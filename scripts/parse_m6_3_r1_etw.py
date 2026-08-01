"""Strict ETW correlation for M6.3-R1 physical artifact reads.

For the Windows Kernel providers used by this harness, the File Read event's
``FileKey`` is the value exposed as Kernel Disk's ``FileObject``.  The parser
therefore requires exact benchmark PID, exact artifact path/FileObject, and
that provider-defined FileKey/FileObject join before it sums disk bytes.
"""
from __future__ import annotations

import argparse
import ctypes
import csv
import json
import os
import re
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from typing import Any


BYTE_FIELDS = ("iosize", "transfersize", "bytecount", "size")
FILE_FIELDS = ("filename", "filepath", "path", "openpath")
FILE_OBJECT_FIELDS = ("fileobject", "fileobjectptr", "filekey")
IRP_FIELDS = ("irp", "irpptr", "irpaddress", "iorequestpacket")


@lru_cache(maxsize=1)
def device_paths() -> dict[str, str]:
    if os.name != "nt":
        return {}
    mappings: dict[str, str] = {}
    buffer = ctypes.create_unicode_buffer(32768)
    for ordinal in range(ord("A"), ord("Z") + 1):
        drive = f"{chr(ordinal)}:"
        if ctypes.windll.kernel32.QueryDosDeviceW(drive, buffer, len(buffer)):
            mappings[buffer.value.lower()] = drive
    return mappings


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def normalized_path(value: str) -> str:
    raw = value.strip().replace("/", "\\")
    for device, drive in device_paths().items():
        if raw.lower() == device or raw.lower().startswith(device + "\\"):
            raw = drive + raw[len(device) :]
            break
    return os.path.normcase(os.path.abspath(raw)).replace("/", "\\")


def fields(event: ET.Element) -> dict[str, str]:
    result: dict[str, str] = {}
    for element in event.iter():
        name = element.attrib.get("Name") or element.attrib.get("name")
        value = element.attrib.get("Value") or element.attrib.get("value")
        text = (element.text or "").strip()
        if name and (value is not None or text):
            result[name.lower()] = value if value is not None else text
        elif local_name(element.tag) not in {"event", "system", "eventdata", "userdata"}:
            if text:
                result[local_name(element.tag)] = text
        if local_name(element.tag) == "execution":
            for key, attribute in element.attrib.items():
                result[key.lower()] = attribute
    return result


def first(data: dict[str, str], names: tuple[str, ...]) -> str | None:
    return next((data[name] for name in names if name in data), None)


def parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value, 0)
    except ValueError:
        return None


def event_identity(event: ET.Element, data: dict[str, str]) -> str:
    parts = []
    for element in event.iter():
        if local_name(element.tag) in {"provider", "task", "opcode", "eventid"}:
            parts.extend(str(value) for value in element.attrib.values())
            if element.text:
                parts.append(element.text)
    parts.extend(data.get(name, "") for name in ("eventname", "taskname", "opcode"))
    return " ".join(parts).lower()


def parse(xml_path: Path, pid: int, artifacts: list[Path]) -> dict[str, Any]:
    artifact_paths = {normalized_path(str(path)): str(path.resolve()) for path in artifacts}
    root = ET.parse(xml_path).getroot()
    events = [item for item in root.iter() if local_name(item.tag) == "event"]
    file_objects: dict[str, str] = {}
    file_reads: list[dict[str, Any]] = []
    disk_reads: dict[str, int] = {}

    for event in events:
        data = fields(event)
        identity = event_identity(event, data)
        event_pid = parse_int(first(data, ("processid", "pid")))
        path_value = first(data, FILE_FIELDS)
        file_object = first(data, FILE_OBJECT_FIELDS)
        irp = first(data, IRP_FIELDS)
        byte_count = parse_int(first(data, BYTE_FIELDS))

        if path_value and file_object:
            normalized = normalized_path(path_value)
            if normalized in artifact_paths:
                file_objects[file_object.lower()] = normalized

        is_file = "kernel-file" in identity or "fileio" in identity or " file " in identity
        is_disk = "kernel-disk" in identity or "diskio" in identity or " disk " in identity
        is_read = "read" in identity
        if is_disk:
            event_id = next(
                ((element.text or "").strip() for element in event.iter() if local_name(element.tag) == "eventid"),
                "",
            )
            is_read = event_id == "10" or is_read
        resolved_path = normalized_path(path_value) if path_value else file_objects.get((file_object or "").lower())
        if is_file and is_read and event_pid == pid and resolved_path in artifact_paths and irp:
            file_reads.append(
                {
                    "irp": irp.lower(),
                    "artifact_path": artifact_paths[resolved_path],
                    "logical_event_bytes": byte_count,
                }
            )
        if is_disk and is_read and irp and byte_count is not None and byte_count >= 0:
            disk_reads[irp.lower()] = disk_reads.get(irp.lower(), 0) + byte_count

    correlated = []
    for item in file_reads:
        disk_bytes = disk_reads.get(item["irp"])
        if disk_bytes is not None:
            correlated.append({**item, "physical_bytes": disk_bytes})

    per_artifact = []
    for artifact in artifacts:
        resolved = str(artifact.resolve())
        rows = [item for item in correlated if item["artifact_path"] == resolved]
        per_artifact.append(
            {
                "path": resolved,
                "correlated_event_count": len(rows),
                "physical_read_bytes": sum(item["physical_bytes"] for item in rows),
            }
        )
    all_artifacts_correlated = bool(per_artifact) and all(row["correlated_event_count"] > 0 for row in per_artifact)
    return {
        "schema": "m6.3-r1-etw-correlation-v1",
        "status": "correlated" if all_artifacts_correlated else "not_measured",
        "pid": pid,
        "artifact_count": len(artifacts),
        "file_read_event_count": len(file_reads),
        "correlated_event_count": len(correlated),
        "physical_read_bytes": sum(item["physical_bytes"] for item in correlated) if all_artifacts_correlated else None,
        "artifacts": per_artifact,
        "correlation": "exact PID + exact artifact path/FileObject + shared IRP + Kernel Disk read byte count",
        "events_lost": 0,
        "buffers_lost": 0,
    }


def csv_payload(row: dict[str, str]) -> dict[str, str]:
    data = {key.strip().lower().replace(" ", ""): (value or "").strip() for key, value in row.items() if key}
    payload = " ".join(
        value
        for key, value in row.items()
        if key and key.strip().lower().replace(" ", "") in {"userdata", "eventdata", "data"} and value
    )
    for match in re.finditer(
        r"(?:^|[;,])\s*([A-Za-z][A-Za-z0-9_ ]*?)\s*(?:=|:)\s*(.*?)\s*(?=(?:[;,]\s*[A-Za-z][A-Za-z0-9_ ]*?\s*(?:=|:))|$)",
        payload,
    ):
        data[match.group(1).strip().lower().replace(" ", "")] = match.group(2).strip()
    trailing = row.get(None, [])
    extra: list[str] = []
    if trailing:
        user_data = next(
            (value for key, value in row.items() if key and key.strip().lower().replace(" ", "") == "userdata"),
            "",
        )
        extra = [str(user_data).strip(), *[str(value).strip() for value in trailing]]
    provider = data.get("eventname", "").strip().lower()
    event_id = parse_int(data.get("eventid"))
    names: tuple[str, ...] = ()
    if provider == "eventtrace" and data.get("type", "").strip().lower() == "header":
        names = (
            "buffersize", "version", "providerversion", "numberofprocessors", "endtime",
            "timerresolution", "maxfilesize", "logfilemode", "bufferswritten", "startbuffers",
            "pointersize", "eventslost", "cpuspeed", "loggername", "logfilename", "boottime",
            "perffreq", "starttime", "reservedflags", "bufferslost", "sessionnamestring",
            "logfilenamestring",
        )
    elif "kernel-file" in provider and event_id == 12:
        names = ("irp", "fileobject", "issuingthreadid", "createoptions", "createattributes", "shareaccess", "filename")
    elif "kernel-file" in provider and event_id == 15:
        names = ("byteoffset", "irp", "fileobject", "filekey", "issuingthreadid", "iosize", "ioflags", "extraflags")
    elif "kernel-disk" in provider and event_id == 10:
        names = ("disknumber", "irpflags", "transfersize", "reserved", "byteoffset", "fileobject", "iorequestpacket", "highresresponsetime")
    for name, value in zip(names, extra):
        data[name] = value.strip('"')
    return data


def trace_summary_losses(summary_path: Path | None) -> tuple[int | None, str | None]:
    if summary_path is None:
        return None, None
    text = summary_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"^Total Events\s+Lost\s+(\d+)\s*$", text, re.MULTILINE)
    if match is None:
        return None, None
    return int(match.group(1)), "tracerpt_summary"


def parse_csv(csv_path: Path, pid: int, artifacts: list[Path], summary_path: Path | None = None) -> dict[str, Any]:
    artifact_paths = {normalized_path(str(path)): str(path.resolve()) for path in artifacts}
    file_objects: dict[str, str] = {}
    file_reads: list[dict[str, Any]] = []
    disk_reads: dict[str, list[int]] = {}
    artifact_file_keys: dict[str, set[str]] = {}
    events_lost: int | None = None
    buffers_lost: int | None = None
    headers: list[str] = []
    row_count = 0

    # tracerpt normally emits UTF-8, but provider-rendered text can contain a
    # legacy byte. Replacement preserves the comma-delimited record and keeps
    # the strict artifact path/PID/key checks intact without loading the trace
    # into memory.
    with csv_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as source:
        reader = csv.DictReader(source)
        headers = list(reader.fieldnames or [])
        for row in reader:
            row_count += 1
            data = csv_payload(row)
            identity = " ".join(
                data.get(name, "")
                for name in ("eventname", "type", "provider", "task", "opcode", "channel", "eventid")
            ).lower()
            event_pid = parse_int(first(data, ("processid", "pid")))
            path_value = first(data, FILE_FIELDS)
            file_object = first(data, FILE_OBJECT_FIELDS)
            irp = first(data, IRP_FIELDS)
            byte_count = parse_int(first(data, BYTE_FIELDS + ("length",)))
            if events_lost is None and "eventslost" in data:
                events_lost = parse_int(data["eventslost"])
            if buffers_lost is None and "bufferslost" in data:
                buffers_lost = parse_int(data["bufferslost"])

            if path_value and file_object:
                normalized = normalized_path(path_value)
                if normalized in artifact_paths:
                    file_objects[file_object.lower()] = normalized
            is_file = "kernel-file" in identity or "fileio" in identity
            is_disk = "kernel-disk" in identity or "diskio" in identity
            event_id = data.get("eventid", "").strip()
            is_read = "read" in identity or (is_file and event_id == "15") or (is_disk and event_id == "10")
            resolved_path = normalized_path(path_value) if path_value else file_objects.get((file_object or "").lower())
            if is_file and is_read and event_pid == pid and resolved_path in artifact_paths and irp:
                file_key = (data.get("filekey") or "").lower()
                file_reads.append(
                    {
                        "file_key": file_key,
                        "artifact_path": artifact_paths[resolved_path],
                        "logical_event_bytes": byte_count,
                    }
                )
                if file_key:
                    artifact_file_keys.setdefault(artifact_paths[resolved_path], set()).add(file_key)
            if is_disk and is_read and event_pid == pid and file_object and byte_count is not None and byte_count >= 0:
                disk_reads.setdefault(file_object.lower(), []).append(byte_count)

    summary_events_lost, summary_loss_source = trace_summary_losses(summary_path)
    if summary_events_lost is not None:
        events_lost = summary_events_lost
    # A FileKey can appear in many logical File Read events.  Each matching
    # Kernel Disk event is physical work and must be attributed once, not once
    # per logical read sharing that key.
    correlated = []
    for artifact_path, file_keys in artifact_file_keys.items():
        for file_key in file_keys:
            for physical_bytes in disk_reads.get(file_key, []):
                correlated.append({"artifact_path": artifact_path, "file_key": file_key, "physical_bytes": physical_bytes})
    per_artifact = []
    for artifact in artifacts:
        resolved = str(artifact.resolve())
        rows = [item for item in correlated if item["artifact_path"] == resolved]
        per_artifact.append(
            {"path": resolved, "correlated_event_count": len(rows), "physical_read_bytes": sum(item["physical_bytes"] for item in rows)}
        )
    loss_counters_valid = events_lost == 0 and (buffers_lost == 0 or summary_loss_source == "tracerpt_summary")
    all_artifacts_correlated = (
        loss_counters_valid and bool(per_artifact) and all(row["correlated_event_count"] > 0 for row in per_artifact)
    )
    return {
        "schema": "m6.3-r1-etw-correlation-v1",
        "status": "correlated" if all_artifacts_correlated else "not_measured",
        "pid": pid,
        "artifact_count": len(artifacts),
        "csv_row_count": row_count,
        "csv_headers": headers,
        "file_read_event_count": len(file_reads),
        "correlated_event_count": len(correlated),
        "physical_read_bytes": sum(item["physical_bytes"] for item in correlated) if all_artifacts_correlated else None,
        "artifacts": per_artifact,
        "events_lost": events_lost,
        "buffers_lost": buffers_lost if buffers_lost is not None else "not_reported",
        "loss_counter_source": summary_loss_source or ("csv_header" if events_lost is not None else "not_measured"),
        "correlation": "exact PID + exact artifact path/FileObject + Kernel File Read.FileKey = Kernel Disk.FileObject + Kernel Disk read byte count",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--xml", type=Path)
    source.add_argument("--csv", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--pid", required=True, type=int)
    parser.add_argument("--artifact", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = parse(args.xml, args.pid, args.artifact) if args.xml else parse_csv(args.csv, args.pid, args.artifact, args.summary)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(result["status"])
    return 0 if result["status"] == "correlated" else 1


if __name__ == "__main__":
    raise SystemExit(main())

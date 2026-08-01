import sys
import tempfile
import unittest
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_m6_3_r1_etw import parse, parse_csv


XML = """<Events>
<Event>
  <System><Provider Name="Microsoft-Windows-Kernel-File"/><Task Name="FileIo"/><Opcode Name="Read"/></System>
  <EventData>
    <Data Name="ProcessId" Value="42"/><Data Name="FileObject" Value="0x1"/>
    <Data Name="FileName" Value="{artifact}"/><Data Name="Irp" Value="0xa"/>
    <Data Name="IoSize" Value="4096"/>
  </EventData>
</Event>
<Event>
  <System><Provider Name="Microsoft-Windows-Kernel-Disk"/><Task Name="DiskIo"/><Opcode Name="Read"/></System>
  <EventData>
    <Data Name="ProcessId" Value="4"/><Data Name="IORequestPacket" Value="0xa"/>
    <Data Name="TransferSize" Value="4096"/>
  </EventData>
</Event>
</Events>"""


class EtwParserTests(unittest.TestCase):
    def test_correlates_exact_pid_path_irp_and_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact.bin"
            artifact.write_bytes(b"x")
            xml = root / "trace.xml"
            xml.write_text(XML.format(artifact=artifact), encoding="utf-8")
            result = parse(xml, 42, [artifact])
            self.assertEqual(result["status"], "correlated", result)
            self.assertEqual(result["physical_read_bytes"], 4096)

    def test_rejects_disk_event_without_matching_irp(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact.bin"
            artifact.write_bytes(b"x")
            xml = root / "trace.xml"
            xml.write_text(XML.format(artifact=artifact).replace('Value="0xa"', 'Value="0xb"', 1), encoding="utf-8")
            result = parse(xml, 42, [artifact])
            self.assertEqual(result["status"], "not_measured")
            self.assertIsNone(result["physical_read_bytes"])

    def test_rejects_wrong_pid_or_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact.bin"
            other = root / "other.bin"
            artifact.write_bytes(b"x")
            other.write_bytes(b"x")
            xml = root / "trace.xml"
            xml.write_text(XML.format(artifact=artifact), encoding="utf-8")
            self.assertEqual(parse(xml, 43, [artifact])["status"], "not_measured")
            self.assertEqual(parse(xml, 42, [other])["status"], "not_measured")

    def test_streams_csv_and_requires_zero_loss_counters(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact.bin"
            artifact.write_bytes(b"x")
            csv_path = root / "trace.csv"
            csv_path.write_text(
                'Event Name,Provider,Event ID,Task,PID,User Data\n'
                'Header,MSNT_SystemTrace,0,Header,4,"EventsLost=0; BuffersLost=0"\n'
                f'Create,Microsoft-Windows-Kernel-File,12,Create,42,"FileObject=0x1; FileName={artifact}"\n'
                'Read,Microsoft-Windows-Kernel-File,15,Read,42,"FileObject=0x1; FileKey=0x2; Irp=0xa; IOSize=4096"\n'
                'Read,Microsoft-Windows-Kernel-Disk,10,Read,42,"FileObject=0x2; IORequestPacket=0xa; TransferSize=4096"\n',
                encoding="utf-8",
            )
            result = parse_csv(csv_path, 42, [artifact])
            self.assertEqual(result["status"], "correlated", result)
            self.assertEqual(result["physical_read_bytes"], 4096)
            csv_path.write_text(csv_path.read_text(encoding="utf-8").replace("EventsLost=0", "EventsLost=1"), encoding="utf-8")
            self.assertEqual(parse_csv(csv_path, 42, [artifact])["status"], "not_measured")

    def test_parses_tracerpt_positional_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact.bin"
            artifact.write_bytes(b"x")
            csv_path = root / "trace.csv"
            header = ["Event Name", "Type", "Event ID", "Version", "Channel", "Level", "Opcode", "Task", "Keyword", "PID", "TID", "Processor Number", "Instance ID", "Parent Instance ID", "Activity ID", "Related Activity ID", "Clock-Time", "Kernel(ms)", "User(ms)", "User Data"]
            def fixed(provider, kind, event_id, pid):
                return [provider, kind, event_id, "", "", "", "", "", "", pid, "", "", "", "", "", "", "", "", ""]
            with csv_path.open("w", encoding="utf-8", newline="") as output:
                writer = csv.writer(output)
                writer.writerow(header)
                writer.writerow(fixed("EventTrace", "Header", "0", "4") + [8192, 1, 1, 8, 0, 0, 0, 0, 1, 1, 8, 0, 1, 0, 0, 0, 0, 0, 0, 0, "session", "trace"])
                writer.writerow(fixed("Microsoft-Windows-Kernel-File", "Info", "12", "0x2a") + ["0xa", "0x1", 1, 0, 0, 1, artifact])
                writer.writerow(fixed("Microsoft-Windows-Kernel-File", "Info", "15", "0x2a") + [0, "0xa", "0x1", "0x2", 1, 4096, 0, 0])
                writer.writerow(fixed("Microsoft-Windows-Kernel-Disk", "Info", "10", "0x2a") + [0, 0, 4096, 0, 0, "0x2", "0xa", 1])
            result = parse_csv(csv_path, 42, [artifact])
            self.assertEqual(result["status"], "correlated", result)
            self.assertEqual(result["physical_read_bytes"], 4096)

    def test_rejects_disk_event_from_another_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact.bin"
            artifact.write_bytes(b"x")
            csv_path = root / "trace.csv"
            csv_path.write_text(
                'Event Name,Provider,Event ID,Task,PID,User Data\n'
                'Header,MSNT_SystemTrace,0,Header,4,"EventsLost=0; BuffersLost=0"\n'
                f'Create,Microsoft-Windows-Kernel-File,12,Create,42,"FileObject=0x1; FileName={artifact}"\n'
                'Read,Microsoft-Windows-Kernel-File,15,Read,42,"FileObject=0x1; FileKey=0x2; Irp=0xa; IOSize=4096"\n'
                'Read,Microsoft-Windows-Kernel-Disk,10,Read,41,"FileObject=0x2; TransferSize=4096"\n',
                encoding="utf-8",
            )
            result = parse_csv(csv_path, 42, [artifact])
            self.assertEqual(result["status"], "not_measured", result)
            self.assertIsNone(result["physical_read_bytes"])

    def test_counts_one_disk_event_once_for_repeated_file_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact.bin"
            artifact.write_bytes(b"x")
            csv_path = root / "trace.csv"
            csv_path.write_text(
                'Event Name,Provider,Event ID,Task,PID,User Data\n'
                'Header,MSNT_SystemTrace,0,Header,4,"EventsLost=0; BuffersLost=0"\n'
                f'Create,Microsoft-Windows-Kernel-File,12,Create,42,"FileObject=0x1; FileName={artifact}"\n'
                'Read,Microsoft-Windows-Kernel-File,15,Read,42,"FileObject=0x1; FileKey=0x2; Irp=0xa; IOSize=4096"\n'
                'Read,Microsoft-Windows-Kernel-File,15,Read,42,"FileObject=0x1; FileKey=0x2; Irp=0xb; IOSize=4096"\n'
                'Read,Microsoft-Windows-Kernel-Disk,10,Read,42,"FileObject=0x2; TransferSize=4096"\n',
                encoding="utf-8",
            )
            result = parse_csv(csv_path, 42, [artifact])
            self.assertEqual(result["status"], "correlated", result)
            self.assertEqual(result["correlated_event_count"], 1)
            self.assertEqual(result["physical_read_bytes"], 4096)


if __name__ == "__main__":
    unittest.main()

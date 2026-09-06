"""Validate and summarize frozen M6.3-R2.2-D5c paired evidence."""
from __future__ import annotations

import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

CONTRACT_SHA256 = "34b5a5a032bf6251b7d724c115bad9db44fc11f08092d235155e32c304d8acd3"
FIXTURES = ["short_english", "short_thai"]
CACHE_LABELS = ["first_process_touch", "likely_warm"]
PAIR_ORDER = [
    ["production_hybrid", "reference_f32"],
    ["reference_f32", "production_hybrid"],
    ["production_hybrid", "reference_f32"],
    ["reference_f32", "production_hybrid"],
    ["production_hybrid", "reference_f32"],
]
WALL_SPEEDUP_MIN_PERCENT = 1.0
PAIR_WINS_MIN = 4
MEMORY_REGRESSION_MAX_PERCENT = 1.0


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def finite_nonnegative(value: object, name: str) -> float:
    number = float(value)
    require(math.isfinite(number) and number >= 0.0, f"invalid {name}")
    return number


def percent_improvement(reference: float, hybrid: float) -> float:
    require(reference > 0.0, "reference metric must be positive")
    return 100.0 * (reference - hybrid) / reference


def percent_regression(reference: float, hybrid: float) -> float:
    require(reference > 0.0, "reference memory metric must be positive")
    return 100.0 * (hybrid - reference) / reference


def expected_identities() -> list[tuple[str, str, int, int, str]]:
    return [
        (fixture, cache_label, pair_index, order_index, mode)
        for fixture in FIXTURES
        for cache_label in CACHE_LABELS
        for pair_index, modes in enumerate(PAIR_ORDER, start=1)
        for order_index, mode in enumerate(modes, start=1)
    ]


def validate(document: dict) -> dict:
    require(document.get("schema") == "m6.3-r2.2-d5c-paired-samples-v1", "sample schema mismatch")
    require(document.get("contract_sha256") == CONTRACT_SHA256, "contract hash mismatch")
    samples = document.get("samples")
    require(isinstance(samples, list) and len(samples) == 40, "D5c requires exactly 40 samples")
    expected = expected_identities()
    for sample, identity in zip(samples, expected, strict=True):
        fixture, cache_label, pair_index, order_index, mode = identity
        require(
            (sample.get("fixture"), sample.get("cache_label"), sample.get("pair"), sample.get("order_index"), sample.get("mode"))
            == (fixture, cache_label, pair_index, order_index, mode),
            "sample ordering/identity mismatch",
        )
        for metric in (
            "wall_seconds",
            "prefill_tokens_per_second",
            "decode_tokens_per_second",
            "peak_working_set_bytes",
            "peak_private_bytes",
            "logical_artifact_bytes_read",
            "expert_payload_bytes_read",
            "candidate_artifact_bytes_read",
        ):
            finite_nonnegative(sample.get(metric), metric)
        require(float(sample["wall_seconds"]) > 0.0, "wall seconds must be positive")
        require(float(sample["prefill_tokens_per_second"]) > 0.0, "prefill throughput must be positive")
        require(float(sample["decode_tokens_per_second"]) > 0.0, "decode throughput must be positive")
        if mode == "reference_f32":
            require(int(sample["candidate_artifact_bytes_read"]) == 0, "reference candidate bytes must be zero")
        else:
            require(int(sample["candidate_artifact_bytes_read"]) > 0, "hybrid candidate bytes must be positive")

    paired: dict[tuple[str, str, int], dict[str, dict]] = defaultdict(dict)
    for sample in samples:
        paired[(sample["fixture"], sample["cache_label"], int(sample["pair"]))][sample["mode"]] = sample
    require(len(paired) == 20, "D5c pair count mismatch")

    wall_speedups: dict[str, list[float]] = {fixture: [] for fixture in FIXTURES}
    prefill_improvements: dict[str, list[float]] = {fixture: [] for fixture in FIXTURES}
    decode_improvements: dict[str, list[float]] = {fixture: [] for fixture in FIXTURES}
    wins: dict[tuple[str, str], int] = defaultdict(int)
    logical_reduction_bytes: list[int] = []
    for (fixture, cache_label, _pair), modes in paired.items():
        require(set(modes) == {"reference_f32", "production_hybrid"}, "incomplete D5c pair")
        reference = modes["reference_f32"]
        hybrid = modes["production_hybrid"]
        ref_wall = float(reference["wall_seconds"])
        hyb_wall = float(hybrid["wall_seconds"])
        wall_speedups[fixture].append(percent_improvement(ref_wall, hyb_wall))
        prefill_improvements[fixture].append(
            100.0 * (float(hybrid["prefill_tokens_per_second"]) - float(reference["prefill_tokens_per_second"]))
            / float(reference["prefill_tokens_per_second"])
        )
        decode_improvements[fixture].append(
            100.0 * (float(hybrid["decode_tokens_per_second"]) - float(reference["decode_tokens_per_second"]))
            / float(reference["decode_tokens_per_second"])
        )
        if hyb_wall < ref_wall:
            wins[(fixture, cache_label)] += 1
        reduction = int(reference["expert_payload_bytes_read"]) - int(hybrid["expert_payload_bytes_read"])
        require(reduction > 0, "hybrid must reduce logical expert payload bytes in every pair")
        logical_reduction_bytes.append(reduction)

    fixture_summary = {}
    for fixture in FIXTURES:
        require(len(wall_speedups[fixture]) == 10, "fixture wall pair count")
        fixture_summary[fixture] = {
            "median_wall_speedup_percent": statistics.median(wall_speedups[fixture]),
            "wall_speedup_percent_by_pair": wall_speedups[fixture],
            "median_prefill_throughput_improvement_percent": statistics.median(prefill_improvements[fixture]),
            "median_decode_throughput_improvement_percent": statistics.median(decode_improvements[fixture]),
        }

    cell_summary = {}
    memory_regressions = []
    for fixture in FIXTURES:
        for cache_label in CACHE_LABELS:
            reference_rows = [s for s in samples if s["fixture"] == fixture and s["cache_label"] == cache_label and s["mode"] == "reference_f32"]
            hybrid_rows = [s for s in samples if s["fixture"] == fixture and s["cache_label"] == cache_label and s["mode"] == "production_hybrid"]
            require(len(reference_rows) == len(hybrid_rows) == 5, "cell sample count")
            ref_ws = statistics.median(int(s["peak_working_set_bytes"]) for s in reference_rows)
            hyb_ws = statistics.median(int(s["peak_working_set_bytes"]) for s in hybrid_rows)
            ref_private = statistics.median(int(s["peak_private_bytes"]) for s in reference_rows)
            hyb_private = statistics.median(int(s["peak_private_bytes"]) for s in hybrid_rows)
            ws_regression = percent_regression(ref_ws, hyb_ws)
            private_regression = percent_regression(ref_private, hyb_private)
            memory_regressions.extend([ws_regression, private_regression])
            key = f"{fixture}:{cache_label}"
            cell_summary[key] = {
                "hybrid_wall_wins": wins[(fixture, cache_label)],
                "reference_peak_working_set_median_bytes": ref_ws,
                "hybrid_peak_working_set_median_bytes": hyb_ws,
                "working_set_regression_percent": ws_regression,
                "reference_peak_private_median_bytes": ref_private,
                "hybrid_peak_private_median_bytes": hyb_private,
                "private_bytes_regression_percent": private_regression,
            }

    wall_gate = all(
        fixture_summary[fixture]["median_wall_speedup_percent"] >= WALL_SPEEDUP_MIN_PERCENT
        for fixture in FIXTURES
    )
    wins_gate = all(summary["hybrid_wall_wins"] >= PAIR_WINS_MIN for summary in cell_summary.values())
    memory_gate = max(memory_regressions) <= MEMORY_REGRESSION_MAX_PERCENT
    logical_gate = all(value > 0 for value in logical_reduction_bytes)
    go = wall_gate and wins_gate and memory_gate and logical_gate
    return {
        "schema": "m6.3-r2.2-d5c-paired-result-v1",
        "contract_sha256": CONTRACT_SHA256,
        "sample_count": len(samples),
        "pair_count": len(paired),
        "fixture_summary": fixture_summary,
        "cell_summary": cell_summary,
        "logical_expert_reduction_bytes_per_pair": logical_reduction_bytes,
        "minimum_logical_expert_reduction_bytes": min(logical_reduction_bytes),
        "worst_memory_regression_percent": max(memory_regressions),
        "gates": {
            "median_wall_speedup_each_fixture": wall_gate,
            "four_of_five_wins_each_cell": wins_gate,
            "memory_regression_within_one_percent": memory_gate,
            "positive_logical_expert_reduction_every_pair": logical_gate,
        },
        "decision": "go" if go else "no_go",
        "go_authorizes": "separate_all_layer_plan_review_only" if go else "nothing",
    }


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: validate_m6_3_r2_2_d5c_result.py SAMPLES.json RESULT.json", file=sys.stderr)
        return 2
    samples_path = Path(sys.argv[1])
    result_path = Path(sys.argv[2])
    require(not result_path.exists(), "D5c result output must be new")
    document = json.loads(samples_path.read_text(encoding="utf-8"))
    result = validate(document)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "samples": result["sample_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

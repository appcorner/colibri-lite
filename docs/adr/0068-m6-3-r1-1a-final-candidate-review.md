# ADR 0068: M6.3-R1.1a Final Candidate Review

## Status

Accepted for R1.1a closure. This ADR closes only the append-only characterization remediation defined by ADR 0059 and resolved to `code_newline` by ADR 0060. It does not revise the historical R1.1 `no_candidate_admitted` outcome, authorize R1.2, or authorize M6.4.

## Evidence

Both pre-registered candidates completed three valid `code_newline` characterization runs. Each valid run preserved exact Layer-0 router IDs, passed the unchanged fixed-logit cap `0.05`, named the first divergence as `layer0.selected_expert_output`, directly consumed packed values and scales, produced zero complete F32 weight materializations, and passed the ADR 0065 nonzero process-correlated physical-I/O gate with zero lost events.

The deterministic measurements were bit-identical within each three-run set. Group-64 recorded fixed-logit max abs error `2.37542390823364258e-2`; group-32 recorded `2.36730575561523438e-2`.

## Decision

Apply the ADR 0059 ranking lexicographically and stop at the first differing criterion:

1. fixed-logit max abs error ascending;
2. routed-expert/MoE checkpoint error ascending;
3. artifact bytes ascending;
4. candidate ID lexical ascending.

Group-32 wins criterion 1 because `2.36730575561523438e-2 < 2.37542390823364258e-2`. Later criteria therefore do not participate in the selection. Record `cpu-safe-rust-int8-group32-layer0-r1-1a` as the R1.1a characterization winner.

The admission outcome remains `no_candidate_admitted`. ADR 0059 made this remediation append-only and explicitly prohibited revising the completed R1.1 outcome or authorizing R1.2. A future promotion of group-32 requires a separately reviewed R1.1 admission amendment or replacement record; it cannot be inferred from this characterization result.

## Measurement limitation

ADR 0065 establishes that candidate-correlated physical disk I/O was observed, not that the complete candidate artifact was device-read. The correlated physical byte counts are a small fraction of each artifact and must not be described as complete cold-device reads.

## Consequences

- R1.1a is complete.
- Group-32 is the deterministic characterization winner.
- Historical R1.1 remains `no_candidate_admitted`.
- R1.2 remains blocked.
- The pre-registration records remain byte-identical; machine-readable closure is stored separately in `models/qwen3-30b-a3b/m6.3-r1-1a-final-review-v1.json`.
- No runtime, artifact format, cache policy, dependency, unsafe boundary, or production default changes.

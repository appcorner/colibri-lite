# Correctness Gates

1. Validate tensor identities, shapes, dtypes, and artifact hashes exactly.
2. Compare router selections exactly whenever the F32 margin is classified
   safe; classify near ties instead of weakening the rule.
3. Compare layer checkpoints and logits against documented absolute/relative
   tolerances.
4. Run deterministic English and Thai fixture outputs and record quality
   deltas.
5. Repeat candidate execution and require deterministic applicable outputs.

The report must name the first failed stage. Tolerances are inherited from the
reference registry or added by ADR with numerical justification.

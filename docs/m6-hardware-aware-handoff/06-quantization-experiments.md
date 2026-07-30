# Quantization Experiment Policy

M6.3 evaluates one expert-weight candidate for one layer. Candidate formats may
include group-wise INT8 or Q4/Q5/Q6, but selection requires recorded artifact
layout, conversion command, tool versions, hashes, license, direct-consumption
proof, and an ADR.

Router, norms, routing weights, residuals, softmax, and accumulation remain
F32 unless separate evidence approves a change. A candidate that reconstructs
a complete expert as F32 during normal execution fails the vertical-slice goal.

Report cold/warm throughput, TTFT, RAM/VRAM peak, physical bytes read/token,
router agreement, checkpoints, logits, and Thai/English quality results.

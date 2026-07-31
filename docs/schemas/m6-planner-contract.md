# M6.2 Planner Contract

`planner-contract-v1.schema.json` defines the JSON documents at the future
`plan` command boundary. It defines contracts only; M6.2-01 neither enumerates
placements nor calculates an estimate.

## Inputs

A request cites immutable hardware and model profile IDs and SHA-256 document
hashes. It also carries the explicit RAM and VRAM budgets plus the workload's
prefill, decode, and context sizes. A planner must use those cited documents;
it must not replace them with ambient host observations.

## Candidate plans and estimates

Each candidate identifies its backend, dense/expert placement, and precision
inventory entry. Its resource record is always explicit, including context
capacity, expert cache, disk bytes per routed token, and startup seconds.

`estimate.status=available` requires positive prefill/decode throughput and
at least one measurement reference. `incomplete` requires named missing
assumptions. `analytical-v1` means a derived prediction, not a measured
end-to-end throughput claim. Later M6.2 tasks must retain provenance for every
derived estimate and report prediction error against recorded benchmarks.

M6.2-02 implements the initial `analytical-v1` math in `clr-core`: model work
per token divided by matching profile rates for compute, RAM, and storage. The
M6.1 model profile does not contain a FLOP count, so that term remains an
explicit model-derived input rather than an implicit architecture constant.

M6.2-03 enumerates the dense locations `ram`/`vram` and expert locations
`ram`/`vram`/`ssd` only when the profile marks every required resource as
`measured`. Mixed RAM/VRAM locations additionally require a measured transfer.
`unavailable` and `not_run` produce no candidate and will be explained by the
later rejection/admission stage.

M6.2-04 compares each enumerated candidate with the caller's exact RAM/VRAM
limits and requested context. A candidate exactly at a byte or token limit is
admitted. All exceeded constraints are emitted in the stable order RAM, VRAM,
then context with the candidate plan ID, calculated requirement, and limit.

## Rejections and ranking

Resource rejection codes require the calculated requirement, the requested
limit, and a unit. Other rejection codes retain a machine-readable category
and human-readable evidence. The schema deliberately does not make an
unsupported or incomplete candidate feasible.

The result's ranking is a unique ordered list of feasible plan IDs. M6.2-05
will enforce that each ranked ID exists and that ordering follows the recorded
deterministic ranking policy.

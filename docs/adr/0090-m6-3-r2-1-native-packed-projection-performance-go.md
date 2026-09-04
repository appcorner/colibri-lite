# ADR 0090: M6.3-R2.1 native packed-projection performance GO

Status: accepted.

## Context

ADR 0087 pre-registered one Layer-0 AVX2+FMA packed-projection hypothesis and
froze quality-before-performance, balanced three-path timing, resource guards,
and no-retry semantics. ADR 0089 closed held-out quality `GO` and authorized
R2.1c official timing only.

R2.1c used one frozen release test binary containing native AVX2+FMA group32,
scalar group32, and reference F32 paths. The matrix completed all 72 fresh
process samples: two fixtures, two views, six balanced triplet permutations,
and three paths per triplet. No automatic retry occurred.

Validated samples SHA-256:
`5f35dd5219079fffad361d0b74344636491359abda69f32668af7138613337ce`.
Validated result SHA-256:
`c1df174cb7fa0ab649797255d0beac2087fe728b78bfa9f621a3c87def663e7e`.
## Result

All frozen performance and resource gates passed.

| Fixture | View | Scalar/native median | Native/F32 median | Native beats scalar | Native beats F32 |
| --- | --- | ---: | ---: | ---: | ---: |
| short_english | compute-only | 3.6810x | 0.5024x | 6/6 | 6/6 |
| short_thai | compute-only | 3.3597x | 0.4497x | 6/6 | 6/6 |
| short_english | load+compute | 2.0800x | 0.0633x | 6/6 | 6/6 |
| short_thai | load+compute | 2.0631x | 0.0605x | 6/6 | 6/6 |

Resource validation preserved zero complete F32 weight materializations, zero
persistent native prepack, zero VRAM use, compute-only zero load bytes, and
expected candidate/reference payload accounting.

## Decision

Close M6.3-R2.1c and R2.1d `GO`. The native AVX2+FMA packed-projection
hypothesis is validated for the measured Layer-0 vertical slice.

Authorize **M6.3-R2.2 design only** for expanded impact modeling/proof. This ADR
does not authorize R2.2 implementation, an all-layer rollout, M6.4, AVX-512,
threading, prefetch, GPU execution, cache-policy changes, or new quantization.

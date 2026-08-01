# ADR 0067: M6.3-R1.1a Reboot-Bound Cold-Cache Protocol

## Status

Accepted after the first otherwise valid group-64 characterization produced
805 candidate File reads but no Kernel-Disk event correlated to the candidate.
That run is invalid admission evidence and was removed through reviewed
cleanup. This ADR is append-only and does not reinterpret its physical-I/O
result as zero or weaken any numerical, telemetry, or admission gate.

## Context

The required fresh conversion reads the source and writes the complete
candidate immediately before characterization. On Windows this can leave the
candidate payload resident in the system filesystem cache. The correlated
candidate process then performs logical reads without a physical device read,
so ADR 0065 correctly returns `not_measured`. Repeating the same sequence is
not evidence that the physical-I/O gate can be satisfied.

An unbuffered reader would change the runtime file-access strategy. Accepting
logical reads or zero physical bytes would weaken a pre-registered gate.
Neither is permitted.

## Decision

Every remaining R1.1a candidate run uses a two-phase, reboot-bound, one-shot
protocol described by the versioned
`m6.3-r1-1a-cold-cache-contract-v1.json` record.

Phase A occurs after a fresh conversion. It verifies the full pre-registered
SHA-256 and size, requires the candidate file to be read-only, and records its
path, volume/file identity, size, modification time, boot marker, and flat run
directory. The run directory and artifact survive the reboot; no artifact is
copied or reconverted between phases.

The machine is then rebooted. Phase B must run within 1,800 seconds of the new
boot. It proves a different boot origin with `GetTickCount64`, verifies only
metadata and stable file identity, performs no payload read or SHA-256 scan,
and writes a one-shot authorization beside the candidate. Clock-derived boot
origins must differ by at least five seconds; the launch must remain in the
same boot session as Phase B.

The ETW collector requires the authorization binding for candidate runs and
atomically consumes it before starting the trace. It then starts ETW before
the candidate process. The candidate's existing full-file verification scan
therefore occurs inside the correlated process and trace. A launch failure,
invalid ETW result, or consumed authorization requires a fresh run directory,
conversion, Phase A, and reboot; authorization reuse is prohibited.

The original ADR 0065 gate remains exact: status `correlated`, candidate disk
read bytes at least one, and zero lost events. Group-64 must close before
group-32 begins. The fixed-logit cap remains `0.05`; control-only checkpoint
budgets remain diagnostics/ranking inputs and are not changed here.

## Consequences

- The protocol measures actual Windows cold-cache physical I/O without adding
  unsafe code, a dependency, unbuffered access, cache-eviction software, or a
  new artifact format.
- Pre-reboot SHA-256 plus stable read-only file identity prevents Phase B from
  silently substituting another artifact without warming its payload.
- A reboot is required for every independent candidate run, so execution is
  slower but the evidence is reproducible and gate-preserving.
- This contract authorizes only a fresh group-64 R1.1a attempt. It does not
  authorize group-32 before group-64 closure, held-out fixtures, R1.2, or M6.4.

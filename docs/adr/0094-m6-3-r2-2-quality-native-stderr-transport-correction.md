# ADR 0094: M6.3-R2.2 quality native-stderr transport correction

- Status: Accepted
- Date: 2026-09-04
- Milestone: M6.3-R2.2b

## Context

ADR 0093 introduced a detached durable worker. Execution ordinal 3 passed its
frozen input preflight and survived independently of the connector, proving the
detachment correction itself works.

The worker later persisted `exit.code=97`, `worker-error.txt`, and no quality
TSV. The error is a PowerShell `NativeCommandError` at the direct native-command
invocation. The test's long-running libtest message was captured in stdout; the
redirected stderr file remained empty. No quality assertion result, panic, or
quality output was persisted.

This is therefore a runner transport failure, not a quality PASS or quality
NO-GO. R2.2b remains undecided and R2.2c remains blocked.

## Decision

One further transport-only correction is allowed. Preserve the exact frozen
release binary, quality harness, reference, sentinel artifacts, host, and all
quality thresholds. No inference/runtime code or gate may change.
The corrected worker MUST launch the frozen test with `Start-Process` using
`-RedirectStandardOutput`, `-RedirectStandardError`, `-Wait`, and `-PassThru`.
The persisted native process `ExitCode` is authoritative. Native stderr must be
stored as raw process output and must not be promoted into a PowerShell error
record merely because it is non-empty.

Use a new durable run root so ordinal-3 evidence remains immutable. Preflight
must again verify all frozen input hashes and require the quality output path to
be absent before launch.

If the native test exits zero and writes the frozen quality TSV, validate the
TSV against the frozen gates. A nonzero native exit is a real test execution
failure and must be inspected as such. A wrapper/preflight error remains a
transport failure and does not by itself determine model quality.

No automatic quality rerun beyond this corrected execution is authorized by
this ADR. R2.2 performance remains blocked until a valid quality PASS exists.

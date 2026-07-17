# Temporary Artifact Policy

## Purpose

Large model workflows must not multiply source, converted, or unified artifact
payloads through implicit copies or nested temporary directories. This policy
applies to every model acquisition, conversion, validation, relocation, and
correctness task.

## Directory ownership

Each task run receives exactly one unique, flat run directory:

```text
D:\tmp\colibri-lite-runs\<task-id>-<run-id>
```

Run IDs must not be reused. A run directory must never contain another temp or
run directory. Component subdirectories that form the final artifact layout
are allowed; nested scratch, cache, staging, download, or `tmp` directories are
prohibited.

Canonical source and artifact roots are separate from run directories. They
are treated as read-only inputs. Tasks receive their paths explicitly and must
reuse them rather than copying them into a run directory.

The stable artifact root must be registered in the tracked model-specific
canonical-root registry. Cleanup requires that registry as a separate input,
rejects any plan naming a different canonical root, and rejects canonical roots
inside the temporary namespace.

## Output lifecycle

1. Write a final output to one sibling `.incomplete` path in the task run.
2. Sync and validate it.
3. Atomically promote it to its reviewed final path.
4. Record final size/hash and post-task disk accounting.
5. Remove the successful run directory automatically.

Failed runs are removed automatically unless explicit debug retention is
enabled before the run. Debug retention is bounded to:

- at most 2 failed runs per task;
- at most 24 hours;
- at most 20 GiB across retained runs.

The oldest retained run is removed before starting another run if any bound
would be exceeded. Large source, dense, expert, or unified payload copies are
never retained as debug output without a separately reviewed disk budget.

## Reproducibility and relocation

Determinism tests write only the new output being tested. Existing source and
canonical artifacts remain read-only. A repeat that would duplicate a complete
payload must use a manifest-root override, hard links, a junction, or an atomic
directory rename when that preserves the semantics under test.

Relocation tests must rename the complete directory or use an alternate root
override. They must not copy the 122 GB unified artifact.

Hard links are allowed only when:

- source and destination are on the same volume;
- the artifact contract depends on bytes, not file identity;
- the final canonical path is protected before cleanup;
- link counts are included in cleanup accounting.

## Disk preflight

Before creating a run directory, record:

- filesystem free bytes;
- canonical source/artifact paths and logical bytes;
- expected new output bytes;
- expected peak temporary bytes;
- hard-linked/shared bytes that require no new allocation;
- retained-debug bytes;
- a safety reserve of the greater of 1 GiB or 5% of expected incremental
  allocation.

The task may start only when:

```text
free bytes >= new output + peak temporary + retained debug + safety reserve
```

Preflight must fail before output creation when the requirement is not met.

## Post-task accounting

After success or failure, record:

- free bytes before and after;
- created logical bytes and file count;
- promoted final bytes and file count;
- removed temporary bytes and file count;
- last-link reclaimable bytes versus shared hard-link bytes;
- retained debug paths, ages, and bytes;
- whether the run directory was removed.

A successful task is not closed while an unreviewed run directory remains.

## Cleanup

`scripts/cleanup_temp_artifacts.py` consumes a reviewed JSON plan and tracked
canonical-root registry. Dry-run is the default; deletion requires `--apply`.
Before either mode it verifies:

- the canonical root manifest and all referenced final paths exist;
- the canonical root is outside the temporary namespace and exactly matches the
  registry path, manifest size/hash, and file count;
- protected source paths exist;
- every candidate remains within the declared temp root;
- candidates do not overlap protected/canonical paths or one another;
- file counts, logical bytes, last-link reclaimable bytes, and shared hard-link
  bytes have not changed since review;
- candidate classifications are explicitly deletable;
- candidates contain no symlink or junction traversal.

Cleanup plans must be regenerated after any filesystem change. The apply mode
must never be invoked until the corresponding dry-run report is reviewed.

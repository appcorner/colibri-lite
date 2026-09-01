# ADR 0077: M6.3-R1.3 Paired Measurement Result

## Status

Accepted. M6.3-R1.3 closes with a valid 20-sample paired measurement set and authorizes R1.4 review only.

## Evidence identity

The final run uses measurement contract v5 SHA-256 `57a02d697231350977a4391a3625d2808b97cb394078290db398e2996195584f` and release binary SHA-256 `69109985e658918e8b0accf3dacb737daba53b6d6f05fb6535b256379805d541`.

The admitted candidate remains `cpu-safe-rust-int8-group32-layer0-r1-1a`, artifact SHA-256 `777820df3bfd7fa918035757245611c033f80eda9c92bbfca577f61ef504b1a2`.

The persisted sample set is `models/qwen3-30b-a3b/m6.3-r1-3-paired-samples-v1.json`, SHA-256 `4995c01a36c93023ed886108a9368b45746a4f6c2f015e2c0e6cfcbc72e5bea4`. It contains exactly 20 valid release-process samples: five F32/candidate pairs for each of `runtime_cache_cold` and `runtime_cache_warm` in the frozen alternating order.

The validator result is `models/qwen3-30b-a3b/m6.3-r1-3-paired-result-v1.json`, SHA-256 `491ea0490e400e6563807ec9000eee2cbe78f908d5c4ce709ef18b5f32f2d04b`.

Repository evidence is LF-canonicalized without changing JSON values. The original Windows run-file hashes (`cbdef7b4...c0c9` samples and `ac994f67...bcf0` result) remain recorded as `source_run_sha256` provenance in the R1.4 decision record.

One interrupted candidate attempt was excluded before admission because `capture.json` was absent even though inference and ETW correlation had completed. ADR 0076 records the audited replacement; the valid four-sample prefix was preserved exactly.

## Gate results

All 20 valid samples exited successfully, retained final argmax `0`, reported correlated ETW, stayed below the 16 GiB working-set ceiling, respected the 18,874,368-byte F32 cache budget, and passed candidate memory accounting. Maximum observed working set was 142,696,448 bytes and maximum private bytes were 138,760,192 bytes.

## Paired result

The only 5/5 directional win was total logical bytes: candidate reads were 0.8919485285% lower in both cache conditions.

Latency and throughput did not produce a directional win. Median candidate-vs-F32 deltas were:

- `runtime_cache_cold`: timed wall `+2.8183%`, TTFT `+9.1717%`, prefill tok/s `-8.4011%`, decode tok/s `-3.4114%`, working set `+0.2994%`, private bytes `+0.4602%`.
- `runtime_cache_warm`: timed wall `+8.7446%`, TTFT `+7.4820%`, prefill tok/s `-6.9612%`, decode tok/s `-6.6696%`, working set `+0.0460%`, private bytes `+0.1242%`.

Physical I/O also has no directional win. Cold pairs include both measured-zero reads and one large candidate read, while all warm reference/candidate physical reads were measured zero, making paired percentages undefined there rather than evidence of an I/O advantage.

## Decision

R1.3 is complete and its evidence is valid. It authorizes R1.4 review but does not authorize M6.4, all-layer rollout, or a production performance claim.
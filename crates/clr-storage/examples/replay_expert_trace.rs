//! Replay a compact ordered expert trace through the production byte-budgeted
//! cache. The input is generated from the authoritative JSON trace by the
//! M5.1-02 Python harness; no model arithmetic or artifact is involved.

use std::{env, fs, process};

use clr_storage::{ExpertCache, ExpertId, ExpertKey};

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let arguments: Vec<String> = env::args().skip(1).collect();
    if arguments.len() != 4 || arguments[0] != "--trace" || arguments[2] != "--budget-bytes" {
        return Err("usage: replay_expert_trace --trace <tsv> --budget-bytes <N>".to_owned());
    }
    let trace = fs::read_to_string(&arguments[1]).map_err(|error| error.to_string())?;
    let budget = arguments[3]
        .parse::<usize>()
        .map_err(|_| "invalid cache budget".to_owned())?;
    let mut cache = ExpertCache::new(budget);
    let mut requests = 0_u64;
    for (line_number, line) in trace.lines().enumerate() {
        if line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != 3 {
            return Err(format!(
                "trace line {} must contain layer, expert, payload",
                line_number + 1
            ));
        }
        let layer = fields[0]
            .parse::<u32>()
            .map_err(|_| "invalid layer".to_owned())?;
        let expert = fields[1]
            .parse::<u32>()
            .map_err(|_| "invalid expert".to_owned())?;
        let payload = fields[2]
            .parse::<usize>()
            .map_err(|_| "invalid payload".to_owned())?;
        let key = ExpertKey {
            layer_index: layer,
            expert_id: ExpertId(expert),
        };
        let lease = cache
            .get_or_load(key, || Ok(vec![0_u8; payload]))
            .map_err(|error| error.to_string())?;
        drop(lease);
        requests += 1;
    }
    let metrics = cache.metrics();
    println!(
        "{{\"requests\":{},\"configured_budget_bytes\":{},\"hits\":{},\"misses\":{},\"loads\":{},\"evictions\":{},\"resident_bytes\":{},\"peak_resident_bytes\":{},\"peak_entry_count\":{},\"bytes_read\":{},\"bytes_served_from_cache\":{},\"bytes_avoided\":{},\"oversized_entry_events\":{},\"blocked_eviction_events\":{}}}",
        requests,
        metrics.configured_budget_bytes,
        metrics.hits,
        metrics.misses,
        metrics.loads,
        metrics.evictions,
        metrics.resident_bytes,
        metrics.peak_resident_bytes,
        metrics.peak_entry_count,
        metrics.bytes_read,
        metrics.bytes_served_from_cache,
        metrics.bytes_avoided,
        metrics.oversized_entry_events,
        metrics.blocked_eviction_events,
    );
    Ok(())
}

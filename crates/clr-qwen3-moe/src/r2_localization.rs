//! Test-binary-only timing support for M6.3-R2.0 localization.
//!
//! This collector observes existing operations. It must not make execution
//! decisions or alter arithmetic, cache policy, expert selection, or layout.

use std::{cell::RefCell, collections::BTreeMap, rc::Rc, time::Instant};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Event {
    pub calls: u64,
    pub total_nanos: u128,
    pub exclusive_nanos: u128,
    pub min_nanos: u128,
    pub median_nanos: u128,
    pub max_nanos: u128,
    pub logical_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Snapshot {
    pub enabled: bool,
    pub events: BTreeMap<String, Event>,
    pub counters: BTreeMap<String, u64>,
}
#[derive(Debug, Default)]
struct Aggregate {
    calls: u64,
    total_nanos: u128,
    exclusive_nanos: u128,
    min_nanos: u128,
    max_nanos: u128,
    logical_bytes: u64,
    samples: Vec<u128>,
}

#[derive(Debug, Clone)]
struct Frame {
    name: &'static str,
    started: Instant,
    child_nanos: u128,
}

#[derive(Debug)]
struct Collector {
    enabled: bool,
    stack: Vec<Frame>,
    events: BTreeMap<&'static str, Aggregate>,
    counters: BTreeMap<&'static str, u64>,
}

impl Collector {
    fn new(enabled: bool) -> Self {
        Self {
            enabled,
            stack: Vec::new(),
            events: BTreeMap::new(),
            counters: BTreeMap::new(),
        }
    }
    fn begin_scope(&mut self, name: &'static str) -> Option<Frame> {
        if !self.enabled {
            return None;
        }
        let frame = Frame {
            name,
            started: Instant::now(),
            child_nanos: 0,
        };
        self.stack.push(frame.clone());
        Some(frame)
    }

    fn finish_scope(&mut self, frame: &Frame) {
        let completed = self
            .stack
            .pop()
            .expect("R2.0 timing scope stack is balanced");
        assert_eq!(completed.name, frame.name, "R2.0 timing scope nesting");
        let total_nanos = completed.started.elapsed().as_nanos();
        let exclusive_nanos = total_nanos.saturating_sub(completed.child_nanos);
        if let Some(parent) = self.stack.last_mut() {
            parent.child_nanos = parent.child_nanos.saturating_add(total_nanos);
        }
        let event = self.events.entry(frame.name).or_default();
        event.calls = event.calls.saturating_add(1);
        event.total_nanos = event.total_nanos.saturating_add(total_nanos);
        event.exclusive_nanos = event.exclusive_nanos.saturating_add(exclusive_nanos);
        event.min_nanos = if event.calls == 1 {
            total_nanos
        } else {
            event.min_nanos.min(total_nanos)
        };
        event.max_nanos = event.max_nanos.max(total_nanos);
        event.samples.push(total_nanos);
    }
    fn add_bytes(&mut self, name: &'static str, bytes: u64) {
        if self.enabled {
            let event = self.events.entry(name).or_default();
            event.logical_bytes = event.logical_bytes.saturating_add(bytes);
        }
    }

    fn add_counter(&mut self, name: &'static str, amount: u64) {
        if self.enabled {
            let value = self.counters.entry(name).or_default();
            *value = value.saturating_add(amount);
        }
    }

    fn add_leaf_samples(&mut self, name: &'static str, samples: &[u128]) {
        if !self.enabled || samples.is_empty() {
            return;
        }
        let total_nanos = samples.iter().copied().sum::<u128>();
        if let Some(parent) = self.stack.last_mut() {
            parent.child_nanos = parent.child_nanos.saturating_add(total_nanos);
        }
        let event = self.events.entry(name).or_default();
        event.calls = event.calls.saturating_add(samples.len() as u64);
        event.total_nanos = event.total_nanos.saturating_add(total_nanos);
        event.exclusive_nanos = event.exclusive_nanos.saturating_add(total_nanos);
        let sample_min = *samples.iter().min().expect("non-empty R2.0 samples");
        let sample_max = *samples.iter().max().expect("non-empty R2.0 samples");
        event.min_nanos = if event.samples.is_empty() {
            sample_min
        } else {
            event.min_nanos.min(sample_min)
        };
        event.max_nanos = event.max_nanos.max(sample_max);
        event.samples.extend_from_slice(samples);
    }

    fn snapshot(&self) -> Snapshot {
        let events = self
            .events
            .iter()
            .map(|(name, aggregate)| {
                let mut samples = aggregate.samples.clone();
                samples.sort_unstable();
                let median_nanos = samples.get(samples.len() / 2).copied().unwrap_or_default();
                (
                    (*name).to_owned(),
                    Event {
                        calls: aggregate.calls,
                        total_nanos: aggregate.total_nanos,
                        exclusive_nanos: aggregate.exclusive_nanos,
                        min_nanos: aggregate.min_nanos,
                        median_nanos,
                        max_nanos: aggregate.max_nanos,
                        logical_bytes: aggregate.logical_bytes,
                    },
                )
            })
            .collect();
        Snapshot {
            enabled: self.enabled,
            events,
            counters: self
                .counters
                .iter()
                .map(|(name, value)| ((*name).to_owned(), *value))
                .collect(),
        }
    }
}

thread_local! {
    static ACTIVE: RefCell<Option<Rc<RefCell<Collector>>>> = const { RefCell::new(None) };
}

pub(crate) struct Session {
    collector: Rc<RefCell<Collector>>,
    previous: Option<Rc<RefCell<Collector>>>,
}

pub(crate) struct Scope {
    collector: Option<Rc<RefCell<Collector>>>,
    frame: Option<Frame>,
}

pub(crate) fn start(enabled: bool) -> Session {
    let collector = Rc::new(RefCell::new(Collector::new(enabled)));
    let previous = ACTIVE.with(|active| active.replace(Some(Rc::clone(&collector))));
    Session {
        collector,
        previous,
    }
}
pub(crate) fn finish(session: Session) -> Snapshot {
    let snapshot = session.collector.borrow().snapshot();
    ACTIVE.with(|active| {
        let current = active.replace(session.previous);
        assert!(
            current
                .as_ref()
                .is_some_and(|value| Rc::ptr_eq(value, &session.collector)),
            "R2.0 timing session nesting is balanced"
        );
    });
    snapshot
}

pub(crate) fn scope(name: &'static str) -> Scope {
    ACTIVE.with(|active| {
        let collector = active.borrow().clone();
        let frame = collector
            .as_ref()
            .and_then(|value| value.borrow_mut().begin_scope(name));
        Scope { collector, frame }
    })
}

pub(crate) fn record_bytes(name: &'static str, bytes: u64) {
    ACTIVE.with(|active| {
        if let Some(collector) = active.borrow().as_ref() {
            collector.borrow_mut().add_bytes(name, bytes);
        }
    });
}

pub(crate) fn add_counter(name: &'static str, amount: u64) {
    ACTIVE.with(|active| {
        if let Some(collector) = active.borrow().as_ref() {
            collector.borrow_mut().add_counter(name, amount);
        }
    });
}

pub(crate) fn is_enabled() -> bool {
    ACTIVE.with(|active| {
        active
            .borrow()
            .as_ref()
            .is_some_and(|collector| collector.borrow().enabled)
    })
}

pub(crate) fn record_leaf_samples(name: &'static str, samples: &[u128]) {
    ACTIVE.with(|active| {
        if let Some(collector) = active.borrow().as_ref() {
            collector.borrow_mut().add_leaf_samples(name, samples);
        }
    });
}

impl Drop for Scope {
    fn drop(&mut self) {
        if let (Some(collector), Some(frame)) = (self.collector.take(), self.frame.take()) {
            collector.borrow_mut().finish_scope(&frame);
        }
    }
}

pub(crate) fn calibrate_noop(iterations: usize) -> Event {
    assert!(
        iterations > 0,
        "R2.0 timer calibration iterations must be positive"
    );
    let session = start(true);
    for _ in 0..iterations {
        drop(scope("timer.noop"));
    }
    finish(session)
        .events
        .remove("timer.noop")
        .expect("R2.0 no-op timer event")
}

#[cfg(test)]
mod tests {
    use super::{add_counter, calibrate_noop, finish, record_bytes, scope, start};

    #[test]
    fn disabled_session_records_nothing() {
        let session = start(false);
        drop(scope("expert_total"));
        record_bytes("packed_value_read", 64);
        add_counter("expert_occurrences", 1);
        let snapshot = finish(session);
        assert!(!snapshot.enabled);
        assert!(snapshot.events.is_empty());
        assert!(snapshot.counters.is_empty());
    }
    #[test]
    fn nested_scopes_reconcile_and_keep_bytes() {
        let session = start(true);
        {
            let _parent = scope("expert_total");
            {
                let _child = scope("gate_projection");
                record_bytes("gate_projection", 128);
            }
            add_counter("expert_occurrences", 2);
        }
        let snapshot = finish(session);
        let parent = &snapshot.events["expert_total"];
        let child = &snapshot.events["gate_projection"];
        assert_eq!(parent.calls, 1);
        assert_eq!(child.calls, 1);
        assert!(parent.total_nanos >= child.total_nanos);
        assert_eq!(
            parent.total_nanos,
            parent.exclusive_nanos + child.total_nanos
        );
        assert_eq!(child.logical_bytes, 128);
        assert_eq!(snapshot.counters["expert_occurrences"], 2);
    }

    #[test]
    fn noop_calibration_reports_distribution() {
        let event = calibrate_noop(31);
        assert_eq!(event.calls, 31);
        assert!(event.min_nanos <= event.median_nanos);
        assert!(event.median_nanos <= event.max_nanos);
        assert!(event.total_nanos >= event.max_nanos);
    }

    #[test]
    fn f32_expert_mlp_scopes_preserve_exact_output() {
        let input = [0.25_f32, -0.5];
        let gate = [0.5_f32, 0.25, -0.25, 0.75];
        let up = [0.75_f32, -0.5, 0.25, 0.5];
        let down = [0.5_f32, -0.25, 0.75, 0.125];
        let baseline_session = start(false);
        let baseline = crate::block::expert_mlp(&input, &gate, &up, &down, 2, 2);
        let baseline_snapshot = finish(baseline_session);
        assert!(baseline_snapshot.events.is_empty());
        let session = start(true);
        let observed = crate::block::expert_mlp(&input, &gate, &up, &down, 2, 2);
        let snapshot = finish(session);
        assert_eq!(observed, baseline);
        for name in [
            "gate_projection",
            "up_projection",
            "activation_product",
            "down_projection",
        ] {
            assert_eq!(snapshot.events[name].calls, 2, "{name} calls");
        }
    }
}

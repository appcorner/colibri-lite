//! Feature-gated hierarchical timing for the M5.3-03 full-model study.
//!
//! This module is test-binary instrumentation only. It is deliberately not
//! part of the default runtime and contains no execution decisions. Timing
//! fields are directional evidence; event names, phases, dimensions, and
//! counters are the deterministic parts of the profile contract.

use std::{
    cell::RefCell, collections::BTreeMap, env, fmt::Write as _, fs, path::Path, rc::Rc,
    time::Instant,
};

const PROFILE_SCHEMA: &str = "colibri-qwen3-moe-m5.3-03-compute-profile-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProfileMode {
    Disabled,
    Coarse,
    Detailed,
}

impl ProfileMode {
    fn from_env() -> Self {
        match env::var("COLIBRI_COMPUTE_PROFILE_MODE")
            .unwrap_or_else(|_| "disabled".to_owned())
            .as_str()
        {
            "disabled" => Self::Disabled,
            "coarse" => Self::Coarse,
            "detailed" => Self::Detailed,
            other => panic!(
                "COLIBRI_COMPUTE_PROFILE_MODE must be disabled, coarse, or detailed: {other}"
            ),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Coarse => "coarse",
            Self::Detailed => "detailed",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct EventKey {
    phase: String,
    layer: Option<usize>,
    operation: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct MatrixShape {
    rows: usize,
    outputs: usize,
    inputs: usize,
}

#[derive(Debug, Default, Clone)]
struct MatrixAggregate {
    calls: u64,
    estimated_flops: u128,
    input_bytes: u128,
    weight_bytes: u128,
    output_bytes: u128,
}

#[derive(Debug, Default, Clone)]
struct Aggregate {
    calls: u64,
    total_nanos: u128,
    exclusive_nanos: u128,
    min_nanos: u128,
    max_nanos: u128,
    estimated_flops: u128,
    input_bytes: u128,
    output_bytes: u128,
    matrices: BTreeMap<MatrixShape, MatrixAggregate>,
}

#[derive(Debug, Clone)]
struct ScopeFrame {
    key: EventKey,
    started: Instant,
    child_nanos: u128,
    matrix: Option<MatrixShape>,
}

#[derive(Debug)]
struct Collector {
    mode: ProfileMode,
    phase: String,
    layer: Option<usize>,
    stack: Vec<ScopeFrame>,
    events: BTreeMap<EventKey, Aggregate>,
    scope_count: u64,
}

impl Collector {
    fn new(mode: ProfileMode) -> Self {
        Self {
            mode,
            phase: "initialization".to_owned(),
            layer: None,
            stack: Vec::new(),
            events: BTreeMap::new(),
            scope_count: 0,
        }
    }

    fn accepts(&self, operation: &str) -> bool {
        match self.mode {
            ProfileMode::Disabled => false,
            ProfileMode::Detailed => true,
            ProfileMode::Coarse => {
                operation.starts_with("model.")
                    || operation.starts_with("decoder.layer")
                    || operation.starts_with("attention.")
                    || operation.starts_with("experts.")
                    || operation.starts_with("expert.mlp")
                    || operation.starts_with("lm_head")
                    || operation.starts_with("embedding.")
                    || operation.starts_with("final_norm")
                    || operation.starts_with("cache.")
            }
        }
    }

    fn begin_scope(&mut self, operation: &str, matrix: Option<MatrixShape>) -> Option<ScopeFrame> {
        if !self.accepts(operation) {
            return None;
        }
        let frame = ScopeFrame {
            key: EventKey {
                phase: self.phase.clone(),
                layer: self.layer,
                operation: operation.to_owned(),
            },
            started: Instant::now(),
            child_nanos: 0,
            matrix,
        };
        self.scope_count = self.scope_count.saturating_add(1);
        self.stack.push(frame.clone());
        Some(frame)
    }

    fn finish_scope(&mut self, frame: ScopeFrame) {
        let completed = self.stack.pop().expect("profile scope stack is balanced");
        assert_eq!(
            completed.key, frame.key,
            "profile scope nesting is balanced"
        );
        let total_nanos = frame.started.elapsed().as_nanos();
        let exclusive_nanos = total_nanos.saturating_sub(frame.child_nanos);
        if let Some(parent) = self.stack.last_mut() {
            parent.child_nanos = parent.child_nanos.saturating_add(total_nanos);
        }
        let event = self.events.entry(frame.key).or_default();
        event.calls = event.calls.saturating_add(1);
        event.total_nanos = event.total_nanos.saturating_add(total_nanos);
        event.exclusive_nanos = event.exclusive_nanos.saturating_add(exclusive_nanos);
        event.min_nanos = if event.calls == 1 {
            total_nanos
        } else {
            event.min_nanos.min(total_nanos)
        };
        event.max_nanos = event.max_nanos.max(total_nanos);
        if let Some(shape) = frame.matrix {
            let flops = (shape.rows as u128)
                .saturating_mul(shape.outputs as u128)
                .saturating_mul(shape.inputs as u128)
                .saturating_mul(2);
            let input_bytes = (shape.rows as u128)
                .saturating_mul(shape.inputs as u128)
                .saturating_mul(4);
            let weight_bytes = (shape.outputs as u128)
                .saturating_mul(shape.inputs as u128)
                .saturating_mul(4);
            let output_bytes = (shape.rows as u128)
                .saturating_mul(shape.outputs as u128)
                .saturating_mul(4);
            event.estimated_flops = event.estimated_flops.saturating_add(flops);
            event.input_bytes = event.input_bytes.saturating_add(input_bytes);
            event.output_bytes = event.output_bytes.saturating_add(output_bytes);
            let matrix = event.matrices.entry(shape).or_default();
            matrix.calls = matrix.calls.saturating_add(1);
            matrix.estimated_flops = matrix.estimated_flops.saturating_add(flops);
            matrix.input_bytes = matrix.input_bytes.saturating_add(input_bytes);
            matrix.weight_bytes = matrix.weight_bytes.saturating_add(weight_bytes);
            matrix.output_bytes = matrix.output_bytes.saturating_add(output_bytes);
        }
    }

    fn snapshot(&self) -> ProfileSnapshot {
        let events = self
            .events
            .iter()
            .map(|(key, value)| ProfileEvent {
                phase: key.phase.clone(),
                layer: key.layer,
                operation: key.operation.clone(),
                calls: value.calls,
                total_nanos: value.total_nanos,
                exclusive_nanos: value.exclusive_nanos,
                min_nanos: value.min_nanos,
                max_nanos: value.max_nanos,
                estimated_flops: value.estimated_flops,
                input_bytes: value.input_bytes,
                output_bytes: value.output_bytes,
                matrices: value
                    .matrices
                    .iter()
                    .map(|(shape, matrix)| MatrixEvent {
                        rows: shape.rows,
                        outputs: shape.outputs,
                        inputs: shape.inputs,
                        calls: matrix.calls,
                        estimated_flops: matrix.estimated_flops,
                        input_bytes: matrix.input_bytes,
                        weight_bytes: matrix.weight_bytes,
                        output_bytes: matrix.output_bytes,
                    })
                    .collect(),
            })
            .collect();
        ProfileSnapshot {
            mode: self.mode,
            scope_count: self.scope_count,
            events,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MatrixEvent {
    pub rows: usize,
    pub outputs: usize,
    pub inputs: usize,
    pub calls: u64,
    pub estimated_flops: u128,
    pub input_bytes: u128,
    pub weight_bytes: u128,
    pub output_bytes: u128,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ProfileEvent {
    pub phase: String,
    pub layer: Option<usize>,
    pub operation: String,
    pub calls: u64,
    pub total_nanos: u128,
    pub exclusive_nanos: u128,
    pub min_nanos: u128,
    pub max_nanos: u128,
    pub estimated_flops: u128,
    pub input_bytes: u128,
    pub output_bytes: u128,
    pub matrices: Vec<MatrixEvent>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ProfileSnapshot {
    pub mode: ProfileMode,
    pub scope_count: u64,
    pub events: Vec<ProfileEvent>,
}

thread_local! {
    static ACTIVE: RefCell<Option<Rc<RefCell<Collector>>>> = const { RefCell::new(None) };
}

pub(crate) struct Session {
    collector: Rc<RefCell<Collector>>,
    previous: Option<Rc<RefCell<Collector>>>,
}

pub(crate) fn start_from_env() -> Session {
    let collector = Rc::new(RefCell::new(Collector::new(ProfileMode::from_env())));
    let previous = ACTIVE.with(|active| active.replace(Some(Rc::clone(&collector))));
    Session {
        collector,
        previous,
    }
}

pub(crate) fn finish(session: Session) -> ProfileSnapshot {
    let snapshot = session.collector.borrow().snapshot();
    ACTIVE.with(|active| {
        let replaced = active.replace(session.previous);
        assert!(
            replaced.is_some(),
            "profiling session must remain active until finish"
        );
    });
    snapshot
}

pub(crate) fn set_phase(phase: impl Into<String>) {
    ACTIVE.with(|active| {
        if let Some(collector) = active.borrow().as_ref() {
            collector.borrow_mut().phase = phase.into();
        }
    });
}

pub(crate) fn set_layer(layer: Option<usize>) {
    ACTIVE.with(|active| {
        if let Some(collector) = active.borrow().as_ref() {
            collector.borrow_mut().layer = layer;
        }
    });
}

pub(crate) fn scope(operation: &str) -> Scope {
    scope_with_matrix(operation, None)
}

pub(crate) fn matrix_scope(operation: &str, rows: usize, outputs: usize, inputs: usize) -> Scope {
    scope_with_matrix(
        operation,
        Some(MatrixShape {
            rows,
            outputs,
            inputs,
        }),
    )
}

fn scope_with_matrix(operation: &str, matrix: Option<MatrixShape>) -> Scope {
    ACTIVE.with(|active| {
        let collector = active.borrow().as_ref().cloned();
        let frame = collector
            .as_ref()
            .and_then(|collector| collector.borrow_mut().begin_scope(operation, matrix));
        Scope { collector, frame }
    })
}

pub(crate) struct Scope {
    collector: Option<Rc<RefCell<Collector>>>,
    frame: Option<ScopeFrame>,
}

impl Drop for Scope {
    fn drop(&mut self) {
        if let (Some(collector), Some(frame)) = (self.collector.take(), self.frame.take()) {
            collector.borrow_mut().finish_scope(frame);
        }
    }
}

fn json_string(value: &str) -> String {
    let mut output = String::with_capacity(value.len() + 2);
    output.push('"');
    for character in value.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            character if character.is_control() => {
                write!(output, "\\u{:04x}", character as u32).expect("write JSON escape");
            }
            character => output.push(character),
        }
    }
    output.push('"');
    output
}

impl ProfileSnapshot {
    pub(crate) fn to_json(
        &self,
        fixture_id: &str,
        cache_budget: usize,
        input_token_ids: &[usize],
        generated_token_ids: &[usize],
    ) -> String {
        let mut output = String::new();
        write!(
            output,
            "{{\"schema\":{schema},\"schema_version\":1,\"fixture_id\":{fixture},\"cache_budget_bytes\":{budget},\"input_token_ids\":{input:?},\"generated_token_ids\":{generated:?},\"mode\":{mode},\"scope_count\":{},\"events\":[",
            self.scope_count,
            schema = json_string(PROFILE_SCHEMA),
            fixture = json_string(fixture_id),
            budget = cache_budget,
            input = input_token_ids,
            generated = generated_token_ids,
            mode = json_string(self.mode.name()),
        )
        .expect("write profile JSON header");
        for (index, event) in self.events.iter().enumerate() {
            if index != 0 {
                output.push(',');
            }
            write!(
                output,
                "{{\"phase\":{},\"layer\":{},\"operation\":{},\"calls\":{},\"total_nanos\":{},\"exclusive_nanos\":{},\"min_nanos\":{},\"max_nanos\":{},\"estimated_flops\":{},\"input_bytes\":{},\"output_bytes\":{},\"matrices\":[",
                json_string(&event.phase),
                event
                    .layer
                    .map_or_else(|| "null".to_owned(), |value| value.to_string()),
                json_string(&event.operation),
                event.calls,
                event.total_nanos,
                event.exclusive_nanos,
                event.min_nanos,
                event.max_nanos,
                event.estimated_flops,
                event.input_bytes,
                event.output_bytes,
            )
            .expect("write profile event");
            for (matrix_index, matrix) in event.matrices.iter().enumerate() {
                if matrix_index != 0 {
                    output.push(',');
                }
                write!(
                    output,
                    "{{\"rows\":{},\"outputs\":{},\"inputs\":{},\"calls\":{},\"estimated_flops\":{},\"input_bytes\":{},\"weight_bytes\":{},\"output_bytes\":{}}}",
                    matrix.rows,
                    matrix.outputs,
                    matrix.inputs,
                    matrix.calls,
                    matrix.estimated_flops,
                    matrix.input_bytes,
                    matrix.weight_bytes,
                    matrix.output_bytes,
                )
                .expect("write matrix profile event");
            }
            output.push_str("]}");
        }
        output.push_str("]}\n");
        output
    }
}

pub(crate) fn write_json(
    path: &Path,
    snapshot: &ProfileSnapshot,
    fixture_id: &str,
    cache_budget: usize,
    input_token_ids: &[usize],
    generated_token_ids: &[usize],
) {
    fs::write(
        path,
        snapshot.to_json(
            fixture_id,
            cache_budget,
            input_token_ids,
            generated_token_ids,
        ),
    )
    .expect("write compute profile output");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_scopes_reconcile_inclusive_and_exclusive_time() {
        let collector = Rc::new(RefCell::new(Collector::new(ProfileMode::Detailed)));
        let outer = collector
            .borrow_mut()
            .begin_scope("model.total", None)
            .expect("outer scope");
        let inner = collector
            .borrow_mut()
            .begin_scope(
                "lm_head",
                Some(MatrixShape {
                    rows: 1,
                    outputs: 3,
                    inputs: 2,
                }),
            )
            .expect("inner scope");
        collector.borrow_mut().finish_scope(inner);
        collector.borrow_mut().finish_scope(outer);
        let snapshot = collector.borrow().snapshot();
        assert_eq!(snapshot.events.len(), 2);
        assert!(
            snapshot
                .events
                .iter()
                .all(|event| event.calls == 1 && event.total_nanos >= event.exclusive_nanos)
        );
        let matrix = snapshot
            .events
            .iter()
            .find(|event| event.operation == "lm_head")
            .expect("matrix event")
            .matrices
            .first()
            .expect("matrix shape");
        assert_eq!((matrix.rows, matrix.outputs, matrix.inputs), (1, 3, 2));
        assert_eq!(matrix.estimated_flops, 12);
    }

    #[test]
    fn disabled_mode_records_no_scopes() {
        let mut collector = Collector::new(ProfileMode::Disabled);
        assert!(collector.begin_scope("model.total", None).is_none());
        assert_eq!(collector.scope_count, 0);
    }
}

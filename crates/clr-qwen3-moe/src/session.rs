use std::fmt;

use clr_core::{RuntimeError, Tensor, TensorShape, TensorView, ops::elementwise_add};
use clr_storage::ExpertStore;

use crate::{
    KvCache, Qwen3MoeConfig, Qwen3MoeModel, SeededRng, StreamingModelError, StreamingQwen3MoeModel,
    block::{
        RouterOutput, cached_attention_with_weights, linear, rms_norm, route_tokens, routed_experts,
    },
    cache::{LayerKvUpdate, LayerKvView},
    generation::{greedy_token, sample_token},
    model::embedding_lookup,
    streaming::streaming_routed_experts,
};

/// Error produced by resident or storage-aware generation sessions.
#[derive(Debug)]
pub enum GenerationError {
    /// In-memory model or cache validation failed.
    Runtime(RuntimeError),
    /// Storage-aware execution failed.
    Streaming(StreamingModelError),
}

impl From<RuntimeError> for GenerationError {
    fn from(error: RuntimeError) -> Self {
        Self::Runtime(error)
    }
}

impl From<StreamingModelError> for GenerationError {
    fn from(error: StreamingModelError) -> Self {
        Self::Streaming(error)
    }
}

impl fmt::Display for GenerationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Runtime(error) => write!(formatter, "runtime error: {error}"),
            Self::Streaming(error) => write!(formatter, "streaming error: {error}"),
        }
    }
}

impl std::error::Error for GenerationError {}

/// Numerical evidence returned after filling an empty generation cache.
#[derive(Debug, Clone, PartialEq)]
pub struct PrefillOutput {
    /// Per-token language-model logits `[sequence, vocabulary]`.
    pub logits: Tensor,
    /// Exact expert IDs, grouped by layer in token-major order.
    pub selected_experts: Vec<Vec<usize>>,
}

#[derive(Debug)]
enum GenerationBackend<'a> {
    Resident(&'a Qwen3MoeModel),
    Streaming {
        model: &'a StreamingQwen3MoeModel,
        store: &'a mut ExpertStore,
    },
}

/// Fixed-context autoregressive state for one resident or streaming model.
#[derive(Debug)]
pub struct GenerationSession<'a> {
    backend: GenerationBackend<'a>,
    cache: KvCache,
    sequence: Vec<usize>,
    last_logits: Option<Tensor>,
    rng: SeededRng,
}

impl<'a> GenerationSession<'a> {
    /// Creates an empty fixed-context session for a resident model.
    ///
    /// # Errors
    ///
    /// Returns a structured error for zero capacity, a capacity above the
    /// model context limit, or checked cache-size overflow.
    pub fn resident(
        model: &'a Qwen3MoeModel,
        capacity: usize,
        seed: u64,
    ) -> Result<Self, GenerationError> {
        Ok(Self {
            backend: GenerationBackend::Resident(model),
            cache: cache_for_config(model.config, capacity)?,
            sequence: Vec::with_capacity(capacity),
            last_logits: None,
            rng: SeededRng::new(seed),
        })
    }

    /// Creates an empty fixed-context session with on-demand expert loading.
    ///
    /// # Errors
    ///
    /// Returns a structured error for zero capacity, a capacity above the
    /// model context limit, or checked cache-size overflow.
    pub fn streaming(
        model: &'a StreamingQwen3MoeModel,
        store: &'a mut ExpertStore,
        capacity: usize,
        seed: u64,
    ) -> Result<Self, GenerationError> {
        Ok(Self {
            backend: GenerationBackend::Streaming { model, store },
            cache: cache_for_config(model.config, capacity)?,
            sequence: Vec::with_capacity(capacity),
            last_logits: None,
            rng: SeededRng::new(seed),
        })
    }

    /// Processes a non-empty prompt into an empty fixed-capacity cache.
    ///
    /// Capacity and token IDs are validated before cache or sequence mutation.
    ///
    /// # Errors
    ///
    /// Returns a context, token-ID, numerical, expert-storage, or repeated-
    /// prefill error.
    pub fn prefill(&mut self, token_ids: &[usize]) -> Result<PrefillOutput, GenerationError> {
        if !self.cache.is_empty() || !self.sequence.is_empty() {
            return Err(RuntimeError::InvalidShape {
                reason: "prefill requires an empty generation session",
            }
            .into());
        }
        if token_ids.is_empty() {
            return Err(RuntimeError::InvalidShape {
                reason: "prefill token ID sequence must not be empty",
            }
            .into());
        }
        self.ensure_can_append(token_ids.len())?;
        let vocabulary = self.config().model().vocabulary_size();
        if let Some(token_id) = token_ids.iter().find(|token_id| **token_id >= vocabulary) {
            return Err(RuntimeError::IndexOutOfBounds {
                index: *token_id,
                length: vocabulary,
            }
            .into());
        }

        let mut logits = Vec::with_capacity(token_ids.len() * vocabulary);
        let mut selected_experts = vec![Vec::new(); self.config().model().layer_count()];
        for token_id in token_ids {
            let output = self.forward_one(*token_id)?;
            logits.extend_from_slice(output.logits.data());
            for (layer_selected, token_selected) in
                selected_experts.iter_mut().zip(&output.selected_experts)
            {
                layer_selected.extend_from_slice(token_selected);
            }
            self.last_logits = Some(output.logits);
            self.sequence.push(*token_id);
        }
        Ok(PrefillOutput {
            logits: Tensor::new(TensorShape::new([token_ids.len(), vocabulary]), logits)?,
            selected_experts,
        })
    }

    /// Selects and processes one greedy token using the current cached logits.
    ///
    /// # Errors
    ///
    /// Returns a structured error when prefill has not run, the context is
    /// full, logits are invalid, or model/storage execution fails.
    pub fn decode_greedy(&mut self) -> Result<usize, GenerationError> {
        self.ensure_can_append(1)?;
        let token_id = greedy_token(self.current_logits()?.view())?;
        self.process_decoded_token(token_id)?;
        Ok(token_id)
    }

    /// Selects and processes one reproducibly sampled temperature token.
    ///
    /// RNG state is committed only after model execution and cache append
    /// succeed.
    ///
    /// # Errors
    ///
    /// Returns a structured error when prefill has not run, the context is
    /// full, temperature/logits are invalid, or model/storage execution fails.
    pub fn decode_temperature(&mut self, temperature: f32) -> Result<usize, GenerationError> {
        self.ensure_can_append(1)?;
        let mut next_rng = self.rng;
        let token_id = sample_token(self.current_logits()?.view(), temperature, &mut next_rng)?;
        self.process_decoded_token(token_id)?;
        self.rng = next_rng;
        Ok(token_id)
    }

    /// Returns the initialized token IDs, including later decoded tokens.
    #[must_use]
    pub fn sequence(&self) -> &[usize] {
        &self.sequence
    }

    /// Returns fixed cache accounting and initialized length.
    #[must_use]
    pub const fn cache(&self) -> &KvCache {
        &self.cache
    }

    fn config(&self) -> Qwen3MoeConfig {
        match &self.backend {
            GenerationBackend::Resident(model) => model.config,
            GenerationBackend::Streaming { model, .. } => model.config,
        }
    }

    fn ensure_can_append(&self, count: usize) -> Result<(), RuntimeError> {
        let requested =
            self.cache
                .len()
                .checked_add(count)
                .ok_or(RuntimeError::ArithmeticOverflow {
                    operation: "generation context length",
                })?;
        if requested > self.cache.capacity() {
            return Err(RuntimeError::ContextLengthExceeded {
                requested,
                capacity: self.cache.capacity(),
            });
        }
        Ok(())
    }

    fn current_logits(&self) -> Result<&Tensor, RuntimeError> {
        self.last_logits.as_ref().ok_or(RuntimeError::InvalidShape {
            reason: "decode requires a successful prefill",
        })
    }

    fn process_decoded_token(&mut self, token_id: usize) -> Result<(), GenerationError> {
        let output = self.forward_one(token_id)?;
        self.last_logits = Some(output.logits);
        self.sequence.push(token_id);
        Ok(())
    }

    fn forward_one(&mut self, token_id: usize) -> Result<TokenForward, GenerationError> {
        let output = match &mut self.backend {
            GenerationBackend::Resident(model) => {
                resident_forward_one(model, &self.cache, token_id)?
            }
            GenerationBackend::Streaming { model, store } => {
                streaming_forward_one(model, store, &self.cache, token_id)?
            }
        };
        let updates: Vec<_> = output
            .key_values
            .iter()
            .map(|values| LayerKvUpdate {
                key: &values.key,
                value: &values.value,
            })
            .collect();
        self.cache.append_token(&updates)?;
        Ok(output)
    }
}

fn cache_for_config(config: Qwen3MoeConfig, capacity: usize) -> Result<KvCache, RuntimeError> {
    if capacity == 0 {
        return Err(RuntimeError::InvalidModelConfig {
            field: "context_capacity",
            reason: "must be greater than zero",
        });
    }
    let model_limit = config.model().max_sequence_length();
    if capacity > model_limit {
        return Err(RuntimeError::ContextLengthExceeded {
            requested: capacity,
            capacity: model_limit,
        });
    }
    KvCache::new(
        config.model().layer_count(),
        capacity,
        config.model().key_value_head_count(),
        config.head_dimension(),
    )
}

struct OwnedLayerKv {
    key: Vec<f32>,
    value: Vec<f32>,
}

struct TokenForward {
    logits: Tensor,
    selected_experts: Vec<Vec<usize>>,
    key_values: Vec<OwnedLayerKv>,
}

struct CachedBlockOutput {
    hidden: Tensor,
    selected_experts: Vec<usize>,
    key_values: OwnedLayerKv,
}

#[allow(clippy::too_many_arguments)]
fn cached_block<E, F>(
    hidden: TensorView<'_>,
    config: Qwen3MoeConfig,
    cache: LayerKvView<'_>,
    input_norm_weight: TensorView<'_>,
    query_weight: TensorView<'_>,
    key_weight: TensorView<'_>,
    value_weight: TensorView<'_>,
    output_weight: TensorView<'_>,
    query_norm_weight: TensorView<'_>,
    key_norm_weight: TensorView<'_>,
    post_attention_norm_weight: TensorView<'_>,
    router_weight: TensorView<'_>,
    compute_experts: F,
) -> Result<CachedBlockOutput, E>
where
    E: From<RuntimeError>,
    F: FnOnce(TensorView<'_>, &RouterOutput) -> Result<Tensor, E>,
{
    let input_norm = rms_norm(hidden, input_norm_weight, config.rms_norm_epsilon())?;
    let attention = cached_attention_with_weights(
        input_norm.view(),
        config,
        query_weight,
        key_weight,
        value_weight,
        output_weight,
        query_norm_weight,
        key_norm_weight,
        cache,
    )?;
    let after_attention = elementwise_add(hidden, attention.output.view())?;
    let post_attention_norm = rms_norm(
        after_attention.view(),
        post_attention_norm_weight,
        config.rms_norm_epsilon(),
    )?;
    let router = route_tokens(post_attention_norm.view(), router_weight, config)?;
    let moe_output = compute_experts(post_attention_norm.view(), &router)?;
    let hidden = elementwise_add(after_attention.view(), moe_output.view())?;
    Ok(CachedBlockOutput {
        hidden,
        selected_experts: router.selected_experts,
        key_values: OwnedLayerKv {
            key: attention.key,
            value: attention.value,
        },
    })
}

fn resident_forward_one(
    model: &Qwen3MoeModel,
    cache: &KvCache,
    token_id: usize,
) -> Result<TokenForward, RuntimeError> {
    let mut hidden = embedding_lookup(
        model.token_embeddings.view(),
        &[token_id],
        model.config.model().vocabulary_size(),
    )?;
    let mut selected_experts = Vec::with_capacity(model.blocks.len());
    let mut key_values = Vec::with_capacity(model.blocks.len());
    for (layer, block) in model.blocks.iter().enumerate() {
        let weights = &block.weights;
        let output = cached_block(
            hidden.view(),
            model.config,
            cache.layer(layer)?,
            weights.input_norm.view(),
            weights.query_projection.view(),
            weights.key_projection.view(),
            weights.value_projection.view(),
            weights.output_projection.view(),
            weights.query_norm.view(),
            weights.key_norm.view(),
            weights.post_attention_norm.view(),
            weights.router.view(),
            |normalized, router| {
                routed_experts(
                    normalized,
                    weights.expert_gate_up.view(),
                    weights.expert_down.view(),
                    router,
                    model.config,
                )
            },
        )?;
        hidden = output.hidden;
        selected_experts.push(output.selected_experts);
        key_values.push(output.key_values);
    }
    let final_norm = rms_norm(
        hidden.view(),
        model.final_norm.view(),
        model.config.rms_norm_epsilon(),
    )?;
    let logits = linear(
        final_norm.view(),
        model.language_model_head.view(),
        "cached language model head",
    )?;
    Ok(TokenForward {
        logits,
        selected_experts,
        key_values,
    })
}

fn streaming_forward_one(
    model: &StreamingQwen3MoeModel,
    store: &mut ExpertStore,
    cache: &KvCache,
    token_id: usize,
) -> Result<TokenForward, StreamingModelError> {
    let mut hidden = embedding_lookup(
        model.weights.token_embeddings.view(),
        &[token_id],
        model.config.model().vocabulary_size(),
    )?;
    let mut selected_experts = Vec::with_capacity(model.weights.blocks.len());
    let mut key_values = Vec::with_capacity(model.weights.blocks.len());
    for (layer, weights) in model.weights.blocks.iter().enumerate() {
        let output = cached_block(
            hidden.view(),
            model.config,
            cache.layer(layer)?,
            weights.input_norm.view(),
            weights.query_projection.view(),
            weights.key_projection.view(),
            weights.value_projection.view(),
            weights.output_projection.view(),
            weights.query_norm.view(),
            weights.key_norm.view(),
            weights.post_attention_norm.view(),
            weights.router.view(),
            |normalized, router| {
                streaming_routed_experts(
                    normalized,
                    router,
                    model.config,
                    layer,
                    store,
                    model.layout,
                )
            },
        )?;
        hidden = output.hidden;
        selected_experts.push(output.selected_experts);
        key_values.push(output.key_values);
    }
    let final_norm = rms_norm(
        hidden.view(),
        model.weights.final_norm.view(),
        model.config.rms_norm_epsilon(),
    )?;
    let logits = linear(
        final_norm.view(),
        model.weights.language_model_head.view(),
        "cached language model head",
    )?;
    Ok(TokenForward {
        logits,
        selected_experts,
        key_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Qwen3MoeModelWeightsSpec, test_fixture};

    fn fixture_model() -> Qwen3MoeModel {
        Qwen3MoeModel::new(
            test_fixture::config(),
            Qwen3MoeModelWeightsSpec {
                token_embeddings: test_fixture::token_embeddings(),
                blocks: vec![
                    test_fixture::block_weights(0),
                    test_fixture::block_weights(1),
                ],
                final_norm: test_fixture::final_norm_weight(),
                language_model_head: test_fixture::language_model_head(),
            },
        )
        .expect("fixture model")
    }

    fn assert_tensor_close(actual: &Tensor, expected: &Tensor) {
        assert_eq!(actual.shape(), expected.shape());
        for (index, (actual, expected)) in actual.data().iter().zip(expected.data()).enumerate() {
            let tolerance = 1.0e-6 + 1.0e-5 * expected.abs();
            assert!(
                (actual - expected).abs() <= tolerance,
                "prefill mismatch at logits[{index}]: {actual} vs {expected}"
            );
        }
    }

    #[test]
    fn resident_prefill_matches_stateless_logits_and_experts() {
        let model = fixture_model();
        let prompt = test_fixture::token_ids();
        let expected = model.forward(&prompt).expect("stateless forward");
        let mut session =
            GenerationSession::resident(&model, model.config.model().max_sequence_length(), 0)
                .expect("session");

        let actual = session.prefill(&prompt).expect("prefill");

        assert_tensor_close(&actual.logits, &expected.logits);
        assert_eq!(
            actual.selected_experts,
            expected
                .block_outputs
                .iter()
                .map(|block| block.selected_experts.clone())
                .collect::<Vec<_>>()
        );
        assert_eq!(session.sequence(), prompt);
        assert_eq!(session.cache().len(), prompt.len());
    }

    #[test]
    fn prefill_validates_capacity_and_ids_before_mutation() {
        let model = fixture_model();
        let prompt = test_fixture::token_ids();
        let mut session =
            GenerationSession::resident(&model, prompt.len() - 1, 0).expect("small session");

        assert!(matches!(
            session.prefill(&prompt),
            Err(GenerationError::Runtime(
                RuntimeError::ContextLengthExceeded { .. }
            ))
        ));
        assert!(session.sequence().is_empty());
        assert!(session.cache().is_empty());

        let mut session = GenerationSession::resident(&model, prompt.len(), 0).expect("session");
        assert!(matches!(
            session.prefill(&[model.config.model().vocabulary_size()]),
            Err(GenerationError::Runtime(
                RuntimeError::IndexOutOfBounds { .. }
            ))
        ));
        assert!(session.sequence().is_empty());
        assert!(session.cache().is_empty());
    }

    #[test]
    fn session_rejects_invalid_context_capacity() {
        let model = fixture_model();
        assert!(matches!(
            GenerationSession::resident(&model, 0, 0),
            Err(GenerationError::Runtime(RuntimeError::InvalidModelConfig {
                field: "context_capacity",
                ..
            }))
        ));
        assert!(matches!(
            GenerationSession::resident(&model, model.config.model().max_sequence_length() + 1, 0,),
            Err(GenerationError::Runtime(
                RuntimeError::ContextLengthExceeded { .. }
            ))
        ));
    }

    #[test]
    fn cached_greedy_decode_matches_recomputing_generation() {
        let model = fixture_model();
        let prompt = test_fixture::token_ids();
        let expected = model
            .generate_greedy(&prompt, 4)
            .expect("recomputing generation");
        let mut session =
            GenerationSession::resident(&model, model.config.model().max_sequence_length(), 0)
                .expect("session");
        session.prefill(&prompt).expect("prefill");

        let actual = (0..4)
            .map(|_| session.decode_greedy().expect("decode"))
            .collect::<Vec<_>>();

        assert_eq!(actual, expected);
        assert_eq!(session.cache().len(), prompt.len() + actual.len());
        assert_eq!(&session.sequence()[prompt.len()..], actual);
    }

    #[test]
    fn cached_temperature_decode_matches_recomputing_generation() {
        let model = fixture_model();
        let prompt = test_fixture::token_ids();
        let expected = model
            .generate_temperature(&prompt, 4, 0.8, 42)
            .expect("recomputing generation");
        let mut session =
            GenerationSession::resident(&model, model.config.model().max_sequence_length(), 42)
                .expect("session");
        session.prefill(&prompt).expect("prefill");

        let actual = (0..4)
            .map(|_| session.decode_temperature(0.8).expect("decode"))
            .collect::<Vec<_>>();

        assert_eq!(actual, expected);
    }

    #[test]
    fn decode_requires_prefill_and_preserves_full_session() {
        let model = fixture_model();
        let prompt = test_fixture::token_ids();
        let mut empty = GenerationSession::resident(&model, prompt.len(), 0).expect("session");
        assert!(matches!(
            empty.decode_greedy(),
            Err(GenerationError::Runtime(RuntimeError::InvalidShape { .. }))
        ));
        assert!(empty.sequence().is_empty());
        assert!(empty.cache().is_empty());

        let mut full = GenerationSession::resident(&model, prompt.len(), 0).expect("session");
        full.prefill(&prompt).expect("prefill");
        let sequence = full.sequence().to_vec();
        let cache = full.cache().clone();
        assert!(matches!(
            full.decode_temperature(0.8),
            Err(GenerationError::Runtime(
                RuntimeError::ContextLengthExceeded { .. }
            ))
        ));
        assert_eq!(full.sequence(), sequence);
        assert_eq!(full.cache(), &cache);
    }
}

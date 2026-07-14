use clr_core::{RuntimeError, Tensor, TensorShape, TensorView, ops::elementwise_add};

use crate::Qwen3MoeConfig;
use crate::cache::LayerKvView;

/// Unvalidated tensors required by one sparse Qwen3-MoE decoder block.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3MoeBlockWeightsSpec {
    /// Input RMS normalization weight `[hidden]`.
    pub input_norm: Tensor,
    /// Query projection weight `[query_heads * head_dim, hidden]`.
    pub query_projection: Tensor,
    /// Key projection weight `[kv_heads * head_dim, hidden]`.
    pub key_projection: Tensor,
    /// Value projection weight `[kv_heads * head_dim, hidden]`.
    pub value_projection: Tensor,
    /// Attention output projection weight `[hidden, hidden]`.
    pub output_projection: Tensor,
    /// Per-head query RMS normalization weight `[head_dim]`.
    pub query_norm: Tensor,
    /// Per-head key RMS normalization weight `[head_dim]`.
    pub key_norm: Tensor,
    /// Post-attention RMS normalization weight `[hidden]`.
    pub post_attention_norm: Tensor,
    /// Router projection weight `[experts, hidden]`.
    pub router: Tensor,
    /// Packed expert gate/up weights `[experts, 2 * moe_intermediate, hidden]`.
    pub expert_gate_up: Tensor,
    /// Expert down weights `[experts, hidden, moe_intermediate]`.
    pub expert_down: Tensor,
}

/// Numerical checkpoints produced by one sparse decoder block.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3MoeBlockOutput {
    /// Output of the input RMS normalization.
    pub input_norm: Tensor,
    /// Output of causal grouped-query attention before the residual add.
    pub attention_output: Tensor,
    /// Output of the post-attention RMS normalization.
    pub post_attention_norm: Tensor,
    /// Router logits `[tokens, experts]`.
    pub router_logits: Tensor,
    /// Selected routing probabilities `[tokens, experts_per_token]`.
    pub routing_weights: Tensor,
    /// Exact selected expert IDs in token-major order.
    pub selected_experts: Vec<usize>,
    /// Routed expert output before the residual add.
    pub moe_output: Tensor,
    /// Final decoder block hidden state.
    pub block_output: Tensor,
}

/// Correctness-first implementation of one sparse Qwen3-MoE decoder block.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3MoeBlock {
    pub(crate) config: Qwen3MoeConfig,
    pub(crate) weights: Qwen3MoeBlockWeightsSpec,
}

impl Qwen3MoeBlock {
    /// Validates weights and creates one sparse decoder block.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::ShapeMismatch`] when any weight shape differs
    /// from the shape derived from `config`.
    pub fn new(
        config: Qwen3MoeConfig,
        weights: Qwen3MoeBlockWeightsSpec,
    ) -> Result<Self, RuntimeError> {
        validate_weight_shapes(config, &weights)?;
        Ok(Self { config, weights })
    }

    /// Executes a batch-one sparse decoder block for contiguous token states.
    ///
    /// `hidden_states` must have shape `[sequence, hidden]`. Positions start at
    /// zero and attention is causal with no padding.
    ///
    /// # Errors
    ///
    /// Returns a structured rank/shape/non-finite error when an input or
    /// intermediate violates the correctness-path contract.
    pub fn forward(
        &self,
        hidden_states: TensorView<'_>,
    ) -> Result<Qwen3MoeBlockOutput, RuntimeError> {
        require_rank(hidden_states, 2, "Qwen3-MoE block input")?;
        let hidden_size = self.config.model().hidden_size();
        if hidden_states.shape().dimensions()[1] != hidden_size {
            return Err(RuntimeError::ShapeMismatch {
                operation: "Qwen3-MoE block input",
                expected: [hidden_states.shape().dimensions()[0], hidden_size].into(),
                actual: hidden_states.shape().dimensions().into(),
            });
        }

        let input_norm = rms_norm(
            hidden_states,
            self.weights.input_norm.view(),
            self.config.rms_norm_epsilon(),
        )?;
        let attention_output = attention(input_norm.view(), self.config, &self.weights)?;
        let after_attention = elementwise_add(hidden_states, attention_output.view())?;
        let post_attention_norm = rms_norm(
            after_attention.view(),
            self.weights.post_attention_norm.view(),
            self.config.rms_norm_epsilon(),
        )?;
        let router = route_tokens(
            post_attention_norm.view(),
            self.weights.router.view(),
            self.config,
        )?;
        let moe_output = routed_experts(
            post_attention_norm.view(),
            self.weights.expert_gate_up.view(),
            self.weights.expert_down.view(),
            &router,
            self.config,
        )?;
        let block_output = elementwise_add(after_attention.view(), moe_output.view())?;

        Ok(Qwen3MoeBlockOutput {
            input_norm,
            attention_output,
            post_attention_norm,
            router_logits: router.logits,
            routing_weights: router.weights,
            selected_experts: router.selected_experts,
            moe_output,
            block_output,
        })
    }
}

#[derive(Debug)]
pub(crate) struct RouterOutput {
    pub(crate) logits: Tensor,
    pub(crate) weights: Tensor,
    pub(crate) selected_experts: Vec<usize>,
}

pub(crate) fn rms_norm(
    input: TensorView<'_>,
    weight: TensorView<'_>,
    epsilon: f32,
) -> Result<Tensor, RuntimeError> {
    require_rank(weight, 1, "RMSNorm weight")?;
    let width = weight.shape().dimensions()[0];
    if input.shape().dimensions().last().copied() != Some(width) {
        return Err(RuntimeError::ShapeMismatch {
            operation: "RMSNorm",
            expected: [width].into(),
            actual: input
                .shape()
                .dimensions()
                .last()
                .copied()
                .into_iter()
                .collect(),
        });
    }
    require_finite(input, "RMSNorm")?;

    let mut output = Vec::with_capacity(input.data().len());
    for row in input.data().chunks_exact(width) {
        let square_sum: f32 = row.iter().map(|value| value * value).sum();
        #[allow(clippy::cast_precision_loss)]
        let variance = square_sum / width as f32;
        let inverse_rms = (variance + epsilon).sqrt().recip();
        output.extend(
            row.iter()
                .zip(weight.data())
                .map(|(value, scale)| value * inverse_rms * scale),
        );
    }
    Tensor::new(input.shape().clone(), output)
}

pub(crate) fn attention(
    hidden_states: TensorView<'_>,
    config: Qwen3MoeConfig,
    weights: &Qwen3MoeBlockWeightsSpec,
) -> Result<Tensor, RuntimeError> {
    attention_with_weights(
        hidden_states,
        config,
        weights.query_projection.view(),
        weights.key_projection.view(),
        weights.value_projection.view(),
        weights.output_projection.view(),
        weights.query_norm.view(),
        weights.key_norm.view(),
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_with_weights(
    hidden_states: TensorView<'_>,
    config: Qwen3MoeConfig,
    query_weight: TensorView<'_>,
    key_weight: TensorView<'_>,
    value_weight: TensorView<'_>,
    output_weight: TensorView<'_>,
    query_norm_weight: TensorView<'_>,
    key_norm_weight: TensorView<'_>,
) -> Result<Tensor, RuntimeError> {
    let sequence_length = hidden_states.shape().dimensions()[0];
    let query_head_count = config.model().attention_head_count();
    let key_value_head_count = config.model().key_value_head_count();
    let head_dimension = config.head_dimension();

    let query = linear(hidden_states, query_weight, "query projection")?;
    let key = linear(hidden_states, key_weight, "key projection")?;
    let value = linear(hidden_states, value_weight, "value projection")?;
    let query = Tensor::new(
        TensorShape::new([sequence_length, query_head_count, head_dimension]),
        query.into_data(),
    )?;
    let key = Tensor::new(
        TensorShape::new([sequence_length, key_value_head_count, head_dimension]),
        key.into_data(),
    )?;
    let value = Tensor::new(
        TensorShape::new([sequence_length, key_value_head_count, head_dimension]),
        value.into_data(),
    )?;
    let query = rms_norm(query.view(), query_norm_weight, config.rms_norm_epsilon())?;
    let key = rms_norm(key.view(), key_norm_weight, config.rms_norm_epsilon())?;
    let (query, key) = apply_rotary_embeddings(query.view(), key.view(), config)?;

    let mut attended = vec![0.0; sequence_length * query_head_count * head_dimension];
    // Attention scaling is defined in the model's F32 compute precision.
    #[allow(clippy::cast_precision_loss)]
    let scale = (head_dimension as f32).sqrt().recip();
    let group_count = config.key_value_group_count();
    for query_position in 0..sequence_length {
        for query_head in 0..query_head_count {
            let key_value_head = query_head / group_count;
            let query_offset = (query_position * query_head_count + query_head) * head_dimension;
            let query_values = &query.data()[query_offset..query_offset + head_dimension];
            let mut scores = Vec::with_capacity(query_position + 1);
            for key_position in 0..=query_position {
                let key_offset =
                    (key_position * key_value_head_count + key_value_head) * head_dimension;
                let key_values = &key.data()[key_offset..key_offset + head_dimension];
                let score: f32 = query_values
                    .iter()
                    .zip(key_values)
                    .map(|(query_value, key_value)| query_value * key_value)
                    .sum();
                scores.push(score * scale);
            }
            softmax_slice(&mut scores);

            let output_offset = (query_position * query_head_count + query_head) * head_dimension;
            for (key_position, attention_weight) in scores.iter().enumerate() {
                let value_offset =
                    (key_position * key_value_head_count + key_value_head) * head_dimension;
                for dimension in 0..head_dimension {
                    attended[output_offset + dimension] +=
                        attention_weight * value.data()[value_offset + dimension];
                }
            }
        }
    }
    let attended = Tensor::new(
        TensorShape::new([sequence_length, config.model().query_projection_width()]),
        attended,
    )?;
    linear(
        attended.view(),
        output_weight,
        "attention output projection",
    )
}

pub(crate) struct CachedAttentionOutput {
    pub output: Tensor,
    pub key: Vec<f32>,
    pub value: Vec<f32>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn cached_attention_with_weights(
    hidden_states: TensorView<'_>,
    config: Qwen3MoeConfig,
    query_weight: TensorView<'_>,
    key_weight: TensorView<'_>,
    value_weight: TensorView<'_>,
    output_weight: TensorView<'_>,
    query_norm_weight: TensorView<'_>,
    key_norm_weight: TensorView<'_>,
    cache: LayerKvView<'_>,
) -> Result<CachedAttentionOutput, RuntimeError> {
    let query_head_count = config.model().attention_head_count();
    let key_value_head_count = config.model().key_value_head_count();
    let head_dimension = config.head_dimension();
    let values_per_token = key_value_head_count.checked_mul(head_dimension).ok_or(
        RuntimeError::ArithmeticOverflow {
            operation: "cached attention values per token",
        },
    )?;
    let initialized =
        cache
            .len
            .checked_mul(values_per_token)
            .ok_or(RuntimeError::ArithmeticOverflow {
                operation: "cached attention initialized values",
            })?;
    if hidden_states.shape().dimensions() != [1, config.model().hidden_size()]
        || cache.key.len() != initialized
        || cache.value.len() != initialized
    {
        return Err(RuntimeError::InvalidShape {
            reason: "cached attention input or cache shape mismatch",
        });
    }

    let query = linear(hidden_states, query_weight, "cached query projection")?;
    let key = linear(hidden_states, key_weight, "cached key projection")?;
    let value = linear(hidden_states, value_weight, "cached value projection")?;
    let query = Tensor::new(
        TensorShape::new([1, query_head_count, head_dimension]),
        query.into_data(),
    )?;
    let key = Tensor::new(
        TensorShape::new([1, key_value_head_count, head_dimension]),
        key.into_data(),
    )?;
    let value = Tensor::new(
        TensorShape::new([1, key_value_head_count, head_dimension]),
        value.into_data(),
    )?;
    let query = rms_norm(query.view(), query_norm_weight, config.rms_norm_epsilon())?;
    let key = rms_norm(key.view(), key_norm_weight, config.rms_norm_epsilon())?;
    let (query, key) = apply_rotary_embeddings_at(query.view(), key.view(), config, cache.len)?;

    let mut attended = vec![0.0; query_head_count * head_dimension];
    #[allow(clippy::cast_precision_loss)]
    let scale = (head_dimension as f32).sqrt().recip();
    let group_count = config.key_value_group_count();
    for query_head in 0..query_head_count {
        let key_value_head = query_head / group_count;
        let query_offset = query_head * head_dimension;
        let query_values = &query.data()[query_offset..query_offset + head_dimension];
        let mut scores = Vec::with_capacity(cache.len + 1);
        for key_position in 0..=cache.len {
            let key_offset =
                (key_position * key_value_head_count + key_value_head) * head_dimension;
            let key_values = if key_position == cache.len {
                &key.data()[key_value_head * head_dimension..(key_value_head + 1) * head_dimension]
            } else {
                &cache.key[key_offset..key_offset + head_dimension]
            };
            let score: f32 = query_values
                .iter()
                .zip(key_values)
                .map(|(query_value, key_value)| query_value * key_value)
                .sum();
            scores.push(score * scale);
        }
        softmax_slice(&mut scores);

        for (key_position, attention_weight) in scores.iter().enumerate() {
            let value_offset =
                (key_position * key_value_head_count + key_value_head) * head_dimension;
            let values = if key_position == cache.len {
                &value.data()
                    [key_value_head * head_dimension..(key_value_head + 1) * head_dimension]
            } else {
                &cache.value[value_offset..value_offset + head_dimension]
            };
            for dimension in 0..head_dimension {
                attended[query_offset + dimension] += attention_weight * values[dimension];
            }
        }
    }
    let attended = Tensor::new(
        TensorShape::new([1, config.model().query_projection_width()]),
        attended,
    )?;
    let output = linear(
        attended.view(),
        output_weight,
        "cached attention output projection",
    )?;
    Ok(CachedAttentionOutput {
        output,
        key: key.into_data(),
        value: value.into_data(),
    })
}

fn apply_rotary_embeddings(
    query: TensorView<'_>,
    key: TensorView<'_>,
    config: Qwen3MoeConfig,
) -> Result<(Tensor, Tensor), RuntimeError> {
    apply_rotary_embeddings_at(query, key, config, 0)
}

pub(crate) fn apply_rotary_embeddings_at(
    query: TensorView<'_>,
    key: TensorView<'_>,
    config: Qwen3MoeConfig,
    position_offset: usize,
) -> Result<(Tensor, Tensor), RuntimeError> {
    require_rank(query, 3, "RoPE query")?;
    require_rank(key, 3, "RoPE key")?;
    let sequence_length = query.shape().dimensions()[0];
    let head_dimension = config.head_dimension();
    let query_heads = query.shape().dimensions()[1];
    let key_heads = key.shape().dimensions()[1];
    if query.shape().dimensions()[2] != head_dimension
        || key.shape().dimensions()[0] != sequence_length
        || key.shape().dimensions()[2] != head_dimension
    {
        return Err(RuntimeError::InvalidShape {
            reason: "RoPE input shapes do not match configuration",
        });
    }

    let mut rotated_query = query.data().to_vec();
    let mut rotated_key = key.data().to_vec();
    for position in 0..sequence_length {
        let absolute_position =
            position_offset
                .checked_add(position)
                .ok_or(RuntimeError::ArithmeticOverflow {
                    operation: "RoPE absolute position",
                })?;
        for pair_index in 0..head_dimension / 2 {
            #[allow(clippy::cast_precision_loss)]
            let exponent = (pair_index * 2) as f32 / head_dimension as f32;
            #[allow(clippy::cast_precision_loss)]
            let frequency = absolute_position as f32 / config.rope_theta().powf(exponent);
            let cosine = frequency.cos();
            let sine = frequency.sin();
            rotate_pair_group(
                &mut rotated_query,
                query.data(),
                position,
                query_heads,
                head_dimension,
                pair_index,
                cosine,
                sine,
            );
            rotate_pair_group(
                &mut rotated_key,
                key.data(),
                position,
                key_heads,
                head_dimension,
                pair_index,
                cosine,
                sine,
            );
        }
    }
    Ok((
        Tensor::new(query.shape().clone(), rotated_query)?,
        Tensor::new(key.shape().clone(), rotated_key)?,
    ))
}

#[allow(clippy::too_many_arguments)]
fn rotate_pair_group(
    output: &mut [f32],
    input: &[f32],
    position: usize,
    head_count: usize,
    head_dimension: usize,
    pair_index: usize,
    cosine: f32,
    sine: f32,
) {
    for head in 0..head_count {
        let base = (position * head_count + head) * head_dimension;
        let first = base + pair_index;
        let second = first + head_dimension / 2;
        output[first] = input[first] * cosine - input[second] * sine;
        output[second] = input[second] * cosine + input[first] * sine;
    }
}

pub(crate) fn route_tokens(
    hidden_states: TensorView<'_>,
    router_weight: TensorView<'_>,
    config: Qwen3MoeConfig,
) -> Result<RouterOutput, RuntimeError> {
    let logits = linear(hidden_states, router_weight, "router projection")?;
    require_finite(logits.view(), "router")?;
    let token_count = hidden_states.shape().dimensions()[0];
    let expert_count = config.expert_count();
    let top_k = config.experts_per_token();
    let mut selected_experts = Vec::with_capacity(token_count * top_k);
    let mut selected_weights = Vec::with_capacity(token_count * top_k);

    for token_logits in logits.data().chunks_exact(expert_count) {
        let mut probabilities = token_logits.to_vec();
        softmax_slice(&mut probabilities);
        let mut ranked: Vec<(usize, f32)> = probabilities.into_iter().enumerate().collect();
        ranked.sort_by(|(left_id, left_score), (right_id, right_score)| {
            right_score
                .total_cmp(left_score)
                .then_with(|| left_id.cmp(right_id))
        });
        let selected = &ranked[..top_k];
        let normalizer: f32 = if config.normalize_topk_probabilities() {
            selected.iter().map(|(_, score)| score).sum()
        } else {
            1.0
        };
        for (expert_id, score) in selected {
            selected_experts.push(*expert_id);
            selected_weights.push(score / normalizer);
        }
    }

    Ok(RouterOutput {
        logits,
        weights: Tensor::new(TensorShape::new([token_count, top_k]), selected_weights)?,
        selected_experts,
    })
}

pub(crate) fn routed_experts(
    hidden_states: TensorView<'_>,
    gate_up_weight: TensorView<'_>,
    down_weight: TensorView<'_>,
    router: &RouterOutput,
    config: Qwen3MoeConfig,
) -> Result<Tensor, RuntimeError> {
    let hidden_size = config.model().hidden_size();
    let intermediate_size = config.moe_intermediate_size();
    combine_routed_experts(hidden_states, router, config, |expert_id, occurrences| {
        let gate_up_base = expert_id * 2 * intermediate_size * hidden_size;
        let down_base = expert_id * hidden_size * intermediate_size;
        let gate =
            &gate_up_weight.data()[gate_up_base..gate_up_base + intermediate_size * hidden_size];
        let up = &gate_up_weight.data()[gate_up_base + intermediate_size * hidden_size
            ..gate_up_base + 2 * intermediate_size * hidden_size];
        let down = &down_weight.data()[down_base..down_base + hidden_size * intermediate_size];
        Ok(occurrences
            .iter()
            .map(|(token_index, _)| {
                let input = &hidden_states.data()
                    [token_index * hidden_size..(token_index + 1) * hidden_size];
                expert_mlp(input, gate, up, down, hidden_size, intermediate_size)
            })
            .collect())
    })
}

pub(crate) fn combine_routed_experts<E, F>(
    hidden_states: TensorView<'_>,
    router: &RouterOutput,
    config: Qwen3MoeConfig,
    mut compute: F,
) -> Result<Tensor, E>
where
    E: From<RuntimeError>,
    F: FnMut(usize, &[(usize, usize)]) -> Result<Vec<Vec<f32>>, E>,
{
    let token_count = hidden_states.shape().dimensions()[0];
    let hidden_size = config.model().hidden_size();
    let top_k = config.experts_per_token();
    let mut output = vec![0.0; token_count * hidden_size];
    for expert_id in 0..config.expert_count() {
        let mut occurrences = Vec::new();
        for token_index in 0..token_count {
            for position in 0..top_k {
                if router.selected_experts[token_index * top_k + position] == expert_id {
                    occurrences.push((token_index, position));
                }
            }
        }
        if occurrences.is_empty() {
            continue;
        }
        let expert_outputs = compute(expert_id, &occurrences)?;
        if expert_outputs.len() != occurrences.len()
            || expert_outputs
                .iter()
                .any(|values| values.len() != hidden_size)
        {
            return Err(RuntimeError::InvalidShape {
                reason: "expert callback output shape mismatch",
            }
            .into());
        }
        for ((token_index, position), values) in occurrences.iter().zip(expert_outputs) {
            let weight = router.weights.data()[token_index * top_k + position];
            for (hidden_index, value) in values.into_iter().enumerate() {
                output[token_index * hidden_size + hidden_index] += value * weight;
            }
        }
    }
    Tensor::new(TensorShape::new([token_count, hidden_size]), output).map_err(Into::into)
}

pub(crate) fn expert_mlp(
    input: &[f32],
    gate: &[f32],
    up: &[f32],
    down: &[f32],
    hidden_size: usize,
    intermediate_size: usize,
) -> Vec<f32> {
    let mut activated = vec![0.0; intermediate_size];
    for (intermediate_index, activated_value) in activated.iter_mut().enumerate() {
        let start = intermediate_index * hidden_size;
        let gate_value = dot(input, &gate[start..start + hidden_size]);
        let up_value = dot(input, &up[start..start + hidden_size]);
        *activated_value = gate_value / (1.0 + (-gate_value).exp()) * up_value;
    }
    (0..hidden_size)
        .map(|hidden_index| {
            let start = hidden_index * intermediate_size;
            dot(&activated, &down[start..start + intermediate_size])
        })
        .collect()
}

pub(crate) fn linear(
    input: TensorView<'_>,
    weight: TensorView<'_>,
    operation: &'static str,
) -> Result<Tensor, RuntimeError> {
    require_rank(input, 2, operation)?;
    require_rank(weight, 2, operation)?;
    let row_count = input.shape().dimensions()[0];
    let input_width = input.shape().dimensions()[1];
    let output_width = weight.shape().dimensions()[0];
    if weight.shape().dimensions()[1] != input_width {
        return Err(RuntimeError::ShapeMismatch {
            operation,
            expected: [output_width, input_width].into(),
            actual: weight.shape().dimensions().into(),
        });
    }

    let mut output = vec![0.0; row_count * output_width];
    for row_index in 0..row_count {
        let input_row = &input.data()[row_index * input_width..(row_index + 1) * input_width];
        for output_index in 0..output_width {
            let weight_row =
                &weight.data()[output_index * input_width..(output_index + 1) * input_width];
            output[row_index * output_width + output_index] = dot(input_row, weight_row);
        }
    }
    Tensor::new(TensorShape::new([row_count, output_width]), output)
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left_value, right_value)| left_value * right_value)
        .sum()
}

fn softmax_slice(values: &mut [f32]) {
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut denominator = 0.0;
    for value in &mut *values {
        *value = (*value - maximum).exp();
        denominator += *value;
    }
    for value in values {
        *value /= denominator;
    }
}

fn require_rank(
    input: TensorView<'_>,
    expected: usize,
    context: &'static str,
) -> Result<(), RuntimeError> {
    let actual = input.shape().rank();
    if actual != expected {
        return Err(RuntimeError::RankMismatch {
            context,
            expected,
            actual,
        });
    }
    Ok(())
}

fn require_finite(input: TensorView<'_>, operation: &'static str) -> Result<(), RuntimeError> {
    if let Some(index) = input.data().iter().position(|value| !value.is_finite()) {
        return Err(RuntimeError::NonFiniteInput { operation, index });
    }
    Ok(())
}

fn require_shape(
    tensor: &Tensor,
    expected: &[usize],
    operation: &'static str,
) -> Result<(), RuntimeError> {
    if tensor.shape().dimensions() != expected {
        return Err(RuntimeError::ShapeMismatch {
            operation,
            expected: expected.into(),
            actual: tensor.shape().dimensions().into(),
        });
    }
    Ok(())
}

fn validate_weight_shapes(
    config: Qwen3MoeConfig,
    weights: &Qwen3MoeBlockWeightsSpec,
) -> Result<(), RuntimeError> {
    let hidden = config.model().hidden_size();
    let query_width = config.model().query_projection_width();
    let key_value_width = config.model().key_value_projection_width();
    let experts = config.expert_count();
    let intermediate = config.moe_intermediate_size();
    require_shape(&weights.input_norm, &[hidden], "input norm weight")?;
    require_shape(
        &weights.query_projection,
        &[query_width, hidden],
        "query projection weight",
    )?;
    require_shape(
        &weights.key_projection,
        &[key_value_width, hidden],
        "key projection weight",
    )?;
    require_shape(
        &weights.value_projection,
        &[key_value_width, hidden],
        "value projection weight",
    )?;
    require_shape(
        &weights.output_projection,
        &[hidden, query_width],
        "attention output projection weight",
    )?;
    require_shape(
        &weights.query_norm,
        &[config.head_dimension()],
        "query norm weight",
    )?;
    require_shape(
        &weights.key_norm,
        &[config.head_dimension()],
        "key norm weight",
    )?;
    require_shape(
        &weights.post_attention_norm,
        &[hidden],
        "post-attention norm weight",
    )?;
    require_shape(&weights.router, &[experts, hidden], "router weight")?;
    require_shape(
        &weights.expert_gate_up,
        &[experts, 2 * intermediate, hidden],
        "expert gate/up weight",
    )?;
    require_shape(
        &weights.expert_down,
        &[experts, hidden, intermediate],
        "expert down weight",
    )?;
    Ok(())
}

#[cfg(test)]
mod tests;

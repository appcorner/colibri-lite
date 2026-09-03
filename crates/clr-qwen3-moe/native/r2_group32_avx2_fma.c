#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

static float horizontal_sum_256(__m256 value) {
    __m128 low = _mm256_castps256_ps128(value);
    __m128 high = _mm256_extractf128_ps(value, 1);
    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

int clr_qwen3_moe_group32_avx2_fma(
    const int8_t *values,
    const float *scales,
    const float *input,
    float *output,
    size_t rows,
    size_t columns) {
    if (values == NULL || scales == NULL || input == NULL || output == NULL ||
        rows == 0 || columns == 0 || (columns % 32) != 0) {
        return 1;
    }

    const size_t groups = columns / 32;
    for (size_t row = 0; row < rows; ++row) {
        __m256 accumulator = _mm256_setzero_ps();
        const int8_t *row_values = values + row * columns;
        const float *row_scales = scales + row * groups;

        for (size_t group = 0; group < groups; ++group) {
            const __m256 scale = _mm256_set1_ps(row_scales[group]);
            const size_t base = group * 32;

            for (size_t offset = 0; offset < 32; offset += 8) {
                const __m128i packed = _mm_loadl_epi64(
                    (const __m128i *)(row_values + base + offset));
                const __m256i widened = _mm256_cvtepi8_epi32(packed);
                const __m256 weight = _mm256_cvtepi32_ps(widened);
                const __m256 dequantized = _mm256_mul_ps(weight, scale);
                const __m256 input_values = _mm256_loadu_ps(input + base + offset);
                accumulator = _mm256_fmadd_ps(input_values, dequantized, accumulator);
            }
        }

        output[row] = horizontal_sum_256(accumulator);
    }

    return 0;
}

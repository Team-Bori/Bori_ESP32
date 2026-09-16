#include "mlp.h"

#include <limits.h>
#include <math.h>

static int32_t make_q31_multiplier(float real_multiplier, int *shift)
{
    if (real_multiplier <= 0.0f) {
        *shift = 0;
        return 0;
    }

    int exp = 0;
    float normalized = frexpf(real_multiplier, &exp); // 0.5 <= n < 1.0
    int64_t q31 = (int64_t)llroundf(normalized * 2147483648.0f);

    if (q31 >= 2147483648LL) {
        q31 >>= 1;
        exp += 1;
    }

    *shift = 31 - exp;
    if (*shift < 0) *shift = 0;
    return (int32_t)q31;
}

static int32_t requantize(int32_t acc, int32_t multiplier, int shift)
{
    if (multiplier == 0) {
        return 0;
    }

    int64_t product = (int64_t)acc * (int64_t)multiplier;
    if (shift > 0) {
        const int64_t rounding = 1LL << (shift - 1);
        product = (product >= 0) ? product + rounding : product - rounding;
        product >>= shift;
    }

    if (product > INT32_MAX) return INT32_MAX;
    if (product < INT32_MIN) return INT32_MIN;
    return (int32_t)product;
}

int mlp_predict(const mlp_model_t *model, const int8_t input[MLP_INPUT_SIZE], int32_t logits[MLP_OUTPUT_SIZE])
{
    int8_t hidden_q[MLP_HIDDEN_SIZE];

    const float hidden_requant_scale = (model->header.input_scale * model->header.w1_scale) / model->header.hidden_scale;

    int hidden_shift = 0;
    const int32_t hidden_multiplier = make_q31_multiplier(hidden_requant_scale, &hidden_shift);

    // Layer 1: 64 -> 16, INT8 x INT8 -> INT32
    for (int h = 0; h < MLP_HIDDEN_SIZE; ++h) {
        int32_t acc = model->b1[h];

        for (int i = 0; i < MLP_INPUT_SIZE; ++i) {
            acc += (int32_t)input[i] * (int32_t)model->w1[i * MLP_HIDDEN_SIZE + h];
        }

        int32_t q = requantize(acc, hidden_multiplier, hidden_shift);

        // ReLU + int8 activation range
        if (q <= 0) hidden_q[h] = 0;
        else if (q > 127) hidden_q[h] = 127;
        else hidden_q[h] = (int8_t)q;
    }

    // Layer 2: 16 -> 10
    int best = 0;
    int32_t best_value = INT32_MIN;

    for (int o = 0; o < MLP_OUTPUT_SIZE; ++o) {
        int32_t acc = model->b2[o];

        for (int h = 0; h < MLP_HIDDEN_SIZE; ++h) {
            acc += (int32_t)hidden_q[h] *
                   (int32_t)model->w2[h * MLP_OUTPUT_SIZE + o];
        }

        logits[o] = acc;

        if (acc > best_value) {
            best_value = acc;
            best = o;
        }
    }

    return best;
}

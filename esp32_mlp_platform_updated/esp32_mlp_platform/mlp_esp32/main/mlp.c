#include "mlp.h"

#include <limits.h>

int mlp_predict(const mlp_model_t *model, const int8_t input[MLP_INPUT_SIZE],
                int32_t logits[MLP_OUTPUT_SIZE])
{
    int8_t hidden_q[MLP_HIDDEN_SIZE];

    // Layer 1: 64 -> 16
    for (int h = 0; h < MLP_HIDDEN_SIZE; ++h) {
        int32_t acc = model->b1[h];
        for (int i = 0; i < MLP_INPUT_SIZE; ++i) {
            acc += (int32_t)input[i] *
                   (int32_t)model->w1[i * MLP_HIDDEN_SIZE + h];
        }

        // Requantize + ReLU. Float is used only for scale conversion.
        float real_value = (float)acc *
                           (model->header.input_scale * model->header.w1_scale);
        if (real_value <= 0.0f) {
            hidden_q[h] = 0;
        } else {
            int32_t q = (int32_t)(real_value / model->header.hidden_scale + 0.5f);
            if (q > 127) q = 127;
            hidden_q[h] = (int8_t)q;
        }
    }

    // Layer 2: 16 -> 10. Argmax is valid because all outputs share the same scale.
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

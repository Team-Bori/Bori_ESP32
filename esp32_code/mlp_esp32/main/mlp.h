#pragma once

#include <stdint.h>
#include "model.h"

int mlp_predict(const mlp_model_t *model, const int8_t input[MLP_INPUT_SIZE],
                int32_t logits[MLP_OUTPUT_SIZE]);

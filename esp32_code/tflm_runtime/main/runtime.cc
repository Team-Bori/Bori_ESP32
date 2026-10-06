#include "runtime.h"

#include <math.h>
#include <stdarg.h>
#include <new>
#include <stdio.h>
#include <string.h>

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_partition.h"
#include "esp_timer.h"

#include "flatbuffers/flatbuffers.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/schema/schema_utils.h"

#include "bori_common.h"

static const char *TAG = "TFLM_RT";

/* Arena caps: internal RAM (no PSRAM on this board), byte addressable. */
#define ARENA_CAPS (MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)
#define ARENA_ALIGN 16
/* Left out of an "auto" arena so small allocations (stdio, logging) still succeed. */
#define AUTO_ARENA_SLACK 1024

RuntimeState g_rt;

namespace {

#define BORI_OP(method, name) +1
constexpr unsigned kOpCount = 0
#include "ops.def"
    ;
#undef BORI_OP

using Resolver = tflite::MicroMutableOpResolver<kOpCount>;

Resolver s_resolver;
bool s_resolver_ready = false;

alignas(tflite::MicroInterpreter) uint8_t s_interp_storage[sizeof(tflite::MicroInterpreter)];
tflite::MicroInterpreter *s_interp = nullptr;
uint8_t *s_arena = nullptr;

esp_partition_mmap_handle_t s_mmap_handle;

void set_error(const char *code, const char *fmt, ...) __attribute__((format(printf, 2, 3)));

void set_error(const char *code, const char *fmt, ...)
{
    g_rt.loaded = false;
    g_rt.error_code = code;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_rt.error_message, sizeof(g_rt.error_message), fmt, ap);
    va_end(ap);
}

bool register_ops(void)
{
    if (s_resolver_ready) {
        return true;
    }
#define BORI_OP(method, name)                                         \
    if (s_resolver.Add##method() != kTfLiteOk) {                      \
        ESP_LOGE(TAG, "op registration failed: %s", #name);           \
        return false;                                                 \
    }
#include "ops.def"
#undef BORI_OP
    s_resolver_ready = true;
    return true;
}

uint8_t btf_dtype_from_tflite(TfLiteType t)
{
    switch (t) {
    case kTfLiteInt8:    return BTF_DTYPE_INT8;
    case kTfLiteUInt8:   return BTF_DTYPE_UINT8;
    case kTfLiteInt16:   return BTF_DTYPE_INT16;
    case kTfLiteInt32:   return BTF_DTYPE_INT32;
    case kTfLiteFloat32: return BTF_DTYPE_FLOAT32;
    default:             return 0;
    }
}

bool scale_matches(float a, float b)
{
    if (a == b) {
        return true;
    }
    const float diff = fabsf(a - b);
    return diff <= 1e-6f * fmaxf(fabsf(a), fabsf(b));
}

/* Compares a TFLM tensor with the package descriptor; returns NULL when they match. */
const char *tensor_mismatch(const TfLiteTensor *t, const btf_tensor_desc_t *d)
{
    if (t == nullptr) {
        return "missing";
    }
    if (btf_dtype_from_tflite(t->type) != d->dtype) {
        return "dtype";
    }
    if (t->dims == nullptr || t->dims->size != d->ndim) {
        return "rank";
    }
    for (int i = 0; i < d->ndim; ++i) {
        if (t->dims->data[i] != d->dims[i]) {
            return "shape";
        }
    }
    if (t->bytes != d->bytes) {
        return "bytes";
    }
    if (d->dtype != BTF_DTYPE_FLOAT32) {
        if (!scale_matches(t->params.scale, d->scale) || t->params.zero_point != d->zero_point) {
            return "quantization";
        }
    }
    return nullptr;
}

/* Returns NULL if every op of the model is registered, else the first missing op name. */
const char *find_unsupported_op(const tflite::Model *model, char *buf, size_t len)
{
    const auto *codes = model->operator_codes();
    g_rt.model_ops = codes ? codes->size() : 0;
    if (codes == nullptr) {
        return nullptr;
    }
    for (flatbuffers::uoffset_t i = 0; i < codes->size(); ++i) {
        const tflite::OperatorCode *oc = codes->Get(i);
        const tflite::BuiltinOperator code = tflite::GetBuiltinCode(oc);
        if (code == tflite::BuiltinOperator_CUSTOM) {
            snprintf(buf, len, "CUSTOM(%s)",
                     oc->custom_code() ? oc->custom_code()->c_str() : "?");
            return buf;
        }
        if (s_resolver.FindOp(code) == nullptr) {
            const char *name = tflite::EnumNameBuiltinOperator(code);
            if (name != nullptr && name[0] != '\0') {
                snprintf(buf, len, "%s", name);
            } else {
                snprintf(buf, len, "builtin_%d", (int)code);
            }
            return buf;
        }
    }
    return nullptr;
}

void destroy_interpreter(void)
{
    if (s_interp != nullptr) {
        s_interp->~MicroInterpreter();
        s_interp = nullptr;
    }
    if (s_arena != nullptr) {
        heap_caps_free(s_arena);
        s_arena = nullptr;
    }
    g_rt.arena_alloc_bytes = 0;
}

/* Allocates the arena and the interpreter. Returns kTfLiteOk when tensors are allocated. */
TfLiteStatus create_interpreter(const tflite::Model *model, size_t arena_bytes)
{
    s_arena = static_cast<uint8_t *>(heap_caps_aligned_alloc(ARENA_ALIGN, arena_bytes, ARENA_CAPS));
    if (s_arena == nullptr) {
        return kTfLiteError;
    }
    g_rt.arena_alloc_bytes = arena_bytes;
    s_interp = new (s_interp_storage) tflite::MicroInterpreter(model, s_resolver, s_arena, arena_bytes);
    return s_interp->AllocateTensors();
}

size_t largest_free_block(void)
{
    return heap_caps_get_largest_free_block(ARENA_CAPS);
}

size_t auto_arena_size(void)
{
    const size_t largest = largest_free_block();
    if (largest <= AUTO_ARENA_SLACK + 64) {
        return 0;
    }
    return (largest - AUTO_ARENA_SLACK) & ~(size_t)(ARENA_ALIGN - 1);
}

}  // namespace

int runtime_registered_ops(void)
{
    return (int)kOpCount;
}

void runtime_load(const char *skip_reason)
{
    g_rt.loaded = false;
    g_rt.error_code = nullptr;
    g_rt.error_message[0] = '\0';
    /* Reported as max_arena_bytes even when no package loads (memory budget measurement).
     * Measured again right before the arena is allocated. */
    g_rt.boot_largest_free = largest_free_block();

    const esp_partition_t *part = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA, (esp_partition_subtype_t)BORI_MODEL_PARTITION_SUBTYPE,
        BORI_MODEL_PARTITION_LABEL);
    if (part == nullptr) {
        set_error("package_invalid", "model partition not found (flash the firmware from 0x0)");
        return;
    }
    g_rt.partition_offset = part->address;
    g_rt.partition_size = part->size;

    if (skip_reason != nullptr) {
        set_error("model_crashed", "%s", skip_reason);
        return;
    }

    uint8_t head[BTF_HEADER_MIN_SIZE];
    if (esp_partition_read(part, 0, head, sizeof(head)) != ESP_OK) {
        set_error("package_invalid", "cannot read the model partition");
        return;
    }
    if (head[0] == 0xFF && head[1] == 0xFF && head[2] == 0xFF && head[3] == 0xFF) {
        set_error("no_package", "model partition is empty (write a package at 0x%06x)",
                  (unsigned)part->address);
        return;
    }
    if (memcmp(head, BTF_MAGIC, 4) != 0) {
        set_error("package_invalid", "not a BTF1 package (magic %02x%02x%02x%02x)%s",
                  head[0], head[1], head[2], head[3],
                  memcmp(head, "MLP1", 4) == 0 ? "; MLP1 needs the mlp firmware" : "");
        return;
    }
    uint32_t total = (uint32_t)head[8] | ((uint32_t)head[9] << 8) | ((uint32_t)head[10] << 16) |
                     ((uint32_t)head[11] << 24);
    if (total < BTF_HEADER_MIN_SIZE || total > part->size) {
        set_error("package_invalid", "package size %u does not fit the partition (%u)",
                  (unsigned)total, (unsigned)part->size);
        return;
    }

    const void *mapped = nullptr;
    if (esp_partition_mmap(part, 0, total, ESP_PARTITION_MMAP_DATA, &mapped, &s_mmap_handle) != ESP_OK) {
        set_error("package_invalid", "esp_partition_mmap failed for %u bytes", (unsigned)total);
        return;
    }

    const char *code = nullptr;
    const char *err = btf_parse(static_cast<const uint8_t *>(mapped), total, &g_rt.pkg, &code);
    if (err != nullptr) {
        set_error(code, "%s", err);
        return;
    }
    const btf_package_t &pkg = g_rt.pkg;

    flatbuffers::Verifier verifier(pkg.tflite, pkg.tflite_size);
    if (!tflite::VerifyModelBuffer(verifier)) {
        set_error("package_invalid", "tflite flatbuffer verification failed");
        return;
    }
    const tflite::Model *model = tflite::GetModel(pkg.tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        set_error("package_invalid", "tflite schema version %u (expected %d)",
                  (unsigned)model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }

    if (!register_ops()) {
        set_error("unsupported_op", "op resolver setup failed");
        return;
    }
    char op_name[48];
    const char *missing = find_unsupported_op(model, op_name, sizeof(op_name));
    if (missing != nullptr) {
        set_error("unsupported_op", "op %s is not supported by tflm_runtime (see docs/ADDING_A_MODEL.md)", missing);
        return;
    }

    g_rt.boot_largest_free = largest_free_block();
    const bool auto_arena = pkg.arena_bytes == 0;
    const size_t want = auto_arena ? auto_arena_size() : pkg.arena_bytes;

    TfLiteStatus st = create_interpreter(model, want);
    if (s_arena == nullptr) {
        set_error("arena_alloc_failed", "cannot allocate a %u byte arena; largest free block %u",
                  (unsigned)want, (unsigned)g_rt.boot_largest_free);
        return;
    }
    if (st != kTfLiteOk) {
        destroy_interpreter();
        if (auto_arena) {
            set_error("arena_alloc_failed",
                      "model needs more than the largest free block (%u bytes)",
                      (unsigned)g_rt.boot_largest_free);
            return;
        }
        /* Measure what the model really needs with the biggest arena we can get. */
        size_t required = 0;
        const size_t probe = auto_arena_size();
        if (probe > want && create_interpreter(model, probe) == kTfLiteOk) {
            required = s_interp->arena_used_bytes();
        }
        destroy_interpreter();
        if (required > 0) {
            set_error("arena_too_small", "package arena %u bytes, model needs %u bytes",
                      (unsigned)want, (unsigned)required);
        } else {
            set_error("arena_alloc_failed",
                      "package arena %u bytes is too small and the board cannot provide more "
                      "(largest free block %u)",
                      (unsigned)want, (unsigned)g_rt.boot_largest_free);
        }
        return;
    }
    g_rt.arena_used_bytes = s_interp->arena_used_bytes();

    const size_t n_in = s_interp->inputs_size(), n_out = s_interp->outputs_size();
    if (n_in != 1 || n_out != 1) {
        destroy_interpreter();
        set_error("package_invalid", "model must have exactly 1 input and 1 output (has %u/%u)",
                  (unsigned)n_in, (unsigned)n_out);
        return;
    }
    const char *why = tensor_mismatch(s_interp->input(0), &pkg.input);
    if (why == nullptr) {
        why = tensor_mismatch(s_interp->output(0), &pkg.output);
        if (why != nullptr) {
            destroy_interpreter();
            set_error("package_invalid", "output tensor does not match the package (%s)", why);
            return;
        }
    } else {
        destroy_interpreter();
        set_error("package_invalid", "input tensor does not match the package (%s)", why);
        return;
    }

    g_rt.loaded = true;
    ESP_LOGI(TAG, "package %s loaded: tflite %u bytes, arena %u used / %u allocated%s, %u ops",
             pkg.model_id, (unsigned)pkg.tflite_size, (unsigned)g_rt.arena_used_bytes,
             (unsigned)g_rt.arena_alloc_bytes, auto_arena ? " (auto)" : "",
             (unsigned)g_rt.model_ops);
}

uint8_t *runtime_input(void)
{
    return s_interp ? s_interp->input(0)->data.uint8 : nullptr;
}

size_t runtime_input_bytes(void)
{
    return g_rt.pkg.input.bytes;
}

uint32_t runtime_output_count(void)
{
    return g_rt.pkg.output.bytes / btf_dtype_size(g_rt.pkg.output.dtype);
}

float runtime_output_value(uint32_t index)
{
    const TfLiteTensor *t = s_interp->output(0);
    const float scale = t->params.scale;
    const int32_t zp = t->params.zero_point;
    switch (t->type) {
    case kTfLiteInt8:    return (float)((int32_t)t->data.int8[index] - zp) * scale;
    case kTfLiteUInt8:   return (float)((int32_t)t->data.uint8[index] - zp) * scale;
    case kTfLiteInt16:   return (float)((int32_t)t->data.i16[index] - zp) * scale;
    case kTfLiteInt32:   return (float)(t->data.i32[index] - zp) * scale;
    case kTfLiteFloat32: return t->data.f[index];
    default:             return 0.0f;
    }
}

void runtime_set_demo_input(void)
{
    uint8_t *in = runtime_input();
    const btf_package_t &pkg = g_rt.pkg;
    if (pkg.demo_input != nullptr) {
        memcpy(in, pkg.demo_input, pkg.input.bytes);
        return;
    }
    /* No demo sample: feed the quantized value of 0.0 everywhere. */
    switch (pkg.input.dtype) {
    case BTF_DTYPE_INT8:
        memset(in, (int8_t)pkg.input.zero_point, pkg.input.bytes);
        break;
    case BTF_DTYPE_UINT8:
        memset(in, (uint8_t)pkg.input.zero_point, pkg.input.bytes);
        break;
    default:
        memset(in, 0, pkg.input.bytes);
        break;
    }
}

bool runtime_invoke(int64_t *elapsed_us)
{
    const int64_t start = esp_timer_get_time();
    const TfLiteStatus st = s_interp->Invoke();
    *elapsed_us = esp_timer_get_time() - start;
    if (st != kTfLiteOk) {
        ++g_rt.invoke_errors;
        return false;
    }
    return true;
}

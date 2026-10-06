#include "package.h"

#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "bori_common.h"

static char s_error[96];

static uint16_t rd16(const uint8_t *p)
{
    return (uint16_t)(p[0] | (p[1] << 8));
}

static uint32_t rd32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

float btf_read_f32(const uint8_t *p)
{
    const uint32_t bits = rd32(p);
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

const char *btf_task_name(uint8_t task)
{
    switch (task) {
    case BTF_TASK_CLASSIFICATION: return "classification";
    case BTF_TASK_REGRESSION:     return "regression";
    case BTF_TASK_BINARY_SCORE:   return "binary_score";
    default:                      return "unknown";
    }
}

const char *btf_dtype_name(uint8_t dtype)
{
    switch (dtype) {
    case BTF_DTYPE_INT8:    return "int8";
    case BTF_DTYPE_UINT8:   return "uint8";
    case BTF_DTYPE_INT16:   return "int16";
    case BTF_DTYPE_INT32:   return "int32";
    case BTF_DTYPE_FLOAT32: return "float32";
    default:                return "unknown";
    }
}

uint32_t btf_dtype_size(uint8_t dtype)
{
    switch (dtype) {
    case BTF_DTYPE_INT8:
    case BTF_DTYPE_UINT8:   return 1;
    case BTF_DTYPE_INT16:   return 2;
    case BTF_DTYPE_INT32:
    case BTF_DTYPE_FLOAT32: return 4;
    default:                return 0;
    }
}

const char *btf_label(const btf_package_t *pkg, uint32_t index)
{
    if (pkg->labels == NULL || index >= pkg->label_count) {
        return NULL;
    }
    const char *p = pkg->labels;
    for (uint32_t i = 0; i < index; ++i) {
        p += strlen(p) + 1;
    }
    return p;
}

static const char *fail(const char **code, const char *c, const char *msg)
{
    *code = c;
    return msg;
}

static bool parse_tensor(const uint8_t *p, btf_tensor_desc_t *t)
{
    t->dtype = p[0];
    t->ndim = p[1];
    for (int i = 0; i < BTF_MAX_DIMS; ++i) {
        t->dims[i] = (int32_t)rd32(p + 4 + 4 * i);
    }
    t->scale = btf_read_f32(p + 28);
    t->zero_point = (int32_t)rd32(p + 32);
    t->bytes = rd32(p + 36);

    const uint32_t esize = btf_dtype_size(t->dtype);
    if (esize == 0 || t->ndim == 0 || t->ndim > BTF_MAX_DIMS) {
        return false;
    }
    uint64_t elements = 1;
    for (int i = 0; i < t->ndim; ++i) {
        if (t->dims[i] <= 0) {
            return false;
        }
        elements *= (uint64_t)t->dims[i];
    }
    return elements * esize == t->bytes;
}

static bool valid_model_id(const char *id)
{
    if (id[0] == '\0') {
        return false;
    }
    for (const char *p = id; *p != '\0'; ++p) {
        const bool ok = (*p >= 'a' && *p <= 'z') || (*p >= '0' && *p <= '9') || *p == '_';
        if (!ok) {
            return false;
        }
    }
    return true;
}

const char *btf_parse(const uint8_t *base, uint32_t mapped_size, btf_package_t *pkg, const char **code)
{
    memset(pkg, 0, sizeof(*pkg));
    *code = "package_invalid";

    if (mapped_size < BTF_HEADER_MIN_SIZE || memcmp(base, BTF_MAGIC, 4) != 0) {
        return fail(code, "package_invalid", "bad magic (expected BTF1)");
    }
    const uint16_t version = rd16(base + 4);
    const uint16_t header_size = rd16(base + 6);
    const uint32_t total_size = rd32(base + 8);
    if (version != BTF_FORMAT_VERSION) {
        snprintf(s_error, sizeof(s_error), "unsupported format version %u", version);
        return fail(code, "package_invalid", s_error);
    }
    const uint8_t section_count = base[49];
    const uint32_t table_end = (uint32_t)header_size + (uint32_t)section_count * BTF_SECTION_ENTRY_SIZE;
    if (header_size < BTF_HEADER_MIN_SIZE || total_size < table_end || total_size > mapped_size) {
        snprintf(s_error, sizeof(s_error), "bad sizes: header %u, total %u, partition %u",
                 header_size, (unsigned)total_size, (unsigned)mapped_size);
        return fail(code, "package_invalid", s_error);
    }

    pkg->base = base;
    pkg->total_size = total_size;
    pkg->crc32 = rd32(base + 12);
    const uint32_t actual_crc = bori_crc32(0, base + 16, total_size - 16);
    if (actual_crc != pkg->crc32) {
        snprintf(s_error, sizeof(s_error), "crc32 expected 0x%08x, actual 0x%08x",
                 (unsigned)pkg->crc32, (unsigned)actual_crc);
        return fail(code, "checksum_mismatch", s_error);
    }

    memcpy(pkg->model_id, base + 16, BTF_MODEL_ID_LEN);
    pkg->model_id[BTF_MODEL_ID_LEN] = '\0';
    if (memchr(base + 16, 0, BTF_MODEL_ID_LEN) == NULL || !valid_model_id(pkg->model_id)) {
        return fail(code, "package_invalid", "model_id must be 1-31 chars of a-z, 0-9, _");
    }
    pkg->task = base[48];
    pkg->label_dim = rd16(base + 50);
    pkg->arena_bytes = rd32(base + 52);
    pkg->task_param = btf_read_f32(base + 56);
    if (pkg->task > BTF_TASK_BINARY_SCORE) {
        return fail(code, "package_invalid", "unknown task");
    }
    if (pkg->label_dim == 0 || pkg->label_dim > BTF_MAX_LABEL_DIM) {
        return fail(code, "package_invalid", "label_dim out of range");
    }

    bool have_tensors = false;
    uint32_t seen = 0;
    const uint8_t *eval_inputs = NULL, *eval_expected = NULL;
    uint32_t eval_inputs_size = 0, eval_inputs_count = 0, eval_expected_size = 0, eval_expected_count = 0;
    uint32_t demo_input_size = 0, demo_expected_size = 0;

    for (uint32_t i = 0; i < section_count; ++i) {
        const uint8_t *e = base + header_size + i * BTF_SECTION_ENTRY_SIZE;
        const uint32_t type = rd32(e), offset = rd32(e + 4), size = rd32(e + 8), count = rd32(e + 12);
        if (offset < table_end || offset > total_size || size > total_size - offset) {
            snprintf(s_error, sizeof(s_error), "section %u (type %u) out of bounds", (unsigned)i, (unsigned)type);
            return fail(code, "package_invalid", s_error);
        }
        if (type >= 1 && type <= 31) {
            if (seen & (1u << type)) {
                return fail(code, "package_invalid", "duplicate section");
            }
            seen |= 1u << type;
        }
        const uint8_t *data = base + offset;

        switch (type) {
        case BTF_SEC_TFLITE:
            if (((uintptr_t)data % BTF_TFLITE_ALIGN) != 0 || size < 8 || memcmp(data + 4, "TFL3", 4) != 0) {
                return fail(code, "package_invalid", "tflite section must be 16-byte aligned and start with a TFL3 model");
            }
            pkg->tflite = data;
            pkg->tflite_size = size;
            break;
        case BTF_SEC_TENSORS:
            if (count != 2 || size != 2 * BTF_TENSOR_DESC_SIZE ||
                !parse_tensor(data, &pkg->input) || !parse_tensor(data + BTF_TENSOR_DESC_SIZE, &pkg->output)) {
                return fail(code, "package_invalid", "bad tensor descriptors (need 1 input + 1 output)");
            }
            have_tensors = true;
            break;
        case BTF_SEC_LABELS: {
            uint32_t n = 0, pos = 0;
            while (pos < size && n < count) {
                const uint8_t *nul = memchr(data + pos, 0, size - pos);
                if (nul == NULL) {
                    break;
                }
                pos = (uint32_t)(nul - data) + 1;
                ++n;
            }
            if (n != count) {
                return fail(code, "package_invalid", "labels section is malformed");
            }
            pkg->labels = (const char *)data;
            pkg->labels_size = size;
            pkg->label_count = count;
            break;
        }
        case BTF_SEC_DEMO_INPUT:
            pkg->demo_input = data;
            demo_input_size = size;
            break;
        case BTF_SEC_DEMO_EXPECTED:
            pkg->demo_expected = data;
            demo_expected_size = size;
            break;
        case BTF_SEC_EVAL_INPUTS:
            eval_inputs = data;
            eval_inputs_size = size;
            eval_inputs_count = count;
            break;
        case BTF_SEC_EVAL_EXPECTED:
            eval_expected = data;
            eval_expected_size = size;
            eval_expected_count = count;
            break;
        default:
            break; /* unknown sections are skipped (forward compatible) */
        }
    }

    if (pkg->tflite == NULL || !have_tensors) {
        return fail(code, "package_invalid", "missing tflite or tensors section");
    }

    const uint32_t out_elements = pkg->output.bytes / btf_dtype_size(pkg->output.dtype);
    if (pkg->task == BTF_TASK_REGRESSION) {
        if (pkg->label_dim != out_elements) {
            return fail(code, "package_invalid", "regression label_dim must equal output elements");
        }
    } else {
        if (pkg->label_dim != 1) {
            return fail(code, "package_invalid", "classification/binary_score need label_dim 1");
        }
        if (pkg->task == BTF_TASK_BINARY_SCORE && out_elements != 1) {
            return fail(code, "package_invalid", "binary_score needs a single output value");
        }
        if (pkg->task == BTF_TASK_CLASSIFICATION && pkg->labels != NULL && pkg->label_count != out_elements) {
            return fail(code, "package_invalid", "label count must equal output elements");
        }
    }

    const uint32_t expected_bytes = (uint32_t)pkg->label_dim * 4u;
    if (pkg->demo_input != NULL && demo_input_size != pkg->input.bytes) {
        return fail(code, "package_invalid", "demo input size != input tensor bytes");
    }
    if (pkg->demo_expected != NULL && (pkg->demo_input == NULL || demo_expected_size != expected_bytes)) {
        return fail(code, "package_invalid", "bad demo expected section");
    }
    if (eval_inputs != NULL || eval_expected != NULL) {
        if (eval_inputs == NULL || eval_expected == NULL || eval_inputs_count != eval_expected_count ||
            (uint64_t)eval_inputs_count * pkg->input.bytes != eval_inputs_size ||
            (uint64_t)eval_expected_count * expected_bytes != eval_expected_size) {
            return fail(code, "package_invalid", "eval inputs/expected sections do not match");
        }
        pkg->eval_inputs = eval_inputs;
        pkg->eval_expected = eval_expected;
        pkg->eval_count = eval_inputs_count;
    }

    *code = NULL;
    return NULL;
}

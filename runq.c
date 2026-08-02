/* Inference for Gated DeltaNet in pure C, int8 quantized forward pass. */

#define _POSIX_C_SOURCE 200809L

#include <ctype.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

// ----------------------------------------------------------------------------
// GDN Q8 model

#define GDN_MAGIC UINT32_C(0x47444E65) /* "GDNe" on little-endian systems */
#define GDN_VERSION 2
#define HEADER_SIZE 256

typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int num_heads;
    int head_k_dim;
    int head_v_dim;
    int conv_size;
    int vocab_size;
    int seq_len;
    int shared_classifier;
    int _pad;
} Config;

_Static_assert(sizeof(Config) == 44, "Q8 Config must contain eleven 32-bit integers");

typedef struct {
    int8_t *q;    // quantized values
    float *s;     // scaling factors
    int rows;
    int cols;
} QuantizedTensor;

typedef struct {
    const float *attn_norm;
    QuantizedTensor q_proj;
    QuantizedTensor k_proj;
    QuantizedTensor v_proj;
    const float *a_proj;
    const float *b_proj;
    QuantizedTensor g_proj;
    const float *q_conv;
    const float *k_conv;
    const float *v_conv;
    const float *A;
    const float *dt_bias;
    const float *o_norm;
    QuantizedTensor o_proj;
} MixerWeights;

typedef struct {
    const float *norm;
    QuantizedTensor w1;
    QuantizedTensor w2;
    QuantizedTensor w3;
} FFNWeights;

typedef struct {
    MixerWeights mixer;
    FFNWeights ffn;
} LayerWeights;

typedef struct {
    QuantizedTensor token_embedding_table;
    LayerWeights *layers;
    const float *rms_final_weight;
    QuantizedTensor wcls;
} Weights;

typedef struct {
    float *x, *xb, *xb2;
    float *hb, *hb2;
    float *q, *k, *v;
    float *gate;
    float *beta;
    float *g;
    float *linear_out;
    float *logits;
    float *q_conv_state, *k_conv_state, *v_conv_state;
    float *S;
    QuantizedTensor xq; // quantized activations for W8A8 matmuls
} RunState;

typedef struct {
    Config config;
    int group_size;
    Weights weights;
    RunState state;
    int fd;
    void *mapped_data;
    size_t file_size;
} GDNQ8;

static void *xcalloc(size_t n, size_t size) {
    void *ptr = calloc(n, size);
    if (!ptr) {
        fprintf(stderr, "calloc failed\n");
        exit(EXIT_FAILURE);
    }
    return ptr;
}

static float sigmoidf_(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static float silu(float x) {
    return x * sigmoidf_(x);
}

static void rmsnorm(float *o, const float *x, const float *weight, int size, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < size; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / size + eps);
    for (int i = 0; i < size; i++) o[i] = weight[i] * (ss * x[i]);
}

static void head_rmsnorm_gated(float *x, const float *gate, const float *weight, int n_heads, int head_dim, float eps) {
    for (int h = 0; h < n_heads; h++) {
        float *xh = x + h * head_dim;
        const float *gh = gate + h * head_dim;
        float ss = 0.0f;
        for (int i = 0; i < head_dim; i++) ss += xh[i] * xh[i];
        ss = 1.0f / sqrtf(ss / head_dim + eps);
        for (int i = 0; i < head_dim; i++) xh[i] = weight[i] * (ss * xh[i]) * silu(gh[i]);
    }
}

static float softplus(float x) {
    return x > 20.0f ? x : log1pf(expf(x));
}

static void l2norm(float *x, int size) {
    float ss = 0.0f;
    for (int i = 0; i < size; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss + 1e-6f);
    for (int i = 0; i < size; i++) x[i] *= ss;
}

static void softmax(float *x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) x[i] /= sum;
}

static float matmul_scalar(const float *x, const float *w, int n) {
    float val = 0.0f;
    for (int i = 0; i < n; i++) val += w[i] * x[i];
    return val;
}

// ----------------------------------------------------------------------------
// Quantization functions

static void dequantize(QuantizedTensor *qx, float *x, int n, int group_size) {
    for (int i = 0; i < n; i++) {
        x[i] = (float)qx->q[i] * qx->s[i / group_size];
    }
}

static void quantize(QuantizedTensor *qx, const float *x, int n, int group_size) {
    int num_groups = n / group_size;
    const float Q_MAX = 127.0f;
    for (int group = 0; group < num_groups; group++) {
        int offset = group * group_size;
        float wmax = 0.0f;
        for (int i = 0; i < group_size; i++) {
            float val = fabsf(x[offset + i]);
            if (val > wmax) wmax = val;
        }
        if (wmax == 0.0f) {
            qx->s[group] = 1.0f;
            memset(qx->q + offset, 0, (size_t)group_size);
            continue;
        }
        float scale = wmax / Q_MAX;
        qx->s[group] = scale;
        for (int i = 0; i < group_size; i++) {
            int quantized = (int)roundf(x[offset + i] / scale);
            if (quantized > 127) quantized = 127;
            if (quantized < -127) quantized = -127;
            qx->q[offset + i] = (int8_t)quantized;
        }
    }
}

static void matmul(float *xout, QuantizedTensor *x, const QuantizedTensor *w, int group_size) {
    // W (d,n) @ x (n,) -> xout (d,). Both x and W are Q8_0.
    int n = w->cols;
    int d = w->rows;
    int groups = n / group_size;
    #ifdef _OPENMP
    #pragma omp parallel for if (d > 256)
    #endif
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        size_t q_row = (size_t)i * n;
        size_t s_row = (size_t)i * groups;
        for (int group = 0; group < groups; group++) {
            int32_t ival = 0;
            int offset = group * group_size;
            for (int j = 0; j < group_size; j++) {
                ival += (int32_t)x->q[offset + j] * (int32_t)w->q[q_row + offset + j];
            }
            val += (float)ival * w->s[s_row + group] * x->s[group];
        }
        xout[i] = val;
    }
}

static void dequantize_row(float *out, const QuantizedTensor *table, int row, int group_size) {
    int n = table->cols;
    QuantizedTensor qx = {
        .q = table->q + (size_t)row * n,
        .s = table->s + (size_t)row * (n / group_size),
        .rows = 1,
        .cols = n,
    };
    dequantize(&qx, out, n, group_size);
}

static void malloc_run_state(RunState *s, const Config *p, int group_size) {
    int key_dim = p->num_heads * p->head_k_dim;
    int value_dim = p->num_heads * p->head_v_dim;
    int max_activation = p->hidden_dim;
    if (p->dim > max_activation) max_activation = p->dim;
    if (value_dim > max_activation) max_activation = value_dim;

    s->x = xcalloc(p->dim, sizeof(float));
    s->xb = xcalloc((value_dim > p->dim ? value_dim : p->dim), sizeof(float));
    s->xb2 = xcalloc(p->dim, sizeof(float));
    s->hb = xcalloc(p->hidden_dim, sizeof(float));
    s->hb2 = xcalloc(p->hidden_dim, sizeof(float));
    s->q = xcalloc(key_dim, sizeof(float));
    s->k = xcalloc(key_dim, sizeof(float));
    s->v = xcalloc(value_dim, sizeof(float));
    s->gate = xcalloc(value_dim, sizeof(float));
    s->beta = xcalloc(p->num_heads, sizeof(float));
    s->g = xcalloc(p->num_heads, sizeof(float));
    s->linear_out = xcalloc(value_dim, sizeof(float));
    s->logits = xcalloc(p->vocab_size, sizeof(float));
    s->q_conv_state = xcalloc((size_t)p->n_layers * key_dim * p->conv_size, sizeof(float));
    s->k_conv_state = xcalloc((size_t)p->n_layers * key_dim * p->conv_size, sizeof(float));
    s->v_conv_state = xcalloc((size_t)p->n_layers * value_dim * p->conv_size, sizeof(float));
    s->S = xcalloc((size_t)p->n_layers * p->num_heads * p->head_k_dim * p->head_v_dim, sizeof(float));
    s->xq.q = xcalloc(max_activation, sizeof(int8_t));
    s->xq.s = xcalloc(max_activation / group_size, sizeof(float));
    s->xq.rows = 1;
    s->xq.cols = max_activation;
}

static void free_run_state(RunState *s) {
    free(s->x); free(s->xb); free(s->xb2); free(s->hb); free(s->hb2);
    free(s->q); free(s->k); free(s->v); free(s->gate); free(s->beta); free(s->g);
    free(s->linear_out); free(s->logits); free(s->q_conv_state); free(s->k_conv_state);
    free(s->v_conv_state); free(s->S); free(s->xq.q); free(s->xq.s);
}

static void require_bytes(const uint8_t *ptr, const uint8_t *end, size_t bytes, const char *name) {
    if (ptr > end || bytes > (size_t)(end - ptr)) {
        fprintf(stderr, "truncated Q8 checkpoint while mapping %s\n", name);
        exit(EXIT_FAILURE);
    }
}

static const float *map_fp32(const uint8_t **ptr, const uint8_t *end, size_t count, const char *name) {
    size_t bytes = count * sizeof(float);
    require_bytes(*ptr, end, bytes, name);
    const float *result = (const float *)*ptr;
    *ptr += bytes;
    return result;
}

static QuantizedTensor map_q8(const uint8_t **ptr, const uint8_t *end, int rows, int cols, int group_size, const char *name) {
    QuantizedTensor result = {.rows = rows, .cols = cols};
    size_t q_count = (size_t)rows * cols;
    size_t scale_count = (size_t)rows * (cols / group_size);
    require_bytes(*ptr, end, q_count, name);
    result.q = (int8_t *)*ptr;
    *ptr += q_count;
    require_bytes(*ptr, end, scale_count * sizeof(float), name);
    result.s = (float *)*ptr;
    *ptr += scale_count * sizeof(float);
    return result;
}

static void validate_config(const Config *p, int group_size) {
    int value_dim = p->num_heads * p->head_v_dim;
    if (p->dim <= 0 || p->hidden_dim <= 0 || p->n_layers <= 0 || p->num_heads <= 0
        || p->head_k_dim <= 0 || p->head_v_dim <= 0 || p->conv_size <= 0
        || p->vocab_size <= 0 || group_size <= 0) {
        fprintf(stderr, "invalid Q8 checkpoint dimensions\n");
        exit(EXIT_FAILURE);
    }
    if (p->dim % group_size != 0 || p->hidden_dim % group_size != 0 || value_dim % group_size != 0) {
        fprintf(stderr, "Q8 group size %d does not divide all matrix input dimensions\n", group_size);
        exit(EXIT_FAILURE);
    }
    if (p->shared_classifier != 0 && p->shared_classifier != 1) {
        fprintf(stderr, "invalid shared-classifier flag %d\n", p->shared_classifier);
        exit(EXIT_FAILURE);
    }
}

static void read_checkpoint(const char *path, GDNQ8 *model) {
    model->fd = open(path, O_RDONLY);
    if (model->fd == -1) {
        fprintf(stderr, "could not open checkpoint %s\n", path);
        exit(EXIT_FAILURE);
    }
    struct stat st;
    if (fstat(model->fd, &st) != 0 || st.st_size < HEADER_SIZE) {
        fprintf(stderr, "invalid v2 checkpoint size for %s\n", path);
        exit(EXIT_FAILURE);
    }
    model->file_size = (size_t)st.st_size;
    model->mapped_data = mmap(NULL, model->file_size, PROT_READ, MAP_PRIVATE, model->fd, 0);
    if (model->mapped_data == MAP_FAILED) {
        fprintf(stderr, "mmap failed for %s\n", path);
        exit(EXIT_FAILURE);
    }

    const uint8_t *header = (const uint8_t *)model->mapped_data;
    uint32_t magic;
    int version;
    memcpy(&magic, header, sizeof(magic));
    memcpy(&version, header + sizeof(magic), sizeof(version));
    if (magic != GDN_MAGIC) {
        fprintf(stderr, "bad checkpoint magic: 0x%08x (expected GDNe)\n", magic);
        exit(EXIT_FAILURE);
    }
    if (version != GDN_VERSION) {
        fprintf(stderr, "runq requires checkpoint version 2, got %d\n", version);
        exit(EXIT_FAILURE);
    }
    memcpy(&model->config, header + sizeof(magic) + sizeof(version), sizeof(Config));
    memcpy(&model->group_size, header + sizeof(magic) + sizeof(version) + sizeof(Config), sizeof(int));
    validate_config(&model->config, model->group_size);

    Config *p = &model->config;
    Weights *w = &model->weights;
    int key_dim = p->num_heads * p->head_k_dim;
    int value_dim = p->num_heads * p->head_v_dim;
    const uint8_t *ptr = header + HEADER_SIZE;
    const uint8_t *end = header + model->file_size;

    w->token_embedding_table = map_q8(&ptr, end, p->vocab_size, p->dim, model->group_size, "embedding");
    w->layers = xcalloc(p->n_layers, sizeof(LayerWeights));
    for (int layer = 0; layer < p->n_layers; layer++) {
        MixerWeights *m = &w->layers[layer].mixer;
        FFNWeights *f = &w->layers[layer].ffn;
        m->attn_norm = map_fp32(&ptr, end, p->dim, "attention norm");
        m->q_proj = map_q8(&ptr, end, key_dim, p->dim, model->group_size, "Q projection");
        m->k_proj = map_q8(&ptr, end, key_dim, p->dim, model->group_size, "K projection");
        m->v_proj = map_q8(&ptr, end, value_dim, p->dim, model->group_size, "V projection");
        m->a_proj = map_fp32(&ptr, end, (size_t)p->num_heads * p->dim, "a projection");
        m->b_proj = map_fp32(&ptr, end, (size_t)p->num_heads * p->dim, "b projection");
        m->g_proj = map_q8(&ptr, end, value_dim, p->dim, model->group_size, "G projection");
        m->q_conv = map_fp32(&ptr, end, (size_t)key_dim * p->conv_size, "Q convolution");
        m->k_conv = map_fp32(&ptr, end, (size_t)key_dim * p->conv_size, "K convolution");
        m->v_conv = map_fp32(&ptr, end, (size_t)value_dim * p->conv_size, "V convolution");
        m->A = map_fp32(&ptr, end, p->num_heads, "A");
        m->dt_bias = map_fp32(&ptr, end, p->num_heads, "dt bias");
        m->o_norm = map_fp32(&ptr, end, p->head_v_dim, "output norm");
        m->o_proj = map_q8(&ptr, end, p->dim, value_dim, model->group_size, "O projection");

        f->norm = map_fp32(&ptr, end, p->dim, "FFN norm");
        f->w1 = map_q8(&ptr, end, p->hidden_dim, p->dim, model->group_size, "FFN gate projection");
        f->w2 = map_q8(&ptr, end, p->dim, p->hidden_dim, model->group_size, "FFN down projection");
        f->w3 = map_q8(&ptr, end, p->hidden_dim, p->dim, model->group_size, "FFN up projection");
    }
    w->rms_final_weight = map_fp32(&ptr, end, p->dim, "final norm");
    w->wcls = p->shared_classifier
        ? w->token_embedding_table
        : map_q8(&ptr, end, p->vocab_size, p->dim, model->group_size, "classifier");
    if (ptr != end) {
        fprintf(stderr, "Q8 checkpoint has %zu unexpected trailing bytes\n", (size_t)(end - ptr));
        exit(EXIT_FAILURE);
    }
}

static void free_model(GDNQ8 *model) {
    free(model->weights.layers);
    if (model->mapped_data && model->mapped_data != MAP_FAILED) munmap(model->mapped_data, model->file_size);
    if (model->fd >= 0) close(model->fd);
    free_run_state(&model->state);
}

static void conv_step(float *out, float *state, const float *input, const float *weight, int channels, int kernel) {
    for (int channel = 0; channel < channels; channel++) {
        float *st = state + channel * kernel;
        for (int i = 0; i < kernel - 1; i++) st[i] = st[i + 1];
        st[kernel - 1] = input[channel];
        float value = 0.0f;
        for (int i = 0; i < kernel; i++) value += st[i] * weight[channel * kernel + i];
        out[channel] = silu(value);
    }
}

static void mixer_forward(GDNQ8 *model, const MixerWeights *w, int layer) {
    Config *p = &model->config;
    RunState *s = &model->state;
    int dim = p->dim;
    int key_dim = p->num_heads * p->head_k_dim;
    int value_dim = p->num_heads * p->head_v_dim;
    const float eps = 1e-5f;

    rmsnorm(s->xb, s->x, w->attn_norm, dim, eps);
    quantize(&s->xq, s->xb, dim, model->group_size);
    matmul(s->q, &s->xq, &w->q_proj, model->group_size);
    matmul(s->k, &s->xq, &w->k_proj, model->group_size);
    matmul(s->v, &s->xq, &w->v_proj, model->group_size);
    matmul(s->gate, &s->xq, &w->g_proj, model->group_size);

    conv_step(s->q, s->q_conv_state + (size_t)layer * key_dim * p->conv_size,
              s->q, w->q_conv, key_dim, p->conv_size);
    conv_step(s->k, s->k_conv_state + (size_t)layer * key_dim * p->conv_size,
              s->k, w->k_conv, key_dim, p->conv_size);
    conv_step(s->v, s->v_conv_state + (size_t)layer * value_dim * p->conv_size,
              s->v, w->v_conv, value_dim, p->conv_size);

    for (int head = 0; head < p->num_heads; head++) {
        float *q = s->q + head * p->head_k_dim;
        float *k = s->k + head * p->head_k_dim;
        l2norm(q, p->head_k_dim);
        l2norm(k, p->head_k_dim);
        float scale = 1.0f / sqrtf((float)p->head_k_dim);
        for (int i = 0; i < p->head_k_dim; i++) q[i] *= scale;
        s->beta[head] = sigmoidf_(matmul_scalar(s->xb, w->b_proj + head * dim, dim));
        s->g[head] = expf(w->A[head] * softplus(
            matmul_scalar(s->xb, w->a_proj + head * dim, dim) + w->dt_bias[head]));
    }

    float *S_layer = s->S + (size_t)layer * p->num_heads * p->head_k_dim * p->head_v_dim;
    for (int head = 0; head < p->num_heads; head++) {
        float *S = S_layer + (size_t)head * p->head_k_dim * p->head_v_dim;
        float *q = s->q + head * p->head_k_dim;
        float *k = s->k + head * p->head_k_dim;
        float *v = s->v + head * p->head_v_dim;
        float *out = s->linear_out + head * p->head_v_dim;

        for (int i = 0; i < p->head_k_dim * p->head_v_dim; i++) S[i] *= s->g[head];
        for (int j = 0; j < p->head_v_dim; j++) {
            float prediction = 0.0f;
            for (int i = 0; i < p->head_k_dim; i++) prediction += S[i * p->head_v_dim + j] * k[i];
            s->xb[j] = (v[j] - prediction) * s->beta[head];
        }
        for (int i = 0; i < p->head_k_dim; i++) {
            for (int j = 0; j < p->head_v_dim; j++) S[i * p->head_v_dim + j] += k[i] * s->xb[j];
        }
        for (int j = 0; j < p->head_v_dim; j++) {
            float value = 0.0f;
            for (int i = 0; i < p->head_k_dim; i++) value += q[i] * S[i * p->head_v_dim + j];
            out[j] = value;
        }
    }

    head_rmsnorm_gated(s->linear_out, s->gate, w->o_norm, p->num_heads, p->head_v_dim, eps);
    quantize(&s->xq, s->linear_out, value_dim, model->group_size);
    matmul(s->xb, &s->xq, &w->o_proj, model->group_size);
    for (int i = 0; i < dim; i++) s->x[i] += s->xb[i];
}

static void ffn_forward(GDNQ8 *model, const FFNWeights *w) {
    Config *p = &model->config;
    RunState *s = &model->state;
    const float eps = 1e-5f;

    rmsnorm(s->xb, s->x, w->norm, p->dim, eps);
    quantize(&s->xq, s->xb, p->dim, model->group_size);
    matmul(s->hb, &s->xq, &w->w1, model->group_size);
    matmul(s->hb2, &s->xq, &w->w3, model->group_size);
    for (int i = 0; i < p->hidden_dim; i++) s->hb[i] = silu(s->hb[i]) * s->hb2[i];
    quantize(&s->xq, s->hb, p->hidden_dim, model->group_size);
    matmul(s->xb, &s->xq, &w->w2, model->group_size);
    for (int i = 0; i < p->dim; i++) s->x[i] += s->xb[i];
}

static float *forward(GDNQ8 *model, int token) {
    Config *p = &model->config;
    Weights *w = &model->weights;
    RunState *s = &model->state;

    dequantize_row(s->x, &w->token_embedding_table, token, model->group_size);
    for (int layer = 0; layer < p->n_layers; layer++) {
        mixer_forward(model, &w->layers[layer].mixer, layer);
        ffn_forward(model, &w->layers[layer].ffn);
    }
    rmsnorm(s->x, s->x, w->rms_final_weight, p->dim, 1e-5f);
    quantize(&s->xq, s->x, p->dim, model->group_size);
    matmul(s->logits, &s->xq, &w->wcls, model->group_size);
    return s->logits;
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char **vocab;
    float *vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512];
} Tokenizer;

static int compare_tokens(const void *a, const void *b) {
    return strcmp(((const TokenIndex *)a)->str, ((const TokenIndex *)b)->str);
}

static void build_tokenizer(Tokenizer *t, const char *tokenizer_path, int vocab_size) {
    t->vocab_size = vocab_size;
    t->vocab = xcalloc(vocab_size, sizeof(char *));
    t->vocab_scores = xcalloc(vocab_size, sizeof(float));
    t->sorted_vocab = NULL;
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) {
        fprintf(stderr, "could not load tokenizer %s\n", tokenizer_path);
        exit(EXIT_FAILURE);
    }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) exit(EXIT_FAILURE);
    for (int i = 0; i < vocab_size; i++) {
        int len;
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) exit(EXIT_FAILURE);
        if (fread(&len, sizeof(int), 1, file) != 1) exit(EXIT_FAILURE);
        t->vocab[i] = xcalloc((size_t)len + 1, sizeof(char));
        if (len > 0 && fread(t->vocab[i], len, 1, file) != 1) exit(EXIT_FAILURE);
    }
    fclose(file);
}

static void free_tokenizer(Tokenizer *t) {
    for (int i = 0; i < t->vocab_size; i++) free(t->vocab[i]);
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

static char *decode(Tokenizer *t, int prev_token, int token) {
    char *piece = t->vocab[token];
    if (prev_token == 1 && piece[0] == ' ') piece++;
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char *)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

static void safe_printf(char *piece) {
    if (!piece || piece[0] == '\0') return;
    if (piece[1] == '\0') {
        unsigned char byte_val = (unsigned char)piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) return;
    }
    printf("%s", piece);
}

static int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    TokenIndex token = {.str = str};
    TokenIndex *result = bsearch(&token, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return result ? result->id : -1;
}

static void encode(Tokenizer *t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }

    if (t->sorted_vocab == NULL) {
        // lazily malloc and sort the vocabulary
        t->sorted_vocab = xcalloc(t->vocab_size, sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    char *str_buffer = xcalloc((size_t)t->max_token_length * 2 + 1 + 2, sizeof(char));
    size_t str_len = 0;

    // start at 0 tokens
    *n_tokens = 0;

    // add optional BOS (=1) token, if desired
    if (bos) tokens[(*n_tokens)++] = 1;

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    // process the raw (UTF-8) byte sequence of the input string
    for (char *c = text; *c != '\0'; c++) {
        // reset buffer if the current byte is ASCII or a leading byte
        if ((*c & 0xC0) != 0x80) {
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int i = 0; i < (int)str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        float best_score = -1e10f;
        int best_id = -1;
        int best_idx = -1;

        for (int i = 0; i < (*n_tokens - 1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx + 1; i < (*n_tokens - 1); i++) {
            tokens[i] = tokens[i + 1];
        }
        (*n_tokens)--; // token length decreased
    }

    // add optional EOS (=2) token, if desired
    if (eos) tokens[(*n_tokens)++] = 2;

    free(str_buffer);
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token

typedef struct {
    float prob;
    int index;
} ProbIndex;

typedef struct {
    int vocab_size;
    ProbIndex *probindex;
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

static int sample_argmax(float *probabilities, int n) {
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

static int sample_mult(float *probabilities, int n, float coin) {
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) return i;
    }
    return n - 1;
}

static int compare_prob(const void *a, const void *b) {
    const ProbIndex *pa = (const ProbIndex *)a;
    const ProbIndex *pb = (const ProbIndex *)b;
    return (pa->prob < pb->prob) - (pa->prob > pb->prob);
}

static int sample_topp(float *probabilities, int n, float topp, ProbIndex *probindex, float coin) {
    int n0 = 0;
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare_prob);
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1;
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break;
        }
    }
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) return probindex[i].index;
    }
    return probindex[last_idx].index;
}

static unsigned int random_u32(unsigned long long *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (unsigned int)((*state * UINT64_C(0x2545F4914F6CDD1D)) >> 32);
}

static float random_f32(unsigned long long *state) {
    return (random_u32(state) >> 8) / 16777216.0f;
}

static int sample(Sampler *sampler, float *logits) {
    if (sampler->temperature == 0.0f) return sample_argmax(logits, sampler->vocab_size);
    for (int i = 0; i < sampler->vocab_size; i++) logits[i] /= sampler->temperature;
    softmax(logits, sampler->vocab_size);
    float coin = random_f32(&sampler->rng_state);
    if (sampler->topp <= 0 || sampler->topp >= 1) {
        return sample_mult(logits, sampler->vocab_size, coin);
    }
    return sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
}

static long time_in_ms(void) {
    struct timespec now;
    clock_gettime(CLOCK_REALTIME, &now);
    return now.tv_sec * 1000 + now.tv_nsec / 1000000;
}

// ----------------------------------------------------------------------------
// generation loop

static void generate(GDNQ8 *model, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    int num_prompt_tokens = 0;
    int *prompt_tokens = xcalloc(strlen(prompt) * 2 + 3, sizeof(int));
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    long start = 0;
    int next;
    int token = prompt_tokens[0];
    int pos = 0;
    while (pos < steps) {
        float *logits = forward(model, token);
        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            next = sample(sampler, logits);
        }
        pos++;

        if (next == 2) { break; } // EOS token, finish
        char *piece = decode(tokenizer, token, next);
        safe_printf(piece);
        fflush(stdout);
        token = next;
        if (start == 0) { start = time_in_ms(); }
    }
    printf("\n");

    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "achieved tok/s: %f\n", (pos - 1) / (double)(end - start) * 1000);
    }
    free(prompt_tokens);
}

static void read_stdin(const char *guide, char *buffer, size_t bufsize) {
    printf("%s", guide);
    fflush(stdout);
    if (fgets(buffer, bufsize, stdin) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0';
        }
    } else {
        buffer[0] = '\0';
    }
}

static void apply_repetition_penalty(float *logits, const int *tokens, int num_tokens, float penalty) {
    if (penalty <= 1.0f) return;
    for (int i = 0; i < num_tokens; i++) {
        int already_penalized = 0;
        for (int j = 0; j < i; j++) {
            if (tokens[j] == tokens[i]) {
                already_penalized = 1;
                break;
            }
        }
        if (already_penalized) continue;
        int token = tokens[i];
        logits[token] = logits[token] < 0.0f
            ? logits[token] * penalty
            : logits[token] / penalty;
    }
}

static int would_repeat_ngram(const int *tokens, int num_tokens, int next, int n) {
    if (n < 2 || num_tokens < n) return 0;
    int suffix_start = num_tokens - (n - 1);
    for (int start = 0; start <= num_tokens - n; start++) {
        int matches = 1;
        for (int offset = 0; offset < n - 1; offset++) {
            if (tokens[start + offset] != tokens[suffix_start + offset]) {
                matches = 0;
                break;
            }
        }
        if (matches && tokens[start + n - 1] == next) return 1;
    }
    return 0;
}

static int is_exit_command(const char *prompt) {
    return strcmp(prompt, "/exit") == 0
        || strcmp(prompt, "exit") == 0
        || strcmp(prompt, "quit") == 0;
}

// ----------------------------------------------------------------------------
// chat loop
// GDN is recurrent (no KV cache / pos). Turn boundaries follow run.c: keep
// state across turns and feed EOS after each assistant reply, matching SFT.

static void chat(GDNQ8 *model, Tokenizer *tokenizer, Sampler *sampler,
                 char *cli_user_prompt, char *cli_system_prompt, int steps) {
    const float repetition_penalty = 1.10f;
    const int max_response_tokens = 64;
    char system_prompt[512];
    char user_prompt[512];
    int response_tokens[64];
    int total_steps = 0;
    int first_turn = 1;

    system_prompt[0] = '\0';
    if (cli_system_prompt != NULL) {
        snprintf(system_prompt, sizeof(system_prompt), "%s", cli_system_prompt);
    }

    while (total_steps < steps) {
        if (first_turn) {
            if (cli_system_prompt == NULL) {
                read_stdin("Enter system prompt (optional): ", system_prompt, sizeof(system_prompt));
            }
            if (cli_user_prompt != NULL) {
                snprintf(user_prompt, sizeof(user_prompt), "%s", cli_user_prompt);
            } else {
                read_stdin("User: ", user_prompt, sizeof(user_prompt));
            }
        } else {
            read_stdin("User: ", user_prompt, sizeof(user_prompt));
        }
        if (is_exit_command(user_prompt)) break;
        if (user_prompt[0] == '\0') {
            first_turn = 0;
            continue;
        }

        size_t rendered_size = strlen(user_prompt) + 64;
        if (first_turn && system_prompt[0] != '\0') rendered_size += strlen(system_prompt);
        char *rendered_prompt = xcalloc(rendered_size, sizeof(char));
        if (first_turn && system_prompt[0] != '\0') {
            snprintf(rendered_prompt, rendered_size,
                     "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]",
                     system_prompt, user_prompt);
        } else {
            snprintf(rendered_prompt, rendered_size, "[INST] %s [/INST]", user_prompt);
        }

        int num_prompt_tokens = 0;
        int *prompt_tokens = xcalloc(strlen(rendered_prompt) * 2 + 3, sizeof(int));
        encode(tokenizer, rendered_prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
        free(rendered_prompt);
        if (total_steps + num_prompt_tokens >= steps) {
            free(prompt_tokens);
            fprintf(stderr, "chat context limit reached\n");
            break;
        }

        float *logits = NULL;
        int previous_token = 1;
        for (int i = 0; i < num_prompt_tokens; i++) {
            previous_token = prompt_tokens[i];
            logits = forward(model, previous_token);
            total_steps++;
        }
        free(prompt_tokens);

        printf("Assistant: ");
        fflush(stdout);
        int response_len = 0;
        while (response_len < max_response_tokens && total_steps < steps) {
            apply_repetition_penalty(logits, response_tokens, response_len, repetition_penalty);
            int next = sample(sampler, logits);
            if (next == 1 || next == 2) break;
            if (would_repeat_ngram(response_tokens, response_len, next, 5)) break;

            response_tokens[response_len++] = next;
            safe_printf(decode(tokenizer, previous_token, next));
            fflush(stdout);
            previous_token = next;
            logits = forward(model, next);
            total_steps++;
        }
        printf("\n");

        if (total_steps < steps) {
            forward(model, 2); // EOS between turns, matching SFT
            total_steps++;
        }
        first_turn = 0;
        cli_user_prompt = NULL;
    }
}

// ----------------------------------------------------------------------------
// CLI

static void error_usage(void) {
    fprintf(stderr, "Usage:   runq <checkpoint> [options]\n");
    fprintf(stderr, "Example: runq modelq.bin -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n");
    fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {

    // default parameters
    char *checkpoint_path = NULL;
    char *tokenizer_path = "tokenizer.bin";
    float temperature = 1.0f;
    float topp = 0.9f;
    int steps = 256;
    char *prompt = NULL;
    unsigned long long rng_seed = 0;
    char *mode = "generate";
    char *system_prompt = NULL;

    if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }
    for (int i = 2; i < argc; i += 2) {
        if (i + 1 >= argc) { error_usage(); }
        if (argv[i][0] != '-') { error_usage(); }
        if (strlen(argv[i]) != 2) { error_usage(); }
        if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
        else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
        else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
        else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
        else if (argv[i][1] == 'y') { system_prompt = argv[i + 1]; }
        else { error_usage(); }
    }

    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    // resolve tokenizer path with a few common fallbacks
    {
        const char *candidates[] = {
            tokenizer_path,
            "tokenizer.bin",
            "../tokenizer.bin",
            "../llama2.c/tokenizer.bin",
            "llama2.c/tokenizer.bin",
            NULL,
        };
        tokenizer_path = "tokenizer.bin";
        for (int i = 0; candidates[i]; i++) {
            FILE *file = fopen(candidates[i], "rb");
            if (file) {
                fclose(file);
                tokenizer_path = (char *)candidates[i];
                break;
            }
        }
    }

    GDNQ8 model = {.fd = -1};
    read_checkpoint(checkpoint_path, &model);
    malloc_run_state(&model.state, &model.config, model.group_size);
    if (steps == 0 || steps > model.config.seq_len) steps = model.config.seq_len;

    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, model.config.vocab_size);

    Sampler sampler = {
        .vocab_size = model.config.vocab_size,
        .probindex = xcalloc(model.config.vocab_size, sizeof(ProbIndex)),
        .temperature = temperature,
        .topp = topp,
        .rng_state = rng_seed,
    };

    if (strcmp(mode, "generate") == 0) {
        generate(&model, &tokenizer, &sampler, prompt, steps);
    } else if (strcmp(mode, "chat") == 0) {
        chat(&model, &tokenizer, &sampler, prompt, system_prompt, steps);
    } else {
        fprintf(stderr, "unknown mode: %s\n", mode);
        error_usage();
    }

    free(sampler.probindex);
    free_tokenizer(&tokenizer);
    free_model(&model);
    return 0;
}

/* Inference for the Gated DeltaNet language model in pure C. */

#define _POSIX_C_SOURCE 200809L

#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

typedef struct {
    float *token_embedding_table;
    float *layers;
    float *rms_final_weight;
    float *wcls;
    size_t layer_stride;
    float *data;
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
} RunState;

typedef struct {
    Config config;
    Weights weights;
    RunState state;
} GDN;

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

static void matmul(float *out, const float *x, const float *w, int n, int d) {
    #pragma omp parallel for if (d > 256)
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) val += w[i * n + j] * x[j];
        out[i] = val;
    }
}

static float matmul_scalar(const float *x, const float *w, int n) {
    float val = 0.0f;
    for (int i = 0; i < n; i++) val += w[i] * x[i];
    return val;
}

static void malloc_run_state(RunState *s, const Config *p) {
    int key_dim = p->num_heads * p->head_k_dim;
    int value_dim = p->num_heads * p->head_v_dim;
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
}

static void free_run_state(RunState *s) {
    free(s->x); free(s->xb); free(s->xb2); free(s->hb); free(s->hb2);
    free(s->q); free(s->k); free(s->v); free(s->gate); free(s->beta); free(s->g); free(s->linear_out);
    free(s->logits); free(s->q_conv_state); free(s->k_conv_state);
    free(s->v_conv_state); free(s->S);
}

static size_t layer_stride(const Config *p) {
    int key_dim = p->num_heads * p->head_k_dim;
    int value_dim = p->num_heads * p->head_v_dim;
    return (size_t)p->dim
        + (size_t)key_dim * p->dim
        + (size_t)key_dim * p->dim
        + (size_t)value_dim * p->dim
        + (size_t)p->num_heads * p->dim
        + (size_t)p->num_heads * p->dim
        + (size_t)value_dim * p->dim
        + (size_t)key_dim * p->conv_size
        + (size_t)key_dim * p->conv_size
        + (size_t)value_dim * p->conv_size
        + (size_t)p->num_heads
        + (size_t)p->num_heads
        + (size_t)p->head_v_dim
        + (size_t)p->dim * value_dim
        + (size_t)p->dim
        + (size_t)p->hidden_dim * p->dim
        + (size_t)p->dim * p->hidden_dim
        + (size_t)p->hidden_dim * p->dim;
}

static void read_checkpoint(const char *path, GDN *model) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "could not open checkpoint %s\n", path);
        exit(EXIT_FAILURE);
    }
    uint32_t magic;
    int version;
    if (fread(&magic, sizeof(uint32_t), 1, f) != 1 || fread(&version, sizeof(int), 1, f) != 1) {
        fprintf(stderr, "failed to read checkpoint header\n");
        exit(EXIT_FAILURE);
    }
    if (magic != 0x47444e65) {
        fprintf(stderr, "bad checkpoint magic for GDN: 0x%08x\n", magic);
        exit(EXIT_FAILURE);
    }
    if (version != 0 && version != 1) {
        fprintf(stderr, "run requires an fp32 checkpoint (version 0 or 1), got %d\n", version);
        exit(EXIT_FAILURE);
    }
    if (fread(&model->config, sizeof(Config), 1, f) != 1) {
        fprintf(stderr, "failed to read checkpoint config\n");
        exit(EXIT_FAILURE);
    }
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 256, SEEK_SET);
    size_t n_floats = (size_t)(file_size - 256) / sizeof(float);
    model->weights.data = xcalloc(n_floats, sizeof(float));
    if (fread(model->weights.data, sizeof(float), n_floats, f) != n_floats) {
        fprintf(stderr, "failed to read checkpoint weights\n");
        exit(EXIT_FAILURE);
    }
    fclose(f);

    Config *p = &model->config;
    Weights *w = &model->weights;
    w->layer_stride = layer_stride(p);
    float *ptr = w->data;
    w->token_embedding_table = ptr;
    ptr += (size_t)p->vocab_size * p->dim;
    w->layers = ptr;
    ptr += (size_t)p->n_layers * w->layer_stride;
    w->rms_final_weight = ptr;
    ptr += p->dim;
    w->wcls = p->shared_classifier ? w->token_embedding_table : ptr;
}

static void free_model(GDN *model) {
    free(model->weights.data);
    free_run_state(&model->state);
}

static void conv_step(float *out, float *state, const float *input, const float *weight, int channels, int kernel) {
    for (int c = 0; c < channels; c++) {
        float *st = state + c * kernel;
        for (int j = 0; j < kernel - 1; j++) st[j] = st[j + 1];
        st[kernel - 1] = input[c];
        float val = 0.0f;
        for (int j = 0; j < kernel; j++) val += st[j] * weight[c * kernel + j];
        out[c] = silu(val);
    }
}

static float *forward(GDN *model, int token) {
    Config *p = &model->config;
    Weights *w = &model->weights;
    RunState *s = &model->state;
    int dim = p->dim;
    int key_dim = p->num_heads * p->head_k_dim;
    int value_dim = p->num_heads * p->head_v_dim;
    const float eps = 1e-5f;

    memcpy(s->x, w->token_embedding_table + (size_t)token * dim, dim * sizeof(float));

    for (int l = 0; l < p->n_layers; l++) {
        float *base = w->layers + (size_t)l * w->layer_stride;
        float *attn_norm = base; base += dim;
        float *q_proj = base; base += (size_t)key_dim * dim;
        float *k_proj = base; base += (size_t)key_dim * dim;
        float *v_proj = base; base += (size_t)value_dim * dim;
        float *a_proj = base; base += (size_t)p->num_heads * dim;
        float *b_proj = base; base += (size_t)p->num_heads * dim;
        float *g_proj = base; base += (size_t)value_dim * dim;
        float *q_conv = base; base += (size_t)key_dim * p->conv_size;
        float *k_conv = base; base += (size_t)key_dim * p->conv_size;
        float *v_conv = base; base += (size_t)value_dim * p->conv_size;
        float *A = base; base += p->num_heads;
        float *dt_bias = base; base += p->num_heads;
        float *o_norm = base; base += p->head_v_dim;
        float *o_proj = base; base += (size_t)dim * value_dim;
        float *ffn_norm = base; base += dim;
        float *w1 = base; base += (size_t)p->hidden_dim * dim;
        float *w2 = base; base += (size_t)dim * p->hidden_dim;
        float *w3 = base;

        rmsnorm(s->xb, s->x, attn_norm, dim, eps);
        matmul(s->q, s->xb, q_proj, dim, key_dim);
        matmul(s->k, s->xb, k_proj, dim, key_dim);
        matmul(s->v, s->xb, v_proj, dim, value_dim);
        matmul(s->gate, s->xb, g_proj, dim, value_dim);

        conv_step(s->q, s->q_conv_state + (size_t)l * key_dim * p->conv_size, s->q, q_conv, key_dim, p->conv_size);
        conv_step(s->k, s->k_conv_state + (size_t)l * key_dim * p->conv_size, s->k, k_conv, key_dim, p->conv_size);
        conv_step(s->v, s->v_conv_state + (size_t)l * value_dim * p->conv_size, s->v, v_conv, value_dim, p->conv_size);

        for (int h = 0; h < p->num_heads; h++) {
            float *qh = s->q + h * p->head_k_dim;
            float *kh = s->k + h * p->head_k_dim;
            l2norm(qh, p->head_k_dim);
            l2norm(kh, p->head_k_dim);
            float scale = 1.0f / sqrtf((float)p->head_k_dim);
            for (int i = 0; i < p->head_k_dim; i++) qh[i] *= scale;
            s->beta[h] = sigmoidf_(matmul_scalar(s->xb, b_proj + h * dim, dim));
            s->g[h] = expf(A[h] * softplus(matmul_scalar(s->xb, a_proj + h * dim, dim) + dt_bias[h]));
        }

        float *S_layer = s->S + (size_t)l * p->num_heads * p->head_k_dim * p->head_v_dim;
        for (int h = 0; h < p->num_heads; h++) {
            float *S = S_layer + (size_t)h * p->head_k_dim * p->head_v_dim;
            float *q = s->q + h * p->head_k_dim;
            float *k = s->k + h * p->head_k_dim;
            float *v = s->v + h * p->head_v_dim;
            float *out = s->linear_out + h * p->head_v_dim;

            for (int i = 0; i < p->head_k_dim * p->head_v_dim; i++) S[i] *= s->g[h];
            for (int j = 0; j < p->head_v_dim; j++) {
                float pred = 0.0f;
                for (int i = 0; i < p->head_k_dim; i++) pred += S[i * p->head_v_dim + j] * k[i];
                s->xb[j] = (v[j] - pred) * s->beta[h];
            }
            for (int i = 0; i < p->head_k_dim; i++) {
                for (int j = 0; j < p->head_v_dim; j++) S[i * p->head_v_dim + j] += k[i] * s->xb[j];
            }
            for (int j = 0; j < p->head_v_dim; j++) {
                float val = 0.0f;
                for (int i = 0; i < p->head_k_dim; i++) val += q[i] * S[i * p->head_v_dim + j];
                out[j] = val;
            }
        }

        head_rmsnorm_gated(s->linear_out, s->gate, o_norm, p->num_heads, p->head_v_dim, eps);
        matmul(s->xb, s->linear_out, o_proj, value_dim, dim);
        for (int i = 0; i < dim; i++) s->x[i] += s->xb[i];

        rmsnorm(s->xb, s->x, ffn_norm, dim, eps);
        matmul(s->hb, s->xb, w1, dim, p->hidden_dim);
        matmul(s->hb2, s->xb, w3, dim, p->hidden_dim);
        for (int i = 0; i < p->hidden_dim; i++) s->hb[i] = silu(s->hb[i]) * s->hb2[i];
        matmul(s->xb, s->hb, w2, p->hidden_dim, dim);
        for (int i = 0; i < dim; i++) s->x[i] += s->xb[i];
    }

    rmsnorm(s->x, s->x, w->rms_final_weight, dim, eps);
    matmul(s->logits, s->x, w->wcls, dim, p->vocab_size);
    return s->logits;
}

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
    return strcmp(((TokenIndex *)a)->str, ((TokenIndex *)b)->str);
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
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) return;
    }
    printf("%s", piece);
}

static int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    TokenIndex tok = {.str = str};
    TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res ? res->id : -1;
}

static void encode(Tokenizer *t, const char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    if (t->sorted_vocab == NULL) {
        t->sorted_vocab = xcalloc(t->vocab_size, sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }
    char *str_buffer = xcalloc((size_t)t->max_token_length * 2 + 3, sizeof(char));
    *n_tokens = 0;
    if (bos) tokens[(*n_tokens)++] = 1;
    for (const char *c = text; *c; c++) {
        if ((*c & 0xC0) != 0x80) {
            int len = 1;
            while ((c[len] & 0xC0) == 0x80) len++;
            strncpy(str_buffer, c, len);
            str_buffer[len] = '\0';
            tokens[(*n_tokens)++] = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
        }
    }
    while (1) {
        float best_score = -1e10f;
        int best_id = -1, best_idx = -1;
        for (int i = 0; i < (*n_tokens - 1); i++) {
            snprintf(str_buffer, (size_t)t->max_token_length * 2 + 3, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }
        if (best_idx == -1) break;
        tokens[best_idx] = best_id;
        for (int i = best_idx + 1; i < (*n_tokens - 1); i++) tokens[i] = tokens[i + 1];
        (*n_tokens)--;
    }
    if (eos) tokens[(*n_tokens)++] = 2;
    free(str_buffer);
}

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
    return (unsigned int)((*state * 0x2545F4914F6CDD1Dull) >> 32);
}

static float random_f32(unsigned long long *state) {
    return (random_u32(state) >> 8) / 16777216.0f;
}

static int sample(Sampler *sampler, float *logits) {
    if (sampler->temperature == 0.0f) return sample_argmax(logits, sampler->vocab_size);
    for (int q = 0; q < sampler->vocab_size; q++) logits[q] /= sampler->temperature;
    softmax(logits, sampler->vocab_size);
    float coin = random_f32(&sampler->rng_state);
    if (sampler->topp <= 0 || sampler->topp >= 1) return sample_mult(logits, sampler->vocab_size, coin);
    return sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
}

static long time_in_ms(void) {
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

static void generate(GDN *model, Tokenizer *tokenizer, Sampler *sampler, const char *prompt, int steps) {
    int num_prompt_tokens = 0;
    int *prompt_tokens = xcalloc(strlen(prompt) + 3, sizeof(int));
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "empty prompt after encoding\n");
        exit(EXIT_FAILURE);
    }
    int token = prompt_tokens[0];
    int next = 0;
    long start = 0;
    for (int pos = 0; pos < steps; pos++) {
        float *logits = forward(model, token);
        next = (pos < num_prompt_tokens - 1) ? prompt_tokens[pos + 1] : sample(sampler, logits);
        if (next == 2) break;
        char *piece = decode(tokenizer, token, next);
        safe_printf(piece);
        fflush(stdout);
        token = next;
        if (start == 0) start = time_in_ms();
    }
    printf("\n");
    if (start != 0) {
        long end = time_in_ms();
        fprintf(stderr, "achieved tok/s: %f\n", (steps - 1) / (double)(end - start) * 1000);
    }
    free(prompt_tokens);
}

static const char *default_tokenizer_path(void) {
    static const char *paths[] = {
        "tokenizer.bin",
        "../tokenizer.bin",
        "../llama2.c/tokenizer.bin",
        "llama2.c/tokenizer.bin",
        NULL,
    };
    for (int i = 0; paths[i]; i++) {
        FILE *f = fopen(paths[i], "rb");
        if (f) {
            fclose(f);
            return paths[i];
        }
    }
    return "tokenizer.bin";
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: run <checkpoint> [-z tokenizer.bin] [-t temperature] [-p topp] [-s seed] [-n steps] [-i prompt]\n");
        return 1;
    }
    const char *checkpoint = argv[1];
    const char *tokenizer_path = default_tokenizer_path();
    float temperature = 1.0f;
    float topp = 0.9f;
    int steps = 256;
    const char *prompt = "";
    unsigned long long rng_seed = (unsigned long long)time(NULL);
    for (int i = 2; i < argc; i += 2) {
        if (i + 1 >= argc || argv[i][0] != '-') {
            fprintf(stderr, "Usage: run <checkpoint> [-z tokenizer.bin] [-t temperature] [-p topp] [-s seed] [-n steps] [-i prompt]\n");
            return 1;
        }
        if (argv[i][1] == 'z') tokenizer_path = argv[i + 1];
        else if (argv[i][1] == 't') temperature = atof(argv[i + 1]);
        else if (argv[i][1] == 'p') topp = atof(argv[i + 1]);
        else if (argv[i][1] == 's') rng_seed = strtoull(argv[i + 1], NULL, 10);
        else if (argv[i][1] == 'n') steps = atoi(argv[i + 1]);
        else if (argv[i][1] == 'i') prompt = argv[i + 1];
        else {
            fprintf(stderr, "unknown option %s\n", argv[i]);
            return 1;
        }
    }

    GDN model = {0};
    read_checkpoint(checkpoint, &model);
    malloc_run_state(&model.state, &model.config);

    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, model.config.vocab_size);

    Sampler sampler = {
        .vocab_size = model.config.vocab_size,
        .probindex = xcalloc(model.config.vocab_size, sizeof(ProbIndex)),
        .temperature = temperature,
        .topp = topp,
        .rng_state = rng_seed,
    };

    generate(&model, &tokenizer, &sampler, prompt, steps);

    free(sampler.probindex);
    free_tokenizer(&tokenizer);
    free_model(&model);
    return 0;
}

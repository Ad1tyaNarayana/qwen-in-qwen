#define TESTING
#include "run.c"

static uint32_t read_u32(FILE *file) {
    uint32_t value;
    if (fread(&value, sizeof(value), 1, file) != 1) {
        fprintf(stderr, "invalid evaluation corpus\n");
        exit(1);
    }
    return value;
}

int main(int argc, char **argv) {
    if (argc != 4) return 1;
    Transformer model;
    build_transformer(&model, argv[1]);
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, argv[2], model.config.vocab_size);
    FILE *file = fopen(argv[3], "rb");
    if (!file || model.config.seq_len < 128) return 1;
    uint32_t stories = read_u32(file);
    unsigned long long bytes = 0, targets = 0;
    double nll = 0;
    for (uint32_t story = 0; story < stories; story++) {
        uint32_t length = read_u32(file);
        if (length > 1048576) return 1;
        char *text = calloc(length + 1, 1);
        int *tokens = malloc((length + 3) * sizeof(int));
        if (!text || !tokens || fread(text, 1, length, file) != length) return 1;
        int count = 0;
        encode(&tokenizer, text, 1, 0, tokens, &count);
        for (int i = 0; i < count - 1; i++) {
            float *logits = forward(&model, tokens[i], i % 128);
            float maximum = logits[0];
            for (int j = 1; j < model.config.vocab_size; j++) {
                if (logits[j] > maximum) maximum = logits[j];
            }
            double sum = 0;
            for (int j = 0; j < model.config.vocab_size; j++) {
                sum += exp((double)logits[j] - maximum);
            }
            nll += log(sum) + maximum - logits[tokens[i + 1]];
            targets++;
        }
        bytes += length;
        free(text);
        free(tokens);
    }
    if (!targets || !bytes || !isfinite(nll)) return 1;
    printf("{\"stories\":%u,\"utf8_bytes\":%llu,\"tokens\":%llu,\"total_nll\":%.9f,\"nats_per_token\":%.9f,\"bits_per_byte\":%.9f,\"context\":128}\n",
           stories, bytes, targets, nll, nll / targets, nll / log(2.0) / bytes);
    fclose(file);
    free_tokenizer(&tokenizer);
    free_transformer(&model);
    return 0;
}

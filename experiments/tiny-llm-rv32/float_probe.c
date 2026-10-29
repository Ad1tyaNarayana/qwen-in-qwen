/* This wrapper compiles the locally copied, unmodified llama2.c run.c. */
#define TESTING
#include "run.c"
int main(int argc, char **argv) {
    if (argc < 5) return 2;
    Transformer t;
    Tokenizer z;
    build_transformer(&t, argv[1]);
    build_tokenizer(&z, argv[2], t.config.vocab_size);
    int prompt[4096], n;
    encode(&z, argv[3], 1, 0, prompt, &n);
    int token = prompt[0], steps = atoi(argv[4]);
    FILE *forced = argc == 6 ? fopen(argv[5], "r") : NULL;
    for (int pos = 0; pos < steps; pos++) {
        if (forced && fscanf(forced, "%d", &token) != 1) return 3;
        float *logits = forward(&t, token, pos);
        int next = sample_argmax(logits, t.config.vocab_size);
        printf("@%d %d %d 0 0\n", pos, token, next);
        if (pos + 1 < n) next = prompt[pos + 1];
        printf(">%s\n", decode(&z, token, next));
        token = next;
    }
    if (forced) fclose(forced);
    free_tokenizer(&z);
    free_transformer(&t);
    return 0;
}

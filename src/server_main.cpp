#include "RK35llm.h"
#include "server.h"
#include <iostream>
#include <string>
#include <cstdlib>

static void usage(const char* prog)
{
    fprintf(stderr,
        "Usage: %s <vlm_model> <llm_model> [options]\n"
        "\n"
        "Options:\n"
        "  --host <addr>        bind address (default: 0.0.0.0)\n"
        "  --port <n>           port (default: 8080)\n"
        "  --model-name <name>  model ID returned by /v1/models (default: qwen3-vl-2b)\n"
        "  --max-tokens <n>     max new tokens (default: 2048)\n"
        "  --context <n>        context length (default: 4096)\n"
        "\n"
        "API endpoints:\n"
        "  GET  /health\n"
        "  GET  /props\n"
        "  GET  /v1/models\n"
        "  POST /completion              (llama-server format)\n"
        "  POST /v1/chat/completions     (OpenAI-compatible, supports stream+images)\n"
        "  POST /tokenize\n",
        prog);
}

int main(int argc, char** argv)
{
    if (argc < 3) { usage(argv[0]); return 1; }

    std::string vlm_model  = argv[1];
    std::string llm_model  = argv[2];
    std::string host       = "0.0.0.0";
    int         port       = 8080;
    std::string model_name = "qwen3-vl-2b";
    int         max_tokens = 2048;
    int         context    = 4096;

    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--host"       && i+1 < argc) host       = argv[++i];
        else if (a == "--port"       && i+1 < argc) port       = std::atoi(argv[++i]);
        else if (a == "--model-name" && i+1 < argc) model_name = argv[++i];
        else if (a == "--max-tokens" && i+1 < argc) max_tokens = std::atoi(argv[++i]);
        else if (a == "--context"    && i+1 < argc) context    = std::atoi(argv[++i]);
        else { fprintf(stderr, "Unknown option: %s\n", a.c_str()); usage(argv[0]); return 1; }
    }

    RK35llm model;
    model.SetInfo(false);
    printf("[server] loading models...\n");

    if (!model.LoadModel(vlm_model, llm_model, max_tokens, context)) {
        fprintf(stderr, "[server] failed to load models\n");
        return 1;
    }

    printf("[server] models loaded\n");

    VLM_Server server(model, host, port, model_name);
    return server.Start() ? 0 : 1;
}

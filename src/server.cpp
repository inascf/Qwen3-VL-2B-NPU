#include "server.h"
#include <chrono>
#include <sstream>
#include <iomanip>
#include <random>
#include <ctime>

using json = nlohmann::json;

//──────────────────────────────────────────────
// Constructor
//──────────────────────────────────────────────
VLM_Server::VLM_Server(RK35llm& model,
                       const std::string& host,
                       int port,
                       const std::string& model_name)
    : model_(model), host_(host), port_(port), model_name_(model_name)
{
    model_.SetChatTemplate("", "", "");
    model_.SetSilence(true);
    RegisterRoutes();
}

//──────────────────────────────────────────────
// Start / Stop
//──────────────────────────────────────────────
bool VLM_Server::Start()
{
    printf("[server] listening on %s:%d\n", host_.c_str(), port_);
    return svr_.listen(host_.c_str(), port_);
}

void VLM_Server::Stop()
{
    svr_.stop();
}

//──────────────────────────────────────────────
// Route registration
//──────────────────────────────────────────────
void VLM_Server::RegisterRoutes()
{
    svr_.Get ("/health",             [this](auto& q, auto& r){ OnHealth(q,r); });
    svr_.Get ("/props",              [this](auto& q, auto& r){ OnProps(q,r); });
    svr_.Get ("/v1/models",          [this](auto& q, auto& r){ OnModels(q,r); });
    svr_.Post("/completion",         [this](auto& q, auto& r){ OnCompletion(q,r); });
    svr_.Post("/v1/chat/completions",[this](auto& q, auto& r){ OnChat(q,r); });
    svr_.Post("/tokenize",           [this](auto& q, auto& r){ OnTokenize(q,r); });

    svr_.Options(".*", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin",  "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
        res.status = 204;
    });

    svr_.set_default_headers({{"Access-Control-Allow-Origin", "*"}});
}

//──────────────────────────────────────────────
// GET /health
//──────────────────────────────────────────────
void VLM_Server::OnHealth(const httplib::Request&, httplib::Response& res)
{
    res.set_content("{\"status\":\"ok\"}", "application/json");
}

//──────────────────────────────────────────────
// GET /props
//──────────────────────────────────────────────
void VLM_Server::OnProps(const httplib::Request&, httplib::Response& res)
{
    json body = {
        {"model_alias",   model_name_},
        {"total_slots",   1},
        {"chat_template", "qwen3"},
        {"multimodal",    true}
    };
    res.set_content(body.dump(), "application/json");
}

//──────────────────────────────────────────────
// GET /v1/models
//──────────────────────────────────────────────
void VLM_Server::OnModels(const httplib::Request&, httplib::Response& res)
{
    json body = {
        {"object", "list"},
        {"data", {{
            {"id",       model_name_},
            {"object",   "model"},
            {"created",  (int64_t)time(nullptr)},
            {"owned_by", "local"}
        }}}
    };
    res.set_content(body.dump(), "application/json");
}

//──────────────────────────────────────────────
// POST /tokenize
//──────────────────────────────────────────────
void VLM_Server::OnTokenize(const httplib::Request& req, httplib::Response& res)
{
    json body;
    try { body = json::parse(req.body); } catch(...) {
        res.status = 400;
        res.set_content("{\"error\":\"invalid json\"}", "application/json");
        return;
    }
    std::string content = body.value("content", "");
    int est = (int)(content.size() / 4) + 1;
    json out = {{"tokens", json::array()}, {"estimated_count", est}};
    res.set_content(out.dump(), "application/json");
}

//──────────────────────────────────────────────
// POST /completion
//──────────────────────────────────────────────
void VLM_Server::OnCompletion(const httplib::Request& req, httplib::Response& res)
{
    json body;
    try { body = json::parse(req.body); } catch(...) {
        res.status = 400;
        res.set_content("{\"error\":\"invalid json\"}", "application/json");
        return;
    }

    InferRequest ir;
    ir.prompt     = body.value("prompt", "");
    ir.stream     = body.value("stream", false);
    ir.max_tokens = body.value("n_predict", -1);

    if (body.contains("image_data") && body["image_data"].is_array()
        && !body["image_data"].empty())
    {
        std::string b64 = body["image_data"][0].value("data", "");
        if (!b64.empty()) {
            ir.image_data = Base64DecodeBytes(b64);
            ir.has_image  = !ir.image_data.empty();
        }
    }

    if (!ir.stream) {
        std::string text = RunBlocking(ir);
        json out = {
            {"content",          text},
            {"stop",             true},
            {"model",            model_name_},
            {"tokens_predicted", (int)(text.size() / 4)}
        };
        res.set_content(out.dump(), "application/json");
        return;
    }

    std::string rid = MakeId("cmpl-");
    res.set_chunked_content_provider("text/event-stream",
        [this, ir, rid](size_t, httplib::DataSink& sink) -> bool {
            RunStreaming(ir, rid, sink, false);
            return false;
        });
}

//──────────────────────────────────────────────
// POST /v1/chat/completions
//──────────────────────────────────────────────
void VLM_Server::OnChat(const httplib::Request& req, httplib::Response& res)
{
    json body;
    try { body = json::parse(req.body); } catch(...) {
        res.status = 400;
        res.set_content("{\"error\":\"invalid json\"}", "application/json");
        return;
    }

    if (!body.contains("messages") || !body["messages"].is_array()) {
        res.status = 400;
        res.set_content("{\"error\":\"messages required\"}", "application/json");
        return;
    }

    ParsedChat pc = ParseChatMessages(body["messages"]);

    InferRequest ir;
    ir.prompt     = pc.prompt;
    ir.has_image  = pc.has_image;
    ir.image_data = std::move(pc.image_data);
    ir.stream     = body.value("stream", false);
    ir.max_tokens = body.value("max_tokens", -1);
    ir.thinking   = pc.thinking;

    if (!ir.stream) {
        std::string text = RunBlocking(ir);
        std::string rid  = MakeId("chatcmpl-");
        json out = {
            {"id",      rid},
            {"object",  "chat.completion"},
            {"created", (int64_t)time(nullptr)},
            {"model",   model_name_},
            {"choices", {{
                {"index",         0},
                {"message",       {{"role","assistant"},{"content", text}}},
                {"finish_reason", "stop"}
            }}},
            {"usage", {
                {"prompt_tokens",     0},
                {"completion_tokens", (int)(text.size()/4)},
                {"total_tokens",      (int)(text.size()/4)}
            }}
        };
        res.set_content(out.dump(), "application/json");
        return;
    }

    std::string rid = MakeId("chatcmpl-");
    res.set_chunked_content_provider("text/event-stream",
        [this, ir, rid](size_t, httplib::DataSink& sink) -> bool {
            RunStreaming(ir, rid, sink, true);
            return false;
        });
}

//──────────────────────────────────────────────
// Core: blocking inference
//──────────────────────────────────────────────
std::string VLM_Server::RunBlocking(const InferRequest& ir)
{
    std::lock_guard<std::mutex> lock(model_.GetInferenceMutex());

    model_.ClearHistory();
    model_.SetTokenCallback(nullptr);

    if (ir.has_image && !ir.image_data.empty())
        model_.LoadImageFromMemory(ir.image_data.data(), ir.image_data.size());

    std::string q = ir.prompt;
    if (ir.has_image && q.find("<image>") == std::string::npos)
        q += " <image>";

    return model_.Ask(q);
}

//──────────────────────────────────────────────
// Core: streaming inference (SSE)
//──────────────────────────────────────────────
void VLM_Server::RunStreaming(const InferRequest& ir,
                              const std::string& req_id,
                              httplib::DataSink& sink,
                              bool openai_format)
{
    struct State {
        std::mutex              mu;
        std::condition_variable cv;
        std::deque<std::string> tokens;
        bool                    done  = false;
    };
    auto st = std::make_shared<State>();

    std::thread infer_thread([this, &ir, st]() {
        std::lock_guard<std::mutex> lock(model_.GetInferenceMutex());

        model_.ClearHistory();
        model_.SetTokenCallback([st](const std::string& tok) {
            std::lock_guard<std::mutex> lg(st->mu);
            st->tokens.push_back(tok);
            st->cv.notify_one();
        });

        if (ir.has_image && !ir.image_data.empty())
            model_.LoadImageFromMemory(ir.image_data.data(), ir.image_data.size());

        std::string q = ir.prompt;
        if (ir.has_image && q.find("<image>") == std::string::npos)
            q += " <image>";

        model_.Ask(q);
        model_.SetTokenCallback(nullptr);

        std::lock_guard<std::mutex> lg(st->mu);
        st->done = true;
        st->cv.notify_all();
    });

    int64_t created = (int64_t)time(nullptr);

    while (true) {
        std::unique_lock<std::mutex> lk(st->mu);
        st->cv.wait(lk, [&st]{ return !st->tokens.empty() || st->done; });

        while (!st->tokens.empty()) {
            std::string tok = st->tokens.front();
            st->tokens.pop_front();
            lk.unlock();

            std::string data;
            if (openai_format) {
                json chunk = {
                    {"id",      req_id},
                    {"object",  "chat.completion.chunk"},
                    {"created", created},
                    {"model",   model_name_},
                    {"choices", {{
                        {"index",         0},
                        {"delta",         {{"content", tok}}},
                        {"finish_reason", nullptr}
                    }}}
                };
                data = "data: " + chunk.dump() + "\n\n";
            } else {
                json chunk = {{"content", tok}, {"stop", false}};
                data = "data: " + chunk.dump() + "\n\n";
            }

            if (!sink.write(data.c_str(), data.size())) {
                infer_thread.detach();
                return;
            }

            lk.lock();
        }

        if (st->done && st->tokens.empty()) break;
    }

    // Final stop chunk
    if (openai_format) {
        json last = {
            {"id",      req_id},
            {"object",  "chat.completion.chunk"},
            {"created", created},
            {"model",   model_name_},
            {"choices", {{
                {"index",         0},
                {"delta",         json::object()},
                {"finish_reason", "stop"}
            }}}
        };
        std::string s = "data: " + last.dump() + "\n\n";
        sink.write(s.c_str(), s.size());
    } else {
        std::string_view s = "data: {\"content\":\"\",\"stop\":true}\n\n";
        sink.write(s.data(), s.size());
    }

    std::string_view done_msg = "data: [DONE]\n\n";
    sink.write(done_msg.data(), done_msg.size());

    infer_thread.join();
}

//──────────────────────────────────────────────
// Parse OpenAI messages → Qwen3 formatted prompt
//──────────────────────────────────────────────
VLM_Server::ParsedChat VLM_Server::ParseChatMessages(const json& messages)
{
    ParsedChat out;
    std::string prompt;
    bool has_system = false;

    for (auto& msg : messages) {
        std::string role    = msg.value("role", "user");
        std::string content;

        if (msg["content"].is_string()) {
            content = msg["content"].get<std::string>();
        } else if (msg["content"].is_array()) {
            for (auto& part : msg["content"]) {
                std::string type = part.value("type", "");
                if (type == "text") {
                    content += part.value("text", "");
                } else if (type == "image_url") {
                    std::string url = part["image_url"].value("url", "");
                    if (!url.empty()) {
                        out.image_data = Base64DecodeBytes(url);
                        out.has_image  = !out.image_data.empty();
                        content       += "<image>";
                    }
                }
            }
        }

        if (role == "system") {
            has_system = true;
            prompt += "<|im_start|>system\n" + content + "<|im_end|>\n";
        } else if (role == "user") {
            prompt += "<|im_start|>user\n" + content + "<|im_end|>\n";
        } else if (role == "assistant") {
            prompt += "<|im_start|>assistant\n" + content + "<|im_end|>\n";
        }
    }

    if (!has_system)
        prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n" + prompt;

    prompt += "<|im_start|>assistant\n";
    out.prompt = prompt;
    return out;
}

//──────────────────────────────────────────────
// Base64 decode → raw bytes
//──────────────────────────────────────────────
std::vector<uint8_t> VLM_Server::Base64DecodeBytes(const std::string& b64_or_url)
{
    // Strip "data:image/xxx;base64," prefix if present
    std::string data = b64_or_url;
    size_t comma = data.find(',');
    if (comma != std::string::npos)
        data = data.substr(comma + 1);

    // Remove whitespace
    data.erase(std::remove_if(data.begin(), data.end(), ::isspace), data.end());

    static const int8_t T[256] = {
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,62,-1,-1,-1,63,
        52,53,54,55,56,57,58,59,60,61,-1,-1,-1,-1,-1,-1,
        -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,
        15,16,17,18,19,20,21,22,23,24,25,-1,-1,-1,-1,-1,
        -1,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
        41,42,43,44,45,46,47,48,49,50,51,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1
    };

    std::vector<uint8_t> out;
    out.reserve(data.size() * 3 / 4);

    int val = 0, bits = -8;
    for (unsigned char c : data) {
        if (T[c] == -1) break;
        val = (val << 6) | T[c];
        bits += 6;
        if (bits >= 0) {
            out.push_back((val >> bits) & 0xFF);
            bits -= 8;
        }
    }
    return out;
}

//──────────────────────────────────────────────
// Unique ID
//──────────────────────────────────────────────
std::string VLM_Server::MakeId(const char* prefix)
{
    static std::mt19937_64 rng(
        std::chrono::steady_clock::now().time_since_epoch().count());
    std::ostringstream ss;
    ss << prefix << std::hex << rng();
    return ss.str();
}

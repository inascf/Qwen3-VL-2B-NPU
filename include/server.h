#ifndef VLM_SERVER_H
#define VLM_SERVER_H

#include "RK35llm.h"
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <string>
#include <deque>
#include <atomic>
#include <thread>

class VLM_Server {
public:
    VLM_Server(RK35llm& model,
               const std::string& host = "0.0.0.0",
               int port = 8080,
               const std::string& model_name = "qwen3-vl-2b");

    bool Start();   // blocking
    void Stop();

private:
    RK35llm&          model_;
    std::string       host_;
    int               port_;
    std::string       model_name_;
    httplib::Server   svr_;

    void RegisterRoutes();

    // Route handlers
    void OnHealth      (const httplib::Request&, httplib::Response&);
    void OnProps       (const httplib::Request&, httplib::Response&);
    void OnModels      (const httplib::Request&, httplib::Response&);
    void OnCompletion  (const httplib::Request&, httplib::Response&);
    void OnChat        (const httplib::Request&, httplib::Response&);
    void OnTokenize    (const httplib::Request&, httplib::Response&);

    // Helpers
    struct InferRequest {
        std::string           prompt;
        bool                  has_image = false;
        cv::Mat               image;
        bool                  stream    = false;
        int                   max_tokens = -1;
        bool                  thinking  = false;
        int                   keep_hist = 0;
    };

    // Blocking, non-streaming inference (returns full text)
    std::string RunBlocking(const InferRequest& req);

    // Streaming inference — writes SSE events into sink until done
    void RunStreaming(const InferRequest& req,
                      const std::string& req_id,
                      httplib::DataSink& sink,
                      bool openai_format);

    // Build a fully-formatted Qwen3-VL chat prompt from OpenAI messages array
    // Returns {prompt_text, has_image, decoded_image}
    struct ParsedChat {
        std::string prompt;
        bool        has_image = false;
        cv::Mat     image;
        bool        thinking  = false;
    };
    ParsedChat ParseChatMessages(const nlohmann::json& messages);

    // Base64 helpers
    static std::string Base64Decode(const std::string& in);
    static cv::Mat     DecodeB64Image(const std::string& b64_or_url);

    // Unique ID generator
    static std::string MakeId(const char* prefix = "cmpl-");
};

#endif // VLM_SERVER_H

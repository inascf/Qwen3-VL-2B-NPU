#ifndef VLM_SERVER_H
#define VLM_SERVER_H

#include "RK35llm.h"
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
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

    struct InferRequest {
        std::string          prompt;
        bool                 has_image  = false;
        std::vector<uint8_t> image_data; // raw JPEG/PNG bytes (not decoded)
        bool                 stream     = false;
        int                  max_tokens = -1;
        bool                 thinking   = false;
        int                  keep_hist  = 0;
    };

    std::string RunBlocking (const InferRequest& req);
    void        RunStreaming(const InferRequest& req,
                             const std::string& req_id,
                             httplib::DataSink& sink,
                             bool openai_format);

    struct ParsedChat {
        std::string          prompt;
        bool                 has_image = false;
        std::vector<uint8_t> image_data;
        bool                 thinking  = false;
    };
    ParsedChat ParseChatMessages(const nlohmann::json& messages);

    // Base64 decode → raw bytes
    static std::vector<uint8_t> Base64DecodeBytes(const std::string& in);

    static std::string MakeId(const char* prefix = "cmpl-");
};

#endif // VLM_SERVER_H

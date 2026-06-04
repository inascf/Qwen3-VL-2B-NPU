#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

#include "RK35llm.h"
#include <algorithm>

#define HISTORY true
//----------------------------------------------------------------------------------------
RK35llm::RK35llm(void)
{
    Info    = false;
    Silence = false;
    ImgVec    = nullptr;
    llmHandle = nullptr;
    responseReady_ = false;
    memset(&rknn_app_ctx, 0, sizeof(RKLLM_app_context_t));
    memset(&rkllm_input, 0, sizeof(RKLLMInput));
    memset(&rkllm_infer_params, 0, sizeof(RKLLMInferParam));

    param = rkllm_createDefaultParam();
    param.top_k = 1;
    param.skip_special_token = true;
    param.extend_param.base_domain_id = 1;
    param.img_start   = "<|vision_start|>";
    param.img_end     = "<|vision_end|>";
    param.img_content = "<|image_pad|>";
}
//----------------------------------------------------------------------------------------
RK35llm::~RK35llm(void)
{
    if(rknn_app_ctx.input_attrs != nullptr){
        free(rknn_app_ctx.input_attrs);
        rknn_app_ctx.input_attrs = nullptr;
    }
    if(rknn_app_ctx.output_attrs != nullptr){
        free(rknn_app_ctx.output_attrs);
        rknn_app_ctx.output_attrs = nullptr;
    }
    if(rknn_app_ctx.rknn_ctx != 0){
        rknn_destroy(rknn_app_ctx.rknn_ctx);
        rknn_app_ctx.rknn_ctx = 0;
    }
    if(ImgVec!=nullptr){
        delete[] ImgVec;
        ImgVec = nullptr;
    }

    if(llmHandle!=nullptr) rkllm_destroy(llmHandle);
}
//----------------------------------------------------------------------------------------
void RK35llm::DumpTensorAttr(rknn_tensor_attr* attr)
{
    if(!Info) return;
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
            "zp=%d, scale=%f\n",
            attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
            attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
            get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}
//----------------------------------------------------------------------------------------
int RK35llm::StaticCallback(RKLLMResult* result, void* userdata, LLMCallState state)
{
    if (!userdata) return -1;
    RK35llm* self = static_cast<RK35llm*>(userdata);
    return self->InstanceCallback(result, state);
}
//----------------------------------------------------------------------------------------
int RK35llm::InstanceCallback(RKLLMResult *result, LLMCallState state)
{
    if (state == RKLLM_RUN_FINISH)
    {
        if(!Silence) printf("\n");
        responseReady_ = true;
        responseCv_.notify_all();
    }
    else if (state == RKLLM_RUN_ERROR)
    {
        if(!Silence) printf("[Error during inference]\n");
        responseBuffer_ += "[Error during inference]";
        responseReady_ = true;
        responseCv_.notify_all();
    }
    else if (state == RKLLM_RUN_NORMAL)
    {
        if(!Silence) printf("%s", result->text);
        if (result && result->text) {
            responseBuffer_ += result->text;
            if (tokenCallback_) tokenCallback_(result->text);
        }
    }
    return 0;
}
//----------------------------------------------------------------------------------------
// Pad image to square, then resize to model input dimensions.
// rgb: input RGB pixels (3 bytes/pixel), w x h
bool RK35llm::ProcessRawImage(const uint8_t* rgb, int w, int h)
{
    int tgt_w    = rknn_app_ctx.model_width;
    int tgt_h    = rknn_app_ctx.model_height;
    int channels = 3;

    // Expand to square with gray padding (value 127, matching original 127.5 → uint8)
    int sq = std::max(w, h);
    std::vector<uint8_t> square(sq * sq * channels, 127);
    int x_off = (sq - w) / 2;
    int y_off = (sq - h) / 2;
    for (int row = 0; row < h; row++) {
        memcpy(square.data() + ((row + y_off) * sq + x_off) * channels,
               rgb  + row * w * channels,
               w * channels);
    }

    // Bilinear resize to model input size (same algorithm as cv::INTER_LINEAR)
    resized_img_data_.resize(tgt_w * tgt_h * channels);
    stbir_resize_uint8_linear(
        square.data(), sq,    sq,    0,
        resized_img_data_.data(), tgt_w, tgt_h, 0,
        STBIR_RGB);

    return true;
}
//----------------------------------------------------------------------------------------
int RK35llm::InitImgEnc(const char* model_path)
{
    int ret;
    rknn_context ctx = 0;

    ret = rknn_init(&ctx, (void*)model_path, 0, 0, NULL);
    if (ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    int core_num=3;
    if(Info) printf("\nused NPU cores %d\n", core_num);
    if (core_num == 2) {
        ret = rknn_set_core_mask(ctx, RKNN_NPU_CORE_0_1);
    } else if (core_num == 3) {
        ret = rknn_set_core_mask(ctx, RKNN_NPU_CORE_0_1_2);
    } else {
        ret = rknn_set_core_mask(ctx, RKNN_NPU_CORE_AUTO);
    }
    if (ret < 0) {
        printf("rknn_set_core_mask fail! ret=%d\n", ret);
        return -1;
    }

    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    if(Info) printf("\nmodel input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    if(Info) printf("\nInput tensors:\n");
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (uint32_t i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        DumpTensorAttr(&(input_attrs[i]));
    }

    if(Info) printf("\nOutput tensors:\n");
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (uint32_t i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        DumpTensorAttr(&(output_attrs[i]));
    }

    for (int i = 0; i < 4; i++) {
        if (output_attrs[0].dims[i] > 1) {
            rknn_app_ctx.model_image_token = output_attrs[0].dims[i];
            rknn_app_ctx.model_embed_size = output_attrs[0].dims[i + 1];
            break;
        }
    }
    rknn_app_ctx.rknn_ctx = ctx;
    rknn_app_ctx.io_num = io_num;
    rknn_app_ctx.input_attrs = (rknn_tensor_attr*)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    memcpy(rknn_app_ctx.input_attrs, input_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
    rknn_app_ctx.output_attrs = (rknn_tensor_attr*)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    memcpy(rknn_app_ctx.output_attrs, output_attrs, io_num.n_output * sizeof(rknn_tensor_attr));

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        rknn_app_ctx.model_channel = input_attrs[0].dims[1];
        rknn_app_ctx.model_height  = input_attrs[0].dims[2];
        rknn_app_ctx.model_width   = input_attrs[0].dims[3];
    } else {
        rknn_app_ctx.model_height  = input_attrs[0].dims[1];
        rknn_app_ctx.model_width   = input_attrs[0].dims[2];
        rknn_app_ctx.model_channel = input_attrs[0].dims[3];
    }
    if(Info) printf("\nModel input height=%d, width=%d, channel=%d\n\n",
        rknn_app_ctx.model_height, rknn_app_ctx.model_width, rknn_app_ctx.model_channel);

    if(ImgVec!=nullptr) delete[] ImgVec;
    ImgVec = new float[rknn_app_ctx.model_image_token * rknn_app_ctx.model_embed_size * rknn_app_ctx.io_num.n_output];

    return 0;
}
//----------------------------------------------------------------------------------------
int RK35llm::RunImgEnc(void)
{
    int ret = 0;
    rknn_input inputs[1];
    rknn_output outputs[rknn_app_ctx.io_num.n_output];

    memset(inputs, 0, sizeof(inputs));
    memset(outputs, 0, sizeof(outputs));

    inputs[0].index = 0;
    inputs[0].type  = RKNN_TENSOR_UINT8;
    inputs[0].fmt   = RKNN_TENSOR_NHWC;
    inputs[0].size  = rknn_app_ctx.model_width * rknn_app_ctx.model_height * rknn_app_ctx.model_channel;
    inputs[0].buf   = resized_img_data_.data();

    ret = rknn_inputs_set(rknn_app_ctx.rknn_ctx, 1, inputs);
    if (ret < 0) {
        printf("rknn_input_set fail! ret=%d\n", ret);
        return -1;
    }

    ret = rknn_run(rknn_app_ctx.rknn_ctx, nullptr);
    if (ret < 0) {
        printf("rknn_run fail! ret=%d\n", ret);
        return -1;
    }

    for (uint32_t j=0; j<rknn_app_ctx.io_num.n_output; j++) {
        outputs[j].want_float = 1;
    }
    ret = rknn_outputs_get(rknn_app_ctx.rknn_ctx, rknn_app_ctx.io_num.n_output, outputs, NULL);
    if (ret < 0) {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        return ret;
    }

    if(rknn_app_ctx.io_num.n_output == 1) memcpy(ImgVec, outputs[0].buf, outputs[0].size);
    else {
        for(int i=0; i<rknn_app_ctx.model_image_token; i++){
            for (uint32_t j = 0; j < rknn_app_ctx.io_num.n_output; j++) {
                memcpy(ImgVec + i * rknn_app_ctx.io_num.n_output * rknn_app_ctx.model_embed_size + j * rknn_app_ctx.model_embed_size,
                      (float*)(outputs[j].buf) + i * rknn_app_ctx.model_embed_size, sizeof(float) * rknn_app_ctx.model_embed_size);
            }
        }
    }

    rknn_outputs_release(rknn_app_ctx.rknn_ctx, 1, outputs);
    return ret;
}
//----------------------------------------------------------------------------------------
void RK35llm::SetInfo(bool _Info)
{
    Info = _Info;
}
//----------------------------------------------------------------------------------------
void RK35llm::SetHistory(bool _History)
{
    rkllm_infer_params.mode = RKLLM_INFER_GENERATE;
    rkllm_infer_params.keep_history = 0;
    if(_History){
        rkllm_infer_params.keep_history = 1;
        rkllm_set_chat_template(llmHandle, "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n", "<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n");
    }
}
//----------------------------------------------------------------------------------------
void RK35llm::SetSilence(bool _Silence)
{
    Silence = _Silence;
}
//----------------------------------------------------------------------------------------
void RK35llm::SetTokenCallback(std::function<void(const std::string&)> cb)
{
    std::lock_guard<std::mutex> lk(responseMutex_);
    tokenCallback_ = std::move(cb);
}
//----------------------------------------------------------------------------------------
void RK35llm::SetChatTemplate(const char* system, const char* prefix, const char* postfix)
{
    if (llmHandle) rkllm_set_chat_template(llmHandle, system, prefix, postfix);
}
//----------------------------------------------------------------------------------------
void RK35llm::ClearHistory()
{
    if (llmHandle) rkllm_clear_kv_cache(llmHandle, 0, nullptr, nullptr);
}
//----------------------------------------------------------------------------------------
bool RK35llm::LoadModel(const std::string& VLMmodel, const std::string& LLMmodel, int32_t NewTokens, int32_t ContextLength)
{
    param.model_path = LLMmodel.c_str();
    param.max_new_tokens = NewTokens;
    param.max_context_len = ContextLength;

    int ret = rkllm_init(&llmHandle, &param, RK35llm::StaticCallback);
    if(ret != 0) return false;
    else{
        if(Info) printf("rkllm init success\n");
    }

    #if HISTORY
        rkllm_infer_params.keep_history = 1;
        int setret = rkllm_set_chat_template(llmHandle,
             "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
             "<|im_start|>user\n",
             "<|im_end|>\n<|im_start|>assistant\n");
        if (setret != 0 && Info) {
            printf("rkllm_set_chat_template returned %d\n", setret);
        }
    #else
        rkllm_infer_params.keep_history = 0;
    #endif

    ret = InitImgEnc(VLMmodel.c_str());
    if(ret != 0) return false;

    return true;
}
//----------------------------------------------------------------------------------------
bool RK35llm::LoadImage(const std::string& filepath)
{
    int w, h, c;
    // stb_image reads as RGB natively (no BGR swap needed unlike OpenCV)
    uint8_t* pixels = stbi_load(filepath.c_str(), &w, &h, &c, 3);
    if (!pixels) {
        printf("stbi_load failed: %s\n", filepath.c_str());
        return false;
    }

    bool ok = ProcessRawImage(pixels, w, h);
    stbi_image_free(pixels);

    if (!ok) return false;

    int ret = RunImgEnc();
    if (ret != 0) {
        printf("run_imgenc fail! ret=%d\n", ret);
        return false;
    }
    return true;
}
//----------------------------------------------------------------------------------------
bool RK35llm::LoadImageFromMemory(const void* data, size_t len)
{
    int w, h, c;
    uint8_t* pixels = stbi_load_from_memory(
        static_cast<const stbi_uc*>(data), (int)len, &w, &h, &c, 3);
    if (!pixels) {
        printf("stbi_load_from_memory failed\n");
        return false;
    }

    bool ok = ProcessRawImage(pixels, w, h);
    stbi_image_free(pixels);

    if (!ok) return false;

    int ret = RunImgEnc();
    if (ret != 0) {
        printf("run_imgenc fail! ret=%d\n", ret);
        return false;
    }
    return true;
}
//----------------------------------------------------------------------------------------
std::string RK35llm::Ask(const std::string& Question)
{
    std::string Str="";

    if (!llmHandle) return Str;

    {
        std::lock_guard<std::mutex> lk(responseMutex_);
        responseBuffer_.clear();
        responseReady_ = false;
    }

    if (Question == "clear")
    {
        rkllm_clear_kv_cache(llmHandle, 1, nullptr, nullptr);
        return Str;
    }
    if (Question.find("<image>") == std::string::npos)
    {
        rkllm_input.input_type = RKLLM_INPUT_PROMPT;
        rkllm_input.role = "user";
        rkllm_input.prompt_input = (char*)Question.c_str();
    } else {
        rkllm_input.input_type = RKLLM_INPUT_MULTIMODAL;
        rkllm_input.role = "user";
        rkllm_input.multimodal_input.prompt = (char*)Question.c_str();
        rkllm_input.multimodal_input.image_embed = ImgVec;
        rkllm_input.multimodal_input.n_image_tokens = rknn_app_ctx.model_image_token;
        rkllm_input.multimodal_input.n_image = 1;
        rkllm_input.multimodal_input.image_height = rknn_app_ctx.model_height;
        rkllm_input.multimodal_input.image_width = rknn_app_ctx.model_width;
    }
    if(!Silence) printf("Answer: ");

    int ret = rkllm_run(llmHandle, &rkllm_input, &rkllm_infer_params, this);
    if (ret != 0) {
        std::cerr << "rkllm_run returned " << ret << "\n";
    }

    std::unique_lock<std::mutex> lk(responseMutex_);
    responseCv_.wait(lk, [this]{ return responseReady_; });

    return responseBuffer_;
}
//----------------------------------------------------------------------------------------

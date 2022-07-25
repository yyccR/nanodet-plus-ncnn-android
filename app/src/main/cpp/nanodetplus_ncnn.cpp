//
// Created by yang on 2022/7/21.
//

// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "net.h"

#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif
#include "jni.h"
#include <android/asset_manager_jni.h>
#include <stdlib.h>
#include <float.h>
#include <stdio.h>
#include <vector>

static ncnn::Net nanodetplus;
static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

typedef struct BoxInfo {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
} BoxInfo;

typedef struct HeadInfo_
{
    std::string cls_layer;
    std::string dis_layer;
    int stride;
} HeadInfo;

typedef struct CenterPrior_
{
    int x;
    int y;
    int stride;
} CenterPrior;

inline float fast_exp(float x)
{
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

inline float sigmoid(float x)
{
    return 1.0f / (1.0f + fast_exp(-x));
}

template<typename _Tp>
int activation_function_softmax(const _Tp* src, _Tp* dst, int length)
{
    const _Tp alpha = *std::max_element(src, src + length);
    _Tp denominator{ 0 };

    for (int i = 0; i < length; ++i) {
        dst[i] = fast_exp(src[i] - alpha);
        denominator += dst[i];
    }

    for (int i = 0; i < length; ++i) {
        dst[i] /= denominator;
    }

    return 0;
}

inline std::string jString2String(JNIEnv *env, jstring jstring) {
    const char *js = NULL;
    js = env->GetStringUTFChars(jstring, 0);
    std::string s(js);
    env->ReleaseStringUTFChars(jstring, js);
    return s;
}

void generate_grid_center_priors(const int input_height, const int input_width, std::vector<int>& strides, std::vector<CenterPrior>& center_priors)
{
    for (int i = 0; i < (int)strides.size(); i++)
    {
        int stride = strides[i];
        int feat_w = ceil((float)input_width / stride);
        int feat_h = ceil((float)input_height / stride);
        for (int y = 0; y < feat_h; y++)
        {
            for (int x = 0; x < feat_w; x++)
            {
                CenterPrior ct;
                ct.x = x;
                ct.y = y;
                ct.stride = stride;
                center_priors.push_back(ct);
            }
        }
    }
}

BoxInfo disPred2Bbox(const float*& dfl_det, int label, float score, int x, int y, int stride, float width_ratio, float height_ratio)
{
    float input_size = 416.0;
    float ct_x = x * stride;
    float ct_y = y * stride;
    std::vector<float> dis_pred;
    dis_pred.resize(4);
    for (int i = 0; i < 4; i++)
    {
        float dis = 0;
        float* dis_after_sm = new float[7 + 1];
        activation_function_softmax(dfl_det + i * (7 + 1), dis_after_sm, 7 + 1);
        for (int j = 0; j < (7 + 1); j++)
        {
            dis += j * dis_after_sm[j];
        }
        dis *= stride;
        //std::cout << "dis:" << dis << std::endl;
        dis_pred[i] = dis;
        delete[] dis_after_sm;
    }
    float xmin = (std::max)(ct_x - dis_pred[0], .0f) * width_ratio;
    float ymin = (std::max)(ct_y - dis_pred[1], .0f) * height_ratio;
    float xmax = (std::min)(ct_x + dis_pred[2], input_size) * width_ratio;
    float ymax = (std::min)(ct_y + dis_pred[3], input_size) * height_ratio;

    //std::cout << xmin << "," << ymin << "," << xmax << "," << xmax << "," << std::endl;
    return BoxInfo { xmin, ymin, xmax, ymax, score, label };
}

void decode_infer(ncnn::Mat& feats, int num_class, std::vector<CenterPrior>& center_priors, float threshold, std::vector<std::vector<BoxInfo>>& results, float width_ratio, float height_ratio)
{
    const int num_points = center_priors.size();
    //printf("num_points:%d\n", num_points);

    //cv::Mat debug_heatmap = cv::Mat(feature_h, feature_w, CV_8UC3);
    for (int idx = 0; idx < num_points; idx++)
    {
        const int ct_x = center_priors[idx].x;
        const int ct_y = center_priors[idx].y;
        const int stride = center_priors[idx].stride;

        const float* scores = feats.row(idx);
        float score = 0;
        int cur_label = 0;
        for (int label = 0; label < num_class; label++)
        {
            if (scores[label] > score)
            {
                score = scores[label];
                cur_label = label;
            }
        }
        if (score > threshold)
        {
            //std::cout << "label:" << cur_label << " score:" << score << std::endl;
            const float* bbox_pred = feats.row(idx) + num_class;
            results[cur_label].push_back(disPred2Bbox(bbox_pred, cur_label, score, ct_x, ct_y, stride, width_ratio, height_ratio));
            //debug_heatmap.at<cv::Vec3b>(row, col)[0] = 255;
            //cv::imshow("debug", debug_heatmap);
        }

    }
}


void nms(std::vector<BoxInfo>& input_boxes, float NMS_THRESH)
{
    std::sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
    std::vector<float> vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
                   * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        for (int j = i + 1; j < int(input_boxes.size());) {
            float xx1 = (std::max)(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = (std::max)(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = (std::min)(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = (std::min)(input_boxes[i].y2, input_boxes[j].y2);
            float w = (std::max)(float(0), xx2 - xx1 + 1);
            float h = (std::max)(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= NMS_THRESH) {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            }
            else {
                j++;
            }
        }
    }
}




extern "C" {

// FIXME DeleteGlobalRef is missing for objCls
static jclass objCls = NULL;
static jmethodID constructortorId;
static jfieldID xId;
static jfieldID yId;
static jfieldID wId;
static jfieldID hId;
static jfieldID labelId;
static jfieldID probId;


extern "C"
JNIEXPORT jobjectArray JNICALL
Java_com_example_nanodet_1plus_1ncnn_detector_NanodetplusNcnnDetector_detect(JNIEnv *env,
                                                                             jobject thiz,
                                                                             jobject bitmap,
                                                                             jboolean use_gpu,
                                                                             jint num_classes) {

    nanodetplus.opt.use_vulkan_compute = true;
    // nanodet.opt.use_bf16_storage = true;

    // original pretrained model from https://github.com/RangiLyu/nanodet
    // the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
    //     nanodet.load_param("nanodet-plus-m_320.torchscript.ncnn.param");
    //     nanodet.load_model("nanodet-plus-m_320.torchscript.ncnn.bin");
//    if (nanodetplus.load_param("nanodet-plus-m_416.torchscript.ncnn.param"))
//        exit(-1);
//    if (nanodetplus.load_model("nanodet-plus-m_416.torchscript.ncnn.bin"))
//        exit(-1);

//    int width = bgr.cols;
//    int height = bgr.rows;

    //     const int target_size = 320;
    const int target_size = 416;
    std::vector<int> strides = { 8, 16, 32, 64 };
    const float score_threshold = 0.4f;
    const float nms_threshold = 0.5f;

    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, bitmap, &info);
    const int width = info.width;
    const int height = info.height;
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
        return NULL;
    float width_ratio = (float) width / (float) target_size;
    float height_ratio = (float) height/ (float) target_size;


    // pad to multiple of 32
//    int w = width;
//    int h = height;
//    float scale = 1.f;
//    if (w > h) {
//        scale = (float) target_size / w;
//        w = target_size;
//        h = h * scale;
//    } else {
//        scale = (float) target_size / h;
//        h = target_size;
//        w = w * scale;
//    }

//    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, width, height, w,
//                                                 h);
//    ncnn::Mat in = ncnn::Mat::from_android_bitmap_resize(env, bitmap, ncnn::Mat::PIXEL_BGR, w, h);
    ncnn::Mat in = ncnn::Mat::from_android_bitmap_resize(env, bitmap, ncnn::Mat::PIXEL_BGR, target_size, target_size);

    // pad to target_size rectangle
//    int wpad = (w + 31) / 32 * 32 - w;
//    int hpad = (h + 31) / 32 * 32 - h;
//    ncnn::Mat in_pad;
//    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2,
//                           ncnn::BORDER_CONSTANT, 0.f);

//    const float mean_vals[3] = {103.53f, 116.28f, 123.675f};
//    const float norm_vals[3] = {0.017429f, 0.017507f, 0.017125f};
//    in_pad.substract_mean_normalize(mean_vals, norm_vals);

    const float mean_vals[3] = { 103.53f, 116.28f, 123.675f };
    const float norm_vals[3] = { 0.017429f, 0.017507f, 0.017125f };
//    in_pad.substract_mean_normalize(mean_vals, norm_vals);
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = nanodetplus.create_extractor();
//    ex.input("data", in_pad);
    ex.input("data", in);

    std::vector<std::vector<BoxInfo>> results;
    results.resize(num_classes);

    ncnn::Mat out;
    ex.extract("output", out);
    // printf("%d %d %d \n", out.w, out.h, out.c);

    // generate center priors in format of (x, y, stride)
    std::vector<CenterPrior> center_priors;
    generate_grid_center_priors(target_size, target_size, strides, center_priors);

//    decode_infer(out, center_priors, score_threshold, results);
    decode_infer(out, num_classes, center_priors, score_threshold, results, width_ratio, height_ratio);

    std::vector<BoxInfo> dets;
    for (int i = 0; i < (int)results.size(); i++)
    {
        nms(results[i], nms_threshold);

        for (auto box : results[i])
        {
            dets.push_back(box);
        }
    }
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_nanodet_1plus_1ncnn_detector_NanodetplusNcnnDetector_init(JNIEnv *env,
                                                                           jobject thiz,
                                                                           jobject asset_manager,
                                                                           jstring model_name) {
    ncnn::Option opt;
    opt.lightmode = true;
    opt.num_threads = 4;
    opt.blob_allocator = &g_blob_pool_allocator;
    opt.workspace_allocator = &g_workspace_pool_allocator;
    opt.use_packing_layout = true;

    // use vulkan compute
    if (ncnn::get_gpu_count() != 0)
        opt.use_vulkan_compute = true;

    AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);
    nanodetplus.opt = opt;

    // init param
    {
        std::string param = jString2String(env, model_name) + ".param";
        int ret = nanodetplus.load_param(mgr, param.c_str());
        __android_log_print(ANDROID_LOG_INFO,"ncnn:", "%s", ("load_param: " + param + " success").c_str());
        if (ret != 0) {
            __android_log_print(ANDROID_LOG_DEBUG, "ncnn:", "load_param failed");
            return JNI_FALSE;
        }
    }

    // init bin
    {
        std::string bin = jString2String(env, model_name) + ".bin";
        int ret = nanodetplus.load_model(mgr, bin.c_str());
        __android_log_print(ANDROID_LOG_INFO, "ncnn:", "%s", ("load_bin: " + bin + " success").c_str());
        if (ret != 0) {
            __android_log_print(ANDROID_LOG_DEBUG, "ncnn:", "load_bin failed");
            return JNI_FALSE;
        }
    }

    // init jni glue
    jclass localObjCls = env->FindClass(
            "com/example/nanodet_plus_ncnn/detector/NanodetplusNcnnDetector$Obj");
    objCls = reinterpret_cast<jclass>(env->NewGlobalRef(localObjCls));

    constructortorId = env->GetMethodID(objCls, "<init>", "(Lcom/example/nanodet_plus_ncnn/detector/NanodetplusNcnnDetector;)V");

    xId = env->GetFieldID(objCls, "x", "F");
    yId = env->GetFieldID(objCls, "y", "F");
    wId = env->GetFieldID(objCls, "w", "F");
    hId = env->GetFieldID(objCls, "h", "F");
    labelId = env->GetFieldID(objCls, "label", "I");
    probId = env->GetFieldID(objCls, "prob", "F");

    return JNI_TRUE;
}

}


#ifndef MYFACE_TRT_MYFACE_H
#define MYFACE_TRT_MYFACE_H

#include "assets.h"
#include <opencv2/opencv.hpp>

class MyFace 
{
    struct FaceBox 
	{
        float x;
        float y;
        float w;
        float h;
    };

    struct FaceRes 
	{
        float confidence;
        FaceBox face_box;
        std::vector<cv::Point2f> keypoints;
        bool has_mask = false;
    };

public:
    explicit MyFace();
    ~MyFace();
    void LoadModel(std::string onnx_path);
    std::vector<Bbox> RunModel(cv::Mat& img);
	cv::Mat prepareImage(cv::Mat& input_img);

private:
    void GenerateAnchors();   
    std::vector<FaceRes> postProcess(cv::Mat &vec_Mat, cv::Mat &result_matrix);
    void NmsDetect(std::vector<FaceRes>& detections);
    static float IOUCalculate(const FaceBox& det_a, const FaceBox& det_b);

    int BATCH_SIZE;
    int INPUT_CHANNEL;
    int IMAGE_WIDTH;
    int IMAGE_HEIGHT;
    float obj_threshold;
    float nms_threshold;
    
    cv::Mat refer_matrix;
    int anchor_num = 6;    // 1��ê��λ  ����ê��
    int bbox_head = 4;     // ������     ��Ӧ�� Ԥ��ֵ����(4�� = ���ĵ�����ƫ���� + ���)
    int landmark_head = 6; // �沿��ǵ� ��Ӧ�� Ԥ��ֵ����(6�� = 3������)
    std::vector<int> feature_sizes;
    std::vector<int> feature_steps;
    std::vector<int> feature_maps;
    std::vector<std::vector<int>> anchor_sizes;
    int sum_of_feature;
    
    cv::dnn::Net model;
};

#endif //MYFACE_TRT_MYFACE_H


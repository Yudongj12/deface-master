#include "MyFace.h"
#include <numeric>

MyFace::MyFace() 
{
    BATCH_SIZE = 1;              // (源代码，禁用)Batch Size 输入图像数量，默认 1
    INPUT_CHANNEL = 3;           // 输入图像 通道数，默认 3通道(RGB)
    IMAGE_WIDTH = 640;           // 输入图像 宽度 640(强制，对应 ONNX模型文件)
    IMAGE_HEIGHT = 640;          // 输入图像 高度 640(强制，对应 ONNX模型文件)
    obj_threshold = 0.5;         // 人脸得分/置信度阈值，默认 0.5
    nms_threshold = 0.3;         // NMS阈值，默认 0.45    
    feature_steps = {16};        // 特征图步长/缩放倍数，源代码设为 {16}

    for (const int step : feature_steps) 
	{
        assert(step != 0);

        int feature_map = IMAGE_HEIGHT / step;
        feature_maps.push_back(feature_map);

        int feature_size = feature_map * feature_map; // 单个特征图 所含锚点数(1个锚点位 包含 6种锚)
        feature_sizes.push_back(feature_size);

        std::cout << "—— 锚点数(单个特征图) ——: " << feature_size << std::endl; // 1600
    }

    anchor_sizes = {{8, 16, 32, 64, 128, 256}}; // 锚の尺寸类型

    sum_of_feature = std::accumulate(feature_sizes.begin(), feature_sizes.end(), 0) * anchor_num; // 9600 = 1600 * 6
    std::cout << "—— 锚の总数 ——: " << sum_of_feature << std::endl; 

    GenerateAnchors(); // 生成 特征图对应的 锚(真实中心点坐标 & 真实宽高)
}

MyFace::~MyFace() = default;

// ONNX模型文件 加载函数
void MyFace::LoadModel(std::string onnx_path) 
{ 
    std::cout << "开始加载 ONNX模型文件 ……" << std::endl;

    model = cv::dnn::readNetFromONNX(onnx_path);

    std::cout << "ONNX模型文件 加载完成！" << std::endl;
}

// 函数：图像预处理 & 前向推理 & 后处理 & 生成检测结果
std::vector<Bbox> MyFace::RunModel(cv::Mat& img) 
{
    std::vector<Bbox> result;
    // result.clear();    
	cv::Mat img_prepared = img;

    cv::Scalar mean_mxnet_(0.0, 0.0, 0.0); // 通道去均值，    用于 Blob图像处理
    float alpha_mxnet_ = 1.0;              // 数值缩放比例，  用于 Blob图像处理
    cv::Size size_detection(640, 640);     // 输出图像尺寸，  用于 Blob图像处理
    bool swap_BGR = true;                  // RB通道交换标识，用于 Blob图像处理

	// Blob图像处理：参数1 输入图像；参数2 数值缩放比例；参数3 输出图像尺寸；参数4 通道去均值；参数5 RB通道交换标识；参数6 图像裁剪标识(默认false)；参数7 输出图像深度(CV_32F/CV_8U)
    cv::Mat blob = cv::dnn::blobFromImage(img_prepared, alpha_mxnet_, size_detection, mean_mxnet_, swap_BGR);

    // 设置 模型输入：Blob图像，"input_image" 对应至 ONNX模型文件
    model.setInput(blob, "input_image");

	// 模型推理/前向计算：输出 三维矩阵(1,9600,16) 通道数=1；"complete_model_output" 对应至 ONNX模型文件
    cv::Mat out = model.forward("complete_model_output");

	// 后处理：结合 模型预测值 & 原始锚，根据 人脸得分 筛选锚，进行 人脸框 & 面部标记点解码 & NMS，得到 最终检测结果(用于绘至图像)
    auto faces = postProcess(img, out);
    auto rects = faces;

	// 进一步调整 检测结果，将 人脸得分 & 人脸框の左上点+右下点坐标 & 3个面部标记点坐标 存入 Bbox，再push进 result
    if (rects.size() != 0) 
	{
        for (const auto& rect : rects) 
		{
            Bbox box;

			// 人脸得分
            box.score = rect.confidence;

			// 人脸框の左上点+右下点坐标
            box.x1 = rect.face_box.x - rect.face_box.w / 2;
            box.y1 = rect.face_box.y - rect.face_box.h / 2;
            box.x2 = rect.face_box.x + rect.face_box.w / 2;
            box.y2 = rect.face_box.y + rect.face_box.h / 2;

			// 3个面部标记点坐标
            box.ppoint[0] = rect.keypoints[0].x;
            box.ppoint[1] = rect.keypoints[0].y;
            box.ppoint[2] = rect.keypoints[1].x;
            box.ppoint[3] = rect.keypoints[1].y;
            box.ppoint[4] = rect.keypoints[2].x;
            box.ppoint[5] = rect.keypoints[2].y;

            result.push_back(box);
        }
    }
    else // 防呆设置：检测不到人脸，则存入 0
	{
        Bbox box;

        box.score = 0;
        box.x1 = 0;
        box.y1 = 0;
        box.x2 = 0;
        box.y2 = 0;
        box.ppoint[0] = 0;
        box.ppoint[1] = 0;
        box.ppoint[2] = 0;
        box.ppoint[3] = 0;
        box.ppoint[4] = 0;
        box.ppoint[5] = 0; 

        result.push_back(box);
    }

    return result;
}

// 生成锚：真实中心点坐标 & 真实宽高
void MyFace::GenerateAnchors() 
{  
    refer_matrix = cv::Mat(sum_of_feature, bbox_head, CV_32FC1); // bbox_head = 4
    int line = 0;

    // 遍历 特征图（仅1个，此处可优化！）
	for (size_t feature_map = 0; feature_map < feature_maps.size(); feature_map++) 
	{
        // 单个特征图の 高 & 宽
		for (int height = 0; height < feature_maps[feature_map]; ++height) 
		{
            for (int width = 0; width < feature_maps[feature_map]; ++width) 
			{
                for (int anchor = 0; anchor < anchor_sizes[feature_map].size(); ++anchor) 
				{
                    auto* row = refer_matrix.ptr<float>(line);    

					row[0] = (float)(width + 0.5) * feature_steps[feature_map];  // 锚の中心点 x坐标
					row[1] = (float)(height + 0.5) * feature_steps[feature_map]; // 锚の中心点 y坐标
                    row[2] = anchor_sizes[feature_map][anchor];                  // 锚の真实宽度
                    row[3] = anchor_sizes[feature_map][anchor];                  // 锚の真实高度

                    line++;
                }
            }
        }
    }
}

// 图像预处理函数：输出 尺寸调整后的图像(640×640，两侧补灰边)
cv::Mat MyFace::prepareImage(cv::Mat &input_img) 
{        
    // 计算 缩放倍数
	float ratio = float(IMAGE_WIDTH) / float(input_img.cols) < float(IMAGE_HEIGHT) / float(input_img.rows) ? float(IMAGE_WIDTH) / float(input_img.cols) : float(IMAGE_HEIGHT) / float(input_img.rows);
    
	cv::Mat prepared_img = cv::Mat(cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT), CV_8UC3, cv::Scalar(128, 128, 128)); // 创建 图像模板（灰色，对应 Pytorch训练方式）
    cv::Mat rsz_img;

    cv::resize(input_img, rsz_img, cv::Size(), ratio, ratio); // 原图像 等比例缩放
    rsz_img.copyTo(prepared_img(cv::Rect((int)((IMAGE_WIDTH - rsz_img.cols) / 2), 0, rsz_img.cols, rsz_img.rows))); // 将缩放后的图像 填充至 图像模板
	
    return prepared_img;
}

// 后处理函数：结合 模型预测值 & 原始锚，根据 人脸得分 筛选锚，进行 人脸框 & 面部标记点解码 & NMS，得到 最终检测结果(用于绘至图像)
std::vector<MyFace::FaceRes> MyFace::postProcess(cv::Mat &src_img, cv::Mat &result_matrix)
{
    std::vector<FaceRes> result;
   
	int result_cols = 2 + bbox_head + landmark_head; // 单个锚 对应的 预测值数量 12 = 2 + 4 + 6

	// 遍历 所有锚，筛选出 人脸得分＞阈值的锚，解码对应的 人脸框 & 面部标记点
    for (int item = 0; item < sum_of_feature; ++item) // sum_of_feature 锚の总数(所有特征图) 16800
	{
        float* current_row = (float*)result_matrix.data + item * result_cols; // 单个锚の预测值の起始位置

        if (current_row[1] > obj_threshold) // current_row[1] 人脸置信度得分(是人脸的概率)；obj_threshold 判断阈值
		{           
            FaceRes headbox;
            headbox.confidence = current_row[1];
            auto* anchor = refer_matrix.ptr<float>(item);
            auto* bbox = current_row + 2;
            auto* keyp = current_row + 2 + bbox_head;
            auto* mask = current_row + 2 + bbox_head + landmark_head;

			// 重写 人脸框解码：真实中心点坐标 + 真实宽高
			// anchor[0-3]：锚の 真实中心点坐标 & 真实宽高
			headbox.face_box.x = anchor[0] + bbox[0] * 0.1 * anchor[2];
			headbox.face_box.y = anchor[1] + bbox[1] * 0.1 * anchor[3];
			headbox.face_box.w = anchor[2] * exp(bbox[2] * 0.2);
			headbox.face_box.h = anchor[3] * exp(bbox[3] * 0.2);
           
			// 重写 面部标记点解码：左眼 & 右眼 & 鼻子
			headbox.keypoints = {
				cv::Point2f(anchor[0] + keyp[0] * 0.1 * anchor[2],
							anchor[1] + keyp[1] * 0.1 * anchor[3]),
				cv::Point2f(anchor[0] + keyp[2] * 0.1 * anchor[2],
							anchor[1] + keyp[3] * 0.1 * anchor[3]),
				cv::Point2f(anchor[0] + keyp[4] * 0.1 * anchor[2],
							anchor[1] + keyp[5] * 0.1 * anchor[3])				
			};
			         
            result.push_back(headbox);
        }
    }

	// NMS
    NmsDetect(result);

    return result;
}

// NMS函数：自定义NMS（未使用 OpenCV官方NMS函数）
void MyFace::NmsDetect(std::vector<FaceRes> & detections) 
{
    sort(detections.begin(), detections.end(), [=](const FaceRes& left, const FaceRes& right) {
        return left.confidence > right.confidence;
    });

    for (int i = 0; i < (int)detections.size(); i++)
        for (int j = i + 1; j < (int)detections.size(); j++)
        {
            float iou = IOUCalculate(detections[i].face_box, detections[j].face_box);
            if (iou > nms_threshold)
                detections[j].confidence = 0;
        }

    detections.erase(std::remove_if(detections.begin(), detections.end(), [](const FaceRes& det)
    { return det.confidence == 0; }), detections.end());
}

// 函数：计算 IoU
float MyFace::IOUCalculate(const MyFace::FaceBox & det_a, const MyFace::FaceBox & det_b) 
{
    cv::Point2f center_a(det_a.x + det_a.w / 2, det_a.y + det_a.h / 2);
    cv::Point2f center_b(det_b.x + det_b.w / 2, det_b.y + det_b.h / 2);
    cv::Point2f left_up(std::min(det_a.x, det_b.x), std::min(det_a.y, det_b.y));
    cv::Point2f right_down(std::max(det_a.x + det_a.w, det_b.x + det_b.w), std::max(det_a.y + det_a.h, det_b.y + det_b.h));
    float distance_d = (center_a - center_b).x * (center_a - center_b).x + (center_a - center_b).y * (center_a - center_b).y;
    float distance_c = (left_up - right_down).x * (left_up - right_down).x + (left_up - right_down).y * (left_up - right_down).y;
    float inter_l = det_a.x > det_b.x ? det_a.x : det_b.x;
    float inter_t = det_a.y > det_b.y ? det_a.y : det_b.y;
    float inter_r = det_a.x + det_a.w < det_b.x + det_b.w ? det_a.x + det_a.w : det_b.x + det_b.w;
    float inter_b = det_a.y + det_a.h < det_b.y + det_b.h ? det_a.y + det_a.h : det_b.y + det_b.h;
    if (inter_b < inter_t || inter_r < inter_l)
        return 0;
    float inter_area = (inter_b - inter_t) * (inter_r - inter_l);
    float union_area = det_a.w * det_a.h + det_b.w * det_b.h - inter_area;
    if (union_area == 0)
        return 0;
    else
        return inter_area / union_area - distance_d / distance_c;
}  


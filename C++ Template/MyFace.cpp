#include "MyFace.h"
#include <numeric>

MyFace::MyFace() 
{
    BATCH_SIZE = 1;              // (Դ���룬����)Batch Size ����ͼ��������Ĭ�� 1
    INPUT_CHANNEL = 3;           // ����ͼ�� ͨ������Ĭ�� 3ͨ��(RGB)
    IMAGE_WIDTH = 640;           // ����ͼ�� ��� 640(ǿ�ƣ���Ӧ ONNXģ���ļ�)
    IMAGE_HEIGHT = 640;          // ����ͼ�� �߶� 640(ǿ�ƣ���Ӧ ONNXģ���ļ�)
    obj_threshold = 0.5;         // �����÷�/���Ŷ���ֵ��Ĭ�� 0.5
    nms_threshold = 0.3;         // NMS��ֵ��Ĭ�� 0.45    
    feature_steps = {16};        // ����ͼ����/���ű�����Դ������Ϊ {16}

    for (const int step : feature_steps) 
	{
        assert(step != 0);

        int feature_map = IMAGE_HEIGHT / step;
        feature_maps.push_back(feature_map);

        int feature_size = feature_map * feature_map; // ��������ͼ ����ê����(1��ê��λ ���� 6��ê)
        feature_sizes.push_back(feature_size);

        std::cout << "���� ê����(��������ͼ) ����: " << feature_size << std::endl; // 1600
    }

    anchor_sizes = {{8, 16, 32, 64, 128, 256}}; // ê�γߴ�����

    sum_of_feature = std::accumulate(feature_sizes.begin(), feature_sizes.end(), 0) * anchor_num; // 9600 = 1600 * 6
    std::cout << "���� ê������ ����: " << sum_of_feature << std::endl; 

    GenerateAnchors(); // ���� ����ͼ��Ӧ�� ê(��ʵ���ĵ����� & ��ʵ���)
}

MyFace::~MyFace() = default;

// ONNXģ���ļ� ���غ���
void MyFace::LoadModel(std::string onnx_path) 
{ 
    std::cout << "��ʼ���� ONNXģ���ļ� ����" << std::endl;

    model = cv::dnn::readNetFromONNX(onnx_path);

    std::cout << "ONNXģ���ļ� ������ɣ�" << std::endl;
}

// ������ͼ��Ԥ���� & ǰ������ & ���� & ���ɼ����
std::vector<Bbox> MyFace::RunModel(cv::Mat& img) 
{
    std::vector<Bbox> result;
    // result.clear();    
	cv::Mat img_prepared = img;

    cv::Scalar mean_mxnet_(0.0, 0.0, 0.0); // ͨ��ȥ��ֵ��    ���� Blobͼ����
    float alpha_mxnet_ = 1.0;              // ��ֵ���ű�����  ���� Blobͼ����
    cv::Size size_detection(640, 640);     // ���ͼ��ߴ磬  ���� Blobͼ����
    bool swap_BGR = true;                  // RBͨ��������ʶ������ Blobͼ����

	// Blobͼ��������1 ����ͼ�񣻲���2 ��ֵ���ű���������3 ���ͼ��ߴ磻����4 ͨ��ȥ��ֵ������5 RBͨ��������ʶ������6 ͼ��ü���ʶ(Ĭ��false)������7 ���ͼ�����(CV_32F/CV_8U)
    cv::Mat blob = cv::dnn::blobFromImage(img_prepared, alpha_mxnet_, size_detection, mean_mxnet_, swap_BGR);

    // ���� ģ�����룺Blobͼ��"input_image" ��Ӧ�� ONNXģ���ļ�
    model.setInput(blob, "input_image");

	// ģ������/ǰ����㣺��� ��ά����(1,9600,16) ͨ����=1��"complete_model_output" ��Ӧ�� ONNXģ���ļ�
    cv::Mat out = model.forward("complete_model_output");

	// ������� ģ��Ԥ��ֵ & ԭʼê������ �����÷� ɸѡê������ ������ & �沿��ǵ���� & NMS���õ� ���ռ����(���ڻ���ͼ��)
    auto faces = postProcess(img, out);
    auto rects = faces;

	// ��һ������ ��������� �����÷� & ����������ϵ�+���µ����� & 3���沿��ǵ����� ���� Bbox����push�� result
    if (rects.size() != 0) 
	{
        for (const auto& rect : rects) 
		{
            Bbox box;

			// �����÷�
            box.score = rect.confidence;

			// ����������ϵ�+���µ�����
            box.x1 = rect.face_box.x - rect.face_box.w / 2;
            box.y1 = rect.face_box.y - rect.face_box.h / 2;
            box.x2 = rect.face_box.x + rect.face_box.w / 2;
            box.y2 = rect.face_box.y + rect.face_box.h / 2;

			// 3���沿��ǵ�����
            box.ppoint[0] = rect.keypoints[0].x;
            box.ppoint[1] = rect.keypoints[0].y;
            box.ppoint[2] = rect.keypoints[1].x;
            box.ppoint[3] = rect.keypoints[1].y;
            box.ppoint[4] = rect.keypoints[2].x;
            box.ppoint[5] = rect.keypoints[2].y;

            result.push_back(box);
        }
    }
    else // �������ã���ⲻ������������� 0
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

// ����ê����ʵ���ĵ����� & ��ʵ���
void MyFace::GenerateAnchors() 
{  
    refer_matrix = cv::Mat(sum_of_feature, bbox_head, CV_32FC1); // bbox_head = 4
    int line = 0;

    // ���� ����ͼ����1�����˴����Ż�����
	for (size_t feature_map = 0; feature_map < feature_maps.size(); feature_map++) 
	{
        // ��������ͼ�� �� & ��
		for (int height = 0; height < feature_maps[feature_map]; ++height) 
		{
            for (int width = 0; width < feature_maps[feature_map]; ++width) 
			{
                for (int anchor = 0; anchor < anchor_sizes[feature_map].size(); ++anchor) 
				{
                    auto* row = refer_matrix.ptr<float>(line);    

					row[0] = (float)(width + 0.5) * feature_steps[feature_map];  // ê�����ĵ� x����
					row[1] = (float)(height + 0.5) * feature_steps[feature_map]; // ê�����ĵ� y����
                    row[2] = anchor_sizes[feature_map][anchor];                  // ê����ʵ���
                    row[3] = anchor_sizes[feature_map][anchor];                  // ê����ʵ�߶�

                    line++;
                }
            }
        }
    }
}

// ͼ��Ԥ����������� �ߴ�������ͼ��(640��640�����ಹ�ұ�)
cv::Mat MyFace::prepareImage(cv::Mat &input_img) 
{        
    // ���� ���ű���
	float ratio = float(IMAGE_WIDTH) / float(input_img.cols) < float(IMAGE_HEIGHT) / float(input_img.rows) ? float(IMAGE_WIDTH) / float(input_img.cols) : float(IMAGE_HEIGHT) / float(input_img.rows);
    
	cv::Mat prepared_img = cv::Mat(cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT), CV_8UC3, cv::Scalar(128, 128, 128)); // ���� ͼ��ģ�壨��ɫ����Ӧ Pytorchѵ����ʽ��
    cv::Mat rsz_img;

    cv::resize(input_img, rsz_img, cv::Size(), ratio, ratio); // ԭͼ�� �ȱ�������
    rsz_img.copyTo(prepared_img(cv::Rect((int)((IMAGE_WIDTH - rsz_img.cols) / 2), 0, rsz_img.cols, rsz_img.rows))); // �����ź��ͼ�� ����� ͼ��ģ��
	
    return prepared_img;
}

// ����������� ģ��Ԥ��ֵ & ԭʼê������ �����÷� ɸѡê������ ������ & �沿��ǵ���� & NMS���õ� ���ռ����(���ڻ���ͼ��)
std::vector<MyFace::FaceRes> MyFace::postProcess(cv::Mat &src_img, cv::Mat &result_matrix)
{
    std::vector<FaceRes> result;
   
	int result_cols = 2 + bbox_head + landmark_head; // ����ê ��Ӧ�� Ԥ��ֵ���� 12 = 2 + 4 + 6

	// ���� ����ê��ɸѡ�� �����÷֣���ֵ��ê�������Ӧ�� ������ & �沿��ǵ�
    for (int item = 0; item < sum_of_feature; ++item) // sum_of_feature ê������(��������ͼ) 16800
	{
        float* current_row = (float*)result_matrix.data + item * result_cols; // ����ê��Ԥ��ֵ����ʼλ��

        if (current_row[1] > obj_threshold) // current_row[1] �������Ŷȵ÷�(�������ĸ���)��obj_threshold �ж���ֵ
		{           
            FaceRes headbox;
            headbox.confidence = current_row[1];
            auto* anchor = refer_matrix.ptr<float>(item);
            auto* bbox = current_row + 2;
            auto* keyp = current_row + 2 + bbox_head;
            auto* mask = current_row + 2 + bbox_head + landmark_head;

			// ��д ��������룺��ʵ���ĵ����� + ��ʵ���
			// anchor[0-3]��ê�� ��ʵ���ĵ����� & ��ʵ���
			headbox.face_box.x = anchor[0] + bbox[0] * 0.1 * anchor[2];
			headbox.face_box.y = anchor[1] + bbox[1] * 0.1 * anchor[3];
			headbox.face_box.w = anchor[2] * exp(bbox[2] * 0.2);
			headbox.face_box.h = anchor[3] * exp(bbox[3] * 0.2);
           
			// ��д �沿��ǵ���룺���� & ���� & ����
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

// NMS�������Զ���NMS��δʹ�� OpenCV�ٷ�NMS������
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

// ���������� IoU
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


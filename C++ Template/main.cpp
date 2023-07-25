
// DEFace��C++ OpenCV ONNX�������

/*  ˵��  ����������������������������������������
	�沿��ǵ� Ԥ��ֵ��10��3������(���� & ���� & ����)
	������     Ԥ��ֵ�� 4�����ĵ�����ƫ���� & ���ƫ����
*/

#include <iostream>
#include <string>
#include <fstream>
#include "opencv2/opencv.hpp"
#include "MyFace.h"
#include <io.h>
#include <time.h>

// �ļ���Ѱ������Ѱ�� ָ��·��_path�� �ļ���ʽ=_filter�� ͼ���ļ������� �ַ�������return_(����׺)
std::vector<std::string> Get_File_List(const std::string& _path, const std::string& _filter)
{	
	std::string searching_dir = _path + "*." + _filter;	
	std::vector<std::string> return_;
	int cnt = 0;
	struct _finddata_t fd;
	intptr_t handle;

	if ((handle = _findfirst(searching_dir.c_str(), &fd)) == -1L)
	{
		std::cout << "���棺δ������ָ����ʽ��ͼ���ļ���" << std::endl;
	}
	
	do 
	{
		return_.push_back(fd.name);
		cnt += 1;
	} while (_findnext(handle, &fd) == 0);

	_findclose(handle);
	return return_;
}

// ������
int main(void)
{
	std::string model_path("DEFace_Method1.onnx");   // ONNXģ���ļ�·��
	std::string base_path(".\\image\\");             // �����ͼ��·��
	std::string output_dir(".\\result\\");           // �Ѽ���עͼ��洢·��
	std::string file_filter("jpg");                  // ѡ�� ͼ���ļ���ʽ

	// ɸѡ base_path�ļ����� ��ʽ=file_filter�� ͼ���ļ������ļ���(����׺)���� file_list
	std::vector<std::string> file_list = Get_File_List(base_path, file_filter);

	//std::cout << "�׸��ļ���: " << file_list[0] << std::endl; // ��ӡ����
	//file_list.erase(file_list.begin(), file_list.begin() + 2); // ��� ����file_list�е� ����Ԫ��

	MyFace MyFace;
	MyFace.LoadModel(model_path); // ���� ONNXģ���ļ�
	std::vector<Bbox> result;

	// ���� ���д����ͼ�񣬽��� ������⣬չʾ&�洢 �ѱ�עͼ��
	for (std::string file : file_list)
	{
		std::string image_path = base_path + file; // base_path + file = image/xxx.jpg
		cv::Mat img = cv::imread(image_path); // ��ȡ ԭͼ��
		
		// ���� prepareImage��������� �ߴ�������ͼ��(640��640�����ಹ�ұ�)
		img = MyFace.prepareImage(img);

		if (img.empty())
		{
			std::cout << "���棺�޷���ȡ ͼ���ļ�" << file << std::endl;
			continue;
		}
		
		// ִ�� ǰ������
		result = MyFace.RunModel(img); 

		// ���� ������ & �沿��ǵ�
		for (int aj = 0; aj < result.size(); aj++)
		{
			char name[256]; // �������ʶ����ʾ �����÷�
			cv::Scalar color(255, 0, 0); // ������ɫ��Red

			sprintf_s(name, "%.2f", result[aj].score); // �����÷� ���� �������ʶ			
			cv::putText(img, name, cv::Point(result[aj].x1, result[aj].y1), cv::FONT_HERSHEY_COMPLEX, 0.7, color, 2); // �����ı� ��ԭʼͼ��
			
			cv::Rect box(result[aj].x1, result[aj].y1, result[aj].x2 - result[aj].x1, result[aj].y2 - result[aj].y1); // ������cv::Rect������1 ���ϵ�x���ꣻ����2 ���ϵ�y���ꣻ����3 ��ȣ�����4 �߶�
			cv::rectangle(img, box, cv::Scalar(255, 0, 0), 2, cv::LINE_8, 0); // ���������� ��ԭʼͼ��

			// �����沿��ǵ� ��ԭʼͼ��
			for (int k = 0; k < 3; k++) 
			{
				cv::Point2f key_point = cv::Point2f(result[aj].ppoint[2 * k], result[aj].ppoint[2 * k + 1]); // �沿��ǵ�����
				if (k % 3 == 0)
					cv::circle(img, key_point, 3, cv::Scalar(0, 255, 0), -1);
				else if (k % 3 == 1)
					cv::circle(img, key_point, 3, cv::Scalar(255, 0, 255), -1);
				else
					cv::circle(img, key_point, 3, cv::Scalar(0, 255, 255), -1);
			}
		}
		
		cv::imshow("result", img);      // չʾ �ѱ�עͼ��
		cv::imwrite(output_dir + "Detected-" + file, img); // �洢 �ѱ�עͼ��

		cv::waitKey(0);			
	}

	return 0;
}


// DEFace：C++ OpenCV ONNX部署测试

/*  说明  ————————————————————
	面部标记点 预测值：10，3点坐标(左眼 & 右眼 & 鼻子)
	人脸框     预测值： 4，中心点坐标偏移量 & 宽高偏移量
*/

#include <iostream>
#include <string>
#include <fstream>
#include "opencv2/opencv.hpp"
#include "MyFace.h"
#include <io.h>
#include <time.h>

// 文件搜寻函数：寻找 指定路径_path下 文件格式=_filter的 图像文件，存入 字符串向量return_(含后缀)
std::vector<std::string> Get_File_List(const std::string& _path, const std::string& _filter)
{	
	std::string searching_dir = _path + "*." + _filter;	
	std::vector<std::string> return_;
	int cnt = 0;
	struct _finddata_t fd;
	intptr_t handle;

	if ((handle = _findfirst(searching_dir.c_str(), &fd)) == -1L)
	{
		std::cout << "警告：未搜索到指定格式的图像文件！" << std::endl;
	}
	
	do 
	{
		return_.push_back(fd.name);
		cnt += 1;
	} while (_findnext(handle, &fd) == 0);

	_findclose(handle);
	return return_;
}

// 主函数
int main(void)
{
	std::string model_path("DEFace_Method1.onnx");   // ONNX模型文件路径
	std::string base_path(".\\image\\");             // 待检测图像路径
	std::string output_dir(".\\result\\");           // 已检测标注图像存储路径
	std::string file_filter("jpg");                  // 选择 图像文件格式

	// 筛选 base_path文件夹中 格式=file_filter的 图像文件，将文件名(含后缀)存入 file_list
	std::vector<std::string> file_list = Get_File_List(base_path, file_filter);

	//std::cout << "首个文件名: " << file_list[0] << std::endl; // 打印测试
	//file_list.erase(file_list.begin(), file_list.begin() + 2); // 清除 向量file_list中的 部分元素

	MyFace MyFace;
	MyFace.LoadModel(model_path); // 载入 ONNX模型文件
	std::vector<Bbox> result;

	// 遍历 所有待检测图像，进行 人脸检测，展示&存储 已标注图像
	for (std::string file : file_list)
	{
		std::string image_path = base_path + file; // base_path + file = image/xxx.jpg
		cv::Mat img = cv::imread(image_path); // 读取 原图像
		
		// 调用 prepareImage函数，输出 尺寸调整后的图像(640×640，两侧补灰边)
		img = MyFace.prepareImage(img);

		if (img.empty())
		{
			std::cout << "警告：无法读取 图像文件" << file << std::endl;
			continue;
		}
		
		// 执行 前向推理
		result = MyFace.RunModel(img); 

		// 绘制 人脸框 & 面部标记点
		for (int aj = 0; aj < result.size(); aj++)
		{
			char name[256]; // 人脸框标识：显示 人脸得分
			cv::Scalar color(255, 0, 0); // 文字颜色：Red

			sprintf_s(name, "%.2f", result[aj].score); // 人脸得分 存入 人脸框标识			
			cv::putText(img, name, cv::Point(result[aj].x1, result[aj].y1), cv::FONT_HERSHEY_COMPLEX, 0.7, color, 2); // 插入文本 至原始图像
			
			cv::Rect box(result[aj].x1, result[aj].y1, result[aj].x2 - result[aj].x1, result[aj].y2 - result[aj].y1); // 人脸框；cv::Rect：参数1 左上点x坐标；参数2 左上点y坐标；参数3 宽度；参数4 高度
			cv::rectangle(img, box, cv::Scalar(255, 0, 0), 2, cv::LINE_8, 0); // 插入人脸框 至原始图像

			// 插入面部标记点 至原始图像
			for (int k = 0; k < 3; k++) 
			{
				cv::Point2f key_point = cv::Point2f(result[aj].ppoint[2 * k], result[aj].ppoint[2 * k + 1]); // 面部标记点坐标
				if (k % 3 == 0)
					cv::circle(img, key_point, 3, cv::Scalar(0, 255, 0), -1);
				else if (k % 3 == 1)
					cv::circle(img, key_point, 3, cv::Scalar(255, 0, 255), -1);
				else
					cv::circle(img, key_point, 3, cv::Scalar(0, 255, 255), -1);
			}
		}
		
		cv::imshow("result", img);      // 展示 已标注图像
		cv::imwrite(output_dir + "Detected-" + file, img); // 存储 已标注图像

		cv::waitKey(0);			
	}

	return 0;
}

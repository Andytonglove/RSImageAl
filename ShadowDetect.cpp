// ShadowDetect.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//
#include <stdio.h>
#include <iostream>
#include <string>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

// 此程序用以进行阴影检测，基于OpenCV 4.5.5
// 具体包括阴影检测和阈值分割、区域合并函数，以及区域合并，主程序中对Color和zy数据进行处理

/* 阴影检测：包括HSV和C1C2C3 */
Mat HSV(Mat& src) {
	double thres = -0.95;  // 确定阈值
	int rows = src.rows, cols = src.cols;

	Mat shadow;
	shadow.create(rows, cols, CV_8UC3);
	uchar* pData = shadow.data;

	double mmin = 0, mmax = 0;

	double min;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			double b = src.at<Vec3b>(i, j)[0];
			double g = src.at<Vec3b>(i, j)[1];
			double r = src.at<Vec3b>(i, j)[2];

			min = (r < g) ? r : g;  // 计算最小值
			if (min > b) {
				min = b;
			}
			double v = (b + g + r) / 3;
			double s = 1.0 - (3.0 / (b + g + r)) * min;

			double h;
			if ((r - g) * (r - g) + (r - b) * (g - b) != 0) {
				h = acos((2 * r - g - b) / (2 * pow((r - g) * (r - g) + (r - b) * (g - b), 0.5)));
			} else {
				h = 0.0;
			}
			if (g < b) {
				h = 2 * CV_PI - h;
			}

			double m = (s - v) / (h + s + v);  // HSV
			if (m < thres) {
				shadow.at<Vec3b>(i, j)[0] = 80;  // 紫色
				shadow.at<Vec3b>(i, j)[1] = 0;
				shadow.at<Vec3b>(i, j)[2] = 80;
			}
			else {
				shadow.at<Vec3b>(i, j)[0] = 0;
				shadow.at<Vec3b>(i, j)[1] = 255;
				shadow.at<Vec3b>(i, j)[2] = 255;
			}
		}
	}
	return shadow;
}

Mat C1C2C3(Mat& src, double thres1, double thres2) {
	int rows = src.rows, cols = src.cols;

	Mat C123;
	C123.create(rows, cols, CV_64FC3);

	Mat shadow;
	shadow.create(rows, cols, CV_8UC3);

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			double b = src.at<Vec3b>(i, j)[0];
			double g = src.at<Vec3b>(i, j)[1];
			double r = src.at<Vec3b>(i, j)[2];

			double c1, c2, c3;
			// 进行C1C2C3计算
			if (g > b) {
				c1 = atan(r / g);
			} else {
				c1 = atan(r / b);
			}
			if (r > b) {
				c2 = atan(g / r);
			} else {
				c2 = atan(g / b);
			}
			if (r > g) {
				c3 = atan(b / r);
			} else {
				c3 = atan(b / g);
			}
			// 进行色彩映射
			if (c3 > thres1 && b < thres2) {
				shadow.at<Vec3b>(i, j)[0] = 0;
				shadow.at<Vec3b>(i, j)[1] = 255;
				shadow.at<Vec3b>(i, j)[2] = 255;
			} else {
				shadow.at<Vec3b>(i, j)[0] = 80;  // 紫色
				shadow.at<Vec3b>(i, j)[1] = 0;
				shadow.at<Vec3b>(i, j)[2] = 80;
			}
		}
	}
	return shadow;
}

/* 阈值分割：包括Otsu、状态法和判断分析法 */
Mat Otsu(Mat& src, string filename) {
	/* Otsu阈值分割 */
	int thresh = 0;
	const int Grayscale = 256;
	int graynum[Grayscale] = { 0 };
	int r = src.rows, c = src.cols * 3;
	for (int i = 0; i < r; ++i) {
		const uchar* ptr = src.ptr<uchar>(i);
		for (int j = 0; j < c; ++j) {
			graynum[ptr[j]]++;  // 直方图统计
		}
	}

	double P[Grayscale] = { 0 };
	double PK[Grayscale] = { 0 };
	double MK[Grayscale] = { 0 };
	double srcpixnum = r * c, sumtmpPK = 0, sumtmpMK = 0;
	for (int i = 0; i < Grayscale; ++i) {
		P[i] = graynum[i] / srcpixnum;  // 每个灰度级出现的概率
		PK[i] = sumtmpPK + P[i];  // 概率累计和 
		sumtmpPK = PK[i];
		MK[i] = sumtmpMK + i * P[i];  // 灰度级的累加均值                                                                                                                                                                                                                                                                                                                                                                                                        
		sumtmpMK = MK[i];
	}

	// 计算类间方差
	double Var = 0;
	for (int k = 0; k < Grayscale; ++k) {
		if ((MK[Grayscale - 1] * PK[k] - MK[k]) * (MK[Grayscale - 1] * PK[k] - MK[k]) / (PK[k] * (1 - PK[k])) > Var) {
			Var = (MK[Grayscale - 1] * PK[k] - MK[k]) * (MK[Grayscale - 1] * PK[k] - MK[k]) / (PK[k] * (1 - PK[k]));
			thresh = k;
		}
	}
	cout << thresh;
	// 阈值处理
	Mat dst;
	src.copyTo(dst);
	cvtColor(dst, dst, COLOR_BGR2GRAY);  // 转为单波段
	for (int i = 0; i < r; ++i) {
		uchar* ptr = dst.ptr<uchar>(i);
		for (int j = 0; j < c / 3; ++j) {
			if (ptr[j] > thresh)
				ptr[j] = 255;
			else
				ptr[j] = 0;
		}

	}
	imshow("Otsu-" + filename, dst);
	waitKey(0);
	imwrite("Otsu-" + filename + ".bmp", dst);
	return dst;
}

Mat StateBin(Mat& src, string filename)
{
	/* 状态法（峰谷法）阈值分割 */
	int T = 0; // 阈值
	int nNewThre = 0;
	//初始化定义类1、类2像素总数；类1、类2灰度均值
	int cnt1 = 0, cnt2 = 0, mval1 = 0, mval2 = 0;
	int iter = 0; // 迭代次数
	// 先将彩色图像化为灰度图像
	Mat dst;
	src.copyTo(dst);
	cvtColor(src, dst, COLOR_BGR2GRAY);  // 转为单波段
	int Rows = dst.rows, Cols = dst.cols;

	int nEISize = dst.elemSize(); // 获取每个像素的字节数
	int G = pow(2, double(8 * nEISize)); // 灰度级数
	nNewThre = int(G / 2); // 给阈值赋迭代初值
	// 分配灰度级数个量的内存，储存并计算灰度直方图
	auto* hist = new int[G]; // 灰度统计数组
	for (int i = 0; i < G; i++) {
		hist[i] = 0;
	}
	for (int i = 0; i < Rows; i++) {
		for (int j = 0; j < Cols; j++) {
			int g = dst.at<uchar>(i, j);
			hist[g]++;
		}
	}
	// 迭代求最佳阈值
	for (iter = 0;iter < 100;iter++) {
		/* 初始化 */
		T = nNewThre;
		for (int m = T; m < G; m++) {
			cnt2 += hist[m];
		}
		cnt1 = dst.rows * dst.rows - cnt2;
		// 组1从0开始计数到小于i，组2反向计数从G递减到等于i
		for (int n = T; n < G; n++) {
			mval2 += (double(hist[n]) / cnt2) * n;
		}
		for (int k = 0; k < T; k++) {
			mval1 += (double(hist[k]) / cnt1) * k;
		}
		T = int(mval1 + mval2) / 2; //得新阈值
	}
	for (int i = 0;i < Rows;i++) {
		for (int j = 0;j < Cols;j++) {
			dst.at<uchar>(i, j) = (dst.at<uchar>(i, j) > T) ? (G - 1) : 0; // 特别注意这里最大像素值是G-1！！！
		}
	}
	imshow("状态法-" + filename, dst);
	waitKey(0);
	imwrite("状态法-" + filename + ".bmp", dst);
	return dst;

	delete[]hist;
}

Mat AnalysisBin(Mat& src, string filename)
{
	/* 判断分析法：基本原理：使被阈值区分的两组灰度级之间，组内组间方差比最大 */
	/* 一些值的定义与初始化 */
	double ratio = 0;
	int Thre = 0; // 阈值
	double cnt1 = 0, cnt2 = 0; // 两组像素的总数量
	double mval1 = 0, mval2 = 0, mval = 0; // 两组像素的灰度平均值以及整幅图像的灰度平均值
	double delta1 = 0, delta2 = 0; // 第一组、第二组像素的方差
	double deltaW = 0, deltaB = 0; // 组内方差、组间方差

	// 先将彩色图像化为灰度图像
	Mat dst;
	src.copyTo(dst);
	cvtColor(src, dst, COLOR_BGR2GRAY);  // 转为单波段
	int Rows = dst.rows, Cols = dst.cols;

	int nEISize = dst.elemSize(); // 获取每个像素的字节数
	int G = pow(2, double(8 * nEISize)); // 灰度级数
	// 分配灰度级数个量的内存，储存并计算灰度直方图
	auto* hist = new int[G]; // 灰度统计数组
	for (int i = 0; i < G; i++) {
		hist[i] = 0;
	}
	for (int i = 0; i < Rows; i++) {
		for (int j = 0; j < Cols; j++) {
			int H = dst.at<uchar>(i, j);
			hist[H]++;
		}
	}

	// 通过使组内方差与组间方差之比最大来确定阈值T
	for (int i = 0; i < G; i++) {
		for (int m = i; m < G; m++) {
			cnt2 += hist[m];
		}
		cnt1 = Cols * Rows - cnt2;

		// 组1从0开始计数到小于i，组2反向计数从G递减到等于i
		for (int n = i; n < G; n++) {
			mval2 += (double(hist[n]) / cnt2) * n;
		}
		for (int k = 0; k < i; k++) {
			mval1 += (double(hist[k]) / cnt1) * k;
		}

		// 整幅图像的灰度平均值计算
		mval = (mval1 * cnt1 + mval2 * cnt2) / (cnt1 + cnt2);

		// 两组的方差以及组内方差和组间方差计算，同上理
		for (int p = i; p < G; p++) {
			delta2 += (double(hist[p]) / cnt2) * pow((p - mval2), 2);
		}
		for (int q = 0;q < i;q++) {
			delta1 += (double(hist[q]) / cnt1) * pow((q - mval1), 2);
		}
		deltaW = cnt1 * delta1 + cnt2 * delta2;
		deltaB = cnt1 * cnt2 * pow((mval1 - mval2), 2);
		if ((deltaB / deltaW) > ratio) {
			ratio = deltaB / deltaW;
			Thre = i; // 阈值T计算
		}
		// 重新赋值还原为0
		cnt1 = 0;
		cnt2 = 0;
		mval1 = 0;
		mval2 = 0;
		delta1 = 0;
		delta2 = 0;
		deltaW = 0;
		deltaB = 0;
	}

	// 根据阈值进行二值化处理
	for (int i = 0; i < Rows; i++) {
		for (int j = 0; j < Cols; j++) {
			dst.at<uchar>(i, j) = (dst.at<uchar>(i, j) < Thre) ? 0 : (G - 1); // 通过阈值来赋值
		}
	}
	imshow("判断分析法-" + filename, dst);
	waitKey(0);
	imwrite("判断分析法-" + filename + ".bmp", dst);
	return dst;

	delete[]hist;
}

/* 区域合并 */
void DetailMerge(Mat& src, int type, string filename)
{
	/* 本函数进行微小阴影区域合并 */
	Mat temp = src.clone();

	// 腐蚀和膨胀操作形成连通域
	Mat element1 = getStructuringElement(MORPH_ELLIPSE, Size(2, 2));
	Mat element2 = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));

	// type=0即为处理HSV，为1处理C1C2C3
	if (type == 0) {
		dilate(temp, temp, element1, Point(-1, -1), 6);
		dilate(temp, temp, element2, Point(-1, -1), 3);
		erode(temp, temp, element1, Point(-1, -1), 1);
	}
	else if (type == 1) {
		dilate(temp, temp, element1, Point(-1, -1), 2);
		dilate(temp, temp, element2, Point(-1, -1), 1);
		erode(temp, temp, element1, Point(-1, -1), 1);
	}

	Mat dst = temp.clone();
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	// 寻找连通区域
	findContours(dst, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);

	for (int i = 0; i < contours.size(); i++){
		// 连通区域阈值选取
		if (contourArea(contours[i]) < 100){
			drawContours(dst, contours, i, Scalar(0), -1);
		}
	}

	imshow("阴影检测-区域合并结果-"+ filename, dst);
	waitKey(0);
	imwrite("阴影检测-区域合并结果-" + filename + ".bmp", dst);
}

int main()
{
	Mat bmp = imread("data/Color.bmp");  // 读入bmp位图
	Mat zy = imread("data/zy-3-wd.jpg");  // 读入zy-3图像

	/* 对Color的bmp图像进行处理 */

	// 基于HSV彩色空间的阴影检测
	Mat shadow_HSV = HSV(bmp);
	imshow("HSV", shadow_HSV);
	waitKey(0);
	imwrite("shadow_HSV.bmp", shadow_HSV);

	// 选取效果最佳的两种方法对HSV结果进行阈值分割
	Mat otsu_HSV = Otsu(shadow_HSV, "HSV");
	Mat ana_HSV = AnalysisBin(shadow_HSV, "HSV");
	DetailMerge(otsu_HSV, 0, "HSV");  // 区域合并

	// 基于C1C2C3彩色空间的阴影检测
	Mat shadow_C1C2C3 = C1C2C3(bmp, 0.8, 80);
	imshow("C1C2C3", shadow_C1C2C3);
	waitKey(0);
	imwrite("shadow_C1C2C3.bmp", shadow_C1C2C3);

	// 同上，进行阈值分割
	Mat state_C1C2C3 = StateBin(shadow_C1C2C3, "C1C2C3");
	Mat ana_C1C2C3 = AnalysisBin(shadow_C1C2C3, "C1C2C3");
	DetailMerge(ana_C1C2C3, 1, "C1C2C3");  // 区域合并


	/* 对对zy-3-wd图像进行处理 */
	
	// 基于HSV彩色空间的阴影检测
	Mat zy_HSV = HSV(zy);
	imshow("zy_HSV", zy_HSV);
	waitKey(0);
	imwrite("zy_HSV.bmp", zy_HSV);
	// 选取效果最佳的两种方法对HSV结果进行阈值分割
	Mat otsu_zy_HSV = Otsu(zy_HSV, "zy_HSV");
	Mat ana_zy_HSV = AnalysisBin(zy_HSV, "zy_HSV");

	// 基于C1C2C3彩色空间的阴影检测
	Mat zy_C1C2C3 = C1C2C3(zy, 0.7, 100);
	imshow("zy_C1C2C3", zy_C1C2C3);
	waitKey(0);
	imwrite("zy_C1C2C3.bmp", zy_C1C2C3);
	Mat ana_zy_C1C2C3 = AnalysisBin(zy_C1C2C3, "zy_C1C2C3");  // 阈值分割
}

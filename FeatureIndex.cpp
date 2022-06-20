// FeatureIndex.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <stdio.h>
#include <iostream>
#include <math.h>
#include <string>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

// 本程序为基于特征指数的遥感专题信息提取算法编程，基于OpenCV 4.5.5
// 具体包括多种植被指数、水体指数、建筑指数和阈值分割函数，在main主程序中调用

/* 植被指数：包括RVI、RDVI、SAVI */
Mat RVI(Mat& tm3, Mat& tm4) {
	int rows = tm3.rows, cols = tm3.cols;
	Mat rvi(rows, cols, CV_8UC3);
	double band3, band4;
	for (int i = 0; i < rows; i++){
		for (int j = 0; j < cols; j++){
			band3 = (double)tm3.at<Vec3b>(i, j)[0];
			band4 = (double)tm4.at<Vec3b>(i, j)[0];
			// 植被这里以绿色为代表进行标注，各类指数越大，绿色越深
			rvi.at<Vec3b>(i, j)[1] = (255 - 255 * (band4 / band3));  // 绿色层，RVI越大，绿色越深
			rvi.at<Vec3b>(i, j)[0] = 0;
			rvi.at<Vec3b>(i, j)[2] = 0;
			if (band3 > band4) {
				// NDVI小于0则显示白色
				rvi.at<Vec3b>(i, j)[0] = rvi.at<Vec3b>(i, j)[1] = rvi.at<Vec3b>(i, j)[2] = 255;
			}
		}
	}
	normalize(rvi, rvi, 0, 255, NORM_MINMAX);  // 归一化到线性空间
	imshow("RVI", rvi);  // 展示并保存图片
	waitKey(0);
	imwrite("RVI.bmp", rvi);
	return rvi;
}

Mat NDVI(Mat& tm3, Mat& tm4) {
	int rows = tm3.rows, cols = tm3.cols;
	Mat ndvi(rows, cols, CV_8UC3);
	double band3, band4;
	for (int i = 0; i < rows; i++){
		for (int j = 0; j < cols; j++){
			band3 = (double)tm3.at<Vec3b>(i, j)[0];
			band4 = (double)tm4.at<Vec3b>(i, j)[0];
			ndvi.at<Vec3b>(i, j)[1] = (255 - 255 * (band4 - band3) / (band3 + band4));
			ndvi.at<Vec3b>(i, j)[0] = 0;
			ndvi.at<Vec3b>(i, j)[2] = 0;
			if (band3 > band4) {
				// NDVI小于0则显示白色
				ndvi.at<Vec3b>(i, j)[0] = ndvi.at<Vec3b>(i, j)[1] = ndvi.at<Vec3b>(i, j)[2] = 255;
			}
		}
	}
	imshow("NDVI", ndvi);
	waitKey(0);
	imwrite("NDVI.bmp", ndvi);
	return ndvi;
}

Mat SAVI(Mat& tm3, Mat& tm4) {
	int rows = tm3.rows, cols = tm3.cols;
	double L = 0.5;
	Mat savi(rows, cols, CV_8UC3);
	double band3, band4;
	for (int i = 0; i < rows; i++){
		for (int j = 0; j < cols; j++){
			band3 = (double)tm3.at<Vec3b>(i, j)[0];
			band4 = (double)tm4.at<Vec3b>(i, j)[0];
			savi.at<Vec3b>(i, j)[1] = (255 - 255 * (2 * band4 + 1 - sqrt((2 * band4 + 1) * (2 * band4 + 1) - 8 * (band4 - band3))) / 2);
			savi.at<Vec3b>(i, j)[0] = 0;
			savi.at<Vec3b>(i, j)[2] = 0;
			if (band3 > band4) {
				savi.at<Vec3b>(i, j)[0] = savi.at<Vec3b>(i, j)[1] = savi.at<Vec3b>(i, j)[2] = 255;
			}
		}
	}
	normalize(savi, savi, 0, 255, NORM_MINMAX);
	imshow("SAVI", savi);
	waitKey(0);
	imwrite("SAVI.bmp", savi);
	return savi;
}

/* 水体指数：包括NDWI、MNDWI */
Mat NDWI(Mat& tm2, Mat& tm4)
{
	int rows = tm2.rows, cols = tm2.cols;
	Mat ndwi(rows, cols, CV_8UC3);
	double band2, band4;
	for (int i = 0; i < rows; i++){
		for (int j = 0; j < cols; j++){
			band2 = (double)tm2.at<Vec3b>(i, j)[0];
			band4 = (double)tm4.at<Vec3b>(i, j)[0];
			// 水体这里以蓝色为代表进行标注，各类指数越大，蓝色越深
			ndwi.at<Vec3b>(i, j)[0] = (255 - 255 * (band2 - band4) / (band2 + band4));  // 蓝色层，NDMI越大，蓝色越深
			ndwi.at<Vec3b>(i, j)[2] = 0;
			ndwi.at<Vec3b>(i, j)[1] = 0;
			if (band2 < band4) {
				// NDWI小于0则显示白色
				ndwi.at<Vec3b>(i, j)[0] = ndwi.at<Vec3b>(i, j)[1] = ndwi.at<Vec3b>(i, j)[2] = 255;
			}

		}
	}
	normalize(ndwi, ndwi, 0, 255, NORM_MINMAX);
	imshow("NDWI", ndwi);
	waitKey(0);
	imwrite("NDWI.bmp", ndwi);
	return ndwi;
}

Mat MNDWI(Mat& tm2, Mat& tm5)
{
	int rows = tm2.rows, cols = tm2.cols;
	Mat mndwi(rows, cols, CV_8UC3);
	double band2, band5;
	for (int i = 0; i < rows; i++){
		for (int j = 0; j < cols; j++){
			band2 = (double)tm2.at<Vec3b>(i, j)[0];
			band5 = (double)tm5.at<Vec3b>(i, j)[0];
			mndwi.at<Vec3b>(i, j)[0] = (255 - 255 * (band2 - band5) / (band2 + band5));
			mndwi.at<Vec3b>(i, j)[1] = 0;
			mndwi.at<Vec3b>(i, j)[2] = 0;
			if (band2 < band5) {
				// MNDWI小于0则显示白色
				mndwi.at<Vec3b>(i, j)[0] = mndwi.at<Vec3b>(i, j)[1] = mndwi.at<Vec3b>(i, j)[2] = 255;
			}
		}
	}
	normalize(mndwi, mndwi, 0, 255, NORM_MINMAX);
	imshow("MNDWI", mndwi);
	waitKey(0);
	imwrite("MNDWI.bmp", mndwi);
	return mndwi;
}

/* 建筑指数：包括DBI、NDBI */
Mat DBI(Mat& tm4, Mat& tm7)
{
	int rows = tm4.rows, cols = tm4.cols;
	Mat dbi(rows, cols, CV_8UC3);
	double band4, band7;
	for (int i = 0; i < rows; i++){
		for (int j = 0; j < cols; j++){
			band4 = (double)tm4.at<Vec3b>(i, j)[0];
			band7 = (double)tm7.at<Vec3b>(i, j)[0];
			// 用三通道进行计算，从而显示彩色图像
			dbi.at<Vec3b>(i, j)[0] = 100;
			dbi.at<Vec3b>(i, j)[1] = (255 - 255 * (band7 - band4));
			dbi.at<Vec3b>(i, j)[2] = (255 - 255 * (band7 - band4));

			if (band4 > band7) {
				// DBI小于0则显示白色
				dbi.at<Vec3b>(i, j)[0] = dbi.at<Vec3b>(i, j)[1] = dbi.at<Vec3b>(i, j)[2] = 255;
			}
		}
	}
	normalize(dbi, dbi, 0, 255, NORM_MINMAX);
	imshow("DBI", dbi);
	waitKey(0);
	imwrite("DBI.bmp", dbi);
	return dbi;
}

Mat NDBI(Mat& tm4, Mat& tm5)
{
	int rows = tm4.rows, cols = tm4.cols;
	Mat ndbi(rows, cols, CV_8UC3);
	double band4, band5;
	for (int i = 0; i < rows; i++){
		for (int j = 0; j < cols; j++){
			band4 = (double)tm4.at<Vec3b>(i, j)[0];
			band5 = (double)tm5.at<Vec3b>(i, j)[0];
			ndbi.at<Vec3b>(i, j)[0] = 100;
			ndbi.at<Vec3b>(i, j)[1] = (255 - 255 * (band5 - band4) / (band5 + band4));
			ndbi.at<Vec3b>(i, j)[2] = (255 - 255 * (band5 - band4) / (band5 + band4));

			if (band4 > band5) {
				ndbi.at<Vec3b>(i, j)[0] = ndbi.at<Vec3b>(i, j)[1] = ndbi.at<Vec3b>(i, j)[2] = 255;
			}
		}
	}
	normalize(ndbi, ndbi, 0, 255, NORM_MINMAX);
	imshow("NDBI", ndbi);
	waitKey(0);
	imwrite("NDBI.bmp", ndbi);
	return ndbi;
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
	imshow("Otsu-"+ filename, dst);
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

int main()
{
	// 读取各波段文件
	Mat img_b = imread("data/tm1.tif"); // 蓝波段
	Mat img_g = imread("data/tm2.tif"); // 绿
	Mat img_r = imread("data/tm3.tif"); // 红
	Mat img_nr = imread("data/tm4.tif"); // 近红外
	Mat img_mr = imread("data/tm5.tif"); // 中红外
	Mat img_dr = imread("data/tm7.tif"); // 远红外

	// 进行指数计算
	Mat rvi = RVI(img_r, img_nr);  // 比值植被指数
	Mat ndvi = NDVI(img_r, img_nr);  // 归一化植被指数
	Mat savi = SAVI(img_r, img_nr);  // 土壤调节植被指数
	Mat ndwi = NDWI(img_g, img_nr);  // 归一化水指数
	Mat mndwi = MNDWI(img_g, img_mr);  // 归一化差异水体指数
	Mat dbi = DBI(img_nr, img_dr);  // 差值建筑覆盖指数
	Mat ndbi = NDBI(img_nr, img_mr);  // 归一化差值建筑用地指数

	// 阈值分割
	Mat otsu = imread("data/Color.bmp");
	Mat otsu_color = Otsu(otsu, "Color");  // Otsu
	Mat state_color = StateBin(otsu, "Color");  // 状态法
	Mat analysis_color = AnalysisBin(otsu, "Color");  // 判断分析法

	// 对上面的各类指数中阈值分割效果最好的方法，二值化植被水体建筑指数专题信息图
	Mat rvi_binary = Otsu(rvi, "RVI");
	Mat ndvi_binary = AnalysisBin(ndvi, "NDVI");
	Mat savi_binary = Otsu(savi, "SAVI");
	Mat ndwi_binary = AnalysisBin(ndwi, "NDWI");
	Mat mndwi_binary = Otsu(mndwi, "MNDWI");
	Mat dbi_binary = Otsu(dbi, "DBI");
	Mat ndbi_binary_otsu = Otsu(ndbi, "NDBI");
	Mat ndbi_binary_ana = AnalysisBin(ndbi, "NDBI");

	waitKey(0);
}
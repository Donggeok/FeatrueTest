#include <iostream>
#include <opencv.hpp>

#define M_PI 3.1415926535

void unevenLightCompensate(cv::Mat &image, int blockSize)
{
	if (image.channels() == 3) cvtColor(image, image, 7);
	double average = mean(image)[0];
	int rows_new = ceil(double(image.rows) / double(blockSize));
	int cols_new = ceil(double(image.cols) / double(blockSize));
	cv::Mat blockImage;
	blockImage = cv::Mat::zeros(rows_new, cols_new, CV_32FC1);
	for (int i = 0; i < rows_new; i++)
	{
		for (int j = 0; j < cols_new; j++)
		{
			int rowmin = i*blockSize;
			int rowmax = (i + 1)*blockSize;
			if (rowmax > image.rows) rowmax = image.rows;
			int colmin = j*blockSize;
			int colmax = (j + 1)*blockSize;
			if (colmax > image.cols) colmax = image.cols;
			cv::Mat imageROI = image(cv::Range(rowmin, rowmax), cv::Range(colmin, colmax));
			double temaver = cv::mean(imageROI)[0];
			blockImage.at<float>(i, j) = temaver;
		}
	}
	blockImage = blockImage - average;
	cv::Mat blockImage2;
	resize(blockImage, blockImage2, image.size(), (0, 0), (0, 0), CV_INTER_CUBIC);
	cv::Mat image2;
	image.convertTo(image2, CV_32FC1);
	cv::Mat dst = image2 - blockImage2;
	dst.convertTo(image, CV_8UC1);
}

cv::Scalar calGaussianParam(std::vector<float> samples) {
	cv::Scalar tmp;
	float mu = cv::mean(samples)[0];
	for (float &e : samples) {
		e = powf(e - mu, 2);
	}
	double sigma2 = cv::mean(samples)[0];
	tmp[0] = mu;
	tmp[1] = sigma2;
	return tmp;
}

double Gaussian(cv::Scalar param, float x) {
	double tmp = 2 * param[1];
	double value = exp(-powf((x - param[0]), 2) / tmp) / sqrt(tmp*CV_PI);
	return value;
}

// src为待处理的灰度图，2^kmax为最大窗口，kmax一般为6
double ETVP_FEATURE_Tamura_Coarseness(cv::Mat src, int kmax) {
	int height = src.rows;
	int width = src.cols;
	std::vector<cv::Mat> A = std::vector<cv::Mat>(kmax, cv::Mat::zeros(cv::Size(width ,height), CV_32FC1));
	std::vector<cv::Mat> Eh = std::vector<cv::Mat>(kmax, cv::Mat::zeros(cv::Size(width, height), CV_32FC1));
	std::vector<cv::Mat> Ev = std::vector<cv::Mat>(kmax, cv::Mat::zeros(cv::Size(width, height), CV_32FC1));
	cv::Mat Sbest = cv::Mat::zeros(cv::Size(width, height), CV_8UC1);

	// 计算有效可计算范围内每个点的2^k邻域内的平均灰度值
	for (int i = pow(2, kmax - 1); i < height - pow(2, kmax - 1); ++i) {
		for (int j = pow(2, kmax - 1); j < width - pow(2, kmax - 1); ++j) {
			for (int k = 1; k < kmax; ++k) {
				A[k].at<float>(i, j) = cv::mean(src(cv::Rect(cv::Point(j - pow(2, k - 1) - 1, i - pow(2, k - 1) - 1), cv::Point(j + pow(2, k - 1), i + pow(2, k - 1)))))[0];
			}
		}
	}

	// 对每个像素点计算在水平和垂直方向上不重叠窗口之间的差
	for (int i = pow(2, kmax - 1); i < height - pow(2, kmax - 1); ++i) {
		for (int j = pow(2, kmax - 1); j < width - pow(2, kmax - 1); ++j) {
			for (int k = 1; k < kmax; ++k) {
				Eh[k].at<float>(i, j) = abs(A[k].at<float>(i + pow(2, k - 1), j) - A[k].at<float>(i - pow(2, k - 1), j));
				Ev[k].at<float>(i, j) = abs(A[k].at<float>(i, j + pow(2, k - 1)) - A[k].at<float>(i, j - pow(2, k - 1)));
			}
		}
	}
	// 对每个像素点计算使E达到最大值的k
	int maxkk = 0;
	for (int i = pow(2, kmax - 1); i < height - pow(2, kmax - 1); ++i) {
		for (int j = pow(2, kmax - 1); j < width - pow(2, kmax - 1); ++j) {
			int p = 1, q = 1;
			float maxEh = -1.0, maxEv = -1.0;
			for (int k = 1; k < kmax; ++k) {
				if (maxEh < Eh[k].at<float>(i, j)) {
					maxEh = Eh[k].at<float>(i, j);
					p = k;
				}
				if (maxEv < Ev[k].at<float>(i, j)) {
					maxEv = Ev[k].at<float>(i, j);
					q = k;
				}
			}
			if (maxEh > maxEv) {
				maxkk = p;
			}
			else {
				maxkk = q;
			}
			Sbest.at<uchar>(i, j) = pow(2, maxkk);
		}
	}

	// 所有Sbest的均值作为整幅图片的粗糙度
	return cv::mean(Sbest)[0];

}

// src为待处理的灰度图片
double ETVP_FEATURE_Tamura_Contrast(cv::Mat src) {
	src.convertTo(src, CV_32FC1);

	// 方差
	cv::Scalar mean, dev;
	cv::meanStdDev(src, mean, dev);
	double sigma = dev.val[0];
	double srcM = mean.val[0];

	// 四阶矩
	cv::Mat dst;
	cv::subtract(src, srcM, dst);
	cv::pow(dst, 4, dst);
	double M4 = cv::mean(dst)[0];

	// 峰度
	double alpha4 = M4 / pow(sigma, 4);

	// 对比度
	return sigma / pow(alpha4, 0.25);
}

// theta矩阵为各像素点的角度矩阵，在线性度中会用到
double ETVP_FEATURE_Tamura_Directionality(cv::Mat src, cv::Mat& theta) {
	int height = src.rows;
	int width = src.cols;

	cv::Mat deltaH(height, width, CV_32FC1);
	cv::Mat deltaV(height, width, CV_32FC1);
	src.convertTo(deltaH, deltaH.type());
	src.convertTo(deltaV, deltaV.type());

	float GradientH[3][3] = { -1, 0, 1,
		-1,  0,  1,
		-1,  0,  1
	};

	float GradientV[3][3] = { 1,  1,  1,
		0,  0,  0,
		-1,  -1,  -1
	};

	// 卷积
	cv::Mat matrixH(3, 3, CV_32FC1, GradientH), matrixV(3, 3, CV_32FC1, GradientV);
	filter2D(deltaH, deltaH, deltaH.depth(), matrixH);
	filter2D(deltaV, deltaV, deltaV.depth(), matrixV);

	// 各像素点的方向
	int n = 16, t = 12;
	std::vector<int> Ntheta(n, 0);

	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			float _fh = deltaH.at<float>(y, x);
			float _fv = deltaV.at<float>(y, x);
			if (_fh != 0.0)
			{
				theta.at<float>(y, x) = atan(_fv / _fh) + (M_PI / 2.0 + 0.001); //+0.001 because otherwise sometimes getting -6.12574e-17
			}
			for (int k = 0; k < 16; ++k) {
				if (theta.at<float>(y, x) >= (k * M_PI) / n && theta.at<float>(y, x) < ((2 * k + 1) * M_PI) / 2 / n && (fabs(_fh) + fabs(_fv) >= t)) {
					Ntheta[k]++;
				}
			}
			
		}
	}

	// 假设每幅图片只有一个方向峰值，简化了原著
	long long int s = cv::sum(Ntheta)[0];
	std::vector<double> HD(n, 0.0);
	double maxvalue = -1.0;
	int FIp = 0;
	for (int i = 0; i < n; ++i) {
		HD[i] = (double)Ntheta[i] / s;
		if (maxvalue < HD[i]) {
			FIp = i;
		}
	}
	double Fdir = 0.0;
	for (int i = 0; i < n; ++i) {
		Fdir += pow(i - FIp, 2)*HD[i];
	}
	
	return Fdir;

}

// src为灰度图片，theta为方向度返回的方向矩阵，d为共生矩阵计算时的像素间隔距离
double ETVP_FEATURE_Tamura_Linelikeness(cv::Mat src, cv::Mat theta, int d) {
	int height = src.rows;
	int width = src.cols;

	int n = 16;
	// 构造方向共生矩阵
	cv::Mat PDd0 = cv::Mat::zeros(cv::Size(n, n), CV_32S);
	cv::Mat PDd1 = cv::Mat::zeros(cv::Size(n, n), CV_32S);
	cv::Mat PDd2 = cv::Mat::zeros(cv::Size(n, n), CV_32S);
	cv::Mat PDd3 = cv::Mat::zeros(cv::Size(n, n), CV_32S);
	cv::Mat PDd4 = cv::Mat::zeros(cv::Size(n, n), CV_32S);
	cv::Mat PDd5 = cv::Mat::zeros(cv::Size(n, n), CV_32S);
	cv::Mat PDd6 = cv::Mat::zeros(cv::Size(n, n), CV_32S);
	cv::Mat PDd7 = cv::Mat::zeros(cv::Size(n, n), CV_32S);

	for (int i = d; i < height - d - 2; ++i) {
		for (int j = d; j < width - d - 2; ++j) {
			for (int m1 = 0; m1 < n; ++m1) {
				for (int m2 = 0; m2 < n; ++m2) {
					// 下方向
					if (theta.at<float>(i, j) >= m1 * M_PI / n && theta.at<float>(i, j) < (2 * m1 + 1)*M_PI / 2 / n
						&& theta.at<float>(i + d, j) >= (m2)*M_PI / n &&  theta.at<float>(i + d, j) < (2 * m2 + 1)*M_PI / 2 / n) {
						PDd0.at<int>(m1, m2)++;
					}
					// 上方向
					if (theta.at<float>(i, j) >= m1 * M_PI / n && theta.at<float>(i, j) < (2 * m1 + 1)*M_PI / 2 / n
						&& theta.at<float>(i - d, j) >= (m2)*M_PI / n &&  theta.at<float>(i - d, j) < (2 * m2 + 1)*M_PI / 2 / n) {
						PDd1.at<int>(m1, m2)++;
					}
					// 右方向
					if (theta.at<float>(i, j) >= m1 * M_PI / n && theta.at<float>(i, j) < (2 * m1 + 1)*M_PI / 2 / n
						&& theta.at<float>(i, j + d) >= (m2)*M_PI / n &&  theta.at<float>(i, j + d) < (2 * m2 + 1)*M_PI / 2 / n) {
						PDd2.at<int>(m1, m2)++;
					}
					// 左方向
					if (theta.at<float>(i, j) >= m1 * M_PI / n && theta.at<float>(i, j) < (2 * m1 + 1)*M_PI / 2 / n
						&& theta.at<float>(i, j - d) >= (m2)*M_PI / n &&  theta.at<float>(i, j - d) < (2 * m2 + 1)*M_PI / 2 / n) {
						PDd3.at<int>(m1, m2)++;
					}
					// 右下方向
					if (theta.at<float>(i, j) >= m1 * M_PI / n && theta.at<float>(i, j) < (2 * m1 + 1)*M_PI / 2 / n
						&& theta.at<float>(i + d, j + d) >= (m2)*M_PI / n &&  theta.at<float>(i + d, j + d) < (2 * m2 + 1)*M_PI / 2 / n) {
						PDd4.at<int>(m1, m2)++;
					}
					// 右上方向
					if (theta.at<float>(i, j) >= m1 * M_PI / n && theta.at<float>(i, j) < (2 * m1 + 1)*M_PI / 2 / n
						&& theta.at<float>(i - d, j + d) >= (m2)*M_PI / n &&  theta.at<float>(i - d, j + d) < (2 * m2 + 1)*M_PI / 2 / n) {
						PDd5.at<int>(m1, m2)++;
					}
					// 左下方向
					if (theta.at<float>(i, j) >= m1 * M_PI / n && theta.at<float>(i, j) < (2 * m1 + 1)*M_PI / 2 / n
						&& theta.at<float>(i + d, j - d) >= (m2)*M_PI / n &&  theta.at<float>(i + d, j - d) < (2 * m2 + 1)*M_PI / 2 / n) {
						PDd6.at<int>(m1, m2)++;
					}
					// 左上方向
					if (theta.at<float>(i, j) >= m1 * M_PI / n && theta.at<float>(i, j) < (2 * m1 + 1)*M_PI / 2 / n
						&& theta.at<float>(i - d, j - d) >= (m2)*M_PI / n &&  theta.at<float>(i - d, j - d) < (2 * m2 + 1)*M_PI / 2 / n) {
						PDd7.at<int>(m1, m2)++;
					}
				}
			}
		}
	}
	std::vector<double> f(8, 0.0);
	std::vector<double> g(8, 0.0);
	std::vector<double> tempM(8, 0.0);

	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			f[0] = f[0] + PDd0.at<int>(i, j) * cos((i - j) * 2 * M_PI / n);
			g[0] = g[0] + PDd0.at<int>(i, j);
			f[1] = f[1] + PDd1.at<int>(i, j) * cos((i - j) * 2 * M_PI / n);
			g[1] = g[1] + PDd1.at<int>(i, j);
			f[2] = f[2] + PDd2.at<int>(i, j) * cos((i - j) * 2 * M_PI / n);
			g[2] = g[2] + PDd2.at<int>(i, j);
			f[3] = f[3] + PDd3.at<int>(i, j) * cos((i - j) * 2 * M_PI / n);
			g[3] = g[3] + PDd3.at<int>(i, j);
			f[4] = f[4] + PDd4.at<int>(i, j) * cos((i - j) * 2 * M_PI / n);
			g[4] = g[4] + PDd4.at<int>(i, j);
			f[5] = f[5] + PDd5.at<int>(i, j) * cos((i - j) * 2 * M_PI / n);
			g[5] = g[5] + PDd5.at<int>(i, j);
			f[6] = f[6] + PDd6.at<int>(i, j) * cos((i - j) * 2 * M_PI / n);
			g[6] = g[6] + PDd6.at<int>(i, j);
			f[7] = f[7] + PDd7.at<int>(i, j) * cos((i - j) * 2 * M_PI / n);
			g[7] = g[7] + PDd7.at<int>(i, j);
		}
	}

	double Flin = -1.0;
	for (int i = 0; i < 8; ++i) {
		tempM[i] = f[i] / g[i];
		if (Flin < tempM[i]) {
			Flin = tempM[i];
		}
	}
	
	return Flin;
}

// src为灰度待测图，windowsize为计算规则度的子窗口大小
double ETVP_FEATURE_Tamura_Regularity(cv::Mat src, int windowsize) {
	int height = src.rows;
	int width = src.cols;

	int k = 0;
	std::vector<float> crs, con, dir, lin;
	for (int i = 0; i < height - windowsize; i += windowsize) {
		for (int j = 0; j < width - windowsize; j += windowsize) {
			k++;
			float crsV = ETVP_FEATURE_Tamura_Coarseness(src(cv::Rect(cv::Point(i, j), cv::Point(i + windowsize - 1, j + windowsize - 1))), 5);
			crs.push_back(crsV);
			float conV = ETVP_FEATURE_Tamura_Contrast(src(cv::Rect(cv::Point(i, j), cv::Point(i + windowsize - 1, j + windowsize - 1))));
			con.push_back(conV);
			cv::Mat theta;
			theta.create(cv::Size(src.cols, src.rows), CV_32FC1);
			float dirV = ETVP_FEATURE_Tamura_Directionality(src(cv::Rect(cv::Point(i, j), cv::Point(i + windowsize - 1, j + windowsize - 1))), theta);
			dir.push_back(dirV);
			float linV = ETVP_FEATURE_Tamura_Linelikeness(src(cv::Rect(cv::Point(i, j), cv::Point(i + windowsize - 1, j + windowsize - 1))), theta, 4);
			lin.push_back(linV);
		}
	}

	// 求上述各参数的标准差
	cv::Scalar means, std;
	double Dcrs, Dcon, Ddir, Dlin;
	cv::meanStdDev(crs, means, std);
	Dcrs = std.val[0];
	cv::meanStdDev(con, means, std);
	Dcon = std.val[0];
	cv::meanStdDev(dir, means, std);
	Ddir = std.val[0];
	cv::meanStdDev(lin, means, std);
	Dlin = std.val[0];

	return 1 - (Dcrs + Dcon + Ddir + Dlin) / 4 / 100;
}

double ETVP_FEATURE_Tamura_Roughness(cv::Mat src) {
	return ETVP_FEATURE_Tamura_Coarseness(src, 6) + ETVP_FEATURE_Tamura_Contrast(src);
}


int main()
{
	cv::Mat normalpic = cv::imread("D:\\QQPCmgr\\Desktop\\Data\\normalPic.bmp", 0);
	cv::Mat defectpic = cv::imread("D:\\QQPCmgr\\Desktop\\Data\\defectPic.bmp", 0);

	//double Fcrs = ETVP_FEATURE_Tamura_Coarseness(normalpic, 6);
	//cv::Mat theta;
	//theta.create(cv::Size(normalpic.cols, normalpic.rows), CV_32FC1);
	//double FDir = ETVP_FEATURE_Tamura_Directionality(normalpic, theta);
	//double Flin = ETVP_FEATURE_Tamura_Linelikeness(normalpic, theta, 4);
	//double Freg = ETVP_FEATURE_Tamura_Regularity(normalpic, 64);
	double Frgh = ETVP_FEATURE_Tamura_Roughness(normalpic);

	std::cout << Frgh << std::endl;

	// 预处理：直方图均衡
	//cv::GaussianBlur(defectpic, defectpic, cv::Size(5, 5), 0);
	cv::equalizeHist(normalpic, normalpic);
	cv::equalizeHist(defectpic, defectpic);
	//defectpic = illuminationAdaptive(defectpic);
	//cv::cvtColor(defectpic, defectpic, CV_BGR2GRAY);
	//cv::equalizeHist(defectpic, defectpic);
	//unevenLightCompensate(defectpic, 32);

	//// 梯度图像
	//cv::Mat ngrad_x, ngrad_y, ngrad;
	///// 求 X方向梯度
	//Sobel(normalpic, ngrad_x, CV_16SC1, 1, 0, 3);
	///// 求 Y方向梯度
	//Sobel(normalpic, ngrad_y, CV_16SC1, 0, 1, 3);
	//addWeighted(ngrad_x, 0.5, ngrad_y, 0.5, 0, ngrad);

	//cv::Mat grad_x, grad_y, grad;
	///// 求 X方向梯度
	//Sobel(defectpic, grad_x, CV_16SC1, 1, 0, 3);
	///// 求 Y方向梯度
	//Sobel(defectpic, grad_y, CV_16SC1, 0, 1, 3);
	//addWeighted(grad_x, 0.5, grad_y, 0.5, 0, grad);



	//// 二阶图像
	//cv::Mat nlapMat, dlapMat;
	//cv::Laplacian(normalpic, nlapMat, CV_16S, 3);
	//cv::Laplacian(defectpic, dlapMat, CV_16S, 3);

	////cv::convertScaleAbs(nlapMat, grad);
	////cv::imshow("Laplacian", grad);
	////cv::waitKey(0);

	//// 梯度图像的转换
	//cv::convertScaleAbs(nlapMat, nlapMat);
	//cv::convertScaleAbs(dlapMat, dlapMat);

	//std::vector<float> samples;

	//// 计算1000个正常图像的平均灰度(128)
	//cv::Mat normalSrc = nlapMat.clone();
	//int WindowLen1 = 16;
	//int stepLen = 10;
	//for (int i = 0; i <= normalSrc.rows - WindowLen1; i += stepLen) {
	//	for (int j = 0; j <= normalSrc.cols - WindowLen1; j += stepLen) {
	//		cv::Mat imgROI = normalSrc(cv::Rect(j, i, WindowLen1, WindowLen1));
	//		float meanValue = cv::mean(imgROI)[0];
	//		samples.push_back(meanValue);
	//		//std::cout << "i: " << i << " j: " << j << " meanValue: " << meanValue << std::endl;
	//		//cv::imshow("imgROI", imgROI);
	//		//cv::waitKey(0);
	//	}
	//}

	//cv::Scalar param = calGaussianParam(samples);
	//std::cout << "mean: " << param[0] << ", variance: " << param[1] << std::endl;

	//// 计算1000个非正常图像的平均灰度(128)
	////WindowLen1 = 32;
	//cv::Mat defectResult = dlapMat.clone();
	//stepLen = WindowLen1 - 1;
	//for (int i = 0; i <= defectResult.rows - WindowLen1; i += stepLen) {
	//	for (int j = 0; j <= defectResult.cols - WindowLen1; j += stepLen) {
	//		cv::Mat imgROI = defectResult(cv::Rect(j, i, WindowLen1, WindowLen1));
	//		//cv::equalizeHist(imgROI, imgROI);
	//		//cv::normalize(imgROI, imgROI, 0, 255, cv::NORM_MINMAX);
	//		float meanValue = cv::mean(imgROI)[0];
	//		if (meanValue > param[0] - 5 * sqrt(param[1]) && meanValue < param[0] + 5 * sqrt(param[1])) {
	//			imgROI = cv::Mat::zeros(cv::Size(imgROI.cols, imgROI.rows), imgROI.type());
	//		}
	//		//std::cout << "i: " << i << " j: " << j << " meanValue: " << meanValue << std::endl;
	//		//convertScaleAbs(imgROI, imgROI);
	//		//cv::imshow("imgROI", imgROI);
	//		//cv::waitKey(0);
	//	}
	//}

	////cv::Mat src = illuminationAdaptive(pic);
	////cv::imshow("normalPic", normalpic);
	//cv::resize(defectResult, defectResult, cv::Size(512, 512));
	//cv::imshow("defectpic", defectResult);
	//cv::waitKey(0);

	return 0;
}


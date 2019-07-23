#ifndef TAMURA_H
#define TAMURA_H

#include <iostream>
#include <opencv.hpp>

#define M_PI 3.1415926535

using namespace cv;
using namespace std;

// image为单通道的灰度图, x和y分别是image对应的坐标索引
double getLocalContrast(const Mat& image, int xpos, int ypos)
{
	int xdim = image.cols;
	int ydim = image.rows;
	int ystart = ::std::max(0, ypos - 5);
	int xstart = ::std::max(0, xpos - 5);
	int ystop = ::std::min(ydim, ypos + 6);
	int xstop = ::std::min(xdim, xpos + 6);

	int size = (ystop - ystart)*(xstop - xstart);

	double mean = 0.0, sigma = 0.0, kurtosis = 0.0, tmp = 0.0;

	for (int y = ystart; y<ystop; ++y)
	{
		for (int x = xstart; x<xstop; ++x)
		{
			tmp = image.at<uchar>(y, x);
			mean += tmp;
			sigma += tmp*tmp;
		}
	}
	mean /= size;
	sigma /= size;
	sigma -= mean*mean;
	// finish calculating mean and variance 

	for (int y = ystart; y<ystop; ++y)
	{
		for (int x = xstart; x<xstop; ++x)
		{
			tmp = image.at<uchar>(y, x) - mean;
			tmp *= tmp;
			tmp *= tmp;
			kurtosis += tmp;
		}
	}
	kurtosis /= size;
	//double alpha4=kurtosis/(sigma*sigma);
	//double contrast=sqrt(sigma)/sqrt(sqrt(alpha4));
	//return contrast;

	/// if we don't have this exception, there are numeric problems!
	/// if the region is homogeneous: kurtosis and sigma are numerically very close to zero
	if (kurtosis<numeric_limits<double>::epsilon())
	{
		return 0.0;
	}

	return sigma / sqrt(sqrt(kurtosis));
}


Mat contrast(const Mat &image)
{
	int xdim = image.cols;
	int ydim = image.rows;

	Mat result(ydim, xdim, CV_32FC1);

	double min = numeric_limits<double>::max();
	double max = 0;
	double tmp;

	for (int x = 0; x<xdim; ++x)
	{
		for (int y = 0; y<ydim; ++y)
		{
			tmp = getLocalContrast(image, x, y);
			result.at<float>(y, x) = tmp;
			min = ::std::min(min, tmp);
			max = ::std::max(max, tmp);
		}
	}

	// whether normalize
	/*
	double dMin,dMax;
	minMaxLoc( result, &dMin, &dMax, NULL, NULL);
	if ( std::abs<double>(dMax-dMin)< std::numeric_limits<double>::epsilon())
	{
	return result;
	}
	result = (result - dMin)/(dMax - dMin)*255.0;
	*/

	return result;
}


// 计算方向性
Mat directionality(const Mat &image)
{
	//init
	int xdim = image.cols;
	int ydim = image.rows;

	Mat deltaH(ydim, xdim, CV_32FC1);
	Mat deltaV(ydim, xdim, CV_32FC1);
	image.convertTo(deltaH, deltaH.type());
	image.convertTo(deltaV, deltaV.type());

	float fH[3][3] = { -1, -2, -1,
		0,  0,  0,
		1,  2,  1
	};

	float fV[3][3] = { 1,  0,  -1,
		2,  0,  -2,
		1,  0,  -1
	};

	//step1
	Mat matrixH(3, 3, CV_32FC1, fH), matrixV(3, 3, CV_32FC1, fV);
	filter2D(deltaH, deltaH, deltaH.depth(), matrixH);
	filter2D(deltaV, deltaV, deltaV.depth(), matrixV);

	//step2	
	Mat phi(ydim, xdim, CV_32FC1);
	for (int y = 0; y<ydim; ++y)
	{
		for (int x = 0; x<xdim; ++x)
		{
			float _fh = deltaH.at<float>(y, x);
			float _fv = deltaV.at<float>(y, x);
			if (_fh != 0.0)
			{
				phi.at<float>(y, x) = atan(_fv / _fh) + (M_PI / 2.0 + 0.001); //+0.001 because otherwise sometimes getting -6.12574e-17
			}
		}
	}

	return phi;

	// 关于方向性的计算还有一系列简单的近似方法
	/*
	Mat sobelx( ydim, xdim, CV_32FC1);
	Mat sobely( ydim, xdim, CV_32FC1);
	Sobel( image, sobelx, sobelx.depth(), 1, 0, 3);
	Sobel( image, sobely, sobely.depth(), 0, 1, 3);
	//magnitude( sobelx, sobely, phi);
	Mat mag, angle;
	cartToPolar( sobelx, sobely, mag, angle ); // method one
	phase( sobelx, sobely, angle); // method two
	*/
}

// chn:number of channel, k: the kth channel (default k=0)
template<typename _T> inline _T getPixelValue(const Mat& image, int x, int y, int chn, int k)
{
	return ((_T*)image.data + image.step*y / sizeof(_T))[x*chn + k];
}

template<typename _T> inline void setPixelValue(const Mat& image, int x, int y, int chn, int k, _T _val)
{
	return ((_T*)image.data + image.step*y / sizeof(_T))[x*chn + k] = _val;
}

double efficientLocalMean(const int x, const int y, const int k, const Mat &laufendeSumme)
{
	int k2 = k / 2;

	int dimx = laufendeSumme.cols;
	int dimy = laufendeSumme.rows;

	//wanting average over area: (y-k2,x-k2) ... (y+k2-1, x+k2-1)
	int starty = y - k2;
	int startx = x - k2;
	int stopy = y + k2 - 1;
	int stopx = x + k2 - 1;

	if (starty<0)
		starty = 0;
	if (startx<0)
		startx = 0;
	if (stopx>dimx - 1)
		stopx = dimx - 1;
	if (stopy>dimy - 1)
		stopy = dimy - 1;

	double unten, links, oben, obenlinks;

	if (startx - 1<0)
		links = 0;
	else
		links = laufendeSumme.at<float>(stopy, startx - 1);

	if (starty - 1<0)
		oben = 0;
	else
		oben = laufendeSumme.at<float>(starty - 1, stopx);

	if ((starty - 1 < 0) || (startx - 1 <0))
		obenlinks = 0;
	else
		obenlinks = laufendeSumme.at<float>(starty - 1, startx - 1);

	unten = laufendeSumme.at<float>(stopy, stopx);

	//   cout << "obenlinks=" << obenlinks << " oben=" << oben << " links=" << links << " unten=" <<unten << endl;
	int counter = (stopy - starty + 1)*(stopx - startx + 1);
	return (unten - links - oben + obenlinks) / counter;
}


Mat coarseness(const Mat &image)
{
	const int yDim = image.rows;
	const int xDim = image.cols;

	Mat laufendeSumme(yDim + 1, xDim + 1, CV_32FC1);

	integral(image, laufendeSumme);

	// initialize for running sum calculation
	/*  //感觉像是在计算积分图
	double links, oben, obenlinks;
	for(int y=0;y<yDim;++y)
	{
	for(int x=0;x<xDim;++x)
	{
	if(x<1)
	links=0;
	else
	links=laufendeSumme(x-1,y,0);
	if(y<1)
	oben=0;
	else
	oben=laufendeSumme(x,y-1,0);
	if(y<1 || x<1)
	obenlinks=0;
	else
	obenlinks=laufendeSumme(x-1,y-1,0);
	laufendeSumme.at<float>(y,x)=image(y,x)+links+oben-obenlinks;
	}
	}
	*/

	Mat Ak(xDim, yDim, CV_MAKETYPE(CV_32F, 5)); //CV_MAKETYPE(CV_32F,5)
	Mat Ekh(xDim, yDim, CV_MAKETYPE(CV_32F, 5));
	Mat Ekv(xDim, yDim, CV_MAKETYPE(CV_32F, 5));

	Mat Sbest(xDim, yDim, CV_32FC1);


	//step 1
	int chn = Ak.channels();
	int lenOfk = 1;
	for (int k = 1; k <= 5; ++k)
	{
		lenOfk *= 2;
		for (int y = 0; y<yDim; ++y)
		{
			float* data = (float*)Ak.data + Ak.step*y / sizeof(float);
			for (int x = 0; x<xDim; ++x)
			{
				data[chn*x + k - 1] = efficientLocalMean(x, y, lenOfk, laufendeSumme);
				//Ak(x,y,k-1)=efficientLocalMean(x,y,lenOfk,laufendeSumme);				
			}
		}
	}


	//step 2
	lenOfk = 1;
	for (int k = 1; k <= 5; ++k)
	{
		int k2 = lenOfk;
		lenOfk *= 2;
		for (int y = 0; y<yDim; ++y)
		{
			float* dataA = (float*)Ak.data + Ak.step*y / sizeof(float);
			float* dataH = (float*)Ekh.data + Ekh.step*y / sizeof(float);
			float* dataV = (float*)Ekv.data + Ekv.step*y / sizeof(float);
			for (int x = 0; x<xDim; ++x)
			{
				int posx1 = x + k2;
				int posx2 = x - k2;

				int posy1 = y + k2;
				int posy2 = y - k2;
				if (posx1<xDim && posx2 >= 0)
				{
					dataH[x*chn + k - 1] = fabs(dataA[posx1*chn + k - 1] - dataA[posx2*chn + k - 1]);
				}
				else
				{
					dataH[x*chn + k - 1] = 0;
				}
				if (posy1<yDim && posy2 >= 0)
				{

					dataV[x*chn + k - 1] = fabs(((float*)Ak.data + Ak.step*posy1 / sizeof(float))[x*chn + k - 1] - ((float*)Ak.data + Ak.step*posy2 / sizeof(float))[x*chn + k - 1]);
					//dataV[x*chn+k-1]=fabs(Ak(x,posy1,k-1)-Ak(x,posy2,k-1));
				}
				else
				{
					dataV[x*chn + k - 1] = 0;
				}
			}
		}
	}
	double sum = 0.0;

	//step3
	for (int y = 0; y<yDim; ++y)
	{
		for (int x = 0; x<xDim; ++x)
		{
			double maxE = 0;
			int maxk = 0;
			for (int k = 1; k <= 5; ++k)
			{
				float _fH = getPixelValue<float>(Ekh, x, y, chn, k - 1);
				float _fV = getPixelValue<float>(Ekv, x, y, chn, k - 1);
				if (_fH>maxE)
				{
					maxE = _fH;
					maxk = k;
				}
				if (_fV>maxE)
				{
					maxE = _fV;
					maxk = k;
				}
			}
			setPixelValue<float>(Sbest, x, y, 1, 0, maxk);
			sum += maxk;
		}
	}

	sum /= ((yDim - 32)*(xDim - 32));

	return Sbest;
}

#endif
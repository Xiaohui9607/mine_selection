#include "ImageProcess.h"

constexpr uint32_t  PICHIGHT = 2048;
constexpr uint32_t  PICWIDTH = 640;

ImageProcess::ImageProcess()
{
	m_nWidth = PICWIDTH;
	m_nHigh = PICHIGHT;
}


ImageProcess::~ImageProcess()
{
}

void ImageProcess::ReadImage(Mat &nSrc)
{
	nSrc.copyTo(m_CurrentMat);
	imshow("tt", m_CurrentMat);
	m_RValueMat = Mat::zeros(m_CurrentMat.size(), CV_16UC1);
}

void ImageProcess::ReadImage(char * nPath)
{
	m_CurrentMat = cv::imread(nPath, IMREAD_ANYDEPTH);
	if (m_CurrentMat.empty())
		return;
	imshow("tt", m_CurrentMat);
	m_RValueMat = Mat::zeros(m_CurrentMat.size(), CV_16UC1);
}

uint32_t ImageProcess::getLValue(int x, int y)
{
	return m_CurrentMat.at<ushort>(y, x);
}

uint32_t ImageProcess::getHValue(int x, int y)
{
	return m_CurrentMat.at<ushort>(y, x + m_nWidth);
}

uint32_t ImageProcess::getRValue(int x, int y)
{
	return m_RValueMat.at<ushort>(y, x);
}

void ImageProcess::calculateR(Mat & imgSrc, Mat *R)
{
	int widthOffset = imgSrc.cols / 2;
	ushort nT = 4000;
	for (int i = 0; i < imgSrc.rows; i++) {
		for (int j = 0; j < imgSrc.cols / 2; j++) {
			float TL = imgSrc.at<ushort>(i, j);
			float TH = imgSrc.at<ushort>(i, j + widthOffset);

			if (TL * TH * TL0 * TH0 > 0 && TL < (TL0 - nT) && TH < (TH0 - nT))
			{

				double dL = log(TL / TL0);
				double dH = log(TH / TH0);
				double dR = dL / dH;
				R->at<ushort>(i, j) = (ushort)(1000 * dR);
			}
			else
			{
				R->at<ushort>(i, j) = 0;
			}
		}
	}
}

void ImageProcess::getRValueMat()
{
	//calculateR(m_CurrentMat, m_pTHTLBuffer, &m_RValueMat);
	calculateR(m_CurrentMat, &m_RValueMat);
}

Mat ImageProcess::GetRValueMat()
{
	return m_RValueMat;
}


#pragma once
#include "type.hpp"

class ImageProcess
{
public:
	ImageProcess();
	~ImageProcess();
public:
	void ReadImage(Mat &nSrc);
	void ReadImage(char *nPath);
	uint32_t getLValue(int x, int y);
	uint32_t getHValue(int x, int y);
	uint32_t getRValue(int x, int y);
	void getRValueMat();
	Mat GetRValueMat();
public:
	void calculateR(cv::Mat imgSrc, ushort *Adjust, cv::Mat* R);
	void getTH0TL0(uint32_t nTH0, uint32_t nTL0);
private:
	void calculateR(Mat &imgSrc, Mat *R);
private:
	Mat m_CurrentMat;
	Mat m_RValueMat;
	float TL0{ 55000 };
	float TH0{ 55000 };
	uint32_t m_nWidth{ 640 };
	uint32_t m_nHigh{ 2048 };
	int m_PixlOffset[640]{ 0 };
};


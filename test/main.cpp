#include "type.hpp"
#include "ImageProcess.h"
#include <string>
#include <fstream>
#include <sstream>
#include<iostream>
#include <stdio.h>
#include <io.h>
#include<fstream>
#define MAX_LINE 1024   //定义txt中最大行数。可调整更改

void SetRgb()
{
	shared_ptr<ImageProcess> nImageProcessPtr = make_shared<ImageProcess>();
	//Mat src = imread("D:/ResultImage.tif", IMREAD_ANYDEPTH);
	//Mat buf = Mat::zeros(256,640, CV_16UC3);
	//ushort num = 0;
	//for (int i = 0; i < 256; i++)
	//{
	//	for (int j = 0; j < 640; j++)
	//	{
	//		buf.at<Vec3w>(i, j)[0] = src.at<WORD>(i, j);
	//		buf.at<Vec3w>(i, j)[1] = src.at<WORD>(i, 640 + j);
	//		buf.at<Vec3w>(i, j)[2] = 0;
	//		//buf.at<Vec3w>(i, j)[0];
	//		//buf.at<Vec3w>(i, j)[1];
	//		//buf.at<Vec3w>(i, j)[2];
	//	}
	//}
	//imshow("tt", buf);

	char filename[50];
	std::vector<cv::String> image_files;
	std::string pattern_jpg = "D:/test/src/*.tif";
	cv::glob(pattern_jpg, image_files);
	for (int i = 0; i < image_files.size(); i++)
	{
		cout << image_files[i].c_str() << endl;
		Mat src = imread(image_files[i].c_str(), IMREAD_ANYDEPTH);
		Mat buf = Mat::zeros(256, 640, CV_16UC3);
		nImageProcessPtr->ReadImage(src);
		nImageProcessPtr->getRValueMat();

		for (int i = 0; i < 256; i++)
		{
			for (int j = 0; j < 640; j++)
			{
				buf.at<Vec3w>(i, j)[0] = src.at<WORD>(i, j);
				buf.at<Vec3w>(i, j)[1] = src.at<WORD>(i, 640 + j);
				buf.at<Vec3w>(i, j)[2] = nImageProcessPtr->getRValue(j, i);
			}
		}
		sprintf(filename, "D:/test/dec/0 (%d).tif", i);
		imwrite(filename, buf);
	}
	waitKey(0);
}

void SetText(cv::String x)
{
	if (x.c_str() == "D:/sources/yolo_train/0 (233).txt") {
		std::cout << "fku";
	}
	char buf[MAX_LINE], temp[10000], j, temp1[500];  /*缓冲区*/
	FILE *fp;            /*文件指针*/
		int len, i = 0, l;  /*行字符个数*/
		fp = fopen(x.c_str(), "r+");
		//fp2 = fopen("D:/tem1.txt", "w+");
		if ((fp == NULL))
		{
			perror("fail to read");
			return;
		}
		while (j = fgetc(fp))
		{
			//len = strlen(buf);
			//buf[len - 1] = '\0';  /*去掉换行符*/
			//printf("%s%d ", buf, len - 1);
			temp[i] = j;
			i++;
			if (feof(fp))
				break;
		}
		temp[i] = '\0';
		//cout << "start:" << temp << endl;
		l = strlen(temp);
		int linenum = 1;
		while (linenum < l) {
			if (temp[linenum] == '7'&&temp[linenum + 1] == ' ') {
				temp[linenum - 1] = '0';
				for (int k = linenum; k < l; k++) {
					temp[k] = temp[k + 1];
				}
			}
		    if (temp[linenum] == '5'&&temp[linenum + 1] == ' ') {
				temp[linenum - 1] = '1';
				for (int k = linenum; k < l; k++) {
					temp[k] = temp[k + 1];
				}
			}
			linenum += 38;
		}
		l = strlen(temp);
		int m = 0;//check the num of '.' we found
		for (int pos = 0; pos < l; pos++) {
			if (temp[pos] == '.') {
				m++;
				if (m % 4 == 1) {
					//int t = 0;
					/*while (temp[pos] != ' ') {
						temp1[t] = temp[pos];
						pos++;
						t++;
					}*/
					strncpy(temp1, &temp[pos], 7);
					temp1[7] = '\0';
					//cout <<"temp1:"<< temp1<<endl;
					int ans;
					ans = atoi(temp1 + 1);
					ans = ans - 500000;
					ans = ans * 2;
					char change[10];
					itoa(ans, change, 10);
					/*pos = pos - 6;*/
					if (strlen(change) < 6) {
						temp[pos+1] = '0';
						pos++;
						for (int k = 0; k < 5; k++) {
							temp[pos+1] = change[k];
							pos++;
						}
					}
					else {
						for (int k = 0; k < 6; k++) {
							temp[pos+1] = change[k];
							pos++;
						}
					}
					/*		cout << "ans:" << ans << endl;;
							cout << "temp1:" << temp1 << endl;;
							cout << "change:" << change << endl;*/
				}
			}
		}
		/*for (int k = 0; k < l; k++) {
			cout << temp[k];
		}*/
		fseek(fp, 0, 0);
		cout << "end:" << temp << endl;
		fwrite(temp, 1, strlen(temp), fp);
		fclose(fp);
		i = 0;

	}


int main()
{
	//SetRgb();
	char filename[50];
	std::vector<cv::String> image_files;
	std::string pattern_jpg = "D:/sources/yolo_train/*.txt";
	cv::glob(pattern_jpg, image_files);
	for (int p = 0; p < image_files.size(); p++) {
		SetText(image_files[p]);
	}
	//Mat src = imread("D:/R.tif", IMREAD_COLOR);
	//
	//for (int i = 0; i < src.rows; i++)
	//{
	//	for (int j = 0; j < src.cols; j++)
	//	{
	//		if(src.at<Vec3b>(i, j)[0])
	//			printf ("%c\n",src.at<Vec3b>(i, j)[0]);
	//		std::this_thread::sleep_for(std::chrono::seconds(1));
	//	}
	//}
	//waitKey(0);
	system("pause");
	return 0;
}
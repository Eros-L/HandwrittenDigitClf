#ifndef _IMAGE_CORRECTION_HPP_
#define _IMAGE_CORRECTION_HPP_

#include <iostream>
#include <vector>
#include <cmath>
#include <stack>
#include <algorithm>
#include <fstream>
#include <CImg.h>
#include <Matrix.hpp>

#define uchar unsigned char


class ImageCorrection {
public:
	static cimg_library::CImg<uchar> rgb2grey(const cimg_library::CImg<uchar>&);
	static uchar otsu(const cimg_library::CImg<uchar>&);
	static cimg_library::CImg<short> hough(const cimg_library::CImg<uchar>&, const uchar);
	static std::vector<std::tuple<int, int, int>> vote(const cimg_library::CImg<short>&, const int, const int, const int = 50, const int = 50);
	static cimg_library::CImg<uchar> correct(const cimg_library::CImg<uchar>&, const int sn);
	static cimg_library::CImg<uchar> binary(const cimg_library::CImg<uchar>&, const int = 139, const double = 0.90);

private:
	static cimg_library::CImg<uchar> draw(const cimg_library::CImg<uchar>&, const uchar);
	static cimg_library::CImg<uchar> draw(const cimg_library::CImg<uchar>&, const std::vector<std::tuple<int, int, int>>&);
	static cimg_library::CImg<uchar> warp(const cimg_library::CImg<uchar>&, const std::vector<std::tuple<int, int, int>>&);
	static cimg_library::CImg<uchar> crop(const cimg_library::CImg<uchar>&);
};


#endif
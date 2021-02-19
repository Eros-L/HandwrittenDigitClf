#include <iostream>
#include <ImageCorrection.hpp>

using namespace std;
using namespace cimg_library;

#define CORRECT 0
#define BINARY 1
#define OUT 0


int main() {
	#if CORRECT
		int sn = 0;
		for (sn = 0; sn < 63; ++sn) {
			#if OUT
				ofstream fout("Result/stage1.csv", ios::app);
				fout << sn << ",";
				fout.close();
			#endif
			cout << "processing sn " << sn << endl;
			string path = "ImageData/" + to_string(sn) + ".jpg";
			CImg<uchar> img;
		 	img.load_jpeg(path.c_str());
			CImg<uchar> out = ImageCorrection::correct(img, sn);
		 	out.save_jpeg(("./Correction/" + to_string(sn)+".jpg").c_str());
		}
	#endif
	#if BINARY
		int sn = 0;
		for (sn = 0; sn < 63; ++sn) {
			cout << "processing sn " << sn << endl;
			string path = "Correction/" + to_string(sn) + ".jpg";
			CImg<uchar> img;
		 	img.load_jpeg(path.c_str());
			CImg<uchar> greyscale = ImageCorrection::rgb2grey(img).erode(5);
			CImg<uchar> out = ImageCorrection::binary(greyscale);
		 	out.save_jpeg(("./Binary/" + to_string(sn)+".jpg").c_str());
		}
	#endif
	return 0;
}

// g++ -std=c++11 main.cpp ImageCorrection.cpp Matrix.cpp -o main.exec -I /opt/X11/include -L /opt/X11/lib -lX11 -I .
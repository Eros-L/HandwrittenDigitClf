#include <ImageCorrection.hpp>

using namespace std;
using namespace cimg_library;


const int dirs[8][2] = { {-1, 0}, {-1, 1}, {0, 1}, {1, 1}, {1, 0}, {1, -1}, {0, -1}, {-1, -1} };

CImg<uchar> ImageCorrection::rgb2grey(const CImg<uchar>& src) {
    CImg<uchar> dst(src.width(), src.height(), 1, 1);
    cimg_forXY(dst, x, y) {
        dst(x, y, 0) = 0.2126*src(x, y, 0) + 0.7152*src(x, y, 1) + 0.0722*src(x, y, 2);
    }

    return dst;
}

uchar ImageCorrection::otsu(const CImg<uchar>& img) {
	// calculate the maximum pixel value and minimum pixel value of the image
	uchar maximum = img.max();
	uchar minimum = img.min();
	// greatest g, which is obtained by w0 * w1 * (u0 - u1)^2
	uchar T = 0;
	double G = 0.0;
	// traverse all t, whose value is between minimum and maximum
	for (int t = minimum; t <= maximum; ++t) {
		// estimate foreground pixels and background pixels with threshold t
		int w0 = 0, w1 = 0;
		double u0 = 0.0, u1 = 0.0;
		cimg_forXY(img, x, y) {
			if (img(x, y) > t) {
				w0 += 1;
				u0 += img(x, y);
			} else {
				w1 += 1;
				u1 += img(x, y);
			}
		}
		u0 /= w0;
		u1 /= w1;
		// update the value of optimal threshold
		double g = w0 * w1 * (u0 - u1) * (u0 - u1);
		if (g > G) {
			T = t;
			G = g;
		}
	}

	return T;
}

CImg<short> ImageCorrection::hough(const CImg<uchar>& img, const uchar T) {
	// the maximum value of rho
    int rhoMax = sqrt(img.width()*img.width()+img.height()*img.height());
    // perform frequency counts for lines in hough space
	CImg<short> houghSpace(360, rhoMax, 1, 1, 0);
	cimg_forXY(img, x, y) {
		if (img(x, y) <= T)
			continue;
    	for (int i = 0; i < 8; ++i) {
    		int xx = x + dirs[i][0], yy = y + dirs[i][1];
    		if (xx >= 0 && xx < img.width() && yy >= 0 && yy < img.height()) {
    			if (img(xx, yy) <= T) {
		            cimg_forX(houghSpace, theta) {
		                double alpha = (double(theta)/180) * M_PI;
		                int rho = int(x*cos(alpha) + y*sin(alpha));
		                if (rho >= 0 && rho < rhoMax) {
		                    ++houghSpace(theta, rho);
		                }
		            }
		            break;
    			}
    		}
    	}
    }

    return houghSpace;
}

vector<tuple<int, int, int>> ImageCorrection::vote(const CImg<short>& houghSpace, const int width, const int height, const int thres, const int dis) {
    // models chosen to represent features (models stored in format of tuple<theta, rho, freq>)
    vector<tuple<int, int, int>> models;
    cimg_forXY(houghSpace, theta, rho) {
        // only consider models having frequency larger than thres
        if (houghSpace(theta, rho, 0) > thres) {
            double alpha = (double(theta)/180) * M_PI;
            // scale of x-axis
            int x0 = (double(rho)/cos(alpha));
            int x1 = (double(rho)/cos(alpha)) - (double(height-1)*tan(alpha));
            // scale of y-axis
            int y0 = (double(rho)/sin(alpha));
            int y1 = (double(rho)/sin(alpha)) - (double(width-1)/tan(alpha));
            // ensure that the models are within the display area
            if ((x0 >= 0 && x0 < width) || (x1 >= 0 && x1 < width) || (y0 >= 0 && y0 < height) || (y1 >= 0 && y1 < height)) {
                // only sample one model in a given interval (distances between any 2 points should larger than dis)
                bool sampled = false;
                for (auto& m : models) {
                    if (sqrt((get<0>(m)-theta)*(get<0>(m)-theta) + (get<1>(m)-rho)*(get<1>(m)-rho)) < dis) {
                        sampled = true;
                        if (get<2>(m) < houghSpace(theta, rho, 0)) {
                            m = make_tuple(theta, rho, houghSpace(theta, rho, 0));
                        }
                    }
                }
                // add a new model
                if (!sampled) {
                    models.push_back(make_tuple(theta, rho, houghSpace(theta, rho, 0)));
                }
            }
        }
    }
    // eliminate rebundant model(s)
    if (models.size() > 4) {
        #define point tuple<int, int, int>
        for (vector<point>::iterator it = models.begin(); it != models.end(); ) {
            double alpha = (double(get<0>(*it))/180) * M_PI;
            if (sin(alpha) == 0.0) {
                it = models.erase(it);
            } else {
                ++it;
            }
        }
        sort(models.begin(), models.end(), [&](const point& a, const point& b) -> bool { return get<2>(a) > get<2>(b); } );
        while (models.size() > 4) {
            models.pop_back();
        }
    }

    return models;
}

CImg<uchar> ImageCorrection::correct(const CImg<uchar>& img, const int sn) {
    CImg<uchar> blur = img.get_resize(img.width()*0.08, img.height()*0.08).get_blur_median(5);

    CImg<uchar> greyscale = ImageCorrection::rgb2grey(blur);

    uchar T = ImageCorrection::otsu(greyscale);

    // ImageCorrection::draw(img, T).save_jpeg(("./Segmentation/" + to_string(sn)+".jpg").c_str());

    CImg<short> houghSpace = ImageCorrection::hough(greyscale, T);
    vector<tuple<int, int, int>> models = ImageCorrection::vote(houghSpace, greyscale.width(), greyscale.height());

    // ImageCorrection::draw(img, models).save_jpeg(("./Edge/" + to_string(sn)+".jpg").c_str());

    return ImageCorrection::warp(img, models);
}

CImg<uchar> ImageCorrection::draw(const CImg<uchar>& src, const uchar T) {
    CImg<uchar> greyscale = ImageCorrection::rgb2grey(src);
    CImg<uchar> dst(src);
    cimg_forXY(dst, x, y) {
        if (greyscale(x, y) <= T) {
            dst(x, y, 0) = 0;
            dst(x, y, 1) = 0;
            dst(x, y, 2) = 0;
        }
    }

    return dst;
}

CImg<uchar> ImageCorrection::draw(const CImg<uchar>& src, const vector<tuple<int, int, int>>& models) {
	CImg<uchar> dst(src);
    const uchar YELLOW[] = { 255, 255, 0 };
    for (auto m : models) {
        // arc angle of theta
        double alpha = (double(get<0>(m))/180) * M_PI;
        // slope
        double k = double(-1.0f) / tan(alpha);
        // intercept
        double b = double(get<1>(m)*12.5) / sin(alpha);
        // draw lines
        int x0 = -b / k;
        int x1 = (dst.height()-1-b) / k;
        int y0 = b;
        int y1 = (dst.width()-1)*k + b;
        if (abs(k) > 1) {
            dst.draw_line(x0, 0, x1, dst.height()-1, YELLOW, 1000);
        } else {
            dst.draw_line(0, y0, dst.width()-1, y1, YELLOW, 1000);
        }
        cout << "Line: y = " << k << "x + " << b << endl; 
    }

    return dst;
}

CImg<uchar> ImageCorrection::warp(const CImg<uchar>& src, const vector<tuple<int, int, int>>& models) {
    vector<pair<int, int>> corner;
    for (int i = 0; i < models.size(); ++i) {
        double alpha0 = (double(get<0>(models[i]))/180) * M_PI;
        double k0 = double(-1.0f) / tan(alpha0);
        double b0 = double(get<1>(models[i])*12.5) / sin(alpha0);
        for (int j = i+1; j < models.size(); ++j) {
            // arc angle of theta
            double alpha1 = (double(get<0>(models[j]))/180) * M_PI;
            // slope
            double k1 = double(-1.0f) / tan(alpha1);
            // intercept
            double b1 = double(get<1>(models[j])*12.5) / sin(alpha1);
            // crossover point
            double x = (b1 - b0) / (k0 - k1);
            x = min(4031.0, x);
            double y = (k0*b1 - k1*b0) / (k0 - k1);
            // detect the boundary
            if (x >= 0 && x < src.width() && y >= 0 && y < src.height()) {
                corner.push_back(make_pair(x, y));
            }
        }
    }

    int xmax = 0, ymax = 0;
    int xmin = INT_MAX, ymin = INT_MAX;
    for (int i = 0; i < 4; ++i) {
        xmax = max(xmax, corner[i].first);
        ymax = max(ymax, corner[i].second);
        xmin = min(xmin, corner[i].first);
        ymin = min(ymin, corner[i].second);
    }
    if (xmax - xmin > ymax - ymin) {
        sort(corner.begin(), corner.end(), [&](const pair<int, int>& p1, const pair<int, int>& p2) -> bool {
            return (p1.first < p2.first);
        });
        if (corner[0].second < corner[1].second) {
            swap(corner[0], corner[1]);
        }
        if (corner[2].second < corner[3].second) {
            swap(corner[2], corner[3]);
        }
    } else {
        sort(corner.begin(), corner.end(), [&](const pair<int, int>& p1, const pair<int, int>& p2) -> bool {
            return (p1.second < p2.second);
        });
        if (corner[0].first > corner[1].first) {
            swap(corner[0], corner[1]);
        }
        if (corner[2].first > corner[3].first) {
            swap(corner[2], corner[3]);
        }
    }
    
    // ofstream fout("Result/stage1.csv", ios::app);
    // for (int i = 0; i < 4; ++i) {
    //     fout << "\"" << 3000-corner[i].second << ", " << corner[i].first << "\",";
    // }
    // fout << endl;
    // fout.close();

    CImg<uchar> dst(1200, 1600, 1, 3, 0);
    vector<pair<int, int>> a4{ make_pair(0, 0), make_pair(1199, 0), make_pair(0, 1599), make_pair(1199, 1599) };
    Matrix A1(3, 3, 1), A2(3, 3, 1);
    Matrix b1(3, 3, 1), b2(3, 3, 1);
    for (int i = 0; i < 3; ++i) {
        A1[i][0] = a4[i].first; A1[i][1] = a4[i].second;
        A2[i][0] = a4[3-i].first; A2[i][1] = a4[3-i].second;

        b1[i][0] = corner[i].first; b1[i][1] = corner[i].second;
        b2[i][0] = corner[3-i].first; b2[i][1] = corner[3-i].second;
    }
    Matrix h1 = A1.inverse() * b1;
    Matrix h2 = A2.inverse() * b2;
    cimg_forXY(dst, x, y) {
        Matrix p(1, 3, 1);
        p[0][0] = x, p[0][1] = y;
        Matrix p1;
        if (double(x)/1185.0 + double(y)/1599.0 < 1.0) {
            p1 = p * h1;
        } else {
            p1 = p * h2;
        }
        int x1 = p1[0][0], y1 = p1[0][1];
        if (x1 >= 0 && x1 < src.width() && y1 >= 0 && y1 < src.height()) {
            dst(x, y, 0) = src(x1, y1, 0);
            dst(x, y, 1) = src(x1, y1, 1);
            dst(x, y, 2) = src(x1, y1, 2);
        }
    }

    return dst;
}

CImg<uchar> ImageCorrection::binary(const CImg<uchar>& src, const int block, const double ratio) {
    CImg<long> p(src);
    for (int i = 0; i < p.height(); ++i) {
        for (int j = 1; j < p.width(); ++j) {
            p(j, i) += p(j-1, i);
            if (i != 0) {
                p(j-1, i) += p(j-1, i-1);
            }
        }
        if (i != 0) {
            p(p.width()-1, i) += p(p.width()-1, i-1);
        }
    }
    CImg<uchar> dst(src);
    cimg_forXY(dst, x, y) {
        int x1 = max(0, x - block/2), x2 = min(dst.width()-1, x + block/2);
        int y1 = max(0, y - block/2), y2 = min(dst.height()-1, y + block/2);
        int count = (x2 - x1 + 1) * (y2 - y1 + 1);
        double sum = p(x2, y2);
        if (x1 != 0 && y1 != 0) {
            sum -= p(x2, y1-1) + p(x1-1, y2) - p(x1-1, y1-1);
        } else if (x1 != 0) {
            sum -= p(x1-1, y2);
        } else if (y1 != 0) {
            sum -= p(x2, y1-1);
        }
        if (dst(x, y) < ratio*sum/count) {
            dst(x, y) = 0;
        } else {
            dst(x, y) = 255;
        }
    }

    return ImageCorrection::crop(dst);
}

CImg<uchar> ImageCorrection::crop(const cimg_library::CImg<uchar>& src) {
    CImg<uchar> dst(src);
    stack<pair<int, int>> s;
    int width = dst.width(), height = dst.height();
    for (int i = 0; i < width; ++i) {
        if (dst(i, 0) == 0) {
            s.push(make_pair(i, 0));
        }
        if (dst(i, height-1) == 0) {
            s.push(make_pair(i, height-1));
        }
    }
    for (int i = 0; i < height; ++i) {
        if (dst(0, i) == 0) {
            s.push(make_pair(0, i));
        }
        if (dst(width-1, i) == 0) {
            s.push(make_pair(width-1, i));
        }
    }
    while (!s.empty()) {
        pair<int, int> p = s.top();
        s.pop();
        for (int i = 0; i < 8; ++i) {
            int a = p.first + dirs[i][0], b = p.second + dirs[i][1];
            if (a >= 0 && a < width && b >= 0 && b < height) {
                if (dst(a, b) == 0) {
                    s.push(make_pair(a, b));
                }
            }
        }
        dst(p.first, p.second) = 255;
    }

    return dst;
}

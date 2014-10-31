#ifndef SOBEL_H

#define SOBEL_H
#include <limits>
#include <stdint.h>
#include "opencv2/opencv.hpp"

typedef int16_t gradv_t;

template<class T>
class Normaliser {
    public:
        Normaliser(cv::Mat &m, int boundL, int boundH);
        void update(T v);
        cv::Mat normalise();
    private:
        T min;
        T max;
        cv::Mat &m;
        int boundL;
        int boundH;
};

class SobelFilter {
    public:
        SobelFilter(const cv::Mat &src);

        /* Return copies */
        cv::Mat getGradientX();
        cv::Mat getGradientY();
    private:
        const cv::Mat &src;

        cv::Mat kernelX;
        cv::Mat kernelY;

        cv::Mat gradientXPreS;

        cv::Mat gradientX;
        cv::Mat gradientY;

        gradv_t convolveAt(const cv::Mat &kernel, int x, int y);
        void initKernels();
        void initGradients();
        void pixelLoop();
};

#endif /* end of include guard: SOBEL_H */

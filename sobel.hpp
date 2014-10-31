#ifndef SOBEL_H

#define SOBEL_H
#include <limits>
#include <stdint.h>
#include "opencv2/opencv.hpp"


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

        float convolveAt(const cv::Mat &kernel, int x, int y);
        void initKernels();
        void initGradients();
        void pixelLoop();
};

#endif /* end of include guard: SOBEL_H */

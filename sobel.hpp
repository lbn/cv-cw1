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
        cv::Mat getMagnitude();
        cv::Mat getDirection();
        cv::Mat getGradientX(bool normalised);
        cv::Mat getGradientY(bool normalised);
        cv::Mat getMagnitude(bool normalised);
        cv::Mat getDirection(bool normalised);

    private:
        const cv::Mat &src;

        cv::Mat kernelX;
        cv::Mat kernelY;

        cv::Mat gradientXPreS;

        cv::Mat gradientX;
        cv::Mat gradientY;

        cv::Mat magnitude;
        cv::Mat direction;

        gradv_t convolveAt(const cv::Mat &kernel, int x, int y);
        void initKernels();
        void initGradients();

        void genGradients();
        void genMagAndDir();


        Normaliser<gradv_t> xnorm;
        Normaliser<gradv_t> ynorm;
        Normaliser<gradv_t> mnorm;
        Normaliser<float> dnorm;
};

#endif /* end of include guard: SOBEL_H */

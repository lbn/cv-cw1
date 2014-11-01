#include "sobel.hpp"


SobelFilter::SobelFilter(const cv::Mat &src_) 
    : src(src_),
      xnorm(gradientX,0,255),
      ynorm(gradientY,0,255),
      dnorm(direction,0,255),
      mnorm(magnitude,0,255) {
    initKernels();
    initGradients();

    genGradients();
    genMagAndDir();
}


void SobelFilter::initKernels() {
    kernelX = cv::Mat(3,3,CV_8S);
    kernelY = cv::Mat(3,3,CV_8S);

    int8_t kernelXa[3][3] = {{-1,0,1}
                            ,{-2,0,2}
                            ,{-1,0,1}};

    int8_t kernelYa[3][3] = {{1,2,1}
                            ,{0,0,0}
                            ,{-1,-2,-1}};
    for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
            kernelX.at<int8_t>(a,b) = kernelXa[a][b];
            kernelY.at<int8_t>(a,b) = kernelYa[a][b];
        }
    }
}


void SobelFilter::initGradients() {
    gradientX = cv::Mat(src.size(),CV_16S);
    gradientY = cv::Mat(src.size(),CV_16S);

    magnitude = cv::Mat(src.size(),CV_16S);

    direction = cv::Mat(src.size(),CV_32F);
}


cv::Mat SobelFilter::getGradientX() {
    return getGradientX(false);
}

cv::Mat SobelFilter::getGradientY() {
    return getGradientY(false);
}

cv::Mat SobelFilter::getMagnitude() {
    return getMagnitude(false);
}

cv::Mat SobelFilter::getDirection() {
    return getDirection(false);
}

cv::Mat SobelFilter::getGradientX(bool normalised) {
    if (normalised) {
        return xnorm.normalise();
    } else {
        return gradientX;
    }
}

cv::Mat SobelFilter::getGradientY(bool normalised) {
    if (normalised) {
        return ynorm.normalise();
    } else {
        return gradientY;
    }
}

cv::Mat SobelFilter::getMagnitude(bool normalised) {
    if (normalised) {
        return mnorm.normalise();
    } else {
        return magnitude;
    }
}

cv::Mat SobelFilter::getDirection(bool normalised) {
    if (normalised) {
        return dnorm.normalise();
    } else {
        return direction;
    }
}

template<class T>
Normaliser<T>::Normaliser(cv::Mat &m, int boundL, int boundH) : 
    m(m),
    boundL(boundL),
    boundH(boundH) {

    min = std::numeric_limits<T>::max();
    max = std::numeric_limits<T>::min();
}

template<class T>
void Normaliser<T>::update(T v) {
    if (v < min) {
        min = v;
    }
    if (v > max) {
        max = v;
    }
}

template<class T>
cv::Mat Normaliser<T>::normalise() {
    return (boundH - boundL) * (m-min) / (max-min);
}

void SobelFilter::genGradients() {
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            uint8_t p = src.at<uint8_t>(i,j);

            gradv_t gradXp = convolveAt(kernelX,i,j);
            gradv_t gradYp = convolveAt(kernelY,i,j);
    
            xnorm.update(gradXp);
            ynorm.update(gradYp);

            gradientX.at<gradv_t>(i,j) = gradXp;
            gradientY.at<gradv_t>(i,j) = gradYp;
        }
    }
}


void SobelFilter::genMagAndDir() {
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            gradv_t x = gradientX.at<gradv_t>(i,j);
            gradv_t y = gradientY.at<gradv_t>(i,j);

            gradv_t mag = sqrt(pow(x,2) + pow(y,2));
            magnitude.at<gradv_t>(i,j) = mag;

            // ???
            float th = atan2(y,x);

            mnorm.update(mag);
            dnorm.update(th);
        }
    }
}

gradv_t SobelFilter::convolveAt(const cv::Mat &kernel, int x, int y) {
    gradv_t sum = 0;

    int kxRadius = (kernel.cols - 1) / 2;
    int kyRadius = (kernel.rows - 1) / 2;
    int xMin = kxRadius, yMin = kyRadius;
    int xMax = src.rows - kxRadius, yMax = src.cols - kyRadius;

    if (x < xMax && x >= xMin && y < yMax && y >= yMin) {
        for (int m = -kxRadius; m <= kxRadius; m++) {
            for (int n = -kyRadius; n <= kyRadius; n++) {
                int px = x + m;
                int py = y + n;
                sum += (src.at<uint8_t>(px,py)
                        * kernel.at<int8_t>(m+kxRadius,n+kyRadius));
            }
        }
    }
    return sum;
}

void test(const char *filename) {
    const cv::Mat image = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
    SobelFilter sf(image);

    cv::imwrite("grad_x.jpg",sf.getGradientX(true));
    cv::imwrite("grad_y.jpg",sf.getGradientY(true));
    cv::imwrite("magnitude.jpg",sf.getMagnitude(true));
    cv::imwrite("direction.jpg",sf.getDirection(true));
}

int main(int argc, const char *argv[])
{
    if (argc == 2) {
        test(argv[1]);
    } else {
        std::cerr << "Usage: PROGNAME in.jpg" << std::endl;
        return 1;
    }
    return 0;
}

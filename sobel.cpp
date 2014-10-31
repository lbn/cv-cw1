#include "sobel.hpp"


SobelFilter::SobelFilter(const cv::Mat &src_) 
    : src(src_) {
    initKernels();
    initGradients();

    pixelLoop();
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
    gradientX = cv::Mat(src.size(),CV_32F);
    gradientY = cv::Mat(src.size(),CV_32F);
}


cv::Mat SobelFilter::getGradientX() {
    return gradientX;
}

cv::Mat SobelFilter::getGradientY() {
    return gradientY;
}

void SobelFilter::pixelLoop() {
    float gradXmin = std::numeric_limits<float>::max() , gradXmax = std::numeric_limits<float>::min();
    float gradYmin = std::numeric_limits<float>::max() , gradYmax = std::numeric_limits<float>::min();
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            uint8_t p = src.at<uint8_t>(i,j);

            float gradXp = convolveAt(kernelY.t(),i,j);
            float gradYp = convolveAt(kernelY,i,j);

            if (gradXp < gradXmin) {
                gradXmin = gradXp;
            }
            if (gradYp < gradYmin) {
                gradYmin = gradYp;
            }

            if (gradXp > gradXmax) {
                gradXmax = gradXp;
            }
            if (gradYp > gradYmax) {
                gradYmax = gradYp;
            }

            gradientX.at<float>(i,j) = gradXp;
            gradientY.at<float>(i,j) = gradYp;
        }
    }
    std::cout << "X=" << gradXmin << "   -   " << gradXmax << std::endl;
    std::cout << "Y=" << gradYmin << "   -   " << gradYmax << std::endl;
    gradientX = 255 * (gradientX-gradXmin) / (gradXmax-gradXmin);
    gradientY = 255 * (gradientY-gradYmin) / (gradYmax-gradYmin);
}

float SobelFilter::convolveAt(const cv::Mat &kernel, int x, int y) {
    float sum = 0;

    int kxRadius = (kernel.cols - 1) / 2;
    int kyRadius = (kernel.rows - 1) / 2;
    int xMin = kxRadius, yMin = kyRadius;
    int xMax = src.rows - kxRadius, yMax = src.cols - kyRadius;

    if (x <= xMax && x >= xMin && y <= yMax && y >= yMin) {
        for (int m = -kxRadius; m <= kxRadius; m++) {
            for (int n = -kyRadius; n <= kyRadius; n++) {
                sum += (src.at<uint8_t>(x + m + kxRadius, y + n + kyRadius) 
                        * kernel.at<int8_t>(m+kxRadius,n+kyRadius));
            }
        }
    }
    if (abs(sum) > 255) {
        //std::cout << "ref=x?: " << (&kernel==&kernelX) << ", SUM: " << sum << std::endl;
    }
    return sum;
}

void test(const char *filename) {
    const cv::Mat image = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
    SobelFilter sf(image);

    cv::imwrite("grad_x.jpg",sf.getGradientX());
    cv::imwrite("grad_y.jpg",sf.getGradientY());
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

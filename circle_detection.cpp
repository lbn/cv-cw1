#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>

using namespace cv;

void sobel(Mat img, Mat &dx, Mat &dy, Mat &mag, Mat &dist)
{
	float acc_dx = 0, acc_dy = 0;  				//accumulators
	float k1 [] = {-1,-2,-1,0,0,0,1,2,1}; 		//sobel kernal dx
	float k2 [] = {-1,0,1,-2,0,2,-1,0,1};		//sobel kernal dy

	for(int i=0; i<img.rows; i++) {
		for(int j=0; j<img.cols; j++) {
			acc_dx = acc_dy = 0;

			//apply kernel/mask
			for (int nn=-1; nn<2; nn++) {
				for (int mm = -1; mm < 2; mm++) {
					if (i + nn > 0 && i + nn < img.rows && j + mm > 0 && j + mm < img.cols) {
						acc_dx += (float)img.at<uchar>(i+nn,j+mm) * k1[((mm+1)*3) + nn + 1];
						acc_dy += (float)img.at<uchar>(i+nn,j+mm) * k2[((mm+1)*3) + nn + 1];
					}
				}
			}
			//write final values
			dx.at<float>(i,j) = acc_dx;
			dy.at<float>(i,j) = acc_dy;
			mag.at<float>(i,j) = (sqrtf(acc_dy*acc_dy + acc_dx*acc_dx));// > 150 ? 255 : 0;
			dist.at<float>(i,j) = atan2f(acc_dy,acc_dx);
			// printf("dist : %f \n", dist.at<float>(i,j) / 3.14159265f * 180 );
		}
	}
}



void inc_if_inside(double *** h_space, int x, int y, int height, int width, int r )
{
	if (x>0 && x<width && y> 0 && y<height)
		h_space[y][x][r]++;
}


void local_maxima3(double ***h_space, int y, int x, int r, int height, int width, int depth, double *max) {
    int search_r = 3; 

    for (int i = -search_r; i <= search_r; i++) {
        if (y+i < 0 || y+i >= height) {
            continue;
        }
        for (int j = -search_r; j <= search_r; j++) {
            if (x+j < 0 || x+j >= width) {
                continue;
            }
            for (int k = -search_r; k <= search_r; k++) {
                if (r+k < 0 || r+k >= depth) {
                    continue;
                }
                double current = h_space[y+i][x+j][r+k];
                if (*max < current) {
                    *max = current;
                }
            }
        }
    }
}


int hough_transform(Mat img_data, Mat &h_acc, Mat dist)  
{  

  //Create the accu  
  // double hough_h = ((sqrt(2.0) * (double)(h>w?h:w)) / 2.0);  
  // double _accu_h = hough_h * 2.0; // -r -> +r  
  // double _accu_w = 180;  

  // unsigned int* _accu = (unsigned int*)calloc(_accu_h * _accu_w, sizeof(unsigned int));  

  double center_x = img_data.cols/2;  
  double center_y = img_data.rows/2;  
  double DEG2RAD = 3.14159265f / 180;
  int max_rad = round(((img_data.cols>img_data.rows) ? img_data.rows : img_data.cols) / 6) ; //maximium radius


  int HEIGHT = img_data.rows;
  int WIDTH = img_data.cols;
  int DEPTH = max_rad;

  double ***h_space;

  // Allocate memory
  h_space = new double**[HEIGHT];
  for (int i = 0; i < HEIGHT; ++i) {
    h_space[i] = new double*[WIDTH];

    for (int j = 0; j < WIDTH; ++j)
      h_space[i][j] = new double[DEPTH];
  }



  for(int y=0;y<img_data.rows;y++)  
  {  
       for(int x=0;x<img_data.cols;x++)  
       {  
       	// printf("data point : %f\n", img_data.at<float>(y,x));
            if( (float) img_data.at<float>(y,x) > 250.0 )  //threshold image  
            {  	
            	for (int r=0; r<max_rad; r++)
            	{

            		int x0 = round(x + r * cos(dist.at<float>(y,x)) );
            		int x1 = round(x - r * cos(dist.at<float>(y,x)) );
            		int y0 = round(y + r * sin(dist.at<float>(y,x)) );
            		int y1 = round(y - r * sin(dist.at<float>(y,x)) );


                    inc_if_inside(h_space,x0,y0,HEIGHT, WIDTH, r);
                    inc_if_inside(h_space,x0,y1,HEIGHT, WIDTH, r);
                    inc_if_inside(h_space,x1,y0,HEIGHT, WIDTH, r);
            		inc_if_inside(h_space,x1,y1,HEIGHT, WIDTH, r);
            	}
            }  
       }  
  }  


	//sum up all r to make a 2d mat of hough space to draw.
	for(int y=0;y<img_data.rows;y++)  
	{  
		int sum_r = 0;
		for(int x=0;x<img_data.cols;x++)  
		{  	
			sum_r = 0;
			for (int r=0; r<max_rad; r++)
			{
				sum_r += h_space[y][x][r];
			}
			
			h_acc.at<float>(y,x) = sum_r > 100 ? 255 : 0;;
			// printf("sum_r: %d \n", sum_r);
			// printf("x %d, y %d, sum: %d \n",x,y, sum_r);
		}
	}

    for(int y=0;y<img_data.rows;y++)  
    {  
        int sum_r = 0;
        for(int x=0;x<img_data.cols;x++)  
        {  	
            sum_r = 0;
            for (int r=0; r<max_rad; r++)
            {
                double current = h_space[y][x][r];
                double max = current;
                local_maxima3(h_space,y,x,r,HEIGHT,WIDTH,DEPTH,&max);
                //printf("current: %f, max: %f\n",current,max);
                if (max > current) {
                    // not local maxima
                    continue;
                }
                sum_r = 101;//h_space[y][x][r];
            }

            h_acc.at<float>(y,x) = sum_r > 100 ? 255 : 0;;
            // printf("sum_r: %d \n", sum_r);
            // printf("x %d, y %d, sum: %d \n",x,y, sum_r);
        }
    }


  return 0;  
}  


int main( int argc, char** argv )
{

	char* imageName = argv[1];

	Mat image, img_grey;			//input mat
	Mat dx,dy,mag,dist;
	Mat dx_out, dy_out, dis_out;	//final output mat
	Mat h_acc, h_out;				//hough space matricies

	image = imread( imageName, 1);
	cvtColor( image, img_grey, COLOR_BGR2GRAY );

	dx.create(img_grey.rows, img_grey.cols, CV_32FC1);
	dy.create(img_grey.rows, img_grey.cols, CV_32FC1);
	mag.create(img_grey.rows, img_grey.cols, CV_32FC1);
	dist.create(img_grey.rows, img_grey.cols, CV_32FC1);

	sobel(img_grey, dx, dy, mag, dist);

	//normalize arrays with max and min values of 255 and 0
	normalize(dx, dx_out, 0, 255, NORM_MINMAX, -1, Mat());
	normalize(dy, dy_out, 0, 255, NORM_MINMAX, -1, Mat());
	normalize(dist, dis_out, 0, 255, NORM_MINMAX, -1, Mat());

	// double h_space_h = sqrt(2.0) * (double) (mag.rows>mag.cols ? mag.rows : mag.cols); //-r -> +r
	// double h_space_w = 180;	

	h_acc.create(mag.rows, mag.cols, CV_32FC1);
	// threshold(h_acc,h_out,0,255,THRESH_TOZERO);
	// threshold(h_out,h_acc,0,255,THRESH_TOZERO);


	hough_transform(mag, h_acc, dist);

	// normalize(h_acc, h_out, 0, 255, NORM_MINMAX, -1, Mat());
	// threshold(h_acc,h_out, 200,255,THRESH_TOZERO);

	//save images
	imwrite( "dx.jpg", dx_out );
	imwrite( "dy.jpg", dy_out );
	imwrite( "mag.jpg", mag );
	imwrite( "dist.jpg", dis_out );
	
	imwrite( "h_space.jpg", h_acc);
	 return 0;
}


//to install the missing opencl module 
//sudo apt-get install ocl-icd-opencl-dev 

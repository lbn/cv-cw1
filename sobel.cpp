#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <core/core.hpp>
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
			mag.at<float>(i,j) = (sqrtf(acc_dy*acc_dy + acc_dx*acc_dx)) > 200 ? 255 : 0;
			dist.at<float>(i,j) = atan2f(acc_dy,acc_dx);
			// printf("dist : %f \n", dist.at<float>(i,j) / 3.14159265f * 180 );
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
  int max_rad = round(((img_data.cols>img_data.rows) ? img_data.rows : img_data.cols) / 2) - 2; //maximium radius
  int h_space[img_data.rows][img_data.cols][max_rad];


  for(int y=0;y<img_data.rows;y++)  
  {  
       for(int x=0;x<img_data.cols;x++)  
       {  
       	// printf("data point : %f\n", img_data.at<float>(y,x));
            if( (float) img_data.at<float>(y,x) > 200.0 )  //threshold image  
            {  	
            	double 
            	for (int r=0; r<max_rad; r++)
            	{
            		
            	}
            	 // //for (int t=0; t<180; t++)
            	 // //faster version where:   <G(x,y) - 20  <   theta  <  <G(x,y) + 20
              //    for(int t= (dist.at<float>(y,x) / 3.14159265f * 180 +170); t < dist.at<float>(y,x) / 3.14159265f * 180 + 190 ; t++)  
              //    {  
              //         double r = ( ((double)x - center_x) * cos((double)t * DEG2RAD)) + (((double)y - center_y) * sin((double)t * DEG2RAD));  
              //         h_acc.at<float>((int)round(r+center_y*2),t)++;
              //    }  
            }  
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

	double h_space_h = sqrt(2.0) * (double) (mag.rows>mag.cols ? mag.rows : mag.cols); //-r -> +r
	double h_space_w = 180;	
	h_acc.create(h_space_h, h_space_w, CV_32FC1);
	threshold(h_acc,h_out,0,255,THRESH_TOZERO);
	threshold(h_out,h_acc,0,255,THRESH_TOZERO);
	hough_transform(mag, h_acc, dist);

	normalize(h_acc, h_out, 0, 255, NORM_MINMAX, -1, Mat());
	//threshold(h_out,h_acc,30,255,THRESH_TOZERO);

	//save images
	imwrite( "dx.jpg", dx_out );
	imwrite( "dy.jpg", dy_out );
	imwrite( "mag.jpg", mag );
	imwrite( "dist.jpg", dis_out );
	
	imwrite( "h_space.jpg", h_out);
	 return 0;
}


//to install the missing opencl module 
//sudo apt-get install ocl-icd-opencl-dev 

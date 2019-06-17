#include <thread>
#include <chrono>
#include <time.h>
#include <iostream>
#include <math.h>
#include "imageLoader.cpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/core/utility.hpp>

__global__ void sobelFilterGPU(cv::Mat *srcImg, cv::Mat *desImg, const unsigned int cols, const unsigned int rows){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    float dx, dy;

    if( x > 0 && y > 0 && x < cols-1 && y < rows-1) {
        dx = (-1* srcImg->data[(y-1)*cols + (x-1)]) + (-2*srcImg->data[y*cols+(x-1)]) + (-1*srcImg->data[(y+1)*cols+(x-1)]) +
             (    srcImg->data[(y-1)*cols + (x+1)]) + ( 2*srcImg->data[y*cols+(x+1)]) + (   srcImg->data[(y+1)*cols+(x+1)]);
             
        dy = (    srcImg[(y-1)*cols + (x-1)]) + ( 2*srcImg[(y-1)*cols+x]) + (   srcImg[(y-1)*cols+(x+1)]) +
             (-1* srcImg[(y+1)*cols + (x-1)]) + (-2*srcImg[(y+1)*cols+x]) + (-1*srcImg[(y+1)*cols+(x+1)]);
        
        desImg[y*cols + x] = sqrt( (dx*dx) + (dy*dy) );
    }

}

#define GRIDVAL 20.0 

int main(int argc, char * argv[]){
    if(argc != 2){
        std::cout << argv[0] << "Invalid number of command line arguments. Exiting program" << std::endl;
        std::cout << "Usage: " << argv[0] << " [image.png]"<< std::endl;
        return 1;
    }
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    int cores = devProp.multiProcessorCount;
    switch (devProp.major){
	case 2: // Fermi
		if (devProp.minor == 1) cores *= 48;
		else cores *= 32; break;
	case 3: // Kepler
		cores *= 192; break;
	case 5: // Maxwell
		cores *= 128; break;
	case 6: // Pascal
		if (devProp.minor == 1) cores *= 128;
		else if (devProp.minor == 0) cores *= 64;
		break;
    }
    time_t rawTime;time(&rawTime);
    struct tm* curTime = localtime(&rawTime);
    char timeBuffer[80] = "";
    strftime(timeBuffer, 80, "edge map benchmarks (%c)\n", curTime);
    std::cout << timeBuffer << std::endl;
    std::cout << "GPGPU: " << devProp.name << ", CUDA "<< devProp.major << "."<< devProp.minor <<", "<< devProp.totalGlobalMem / 1048576 << 
                " Mbytes global memory, "<< cores << " CUDA cores\n" <<std::endl;
    std::cout << "OpenCV Version: " << CV_VERSION << std::endl;

    cv::Mat origImg = cv::imread(argv[1]);
    cv::Mat * destImg;

    const size_t size = sizeof(origImg);
    cudaMalloc((void **)&destImg, size);
    cudaMemcpy(destImg, &origImg, size,cudaMemcpyHostToDevice);



    return 0;
}
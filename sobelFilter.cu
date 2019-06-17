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

#define GRIDVAL 20.0 

__global__ void sobelFilterGPU(unsigned char* orig, unsigned char* cpu, const unsigned int width, const unsigned int height){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    float dx, dy;
    if( x > 0 && y > 0 && x < width-1 && y < height-1) {
        dx = (-1* orig[(y-1)*width + (x-1)]) + (-2*orig[y*width+(x-1)]) + (-1*orig[(y+1)*width+(x-1)]) +
             (    orig[(y-1)*width + (x+1)]) + ( 2*orig[y*width+(x+1)]) + (   orig[(y+1)*width+(x+1)]);
             
        dy = (    orig[(y-1)*width + (x-1)]) + ( 2*orig[(y-1)*width+x]) + (   orig[(y-1)*width+(x+1)]) +
             (-1* orig[(y+1)*width + (x-1)]) + (-2*orig[(y+1)*width+x]) + (-1*orig[(y+1)*width+(x+1)]);
        
        cpu[y*width + x] = sqrt( (dx*dx) + (dy*dy) );
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
    cv::cvtColor(origImg, origImg, cv::COLOR_RGB2GRAY);
    
    unsigned char *gpu_orig, *gpu_sobel;
    cudaMalloc( (void**)&gpu_orig, (origImg.cols * origImg.rows));
    cudaMalloc( (void**)&gpu_sobel, (origImg.cols * origImg.rows));

    cudaMemcpy(gpu_orig, origImg.data, (origImg.cols*origImg.rows), cudaMemcpyHostToDevice);
    cudaMemset(gpu_sobel, 0, (origImg.cols*origImg.rows));

    dim3 threadsPerBlock(GRIDVAL, GRIDVAL, 1);
    dim3 numBlocks(ceil(origImg.cols/GRIDVAL), ceil(origImg.rows/GRIDVAL), 1);
    
    auto c = std::chrono::system_clock::now();

    sobelFilterGPU<<<numBlocks, threadsPerBlock>>>(gpu_orig, gpu_sobel, origImg.cols, origImg.rows);

    cudaError_t cudaerror = cudaDeviceSynchronize(); // waits for completion, returns error code
    if ( cudaerror != cudaSuccess ) fprintf( stderr, "Cuda failed to synchronize: %s\n", cudaGetErrorName( cudaerror ) ); // if error, output error
    std::chrono::duration<double> time_gpu = std::chrono::system_clock::now() - c;

    cudaMemcpy(origImg. data, gpu_sobel, (origImg.cols*origImg.rows), cudaMemcpyDeviceToHost);

    /** Output runtimes of each method of sobel filtering **/
    std::cout << "\nProcessing "<< argv[1] << ": "<<origImg.rows<<" rows x "<<origImg.cols << " columns" << std::endl;
    std::cout << "CUDA execution time   = " << 1000*time_gpu.count() <<" msec"<<std::endl;

    cv::imwrite("outImg.png",origImg);    
    cudaFree(gpu_orig); cudaFree(gpu_sobel);

    return 0;
}
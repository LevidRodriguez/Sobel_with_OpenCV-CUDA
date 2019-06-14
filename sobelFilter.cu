/*************************************************************************************************
 * File: sobelFilter.cu
 * Date: 09/27/2017
 * 
 * Compiling: Requires a Nvidia CUDA capable graphics card and the Nvidia GPU Computing Toolkit.
 *            Linux: nvcc -Wno-deprecated-gpu-targets -O3 -o edge sobelFilter.cu lodepng.cpp -Xcompiler -fopenmp
 * 
 * Usage:   Linux: >> edge [filename.png]
 * 
 * Description: This file is meant to handle all the sobel filter functions as well as the main
 *      function. Each sobel filter function runs in a different way than the others, one is a basic
 *      sobel filter running through just the cpu on a single thread, another runs through openmp 
 *      to parallelize the single thread cpu function, and the last one runs through a NVIDIA gpu
 *      to parallelize the function onto the many cores available on the gpu.
 *************************************************************************************************/

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

/************************************************************************************************
 * void sobel_gpu(const byte*, byte*, uint, uint);
 * - This function runs on the GPU, it works on a 2D grid giving the current x, y pair being worked
 * - on, the const byte* is the original image being processed and the second byte* is the image
 * - being created using the sobel filter. This function runs through a given x, y pair and uses 
 * - a sobel filter to find whether or not the current pixel is an edge, the more of an edge it is
 * - the higher the value returned will be
 * 
 * Inputs: const byte* orig : the original image being evaluated
 *                byte* cpu : the image being created using the sobel filter
 *               uint width : the width of the image
 *              uint height : the height of the image
 * 
 ***********************************************************************************************/
__global__ void sobel_gpu(const byte* orig, byte* cpu, const unsigned int width, const unsigned int height) {
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
/************************************************************************************************
 * int main(int, char*[])
 * - This function is our program's entry point. The function passes in the command line arguments
 * - and if there are exactly 2 command line arguments, the program will continue, otherwise it
 * - will exit with error code 1. If the program continues, it will read in the file given by 
 * - command line argument #2 and store as an array of bytes, after some header information is
 * - outputted, the sobel filter will run in 3 different functions on the original image and
 * - 3 new images will be created, each containing a sobel filter created using just the CPU, 
 * - OMP, and the GPU, then the image will be written out to a file with an appropriate indicator
 * - appended to the end of the filename.
 * 
 * Inputs:    int argc : the number of command line arguments
 *         char*argv[] : an array containing the command line arguments
 * Outputs:   returns 0: code ran successful, no issues came up
 *            returns 1: invalid number of command line arguments
 *            returns 2: unable to process input image
 *            returns 3: unable to write output image
 * 
 ***********************************************************************************************/
int main(int argc, char*argv[]) {
    /** Check command line arguments **/
    if(argc != 2) {
        printf("%s: Invalid number of command line arguments. Exiting program\n", argv[0]);
        printf("Usage: %s [image.png]", argv[0]);
        return 1;
    }
    /** Gather CUDA device properties **/
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, 0);
	int cores = devProp.multiProcessorCount;
	switch (devProp.major)
	{
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
    
    /** Print out some header information (# of hardware threads, GPU info, etc) **/
    time_t rawTime;time(&rawTime);
    struct tm* curTime = localtime(&rawTime);
    char timeBuffer[80] = "";
    strftime(timeBuffer, 80, "edge map benchmarks (%c)\n", curTime);
    std::cout << timeBuffer << std::endl;
    std::cout << "GPGPU: " << devProp.name << ", CUDA "<< devProp.major << "."<< devProp.minor <<", "<< devProp.totalGlobalMem / 1048576 << 
                " Mbytes global memory, "<< cores << " CUDA cores\n" <<std::endl;
    std::cout << "OpenCV Version: " << CV_VERSION << std::endl;

    /** Load our img and allocate space for our modified images **/
    imgData origImg = loadImage(argv[1]);
    // imgData origImg = cv::imread(argv[1]);
    
    imgData gpuImg(new byte[origImg.width*origImg.height], origImg.width, origImg.height);
    
    /** Use the GPU to parallelize it further **/
    /** Allocate space in the GPU for our original img, new img, and dimensions **/
    // byte *gpu_orig, *gpu_sobel;
    cv::Mat *gpu_orig, *gpu_sobel;
    cudaMalloc( (void**)&gpu_orig, (origImg.width * origImg.height));
    cudaMalloc( (void**)&gpu_sobel, (origImg.width * origImg.height));
    /** Transfer over the memory from host to device and memset the sobel array to 0s **/
    cudaMemcpy(gpu_orig, origImg.pixels, (origImg.width*origImg.height), cudaMemcpyHostToDevice);
    cudaMemset(gpu_sobel, 0, (origImg.width*origImg.height));
   
    /** set up the dim3's for the gpu to use as arguments (threads per block & num of blocks)**/
    dim3 threadsPerBlock(GRIDVAL, GRIDVAL, 1);
    dim3 numBlocks(ceil(origImg.width/GRIDVAL), ceil(origImg.height/GRIDVAL), 1);

    /** We first run the sobel filter on just the CPU using only 1 thread **/
    auto c = std::chrono::system_clock::now();
    /** Run the sobel filter using the CPU **/
    sobel_gpu<<<numBlocks, threadsPerBlock>>>(gpu_orig, gpu_sobel, origImg.width, origImg.height);
    cudaError_t cudaerror = cudaDeviceSynchronize(); // waits for completion, returns error code
    if ( cudaerror != cudaSuccess ) fprintf( stderr, "Cuda failed to synchronize: %s\n", cudaGetErrorName( cudaerror ) ); // if error, output error
    std::chrono::duration<double> time_gpu = std::chrono::system_clock::now() - c;
    /** Copy data back to CPU from GPU **/
    cudaMemcpy(gpuImg.pixels, gpu_sobel, (origImg.width*origImg.height), cudaMemcpyDeviceToHost);

    /** Output runtimes of each method of sobel filtering **/
    std::cout << "\nProcessing "<< argv[1] << ": "<<origImg.height<<" rows x "<<origImg.width << " columns" << std::endl;
    // printf(, ,  );
    std::cout << "CUDA execution time   = " << 1000*time_gpu.count() <<" msec"<<std::endl;
    // printf(, 5, );
    // printf("\n");

    /** Output the images of each sobel filter with an appropriate string appended to the original image name **/
    writeImage(argv[1], "gpu", gpuImg);
    
    /** Free any memory leftover.. gpuImig, cpuImg, and ompImg get their pixels free'd while writing **/
    cudaFree(gpu_orig); cudaFree(gpu_sobel);
    return 0;
}
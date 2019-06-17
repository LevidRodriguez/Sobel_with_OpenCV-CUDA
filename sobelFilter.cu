#include <thread>
#include <chrono>
#include <time.h>
#include <iostream>
#include <math.h>
// #include "imageLoader.cpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/core/utility.hpp>

#define GridSize 20.0 
void sobel_cpu(unsigned char* orig, unsigned char* cpu, const unsigned int width, const unsigned int height);

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

int main(int argc, char * argv[]){
    if(argc != 2){
        std::cout << argv[0] << "Invalid number of command line arguments. Exiting program" << std::endl;
        std::cout << "Usage: " << argv[0] << " [image.png]"<< std::endl;
        return 1;
    }
    // Verify GPU, CUDA and OpenCV version
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
    // Load Image
    cv::Mat origImg = cv::imread(argv[1]);
    cv::cvtColor(origImg, origImg, cv::COLOR_RGB2GRAY);
    unsigned char *gpu_orig, *gpu_sobel, *cpu_sobel;
    auto c = std::chrono::system_clock::now();
    sobel_cpu(origImg.data, cpu_sobel, origImg.cols, origImg.cols);
    std::chrono::duration<double> time_cpu = std::chrono::system_clock::now() - c;    

    // Allocate memory for the images in GPU memory 
    cudaMalloc( (void**)&gpu_orig, (origImg.cols * origImg.rows));
    cudaMalloc( (void**)&gpu_sobel, (origImg.cols * origImg.rows));
    // Transfiera del host al device y configura la matriz resultante a 0s
    cudaMemcpy(gpu_orig, origImg.data, (origImg.cols*origImg.rows), cudaMemcpyHostToDevice);
    cudaMemset(gpu_sobel, 0, (origImg.cols*origImg.rows));
    // configura los dim3 para que gpu los use como argumentos, hilos por bloque y número de bloques
    dim3 threadsPerBlock(GridSize, GridSize, 1);
    dim3 numBlocks(ceil(origImg.cols/GridSize), ceil(origImg.rows/GridSize), 1);
    
    // Ejecutar el filtro sobel utilizando la CPU.
    c = std::chrono::system_clock::now();
    sobelFilterGPU<<<numBlocks, threadsPerBlock>>>(gpu_orig, gpu_sobel, origImg.cols, origImg.rows);
    cudaError_t cudaerror = cudaDeviceSynchronize(); // waits for completion, returns error code
    // if error, output error
    if ( cudaerror != cudaSuccess ) 
        std::cout <<  "Cuda failed to synchronize: " << cudaGetErrorName( cudaerror ) <<std::endl;
    
    std::chrono::duration<double> time_gpu = std::chrono::system_clock::now() - c;
    // Copia los datos al CPU desde la GPU, del device al host
    cudaMemcpy(origImg. data, gpu_sobel, (origImg.cols*origImg.rows), cudaMemcpyDeviceToHost);

    /** Output runtimes of each method of sobel filtering **/
    std::cout << "\nProcessing "<< argv[1] << ": "<<origImg.rows<<" rows x "<<origImg.cols << " columns" << std::endl;
    std::cout << "CPU execution time   = " << 1000*time_cpu.count() <<" msec"<<std::endl;
    std::cout << "CUDA execution time   = " << 1000*time_gpu.count() <<" msec"<<std::endl;

    // Save results
    cv::imwrite("outImg.png",origImg);    
    cudaFree(gpu_orig); cudaFree(gpu_sobel);

    return 0;
}

void sobel_cpu(unsigned char* orig, unsigned char* cpu,const unsigned int width, const unsigned int height) {
    for(int y = 1; y < height-1; y++) {
        for(int x = 1; x < width-1; x++) {
            int dx = (-1*orig[(y-1)*width + (x-1)]) + (-2*orig[y*width+(x-1)]) + (-1*orig[(y+1)*width+(x-1)]) +
                     (orig[(y-1)*width + (x+1)]) + (2*orig[y*width+(x+1)]) + (orig[(y+1)*width+(x+1)]);
            int dy = (orig[(y-1)*width + (x-1)]) + (2*orig[(y-1)*width+x]) + (orig[(y-1)*width+(x+1)]) +
                (-1*orig[(y+1)*width + (x-1)]) + (-2*orig[(y+1)*width+x]) + (-1*orig[(y+1)*width+(x+1)]);
                cpu[y*width + x] = sqrt((dx*dx)+(dy*dy));
            }
        }
    }
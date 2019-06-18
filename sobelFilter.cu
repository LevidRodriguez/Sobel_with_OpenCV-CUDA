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
void sobelFilterCPU(cv::Mat srcImg, cv::Mat dstImg, const unsigned int width, const unsigned int height);
void sobelFilterOpenCV(cv::Mat srcImg, cv::Mat dstImg);

__global__ void sobelFilterGPU(unsigned char* srcImg, unsigned char* dstImg, const unsigned int width, const unsigned int height){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    float dx, dy;
    if( x > 0 && y > 0 && x < width-1 && y < height-1) {
        dx = (-1* srcImg[(y-1)*width + (x-1)]) + (-2*srcImg[y*width+(x-1)]) + (-1*srcImg[(y+1)*width+(x-1)]) +
             (    srcImg[(y-1)*width + (x+1)]) + ( 2*srcImg[y*width+(x+1)]) + (   srcImg[(y+1)*width+(x+1)]);
             
        dy = (    srcImg[(y-1)*width + (x-1)]) + ( 2*srcImg[(y-1)*width+x]) + (   srcImg[(y-1)*width+(x+1)]) +
             (-1* srcImg[(y+1)*width + (x-1)]) + (-2*srcImg[(y+1)*width+x]) + (-1*srcImg[(y+1)*width+(x+1)]);
        
        dstImg[y*width + x] = sqrt( (dx*dx) + (dy*dy) );
    }
}

int main(int argc, char * argv[]){
    if(argc != 2){
        std::cout << argv[0] << "Invalid number of command line arguments. Exiting program" << std::endl;
        std::cout << "Usage: " << argv[0] << " [image.png]"<< std::endl;
        return 1;
    }
    // Verifica las versiones de GPU, CUDA y OpenCV.
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
    
    // Cargar imagen y la transforma a escala de grises
    cv::Mat srcImg = cv::imread(argv[1]); 
    cv::cvtColor(srcImg, srcImg, cv::COLOR_RGB2GRAY);
    cv::Mat sobel_cpu = cv::Mat::zeros(srcImg.size(),srcImg.type());
    cv::Mat sobel_opencv = cv::Mat::zeros(srcImg.size(), srcImg.type());
    
    unsigned char *gpu_orig, *gpu_sobel;
    auto c = std::chrono::system_clock::now();
    /******************************************---START CPU---****************************************************/
    sobelFilterCPU(srcImg, sobel_cpu, srcImg.cols, srcImg.rows);
    std::chrono::duration<double> time_cpu = std::chrono::system_clock::now() - c;    
    /******************************************---END CPU---******************************************************/
    
    /******************************************---START OPENCV---****************************************************/
    c = std::chrono::system_clock::now();
    sobelFilterOpenCV(srcImg, sobel_opencv);
    std::chrono::duration<double> time_opencv = std::chrono::system_clock::now() - c;    
    /******************************************---END OPENCV---******************************************************/

    /******************************************---SETUP GPU---****************************************************/
    // Asignar memoria para las imágenes en memoria GPU.
    cudaMalloc( (void**)&gpu_orig, (srcImg.cols * srcImg.rows));
    cudaMalloc( (void**)&gpu_sobel, (srcImg.cols * srcImg.rows));
    // Transfiera del host al device y configura la matriz resultante a 0s
    cudaMemcpy(gpu_orig, srcImg.data, (srcImg.cols*srcImg.rows), cudaMemcpyHostToDevice);
    cudaMemset(gpu_sobel, 0, (srcImg.cols*srcImg.rows));
    // configura los dim3 para que gpu los use como argumentos, hilos por bloque y número de bloques
    dim3 threadsPerBlock(GridSize, GridSize, 1);
    dim3 numBlocks(ceil(srcImg.cols/GridSize), ceil(srcImg.rows/GridSize), 1);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    /******************************************---START GPU---****************************************************/
    // Ejecutar el filtro sobel utilizando la GPU.
    c = std::chrono::system_clock::now();
    sobelFilterGPU<<< numBlocks, threadsPerBlock, 0, stream >>>(gpu_orig, gpu_sobel, srcImg.cols, srcImg.rows);
    cudaError_t cudaerror = cudaDeviceSynchronize(); // waits for completion, returns error code
    // if error, output error
    if ( cudaerror != cudaSuccess ) 
        std::cout <<  "Cuda failed to synchronize: " << cudaGetErrorName( cudaerror ) <<std::endl;
    std::chrono::duration<double> time_gpu = std::chrono::system_clock::now() - c;
    /******************************************---END GPU---****************************************************/
    // Copia los datos al CPU desde la GPU, del device al host
    cudaMemcpy(srcImg. data, gpu_sobel, (srcImg.cols*srcImg.rows), cudaMemcpyDeviceToHost);
    /** Tiempos de ejecución de cada método de filtrado por sobel **/
    std::cout << "\nProcessing "<< argv[1] << ": "<<srcImg.rows<<" rows x "<<srcImg.cols << " columns" << std::endl;
    std::cout << "CPU execution time   = " << 1000*time_cpu.count() <<" msec"<<std::endl;
    std::cout << "OPENCV execution time   = " << 1000*time_opencv.count() <<" msec"<<std::endl;
    std::cout << "CUDA execution time   = " << 1000*time_gpu.count() <<" msec"<<std::endl;
    // Save results
    cv::imwrite("outImgCPU.png",sobel_cpu);    
    cv::imwrite("outImgOpenCV.png",sobel_opencv);
    cv::imwrite("outImgGPU.png",srcImg);
    cudaStreamDestroy(stream);    
    cudaFree(gpu_orig); cudaFree(gpu_sobel);

    return 0;
}

void sobelFilterCPU(cv::Mat srcImg, cv::Mat dstImg, const unsigned int width, const unsigned int height){
    // cv::cvtColor(orig, cpu, cv::COLOR_RGB2GRAY);
    for(int y = 1; y < srcImg.rows-1; y++) {
        for(int x = 1; x < srcImg.cols-1; x++) {
            int dx = (-1*srcImg.data[(y-1)*width + (x-1)]) + (-2*srcImg.data[y*width+(x-1)]) + (-1*srcImg.data[(y+1)*width+(x-1)]) +
            (srcImg.data[(y-1)*width + (x+1)]) + (2*srcImg.data[y*width+(x+1)]) + (srcImg.data[(y+1)*width+(x+1)]);
            int dy = (srcImg.data[(y-1)*width + (x-1)]) + (2*srcImg.data[(y-1)*width+x]) + (srcImg.data[(y-1)*width+(x+1)]) +
            (-1*srcImg.data[(y+1)*width + (x-1)]) + (-2*srcImg.data[(y+1)*width+x]) + (-1*srcImg.data[(y+1)*width+(x+1)]);
            // int sum = abs(dx) + abs(dy);
            int sum = sqrt((dx*dx)+(dy*dy));
            // sum = sum>255?255:sum;
            // cpu[y*width + x] = sqrt((dx*dx)+(dy*dy));
            dstImg.at<uchar>(y,x) = sum;
        }
    }
}

void sobelFilterOpenCV(cv::Mat srcImg, cv::Mat dstImg){
    cv::Mat grad_x, grad_y, abs_grad_x, abs_grad_y;
    // Gradiente X
    cv::Sobel(srcImg, grad_x, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
    cv::convertScaleAbs(grad_x, abs_grad_x);
    // Gradiente Y
    cv::Sobel(srcImg, grad_y, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);
    cv::convertScaleAbs(grad_y, abs_grad_y);

    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dstImg );
}
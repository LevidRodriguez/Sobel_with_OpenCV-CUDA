#include <thread>
#include <chrono>
#include <time.h>
#include <iostream>
#include <math.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/core/utility.hpp>

// Numero de hilos por bloque
#define threadsNumber 30.0 

void sobelFilterCPU(cv::Mat srcImg, cv::Mat dstImg, const unsigned int width, const unsigned int height);
void sobelFilterOpenCV(cv::Mat srcImg, cv::Mat dstImg);

__global__ void sobelFilterGPU(unsigned char* srcImg, unsigned char* dstImg, const unsigned int width, const unsigned int height){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if( x > 0 && y > 0 && x < width-1 && y < height-1) {
        float dx = (-1* srcImg[(y-1)*width + (x-1)]) + (-2*srcImg[y*width+(x-1)]) + (-1*srcImg[(y+1)*width+(x-1)]) +
             (    srcImg[(y-1)*width + (x+1)]) + ( 2*srcImg[y*width+(x+1)]) + (   srcImg[(y+1)*width+(x+1)]);
             
        float dy = (    srcImg[(y-1)*width + (x-1)]) + ( 2*srcImg[(y-1)*width+x]) + (   srcImg[(y-1)*width+(x+1)]) +
             (-1* srcImg[(y+1)*width + (x-1)]) + (-2*srcImg[(y+1)*width+x]) + (-1*srcImg[(y+1)*width+(x+1)]);
        dstImg[y*width + x] = sqrt( (dx*dx) + (dy*dy) ) > 255 ? 255 : sqrt( (dx*dx) + (dy*dy) );
    }
}

int main(int argc, char * argv[]){
    if(argc != 2){
        std::cout << argv[0] << "Invalid number of command line arguments. Exiting program" << std::endl;
        std::cout << "Usage: " << argv[0] << " [image.png]"<< std::endl;
        return 1;
    }
    // Verifica las versiones de GPU, CUDA y OpenCV.
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    
    time_t rawTime; time(&rawTime);
    struct tm* curTime = localtime(&rawTime);
    char timeBuffer[80] = "";
    strftime(timeBuffer, 80, "---------- %c ----------", curTime);
    std::cout << timeBuffer << std::endl;
    
    std::cout << "GPU: " << deviceProp.name << ", CUDA "<< deviceProp.major << "."<< deviceProp.minor <<", "<< deviceProp.totalGlobalMem / 1048576 << 
                " Mbytes " <<std::endl; //<< cores << " CUDA cores\n" <<std::endl;
    std::cout << "OpenCV Version: " << CV_VERSION << std::endl;
    
    // Cargar imagen y la transforma a escala de grises
    cv::Mat srcImg = cv::imread(argv[1]); 
    cv::cvtColor(srcImg, srcImg, cv::COLOR_RGB2GRAY);
    cv::Mat sobel_cpu = cv::Mat::zeros(srcImg.size(),srcImg.type());
    cv::Mat sobel_opencv = cv::Mat::zeros(srcImg.size(), srcImg.type());

    unsigned char *gpu_src, *gpu_sobel;
    auto start_time = std::chrono::system_clock::now();
    // ---START CPU
    sobelFilterCPU(srcImg, sobel_cpu, srcImg.cols, srcImg.rows);
    std::chrono::duration<double> time_cpu = std::chrono::system_clock::now() - start_time;    
    // ---END CPU
    
    // ---START OPENCV
    start_time = std::chrono::system_clock::now();
    sobelFilterOpenCV(srcImg, sobel_opencv);
    std::chrono::duration<double> time_opencv = std::chrono::system_clock::now() - start_time;    
    // ---END OPENCV

    // ---SETUP GPU
    // Eventos
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //Streams
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    // Asignar memoria para las imágenes en memoria GPU.
    cudaMalloc( (void**)&gpu_src, (srcImg.cols * srcImg.rows));
    cudaMalloc( (void**)&gpu_sobel, (srcImg.cols * srcImg.rows));

    // Transfiera del host al device y configura la matriz resultante a 0s
    cudaMemcpy(gpu_src, srcImg.data, (srcImg.cols*srcImg.rows), cudaMemcpyHostToDevice);
    cudaMemset(gpu_sobel, 0, (srcImg.cols*srcImg.rows));

    // configura los dim3 para que el gpu los use como argumentos, hilos por bloque y número de bloques
    dim3 threadsPerBlock(threadsNumber, threadsNumber, 1);
    dim3 numBlocks(ceil(srcImg.cols/threadsNumber), ceil(srcImg.rows/threadsNumber), 1);
    
    // ---START GPU
    // Ejecutar el filtro sobel utilizando la GPU.
    cudaEventRecord(start);
    start_time = std::chrono::system_clock::now();
    sobelFilterGPU<<< numBlocks, threadsPerBlock, 0, stream >>>(gpu_src, gpu_sobel, srcImg.cols, srcImg.rows);
    cudaError_t cudaerror = cudaDeviceSynchronize(); // waits for completion, returns error code
    // if error, output error
    if ( cudaerror != cudaSuccess ) 
        std::cout <<  "Cuda failed to synchronize: " << cudaGetErrorName( cudaerror ) <<std::endl;
    std::chrono::duration<double> time_gpu = std::chrono::system_clock::now() - start_time;
    // ---END GPU
    
    // Copia los datos al CPU desde la GPU, del device al host
    cudaMemcpy(srcImg.data, gpu_sobel, (srcImg.cols*srcImg.rows), cudaMemcpyDeviceToHost);
    // Libera recursos
    cudaEventRecord(stop);
    float time_milliseconds =0;
    cudaEventElapsedTime(&time_milliseconds, start, stop);
    cudaStreamDestroy(stream);    
    cudaFree(gpu_src); 
    cudaFree(gpu_sobel);
    /** Tiempos de ejecución de cada método de filtrado por sobel **/
    std::cout << "Archivo: "<< argv[1] << ": "<<srcImg.rows<<" rows x "<<srcImg.cols << " columns" << std::endl;
    std::cout << "CPU execution time   = " << 1000*time_cpu.count() <<" msec"<<std::endl;
    std::cout << "OPENCV execution time   = " << 1000*time_opencv.count() <<" msec"<<std::endl;
    std::cout << "CUDA execution time   = " << 1000*time_gpu.count() <<" msec"<<std::endl;
    
    // Guarda resultados
    cv::imwrite("outImgCPU.png",sobel_cpu);    
    cv::imwrite("outImgOpenCV.png",sobel_opencv);
    cv::imwrite("outImgGPU.png",srcImg);

    return 0;
}

void sobelFilterCPU(cv::Mat srcImg, cv::Mat dstImg, const unsigned int width, const unsigned int height){
    for(int y = 1; y < srcImg.rows-1; y++) {
        for(int x = 1; x < srcImg.cols-1; x++) {
            float dx = (-1*srcImg.data[(y-1)*width + (x-1)]) + (-2*srcImg.data[y*width+(x-1)]) + (-1*srcImg.data[(y+1)*width+(x-1)]) +
            (srcImg.data[(y-1)*width + (x+1)]) + (2*srcImg.data[y*width+(x+1)]) + (srcImg.data[(y+1)*width+(x+1)]);
            
            float dy = (srcImg.data[(y-1)*width + (x-1)]) + (2*srcImg.data[(y-1)*width+x]) + (srcImg.data[(y-1)*width+(x+1)]) +
            (-1*srcImg.data[(y+1)*width + (x-1)]) + (-2*srcImg.data[(y+1)*width+x]) + (-1*srcImg.data[(y+1)*width+(x+1)]);
            
            dstImg.at<uchar>(y,x) = sqrt( (dx*dx) + (dy*dy) ) > 255 ? 255 : sqrt( (dx*dx) + (dy*dy) );
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
    // Une los gradientes 
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dstImg );
}
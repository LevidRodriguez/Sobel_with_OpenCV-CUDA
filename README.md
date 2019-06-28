# Sobel_with_OpenCV-CUDA
Using CUDA C/C++ to achieve Sobel Filter without using the built in function in OpenCV and reduce execution time.
## *Comparacion del filtro de Sobel implementado en CPU, OpenCV y CUDA*

Para compilar:
 1. Clonar repositorio
  ``` git clone https://github.com/LevidRodriguez/Sobel_with_OpenCV-CUDA.git ```
 2. Entrar al directorio
  ``` cd Sobel_with_OpenCV-CUDA ```
 3. Configurar CUDA (Para Google Colab)
  ``` 
  pip install git+git://github.com/andreinechaev/nvcc4jupyter.git 
  load_ext nvcc_plugin  
  
  ```
 4. Compilar
  ``` make all ```
 5. Use: ``` ./sobelFilter <images/src_image> ```
 
 Los resultados son guardados en el mismo directorio como:
 
1. ``` outImgCPU.png ``` --- Resultado de aplicar el filtro de sobel en CPU
2. ``` outImgOpenCV.png ``` --- Resultado de aplicar el filtro de sobel con OpenCV
3. ```outImgGPU.png ``` --- Resultado de aplicar el filtro de sobel heterogéneo (GPU + CPU)
 
 
  

#include <iostream>
#include <stdlib.h>
#include <sys/time.h>

#include <math.h>

#include <IL/il.h>
using namespace std;

__global__ void sobel(unsigned char *data,unsigned char *out,std::size_t rows, std::size_t cols){
	auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  auto idy = blockIdx.y * blockDim.y + threadIdx.y;
	int h,v,res;
	if(idx > 0 && idx < (rows-1) && idy > 0 && idy < (cols-1)){
		for(int c = 0 ; c < 3 ; ++c) {
			// Horizontal
			  h =     data[((idx - 1) * cols + idy - 1) * 3 + c] -   data[((idx - 1) * cols + idy + 1) * 3 + c]
			  + 2 * data[( idx      * cols + idy - 1) * 3 + c] - 2 * data[( idx      * cols + idy + 1) * 3 + c]
			  +     data[((idx + 1) * cols + idy - 1) * 3 + c] -     data[((idx + 1) * cols + idy + 1) * 3 + c];

			// Vertical

			  v =     data[((idx - 1) * cols + idy - 1) * 3 + c] -   data[((idx + 1) * cols + idy - 1) * 3 + c]
			  + 2 * data[((idx - 1) * cols + idy    ) * 3 + c] - 2 * data[((idx + 1) * cols + idy    ) * 3 + c]
			  +     data[((idx - 1) * cols + idy + 1) * 3 + c] -     data[((idx + 1) * cols + idy + 1) * 3 + c];

		  	res = h*h + v*v;
			  res = res > 255*255 ? 255*255 : res;
				out[(idx * cols + idy) * 3 + c] = sqrtf(res);
		}

  }
}

int main() {
  unsigned int image;
  ilInit();
  ilGenImages(1, &image);
  ilBindImage(image);
  ilLoadImage("4v9mo.jpg");
  auto cols = ilGetInteger(IL_IMAGE_WIDTH);
  auto rows = ilGetInteger(IL_IMAGE_HEIGHT);
  auto bpp = ilGetInteger(IL_IMAGE_BYTES_PER_PIXEL);
  // Récupération des données de l'image
  unsigned char* data = ilGetData();
  auto size_img = cols * rows * bpp;
  //Traitement de l'image
  unsigned char* out = (unsigned char*)malloc(size_img);
  unsigned char* out_d;
  unsigned char* data_d;
  cudaMalloc(&out_d,size_img);
  cudaMalloc(&data_d,size_img);
  cudaMemcpy(data_d,data,size_img,cudaMemcpyHostToDevice);
  //sobel<<<grid, block>>>(data_d,out_d,rows,cols);
  struct timeval start, stop;

  dim3 block(32,32);
  dim3 grid( ( rows - 1) / block.x + 1 , ( cols - 1 ) / block.y + 1 );
  gettimeofday(&start, 0);
  sobel<<<grid, block>>>(data_d,out_d,rows,cols);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess){
		cout << "Error : " << cudaGetErrorString(err) << endl;
		exit(EXIT_FAILURE);
  }
  gettimeofday(&stop, 0);
  cout << "elapsed time: " << (((stop.tv_sec*1000000+stop.tv_usec) - (start.tv_sec*1000000+start.tv_usec))/1000) << " ms " << endl;
  cudaMemcpy(out,out_d,size_img,cudaMemcpyDeviceToHost);

  ilSetData(out);
  // Sauvegarde de l'image
  ilEnable(IL_FILE_OVERWRITE);
  ilSaveImage("out.jpg");
  ilDeleteImages(1, &image);
  free(out);
  cudaFree(out_d);
  cudaFree(data_d);
  return 0;
}

#include <iostream>
#include <stdlib.h>
#include <sys/time.h>

#include <math.h>

#include <IL/il.h>
using namespace std;

__global__ void sobel_shared(unsigned char *s,unsigned char *out, size_t width, size_t height){
	//
	auto idxg = blockIdx.x * blockDim.x + threadIdx.x;
  auto idyg = blockIdx.y * blockDim.y + threadIdx.y;

	auto idx = threadIdx.x;
	auto idy = threadIdx.y;
	auto rows = blockDim.x;
	auto cols = blockDim.y;

	extern __shared__ unsigned char data[];

	if(idxg < height && idyg < width){
		data[idx*cols +idy] = s[idxg*width+idyg];
	}
	__syncthreads();

	if(idx > 0 && idx < (rows-1) && idy > 0 && idy < (cols-1)){

		// Horizontal
	          auto h =     data[((idx - 1) * cols + idy - 1) ] -     data[((idx - 1) * cols + idy + 1)]
		  + 2 * data[( idx      * cols + idy - 1)] - 2 * data[( idx      * cols + idy + 1) ]
		  +     data[((idx + 1) * cols + idy - 1)] -     data[((idx + 1) * cols + idy + 1) ];

		// Vertical

		  auto v =     data[((idx - 1) * cols + idy - 1)  ] -     data[((idx + 1) * cols + idy - 1)]
		  + 2 * data[((idx - 1) * cols + idy    ) ] - 2 * data[((idx + 1) * cols + idy    ) ]
		  +     data[((idx - 1) * cols + idy + 1) ] -     data[((idx + 1) * cols + idy + 1) ];

		 h = h > 255 ? 255 : h;
		 v = v > 255 ? 255 : v;

		auto res = h*h + v*v;
		res = res > 255*255 ? res = 255*255 : res;
		if((idxg >1 && idxg < (height-1)) && (idyg > 1 && idyg < (width-1))){
			out[idxg * width + idyg] = sqrtf(res);
		}
       }
}

int main() {
  unsigned int image;
  ilInit();
  ilGenImages(1, &image);
  ilBindImage(image);
  ilLoadImage("in.jpg");
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
  cols = cols * bpp;
  dim3 grid( ( rows - 1) / block.x + 1 , ( cols - 1 ) / block.y + 1 );
  gettimeofday(&start, 0);
  sobel_shared<<<grid, block,1024>>>(data_d,out_d,rows,cols);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess){
		cout << "Error : " << cudaGetErrorString(err) << endl;
		exit(EXIT_FAILURE);
  }
  gettimeofday(&stop, 0);
  cout << "elapsed time: " << (((stop.tv_sec*1000000+stop.tv_usec) - (start.tv_sec*1000000+start.tv_usec))/1000) << " ms " << endl;
  cudaMemcpy(out,out_d,size_img,cudaMemcpyDeviceToHost);
  //Placement des données dans l'image
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

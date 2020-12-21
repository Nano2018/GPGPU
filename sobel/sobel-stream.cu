#include <iostream>
#include <stdlib.h>
#include <sys/time.h>

#include <math.h>

#include <IL/il.h>
using namespace std;

__global__ void sobel(unsigned char *data,unsigned char *out,size_t rows,size_t cols){
	auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  auto idy = blockIdx.y * blockDim.y + threadIdx.y;
	int h,v,res;
	if( idx > 0 && idx < rows-1 && idy > 0 && idy < cols-1){
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
			out[(idx * cols + idy) * 3 + c] =sqrtf(res);
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
  auto size_img = cols * rows * bpp;
  // Récupération des données de l'image
  unsigned char* data = ilGetData();
  //forcer les données a etre dans un espace contigu en mémoire.
  cudaHostRegister( data, size_img, cudaHostRegisterDefault);
  // Traitement de l'image
  unsigned char* out = (unsigned char*)malloc(size_img);
  unsigned char* out_d;
  unsigned char* data_d;

  cudaError_t err = cudaMalloc( &out_d, size_img );
  if( err != cudaSuccess ) {
  	cerr << "Error: " << cudaGetErrorString(err) << endl;
		exit(EXIT_FAILURE);
  }
  err = cudaMalloc(&data_d,size_img+(2*cols*bpp));
	if( err != cudaSuccess ) {
  	cerr << "Error: " << cudaGetErrorString(err) << endl;
		exit(EXIT_FAILURE);
  }
  //utilisation des streams:
	cudaEvent_t start, stop;
  cudaEventCreate( &start );
  cudaEventCreate( &stop );
  cudaEventRecord( start );
	cudaStream_t streams[2];
  for( std::size_t i = 0 ; i < 2 ; ++i ){
    cudaStreamCreate(&streams[ i ] );
  }

 /*****************************************/
 	auto size = bpp*cols;
	auto offset = 0;
  for( size_t i = 0 ; i < 2 ; ++i ){
		if(i == 1){
			offset = bpp*cols;
		}
    cudaMemcpyAsync(data_d + i*size_img/2+offset , data + i*size_img/2-offset , size_img/2+size, cudaMemcpyHostToDevice, streams[i] );
  	/*cudaDeviceSynchronize();
    err = cudaGetLastError();
		if(err != cudaSuccess) {
      cerr << "Error: " << cudaGetErrorString(err) << endl;
      exit(EXIT_FAILURE);
    }*/
   }

 /****************************************/
  //lancement du kenenel avec les streams.
  dim3 t( 32, 32 );
  //int rows_bis[2]; rows_bis[0] = rows  int cols, rows, bpp;_bis[1] = rows/2;
  dim3 b( ( rows - 1) / (t.x) + 1 , ( cols - 1 ) / (t.y) + 1 );
	for(size_t i=0; i<2; i++){
     sobel<<< b, t, 0,streams[i] >>>( data_d + i*size_img/2, out_d + i*size_img/2, rows/2+1, cols );
     /*cudaDeviceSynchronize();
     err = cudaGetLastError();
     if(err != cudaSuccess){
        std::cout << "Error kernel : " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
     }*/
  }

	size = 0;
  for( size_t i = 0 ; i < 2 ; ++i ){
  	if(i == 1 && size_img%2 != 0){
  		size = 1;
  	}
    cudaMemcpyAsync(out + i*size_img/2, out_d + i*size_img/2 ,size_img/2+size, cudaMemcpyDeviceToHost, streams[i] );
		/*cudaDeviceSynchronize();
  	err = cudaGetLastError();
		if( err != cudaSuccess ) {
		  	cerr << "Error cudaMemcpyAsyncDeviceToHost: " << cudaGetErrorString(err)<< endl;
				exit(EXIT_FAILURE);
		}*/
   }
	cudaEventRecord( stop );
  cudaEventSynchronize( stop );
  float duration = 0.0f;
  cudaEventElapsedTime( &duration, start, stop );
 	cout << "time: " << duration << "ms\n";
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

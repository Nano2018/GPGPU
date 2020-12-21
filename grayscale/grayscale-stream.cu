#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
using namespace std;
__global__ void grayscale(unsigned char *rgb, unsigned char *out, std::size_t rows, std::size_t cols){
	auto idx = blockIdx.x * blockDim.x + threadIdx.x;
	auto idy = blockIdx.y * blockDim.y + threadIdx.y;
	if(idx < rows && idy < cols){
		out[idx * cols + idy] = (
			 307 * rgb[ 3 * ( idx * cols + idy ) ]
		       + 604 * rgb[ 3 * ( idx * cols + idy ) + 1 ]
		       + 113 * rgb[  3 * ( idx * cols + idy ) + 2 ]
		       ) / 1024;
	}
}

int main()
{
  cv::Mat m_in = cv::imread("4v9mo.jpg", cv::IMREAD_UNCHANGED );
  auto rgb = m_in.data;
  auto rows = m_in.rows;
  auto cols = m_in.cols;
  auto bpp = 3;
  auto size_img = cols*rows*bpp;
  auto size_matrix = cols*rows;
  cudaHostRegister( rgb,size_img, cudaHostRegisterDefault);
  std::vector< unsigned char > out( size_matrix );
  cv::Mat m_out(rows,cols, CV_8UC1, out.data());
  unsigned char* out_d;
  unsigned char* data_d;
  cudaError_t err = cudaMalloc( &out_d, size_matrix );
  if( err != cudaSuccess ) {
    cerr << "Error cudaMalloc: " << cudaGetErrorString(err) << endl;
    exit(EXIT_FAILURE);
  }
  err = cudaMalloc(&data_d,size_img+(2*size_matrix));
  if( err != cudaSuccess ) {
    cerr << "Error cudaMalloc: " << cudaGetErrorString(err) << endl;
    exit(EXIT_FAILURE);
  }
  cudaEvent_t start, stop;
  cudaEventCreate( &start );
  cudaEventCreate( &stop );
  cudaEventRecord( start );
  //streams:
  cudaStream_t streams[2];
  for( std::size_t i = 0 ; i < 2 ; ++i ){
    cudaStreamCreate(&streams[ i ] );
  }

  auto size = bpp*cols;
  auto offset = 0;
  for( size_t i = 0 ; i < 2 ; ++i ){
    if(i == 1){
      offset = bpp*cols;
    }
    cudaMemcpyAsync(data_d + i*size_img/2+offset , rgb + i*size_img/2-offset , size_img/2+size, cudaMemcpyHostToDevice, streams[i] );
    /*cudaDeviceSynchronize();
    err = cudaGetLastError();
    if( err != cudaSuccess ) {
      cerr << "Error cudaMemcpyAsync: " << cudaGetErrorString(err) << endl;
      exit(EXIT_FAILURE);
    }*/
   }

  //lancement du kernel avec les streams.
  dim3 t( 32, 32 );
  dim3 b( ( rows - 1) / (t.x) + 1 , ( cols - 1 ) / (t.y) + 1 );
  for(size_t i=0; i<2; i++){
     grayscale<<< b, t, 0,streams[i] >>>( data_d + i*size_img/2, out_d + i*size_matrix/2, rows/2+1, cols );
     /*cudaDeviceSynchronize();
     err = cudaGetLastError();
     if(err != cudaSuccess){
        std::cout << "Error kernel : " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
     }*/
  }

  for( size_t i = 0 ; i < 2 ; ++i ){
    cudaMemcpyAsync(out.data() + i*size_matrix/2, out_d + i*size_matrix/2 ,size_matrix/2, cudaMemcpyDeviceToHost, streams[i] );
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
  cv::imwrite( "out.jpg", m_out );
  cudaFree(data_d);
  cudaFree(out_d);
  return 0;
}

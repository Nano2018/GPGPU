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
  std::vector< unsigned char > g( rows * cols );
  cv::Mat m_out(rows,cols, CV_8UC1, g.data());
  unsigned char *rgb_d = nullptr;
  unsigned char *m_d = nullptr;
  cudaMalloc(&m_d, rows * cols);
  cudaMalloc(&rgb_d, 3 * rows * cols );
  cudaMemcpy(rgb_d,rgb,rows * cols * 3,cudaMemcpyHostToDevice);

  cudaEvent_t start, stop;
  cudaEventCreate( &start );
  cudaEventCreate( &stop );
  cudaEventRecord( start );
  dim3 block(32,32);
  dim3 grid( ( rows - 1) / block.x + 1 , ( cols - 1 ) /block.y + 1 );
  grayscale<<<grid,block>>>(rgb_d,m_d,rows,cols);
  //gestion des erreurs.
  /*cudaDeviceSynchronize();
  cudaError err = cudaGetLastError();     
  if(err != cudaSuccess){
	cerr << cudaGetErrorString(err) << endl;
	exit(EXIT_FAILURE);
  }*/
	cudaMemcpy(g.data(),m_d,rows * cols,cudaMemcpyDeviceToHost);
  cudaEventRecord( stop );
  cudaEventSynchronize( stop );
  float duration = 0.0f;
  cudaEventElapsedTime( &duration, start, stop );
  cout << "time: " << duration << "ms\n";
	//cudaMemcpy(g.data(),m_d,rows * cols,cudaMemcpyDeviceToHost);
  cv::imwrite( "out.jpg", m_out );
  cudaFree(rgb_d);
  cudaFree(m_d);
  return 0;
}

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
using namespace std;
__global__ void grayscale_sobel( unsigned char * in, unsigned char * out, std::size_t w, std::size_t h ) {
  auto i = blockIdx.x * (blockDim.x-2) + threadIdx.x;
  auto j = blockIdx.y * (blockDim.y-2) + threadIdx.y;

  auto li = threadIdx.x;
  auto lj = threadIdx.y;

  extern __shared__ unsigned char sh[];

  if( i < w && j < h ) {
    sh[ lj * blockDim.x + li ] = (
			 307 * in[ 3 * ( j * w + i ) ]
			 + 604 * in[ 3 * ( j * w + i ) + 1 ]
			 + 113 * in[  3 * ( j * w + i ) + 2 ]
			 ) / 1024;
  }

  __syncthreads();

  auto ww = blockDim.x;

  if( li > 0 && li < (blockDim.x - 1) && lj > 0 && lj < (blockDim.y - 1) )
  {
    auto hh = sh[ (lj-1)*ww + li - 1 ] - sh[ (lj-1)*ww + li + 1 ]
           + 2 * sh[ lj*ww + li - 1 ] - 2* sh[ lj*ww+li+1 ]
           + sh[ (lj+1)*ww + li -1] - sh[ (lj+1)*ww +li + 1 ];
    auto vv = sh[ (lj-1)*ww + li - 1 ] - sh[ (lj+1)*ww + li - 1 ]
           + 2 * sh[ (lj-1)*ww + li  ] - 2* sh[ (lj+1)*ww+li ]
           + sh[ (lj-1)*ww + li +1] - sh[ (lj+1)*ww +li + 1 ];

    auto res = hh * hh + vv * vv;
    res = res > 255*255 ? res = 255*255 : res;
    out[ j * w + i ] = sqrt( (float)res );

  }
}

int main()
{
  cv::Mat m_in = cv::imread("4v9mo.jpg", cv::IMREAD_UNCHANGED );
  auto rgb = m_in.data;
  auto rows = m_in.rows;
  auto cols = m_in.cols;
  std::vector< unsigned char > g( rows * cols );
  cv::Mat m_out( rows, cols, CV_8UC1, g.data() );
  unsigned char * rgb_d;
  unsigned char * g_d;
  unsigned char * out_d;
  cudaMalloc( &rgb_d, 3 * rows * cols );
  cudaMalloc( &g_d, rows * cols );
  cudaMalloc( &out_d, rows * cols );
  cudaMemcpy( rgb_d, rgb, 3 * rows * cols, cudaMemcpyHostToDevice );
  cudaEvent_t start, stop;
  cudaEventCreate( &start );
  cudaEventCreate( &stop );
  cudaEventRecord( start );
  dim3 t( 32, 32 );
  dim3 b( ( cols - 1) / (t.x-2) + 1 , ( rows - 1 ) / (t.y-2) + 1 );
  grayscale_sobel<<< b, t, t.x*t.y >>>( rgb_d, g_d, cols, rows );
  cudaEventRecord( stop );
  cudaEventSynchronize( stop );
  float duration = 0.0f;
  cudaEventElapsedTime( &duration, start, stop );
 	cout << "time: " << duration << "ms\n";
  cudaDeviceSynchronize();
  /*auto err = cudaGetLastError();
  if( err != cudaSuccess )
  {
    std::cout << cudaGetErrorString( err );
  }*/

  cudaMemcpy( g.data(), g_d, rows * cols, cudaMemcpyDeviceToHost );
  cv::imwrite( "out.jpg", m_out );
  cudaFree( rgb_d);
  cudaFree( g_d);
  return 0;
}

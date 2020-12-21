#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
using namespace std;

/**
//version pour l'image grise.
filtres:
// {0,0,0,0,12,0,0,0,0}; filtre lighten.
// {0,0,0,0,6,0,0,0,0}; filtre darken.
// {0,0,0,0,9,0,0,0,0}; filtre identity.

*/
__global__ void filter(unsigned char *datas, unsigned char *out, size_t height, size_t width) {
  auto idxg = blockIdx.x * blockDim.x + threadIdx.x;
  auto idyg = blockIdx.y * blockDim.y + threadIdx.y;
	auto idx = threadIdx.x;
	auto idy = threadIdx.y;
	auto rows = blockDim.x;
	auto cols = blockDim.y;
  extern __shared__ unsigned char sh[];

	if(idxg < height && idyg < width){
		sh[idx*cols +idy] = datas[3*(idxg*width+idyg)];
	}
  __syncthreads();
  
  unsigned char sharp[9] ={1,1,1,1,1,1,1,1,1};
  if(idxg > 0 && idyg > 0 && idxg<(height-1) && idyg<(width-1)){
      auto sum = 0;
      auto cpt = 0;
      for (auto i=-1; i<2; i++){
  	    for(auto j=-1; j<2; j++){
          //si le pixel n'est pas accessible depuis la mémoire partagée
          if((idx+i)>(rows-1) || (idy+j)>(cols-1) || (idx+i) < 0 || (idy+j) < 0){
            sum += datas[3*((idxg+i)*width+idyg+j)] * sharp[cpt];
          }else{
            sum += sh[(idx+i)*cols+(idy+j)] * sharp[cpt];
          }
          cpt++;
  	    }
      }
      out[idxg * width + idyg] = sum/9;
    }
}

int main(int argc, char **argv)
{
  cv::Mat m_in = cv::imread("maison.jpg", cv::IMREAD_UNCHANGED );
  auto datas = m_in.data;
  auto rows = m_in.rows;
  auto cols = m_in.cols;
  auto size_img = 3*cols*rows;
  auto size_matrix = cols*rows;

  vector<unsigned char> out(size_matrix);
  cv::Mat m_out(rows,cols, CV_8UC1, out.data());
  unsigned char * rgb_d;
  unsigned char * out_d;
  cudaError_t err;
  err = cudaMalloc( &rgb_d, size_img );
  if(err != cudaSuccess){
    cout << "ERROR cudaMalloc: " << cudaGetErrorString(err) << endl;
    exit(EXIT_FAILURE);
  }
  err = cudaMalloc( &out_d, size_matrix );
  if(err != cudaSuccess){
    cout << "ERROR cudaMalloc: " << cudaGetErrorString(err) << endl;
    exit(EXIT_FAILURE);
  }
  cudaMemcpy( rgb_d, datas, size_img, cudaMemcpyHostToDevice );
  //evènement pour calculer le temps mis par le gpu.
  cudaEvent_t start, stop;
  cudaEventCreate( &start );
  cudaEventCreate( &stop );
  cudaEventRecord( start );
  dim3 t( 32, 32 );
  dim3 b( ( rows - 1) /t.x + 1 , ( cols - 1 ) / t.y + 1 );

  filter<<< b, t,1024 >>>( rgb_d, out_d, rows, cols );
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if( err != cudaSuccess )
  {
    cout << "Error kernel: " << cudaGetErrorString( err ) << endl;
    exit(EXIT_FAILURE);
  }
  cudaMemcpy( out.data(), out_d, size_matrix, cudaMemcpyDeviceToHost );
  cudaEventRecord( stop );
  cudaEventSynchronize( stop );
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop );
  cout << " elapsed time : " << elapsedTime << " ms" << endl;
  cudaEventDestroy( start );
  cudaEventDestroy( stop );
  cv::imwrite( "out.jpg",m_out);
  cudaFree(rgb_d);
  cudaFree(out_d);
  return 0;
}

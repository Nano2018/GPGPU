#include <opencv2/opencv.hpp>
#include <vector>
using namespace std;

__global__ void gray_sobel(unsigned char *datas,unsigned char *out, size_t height, size_t width){
  auto idxg = blockIdx.x * blockDim.x + threadIdx.x;
  auto idyg = blockIdx.y * blockDim.y + threadIdx.y;
  auto idx = threadIdx.x;
  auto idy = threadIdx.y;
  auto rows = blockDim.x;
  auto cols = blockDim.y;
  extern __shared__ unsigned char sh[];
  if(idxg < height && idyg < width){
    sh[idx*cols +idy] = (
       307 * datas[ 3 * ( idxg*width+idyg ) ]
       + 604 * datas[ 3 * ( idxg*width+idyg ) + 1 ]
       + 113 * datas[  3 * ( idxg*width+idyg ) + 2 ]
       ) / 1024;
  }
  //synchronisation des threads.
  __syncthreads();
  unsigned char h,v,res;
  if(idx > 0 && idx < (rows-1) && idy > 0 && idy < (cols-1)){
     // Horizontal
        h =     sh[((idx - 1) * cols + idy - 1)] -   sh[((idx - 1) * cols + idy + 1)]
        + 2 * sh[( idx      * cols + idy - 1)] - 2 * sh[( idx      * cols + idy + 1)]
        +     sh[((idx + 1) * cols + idy - 1)] -     sh[((idx + 1) * cols + idy + 1)];
      // Vertical
        v =     sh[((idx - 1) * cols + idy - 1)] -   sh[((idx + 1) * cols + idy - 1)]
        + 2 * sh[((idx - 1) * cols + idy    )] - 2 * sh[((idx + 1) * cols + idy    )]
        +     sh[((idx - 1) * cols + idy + 1)] -     sh[((idx + 1) * cols + idy + 1)];
       res = h*h + v*v;
       res = res > 255*255 ? 255*255 : res;
      if(idxg >0 && idxg < (height-1) && idyg > 0 && idyg < (width-1)){
           out[(idxg * width + idyg)] = sqrtf(res);
      }
  }
}

int main()
{
  cv::Mat m_in = cv::imread("4v9mo.jpg", cv::IMREAD_UNCHANGED );
  auto datas = m_in.data;
  auto rows = m_in.rows;
  auto cols = m_in.cols;
  auto bpp = 3;
  auto size_img = rows * cols * bpp;
  auto size_matrix = cols * rows;
  cudaHostRegister(datas, size_img, cudaHostRegisterDefault);
  vector< unsigned char > out( size_matrix );
  cv::Mat m_out( rows, cols, CV_8UC1, out.data() );
  unsigned char *rgb_d;
  unsigned char *out_d;
  //cudaError_t err;
  cudaMalloc( &rgb_d,size_img+(2*bpp*cols));
  cudaMalloc( &out_d, size_matrix);
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
    cudaMemcpyAsync(rgb_d + i*size_img/2+offset , datas + i*size_img/2-offset , size_img/2+size, cudaMemcpyHostToDevice, streams[i] );
    /*cudaDeviceSynchronize();
    err = cudaGetLastError();
    if( err != cudaSuccess ) {
      cerr << "Error cudaMemcpyAsync: " << cudaGetErrorString(err) << endl;
      exit(EXIT_FAILURE);
    }*/
   }

 /****************************************/
  cudaEvent_t start, stop;
  cudaEventCreate( &start );
  cudaEventCreate( &stop );
  cudaEventRecord( start );
  //lancement du kenenel avec les streams.
  dim3 t( 32, 32 );
  dim3 b( ( rows/2 - 1) / (t.x) + 1 , ( cols - 1 ) / (t.y) + 1 );
  for(size_t i=0; i<2; i++){
     gray_sobel<<< b, t,1024,streams[i] >>>( rgb_d + i*(size_img/2 + (bpp*cols)), out_d + i*size_matrix/2, rows/2, cols );
     /*cudaDeviceSynchronize();
     err = cudaGetLastError();
     if(err != cudaSuccess){
        std::cout << "Error kernel : " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
     }*/
  }
  cudaEventRecord( stop );
  cudaEventSynchronize( stop );
  float duration = 0.0f;
  cudaEventElapsedTime( &duration, start, stop );
 	cout << "time: " << duration << "ms\n";
  /***************************************/
  for( size_t i = 0 ; i < 2 ; ++i ){
    cudaMemcpyAsync(out.data()+i*size_matrix/2, out_d + i*size_matrix/2 ,size_matrix/2, cudaMemcpyDeviceToHost, streams[i] );
    /*cudaDeviceSynchronize();
    err = cudaGetLastError();
    if( err != cudaSuccess ) {
        cerr << "Error cudaMemcpyAsyncDeviceToHost: " << cudaGetErrorString(err)<< endl;
        exit(EXIT_FAILURE);
    }*/
   }
   for( std::size_t i = 0 ; i < 2 ; ++i )
  {
    cudaStreamDestroy( streams[ i ] );
  }
  /**************************************/
  cv::imwrite( "out.jpg", m_out );

  cudaFree(rgb_d);
  cudaFree(out_d);
  return 0;
}

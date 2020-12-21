#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
using namespace std;

__global__ void filter(unsigned char *datas, unsigned char *out, size_t rows, size_t cols) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  auto idy = blockIdx.y * blockDim.y + threadIdx.y;
  if(idx < rows && idy < cols){
      out[idx*cols+idy] = datas[3*(idx*cols+idy)];
  }
}

__global__ void rotation(unsigned char *datas, unsigned char *out, size_t rows, size_t cols,size_t choice) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  auto idy = blockIdx.y * blockDim.y + threadIdx.y;
  if(idx < rows && idy < cols){
      if(choice==1){
          out[idy*rows+idx] = datas[idx*cols+cols-1-idy];
      }else{
          out[idy*rows+idx] = datas[(rows-1-idx)*cols+idy];
      }
  }
}
int main(int argc, char **argv){
  cout << "ce programme permet de faire pivoter à gauche ou à droite une image !!" << endl;
  cout << "pour une rotation à gauche tapez 1" << endl;
  cout << "pour une rotation à droite tapez 2" << endl;
  int choice = 0;
  char ch[1];
  while(choice < 1 || choice > 2){
    cout << "tapez votre choix : ";
    cin >> ch;
    choice = atoi(ch);
  }
  cv::Mat m_in = cv::imread("maison.jpg", cv::IMREAD_UNCHANGED );
  auto datas = m_in.data;
  auto rows = m_in.rows;
  auto cols = m_in.cols;
  auto size_img = 3*cols*rows;
  auto size_matrix = cols*rows;
  // Récupération des données de l'image
  vector<unsigned char> out(size_matrix);
  cv::Mat m_out(rows,cols, CV_8UC1, out.data());
  unsigned char * rgb_d;
  unsigned char * out_d;
  unsigned char * outf_d;
  // deux tableaux out_d et outd_f
  // le premier pour récupérer les données calculées sur le premier kernel, le deuxième sur le second kernel
  cudaMalloc( &rgb_d, size_img );
  cudaMalloc( &out_d, size_matrix );
  cudaMalloc( &outf_d, size_matrix );
  cudaMemcpy( rgb_d, datas, size_img, cudaMemcpyHostToDevice );
  dim3 t( 32, 32 );
  dim3 b( ( rows - 1) /32 + 1 , ( cols - 1 ) / 32 + 1 );
  //evènement pour calculer le temps mis par le gpu.
  cudaEvent_t start, stop;
  cudaEventCreate( &start );
  cudaEventCreate( &stop );
  cudaEventRecord( start );
  filter<<< b, t >>>( rgb_d, outf_d, rows, cols );
  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  if( err != cudaSuccess )
  {
    std::cout <<"error in kernel filter : " << cudaGetErrorString( err ) << endl;
    exit(EXIT_FAILURE);
  }
  rotation<<<b,t>>>(outf_d,out_d,rows,cols,choice);
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if( err != cudaSuccess )
  {
    std::cout << "error in rotation kernel: " << cudaGetErrorString( err ) << endl;
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
  cudaFree( rgb_d);
  cudaFree(out_d);
  cudaFree(outf_d);
  return 0;
}


#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <complex.h>
#include <cuda_runtime.h>
#include <utility>
#include <sys/time.h>
#define K 3
#define BLCH 8
#define BLCW 32

__constant__ float filter[K*K];

__global__ void compute_gpu(float *img, float *out, int bh, int bw, int imgH, int imgW, int imgN, int nF, int convH, int convW){
    
    int idX = blockDim.x * blockIdx.x + threadIdx.x;
    int idY = blockDim.y * blockIdx.y + threadIdx.y;
    
    for (int i = 0; i < imgN; i++){
        int con = i * convW * convH;
        int imgg = i * imgW * imgH;
        if (idX < convH && idY < convW){
            int ind = idY * convW + idX + con;
            int inm = idY * imgW + idX + imgg;
            for (int fi = 0; fi < nF; fi++){
                inm += fi * imgW;
                int inf = fi*nF;
                for (int fj = 0; fj < nF; fj++){
                    inf += fj;
                    inm += fj;
                    float value = img[inm];
                    if (value != 0){
                        out[ind] += value * filter[inf];
                    }
                }
            }
        }
    }
}


int main(int argc, char **argv){

    //create parameters
    int imgH = 2048;
    int imgW = 2048;
    int imgN = 10;
    int blcH = BLCH;
    int blcW = BLCW;
    int k    = K;
    int s    = 1;
    int nB   = (imgH * imgW) / (blcH * blcW);
    //int nT   = (blcW+k) * (blcH+k);
    int nT   = blcW * blcH;
    int imgDims = imgH * imgW * imgN;
    int imgSize = imgDims * sizeof(float);
    
    int num=0;
    srand (time(NULL));
    // create host array that can hold pixel intensity values
    float *h_img = new float[imgDims];
    float *h_imgcsr = new float[imgDims];
    
    for(int i=0; i<imgH; i++){
        for (int j=0; j<imgW; j++){
            if (rand() % 10 == 0){
                num++;
                h_img[i*imgW+j] = (float)(rand()%10485)/10485;
            }
            else{
                h_img[i*imgW+j] = 0.0;
            }
        }
    }
    
    int *pos = new int[imgH];
    int *coor = new int[num];
    int index_p=0;
    int index_c=0;
    for(int i=0; i<imgH; i++){
        for (int j=0; j<imgW; j++){
            if (h_img[i*imgW+j] != 0){
                coor[index_p] = j;
                h_imgcsr[index_p] = h_img[i*imgW+j];
                index_p++;
                
            }
        }
        pos[i] = index - 1;
    }
    
    
    // create filter and copy to constant memory
    int filterDims = k * k;
    int filterSize = filterDims * sizeof(float);
    float *filter = new float[filterDims];
    for(int i=0; i<filterDims; i++){
        filter[i] = (float)(rand()%10485)/10485;
    }
    
    // create host and device array that holds the convoluted matrix
    int convH = ( (imgH - k) / s ) + 1;
    int convW = ( (imgW - k) / s ) + 1;
    int convDims = convH * convW;
    int convSize = convDims * sizeof(float);
    float *h_convolved = new float[convDims];
    for(int i=0; i<convDims; i++){
        h_convolved[i] = 0.0;
    }
    
    // create device array that can hold pixel intensity values in GPU GM
    float *d_img;
    float *d_convolved;
    
    struct timeval starttime, endtime;
    double elapsed = 0.0;
    for (int i = 0; i<10000; i++){
        gettimeofday(&starttime,NULL);
        // call the kernel
        compute_gpu_tiled<<<nB, nT>>>(d_img, d_convolved, blcH, blcW, imgH, imgW, imgN, k, convH, convW);
        gettimeofday(&endtime,NULL);
        elapsed += ((endtime.tv_sec-starttime.tv_sec)*1000000 + endtime.tv_usec-starttime.tv_usec)/1000000.0;
        cudaDeviceReset();
    }
    printf("Input imgH: %d imgW: %d imgN: %d\n", imgH, imgW, imgN);
    printf("Tile width: %d height: %d\n", blcW, blcH);
    printf("Block number: %d, block size: %d \n", nB, nT);
    printf("time: %f \n", &elapsed);
    delete h_img;
    delete h_convolved;
    delete pos;
    delete h_imgcsr;
    delete coor;
    return 0;
}


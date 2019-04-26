#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <complex.h>
#include <cuda_runtime.h>
#include <utility>
#include <sys/time.h>
#define K 3

__constant__ float filter[K*K];

int compute_naive(float *img, float *f, float * out, int bh, int bw, int imgH, int imgW, int imgN, int nF, int convH, int convW) {
    
    for (int i=0; i<imgN; i++){
        int con = i * convH * convW;
        int imgg = i * imgW * imgH;
        for (int j=0; j<convH; j++){
            int ind = j * convW;
            int inm = j * imgW;
            for (int k=0; k<convW; k++){
                ind += k;
                inm += k;
                for (int fi=0; fi<imgN; fi++){
                    int inf = fi * nF;
                    inm += fi * imgW;
                    for (int fj=0; fj<imgN; fj++){
                        inf += fj;
                        inm += fj;
                        out[ind] += img[inm+imgg]*f[inf];
                    }
                }
            }
        }
    }
         
    return 0;
}

int compute_naive_tiled(float *img, float *f, float * out, int bh, int bw, int imgH, int imgW, int imgN, int nF, int convH, int convW){
    int n = (int)sqrt((double)imgN);
    for (int ni=0; ni<n: ni++){
        for (int nj=0; nj<n: nj++){
            int i = ni * n + nj;
            int con = i * convH * convW;
            int imgg = i * imgW * imgH;
            for (int j=0; j<convH; j++){
                int ind = j * convW;
                int inm = j * imgW;
                for (int k=0; k<convW; k++){
                    ind += k;
                    inm += k;
                    for (int fi=0; fi<imgN; fi++){
                        int inf = fi * nF;
                        inm += fi * imgW;
                        for (int fj=0; fj<imgN; fj++){
                            inf += fj;
                            inm += fj;
                            out[ind] += img[inm+imgg]*f[inf];
                        }
                    }
                }
            }
        }
    }
    
    
}

__global__ void compute_gpu(float *img, float *out, int bh, int bw, int imgH, int imgW, int imgN, int nF, int convH, int convW){
    
    int idX = blockDim.x * blockIdx.x + threadIdx.x;
    int idY = blockDim.y * blockIdx.y + threadIdx.y;
    
    int n = (int)sqrt((double)imgN);
    if (idX < n && idY < n){
        int i = idX * bw + idY;
        int con = i * convH * convW;
        int imgg = i * imgW * imgH;
        for (int j=0; j<convH; j++){
            int ind = j * convW;
            int inm = j * imgW;
            for (int k=0; k<convW; k++){
                ind += k;
                inm += k;
                for (int fi=0; fi<imgN; fi++){
                    int inf = fi * nF;
                    inm += fi * imgW;
                    for (int fj=0; fj<imgN; fj++){
                        inf += fj;
                        inm += fj;
                        out[ind] += img[inm+imgg]*f[inf];
                    }
                }
            }
        }
    }
}


int main(int argc, char **argv){

    //create parameters
    int imgH = 10;
    int imgW = 10;
    int imgN = 4096;
    int blcW = 16;
    int blcH = 16;
    int k    = K;
    int s    = 1;
    int nB   = (imgH * imgW) / (blcH * blcW);
    int nT   = blcW * blcH;
    int imgDims = imgH * imgW * imgN;
    int imgSize = imgDims * sizeof(float);
    
    
    srand (time(NULL));
    // create host array that can hold pixel intensity values
    float *h_img = new float[imgDims];
    for(int i=0; i<imgDims; i++){
        h_img[i] = (float)(rand()%10485)/10485;
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
    for (int i = 0; i<100; i++){
        gettimeofday(&starttime,NULL);
        // call the kernel
        compute_gpu<<<nB, nT>>>(d_img, d_convolved, blcH, blcW, imgH, imgW, imgN, k, convH, convW)
        gettimeofday(&endtime,NULL);
        elapsed += ((endtime.tv_sec-starttime.tv_sec)*1000000 + endtime.tv_usec-starttime.tv_usec)/1000000.0;
        cudaDeviceReset();
    }
    printf("Input imgH: %d imgW: %d imgN: %d\n", &imgH, &imgW, &imgN);
    printf("Tile width: %d height: %d\n", &blcW, &blcH);
    printf("Block number: %d, block size: %d \n", &nB, &nT);
    printf("time: %f \n", &elapsed);
    delete h_img;
    delete h_convolved;
    return 0;
}


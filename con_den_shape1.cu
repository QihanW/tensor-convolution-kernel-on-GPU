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

int compute_tiled_naive(float *img, float *f, float * out, int bh, int bw, int imgH, int imgW, int imgN, int nF, int convH, int convW) {
    //compute tile num
    int bbw = imgW / bw;
    int bbh = imgH / bh;
    for (int i = 0; i < imgN; i++){
        int con = i * convW * convH;
        int imgg = i * imgW * imgH;
        //compute center tiles
        for (int j = 0; j < bbh-1; j++){
            for (int k = 0; k < bbw-1; k++){
                for (int mi = 0; mi < bh; mi++){
                    int inm = imgW * (j*bh + mi) + k*bw;
                    int ind = convW * (j*bh + mi) + k*bw;
                    for (int mj = 0; mj < bw; mj++){
                        inm += mj;
                        ind += mj;
                        for (int fi = 0; fi < nF; fi++){
                            int inf = fi * convW;
                            inm += fi * imgW;
                            for (int fj = 0; fj < nF; fj++){
                                inf += fj;
                                inm += fj;
                                out[ind + con] += img[inm + imgg] * f[inf];
                            }
                        }
                    }
                }
            }
        }
        //compute right most tiles
        for (int j = 0; j < bbh-1; j++){
            for (int mi = 0; mi < bh; mi++){
                int inm = imgW * (j*bh + mi) + (bbw-1)*bw;
                int ind = convW * (j*bh + mi) + (bbw-1)*bw;
                for (int mj = 0; mj < (bw - nF + 1); mj++){
                    inm += mj;
                    ind += mj;
                    for (int fi = 0; fi < nF; fi++){
                        int inf = fi * nF;
                        inm += fi * imgW;
                        for (int fj = 0; fj < nF; fj++){
                            inf += fj;
                            inm += fj;
                            out[ind + con] += img[inm + imgg] * f[inf];
                        }
                    }
                }
            }
        }
        //compute bottom tiles
        for (int j = 0; j < bbw-1; j++){
            for (int mi = 0; mi < (bh -nF + 1); mi++){
                int inm = imgW * ((bbh-1)*bh + mi) + (j-1)*bw;
                int ind = convW * ((bbh-1)*bh + mi) + (j-1)*bw;
                for (int mj = 0; mj < bw; mj++){
                    inm += mj;
                    ind += mj;
                    for (int fi = 0; fi < nF; fi++){
                        int inf = fi * nF;
                        inm += fi * imgW;
                        for (int fj = 0; fj < nF; fj++){
                            inf += fj;
                            inm += fj;
                            out[ind + con] += img[inm + imgg] * f[inf];
                        }
                    }
                }
            }
        }
        //compute the final tile
        for (int mi = 0; mi < (bh -nF + 1); mi++){
            int inm = imgW * ((bbh-1)*bh + mi) + (bbw-1)*bw;
            int ind = convW * ((bbh-1)*bh + mi) + (bbw-1)*bw;
            for (int mj = 0; mj < (bw - nF + 1); mj++){
                inm += mj;
                ind += mj;
                for (int fi = 0; fi < nF; fi++){
                    int inf = fi * nF;
                    inm += fi * imgW;
                    for (int fj = 0; fj < nF; fj++){
                        inf += fj;
                        inm += fj;
                        out[ind + con] += img[inm + imgg] * f[inf];
                    }
                }
            }
        }
        
    }
    
    return 0;
}

__global__ void compute_gpu_naive(float *img, float *out, int bh, int bw, int imgH, int imgW, int imgN, int nF, int convH, int convW){
    
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
                    out[ind] += img[fj + inm] * filter[inf + fj];
                }
            }
        }
    }
}

__global__ void compute_gpu_tiled(float *img, float *out, int bh, int bw, int imgH, int imgW, int imgN, int nF, int convH, int convW){
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int nT = blockDim.x;
    int nB = gridDim.x;
    
    int bbw = imgW / bw;
    int bbh = imgH / bh;
    
    
    for (int i = 0; i < imgN; i++){
        int con = i * convW * convH;
        int imgg = i * imgW * imgH;
        //compute center tiles
        if(bx < (bbh-1) && by < (bbw-1)){
            if(tx < bh && ty < bw){
                int inm = imgW * (bx*bh + tx) + by*bw + ty;
                int ind = convW * (bx*bh + ty) + by*bw + ty;
                for (int fi = 0; fi < nF; fi++){
                    int inf = fi * convW;
                    inm += fi * imgW;
                    for (int fj = 0; fj < nF; fj++){
                        inf += fj;
                        inm += fj;
                        out[ind + con] += img[inm + imgg] * filter[inf];
                    }
                }
            }
            __syncthreads();
        }
        //compute right most tiles
        if(bx < (bbh-1) && by == (bbw-1)){
            if(tx < bh && ty < (bw - nF + 1)){
                int inm = imgW * (bx*bh + tx) + by*bw + ty;
                int ind = convW * (bx*bh + ty) + by*bw + ty;
                for (int fi = 0; fi < nF; fi++){
                    int inf = fi * convW;
                    inm += fi * imgW;
                    for (int fj = 0; fj < nF; fj++){
                        inf += fj;
                        inm += fj;
                        out[ind + con] += img[inm + imgg] * filter[inf];
                    }
                }
            }
            __syncthreads();
        }
        //compute the bottom tiles
        if(bx == (bbh-1) && by < (bbw-1)){
            if(tx < (bh -nF + 1) && ty < bw){
                int inm = imgW * (bx*bh + tx) + by*bw + ty;
                int ind = convW * (bx*bh + ty) + by*bw + ty;
                for (int fi = 0; fi < nF; fi++){
                    int inf = fi * convW;
                    inm += fi * imgW;
                    for (int fj = 0; fj < nF; fj++){
                        inf += fj;
                        inm += fj;
                        out[ind + con] += img[inm + imgg] * filter[inf];
                    }
                }
            }
            __syncthreads();
        }
        //compute the final tile
        if(bx == (bbh-1) && by == (bbw-1)){
            if(tx < (bh - nF + 1) && ty < (bw - nF + 1)){
                int inm = imgW * (bx*bh + tx) + by*bw + ty;
                int ind = convW * (bx*bh + ty) + by*bw + ty;
                for (int fi = 0; fi < nF; fi++){
                    int inf = fi * convW;
                    inm += fi * imgW;
                    for (int fj = 0; fj < nF; fj++){
                        inf += fj;
                        inm += fj;
                        out[ind + con] += img[inm + imgg] * filter[inf];
                    }
                }
            }
            __syncthreads();
        }
    }
}

//remember the threads number to be larger than tile size
__global__ void compute_gpu_sm(float *img, float *out, int bh, int bw, int imgH, int imgW, int imgN, int nF, int convH, int convW){
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int nT = blockDim.x;
    int nB = gridDim.x;
    
    int bbw = imgW / bw;
    int bbh = imgH / bh;
    
    __shared__ float sm[ (BLCH + K - 1) * (BLCW + K - 1) ];
    
    for (int i = 0; i < imgN; i++){
        int con = i * convW * convH;
        int imgg = i * imgW * imgH;
        //compute center tiles
        if(bx < (bbh-1) && by < (bbw-1)){
        
            if(tx < (bh+nF-1) && ty < (bw+nF-1)){
                int inm = imgW * (bx*bh + tx) + by*bw + ty;
                //set shared memory
                sm[tx*bw+ty] = img[inm];
            }
            __syncthreads();
            
           if(tx < bh && ty < bw){
                int ind = convW * (bx*bh + ty) + by*bw + ty;
                for (int fi = 0; fi < nF; fi++){
                    int inf = fi * convW;
                    int ins = (tx + fi)*bw + ty;
                    for (int fj = 0; fj < nF; fj++){
                        inf += fj;
                        ins += fj ;
                        out[ind + con] += sm[ins] * filter[inf];
                    }
                }
            }
            __syncthreads();
        }
        //compute right most tiles
        if(bx < (bbh-1) && by == (bbw-1)){
            if(tx < (bh+nF-1) && ty < bw){
                int inm = imgW * (bx*bh + tx) + by*bw + ty;
                //set shared memory
                sm[tx*bw+ty] = img[inm];
            }
            __syncthreads();
            
            if(tx < bh && ty < (bw - nF + 1)){
                int ind = convW * (bx*bh + ty) + by*bw + ty;
                for (int fi = 0; fi < nF; fi++){
                    int inf = fi * convW;
                    int ins = (tx + fi)*bw + ty;
                    for (int fj = 0; fj < nF; fj++){
                        inf += fj;
                        ins += fj;
                        out[ind + con] += sm[ins] * filter[inf];
                    }
                }
            }
            __syncthreads();
        }
        //compute the bottom tiles
        if(bx == (bbh-1) && by < (bbw-1)){
            if(tx < bh && ty < (bw+nF-1)){
                int inm = imgW * (bx*bh + tx) + by*bw + ty;
                //set shared memory
                sm[tx*bw+ty] = img[inm];
            }
            __syncthreads();
            
            if(tx < (bh -nF + 1) && ty < bw){
                int ind = convW * (bx*bh + ty) + by*bw + ty;
                for (int fi = 0; fi < nF; fi++){
                    int inf = fi * convW;
                    int ins = (tx + fi)*bw + ty;
                    for (int fj = 0; fj < nF; fj++){
                        inf += fj;
                        ins += fj;
                        out[ind + con] += sm[ins] * filter[inf];
                    }
                }
            }
            __syncthreads();
        }
        //compute the final tile
        if(bx == (bbh-1) && by == (bbw-1)){
            if(tx < bh && ty < bw){
                int inm = imgW * (bx*bh + tx) + by*bw + ty;
                //set shared memory
                sm[tx*bw+ty] = img[inm];
            }
            __syncthreads();
            
            if(tx < (bh - nF + 1) && ty < (bw - nF + 1)){
                int ind = convW * (bx*bh + ty) + by*bw + ty;
                for (int fi = 0; fi < nF; fi++){
                    int inf = fi * convW;
                    int ins = (tx + fi)*bw + ty;
                    for (int fj = 0; fj < nF; fj++){
                        inf += fj;
                        ins += fj;
                        out[ind + con] += sm[ins] * filter[inf];
                    }
                }
            }
            __syncthreads();
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
    
    // create device array that can hold pixel intensity values in GPU GM
    float *d_img;
    float *d_convolved;
    
    cudaMalloc((void **) &d_img, imgSize);
    cudaMemcpy(d_img, h_img, imgSize, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(cm, filter, filterSize);
    cudaMalloc((void **) &d_convolved, convSize);
    cudaMemcpy(d_convolved, h_convolved, convSize, cudaMemcpyHostToDevice);
    
    struct timeval starttime, endtime;
    double elapsed = 0.0;
    for (int i = 0; i<10000; i++){
        for(int i=0; i<convDims; i++){
            h_convolved[i] = 0.0;
        }
        gettimeofday(&starttime,NULL);
        // call the kernel
        compute_gpu_tiled<<<nB, nT>>>(d_img, d_convolved, blcH, blcW, imgH, imgW, imgN, k, convH, convW);
        gettimeofday(&endtime,NULL);
        elapsed += ((endtime.tv_sec-starttime.tv_sec)*1000000 + endtime.tv_usec-starttime.tv_usec)/1000000.0;
        cudaMemcpy(h_convolved, d_convolved, convSize, cudaMemcpyDeviceToHost);
        cudaDeviceReset();
    }
    printf("Input imgH: %d imgW: %d imgN: %d\n", imgH, imgW, imgN);
    printf("Tile width: %d height: %d\n", blcW, blcH);
    printf("Block number: %d, block size: %d \n", nB, nT);
    printf("time: %f \n", &elapsed);
    delete h_img;
    delete h_convolved;
    return 0;
}


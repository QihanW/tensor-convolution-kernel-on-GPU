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

int compute(float *img, float *f, float * out, int bh, int bw, int imgH, int imgW, int imgN, int nF, int convH, int convW) {

    int inm1, inm2, inm3, inm4, inf, ind1, ind2, ind3;
    inm1 = 0;
    inf = 0;
    ind1 = 0;

    for (int mi = 0; mi < imgN; mi++){
        ind1 += convW * convH;
        inm1 += imgW * imgH;
        for (int mj = 0; mj < convH; mj++){
            ind2 = ind1 + convW * mj;
            inm2 = inm1 + imgW * mj;
            for (int mk = 0; mk < convW; mk++){
                ind3 = ind2 + mk;
                inm3 = inm2 + mk;
                for (int fi = 0; fi < nF; fi++){
                    inm4 = inm3 + imgW * fi;
                    inf = ind3*nF*nF + fi*nF;
                    for (int fj = 0; fj < nF; fj++){
                        out[ind3] += img[inm4+fj] * f[inf+fj];
                    }
                }
            }
        }
    }
    return 0;
}

int compute_csr(float *img, float *f_csr, int *pos, int *coor, float * out, int bh, int bw, int imgH, int imgW, int imgN, int nF, int convH, int convW) {

    int inm1, inm2, inm3, inm4, inf, inf1, inf2, ind1, ind2, ind3;

    for (int mi = 0; mi < imgN; mi++){
        ind1 = mi * convW * convH;
        inm1 = mi * imgW * imgH;
        for (int mj = 0; mj < convH; mj++){
            ind2 = ind1 + convW * mj;
            inm2 = inm1 + imgW * mj;
            for (int mk = 0; mk < convW; mk++){
                ind3 = ind2 + mk;
                inm3 = inm2 + mk;
                inf = nF*nF*ind3;
                for (int fi = 0; fi < nF; fi++){
                    inf1 = pos[fi+inf];
                    inf2 = pos[fi+1+inf];
                    inm4 = inm3 + fi*imgW;
                    for (int fj = inf1; fj < inf2; fj++){
                        out[ind3] += img[inm4+coor[fj]] * f_csr[fj+inf];
                    }

                }
            }
        }
    }
    return 0;
}



__global__ void compute_gpu_csr(float *img, float *f_csr, int *pos, int *coor, float *out, int bh, int bw, int imgH, int imgW, int imgN, int nF, int convH, int convW){
    
    int idX = blockDim.x * blockIdx.x + threadIdx.x;
    int idY = blockDim.y * blockIdx.y + threadIdx.y;
    
    int con, imgg, inm1, inm2, inm3, inm4, inf, inf1, inf2, ind1, ind2, ind3;

    for (int mi = 0; mi < imgN; mi++){
        ind1 = mi * convW * convH;
        inm1 = mi * imgW * imgH;
        if (idX < convH && idY < convW){
            ind2 = ind1 + convW * idX;
            inm2 = inm1 + imgW * idX;
            ind3 = ind2 + idY;
            inm3 = inm2 + idY;
            inf = nF*nF*ind3;
            for (int fi = 0; fi < nF; fi++){
                inf1 = pos[fi+inf];
                inf2 = pos[fi+1+inf];
                inm4 = inm3 + fi*imgW;
                for (int fj = inf1; fj < inf2; fj++){
                    out[ind3] += img[inm4+coor[fj]] * f_csr[fj+inf];
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
    for(int i=0; i<imgDims; i++){
        h_img[i] = (float)(rand()%10485)/10485;
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

    // create filter and copy to constant memory
    int filterDims = k * k * convDims;
    int filterSize = filterDims * sizeof(float);
    float *filter = new float[filterDims];
    num = 0;
    for(int i=0; i<filterDims; i++){
        if(rand()/40 == 0){
            num++;
            filter[i] = (float)(rand()%10485)/10485;
        }
        else{
            filter[i] = 0;
        }   
    }
    
    //create index arrays of CSR
    int *pos = new int[(k+1)*convDims];
    int *coor = new int[num];
    float *h_fcsr = new float[num];
    int csrposSize = (k+1)*convDims*sizeof(int);
    int csrcooSize = num*sizeof(int);
    for(int ki=0; ki<convDims; ki++){
        pos[0+ki*convDims] = 0;
        int index_p=0;
        int z = ki*convDims;
        for(int i=0; i<nF; i++){
            for (int j=0; j<nF; j++){
                if (filter[i*nF+j+z] != 0){
                    coor[index_p+k] = j;
                    h_fcsr[index_p+k] = filter[i*nF+j+z];
                    index_p++;
                }
            }
            pos[i] = index_p;
        }
    }


    // create device array that can hold pixel intensity values in GPU GM
    float *d_img;
    float *d_convolved;
    float *d_filter;
    int *d_pos;
    int *d_coor;
    cudaMemcpyToSymbol(d_filter, h_fcsr, filterSize);
    cudaMalloc((void **) &d_convolved, convSize);
    cudaMemcpy(d_convolved, h_convolved, convSize, cudaMemcpyHostToDevice);
    cudaMalloc((void **) &d_img, imgSize);
    cudaMemcpy(d_img, h_img, imgSize, cudaMemcpyHostToDevice);
    cudaMalloc((void **) &d_pos, csrposSize);
    cudaMemcpy(d_pos, pos, csrposSize, cudaMemcpyHostToDevice);
    cudaMalloc((void **) &d_coor, csrcooSize);
    cudaMemcpy(d_coor, coor, csrcooSize, cudaMemcpyHostToDevice);
    
    struct timeval starttime, endtime;
    double elapsed = 0.0;
    for (int i = 0; i<10000; i++){
        gettimeofday(&starttime,NULL);
        // call the kernel
        //compute_gpu<<<nB, nT>>>(d_img, d_convolved, blcH, blcW, imgH, imgW, imgN, k, convH, convW);
        compute_gpu_csr<<<nB, nT>>>(d_img, d_filter, d_pos, d_coor, d_convolved, blcH, blcW, imgH, imgW, imgN, k, convH, convW);
        gettimeofday(&endtime,NULL);
        elapsed += ((endtime.tv_sec-starttime.tv_sec)*1000000 + endtime.tv_usec-starttime.tv_usec)/1000000.0;
    }
    
    cudaMemcpy(h_convolved, d_convolved, convSize, cudaMemcpyDeviceToHost);
    cudaDeviceReset();
    printf("Input imgH: %d imgW: %d imgN: %d\n", imgH, imgW, imgN);
    printf("Tile width: %d height: %d\n", blcW, blcH);
    printf("Block number: %d, block size: %d \n", nB, nT);
    printf("time: %f \n", elapsed);
    delete h_img;
    delete h_convolved;
    delete h_pos;
    delete h_fcsr;
    delete h_coor;
    return 0;
}

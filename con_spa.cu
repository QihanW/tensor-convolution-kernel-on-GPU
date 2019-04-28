
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

int compute_csr(float *img_csr, float *f, float * out, int *pos, int *coor, int num, int bh, int bw, int imgH, int imgW, int imgN, int nF, int convH, int convW) {

    int con, imgg, ind1, ind2, pj, infi, infj;

    for (int i = 0; i < imgN; i++){
        con = i * convW * convH;
        imgg = i * imgW * imgH;
        //Visit input image by csr
        for (int pi = 0; pi<imgH; pi++){
            ind1 = pos[pi+i*(imgH+1)];
            ind2 = pos[pi+1+i*(imgH+1)];
            for(int ci = ind1; ci<ind2; ci++){
                pj = coor[ci+i*num];
                float value = img_csr[ci+imgg];
                //For every element, compute all filters
                for (int fi = pi-nF+1; fi<=pi; fi++){
                    if (fi>0 && fi<convH){
                        for (int fj = pj-nF+1; fj<=pj; fj++){
                            if (fj>0 && fj<convW){
                                infi = fi-(pi-nF+1);
                                infj = fj-(pj-nF+1);
                                out[fi*convW+fj+con] = value * f[infi*nF+infj];
                            }
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
                    float value = img[inm+imgg];
                    if (value != 0){
                        out[ind+con] += value * filter[inf];
                    }
                }
            }
        }
    }
}

__global__ void compute_gpu_csr(float *img_csr, float * out, int *pos, int *coor, int num, int bh, int bw, int imgH, int imgW, int imgN, int nF, int convH, int convW) {
    
    int pi = blockDim.x * blockIdx.x + threadIdx.x;

    int con, imgg, ind1, ind2, pj, infi, infj;

    for (int i = 0; i < imgN; i++){
        con = i * convW * convH;
        imgg = i * imgW * imgH;
        //Visit input image by csr
        if(pi<imgH){
            ind1 = pos[pi+i*(imgH+1)];
            ind2 = pos[pi+1+i*(imgH+1)];
            for(int ci = ind1; ci<ind2; ci++){
                pj = coor[ci+i*num];
                float value = img_csr[ci+imgg];
                //For every element, compute all filters
                for (int fi = pi-nF+1; fi<=pi; fi++){
                    if (fi>0 && fi<convH){
                        for (int fj = pj-nF+1; fj<=pj; fj++){
                            if (fj>0 && fj<convW){
                                infi = fi-(pi-nF+1);
                                infj = fj-(pj-nF+1);
                                out[fi*convW+fj+con] = value * filter[infi*nF+infj];
                            }
                        }
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
    
    
    for(int k=0; k<imgN; k++){
        for(int i=0; i<imgH; i++){
        for (int j=0; j<imgW; j++){
            if (rand() % 10 == 0){
                num++;
                h_img[i*imgW+j+k*imgH * imgW] = (float)(rand()%10485)/10485;
            }
            else{
                h_img[i*imgW+j+k*imgH * imgW] = 0.0;
            }
        }
    }
    }
    
    //create index arrays of CSR
    int *pos = new int[(imgH+1)*imgN];
    int *coor = new int[num*imgN];
    float *h_imgcsr = new float[num*imgN];
    int csrimgSize = num*imgN*sizeof(float);
    int csrposSize = (imgH+1)*imgN*sizeof(int);
    int csrcooSize = num*imgN*sizeof(int);
    for(int k=0; k<imgN; k++){
        pos[0+k*imgH * imgW] = 0;
        int index_p=0;
        int index_c=0;
        int z = k*imgH * imgW;
        for(int i=0; i<imgH; i++){
            for (int j=0; j<imgW; j++){
                if (h_img[i*imgW+j+k] != 0){
                    coor[index_p+k] = j;
                    h_imgcsr[index_p+k] = h_img[i*imgW+j+k];
                    index_p++;
                
                }
            }
            pos[i] = index_p;
        }
    }
    
    
    // create filter and copy to constant memory
    int filterDims = k * k;
    int filterSize = filterDims * sizeof(float);
    float *f = new float[filterDims];
    for(int i=0; i<filterDims; i++){
        f[i] = (float)(rand()%10485)/10485;
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
    float *d_img_csr;
    float *d_convolved;
    int *d_pos;
    int *d_coor;
    cudaMemcpyToSymbol(filter, f, filterSize);
    cudaMalloc((void **) &d_convolved, convSize);
    cudaMemcpy(d_convolved, h_convolved, convSize, cudaMemcpyHostToDevice);
    cudaMalloc((void **) &d_img_csr, csrimgSize);
    cudaMemcpy(d_img_csr, h_imgcsr, csrimgSize, cudaMemcpyHostToDevice);
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
        compute_gpu_csr<<<nB, nT>>>(d_img_csr, d_convolved, d_pos, d_coor, num, blcH, blcW, imgH, imgW, imgN, k, convH, convW);
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
    delete pos;
    delete h_imgcsr;
    delete coor;
    return 0;
}


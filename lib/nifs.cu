#include "erl_nif.h"
#include "cublas.h"
#include "stdio.h"
#include "time.h"


#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define IDX3C(c,i,j,in_h,in_w) ((c)*((in_h)*(in_w)) + (i)*(in_w) +(j))
#define IDX4C(n,c,i,j,in_c,in_h,in_w) ((n)*((in_c)*(in_h)*(in_w)) + (c)*((in_h)*(in_w)) + (i)*(in_w) +(j))
#define BREAK return(enif_make_int(env, 0));
#define PI 3.14159265358979323846
#define SIGMOID(x)  (1 / (1+exp(-1*x)))
#define DEBUG 0
#define DISP(x) if(DEBUG){printf(x);fflush(stdout);}
#define CHECK(call)                                   \
{                                                     \
    const cudaError_t error = call;                   \
    if (error != cudaSuccess)                         \
    {                                                 \
        return enif_make_int(env,10000+(int)error);   \
    }                                                 \
}
#define CUBLAS(call)                                  \
{                                                     \
    const cublasStatus error = call;                  \
    if (error != CUBLAS_STATUS_SUCCESS)               \
    {                                                 \
        return enif_make_int(env,11000+(int)error);   \
    }                                                 \
}

__global__ void pooling_kernel(float *a, float *b, float *c, int st, int in_c, int in_h, int in_w, int n)
{
    int tid = threadIdx.x;
    int n1,c1,h1,w1,h2,w2,in_h2,in_w2,start_h1,end_h1,start_w1,end_w1,max_h,max_w;
    float max,fmax_h,fmax_w;
    if(tid < n)
    {   
        n1 = tid;
        in_h2 = in_h / st;
        in_w2 = in_w / st;
        for(c1=0;c1<in_c;c1++){
            for(w2=0;w2<in_w2;w2++){
                for(h2=0;h2<in_h2;h2++){
                    max = 0.0;
                    start_h1 = st*h2;
                    end_h1 = st*(h2+1);
                    start_w1 = st*w2;
                    end_w1 = st*(w2+1);
                    for(h1=start_h1;h1<end_h1;h1++){
                        for(w1=start_w1;w1<end_w1;w1++){
                            if(a[IDX4C(n1,c1,h1,w1,in_c,in_h,in_w)] >= max){
                                max = a[IDX4C(n1,c1,h1,w1,in_c,in_h,in_w)];
                                max_h = h1;
                                max_w = w1;
                            }
                        }
                    }
                    b[IDX4C(n1,c1,h2,w2,in_c,in_h2,in_w2)] = max;
                    fmax_h = (float)max_h;
                    fmax_w = (float)max_w;
                    c[IDX4C(n1,c1,h2,w2,in_c,in_h2,in_w2)] = fmax_h * 1000.0 + fmax_w; 
                }
            }
        }
    }
}
  
  /*
  1st arg in_n of tensor
  2nd arg in_c of tensor
  3rd arg in_h of tensor
  4th arg in_w of tensor
  5th arg binary of tensor
  6th arg stride 

  return list [ts1,ts2]
  ts1 is result data for forward
  ts2 is result data dor backward. this is sparse matrix 
  e.g. 
  |0.1,0.2,0.3,0.4|
  |0.5,0.6,0.7,0.8|
  |0.9,1.0,1.1,1.2|
  |1.3,1.4,1.5,1.6|
  
  ts1
  |0.6,0.8|
  |1.4,1.6|

  ts2
  each element is  row*1000+col
  |1.0*1000+1.0,1.0*1000*3.0|
  |3.0*1000+1.0,3.0*1000+3.0|
  
  */
static ERL_NIF_TERM
pooling1(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary  a_bin;
    ERL_NIF_TERM  b_bin,c_bin,tuple;
    int in_n,in_c,in_h,in_w,st, n1, n2;
    float *a,*b, *c;
    float *dev_a, *dev_b, *dev_c;
  
    DISP("pooling1")
    if (!enif_get_int(env, argv[0], &in_n)) return enif_make_int(env,1);
    if (!enif_get_int(env, argv[1], &in_c)) return enif_make_int(env,2);
    if (!enif_get_int(env, argv[2], &in_h)) return enif_make_int(env,3);
    if (!enif_get_int(env, argv[3], &in_w)) return enif_make_int(env,4);
    if (!enif_inspect_binary(env, argv[4], &a_bin )) return enif_make_int(env,5);
    if (!enif_get_int(env, argv[5], &st)) return enif_make_int(env,6);

    n1 = in_n * in_c * in_h * in_w;
    n2 = in_n * in_c * (in_h / st) * (in_w / st);
    a = (float *) a_bin.data;
    b = (float *) enif_make_new_binary(env,  n2 * sizeof(float), &b_bin);
    c = (float *) enif_make_new_binary(env,  n2 * sizeof(float), &c_bin);

   
    // Allocate for GPU
    CHECK(cudaMalloc((void**)&dev_a, n1 * sizeof(float)));
    CHECK(cudaMalloc((void**)&dev_b, n2 * sizeof(float)));
    CHECK(cudaMalloc((void**)&dev_c, n2 * sizeof(float)));
  
    // copy from host a,b to GPU dev_a, dev_b
    CHECK(cudaMemcpy(dev_a, a, n1 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_b, b, n2 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_c, c, n2 * sizeof(float), cudaMemcpyHostToDevice));
  
    pooling_kernel << <1, in_n>> >(dev_a, dev_b, dev_c, st, in_c, in_h, in_w, in_n);
  
    // copy to host b,c from GPU dev_b,dev_c
    CHECK(cudaMemcpy(b, dev_b, n2 * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(c, dev_c, n2 * sizeof(float), cudaMemcpyDeviceToHost));
      

    // return forward data and backward data with tuple {b_bin,c_bin} 
    tuple = enif_make_tuple2(env,b_bin,c_bin);
    
    // free 
    cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

    return(tuple);
}


__global__ void unpooling_kernel(float *a, float *b, float *c, int st, int in_c, int in_h, int in_w, int n)
{
    int tid = threadIdx.x;
    int n1,c1,h1,w1,h2,w2,start_h1,end_h1,start_w1,end_w1,max_h,max_w,in_h1,in_w1;
    float loss,elt;
    if(tid < n)
    {   
        n1 = tid;
        in_h1 = in_h * st;
        in_w1 = in_w * st;
        for(c1=0;c1<in_c;c1++){
            for(h2=0;h2<in_h;h2++){
                for(w2=0;w2<in_w;w2++){
                    start_h1 = st*h2;
                    end_h1 = st*(h2+1);
                    start_w1 = st*w2;
                    end_w1 = st*(w2+1);
                    elt = a[IDX4C(n1,c1,h2,w2,in_c,in_h,in_w)];
                    loss = b[IDX4C(n1,c1,h2,w2,in_c,in_h,in_w)];
                    max_h = (int) floor(elt / 1000.0);
                    max_w = (int) fmodf(elt,1000.0);
                    for(h1=start_h1;h1<end_h1;h1++){
                        for(w1=start_w1;w1<end_w1;w1++){
                            if(h1 == max_h && w1 == max_w){
                                c[IDX4C(n1,c1,h1,w1,in_c,in_h1,in_w1)] = loss;
                            }
                            else{
                                c[IDX4C(n1,c1,h1,w1,in_c,in_h1,in_w1)] = 0.0;
                            }
                        }
                    }
                }
            }
        }
    }
}
  
/*
1st arg in_n of sparse-tensor
2nd arg in_c of sparse-tensor
3rd arg in_h of sparse-tensor
4th arg in_w of sparse-tensor
5th arg binary of sparse-tensor
6th arg binary of loss-tensor
7th arg stride 

return gradiate tensor
e.g.
ts1 index-tensor
  each element is  row*1000+col
  |1.0*1000+1.0,1.0*1000*3.0|
  |3.0*1000+1.0,3.0*1000+3.0|
ts2 loss-tensor
  |0.1,0.2|
  |0.3,0.4|

return
  |0.0,0.0,0.0,0.0|
  |0.0,0.1,0.0,0.2|
  |0.0,0.0,0.0,0.0|
  |0.0,3.4,0.0,0.4|

*/
static ERL_NIF_TERM
unpooling1(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary  a_bin,b_bin;
    ERL_NIF_TERM  c_bin;
    int in_n,in_c,in_h,in_w,st, n1, n2;
    float *a,*b, *c;
    float *dev_a, *dev_b, *dev_c;
  
    DISP("unpooling")
    if (!enif_get_int(env, argv[0], &in_n)) return enif_make_int(env,1);
    if (!enif_get_int(env, argv[1], &in_c)) return enif_make_int(env,2);
    if (!enif_get_int(env, argv[2], &in_h)) return enif_make_int(env,3);
    if (!enif_get_int(env, argv[3], &in_w)) return enif_make_int(env,4);
    if (!enif_inspect_binary(env, argv[4], &a_bin )) return enif_make_int(env,5);
    if (!enif_inspect_binary(env, argv[5], &b_bin )) return enif_make_int(env,6);
    if (!enif_get_int(env, argv[6], &st)) return enif_make_int(env,7);

    n1 = in_n * in_c * in_h * in_w;
    n2 = in_n * in_c * (in_h * st) * (in_w * st);
    a = (float *) a_bin.data;
    b = (float *) b_bin.data;
    c = (float *) enif_make_new_binary(env,  n2 * sizeof(float), &c_bin);


      
    // Allocate for GPU
    CHECK(cudaMalloc((void**)&dev_a, n1 * sizeof(float)));
    CHECK(cudaMalloc((void**)&dev_b, n1 * sizeof(float)));
    CHECK(cudaMalloc((void**)&dev_c, n2 * sizeof(float)));

  
    // copy from host a,b to GPU dev_a, dev_b
    CHECK(cudaMemcpy(dev_a, a, n1 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_b, b, n1 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_c, c, n2 * sizeof(float), cudaMemcpyHostToDevice));
  
    unpooling_kernel << <1, in_n>> >(dev_a, dev_b, dev_c, st, in_c, in_h, in_w, in_n);
  
    // copy to host d from GPU dev_d
    CHECK(cudaMemcpy(c, dev_c, n2 * sizeof(float), cudaMemcpyDeviceToHost));

    // free 
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return(c_bin);
}

  
__global__ void convolute_kernel(float *a, float *b, float *c, int filt_h, int filt_w, int st, int pad, int in_c, int in_h, int in_w, int n)
{
    int tid = threadIdx.x;
    int n1,c1,h1,w1,h2,w2,oh,ow,start_h1,end_h1,start_w1,end_w1;
    float sum,elt1,elt2;
    
    if(tid < n)
    {   
        n1 = tid;
        oh = (in_h+2*pad-filt_h)/st + 1;
        ow = (in_w+2*pad-filt_w)/st + 1;
        for(w2=0;w2<ow;w2++){
            for(h2=0;h2<oh;h2++){
                sum = 0.0;
                start_h1 = st*h2-pad;
                end_h1 = start_h1 + filt_h;
                start_w1 = st*w2-pad;
                end_w1 = start_w1 + filt_w;
                for(c1=0;c1<in_c;c1++){
                    for(h1=start_h1;h1<end_h1;h1++){
                        for(w1=start_w1;w1<end_w1;w1++){
                            if(h1 >= 0 && h1 < in_h && w1 >= 0 && w1 < in_w){
                                elt1 = a[IDX4C(n1,c1,h1,w1,in_c,in_h,in_w)];
                                elt2 = b[IDX3C(c1,h1-start_h1,w1-start_w1,filt_h,filt_w)];
                                sum = sum + elt1*elt2;
                            }
                        }
                    }
                }
                c[IDX4C(n1,0,h2,w2,in_c,oh,ow)] = sum;   
              }
          }
    }
}
  
/*
1st arg in_n of input tensor
2nd arg in_c of input tensor
3rd arg in_h of input tensor
4th arg in_w of input tensor
5th arg filt_h of filter tensor
6th arg filt_w of filter tensor
7th arg binary of input tensor
8th arg binary of filter tensor
9th arg stride
10th arg padding   
*/
static ERL_NIF_TERM
convolute1(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary  a_bin,b_bin;
    ERL_NIF_TERM  c_bin;
    int in_n,in_c,in_h,in_w,filt_h, filt_w, st,pad, n1, n2, n3, oh, ow;
    float *a,*b, *c;
    float *dev_a, *dev_b, *dev_c;
  
    DISP("convolute1")
    if (!enif_get_int(env, argv[0], &in_n)) return enif_make_int(env,1);
    if (!enif_get_int(env, argv[1], &in_c)) return enif_make_int(env,2);
    if (!enif_get_int(env, argv[2], &in_h)) return enif_make_int(env,3);
    if (!enif_get_int(env, argv[3], &in_w)) return enif_make_int(env,4);
    if (!enif_get_int(env, argv[4], &filt_h)) return enif_make_int(env,5);
    if (!enif_get_int(env, argv[5], &filt_w)) return enif_make_int(env,6);
    if (!enif_inspect_binary(env, argv[6], &a_bin )) return enif_make_int(env,7);
    if (!enif_inspect_binary(env, argv[7], &b_bin )) return enif_make_int(env,8);
    if (!enif_get_int(env, argv[8], &st)) return enif_make_int(env,9);
    if (!enif_get_int(env, argv[9], &pad)) return enif_make_int(env,10);

    
    n1 = in_n * in_c * in_h * in_w;
    n2 = in_c * filt_h * filt_w;
    oh = (in_h+2*pad-filt_h)/st + 1;
    ow = (in_w+2*pad-filt_w)/st + 1;
    n3 = in_n * oh * ow;
    a = (float *) a_bin.data;
    b = (float *) b_bin.data;
    c = (float *) enif_make_new_binary(env,  n3 * sizeof(float), &c_bin);
  
    // Allocate for GPU
    CHECK(cudaMalloc((void**)&dev_a, n1 * sizeof(float)));
    CHECK(cudaMalloc((void**)&dev_b, n2 * sizeof(float)));
    CHECK(cudaMalloc((void**)&dev_c, n3 * sizeof(float)));

  
    // copy from host a,b,c to GPU dev_a, dev_b, dev_c
    CHECK(cudaMemcpy(dev_a, a, n1 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_b, b, n2 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_c, c, n3 * sizeof(float), cudaMemcpyHostToDevice));

    convolute_kernel << <1, in_n>> >(dev_a, dev_b, dev_c, filt_h, filt_w, st, pad, in_c, in_h, in_w, in_n);
  
    // copy to host c from GPU dev_c
    CHECK(cudaMemcpy(c, dev_c, n3 * sizeof(float), cudaMemcpyDeviceToHost));

    // free 
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    
    return(c_bin);
}

  
__global__ void deconvolute1_kernel(float *a, float *b, float *c, int filt_h, int filt_w, int st, int pad, int in_c, int in_h, int in_w, int n)
{
    int tid = threadIdx.x;
    int n1,c1,h1,w1,h2,w2,oh,ow,start_h1,end_h1,start_w1,end_w1;
    float sum,elt1,elt2;
    if(tid < n)
    {   
        n1 = tid;
        oh = (in_h+2*pad-filt_h)/st + 1;
        ow = (in_w+2*pad-filt_w)/st + 1;
        for(w2=0;w2<ow;w2++){
            for(h2=0;h2<oh;h2++){
                sum = 0.0;
                start_h1 = st*h2-pad;
                end_h1 = start_h1 + filt_h;
                start_w1 = st*w2-pad;
                end_w1 = start_w1 + filt_w;
                for(c1=0;c1<in_c;c1++){
                    for(h1=start_h1;h1<end_h1;h1++){
                        for(w1=start_w1;w1<end_w1;w1++){
                            if(h1 >= 0 && h1 < in_h && w1 >= 0 && w1 < in_w){
                                elt1 = a[IDX4C(n1,0,h1,w1,in_c,in_h,in_w)];
                                elt2 = b[IDX3C(c1,h1-start_h1,w1-start_w1,filt_h,filt_w)];
                                sum = sum + elt1*elt2;
                            }
                        }
                    }
                    c[IDX4C(n1,c1,h2,w2,in_c,oh,ow)] = sum;  
                }
                 
            }
        }
    }
}
  
/*
1st arg in_n of input tensor
2nd arg in_c of input tensor
3rd arg in_h of input tensor
4th arg in_w of input tensor
5th arg filt_h of filter tensor
6th arg filt_w of filter tensor
7th arg binary of input tensor
8th arg binary of filter tensor
9th arg stride
10th arg padding   
*/
static ERL_NIF_TERM
deconvolute1(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary  a_bin,b_bin;
    ERL_NIF_TERM  c_bin;
    int in_n,in_c,in_h,in_w,filt_h, filt_w, st,pad, pad1, n1, n2, n3, oh, ow, i,j,k;
    float *a,*b, *b1, *c;
    float *dev_a, *dev_b, *dev_c;
  
    DISP("deconvolute1")
    if (!enif_get_int(env, argv[0], &in_n)) return enif_make_int(env,1);
    if (!enif_get_int(env, argv[1], &in_c)) return enif_make_int(env,2);
    if (!enif_get_int(env, argv[2], &in_h)) return enif_make_int(env,3);
    if (!enif_get_int(env, argv[3], &in_w)) return enif_make_int(env,4);
    if (!enif_get_int(env, argv[4], &filt_h)) return enif_make_int(env,5);
    if (!enif_get_int(env, argv[5], &filt_w)) return enif_make_int(env,6);
    if (!enif_inspect_binary(env, argv[6], &a_bin )) return enif_make_int(env,7);
    if (!enif_inspect_binary(env, argv[7], &b_bin )) return enif_make_int(env,8);
    if (!enif_get_int(env, argv[8], &st)) return enif_make_int(env,9);
    if (!enif_get_int(env, argv[9], &pad)) return enif_make_int(env,10);

    

    n1 = in_n * in_c * in_h * in_w;
    n2 = in_c * filt_h * filt_w;
    pad1 = filt_h - 1 + pad;
    oh = (in_h+2*pad1-filt_h)/st + 1;
    ow = (in_w+2*pad1-filt_w)/st + 1;
    n3 = in_n * oh * ow;
    a = (float *) a_bin.data;
    b = (float *) b_bin.data;
    b1 = (float *) enif_alloc(n2 * sizeof(float));
    c = (float *) enif_make_new_binary(env,  n3 * sizeof(float), &c_bin);
  
      
    //rotate 180 degree  
    for(i=0;i<in_c;i++){
        for(j=0;j<filt_h;j++){
            for(k=0;k<filt_w;k++){
                //if(IDX3C(i,filt_h-j-1,filt_w-k-1,filt_h,filt_w) >= n2) return enif_make_int(env,11001);
                b1[IDX3C(i,filt_h-j-1,filt_w-k-1,filt_h,filt_w)] = b[IDX3C(i,j,k,filt_h,filt_w)];
            }
        }
    }
        

    /*
    for(i=0;i<in_c;i++){
        for(j=0;j<filt_h;j++){
            for(k=0;k<filt_w;k++){
                printf("%f",  b1[IDX3C(i,j,k,filt_h,filt_w)]);
            }
        }
    }
    */
    // Allocate for GPU
    CHECK(cudaMalloc((void**)&dev_a, n1 * sizeof(float)));
    CHECK(cudaMalloc((void**)&dev_b, n2 * sizeof(float)));
    CHECK(cudaMalloc((void**)&dev_c, n3 * sizeof(float)));

  
    // copy from host a,b1,c to GPU dev_a, dev_b, dev_c
    CHECK(cudaMemcpy(dev_a, a, n1 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_b, b1, n2 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_c, c, n3 * sizeof(float), cudaMemcpyHostToDevice));

    deconvolute1_kernel << <1, in_n>> >(dev_a, dev_b, dev_c, filt_h, filt_w, st, pad1, in_c, in_h, in_w, in_n);
  
    // copy to host c from GPU dev_c
    CHECK(cudaMemcpy(c, dev_c, n3 * sizeof(float), cudaMemcpyDeviceToHost));

    // free 
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    enif_free(b1);
    
    
    return(c_bin);
}


__global__ void deconvolute2_kernel(float *a, float *b, float *c, int filt_h, int filt_w, int st, int pad, int in_c, int in_h, int in_w, int n)
{
    int tid = threadIdx.x;
    int n1,c1,h1,w1,h2,w2,oh,ow,start_h1,end_h1,start_w1,end_w1;
    float sum,elt1,elt2;
    if(tid < n)
    {   
        n1 = tid;
        oh = (in_h+2*pad-filt_h)/st + 1;
        ow = (in_w+2*pad-filt_w)/st + 1;

        //convulute
        for(w2=0;w2<ow;w2++){
            for(h2=0;h2<oh;h2++){
                sum = 0.0;
                start_h1 = st*h2-pad;
                end_h1 = start_h1 + filt_h;
                start_w1 = st*w2-pad;
                end_w1 = start_w1 + filt_w;
                for(c1=0;c1<in_c;c1++){
                    for(h1=start_h1;h1<end_h1;h1++){
                        for(w1=start_w1;w1<end_w1;w1++){
                            if(h1 >= 0 && h1 < in_h && w1 >= 0 && w1 < in_w){
                                elt1 = a[IDX4C(n1,0,h1,w1,in_c,in_h,in_w)];
                                elt2 = b[IDX3C(c1,h1-start_h1,w1-start_w1,filt_h,filt_w)];
                                sum = sum + elt1*elt2;
                            }
                        }
                    }
                    c[IDX4C(n1,c1,h2,w2,in_c,oh,ow)] = sum;  
                }
                 
            }
        }
    }
}
  
/*
1st arg in_n of input tensor
2nd arg in_c of input tensor
3rd arg in_h of input tensor
4th arg in_w of input tensor
5th arg filt_h of filter tensor
6th arg filt_w of filter tensor
7th arg binary of input tensor
8th arg binary of filter tensor
9th arg stride
10th arg padding   
*/
static ERL_NIF_TERM
deconvolute2(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary  a_bin,b_bin;
    ERL_NIF_TERM  c_bin;
    int in_n,in_c,in_h,in_w,filt_h, filt_w, st,pad, pad1, n1, n2, n3, oh, ow, i,j,k;
    float *a,*b, *b1, *c;
    float *dev_a, *dev_b, *dev_c;
    // when st > 1 dilate loss tensor
    int in_h1,in_w1,l,k1,l1;
    float *a1;
    float elt;
  
    DISP("deconvolute2")
    if (!enif_get_int(env, argv[0], &in_n)) return enif_make_int(env,1);
    if (!enif_get_int(env, argv[1], &in_c)) return enif_make_int(env,2);
    if (!enif_get_int(env, argv[2], &in_h)) return enif_make_int(env,3);
    if (!enif_get_int(env, argv[3], &in_w)) return enif_make_int(env,4);
    if (!enif_get_int(env, argv[4], &filt_h)) return enif_make_int(env,5);
    if (!enif_get_int(env, argv[5], &filt_w)) return enif_make_int(env,6);
    if (!enif_inspect_binary(env, argv[6], &a_bin )) return enif_make_int(env,7);
    if (!enif_inspect_binary(env, argv[7], &b_bin )) return enif_make_int(env,8);
    if (!enif_get_int(env, argv[8], &st)) return enif_make_int(env,9);
    if (!enif_get_int(env, argv[9], &pad)) return enif_make_int(env,10);

        
    // reset size for dilate
    in_h1 = filt_h * st * (in_h - 1) + filt_h;
    in_w1 = filt_w * st * (in_w - 1) + filt_w;

    n1 = in_n * in_c * in_h1 * in_w1;
    n2 = in_c * filt_h * filt_w;
    pad1 = filt_h - 1 + pad;
    oh = (in_h1+2*pad1-filt_h)/st + 1;
    ow = (in_w1+2*pad1-filt_w)/st + 1;
    n3 = in_n * oh * ow;
    a = (float *) a_bin.data;
    b = (float *) b_bin.data;
    a1 = (float *) enif_alloc(n1 * sizeof(float));
    b1 = (float *) enif_alloc(n2 * sizeof(float));
    c = (float *) enif_make_new_binary(env,  n3 * sizeof(float), &c_bin);

    //rotate 180 degree  
    for(i=0;i<in_c;i++){
        for(j=0;j<filt_h;j++){
            for(k=0;k<filt_w;k++){
                //if(IDX3C(i,filt_h-j-1,filt_w-k-1,filt_h,filt_w) >= n2) return enif_make_int(env,11001);
                b1[IDX3C(i,filt_h-j-1,filt_w-k-1,filt_h,filt_w)] = b[IDX3C(i,j,k,filt_h,filt_w)];
            }
        }
    }

    // dilate 
    for(i=0;i<n1;i++){
        a1[i] = 0.0;
    }
    
    for(i=0;i<in_n;i++){
        for(j=0;j<in_c;j++){
            for(k=0;k<in_h;k++){
                for(l=0;l<in_w;l++){
                    elt = a[IDX4C(i,j,k,l,in_c,in_h,in_w)];
                    k1 = k * filt_h * st;
                    l1 = l * filt_w * st;
                    a1[IDX4C(i,j,k1,l1,in_c,in_h1,in_w1)] = elt;
                }
            }
        }
    }
    
    CHECK(cudaMalloc((void**)&dev_a, n1 * sizeof(float)));
    CHECK(cudaMalloc((void**)&dev_b, n2 * sizeof(float)));
    CHECK(cudaMalloc((void**)&dev_c, n3 * sizeof(float)));

    deconvolute2_kernel << <1, in_n>> >(dev_a, dev_b, dev_c, filt_h, filt_w, st, pad1, in_c, in_h1, in_w1, in_n);

    CHECK(cudaMemcpy(dev_a, a1, n1 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_b, b1, n2 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_c, c, n3 * sizeof(float), cudaMemcpyHostToDevice));

    deconvolute2_kernel << <1, in_n>> >(dev_a, dev_b, dev_c, filt_h, filt_w, st, pad1, in_c, in_h1, in_w1, in_n);
  
    // copy to host c from GPU dev_c
    CHECK(cudaMemcpy(c, dev_c, n3 * sizeof(float), cudaMemcpyDeviceToHost));

    // free 
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    enif_free(a1);
    enif_free(b1);
  
    return(c_bin);
}


__global__ void gradfilter_kernel(float *a, float *b, float *c, int filt_h, int filt_w, int loss_h, int loss_w, int st, int pad, int in_c, int in_h, int in_w, int n)
{
    int tid = threadIdx.x;
    int n1,c1,h1,w1,h2,w2,start_h1,end_h1,start_w1,end_w1;
    float sum,elt1,elt2;
    if(tid < n)
    {   
        n1 = tid;
        for(h2=0;h2<loss_h;h2++){
            for(w2=0;w2<loss_w;w2++){
                sum = 0.0;
                start_h1 = st*h2-pad;
                end_h1 = start_h1 + filt_h;
                start_w1 = st*w2-pad;
                end_w1 = start_w1 + filt_w;

                for(c1=0;c1<in_c;c1++){
                    for(h1=start_h1;h1<end_h1;h1=h1+st){
                        for(w1=start_w1;w1<end_w1;w1=w1+st){
                            if(h1 >= 0 && h1 < in_h && w1 >= 0 && w1 < in_w){
                                elt1 = a[IDX4C(n1,c1,h1,w1,in_c,in_h,in_w)];
                                elt2 = b[IDX4C(n1,0,h1-start_h1,w1-start_w1,in_c,loss_h,loss_w)];
                                sum = sum + elt1*elt2;
                            }
                        }
                    }
                    c[IDX3C(c1,h2,w2,filt_h,filt_w)] = sum;
                }
            }
        }
    }
}
  
/*
1st arg in_n of input tensor
2nd arg in_c of input tensor
3rd arg in_h of input tensor
4th arg in_w of input tensor
5th arg filt_h of filter tensor
6th arg filt_w of filter tensor
7th arg loss_h of loss tensor
8th arg loss_w of loss tensor
9th arg binary of input tensor
10th arg binary of loss tensor
11th arg stride
12th arg padding   
*/
static ERL_NIF_TERM
gradfilter1(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary  a_bin,b_bin;
    ERL_NIF_TERM  c_bin;
    int in_n,in_c,in_h,in_w,filt_h,filt_w,loss_h,loss_w,st,pad,n1,n2,n3;
    float *a,*b,*c;
    float *dev_a, *dev_b, *dev_c;
  
    DISP("gradfilter1")
    if (!enif_get_int(env, argv[0], &in_n)) return enif_make_int(env,1);
    if (!enif_get_int(env, argv[1], &in_c)) return enif_make_int(env,2);
    if (!enif_get_int(env, argv[2], &in_h)) return enif_make_int(env,3);
    if (!enif_get_int(env, argv[3], &in_w)) return enif_make_int(env,4);
    if (!enif_get_int(env, argv[4], &filt_h)) return enif_make_int(env,5);
    if (!enif_get_int(env, argv[5], &filt_w)) return enif_make_int(env,6);
    if (!enif_get_int(env, argv[6], &loss_h)) return enif_make_int(env,7);
    if (!enif_get_int(env, argv[7], &loss_w)) return enif_make_int(env,8);
    if (!enif_inspect_binary(env, argv[8], &a_bin )) return enif_make_int(env,9);
    if (!enif_inspect_binary(env, argv[9], &b_bin )) return enif_make_int(env,10);
    if (!enif_get_int(env, argv[10], &st)) return enif_make_int(env,10);
    if (!enif_get_int(env, argv[11], &pad)) return enif_make_int(env,11);

    n1 = in_n * in_c * in_h * in_w;
    n2 = in_n * loss_h * loss_w;
    n3 = in_c * filt_h * filt_w;
    a = (float *) a_bin.data;
    b = (float *) b_bin.data;
    c = (float *) enif_make_new_binary(env,  n3 * sizeof(float), &c_bin);
  
      
    // Allocate for GPU
    CHECK(cudaMalloc((void**)&dev_a, n1 * sizeof(float)));
    CHECK(cudaMalloc((void**)&dev_b, n2 * sizeof(float)));
    CHECK(cudaMalloc((void**)&dev_c, n3 * sizeof(float)));

    
    // copy from host a,b,c to GPU dev_a, dev_b, dev_c
    CHECK(cudaMemcpy(dev_a, a, n1 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_b, b, n2 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_c, c, n3 * sizeof(float), cudaMemcpyHostToDevice));

    gradfilter_kernel << <1, in_n>> >(dev_a, dev_b, dev_c, filt_h, filt_w, loss_h, loss_w, st, pad, in_c, in_h, in_w, in_n);
  
    // copy to host d from GPU dev_d
    CHECK(cudaMemcpy(c, dev_c, n3 * sizeof(float), cudaMemcpyDeviceToHost));

    // free 
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
  
    return(c_bin);
}


__global__ void full_kernel(float *a, float *b, int in_h, int in_w, int n)
{
    int tid = threadIdx.x;
    int n1,i,j;
    float elt;
    if(tid < n)
    {   
        n1 = tid;
        for(i=0;i<in_h;i++){
            for(j=0;j<in_w;j++){
                elt = a[IDX4C(n1,0,i,j,1,in_h,in_w)];
                b[IDX2C(n1,i*in_w + j,n)] = elt;
            }
        }
    }
}
  
/*
1st arg in_n of input tensor
2rd arg in_h of input tensor
3rd arg in_w of input tensor
4th arg binary of input tensor
*/
static ERL_NIF_TERM
full1(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary  a_bin;
    ERL_NIF_TERM  b_bin;
    int in_n,in_h,in_w,n1;
    float *a,*b;
    float *dev_a, *dev_b;

    DISP("full1")
    if (!enif_get_int(env, argv[0], &in_n)) return enif_make_int(env,1);
    if (!enif_get_int(env, argv[1], &in_h)) return enif_make_int(env,2);
    if (!enif_get_int(env, argv[2], &in_w)) return enif_make_int(env,3);
    if (!enif_inspect_binary(env, argv[3], &a_bin )) return enif_make_int(env,4);

    //printf("%d %d %d \n\r", in_n, in_h, in_w);

    // in_c is allways 1 
    n1 = in_n * in_h * in_w;
    a = (float *) a_bin.data;
    b = (float *) enif_make_new_binary(env,  n1 * sizeof(float), &b_bin);
  
      
    // Allocate for GPU
    CHECK(cudaMalloc((void**)&dev_a, n1 * sizeof(float)));
    CHECK(cudaMalloc((void**)&dev_b, n1 * sizeof(float)));
  
    // copy from host a,b to GPU dev_a, dev_b
    CHECK(cudaMemcpy(dev_a, a, n1 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_b, b, n1 * sizeof(float), cudaMemcpyHostToDevice));

    full_kernel << <1, in_n>> >(dev_a, dev_b, in_h, in_w, in_n);
  
    // copy to host d from GPU dev_d
    CHECK(cudaMemcpy(b, dev_b, n1 * sizeof(float), cudaMemcpyDeviceToHost));

    // free 
    cudaFree(dev_a);
    cudaFree(dev_b);
  
    return(b_bin);
}


__global__ void unfull_kernel(float *a, float *b, int in_h, int in_w, int n)
{
    int tid = threadIdx.x;
    int n1,i,j;
    float elt;
    if(tid < n)
    {   
        n1 = tid;
        for(i=0;i<in_h;i++){
            for(j=0;j<in_w;j++){
                elt = a[IDX4C(n1,0,i,j,1,in_h,in_w)];
                b[IDX2C(n1,i*in_w + j,n)] = elt;
            }
        }
    }
}
  
/*
1st arg in_n of input tensor
2rd arg in_h of input tensor
3th arg in_w of input tensor
4th arg binary of input tensor
*/
static ERL_NIF_TERM
unfull1(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary  a_bin;
    ERL_NIF_TERM  b_bin;
    int in_n,in_h,in_w,n1;
    float *a,*b;
    float *dev_a, *dev_b;
    
    DISP("unfull1")
    if (!enif_get_int(env, argv[0], &in_n)) return enif_make_int(env,1);
    if (!enif_get_int(env, argv[1], &in_h)) return enif_make_int(env,2);
    if (!enif_get_int(env, argv[2], &in_w)) return enif_make_int(env,3);
    if (!enif_inspect_binary(env, argv[3], &a_bin )) return enif_make_int(env,4);

    // in_c is allways 1 
    n1 = in_n * in_h * in_w;
    a = (float *) a_bin.data;
    b = (float *) enif_make_new_binary(env,  n1 * sizeof(float), &b_bin);
  
      
      // Allocate for GPU
    CHECK(cudaMalloc((void**)&dev_a, n1 * sizeof(float)));
    CHECK(cudaMalloc((void**)&dev_b, n1 * sizeof(float)));
  
    // copy from host a,b1,c to GPU dev_a, dev_b, dev_c
    CHECK(cudaMemcpy(dev_a, a, n1 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_b, b, n1 * sizeof(float), cudaMemcpyHostToDevice));

    unfull_kernel << <1, in_n>> >(dev_a, dev_b, in_h, in_w, in_n);
  
    // copy to host d from GPU dev_d
    CHECK(cudaMemcpy(b, dev_b, n1 * sizeof(float), cudaMemcpyDeviceToHost));

    // free 
    cudaFree(dev_a);
    cudaFree(dev_b);
  
    return(b_bin);
}



static ERL_NIF_TERM
print1(ErlNifEnv *env, int argc, const ERL_NIF_TERM *argv) {
    ErlNifBinary  a_bin;
    ERL_NIF_TERM  result;
    float        *a;
    int r,c,i,j;


    DISP("print1")
    if (!enif_get_int(env, argv[0], &r )) return enif_make_int(env,1);
    if (!enif_get_int(env, argv[1], &c )) return enif_make_int(env,2);
    if (!enif_inspect_binary(env, argv[2], &a_bin )) return enif_make_int(env,3);
    
    a  = (float *) a_bin.data;
  
    for(i=0;i<r;i++){
        for(j=0;j<c;j++){
            printf("%f ", a[IDX2C(i,j,r)]);
        }
        printf("\n\r"); 
    }
    printf("\n\r"); 
    result = enif_make_atom(env,"true");

    return result;
}


static ERL_NIF_TERM
new1(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    int n,i;
    ERL_NIF_TERM a_bin;
    float *a;
    double d;

    DISP("new1")
    if (!enif_get_int(env, argv[0], &n)) return enif_make_int(env,1);
    if (!enif_get_double(env, argv[1], &d)) return enif_make_int(env,2);
    a = (float *) enif_make_new_binary(env, n * sizeof(float), &a_bin);

    // Set matrix data 
    for(i=0;i<n;i++){
        a[i] = (float)d;
    }

    return(a_bin);
}



static ERL_NIF_TERM
new2(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    int r1,c1,i,j;
    ERL_NIF_TERM head, list, a_bin;
    float *a;
    double d;

    DISP("new2")
    if (!enif_get_int(env, argv[0], &r1)) return enif_make_int(env,1);
    if (!enif_get_int(env, argv[1], &c1)) return enif_make_int(env,2);
    a = (float *) enif_make_new_binary(env, r1 * c1 * sizeof(float), &a_bin);

    // Set matrix data 
    list = argv[2]; /* matrix1 */
    for(i=0;i<r1;i++){
        for(j=0;j<c1;j++){
            enif_get_list_cell(env, list, &head, &list);
            enif_get_double(env,head,&d);
            a[IDX2C(i,j,r1)] = (float)d;
        }
    }

    return(a_bin);
}


static ERL_NIF_TERM
new3(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    int c,h,w,i,j,k;
    ERL_NIF_TERM head, list, a_bin;
    float *a;
    double d;

    DISP("new3")
    if (!enif_get_int(env, argv[0], &c)) return enif_make_int(env,1);
    if (!enif_get_int(env, argv[1], &h)) return enif_make_int(env,2);
    if (!enif_get_int(env, argv[2], &w)) return enif_make_int(env,3);
    a = (float *) enif_make_new_binary(env, c * h * w *  sizeof(float), &a_bin);

    // Set matrix data 
    list = argv[3]; /* matrix1 */
    for(i=0;i<c;i++){
        for(j=0;j<h;j++){
            for(k=0;k<w;k++){
                enif_get_list_cell(env, list, &head, &list);
                enif_get_double(env,head,&d);
                a[IDX3C(i,j,k,h,w)] = (float)d;
            }
        }
    }

    return(a_bin);
}



static ERL_NIF_TERM
new4(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    int n,c,h,w,i,j,k,l;
    ERL_NIF_TERM head, list, a_bin;
    float *a;
    double d;

    DISP("new4")
    if (!enif_get_int(env, argv[0], &n)) return enif_make_int(env,1);
    if (!enif_get_int(env, argv[1], &c)) return enif_make_int(env,2);
    if (!enif_get_int(env, argv[2], &h)) return enif_make_int(env,3);
    if (!enif_get_int(env, argv[3], &w)) return enif_make_int(env,4);
    a = (float *) enif_make_new_binary(env, n * c * h * w *  sizeof(float), &a_bin);

    // Set matrix data 
    list = argv[4]; /* matrix1 */
    for(i=0;i<n;i++){
        for(j=0;j<c;j++){
            for(k=0;k<h;k++){
                for(l=0;l<w;l++){
                    enif_get_list_cell(env, list, &head, &list);
                    enif_get_double(env,head,&d);
                    a[IDX4C(i,j,k,l,c,h,w)] = (float)d;
                }
            }
        }
    }

    return(a_bin);
}



static ERL_NIF_TERM
rand1(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    int n,i;
    float x,y,val;
    float *result_data;
    ERL_NIF_TERM result;

    DISP("rand1")
    if (!enif_get_int(env, argv[0], &n)) return enif_make_int(env,1);
    result_data = (float *) enif_make_new_binary(env, n * sizeof(float), &result);

    srand((unsigned) time(NULL));
    for(i=0;i<n;i++){
        //box_muller
        x = (float)rand()/(float)RAND_MAX;
        y = (float)rand()/(float)RAND_MAX;
        val = sqrt(-2.0 * log(x)) * cos(2.0 * PI * y);
        result_data[i] = val;
    }
    return(result);
}



static ERL_NIF_TERM
mult1(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary  a_bin, b_bin;
    ERL_NIF_TERM  c_bin;
    int r1, c1, r2, c2, n, i, j;
    float *a,*b,*c;
    float* devPtrA;
    float* devPtrB;
    float* devPtrC;

    DISP("mult1")
    if (!enif_get_int(env, argv[0], &r1)) return enif_make_int(env,1);
    if (!enif_get_int(env, argv[1], &c1)) return enif_make_int(env,2);
    if (!enif_inspect_binary(env, argv[2], &a_bin )) return enif_make_int(env,3);
    if (!enif_get_int(env, argv[3], &r2)) return enif_make_int(env,4);
    if (!enif_get_int(env, argv[4], &c2)) return enif_make_int(env,5);
    if (!enif_inspect_binary(env, argv[5], &b_bin)) return enif_make_int(env,6);
    n = r1*c2;
    a = (float *) a_bin.data;
    b = (float *) b_bin.data;
    c = (float *) enif_make_new_binary(env, n * sizeof(float), &c_bin);

    for(j=0;j<c2;j++)
        for(i=0;i<r1;i++)
            c[IDX2C(i,j,r1)] = 0.0;


    // Initialize CUBLAS
    cublasInit();

    CUBLAS(cublasAlloc (r1*c1, sizeof(*a), (void**)&devPtrA));
    CUBLAS(cublasAlloc (r2*c2, sizeof(*b), (void**)&devPtrB));
    CUBLAS(cublasAlloc (r1*c2, sizeof(*c), (void**)&devPtrC));

    CUBLAS(cublasSetMatrix (r1, c1, sizeof(*a), a, r1, devPtrA, r1));
    CUBLAS(cublasSetMatrix (r2, c2, sizeof(*b), b, r2, devPtrB, r2));
    CUBLAS(cublasSetMatrix (r1, c2, sizeof(*c), c, r1, devPtrC, r1));


    //Sgemm
    cublasSgemm('N', 'N', r1, c2, c1, 1.0, devPtrA, r1, devPtrB, r2, 0.0, devPtrC, r1);


    CUBLAS(cublasGetMatrix (r1, c2, sizeof(*c), devPtrC, r1, c, r1));
    
    // Shutdown CUBLAS
    cublasFree(devPtrA);
    cublasFree(devPtrB);
    cublasFree(devPtrC);
    cublasShutdown();
    

    return(c_bin);

}


__global__ void add1_kernel(float *a, float *b, float *c, int n)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < n)
	{
		c[tid] = a[tid] + b[tid];
		tid += blockDim.x * gridDim.x;
	}
}



static ERL_NIF_TERM
add1(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary  a_bin, b_bin;
    ERL_NIF_TERM  c_bin;
    int n;
    float *a,*b,*c;
    float *dev_a, *dev_b, *dev_c;

    DISP("add1")
    if (!enif_get_int(env, argv[0], &n)) return enif_make_int(env,1);
    if (!enif_inspect_binary(env, argv[1], &a_bin )) return enif_make_int(env,2);
    if (!enif_inspect_binary(env, argv[2], &b_bin)) return enif_make_int(env,3);


    a = (float *) a_bin.data;
    b = (float *) b_bin.data;
    c = (float *) enif_make_new_binary(env, n * sizeof(float), &c_bin);

    // Allocate for GPU
	CHECK(cudaMalloc((void**)&dev_a, n * sizeof(float)));
	CHECK(cudaMalloc((void**)&dev_b, n * sizeof(float)));
	CHECK(cudaMalloc((void**)&dev_c, n * sizeof(float)));


    // copy from host a,b to GPU dev_a, dev_b
	CHECK(cudaMemcpy(dev_a, a, n * sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(dev_b, b, n * sizeof(float), cudaMemcpyHostToDevice));

	add1_kernel << <128, 128 >> >(dev_a, dev_b, dev_c, n);

	// copy to host c from GPU dev_c
	CHECK(cudaMemcpy(c, dev_c, n * sizeof(float), cudaMemcpyDeviceToHost));

    // free 
    cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

    return(c_bin);
}

__global__ void sub1_kernel(float *a, float *b, float *c, int n)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < n)
	{
		c[tid] = a[tid] - b[tid];
		tid += blockDim.x * gridDim.x;
	}
}
static ERL_NIF_TERM
sub1(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary  a_bin, b_bin;
    ERL_NIF_TERM  c_bin;
    int n;
    float *a,*b,*c;
    float *dev_a, *dev_b, *dev_c;

    DISP("sub1")
    if (!enif_get_int(env, argv[0], &n)) return enif_make_int(env,1);
    if (!enif_inspect_binary(env, argv[1], &a_bin )) return enif_make_int(env,2);
    if (!enif_inspect_binary(env, argv[2], &b_bin)) return enif_make_int(env,3);

    a = (float *) a_bin.data;
    b = (float *) b_bin.data;
    c = (float *) enif_make_new_binary(env, n * sizeof(float), &c_bin);

    	// Allocate for GPU
	CHECK(cudaMalloc((void**)&dev_a, n * sizeof(float)));
	CHECK(cudaMalloc((void**)&dev_b, n * sizeof(float)));
	CHECK(cudaMalloc((void**)&dev_c, n * sizeof(float)));


    // copy from host a,b to GPU dev_a, dev_b
	CHECK(cudaMemcpy(dev_a, a, n * sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(dev_b, b, n * sizeof(float), cudaMemcpyHostToDevice));

	sub1_kernel << <128, 128 >> >(dev_a, dev_b, dev_c, n);

	// copy to host c from GPU dev_c
	CHECK(cudaMemcpy(c, dev_c, n * sizeof(float), cudaMemcpyDeviceToHost));

    // free 
    cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

    return(c_bin);
}




__global__ void emult1_kernel(float *a, float *b, float *c, int n)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < n)
	{
		c[tid] = a[tid] * b[tid];
		tid += blockDim.x * gridDim.x;
	}
}


static ERL_NIF_TERM
emult1(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary  a_bin, b_bin;
    ERL_NIF_TERM  c_bin;
    int r1, c1, n;
    float *a,*b,*c;
    float *dev_a, *dev_b, *dev_c;

    DISP("emult1")
    if (!enif_get_int(env, argv[0], &r1)) return enif_make_int(env,1);
    if (!enif_get_int(env, argv[1], &c1)) return enif_make_int(env,2);
    if (!enif_inspect_binary(env, argv[2], &a_bin )) return enif_make_int(env,3);
    if (!enif_inspect_binary(env, argv[3], &b_bin)) return enif_make_int(env,4);
    n = r1*c1;
    a = (float *) a_bin.data;
    b = (float *) b_bin.data;
    c = (float *) enif_make_new_binary(env, n * sizeof(float), &c_bin);

    	// Allocate for GPU
	CHECK(cudaMalloc((void**)&dev_a, n * sizeof(float)));
	CHECK(cudaMalloc((void**)&dev_b, n * sizeof(float)));
	CHECK(cudaMalloc((void**)&dev_c, n * sizeof(float)));


    // copy from host a,b to GPU dev_a, dev_b
	CHECK(cudaMemcpy(dev_a, a, n * sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(dev_b, b, n * sizeof(float), cudaMemcpyHostToDevice));

	emult1_kernel << <128, 128 >> >(dev_a, dev_b, dev_c, n);

	// copy to host c from GPU dev_c
	CHECK(cudaMemcpy(c, dev_c, n * sizeof(float), cudaMemcpyDeviceToHost));

    // free 
    cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

    return(c_bin);
}


static ERL_NIF_TERM
transpose1(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary  a_bin;
    ERL_NIF_TERM  b_bin;
    int r1, c1, n, i, j;
    float *a,*b;

    DISP("transpose1")
    if (!enif_get_int(env, argv[0], &r1)) return enif_make_int(env,1);
    if (!enif_get_int(env, argv[1], &c1)) return enif_make_int(env,2);
    if (!enif_inspect_binary(env, argv[2], &a_bin )) return enif_make_int(env,3);
    n = r1*c1;
    a = (float *) a_bin.data;
    b = (float *) enif_make_new_binary(env, n * sizeof(float), &b_bin);

    for(i=0;i<r1;i++){
        for(j=0;j<c1;j++){
            b[IDX2C(j,i,c1)] = a[IDX2C(i,j,r1)];
        }
    }

    return(b_bin);
}


static ERL_NIF_TERM
ident1(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    int n,i,j;
    ERL_NIF_TERM a_bin;
    float *a;

    DISP("ident1")
    if (!enif_get_int(env, argv[0], &n)) return enif_make_int(env,1);
    a = (float *) enif_make_new_binary(env, n * n * sizeof(float), &a_bin);

    // Set matrix data 
    for(i=0;i<n;i++){
        for(j=0;j<n;j++){
            if(i==j)
                a[IDX2C(i,j,n)] = 1.0;
            else
                a[IDX2C(i,j,n)] = 0.0;
        }
    }

    return(a_bin);
}




__global__ void sigmoid_kernel(float *a, float *b, int n)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < n)
	{   
        b[tid] = SIGMOID(a[tid]);
		tid += blockDim.x * gridDim.x;
	}
}

static ERL_NIF_TERM
activate_sigmoid(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary  a_bin;
    ERL_NIF_TERM  b_bin;
    int r1, c1, n;
    float *a,*b;
    float *dev_a, *dev_b;

    DISP("activate_sigmoid")
    if (!enif_get_int(env, argv[0], &r1)) return enif_make_int(env,1);
    if (!enif_get_int(env, argv[1], &c1)) return enif_make_int(env,2);
    if (!enif_inspect_binary(env, argv[2], &a_bin )) return enif_make_int(env,3);
    n = r1*c1;
    a = (float *) a_bin.data;
    b = (float *) enif_make_new_binary(env, n * sizeof(float), &b_bin);

    	// Allocate for GPU
	CHECK(cudaMalloc((void**)&dev_a, n * sizeof(float)));
	CHECK(cudaMalloc((void**)&dev_b, n * sizeof(float)));


    // copy from host a,b to GPU dev_a, dev_b
	CHECK(cudaMemcpy(dev_a, a, n * sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(dev_b, b, n * sizeof(float), cudaMemcpyHostToDevice));

	sigmoid_kernel << <128, 128 >> >(dev_a, dev_b, n);

	// copy to host c from GPU dev_c
    CHECK(cudaMemcpy(b, dev_b, n * sizeof(float), cudaMemcpyDeviceToHost));
    
    // free 
    cudaFree(dev_a);
    cudaFree(dev_b);

    return(b_bin);
}



__global__ void tanh_kernel(float *a, float *b, int n)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < n)
	{
		b[tid] = tanh(a[tid]);
		tid += blockDim.x * gridDim.x;
	}
}


static ERL_NIF_TERM
activate_tanh(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary  a_bin;
    ERL_NIF_TERM  b_bin;
    int r1, c1, n;
    float *a,*b;
    float *dev_a, *dev_b;

    DISP("activate_tanh")
    if (!enif_get_int(env, argv[0], &r1)) return enif_make_int(env,1);
    if (!enif_get_int(env, argv[1], &c1)) return enif_make_int(env,2);
    if (!enif_inspect_binary(env, argv[2], &a_bin )) return enif_make_int(env,3);
    n = r1*c1;
    a = (float *) a_bin.data;
    b = (float *) enif_make_new_binary(env, n * sizeof(float), &b_bin);

    	// Allocate for GPU
	CHECK(cudaMalloc((void**)&dev_a, n * sizeof(float)));
	CHECK(cudaMalloc((void**)&dev_b, n * sizeof(float)));


    // copy from host a,b to GPU dev_a, dev_b
	CHECK(cudaMemcpy(dev_a, a, n * sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(dev_b, b, n * sizeof(float), cudaMemcpyHostToDevice));

	tanh_kernel << <128, 128 >> >(dev_a, dev_b, n);

	// copy to host c from GPU dev_c
    CHECK(cudaMemcpy(b, dev_b, n * sizeof(float), cudaMemcpyDeviceToHost));
    
    // free 
    cudaFree(dev_a);
    cudaFree(dev_b);

    return(b_bin);
}



__global__ void relu_kernel(float *a, float *b, int n)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < n)
	{   
        if(a[tid] >= 0)
		    b[tid] = a[tid];
        else 
            b[tid] = 0.0;
		tid += blockDim.x * gridDim.x;
	}
}


static ERL_NIF_TERM
activate_relu(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary  a_bin;
    ERL_NIF_TERM  b_bin;
    int r1, c1, n;
    float *a,*b;
    float *dev_a, *dev_b;

    DISP("activate_relu")
    if (!enif_get_int(env, argv[0], &r1)) return enif_make_int(env,1);
    if (!enif_get_int(env, argv[1], &c1)) return enif_make_int(env,2);
    if (!enif_inspect_binary(env, argv[2], &a_bin )) return enif_make_int(env,3);
    n = r1*c1;
    a = (float *) a_bin.data;
    b = (float *) enif_make_new_binary(env, n * sizeof(float), &b_bin);

    	// Allocate for GPU
	CHECK(cudaMalloc((void**)&dev_a, n * sizeof(float)));
	CHECK(cudaMalloc((void**)&dev_b, n * sizeof(float)));


    // copy from host a,b to GPU dev_a, dev_b
	CHECK(cudaMemcpy(dev_a, a, n * sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(dev_b, b, n * sizeof(float), cudaMemcpyHostToDevice));

	relu_kernel << <128, 128 >> >(dev_a, dev_b, n);

	// copy to host c from GPU dev_c
    CHECK(cudaMemcpy(b, dev_b, n * sizeof(float), cudaMemcpyDeviceToHost));
    
    // free 
    cudaFree(dev_a);
    cudaFree(dev_b);

    return(b_bin);
}

static ERL_NIF_TERM
activate_softmax(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary  a_bin;
    ERL_NIF_TERM  b_bin;
    int r1, c1, n, i, j, k;
    float *a,*b;
    float max,sum;

    DISP("activate_softmax")
    if (!enif_get_int(env, argv[0], &r1)) return enif_make_int(env,1);
    if (!enif_get_int(env, argv[1], &c1)) return enif_make_int(env,2);
    if (!enif_inspect_binary(env, argv[2], &a_bin )) return enif_make_int(env,3);
    n = r1*c1;
    a = (float *) a_bin.data;
    b = (float *) enif_make_new_binary(env, n * sizeof(float), &b_bin);

    //calculate softmax
    for(i=0;i<r1;i++){
        for(j=0;j<c1;j++){
            max = 0.0;
            for(k=0;k<c1;k++){
                if(a[IDX2C(i,k,r1)] > max)
                    max = a[IDX2C(i,k,r1)];
            }
            sum = 0.0;
            for(k=0;k<c1;k++){
                sum = sum + exp(a[IDX2C(i,k,r1)] - max);
            }
            b[IDX2C(i,j,r1)] = exp(a[IDX2C(i,j,r1)] - max) / sum;
        }
    }


    return(b_bin);
}



__global__ void differ_sigmoid_kernel(float *a, float *b, float *c, int n)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < n)
	{   
        
		c[tid] = a[tid] * ((1 - SIGMOID(b[tid])) * SIGMOID(b[tid]));
		tid += blockDim.x * gridDim.x;
	}
}

static ERL_NIF_TERM
differ_sigmoid(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary  a_bin, b_bin;
    ERL_NIF_TERM  c_bin;
    int r1, c1, n;
    float *a,*b,*c;
    float *dev_a, *dev_b, *dev_c;

    DISP("differ_sigmoid")
    if (!enif_get_int(env, argv[0], &r1)) return enif_make_int(env,1);
    if (!enif_get_int(env, argv[1], &c1)) return enif_make_int(env,2);
    if (!enif_inspect_binary(env, argv[2], &a_bin )) return enif_make_int(env,3);
    if (!enif_inspect_binary(env, argv[3], &b_bin)) return enif_make_int(env,4);
    n = r1*c1;
    a = (float *) a_bin.data;
    b = (float *) b_bin.data;
    c = (float *) enif_make_new_binary(env, n * sizeof(float), &c_bin);

    	// Allocate for GPU
	CHECK(cudaMalloc((void**)&dev_a, n * sizeof(float)));
	CHECK(cudaMalloc((void**)&dev_b, n * sizeof(float)));
	CHECK(cudaMalloc((void**)&dev_c, n * sizeof(float)));


    // copy from host a,b to GPU dev_a, dev_b
	CHECK(cudaMemcpy(dev_a, a, n * sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(dev_b, b, n * sizeof(float), cudaMemcpyHostToDevice));

	differ_sigmoid_kernel << <128, 128 >> >(dev_a, dev_b, dev_c, n);

	// copy to host c from GPU dev_c
	CHECK(cudaMemcpy(c, dev_c, n * sizeof(float), cudaMemcpyDeviceToHost));

    // free 
    cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

    return(c_bin);
}


__global__ void differ_tanh_kernel(float *a, float *b, float *c, int n)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < n)
	{   
        c[tid] = a[tid] * (1/(cosh(b[tid]) * cosh(b[tid])));
		tid += blockDim.x * gridDim.x;
	}
}

static ERL_NIF_TERM
differ_tanh(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary  a_bin, b_bin;
    ERL_NIF_TERM  c_bin;
    int r1, c1, n;
    float *a,*b,*c;
    float *dev_a, *dev_b, *dev_c;

    DISP("differ_tanh")
    if (!enif_get_int(env, argv[0], &r1)) return enif_make_int(env,1);
    if (!enif_get_int(env, argv[1], &c1)) return enif_make_int(env,2);
    if (!enif_inspect_binary(env, argv[2], &a_bin )) return enif_make_int(env,3);
    if (!enif_inspect_binary(env, argv[3], &b_bin)) return enif_make_int(env,4);
    n = r1*c1;
    a = (float *) a_bin.data;
    b = (float *) b_bin.data;
    c = (float *) enif_make_new_binary(env, n * sizeof(float), &c_bin);

    	// Allocate for GPU
	CHECK(cudaMalloc((void**)&dev_a, n * sizeof(float)));
	CHECK(cudaMalloc((void**)&dev_b, n * sizeof(float)));
	CHECK(cudaMalloc((void**)&dev_c, n * sizeof(float)));


    // copy from host a,b to GPU dev_a, dev_b
	CHECK(cudaMemcpy(dev_a, a, n * sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(dev_b, b, n * sizeof(float), cudaMemcpyHostToDevice));

	differ_tanh_kernel << <128, 128 >> >(dev_a, dev_b, dev_c, n);

	// copy to host c from GPU dev_c
	CHECK(cudaMemcpy(c, dev_c, n * sizeof(float), cudaMemcpyDeviceToHost));

    // free 
    cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

    return(c_bin);
}



__global__ void differ_relu_kernel(float *a, float *b, float *c, int n)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < n)
	{   
        if(b[tid] >= 0)
		    c[tid] = a[tid];
        else 
            c[tid] = 0.0;
		tid += blockDim.x * gridDim.x;
	}
}

static ERL_NIF_TERM
differ_relu(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary  a_bin, b_bin;
    ERL_NIF_TERM  c_bin;
    int r1, c1, n;
    float *a,*b,*c;
    float *dev_a, *dev_b, *dev_c;

    DISP("differ_relu")
    if (!enif_get_int(env, argv[0], &r1)) return enif_make_int(env,1);
    if (!enif_get_int(env, argv[1], &c1)) return enif_make_int(env,2);
    if (!enif_inspect_binary(env, argv[2], &a_bin )) return enif_make_int(env,3);
    if (!enif_inspect_binary(env, argv[3], &b_bin)) return enif_make_int(env,4);
    n = r1*c1;
    a = (float *) a_bin.data;
    b = (float *) b_bin.data;
    c = (float *) enif_make_new_binary(env, n * sizeof(float), &c_bin);

    	// Allocate for GPU
	CHECK(cudaMalloc((void**)&dev_a, n * sizeof(float)));
	CHECK(cudaMalloc((void**)&dev_b, n * sizeof(float)));
	CHECK(cudaMalloc((void**)&dev_c, n * sizeof(float)));


    // copy from host a,b to GPU dev_a, dev_b
	CHECK(cudaMemcpy(dev_a, a, n * sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(dev_b, b, n * sizeof(float), cudaMemcpyHostToDevice));

	differ_relu_kernel << <128, 128 >> >(dev_a, dev_b, dev_c, n);

	// copy to host c from GPU dev_c
	CHECK(cudaMemcpy(c, dev_c, n * sizeof(float), cudaMemcpyDeviceToHost));

    // free 
    cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

    return(c_bin);
}


__global__ void smult_kernel(float d, float *a, float *b, int n)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < n)
	{
		b[tid] = d * a[tid];
		tid += blockDim.x * gridDim.x;
	}
}


static ERL_NIF_TERM
smult1(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary  a_bin;
    ERL_NIF_TERM  b_bin;
    int n;
    float *a,*b;
    float *dev_a, *dev_b;
    double s;

    DISP("smult1")
    if (!enif_get_double(env, argv[0], &s)) return enif_make_int(env,1);
    if (!enif_get_int(env, argv[1], &n)) return enif_make_int(env,2);
    if (!enif_inspect_binary(env, argv[2], &a_bin )) return enif_make_int(env,3);
    a = (float *) a_bin.data;
    b = (float *) enif_make_new_binary(env, n * sizeof(float), &b_bin);

    	// Allocate for GPU
	CHECK(cudaMalloc((void**)&dev_a, n * sizeof(float)));
	CHECK(cudaMalloc((void**)&dev_b, n * sizeof(float)));


    // copy from host a,b to GPU dev_a, dev_b
	CHECK(cudaMemcpy(dev_a, a, n * sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(dev_b, b, n * sizeof(float), cudaMemcpyHostToDevice));

	smult_kernel << <128, 128 >> >((float)s,dev_a, dev_b, n);

	// copy to host c from GPU dev_c
	CHECK(cudaMemcpy(b, dev_b, n * sizeof(float), cudaMemcpyDeviceToHost));

    // free 
    cudaFree(dev_a);
	cudaFree(dev_b);

    return(b_bin);
}


static ERL_NIF_TERM
trace1(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary  a_bin;
    ERL_NIF_TERM  result;
    int r1, c1, i, j;
    float *a;
    float trace;

    DISP("trace1")
    if (!enif_get_int(env, argv[0], &r1)) return enif_make_int(env,1);
    if (!enif_get_int(env, argv[1], &c1)) return enif_make_int(env,2);
    if (!enif_inspect_binary(env, argv[2], &a_bin )) return enif_make_int(env,3);
    a = (float *) a_bin.data;
    
    trace = 0.0;
    for(i=0;i<r1;i++){
        for(j=0;j<c1;j++){
            if(i==j)
                trace = trace + a[IDX2C(i,j,r1)];
        }
    }

    result = enif_make_double(env,trace);

    return(result);
}


static ERL_NIF_TERM
mean_square(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary  a_bin,b_bin;
    ERL_NIF_TERM  result;
    int r1, c1, i, j;
    float *a, *b;
    float d,s;

    DISP("mean_square")
    if (!enif_get_int(env, argv[0], &r1)) return enif_make_int(env,1);
    if (!enif_get_int(env, argv[1], &c1)) return enif_make_int(env,2);
    if (!enif_inspect_binary(env, argv[2], &a_bin )) return enif_make_int(env,3);
    if (!enif_inspect_binary(env, argv[3], &b_bin )) return enif_make_int(env,4);

    a = (float *) a_bin.data;
    b = (float *) b_bin.data;

    s = 0.0;
    for(i=0;i<r1;i++){
        for (j=0;j<c1;j++){
            d = a[IDX2C(i,j,r1)] -  b[IDX2C(i,j,r1)];
            s = s + d*d;            
        }
    } 
    s = s / (2.0*(float(r1)));
    result = enif_make_double(env,s);
    return(result);
}

static ERL_NIF_TERM
cross_entropy(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary  a_bin,b_bin;
    ERL_NIF_TERM  result;
    int r1, c1, i, j;
    float *a, *b;
    float d,s,delta;

    DISP("cross_entropy")
    if (!enif_get_int(env, argv[0], &r1)) return enif_make_int(env,1);
    if (!enif_get_int(env, argv[1], &c1)) return enif_make_int(env,2);
    if (!enif_inspect_binary(env, argv[2], &a_bin )) return enif_make_int(env,3);
    if (!enif_inspect_binary(env, argv[3], &b_bin )) return enif_make_int(env,4);

    a = (float *) a_bin.data;
    b = (float *) b_bin.data;

    
    delta = 1e-7;
    s = 0.0;
    for(i=0;i<r1;i++){
        for (j=0;j<c1;j++){
            d = fabsf(a[IDX2C(i,j,r1)])+delta;
            s = s + b[IDX2C(i,j,r1)] * log(d); 
        }
    }
    s = s / (float)r1;
    result = enif_make_double(env,s);
    return(result);
}





static ERL_NIF_TERM
elt1(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary  a_bin;
    ERL_NIF_TERM  result;
    int r1, c1, i, j;
    float *a;

    DISP("elt1")
    if (!enif_get_int(env, argv[0], &r1)) return enif_make_int(env,1);
    if (!enif_get_int(env, argv[1], &c1)) return enif_make_int(env,2);
    if (!enif_get_int(env, argv[2], &i)) enif_make_int(env,3);
    if (!enif_get_int(env, argv[3], &j)) return enif_make_int(env,4);
    if (!enif_inspect_binary(env, argv[4], &a_bin )) return enif_make_int(env,5);
    a = (float *) a_bin.data;
    
    result = enif_make_double(env,(double)a[IDX2C(i,j,r1)]);

    return(result);
}

static ERL_NIF_TERM
set1(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary  a_bin;
    ERL_NIF_TERM  b_bin;
    int r1, c1, n, i, j, x, y;
    float *a,*b;
    double val;

    DISP("set1")
    if (!enif_get_int(env, argv[0], &r1)) return enif_make_int(env,1);
    if (!enif_get_int(env, argv[1], &c1)) return enif_make_int(env,2);
    if (!enif_inspect_binary(env, argv[2], &a_bin )) return enif_make_int(env,3);
    if (!enif_get_int(env, argv[3], &x)) return enif_make_int(env,4);
    if (!enif_get_int(env, argv[4], &y)) return enif_make_int(env,5);
    if (!enif_get_double(env, argv[5], &val)) return enif_make_int(env,6);


    n = r1*c1;
    a = (float *) a_bin.data;
    b = (float *) enif_make_new_binary(env, n * sizeof(float), &b_bin);

    for(i=0;i<r1;i++){
        for(j=0;j<c1;j++){
            if(i==x && j==y)
                b[IDX2C(i,j,r1)] = (float)val;
            else 
                b[IDX2C(i,j,r1)] = a[IDX2C(i,j,r1)];
        }
    }


    return(b_bin);
}

static ERL_NIF_TERM
average1(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary  a_bin;
    ERL_NIF_TERM  b_bin;
    int r1, c1, i, j;
    float *a,*b;
    float sum;

    DISP("average1")
    if (!enif_get_int(env, argv[0], &r1)) return enif_make_int(env,1);
    if (!enif_get_int(env, argv[1], &c1)) return enif_make_int(env,2);
    if (!enif_inspect_binary(env, argv[2], &a_bin )) return enif_make_int(env,3);

    a = (float *) a_bin.data;
    b = (float *) enif_make_new_binary(env, c1 * sizeof(float), &b_bin);

    for(j=0;j<c1;j++){
        sum = 0.0;
        for(i=0;i<r1;i++){
            sum = sum + a[IDX2C(i,j,r1)];
        }
        b[j] = sum / (float)r1;
    }


    return(b_bin);
}

/*
1st arg row-size of matrix
2nd arg col-size of matrix
3rd arg matrix data binary
*/


static ERL_NIF_TERM
sum1(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary  a_bin;
    ERL_NIF_TERM  result;
    int r1, c1, i, j;
    float *a;
    float sum;

    DISP("sum1")
    if (!enif_get_int(env, argv[0], &r1)) return enif_make_int(env,1);
    if (!enif_get_int(env, argv[1], &c1)) return enif_make_int(env,2);
    if (!enif_inspect_binary(env, argv[2], &a_bin )) return enif_make_int(env,3);
    a = (float *) a_bin.data;
    
    sum = 0.0;
    for(i=0;i<r1;i++){
        for(j=0;j<c1;j++){
            sum = sum + a[IDX2C(i,j,r1)];
        }
    }

    result = enif_make_double(env,sum);

    return(result);
}

/*
transfer 2 DIm matrix to list 
*/
static ERL_NIF_TERM
to_list1(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary  a_bin;
    ERL_NIF_TERM  head,list;
    int r1, c1, i, j;
    float *a;

    DISP("to_list1")
    if (!enif_get_int(env, argv[0], &r1)) return enif_make_int(env,1);
    if (!enif_get_int(env, argv[1], &c1)) return enif_make_int(env,2);
    if (!enif_inspect_binary(env, argv[2], &a_bin )) return enif_make_int(env,3);
    a = (float *) a_bin.data;

    
    list = enif_make_list(env, 0);
    for(i=r1-1;i>=0;i--){
        for(j=c1-1;j>=0;j--){
            head = enif_make_double(env,(double)a[IDX2C(i,j,r1)]);
            list = enif_make_list_cell(env,head,list);
        }
    }

    return(list);
}
/*
transfer 3 DIm matrix to list
*/

static ERL_NIF_TERM
to_list2(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary  a_bin;
    ERL_NIF_TERM  head,list;
    int c, h, w, i, j, k;
    float *a;

    DISP("to_list2")
    if (!enif_get_int(env, argv[0], &c)) return enif_make_int(env,1);
    if (!enif_get_int(env, argv[1], &h)) return enif_make_int(env,2);
    if (!enif_get_int(env, argv[2], &w)) return enif_make_int(env,3);
    if (!enif_inspect_binary(env, argv[3], &a_bin )) return enif_make_int(env,4);
   
    a = (float *) a_bin.data;
    
    list = enif_make_list(env, 0);
    for(i=c-1;i>=0;i--){
        for(j=h-1;j>=0;j--){
            for(k=w-1;k>=0;k--){
                head = enif_make_double(env,(double)a[IDX3C(i,j,k,h,w)]);
                list = enif_make_list_cell(env,head,list);
            }
        }
    }

    return(list);
}
/*
transfer 4 DIm matrix to list
*/
static ERL_NIF_TERM
to_list3(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary  a_bin;
    ERL_NIF_TERM  head,list;
    int n, c, h, w, i, j, k, l;
    float *a;

    DISP("to_list3")
    if (!enif_get_int(env, argv[0], &n)) return enif_make_int(env,1);
    if (!enif_get_int(env, argv[1], &c)) return enif_make_int(env,2);
    if (!enif_get_int(env, argv[2], &h)) return enif_make_int(env,3);
    if (!enif_get_int(env, argv[3], &w)) return enif_make_int(env,4);
    if (!enif_inspect_binary(env, argv[4], &a_bin )) return enif_make_badarg(env);
    a = (float *) a_bin.data;

    
    list = enif_make_list(env, 0);
    for(i=n-1;i>=0;i--){
        for(j=c-1;j>=0;j--){
            for(k=h-1;k>=0;k--){
                for(l=w-1;l>=0;l--){
                    head = enif_make_double(env,(double)a[IDX4C(i,j,k,l,c,h,w)]);
                    list = enif_make_list_cell(env,head,list);
                }
            }
        }
    }

    return(list);
}

__global__ void sgd1_kernel(float *a, float *b, float *c, float lr, int n)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < n)
	{
        if(a[tid] != 0.0)
            c[tid] = a[tid] - b[tid]*lr;
        else 
            c[tid] = 0.0;
		tid += blockDim.x * gridDim.x;
	}
}
/*
w - g*lr |> dropout()
for sgd
w is weight matrix.
g is gradient matrix.
when element of w is zero result is zero. This means dropout.
return updated weight matrix.

1st arg is size of vectorized matrix
2nd arg is weight matrix or tensor
3rd arg is gradient matrix or tensor
4th arg is learning rate
5th arg is dropout rate
*/
static ERL_NIF_TERM
sgd1(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary  a_bin, b_bin;
    ERL_NIF_TERM  c_bin;
    int n,r;
    float *a,*b,*c,*dev_a, *dev_b, *dev_c;
    float lr,dr,randfloat;
    double learning_rate,dropout_rate;

    DISP("sgd1")
    if (!enif_get_int(env, argv[0], &n)) return enif_make_int(env,1);
    if (!enif_inspect_binary(env, argv[1], &a_bin )) return enif_make_int(env,2);
    if (!enif_inspect_binary(env, argv[2], &b_bin)) return enif_make_int(env,3);
    if (!enif_get_double(env, argv[3], &learning_rate)) return enif_make_int(env,4);
    if (!enif_get_double(env, argv[4], &dropout_rate)) return enif_make_int(env,5);


    a = (float *) a_bin.data;
    b = (float *) b_bin.data;
    c = (float *) enif_make_new_binary(env, n * sizeof(float), &c_bin);
    lr = (float) learning_rate;
    dr = (float) dropout_rate;

    	// Allocate for GPU
	CHECK(cudaMalloc((void**)&dev_a, n * sizeof(float)));
	CHECK(cudaMalloc((void**)&dev_b, n * sizeof(float)));
	CHECK(cudaMalloc((void**)&dev_c, n * sizeof(float)));


    // copy from host a,b to GPU dev_a, dev_b
	CHECK(cudaMemcpy(dev_a, a, n * sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(dev_b, b, n * sizeof(float), cudaMemcpyHostToDevice));

	sgd1_kernel << <128, 128 >> >(dev_a, dev_b, dev_c, lr, n);

	// copy to host c from GPU dev_c
	CHECK(cudaMemcpy(c, dev_c, n * sizeof(float), cudaMemcpyDeviceToHost));


    // dropout
    randfloat = (double)(rand() % 100) / 100.0;
    if(dr != 0.0 && dr < randfloat){
        r = rand() % n;
        c[r] = 0.0;
    }

    // free 
    cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

    return(c_bin);
}


/*
  def momentum(v, g, lr) do
    Matrex.apply(v, g, fn v, g -> 0.5 * v - lr * g end)
  end
*/
__global__ void momentum_kernel(float *a, float *b, float *c, float *d, float *e, float lr, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < n)
    {   
        d[tid] = (0.5 * b[tid]) - (lr * c[tid]);
        if(a[tid] != 0.0)
            e[tid] = a[tid] + d[tid];
        else 
            e[tid] = 0.0;
        tid += blockDim.x * gridDim.x;
    }
}

/*
1st arg row-size of vectorized each-matrix
2nd arg wight-matrix
3rd arg v-matrix
4th arg g-matrix
5th arg learning rate
6th arg deropout rate
return tuple
*/
static ERL_NIF_TERM
momentum1(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary  a_bin,b_bin,c_bin;
    ERL_NIF_TERM  d_bin,e_bin,tuple;
    int n,r;
    float *a,*b,*c,*d,*e;
    float *dev_a, *dev_b, *dev_c ,*dev_d, *dev_e;
    float lr,dr,randfloat;
    double learning_rate,dropout_rate;
  
    DISP("momentum1")
    if (!enif_get_int(env, argv[0], &n)) return enif_make_int(env,1);
    if (!enif_inspect_binary(env, argv[1], &a_bin)) return enif_make_int(env,2);
    if (!enif_inspect_binary(env, argv[2], &b_bin )) return enif_make_int(env,3);
    if (!enif_inspect_binary(env, argv[3], &c_bin )) return enif_make_int(env,4);
    if (!enif_get_double(env, argv[4], &learning_rate)) return enif_make_int(env,5);
    if (!enif_get_double(env, argv[5], &dropout_rate)) return enif_make_int(env,6);

    a = (float *) a_bin.data;
    b = (float *) b_bin.data;
    c = (float *) c_bin.data;
    d = (float *) enif_make_new_binary(env, n * sizeof(float), &d_bin);
    e = (float *) enif_make_new_binary(env, n * sizeof(float), &e_bin);
    lr = (float) learning_rate;
    dr = (float) dropout_rate;
    
  
    // Allocate for GPU
    CHECK(cudaMalloc((void**)&dev_a, n * sizeof(float)));
    CHECK(cudaMalloc((void**)&dev_b, n * sizeof(float)));
    CHECK(cudaMalloc((void**)&dev_c, n * sizeof(float)));
    CHECK(cudaMalloc((void**)&dev_d, n * sizeof(float)));
    CHECK(cudaMalloc((void**)&dev_e, n * sizeof(float)));
  
    // copy from host a,b to GPU dev_a, dev_b
    CHECK(cudaMemcpy(dev_a, a, n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_b, b, n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_c, c, n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_d, d, n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_e, e, n * sizeof(float), cudaMemcpyHostToDevice));
  
    momentum_kernel << <128, 128 >> >(dev_a, dev_b, dev_c, dev_d, dev_e, lr, n);
  
    // copy to host d from GPU dev_d
    CHECK(cudaMemcpy(d, dev_d, n * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(e, dev_e, n * sizeof(float), cudaMemcpyDeviceToHost));

    // dropout
    randfloat = (double)(rand() % 100) / 100.0;
    if(dr != 0.0 && dr < randfloat){
        r = rand() % n;
        e[r] = 0.0;
    }

    // free 
    cudaFree(dev_a);
	cudaFree(dev_b);
    cudaFree(dev_c);
    cudaFree(dev_d);
	cudaFree(dev_e);
    
    tuple = enif_make_tuple2(env,d_bin,e_bin);
    return(tuple);
}

  /*
    h1 = h + grad*grad
    w1 = w - lr * 1/sqrt(h1) * grad 
  */
  
__global__ void adagrad_kernel(float *a, float *b, float *c, float *d, float *e, float lr, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < n)
    {   
        d[tid] = b[tid] + c[tid]*c[tid];

        if(d[tid] != 0)
            e[tid] = a[tid] - (lr * (1 / sqrt(d[tid])) * c[tid]);
        else 
            e[tid] = a[tid] - (lr * c[tid]);
        tid += blockDim.x * gridDim.x;
    }
}
 
/*
1st arg row-size of vectorized each-matrix
2nd arg wight-matrix (a_bin)
3rd arg h-matrix     (b_bin)
4th arg grad-matrix  (c_bin)
5th arg learning rate
6th arg deropout rate
return tuple {new-h,new-w}
*/
static ERL_NIF_TERM
adagrad1(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary  a_bin,b_bin,c_bin;
    ERL_NIF_TERM  d_bin,e_bin,tuple;
    int n,r;
    float *a,*b,*c,*d,*e;
    float *dev_a, *dev_b, *dev_c, *dev_d, *dev_e;
    float lr,dr,randfloat;
    double learning_rate,dropout_rate;
    
    DISP("adagrad1")
    if (!enif_get_int(env, argv[0], &n)) return enif_make_int(env,1);
    if (!enif_inspect_binary(env, argv[1], &a_bin)) return enif_make_int(env,2);
    if (!enif_inspect_binary(env, argv[2], &b_bin)) return enif_make_int(env,3);
    if (!enif_inspect_binary(env, argv[3], &c_bin)) return enif_make_int(env,4);
    if (!enif_get_double(env, argv[4], &learning_rate)) return enif_make_int(env,5);
    if (!enif_get_double(env, argv[5], &dropout_rate)) return enif_make_int(env,6);

    a = (float *) a_bin.data;
    b = (float *) b_bin.data;
    c = (float *) c_bin.data;
    d = (float *) enif_make_new_binary(env, n * sizeof(float), &d_bin);
    e = (float *) enif_make_new_binary(env, n * sizeof(float), &e_bin);
    lr = (float) learning_rate;
    dr = (float) dropout_rate;
  
    // Allocate for GPU
    CHECK(cudaMalloc((void**)&dev_a, n * sizeof(float)));
    CHECK(cudaMalloc((void**)&dev_b, n * sizeof(float)));
    CHECK(cudaMalloc((void**)&dev_c, n * sizeof(float)));
    CHECK(cudaMalloc((void**)&dev_d, n * sizeof(float)));
    CHECK(cudaMalloc((void**)&dev_e, n * sizeof(float)));
  
    // copy from host a,b to GPU dev_a, dev_b
    CHECK(cudaMemcpy(dev_a, a, n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_b, b, n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_c, c, n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_d, d, n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_e, e, n * sizeof(float), cudaMemcpyHostToDevice));
  
    adagrad_kernel << <128, 128 >> >(dev_a, dev_b, dev_c, dev_d, dev_e, lr, n);
  
    // copy to host c from GPU dev_c
    CHECK(cudaMemcpy(c, dev_d, n * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(c, dev_e, n * sizeof(float), cudaMemcpyDeviceToHost));

    // dropout
    randfloat = (double)(rand() % 100) / 100.0;
    if(dr != 0.0 && dr < randfloat){
        r = rand() % n;
        e[r] = 0.0;
    }

    // free 
    cudaFree(dev_a);
	cudaFree(dev_b);
    cudaFree(dev_c);
    cudaFree(dev_d);
	cudaFree(dev_e);
    
    tuple = enif_make_tuple2(env,d_bin,e_bin);
    return(tuple);
}

/*
1st arg row-size of matrix
2nd arg col-size of matris
3rd arg predicted matrix
4th arg list of label. each element is integer
*/

static ERL_NIF_TERM
accuracy1(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary  a_bin;
    ERL_NIF_TERM  head,list,result;
    int r1, c1, i, j, n, index,sum;
    float *a;
    double max,rate;
  
    DISP("accuracy1")
    if (!enif_get_int(env, argv[0], &r1)) return enif_make_int(env,1);
    if (!enif_get_int(env, argv[1], &c1)) return enif_make_int(env,2);
    if (!enif_inspect_binary(env, argv[2], &a_bin )) return enif_make_int(env,3);

    a = (float *) a_bin.data;
    

    // calculate accuracy
    sum = 0;
    list = argv[3]; 
    for(i=0;i<r1;i++){
        max = 0.0;
        enif_get_list_cell(env, list, &head, &list);
        enif_get_int(env,head,&n);
        for(j=0;j<c1;j++){
            if(a[IDX2C(i,j,r1)] > max){
                max = a[IDX2C(i,j,r1)];
                index = j;
            }
        }
        if(index == n)
            sum++;
    }
    rate = (double)sum / (double)r1;
    result = enif_make_double(env,rate);
    return(result);
}


static ERL_NIF_TERM
random_select1(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary  a_bin,b_bin;
    ERL_NIF_TERM  c_bin,d_bin,tuple;
    int r1, c1, r2, c2, i, j, n, r;
    float *a, *b, *c, *d;
  
    DISP("random_select1")
    if (!enif_get_int(env, argv[0], &r1)) return enif_make_int(env,1);
    if (!enif_get_int(env, argv[1], &c1)) return enif_make_int(env,2);
    if (!enif_inspect_binary(env, argv[2], &a_bin )) return enif_make_int(env,3);
    if (!enif_get_int(env, argv[3], &r2)) return enif_make_int(env,4);
    if (!enif_get_int(env, argv[4], &c2)) return enif_make_int(env,5);
    if (!enif_inspect_binary(env, argv[5], &b_bin )) return enif_make_int(env,6);
    if (!enif_get_int(env, argv[6], &n)) return enif_make_int(env,7);

    a = (float *) a_bin.data;
    b = (float *) b_bin.data;
    c = (float *) enif_make_new_binary(env, n*c1 * sizeof(float), &c_bin);
    d = (float *) enif_make_new_binary(env, n*c2 * sizeof(float), &d_bin);


    // random-select
    for(i=0;i<n;i++){
        r = rand() % r1;
        for(j=0;j<c1;j++){
            c[IDX2C(i,j,n)] = a[IDX2C(r,j,r1)];
        }
        for(j=0;j<c2;j++){
            d[IDX2C(i,j,n)] = b[IDX2C(r,j,r2)];
        }    
    }

    tuple = enif_make_tuple2(env,c_bin,d_bin);
    return(tuple);
}

static ERL_NIF_TERM
random_select2(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary  a_bin,b_bin;
    ERL_NIF_TERM  c_bin,d_bin,tuple;
    int n1,c1,h1,w1,r2,c2, i, j, k, l, n, r;
    float *a, *b, *c, *d;
  
    DISP("random_select2")
    if (!enif_get_int(env, argv[0], &n1)) return enif_make_int(env,1);
    if (!enif_get_int(env, argv[1], &c1)) return enif_make_int(env,2);
    if (!enif_get_int(env, argv[2], &h1)) return enif_make_int(env,3);
    if (!enif_get_int(env, argv[3], &w1)) return enif_make_int(env,4);
    if (!enif_inspect_binary(env, argv[4], &a_bin )) return enif_make_int(env,5);
    if (!enif_get_int(env, argv[5], &r2)) return enif_make_int(env,6);
    if (!enif_get_int(env, argv[6], &c2)) return enif_make_int(env,7);
    if (!enif_inspect_binary(env, argv[7], &b_bin )) return enif_make_int(env,8);
    if (!enif_get_int(env, argv[8], &n)) return enif_make_int(env,9);

    a = (float *) a_bin.data;
    b = (float *) b_bin.data;
    c = (float *) enif_make_new_binary(env, n*c1*h1*w1 * sizeof(float), &c_bin);
    d = (float *) enif_make_new_binary(env, n*r2*c2 * sizeof(float), &d_bin);

    // random-select
    for(i=0;i<n;i++){
        r = rand() % n1;
        for(j=0;j<c1;j++){
            for(k=0;k<h1;k++){
                for(l=0;l<w1;l++){
                    c[IDX4C(i,j,k,l,c1,h1,w1)] = a[IDX4C(r,j,k,l,c1,h1,w1)];
                }
            }
        }
        for(j=0;j<c2;j++){
            d[IDX2C(i,j,n)] = b[IDX2C(r,j,r2)];
        }    
    }

    tuple = enif_make_tuple2(env,c_bin,d_bin);
    return(tuple);
}


// define the array of ErlNifFunc
static ErlNifFunc nif_funcs[] = {
  // {erl_function_name, erl_function_arity, c_function}
  {"print1", 3, print1},
  {"mult1", 6, mult1},
  {"new1", 2, new1},
  {"new2", 3, new2},
  {"new3", 4, new3},
  {"new4", 5, new4},
  {"rand1", 1, rand1},
  {"add1", 3, add1},
  {"sub1", 3, sub1},
  {"emult1", 4, emult1},
  {"transpose1", 3, transpose1},
  {"ident1", 1, ident1},
  {"activate_sigmoid", 3 ,activate_sigmoid},
  {"activate_tanh", 3 , activate_tanh},
  {"activate_relu", 3, activate_relu},
  {"activate_softmax", 3, activate_softmax},
  {"differ_sigmoid", 4, differ_sigmoid},
  {"differ_tanh", 4, differ_tanh},
  {"differ_relu", 4, differ_relu},
  {"smult1", 3, smult1},
  {"trace1", 3, trace1},
  {"mean_square", 4, mean_square},
  {"cross_entropy", 4, cross_entropy},
  {"elt1", 5, elt1},
  {"set1", 6, set1},
  {"average1", 3, average1},
  {"sum1", 3, sum1},
  {"to_list1", 3, to_list1},
  {"to_list2", 4, to_list2},
  {"to_list3", 5, to_list3},
  {"momentum1", 6, momentum1},
  {"adagrad1", 6, adagrad1},
  {"accuracy1", 4, accuracy1},
  {"pooling1", 6, pooling1},
  {"unpooling1", 7, unpooling1},
  {"convolute1", 10, convolute1},
  {"deconvolute1", 10, deconvolute1},
  {"deconvolute2", 10, deconvolute2},
  {"gradfilter1", 12, gradfilter1},
  {"full1", 4, full1},
  {"unfull1", 4, unfull1},
  {"sgd1", 5, sgd1},
  {"random_select1", 7, random_select1},
  {"random_select2", 9, random_select2}
};

ERL_NIF_INIT(Elixir.Cumatrix, nif_funcs, NULL, NULL, NULL, NULL)


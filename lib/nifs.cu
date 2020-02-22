#include "erl_nif.h"
#include "cublas.h"
#include "cudnn.h"
#include "stdio.h"
#include "time.h"


#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define DEBUG return(enif_make_int(env, 0));
#define PI 3.14159265358979323846
#define SIGMOID(x)  (1 / (1+exp(-1*x)))
#define IDX3C(i,j,k,ld,ld1) ((k)*(ld*ld1-1)+((j)*(ld))+(i))


static ERL_NIF_TERM
print1(ErlNifEnv *env, int argc, const ERL_NIF_TERM *argv) {
    ErlNifBinary  a_bin;
    ERL_NIF_TERM  result;
    float        *a;
    int r,c,i,j;


    if (!enif_get_int(env, argv[0], &r )) return enif_make_badarg(env);
    if (!enif_get_int(env, argv[1], &c )) return enif_make_badarg(env);
    if (!enif_inspect_binary(env, argv[2], &a_bin )) return enif_make_badarg(env);
    
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

    enif_get_int(env, argv[0], &n);
    enif_get_double(env, argv[1], &d);
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

    enif_get_int(env, argv[0], &r1);
    enif_get_int(env, argv[1], &c1);
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
rand1(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    int n,i;
    float x,y,val;
    float *result_data;
    ERL_NIF_TERM result;

    enif_get_int(env, argv[0], &n);
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
mult(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary  a_bin, b_bin;
    ERL_NIF_TERM  c_bin;
    int r1, c1, r2, c2, n, i, j;
    float *a,*b,*c;
    cublasStatus stat;
    float* devPtrA;
    float* devPtrB;
    float* devPtrC;


    if (!enif_get_int(env, argv[0], &r1)) return enif_make_badarg(env);
    if (!enif_get_int(env, argv[1], &c1)) return enif_make_badarg(env);
    if (!enif_inspect_binary(env, argv[2], &a_bin )) return enif_make_badarg(env);
    if (!enif_get_int(env, argv[3], &r2)) return enif_make_badarg(env);
    if (!enif_get_int(env, argv[4], &c2)) return enif_make_badarg(env);
    if (!enif_inspect_binary(env, argv[5], &b_bin)) return enif_make_badarg(env);
    n = r1*c2;
    a = (float *) a_bin.data;
    b = (float *) b_bin.data;
    c = (float *) enif_make_new_binary(env, n * sizeof(float), &c_bin);

    for(j=0;j<c2;j++)
        for(i=0;i<r1;i++)
            c[IDX2C(i,j,r1)] = 0.0;


    // Initialize CUBLAS
    cublasInit();

    stat = cublasAlloc (r1*c1, sizeof(*a), (void**)&devPtrA);
    if(stat != CUBLAS_STATUS_SUCCESS)
        return(enif_make_int(env, 0)); 
    stat = cublasAlloc (r2*c2, sizeof(*b), (void**)&devPtrB);
    if(stat != CUBLAS_STATUS_SUCCESS)
        return(enif_make_int(env, 0)); 
    stat = cublasAlloc (r1*c2, sizeof(*c), (void**)&devPtrC);
    if(stat != CUBLAS_STATUS_SUCCESS)
        return(enif_make_int(env, 0)); 

    stat = cublasSetMatrix (r1, c1, sizeof(*a), a, r1, devPtrA, r1);
    if(stat != CUBLAS_STATUS_SUCCESS)
        return(enif_make_int(env, 0)); 
    stat = cublasSetMatrix (r2, c2, sizeof(*b), b, r2, devPtrB, r2);
    if(stat != CUBLAS_STATUS_SUCCESS)
        return(enif_make_int(env, 0)); 
    stat = cublasSetMatrix (r1, c2, sizeof(*c), c, r1, devPtrC, r1);
    if(stat != CUBLAS_STATUS_SUCCESS)
        return(enif_make_int(env, 0)); 


    //Sgemm
    cublasSgemm('N', 'N', r1, c2, c1, 1.0, devPtrA, r1, devPtrB, r2, 0.0, devPtrC, r1);


    stat = cublasGetMatrix (r1, c2, sizeof(*c), devPtrC, r1, c, r1);
    if(stat != CUBLAS_STATUS_SUCCESS){
        cublasFree(devPtrA);
        cublasFree(devPtrB);
        cublasFree(devPtrC);
        cublasShutdown();
        return(enif_make_int(env, 0)); 
    }

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
    int r1, c1, n;
    float *a,*b,*c;
    float *dev_a, *dev_b, *dev_c;

    if (!enif_get_int(env, argv[0], &r1)) return enif_make_badarg(env);
    if (!enif_get_int(env, argv[1], &c1)) return enif_make_badarg(env);
    if (!enif_inspect_binary(env, argv[2], &a_bin )) return enif_make_badarg(env);
    if (!enif_inspect_binary(env, argv[3], &b_bin)) return enif_make_badarg(env);
    n = r1*c1;
    a = (float *) a_bin.data;
    b = (float *) b_bin.data;
    c = (float *) enif_make_new_binary(env, n * sizeof(float), &c_bin);

    	// Allocate for GPU
	cudaMalloc((void**)&dev_a, n * sizeof(float));
	cudaMalloc((void**)&dev_b, n * sizeof(float));
	cudaMalloc((void**)&dev_c, n * sizeof(float));


    // copy from host a,b to GPU dev_a, dev_b
	cudaMemcpy(dev_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

	add1_kernel << <128, 128 >> >(dev_a, dev_b, dev_c, n);

	// copy to host c from GPU dev_c
	cudaMemcpy(c, dev_c, n * sizeof(float), cudaMemcpyDeviceToHost);

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
    int r1, c1, n;
    float *a,*b,*c;
    float *dev_a, *dev_b, *dev_c;

    if (!enif_get_int(env, argv[0], &r1)) return enif_make_badarg(env);
    if (!enif_get_int(env, argv[1], &c1)) return enif_make_badarg(env);
    if (!enif_inspect_binary(env, argv[2], &a_bin )) return enif_make_badarg(env);
    if (!enif_inspect_binary(env, argv[3], &b_bin)) return enif_make_badarg(env);
    n = r1*c1;
    a = (float *) a_bin.data;
    b = (float *) b_bin.data;
    c = (float *) enif_make_new_binary(env, n * sizeof(float), &c_bin);

    	// Allocate for GPU
	cudaMalloc((void**)&dev_a, n * sizeof(float));
	cudaMalloc((void**)&dev_b, n * sizeof(float));
	cudaMalloc((void**)&dev_c, n * sizeof(float));


    // copy from host a,b to GPU dev_a, dev_b
	cudaMemcpy(dev_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

	sub1_kernel << <128, 128 >> >(dev_a, dev_b, dev_c, n);

	// copy to host c from GPU dev_c
	cudaMemcpy(c, dev_c, n * sizeof(float), cudaMemcpyDeviceToHost);

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

    if (!enif_get_int(env, argv[0], &r1)) return enif_make_badarg(env);
    if (!enif_get_int(env, argv[1], &c1)) return enif_make_badarg(env);
    if (!enif_inspect_binary(env, argv[2], &a_bin )) return enif_make_badarg(env);
    if (!enif_inspect_binary(env, argv[3], &b_bin)) return enif_make_badarg(env);
    n = r1*c1;
    a = (float *) a_bin.data;
    b = (float *) b_bin.data;
    c = (float *) enif_make_new_binary(env, n * sizeof(float), &c_bin);

    	// Allocate for GPU
	cudaMalloc((void**)&dev_a, n * sizeof(float));
	cudaMalloc((void**)&dev_b, n * sizeof(float));
	cudaMalloc((void**)&dev_c, n * sizeof(float));


    // copy from host a,b to GPU dev_a, dev_b
	cudaMemcpy(dev_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

	emult1_kernel << <128, 128 >> >(dev_a, dev_b, dev_c, n);

	// copy to host c from GPU dev_c
	cudaMemcpy(c, dev_c, n * sizeof(float), cudaMemcpyDeviceToHost);

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

    if (!enif_get_int(env, argv[0], &r1)) return enif_make_badarg(env);
    if (!enif_get_int(env, argv[1], &c1)) return enif_make_badarg(env);
    if (!enif_inspect_binary(env, argv[2], &a_bin )) return enif_make_badarg(env);
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

    enif_get_int(env, argv[0], &n);
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

    if (!enif_get_int(env, argv[0], &r1)) return enif_make_badarg(env);
    if (!enif_get_int(env, argv[1], &c1)) return enif_make_badarg(env);
    if (!enif_inspect_binary(env, argv[2], &a_bin )) return enif_make_badarg(env);
    n = r1*c1;
    a = (float *) a_bin.data;
    b = (float *) enif_make_new_binary(env, n * sizeof(float), &b_bin);

    	// Allocate for GPU
	cudaMalloc((void**)&dev_a, n * sizeof(float));
	cudaMalloc((void**)&dev_b, n * sizeof(float));


    // copy from host a,b to GPU dev_a, dev_b
	cudaMemcpy(dev_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

	sigmoid_kernel << <128, 128 >> >(dev_a, dev_b, n);

	// copy to host c from GPU dev_c
	cudaMemcpy(b, dev_b, n * sizeof(float), cudaMemcpyDeviceToHost);

    return(b_bin);
}


__global__ void differ_sigmoid_kernel(float *a, float *b, int n)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < n)
	{   
        b[tid] = (1 - SIGMOID(a[tid])) * SIGMOID(a[tid]);
		tid += blockDim.x * gridDim.x;
	}
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

    if (!enif_get_int(env, argv[0], &r1)) return enif_make_badarg(env);
    if (!enif_get_int(env, argv[1], &c1)) return enif_make_badarg(env);
    if (!enif_inspect_binary(env, argv[2], &a_bin )) return enif_make_badarg(env);
    n = r1*c1;
    a = (float *) a_bin.data;
    b = (float *) enif_make_new_binary(env, n * sizeof(float), &b_bin);

    	// Allocate for GPU
	cudaMalloc((void**)&dev_a, n * sizeof(float));
	cudaMalloc((void**)&dev_b, n * sizeof(float));


    // copy from host a,b to GPU dev_a, dev_b
	cudaMemcpy(dev_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

	tanh_kernel << <128, 128 >> >(dev_a, dev_b, n);

	// copy to host c from GPU dev_c
	cudaMemcpy(b, dev_b, n * sizeof(float), cudaMemcpyDeviceToHost);

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

    if (!enif_get_int(env, argv[0], &r1)) return enif_make_badarg(env);
    if (!enif_get_int(env, argv[1], &c1)) return enif_make_badarg(env);
    if (!enif_inspect_binary(env, argv[2], &a_bin )) return enif_make_badarg(env);
    n = r1*c1;
    a = (float *) a_bin.data;
    b = (float *) enif_make_new_binary(env, n * sizeof(float), &b_bin);

    	// Allocate for GPU
	cudaMalloc((void**)&dev_a, n * sizeof(float));
	cudaMalloc((void**)&dev_b, n * sizeof(float));


    // copy from host a,b to GPU dev_a, dev_b
	cudaMemcpy(dev_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

	relu_kernel << <128, 128 >> >(dev_a, dev_b, n);

	// copy to host c from GPU dev_c
	cudaMemcpy(b, dev_b, n * sizeof(float), cudaMemcpyDeviceToHost);

    return(b_bin);
}

static ERL_NIF_TERM
activate_softmax(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary  a_bin;
    ERL_NIF_TERM  b_bin;
    int r1, c1, n, i, j, k;
    float *a,*b;
    float max,sum;

    if (!enif_get_int(env, argv[0], &r1)) return enif_make_badarg(env);
    if (!enif_get_int(env, argv[1], &c1)) return enif_make_badarg(env);
    if (!enif_inspect_binary(env, argv[2], &a_bin )) return enif_make_badarg(env);
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

    if (!enif_get_int(env, argv[0], &r1)) return enif_make_badarg(env);
    if (!enif_get_int(env, argv[1], &c1)) return enif_make_badarg(env);
    if (!enif_inspect_binary(env, argv[2], &a_bin )) return enif_make_badarg(env);
    if (!enif_inspect_binary(env, argv[3], &b_bin)) return enif_make_badarg(env);
    n = r1*c1;
    a = (float *) a_bin.data;
    b = (float *) b_bin.data;
    c = (float *) enif_make_new_binary(env, n * sizeof(float), &c_bin);

    	// Allocate for GPU
	cudaMalloc((void**)&dev_a, n * sizeof(float));
	cudaMalloc((void**)&dev_b, n * sizeof(float));
	cudaMalloc((void**)&dev_c, n * sizeof(float));


    // copy from host a,b to GPU dev_a, dev_b
	cudaMemcpy(dev_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

	differ_sigmoid_kernel << <128, 128 >> >(dev_a, dev_b, dev_c, n);

	// copy to host c from GPU dev_c
	cudaMemcpy(c, dev_c, n * sizeof(float), cudaMemcpyDeviceToHost);

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

    if (!enif_get_int(env, argv[0], &r1)) return enif_make_badarg(env);
    if (!enif_get_int(env, argv[1], &c1)) return enif_make_badarg(env);
    if (!enif_inspect_binary(env, argv[2], &a_bin )) return enif_make_badarg(env);
    if (!enif_inspect_binary(env, argv[3], &b_bin)) return enif_make_badarg(env);
    n = r1*c1;
    a = (float *) a_bin.data;
    b = (float *) b_bin.data;
    c = (float *) enif_make_new_binary(env, n * sizeof(float), &c_bin);

    	// Allocate for GPU
	cudaMalloc((void**)&dev_a, n * sizeof(float));
	cudaMalloc((void**)&dev_b, n * sizeof(float));
	cudaMalloc((void**)&dev_c, n * sizeof(float));


    // copy from host a,b to GPU dev_a, dev_b
	cudaMemcpy(dev_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

	differ_tanh_kernel << <128, 128 >> >(dev_a, dev_b, dev_c, n);

	// copy to host c from GPU dev_c
	cudaMemcpy(c, dev_c, n * sizeof(float), cudaMemcpyDeviceToHost);

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

    if (!enif_get_int(env, argv[0], &r1)) return enif_make_badarg(env);
    if (!enif_get_int(env, argv[1], &c1)) return enif_make_badarg(env);
    if (!enif_inspect_binary(env, argv[2], &a_bin )) return enif_make_badarg(env);
    if (!enif_inspect_binary(env, argv[3], &b_bin)) return enif_make_badarg(env);
    n = r1*c1;
    a = (float *) a_bin.data;
    b = (float *) b_bin.data;
    c = (float *) enif_make_new_binary(env, n * sizeof(float), &c_bin);

    	// Allocate for GPU
	cudaMalloc((void**)&dev_a, n * sizeof(float));
	cudaMalloc((void**)&dev_b, n * sizeof(float));
	cudaMalloc((void**)&dev_c, n * sizeof(float));


    // copy from host a,b to GPU dev_a, dev_b
	cudaMemcpy(dev_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

	differ_relu_kernel << <128, 128 >> >(dev_a, dev_b, dev_c, n);

	// copy to host c from GPU dev_c
	cudaMemcpy(c, dev_c, n * sizeof(float), cudaMemcpyDeviceToHost);

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
    int r1, c1, n;
    float *a,*b;
    float *dev_a, *dev_b;
    double s;


    if (!enif_get_double(env, argv[0], &s)) return enif_make_badarg(env);
    if (!enif_get_int(env, argv[1], &r1)) return enif_make_badarg(env);
    if (!enif_get_int(env, argv[2], &c1)) return enif_make_badarg(env);
    if (!enif_inspect_binary(env, argv[3], &a_bin )) return enif_make_badarg(env);
    n = r1*c1;
    a = (float *) a_bin.data;
    b = (float *) enif_make_new_binary(env, n * sizeof(float), &b_bin);

    	// Allocate for GPU
	cudaMalloc((void**)&dev_a, n * sizeof(float));
	cudaMalloc((void**)&dev_b, n * sizeof(float));


    // copy from host a,b to GPU dev_a, dev_b
	cudaMemcpy(dev_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

	smult_kernel << <128, 128 >> >((float)s,dev_a, dev_b, n);

	// copy to host c from GPU dev_c
	cudaMemcpy(b, dev_b, n * sizeof(float), cudaMemcpyDeviceToHost);

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

    if (!enif_get_int(env, argv[0], &r1)) return enif_make_badarg(env);
    if (!enif_get_int(env, argv[1], &c1)) return enif_make_badarg(env);
    if (!enif_inspect_binary(env, argv[2], &a_bin )) return enif_make_badarg(env);
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


    if (!enif_get_int(env, argv[0], &r1)) return enif_make_badarg(env);
    if (!enif_get_int(env, argv[1], &c1)) return enif_make_badarg(env);
    if (!enif_inspect_binary(env, argv[2], &a_bin )) return enif_make_badarg(env);
    if (!enif_inspect_binary(env, argv[3], &b_bin )) return enif_make_badarg(env);

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


    if (!enif_get_int(env, argv[0], &r1)) return enif_make_badarg(env);
    if (!enif_get_int(env, argv[1], &c1)) return enif_make_badarg(env);
    if (!enif_inspect_binary(env, argv[2], &a_bin )) return enif_make_badarg(env);
    if (!enif_inspect_binary(env, argv[3], &b_bin )) return enif_make_badarg(env);

    a = (float *) a_bin.data;
    b = (float *) b_bin.data;

    
    delta = 1e-7;
    s = 0.0;
    for(i=0;i<r1;i++){
        for (j=0;j<c1;j++){
            d = a[IDX2C(i,j,r1)];
            s = s + b[IDX2C(i,j,r1)] * log(d+delta);             
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

    if (!enif_get_int(env, argv[0], &r1)) return enif_make_badarg(env);
    if (!enif_get_int(env, argv[1], &c1)) return enif_make_badarg(env);
    if (!enif_get_int(env, argv[2], &i)) return enif_make_badarg(env);
    if (!enif_get_int(env, argv[3], &j)) return enif_make_badarg(env);
    if (!enif_inspect_binary(env, argv[4], &a_bin )) return enif_make_badarg(env);
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

    if (!enif_get_int(env, argv[0], &r1)) return enif_make_badarg(env);
    if (!enif_get_int(env, argv[1], &c1)) return enif_make_badarg(env);
    if (!enif_inspect_binary(env, argv[2], &a_bin )) return enif_make_badarg(env);
    if (!enif_get_int(env, argv[3], &x)) return enif_make_badarg(env);
    if (!enif_get_int(env, argv[4], &y)) return enif_make_badarg(env);
    if (!enif_get_double(env, argv[5], &val)) return enif_make_badarg(env);


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

    if (!enif_get_int(env, argv[0], &r1)) return enif_make_badarg(env);
    if (!enif_get_int(env, argv[1], &c1)) return enif_make_badarg(env);
    if (!enif_inspect_binary(env, argv[2], &a_bin )) return enif_make_badarg(env);

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


static ERL_NIF_TERM
sum1(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary  a_bin;
    ERL_NIF_TERM  result;
    int r1, c1, i, j;
    float *a;
    float sum;

    if (!enif_get_int(env, argv[0], &r1)) return enif_make_badarg(env);
    if (!enif_get_int(env, argv[1], &c1)) return enif_make_badarg(env);
    if (!enif_inspect_binary(env, argv[2], &a_bin )) return enif_make_badarg(env);
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

static ERL_NIF_TERM
to_list1(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary  a_bin;
    ERL_NIF_TERM  head,list;
    int r1, c1, i, j;
    float *a;


    if (!enif_get_int(env, argv[0], &r1)) return enif_make_badarg(env);
    if (!enif_get_int(env, argv[1], &c1)) return enif_make_badarg(env);
    if (!enif_inspect_binary(env, argv[2], &a_bin )) return enif_make_badarg(env);
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
  def momentum(v, g, lr) do
    Matrex.apply(v, g, fn v, g -> 0.5 * v - lr * g end)
  end
*/
  __global__ void momentum_kernel(float *a, float *b, float *c, float lr, int n)
  {
      int tid = threadIdx.x + blockIdx.x * blockDim.x;
      while (tid < n)
      {   
          c[tid] = (0.5 * a[tid]) - (lr * b[tid]);
          tid += blockDim.x * gridDim.x;
      }
  }
  
  static ERL_NIF_TERM
  momentum1(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
      ErlNifBinary  a_bin,b_bin;
      ERL_NIF_TERM  c_bin;
      int r1, c1, n;
      float *a,*b, *c;
      float *dev_a, *dev_b, *dev_c;
      double lr;
  
      if (!enif_get_int(env, argv[0], &r1)) return enif_make_badarg(env);
      if (!enif_get_int(env, argv[1], &c1)) return enif_make_badarg(env);
      if (!enif_inspect_binary(env, argv[2], &a_bin )) return enif_make_badarg(env);
      if (!enif_inspect_binary(env, argv[3], &b_bin )) return enif_make_badarg(env);
      if (!enif_get_double(env, argv[4], &lr)) return enif_make_badarg(env);

      n = r1*c1;
      a = (float *) a_bin.data;
      b = (float *) b_bin.data;
      c = (float *) enif_make_new_binary(env, n * sizeof(float), &c_bin);
  
          // Allocate for GPU
      cudaMalloc((void**)&dev_a, n * sizeof(float));
      cudaMalloc((void**)&dev_b, n * sizeof(float));
      cudaMalloc((void**)&dev_c, n * sizeof(float));
  
      // copy from host a,b to GPU dev_a, dev_b
      cudaMemcpy(dev_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(dev_b, b, n * sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(dev_c, c, n * sizeof(float), cudaMemcpyHostToDevice);
  
      momentum_kernel << <128, 128 >> >(dev_a, dev_b, dev_c, float(lr), n);
  
      // copy to host c from GPU dev_c
      cudaMemcpy(c, dev_c, n * sizeof(float), cudaMemcpyDeviceToHost);
  
      return(c_bin);
  }

  /*
  def adagrad(w, g, h, lr) do
    CM.sub(w, Matrex.apply(g, h, fn g, h -> lr * (1 / adagrad_sqrt(h)) * g end))
  end

  def adagrad_sqrt(x) do
    if x != 0 do
      :math.sqrt(x)
    else
      1
    end
  end
  */
  
  __global__ void adagrad_kernel(float *a, float *b, float *c, float h, float lr, int n)
  {
      int tid = threadIdx.x + blockIdx.x * blockDim.x;
      while (tid < n)
      {   
          if(h != 0)
            c[tid] = a[tid] - (lr * (1 / sqrt(h)) * b[tid]);
          else 
            c[tid] = a[tid] - (lr * b[tid]);
          tid += blockDim.x * gridDim.x;
      }
  }
  
  static ERL_NIF_TERM
  adagrad1(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
      ErlNifBinary  a_bin,b_bin;
      ERL_NIF_TERM  c_bin;
      int r1, c1, n;
      float *a,*b, *c;
      float *dev_a, *dev_b, *dev_c;
      double h,lr;
  
      if (!enif_get_int(env, argv[0], &r1)) return enif_make_badarg(env);
      if (!enif_get_int(env, argv[1], &c1)) return enif_make_badarg(env);
      if (!enif_inspect_binary(env, argv[2], &a_bin )) return enif_make_badarg(env);
      if (!enif_inspect_binary(env, argv[3], &b_bin )) return enif_make_badarg(env);
      if (!enif_get_double(env, argv[4], &h)) return enif_make_badarg(env);
      if (!enif_get_double(env, argv[5], &lr)) return enif_make_badarg(env);

      n = r1*c1;
      a = (float *) a_bin.data;
      b = (float *) b_bin.data;
      c = (float *) enif_make_new_binary(env, n * sizeof(float), &c_bin);
  
          // Allocate for GPU
      cudaMalloc((void**)&dev_a, n * sizeof(float));
      cudaMalloc((void**)&dev_b, n * sizeof(float));
      cudaMalloc((void**)&dev_c, n * sizeof(float));
  
      // copy from host a,b to GPU dev_a, dev_b
      cudaMemcpy(dev_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(dev_b, b, n * sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(dev_c, c, n * sizeof(float), cudaMemcpyHostToDevice);
  
      adagrad_kernel << <128, 128 >> >(dev_a, dev_b, dev_c, float(h), float(lr), n);
  
      // copy to host c from GPU dev_c
      cudaMemcpy(c, dev_c, n * sizeof(float), cudaMemcpyDeviceToHost);
  
      return(c_bin);
  }

  static ERL_NIF_TERM
  accuracy1(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
      ErlNifBinary  a_bin;
      ERL_NIF_TERM  head,list,result;
      int r1, c1, i, j, n, index,sum;
      float *a;
      double max,rate;
  
      if (!enif_get_int(env, argv[0], &r1)) return enif_make_badarg(env);
      if (!enif_get_int(env, argv[1], &c1)) return enif_make_badarg(env);
      if (!enif_inspect_binary(env, argv[2], &a_bin )) return enif_make_badarg(env);

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


  
  __global__ void dev_const(float *px, float k) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    px[tid] = k;
  }
  
  __global__ void dev_iota(float *px) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    px[tid] = tid;
  }
  
  
  
  /*
  1st arg n of input tensor
  2nd arg c of input tensor
  3rd arg h of input tensor
  4th arg w of input tensor
  5th arg k of filter tensor
  6th arg c of filter tensor
  7th arg h of filter tensor
  8th arg w of filter tensor
  9th arg binary of input tensor
  10th arg binary of filter tensor1
  11th arg h of padding 
  12th arg w of padding
  13th arg h of stride 
  14th arg w of stride
  */
  static ERL_NIF_TERM
  convolute1(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary  a_bin,b_bin;
    ERL_NIF_TERM  c_bin;
    int in_n,in_c,in_h,in_w;
    int filt_k,filt_c,filt_h,filt_w;
    int pad_h,pad_w,str_h,str_w,dil_h,dil_w; 
    int out_n,out_c,out_h,out_w;
    float *in_data,*filt_data,*out_data;

    if (!enif_get_int(env, argv[0], &in_n)) return enif_make_badarg(env);
    if (!enif_get_int(env, argv[1], &in_c)) return enif_make_badarg(env);
    if (!enif_get_int(env, argv[2], &in_h)) return enif_make_badarg(env);
    if (!enif_get_int(env, argv[3], &in_w)) return enif_make_badarg(env);
    if (!enif_get_int(env, argv[4], &filt_k)) return enif_make_badarg(env);
    if (!enif_get_int(env, argv[5], &filt_c)) return enif_make_badarg(env);
    if (!enif_get_int(env, argv[6], &filt_h)) return enif_make_badarg(env);
    if (!enif_get_int(env, argv[7], &filt_w)) return enif_make_badarg(env);
    if (!enif_inspect_binary(env, argv[8], &a_bin )) return enif_make_badarg(env);
    if (!enif_inspect_binary(env, argv[9], &b_bin )) return enif_make_badarg(env);
    if (!enif_get_int(env, argv[10], &pad_h)) return enif_make_badarg(env);
    if (!enif_get_int(env, argv[11], &pad_w)) return enif_make_badarg(env);
    if (!enif_get_int(env, argv[12], &str_h)) return enif_make_badarg(env);
    if (!enif_get_int(env, argv[13], &str_w)) return enif_make_badarg(env);


    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    
    // input
    /*
    cudnnSetTensor4dDescriptor( )
    1st arg tensorDesc input/Output. Handle to a previously created tensor descriptor. 
    2nd arg Format input. Type of format. 
    3rd arg n Input. Number of images. 
    4th arg c Input. Number of feature maps per image. 
    5th arg h Input. Height of each feature map.
    6th arg w Input. Width of each feature map. 
    */
    cudnnTensorDescriptor_t in_desc;
    cudnnCreateTensorDescriptor(&in_desc);
    cudnnSetTensor4dDescriptor(
          in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          in_n, in_c, in_h, in_w);
  

    in_data = (float *) a_bin.data;
    cudaMalloc(&in_data, in_n * in_c * in_h * in_w * sizeof(float));
        
    // filter
    /*
    cudnnSetFilter4dDescriptor( )
    1st arg filterDesc Input/Output. Handle to a previously created filter descriptor.
    2nd arg datatype Input. Data type.
    3rd arg format Input. Type of format.
    4th arg k Input. Number of output feature maps.
    5th arg c Input. Number of input feature maps.
    6th arg h Input. Height of each filter.
    7th arg w Input. Width of each filter.
    */
    cudnnFilterDescriptor_t filt_desc;
    cudnnCreateFilterDescriptor(&filt_desc);
    cudnnSetFilter4dDescriptor(
          filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
          filt_k, filt_c, filt_h, filt_w);
  
    filt_data = (float *) b_bin.data;
    cudaMalloc(
        &filt_data, filt_k * filt_c * filt_h * filt_w * sizeof(float));
    
    // convolution
    dil_h = 1;
    dil_w = 1;
    
    /*
    cudnnSetConvolution2dDescriptor( )
    1st arg convDesc Input/Output. Handle to a previously created convolution descriptor.
    2nd arg pad_h Input. zero-padding height: number of rows of zeros implicitly concatenated onto the top and onto the bottom of input images.
    3rd arg pad_w Input. zero-padding width: number of columns of zeros implicitly concatenated onto the left and onto the right of input images.
    4th arg u Input. Vertical filter stride.
    5th arg v Input. Horizontal filter stride.
    6th arg dilation_h Input. Filter height dilation.
    7th arg dilation_w Input. Filter width dilation.
    8th arg mode Input. Selects between CUDNN_CONVOLUTION and CUDNN_CROSS_CORRELATION.
    9th arg computeType Input. compute precision.
    */
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnCreateConvolutionDescriptor(&conv_desc);
    cudnnSetConvolution2dDescriptor(
          conv_desc,
          pad_h, pad_w, str_h, str_w, dil_h, dil_w,
          CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT);
  
    // output
    
    cudnnGetConvolution2dForwardOutputDim(
          conv_desc, in_desc, filt_desc,
          &out_n, &out_c, &out_h, &out_w);
  
    
  
    cudnnTensorDescriptor_t out_desc;
    cudnnCreateTensorDescriptor(&out_desc);
    cudnnSetTensor4dDescriptor(
          out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          out_n, out_c, out_h, out_w);
  
    out_data = (float *) enif_make_new_binary(env, out_n * out_c * out_h * out_w * sizeof(float), &c_bin);
    cudaMalloc(
          &out_data, out_n * out_c * out_h * out_w * sizeof(float));

    /*
    cudnnGetConvolutionForwardAlgorithm( )
    1st arg convDesc Input. Handle to a previously created convolution descriptor.
    2nd arg inputTensorDesc Input. Handle to a previously initialized tensor descriptor.
    3rd arg filterDesc Input. Handle to a previously initialized filter descriptor.
    4th arg n Output. Number of output images.
    5th arg c Output. Number of output feature maps per image.
    6th arg h Output. Height of each output feature map.
    7th arg w Output. Width of each output feature map.
    */
  
    // algorithm
    cudnnConvolutionFwdAlgo_t algo;
    cudnnGetConvolutionForwardAlgorithm(
          cudnn,
          in_desc, filt_desc, conv_desc, out_desc,
          CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo);

    /*
    cudnnGetConvolutionForwardWorkspaceSize( )
    1st arg handle Input. Handle to a previously created cuDNN context.
    2nd arg xDesc Input. Handle to the previously initialized x tensor descriptor.
    3rd arg wDesc Input. Handle to a previously initialized filter descriptor.
    4th arg convDesc Input. Previously initialized convolution descriptor.
    5th arg yDesc Input. Handle to the previously initialized y tensor descriptor.
    6th arg algo Input. Enumerant that specifies the chosen convolution algorithm
    7th arg sizeInBytes Output. Amount of GPU memory needed as workspace to be able to execute a forward convolution with the specified algo
    */
  
    // workspace
    size_t ws_size;
    cudnnGetConvolutionForwardWorkspaceSize(
          cudnn, in_desc, filt_desc, conv_desc, out_desc, algo, &ws_size);
  
    float *ws_data;
    cudaMalloc(&ws_data, ws_size);
  
    
    /*
    cudnnConvolutionForward( )
    1st arg handle Input. Handle to a previously created cuDNN context.
    2nd arg alpha Input. Pointers to scaling factors (in host memory) used to blend the computation result with prior value in the output layer as follows: dstValue = alpha[0]*result + beta[0]*priorDstValue
    3rd arg xDesc Input. Handle to a previously initialized tensor descriptor.
    4th arg x Input. Data pointer to GPU memory associated with the tensor descriptor
    5th arg wDesc Input. Handle to a previously initialized filter descriptor.
    6th arg w Input. Data pointer to GPU memory associated with the filter descriptor
    7th arg convDesc Input. Previously initialized convolution descriptor.
    8th arg algo Input. Enumerant that specifies which convolution algorithm shoud be used to compute the results.
    9th arg workSpace Input. Data pointer to GPU memory to a workspace needed to able to execute the specified algorithm. If no workspace is needed for a particular algorithm, that pointer can be nil.
    10th arg workSpaceSizeInBytes Input. Specifies the size in bytes of the provided
    11th arg beta Input. Pointers to scaling factors (in host memory) used to blend the computation result with prior value in the output layer as follows: dstValue = alpha[0]*result + beta[0]*priorDstValue
    12th arg yDesc Input. Handle to a previously initialized tensor descriptor.
    13th arg y Input/Output. Data pointer to GPU memory associated with the tensor descriptor yDesc that carries the result of the convolution.
    */

    // perform
    float alpha = 1.f;
    float beta = 0.f;
    cudnnConvolutionForward(
        cudnn,
        &alpha, in_desc, in_data, filt_desc, filt_data,
        conv_desc, algo, ws_data, ws_size,
        &beta, out_desc, out_data);
  
    
    // finalizing
    cudaFree(ws_data);
    cudaFree(out_data);
    cudnnDestroyTensorDescriptor(out_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudaFree(filt_data);
    cudnnDestroyFilterDescriptor(filt_desc);
    cudaFree(in_data);
    cudnnDestroyTensorDescriptor(in_desc);
    cudnnDestroy(cudnn);

    return(c_bin);
  }

  
// define the array of ErlNifFunc
static ErlNifFunc nif_funcs[] = {
  // {erl_function_name, erl_function_arity, c_function}
  {"print1", 3, print1},
  {"mult1", 6, mult},
  {"new1", 2, new1},
  {"new2", 3, new2},
  {"rand1", 1, rand1},
  {"add1", 4, add1},
  {"sub1", 4, sub1},
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
  {"smult1", 4, smult1},
  {"trace1", 3, trace1},
  {"mean_square", 4, mean_square},
  {"cross_entropy", 4, cross_entropy},
  {"elt1", 5, elt1},
  {"set1", 6, set1},
  {"average1", 3, average1},
  {"sum1", 3, sum1},
  {"to_list1", 3, to_list1},
  {"momentum1", 5, momentum1},
  {"adagrad1", 6, adagrad1},
  {"accuracy1", 4, accuracy1},
  {"convolute1", 14, convolute1}
};

ERL_NIF_INIT(Elixir.Cumatrix, nif_funcs, NULL, NULL, NULL, NULL)


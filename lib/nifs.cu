#include "erl_nif.h"
#include "cublas.h"
#include "stdio.h"
#include "time.h"


#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define DEBUG return(enif_make_int(env, 0));
#define PI 3.14159265358979323846
#define SIGMOID(x)  (1 / (1+exp(-1*x)))



static ERL_NIF_TERM
new1(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    int n,i;
    ERL_NIF_TERM head, list;

    enif_get_int(env, argv[0], &n);
    list = enif_make_list(env, 0);
    head = enif_make_double(env,0.0);
    for(i=0;i<n;i++){
        list = enif_make_list_cell(env,head,list);
    }
    return(list);
}


static ERL_NIF_TERM
rand1(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    int n,i;
    double x,y,val;
    ERL_NIF_TERM head, list;

    enif_get_int(env, argv[0], &n);
    enif_get_double(env, argv[1], &val);
    head = enif_make_double(env,val);
    list = enif_make_list(env, 0);
    for(i=0;i<n;i++){
        //box_muller
        x = (double)rand()/(double)RAND_MAX;
        y = (double)rand()/(double)RAND_MAX;
        val = sqrt(-2.0 * log(x)) * cos(2.0 * PI * y);
        head = enif_make_double(env,val);
        list = enif_make_list_cell(env,head,list);
    }
    return(list);
}



static ERL_NIF_TERM
mult(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    int r1, c1, r2, c2, i, j;
    double d;
    cublasStatus stat;
    float* devPtrA;
    float* devPtrB;
    float* devPtrC;
    float* a = 0;
    float* b = 0;
    float* c = 0;
    ERL_NIF_TERM head, list;

    enif_get_int(env, argv[0], &r1);
    enif_get_int(env, argv[1], &c1);
    enif_get_int(env, argv[3], &r2);
    enif_get_int(env, argv[4], &c2);
    
    if(c1 != r2)
      return(enif_make_int(env, 0)); /*error*/
        

    // Memory Allocation
    a = (float *)malloc (r1 * c1 * sizeof (*a));
    if (!a)
        return(enif_make_int(env, 0)); /*error*/
    b = (float *)malloc (r2 * c2 * sizeof (*b));
    if (!b)
        return(enif_make_int(env, 0)); /*error*/
    c = (float *)malloc (r1 * c2 * sizeof (*c));
    if (!c)
        return(enif_make_int(env, 0)); /*error*/


    // Set matrix data to cublas 
    list = argv[2]; /* matrix1 */
    for(j=0;j<c1;j++){
        for(i=0;i<r1;i++){
            enif_get_list_cell(env, list, &head, &list);
            enif_get_double(env,head,&d);
            a[IDX2C(i,j,r1)] = (float)d;
        }
    }

    list = argv[5]; /* matrix2 */
    for (j=0;j<c2;j++){
        for (i=0;i<r2;i++){
            enif_get_list_cell(env, list, &head, &list);
            enif_get_double(env,head,&d);
            b[IDX2C(i,j,r2)] = (float)d;
        }
    }
    for(j=0;j<c2;j++)
        for(i=0;i<r1;i++)
            c[IDX2C(i,j,r1)] = 0.0;

    // Initialize CUBLAS
    cublasInit();

    stat = cublasAlloc (r1*c1, sizeof(*a), (void**)&devPtrA);
    if(stat != CUBLAS_STATUS_SUCCESS)
        return(enif_make_int(env, 0)); /*error*/
    stat = cublasAlloc (r2*c2, sizeof(*b), (void**)&devPtrB);
    if(stat != CUBLAS_STATUS_SUCCESS)
        return(enif_make_int(env, 0)); /*error*/
    stat = cublasAlloc (r1*c2, sizeof(*c), (void**)&devPtrC);
    if(stat != CUBLAS_STATUS_SUCCESS)
        return(enif_make_int(env, 0)); /*error*/

    stat = cublasSetMatrix (r1, c1, sizeof(*a), a, r1, devPtrA, r1);
    if(stat != CUBLAS_STATUS_SUCCESS)
        return(enif_make_int(env, 0)); /*error*/
    stat = cublasSetMatrix (r2, c2, sizeof(*b), b, r2, devPtrB, r2);
    if(stat != CUBLAS_STATUS_SUCCESS)
        return(enif_make_int(env, 0)); /*error*/
    stat = cublasSetMatrix (r1, c2, sizeof(*c), c, r1, devPtrC, r1);
    if(stat != CUBLAS_STATUS_SUCCESS)
        return(enif_make_int(env, 0)); /*error*/


    //Sgemm
    cublasSgemm('N', 'N', r1, c2, c1, 1.0, devPtrA, r1, devPtrB, r2, 0.0, devPtrC, r1);


    stat = cublasGetMatrix (r1, c2, sizeof(*c), devPtrC, r1, c, r1);
    if(stat != CUBLAS_STATUS_SUCCESS){
        cublasFree(devPtrA);
        cublasFree(devPtrB);
        cublasFree(devPtrC);
        cublasShutdown();
        return(enif_make_int(env, 0)); /*error*/
    }

    // Shutdown CUBLAS
    cublasFree(devPtrA);
    cublasFree(devPtrB);
    cublasFree(devPtrC);
    cublasShutdown();

    // Set matrix After sgemm
    list = enif_make_list(env, 0);
    for(j=c2-1;j>=0;j--){
        for (i=r1-1;i>=0;i--){
            head = enif_make_double(env,(double)c[IDX2C(i,j,r1)]);
            list = enif_make_list_cell(env,head,list);
        }
    }

    free(a);
    free(b);
    free(c);
    return list;
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
    int r1, c1, i, j, n;
    double d;
    float* a = 0;
    float* b = 0;
    float* c = 0;
    float *dev_a, *dev_b, *dev_c;

    ERL_NIF_TERM head, list;

    enif_get_int(env, argv[0], &r1);
    enif_get_int(env, argv[1], &c1);
    n = r1*c1;

    // Memory Allocation
    a = (float *)malloc (r1 * c1 * sizeof (*a));
    if (!a)
        return(enif_make_int(env, 0)); /*error*/
    b = (float *)malloc (r1 * c1 * sizeof (*b));
    if (!b)
        return(enif_make_int(env, 0)); /*error*/
    c = (float *)malloc (r1 * c1 * sizeof (*c));
    if (!c)
        return(enif_make_int(env, 0)); /*error*/
    

	// Allocate for GPU
	cudaMalloc((void**)&dev_a, r1*c1 * sizeof(float));
	cudaMalloc((void**)&dev_b, r1*c1 * sizeof(float));
	cudaMalloc((void**)&dev_c, r1*c1 * sizeof(float));


    // Set matrix data to cublas 
    list = argv[2]; /* matrix1 */
    for(j=0;j<c1;j++){
        for(i=0;i<r1;i++){
            enif_get_list_cell(env, list, &head, &list);
            enif_get_double(env,head,&d);
            a[IDX2C(i,j,r1)] = (float)d;
        }
    }

    list = argv[3]; /* matrix2 */
    for (j=0;j<c1;j++){
        for (i=0;i<r1;i++){
            enif_get_list_cell(env, list, &head, &list);
            enif_get_double(env,head,&d);
            b[IDX2C(i,j,r1)] = (float)d;
        }
    }
    
    
    // copy from host a,b to GPU dev_a, dev_b
	cudaMemcpy(dev_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

	add1_kernel << <128, 128 >> >(dev_a, dev_b, dev_c, n);

	// copy to host c from GPU dev_c
	cudaMemcpy(c, dev_c, n * sizeof(float), cudaMemcpyDeviceToHost);



    // Set matrix After kernel
    list = enif_make_list(env, 0);
    for(j=c1-1;j>=0;j--){
        for (i=r1-1;i>=0;i--){
            head = enif_make_double(env,(double)c[IDX2C(i,j,r1)]);
            list = enif_make_list_cell(env,head,list);
        }
    }

    cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

    free(a);
    free(b);
    free(c);
    return list;
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
    int r1, c1, i, j, n;
    double d;
    float* a = 0;
    float* b = 0;
    float* c = 0;
    float *dev_a, *dev_b, *dev_c;

    ERL_NIF_TERM head, list;

    enif_get_int(env, argv[0], &r1);
    enif_get_int(env, argv[1], &c1);
    n = r1*c1;

    // Memory Allocation
    a = (float *)malloc (r1 * c1 * sizeof (*a));
    if (!a)
        return(enif_make_int(env, 0)); /*error*/
    b = (float *)malloc (r1 * c1 * sizeof (*b));
    if (!b)
        return(enif_make_int(env, 0)); /*error*/
    c = (float *)malloc (r1 * c1 * sizeof (*c));
    if (!c)
        return(enif_make_int(env, 0)); /*error*/
    

	// Allocate for GPU
	cudaMalloc((void**)&dev_a, r1*c1 * sizeof(float));
	cudaMalloc((void**)&dev_b, r1*c1 * sizeof(float));
	cudaMalloc((void**)&dev_c, r1*c1 * sizeof(float));

    // Set matrix data to cublas 
    list = argv[2]; /* matrix1 */
    for(j=0;j<c1;j++){
        for(i=0;i<r1;i++){
            enif_get_list_cell(env, list, &head, &list);
            enif_get_double(env,head,&d);
            a[IDX2C(i,j,r1)] = (float)d;
        }
    }
    


    list = argv[3]; /* matrix2 */
    for (j=0;j<c1;j++){
        for (i=0;i<r1;i++){
            enif_get_list_cell(env, list, &head, &list);
            enif_get_double(env,head,&d);
            b[IDX2C(i,j,r1)] = (float)d;
        }
    }
    
    
    // copy from host a,b to GPU dev_a, dev_b
	cudaMemcpy(dev_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

	sub1_kernel << <128, 128 >> >(dev_a, dev_b, dev_c, n);

	// copy to host c from GPU dev_c
	cudaMemcpy(c, dev_c, n * sizeof(float), cudaMemcpyDeviceToHost);



    // Set matrix After kernel
    list = enif_make_list(env, 0);
    for(j=c1-1;j>=0;j--){
        for (i=r1-1;i>=0;i--){
            head = enif_make_double(env,(double)c[IDX2C(i,j,r1)]);
            list = enif_make_list_cell(env,head,list);
        }
    }

    cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

    free(a);
    free(b);
    free(c);
    return list;
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
    int r1, c1, i, j, n;
    double d;
    float* a = 0;
    float* b = 0;
    float* c = 0;
    float *dev_a, *dev_b, *dev_c;

    ERL_NIF_TERM head, list;

    enif_get_int(env, argv[0], &r1);
    enif_get_int(env, argv[1], &c1);
    n = r1*c1;

    // Memory Allocation
    a = (float *)malloc (r1 * c1 * sizeof (*a));
    if (!a)
        return(enif_make_int(env, 0)); /*error*/
    b = (float *)malloc (r1 * c1 * sizeof (*b));
    if (!b)
        return(enif_make_int(env, 0)); /*error*/
    c = (float *)malloc (r1 * c1 * sizeof (*c));
    if (!c)
        return(enif_make_int(env, 0)); /*error*/
    

	// Allocate for GPU
	cudaMalloc((void**)&dev_a, r1*c1 * sizeof(float));
	cudaMalloc((void**)&dev_b, r1*c1 * sizeof(float));
	cudaMalloc((void**)&dev_c, r1*c1 * sizeof(float));


    // Set matrix data to cublas 
    list = argv[2]; /* matrix1 */
    for(j=0;j<c1;j++){
        for(i=0;i<r1;i++){
            enif_get_list_cell(env, list, &head, &list);
            enif_get_double(env,head,&d);
            a[IDX2C(i,j,r1)] = (float)d;
        }
    }

    list = argv[3]; /* matrix2 */
    for (j=0;j<c1;j++){
        for (i=0;i<r1;i++){
            enif_get_list_cell(env, list, &head, &list);
            enif_get_double(env,head,&d);
            b[IDX2C(i,j,r1)] = (float)d;
        }
    }
    
    
    // copy from host a,b to GPU dev_a, dev_b
	cudaMemcpy(dev_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

	emult1_kernel << <128, 128 >> >(dev_a, dev_b, dev_c, n);

	// copy to host c from GPU dev_c
	cudaMemcpy(c, dev_c, n * sizeof(float), cudaMemcpyDeviceToHost);



    // Set matrix After kernel
    list = enif_make_list(env, 0);
    for(j=c1-1;j>=0;j--){
        for (i=r1-1;i>=0;i--){
            head = enif_make_double(env,(double)c[IDX2C(i,j,r1)]);
            list = enif_make_list_cell(env,head,list);
        }
    }

    cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

    free(a);
    free(b);
    free(c);
    return list;

}


__global__ void badd1_kernel(float *a, float *b, float *c, int n)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < n)
	{
		c[tid] = a[tid] + b[tid];
		tid += blockDim.x * gridDim.x;
	}
}


static ERL_NIF_TERM
badd1(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    int r1, c1, i, j, n;
    double d;
    float* a = 0;
    float* b = 0;
    float* c = 0;
    float *dev_a, *dev_b, *dev_c;

    ERL_NIF_TERM head, list;

    enif_get_int(env, argv[0], &r1);
    enif_get_int(env, argv[1], &c1);
    n = r1*c1;

    // Memory Allocation
    a = (float *)malloc (r1 * c1 * sizeof (*a));
    if (!a)
        return(enif_make_int(env, 0)); /*error*/
    b = (float *)malloc (r1 * c1 * sizeof (*b));
    if (!b)
        return(enif_make_int(env, 0)); /*error*/
    c = (float *)malloc (r1 * c1 * sizeof (*c));
    if (!c)
        return(enif_make_int(env, 0)); /*error*/
    

	// Allocate for GPU
	cudaMalloc((void**)&dev_a, r1*c1 * sizeof(float));
	cudaMalloc((void**)&dev_b, r1*c1 * sizeof(float));
	cudaMalloc((void**)&dev_c, r1*c1 * sizeof(float));


    // Set matrix data to cublas 
    list = argv[2]; /* matrix1 */
    for(j=0;j<c1;j++){
        for(i=0;i<r1;i++){
            enif_get_list_cell(env, list, &head, &list);
            enif_get_double(env,head,&d);
            a[IDX2C(i,j,r1)] = (float)d;
        }
    }

    for (i=0;i<r1;i++){
        list = argv[3]; /* matrix2 row vector bias */
        for (j=0;j<c1;j++){
            enif_get_list_cell(env, list, &head, &list);
            enif_get_double(env,head,&d);
            b[IDX2C(i,j,r1)] = (float)d;
        }
    }
    
    
    // copy from host a,b to GPU dev_a, dev_b
	cudaMemcpy(dev_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

	badd1_kernel << <128, 128 >> >(dev_a, dev_b, dev_c, n);

	// copy to host c from GPU dev_c
	cudaMemcpy(c, dev_c, n * sizeof(float), cudaMemcpyDeviceToHost);



    // Set matrix After kernel
    list = enif_make_list(env, 0);
    for(j=c1-1;j>=0;j--){
        for (i=r1-1;i>=0;i--){
            head = enif_make_double(env,(double)c[IDX2C(i,j,r1)]);
            list = enif_make_list_cell(env,head,list);
        }
    }

    cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

    free(a);
    free(b);
    free(c);
    return list;
}


static ERL_NIF_TERM
transpose1(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    int r,c,i,j;
    float* a = 0;
    double d;
    ERL_NIF_TERM head, list;

    enif_get_int(env, argv[0], &r);
    enif_get_int(env, argv[1], &c);
    
    // Memory Allocation
    a = (float *)malloc (r * c * sizeof (*a));

    list = argv[2]; /* matrix1 */
    for(j=0;j<c;j++){
        for(i=0;i<r;i++){
            enif_get_list_cell(env,list ,&head, &list);
            enif_get_double(env,head,&d);
            a[IDX2C(i,j,r)] = (float)d;
        }
    }
    // Set transposed matrix to list 
    list = enif_make_list(env, 0);
    for(i=r-1;i>=0;i--){
        for (j=c-1;j>=0;j--){
            head = enif_make_double(env,(double)a[IDX2C(i,j,r)]);
            list = enif_make_list_cell(env,head,list);
        }
    }
    free(a);
    return(list);
}

static ERL_NIF_TERM
ident1(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    int r,i,j;
    ERL_NIF_TERM head, list;

    enif_get_int(env, argv[0], &r);
    
    
    // Set ident matrix to list 
    list = enif_make_list(env, 0);
    for(i=r-1;i>=0;i--){
        for (j=r-1;j>=0;j--){

            if(i == j) 
                head = enif_make_double(env,1.0);
            else
                head = enif_make_double(env,0.0);

            list = enif_make_list_cell(env,head,list);
        }
    }
    return(list);
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
    int r1, c1, i, j, n;
    double d;
    float* a = 0;
    float* b = 0;
    float *dev_a, *dev_b;

    ERL_NIF_TERM head, list;

    enif_get_int(env, argv[0], &r1);
    enif_get_int(env, argv[1], &c1);
    n = r1*c1;

    // Memory Allocation
    a = (float *)malloc (r1 * c1 * sizeof (*a));
    b = (float *)malloc (r1 * c1 * sizeof (*b));
    
	// Allocate for GPU
	cudaMalloc((void**)&dev_a, r1*c1 * sizeof(float));
	cudaMalloc((void**)&dev_b, r1*c1 * sizeof(float));


    // Set matrix data to cublas 
    list = argv[2]; /* matrix1 */
    for(j=0;j<c1;j++){
        for(i=0;i<r1;i++){
            enif_get_list_cell(env, list, &head, &list);
            enif_get_double(env,head,&d);
            a[IDX2C(i,j,r1)] = (float)d;
        }
    }

    
    // copy from host a,b to GPU dev_a, dev_b
	cudaMemcpy(dev_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

	sigmoid_kernel << <128, 128 >> >(dev_a, dev_b , n);

	// copy to host c from GPU dev_c
	cudaMemcpy(b, dev_b, n * sizeof(float), cudaMemcpyDeviceToHost);



    // Set matrix After kernel
    list = enif_make_list(env, 0);
    for(j=c1-1;j>=0;j--){
        for (i=r1-1;i>=0;i--){
            head = enif_make_double(env,(double)b[IDX2C(i,j,r1)]);
            list = enif_make_list_cell(env,head,list);
        }
    }

    cudaFree(dev_a);
	cudaFree(dev_b);

    free(a);
    free(b);
    return list;
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
    int r1, c1, i, j, n;
    double d;
    float* a = 0;
    float* b = 0;
    float *dev_a, *dev_b;

    ERL_NIF_TERM head, list;

    enif_get_int(env, argv[0], &r1);
    enif_get_int(env, argv[1], &c1);
    n = r1*c1;

    // Memory Allocation
    a = (float *)malloc (r1 * c1 * sizeof (*a));
    b = (float *)malloc (r1 * c1 * sizeof (*b));
    
	// Allocate for GPU
	cudaMalloc((void**)&dev_a, r1*c1 * sizeof(float));
	cudaMalloc((void**)&dev_b, r1*c1 * sizeof(float));


    // Set matrix data to cublas 
    list = argv[2]; /* matrix1 */
    for(j=0;j<c1;j++){
        for(i=0;i<r1;i++){
            enif_get_list_cell(env, list, &head, &list);
            enif_get_double(env,head,&d);
            a[IDX2C(i,j,r1)] = (float)d;
        }
    }

    
    // copy from host a,b to GPU dev_a, dev_b
	cudaMemcpy(dev_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

	tanh_kernel << <128, 128 >> >(dev_a, dev_b , n);

	// copy to host c from GPU dev_c
	cudaMemcpy(b, dev_b, n * sizeof(float), cudaMemcpyDeviceToHost);



    // Set matrix After kernel
    list = enif_make_list(env, 0);
    for(j=c1-1;j>=0;j--){
        for (i=r1-1;i>=0;i--){
            head = enif_make_double(env,(double)b[IDX2C(i,j,r1)]);
            list = enif_make_list_cell(env,head,list);
        }
    }

    cudaFree(dev_a);
	cudaFree(dev_b);

    free(a);
    free(b);
    return list;
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
    int r1, c1, i, j, n;
    double d;
    float* a = 0;
    float* b = 0;
    float *dev_a, *dev_b;

    ERL_NIF_TERM head, list;

    enif_get_int(env, argv[0], &r1);
    enif_get_int(env, argv[1], &c1);
    n = r1*c1;

    // Memory Allocation
    a = (float *)malloc (r1 * c1 * sizeof (*a));
    b = (float *)malloc (r1 * c1 * sizeof (*b));
    
	// Allocate for GPU
	cudaMalloc((void**)&dev_a, r1*c1 * sizeof(float));
	cudaMalloc((void**)&dev_b, r1*c1 * sizeof(float));


    // Set matrix data to cublas 
    list = argv[2]; /* matrix1 */
    for(j=0;j<c1;j++){
        for(i=0;i<r1;i++){
            enif_get_list_cell(env, list, &head, &list);
            enif_get_double(env,head,&d);
            a[IDX2C(i,j,r1)] = (float)d;
        }
    }

    
    // copy from host a,b to GPU dev_a, dev_b
	cudaMemcpy(dev_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

	relu_kernel << <128, 128 >> >(dev_a, dev_b , n);

	// copy to host c from GPU dev_c
	cudaMemcpy(b, dev_b, n * sizeof(float), cudaMemcpyDeviceToHost);



    // Set matrix After kernel
    list = enif_make_list(env, 0);
    for(j=c1-1;j>=0;j--){
        for (i=r1-1;i>=0;i--){
            head = enif_make_double(env,(double)b[IDX2C(i,j,r1)]);
            list = enif_make_list_cell(env,head,list);
        }
    }

    cudaFree(dev_a);
	cudaFree(dev_b);

    free(a);
    free(b);
    return list;
}



static ERL_NIF_TERM
activate_softmax(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    int r1, c1, i, j, k;
    double d, max, sum;
    float* a = 0;
    float* b = 0;

    ERL_NIF_TERM head, list;

    enif_get_int(env, argv[0], &r1);
    enif_get_int(env, argv[1], &c1);
    
    // Memory Allocation
    a = (float *)malloc (r1 * c1 * sizeof (*a));
    b = (float *)malloc (r1 * c1 * sizeof (*b));
    
	
    // Set matrix data to cublas 
    list = argv[2]; /* matrix1 */
    for(j=0;j<c1;j++){
        for(i=0;i<r1;i++){
            enif_get_list_cell(env, list, &head, &list);
            enif_get_double(env,head,&d);
            a[IDX2C(i,j,r1)] = (float)d;
        }
    }

    
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
    
    // Set matrix 
    list = enif_make_list(env, 0);
    for(j=c1-1;j>=0;j--){
        for (i=r1-1;i>=0;i--){
            head = enif_make_double(env,(double)b[IDX2C(i,j,r1)]);
            list = enif_make_list_cell(env,head,list);
        }
    }

    
    free(a);
    free(b);
    return list;
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
    int r1, c1, i, j, n;
    double d;
    float* a = 0;
    float* b = 0;
    float* c = 0;
    float *dev_a, *dev_b, *dev_c;

    ERL_NIF_TERM head, list;

    enif_get_int(env, argv[0], &r1);
    enif_get_int(env, argv[1], &c1);
    n = r1*c1;

    // Memory Allocation
    a = (float *)malloc (r1 * c1 * sizeof (*a));
    b = (float *)malloc (r1 * c1 * sizeof (*b));
    c = (float *)malloc (r1 * c1 * sizeof (*c));
    
	// Allocate for GPU
	cudaMalloc((void**)&dev_a, r1*c1 * sizeof(float));
	cudaMalloc((void**)&dev_b, r1*c1 * sizeof(float));
    cudaMalloc((void**)&dev_c, r1*c1 * sizeof(float));

    // Set matrix data to cublas 
    list = argv[2]; /* matrix1 */
    for(j=0;j<c1;j++){
        for(i=0;i<r1;i++){
            enif_get_list_cell(env, list, &head, &list);
            enif_get_double(env,head,&d);
            a[IDX2C(i,j,r1)] = (float)d;
        }
    }

    list = argv[3]; /* matrix2 */
    for(j=0;j<c1;j++){
        for(i=0;i<r1;i++){
            enif_get_list_cell(env, list, &head, &list);
            enif_get_double(env,head,&d);
            b[IDX2C(i,j,r1)] = (float)d;
        }
    }


    // copy from host a,b to GPU dev_a, dev_b
	cudaMemcpy(dev_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, c, n * sizeof(float), cudaMemcpyHostToDevice);

	differ_sigmoid_kernel << <128, 128 >> >(dev_a, dev_b ,dev_c, n);

	// copy to host c from GPU dev_c
	cudaMemcpy(c, dev_c, n * sizeof(float), cudaMemcpyDeviceToHost);



    // Set matrix After kernel
    list = enif_make_list(env, 0);
    for(j=c1-1;j>=0;j--){
        for (i=r1-1;i>=0;i--){
            head = enif_make_double(env,(double)c[IDX2C(i,j,r1)]);
            list = enif_make_list_cell(env,head,list);
        }
    }

    cudaFree(dev_a);
	cudaFree(dev_b);
    cudaFree(dev_c);

    free(a);
    free(b);
    free(c);
    return list;
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
    int r1, c1, i, j, n;
    double d;
    float* a = 0;
    float* b = 0;
    float* c = 0;
    float *dev_a, *dev_b, *dev_c;

    ERL_NIF_TERM head, list;

    enif_get_int(env, argv[0], &r1);
    enif_get_int(env, argv[1], &c1);
    n = r1*c1;

    // Memory Allocation
    a = (float *)malloc (r1 * c1 * sizeof (*a));
    b = (float *)malloc (r1 * c1 * sizeof (*b));
    c = (float *)malloc (r1 * c1 * sizeof (*c));
    
	// Allocate for GPU
	cudaMalloc((void**)&dev_a, r1*c1 * sizeof(float));
	cudaMalloc((void**)&dev_b, r1*c1 * sizeof(float));
    cudaMalloc((void**)&dev_c, r1*c1 * sizeof(float));

    // Set matrix data to cublas 
    list = argv[2]; /* matrix1 */
    for(j=0;j<c1;j++){
        for(i=0;i<r1;i++){
            enif_get_list_cell(env, list, &head, &list);
            enif_get_double(env,head,&d);
            a[IDX2C(i,j,r1)] = (float)d;
        }
    }

    list = argv[3]; /* matrix2 */
    for(j=0;j<c1;j++){
        for(i=0;i<r1;i++){
            enif_get_list_cell(env, list, &head, &list);
            enif_get_double(env,head,&d);
            b[IDX2C(i,j,r1)] = (float)d;
        }
    }


    // copy from host a,b to GPU dev_a, dev_b
	cudaMemcpy(dev_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, c, n * sizeof(float), cudaMemcpyHostToDevice);

	differ_tanh_kernel << <128, 128 >> >(dev_a, dev_b ,dev_c, n);

	// copy to host c from GPU dev_c
	cudaMemcpy(c, dev_c, n * sizeof(float), cudaMemcpyDeviceToHost);



    // Set matrix After kernel
    list = enif_make_list(env, 0);
    for(j=c1-1;j>=0;j--){
        for (i=r1-1;i>=0;i--){
            head = enif_make_double(env,(double)c[IDX2C(i,j,r1)]);
            list = enif_make_list_cell(env,head,list);
        }
    }

    cudaFree(dev_a);
	cudaFree(dev_b);
    cudaFree(dev_c);

    free(a);
    free(b);
    free(c);
    return list;
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
    int r1, c1, i, j, n;
    double d;
    float* a = 0;
    float* b = 0;
    float* c = 0;
    float *dev_a, *dev_b, *dev_c;

    ERL_NIF_TERM head, list;

    enif_get_int(env, argv[0], &r1);
    enif_get_int(env, argv[1], &c1);
    n = r1*c1;

    // Memory Allocation
    a = (float *)malloc (r1 * c1 * sizeof (*a));
    b = (float *)malloc (r1 * c1 * sizeof (*b));
    c = (float *)malloc (r1 * c1 * sizeof (*c));
    
	// Allocate for GPU
	cudaMalloc((void**)&dev_a, r1*c1 * sizeof(float));
	cudaMalloc((void**)&dev_b, r1*c1 * sizeof(float));
    cudaMalloc((void**)&dev_c, r1*c1 * sizeof(float));

    // Set matrix data to cublas 
    list = argv[2]; /* matrix1 */
    for(j=0;j<c1;j++){
        for(i=0;i<r1;i++){
            enif_get_list_cell(env, list, &head, &list);
            enif_get_double(env,head,&d);
            a[IDX2C(i,j,r1)] = (float)d;
        }
    }

    list = argv[3]; /* matrix2 */
    for(j=0;j<c1;j++){
        for(i=0;i<r1;i++){
            enif_get_list_cell(env, list, &head, &list);
            enif_get_double(env,head,&d);
            b[IDX2C(i,j,r1)] = (float)d;
        }
    }


    // copy from host a,b to GPU dev_a, dev_b
	cudaMemcpy(dev_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, c, n * sizeof(float), cudaMemcpyHostToDevice);

	differ_relu_kernel << <128, 128 >> >(dev_a, dev_b ,dev_c, n);

	// copy to host c from GPU dev_c
	cudaMemcpy(c, dev_c, n * sizeof(float), cudaMemcpyDeviceToHost);



    // Set matrix After kernel
    list = enif_make_list(env, 0);
    for(j=c1-1;j>=0;j--){
        for (i=r1-1;i>=0;i--){
            head = enif_make_double(env,(double)c[IDX2C(i,j,r1)]);
            list = enif_make_list_cell(env,head,list);
        }
    }

    cudaFree(dev_a);
	cudaFree(dev_b);
    cudaFree(dev_c);

    free(a);
    free(b);
    free(c);
    return list;
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
    int r1, c1, i, j, n;
    double s,d;
    float* a = 0;
    float* b = 0;
    float *dev_a, *dev_b;

    ERL_NIF_TERM head, list;

    enif_get_double(env, argv[0], &s);
    enif_get_int(env, argv[1], &r1);
    enif_get_int(env, argv[2], &c1);
    n = r1*c1;

    // Memory Allocation
    a = (float *)malloc (r1 * c1 * sizeof (*a));
    b = (float *)malloc (r1 * c1 * sizeof (*b));
    
	// Allocate for GPU
	cudaMalloc((void**)&dev_a, r1*c1 * sizeof(float));
	cudaMalloc((void**)&dev_b, r1*c1 * sizeof(float));


    // Set matrix data to cublas 
    list = argv[3]; /* matrix1 */
    for(j=0;j<c1;j++){
        for(i=0;i<r1;i++){
            enif_get_list_cell(env, list, &head, &list);
            enif_get_double(env,head,&d);
            a[IDX2C(i,j,r1)] = (float)d;
        }
    }

    
    // copy from host a,b to GPU dev_a, dev_b
	cudaMemcpy(dev_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

	smult_kernel << <128, 128 >> >((float)s, dev_a, dev_b , n);

	// copy to host c from GPU dev_c
	cudaMemcpy(b, dev_b, n * sizeof(float), cudaMemcpyDeviceToHost);



    // Set matrix After kernel
    list = enif_make_list(env, 0);
    for(j=c1-1;j>=0;j--){
        for (i=r1-1;i>=0;i--){
            head = enif_make_double(env,(double)b[IDX2C(i,j,r1)]);
            list = enif_make_list_cell(env,head,list);
        }
    }

    cudaFree(dev_a);
	cudaFree(dev_b);

    free(a);
    free(b);
    return list;
}

static ERL_NIF_TERM
trace1(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    int r1, c1, i, j;
    double t,d;
    float* a = 0;
    ERL_NIF_TERM head, list;

    enif_get_int(env, argv[0], &r1);
    enif_get_int(env, argv[1], &c1);
    
    // Memory Allocation
    a = (float *)malloc (r1 * c1 * sizeof (*a));
    
    // Set matrix data to a 
    list = argv[2]; /* matrix1 */
    for(j=0;j<c1;j++){
        for(i=0;i<r1;i++){
            enif_get_list_cell(env, list, &head, &list);
            enif_get_double(env,head,&d);
            a[IDX2C(i,j,r1)] = (float)d;
        }
    }

    

    // Culcurate trace of matrix
    t = 0;
    for(j=c1-1;j>=0;j--){
        for (i=r1-1;i>=0;i--){
            if(i==j)
                t = t + (double)a[IDX2C(i,j,r1)];
        }
    }

    free(a);
    return enif_make_double(env,t);
}

static ERL_NIF_TERM
mean_square(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    int r1, c1, i, j;
    double s,d;
    float* a = 0;
    float* b = 0;

    ERL_NIF_TERM head, list;

    enif_get_int(env, argv[0], &r1);
    enif_get_int(env, argv[1], &c1);
    
    // Memory Allocation
    a = (float *)malloc (r1 * c1 * sizeof (*a));
    b = (float *)malloc (r1 * c1 * sizeof (*b));
    
    
    // Set matrix data to cuda 
    list = argv[2]; /* matrix1 */
    for(j=0;j<c1;j++){
        for(i=0;i<r1;i++){
            enif_get_list_cell(env, list, &head, &list);
            enif_get_double(env,head,&d);
            a[IDX2C(i,j,r1)] = (float)d;
        }
    }
    


    list = argv[3]; /* matrix2 */
    for (j=0;j<c1;j++){
        for (i=0;i<r1;i++){
            enif_get_list_cell(env, list, &head, &list);
            enif_get_double(env,head,&d);
            b[IDX2C(i,j,r1)] = (float)d;
        }
    }
    
   

    // Set matrix After kernel
    list = enif_make_list(env, 0);
    for(i=r1-1;i>=0;i--){
        s = 0.0;
        for (j=c1-1;j>=0;j--){
            d = (double)a[IDX2C(i,j,r1)] -  (double)b[IDX2C(i,j,r1)];
            s = s + d*d;            
        }
        head = enif_make_double(env,s/2);
        list = enif_make_list_cell(env,head,list);
    }

    free(a);
    free(b);
    return list;
}

static ERL_NIF_TERM
cross_entropy(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    int r1, c1, i, j;
    double s,d,delta;
    float* a = 0;
    float* b = 0;

    ERL_NIF_TERM head, list;

    enif_get_int(env, argv[0], &r1);
    enif_get_int(env, argv[1], &c1);
    
    // Memory Allocation
    a = (float *)malloc (r1 * c1 * sizeof (*a));
    b = (float *)malloc (r1 * c1 * sizeof (*b));
    
    
    // Set matrix data to cuda 
    list = argv[2]; /* matrix1 */
    for(j=0;j<c1;j++){
        for(i=0;i<r1;i++){
            enif_get_list_cell(env, list, &head, &list);
            enif_get_double(env,head,&d);
            a[IDX2C(i,j,r1)] = (float)d;
        }
    }
    


    list = argv[3]; /* matrix2 */
    for (j=0;j<c1;j++){
        for (i=0;i<r1;i++){
            enif_get_list_cell(env, list, &head, &list);
            enif_get_double(env,head,&d);
            b[IDX2C(i,j,r1)] = (float)d;
        }
    }
    
   

    // Set matrix After kernel
    list = enif_make_list(env, 0);
    delta = 1e-7;
    for(i=r1-1;i>=0;i--){
        s = 0.0;
        for (j=c1-1;j>=0;j--){
            d = (double)a[IDX2C(i,j,r1)];
            s = s + (double)b[IDX2C(i,j,r1)] * log(d+delta);             
        }
        head = enif_make_double(env,-1.0*s);
        list = enif_make_list_cell(env,head,list);
    }

    free(a);
    free(b);
    return list;
}


// define the array of ErlNifFunc
static ErlNifFunc nif_funcs[] = {
  // {erl_function_name, erl_function_arity, c_function}
  {"mult1", 6, mult},
  {"new1", 2, new1},
  {"rand1", 1, rand1},
  {"add1", 4, add1},
  {"sub1", 4, sub1},
  {"emult1", 4, emult1},
  {"badd1", 4 ,badd1},
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
  {"cross_entropy", 4, cross_entropy}
};

ERL_NIF_INIT(Elixir.Cumatrix, nif_funcs, NULL, NULL, NULL, NULL)


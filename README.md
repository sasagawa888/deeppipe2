# DeepPipe2
DeepPipe2 is a library of Deep-Learning by Elixir using GPU. It is fast because it uses CUDA/cuBLAS. 

Currently, I am improving to use CNN.

# getting started
DeepPipe2 requires GPU. Nvidia's GeForce is easy. The game PC turns into an AI experiment tool.

If you have not yet installed CUDA, install it from the Nvidia website.
https://developer.nvidia.com/cuda-downloads

Linux is recommended for OS. DeepPipe2 does not work on Windows.
After installing CUDA, make a clone of DeepPipe2.
git clone https://github.com/sasagawa888/deeppipe2.git

Change to the deeppipe2 directory with the CD command. And enter make from terminal. Installation is completed. Start with iex -S mix.MNIST data and sample code are included. Enter Test.sgd (100,100).

Network descriptions are Elixir-like pipeline operators. See the test.ex file. The random number given to the weight matrix and bias affects learning. The learning rate also has an effect. The default for each is 0.1. You can change it with w(300,100,0.2,0.5) in defnetwork. In this example, the multiple to the random number is 0.2 and the learning rate is 0.5. It is important to find these parameters in Deep-Learning.

Please enjoy.



## install
require CUDA.

Make clone or download.

on terminal 

```
$make
$iex -S mix

```

## example
MNIST 100 mini batch size, 100 epocs

```
iex(1)> Test.sgd(100,100)
preparing data
ready
-2.285336971282959
-2.1092705726623535
-1.9825358390808105
...
-0.2589882016181946
-0.22275760769844055
-0.2696773409843445
accuracy rate = 0.841
end
:ok
iex(2)> 

```

## time

```
iex(2)> require(Time)
Time
iex(3)> Time.time(Test.sgd(100,100))
preparing data
ready
-2.0867862701416016
-2.009648084640503
-1.9592150449752808
...
-0.19911997020244598
-0.3847019076347351
accuracy rate = 0.84
end
"time: 17937936 micro second"
"-------------"
:ok

```

## confirmation
I confirmed following OS and CUDA

Linux Mint 18.1 “Sarah” MATE


```
nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2019 NVIDIA Corporation
Built on Wed_Oct_23_19:24:38_PDT_2019
Cuda compilation tools, release 10.2, V10.2.89
```


## network example
see test.ex

```
# for DNN test
  defnetwork init_network1(_x) do
    _x |> w(784,300) |> b(300) |> relu
    |> w(300,100) |> b(100) |> relu
    |> w(100,10) |> b(10) |> softmax
  end

```

## specification:

### data structure
#### network
[{:weight,w,ir,lr,v},{:bias,b,ir,lr},{:function,name}, ...]
##### weight
{:weight,w,ir,lr,v} w is matrix, ir is rate for initial random number,lr is learning rate, v is for momentum,adagrad,adam
##### bias
{:bias,b,ir,lr,v} b is row vector
##### function
{:function,name} name is function name within sigmoid tanh relu

### module macros
defnetwork is macros to describe network
argument must have under bar to avoid warning message

##### w(m,n)
weight matrix size(m,n). elements are Gaussian distribution random float
##### w(m,n,,ir,lr)
ir is rate for random number. (default is 0.1)
lr is learning rate (default is 0.1)

##### b(n)
bias row_vector size(n). elements are all zero

#### b(n,ir,lr)
ir is rate for random number. (default is 0.1)
lr is learning rate (default is 0.1)

##### function
sigmoid,tanh,relu,softmax

## module Deeppipe
- forward/3

```
return all middle data
1st arg is input data matrix
2nd arg is network list
3rd arg is generated middle layer result
```

- gradient/3

```
gradient with backpropagation
1st arg is input data matrix
2nd arg is network list
3rd arg is train matrix
```

- learn/2 learn/3

```
learning/2 
1st arg is old network list
2nd arg is network with gradient
generate new network with leared weight and bias
update method is sgd

learning/3
added update method to 3rd arg
update method is momentam, adagrad

```




## module cuMatrix
Caution, each element of matrix  must be float number.

- Cumatrix.new(r,c) 
generate matrix with given row col size. Each elements are zero.

- Cumatrix.new(r,c,val)
generate matrix with given row col size. Each elements are val.

- Cumatrix.new(list)
generate matrix with given list. e.g. [[1,2],[3,4]].

- Cumatrix.rand(r,c)
generate matrix with random (Box-muller).

- Cumatrix.add(mt1,mt2)
generate matrix mt1+mt2. if mt1 or mt2 is row vector, expand size to matrix. This function is for bias add in DL.

- Cumatrix.sub(mt1,mt2)
generate matrix mt1-mt2.

- Cumatrix.mult(mt1,mt2)
generate matrix  mt1*mt2 with cuBLAS. 
if mt1 or mt2  is float number. generate matrix that each element is s*elt(x,y)

- Cumatrix.emult(mt1,mt2)
generate Hadamard matrix.

- Cumatrix.transpose(mt)
generate transposed matrix

- Cumatrix.ident(n) 
generate ident matrix of size n.

- Cumatrix.activate(mt,fun)
apply fun to mt. fun is :sigmoid, :tanh, :relu :softmax

- Cumatrix.size(mt)
return tuple {rowsize,colsize}

- Cumatrix.sum(mt)
return sum of elements

- Cumatrix.to_list(mt)
return list that transformed from matrix

- Cumatrix.trace(mt)
return float number. It is trace of matrix.

- Cumatrix.print(mt)
print matrix mt

- Cumatrix.elt(mt,r,c) 
pick up element of mt(r,c) index is one base

- Cumatrix.set(mt,x,y,val)
elt(mt,x,y) := val. 

- Cumatrix.average(mt)
caluculate average of row-vector and generate row-vector that each element is average.
For Deep-Learning.  

- Cumatrix.loss(mt1,mt2,fun)
generate float that is average of loss. fun is :square or :cross.
:square means mean_square function, and :cross means cross_entropy function.
mt1 is calculated data matrix , mt2 is teacher data matrix.
each data is row vector.

- Cumatrix.diff(mt1,mt2,fun)
for each element multiply differntial of mt2 and mt1. fun is :sigmoid :tanh, :relu.

- Cumatrix.momentum(mt1,mt2,lr)
for each element 0.5 * mt1(x,y) - lr * mt2(x,y). for learn/3 in DeepPipe

- Cumatrix.adagrad(mt1,mt2,h,lr)
for each element mt1(x,y) - lr * (1 / adagrad_sqrt(h)) * mt2(x,y). 
when h != 0 adagrad_sqrt(x) = sqrt(x).
when h == 0 adagrad_sqrt(x) = 1.

- Cumatrix.accuracy(mt1,ls) 
return accuracy rate as float number.
mt1 is set of row-vector.Each row-vector is onehot.
ls is list each element is label integer number.

```
e.g.

iex(1)> a = Cumatrix.new([[0.0,0.0,1.0],[0.0,0.1,0.3]])
{2, 3,
 <<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 205, 204, 204, 61, 0, 0, 128, 63, 154,
   153, 153, 62>>}
iex(3)> Cumatrix.accuracy(a,[2,2])
1.0
iex(4)> Cumatrix.accuracy(a,[2,1])
0.5
iex(5)> 

```

## CNN
Now under constrction.

data structure is 4 dimensions tensor (N,C,H,W).
N is mini batch size.
C is channel.
H is hight of image.
W is width of image.

- Cumatrix.rand(n,c,h,w)
generate 4 dimensions data.

- Cumatrix.new(ls)
ls is list that express 4 dimension data

- Cumatrix.to_list(tensor)

- Cumatrix.pooling(x)
pooling with size n


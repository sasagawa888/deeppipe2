# DeepPipe2
DeepPipe2 is a library of Deep-Learning by Elixir using GPU. It is fast because it uses CUDA/cuBLAS. 

Currently, I am improving to use CNN.

# getting started
DeepPipe2 requires GPU. Nvidia's GeForce is easy. The game PC turns into an AI experiment tool.

```
(1)install CUDA
If you have not yet installed CUDA, install it from the Nvidia website.
https://developer.nvidia.com/cuda-downloads

(2)make clone 
Linux is recommended for OS. DeepPipe2 does not work on Windows.
After installing CUDA, make a clone of DeepPipe2.
git clone https://github.com/sasagawa888/deeppipe2.git

(3)invoke Deeppipe2
Change to the deeppipe2 directory with the CD command. 
Start with iex -S mix. mix compiles CUDA code automatically.

(4)dwonload dataset
iex(1)> Deeppipe.download(:mnist)
Deeppipe will prepare MNIST dataset 
Enter on iex.
iex(2)> Test.sgd(100,100).


When you test other dataset use Deeppipe.download(x)
x is dataset name atom. 
:fashion  (Fashion-MNIST)
:cifar10  (CIFAR10)
:iris     (IRIS)

Network descriptions are Elixir-like pipeline operators. 
See the test.ex file. The random number given to the weight matrix and bias affects learning.
The learning rate also has an effect. The default for each is 0.1.
You can change it with w(300,100,0.2,0.5) in defnetwork.
In this example, the multiple to the random number is 0.2 and the learning rate is 0.5.
It is important to find these parameters in Deep-Learning.
```

Please enjoy.

## install
require CUDA.

Make clone or download.

on terminal 

```
$iex -S mix

```

## example
MNIST 100 mini batch size, 100 times repeat.

```
iex(1)> Test.sgd(100,100)
preparing data
learning start
100 2.204291343688965
99 2.0511350631713867
98 1.851050615310669
97 1.7592202425003052
96 1.7395706176757813
95 1.6045866012573242
94 1.5952904224395752
93 1.4877655506134033
...
7 0.4388183653354645
6 0.49198096990585327
5 0.5398643016815186
4 0.43334200978279114
3 0.47159287333488464
2 0.37558087706565857
1 0.37841811776161194
learning end
accuracy rate = 0.8495
"time: 4.006799 second"
"-------------"
:ok
iex(2)> 

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
update method is sgd, momentam, adagrad

```

- train/9

```
1st arg network
2nd arg train image list
3rd arg train onehot list
4th arg test image list
5th arg test labeel list
6th arg loss function (;cross or :squre)
7th arg minibatch size
8th arg lewarning method
9th arg repeat number

```

## CNN
Now testing.

```
# for CNN test for Fashion-MNIST
  defnetwork init_network4(_x) do
    _x
    |> f(5, 5, 1, 12, {1,1}, 1, 0.5, 0.0001)
    |> pooling(2,2)
    |> f(3, 3, 12, 12, {1,1}, 1, 0.5, 0.0001)
    |> f(2, 2, 12, 12, {1,1}, 1, 0.5, 0.0001)
    |> pooling(2,2)
    |> f(3, 3, 12, 12, {1,1}, 0, 0.5, 0.0001)
    |> relu
    |> full
    |> w(300, 10, 0.1, 0.001)
    |> softmax
  end

```


## Hardware 
recommended  memory 8GB or moreover.

recommended GTX960 GPU or moreover.

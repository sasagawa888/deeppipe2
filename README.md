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

(4)download dataset
iex(1)> Deeppipe.download(:mnist)
Deeppipe will prepare MNIST dataset 

(5)learning
iex(2)> MNIST.sgd(100,10).


When you test other dataset use Deeppipe.download(x)
x is dataset name atom. 
:fashion  (Fashion-MNIST)
:cifar10  (CIFAR10)
:iris     (IRIS)

Network descriptions are Elixir-like pipeline operators. 
See the mnist.ex file. The random number given to the weight matrix and bias affects learning.
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
MNIST 100 mini batch size, 2 epochs.

```
# for DNN test
  defnetwork init_network1(_x) do
    _x 
    |> w(784,300) 
    |> b(300) 
    |> relu
    |> w(300,100) 
    |> b(100) 
    |> relu
    |> w(100,10) 
    |> b(10) 
    |> softmax
  end

iex(1)> MNIST.sgd(100,2)
preparing data

epoch 1
[##################################################](100%)
loss = 0.19861093163490295

epoch 2
[##################################################](100%)
loss = 0.157775416970253

learning end
accuracy rate = 94.96%
time: 30.223084 second
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
see mnist.ex

```
# for DNN test
  defnetwork init_network1(_x) do
    _x 
    |> w(784,300) 
    |> b(300) 
    |> relu
    |> w(300,100) 
    |> b(100) 
    |> relu
    |> w(100,10) 
    |> b(10) 
    |> softmax
  end
```



## module Deeppipe  main functions
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

- learn/3

```
learning/3 
1st arg is old network list
2nd arg is network with gradient
3rd arg is update method
generate new network with leared weight and bias
update method is sgd, momentam, adagrad

```

- train/9

```
1st arg network
2nd arg train image list
3rd arg train onehot list
4th arg test image list
5th arg test label list
6th arg loss function (:cross or :square)
7th arg minibatch size
8th arg learning method (:sgd :momentum :adagrad)
9th arg epochs number

```

## CNN
Now testing.

```

 for CNN test for MNIST
  defnetwork init_network4(_x) do
    _x
    |> f(3, 3, 1, 6, {1, 1}, 0, 0.1, 0.001)
    |> f(3, 3, 6, 12, {1, 1}, 0, 0.1, 0.001)
    |> pooling(2, 2)
    |> relu
    |> full
    |> w(1728, 10, 0.1, 0.001)
    |> softmax
  end




mini batch size 100, 3 epoch

MNIST.cnn(100,3)
preparing data

epoch 1
[##################################################](100%)
random loss = 0.6510372757911682
accuracy rate = 82.94%

epoch 2
[##################################################](100%)
random loss = 0.4810127913951874
accuracy rate = 85.8%

epoch 3
[##################################################](100%)
random loss = 0.5048774480819702
accuracy rate = 87.07000000000001%
time: 511.20555 second
:ok



```


## Hardware 
recommended  memory 8GB or moreover.

recommended GTX960 GPU or moreover.

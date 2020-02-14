# DeepPipe2
Deep Pipe2 is based on Deep Pipe.
It uses GPU with CUDA/cuBLAS

Now,under construction. I'm improving.

## install
require CUDA

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
 e.g. [{:weight,w,ir,lr,v},{:bias,b,ir,lr},{:function,name}]
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

####b(n,ir,lr)
ir is rate for random number. (default is 0.1)
lr is learning rate (default is 0.1)

##### function
sigmoid,tanh,relu,softmax

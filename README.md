# DeepPipe2
Deep Pipe2 is based on Deep Pipe.
It uses GPU with CUDA/cuBLAS

Now,under construction. I'm improving.

## install
require CUDA

Make clone or dowload.

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
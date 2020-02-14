defmodule Deeppipe do
  alias Cumatrix, as: CM
  
  # for debug
  def stop() do
    raise("stop")
  end


  # forward
  # return all middle data
  # 1st arg is input data matrix
  # 2nd arg is netward list
  # 3rd arg is generated middle layer result
  def forward(_, [], res) do
    res
  end

  def forward(x, [{:weight, w, _, _} | rest], res) do
    x1 = CM.mult(x, w)
    forward(x1, rest, [x1 | res])
  end

  def forward(x, [{:bias, b, _, _} | rest], res) do
    x1 = CM.add(x, b)
    forward(x1, rest, [x1 | res])
  end

  def forward(x, [{:function, name} | rest], res) do
    x1 = CM.activate(x, name)
    forward(x1, rest, [x1 | res])
  end

 
  # gradient with backpropagation
  # 1st arg is input data matrix
  # 2nd arg is network list
  # 3rd arg is teeacher matrix
  def gradient(x, network, t) do
    [x1 | x2] = forward(x, network, [x])
    loss = CM.sub(x1, t)
    network1 = Enum.reverse(network)
    backward(loss, network1, x2, [])
  end

  # backward
  # calculate grad with gackpropagation
  # 1st arg is loss matrix
  # 2nd arg is network list
  # 3rd arg is generated new network with calulated gradient
  # l loss matrix
  # u input data matrix at each layer
  defp backward(_, [], _, res) do
    res
  end

  defp backward(l, [{:function, :softmax} | rest], [_ | us], res) do
    backward(l, rest, us, [{:function, :softmax} | res])
  end

  defp backward(l, [{:function, name} | rest], [u | us], res) do
    l1 = CM.diff(l, u, name)
    backward(l1, rest, us, [{:function, name} | res])
  end

  defp backward(l, [{:bias, _, lr, v} | rest], [_ | us], res) do
    b1 = CM.average(l)
    backward(l, rest, us, [{:bias, b1, lr, v} | res])
  end

  defp backward(l, [{:weight, w, lr, v} | rest], [u | us], res) do
    {n, _} = CM.size(l)
    w1 = CM.mult(CM.transpose(u), l) |> CM.mult(1 / n)
    l1 = CM.mult(l, CM.transpose(w))
    backward(l1, rest, us, [{:weight, w1, lr, v} | res])
  end

  # ------- learning -------
  # learning/2 
  # 1st arg is old network list
  # 2nd arg is network with gradient
  # generate new network with leared weight and bias
  # update method is sgd
  #
  # learning/3 now under construction
  # added update method to 3rd arg
  # update method is momentam, adagrad, adam

  # --------sgd----------
  def learning([], _) do
    []
  end

  def learning([{:weight, w, lr, v} | rest], [{:weight, w1, _, _} | rest1]) do
    w2 = CM.sub(w,CM.mult(w1,lr))
    [{:weight, w2, lr, v} | learning(rest, rest1)]
  end

  def learning([{:bias, w, lr, v} | rest], [{:bias, w1, _, _} | rest1]) do
    w2 = CM.sub(w,CM.mult(w1,lr))
    [{:bias, w2, lr, v} | learning(rest, rest1)]
  end

  def learning([network | rest], [_ | rest1]) do
    [network | learning(rest, rest1)]
  end

  # average loss (scalar)
  def loss(y, t, :cross) do
    CM.loss(y,t,:cross) |> CM.average() |> CM.elt(1,1)
  end

  def loss(y, t, :square) do
    CM.loss(y,t,:square) |> CM.average() |> CM.elt(1,1)
  end

  def loss(y, t) do
     CM.loss(y,t,:square) |> CM.average() |> CM.elt(1,1) 
  end


  # print predict of test data
  def accuracy(image, network, label) do
    forward(image, network, []) |> hd() |> CM.to_list() |> score(label, 0)
  end

  
  defp score([], [], correct) do
    correct
  end

  defp score([x | xs], [l | ls], correct) do
    if MNIST.onehot_to_num(x) == l do
      score(xs, ls, correct + 1)
    else
      score(xs, ls, correct)
    end
  end

  # select random data from image data and train data 
  # size of m. range from 0 to n
  # and generate tuple of two matrix
  def random_select(image, train, m, n) do
    random_select1(image, train, [], [], m, n)
  end

  defp random_select1(_, _, res1, res2, 0, _) do
    mt1 = CM.new(res1)
    mt2 = CM.new(res2)
    {mt1, mt2}
  end

  defp random_select1(image, train, res1, res2, m, n) do
    i = :rand.uniform(n-1)
    image1 = Enum.at(image, i)
    train1 = Enum.at(train, i)
    random_select1(image, train, [image1 | res1], [train1 | res2], m - 1, n)
  end

  # save/load to file
  def save(file, network) do
    network1 = save1(network)
    File.write(file, inspect(network1, limit: :infinity))
  end

  def save1([]) do
    []
  end

  def save1([{:weight, w, lr, v} | rest]) do
    [{:weight, CM.to_list(w), lr, v} | save1(rest)]
  end

  def save1([{:bias, w, lr, v} | rest]) do
    [{:bias, CM.to_list(w), lr, v} | save1(rest)]
  end

  def save1([{:function, name} | rest]) do
    [{:function, name} | save1(rest)]
  end

  def save1([network | rest]) do
    [network | save1(rest)]
  end

  def load(file) do
    Code.eval_file(file) |> elem(0) |> load1
  end

  def load1([]) do
    []
  end

  def load1([{:weight, w, lr, v} | rest]) do
    [{:weight, CM.new(w), lr, v} | load1(rest)]
  end

  def load1([{:bias, w, lr, v} | rest]) do
    [{:bias, CM.new(w), lr, v} | load1(rest)]
  end

  def load1([{:function, name} | rest]) do
    [{:function, name} | load1(rest)]
  end

  def load1([network | rest]) do
    [network | load1(rest)]
  end

  # basic I/O
  def print(x) do
    cond do 
      is_number(x) || is_atom(x) -> :io.write(x)
      is_matrix(x) -> CM.print(x)
      true ->  print1(x)
      IO.puts("")
    end
  end

  def print1([]) do true end
  def print1([x|xs]) do 
    print2(x)
    print1(xs)
  end 

  

  def print2({:weight, w, _, _}) do
    CM.print(w)
  end

  def print2({:bias, w, _, _}) do
    CM.print(w)
  end

  def print2({:function, name}) do
    :io.write(name)
  end

  def print2(x) do
    if is_matrix(x) do 
      CM.print(x)
    else 
      :io.write(x)
    end 
  end

  def is_matrix({r,c,_}) do
    if is_integer(r) && is_integer(c) do 
      true
    else 
      false
    end 
  end 
  def is_matrix(_) do false end 
  
  def newline() do
    IO.puts("")
  end


end

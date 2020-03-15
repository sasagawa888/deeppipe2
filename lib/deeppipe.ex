defmodule Deeppipe do
  alias Cumatrix, as: CM

  # for debug
  def stop() do
    raise("stop")
  end

  # forward
  # return all middle data
  # 1st arg is input data matrix
  # 2nd arg is network list
  # 3rd arg is generated middle layer result
  def forward(_, [], res) do
    res
  end

  def forward(x, [{:weight, w, _, _, _} | rest], res) do
    x1 = CM.mult(x, w)
    forward(x1, rest, [x1 | res])
  end

  def forward(x, [{:bias, b, _, _, _} | rest], res) do
    x1 = CM.add(x, b)
    forward(x1, rest, [x1 | res])
  end

  def forward(x, [{:function, name} | rest], res) do
    x1 = CM.activate(x, name)
    forward(x1, rest, [x1 | res])
  end

  def forward(x, [{:filter, w, st, pad, _, _} | rest], res) do
    x1 = CM.convolute(x, w, st, pad)
    forward(x1, rest, [x1 | res])
  end

  def forward(x, [{:pooling, st} | rest], [_ | res]) do
    [x1, x2] = CM.pooling(x, st)
    forward(x1, rest, [x2 | res])
  end

  def forward(x, [{:full} | rest], res) do
    x1 = CM.full(x)
    forward(x1, rest, [x1 | res])
  end

  # gradient with backpropagation
  # 1st arg is input data matrix
  # 2nd arg is network list
  # 3rd arg is train matrix
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

  defp backward(l, [{:bias, _, ir, lr, v} | rest], [_ | us], res) do
    b1 = CM.average(l)
    backward(l, rest, us, [{:bias, b1, ir, lr, v} | res])
  end

  defp backward(l, [{:weight, w, ir, lr, v} | rest], [u | us], res) do
    {n, _} = CM.size(l)
    w1 = CM.mult(CM.transpose(u), l) |> CM.mult(1 / n)
    l1 = CM.mult(l, CM.transpose(w))
    backward(l1, rest, us, [{:weight, w1, ir, lr, v} | res])
  end

  defp backward(l, [{:filter, w, st, pad, ir, lr, v} | rest], [u | us], res) do
    w1 = CM.gradfilter(u, w, l, st, pad)
    l1 = CM.deconvolute(l, w, st, pad)
    backward(l1, rest, us, [{:filter, w1, st, ir, lr, v} | res])
  end

  defp backward(l, [{:pooling, st} | rest], [u | us], res) do
    l1 = CM.unpooling(u, l, st)
    backward(l1, rest, us, [{:pooling, st} | res])
  end

  defp backward(l, [{:full} | rest], [u | us], res) do
    {_, _, h, w} = CM.size(hd(u))
    l1 = CM.unfull(l, h, w)
    backward(l1, rest, us, [{:full} | res])
  end

  # ------- learning -------
  # learning/2 
  # 1st arg is old network list
  # 2nd arg is network with gradient
  # generate new network with leared weight and bias
  # update method is sgd
  #
  # learning/3
  # added update method to 3rd arg
  # update method is momentam, adagrad

  # --------sgd----------
  def learning([], _) do
    []
  end

  def learning([{:weight, w, ir, lr, v} | rest], [{:weight, w1, _, _, _} | rest1]) do
    w2 = CM.sub(w, CM.mult(w1, lr))
    [{:weight, w2, ir, lr, v} | learning(rest, rest1)]
  end

  def learning([{:bias, w, ir, lr, v} | rest], [{:bias, w1, _, _, _} | rest1]) do
    w2 = CM.sub(w, CM.mult(w1, lr))
    [{:bias, w2, ir, lr, v} | learning(rest, rest1)]
  end

  def learning([{:filter, w, st, pad, ir, lr, v} | rest], [{:filter, w1, _, _, _, _, _} | rest1]) do
    w2 = CM.sub(w, CM.mult(w1, lr))
    [{:filter, w2, st, pad, ir, lr, v} | learning(rest, rest1)]
  end


  def learning([network | rest], [_ | rest1]) do
    [network | learning(rest, rest1)]
  end

  # --------momentum-------------
  def learning([], _, :momentum) do
    []
  end

  def learning([{:weight, w, ir, lr, v} | rest], [{:weight, w1, _, _, _} | rest1], :momentum) do
    v1 = CM.momentum(v, w1, lr)
    [{:weight, CM.add(w, v1), ir, lr, v1} | learning(rest, rest1, :momentum)]
  end

  def learning([{:bias, w, ir, lr, v} | rest], [{:bias, w1, _, _, _} | rest1], :momentum) do
    v1 = CM.momentum(v, w1, lr)
    [{:bias, CM.add(w, v1), ir, lr, v1} | learning(rest, rest1, :momentum)]
  end

  def learning([{:filter, w, st, pad, ir, lr, v} | rest], [{:filter, w1, _, _, _, _, _} | rest1], :momentum) do
    w2 = CM.sub(w, CM.mult(w1, lr))
    [{:filter, w2, st, pad, ir, lr, v} | learning(rest, rest1, :momentum)]
  end

  def learning([network | rest], [_ | rest1], :momentum) do
    [network | learning(rest, rest1, :momentum)]
  end

  # --------AdaGrad--------------
  def learning([], _, :adagrad) do
    []
  end

  def learning([{:weight, w, ir, lr, h} | rest], [{:weight, w1, _, _, _} | rest1], :adagrad) do
    h1 = CM.add(h, CM.emult(w1, w1))
    [{:weight, CM.adagrad(w, w1, h1, lr), ir, lr, h1} | learning(rest, rest1, :adagrad)]
  end

  def learning([{:bias, w, ir, lr, h} | rest], [{:bias, w1, _, _, _} | rest1], :adagrad) do
    h1 = CM.add(h, CM.emult(w1, w1))
    [{:bias, CM.adagrad(w, w1, h1, lr), ir, lr, h1} | learning(rest, rest1, :adagrad)]
  end

  def learning([{:filter, w, st, pad, ir, lr, v} | rest], [{:filter, w1, _, _, _, _, _} | rest1], :adagrad) do
    w2 = CM.sub(w, CM.mult(w1, lr))
    [{:filter, w2, st, pad, ir, lr, v} | learning(rest, rest1, :adagrad)]
  end



  def learning([network | rest], [_ | rest1], :adagrad) do
    [network | learning(rest, rest1, :adagrad)]
  end

  # calculate accurace 
  def accuracy(image, network, label) do
    [y | _] = forward(image, network, [])
    CM.accuracy(y, label)
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
    i = :rand.uniform(n - 1)
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
      is_number(x) || is_atom(x) ->
        :io.write(x)

      CM.is_matrix(x) ->
        CM.print(x)

      true ->
        print1(x)
        IO.puts("")
    end
  end

  def print1([]) do
    true
  end

  def print1([x | xs]) do
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
    if CM.is_matrix(x) do
      CM.print(x)
    else
      :io.write(x)
    end
  end

  def newline() do
    IO.puts("")
  end
end

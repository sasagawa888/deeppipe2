defmodule Deeppipe do
  alias Cumatrix, as: CM

  # for debug
  def stop() do
    raise("stop")
  end

  # garbage collection
  def gbc() do
    :erlang.garbage_collect()
  end

  # forward
  # return all middle data
  # 1st arg is input data matrix
  # 2nd arg is network list
  # 3rd arg is generated middle layer result
  def forward(_, [], res) do
    res
  end

  def forward(x, [{:weight, w, _, _, _, _} | rest], res) do
    # IO.puts("FD weight")
    x1 = CM.mult(x, w)
    gbc()
    forward(x1, rest, [x1 | res])
  end

  def forward(x, [{:bias, b, _, _, _, _} | rest], res) do
    # IO.puts("FD bias")
    x1 = CM.add(x, b)
    gbc()
    forward(x1, rest, [x1 | res])
  end

  def forward(x, [{:function, name} | rest], res) do
    # IO.puts("FD function")
    x1 = CM.activate(x, name)
    gbc()
    forward(x1, rest, [x1 | res])
  end

  def forward(x, [{:filter, w, st, pad, _, _, _} | rest], res) do
    # IO.puts("FD filter")
    x1 = CM.convolute(x, w, st, pad)
    gbc()
    forward(x1, rest, [x1 | res])
  end

  def forward(x, [{:pooling, st} | rest], [_ | res]) do
    # IO.puts("FD pooling")
    {x1, x2} = CM.pooling(x, st)
    gbc()
    forward(x1, rest, [x1, x2 | res])
  end

  def forward(x, [{:full} | rest], res) do
    # IO.puts("FD full")
    x1 = CM.full(x)
    gbc()
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
    result = backward(loss, network1, x2, [])
    result
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
    # IO.puts("BK softmax")
    backward(l, rest, us, [{:function, :softmax} | res])
  end

  defp backward(l, [{:function, name} | rest], [u | us], res) do
    # IO.puts("BK function")
    l1 = CM.diff(l, u, name)
    gbc()
    backward(l1, rest, us, [{:function, name} | res])
  end

  defp backward(l, [{:bias, _, ir, lr, dr, v} | rest], [_ | us], res) do
    # IO.puts("BK bias")
    b1 = CM.average(l)
    gbc()
    backward(l, rest, us, [{:bias, b1, ir, lr, dr, v} | res])
  end

  defp backward(l, [{:weight, w, ir, lr, dr, v} | rest], [u | us], res) do
    # IO.puts("BK weight")
    {n, _} = CM.size(l)
    w1 = CM.mult(CM.transpose(u), l) |> CM.mult(1 / n)
    l1 = CM.mult(l, CM.transpose(w))
    gbc()
    backward(l1, rest, us, [{:weight, w1, ir, lr, dr, v} | res])
  end

  defp backward(l, [{:filter, w, st, pad, ir, lr, v} | rest], [u | us], res) do
    # IO.puts("BK filter")
    w1 = CM.gradfilter(u, w, l, st, pad)
    l1 = CM.deconvolute(l, w, st, pad)
    gbc()
    backward(l1, rest, us, [{:filter, w1, st, pad, ir, lr, v} | res])
  end

  defp backward(l, [{:pooling, st} | rest], [u | us], res) do
    # IO.puts("BK pooling")
    l1 = CM.unpooling(u, l, st)
    gbc()
    backward(l1, rest, us, [{:pooling, st} | res])
  end

  defp backward(l, [{:full} | rest], [u | us], res) do
    # IO.puts("BK full")
    {_, c, h, w} = CM.size(u)
    l1 = CM.unfull(l, c, h, w)
    gbc()
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

  def learning([{:weight, w, ir, lr, dr, v} | rest], [{:weight, w1, _, _, _, _} | rest1]) do
    # IO.puts("LN weight")
    w2 = CM.sgd(w, w1, lr, dr)
    [{:weight, w2, ir, lr, dr, v} | learning(rest, rest1)]
  end

  def learning([{:bias, w, ir, lr, dr, v} | rest], [{:bias, w1, _, _, _, _} | rest1]) do
    # IO.puts("LN bias")
    w2 = CM.sgd(w, w1, lr, dr)
    [{:bias, w2, ir, lr, dr, v} | learning(rest, rest1)]
  end

  def learning([{:filter, w, st, pad, ir, lr, v} | rest], [{:filter, w1, _, _, _, _, _} | rest1]) do
    # IO.puts("LN filter")
    w2 = CM.sub(w, CM.mult(w1, lr))
    # w2 |> CM.to_list() |> IO.inspect()
    [{:filter, w2, st, pad, ir, lr, v} | learning(rest, rest1)]
  end

  def learning([network | rest], [_ | rest1]) do
    # IO.puts("LN else")
    # IO.inspect(network)
    [network | learning(rest, rest1)]
  end

  # --------momentum-------------
  def learning([], _, :momentum) do
    []
  end

  def learning(
        [{:weight, w, ir, lr, dr, v} | rest],
        [{:weight, w1, _, _, _, _} | rest1],
        :momentum
      ) do
    {v1, w2} = CM.momentum(w, v, w1, lr, dr)
    [{:weight, w2, ir, lr, dr, v1} | learning(rest, rest1, :momentum)]
  end

  def learning([{:bias, w, ir, lr, dr, v} | rest], [{:bias, w1, _, _, _} | rest1], :momentum) do
    {v1, w2} = CM.momentum(w, v, w1, lr, dr)
    [{:bias, w2, ir, lr, v1} | learning(rest, rest1, :momentum)]
  end

  def learning(
        [{:filter, w, st, pad, ir, lr, v} | rest],
        [{:filter, w1, _, _, _, _, _} | rest1],
        :momentum
      ) do
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

  def learning(
        [{:weight, w, ir, lr, dr, h} | rest],
        [{:weight, w1, _, _, _, _} | rest1],
        :adagrad
      ) do
    {h1, w2} = CM.adagrad(w, h, w1, lr, dr)
    [{:weight, w2, ir, lr, dr, h1} | learning(rest, rest1, :adagrad)]
  end

  def learning([{:bias, w, ir, lr, dr, h} | rest], [{:bias, w1, _, _, _, _} | rest1], :adagrad) do
    {h1, w2} = CM.adagrad(w, h, w1, lr, dr)
    [{:bias, w2, ir, lr, dr, h1} | learning(rest, rest1, :adagrad)]
  end

  def learning(
        [{:filter, w, st, pad, ir, lr, v} | rest],
        [{:filter, w1, _, _, _, _, _} | rest1],
        :adagrad
      ) do
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

  # e.g. to_onehot(1,9) => [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  def to_onehot(x, n) do
    to_onehot1(x, n, [])
  end

  def to_onehot1(_, -1, res) do
    res
  end

  def to_onehot1(x, x, res) do
    to_onehot1(x, x - 1, [1.0 | res])
  end

  def to_onehot1(x, c, res) do
    to_onehot1(x, c - 1, [0.0 | res])
  end

  def onehot_to_num([x]) do
    onehot_to_num1(x, 0)
  end

  def onehot_to_num(x) do
    onehot_to_num1(x, 0)
  end

  def onehot_to_num1([x | xs], n) do
    if x == Enum.max([x | xs]) do
      n
    else
      onehot_to_num1(xs, n + 1)
    end
  end

  def normalize(x, bias, div) do
    Enum.map(x, fn z -> (z + bias) / div end)
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

      CM.is_tensor(x) ->
        x |> CM.to_list() |> IO.inspect()

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

  def print2({:filter, w, _, _, _, _, _}) do
    w |> CM.to_list() |> IO.inspect()
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

  # -----------------------------
  # numerical gradient
  # 1st arg input tensor
  # 2nd arg network
  # 3rd arg train tensor
  def numerical_gradient(x, network, t) do
    numerical_gradient1(x, network, t, [], [])
  end

  def numerical_gradient1(_, [], _, _, res) do
    Enum.reverse(res)
  end

  def numerical_gradient1(x, [{:bias, w, ir, lr, dr, v} | rest], t, before, res) do
    # IO.puts("ngrad bias")
    w1 = numerical_gradient_bias(x, w, t, before, {:bias, w, ir, lr, dr, v}, rest)

    numerical_gradient1(x, rest, t, [{:bias, w, ir, lr, dr, v} | before], [
      {:bias, w1, ir, lr, dr, v} | res
    ])
  end

  def numerical_gradient1(x, [{:weight, w, ir, lr, dr, v} | rest], t, before, res) do
    # IO.puts("ngrad wight")
    w1 = numerical_gradient_matrix(x, w, t, before, {:weight, w, ir, lr, dr, v}, rest)

    numerical_gradient1(x, rest, t, [{:weight, w1, ir, lr, dr, v} | before], [
      {:weight, w1, ir, lr, dr, v} | res
    ])
  end

  def numerical_gradient1(x, [{:filter, w, st, pad, ir, lr, v} | rest], t, before, res) do
    # IO.puts("ngrad filter")
    w1 = numerical_gradient_filter(x, w, t, before, {:filter, w, st, pad, ir, lr, v}, rest)

    numerical_gradient1(x, rest, t, [{:filter, w, st, pad, ir, lr, v} | before], [
      {:filter, w1, st, pad, ir, lr, v} | res
    ])
  end

  def numerical_gradient1(x, [y | rest], t, before, res) do
    # IO.puts("ngrad else")
    numerical_gradient1(x, rest, t, [y | before], [y | res])
  end

  # calc numerical gradient of bias
  def numerical_gradient_bias(x, w, t, before, now, rest) do
    {_, c} = Cumatrix.size(w)

    for r1 <- 1..1 do
      for c1 <- 1..c do
        numerical_gradient_bias1(x, t, r1, c1, before, now, rest)
      end
    end
    |> CM.new()
  end

  def numerical_gradient_bias1(x, t, r, c, before, {:bias, w, ir, lr, dr, v}, rest) do
    delta = 0.0001
    w1 = CM.add_diff(w, r, c, delta)
    network0 = Enum.reverse(before) ++ [{:bias, w, ir, lr, dr, v}] ++ rest
    network1 = Enum.reverse(before) ++ [{:bias, w1, ir, lr, dr, v}] ++ rest
    [y0 | _] = forward(x, network0, [])
    [y1 | _] = forward(x, network1, [])
    (CM.loss(y1, t, :cross) - CM.loss(y0, t, :cross)) / delta
  end

  # calc numerical gradient of matrix
  def numerical_gradient_matrix(x, w, t, before, now, rest) do
    {r, c} = Cumatrix.size(w)

    for r1 <- 1..r do
      for c1 <- 1..c do
        numerical_gradient_matrix1(x, t, r1, c1, before, now, rest)
      end
    end
    |> CM.new()
  end

  def numerical_gradient_matrix1(x, t, r, c, before, {:weight, w, ir, lr, dr, v}, rest) do
    delta = 0.0001
    w1 = CM.add_diff(w, r, c, delta)
    network0 = Enum.reverse(before) ++ [{:weight, w, ir, lr, dr, v}] ++ rest
    network1 = Enum.reverse(before) ++ [{:weight, w1, ir, lr, dr, v}] ++ rest
    [y0 | _] = forward(x, network0, [])
    [y1 | _] = forward(x, network1, [])
    (CM.loss(y1, t, :cross) - CM.loss(y0, t, :cross)) / delta
  end

  # calc numerical gradient of filter
  def numerical_gradient_filter(x, w, t, before, now, rest) do
    {c, h, w} = Cumatrix.size(w)

    for c1 <- 1..c do
      for h1 <- 1..h do
        for w1 <- 1..w do
          numerical_gradient_filter1(x, t, c1, h1, w1, before, now, rest)
        end
      end
    end
    |> CM.new()
  end

  def numerical_gradient_filter1(x, t, c, h, w, before, {:filter, m, st, pad, ir, lr, v}, rest) do
    delta = 0.0001
    m1 = CM.add_diff(m, c, h, w, delta)
    network0 = Enum.reverse(before) ++ [{:filter, m, st, pad, ir, lr, v}] ++ rest
    network1 = Enum.reverse(before) ++ [{:filter, m1, st, pad, ir, lr, v}] ++ rest
    [y0 | _] = forward(x, network0, [])
    [y1 | _] = forward(x, network1, [])
    (CM.loss(y1, t, :cross) - CM.loss(y0, t, :cross)) / delta
  end
end

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
    forward(x1, rest, [x1 | res])
  end

  def forward(x, [{:bias, b, _, _, _, _} | rest], res) do
    # IO.puts("FD bias")
    x1 = CM.add(x, b)
    forward(x1, rest, [x1 | res])
  end

  def forward(x, [{:function, name} | rest], res) do
    # IO.puts("FD function")
    x1 = CM.activate(x, name)
    forward(x1, rest, [x1 | res])
  end

  def forward(x, [{:filter, w, {h,_}, pad, _, _, _, _} | rest], res) do
    # IO.puts("FD filter")
    x1 = CM.convolute(x, w, h, pad)
    forward(x1, rest, [x1 | res])
  end

  def forward(x, [{:pooling, st_h, st_w} | rest], [_ | res]) do
    # IO.puts("FD pooling")
    {x1, x2} = CM.pooling(x, st_h, st_w)
    forward(x1, rest, [x1, x2 | res])
  end

  def forward(x, [{:full} | rest], res) do
    # IO.puts("FD full")
    x1 = CM.full(x)
    forward(x1, rest, [x1 | res])
  end

  def forward(x, [{:analizer, n} | rest], res) do
    # IO.puts("FD analizer")
    CM.analizer(x, n)
    forward(x, rest, res)
  end

  def forward(x, [{:visualizer, n, c} | rest], res) do
    # IO.puts("FD visualizer")
    CM.visualizer(x, n, c)
    forward(x, rest, res)
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
    backward(l1, rest, us, [{:function, name} | res])
  end

  defp backward(l, [{:bias, _, ir, lr, dr, v} | rest], [_ | us], res) do
    # IO.puts("BK bias")
    b1 = CM.average(l)
    backward(l, rest, us, [{:bias, b1, ir, lr, dr, v} | res])
  end

  defp backward(l, [{:weight, w, ir, lr, dr, v} | rest], [u | us], res) do
    # IO.puts("BK weight")
    {n, _} = CM.size(l)
    w1 = CM.mult(CM.transpose(u), l) |> CM.mult(1 / n)
    l1 = CM.mult(l, CM.transpose(w))
    backward(l1, rest, us, [{:weight, w1, ir, lr, dr, v} | res])
  end

  defp backward(l, [{:filter, w, {st_h,st_w}, pad, ir, lr, dr, v} | rest], [u | us], res) do
    # IO.puts("BK filter")
    w1 = CM.gradfilter(u, w, l, st_h, pad)
    l1 = CM.deconvolute(l, w, st_h, pad)
    backward(l1, rest, us, [{:filter, w1, {st_h,st_w}, pad, ir, lr, dr, v} | res])
  end

  defp backward(l, [{:pooling, h, w} | rest], [u | us], res) do
    # IO.puts("BK pooling")
    l1 = CM.unpooling(u, l, h)
    backward(l1, rest, us, [{:pooling, h, w} | res])
  end

  defp backward(l, [{:full} | rest], [u | us], res) do
    # IO.puts("BK full")
    {_, c, h, w} = CM.size(u)
    l1 = CM.unfull(l, c, h, w)
    backward(l1, rest, us, [{:full} | res])
  end

  defp backward(l, [{:analizer, n} | rest], us, res) do
    # IO.puts("BK analizer")
    CM.analizer(l, -n)
    backward(l, rest, us, [{:analizer, n} | res])
  end

  defp backward(l, [{:visualizer, n, c} | rest], us, res) do
    # IO.puts("BK visualizer")
    backward(l, rest, us, [{:visualizer, n, c} | res])
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

  def learning([{:filter, w, {st_h,st_w}, pad, ir, lr, dr, v} | rest], [
        {:filter, w1, _, _, _, _, _, _} | rest1
      ]) do
    #IO.puts("LN filter")
    w2 = CM.sgd(w, w1, lr, dr)
    # w2 |> CM.to_list() |> IO.inspect()
    [{:filter, w2, {st_h,st_w}, pad, ir, lr, dr, v} | learning(rest, rest1)]
  end

  def learning([network | rest], [_ | rest1]) do
    # IO.puts("LN else")
    # IO.inspect(network)
    [network | learning(rest, rest1)]
  end

  def learning(network1, network2, :sgd) do
    learning(network1, network2)
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
    #IO.puts("LMom weight")
    {v1, w2} = CM.momentum(w, v, w1, lr, dr)
    [{:weight, w2, ir, lr, dr, v1} | learning(rest, rest1, :momentum)]
  end

  def learning([{:bias, w, ir, lr, dr, v} | rest], [{:bias, w1, _, _, _} | rest1], :momentum) do
    #IO.puts("LMom bias")
    {v1, w2} = CM.momentum(w, v, w1, lr, dr)
    [{:bias, w2, ir, lr, v1} | learning(rest, rest1, :momentum)]
  end

  def learning(
        [{:filter, w, {st_h,st_w}, pad, ir, lr, dr, v} | rest],
        [{:filter, w1, _, _, _, _, _, _} | rest1],
        :momentum
      ) do
    #IO.puts("LMom filter")
    {v1, w2} = CM.momentum(w, v, w1, lr, dr)
    [{:filter, w2, {st_h,st_w}, pad, ir, lr, dr, v1} | learning(rest, rest1, :momentum)]
  end

  def learning([network | rest], [_ | rest1], :momentum) do
    #IO.puts("LMom else")
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
        [{:filter, w, {st_h,st_w}, pad, ir, lr, dr, h} | rest],
        [{:filter, w1, _, _, _, _, _, _} | rest1],
        :adagrad
      ) do
    {h1, w2} = CM.adagrad(w, h, w1, lr, dr)
    [{:filter, w2, {st_h,st_w}, pad, ir, lr, dr, h1} | learning(rest, rest1, :adagrad)]
  end

  def learning([network | rest], [_ | rest1], :adagrad) do
    [network | learning(rest, rest1, :adagrad)]
  end

  # ----------train-----------
  # 1st arg network
  # 2nd arg train image list
  # 3rd arg train onehot list
  # 4th arg test image list
  # 5th arg test labeel list
  # 6th arg loss function (;cross or :squre)
  # 7th arg learning method
  # 8th arg minibatch size
  # 9th arg repeat number

  # automaticaly save network to temp.ex

  def train(network, tr_imag, tr_onehot, ts_imag, ts_label, loss_func, method, m, n) do
    IO.puts("preparing data")
    train_image = tr_imag |> CM.new()
    train_onehot = tr_onehot |> CM.new()
    test_image = ts_imag |> CM.new()

    {time, dict} =
      :timer.tc(fn ->
        train1(network, train_image, train_onehot, test_image, ts_label, loss_func, method, m, n)
      end)

    IO.inspect("time: #{time / 1_000_000} second")
    IO.inspect("-------------")
    dict
  end

  def train1(network, train_image, train_onehot, test_image, test_label, loss_func, method, m, n) do
    IO.puts("learning start")
    IO.puts("count down: loss:")
    network1 = train2(train_image, network, train_onehot, loss_func, method, m, n)
    correct = accuracy(test_image, network1, test_label)
    IO.puts("learning end")
    IO.write("accuracy rate = ")
    IO.puts(correct)
    save("temp.ex", network1)
  end

  def train2(_, network, _, _, _, _, 0) do
    network
  end

  def train2(image, network, train, loss_func, method, m, n) do
    {image1, train1} = CM.random_select(image, train, m)
    network1 = gradient(image1, network, train1)
    network2 = learning(network, network1, method)
    [y | _] = forward(image1, network2, [])
    loss = CM.loss(y, train1, loss_func)
    IO.write(n)
    IO.write(" ")
    IO.puts(loss)
    train2(image, network2, train, loss_func, method, m, n - 1)
  end

  # retrain
  # load network from file and restart learning
  def retrain(file, tr_imag, tr_onehot, ts_imag, ts_label, loss_func, method, m, n) do
    IO.puts("preparing data")
    network = load(file)
    train_image = tr_imag |> CM.new()
    train_onehot = tr_onehot |> CM.new()
    test_image = ts_imag |> CM.new()

    {time, dict} =
      :timer.tc(fn ->
        train1(network, train_image, train_onehot, test_image, ts_label, loss_func, method, m, n)
      end)

    IO.inspect("time: #{time / 1_000_000} second")
    IO.inspect("-------------")
    dict
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

  def save1([{:weight, w, ir, lr, dr, v} | rest]) do
    [{:weight, CM.to_list(w), ir, lr, dr, CM.to_list(v)} | save1(rest)]
  end

  def save1([{:bias, w, ir, lr, dr, v} | rest]) do
    [{:bias, CM.to_list(w), ir, lr, dr, CM.to_list(v)} | save1(rest)]
  end

  def save1([{:filter, w, {st_h,st_w}, pad, ir, lr, dr, v} | rest]) do
    [{:filter, CM.to_list(w), {st_h,st_w}, pad, ir, lr, dr, CM.to_list(v)} | save1(rest)]
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

  def load1([{:weight, w, ir, lr, dr, v} | rest]) do
    [{:weight, CM.new(w), ir, lr, dr, CM.new(v)} | load1(rest)]
  end

  def load1([{:bias, w, ir, lr, dr, v} | rest]) do
    [{:bias, CM.new(w), ir, lr, dr, CM.new(v)} | load1(rest)]
  end

  def load1([{:filter, w, {st_h,st_w}, pad, ir, lr, dr, v} | rest]) do
    [{:filter, CM.new(w), {st_h,st_w}, pad, ir, lr, dr, CM.new(v)} | load1(rest)]
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

  def print2({:weight, w, _, _, _, _}) do
    IO.puts("weight")
    CM.print(w)
  end

  def print2({:bias, w, _, _, _, _}) do
    IO.puts("bias")
    CM.print(w)
  end

  def print2({:function, name}) do
    :io.write(name)
  end

  def print2({:filter, w, _, _, _, _, _, _}) do
    IO.puts("filter")
    CM.print(w)
  end

  def print2(x) do
    if CM.is_matrix(x) do
      CM.print(x)
    else
      :io.write(x)
      IO.puts("")
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

  def numerical_gradient1(x, [{:filter, w, {st_h,st_w}, pad, ir, lr, dr, v} | rest], t, before, res) do
    # IO.puts("ngrad filter")
    w1 = numerical_gradient_filter(x, w, t, before, {:filter, w, {st_h,st_w}, pad, ir, lr, dr, v}, rest)

    numerical_gradient1(x, rest, t, [{:filter, w, {st_h,st_w}, pad, ir, lr, dr, v} | before], [
      {:filter, w1, {st_h,st_w}, pad, ir, lr, dr, v} | res
    ])
  end

  def numerical_gradient1(x, [{:analizer, n} | rest], t, before, res) do
    # IO.puts("FD analizer")
    CM.analizer(x, n)

    numerical_gradient1(x, rest, t, [{:analizer, n} | before], [
      {:analizer, n} | res
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
    {n, c, h, w} = Cumatrix.size(w)

    for n1 <- 1..n do
      for c1 <- 1..c do
        for h1 <- 1..h do
          for w1 <- 1..w do
            numerical_gradient_filter1(x, t, n1, c1, h1, w1, before, now, rest)
          end
        end
      end
    end
    |> CM.new()
  end

  def numerical_gradient_filter1(
        x,
        t,
        n,
        c,
        h,
        w,
        before,
        {:filter, m, {st_h,st_w}, pad, ir, lr, dr, v},
        rest
      ) do
    delta = 0.0001
    m1 = CM.add_diff(m, n, c, h, w, delta)
    network0 = Enum.reverse(before) ++ [{:filter, m, {st_h,st_w}, pad, ir, lr, dr, v}] ++ rest
    network1 = Enum.reverse(before) ++ [{:filter, m1, {st_h,st_w}, pad, ir, lr, dr, v}] ++ rest
    [y0 | _] = forward(x, network0, [])
    [y1 | _] = forward(x, network1, [])
    (CM.loss(y1, t, :cross) - CM.loss(y0, t, :cross)) / delta
  end
end

defmodule Deeppipe do
  alias Cumatrix, as: CM

  @moduledoc """
  main module of DeepPipe2.

  functions for Deep-Learning.


  """

  @doc """
  for debug
  forcely stop
  """
  def stop() do
    raise("stop")
  end

  @doc """
  for debug
  invoke garbage collection forcely.
  """
  def gbc() do
    :erlang.garbage_collect()
  end

  # for dropout. push mask-tensor to input data as tuple.
  # e.g.  [after-data,{befor-data,mask-tensor}|befors]
  # backward with dropout require mask-tensor.
  defp push(x, y, []) do
    [x, y]
  end

  defp push(x, y, [z | zs]) do
    [x, {z, y} | zs]
  end

  @doc """
  forward
  return all middle data
  ```
  1st arg is input data matrix
  2nd arg is network list
  3rd arg is generated middle layer result
  ```
  """
  def forward(_, [], res) do
    res
  end

  def forward(x, [{:weight, w, _, _, dr, _} | rest], res) do
    # IO.puts("FD weight")
    if dr == 0.0 do
      x1 = CM.mult(x, w)
      forward(x1, rest, [x1 | res])
    else
      mw = CM.dropout(w, dr)
      x1 = CM.mult(x, CM.emult(w, mw))
      forward(x1, rest, push(x1, mw, res))
    end
  end

  def forward(x, [{:bias, b, _, _, dr, _} | rest], res) do
    # IO.puts("FD bias")
    if dr == 0.0 do
      x1 = CM.add(x, b)
      forward(x1, rest, [x1 | res])
    else
      mw = CM.dropout(b, dr)
      x1 = CM.add(x, CM.emult(b, mw))
      forward(x1, rest, push(x1, mw, res))
    end
  end

  def forward(x, [{:function, name} | rest], res) do
    # IO.puts("FD function")
    x1 = CM.activate(x, name)
    forward(x1, rest, [x1 | res])
  end

  def forward(x, [{:filter, w, {st_h, st_w}, pad, _, _, dr, _} | rest], res) do
    # IO.puts("FD filter")
    if dr == 0.0 do
      x1 = CM.convolute(x, w, st_h, st_w, pad)
      forward(x1, rest, [x1 | res])
    else
      mw = CM.dropout(w, dr)
      x1 = CM.convolute(x, CM.emult(w, mw), st_h, st_w, pad)
      forward(x1, rest, push(x1, mw, res))
    end
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

  def forward(x, [{:rnn, network} | rest], res) do
    # IO.puts("FD rnn")
    {n, r, c} = CM.size(x)
    # each element of y is 0.0
    y = CM.new(n, c)
    [x1 | x2] = forward_rnn(x, y, network, [], 0, r)
    forward(x1, rest, [x2 | res])
  end

  # for RNN
  # pick up nth row from x 3Dtensor 
  # store intermediate data as list for backward
  def forward_rnn(_, y, _, res, _, 0) do
    [y | res]
  end

  def forward_rnn(x, y, network, res, n, r) do
    x1 = CM.pickup(x, n) |> CM.add(y)
    [x2 | x3] = forward(x1, network, [x1])
    forward_rnn(x, x2, network, [x3 | res], n + 1, r - 1)
  end

  @doc """
  gradient with backpropagation
  ```
  1st arg is input data matrix
  2nd arg is network list
  3rd arg is train matrix
  ```
  """
  def gradient(x, network, t) do
    [x1 | x2] = forward(x, network, [x])
    loss = CM.sub(x1, t)
    network1 = Enum.reverse(network)
    {_, result} = backward(loss, network1, x2, [])
    result
  end

  # backward
  # calculate grad with gackpropagation
  # 1st arg is loss matrix
  # 2nd arg is network list
  # 3rd arg is generated new network with calulated gradient
  # var l is loss matrix
  # var u is input data matrix or tesnro at each layer
  # return {loss,calculated-network}
  defp backward(l, [], _, res) do
    {l, res}
  end

  defp backward(l, [{:weight, w, ir, lr, dr, v} | rest], [u | us], res) do
    # IO.puts("BK weight")
    if dr == 0.0 do
      {n, _} = CM.size(l)
      w1 = CM.mult(CM.transpose(u), l) |> CM.mult(1 / n)
      l1 = CM.mult(l, CM.transpose(w))
      backward(l1, rest, us, [{:weight, w1, ir, lr, 0.0, v} | res])
    else
      {n, _} = CM.size(l)
      {u1, mw} = u
      w1 = CM.mult(CM.transpose(u1), l) |> CM.mult(1 / n) |> CM.emult(mw)
      l1 = CM.mult(l, CM.transpose(CM.emult(w, mw)))
      backward(l1, rest, us, [{:weight, w1, ir, lr, dr, v} | res])
    end
  end

  defp backward(l, [{:bias, _, ir, lr, dr, v} | rest], [u | us], res) do
    # IO.puts("BK bias")
    if dr == 0.0 do
      b1 = CM.average(l)
      backward(l, rest, us, [{:bias, b1, ir, lr, 0.0, v} | res])
    else
      {_, mw} = u
      b1 = CM.average(l) |> CM.emult(mw)
      backward(l, rest, us, [{:bias, b1, ir, lr, dr, v} | res])
    end
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

  defp backward(l, [{:filter, w, {st_h, st_w}, pad, ir, lr, dr, v} | rest], [u | us], res) do
    # IO.puts("BK filter")
    if dr == 0.0 do
      w1 = CM.gradfilter(u, w, l, st_h, st_w, pad)
      l1 = CM.deconvolute(l, w, st_h, st_w, pad)
      backward(l1, rest, us, [{:filter, w1, {st_h, st_w}, pad, ir, lr, 0.0, v} | res])
    else
      {u1, mw} = u
      w1 = CM.gradfilter(u1, w, l, st_h, st_w, pad) |> CM.emult(mw)
      l1 = CM.deconvolute(l, CM.emult(w, mw), st_h, st_w, pad)
      backward(l1, rest, us, [{:filter, w1, {st_h, st_w}, pad, ir, lr, dr, v} | res])
    end
  end

  defp backward(l, [{:pooling, st_h, st_w} | rest], [u | us], res) do
    # IO.puts("BK pooling")
    l1 = CM.unpooling(u, l, st_h, st_w)
    backward(l1, rest, us, [{:pooling, st_h, st_w} | res])
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

  defp backward(l, [{:rnn, network} | rest], [u | us], res) do
    # IO.puts("BK network"
    # u is intermediate data list. length of u means recursive nest number
    n = length(u)
    network1 = backward_rnn(l, network, u, n)
    backward(l, rest, us, [{:rnn, network1} | res])
  end

  # for RNN
  # backward recursively and generate average weight
  # l is loss matrix
  # network is partial network 
  # u is imidiate data list
  # n is recursive number
  def backward_rnn(l, network, u, n) do
    backward_rnn1(l, network, u, [], n) |> backward_rnn2()
  end

  def backward_rnn1(_, _, _, res, 0) do
    res
  end

  def backward_rnn1(l, network, [u | us], res, n) do
    {l1, g1} = backward(l, network, u, [])
    backward_rnn1(l1, network, us, [g1 | res], n - 1)
  end

  # average of many networks
  # case of one network
  def backward_rnn2([g]) do
    [g]
  end

  # case of two network
  def backward_rnn2([g1, g2]) do
    backward_rnn3(g1, g2)
  end

  # case of many network
  def backward_rnn2([g1, g2 | gs]) do
    backward_rnn2([backward_rnn3(g1, g2) | gs])
  end

  # average of two networks
  def backward_rnn3([], []) do
    []
  end

  def backward_rnn3([{:weight, w1, ir, lr, dr, v} | rest1], [{:weight, w2, _, _, _, _} | rest2]) do
    w3 = CM.average(w1, w2)
    [{:weight, w3, ir, lr, dr, v} | backward_rnn3(rest1, rest2)]
  end

  def backward_rnn3([{:bias, b1, ir, lr, dr, v} | rest1], [{:bias, b2, _, _, _, _} | rest2]) do
    b3 = CM.average(b1, b2)
    [{:bias, b3, ir, lr, dr, v} | backward_rnn3(rest1, rest2)]
  end

  def backward_rnn3([{:filter, w1, {st_h, st_w}, pad, ir, lr, dr, v} | rest1], [
        {:filter, w2, _, _, _, _, _, _} | rest2
      ]) do
    w3 = CM.average(w1, w2)
    [{:filter, w3, {st_h, st_w}, pad, ir, lr, dr, v} | backward_rnn3(rest1, rest2)]
  end

  def backward_rnn3([network | rest1], [network | rest2]) do
    [network | backward_rnn3(rest1, rest2)]
  end

  @doc """
  learning(network1,network2,update_method)
  learning/3
  generate new network with leared weight,bias and filter
  update method is :sgd, :momentum, :rms, :adagrad ,:adam 
  """
  # --------sgd----------
  def learning([], _, :sgd) do
    []
  end

  def learning([{:weight, w, ir, lr, dr, v} | rest], [{:weight, w1, _, _, _, _} | rest1], :sgd) do
    # IO.puts("LN weight")
    w2 = CM.sgd(w, w1, lr)
    [{:weight, w2, ir, lr, dr, v} | learning(rest, rest1, :sgd)]
  end

  def learning([{:bias, w, ir, lr, dr, v} | rest], [{:bias, w1, _, _, _, _} | rest1], :sgd) do
    # IO.puts("LN bias")
    w2 = CM.sgd(w, w1, lr)
    [{:bias, w2, ir, lr, dr, v} | learning(rest, rest1, :sgd)]
  end

  def learning(
        [{:filter, w, {st_h, st_w}, pad, ir, lr, dr, v} | rest],
        [
          {:filter, w1, _, _, _, _, _, _} | rest1
        ],
        :sgd
      ) do
    # IO.puts("LN filter")
    w2 = CM.sgd(w, w1, lr)
    # w2 |> CM.to_list() |> IO.inspect()
    [{:filter, w2, {st_h, st_w}, pad, ir, lr, dr, v} | learning(rest, rest1, :sgd)]
  end

  def learning([{:rnn, network} | rest], [{:rnn, network1} | rest1], :sgd) do
    [{:rnn, learning(network, network1, :sgd)} | learning(rest, rest1, :sgd)]
  end

  def learning([network | rest], [_ | rest1], :sgd) do
    # IO.puts("LN else")
    # IO.inspect(network)
    [network | learning(rest, rest1, :sgd)]
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
    # IO.puts("LMom weight")
    {v1, w2} = CM.momentum(w, v, w1, lr)
    [{:weight, w2, ir, lr, dr, v1} | learning(rest, rest1, :momentum)]
  end

  def learning([{:bias, w, ir, lr, dr, v} | rest], [{:bias, w1, _, _, _} | rest1], :momentum) do
    # IO.puts("LMom bias")
    {v1, w2} = CM.momentum(w, v, w1, lr)
    [{:bias, w2, ir, lr, dr, v1} | learning(rest, rest1, :momentum)]
  end

  def learning(
        [{:filter, w, {st_h, st_w}, pad, ir, lr, dr, v} | rest],
        [{:filter, w1, _, _, _, _, _, _} | rest1],
        :momentum
      ) do
    # IO.puts("LMom filter")
    {v1, w2} = CM.momentum(w, v, w1, lr)
    [{:filter, w2, {st_h, st_w}, pad, ir, lr, dr, v1} | learning(rest, rest1, :momentum)]
  end

  def learning([{:rnn, network} | rest], [{:rnn, network1} | rest1], :momentum) do
    [{:rnn, learning(network, network1, :momentum)} | learning(rest, rest1, :momentum)]
  end

  def learning([network | rest], [_ | rest1], :momentum) do
    # IO.puts("LMom else")
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
    {h1, w2} = CM.adagrad(w, h, w1, lr)
    [{:weight, w2, ir, lr, dr, h1} | learning(rest, rest1, :adagrad)]
  end

  def learning([{:bias, w, ir, lr, dr, h} | rest], [{:bias, w1, _, _, _, _} | rest1], :adagrad) do
    {h1, w2} = CM.adagrad(w, h, w1, lr)
    [{:bias, w2, ir, lr, dr, h1} | learning(rest, rest1, :adagrad)]
  end

  def learning(
        [{:filter, w, {st_h, st_w}, pad, ir, lr, dr, h} | rest],
        [{:filter, w1, _, _, _, _, _, _} | rest1],
        :adagrad
      ) do
    {h1, w2} = CM.adagrad(w, h, w1, lr)
    [{:filter, w2, {st_h, st_w}, pad, ir, lr, dr, h1} | learning(rest, rest1, :adagrad)]
  end

  def learning([{:rnn, network} | rest], [{:rnn, network1} | rest1], :adagrad) do
    [{:rnn, learning(network, network1, :adagrad)} | learning(rest, rest1, :adagrad)]
  end

  def learning([network | rest], [_ | rest1], :adagrad) do
    [network | learning(rest, rest1, :adagrad)]
  end

  # --------RMSprop--------------
  def learning([], _, :rms) do
    []
  end

  def learning(
        [{:weight, w, ir, lr, dr, h} | rest],
        [{:weight, w1, _, _, _, _} | rest1],
        :rms
      ) do
    {h1, w2} = CM.rms(w, h, w1, lr)
    [{:weight, w2, ir, lr, dr, h1} | learning(rest, rest1, :rms)]
  end

  def learning([{:bias, w, ir, lr, dr, h} | rest], [{:bias, w1, _, _, _, _} | rest1], :rms) do
    {h1, w2} = CM.rms(w, h, w1, lr)
    [{:bias, w2, ir, lr, dr, h1} | learning(rest, rest1, :rms)]
  end

  def learning(
        [{:filter, w, {st_h, st_w}, pad, ir, lr, dr, h} | rest],
        [{:filter, w1, _, _, _, _, _, _} | rest1],
        :rms
      ) do
    {h1, w2} = CM.rms(w, h, w1, lr)
    [{:filter, w2, {st_h, st_w}, pad, ir, lr, dr, h1} | learning(rest, rest1, :rms)]
  end

  def learning([{:rnn, network} | rest], [{:rnn, network1} | rest1], :rms) do
    [{:rnn, learning(network, network1, :rms)} | learning(rest, rest1, :rms)]
  end

  def learning([network | rest], [_ | rest1], :rms) do
    [network | learning(rest, rest1, :rms)]
  end

  # --------Adam--------------
  def learning([], _, :adam) do
    []
  end

  def learning(
        [{:weight, w, ir, lr, dr, h} | rest],
        [{:weight, w1, _, _, _, _} | rest1],
        :adam
      ) do
    if CM.is_matrix(h) || CM.is_tensor(h) do
      {m1, v1, w2} = CM.adam(w, h, h, w1, lr)
      [{:weight, w2, ir, lr, dr, {m1, v1}} | learning(rest, rest1, :adam)]
    else
      {m, v} = h
      {m1, v1, w2} = CM.adam(w, m, v, w1, lr)
      [{:weight, w2, ir, lr, dr, {m1, v1}} | learning(rest, rest1, :adam)]
    end
  end

  def learning([{:bias, w, ir, lr, dr, h} | rest], [{:bias, w1, _, _, _, _} | rest1], :adam) do
    if CM.is_matrix(h) || CM.is_tensor(h) do
      {m1, v1, w2} = CM.adam(w, h, h, w1, lr)
      [{:bias, w2, ir, lr, dr, {m1, v1}} | learning(rest, rest1, :adam)]
    else
      {m, v} = h
      {m1, v1, w2} = CM.adam(w, m, v, w1, lr)
      [{:bias, w2, ir, lr, dr, {m1, v1}} | learning(rest, rest1, :adam)]
    end
  end

  def learning(
        [{:filter, w, {st_h, st_w}, pad, ir, lr, dr, h} | rest],
        [{:filter, w1, _, _, _, _, _, _} | rest1],
        :adam
      ) do
    if CM.is_matrix(h) || CM.is_tensor(h) do
      {m1, v1, w2} = CM.adam(w, h, h, w1, lr)
      [{:filter, w2, {st_h, st_w}, pad, ir, lr, dr, {m1, v1}} | learning(rest, rest1, :adam)]
    else
      {m, v} = h
      {m1, v1, w2} = CM.adam(w, m, v, w1, lr)
      [{:filter, w2, {st_h, st_w}, pad, ir, lr, dr, {m1, v1}} | learning(rest, rest1, :adam)]
    end
  end

  def learning([{:rnn, network} | rest], [{:rnn, network1} | rest1], :adam) do
    [{:rnn, learning(network, network1, :adam)} | learning(rest, rest1, :adam)]
  end

  def learning([network | rest], [_ | rest1], :adam) do
    [network | learning(rest, rest1, :adam)]
  end

  # ------------train------------
  defp repeat(size, mini) do
    if rem(size, mini) == 0 do
      div(size, mini)
    else
      div(size, mini) + 1
    end
  end

  @doc """
  ```
  1st arg network
  2nd arg train image list
  3rd arg train onehot list
  4th arg test image list
  5th arg test label list
  6th arg loss function (:cross or :square)
  7th arg learning method (:sgd :momentum :rms :adagrad :adam)
  8th arg minibatch size
  9th arg epochs 
  ```
  automaticaly save network to temp.ex
  """
  def train(network, tr_imag, tr_onehot, ts_imag, ts_label, loss_func, method, m, e) do
    IO.puts("preparing data")
    train_image = tr_imag |> CM.new() |> CM.standardize()
    train_onehot = tr_onehot |> CM.new()
    n = repeat(length(tr_onehot), m)

    {time, network1} =
      :timer.tc(fn ->
        train1(
          network,
          train_image,
          train_onehot,
          ts_imag,
          ts_label,
          loss_func,
          method,
          m,
          n,
          e,
          1
        )
      end)

    save("temp.ex", network1)
    IO.puts("time: #{time / 1_000_000} second")
    :ok
  end

  defp train1(network, _, _, _, _, _, _, _, _, 0, _) do
    network
  end

  defp train1(
         network,
         train_image,
         train_onehot,
         ts_imag,
         ts_label,
         loss_func,
         method,
         m,
         n,
         epoch,
         count
       ) do
    IO.puts("\nepoch #{count}")
    network1 = train2(network, train_image, train_onehot, loss_func, method, m, n, n)
    {train_image1, train_onehot1} = CM.random_select(train_image, train_onehot, m)
    [y | _] = forward(train_image1, network1, [])
    loss = CM.loss(y, train_onehot1, loss_func)
    IO.puts("random loss = #{loss}")
    rate = accuracy(ts_imag, network1, ts_label, m)
    IO.puts("accuracy rate = #{rate * 100}%")

    train1(
      network1,
      train_image,
      train_onehot,
      ts_imag,
      ts_label,
      loss_func,
      method,
      m,
      n,
      epoch - 1,
      count + 1
    )
  end

  defp train2(network, _, _, _, _, _, 0, _) do
    progress(1.0)
    newline()
    network
  end

  defp train2(network, train_image, train_onehot, loss_func, method, m, n, all) do
    {train_image1, train_onehot1} = CM.random_select(train_image, train_onehot, m)
    network1 = gradient(train_image1, network, train_onehot1)
    network2 = learning(network, network1, method)
    rate = (all - n) / all
    progress(rate)
    train2(network2, train_image, train_onehot, loss_func, method, m, n - 1, all)
  end

  @doc """
  retrain
  load network from file and restart learning
  """
  def retrain(file, tr_imag, tr_onehot, ts_imag, ts_label, loss_func, method, m, e) do
    IO.puts("preparing data")
    network = load(file)
    train_image = tr_imag |> CM.new() |> CM.standardize()
    train_onehot = tr_onehot |> CM.new()
    n = div(length(tr_onehot), m)

    {time, network1} =
      :timer.tc(fn ->
        train1(
          network,
          train_image,
          train_onehot,
          ts_imag,
          ts_label,
          loss_func,
          method,
          m,
          n,
          e,
          1
        )
      end)

    save("temp.ex", network1)
    IO.puts("time: #{time / 1_000_000} second")
    :ok
  end

  @doc """
  for easy test to decide parameter
  ```
  1st arg network
  2nd arg train image list
  3rd arg train onehot list
  4th arg test image list
  5th arg test label list
  6th arg loss function (:cross or :square)
  7th arg learning method (:sgd :momentum :adagrad :adam)
  8th arg minibatch size
  9th arg repeat number
  ```
  automaticaly save network to temp.ex
  """
  def try(network, tr_imag, tr_onehot, ts_imag, ts_label, loss_func, method, m, n) do
    IO.puts("preparing data")
    train_image = tr_imag |> CM.new() |> CM.standardize()
    train_onehot = tr_onehot |> CM.new()

    {time, network1} =
      :timer.tc(fn ->
        try1(network, train_image, train_onehot, loss_func, method, m, n)
      end)

    rate = accuracy(ts_imag, network1, ts_label, m)
    IO.puts("accuracy rate = #{rate * 100}%")

    IO.inspect("time: #{time / 1_000_000} second")
    :ok
  end

  defp try1(network, train_image, train_onehot, loss_func, method, m, n) do
    IO.puts("count down: loss:")
    network1 = try2(train_image, network, train_onehot, loss_func, method, m, n)
    save("temp.ex", network1)
    network1
  end

  defp try2(_, network, _, _, _, _, 0) do
    network
  end

  defp try2(image, network, train, loss_func, method, m, n) do
    {image1, train1} = CM.random_select(image, train, m)
    network1 = gradient(image1, network, train1)
    network2 = learning(network, network1, method)
    [y | _] = forward(image1, network2, [])
    loss = CM.loss(y, train1, loss_func)
    IO.write(n)
    IO.write(" ")
    IO.puts(loss)
    try2(image, network2, train, loss_func, method, m, n - 1)
  end

  @doc """
  retry
  load network from file and restart learning
  """
  def retry(file, tr_imag, tr_onehot, ts_imag, ts_label, loss_func, method, m, n) do
    IO.puts("preparing data")
    network = load(file)
    train_image = tr_imag |> CM.new() |> CM.standardize()
    train_onehot = tr_onehot |> CM.new()

    {time, network1} =
      :timer.tc(fn ->
        try1(network, train_image, train_onehot, loss_func, method, m, n)
      end)

    correct = accuracy(ts_imag, network1, ts_label, m)
    IO.puts("learning end")
    IO.write("accuracy rate = ")
    IO.puts(correct)

    IO.inspect("time: #{time / 1_000_000} second")
    IO.inspect("-------------")
    :ok
  end

  @doc """
  calculate accuracy
  1st arg  list of image
  2nd arg  network
  3rd arg  list of label
  4th arg  mini batch size
  """
  def accuracy(image, network, label, m) do
    accuracy1(image, network, label, m, length(label), 0)
  end

  defp accuracy1([], _, [], _, total, correct) do
    correct / total
  end

  defp accuracy1(image, network, label, m, total, correct) do
    n = min(length(label), m)
    image1 = Enum.take(image, n) |> CM.new() |> CM.standardize()
    label1 = Enum.take(label, n)
    [y | _] = forward(image1, network, [])
    n1 = CM.correct(y, label1)
    accuracy1(Enum.drop(image, n), network, Enum.drop(label, n), m, total, correct + n1)
  end

  @doc """
  select random data from image data and train data 
  size of m. range from 0 to n
  and generate tuple of two matrix
  """
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

  @doc """
  translate from number to onehot-list
  iex(1)> Deeppipe.to_onehot(1,9)
  [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  """
  def to_onehot(x, n) do
    to_onehot1(x, n, [])
  end

  defp to_onehot1(_, -1, res) do
    res
  end

  defp to_onehot1(x, x, res) do
    to_onehot1(x, x - 1, [1.0 | res])
  end

  defp to_onehot1(x, c, res) do
    to_onehot1(x, c - 1, [0.0 | res])
  end

  @doc """
  normalize dataset element
  normalize(x,bias,div)
  ```
  x + bias / div
  e.g. bias = -127, div = 255
  0~255 => -0.5~0.5 
  ```
  """
  def normalize(x, bias, div) do
    Enum.map(x, fn z -> (z + bias) / div end)
  end

  @doc """
  save network to file
  """
  def save(file, network) do
    network1 = save1(network)
    File.write(file, inspect(network1, limit: :infinity))
  end

  defp save1([]) do
    []
  end

  defp save1([{:weight, w, ir, lr, dr, h} | rest]) do
    if CM.is_matrix(h) || CM.is_tensor(h) do
      [{:weight, CM.to_list(w), ir, lr, dr, CM.to_list(h)} | save1(rest)]
    else
      {m, v} = h
      [{:weight, CM.to_list(w), ir, lr, dr, {CM.to_list(m), CM.to_list(v)}} | save1(rest)]
    end
  end

  defp save1([{:bias, w, ir, lr, dr, h} | rest]) do
    if CM.is_matrix(h) || CM.is_tensor(h) do
      [{:bias, CM.to_list(w), ir, lr, dr, CM.to_list(h)} | save1(rest)]
    else
      {m, v} = h
      [{:bias, CM.to_list(w), ir, lr, dr, {CM.to_list(m), CM.to_list(v)}} | save1(rest)]
    end
  end

  defp save1([{:filter, w, {st_h, st_w}, pad, ir, lr, dr, h} | rest]) do
    if CM.is_matrix(h) || CM.is_tensor(h) do
      [{:filter, CM.to_list(w), {st_h, st_w}, pad, ir, lr, dr, CM.to_list(h)} | save1(rest)]
    else
      {m, v} = h

      [
        {:filter, CM.to_list(w), {st_h, st_w}, pad, ir, lr, dr, {CM.to_list(m), CM.to_list(v)}}
        | save1(rest)
      ]
    end
  end

  defp save1([{:function, name} | rest]) do
    [{:function, name} | save1(rest)]
  end

  defp save1([network | rest]) do
    [network | save1(rest)]
  end

  @doc """
  load network from file
  """
  def load(file) do
    Code.eval_file(file) |> elem(0) |> load1
  end

  defp load1([]) do
    []
  end

  defp load1([{:weight, w, ir, lr, dr, h} | rest]) do
    if !is_tuple(h) do
      [{:weight, CM.new(w), ir, lr, dr, CM.new(h)} | load1(rest)]
    else
      {m, v} = h
      [{:weight, CM.new(w), ir, lr, dr, {CM.new(m), CM.new(v)}} | load1(rest)]
    end
  end

  defp load1([{:bias, w, ir, lr, dr, h} | rest]) do
    if !is_tuple(h) do
      [{:bias, CM.new(w), ir, lr, dr, CM.new(h)} | load1(rest)]
    else
      {m, v} = h
      [{:bias, CM.new(w), ir, lr, dr, {CM.new(m), CM.new(v)}} | load1(rest)]
    end
  end

  defp load1([{:filter, w, {st_h, st_w}, pad, ir, lr, dr, h} | rest]) do
    if !is_tuple(h) do
      [{:filter, CM.new(w), {st_h, st_w}, pad, ir, lr, dr, CM.new(h)} | load1(rest)]
    else
      {m, v} = h
      [{:filter, CM.new(w), {st_h, st_w}, pad, ir, lr, dr, {CM.new(m), CM.new(v)}} | load1(rest)]
    end
  end

  defp load1([{:function, name} | rest]) do
    [{:function, name} | load1(rest)]
  end

  defp load1([network | rest]) do
    [network | load1(rest)]
  end

  @doc """
  display network
  """
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

  defp print1([]) do
    true
  end

  defp print1([x | xs]) do
    print2(x)
    print1(xs)
  end

  defp print2({:weight, w, _, _, _, _}) do
    IO.puts("weight")
    CM.print(w)
  end

  defp print2({:bias, w, _, _, _, _}) do
    IO.puts("bias")
    CM.print(w)
  end

  defp print2({:function, name}) do
    :io.write(name)
  end

  defp print2({:filter, w, _, _, _, _, _, _}) do
    IO.puts("filter")
    CM.print(w)
  end

  defp print2(x) do
    if CM.is_matrix(x) do
      CM.print(x)
    else
      :io.write(x)
      IO.puts("")
    end
  end

  @doc """
  display newline
  """
  def newline() do
    IO.puts("")
  end

  def progress(r) do
    size = 50
    done = round(size * r)
    yet = size - done
    done_str = String.duplicate("#", done)
    yet_str = String.duplicate(" ", yet)
    IO.write("\r[")
    IO.write(done_str)
    IO.write(yet_str)
    IO.write("](#{round(r * 100)}%)")
  end

  @doc """
  download(x)
  ```
  case x
  :mnist   download and decompress MNIST dataset
  :fashon  download and decompress Fashion-MNIST dataset
  :cifar10 download and decompress CIFAR10 dataset
  :iris    download iris dataset
  ```
  """
  def download(:mnist) do
    Application.ensure_all_started(:inets)
    base_url = 'http://yann.lecun.com/exdb/mnist/'

    {:ok, resp} =
      :httpc.request(:get, {base_url ++ 'train-images-idx3-ubyte.gz', []}, [],
        body_format: :binary
      )

    {{_, 200, 'OK'}, _headers, body} = resp
    Mix.shell().cmd("mkdir mnist")
    File.write!("mnist/train-images-idx3-ubyte.gz", body)

    {:ok, resp} =
      :httpc.request(:get, {base_url ++ 'train-labels-idx1-ubyte.gz', []}, [],
        body_format: :binary
      )

    {{_, 200, 'OK'}, _headers, body} = resp
    File.write!("mnist/train-labels-idx1-ubyte.gz", body)

    {:ok, resp} =
      :httpc.request(:get, {base_url ++ 't10k-images-idx3-ubyte.gz', []}, [], body_format: :binary)

    {{_, 200, 'OK'}, _headers, body} = resp
    File.write!("mnist/t10k-images-idx3-ubyte.gz", body)

    {:ok, resp} =
      :httpc.request(:get, {base_url ++ 't10k-labels-idx1-ubyte.gz', []}, [], body_format: :binary)

    {{_, 200, 'OK'}, _headers, body} = resp
    File.write!("mnist/t10k-labels-idx1-ubyte.gz", body)
    Mix.shell().cmd("gzip -d mnist/train-images-idx3-ubyte.gz")
    Mix.shell().cmd("gzip -d mnist/train-labels-idx1-ubyte.gz")
    Mix.shell().cmd("gzip -d mnist/t10k-images-idx3-ubyte.gz")
    Mix.shell().cmd("gzip -d mnist/t10k-labels-idx1-ubyte.gz")
    :ok
  end

  def download(:fashion) do
    Application.ensure_all_started(:inets)
    base_url = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'

    {:ok, resp} =
      :httpc.request(:get, {base_url ++ 'train-images-idx3-ubyte.gz', []}, [],
        body_format: :binary
      )

    {{_, 200, 'OK'}, _headers, body} = resp
    Mix.shell().cmd("mkdir fashion")
    File.write!("fashion/train-images-idx3-ubyte.gz", body)

    {:ok, resp} =
      :httpc.request(:get, {base_url ++ 'train-labels-idx1-ubyte.gz', []}, [],
        body_format: :binary
      )

    {{_, 200, 'OK'}, _headers, body} = resp
    File.write!("fashion/train-labels-idx1-ubyte.gz", body)

    {:ok, resp} =
      :httpc.request(:get, {base_url ++ 't10k-images-idx3-ubyte.gz', []}, [], body_format: :binary)

    {{_, 200, 'OK'}, _headers, body} = resp
    File.write!("fashion/t10k-images-idx3-ubyte.gz", body)

    {:ok, resp} =
      :httpc.request(:get, {base_url ++ 't10k-labels-idx1-ubyte.gz', []}, [], body_format: :binary)

    {{_, 200, 'OK'}, _headers, body} = resp
    File.write!("fashion/t10k-labels-idx1-ubyte.gz", body)
    Mix.shell().cmd("gzip -d fashion/train-images-idx3-ubyte.gz")
    Mix.shell().cmd("gzip -d fashion/train-labels-idx1-ubyte.gz")
    Mix.shell().cmd("gzip -d fashion/t10k-images-idx3-ubyte.gz")
    Mix.shell().cmd("gzip -d fashion/t10k-labels-idx1-ubyte.gz")
    :ok
  end

  def download(:iris) do
    Application.ensure_all_started(:inets)
    base_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/'
    {:ok, resp} = :httpc.request(:get, {base_url ++ 'iris.data', []}, [], body_format: :binary)
    {{_, 200, 'OK'}, _headers, body} = resp
    Mix.shell().cmd("mkdir iris")
    File.write!("iris/iris.data", body)
    :ok
  end

  def download(:cifar10) do
    IO.puts("wait few minutes")
    Application.ensure_all_started(:inets)
    base_url = 'https://www.cs.toronto.edu/~kriz/'

    {:ok, resp} =
      :httpc.request(:get, {base_url ++ 'cifar-10-binary.tar.gz', []}, [], body_format: :binary)

    {{_, 200, 'OK'}, _headers, body} = resp
    File.write!("cifar-10-binary.tar.gz", body)
    Mix.shell().cmd("tar xzvf cifar-10-binary.tar.gz")
    Mix.shell().cmd("rm *.tar.gz")
    :ok
  end

  @doc """
  numerical_gradient(ts,network,train)
  numerical gradient for debug
  1st arg input tensor
  2nd arg network
  3rd arg train matrix
  """
  def numerical_gradient(x, network, t) do
    numerical_gradient1(x, network, t, [], [])
  end

  defp numerical_gradient1(_, [], _, _, res) do
    Enum.reverse(res)
  end

  defp numerical_gradient1(x, [{:bias, w, ir, lr, dr, v} | rest], t, before, res) do
    # IO.puts("ngrad bias")
    w1 = numerical_gradient_bias(x, w, t, before, {:bias, w, ir, lr, dr, v}, rest)

    numerical_gradient1(x, rest, t, [{:bias, w, ir, lr, dr, v} | before], [
      {:bias, w1, ir, lr, dr, v} | res
    ])
  end

  defp numerical_gradient1(x, [{:weight, w, ir, lr, dr, v} | rest], t, before, res) do
    # IO.puts("ngrad wight")
    w1 = numerical_gradient_matrix(x, w, t, before, {:weight, w, ir, lr, dr, v}, rest)

    numerical_gradient1(x, rest, t, [{:weight, w1, ir, lr, dr, v} | before], [
      {:weight, w1, ir, lr, dr, v} | res
    ])
  end

  defp numerical_gradient1(
         x,
         [{:filter, w, {st_h, st_w}, pad, ir, lr, dr, v} | rest],
         t,
         before,
         res
       ) do
    # IO.puts("ngrad filter")
    w1 =
      numerical_gradient_filter(
        x,
        w,
        t,
        before,
        {:filter, w, {st_h, st_w}, pad, ir, lr, dr, v},
        rest
      )

    numerical_gradient1(x, rest, t, [{:filter, w, {st_h, st_w}, pad, ir, lr, dr, v} | before], [
      {:filter, w1, {st_h, st_w}, pad, ir, lr, dr, v} | res
    ])
  end

  defp numerical_gradient1(x, [{:analizer, n} | rest], t, before, res) do
    # IO.puts("FD analizer")
    CM.analizer(x, n)

    numerical_gradient1(x, rest, t, [{:analizer, n} | before], [
      {:analizer, n} | res
    ])
  end

  defp numerical_gradient1(x, [y | rest], t, before, res) do
    # IO.puts("ngrad else")
    numerical_gradient1(x, rest, t, [y | before], [y | res])
  end

  # calc numerical gradient of bias
  defp numerical_gradient_bias(x, w, t, before, now, rest) do
    {_, c} = Cumatrix.size(w)

    for r1 <- 1..1 do
      for c1 <- 1..c do
        numerical_gradient_bias1(x, t, r1, c1, before, now, rest)
      end
    end
    |> CM.new()
  end

  defp numerical_gradient_bias1(x, t, r, c, before, {:bias, w, ir, lr, dr, v}, rest) do
    delta = 0.0001
    w1 = CM.add_diff(w, r, c, delta)
    network0 = Enum.reverse(before) ++ [{:bias, w, ir, lr, dr, v}] ++ rest
    network1 = Enum.reverse(before) ++ [{:bias, w1, ir, lr, dr, v}] ++ rest
    [y0 | _] = forward(x, network0, [])
    [y1 | _] = forward(x, network1, [])
    (CM.loss(y1, t, :cross) - CM.loss(y0, t, :cross)) / delta
  end

  # calc numerical gradient of matrix
  defp numerical_gradient_matrix(x, w, t, before, now, rest) do
    {r, c} = Cumatrix.size(w)

    for r1 <- 1..r do
      for c1 <- 1..c do
        numerical_gradient_matrix1(x, t, r1, c1, before, now, rest)
      end
    end
    |> CM.new()
  end

  defp numerical_gradient_matrix1(x, t, r, c, before, {:weight, w, ir, lr, dr, v}, rest) do
    delta = 0.0001
    w1 = CM.add_diff(w, r, c, delta)
    network0 = Enum.reverse(before) ++ [{:weight, w, ir, lr, dr, v}] ++ rest
    network1 = Enum.reverse(before) ++ [{:weight, w1, ir, lr, dr, v}] ++ rest
    [y0 | _] = forward(x, network0, [])
    [y1 | _] = forward(x, network1, [])
    (CM.loss(y1, t, :cross) - CM.loss(y0, t, :cross)) / delta
  end

  # calc numerical gradient of filter
  defp numerical_gradient_filter(x, w, t, before, now, rest) do
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

  defp numerical_gradient_filter1(
         x,
         t,
         n,
         c,
         h,
         w,
         before,
         {:filter, m, {st_h, st_w}, pad, ir, lr, dr, v},
         rest
       ) do
    delta = 0.0001
    m1 = CM.add_diff(m, n, c, h, w, delta)
    network0 = Enum.reverse(before) ++ [{:filter, m, {st_h, st_w}, pad, ir, lr, dr, v}] ++ rest
    network1 = Enum.reverse(before) ++ [{:filter, m1, {st_h, st_w}, pad, ir, lr, dr, v}] ++ rest
    [y0 | _] = forward(x, network0, [])
    [y1 | _] = forward(x, network1, [])
    (CM.loss(y1, t, :cross) - CM.loss(y0, t, :cross)) / delta
  end
end

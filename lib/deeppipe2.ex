defmodule Deeppipe do
  alias Cumatrix, as: CM
  
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
    cond do
      name == :sigmoid ->
        x1 = CM.activate(x, :sigmoid)
        forward(x1, rest, [x1 | res])

      name == :tanh ->
        x1 = CM.activate(x, :tanh)
        forward(x1, rest, [x1 | res])

      name == :relu ->
        x1 = CM.activate(x, :relu)
        forward(x1, rest, [x1 | res])

      name == :softmax ->
        x1 = CM.activate(x, :softmax)
        forward(x1, rest, [x1 | res])

      true ->
        raise "not exist function"
    end
  end

  # numerical gradient
  # 1st arg is input data matrix
  # 2nd arg is network list
  # 3rd arg is teacher data matrix
  # 4th arg is loss function 
  def ngrad(x, network, t, :square) do
    ngrad(x, network, t)
  end

  def ngrad(x, network, t, :cross) do
    ngrad1(x, network, t, [], [], :cross)
  end

  def ngrad(x, network, t) do
    ngrad1(x, network, t, [], [])
  end

  def ngrad1(_, [], _, _, res) do
    Enum.reverse(res)
  end

  # 1st arg is input data matrix
  # 2nd arg is network list
  # 3rd arg is teacher data matrix
  # 4th arg is network list that is already calculeted
  # 5th arg is generated network list with calculated gradient 
  def ngrad1(x, [{:weight, w, lr, v} | rest], t, before, res) do
    w1 = ngrad_mt(x, w, t, before, {:weight, w, lr, v}, rest)
    ngrad1(x, rest, t, [{:weight, w1, lr, v} | before], [{:weight, w1, lr, v} | res])
  end

  def ngrad1(x, [{:bias, w, lr, v} | rest], t, before, res) do
    w1 = ngrad_mt(x, w, t, before, {:bias, w, lr, v}, rest)
    ngrad1(x, rest, t, [{:bias, w, lr, v} | before], [{:bias, w1, lr, v} | res])
  end

  def ngrad1(x, [y | rest], t, before, res) do
    ngrad1(x, rest, t, [y | before], [y | res])
  end

  def ngrad1(_, [], _, _, res, :cross) do
    Enum.reverse(res)
  end

  def ngrad1(x, [{:weight, w, lr, v} | rest], t, before, res, :cross) do
    w1 = ngrad_mt(x, w, t, before, {:weight, w, lr, v}, rest, :cross)

    ngrad1(
      x,
      rest,
      t,
      [{:weight, w, lr, v} | before],
      [{:weight, w1, lr, v} | res],
      :cross
    )
  end

  def ngrad1(x, [{:bias, w, lr, v} | rest], t, before, res, :cross) do
    w1 = ngrad_mt(x, w, t, before, {:bias, w, lr, v}, rest, :cross)

    ngrad1(
      x,
      rest,
      t,
      [{:bias, w, lr, v} | before],
      [{:bias, w1, lr, v} | res],
      :cross
    )
  end

  def ngrad1(x, [y | rest], t, before, res, :cross) do
    ngrad1(x, rest, t, [y | before], [y | res], :cross)
  end

 
  # calculate numerical gradient of weigth,bias matrix
  # naively calculete each element of wight or bias element 
  # 7th arg is option for closs-entropy loss function
  def ngrad_mt(x, w, t, before, now, rest) do
    {r, c} = CM.size(w)

    Enum.map(
      1..r,
      fn x1 ->
        Enum.map(
          1..c,
          fn y1 -> ngrad_mt1(x, t, x1, y1, before, now, rest) end
        )
      end
    )
    |> CM.new()
  end

  def ngrad_mt(x, w, t, before, now, rest, :cross) do
    {r, c} = CM.size(w)

    Enum.map(
      1..r,
      fn x1 ->
        Enum.map(
          1..c,
          fn y1 -> ngrad_mt1(x, t, x1, y1, before, now, rest, :cross) end
        )
      end
    )
    |> CM.new()
  end

  def ngrad_mt1(x, t, r, c, before, {type, w, lr, v}, rest) do
    h = 0.0001
    w1 = CM.minus(w, r, c, h)
    network0 = Enum.reverse(before) ++ [{type, w, lr, v}] ++ rest
    network1 = Enum.reverse(before) ++ [{type, w1, lr, v}] ++ rest
    [y0|_] = forward(x, network0, [])
    [y1|_] = forward(x, network1, [])
    (CM.loss(y1, t, :square) - CM.loss(y0, t, :square)) / h
  end

  def ngrad_mt1(x, t, r, c, before, {type, w, st, lr, v}, rest) do
    h = 0.0001
    w1 = CM.minus(w, r, c, h)
    network0 = Enum.reverse(before) ++ [{type, w, st, lr, v}] ++ rest
    network1 = Enum.reverse(before) ++ [{type, w1, st, lr, v}] ++ rest
    [y0|_] = forward(x, network0,[])
    [y1|_] = forward(x, network1,[])
    (CM.loss(y1, t, :square) - CM.loss(y0, t, :square)) / h
  end


  defp ngrad_mt1(x, t, r, c, before, {type, w, lr, v}, rest, :cross) do
    h = 0.0001
    w1 = CM.minus(w, r, c, h)
    network0 = Enum.reverse(before) ++ [{type, w, lr, v}] ++ rest
    network1 = Enum.reverse(before) ++ [{type, w1, lr, v}] ++ rest
    [y0 | _] = forward(x, network0,[])
    [y1 | _] = forward(x, network1,[])
    (CM.loss(y1, t, :cross) - CM.loss(y0, t, :cross)) / h
  end

  defp ngrad_mt1(x, t, r, c, before, {type, w, st, lr, v}, rest, :cross) do
    h = 0.0001
    w1 = CM.minus(w, r, c, h)
    network0 = Enum.reverse(before) ++ [{type, w, st, lr, v}] ++ rest
    network1 = Enum.reverse(before) ++ [{type, w1, st, lr, v}] ++ rest
    [y0 | _] = forward(x, network0,[])
    [y1 | _] = forward(x, network1,[])
    (CM.loss(y1, t, :cross) - CM.loss(y0, t, :cross)) / h
  end

  # gradient with backpropagation
  # 1st arg is input data matrix
  # 2nd arg is network list
  # 3rd arg is teeacher matrix
  def grad(x, network, t) do
    [x1|x2] = forward(x, network, [x])
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
    w1 = CM.mult(CM.transpose(u), l) |> CM.emult(1 / n)
    l1 = CM.mult(l, CM.transpose(w))
    backward(l1, rest, us, [{:weight, w1, lr, v} | res])
  end
end

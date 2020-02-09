defmodule Deeppipe2 do
  # common
  def is_matrix({r,c,dt}) do
    if is_integer(r) && is_integer(c) && is_list(dt) do
      true 
    else 
      false
    end 
  end

  # forward
  def forward(x, []) do
    x
  end

  def forward(x, [{:weight, w, _, _} | rest]) do
    Cumatrix.mult(x, w) |> forward(rest)
  end

  def forward(x, [{:bias, b, _, _} | rest]) do
    Cumatrix.add(x, b) |> forward(rest)
  end

  def forward(x, [{:function, name} | rest]) do
    cond do
      name == :sigmoid -> Cumatrix.activate(x,:sigmoid) |> forward(rest)
      name == :tanh -> Cumatrix.activate(x,:tanh) |> forward(rest)
      name == :relu -> Cumatrix.activate(x,:relu) |> forward(rest)
      name == :softmax -> Cumatrix.activate(x,:softmax) |> forward(rest)
      true -> raise "not exist function" 
    end 
  end

  # forward for backpropagation
  # this store all middle data
  def forward_for_back(_, [], res) do
    res
  end

  def forward_for_back(x, [{:weight, w, _, _} | rest], res) do
    x1 = Cumatrix.mult(x, w)
    forward_for_back(x1, rest, [x1 | res])
  end

  def forward_for_back(x, [{:bias, b, _, _} | rest], res) do
    x1 = Cumatrix.add(x, b)
    forward_for_back(x1, rest, [x1 | res])
  end

  def forward_for_back(x, [{:function, name} | rest], res) do
    cond do
      name == :sigmoid -> 
        x1 = Cumatrix.activate(x,:sigmoid)
        forward_for_back(x1, rest, [x1 | res])
      name == :tanh -> 
        x1 = Cumatrix.activate(x,:tanh)
        forward_for_back(x1, rest, [x1 | res])
      name == :relu -> 
        x1 = Cumatrix.activate(x,:relu)
        forward_for_back(x1, rest, [x1 | res])
      name == :softmax -> 
        x1 = Cumatrix.activate(x,:softmax)
        forward_for_back(x1, rest, [x1 | res])
      true -> 
        raise "not exist function"
    end 
  end


# numerical gradient
  def numerical_gradient(x, network, t, :square) do
    numerical_gradient(x, network, t)
  end

  def numerical_gradient(x, network, t, :cross) do
    numerical_gradient1(x, network, t, [], [], :cross)
  end

  def numerical_gradient(x, network, t) do
    numerical_gradient1(x, network, t, [], [])
  end

  def numerical_gradient1(_, [], _, _, res) do
    Enum.reverse(res)
  end

  def numerical_gradient1(x, [{:weight, w, lr, v} | rest], t, before, res) do
    w1 = numerical_gradient_matrix(x, w, t, before, {:weight, w, lr, v}, rest)
    numerical_gradient1(x, rest, t, [{:weight, w1, lr, v} | before], [{:weight, w1, lr, v} | res])
  end

  def numerical_gradient1(x, [{:bias, w, lr, v} | rest], t, before, res) do
    w1 = numerical_gradient_matrix(x, w, t, before, {:bias, w, lr, v}, rest)
    numerical_gradient1(x, rest, t, [{:bias, w, lr, v} | before], [{:bias, w1, lr, v} | res])
  end

  def numerical_gradient1(x, [y | rest], t, before, res) do
    numerical_gradient1(x, rest, t, [y | before], [y | res])
  end

  # calc numerical gradient of weigth,bias matrix
  def numerical_gradient_matrix(x, w, t, before, now, rest) do
    {r, c} = Cumatrix.size(w)

    Enum.map(
      1..r,
      fn x1 ->
        Enum.map(
          1..c,
          fn y1 -> numerical_gradient_matrix1(x, t, x1, y1, before, now, rest) end
        )
      end
    )
    |> Cumatrix.new()
  end

  def numerical_gradient_matrix1(x, t, r, c, before, {type, w, lr, v}, rest) do
    h = 0.0001
    w1 = Cumatrix.minus(w, r, c, h)
    network0 = Enum.reverse(before) ++ [{type, w, lr, v}] ++ rest
    network1 = Enum.reverse(before) ++ [{type, w1, lr, v}] ++ rest
    y0 = forward(x, network0)
    y1 = forward(x, network1)
    (Cumatrix.loss(y1, t, :square) - Cumatrix.loss(y0, t, :square)) / h
  end

  def numerical_gradient_matrix1(x, t, r, c, before, {type, w, st, lr, v}, rest) do
    h = 0.0001
    w1 = Cumatrix.minus(w, r, c, h)
    network0 = Enum.reverse(before) ++ [{type, w, st, lr, v}] ++ rest
    network1 = Enum.reverse(before) ++ [{type, w1, st, lr, v}] ++ rest
    y0 = forward(x, network0)
    y1 = forward(x, network1)
    (Cumatrix.loss(y1, t, :square) - Cumatrix.loss(y0, t, :square)) / h
  end

  def numerical_gradient1(_, [], _, _, res, :cross) do
    Enum.reverse(res)
  end

  
  def numerical_gradient1(x, [{:weight, w, lr, v} | rest], t, before, res, :cross) do
    w1 = numerical_gradient_matrix(x, w, t, before, {:weight, w, lr, v}, rest, :cross)

    numerical_gradient1(
      x,
      rest,
      t,
      [{:weight, w, lr, v} | before],
      [{:weight, w1, lr, v} | res],
      :cross
    )
  end

  def numerical_gradient1(x, [{:bias, w, lr, v} | rest], t, before, res, :cross) do
    w1 = numerical_gradient_matrix(x, w, t, before, {:bias, w, lr, v}, rest, :cross)

    numerical_gradient1(
      x,
      rest,
      t,
      [{:bias, w, lr, v} | before],
      [{:bias, w1, lr, v} | res],
      :cross
    )
  end

  def numerical_gradient1(x, [y | rest], t, before, res, :cross) do
    numerical_gradient1(x, rest, t, [y | before], [y | res], :cross)
  end


# calc numerical gradient of filter,weigth,bias matrix
  defp numerical_gradient_matrix(x, w, t, before, now, rest, :cross) do
    {r, c} = w[:size]

    Enum.map(
      1..r,
      fn x1 ->
        Enum.map(
          1..c,
          fn y1 -> numerical_gradient_matrix1(x, t, x1, y1, before, now, rest, :cross) end
        )
      end
    )
    |> Cumatrix.new()
  end

  defp numerical_gradient_matrix1(x, t, r, c, before, {type, w, lr, v}, rest, :cross) do
    h = 0.0001
    w1 = Cumatrix.minus(w, r, c, h)
    network0 = Enum.reverse(before) ++ [{type, w, lr, v}] ++ rest
    network1 = Enum.reverse(before) ++ [{type, w1, lr, v}] ++ rest
    y0 = forward(x, network0)
    y1 = forward(x, network1)
    (Cumatrix.loss(y1, t, :cross) - Cumatrix.loss(y0, t, :cross)) / h
  end

  defp numerical_gradient_matrix1(x, t, r, c, before, {type, w, st, lr, v}, rest, :cross) do
    h = 0.0001
    w1 = Cumatrix.minus(w, r, c, h)
    network0 = Enum.reverse(before) ++ [{type, w, st, lr, v}] ++ rest
    network1 = Enum.reverse(before) ++ [{type, w1, st, lr, v}] ++ rest
    y0 = forward(x, network0)
    y1 = forward(x, network1)
    (Cumatrix.loss(y1, t, :cross) - Cumatrix.loss(y0, t, :cross)) / h
  end

  # backpropagation
  defp backpropagation(_, [], _, res) do
    res
  end

  defp backpropagation(l, [{:function, :softmax} | rest], [_ | us], res) do
    backpropagation(l, rest, us, [{:function,:softmax} | res])
  end

  defp backpropagation(l, [{:function,name} | rest], [u | us], res) do
      l1 = Cumatrix.diff(l, u, name)
      backpropagation(l1, rest, us, [{:function, name} | res])
  end

  defp backpropagation(l, [{:bias, _, lr, v} | rest], [_ | us], res) do
    b1 = Cumatrix.average(l)
    backpropagation(l, rest, us, [{:bias, b1, lr, v} | res])
  end

  defp backpropagation(l, [{:weight, w, lr, v} | rest], [u | us], res) do
    {n, _} = Cumatrix.size(l)
    w1 = Cumatrix.mult(Cumatrix.transpose(u), l) |>  Cumatrix.emult(1/n)
    l1 = Cumatrix.mult(l, Cumatrix.transpose(w))
    backpropagation(l1, rest, us, [{:weight, w1, lr, v} | res])
  end
  
end

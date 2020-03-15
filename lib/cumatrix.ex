# test nvcc
defmodule Cumatrix do
  @on_load :load_nifs

  def load_nifs do
    :erlang.load_nif('./lib/nifs', 0)
  end

  def print1(_a, _b, _c) do
    raise "NIF print1/3 not implemented"
  end

  def new1(_a, _b) do
    raise "NIF new1/2 not implemented"
  end

  def new2(_a, _b, _c) do
    raise "NIF new2/3 not implemented"
  end

  def new3(_a, _b, _c, _d) do
    raise "NIF new3/4 not implemented"
  end

  def new4(_a, _b, _c, _d, _e) do
    raise "NIF new4/5 not implemented"
  end

  def rand1(_a) do
    raise "NIF rand1/1 not implemented"
  end

  def mult1(_a, _b, _c, _d, _e, _f) do
    raise "NIF mult/6 not implemented"
  end

  def add1(_a, _b, _c, _d) do
    raise "NIF add1/4 not implemented"
  end

  def sub1(_a, _b, _c, _d) do
    raise "NIF sub1/4 not implemented"
  end

  def emult1(_a, _b, _c, _d) do
    raise "NIF emult1/4 not implemented"
  end

  def transpose1(_a, _b, _c) do
    raise "NIF transpose1/3 not implemented"
  end

  def ident1(_a) do
    raise "NIF ident1/3 not implemented"
  end

  def activate_sigmoid(_a, _b, _c) do
    raise "NIF activate_sigmoid/3 not implemented"
  end

  def activate_tanh(_a, _b, _c) do
    raise "NIF activate_tanh/3 not implemented"
  end

  def activate_relu(_a, _b, _c) do
    raise "NIF activate_relu/3 not implemented"
  end

  def activate_softmax(_a, _b, _c) do
    raise "NIF activate_softmax/3 not implemented"
  end

  def differ_sigmoid(_a, _b, _c, _d) do
    raise "NIF differ_sigmoid/4 not implemented"
  end

  def differ_tanh(_a, _b, _c, _d) do
    raise "NIF differ_tanh/4 not implemented"
  end

  def differ_relu(_a, _b, _c, _d) do
    raise "NIF differ_relu/4 not implemented"
  end

  def smult1(_a, _b, _c) do
    raise "NIF smult1/3 not implemented"
  end

  def trace1(_a, _b, _c) do
    raise "NIF trace1/3 not implemented"
  end

  def mean_square(_a, _b, _c, _d) do
    raise "NIF mean_square/4 not implemented"
  end

  def cross_entropy(_a, _b, _c, _d) do
    raise "NIF mean_square/4 not implemented"
  end

  def elt1(_a, _b, _c, _d, _e) do
    raise "NIF elt1/5 not implemented"
  end

  def set1(_a, _b, _c, _d, _e, _f) do
    raise "NIF set1/6 not implemented"
  end

  def average1(_a, _b, _c) do
    raise "NIF average1/3 not implemented"
  end

  def sum1(_a, _b, _c) do
    raise "NIF sum1/3 not implemented"
  end

  def to_list1(_a, _b, _c) do
    raise "NIF to_list1/3 not implemented"
  end

  def to_list2(_a, _b, _c, _d) do
    raise "NIF to_list2/4 not implemented"
  end

  def to_list3(_a, _b, _c, _d, _e) do
    raise "NIF to_list3/5 not implemented"
  end

  def momentum1(_a, _b, _c, _d, _e) do
    raise "NIF momentum1/5 not implemented"
  end

  def adagrad1(_a, _b, _c, _d, _e, _f) do
    raise "NIF adagrad1/6 not implemented"
  end

  def accuracy1(_a, _b, _c, _d) do
    raise "NIF accuracy/4 not implemented"
  end

  def pooling1(_1, _2, _3, _4, _5, _6) do
    raise "NIF pooling1/6 not implemented"
  end

  def unpooling1(_1, _2, _3, _4, _5, _6, _7) do
    raise "NIF unpooling1/7 not implemented"
  end

  def convolute1(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10) do
    raise "NIF convolute1/10 not implemented"
  end

  def deconvolute1(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10) do
    raise "NIF deconvolute1/10 not implemented"
  end

  def gradfilter1(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11) do
    raise "NIF gradfilter1/11 not implemented"
  end

  def full1(_1, _2, _3, _4) do
    raise "NIF full1/4 not implemented"
  end

  def unfull1(_1, _2, _3, _4) do
    raise "NIF unfull1/4 not implemented"
  end

  # ----------------------------------------------------------------
  def mult({r1, c1, dt1}, {r2, c2, dt2}) do
    {r1, c2, mult1(r1, c1, dt1, r2, c2, dt2)}
  end

  def mult(s, {r, c, dt}) when is_float(s) do
    {r, c, smult1(s, r * c, dt)}
  end

  def mult({r, c, dt}, s) when is_float(s) do
    {r, c, smult1(s, r * c, dt)}
  end

  def mult(s, {c, h, w, dt}) when is_float(s) do
    {c, h, w, smult1(s, c * h * w, dt)}
  end

  def mult({c, h, w, dt}, s) when is_float(s) do
    {c, h, w, smult1(s, c * h * w, dt)}
  end

  def mult(s, {n, c, h, w, dt}) when is_float(s) do
    {n, c, h, w, smult1(s, n * c * h * w, dt)}
  end

  def mult({n, c, h, w, dt}, s) when is_float(s) do
    {n, c, h, w, smult1(s, n * c * h * w, dt)}
  end

  def mult(_, _) do
    raise "mult illegal data type"
  end

  def new(r, c) do
    {r, c, new1(r * c, 0.0)}
  end

  def new(r, c, val) do
    {r, c, new1(r * c, val)}
  end

  # list to matrix
  def new(ls) when is_list(ls) do
    cond do
      list_dim(ls) == 2 ->
        r = length(ls)
        c = length(hd(ls))
        ls1 = ls |> flatten()
        {r, c, new2(r, c, ls1)}

      list_dim(ls) == 3 ->
        c = length(ls)
        h = length(hd(ls))
        w = length(hd(hd(ls)))
        ls1 = ls |> flatten()
        {c, h, w, new3(c, h, w, ls1)}

      list_dim(ls) == 4 ->
        n = length(ls)
        c = length(hd(ls))
        h = length(hd(hd(ls)))
        w = length(hd(hd(hd(ls))))
        ls1 = ls |> flatten()
        {n, c, h, w, new4(n, c, h, w, ls1)}
    end
  end

  def rand(r, c) do
    {r, c, rand1(r * c)}
  end

  def rand(c, h, w) do
    {c, h, w, rand1(c * h * w)}
  end

  def rand(n, c, h, w) do
    {n, c, h, w, rand1(n * c * h * w)}
  end

  def add({r1, c1, dt1}, {r1, c1, dt2}) do
    {r1, c1, add1(r1, c1, dt1, dt2)}
  end

  def add({r1, c1, dt1}, {1, c1, dt2}) do
    {r1, c1, add1(r1, c1, dt1, expand({r1, c1, dt2}))}
  end

  def add({1, c1, dt1}, {r1, c1, dt2}) do
    {r1, c1, add1(r1, c1, expand({r1, c1, dt1}), dt2)}
  end

  def add(_, _) do
    raise "add illegal data type"
  end

  def expand({r, c, dt}) do
    dt1 = expand1(r, dt)
    transpose1(r, c, dt1)
  end

  def expand1(0, _) do
    <<>>
  end

  def expand1(n, dt) do
    dt <> expand1(n - 1, dt)
  end

  def sub({r1, c1, dt1}, {r1, c1, dt2}) do
    {r1, c1, sub1(r1, c1, dt1, dt2)}
  end

  def sub(_, _) do
    raise "sub illegal data type"
  end

  def emult({r1, c1, dt1}, {r1, c1, dt2}) do
    {r1, c1, emult1(r1, c1, dt1, dt2)}
  end

  def emult(_, _) do
    raise "emult ilegal data type"
  end

  def elt({r, c, dt}, x, y) do
    elt1(r, c, x - 1, y - 1, dt)
  end

  def set({r, c, dt}, x, y, val) do
    {r, c, set1(r, c, dt, x - 1, y - 1, val)}
  end

  @doc """
  iex(1)> Cumatrix.flatten([[1,2],[3,4]])
  [1, 2, 3, 4]
  """
  def flatten([]) do
    []
  end

  def flatten([l | ls]) do
    if is_nestlist(l) do
      flatten(l) ++ flatten(ls)
    else
      l ++ flatten(ls)
    end
  end

  def is_nestlist([l | _]) do
    if is_list(l) do
      true
    else
      false
    end
  end

  @doc """
  iex(1)>  Cumatrix.list_dim([[1,2],[3,4]])
  2
  iex(2)>  Cumatrix.list_dim([[[1,2],[2,3]]])      
  3
  """
  def list_dim([l | _]) do
    if is_list(l) do
      1 + list_dim(l)
    else
      1
    end
  end

  def transpose({r, c, dt}) do
    {c, r, transpose1(r, c, dt)}
  end

  def ident(r) do
    if !is_number(r) || !is_number(r) || r <= 0 do
      raise "ident illegal size"
    end

    {r, r, ident1(r)}
  end

  def activate({r, c, dt}, :sigmoid) do
    {r, c, activate_sigmoid(r, c, dt)}
  end

  def activate({r, c, dt}, :tanh) do
    {r, c, activate_tanh(r, c, dt)}
  end

  def activate({r, c, dt}, :relu) do
    {r, c, activate_relu(r, c, dt)}
  end

  def activate({r, c, dt}, :softmax) do
    {r, c, activate_softmax(r, c, dt)}
  end

  def activate(_, _) do
    raise "activate illegal argument"
  end

  def diff({r, c, dt1}, {r, c, dt2}, :sigmoid) do
    {r, c, differ_sigmoid(r, c, dt1, dt2)}
  end

  def diff({r, c, dt1}, {r, c, dt2}, :tanh) do
    {r, c, differ_tanh(r, c, dt1, dt2)}
  end

  def diff({r, c, dt1}, {r, c, dt2}, :relu) do
    {r, c, differ_relu(r, c, dt1, dt2)}
  end

  def diff(_, _, _) do
    raise "differ illegal argument"
  end

  def size({r, c, _}) do
    {r, c}
  end

  def size({c, h, w, _}) do
    {c, h, w}
  end

  def size({n, c, h, w, _}) do
    {n, c, h, w}
  end

  def average({r, c, dt}) do
    {1, c, average1(r, c, dt)}
  end

  def sum({r, c, dt}) do
    sum1(r, c, dt)
  end

  def to_list({r, c, dt}) do
    to_list1(r, c, dt) |> Enum.chunk_every(c)
  end

  def to_list({c, h, w, dt}) do
    to_list2(c, h, w, dt)
    |> conv_dim([c, h, w])
  end

  def to_list({n, c, h, w, dt}) do
    to_list3(n, c, h, w, dt)
    |> conv_dim([n, c, h, w])
  end

  def conv_dim(ls, [_]) do
    ls
  end

  def conv_dim(ls, [d | ds]) do
    dim = div(length(ls), d)
    Enum.chunk_every(ls, dim) |> Enum.map(fn x -> conv_dim(x, ds) end)
  end

  def trace({r, c, dt}) do
    if r != c do
      raise "trace not square matrix"
    end

    trace1(r, c, dt)
  end

  def loss({r1, c1, dt1}, {r1, c1, dt2}, :square) do
    mean_square(r1, c1, dt1, dt2)
  end

  def loss({r1, c1, dt1}, {r1, c1, dt2}, :cross) do
    cross_entropy(r1, c1, dt1, dt2)
  end

  def momentum({r1, c1, dt1}, {r1, c1, dt2}, lr) do
    {r1, c1, momentum1(r1, c1, dt1, dt2, lr)}
  end

  def momentum(_, _, _) do
    raise "momentum illegal argument"
  end

  def adagrad({r1, c1, dt1}, {r1, c1, dt2}, h, lr) do
    {r1, c1, adagrad1(r1, c1, dt1, dt2, h, lr)}
  end

  def adagrad(_, _, _, _) do
    raise "adagrad illegal argument"
  end

  def accuracy({r1, c1, dt1}, ls) do
    if length(ls) != r1 do
      raise "accuracy illegal argument"
    else
      accuracy1(r1, c1, dt1, ls)
    end
  end

  def accurace(_, _) do
    raise "accuracy illegal argument"
  end

  def print({r, c, dt}) do
    print1(r, c, dt)
  end

  def pooling({n, c, h, w, dt}, st) do
    if rem(h, st) != 0 || rem(w, st) != 0 do
      raise "pooling illegal argument"
    else
      {n, c, div(h, st), div(w, st), pooling1(n, c, h, w, dt, st)}
    end
  end

  def unpooling({n1, c1, h1, w1, d1}, {n1, _, _, _, d2}, st) do
    unpooling1(n1, c1, h1, w1, d1, d2, st)
  end

  def convolute({n, c, h1, w1, dt1}, {c, h2, w2, dt2}, st, pad) do
    oh = div(h1 + 2 * pad - h2, st) + 1
    ow = div(w1 + 2 * pad - w2, st) + 1
    {n, 1, oh, ow, convolute1(n, c, h1, w1, h2, w2, dt1, dt2, st, pad)}
  end

  def deconvolute({n, c, oh, ow, dt1}, {c, h2, w2, dt2}, st, pad) do
    h1 = (oh-1)*st - 2*pad + h2
    w1 = (ow-1)*st - 2*pad + h2
    {n, c, h1, w1, deconvolute1(n, c, h1, w1, h2, w2, dt1, dt2, st, pad)}
  end

  def gradfilter({n1, c1, h1, w1, dt1}, {c1, h2, w2, dt2}, {n1, c1, _, _, dt3}, st, pad) do
    {c1,h2,w2,gradfilter1(n1, c1, h1, w1, h2, w2, dt1, dt2, dt3, st, pad)}
  end
  def gradfilter(_,_,_,_) do
    raise "gradfilter illegal data form"
  end 

  def full({n1, 1, h1, w1, dt1}) do
    {n1, h1 * w1, full1(n1, h1, w1, dt1)}
  end

  def unfull({r, _, dt1}, h, w) do
    {r, 1, h, w, unfull1(r, h, w, dt1)}
  end

  def is_matrix({r, c, dt}) do
    if is_integer(r) && is_integer(c) && is_binary(dt) do
      true
    else
      false
    end
  end

  def is_matrix(_) do
    false
  end

  def is_tensor({n, c, h, w, dt}) do
    if is_integer(n) && is_integer(c) && is_integer(h) && is_integer(w) && is_binary(dt) do
      true
    else
      false
    end
  end

  def is_tensor(_) do
    false
  end
end

defmodule Time do
  @moduledoc """
  macro for measure execution time
  """
  defmacro time(exp) do
    quote do
      {time, dict} = :timer.tc(fn -> unquote(exp) end)
      IO.inspect("time: #{time} micro second")
      IO.inspect("-------------")
      dict
    end
  end
end

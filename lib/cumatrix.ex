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

  def badd1(_a, _b, _c, _d) do
    raise "NIF badd1/4 not implemented"
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

  def smult1(_a, _b, _c, _d) do
    raise "NIF smult1/4 not implemented"
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

  def sum1(_a, _b, _c) do
    raise "NIF sum1/3 not implemented"
  end 

  def to_list1(_a, _b, _c) do
    raise "NIF to_list1/3 not implemented"
  end 


#----------------------------------------------------------------
  def mult({r1, c1, dt1}, {r2, c2, dt2}) do
    {r1, c2, mult1(r1, c1, dt1, r2, c2, dt2)}
  end
  def mult(s,{r,c,dt}) when is_float(s) do
    {r,c, smult1(s,r,c,dt)}
  end
  def mult({r,c,dt},s) when is_float(s) do
    {r,c, smult1(s,r,c,dt)}
  end
  def mult(_,_) do
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
    r = length(ls)
    c = length(hd(ls))
    ls1 = ls |> to_col_vec() |> to_flat_vec()
    {r,c,new2(r,c,ls1)}
  end

  def rand(r, c) do
    {r, c, rand1(r * c)}
  end

  def add({r1, c1, dt1}, {r1, c1, dt2}) do
    {r1, c1, add1(r1, c1, dt1, dt2)}
  end
  def add({r1, c1, dt1}, {1, c1, dt2}) do
    {r1,c1, add1(r1,c1,dt1,expand({r1,c1,dt2}))}
  end 
  def add({1, c1, dt1}, {r1, c1, dt2}) do
    {r1,c1, add1(r1,c1,expand({r1,c1,dt1}),dt2)}
  end 
  def add(_,_) do 
    raise "add illegal data type"
  end 

  def expand({r,c,dt}) do
    dt1 = expand1(r,dt)
    transpose1(r,c,dt1)
  end 
  def expand1(0,_) do <<>> end 
  def expand1(n,dt) do
    dt <> expand1(n-1, dt)
  end 

  def sub({r1, c1, dt1}, {r1, c1, dt2}) do
    {r1, c1, sub1(r1, c1, dt1, dt2)}
  end
  def sub(_,_) do 
    raise "sub illegal data type"
  end 



  def emult({r1, c1, dt1}, {r2, c2, dt2}) do
    if r1 != r2 || c1 != c2 do
      raise "emult size mismatch"
    end

    {r1, c1, emult1(r1, c1, dt1, dt2)}
  end

  def elt({r, c, dt}, x, y) do
    elt1(r,c,x-1,y-1,dt)
  end

  @doc """
  iex(1)> Cumatrix.to_col_vec([[1,2],[3,4]])
  [[1, 3], [2, 4]]
  """
  def to_col_vec(ls) do
    to_col_vec1(ls, 0, length(hd(ls)))
  end

  def to_col_vec1(_, pos, pos) do
    []
  end

  def to_col_vec1(ls, pos, max) do
    [Enum.map(ls, fn x -> Enum.at(x, pos) end) | to_col_vec1(ls, pos + 1, max)]
  end

  @doc """
  iex(1)> Cumatrix.to_flat_vec([[1,2],[3,4]])
  [1, 2, 3, 4]
  """
  def to_flat_vec([]) do
    []
  end

  def to_flat_vec([l | ls]) do
    l ++ to_flat_vec(ls)
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

  def sum({r, c, dt}) do
    sum1(r,c,dt)
  end

  def to_list({r,c,dt}) do
    to_list1(r,c,dt) |> Enum.chunk_every(c)
  end 

  def trace({r, c, dt}) do
    if r != c do
      raise "trace not square matrix"
    end

    trace1(r, c, dt)
  end

  def loss({r1, c1, dt1}, {r1, c1, dt2}, :square) do
    {r1, 1, mean_square(r1, c1, dt1, dt2)}
  end

  def loss({r1, c1, dt1}, {r1, c1, dt2}, :cross) do
    {r1, 1, cross_entropy(r1, c1, dt1, dt2)}
  end

  def print({r, c, dt}) do
    print1(r,c,dt)
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

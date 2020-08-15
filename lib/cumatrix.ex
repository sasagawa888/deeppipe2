defmodule Cumatrix do
  @moduledoc """
  Calculate matrix or tensor using CUDA and CUBLAS library.

  Caution, each element of matrix  must be float number.

  tensor data structure is 4-dimensions tensor (N,C,H,W) or 3-dimension tensor (C,H,W)
  N is mini batch size.
  C is channel.
  H is hight of image.
  W is width of image.

  error code
  N<10000  bad argument error   N is argument number. 
  10000<= N <11000 CUDA error   N-10000 is error code of CUDA.
  11000 < N  cuBLAS error  N-11000 is error code of cuBLAS.
  """

  @on_load :load_nifs

  def load_nifs do
    priv_dir =
      case :code.priv_dir(:deeppipe2) do
        {:error, _} -> "priv"
        path -> path
      end

    file = :filename.join(priv_dir, "nifs")
    :erlang.load_nif(String.to_charlist(file), 0)
  end

  defp new1(_1, _2) do
    raise "NIF new1/2 not implemented"
  end

  defp new2(_1, _2, _3) do
    raise "NIF new2/3 not implemented"
  end

  defp new3(_1, _2, _3, _4) do
    raise "NIF new3/4 not implemented"
  end

  defp new4(_1, _2, _3, _4, _5) do
    raise "NIF new4/5 not implemented"
  end

  defp rand1(_1) do
    raise "NIF rand1/1 not implemented"
  end

  defp mult1(_1, _2, _3, _4, _5, _6) do
    raise "NIF mult/6 not implemented"
  end

  defp add1(_1, _2, _3) do
    raise "NIF add1/3 not implemented"
  end

  defp sub1(_1, _2, _3) do
    raise "NIF sub1/3 not implemented"
  end

  defp emult1(_1, _2, _3) do
    raise "NIF emult1/3 not implemented"
  end

  defp ediv1(_1, _2, _3) do
    raise "NIF div1/3 not implemented"
  end

  defp transpose1(_1, _2, _3) do
    raise "NIF transpose1/3 not implemented"
  end

  defp ident1(_1) do
    raise "NIF ident1/3 not implemented"
  end

  defp activate_sigmoid(_1, _2) do
    raise "NIF activate_sigmoid/2 not implemented"
  end

  defp activate_tanh(_1, _2) do
    raise "NIF activate_tanh/2 not implemented"
  end

  defp activate_relu(_1, _2) do
    raise "NIF activate_relu/2 not implemented"
  end

  defp activate_softmax(_1, _2, _3) do
    raise "NIF activate_softmax/3 not implemented"
  end

  defp differ_sigmoid(_1, _2, _3) do
    raise "NIF differ_sigmoid/3 not implemented"
  end

  defp differ_tanh(_1, _2, _3) do
    raise "NIF differ_tanh/3 not implemented"
  end

  defp differ_relu(_1, _2, _3) do
    raise "NIF differ_relu/3 not implemented"
  end

  defp smult1(_1, _2, _3) do
    raise "NIF smult1/3 not implemented"
  end

  defp trace1(_1, _2, _3) do
    raise "NIF trace1/3 not implemented"
  end

  defp mean_square(_1, _2, _3, _4) do
    raise "NIF mean_square/4 not implemented"
  end

  defp cross_entropy(_1, _2, _3, _4) do
    raise "NIF mean_square/4 not implemented"
  end

  defp elt1(_1, _2, _3, _4, _5) do
    raise "NIF elt1/5 not implemented"
  end

  defp set1(_1, _2, _3, _4, _5, _6) do
    raise "NIF set1/6 not implemented"
  end

  defp add_diff1(_1, _2, _3, _4, _5, _6) do
    raise "NIF add_diff1/6 not implemented"
  end

  defp add_diff2(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10) do
    raise "NIF add_diff2/10 not implemented"
  end

  defp average1(_1, _2, _3) do
    raise "NIF average1/3 not implemented"
  end

  defp sum1(_1, _2, _3) do
    raise "NIF sum1/3 not implemented"
  end

  defp to_list1(_1, _2, _3) do
    raise "NIF to_list1/3 not implemented"
  end

  defp to_list2(_1, _2, _3, _4) do
    raise "NIF to_list2/4 not implemented"
  end

  defp to_list3(_1, _2, _3, _4, _5) do
    raise "NIF to_list3/5 not implemented"
  end

  defp dropout1(_1, _2) do
    raise "NIF dropout1/2 not implemented"
  end

  defp sgd1(_1, _2, _3, _4) do
    raise "NIF sgd1/4 not implemented"
  end

  defp momentum1(_1, _2, _3, _4, _5) do
    raise "NIF momentum1/5 not implemented"
  end

  defp adagrad1(_1, _2, _3, _4, _5) do
    raise "NIF adagrad1/5 not implemented"
  end

  defp rms1(_1, _2, _3, _4, _5) do
    raise "NIF rms1/5 not implemented"
  end

  defp adam1(_1, _2, _3, _4, _5, _6) do
    raise "NIF adam1/5 not implemented"
  end

  defp accuracy1(_1, _2, _3, _4) do
    raise "NIF accuracy/4 not implemented"
  end

  defp correct1(_1, _2, _3, _4) do
    raise "NIF accuracy/4 not implemented"
  end

  defp pooling1(_1, _2, _3, _4, _5, _6, _7) do
    raise "NIF pooling1/7 not implemented"
  end

  defp unpooling1(_1, _2, _3, _4, _5, _6, _7, _8) do
    raise "NIF unpooling1/8 not implemented"
  end

  defp convolute1(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13) do
    raise "NIF convolute1/13 not implemented"
  end

  defp deconvolute1(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13) do
    raise "NIF deconvolute1/13 not implemented"
  end

  defp deconvolute2(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13) do
    raise "NIF deconvolute2/13 not implemented"
  end

  defp gradfilter1(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16) do
    raise "NIF gradfilter1/16 not implemented"
  end

  defp gradfilter2(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16) do
    raise "NIF gradfilter2/16 not implemented"
  end

  defp full1(_1, _2, _3, _4, _5) do
    raise "NIF full1/5 not implemented"
  end

  defp unfull1(_1, _2, _3, _4, _5) do
    raise "NIF unfull1/5 not implemented"
  end

  defp random_select1(_1, _2, _3, _4, _5, _6, _7) do
    raise "NIF random_select1/7 not implemented"
  end

  defp random_select2(_1, _2, _3, _4, _5, _6, _7, _8, _9) do
    raise "NIF random_select2/9 not implemented"
  end

  defp random_select3(_1, _2, _3, _4, _5, _6, _7, _8) do
    raise "NIF random_select3/8 not implemented"
  end

  defp is_near1(_1, _2, _3) do
    raise "NIF is_near1/3 not implemented"
  end

  defp is_equal1(_1, _2, _3) do
    raise "NIF is_equal1/3 not implemented"
  end

  defp analizer1(_1, _2, _3) do
    raise "NIF analizer1/3 not implemented"
  end

  defp standardize1(_1, _2, _3, _4, _5) do
    raise "NIF normalizer1/3 not implemented"
  end

  defp pickup1(_1, _2, _3, _4, _5) do
    raise "NIF pickup1/3 not implemented"
  end

  defp copy1(_1, _2) do
    raise "NIF copy1/2 not implemented"
  end

  defp slice1(_1, _2, _3) do
    raise "NIF slice1/3 not implemented"
  end

  defp unslice1(_1, _2, _3, _4, _5, _6) do
    raise "NIF unslice1/6 not implemented"
  end

  # ----------------------------------------------------------------
  @doc """
  generate matrix  mt1*mt2 with cuBLAS. 
  if mt1 or mt2  is float number. generate matrix that each element is s*elt(x,y)
  """
  # c1 == r2 
  def mult({r1, c1, dt1}, {c1, c2, dt2}) do
    result = mult1(r1, c1, dt1, c1, c2, dt2)

    if !is_integer(result) do
      {r1, c2, result}
    else
      error("mult", result)
    end
  end

  def mult(s, {r, c, dt}) when is_float(s) do
    result = smult1(s, r * c, dt)

    if !is_integer(result) do
      {r, c, result}
    else
      error("smult", result)
    end
  end

  def mult({r, c, dt}, s) when is_float(s) do
    result = smult1(s, r * c, dt)

    if !is_integer(result) do
      {r, c, result}
    else
      error("smult", result)
    end
  end

  def mult(s, {c, h, w, dt}) when is_float(s) do
    result = smult1(s, c * h * w, dt)

    if !is_integer(result) do
      {c, h, w, result}
    else
      error("smult", result)
    end
  end

  def mult({c, h, w, dt}, s) when is_float(s) do
    result = smult1(s, c * h * w, dt)

    if !is_integer(result) do
      {c, h, w, result}
    else
      error("smult", result)
    end
  end

  def mult(s, {n, c, h, w, dt}) when is_float(s) do
    result = smult1(s, n * c * h * w, dt)

    if !is_integer(result) do
      {n, c, h, w, result}
    else
      error("smult", result)
    end
  end

  def mult({n, c, h, w, dt}, s) when is_float(s) do
    result = smult1(s, n * c * h * w, dt)

    if !is_integer(result) do
      {n, c, h, w, result}
    else
      error("smult", result)
    end
  end

  def mult(a, b) do
    IO.inspect(a)
    IO.inspect(b)
    raise "mult illegal data type"
  end

  @doc """
  new(r,c) 
  generate matrix with given row col size. Each elements are zero.
  """
  def new(r, c) do
    result = new1(r * c, 0.0)

    if !is_integer(result) do
      {r, c, result}
    else
      error("new1", result)
    end
  end

  @doc """
  new(r,c,val)
  generate matrix with given row col size. Each elements are val.
  """
  def new(r, c, val) when is_float(val) do
    result = new1(r * c, val)

    if !is_integer(result) do
      {r, c, result}
    else
      error("new1", result)
    end
  end

  @doc """
  new(c,h,w)
  generate tensor with given size
  """
  def new(c, h, w) when is_integer(w) do
    result = new1(c * h * w, 0.0)

    if !is_integer(result) do
      {c, h, w, result}
    else
      error("new1", result)
    end
  end

  @doc """
  new(c,h,w,val)
  generate matrix with given row col size. Each elements are val.
  """
  def new(c, h, w, val) when is_float(val) do
    result = new1(c * h * w, val)

    if !is_integer(result) do
      {c, h, w, result}
    else
      error("new1", result)
    end
  end

  @doc """
  new(n,c,h,w,val)
  generate tensor with given size.
  """
  def new(n, c, h, w) when is_integer(w) do
    result = new1(n * c * h * w, 0.0)

    if !is_integer(result) do
      {n, c, h, w, result}
    else
      error("new1", result)
    end
  end

  @doc """
  new(n,c,h,w,val)
  generate tensor with given size.Each elements are val.
  """
  def new(n, c, h, w, val) when is_float(val) do
    result = new1(n * c * h * w, val)

    if !is_integer(result) do
      {n, c, h, w, result}
    else
      error("new1", result)
    end
  end

  @doc """
  new(list)
  generate matrix with given list. e.g. [[1,2],[3,4]].
  ls is also list that express 4-dimension or 3-dimension data
  """
  def new(ls) when is_list(ls) do
    cond do
      list_dim(ls) == 2 ->
        r = length(ls)
        c = length(hd(ls))
        ls1 = ls |> flatten()
        result = new2(r, c, ls1)

        if !is_integer(result) do
          {r, c, result}
        else
          error("new2", result)
        end

      list_dim(ls) == 3 ->
        c = length(ls)
        h = length(hd(ls))
        w = length(hd(hd(ls)))
        ls1 = ls |> flatten()
        result = new3(c, h, w, ls1)

        if !is_integer(result) do
          {c, h, w, result}
        else
          error("new3", result)
        end

      list_dim(ls) == 4 ->
        n = length(ls)
        c = length(hd(ls))
        h = length(hd(hd(ls)))
        w = length(hd(hd(hd(ls))))
        ls1 = ls |> flatten()
        result = new4(n, c, h, w, ls1)

        if !is_integer(result) do
          {n, c, h, w, result}
        else
          error("new4", result)
        end
    end
  end

  @doc """
  rand(r,c)
  generate matrix with random (Box-muller).
  """
  def rand(r, c) do
    result = rand1(r * c)

    if !is_integer(result) do
      {r, c, result}
    else
      error("rand1", result)
    end
  end

  @doc """
  rand(c,h,w)
  generate 3 dimensions data.
  """
  def rand(c, h, w) do
    result = rand1(c * h * w)

    if !is_integer(result) do
      {c, h, w, result}
    else
      error("rand1", result)
    end
  end

  @doc """
  rand(n,c,h,w)
  generate 4 dimensions data.
  """
  def rand(n, c, h, w) do
    result = rand1(n * c * h * w)

    if !is_integer(result) do
      {n, c, h, w, result}
    else
      error("rand1", result)
    end
  end

  @doc """
  add(mt1,mt2)
  generate matrix mt1+mt2.
  if mt1 or mt2 is row vector, expand size to matrix. 
  This function is for bias add in DL.
  """
  def add({r1, c1, dt1}, {r1, c1, dt2}) do
    result = add1(r1 * c1, dt1, dt2)

    if !is_integer(result) do
      {r1, c1, result}
    else
      error("add1", result)
    end
  end

  def add({r1, c1, dt1}, {1, c1, dt2}) do
    result = add1(r1 * c1, dt1, expand({r1, c1, dt2}))

    if !is_integer(result) do
      {r1, c1, result}
    else
      error("add1+expand", result)
    end
  end

  def add({1, c1, dt1}, {r1, c1, dt2}) do
    result = add1(r1 * c1, expand({r1, c1, dt1}), dt2)

    if !is_integer(result) do
      {r1, c1, result}
    else
      error("add1+expand", result)
    end
  end

  def add({c1, h1, w1, dt1}, {c1, h1, w1, dt2}) do
    result = add1(c1 * h1 * w1, dt1, dt2)

    if !is_integer(result) do
      {c1, h1, w1, result}
    else
      error("add1", result)
    end
  end

  def add({n1, c1, h1, w1, dt1}, {n1, c1, h1, w1, dt2}) do
    result = add1(n1 * c1 * h1 * w1, dt1, dt2)

    if !is_integer(result) do
      {n1, c1, h1, w1, result}
    else
      error("add1", result)
    end
  end

  def add(_, _) do
    raise "add illegal data type"
  end

  defp expand({r, c, dt}) do
    dt1 = expand1(r, dt)
    transpose1(r, c, dt1)
  end

  defp expand1(0, _) do
    <<>>
  end

  defp expand1(n, dt) do
    dt <> expand1(n - 1, dt)
  end

  @doc """
  sub(mt1,mt2)
  generate matrix mt1-mt2.
  It is possible to adapt tensor
  """
  def sub({r1, c1, dt1}, {r1, c1, dt2}) do
    result = sub1(r1 * c1, dt1, dt2)

    if !is_integer(result) do
      {r1, c1, result}
    else
      error("sub1", result)
    end
  end

  def sub({c1, h1, w1, dt1}, {c1, h1, w1, dt2}) do
    result = sub1(c1 * h1 * w1, dt1, dt2)

    if !is_integer(result) do
      {c1, h1, w1, result}
    else
      error("sub1", result)
    end
  end

  def sub({n1, c1, h1, w1, dt1}, {n1, c1, h1, w1, dt2}) do
    result = sub1(n1 * c1 * h1 * w1, dt1, dt2)

    if !is_integer(result) do
      {n1, c1, h1, w1, result}
    else
      error("sub1", result)
    end
  end

  def sub(_, _) do
    raise "sub illegal data type"
  end

  @doc """
  emult(mt1,mt2)
  generate Hadamard matrix.
  """
  def emult({r1, c1, dt1}, {r1, c1, dt2}) do
    result = emult1(r1 * c1, dt1, dt2)

    if !is_integer(result) do
      {r1, c1, result}
    else
      error("emult1", result)
    end
  end

  def emult({n, c, h, w, dt1}, {n, c, h, w, dt2}) do
    result = emult1(n * c * h * w, dt1, dt2)

    if !is_integer(result) do
      {n, c, h, w, result}
    else
      error("emult1", result)
    end
  end

  def emult(_, _) do
    raise "emult ilegal data type"
  end

  @doc """
  ediv(mt1,mt2)
  generate differntial Hadamard matrix.
  """
  def ediv({r1, c1, dt1}, {r1, c1, dt2}) do
    result = ediv1(r1 * c1, dt1, dt2)

    if !is_integer(result) do
      {r1, c1, result}
    else
      error("ediv1", result)
    end
  end

  def ediv({n, c, h, w, dt1}, {n, c, h, w, dt2}) do
    result = ediv1(n * c * h * w, dt1, dt2)

    if !is_integer(result) do
      {n, c, h, w, result}
    else
      error("ediv1", result)
    end
  end

  def ediv(_, _) do
    raise "ediv ilegal data type"
  end


  @doc """
  elt(r,c,mt)
  pick up element of mt(r,c) index is one base
  """
  def elt({r, c, dt}, x, y) do
    result = elt1(r, c, x - 1, y - 1, dt)

    if !is_integer(result) do
      result
    else
      error("elt1", result)
    end
  end

  @doc """
  set(mt,r,c,val)
  elt(mt,x,y) := val. 
  """
  def set({r, c, dt}, x, y, val) do
    result = set1(r, c, dt, x - 1, y - 1, val)

    if !is_integer(result) do
      {r, c, result}
    else
      error("set1", result)
    end
  end

  @doc """
  add_diff(mt,r,c,val)
  elt(mt,x,y) := elt(mt,x,y + val. 
  """
  def add_diff({r, c, dt}, x, y, val) do
    result = add_diff1(r, c, dt, x - 1, y - 1, val)

    if !is_integer(result) do
      {r, c, result}
    else
      error("add_diff1", result)
    end
  end

  def add_diff({n, c, h, w, dt}, n1, c1, h1, w1, val) do
    result = add_diff2(n, c, h, w, dt, n1 - 1, c1 - 1, h1 - 1, w1 - 1, val)

    if !is_integer(result) do
      {n, c, h, w, result}
    else
      error("add_diff2", result)
    end
  end

  # iex(1)> Cumatrix.flatten([[1,2],[3,4]])
  # [1, 2, 3, 4]
  defp flatten([]) do
    []
  end

  defp flatten([l | ls]) do
    cond do
      is_number(l) -> [l | flatten(ls)]
      is_list(l) -> flatten(l) ++ flatten(ls)
      true -> l ++ ls
    end
  end

  # iex(1)>  Cumatrix.list_dim([[1,2],[3,4]])
  # 2
  # iex(2)>  Cumatrix.list_dim([[[1,2],[2,3]]])      
  # 3
  defp list_dim([l | _]) do
    if is_list(l) do
      1 + list_dim(l)
    else
      1
    end
  end

  @doc """
  iex(1)> Cumatrix.reshape([1,2,3,4,5,6],[2,3])
  [[1, 2, 3], [4, 5, 6]]
  iex(2)> Cumatrix.reshape([1,2,3,4,5,6],[1,2,3])
  [[[1, 2, 3], [4, 5, 6]]]
  """
  def reshape(x, i) do
    flatten(x) |> reshape1(i)
  end

  defp reshape1(x, [_]) do
    x
  end

  defp reshape1(x, [i | is]) do
    reshape2(x, i) |> Enum.map(fn y -> reshape1(y, is) end)
  end

  defp reshape2(x, n) do
    col = div(length(x), n)
    Enum.chunk_every(x, col)
  end

  @doc """
  iex(1)> Cumatrix.nth([1,2,3],2)
  2
  """
  def nth([x | _], 1) do
    x
  end

  def nth([_ | xs], n) do
    nth(xs, n - 1)
  end

  def nth([], _) do
    raise "nth error"
  end

  @doc """
  transpose(mt)
  generate transposed matrix
  """
  def transpose({r, c, dt}) do
    result = transpose1(r, c, dt)

    if !is_integer(result) do
      {c, r, result}
    else
      error("transpose1", result)
    end
  end

  @doc """
  ident(n)
  generate ident matrix of size n.
  """
  def ident(r) do
    if !is_number(r) || !is_number(r) || r <= 0 do
      raise "ident illegal size"
    end

    result = ident1(r)

    if !is_integer(result) do
      {r, r, result}
    else
      error("ident1", result)
    end
  end

  @doc """
  activate(mt,fun)
  apply fun to mt. fun is :sigmoid, :tanh, :relu :softmax
  """
  def activate({r, c, dt}, :sigmoid) do
    result = activate_sigmoid(r * c, dt)

    if !is_integer(result) do
      {r, c, result}
    else
      error("activate_sigmoid", result)
    end
  end

  def activate({n, c, h, w, dt}, :sigmoid) do
    result = activate_sigmoid(n * c * h * w, dt)

    if !is_integer(result) do
      {n, c, h, w, result}
    else
      error("activate_sigmoid", result)
    end
  end

  def activate({r, c, dt}, :tanh) do
    result = activate_tanh(r * c, dt)

    if !is_integer(result) do
      {r, c, result}
    else
      error("activate_tanh", result)
    end
  end

  def activate({n, c, h, w, dt}, :tanh) do
    result = activate_tanh(n * c * h * w, dt)

    if !is_integer(result) do
      {n, c, h, w, result}
    else
      error("activate_tanh", result)
    end
  end

  def activate({r, c, dt}, :relu) do
    result = activate_relu(r * c, dt)

    if !is_integer(result) do
      {r, c, result}
    else
      error("activate_relu", result)
    end
  end

  def activate({n, c, h, w, dt}, :relu) do
    result = activate_relu(n * c * h * w, dt)

    if !is_integer(result) do
      {n, c, h, w, result}
    else
      error("activate_relu", result)
    end
  end

  def activate({r, c, dt}, :softmax) do
    result = activate_softmax(r, c, dt)

    if !is_integer(result) do
      a = {r, c, result}
      # inspect softmax fordebug
      # to_list(a) |> IO.inspect()
      a
    else
      error("activate_softmax", result)
    end
  end

  def activate(_, _) do
    raise "activate illegal argument"
  end

  @doc """
  diff(mt1,mt2,fun)
  for each element multiply differntial of mt2 and mt1. fun is :sigmoid :tanh, :relu.
  """
  def diff({r, c, dt1}, {r, c, dt2}, :sigmoid) do
    result = differ_sigmoid(r * c, dt1, dt2)

    if !is_integer(result) do
      {r, c, result}
    else
      error("differ_sigmoid", result)
    end
  end

  def diff({n, c, h, w, dt1}, {n, c, h, w, dt2}, :sigmoid) do
    result = differ_sigmoid(n * c * h * w, dt1, dt2)

    if !is_integer(result) do
      {n, c, h, w, result}
    else
      error("differ_sigmoid", result)
    end
  end

  def diff({r, c, dt1}, {r, c, dt2}, :tanh) do
    result = differ_tanh(r * c, dt1, dt2)

    if !is_integer(result) do
      {r, c, result}
    else
      error("differ_tanh", result)
    end
  end

  def diff({n, c, h, w, dt1}, {n, c, h, w, dt2}, :tanh) do
    result = differ_tanh(n * c * h * w, dt1, dt2)

    if !is_integer(result) do
      {n, c, h, w, result}
    else
      error("differ_tanh", result)
    end
  end

  def diff({r, c, dt1}, {r, c, dt2}, :relu) do
    result = differ_relu(r * c, dt1, dt2)

    if !is_integer(result) do
      {r, c, result}
    else
      error("differ_relu", result)
    end
  end

  def diff({n, c, h, w, dt1}, {n, c, h, w, dt2}, :relu) do
    result = differ_relu(n * c * h * w, dt1, dt2)

    if !is_integer(result) do
      {n, c, h, w, result}
    else
      error("differ_relu", result)
    end
  end

  def diff(a, b, c) do
    IO.inspect(a)
    IO.inspect(b)
    IO.inspect(c)
    raise "differ illegal argument"
  end

  @doc """
  size(mt) or size(tensor)
  return tuple {rowsize,colsize}
  """
  def size({r, c, _}) do
    {r, c}
  end

  def size({c, h, w, _}) do
    {c, h, w}
  end

  def size({n, c, h, w, _}) do
    {n, c, h, w}
  end

  @doc """
  average(mt)
  caluculate average of row-vector and generate row-vector that each element is average.
  For Deep-Learning.  
  """
  def average({r, c, dt}) do
    result = average1(r, c, dt)

    if !is_integer(result) do
      {1, c, result}
    else
      error("average1", result)
    end
  end

  @doc """
  average(mt1,mt2)
  caluculate average of each element mt1 and mt2
  """
  def average(mt1, mt2) do
    add(mt1, mt2) |> mult(0.5)
  end

  @doc """
  sum(mt)
  return sum of elements
  """
  def sum({r, c, dt}) do
    result = sum1(r, c, dt)

    if !is_integer(result) do
      result
    else
      error("sum1", result)
    end
  end

  @doc """
  to_list(mt)
  return list that transformed from matrix
  to_list(tensor)
  tensor is 3-dimension or 4-dimension
  """
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

  defp conv_dim(ls, [_]) do
    ls
  end

  defp conv_dim(ls, [d | ds]) do
    dim = div(length(ls), d)
    Enum.chunk_every(ls, dim) |> Enum.map(fn x -> conv_dim(x, ds) end)
  end

  @doc """
  trace(mt)
  return float number. It is trace of matrix.
  """
  def trace({r, c, dt}) do
    if r != c do
      raise "trace not square matrix"
    end

    result = trace1(r, c, dt)

    if !is_integer(result) do
      result
    else
      error("trace1", result)
    end
  end

  @doc """
  loss(mt1,mt2) mt1 is forwarded-matrix. mt2 is train-matrix.
  generate float that is average of loss. fun is :square or :cross.
  :square means mean_square function, and :cross means cross_entropy function.
  mt1 is calculated data matrix , mt2 is train data matrix.
  each data is row vector.
  """
  def loss({r1, c1, dt1}, {r1, c1, dt2}, :square) do
    result = mean_square(r1, c1, dt1, dt2)

    if !is_integer(result) do
      result
    else
      error("mean_square", result)
    end
  end

  def loss({r1, c1, dt1}, {r1, c1, dt2}, :cross) do
    result = cross_entropy(r1, c1, dt1, dt2)

    if !is_integer(result) do
      result
    else
      error("cross_entropy", result)
    end
  end

  @doc """
  generate mask matrix or tensor for dropout
  """
  def dropout({r, c, _}, dr) do
    result = dropout1(r * c, dr)

    if !is_integer(result) do
      {r, c, result}
    else
      error("dropout1", result)
    end
  end

  def dropout({n, c, h, w, _}, dr) do
    result = dropout1(n * c * h * w, dr)

    if !is_integer(result) do
      {n, c, h, w, result}
    else
      error("dropout1", result)
    end
  end

  @doc """
  sgd(mt1,mt2,lr,dr)
  element of mt1 - element of mt2*lr. and dropout with rate dr.
  """
  def sgd({r1, c1, dt1}, {r1, c1, dt2}, lr) do
    result = sgd1(r1 * c1, dt1, dt2, lr)

    if !is_integer(result) do
      {r1, c1, result}
    else
      error("sgd1", result)
    end
  end

  def sgd({c1, h1, w1, dt1}, {c1, h1, w1, dt2}, lr) do
    result = sgd1(c1 * h1 * w1, dt1, dt2, lr)

    if !is_integer(result) do
      {c1, h1, w1, result}
    else
      error("sgd1", result)
    end
  end

  def sgd({n1, c1, h1, w1, dt1}, {n1, c1, h1, w1, dt2}, lr) do
    result = sgd1(n1 * c1 * h1 * w1, dt1, dt2, lr)

    if !is_integer(result) do
      {n1, c1, h1, w1, result}
    else
      error("sgd1", result)
    end
  end

  def sgd(_, _, _, _) do
    raise "sgd illegal data type"
  end

  @doc """
  momentum(mt1,mt2,mt3,lr)
  for each element
  v = 0.5 * mt2(x,y) - lr * mt3(x,y).
  w = mt1 + v.
  return tuple {v,w}
  for learn/3 in DeepPipe2
  """
  def momentum({r1, c1, dt1}, {r1, c1, dt2}, {r1, c1, dt3}, lr) do
    result = momentum1(r1 * c1, dt1, dt2, dt3, lr)

    if !is_integer(result) do
      {v1, w1} = result
      {{r1, c1, v1}, {r1, c1, w1}}
    else
      error("momentum1", result)
    end
  end

  def momentum({n, c, h, w, dt1}, {n, c, h, w, dt2}, {n, c, h, w, dt3}, lr) do
    result = momentum1(n * c * h * w, dt1, dt2, dt3, lr)

    if !is_integer(result) do
      {v1, w1} = result
      {{n, c, h, w, v1}, {n, c, h, w, w1}}
    else
      error("momentum1", result)
    end
  end

  def momentum(_, _, _, _) do
    raise "momentum illegal argument"
  end

  @doc """
  adagrad(weight,h,grad,lr)
  adagrad optimizer 
  return tuple(h,w)
  for learn/3 in DeepPipe2
  """
  def adagrad({r1, c1, dt1}, {r1, c1, dt2}, {r1, c1, dt3}, lr) do
    result = adagrad1(r1 * c1, dt1, dt2, dt3, lr)

    if !is_integer(result) do
      {dth, dtw} = result
      {{r1, c1, dth}, {r1, c1, dtw}}
    else
      error("adagrad1", result)
    end
  end

  def adagrad({n, c, h, w, dt1}, {n, c, h, w, dt2}, {n, c, h, w, dt3}, lr) do
    result = adagrad1(n * c * h * w, dt1, dt2, dt3, lr)

    if !is_integer(result) do
      {dth, dtw} = result
      {{n, c, h, w, dth}, {n, c, h, w, dtw}}
    else
      error("adagrad1", result)
    end
  end

  def adagrad(_, _, _, _) do
    raise "adagrad illegal argument"
  end

  @doc """
  rms(weight,h,grad,lr)
  RMSprop optimizer 
  return tuple(h,w)
  for learn/3 in DeepPipe2
  """
  def rms({r1, c1, dt1}, {r1, c1, dt2}, {r1, c1, dt3}, lr) do
    result = rms1(r1 * c1, dt1, dt2, dt3, lr)

    if !is_integer(result) do
      {dth, dtw} = result
      {{r1, c1, dth}, {r1, c1, dtw}}
    else
      error("rms1", result)
    end
  end

  def rms({n, c, h, w, dt1}, {n, c, h, w, dt2}, {n, c, h, w, dt3}, lr) do
    result = rms1(n * c * h * w, dt1, dt2, dt3, lr)

    if !is_integer(result) do
      {dth, dtw} = result
      {{n, c, h, w, dth}, {n, c, h, w, dtw}}
    else
      error("rms1", result)
    end
  end

  def rms(_, _, _, _) do
    raise "rms illegal argument"
  end

  @doc """
  adam(w,m,v,grad,lr)
  adam optimizer
  return tuple(m1,v1,w1)
  for learn/3 in DeepPipe2
  """
  def adam({r1, c1, dt1}, {r1, c1, dt2}, {r1, c1, dt3}, {r1, c1, dt4}, lr) do
    result = adam1(r1 * c1, dt1, dt2, dt3, dt4, lr)

    if !is_integer(result) do
      {dtm, dtv, dtw} = result
      {{r1, c1, dtm}, {r1, c1, dtv}, {r1, c1, dtw}}
    else
      error("adam1", result)
    end
  end

  def adam({n, c, h, w, dt1}, {n, c, h, w, dt2}, {n, c, h, w, dt3}, {n, c, h, w, dt4}, lr) do
    result = adam1(n * c * h * w, dt1, dt2, dt3, dt4, lr)

    if !is_integer(result) do
      {dtm, dtv, dtw} = result
      {{n, c, h, w, dtm}, {n, c, h, w, dtv}, {n, c, h, w, dtw}}
    else
      error("adam1", result)
    end
  end

  def adam(_, _, _, _, _) do
    raise "adam illegal argument"
  end

  @doc """
  accuracy(mt1,ls) 
  return accuracy rate as float number.
  mt1 is set of row-vector.Each row-vector is onehot.
  ls is list each element is label integer number.

  e.g.

  iex(1)> a = Cumatrix.new([[0.0,0.0,1.0],[0.0,0.1,0.3]])
  {2, 3,
  <<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 205, 204, 204, 61, 0, 0, 128, 63, 154,
   153, 153, 62>>}
  iex(3)> Cumatrix.accuracy(a,[2,2])
  1.0
  iex(4)> Cumatrix.accuracy(a,[2,1])
  0.5
  iex(5)> 
  """
  def accuracy({r1, c1, dt1}, ls) do
    if length(ls) != r1 do
      raise "accuracy illegal argument"
    else
      result = accuracy1(r1, c1, dt1, ls)

      if !is_integer(result) do
        result
      else
        error("accuracy1", result)
      end
    end
  end

  def accuracy(_, _) do
    raise "accuracy illegal argument"
  end

  @doc """
  correct(mt1,ls) 
  return correct number as integer number.
  mt1 is set of row-vector.Each row-vector is onehot.
  ls is list each element is label integer number.

  e.g.

  iex(1)> a = Cumatrix.new([[0.0,0.0,1.0],[0.0,0.1,0.3]])
  {2, 3,
  <<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 205, 204, 204, 61, 0, 0, 128, 63, 154,
   153, 153, 62>>}
  iex(3)> Cumatrix.correct(a,[2,2])
  2.0
  iex(4)> Cumatrix.correct(a,[2,1])
  1.0
  iex(5)> 
  """
  def correct({r1, c1, dt1}, ls) do
    if length(ls) != r1 do
      raise "correct illegal argument"
    else
      result = correct1(r1, c1, dt1, ls)

      if !is_integer(result) do
        result
      else
        error("correct1", result)
      end
    end
  end

  def correct(_, _) do
    raise "correct illegal argument"
  end

  @doc """
  random_select(mt1,mt2,n)
  select same row data from matrix(mt1) and matrix(mt2)
  also for tensor arg1.
  """
  # matrix
  def random_select({r1, c1, dt1}, {r2, c2, dt2}, n) do
    result = random_select1(r1, c1, dt1, r2, c2, dt2, n)

    if !is_integer(result) do
      {dt3, dt4} = result
      {{n, c1, dt3}, {n, c2, dt4}}
    else
      error("random_select", result)
    end
  end

  # 4D tensor
  def random_select({n1, c1, h1, w1, dt1}, {r2, c2, dt2}, n) do
    result = random_select2(n1, c1, h1, w1, dt1, r2, c2, dt2, n)

    if !is_integer(result) do
      {dt3, dt4} = result
      {{n, c1, h1, w1, dt3}, {n, c2, dt4}}
    else
      error("random_select", result)
    end
  end

  # 3D tensor
  def random_select({n1, h1, w1, dt1}, {r2, c2, dt2}, n) do
    result = random_select3(n1, h1, w1, dt1, r2, c2, dt2, n)

    if !is_integer(result) do
      {dt3, dt4} = result
      {{n, h1, w1, dt3}, {n, c2, dt4}}
    else
      error("random_select", result)
    end
  end

  @doc """
  print(mt) print(ts)
  print matrix mt or tensor ts
  """
  def print(x) do
    x |> to_list() |> IO.inspect()
  end

  @doc """
  pooling(tensor,st_h,st_w)
  pooling with stride st_w st_w. size of H and W must be less 1000. max 999*999. return tuple {tensor-for-forward,tensor-for-backward}
  """
  def pooling({n, c, h, w, dt}, st_h, st_w) do
    if rem(h, st_h) != 0 || rem(w, st_w) != 0 do
      raise "pooling illegal argument " <> Integer.to_string(h) <> "," <> Integer.to_string(w)
    else
      result = pooling1(n, c, h, w, dt, st_h, st_w)

      if !is_integer(result) do
        {f, b} = result
        h1 = div(h, st_h)
        w1 = div(w, st_w)
        {{n, c, h1, w1, f}, {n, c, h1, w1, b}}
      else
        error("pooling1", result)
      end
    end
  end

  @doc """
  unpooing(ts1,ts2,st_h,st_w)
  unpooling with stride st.
  ts1 is sparse tensor that save index of max element. ts2 is loss tensor.
  """
  def unpooling({n1, c1, h1, w1, d1}, {n1, c1, h1, w1, d2}, st_h, st_w) do
    result = unpooling1(n1, c1, h1, w1, d1, d2, st_h, st_w)

    if !is_integer(result) do
      h2 = h1 * st_h
      w2 = w1 * st_w
      {n1, c1, h2, w2, result}
    else
      error("unpooling1", result)
    end
  end

  @doc """
  convolute(ts1,ts2,st_h,st_w,pad)
  convolution with input-tensor(ts1), filter-tensor(ts2), stride(st_h,st_w), padding(pad)
  """
  def convolute({n1, c1, h1, w1, dt1}, {n2, c2, h2, w2, dt2}, st_h, st_w, pad) do
    oh = div(h1 + 2 * pad - h2, st_h) + 1
    ow = div(w1 + 2 * pad - w2, st_w) + 1
    result = convolute1(n1, c1, h1, w1, n2, c2, h2, w2, dt1, dt2, st_h, st_w, pad)

    if !is_integer(result) do
      {n1, n2, oh, ow, result}
    else
      error("convolute1", result)
    end
  end

  @doc """
  deconvolute(ts1,ts2,st_h,st_w,pad)
  deconvolution with input-tensor(ts1), filter-tensor(ts2), stride(st_h,st_w), padding(pad)
  1st arg loss-tensor
  2nd arg filter-tesnor
  """
  def deconvolute({n, c1, oh, ow, dt1}, {n2, c2, h2, w2, dt2}, st_h, st_w, pad) do
    h1 = (oh - 1) * st_h - 2 * pad + h2
    w1 = (ow - 1) * st_w - 2 * pad + h2

    if st_h == 1 && st_w == 1 do
      result = deconvolute1(n, c1, oh, ow, n2, c2, h2, w2, dt1, dt2, st_h, st_w, pad)

      if !is_integer(result) do
        {n, c2, h1, w1, result}
      else
        error("deconvolute1", result)
      end
    else
      result = deconvolute2(n, c1, oh, ow, n2, c2, h2, w2, dt1, dt2, st_h, st_w, pad)

      if !is_integer(result) do
        {n, c2, h1, w1, result}
      else
        error("deconvolute2", result)
      end
    end
  end

  @doc """
  gradfilter(ts1,ts2,ts3,st_h,st_w,pad)
  gradient by backpropagation. ts1 is input-tesor, ts2 is filter-tensor, ts3 is loss-tensor, st_h and st_w are stride size, pad is padding size.
  calculate gradient of filter.
  ```
  1st arg input tensor
  2nd arg filter tensor
  3rd arg loss tensor
  4th arg stride_hight
  5th arg stride_width
  6th arg padding size
  ```
  """
  def gradfilter(
        {n1, c1, h1, w1, dt1},
        {n2, c2, h2, w2, _},
        {n1, c3, h3, w3, dt3},
        st_h,
        st_w,
        pad
      ) do
    if st_h == 1 && st_w == 1 do
      result = gradfilter1(n1, c1, h1, w1, n2, c2, h2, w2, c3, h3, w3, dt1, dt3, st_h, st_w, pad)

      if !is_integer(result) do
        {n2, c2, h2, w2, result}
      else
        error("gradfilter1", result)
      end
    else
      result = gradfilter2(n1, c1, h1, w1, n2, c2, h2, w2, c3, h3, w3, dt1, dt3, st_h, st_w, pad)

      if !is_integer(result) do
        {n2, c2, h2, w2, result}
      else
        error("gradfilter2", result)
      end
    end
  end

  def gradfilter(_, _, _, _, _, _) do
    raise "gradfilter illegal data form"
  end

  @doc """
  full(ts) 
  transfer from 4 DIM tensor to matrix.
  """
  def full({n1, c1, h1, w1, dt1}) do
    result = full1(n1, c1, h1, w1, dt1)

    if !is_integer(result) do
      {n1, c1 * h1 * w1, result}
    else
      error("full1", result)
    end
  end

  @doc """
  unfull(mt,h,w)
  transfer from matrix to 4 DIM tensor. tensor(N,C,H,W). N is row size of matrix. C is 1.
  """
  def unfull({r, _, dt1}, c, h, w) do
    result = unfull1(r, c, h, w, dt1)

    if !is_integer(result) do
      {r, c, h, w, result}
    else
      error("unfull1", result)
    end
  end

  defp error(func, n) do
    cond do
      n < 10000 -> raise func <> " bad argument error" <> Integer.to_string(n)
      n >= 10000 && n < 11000 -> raise func <> "cuda error" <> Integer.to_string(n - 10000)
      true -> raise func <> "cuBLAS error" <> Integer.to_string(n - 11000)
    end
  end

  def is_near({r, c, dt1}, {r, c, dt2}) do
    if is_near1(r * c, dt1, dt2) == 1 do
      true
    else
      false
    end
  end

  @doc """
  is_near(mt1,mt2) is_near(ts1,ts2)
  for debug
  """
  def is_near({c, h, w, dt1}, {c, h, w, dt2}) do
    if is_near1(c * h * w, dt1, dt2) == 1 do
      true
    else
      false
    end
  end

  def is_near({n, c, h, w, dt1}, {n, c, h, w, dt2}) do
    if is_near1(n * c * h * w, dt1, dt2) == 1 do
      true
    else
      false
    end
  end

  def is_near(_, _) do
    false
  end

  @doc """
  if_equal(mt1,mt2) is_equal(ts1,ts2)
  for debug
  """
  def is_equal({r, c, dt1}, {r, c, dt2}) do
    if is_equal1(r * c, dt1, dt2) == 1 do
      true
    else
      false
    end
  end

  def is_equal({c, h, w, dt1}, {c, h, w, dt2}) do
    if is_equal1(c * h * w, dt1, dt2) == 1 do
      true
    else
      false
    end
  end

  def is_equal({n, c, h, w, dt1}, {n, c, h, w, dt2}) do
    if is_equal1(n * c * h * w, dt1, dt2) == 1 do
      true
    else
      false
    end
  end

  def is_equal(_, _) do
    false
  end

  @doc """
  analizer(mt,id) analizer(ts,id)
  display id-number,max-element,min-element,average.
  for debug.
  """
  def analizer({n, c, h, w, dt}, id) do
    cond do
      analizer1(n * c * h * w, dt, id) == 9999 -> raise "analizer NAN"
      analizer1(n * c * h * w, dt, id) == 9998 -> raise "analizer INF"
      true -> true
    end
  end

  def analizer({r, c, dt}, id) do
    cond do
      analizer1(r * c, dt, id) == 9999 -> raise "analizer NAN"
      analizer1(r * c, dt, id) == 9998 -> raise "analizer INF"
      true -> true
    end
  end

  @doc """
  visualizer(ts,n,c)
  display heatmap nth and c channel data.
  It depends on Matrex.heatmap
  """
  def visualizer(x, n, c) do
    x |> to_list() |> nth(n) |> nth(c) |> Matrex.new() |> Matrex.heatmap(:color256, [])
  end

  @doc """
  normalizer(ts)
  calculate average of nth data and sub each elemet the average.
  when 3D tensor or matrix , nothing to do
  """
  def standardize({n, c, h, w, dt}) do
    {n, c, h, w, standardize1(n, c, h, w, dt)}
  end

  def standardize({n, h, w, dt}) do
    {n, h, w, dt}
  end

  def standardize({r, c, dt}) do
    {r, c, dt}
  end

  @doc """
  pickup(3Dtensor,nth)
  translate 3Dtensor to matrix. for RNN
  """
  def pickup({n, r, c, dt}, nth) do
    result = pickup1(n, r, c, dt, nth)

    if !is_integer(result) do
      {n, c, result}
    else
      error("pickup1", result)
    end
  end

  @doc """
  copy(x)
  return copy of matrix or tensor x
  """
  def copy({r, c, dt}) do
    result = copy1(r * c, dt)

    if !is_integer(result) do
      {r, c, result}
    else
      error("copy1", result)
    end
  end

  def copy({n, r, c, dt}) do
    result = copy1(n * r * c, dt)

    if !is_integer(result) do
      {n, r, c, result}
    else
      error("copy1", result)
    end
  end

  def copy({n, c, h, w, dt}) do
    result = copy1(n * c * h * w, dt)

    if !is_integer(result) do
      {n, c, h, w, result}
    else
      error("copy1", result)
    end
  end

  def slice({r, c, dt}) do
    result = slice1(r, c, dt)

    if !is_integer(result) do
      {dt1, dt2, dt3, dt4} = result
      c1 = div(c, 4)
      {{r, c1, dt1}, {r, c1, dt2}, {r, c1, dt3}, {r, c1, dt4}}
    else
      error("copy1", result)
    end
  end

  def unslice({r, c, dt1}, {r, c, dt2}, {r, c, dt3}, {r, c, dt4}) do
    result = unslice1(r, c, dt1, dt2, dt3, dt4)

    if !is_integer(result) do
      {r, 4 * c, result}
    else
      error("unslice1", result)
    end
  end

  @doc """
  is_matrix(x)
  if x is matrix return true
  else return false
  """
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

  @doc """
  is_tesnsor(x)
  if x is tensor return true
  else return false
  """
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

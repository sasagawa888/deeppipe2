# test nvcc
defmodule Cumatrix do
  @on_load :load_nifs

  def load_nifs do
    :erlang.load_nif('./lib/nifs', 0)
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

  def add1(_a, _b, _c) do
    raise "NIF add1/3 not implemented"
  end

  def sub1(_a, _b, _c) do
    raise "NIF sub1/3 not implemented"
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

  def activate_sigmoid(_a, _b) do
    raise "NIF activate_sigmoid/2 not implemented"
  end

  def activate_tanh(_a, _b) do
    raise "NIF activate_tanh/2 not implemented"
  end

  def activate_relu(_a, _b) do
    raise "NIF activate_relu/2 not implemented"
  end

  def activate_softmax(_a, _b, _c) do
    raise "NIF activate_softmax/3 not implemented"
  end

  def differ_sigmoid(_a, _b, _c) do
    raise "NIF differ_sigmoid/3 not implemented"
  end

  def differ_tanh(_a, _b, _c) do
    raise "NIF differ_tanh/3 not implemented"
  end

  def differ_relu(_a, _b, _c) do
    raise "NIF differ_relu/3 not implemented"
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

  def add_diff1(_a, _b, _c, _d, _e, _f) do
    raise "NIF add_diff1/6 not implemented"
  end

  def add_diff2(_a, _b, _c, _d, _e, _f, _g, _h, _i, _j) do
    raise "NIF add_diff2/10 not implemented"
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

  def sgd1(_1, _2, _3, _4, _5) do
    raise "NIF sgd1/5 not implemented"
  end

  def momentum1(_a, _b, _c, _d, _e, _f) do
    raise "NIF momentum1/6 not implemented"
  end

  def adagrad1(_a, _b, _c, _d, _e, _f) do
    raise "NIF adagrad1/6 not implemented"
  end

  def accuracy1(_a, _b, _c, _d) do
    raise "NIF accuracy/4 not implemented"
  end

  def pooling1(_1, _2, _3, _4, _5, _6, _7) do
    raise "NIF pooling1/7 not implemented"
  end

  def unpooling1(_1, _2, _3, _4, _5, _6, _7, _8) do
    raise "NIF unpooling1/8 not implemented"
  end

  def convolute1(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13) do
    raise "NIF convolute1/13 not implemented"
  end

  def deconvolute1(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12) do
    raise "NIF deconvolute1/12 not implemented"
  end

  def deconvolute2(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12) do
    raise "NIF deconvolute2/12 not implemented"
  end

  def gradfilter1(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15) do
    raise "NIF gradfilter1/15 not implemented"
  end

  def gradfilter2(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15) do
    raise "NIF gradfilter2/15 not implemented"
  end

  def full1(_1, _2, _3, _4, _5) do
    raise "NIF full1/5 not implemented"
  end

  def unfull1(_1, _2, _3, _4, _5) do
    raise "NIF unfull1/5 not implemented"
  end

  def random_select1(_1, _2, _3, _4, _5, _6, _7) do
    raise "NIF random_select1/7 not implemented"
  end

  def random_select2(_1, _2, _3, _4, _5, _6, _7, _8, _9) do
    raise "NIF random_select1/9 not implemented"
  end

  def is_near1(_1, _2, _3) do
    raise "NIF is_near1/3 not implemented"
  end

  def is_equal1(_1, _2, _3) do
    raise "NIF is_equal1/3 not implemented"
  end

  def analizer1(_1, _2, _3) do
    raise "NIF analizer1/3 not implemented"
  end

  # ----------------------------------------------------------------
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

  def new(r, c) do
    result = new1(r * c, 0.0)

    if !is_integer(result) do
      {r, c, result}
    else
      error("new1", result)
    end
  end

  def new(r, c, val) when is_float(val) do
    result = new1(r * c, val)

    if !is_integer(result) do
      {r, c, result}
    else
      error("new1", result)
    end
  end

  def new(c, h, w) when is_integer(w) do
    result = new1(c * h * w, 0.0)

    if !is_integer(result) do
      {c, h, w, result}
    else
      error("new1", result)
    end
  end

  def new(c, h, w, val) when is_float(val) do
    result = new1(c * h * w, val)

    if !is_integer(result) do
      {c, h, w, result}
    else
      error("new1", result)
    end
  end

  def new(n, c, h, w) when is_integer(w) do
    result = new1(n * c * h * w, 0.0)

    if !is_integer(result) do
      {n, c, h, w, result}
    else
      error("new1", result)
    end
  end

  def new(n, c, h, w, val) when is_float(val) do
    result = new1(n * c * h * w, val)

    if !is_integer(result) do
      {n, c, h, w, result}
    else
      error("new1", result)
    end
  end

  # list to matrix
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

  def rand(r, c) do
    result = rand1(r * c)

    if !is_integer(result) do
      {r, c, result}
    else
      error("rand1", result)
    end
  end

  def rand(c, h, w) do
    result = rand1(c * h * w)

    if !is_integer(result) do
      {c, h, w, result}
    else
      error("rand1", result)
    end
  end

  def rand(n, c, h, w) do
    result = rand1(n * c * h * w)

    if !is_integer(result) do
      {n, c, h, w, result}
    else
      error("rand1", result)
    end
  end

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

  def emult({r1, c1, dt1}, {r1, c1, dt2}) do
    result = emult1(r1, c1, dt1, dt2)

    if !is_integer(result) do
      {r1, c1, result}
    else
      error("emult1", result)
    end
  end

  def emult(_, _) do
    raise "emult ilegal data type"
  end

  def elt({r, c, dt}, x, y) do
    result = elt1(r, c, x - 1, y - 1, dt)

    if !is_integer(result) do
      result
    else
      error("elt1", result)
    end
  end

  def set({r, c, dt}, x, y, val) do
    result = set1(r, c, dt, x - 1, y - 1, val)

    if !is_integer(result) do
      {r, c, result}
    else
      error("set1", result)
    end
  end

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

  @doc """
  iex(1)> Cumatrix.flatten([[1,2],[3,4]])
  [1, 2, 3, 4]
  """
  def flatten([]) do
    []
  end

  def flatten([l | ls]) do
    cond do
      is_number(l) -> [l | flatten(ls)]
      is_list(l) -> flatten(l) ++ flatten(ls)
      true -> l ++ ls
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

  def reshape(x, i) do
    flatten(x) |> reshape1(i)
  end

  def reshape1(x, [_]) do
    x
  end

  def reshape1(x, [i | is]) do
    reshape2(x, i) |> Enum.map(fn y -> reshape1(y, is) end)
  end

  def reshape2(x, n) do
    col = div(length(x), n)
    Enum.chunk_every(x, col)
  end

  def nth([x | _], 1) do
    x
  end

  def nth([_ | xs], n) do
    nth(xs, n - 1)
  end

  def nth([], _) do
    raise "nth error"
  end

  def transpose({r, c, dt}) do
    result = transpose1(r, c, dt)

    if !is_integer(result) do
      {c, r, result}
    else
      error("transpose1", result)
    end
  end

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
    result = average1(r, c, dt)

    if !is_integer(result) do
      {1, c, result}
    else
      error("average1", result)
    end
  end

  def sum({r, c, dt}) do
    result = sum1(r, c, dt)

    if !is_integer(result) do
      result
    else
      error("sum1", result)
    end
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

    result = trace1(r, c, dt)

    if !is_integer(result) do
      result
    else
      error("trace1", result)
    end
  end

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

  def sgd({r1, c1, dt1}, {r1, c1, dt2}, lr, dr) do
    result = sgd1(r1 * c1, dt1, dt2, lr, dr)

    if !is_integer(result) do
      {r1, c1, result}
    else
      error("sgd1", result)
    end
  end

  def sgd({c1, h1, w1, dt1}, {c1, h1, w1, dt2}, lr, dr) do
    result = sgd1(c1 * h1 * w1, dt1, dt2, lr, dr)

    if !is_integer(result) do
      {c1, h1, w1, result}
    else
      error("sgd1", result)
    end
  end

  def sgd({n1, c1, h1, w1, dt1}, {n1, c1, h1, w1, dt2}, lr, dr) do
    result = sgd1(n1 * c1 * h1 * w1, dt1, dt2, lr, dr)

    if !is_integer(result) do
      {n1, c1, h1, w1, result}
    else
      error("sgd1", result)
    end
  end

  def sgd(_, _) do
    raise "sgd illegal data type"
  end

  def momentum({r1, c1, dt1}, {r1, c1, dt2}, {r1, c1, dt3}, lr, dr) do
    result = momentum1(r1 * c1, dt1, dt2, dt3, lr, dr)

    if !is_integer(result) do
      {v1, w1} = result
      {{r1, c1, v1}, {r1, c1, w1}}
    else
      error("momentum1", result)
    end
  end

  def momentum({n, c, h, w, dt1}, {n, c, h, w, dt2}, {n, c, h, w, dt3}, lr, dr) do
    result = momentum1(n * c * h * w, dt1, dt2, dt3, lr, dr)

    if !is_integer(result) do
      {v1, w1} = result
      {{n, c, h, w, v1}, {n, c, h, w, w1}}
    else
      error("momentum1", result)
    end
  end

  def momentum(_, _, _, _, _) do
    raise "momentum illegal argument"
  end

  def adagrad({r1, c1, dt1}, {r1, c1, dt2}, {r1, c1, dt3}, lr, dr) do
    result = adagrad1(r1 * c1, dt1, dt2, dt3, lr, dr)

    if !is_integer(result) do
      {dth, dtw} = result
      {{r1, c1, dth}, {r1, c1, dtw}}
    else
      error("adagrad1", result)
    end
  end

  def adagrad({n, c, h, w, dt1}, {n, c, h, w, dt2}, {n, c, h, w, dt3}, lr, dr) do
    result = adagrad1(n * c * h * w, dt1, dt2, dt3, lr, dr)

    if !is_integer(result) do
      {dth, dtw} = result
      {{n,c, h, w, dth}, {n,c, h, w, dtw}}
    else
      error("adagrad1", result)
    end
  end

  def adagrad(_, _, _, _, _) do
    raise "adagrad illegal argument"
  end

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

  def accurace(_, _) do
    raise "accuracy illegal argument"
  end

  def random_select({r1, c1, dt1}, {r2, c2, dt2}, n) do
    result = random_select1(r1, c1, dt1, r2, c2, dt2, n)

    if !is_integer(result) do
      {dt3, dt4} = result
      {{n, c1, dt3}, {n, c2, dt4}}
    else
      error("random_select", result)
    end
  end

  def random_select({n1, c1, h1, w1, dt1}, {r2, c2, dt2}, n) do
    result = random_select2(n1, c1, h1, w1, dt1, r2, c2, dt2, n)

    if !is_integer(result) do
      {dt3, dt4} = result
      {{n, c1, h1, w1, dt3}, {n, c2, dt4}}
    else
      error("random_select", result)
    end
  end

  def print(x) do
    x |> to_list() |> IO.inspect()
  end

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

  def convolute({n1, c1, h1, w1, dt1}, {n2, c2, h2, w2, dt2}, st_h, st_w, pad) do
    oh = div(h1 + 2 * pad - h2, st_h) + 1
    ow = div(w1 + 2 * pad - w2, st_w) + 1
    result = convolute1(n1, c1, h1, w1, n2, c2, h2, w2, dt1, dt2, st_h,st_w, pad)

    if !is_integer(result) do
      {n1, n2, oh, ow, result}
    else
      error("convolute1", result)
    end
  end

  # 1st arg loss-tensor
  # 2nd arg filter-tesnor
  def deconvolute({n, c1, oh, ow, dt1}, {n2, c2, h2, w2, dt2}, st, pad) do
    h1 = (oh - 1) * st - 2 * pad + h2
    w1 = (ow - 1) * st - 2 * pad + h2

    if st == 1 do
      result = deconvolute1(n, c1, oh, ow, n2, c2, h2, w2, dt1, dt2, st, pad)

      if !is_integer(result) do
        {n, c2, h1, w1, result}
      else
        error("deconvolute1", result)
      end
    else
      result = deconvolute2(n, c1, oh, ow, n2, c2, h2, w2, dt1, dt2, st, pad)

      if !is_integer(result) do
        {n, c2, h1, w1, result}
      else
        error("deconvolute2", result)
      end
    end
  end

  # 1st arg input tensor
  # 2nd arg filter tensor
  # 3rd arg loss tensor
  # 4th arg stride
  # 5th arg padding
  def gradfilter({n1, c1, h1, w1, dt1}, {n2, c2, h2, w2, _}, {n1, c3, h3, w3, dt3}, st, pad) do
    if st == 1 do
      result = gradfilter1(n1, c1, h1, w1, n2, c2, h2, w2, c3, h3, w3, dt1, dt3, st, pad)

      if !is_integer(result) do
        {n2, c2, h2, w2, result}
      else
        error("gradfilter1", result)
      end
    else
      result = gradfilter2(n1, c1, h1, w1, n2, c2, h2, w2, c3, h3, w3, dt1, dt3, st, pad)

      if !is_integer(result) do
        {n2, c2, h2, w2, result}
      else
        error("gradfilter2", result)
      end
    end
  end

  def gradfilter(_, _, _, _) do
    raise "gradfilter illegal data form"
  end

  def full({n1, c1, h1, w1, dt1}) do
    result = full1(n1, c1, h1, w1, dt1)

    if !is_integer(result) do
      {n1, c1 * h1 * w1, result}
    else
      error("full1", result)
    end
  end

  def unfull({r, _, dt1}, c, h, w) do
    result = unfull1(r, c, h, w, dt1)

    if !is_integer(result) do
      {r, c, h, w, result}
    else
      error("unfull1", result)
    end
  end

  def error(func, n) do
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

  def visualizer(x, n, c) do
    x |> to_list() |> nth(n) |> nth(c) |> Matrex.new() |> Matrex.heatmap(:color256, [])
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

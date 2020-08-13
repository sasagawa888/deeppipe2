defmodule Network do
  alias Cumatrix, as: CM

  @moduledoc """
  defnetwork is macros to describe network

  argument must have under bar to avoid warning message

  ```
  defnetwork  name(_x) do
  _x |> element of network |> ...
  end

  ```

  element 

  - w(r,c)  weight matrix row-size is r col-size is c. initial val is random * 0.1, default learning late 0.1
  - w(r,c,ir,lr) ir is initial rate to multiple randam, lr is learning rate.
  - w(r,c,ir,lr,dr) dr is dropout rate.
  - b(n) bias row vector size n.  initial val is randam * 0.1, default learning late 0.1 
  - b(n,ir,lr) ir is initial rate to multiple randam, lr is learning rate.
  - b(n,ir,lr,dp) dr is dropout rate.
  - activate function  leru sigmoid tanh softmax
  - f(r,c) filter matrix row-size is r col-size is c. input and output channel is 1, initial val random * 0.1, default learning late 0.1
  - f(r,c,i)  filter matrix. i input channel.
  - f(r,c,i,o) filter matrix. o output channel
  - f(r,c,i,o,{st_h,st_w}) filter matrix. st_h and st_w are stride size od hight and width.
  - f(r,c,i,o,{st_h,st_w},pad) filter matrix. pad is padding size. 
  - f(r,c,i,o,{st_h,st_w},pad,{:xcavier,dim},lr) filter matrix. generate initial element by Xavier method. Dim is dimension of input, lr is learning rate.
  - f(r,c,i,o,{st_h,st_w},pad,{:he,dim},lr) filter matrix. generate initial element by He method. Dim is dimension of input, lr is learning rate.
  - f(r,c,i,o,{st_h,st_w},pad,ir,lr) filter matrix. ir is rate for initial val, lr is learning rate.
  - f(r,c,i,o,{st_h,st_w},pad,ir,lr,dr) filter matrix. dr is dropout rate.
  - pooling(st_h,st_w) st_h and st_w are pooling size.
  - full    convert from image of CNN to matrix for DNN.

  for debug
  - analizer(n)  calculate max min average of data and display n max min average
  - visualizer(n,c)  display a data(n th, c channel) as graphics 


  data structure
  ```
  network
  [{:weight,w,ir,lr,dr,v},{:bias,b,ir,lr,dr,v},{:function,name},{:filter,w,{st_h,st_w},pad,ir,lr,dr,v} ...]
  weight
  {:weight,w,ir,lr,dp,v,mask} w is matrix, ir is rate for initial random number,
  lr is learning rate, dp is dropout rate.
  bias
  {:bias,b,ir,lr,dp,v,mask} b is row vector
  function
  {:function,name} name is function name within sigmoid tanh relu softmax
  filter
  {:filter,w,{st_h,st_w},pad,ir,lr,dr,v,mask}
  pooling
  {:pooling,st_,st_w}
  ```

  """

  defmacro defnetwork(name, do: body) do
    {_, _, [{arg, _, _}]} = name
    body1 = parse(body, arg)

    quote do
      def unquote(name) do
        unquote(body1)
      end
    end
  end

  # weight
  # cw mean constant weight for gradient check
  def parse({:cw, _, [m]}, _) do
    quote do
      [{:weight, CM.new(unquote(m)), 0.1, 0.1, 0.0, CM.new(1, 1)}]
    end
  end

  def parse({:w, _, [x, y]}, _) do
    quote do
      [
        {:weight, CM.rand(unquote(x), unquote(y)) |> CM.mult(0.1), 0.1, 0.1, 0.0,
         CM.new(unquote(x), unquote(y))}
      ]
    end
  end

  def parse({:w, _, [x, y, ir]}, _) do
    quote do
      [
        {:weight, CM.rand(unquote(x), unquote(y)) |> CM.mult(unquote(ir)), unquote(ir), 0.1, 0.0,
         CM.new(unquote(x), unquote(y))}
      ]
    end
  end

  def parse({:w, _, [x, y, {:xavier, dim}, lr]}, _) do
    quote do
      [
        {:weight, CM.rand(unquote(x), unquote(y)) |> CM.mult(:math.sqrt(1 / unquote(dim))),
         {:xavier, unquote(dim)}, unquote(lr), 0.0, CM.new(unquote(x), unquote(y))}
      ]
    end
  end

  def parse({:w, _, [x, y, {:he, dim}, lr]}, _) do
    quote do
      [
        {:weight, CM.rand(unquote(x), unquote(y)) |> CM.mult(:math.sqrt(2 / unquote(dim))),
         {:he, unquote(dim)}, unquote(lr), 0.0, CM.new(unquote(x), unquote(y))}
      ]
    end
  end

  def parse({:w, _, [x, y, ir, lr]}, _) do
    quote do
      [
        {:weight, CM.rand(unquote(x), unquote(y)) |> CM.mult(unquote(ir)), unquote(ir),
         unquote(lr), 0.0, CM.new(unquote(x), unquote(y))}
      ]
    end
  end

  def parse({:w, _, [x, y, {:xavier, dim}, lr, dr]}, _) do
    quote do
      [
        {:weight, CM.rand(unquote(x), unquote(y)) |> CM.mult(:math.sqrt(1 / unquote(dim))),
         {:xavier, unquote(dim)}, unquote(lr), unquote(dr), CM.new(unquote(x), unquote(y))}
      ]
    end
  end

  def parse({:w, _, [x, y, {:he, dim}, lr, dr]}, _) do
    quote do
      [
        {:weight, CM.rand(unquote(x), unquote(y)) |> CM.mult(:math.sqrt(2 / unquote(dim))),
         {:he, unquote(dim)}, unquote(lr), unquote(dr), CM.new(unquote(x), unquote(y))}
      ]
    end
  end

  def parse({:w, _, [x, y, ir, lr, dr]}, _) do
    quote do
      [
        {:weight, CM.rand(unquote(x), unquote(y)) |> CM.mult(unquote(ir)), unquote(ir),
         unquote(lr), unquote(dr), CM.new(unquote(x), unquote(y))}
      ]
    end
  end

  # bias
  # cb means constant bias for gradient check
  def parse({:cb, _, [m]}, _) do
    quote do
      [{:bias, CM.new(unquote(m)), 0.1, 0.1, 0.0, CM.new(1, 1)}]
    end
  end

  def parse({:b, _, [x]}, _) do
    quote do
      [{:bias, CM.new(1, unquote(x)) |> CM.mult(0.1), 0.1, 0.1, 0.0, CM.new(1, unquote(x))}]
    end
  end

  def parse({:b, _, [x, ir]}, _) do
    quote do
      [
        {:bias, CM.rand(1, unquote(x)) |> CM.mult(unquote(ir)), unquote(ir), 0.1, 0.0,
         CM.new(1, unquote(x))}
      ]
    end
  end

  def parse({:b, _, [x, ir, lr]}, _) do
    quote do
      [
        {:bias, CM.rand(1, unquote(x)) |> CM.mult(unquote(ir)), unquote(ir), unquote(lr), 0.0,
         CM.new(1, unquote(x))}
      ]
    end
  end

  def parse({:b, _, [x, ir, lr, dr]}, _) do
    quote do
      [
        {:bias, CM.rand(1, unquote(x)) |> CM.mult(unquote(ir)), unquote(ir), unquote(lr),
         unquote(dr), CM.new(1, unquote(x))}
      ]
    end
  end

  # sigmoid
  def parse({:sigmoid, _, nil}, _) do
    quote do
      [{:function, :sigmoid}]
    end
  end

  # identity
  def parse({:tanh, _, nil}, _) do
    quote do
      [{:function, :tanh}]
    end
  end

  # relu
  def parse({:relu, _, nil}, _) do
    quote do
      [{:function, :relu}]
    end
  end

  # softmax
  def parse({:softmax, _, nil}, _) do
    quote do
      [{:function, :softmax}]
    end
  end

  # filter
  # cf means constant filter for gradient check
  def parse({:cf, _, [m]}, _) do
    quote do
      [{:filter, CM.new(unquote(m)), 1, 0, 0.1, 0.1, CM.new(1, 3, 3)}]
    end
  end

  # {:filter,filter-matrix,stride,padding,init_rate,learning_rate,dropout_rate,v}
  def parse({:f, _, [x, y]}, _) do
    quote do
      [
        {:filter, CM.rand(1, 1, unquote(x), unquote(y)) |> CM.mult(0.1), 1, 0, 0.1, 0.1, 0.0,
         CM.new(1, 1, unquote(x), unquote(y))}
      ]
    end
  end

  def parse({:f, _, [x, y, c]}, _) do
    quote do
      [
        {:filter, CM.rand(1, unquote(c), unquote(x), unquote(y)) |> CM.mult(0.1), 1, 0, 0.1, 0.1,
         0.0, CM.new(1, unquote(c), unquote(x), unquote(y))}
      ]
    end
  end

  def parse({:f, _, [x, y, c, n]}, _) do
    quote do
      [
        {:filter, CM.rand(unquote(n), unquote(c), unquote(x), unquote(y)) |> CM.mult(0.1), 1, 0,
         0.0, 0.1, 0.1, CM.new(unquote(n), unquote(c), unquote(x), unquote(y))}
      ]
    end
  end

  def parse({:f, _, [x, y, c, n, {h, w}]}, _) do
    quote do
      [
        {:filter, CM.rand(unquote(n), unquote(c), unquote(x), unquote(y)) |> CM.mult(0.1),
         {unquote(h), unquote(w)}, 0, 0.1, 0.1, 0.0,
         CM.new(unquote(n), unquote(c), unquote(x), unquote(y))}
      ]
    end
  end

  def parse({:f, _, [x, y, c, n, {h, w}, pad]}, _) do
    quote do
      [
        {:filter, CM.rand(unquote(n), unquote(c), unquote(x), unquote(y)) |> CM.mult(0.1),
         {unquote(h), unquote(w)}, unquote(pad), 0.1, 0.1, 0.0,
         CM.new(unquote(n), unquote(c), unquote(x), unquote(y))}
      ]
    end
  end

  def parse({:f, _, [x, y, c, n, {h, w}, pad, {:xavier, dim}]}, _) do
    quote do
      [
        {:filter,
         CM.rand(unquote(n), unquote(c), unquote(x), unquote(y))
         |> CM.mult(:math.sqrt(1 / unquote(dim))), {unquote(h), unquote(w)}, unquote(pad),
         {:xavier, unquote(dim)}, 0.1, 0.0,
         CM.new(unquote(n), unquote(c), unquote(x), unquote(y))}
      ]
    end
  end

  def parse({:f, _, [x, y, c, n, {h, w}, pad, {:he, dim}]}, _) do
    quote do
      [
        {:filter,
         CM.rand(unquote(n), unquote(c), unquote(x), unquote(y))
         |> CM.mult(:math.sqrt(2 / unquote(dim))), {unquote(h), unquote(w)}, unquote(pad),
         {:he, unquote(dim)}, 0.1, 0.0, CM.new(unquote(n), unquote(c), unquote(x), unquote(y))}
      ]
    end
  end

  def parse({:f, _, [x, y, c, n, {h, w}, pad, {:xavier, dim}, lr]}, _) do
    quote do
      [
        {:filter,
         CM.rand(unquote(n), unquote(c), unquote(x), unquote(y))
         |> CM.mult(:math.sqrt(1 / unquote(dim))), {unquote(h), unquote(w)}, unquote(pad),
         {:xavier, unquote(dim)}, unquote(lr), 0.0,
         CM.new(unquote(n), unquote(c), unquote(x), unquote(y))}
      ]
    end
  end

  def parse({:f, _, [x, y, c, n, {h, w}, pad, {:he, dim}, lr]}, _) do
    quote do
      [
        {:filter,
         CM.rand(unquote(n), unquote(c), unquote(x), unquote(y))
         |> CM.mult(:math.sqrt(2 / unquote(dim))), {unquote(h), unquote(w)}, unquote(pad),
         {:he, unquote(dim)}, unquote(lr), 0.0,
         CM.new(unquote(n), unquote(c), unquote(x), unquote(y))}
      ]
    end
  end

  def parse({:f, _, [x, y, c, n, {h, w}, pad, ir, lr]}, _) do
    quote do
      [
        {:filter, CM.rand(unquote(n), unquote(c), unquote(x), unquote(y)) |> CM.mult(unquote(ir)),
         {unquote(h), unquote(w)}, unquote(pad), unquote(ir), unquote(lr), 0.0,
         CM.new(unquote(n), unquote(c), unquote(x), unquote(y))}
      ]
    end
  end

  def parse({:f, _, [x, y, c, n, {h, w}, pad, {:xavier, dim}, lr, dr]}, _) do
    quote do
      [
        {:filter,
         CM.rand(unquote(n), unquote(c), unquote(x), unquote(y))
         |> CM.mult(:math.sqrt(1 / unquote(dim))), {unquote(h), unquote(w)}, unquote(pad),
         {:xavier, unquote(dim)}, unquote(lr), unquote(dr),
         CM.new(unquote(n), unquote(c), unquote(x), unquote(y))}
      ]
    end
  end

  def parse({:f, _, [x, y, c, n, {h, w}, pad, {:he, dim}, lr, dr]}, _) do
    quote do
      [
        {:filter,
         CM.rand(unquote(n), unquote(c), unquote(x), unquote(y))
         |> CM.mult(:math.sqrt(2 / unquote(dim))), {unquote(h), unquote(w)}, unquote(pad),
         {:he, unquote(dim)}, unquote(lr), unquote(dr),
         CM.new(unquote(n), unquote(c), unquote(x), unquote(y))}
      ]
    end
  end

  def parse({:f, _, [x, y, c, n, {h, w}, pad, ir, lr, dr]}, _) do
    quote do
      [
        {:filter, CM.rand(unquote(n), unquote(c), unquote(x), unquote(y)) |> CM.mult(unquote(ir)),
         {unquote(h), unquote(w)}, unquote(pad), unquote(ir), unquote(lr), unquote(dr),
         CM.new(unquote(n), unquote(c), unquote(x), unquote(y))}
      ]
    end
  end

  # pooling
  def parse({:pooling, _, [h, w]}, _) do
    quote do
      [{:pooling, unquote(h), unquote(w)}]
    end
  end

  # flll connection
  def parse({:full, _, nil}, _) do
    quote do
      [{:full}]
    end
  end

  # analizer for debug
  def parse({:analizer, _, [x]}, _) do
    quote do
      [{:analizer, unquote(x)}]
    end
  end

  # visualizer for debug
  def parse({:visualizer, _, [n, c]}, _) do
    quote do
      [{:visualizer, unquote(n), unquote(c)}]
    end
  end

  # RNN
  def parse({:rnn, _, [x, n]}, _) do
    rnn(x, 1, n)
  end

  # LSTM
  def parse({:lstm, _, [x, y]}, _) do
    quote do
      [{:lstm, unquote(x), unquote(y), 0}]
    end
  end

  def parse({x, _, nil}, _) do
    x
  end

  def parse({:|>, _, exp}, arg) do
    parse(exp, arg)
  end

  def parse([{arg, _, nil}, exp], arg) do
    parse(exp, arg)
  end

  def parse([exp1, exp2], arg) do
    Enum.reverse(parse(exp2, arg) ++ Enum.reverse(parse(exp1, arg)))
  end

  def parse(x, _) do
    IO.write("Syntax error in defnetwork  ")
    IO.inspect(x)
    raise ""
  end

  def rnn(_, m, n) when m > n do
    []
  end

  # data structure RNN = {:rnn, nth, Wx, Wh, b, ir, lr, dr, v}
  # nth means nth of recursive
  # Wx is weigth matrix for input x
  # Wh is weight matrix for h (output t-1)
  # ir is initial rate
  # lr is learning rate
  # dr is dropout rate
  # v = matrix for learning momentum
  def rnn(x, m, n) do
    [
      quote do
        {:rnn, unquote(m), CM.rand(unquote(x), unquote(x)), CM.rand(unquote(x), unquote(x)),
         CM.rand(1, unquote(x)), 0.1, 0.1, 0.0, CM.rand(unquote(x), unquote(x))}
      end
      | rnn(x, m + 1, n - 1)
    ]
  end
end

defmodule Network do
  alias Cumatrix, as: CM

  @moduledoc """
  defnetwork generates neural network(nn)
  nn is list.
  e.g. [{:weight,w,ir,lr,v},{:bias,b,ir,lr},{:function,name}]
  each element is following.
  - weight
  {:weight,w,lr,v} w is matrix, ir is init rate for random element, lr is learning rate, v is for momentum,adagrad,adam
  - bias
  {:bias,b,lr,v} b is row vector
  - function
  {:function,name} 
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
      {:weight, CM.new(unquote(m)), 0.1, 0.1, 0.0, CM.new(1, 1)}
    end
  end

  def parse({:w, _, [x, y]}, _) do
    quote do
      {:weight, CM.rand(unquote(x), unquote(y)) |> CM.mult(0.1), 0.1, 0.11, 0.0,
       CM.new(unquote(x), unquote(y))}
    end
  end

  def parse({:w, _, [x, y, ir]}, _) do
    quote do
      {:weight, CM.rand(unquote(x), unquote(y)) |> CM.mult(unquote(ir)), unquote(ir), 0.1, 0.0,
       CM.new(unquote(x), unquote(y))}
    end
  end

  def parse({:w, _, [x, y, ir, lr]}, _) do
    quote do
      {:weight, CM.rand(unquote(x), unquote(y)) |> CM.mult(unquote(ir)), unquote(ir), unquote(lr),
       0.0, CM.new(unquote(x), unquote(y))}
    end
  end

  def parse({:w, _, [x, y, ir, lr, dr]}, _) do
    quote do
      {:weight, CM.rand(unquote(x), unquote(y)) |> CM.mult(unquote(ir)), unquote(ir), unquote(lr),
       unquote(dr), CM.new(unquote(x), unquote(y))}
    end
  end

  # bias
  # cb means constant bias for gradient check
  def parse({:cb, _, [m]}, _) do
    quote do
      {:bias, CM.new(unquote(m)), 0.1, 0.1, 0.0, CM.new(1, 1)}
    end
  end

  def parse({:b, _, [x]}, _) do
    quote do
      {:bias, CM.new(1, unquote(x)) |> CM.mult(0.1), 0.1, 0.1, 0.0, CM.new(1, unquote(x))}
    end
  end

  def parse({:b, _, [x, ir]}, _) do
    quote do
      {:bias, CM.rand(1, unquote(x)) |> CM.mult(unquote(ir)), unquote(ir), 0.1, 0.0,
       CM.new(1, unquote(x))}
    end
  end

  def parse({:b, _, [x, ir, lr]}, _) do
    quote do
      {:bias, CM.rand(1, unquote(x)) |> CM.mult(unquote(ir)), unquote(ir), unquote(lr), 0.0,
       CM.new(1, unquote(x))}
    end
  end

  def parse({:b, _, [x, ir, lr, dr]}, _) do
    quote do
      {:bias, CM.rand(1, unquote(x)) |> CM.mult(unquote(ir)), unquote(ir), unquote(lr),
       unquote(dr), CM.new(1, unquote(x))}
    end
  end

  # sigmoid
  def parse({:sigmoid, _, nil}, _) do
    quote do
      {:function, :sigmoid}
    end
  end

  # identity
  def parse({:tanh, _, nil}, _) do
    quote do
      {:function, :tanh}
    end
  end

  # relu
  def parse({:relu, _, nil}, _) do
    quote do
      {:function, :relu}
    end
  end

  # softmax
  def parse({:softmax, _, nil}, _) do
    quote do
      {:function, :softmax}
    end
  end

  # filter
  # cf means constant filter for gradient check
  def parse({:cf, _, [m]}, _) do
    quote do
      {:filter, CM.new(unquote(m)), 1, 0, 0.1, 0.1, CM.new(1, 3, 3)}
    end
  end

  # {:filter,filter-matrix,stride,padding,init_rate,learning_rate,dropout_rate,v}
  def parse({:f, _, [x, y]}, _) do
    quote do
      {:filter, CM.rand(1, 1, unquote(x), unquote(y)) |> CM.mult(0.1), 1, 0, 0.1, 0.1, 0.0,
       CM.new(1, 1, unquote(x), unquote(y))}
    end
  end

  def parse({:f, _, [x, y, c]}, _) do
    quote do
      {:filter, CM.rand(1, unquote(c), unquote(x), unquote(y)) |> CM.mult(0.1), 1, 0, 0.1, 0.1,
       0.0, CM.new(1, unquote(c), unquote(x), unquote(y))}
    end
  end

  def parse({:f, _, [x, y, c, n]}, _) do
    quote do
      {:filter, CM.rand(unquote(n), unquote(c), unquote(x), unquote(y)) |> CM.mult(0.1), 1, 0,
       0.0, 0.1, 0.1, CM.new(unquote(n), unquote(c), unquote(x), unquote(y))}
    end
  end

  def parse({:f, _, [x, y, c, n, st]}, _) do
    quote do
      {:filter, CM.rand(unquote(n), unquote(c), unquote(x), unquote(y)) |> CM.mult(0.1),
       unquote(st), 0, 0.1, 0.1, 0.0, CM.new(unquote(n), unquote(c), unquote(x), unquote(y))}
    end
  end

  def parse({:f, _, [x, y, c, n, st, pad]}, _) do
    quote do
      {:filter, CM.rand(unquote(n), unquote(c), unquote(x), unquote(y)) |> CM.mult(0.1),
       unquote(st), unquote(pad), 0.1, 0.1, 0.0,
       CM.new(unquote(n), unquote(c), unquote(x), unquote(y))}
    end
  end

  def parse({:f, _, [x, y, c, n, st, pad, ir, lr]}, _) do
    quote do
      {:filter, CM.rand(unquote(n), unquote(c), unquote(x), unquote(y)) |> CM.mult(unquote(ir)),
       unquote(st), unquote(pad), unquote(ir), unquote(lr), 0.0,
       CM.new(unquote(n), unquote(c), unquote(x), unquote(y))}
    end
  end

  def parse({:f, _, [x, y, c, n, st, pad, ir, lr, dr]}, _) do
    quote do
      {:filter, CM.rand(unquote(n), unquote(c), unquote(x), unquote(y)) |> CM.mult(unquote(ir)),
       unquote(st), unquote(pad), unquote(ir), unquote(lr), unquote(dr),
       CM.new(unquote(n), unquote(c), unquote(x), unquote(y))}
    end
  end

  # pooling
  def parse({:pooling, _, [x]}, _) do
    quote do
      {:pooling, unquote(x)}
    end
  end

  # flll connection
  def parse({:full, _, nil}, _) do
    quote do
      {:full}
    end
  end

  # analizer for debug
  def parse({:analizer, _, [x]}, _) do
    quote do
      {:analizer, unquote(x)}
    end
  end

  # visualizer for debug
  def parse({:visualizer, _, [n, c]}, _) do
    quote do
      {:visualizer, unquote(n), unquote(c)}
    end
  end

  def parse({x, _, nil}, _) do
    x
  end

  def parse({:|>, _, exp}, arg) do
    parse(exp, arg)
  end

  def parse([{arg, _, nil}, exp], arg) do
    [parse(exp, arg)]
  end

  def parse([exp1, exp2], arg) do
    Enum.reverse([parse(exp2, arg)] ++ Enum.reverse(parse(exp1, arg)))
  end

  def parse(x, _) do
    :io.write(x)
    raise "Syntax error in defnetwork"
  end
end

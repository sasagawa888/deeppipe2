
defmodule Network do
  @moduledoc """
  defnetwork generates neural network(nn)
  nn is list.
  e.g. [{:weight,w,lr,v},{:bias,b,lr},{:function,name}]
  each element is following.
  - weight
  {:weight,w,lr,v} w is matrix, lr is learning rate, v is for momentum,adagrad,adam
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

  # filter
  def parse({:f, _, [x, y]}, _) do
    quote do
      {:filter, Cumatrix.new(unquote(x), unquote(y), 0.1), 1, 0.1,
       Cumatrix.new(unquote(x), unquote(y),0.0)}
    end
  end

  def parse({:f, _, [x, y, lr]}, _) do
    quote do
      {:filter, Cumatrix.new(unquote(x), unquote(y), 0.1), 1, unquote(lr),
       Cumatrix.new(unquote(x), unquote(y),0.0)}
    end
  end

  def parse({:f, _, [x, y, lr, z]}, _) do
    quote do
      {:filter, Cumatrix.new(unquote(x), unquote(y), unquote(z)), 1, unquote(lr),
       Cumatrix.new(unquote(x), unquote(y), 0.0)}
    end
  end

  def parse({:f, _, [x, y, lr, z, st]}, _) do
    quote do
      {:filter, Cumatrix.new(unquote(x), unquote(y), unquote(z)), unquote(st), unquote(lr),
       Cumatrix.new(unquote(x), unquote(y), 0.0)}
    end
  end

  # pooling
  def parse({:pool, _, [x]}, _) do
    quote do
      {:pooling, unquote(x)}
    end
  end

  # padding
  def parse({:pad, _, [x]}, _) do
    quote do
      {:padding, unquote(x)}
    end
  end

  # constant weight for test
  def parse({:cw, _, [x]}, _) do
    quote do
      {:weight, Matrex.new(unquote(x)), 0.1, 0}
    end
  end

  # constant filter for test
  def parse({:cf, _, [x]}, _) do
    quote do
      {:filter, Matrex.new(unquote(x)), 1, 0.1, 0}
    end
  end

  # constant bias for test
  def parse({:cb, _, [x]}, _) do
    quote do
      {:bias, Matrex.new(unquote(x)), 0.1, 0}
    end
  end

  # weight
  def parse({:w, _, [x, y]}, _) do
    quote do
      {:weight, Cumatrix.new(unquote(x), unquote(y), 0.1), 0.1,
       Cumatrix.new(unquote(x), unquote(y),0.0)}
    end
  end

  def parse({:w, _, [x, y, lr]}, _) do
    quote do
      {:weight, Cumatrix.new(unquote(x), unquote(y), 0.1), unquote(lr),
       Cumatrix.new(unquote(x), unquote(y), 0.0)}
    end
  end

  def parse({:w, _, [x, y, lr, z]}, _) do
    quote do
      {:weight, Cumatrix.new(unquote(x), unquote(y), unquote(z)), unquote(lr),
       Cumatrix.new(unquote(x), unquote(y),0.0)}
    end
  end

  

  # bias
  def parse({:b, _, [x]}, _) do
    quote do
      {:bias, Cumatrix.new(1, unquote(x),0.0), 0.1, Cumatrix.new(1, unquote(x),0.0)}
    end
  end

  def parse({:b, _, [x, lr]}, _) do
    quote do
      {:bias, Cumatrix.new(1, unquote(x),0.0), unquote(lr), Cumatrix.new(1, unquote(x),0.0)}
    end
  end

  # sigmoid
  def parse({:sigmoid, _, nil}, _) do
    quote do
      {:function,:sigmoid}
    end
  end

  # identity
  def parse({:ident, _, nil}, _) do
    quote do
      {:function,:ident}
    end
  end

  # relu
  def parse({:relu, _, nil}, _) do
    quote do
      {:function,:relu}
    end
  end

  # softmax
  def parse({:softmax, _, nil}, _) do
    quote do
      {:function,:softmax}
    end 
  end

  # flatten
  def parse({:flatten, _, nil}, _) do
    quote do
      {:flatten}
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
    IO.puts("Syntax error in defnetwork")
  end

end 
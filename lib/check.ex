defmodule Check do
  import Network
  alias Deeppipe, as: DP
  alias Cumatrix, as: CM

  @moduledoc """
  gradient check 

  for debug
  """
  # for grad confirmation 
  defnetwork test_network0(_x) do
    _x
    |> f(3, 3, 1, 2, {1, 1}, 0, 0.1, 0.1)
    |> relu
    |> f(3, 3, 2, 2, {1, 1}, 0, 0.1, 0.1)
    |> relu
    |> f(3, 3, 2, 2, {1, 1}, 0, 0.1, 0.1)
    |> relu
    |> pooling(2, 2)
    |> full
    |> w(242, 10, 0.1, 0.1)
    |> softmax
  end

  def test() do
    data = CM.rand(2, 1, 28, 28)

    train =
      [
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
      ]
      |> CM.new()

    network = test_network0(0)
    IO.puts("compute numerical gradient")
    network1 = DP.numerical_gradient(data, network, train)
    IO.puts("compute backpropagation")
    network2 = DP.gradient(data, network, train)
    test1(network1, network2, 1)
  end

  defp test1([], [], _) do
    true
  end

  defp test1([{:filter, x, _, _, _, _, _, _} | xs], [{:filter, y, _, _, _, _, _, _} | ys], n) do
    if CM.is_near(x, y) == 1 do
      IO.write(n)
      IO.puts(" filter layer ok")
      test1(xs, ys, n + 1)
    else
      IO.write(n)
      IO.puts(" filter layer error")
      x |> CM.to_list() |> IO.inspect()
      y |> CM.to_list() |> IO.inspect()
      test1(xs, ys, n + 1)
    end
  end

  defp test1([{:weight, x, _, _, _, _} | xs], [{:weight, y, _, _, _, _} | ys], n) do
    if CM.is_near(x, y) == 1 do
      IO.write(n)
      IO.puts(" weight layer ok")
      test1(xs, ys, n + 1)
    else
      IO.write(n)
      IO.puts(" weight layer error")
      test1(xs, ys, n + 1)
    end
  end

  defp test1([{:bias, x, _, _, _, _} | xs], [{:bias, y, _, _, _, _} | ys], n) do
    if CM.is_near(x, y) == 1 do
      IO.write(n)
      IO.puts(" bias layer ok")
      test1(xs, ys, n + 1)
    else
      IO.write(n)
      IO.puts(" bias layer error")
      test1(xs, ys, n + 1)
    end
  end

  defp test1([x | xs], [_ | ys], n) do
    IO.write(n)
    IO.inspect(x)
    test1(xs, ys, n + 1)
  end

  # ---tested-------------

  defnetwork test_network1(_x) do
    _x
    |> f(2, 2, 1, 1, {1, 1}, 1)
    |> f(2, 2, 1, 1, {1, 1}, 1)
    |> pooling(2, 2)
    |> full
    |> softmax
  end

  defnetwork test_network2(_x) do
    _x
    |> f(2, 2, 1, 1, {2, 2}, 0)
    |> full
    |> softmax
  end

  defnetwork test_network4(_x) do
    _x
    |> f(2, 2, 2, 1, {1, 1})
    |> full
    |> w(4, 8)
    |> softmax
  end

  defnetwork test_network5(_x) do
    _x
    |> f(2, 2, 2, 2, {1, 1}, 1)
    |> full
    |> w(18, 4)
    |> softmax
  end

  defnetwork test_network3(_x) do
    _x
    |> f(2, 2, 2, 2, {1, 1}, 1)
    |> f(2, 2, 2, 1, {1, 1}, 1)
    |> pooling(2, 2)
    |> full
    |> softmax
  end

  defnetwork test_network6(_x) do
    _x
    |> f(2, 2, 2, 2, {1, 1}, 1)
    |> f(2, 2, 2, 2, {1, 1}, 1)
    |> pooling(2, 2)
    |> full
    |> w(32, 4)
    |> softmax
  end
end

defmodule Check do
  import Network
  alias Deeppipe, as: DP
  alias Cumatrix, as: CM

  defnetwork init_network1(_x) do
    _x
    |> f(3, 3, 3, 1, 1, 0.01, 0.001)
    |> pooling(2)
    |> f(5, 5, 1, 1, 1, 0.01, 0.01)
    |> f(5, 5, 1, 1, 1, 0.01, 0.01)
    |> sigmoid
    |> full
    |> w(144, 10, 0.1, 0.01)
    |> b(10, 0.1, 0.01)
    |> softmax
  end

  def fd() do
    data = CM.rand(1, 3, 32, 32)
    network = init_network1(0)
    [y | _] = DP.forward(data, network, [])
    y |> CM.to_list()
  end

  # for grad confirmation
  defnetwork test_network0(_x) do
    _x
    |> f(2, 2, 2)
    |> pooling(2)
    |> f(2, 2, 1, 2)
    |> full
    |> w(4, 4)
    |> b(4)
    |> softmax
  end

  def test() do
    data = CM.rand(1, 3, 32, 32) |> CM.mult(0.1)
    train = [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] |> CM.new()
    network1 = DP.numerical_gradient(data, init_network1(0), train)
    network2 = DP.gradient(data, init_network1(0), train)
    test1(network1, network2, 1)
  end

  def test1([], [], _) do
    true
  end

  def test1([{:filter, x, _, _, _, _, _} | xs], [{:filter, y, _, _, _, _, _} | ys], n) do
    if CM.is_near(x, y) do
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

  def test1([{:weight, x, _, _, _, _} | xs], [{:weight, y, _, _, _, _} | ys], n) do
    if CM.is_near(x, y) do
      IO.write(n)
      IO.puts(" weight layer ok")
      test1(xs, ys, n + 1)
    else
      IO.write(n)
      IO.puts(" weight layer error")
      x |> CM.to_list() |> IO.inspect()
      y |> CM.to_list() |> IO.inspect()
      test1(xs, ys, n + 1)
    end
  end

  def test1([{:bias, x, _, _, _, _} | xs], [{:bias, y, _, _, _, _} | ys], n) do
    if CM.is_near(x, y) do
      IO.write(n)
      IO.puts(" bias layer ok")
      test1(xs, ys, n + 1)
    else
      IO.write(n)
      IO.puts(" bias layer error")
      x |> CM.to_list() |> IO.inspect()
      y |> CM.to_list() |> IO.inspect()
      test1(xs, ys, n + 1)
    end
  end

  def test1([x | xs], [_ | ys], n) do
    IO.write(n)
    IO.inspect(x)
    test1(xs, ys, n + 1)
  end
end

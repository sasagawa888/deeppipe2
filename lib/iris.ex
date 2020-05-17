defmodule Iris do
  import Network
  alias Deeppipe, as: DP
  alias Cumatrix, as: CM

  defnetwork init_network0(_x) do
    _x
    |> w(4, 100)
    |> b(100)
    |> relu
    |> w(100, 50)
    |> b(50)
    |> relu
    |> w(50, 3)
    |> b(3)
    |> softmax
  end

  def sgd(m, n) do
    IO.puts("preparing data")
    image = train_image()
    label = train_label_onehot()
    network = init_network0(0)
    IO.puts("ready")
    network1 = sgd1(image, network, label, m, n)
    image1 = image |> CM.new()
    label1 = train_label()
    correct = DP.accuracy(image1, network1, label1)
    IO.write("accuracy rate = ")
    IO.puts(correct)
    IO.puts("end")
  end

  def sgd1(_, network, _, _, 0) do
    network
  end

  def sgd1(image, network, train, m, n) do
    {image1, train1} = DP.random_select(image, train, m, 150)
    network1 = DP.gradient(image1, network, train1)
    network2 = DP.learning(network, network1, :momentum)
    [y | _] = DP.forward(image1, network2, [])
    loss = CM.loss(y, train1, :cross)
    IO.puts(loss)
    sgd1(image, network2, train, m, n - 1)
  end

  def train_image() do
    {_, x} = File.read("iris/iris.data")

    x
    |> String.split("\n")
    |> Enum.take(150)
    |> Enum.map(fn y -> train_image1(y) end)
  end

  def train_image1(x) do
    x1 = x |> String.split(",") |> Enum.take(4)

    x1
    |> Enum.map(fn y -> String.to_float(y) end)
    |> DP.normalize(0, 1)
  end

  def train_label() do
    {_, x} = File.read("iris/iris.data")

    x
    |> String.split("\n")
    |> Enum.take(150)
    |> Enum.map(fn y -> train_label1(y) end)
  end

  def train_label1(x) do
    [x1] = x |> String.split(",") |> Enum.drop(4)

    cond do
      x1 == "Iris-setosa" -> 0
      x1 == "Iris-versicolor" -> 1
      x1 == "Iris-virginica" -> 2
    end
  end

  def train_label_onehot() do
    train_label() |> Enum.map(fn x -> DP.to_onehot(x, 2) end)
  end
end

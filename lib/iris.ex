defmodule Iris do
  import Network
  alias Deeppipe, as: DP

  @moduledoc """
  test with iris dataset
  """

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
    image = train_image()
    onehot = train_label_onehot()
    network = init_network0(0)
    test_image = image
    test_label = train_label()
    DP.train(network, image, onehot, test_image, test_label, :cross, :sgd, m, n)
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

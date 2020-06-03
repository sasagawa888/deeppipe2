defmodule Fashion do
  import Network
  alias Deeppipe, as: DP

  @moduledoc """
  test with Fashion-MNIST dataset.
  """

  # for DNN test sgd
  defnetwork init_network1(_x) do
    _x
    |> w(784, 300)
    |> b(300)
    |> tanh
    |> w(300, 100)
    |> b(100)
    |> tanh
    |> w(100, 10)
    |> b(10)
    |> softmax
  end
  # adagrad(300,14) acc=86.2% lr=0.008
  # for CNN test for Fashion-MNIST
  defnetwork init_network9(_x) do
    _x
    |> f(3, 3, 1, 32, {1, 1}, 0, {:he,784}, 0.008)
    |> relu
    |> f(3, 3, 32, 64, {1, 1}, 0, {:he,21632}, 0.008)
    |> relu
    |> pooling(2, 2)
    |> full
    |> w(9216, 128, {:he,9216}, 0.008, 0.25)
    |> w(128, 10, {:he,128}, 0.008, 0.25)
    |> softmax
  end

  def sgd(m, n) do
    image = train_image(60000, :flatten)
    onehot = train_label_onehot(60000)
    network = init_network1(0)
    test_image = test_image(10000, :flatten)
    test_label = test_label(10000)
    DP.train(network, image, onehot, test_image, test_label, :cross, :sgd, m, n)
  end

  def momentum(m, n) do
    image = train_image(10000, :structure)
    onehot = train_label_onehot(10000)

    network = init_network9(0)
    test_image = test_image(1000, :structure)
    test_label = test_label(1000)
    DP.train(network, image, onehot, test_image, test_label, :cross, :momentum, m, n)
  end

  def adagrad(m, n) do
    image = train_image(10000, :structure)
    onehot = train_label_onehot(10000)

    network = init_network9(0)
    test_image = test_image(1000, :structure)
    test_label = test_label(1000)
    DP.train(network, image, onehot, test_image, test_label, :cross, :adagrad, m, n)
  end

  def readagrad(m, n) do
    image = train_image(10000, :structure)
    onehot = train_label_onehot(10000)
    # test_image = train_image(10000, :structure)
    # test_label = train_label(10000)
    test_image = test_image(1000, :structure)
    test_label = test_label(1000)
    DP.retrain("temp.ex", image, onehot, test_image, test_label, :cross, :adagrad, m, n)
  end

  # Fashion-MNIST
  def try(m, n) do
    image = train_image(10000, :structure)
    onehot = train_label_onehot(10000)
    network = init_network9(0)
    test_image = test_image(100, :structure)
    test_label = test_label(100)
    DP.try(network, image, onehot, test_image, test_label, :cross, :adagrad, m, n)
  end

  def retry(m, n) do
    image = train_image(60000, :structure)
    onehot = train_label_onehot(60000)
    test_image = test_image(10000, :structure)
    test_label = test_label(10000)
    DP.retry("temp.ex", image, onehot, test_image, test_label, :cross, :adagrad, m, n)
  end

  # structure from flat vector to matrix(r,c) as 1 channel 
  def structure(x, r, c) do
    [structure1(x, r, c)]
  end

  def structure0(x, r, c) do
    structure1(x, r, c)
  end

  def structure1(_, 0, _) do
    []
  end

  def structure1(x, r, c) do
    [Enum.take(x, c) | structure1(Enum.drop(x, c), r - 1, c)]
  end

  # get n datas from train-label
  def train_label(n) do
    Enum.take(train_label(), n)
  end

  # transfer from train-label to onehot list
  def train_label_onehot(n) do
    Enum.take(train_label(), n) |> Enum.map(fn y -> DP.to_onehot(y, 9) end)
  end

  # get n datas from train-image with normalization
  def train_image(n, :structure) do
    train_image()
    |> Enum.take(n)
    |> Enum.map(fn x -> structure(DP.normalize(x, 0, 255), 28, 28) end)
  end

  # get n datas from train-image as flatten list
  def train_image(n, :flatten) do
    train_image()
    |> Enum.take(n)
    |> Enum.map(fn x -> DP.normalize(x, 0, 255) end)
  end

  # get n datas from test-label 
  def test_label(n) do
    Enum.take(test_label(), n)
  end

  # transfer from test-label to onehot list
  def test_label_onehot(n) do
    Enum.take(test_label(), n) |> Enum.map(fn y -> DP.to_onehot(y, 9) end)
  end

  # get n datas from test-image with normalization as structured list
  def test_image(n) do
    test_image()
    |> Enum.take(n)
    |> Enum.map(fn x -> structure(DP.normalize(x, 0, 255), 28, 28) end)
  end

  def test_image(n, :structure) do
    test_image()
    |> Enum.take(n)
    |> Enum.map(fn x -> structure(DP.normalize(x, 0, 255), 28, 28) end)
  end

  # get n datas from train-image as flatten list
  def test_image(n, :flatten) do
    test_image()
    |> Enum.take(n)
    |> Enum.map(fn x -> DP.normalize(x, 0, 255) end)
  end

  def train_label() do
    {:ok, <<0, 0, 8, 1, 0, 0, 234, 96, label::binary>>} =
      File.read("fashion/train-labels-idx1-ubyte")

    label |> String.to_charlist()
  end

  def train_image() do
    {:ok, <<0, 0, 8, 3, 0, 0, 234, 96, 0, 0, 0, 28, 0, 0, 0, 28, image::binary>>} =
      File.read("fashion/train-images-idx3-ubyte")

    byte_to_list(image)
  end

  def test_label() do
    {:ok, <<0, 0, 8, 1, 0, 0, 39, 16, label::binary>>} =
      File.read("fashion/t10k-labels-idx1-ubyte")

    label |> String.to_charlist()
  end

  def test_image() do
    {:ok, <<0, 0, 8, 3, 0, 0, 39, 16, 0, 0, 0, 28, 0, 0, 0, 28, image::binary>>} =
      File.read("fashion/t10k-images-idx3-ubyte")

    byte_to_list(image)
  end

  def byte_to_list(bin) do
    byte_to_list1(bin, 784, [], [])
  end

  def byte_to_list1(<<>>, _, ls, res) do
    [Enum.reverse(ls) | res] |> Enum.reverse()
  end

  def byte_to_list1(bin, 0, ls, res) do
    byte_to_list1(bin, 784, [], [Enum.reverse(ls) | res])
  end

  def byte_to_list1(<<b, bs::binary>>, n, ls, res) do
    byte_to_list1(bs, n - 1, [b | ls], res)
  end
end

defmodule Fashion do
  import Network
  alias Deeppipe, as: DP
  alias Cumatrix, as: CM

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
    # 28*28*1=784
    |> f(3, 3, 1, 32, {1, 1}, 0, {:he, 784}, 0.008)
    |> relu
    # 26*26*32=21632
    |> f(3, 3, 32, 64, {1, 1}, 0, {:he, 21632}, 0.008)
    |> relu
    |> pooling(2, 2)
    |> full
    |> w(9216, 128, {:he, 9216}, 0.008, 0.25)
    |> w(128, 10, {:he, 128}, 0.008, 0.25)
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
    image = train_image(60000, :structure)
    onehot = train_label_onehot(60000)

    network = init_network9(0)
    test_image = test_image(10000, :structure)
    test_label = test_label(10000)
    DP.train(network, image, onehot, test_image, test_label, :cross, :momentum, m, n)
  end

  def adagrad(m, n) do
    image = train_image(60000, :structure)
    onehot = train_label_onehot(60000)

    network = init_network9(0)
    test_image = test_image(10000, :structure)
    test_label = test_label(10000)
    DP.train(network, image, onehot, test_image, test_label, :cross, :adagrad, m, n)
  end

  def readagrad(m, n) do
    image = train_image(60000, :structure)
    onehot = train_label_onehot(60000)
    test_image = test_image(10000, :structure)
    test_label = test_label(10000)
    DP.retrain("temp.ex", image, onehot, test_image, test_label, :cross, :adagrad, m, n)
  end

  def adam(m, n) do
    image = train_image(60000, :structure)
    onehot = train_label_onehot(60000)

    network = init_network9(0)
    test_image = test_image(10000, :structure)
    test_label = test_label(10000)
    DP.train(network, image, onehot, test_image, test_label, :cross, :adam, m, n)
  end

  def try(m, n) do
    image = train_image(60000, :structure)
    onehot = train_label_onehot(60000)
    network = init_network9(0)
    test_image = test_image(1000, :structure)
    test_label = test_label(1000)
    DP.try(network, image, onehot, test_image, test_label, :cross, :adagrad, m, n)
  end

  def retry(m, n) do
    image = train_image(60000, :structure)
    onehot = train_label_onehot(60000)
    test_image = test_image(1000, :structure)
    test_label = test_label(1000)
    DP.retry("temp.ex", image, onehot, test_image, test_label, :cross, :adagrad, m, n)
  end

  
  @doc """
  get n datas from train-label
  """
  def train_label(n) do
    Enum.take(train_label(), n)
  end

  @doc """
  transfer from train-label to onehot list
  """
  def train_label_onehot(n) do
    Enum.take(train_label(), n) |> Enum.map(fn y -> DP.to_onehot(y, 9) end)
  end

  @doc """
  get n datas from train-image with normalization
  """
  def train_image(n, :structure) do
    train_image()
    |> Enum.take(n*28*28)
    |> DP.normalize(0,255)
    |> CM.reshape([n,1,28,28])
  end

  @doc """
  get n datas from train-image as flatten list
  """
  def train_image(n, :flatten) do
    train_image()
    |> Enum.take(n*784)
    |> DP.normalize(0, 255)
    |> CM.reshape([n,784])
  end

  @doc """
  get n datas from test-label 
  """
  def test_label(n) do
    Enum.take(test_label(), n)
  end

  @doc """
  transfer from test-label to onehot list
  """
  def test_label_onehot(n) do
    test_label()
    |> Enum.take(n) 
    |> Enum.map(fn y -> DP.to_onehot(y, 9) end)
  end

  @doc """
  get n datas from test-image with normalization as structured list
  """
  def test_image(n) do
    test_image()
    |> Enum.take(n*28*28)
    |> DP.normalize(0, 255)
    |> CM.reshape([n,1,28,28])
  end

  @doc """
  get n datas from test-image with normalization as structured list or matrix
  1st arg is size of data 
  2nd arg is :structure or :flatten
  """
  def test_image(n, :structure) do
    test_image()
    |> Enum.take(n*28*28)
    |> DP.normalize(0, 255)
    |> CM.reshape([n,1,28,28])
  end

  # get n datas from train-image as flatten list
  def test_image(n, :flatten) do
    test_image()
    |> Enum.take(n*784)
    |> DP.normalize(0, 255)
    |> CM.reshape([n,784])
  end

  @doc """
  get train label data
  """
  def train_label() do
    {:ok, <<0, 0, 8, 1, 0, 0, 234, 96, label::binary>>} =
      File.read("fashion/train-labels-idx1-ubyte")

    label |> String.to_charlist()
  end

  @doc """
  get train image data
  """
  def train_image() do
    {:ok, <<0, 0, 8, 3, 0, 0, 234, 96, 0, 0, 0, 28, 0, 0, 0, 28, image::binary>>} =
      File.read("fashion/train-images-idx3-ubyte")

    image |> :binary.bin_to_list()
  end

  @doc """
  get test label data
  """
  def test_label() do
    {:ok, <<0, 0, 8, 1, 0, 0, 39, 16, label::binary>>} =
      File.read("fashion/t10k-labels-idx1-ubyte")

    label |> String.to_charlist()
  end

  @doc """
  get test image data
  """
  def test_image() do
    {:ok, <<0, 0, 8, 3, 0, 0, 39, 16, 0, 0, 0, 28, 0, 0, 0, 28, image::binary>>} =
      File.read("fashion/t10k-images-idx3-ubyte")

    image |> :binary.bin_to_list()
  end

end

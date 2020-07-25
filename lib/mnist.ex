defmodule MNIST do
  @moduledoc """
  test with MNIST dataset
  """
  import Network
  alias Deeppipe, as: DP
  alias Cumatrix, as: CM

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

  # same meaning as init_network1
  defnetwork init_network11(_x) do
    _x
    |> w(784, 300, 0.1, 0.1, 0.0)
    |> b(300, 0.1, 0.1, 0.0)
    |> tanh
    |> w(300, 100, 0.1, 0.1, 0.0)
    |> b(100, 0.1, 0.0)
    |> tanh
    |> w(100, 10, 0.1, 0.1, 0.0)
    |> b(10, 0.1, 0.0)
    |> softmax
  end

  # for momentum
  defnetwork init_network2(_x) do
    _x
    |> w(784, 300)
    |> b(300)
    |> relu
    |> w(300, 100)
    |> b(100)
    |> sigmoid
    |> w(100, 10)
    |> b(10)
    |> softmax
  end

  # for adagrad/adam
  defnetwork init_network3(_x) do
    _x
    |> w(784, 300, 0.1, 0.01)
    |> b(300, 0.1, 0.01)
    |> relu
    |> w(300, 100, 0.1, 0.01)
    |> b(100, 0.1, 0.01)
    |> relu
    |> w(100, 10, 0.1, 0.01, 0.25)
    |> b(10, 0.1, 0.01)
    |> softmax
  end

  # CNN test for MNIST
  defnetwork init_network4(_x) do
    _x
    |> f(3, 3, 1, 6, {1, 1}, 0, {:he, 728}, 0.001)
    |> relu
    |> f(3, 3, 6, 12, {1, 1}, 0, {:he, 4056}, 0.001)
    |> relu
    |> pooling(2, 2)
    |> relu
    |> full
    |> w(1728, 10, {:he, 1728}, 0.001)
    |> softmax
  end

  # convolution filter (2,2) 1ch, stride=2
  defnetwork init_network5(_x) do
    _x
    |> f(2, 2, 1, 1, {2, 2})
    |> f(2, 2, 1, 1, {2, 2})
    |> full
    |> w(49, 10)
    |> softmax
  end

  # convolution filter (4,4) 1ch, stride=1, padding=1
  defnetwork init_network6(_x) do
    _x
    |> f(4, 4, 1, 1, {1, 1})
    |> full
    |> w(625, 300)
    |> b(300)
    |> relu
    |> w(300, 100)
    |> b(100)
    |> relu
    |> w(100, 10)
    |> b(10)
    |> softmax
  end

  # dropout test
  # dropout rate 25% initial-rate =0.1 learning-rate=0.1
  defnetwork init_network7(_x) do
    _x
    |> w(784, 300, 0.1, 0.1, 0.25)
    |> b(300)
    |> relu
    |> w(300, 100)
    |> b(100)
    |> relu
    |> w(100, 10)
    |> b(10)
    |> softmax
  end

  # long network test
  defnetwork init_network8(_x) do
    _x
    |> w(784, 600)
    |> b(600)
    |> relu
    |> w(600, 500)
    |> b(500)
    |> relu
    |> w(500, 400)
    |> b(400)
    |> relu
    |> w(400, 300)
    |> b(300)
    |> relu
    |> w(300, 100)
    |> b(100)
    |> relu
    |> w(100, 10)
    |> b(10)
    |> softmax
  end

  @doc """
  MNIST minibatch size , n epocs
  SGD optimizer
  """
  def sgd(m, n) do
    image = train_image(60000, :flatten)
    onehot = train_label_onehot(60000)
    network = init_network1(0)
    test_image = test_image(10000, :flatten)
    test_label = test_label(10000)
    DP.train(network, image, onehot, test_image, test_label, :cross, :sgd, m, n)
  end

  @doc """
  MNIST minibatch size , n epocs
  SGD optimizer
  initialize network from file temp.ex
  """
  def resgd(m, n) do
    image = train_image(60000, :flatten)
    onehot = train_label_onehot(60000)
    test_image = test_image(10000, :flatten)
    test_label = test_label(10000)
    DP.retrain("temp.ex", image, onehot, test_image, test_label, :cross, :sgd, m, n)
  end

  @doc """
  MNIST minibatch size , n epocs
  momentum optimizer
  """
  def momentum(m, n) do
    image = train_image(60000, :flatten)
    onehot = train_label_onehot(60000)
    network = init_network2(0)
    test_image = test_image(10000, :flatten)
    test_label = test_label(10000)
    DP.train(network, image, onehot, test_image, test_label, :cross, :momentum, m, n)
  end

  @doc """
  MNIST minibatch size , n epocs
  adagrad optimizer
  """
  def adagrad(m, n) do
    image = train_image(60000, :flatten)
    onehot = train_label_onehot(60000)
    network = init_network3(0)
    test_image = test_image(10000, :flatten)
    test_label = test_label(10000)
    DP.train(network, image, onehot, test_image, test_label, :cross, :adagrad, m, n)
  end

  @doc """
  MNIST minibatch size , n epocs
  RMSprop optimizer
  """
  def rms(m, n) do
    image = train_image(60000, :flatten)
    onehot = train_label_onehot(60000)
    network = init_network3(0)
    test_image = test_image(10000, :flatten)
    test_label = test_label(10000)
    DP.train(network, image, onehot, test_image, test_label, :cross, :rms, m, n)
  end

  @doc """
  MNIST minibatch size , n epocs
  adam optimizer
  """
  def adam(m, n) do
    image = train_image(60000, :flatten)
    onehot = train_label_onehot(60000)
    network = init_network3(0)
    test_image = test_image(10000, :flatten)
    test_label = test_label(10000)
    DP.train(network, image, onehot, test_image, test_label, :cross, :adam, m, n)
  end

  @doc """
  MNIST minibatch size , n epocs
  adam optimizer
  initialize network from file temp.ex
  """
  def readam(m, n) do
    image = train_image(60000, :flatten)
    onehot = train_label_onehot(60000)
    test_image = test_image(10000, :flatten)
    test_label = test_label(10000)
    DP.retrain("temp.ex", image, onehot, test_image, test_label, :cross, :adam, m, n)
  end

  @doc """
  MNIST minibatch size , n epocs
  Adam optimizer
  CNN network
  """
  def cnn(m, n) do
    image = train_image(60000, :structure)
    onehot = train_label_onehot(60000)
    network = init_network4(0)
    test_image = test_image(10000, :structure)
    test_label = test_label(10000)
    DP.train(network, image, onehot, test_image, test_label, :cross, :adam, m, n)
  end

  @doc """
  MNIST minibatch size , n epocs
  Adam optimizer
  CNN network
  initialize network from file temp.ex
  """
  def recnn(m, n) do
    image = train_image(60000, :structure)
    onehot = train_label_onehot(60000)
    test_image = test_image(10000, :structure)
    test_label = test_label(10000)
    DP.retrain("temp.ex", image, onehot, test_image, test_label, :cross, :adam, m, n)
  end

  @doc """
  MNIST minibatch size , n epocs
  SGD optimizer
  CNN network stride-test
  """
  def st(m, n) do
    image = train_image(60000, :structure)
    onehot = train_label_onehot(60000)
    network = init_network5(0)
    test_image = test_image(10000, :structure)
    test_label = test_label(10000)
    DP.train(network, image, onehot, test_image, test_label, :cross, :sgd, m, n)
  end

  @doc """
  MNIST minibatch size , n epocs
  SGD optimizer
  CNN network padding-test
  """
  def pad(m, n) do
    image = train_image(60000, :structure)
    onehot = train_label_onehot(60000)
    network = init_network6(0)
    test_image = test_image(10000, :structure)
    test_label = test_label(10000)
    DP.train(network, image, onehot, test_image, test_label, :cross, :sgd, m, n)
  end

  @doc """
  MNIST minibatch size , n epocs
  SGD optimizer
  CNN network dropout-test
  """
  def drop(m, n) do
    image = train_image(60000, :flatten)
    onehot = train_label_onehot(60000)
    network = init_network7(0)
    test_image = test_image(10000, :flatten)
    test_label = test_label(10000)
    DP.train(network, image, onehot, test_image, test_label, :cross, :sgd, m, n)
  end

  @doc """
  MNIST minibatch size , n epocs
  SGD optimizer
  long DNN test
  """
  def long(m, n) do
    image = train_image(60000, :flatten)
    onehot = train_label_onehot(60000)
    network = init_network8(0)
    test_image = test_image(10000, :flatten)
    test_label = test_label(10000)
    DP.train(network, image, onehot, test_image, test_label, :cross, :sgd, m, n)
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
    |> Enum.take(n * 28 * 28)
    |> DP.normalize(0, 255)
    |> CM.reshape([n, 1, 28, 28])
  end

  @doc """
  get n datas from train-image as flatten list
  """
  def train_image(n, :flatten) do
    train_image()
    |> Enum.take(n * 784)
    |> DP.normalize(0, 255)
    |> CM.reshape([n, 784])
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
    |> Enum.take(n * 28 * 28)
    |> DP.normalize(0, 255)
    |> CM.reshape([n, 1, 28, 28])
  end

  @doc """
  get n datas from test-image with normalization as structured list or matrix
  1st arg is size of data 
  2nd arg is :structure or :flatten
  """
  def test_image(n, :structure) do
    test_image()
    |> Enum.take(n * 28 * 28)
    |> DP.normalize(0, 255)
    |> CM.reshape([n, 1, 28, 28])
  end

  # get n datas from train-image as flatten list
  def test_image(n, :flatten) do
    test_image()
    |> Enum.take(n * 784)
    |> DP.normalize(0, 255)
    |> CM.reshape([n, 784])
  end

  @doc """
  get train label data
  """
  def train_label() do
    {:ok, <<0, 0, 8, 1, 0, 0, 234, 96, label::binary>>} =
      File.read("mnist/train-labels-idx1-ubyte")

    label |> String.to_charlist()
  end

  @doc """
  get train image data
  """
  def train_image() do
    {:ok, <<0, 0, 8, 3, 0, 0, 234, 96, 0, 0, 0, 28, 0, 0, 0, 28, image::binary>>} =
      File.read("mnist/train-images-idx3-ubyte")

    image |> :binary.bin_to_list()
  end

  @doc """
  get test label data
  """
  def test_label() do
    {:ok, <<0, 0, 8, 1, 0, 0, 39, 16, label::binary>>} = File.read("mnist/t10k-labels-idx1-ubyte")

    label |> String.to_charlist()
  end

  @doc """
  get test image data
  """
  def test_image() do
    {:ok, <<0, 0, 8, 3, 0, 0, 39, 16, 0, 0, 0, 28, 0, 0, 0, 28, image::binary>>} =
      File.read("mnist/t10k-images-idx3-ubyte")

    image |> :binary.bin_to_list()
  end
end

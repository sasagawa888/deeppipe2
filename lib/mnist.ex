defmodule MNIST do
  @moduledoc """
  test with MNIST dataset
  """
  import Network
  alias Deeppipe, as: DP

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

  # for adagrad
  defnetwork init_network3(_x) do
    _x
    |> w(784, 300, 0.1, 0.01)
    |> b(300, 0.1, 0.1)
    |> relu
    |> w(300, 100, 0.1, 0.01)
    |> b(100, 0.1, 0.1)
    |> relu
    |> w(100, 10, 0.1, 0.01,0.25)
    |> b(10, 0.1, 0.01)
    |> softmax
  end

  # for CNN test for MNIST
  defnetwork init_network4(_x) do
    _x
    # |> analizer(1)
    |> f(3, 3, 1, 6, {1, 1}, 0, 0.1, 0.001)
    |> f(3, 3, 6, 12, {1, 1}, 0, 0.1, 0.001)
    |> pooling(2, 2)
    |> relu
    # |> visualizer(1,1)
    |> full
    |> w(1728, 10, 0.1, 0.001)
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

  # for CNN test for Fashion-MNIST
  defnetwork init_network9(_x) do
    _x
    # |> analizer(1)
    |> f(5, 5, 1, 12, {1, 1}, 1, 0.1, 0.0005)
    |> pooling(2, 2)
    |> f(3, 3, 12, 12, {1, 1}, 1, 0.1, 0.0005)
    |> f(2, 2, 12, 12, {1, 1}, 1, 0.1, 0.0005)
    |> pooling(2, 2)
    |> f(3, 3, 12, 12, {1, 1}, 0, 0.1, 0.0005)
    |> relu
    # |> visualizer(1,1)
    |> full
    |> w(300, 10, 0.1, 0.0005)
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

  def resgd(m, n) do
    image = train_image(60000, :flatten)
    onehot = train_label_onehot(60000)
    test_image = test_image(10000, :flatten)
    test_label = test_label(10000)
    DP.retrain("temp.ex", image, onehot, test_image, test_label, :cross, :sgd, m, n)
  end

  def momentum(m, n) do
    image = train_image(60000, :flatten)
    onehot = train_label_onehot(60000)
    network = init_network2(0)
    test_image = test_image(10000, :flatten)
    test_label = test_label(10000)
    DP.train(network, image, onehot, test_image, test_label, :cross, :momentum, m, n)
  end

  def adagrad(m, n) do
    image = train_image(60000, :flatten)
    onehot = train_label_onehot(60000)
    network = init_network3(0)
    test_image = test_image(10000, :flatten)
    test_label = test_label(10000)
    DP.train(network, image, onehot, test_image, test_label, :cross, :adagrad, m, n)
  end

  def cnn(m, n) do
    image = train_image(60000, :structure)
    onehot = train_label_onehot(60000)
    network = init_network4(0)
    test_image = test_image(10000, :structure)
    test_label = test_label(10000)
    DP.train(network, image, onehot, test_image, test_label, :cross, :adagrad, m, n)
  end

  def recnn(m, n) do
    image = train_image(60000, :structure)
    onehot = train_label_onehot(60000)
    test_image = test_image(10000, :structure)
    test_label = test_label(10000)
    DP.retrain("temp.ex", image, onehot, test_image, test_label, :cross, :adagrad, m, n)
  end

  def st(m, n) do
    image = train_image(60000, :structure)
    onehot = train_label_onehot(60000)
    network = init_network5(0)
    test_image = test_image(10000, :structure)
    test_label = test_label(10000)
    DP.train(network, image, onehot, test_image, test_label, :cross, :sgd, m, n)
  end

  def pad(m, n) do
    image = train_image(60000, :structure)
    onehot = train_label_onehot(60000)
    network = init_network6(0)
    test_image = test_image(10000, :structure)
    test_label = test_label(10000)
    DP.train(network, image, onehot, test_image, test_label, :cross, :sgd, m, n)
  end

  def drop(m, n) do
    image = train_image(60000, :flatten)
    onehot = train_label_onehot(60000)
    network = init_network7(0)
    test_image = test_image(10000, :flatten)
    test_label = test_label(10000)
    DP.train(network, image, onehot, test_image, test_label, :cross, :sgd, m, n)
  end

  def long(m, n) do
    image = train_image(60000, :flatten)
    onehot = train_label_onehot(60000)
    network = init_network8(0)
    test_image = test_image(10000, :flatten)
    test_label = test_label(10000)
    DP.train(network, image, onehot, test_image, test_label, :cross, :sgd, m, n)
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
      File.read("mnist/train-labels-idx1-ubyte")

    label |> String.to_charlist()
  end

  def train_image() do
    {:ok, <<0, 0, 8, 3, 0, 0, 234, 96, 0, 0, 0, 28, 0, 0, 0, 28, image::binary>>} =
      File.read("mnist/train-images-idx3-ubyte")

    byte_to_list(image)
  end

  def test_label() do
    {:ok, <<0, 0, 8, 1, 0, 0, 39, 16, label::binary>>} = File.read("mnist/t10k-labels-idx1-ubyte")
    label |> String.to_charlist()
  end

  def test_image() do
    {:ok, <<0, 0, 8, 3, 0, 0, 39, 16, 0, 0, 0, 28, 0, 0, 0, 28, image::binary>>} =
      File.read("mnist/t10k-images-idx3-ubyte")

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

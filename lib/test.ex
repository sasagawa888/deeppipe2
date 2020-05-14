defmodule Test do
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
    |> w(784, 300, 0.1, 0.1)
    |> b(300, 0.1, 0.1)
    |> relu
    |> w(300, 100, 0.1, 0.1)
    |> b(100, 0.3, 0.1)
    |> relu
    |> w(100, 10, 0.1, 0.1)
    |> b(10, 0.1, 0.1)
    |> softmax
  end

  # for CNN test for MNIST
  defnetwork init_network4(_x) do
    _x
    # |> visualizer(1,1)
    |> f(5, 5, 1, 12, {1, 1}, 1, 0.5, 0.0001)
    |> pooling(2, 2)
    |> f(3, 3, 12, 12, {1, 1}, 1, 0.5, 0.0001)
    |> f(2, 2, 12, 12, {1, 1}, 1, 0.5, 0.0001)
    |> pooling(2, 2)
    |> f(3, 3, 12, 12, {1, 1}, 0, 0.5, 0.0001)
    |> relu
    # |> visualizer(1,1)
    |> full
    |> w(300, 10, 0.1, 0.001)
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
  # dropout rate 50% initial-rate =0.1 learning-rate=0.1
  defnetwork init_network7(_x) do
    _x
    |> w(784, 300, 0.1, 0.1, 0.5)
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

  def sgd(m, n) do
    image = MNIST.train_image(60000, :flatten)
    onehot = MNIST.train_label_onehot(60000)
    network = init_network1(0)
    test_image = MNIST.test_image(2000, :flatten)
    test_label = MNIST.test_label(2000)
    DP.train(network, image, onehot, test_image, test_label, :cross, :sgd, m, n)
  end

  def momentum(m, n) do
    image = MNIST.train_image(3000, :flatten)
    onehot = MNIST.train_label_onehot(3000)
    network = init_network2(0)
    test_image = MNIST.test_image(1000, :flatten)
    test_label = MNIST.test_label(1000)
    DP.train(network, image, onehot, test_image, test_label, :cross, :momentum, m, n)
  end

  def adagrad(m, n) do
    image = MNIST.train_image(3000, :flatten)
    onehot = MNIST.train_label_onehot(3000)
    network = init_network3(0)
    test_image = MNIST.test_image(1000, :flatten)
    test_label = MNIST.test_label(1000)
    DP.train(network, image, onehot, test_image, test_label, :cross, :adagrad, m, n)
  end

  def cnn(m, n) do
    image = MNIST.train_image(3000, :structure)
    onehot = MNIST.train_label_onehot(3000)
    network = init_network4(0)
    test_image = MNIST.test_image(1000, :structure)
    test_label = MNIST.test_label(1000)
    DP.train(network, image, onehot, test_image, test_label, :cross, :sgd, m, n)
  end

  def recnn(m, n) do
    image = MNIST.train_image(3000, :structure)
    onehot = MNIST.train_label_onehot(3000)
    test_image = MNIST.test_image(1000, :structure)
    test_label = MNIST.test_label(1000)
    DP.retrain("temp.ex", image, onehot, test_image, test_label, :cross, :sgd, m, n)
  end

  def st(m, n) do
    image = MNIST.train_image(3000, :structure)
    onehot = MNIST.train_label_onehot(3000)
    network = init_network5(0)
    test_image = MNIST.test_image(1000, :structure)
    test_label = MNIST.test_label(1000)
    DP.train(network, image, onehot, test_image, test_label, :cross, :sgd, m, n)
  end

  def pad(m, n) do
    image = MNIST.train_image(3000, :structure)
    onehot = MNIST.train_label_onehot(3000)
    network = init_network6(0)
    test_image = MNIST.test_image(1000, :structure)
    test_label = MNIST.test_label(1000)
    DP.train(network, image, onehot, test_image, test_label, :cross, :sgd, m, n)
  end

  def drop(m, n) do
    image = MNIST.train_image(3000, :flatten)
    onehot = MNIST.train_label_onehot(3000)
    network = init_network7(0)
    test_image = MNIST.test_image(1000, :flatten)
    test_label = MNIST.test_label(1000)
    DP.train(network, image, onehot, test_image, test_label, :cross, :sgd, m, n)
  end

  def long(m, n) do
    image = MNIST.train_image(3000, :flatten)
    onehot = MNIST.train_label_onehot(3000)
    network = init_network8(0)
    test_image = MNIST.test_image(1000, :flatten)
    test_label = MNIST.test_label(1000)
    DP.train(network, image, onehot, test_image, test_label, :cross, :sgd, m, n)
  end

  def resgd(m, n) do
    image = MNIST.train_image(3000, :flatten)
    onehot = MNIST.train_label_onehot(3000)
    test_image = MNIST.test_image(1000, :flatten)
    test_label = MNIST.test_label(1000)
    DP.retrain("temp.ex", image, onehot, test_image, test_label, :cross, :sgd, m, n)
  end

  # Fashon-MNIST
  def fashion(m, n) do
    image = Fashon.train_image(3000, :structure)
    onehot = Fashon.train_label_onehot(3000)
    network = init_network4(0)
    test_image = Fashon.test_image(1000, :structure)
    test_label = Fashon.test_label(1000)
    DP.train(network, image, onehot, test_image, test_label, :cross, :sgd, m, n)
  end

  def refashion(m, n) do
    image = Fashon.train_image(3000, :structure)
    onehot = Fashon.train_label_onehot(3000)
    test_image = Fashon.test_image(1000, :structure)
    test_label = Fashon.test_label(1000)
    DP.retrain("temp.ex", image, onehot, test_image, test_label, :cross, :sgd, m, n)
  end
end

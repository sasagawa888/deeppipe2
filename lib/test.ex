defmodule Test do
  import Network
  alias Deeppipe, as: DP
  alias Cumatrix, as: CM

  # for DNN test
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

  defnetwork init_network3(_x) do
    _x
    |> w(784, 300, 1.0, 0.2)
    |> b(300, 1.0, 0.2)
    |> sigmoid
    |> w(300, 100, 1.0, 0.1)
    |> b(100, 1.0, 0.1)
    |> tanh
    |> w(100, 10)
    |> b(10)
    |> softmax
  end

  # for CNN test
  defnetwork init_network4(_x) do
    _x
    |> f(3, 3, 1, 12, 1, 0, 0.1, 0.01)
    |> f(5, 5, 12, 12, 1, 0, 0.1, 0.01)
    |> full
    |> w(5808, 10)
    |> b(10)
    # |> analizer(1)
    |> softmax
  end

  # convolution filter (2,2) 1ch, stride=2
  defnetwork init_network5(_x) do
    _x
    |> f(2, 2, 1, 1, 2)
    |> f(2, 2, 1, 1, 2)
    |> full
    |> w(49, 10)
    |> softmax
  end

  # convolution filter (4,4) 1ch, stride=1, padding=1
  defnetwork init_network6(_x) do
    _x
    |> f(4, 4, 1, 1, 1)
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
    IO.puts("preparing data")
    image = MNIST.train_image(60000, :flatten) |> CM.new()
    label = MNIST.train_label_onehot(60000) |> CM.new()
    network = init_network1(0)
    IO.puts("ready")
    network1 = sgd1(image, network, label, m, n)
    test_image = MNIST.test_image(2000, :flatten) |> CM.new()
    test_label = MNIST.test_label(2000)
    correct = DP.accuracy(test_image, network1, test_label)
    IO.write("accuracy rate = ")
    IO.puts(correct)
    IO.puts("end")
  end

  def sgd1(_, network, _, _, 0) do
    network
  end

  def sgd1(image, network, train, m, n) do
    {image1, train1} = CM.random_select(image, train, m)
    network1 = DP.gradient(image1, network, train1)
    network2 = DP.learning(network, network1)
    [y | _] = DP.forward(image1, network2, [])
    loss = CM.loss(y, train1, :cross)
    IO.puts(loss)
    sgd1(image, network2, train, m, n - 1)
  end

  def momentum(m, n) do
    IO.puts("preparing data")
    image = MNIST.train_image(3000, :flatten) |> CM.new()
    label = MNIST.train_label_onehot(3000) |> CM.new()
    network = init_network2(0)
    IO.puts("ready")
    network1 = sgd1(image, network, label, m, n)
    test_image = MNIST.test_image(1000, :flatten) |> CM.new()
    test_label = MNIST.test_label(1000)
    correct = DP.accuracy(test_image, network1, test_label)
    IO.write("accuracy rate = ")
    IO.puts(correct)
    IO.puts("end")
  end

  def momentum1(_, network, _, _, 0) do
    network
  end

  def momentum1(image, network, train, m, n) do
    {image1, train1} = DP.random_select(image, train, m, 2000)
    network1 = DP.gradient(image1, network, train1)
    network2 = DP.learning(network, network1, :momentum)
    [y | _] = DP.forward(image1, network2, [])
    loss = CM.loss(y, train1, :cross)
    IO.puts(loss)
    sgd1(image, network2, train, m, n - 1)
  end

  def adagrad(m, n) do
    IO.puts("preparing data")
    image = MNIST.train_image(3000, :flatten) |> CM.new()
    label = MNIST.train_label_onehot(3000) |> CM.new()
    network = init_network3(0)
    IO.puts("ready")
    network1 = sgd1(image, network, label, m, n)
    test_image = MNIST.test_image(1000, :flatten) |> CM.new()
    test_label = MNIST.test_label(1000)
    correct = DP.accuracy(test_image, network1, test_label)
    IO.write("accuracy rate = ")
    IO.puts(correct)
    IO.puts("end")
  end

  def adagrad1(_, network, _, _, 0) do
    network
  end

  def adagrad1(image, network, train, m, n) do
    {image1, train1} = DP.random_select(image, train, m, 2000)
    network1 = DP.gradient(image1, network, train1)
    network2 = DP.learning(network, network1, :adagrad)
    [y | _] = DP.forward(image1, network2, [])
    loss = CM.loss(y, train1, :cross)
    IO.puts(loss)
    sgd1(image, network2, train, m, n - 1)
  end

  def cnn(m, n) do
    IO.puts("preparing data")
    image = MNIST.train_image(3000, :structure) |> CM.new()
    label = MNIST.train_label_onehot(3000) |> CM.new()
    network = init_network4(0)
    IO.puts("ready")
    network1 = cnn1(image, network, label, m, n)
    test_image = MNIST.test_image(1000, :structure) |> CM.new()
    test_label = MNIST.test_label(1000)
    correct = DP.accuracy(test_image, network1, test_label)
    IO.write("accuracy rate = ")
    IO.puts(correct)
    IO.puts("end")
  end

  def cnn1(_, network, _, _, 0) do
    network
  end

  def cnn1(image, network, train, m, n) do
    {image1, train1} = CM.random_select(image, train, m)
    network1 = DP.gradient(image1, network, train1)
    network2 = DP.learning(network, network1)
    [y | _] = DP.forward(image1, network2, [])
    loss = CM.loss(y, train1, :cross)
    IO.puts(loss)
    cnn1(image, network2, train, m, n - 1)
  end

  def st(m, n) do
    DP.gbc()
    IO.puts("preparing data")
    image = MNIST.train_image(3000, :structure) |> CM.new()
    label = MNIST.train_label_onehot(3000) |> CM.new()
    network = init_network5(0)
    IO.puts("ready")
    network1 = cnn1(image, network, label, m, n)
    test_image = MNIST.test_image(1000, :structure) |> CM.new()
    test_label = MNIST.test_label(1000)
    correct = DP.accuracy(test_image, network1, test_label)
    IO.write("accuracy rate = ")
    IO.puts(correct)
    IO.puts("end")
  end

  def pad(m, n) do
    IO.puts("preparing data")
    image = MNIST.train_image(3000, :structure) |> CM.new()
    label = MNIST.train_label_onehot(3000) |> CM.new()
    network = init_network6(0)
    IO.puts("ready")
    network1 = cnn1(image, network, label, m, n)
    test_image = MNIST.test_image(1000, :structure) |> CM.new()
    test_label = MNIST.test_label(1000)
    correct = DP.accuracy(test_image, network1, test_label)
    IO.write("accuracy rate = ")
    IO.puts(correct)
    IO.puts("end")
  end

  def drop(m, n) do
    IO.puts("preparing data")
    image = MNIST.train_image(3000, :flatten) |> CM.new()
    label = MNIST.train_label_onehot(3000) |> CM.new()
    network = init_network7(0)
    IO.puts("ready")
    network1 = drop1(image, network, label, m, n)
    test_image = MNIST.test_image(1000, :flatten) |> CM.new()
    test_label = MNIST.test_label(1000)
    correct = DP.accuracy(test_image, network1, test_label)
    IO.write("accuracy rate = ")
    IO.puts(correct)
    IO.puts("end")
  end

  def drop1(_, network, _, _, 0) do
    network
  end

  def drop1(image, network, train, m, n) do
    {image1, train1} = CM.random_select(image, train, m)
    network1 = DP.gradient(image1, network, train1)
    network2 = DP.learning(network, network1)
    [y | _] = DP.forward(image1, network2, [])
    loss = CM.loss(y, train1, :cross)
    IO.puts(loss)
    drop1(image, network2, train, m, n - 1)
  end

  def long(m, n) do
    IO.puts("preparing data")
    image = MNIST.train_image(3000, :flatten) |> CM.new()
    label = MNIST.train_label_onehot(3000) |> CM.new()
    network = init_network8(0)
    IO.puts("ready")
    network1 = drop1(image, network, label, m, n)
    test_image = MNIST.test_image(1000, :flatten) |> CM.new()
    test_label = MNIST.test_label(1000)
    correct = DP.accuracy(test_image, network1, test_label)
    IO.write("accuracy rate = ")
    IO.puts(correct)
    IO.puts("end")
  end

  def long1(_, network, _, _, 0) do
    network
  end

  def long1(image, network, train, m, n) do
    {image1, train1} = CM.random_select(image, train, m)
    network1 = DP.gradient(image1, network, train1)
    network2 = DP.learning(network, network1)
    [y | _] = DP.forward(image1, network2, [])
    loss = CM.loss(y, train1, :cross)
    IO.puts(loss)
    drop1(image, network2, train, m, n - 1)
  end
end

defmodule Test do
  import Network
  alias Deeppipe, as: DP
  alias Cumatrix, as: CM


  # for DNN test
  defnetwork init_network1(_x) do
    _x |> w(784,300) |> b(300) |> relu
    |> w(300,100) |> b(100) |> relu
    |> w(100,10) |> b(10) |> softmax
  end

  

  def sgd(m, n) do
    IO.puts("preparing data")
    image = MNIST.train_image(3000,:flatten)
    label = MNIST.train_label_onehot(3000)
    network = init_network1(0)
    IO.puts("ready")
    network1 = sgd1(image, network, label, m, n)
    test_image = MNIST.test_image(1000,:flatten) |> CM.new()
    test_label = MNIST.test_label(1000)
    correct = DP.accuracy(test_image, network1, test_label)
    IO.write("accuracy rate = ")
    IO.puts(correct / 1000)
    IO.puts("end")
  end

  def sgd1(_, network, _, _, 0) do
    network
  end

  def sgd1(image, network, train, m, n) do
    {image1, train1} = DP.random_select(image, train, m, 2000)
    network1 = DP.gradient(image1, network, train1)
    network2 = DP.learning(network, network1,:momentum)
    [y|_] = DP.forward(image1, network2 ,[])
    loss = CM.loss(y, train1, :cross)
    DP.print(loss)
    DP.newline()
    sgd1(image, network2, train, m, n - 1)
  end


  def momentum(m, n) do
    IO.puts("preparing data")
    image = MNIST.train_image(3000,:flatten)
    label = MNIST.train_label_onehot(3000)
    network = init_network1(0)
    IO.puts("ready")
    network1 = sgd1(image, network, label, m, n)
    test_image = MNIST.test_image(1000,:flatten) |> CM.new()
    test_label = MNIST.test_label(1000)
    correct = DP.accuracy(test_image, network1, test_label)
    IO.write("accuracy rate = ")
    IO.puts(correct / 1000)
    IO.puts("end")
  end

  def momentum1(_, network, _, _, 0) do
    network
  end

  def momentum1(image, network, train, m, n) do
    {image1, train1} = DP.random_select(image, train, m, 2000)
    network1 = DP.gradient(image1, network, train1)
    network2 = DP.learning(network, network1,:momentum)
    [y|_] = DP.forward(image1, network2 ,[])
    loss = CM.loss(y, train1, :cross)
    DP.print(loss)
    DP.newline()
    sgd1(image, network2, train, m, n - 1)
  end

  def adagrad(m, n) do
    IO.puts("preparing data")
    image = MNIST.train_image(3000,:flatten)
    label = MNIST.train_label_onehot(3000)
    network = init_network1(0)
    IO.puts("ready")
    network1 = sgd1(image, network, label, m, n)
    test_image = MNIST.test_image(1000,:flatten) |> CM.new()
    test_label = MNIST.test_label(1000)
    correct = DP.accuracy(test_image, network1, test_label)
    IO.write("accuracy rate = ")
    IO.puts(correct / 1000)
    IO.puts("end")
  end

  def adagrad1(_, network, _, _, 0) do
    network
  end

  def adagrad1(image, network, train, m, n) do
    {image1, train1} = DP.random_select(image, train, m, 2000)
    network1 = DP.gradient(image1, network, train1)
    network2 = DP.learning(network, network1,:adagrad)
    [y|_] = DP.forward(image1, network2 ,[])
    loss = CM.loss(y, train1, :cross)
    DP.print(loss)
    DP.newline()
    sgd1(image, network2, train, m, n - 1)
  end

end

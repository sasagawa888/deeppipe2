defmodule Test do
  import Network
  alias Deeppipe, as: DP
  alias Cumatrix, as: CM


  # for DNN test
  defnetwork init_network1(_x) do
    _x |> w(784,300,1.0) |> b(300,1.0) |> sigmoid
    |> w(300,100,1.0) |> b(100,1.0) |> sigmoid
    |> w(100,10,0.5) |> b(10,0.5) |> softmax
  end


  def sgd(m, n) do
    IO.puts("preparing data")
    image = MNIST.train_image(3000,:flatten)
    label = MNIST.train_label_onehot(3000)
    network = init_network1(0)
    test_image = MNIST.test_image(1000,:flatten) |> CM.new()
    test_label = MNIST.test_label(1000)
    IO.puts("ready")
    network1 = sgd1(image, network, label, m, n)
    correct = DP.accuracy(test_image, network1, test_label)
    IO.write("accuracy rate = ")
    IO.puts(correct / 1000)
  end

  def sgd1(_, network, _, _, 0) do
    network
  end

  def sgd1(image, network, train, m, n) do
    {image1, train1} = DP.random_select(image, train, m, 2000)
    network1 = DP.gradient(image1, network, train1)
    network2 = DP.learning(network, network1)
    [y|_] = DP.forward(image1, network2 ,[])
    loss = DP.loss(y, train1)
    DP.print(loss)
    DP.newline()
    sgd1(image, network2, train, m, n - 1)
  end
end

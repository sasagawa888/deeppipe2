defmodule Test do
  import Network
  alias Deeppipe, as: DP
  alias Cumatrix, as: CM


  # debug
  defnetwork debug(_x) do
    _x
    |> cw([[0.5,0.7,0.2],
           [0.1,0.2,0.3],
           [0.3,0.4,0.1]])
    |> cb([[0.1,0.2,0.3]])
    |> softmax
  end 

  def test() do
    y = CM.new([[0.6,0.5,0.4]])
    t = CM.new([[0.0,0.1,0.0]])
    DP.gradient(y,debug(0),t)
  end 

  # for DNN test
  defnetwork init_network1(_x) do
    _x
    |> w(784, 300, 0.2)
    |> b(300, 0.2)
    |> sigmoid
    |> w(300, 100, 0.2)
    |> b(100, 0.2)
    |> sigmoid
    |> w(100, 10, 0.2)
    |> b(10, 0.2)
    |> softmax
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

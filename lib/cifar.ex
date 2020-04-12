defmodule CIFAR do
  import Network
  alias Deeppipe, as: DP
  alias Cumatrix, as: CM
  

  # for CNN test
  defnetwork init_network1(_x) do
    _x
    |> f(3, 3, 3, 1, 1)
    |> pooling(2)
    |> f(5, 5, 1, 1, 2)
    |> pooling(2)
    |> f(3, 3, 1, 1, 1)
    |> f(3, 3, 1, 1, 1)
    |> pooling(2)
    |> full
    |> sigmoid
    |> w(16, 10)
    |> b(10)
    |> softmax
  end

  def sgd(m, n) do
    IO.puts("preparing data")
    image = train_image(3000) |> CM.new()
    label = train_label_onehot(3000) |> CM.new()
    network = init_network1(0)
    IO.puts("ready")
    network1 = sgd1(image, network, label, m, n)
    test_image = test_image(30) |> CM.new()
    test_label = test_label(30)
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
    # [x | _] = DP.forward(image1, network, [])
    # CM.print(x)
    network1 = DP.gradient(image1, network, train1)
    network2 = DP.learning(network, network1)
    # DP.print(network2)
    [y | _] = DP.forward(image1, network2, [])
    # CM.print(z)
    loss = CM.loss(y, train1, :cross)
    IO.puts(loss)
    # IO.puts(n)
    sgd1(image, network2, train, m, n - 1)
  end

  # transfer from train-label to onehot list
  def train_label_onehot(n) do
    Enum.take(test_label(), n) |> Enum.map(fn y -> DP.to_onehot(y, 9) end)
  end

  # transfer from test-label to onehot list
  def test_label_onehot(n) do
    Enum.take(test_label(), n) |> Enum.map(fn y -> DP.to_onehot(y, 9) end)
  end

  def train_label() do
    {:ok, <<label, rest::binary>>} = File.read("data_batch_1.bin")
    [label | train_label1(rest)]
  end

  # 36*36*3 = 3072
  def train_label1(<<>>) do
    []
  end

  def train_label1(x) do
    result = train_label2(x, 3072)

    if result != <<>> do
      <<label, rest::binary>> = result
      [label | train_label1(rest)]
    else
      []
    end
  end

  def test_label(n) do
    test_label() |> Enum.take(n)
  end

  def test_label() do
    {:ok, <<label, rest::binary>>} = File.read("test_batch.bin")
    [label | train_label1(rest)]
  end

  # skip data
  def train_label2(<<rest::binary>>, 0) do
    rest
  end

  def train_label2(<<_, rest::binary>>, n) do
    train_label2(rest, n - 1)
  end

  def train_image(n) do
    train_image() |> Enum.take(n)
  end

  def train_image() do
    {:ok, bin} = File.read("data_batch_1.bin")
    train_image1(bin)
  end

  def test_image(n) do
    train_image() |> Enum.take(n)
  end

  def test_image() do
    {:ok, bin} = File.read("test_batch.bin")
    train_image1(bin)
  end

  # get RGB 3ch data
  def train_image1(<<>>) do
    []
  end

  def train_image1(<<_, rest::binary>>) do
    {image, other} = train_image2(rest, 3, [])
    [image | train_image1(other)]
  end

  # get one RGB data
  def train_image2(x, 0, res) do
    {Enum.reverse(res), x}
  end

  def train_image2(x, n, res) do
    {image, rest} = train_image3(x, 32, [])
    train_image2(rest, n - 1, [image | res])
  end

  # get one image 2D data
  def train_image3(x, 0, res) do
    {Enum.reverse(res), x}
  end

  def train_image3(x, n, res) do
    {image, rest} = train_image4(x, 32, [])
    train_image3(rest, n - 1, [image | res])
  end

  # get one row vector
  def train_image4(x, 0, res) do
    # {Enum.reverse(res) , x}
    {Enum.reverse(res) |> DP.normalize(-128, 256), x}
  end

  def train_image4(<<x, xs::binary>>, n, res) do
    train_image4(xs, n - 1, [x | res])
  end
end

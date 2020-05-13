defmodule CIFAR do
  import Network
  alias Deeppipe, as: DP
  alias Cumatrix, as: CM

  # for CNN test
  # CIFAR.momentum(400,500) 20epochs mini batch size 400

  defnetwork init_network1(_x) do
    _x
    #|> analizer(1)
    |> f(3, 3, 3, 8, 1, 1, 0.3, 0.01)
    #|> analizer(2)
    |> relu
    #|> analizer(3)
    |> f(3, 3, 8, 8, 1, 1, 0.3, 0.01)
    #|> analizer(4)
    |> pooling(2)
    #|> analizer(5)
    |> f(3, 3, 8, 16, 1, 1, 0.3, 0.01)
    #|> analizer(6)
    |> relu
    #|> analizer(7)
    |> f(3, 3, 16, 16, 1, 1, 0.3, 0.01)
    #|> analizer(8)
    |> relu
    #|> analizer(9)
    |> poolingt(2)
    |> full
    |> w(1024, 10, 0.3, 0.01)
    |> b(10, 0.3, 0.01)
    |> softmax
  end

  # sgd/2 train network and save network temp.ex
  def sgd(m, n) do
    image = train_image(10000)
    onehot = train_label_onehot(10000)
    test_image = test_image(300)
    test_label = test_label(300)
    network = init_network1(0)
    DP.train(network, image, onehot, test_image, test_label, :cross, :sgd, m, n)
  end

  # resgd/2 load network from temp.ex and restart training
  def resgd(m, n) do
    image = train_image(10000)
    onehot = train_label_onehot(10000)
    test_image = test_image(300)
    test_label = test_label(300)
    #test_image = train_image(300)
    #test_label = train_label(300)
    DP.retrain("temp.ex", image, onehot, test_image, test_label, :cross, :sgd, m, n)
  end

  # momentum/2 train network and save network temp.ex
  def momentum(m, n) do
    image = train_image(10000)
    onehot = train_label_onehot(10000)
    test_image = test_image(300)
    test_label = test_label(300)
    network = init_network1(0)
    DP.train(network, image, onehot, test_image, test_label, :cross, :momentum, m, n)
  end

  # remomentum/2 load network from temp.ex and restart training
  def remomentum(m, n) do
    image = train_image(10000)
    onehot = train_label_onehot(10000)
    test_image = test_image(300)
    test_label = test_label(300)
    #test_image = train_image(300)
    #test_label = train_label(300)
    DP.retrain("temp.ex", image, onehot, test_image, test_label, :cross, :momentum, m, n)
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

  def train_label(n) do
    train_label() |> Enum.take(n)
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
    {Enum.reverse(res) |> DP.normalize(-128, 128), x}
  end

  def train_image4(<<x, xs::binary>>, n, res) do
    train_image4(xs, n - 1, [x | res])
  end

  def heatmap(n) do
    train_rgb(n) |> Matrex.new() |> Matrex.heatmap(:color24bit, [])
  end

  def heatmapr(n) do
    train_r(n) |> Matrex.new() |> Matrex.heatmap(:color24bit, [])
  end

  def heatmapg(n) do
    train_g(n) |> Matrex.new() |> Matrex.heatmap(:color24bit, [])
  end

  def heatmapb(n) do
    train_b(n) |> Matrex.new() |> Matrex.heatmap(:color24bit, [])
  end

  def train_rgb(n) do
    {:ok, bin} = File.read("data_batch_1.bin")
    train_rgb1(bin) |> CM.nth(n) |> composit() |> CM.reshape([32, 32])
  end

  def train_r(n) do
    {:ok, bin} = File.read("data_batch_1.bin")
    train_rgb1(bin) |> CM.nth(n) |> CM.nth(1) |> CM.reshape([32, 32])
  end

  def train_g(n) do
    {:ok, bin} = File.read("data_batch_1.bin")
    train_rgb1(bin) |> CM.nth(n) |> CM.nth(2) |> CM.reshape([32, 32])
  end

  def train_b(n) do
    {:ok, bin} = File.read("data_batch_1.bin")
    train_rgb1(bin) |> CM.nth(n) |> CM.nth(3) |> CM.reshape([32, 32])
  end

  # get RGB 3ch data
  def train_rgb1(<<>>) do
    []
  end

  def train_rgb1(<<_, rest::binary>>) do
    {image, other} = train_rgb2(rest, 3, [])
    [image | train_rgb1(other)]
  end

  # get one RGB data
  def train_rgb2(x, 0, res) do
    {Enum.reverse(res), x}
  end

  def train_rgb2(x, n, res) do
    {image, rest} = train_rgb3(x, 1024, [])
    train_rgb2(rest, n - 1, [image | res])
  end

  # get one image vector data
  def train_rgb3(x, 0, res) do
    {Enum.reverse(res), x}
  end

  def train_rgb3(<<x, xs::binary>>, n, res) do
    train_rgb3(xs, n - 1, [x | res])
  end

  def composit([r, g, b]) do
    composit1(r, g, b)
  end

  def composit1([], [], []) do
    []
  end

  def composit1([r | rs], [g | gs], [b | bs]) do
    [r * 256 * 256 + g * 256 + b | composit1(rs, gs, bs)]
  end
end

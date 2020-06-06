defmodule CIFAR do
  import Network
  alias Deeppipe, as: DP
  alias Cumatrix, as: CM

  @moduledoc """
  test with CIFAR10 dataset
  """

  # for CNN test
  # CIFAR.adagrad(100,20) 20epochs mini batch size 100 for batch_data1

  defnetwork init_network1(_x) do
    _x
    |> f(3, 3, 3, 32, {1, 1}, 1, {:he, 1024}, 0.01)
    |> relu
    |> f(3, 3, 32, 32, {1, 1}, 1, {:he, 32768}, 0.01)
    |> pooling(2, 2)
    |> f(3, 3, 32, 64, {1, 1}, 1, {:he, 32768}, 0.01)
    |> relu
    |> f(3, 3, 64, 64, {1, 1}, 1, {:he, 65536}, 0.01)
    |> relu
    |> pooling(2, 2)
    |> f(3, 3, 64, 64, {1, 1}, 1, {:he, 32768}, 0.01)
    |> f(3, 3, 64, 64, {1, 1}, 1, {:he, 32768}, 0.01)
    |> full
    |> w(4096, 100, {:he, 4098}, 0.01)
    |> w(100, 10, {:he, 100}, 0.01)
    |> softmax
  end

  def adagrad(m, n) do
    image = train_image_batch1()
    onehot = train_label_onehot1()
    network = init_network1(0)
    test_image = test_image(1000)
    test_label = test_label(1000)
    DP.train(network, image, onehot, test_image, test_label, :cross, :adagrad, m, n)
  end

  def readagrad(m, n) do
    image = train_image_batch1()
    onehot = train_label_onehot1()
    test_image = test_image(1000)
    test_label = test_label(1000)
    DP.retrain("temp.ex", image, onehot, test_image, test_label, :cross, :adagrad, m, n)
  end

  # transfer from train-label to onehot list
  def train_label_onehot1() do
    train_label_batch1() |> Enum.map(fn y -> DP.to_onehot(y, 9) end)
  end

  def train_label_onehot2() do
    train_label_batch2() |> Enum.map(fn y -> DP.to_onehot(y, 9) end)
  end

  def train_label_onehot3() do
    train_label_batch3() |> Enum.map(fn y -> DP.to_onehot(y, 9) end)
  end

  def train_label_onehot4() do
    train_label_batch4() |> Enum.map(fn y -> DP.to_onehot(y, 9) end)
  end

  def train_label_onehot5() do
    train_label_batch5() |> Enum.map(fn y -> DP.to_onehot(y, 9) end)
  end

  def train_label_batch1() do
    {:ok, <<label, rest::binary>>} = File.read("cifar-10-batches-bin/data_batch_1.bin")
    [label | train_label1(rest)]
  end

  def train_label_batch2() do
    {:ok, <<label, rest::binary>>} = File.read("cifar-10-batches-bin/data_batch_2.bin")
    [label | train_label1(rest)]
  end

  def train_label_batch3() do
    {:ok, <<label, rest::binary>>} = File.read("cifar-10-batches-bin/data_batch_3.bin")
    [label | train_label1(rest)]
  end

  def train_label_batch4() do
    {:ok, <<label, rest::binary>>} = File.read("cifar-10-batches-bin/data_batch_4.bin")
    [label | train_label1(rest)]
  end

  def train_label_batch5() do
    {:ok, <<label, rest::binary>>} = File.read("cifar-10-batches-bin/data_batch_5.bin")
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
    {:ok, <<label, rest::binary>>} = File.read("cifar-10-batches-bin/test_batch.bin")
    [label | train_label1(rest)]
  end

  # skip data
  def train_label2(<<rest::binary>>, 0) do
    rest
  end

  def train_label2(<<_, rest::binary>>, n) do
    train_label2(rest, n - 1)
  end

  def train_image() do
    train_image_batch1()
  end

  def train_image_batch1() do
    {:ok, bin} = File.read("cifar-10-batches-bin/data_batch_1.bin")
    train_image1(bin)
  end

  def train_image_batch2() do
    {:ok, bin} = File.read("cifar-10-batches-bin/data_batch_2.bin")
    train_image1(bin)
  end

  def train_image_batch3() do
    {:ok, bin} = File.read("cifar-10-batches-bin/data_batch_3.bin")
    train_image1(bin)
  end

  def train_image_batch4() do
    {:ok, bin} = File.read("cifar-10-batches-bin/data_batch_4.bin")
    train_image1(bin)
  end

  def train_image_batch5() do
    {:ok, bin} = File.read("cifar-10-batches-bin/data_batch_5.bin")
    train_image1(bin)
  end

  def test_image(n) do
    train_image() |> Enum.take(n)
  end

  def test_image() do
    {:ok, bin} = File.read("cifar-10-batches-bin/test_batch.bin")
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
    {:ok, bin} = File.read("cifar-10-batches-bin/data_batch_1.bin")
    train_rgb1(bin) |> CM.nth(n) |> composit() |> CM.reshape([32, 32])
  end

  def train_r(n) do
    {:ok, bin} = File.read("cifar-10-batches-bin/data_batch_1.bin")
    train_rgb1(bin) |> CM.nth(n) |> CM.nth(1) |> CM.reshape([32, 32])
  end

  def train_g(n) do
    {:ok, bin} = File.read("cifar-10-batches-bin/data_batch_1.bin")
    train_rgb1(bin) |> CM.nth(n) |> CM.nth(2) |> CM.reshape([32, 32])
  end

  def train_b(n) do
    {:ok, bin} = File.read("cifar-10-batches-bin/data_batch_1.bin")
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

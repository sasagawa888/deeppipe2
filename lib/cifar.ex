defmodule CIFAR do
  import Network
  alias Deeppipe, as: DP
  alias Cumatrix, as: CM

  @moduledoc """
  test with CIFAR10 dataset
  """

  # for CNN test
  # CIFAR.adagrad(100,20) 20epochs mini batch size 100 for all batch_data

  defnetwork init_network1(_x) do
    _x
    |> f(3, 3, 3, 8, {1, 1}, 1, 0.1, 0.0001)
    |> relu
    |> f(3, 3, 8, 8, {1, 1}, 1, 0.1, 0.0001)
    |> pooling(2, 2)
    |> f(3, 3, 8, 16, {1, 1}, 1, 0.1, 0.0001)
    |> relu
    |> f(3, 3, 16, 16, {1, 1}, 1, 0.1, 0.0001)
    |> relu
    |> pooling(2, 2)
    |> f(3, 3, 16, 32, {1, 1}, 1, 0.1, 0.0001)
    |> f(3, 3, 32, 32, {1, 1}, 1, 0.1, 0.0001)
    |> f(3, 3, 32, 32, {1, 1}, 1, 0.1, 0.0001)
    |> full
    |> w(2048, 1000, 0.1, 0.0001)
    |> w(1000, 100, 0.1, 0.0001)
    |> w(100, 10, 0.1, 0.0001)
    |> b(10, 0.1, 0.0001)
    |> softmax
  end 



  # adagrad/2 train network and save network temp.ex
  def adagrad(m, epoch) do
    test_image = test_image(1000)
    test_label = test_label(1000)
    network = init_network1(0)
    n = div(10000,m)
  
    {time, network1} =
      :timer.tc(fn ->
        adagrad1(network, m, n, epoch)
      end)

    correct = DP.accuracy(test_image, network1, test_label,m)
    IO.puts("learning end")
    IO.write("accuracy rate = ")
    IO.puts(correct)

    IO.inspect("time: #{time / 1_000_000} second")
    IO.inspect("-------------")
  end

  def adagrad1(network,_,_,0) do network end 
  def adagrad1(network,m,n,epoch) do
    IO.write("epocs--- ")
    IO.puts(epoch)
    network1 = DP.batch_train(network, train_image_batch1(), train_label_onehot1(), :cross, :adagrad, m, n)
    network2 = DP.batch_train(network1, train_image_batch2(), train_label_onehot2(), :cross, :adagrad, m, n)
    network3 = DP.batch_train(network2, train_image_batch3(), train_label_onehot3(), :cross, :adagrad, m, n)
    network4 = DP.batch_train(network3, train_image_batch4(), train_label_onehot4(), :cross, :adagrad, m, n)
    network5 = DP.batch_train(network4, train_image_batch5(), train_label_onehot5(), :cross, :adagrad, m, n)
    adagrad1(network5,m,n,epoch-1)
  end 

  # adagrad/2 load network from temp.ex and restart training
  def readagrad(m, epoch) do
    test_image = test_image(1000)
    test_label = test_label(1000)
    network = DP.load("temp.ex")
    n = div(10000,m)
  
    {time, network1} =
      :timer.tc(fn ->
        adagrad1(network, m, n, epoch)
      end)

    correct = DP.accuracy(test_image, network1, test_label,m)
    IO.puts("learning end")
    IO.write("accuracy rate = ")
    IO.puts(correct)

    IO.inspect("time: #{time / 1_000_000} second")
    IO.inspect("-------------")
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

defmodule CIFAR do
  import Network
  alias Deeppipe, as: DP

  @moduledoc """
  test with CIFAR10 dataset
  """

  # for CNN for CIFAR10
  # e.g. CIFAR.adam(300,20)    about 1.5 hours by GTX960
  
  defnetwork init_network1(_x) do
    _x
    |> f(3, 3, 3, 32, {1, 1}, 1, {:he, 1024}, 0.001)
    |> relu
    |> f(3, 3, 32, 32, {1, 1}, 1, {:he, 32768}, 0.001)
    |> pooling(2, 2)
    |> f(3, 3, 32, 64, {1, 1}, 1, {:he, 32768}, 0.001)
    |> relu
    |> f(3, 3, 64, 64, {1, 1}, 1, {:he, 65536}, 0.001)
    |> relu
    |> pooling(2, 2)
    |> f(3, 3, 64, 64, {1, 1}, 1, {:he, 32768}, 0.001)
    |> f(3, 3, 64, 64, {1, 1}, 1, {:he, 32768}, 0.001)
    |> full
    |> w(4096, 100, {:he, 4098}, 0.001, 0.25)
    |> w(100, 10, {:he, 100}, 0.001, 0.25)
    |> softmax
  end

  @doc """
  train with CIFAR10 dataset. optimizer is Adam 
  CIFAR.adam(300,20) about 1.5 hours by GTX960
  """
  def adam(m, n) do
    image = train_image_batch1()
    onehot = train_label_onehot1()
    network = init_network1(0)
    test_image = test_image(1000)
    test_label = test_label(1000)
    DP.train(network, image, onehot, test_image, test_label, :cross, :adam, m, n)
  end

  @doc """
  train agrain with CIFAR10data set.
  initialize network from file temp.ex
  """
  def readam(m, n) do
    image = train_image_batch1()
    onehot = train_label_onehot1()
    test_image = test_image(1000)
    test_label = test_label(1000)
    DP.retrain("temp.ex", image, onehot, test_image, test_label, :cross, :adam, m, n)
  end

  @doc """
  get n-size train image data from batch1 file
  """
  def train_image(n) do
    train_image_batch1() |> Enum.take(n)
  end 

  @doc """
  get n-size train label data from batch1 file
  """
  def train_label(n) do
    train_label_batch1() |> Enum.take(n)
  end 

  @doc """
  get n-size train label data as onehot from batch1 file
  """
  def train_label_onehot1() do
    train_label_batch1() |> Enum.map(fn y -> DP.to_onehot(y, 9) end)
  end

  @doc """
  get n-size train label data as onehot from batch2 file
  """
  def train_label_onehot2() do
    train_label_batch2() |> Enum.map(fn y -> DP.to_onehot(y, 9) end)
  end

  @doc """
  get n-size train label data as onehot from batch3 file
  """
  def train_label_onehot3() do
    train_label_batch3() |> Enum.map(fn y -> DP.to_onehot(y, 9) end)
  end

  @doc """
  get n-size train label data as onehot from batch4 file
  """
  def train_label_onehot4() do
    train_label_batch4() |> Enum.map(fn y -> DP.to_onehot(y, 9) end)
  end

  @doc """
  get n-size train label data as onehot from batch5 file
  """
  def train_label_onehot5() do
    train_label_batch5() |> Enum.map(fn y -> DP.to_onehot(y, 9) end)
  end

  defp train_label_batch1() do
    {:ok, <<label, rest::binary>>} = File.read("cifar-10-batches-bin/data_batch_1.bin")
    [label | train_label1(rest)]
  end

  defp train_label_batch2() do
    {:ok, <<label, rest::binary>>} = File.read("cifar-10-batches-bin/data_batch_2.bin")
    [label | train_label1(rest)]
  end

  defp train_label_batch3() do
    {:ok, <<label, rest::binary>>} = File.read("cifar-10-batches-bin/data_batch_3.bin")
    [label | train_label1(rest)]
  end

  defp train_label_batch4() do
    {:ok, <<label, rest::binary>>} = File.read("cifar-10-batches-bin/data_batch_4.bin")
    [label | train_label1(rest)]
  end

  defp train_label_batch5() do
    {:ok, <<label, rest::binary>>} = File.read("cifar-10-batches-bin/data_batch_5.bin")
    [label | train_label1(rest)]
  end

  # 36*36*3 = 3072
  defp train_label1(<<>>) do
    []
  end

  defp train_label1(x) do
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
  defp train_label2(<<rest::binary>>, 0) do
    rest
  end

  defp train_label2(<<_, rest::binary>>, n) do
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
    test_image() |> Enum.take(n)
  end

  def test_image() do
    {:ok, bin} = File.read("cifar-10-batches-bin/test_batch.bin")
    train_image1(bin)
  end

  # get RGB 3ch data
  defp train_image1(<<>>) do
    []
  end

  defp train_image1(<<_, rest::binary>>) do
    {image, other} = train_image2(rest, 3, [])
    [image | train_image1(other)]
  end

  # get one RGB data
  defp train_image2(x, 0, res) do
    {Enum.reverse(res), x}
  end

  defp train_image2(x, n, res) do
    {image, rest} = train_image3(x, 32, [])
    train_image2(rest, n - 1, [image | res])
  end

  # get one image 2D data
  defp train_image3(x, 0, res) do
    {Enum.reverse(res), x}
  end

  defp train_image3(x, n, res) do
    {image, rest} = train_image4(x, 32, [])
    train_image3(rest, n - 1, [image | res])
  end

  # get one row vector
  defp train_image4(x, 0, res) do
    # {Enum.reverse(res) , x}
    {Enum.reverse(res) |> DP.normalize(-128, 128), x}
  end

  defp train_image4(<<x, xs::binary>>, n, res) do
    train_image4(xs, n - 1, [x | res])
  end

  
end

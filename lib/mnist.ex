defmodule MNIST do
  # structure from flat vector to matrix(r,c)
  def structure([x], r, c) do
    structure1(x, r, c)
  end

  def structure0(x, r, c) do
    structure1(x, r, c)
  end

  def structure1(_, 0, _) do
    []
  end

  def structure1(x, r, c) do
    [Enum.take(x, c) | structure1(Enum.drop(x, c), r - 1, c)]
  end

  # get n datas from train-label
  def train_label(n) do
    Enum.take(train_label(), n)
  end

  # transfer from train-label to onehot list
  def train_label_onehot(n) do
    Enum.take(train_label(), n) |> Enum.map(fn y -> to_onehot0(y) end)
  end

  # get n datas from train-image with normalization
  def train_image(n) do
    train_image()
    |> Enum.take(n)
    |> Enum.map(fn x -> structure(MNIST.normalize(x, 255), 28, 28) end)
  end

  # get n datas from train-image as flatten list
  def train_image(n, :flatten) do
    train_image()
    |> Enum.take(n)
    |> Enum.map(fn x -> hd(MNIST.normalize(x, 255)) end)
  end

  # get n datas from test-label 
  def test_label(n) do
    Enum.take(test_label(), n)
  end

  # transfer from test-label to onehot list
  def test_label_onehot(n) do
    Enum.take(test_label(), n) |> Enum.map(fn y -> to_onehot0(y) end)
  end

  # get n datas from test-image with normalization
  def test_image(n) do
    test_image()
    |> Enum.take(n)
    |> Enum.map(fn x -> structure(MNIST.normalize(x, 255), 28, 28) end)
  end

  # get n datas from train-image as flatten list
  def test_image(n, :flatten) do
    test_image()
    |> Enum.take(n)
    |> Enum.map(fn x -> hd(MNIST.normalize(x, 255)) end)
  end

  def train_label() do
    {:ok, <<0, 0, 8, 1, 0, 0, 234, 96, label::binary>>} = File.read("train-labels-idx1-ubyte")
    label |> String.to_charlist()
  end

  def train_image() do
    {:ok, <<0, 0, 8, 3, 0, 0, 234, 96, 0, 0, 0, 28, 0, 0, 0, 28, image::binary>>} =
      File.read("train-images-idx3-ubyte")

    byte_to_list(image)
  end

  def test_label() do
    {:ok, <<0, 0, 8, 1, 0, 0, 39, 16, label::binary>>} = File.read("t10k-labels-idx1-ubyte")
    label |> String.to_charlist()
  end

  def test_image() do
    {:ok, <<0, 0, 8, 3, 0, 0, 39, 16, 0, 0, 0, 28, 0, 0, 0, 28, image::binary>>} =
      File.read("t10k-images-idx3-ubyte")

    byte_to_list(image)
  end

  def byte_to_list(bin) do
    byte_to_list1(bin, 784, [], [])
  end

  def byte_to_list1(<<>>, _, ls, res) do
    [Enum.reverse(ls) | res] |> Enum.reverse()
  end

  def byte_to_list1(bin, 0, ls, res) do
    byte_to_list1(bin, 784, [], [Enum.reverse(ls) | res])
  end

  def byte_to_list1(<<b, bs::binary>>, n, ls, res) do
    byte_to_list1(bs, n - 1, [b | ls], res)
  end

  def normalize(x, y) do
    [Enum.map(x, fn z -> z / y end)]
  end

  # e.g. 1 => [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  def to_onehot0(x) do
    to_onehot1(x, 9, [])
  end

  def to_onehot(x) do
    [to_onehot1(x, 9, [])]
  end

  def to_onehot1(_, -1, res) do
    res
  end

  def to_onehot1(x, x, res) do
    to_onehot1(x, x - 1, [1.0 | res])
  end

  def to_onehot1(x, c, res) do
    to_onehot1(x, c - 1, [0.0 | res])
  end

  def onehot_to_num([x]) do
    onehot_to_num1(x, 0)
  end

  def onehot_to_num(x) do
    onehot_to_num1(x, 0)
  end

  def onehot_to_num1([x | xs], n) do
    if x == Enum.max([x | xs]) do
      n
    else
      onehot_to_num1(xs, n + 1)
    end
  end
end

defmodule Fashon do
  alias Deeppipe, as: DP

  # structure from flat vector to matrix(r,c) as 1 channel 
  def structure(x, r, c) do
    [structure1(x, r, c)]
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
    Enum.take(train_label(), n) |> Enum.map(fn y -> DP.to_onehot(y, 9) end)
  end

  # get n datas from train-image with normalization
  def train_image(n, :structure) do
    train_image()
    |> Enum.take(n)
    |> Enum.map(fn x -> structure(DP.normalize(x, 0, 255), 28, 28) end)
  end

  # get n datas from train-image as flatten list
  def train_image(n, :flatten) do
    train_image()
    |> Enum.take(n)
    |> Enum.map(fn x -> DP.normalize(x, 0, 255) end)
  end

  # get n datas from test-label 
  def test_label(n) do
    Enum.take(test_label(), n)
  end

  # transfer from test-label to onehot list
  def test_label_onehot(n) do
    Enum.take(test_label(), n) |> Enum.map(fn y -> DP.to_onehot(y, 9) end)
  end

  # get n datas from test-image with normalization as structured list
  def test_image(n) do
    test_image()
    |> Enum.take(n)
    |> Enum.map(fn x -> structure(DP.normalize(x, 0, 255), 28, 28) end)
  end

  def test_image(n, :structure) do
    test_image()
    |> Enum.take(n)
    |> Enum.map(fn x -> structure(DP.normalize(x, 0, 255), 28, 28) end)
  end

  # get n datas from train-image as flatten list
  def test_image(n, :flatten) do
    test_image()
    |> Enum.take(n)
    |> Enum.map(fn x -> DP.normalize(x, 0, 255) end)
  end

  def train_label() do
    {:ok, <<0, 0, 8, 1, 0, 0, 234, 96, label::binary>>} =
      File.read("fashion/train-labels-idx1-ubyte")

    label |> String.to_charlist()
  end

  def train_image() do
    {:ok, <<0, 0, 8, 3, 0, 0, 234, 96, 0, 0, 0, 28, 0, 0, 0, 28, image::binary>>} =
      File.read("fashion/train-images-idx3-ubyte")

    byte_to_list(image)
  end

  def test_label() do
    {:ok, <<0, 0, 8, 1, 0, 0, 39, 16, label::binary>>} =
      File.read("fashion/t10k-labels-idx1-ubyte")

    label |> String.to_charlist()
  end

  def test_image() do
    {:ok, <<0, 0, 8, 3, 0, 0, 39, 16, 0, 0, 0, 28, 0, 0, 0, 28, image::binary>>} =
      File.read("fashion/t10k-images-idx3-ubyte")

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
end

defmodule NLP do
  import Network
  alias Deeppipe, as: DP

  @moduledoc """
  for natural language
  """

  defnetwork init1(_x) do
    _x
    |> lstm(29, 14)
    |> w(29, 2)
    |> softmax
  end

  def sgd(m, n) do
    image = train_image()
    onehot = train_label_onehot()
    network = init1(0)
    test_image = train_image()
    test_label = train_label()
    DP.train(network, image, onehot, test_image, test_label, :cross, :sgd, m, n)
  end

  def train_image() do
    {:ok, dt} = File.read("rnn/train.exs")
    dt |> String.replace("\n", "") |> preprocess()
  end

  def train_label() do
    {:ok, dt} = File.read("rnn/train-label.exs")

    dt
    |> String.replace("\n", " ")
    |> String.split(" ")
    |> butlast()
    |> Enum.map(fn x -> String.to_integer(x) end)
  end

  def train_label_onehot() do
    ls = train_label()
    dim = Enum.max(ls)
    ls |> Enum.map(fn x -> DP.to_onehot(x, dim) end)
  end

  def test_image() do
    {:ok, dt} = File.read("rnn/test.exs")
    dt |> String.replace("\n", "") |> preprocess()
  end

  def test_label() do
    {:ok, dt} = File.read("rnn/test-label.exs")

    dt
    |> String.replace("\n", " ")
    |> String.split(" ")
    |> butlast()
    |> Enum.map(fn x -> String.to_integer(x) end)
  end

  @doc """
  transform sentences to matrix. Each element is onehot_vector.
  iex(1)> NLP.preprocess("I love you.you love me?")
  [
  [
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
  ],
  [
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
  ]
  ]
  """
  def preprocess(text) do
    {text1, dic, _} = preprocess1(text)
    maxlen = text1 |> Enum.map(fn x -> length(x) end) |> Enum.max()
    count = length(dic)

    text1
    |> Enum.map(fn x -> addzero(x, maxlen - length(x)) end)
    |> Enum.map(fn x -> Enum.map(x, fn y -> DP.to_onehot(y, count) end) end)
  end

  defp addzero1(0) do
    []
  end

  defp addzero1(n) do
    [0 | addzero1(n - 1)]
  end

  defp addzero(ls, n) do
    ls ++ addzero1(n)
  end

  @doc """
  generate corpus,dic of word->ID and dic of ID->word from sentences as text.
  iex(2)> NLP.preprocess1("I love you.you love me?")
  {[[1, 2, 3, 4], [3, 2, 5, 6]], [I: 1, love: 2, you: 3, ".": 4, me: 5, "?": 6],
  [{1, :I}, {2, :love}, {3, :you}, {4, :.}, {5, :me}, {6, :"?"}]}

  """
  def preprocess1(text) do
    dic =
      text
      |> String.replace(".", " . ")
      |> String.replace("?", " ? ")
      |> String.split(" ")
      |> butlast()
      |> Enum.map(fn x -> String.to_atom(x) end)
      |> word_to_id()

    text1 =
      text
      |> String.replace(".", ".EOS")
      |> String.replace("?", "?EOS")
      |> String.split("EOS")
      |> butlast()
      |> Enum.map(fn x -> preprocess2(x, dic) end)

    {text1, dic, id_to_word(dic)}
  end

  def preprocess2(text, dic) do
    text1 =
      text
      |> String.replace(".", " .")
      |> String.replace("?", " ?")
      |> String.split(" ")
      |> Enum.map(fn x -> String.to_atom(x) end)

    corpus(text1, dic)
  end

  def butlast(ls) do
    ls
    |> Enum.reverse()
    |> Enum.drop(1)
    |> Enum.reverse()
  end

  def corpus([], _) do
    []
  end

  def corpus([l | ls], dic) do
    [dic[l] | corpus(ls, dic)]
  end

  def word_to_id(ls) do
    word_to_id1(ls, 1, [])
  end

  def word_to_id1([], _, dic) do
    Enum.reverse(dic)
  end

  def word_to_id1([l | ls], n, dic) do
    if dic[l] != nil do
      word_to_id1(ls, n, dic)
    else
      word_to_id1(ls, n + 1, Keyword.put(dic, l, n))
    end
  end

  def id_to_word([]) do
    []
  end

  def id_to_word([{word, id} | ls]) do
    [{id, word} | id_to_word(ls)]
  end
end

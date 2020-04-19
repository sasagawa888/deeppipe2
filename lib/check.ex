defmodule Check do
  import Network
  alias Deeppipe, as: DP
  alias Cumatrix, as: CM

  # for grad confarmation
  defnetwork test_network0(_x) do
    _x
    |> cf([[[0.1, 0.1], [0.1, 0.1]], [[0.1, 0.1], [0.1, 0.1]]])
    |> cf([[[0.2, 0.2], [0.2, 0.2]]])
    |> full
    |> softmax
  end

  def fd() do
    data =
      [
        [
          [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 0.10, 0.11, 0.12],
            [0.13, 0.14, 0.15, 0.16]
          ],
          [
            [0.17, 0.18, 0.19, 0.20],
            [0.21, 0.22, 0.23, 0.24],
            [0.25, 0.26, 0.27, 0.28],
            [0.29, 0.30, 0.31, 0.32]
          ]
        ]
      ]
      |> CM.new()

    [y|_] = DP.forward(data, test_network0(0),[])
    y |> CM.to_list()
  end

  def grad() do
    data =
      [
        [
          [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 0.10, 0.11, 0.12],
            [0.13, 0.14, 0.15, 0.16]
          ],
          [
            [0.17, 0.18, 0.19, 0.20],
            [0.21, 0.22, 0.23, 0.24],
            [0.25, 0.26, 0.27, 0.28],
            [0.29, 0.30, 0.31, 0.32]
          ]
        ]
      ]
      |> CM.new()

    train = [[0.0,1.0,0.0,0.0]] |> CM.new()
    network1 = DP.numerical_gradient(data, test_network0(0), train)
    network1 |> DP.print()
  end

  def back() do
    data =
      [
        [
          [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 0.10, 0.11, 0.12],
            [0.13, 0.14, 0.15, 0.16]
          ],
          [
            [0.17, 0.18, 0.19, 0.20],
            [0.21, 0.22, 0.23, 0.24],
            [0.25, 0.26, 0.27, 0.28],
            [0.29, 0.30, 0.31, 0.32]
          ]
        ]
      ]
      |> CM.new()

    train = [[0.0,1.0,0.0,0.0]] |> CM.new()
    network1 = DP.gradient(data, test_network0(0), train)
    network1 |> DP.print()
  end
end 
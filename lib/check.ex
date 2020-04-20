defmodule Check do
  import Network
  alias Deeppipe, as: DP
  alias Cumatrix, as: CM

  # for grad confirmation
  defnetwork test_network0(_x) do
    _x
    |> cf([[[0.1, 0.1], [0.1, 0.1]], [[0.1, 0.1], [0.1, 0.1]]])
    |> cf([[[0.2, 0.2], [0.2, 0.2]]])
    |> full
    |> softmax
  end

  def test() do
    data = CM.rand(1,2,4,4)
    train = [[0.0,1.0,0.0,0.0]] |> CM.new()
    network1 = DP.numerical_gradient(data, test_network0(0), train)
    network2 = DP.gradient(data, test_network0(0), train)
    test1(network1,network2,1)
  end 

  def test1([],[],_) do true end 
  def test1([x | xs],[y | ys],n) do
    cond do
      (CM.is_matrix(x) && CM.is_matrix(y)) || (CM.is_matrix(x) && CM.is_tensor(y)) -> 
        if CM.is_near(x,y) do 
          test1(xs,ys,n+1)
        else
          IO.puts(n)
          x |> CM.to_list() |> IO.inspect()
          y |> CM.to_list() |> IO.inspect()
          test1(xs,ys,n+1)
      end 
      true -> test1(xs,ys,n+1)
    end
  end  


  
end 
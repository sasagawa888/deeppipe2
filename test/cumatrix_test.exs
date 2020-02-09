defmodule CumatrixTest do
  use ExUnit.Case
  doctest Cumatrix

  test "matrix" do
    assert Cumatrix.new([[1.0, 2.0], [3.0, 4.0]]) == {2, 2, <<0, 0, 128, 63, 0, 0, 64, 64, 0, 0, 0, 64, 0, 0, 128, 64>>}
    m = Cumatrix.new([[1.0, 2.0], [3.0, 4.0]])
    assert Cumatrix.add(m,m) == Cumatrix.new([[2.0,4.0],[6.0,8.0]])
    assert Cumatrix.sub(m,m) == Cumatrix.new([[0.0,0.0],[0.0,0.0]])
    assert Cumatrix.mult(m,m) == Cumatrix.new([[7.0,10.0],[15.0,22.0]])
    assert Cumatrix.new(2,3) == Cumatrix.new([[0.0,0.0,0.0],[0.0,0.0,0.0]]) 
    assert Cumatrix.new(2,3,0.1) == Cumatrix.new([[0.1,0.1,0.1],[0.1,0.1,0.1]]) 
    assert Cumatrix.elt(m, 1, 2) == 2.0
    assert Cumatrix.size(m) == {2, 2}
    m1 = Cumatrix.new([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]])
    assert Cumatrix.emult(m1,m1) == Cumatrix.new([[1.0,4.0,9.0],[16.0,25.0,36.0]]) 
    assert Cumatrix.sum(m) == 10.0
    assert Cumatrix.trace(m) == 5.0
    assert Cumatrix.ident(3) == Cumatrix.new([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
    assert Cumatrix.transpose(m1) == Cumatrix.new([[1.0,4.0], [-2.0,5.0],[3.0,-6.0]])
    m2 = Cumatrix.new([[1010.0,1000.0,990.0],[1010.0,1000.0,990.0]])
    assert Cumatrix.activate(m2, :softmax) ==
      Cumatrix.new([[0.9999545812606812,4.539786823443137e-5,2.0610599893444714e-9],[0.9999545812606812,4.539786823443137e-5,2.0610599893444714e-9]])
    assert Cumatrix.mult(2.0,m1) == Cumatrix.new([[2.0, -4.0, 6.0], [8.0, 10.0, -12.0]])
    assert Cumatrix.mult(m1,2.0) == Cumatrix.new([[2.0, -4.0, 6.0], [8.0, 10.0, -12.0]])
    m3 = Cumatrix.new([[1.0,2.0]])
    assert Cumatrix.add(m,m3) == Cumatrix.new([[2.0, 4.0], [4.0, 6.0]])
    assert Cumatrix.to_list(m1) == [[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]]
  end

  
end

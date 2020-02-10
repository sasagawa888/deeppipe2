defmodule CumatrixTest do
  use ExUnit.Case
  doctest Cumatrix

  test "matrix" do
    assert Cumatrix.new([[1.0, 2.0], [3.0, 4.0]]) ==
             {2, 2, <<0, 0, 128, 63, 0, 0, 64, 64, 0, 0, 0, 64, 0, 0, 128, 64>>}

    m = Cumatrix.new([[1.0, 2.0], [3.0, 4.0]])
    assert Cumatrix.add(m, m) == Cumatrix.new([[2.0, 4.0], [6.0, 8.0]])
    assert Cumatrix.sub(m, m) == Cumatrix.new([[0.0, 0.0], [0.0, 0.0]])
    assert Cumatrix.mult(m, m) == Cumatrix.new([[7.0, 10.0], [15.0, 22.0]])
    assert Cumatrix.new(2, 3) == Cumatrix.new([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    assert Cumatrix.new(2, 3, 0.1) == Cumatrix.new([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])
    assert Cumatrix.elt(m, 1, 2) == 2.0
    assert Cumatrix.size(m) == {2, 2}
    m1 = Cumatrix.new([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]])
    assert Cumatrix.emult(m1, m1) == Cumatrix.new([[1.0, 4.0, 9.0], [16.0, 25.0, 36.0]])
    assert Cumatrix.sum(m) == 10.0
    assert Cumatrix.trace(m) == 5.0
    assert Cumatrix.ident(3) == Cumatrix.new([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert Cumatrix.transpose(m1) == Cumatrix.new([[1.0, 4.0], [-2.0, 5.0], [3.0, -6.0]])
    m2 = Cumatrix.new([[1010.0, 1000.0, 990.0], [1010.0, 1000.0, 990.0]])

    assert Cumatrix.activate(m2, :softmax) ==
             Cumatrix.new([
               [0.9999545812606812, 4.539786823443137e-5, 2.0610599893444714e-9],
               [0.9999545812606812, 4.539786823443137e-5, 2.0610599893444714e-9]
             ])

    assert Cumatrix.mult(2.0, m1) == Cumatrix.new([[2.0, -4.0, 6.0], [8.0, 10.0, -12.0]])
    assert Cumatrix.mult(m1, 2.0) == Cumatrix.new([[2.0, -4.0, 6.0], [8.0, 10.0, -12.0]])
    m3 = Cumatrix.new([[1.0, 2.0]])
    assert Cumatrix.add(m, m3) == Cumatrix.new([[2.0, 4.0], [4.0, 6.0]])
    assert Cumatrix.to_list(m1) == [[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]]
    assert Cumatrix.minus(m1, 2, 3, 1.0) == Cumatrix.new([[1.0, -2.0, 3.0], [4.0, 5.0, -7.0]])
    m4 = Cumatrix.new([[1.0, 2.0, 3.0], [2.0, 3.0, 1.2]])
    m5 = Cumatrix.new([[1.1, 2.9, 2.0], [2.2, 3.1, 1.2]])

    assert Cumatrix.loss(m4, m5, :square) ==
             Cumatrix.new([[0.9100000858306885], [0.02499999850988388]])

    assert Cumatrix.loss(m4, m5, :cross) ==
             Cumatrix.new([[4.2073516845703125], [5.149408340454102]])

    m6 = Cumatrix.new([[0.1, 0.9, 0.1], [0.2, 0.1, 1.2]])

    assert Cumatrix.activate(m6, :sigmoid) ==
             Cumatrix.new([
               [0.5249791741371155, 0.7109494805335999, 0.5249791741371155],
               [0.5498339533805847, 0.5249791741371155, 0.7685248255729675]
             ])

    assert Cumatrix.activate(m6, :tanh) ==
             Cumatrix.new([
               [0.0996680036187172, 0.7162978649139404, 0.0996680036187172],
               [0.1973753273487091, 0.0996680036187172, 0.8336546421051025]
             ])

    assert Cumatrix.activate(m6, :relu) ==
             Cumatrix.new([
               [0.10000000149011612, 0.8999999761581421, 0.10000000149011612],
               [0.20000000298023224, 0.10000000149011612, 1.2000000476837158]
             ])
  end
end
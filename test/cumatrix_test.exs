defmodule CumatrixTest do
  use ExUnit.Case
  import ExUnit.CaptureIO
  doctest Cumatrix

  test "matrix" do
    assert Cumatrix.new([[1, 2], [3, 4]]) == {2, 2, [1, 3, 2, 4]}
    m = Cumatrix.new([[1.0, 2.0], [3.0, 4.0]])
    assert Cumatrix.elt(m, 1, 2) == 2
    assert Cumatrix.size(m) == {2, 2}
    assert Cumatrix.sum(m) == 10.0
    assert Cumatrix.trace(m) == 5.0
  end

  test "total test" do
    m = Cumatrix.new([[1.0, 2.0], [3.0, 4.0]])
    m1 = Cumatrix.mult(m, m)
    m2 = Cumatrix.add(m, m)
    m3 = Cumatrix.sub(m, m)
    m4 = Cumatrix.new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    m5 = Cumatrix.transpose(m4)
    m6 = Cumatrix.ident(3)
    m7 = Cumatrix.new([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]])
    m8 = Cumatrix.emult(m7, m7)
    m9 = Cumatrix.new(([[1.0,2.0,3.0]]))
    m10 = Cumatrix.badd(m7,m9)
    m11 = Cumatrix.new([[1010.0,1000.0,990.0],[1010.0,1000.0,990.0]])
    assert capture_io(fn -> Cumatrix.print(m) end) == "| 1.0 2.0 |\n| 3.0 4.0 |\n"
    assert capture_io(fn -> Cumatrix.print(m1) end) == "| 7.0 10.0 |\n| 15.0 22.0 |\n"
    assert capture_io(fn -> Cumatrix.print(m2) end) == "| 2.0 4.0 |\n| 6.0 8.0 |\n"
    assert capture_io(fn -> Cumatrix.print(m3) end) == "| 0.0 0.0 |\n| 0.0 0.0 |\n"
    assert capture_io(fn -> Cumatrix.print(m5) end) == "| 1.0 4.0 |\n| 2.0 5.0 |\n| 3.0 6.0 |\n"

    assert capture_io(fn -> Cumatrix.print(m6) end) ==
             "| 1.0 0.0 0.0 |\n| 0.0 1.0 0.0 |\n| 0.0 0.0 1.0 |\n"

    assert capture_io(fn -> Cumatrix.activate(m4, :sigmoid) |> Cumatrix.print() end) ==
             "| 0.7310585975646973 0.8807970285415649 0.9525741338729858 |\n| 0.9820137619972229 0.9933071732521057 0.9975274205207825 |\n"

    assert capture_io(fn -> Cumatrix.activate(m4, :tanh) |> Cumatrix.print() end) ==
             "| 0.7615941762924194 0.9640275835990906 0.9950547814369202 |\n| 0.9993293285369873 0.9999092221260071 0.9999877214431763 |\n"

    assert capture_io(fn -> Cumatrix.activate(m7, :relu) |> Cumatrix.print() end) ==
             "| 1.0 0.0 3.0 |\n| 4.0 5.0 0.0 |\n"

    assert capture_io(fn -> Cumatrix.activate(m11, :softmax) |> Cumatrix.print() end) ==
             "| 0.9999545812606812 4.539786823443137e-5 2.0610599893444714e-9 |\n| 0.9999545812606812 4.539786823443137e-5 2.0610599893444714e-9 |\n"

    assert capture_io(fn -> Cumatrix.smult(2.0, m7) |> Cumatrix.print() end) ==
             "| 2.0 -4.0 6.0 |\n| 8.0 10.0 -12.0 |\n"

    t = Cumatrix.new([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    y = Cumatrix.new([[0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]])

    assert capture_io(fn -> Cumatrix.loss(y, t, :square) |> Cumatrix.print() end) ==
             "| 0.09749999098479775 |\n"

    assert capture_io(fn -> Cumatrix.loss(y, t, :cross) |> Cumatrix.print() end) ==
             "| 0.5108254173629155 |\n"

    assert capture_io(fn -> Cumatrix.print(m8) end) == "| 1.0 4.0 9.0 |\n| 16.0 25.0 36.0 |\n"

    assert capture_io(fn -> Cumatrix.print(m10) end) == "| 2.0 0.0 6.0 |\n| 5.0 7.0 -3.0 |\n"
  end
end

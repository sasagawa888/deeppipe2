defmodule Deeppipe2 do
  # common
  def is_matrix({r,c,dt}) do
    if is_integer(r) && is_integer(c) && is_list(dt) do
      true 
    else 
      false
    end 
  end

  # forward
  def forward(x, []) do
    x
  end

  def forward(x, [{:weight, w, _, _} | rest]) do
    Cumatrix.mult(x, w) |> forward(rest)
  end

  def forward(x, [{:bias, b, _, _} | rest]) do
    Cumatrix.badd(x, b) |> forward(rest)
  end

  def forward(x, [{:function, name} | rest]) do
    cond do
      name == :sigmoid -> Cumatrix.activate(x,:sigmoid) |> forward(rest)
      name == :tanh -> Cumatrix.activate(x,:tanh) |> forward(rest)
      name == :relu -> Cumatrix.activate(x,:relu) |> forward(rest)
      name == :softmax -> Cumatrix.activate(x,:softmax) |> forward(rest)
      true -> raise "not exist function" 
    end 
  end
end

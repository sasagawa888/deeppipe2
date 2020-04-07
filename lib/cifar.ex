defmodule CIFAR do
    alias Deeppipe, as: DP

    
    def train_label() do
      {:ok, <<label, rest::binary>>} = File.read("data_batch_1.bin")
      [label | train_label1(rest)]
    end
  
    # 36*36*3 = 3072
    def train_label1(<<>>) do [] end
    def train_label1(x) do
      result = train_label2(x,3072)
      if result != <<>> do 
        <<label, rest::binary>> = result
        [label | train_label1(rest)]
      else
        []
      end 
    end 

    def test_label() do
      {:ok, <<label, rest::binary>>} = File.read("test_batch.bin")
      [label | train_label1(rest)]
    end 

    # skip data
    def train_label2(<<rest::binary>>, 0) do
        rest
    end 
    def train_label2(<<_,rest::binary>>, n) do 
      train_label2(rest,n-1)
    end 

    def train_image() do
      {:ok, bin} = File.read("data_batch_1.bin")
      train_image1(bin)
    end

    def test_image() do
      {:ok, bin} = File.read("test_batch.bin")
      train_image1(bin)
    end

    # get RGB 3ch data
    def train_image1(<<>>) do [] end 
    def train_image1(<<_, rest::binary>>) do
      {image, other} = train_image2(rest,3,[])
      [image | train_image1(other)]
    end 

    # get one RGB data
    def train_image2(x, 0, res) do {Enum.reverse(res),x} end 
    def train_image2(x, n, res) do
      {image, rest} = train_image3(x,32,[])
      train_image2(rest, n-1, [image|res])
    end 

    # get one image 2D data
    def train_image3(x, 0, res) do {Enum.reverse(res),x} end 
    def train_image3(x, n, res) do
      {image, rest} = train_image4(x,32,[]) 
      train_image3(rest, n-1, [image|res])
    end 

    # get one row vector
    def train_image4(x, 0, res) do {Enum.reverse(res) |> DP.normalize(256) ,x} end 
    def train_image4(<<x,xs::binary>>, n, res) do 
      train_image4(xs,n-1,[x|res])
    end 


  
end

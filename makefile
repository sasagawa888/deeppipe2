nifs: ./lib/nifs.cu
	@mkdir -p priv
	nvcc -shared --compiler-options '-fPIC' -lcublas -o ./priv/nifs.so ./lib/nifs.cu

nifs: ./lib/nifs.cu
	nvcc -shared --compiler-options '-fPIC' -lcublas -o ./lib/nifs.so ./lib/nifs.cu

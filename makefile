nifs: ./lib/nifs.cu
	nvcc -shared --compiler-options '-fPIC' -lcublas -lcudnn -o ./lib/nifs.so ./lib/nifs.cu

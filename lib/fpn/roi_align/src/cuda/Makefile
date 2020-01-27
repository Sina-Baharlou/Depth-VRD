all: roi_align_kernel.cu roi_align_kernel.h
	/usr/local/cuda/bin/nvcc -c -o roi_align.cu.o roi_align_kernel.cu --compiler-options -fPIC -gencode arch=compute_37,code=sm_37
clean:
	rm roi_align.cu.o

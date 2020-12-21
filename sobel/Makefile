CC=gcc
CFLAGS=-O3 -march=native
LDLIBS=-lm -lIL

sobel: sobel.cu
	nvcc -o $@ $< $(LDLIBS)

sobel-shared : sobel_shared.cu
	  nvcc -o $@ $< $(LDLIBS)

sobelc : sobel.c
	gcc -o $@ $< $(LDLIBS)

sobel-stream : sobel-stream.cu
	nvcc -o $@ $< $(LDLIBS)
clean:
	rm sobel

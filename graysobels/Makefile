CC=nvcc
CFLAGS=-O3 -march=native
LDLIBS=`pkg-config --libs opencv`

gray_sobel_stream: gray_sobel_stream.cu
	$(CC) -o $@ $< $(LDLIBS)

gray_sobel: gray_sobel.cu
	$(CC) -o $@ $< $(LDLIBS)
clean:
	rm sobel

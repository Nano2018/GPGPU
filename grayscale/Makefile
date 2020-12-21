CXX=g++
CXXFLAGS=-O3 -march=native
LDLIBS=`pkg-config --libs opencv`


grayscale: grayscale.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

grayscale-cu: grayscale.cu
	nvcc -o $@ $< $(LDLIBS)

grayscale-stream-cu : grayscale-stream.cu
	nvcc -o $@ $< $(LDLIBS)

.PHONY: clean

clean:
	rm grayscale

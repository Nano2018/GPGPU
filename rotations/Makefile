CXX=nvcc
CXXFLAGS=-O3 -march=native
LDLIBS= `pkg-config --libs opencv`
rotation: rotation.cu
	$(CXX) -o $@ $< $(LDLIBS)

.PHONY: clean

clean:
	rm sharpen

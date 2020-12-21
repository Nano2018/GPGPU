CXX=nvcc
CXXFLAGS=-O3 -march=native
LDLIBS=`pkg-config --libs opencv`
LIBS = -lm -lIL
sharpen: sharpen.cu
	$(CXX) -o $@ $< $(LDLIBS)

sharpen_bis : sharpen_bis.cu
	$(CXX) -o $@ $< $(LIBS)

.PHONY: clean

clean:
	rm sharpen

NVCC   := nvcc
SM     ?= 75   # override: make SM=80 (A100), SM=70 (V100), SM=89 (L4)

OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4 2>/dev/null || pkg-config --cflags opencv 2>/dev/null)
OPENCV_LIBS   := $(shell pkg-config --libs   opencv4 2>/dev/null || pkg-config --libs   opencv 2>/dev/null)

NVCCFLAGS := -arch=sm_$(SM) -O2 --std=c++17 -Iinclude $(OPENCV_CFLAGS)

SRCS := main.cu \
        src/bvh.cu \
        src/forces.cu \
        src/graphic.cu \
        src/morton.cu \
        src/tree.cu

OBJS := $(patsubst %.cu, build/%.o, $(SRCS))

TARGET := cluster

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) -arch=sm_$(SM) $^ -o $@ $(OPENCV_LIBS)

build/%.o: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

clean:
	rm -rf build $(TARGET) cluster.mp4

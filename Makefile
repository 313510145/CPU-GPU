.PHONY: all check clean

CXX = nvcc
CXX_FLAG = -std=c++17 -g -G -Xcompiler -fopenmp
INCLUDE_PATH = -Iheader -Iheader_cuda

CPPCHECK = ../cppcheck-2.17.1/cppcheck
CPPCHECK_FLAG = --enable=all --check-level=exhaustive --inconclusive --suppress=missingIncludeSystem

SOURCE_DIRECTORY = source
OBJECT_DIRECTORY = object

SOURCE_CUDA_DIRECTORY = source_cuda
OBJECT_CUDA_DIRECTORY = object_cuda

SOURCE = $(wildcard $(SOURCE_DIRECTORY)/*.cpp)
OBJECT = $(patsubst $(SOURCE_DIRECTORY)/%.cpp, $(OBJECT_DIRECTORY)/%.o, $(SOURCE))
DEPENDENCY = $(patsubst $(SOURCE_DIRECTORY)/%.cpp, $(OBJECT_DIRECTORY)/%.d, $(SOURCE))

SOURCE_CUDA = $(wildcard $(SOURCE_CUDA_DIRECTORY)/*.cu)
OBJECT_CUDA = $(patsubst $(SOURCE_CUDA_DIRECTORY)/%.cu, $(OBJECT_CUDA_DIRECTORY)/%.o, $(SOURCE_CUDA))
DEPENDENCY_CUDA = $(patsubst $(SOURCE_CUDA_DIRECTORY)/%.cu, $(OBJECT_CUDA_DIRECTORY)/%.d, $(SOURCE_CUDA))

TARGET = library
MAIN = main.cu

all: $(TARGET)

$(OBJECT_DIRECTORY):
	mkdir $(OBJECT_DIRECTORY)

$(OBJECT_CUDA_DIRECTORY):
	mkdir $(OBJECT_CUDA_DIRECTORY)

$(TARGET): $(MAIN) $(OBJECT) $(OBJECT_CUDA)
	$(CXX) $(CXX_FLAG) $(INCLUDE_PATH) $^ -o $@

-include $(DEPENDENCY) $(DEPENDENCY_CUDA)

$(OBJECT_DIRECTORY)/%.o: $(SOURCE_DIRECTORY)/%.cpp | $(OBJECT_DIRECTORY)
	$(CXX) $(CXX_FLAG) $(INCLUDE_PATH) -MMD -c $< -o $@

$(OBJECT_CUDA_DIRECTORY)/%.o: $(SOURCE_CUDA_DIRECTORY)/%.cu | $(OBJECT_CUDA_DIRECTORY)
	$(CXX) $(CXX_FLAG) $(INCLUDE_PATH) -MMD -c $< -o $@

check: $(CPPCHECK)
	./$(CPPCHECK) $(CPPCHECK_FLAG) $(INCLUDE_PATH) $(MAIN) $(SOURCE) $(SOURCE_CUDA)

clean:
	rm -rf $(TARGET) $(OBJECT_DIRECTORY) $(OBJECT_CUDA_DIRECTORY)

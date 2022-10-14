CUDA_DIR=$(patsubst %bin/nvcc, %, $(shell which nvcc))
CUDA_LIB=${CUDA_DIR}/lib64
LOCAL_LIB=lib
LOCAL_INCLUDE=include
LOCAL_OBJ=obj
LOCAL_BIN=bin
GCC=gcc-10
CXX=g++-10 -g -D__CUDA__=1
MPICXX=mpic++
NVCC=nvcc -arch=compute_35 --compiler-bindir /usr/bin/g++-10 -g

CPP_EXE_SRC=$(wildcard util/*.cpp)
CPP_EXE_OBJ=$(patsubst util/%.cpp, ${LOCAL_OBJ}/%.o, ${CPP_EXE_SRC})
CPP_EXE_RUN=$(patsubst util/%.cpp, ${LOCAL_BIN}/%_run, ${CPP_EXE_SRC})

CUDA_EXE_SRC=$(wildcard util/*.cu)
CUDA_EXE_OBJ=$(patsubst util/%.cu, ${LOCAL_OBJ}/%_cu.o, ${CUDA_EXE_SRC})
CUDA_EXE_RUN=$(patsubst util/%.cu, ${LOCAL_BIN}/%_cu, ${CUDA_EXE_SRC})

COMMON_LIB_SRC=$(wildcard src/common/*.cc)
COMMON_LIB_OBJ=$(patsubst src/common/%.cc, ${LOCAL_OBJ}/%.o, ${COMMON_LIB_SRC})
COMMON_LIB=$(patsubst src/common/%.cc, ${LOCAL_LIB}/lib%.so, ${COMMON_LIB_SRC})

COMMON_C_LIB_SRC=$(wildcard src/common/*.c)
COMMON_C_LIB_OBJ=$(patsubst src/common/%.c, ${LOCAL_OBJ}/%.o, ${COMMON_C_LIB_SRC})
COMMON_C_LIB=$(patsubst src/common/%.c, ${LOCAL_LIB}/lib%.so, ${COMMON_C_LIB_SRC})

CPP_LIB_SRC=$(wildcard src/cpu/*.cc)
CPP_LIB_OBJ=$(patsubst src/cpu/%.cc, ${LOCAL_OBJ}/%.o, ${CPP_LIB_SRC})
CPP_LIB=$(patsubst src/cpu/%.cc, ${LOCAL_LIB}/lib%.so,  ${CPP_LIB_SRC})

CUDA_WRAP_LIB_SRC=$(wildcard src/gpu/*.cu)
CUDA_WRAP_LIB_OBJ=$(patsubst src/gpu/%.cu, ${LOCAL_OBJ}/%_cu.o, ${CUDA_WRAP_LIB_SRC})
CUDA_WRAP_LIB=$(patsubst src/gpu/%.cu, ${LOCAL_LIB}/lib%_cu.a,  ${CUDA_WRAP_LIB_SRC})

LINK_FLAGS_EXT=$(shell pkg-config --libs opencv4 hdf5 tbb fftw3) -lfftw3_mpi -lcholmod -lfftw3_threads -lm -lpthread -lconfig++
LINK_FLAGS_EXT_CU=$(shell pkg-config --libs opencv4 hdf5)
#LINK_FLAGS+=$(patsubst ${LOCAL_LIB}/lib%.so, -l%, ${CPP_LIB})
LINK_FLAGS_CUDA_WRAP=$(patsubst ${LOCAL_LIB}/lib%.a, -l%, ${CUDA_WRAP_LIB})
LINK_FLAGS= -L${LOCAL_LIB} $(patsubst ${LOCAL_LIB}/lib%.so, -l%, ${COMMON_LIB} ${COMMON_C_LIB})
LINK_FLAGS+=${LINK_FLAGS_CUDA_WRAP}
INCLUDE_FLAGS=-I${LOCAL_INCLUDE} $(shell pkg-config --cflags opencv4 hdf5 mpi) -I/usr/include/suitesparse

all: print ${COMMON_LIB_OBJ} ${CUDA_WRAP_LIB_OBJ} ${CPP_EXE_OBJ} ${COMMON_C_LIB_OBJ} ${CUDA_EXE_OBJ}  ${COMMON_C_LIB} ${COMMON_LIB} ${CUDA_WRAP_LIB} ${CPP_EXE_RUN} ${CUDA_EXE_RUN}
print:
	@echo CUDA: ${CUDA_DIR}
	@echo exes: ${CPP_EXE_RUN} ${CUDA_EXE_RUN}
	@echo libs: ${COMMON_LIB} 
	@echo cuda_libs: ${CUDA_WRAP_LIB} 
	@mkdir -p ${LOCAL_INCLUDE}
	@mkdir -p ${LOCAL_LIB}
	@mkdir -p ${LOCAL_OBJ}

${LOCAL_BIN}/cdi_run: ${LOCAL_OBJ}/cdi.o ${COMMON_LIB} ${CUDA_WRAP_LIB}
	${MPICXX} $< -o $@ ${LINK_FLAGS_EXT} ${LINK_FLAGS} -L${CUDA_LIB} -lcufft -lcudart
${LOCAL_BIN}/test_run: ${LOCAL_OBJ}/test.o ${COMMON_LIB} ${CUDA_WRAP_LIB}
	${MPICXX} $< -o $@ ${LINK_FLAGS_EXT} ${LINK_FLAGS} -L${CUDA_LIB} -lcufft -lcudart


${LOCAL_BIN}/%_run: ${LOCAL_OBJ}/%.o ${COMMON_LIB} ${CUDA_WRAP_LIB}
	${CXX} $< -o $@ ${LINK_FLAGS_EXT} ${LINK_FLAGS} -L${CUDA_LIB} -lcufft -lcudart

${LOCAL_BIN}/%_cu: ${LOCAL_OBJ}/%_cu.o ${COMMON_LIB} ${CUDA_WRAP_LIB}
	${CXX} $< -o $@ ${LINK_FLAGS_EXT_CU} ${LINK_FLAGS} -L${CUDA_LIB} -lcufft -lcudart

${LOCAL_LIB}/lib%.so: ${LOCAL_OBJ}/%.o ${LOCAL_LIB}/libreadCXI.so
	${CXX} -shared $< -o $@ ${LINK_FLAGS_EXT} -L${LOCAL_LIB} -lreadCXI

${LOCAL_LIB}/libreadCXI.so: ${LOCAL_OBJ}/readCXI.o
	${MPICXX} -shared $< -o $@ ${LINK_FLAGS_EXT}

${LOCAL_OBJ}/%.o: src/common/%.c
	${GCC} -c -fPIC $< ${INCLUDE_FLAGS} -o $@

${LOCAL_OBJ}/%.o: util/%.cpp
	${CXX} -c $< ${INCLUDE_FLAGS} -o $@

${LOCAL_OBJ}/%_cu.o: util/%.cu
	${NVCC} -c $< $(patsubst -pthread%, %, ${INCLUDE_FLAGS}) -o $@

${LOCAL_OBJ}/%.o: util/%.cpp
	${CXX} -c $< ${INCLUDE_FLAGS} -o $@

${LOCAL_OBJ}/%.o: src/common/%.cc
	${CXX} -c -fPIC $< ${INCLUDE_FLAGS} -o $@

${LOCAL_OBJ}/sparse.o: src/common/sparse.cc
	${CXX} -c -fPIC $< ${INCLUDE_FLAGS} -o $@

${LOCAL_LIB}/lib%_cu.a: ${LOCAL_OBJ}/%_cu.o
	ar cr $@ $<
	ranlib $@

${LOCAL_OBJ}/%_cu.o: src/gpu/%.cu
	${NVCC} -c $< -o $@ $(patsubst -pthread%, %, ${INCLUDE_FLAGS})
clean:
	rm lib/* obj/* bin/*

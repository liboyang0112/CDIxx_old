CUDA_DIR=$(patsubst %bin/nvcc, %, $(shell which nvcc))
CUDA_LIB=-L${CUDA_DIR}/lib64 -lcufft -lcudadevrt -lcudart
LOCAL_LIB=lib
LOCAL_INCLUDE=include
LOCAL_OBJ=obj
LOCAL_BIN=bin
GCC=gcc-10 -Ofast
CXX=g++-10 -g -D__CUDA__=1
MPICXX=mpic++
NVCC=nvcc -arch=sm_35 -Xptxas=-O3# --compiler-bindir /usr/bin/g++-10 -g -rdc=true

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

CUDA_WRAP_LIB_SRC=$(wildcard src/gpu/*)
CUDA_WRAP_LIB_OBJ=$(patsubst src/gpu/%cc, ${LOCAL_OBJ}/gpu/%o, $(patsubst src/gpu/%.cu, ${LOCAL_OBJ}/gpu/%_cu.o, ${CUDA_WRAP_LIB_SRC}))
CUDA_WRAP_LIB=${LOCAL_LIB}/libcudaWrap.a
#CUDA_WRAP_LIB=${LOCAL_LIB}/libcudaWrap.so

LINK_FLAGS_EXT=$(shell pkg-config --libs opencv4 hdf5 tbb fftw3) -lfftw3_mpi -lcholmod -lfftw3_threads -lm -lpthread -lconfig++
LINK_FLAGS_EXT_CU=$(shell pkg-config --libs opencv4 hdf5)
#LINK_FLAGS+=$(patsubst ${LOCAL_LIB}/lib%.so, -l%, ${CPP_LIB})
LINK_FLAGS_CUDA_WRAP=$(patsubst ${LOCAL_LIB}/lib%.a, -l%, ${CUDA_WRAP_LIB})
LINK_FLAGS= -L${LOCAL_LIB} ${LINK_FLAGS_CUDA_WRAP} $(patsubst ${LOCAL_LIB}/lib%.so, -l%, ${COMMON_LIB} ${COMMON_C_LIB})
INCLUDE_FLAGS=-I${LOCAL_INCLUDE} $(shell pkg-config --cflags opencv4 hdf5 mpi) -I/usr/include/suitesparse

all: print
	make -j16 objs
	make -j16 commonlibs
	make -j16 cudalibs
	make -j16 exes

print:
	@echo CUDA: ${CUDA_DIR}
	@echo objs: ${COMMON_LIB_OBJ} ${CUDA_WRAP_LIB_OBJ} ${CPP_EXE_OBJ} ${COMMON_C_LIB_OBJ} ${CUDA_EXE_OBJ}
	@echo exes: ${CPP_EXE_RUN} ${CUDA_EXE_RUN}
	@echo libs: ${COMMON_LIB} 
	@echo cuda_libs: ${CUDA_WRAP_LIB} 
	@mkdir -p ${LOCAL_INCLUDE}
	@mkdir -p ${LOCAL_LIB}
	@mkdir -p ${LOCAL_OBJ}
	@mkdir -p ${LOCAL_OBJ}/gpu

exes: ${CPP_EXE_RUN} ${CUDA_EXE_RUN}
cudalibs: ${CUDA_WRAP_LIB}
commonlibs: ${COMMON_C_LIB} ${COMMON_LIB} 
objs: ${COMMON_LIB_OBJ} ${CUDA_WRAP_LIB_OBJ} ${CPP_EXE_OBJ} ${COMMON_C_LIB_OBJ} ${CUDA_EXE_OBJ}


${LOCAL_BIN}/cdi_run: ${LOCAL_OBJ}/cdi.o ${LOCAL_OBJ}/gpu/cudaWraplink.o
	${MPICXX} $^ -o $@ ${LINK_FLAGS} ${LINK_FLAGS_EXT} ${CUDA_LIB}
${LOCAL_BIN}/test_run: ${LOCAL_OBJ}/test.o ${LOCAL_OBJ}/gpu/cudaWraplink.o
	${MPICXX} $^ -o $@ ${LINK_FLAGS} ${LINK_FLAGS_EXT} ${CUDA_LIB}


${LOCAL_BIN}/%_run: ${LOCAL_OBJ}/%.o ${LOCAL_OBJ}/gpu/cudaWraplink.o
	${CXX} $^ -o $@  ${LINK_FLAGS} ${LINK_FLAGS_EXT} ${CUDA_LIB}

${LOCAL_BIN}/%_cu: ${LOCAL_OBJ}/%_cu.o ${LOCAL_OBJ}/%link_cu.o
	${CXX} $^ -o $@ ${LINK_FLAGS} ${LINK_FLAGS_EXT_CU} ${CUDA_LIB}

${LOCAL_LIB}/lib%.so: ${LOCAL_OBJ}/%.o
	${CXX} -shared $< -o $@ ${LINK_FLAGS_EXT} -L${LOCAL_LIB} -lreadCXI

${LOCAL_LIB}/libreadCXI.so: ${LOCAL_OBJ}/readCXI.o
	${MPICXX} -shared $< -o $@ ${LINK_FLAGS_EXT}

${LOCAL_OBJ}/%.o: src/common/%.c
	${GCC} -c -fPIC $< ${INCLUDE_FLAGS} -o $@

${LOCAL_OBJ}/%.o: util/%.cpp
	${CXX} -c $< ${INCLUDE_FLAGS} -o $@

${LOCAL_OBJ}/%_cu.o: util/%.cu
	${NVCC} -rdc=true -c $< $(patsubst -pthread%, %, ${INCLUDE_FLAGS}) -o $@

${LOCAL_OBJ}/%link_cu.o: ${LOCAL_OBJ}/%_cu.o ${CUDA_WRAP_LIB_OBJ}
	${NVCC} -dlink $^ -o $@

${LOCAL_OBJ}/%.o: util/%.cpp
	${CXX} -c $< ${INCLUDE_FLAGS} -o $@

${LOCAL_OBJ}/%.o: src/common/%.cc
	${CXX} -c -fPIC $< ${INCLUDE_FLAGS} -o $@

${LOCAL_LIB}/libcudaWrap.so: ${CUDA_WRAP_LIB_OBJ} ${LOCAL_OBJ}/gpu/cudaWraplink.o
	${CXX}  -shared -o $@ $^ -L${LOCAL_LIB} -lsparse -lformat ${CUDA_LIB} ${LINK_FLAGS_EXT}

${LOCAL_LIB}/libcudaWrap.a: ${CUDA_WRAP_LIB_OBJ}# ${LOCAL_OBJ}/gpu/cudaWraplink.o
	ar cr $@ $^

${LOCAL_OBJ}/gpu/%_cu.o: src/gpu/%.cu
	#${NVCC} -rdc=true -Xcompiler '-fPIC' -c $< -o $@ $(patsubst -pthread%, %, ${INCLUDE_FLAGS})
	${NVCC} -rdc=true -c $< -o $@ $(patsubst -pthread%, %, ${INCLUDE_FLAGS})

${LOCAL_OBJ}/gpu/%.o: src/gpu/%.cc
	#${NVCC} -rdc=true -Xcompiler '-fPIC' -c $< -o $@ $(patsubst -pthread%, %, ${INCLUDE_FLAGS})
	${CXX} -c $< -o $@ $(patsubst -pthread%, %, ${INCLUDE_FLAGS})

${LOCAL_OBJ}/gpu/cudaWraplink.o: ${CUDA_WRAP_LIB_OBJ}
	#${NVCC} -Xcompiler '-fPIC' -dlink $^ -o $@
	${NVCC} -dlink $^ -o $@
clean:
	rm -r lib/* obj/* bin/*

CUDA_DIR=$(patsubst %bin/nvcc, %, $(shell which nvcc))
CUDA_LIB=-L${CUDA_DIR}/lib64 -lcufft -lcudadevrt -lcudart
LOCAL_LIB=lib
LOCAL_INCLUDE=include
LOCAL_OBJ=obj
LOCAL_BIN=bin
GCC=gcc-10 -Ofast
CXX=g++-10 -g -D__CUDA__=1
MPICXX=mpic++
NVCC=nvcc -Xptxas=-O3 -g# --compiler-bindir /usr/bin/g++-10 -g -rdc=true

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

CUDA_EXT_LIB_SRC=$(wildcard src/gpu_ext/*)
CUDA_EXT_LIB_OBJ=$(patsubst src/gpu_ext/%cc, ${LOCAL_OBJ}/gpu_ext/%o, $(patsubst src/gpu_ext/%.cu, ${LOCAL_OBJ}/gpu_ext/%_cu.o, ${CUDA_EXT_LIB_SRC}))
CUDA_EXT_LIB=${LOCAL_LIB}/libcudaExt.so

TORCH_WRAP_LIB_SRC=$(wildcard src/torch/*)
TORCH_WRAP_LIB_OBJ=$(patsubst src/torch/%cc, ${LOCAL_OBJ}/torch/%o, ${TORCH_WRAP_LIB_SRC})
TORCH_WRAP_LIB=${LOCAL_LIB}/libtorchWrap.so

VTK_WRAP_LIB_SRC=$(wildcard src/vtk/*)
VTK_WRAP_LIB_OBJ=$(patsubst src/vtk/%cc, ${LOCAL_OBJ}/vtk/%o, ${VTK_WRAP_LIB_SRC})
VTK_WRAP_LIB=${LOCAL_LIB}/libvtkWrap.so

LINK_FLAGS_EXT=$(shell pkg-config --libs opencv4 hdf5 tbb fftw3) -lfftw3_mpi -lcholmod -lfftw3_threads -lm -lpthread -lconfig++ -lz
LINK_FLAGS_EXT_CU=$(shell pkg-config --libs opencv4 hdf5)
#LINK_FLAGS+=$(patsubst ${LOCAL_LIB}/lib%.so, -l%, ${CPP_LIB})
LINK_FLAGS_CUDA_WRAP=$(patsubst ${LOCAL_LIB}/lib%.a, -l%, ${CUDA_WRAP_LIB})
LINK_FLAGS_CUDA_EXT=$(patsubst ${LOCAL_LIB}/lib%.so, -l%, ${CUDA_EXT_LIB})
LINK_FLAGS_TORCH_WRAP=$(patsubst ${LOCAL_LIB}/lib%.so, -l%, ${TORCH_WRAP_LIB})
LINK_FLAGS_VTK_WRAP=$(patsubst ${LOCAL_LIB}/lib%.so, -l%, ${VTK_WRAP_LIB})
LINK_FLAGS= -L${LOCAL_LIB} -lvtkWrap  -lmmio  -lcnpy -lcommon -limageReader -lsparse  -lwrite -lcudaExt -ltorchWrap -lreadCXI -lreadConfig -lmemManager 
#LINK_FLAGS= -L${LOCAL_LIB} ${LINK_FLAGS_CUDA_WRAP} ${LINK_FLAGS_VTK_WRAP} $(patsubst ${LOCAL_LIB}/lib%.so, -l%, ${COMMON_LIB} ${COMMON_C_LIB}) ${LINK_FLAGS_CUDA_EXT} -lmemManager -lreadConfig ${LINK_FLAGS_TORCH_WRAP}
LINK_FLAGS_TORCH= -L${torch}/lib -lc10_cuda -lcaffe2_nvrtc -lshm -ltorch_cpu -ltorch_cuda_linalg -ltorch_cuda -ltorch_global_deps -ltorch_python -ltorch -lc10
VTK_INCLUDE=-I /usr/include/vtk-9.1
VTK_LIBS=-lvtkCommonColor-9.1 -lvtkCommonCore-9.1 -lvtkCommonDataModel-9.1 -lvtkFiltersGeometry-9.1\
				 -lvtkIOXML-9.1 -lvtkInteractionStyle-9.1 -lvtkRenderingContextOpenGL2-9.1 -lvtkRenderingCore-9.1\
				 -lvtkRenderingFreeType-9.1 -lvtkRenderingGL2PSOpenGL2-9.1 -lvtkRenderingOpenGL2-9.1\
				 -lvtkRenderingVolumeOpenGL2-9.1 -lvtkCommonExecutionModel-9.1 -lvtkRenderingVolume-9.1

INCLUDE_FLAGS=-I${LOCAL_INCLUDE} $(shell pkg-config --cflags opencv4 hdf5 mpi) -I/usr/include/suitesparse
INCLUDE_FLAGS_TORCH=-I${torch}/include -I${torch}/include/torch/csrc/api/include

all: print
	make -j16 objs
	make -j16 commonlibs
	make -j16 cudalibs
	make -j16 exes

print:
	@echo CUDA: ${CUDA_DIR}
	@echo objs: ${CUDA_EXE_OBJ} ${CUDA_WRAP_LIB_OBJ} ${TORCH_WRAP_LIB_OBJ} ${CPP_EXE_OBJ} ${COMMON_LIB_OBJ} ${COMMON_C_LIB_OBJ} ${CUDA_EXT_LIB_OBJ}
	@echo exes: ${CPP_EXE_RUN} ${CUDA_EXE_RUN}
	@echo libs: ${COMMON_LIB} 
	@echo cuda_libs: ${CUDA_WRAP_LIB} 
	@echo cuda_extlibs: ${CUDA_EXT_LIB} 
	@echo torch_libs: ${TORCH_WRAP_LIB} 
	@mkdir -p ${LOCAL_INCLUDE}
	@mkdir -p ${LOCAL_LIB}
	@mkdir -p ${LOCAL_OBJ}
	@mkdir -p ${LOCAL_OBJ}/gpu
	@mkdir -p ${LOCAL_OBJ}/gpu_ext
	@mkdir -p ${LOCAL_OBJ}/torch
	@mkdir -p ${LOCAL_OBJ}/vtk

exes: ${CPP_EXE_RUN} ${CUDA_EXE_RUN}
cudalibs: ${CUDA_WRAP_LIB} ${TORCH_WRAP_LIB} ${VTK_WRAP_LIB} ${CUDA_EXT_LIB}
commonlibs: ${COMMON_C_LIB} ${COMMON_LIB} 
objs: ${COMMON_LIB_OBJ} ${TORCH_WRAP_LIB_OBJ} ${CUDA_WRAP_LIB_OBJ} ${CPP_EXE_OBJ} ${COMMON_C_LIB_OBJ} ${CUDA_EXE_OBJ} ${CUDA_EXT_LIB_OBJ}


${LOCAL_BIN}/%_run: ${LOCAL_OBJ}/%.o
	${CXX} $^ -o $@  ${LINK_FLAGS} ${LINK_FLAGS_EXT} ${CUDA_LIB} ${VTK_LIBS}

${LOCAL_BIN}/%_cu: ${LOCAL_OBJ}/%_cu.o ${LOCAL_OBJ}/%link_cu.o
	${CXX} $^ -o $@ ${LINK_FLAGS} ${LINK_FLAGS_EXT_CU} ${CUDA_LIB} ${LINK_FLAGS_TORCH} ${VTK_LIBS} ${LINK_FLAGS_CUDA_WRAP}

${LOCAL_LIB}/lib%.so: ${LOCAL_OBJ}/%.o
	${CXX} -shared $< -o $@ ${LINK_FLAGS_EXT} -L${LOCAL_LIB} -lreadCXI

${LOCAL_LIB}/libreadCXI.so: ${LOCAL_OBJ}/readCXI.o
	${MPICXX} -shared $< -o $@ ${LINK_FLAGS_EXT}

${LOCAL_OBJ}/%.o: src/common/%.c
	${GCC} -c -fPIC $< ${INCLUDE_FLAGS} -o $@

${LOCAL_OBJ}/%.o: util/%.cc
	${CXX} -c $< ${INCLUDE_FLAGS} -o $@

${LOCAL_OBJ}/%_cu.o: util/%.cu
	${NVCC} -rdc=true -c $< $(patsubst -pthread%, %, ${INCLUDE_FLAGS}) -o $@

${LOCAL_OBJ}/%link_cu.o: ${LOCAL_OBJ}/%_cu.o ${CUDA_WRAP_LIB_OBJ}
	${NVCC} -dlink $^ -o $@

${LOCAL_OBJ}/%.o: util/%.cpp
	${CXX} -c $< ${INCLUDE_FLAGS} -o $@

${LOCAL_OBJ}/%.o: src/common/%.cc
	${CXX} -c -fPIC $< ${INCLUDE_FLAGS} -o $@
	
${TORCH_WRAP_LIB}: ${TORCH_WRAP_LIB_OBJ}
	${CXX} -shared $^ -o $@ ${LINK_FLAGS_EXT} -L${LOCAL_LIB} ${LINK_FLAGS_TORCH}

${VTK_WRAP_LIB}: ${VTK_WRAP_LIB_OBJ}
	${CXX} -shared $^ -o $@ ${VTK_LIBS}

${LOCAL_LIB}/libcudaExt.so: ${CUDA_EXT_LIB_OBJ} ${LOCAL_OBJ}/gpu/cudaWraplink.o
	${CXX}  -shared -o $@ $^ -L${LOCAL_LIB} -lsparse ${CUDA_LIB} ${LINK_FLAGS_EXT} ${CUDA_WRAP_LIB}

${CUDA_WRAP_LIB}: ${CUDA_WRAP_LIB_OBJ}# ${LOCAL_OBJ}/gpu/cudaWraplink.o
	ar cr $@ $^

${LOCAL_OBJ}/torch/%.o: src/torch/%.cc
	${CXX} -D_GLIBCXX_USE_CXX11_ABI=0 -c -fPIC $< ${INCLUDE_FLAGS} ${INCLUDE_FLAGS_TORCH} -o $@ # may use this with older torch version

${LOCAL_OBJ}/vtk/%.o: src/vtk/%.cc
	${CXX} -c -fPIC $< ${INCLUDE_FLAGS} ${VTK_INCLUDE} -o $@

${LOCAL_OBJ}/gpu_ext/%_cu.o: src/gpu_ext/%.cu
	${NVCC} -rdc=true -Xcompiler '-fPIC' -c $< -o $@ $(patsubst -pthread%, %, ${INCLUDE_FLAGS})

#${LOCAL_OBJ}/gpu_ext/cudaExtlink.o: 
#	${NVCC} -Xcompiler '-fPIC' -dlink $^ -o $@

${LOCAL_OBJ}/gpu/%_cu.o: src/gpu/%.cu
	#${NVCC} -rdc=true -Xcompiler '-fPIC' -c $< -o $@ $(patsubst -pthread%, %, ${INCLUDE_FLAGS})
	${NVCC} -rdc=true -Xcompiler '-fPIC' -c $< -o $@ $(patsubst -pthread%, %, ${INCLUDE_FLAGS})

${LOCAL_OBJ}/gpu/%.o: src/gpu/%.cc
	#${NVCC} -rdc=true -Xcompiler '-fPIC' -c $< -o $@ $(patsubst -pthread%, %, ${INCLUDE_FLAGS})
	${CXX} -c $< -o $@ $(patsubst -pthread%, %, ${INCLUDE_FLAGS})

${LOCAL_OBJ}/gpu/cudaWraplink.o: ${CUDA_WRAP_LIB_OBJ} ${CUDA_EXT_LIB_OBJ}
	${NVCC} -Xcompiler '-fPIC' -dlink $^ -o $@
	#${NVCC} -dlink $^ -o $@
clean:
	rm -r lib/* obj/* bin/*

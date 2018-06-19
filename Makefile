ROOT_DIR= $(shell pwd)
TARGETS= toolkits/bc toolkits/bfs toolkits/cc toolkits/pagerank toolkits/sssp toolkits/partition
MACROS=
#MACROS= -D PRINT_DEBUG_MESSAGES

MPICXX= mpicxx
CXXFLAGS= -O3 -std=c++14 -g -fopenmp -march=native -I$(ROOT_DIR) $(MACROS)
SYSLIBS= -lnuma -lzstd
HEADERS= $(shell find . -name '*.hpp')

all: $(TARGETS)

toolkits/%: toolkits/%.cpp $(HEADERS)
	$(MPICXX) $(CXXFLAGS) -o $@ $< $(SYSLIBS)

clean: 
	rm -f $(TARGETS)


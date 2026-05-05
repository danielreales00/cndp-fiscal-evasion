# Makefile for CNDP Solver (C++ implementation)
# Targets macOS ARM64 with Apple clang, HiGHS via Homebrew, optional OpenMP

CXX := clang++
CXXFLAGS := -std=c++17 -O3 -march=native -Wall -Wextra

TARGET := cndp_solver
SRC := cndp_solver.cpp

# --- HiGHS (ILP solver) ---
# Try to detect HiGHS installation via Homebrew
HIGHS_PREFIX := $(shell brew --prefix highs 2>/dev/null)
ifdef HIGHS_PREFIX
    HIGHS_CFLAGS := -DUSE_HIGHS -I$(HIGHS_PREFIX)/include -I$(HIGHS_PREFIX)/include/highs
    HIGHS_LDFLAGS := -L$(HIGHS_PREFIX)/lib -lhighs -Wl,-rpath,$(HIGHS_PREFIX)/lib
else
    # Fallback: check standard /opt/homebrew paths
    ifneq ($(wildcard /opt/homebrew/include/Highs.h),)
        HIGHS_CFLAGS := -DUSE_HIGHS -I/opt/homebrew/include
        HIGHS_LDFLAGS := -L/opt/homebrew/lib -lhighs -Wl,-rpath,/opt/homebrew/lib
    else ifneq ($(wildcard /usr/local/include/Highs.h),)
        HIGHS_CFLAGS := -DUSE_HIGHS -I/usr/local/include
        HIGHS_LDFLAGS := -L/usr/local/lib -lhighs -Wl,-rpath,/usr/local/lib
    else
        $(warning HiGHS not found. Building without ILP support. Install with: brew install highs)
        HIGHS_CFLAGS :=
        HIGHS_LDFLAGS :=
    endif
endif

# --- OpenMP ---
# On macOS with Apple clang, OpenMP requires libomp from Homebrew
OMP_PREFIX := $(shell brew --prefix libomp 2>/dev/null)
ifdef OMP_PREFIX
    OMP_CFLAGS := -Xclang -fopenmp -D_OPENMP -I$(OMP_PREFIX)/include
    OMP_LDFLAGS := -L$(OMP_PREFIX)/lib -lomp
else
    # Fallback paths
    ifneq ($(wildcard /opt/homebrew/opt/libomp/include/omp.h),)
        OMP_CFLAGS := -Xclang -fopenmp -D_OPENMP -I/opt/homebrew/opt/libomp/include
        OMP_LDFLAGS := -L/opt/homebrew/opt/libomp/lib -lomp
    else
        $(warning OpenMP (libomp) not found. Building without parallelism. Install with: brew install libomp)
        OMP_CFLAGS :=
        OMP_LDFLAGS :=
    endif
endif

# --- Combine flags ---
ALL_CFLAGS := $(CXXFLAGS) $(HIGHS_CFLAGS) $(OMP_CFLAGS)
ALL_LDFLAGS := $(HIGHS_LDFLAGS) $(OMP_LDFLAGS)

# --- Targets ---
.PHONY: all clean info

all: $(TARGET)

$(TARGET): $(SRC) graph.h union_find.h
	$(CXX) $(ALL_CFLAGS) -o $@ $(SRC) $(ALL_LDFLAGS)

# Build without HiGHS (heuristics only)
no-highs:
	$(CXX) $(CXXFLAGS) $(OMP_CFLAGS) -o $(TARGET) $(SRC) $(OMP_LDFLAGS)

# Build without OpenMP (sequential)
no-omp:
	$(CXX) $(CXXFLAGS) $(HIGHS_CFLAGS) -o $(TARGET) $(SRC) $(HIGHS_LDFLAGS)

# Build minimal (no HiGHS, no OpenMP)
minimal:
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)

info:
	@echo "CXX:          $(CXX)"
	@echo "CXXFLAGS:     $(CXXFLAGS)"
	@echo "HIGHS_PREFIX: $(HIGHS_PREFIX)"
	@echo "HIGHS_CFLAGS: $(HIGHS_CFLAGS)"
	@echo "HIGHS_LDFLAGS:$(HIGHS_LDFLAGS)"
	@echo "OMP_PREFIX:   $(OMP_PREFIX)"
	@echo "OMP_CFLAGS:   $(OMP_CFLAGS)"
	@echo "OMP_LDFLAGS:  $(OMP_LDFLAGS)"

# Compilation de la bibliothèque partagée (Linux / macOS). Sous Windows : voir CMakeLists.txt.
CXX ?= g++
CXXFLAGS ?= -O3 -march=native -std=c++17 -fPIC -fvisibility=hidden -Wall -Wextra -Wno-unused-parameter
UNAME := $(shell uname)
ifeq ($(UNAME),Darwin)
  LIB = libmoteur.dylib
else
  LIB = libmoteur.so
endif

$(LIB): moteur.cpp
	$(CXX) $(CXXFLAGS) -shared -o $(LIB) moteur.cpp

clean:
	rm -f libmoteur.so libmoteur.dylib moteur.dll

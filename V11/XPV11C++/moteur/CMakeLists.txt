# Compilation multiplateforme (Linux, macOS, Windows/MSVC) :
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build --config Release
# La bibliothèque produite (libmoteur.so, libmoteur.dylib ou moteur.dll) est copiée à côté
# de moteur.cpp, où moteur_rapide.py la cherche.
cmake_minimum_required(VERSION 3.15)
project(moteur CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_library(moteur SHARED moteur.cpp)
if(MSVC)
  target_compile_options(moteur PRIVATE /O2 /fp:precise)
else()
  target_compile_options(moteur PRIVATE -O3 -march=native -fvisibility=hidden)
endif()
add_custom_command(TARGET moteur POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:moteur> ${CMAKE_CURRENT_SOURCE_DIR}/)

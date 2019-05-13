project(ExaJet)

cmake_minimum_required(VERSION 3.10)

set(gdt_dir ../gdt)

include_directories(${gdt_dir})

add_executable(exa
  src/main.cpp
  )
target_link_libraries(exa
  tbb
  )
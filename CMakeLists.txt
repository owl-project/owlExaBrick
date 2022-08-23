# ======================================================================== #
# Copyright 2018-2019 Ingo Wald                                            #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ======================================================================== #

project(ExaJet)

cmake_minimum_required(VERSION 2.8)

# ------------------------------------------------------------------
# first, include owl project to do some general configuration stuff
# (build modes, glut, optix, etc)
# ------------------------------------------------------------------

mark_as_advanced(CUDA_SDK_ROOT_DIR)
set(owl_dir ${CMAKE_CURRENT_SOURCE_DIR}/submodules/owl)
add_subdirectory(${owl_dir} EXCLUDE_FROM_ALL)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${owl_dir}/owl/cmake/")
include_directories(${OWL_INCLUDES})

include(configure_cuda)



# ------------------------------------------------------------------
# import glui, for user interface
# ------------------------------------------------------------------
find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIR})
set(glui_dir ${PROJECT_SOURCE_DIR}/submodules/glui)
include_directories(${glui_dir}/glui/include)
add_subdirectory(${glui_dir} glui EXCLUDE_FROM_ALL)



# ------------------------------------------------------------------
# import owl, for owl stuff
# ------------------------------------------------------------------
include_directories(${OWL_INCLUDES})
include_directories(${OWL_VIEWER_INCLUDES})
add_definitions(${OWL_CXX_FLAGS})
include_directories("${CMAKE_CURRENT_SOURCE_DIR}")



# ------------------------------------------------------------------
# and final build rules for the project
# ------------------------------------------------------------------
option(BASIS_METHOD "Interpolate via Basis Function Method?" ON)
if (BASIS_METHOD)
  add_definitions(-DBASIS_METHOD=1)
endif()

option(ANALYTIC_GRADIENTS "Compute analytic gradients (not central differences) when using Basis Function Method" ON)
if (ANALYTIC_GRADIENTS)
  add_definitions(-DANALYTIC_GRADIENTS=1)
endif()

option(ALLOW_EMPTY_CELLS "Allow empty cells in bricks?" OFF)
if (ALLOW_EMPTY_CELLS)
  add_definitions(-DALLOW_EMPTY_CELLS=1)
endif()

option(ENABLE_GRADIENT_SHADING "Enable gradient shading" ON)
if (ENABLE_GRADIENT_SHADING)
    add_definitions(-DENABLE_GRADIENT_SHADING=1)
endif()

option(FAST_CENTRAL_DIFFERENCES "In CD gradient mode, clamp sample positions at local brick boundaries" OFF)
if (FAST_CENTRAL_DIFFERENCES)
  add_definitions(-DFAST_CENTRAL_DIFFERENCES=1)
endif()

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${GLUT_INCLUDE_DIR})

add_executable(exaBuilder
  builder/builder.cpp
  )
target_link_libraries(exaBuilder
  ${TBB_LIBRARIES}
  ${OWL_LIBRARIES}
  ${OWL_VIEWER_LIBRARIES}
)

cuda_compile_and_embed(
  ptxCode programs/exabrick.cu
  )
add_executable(exaViewer
  glutViewer/Camera.cpp
  glutViewer/FlyMode.cpp
  glutViewer/InspectMode.cpp
  glutViewer/OWLViewer.cpp

  exa/viewer.cpp
  exa/TriangleMesh.cpp
  exa/ScalarField.cpp
  exa/Config.cpp
  exa/OptixRenderer.cpp
  exa/ExaBricks.cpp
  exa/Regions.cpp
  exa/Texture.cpp
  exa/ColorMapper.cpp
  ${ptxCode}
  )
target_link_libraries(exaViewer
  glui_static

  ${OWL_LIBRARIES}
  ${OWL_VIEWER_LIBRARIES}
  ${TBB_LIBRARIES}
  ${GLUT_LIBRARIES}
  ${OPENGL_LIBRARIES}
  ${CMAKE_DL_LIBS}
  )




add_executable(exaRawToCells
  tools/fromVolume/raw2cells.cpp
  )
target_link_libraries(exaRawToCells
  ${TBB_LIBRARIES}
  ${OWL_LIBRARIES}
  )


# ------------------------------------------------------------------
# Tool to create some simple test data sets in the exa format
# ------------------------------------------------------------------

add_executable(exaArtificial
  tools/artificial/artificial.cpp
  )

# ------------------------------------------------------------------
# Flash converter depends on HDF5, built optionally,
# and only if HDF5 was found
# ------------------------------------------------------------------

option(FLASH_TO_CELLS "Build the exaFlashToCells tool (needs HDF5)" OFF)
if (FLASH_TO_CELLS)
  find_package(HDF5 COMPONENTS CXX)
  if (HDF5_FOUND)
    add_executable(exaFlashToCells
      tools/fromFlash/flash2cells.cpp
      )
    include_directories(${HDF5_CXX_INCLUDE_DIRS})
    target_link_libraries(exaFlashToCells ${HDF5_LIBRARIES})
  endif()
endif()

# ------------------------------------------------------------------
# Flash converter depends on HDF5, built optionally,
# and only if HDF5 was found
# ------------------------------------------------------------------

option(LANL_TO_CELLS "Build the readVTU tool (loads LANL ocean)" OFF)
if (LANL_TO_CELLS)
  add_subdirectory(tools/convertLanlOcean)
endif()

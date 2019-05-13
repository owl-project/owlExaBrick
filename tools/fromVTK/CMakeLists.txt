project(fromVTK)

cmake_minimum_required(VERSION 3.3 FATAL_ERROR)

include_directories(../../submodules/gdt)

find_package(VTK COMPONENTS 
  vtkCommonColor
  vtkCommonCore
  vtkIOXML
  vtkInteractionStyle
  vtkRenderingCore
  vtkCommonDataModel
  vtkIOLegacy
  vtkRenderingFreeType
  vtkRenderingOpenGL2 QUIET)
if (NOT VTK_FOUND)
  message("Skipping fromVTK (vtk not found): ${VTK_NOT_FOUND_MESSAGE}")
  return ()
endif()
message (STATUS "VTK_VERSION: ${VTK_VERSION}")
if (VTK_VERSION VERSION_LESS "8.90.0")
  # old system
  include(${VTK_USE_FILE})
  add_executable(fromVTK MACOSX_BUNDLE fromVTK.cpp )
  target_link_libraries(fromVTK PRIVATE ${VTK_LIBRARIES})
else ()
  # include all components
  add_executable(fromVTK MACOSX_BUNDLE fromVTK.cpp )
  target_link_libraries(fromVTK PRIVATE ${VTK_LIBRARIES})
  # vtk_module_autoinit is needed
  vtk_module_autoinit(
    TARGETS fromVTK
    MODULES ${VTK_LIBRARIES}
    )
endif () 

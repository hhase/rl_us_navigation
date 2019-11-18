cmake_minimum_required(VERSION 3.3.0)
project(tracking_plug_in)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(ImFusionLib REQUIRED)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)


# Define and configure the targets
set(PLUGIN_SOURCES
	src/algorithm.cpp
	src/controller.cpp
	src/factory.cpp
	src/plugin.cpp
	src/controller.ui
)

set(PLUGIN_HEADERS
        include/${PROJECT_NAME}/algorithm.hpp
        include/${PROJECT_NAME}/controller.hpp
        include/${PROJECT_NAME}/factory.hpp
        include/${PROJECT_NAME}/plugin.hpp
)



add_library(${PROJECT_NAME} SHARED
        ${PLUGIN_SOURCES}
        ${PLUGIN_HEADERS}
        ${PLUGIN_FORMS}
        )

target_include_directories(${PROJECT_NAME} PRIVATE
      include
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_BINARY_DIR}
      )

target_link_libraries(${PROJECT_NAME}
      ImFusionLib
      ImFusionCT
      ImFusionDicom
      )

###### INSTALLATION
include(GNUInstallDirs)

install(TARGETS ${PROJECT_NAME}
  EXPORT "${PROJECT_NAME}Targets"
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

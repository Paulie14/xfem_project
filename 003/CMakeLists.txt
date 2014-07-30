##                                                                       ##
#           A simple CMakeLists.txt file for use with deal.II             #
##                                                                       ##

#
# Set the name of the project and target:
#
SET(TARGET "main")

#
# Declare all source files the target consists of:
#
SET(TARGET_SRC
  ${TARGET}.cc
  system.cc   
  parameters.cc    
  well.cc
  data_cell.cc
  xfevalues.cc
  mapping.cc
  model_base.cc
  model.cc
  bem_model.cc
  exact_model.cc
  adaptive_integration.cc    
  xmodel.cc    
  simple_models.cc        
  comparing.cc
  # You can specify additional files here!
  )

#
# (Optional)
# Specify a list of files (file globs) that will be removed
# with the "make runclean" and "make distclean" targets.
# If empty, sensible default values will be used.
#
SET(CLEAN_UP_FILES
  # a custom list of globs, e.g. *.log *.vtk
  )

#
# (Optional)
# A custom command line that should be invoked by "make run". If empty,
# ./${TARGET} will be invoked.
#
SET(TARGET_RUN
  # a custom command line, e.g. mpirun -np 2 ${TARGET}
  )

#
# It is good practice to specify a version requirement:
#
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

#
# Find and import the deal.II project configuration:
#
FIND_PACKAGE(deal.II 8.0 REQUIRED
  HINTS
    ${DEAL_II_DIRS} ../ ../../ $ENV{DEAL_II_DIR}
    $ENV{HOME}/local/deal.II
  #
  # If the deal.II library cannot be found (because it is not installed at a
  # default location or your project resides at an uncommon place), you
  # can specify additional hints for search paths here, e.g.
  # "$ENV{HOME}/workspace/deal.II"
  #
  )

#
# And let deal.II do the rest:
#
DEAL_II_INITIALIZE_CACHED_VARIABLES()

#SET(CMAKE_CXX_FLAGS "${DEAL_II_CXX_FLAGS} -std=c++11")
MESSAGE("CMAKE_CXX_FLAGS:  ${CMAKE_CXX_FLAGS}")

PROJECT(${TARGET})
DEAL_II_INVOKE_AUTOPILOT()

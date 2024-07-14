# Define source files
set(APP_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/config.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/libs/curve_fit.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/utils/program_options.cpp
)

# Define headers
set(APP_HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/data_set.h
  ${CMAKE_CURRENT_SOURCE_DIR}/config.h
  ${CMAKE_CURRENT_SOURCE_DIR}/model_tester.h
  ${CMAKE_CURRENT_SOURCE_DIR}/models/models.h
  ${CMAKE_CURRENT_SOURCE_DIR}/libs/curve_fit.h
  ${CMAKE_CURRENT_SOURCE_DIR}/models/math_util.h
  ${CMAKE_CURRENT_SOURCE_DIR}/utils/program_options.h
)

# Add executable
add_executable(${PROJECT_NAME} ${APP_SOURCES} ${APP_HEADERS})

# Include directories
target_include_directories(${PROJECT_NAME} PUBLIC 
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/models
  ${CMAKE_CURRENT_SOURCE_DIR}/libs
  ${CMAKE_CURRENT_SOURCE_DIR}/utils
  ${JSONCPP_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
  PkgConfig::JSONCPP
)

cmake_minimum_required(VERSION 2.8.3)
project(range_reg)

find_package(catkin REQUIRED 
  COMPONENTS 
  message_generation 
  std_msgs 
  sensor_msgs
  offscreen_render
)

#generate_messages(DEPENDENCIES std_msgs sensor_msgs)

catkin_package(
  CATKIN_DEPENDS 
  message_runtime
  std_msgs
  sensor_msgs
  offscreen_render
  )



#add_executable(range_reg src/main.cpp)
#add_dependencies(message_program ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# DeepSeek V4.1 Flash external model extension.

gufo_extension_target(
  NAME gufo_deepseek41flash
  SOURCES
    extension.cpp
    probe.cpp
  LINK_LIBRARIES
    gufo_ext_runtime
)

if(TARGET gufo_deepseek41flash)
  target_include_directories(gufo_deepseek41flash PUBLIC
    "$<BUILD_INTERFACE:${GUFO_EXTENSION_BUILD_ROOT}>"
  )
endif()

install(FILES
  "${GUFO_EXTENSION_BUILD_ROOT}/LICENSE"
  "${GUFO_EXTENSION_BUILD_ROOT}/LICENSES/gufo.txt"
  DESTINATION share/licenses/gufo/extensions/deepseek41flash)

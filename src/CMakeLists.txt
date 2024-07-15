
include_directories(${FLYWAVE_INCLUDE_DIRS})

add_definitions(-DHAVE_CONFIG_H -DIPOPTLIB_BUILD)

FILE( GLOB cipopt_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.c )
FILE( GLOB cipopt_HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.h )

ADD_LIBRARY(cipopt STATIC
            ${cipopt_HEADER_FILES}
            ${cipopt_SOURCE_FILES}
          )

TARGET_INCLUDE_DIRECTORIES(cipopt PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>)

SOURCE_GROUP("src" FILES ${cipopt_SOURCE_FILES})
SOURCE_GROUP("include" FILES ${cipopt_HEADER_FILES})

SET_TARGET_PROPERTIES(cipopt
                    PROPERTIES PUBLIC_HEADER "${headers}")


ADD_EXECUTABLE(cipopt_test test_ipopt_.c)

IF(MINGW)
     TARGET_LINK_LIBRARIES(cipopt_test -Wl,--start-group stdc++ ${FLYWAVE_LIBRARY_DEPES} cipopt  -Wl,--end-group)
ELSEIF(UNIX AND NOT APPLE)
    TARGET_LINK_LIBRARIES(cipopt_test -Wl,--start-group cipopt pthread dl stdc++ m ${FLYWAVE_LIBRARY_DEPES} -Wl,--end-group)
ELSE()
    TARGET_LINK_LIBRARIES(cipopt_test cipopt ${FLYWAVE_LIBRARY_DEPES})
ENDIF()
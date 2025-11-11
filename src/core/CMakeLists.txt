add_library(core SHARED
    CTL.cpp
    dimensions_utils.cpp
    reader.cpp
)

target_include_directories(core
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

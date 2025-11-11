add_library(core SHARED
        ETL.cpp
        dimensions_utils.cpp
        context.cpp  
        mem_planner.cpp  
        reader.cpp
)

target_include_directories(core
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

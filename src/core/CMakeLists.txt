add_library(core SHARED
        ETL.cpp
        dimensions_utils.cpp
        context.cpp  
        mem_planner.cpp  
        reader.cpp
        solve_perms.cpp
)

target_link_libraries(core
    PRIVATE
        z3
)
target_include_directories(core
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

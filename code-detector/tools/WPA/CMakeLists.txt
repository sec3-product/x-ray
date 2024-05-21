add_definitions(-DDebug)

if(DEFINED IN_SOURCE_BUILD)
    set(LLVM_LINK_COMPONENTS BitWriter Core IPO IrReader InstCombine Instrumentation Target Linker Analysis ScalarOpts Support)
    add_llvm_tool( wpa wpa.cpp )
else()
    llvm_map_components_to_libnames(llvm_libs bitwriter core ipo irreader instcombine instrumentation target linker analysis scalaropts support transformutils)
    add_executable( wpa EXCLUDE_FROM_ALL wpa.cpp  )

    target_link_libraries( wpa aser ${llvm_libs} )

    set_target_properties( wpa PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin )
endif()

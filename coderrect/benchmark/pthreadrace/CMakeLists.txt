cmake_minimum_required(VERSION 3.10)
project(pthreadrace)

set(CMAKE_CXX_STANDARD 11)

find_package(Threads)

find_package(Boost REQUIRED COMPONENTS thread)
include_directories(${Boost_INCLUDE_DIR})

add_executable(sem
        sem.cpp)
target_link_libraries(sem ${CMAKE_THREAD_LIBS_INIT})


add_executable(lock_guard
        lock_guard.cpp)
target_link_libraries(lock_guard ${CMAKE_THREAD_LIBS_INIT})


add_executable(helloworld
        helloworld.cpp)
target_link_libraries(helloworld ${CMAKE_THREAD_LIBS_INIT})

add_executable(std_thread std_thread.cpp)
target_link_libraries(std_thread ${CMAKE_THREAD_LIBS_INIT})

#boost thread test cases
add_executable(boost_thread boost_thread.cpp)
target_link_libraries(boost_thread ${CMAKE_THREAD_LIBS_INIT} ${Boost_LIBRARIES})
target_link_libraries(boost_thread  ${Boost_LIBRARIES})

add_executable(boost_thread_func boost_thread_func.cpp)
target_link_libraries(boost_thread_func ${CMAKE_THREAD_LIBS_INIT} ${Boost_LIBRARIES})
target_link_libraries(boost_thread_func ${Boost_LIBRARIES})

add_executable(boost_thread_lambda boost_thread_lambda.cpp)
target_link_libraries(boost_thread_lambda ${CMAKE_THREAD_LIBS_INIT} ${Boost_LIBRARIES})
target_link_libraries(boost_thread_lambda  ${Boost_LIBRARIES})

add_executable(boost_thread_memptr boost_thread_memptr.cpp)
target_link_libraries(boost_thread_memptr ${CMAKE_THREAD_LIBS_INIT} ${Boost_LIBRARIES})
target_link_libraries(boost_thread_memptr  ${Boost_LIBRARIES})

add_executable(boost_thread_virtual_memptr boost_thread_virtual_memptr.cpp)
target_link_libraries(boost_thread_virtual_memptr ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(boost_thread_virtual_memptr  ${Boost_LIBRARIES})

add_executable(rwlock_branch rwlock_branch.cpp)
target_link_libraries(rwlock_branch ${CMAKE_THREAD_LIBS_INIT})

add_executable(rwlock_norace rwlock_norace.cpp)
target_link_libraries(rwlock_norace ${CMAKE_THREAD_LIBS_INIT})

add_executable(rwlock_race
        rwlock_race.cpp)
target_link_libraries(rwlock_race ${CMAKE_THREAD_LIBS_INIT})


add_executable(rwlock_rdlock_write_race2
        rwlock_rdlock_write_race2.cpp)
target_link_libraries(rwlock_rdlock_write_race2 ${CMAKE_THREAD_LIBS_INIT})


add_executable(deadlock
        deadlock.cpp)
target_link_libraries(deadlock ${CMAKE_THREAD_LIBS_INIT})


add_library(ringbuffer STATIC
            ringbuffer_lib.cpp
            ringbuffer_lib.h)


add_executable(ownershipxfer
        ownershipxfer.cpp
        )
target_link_libraries(ownershipxfer ${CMAKE_THREAD_LIBS_INIT})


add_executable(ownershipxfer2
        ownershipxfer2.cpp
        )
target_link_libraries(ownershipxfer2 ${CMAKE_THREAD_LIBS_INIT})


add_executable(ownershipxfer3
        ownershipxfer3.cpp
        )
target_link_libraries(ownershipxfer3 ${CMAKE_THREAD_LIBS_INIT})


add_executable(ownershipxfer4
        ownershipxfer4.cpp
        )
target_link_libraries(ownershipxfer4 ${CMAKE_THREAD_LIBS_INIT})


add_executable(signal_race
        signal_race.cpp
        )
target_link_libraries(signal_race ${CMAKE_THREAD_LIBS_INIT})


add_executable(signal_sigaction_race
        signal_sigaction_race.cpp
        )
target_link_libraries(signal_sigaction_race ${CMAKE_THREAD_LIBS_INIT})


add_executable(homemade_spinlock_norace
        homemade_spinlock_norace.cpp
        )
target_link_libraries(homemade_spinlock_norace ${CMAKE_THREAD_LIBS_INIT})


add_executable(pthread_spinlock_norace
        pthread_spinlock_norace.cpp
        )
target_link_libraries(pthread_spinlock_norace ${CMAKE_THREAD_LIBS_INIT})


add_executable(pthread_spinlock_race
        pthread_spinlock_race.cpp
        )
target_link_libraries(pthread_spinlock_race ${CMAKE_THREAD_LIBS_INIT})


add_executable(memcached_state_machine_norace
        memcached_state_machine_norace.cpp
        )
target_link_libraries(memcached_state_machine_norace ${CMAKE_THREAD_LIBS_INIT})


add_executable(memcached_complex_norace
        memcached_complex_norace.cpp
        )
target_link_libraries(memcached_complex_norace ${CMAKE_THREAD_LIBS_INIT})



add_executable(std_thread_func
        std_thread_func.cpp
        )
target_link_libraries(std_thread_func ${CMAKE_THREAD_LIBS_INIT})


add_executable(std_thread_memptr
		std_thread_memptr.cpp)
target_link_libraries(std_thread_memptr ${CMAKE_THREAD_LIBS_INIT})


add_executable(std_thread_vector_race
		std_thread_vector_race.cpp
		)
target_link_libraries(std_thread_vector_race ${CMAKE_THREAD_LIBS_INIT})

add_executable(std_vector_race  std_vector_race.cpp)
target_link_libraries(std_vector_race ${CMAKE_THREAD_LIBS_INIT})

add_executable(cpp_vtable_no_race cpp_vtable_no_race.cpp)
target_link_libraries(cpp_vtable_no_race ${CMAKE_THREAD_LIBS_INIT})



add_executable(pthread_vector_race
        pthread_vector_race.cpp
        )
target_link_libraries(pthread_vector_race ${CMAKE_THREAD_LIBS_INIT})


add_executable(std_mutex_norace
        std_mutex_norace.cpp
        )
target_link_libraries(std_mutex_norace ${CMAKE_THREAD_LIBS_INIT})


add_executable(std_atomic_norace
        std_atomic_norace.cpp
        )
target_link_libraries(std_atomic_norace ${CMAKE_THREAD_LIBS_INIT})


add_executable(std_async_race
        std_async_race.cpp
        )
target_link_libraries(std_async_race ${CMAKE_THREAD_LIBS_INIT})


add_executable(std_thread_lambda_race
        std_thread_lambda_race.cpp
        )
target_link_libraries(std_thread_lambda_race ${CMAKE_THREAD_LIBS_INIT})


add_executable(pthread_vector_jeff_race
        pthread_vector_jeff_race.cpp
        )
target_link_libraries(pthread_vector_jeff_race ${CMAKE_THREAD_LIBS_INIT})


add_executable(signal_sigaction_norace
        signal_sigaction_norace.cpp
        )
target_link_libraries(signal_sigaction_norace ${CMAKE_THREAD_LIBS_INIT})

add_executable(signal_sigaction2_norace
        signal_sigaction2_norace.cpp
        )
target_link_libraries(signal_sigaction2_norace ${CMAKE_THREAD_LIBS_INIT})

add_executable(signal_single_thread_norace
        signal_single_thread_norace.cpp
        )
target_link_libraries(signal_single_thread_norace ${CMAKE_THREAD_LIBS_INIT})


add_executable(jieyu_aget1_deadlock
        jieyu_aget1_deadlock.cpp
        )
target_link_libraries(jieyu_aget1_deadlock ${CMAKE_THREAD_LIBS_INIT})

add_executable(jieyu_httrack_race
        jieyu_httrack_race.cpp
        )
target_link_libraries(jieyu_httrack_race ${CMAKE_THREAD_LIBS_INIT})

add_executable(repeat_call_stack_main_norace
	repeat_call_stack_main_norace.cpp)
target_link_libraries(repeat_call_stack_main_norace ${CMAKE_THREAD_LIBS_INIT})

add_executable(repeat_call_stack_main_norace2
	repeat_call_stack_main_norace2.cpp)
target_link_libraries(repeat_call_stack_main_norace2 ${CMAKE_THREAD_LIBS_INIT})

add_executable(repeat_call_stack_main_race
	repeat_call_stack_main_race.cpp)
target_link_libraries(repeat_call_stack_main_race ${CMAKE_THREAD_LIBS_INIT})

add_executable(repeat_call_stack_main_race2
	repeat_call_stack_main_race2.cpp)
target_link_libraries(repeat_call_stack_main_race2 ${CMAKE_THREAD_LIBS_INIT})

add_executable(struct_field_race
	struct_field_race.cpp)
target_link_libraries(struct_field_race ${CMAKE_THREAD_LIBS_INIT})

add_executable(path_sensitivity_redis_norace
	path_sensitivity_redis_norace.cpp)
target_link_libraries(path_sensitivity_redis_norace ${CMAKE_THREAD_LIBS_INIT})

add_executable(path_sensitivity_redis_race
	path_sensitivity_redis_race.cpp)
target_link_libraries(path_sensitivity_redis_race ${CMAKE_THREAD_LIBS_INIT})

add_executable(path_sensitivity_redis2_race
	path_sensitivity_redis2_race.cpp)
target_link_libraries(path_sensitivity_redis2_race ${CMAKE_THREAD_LIBS_INIT})

add_executable(path_sensitivity_exit_norace
	path_sensitivity_exit_norace.cpp)
target_link_libraries(path_sensitivity_exit_norace ${CMAKE_THREAD_LIBS_INIT})

add_executable(path_sensitivity_exit2_race
	path_sensitivity_exit2_race.cpp)
target_link_libraries(path_sensitivity_exit2_race ${CMAKE_THREAD_LIBS_INIT})

add_executable(path_sensitivity_switch_norace
	path_sensitivity_switch_norace.cpp)
target_link_libraries(path_sensitivity_switch_norace ${CMAKE_THREAD_LIBS_INIT})

add_executable(path_sensitivity_branch1_norace
	path_sensitivity_branch1_norace.cpp)
target_link_libraries(path_sensitivity_branch1_norace ${CMAKE_THREAD_LIBS_INIT})

add_executable(path_sensitivity_branch1_race
	path_sensitivity_branch1_race.cpp)
target_link_libraries(path_sensitivity_branch1_race ${CMAKE_THREAD_LIBS_INIT})

add_executable(loop_unroll_race
	loop_unroll_race.cpp)
target_link_libraries(loop_unroll_race ${CMAKE_THREAD_LIBS_INIT})

add_executable(loop_unroll_norace
	loop_unroll_norace.cpp)
target_link_libraries(loop_unroll_norace ${CMAKE_THREAD_LIBS_INIT})

add_executable(toctou_null_deref
	toctou_null_deref.cpp)
target_link_libraries(toctou_null_deref ${CMAKE_THREAD_LIBS_INIT})

add_executable(toctou_windows_2003
	toctou_windows_2003.cpp)
target_link_libraries(toctou_windows_2003 ${CMAKE_THREAD_LIBS_INIT})

add_executable(toctou_windows_2003_synced
	toctou_windows_2003_synced.cpp)
target_link_libraries(toctou_windows_2003_synced ${CMAKE_THREAD_LIBS_INIT})

add_executable(toctou_openfile
	toctou_openfile.cpp)
target_link_libraries(toctou_openfile ${CMAKE_THREAD_LIBS_INIT})


#include <thread>

extern "C" {
    void __coderrect_stub_thread_create_no_origin(void *, void *, void *(*)(void *), void *);
}

namespace coderrect_stub {
    __attribute__((always_inline))
    static void* __coderrect_execute_thread_cb(void* __p)
    {
      std::thread::_State_ptr __t{ static_cast<std::thread::_State*>(__p) };
      __t->_M_run();
      return nullptr;
    }
}

__attribute__((always_inline))
void std::thread::_M_start_thread(std::unique_ptr<std::thread::_State, std::default_delete<std::thread::_State> > invoker, void (*)()) {
    __coderrect_stub_thread_create_no_origin(NULL, NULL, &coderrect_stub::__coderrect_execute_thread_cb, invoker.get());
}
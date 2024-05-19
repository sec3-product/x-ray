!!!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!!!
!!! Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
!!! and DataRaceBench project contributors. See the DataRaceBench/COPYRIGHT file for details.
!!!
!!! SPDX-License-Identifier: (BSD-3-Clause)
!!!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!!!

!This example is modified version of nestable_lock.1.c example, OpenMP 5.0 Application Programming Examples.
!A nested lock can be locked several times. It doesn't unlock until you have unset it as many times as the
!number of calls to omp_set_nest_lock.
!incr_b is called at line 54 and line 59. So, it needs a nest_lock for p%b@35:5. No data race.

module DRB118
    use OMP_LIB, ONLY: OMP_NEST_LOCK_KIND
    type pair
        integer a
        integer b
        integer (OMP_NEST_LOCK_KIND) lck
    end type
end module

subroutine incr_a(p, a)
    use DRB118
    type(pair) :: p
    integer a
    p%a = p%a + 1
end subroutine

subroutine incr_b(p, b)
    use omp_lib
    use DRB118
    type(pair) :: p
    integer b
    call OMP_SET_NEST_LOCK(p%lck)
    p%b = p%b + 1
    call OMP_UNSET_NEST_LOCK(p%lck)
end subroutine

program DRB118_nestlock_orig_no
    use omp_lib
    use DRB118
    implicit none

    integer :: a, b

    type(pair) :: p
    p%a = 0
    p%b = 0
    call omp_init_nest_lock(p%lck);

    !$omp parallel sections
    !$omp section
        call omp_set_nest_lock(p%lck)
        call incr_b(p, a)
        call incr_a(p, b)
        call omp_unset_nest_lock(p%lck)

    !$omp section
        call incr_b(p, b);

    !$omp end parallel sections

    call omp_destroy_nest_lock(p%lck)

    print*,p%b

end program

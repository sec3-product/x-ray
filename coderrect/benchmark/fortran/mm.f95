program mm
    use omp_lib
        implicit none
        integer :: i, j, k, N, sum
        double precision :: start, end
        integer, allocatable, dimension(:,:) :: a, b, c

        N = 1000
        allocate ( a(N,N), b(N,N), c(N,N) )

        ! fill a and b
        do i = 1, N
            a(i, i) = i
            b(N-i,i) = i
        end do

        ! do matrix multiplication on single thread
        sum = 0
        start = omp_get_wtime()
        do i = 1, N
            do j = 1, N
                do k = 1, N
                    a(i,j) = a(i,j) + b(i,k) * c(k,j)
                end do
                sum = sum + a(i,j)
            end do
        end do
        end = omp_get_wtime()

        print *, "sum",sum
        write(*, fmt="(F8.5,A)") end-start, " seconds on 1 thread"

        ! do matrix multiplication in parallel
        sum = 0
        start = omp_get_wtime()
        !$omp parallel do
        do i = 1, N
            do j = 1, N
                do k = 1, N
                    a(i,j) = a(i,j) + b(i,k) * c(k,j)
                end do
                sum = sum + a(i, j)
            end do
        end do
        !$omp end parallel do
        end = omp_get_wtime()

        print *, "sum",sum
        write(*, fmt="(F8.5,A,I3,A)") end-start, " seconds on ", omp_get_max_threads(), " threads"

    end program
    

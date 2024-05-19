; ModuleID = '/tmp/DRB135-taskdep-mutexinoutset-orig-omp50-no-c3570f.ll'
source_filename = "/tmp/DRB135-taskdep-mutexinoutset-orig-omp50-no-c3570f.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%astruct.dt58 = type <{ i8*, i8*, i8*, i8* }>

@.C312_MAIN_ = internal constant i32 25
@.C284_MAIN_ = internal constant i64 0
@.C352_MAIN_ = internal constant i32 6
@.C349_MAIN_ = internal constant [71 x i8] c"micro-benchmarks-fortran/DRB135-taskdep-mutexinoutset-orig-omp50-no.f95"
@.C351_MAIN_ = internal constant i32 41
@.C301_MAIN_ = internal constant i32 3
@.C300_MAIN_ = internal constant i32 2
@.C285_MAIN_ = internal constant i32 1
@.C283_MAIN_ = internal constant i32 0
@.C301___nv_MAIN__F1L18_1 = internal constant i32 3
@.C300___nv_MAIN__F1L18_1 = internal constant i32 2
@.C283___nv_MAIN__F1L18_1 = internal constant i32 0
@.C285___nv_MAIN__F1L18_1 = internal constant i32 1
@.C285___nv_MAIN_F1L20_2 = internal constant i32 1
@.C300___nv_MAIN_F1L23_3 = internal constant i32 2
@.C301___nv_MAIN_F1L26_4 = internal constant i32 3

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__390 = alloca i32, align 4
  %c_315 = alloca i32, align 4
  %.uplevelArgPack0001_375 = alloca %astruct.dt58, align 16
  %a_313 = alloca i32, align 4
  %b_314 = alloca i32, align 4
  %d_316 = alloca i32, align 4
  %z__io_354 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 2, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 3, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !14, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 2, metadata !15, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 3, metadata !16, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !17, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !18, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 2, metadata !19, metadata !DIExpression()), !dbg !10
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !20
  store i32 %0, i32* %__gtid_MAIN__390, align 4, !dbg !20
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !21
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !21
  call void (i8*, ...) %2(i8* %1), !dbg !21
  br label %L.LB1_370

L.LB1_370:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %c_315, metadata !22, metadata !DIExpression()), !dbg !10
  %3 = bitcast i32* %c_315 to i8*, !dbg !23
  %4 = bitcast %astruct.dt58* %.uplevelArgPack0001_375 to i8**, !dbg !23
  store i8* %3, i8** %4, align 8, !dbg !23
  call void @llvm.dbg.declare(metadata i32* %a_313, metadata !24, metadata !DIExpression()), !dbg !10
  %5 = bitcast i32* %a_313 to i8*, !dbg !23
  %6 = bitcast %astruct.dt58* %.uplevelArgPack0001_375 to i8*, !dbg !23
  %7 = getelementptr i8, i8* %6, i64 8, !dbg !23
  %8 = bitcast i8* %7 to i8**, !dbg !23
  store i8* %5, i8** %8, align 8, !dbg !23
  call void @llvm.dbg.declare(metadata i32* %b_314, metadata !25, metadata !DIExpression()), !dbg !10
  %9 = bitcast i32* %b_314 to i8*, !dbg !23
  %10 = bitcast %astruct.dt58* %.uplevelArgPack0001_375 to i8*, !dbg !23
  %11 = getelementptr i8, i8* %10, i64 16, !dbg !23
  %12 = bitcast i8* %11 to i8**, !dbg !23
  store i8* %9, i8** %12, align 8, !dbg !23
  call void @llvm.dbg.declare(metadata i32* %d_316, metadata !26, metadata !DIExpression()), !dbg !10
  %13 = bitcast i32* %d_316 to i8*, !dbg !23
  %14 = bitcast %astruct.dt58* %.uplevelArgPack0001_375 to i8*, !dbg !23
  %15 = getelementptr i8, i8* %14, i64 24, !dbg !23
  %16 = bitcast i8* %15 to i8**, !dbg !23
  store i8* %13, i8** %16, align 8, !dbg !23
  br label %L.LB1_388, !dbg !23

L.LB1_388:                                        ; preds = %L.LB1_370
  %17 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L18_1_ to i64*, !dbg !23
  %18 = bitcast %astruct.dt58* %.uplevelArgPack0001_375 to i64*, !dbg !23
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %17, i64* %18), !dbg !23
  call void (...) @_mp_bcs_nest(), !dbg !27
  %19 = bitcast i32* @.C351_MAIN_ to i8*, !dbg !27
  %20 = bitcast [71 x i8]* @.C349_MAIN_ to i8*, !dbg !27
  %21 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !27
  call void (i8*, i8*, i64, ...) %21(i8* %19, i8* %20, i64 71), !dbg !27
  %22 = bitcast i32* @.C352_MAIN_ to i8*, !dbg !27
  %23 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !27
  %24 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !27
  %25 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !27
  %26 = call i32 (i8*, i8*, i8*, i8*, ...) %25(i8* %22, i8* null, i8* %23, i8* %24), !dbg !27
  call void @llvm.dbg.declare(metadata i32* %z__io_354, metadata !28, metadata !DIExpression()), !dbg !10
  store i32 %26, i32* %z__io_354, align 4, !dbg !27
  %27 = load i32, i32* %d_316, align 4, !dbg !27
  call void @llvm.dbg.value(metadata i32 %27, metadata !26, metadata !DIExpression()), !dbg !10
  %28 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !27
  %29 = call i32 (i32, i32, ...) %28(i32 %27, i32 25), !dbg !27
  store i32 %29, i32* %z__io_354, align 4, !dbg !27
  %30 = call i32 (...) @f90io_ldw_end(), !dbg !27
  store i32 %30, i32* %z__io_354, align 4, !dbg !27
  call void (...) @_mp_ecs_nest(), !dbg !27
  ret void, !dbg !20
}

define internal void @__nv_MAIN__F1L18_1_(i32* %__nv_MAIN__F1L18_1Arg0, i64* %__nv_MAIN__F1L18_1Arg1, i64* %__nv_MAIN__F1L18_1Arg2) #0 !dbg !29 {
L.entry:
  %__gtid___nv_MAIN__F1L18_1__423 = alloca i32, align 4
  %.s0000_418 = alloca i32, align 4
  %.s0001_419 = alloca i32, align 4
  %.s0002_429 = alloca i32, align 4
  %.z0371_428 = alloca i8*, align 8
  %.s0003_453 = alloca i32, align 4
  %.z0371_452 = alloca i8*, align 8
  %.s0004_463 = alloca i32, align 4
  %.z0371_462 = alloca i8*, align 8
  %.s0005_473 = alloca i32, align 4
  %.z0371_472 = alloca i8*, align 8
  %.s0006_483 = alloca i32, align 4
  %.z0371_482 = alloca i8*, align 8
  %.s0007_493 = alloca i32, align 4
  %.z0371_492 = alloca i8*, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L18_1Arg0, metadata !33, metadata !DIExpression()), !dbg !34
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L18_1Arg1, metadata !35, metadata !DIExpression()), !dbg !34
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L18_1Arg2, metadata !36, metadata !DIExpression()), !dbg !34
  call void @llvm.dbg.value(metadata i32 1, metadata !37, metadata !DIExpression()), !dbg !34
  call void @llvm.dbg.value(metadata i32 2, metadata !38, metadata !DIExpression()), !dbg !34
  call void @llvm.dbg.value(metadata i32 3, metadata !39, metadata !DIExpression()), !dbg !34
  call void @llvm.dbg.value(metadata i32 0, metadata !40, metadata !DIExpression()), !dbg !34
  call void @llvm.dbg.value(metadata i32 1, metadata !41, metadata !DIExpression()), !dbg !34
  call void @llvm.dbg.value(metadata i32 2, metadata !42, metadata !DIExpression()), !dbg !34
  call void @llvm.dbg.value(metadata i32 3, metadata !43, metadata !DIExpression()), !dbg !34
  call void @llvm.dbg.value(metadata i32 0, metadata !44, metadata !DIExpression()), !dbg !34
  call void @llvm.dbg.value(metadata i32 1, metadata !45, metadata !DIExpression()), !dbg !34
  call void @llvm.dbg.value(metadata i32 2, metadata !46, metadata !DIExpression()), !dbg !34
  %0 = load i32, i32* %__nv_MAIN__F1L18_1Arg0, align 4, !dbg !47
  store i32 %0, i32* %__gtid___nv_MAIN__F1L18_1__423, align 4, !dbg !47
  br label %L.LB2_417

L.LB2_417:                                        ; preds = %L.entry
  br label %L.LB2_319

L.LB2_319:                                        ; preds = %L.LB2_417
  store i32 -1, i32* %.s0000_418, align 4, !dbg !48
  store i32 0, i32* %.s0001_419, align 4, !dbg !48
  %1 = load i32, i32* %__gtid___nv_MAIN__F1L18_1__423, align 4, !dbg !48
  %2 = call i32 @__kmpc_single(i64* null, i32 %1), !dbg !48
  %3 = icmp eq i32 %2, 0, !dbg !48
  br i1 %3, label %L.LB2_362, label %L.LB2_321, !dbg !48

L.LB2_321:                                        ; preds = %L.LB2_319
  store i32 1, i32* %.s0002_429, align 4, !dbg !49
  %4 = load i32, i32* %__gtid___nv_MAIN__F1L18_1__423, align 4, !dbg !50
  %5 = load i32, i32* %.s0002_429, align 4, !dbg !50
  %6 = bitcast void (i32, i64*)* @__nv_MAIN_F1L20_2_ to i64*, !dbg !50
  %7 = call i8* @__kmpc_omp_task_alloc(i64* null, i32 %4, i32 %5, i32 40, i32 32, i64* %6), !dbg !50
  store i8* %7, i8** %.z0371_428, align 8, !dbg !50
  %8 = load i64, i64* %__nv_MAIN__F1L18_1Arg2, align 8, !dbg !50
  %9 = load i8*, i8** %.z0371_428, align 8, !dbg !50
  %10 = bitcast i8* %9 to i64**, !dbg !50
  %11 = load i64*, i64** %10, align 8, !dbg !50
  store i64 %8, i64* %11, align 8, !dbg !50
  %12 = bitcast i64* %__nv_MAIN__F1L18_1Arg2 to i8*, !dbg !47
  %13 = getelementptr i8, i8* %12, i64 8, !dbg !47
  %14 = bitcast i8* %13 to i64*, !dbg !47
  %15 = load i64, i64* %14, align 8, !dbg !47
  %16 = bitcast i64* %11 to i8*, !dbg !47
  %17 = getelementptr i8, i8* %16, i64 8, !dbg !47
  %18 = bitcast i8* %17 to i64*, !dbg !47
  store i64 %15, i64* %18, align 8, !dbg !47
  %19 = bitcast i64* %__nv_MAIN__F1L18_1Arg2 to i8*, !dbg !47
  %20 = getelementptr i8, i8* %19, i64 16, !dbg !47
  %21 = bitcast i8* %20 to i64*, !dbg !47
  %22 = load i64, i64* %21, align 8, !dbg !47
  %23 = bitcast i64* %11 to i8*, !dbg !47
  %24 = getelementptr i8, i8* %23, i64 16, !dbg !47
  %25 = bitcast i8* %24 to i64*, !dbg !47
  store i64 %22, i64* %25, align 8, !dbg !47
  %26 = bitcast i64* %__nv_MAIN__F1L18_1Arg2 to i8*, !dbg !47
  %27 = getelementptr i8, i8* %26, i64 24, !dbg !47
  %28 = bitcast i8* %27 to i64*, !dbg !47
  %29 = load i64, i64* %28, align 8, !dbg !47
  %30 = bitcast i64* %11 to i8*, !dbg !47
  %31 = getelementptr i8, i8* %30, i64 24, !dbg !47
  %32 = bitcast i8* %31 to i64*, !dbg !47
  store i64 %29, i64* %32, align 8, !dbg !47
  %33 = load i32, i32* %__gtid___nv_MAIN__F1L18_1__423, align 4, !dbg !50
  %34 = load i8*, i8** %.z0371_428, align 8, !dbg !50
  %35 = bitcast i8* %34 to i64*, !dbg !50
  call void @__kmpc_omp_task(i64* null, i32 %33, i64* %35), !dbg !50
  br label %L.LB2_363

L.LB2_363:                                        ; preds = %L.LB2_321
  store i32 1, i32* %.s0003_453, align 4, !dbg !51
  %36 = load i32, i32* %__gtid___nv_MAIN__F1L18_1__423, align 4, !dbg !52
  %37 = load i32, i32* %.s0003_453, align 4, !dbg !52
  %38 = bitcast void (i32, i64*)* @__nv_MAIN_F1L23_3_ to i64*, !dbg !52
  %39 = call i8* @__kmpc_omp_task_alloc(i64* null, i32 %36, i32 %37, i32 40, i32 32, i64* %38), !dbg !52
  store i8* %39, i8** %.z0371_452, align 8, !dbg !52
  %40 = load i64, i64* %__nv_MAIN__F1L18_1Arg2, align 8, !dbg !52
  %41 = load i8*, i8** %.z0371_452, align 8, !dbg !52
  %42 = bitcast i8* %41 to i64**, !dbg !52
  %43 = load i64*, i64** %42, align 8, !dbg !52
  store i64 %40, i64* %43, align 8, !dbg !52
  %44 = bitcast i64* %__nv_MAIN__F1L18_1Arg2 to i8*, !dbg !47
  %45 = getelementptr i8, i8* %44, i64 8, !dbg !47
  %46 = bitcast i8* %45 to i64*, !dbg !47
  %47 = load i64, i64* %46, align 8, !dbg !47
  %48 = bitcast i64* %43 to i8*, !dbg !47
  %49 = getelementptr i8, i8* %48, i64 8, !dbg !47
  %50 = bitcast i8* %49 to i64*, !dbg !47
  store i64 %47, i64* %50, align 8, !dbg !47
  %51 = bitcast i64* %__nv_MAIN__F1L18_1Arg2 to i8*, !dbg !47
  %52 = getelementptr i8, i8* %51, i64 16, !dbg !47
  %53 = bitcast i8* %52 to i64*, !dbg !47
  %54 = load i64, i64* %53, align 8, !dbg !47
  %55 = bitcast i64* %43 to i8*, !dbg !47
  %56 = getelementptr i8, i8* %55, i64 16, !dbg !47
  %57 = bitcast i8* %56 to i64*, !dbg !47
  store i64 %54, i64* %57, align 8, !dbg !47
  %58 = bitcast i64* %__nv_MAIN__F1L18_1Arg2 to i8*, !dbg !47
  %59 = getelementptr i8, i8* %58, i64 24, !dbg !47
  %60 = bitcast i8* %59 to i64*, !dbg !47
  %61 = load i64, i64* %60, align 8, !dbg !47
  %62 = bitcast i64* %43 to i8*, !dbg !47
  %63 = getelementptr i8, i8* %62, i64 24, !dbg !47
  %64 = bitcast i8* %63 to i64*, !dbg !47
  store i64 %61, i64* %64, align 8, !dbg !47
  %65 = load i32, i32* %__gtid___nv_MAIN__F1L18_1__423, align 4, !dbg !52
  %66 = load i8*, i8** %.z0371_452, align 8, !dbg !52
  %67 = bitcast i8* %66 to i64*, !dbg !52
  call void @__kmpc_omp_task(i64* null, i32 %65, i64* %67), !dbg !52
  br label %L.LB2_364

L.LB2_364:                                        ; preds = %L.LB2_363
  store i32 1, i32* %.s0004_463, align 4, !dbg !53
  %68 = load i32, i32* %__gtid___nv_MAIN__F1L18_1__423, align 4, !dbg !54
  %69 = load i32, i32* %.s0004_463, align 4, !dbg !54
  %70 = bitcast void (i32, i64*)* @__nv_MAIN_F1L26_4_ to i64*, !dbg !54
  %71 = call i8* @__kmpc_omp_task_alloc(i64* null, i32 %68, i32 %69, i32 40, i32 32, i64* %70), !dbg !54
  store i8* %71, i8** %.z0371_462, align 8, !dbg !54
  %72 = load i64, i64* %__nv_MAIN__F1L18_1Arg2, align 8, !dbg !54
  %73 = load i8*, i8** %.z0371_462, align 8, !dbg !54
  %74 = bitcast i8* %73 to i64**, !dbg !54
  %75 = load i64*, i64** %74, align 8, !dbg !54
  store i64 %72, i64* %75, align 8, !dbg !54
  %76 = bitcast i64* %__nv_MAIN__F1L18_1Arg2 to i8*, !dbg !47
  %77 = getelementptr i8, i8* %76, i64 8, !dbg !47
  %78 = bitcast i8* %77 to i64*, !dbg !47
  %79 = load i64, i64* %78, align 8, !dbg !47
  %80 = bitcast i64* %75 to i8*, !dbg !47
  %81 = getelementptr i8, i8* %80, i64 8, !dbg !47
  %82 = bitcast i8* %81 to i64*, !dbg !47
  store i64 %79, i64* %82, align 8, !dbg !47
  %83 = bitcast i64* %__nv_MAIN__F1L18_1Arg2 to i8*, !dbg !47
  %84 = getelementptr i8, i8* %83, i64 16, !dbg !47
  %85 = bitcast i8* %84 to i64*, !dbg !47
  %86 = load i64, i64* %85, align 8, !dbg !47
  %87 = bitcast i64* %75 to i8*, !dbg !47
  %88 = getelementptr i8, i8* %87, i64 16, !dbg !47
  %89 = bitcast i8* %88 to i64*, !dbg !47
  store i64 %86, i64* %89, align 8, !dbg !47
  %90 = bitcast i64* %__nv_MAIN__F1L18_1Arg2 to i8*, !dbg !47
  %91 = getelementptr i8, i8* %90, i64 24, !dbg !47
  %92 = bitcast i8* %91 to i64*, !dbg !47
  %93 = load i64, i64* %92, align 8, !dbg !47
  %94 = bitcast i64* %75 to i8*, !dbg !47
  %95 = getelementptr i8, i8* %94, i64 24, !dbg !47
  %96 = bitcast i8* %95 to i64*, !dbg !47
  store i64 %93, i64* %96, align 8, !dbg !47
  %97 = load i32, i32* %__gtid___nv_MAIN__F1L18_1__423, align 4, !dbg !54
  %98 = load i8*, i8** %.z0371_462, align 8, !dbg !54
  %99 = bitcast i8* %98 to i64*, !dbg !54
  call void @__kmpc_omp_task(i64* null, i32 %97, i64* %99), !dbg !54
  br label %L.LB2_365

L.LB2_365:                                        ; preds = %L.LB2_364
  store i32 1, i32* %.s0005_473, align 4, !dbg !55
  %100 = load i32, i32* %__gtid___nv_MAIN__F1L18_1__423, align 4, !dbg !56
  %101 = load i32, i32* %.s0005_473, align 4, !dbg !56
  %102 = bitcast void (i32, i64*)* @__nv_MAIN_F1L29_5_ to i64*, !dbg !56
  %103 = call i8* @__kmpc_omp_task_alloc(i64* null, i32 %100, i32 %101, i32 40, i32 32, i64* %102), !dbg !56
  store i8* %103, i8** %.z0371_472, align 8, !dbg !56
  %104 = load i64, i64* %__nv_MAIN__F1L18_1Arg2, align 8, !dbg !56
  %105 = load i8*, i8** %.z0371_472, align 8, !dbg !56
  %106 = bitcast i8* %105 to i64**, !dbg !56
  %107 = load i64*, i64** %106, align 8, !dbg !56
  store i64 %104, i64* %107, align 8, !dbg !56
  %108 = bitcast i64* %__nv_MAIN__F1L18_1Arg2 to i8*, !dbg !47
  %109 = getelementptr i8, i8* %108, i64 8, !dbg !47
  %110 = bitcast i8* %109 to i64*, !dbg !47
  %111 = load i64, i64* %110, align 8, !dbg !47
  %112 = bitcast i64* %107 to i8*, !dbg !47
  %113 = getelementptr i8, i8* %112, i64 8, !dbg !47
  %114 = bitcast i8* %113 to i64*, !dbg !47
  store i64 %111, i64* %114, align 8, !dbg !47
  %115 = bitcast i64* %__nv_MAIN__F1L18_1Arg2 to i8*, !dbg !47
  %116 = getelementptr i8, i8* %115, i64 16, !dbg !47
  %117 = bitcast i8* %116 to i64*, !dbg !47
  %118 = load i64, i64* %117, align 8, !dbg !47
  %119 = bitcast i64* %107 to i8*, !dbg !47
  %120 = getelementptr i8, i8* %119, i64 16, !dbg !47
  %121 = bitcast i8* %120 to i64*, !dbg !47
  store i64 %118, i64* %121, align 8, !dbg !47
  %122 = bitcast i64* %__nv_MAIN__F1L18_1Arg2 to i8*, !dbg !47
  %123 = getelementptr i8, i8* %122, i64 24, !dbg !47
  %124 = bitcast i8* %123 to i64*, !dbg !47
  %125 = load i64, i64* %124, align 8, !dbg !47
  %126 = bitcast i64* %107 to i8*, !dbg !47
  %127 = getelementptr i8, i8* %126, i64 24, !dbg !47
  %128 = bitcast i8* %127 to i64*, !dbg !47
  store i64 %125, i64* %128, align 8, !dbg !47
  %129 = load i32, i32* %__gtid___nv_MAIN__F1L18_1__423, align 4, !dbg !56
  %130 = load i8*, i8** %.z0371_472, align 8, !dbg !56
  %131 = bitcast i8* %130 to i64*, !dbg !56
  call void @__kmpc_omp_task(i64* null, i32 %129, i64* %131), !dbg !56
  br label %L.LB2_366

L.LB2_366:                                        ; preds = %L.LB2_365
  store i32 1, i32* %.s0006_483, align 4, !dbg !57
  %132 = load i32, i32* %__gtid___nv_MAIN__F1L18_1__423, align 4, !dbg !58
  %133 = load i32, i32* %.s0006_483, align 4, !dbg !58
  %134 = bitcast void (i32, i64*)* @__nv_MAIN_F1L32_6_ to i64*, !dbg !58
  %135 = call i8* @__kmpc_omp_task_alloc(i64* null, i32 %132, i32 %133, i32 40, i32 32, i64* %134), !dbg !58
  store i8* %135, i8** %.z0371_482, align 8, !dbg !58
  %136 = load i64, i64* %__nv_MAIN__F1L18_1Arg2, align 8, !dbg !58
  %137 = load i8*, i8** %.z0371_482, align 8, !dbg !58
  %138 = bitcast i8* %137 to i64**, !dbg !58
  %139 = load i64*, i64** %138, align 8, !dbg !58
  store i64 %136, i64* %139, align 8, !dbg !58
  %140 = bitcast i64* %__nv_MAIN__F1L18_1Arg2 to i8*, !dbg !47
  %141 = getelementptr i8, i8* %140, i64 8, !dbg !47
  %142 = bitcast i8* %141 to i64*, !dbg !47
  %143 = load i64, i64* %142, align 8, !dbg !47
  %144 = bitcast i64* %139 to i8*, !dbg !47
  %145 = getelementptr i8, i8* %144, i64 8, !dbg !47
  %146 = bitcast i8* %145 to i64*, !dbg !47
  store i64 %143, i64* %146, align 8, !dbg !47
  %147 = bitcast i64* %__nv_MAIN__F1L18_1Arg2 to i8*, !dbg !47
  %148 = getelementptr i8, i8* %147, i64 16, !dbg !47
  %149 = bitcast i8* %148 to i64*, !dbg !47
  %150 = load i64, i64* %149, align 8, !dbg !47
  %151 = bitcast i64* %139 to i8*, !dbg !47
  %152 = getelementptr i8, i8* %151, i64 16, !dbg !47
  %153 = bitcast i8* %152 to i64*, !dbg !47
  store i64 %150, i64* %153, align 8, !dbg !47
  %154 = bitcast i64* %__nv_MAIN__F1L18_1Arg2 to i8*, !dbg !47
  %155 = getelementptr i8, i8* %154, i64 24, !dbg !47
  %156 = bitcast i8* %155 to i64*, !dbg !47
  %157 = load i64, i64* %156, align 8, !dbg !47
  %158 = bitcast i64* %139 to i8*, !dbg !47
  %159 = getelementptr i8, i8* %158, i64 24, !dbg !47
  %160 = bitcast i8* %159 to i64*, !dbg !47
  store i64 %157, i64* %160, align 8, !dbg !47
  %161 = load i32, i32* %__gtid___nv_MAIN__F1L18_1__423, align 4, !dbg !58
  %162 = load i8*, i8** %.z0371_482, align 8, !dbg !58
  %163 = bitcast i8* %162 to i64*, !dbg !58
  call void @__kmpc_omp_task(i64* null, i32 %161, i64* %163), !dbg !58
  br label %L.LB2_367

L.LB2_367:                                        ; preds = %L.LB2_366
  store i32 1, i32* %.s0007_493, align 4, !dbg !59
  %164 = load i32, i32* %__gtid___nv_MAIN__F1L18_1__423, align 4, !dbg !60
  %165 = load i32, i32* %.s0007_493, align 4, !dbg !60
  %166 = bitcast void (i32, i64*)* @__nv_MAIN_F1L35_7_ to i64*, !dbg !60
  %167 = call i8* @__kmpc_omp_task_alloc(i64* null, i32 %164, i32 %165, i32 40, i32 32, i64* %166), !dbg !60
  store i8* %167, i8** %.z0371_492, align 8, !dbg !60
  %168 = load i64, i64* %__nv_MAIN__F1L18_1Arg2, align 8, !dbg !60
  %169 = load i8*, i8** %.z0371_492, align 8, !dbg !60
  %170 = bitcast i8* %169 to i64**, !dbg !60
  %171 = load i64*, i64** %170, align 8, !dbg !60
  store i64 %168, i64* %171, align 8, !dbg !60
  %172 = bitcast i64* %__nv_MAIN__F1L18_1Arg2 to i8*, !dbg !47
  %173 = getelementptr i8, i8* %172, i64 8, !dbg !47
  %174 = bitcast i8* %173 to i64*, !dbg !47
  %175 = load i64, i64* %174, align 8, !dbg !47
  %176 = bitcast i64* %171 to i8*, !dbg !47
  %177 = getelementptr i8, i8* %176, i64 8, !dbg !47
  %178 = bitcast i8* %177 to i64*, !dbg !47
  store i64 %175, i64* %178, align 8, !dbg !47
  %179 = bitcast i64* %__nv_MAIN__F1L18_1Arg2 to i8*, !dbg !47
  %180 = getelementptr i8, i8* %179, i64 16, !dbg !47
  %181 = bitcast i8* %180 to i64*, !dbg !47
  %182 = load i64, i64* %181, align 8, !dbg !47
  %183 = bitcast i64* %171 to i8*, !dbg !47
  %184 = getelementptr i8, i8* %183, i64 16, !dbg !47
  %185 = bitcast i8* %184 to i64*, !dbg !47
  store i64 %182, i64* %185, align 8, !dbg !47
  %186 = bitcast i64* %__nv_MAIN__F1L18_1Arg2 to i8*, !dbg !47
  %187 = getelementptr i8, i8* %186, i64 24, !dbg !47
  %188 = bitcast i8* %187 to i64*, !dbg !47
  %189 = load i64, i64* %188, align 8, !dbg !47
  %190 = bitcast i64* %171 to i8*, !dbg !47
  %191 = getelementptr i8, i8* %190, i64 24, !dbg !47
  %192 = bitcast i8* %191 to i64*, !dbg !47
  store i64 %189, i64* %192, align 8, !dbg !47
  %193 = load i32, i32* %__gtid___nv_MAIN__F1L18_1__423, align 4, !dbg !60
  %194 = load i8*, i8** %.z0371_492, align 8, !dbg !60
  %195 = bitcast i8* %194 to i64*, !dbg !60
  call void @__kmpc_omp_task(i64* null, i32 %193, i64* %195), !dbg !60
  br label %L.LB2_368

L.LB2_368:                                        ; preds = %L.LB2_367
  %196 = load i32, i32* %__gtid___nv_MAIN__F1L18_1__423, align 4, !dbg !61
  store i32 %196, i32* %.s0000_418, align 4, !dbg !61
  store i32 1, i32* %.s0001_419, align 4, !dbg !61
  %197 = load i32, i32* %__gtid___nv_MAIN__F1L18_1__423, align 4, !dbg !61
  call void @__kmpc_end_single(i64* null, i32 %197), !dbg !61
  br label %L.LB2_362

L.LB2_362:                                        ; preds = %L.LB2_368, %L.LB2_319
  br label %L.LB2_346

L.LB2_346:                                        ; preds = %L.LB2_362
  %198 = load i32, i32* %__gtid___nv_MAIN__F1L18_1__423, align 4, !dbg !61
  call void @__kmpc_barrier(i64* null, i32 %198), !dbg !61
  br label %L.LB2_347

L.LB2_347:                                        ; preds = %L.LB2_346
  ret void, !dbg !47
}

define internal void @__nv_MAIN_F1L20_2_(i32 %__nv_MAIN_F1L20_2Arg0.arg, i64* %__nv_MAIN_F1L20_2Arg1) #0 !dbg !62 {
L.entry:
  %__nv_MAIN_F1L20_2Arg0.addr = alloca i32, align 4
  %.S0000_518 = alloca i8*, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L20_2Arg0.addr, metadata !65, metadata !DIExpression()), !dbg !66
  store i32 %__nv_MAIN_F1L20_2Arg0.arg, i32* %__nv_MAIN_F1L20_2Arg0.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L20_2Arg0.addr, metadata !67, metadata !DIExpression()), !dbg !66
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L20_2Arg1, metadata !68, metadata !DIExpression()), !dbg !66
  call void @llvm.dbg.value(metadata i32 1, metadata !69, metadata !DIExpression()), !dbg !66
  call void @llvm.dbg.value(metadata i32 2, metadata !70, metadata !DIExpression()), !dbg !66
  call void @llvm.dbg.value(metadata i32 3, metadata !71, metadata !DIExpression()), !dbg !66
  call void @llvm.dbg.value(metadata i32 0, metadata !72, metadata !DIExpression()), !dbg !66
  call void @llvm.dbg.value(metadata i32 1, metadata !73, metadata !DIExpression()), !dbg !66
  call void @llvm.dbg.value(metadata i32 2, metadata !74, metadata !DIExpression()), !dbg !66
  call void @llvm.dbg.value(metadata i32 3, metadata !75, metadata !DIExpression()), !dbg !66
  call void @llvm.dbg.value(metadata i32 0, metadata !76, metadata !DIExpression()), !dbg !66
  call void @llvm.dbg.value(metadata i32 1, metadata !77, metadata !DIExpression()), !dbg !66
  call void @llvm.dbg.value(metadata i32 2, metadata !78, metadata !DIExpression()), !dbg !66
  %0 = bitcast i64* %__nv_MAIN_F1L20_2Arg1 to i8**, !dbg !79
  %1 = load i8*, i8** %0, align 8, !dbg !79
  store i8* %1, i8** %.S0000_518, align 8, !dbg !79
  br label %L.LB4_522

L.LB4_522:                                        ; preds = %L.entry
  br label %L.LB4_324

L.LB4_324:                                        ; preds = %L.LB4_522
  %2 = load i8*, i8** %.S0000_518, align 8, !dbg !80
  %3 = bitcast i8* %2 to i32**, !dbg !80
  %4 = load i32*, i32** %3, align 8, !dbg !80
  store i32 1, i32* %4, align 4, !dbg !80
  br label %L.LB4_325

L.LB4_325:                                        ; preds = %L.LB4_324
  ret void, !dbg !81
}

define internal void @__nv_MAIN_F1L23_3_(i32 %__nv_MAIN_F1L23_3Arg0.arg, i64* %__nv_MAIN_F1L23_3Arg1) #0 !dbg !82 {
L.entry:
  %__nv_MAIN_F1L23_3Arg0.addr = alloca i32, align 4
  %.S0000_518 = alloca i8*, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L23_3Arg0.addr, metadata !83, metadata !DIExpression()), !dbg !84
  store i32 %__nv_MAIN_F1L23_3Arg0.arg, i32* %__nv_MAIN_F1L23_3Arg0.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L23_3Arg0.addr, metadata !85, metadata !DIExpression()), !dbg !84
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L23_3Arg1, metadata !86, metadata !DIExpression()), !dbg !84
  call void @llvm.dbg.value(metadata i32 1, metadata !87, metadata !DIExpression()), !dbg !84
  call void @llvm.dbg.value(metadata i32 2, metadata !88, metadata !DIExpression()), !dbg !84
  call void @llvm.dbg.value(metadata i32 3, metadata !89, metadata !DIExpression()), !dbg !84
  call void @llvm.dbg.value(metadata i32 0, metadata !90, metadata !DIExpression()), !dbg !84
  call void @llvm.dbg.value(metadata i32 1, metadata !91, metadata !DIExpression()), !dbg !84
  call void @llvm.dbg.value(metadata i32 2, metadata !92, metadata !DIExpression()), !dbg !84
  call void @llvm.dbg.value(metadata i32 3, metadata !93, metadata !DIExpression()), !dbg !84
  call void @llvm.dbg.value(metadata i32 0, metadata !94, metadata !DIExpression()), !dbg !84
  call void @llvm.dbg.value(metadata i32 1, metadata !95, metadata !DIExpression()), !dbg !84
  call void @llvm.dbg.value(metadata i32 2, metadata !96, metadata !DIExpression()), !dbg !84
  %0 = bitcast i64* %__nv_MAIN_F1L23_3Arg1 to i8**, !dbg !97
  %1 = load i8*, i8** %0, align 8, !dbg !97
  store i8* %1, i8** %.S0000_518, align 8, !dbg !97
  br label %L.LB5_528

L.LB5_528:                                        ; preds = %L.entry
  br label %L.LB5_328

L.LB5_328:                                        ; preds = %L.LB5_528
  %2 = load i8*, i8** %.S0000_518, align 8, !dbg !98
  %3 = getelementptr i8, i8* %2, i64 8, !dbg !98
  %4 = bitcast i8* %3 to i32**, !dbg !98
  %5 = load i32*, i32** %4, align 8, !dbg !98
  store i32 2, i32* %5, align 4, !dbg !98
  br label %L.LB5_329

L.LB5_329:                                        ; preds = %L.LB5_328
  ret void, !dbg !99
}

define internal void @__nv_MAIN_F1L26_4_(i32 %__nv_MAIN_F1L26_4Arg0.arg, i64* %__nv_MAIN_F1L26_4Arg1) #0 !dbg !100 {
L.entry:
  %__nv_MAIN_F1L26_4Arg0.addr = alloca i32, align 4
  %.S0000_518 = alloca i8*, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L26_4Arg0.addr, metadata !101, metadata !DIExpression()), !dbg !102
  store i32 %__nv_MAIN_F1L26_4Arg0.arg, i32* %__nv_MAIN_F1L26_4Arg0.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L26_4Arg0.addr, metadata !103, metadata !DIExpression()), !dbg !102
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L26_4Arg1, metadata !104, metadata !DIExpression()), !dbg !102
  call void @llvm.dbg.value(metadata i32 1, metadata !105, metadata !DIExpression()), !dbg !102
  call void @llvm.dbg.value(metadata i32 2, metadata !106, metadata !DIExpression()), !dbg !102
  call void @llvm.dbg.value(metadata i32 3, metadata !107, metadata !DIExpression()), !dbg !102
  call void @llvm.dbg.value(metadata i32 0, metadata !108, metadata !DIExpression()), !dbg !102
  call void @llvm.dbg.value(metadata i32 1, metadata !109, metadata !DIExpression()), !dbg !102
  call void @llvm.dbg.value(metadata i32 2, metadata !110, metadata !DIExpression()), !dbg !102
  call void @llvm.dbg.value(metadata i32 3, metadata !111, metadata !DIExpression()), !dbg !102
  call void @llvm.dbg.value(metadata i32 0, metadata !112, metadata !DIExpression()), !dbg !102
  call void @llvm.dbg.value(metadata i32 1, metadata !113, metadata !DIExpression()), !dbg !102
  call void @llvm.dbg.value(metadata i32 2, metadata !114, metadata !DIExpression()), !dbg !102
  %0 = bitcast i64* %__nv_MAIN_F1L26_4Arg1 to i8**, !dbg !115
  %1 = load i8*, i8** %0, align 8, !dbg !115
  store i8* %1, i8** %.S0000_518, align 8, !dbg !115
  br label %L.LB6_534

L.LB6_534:                                        ; preds = %L.entry
  br label %L.LB6_332

L.LB6_332:                                        ; preds = %L.LB6_534
  %2 = load i8*, i8** %.S0000_518, align 8, !dbg !116
  %3 = getelementptr i8, i8* %2, i64 16, !dbg !116
  %4 = bitcast i8* %3 to i32**, !dbg !116
  %5 = load i32*, i32** %4, align 8, !dbg !116
  store i32 3, i32* %5, align 4, !dbg !116
  br label %L.LB6_333

L.LB6_333:                                        ; preds = %L.LB6_332
  ret void, !dbg !117
}

define internal void @__nv_MAIN_F1L29_5_(i32 %__nv_MAIN_F1L29_5Arg0.arg, i64* %__nv_MAIN_F1L29_5Arg1) #0 !dbg !118 {
L.entry:
  %__nv_MAIN_F1L29_5Arg0.addr = alloca i32, align 4
  %.S0000_518 = alloca i8*, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L29_5Arg0.addr, metadata !119, metadata !DIExpression()), !dbg !120
  store i32 %__nv_MAIN_F1L29_5Arg0.arg, i32* %__nv_MAIN_F1L29_5Arg0.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L29_5Arg0.addr, metadata !121, metadata !DIExpression()), !dbg !120
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L29_5Arg1, metadata !122, metadata !DIExpression()), !dbg !120
  call void @llvm.dbg.value(metadata i32 1, metadata !123, metadata !DIExpression()), !dbg !120
  call void @llvm.dbg.value(metadata i32 2, metadata !124, metadata !DIExpression()), !dbg !120
  call void @llvm.dbg.value(metadata i32 3, metadata !125, metadata !DIExpression()), !dbg !120
  call void @llvm.dbg.value(metadata i32 0, metadata !126, metadata !DIExpression()), !dbg !120
  call void @llvm.dbg.value(metadata i32 1, metadata !127, metadata !DIExpression()), !dbg !120
  call void @llvm.dbg.value(metadata i32 2, metadata !128, metadata !DIExpression()), !dbg !120
  call void @llvm.dbg.value(metadata i32 3, metadata !129, metadata !DIExpression()), !dbg !120
  call void @llvm.dbg.value(metadata i32 0, metadata !130, metadata !DIExpression()), !dbg !120
  call void @llvm.dbg.value(metadata i32 1, metadata !131, metadata !DIExpression()), !dbg !120
  call void @llvm.dbg.value(metadata i32 2, metadata !132, metadata !DIExpression()), !dbg !120
  %0 = bitcast i64* %__nv_MAIN_F1L29_5Arg1 to i8**, !dbg !133
  %1 = load i8*, i8** %0, align 8, !dbg !133
  store i8* %1, i8** %.S0000_518, align 8, !dbg !133
  br label %L.LB7_540

L.LB7_540:                                        ; preds = %L.entry
  br label %L.LB7_336

L.LB7_336:                                        ; preds = %L.LB7_540
  %2 = load i8*, i8** %.S0000_518, align 8, !dbg !134
  %3 = getelementptr i8, i8* %2, i64 8, !dbg !134
  %4 = bitcast i8* %3 to i32**, !dbg !134
  %5 = load i32*, i32** %4, align 8, !dbg !134
  %6 = load i32, i32* %5, align 4, !dbg !134
  %7 = load i8*, i8** %.S0000_518, align 8, !dbg !134
  %8 = bitcast i8* %7 to i32**, !dbg !134
  %9 = load i32*, i32** %8, align 8, !dbg !134
  %10 = load i32, i32* %9, align 4, !dbg !134
  %11 = add nsw i32 %6, %10, !dbg !134
  %12 = load i8*, i8** %.S0000_518, align 8, !dbg !134
  %13 = bitcast i8* %12 to i32**, !dbg !134
  %14 = load i32*, i32** %13, align 8, !dbg !134
  store i32 %11, i32* %14, align 4, !dbg !134
  br label %L.LB7_337

L.LB7_337:                                        ; preds = %L.LB7_336
  ret void, !dbg !135
}

define internal void @__nv_MAIN_F1L32_6_(i32 %__nv_MAIN_F1L32_6Arg0.arg, i64* %__nv_MAIN_F1L32_6Arg1) #0 !dbg !136 {
L.entry:
  %__nv_MAIN_F1L32_6Arg0.addr = alloca i32, align 4
  %.S0000_518 = alloca i8*, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L32_6Arg0.addr, metadata !137, metadata !DIExpression()), !dbg !138
  store i32 %__nv_MAIN_F1L32_6Arg0.arg, i32* %__nv_MAIN_F1L32_6Arg0.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L32_6Arg0.addr, metadata !139, metadata !DIExpression()), !dbg !138
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L32_6Arg1, metadata !140, metadata !DIExpression()), !dbg !138
  call void @llvm.dbg.value(metadata i32 1, metadata !141, metadata !DIExpression()), !dbg !138
  call void @llvm.dbg.value(metadata i32 2, metadata !142, metadata !DIExpression()), !dbg !138
  call void @llvm.dbg.value(metadata i32 3, metadata !143, metadata !DIExpression()), !dbg !138
  call void @llvm.dbg.value(metadata i32 0, metadata !144, metadata !DIExpression()), !dbg !138
  call void @llvm.dbg.value(metadata i32 1, metadata !145, metadata !DIExpression()), !dbg !138
  call void @llvm.dbg.value(metadata i32 2, metadata !146, metadata !DIExpression()), !dbg !138
  call void @llvm.dbg.value(metadata i32 3, metadata !147, metadata !DIExpression()), !dbg !138
  call void @llvm.dbg.value(metadata i32 0, metadata !148, metadata !DIExpression()), !dbg !138
  call void @llvm.dbg.value(metadata i32 1, metadata !149, metadata !DIExpression()), !dbg !138
  call void @llvm.dbg.value(metadata i32 2, metadata !150, metadata !DIExpression()), !dbg !138
  %0 = bitcast i64* %__nv_MAIN_F1L32_6Arg1 to i8**, !dbg !151
  %1 = load i8*, i8** %0, align 8, !dbg !151
  store i8* %1, i8** %.S0000_518, align 8, !dbg !151
  br label %L.LB8_546

L.LB8_546:                                        ; preds = %L.entry
  br label %L.LB8_340

L.LB8_340:                                        ; preds = %L.LB8_546
  %2 = load i8*, i8** %.S0000_518, align 8, !dbg !152
  %3 = getelementptr i8, i8* %2, i64 16, !dbg !152
  %4 = bitcast i8* %3 to i32**, !dbg !152
  %5 = load i32*, i32** %4, align 8, !dbg !152
  %6 = load i32, i32* %5, align 4, !dbg !152
  %7 = load i8*, i8** %.S0000_518, align 8, !dbg !152
  %8 = bitcast i8* %7 to i32**, !dbg !152
  %9 = load i32*, i32** %8, align 8, !dbg !152
  %10 = load i32, i32* %9, align 4, !dbg !152
  %11 = add nsw i32 %6, %10, !dbg !152
  %12 = load i8*, i8** %.S0000_518, align 8, !dbg !152
  %13 = bitcast i8* %12 to i32**, !dbg !152
  %14 = load i32*, i32** %13, align 8, !dbg !152
  store i32 %11, i32* %14, align 4, !dbg !152
  br label %L.LB8_341

L.LB8_341:                                        ; preds = %L.LB8_340
  ret void, !dbg !153
}

define internal void @__nv_MAIN_F1L35_7_(i32 %__nv_MAIN_F1L35_7Arg0.arg, i64* %__nv_MAIN_F1L35_7Arg1) #0 !dbg !154 {
L.entry:
  %__nv_MAIN_F1L35_7Arg0.addr = alloca i32, align 4
  %.S0000_518 = alloca i8*, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L35_7Arg0.addr, metadata !155, metadata !DIExpression()), !dbg !156
  store i32 %__nv_MAIN_F1L35_7Arg0.arg, i32* %__nv_MAIN_F1L35_7Arg0.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L35_7Arg0.addr, metadata !157, metadata !DIExpression()), !dbg !156
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L35_7Arg1, metadata !158, metadata !DIExpression()), !dbg !156
  call void @llvm.dbg.value(metadata i32 1, metadata !159, metadata !DIExpression()), !dbg !156
  call void @llvm.dbg.value(metadata i32 2, metadata !160, metadata !DIExpression()), !dbg !156
  call void @llvm.dbg.value(metadata i32 3, metadata !161, metadata !DIExpression()), !dbg !156
  call void @llvm.dbg.value(metadata i32 0, metadata !162, metadata !DIExpression()), !dbg !156
  call void @llvm.dbg.value(metadata i32 1, metadata !163, metadata !DIExpression()), !dbg !156
  call void @llvm.dbg.value(metadata i32 2, metadata !164, metadata !DIExpression()), !dbg !156
  call void @llvm.dbg.value(metadata i32 3, metadata !165, metadata !DIExpression()), !dbg !156
  call void @llvm.dbg.value(metadata i32 0, metadata !166, metadata !DIExpression()), !dbg !156
  call void @llvm.dbg.value(metadata i32 1, metadata !167, metadata !DIExpression()), !dbg !156
  call void @llvm.dbg.value(metadata i32 2, metadata !168, metadata !DIExpression()), !dbg !156
  %0 = bitcast i64* %__nv_MAIN_F1L35_7Arg1 to i8**, !dbg !169
  %1 = load i8*, i8** %0, align 8, !dbg !169
  store i8* %1, i8** %.S0000_518, align 8, !dbg !169
  br label %L.LB9_552

L.LB9_552:                                        ; preds = %L.entry
  br label %L.LB9_344

L.LB9_344:                                        ; preds = %L.LB9_552
  %2 = load i8*, i8** %.S0000_518, align 8, !dbg !170
  %3 = bitcast i8* %2 to i32**, !dbg !170
  %4 = load i32*, i32** %3, align 8, !dbg !170
  %5 = load i32, i32* %4, align 4, !dbg !170
  %6 = load i8*, i8** %.S0000_518, align 8, !dbg !170
  %7 = getelementptr i8, i8* %6, i64 24, !dbg !170
  %8 = bitcast i8* %7 to i32**, !dbg !170
  %9 = load i32*, i32** %8, align 8, !dbg !170
  store i32 %5, i32* %9, align 4, !dbg !170
  br label %L.LB9_345

L.LB9_345:                                        ; preds = %L.LB9_344
  ret void, !dbg !171
}

declare void @__kmpc_barrier(i64*, i32) #0

declare void @__kmpc_end_single(i64*, i32) #0

declare void @__kmpc_omp_task(i64*, i32, i64*) #0

declare i8* @__kmpc_omp_task_alloc(i64*, i32, i32, i32, i32, i64*) #0

declare signext i32 @__kmpc_single(i64*, i32) #0

declare void @_mp_ecs_nest(...) #0

declare signext i32 @f90io_ldw_end(...) #0

declare signext i32 @f90io_sc_i_ldw(...) #0

declare signext i32 @f90io_print_init(...) #0

declare void @f90io_src_info03a(...) #0

declare void @_mp_bcs_nest(...) #0

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare void @fort_init(...) #0

declare signext i32 @__kmpc_global_thread_num(i64*) #0

declare void @__kmpc_fork_call(i64*, i32, i64*, i64*, ...) #0

declare void @__kmpc_end_serialized_parallel(i64*, i32) #0

declare void @__kmpc_serialized_parallel(i64*, i32) #0

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { "no-frame-pointer-elim-non-leaf" }
attributes #1 = { nounwind readnone speculatable }

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !4, globals: !4, imports: !4)
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB135-taskdep-mutexinoutset-orig-omp50-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb135_taskdep_mutexinoutset_orig_no_omp50", scope: !2, file: !3, line: 12, type: !6, scopeLine: 12, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!6 = !DISubroutineType(cc: DW_CC_program, types: !7)
!7 = !{null}
!8 = !DILocalVariable(name: "omp_sched_static", scope: !5, file: !3, type: !9)
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 0, scope: !5)
!11 = !DILocalVariable(name: "omp_sched_dynamic", scope: !5, file: !3, type: !9)
!12 = !DILocalVariable(name: "omp_sched_guided", scope: !5, file: !3, type: !9)
!13 = !DILocalVariable(name: "omp_proc_bind_false", scope: !5, file: !3, type: !9)
!14 = !DILocalVariable(name: "omp_proc_bind_true", scope: !5, file: !3, type: !9)
!15 = !DILocalVariable(name: "omp_proc_bind_master", scope: !5, file: !3, type: !9)
!16 = !DILocalVariable(name: "omp_proc_bind_close", scope: !5, file: !3, type: !9)
!17 = !DILocalVariable(name: "omp_lock_hint_none", scope: !5, file: !3, type: !9)
!18 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !5, file: !3, type: !9)
!19 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !5, file: !3, type: !9)
!20 = !DILocation(line: 42, column: 1, scope: !5)
!21 = !DILocation(line: 12, column: 1, scope: !5)
!22 = !DILocalVariable(name: "c", scope: !5, file: !3, type: !9)
!23 = !DILocation(line: 18, column: 1, scope: !5)
!24 = !DILocalVariable(name: "a", scope: !5, file: !3, type: !9)
!25 = !DILocalVariable(name: "b", scope: !5, file: !3, type: !9)
!26 = !DILocalVariable(name: "d", scope: !5, file: !3, type: !9)
!27 = !DILocation(line: 41, column: 1, scope: !5)
!28 = !DILocalVariable(scope: !5, file: !3, type: !9, flags: DIFlagArtificial)
!29 = distinct !DISubprogram(name: "__nv_MAIN__F1L18_1", scope: !2, file: !3, line: 18, type: !30, scopeLine: 18, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!30 = !DISubroutineType(types: !31)
!31 = !{null, !9, !32, !32}
!32 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!33 = !DILocalVariable(name: "__nv_MAIN__F1L18_1Arg0", arg: 1, scope: !29, file: !3, type: !9)
!34 = !DILocation(line: 0, scope: !29)
!35 = !DILocalVariable(name: "__nv_MAIN__F1L18_1Arg1", arg: 2, scope: !29, file: !3, type: !32)
!36 = !DILocalVariable(name: "__nv_MAIN__F1L18_1Arg2", arg: 3, scope: !29, file: !3, type: !32)
!37 = !DILocalVariable(name: "omp_sched_static", scope: !29, file: !3, type: !9)
!38 = !DILocalVariable(name: "omp_sched_dynamic", scope: !29, file: !3, type: !9)
!39 = !DILocalVariable(name: "omp_sched_guided", scope: !29, file: !3, type: !9)
!40 = !DILocalVariable(name: "omp_proc_bind_false", scope: !29, file: !3, type: !9)
!41 = !DILocalVariable(name: "omp_proc_bind_true", scope: !29, file: !3, type: !9)
!42 = !DILocalVariable(name: "omp_proc_bind_master", scope: !29, file: !3, type: !9)
!43 = !DILocalVariable(name: "omp_proc_bind_close", scope: !29, file: !3, type: !9)
!44 = !DILocalVariable(name: "omp_lock_hint_none", scope: !29, file: !3, type: !9)
!45 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !29, file: !3, type: !9)
!46 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !29, file: !3, type: !9)
!47 = !DILocation(line: 39, column: 1, scope: !29)
!48 = !DILocation(line: 19, column: 1, scope: !29)
!49 = !DILocation(line: 20, column: 1, scope: !29)
!50 = !DILocation(line: 22, column: 1, scope: !29)
!51 = !DILocation(line: 23, column: 1, scope: !29)
!52 = !DILocation(line: 25, column: 1, scope: !29)
!53 = !DILocation(line: 26, column: 1, scope: !29)
!54 = !DILocation(line: 28, column: 1, scope: !29)
!55 = !DILocation(line: 29, column: 1, scope: !29)
!56 = !DILocation(line: 31, column: 1, scope: !29)
!57 = !DILocation(line: 32, column: 1, scope: !29)
!58 = !DILocation(line: 34, column: 1, scope: !29)
!59 = !DILocation(line: 35, column: 1, scope: !29)
!60 = !DILocation(line: 37, column: 1, scope: !29)
!61 = !DILocation(line: 38, column: 1, scope: !29)
!62 = distinct !DISubprogram(name: "__nv_MAIN_F1L20_2", scope: !2, file: !3, line: 20, type: !63, scopeLine: 20, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!63 = !DISubroutineType(types: !64)
!64 = !{null, !9, !32}
!65 = !DILocalVariable(name: "__nv_MAIN_F1L20_2Arg0", scope: !62, file: !3, type: !9)
!66 = !DILocation(line: 0, scope: !62)
!67 = !DILocalVariable(name: "__nv_MAIN_F1L20_2Arg0", arg: 1, scope: !62, file: !3, type: !9)
!68 = !DILocalVariable(name: "__nv_MAIN_F1L20_2Arg1", arg: 2, scope: !62, file: !3, type: !32)
!69 = !DILocalVariable(name: "omp_sched_static", scope: !62, file: !3, type: !9)
!70 = !DILocalVariable(name: "omp_sched_dynamic", scope: !62, file: !3, type: !9)
!71 = !DILocalVariable(name: "omp_sched_guided", scope: !62, file: !3, type: !9)
!72 = !DILocalVariable(name: "omp_proc_bind_false", scope: !62, file: !3, type: !9)
!73 = !DILocalVariable(name: "omp_proc_bind_true", scope: !62, file: !3, type: !9)
!74 = !DILocalVariable(name: "omp_proc_bind_master", scope: !62, file: !3, type: !9)
!75 = !DILocalVariable(name: "omp_proc_bind_close", scope: !62, file: !3, type: !9)
!76 = !DILocalVariable(name: "omp_lock_hint_none", scope: !62, file: !3, type: !9)
!77 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !62, file: !3, type: !9)
!78 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !62, file: !3, type: !9)
!79 = !DILocation(line: 20, column: 1, scope: !62)
!80 = !DILocation(line: 21, column: 1, scope: !62)
!81 = !DILocation(line: 22, column: 1, scope: !62)
!82 = distinct !DISubprogram(name: "__nv_MAIN_F1L23_3", scope: !2, file: !3, line: 23, type: !63, scopeLine: 23, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!83 = !DILocalVariable(name: "__nv_MAIN_F1L23_3Arg0", scope: !82, file: !3, type: !9)
!84 = !DILocation(line: 0, scope: !82)
!85 = !DILocalVariable(name: "__nv_MAIN_F1L23_3Arg0", arg: 1, scope: !82, file: !3, type: !9)
!86 = !DILocalVariable(name: "__nv_MAIN_F1L23_3Arg1", arg: 2, scope: !82, file: !3, type: !32)
!87 = !DILocalVariable(name: "omp_sched_static", scope: !82, file: !3, type: !9)
!88 = !DILocalVariable(name: "omp_sched_dynamic", scope: !82, file: !3, type: !9)
!89 = !DILocalVariable(name: "omp_sched_guided", scope: !82, file: !3, type: !9)
!90 = !DILocalVariable(name: "omp_proc_bind_false", scope: !82, file: !3, type: !9)
!91 = !DILocalVariable(name: "omp_proc_bind_true", scope: !82, file: !3, type: !9)
!92 = !DILocalVariable(name: "omp_proc_bind_master", scope: !82, file: !3, type: !9)
!93 = !DILocalVariable(name: "omp_proc_bind_close", scope: !82, file: !3, type: !9)
!94 = !DILocalVariable(name: "omp_lock_hint_none", scope: !82, file: !3, type: !9)
!95 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !82, file: !3, type: !9)
!96 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !82, file: !3, type: !9)
!97 = !DILocation(line: 23, column: 1, scope: !82)
!98 = !DILocation(line: 24, column: 1, scope: !82)
!99 = !DILocation(line: 25, column: 1, scope: !82)
!100 = distinct !DISubprogram(name: "__nv_MAIN_F1L26_4", scope: !2, file: !3, line: 26, type: !63, scopeLine: 26, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!101 = !DILocalVariable(name: "__nv_MAIN_F1L26_4Arg0", scope: !100, file: !3, type: !9)
!102 = !DILocation(line: 0, scope: !100)
!103 = !DILocalVariable(name: "__nv_MAIN_F1L26_4Arg0", arg: 1, scope: !100, file: !3, type: !9)
!104 = !DILocalVariable(name: "__nv_MAIN_F1L26_4Arg1", arg: 2, scope: !100, file: !3, type: !32)
!105 = !DILocalVariable(name: "omp_sched_static", scope: !100, file: !3, type: !9)
!106 = !DILocalVariable(name: "omp_sched_dynamic", scope: !100, file: !3, type: !9)
!107 = !DILocalVariable(name: "omp_sched_guided", scope: !100, file: !3, type: !9)
!108 = !DILocalVariable(name: "omp_proc_bind_false", scope: !100, file: !3, type: !9)
!109 = !DILocalVariable(name: "omp_proc_bind_true", scope: !100, file: !3, type: !9)
!110 = !DILocalVariable(name: "omp_proc_bind_master", scope: !100, file: !3, type: !9)
!111 = !DILocalVariable(name: "omp_proc_bind_close", scope: !100, file: !3, type: !9)
!112 = !DILocalVariable(name: "omp_lock_hint_none", scope: !100, file: !3, type: !9)
!113 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !100, file: !3, type: !9)
!114 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !100, file: !3, type: !9)
!115 = !DILocation(line: 26, column: 1, scope: !100)
!116 = !DILocation(line: 27, column: 1, scope: !100)
!117 = !DILocation(line: 28, column: 1, scope: !100)
!118 = distinct !DISubprogram(name: "__nv_MAIN_F1L29_5", scope: !2, file: !3, line: 29, type: !63, scopeLine: 29, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!119 = !DILocalVariable(name: "__nv_MAIN_F1L29_5Arg0", scope: !118, file: !3, type: !9)
!120 = !DILocation(line: 0, scope: !118)
!121 = !DILocalVariable(name: "__nv_MAIN_F1L29_5Arg0", arg: 1, scope: !118, file: !3, type: !9)
!122 = !DILocalVariable(name: "__nv_MAIN_F1L29_5Arg1", arg: 2, scope: !118, file: !3, type: !32)
!123 = !DILocalVariable(name: "omp_sched_static", scope: !118, file: !3, type: !9)
!124 = !DILocalVariable(name: "omp_sched_dynamic", scope: !118, file: !3, type: !9)
!125 = !DILocalVariable(name: "omp_sched_guided", scope: !118, file: !3, type: !9)
!126 = !DILocalVariable(name: "omp_proc_bind_false", scope: !118, file: !3, type: !9)
!127 = !DILocalVariable(name: "omp_proc_bind_true", scope: !118, file: !3, type: !9)
!128 = !DILocalVariable(name: "omp_proc_bind_master", scope: !118, file: !3, type: !9)
!129 = !DILocalVariable(name: "omp_proc_bind_close", scope: !118, file: !3, type: !9)
!130 = !DILocalVariable(name: "omp_lock_hint_none", scope: !118, file: !3, type: !9)
!131 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !118, file: !3, type: !9)
!132 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !118, file: !3, type: !9)
!133 = !DILocation(line: 29, column: 1, scope: !118)
!134 = !DILocation(line: 30, column: 1, scope: !118)
!135 = !DILocation(line: 31, column: 1, scope: !118)
!136 = distinct !DISubprogram(name: "__nv_MAIN_F1L32_6", scope: !2, file: !3, line: 32, type: !63, scopeLine: 32, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!137 = !DILocalVariable(name: "__nv_MAIN_F1L32_6Arg0", scope: !136, file: !3, type: !9)
!138 = !DILocation(line: 0, scope: !136)
!139 = !DILocalVariable(name: "__nv_MAIN_F1L32_6Arg0", arg: 1, scope: !136, file: !3, type: !9)
!140 = !DILocalVariable(name: "__nv_MAIN_F1L32_6Arg1", arg: 2, scope: !136, file: !3, type: !32)
!141 = !DILocalVariable(name: "omp_sched_static", scope: !136, file: !3, type: !9)
!142 = !DILocalVariable(name: "omp_sched_dynamic", scope: !136, file: !3, type: !9)
!143 = !DILocalVariable(name: "omp_sched_guided", scope: !136, file: !3, type: !9)
!144 = !DILocalVariable(name: "omp_proc_bind_false", scope: !136, file: !3, type: !9)
!145 = !DILocalVariable(name: "omp_proc_bind_true", scope: !136, file: !3, type: !9)
!146 = !DILocalVariable(name: "omp_proc_bind_master", scope: !136, file: !3, type: !9)
!147 = !DILocalVariable(name: "omp_proc_bind_close", scope: !136, file: !3, type: !9)
!148 = !DILocalVariable(name: "omp_lock_hint_none", scope: !136, file: !3, type: !9)
!149 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !136, file: !3, type: !9)
!150 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !136, file: !3, type: !9)
!151 = !DILocation(line: 32, column: 1, scope: !136)
!152 = !DILocation(line: 33, column: 1, scope: !136)
!153 = !DILocation(line: 34, column: 1, scope: !136)
!154 = distinct !DISubprogram(name: "__nv_MAIN_F1L35_7", scope: !2, file: !3, line: 35, type: !63, scopeLine: 35, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!155 = !DILocalVariable(name: "__nv_MAIN_F1L35_7Arg0", scope: !154, file: !3, type: !9)
!156 = !DILocation(line: 0, scope: !154)
!157 = !DILocalVariable(name: "__nv_MAIN_F1L35_7Arg0", arg: 1, scope: !154, file: !3, type: !9)
!158 = !DILocalVariable(name: "__nv_MAIN_F1L35_7Arg1", arg: 2, scope: !154, file: !3, type: !32)
!159 = !DILocalVariable(name: "omp_sched_static", scope: !154, file: !3, type: !9)
!160 = !DILocalVariable(name: "omp_sched_dynamic", scope: !154, file: !3, type: !9)
!161 = !DILocalVariable(name: "omp_sched_guided", scope: !154, file: !3, type: !9)
!162 = !DILocalVariable(name: "omp_proc_bind_false", scope: !154, file: !3, type: !9)
!163 = !DILocalVariable(name: "omp_proc_bind_true", scope: !154, file: !3, type: !9)
!164 = !DILocalVariable(name: "omp_proc_bind_master", scope: !154, file: !3, type: !9)
!165 = !DILocalVariable(name: "omp_proc_bind_close", scope: !154, file: !3, type: !9)
!166 = !DILocalVariable(name: "omp_lock_hint_none", scope: !154, file: !3, type: !9)
!167 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !154, file: !3, type: !9)
!168 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !154, file: !3, type: !9)
!169 = !DILocation(line: 35, column: 1, scope: !154)
!170 = !DILocation(line: 36, column: 1, scope: !154)
!171 = !DILocation(line: 37, column: 1, scope: !154)

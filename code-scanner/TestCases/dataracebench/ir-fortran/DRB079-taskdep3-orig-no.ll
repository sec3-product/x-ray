; ModuleID = '/tmp/DRB079-taskdep3-orig-no-cd9d8b.ll'
source_filename = "/tmp/DRB079-taskdep3-orig-no-cd9d8b.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.STATICS1 = type <{ [84 x i8] }>
%astruct.dt62 = type <{ i8*, i8*, i8* }>

@.STATICS1 = internal global %struct.STATICS1 <{ [84 x i8] c"\FB\FF\FF\FF\03\00\00\00j =\00\EB\FF\FF\FF\00\00\00\00\03\00\00\00\00\00\00\00\01\00\00\00\F7\FF\FF\FF\00\00\00\00\02\00\00\00\FB\FF\FF\FF\03\00\00\00k =\00\EB\FF\FF\FF\00\00\00\00\03\00\00\00\00\00\00\00\01\00\00\00\FF\FF\FF\FF\00\00\00\00" }>, align 16
@.C308_MAIN_ = internal constant i32 14
@.C349_MAIN_ = internal constant [14 x i8] c"Race Condition"
@.C347_MAIN_ = internal constant i32 36
@.C309_MAIN_ = internal constant i32 25
@.C284_MAIN_ = internal constant i64 0
@.C338_MAIN_ = internal constant i32 6
@.C334_MAIN_ = internal constant [52 x i8] c"micro-benchmarks-fortran/DRB079-taskdep3-orig-no.f95"
@.C336_MAIN_ = internal constant i32 32
@.C300_MAIN_ = internal constant i32 3
@.C285_MAIN_ = internal constant i32 1
@.C283_MAIN_ = internal constant i32 0
@.C300___nv_MAIN__F1L17_1 = internal constant i32 3
@.C283___nv_MAIN__F1L17_1 = internal constant i32 0
@.C285___nv_MAIN__F1L17_1 = internal constant i32 1
@.C285___nv_MAIN_F1L19_2 = internal constant i32 1
@.C300___nv_MAIN_F1L19_2 = internal constant i32 3

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__379 = alloca i32, align 4
  %i_310 = alloca i32, align 4
  %.uplevelArgPack0001_368 = alloca %astruct.dt62, align 16
  %j_311 = alloca i32, align 4
  %k_312 = alloca i32, align 4
  %z__io_340 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 3, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 3, metadata !14, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !15, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !16, metadata !DIExpression()), !dbg !10
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !17
  store i32 %0, i32* %__gtid_MAIN__379, align 4, !dbg !17
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !18
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !18
  call void (i8*, ...) %2(i8* %1), !dbg !18
  br label %L.LB1_362

L.LB1_362:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %i_310, metadata !19, metadata !DIExpression()), !dbg !10
  store i32 0, i32* %i_310, align 4, !dbg !20
  %3 = bitcast i32* %i_310 to i8*, !dbg !21
  %4 = bitcast %astruct.dt62* %.uplevelArgPack0001_368 to i8**, !dbg !21
  store i8* %3, i8** %4, align 8, !dbg !21
  call void @llvm.dbg.declare(metadata i32* %j_311, metadata !22, metadata !DIExpression()), !dbg !10
  %5 = bitcast i32* %j_311 to i8*, !dbg !21
  %6 = bitcast %astruct.dt62* %.uplevelArgPack0001_368 to i8*, !dbg !21
  %7 = getelementptr i8, i8* %6, i64 8, !dbg !21
  %8 = bitcast i8* %7 to i8**, !dbg !21
  store i8* %5, i8** %8, align 8, !dbg !21
  call void @llvm.dbg.declare(metadata i32* %k_312, metadata !23, metadata !DIExpression()), !dbg !10
  %9 = bitcast i32* %k_312 to i8*, !dbg !21
  %10 = bitcast %astruct.dt62* %.uplevelArgPack0001_368 to i8*, !dbg !21
  %11 = getelementptr i8, i8* %10, i64 16, !dbg !21
  %12 = bitcast i8* %11 to i8**, !dbg !21
  store i8* %9, i8** %12, align 8, !dbg !21
  br label %L.LB1_377, !dbg !21

L.LB1_377:                                        ; preds = %L.LB1_362
  %13 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L17_1_ to i64*, !dbg !21
  %14 = bitcast %astruct.dt62* %.uplevelArgPack0001_368 to i64*, !dbg !21
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %13, i64* %14), !dbg !21
  call void (...) @_mp_bcs_nest(), !dbg !24
  %15 = bitcast i32* @.C336_MAIN_ to i8*, !dbg !24
  %16 = bitcast [52 x i8]* @.C334_MAIN_ to i8*, !dbg !24
  %17 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !24
  call void (i8*, i8*, i64, ...) %17(i8* %15, i8* %16, i64 52), !dbg !24
  %18 = bitcast i32* @.C338_MAIN_ to i8*, !dbg !24
  %19 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !24
  %20 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !24
  %21 = bitcast %struct.STATICS1* @.STATICS1 to i8*, !dbg !24
  %22 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !24
  %23 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %22(i8* %18, i8* null, i8* %19, i8* %20, i8* %21, i8* null, i64 0), !dbg !24
  call void @llvm.dbg.declare(metadata i32* %z__io_340, metadata !25, metadata !DIExpression()), !dbg !10
  store i32 %23, i32* %z__io_340, align 4, !dbg !24
  %24 = load i32, i32* %j_311, align 4, !dbg !24
  call void @llvm.dbg.value(metadata i32 %24, metadata !22, metadata !DIExpression()), !dbg !10
  %25 = bitcast i32 (...)* @f90io_sc_i_fmt_write to i32 (i32, i32, ...)*, !dbg !24
  %26 = call i32 (i32, i32, ...) %25(i32 %24, i32 25), !dbg !24
  store i32 %26, i32* %z__io_340, align 4, !dbg !24
  %27 = load i32, i32* %k_312, align 4, !dbg !24
  call void @llvm.dbg.value(metadata i32 %27, metadata !23, metadata !DIExpression()), !dbg !10
  %28 = bitcast i32 (...)* @f90io_sc_i_fmt_write to i32 (i32, i32, ...)*, !dbg !24
  %29 = call i32 (i32, i32, ...) %28(i32 %27, i32 25), !dbg !24
  store i32 %29, i32* %z__io_340, align 4, !dbg !24
  %30 = call i32 (...) @f90io_fmtw_end(), !dbg !24
  store i32 %30, i32* %z__io_340, align 4, !dbg !24
  call void (...) @_mp_ecs_nest(), !dbg !24
  %31 = load i32, i32* %j_311, align 4, !dbg !26
  call void @llvm.dbg.value(metadata i32 %31, metadata !22, metadata !DIExpression()), !dbg !10
  %32 = icmp eq i32 %31, 1, !dbg !26
  br i1 %32, label %L.LB1_360, label %L.LB1_408, !dbg !26

L.LB1_408:                                        ; preds = %L.LB1_377
  %33 = load i32, i32* %k_312, align 4, !dbg !26
  call void @llvm.dbg.value(metadata i32 %33, metadata !23, metadata !DIExpression()), !dbg !10
  %34 = icmp eq i32 %33, 1, !dbg !26
  br i1 %34, label %L.LB1_360, label %L.LB1_409, !dbg !26

L.LB1_409:                                        ; preds = %L.LB1_408
  call void (...) @_mp_bcs_nest(), !dbg !27
  %35 = bitcast i32* @.C347_MAIN_ to i8*, !dbg !27
  %36 = bitcast [52 x i8]* @.C334_MAIN_ to i8*, !dbg !27
  %37 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !27
  call void (i8*, i8*, i64, ...) %37(i8* %35, i8* %36, i64 52), !dbg !27
  %38 = bitcast i32* @.C338_MAIN_ to i8*, !dbg !27
  %39 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !27
  %40 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !27
  %41 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !27
  %42 = call i32 (i8*, i8*, i8*, i8*, ...) %41(i8* %38, i8* null, i8* %39, i8* %40), !dbg !27
  store i32 %42, i32* %z__io_340, align 4, !dbg !27
  %43 = bitcast [14 x i8]* @.C349_MAIN_ to i8*, !dbg !27
  %44 = bitcast i32 (...)* @f90io_sc_ch_ldw to i32 (i8*, i32, i64, ...)*, !dbg !27
  %45 = call i32 (i8*, i32, i64, ...) %44(i8* %43, i32 14, i64 14), !dbg !27
  store i32 %45, i32* %z__io_340, align 4, !dbg !27
  %46 = call i32 (...) @f90io_ldw_end(), !dbg !27
  store i32 %46, i32* %z__io_340, align 4, !dbg !27
  call void (...) @_mp_ecs_nest(), !dbg !27
  br label %L.LB1_360

L.LB1_360:                                        ; preds = %L.LB1_409, %L.LB1_408, %L.LB1_377
  ret void, !dbg !17
}

define internal void @__nv_MAIN__F1L17_1_(i32* %__nv_MAIN__F1L17_1Arg0, i64* %__nv_MAIN__F1L17_1Arg1, i64* %__nv_MAIN__F1L17_1Arg2) #0 !dbg !28 {
L.entry:
  %__gtid___nv_MAIN__F1L17_1__419 = alloca i32, align 4
  %.s0000_414 = alloca i32, align 4
  %.s0001_415 = alloca i32, align 4
  %.s0002_425 = alloca i32, align 4
  %.z0364_424 = alloca i8*, align 8
  %.s0003_449 = alloca i32, align 4
  %.z0364_448 = alloca i8*, align 8
  %.s0004_459 = alloca i32, align 4
  %.z0364_458 = alloca i8*, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L17_1Arg0, metadata !32, metadata !DIExpression()), !dbg !33
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L17_1Arg1, metadata !34, metadata !DIExpression()), !dbg !33
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L17_1Arg2, metadata !35, metadata !DIExpression()), !dbg !33
  call void @llvm.dbg.value(metadata i32 1, metadata !36, metadata !DIExpression()), !dbg !33
  call void @llvm.dbg.value(metadata i32 3, metadata !37, metadata !DIExpression()), !dbg !33
  call void @llvm.dbg.value(metadata i32 0, metadata !38, metadata !DIExpression()), !dbg !33
  call void @llvm.dbg.value(metadata i32 1, metadata !39, metadata !DIExpression()), !dbg !33
  call void @llvm.dbg.value(metadata i32 3, metadata !40, metadata !DIExpression()), !dbg !33
  call void @llvm.dbg.value(metadata i32 0, metadata !41, metadata !DIExpression()), !dbg !33
  call void @llvm.dbg.value(metadata i32 1, metadata !42, metadata !DIExpression()), !dbg !33
  %0 = load i32, i32* %__nv_MAIN__F1L17_1Arg0, align 4, !dbg !43
  store i32 %0, i32* %__gtid___nv_MAIN__F1L17_1__419, align 4, !dbg !43
  br label %L.LB2_413

L.LB2_413:                                        ; preds = %L.entry
  br label %L.LB2_315

L.LB2_315:                                        ; preds = %L.LB2_413
  store i32 -1, i32* %.s0000_414, align 4, !dbg !44
  store i32 0, i32* %.s0001_415, align 4, !dbg !44
  %1 = load i32, i32* %__gtid___nv_MAIN__F1L17_1__419, align 4, !dbg !44
  %2 = call i32 @__kmpc_single(i64* null, i32 %1), !dbg !44
  %3 = icmp eq i32 %2, 0, !dbg !44
  br i1 %3, label %L.LB2_356, label %L.LB2_317, !dbg !44

L.LB2_317:                                        ; preds = %L.LB2_315
  store i32 1, i32* %.s0002_425, align 4, !dbg !45
  %4 = load i32, i32* %__gtid___nv_MAIN__F1L17_1__419, align 4, !dbg !46
  %5 = load i32, i32* %.s0002_425, align 4, !dbg !46
  %6 = bitcast void (i32, i64*)* @__nv_MAIN_F1L19_2_ to i64*, !dbg !46
  %7 = call i8* @__kmpc_omp_task_alloc(i64* null, i32 %4, i32 %5, i32 40, i32 24, i64* %6), !dbg !46
  store i8* %7, i8** %.z0364_424, align 8, !dbg !46
  %8 = load i64, i64* %__nv_MAIN__F1L17_1Arg2, align 8, !dbg !46
  %9 = load i8*, i8** %.z0364_424, align 8, !dbg !46
  %10 = bitcast i8* %9 to i64**, !dbg !46
  %11 = load i64*, i64** %10, align 8, !dbg !46
  store i64 %8, i64* %11, align 8, !dbg !46
  %12 = bitcast i64* %__nv_MAIN__F1L17_1Arg2 to i8*, !dbg !43
  %13 = getelementptr i8, i8* %12, i64 8, !dbg !43
  %14 = bitcast i8* %13 to i64*, !dbg !43
  %15 = load i64, i64* %14, align 8, !dbg !43
  %16 = bitcast i64* %11 to i8*, !dbg !43
  %17 = getelementptr i8, i8* %16, i64 8, !dbg !43
  %18 = bitcast i8* %17 to i64*, !dbg !43
  store i64 %15, i64* %18, align 8, !dbg !43
  %19 = bitcast i64* %__nv_MAIN__F1L17_1Arg2 to i8*, !dbg !43
  %20 = getelementptr i8, i8* %19, i64 16, !dbg !43
  %21 = bitcast i8* %20 to i64*, !dbg !43
  %22 = load i64, i64* %21, align 8, !dbg !43
  %23 = bitcast i64* %11 to i8*, !dbg !43
  %24 = getelementptr i8, i8* %23, i64 16, !dbg !43
  %25 = bitcast i8* %24 to i64*, !dbg !43
  store i64 %22, i64* %25, align 8, !dbg !43
  %26 = load i32, i32* %__gtid___nv_MAIN__F1L17_1__419, align 4, !dbg !46
  %27 = load i8*, i8** %.z0364_424, align 8, !dbg !46
  %28 = bitcast i8* %27 to i64*, !dbg !46
  call void @__kmpc_omp_task(i64* null, i32 %26, i64* %28), !dbg !46
  br label %L.LB2_357

L.LB2_357:                                        ; preds = %L.LB2_317
  store i32 1, i32* %.s0003_449, align 4, !dbg !47
  %29 = load i32, i32* %__gtid___nv_MAIN__F1L17_1__419, align 4, !dbg !48
  %30 = load i32, i32* %.s0003_449, align 4, !dbg !48
  %31 = bitcast void (i32, i64*)* @__nv_MAIN_F1L23_3_ to i64*, !dbg !48
  %32 = call i8* @__kmpc_omp_task_alloc(i64* null, i32 %29, i32 %30, i32 40, i32 24, i64* %31), !dbg !48
  store i8* %32, i8** %.z0364_448, align 8, !dbg !48
  %33 = load i64, i64* %__nv_MAIN__F1L17_1Arg2, align 8, !dbg !48
  %34 = load i8*, i8** %.z0364_448, align 8, !dbg !48
  %35 = bitcast i8* %34 to i64**, !dbg !48
  %36 = load i64*, i64** %35, align 8, !dbg !48
  store i64 %33, i64* %36, align 8, !dbg !48
  %37 = bitcast i64* %__nv_MAIN__F1L17_1Arg2 to i8*, !dbg !43
  %38 = getelementptr i8, i8* %37, i64 8, !dbg !43
  %39 = bitcast i8* %38 to i64*, !dbg !43
  %40 = load i64, i64* %39, align 8, !dbg !43
  %41 = bitcast i64* %36 to i8*, !dbg !43
  %42 = getelementptr i8, i8* %41, i64 8, !dbg !43
  %43 = bitcast i8* %42 to i64*, !dbg !43
  store i64 %40, i64* %43, align 8, !dbg !43
  %44 = bitcast i64* %__nv_MAIN__F1L17_1Arg2 to i8*, !dbg !43
  %45 = getelementptr i8, i8* %44, i64 16, !dbg !43
  %46 = bitcast i8* %45 to i64*, !dbg !43
  %47 = load i64, i64* %46, align 8, !dbg !43
  %48 = bitcast i64* %36 to i8*, !dbg !43
  %49 = getelementptr i8, i8* %48, i64 16, !dbg !43
  %50 = bitcast i8* %49 to i64*, !dbg !43
  store i64 %47, i64* %50, align 8, !dbg !43
  %51 = load i32, i32* %__gtid___nv_MAIN__F1L17_1__419, align 4, !dbg !48
  %52 = load i8*, i8** %.z0364_448, align 8, !dbg !48
  %53 = bitcast i8* %52 to i64*, !dbg !48
  call void @__kmpc_omp_task(i64* null, i32 %51, i64* %53), !dbg !48
  br label %L.LB2_358

L.LB2_358:                                        ; preds = %L.LB2_357
  store i32 1, i32* %.s0004_459, align 4, !dbg !49
  %54 = load i32, i32* %__gtid___nv_MAIN__F1L17_1__419, align 4, !dbg !50
  %55 = load i32, i32* %.s0004_459, align 4, !dbg !50
  %56 = bitcast void (i32, i64*)* @__nv_MAIN_F1L26_4_ to i64*, !dbg !50
  %57 = call i8* @__kmpc_omp_task_alloc(i64* null, i32 %54, i32 %55, i32 40, i32 24, i64* %56), !dbg !50
  store i8* %57, i8** %.z0364_458, align 8, !dbg !50
  %58 = load i64, i64* %__nv_MAIN__F1L17_1Arg2, align 8, !dbg !50
  %59 = load i8*, i8** %.z0364_458, align 8, !dbg !50
  %60 = bitcast i8* %59 to i64**, !dbg !50
  %61 = load i64*, i64** %60, align 8, !dbg !50
  store i64 %58, i64* %61, align 8, !dbg !50
  %62 = bitcast i64* %__nv_MAIN__F1L17_1Arg2 to i8*, !dbg !43
  %63 = getelementptr i8, i8* %62, i64 8, !dbg !43
  %64 = bitcast i8* %63 to i64*, !dbg !43
  %65 = load i64, i64* %64, align 8, !dbg !43
  %66 = bitcast i64* %61 to i8*, !dbg !43
  %67 = getelementptr i8, i8* %66, i64 8, !dbg !43
  %68 = bitcast i8* %67 to i64*, !dbg !43
  store i64 %65, i64* %68, align 8, !dbg !43
  %69 = bitcast i64* %__nv_MAIN__F1L17_1Arg2 to i8*, !dbg !43
  %70 = getelementptr i8, i8* %69, i64 16, !dbg !43
  %71 = bitcast i8* %70 to i64*, !dbg !43
  %72 = load i64, i64* %71, align 8, !dbg !43
  %73 = bitcast i64* %61 to i8*, !dbg !43
  %74 = getelementptr i8, i8* %73, i64 16, !dbg !43
  %75 = bitcast i8* %74 to i64*, !dbg !43
  store i64 %72, i64* %75, align 8, !dbg !43
  %76 = load i32, i32* %__gtid___nv_MAIN__F1L17_1__419, align 4, !dbg !50
  %77 = load i8*, i8** %.z0364_458, align 8, !dbg !50
  %78 = bitcast i8* %77 to i64*, !dbg !50
  call void @__kmpc_omp_task(i64* null, i32 %76, i64* %78), !dbg !50
  br label %L.LB2_359

L.LB2_359:                                        ; preds = %L.LB2_358
  %79 = load i32, i32* %__gtid___nv_MAIN__F1L17_1__419, align 4, !dbg !51
  store i32 %79, i32* %.s0000_414, align 4, !dbg !51
  store i32 1, i32* %.s0001_415, align 4, !dbg !51
  %80 = load i32, i32* %__gtid___nv_MAIN__F1L17_1__419, align 4, !dbg !51
  call void @__kmpc_end_single(i64* null, i32 %80), !dbg !51
  br label %L.LB2_356

L.LB2_356:                                        ; preds = %L.LB2_359, %L.LB2_315
  br label %L.LB2_331

L.LB2_331:                                        ; preds = %L.LB2_356
  %81 = load i32, i32* %__gtid___nv_MAIN__F1L17_1__419, align 4, !dbg !51
  call void @__kmpc_barrier(i64* null, i32 %81), !dbg !51
  br label %L.LB2_332

L.LB2_332:                                        ; preds = %L.LB2_331
  ret void, !dbg !43
}

define internal void @__nv_MAIN_F1L19_2_(i32 %__nv_MAIN_F1L19_2Arg0.arg, i64* %__nv_MAIN_F1L19_2Arg1) #0 !dbg !52 {
L.entry:
  %__nv_MAIN_F1L19_2Arg0.addr = alloca i32, align 4
  %.S0000_482 = alloca i8*, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L19_2Arg0.addr, metadata !55, metadata !DIExpression()), !dbg !56
  store i32 %__nv_MAIN_F1L19_2Arg0.arg, i32* %__nv_MAIN_F1L19_2Arg0.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L19_2Arg0.addr, metadata !57, metadata !DIExpression()), !dbg !56
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L19_2Arg1, metadata !58, metadata !DIExpression()), !dbg !56
  call void @llvm.dbg.value(metadata i32 1, metadata !59, metadata !DIExpression()), !dbg !56
  call void @llvm.dbg.value(metadata i32 3, metadata !60, metadata !DIExpression()), !dbg !56
  call void @llvm.dbg.value(metadata i32 0, metadata !61, metadata !DIExpression()), !dbg !56
  call void @llvm.dbg.value(metadata i32 1, metadata !62, metadata !DIExpression()), !dbg !56
  call void @llvm.dbg.value(metadata i32 3, metadata !63, metadata !DIExpression()), !dbg !56
  call void @llvm.dbg.value(metadata i32 0, metadata !64, metadata !DIExpression()), !dbg !56
  call void @llvm.dbg.value(metadata i32 1, metadata !65, metadata !DIExpression()), !dbg !56
  %0 = bitcast i64* %__nv_MAIN_F1L19_2Arg1 to i8**, !dbg !66
  %1 = load i8*, i8** %0, align 8, !dbg !66
  store i8* %1, i8** %.S0000_482, align 8, !dbg !66
  br label %L.LB4_486

L.LB4_486:                                        ; preds = %L.entry
  br label %L.LB4_320

L.LB4_320:                                        ; preds = %L.LB4_486
  %2 = bitcast i32* @.C300___nv_MAIN_F1L19_2 to i8*, !dbg !67
  %3 = bitcast void (...)* @sleep_ to void (i8*, ...)*, !dbg !67
  call void (i8*, ...) %3(i8* %2), !dbg !67
  %4 = load i8*, i8** %.S0000_482, align 8, !dbg !68
  %5 = bitcast i8* %4 to i32**, !dbg !68
  %6 = load i32*, i32** %5, align 8, !dbg !68
  store i32 1, i32* %6, align 4, !dbg !68
  br label %L.LB4_322

L.LB4_322:                                        ; preds = %L.LB4_320
  ret void, !dbg !69
}

define internal void @__nv_MAIN_F1L23_3_(i32 %__nv_MAIN_F1L23_3Arg0.arg, i64* %__nv_MAIN_F1L23_3Arg1) #0 !dbg !70 {
L.entry:
  %__nv_MAIN_F1L23_3Arg0.addr = alloca i32, align 4
  %.S0000_482 = alloca i8*, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L23_3Arg0.addr, metadata !71, metadata !DIExpression()), !dbg !72
  store i32 %__nv_MAIN_F1L23_3Arg0.arg, i32* %__nv_MAIN_F1L23_3Arg0.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L23_3Arg0.addr, metadata !73, metadata !DIExpression()), !dbg !72
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L23_3Arg1, metadata !74, metadata !DIExpression()), !dbg !72
  call void @llvm.dbg.value(metadata i32 1, metadata !75, metadata !DIExpression()), !dbg !72
  call void @llvm.dbg.value(metadata i32 3, metadata !76, metadata !DIExpression()), !dbg !72
  call void @llvm.dbg.value(metadata i32 0, metadata !77, metadata !DIExpression()), !dbg !72
  call void @llvm.dbg.value(metadata i32 1, metadata !78, metadata !DIExpression()), !dbg !72
  call void @llvm.dbg.value(metadata i32 3, metadata !79, metadata !DIExpression()), !dbg !72
  call void @llvm.dbg.value(metadata i32 0, metadata !80, metadata !DIExpression()), !dbg !72
  call void @llvm.dbg.value(metadata i32 1, metadata !81, metadata !DIExpression()), !dbg !72
  %0 = bitcast i64* %__nv_MAIN_F1L23_3Arg1 to i8**, !dbg !82
  %1 = load i8*, i8** %0, align 8, !dbg !82
  store i8* %1, i8** %.S0000_482, align 8, !dbg !82
  br label %L.LB5_493

L.LB5_493:                                        ; preds = %L.entry
  br label %L.LB5_325

L.LB5_325:                                        ; preds = %L.LB5_493
  %2 = load i8*, i8** %.S0000_482, align 8, !dbg !83
  %3 = bitcast i8* %2 to i32**, !dbg !83
  %4 = load i32*, i32** %3, align 8, !dbg !83
  %5 = load i32, i32* %4, align 4, !dbg !83
  %6 = load i8*, i8** %.S0000_482, align 8, !dbg !83
  %7 = getelementptr i8, i8* %6, i64 8, !dbg !83
  %8 = bitcast i8* %7 to i32**, !dbg !83
  %9 = load i32*, i32** %8, align 8, !dbg !83
  store i32 %5, i32* %9, align 4, !dbg !83
  br label %L.LB5_326

L.LB5_326:                                        ; preds = %L.LB5_325
  ret void, !dbg !84
}

define internal void @__nv_MAIN_F1L26_4_(i32 %__nv_MAIN_F1L26_4Arg0.arg, i64* %__nv_MAIN_F1L26_4Arg1) #0 !dbg !85 {
L.entry:
  %__nv_MAIN_F1L26_4Arg0.addr = alloca i32, align 4
  %.S0000_482 = alloca i8*, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L26_4Arg0.addr, metadata !86, metadata !DIExpression()), !dbg !87
  store i32 %__nv_MAIN_F1L26_4Arg0.arg, i32* %__nv_MAIN_F1L26_4Arg0.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L26_4Arg0.addr, metadata !88, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L26_4Arg1, metadata !89, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 1, metadata !90, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 3, metadata !91, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 0, metadata !92, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 1, metadata !93, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 3, metadata !94, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 0, metadata !95, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 1, metadata !96, metadata !DIExpression()), !dbg !87
  %0 = bitcast i64* %__nv_MAIN_F1L26_4Arg1 to i8**, !dbg !97
  %1 = load i8*, i8** %0, align 8, !dbg !97
  store i8* %1, i8** %.S0000_482, align 8, !dbg !97
  br label %L.LB6_499

L.LB6_499:                                        ; preds = %L.entry
  br label %L.LB6_329

L.LB6_329:                                        ; preds = %L.LB6_499
  %2 = load i8*, i8** %.S0000_482, align 8, !dbg !98
  %3 = bitcast i8* %2 to i32**, !dbg !98
  %4 = load i32*, i32** %3, align 8, !dbg !98
  %5 = load i32, i32* %4, align 4, !dbg !98
  %6 = load i8*, i8** %.S0000_482, align 8, !dbg !98
  %7 = getelementptr i8, i8* %6, i64 16, !dbg !98
  %8 = bitcast i8* %7 to i32**, !dbg !98
  %9 = load i32*, i32** %8, align 8, !dbg !98
  store i32 %5, i32* %9, align 4, !dbg !98
  br label %L.LB6_330

L.LB6_330:                                        ; preds = %L.LB6_329
  ret void, !dbg !99
}

declare void @sleep_(...) #0

declare void @__kmpc_barrier(i64*, i32) #0

declare void @__kmpc_end_single(i64*, i32) #0

declare void @__kmpc_omp_task(i64*, i32, i64*) #0

declare i8* @__kmpc_omp_task_alloc(i64*, i32, i32, i32, i32, i64*) #0

declare signext i32 @__kmpc_single(i64*, i32) #0

declare signext i32 @f90io_ldw_end(...) #0

declare signext i32 @f90io_sc_ch_ldw(...) #0

declare signext i32 @f90io_print_init(...) #0

declare void @_mp_ecs_nest(...) #0

declare signext i32 @f90io_fmtw_end(...) #0

declare signext i32 @f90io_sc_i_fmt_write(...) #0

declare signext i32 @f90io_fmtw_inita(...) #0

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
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB079-taskdep3-orig-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb079_taskdep3_orig_no", scope: !2, file: !3, line: 10, type: !6, scopeLine: 10, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!6 = !DISubroutineType(cc: DW_CC_program, types: !7)
!7 = !{null}
!8 = !DILocalVariable(name: "omp_sched_static", scope: !5, file: !3, type: !9)
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 0, scope: !5)
!11 = !DILocalVariable(name: "omp_sched_guided", scope: !5, file: !3, type: !9)
!12 = !DILocalVariable(name: "omp_proc_bind_false", scope: !5, file: !3, type: !9)
!13 = !DILocalVariable(name: "omp_proc_bind_true", scope: !5, file: !3, type: !9)
!14 = !DILocalVariable(name: "omp_proc_bind_close", scope: !5, file: !3, type: !9)
!15 = !DILocalVariable(name: "omp_lock_hint_none", scope: !5, file: !3, type: !9)
!16 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !5, file: !3, type: !9)
!17 = !DILocation(line: 39, column: 1, scope: !5)
!18 = !DILocation(line: 10, column: 1, scope: !5)
!19 = !DILocalVariable(name: "i", scope: !5, file: !3, type: !9)
!20 = !DILocation(line: 15, column: 1, scope: !5)
!21 = !DILocation(line: 17, column: 1, scope: !5)
!22 = !DILocalVariable(name: "j", scope: !5, file: !3, type: !9)
!23 = !DILocalVariable(name: "k", scope: !5, file: !3, type: !9)
!24 = !DILocation(line: 32, column: 1, scope: !5)
!25 = !DILocalVariable(scope: !5, file: !3, type: !9, flags: DIFlagArtificial)
!26 = !DILocation(line: 35, column: 1, scope: !5)
!27 = !DILocation(line: 36, column: 1, scope: !5)
!28 = distinct !DISubprogram(name: "__nv_MAIN__F1L17_1", scope: !2, file: !3, line: 17, type: !29, scopeLine: 17, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!29 = !DISubroutineType(types: !30)
!30 = !{null, !9, !31, !31}
!31 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!32 = !DILocalVariable(name: "__nv_MAIN__F1L17_1Arg0", arg: 1, scope: !28, file: !3, type: !9)
!33 = !DILocation(line: 0, scope: !28)
!34 = !DILocalVariable(name: "__nv_MAIN__F1L17_1Arg1", arg: 2, scope: !28, file: !3, type: !31)
!35 = !DILocalVariable(name: "__nv_MAIN__F1L17_1Arg2", arg: 3, scope: !28, file: !3, type: !31)
!36 = !DILocalVariable(name: "omp_sched_static", scope: !28, file: !3, type: !9)
!37 = !DILocalVariable(name: "omp_sched_guided", scope: !28, file: !3, type: !9)
!38 = !DILocalVariable(name: "omp_proc_bind_false", scope: !28, file: !3, type: !9)
!39 = !DILocalVariable(name: "omp_proc_bind_true", scope: !28, file: !3, type: !9)
!40 = !DILocalVariable(name: "omp_proc_bind_close", scope: !28, file: !3, type: !9)
!41 = !DILocalVariable(name: "omp_lock_hint_none", scope: !28, file: !3, type: !9)
!42 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !28, file: !3, type: !9)
!43 = !DILocation(line: 30, column: 1, scope: !28)
!44 = !DILocation(line: 18, column: 1, scope: !28)
!45 = !DILocation(line: 19, column: 1, scope: !28)
!46 = !DILocation(line: 22, column: 1, scope: !28)
!47 = !DILocation(line: 23, column: 1, scope: !28)
!48 = !DILocation(line: 25, column: 1, scope: !28)
!49 = !DILocation(line: 26, column: 1, scope: !28)
!50 = !DILocation(line: 28, column: 1, scope: !28)
!51 = !DILocation(line: 29, column: 1, scope: !28)
!52 = distinct !DISubprogram(name: "__nv_MAIN_F1L19_2", scope: !2, file: !3, line: 19, type: !53, scopeLine: 19, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!53 = !DISubroutineType(types: !54)
!54 = !{null, !9, !31}
!55 = !DILocalVariable(name: "__nv_MAIN_F1L19_2Arg0", scope: !52, file: !3, type: !9)
!56 = !DILocation(line: 0, scope: !52)
!57 = !DILocalVariable(name: "__nv_MAIN_F1L19_2Arg0", arg: 1, scope: !52, file: !3, type: !9)
!58 = !DILocalVariable(name: "__nv_MAIN_F1L19_2Arg1", arg: 2, scope: !52, file: !3, type: !31)
!59 = !DILocalVariable(name: "omp_sched_static", scope: !52, file: !3, type: !9)
!60 = !DILocalVariable(name: "omp_sched_guided", scope: !52, file: !3, type: !9)
!61 = !DILocalVariable(name: "omp_proc_bind_false", scope: !52, file: !3, type: !9)
!62 = !DILocalVariable(name: "omp_proc_bind_true", scope: !52, file: !3, type: !9)
!63 = !DILocalVariable(name: "omp_proc_bind_close", scope: !52, file: !3, type: !9)
!64 = !DILocalVariable(name: "omp_lock_hint_none", scope: !52, file: !3, type: !9)
!65 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !52, file: !3, type: !9)
!66 = !DILocation(line: 19, column: 1, scope: !52)
!67 = !DILocation(line: 20, column: 1, scope: !52)
!68 = !DILocation(line: 21, column: 1, scope: !52)
!69 = !DILocation(line: 22, column: 1, scope: !52)
!70 = distinct !DISubprogram(name: "__nv_MAIN_F1L23_3", scope: !2, file: !3, line: 23, type: !53, scopeLine: 23, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!71 = !DILocalVariable(name: "__nv_MAIN_F1L23_3Arg0", scope: !70, file: !3, type: !9)
!72 = !DILocation(line: 0, scope: !70)
!73 = !DILocalVariable(name: "__nv_MAIN_F1L23_3Arg0", arg: 1, scope: !70, file: !3, type: !9)
!74 = !DILocalVariable(name: "__nv_MAIN_F1L23_3Arg1", arg: 2, scope: !70, file: !3, type: !31)
!75 = !DILocalVariable(name: "omp_sched_static", scope: !70, file: !3, type: !9)
!76 = !DILocalVariable(name: "omp_sched_guided", scope: !70, file: !3, type: !9)
!77 = !DILocalVariable(name: "omp_proc_bind_false", scope: !70, file: !3, type: !9)
!78 = !DILocalVariable(name: "omp_proc_bind_true", scope: !70, file: !3, type: !9)
!79 = !DILocalVariable(name: "omp_proc_bind_close", scope: !70, file: !3, type: !9)
!80 = !DILocalVariable(name: "omp_lock_hint_none", scope: !70, file: !3, type: !9)
!81 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !70, file: !3, type: !9)
!82 = !DILocation(line: 23, column: 1, scope: !70)
!83 = !DILocation(line: 24, column: 1, scope: !70)
!84 = !DILocation(line: 25, column: 1, scope: !70)
!85 = distinct !DISubprogram(name: "__nv_MAIN_F1L26_4", scope: !2, file: !3, line: 26, type: !53, scopeLine: 26, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!86 = !DILocalVariable(name: "__nv_MAIN_F1L26_4Arg0", scope: !85, file: !3, type: !9)
!87 = !DILocation(line: 0, scope: !85)
!88 = !DILocalVariable(name: "__nv_MAIN_F1L26_4Arg0", arg: 1, scope: !85, file: !3, type: !9)
!89 = !DILocalVariable(name: "__nv_MAIN_F1L26_4Arg1", arg: 2, scope: !85, file: !3, type: !31)
!90 = !DILocalVariable(name: "omp_sched_static", scope: !85, file: !3, type: !9)
!91 = !DILocalVariable(name: "omp_sched_guided", scope: !85, file: !3, type: !9)
!92 = !DILocalVariable(name: "omp_proc_bind_false", scope: !85, file: !3, type: !9)
!93 = !DILocalVariable(name: "omp_proc_bind_true", scope: !85, file: !3, type: !9)
!94 = !DILocalVariable(name: "omp_proc_bind_close", scope: !85, file: !3, type: !9)
!95 = !DILocalVariable(name: "omp_lock_hint_none", scope: !85, file: !3, type: !9)
!96 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !85, file: !3, type: !9)
!97 = !DILocation(line: 26, column: 1, scope: !85)
!98 = !DILocation(line: 27, column: 1, scope: !85)
!99 = !DILocation(line: 28, column: 1, scope: !85)

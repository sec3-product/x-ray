; ModuleID = '/tmp/DRB091-threadprivate2-orig-no-06e89c.ll'
source_filename = "/tmp/DRB091-threadprivate2-orig-no-06e89c.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct__cs_unspc_ = type <{ [32 x i8] }>
%struct_drb091_3_ = type <{ [4 x i8] }>
%struct_drb091_0_ = type <{ [4 x i8] }>
%astruct.dt65 = type <{ i8*, i8*, i8* }>

@.C336_MAIN_ = internal constant [6 x i8] c"sum1 ="
@.C307_MAIN_ = internal constant i32 25
@.C306_MAIN_ = internal constant i32 14
@.C335_MAIN_ = internal constant [5 x i8] c"sum ="
@.C284_MAIN_ = internal constant i64 0
@.C332_MAIN_ = internal constant i32 6
@.C329_MAIN_ = internal constant [58 x i8] c"micro-benchmarks-fortran/DRB091-threadprivate2-orig-no.f95"
@.C331_MAIN_ = internal constant i32 42
@.C285_MAIN_ = internal constant i32 1
@.C313_MAIN_ = internal constant i32 1000
@.C283_MAIN_ = internal constant i32 0
@.C285___nv_MAIN__F1L27_1 = internal constant i32 1
@.C283___nv_MAIN__F1L27_1 = internal constant i32 0
@__cs_unspc_ = common global %struct__cs_unspc_ zeroinitializer, align 64
@_drb091_3_ = common global %struct_drb091_3_ zeroinitializer, align 64, !dbg !0
@_drb091_0_ = common global %struct_drb091_0_ zeroinitializer, align 64, !dbg !7
@TPp_drb091_3_ = common global i8* null, align 64

; Function Attrs: noinline
define float @drb091_() #0 {
.L.entry:
  ret float undef
}

define void @MAIN_() #1 !dbg !12 {
L.entry:
  %__gtid_MAIN__380 = alloca i32, align 4
  %.T0373_373 = alloca i8*, align 8
  %len_314 = alloca i32, align 4
  %sum_315 = alloca i32, align 4
  %.uplevelArgPack0001_369 = alloca %astruct.dt65, align 16
  %.dY0002_359 = alloca i32, align 4
  %i_312 = alloca i32, align 4
  %z__io_334 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !34, metadata !DIExpression()), !dbg !35
  call void @llvm.dbg.value(metadata i32 0, metadata !36, metadata !DIExpression()), !dbg !35
  call void @llvm.dbg.value(metadata i32 1, metadata !37, metadata !DIExpression()), !dbg !35
  call void @llvm.dbg.value(metadata i32 0, metadata !38, metadata !DIExpression()), !dbg !35
  call void @llvm.dbg.value(metadata i32 1, metadata !39, metadata !DIExpression()), !dbg !35
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !40
  store i32 %0, i32* %__gtid_MAIN__380, align 4, !dbg !40
  %1 = load i32, i32* %__gtid_MAIN__380, align 4, !dbg !40
  %2 = bitcast %struct_drb091_3_* @_drb091_3_ to i64*, !dbg !40
  %3 = bitcast i8** @TPp_drb091_3_ to i64*, !dbg !40
  %4 = call i8* @__kmpc_threadprivate_cached(i64* null, i32 %1, i64* %2, i64 4, i64* %3), !dbg !40
  store i8* %4, i8** %.T0373_373, align 8, !dbg !40
  %5 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !41
  %6 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !41
  call void (i8*, ...) %6(i8* %5), !dbg !41
  br label %L.LB2_362

L.LB2_362:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %len_314, metadata !42, metadata !DIExpression()), !dbg !35
  store i32 1000, i32* %len_314, align 4, !dbg !43
  call void @llvm.dbg.declare(metadata i32* %sum_315, metadata !44, metadata !DIExpression()), !dbg !35
  store i32 0, i32* %sum_315, align 4, !dbg !45
  %7 = load i8*, i8** %.T0373_373, align 8, !dbg !46
  %8 = bitcast %astruct.dt65* %.uplevelArgPack0001_369 to i8**, !dbg !46
  store i8* %7, i8** %8, align 8, !dbg !46
  %9 = bitcast i32* %len_314 to i8*, !dbg !46
  %10 = bitcast %astruct.dt65* %.uplevelArgPack0001_369 to i8*, !dbg !46
  %11 = getelementptr i8, i8* %10, i64 8, !dbg !46
  %12 = bitcast i8* %11 to i8**, !dbg !46
  store i8* %9, i8** %12, align 8, !dbg !46
  %13 = bitcast i32* %sum_315 to i8*, !dbg !46
  %14 = bitcast %astruct.dt65* %.uplevelArgPack0001_369 to i8*, !dbg !46
  %15 = getelementptr i8, i8* %14, i64 16, !dbg !46
  %16 = bitcast i8* %15 to i8**, !dbg !46
  store i8* %13, i8** %16, align 8, !dbg !46
  br label %L.LB2_378, !dbg !46

L.LB2_378:                                        ; preds = %L.LB2_362
  %17 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L27_1_ to i64*, !dbg !46
  %18 = bitcast %astruct.dt65* %.uplevelArgPack0001_369 to i64*, !dbg !46
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %17, i64* %18), !dbg !46
  %19 = load i32, i32* %len_314, align 4, !dbg !47
  call void @llvm.dbg.value(metadata i32 %19, metadata !42, metadata !DIExpression()), !dbg !35
  store i32 %19, i32* %.dY0002_359, align 4, !dbg !47
  call void @llvm.dbg.declare(metadata i32* %i_312, metadata !48, metadata !DIExpression()), !dbg !35
  store i32 1, i32* %i_312, align 4, !dbg !47
  %20 = load i32, i32* %.dY0002_359, align 4, !dbg !47
  %21 = icmp sle i32 %20, 0, !dbg !47
  br i1 %21, label %L.LB2_358, label %L.LB2_357, !dbg !47

L.LB2_357:                                        ; preds = %L.LB2_357, %L.LB2_378
  %22 = bitcast %struct_drb091_0_* @_drb091_0_ to i32*, !dbg !49
  %23 = load i32, i32* %22, align 4, !dbg !49
  %24 = load i32, i32* %i_312, align 4, !dbg !49
  call void @llvm.dbg.value(metadata i32 %24, metadata !48, metadata !DIExpression()), !dbg !35
  %25 = add nsw i32 %23, %24, !dbg !49
  %26 = bitcast %struct_drb091_0_* @_drb091_0_ to i32*, !dbg !49
  store i32 %25, i32* %26, align 4, !dbg !49
  %27 = load i32, i32* %i_312, align 4, !dbg !50
  call void @llvm.dbg.value(metadata i32 %27, metadata !48, metadata !DIExpression()), !dbg !35
  %28 = add nsw i32 %27, 1, !dbg !50
  store i32 %28, i32* %i_312, align 4, !dbg !50
  %29 = load i32, i32* %.dY0002_359, align 4, !dbg !50
  %30 = sub nsw i32 %29, 1, !dbg !50
  store i32 %30, i32* %.dY0002_359, align 4, !dbg !50
  %31 = load i32, i32* %.dY0002_359, align 4, !dbg !50
  %32 = icmp sgt i32 %31, 0, !dbg !50
  br i1 %32, label %L.LB2_357, label %L.LB2_358, !dbg !50

L.LB2_358:                                        ; preds = %L.LB2_357, %L.LB2_378
  call void (...) @_mp_bcs_nest(), !dbg !51
  %33 = bitcast i32* @.C331_MAIN_ to i8*, !dbg !51
  %34 = bitcast [58 x i8]* @.C329_MAIN_ to i8*, !dbg !51
  %35 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !51
  call void (i8*, i8*, i64, ...) %35(i8* %33, i8* %34, i64 58), !dbg !51
  %36 = bitcast i32* @.C332_MAIN_ to i8*, !dbg !51
  %37 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !51
  %38 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !51
  %39 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !51
  %40 = call i32 (i8*, i8*, i8*, i8*, ...) %39(i8* %36, i8* null, i8* %37, i8* %38), !dbg !51
  call void @llvm.dbg.declare(metadata i32* %z__io_334, metadata !52, metadata !DIExpression()), !dbg !35
  store i32 %40, i32* %z__io_334, align 4, !dbg !51
  %41 = bitcast [5 x i8]* @.C335_MAIN_ to i8*, !dbg !51
  %42 = bitcast i32 (...)* @f90io_sc_ch_ldw to i32 (i8*, i32, i64, ...)*, !dbg !51
  %43 = call i32 (i8*, i32, i64, ...) %42(i8* %41, i32 14, i64 5), !dbg !51
  store i32 %43, i32* %z__io_334, align 4, !dbg !51
  %44 = load i32, i32* %sum_315, align 4, !dbg !51
  call void @llvm.dbg.value(metadata i32 %44, metadata !44, metadata !DIExpression()), !dbg !35
  %45 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !51
  %46 = call i32 (i32, i32, ...) %45(i32 %44, i32 25), !dbg !51
  store i32 %46, i32* %z__io_334, align 4, !dbg !51
  %47 = bitcast [6 x i8]* @.C336_MAIN_ to i8*, !dbg !51
  %48 = bitcast i32 (...)* @f90io_sc_ch_ldw to i32 (i8*, i32, i64, ...)*, !dbg !51
  %49 = call i32 (i8*, i32, i64, ...) %48(i8* %47, i32 14, i64 6), !dbg !51
  store i32 %49, i32* %z__io_334, align 4, !dbg !51
  %50 = bitcast %struct_drb091_0_* @_drb091_0_ to i32*, !dbg !51
  %51 = load i32, i32* %50, align 4, !dbg !51
  %52 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !51
  %53 = call i32 (i32, i32, ...) %52(i32 %51, i32 25), !dbg !51
  store i32 %53, i32* %z__io_334, align 4, !dbg !51
  %54 = call i32 (...) @f90io_ldw_end(), !dbg !51
  store i32 %54, i32* %z__io_334, align 4, !dbg !51
  call void (...) @_mp_ecs_nest(), !dbg !51
  ret void, !dbg !40
}

define internal void @__nv_MAIN__F1L27_1_(i32* %__nv_MAIN__F1L27_1Arg0, i64* %__nv_MAIN__F1L27_1Arg1, i64* %__nv_MAIN__F1L27_1Arg2) #1 !dbg !19 {
L.entry:
  %__gtid___nv_MAIN__F1L27_1__434 = alloca i32, align 4
  %.T0431_431 = alloca i8*, align 8
  %.i0000p_321 = alloca i32, align 4
  %i_320 = alloca i32, align 4
  %.du0001p_348 = alloca i32, align 4
  %.de0001p_349 = alloca i32, align 4
  %.di0001p_350 = alloca i32, align 4
  %.ds0001p_351 = alloca i32, align 4
  %.dl0001p_353 = alloca i32, align 4
  %.dl0001p.copy_447 = alloca i32, align 4
  %.de0001p.copy_448 = alloca i32, align 4
  %.ds0001p.copy_449 = alloca i32, align 4
  %.dX0001p_352 = alloca i32, align 4
  %.dY0001p_347 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L27_1Arg0, metadata !53, metadata !DIExpression()), !dbg !54
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L27_1Arg1, metadata !55, metadata !DIExpression()), !dbg !54
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L27_1Arg2, metadata !56, metadata !DIExpression()), !dbg !54
  call void @llvm.dbg.value(metadata i32 1, metadata !57, metadata !DIExpression()), !dbg !54
  call void @llvm.dbg.value(metadata i32 0, metadata !58, metadata !DIExpression()), !dbg !54
  call void @llvm.dbg.value(metadata i32 1, metadata !59, metadata !DIExpression()), !dbg !54
  call void @llvm.dbg.value(metadata i32 0, metadata !60, metadata !DIExpression()), !dbg !54
  call void @llvm.dbg.value(metadata i32 1, metadata !61, metadata !DIExpression()), !dbg !54
  %0 = load i32, i32* %__nv_MAIN__F1L27_1Arg0, align 4, !dbg !62
  store i32 %0, i32* %__gtid___nv_MAIN__F1L27_1__434, align 4, !dbg !62
  %1 = load i32, i32* %__gtid___nv_MAIN__F1L27_1__434, align 4, !dbg !62
  %2 = bitcast %struct_drb091_3_* @_drb091_3_ to i64*, !dbg !62
  %3 = bitcast i8** @TPp_drb091_3_ to i64*, !dbg !62
  %4 = call i8* @__kmpc_threadprivate_cached(i64* null, i32 %1, i64* %2, i64 4, i64* %3), !dbg !62
  store i8* %4, i8** %.T0431_431, align 8, !dbg !62
  br label %L.LB3_427

L.LB3_427:                                        ; preds = %L.entry
  br label %L.LB3_318

L.LB3_318:                                        ; preds = %L.LB3_427
  %5 = load i8*, i8** %.T0431_431, align 8, !dbg !63
  %6 = bitcast i64* %__nv_MAIN__F1L27_1Arg2 to i8**, !dbg !63
  %7 = load i8*, i8** %6, align 8, !dbg !63
  %8 = icmp eq i8* %5, %7, !dbg !63
  br i1 %8, label %L.LB3_428, label %L.LB3_483, !dbg !63

L.LB3_483:                                        ; preds = %L.LB3_318
  %9 = load i8*, i8** %.T0431_431, align 8, !dbg !63
  %10 = bitcast i64* %__nv_MAIN__F1L27_1Arg2 to i8**, !dbg !63
  %11 = load i8*, i8** %10, align 8, !dbg !63
  %12 = call i8* @memcpy(i8* %9, i8* %11, i64 4), !dbg !63
  br label %L.LB3_428

L.LB3_428:                                        ; preds = %L.LB3_483, %L.LB3_318
  %13 = load i32, i32* %__gtid___nv_MAIN__F1L27_1__434, align 4, !dbg !63
  call void @__kmpc_barrier(i64* null, i32 %13), !dbg !63
  br label %L.LB3_319

L.LB3_319:                                        ; preds = %L.LB3_428
  store i32 0, i32* %.i0000p_321, align 4, !dbg !64
  call void @llvm.dbg.declare(metadata i32* %i_320, metadata !65, metadata !DIExpression()), !dbg !62
  store i32 1, i32* %i_320, align 4, !dbg !64
  %14 = bitcast i64* %__nv_MAIN__F1L27_1Arg2 to i8*, !dbg !64
  %15 = getelementptr i8, i8* %14, i64 8, !dbg !64
  %16 = bitcast i8* %15 to i32**, !dbg !64
  %17 = load i32*, i32** %16, align 8, !dbg !64
  %18 = load i32, i32* %17, align 4, !dbg !64
  store i32 %18, i32* %.du0001p_348, align 4, !dbg !64
  %19 = bitcast i64* %__nv_MAIN__F1L27_1Arg2 to i8*, !dbg !64
  %20 = getelementptr i8, i8* %19, i64 8, !dbg !64
  %21 = bitcast i8* %20 to i32**, !dbg !64
  %22 = load i32*, i32** %21, align 8, !dbg !64
  %23 = load i32, i32* %22, align 4, !dbg !64
  store i32 %23, i32* %.de0001p_349, align 4, !dbg !64
  store i32 1, i32* %.di0001p_350, align 4, !dbg !64
  %24 = load i32, i32* %.di0001p_350, align 4, !dbg !64
  store i32 %24, i32* %.ds0001p_351, align 4, !dbg !64
  store i32 1, i32* %.dl0001p_353, align 4, !dbg !64
  %25 = load i32, i32* %.dl0001p_353, align 4, !dbg !64
  store i32 %25, i32* %.dl0001p.copy_447, align 4, !dbg !64
  %26 = load i32, i32* %.de0001p_349, align 4, !dbg !64
  store i32 %26, i32* %.de0001p.copy_448, align 4, !dbg !64
  %27 = load i32, i32* %.ds0001p_351, align 4, !dbg !64
  store i32 %27, i32* %.ds0001p.copy_449, align 4, !dbg !64
  %28 = load i32, i32* %__gtid___nv_MAIN__F1L27_1__434, align 4, !dbg !64
  %29 = bitcast i32* %.i0000p_321 to i64*, !dbg !64
  %30 = bitcast i32* %.dl0001p.copy_447 to i64*, !dbg !64
  %31 = bitcast i32* %.de0001p.copy_448 to i64*, !dbg !64
  %32 = bitcast i32* %.ds0001p.copy_449 to i64*, !dbg !64
  %33 = load i32, i32* %.ds0001p.copy_449, align 4, !dbg !64
  call void @__kmpc_for_static_init_4(i64* null, i32 %28, i32 34, i64* %29, i64* %30, i64* %31, i64* %32, i32 %33, i32 1), !dbg !64
  %34 = load i32, i32* %.dl0001p.copy_447, align 4, !dbg !64
  store i32 %34, i32* %.dl0001p_353, align 4, !dbg !64
  %35 = load i32, i32* %.de0001p.copy_448, align 4, !dbg !64
  store i32 %35, i32* %.de0001p_349, align 4, !dbg !64
  %36 = load i32, i32* %.ds0001p.copy_449, align 4, !dbg !64
  store i32 %36, i32* %.ds0001p_351, align 4, !dbg !64
  %37 = load i32, i32* %.dl0001p_353, align 4, !dbg !64
  store i32 %37, i32* %i_320, align 4, !dbg !64
  %38 = load i32, i32* %i_320, align 4, !dbg !64
  call void @llvm.dbg.value(metadata i32 %38, metadata !65, metadata !DIExpression()), !dbg !62
  store i32 %38, i32* %.dX0001p_352, align 4, !dbg !64
  %39 = load i32, i32* %.dX0001p_352, align 4, !dbg !64
  %40 = load i32, i32* %.du0001p_348, align 4, !dbg !64
  %41 = icmp sgt i32 %39, %40, !dbg !64
  br i1 %41, label %L.LB3_346, label %L.LB3_484, !dbg !64

L.LB3_484:                                        ; preds = %L.LB3_319
  %42 = load i32, i32* %.dX0001p_352, align 4, !dbg !64
  store i32 %42, i32* %i_320, align 4, !dbg !64
  %43 = load i32, i32* %.di0001p_350, align 4, !dbg !64
  %44 = load i32, i32* %.de0001p_349, align 4, !dbg !64
  %45 = load i32, i32* %.dX0001p_352, align 4, !dbg !64
  %46 = sub nsw i32 %44, %45, !dbg !64
  %47 = add nsw i32 %43, %46, !dbg !64
  %48 = load i32, i32* %.di0001p_350, align 4, !dbg !64
  %49 = sdiv i32 %47, %48, !dbg !64
  store i32 %49, i32* %.dY0001p_347, align 4, !dbg !64
  %50 = load i32, i32* %.dY0001p_347, align 4, !dbg !64
  %51 = icmp sle i32 %50, 0, !dbg !64
  br i1 %51, label %L.LB3_356, label %L.LB3_355, !dbg !64

L.LB3_355:                                        ; preds = %L.LB3_355, %L.LB3_484
  %52 = load i32, i32* %i_320, align 4, !dbg !66
  call void @llvm.dbg.value(metadata i32 %52, metadata !65, metadata !DIExpression()), !dbg !62
  %53 = load i8*, i8** %.T0431_431, align 8, !dbg !66
  %54 = bitcast i8* %53 to i32*, !dbg !66
  %55 = load i32, i32* %54, align 4, !dbg !66
  %56 = add nsw i32 %52, %55, !dbg !66
  %57 = load i8*, i8** %.T0431_431, align 8, !dbg !66
  %58 = bitcast i8* %57 to i32*, !dbg !66
  store i32 %56, i32* %58, align 4, !dbg !66
  %59 = load i32, i32* %.di0001p_350, align 4, !dbg !67
  %60 = load i32, i32* %i_320, align 4, !dbg !67
  call void @llvm.dbg.value(metadata i32 %60, metadata !65, metadata !DIExpression()), !dbg !62
  %61 = add nsw i32 %59, %60, !dbg !67
  store i32 %61, i32* %i_320, align 4, !dbg !67
  %62 = load i32, i32* %.dY0001p_347, align 4, !dbg !67
  %63 = sub nsw i32 %62, 1, !dbg !67
  store i32 %63, i32* %.dY0001p_347, align 4, !dbg !67
  %64 = load i32, i32* %.dY0001p_347, align 4, !dbg !67
  %65 = icmp sgt i32 %64, 0, !dbg !67
  br i1 %65, label %L.LB3_355, label %L.LB3_356, !dbg !67

L.LB3_356:                                        ; preds = %L.LB3_355, %L.LB3_484
  br label %L.LB3_346

L.LB3_346:                                        ; preds = %L.LB3_356, %L.LB3_319
  %66 = load i32, i32* %__gtid___nv_MAIN__F1L27_1__434, align 4, !dbg !67
  call void @__kmpc_for_static_fini(i64* null, i32 %66), !dbg !67
  br label %L.LB3_322

L.LB3_322:                                        ; preds = %L.LB3_346
  %67 = load i32, i32* %__gtid___nv_MAIN__F1L27_1__434, align 4, !dbg !68
  call void @__kmpc_barrier(i64* null, i32 %67), !dbg !68
  %68 = load i32, i32* %__gtid___nv_MAIN__F1L27_1__434, align 4, !dbg !69
  %69 = bitcast %struct__cs_unspc_* @__cs_unspc_ to i64*, !dbg !69
  call void @__kmpc_critical(i64* null, i32 %68, i64* %69), !dbg !69
  %70 = load i8*, i8** %.T0431_431, align 8, !dbg !69
  %71 = bitcast i8* %70 to i32*, !dbg !69
  %72 = load i32, i32* %71, align 4, !dbg !69
  %73 = bitcast i64* %__nv_MAIN__F1L27_1Arg2 to i8*, !dbg !69
  %74 = getelementptr i8, i8* %73, i64 16, !dbg !69
  %75 = bitcast i8* %74 to i32**, !dbg !69
  %76 = load i32*, i32** %75, align 8, !dbg !69
  %77 = load i32, i32* %76, align 4, !dbg !69
  %78 = add nsw i32 %72, %77, !dbg !69
  %79 = bitcast i64* %__nv_MAIN__F1L27_1Arg2 to i8*, !dbg !69
  %80 = getelementptr i8, i8* %79, i64 16, !dbg !69
  %81 = bitcast i8* %80 to i32**, !dbg !69
  %82 = load i32*, i32** %81, align 8, !dbg !69
  store i32 %78, i32* %82, align 4, !dbg !69
  %83 = load i32, i32* %__gtid___nv_MAIN__F1L27_1__434, align 4, !dbg !69
  %84 = bitcast %struct__cs_unspc_* @__cs_unspc_ to i64*, !dbg !69
  call void @__kmpc_end_critical(i64* null, i32 %83, i64* %84), !dbg !69
  br label %L.LB3_327

L.LB3_327:                                        ; preds = %L.LB3_322
  ret void, !dbg !62
}

declare void @__kmpc_end_critical(i64*, i32, i64*) #1

declare void @__kmpc_critical(i64*, i32, i64*) #1

declare void @__kmpc_for_static_fini(i64*, i32) #1

declare void @__kmpc_for_static_init_4(i64*, i32, i32, i64*, i64*, i64*, i64*, i32, i32) #1

declare void @__kmpc_barrier(i64*, i32) #1

declare i8* @memcpy(i8*, i8*, i64) #1

declare void @_mp_ecs_nest(...) #1

declare signext i32 @f90io_ldw_end(...) #1

declare signext i32 @f90io_sc_i_ldw(...) #1

declare signext i32 @f90io_sc_ch_ldw(...) #1

declare signext i32 @f90io_print_init(...) #1

declare void @f90io_src_info03a(...) #1

declare void @_mp_bcs_nest(...) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

declare void @fort_init(...) #1

declare signext i32 @__kmpc_global_thread_num(i64*) #1

declare i8* @__kmpc_threadprivate_cached(i64*, i32, i64*, i64, i64*) #1

declare void @__kmpc_fork_call(i64*, i32, i64*, i64*, ...) #1

declare void @__kmpc_end_serialized_parallel(i64*, i32) #1

declare void @__kmpc_serialized_parallel(i64*, i32) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { noinline }
attributes #1 = { "no-frame-pointer-elim-non-leaf" }
attributes #2 = { nounwind readnone speculatable }

!llvm.module.flags = !{!32, !33}
!llvm.dbg.cu = !{!3}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "sum0", scope: !2, file: !4, type: !9, isLocal: false, isDefinition: true)
!2 = !DIModule(scope: !3, name: "drb091")
!3 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !4, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !6, imports: !30)
!4 = !DIFile(filename: "micro-benchmarks-fortran/DRB091-threadprivate2-orig-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!5 = !{}
!6 = !{!7, !0, !10, !17, !23}
!7 = !DIGlobalVariableExpression(var: !8, expr: !DIExpression())
!8 = distinct !DIGlobalVariable(name: "sum1", scope: !2, file: !4, type: !9, isLocal: false, isDefinition: true)
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DIGlobalVariableExpression(var: !11, expr: !DIExpression())
!11 = distinct !DIGlobalVariable(name: "TPp_drb091$3", scope: !12, file: !4, type: !15, isLocal: false, isDefinition: true)
!12 = distinct !DISubprogram(name: "drb091_threadprivate2_orig_no", scope: !3, file: !4, line: 18, type: !13, scopeLine: 18, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !3)
!13 = !DISubroutineType(cc: DW_CC_program, types: !14)
!14 = !{null}
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !16, size: 64, align: 64)
!16 = !DIBasicType(name: "any", encoding: DW_ATE_signed)
!17 = !DIGlobalVariableExpression(var: !18, expr: !DIExpression())
!18 = distinct !DIGlobalVariable(name: "TPp_drb091$3", scope: !19, file: !4, type: !15, isLocal: false, isDefinition: true)
!19 = distinct !DISubprogram(name: "__nv_MAIN__F1L27_1", scope: !3, file: !4, line: 27, type: !20, scopeLine: 27, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !3)
!20 = !DISubroutineType(types: !21)
!21 = !{null, !9, !22, !22}
!22 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!23 = !DIGlobalVariableExpression(var: !24, expr: !DIExpression())
!24 = distinct !DIGlobalVariable(name: "__cs_unspc", scope: !25, type: !26, isLocal: false, isDefinition: true)
!25 = distinct !DICommonBlock(scope: !19, declaration: !24, name: "__cs_unspc")
!26 = !DICompositeType(tag: DW_TAG_array_type, baseType: !27, size: 256, align: 8, elements: !28)
!27 = !DIBasicType(name: "byte", size: 8, align: 8, encoding: DW_ATE_signed)
!28 = !{!29}
!29 = !DISubrange(count: 32)
!30 = !{!31}
!31 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !12, entity: !2, file: !4, line: 18)
!32 = !{i32 2, !"Dwarf Version", i32 4}
!33 = !{i32 2, !"Debug Info Version", i32 3}
!34 = !DILocalVariable(name: "omp_sched_static", scope: !12, file: !4, type: !9)
!35 = !DILocation(line: 0, scope: !12)
!36 = !DILocalVariable(name: "omp_proc_bind_false", scope: !12, file: !4, type: !9)
!37 = !DILocalVariable(name: "omp_proc_bind_true", scope: !12, file: !4, type: !9)
!38 = !DILocalVariable(name: "omp_lock_hint_none", scope: !12, file: !4, type: !9)
!39 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !12, file: !4, type: !9)
!40 = !DILocation(line: 44, column: 1, scope: !12)
!41 = !DILocation(line: 18, column: 1, scope: !12)
!42 = !DILocalVariable(name: "len", scope: !12, file: !4, type: !9)
!43 = !DILocation(line: 24, column: 1, scope: !12)
!44 = !DILocalVariable(name: "sum", scope: !12, file: !4, type: !9)
!45 = !DILocation(line: 25, column: 1, scope: !12)
!46 = !DILocation(line: 27, column: 1, scope: !12)
!47 = !DILocation(line: 38, column: 1, scope: !12)
!48 = !DILocalVariable(name: "i", scope: !12, file: !4, type: !9)
!49 = !DILocation(line: 39, column: 1, scope: !12)
!50 = !DILocation(line: 40, column: 1, scope: !12)
!51 = !DILocation(line: 42, column: 1, scope: !12)
!52 = !DILocalVariable(scope: !12, file: !4, type: !9, flags: DIFlagArtificial)
!53 = !DILocalVariable(name: "__nv_MAIN__F1L27_1Arg0", arg: 1, scope: !19, file: !4, type: !9)
!54 = !DILocation(line: 0, scope: !19)
!55 = !DILocalVariable(name: "__nv_MAIN__F1L27_1Arg1", arg: 2, scope: !19, file: !4, type: !22)
!56 = !DILocalVariable(name: "__nv_MAIN__F1L27_1Arg2", arg: 3, scope: !19, file: !4, type: !22)
!57 = !DILocalVariable(name: "omp_sched_static", scope: !19, file: !4, type: !9)
!58 = !DILocalVariable(name: "omp_proc_bind_false", scope: !19, file: !4, type: !9)
!59 = !DILocalVariable(name: "omp_proc_bind_true", scope: !19, file: !4, type: !9)
!60 = !DILocalVariable(name: "omp_lock_hint_none", scope: !19, file: !4, type: !9)
!61 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !19, file: !4, type: !9)
!62 = !DILocation(line: 36, column: 1, scope: !19)
!63 = !DILocation(line: 27, column: 1, scope: !19)
!64 = !DILocation(line: 29, column: 1, scope: !19)
!65 = !DILocalVariable(name: "i", scope: !19, file: !4, type: !9)
!66 = !DILocation(line: 30, column: 1, scope: !19)
!67 = !DILocation(line: 31, column: 1, scope: !19)
!68 = !DILocation(line: 32, column: 1, scope: !19)
!69 = !DILocation(line: 34, column: 1, scope: !19)

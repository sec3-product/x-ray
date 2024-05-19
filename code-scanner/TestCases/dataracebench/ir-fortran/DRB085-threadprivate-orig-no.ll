; ModuleID = '/tmp/DRB085-threadprivate-orig-no-d9df36.ll'
source_filename = "/tmp/DRB085-threadprivate-orig-no-d9df36.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct__cs_unspc_ = type <{ [32 x i8] }>
%struct_drb085_3_ = type <{ [8 x i8] }>
%struct_drb085_2_ = type <{ [8 x i8] }>
%astruct.dt66 = type <{ i8*, i8*, i8* }>

@.C337_MAIN_ = internal constant [6 x i8] c"sum1 ="
@.C341_MAIN_ = internal constant i32 26
@.C306_MAIN_ = internal constant i32 14
@.C336_MAIN_ = internal constant [6 x i8] c"sum = "
@.C333_MAIN_ = internal constant i32 6
@.C330_MAIN_ = internal constant [57 x i8] c"micro-benchmarks-fortran/DRB085-threadprivate-orig-no.f95"
@.C332_MAIN_ = internal constant i32 47
@.C286_MAIN_ = internal constant i64 1
@.C284_MAIN_ = internal constant i64 0
@.C314_MAIN_ = internal constant i32 1000
@.C283_MAIN_ = internal constant i32 0
@.C286___nv_MAIN__F1L32_1 = internal constant i64 1
@.C283___nv_MAIN__F1L32_1 = internal constant i32 0
@__cs_unspc_ = common global %struct__cs_unspc_ zeroinitializer, align 64
@_drb085_3_ = common global %struct_drb085_3_ zeroinitializer, align 64, !dbg !0
@_drb085_2_ = common global %struct_drb085_2_ zeroinitializer, align 64, !dbg !7
@TPp_drb085_3_ = common global i8* null, align 64

; Function Attrs: noinline
define float @drb085_() #0 {
.L.entry:
  ret float undef
}

define void @drb085_foo_(i64* %i) #1 !dbg !12 {
L.entry:
  %__gtid_drb085_foo__316 = alloca i32, align 4
  %.T0312_312 = alloca i8*, align 8
  call void @llvm.dbg.declare(metadata i64* %i, metadata !39, metadata !DIExpression()), !dbg !40
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !41
  store i32 %0, i32* %__gtid_drb085_foo__316, align 4, !dbg !41
  %1 = load i32, i32* %__gtid_drb085_foo__316, align 4, !dbg !41
  %2 = bitcast %struct_drb085_3_* @_drb085_3_ to i64*, !dbg !41
  %3 = bitcast i8** @TPp_drb085_3_ to i64*, !dbg !41
  %4 = call i8* @__kmpc_threadprivate_cached(i64* null, i32 %1, i64* %2, i64 8, i64* %3), !dbg !41
  store i8* %4, i8** %.T0312_312, align 8, !dbg !41
  br label %L.LB2_308

L.LB2_308:                                        ; preds = %L.entry
  %5 = load i64, i64* %i, align 8, !dbg !42
  %6 = load i8*, i8** %.T0312_312, align 8, !dbg !42
  %7 = bitcast i8* %6 to i64*, !dbg !42
  %8 = load i64, i64* %7, align 8, !dbg !42
  %9 = add nsw i64 %5, %8, !dbg !42
  %10 = load i8*, i8** %.T0312_312, align 8, !dbg !42
  %11 = bitcast i8* %10 to i64*, !dbg !42
  store i64 %9, i64* %11, align 8, !dbg !42
  ret void, !dbg !41
}

define void @MAIN_() #1 !dbg !19 {
L.entry:
  %__gtid_MAIN__382 = alloca i32, align 4
  %.T0375_375 = alloca i8*, align 8
  %len_315 = alloca i32, align 4
  %sum_316 = alloca i64, align 8
  %.uplevelArgPack0001_371 = alloca %astruct.dt66, align 16
  %.dY0002_361 = alloca i64, align 8
  %i_313 = alloca i64, align 8
  %z__io_335 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !43, metadata !DIExpression()), !dbg !44
  call void @llvm.dbg.value(metadata i32 0, metadata !45, metadata !DIExpression()), !dbg !44
  call void @llvm.dbg.value(metadata i32 1, metadata !46, metadata !DIExpression()), !dbg !44
  call void @llvm.dbg.value(metadata i32 0, metadata !47, metadata !DIExpression()), !dbg !44
  call void @llvm.dbg.value(metadata i32 1, metadata !48, metadata !DIExpression()), !dbg !44
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !49
  store i32 %0, i32* %__gtid_MAIN__382, align 4, !dbg !49
  %1 = load i32, i32* %__gtid_MAIN__382, align 4, !dbg !49
  %2 = bitcast %struct_drb085_3_* @_drb085_3_ to i64*, !dbg !49
  %3 = bitcast i8** @TPp_drb085_3_ to i64*, !dbg !49
  %4 = call i8* @__kmpc_threadprivate_cached(i64* null, i32 %1, i64* %2, i64 8, i64* %3), !dbg !49
  store i8* %4, i8** %.T0375_375, align 8, !dbg !49
  %5 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !50
  %6 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !50
  call void (i8*, ...) %6(i8* %5), !dbg !50
  br label %L.LB3_364

L.LB3_364:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %len_315, metadata !51, metadata !DIExpression()), !dbg !44
  store i32 1000, i32* %len_315, align 4, !dbg !52
  call void @llvm.dbg.declare(metadata i64* %sum_316, metadata !53, metadata !DIExpression()), !dbg !44
  store i64 0, i64* %sum_316, align 8, !dbg !54
  %7 = load i8*, i8** %.T0375_375, align 8, !dbg !55
  %8 = bitcast %astruct.dt66* %.uplevelArgPack0001_371 to i8**, !dbg !55
  store i8* %7, i8** %8, align 8, !dbg !55
  %9 = bitcast i32* %len_315 to i8*, !dbg !55
  %10 = bitcast %astruct.dt66* %.uplevelArgPack0001_371 to i8*, !dbg !55
  %11 = getelementptr i8, i8* %10, i64 8, !dbg !55
  %12 = bitcast i8* %11 to i8**, !dbg !55
  store i8* %9, i8** %12, align 8, !dbg !55
  %13 = bitcast i64* %sum_316 to i8*, !dbg !55
  %14 = bitcast %astruct.dt66* %.uplevelArgPack0001_371 to i8*, !dbg !55
  %15 = getelementptr i8, i8* %14, i64 16, !dbg !55
  %16 = bitcast i8* %15 to i8**, !dbg !55
  store i8* %13, i8** %16, align 8, !dbg !55
  br label %L.LB3_380, !dbg !55

L.LB3_380:                                        ; preds = %L.LB3_364
  %17 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L32_1_ to i64*, !dbg !55
  %18 = bitcast %astruct.dt66* %.uplevelArgPack0001_371 to i64*, !dbg !55
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %17, i64* %18), !dbg !55
  %19 = load i32, i32* %len_315, align 4, !dbg !56
  call void @llvm.dbg.value(metadata i32 %19, metadata !51, metadata !DIExpression()), !dbg !44
  %20 = sext i32 %19 to i64, !dbg !56
  store i64 %20, i64* %.dY0002_361, align 8, !dbg !56
  call void @llvm.dbg.declare(metadata i64* %i_313, metadata !57, metadata !DIExpression()), !dbg !44
  store i64 1, i64* %i_313, align 8, !dbg !56
  %21 = load i64, i64* %.dY0002_361, align 8, !dbg !56
  %22 = icmp sle i64 %21, 0, !dbg !56
  br i1 %22, label %L.LB3_360, label %L.LB3_359, !dbg !56

L.LB3_359:                                        ; preds = %L.LB3_359, %L.LB3_380
  %23 = bitcast %struct_drb085_2_* @_drb085_2_ to i64*, !dbg !58
  %24 = load i64, i64* %23, align 8, !dbg !58
  %25 = load i64, i64* %i_313, align 8, !dbg !58
  call void @llvm.dbg.value(metadata i64 %25, metadata !57, metadata !DIExpression()), !dbg !44
  %26 = add nsw i64 %24, %25, !dbg !58
  %27 = bitcast %struct_drb085_2_* @_drb085_2_ to i64*, !dbg !58
  store i64 %26, i64* %27, align 8, !dbg !58
  %28 = load i64, i64* %i_313, align 8, !dbg !59
  call void @llvm.dbg.value(metadata i64 %28, metadata !57, metadata !DIExpression()), !dbg !44
  %29 = add nsw i64 %28, 1, !dbg !59
  store i64 %29, i64* %i_313, align 8, !dbg !59
  %30 = load i64, i64* %.dY0002_361, align 8, !dbg !59
  %31 = sub nsw i64 %30, 1, !dbg !59
  store i64 %31, i64* %.dY0002_361, align 8, !dbg !59
  %32 = load i64, i64* %.dY0002_361, align 8, !dbg !59
  %33 = icmp sgt i64 %32, 0, !dbg !59
  br i1 %33, label %L.LB3_359, label %L.LB3_360, !dbg !59

L.LB3_360:                                        ; preds = %L.LB3_359, %L.LB3_380
  call void (...) @_mp_bcs_nest(), !dbg !60
  %34 = bitcast i32* @.C332_MAIN_ to i8*, !dbg !60
  %35 = bitcast [57 x i8]* @.C330_MAIN_ to i8*, !dbg !60
  %36 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !60
  call void (i8*, i8*, i64, ...) %36(i8* %34, i8* %35, i64 57), !dbg !60
  %37 = bitcast i32* @.C333_MAIN_ to i8*, !dbg !60
  %38 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !60
  %39 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !60
  %40 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !60
  %41 = call i32 (i8*, i8*, i8*, i8*, ...) %40(i8* %37, i8* null, i8* %38, i8* %39), !dbg !60
  call void @llvm.dbg.declare(metadata i32* %z__io_335, metadata !61, metadata !DIExpression()), !dbg !44
  store i32 %41, i32* %z__io_335, align 4, !dbg !60
  %42 = bitcast [6 x i8]* @.C336_MAIN_ to i8*, !dbg !60
  %43 = bitcast i32 (...)* @f90io_sc_ch_ldw to i32 (i8*, i32, i64, ...)*, !dbg !60
  %44 = call i32 (i8*, i32, i64, ...) %43(i8* %42, i32 14, i64 6), !dbg !60
  store i32 %44, i32* %z__io_335, align 4, !dbg !60
  %45 = load i64, i64* %sum_316, align 8, !dbg !60
  call void @llvm.dbg.value(metadata i64 %45, metadata !53, metadata !DIExpression()), !dbg !44
  %46 = bitcast i32 (...)* @f90io_sc_l_ldw to i32 (i64, i32, ...)*, !dbg !60
  %47 = call i32 (i64, i32, ...) %46(i64 %45, i32 26), !dbg !60
  store i32 %47, i32* %z__io_335, align 4, !dbg !60
  %48 = bitcast [6 x i8]* @.C337_MAIN_ to i8*, !dbg !60
  %49 = bitcast i32 (...)* @f90io_sc_ch_ldw to i32 (i8*, i32, i64, ...)*, !dbg !60
  %50 = call i32 (i8*, i32, i64, ...) %49(i8* %48, i32 14, i64 6), !dbg !60
  store i32 %50, i32* %z__io_335, align 4, !dbg !60
  %51 = bitcast %struct_drb085_2_* @_drb085_2_ to i64*, !dbg !60
  %52 = load i64, i64* %51, align 8, !dbg !60
  %53 = bitcast i32 (...)* @f90io_sc_l_ldw to i32 (i64, i32, ...)*, !dbg !60
  %54 = call i32 (i64, i32, ...) %53(i64 %52, i32 26), !dbg !60
  store i32 %54, i32* %z__io_335, align 4, !dbg !60
  %55 = call i32 (...) @f90io_ldw_end(), !dbg !60
  store i32 %55, i32* %z__io_335, align 4, !dbg !60
  call void (...) @_mp_ecs_nest(), !dbg !60
  ret void, !dbg !49
}

define internal void @__nv_MAIN__F1L32_1_(i32* %__nv_MAIN__F1L32_1Arg0, i64* %__nv_MAIN__F1L32_1Arg1, i64* %__nv_MAIN__F1L32_1Arg2) #1 !dbg !24 {
L.entry:
  %__gtid___nv_MAIN__F1L32_1__434 = alloca i32, align 4
  %.T0431_431 = alloca i8*, align 8
  %.i0000p_322 = alloca i32, align 4
  %i_321 = alloca i64, align 8
  %.du0001p_350 = alloca i64, align 8
  %.de0001p_351 = alloca i64, align 8
  %.di0001p_352 = alloca i64, align 8
  %.ds0001p_353 = alloca i64, align 8
  %.dl0001p_355 = alloca i64, align 8
  %.dl0001p.copy_447 = alloca i64, align 8
  %.de0001p.copy_448 = alloca i64, align 8
  %.ds0001p.copy_449 = alloca i64, align 8
  %.dX0001p_354 = alloca i64, align 8
  %.dY0001p_349 = alloca i64, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L32_1Arg0, metadata !62, metadata !DIExpression()), !dbg !63
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L32_1Arg1, metadata !64, metadata !DIExpression()), !dbg !63
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L32_1Arg2, metadata !65, metadata !DIExpression()), !dbg !63
  call void @llvm.dbg.value(metadata i32 1, metadata !66, metadata !DIExpression()), !dbg !63
  call void @llvm.dbg.value(metadata i32 0, metadata !67, metadata !DIExpression()), !dbg !63
  call void @llvm.dbg.value(metadata i32 1, metadata !68, metadata !DIExpression()), !dbg !63
  call void @llvm.dbg.value(metadata i32 0, metadata !69, metadata !DIExpression()), !dbg !63
  call void @llvm.dbg.value(metadata i32 1, metadata !70, metadata !DIExpression()), !dbg !63
  %0 = load i32, i32* %__nv_MAIN__F1L32_1Arg0, align 4, !dbg !71
  store i32 %0, i32* %__gtid___nv_MAIN__F1L32_1__434, align 4, !dbg !71
  %1 = load i32, i32* %__gtid___nv_MAIN__F1L32_1__434, align 4, !dbg !71
  %2 = bitcast %struct_drb085_3_* @_drb085_3_ to i64*, !dbg !71
  %3 = bitcast i8** @TPp_drb085_3_ to i64*, !dbg !71
  %4 = call i8* @__kmpc_threadprivate_cached(i64* null, i32 %1, i64* %2, i64 8, i64* %3), !dbg !71
  store i8* %4, i8** %.T0431_431, align 8, !dbg !71
  br label %L.LB4_427

L.LB4_427:                                        ; preds = %L.entry
  br label %L.LB4_319

L.LB4_319:                                        ; preds = %L.LB4_427
  %5 = load i8*, i8** %.T0431_431, align 8, !dbg !72
  %6 = bitcast i64* %__nv_MAIN__F1L32_1Arg2 to i8**, !dbg !72
  %7 = load i8*, i8** %6, align 8, !dbg !72
  %8 = icmp eq i8* %5, %7, !dbg !72
  br i1 %8, label %L.LB4_428, label %L.LB4_483, !dbg !72

L.LB4_483:                                        ; preds = %L.LB4_319
  %9 = load i8*, i8** %.T0431_431, align 8, !dbg !72
  %10 = bitcast i64* %__nv_MAIN__F1L32_1Arg2 to i8**, !dbg !72
  %11 = load i8*, i8** %10, align 8, !dbg !72
  %12 = call i8* @memcpy(i8* %9, i8* %11, i64 8), !dbg !72
  br label %L.LB4_428

L.LB4_428:                                        ; preds = %L.LB4_483, %L.LB4_319
  %13 = load i32, i32* %__gtid___nv_MAIN__F1L32_1__434, align 4, !dbg !72
  call void @__kmpc_barrier(i64* null, i32 %13), !dbg !72
  br label %L.LB4_320

L.LB4_320:                                        ; preds = %L.LB4_428
  store i32 0, i32* %.i0000p_322, align 4, !dbg !73
  call void @llvm.dbg.declare(metadata i64* %i_321, metadata !74, metadata !DIExpression()), !dbg !71
  store i64 1, i64* %i_321, align 8, !dbg !73
  %14 = bitcast i64* %__nv_MAIN__F1L32_1Arg2 to i8*, !dbg !73
  %15 = getelementptr i8, i8* %14, i64 8, !dbg !73
  %16 = bitcast i8* %15 to i32**, !dbg !73
  %17 = load i32*, i32** %16, align 8, !dbg !73
  %18 = load i32, i32* %17, align 4, !dbg !73
  %19 = sext i32 %18 to i64, !dbg !73
  store i64 %19, i64* %.du0001p_350, align 8, !dbg !73
  %20 = bitcast i64* %__nv_MAIN__F1L32_1Arg2 to i8*, !dbg !73
  %21 = getelementptr i8, i8* %20, i64 8, !dbg !73
  %22 = bitcast i8* %21 to i32**, !dbg !73
  %23 = load i32*, i32** %22, align 8, !dbg !73
  %24 = load i32, i32* %23, align 4, !dbg !73
  %25 = sext i32 %24 to i64, !dbg !73
  store i64 %25, i64* %.de0001p_351, align 8, !dbg !73
  store i64 1, i64* %.di0001p_352, align 8, !dbg !73
  %26 = load i64, i64* %.di0001p_352, align 8, !dbg !73
  store i64 %26, i64* %.ds0001p_353, align 8, !dbg !73
  store i64 1, i64* %.dl0001p_355, align 8, !dbg !73
  %27 = load i64, i64* %.dl0001p_355, align 8, !dbg !73
  store i64 %27, i64* %.dl0001p.copy_447, align 8, !dbg !73
  %28 = load i64, i64* %.de0001p_351, align 8, !dbg !73
  store i64 %28, i64* %.de0001p.copy_448, align 8, !dbg !73
  %29 = load i64, i64* %.ds0001p_353, align 8, !dbg !73
  store i64 %29, i64* %.ds0001p.copy_449, align 8, !dbg !73
  %30 = load i32, i32* %__gtid___nv_MAIN__F1L32_1__434, align 4, !dbg !73
  %31 = bitcast i32* %.i0000p_322 to i64*, !dbg !73
  %32 = load i64, i64* %.ds0001p.copy_449, align 8, !dbg !73
  call void @__kmpc_for_static_init_8(i64* null, i32 %30, i32 34, i64* %31, i64* %.dl0001p.copy_447, i64* %.de0001p.copy_448, i64* %.ds0001p.copy_449, i64 %32, i64 1), !dbg !73
  %33 = load i64, i64* %.dl0001p.copy_447, align 8, !dbg !73
  store i64 %33, i64* %.dl0001p_355, align 8, !dbg !73
  %34 = load i64, i64* %.de0001p.copy_448, align 8, !dbg !73
  store i64 %34, i64* %.de0001p_351, align 8, !dbg !73
  %35 = load i64, i64* %.ds0001p.copy_449, align 8, !dbg !73
  store i64 %35, i64* %.ds0001p_353, align 8, !dbg !73
  %36 = load i64, i64* %.dl0001p_355, align 8, !dbg !73
  store i64 %36, i64* %i_321, align 8, !dbg !73
  %37 = load i64, i64* %i_321, align 8, !dbg !73
  call void @llvm.dbg.value(metadata i64 %37, metadata !74, metadata !DIExpression()), !dbg !71
  store i64 %37, i64* %.dX0001p_354, align 8, !dbg !73
  %38 = load i64, i64* %.dX0001p_354, align 8, !dbg !73
  %39 = load i64, i64* %.du0001p_350, align 8, !dbg !73
  %40 = icmp sgt i64 %38, %39, !dbg !73
  br i1 %40, label %L.LB4_348, label %L.LB4_484, !dbg !73

L.LB4_484:                                        ; preds = %L.LB4_320
  %41 = load i64, i64* %.dX0001p_354, align 8, !dbg !73
  store i64 %41, i64* %i_321, align 8, !dbg !73
  %42 = load i64, i64* %.di0001p_352, align 8, !dbg !73
  %43 = load i64, i64* %.de0001p_351, align 8, !dbg !73
  %44 = load i64, i64* %.dX0001p_354, align 8, !dbg !73
  %45 = sub nsw i64 %43, %44, !dbg !73
  %46 = add nsw i64 %42, %45, !dbg !73
  %47 = load i64, i64* %.di0001p_352, align 8, !dbg !73
  %48 = sdiv i64 %46, %47, !dbg !73
  store i64 %48, i64* %.dY0001p_349, align 8, !dbg !73
  %49 = load i64, i64* %.dY0001p_349, align 8, !dbg !73
  %50 = icmp sle i64 %49, 0, !dbg !73
  br i1 %50, label %L.LB4_358, label %L.LB4_357, !dbg !73

L.LB4_357:                                        ; preds = %L.LB4_357, %L.LB4_484
  call void @drb085_foo_(i64* %i_321), !dbg !75
  %51 = load i64, i64* %.di0001p_352, align 8, !dbg !76
  %52 = load i64, i64* %i_321, align 8, !dbg !76
  call void @llvm.dbg.value(metadata i64 %52, metadata !74, metadata !DIExpression()), !dbg !71
  %53 = add nsw i64 %51, %52, !dbg !76
  store i64 %53, i64* %i_321, align 8, !dbg !76
  %54 = load i64, i64* %.dY0001p_349, align 8, !dbg !76
  %55 = sub nsw i64 %54, 1, !dbg !76
  store i64 %55, i64* %.dY0001p_349, align 8, !dbg !76
  %56 = load i64, i64* %.dY0001p_349, align 8, !dbg !76
  %57 = icmp sgt i64 %56, 0, !dbg !76
  br i1 %57, label %L.LB4_357, label %L.LB4_358, !dbg !76

L.LB4_358:                                        ; preds = %L.LB4_357, %L.LB4_484
  br label %L.LB4_348

L.LB4_348:                                        ; preds = %L.LB4_358, %L.LB4_320
  %58 = load i32, i32* %__gtid___nv_MAIN__F1L32_1__434, align 4, !dbg !76
  call void @__kmpc_for_static_fini(i64* null, i32 %58), !dbg !76
  br label %L.LB4_323

L.LB4_323:                                        ; preds = %L.LB4_348
  %59 = load i32, i32* %__gtid___nv_MAIN__F1L32_1__434, align 4, !dbg !77
  call void @__kmpc_barrier(i64* null, i32 %59), !dbg !77
  %60 = load i32, i32* %__gtid___nv_MAIN__F1L32_1__434, align 4, !dbg !78
  %61 = bitcast %struct__cs_unspc_* @__cs_unspc_ to i64*, !dbg !78
  call void @__kmpc_critical(i64* null, i32 %60, i64* %61), !dbg !78
  %62 = load i8*, i8** %.T0431_431, align 8, !dbg !78
  %63 = bitcast i8* %62 to i64*, !dbg !78
  %64 = load i64, i64* %63, align 8, !dbg !78
  %65 = bitcast i64* %__nv_MAIN__F1L32_1Arg2 to i8*, !dbg !78
  %66 = getelementptr i8, i8* %65, i64 16, !dbg !78
  %67 = bitcast i8* %66 to i64**, !dbg !78
  %68 = load i64*, i64** %67, align 8, !dbg !78
  %69 = load i64, i64* %68, align 8, !dbg !78
  %70 = add nsw i64 %64, %69, !dbg !78
  %71 = bitcast i64* %__nv_MAIN__F1L32_1Arg2 to i8*, !dbg !78
  %72 = getelementptr i8, i8* %71, i64 16, !dbg !78
  %73 = bitcast i8* %72 to i64**, !dbg !78
  %74 = load i64*, i64** %73, align 8, !dbg !78
  store i64 %70, i64* %74, align 8, !dbg !78
  %75 = load i32, i32* %__gtid___nv_MAIN__F1L32_1__434, align 4, !dbg !78
  %76 = bitcast %struct__cs_unspc_* @__cs_unspc_ to i64*, !dbg !78
  call void @__kmpc_end_critical(i64* null, i32 %75, i64* %76), !dbg !78
  br label %L.LB4_328

L.LB4_328:                                        ; preds = %L.LB4_323
  ret void, !dbg !71
}

declare void @__kmpc_end_critical(i64*, i32, i64*) #1

declare void @__kmpc_critical(i64*, i32, i64*) #1

declare void @__kmpc_for_static_fini(i64*, i32) #1

declare void @__kmpc_for_static_init_8(i64*, i32, i32, i64*, i64*, i64*, i64*, i64, i64) #1

declare void @__kmpc_barrier(i64*, i32) #1

declare i8* @memcpy(i8*, i8*, i64) #1

declare void @_mp_ecs_nest(...) #1

declare signext i32 @f90io_ldw_end(...) #1

declare signext i32 @f90io_sc_l_ldw(...) #1

declare signext i32 @f90io_sc_ch_ldw(...) #1

declare signext i32 @f90io_print_init(...) #1

declare void @f90io_src_info03a(...) #1

declare void @_mp_bcs_nest(...) #1

declare void @fort_init(...) #1

declare void @__kmpc_fork_call(i64*, i32, i64*, i64*, ...) #1

declare void @__kmpc_end_serialized_parallel(i64*, i32) #1

declare void @__kmpc_serialized_parallel(i64*, i32) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

declare signext i32 @__kmpc_global_thread_num(i64*) #1

declare i8* @__kmpc_threadprivate_cached(i64*, i32, i64*, i64, i64*) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { noinline }
attributes #1 = { "no-frame-pointer-elim-non-leaf" }
attributes #2 = { nounwind readnone speculatable }

!llvm.module.flags = !{!37, !38}
!llvm.dbg.cu = !{!3}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "sum0", scope: !2, file: !4, type: !9, isLocal: false, isDefinition: true)
!2 = !DIModule(scope: !3, name: "drb085")
!3 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !4, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !6, imports: !35)
!4 = !DIFile(filename: "micro-benchmarks-fortran/DRB085-threadprivate-orig-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!5 = !{}
!6 = !{!7, !0, !10, !17, !22, !28}
!7 = !DIGlobalVariableExpression(var: !8, expr: !DIExpression())
!8 = distinct !DIGlobalVariable(name: "sum1", scope: !2, file: !4, type: !9, isLocal: false, isDefinition: true)
!9 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!10 = !DIGlobalVariableExpression(var: !11, expr: !DIExpression())
!11 = distinct !DIGlobalVariable(name: "TPp_drb085$3", scope: !12, file: !4, type: !15, isLocal: false, isDefinition: true)
!12 = distinct !DISubprogram(name: "foo", scope: !2, file: !4, line: 16, type: !13, scopeLine: 16, spFlags: DISPFlagDefinition, unit: !3)
!13 = !DISubroutineType(types: !14)
!14 = !{null, !9}
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !16, size: 64, align: 64)
!16 = !DIBasicType(name: "any", encoding: DW_ATE_signed)
!17 = !DIGlobalVariableExpression(var: !18, expr: !DIExpression())
!18 = distinct !DIGlobalVariable(name: "TPp_drb085$3", scope: !19, file: !4, type: !15, isLocal: false, isDefinition: true)
!19 = distinct !DISubprogram(name: "drb085_threadprivate_orig_no", scope: !3, file: !4, line: 22, type: !20, scopeLine: 22, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !3)
!20 = !DISubroutineType(cc: DW_CC_program, types: !21)
!21 = !{null}
!22 = !DIGlobalVariableExpression(var: !23, expr: !DIExpression())
!23 = distinct !DIGlobalVariable(name: "TPp_drb085$3", scope: !24, file: !4, type: !15, isLocal: false, isDefinition: true)
!24 = distinct !DISubprogram(name: "__nv_MAIN__F1L32_1", scope: !3, file: !4, line: 32, type: !25, scopeLine: 32, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !3)
!25 = !DISubroutineType(types: !26)
!26 = !{null, !27, !9, !9}
!27 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!28 = !DIGlobalVariableExpression(var: !29, expr: !DIExpression())
!29 = distinct !DIGlobalVariable(name: "__cs_unspc", scope: !30, type: !31, isLocal: false, isDefinition: true)
!30 = distinct !DICommonBlock(scope: !24, declaration: !29, name: "__cs_unspc")
!31 = !DICompositeType(tag: DW_TAG_array_type, baseType: !32, size: 256, align: 8, elements: !33)
!32 = !DIBasicType(name: "byte", size: 8, align: 8, encoding: DW_ATE_signed)
!33 = !{!34}
!34 = !DISubrange(count: 32)
!35 = !{!36}
!36 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !19, entity: !2, file: !4, line: 22)
!37 = !{i32 2, !"Dwarf Version", i32 4}
!38 = !{i32 2, !"Debug Info Version", i32 3}
!39 = !DILocalVariable(name: "i", arg: 1, scope: !12, file: !4, type: !9)
!40 = !DILocation(line: 0, scope: !12)
!41 = !DILocation(line: 19, column: 1, scope: !12)
!42 = !DILocation(line: 18, column: 1, scope: !12)
!43 = !DILocalVariable(name: "omp_sched_static", scope: !19, file: !4, type: !27)
!44 = !DILocation(line: 0, scope: !19)
!45 = !DILocalVariable(name: "omp_proc_bind_false", scope: !19, file: !4, type: !27)
!46 = !DILocalVariable(name: "omp_proc_bind_true", scope: !19, file: !4, type: !27)
!47 = !DILocalVariable(name: "omp_lock_hint_none", scope: !19, file: !4, type: !27)
!48 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !19, file: !4, type: !27)
!49 = !DILocation(line: 48, column: 1, scope: !19)
!50 = !DILocation(line: 22, column: 1, scope: !19)
!51 = !DILocalVariable(name: "len", scope: !19, file: !4, type: !27)
!52 = !DILocation(line: 29, column: 1, scope: !19)
!53 = !DILocalVariable(name: "sum", scope: !19, file: !4, type: !9)
!54 = !DILocation(line: 30, column: 1, scope: !19)
!55 = !DILocation(line: 32, column: 1, scope: !19)
!56 = !DILocation(line: 43, column: 1, scope: !19)
!57 = !DILocalVariable(name: "i", scope: !19, file: !4, type: !9)
!58 = !DILocation(line: 44, column: 1, scope: !19)
!59 = !DILocation(line: 45, column: 1, scope: !19)
!60 = !DILocation(line: 47, column: 1, scope: !19)
!61 = !DILocalVariable(scope: !19, file: !4, type: !27, flags: DIFlagArtificial)
!62 = !DILocalVariable(name: "__nv_MAIN__F1L32_1Arg0", arg: 1, scope: !24, file: !4, type: !27)
!63 = !DILocation(line: 0, scope: !24)
!64 = !DILocalVariable(name: "__nv_MAIN__F1L32_1Arg1", arg: 2, scope: !24, file: !4, type: !9)
!65 = !DILocalVariable(name: "__nv_MAIN__F1L32_1Arg2", arg: 3, scope: !24, file: !4, type: !9)
!66 = !DILocalVariable(name: "omp_sched_static", scope: !24, file: !4, type: !27)
!67 = !DILocalVariable(name: "omp_proc_bind_false", scope: !24, file: !4, type: !27)
!68 = !DILocalVariable(name: "omp_proc_bind_true", scope: !24, file: !4, type: !27)
!69 = !DILocalVariable(name: "omp_lock_hint_none", scope: !24, file: !4, type: !27)
!70 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !24, file: !4, type: !27)
!71 = !DILocation(line: 41, column: 1, scope: !24)
!72 = !DILocation(line: 32, column: 1, scope: !24)
!73 = !DILocation(line: 34, column: 1, scope: !24)
!74 = !DILocalVariable(name: "i", scope: !24, file: !4, type: !9)
!75 = !DILocation(line: 35, column: 1, scope: !24)
!76 = !DILocation(line: 36, column: 1, scope: !24)
!77 = !DILocation(line: 37, column: 1, scope: !24)
!78 = !DILocation(line: 39, column: 1, scope: !24)

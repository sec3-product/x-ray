; ModuleID = '/tmp/DRB164-simdmissinglock1-orig-gpu-yes-c812a4.ll'
source_filename = "/tmp/DRB164-simdmissinglock1-orig-gpu-yes-c812a4.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct_drb163_0_ = type <{ [72 x i8] }>
%astruct.dt157 = type <{ i8*, i8* }>

@.C307_MAIN_ = internal constant i32 25
@.C310_MAIN_ = internal constant i64 16
@.C284_MAIN_ = internal constant i64 0
@.C346_MAIN_ = internal constant i32 6
@.C343_MAIN_ = internal constant [65 x i8] c"micro-benchmarks-fortran/DRB164-simdmissinglock1-orig-gpu-yes.f95"
@.C345_MAIN_ = internal constant i32 37
@.C308_MAIN_ = internal constant i32 17
@.C309_MAIN_ = internal constant i32 20
@.C306_MAIN_ = internal constant i32 16
@.C285_MAIN_ = internal constant i32 1
@.C283_MAIN_ = internal constant i32 0
@.C306___nv_MAIN__F1L25_1 = internal constant i32 16
@.C308___nv_MAIN__F1L25_1 = internal constant i32 17
@.C309___nv_MAIN__F1L25_1 = internal constant i32 20
@.C285___nv_MAIN__F1L25_1 = internal constant i32 1
@.C283___nv_MAIN__F1L25_1 = internal constant i32 0
@.C306___nv_MAIN_F1L26_2 = internal constant i32 16
@.C308___nv_MAIN_F1L26_2 = internal constant i32 17
@.C309___nv_MAIN_F1L26_2 = internal constant i32 20
@.C285___nv_MAIN_F1L26_2 = internal constant i32 1
@.C283___nv_MAIN_F1L26_2 = internal constant i32 0
@.C306___nv_MAIN_F1L27_3 = internal constant i32 16
@.C308___nv_MAIN_F1L27_3 = internal constant i32 17
@.C285___nv_MAIN_F1L27_3 = internal constant i32 1
@.C283___nv_MAIN_F1L27_3 = internal constant i32 0
@_drb163_0_ = common global %struct_drb163_0_ zeroinitializer, align 64, !dbg !0, !dbg !7, !dbg !10

; Function Attrs: noinline
define float @drb163_() #0 {
.L.entry:
  ret float undef
}

define void @MAIN_() #1 !dbg !14 {
L.entry:
  %__gtid_MAIN__401 = alloca i32, align 4
  %.dY0001_358 = alloca i32, align 4
  %z__io_348 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !22, metadata !DIExpression()), !dbg !23
  call void @llvm.dbg.value(metadata i32 0, metadata !24, metadata !DIExpression()), !dbg !23
  call void @llvm.dbg.value(metadata i32 1, metadata !25, metadata !DIExpression()), !dbg !23
  call void @llvm.dbg.value(metadata i32 0, metadata !26, metadata !DIExpression()), !dbg !23
  call void @llvm.dbg.value(metadata i32 1, metadata !27, metadata !DIExpression()), !dbg !23
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !28
  store i32 %0, i32* %__gtid_MAIN__401, align 4, !dbg !28
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !29
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !29
  call void (i8*, ...) %2(i8* %1), !dbg !29
  br label %L.LB2_387

L.LB2_387:                                        ; preds = %L.entry
  store i32 16, i32* %.dY0001_358, align 4, !dbg !30
  %3 = bitcast %struct_drb163_0_* @_drb163_0_ to i8*, !dbg !30
  %4 = getelementptr i8, i8* %3, i64 64, !dbg !30
  %5 = bitcast i8* %4 to i32*, !dbg !30
  store i32 1, i32* %5, align 4, !dbg !30
  br label %L.LB2_356

L.LB2_356:                                        ; preds = %L.LB2_356, %L.LB2_387
  %6 = bitcast %struct_drb163_0_* @_drb163_0_ to i8*, !dbg !31
  %7 = getelementptr i8, i8* %6, i64 64, !dbg !31
  %8 = bitcast i8* %7 to i32*, !dbg !31
  %9 = load i32, i32* %8, align 4, !dbg !31
  %10 = sext i32 %9 to i64, !dbg !31
  %11 = bitcast %struct_drb163_0_* @_drb163_0_ to i8*, !dbg !31
  %12 = getelementptr i8, i8* %11, i64 -4, !dbg !31
  %13 = bitcast i8* %12 to i32*, !dbg !31
  %14 = getelementptr i32, i32* %13, i64 %10, !dbg !31
  store i32 0, i32* %14, align 4, !dbg !31
  %15 = bitcast %struct_drb163_0_* @_drb163_0_ to i8*, !dbg !32
  %16 = getelementptr i8, i8* %15, i64 64, !dbg !32
  %17 = bitcast i8* %16 to i32*, !dbg !32
  %18 = load i32, i32* %17, align 4, !dbg !32
  %19 = add nsw i32 %18, 1, !dbg !32
  %20 = bitcast %struct_drb163_0_* @_drb163_0_ to i8*, !dbg !32
  %21 = getelementptr i8, i8* %20, i64 64, !dbg !32
  %22 = bitcast i8* %21 to i32*, !dbg !32
  store i32 %19, i32* %22, align 4, !dbg !32
  %23 = load i32, i32* %.dY0001_358, align 4, !dbg !32
  %24 = sub nsw i32 %23, 1, !dbg !32
  store i32 %24, i32* %.dY0001_358, align 4, !dbg !32
  %25 = load i32, i32* %.dY0001_358, align 4, !dbg !32
  %26 = icmp sgt i32 %25, 0, !dbg !32
  br i1 %26, label %L.LB2_356, label %L.LB2_419, !dbg !32

L.LB2_419:                                        ; preds = %L.LB2_356
  call void @__nv_MAIN__F1L25_1_(i32* %__gtid_MAIN__401, i64* null, i64* null), !dbg !33
  call void (...) @_mp_bcs_nest(), !dbg !34
  %27 = bitcast i32* @.C345_MAIN_ to i8*, !dbg !34
  %28 = bitcast [65 x i8]* @.C343_MAIN_ to i8*, !dbg !34
  %29 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !34
  call void (i8*, i8*, i64, ...) %29(i8* %27, i8* %28, i64 65), !dbg !34
  %30 = bitcast i32* @.C346_MAIN_ to i8*, !dbg !34
  %31 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !34
  %32 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !34
  %33 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !34
  %34 = call i32 (i8*, i8*, i8*, i8*, ...) %33(i8* %30, i8* null, i8* %31, i8* %32), !dbg !34
  call void @llvm.dbg.declare(metadata i32* %z__io_348, metadata !35, metadata !DIExpression()), !dbg !23
  store i32 %34, i32* %z__io_348, align 4, !dbg !34
  %35 = bitcast %struct_drb163_0_* @_drb163_0_ to i8*, !dbg !34
  %36 = getelementptr i8, i8* %35, i64 60, !dbg !34
  %37 = bitcast i8* %36 to i32*, !dbg !34
  %38 = load i32, i32* %37, align 4, !dbg !34
  %39 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !34
  %40 = call i32 (i32, i32, ...) %39(i32 %38, i32 25), !dbg !34
  store i32 %40, i32* %z__io_348, align 4, !dbg !34
  %41 = call i32 (...) @f90io_ldw_end(), !dbg !34
  store i32 %41, i32* %z__io_348, align 4, !dbg !34
  call void (...) @_mp_ecs_nest(), !dbg !34
  ret void, !dbg !28
}

define internal void @__nv_MAIN__F1L25_1_(i32* %__nv_MAIN__F1L25_1Arg0, i64* %__nv_MAIN__F1L25_1Arg1, i64* %__nv_MAIN__F1L25_1Arg2) #1 !dbg !36 {
L.entry:
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L25_1Arg0, metadata !40, metadata !DIExpression()), !dbg !41
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L25_1Arg1, metadata !42, metadata !DIExpression()), !dbg !41
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L25_1Arg2, metadata !43, metadata !DIExpression()), !dbg !41
  call void @llvm.dbg.value(metadata i32 1, metadata !44, metadata !DIExpression()), !dbg !41
  call void @llvm.dbg.value(metadata i32 0, metadata !45, metadata !DIExpression()), !dbg !41
  call void @llvm.dbg.value(metadata i32 1, metadata !46, metadata !DIExpression()), !dbg !41
  call void @llvm.dbg.value(metadata i32 0, metadata !47, metadata !DIExpression()), !dbg !41
  call void @llvm.dbg.value(metadata i32 1, metadata !48, metadata !DIExpression()), !dbg !41
  br label %L.LB3_423

L.LB3_423:                                        ; preds = %L.entry
  br label %L.LB3_317

L.LB3_317:                                        ; preds = %L.LB3_423
  %0 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN_F1L26_2_ to i64*, !dbg !49
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_teams(i64* null, i32 1, i64* %0, i64* %__nv_MAIN__F1L25_1Arg2), !dbg !49
  br label %L.LB3_341

L.LB3_341:                                        ; preds = %L.LB3_317
  ret void, !dbg !50
}

define internal void @__nv_MAIN_F1L26_2_(i32* %__nv_MAIN_F1L26_2Arg0, i64* %__nv_MAIN_F1L26_2Arg1, i64* %__nv_MAIN_F1L26_2Arg2) #1 !dbg !51 {
L.entry:
  %__gtid___nv_MAIN_F1L26_2__458 = alloca i32, align 4
  %.i0000p_324 = alloca i32, align 4
  %.i0001p_325 = alloca i32, align 4
  %.i0002p_326 = alloca i32, align 4
  %.i0003p_327 = alloca i32, align 4
  %i_323 = alloca i32, align 4
  %.du0002_362 = alloca i32, align 4
  %.de0002_363 = alloca i32, align 4
  %.di0002_364 = alloca i32, align 4
  %.ds0002_365 = alloca i32, align 4
  %.dl0002_367 = alloca i32, align 4
  %.dl0002.copy_452 = alloca i32, align 4
  %.de0002.copy_453 = alloca i32, align 4
  %.ds0002.copy_454 = alloca i32, align 4
  %.dX0002_366 = alloca i32, align 4
  %.dY0002_361 = alloca i32, align 4
  %.uplevelArgPack0001_477 = alloca %astruct.dt157, align 16
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L26_2Arg0, metadata !52, metadata !DIExpression()), !dbg !53
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L26_2Arg1, metadata !54, metadata !DIExpression()), !dbg !53
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L26_2Arg2, metadata !55, metadata !DIExpression()), !dbg !53
  call void @llvm.dbg.value(metadata i32 1, metadata !56, metadata !DIExpression()), !dbg !53
  call void @llvm.dbg.value(metadata i32 0, metadata !57, metadata !DIExpression()), !dbg !53
  call void @llvm.dbg.value(metadata i32 1, metadata !58, metadata !DIExpression()), !dbg !53
  call void @llvm.dbg.value(metadata i32 0, metadata !59, metadata !DIExpression()), !dbg !53
  call void @llvm.dbg.value(metadata i32 1, metadata !60, metadata !DIExpression()), !dbg !53
  %0 = load i32, i32* %__nv_MAIN_F1L26_2Arg0, align 4, !dbg !61
  store i32 %0, i32* %__gtid___nv_MAIN_F1L26_2__458, align 4, !dbg !61
  br label %L.LB5_441

L.LB5_441:                                        ; preds = %L.entry
  br label %L.LB5_320

L.LB5_320:                                        ; preds = %L.LB5_441
  br label %L.LB5_321

L.LB5_321:                                        ; preds = %L.LB5_320
  br label %L.LB5_322

L.LB5_322:                                        ; preds = %L.LB5_321
  store i32 0, i32* %.i0000p_324, align 4, !dbg !62
  store i32 1, i32* %.i0001p_325, align 4, !dbg !62
  store i32 20, i32* %.i0002p_326, align 4, !dbg !62
  store i32 1, i32* %.i0003p_327, align 4, !dbg !62
  %1 = load i32, i32* %.i0001p_325, align 4, !dbg !62
  call void @llvm.dbg.declare(metadata i32* %i_323, metadata !63, metadata !DIExpression()), !dbg !61
  store i32 %1, i32* %i_323, align 4, !dbg !62
  %2 = load i32, i32* %.i0002p_326, align 4, !dbg !62
  store i32 %2, i32* %.du0002_362, align 4, !dbg !62
  %3 = load i32, i32* %.i0002p_326, align 4, !dbg !62
  store i32 %3, i32* %.de0002_363, align 4, !dbg !62
  store i32 1, i32* %.di0002_364, align 4, !dbg !62
  %4 = load i32, i32* %.di0002_364, align 4, !dbg !62
  store i32 %4, i32* %.ds0002_365, align 4, !dbg !62
  %5 = load i32, i32* %.i0001p_325, align 4, !dbg !62
  store i32 %5, i32* %.dl0002_367, align 4, !dbg !62
  %6 = load i32, i32* %.dl0002_367, align 4, !dbg !62
  store i32 %6, i32* %.dl0002.copy_452, align 4, !dbg !62
  %7 = load i32, i32* %.de0002_363, align 4, !dbg !62
  store i32 %7, i32* %.de0002.copy_453, align 4, !dbg !62
  %8 = load i32, i32* %.ds0002_365, align 4, !dbg !62
  store i32 %8, i32* %.ds0002.copy_454, align 4, !dbg !62
  %9 = load i32, i32* %__gtid___nv_MAIN_F1L26_2__458, align 4, !dbg !62
  %10 = bitcast i32* %.i0000p_324 to i64*, !dbg !62
  %11 = bitcast i32* %.dl0002.copy_452 to i64*, !dbg !62
  %12 = bitcast i32* %.de0002.copy_453 to i64*, !dbg !62
  %13 = bitcast i32* %.ds0002.copy_454 to i64*, !dbg !62
  %14 = load i32, i32* %.ds0002.copy_454, align 4, !dbg !62
  call void @__kmpc_for_static_init_4(i64* null, i32 %9, i32 92, i64* %10, i64* %11, i64* %12, i64* %13, i32 %14, i32 1), !dbg !62
  %15 = load i32, i32* %.dl0002.copy_452, align 4, !dbg !62
  store i32 %15, i32* %.dl0002_367, align 4, !dbg !62
  %16 = load i32, i32* %.de0002.copy_453, align 4, !dbg !62
  store i32 %16, i32* %.de0002_363, align 4, !dbg !62
  %17 = load i32, i32* %.ds0002.copy_454, align 4, !dbg !62
  store i32 %17, i32* %.ds0002_365, align 4, !dbg !62
  %18 = load i32, i32* %.dl0002_367, align 4, !dbg !62
  store i32 %18, i32* %i_323, align 4, !dbg !62
  %19 = load i32, i32* %i_323, align 4, !dbg !62
  call void @llvm.dbg.value(metadata i32 %19, metadata !63, metadata !DIExpression()), !dbg !61
  store i32 %19, i32* %.dX0002_366, align 4, !dbg !62
  %20 = load i32, i32* %.dX0002_366, align 4, !dbg !62
  %21 = load i32, i32* %.du0002_362, align 4, !dbg !62
  %22 = icmp sgt i32 %20, %21, !dbg !62
  br i1 %22, label %L.LB5_360, label %L.LB5_504, !dbg !62

L.LB5_504:                                        ; preds = %L.LB5_322
  %23 = load i32, i32* %.du0002_362, align 4, !dbg !62
  %24 = load i32, i32* %.de0002_363, align 4, !dbg !62
  %25 = icmp slt i32 %23, %24, !dbg !62
  %26 = select i1 %25, i32 %23, i32 %24, !dbg !62
  store i32 %26, i32* %.de0002_363, align 4, !dbg !62
  %27 = load i32, i32* %.dX0002_366, align 4, !dbg !62
  store i32 %27, i32* %i_323, align 4, !dbg !62
  %28 = load i32, i32* %.di0002_364, align 4, !dbg !62
  %29 = load i32, i32* %.de0002_363, align 4, !dbg !62
  %30 = load i32, i32* %.dX0002_366, align 4, !dbg !62
  %31 = sub nsw i32 %29, %30, !dbg !62
  %32 = add nsw i32 %28, %31, !dbg !62
  %33 = load i32, i32* %.di0002_364, align 4, !dbg !62
  %34 = sdiv i32 %32, %33, !dbg !62
  store i32 %34, i32* %.dY0002_361, align 4, !dbg !62
  %35 = load i32, i32* %i_323, align 4, !dbg !62
  call void @llvm.dbg.value(metadata i32 %35, metadata !63, metadata !DIExpression()), !dbg !61
  store i32 %35, i32* %.i0001p_325, align 4, !dbg !62
  %36 = load i32, i32* %.de0002_363, align 4, !dbg !62
  store i32 %36, i32* %.i0002p_326, align 4, !dbg !62
  %37 = bitcast i32* %.i0001p_325 to i8*, !dbg !62
  %38 = bitcast %astruct.dt157* %.uplevelArgPack0001_477 to i8**, !dbg !62
  store i8* %37, i8** %38, align 8, !dbg !62
  %39 = bitcast i32* %.i0002p_326 to i8*, !dbg !62
  %40 = bitcast %astruct.dt157* %.uplevelArgPack0001_477 to i8*, !dbg !62
  %41 = getelementptr i8, i8* %40, i64 8, !dbg !62
  %42 = bitcast i8* %41 to i8**, !dbg !62
  store i8* %39, i8** %42, align 8, !dbg !62
  br label %L.LB5_482, !dbg !62

L.LB5_482:                                        ; preds = %L.LB5_504
  %43 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN_F1L27_3_ to i64*, !dbg !62
  %44 = bitcast %astruct.dt157* %.uplevelArgPack0001_477 to i64*, !dbg !62
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %43, i64* %44), !dbg !62
  br label %L.LB5_360

L.LB5_360:                                        ; preds = %L.LB5_482, %L.LB5_322
  %45 = load i32, i32* %__gtid___nv_MAIN_F1L26_2__458, align 4, !dbg !64
  call void @__kmpc_for_static_fini(i64* null, i32 %45), !dbg !64
  br label %L.LB5_338

L.LB5_338:                                        ; preds = %L.LB5_360
  br label %L.LB5_339

L.LB5_339:                                        ; preds = %L.LB5_338
  br label %L.LB5_340

L.LB5_340:                                        ; preds = %L.LB5_339
  ret void, !dbg !61
}

define internal void @__nv_MAIN_F1L27_3_(i32* %__nv_MAIN_F1L27_3Arg0, i64* %__nv_MAIN_F1L27_3Arg1, i64* %__nv_MAIN_F1L27_3Arg2) #1 !dbg !65 {
L.entry:
  %__gtid___nv_MAIN_F1L27_3__524 = alloca i32, align 4
  %.i0004p_332 = alloca i32, align 4
  %i_331 = alloca i32, align 4
  %.du0003p_374 = alloca i32, align 4
  %.de0003p_375 = alloca i32, align 4
  %.di0003p_376 = alloca i32, align 4
  %.ds0003p_377 = alloca i32, align 4
  %.dl0003p_379 = alloca i32, align 4
  %.dl0003p.copy_518 = alloca i32, align 4
  %.de0003p.copy_519 = alloca i32, align 4
  %.ds0003p.copy_520 = alloca i32, align 4
  %.dX0003p_378 = alloca i32, align 4
  %.dY0003p_373 = alloca i32, align 4
  %.i0005p_335 = alloca i32, align 4
  %.dY0004p_385 = alloca i32, align 4
  %j_334 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L27_3Arg0, metadata !66, metadata !DIExpression()), !dbg !67
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L27_3Arg1, metadata !68, metadata !DIExpression()), !dbg !67
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L27_3Arg2, metadata !69, metadata !DIExpression()), !dbg !67
  call void @llvm.dbg.value(metadata i32 1, metadata !70, metadata !DIExpression()), !dbg !67
  call void @llvm.dbg.value(metadata i32 0, metadata !71, metadata !DIExpression()), !dbg !67
  call void @llvm.dbg.value(metadata i32 1, metadata !72, metadata !DIExpression()), !dbg !67
  call void @llvm.dbg.value(metadata i32 0, metadata !73, metadata !DIExpression()), !dbg !67
  call void @llvm.dbg.value(metadata i32 1, metadata !74, metadata !DIExpression()), !dbg !67
  %0 = load i32, i32* %__nv_MAIN_F1L27_3Arg0, align 4, !dbg !75
  store i32 %0, i32* %__gtid___nv_MAIN_F1L27_3__524, align 4, !dbg !75
  br label %L.LB7_508

L.LB7_508:                                        ; preds = %L.entry
  br label %L.LB7_330

L.LB7_330:                                        ; preds = %L.LB7_508
  store i32 0, i32* %.i0004p_332, align 4, !dbg !76
  %1 = bitcast i64* %__nv_MAIN_F1L27_3Arg2 to i32**, !dbg !76
  %2 = load i32*, i32** %1, align 8, !dbg !76
  %3 = load i32, i32* %2, align 4, !dbg !76
  call void @llvm.dbg.declare(metadata i32* %i_331, metadata !77, metadata !DIExpression()), !dbg !75
  store i32 %3, i32* %i_331, align 4, !dbg !76
  %4 = bitcast i64* %__nv_MAIN_F1L27_3Arg2 to i8*, !dbg !76
  %5 = getelementptr i8, i8* %4, i64 8, !dbg !76
  %6 = bitcast i8* %5 to i32**, !dbg !76
  %7 = load i32*, i32** %6, align 8, !dbg !76
  %8 = load i32, i32* %7, align 4, !dbg !76
  store i32 %8, i32* %.du0003p_374, align 4, !dbg !76
  %9 = bitcast i64* %__nv_MAIN_F1L27_3Arg2 to i8*, !dbg !76
  %10 = getelementptr i8, i8* %9, i64 8, !dbg !76
  %11 = bitcast i8* %10 to i32**, !dbg !76
  %12 = load i32*, i32** %11, align 8, !dbg !76
  %13 = load i32, i32* %12, align 4, !dbg !76
  store i32 %13, i32* %.de0003p_375, align 4, !dbg !76
  store i32 1, i32* %.di0003p_376, align 4, !dbg !76
  %14 = load i32, i32* %.di0003p_376, align 4, !dbg !76
  store i32 %14, i32* %.ds0003p_377, align 4, !dbg !76
  %15 = bitcast i64* %__nv_MAIN_F1L27_3Arg2 to i32**, !dbg !76
  %16 = load i32*, i32** %15, align 8, !dbg !76
  %17 = load i32, i32* %16, align 4, !dbg !76
  store i32 %17, i32* %.dl0003p_379, align 4, !dbg !76
  %18 = load i32, i32* %.dl0003p_379, align 4, !dbg !76
  store i32 %18, i32* %.dl0003p.copy_518, align 4, !dbg !76
  %19 = load i32, i32* %.de0003p_375, align 4, !dbg !76
  store i32 %19, i32* %.de0003p.copy_519, align 4, !dbg !76
  %20 = load i32, i32* %.ds0003p_377, align 4, !dbg !76
  store i32 %20, i32* %.ds0003p.copy_520, align 4, !dbg !76
  %21 = load i32, i32* %__gtid___nv_MAIN_F1L27_3__524, align 4, !dbg !76
  %22 = bitcast i32* %.i0004p_332 to i64*, !dbg !76
  %23 = bitcast i32* %.dl0003p.copy_518 to i64*, !dbg !76
  %24 = bitcast i32* %.de0003p.copy_519 to i64*, !dbg !76
  %25 = bitcast i32* %.ds0003p.copy_520 to i64*, !dbg !76
  %26 = load i32, i32* %.ds0003p.copy_520, align 4, !dbg !76
  call void @__kmpc_for_static_init_4(i64* null, i32 %21, i32 34, i64* %22, i64* %23, i64* %24, i64* %25, i32 %26, i32 1), !dbg !76
  %27 = load i32, i32* %.dl0003p.copy_518, align 4, !dbg !76
  store i32 %27, i32* %.dl0003p_379, align 4, !dbg !76
  %28 = load i32, i32* %.de0003p.copy_519, align 4, !dbg !76
  store i32 %28, i32* %.de0003p_375, align 4, !dbg !76
  %29 = load i32, i32* %.ds0003p.copy_520, align 4, !dbg !76
  store i32 %29, i32* %.ds0003p_377, align 4, !dbg !76
  %30 = load i32, i32* %.dl0003p_379, align 4, !dbg !76
  store i32 %30, i32* %i_331, align 4, !dbg !76
  %31 = load i32, i32* %i_331, align 4, !dbg !76
  call void @llvm.dbg.value(metadata i32 %31, metadata !77, metadata !DIExpression()), !dbg !75
  store i32 %31, i32* %.dX0003p_378, align 4, !dbg !76
  %32 = load i32, i32* %.dX0003p_378, align 4, !dbg !76
  %33 = load i32, i32* %.du0003p_374, align 4, !dbg !76
  %34 = icmp sgt i32 %32, %33, !dbg !76
  br i1 %34, label %L.LB7_372, label %L.LB7_536, !dbg !76

L.LB7_536:                                        ; preds = %L.LB7_330
  %35 = load i32, i32* %.dX0003p_378, align 4, !dbg !76
  store i32 %35, i32* %i_331, align 4, !dbg !76
  %36 = load i32, i32* %.di0003p_376, align 4, !dbg !76
  %37 = load i32, i32* %.de0003p_375, align 4, !dbg !76
  %38 = load i32, i32* %.dX0003p_378, align 4, !dbg !76
  %39 = sub nsw i32 %37, %38, !dbg !76
  %40 = add nsw i32 %36, %39, !dbg !76
  %41 = load i32, i32* %.di0003p_376, align 4, !dbg !76
  %42 = sdiv i32 %40, %41, !dbg !76
  store i32 %42, i32* %.dY0003p_373, align 4, !dbg !76
  %43 = load i32, i32* %.dY0003p_373, align 4, !dbg !76
  %44 = icmp sle i32 %43, 0, !dbg !76
  br i1 %44, label %L.LB7_382, label %L.LB7_381, !dbg !76

L.LB7_381:                                        ; preds = %L.LB7_336, %L.LB7_536
  br label %L.LB7_333

L.LB7_333:                                        ; preds = %L.LB7_381
  store i32 17, i32* %.i0005p_335, align 4, !dbg !78
  store i32 16, i32* %.dY0004p_385, align 4, !dbg !78
  call void @llvm.dbg.declare(metadata i32* %j_334, metadata !79, metadata !DIExpression()), !dbg !75
  store i32 1, i32* %j_334, align 4, !dbg !78
  br label %L.LB7_383

L.LB7_383:                                        ; preds = %L.LB7_383, %L.LB7_333
  %45 = load i32, i32* %j_334, align 4, !dbg !80
  call void @llvm.dbg.value(metadata i32 %45, metadata !79, metadata !DIExpression()), !dbg !75
  %46 = sext i32 %45 to i64, !dbg !80
  %47 = bitcast %struct_drb163_0_* @_drb163_0_ to i8*, !dbg !80
  %48 = getelementptr i8, i8* %47, i64 -4, !dbg !80
  %49 = bitcast i8* %48 to i32*, !dbg !80
  %50 = getelementptr i32, i32* %49, i64 %46, !dbg !80
  %51 = load i32, i32* %50, align 4, !dbg !80
  %52 = add nsw i32 %51, 1, !dbg !80
  %53 = load i32, i32* %j_334, align 4, !dbg !80
  call void @llvm.dbg.value(metadata i32 %53, metadata !79, metadata !DIExpression()), !dbg !75
  %54 = sext i32 %53 to i64, !dbg !80
  %55 = bitcast %struct_drb163_0_* @_drb163_0_ to i8*, !dbg !80
  %56 = getelementptr i8, i8* %55, i64 -4, !dbg !80
  %57 = bitcast i8* %56 to i32*, !dbg !80
  %58 = getelementptr i32, i32* %57, i64 %54, !dbg !80
  store i32 %52, i32* %58, align 4, !dbg !80
  %59 = load i32, i32* %j_334, align 4, !dbg !81
  call void @llvm.dbg.value(metadata i32 %59, metadata !79, metadata !DIExpression()), !dbg !75
  %60 = add nsw i32 %59, 1, !dbg !81
  store i32 %60, i32* %j_334, align 4, !dbg !81
  %61 = load i32, i32* %.dY0004p_385, align 4, !dbg !81
  %62 = sub nsw i32 %61, 1, !dbg !81
  store i32 %62, i32* %.dY0004p_385, align 4, !dbg !81
  %63 = load i32, i32* %.dY0004p_385, align 4, !dbg !81
  %64 = icmp sgt i32 %63, 0, !dbg !81
  br i1 %64, label %L.LB7_383, label %L.LB7_336, !dbg !81

L.LB7_336:                                        ; preds = %L.LB7_383
  %65 = load i32, i32* %.di0003p_376, align 4, !dbg !75
  %66 = load i32, i32* %i_331, align 4, !dbg !75
  call void @llvm.dbg.value(metadata i32 %66, metadata !77, metadata !DIExpression()), !dbg !75
  %67 = add nsw i32 %65, %66, !dbg !75
  store i32 %67, i32* %i_331, align 4, !dbg !75
  %68 = load i32, i32* %.dY0003p_373, align 4, !dbg !75
  %69 = sub nsw i32 %68, 1, !dbg !75
  store i32 %69, i32* %.dY0003p_373, align 4, !dbg !75
  %70 = load i32, i32* %.dY0003p_373, align 4, !dbg !75
  %71 = icmp sgt i32 %70, 0, !dbg !75
  br i1 %71, label %L.LB7_381, label %L.LB7_382, !dbg !75

L.LB7_382:                                        ; preds = %L.LB7_336, %L.LB7_536
  br label %L.LB7_372

L.LB7_372:                                        ; preds = %L.LB7_382, %L.LB7_330
  %72 = load i32, i32* %__gtid___nv_MAIN_F1L27_3__524, align 4, !dbg !75
  call void @__kmpc_for_static_fini(i64* null, i32 %72), !dbg !75
  br label %L.LB7_337

L.LB7_337:                                        ; preds = %L.LB7_372
  ret void, !dbg !75
}

declare void @__kmpc_for_static_fini(i64*, i32) #1

declare void @__kmpc_fork_call(i64*, i32, i64*, i64*, ...) #1

declare void @__kmpc_end_serialized_parallel(i64*, i32) #1

declare void @__kmpc_serialized_parallel(i64*, i32) #1

declare void @__kmpc_for_static_init_4(i64*, i32, i32, i64*, i64*, i64*, i64*, i32, i32) #1

declare void @__kmpc_fork_teams(i64*, i32, i64*, i64*, ...) #1

declare void @_mp_ecs_nest(...) #1

declare signext i32 @f90io_ldw_end(...) #1

declare signext i32 @f90io_sc_i_ldw(...) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

declare signext i32 @f90io_print_init(...) #1

declare void @f90io_src_info03a(...) #1

declare void @_mp_bcs_nest(...) #1

declare void @fort_init(...) #1

declare signext i32 @__kmpc_global_thread_num(i64*) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { noinline }
attributes #1 = { "no-frame-pointer-elim-non-leaf" }
attributes #2 = { nounwind readnone speculatable }

!llvm.module.flags = !{!20, !21}
!llvm.dbg.cu = !{!3}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "var", scope: !2, file: !4, type: !17, isLocal: false, isDefinition: true)
!2 = !DIModule(scope: !3, name: "drb163")
!3 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !4, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !6, imports: !12)
!4 = !DIFile(filename: "micro-benchmarks-fortran/DRB164-simdmissinglock1-orig-gpu-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!5 = !{}
!6 = !{!0, !7, !10}
!7 = !DIGlobalVariableExpression(var: !8, expr: !DIExpression(DW_OP_plus_uconst, 64))
!8 = distinct !DIGlobalVariable(name: "i", scope: !2, file: !4, type: !9, isLocal: false, isDefinition: true)
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DIGlobalVariableExpression(var: !11, expr: !DIExpression(DW_OP_plus_uconst, 68))
!11 = distinct !DIGlobalVariable(name: "j", scope: !2, file: !4, type: !9, isLocal: false, isDefinition: true)
!12 = !{!13}
!13 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !14, entity: !2, file: !4, line: 16)
!14 = distinct !DISubprogram(name: "drb163_simdmissinglock1_orig_gpu_no", scope: !3, file: !4, line: 16, type: !15, scopeLine: 16, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !3)
!15 = !DISubroutineType(cc: DW_CC_program, types: !16)
!16 = !{null}
!17 = !DICompositeType(tag: DW_TAG_array_type, baseType: !9, size: 512, align: 32, elements: !18)
!18 = !{!19}
!19 = !DISubrange(count: 16, lowerBound: 1)
!20 = !{i32 2, !"Dwarf Version", i32 4}
!21 = !{i32 2, !"Debug Info Version", i32 3}
!22 = !DILocalVariable(name: "omp_sched_static", scope: !14, file: !4, type: !9)
!23 = !DILocation(line: 0, scope: !14)
!24 = !DILocalVariable(name: "omp_proc_bind_false", scope: !14, file: !4, type: !9)
!25 = !DILocalVariable(name: "omp_proc_bind_true", scope: !14, file: !4, type: !9)
!26 = !DILocalVariable(name: "omp_lock_hint_none", scope: !14, file: !4, type: !9)
!27 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !14, file: !4, type: !9)
!28 = !DILocation(line: 39, column: 1, scope: !14)
!29 = !DILocation(line: 16, column: 1, scope: !14)
!30 = !DILocation(line: 21, column: 1, scope: !14)
!31 = !DILocation(line: 22, column: 1, scope: !14)
!32 = !DILocation(line: 23, column: 1, scope: !14)
!33 = !DILocation(line: 35, column: 1, scope: !14)
!34 = !DILocation(line: 37, column: 1, scope: !14)
!35 = !DILocalVariable(scope: !14, file: !4, type: !9, flags: DIFlagArtificial)
!36 = distinct !DISubprogram(name: "__nv_MAIN__F1L25_1", scope: !3, file: !4, line: 25, type: !37, scopeLine: 25, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !3)
!37 = !DISubroutineType(types: !38)
!38 = !{null, !9, !39, !39}
!39 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!40 = !DILocalVariable(name: "__nv_MAIN__F1L25_1Arg0", arg: 1, scope: !36, file: !4, type: !9)
!41 = !DILocation(line: 0, scope: !36)
!42 = !DILocalVariable(name: "__nv_MAIN__F1L25_1Arg1", arg: 2, scope: !36, file: !4, type: !39)
!43 = !DILocalVariable(name: "__nv_MAIN__F1L25_1Arg2", arg: 3, scope: !36, file: !4, type: !39)
!44 = !DILocalVariable(name: "omp_sched_static", scope: !36, file: !4, type: !9)
!45 = !DILocalVariable(name: "omp_proc_bind_false", scope: !36, file: !4, type: !9)
!46 = !DILocalVariable(name: "omp_proc_bind_true", scope: !36, file: !4, type: !9)
!47 = !DILocalVariable(name: "omp_lock_hint_none", scope: !36, file: !4, type: !9)
!48 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !36, file: !4, type: !9)
!49 = !DILocation(line: 26, column: 1, scope: !36)
!50 = !DILocation(line: 35, column: 1, scope: !36)
!51 = distinct !DISubprogram(name: "__nv_MAIN_F1L26_2", scope: !3, file: !4, line: 26, type: !37, scopeLine: 26, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !3)
!52 = !DILocalVariable(name: "__nv_MAIN_F1L26_2Arg0", arg: 1, scope: !51, file: !4, type: !9)
!53 = !DILocation(line: 0, scope: !51)
!54 = !DILocalVariable(name: "__nv_MAIN_F1L26_2Arg1", arg: 2, scope: !51, file: !4, type: !39)
!55 = !DILocalVariable(name: "__nv_MAIN_F1L26_2Arg2", arg: 3, scope: !51, file: !4, type: !39)
!56 = !DILocalVariable(name: "omp_sched_static", scope: !51, file: !4, type: !9)
!57 = !DILocalVariable(name: "omp_proc_bind_false", scope: !51, file: !4, type: !9)
!58 = !DILocalVariable(name: "omp_proc_bind_true", scope: !51, file: !4, type: !9)
!59 = !DILocalVariable(name: "omp_lock_hint_none", scope: !51, file: !4, type: !9)
!60 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !51, file: !4, type: !9)
!61 = !DILocation(line: 34, column: 1, scope: !51)
!62 = !DILocation(line: 27, column: 1, scope: !51)
!63 = !DILocalVariable(name: "i", scope: !51, file: !4, type: !9)
!64 = !DILocation(line: 33, column: 1, scope: !51)
!65 = distinct !DISubprogram(name: "__nv_MAIN_F1L27_3", scope: !3, file: !4, line: 27, type: !37, scopeLine: 27, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !3)
!66 = !DILocalVariable(name: "__nv_MAIN_F1L27_3Arg0", arg: 1, scope: !65, file: !4, type: !9)
!67 = !DILocation(line: 0, scope: !65)
!68 = !DILocalVariable(name: "__nv_MAIN_F1L27_3Arg1", arg: 2, scope: !65, file: !4, type: !39)
!69 = !DILocalVariable(name: "__nv_MAIN_F1L27_3Arg2", arg: 3, scope: !65, file: !4, type: !39)
!70 = !DILocalVariable(name: "omp_sched_static", scope: !65, file: !4, type: !9)
!71 = !DILocalVariable(name: "omp_proc_bind_false", scope: !65, file: !4, type: !9)
!72 = !DILocalVariable(name: "omp_proc_bind_true", scope: !65, file: !4, type: !9)
!73 = !DILocalVariable(name: "omp_lock_hint_none", scope: !65, file: !4, type: !9)
!74 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !65, file: !4, type: !9)
!75 = !DILocation(line: 33, column: 1, scope: !65)
!76 = !DILocation(line: 27, column: 1, scope: !65)
!77 = !DILocalVariable(name: "i", scope: !65, file: !4, type: !9)
!78 = !DILocation(line: 29, column: 1, scope: !65)
!79 = !DILocalVariable(name: "j", scope: !65, file: !4, type: !9)
!80 = !DILocation(line: 30, column: 1, scope: !65)
!81 = !DILocation(line: 31, column: 1, scope: !65)

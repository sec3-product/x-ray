; ModuleID = '/tmp/DRB153-missinglock2-orig-gpu-yes-0bc89c.ll'
source_filename = "/tmp/DRB153-missinglock2-orig-gpu-yes-0bc89c.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%astruct.dt58 = type <{ i8* }>
%astruct.dt100 = type <{ [8 x i8] }>
%astruct.dt154 = type <{ [8 x i8], i8*, i8* }>

@.C305_MAIN_ = internal constant i32 25
@.C284_MAIN_ = internal constant i64 0
@.C336_MAIN_ = internal constant i32 6
@.C334_MAIN_ = internal constant [61 x i8] c"micro-benchmarks-fortran/DRB153-missinglock2-orig-gpu-yes.f95"
@.C306_MAIN_ = internal constant i32 28
@.C317_MAIN_ = internal constant i32 100
@.C285_MAIN_ = internal constant i32 1
@.C283_MAIN_ = internal constant i32 0
@.C317___nv_MAIN__F1L18_1 = internal constant i32 100
@.C283___nv_MAIN__F1L18_1 = internal constant i32 0
@.C285___nv_MAIN__F1L18_1 = internal constant i32 1
@.C317___nv_MAIN_F1L19_2 = internal constant i32 100
@.C285___nv_MAIN_F1L19_2 = internal constant i32 1
@.C283___nv_MAIN_F1L19_2 = internal constant i32 0
@.C285___nv_MAIN_F1L21_3 = internal constant i32 1
@.C283___nv_MAIN_F1L21_3 = internal constant i32 0

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__380 = alloca i32, align 4
  %var_307 = alloca i32, align 4
  %.uplevelArgPack0001_377 = alloca %astruct.dt58, align 8
  %z__io_338 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !14, metadata !DIExpression()), !dbg !10
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !15
  store i32 %0, i32* %__gtid_MAIN__380, align 4, !dbg !15
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !16
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !16
  call void (i8*, ...) %2(i8* %1), !dbg !16
  br label %L.LB1_371

L.LB1_371:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %var_307, metadata !17, metadata !DIExpression()), !dbg !10
  store i32 0, i32* %var_307, align 4, !dbg !18
  %3 = bitcast i32* %var_307 to i8*, !dbg !19
  %4 = bitcast %astruct.dt58* %.uplevelArgPack0001_377 to i8**, !dbg !19
  store i8* %3, i8** %4, align 8, !dbg !19
  %5 = bitcast %astruct.dt58* %.uplevelArgPack0001_377 to i64*, !dbg !19
  call void @__nv_MAIN__F1L18_1_(i32* %__gtid_MAIN__380, i64* null, i64* %5), !dbg !19
  call void (...) @_mp_bcs_nest(), !dbg !20
  %6 = bitcast i32* @.C306_MAIN_ to i8*, !dbg !20
  %7 = bitcast [61 x i8]* @.C334_MAIN_ to i8*, !dbg !20
  %8 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !20
  call void (i8*, i8*, i64, ...) %8(i8* %6, i8* %7, i64 61), !dbg !20
  %9 = bitcast i32* @.C336_MAIN_ to i8*, !dbg !20
  %10 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !20
  %11 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !20
  %12 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !20
  %13 = call i32 (i8*, i8*, i8*, i8*, ...) %12(i8* %9, i8* null, i8* %10, i8* %11), !dbg !20
  call void @llvm.dbg.declare(metadata i32* %z__io_338, metadata !21, metadata !DIExpression()), !dbg !10
  store i32 %13, i32* %z__io_338, align 4, !dbg !20
  %14 = load i32, i32* %var_307, align 4, !dbg !20
  call void @llvm.dbg.value(metadata i32 %14, metadata !17, metadata !DIExpression()), !dbg !10
  %15 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !20
  %16 = call i32 (i32, i32, ...) %15(i32 %14, i32 25), !dbg !20
  store i32 %16, i32* %z__io_338, align 4, !dbg !20
  %17 = call i32 (...) @f90io_ldw_end(), !dbg !20
  store i32 %17, i32* %z__io_338, align 4, !dbg !20
  call void (...) @_mp_ecs_nest(), !dbg !20
  ret void, !dbg !15
}

define internal void @__nv_MAIN__F1L18_1_(i32* %__nv_MAIN__F1L18_1Arg0, i64* %__nv_MAIN__F1L18_1Arg1, i64* %__nv_MAIN__F1L18_1Arg2) #0 !dbg !22 {
L.entry:
  %__gtid___nv_MAIN__F1L18_1__404 = alloca i32, align 4
  %.uplevelArgPack0002_400 = alloca %astruct.dt100, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L18_1Arg0, metadata !26, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L18_1Arg1, metadata !28, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L18_1Arg2, metadata !29, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.value(metadata i32 1, metadata !30, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.value(metadata i32 0, metadata !31, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.value(metadata i32 1, metadata !32, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.value(metadata i32 0, metadata !33, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.value(metadata i32 1, metadata !34, metadata !DIExpression()), !dbg !27
  %0 = load i32, i32* %__nv_MAIN__F1L18_1Arg0, align 4, !dbg !35
  store i32 %0, i32* %__gtid___nv_MAIN__F1L18_1__404, align 4, !dbg !35
  br label %L.LB2_395

L.LB2_395:                                        ; preds = %L.entry
  br label %L.LB2_311

L.LB2_311:                                        ; preds = %L.LB2_395
  %1 = load i64, i64* %__nv_MAIN__F1L18_1Arg2, align 8, !dbg !36
  %2 = bitcast %astruct.dt100* %.uplevelArgPack0002_400 to i64*, !dbg !36
  store i64 %1, i64* %2, align 8, !dbg !36
  %3 = load i32, i32* %__gtid___nv_MAIN__F1L18_1__404, align 4, !dbg !36
  call void @__kmpc_push_num_teams(i64* null, i32 %3, i32 1, i32 0), !dbg !36
  %4 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN_F1L19_2_ to i64*, !dbg !36
  %5 = bitcast %astruct.dt100* %.uplevelArgPack0002_400 to i64*, !dbg !36
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_teams(i64* null, i32 1, i64* %4, i64* %5), !dbg !36
  br label %L.LB2_332

L.LB2_332:                                        ; preds = %L.LB2_311
  ret void, !dbg !35
}

define internal void @__nv_MAIN_F1L19_2_(i32* %__nv_MAIN_F1L19_2Arg0, i64* %__nv_MAIN_F1L19_2Arg1, i64* %__nv_MAIN_F1L19_2Arg2) #0 !dbg !37 {
L.entry:
  %__gtid___nv_MAIN_F1L19_2__444 = alloca i32, align 4
  %.i0000p_319 = alloca i32, align 4
  %.i0001p_320 = alloca i32, align 4
  %.i0002p_321 = alloca i32, align 4
  %.i0003p_322 = alloca i32, align 4
  %i_318 = alloca i32, align 4
  %.du0001_349 = alloca i32, align 4
  %.de0001_350 = alloca i32, align 4
  %.di0001_351 = alloca i32, align 4
  %.ds0001_352 = alloca i32, align 4
  %.dl0001_354 = alloca i32, align 4
  %.dl0001.copy_438 = alloca i32, align 4
  %.de0001.copy_439 = alloca i32, align 4
  %.ds0001.copy_440 = alloca i32, align 4
  %.dX0001_353 = alloca i32, align 4
  %.dY0001_348 = alloca i32, align 4
  %.uplevelArgPack0003_463 = alloca %astruct.dt154, align 16
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L19_2Arg0, metadata !38, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L19_2Arg1, metadata !40, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L19_2Arg2, metadata !41, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.value(metadata i32 1, metadata !42, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.value(metadata i32 0, metadata !43, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.value(metadata i32 1, metadata !44, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.value(metadata i32 0, metadata !45, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.value(metadata i32 1, metadata !46, metadata !DIExpression()), !dbg !39
  %0 = load i32, i32* %__nv_MAIN_F1L19_2Arg0, align 4, !dbg !47
  store i32 %0, i32* %__gtid___nv_MAIN_F1L19_2__444, align 4, !dbg !47
  br label %L.LB4_427

L.LB4_427:                                        ; preds = %L.entry
  br label %L.LB4_314

L.LB4_314:                                        ; preds = %L.LB4_427
  br label %L.LB4_315

L.LB4_315:                                        ; preds = %L.LB4_314
  br label %L.LB4_316

L.LB4_316:                                        ; preds = %L.LB4_315
  store i32 0, i32* %.i0000p_319, align 4, !dbg !48
  store i32 1, i32* %.i0001p_320, align 4, !dbg !48
  store i32 100, i32* %.i0002p_321, align 4, !dbg !48
  store i32 1, i32* %.i0003p_322, align 4, !dbg !48
  %1 = load i32, i32* %.i0001p_320, align 4, !dbg !48
  call void @llvm.dbg.declare(metadata i32* %i_318, metadata !49, metadata !DIExpression()), !dbg !47
  store i32 %1, i32* %i_318, align 4, !dbg !48
  %2 = load i32, i32* %.i0002p_321, align 4, !dbg !48
  store i32 %2, i32* %.du0001_349, align 4, !dbg !48
  %3 = load i32, i32* %.i0002p_321, align 4, !dbg !48
  store i32 %3, i32* %.de0001_350, align 4, !dbg !48
  store i32 1, i32* %.di0001_351, align 4, !dbg !48
  %4 = load i32, i32* %.di0001_351, align 4, !dbg !48
  store i32 %4, i32* %.ds0001_352, align 4, !dbg !48
  %5 = load i32, i32* %.i0001p_320, align 4, !dbg !48
  store i32 %5, i32* %.dl0001_354, align 4, !dbg !48
  %6 = load i32, i32* %.dl0001_354, align 4, !dbg !48
  store i32 %6, i32* %.dl0001.copy_438, align 4, !dbg !48
  %7 = load i32, i32* %.de0001_350, align 4, !dbg !48
  store i32 %7, i32* %.de0001.copy_439, align 4, !dbg !48
  %8 = load i32, i32* %.ds0001_352, align 4, !dbg !48
  store i32 %8, i32* %.ds0001.copy_440, align 4, !dbg !48
  %9 = load i32, i32* %__gtid___nv_MAIN_F1L19_2__444, align 4, !dbg !48
  %10 = bitcast i32* %.i0000p_319 to i64*, !dbg !48
  %11 = bitcast i32* %.dl0001.copy_438 to i64*, !dbg !48
  %12 = bitcast i32* %.de0001.copy_439 to i64*, !dbg !48
  %13 = bitcast i32* %.ds0001.copy_440 to i64*, !dbg !48
  %14 = load i32, i32* %.ds0001.copy_440, align 4, !dbg !48
  call void @__kmpc_for_static_init_4(i64* null, i32 %9, i32 92, i64* %10, i64* %11, i64* %12, i64* %13, i32 %14, i32 1), !dbg !48
  %15 = load i32, i32* %.dl0001.copy_438, align 4, !dbg !48
  store i32 %15, i32* %.dl0001_354, align 4, !dbg !48
  %16 = load i32, i32* %.de0001.copy_439, align 4, !dbg !48
  store i32 %16, i32* %.de0001_350, align 4, !dbg !48
  %17 = load i32, i32* %.ds0001.copy_440, align 4, !dbg !48
  store i32 %17, i32* %.ds0001_352, align 4, !dbg !48
  %18 = load i32, i32* %.dl0001_354, align 4, !dbg !48
  store i32 %18, i32* %i_318, align 4, !dbg !48
  %19 = load i32, i32* %i_318, align 4, !dbg !48
  call void @llvm.dbg.value(metadata i32 %19, metadata !49, metadata !DIExpression()), !dbg !47
  store i32 %19, i32* %.dX0001_353, align 4, !dbg !48
  %20 = load i32, i32* %.dX0001_353, align 4, !dbg !48
  %21 = load i32, i32* %.du0001_349, align 4, !dbg !48
  %22 = icmp sgt i32 %20, %21, !dbg !48
  br i1 %22, label %L.LB4_347, label %L.LB4_494, !dbg !48

L.LB4_494:                                        ; preds = %L.LB4_316
  %23 = load i32, i32* %.du0001_349, align 4, !dbg !48
  %24 = load i32, i32* %.de0001_350, align 4, !dbg !48
  %25 = icmp slt i32 %23, %24, !dbg !48
  %26 = select i1 %25, i32 %23, i32 %24, !dbg !48
  store i32 %26, i32* %.de0001_350, align 4, !dbg !48
  %27 = load i32, i32* %.dX0001_353, align 4, !dbg !48
  store i32 %27, i32* %i_318, align 4, !dbg !48
  %28 = load i32, i32* %.di0001_351, align 4, !dbg !48
  %29 = load i32, i32* %.de0001_350, align 4, !dbg !48
  %30 = load i32, i32* %.dX0001_353, align 4, !dbg !48
  %31 = sub nsw i32 %29, %30, !dbg !48
  %32 = add nsw i32 %28, %31, !dbg !48
  %33 = load i32, i32* %.di0001_351, align 4, !dbg !48
  %34 = sdiv i32 %32, %33, !dbg !48
  store i32 %34, i32* %.dY0001_348, align 4, !dbg !48
  %35 = load i32, i32* %i_318, align 4, !dbg !48
  call void @llvm.dbg.value(metadata i32 %35, metadata !49, metadata !DIExpression()), !dbg !47
  store i32 %35, i32* %.i0001p_320, align 4, !dbg !48
  %36 = load i32, i32* %.de0001_350, align 4, !dbg !48
  store i32 %36, i32* %.i0002p_321, align 4, !dbg !48
  %37 = load i64, i64* %__nv_MAIN_F1L19_2Arg2, align 8, !dbg !48
  %38 = bitcast %astruct.dt154* %.uplevelArgPack0003_463 to i64*, !dbg !48
  store i64 %37, i64* %38, align 8, !dbg !48
  %39 = bitcast i32* %.i0001p_320 to i8*, !dbg !48
  %40 = bitcast %astruct.dt154* %.uplevelArgPack0003_463 to i8*, !dbg !48
  %41 = getelementptr i8, i8* %40, i64 8, !dbg !48
  %42 = bitcast i8* %41 to i8**, !dbg !48
  store i8* %39, i8** %42, align 8, !dbg !48
  %43 = bitcast i32* %.i0002p_321 to i8*, !dbg !48
  %44 = bitcast %astruct.dt154* %.uplevelArgPack0003_463 to i8*, !dbg !48
  %45 = getelementptr i8, i8* %44, i64 16, !dbg !48
  %46 = bitcast i8* %45 to i8**, !dbg !48
  store i8* %43, i8** %46, align 8, !dbg !48
  br label %L.LB4_470, !dbg !48

L.LB4_470:                                        ; preds = %L.LB4_494
  %47 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN_F1L21_3_ to i64*, !dbg !48
  %48 = bitcast %astruct.dt154* %.uplevelArgPack0003_463 to i64*, !dbg !48
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %47, i64* %48), !dbg !48
  br label %L.LB4_347

L.LB4_347:                                        ; preds = %L.LB4_470, %L.LB4_316
  %49 = load i32, i32* %__gtid___nv_MAIN_F1L19_2__444, align 4, !dbg !50
  call void @__kmpc_for_static_fini(i64* null, i32 %49), !dbg !50
  br label %L.LB4_329

L.LB4_329:                                        ; preds = %L.LB4_347
  br label %L.LB4_330

L.LB4_330:                                        ; preds = %L.LB4_329
  br label %L.LB4_331

L.LB4_331:                                        ; preds = %L.LB4_330
  ret void, !dbg !47
}

define internal void @__nv_MAIN_F1L21_3_(i32* %__nv_MAIN_F1L21_3Arg0, i64* %__nv_MAIN_F1L21_3Arg1, i64* %__nv_MAIN_F1L21_3Arg2) #0 !dbg !51 {
L.entry:
  %__gtid___nv_MAIN_F1L21_3__515 = alloca i32, align 4
  %.i0004p_327 = alloca i32, align 4
  %i_326 = alloca i32, align 4
  %.du0002p_361 = alloca i32, align 4
  %.de0002p_362 = alloca i32, align 4
  %.di0002p_363 = alloca i32, align 4
  %.ds0002p_364 = alloca i32, align 4
  %.dl0002p_366 = alloca i32, align 4
  %.dl0002p.copy_509 = alloca i32, align 4
  %.de0002p.copy_510 = alloca i32, align 4
  %.ds0002p.copy_511 = alloca i32, align 4
  %.dX0002p_365 = alloca i32, align 4
  %.dY0002p_360 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L21_3Arg0, metadata !52, metadata !DIExpression()), !dbg !53
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L21_3Arg1, metadata !54, metadata !DIExpression()), !dbg !53
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L21_3Arg2, metadata !55, metadata !DIExpression()), !dbg !53
  call void @llvm.dbg.value(metadata i32 1, metadata !56, metadata !DIExpression()), !dbg !53
  call void @llvm.dbg.value(metadata i32 0, metadata !57, metadata !DIExpression()), !dbg !53
  call void @llvm.dbg.value(metadata i32 1, metadata !58, metadata !DIExpression()), !dbg !53
  call void @llvm.dbg.value(metadata i32 0, metadata !59, metadata !DIExpression()), !dbg !53
  call void @llvm.dbg.value(metadata i32 1, metadata !60, metadata !DIExpression()), !dbg !53
  %0 = load i32, i32* %__nv_MAIN_F1L21_3Arg0, align 4, !dbg !61
  store i32 %0, i32* %__gtid___nv_MAIN_F1L21_3__515, align 4, !dbg !61
  br label %L.LB6_498

L.LB6_498:                                        ; preds = %L.entry
  br label %L.LB6_325

L.LB6_325:                                        ; preds = %L.LB6_498
  store i32 0, i32* %.i0004p_327, align 4, !dbg !62
  %1 = bitcast i64* %__nv_MAIN_F1L21_3Arg2 to i8*, !dbg !62
  %2 = getelementptr i8, i8* %1, i64 8, !dbg !62
  %3 = bitcast i8* %2 to i32**, !dbg !62
  %4 = load i32*, i32** %3, align 8, !dbg !62
  %5 = load i32, i32* %4, align 4, !dbg !62
  call void @llvm.dbg.declare(metadata i32* %i_326, metadata !63, metadata !DIExpression()), !dbg !61
  store i32 %5, i32* %i_326, align 4, !dbg !62
  %6 = bitcast i64* %__nv_MAIN_F1L21_3Arg2 to i8*, !dbg !62
  %7 = getelementptr i8, i8* %6, i64 16, !dbg !62
  %8 = bitcast i8* %7 to i32**, !dbg !62
  %9 = load i32*, i32** %8, align 8, !dbg !62
  %10 = load i32, i32* %9, align 4, !dbg !62
  store i32 %10, i32* %.du0002p_361, align 4, !dbg !62
  %11 = bitcast i64* %__nv_MAIN_F1L21_3Arg2 to i8*, !dbg !62
  %12 = getelementptr i8, i8* %11, i64 16, !dbg !62
  %13 = bitcast i8* %12 to i32**, !dbg !62
  %14 = load i32*, i32** %13, align 8, !dbg !62
  %15 = load i32, i32* %14, align 4, !dbg !62
  store i32 %15, i32* %.de0002p_362, align 4, !dbg !62
  store i32 1, i32* %.di0002p_363, align 4, !dbg !62
  %16 = load i32, i32* %.di0002p_363, align 4, !dbg !62
  store i32 %16, i32* %.ds0002p_364, align 4, !dbg !62
  %17 = bitcast i64* %__nv_MAIN_F1L21_3Arg2 to i8*, !dbg !62
  %18 = getelementptr i8, i8* %17, i64 8, !dbg !62
  %19 = bitcast i8* %18 to i32**, !dbg !62
  %20 = load i32*, i32** %19, align 8, !dbg !62
  %21 = load i32, i32* %20, align 4, !dbg !62
  store i32 %21, i32* %.dl0002p_366, align 4, !dbg !62
  %22 = load i32, i32* %.dl0002p_366, align 4, !dbg !62
  store i32 %22, i32* %.dl0002p.copy_509, align 4, !dbg !62
  %23 = load i32, i32* %.de0002p_362, align 4, !dbg !62
  store i32 %23, i32* %.de0002p.copy_510, align 4, !dbg !62
  %24 = load i32, i32* %.ds0002p_364, align 4, !dbg !62
  store i32 %24, i32* %.ds0002p.copy_511, align 4, !dbg !62
  %25 = load i32, i32* %__gtid___nv_MAIN_F1L21_3__515, align 4, !dbg !62
  %26 = bitcast i32* %.i0004p_327 to i64*, !dbg !62
  %27 = bitcast i32* %.dl0002p.copy_509 to i64*, !dbg !62
  %28 = bitcast i32* %.de0002p.copy_510 to i64*, !dbg !62
  %29 = bitcast i32* %.ds0002p.copy_511 to i64*, !dbg !62
  %30 = load i32, i32* %.ds0002p.copy_511, align 4, !dbg !62
  call void @__kmpc_for_static_init_4(i64* null, i32 %25, i32 34, i64* %26, i64* %27, i64* %28, i64* %29, i32 %30, i32 1), !dbg !62
  %31 = load i32, i32* %.dl0002p.copy_509, align 4, !dbg !62
  store i32 %31, i32* %.dl0002p_366, align 4, !dbg !62
  %32 = load i32, i32* %.de0002p.copy_510, align 4, !dbg !62
  store i32 %32, i32* %.de0002p_362, align 4, !dbg !62
  %33 = load i32, i32* %.ds0002p.copy_511, align 4, !dbg !62
  store i32 %33, i32* %.ds0002p_364, align 4, !dbg !62
  %34 = load i32, i32* %.dl0002p_366, align 4, !dbg !62
  store i32 %34, i32* %i_326, align 4, !dbg !62
  %35 = load i32, i32* %i_326, align 4, !dbg !62
  call void @llvm.dbg.value(metadata i32 %35, metadata !63, metadata !DIExpression()), !dbg !61
  store i32 %35, i32* %.dX0002p_365, align 4, !dbg !62
  %36 = load i32, i32* %.dX0002p_365, align 4, !dbg !62
  %37 = load i32, i32* %.du0002p_361, align 4, !dbg !62
  %38 = icmp sgt i32 %36, %37, !dbg !62
  br i1 %38, label %L.LB6_359, label %L.LB6_524, !dbg !62

L.LB6_524:                                        ; preds = %L.LB6_325
  %39 = load i32, i32* %.dX0002p_365, align 4, !dbg !62
  store i32 %39, i32* %i_326, align 4, !dbg !62
  %40 = load i32, i32* %.di0002p_363, align 4, !dbg !62
  %41 = load i32, i32* %.de0002p_362, align 4, !dbg !62
  %42 = load i32, i32* %.dX0002p_365, align 4, !dbg !62
  %43 = sub nsw i32 %41, %42, !dbg !62
  %44 = add nsw i32 %40, %43, !dbg !62
  %45 = load i32, i32* %.di0002p_363, align 4, !dbg !62
  %46 = sdiv i32 %44, %45, !dbg !62
  store i32 %46, i32* %.dY0002p_360, align 4, !dbg !62
  %47 = load i32, i32* %.dY0002p_360, align 4, !dbg !62
  %48 = icmp sle i32 %47, 0, !dbg !62
  br i1 %48, label %L.LB6_369, label %L.LB6_368, !dbg !62

L.LB6_368:                                        ; preds = %L.LB6_368, %L.LB6_524
  %49 = bitcast i64* %__nv_MAIN_F1L21_3Arg2 to i32**, !dbg !64
  %50 = load i32*, i32** %49, align 8, !dbg !64
  %51 = load i32, i32* %50, align 4, !dbg !64
  %52 = add nsw i32 %51, 1, !dbg !64
  %53 = bitcast i64* %__nv_MAIN_F1L21_3Arg2 to i32**, !dbg !64
  %54 = load i32*, i32** %53, align 8, !dbg !64
  store i32 %52, i32* %54, align 4, !dbg !64
  %55 = load i32, i32* %.di0002p_363, align 4, !dbg !61
  %56 = load i32, i32* %i_326, align 4, !dbg !61
  call void @llvm.dbg.value(metadata i32 %56, metadata !63, metadata !DIExpression()), !dbg !61
  %57 = add nsw i32 %55, %56, !dbg !61
  store i32 %57, i32* %i_326, align 4, !dbg !61
  %58 = load i32, i32* %.dY0002p_360, align 4, !dbg !61
  %59 = sub nsw i32 %58, 1, !dbg !61
  store i32 %59, i32* %.dY0002p_360, align 4, !dbg !61
  %60 = load i32, i32* %.dY0002p_360, align 4, !dbg !61
  %61 = icmp sgt i32 %60, 0, !dbg !61
  br i1 %61, label %L.LB6_368, label %L.LB6_369, !dbg !61

L.LB6_369:                                        ; preds = %L.LB6_368, %L.LB6_524
  br label %L.LB6_359

L.LB6_359:                                        ; preds = %L.LB6_369, %L.LB6_325
  %62 = load i32, i32* %__gtid___nv_MAIN_F1L21_3__515, align 4, !dbg !61
  call void @__kmpc_for_static_fini(i64* null, i32 %62), !dbg !61
  br label %L.LB6_328

L.LB6_328:                                        ; preds = %L.LB6_359
  ret void, !dbg !61
}

declare void @__kmpc_for_static_fini(i64*, i32) #0

declare void @__kmpc_fork_call(i64*, i32, i64*, i64*, ...) #0

declare void @__kmpc_end_serialized_parallel(i64*, i32) #0

declare void @__kmpc_serialized_parallel(i64*, i32) #0

declare void @__kmpc_for_static_init_4(i64*, i32, i32, i64*, i64*, i64*, i64*, i32, i32) #0

declare void @__kmpc_fork_teams(i64*, i32, i64*, i64*, ...) #0

declare void @__kmpc_push_num_teams(i64*, i32, i32, i32) #0

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

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { "no-frame-pointer-elim-non-leaf" }
attributes #1 = { nounwind readnone speculatable }

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !4, globals: !4, imports: !4)
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB153-missinglock2-orig-gpu-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb153_missinglock2_orig_gpu_yes", scope: !2, file: !3, line: 11, type: !6, scopeLine: 11, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!6 = !DISubroutineType(cc: DW_CC_program, types: !7)
!7 = !{null}
!8 = !DILocalVariable(name: "omp_sched_static", scope: !5, file: !3, type: !9)
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 0, scope: !5)
!11 = !DILocalVariable(name: "omp_proc_bind_false", scope: !5, file: !3, type: !9)
!12 = !DILocalVariable(name: "omp_proc_bind_true", scope: !5, file: !3, type: !9)
!13 = !DILocalVariable(name: "omp_lock_hint_none", scope: !5, file: !3, type: !9)
!14 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !5, file: !3, type: !9)
!15 = !DILocation(line: 29, column: 1, scope: !5)
!16 = !DILocation(line: 11, column: 1, scope: !5)
!17 = !DILocalVariable(name: "var", scope: !5, file: !3, type: !9)
!18 = !DILocation(line: 16, column: 1, scope: !5)
!19 = !DILocation(line: 26, column: 1, scope: !5)
!20 = !DILocation(line: 28, column: 1, scope: !5)
!21 = !DILocalVariable(scope: !5, file: !3, type: !9, flags: DIFlagArtificial)
!22 = distinct !DISubprogram(name: "__nv_MAIN__F1L18_1", scope: !2, file: !3, line: 18, type: !23, scopeLine: 18, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!23 = !DISubroutineType(types: !24)
!24 = !{null, !9, !25, !25}
!25 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!26 = !DILocalVariable(name: "__nv_MAIN__F1L18_1Arg0", arg: 1, scope: !22, file: !3, type: !9)
!27 = !DILocation(line: 0, scope: !22)
!28 = !DILocalVariable(name: "__nv_MAIN__F1L18_1Arg1", arg: 2, scope: !22, file: !3, type: !25)
!29 = !DILocalVariable(name: "__nv_MAIN__F1L18_1Arg2", arg: 3, scope: !22, file: !3, type: !25)
!30 = !DILocalVariable(name: "omp_sched_static", scope: !22, file: !3, type: !9)
!31 = !DILocalVariable(name: "omp_proc_bind_false", scope: !22, file: !3, type: !9)
!32 = !DILocalVariable(name: "omp_proc_bind_true", scope: !22, file: !3, type: !9)
!33 = !DILocalVariable(name: "omp_lock_hint_none", scope: !22, file: !3, type: !9)
!34 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !22, file: !3, type: !9)
!35 = !DILocation(line: 26, column: 1, scope: !22)
!36 = !DILocation(line: 19, column: 1, scope: !22)
!37 = distinct !DISubprogram(name: "__nv_MAIN_F1L19_2", scope: !2, file: !3, line: 19, type: !23, scopeLine: 19, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!38 = !DILocalVariable(name: "__nv_MAIN_F1L19_2Arg0", arg: 1, scope: !37, file: !3, type: !9)
!39 = !DILocation(line: 0, scope: !37)
!40 = !DILocalVariable(name: "__nv_MAIN_F1L19_2Arg1", arg: 2, scope: !37, file: !3, type: !25)
!41 = !DILocalVariable(name: "__nv_MAIN_F1L19_2Arg2", arg: 3, scope: !37, file: !3, type: !25)
!42 = !DILocalVariable(name: "omp_sched_static", scope: !37, file: !3, type: !9)
!43 = !DILocalVariable(name: "omp_proc_bind_false", scope: !37, file: !3, type: !9)
!44 = !DILocalVariable(name: "omp_proc_bind_true", scope: !37, file: !3, type: !9)
!45 = !DILocalVariable(name: "omp_lock_hint_none", scope: !37, file: !3, type: !9)
!46 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !37, file: !3, type: !9)
!47 = !DILocation(line: 25, column: 1, scope: !37)
!48 = !DILocation(line: 21, column: 1, scope: !37)
!49 = !DILocalVariable(name: "i", scope: !37, file: !3, type: !9)
!50 = !DILocation(line: 23, column: 1, scope: !37)
!51 = distinct !DISubprogram(name: "__nv_MAIN_F1L21_3", scope: !2, file: !3, line: 21, type: !23, scopeLine: 21, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!52 = !DILocalVariable(name: "__nv_MAIN_F1L21_3Arg0", arg: 1, scope: !51, file: !3, type: !9)
!53 = !DILocation(line: 0, scope: !51)
!54 = !DILocalVariable(name: "__nv_MAIN_F1L21_3Arg1", arg: 2, scope: !51, file: !3, type: !25)
!55 = !DILocalVariable(name: "__nv_MAIN_F1L21_3Arg2", arg: 3, scope: !51, file: !3, type: !25)
!56 = !DILocalVariable(name: "omp_sched_static", scope: !51, file: !3, type: !9)
!57 = !DILocalVariable(name: "omp_proc_bind_false", scope: !51, file: !3, type: !9)
!58 = !DILocalVariable(name: "omp_proc_bind_true", scope: !51, file: !3, type: !9)
!59 = !DILocalVariable(name: "omp_lock_hint_none", scope: !51, file: !3, type: !9)
!60 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !51, file: !3, type: !9)
!61 = !DILocation(line: 23, column: 1, scope: !51)
!62 = !DILocation(line: 21, column: 1, scope: !51)
!63 = !DILocalVariable(name: "i", scope: !51, file: !3, type: !9)
!64 = !DILocation(line: 22, column: 1, scope: !51)

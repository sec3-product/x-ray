; ModuleID = '/tmp/DRB151-missinglock3-orig-gpu-yes-3306d8.ll'
source_filename = "/tmp/DRB151-missinglock3-orig-gpu-yes-3306d8.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%astruct.dt58 = type <{ i8* }>
%astruct.dt100 = type <{ [8 x i8] }>
%astruct.dt154 = type <{ [8 x i8], i8*, i8* }>

@.C305_MAIN_ = internal constant i32 25
@.C284_MAIN_ = internal constant i64 0
@.C336_MAIN_ = internal constant i32 6
@.C333_MAIN_ = internal constant [61 x i8] c"micro-benchmarks-fortran/DRB151-missinglock3-orig-gpu-yes.f95"
@.C335_MAIN_ = internal constant i32 26
@.C316_MAIN_ = internal constant i32 100
@.C285_MAIN_ = internal constant i32 1
@.C283_MAIN_ = internal constant i32 0
@.C316___nv_MAIN__F1L18_1 = internal constant i32 100
@.C285___nv_MAIN__F1L18_1 = internal constant i32 1
@.C283___nv_MAIN__F1L18_1 = internal constant i32 0
@.C316___nv_MAIN_F1L19_2 = internal constant i32 100
@.C285___nv_MAIN_F1L19_2 = internal constant i32 1
@.C283___nv_MAIN_F1L19_2 = internal constant i32 0
@.C285___nv_MAIN_F1L20_3 = internal constant i32 1
@.C283___nv_MAIN_F1L20_3 = internal constant i32 0

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__380 = alloca i32, align 4
  %var_306 = alloca i32, align 4
  %.uplevelArgPack0001_376 = alloca %astruct.dt58, align 8
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
  call void @llvm.dbg.declare(metadata i32* %var_306, metadata !17, metadata !DIExpression()), !dbg !10
  %3 = bitcast i32* %var_306 to i8*, !dbg !18
  %4 = bitcast %astruct.dt58* %.uplevelArgPack0001_376 to i8**, !dbg !18
  store i8* %3, i8** %4, align 8, !dbg !18
  %5 = bitcast %astruct.dt58* %.uplevelArgPack0001_376 to i64*, !dbg !18
  call void @__nv_MAIN__F1L18_1_(i32* %__gtid_MAIN__380, i64* null, i64* %5), !dbg !18
  call void (...) @_mp_bcs_nest(), !dbg !19
  %6 = bitcast i32* @.C335_MAIN_ to i8*, !dbg !19
  %7 = bitcast [61 x i8]* @.C333_MAIN_ to i8*, !dbg !19
  %8 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !19
  call void (i8*, i8*, i64, ...) %8(i8* %6, i8* %7, i64 61), !dbg !19
  %9 = bitcast i32* @.C336_MAIN_ to i8*, !dbg !19
  %10 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !19
  %11 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !19
  %12 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !19
  %13 = call i32 (i8*, i8*, i8*, i8*, ...) %12(i8* %9, i8* null, i8* %10, i8* %11), !dbg !19
  call void @llvm.dbg.declare(metadata i32* %z__io_338, metadata !20, metadata !DIExpression()), !dbg !10
  store i32 %13, i32* %z__io_338, align 4, !dbg !19
  %14 = load i32, i32* %var_306, align 4, !dbg !19
  call void @llvm.dbg.value(metadata i32 %14, metadata !17, metadata !DIExpression()), !dbg !10
  %15 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !19
  %16 = call i32 (i32, i32, ...) %15(i32 %14, i32 25), !dbg !19
  store i32 %16, i32* %z__io_338, align 4, !dbg !19
  %17 = call i32 (...) @f90io_ldw_end(), !dbg !19
  store i32 %17, i32* %z__io_338, align 4, !dbg !19
  call void (...) @_mp_ecs_nest(), !dbg !19
  ret void, !dbg !15
}

define internal void @__nv_MAIN__F1L18_1_(i32* %__nv_MAIN__F1L18_1Arg0, i64* %__nv_MAIN__F1L18_1Arg1, i64* %__nv_MAIN__F1L18_1Arg2) #0 !dbg !21 {
L.entry:
  %.uplevelArgPack0002_400 = alloca %astruct.dt100, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L18_1Arg0, metadata !25, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L18_1Arg1, metadata !27, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L18_1Arg2, metadata !28, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.value(metadata i32 1, metadata !29, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.value(metadata i32 0, metadata !30, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.value(metadata i32 1, metadata !31, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.value(metadata i32 0, metadata !32, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.value(metadata i32 1, metadata !33, metadata !DIExpression()), !dbg !26
  br label %L.LB2_395

L.LB2_395:                                        ; preds = %L.entry
  br label %L.LB2_310

L.LB2_310:                                        ; preds = %L.LB2_395
  %0 = load i64, i64* %__nv_MAIN__F1L18_1Arg2, align 8, !dbg !34
  %1 = bitcast %astruct.dt100* %.uplevelArgPack0002_400 to i64*, !dbg !34
  store i64 %0, i64* %1, align 8, !dbg !34
  %2 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN_F1L19_2_ to i64*, !dbg !34
  %3 = bitcast %astruct.dt100* %.uplevelArgPack0002_400 to i64*, !dbg !34
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_teams(i64* null, i32 1, i64* %2, i64* %3), !dbg !34
  br label %L.LB2_331

L.LB2_331:                                        ; preds = %L.LB2_310
  ret void, !dbg !35
}

define internal void @__nv_MAIN_F1L19_2_(i32* %__nv_MAIN_F1L19_2Arg0, i64* %__nv_MAIN_F1L19_2Arg1, i64* %__nv_MAIN_F1L19_2Arg2) #0 !dbg !36 {
L.entry:
  %__gtid___nv_MAIN_F1L19_2__435 = alloca i32, align 4
  %.i0000p_318 = alloca i32, align 4
  %.i0001p_319 = alloca i32, align 4
  %.i0002p_320 = alloca i32, align 4
  %.i0003p_321 = alloca i32, align 4
  %i_317 = alloca i32, align 4
  %.du0001_349 = alloca i32, align 4
  %.de0001_350 = alloca i32, align 4
  %.di0001_351 = alloca i32, align 4
  %.ds0001_352 = alloca i32, align 4
  %.dl0001_354 = alloca i32, align 4
  %.dl0001.copy_429 = alloca i32, align 4
  %.de0001.copy_430 = alloca i32, align 4
  %.ds0001.copy_431 = alloca i32, align 4
  %.dX0001_353 = alloca i32, align 4
  %.dY0001_348 = alloca i32, align 4
  %.uplevelArgPack0003_454 = alloca %astruct.dt154, align 16
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L19_2Arg0, metadata !37, metadata !DIExpression()), !dbg !38
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L19_2Arg1, metadata !39, metadata !DIExpression()), !dbg !38
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L19_2Arg2, metadata !40, metadata !DIExpression()), !dbg !38
  call void @llvm.dbg.value(metadata i32 1, metadata !41, metadata !DIExpression()), !dbg !38
  call void @llvm.dbg.value(metadata i32 0, metadata !42, metadata !DIExpression()), !dbg !38
  call void @llvm.dbg.value(metadata i32 1, metadata !43, metadata !DIExpression()), !dbg !38
  call void @llvm.dbg.value(metadata i32 0, metadata !44, metadata !DIExpression()), !dbg !38
  call void @llvm.dbg.value(metadata i32 1, metadata !45, metadata !DIExpression()), !dbg !38
  %0 = load i32, i32* %__nv_MAIN_F1L19_2Arg0, align 4, !dbg !46
  store i32 %0, i32* %__gtid___nv_MAIN_F1L19_2__435, align 4, !dbg !46
  br label %L.LB4_418

L.LB4_418:                                        ; preds = %L.entry
  br label %L.LB4_313

L.LB4_313:                                        ; preds = %L.LB4_418
  br label %L.LB4_314

L.LB4_314:                                        ; preds = %L.LB4_313
  br label %L.LB4_315

L.LB4_315:                                        ; preds = %L.LB4_314
  store i32 0, i32* %.i0000p_318, align 4, !dbg !47
  store i32 1, i32* %.i0001p_319, align 4, !dbg !47
  store i32 100, i32* %.i0002p_320, align 4, !dbg !47
  store i32 1, i32* %.i0003p_321, align 4, !dbg !47
  %1 = load i32, i32* %.i0001p_319, align 4, !dbg !47
  call void @llvm.dbg.declare(metadata i32* %i_317, metadata !48, metadata !DIExpression()), !dbg !46
  store i32 %1, i32* %i_317, align 4, !dbg !47
  %2 = load i32, i32* %.i0002p_320, align 4, !dbg !47
  store i32 %2, i32* %.du0001_349, align 4, !dbg !47
  %3 = load i32, i32* %.i0002p_320, align 4, !dbg !47
  store i32 %3, i32* %.de0001_350, align 4, !dbg !47
  store i32 1, i32* %.di0001_351, align 4, !dbg !47
  %4 = load i32, i32* %.di0001_351, align 4, !dbg !47
  store i32 %4, i32* %.ds0001_352, align 4, !dbg !47
  %5 = load i32, i32* %.i0001p_319, align 4, !dbg !47
  store i32 %5, i32* %.dl0001_354, align 4, !dbg !47
  %6 = load i32, i32* %.dl0001_354, align 4, !dbg !47
  store i32 %6, i32* %.dl0001.copy_429, align 4, !dbg !47
  %7 = load i32, i32* %.de0001_350, align 4, !dbg !47
  store i32 %7, i32* %.de0001.copy_430, align 4, !dbg !47
  %8 = load i32, i32* %.ds0001_352, align 4, !dbg !47
  store i32 %8, i32* %.ds0001.copy_431, align 4, !dbg !47
  %9 = load i32, i32* %__gtid___nv_MAIN_F1L19_2__435, align 4, !dbg !47
  %10 = bitcast i32* %.i0000p_318 to i64*, !dbg !47
  %11 = bitcast i32* %.dl0001.copy_429 to i64*, !dbg !47
  %12 = bitcast i32* %.de0001.copy_430 to i64*, !dbg !47
  %13 = bitcast i32* %.ds0001.copy_431 to i64*, !dbg !47
  %14 = load i32, i32* %.ds0001.copy_431, align 4, !dbg !47
  call void @__kmpc_for_static_init_4(i64* null, i32 %9, i32 92, i64* %10, i64* %11, i64* %12, i64* %13, i32 %14, i32 1), !dbg !47
  %15 = load i32, i32* %.dl0001.copy_429, align 4, !dbg !47
  store i32 %15, i32* %.dl0001_354, align 4, !dbg !47
  %16 = load i32, i32* %.de0001.copy_430, align 4, !dbg !47
  store i32 %16, i32* %.de0001_350, align 4, !dbg !47
  %17 = load i32, i32* %.ds0001.copy_431, align 4, !dbg !47
  store i32 %17, i32* %.ds0001_352, align 4, !dbg !47
  %18 = load i32, i32* %.dl0001_354, align 4, !dbg !47
  store i32 %18, i32* %i_317, align 4, !dbg !47
  %19 = load i32, i32* %i_317, align 4, !dbg !47
  call void @llvm.dbg.value(metadata i32 %19, metadata !48, metadata !DIExpression()), !dbg !46
  store i32 %19, i32* %.dX0001_353, align 4, !dbg !47
  %20 = load i32, i32* %.dX0001_353, align 4, !dbg !47
  %21 = load i32, i32* %.du0001_349, align 4, !dbg !47
  %22 = icmp sgt i32 %20, %21, !dbg !47
  br i1 %22, label %L.LB4_347, label %L.LB4_485, !dbg !47

L.LB4_485:                                        ; preds = %L.LB4_315
  %23 = load i32, i32* %.du0001_349, align 4, !dbg !47
  %24 = load i32, i32* %.de0001_350, align 4, !dbg !47
  %25 = icmp slt i32 %23, %24, !dbg !47
  %26 = select i1 %25, i32 %23, i32 %24, !dbg !47
  store i32 %26, i32* %.de0001_350, align 4, !dbg !47
  %27 = load i32, i32* %.dX0001_353, align 4, !dbg !47
  store i32 %27, i32* %i_317, align 4, !dbg !47
  %28 = load i32, i32* %.di0001_351, align 4, !dbg !47
  %29 = load i32, i32* %.de0001_350, align 4, !dbg !47
  %30 = load i32, i32* %.dX0001_353, align 4, !dbg !47
  %31 = sub nsw i32 %29, %30, !dbg !47
  %32 = add nsw i32 %28, %31, !dbg !47
  %33 = load i32, i32* %.di0001_351, align 4, !dbg !47
  %34 = sdiv i32 %32, %33, !dbg !47
  store i32 %34, i32* %.dY0001_348, align 4, !dbg !47
  %35 = load i32, i32* %i_317, align 4, !dbg !47
  call void @llvm.dbg.value(metadata i32 %35, metadata !48, metadata !DIExpression()), !dbg !46
  store i32 %35, i32* %.i0001p_319, align 4, !dbg !47
  %36 = load i32, i32* %.de0001_350, align 4, !dbg !47
  store i32 %36, i32* %.i0002p_320, align 4, !dbg !47
  %37 = load i64, i64* %__nv_MAIN_F1L19_2Arg2, align 8, !dbg !47
  %38 = bitcast %astruct.dt154* %.uplevelArgPack0003_454 to i64*, !dbg !47
  store i64 %37, i64* %38, align 8, !dbg !47
  %39 = bitcast i32* %.i0001p_319 to i8*, !dbg !47
  %40 = bitcast %astruct.dt154* %.uplevelArgPack0003_454 to i8*, !dbg !47
  %41 = getelementptr i8, i8* %40, i64 8, !dbg !47
  %42 = bitcast i8* %41 to i8**, !dbg !47
  store i8* %39, i8** %42, align 8, !dbg !47
  %43 = bitcast i32* %.i0002p_320 to i8*, !dbg !47
  %44 = bitcast %astruct.dt154* %.uplevelArgPack0003_454 to i8*, !dbg !47
  %45 = getelementptr i8, i8* %44, i64 16, !dbg !47
  %46 = bitcast i8* %45 to i8**, !dbg !47
  store i8* %43, i8** %46, align 8, !dbg !47
  br label %L.LB4_461, !dbg !47

L.LB4_461:                                        ; preds = %L.LB4_485
  %47 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN_F1L20_3_ to i64*, !dbg !47
  %48 = bitcast %astruct.dt154* %.uplevelArgPack0003_454 to i64*, !dbg !47
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %47, i64* %48), !dbg !47
  br label %L.LB4_347

L.LB4_347:                                        ; preds = %L.LB4_461, %L.LB4_315
  %49 = load i32, i32* %__gtid___nv_MAIN_F1L19_2__435, align 4, !dbg !49
  call void @__kmpc_for_static_fini(i64* null, i32 %49), !dbg !49
  br label %L.LB4_328

L.LB4_328:                                        ; preds = %L.LB4_347
  br label %L.LB4_329

L.LB4_329:                                        ; preds = %L.LB4_328
  br label %L.LB4_330

L.LB4_330:                                        ; preds = %L.LB4_329
  ret void, !dbg !46
}

define internal void @__nv_MAIN_F1L20_3_(i32* %__nv_MAIN_F1L20_3Arg0, i64* %__nv_MAIN_F1L20_3Arg1, i64* %__nv_MAIN_F1L20_3Arg2) #0 !dbg !50 {
L.entry:
  %__gtid___nv_MAIN_F1L20_3__506 = alloca i32, align 4
  %.i0004p_326 = alloca i32, align 4
  %i_325 = alloca i32, align 4
  %.du0002p_361 = alloca i32, align 4
  %.de0002p_362 = alloca i32, align 4
  %.di0002p_363 = alloca i32, align 4
  %.ds0002p_364 = alloca i32, align 4
  %.dl0002p_366 = alloca i32, align 4
  %.dl0002p.copy_500 = alloca i32, align 4
  %.de0002p.copy_501 = alloca i32, align 4
  %.ds0002p.copy_502 = alloca i32, align 4
  %.dX0002p_365 = alloca i32, align 4
  %.dY0002p_360 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L20_3Arg0, metadata !51, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L20_3Arg1, metadata !53, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L20_3Arg2, metadata !54, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.value(metadata i32 1, metadata !55, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.value(metadata i32 0, metadata !56, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.value(metadata i32 1, metadata !57, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.value(metadata i32 0, metadata !58, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.value(metadata i32 1, metadata !59, metadata !DIExpression()), !dbg !52
  %0 = load i32, i32* %__nv_MAIN_F1L20_3Arg0, align 4, !dbg !60
  store i32 %0, i32* %__gtid___nv_MAIN_F1L20_3__506, align 4, !dbg !60
  br label %L.LB6_489

L.LB6_489:                                        ; preds = %L.entry
  br label %L.LB6_324

L.LB6_324:                                        ; preds = %L.LB6_489
  store i32 0, i32* %.i0004p_326, align 4, !dbg !61
  %1 = bitcast i64* %__nv_MAIN_F1L20_3Arg2 to i8*, !dbg !61
  %2 = getelementptr i8, i8* %1, i64 8, !dbg !61
  %3 = bitcast i8* %2 to i32**, !dbg !61
  %4 = load i32*, i32** %3, align 8, !dbg !61
  %5 = load i32, i32* %4, align 4, !dbg !61
  call void @llvm.dbg.declare(metadata i32* %i_325, metadata !62, metadata !DIExpression()), !dbg !60
  store i32 %5, i32* %i_325, align 4, !dbg !61
  %6 = bitcast i64* %__nv_MAIN_F1L20_3Arg2 to i8*, !dbg !61
  %7 = getelementptr i8, i8* %6, i64 16, !dbg !61
  %8 = bitcast i8* %7 to i32**, !dbg !61
  %9 = load i32*, i32** %8, align 8, !dbg !61
  %10 = load i32, i32* %9, align 4, !dbg !61
  store i32 %10, i32* %.du0002p_361, align 4, !dbg !61
  %11 = bitcast i64* %__nv_MAIN_F1L20_3Arg2 to i8*, !dbg !61
  %12 = getelementptr i8, i8* %11, i64 16, !dbg !61
  %13 = bitcast i8* %12 to i32**, !dbg !61
  %14 = load i32*, i32** %13, align 8, !dbg !61
  %15 = load i32, i32* %14, align 4, !dbg !61
  store i32 %15, i32* %.de0002p_362, align 4, !dbg !61
  store i32 1, i32* %.di0002p_363, align 4, !dbg !61
  %16 = load i32, i32* %.di0002p_363, align 4, !dbg !61
  store i32 %16, i32* %.ds0002p_364, align 4, !dbg !61
  %17 = bitcast i64* %__nv_MAIN_F1L20_3Arg2 to i8*, !dbg !61
  %18 = getelementptr i8, i8* %17, i64 8, !dbg !61
  %19 = bitcast i8* %18 to i32**, !dbg !61
  %20 = load i32*, i32** %19, align 8, !dbg !61
  %21 = load i32, i32* %20, align 4, !dbg !61
  store i32 %21, i32* %.dl0002p_366, align 4, !dbg !61
  %22 = load i32, i32* %.dl0002p_366, align 4, !dbg !61
  store i32 %22, i32* %.dl0002p.copy_500, align 4, !dbg !61
  %23 = load i32, i32* %.de0002p_362, align 4, !dbg !61
  store i32 %23, i32* %.de0002p.copy_501, align 4, !dbg !61
  %24 = load i32, i32* %.ds0002p_364, align 4, !dbg !61
  store i32 %24, i32* %.ds0002p.copy_502, align 4, !dbg !61
  %25 = load i32, i32* %__gtid___nv_MAIN_F1L20_3__506, align 4, !dbg !61
  %26 = bitcast i32* %.i0004p_326 to i64*, !dbg !61
  %27 = bitcast i32* %.dl0002p.copy_500 to i64*, !dbg !61
  %28 = bitcast i32* %.de0002p.copy_501 to i64*, !dbg !61
  %29 = bitcast i32* %.ds0002p.copy_502 to i64*, !dbg !61
  %30 = load i32, i32* %.ds0002p.copy_502, align 4, !dbg !61
  call void @__kmpc_for_static_init_4(i64* null, i32 %25, i32 34, i64* %26, i64* %27, i64* %28, i64* %29, i32 %30, i32 1), !dbg !61
  %31 = load i32, i32* %.dl0002p.copy_500, align 4, !dbg !61
  store i32 %31, i32* %.dl0002p_366, align 4, !dbg !61
  %32 = load i32, i32* %.de0002p.copy_501, align 4, !dbg !61
  store i32 %32, i32* %.de0002p_362, align 4, !dbg !61
  %33 = load i32, i32* %.ds0002p.copy_502, align 4, !dbg !61
  store i32 %33, i32* %.ds0002p_364, align 4, !dbg !61
  %34 = load i32, i32* %.dl0002p_366, align 4, !dbg !61
  store i32 %34, i32* %i_325, align 4, !dbg !61
  %35 = load i32, i32* %i_325, align 4, !dbg !61
  call void @llvm.dbg.value(metadata i32 %35, metadata !62, metadata !DIExpression()), !dbg !60
  store i32 %35, i32* %.dX0002p_365, align 4, !dbg !61
  %36 = load i32, i32* %.dX0002p_365, align 4, !dbg !61
  %37 = load i32, i32* %.du0002p_361, align 4, !dbg !61
  %38 = icmp sgt i32 %36, %37, !dbg !61
  br i1 %38, label %L.LB6_359, label %L.LB6_515, !dbg !61

L.LB6_515:                                        ; preds = %L.LB6_324
  %39 = load i32, i32* %.dX0002p_365, align 4, !dbg !61
  store i32 %39, i32* %i_325, align 4, !dbg !61
  %40 = load i32, i32* %.di0002p_363, align 4, !dbg !61
  %41 = load i32, i32* %.de0002p_362, align 4, !dbg !61
  %42 = load i32, i32* %.dX0002p_365, align 4, !dbg !61
  %43 = sub nsw i32 %41, %42, !dbg !61
  %44 = add nsw i32 %40, %43, !dbg !61
  %45 = load i32, i32* %.di0002p_363, align 4, !dbg !61
  %46 = sdiv i32 %44, %45, !dbg !61
  store i32 %46, i32* %.dY0002p_360, align 4, !dbg !61
  %47 = load i32, i32* %.dY0002p_360, align 4, !dbg !61
  %48 = icmp sle i32 %47, 0, !dbg !61
  br i1 %48, label %L.LB6_369, label %L.LB6_368, !dbg !61

L.LB6_368:                                        ; preds = %L.LB6_368, %L.LB6_515
  %49 = bitcast i64* %__nv_MAIN_F1L20_3Arg2 to i32**, !dbg !63
  %50 = load i32*, i32** %49, align 8, !dbg !63
  %51 = load i32, i32* %50, align 4, !dbg !63
  %52 = add nsw i32 %51, 1, !dbg !63
  %53 = bitcast i64* %__nv_MAIN_F1L20_3Arg2 to i32**, !dbg !63
  %54 = load i32*, i32** %53, align 8, !dbg !63
  store i32 %52, i32* %54, align 4, !dbg !63
  %55 = load i32, i32* %.di0002p_363, align 4, !dbg !60
  %56 = load i32, i32* %i_325, align 4, !dbg !60
  call void @llvm.dbg.value(metadata i32 %56, metadata !62, metadata !DIExpression()), !dbg !60
  %57 = add nsw i32 %55, %56, !dbg !60
  store i32 %57, i32* %i_325, align 4, !dbg !60
  %58 = load i32, i32* %.dY0002p_360, align 4, !dbg !60
  %59 = sub nsw i32 %58, 1, !dbg !60
  store i32 %59, i32* %.dY0002p_360, align 4, !dbg !60
  %60 = load i32, i32* %.dY0002p_360, align 4, !dbg !60
  %61 = icmp sgt i32 %60, 0, !dbg !60
  br i1 %61, label %L.LB6_368, label %L.LB6_369, !dbg !60

L.LB6_369:                                        ; preds = %L.LB6_368, %L.LB6_515
  br label %L.LB6_359

L.LB6_359:                                        ; preds = %L.LB6_369, %L.LB6_324
  %62 = load i32, i32* %__gtid___nv_MAIN_F1L20_3__506, align 4, !dbg !60
  call void @__kmpc_for_static_fini(i64* null, i32 %62), !dbg !60
  br label %L.LB6_327

L.LB6_327:                                        ; preds = %L.LB6_359
  ret void, !dbg !60
}

declare void @__kmpc_for_static_fini(i64*, i32) #0

declare void @__kmpc_fork_call(i64*, i32, i64*, i64*, ...) #0

declare void @__kmpc_end_serialized_parallel(i64*, i32) #0

declare void @__kmpc_serialized_parallel(i64*, i32) #0

declare void @__kmpc_for_static_init_4(i64*, i32, i32, i64*, i64*, i64*, i64*, i32, i32) #0

declare void @__kmpc_fork_teams(i64*, i32, i64*, i64*, ...) #0

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
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB151-missinglock3-orig-gpu-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb151_missinglock3_orig_gpu_yes", scope: !2, file: !3, line: 12, type: !6, scopeLine: 12, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!6 = !DISubroutineType(cc: DW_CC_program, types: !7)
!7 = !{null}
!8 = !DILocalVariable(name: "omp_sched_static", scope: !5, file: !3, type: !9)
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 0, scope: !5)
!11 = !DILocalVariable(name: "omp_proc_bind_false", scope: !5, file: !3, type: !9)
!12 = !DILocalVariable(name: "omp_proc_bind_true", scope: !5, file: !3, type: !9)
!13 = !DILocalVariable(name: "omp_lock_hint_none", scope: !5, file: !3, type: !9)
!14 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !5, file: !3, type: !9)
!15 = !DILocation(line: 27, column: 1, scope: !5)
!16 = !DILocation(line: 12, column: 1, scope: !5)
!17 = !DILocalVariable(name: "var", scope: !5, file: !3, type: !9)
!18 = !DILocation(line: 24, column: 1, scope: !5)
!19 = !DILocation(line: 26, column: 1, scope: !5)
!20 = !DILocalVariable(scope: !5, file: !3, type: !9, flags: DIFlagArtificial)
!21 = distinct !DISubprogram(name: "__nv_MAIN__F1L18_1", scope: !2, file: !3, line: 18, type: !22, scopeLine: 18, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!22 = !DISubroutineType(types: !23)
!23 = !{null, !9, !24, !24}
!24 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!25 = !DILocalVariable(name: "__nv_MAIN__F1L18_1Arg0", arg: 1, scope: !21, file: !3, type: !9)
!26 = !DILocation(line: 0, scope: !21)
!27 = !DILocalVariable(name: "__nv_MAIN__F1L18_1Arg1", arg: 2, scope: !21, file: !3, type: !24)
!28 = !DILocalVariable(name: "__nv_MAIN__F1L18_1Arg2", arg: 3, scope: !21, file: !3, type: !24)
!29 = !DILocalVariable(name: "omp_sched_static", scope: !21, file: !3, type: !9)
!30 = !DILocalVariable(name: "omp_proc_bind_false", scope: !21, file: !3, type: !9)
!31 = !DILocalVariable(name: "omp_proc_bind_true", scope: !21, file: !3, type: !9)
!32 = !DILocalVariable(name: "omp_lock_hint_none", scope: !21, file: !3, type: !9)
!33 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !21, file: !3, type: !9)
!34 = !DILocation(line: 19, column: 1, scope: !21)
!35 = !DILocation(line: 24, column: 1, scope: !21)
!36 = distinct !DISubprogram(name: "__nv_MAIN_F1L19_2", scope: !2, file: !3, line: 19, type: !22, scopeLine: 19, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!37 = !DILocalVariable(name: "__nv_MAIN_F1L19_2Arg0", arg: 1, scope: !36, file: !3, type: !9)
!38 = !DILocation(line: 0, scope: !36)
!39 = !DILocalVariable(name: "__nv_MAIN_F1L19_2Arg1", arg: 2, scope: !36, file: !3, type: !24)
!40 = !DILocalVariable(name: "__nv_MAIN_F1L19_2Arg2", arg: 3, scope: !36, file: !3, type: !24)
!41 = !DILocalVariable(name: "omp_sched_static", scope: !36, file: !3, type: !9)
!42 = !DILocalVariable(name: "omp_proc_bind_false", scope: !36, file: !3, type: !9)
!43 = !DILocalVariable(name: "omp_proc_bind_true", scope: !36, file: !3, type: !9)
!44 = !DILocalVariable(name: "omp_lock_hint_none", scope: !36, file: !3, type: !9)
!45 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !36, file: !3, type: !9)
!46 = !DILocation(line: 23, column: 1, scope: !36)
!47 = !DILocation(line: 20, column: 1, scope: !36)
!48 = !DILocalVariable(name: "i", scope: !36, file: !3, type: !9)
!49 = !DILocation(line: 22, column: 1, scope: !36)
!50 = distinct !DISubprogram(name: "__nv_MAIN_F1L20_3", scope: !2, file: !3, line: 20, type: !22, scopeLine: 20, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!51 = !DILocalVariable(name: "__nv_MAIN_F1L20_3Arg0", arg: 1, scope: !50, file: !3, type: !9)
!52 = !DILocation(line: 0, scope: !50)
!53 = !DILocalVariable(name: "__nv_MAIN_F1L20_3Arg1", arg: 2, scope: !50, file: !3, type: !24)
!54 = !DILocalVariable(name: "__nv_MAIN_F1L20_3Arg2", arg: 3, scope: !50, file: !3, type: !24)
!55 = !DILocalVariable(name: "omp_sched_static", scope: !50, file: !3, type: !9)
!56 = !DILocalVariable(name: "omp_proc_bind_false", scope: !50, file: !3, type: !9)
!57 = !DILocalVariable(name: "omp_proc_bind_true", scope: !50, file: !3, type: !9)
!58 = !DILocalVariable(name: "omp_lock_hint_none", scope: !50, file: !3, type: !9)
!59 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !50, file: !3, type: !9)
!60 = !DILocation(line: 22, column: 1, scope: !50)
!61 = !DILocation(line: 20, column: 1, scope: !50)
!62 = !DILocalVariable(name: "i", scope: !50, file: !3, type: !9)
!63 = !DILocation(line: 21, column: 1, scope: !50)

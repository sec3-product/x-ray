; ModuleID = '/tmp/DRB145-atomiccritical-orig-gpu-no-7c9598.ll'
source_filename = "/tmp/DRB145-atomiccritical-orig-gpu-no-7c9598.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%astruct.dt56 = type <{ i8* }>
%astruct.dt89 = type <{ [8 x i8] }>
%astruct.dt143 = type <{ [8 x i8], i8*, i8*, i8* }>

@.C328_MAIN_ = internal constant i32 101
@.C316_MAIN_ = internal constant i32 200
@.C285_MAIN_ = internal constant i32 1
@.C283_MAIN_ = internal constant i32 0
@.C328___nv_MAIN__F1L18_1 = internal constant i32 101
@.C316___nv_MAIN__F1L18_1 = internal constant i32 200
@.C285___nv_MAIN__F1L18_1 = internal constant i32 1
@.C283___nv_MAIN__F1L18_1 = internal constant i32 0
@.C328___nv_MAIN_F1L19_2 = internal constant i32 101
@.C316___nv_MAIN_F1L19_2 = internal constant i32 200
@.C285___nv_MAIN_F1L19_2 = internal constant i32 1
@.C283___nv_MAIN_F1L19_2 = internal constant i32 0
@.C328___nv_MAIN_F1L20_3 = internal constant i32 101
@.C285___nv_MAIN_F1L20_3 = internal constant i32 1
@.C283___nv_MAIN_F1L20_3 = internal constant i32 0

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__372 = alloca i32, align 4
  %var_305 = alloca i32, align 4
  %.uplevelArgPack0001_369 = alloca %astruct.dt56, align 8
  call void @llvm.dbg.value(metadata i32 1, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !14, metadata !DIExpression()), !dbg !10
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !15
  store i32 %0, i32* %__gtid_MAIN__372, align 4, !dbg !15
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !16
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !16
  call void (i8*, ...) %2(i8* %1), !dbg !16
  br label %L.LB1_363

L.LB1_363:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %var_305, metadata !17, metadata !DIExpression()), !dbg !10
  store i32 0, i32* %var_305, align 4, !dbg !18
  %3 = bitcast i32* %var_305 to i8*, !dbg !19
  %4 = bitcast %astruct.dt56* %.uplevelArgPack0001_369 to i8**, !dbg !19
  store i8* %3, i8** %4, align 8, !dbg !19
  %5 = bitcast %astruct.dt56* %.uplevelArgPack0001_369 to i64*, !dbg !19
  call void @__nv_MAIN__F1L18_1_(i32* %__gtid_MAIN__372, i64* null, i64* %5), !dbg !19
  ret void, !dbg !15
}

define internal void @__nv_MAIN__F1L18_1_(i32* %__nv_MAIN__F1L18_1Arg0, i64* %__nv_MAIN__F1L18_1Arg1, i64* %__nv_MAIN__F1L18_1Arg2) #0 !dbg !20 {
L.entry:
  %.uplevelArgPack0002_386 = alloca %astruct.dt89, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L18_1Arg0, metadata !24, metadata !DIExpression()), !dbg !25
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L18_1Arg1, metadata !26, metadata !DIExpression()), !dbg !25
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L18_1Arg2, metadata !27, metadata !DIExpression()), !dbg !25
  call void @llvm.dbg.value(metadata i32 1, metadata !28, metadata !DIExpression()), !dbg !25
  call void @llvm.dbg.value(metadata i32 0, metadata !29, metadata !DIExpression()), !dbg !25
  call void @llvm.dbg.value(metadata i32 1, metadata !30, metadata !DIExpression()), !dbg !25
  call void @llvm.dbg.value(metadata i32 0, metadata !31, metadata !DIExpression()), !dbg !25
  call void @llvm.dbg.value(metadata i32 1, metadata !32, metadata !DIExpression()), !dbg !25
  br label %L.LB2_381

L.LB2_381:                                        ; preds = %L.entry
  br label %L.LB2_309

L.LB2_309:                                        ; preds = %L.LB2_381
  %0 = load i64, i64* %__nv_MAIN__F1L18_1Arg2, align 8, !dbg !33
  %1 = bitcast %astruct.dt89* %.uplevelArgPack0002_386 to i64*, !dbg !33
  store i64 %0, i64* %1, align 8, !dbg !33
  %2 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN_F1L19_2_ to i64*, !dbg !33
  %3 = bitcast %astruct.dt89* %.uplevelArgPack0002_386 to i64*, !dbg !33
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_teams(i64* null, i32 1, i64* %2, i64* %3), !dbg !33
  br label %L.LB2_333

L.LB2_333:                                        ; preds = %L.LB2_309
  ret void, !dbg !34
}

define internal void @__nv_MAIN_F1L19_2_(i32* %__nv_MAIN_F1L19_2Arg0, i64* %__nv_MAIN_F1L19_2Arg1, i64* %__nv_MAIN_F1L19_2Arg2) #0 !dbg !35 {
L.entry:
  %__gtid___nv_MAIN_F1L19_2__422 = alloca i32, align 4
  %var_313 = alloca i32, align 4
  %.i0000p_318 = alloca i32, align 4
  %.i0001p_319 = alloca i32, align 4
  %.i0002p_320 = alloca i32, align 4
  %.i0003p_321 = alloca i32, align 4
  %i_317 = alloca i32, align 4
  %.du0001_340 = alloca i32, align 4
  %.de0001_341 = alloca i32, align 4
  %.di0001_342 = alloca i32, align 4
  %.ds0001_343 = alloca i32, align 4
  %.dl0001_345 = alloca i32, align 4
  %.dl0001.copy_416 = alloca i32, align 4
  %.de0001.copy_417 = alloca i32, align 4
  %.ds0001.copy_418 = alloca i32, align 4
  %.dX0001_344 = alloca i32, align 4
  %.dY0001_339 = alloca i32, align 4
  %.uplevelArgPack0003_441 = alloca %astruct.dt143, align 16
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L19_2Arg0, metadata !36, metadata !DIExpression()), !dbg !37
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L19_2Arg1, metadata !38, metadata !DIExpression()), !dbg !37
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L19_2Arg2, metadata !39, metadata !DIExpression()), !dbg !37
  call void @llvm.dbg.value(metadata i32 1, metadata !40, metadata !DIExpression()), !dbg !37
  call void @llvm.dbg.value(metadata i32 0, metadata !41, metadata !DIExpression()), !dbg !37
  call void @llvm.dbg.value(metadata i32 1, metadata !42, metadata !DIExpression()), !dbg !37
  call void @llvm.dbg.value(metadata i32 0, metadata !43, metadata !DIExpression()), !dbg !37
  call void @llvm.dbg.value(metadata i32 1, metadata !44, metadata !DIExpression()), !dbg !37
  %0 = load i32, i32* %__nv_MAIN_F1L19_2Arg0, align 4, !dbg !45
  store i32 %0, i32* %__gtid___nv_MAIN_F1L19_2__422, align 4, !dbg !45
  br label %L.LB4_404

L.LB4_404:                                        ; preds = %L.entry
  br label %L.LB4_312

L.LB4_312:                                        ; preds = %L.LB4_404
  call void @llvm.dbg.declare(metadata i32* %var_313, metadata !46, metadata !DIExpression()), !dbg !45
  store i32 0, i32* %var_313, align 4, !dbg !47
  br label %L.LB4_314

L.LB4_314:                                        ; preds = %L.LB4_312
  br label %L.LB4_315

L.LB4_315:                                        ; preds = %L.LB4_314
  store i32 0, i32* %.i0000p_318, align 4, !dbg !48
  store i32 1, i32* %.i0001p_319, align 4, !dbg !48
  store i32 200, i32* %.i0002p_320, align 4, !dbg !48
  store i32 1, i32* %.i0003p_321, align 4, !dbg !48
  %1 = load i32, i32* %.i0001p_319, align 4, !dbg !48
  call void @llvm.dbg.declare(metadata i32* %i_317, metadata !49, metadata !DIExpression()), !dbg !45
  store i32 %1, i32* %i_317, align 4, !dbg !48
  %2 = load i32, i32* %.i0002p_320, align 4, !dbg !48
  store i32 %2, i32* %.du0001_340, align 4, !dbg !48
  %3 = load i32, i32* %.i0002p_320, align 4, !dbg !48
  store i32 %3, i32* %.de0001_341, align 4, !dbg !48
  store i32 1, i32* %.di0001_342, align 4, !dbg !48
  %4 = load i32, i32* %.di0001_342, align 4, !dbg !48
  store i32 %4, i32* %.ds0001_343, align 4, !dbg !48
  %5 = load i32, i32* %.i0001p_319, align 4, !dbg !48
  store i32 %5, i32* %.dl0001_345, align 4, !dbg !48
  %6 = load i32, i32* %.dl0001_345, align 4, !dbg !48
  store i32 %6, i32* %.dl0001.copy_416, align 4, !dbg !48
  %7 = load i32, i32* %.de0001_341, align 4, !dbg !48
  store i32 %7, i32* %.de0001.copy_417, align 4, !dbg !48
  %8 = load i32, i32* %.ds0001_343, align 4, !dbg !48
  store i32 %8, i32* %.ds0001.copy_418, align 4, !dbg !48
  %9 = load i32, i32* %__gtid___nv_MAIN_F1L19_2__422, align 4, !dbg !48
  %10 = bitcast i32* %.i0000p_318 to i64*, !dbg !48
  %11 = bitcast i32* %.dl0001.copy_416 to i64*, !dbg !48
  %12 = bitcast i32* %.de0001.copy_417 to i64*, !dbg !48
  %13 = bitcast i32* %.ds0001.copy_418 to i64*, !dbg !48
  %14 = load i32, i32* %.ds0001.copy_418, align 4, !dbg !48
  call void @__kmpc_for_static_init_4(i64* null, i32 %9, i32 92, i64* %10, i64* %11, i64* %12, i64* %13, i32 %14, i32 1), !dbg !48
  %15 = load i32, i32* %.dl0001.copy_416, align 4, !dbg !48
  store i32 %15, i32* %.dl0001_345, align 4, !dbg !48
  %16 = load i32, i32* %.de0001.copy_417, align 4, !dbg !48
  store i32 %16, i32* %.de0001_341, align 4, !dbg !48
  %17 = load i32, i32* %.ds0001.copy_418, align 4, !dbg !48
  store i32 %17, i32* %.ds0001_343, align 4, !dbg !48
  %18 = load i32, i32* %.dl0001_345, align 4, !dbg !48
  store i32 %18, i32* %i_317, align 4, !dbg !48
  %19 = load i32, i32* %i_317, align 4, !dbg !48
  call void @llvm.dbg.value(metadata i32 %19, metadata !49, metadata !DIExpression()), !dbg !45
  store i32 %19, i32* %.dX0001_344, align 4, !dbg !48
  %20 = load i32, i32* %.dX0001_344, align 4, !dbg !48
  %21 = load i32, i32* %.du0001_340, align 4, !dbg !48
  %22 = icmp sgt i32 %20, %21, !dbg !48
  br i1 %22, label %L.LB4_338, label %L.LB4_476, !dbg !48

L.LB4_476:                                        ; preds = %L.LB4_315
  %23 = load i32, i32* %.du0001_340, align 4, !dbg !48
  %24 = load i32, i32* %.de0001_341, align 4, !dbg !48
  %25 = icmp slt i32 %23, %24, !dbg !48
  %26 = select i1 %25, i32 %23, i32 %24, !dbg !48
  store i32 %26, i32* %.de0001_341, align 4, !dbg !48
  %27 = load i32, i32* %.dX0001_344, align 4, !dbg !48
  store i32 %27, i32* %i_317, align 4, !dbg !48
  %28 = load i32, i32* %.di0001_342, align 4, !dbg !48
  %29 = load i32, i32* %.de0001_341, align 4, !dbg !48
  %30 = load i32, i32* %.dX0001_344, align 4, !dbg !48
  %31 = sub nsw i32 %29, %30, !dbg !48
  %32 = add nsw i32 %28, %31, !dbg !48
  %33 = load i32, i32* %.di0001_342, align 4, !dbg !48
  %34 = sdiv i32 %32, %33, !dbg !48
  store i32 %34, i32* %.dY0001_339, align 4, !dbg !48
  %35 = load i32, i32* %i_317, align 4, !dbg !48
  call void @llvm.dbg.value(metadata i32 %35, metadata !49, metadata !DIExpression()), !dbg !45
  store i32 %35, i32* %.i0001p_319, align 4, !dbg !48
  %36 = load i32, i32* %.de0001_341, align 4, !dbg !48
  store i32 %36, i32* %.i0002p_320, align 4, !dbg !48
  %37 = load i64, i64* %__nv_MAIN_F1L19_2Arg2, align 8, !dbg !48
  %38 = bitcast %astruct.dt143* %.uplevelArgPack0003_441 to i64*, !dbg !48
  store i64 %37, i64* %38, align 8, !dbg !48
  %39 = bitcast i32* %var_313 to i8*, !dbg !48
  %40 = bitcast %astruct.dt143* %.uplevelArgPack0003_441 to i8*, !dbg !48
  %41 = getelementptr i8, i8* %40, i64 8, !dbg !48
  %42 = bitcast i8* %41 to i8**, !dbg !48
  store i8* %39, i8** %42, align 8, !dbg !48
  %43 = bitcast i32* %.i0001p_319 to i8*, !dbg !48
  %44 = bitcast %astruct.dt143* %.uplevelArgPack0003_441 to i8*, !dbg !48
  %45 = getelementptr i8, i8* %44, i64 16, !dbg !48
  %46 = bitcast i8* %45 to i8**, !dbg !48
  store i8* %43, i8** %46, align 8, !dbg !48
  %47 = bitcast i32* %.i0002p_320 to i8*, !dbg !48
  %48 = bitcast %astruct.dt143* %.uplevelArgPack0003_441 to i8*, !dbg !48
  %49 = getelementptr i8, i8* %48, i64 24, !dbg !48
  %50 = bitcast i8* %49 to i8**, !dbg !48
  store i8* %47, i8** %50, align 8, !dbg !48
  br label %L.LB4_450, !dbg !48

L.LB4_450:                                        ; preds = %L.LB4_476
  %51 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN_F1L20_3_ to i64*, !dbg !48
  %52 = bitcast %astruct.dt143* %.uplevelArgPack0003_441 to i64*, !dbg !48
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %51, i64* %52), !dbg !48
  br label %L.LB4_338

L.LB4_338:                                        ; preds = %L.LB4_450, %L.LB4_315
  %53 = load i32, i32* %__gtid___nv_MAIN_F1L19_2__422, align 4, !dbg !50
  call void @__kmpc_for_static_fini(i64* null, i32 %53), !dbg !50
  br label %L.LB4_330

L.LB4_330:                                        ; preds = %L.LB4_338
  br label %L.LB4_331

L.LB4_331:                                        ; preds = %L.LB4_330
  %54 = call i32 (...) @_mp_bcs_nest_red(), !dbg !45
  %55 = call i32 (...) @_mp_bcs_nest_red(), !dbg !45
  %56 = load i32, i32* %var_313, align 4, !dbg !45
  call void @llvm.dbg.value(metadata i32 %56, metadata !46, metadata !DIExpression()), !dbg !45
  %57 = bitcast i64* %__nv_MAIN_F1L19_2Arg2 to i32**, !dbg !45
  %58 = load i32*, i32** %57, align 8, !dbg !45
  %59 = load i32, i32* %58, align 4, !dbg !45
  %60 = add nsw i32 %56, %59, !dbg !45
  %61 = bitcast i64* %__nv_MAIN_F1L19_2Arg2 to i32**, !dbg !45
  %62 = load i32*, i32** %61, align 8, !dbg !45
  store i32 %60, i32* %62, align 4, !dbg !45
  %63 = call i32 (...) @_mp_ecs_nest_red(), !dbg !45
  %64 = call i32 (...) @_mp_ecs_nest_red(), !dbg !45
  br label %L.LB4_332

L.LB4_332:                                        ; preds = %L.LB4_331
  ret void, !dbg !45
}

define internal void @__nv_MAIN_F1L20_3_(i32* %__nv_MAIN_F1L20_3Arg0, i64* %__nv_MAIN_F1L20_3Arg1, i64* %__nv_MAIN_F1L20_3Arg2) #0 !dbg !51 {
L.entry:
  %__gtid___nv_MAIN_F1L20_3__498 = alloca i32, align 4
  %var_325 = alloca i32, align 4
  %.i0004p_327 = alloca i32, align 4
  %i_326 = alloca i32, align 4
  %.du0002p_352 = alloca i32, align 4
  %.de0002p_353 = alloca i32, align 4
  %.di0002p_354 = alloca i32, align 4
  %.ds0002p_355 = alloca i32, align 4
  %.dl0002p_357 = alloca i32, align 4
  %.dl0002p.copy_492 = alloca i32, align 4
  %.de0002p.copy_493 = alloca i32, align 4
  %.ds0002p.copy_494 = alloca i32, align 4
  %.dX0002p_356 = alloca i32, align 4
  %.dY0002p_351 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L20_3Arg0, metadata !52, metadata !DIExpression()), !dbg !53
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L20_3Arg1, metadata !54, metadata !DIExpression()), !dbg !53
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L20_3Arg2, metadata !55, metadata !DIExpression()), !dbg !53
  call void @llvm.dbg.value(metadata i32 1, metadata !56, metadata !DIExpression()), !dbg !53
  call void @llvm.dbg.value(metadata i32 0, metadata !57, metadata !DIExpression()), !dbg !53
  call void @llvm.dbg.value(metadata i32 1, metadata !58, metadata !DIExpression()), !dbg !53
  call void @llvm.dbg.value(metadata i32 0, metadata !59, metadata !DIExpression()), !dbg !53
  call void @llvm.dbg.value(metadata i32 1, metadata !60, metadata !DIExpression()), !dbg !53
  %0 = load i32, i32* %__nv_MAIN_F1L20_3Arg0, align 4, !dbg !61
  store i32 %0, i32* %__gtid___nv_MAIN_F1L20_3__498, align 4, !dbg !61
  br label %L.LB6_480

L.LB6_480:                                        ; preds = %L.entry
  br label %L.LB6_324

L.LB6_324:                                        ; preds = %L.LB6_480
  call void @llvm.dbg.declare(metadata i32* %var_325, metadata !62, metadata !DIExpression()), !dbg !61
  store i32 0, i32* %var_325, align 4, !dbg !63
  store i32 0, i32* %.i0004p_327, align 4, !dbg !63
  %1 = bitcast i64* %__nv_MAIN_F1L20_3Arg2 to i8*, !dbg !63
  %2 = getelementptr i8, i8* %1, i64 16, !dbg !63
  %3 = bitcast i8* %2 to i32**, !dbg !63
  %4 = load i32*, i32** %3, align 8, !dbg !63
  %5 = load i32, i32* %4, align 4, !dbg !63
  call void @llvm.dbg.declare(metadata i32* %i_326, metadata !64, metadata !DIExpression()), !dbg !61
  store i32 %5, i32* %i_326, align 4, !dbg !63
  %6 = bitcast i64* %__nv_MAIN_F1L20_3Arg2 to i8*, !dbg !63
  %7 = getelementptr i8, i8* %6, i64 24, !dbg !63
  %8 = bitcast i8* %7 to i32**, !dbg !63
  %9 = load i32*, i32** %8, align 8, !dbg !63
  %10 = load i32, i32* %9, align 4, !dbg !63
  store i32 %10, i32* %.du0002p_352, align 4, !dbg !63
  %11 = bitcast i64* %__nv_MAIN_F1L20_3Arg2 to i8*, !dbg !63
  %12 = getelementptr i8, i8* %11, i64 24, !dbg !63
  %13 = bitcast i8* %12 to i32**, !dbg !63
  %14 = load i32*, i32** %13, align 8, !dbg !63
  %15 = load i32, i32* %14, align 4, !dbg !63
  store i32 %15, i32* %.de0002p_353, align 4, !dbg !63
  store i32 1, i32* %.di0002p_354, align 4, !dbg !63
  %16 = load i32, i32* %.di0002p_354, align 4, !dbg !63
  store i32 %16, i32* %.ds0002p_355, align 4, !dbg !63
  %17 = bitcast i64* %__nv_MAIN_F1L20_3Arg2 to i8*, !dbg !63
  %18 = getelementptr i8, i8* %17, i64 16, !dbg !63
  %19 = bitcast i8* %18 to i32**, !dbg !63
  %20 = load i32*, i32** %19, align 8, !dbg !63
  %21 = load i32, i32* %20, align 4, !dbg !63
  store i32 %21, i32* %.dl0002p_357, align 4, !dbg !63
  %22 = load i32, i32* %.dl0002p_357, align 4, !dbg !63
  store i32 %22, i32* %.dl0002p.copy_492, align 4, !dbg !63
  %23 = load i32, i32* %.de0002p_353, align 4, !dbg !63
  store i32 %23, i32* %.de0002p.copy_493, align 4, !dbg !63
  %24 = load i32, i32* %.ds0002p_355, align 4, !dbg !63
  store i32 %24, i32* %.ds0002p.copy_494, align 4, !dbg !63
  %25 = load i32, i32* %__gtid___nv_MAIN_F1L20_3__498, align 4, !dbg !63
  %26 = bitcast i32* %.i0004p_327 to i64*, !dbg !63
  %27 = bitcast i32* %.dl0002p.copy_492 to i64*, !dbg !63
  %28 = bitcast i32* %.de0002p.copy_493 to i64*, !dbg !63
  %29 = bitcast i32* %.ds0002p.copy_494 to i64*, !dbg !63
  %30 = load i32, i32* %.ds0002p.copy_494, align 4, !dbg !63
  call void @__kmpc_for_static_init_4(i64* null, i32 %25, i32 34, i64* %26, i64* %27, i64* %28, i64* %29, i32 %30, i32 1), !dbg !63
  %31 = load i32, i32* %.dl0002p.copy_492, align 4, !dbg !63
  store i32 %31, i32* %.dl0002p_357, align 4, !dbg !63
  %32 = load i32, i32* %.de0002p.copy_493, align 4, !dbg !63
  store i32 %32, i32* %.de0002p_353, align 4, !dbg !63
  %33 = load i32, i32* %.ds0002p.copy_494, align 4, !dbg !63
  store i32 %33, i32* %.ds0002p_355, align 4, !dbg !63
  %34 = load i32, i32* %.dl0002p_357, align 4, !dbg !63
  store i32 %34, i32* %i_326, align 4, !dbg !63
  %35 = load i32, i32* %i_326, align 4, !dbg !63
  call void @llvm.dbg.value(metadata i32 %35, metadata !64, metadata !DIExpression()), !dbg !61
  store i32 %35, i32* %.dX0002p_356, align 4, !dbg !63
  %36 = load i32, i32* %.dX0002p_356, align 4, !dbg !63
  %37 = load i32, i32* %.du0002p_352, align 4, !dbg !63
  %38 = icmp sgt i32 %36, %37, !dbg !63
  br i1 %38, label %L.LB6_350, label %L.LB6_508, !dbg !63

L.LB6_508:                                        ; preds = %L.LB6_324
  %39 = load i32, i32* %.dX0002p_356, align 4, !dbg !63
  store i32 %39, i32* %i_326, align 4, !dbg !63
  %40 = load i32, i32* %.di0002p_354, align 4, !dbg !63
  %41 = load i32, i32* %.de0002p_353, align 4, !dbg !63
  %42 = load i32, i32* %.dX0002p_356, align 4, !dbg !63
  %43 = sub nsw i32 %41, %42, !dbg !63
  %44 = add nsw i32 %40, %43, !dbg !63
  %45 = load i32, i32* %.di0002p_354, align 4, !dbg !63
  %46 = sdiv i32 %44, %45, !dbg !63
  store i32 %46, i32* %.dY0002p_351, align 4, !dbg !63
  %47 = load i32, i32* %.dY0002p_351, align 4, !dbg !63
  %48 = icmp sle i32 %47, 0, !dbg !63
  br i1 %48, label %L.LB6_360, label %L.LB6_359, !dbg !63

L.LB6_359:                                        ; preds = %L.LB6_361, %L.LB6_508
  %49 = load i32, i32* %var_325, align 4, !dbg !65
  call void @llvm.dbg.value(metadata i32 %49, metadata !62, metadata !DIExpression()), !dbg !61
  %50 = icmp sge i32 %49, 101, !dbg !65
  br i1 %50, label %L.LB6_361, label %L.LB6_509, !dbg !65

L.LB6_509:                                        ; preds = %L.LB6_359
  %51 = load i32, i32* %var_325, align 4, !dbg !66
  call void @llvm.dbg.value(metadata i32 %51, metadata !62, metadata !DIExpression()), !dbg !61
  %52 = add nsw i32 %51, 1, !dbg !66
  store i32 %52, i32* %var_325, align 4, !dbg !66
  br label %L.LB6_361

L.LB6_361:                                        ; preds = %L.LB6_509, %L.LB6_359
  %53 = load i32, i32* %.di0002p_354, align 4, !dbg !61
  %54 = load i32, i32* %i_326, align 4, !dbg !61
  call void @llvm.dbg.value(metadata i32 %54, metadata !64, metadata !DIExpression()), !dbg !61
  %55 = add nsw i32 %53, %54, !dbg !61
  store i32 %55, i32* %i_326, align 4, !dbg !61
  %56 = load i32, i32* %.dY0002p_351, align 4, !dbg !61
  %57 = sub nsw i32 %56, 1, !dbg !61
  store i32 %57, i32* %.dY0002p_351, align 4, !dbg !61
  %58 = load i32, i32* %.dY0002p_351, align 4, !dbg !61
  %59 = icmp sgt i32 %58, 0, !dbg !61
  br i1 %59, label %L.LB6_359, label %L.LB6_360, !dbg !61

L.LB6_360:                                        ; preds = %L.LB6_361, %L.LB6_508
  br label %L.LB6_350

L.LB6_350:                                        ; preds = %L.LB6_360, %L.LB6_324
  %60 = load i32, i32* %__gtid___nv_MAIN_F1L20_3__498, align 4, !dbg !61
  call void @__kmpc_for_static_fini(i64* null, i32 %60), !dbg !61
  %61 = call i32 (...) @_mp_bcs_nest_red(), !dbg !61
  %62 = call i32 (...) @_mp_bcs_nest_red(), !dbg !61
  %63 = load i32, i32* %var_325, align 4, !dbg !61
  call void @llvm.dbg.value(metadata i32 %63, metadata !62, metadata !DIExpression()), !dbg !61
  %64 = bitcast i64* %__nv_MAIN_F1L20_3Arg2 to i8*, !dbg !61
  %65 = getelementptr i8, i8* %64, i64 8, !dbg !61
  %66 = bitcast i8* %65 to i32**, !dbg !61
  %67 = load i32*, i32** %66, align 8, !dbg !61
  %68 = load i32, i32* %67, align 4, !dbg !61
  %69 = add nsw i32 %63, %68, !dbg !61
  %70 = bitcast i64* %__nv_MAIN_F1L20_3Arg2 to i8*, !dbg !61
  %71 = getelementptr i8, i8* %70, i64 8, !dbg !61
  %72 = bitcast i8* %71 to i32**, !dbg !61
  %73 = load i32*, i32** %72, align 8, !dbg !61
  store i32 %69, i32* %73, align 4, !dbg !61
  %74 = call i32 (...) @_mp_ecs_nest_red(), !dbg !61
  %75 = call i32 (...) @_mp_ecs_nest_red(), !dbg !61
  br label %L.LB6_329

L.LB6_329:                                        ; preds = %L.LB6_350
  ret void, !dbg !61
}

declare signext i32 @_mp_ecs_nest_red(...) #0

declare signext i32 @_mp_bcs_nest_red(...) #0

declare void @__kmpc_for_static_fini(i64*, i32) #0

declare void @__kmpc_fork_call(i64*, i32, i64*, i64*, ...) #0

declare void @__kmpc_end_serialized_parallel(i64*, i32) #0

declare void @__kmpc_serialized_parallel(i64*, i32) #0

declare void @__kmpc_for_static_init_4(i64*, i32, i32, i64*, i64*, i64*, i64*, i32, i32) #0

declare void @__kmpc_fork_teams(i64*, i32, i64*, i64*, ...) #0

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
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB145-atomiccritical-orig-gpu-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb145_atomiccritical_orig_gpu_no", scope: !2, file: !3, line: 11, type: !6, scopeLine: 11, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!6 = !DISubroutineType(cc: DW_CC_program, types: !7)
!7 = !{null}
!8 = !DILocalVariable(name: "omp_sched_static", scope: !5, file: !3, type: !9)
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 0, scope: !5)
!11 = !DILocalVariable(name: "omp_proc_bind_false", scope: !5, file: !3, type: !9)
!12 = !DILocalVariable(name: "omp_proc_bind_true", scope: !5, file: !3, type: !9)
!13 = !DILocalVariable(name: "omp_lock_hint_none", scope: !5, file: !3, type: !9)
!14 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !5, file: !3, type: !9)
!15 = !DILocation(line: 28, column: 1, scope: !5)
!16 = !DILocation(line: 11, column: 1, scope: !5)
!17 = !DILocalVariable(name: "var", scope: !5, file: !3, type: !9)
!18 = !DILocation(line: 16, column: 1, scope: !5)
!19 = !DILocation(line: 26, column: 1, scope: !5)
!20 = distinct !DISubprogram(name: "__nv_MAIN__F1L18_1", scope: !2, file: !3, line: 18, type: !21, scopeLine: 18, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!21 = !DISubroutineType(types: !22)
!22 = !{null, !9, !23, !23}
!23 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!24 = !DILocalVariable(name: "__nv_MAIN__F1L18_1Arg0", arg: 1, scope: !20, file: !3, type: !9)
!25 = !DILocation(line: 0, scope: !20)
!26 = !DILocalVariable(name: "__nv_MAIN__F1L18_1Arg1", arg: 2, scope: !20, file: !3, type: !23)
!27 = !DILocalVariable(name: "__nv_MAIN__F1L18_1Arg2", arg: 3, scope: !20, file: !3, type: !23)
!28 = !DILocalVariable(name: "omp_sched_static", scope: !20, file: !3, type: !9)
!29 = !DILocalVariable(name: "omp_proc_bind_false", scope: !20, file: !3, type: !9)
!30 = !DILocalVariable(name: "omp_proc_bind_true", scope: !20, file: !3, type: !9)
!31 = !DILocalVariable(name: "omp_lock_hint_none", scope: !20, file: !3, type: !9)
!32 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !20, file: !3, type: !9)
!33 = !DILocation(line: 19, column: 1, scope: !20)
!34 = !DILocation(line: 26, column: 1, scope: !20)
!35 = distinct !DISubprogram(name: "__nv_MAIN_F1L19_2", scope: !2, file: !3, line: 19, type: !21, scopeLine: 19, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!36 = !DILocalVariable(name: "__nv_MAIN_F1L19_2Arg0", arg: 1, scope: !35, file: !3, type: !9)
!37 = !DILocation(line: 0, scope: !35)
!38 = !DILocalVariable(name: "__nv_MAIN_F1L19_2Arg1", arg: 2, scope: !35, file: !3, type: !23)
!39 = !DILocalVariable(name: "__nv_MAIN_F1L19_2Arg2", arg: 3, scope: !35, file: !3, type: !23)
!40 = !DILocalVariable(name: "omp_sched_static", scope: !35, file: !3, type: !9)
!41 = !DILocalVariable(name: "omp_proc_bind_false", scope: !35, file: !3, type: !9)
!42 = !DILocalVariable(name: "omp_proc_bind_true", scope: !35, file: !3, type: !9)
!43 = !DILocalVariable(name: "omp_lock_hint_none", scope: !35, file: !3, type: !9)
!44 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !35, file: !3, type: !9)
!45 = !DILocation(line: 25, column: 1, scope: !35)
!46 = !DILocalVariable(name: "var", scope: !35, file: !3, type: !9)
!47 = !DILocation(line: 19, column: 1, scope: !35)
!48 = !DILocation(line: 20, column: 1, scope: !35)
!49 = !DILocalVariable(name: "i", scope: !35, file: !3, type: !9)
!50 = !DILocation(line: 24, column: 1, scope: !35)
!51 = distinct !DISubprogram(name: "__nv_MAIN_F1L20_3", scope: !2, file: !3, line: 20, type: !21, scopeLine: 20, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!52 = !DILocalVariable(name: "__nv_MAIN_F1L20_3Arg0", arg: 1, scope: !51, file: !3, type: !9)
!53 = !DILocation(line: 0, scope: !51)
!54 = !DILocalVariable(name: "__nv_MAIN_F1L20_3Arg1", arg: 2, scope: !51, file: !3, type: !23)
!55 = !DILocalVariable(name: "__nv_MAIN_F1L20_3Arg2", arg: 3, scope: !51, file: !3, type: !23)
!56 = !DILocalVariable(name: "omp_sched_static", scope: !51, file: !3, type: !9)
!57 = !DILocalVariable(name: "omp_proc_bind_false", scope: !51, file: !3, type: !9)
!58 = !DILocalVariable(name: "omp_proc_bind_true", scope: !51, file: !3, type: !9)
!59 = !DILocalVariable(name: "omp_lock_hint_none", scope: !51, file: !3, type: !9)
!60 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !51, file: !3, type: !9)
!61 = !DILocation(line: 24, column: 1, scope: !51)
!62 = !DILocalVariable(name: "var", scope: !51, file: !3, type: !9)
!63 = !DILocation(line: 20, column: 1, scope: !51)
!64 = !DILocalVariable(name: "i", scope: !51, file: !3, type: !9)
!65 = !DILocation(line: 21, column: 1, scope: !51)
!66 = !DILocation(line: 22, column: 1, scope: !51)

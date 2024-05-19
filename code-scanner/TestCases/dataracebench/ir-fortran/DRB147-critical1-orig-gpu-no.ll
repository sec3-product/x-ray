; ModuleID = '/tmp/DRB147-critical1-orig-gpu-no-bb59fb.ll'
source_filename = "/tmp/DRB147-critical1-orig-gpu-no-bb59fb.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%astruct.dt58 = type <{ i8* }>
%astruct.dt100 = type <{ [8 x i8] }>
%astruct.dt154 = type <{ [8 x i8], i8*, i8* }>

@.C309_MAIN_ = internal constant i32 25
@.C284_MAIN_ = internal constant i64 0
@.C340_MAIN_ = internal constant i32 6
@.C337_MAIN_ = internal constant [57 x i8] c"micro-benchmarks-fortran/DRB147-critical1-orig-gpu-no.f95"
@.C339_MAIN_ = internal constant i32 30
@.C300_MAIN_ = internal constant i32 2
@.C320_MAIN_ = internal constant i32 100
@.C285_MAIN_ = internal constant i32 1
@.C283_MAIN_ = internal constant i32 0
@.C300___nv_MAIN__F1L19_1 = internal constant i32 2
@.C285___nv_MAIN__F1L19_1 = internal constant i32 1
@.C320___nv_MAIN__F1L19_1 = internal constant i32 100
@.C283___nv_MAIN__F1L19_1 = internal constant i32 0
@.C300___nv_MAIN_F1L20_2 = internal constant i32 2
@.C285___nv_MAIN_F1L20_2 = internal constant i32 1
@.C320___nv_MAIN_F1L20_2 = internal constant i32 100
@.C283___nv_MAIN_F1L20_2 = internal constant i32 0
@.C300___nv_MAIN_F1L21_3 = internal constant i32 2
@.C285___nv_MAIN_F1L21_3 = internal constant i32 1
@.C283___nv_MAIN_F1L21_3 = internal constant i32 0

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__384 = alloca i32, align 4
  %var_310 = alloca i32, align 4
  %.uplevelArgPack0001_381 = alloca %astruct.dt58, align 8
  %z__io_342 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 2, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 2, metadata !14, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !15, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !16, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 2, metadata !17, metadata !DIExpression()), !dbg !10
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !18
  store i32 %0, i32* %__gtid_MAIN__384, align 4, !dbg !18
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !19
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !19
  call void (i8*, ...) %2(i8* %1), !dbg !19
  br label %L.LB1_375

L.LB1_375:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %var_310, metadata !20, metadata !DIExpression()), !dbg !10
  store i32 0, i32* %var_310, align 4, !dbg !21
  %3 = bitcast i32* %var_310 to i8*, !dbg !22
  %4 = bitcast %astruct.dt58* %.uplevelArgPack0001_381 to i8**, !dbg !22
  store i8* %3, i8** %4, align 8, !dbg !22
  %5 = bitcast %astruct.dt58* %.uplevelArgPack0001_381 to i64*, !dbg !22
  call void @__nv_MAIN__F1L19_1_(i32* %__gtid_MAIN__384, i64* null, i64* %5), !dbg !22
  call void (...) @_mp_bcs_nest(), !dbg !23
  %6 = bitcast i32* @.C339_MAIN_ to i8*, !dbg !23
  %7 = bitcast [57 x i8]* @.C337_MAIN_ to i8*, !dbg !23
  %8 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !23
  call void (i8*, i8*, i64, ...) %8(i8* %6, i8* %7, i64 57), !dbg !23
  %9 = bitcast i32* @.C340_MAIN_ to i8*, !dbg !23
  %10 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !23
  %11 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !23
  %12 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !23
  %13 = call i32 (i8*, i8*, i8*, i8*, ...) %12(i8* %9, i8* null, i8* %10, i8* %11), !dbg !23
  call void @llvm.dbg.declare(metadata i32* %z__io_342, metadata !24, metadata !DIExpression()), !dbg !10
  store i32 %13, i32* %z__io_342, align 4, !dbg !23
  %14 = load i32, i32* %var_310, align 4, !dbg !23
  call void @llvm.dbg.value(metadata i32 %14, metadata !20, metadata !DIExpression()), !dbg !10
  %15 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !23
  %16 = call i32 (i32, i32, ...) %15(i32 %14, i32 25), !dbg !23
  store i32 %16, i32* %z__io_342, align 4, !dbg !23
  %17 = call i32 (...) @f90io_ldw_end(), !dbg !23
  store i32 %17, i32* %z__io_342, align 4, !dbg !23
  call void (...) @_mp_ecs_nest(), !dbg !23
  ret void, !dbg !18
}

define internal void @__nv_MAIN__F1L19_1_(i32* %__nv_MAIN__F1L19_1Arg0, i64* %__nv_MAIN__F1L19_1Arg1, i64* %__nv_MAIN__F1L19_1Arg2) #0 !dbg !25 {
L.entry:
  %.uplevelArgPack0002_404 = alloca %astruct.dt100, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L19_1Arg0, metadata !29, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L19_1Arg1, metadata !31, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L19_1Arg2, metadata !32, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.value(metadata i32 1, metadata !33, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.value(metadata i32 2, metadata !34, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.value(metadata i32 0, metadata !35, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.value(metadata i32 1, metadata !36, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.value(metadata i32 2, metadata !37, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.value(metadata i32 0, metadata !38, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.value(metadata i32 1, metadata !39, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.value(metadata i32 2, metadata !40, metadata !DIExpression()), !dbg !30
  br label %L.LB2_399

L.LB2_399:                                        ; preds = %L.entry
  br label %L.LB2_314

L.LB2_314:                                        ; preds = %L.LB2_399
  %0 = load i64, i64* %__nv_MAIN__F1L19_1Arg2, align 8, !dbg !41
  %1 = bitcast %astruct.dt100* %.uplevelArgPack0002_404 to i64*, !dbg !41
  store i64 %0, i64* %1, align 8, !dbg !41
  %2 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN_F1L20_2_ to i64*, !dbg !41
  %3 = bitcast %astruct.dt100* %.uplevelArgPack0002_404 to i64*, !dbg !41
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_teams(i64* null, i32 1, i64* %2, i64* %3), !dbg !41
  br label %L.LB2_335

L.LB2_335:                                        ; preds = %L.LB2_314
  ret void, !dbg !42
}

define internal void @__nv_MAIN_F1L20_2_(i32* %__nv_MAIN_F1L20_2Arg0, i64* %__nv_MAIN_F1L20_2Arg1, i64* %__nv_MAIN_F1L20_2Arg2) #0 !dbg !43 {
L.entry:
  %__gtid___nv_MAIN_F1L20_2__439 = alloca i32, align 4
  %.i0000p_322 = alloca i32, align 4
  %.i0001p_323 = alloca i32, align 4
  %.i0002p_324 = alloca i32, align 4
  %.i0003p_325 = alloca i32, align 4
  %i_321 = alloca i32, align 4
  %.du0001_353 = alloca i32, align 4
  %.de0001_354 = alloca i32, align 4
  %.di0001_355 = alloca i32, align 4
  %.ds0001_356 = alloca i32, align 4
  %.dl0001_358 = alloca i32, align 4
  %.dl0001.copy_433 = alloca i32, align 4
  %.de0001.copy_434 = alloca i32, align 4
  %.ds0001.copy_435 = alloca i32, align 4
  %.dX0001_357 = alloca i32, align 4
  %.dY0001_352 = alloca i32, align 4
  %.uplevelArgPack0003_458 = alloca %astruct.dt154, align 16
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L20_2Arg0, metadata !44, metadata !DIExpression()), !dbg !45
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L20_2Arg1, metadata !46, metadata !DIExpression()), !dbg !45
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L20_2Arg2, metadata !47, metadata !DIExpression()), !dbg !45
  call void @llvm.dbg.value(metadata i32 1, metadata !48, metadata !DIExpression()), !dbg !45
  call void @llvm.dbg.value(metadata i32 2, metadata !49, metadata !DIExpression()), !dbg !45
  call void @llvm.dbg.value(metadata i32 0, metadata !50, metadata !DIExpression()), !dbg !45
  call void @llvm.dbg.value(metadata i32 1, metadata !51, metadata !DIExpression()), !dbg !45
  call void @llvm.dbg.value(metadata i32 2, metadata !52, metadata !DIExpression()), !dbg !45
  call void @llvm.dbg.value(metadata i32 0, metadata !53, metadata !DIExpression()), !dbg !45
  call void @llvm.dbg.value(metadata i32 1, metadata !54, metadata !DIExpression()), !dbg !45
  call void @llvm.dbg.value(metadata i32 2, metadata !55, metadata !DIExpression()), !dbg !45
  %0 = load i32, i32* %__nv_MAIN_F1L20_2Arg0, align 4, !dbg !56
  store i32 %0, i32* %__gtid___nv_MAIN_F1L20_2__439, align 4, !dbg !56
  br label %L.LB4_422

L.LB4_422:                                        ; preds = %L.entry
  br label %L.LB4_317

L.LB4_317:                                        ; preds = %L.LB4_422
  br label %L.LB4_318

L.LB4_318:                                        ; preds = %L.LB4_317
  br label %L.LB4_319

L.LB4_319:                                        ; preds = %L.LB4_318
  store i32 0, i32* %.i0000p_322, align 4, !dbg !57
  store i32 0, i32* %.i0001p_323, align 4, !dbg !57
  store i32 100, i32* %.i0002p_324, align 4, !dbg !57
  store i32 1, i32* %.i0003p_325, align 4, !dbg !57
  %1 = load i32, i32* %.i0001p_323, align 4, !dbg !57
  call void @llvm.dbg.declare(metadata i32* %i_321, metadata !58, metadata !DIExpression()), !dbg !56
  store i32 %1, i32* %i_321, align 4, !dbg !57
  %2 = load i32, i32* %.i0002p_324, align 4, !dbg !57
  store i32 %2, i32* %.du0001_353, align 4, !dbg !57
  %3 = load i32, i32* %.i0002p_324, align 4, !dbg !57
  store i32 %3, i32* %.de0001_354, align 4, !dbg !57
  store i32 1, i32* %.di0001_355, align 4, !dbg !57
  %4 = load i32, i32* %.di0001_355, align 4, !dbg !57
  store i32 %4, i32* %.ds0001_356, align 4, !dbg !57
  %5 = load i32, i32* %.i0001p_323, align 4, !dbg !57
  store i32 %5, i32* %.dl0001_358, align 4, !dbg !57
  %6 = load i32, i32* %.dl0001_358, align 4, !dbg !57
  store i32 %6, i32* %.dl0001.copy_433, align 4, !dbg !57
  %7 = load i32, i32* %.de0001_354, align 4, !dbg !57
  store i32 %7, i32* %.de0001.copy_434, align 4, !dbg !57
  %8 = load i32, i32* %.ds0001_356, align 4, !dbg !57
  store i32 %8, i32* %.ds0001.copy_435, align 4, !dbg !57
  %9 = load i32, i32* %__gtid___nv_MAIN_F1L20_2__439, align 4, !dbg !57
  %10 = bitcast i32* %.i0000p_322 to i64*, !dbg !57
  %11 = bitcast i32* %.dl0001.copy_433 to i64*, !dbg !57
  %12 = bitcast i32* %.de0001.copy_434 to i64*, !dbg !57
  %13 = bitcast i32* %.ds0001.copy_435 to i64*, !dbg !57
  %14 = load i32, i32* %.ds0001.copy_435, align 4, !dbg !57
  call void @__kmpc_for_static_init_4(i64* null, i32 %9, i32 92, i64* %10, i64* %11, i64* %12, i64* %13, i32 %14, i32 1), !dbg !57
  %15 = load i32, i32* %.dl0001.copy_433, align 4, !dbg !57
  store i32 %15, i32* %.dl0001_358, align 4, !dbg !57
  %16 = load i32, i32* %.de0001.copy_434, align 4, !dbg !57
  store i32 %16, i32* %.de0001_354, align 4, !dbg !57
  %17 = load i32, i32* %.ds0001.copy_435, align 4, !dbg !57
  store i32 %17, i32* %.ds0001_356, align 4, !dbg !57
  %18 = load i32, i32* %.dl0001_358, align 4, !dbg !57
  store i32 %18, i32* %i_321, align 4, !dbg !57
  %19 = load i32, i32* %i_321, align 4, !dbg !57
  call void @llvm.dbg.value(metadata i32 %19, metadata !58, metadata !DIExpression()), !dbg !56
  store i32 %19, i32* %.dX0001_357, align 4, !dbg !57
  %20 = load i32, i32* %.dX0001_357, align 4, !dbg !57
  %21 = load i32, i32* %.du0001_353, align 4, !dbg !57
  %22 = icmp sgt i32 %20, %21, !dbg !57
  br i1 %22, label %L.LB4_351, label %L.LB4_489, !dbg !57

L.LB4_489:                                        ; preds = %L.LB4_319
  %23 = load i32, i32* %.du0001_353, align 4, !dbg !57
  %24 = load i32, i32* %.de0001_354, align 4, !dbg !57
  %25 = icmp slt i32 %23, %24, !dbg !57
  %26 = select i1 %25, i32 %23, i32 %24, !dbg !57
  store i32 %26, i32* %.de0001_354, align 4, !dbg !57
  %27 = load i32, i32* %.dX0001_357, align 4, !dbg !57
  store i32 %27, i32* %i_321, align 4, !dbg !57
  %28 = load i32, i32* %.di0001_355, align 4, !dbg !57
  %29 = load i32, i32* %.de0001_354, align 4, !dbg !57
  %30 = load i32, i32* %.dX0001_357, align 4, !dbg !57
  %31 = sub nsw i32 %29, %30, !dbg !57
  %32 = add nsw i32 %28, %31, !dbg !57
  %33 = load i32, i32* %.di0001_355, align 4, !dbg !57
  %34 = sdiv i32 %32, %33, !dbg !57
  store i32 %34, i32* %.dY0001_352, align 4, !dbg !57
  %35 = load i32, i32* %i_321, align 4, !dbg !57
  call void @llvm.dbg.value(metadata i32 %35, metadata !58, metadata !DIExpression()), !dbg !56
  store i32 %35, i32* %.i0001p_323, align 4, !dbg !57
  %36 = load i32, i32* %.de0001_354, align 4, !dbg !57
  store i32 %36, i32* %.i0002p_324, align 4, !dbg !57
  %37 = load i64, i64* %__nv_MAIN_F1L20_2Arg2, align 8, !dbg !57
  %38 = bitcast %astruct.dt154* %.uplevelArgPack0003_458 to i64*, !dbg !57
  store i64 %37, i64* %38, align 8, !dbg !57
  %39 = bitcast i32* %.i0001p_323 to i8*, !dbg !57
  %40 = bitcast %astruct.dt154* %.uplevelArgPack0003_458 to i8*, !dbg !57
  %41 = getelementptr i8, i8* %40, i64 8, !dbg !57
  %42 = bitcast i8* %41 to i8**, !dbg !57
  store i8* %39, i8** %42, align 8, !dbg !57
  %43 = bitcast i32* %.i0002p_324 to i8*, !dbg !57
  %44 = bitcast %astruct.dt154* %.uplevelArgPack0003_458 to i8*, !dbg !57
  %45 = getelementptr i8, i8* %44, i64 16, !dbg !57
  %46 = bitcast i8* %45 to i8**, !dbg !57
  store i8* %43, i8** %46, align 8, !dbg !57
  br label %L.LB4_465, !dbg !57

L.LB4_465:                                        ; preds = %L.LB4_489
  %47 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN_F1L21_3_ to i64*, !dbg !57
  %48 = bitcast %astruct.dt154* %.uplevelArgPack0003_458 to i64*, !dbg !57
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %47, i64* %48), !dbg !57
  br label %L.LB4_351

L.LB4_351:                                        ; preds = %L.LB4_465, %L.LB4_319
  %49 = load i32, i32* %__gtid___nv_MAIN_F1L20_2__439, align 4, !dbg !59
  call void @__kmpc_for_static_fini(i64* null, i32 %49), !dbg !59
  br label %L.LB4_332

L.LB4_332:                                        ; preds = %L.LB4_351
  br label %L.LB4_333

L.LB4_333:                                        ; preds = %L.LB4_332
  br label %L.LB4_334

L.LB4_334:                                        ; preds = %L.LB4_333
  ret void, !dbg !56
}

define internal void @__nv_MAIN_F1L21_3_(i32* %__nv_MAIN_F1L21_3Arg0, i64* %__nv_MAIN_F1L21_3Arg1, i64* %__nv_MAIN_F1L21_3Arg2) #0 !dbg !60 {
L.entry:
  %__gtid___nv_MAIN_F1L21_3__510 = alloca i32, align 4
  %.i0004p_330 = alloca i32, align 4
  %i_329 = alloca i32, align 4
  %.du0002p_365 = alloca i32, align 4
  %.de0002p_366 = alloca i32, align 4
  %.di0002p_367 = alloca i32, align 4
  %.ds0002p_368 = alloca i32, align 4
  %.dl0002p_370 = alloca i32, align 4
  %.dl0002p.copy_504 = alloca i32, align 4
  %.de0002p.copy_505 = alloca i32, align 4
  %.ds0002p.copy_506 = alloca i32, align 4
  %.dX0002p_369 = alloca i32, align 4
  %.dY0002p_364 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L21_3Arg0, metadata !61, metadata !DIExpression()), !dbg !62
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L21_3Arg1, metadata !63, metadata !DIExpression()), !dbg !62
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L21_3Arg2, metadata !64, metadata !DIExpression()), !dbg !62
  call void @llvm.dbg.value(metadata i32 1, metadata !65, metadata !DIExpression()), !dbg !62
  call void @llvm.dbg.value(metadata i32 2, metadata !66, metadata !DIExpression()), !dbg !62
  call void @llvm.dbg.value(metadata i32 0, metadata !67, metadata !DIExpression()), !dbg !62
  call void @llvm.dbg.value(metadata i32 1, metadata !68, metadata !DIExpression()), !dbg !62
  call void @llvm.dbg.value(metadata i32 2, metadata !69, metadata !DIExpression()), !dbg !62
  call void @llvm.dbg.value(metadata i32 0, metadata !70, metadata !DIExpression()), !dbg !62
  call void @llvm.dbg.value(metadata i32 1, metadata !71, metadata !DIExpression()), !dbg !62
  call void @llvm.dbg.value(metadata i32 2, metadata !72, metadata !DIExpression()), !dbg !62
  %0 = load i32, i32* %__nv_MAIN_F1L21_3Arg0, align 4, !dbg !73
  store i32 %0, i32* %__gtid___nv_MAIN_F1L21_3__510, align 4, !dbg !73
  br label %L.LB6_493

L.LB6_493:                                        ; preds = %L.entry
  br label %L.LB6_328

L.LB6_328:                                        ; preds = %L.LB6_493
  store i32 0, i32* %.i0004p_330, align 4, !dbg !74
  %1 = bitcast i64* %__nv_MAIN_F1L21_3Arg2 to i8*, !dbg !74
  %2 = getelementptr i8, i8* %1, i64 8, !dbg !74
  %3 = bitcast i8* %2 to i32**, !dbg !74
  %4 = load i32*, i32** %3, align 8, !dbg !74
  %5 = load i32, i32* %4, align 4, !dbg !74
  call void @llvm.dbg.declare(metadata i32* %i_329, metadata !75, metadata !DIExpression()), !dbg !73
  store i32 %5, i32* %i_329, align 4, !dbg !74
  %6 = bitcast i64* %__nv_MAIN_F1L21_3Arg2 to i8*, !dbg !74
  %7 = getelementptr i8, i8* %6, i64 16, !dbg !74
  %8 = bitcast i8* %7 to i32**, !dbg !74
  %9 = load i32*, i32** %8, align 8, !dbg !74
  %10 = load i32, i32* %9, align 4, !dbg !74
  store i32 %10, i32* %.du0002p_365, align 4, !dbg !74
  %11 = bitcast i64* %__nv_MAIN_F1L21_3Arg2 to i8*, !dbg !74
  %12 = getelementptr i8, i8* %11, i64 16, !dbg !74
  %13 = bitcast i8* %12 to i32**, !dbg !74
  %14 = load i32*, i32** %13, align 8, !dbg !74
  %15 = load i32, i32* %14, align 4, !dbg !74
  store i32 %15, i32* %.de0002p_366, align 4, !dbg !74
  store i32 1, i32* %.di0002p_367, align 4, !dbg !74
  %16 = load i32, i32* %.di0002p_367, align 4, !dbg !74
  store i32 %16, i32* %.ds0002p_368, align 4, !dbg !74
  %17 = bitcast i64* %__nv_MAIN_F1L21_3Arg2 to i8*, !dbg !74
  %18 = getelementptr i8, i8* %17, i64 8, !dbg !74
  %19 = bitcast i8* %18 to i32**, !dbg !74
  %20 = load i32*, i32** %19, align 8, !dbg !74
  %21 = load i32, i32* %20, align 4, !dbg !74
  store i32 %21, i32* %.dl0002p_370, align 4, !dbg !74
  %22 = load i32, i32* %.dl0002p_370, align 4, !dbg !74
  store i32 %22, i32* %.dl0002p.copy_504, align 4, !dbg !74
  %23 = load i32, i32* %.de0002p_366, align 4, !dbg !74
  store i32 %23, i32* %.de0002p.copy_505, align 4, !dbg !74
  %24 = load i32, i32* %.ds0002p_368, align 4, !dbg !74
  store i32 %24, i32* %.ds0002p.copy_506, align 4, !dbg !74
  %25 = load i32, i32* %__gtid___nv_MAIN_F1L21_3__510, align 4, !dbg !74
  %26 = bitcast i32* %.i0004p_330 to i64*, !dbg !74
  %27 = bitcast i32* %.dl0002p.copy_504 to i64*, !dbg !74
  %28 = bitcast i32* %.de0002p.copy_505 to i64*, !dbg !74
  %29 = bitcast i32* %.ds0002p.copy_506 to i64*, !dbg !74
  %30 = load i32, i32* %.ds0002p.copy_506, align 4, !dbg !74
  call void @__kmpc_for_static_init_4(i64* null, i32 %25, i32 34, i64* %26, i64* %27, i64* %28, i64* %29, i32 %30, i32 1), !dbg !74
  %31 = load i32, i32* %.dl0002p.copy_504, align 4, !dbg !74
  store i32 %31, i32* %.dl0002p_370, align 4, !dbg !74
  %32 = load i32, i32* %.de0002p.copy_505, align 4, !dbg !74
  store i32 %32, i32* %.de0002p_366, align 4, !dbg !74
  %33 = load i32, i32* %.ds0002p.copy_506, align 4, !dbg !74
  store i32 %33, i32* %.ds0002p_368, align 4, !dbg !74
  %34 = load i32, i32* %.dl0002p_370, align 4, !dbg !74
  store i32 %34, i32* %i_329, align 4, !dbg !74
  %35 = load i32, i32* %i_329, align 4, !dbg !74
  call void @llvm.dbg.value(metadata i32 %35, metadata !75, metadata !DIExpression()), !dbg !73
  store i32 %35, i32* %.dX0002p_369, align 4, !dbg !74
  %36 = load i32, i32* %.dX0002p_369, align 4, !dbg !74
  %37 = load i32, i32* %.du0002p_365, align 4, !dbg !74
  %38 = icmp sgt i32 %36, %37, !dbg !74
  br i1 %38, label %L.LB6_363, label %L.LB6_521, !dbg !74

L.LB6_521:                                        ; preds = %L.LB6_328
  %39 = load i32, i32* %.dX0002p_369, align 4, !dbg !74
  store i32 %39, i32* %i_329, align 4, !dbg !74
  %40 = load i32, i32* %.di0002p_367, align 4, !dbg !74
  %41 = load i32, i32* %.de0002p_366, align 4, !dbg !74
  %42 = load i32, i32* %.dX0002p_369, align 4, !dbg !74
  %43 = sub nsw i32 %41, %42, !dbg !74
  %44 = add nsw i32 %40, %43, !dbg !74
  %45 = load i32, i32* %.di0002p_367, align 4, !dbg !74
  %46 = sdiv i32 %44, %45, !dbg !74
  store i32 %46, i32* %.dY0002p_364, align 4, !dbg !74
  %47 = load i32, i32* %.dY0002p_364, align 4, !dbg !74
  %48 = icmp sle i32 %47, 0, !dbg !74
  br i1 %48, label %L.LB6_373, label %L.LB6_372, !dbg !74

L.LB6_372:                                        ; preds = %L.LB6_372, %L.LB6_521
  %49 = call i32 (...) @_mp_bcs_nest_red(), !dbg !76
  %50 = bitcast i64* %__nv_MAIN_F1L21_3Arg2 to i32**, !dbg !76
  %51 = load i32*, i32** %50, align 8, !dbg !76
  %52 = load i32, i32* %51, align 4, !dbg !76
  %53 = add nsw i32 %52, 1, !dbg !76
  %54 = bitcast i64* %__nv_MAIN_F1L21_3Arg2 to i32**, !dbg !76
  %55 = load i32*, i32** %54, align 8, !dbg !76
  store i32 %53, i32* %55, align 4, !dbg !76
  %56 = call i32 (...) @_mp_ecs_nest_red(), !dbg !76
  %57 = call i32 (...) @_mp_bcs_nest_red(), !dbg !77
  %58 = bitcast i64* %__nv_MAIN_F1L21_3Arg2 to i32**, !dbg !77
  %59 = load i32*, i32** %58, align 8, !dbg !77
  %60 = load i32, i32* %59, align 4, !dbg !77
  %61 = sub nsw i32 %60, 2, !dbg !77
  %62 = bitcast i64* %__nv_MAIN_F1L21_3Arg2 to i32**, !dbg !77
  %63 = load i32*, i32** %62, align 8, !dbg !77
  store i32 %61, i32* %63, align 4, !dbg !77
  %64 = call i32 (...) @_mp_ecs_nest_red(), !dbg !77
  %65 = load i32, i32* %.di0002p_367, align 4, !dbg !73
  %66 = load i32, i32* %i_329, align 4, !dbg !73
  call void @llvm.dbg.value(metadata i32 %66, metadata !75, metadata !DIExpression()), !dbg !73
  %67 = add nsw i32 %65, %66, !dbg !73
  store i32 %67, i32* %i_329, align 4, !dbg !73
  %68 = load i32, i32* %.dY0002p_364, align 4, !dbg !73
  %69 = sub nsw i32 %68, 1, !dbg !73
  store i32 %69, i32* %.dY0002p_364, align 4, !dbg !73
  %70 = load i32, i32* %.dY0002p_364, align 4, !dbg !73
  %71 = icmp sgt i32 %70, 0, !dbg !73
  br i1 %71, label %L.LB6_372, label %L.LB6_373, !dbg !73

L.LB6_373:                                        ; preds = %L.LB6_372, %L.LB6_521
  br label %L.LB6_363

L.LB6_363:                                        ; preds = %L.LB6_373, %L.LB6_328
  %72 = load i32, i32* %__gtid___nv_MAIN_F1L21_3__510, align 4, !dbg !73
  call void @__kmpc_for_static_fini(i64* null, i32 %72), !dbg !73
  br label %L.LB6_331

L.LB6_331:                                        ; preds = %L.LB6_363
  ret void, !dbg !73
}

declare signext i32 @_mp_ecs_nest_red(...) #0

declare signext i32 @_mp_bcs_nest_red(...) #0

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
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB147-critical1-orig-gpu-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb147_critical1_orig_gpu_no", scope: !2, file: !3, line: 12, type: !6, scopeLine: 12, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!6 = !DISubroutineType(cc: DW_CC_program, types: !7)
!7 = !{null}
!8 = !DILocalVariable(name: "omp_sched_static", scope: !5, file: !3, type: !9)
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 0, scope: !5)
!11 = !DILocalVariable(name: "omp_sched_dynamic", scope: !5, file: !3, type: !9)
!12 = !DILocalVariable(name: "omp_proc_bind_false", scope: !5, file: !3, type: !9)
!13 = !DILocalVariable(name: "omp_proc_bind_true", scope: !5, file: !3, type: !9)
!14 = !DILocalVariable(name: "omp_proc_bind_master", scope: !5, file: !3, type: !9)
!15 = !DILocalVariable(name: "omp_lock_hint_none", scope: !5, file: !3, type: !9)
!16 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !5, file: !3, type: !9)
!17 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !5, file: !3, type: !9)
!18 = !DILocation(line: 31, column: 1, scope: !5)
!19 = !DILocation(line: 12, column: 1, scope: !5)
!20 = !DILocalVariable(name: "var", scope: !5, file: !3, type: !9)
!21 = !DILocation(line: 17, column: 1, scope: !5)
!22 = !DILocation(line: 28, column: 1, scope: !5)
!23 = !DILocation(line: 30, column: 1, scope: !5)
!24 = !DILocalVariable(scope: !5, file: !3, type: !9, flags: DIFlagArtificial)
!25 = distinct !DISubprogram(name: "__nv_MAIN__F1L19_1", scope: !2, file: !3, line: 19, type: !26, scopeLine: 19, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!26 = !DISubroutineType(types: !27)
!27 = !{null, !9, !28, !28}
!28 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!29 = !DILocalVariable(name: "__nv_MAIN__F1L19_1Arg0", arg: 1, scope: !25, file: !3, type: !9)
!30 = !DILocation(line: 0, scope: !25)
!31 = !DILocalVariable(name: "__nv_MAIN__F1L19_1Arg1", arg: 2, scope: !25, file: !3, type: !28)
!32 = !DILocalVariable(name: "__nv_MAIN__F1L19_1Arg2", arg: 3, scope: !25, file: !3, type: !28)
!33 = !DILocalVariable(name: "omp_sched_static", scope: !25, file: !3, type: !9)
!34 = !DILocalVariable(name: "omp_sched_dynamic", scope: !25, file: !3, type: !9)
!35 = !DILocalVariable(name: "omp_proc_bind_false", scope: !25, file: !3, type: !9)
!36 = !DILocalVariable(name: "omp_proc_bind_true", scope: !25, file: !3, type: !9)
!37 = !DILocalVariable(name: "omp_proc_bind_master", scope: !25, file: !3, type: !9)
!38 = !DILocalVariable(name: "omp_lock_hint_none", scope: !25, file: !3, type: !9)
!39 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !25, file: !3, type: !9)
!40 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !25, file: !3, type: !9)
!41 = !DILocation(line: 20, column: 1, scope: !25)
!42 = !DILocation(line: 28, column: 1, scope: !25)
!43 = distinct !DISubprogram(name: "__nv_MAIN_F1L20_2", scope: !2, file: !3, line: 20, type: !26, scopeLine: 20, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!44 = !DILocalVariable(name: "__nv_MAIN_F1L20_2Arg0", arg: 1, scope: !43, file: !3, type: !9)
!45 = !DILocation(line: 0, scope: !43)
!46 = !DILocalVariable(name: "__nv_MAIN_F1L20_2Arg1", arg: 2, scope: !43, file: !3, type: !28)
!47 = !DILocalVariable(name: "__nv_MAIN_F1L20_2Arg2", arg: 3, scope: !43, file: !3, type: !28)
!48 = !DILocalVariable(name: "omp_sched_static", scope: !43, file: !3, type: !9)
!49 = !DILocalVariable(name: "omp_sched_dynamic", scope: !43, file: !3, type: !9)
!50 = !DILocalVariable(name: "omp_proc_bind_false", scope: !43, file: !3, type: !9)
!51 = !DILocalVariable(name: "omp_proc_bind_true", scope: !43, file: !3, type: !9)
!52 = !DILocalVariable(name: "omp_proc_bind_master", scope: !43, file: !3, type: !9)
!53 = !DILocalVariable(name: "omp_lock_hint_none", scope: !43, file: !3, type: !9)
!54 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !43, file: !3, type: !9)
!55 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !43, file: !3, type: !9)
!56 = !DILocation(line: 27, column: 1, scope: !43)
!57 = !DILocation(line: 21, column: 1, scope: !43)
!58 = !DILocalVariable(name: "i", scope: !43, file: !3, type: !9)
!59 = !DILocation(line: 26, column: 1, scope: !43)
!60 = distinct !DISubprogram(name: "__nv_MAIN_F1L21_3", scope: !2, file: !3, line: 21, type: !26, scopeLine: 21, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!61 = !DILocalVariable(name: "__nv_MAIN_F1L21_3Arg0", arg: 1, scope: !60, file: !3, type: !9)
!62 = !DILocation(line: 0, scope: !60)
!63 = !DILocalVariable(name: "__nv_MAIN_F1L21_3Arg1", arg: 2, scope: !60, file: !3, type: !28)
!64 = !DILocalVariable(name: "__nv_MAIN_F1L21_3Arg2", arg: 3, scope: !60, file: !3, type: !28)
!65 = !DILocalVariable(name: "omp_sched_static", scope: !60, file: !3, type: !9)
!66 = !DILocalVariable(name: "omp_sched_dynamic", scope: !60, file: !3, type: !9)
!67 = !DILocalVariable(name: "omp_proc_bind_false", scope: !60, file: !3, type: !9)
!68 = !DILocalVariable(name: "omp_proc_bind_true", scope: !60, file: !3, type: !9)
!69 = !DILocalVariable(name: "omp_proc_bind_master", scope: !60, file: !3, type: !9)
!70 = !DILocalVariable(name: "omp_lock_hint_none", scope: !60, file: !3, type: !9)
!71 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !60, file: !3, type: !9)
!72 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !60, file: !3, type: !9)
!73 = !DILocation(line: 26, column: 1, scope: !60)
!74 = !DILocation(line: 21, column: 1, scope: !60)
!75 = !DILocalVariable(name: "i", scope: !60, file: !3, type: !9)
!76 = !DILocation(line: 23, column: 1, scope: !60)
!77 = !DILocation(line: 25, column: 1, scope: !60)

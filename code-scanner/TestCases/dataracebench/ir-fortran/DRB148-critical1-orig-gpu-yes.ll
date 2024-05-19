; ModuleID = '/tmp/DRB148-critical1-orig-gpu-yes-23da17.ll'
source_filename = "/tmp/DRB148-critical1-orig-gpu-yes-23da17.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct__cs_addlock_ = type <{ [32 x i8] }>
%astruct.dt61 = type <{ i8* }>
%astruct.dt115 = type <{ [8 x i8] }>
%astruct.dt169 = type <{ [8 x i8], i8*, i8* }>

@.C305_MAIN_ = internal constant i32 25
@.C284_MAIN_ = internal constant i64 0
@.C340_MAIN_ = internal constant i32 6
@.C337_MAIN_ = internal constant [58 x i8] c"micro-benchmarks-fortran/DRB148-critical1-orig-gpu-yes.f95"
@.C339_MAIN_ = internal constant i32 30
@.C316_MAIN_ = internal constant i32 100
@.C285_MAIN_ = internal constant i32 1
@.C283_MAIN_ = internal constant i32 0
@.C316___nv_MAIN__F1L20_1 = internal constant i32 100
@.C285___nv_MAIN__F1L20_1 = internal constant i32 1
@.C283___nv_MAIN__F1L20_1 = internal constant i32 0
@.C316___nv_MAIN_F1L21_2 = internal constant i32 100
@.C285___nv_MAIN_F1L21_2 = internal constant i32 1
@.C283___nv_MAIN_F1L21_2 = internal constant i32 0
@.C285___nv_MAIN_F1L22_3 = internal constant i32 1
@.C283___nv_MAIN_F1L22_3 = internal constant i32 0
@__cs_addlock_ = common global %struct__cs_addlock_ zeroinitializer, align 64

define void @MAIN_() #0 !dbg !18 {
L.entry:
  %__gtid_MAIN__384 = alloca i32, align 4
  %var_306 = alloca i32, align 4
  %.uplevelArgPack0001_381 = alloca %astruct.dt61, align 8
  %z__io_342 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !21, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i32 0, metadata !23, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i32 1, metadata !24, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i32 0, metadata !25, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i32 1, metadata !26, metadata !DIExpression()), !dbg !22
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !27
  store i32 %0, i32* %__gtid_MAIN__384, align 4, !dbg !27
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !28
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !28
  call void (i8*, ...) %2(i8* %1), !dbg !28
  br label %L.LB1_375

L.LB1_375:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %var_306, metadata !29, metadata !DIExpression()), !dbg !22
  store i32 0, i32* %var_306, align 4, !dbg !30
  %3 = bitcast i32* %var_306 to i8*, !dbg !31
  %4 = bitcast %astruct.dt61* %.uplevelArgPack0001_381 to i8**, !dbg !31
  store i8* %3, i8** %4, align 8, !dbg !31
  %5 = bitcast %astruct.dt61* %.uplevelArgPack0001_381 to i64*, !dbg !31
  call void @__nv_MAIN__F1L20_1_(i32* %__gtid_MAIN__384, i64* null, i64* %5), !dbg !31
  call void (...) @_mp_bcs_nest(), !dbg !32
  %6 = bitcast i32* @.C339_MAIN_ to i8*, !dbg !32
  %7 = bitcast [58 x i8]* @.C337_MAIN_ to i8*, !dbg !32
  %8 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !32
  call void (i8*, i8*, i64, ...) %8(i8* %6, i8* %7, i64 58), !dbg !32
  %9 = bitcast i32* @.C340_MAIN_ to i8*, !dbg !32
  %10 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !32
  %11 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !32
  %12 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !32
  %13 = call i32 (i8*, i8*, i8*, i8*, ...) %12(i8* %9, i8* null, i8* %10, i8* %11), !dbg !32
  call void @llvm.dbg.declare(metadata i32* %z__io_342, metadata !33, metadata !DIExpression()), !dbg !22
  store i32 %13, i32* %z__io_342, align 4, !dbg !32
  %14 = load i32, i32* %var_306, align 4, !dbg !32
  call void @llvm.dbg.value(metadata i32 %14, metadata !29, metadata !DIExpression()), !dbg !22
  %15 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !32
  %16 = call i32 (i32, i32, ...) %15(i32 %14, i32 25), !dbg !32
  store i32 %16, i32* %z__io_342, align 4, !dbg !32
  %17 = call i32 (...) @f90io_ldw_end(), !dbg !32
  store i32 %17, i32* %z__io_342, align 4, !dbg !32
  call void (...) @_mp_ecs_nest(), !dbg !32
  ret void, !dbg !27
}

define internal void @__nv_MAIN__F1L20_1_(i32* %__nv_MAIN__F1L20_1Arg0, i64* %__nv_MAIN__F1L20_1Arg1, i64* %__nv_MAIN__F1L20_1Arg2) #0 !dbg !34 {
L.entry:
  %.uplevelArgPack0002_405 = alloca %astruct.dt115, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L20_1Arg0, metadata !35, metadata !DIExpression()), !dbg !36
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L20_1Arg1, metadata !37, metadata !DIExpression()), !dbg !36
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L20_1Arg2, metadata !38, metadata !DIExpression()), !dbg !36
  call void @llvm.dbg.value(metadata i32 1, metadata !39, metadata !DIExpression()), !dbg !36
  call void @llvm.dbg.value(metadata i32 0, metadata !40, metadata !DIExpression()), !dbg !36
  call void @llvm.dbg.value(metadata i32 1, metadata !41, metadata !DIExpression()), !dbg !36
  call void @llvm.dbg.value(metadata i32 0, metadata !42, metadata !DIExpression()), !dbg !36
  call void @llvm.dbg.value(metadata i32 1, metadata !43, metadata !DIExpression()), !dbg !36
  br label %L.LB2_400

L.LB2_400:                                        ; preds = %L.entry
  br label %L.LB2_310

L.LB2_310:                                        ; preds = %L.LB2_400
  %0 = load i64, i64* %__nv_MAIN__F1L20_1Arg2, align 8, !dbg !44
  %1 = bitcast %astruct.dt115* %.uplevelArgPack0002_405 to i64*, !dbg !44
  store i64 %0, i64* %1, align 8, !dbg !44
  %2 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN_F1L21_2_ to i64*, !dbg !44
  %3 = bitcast %astruct.dt115* %.uplevelArgPack0002_405 to i64*, !dbg !44
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_teams(i64* null, i32 1, i64* %2, i64* %3), !dbg !44
  br label %L.LB2_335

L.LB2_335:                                        ; preds = %L.LB2_310
  ret void, !dbg !45
}

define internal void @__nv_MAIN_F1L21_2_(i32* %__nv_MAIN_F1L21_2Arg0, i64* %__nv_MAIN_F1L21_2Arg1, i64* %__nv_MAIN_F1L21_2Arg2) #0 !dbg !46 {
L.entry:
  %__gtid___nv_MAIN_F1L21_2__440 = alloca i32, align 4
  %.i0000p_318 = alloca i32, align 4
  %.i0001p_319 = alloca i32, align 4
  %.i0002p_320 = alloca i32, align 4
  %.i0003p_321 = alloca i32, align 4
  %i_317 = alloca i32, align 4
  %.du0001_353 = alloca i32, align 4
  %.de0001_354 = alloca i32, align 4
  %.di0001_355 = alloca i32, align 4
  %.ds0001_356 = alloca i32, align 4
  %.dl0001_358 = alloca i32, align 4
  %.dl0001.copy_434 = alloca i32, align 4
  %.de0001.copy_435 = alloca i32, align 4
  %.ds0001.copy_436 = alloca i32, align 4
  %.dX0001_357 = alloca i32, align 4
  %.dY0001_352 = alloca i32, align 4
  %.uplevelArgPack0003_459 = alloca %astruct.dt169, align 16
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L21_2Arg0, metadata !47, metadata !DIExpression()), !dbg !48
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L21_2Arg1, metadata !49, metadata !DIExpression()), !dbg !48
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L21_2Arg2, metadata !50, metadata !DIExpression()), !dbg !48
  call void @llvm.dbg.value(metadata i32 1, metadata !51, metadata !DIExpression()), !dbg !48
  call void @llvm.dbg.value(metadata i32 0, metadata !52, metadata !DIExpression()), !dbg !48
  call void @llvm.dbg.value(metadata i32 1, metadata !53, metadata !DIExpression()), !dbg !48
  call void @llvm.dbg.value(metadata i32 0, metadata !54, metadata !DIExpression()), !dbg !48
  call void @llvm.dbg.value(metadata i32 1, metadata !55, metadata !DIExpression()), !dbg !48
  %0 = load i32, i32* %__nv_MAIN_F1L21_2Arg0, align 4, !dbg !56
  store i32 %0, i32* %__gtid___nv_MAIN_F1L21_2__440, align 4, !dbg !56
  br label %L.LB4_423

L.LB4_423:                                        ; preds = %L.entry
  br label %L.LB4_313

L.LB4_313:                                        ; preds = %L.LB4_423
  br label %L.LB4_314

L.LB4_314:                                        ; preds = %L.LB4_313
  br label %L.LB4_315

L.LB4_315:                                        ; preds = %L.LB4_314
  store i32 0, i32* %.i0000p_318, align 4, !dbg !57
  store i32 1, i32* %.i0001p_319, align 4, !dbg !57
  store i32 100, i32* %.i0002p_320, align 4, !dbg !57
  store i32 1, i32* %.i0003p_321, align 4, !dbg !57
  %1 = load i32, i32* %.i0001p_319, align 4, !dbg !57
  call void @llvm.dbg.declare(metadata i32* %i_317, metadata !58, metadata !DIExpression()), !dbg !56
  store i32 %1, i32* %i_317, align 4, !dbg !57
  %2 = load i32, i32* %.i0002p_320, align 4, !dbg !57
  store i32 %2, i32* %.du0001_353, align 4, !dbg !57
  %3 = load i32, i32* %.i0002p_320, align 4, !dbg !57
  store i32 %3, i32* %.de0001_354, align 4, !dbg !57
  store i32 1, i32* %.di0001_355, align 4, !dbg !57
  %4 = load i32, i32* %.di0001_355, align 4, !dbg !57
  store i32 %4, i32* %.ds0001_356, align 4, !dbg !57
  %5 = load i32, i32* %.i0001p_319, align 4, !dbg !57
  store i32 %5, i32* %.dl0001_358, align 4, !dbg !57
  %6 = load i32, i32* %.dl0001_358, align 4, !dbg !57
  store i32 %6, i32* %.dl0001.copy_434, align 4, !dbg !57
  %7 = load i32, i32* %.de0001_354, align 4, !dbg !57
  store i32 %7, i32* %.de0001.copy_435, align 4, !dbg !57
  %8 = load i32, i32* %.ds0001_356, align 4, !dbg !57
  store i32 %8, i32* %.ds0001.copy_436, align 4, !dbg !57
  %9 = load i32, i32* %__gtid___nv_MAIN_F1L21_2__440, align 4, !dbg !57
  %10 = bitcast i32* %.i0000p_318 to i64*, !dbg !57
  %11 = bitcast i32* %.dl0001.copy_434 to i64*, !dbg !57
  %12 = bitcast i32* %.de0001.copy_435 to i64*, !dbg !57
  %13 = bitcast i32* %.ds0001.copy_436 to i64*, !dbg !57
  %14 = load i32, i32* %.ds0001.copy_436, align 4, !dbg !57
  call void @__kmpc_for_static_init_4(i64* null, i32 %9, i32 92, i64* %10, i64* %11, i64* %12, i64* %13, i32 %14, i32 1), !dbg !57
  %15 = load i32, i32* %.dl0001.copy_434, align 4, !dbg !57
  store i32 %15, i32* %.dl0001_358, align 4, !dbg !57
  %16 = load i32, i32* %.de0001.copy_435, align 4, !dbg !57
  store i32 %16, i32* %.de0001_354, align 4, !dbg !57
  %17 = load i32, i32* %.ds0001.copy_436, align 4, !dbg !57
  store i32 %17, i32* %.ds0001_356, align 4, !dbg !57
  %18 = load i32, i32* %.dl0001_358, align 4, !dbg !57
  store i32 %18, i32* %i_317, align 4, !dbg !57
  %19 = load i32, i32* %i_317, align 4, !dbg !57
  call void @llvm.dbg.value(metadata i32 %19, metadata !58, metadata !DIExpression()), !dbg !56
  store i32 %19, i32* %.dX0001_357, align 4, !dbg !57
  %20 = load i32, i32* %.dX0001_357, align 4, !dbg !57
  %21 = load i32, i32* %.du0001_353, align 4, !dbg !57
  %22 = icmp sgt i32 %20, %21, !dbg !57
  br i1 %22, label %L.LB4_351, label %L.LB4_490, !dbg !57

L.LB4_490:                                        ; preds = %L.LB4_315
  %23 = load i32, i32* %.du0001_353, align 4, !dbg !57
  %24 = load i32, i32* %.de0001_354, align 4, !dbg !57
  %25 = icmp slt i32 %23, %24, !dbg !57
  %26 = select i1 %25, i32 %23, i32 %24, !dbg !57
  store i32 %26, i32* %.de0001_354, align 4, !dbg !57
  %27 = load i32, i32* %.dX0001_357, align 4, !dbg !57
  store i32 %27, i32* %i_317, align 4, !dbg !57
  %28 = load i32, i32* %.di0001_355, align 4, !dbg !57
  %29 = load i32, i32* %.de0001_354, align 4, !dbg !57
  %30 = load i32, i32* %.dX0001_357, align 4, !dbg !57
  %31 = sub nsw i32 %29, %30, !dbg !57
  %32 = add nsw i32 %28, %31, !dbg !57
  %33 = load i32, i32* %.di0001_355, align 4, !dbg !57
  %34 = sdiv i32 %32, %33, !dbg !57
  store i32 %34, i32* %.dY0001_352, align 4, !dbg !57
  %35 = load i32, i32* %i_317, align 4, !dbg !57
  call void @llvm.dbg.value(metadata i32 %35, metadata !58, metadata !DIExpression()), !dbg !56
  store i32 %35, i32* %.i0001p_319, align 4, !dbg !57
  %36 = load i32, i32* %.de0001_354, align 4, !dbg !57
  store i32 %36, i32* %.i0002p_320, align 4, !dbg !57
  %37 = load i64, i64* %__nv_MAIN_F1L21_2Arg2, align 8, !dbg !57
  %38 = bitcast %astruct.dt169* %.uplevelArgPack0003_459 to i64*, !dbg !57
  store i64 %37, i64* %38, align 8, !dbg !57
  %39 = bitcast i32* %.i0001p_319 to i8*, !dbg !57
  %40 = bitcast %astruct.dt169* %.uplevelArgPack0003_459 to i8*, !dbg !57
  %41 = getelementptr i8, i8* %40, i64 8, !dbg !57
  %42 = bitcast i8* %41 to i8**, !dbg !57
  store i8* %39, i8** %42, align 8, !dbg !57
  %43 = bitcast i32* %.i0002p_320 to i8*, !dbg !57
  %44 = bitcast %astruct.dt169* %.uplevelArgPack0003_459 to i8*, !dbg !57
  %45 = getelementptr i8, i8* %44, i64 16, !dbg !57
  %46 = bitcast i8* %45 to i8**, !dbg !57
  store i8* %43, i8** %46, align 8, !dbg !57
  br label %L.LB4_466, !dbg !57

L.LB4_466:                                        ; preds = %L.LB4_490
  %47 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN_F1L22_3_ to i64*, !dbg !57
  %48 = bitcast %astruct.dt169* %.uplevelArgPack0003_459 to i64*, !dbg !57
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %47, i64* %48), !dbg !57
  br label %L.LB4_351

L.LB4_351:                                        ; preds = %L.LB4_466, %L.LB4_315
  %49 = load i32, i32* %__gtid___nv_MAIN_F1L21_2__440, align 4, !dbg !59
  call void @__kmpc_for_static_fini(i64* null, i32 %49), !dbg !59
  br label %L.LB4_332

L.LB4_332:                                        ; preds = %L.LB4_351
  br label %L.LB4_333

L.LB4_333:                                        ; preds = %L.LB4_332
  br label %L.LB4_334

L.LB4_334:                                        ; preds = %L.LB4_333
  ret void, !dbg !56
}

define internal void @__nv_MAIN_F1L22_3_(i32* %__nv_MAIN_F1L22_3Arg0, i64* %__nv_MAIN_F1L22_3Arg1, i64* %__nv_MAIN_F1L22_3Arg2) #0 !dbg !9 {
L.entry:
  %__gtid___nv_MAIN_F1L22_3__511 = alloca i32, align 4
  %.i0004p_326 = alloca i32, align 4
  %i_325 = alloca i32, align 4
  %.du0002p_365 = alloca i32, align 4
  %.de0002p_366 = alloca i32, align 4
  %.di0002p_367 = alloca i32, align 4
  %.ds0002p_368 = alloca i32, align 4
  %.dl0002p_370 = alloca i32, align 4
  %.dl0002p.copy_505 = alloca i32, align 4
  %.de0002p.copy_506 = alloca i32, align 4
  %.ds0002p.copy_507 = alloca i32, align 4
  %.dX0002p_369 = alloca i32, align 4
  %.dY0002p_364 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L22_3Arg0, metadata !60, metadata !DIExpression()), !dbg !61
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L22_3Arg1, metadata !62, metadata !DIExpression()), !dbg !61
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L22_3Arg2, metadata !63, metadata !DIExpression()), !dbg !61
  call void @llvm.dbg.value(metadata i32 1, metadata !64, metadata !DIExpression()), !dbg !61
  call void @llvm.dbg.value(metadata i32 0, metadata !65, metadata !DIExpression()), !dbg !61
  call void @llvm.dbg.value(metadata i32 1, metadata !66, metadata !DIExpression()), !dbg !61
  call void @llvm.dbg.value(metadata i32 0, metadata !67, metadata !DIExpression()), !dbg !61
  call void @llvm.dbg.value(metadata i32 1, metadata !68, metadata !DIExpression()), !dbg !61
  %0 = load i32, i32* %__nv_MAIN_F1L22_3Arg0, align 4, !dbg !69
  store i32 %0, i32* %__gtid___nv_MAIN_F1L22_3__511, align 4, !dbg !69
  br label %L.LB6_494

L.LB6_494:                                        ; preds = %L.entry
  br label %L.LB6_324

L.LB6_324:                                        ; preds = %L.LB6_494
  store i32 0, i32* %.i0004p_326, align 4, !dbg !70
  %1 = bitcast i64* %__nv_MAIN_F1L22_3Arg2 to i8*, !dbg !70
  %2 = getelementptr i8, i8* %1, i64 8, !dbg !70
  %3 = bitcast i8* %2 to i32**, !dbg !70
  %4 = load i32*, i32** %3, align 8, !dbg !70
  %5 = load i32, i32* %4, align 4, !dbg !70
  call void @llvm.dbg.declare(metadata i32* %i_325, metadata !71, metadata !DIExpression()), !dbg !69
  store i32 %5, i32* %i_325, align 4, !dbg !70
  %6 = bitcast i64* %__nv_MAIN_F1L22_3Arg2 to i8*, !dbg !70
  %7 = getelementptr i8, i8* %6, i64 16, !dbg !70
  %8 = bitcast i8* %7 to i32**, !dbg !70
  %9 = load i32*, i32** %8, align 8, !dbg !70
  %10 = load i32, i32* %9, align 4, !dbg !70
  store i32 %10, i32* %.du0002p_365, align 4, !dbg !70
  %11 = bitcast i64* %__nv_MAIN_F1L22_3Arg2 to i8*, !dbg !70
  %12 = getelementptr i8, i8* %11, i64 16, !dbg !70
  %13 = bitcast i8* %12 to i32**, !dbg !70
  %14 = load i32*, i32** %13, align 8, !dbg !70
  %15 = load i32, i32* %14, align 4, !dbg !70
  store i32 %15, i32* %.de0002p_366, align 4, !dbg !70
  store i32 1, i32* %.di0002p_367, align 4, !dbg !70
  %16 = load i32, i32* %.di0002p_367, align 4, !dbg !70
  store i32 %16, i32* %.ds0002p_368, align 4, !dbg !70
  %17 = bitcast i64* %__nv_MAIN_F1L22_3Arg2 to i8*, !dbg !70
  %18 = getelementptr i8, i8* %17, i64 8, !dbg !70
  %19 = bitcast i8* %18 to i32**, !dbg !70
  %20 = load i32*, i32** %19, align 8, !dbg !70
  %21 = load i32, i32* %20, align 4, !dbg !70
  store i32 %21, i32* %.dl0002p_370, align 4, !dbg !70
  %22 = load i32, i32* %.dl0002p_370, align 4, !dbg !70
  store i32 %22, i32* %.dl0002p.copy_505, align 4, !dbg !70
  %23 = load i32, i32* %.de0002p_366, align 4, !dbg !70
  store i32 %23, i32* %.de0002p.copy_506, align 4, !dbg !70
  %24 = load i32, i32* %.ds0002p_368, align 4, !dbg !70
  store i32 %24, i32* %.ds0002p.copy_507, align 4, !dbg !70
  %25 = load i32, i32* %__gtid___nv_MAIN_F1L22_3__511, align 4, !dbg !70
  %26 = bitcast i32* %.i0004p_326 to i64*, !dbg !70
  %27 = bitcast i32* %.dl0002p.copy_505 to i64*, !dbg !70
  %28 = bitcast i32* %.de0002p.copy_506 to i64*, !dbg !70
  %29 = bitcast i32* %.ds0002p.copy_507 to i64*, !dbg !70
  %30 = load i32, i32* %.ds0002p.copy_507, align 4, !dbg !70
  call void @__kmpc_for_static_init_4(i64* null, i32 %25, i32 34, i64* %26, i64* %27, i64* %28, i64* %29, i32 %30, i32 1), !dbg !70
  %31 = load i32, i32* %.dl0002p.copy_505, align 4, !dbg !70
  store i32 %31, i32* %.dl0002p_370, align 4, !dbg !70
  %32 = load i32, i32* %.de0002p.copy_506, align 4, !dbg !70
  store i32 %32, i32* %.de0002p_366, align 4, !dbg !70
  %33 = load i32, i32* %.ds0002p.copy_507, align 4, !dbg !70
  store i32 %33, i32* %.ds0002p_368, align 4, !dbg !70
  %34 = load i32, i32* %.dl0002p_370, align 4, !dbg !70
  store i32 %34, i32* %i_325, align 4, !dbg !70
  %35 = load i32, i32* %i_325, align 4, !dbg !70
  call void @llvm.dbg.value(metadata i32 %35, metadata !71, metadata !DIExpression()), !dbg !69
  store i32 %35, i32* %.dX0002p_369, align 4, !dbg !70
  %36 = load i32, i32* %.dX0002p_369, align 4, !dbg !70
  %37 = load i32, i32* %.du0002p_365, align 4, !dbg !70
  %38 = icmp sgt i32 %36, %37, !dbg !70
  br i1 %38, label %L.LB6_363, label %L.LB6_529, !dbg !70

L.LB6_529:                                        ; preds = %L.LB6_324
  %39 = load i32, i32* %.dX0002p_369, align 4, !dbg !70
  store i32 %39, i32* %i_325, align 4, !dbg !70
  %40 = load i32, i32* %.di0002p_367, align 4, !dbg !70
  %41 = load i32, i32* %.de0002p_366, align 4, !dbg !70
  %42 = load i32, i32* %.dX0002p_369, align 4, !dbg !70
  %43 = sub nsw i32 %41, %42, !dbg !70
  %44 = add nsw i32 %40, %43, !dbg !70
  %45 = load i32, i32* %.di0002p_367, align 4, !dbg !70
  %46 = sdiv i32 %44, %45, !dbg !70
  store i32 %46, i32* %.dY0002p_364, align 4, !dbg !70
  %47 = load i32, i32* %.dY0002p_364, align 4, !dbg !70
  %48 = icmp sle i32 %47, 0, !dbg !70
  br i1 %48, label %L.LB6_373, label %L.LB6_372, !dbg !70

L.LB6_372:                                        ; preds = %L.LB6_372, %L.LB6_529
  %49 = load i32, i32* %__gtid___nv_MAIN_F1L22_3__511, align 4, !dbg !72
  %50 = bitcast %struct__cs_addlock_* @__cs_addlock_ to i64*, !dbg !72
  call void @__kmpc_critical(i64* null, i32 %49, i64* %50), !dbg !72
  %51 = bitcast i64* %__nv_MAIN_F1L22_3Arg2 to i32**, !dbg !72
  %52 = load i32*, i32** %51, align 8, !dbg !72
  %53 = load i32, i32* %52, align 4, !dbg !72
  %54 = add nsw i32 %53, 1, !dbg !72
  %55 = bitcast i64* %__nv_MAIN_F1L22_3Arg2 to i32**, !dbg !72
  %56 = load i32*, i32** %55, align 8, !dbg !72
  store i32 %54, i32* %56, align 4, !dbg !72
  %57 = load i32, i32* %__gtid___nv_MAIN_F1L22_3__511, align 4, !dbg !72
  %58 = bitcast %struct__cs_addlock_* @__cs_addlock_ to i64*, !dbg !72
  call void @__kmpc_end_critical(i64* null, i32 %57, i64* %58), !dbg !72
  %59 = load i32, i32* %.di0002p_367, align 4, !dbg !69
  %60 = load i32, i32* %i_325, align 4, !dbg !69
  call void @llvm.dbg.value(metadata i32 %60, metadata !71, metadata !DIExpression()), !dbg !69
  %61 = add nsw i32 %59, %60, !dbg !69
  store i32 %61, i32* %i_325, align 4, !dbg !69
  %62 = load i32, i32* %.dY0002p_364, align 4, !dbg !69
  %63 = sub nsw i32 %62, 1, !dbg !69
  store i32 %63, i32* %.dY0002p_364, align 4, !dbg !69
  %64 = load i32, i32* %.dY0002p_364, align 4, !dbg !69
  %65 = icmp sgt i32 %64, 0, !dbg !69
  br i1 %65, label %L.LB6_372, label %L.LB6_373, !dbg !69

L.LB6_373:                                        ; preds = %L.LB6_372, %L.LB6_529
  br label %L.LB6_363

L.LB6_363:                                        ; preds = %L.LB6_373, %L.LB6_324
  %66 = load i32, i32* %__gtid___nv_MAIN_F1L22_3__511, align 4, !dbg !69
  call void @__kmpc_for_static_fini(i64* null, i32 %66), !dbg !69
  br label %L.LB6_331

L.LB6_331:                                        ; preds = %L.LB6_363
  ret void, !dbg !69
}

declare void @__kmpc_end_critical(i64*, i32, i64*) #0

declare void @__kmpc_critical(i64*, i32, i64*) #0

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
!2 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !4, globals: !5, imports: !4)
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB148-critical1-orig-gpu-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = !{!6}
!6 = !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!7 = distinct !DIGlobalVariable(name: "__cs_addlock", scope: !8, type: !14, isLocal: false, isDefinition: true)
!8 = distinct !DICommonBlock(scope: !9, declaration: !7, name: "__cs_addlock")
!9 = distinct !DISubprogram(name: "__nv_MAIN_F1L22_3", scope: !2, file: !3, line: 22, type: !10, scopeLine: 22, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!10 = !DISubroutineType(types: !11)
!11 = !{null, !12, !13, !13}
!12 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!13 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!14 = !DICompositeType(tag: DW_TAG_array_type, baseType: !15, size: 256, align: 8, elements: !16)
!15 = !DIBasicType(name: "byte", size: 8, align: 8, encoding: DW_ATE_signed)
!16 = !{!17}
!17 = !DISubrange(count: 32)
!18 = distinct !DISubprogram(name: "drb148_critical1_orig_gpu_yes", scope: !2, file: !3, line: 13, type: !19, scopeLine: 13, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!19 = !DISubroutineType(cc: DW_CC_program, types: !20)
!20 = !{null}
!21 = !DILocalVariable(name: "omp_sched_static", scope: !18, file: !3, type: !12)
!22 = !DILocation(line: 0, scope: !18)
!23 = !DILocalVariable(name: "omp_proc_bind_false", scope: !18, file: !3, type: !12)
!24 = !DILocalVariable(name: "omp_proc_bind_true", scope: !18, file: !3, type: !12)
!25 = !DILocalVariable(name: "omp_lock_hint_none", scope: !18, file: !3, type: !12)
!26 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !18, file: !3, type: !12)
!27 = !DILocation(line: 31, column: 1, scope: !18)
!28 = !DILocation(line: 13, column: 1, scope: !18)
!29 = !DILocalVariable(name: "var", scope: !18, file: !3, type: !12)
!30 = !DILocation(line: 18, column: 1, scope: !18)
!31 = !DILocation(line: 28, column: 1, scope: !18)
!32 = !DILocation(line: 30, column: 1, scope: !18)
!33 = !DILocalVariable(scope: !18, file: !3, type: !12, flags: DIFlagArtificial)
!34 = distinct !DISubprogram(name: "__nv_MAIN__F1L20_1", scope: !2, file: !3, line: 20, type: !10, scopeLine: 20, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!35 = !DILocalVariable(name: "__nv_MAIN__F1L20_1Arg0", arg: 1, scope: !34, file: !3, type: !12)
!36 = !DILocation(line: 0, scope: !34)
!37 = !DILocalVariable(name: "__nv_MAIN__F1L20_1Arg1", arg: 2, scope: !34, file: !3, type: !13)
!38 = !DILocalVariable(name: "__nv_MAIN__F1L20_1Arg2", arg: 3, scope: !34, file: !3, type: !13)
!39 = !DILocalVariable(name: "omp_sched_static", scope: !34, file: !3, type: !12)
!40 = !DILocalVariable(name: "omp_proc_bind_false", scope: !34, file: !3, type: !12)
!41 = !DILocalVariable(name: "omp_proc_bind_true", scope: !34, file: !3, type: !12)
!42 = !DILocalVariable(name: "omp_lock_hint_none", scope: !34, file: !3, type: !12)
!43 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !34, file: !3, type: !12)
!44 = !DILocation(line: 21, column: 1, scope: !34)
!45 = !DILocation(line: 28, column: 1, scope: !34)
!46 = distinct !DISubprogram(name: "__nv_MAIN_F1L21_2", scope: !2, file: !3, line: 21, type: !10, scopeLine: 21, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!47 = !DILocalVariable(name: "__nv_MAIN_F1L21_2Arg0", arg: 1, scope: !46, file: !3, type: !12)
!48 = !DILocation(line: 0, scope: !46)
!49 = !DILocalVariable(name: "__nv_MAIN_F1L21_2Arg1", arg: 2, scope: !46, file: !3, type: !13)
!50 = !DILocalVariable(name: "__nv_MAIN_F1L21_2Arg2", arg: 3, scope: !46, file: !3, type: !13)
!51 = !DILocalVariable(name: "omp_sched_static", scope: !46, file: !3, type: !12)
!52 = !DILocalVariable(name: "omp_proc_bind_false", scope: !46, file: !3, type: !12)
!53 = !DILocalVariable(name: "omp_proc_bind_true", scope: !46, file: !3, type: !12)
!54 = !DILocalVariable(name: "omp_lock_hint_none", scope: !46, file: !3, type: !12)
!55 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !46, file: !3, type: !12)
!56 = !DILocation(line: 27, column: 1, scope: !46)
!57 = !DILocation(line: 22, column: 1, scope: !46)
!58 = !DILocalVariable(name: "i", scope: !46, file: !3, type: !12)
!59 = !DILocation(line: 26, column: 1, scope: !46)
!60 = !DILocalVariable(name: "__nv_MAIN_F1L22_3Arg0", arg: 1, scope: !9, file: !3, type: !12)
!61 = !DILocation(line: 0, scope: !9)
!62 = !DILocalVariable(name: "__nv_MAIN_F1L22_3Arg1", arg: 2, scope: !9, file: !3, type: !13)
!63 = !DILocalVariable(name: "__nv_MAIN_F1L22_3Arg2", arg: 3, scope: !9, file: !3, type: !13)
!64 = !DILocalVariable(name: "omp_sched_static", scope: !9, file: !3, type: !12)
!65 = !DILocalVariable(name: "omp_proc_bind_false", scope: !9, file: !3, type: !12)
!66 = !DILocalVariable(name: "omp_proc_bind_true", scope: !9, file: !3, type: !12)
!67 = !DILocalVariable(name: "omp_lock_hint_none", scope: !9, file: !3, type: !12)
!68 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !9, file: !3, type: !12)
!69 = !DILocation(line: 26, column: 1, scope: !9)
!70 = !DILocation(line: 22, column: 1, scope: !9)
!71 = !DILocalVariable(name: "i", scope: !9, file: !3, type: !12)
!72 = !DILocation(line: 24, column: 1, scope: !9)

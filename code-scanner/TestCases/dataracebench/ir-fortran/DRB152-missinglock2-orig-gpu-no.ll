; ModuleID = '/tmp/DRB152-missinglock2-orig-gpu-no-3dca42.ll'
source_filename = "/tmp/DRB152-missinglock2-orig-gpu-no-3dca42.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%astruct.dt62 = type <{ i8*, i8* }>
%astruct.dt95 = type <{ [16 x i8] }>
%astruct.dt149 = type <{ [16 x i8], i8*, i8* }>

@.C320_MAIN_ = internal constant i32 100
@.C285_MAIN_ = internal constant i32 1
@.C283_MAIN_ = internal constant i32 0
@.C320___nv_MAIN__F1L18_1 = internal constant i32 100
@.C283___nv_MAIN__F1L18_1 = internal constant i32 0
@.C285___nv_MAIN__F1L18_1 = internal constant i32 1
@.C320___nv_MAIN_F1L19_2 = internal constant i32 100
@.C285___nv_MAIN_F1L19_2 = internal constant i32 1
@.C283___nv_MAIN_F1L19_2 = internal constant i32 0
@.C285___nv_MAIN_F1L21_3 = internal constant i32 1
@.C283___nv_MAIN_F1L21_3 = internal constant i32 0

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__376 = alloca i32, align 4
  %var_310 = alloca i32, align 4
  %lck_309 = alloca i32, align 4
  %.uplevelArgPack0001_370 = alloca %astruct.dt62, align 16
  call void @llvm.dbg.value(metadata i32 1, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !14, metadata !DIExpression()), !dbg !10
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !15
  store i32 %0, i32* %__gtid_MAIN__376, align 4, !dbg !15
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !16
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !16
  call void (i8*, ...) %2(i8* %1), !dbg !16
  br label %L.LB1_364

L.LB1_364:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %var_310, metadata !17, metadata !DIExpression()), !dbg !10
  store i32 0, i32* %var_310, align 4, !dbg !18
  call void @llvm.dbg.declare(metadata i32* %lck_309, metadata !19, metadata !DIExpression()), !dbg !10
  %3 = bitcast i32* %lck_309 to i8*, !dbg !20
  %4 = bitcast %astruct.dt62* %.uplevelArgPack0001_370 to i8**, !dbg !20
  store i8* %3, i8** %4, align 8, !dbg !20
  %5 = bitcast i32* %var_310 to i8*, !dbg !20
  %6 = bitcast %astruct.dt62* %.uplevelArgPack0001_370 to i8*, !dbg !20
  %7 = getelementptr i8, i8* %6, i64 8, !dbg !20
  %8 = bitcast i8* %7 to i8**, !dbg !20
  store i8* %5, i8** %8, align 8, !dbg !20
  %9 = bitcast %astruct.dt62* %.uplevelArgPack0001_370 to i64*, !dbg !20
  call void @__nv_MAIN__F1L18_1_(i32* %__gtid_MAIN__376, i64* null, i64* %9), !dbg !20
  ret void, !dbg !15
}

define internal void @__nv_MAIN__F1L18_1_(i32* %__nv_MAIN__F1L18_1Arg0, i64* %__nv_MAIN__F1L18_1Arg1, i64* %__nv_MAIN__F1L18_1Arg2) #0 !dbg !21 {
L.entry:
  %__gtid___nv_MAIN__F1L18_1__394 = alloca i32, align 4
  %.uplevelArgPack0002_390 = alloca %astruct.dt95, align 16
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L18_1Arg0, metadata !25, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L18_1Arg1, metadata !27, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L18_1Arg2, metadata !28, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.value(metadata i32 1, metadata !29, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.value(metadata i32 0, metadata !30, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.value(metadata i32 1, metadata !31, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.value(metadata i32 0, metadata !32, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.value(metadata i32 1, metadata !33, metadata !DIExpression()), !dbg !26
  %0 = load i32, i32* %__nv_MAIN__F1L18_1Arg0, align 4, !dbg !34
  store i32 %0, i32* %__gtid___nv_MAIN__F1L18_1__394, align 4, !dbg !34
  br label %L.LB2_385

L.LB2_385:                                        ; preds = %L.entry
  br label %L.LB2_314

L.LB2_314:                                        ; preds = %L.LB2_385
  %1 = load i64, i64* %__nv_MAIN__F1L18_1Arg2, align 8, !dbg !35
  %2 = bitcast %astruct.dt95* %.uplevelArgPack0002_390 to i64*, !dbg !35
  store i64 %1, i64* %2, align 8, !dbg !35
  %3 = bitcast i64* %__nv_MAIN__F1L18_1Arg2 to i8*, !dbg !34
  %4 = getelementptr i8, i8* %3, i64 8, !dbg !34
  %5 = bitcast i8* %4 to i64*, !dbg !34
  %6 = load i64, i64* %5, align 8, !dbg !34
  %7 = bitcast %astruct.dt95* %.uplevelArgPack0002_390 to i8*, !dbg !34
  %8 = getelementptr i8, i8* %7, i64 8, !dbg !34
  %9 = bitcast i8* %8 to i64*, !dbg !34
  store i64 %6, i64* %9, align 8, !dbg !34
  %10 = load i32, i32* %__gtid___nv_MAIN__F1L18_1__394, align 4, !dbg !35
  call void @__kmpc_push_num_teams(i64* null, i32 %10, i32 1, i32 0), !dbg !35
  %11 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN_F1L19_2_ to i64*, !dbg !35
  %12 = bitcast %astruct.dt95* %.uplevelArgPack0002_390 to i64*, !dbg !35
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_teams(i64* null, i32 1, i64* %11, i64* %12), !dbg !35
  br label %L.LB2_335

L.LB2_335:                                        ; preds = %L.LB2_314
  ret void, !dbg !34
}

define internal void @__nv_MAIN_F1L19_2_(i32* %__nv_MAIN_F1L19_2Arg0, i64* %__nv_MAIN_F1L19_2Arg1, i64* %__nv_MAIN_F1L19_2Arg2) #0 !dbg !36 {
L.entry:
  %__gtid___nv_MAIN_F1L19_2__436 = alloca i32, align 4
  %.i0000p_322 = alloca i32, align 4
  %.i0001p_323 = alloca i32, align 4
  %.i0002p_324 = alloca i32, align 4
  %.i0003p_325 = alloca i32, align 4
  %i_321 = alloca i32, align 4
  %.du0001_342 = alloca i32, align 4
  %.de0001_343 = alloca i32, align 4
  %.di0001_344 = alloca i32, align 4
  %.ds0001_345 = alloca i32, align 4
  %.dl0001_347 = alloca i32, align 4
  %.dl0001.copy_430 = alloca i32, align 4
  %.de0001.copy_431 = alloca i32, align 4
  %.ds0001.copy_432 = alloca i32, align 4
  %.dX0001_346 = alloca i32, align 4
  %.dY0001_341 = alloca i32, align 4
  %.uplevelArgPack0003_455 = alloca %astruct.dt149, align 16
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L19_2Arg0, metadata !37, metadata !DIExpression()), !dbg !38
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L19_2Arg1, metadata !39, metadata !DIExpression()), !dbg !38
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L19_2Arg2, metadata !40, metadata !DIExpression()), !dbg !38
  call void @llvm.dbg.value(metadata i32 1, metadata !41, metadata !DIExpression()), !dbg !38
  call void @llvm.dbg.value(metadata i32 0, metadata !42, metadata !DIExpression()), !dbg !38
  call void @llvm.dbg.value(metadata i32 1, metadata !43, metadata !DIExpression()), !dbg !38
  call void @llvm.dbg.value(metadata i32 0, metadata !44, metadata !DIExpression()), !dbg !38
  call void @llvm.dbg.value(metadata i32 1, metadata !45, metadata !DIExpression()), !dbg !38
  %0 = load i32, i32* %__nv_MAIN_F1L19_2Arg0, align 4, !dbg !46
  store i32 %0, i32* %__gtid___nv_MAIN_F1L19_2__436, align 4, !dbg !46
  br label %L.LB4_419

L.LB4_419:                                        ; preds = %L.entry
  br label %L.LB4_317

L.LB4_317:                                        ; preds = %L.LB4_419
  br label %L.LB4_318

L.LB4_318:                                        ; preds = %L.LB4_317
  br label %L.LB4_319

L.LB4_319:                                        ; preds = %L.LB4_318
  store i32 0, i32* %.i0000p_322, align 4, !dbg !47
  store i32 1, i32* %.i0001p_323, align 4, !dbg !47
  store i32 100, i32* %.i0002p_324, align 4, !dbg !47
  store i32 1, i32* %.i0003p_325, align 4, !dbg !47
  %1 = load i32, i32* %.i0001p_323, align 4, !dbg !47
  call void @llvm.dbg.declare(metadata i32* %i_321, metadata !48, metadata !DIExpression()), !dbg !46
  store i32 %1, i32* %i_321, align 4, !dbg !47
  %2 = load i32, i32* %.i0002p_324, align 4, !dbg !47
  store i32 %2, i32* %.du0001_342, align 4, !dbg !47
  %3 = load i32, i32* %.i0002p_324, align 4, !dbg !47
  store i32 %3, i32* %.de0001_343, align 4, !dbg !47
  store i32 1, i32* %.di0001_344, align 4, !dbg !47
  %4 = load i32, i32* %.di0001_344, align 4, !dbg !47
  store i32 %4, i32* %.ds0001_345, align 4, !dbg !47
  %5 = load i32, i32* %.i0001p_323, align 4, !dbg !47
  store i32 %5, i32* %.dl0001_347, align 4, !dbg !47
  %6 = load i32, i32* %.dl0001_347, align 4, !dbg !47
  store i32 %6, i32* %.dl0001.copy_430, align 4, !dbg !47
  %7 = load i32, i32* %.de0001_343, align 4, !dbg !47
  store i32 %7, i32* %.de0001.copy_431, align 4, !dbg !47
  %8 = load i32, i32* %.ds0001_345, align 4, !dbg !47
  store i32 %8, i32* %.ds0001.copy_432, align 4, !dbg !47
  %9 = load i32, i32* %__gtid___nv_MAIN_F1L19_2__436, align 4, !dbg !47
  %10 = bitcast i32* %.i0000p_322 to i64*, !dbg !47
  %11 = bitcast i32* %.dl0001.copy_430 to i64*, !dbg !47
  %12 = bitcast i32* %.de0001.copy_431 to i64*, !dbg !47
  %13 = bitcast i32* %.ds0001.copy_432 to i64*, !dbg !47
  %14 = load i32, i32* %.ds0001.copy_432, align 4, !dbg !47
  call void @__kmpc_for_static_init_4(i64* null, i32 %9, i32 92, i64* %10, i64* %11, i64* %12, i64* %13, i32 %14, i32 1), !dbg !47
  %15 = load i32, i32* %.dl0001.copy_430, align 4, !dbg !47
  store i32 %15, i32* %.dl0001_347, align 4, !dbg !47
  %16 = load i32, i32* %.de0001.copy_431, align 4, !dbg !47
  store i32 %16, i32* %.de0001_343, align 4, !dbg !47
  %17 = load i32, i32* %.ds0001.copy_432, align 4, !dbg !47
  store i32 %17, i32* %.ds0001_345, align 4, !dbg !47
  %18 = load i32, i32* %.dl0001_347, align 4, !dbg !47
  store i32 %18, i32* %i_321, align 4, !dbg !47
  %19 = load i32, i32* %i_321, align 4, !dbg !47
  call void @llvm.dbg.value(metadata i32 %19, metadata !48, metadata !DIExpression()), !dbg !46
  store i32 %19, i32* %.dX0001_346, align 4, !dbg !47
  %20 = load i32, i32* %.dX0001_346, align 4, !dbg !47
  %21 = load i32, i32* %.du0001_342, align 4, !dbg !47
  %22 = icmp sgt i32 %20, %21, !dbg !47
  br i1 %22, label %L.LB4_340, label %L.LB4_487, !dbg !47

L.LB4_487:                                        ; preds = %L.LB4_319
  %23 = load i32, i32* %.du0001_342, align 4, !dbg !47
  %24 = load i32, i32* %.de0001_343, align 4, !dbg !47
  %25 = icmp slt i32 %23, %24, !dbg !47
  %26 = select i1 %25, i32 %23, i32 %24, !dbg !47
  store i32 %26, i32* %.de0001_343, align 4, !dbg !47
  %27 = load i32, i32* %.dX0001_346, align 4, !dbg !47
  store i32 %27, i32* %i_321, align 4, !dbg !47
  %28 = load i32, i32* %.di0001_344, align 4, !dbg !47
  %29 = load i32, i32* %.de0001_343, align 4, !dbg !47
  %30 = load i32, i32* %.dX0001_346, align 4, !dbg !47
  %31 = sub nsw i32 %29, %30, !dbg !47
  %32 = add nsw i32 %28, %31, !dbg !47
  %33 = load i32, i32* %.di0001_344, align 4, !dbg !47
  %34 = sdiv i32 %32, %33, !dbg !47
  store i32 %34, i32* %.dY0001_341, align 4, !dbg !47
  %35 = load i32, i32* %i_321, align 4, !dbg !47
  call void @llvm.dbg.value(metadata i32 %35, metadata !48, metadata !DIExpression()), !dbg !46
  store i32 %35, i32* %.i0001p_323, align 4, !dbg !47
  %36 = load i32, i32* %.de0001_343, align 4, !dbg !47
  store i32 %36, i32* %.i0002p_324, align 4, !dbg !47
  %37 = load i64, i64* %__nv_MAIN_F1L19_2Arg2, align 8, !dbg !47
  %38 = bitcast %astruct.dt149* %.uplevelArgPack0003_455 to i64*, !dbg !47
  store i64 %37, i64* %38, align 8, !dbg !47
  %39 = bitcast i64* %__nv_MAIN_F1L19_2Arg2 to i8*, !dbg !46
  %40 = getelementptr i8, i8* %39, i64 8, !dbg !46
  %41 = bitcast i8* %40 to i64*, !dbg !46
  %42 = load i64, i64* %41, align 8, !dbg !46
  %43 = bitcast %astruct.dt149* %.uplevelArgPack0003_455 to i8*, !dbg !46
  %44 = getelementptr i8, i8* %43, i64 8, !dbg !46
  %45 = bitcast i8* %44 to i64*, !dbg !46
  store i64 %42, i64* %45, align 8, !dbg !46
  %46 = bitcast i32* %.i0001p_323 to i8*, !dbg !47
  %47 = bitcast %astruct.dt149* %.uplevelArgPack0003_455 to i8*, !dbg !47
  %48 = getelementptr i8, i8* %47, i64 16, !dbg !47
  %49 = bitcast i8* %48 to i8**, !dbg !47
  store i8* %46, i8** %49, align 8, !dbg !47
  %50 = bitcast i32* %.i0002p_324 to i8*, !dbg !47
  %51 = bitcast %astruct.dt149* %.uplevelArgPack0003_455 to i8*, !dbg !47
  %52 = getelementptr i8, i8* %51, i64 24, !dbg !47
  %53 = bitcast i8* %52 to i8**, !dbg !47
  store i8* %50, i8** %53, align 8, !dbg !47
  br label %L.LB4_462, !dbg !47

L.LB4_462:                                        ; preds = %L.LB4_487
  %54 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN_F1L21_3_ to i64*, !dbg !47
  %55 = bitcast %astruct.dt149* %.uplevelArgPack0003_455 to i64*, !dbg !47
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %54, i64* %55), !dbg !47
  br label %L.LB4_340

L.LB4_340:                                        ; preds = %L.LB4_462, %L.LB4_319
  %56 = load i32, i32* %__gtid___nv_MAIN_F1L19_2__436, align 4, !dbg !49
  call void @__kmpc_for_static_fini(i64* null, i32 %56), !dbg !49
  br label %L.LB4_332

L.LB4_332:                                        ; preds = %L.LB4_340
  br label %L.LB4_333

L.LB4_333:                                        ; preds = %L.LB4_332
  br label %L.LB4_334

L.LB4_334:                                        ; preds = %L.LB4_333
  ret void, !dbg !46
}

define internal void @__nv_MAIN_F1L21_3_(i32* %__nv_MAIN_F1L21_3Arg0, i64* %__nv_MAIN_F1L21_3Arg1, i64* %__nv_MAIN_F1L21_3Arg2) #0 !dbg !50 {
L.entry:
  %__gtid___nv_MAIN_F1L21_3__508 = alloca i32, align 4
  %.i0004p_330 = alloca i32, align 4
  %i_329 = alloca i32, align 4
  %.du0002p_354 = alloca i32, align 4
  %.de0002p_355 = alloca i32, align 4
  %.di0002p_356 = alloca i32, align 4
  %.ds0002p_357 = alloca i32, align 4
  %.dl0002p_359 = alloca i32, align 4
  %.dl0002p.copy_502 = alloca i32, align 4
  %.de0002p.copy_503 = alloca i32, align 4
  %.ds0002p.copy_504 = alloca i32, align 4
  %.dX0002p_358 = alloca i32, align 4
  %.dY0002p_353 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L21_3Arg0, metadata !51, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L21_3Arg1, metadata !53, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L21_3Arg2, metadata !54, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.value(metadata i32 1, metadata !55, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.value(metadata i32 0, metadata !56, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.value(metadata i32 1, metadata !57, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.value(metadata i32 0, metadata !58, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.value(metadata i32 1, metadata !59, metadata !DIExpression()), !dbg !52
  %0 = load i32, i32* %__nv_MAIN_F1L21_3Arg0, align 4, !dbg !60
  store i32 %0, i32* %__gtid___nv_MAIN_F1L21_3__508, align 4, !dbg !60
  br label %L.LB6_491

L.LB6_491:                                        ; preds = %L.entry
  br label %L.LB6_328

L.LB6_328:                                        ; preds = %L.LB6_491
  store i32 0, i32* %.i0004p_330, align 4, !dbg !61
  %1 = bitcast i64* %__nv_MAIN_F1L21_3Arg2 to i8*, !dbg !61
  %2 = getelementptr i8, i8* %1, i64 16, !dbg !61
  %3 = bitcast i8* %2 to i32**, !dbg !61
  %4 = load i32*, i32** %3, align 8, !dbg !61
  %5 = load i32, i32* %4, align 4, !dbg !61
  call void @llvm.dbg.declare(metadata i32* %i_329, metadata !62, metadata !DIExpression()), !dbg !60
  store i32 %5, i32* %i_329, align 4, !dbg !61
  %6 = bitcast i64* %__nv_MAIN_F1L21_3Arg2 to i8*, !dbg !61
  %7 = getelementptr i8, i8* %6, i64 24, !dbg !61
  %8 = bitcast i8* %7 to i32**, !dbg !61
  %9 = load i32*, i32** %8, align 8, !dbg !61
  %10 = load i32, i32* %9, align 4, !dbg !61
  store i32 %10, i32* %.du0002p_354, align 4, !dbg !61
  %11 = bitcast i64* %__nv_MAIN_F1L21_3Arg2 to i8*, !dbg !61
  %12 = getelementptr i8, i8* %11, i64 24, !dbg !61
  %13 = bitcast i8* %12 to i32**, !dbg !61
  %14 = load i32*, i32** %13, align 8, !dbg !61
  %15 = load i32, i32* %14, align 4, !dbg !61
  store i32 %15, i32* %.de0002p_355, align 4, !dbg !61
  store i32 1, i32* %.di0002p_356, align 4, !dbg !61
  %16 = load i32, i32* %.di0002p_356, align 4, !dbg !61
  store i32 %16, i32* %.ds0002p_357, align 4, !dbg !61
  %17 = bitcast i64* %__nv_MAIN_F1L21_3Arg2 to i8*, !dbg !61
  %18 = getelementptr i8, i8* %17, i64 16, !dbg !61
  %19 = bitcast i8* %18 to i32**, !dbg !61
  %20 = load i32*, i32** %19, align 8, !dbg !61
  %21 = load i32, i32* %20, align 4, !dbg !61
  store i32 %21, i32* %.dl0002p_359, align 4, !dbg !61
  %22 = load i32, i32* %.dl0002p_359, align 4, !dbg !61
  store i32 %22, i32* %.dl0002p.copy_502, align 4, !dbg !61
  %23 = load i32, i32* %.de0002p_355, align 4, !dbg !61
  store i32 %23, i32* %.de0002p.copy_503, align 4, !dbg !61
  %24 = load i32, i32* %.ds0002p_357, align 4, !dbg !61
  store i32 %24, i32* %.ds0002p.copy_504, align 4, !dbg !61
  %25 = load i32, i32* %__gtid___nv_MAIN_F1L21_3__508, align 4, !dbg !61
  %26 = bitcast i32* %.i0004p_330 to i64*, !dbg !61
  %27 = bitcast i32* %.dl0002p.copy_502 to i64*, !dbg !61
  %28 = bitcast i32* %.de0002p.copy_503 to i64*, !dbg !61
  %29 = bitcast i32* %.ds0002p.copy_504 to i64*, !dbg !61
  %30 = load i32, i32* %.ds0002p.copy_504, align 4, !dbg !61
  call void @__kmpc_for_static_init_4(i64* null, i32 %25, i32 34, i64* %26, i64* %27, i64* %28, i64* %29, i32 %30, i32 1), !dbg !61
  %31 = load i32, i32* %.dl0002p.copy_502, align 4, !dbg !61
  store i32 %31, i32* %.dl0002p_359, align 4, !dbg !61
  %32 = load i32, i32* %.de0002p.copy_503, align 4, !dbg !61
  store i32 %32, i32* %.de0002p_355, align 4, !dbg !61
  %33 = load i32, i32* %.ds0002p.copy_504, align 4, !dbg !61
  store i32 %33, i32* %.ds0002p_357, align 4, !dbg !61
  %34 = load i32, i32* %.dl0002p_359, align 4, !dbg !61
  store i32 %34, i32* %i_329, align 4, !dbg !61
  %35 = load i32, i32* %i_329, align 4, !dbg !61
  call void @llvm.dbg.value(metadata i32 %35, metadata !62, metadata !DIExpression()), !dbg !60
  store i32 %35, i32* %.dX0002p_358, align 4, !dbg !61
  %36 = load i32, i32* %.dX0002p_358, align 4, !dbg !61
  %37 = load i32, i32* %.du0002p_354, align 4, !dbg !61
  %38 = icmp sgt i32 %36, %37, !dbg !61
  br i1 %38, label %L.LB6_352, label %L.LB6_517, !dbg !61

L.LB6_517:                                        ; preds = %L.LB6_328
  %39 = load i32, i32* %.dX0002p_358, align 4, !dbg !61
  store i32 %39, i32* %i_329, align 4, !dbg !61
  %40 = load i32, i32* %.di0002p_356, align 4, !dbg !61
  %41 = load i32, i32* %.de0002p_355, align 4, !dbg !61
  %42 = load i32, i32* %.dX0002p_358, align 4, !dbg !61
  %43 = sub nsw i32 %41, %42, !dbg !61
  %44 = add nsw i32 %40, %43, !dbg !61
  %45 = load i32, i32* %.di0002p_356, align 4, !dbg !61
  %46 = sdiv i32 %44, %45, !dbg !61
  store i32 %46, i32* %.dY0002p_353, align 4, !dbg !61
  %47 = load i32, i32* %.dY0002p_353, align 4, !dbg !61
  %48 = icmp sle i32 %47, 0, !dbg !61
  br i1 %48, label %L.LB6_362, label %L.LB6_361, !dbg !61

L.LB6_361:                                        ; preds = %L.LB6_361, %L.LB6_517
  %49 = bitcast i64* %__nv_MAIN_F1L21_3Arg2 to i64**, !dbg !63
  %50 = load i64*, i64** %49, align 8, !dbg !63
  call void @omp_set_lock_(i64* %50), !dbg !63
  %51 = bitcast i64* %__nv_MAIN_F1L21_3Arg2 to i8*, !dbg !64
  %52 = getelementptr i8, i8* %51, i64 8, !dbg !64
  %53 = bitcast i8* %52 to i32**, !dbg !64
  %54 = load i32*, i32** %53, align 8, !dbg !64
  %55 = load i32, i32* %54, align 4, !dbg !64
  %56 = add nsw i32 %55, 1, !dbg !64
  %57 = bitcast i64* %__nv_MAIN_F1L21_3Arg2 to i8*, !dbg !64
  %58 = getelementptr i8, i8* %57, i64 8, !dbg !64
  %59 = bitcast i8* %58 to i32**, !dbg !64
  %60 = load i32*, i32** %59, align 8, !dbg !64
  store i32 %56, i32* %60, align 4, !dbg !64
  %61 = bitcast i64* %__nv_MAIN_F1L21_3Arg2 to i64**, !dbg !65
  %62 = load i64*, i64** %61, align 8, !dbg !65
  call void @omp_unset_lock_(i64* %62), !dbg !65
  %63 = load i32, i32* %.di0002p_356, align 4, !dbg !60
  %64 = load i32, i32* %i_329, align 4, !dbg !60
  call void @llvm.dbg.value(metadata i32 %64, metadata !62, metadata !DIExpression()), !dbg !60
  %65 = add nsw i32 %63, %64, !dbg !60
  store i32 %65, i32* %i_329, align 4, !dbg !60
  %66 = load i32, i32* %.dY0002p_353, align 4, !dbg !60
  %67 = sub nsw i32 %66, 1, !dbg !60
  store i32 %67, i32* %.dY0002p_353, align 4, !dbg !60
  %68 = load i32, i32* %.dY0002p_353, align 4, !dbg !60
  %69 = icmp sgt i32 %68, 0, !dbg !60
  br i1 %69, label %L.LB6_361, label %L.LB6_362, !dbg !60

L.LB6_362:                                        ; preds = %L.LB6_361, %L.LB6_517
  br label %L.LB6_352

L.LB6_352:                                        ; preds = %L.LB6_362, %L.LB6_328
  %70 = load i32, i32* %__gtid___nv_MAIN_F1L21_3__508, align 4, !dbg !60
  call void @__kmpc_for_static_fini(i64* null, i32 %70), !dbg !60
  br label %L.LB6_331

L.LB6_331:                                        ; preds = %L.LB6_352
  ret void, !dbg !60
}

declare void @__kmpc_for_static_fini(i64*, i32) #0

declare void @__kmpc_fork_call(i64*, i32, i64*, i64*, ...) #0

declare void @__kmpc_end_serialized_parallel(i64*, i32) #0

declare void @__kmpc_serialized_parallel(i64*, i32) #0

declare void @__kmpc_for_static_init_4(i64*, i32, i32, i64*, i64*, i64*, i64*, i32, i32) #0

declare void @__kmpc_fork_teams(i64*, i32, i64*, i64*, ...) #0

declare void @__kmpc_push_num_teams(i64*, i32, i32, i32) #0

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare void @fort_init(...) #0

declare signext i32 @__kmpc_global_thread_num(i64*) #0

declare void @omp_unset_lock_(i64*) #0

declare void @omp_set_lock_(i64*) #0

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { "no-frame-pointer-elim-non-leaf" }
attributes #1 = { nounwind readnone speculatable }

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !4, globals: !4, imports: !4)
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB152-missinglock2-orig-gpu-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb152_missinglock2_orig_gpu_no", scope: !2, file: !3, line: 10, type: !6, scopeLine: 10, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
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
!16 = !DILocation(line: 10, column: 1, scope: !5)
!17 = !DILocalVariable(name: "var", scope: !5, file: !3, type: !9)
!18 = !DILocation(line: 16, column: 1, scope: !5)
!19 = !DILocalVariable(name: "lck", scope: !5, file: !3, type: !9)
!20 = !DILocation(line: 28, column: 1, scope: !5)
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
!34 = !DILocation(line: 28, column: 1, scope: !21)
!35 = !DILocation(line: 19, column: 1, scope: !21)
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
!46 = !DILocation(line: 27, column: 1, scope: !36)
!47 = !DILocation(line: 21, column: 1, scope: !36)
!48 = !DILocalVariable(name: "i", scope: !36, file: !3, type: !9)
!49 = !DILocation(line: 25, column: 1, scope: !36)
!50 = distinct !DISubprogram(name: "__nv_MAIN_F1L21_3", scope: !2, file: !3, line: 21, type: !22, scopeLine: 21, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!51 = !DILocalVariable(name: "__nv_MAIN_F1L21_3Arg0", arg: 1, scope: !50, file: !3, type: !9)
!52 = !DILocation(line: 0, scope: !50)
!53 = !DILocalVariable(name: "__nv_MAIN_F1L21_3Arg1", arg: 2, scope: !50, file: !3, type: !24)
!54 = !DILocalVariable(name: "__nv_MAIN_F1L21_3Arg2", arg: 3, scope: !50, file: !3, type: !24)
!55 = !DILocalVariable(name: "omp_sched_static", scope: !50, file: !3, type: !9)
!56 = !DILocalVariable(name: "omp_proc_bind_false", scope: !50, file: !3, type: !9)
!57 = !DILocalVariable(name: "omp_proc_bind_true", scope: !50, file: !3, type: !9)
!58 = !DILocalVariable(name: "omp_lock_hint_none", scope: !50, file: !3, type: !9)
!59 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !50, file: !3, type: !9)
!60 = !DILocation(line: 25, column: 1, scope: !50)
!61 = !DILocation(line: 21, column: 1, scope: !50)
!62 = !DILocalVariable(name: "i", scope: !50, file: !3, type: !9)
!63 = !DILocation(line: 22, column: 1, scope: !50)
!64 = !DILocation(line: 23, column: 1, scope: !50)
!65 = !DILocation(line: 24, column: 1, scope: !50)

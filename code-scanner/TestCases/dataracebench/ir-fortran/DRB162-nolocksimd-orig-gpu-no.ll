; ModuleID = '/tmp/DRB162-nolocksimd-orig-gpu-no-ea0901.ll'
source_filename = "/tmp/DRB162-nolocksimd-orig-gpu-no-ea0901.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.BSS1 = type <{ [32 x i8] }>
%astruct.dt61 = type <{ i8* }>
%astruct.dt103 = type <{ [8 x i8] }>
%astruct.dt157 = type <{ [8 x i8], i8*, i8* }>

@.BSS1 = internal global %struct.BSS1 zeroinitializer, align 32, !dbg !0
@.C308_MAIN_ = internal constant i32 25
@.C284_MAIN_ = internal constant i64 0
@.C348_MAIN_ = internal constant i32 6
@.C345_MAIN_ = internal constant [58 x i8] c"micro-benchmarks-fortran/DRB162-nolocksimd-orig-gpu-no.f95"
@.C347_MAIN_ = internal constant i32 37
@.C336_MAIN_ = internal constant i32 9
@.C311_MAIN_ = internal constant i64 8
@.C286_MAIN_ = internal constant i64 1
@.C309_MAIN_ = internal constant i32 20
@.C317_MAIN_ = internal constant i32 1048
@.C300_MAIN_ = internal constant i32 8
@.C285_MAIN_ = internal constant i32 1
@.C283_MAIN_ = internal constant i32 0
@.C300___nv_MAIN__F1L21_1 = internal constant i32 8
@.C336___nv_MAIN__F1L21_1 = internal constant i32 9
@.C311___nv_MAIN__F1L21_1 = internal constant i64 8
@.C286___nv_MAIN__F1L21_1 = internal constant i64 1
@.C309___nv_MAIN__F1L21_1 = internal constant i32 20
@.C283___nv_MAIN__F1L21_1 = internal constant i32 0
@.C317___nv_MAIN__F1L21_1 = internal constant i32 1048
@.C285___nv_MAIN__F1L21_1 = internal constant i32 1
@.C300___nv_MAIN_F1L22_2 = internal constant i32 8
@.C336___nv_MAIN_F1L22_2 = internal constant i32 9
@.C311___nv_MAIN_F1L22_2 = internal constant i64 8
@.C286___nv_MAIN_F1L22_2 = internal constant i64 1
@.C309___nv_MAIN_F1L22_2 = internal constant i32 20
@.C285___nv_MAIN_F1L22_2 = internal constant i32 1
@.C283___nv_MAIN_F1L22_2 = internal constant i32 0
@.C300___nv_MAIN_F1L24_3 = internal constant i32 8
@.C336___nv_MAIN_F1L24_3 = internal constant i32 9
@.C285___nv_MAIN_F1L24_3 = internal constant i32 1
@.C283___nv_MAIN_F1L24_3 = internal constant i32 0
@.C311___nv_MAIN_F1L24_3 = internal constant i64 8
@.C286___nv_MAIN_F1L24_3 = internal constant i64 1

define void @MAIN_() #0 !dbg !2 {
L.entry:
  %__gtid_MAIN__416 = alloca i32, align 4
  %.dY0001_361 = alloca i32, align 4
  %i_312 = alloca i32, align 4
  %.uplevelArgPack0001_413 = alloca %astruct.dt61, align 8
  %.dY0007_397 = alloca i32, align 4
  %z__io_350 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 8, metadata !25, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.value(metadata i32 1, metadata !27, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.value(metadata i32 0, metadata !28, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.value(metadata i32 1, metadata !29, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.value(metadata i32 0, metadata !30, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.value(metadata i32 1, metadata !31, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.value(metadata i32 8, metadata !32, metadata !DIExpression()), !dbg !26
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !33
  store i32 %0, i32* %__gtid_MAIN__416, align 4, !dbg !33
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !34
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !34
  call void (i8*, ...) %2(i8* %1), !dbg !34
  br label %L.LB1_400

L.LB1_400:                                        ; preds = %L.entry
  store i32 8, i32* %.dY0001_361, align 4, !dbg !35
  call void @llvm.dbg.declare(metadata i32* %i_312, metadata !36, metadata !DIExpression()), !dbg !26
  store i32 1, i32* %i_312, align 4, !dbg !35
  br label %L.LB1_359

L.LB1_359:                                        ; preds = %L.LB1_359, %L.LB1_400
  %3 = load i32, i32* %i_312, align 4, !dbg !37
  call void @llvm.dbg.value(metadata i32 %3, metadata !36, metadata !DIExpression()), !dbg !26
  %4 = sext i32 %3 to i64, !dbg !37
  %5 = bitcast %struct.BSS1* @.BSS1 to i8*, !dbg !37
  %6 = getelementptr i8, i8* %5, i64 -4, !dbg !37
  %7 = bitcast i8* %6 to i32*, !dbg !37
  %8 = getelementptr i32, i32* %7, i64 %4, !dbg !37
  store i32 0, i32* %8, align 4, !dbg !37
  %9 = load i32, i32* %i_312, align 4, !dbg !38
  call void @llvm.dbg.value(metadata i32 %9, metadata !36, metadata !DIExpression()), !dbg !26
  %10 = add nsw i32 %9, 1, !dbg !38
  store i32 %10, i32* %i_312, align 4, !dbg !38
  %11 = load i32, i32* %.dY0001_361, align 4, !dbg !38
  %12 = sub nsw i32 %11, 1, !dbg !38
  store i32 %12, i32* %.dY0001_361, align 4, !dbg !38
  %13 = load i32, i32* %.dY0001_361, align 4, !dbg !38
  %14 = icmp sgt i32 %13, 0, !dbg !38
  br i1 %14, label %L.LB1_359, label %L.LB1_429, !dbg !38

L.LB1_429:                                        ; preds = %L.LB1_359
  %15 = bitcast %astruct.dt61* %.uplevelArgPack0001_413 to i64*, !dbg !39
  call void @__nv_MAIN__F1L21_1_(i32* %__gtid_MAIN__416, i64* null, i64* %15), !dbg !39
  store i32 8, i32* %.dY0007_397, align 4, !dbg !40
  store i32 1, i32* %i_312, align 4, !dbg !40
  br label %L.LB1_395

L.LB1_395:                                        ; preds = %L.LB1_398, %L.LB1_429
  %16 = load i32, i32* %i_312, align 4, !dbg !41
  call void @llvm.dbg.value(metadata i32 %16, metadata !36, metadata !DIExpression()), !dbg !26
  %17 = sext i32 %16 to i64, !dbg !41
  %18 = bitcast %struct.BSS1* @.BSS1 to i8*, !dbg !41
  %19 = getelementptr i8, i8* %18, i64 -4, !dbg !41
  %20 = bitcast i8* %19 to i32*, !dbg !41
  %21 = getelementptr i32, i32* %20, i64 %17, !dbg !41
  %22 = load i32, i32* %21, align 4, !dbg !41
  %23 = icmp eq i32 %22, 20, !dbg !41
  br i1 %23, label %L.LB1_398, label %L.LB1_430, !dbg !41

L.LB1_430:                                        ; preds = %L.LB1_395
  call void (...) @_mp_bcs_nest(), !dbg !42
  %24 = bitcast i32* @.C347_MAIN_ to i8*, !dbg !42
  %25 = bitcast [58 x i8]* @.C345_MAIN_ to i8*, !dbg !42
  %26 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !42
  call void (i8*, i8*, i64, ...) %26(i8* %24, i8* %25, i64 58), !dbg !42
  %27 = bitcast i32* @.C348_MAIN_ to i8*, !dbg !42
  %28 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !42
  %29 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !42
  %30 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !42
  %31 = call i32 (i8*, i8*, i8*, i8*, ...) %30(i8* %27, i8* null, i8* %28, i8* %29), !dbg !42
  call void @llvm.dbg.declare(metadata i32* %z__io_350, metadata !43, metadata !DIExpression()), !dbg !26
  store i32 %31, i32* %z__io_350, align 4, !dbg !42
  %32 = load i32, i32* %i_312, align 4, !dbg !42
  call void @llvm.dbg.value(metadata i32 %32, metadata !36, metadata !DIExpression()), !dbg !26
  %33 = sext i32 %32 to i64, !dbg !42
  %34 = bitcast %struct.BSS1* @.BSS1 to i8*, !dbg !42
  %35 = getelementptr i8, i8* %34, i64 -4, !dbg !42
  %36 = bitcast i8* %35 to i32*, !dbg !42
  %37 = getelementptr i32, i32* %36, i64 %33, !dbg !42
  %38 = load i32, i32* %37, align 4, !dbg !42
  %39 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !42
  %40 = call i32 (i32, i32, ...) %39(i32 %38, i32 25), !dbg !42
  store i32 %40, i32* %z__io_350, align 4, !dbg !42
  %41 = call i32 (...) @f90io_ldw_end(), !dbg !42
  store i32 %41, i32* %z__io_350, align 4, !dbg !42
  call void (...) @_mp_ecs_nest(), !dbg !42
  br label %L.LB1_398

L.LB1_398:                                        ; preds = %L.LB1_430, %L.LB1_395
  %42 = load i32, i32* %i_312, align 4, !dbg !44
  call void @llvm.dbg.value(metadata i32 %42, metadata !36, metadata !DIExpression()), !dbg !26
  %43 = add nsw i32 %42, 1, !dbg !44
  store i32 %43, i32* %i_312, align 4, !dbg !44
  %44 = load i32, i32* %.dY0007_397, align 4, !dbg !44
  %45 = sub nsw i32 %44, 1, !dbg !44
  store i32 %45, i32* %.dY0007_397, align 4, !dbg !44
  %46 = load i32, i32* %.dY0007_397, align 4, !dbg !44
  %47 = icmp sgt i32 %46, 0, !dbg !44
  br i1 %47, label %L.LB1_395, label %L.LB1_431, !dbg !44

L.LB1_431:                                        ; preds = %L.LB1_398
  ret void, !dbg !33
}

define internal void @__nv_MAIN__F1L21_1_(i32* %__nv_MAIN__F1L21_1Arg0, i64* %__nv_MAIN__F1L21_1Arg1, i64* %__nv_MAIN__F1L21_1Arg2) #0 !dbg !45 {
L.entry:
  %__gtid___nv_MAIN__F1L21_1__444 = alloca i32, align 4
  %.uplevelArgPack0002_440 = alloca %astruct.dt103, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L21_1Arg0, metadata !46, metadata !DIExpression()), !dbg !47
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L21_1Arg1, metadata !48, metadata !DIExpression()), !dbg !47
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L21_1Arg2, metadata !49, metadata !DIExpression()), !dbg !47
  call void @llvm.dbg.value(metadata i32 8, metadata !50, metadata !DIExpression()), !dbg !47
  call void @llvm.dbg.value(metadata i32 1, metadata !51, metadata !DIExpression()), !dbg !47
  call void @llvm.dbg.value(metadata i32 0, metadata !52, metadata !DIExpression()), !dbg !47
  call void @llvm.dbg.value(metadata i32 1, metadata !53, metadata !DIExpression()), !dbg !47
  call void @llvm.dbg.value(metadata i32 0, metadata !54, metadata !DIExpression()), !dbg !47
  call void @llvm.dbg.value(metadata i32 1, metadata !55, metadata !DIExpression()), !dbg !47
  call void @llvm.dbg.value(metadata i32 8, metadata !56, metadata !DIExpression()), !dbg !47
  %0 = load i32, i32* %__nv_MAIN__F1L21_1Arg0, align 4, !dbg !57
  store i32 %0, i32* %__gtid___nv_MAIN__F1L21_1__444, align 4, !dbg !57
  br label %L.LB2_435

L.LB2_435:                                        ; preds = %L.entry
  br label %L.LB2_316

L.LB2_316:                                        ; preds = %L.LB2_435
  %1 = load i64, i64* %__nv_MAIN__F1L21_1Arg2, align 8, !dbg !58
  %2 = bitcast %astruct.dt103* %.uplevelArgPack0002_440 to i64*, !dbg !58
  store i64 %1, i64* %2, align 8, !dbg !58
  %3 = load i32, i32* %__gtid___nv_MAIN__F1L21_1__444, align 4, !dbg !58
  call void @__kmpc_push_num_teams(i64* null, i32 %3, i32 1, i32 1048), !dbg !58
  %4 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN_F1L22_2_ to i64*, !dbg !58
  %5 = bitcast %astruct.dt103* %.uplevelArgPack0002_440 to i64*, !dbg !58
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_teams(i64* null, i32 1, i64* %4, i64* %5), !dbg !58
  br label %L.LB2_343

L.LB2_343:                                        ; preds = %L.LB2_316
  ret void, !dbg !57
}

define internal void @__nv_MAIN_F1L22_2_(i32* %__nv_MAIN_F1L22_2Arg0, i64* %__nv_MAIN_F1L22_2Arg1, i64* %__nv_MAIN_F1L22_2Arg2) #0 !dbg !59 {
L.entry:
  %__gtid___nv_MAIN_F1L22_2__484 = alloca i32, align 4
  %.i0000p_324 = alloca i32, align 4
  %.i0001p_325 = alloca i32, align 4
  %.i0002p_326 = alloca i32, align 4
  %.i0003p_327 = alloca i32, align 4
  %i_323 = alloca i32, align 4
  %.du0002_365 = alloca i32, align 4
  %.de0002_366 = alloca i32, align 4
  %.di0002_367 = alloca i32, align 4
  %.ds0002_368 = alloca i32, align 4
  %.dl0002_370 = alloca i32, align 4
  %.dl0002.copy_478 = alloca i32, align 4
  %.de0002.copy_479 = alloca i32, align 4
  %.ds0002.copy_480 = alloca i32, align 4
  %.dX0002_369 = alloca i32, align 4
  %.dY0002_364 = alloca i32, align 4
  %.uplevelArgPack0003_503 = alloca %astruct.dt157, align 16
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L22_2Arg0, metadata !60, metadata !DIExpression()), !dbg !61
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L22_2Arg1, metadata !62, metadata !DIExpression()), !dbg !61
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L22_2Arg2, metadata !63, metadata !DIExpression()), !dbg !61
  call void @llvm.dbg.value(metadata i32 8, metadata !64, metadata !DIExpression()), !dbg !61
  call void @llvm.dbg.value(metadata i32 1, metadata !65, metadata !DIExpression()), !dbg !61
  call void @llvm.dbg.value(metadata i32 0, metadata !66, metadata !DIExpression()), !dbg !61
  call void @llvm.dbg.value(metadata i32 1, metadata !67, metadata !DIExpression()), !dbg !61
  call void @llvm.dbg.value(metadata i32 0, metadata !68, metadata !DIExpression()), !dbg !61
  call void @llvm.dbg.value(metadata i32 1, metadata !69, metadata !DIExpression()), !dbg !61
  call void @llvm.dbg.value(metadata i32 8, metadata !70, metadata !DIExpression()), !dbg !61
  %0 = load i32, i32* %__nv_MAIN_F1L22_2Arg0, align 4, !dbg !71
  store i32 %0, i32* %__gtid___nv_MAIN_F1L22_2__484, align 4, !dbg !71
  br label %L.LB4_467

L.LB4_467:                                        ; preds = %L.entry
  br label %L.LB4_320

L.LB4_320:                                        ; preds = %L.LB4_467
  br label %L.LB4_321

L.LB4_321:                                        ; preds = %L.LB4_320
  br label %L.LB4_322

L.LB4_322:                                        ; preds = %L.LB4_321
  store i32 0, i32* %.i0000p_324, align 4, !dbg !72
  store i32 1, i32* %.i0001p_325, align 4, !dbg !72
  store i32 20, i32* %.i0002p_326, align 4, !dbg !72
  store i32 1, i32* %.i0003p_327, align 4, !dbg !72
  %1 = load i32, i32* %.i0001p_325, align 4, !dbg !72
  call void @llvm.dbg.declare(metadata i32* %i_323, metadata !73, metadata !DIExpression()), !dbg !71
  store i32 %1, i32* %i_323, align 4, !dbg !72
  %2 = load i32, i32* %.i0002p_326, align 4, !dbg !72
  store i32 %2, i32* %.du0002_365, align 4, !dbg !72
  %3 = load i32, i32* %.i0002p_326, align 4, !dbg !72
  store i32 %3, i32* %.de0002_366, align 4, !dbg !72
  store i32 1, i32* %.di0002_367, align 4, !dbg !72
  %4 = load i32, i32* %.di0002_367, align 4, !dbg !72
  store i32 %4, i32* %.ds0002_368, align 4, !dbg !72
  %5 = load i32, i32* %.i0001p_325, align 4, !dbg !72
  store i32 %5, i32* %.dl0002_370, align 4, !dbg !72
  %6 = load i32, i32* %.dl0002_370, align 4, !dbg !72
  store i32 %6, i32* %.dl0002.copy_478, align 4, !dbg !72
  %7 = load i32, i32* %.de0002_366, align 4, !dbg !72
  store i32 %7, i32* %.de0002.copy_479, align 4, !dbg !72
  %8 = load i32, i32* %.ds0002_368, align 4, !dbg !72
  store i32 %8, i32* %.ds0002.copy_480, align 4, !dbg !72
  %9 = load i32, i32* %__gtid___nv_MAIN_F1L22_2__484, align 4, !dbg !72
  %10 = bitcast i32* %.i0000p_324 to i64*, !dbg !72
  %11 = bitcast i32* %.dl0002.copy_478 to i64*, !dbg !72
  %12 = bitcast i32* %.de0002.copy_479 to i64*, !dbg !72
  %13 = bitcast i32* %.ds0002.copy_480 to i64*, !dbg !72
  %14 = load i32, i32* %.ds0002.copy_480, align 4, !dbg !72
  call void @__kmpc_for_static_init_4(i64* null, i32 %9, i32 92, i64* %10, i64* %11, i64* %12, i64* %13, i32 %14, i32 1), !dbg !72
  %15 = load i32, i32* %.dl0002.copy_478, align 4, !dbg !72
  store i32 %15, i32* %.dl0002_370, align 4, !dbg !72
  %16 = load i32, i32* %.de0002.copy_479, align 4, !dbg !72
  store i32 %16, i32* %.de0002_366, align 4, !dbg !72
  %17 = load i32, i32* %.ds0002.copy_480, align 4, !dbg !72
  store i32 %17, i32* %.ds0002_368, align 4, !dbg !72
  %18 = load i32, i32* %.dl0002_370, align 4, !dbg !72
  store i32 %18, i32* %i_323, align 4, !dbg !72
  %19 = load i32, i32* %i_323, align 4, !dbg !72
  call void @llvm.dbg.value(metadata i32 %19, metadata !73, metadata !DIExpression()), !dbg !71
  store i32 %19, i32* %.dX0002_369, align 4, !dbg !72
  %20 = load i32, i32* %.dX0002_369, align 4, !dbg !72
  %21 = load i32, i32* %.du0002_365, align 4, !dbg !72
  %22 = icmp sgt i32 %20, %21, !dbg !72
  br i1 %22, label %L.LB4_363, label %L.LB4_534, !dbg !72

L.LB4_534:                                        ; preds = %L.LB4_322
  %23 = load i32, i32* %.du0002_365, align 4, !dbg !72
  %24 = load i32, i32* %.de0002_366, align 4, !dbg !72
  %25 = icmp slt i32 %23, %24, !dbg !72
  %26 = select i1 %25, i32 %23, i32 %24, !dbg !72
  store i32 %26, i32* %.de0002_366, align 4, !dbg !72
  %27 = load i32, i32* %.dX0002_369, align 4, !dbg !72
  store i32 %27, i32* %i_323, align 4, !dbg !72
  %28 = load i32, i32* %.di0002_367, align 4, !dbg !72
  %29 = load i32, i32* %.de0002_366, align 4, !dbg !72
  %30 = load i32, i32* %.dX0002_369, align 4, !dbg !72
  %31 = sub nsw i32 %29, %30, !dbg !72
  %32 = add nsw i32 %28, %31, !dbg !72
  %33 = load i32, i32* %.di0002_367, align 4, !dbg !72
  %34 = sdiv i32 %32, %33, !dbg !72
  store i32 %34, i32* %.dY0002_364, align 4, !dbg !72
  %35 = load i32, i32* %i_323, align 4, !dbg !72
  call void @llvm.dbg.value(metadata i32 %35, metadata !73, metadata !DIExpression()), !dbg !71
  store i32 %35, i32* %.i0001p_325, align 4, !dbg !72
  %36 = load i32, i32* %.de0002_366, align 4, !dbg !72
  store i32 %36, i32* %.i0002p_326, align 4, !dbg !72
  %37 = load i64, i64* %__nv_MAIN_F1L22_2Arg2, align 8, !dbg !72
  %38 = bitcast %astruct.dt157* %.uplevelArgPack0003_503 to i64*, !dbg !72
  store i64 %37, i64* %38, align 8, !dbg !72
  %39 = bitcast i32* %.i0001p_325 to i8*, !dbg !72
  %40 = bitcast %astruct.dt157* %.uplevelArgPack0003_503 to i8*, !dbg !72
  %41 = getelementptr i8, i8* %40, i64 8, !dbg !72
  %42 = bitcast i8* %41 to i8**, !dbg !72
  store i8* %39, i8** %42, align 8, !dbg !72
  %43 = bitcast i32* %.i0002p_326 to i8*, !dbg !72
  %44 = bitcast %astruct.dt157* %.uplevelArgPack0003_503 to i8*, !dbg !72
  %45 = getelementptr i8, i8* %44, i64 16, !dbg !72
  %46 = bitcast i8* %45 to i8**, !dbg !72
  store i8* %43, i8** %46, align 8, !dbg !72
  br label %L.LB4_510, !dbg !72

L.LB4_510:                                        ; preds = %L.LB4_534
  %47 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN_F1L24_3_ to i64*, !dbg !72
  %48 = bitcast %astruct.dt157* %.uplevelArgPack0003_503 to i64*, !dbg !72
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %47, i64* %48), !dbg !72
  br label %L.LB4_363

L.LB4_363:                                        ; preds = %L.LB4_510, %L.LB4_322
  %49 = load i32, i32* %__gtid___nv_MAIN_F1L22_2__484, align 4, !dbg !74
  call void @__kmpc_for_static_fini(i64* null, i32 %49), !dbg !74
  br label %L.LB4_340

L.LB4_340:                                        ; preds = %L.LB4_363
  br label %L.LB4_341

L.LB4_341:                                        ; preds = %L.LB4_340
  br label %L.LB4_342

L.LB4_342:                                        ; preds = %L.LB4_341
  ret void, !dbg !71
}

define internal void @__nv_MAIN_F1L24_3_(i32* %__nv_MAIN_F1L24_3Arg0, i64* %__nv_MAIN_F1L24_3Arg1, i64* %__nv_MAIN_F1L24_3Arg2) #0 !dbg !17 {
L.entry:
  %__gtid___nv_MAIN_F1L24_3__559 = alloca i32, align 4
  %.dY0003p_376 = alloca i64, align 8
  %"i$a_356" = alloca i64, align 8
  %var_331 = alloca [8 x i32], align 16
  %.i0004p_333 = alloca i32, align 4
  %i_332 = alloca i32, align 4
  %.du0004p_380 = alloca i32, align 4
  %.de0004p_381 = alloca i32, align 4
  %.di0004p_382 = alloca i32, align 4
  %.ds0004p_383 = alloca i32, align 4
  %.dl0004p_385 = alloca i32, align 4
  %.dl0004p.copy_553 = alloca i32, align 4
  %.de0004p.copy_554 = alloca i32, align 4
  %.ds0004p.copy_555 = alloca i32, align 4
  %.dX0004p_384 = alloca i32, align 4
  %.dY0004p_379 = alloca i32, align 4
  %.i0005p_337 = alloca i32, align 4
  %.dY0005p_391 = alloca i32, align 4
  %j_335 = alloca i32, align 4
  %.dY0006p_394 = alloca i64, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L24_3Arg0, metadata !75, metadata !DIExpression()), !dbg !76
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L24_3Arg1, metadata !77, metadata !DIExpression()), !dbg !76
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L24_3Arg2, metadata !78, metadata !DIExpression()), !dbg !76
  call void @llvm.dbg.value(metadata i32 8, metadata !79, metadata !DIExpression()), !dbg !76
  call void @llvm.dbg.value(metadata i32 1, metadata !80, metadata !DIExpression()), !dbg !76
  call void @llvm.dbg.value(metadata i32 0, metadata !81, metadata !DIExpression()), !dbg !76
  call void @llvm.dbg.value(metadata i32 1, metadata !82, metadata !DIExpression()), !dbg !76
  call void @llvm.dbg.value(metadata i32 0, metadata !83, metadata !DIExpression()), !dbg !76
  call void @llvm.dbg.value(metadata i32 1, metadata !84, metadata !DIExpression()), !dbg !76
  call void @llvm.dbg.value(metadata i32 8, metadata !85, metadata !DIExpression()), !dbg !76
  %0 = load i32, i32* %__nv_MAIN_F1L24_3Arg0, align 4, !dbg !86
  store i32 %0, i32* %__gtid___nv_MAIN_F1L24_3__559, align 4, !dbg !86
  br label %L.LB6_538

L.LB6_538:                                        ; preds = %L.entry
  br label %L.LB6_330

L.LB6_330:                                        ; preds = %L.LB6_538
  store i64 8, i64* %.dY0003p_376, align 8, !dbg !87
  call void @llvm.dbg.declare(metadata i64* %"i$a_356", metadata !88, metadata !DIExpression()), !dbg !76
  store i64 1, i64* %"i$a_356", align 8, !dbg !87
  br label %L.LB6_374

L.LB6_374:                                        ; preds = %L.LB6_374, %L.LB6_330
  %1 = load i64, i64* %"i$a_356", align 8, !dbg !87
  call void @llvm.dbg.value(metadata i64 %1, metadata !88, metadata !DIExpression()), !dbg !76
  call void @llvm.dbg.declare(metadata [8 x i32]* %var_331, metadata !89, metadata !DIExpression()), !dbg !86
  %2 = bitcast [8 x i32]* %var_331 to i8*, !dbg !87
  %3 = getelementptr i8, i8* %2, i64 -4, !dbg !87
  %4 = bitcast i8* %3 to i32*, !dbg !87
  %5 = getelementptr i32, i32* %4, i64 %1, !dbg !87
  store i32 0, i32* %5, align 4, !dbg !87
  %6 = load i64, i64* %"i$a_356", align 8, !dbg !87
  call void @llvm.dbg.value(metadata i64 %6, metadata !88, metadata !DIExpression()), !dbg !76
  %7 = add nsw i64 %6, 1, !dbg !87
  store i64 %7, i64* %"i$a_356", align 8, !dbg !87
  %8 = load i64, i64* %.dY0003p_376, align 8, !dbg !87
  %9 = sub nsw i64 %8, 1, !dbg !87
  store i64 %9, i64* %.dY0003p_376, align 8, !dbg !87
  %10 = load i64, i64* %.dY0003p_376, align 8, !dbg !87
  %11 = icmp sgt i64 %10, 0, !dbg !87
  br i1 %11, label %L.LB6_374, label %L.LB6_574, !dbg !87

L.LB6_574:                                        ; preds = %L.LB6_374
  store i32 0, i32* %.i0004p_333, align 4, !dbg !87
  %12 = bitcast i64* %__nv_MAIN_F1L24_3Arg2 to i8*, !dbg !87
  %13 = getelementptr i8, i8* %12, i64 8, !dbg !87
  %14 = bitcast i8* %13 to i32**, !dbg !87
  %15 = load i32*, i32** %14, align 8, !dbg !87
  %16 = load i32, i32* %15, align 4, !dbg !87
  call void @llvm.dbg.declare(metadata i32* %i_332, metadata !90, metadata !DIExpression()), !dbg !86
  store i32 %16, i32* %i_332, align 4, !dbg !87
  %17 = bitcast i64* %__nv_MAIN_F1L24_3Arg2 to i8*, !dbg !87
  %18 = getelementptr i8, i8* %17, i64 16, !dbg !87
  %19 = bitcast i8* %18 to i32**, !dbg !87
  %20 = load i32*, i32** %19, align 8, !dbg !87
  %21 = load i32, i32* %20, align 4, !dbg !87
  store i32 %21, i32* %.du0004p_380, align 4, !dbg !87
  %22 = bitcast i64* %__nv_MAIN_F1L24_3Arg2 to i8*, !dbg !87
  %23 = getelementptr i8, i8* %22, i64 16, !dbg !87
  %24 = bitcast i8* %23 to i32**, !dbg !87
  %25 = load i32*, i32** %24, align 8, !dbg !87
  %26 = load i32, i32* %25, align 4, !dbg !87
  store i32 %26, i32* %.de0004p_381, align 4, !dbg !87
  store i32 1, i32* %.di0004p_382, align 4, !dbg !87
  %27 = load i32, i32* %.di0004p_382, align 4, !dbg !87
  store i32 %27, i32* %.ds0004p_383, align 4, !dbg !87
  %28 = bitcast i64* %__nv_MAIN_F1L24_3Arg2 to i8*, !dbg !87
  %29 = getelementptr i8, i8* %28, i64 8, !dbg !87
  %30 = bitcast i8* %29 to i32**, !dbg !87
  %31 = load i32*, i32** %30, align 8, !dbg !87
  %32 = load i32, i32* %31, align 4, !dbg !87
  store i32 %32, i32* %.dl0004p_385, align 4, !dbg !87
  %33 = load i32, i32* %.dl0004p_385, align 4, !dbg !87
  store i32 %33, i32* %.dl0004p.copy_553, align 4, !dbg !87
  %34 = load i32, i32* %.de0004p_381, align 4, !dbg !87
  store i32 %34, i32* %.de0004p.copy_554, align 4, !dbg !87
  %35 = load i32, i32* %.ds0004p_383, align 4, !dbg !87
  store i32 %35, i32* %.ds0004p.copy_555, align 4, !dbg !87
  %36 = load i32, i32* %__gtid___nv_MAIN_F1L24_3__559, align 4, !dbg !87
  %37 = bitcast i32* %.i0004p_333 to i64*, !dbg !87
  %38 = bitcast i32* %.dl0004p.copy_553 to i64*, !dbg !87
  %39 = bitcast i32* %.de0004p.copy_554 to i64*, !dbg !87
  %40 = bitcast i32* %.ds0004p.copy_555 to i64*, !dbg !87
  %41 = load i32, i32* %.ds0004p.copy_555, align 4, !dbg !87
  call void @__kmpc_for_static_init_4(i64* null, i32 %36, i32 34, i64* %37, i64* %38, i64* %39, i64* %40, i32 %41, i32 1), !dbg !87
  %42 = load i32, i32* %.dl0004p.copy_553, align 4, !dbg !87
  store i32 %42, i32* %.dl0004p_385, align 4, !dbg !87
  %43 = load i32, i32* %.de0004p.copy_554, align 4, !dbg !87
  store i32 %43, i32* %.de0004p_381, align 4, !dbg !87
  %44 = load i32, i32* %.ds0004p.copy_555, align 4, !dbg !87
  store i32 %44, i32* %.ds0004p_383, align 4, !dbg !87
  %45 = load i32, i32* %.dl0004p_385, align 4, !dbg !87
  store i32 %45, i32* %i_332, align 4, !dbg !87
  %46 = load i32, i32* %i_332, align 4, !dbg !87
  call void @llvm.dbg.value(metadata i32 %46, metadata !90, metadata !DIExpression()), !dbg !86
  store i32 %46, i32* %.dX0004p_384, align 4, !dbg !87
  %47 = load i32, i32* %.dX0004p_384, align 4, !dbg !87
  %48 = load i32, i32* %.du0004p_380, align 4, !dbg !87
  %49 = icmp sgt i32 %47, %48, !dbg !87
  br i1 %49, label %L.LB6_378, label %L.LB6_575, !dbg !87

L.LB6_575:                                        ; preds = %L.LB6_574
  %50 = load i32, i32* %.dX0004p_384, align 4, !dbg !87
  store i32 %50, i32* %i_332, align 4, !dbg !87
  %51 = load i32, i32* %.di0004p_382, align 4, !dbg !87
  %52 = load i32, i32* %.de0004p_381, align 4, !dbg !87
  %53 = load i32, i32* %.dX0004p_384, align 4, !dbg !87
  %54 = sub nsw i32 %52, %53, !dbg !87
  %55 = add nsw i32 %51, %54, !dbg !87
  %56 = load i32, i32* %.di0004p_382, align 4, !dbg !87
  %57 = sdiv i32 %55, %56, !dbg !87
  store i32 %57, i32* %.dY0004p_379, align 4, !dbg !87
  %58 = load i32, i32* %.dY0004p_379, align 4, !dbg !87
  %59 = icmp sle i32 %58, 0, !dbg !87
  br i1 %59, label %L.LB6_388, label %L.LB6_387, !dbg !87

L.LB6_387:                                        ; preds = %L.LB6_338, %L.LB6_575
  br label %L.LB6_334

L.LB6_334:                                        ; preds = %L.LB6_387
  store i32 9, i32* %.i0005p_337, align 4, !dbg !91
  store i32 8, i32* %.dY0005p_391, align 4, !dbg !91
  call void @llvm.dbg.declare(metadata i32* %j_335, metadata !92, metadata !DIExpression()), !dbg !86
  store i32 1, i32* %j_335, align 4, !dbg !91
  br label %L.LB6_389

L.LB6_389:                                        ; preds = %L.LB6_389, %L.LB6_334
  %60 = load i32, i32* %j_335, align 4, !dbg !93
  call void @llvm.dbg.value(metadata i32 %60, metadata !92, metadata !DIExpression()), !dbg !86
  %61 = sext i32 %60 to i64, !dbg !93
  %62 = bitcast [8 x i32]* %var_331 to i8*, !dbg !93
  %63 = getelementptr i8, i8* %62, i64 -4, !dbg !93
  %64 = bitcast i8* %63 to i32*, !dbg !93
  %65 = getelementptr i32, i32* %64, i64 %61, !dbg !93
  %66 = load i32, i32* %65, align 4, !dbg !93
  %67 = add nsw i32 %66, 1, !dbg !93
  %68 = load i32, i32* %j_335, align 4, !dbg !93
  call void @llvm.dbg.value(metadata i32 %68, metadata !92, metadata !DIExpression()), !dbg !86
  %69 = sext i32 %68 to i64, !dbg !93
  %70 = bitcast [8 x i32]* %var_331 to i8*, !dbg !93
  %71 = getelementptr i8, i8* %70, i64 -4, !dbg !93
  %72 = bitcast i8* %71 to i32*, !dbg !93
  %73 = getelementptr i32, i32* %72, i64 %69, !dbg !93
  store i32 %67, i32* %73, align 4, !dbg !93
  %74 = load i32, i32* %j_335, align 4, !dbg !94
  call void @llvm.dbg.value(metadata i32 %74, metadata !92, metadata !DIExpression()), !dbg !86
  %75 = add nsw i32 %74, 1, !dbg !94
  store i32 %75, i32* %j_335, align 4, !dbg !94
  %76 = load i32, i32* %.dY0005p_391, align 4, !dbg !94
  %77 = sub nsw i32 %76, 1, !dbg !94
  store i32 %77, i32* %.dY0005p_391, align 4, !dbg !94
  %78 = load i32, i32* %.dY0005p_391, align 4, !dbg !94
  %79 = icmp sgt i32 %78, 0, !dbg !94
  br i1 %79, label %L.LB6_389, label %L.LB6_338, !dbg !94

L.LB6_338:                                        ; preds = %L.LB6_389
  %80 = load i32, i32* %.di0004p_382, align 4, !dbg !86
  %81 = load i32, i32* %i_332, align 4, !dbg !86
  call void @llvm.dbg.value(metadata i32 %81, metadata !90, metadata !DIExpression()), !dbg !86
  %82 = add nsw i32 %80, %81, !dbg !86
  store i32 %82, i32* %i_332, align 4, !dbg !86
  %83 = load i32, i32* %.dY0004p_379, align 4, !dbg !86
  %84 = sub nsw i32 %83, 1, !dbg !86
  store i32 %84, i32* %.dY0004p_379, align 4, !dbg !86
  %85 = load i32, i32* %.dY0004p_379, align 4, !dbg !86
  %86 = icmp sgt i32 %85, 0, !dbg !86
  br i1 %86, label %L.LB6_387, label %L.LB6_388, !dbg !86

L.LB6_388:                                        ; preds = %L.LB6_338, %L.LB6_575
  br label %L.LB6_378

L.LB6_378:                                        ; preds = %L.LB6_388, %L.LB6_574
  %87 = load i32, i32* %__gtid___nv_MAIN_F1L24_3__559, align 4, !dbg !86
  call void @__kmpc_for_static_fini(i64* null, i32 %87), !dbg !86
  %88 = call i32 (...) @_mp_bcs_nest_red(), !dbg !86
  %89 = call i32 (...) @_mp_bcs_nest_red(), !dbg !86
  store i64 8, i64* %.dY0006p_394, align 8, !dbg !86
  store i64 1, i64* %"i$a_356", align 8, !dbg !86
  br label %L.LB6_392

L.LB6_392:                                        ; preds = %L.LB6_392, %L.LB6_378
  %90 = load i64, i64* %"i$a_356", align 8, !dbg !86
  call void @llvm.dbg.value(metadata i64 %90, metadata !88, metadata !DIExpression()), !dbg !76
  %91 = bitcast %struct.BSS1* @.BSS1 to i8*, !dbg !86
  %92 = getelementptr i8, i8* %91, i64 -4, !dbg !86
  %93 = bitcast i8* %92 to i32*, !dbg !86
  %94 = getelementptr i32, i32* %93, i64 %90, !dbg !86
  %95 = load i32, i32* %94, align 4, !dbg !86
  %96 = load i64, i64* %"i$a_356", align 8, !dbg !86
  call void @llvm.dbg.value(metadata i64 %96, metadata !88, metadata !DIExpression()), !dbg !76
  %97 = bitcast [8 x i32]* %var_331 to i8*, !dbg !86
  %98 = getelementptr i8, i8* %97, i64 -4, !dbg !86
  %99 = bitcast i8* %98 to i32*, !dbg !86
  %100 = getelementptr i32, i32* %99, i64 %96, !dbg !86
  %101 = load i32, i32* %100, align 4, !dbg !86
  %102 = add nsw i32 %95, %101, !dbg !86
  %103 = load i64, i64* %"i$a_356", align 8, !dbg !86
  call void @llvm.dbg.value(metadata i64 %103, metadata !88, metadata !DIExpression()), !dbg !76
  %104 = bitcast %struct.BSS1* @.BSS1 to i8*, !dbg !86
  %105 = getelementptr i8, i8* %104, i64 -4, !dbg !86
  %106 = bitcast i8* %105 to i32*, !dbg !86
  %107 = getelementptr i32, i32* %106, i64 %103, !dbg !86
  store i32 %102, i32* %107, align 4, !dbg !86
  %108 = load i64, i64* %"i$a_356", align 8, !dbg !86
  call void @llvm.dbg.value(metadata i64 %108, metadata !88, metadata !DIExpression()), !dbg !76
  %109 = add nsw i64 %108, 1, !dbg !86
  store i64 %109, i64* %"i$a_356", align 8, !dbg !86
  %110 = load i64, i64* %.dY0006p_394, align 8, !dbg !86
  %111 = sub nsw i64 %110, 1, !dbg !86
  store i64 %111, i64* %.dY0006p_394, align 8, !dbg !86
  %112 = load i64, i64* %.dY0006p_394, align 8, !dbg !86
  %113 = icmp sgt i64 %112, 0, !dbg !86
  br i1 %113, label %L.LB6_392, label %L.LB6_576, !dbg !86

L.LB6_576:                                        ; preds = %L.LB6_392
  %114 = call i32 (...) @_mp_ecs_nest_red(), !dbg !86
  %115 = call i32 (...) @_mp_ecs_nest_red(), !dbg !86
  br label %L.LB6_339

L.LB6_339:                                        ; preds = %L.LB6_576
  ret void, !dbg !86
}

declare signext i32 @_mp_ecs_nest_red(...) #0

declare signext i32 @_mp_bcs_nest_red(...) #0

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

!llvm.module.flags = !{!23, !24}
!llvm.dbg.cu = !{!4}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "var", scope: !2, file: !3, type: !9, isLocal: true, isDefinition: true)
!2 = distinct !DISubprogram(name: "drb162_nolocksimd_orig_gpu_no", scope: !4, file: !3, line: 10, type: !21, scopeLine: 10, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !4)
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB162-nolocksimd-orig-gpu-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !6, imports: !5)
!5 = !{}
!6 = !{!0, !7, !13, !15}
!7 = !DIGlobalVariableExpression(var: !8, expr: !DIExpression())
!8 = distinct !DIGlobalVariable(name: "var", scope: !4, file: !3, type: !9, isLocal: true, isDefinition: true)
!9 = !DICompositeType(tag: DW_TAG_array_type, baseType: !10, size: 256, align: 32, elements: !11)
!10 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!11 = !{!12}
!12 = !DISubrange(count: 8, lowerBound: 1)
!13 = !DIGlobalVariableExpression(var: !14, expr: !DIExpression())
!14 = distinct !DIGlobalVariable(name: "var", scope: !4, file: !3, type: !9, isLocal: true, isDefinition: true)
!15 = !DIGlobalVariableExpression(var: !16, expr: !DIExpression())
!16 = distinct !DIGlobalVariable(name: "var", scope: !17, file: !3, type: !9, isLocal: true, isDefinition: true)
!17 = distinct !DISubprogram(name: "__nv_MAIN_F1L24_3", scope: !4, file: !3, line: 24, type: !18, scopeLine: 24, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !4)
!18 = !DISubroutineType(types: !19)
!19 = !{null, !10, !20, !20}
!20 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!21 = !DISubroutineType(cc: DW_CC_program, types: !22)
!22 = !{null}
!23 = !{i32 2, !"Dwarf Version", i32 4}
!24 = !{i32 2, !"Debug Info Version", i32 3}
!25 = !DILocalVariable(name: "omp_nest_lock_kind", scope: !2, file: !3, type: !10)
!26 = !DILocation(line: 0, scope: !2)
!27 = !DILocalVariable(name: "omp_sched_static", scope: !2, file: !3, type: !10)
!28 = !DILocalVariable(name: "omp_proc_bind_false", scope: !2, file: !3, type: !10)
!29 = !DILocalVariable(name: "omp_proc_bind_true", scope: !2, file: !3, type: !10)
!30 = !DILocalVariable(name: "omp_lock_hint_none", scope: !2, file: !3, type: !10)
!31 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !2, file: !3, type: !10)
!32 = !DILocalVariable(name: "omp_lock_hint_speculative", scope: !2, file: !3, type: !10)
!33 = !DILocation(line: 41, column: 1, scope: !2)
!34 = !DILocation(line: 10, column: 1, scope: !2)
!35 = !DILocation(line: 17, column: 1, scope: !2)
!36 = !DILocalVariable(name: "i", scope: !2, file: !3, type: !10)
!37 = !DILocation(line: 18, column: 1, scope: !2)
!38 = !DILocation(line: 19, column: 1, scope: !2)
!39 = !DILocation(line: 33, column: 1, scope: !2)
!40 = !DILocation(line: 35, column: 1, scope: !2)
!41 = !DILocation(line: 36, column: 1, scope: !2)
!42 = !DILocation(line: 37, column: 1, scope: !2)
!43 = !DILocalVariable(scope: !2, file: !3, type: !10, flags: DIFlagArtificial)
!44 = !DILocation(line: 39, column: 1, scope: !2)
!45 = distinct !DISubprogram(name: "__nv_MAIN__F1L21_1", scope: !4, file: !3, line: 21, type: !18, scopeLine: 21, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !4)
!46 = !DILocalVariable(name: "__nv_MAIN__F1L21_1Arg0", arg: 1, scope: !45, file: !3, type: !10)
!47 = !DILocation(line: 0, scope: !45)
!48 = !DILocalVariable(name: "__nv_MAIN__F1L21_1Arg1", arg: 2, scope: !45, file: !3, type: !20)
!49 = !DILocalVariable(name: "__nv_MAIN__F1L21_1Arg2", arg: 3, scope: !45, file: !3, type: !20)
!50 = !DILocalVariable(name: "omp_nest_lock_kind", scope: !45, file: !3, type: !10)
!51 = !DILocalVariable(name: "omp_sched_static", scope: !45, file: !3, type: !10)
!52 = !DILocalVariable(name: "omp_proc_bind_false", scope: !45, file: !3, type: !10)
!53 = !DILocalVariable(name: "omp_proc_bind_true", scope: !45, file: !3, type: !10)
!54 = !DILocalVariable(name: "omp_lock_hint_none", scope: !45, file: !3, type: !10)
!55 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !45, file: !3, type: !10)
!56 = !DILocalVariable(name: "omp_lock_hint_speculative", scope: !45, file: !3, type: !10)
!57 = !DILocation(line: 33, column: 1, scope: !45)
!58 = !DILocation(line: 22, column: 1, scope: !45)
!59 = distinct !DISubprogram(name: "__nv_MAIN_F1L22_2", scope: !4, file: !3, line: 22, type: !18, scopeLine: 22, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !4)
!60 = !DILocalVariable(name: "__nv_MAIN_F1L22_2Arg0", arg: 1, scope: !59, file: !3, type: !10)
!61 = !DILocation(line: 0, scope: !59)
!62 = !DILocalVariable(name: "__nv_MAIN_F1L22_2Arg1", arg: 2, scope: !59, file: !3, type: !20)
!63 = !DILocalVariable(name: "__nv_MAIN_F1L22_2Arg2", arg: 3, scope: !59, file: !3, type: !20)
!64 = !DILocalVariable(name: "omp_nest_lock_kind", scope: !59, file: !3, type: !10)
!65 = !DILocalVariable(name: "omp_sched_static", scope: !59, file: !3, type: !10)
!66 = !DILocalVariable(name: "omp_proc_bind_false", scope: !59, file: !3, type: !10)
!67 = !DILocalVariable(name: "omp_proc_bind_true", scope: !59, file: !3, type: !10)
!68 = !DILocalVariable(name: "omp_lock_hint_none", scope: !59, file: !3, type: !10)
!69 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !59, file: !3, type: !10)
!70 = !DILocalVariable(name: "omp_lock_hint_speculative", scope: !59, file: !3, type: !10)
!71 = !DILocation(line: 32, column: 1, scope: !59)
!72 = !DILocation(line: 24, column: 1, scope: !59)
!73 = !DILocalVariable(name: "i", scope: !59, file: !3, type: !10)
!74 = !DILocation(line: 30, column: 1, scope: !59)
!75 = !DILocalVariable(name: "__nv_MAIN_F1L24_3Arg0", arg: 1, scope: !17, file: !3, type: !10)
!76 = !DILocation(line: 0, scope: !17)
!77 = !DILocalVariable(name: "__nv_MAIN_F1L24_3Arg1", arg: 2, scope: !17, file: !3, type: !20)
!78 = !DILocalVariable(name: "__nv_MAIN_F1L24_3Arg2", arg: 3, scope: !17, file: !3, type: !20)
!79 = !DILocalVariable(name: "omp_nest_lock_kind", scope: !17, file: !3, type: !10)
!80 = !DILocalVariable(name: "omp_sched_static", scope: !17, file: !3, type: !10)
!81 = !DILocalVariable(name: "omp_proc_bind_false", scope: !17, file: !3, type: !10)
!82 = !DILocalVariable(name: "omp_proc_bind_true", scope: !17, file: !3, type: !10)
!83 = !DILocalVariable(name: "omp_lock_hint_none", scope: !17, file: !3, type: !10)
!84 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !17, file: !3, type: !10)
!85 = !DILocalVariable(name: "omp_lock_hint_speculative", scope: !17, file: !3, type: !10)
!86 = !DILocation(line: 30, column: 1, scope: !17)
!87 = !DILocation(line: 24, column: 1, scope: !17)
!88 = !DILocalVariable(scope: !17, file: !3, type: !20, flags: DIFlagArtificial)
!89 = !DILocalVariable(name: "var", scope: !17, file: !3, type: !9)
!90 = !DILocalVariable(name: "i", scope: !17, file: !3, type: !10)
!91 = !DILocation(line: 26, column: 1, scope: !17)
!92 = !DILocalVariable(name: "j", scope: !17, file: !3, type: !10)
!93 = !DILocation(line: 27, column: 1, scope: !17)
!94 = !DILocation(line: 28, column: 1, scope: !17)

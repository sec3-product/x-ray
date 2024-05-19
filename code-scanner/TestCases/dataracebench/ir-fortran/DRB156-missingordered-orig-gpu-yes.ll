; ModuleID = '/tmp/DRB156-missingordered-orig-gpu-yes-c20890.ll'
source_filename = "/tmp/DRB156-missingordered-orig-gpu-yes-c20890.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.BSS1 = type <{ [400 x i8] }>
%astruct.dt61 = type <{ i8* }>
%astruct.dt103 = type <{ [8 x i8] }>
%astruct.dt157 = type <{ [8 x i8], i8*, i8* }>

@.BSS1 = internal global %struct.BSS1 zeroinitializer, align 32, !dbg !0
@.C309_MAIN_ = internal constant i32 25
@.C312_MAIN_ = internal constant i64 100
@.C284_MAIN_ = internal constant i64 0
@.C341_MAIN_ = internal constant i32 6
@.C338_MAIN_ = internal constant [63 x i8] c"micro-benchmarks-fortran/DRB156-missingordered-orig-gpu-yes.f95"
@.C340_MAIN_ = internal constant i32 29
@.C300_MAIN_ = internal constant i32 2
@.C311_MAIN_ = internal constant i32 100
@.C285_MAIN_ = internal constant i32 1
@.C283_MAIN_ = internal constant i32 0
@.C285___nv_MAIN__F1L21_1 = internal constant i32 1
@.C311___nv_MAIN__F1L21_1 = internal constant i32 100
@.C300___nv_MAIN__F1L21_1 = internal constant i32 2
@.C283___nv_MAIN__F1L21_1 = internal constant i32 0
@.C285___nv_MAIN_F1L22_2 = internal constant i32 1
@.C311___nv_MAIN_F1L22_2 = internal constant i32 100
@.C300___nv_MAIN_F1L22_2 = internal constant i32 2
@.C283___nv_MAIN_F1L22_2 = internal constant i32 0
@.C285___nv_MAIN_F1L23_3 = internal constant i32 1
@.C283___nv_MAIN_F1L23_3 = internal constant i32 0

define void @MAIN_() #0 !dbg !2 {
L.entry:
  %__gtid_MAIN__395 = alloca i32, align 4
  %.dY0001_353 = alloca i32, align 4
  %i_313 = alloca i32, align 4
  %.uplevelArgPack0001_392 = alloca %astruct.dt61, align 8
  %z__io_343 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !25, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.value(metadata i32 2, metadata !27, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.value(metadata i32 0, metadata !28, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.value(metadata i32 1, metadata !29, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.value(metadata i32 2, metadata !30, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.value(metadata i32 0, metadata !31, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.value(metadata i32 1, metadata !32, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.value(metadata i32 2, metadata !33, metadata !DIExpression()), !dbg !26
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !34
  store i32 %0, i32* %__gtid_MAIN__395, align 4, !dbg !34
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !35
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !35
  call void (i8*, ...) %2(i8* %1), !dbg !35
  br label %L.LB1_379

L.LB1_379:                                        ; preds = %L.entry
  store i32 100, i32* %.dY0001_353, align 4, !dbg !36
  call void @llvm.dbg.declare(metadata i32* %i_313, metadata !37, metadata !DIExpression()), !dbg !26
  store i32 1, i32* %i_313, align 4, !dbg !36
  br label %L.LB1_351

L.LB1_351:                                        ; preds = %L.LB1_351, %L.LB1_379
  %3 = load i32, i32* %i_313, align 4, !dbg !38
  call void @llvm.dbg.value(metadata i32 %3, metadata !37, metadata !DIExpression()), !dbg !26
  %4 = sext i32 %3 to i64, !dbg !38
  %5 = bitcast %struct.BSS1* @.BSS1 to i8*, !dbg !38
  %6 = getelementptr i8, i8* %5, i64 -4, !dbg !38
  %7 = bitcast i8* %6 to i32*, !dbg !38
  %8 = getelementptr i32, i32* %7, i64 %4, !dbg !38
  store i32 1, i32* %8, align 4, !dbg !38
  %9 = load i32, i32* %i_313, align 4, !dbg !39
  call void @llvm.dbg.value(metadata i32 %9, metadata !37, metadata !DIExpression()), !dbg !26
  %10 = add nsw i32 %9, 1, !dbg !39
  store i32 %10, i32* %i_313, align 4, !dbg !39
  %11 = load i32, i32* %.dY0001_353, align 4, !dbg !39
  %12 = sub nsw i32 %11, 1, !dbg !39
  store i32 %12, i32* %.dY0001_353, align 4, !dbg !39
  %13 = load i32, i32* %.dY0001_353, align 4, !dbg !39
  %14 = icmp sgt i32 %13, 0, !dbg !39
  br i1 %14, label %L.LB1_351, label %L.LB1_413, !dbg !39

L.LB1_413:                                        ; preds = %L.LB1_351
  %15 = bitcast %astruct.dt61* %.uplevelArgPack0001_392 to i64*, !dbg !40
  call void @__nv_MAIN__F1L21_1_(i32* %__gtid_MAIN__395, i64* null, i64* %15), !dbg !40
  call void (...) @_mp_bcs_nest(), !dbg !41
  %16 = bitcast i32* @.C340_MAIN_ to i8*, !dbg !41
  %17 = bitcast [63 x i8]* @.C338_MAIN_ to i8*, !dbg !41
  %18 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !41
  call void (i8*, i8*, i64, ...) %18(i8* %16, i8* %17, i64 63), !dbg !41
  %19 = bitcast i32* @.C341_MAIN_ to i8*, !dbg !41
  %20 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !41
  %21 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !41
  %22 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !41
  %23 = call i32 (i8*, i8*, i8*, i8*, ...) %22(i8* %19, i8* null, i8* %20, i8* %21), !dbg !41
  call void @llvm.dbg.declare(metadata i32* %z__io_343, metadata !42, metadata !DIExpression()), !dbg !26
  store i32 %23, i32* %z__io_343, align 4, !dbg !41
  %24 = bitcast %struct.BSS1* @.BSS1 to i8*, !dbg !41
  %25 = getelementptr i8, i8* %24, i64 396, !dbg !41
  %26 = bitcast i8* %25 to i32*, !dbg !41
  %27 = load i32, i32* %26, align 4, !dbg !41
  %28 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !41
  %29 = call i32 (i32, i32, ...) %28(i32 %27, i32 25), !dbg !41
  store i32 %29, i32* %z__io_343, align 4, !dbg !41
  %30 = call i32 (...) @f90io_ldw_end(), !dbg !41
  store i32 %30, i32* %z__io_343, align 4, !dbg !41
  call void (...) @_mp_ecs_nest(), !dbg !41
  ret void, !dbg !34
}

define internal void @__nv_MAIN__F1L21_1_(i32* %__nv_MAIN__F1L21_1Arg0, i64* %__nv_MAIN__F1L21_1Arg1, i64* %__nv_MAIN__F1L21_1Arg2) #0 !dbg !43 {
L.entry:
  %.uplevelArgPack0002_422 = alloca %astruct.dt103, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L21_1Arg0, metadata !44, metadata !DIExpression()), !dbg !45
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L21_1Arg1, metadata !46, metadata !DIExpression()), !dbg !45
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L21_1Arg2, metadata !47, metadata !DIExpression()), !dbg !45
  call void @llvm.dbg.value(metadata i32 1, metadata !48, metadata !DIExpression()), !dbg !45
  call void @llvm.dbg.value(metadata i32 2, metadata !49, metadata !DIExpression()), !dbg !45
  call void @llvm.dbg.value(metadata i32 0, metadata !50, metadata !DIExpression()), !dbg !45
  call void @llvm.dbg.value(metadata i32 1, metadata !51, metadata !DIExpression()), !dbg !45
  call void @llvm.dbg.value(metadata i32 2, metadata !52, metadata !DIExpression()), !dbg !45
  call void @llvm.dbg.value(metadata i32 0, metadata !53, metadata !DIExpression()), !dbg !45
  call void @llvm.dbg.value(metadata i32 1, metadata !54, metadata !DIExpression()), !dbg !45
  call void @llvm.dbg.value(metadata i32 2, metadata !55, metadata !DIExpression()), !dbg !45
  br label %L.LB2_417

L.LB2_417:                                        ; preds = %L.entry
  br label %L.LB2_316

L.LB2_316:                                        ; preds = %L.LB2_417
  %0 = load i64, i64* %__nv_MAIN__F1L21_1Arg2, align 8, !dbg !56
  %1 = bitcast %astruct.dt103* %.uplevelArgPack0002_422 to i64*, !dbg !56
  store i64 %0, i64* %1, align 8, !dbg !56
  %2 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN_F1L22_2_ to i64*, !dbg !56
  %3 = bitcast %astruct.dt103* %.uplevelArgPack0002_422 to i64*, !dbg !56
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_teams(i64* null, i32 1, i64* %2, i64* %3), !dbg !56
  br label %L.LB2_336

L.LB2_336:                                        ; preds = %L.LB2_316
  ret void, !dbg !57
}

define internal void @__nv_MAIN_F1L22_2_(i32* %__nv_MAIN_F1L22_2Arg0, i64* %__nv_MAIN_F1L22_2Arg1, i64* %__nv_MAIN_F1L22_2Arg2) #0 !dbg !58 {
L.entry:
  %__gtid___nv_MAIN_F1L22_2__456 = alloca i32, align 4
  %.i0000p_323 = alloca i32, align 4
  %.i0001p_324 = alloca i32, align 4
  %.i0002p_325 = alloca i32, align 4
  %.i0003p_326 = alloca i32, align 4
  %i_322 = alloca i32, align 4
  %.du0002_357 = alloca i32, align 4
  %.de0002_358 = alloca i32, align 4
  %.di0002_359 = alloca i32, align 4
  %.ds0002_360 = alloca i32, align 4
  %.dl0002_362 = alloca i32, align 4
  %.dl0002.copy_450 = alloca i32, align 4
  %.de0002.copy_451 = alloca i32, align 4
  %.ds0002.copy_452 = alloca i32, align 4
  %.dX0002_361 = alloca i32, align 4
  %.dY0002_356 = alloca i32, align 4
  %.uplevelArgPack0003_475 = alloca %astruct.dt157, align 16
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L22_2Arg0, metadata !59, metadata !DIExpression()), !dbg !60
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L22_2Arg1, metadata !61, metadata !DIExpression()), !dbg !60
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L22_2Arg2, metadata !62, metadata !DIExpression()), !dbg !60
  call void @llvm.dbg.value(metadata i32 1, metadata !63, metadata !DIExpression()), !dbg !60
  call void @llvm.dbg.value(metadata i32 2, metadata !64, metadata !DIExpression()), !dbg !60
  call void @llvm.dbg.value(metadata i32 0, metadata !65, metadata !DIExpression()), !dbg !60
  call void @llvm.dbg.value(metadata i32 1, metadata !66, metadata !DIExpression()), !dbg !60
  call void @llvm.dbg.value(metadata i32 2, metadata !67, metadata !DIExpression()), !dbg !60
  call void @llvm.dbg.value(metadata i32 0, metadata !68, metadata !DIExpression()), !dbg !60
  call void @llvm.dbg.value(metadata i32 1, metadata !69, metadata !DIExpression()), !dbg !60
  call void @llvm.dbg.value(metadata i32 2, metadata !70, metadata !DIExpression()), !dbg !60
  %0 = load i32, i32* %__nv_MAIN_F1L22_2Arg0, align 4, !dbg !71
  store i32 %0, i32* %__gtid___nv_MAIN_F1L22_2__456, align 4, !dbg !71
  br label %L.LB4_439

L.LB4_439:                                        ; preds = %L.entry
  br label %L.LB4_319

L.LB4_319:                                        ; preds = %L.LB4_439
  br label %L.LB4_320

L.LB4_320:                                        ; preds = %L.LB4_319
  br label %L.LB4_321

L.LB4_321:                                        ; preds = %L.LB4_320
  store i32 0, i32* %.i0000p_323, align 4, !dbg !72
  store i32 2, i32* %.i0001p_324, align 4, !dbg !72
  store i32 100, i32* %.i0002p_325, align 4, !dbg !72
  store i32 1, i32* %.i0003p_326, align 4, !dbg !72
  %1 = load i32, i32* %.i0001p_324, align 4, !dbg !72
  call void @llvm.dbg.declare(metadata i32* %i_322, metadata !73, metadata !DIExpression()), !dbg !71
  store i32 %1, i32* %i_322, align 4, !dbg !72
  %2 = load i32, i32* %.i0002p_325, align 4, !dbg !72
  store i32 %2, i32* %.du0002_357, align 4, !dbg !72
  %3 = load i32, i32* %.i0002p_325, align 4, !dbg !72
  store i32 %3, i32* %.de0002_358, align 4, !dbg !72
  store i32 1, i32* %.di0002_359, align 4, !dbg !72
  %4 = load i32, i32* %.di0002_359, align 4, !dbg !72
  store i32 %4, i32* %.ds0002_360, align 4, !dbg !72
  %5 = load i32, i32* %.i0001p_324, align 4, !dbg !72
  store i32 %5, i32* %.dl0002_362, align 4, !dbg !72
  %6 = load i32, i32* %.dl0002_362, align 4, !dbg !72
  store i32 %6, i32* %.dl0002.copy_450, align 4, !dbg !72
  %7 = load i32, i32* %.de0002_358, align 4, !dbg !72
  store i32 %7, i32* %.de0002.copy_451, align 4, !dbg !72
  %8 = load i32, i32* %.ds0002_360, align 4, !dbg !72
  store i32 %8, i32* %.ds0002.copy_452, align 4, !dbg !72
  %9 = load i32, i32* %__gtid___nv_MAIN_F1L22_2__456, align 4, !dbg !72
  %10 = bitcast i32* %.i0000p_323 to i64*, !dbg !72
  %11 = bitcast i32* %.dl0002.copy_450 to i64*, !dbg !72
  %12 = bitcast i32* %.de0002.copy_451 to i64*, !dbg !72
  %13 = bitcast i32* %.ds0002.copy_452 to i64*, !dbg !72
  %14 = load i32, i32* %.ds0002.copy_452, align 4, !dbg !72
  call void @__kmpc_for_static_init_4(i64* null, i32 %9, i32 92, i64* %10, i64* %11, i64* %12, i64* %13, i32 %14, i32 1), !dbg !72
  %15 = load i32, i32* %.dl0002.copy_450, align 4, !dbg !72
  store i32 %15, i32* %.dl0002_362, align 4, !dbg !72
  %16 = load i32, i32* %.de0002.copy_451, align 4, !dbg !72
  store i32 %16, i32* %.de0002_358, align 4, !dbg !72
  %17 = load i32, i32* %.ds0002.copy_452, align 4, !dbg !72
  store i32 %17, i32* %.ds0002_360, align 4, !dbg !72
  %18 = load i32, i32* %.dl0002_362, align 4, !dbg !72
  store i32 %18, i32* %i_322, align 4, !dbg !72
  %19 = load i32, i32* %i_322, align 4, !dbg !72
  call void @llvm.dbg.value(metadata i32 %19, metadata !73, metadata !DIExpression()), !dbg !71
  store i32 %19, i32* %.dX0002_361, align 4, !dbg !72
  %20 = load i32, i32* %.dX0002_361, align 4, !dbg !72
  %21 = load i32, i32* %.du0002_357, align 4, !dbg !72
  %22 = icmp sgt i32 %20, %21, !dbg !72
  br i1 %22, label %L.LB4_355, label %L.LB4_506, !dbg !72

L.LB4_506:                                        ; preds = %L.LB4_321
  %23 = load i32, i32* %.du0002_357, align 4, !dbg !72
  %24 = load i32, i32* %.de0002_358, align 4, !dbg !72
  %25 = icmp slt i32 %23, %24, !dbg !72
  %26 = select i1 %25, i32 %23, i32 %24, !dbg !72
  store i32 %26, i32* %.de0002_358, align 4, !dbg !72
  %27 = load i32, i32* %.dX0002_361, align 4, !dbg !72
  store i32 %27, i32* %i_322, align 4, !dbg !72
  %28 = load i32, i32* %.di0002_359, align 4, !dbg !72
  %29 = load i32, i32* %.de0002_358, align 4, !dbg !72
  %30 = load i32, i32* %.dX0002_361, align 4, !dbg !72
  %31 = sub nsw i32 %29, %30, !dbg !72
  %32 = add nsw i32 %28, %31, !dbg !72
  %33 = load i32, i32* %.di0002_359, align 4, !dbg !72
  %34 = sdiv i32 %32, %33, !dbg !72
  store i32 %34, i32* %.dY0002_356, align 4, !dbg !72
  %35 = load i32, i32* %i_322, align 4, !dbg !72
  call void @llvm.dbg.value(metadata i32 %35, metadata !73, metadata !DIExpression()), !dbg !71
  store i32 %35, i32* %.i0001p_324, align 4, !dbg !72
  %36 = load i32, i32* %.de0002_358, align 4, !dbg !72
  store i32 %36, i32* %.i0002p_325, align 4, !dbg !72
  %37 = load i64, i64* %__nv_MAIN_F1L22_2Arg2, align 8, !dbg !72
  %38 = bitcast %astruct.dt157* %.uplevelArgPack0003_475 to i64*, !dbg !72
  store i64 %37, i64* %38, align 8, !dbg !72
  %39 = bitcast i32* %.i0001p_324 to i8*, !dbg !72
  %40 = bitcast %astruct.dt157* %.uplevelArgPack0003_475 to i8*, !dbg !72
  %41 = getelementptr i8, i8* %40, i64 8, !dbg !72
  %42 = bitcast i8* %41 to i8**, !dbg !72
  store i8* %39, i8** %42, align 8, !dbg !72
  %43 = bitcast i32* %.i0002p_325 to i8*, !dbg !72
  %44 = bitcast %astruct.dt157* %.uplevelArgPack0003_475 to i8*, !dbg !72
  %45 = getelementptr i8, i8* %44, i64 16, !dbg !72
  %46 = bitcast i8* %45 to i8**, !dbg !72
  store i8* %43, i8** %46, align 8, !dbg !72
  br label %L.LB4_482, !dbg !72

L.LB4_482:                                        ; preds = %L.LB4_506
  %47 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN_F1L23_3_ to i64*, !dbg !72
  %48 = bitcast %astruct.dt157* %.uplevelArgPack0003_475 to i64*, !dbg !72
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %47, i64* %48), !dbg !72
  br label %L.LB4_355

L.LB4_355:                                        ; preds = %L.LB4_482, %L.LB4_321
  %49 = load i32, i32* %__gtid___nv_MAIN_F1L22_2__456, align 4, !dbg !74
  call void @__kmpc_for_static_fini(i64* null, i32 %49), !dbg !74
  br label %L.LB4_333

L.LB4_333:                                        ; preds = %L.LB4_355
  br label %L.LB4_334

L.LB4_334:                                        ; preds = %L.LB4_333
  br label %L.LB4_335

L.LB4_335:                                        ; preds = %L.LB4_334
  ret void, !dbg !71
}

define internal void @__nv_MAIN_F1L23_3_(i32* %__nv_MAIN_F1L23_3Arg0, i64* %__nv_MAIN_F1L23_3Arg1, i64* %__nv_MAIN_F1L23_3Arg2) #0 !dbg !17 {
L.entry:
  %__gtid___nv_MAIN_F1L23_3__527 = alloca i32, align 4
  %.i0004p_331 = alloca i32, align 4
  %i_330 = alloca i32, align 4
  %.du0003p_369 = alloca i32, align 4
  %.de0003p_370 = alloca i32, align 4
  %.di0003p_371 = alloca i32, align 4
  %.ds0003p_372 = alloca i32, align 4
  %.dl0003p_374 = alloca i32, align 4
  %.dl0003p.copy_521 = alloca i32, align 4
  %.de0003p.copy_522 = alloca i32, align 4
  %.ds0003p.copy_523 = alloca i32, align 4
  %.dX0003p_373 = alloca i32, align 4
  %.dY0003p_368 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L23_3Arg0, metadata !75, metadata !DIExpression()), !dbg !76
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L23_3Arg1, metadata !77, metadata !DIExpression()), !dbg !76
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L23_3Arg2, metadata !78, metadata !DIExpression()), !dbg !76
  call void @llvm.dbg.value(metadata i32 1, metadata !79, metadata !DIExpression()), !dbg !76
  call void @llvm.dbg.value(metadata i32 2, metadata !80, metadata !DIExpression()), !dbg !76
  call void @llvm.dbg.value(metadata i32 0, metadata !81, metadata !DIExpression()), !dbg !76
  call void @llvm.dbg.value(metadata i32 1, metadata !82, metadata !DIExpression()), !dbg !76
  call void @llvm.dbg.value(metadata i32 2, metadata !83, metadata !DIExpression()), !dbg !76
  call void @llvm.dbg.value(metadata i32 0, metadata !84, metadata !DIExpression()), !dbg !76
  call void @llvm.dbg.value(metadata i32 1, metadata !85, metadata !DIExpression()), !dbg !76
  call void @llvm.dbg.value(metadata i32 2, metadata !86, metadata !DIExpression()), !dbg !76
  %0 = load i32, i32* %__nv_MAIN_F1L23_3Arg0, align 4, !dbg !87
  store i32 %0, i32* %__gtid___nv_MAIN_F1L23_3__527, align 4, !dbg !87
  br label %L.LB6_510

L.LB6_510:                                        ; preds = %L.entry
  br label %L.LB6_329

L.LB6_329:                                        ; preds = %L.LB6_510
  store i32 0, i32* %.i0004p_331, align 4, !dbg !88
  %1 = bitcast i64* %__nv_MAIN_F1L23_3Arg2 to i8*, !dbg !88
  %2 = getelementptr i8, i8* %1, i64 8, !dbg !88
  %3 = bitcast i8* %2 to i32**, !dbg !88
  %4 = load i32*, i32** %3, align 8, !dbg !88
  %5 = load i32, i32* %4, align 4, !dbg !88
  call void @llvm.dbg.declare(metadata i32* %i_330, metadata !89, metadata !DIExpression()), !dbg !87
  store i32 %5, i32* %i_330, align 4, !dbg !88
  %6 = bitcast i64* %__nv_MAIN_F1L23_3Arg2 to i8*, !dbg !88
  %7 = getelementptr i8, i8* %6, i64 16, !dbg !88
  %8 = bitcast i8* %7 to i32**, !dbg !88
  %9 = load i32*, i32** %8, align 8, !dbg !88
  %10 = load i32, i32* %9, align 4, !dbg !88
  store i32 %10, i32* %.du0003p_369, align 4, !dbg !88
  %11 = bitcast i64* %__nv_MAIN_F1L23_3Arg2 to i8*, !dbg !88
  %12 = getelementptr i8, i8* %11, i64 16, !dbg !88
  %13 = bitcast i8* %12 to i32**, !dbg !88
  %14 = load i32*, i32** %13, align 8, !dbg !88
  %15 = load i32, i32* %14, align 4, !dbg !88
  store i32 %15, i32* %.de0003p_370, align 4, !dbg !88
  store i32 1, i32* %.di0003p_371, align 4, !dbg !88
  %16 = load i32, i32* %.di0003p_371, align 4, !dbg !88
  store i32 %16, i32* %.ds0003p_372, align 4, !dbg !88
  %17 = bitcast i64* %__nv_MAIN_F1L23_3Arg2 to i8*, !dbg !88
  %18 = getelementptr i8, i8* %17, i64 8, !dbg !88
  %19 = bitcast i8* %18 to i32**, !dbg !88
  %20 = load i32*, i32** %19, align 8, !dbg !88
  %21 = load i32, i32* %20, align 4, !dbg !88
  store i32 %21, i32* %.dl0003p_374, align 4, !dbg !88
  %22 = load i32, i32* %.dl0003p_374, align 4, !dbg !88
  store i32 %22, i32* %.dl0003p.copy_521, align 4, !dbg !88
  %23 = load i32, i32* %.de0003p_370, align 4, !dbg !88
  store i32 %23, i32* %.de0003p.copy_522, align 4, !dbg !88
  %24 = load i32, i32* %.ds0003p_372, align 4, !dbg !88
  store i32 %24, i32* %.ds0003p.copy_523, align 4, !dbg !88
  %25 = load i32, i32* %__gtid___nv_MAIN_F1L23_3__527, align 4, !dbg !88
  %26 = bitcast i32* %.i0004p_331 to i64*, !dbg !88
  %27 = bitcast i32* %.dl0003p.copy_521 to i64*, !dbg !88
  %28 = bitcast i32* %.de0003p.copy_522 to i64*, !dbg !88
  %29 = bitcast i32* %.ds0003p.copy_523 to i64*, !dbg !88
  %30 = load i32, i32* %.ds0003p.copy_523, align 4, !dbg !88
  call void @__kmpc_for_static_init_4(i64* null, i32 %25, i32 34, i64* %26, i64* %27, i64* %28, i64* %29, i32 %30, i32 1), !dbg !88
  %31 = load i32, i32* %.dl0003p.copy_521, align 4, !dbg !88
  store i32 %31, i32* %.dl0003p_374, align 4, !dbg !88
  %32 = load i32, i32* %.de0003p.copy_522, align 4, !dbg !88
  store i32 %32, i32* %.de0003p_370, align 4, !dbg !88
  %33 = load i32, i32* %.ds0003p.copy_523, align 4, !dbg !88
  store i32 %33, i32* %.ds0003p_372, align 4, !dbg !88
  %34 = load i32, i32* %.dl0003p_374, align 4, !dbg !88
  store i32 %34, i32* %i_330, align 4, !dbg !88
  %35 = load i32, i32* %i_330, align 4, !dbg !88
  call void @llvm.dbg.value(metadata i32 %35, metadata !89, metadata !DIExpression()), !dbg !87
  store i32 %35, i32* %.dX0003p_373, align 4, !dbg !88
  %36 = load i32, i32* %.dX0003p_373, align 4, !dbg !88
  %37 = load i32, i32* %.du0003p_369, align 4, !dbg !88
  %38 = icmp sgt i32 %36, %37, !dbg !88
  br i1 %38, label %L.LB6_367, label %L.LB6_541, !dbg !88

L.LB6_541:                                        ; preds = %L.LB6_329
  %39 = load i32, i32* %.dX0003p_373, align 4, !dbg !88
  store i32 %39, i32* %i_330, align 4, !dbg !88
  %40 = load i32, i32* %.di0003p_371, align 4, !dbg !88
  %41 = load i32, i32* %.de0003p_370, align 4, !dbg !88
  %42 = load i32, i32* %.dX0003p_373, align 4, !dbg !88
  %43 = sub nsw i32 %41, %42, !dbg !88
  %44 = add nsw i32 %40, %43, !dbg !88
  %45 = load i32, i32* %.di0003p_371, align 4, !dbg !88
  %46 = sdiv i32 %44, %45, !dbg !88
  store i32 %46, i32* %.dY0003p_368, align 4, !dbg !88
  %47 = load i32, i32* %.dY0003p_368, align 4, !dbg !88
  %48 = icmp sle i32 %47, 0, !dbg !88
  br i1 %48, label %L.LB6_377, label %L.LB6_376, !dbg !88

L.LB6_376:                                        ; preds = %L.LB6_376, %L.LB6_541
  %49 = load i32, i32* %i_330, align 4, !dbg !90
  call void @llvm.dbg.value(metadata i32 %49, metadata !89, metadata !DIExpression()), !dbg !87
  %50 = sext i32 %49 to i64, !dbg !90
  %51 = bitcast %struct.BSS1* @.BSS1 to i8*, !dbg !90
  %52 = getelementptr i8, i8* %51, i64 -8, !dbg !90
  %53 = bitcast i8* %52 to i32*, !dbg !90
  %54 = getelementptr i32, i32* %53, i64 %50, !dbg !90
  %55 = load i32, i32* %54, align 4, !dbg !90
  %56 = add nsw i32 %55, 1, !dbg !90
  %57 = load i32, i32* %i_330, align 4, !dbg !90
  call void @llvm.dbg.value(metadata i32 %57, metadata !89, metadata !DIExpression()), !dbg !87
  %58 = sext i32 %57 to i64, !dbg !90
  %59 = bitcast %struct.BSS1* @.BSS1 to i8*, !dbg !90
  %60 = getelementptr i8, i8* %59, i64 -4, !dbg !90
  %61 = bitcast i8* %60 to i32*, !dbg !90
  %62 = getelementptr i32, i32* %61, i64 %58, !dbg !90
  store i32 %56, i32* %62, align 4, !dbg !90
  %63 = load i32, i32* %.di0003p_371, align 4, !dbg !87
  %64 = load i32, i32* %i_330, align 4, !dbg !87
  call void @llvm.dbg.value(metadata i32 %64, metadata !89, metadata !DIExpression()), !dbg !87
  %65 = add nsw i32 %63, %64, !dbg !87
  store i32 %65, i32* %i_330, align 4, !dbg !87
  %66 = load i32, i32* %.dY0003p_368, align 4, !dbg !87
  %67 = sub nsw i32 %66, 1, !dbg !87
  store i32 %67, i32* %.dY0003p_368, align 4, !dbg !87
  %68 = load i32, i32* %.dY0003p_368, align 4, !dbg !87
  %69 = icmp sgt i32 %68, 0, !dbg !87
  br i1 %69, label %L.LB6_376, label %L.LB6_377, !dbg !87

L.LB6_377:                                        ; preds = %L.LB6_376, %L.LB6_541
  br label %L.LB6_367

L.LB6_367:                                        ; preds = %L.LB6_377, %L.LB6_329
  %70 = load i32, i32* %__gtid___nv_MAIN_F1L23_3__527, align 4, !dbg !87
  call void @__kmpc_for_static_fini(i64* null, i32 %70), !dbg !87
  br label %L.LB6_332

L.LB6_332:                                        ; preds = %L.LB6_367
  ret void, !dbg !87
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

!llvm.module.flags = !{!23, !24}
!llvm.dbg.cu = !{!4}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "var", scope: !2, file: !3, type: !9, isLocal: true, isDefinition: true)
!2 = distinct !DISubprogram(name: "drb156_missingordered_orig_gpu_yes", scope: !4, file: !3, line: 10, type: !21, scopeLine: 10, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !4)
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB156-missingordered-orig-gpu-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !6, imports: !5)
!5 = !{}
!6 = !{!0, !7, !13, !15}
!7 = !DIGlobalVariableExpression(var: !8, expr: !DIExpression())
!8 = distinct !DIGlobalVariable(name: "var", scope: !4, file: !3, type: !9, isLocal: true, isDefinition: true)
!9 = !DICompositeType(tag: DW_TAG_array_type, baseType: !10, size: 3200, align: 32, elements: !11)
!10 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!11 = !{!12}
!12 = !DISubrange(count: 100, lowerBound: 1)
!13 = !DIGlobalVariableExpression(var: !14, expr: !DIExpression())
!14 = distinct !DIGlobalVariable(name: "var", scope: !4, file: !3, type: !9, isLocal: true, isDefinition: true)
!15 = !DIGlobalVariableExpression(var: !16, expr: !DIExpression())
!16 = distinct !DIGlobalVariable(name: "var", scope: !17, file: !3, type: !9, isLocal: true, isDefinition: true)
!17 = distinct !DISubprogram(name: "__nv_MAIN_F1L23_3", scope: !4, file: !3, line: 23, type: !18, scopeLine: 23, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !4)
!18 = !DISubroutineType(types: !19)
!19 = !{null, !10, !20, !20}
!20 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!21 = !DISubroutineType(cc: DW_CC_program, types: !22)
!22 = !{null}
!23 = !{i32 2, !"Dwarf Version", i32 4}
!24 = !{i32 2, !"Debug Info Version", i32 3}
!25 = !DILocalVariable(name: "omp_sched_static", scope: !2, file: !3, type: !10)
!26 = !DILocation(line: 0, scope: !2)
!27 = !DILocalVariable(name: "omp_sched_dynamic", scope: !2, file: !3, type: !10)
!28 = !DILocalVariable(name: "omp_proc_bind_false", scope: !2, file: !3, type: !10)
!29 = !DILocalVariable(name: "omp_proc_bind_true", scope: !2, file: !3, type: !10)
!30 = !DILocalVariable(name: "omp_proc_bind_master", scope: !2, file: !3, type: !10)
!31 = !DILocalVariable(name: "omp_lock_hint_none", scope: !2, file: !3, type: !10)
!32 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !2, file: !3, type: !10)
!33 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !2, file: !3, type: !10)
!34 = !DILocation(line: 30, column: 1, scope: !2)
!35 = !DILocation(line: 10, column: 1, scope: !2)
!36 = !DILocation(line: 17, column: 1, scope: !2)
!37 = !DILocalVariable(name: "i", scope: !2, file: !3, type: !10)
!38 = !DILocation(line: 18, column: 1, scope: !2)
!39 = !DILocation(line: 19, column: 1, scope: !2)
!40 = !DILocation(line: 27, column: 1, scope: !2)
!41 = !DILocation(line: 29, column: 1, scope: !2)
!42 = !DILocalVariable(scope: !2, file: !3, type: !10, flags: DIFlagArtificial)
!43 = distinct !DISubprogram(name: "__nv_MAIN__F1L21_1", scope: !4, file: !3, line: 21, type: !18, scopeLine: 21, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !4)
!44 = !DILocalVariable(name: "__nv_MAIN__F1L21_1Arg0", arg: 1, scope: !43, file: !3, type: !10)
!45 = !DILocation(line: 0, scope: !43)
!46 = !DILocalVariable(name: "__nv_MAIN__F1L21_1Arg1", arg: 2, scope: !43, file: !3, type: !20)
!47 = !DILocalVariable(name: "__nv_MAIN__F1L21_1Arg2", arg: 3, scope: !43, file: !3, type: !20)
!48 = !DILocalVariable(name: "omp_sched_static", scope: !43, file: !3, type: !10)
!49 = !DILocalVariable(name: "omp_sched_dynamic", scope: !43, file: !3, type: !10)
!50 = !DILocalVariable(name: "omp_proc_bind_false", scope: !43, file: !3, type: !10)
!51 = !DILocalVariable(name: "omp_proc_bind_true", scope: !43, file: !3, type: !10)
!52 = !DILocalVariable(name: "omp_proc_bind_master", scope: !43, file: !3, type: !10)
!53 = !DILocalVariable(name: "omp_lock_hint_none", scope: !43, file: !3, type: !10)
!54 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !43, file: !3, type: !10)
!55 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !43, file: !3, type: !10)
!56 = !DILocation(line: 22, column: 1, scope: !43)
!57 = !DILocation(line: 27, column: 1, scope: !43)
!58 = distinct !DISubprogram(name: "__nv_MAIN_F1L22_2", scope: !4, file: !3, line: 22, type: !18, scopeLine: 22, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !4)
!59 = !DILocalVariable(name: "__nv_MAIN_F1L22_2Arg0", arg: 1, scope: !58, file: !3, type: !10)
!60 = !DILocation(line: 0, scope: !58)
!61 = !DILocalVariable(name: "__nv_MAIN_F1L22_2Arg1", arg: 2, scope: !58, file: !3, type: !20)
!62 = !DILocalVariable(name: "__nv_MAIN_F1L22_2Arg2", arg: 3, scope: !58, file: !3, type: !20)
!63 = !DILocalVariable(name: "omp_sched_static", scope: !58, file: !3, type: !10)
!64 = !DILocalVariable(name: "omp_sched_dynamic", scope: !58, file: !3, type: !10)
!65 = !DILocalVariable(name: "omp_proc_bind_false", scope: !58, file: !3, type: !10)
!66 = !DILocalVariable(name: "omp_proc_bind_true", scope: !58, file: !3, type: !10)
!67 = !DILocalVariable(name: "omp_proc_bind_master", scope: !58, file: !3, type: !10)
!68 = !DILocalVariable(name: "omp_lock_hint_none", scope: !58, file: !3, type: !10)
!69 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !58, file: !3, type: !10)
!70 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !58, file: !3, type: !10)
!71 = !DILocation(line: 26, column: 1, scope: !58)
!72 = !DILocation(line: 23, column: 1, scope: !58)
!73 = !DILocalVariable(name: "i", scope: !58, file: !3, type: !10)
!74 = !DILocation(line: 25, column: 1, scope: !58)
!75 = !DILocalVariable(name: "__nv_MAIN_F1L23_3Arg0", arg: 1, scope: !17, file: !3, type: !10)
!76 = !DILocation(line: 0, scope: !17)
!77 = !DILocalVariable(name: "__nv_MAIN_F1L23_3Arg1", arg: 2, scope: !17, file: !3, type: !20)
!78 = !DILocalVariable(name: "__nv_MAIN_F1L23_3Arg2", arg: 3, scope: !17, file: !3, type: !20)
!79 = !DILocalVariable(name: "omp_sched_static", scope: !17, file: !3, type: !10)
!80 = !DILocalVariable(name: "omp_sched_dynamic", scope: !17, file: !3, type: !10)
!81 = !DILocalVariable(name: "omp_proc_bind_false", scope: !17, file: !3, type: !10)
!82 = !DILocalVariable(name: "omp_proc_bind_true", scope: !17, file: !3, type: !10)
!83 = !DILocalVariable(name: "omp_proc_bind_master", scope: !17, file: !3, type: !10)
!84 = !DILocalVariable(name: "omp_lock_hint_none", scope: !17, file: !3, type: !10)
!85 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !17, file: !3, type: !10)
!86 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !17, file: !3, type: !10)
!87 = !DILocation(line: 25, column: 1, scope: !17)
!88 = !DILocation(line: 23, column: 1, scope: !17)
!89 = !DILocalVariable(name: "i", scope: !17, file: !3, type: !10)
!90 = !DILocation(line: 24, column: 1, scope: !17)

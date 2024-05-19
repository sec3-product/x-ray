; ModuleID = '/tmp/DRB157-missingorderedsimd-orig-gpu-yes-ade473.ll'
source_filename = "/tmp/DRB157-missingorderedsimd-orig-gpu-yes-ade473.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.BSS1 = type <{ [400 x i8] }>
%astruct.dt61 = type <{ i8* }>
%astruct.dt103 = type <{ [8 x i8] }>
%astruct.dt157 = type <{ [8 x i8], i8*, i8* }>

@.BSS1 = internal global %struct.BSS1 zeroinitializer, align 32, !dbg !0
@.C306_MAIN_ = internal constant i32 25
@.C342_MAIN_ = internal constant i64 98
@.C284_MAIN_ = internal constant i64 0
@.C339_MAIN_ = internal constant i32 6
@.C336_MAIN_ = internal constant [67 x i8] c"micro-benchmarks-fortran/DRB157-missingorderedsimd-orig-gpu-yes.f95"
@.C338_MAIN_ = internal constant i32 30
@.C305_MAIN_ = internal constant i32 16
@.C307_MAIN_ = internal constant i32 17
@.C309_MAIN_ = internal constant i32 100
@.C285_MAIN_ = internal constant i32 1
@.C283_MAIN_ = internal constant i32 0
@.C305___nv_MAIN__F1L22_1 = internal constant i32 16
@.C285___nv_MAIN__F1L22_1 = internal constant i32 1
@.C309___nv_MAIN__F1L22_1 = internal constant i32 100
@.C307___nv_MAIN__F1L22_1 = internal constant i32 17
@.C283___nv_MAIN__F1L22_1 = internal constant i32 0
@.C305___nv_MAIN_F1L23_2 = internal constant i32 16
@.C285___nv_MAIN_F1L23_2 = internal constant i32 1
@.C309___nv_MAIN_F1L23_2 = internal constant i32 100
@.C307___nv_MAIN_F1L23_2 = internal constant i32 17
@.C283___nv_MAIN_F1L23_2 = internal constant i32 0
@.C305___nv_MAIN_F1L24_3 = internal constant i32 16
@.C285___nv_MAIN_F1L24_3 = internal constant i32 1
@.C283___nv_MAIN_F1L24_3 = internal constant i32 0

define void @MAIN_() #0 !dbg !2 {
L.entry:
  %__gtid_MAIN__394 = alloca i32, align 4
  %.dY0001_352 = alloca i32, align 4
  %i_311 = alloca i32, align 4
  %.uplevelArgPack0001_391 = alloca %astruct.dt61, align 8
  %z__io_341 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !25, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.value(metadata i32 0, metadata !27, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.value(metadata i32 1, metadata !28, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.value(metadata i32 0, metadata !29, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.value(metadata i32 1, metadata !30, metadata !DIExpression()), !dbg !26
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !31
  store i32 %0, i32* %__gtid_MAIN__394, align 4, !dbg !31
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !32
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !32
  call void (i8*, ...) %2(i8* %1), !dbg !32
  br label %L.LB1_378

L.LB1_378:                                        ; preds = %L.entry
  store i32 100, i32* %.dY0001_352, align 4, !dbg !33
  call void @llvm.dbg.declare(metadata i32* %i_311, metadata !34, metadata !DIExpression()), !dbg !26
  store i32 1, i32* %i_311, align 4, !dbg !33
  br label %L.LB1_350

L.LB1_350:                                        ; preds = %L.LB1_350, %L.LB1_378
  %3 = load i32, i32* %i_311, align 4, !dbg !35
  call void @llvm.dbg.value(metadata i32 %3, metadata !34, metadata !DIExpression()), !dbg !26
  %4 = sext i32 %3 to i64, !dbg !35
  %5 = bitcast %struct.BSS1* @.BSS1 to i8*, !dbg !35
  %6 = getelementptr i8, i8* %5, i64 -4, !dbg !35
  %7 = bitcast i8* %6 to i32*, !dbg !35
  %8 = getelementptr i32, i32* %7, i64 %4, !dbg !35
  store i32 1, i32* %8, align 4, !dbg !35
  %9 = load i32, i32* %i_311, align 4, !dbg !36
  call void @llvm.dbg.value(metadata i32 %9, metadata !34, metadata !DIExpression()), !dbg !26
  %10 = add nsw i32 %9, 1, !dbg !36
  store i32 %10, i32* %i_311, align 4, !dbg !36
  %11 = load i32, i32* %.dY0001_352, align 4, !dbg !36
  %12 = sub nsw i32 %11, 1, !dbg !36
  store i32 %12, i32* %.dY0001_352, align 4, !dbg !36
  %13 = load i32, i32* %.dY0001_352, align 4, !dbg !36
  %14 = icmp sgt i32 %13, 0, !dbg !36
  br i1 %14, label %L.LB1_350, label %L.LB1_413, !dbg !36

L.LB1_413:                                        ; preds = %L.LB1_350
  %15 = bitcast %astruct.dt61* %.uplevelArgPack0001_391 to i64*, !dbg !37
  call void @__nv_MAIN__F1L22_1_(i32* %__gtid_MAIN__394, i64* null, i64* %15), !dbg !37
  call void (...) @_mp_bcs_nest(), !dbg !38
  %16 = bitcast i32* @.C338_MAIN_ to i8*, !dbg !38
  %17 = bitcast [67 x i8]* @.C336_MAIN_ to i8*, !dbg !38
  %18 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !38
  call void (i8*, i8*, i64, ...) %18(i8* %16, i8* %17, i64 67), !dbg !38
  %19 = bitcast i32* @.C339_MAIN_ to i8*, !dbg !38
  %20 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !38
  %21 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !38
  %22 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !38
  %23 = call i32 (i8*, i8*, i8*, i8*, ...) %22(i8* %19, i8* null, i8* %20, i8* %21), !dbg !38
  call void @llvm.dbg.declare(metadata i32* %z__io_341, metadata !39, metadata !DIExpression()), !dbg !26
  store i32 %23, i32* %z__io_341, align 4, !dbg !38
  %24 = bitcast %struct.BSS1* @.BSS1 to i8*, !dbg !38
  %25 = getelementptr i8, i8* %24, i64 388, !dbg !38
  %26 = bitcast i8* %25 to i32*, !dbg !38
  %27 = load i32, i32* %26, align 4, !dbg !38
  %28 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !38
  %29 = call i32 (i32, i32, ...) %28(i32 %27, i32 25), !dbg !38
  store i32 %29, i32* %z__io_341, align 4, !dbg !38
  %30 = call i32 (...) @f90io_ldw_end(), !dbg !38
  store i32 %30, i32* %z__io_341, align 4, !dbg !38
  call void (...) @_mp_ecs_nest(), !dbg !38
  ret void, !dbg !31
}

define internal void @__nv_MAIN__F1L22_1_(i32* %__nv_MAIN__F1L22_1Arg0, i64* %__nv_MAIN__F1L22_1Arg1, i64* %__nv_MAIN__F1L22_1Arg2) #0 !dbg !40 {
L.entry:
  %.uplevelArgPack0002_422 = alloca %astruct.dt103, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L22_1Arg0, metadata !41, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L22_1Arg1, metadata !43, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L22_1Arg2, metadata !44, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.value(metadata i32 1, metadata !45, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.value(metadata i32 0, metadata !46, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.value(metadata i32 1, metadata !47, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.value(metadata i32 0, metadata !48, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.value(metadata i32 1, metadata !49, metadata !DIExpression()), !dbg !42
  br label %L.LB2_417

L.LB2_417:                                        ; preds = %L.entry
  br label %L.LB2_314

L.LB2_314:                                        ; preds = %L.LB2_417
  %0 = load i64, i64* %__nv_MAIN__F1L22_1Arg2, align 8, !dbg !50
  %1 = bitcast %astruct.dt103* %.uplevelArgPack0002_422 to i64*, !dbg !50
  store i64 %0, i64* %1, align 8, !dbg !50
  %2 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN_F1L23_2_ to i64*, !dbg !50
  %3 = bitcast %astruct.dt103* %.uplevelArgPack0002_422 to i64*, !dbg !50
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_teams(i64* null, i32 1, i64* %2, i64* %3), !dbg !50
  br label %L.LB2_334

L.LB2_334:                                        ; preds = %L.LB2_314
  ret void, !dbg !51
}

define internal void @__nv_MAIN_F1L23_2_(i32* %__nv_MAIN_F1L23_2Arg0, i64* %__nv_MAIN_F1L23_2Arg1, i64* %__nv_MAIN_F1L23_2Arg2) #0 !dbg !52 {
L.entry:
  %__gtid___nv_MAIN_F1L23_2__456 = alloca i32, align 4
  %.i0000p_321 = alloca i32, align 4
  %.i0001p_322 = alloca i32, align 4
  %.i0002p_323 = alloca i32, align 4
  %.i0003p_324 = alloca i32, align 4
  %i_320 = alloca i32, align 4
  %.du0002_356 = alloca i32, align 4
  %.de0002_357 = alloca i32, align 4
  %.di0002_358 = alloca i32, align 4
  %.ds0002_359 = alloca i32, align 4
  %.dl0002_361 = alloca i32, align 4
  %.dl0002.copy_450 = alloca i32, align 4
  %.de0002.copy_451 = alloca i32, align 4
  %.ds0002.copy_452 = alloca i32, align 4
  %.dX0002_360 = alloca i32, align 4
  %.dY0002_355 = alloca i32, align 4
  %.uplevelArgPack0003_475 = alloca %astruct.dt157, align 16
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L23_2Arg0, metadata !53, metadata !DIExpression()), !dbg !54
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L23_2Arg1, metadata !55, metadata !DIExpression()), !dbg !54
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L23_2Arg2, metadata !56, metadata !DIExpression()), !dbg !54
  call void @llvm.dbg.value(metadata i32 1, metadata !57, metadata !DIExpression()), !dbg !54
  call void @llvm.dbg.value(metadata i32 0, metadata !58, metadata !DIExpression()), !dbg !54
  call void @llvm.dbg.value(metadata i32 1, metadata !59, metadata !DIExpression()), !dbg !54
  call void @llvm.dbg.value(metadata i32 0, metadata !60, metadata !DIExpression()), !dbg !54
  call void @llvm.dbg.value(metadata i32 1, metadata !61, metadata !DIExpression()), !dbg !54
  %0 = load i32, i32* %__nv_MAIN_F1L23_2Arg0, align 4, !dbg !62
  store i32 %0, i32* %__gtid___nv_MAIN_F1L23_2__456, align 4, !dbg !62
  br label %L.LB4_439

L.LB4_439:                                        ; preds = %L.entry
  br label %L.LB4_317

L.LB4_317:                                        ; preds = %L.LB4_439
  br label %L.LB4_318

L.LB4_318:                                        ; preds = %L.LB4_317
  br label %L.LB4_319

L.LB4_319:                                        ; preds = %L.LB4_318
  store i32 0, i32* %.i0000p_321, align 4, !dbg !63
  store i32 17, i32* %.i0001p_322, align 4, !dbg !63
  store i32 100, i32* %.i0002p_323, align 4, !dbg !63
  store i32 1, i32* %.i0003p_324, align 4, !dbg !63
  %1 = load i32, i32* %.i0001p_322, align 4, !dbg !63
  call void @llvm.dbg.declare(metadata i32* %i_320, metadata !64, metadata !DIExpression()), !dbg !62
  store i32 %1, i32* %i_320, align 4, !dbg !63
  %2 = load i32, i32* %.i0002p_323, align 4, !dbg !63
  store i32 %2, i32* %.du0002_356, align 4, !dbg !63
  %3 = load i32, i32* %.i0002p_323, align 4, !dbg !63
  store i32 %3, i32* %.de0002_357, align 4, !dbg !63
  store i32 1, i32* %.di0002_358, align 4, !dbg !63
  %4 = load i32, i32* %.di0002_358, align 4, !dbg !63
  store i32 %4, i32* %.ds0002_359, align 4, !dbg !63
  %5 = load i32, i32* %.i0001p_322, align 4, !dbg !63
  store i32 %5, i32* %.dl0002_361, align 4, !dbg !63
  %6 = load i32, i32* %.dl0002_361, align 4, !dbg !63
  store i32 %6, i32* %.dl0002.copy_450, align 4, !dbg !63
  %7 = load i32, i32* %.de0002_357, align 4, !dbg !63
  store i32 %7, i32* %.de0002.copy_451, align 4, !dbg !63
  %8 = load i32, i32* %.ds0002_359, align 4, !dbg !63
  store i32 %8, i32* %.ds0002.copy_452, align 4, !dbg !63
  %9 = load i32, i32* %__gtid___nv_MAIN_F1L23_2__456, align 4, !dbg !63
  %10 = bitcast i32* %.i0000p_321 to i64*, !dbg !63
  %11 = bitcast i32* %.dl0002.copy_450 to i64*, !dbg !63
  %12 = bitcast i32* %.de0002.copy_451 to i64*, !dbg !63
  %13 = bitcast i32* %.ds0002.copy_452 to i64*, !dbg !63
  %14 = load i32, i32* %.ds0002.copy_452, align 4, !dbg !63
  call void @__kmpc_for_static_init_4(i64* null, i32 %9, i32 92, i64* %10, i64* %11, i64* %12, i64* %13, i32 %14, i32 1), !dbg !63
  %15 = load i32, i32* %.dl0002.copy_450, align 4, !dbg !63
  store i32 %15, i32* %.dl0002_361, align 4, !dbg !63
  %16 = load i32, i32* %.de0002.copy_451, align 4, !dbg !63
  store i32 %16, i32* %.de0002_357, align 4, !dbg !63
  %17 = load i32, i32* %.ds0002.copy_452, align 4, !dbg !63
  store i32 %17, i32* %.ds0002_359, align 4, !dbg !63
  %18 = load i32, i32* %.dl0002_361, align 4, !dbg !63
  store i32 %18, i32* %i_320, align 4, !dbg !63
  %19 = load i32, i32* %i_320, align 4, !dbg !63
  call void @llvm.dbg.value(metadata i32 %19, metadata !64, metadata !DIExpression()), !dbg !62
  store i32 %19, i32* %.dX0002_360, align 4, !dbg !63
  %20 = load i32, i32* %.dX0002_360, align 4, !dbg !63
  %21 = load i32, i32* %.du0002_356, align 4, !dbg !63
  %22 = icmp sgt i32 %20, %21, !dbg !63
  br i1 %22, label %L.LB4_354, label %L.LB4_506, !dbg !63

L.LB4_506:                                        ; preds = %L.LB4_319
  %23 = load i32, i32* %.du0002_356, align 4, !dbg !63
  %24 = load i32, i32* %.de0002_357, align 4, !dbg !63
  %25 = icmp slt i32 %23, %24, !dbg !63
  %26 = select i1 %25, i32 %23, i32 %24, !dbg !63
  store i32 %26, i32* %.de0002_357, align 4, !dbg !63
  %27 = load i32, i32* %.dX0002_360, align 4, !dbg !63
  store i32 %27, i32* %i_320, align 4, !dbg !63
  %28 = load i32, i32* %.di0002_358, align 4, !dbg !63
  %29 = load i32, i32* %.de0002_357, align 4, !dbg !63
  %30 = load i32, i32* %.dX0002_360, align 4, !dbg !63
  %31 = sub nsw i32 %29, %30, !dbg !63
  %32 = add nsw i32 %28, %31, !dbg !63
  %33 = load i32, i32* %.di0002_358, align 4, !dbg !63
  %34 = sdiv i32 %32, %33, !dbg !63
  store i32 %34, i32* %.dY0002_355, align 4, !dbg !63
  %35 = load i32, i32* %i_320, align 4, !dbg !63
  call void @llvm.dbg.value(metadata i32 %35, metadata !64, metadata !DIExpression()), !dbg !62
  store i32 %35, i32* %.i0001p_322, align 4, !dbg !63
  %36 = load i32, i32* %.de0002_357, align 4, !dbg !63
  store i32 %36, i32* %.i0002p_323, align 4, !dbg !63
  %37 = load i64, i64* %__nv_MAIN_F1L23_2Arg2, align 8, !dbg !63
  %38 = bitcast %astruct.dt157* %.uplevelArgPack0003_475 to i64*, !dbg !63
  store i64 %37, i64* %38, align 8, !dbg !63
  %39 = bitcast i32* %.i0001p_322 to i8*, !dbg !63
  %40 = bitcast %astruct.dt157* %.uplevelArgPack0003_475 to i8*, !dbg !63
  %41 = getelementptr i8, i8* %40, i64 8, !dbg !63
  %42 = bitcast i8* %41 to i8**, !dbg !63
  store i8* %39, i8** %42, align 8, !dbg !63
  %43 = bitcast i32* %.i0002p_323 to i8*, !dbg !63
  %44 = bitcast %astruct.dt157* %.uplevelArgPack0003_475 to i8*, !dbg !63
  %45 = getelementptr i8, i8* %44, i64 16, !dbg !63
  %46 = bitcast i8* %45 to i8**, !dbg !63
  store i8* %43, i8** %46, align 8, !dbg !63
  br label %L.LB4_482, !dbg !63

L.LB4_482:                                        ; preds = %L.LB4_506
  %47 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN_F1L24_3_ to i64*, !dbg !63
  %48 = bitcast %astruct.dt157* %.uplevelArgPack0003_475 to i64*, !dbg !63
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %47, i64* %48), !dbg !63
  br label %L.LB4_354

L.LB4_354:                                        ; preds = %L.LB4_482, %L.LB4_319
  %49 = load i32, i32* %__gtid___nv_MAIN_F1L23_2__456, align 4, !dbg !65
  call void @__kmpc_for_static_fini(i64* null, i32 %49), !dbg !65
  br label %L.LB4_331

L.LB4_331:                                        ; preds = %L.LB4_354
  br label %L.LB4_332

L.LB4_332:                                        ; preds = %L.LB4_331
  br label %L.LB4_333

L.LB4_333:                                        ; preds = %L.LB4_332
  ret void, !dbg !62
}

define internal void @__nv_MAIN_F1L24_3_(i32* %__nv_MAIN_F1L24_3Arg0, i64* %__nv_MAIN_F1L24_3Arg1, i64* %__nv_MAIN_F1L24_3Arg2) #0 !dbg !17 {
L.entry:
  %__gtid___nv_MAIN_F1L24_3__527 = alloca i32, align 4
  %.i0004p_329 = alloca i32, align 4
  %i_328 = alloca i32, align 4
  %.du0003p_368 = alloca i32, align 4
  %.de0003p_369 = alloca i32, align 4
  %.di0003p_370 = alloca i32, align 4
  %.ds0003p_371 = alloca i32, align 4
  %.dl0003p_373 = alloca i32, align 4
  %.dl0003p.copy_521 = alloca i32, align 4
  %.de0003p.copy_522 = alloca i32, align 4
  %.ds0003p.copy_523 = alloca i32, align 4
  %.dX0003p_372 = alloca i32, align 4
  %.dY0003p_367 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L24_3Arg0, metadata !66, metadata !DIExpression()), !dbg !67
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L24_3Arg1, metadata !68, metadata !DIExpression()), !dbg !67
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L24_3Arg2, metadata !69, metadata !DIExpression()), !dbg !67
  call void @llvm.dbg.value(metadata i32 1, metadata !70, metadata !DIExpression()), !dbg !67
  call void @llvm.dbg.value(metadata i32 0, metadata !71, metadata !DIExpression()), !dbg !67
  call void @llvm.dbg.value(metadata i32 1, metadata !72, metadata !DIExpression()), !dbg !67
  call void @llvm.dbg.value(metadata i32 0, metadata !73, metadata !DIExpression()), !dbg !67
  call void @llvm.dbg.value(metadata i32 1, metadata !74, metadata !DIExpression()), !dbg !67
  %0 = load i32, i32* %__nv_MAIN_F1L24_3Arg0, align 4, !dbg !75
  store i32 %0, i32* %__gtid___nv_MAIN_F1L24_3__527, align 4, !dbg !75
  br label %L.LB6_510

L.LB6_510:                                        ; preds = %L.entry
  br label %L.LB6_327

L.LB6_327:                                        ; preds = %L.LB6_510
  store i32 0, i32* %.i0004p_329, align 4, !dbg !76
  %1 = bitcast i64* %__nv_MAIN_F1L24_3Arg2 to i8*, !dbg !76
  %2 = getelementptr i8, i8* %1, i64 8, !dbg !76
  %3 = bitcast i8* %2 to i32**, !dbg !76
  %4 = load i32*, i32** %3, align 8, !dbg !76
  %5 = load i32, i32* %4, align 4, !dbg !76
  call void @llvm.dbg.declare(metadata i32* %i_328, metadata !77, metadata !DIExpression()), !dbg !75
  store i32 %5, i32* %i_328, align 4, !dbg !76
  %6 = bitcast i64* %__nv_MAIN_F1L24_3Arg2 to i8*, !dbg !76
  %7 = getelementptr i8, i8* %6, i64 16, !dbg !76
  %8 = bitcast i8* %7 to i32**, !dbg !76
  %9 = load i32*, i32** %8, align 8, !dbg !76
  %10 = load i32, i32* %9, align 4, !dbg !76
  store i32 %10, i32* %.du0003p_368, align 4, !dbg !76
  %11 = bitcast i64* %__nv_MAIN_F1L24_3Arg2 to i8*, !dbg !76
  %12 = getelementptr i8, i8* %11, i64 16, !dbg !76
  %13 = bitcast i8* %12 to i32**, !dbg !76
  %14 = load i32*, i32** %13, align 8, !dbg !76
  %15 = load i32, i32* %14, align 4, !dbg !76
  store i32 %15, i32* %.de0003p_369, align 4, !dbg !76
  store i32 1, i32* %.di0003p_370, align 4, !dbg !76
  %16 = load i32, i32* %.di0003p_370, align 4, !dbg !76
  store i32 %16, i32* %.ds0003p_371, align 4, !dbg !76
  %17 = bitcast i64* %__nv_MAIN_F1L24_3Arg2 to i8*, !dbg !76
  %18 = getelementptr i8, i8* %17, i64 8, !dbg !76
  %19 = bitcast i8* %18 to i32**, !dbg !76
  %20 = load i32*, i32** %19, align 8, !dbg !76
  %21 = load i32, i32* %20, align 4, !dbg !76
  store i32 %21, i32* %.dl0003p_373, align 4, !dbg !76
  %22 = load i32, i32* %.dl0003p_373, align 4, !dbg !76
  store i32 %22, i32* %.dl0003p.copy_521, align 4, !dbg !76
  %23 = load i32, i32* %.de0003p_369, align 4, !dbg !76
  store i32 %23, i32* %.de0003p.copy_522, align 4, !dbg !76
  %24 = load i32, i32* %.ds0003p_371, align 4, !dbg !76
  store i32 %24, i32* %.ds0003p.copy_523, align 4, !dbg !76
  %25 = load i32, i32* %__gtid___nv_MAIN_F1L24_3__527, align 4, !dbg !76
  %26 = bitcast i32* %.i0004p_329 to i64*, !dbg !76
  %27 = bitcast i32* %.dl0003p.copy_521 to i64*, !dbg !76
  %28 = bitcast i32* %.de0003p.copy_522 to i64*, !dbg !76
  %29 = bitcast i32* %.ds0003p.copy_523 to i64*, !dbg !76
  %30 = load i32, i32* %.ds0003p.copy_523, align 4, !dbg !76
  call void @__kmpc_for_static_init_4(i64* null, i32 %25, i32 34, i64* %26, i64* %27, i64* %28, i64* %29, i32 %30, i32 1), !dbg !76
  %31 = load i32, i32* %.dl0003p.copy_521, align 4, !dbg !76
  store i32 %31, i32* %.dl0003p_373, align 4, !dbg !76
  %32 = load i32, i32* %.de0003p.copy_522, align 4, !dbg !76
  store i32 %32, i32* %.de0003p_369, align 4, !dbg !76
  %33 = load i32, i32* %.ds0003p.copy_523, align 4, !dbg !76
  store i32 %33, i32* %.ds0003p_371, align 4, !dbg !76
  %34 = load i32, i32* %.dl0003p_373, align 4, !dbg !76
  store i32 %34, i32* %i_328, align 4, !dbg !76
  %35 = load i32, i32* %i_328, align 4, !dbg !76
  call void @llvm.dbg.value(metadata i32 %35, metadata !77, metadata !DIExpression()), !dbg !75
  store i32 %35, i32* %.dX0003p_372, align 4, !dbg !76
  %36 = load i32, i32* %.dX0003p_372, align 4, !dbg !76
  %37 = load i32, i32* %.du0003p_368, align 4, !dbg !76
  %38 = icmp sgt i32 %36, %37, !dbg !76
  br i1 %38, label %L.LB6_366, label %L.LB6_543, !dbg !76

L.LB6_543:                                        ; preds = %L.LB6_327
  %39 = load i32, i32* %.dX0003p_372, align 4, !dbg !76
  store i32 %39, i32* %i_328, align 4, !dbg !76
  %40 = load i32, i32* %.di0003p_370, align 4, !dbg !76
  %41 = load i32, i32* %.de0003p_369, align 4, !dbg !76
  %42 = load i32, i32* %.dX0003p_372, align 4, !dbg !76
  %43 = sub nsw i32 %41, %42, !dbg !76
  %44 = add nsw i32 %40, %43, !dbg !76
  %45 = load i32, i32* %.di0003p_370, align 4, !dbg !76
  %46 = sdiv i32 %44, %45, !dbg !76
  store i32 %46, i32* %.dY0003p_367, align 4, !dbg !76
  %47 = load i32, i32* %.dY0003p_367, align 4, !dbg !76
  %48 = icmp sle i32 %47, 0, !dbg !76
  br i1 %48, label %L.LB6_376, label %L.LB6_375, !dbg !76

L.LB6_375:                                        ; preds = %L.LB6_375, %L.LB6_543
  %49 = load i32, i32* %i_328, align 4, !dbg !78
  call void @llvm.dbg.value(metadata i32 %49, metadata !77, metadata !DIExpression()), !dbg !75
  %50 = sext i32 %49 to i64, !dbg !78
  %51 = bitcast %struct.BSS1* @.BSS1 to i8*, !dbg !78
  %52 = getelementptr i8, i8* %51, i64 -68, !dbg !78
  %53 = bitcast i8* %52 to i32*, !dbg !78
  %54 = getelementptr i32, i32* %53, i64 %50, !dbg !78
  %55 = load i32, i32* %54, align 4, !dbg !78
  %56 = add nsw i32 %55, 1, !dbg !78
  %57 = load i32, i32* %i_328, align 4, !dbg !78
  call void @llvm.dbg.value(metadata i32 %57, metadata !77, metadata !DIExpression()), !dbg !75
  %58 = sext i32 %57 to i64, !dbg !78
  %59 = bitcast %struct.BSS1* @.BSS1 to i8*, !dbg !78
  %60 = getelementptr i8, i8* %59, i64 -4, !dbg !78
  %61 = bitcast i8* %60 to i32*, !dbg !78
  %62 = getelementptr i32, i32* %61, i64 %58, !dbg !78
  store i32 %56, i32* %62, align 4, !dbg !78
  %63 = load i32, i32* %.di0003p_370, align 4, !dbg !75
  %64 = load i32, i32* %i_328, align 4, !dbg !75
  call void @llvm.dbg.value(metadata i32 %64, metadata !77, metadata !DIExpression()), !dbg !75
  %65 = add nsw i32 %63, %64, !dbg !75
  store i32 %65, i32* %i_328, align 4, !dbg !75
  %66 = load i32, i32* %.dY0003p_367, align 4, !dbg !75
  %67 = sub nsw i32 %66, 1, !dbg !75
  store i32 %67, i32* %.dY0003p_367, align 4, !dbg !75
  %68 = load i32, i32* %.dY0003p_367, align 4, !dbg !75
  %69 = icmp sgt i32 %68, 0, !dbg !75
  br i1 %69, label %L.LB6_375, label %L.LB6_376, !dbg !75

L.LB6_376:                                        ; preds = %L.LB6_375, %L.LB6_543
  br label %L.LB6_366

L.LB6_366:                                        ; preds = %L.LB6_376, %L.LB6_327
  %70 = load i32, i32* %__gtid___nv_MAIN_F1L24_3__527, align 4, !dbg !75
  call void @__kmpc_for_static_fini(i64* null, i32 %70), !dbg !75
  br label %L.LB6_330

L.LB6_330:                                        ; preds = %L.LB6_366
  ret void, !dbg !75
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
!2 = distinct !DISubprogram(name: "drb157_missingorderedsimd_orig_gpu_yes", scope: !4, file: !3, line: 11, type: !21, scopeLine: 11, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !4)
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB157-missingorderedsimd-orig-gpu-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
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
!17 = distinct !DISubprogram(name: "__nv_MAIN_F1L24_3", scope: !4, file: !3, line: 24, type: !18, scopeLine: 24, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !4)
!18 = !DISubroutineType(types: !19)
!19 = !{null, !10, !20, !20}
!20 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!21 = !DISubroutineType(cc: DW_CC_program, types: !22)
!22 = !{null}
!23 = !{i32 2, !"Dwarf Version", i32 4}
!24 = !{i32 2, !"Debug Info Version", i32 3}
!25 = !DILocalVariable(name: "omp_sched_static", scope: !2, file: !3, type: !10)
!26 = !DILocation(line: 0, scope: !2)
!27 = !DILocalVariable(name: "omp_proc_bind_false", scope: !2, file: !3, type: !10)
!28 = !DILocalVariable(name: "omp_proc_bind_true", scope: !2, file: !3, type: !10)
!29 = !DILocalVariable(name: "omp_lock_hint_none", scope: !2, file: !3, type: !10)
!30 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !2, file: !3, type: !10)
!31 = !DILocation(line: 31, column: 1, scope: !2)
!32 = !DILocation(line: 11, column: 1, scope: !2)
!33 = !DILocation(line: 18, column: 1, scope: !2)
!34 = !DILocalVariable(name: "i", scope: !2, file: !3, type: !10)
!35 = !DILocation(line: 19, column: 1, scope: !2)
!36 = !DILocation(line: 20, column: 1, scope: !2)
!37 = !DILocation(line: 28, column: 1, scope: !2)
!38 = !DILocation(line: 30, column: 1, scope: !2)
!39 = !DILocalVariable(scope: !2, file: !3, type: !10, flags: DIFlagArtificial)
!40 = distinct !DISubprogram(name: "__nv_MAIN__F1L22_1", scope: !4, file: !3, line: 22, type: !18, scopeLine: 22, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !4)
!41 = !DILocalVariable(name: "__nv_MAIN__F1L22_1Arg0", arg: 1, scope: !40, file: !3, type: !10)
!42 = !DILocation(line: 0, scope: !40)
!43 = !DILocalVariable(name: "__nv_MAIN__F1L22_1Arg1", arg: 2, scope: !40, file: !3, type: !20)
!44 = !DILocalVariable(name: "__nv_MAIN__F1L22_1Arg2", arg: 3, scope: !40, file: !3, type: !20)
!45 = !DILocalVariable(name: "omp_sched_static", scope: !40, file: !3, type: !10)
!46 = !DILocalVariable(name: "omp_proc_bind_false", scope: !40, file: !3, type: !10)
!47 = !DILocalVariable(name: "omp_proc_bind_true", scope: !40, file: !3, type: !10)
!48 = !DILocalVariable(name: "omp_lock_hint_none", scope: !40, file: !3, type: !10)
!49 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !40, file: !3, type: !10)
!50 = !DILocation(line: 23, column: 1, scope: !40)
!51 = !DILocation(line: 28, column: 1, scope: !40)
!52 = distinct !DISubprogram(name: "__nv_MAIN_F1L23_2", scope: !4, file: !3, line: 23, type: !18, scopeLine: 23, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !4)
!53 = !DILocalVariable(name: "__nv_MAIN_F1L23_2Arg0", arg: 1, scope: !52, file: !3, type: !10)
!54 = !DILocation(line: 0, scope: !52)
!55 = !DILocalVariable(name: "__nv_MAIN_F1L23_2Arg1", arg: 2, scope: !52, file: !3, type: !20)
!56 = !DILocalVariable(name: "__nv_MAIN_F1L23_2Arg2", arg: 3, scope: !52, file: !3, type: !20)
!57 = !DILocalVariable(name: "omp_sched_static", scope: !52, file: !3, type: !10)
!58 = !DILocalVariable(name: "omp_proc_bind_false", scope: !52, file: !3, type: !10)
!59 = !DILocalVariable(name: "omp_proc_bind_true", scope: !52, file: !3, type: !10)
!60 = !DILocalVariable(name: "omp_lock_hint_none", scope: !52, file: !3, type: !10)
!61 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !52, file: !3, type: !10)
!62 = !DILocation(line: 27, column: 1, scope: !52)
!63 = !DILocation(line: 24, column: 1, scope: !52)
!64 = !DILocalVariable(name: "i", scope: !52, file: !3, type: !10)
!65 = !DILocation(line: 26, column: 1, scope: !52)
!66 = !DILocalVariable(name: "__nv_MAIN_F1L24_3Arg0", arg: 1, scope: !17, file: !3, type: !10)
!67 = !DILocation(line: 0, scope: !17)
!68 = !DILocalVariable(name: "__nv_MAIN_F1L24_3Arg1", arg: 2, scope: !17, file: !3, type: !20)
!69 = !DILocalVariable(name: "__nv_MAIN_F1L24_3Arg2", arg: 3, scope: !17, file: !3, type: !20)
!70 = !DILocalVariable(name: "omp_sched_static", scope: !17, file: !3, type: !10)
!71 = !DILocalVariable(name: "omp_proc_bind_false", scope: !17, file: !3, type: !10)
!72 = !DILocalVariable(name: "omp_proc_bind_true", scope: !17, file: !3, type: !10)
!73 = !DILocalVariable(name: "omp_lock_hint_none", scope: !17, file: !3, type: !10)
!74 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !17, file: !3, type: !10)
!75 = !DILocation(line: 26, column: 1, scope: !17)
!76 = !DILocation(line: 24, column: 1, scope: !17)
!77 = !DILocalVariable(name: "i", scope: !17, file: !3, type: !10)
!78 = !DILocation(line: 25, column: 1, scope: !17)

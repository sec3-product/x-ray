; ModuleID = '/tmp/DRB161-nolocksimd-orig-gpu-yes-083abf.ll'
source_filename = "/tmp/DRB161-nolocksimd-orig-gpu-yes-083abf.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.BSS1 = type <{ [32 x i8] }>
%astruct.dt61 = type <{ i8* }>
%astruct.dt103 = type <{ [8 x i8] }>
%astruct.dt157 = type <{ [8 x i8], i8*, i8* }>

@.BSS1 = internal global %struct.BSS1 zeroinitializer, align 32, !dbg !0
@.C308_MAIN_ = internal constant i32 25
@.C311_MAIN_ = internal constant i64 8
@.C284_MAIN_ = internal constant i64 0
@.C347_MAIN_ = internal constant i32 6
@.C344_MAIN_ = internal constant [59 x i8] c"micro-benchmarks-fortran/DRB161-nolocksimd-orig-gpu-yes.f95"
@.C346_MAIN_ = internal constant i32 37
@.C335_MAIN_ = internal constant i32 9
@.C309_MAIN_ = internal constant i32 20
@.C317_MAIN_ = internal constant i32 1048
@.C300_MAIN_ = internal constant i32 8
@.C285_MAIN_ = internal constant i32 1
@.C283_MAIN_ = internal constant i32 0
@.C300___nv_MAIN__F1L23_1 = internal constant i32 8
@.C335___nv_MAIN__F1L23_1 = internal constant i32 9
@.C309___nv_MAIN__F1L23_1 = internal constant i32 20
@.C283___nv_MAIN__F1L23_1 = internal constant i32 0
@.C317___nv_MAIN__F1L23_1 = internal constant i32 1048
@.C285___nv_MAIN__F1L23_1 = internal constant i32 1
@.C300___nv_MAIN_F1L24_2 = internal constant i32 8
@.C335___nv_MAIN_F1L24_2 = internal constant i32 9
@.C309___nv_MAIN_F1L24_2 = internal constant i32 20
@.C285___nv_MAIN_F1L24_2 = internal constant i32 1
@.C283___nv_MAIN_F1L24_2 = internal constant i32 0
@.C300___nv_MAIN_F1L26_3 = internal constant i32 8
@.C335___nv_MAIN_F1L26_3 = internal constant i32 9
@.C285___nv_MAIN_F1L26_3 = internal constant i32 1
@.C283___nv_MAIN_F1L26_3 = internal constant i32 0

define void @MAIN_() #0 !dbg !2 {
L.entry:
  %__gtid_MAIN__404 = alloca i32, align 4
  %.dY0001_359 = alloca i32, align 4
  %i_312 = alloca i32, align 4
  %.uplevelArgPack0001_401 = alloca %astruct.dt61, align 8
  %z__io_349 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 8, metadata !25, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.value(metadata i32 1, metadata !27, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.value(metadata i32 0, metadata !28, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.value(metadata i32 1, metadata !29, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.value(metadata i32 0, metadata !30, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.value(metadata i32 1, metadata !31, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.value(metadata i32 8, metadata !32, metadata !DIExpression()), !dbg !26
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !33
  store i32 %0, i32* %__gtid_MAIN__404, align 4, !dbg !33
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !34
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !34
  call void (i8*, ...) %2(i8* %1), !dbg !34
  br label %L.LB1_388

L.LB1_388:                                        ; preds = %L.entry
  store i32 8, i32* %.dY0001_359, align 4, !dbg !35
  call void @llvm.dbg.declare(metadata i32* %i_312, metadata !36, metadata !DIExpression()), !dbg !26
  store i32 1, i32* %i_312, align 4, !dbg !35
  br label %L.LB1_357

L.LB1_357:                                        ; preds = %L.LB1_357, %L.LB1_388
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
  %11 = load i32, i32* %.dY0001_359, align 4, !dbg !38
  %12 = sub nsw i32 %11, 1, !dbg !38
  store i32 %12, i32* %.dY0001_359, align 4, !dbg !38
  %13 = load i32, i32* %.dY0001_359, align 4, !dbg !38
  %14 = icmp sgt i32 %13, 0, !dbg !38
  br i1 %14, label %L.LB1_357, label %L.LB1_422, !dbg !38

L.LB1_422:                                        ; preds = %L.LB1_357
  %15 = bitcast %astruct.dt61* %.uplevelArgPack0001_401 to i64*, !dbg !39
  call void @__nv_MAIN__F1L23_1_(i32* %__gtid_MAIN__404, i64* null, i64* %15), !dbg !39
  call void (...) @_mp_bcs_nest(), !dbg !40
  %16 = bitcast i32* @.C346_MAIN_ to i8*, !dbg !40
  %17 = bitcast [59 x i8]* @.C344_MAIN_ to i8*, !dbg !40
  %18 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !40
  call void (i8*, i8*, i64, ...) %18(i8* %16, i8* %17, i64 59), !dbg !40
  %19 = bitcast i32* @.C347_MAIN_ to i8*, !dbg !40
  %20 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !40
  %21 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !40
  %22 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !40
  %23 = call i32 (i8*, i8*, i8*, i8*, ...) %22(i8* %19, i8* null, i8* %20, i8* %21), !dbg !40
  call void @llvm.dbg.declare(metadata i32* %z__io_349, metadata !41, metadata !DIExpression()), !dbg !26
  store i32 %23, i32* %z__io_349, align 4, !dbg !40
  %24 = bitcast %struct.BSS1* @.BSS1 to i8*, !dbg !40
  %25 = getelementptr i8, i8* %24, i64 28, !dbg !40
  %26 = bitcast i8* %25 to i32*, !dbg !40
  %27 = load i32, i32* %26, align 4, !dbg !40
  %28 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !40
  %29 = call i32 (i32, i32, ...) %28(i32 %27, i32 25), !dbg !40
  store i32 %29, i32* %z__io_349, align 4, !dbg !40
  %30 = call i32 (...) @f90io_ldw_end(), !dbg !40
  store i32 %30, i32* %z__io_349, align 4, !dbg !40
  call void (...) @_mp_ecs_nest(), !dbg !40
  ret void, !dbg !33
}

define internal void @__nv_MAIN__F1L23_1_(i32* %__nv_MAIN__F1L23_1Arg0, i64* %__nv_MAIN__F1L23_1Arg1, i64* %__nv_MAIN__F1L23_1Arg2) #0 !dbg !42 {
L.entry:
  %__gtid___nv_MAIN__F1L23_1__435 = alloca i32, align 4
  %.uplevelArgPack0002_431 = alloca %astruct.dt103, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L23_1Arg0, metadata !43, metadata !DIExpression()), !dbg !44
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L23_1Arg1, metadata !45, metadata !DIExpression()), !dbg !44
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L23_1Arg2, metadata !46, metadata !DIExpression()), !dbg !44
  call void @llvm.dbg.value(metadata i32 8, metadata !47, metadata !DIExpression()), !dbg !44
  call void @llvm.dbg.value(metadata i32 1, metadata !48, metadata !DIExpression()), !dbg !44
  call void @llvm.dbg.value(metadata i32 0, metadata !49, metadata !DIExpression()), !dbg !44
  call void @llvm.dbg.value(metadata i32 1, metadata !50, metadata !DIExpression()), !dbg !44
  call void @llvm.dbg.value(metadata i32 0, metadata !51, metadata !DIExpression()), !dbg !44
  call void @llvm.dbg.value(metadata i32 1, metadata !52, metadata !DIExpression()), !dbg !44
  call void @llvm.dbg.value(metadata i32 8, metadata !53, metadata !DIExpression()), !dbg !44
  %0 = load i32, i32* %__nv_MAIN__F1L23_1Arg0, align 4, !dbg !54
  store i32 %0, i32* %__gtid___nv_MAIN__F1L23_1__435, align 4, !dbg !54
  br label %L.LB2_426

L.LB2_426:                                        ; preds = %L.entry
  br label %L.LB2_316

L.LB2_316:                                        ; preds = %L.LB2_426
  %1 = load i64, i64* %__nv_MAIN__F1L23_1Arg2, align 8, !dbg !55
  %2 = bitcast %astruct.dt103* %.uplevelArgPack0002_431 to i64*, !dbg !55
  store i64 %1, i64* %2, align 8, !dbg !55
  %3 = load i32, i32* %__gtid___nv_MAIN__F1L23_1__435, align 4, !dbg !55
  call void @__kmpc_push_num_teams(i64* null, i32 %3, i32 1, i32 1048), !dbg !55
  %4 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN_F1L24_2_ to i64*, !dbg !55
  %5 = bitcast %astruct.dt103* %.uplevelArgPack0002_431 to i64*, !dbg !55
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_teams(i64* null, i32 1, i64* %4, i64* %5), !dbg !55
  br label %L.LB2_342

L.LB2_342:                                        ; preds = %L.LB2_316
  ret void, !dbg !54
}

define internal void @__nv_MAIN_F1L24_2_(i32* %__nv_MAIN_F1L24_2Arg0, i64* %__nv_MAIN_F1L24_2Arg1, i64* %__nv_MAIN_F1L24_2Arg2) #0 !dbg !56 {
L.entry:
  %__gtid___nv_MAIN_F1L24_2__474 = alloca i32, align 4
  %.i0000p_324 = alloca i32, align 4
  %.i0001p_325 = alloca i32, align 4
  %.i0002p_326 = alloca i32, align 4
  %.i0003p_327 = alloca i32, align 4
  %i_323 = alloca i32, align 4
  %.du0002_363 = alloca i32, align 4
  %.de0002_364 = alloca i32, align 4
  %.di0002_365 = alloca i32, align 4
  %.ds0002_366 = alloca i32, align 4
  %.dl0002_368 = alloca i32, align 4
  %.dl0002.copy_468 = alloca i32, align 4
  %.de0002.copy_469 = alloca i32, align 4
  %.ds0002.copy_470 = alloca i32, align 4
  %.dX0002_367 = alloca i32, align 4
  %.dY0002_362 = alloca i32, align 4
  %.uplevelArgPack0003_493 = alloca %astruct.dt157, align 16
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L24_2Arg0, metadata !57, metadata !DIExpression()), !dbg !58
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L24_2Arg1, metadata !59, metadata !DIExpression()), !dbg !58
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L24_2Arg2, metadata !60, metadata !DIExpression()), !dbg !58
  call void @llvm.dbg.value(metadata i32 8, metadata !61, metadata !DIExpression()), !dbg !58
  call void @llvm.dbg.value(metadata i32 1, metadata !62, metadata !DIExpression()), !dbg !58
  call void @llvm.dbg.value(metadata i32 0, metadata !63, metadata !DIExpression()), !dbg !58
  call void @llvm.dbg.value(metadata i32 1, metadata !64, metadata !DIExpression()), !dbg !58
  call void @llvm.dbg.value(metadata i32 0, metadata !65, metadata !DIExpression()), !dbg !58
  call void @llvm.dbg.value(metadata i32 1, metadata !66, metadata !DIExpression()), !dbg !58
  call void @llvm.dbg.value(metadata i32 8, metadata !67, metadata !DIExpression()), !dbg !58
  %0 = load i32, i32* %__nv_MAIN_F1L24_2Arg0, align 4, !dbg !68
  store i32 %0, i32* %__gtid___nv_MAIN_F1L24_2__474, align 4, !dbg !68
  br label %L.LB4_457

L.LB4_457:                                        ; preds = %L.entry
  br label %L.LB4_320

L.LB4_320:                                        ; preds = %L.LB4_457
  br label %L.LB4_321

L.LB4_321:                                        ; preds = %L.LB4_320
  br label %L.LB4_322

L.LB4_322:                                        ; preds = %L.LB4_321
  store i32 0, i32* %.i0000p_324, align 4, !dbg !69
  store i32 1, i32* %.i0001p_325, align 4, !dbg !69
  store i32 20, i32* %.i0002p_326, align 4, !dbg !69
  store i32 1, i32* %.i0003p_327, align 4, !dbg !69
  %1 = load i32, i32* %.i0001p_325, align 4, !dbg !69
  call void @llvm.dbg.declare(metadata i32* %i_323, metadata !70, metadata !DIExpression()), !dbg !68
  store i32 %1, i32* %i_323, align 4, !dbg !69
  %2 = load i32, i32* %.i0002p_326, align 4, !dbg !69
  store i32 %2, i32* %.du0002_363, align 4, !dbg !69
  %3 = load i32, i32* %.i0002p_326, align 4, !dbg !69
  store i32 %3, i32* %.de0002_364, align 4, !dbg !69
  store i32 1, i32* %.di0002_365, align 4, !dbg !69
  %4 = load i32, i32* %.di0002_365, align 4, !dbg !69
  store i32 %4, i32* %.ds0002_366, align 4, !dbg !69
  %5 = load i32, i32* %.i0001p_325, align 4, !dbg !69
  store i32 %5, i32* %.dl0002_368, align 4, !dbg !69
  %6 = load i32, i32* %.dl0002_368, align 4, !dbg !69
  store i32 %6, i32* %.dl0002.copy_468, align 4, !dbg !69
  %7 = load i32, i32* %.de0002_364, align 4, !dbg !69
  store i32 %7, i32* %.de0002.copy_469, align 4, !dbg !69
  %8 = load i32, i32* %.ds0002_366, align 4, !dbg !69
  store i32 %8, i32* %.ds0002.copy_470, align 4, !dbg !69
  %9 = load i32, i32* %__gtid___nv_MAIN_F1L24_2__474, align 4, !dbg !69
  %10 = bitcast i32* %.i0000p_324 to i64*, !dbg !69
  %11 = bitcast i32* %.dl0002.copy_468 to i64*, !dbg !69
  %12 = bitcast i32* %.de0002.copy_469 to i64*, !dbg !69
  %13 = bitcast i32* %.ds0002.copy_470 to i64*, !dbg !69
  %14 = load i32, i32* %.ds0002.copy_470, align 4, !dbg !69
  call void @__kmpc_for_static_init_4(i64* null, i32 %9, i32 92, i64* %10, i64* %11, i64* %12, i64* %13, i32 %14, i32 1), !dbg !69
  %15 = load i32, i32* %.dl0002.copy_468, align 4, !dbg !69
  store i32 %15, i32* %.dl0002_368, align 4, !dbg !69
  %16 = load i32, i32* %.de0002.copy_469, align 4, !dbg !69
  store i32 %16, i32* %.de0002_364, align 4, !dbg !69
  %17 = load i32, i32* %.ds0002.copy_470, align 4, !dbg !69
  store i32 %17, i32* %.ds0002_366, align 4, !dbg !69
  %18 = load i32, i32* %.dl0002_368, align 4, !dbg !69
  store i32 %18, i32* %i_323, align 4, !dbg !69
  %19 = load i32, i32* %i_323, align 4, !dbg !69
  call void @llvm.dbg.value(metadata i32 %19, metadata !70, metadata !DIExpression()), !dbg !68
  store i32 %19, i32* %.dX0002_367, align 4, !dbg !69
  %20 = load i32, i32* %.dX0002_367, align 4, !dbg !69
  %21 = load i32, i32* %.du0002_363, align 4, !dbg !69
  %22 = icmp sgt i32 %20, %21, !dbg !69
  br i1 %22, label %L.LB4_361, label %L.LB4_524, !dbg !69

L.LB4_524:                                        ; preds = %L.LB4_322
  %23 = load i32, i32* %.du0002_363, align 4, !dbg !69
  %24 = load i32, i32* %.de0002_364, align 4, !dbg !69
  %25 = icmp slt i32 %23, %24, !dbg !69
  %26 = select i1 %25, i32 %23, i32 %24, !dbg !69
  store i32 %26, i32* %.de0002_364, align 4, !dbg !69
  %27 = load i32, i32* %.dX0002_367, align 4, !dbg !69
  store i32 %27, i32* %i_323, align 4, !dbg !69
  %28 = load i32, i32* %.di0002_365, align 4, !dbg !69
  %29 = load i32, i32* %.de0002_364, align 4, !dbg !69
  %30 = load i32, i32* %.dX0002_367, align 4, !dbg !69
  %31 = sub nsw i32 %29, %30, !dbg !69
  %32 = add nsw i32 %28, %31, !dbg !69
  %33 = load i32, i32* %.di0002_365, align 4, !dbg !69
  %34 = sdiv i32 %32, %33, !dbg !69
  store i32 %34, i32* %.dY0002_362, align 4, !dbg !69
  %35 = load i32, i32* %i_323, align 4, !dbg !69
  call void @llvm.dbg.value(metadata i32 %35, metadata !70, metadata !DIExpression()), !dbg !68
  store i32 %35, i32* %.i0001p_325, align 4, !dbg !69
  %36 = load i32, i32* %.de0002_364, align 4, !dbg !69
  store i32 %36, i32* %.i0002p_326, align 4, !dbg !69
  %37 = load i64, i64* %__nv_MAIN_F1L24_2Arg2, align 8, !dbg !69
  %38 = bitcast %astruct.dt157* %.uplevelArgPack0003_493 to i64*, !dbg !69
  store i64 %37, i64* %38, align 8, !dbg !69
  %39 = bitcast i32* %.i0001p_325 to i8*, !dbg !69
  %40 = bitcast %astruct.dt157* %.uplevelArgPack0003_493 to i8*, !dbg !69
  %41 = getelementptr i8, i8* %40, i64 8, !dbg !69
  %42 = bitcast i8* %41 to i8**, !dbg !69
  store i8* %39, i8** %42, align 8, !dbg !69
  %43 = bitcast i32* %.i0002p_326 to i8*, !dbg !69
  %44 = bitcast %astruct.dt157* %.uplevelArgPack0003_493 to i8*, !dbg !69
  %45 = getelementptr i8, i8* %44, i64 16, !dbg !69
  %46 = bitcast i8* %45 to i8**, !dbg !69
  store i8* %43, i8** %46, align 8, !dbg !69
  br label %L.LB4_500, !dbg !69

L.LB4_500:                                        ; preds = %L.LB4_524
  %47 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN_F1L26_3_ to i64*, !dbg !69
  %48 = bitcast %astruct.dt157* %.uplevelArgPack0003_493 to i64*, !dbg !69
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %47, i64* %48), !dbg !69
  br label %L.LB4_361

L.LB4_361:                                        ; preds = %L.LB4_500, %L.LB4_322
  %49 = load i32, i32* %__gtid___nv_MAIN_F1L24_2__474, align 4, !dbg !71
  call void @__kmpc_for_static_fini(i64* null, i32 %49), !dbg !71
  br label %L.LB4_339

L.LB4_339:                                        ; preds = %L.LB4_361
  br label %L.LB4_340

L.LB4_340:                                        ; preds = %L.LB4_339
  br label %L.LB4_341

L.LB4_341:                                        ; preds = %L.LB4_340
  ret void, !dbg !68
}

define internal void @__nv_MAIN_F1L26_3_(i32* %__nv_MAIN_F1L26_3Arg0, i64* %__nv_MAIN_F1L26_3Arg1, i64* %__nv_MAIN_F1L26_3Arg2) #0 !dbg !17 {
L.entry:
  %__gtid___nv_MAIN_F1L26_3__545 = alloca i32, align 4
  %.i0004p_332 = alloca i32, align 4
  %i_331 = alloca i32, align 4
  %.du0003p_375 = alloca i32, align 4
  %.de0003p_376 = alloca i32, align 4
  %.di0003p_377 = alloca i32, align 4
  %.ds0003p_378 = alloca i32, align 4
  %.dl0003p_380 = alloca i32, align 4
  %.dl0003p.copy_539 = alloca i32, align 4
  %.de0003p.copy_540 = alloca i32, align 4
  %.ds0003p.copy_541 = alloca i32, align 4
  %.dX0003p_379 = alloca i32, align 4
  %.dY0003p_374 = alloca i32, align 4
  %.i0005p_336 = alloca i32, align 4
  %.dY0004p_386 = alloca i32, align 4
  %j_334 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L26_3Arg0, metadata !72, metadata !DIExpression()), !dbg !73
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L26_3Arg1, metadata !74, metadata !DIExpression()), !dbg !73
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L26_3Arg2, metadata !75, metadata !DIExpression()), !dbg !73
  call void @llvm.dbg.value(metadata i32 8, metadata !76, metadata !DIExpression()), !dbg !73
  call void @llvm.dbg.value(metadata i32 1, metadata !77, metadata !DIExpression()), !dbg !73
  call void @llvm.dbg.value(metadata i32 0, metadata !78, metadata !DIExpression()), !dbg !73
  call void @llvm.dbg.value(metadata i32 1, metadata !79, metadata !DIExpression()), !dbg !73
  call void @llvm.dbg.value(metadata i32 0, metadata !80, metadata !DIExpression()), !dbg !73
  call void @llvm.dbg.value(metadata i32 1, metadata !81, metadata !DIExpression()), !dbg !73
  call void @llvm.dbg.value(metadata i32 8, metadata !82, metadata !DIExpression()), !dbg !73
  %0 = load i32, i32* %__nv_MAIN_F1L26_3Arg0, align 4, !dbg !83
  store i32 %0, i32* %__gtid___nv_MAIN_F1L26_3__545, align 4, !dbg !83
  br label %L.LB6_528

L.LB6_528:                                        ; preds = %L.entry
  br label %L.LB6_330

L.LB6_330:                                        ; preds = %L.LB6_528
  store i32 0, i32* %.i0004p_332, align 4, !dbg !84
  %1 = bitcast i64* %__nv_MAIN_F1L26_3Arg2 to i8*, !dbg !84
  %2 = getelementptr i8, i8* %1, i64 8, !dbg !84
  %3 = bitcast i8* %2 to i32**, !dbg !84
  %4 = load i32*, i32** %3, align 8, !dbg !84
  %5 = load i32, i32* %4, align 4, !dbg !84
  call void @llvm.dbg.declare(metadata i32* %i_331, metadata !85, metadata !DIExpression()), !dbg !83
  store i32 %5, i32* %i_331, align 4, !dbg !84
  %6 = bitcast i64* %__nv_MAIN_F1L26_3Arg2 to i8*, !dbg !84
  %7 = getelementptr i8, i8* %6, i64 16, !dbg !84
  %8 = bitcast i8* %7 to i32**, !dbg !84
  %9 = load i32*, i32** %8, align 8, !dbg !84
  %10 = load i32, i32* %9, align 4, !dbg !84
  store i32 %10, i32* %.du0003p_375, align 4, !dbg !84
  %11 = bitcast i64* %__nv_MAIN_F1L26_3Arg2 to i8*, !dbg !84
  %12 = getelementptr i8, i8* %11, i64 16, !dbg !84
  %13 = bitcast i8* %12 to i32**, !dbg !84
  %14 = load i32*, i32** %13, align 8, !dbg !84
  %15 = load i32, i32* %14, align 4, !dbg !84
  store i32 %15, i32* %.de0003p_376, align 4, !dbg !84
  store i32 1, i32* %.di0003p_377, align 4, !dbg !84
  %16 = load i32, i32* %.di0003p_377, align 4, !dbg !84
  store i32 %16, i32* %.ds0003p_378, align 4, !dbg !84
  %17 = bitcast i64* %__nv_MAIN_F1L26_3Arg2 to i8*, !dbg !84
  %18 = getelementptr i8, i8* %17, i64 8, !dbg !84
  %19 = bitcast i8* %18 to i32**, !dbg !84
  %20 = load i32*, i32** %19, align 8, !dbg !84
  %21 = load i32, i32* %20, align 4, !dbg !84
  store i32 %21, i32* %.dl0003p_380, align 4, !dbg !84
  %22 = load i32, i32* %.dl0003p_380, align 4, !dbg !84
  store i32 %22, i32* %.dl0003p.copy_539, align 4, !dbg !84
  %23 = load i32, i32* %.de0003p_376, align 4, !dbg !84
  store i32 %23, i32* %.de0003p.copy_540, align 4, !dbg !84
  %24 = load i32, i32* %.ds0003p_378, align 4, !dbg !84
  store i32 %24, i32* %.ds0003p.copy_541, align 4, !dbg !84
  %25 = load i32, i32* %__gtid___nv_MAIN_F1L26_3__545, align 4, !dbg !84
  %26 = bitcast i32* %.i0004p_332 to i64*, !dbg !84
  %27 = bitcast i32* %.dl0003p.copy_539 to i64*, !dbg !84
  %28 = bitcast i32* %.de0003p.copy_540 to i64*, !dbg !84
  %29 = bitcast i32* %.ds0003p.copy_541 to i64*, !dbg !84
  %30 = load i32, i32* %.ds0003p.copy_541, align 4, !dbg !84
  call void @__kmpc_for_static_init_4(i64* null, i32 %25, i32 34, i64* %26, i64* %27, i64* %28, i64* %29, i32 %30, i32 1), !dbg !84
  %31 = load i32, i32* %.dl0003p.copy_539, align 4, !dbg !84
  store i32 %31, i32* %.dl0003p_380, align 4, !dbg !84
  %32 = load i32, i32* %.de0003p.copy_540, align 4, !dbg !84
  store i32 %32, i32* %.de0003p_376, align 4, !dbg !84
  %33 = load i32, i32* %.ds0003p.copy_541, align 4, !dbg !84
  store i32 %33, i32* %.ds0003p_378, align 4, !dbg !84
  %34 = load i32, i32* %.dl0003p_380, align 4, !dbg !84
  store i32 %34, i32* %i_331, align 4, !dbg !84
  %35 = load i32, i32* %i_331, align 4, !dbg !84
  call void @llvm.dbg.value(metadata i32 %35, metadata !85, metadata !DIExpression()), !dbg !83
  store i32 %35, i32* %.dX0003p_379, align 4, !dbg !84
  %36 = load i32, i32* %.dX0003p_379, align 4, !dbg !84
  %37 = load i32, i32* %.du0003p_375, align 4, !dbg !84
  %38 = icmp sgt i32 %36, %37, !dbg !84
  br i1 %38, label %L.LB6_373, label %L.LB6_557, !dbg !84

L.LB6_557:                                        ; preds = %L.LB6_330
  %39 = load i32, i32* %.dX0003p_379, align 4, !dbg !84
  store i32 %39, i32* %i_331, align 4, !dbg !84
  %40 = load i32, i32* %.di0003p_377, align 4, !dbg !84
  %41 = load i32, i32* %.de0003p_376, align 4, !dbg !84
  %42 = load i32, i32* %.dX0003p_379, align 4, !dbg !84
  %43 = sub nsw i32 %41, %42, !dbg !84
  %44 = add nsw i32 %40, %43, !dbg !84
  %45 = load i32, i32* %.di0003p_377, align 4, !dbg !84
  %46 = sdiv i32 %44, %45, !dbg !84
  store i32 %46, i32* %.dY0003p_374, align 4, !dbg !84
  %47 = load i32, i32* %.dY0003p_374, align 4, !dbg !84
  %48 = icmp sle i32 %47, 0, !dbg !84
  br i1 %48, label %L.LB6_383, label %L.LB6_382, !dbg !84

L.LB6_382:                                        ; preds = %L.LB6_337, %L.LB6_557
  br label %L.LB6_333

L.LB6_333:                                        ; preds = %L.LB6_382
  store i32 9, i32* %.i0005p_336, align 4, !dbg !86
  store i32 8, i32* %.dY0004p_386, align 4, !dbg !86
  call void @llvm.dbg.declare(metadata i32* %j_334, metadata !87, metadata !DIExpression()), !dbg !83
  store i32 1, i32* %j_334, align 4, !dbg !86
  br label %L.LB6_384

L.LB6_384:                                        ; preds = %L.LB6_384, %L.LB6_333
  %49 = load i32, i32* %j_334, align 4, !dbg !88
  call void @llvm.dbg.value(metadata i32 %49, metadata !87, metadata !DIExpression()), !dbg !83
  %50 = sext i32 %49 to i64, !dbg !88
  %51 = bitcast %struct.BSS1* @.BSS1 to i8*, !dbg !88
  %52 = getelementptr i8, i8* %51, i64 -4, !dbg !88
  %53 = bitcast i8* %52 to i32*, !dbg !88
  %54 = getelementptr i32, i32* %53, i64 %50, !dbg !88
  %55 = load i32, i32* %54, align 4, !dbg !88
  %56 = add nsw i32 %55, 1, !dbg !88
  %57 = load i32, i32* %j_334, align 4, !dbg !88
  call void @llvm.dbg.value(metadata i32 %57, metadata !87, metadata !DIExpression()), !dbg !83
  %58 = sext i32 %57 to i64, !dbg !88
  %59 = bitcast %struct.BSS1* @.BSS1 to i8*, !dbg !88
  %60 = getelementptr i8, i8* %59, i64 -4, !dbg !88
  %61 = bitcast i8* %60 to i32*, !dbg !88
  %62 = getelementptr i32, i32* %61, i64 %58, !dbg !88
  store i32 %56, i32* %62, align 4, !dbg !88
  %63 = load i32, i32* %j_334, align 4, !dbg !89
  call void @llvm.dbg.value(metadata i32 %63, metadata !87, metadata !DIExpression()), !dbg !83
  %64 = add nsw i32 %63, 1, !dbg !89
  store i32 %64, i32* %j_334, align 4, !dbg !89
  %65 = load i32, i32* %.dY0004p_386, align 4, !dbg !89
  %66 = sub nsw i32 %65, 1, !dbg !89
  store i32 %66, i32* %.dY0004p_386, align 4, !dbg !89
  %67 = load i32, i32* %.dY0004p_386, align 4, !dbg !89
  %68 = icmp sgt i32 %67, 0, !dbg !89
  br i1 %68, label %L.LB6_384, label %L.LB6_337, !dbg !89

L.LB6_337:                                        ; preds = %L.LB6_384
  %69 = load i32, i32* %.di0003p_377, align 4, !dbg !83
  %70 = load i32, i32* %i_331, align 4, !dbg !83
  call void @llvm.dbg.value(metadata i32 %70, metadata !85, metadata !DIExpression()), !dbg !83
  %71 = add nsw i32 %69, %70, !dbg !83
  store i32 %71, i32* %i_331, align 4, !dbg !83
  %72 = load i32, i32* %.dY0003p_374, align 4, !dbg !83
  %73 = sub nsw i32 %72, 1, !dbg !83
  store i32 %73, i32* %.dY0003p_374, align 4, !dbg !83
  %74 = load i32, i32* %.dY0003p_374, align 4, !dbg !83
  %75 = icmp sgt i32 %74, 0, !dbg !83
  br i1 %75, label %L.LB6_382, label %L.LB6_383, !dbg !83

L.LB6_383:                                        ; preds = %L.LB6_337, %L.LB6_557
  br label %L.LB6_373

L.LB6_373:                                        ; preds = %L.LB6_383, %L.LB6_330
  %76 = load i32, i32* %__gtid___nv_MAIN_F1L26_3__545, align 4, !dbg !83
  call void @__kmpc_for_static_fini(i64* null, i32 %76), !dbg !83
  br label %L.LB6_338

L.LB6_338:                                        ; preds = %L.LB6_373
  ret void, !dbg !83
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

!llvm.module.flags = !{!23, !24}
!llvm.dbg.cu = !{!4}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "var", scope: !2, file: !3, type: !9, isLocal: true, isDefinition: true)
!2 = distinct !DISubprogram(name: "drb161_nolocksimd_orig_gpu_yes", scope: !4, file: !3, line: 12, type: !21, scopeLine: 12, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !4)
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB161-nolocksimd-orig-gpu-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
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
!17 = distinct !DISubprogram(name: "__nv_MAIN_F1L26_3", scope: !4, file: !3, line: 26, type: !18, scopeLine: 26, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !4)
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
!33 = !DILocation(line: 38, column: 1, scope: !2)
!34 = !DILocation(line: 12, column: 1, scope: !2)
!35 = !DILocation(line: 19, column: 1, scope: !2)
!36 = !DILocalVariable(name: "i", scope: !2, file: !3, type: !10)
!37 = !DILocation(line: 20, column: 1, scope: !2)
!38 = !DILocation(line: 21, column: 1, scope: !2)
!39 = !DILocation(line: 35, column: 1, scope: !2)
!40 = !DILocation(line: 37, column: 1, scope: !2)
!41 = !DILocalVariable(scope: !2, file: !3, type: !10, flags: DIFlagArtificial)
!42 = distinct !DISubprogram(name: "__nv_MAIN__F1L23_1", scope: !4, file: !3, line: 23, type: !18, scopeLine: 23, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !4)
!43 = !DILocalVariable(name: "__nv_MAIN__F1L23_1Arg0", arg: 1, scope: !42, file: !3, type: !10)
!44 = !DILocation(line: 0, scope: !42)
!45 = !DILocalVariable(name: "__nv_MAIN__F1L23_1Arg1", arg: 2, scope: !42, file: !3, type: !20)
!46 = !DILocalVariable(name: "__nv_MAIN__F1L23_1Arg2", arg: 3, scope: !42, file: !3, type: !20)
!47 = !DILocalVariable(name: "omp_nest_lock_kind", scope: !42, file: !3, type: !10)
!48 = !DILocalVariable(name: "omp_sched_static", scope: !42, file: !3, type: !10)
!49 = !DILocalVariable(name: "omp_proc_bind_false", scope: !42, file: !3, type: !10)
!50 = !DILocalVariable(name: "omp_proc_bind_true", scope: !42, file: !3, type: !10)
!51 = !DILocalVariable(name: "omp_lock_hint_none", scope: !42, file: !3, type: !10)
!52 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !42, file: !3, type: !10)
!53 = !DILocalVariable(name: "omp_lock_hint_speculative", scope: !42, file: !3, type: !10)
!54 = !DILocation(line: 35, column: 1, scope: !42)
!55 = !DILocation(line: 24, column: 1, scope: !42)
!56 = distinct !DISubprogram(name: "__nv_MAIN_F1L24_2", scope: !4, file: !3, line: 24, type: !18, scopeLine: 24, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !4)
!57 = !DILocalVariable(name: "__nv_MAIN_F1L24_2Arg0", arg: 1, scope: !56, file: !3, type: !10)
!58 = !DILocation(line: 0, scope: !56)
!59 = !DILocalVariable(name: "__nv_MAIN_F1L24_2Arg1", arg: 2, scope: !56, file: !3, type: !20)
!60 = !DILocalVariable(name: "__nv_MAIN_F1L24_2Arg2", arg: 3, scope: !56, file: !3, type: !20)
!61 = !DILocalVariable(name: "omp_nest_lock_kind", scope: !56, file: !3, type: !10)
!62 = !DILocalVariable(name: "omp_sched_static", scope: !56, file: !3, type: !10)
!63 = !DILocalVariable(name: "omp_proc_bind_false", scope: !56, file: !3, type: !10)
!64 = !DILocalVariable(name: "omp_proc_bind_true", scope: !56, file: !3, type: !10)
!65 = !DILocalVariable(name: "omp_lock_hint_none", scope: !56, file: !3, type: !10)
!66 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !56, file: !3, type: !10)
!67 = !DILocalVariable(name: "omp_lock_hint_speculative", scope: !56, file: !3, type: !10)
!68 = !DILocation(line: 34, column: 1, scope: !56)
!69 = !DILocation(line: 26, column: 1, scope: !56)
!70 = !DILocalVariable(name: "i", scope: !56, file: !3, type: !10)
!71 = !DILocation(line: 32, column: 1, scope: !56)
!72 = !DILocalVariable(name: "__nv_MAIN_F1L26_3Arg0", arg: 1, scope: !17, file: !3, type: !10)
!73 = !DILocation(line: 0, scope: !17)
!74 = !DILocalVariable(name: "__nv_MAIN_F1L26_3Arg1", arg: 2, scope: !17, file: !3, type: !20)
!75 = !DILocalVariable(name: "__nv_MAIN_F1L26_3Arg2", arg: 3, scope: !17, file: !3, type: !20)
!76 = !DILocalVariable(name: "omp_nest_lock_kind", scope: !17, file: !3, type: !10)
!77 = !DILocalVariable(name: "omp_sched_static", scope: !17, file: !3, type: !10)
!78 = !DILocalVariable(name: "omp_proc_bind_false", scope: !17, file: !3, type: !10)
!79 = !DILocalVariable(name: "omp_proc_bind_true", scope: !17, file: !3, type: !10)
!80 = !DILocalVariable(name: "omp_lock_hint_none", scope: !17, file: !3, type: !10)
!81 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !17, file: !3, type: !10)
!82 = !DILocalVariable(name: "omp_lock_hint_speculative", scope: !17, file: !3, type: !10)
!83 = !DILocation(line: 32, column: 1, scope: !17)
!84 = !DILocation(line: 26, column: 1, scope: !17)
!85 = !DILocalVariable(name: "i", scope: !17, file: !3, type: !10)
!86 = !DILocation(line: 28, column: 1, scope: !17)
!87 = !DILocalVariable(name: "j", scope: !17, file: !3, type: !10)
!88 = !DILocation(line: 29, column: 1, scope: !17)
!89 = !DILocation(line: 30, column: 1, scope: !17)

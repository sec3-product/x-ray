; ModuleID = '/tmp/DRB155-missingordered-orig-gpu-no-eb548d.ll'
source_filename = "/tmp/DRB155-missingordered-orig-gpu-no-eb548d.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.BSS1 = type <{ [400 x i8] }>
%astruct.dt69 = type <{ i8* }>
%astruct.dt111 = type <{ [8 x i8] }>

@.BSS1 = internal global %struct.BSS1 zeroinitializer, align 32, !dbg !0
@.C309_MAIN_ = internal constant i32 14
@.C331_MAIN_ = internal constant [17 x i8] c"Data Race Present"
@.C284_MAIN_ = internal constant i64 0
@.C328_MAIN_ = internal constant i32 6
@.C325_MAIN_ = internal constant [62 x i8] c"micro-benchmarks-fortran/DRB155-missingordered-orig-gpu-no.f95"
@.C327_MAIN_ = internal constant i32 34
@.C300_MAIN_ = internal constant i32 2
@.C311_MAIN_ = internal constant i32 100
@.C285_MAIN_ = internal constant i32 1
@.C283_MAIN_ = internal constant i32 0
@.C285___nv_MAIN__F1L22_1 = internal constant i32 1
@.C311___nv_MAIN__F1L22_1 = internal constant i32 100
@.C300___nv_MAIN__F1L22_1 = internal constant i32 2
@.C283___nv_MAIN__F1L22_1 = internal constant i32 0
@.C285___nv_MAIN_F1L23_2 = internal constant i32 1
@.C311___nv_MAIN_F1L23_2 = internal constant i32 100
@.C300___nv_MAIN_F1L23_2 = internal constant i32 2
@.C283___nv_MAIN_F1L23_2 = internal constant i32 0

define void @MAIN_() #0 !dbg !2 {
L.entry:
  %__gtid_MAIN__380 = alloca i32, align 4
  %.dY0001_341 = alloca i32, align 4
  %i_313 = alloca i32, align 4
  %.uplevelArgPack0001_377 = alloca %astruct.dt69, align 8
  %.dY0003_361 = alloca i32, align 4
  %z__io_330 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !23, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.value(metadata i32 2, metadata !25, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.value(metadata i32 0, metadata !26, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.value(metadata i32 1, metadata !27, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.value(metadata i32 2, metadata !28, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.value(metadata i32 0, metadata !29, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.value(metadata i32 1, metadata !30, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.value(metadata i32 2, metadata !31, metadata !DIExpression()), !dbg !24
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !32
  store i32 %0, i32* %__gtid_MAIN__380, align 4, !dbg !32
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !33
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !33
  call void (i8*, ...) %2(i8* %1), !dbg !33
  br label %L.LB1_364

L.LB1_364:                                        ; preds = %L.entry
  store i32 100, i32* %.dY0001_341, align 4, !dbg !34
  call void @llvm.dbg.declare(metadata i32* %i_313, metadata !35, metadata !DIExpression()), !dbg !24
  store i32 1, i32* %i_313, align 4, !dbg !34
  br label %L.LB1_339

L.LB1_339:                                        ; preds = %L.LB1_339, %L.LB1_364
  %3 = load i32, i32* %i_313, align 4, !dbg !36
  call void @llvm.dbg.value(metadata i32 %3, metadata !35, metadata !DIExpression()), !dbg !24
  %4 = sext i32 %3 to i64, !dbg !36
  %5 = bitcast %struct.BSS1* @.BSS1 to i8*, !dbg !36
  %6 = getelementptr i8, i8* %5, i64 -4, !dbg !36
  %7 = bitcast i8* %6 to i32*, !dbg !36
  %8 = getelementptr i32, i32* %7, i64 %4, !dbg !36
  store i32 1, i32* %8, align 4, !dbg !36
  %9 = load i32, i32* %i_313, align 4, !dbg !37
  call void @llvm.dbg.value(metadata i32 %9, metadata !35, metadata !DIExpression()), !dbg !24
  %10 = add nsw i32 %9, 1, !dbg !37
  store i32 %10, i32* %i_313, align 4, !dbg !37
  %11 = load i32, i32* %.dY0001_341, align 4, !dbg !37
  %12 = sub nsw i32 %11, 1, !dbg !37
  store i32 %12, i32* %.dY0001_341, align 4, !dbg !37
  %13 = load i32, i32* %.dY0001_341, align 4, !dbg !37
  %14 = icmp sgt i32 %13, 0, !dbg !37
  br i1 %14, label %L.LB1_339, label %L.LB1_396, !dbg !37

L.LB1_396:                                        ; preds = %L.LB1_339
  %15 = bitcast %astruct.dt69* %.uplevelArgPack0001_377 to i64*, !dbg !38
  call void @__nv_MAIN__F1L22_1_(i32* %__gtid_MAIN__380, i64* null, i64* %15), !dbg !38
  store i32 100, i32* %.dY0003_361, align 4, !dbg !39
  store i32 1, i32* %i_313, align 4, !dbg !39
  br label %L.LB1_359

L.LB1_359:                                        ; preds = %L.LB1_362, %L.LB1_396
  %16 = load i32, i32* %i_313, align 4, !dbg !40
  call void @llvm.dbg.value(metadata i32 %16, metadata !35, metadata !DIExpression()), !dbg !24
  %17 = sext i32 %16 to i64, !dbg !40
  %18 = bitcast %struct.BSS1* @.BSS1 to i8*, !dbg !40
  %19 = getelementptr i8, i8* %18, i64 -4, !dbg !40
  %20 = bitcast i8* %19 to i32*, !dbg !40
  %21 = getelementptr i32, i32* %20, i64 %17, !dbg !40
  %22 = load i32, i32* %21, align 4, !dbg !40
  %23 = load i32, i32* %i_313, align 4, !dbg !40
  call void @llvm.dbg.value(metadata i32 %23, metadata !35, metadata !DIExpression()), !dbg !24
  %24 = icmp eq i32 %22, %23, !dbg !40
  br i1 %24, label %L.LB1_362, label %L.LB1_397, !dbg !40

L.LB1_397:                                        ; preds = %L.LB1_359
  call void (...) @_mp_bcs_nest(), !dbg !41
  %25 = bitcast i32* @.C327_MAIN_ to i8*, !dbg !41
  %26 = bitcast [62 x i8]* @.C325_MAIN_ to i8*, !dbg !41
  %27 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !41
  call void (i8*, i8*, i64, ...) %27(i8* %25, i8* %26, i64 62), !dbg !41
  %28 = bitcast i32* @.C328_MAIN_ to i8*, !dbg !41
  %29 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !41
  %30 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !41
  %31 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !41
  %32 = call i32 (i8*, i8*, i8*, i8*, ...) %31(i8* %28, i8* null, i8* %29, i8* %30), !dbg !41
  call void @llvm.dbg.declare(metadata i32* %z__io_330, metadata !42, metadata !DIExpression()), !dbg !24
  store i32 %32, i32* %z__io_330, align 4, !dbg !41
  %33 = bitcast [17 x i8]* @.C331_MAIN_ to i8*, !dbg !41
  %34 = bitcast i32 (...)* @f90io_sc_ch_ldw to i32 (i8*, i32, i64, ...)*, !dbg !41
  %35 = call i32 (i8*, i32, i64, ...) %34(i8* %33, i32 14, i64 17), !dbg !41
  store i32 %35, i32* %z__io_330, align 4, !dbg !41
  %36 = call i32 (...) @f90io_ldw_end(), !dbg !41
  store i32 %36, i32* %z__io_330, align 4, !dbg !41
  call void (...) @_mp_ecs_nest(), !dbg !41
  br label %L.LB1_362

L.LB1_362:                                        ; preds = %L.LB1_397, %L.LB1_359
  %37 = load i32, i32* %i_313, align 4, !dbg !43
  call void @llvm.dbg.value(metadata i32 %37, metadata !35, metadata !DIExpression()), !dbg !24
  %38 = add nsw i32 %37, 1, !dbg !43
  store i32 %38, i32* %i_313, align 4, !dbg !43
  %39 = load i32, i32* %.dY0003_361, align 4, !dbg !43
  %40 = sub nsw i32 %39, 1, !dbg !43
  store i32 %40, i32* %.dY0003_361, align 4, !dbg !43
  %41 = load i32, i32* %.dY0003_361, align 4, !dbg !43
  %42 = icmp sgt i32 %41, 0, !dbg !43
  br i1 %42, label %L.LB1_359, label %L.LB1_398, !dbg !43

L.LB1_398:                                        ; preds = %L.LB1_362
  ret void, !dbg !32
}

define internal void @__nv_MAIN__F1L22_1_(i32* %__nv_MAIN__F1L22_1Arg0, i64* %__nv_MAIN__F1L22_1Arg1, i64* %__nv_MAIN__F1L22_1Arg2) #0 !dbg !44 {
L.entry:
  %__gtid___nv_MAIN__F1L22_1__412 = alloca i32, align 4
  %.uplevelArgPack0002_407 = alloca %astruct.dt111, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L22_1Arg0, metadata !45, metadata !DIExpression()), !dbg !46
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L22_1Arg1, metadata !47, metadata !DIExpression()), !dbg !46
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L22_1Arg2, metadata !48, metadata !DIExpression()), !dbg !46
  call void @llvm.dbg.value(metadata i32 1, metadata !49, metadata !DIExpression()), !dbg !46
  call void @llvm.dbg.value(metadata i32 2, metadata !50, metadata !DIExpression()), !dbg !46
  call void @llvm.dbg.value(metadata i32 0, metadata !51, metadata !DIExpression()), !dbg !46
  call void @llvm.dbg.value(metadata i32 1, metadata !52, metadata !DIExpression()), !dbg !46
  call void @llvm.dbg.value(metadata i32 2, metadata !53, metadata !DIExpression()), !dbg !46
  call void @llvm.dbg.value(metadata i32 0, metadata !54, metadata !DIExpression()), !dbg !46
  call void @llvm.dbg.value(metadata i32 1, metadata !55, metadata !DIExpression()), !dbg !46
  call void @llvm.dbg.value(metadata i32 2, metadata !56, metadata !DIExpression()), !dbg !46
  %0 = load i32, i32* %__nv_MAIN__F1L22_1Arg0, align 4, !dbg !57
  store i32 %0, i32* %__gtid___nv_MAIN__F1L22_1__412, align 4, !dbg !57
  br label %L.LB2_402

L.LB2_402:                                        ; preds = %L.entry
  br label %L.LB2_316

L.LB2_316:                                        ; preds = %L.LB2_402
  %1 = load i64, i64* %__nv_MAIN__F1L22_1Arg2, align 8, !dbg !58
  %2 = bitcast %astruct.dt111* %.uplevelArgPack0002_407 to i64*, !dbg !58
  store i64 %1, i64* %2, align 8, !dbg !58
  br label %L.LB2_410, !dbg !58

L.LB2_410:                                        ; preds = %L.LB2_316
  %3 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN_F1L23_2_ to i64*, !dbg !58
  %4 = bitcast %astruct.dt111* %.uplevelArgPack0002_407 to i64*, !dbg !58
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %3, i64* %4), !dbg !58
  br label %L.LB2_323

L.LB2_323:                                        ; preds = %L.LB2_410
  ret void, !dbg !57
}

define internal void @__nv_MAIN_F1L23_2_(i32* %__nv_MAIN_F1L23_2Arg0, i64* %__nv_MAIN_F1L23_2Arg1, i64* %__nv_MAIN_F1L23_2Arg2) #0 !dbg !15 {
L.entry:
  %__gtid___nv_MAIN_F1L23_2__454 = alloca i32, align 4
  %.i0000p_321 = alloca i32, align 4
  %i_320 = alloca i32, align 4
  %.dY0002p_344 = alloca i32, align 4
  %.du0002p_349 = alloca i32, align 4
  %.de0002p_350 = alloca i32, align 4
  %.di0002p_351 = alloca i32, align 4
  %.ds0002p_352 = alloca i32, align 4
  %.dx0002p_354 = alloca i32, align 4
  %.dl0002p_355 = alloca i32, align 4
  %.dU0002p_356 = alloca i32, align 4
  %.dl0002p.copy_448 = alloca i32, align 4
  %.dU0002p.copy_449 = alloca i32, align 4
  %.ds0002p.copy_450 = alloca i32, align 4
  %.dX0002p_353 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L23_2Arg0, metadata !59, metadata !DIExpression()), !dbg !60
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L23_2Arg1, metadata !61, metadata !DIExpression()), !dbg !60
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L23_2Arg2, metadata !62, metadata !DIExpression()), !dbg !60
  call void @llvm.dbg.value(metadata i32 1, metadata !63, metadata !DIExpression()), !dbg !60
  call void @llvm.dbg.value(metadata i32 2, metadata !64, metadata !DIExpression()), !dbg !60
  call void @llvm.dbg.value(metadata i32 0, metadata !65, metadata !DIExpression()), !dbg !60
  call void @llvm.dbg.value(metadata i32 1, metadata !66, metadata !DIExpression()), !dbg !60
  call void @llvm.dbg.value(metadata i32 2, metadata !67, metadata !DIExpression()), !dbg !60
  call void @llvm.dbg.value(metadata i32 0, metadata !68, metadata !DIExpression()), !dbg !60
  call void @llvm.dbg.value(metadata i32 1, metadata !69, metadata !DIExpression()), !dbg !60
  call void @llvm.dbg.value(metadata i32 2, metadata !70, metadata !DIExpression()), !dbg !60
  %0 = load i32, i32* %__nv_MAIN_F1L23_2Arg0, align 4, !dbg !71
  store i32 %0, i32* %__gtid___nv_MAIN_F1L23_2__454, align 4, !dbg !71
  br label %L.LB4_437

L.LB4_437:                                        ; preds = %L.entry
  br label %L.LB4_319

L.LB4_319:                                        ; preds = %L.LB4_437
  store i32 0, i32* %.i0000p_321, align 4, !dbg !72
  call void @llvm.dbg.declare(metadata i32* %i_320, metadata !73, metadata !DIExpression()), !dbg !71
  store i32 2, i32* %i_320, align 4, !dbg !72
  store i32 100, i32* %.dY0002p_344, align 4, !dbg !72
  store i32 2, i32* %i_320, align 4, !dbg !72
  store i32 100, i32* %.du0002p_349, align 4, !dbg !72
  store i32 100, i32* %.de0002p_350, align 4, !dbg !72
  store i32 1, i32* %.di0002p_351, align 4, !dbg !72
  %1 = load i32, i32* %.di0002p_351, align 4, !dbg !72
  store i32 %1, i32* %.ds0002p_352, align 4, !dbg !72
  store i32 2, i32* %.dx0002p_354, align 4, !dbg !72
  store i32 2, i32* %.dl0002p_355, align 4, !dbg !72
  store i32 100, i32* %.dU0002p_356, align 4, !dbg !72
  %2 = load i32, i32* %.dl0002p_355, align 4, !dbg !72
  store i32 %2, i32* %.dl0002p.copy_448, align 4, !dbg !72
  %3 = load i32, i32* %.dU0002p_356, align 4, !dbg !72
  store i32 %3, i32* %.dU0002p.copy_449, align 4, !dbg !72
  %4 = load i32, i32* %.ds0002p_352, align 4, !dbg !72
  store i32 %4, i32* %.ds0002p.copy_450, align 4, !dbg !72
  %5 = load i32, i32* %__gtid___nv_MAIN_F1L23_2__454, align 4, !dbg !72
  %6 = load i32, i32* %.dl0002p.copy_448, align 4, !dbg !72
  %7 = load i32, i32* %.dU0002p.copy_449, align 4, !dbg !72
  %8 = load i32, i32* %.ds0002p.copy_450, align 4, !dbg !72
  call void @__kmpc_dispatch_init_4(i64* null, i32 %5, i32 66, i32 %6, i32 %7, i32 %8, i32 0), !dbg !72
  %9 = load i32, i32* %.dl0002p.copy_448, align 4, !dbg !72
  store i32 %9, i32* %.dl0002p_355, align 4, !dbg !72
  %10 = load i32, i32* %.dU0002p.copy_449, align 4, !dbg !72
  store i32 %10, i32* %.dU0002p_356, align 4, !dbg !72
  %11 = load i32, i32* %.ds0002p.copy_450, align 4, !dbg !72
  store i32 %11, i32* %.ds0002p_352, align 4, !dbg !72
  br label %L.LB4_342

L.LB4_342:                                        ; preds = %L.LB4_358, %L.LB4_319
  %12 = load i32, i32* %__gtid___nv_MAIN_F1L23_2__454, align 4, !dbg !72
  %13 = bitcast i32* %.i0000p_321 to i64*, !dbg !72
  %14 = bitcast i32* %.dx0002p_354 to i64*, !dbg !72
  %15 = bitcast i32* %.de0002p_350 to i64*, !dbg !72
  %16 = bitcast i32* %.ds0002p_352 to i64*, !dbg !72
  %17 = call i32 @__kmpc_dispatch_next_4(i64* null, i32 %12, i64* %13, i64* %14, i64* %15, i64* %16), !dbg !72
  %18 = icmp eq i32 %17, 0, !dbg !72
  br i1 %18, label %L.LB4_343, label %L.LB4_492, !dbg !72

L.LB4_492:                                        ; preds = %L.LB4_342
  %19 = load i32, i32* %.dx0002p_354, align 4, !dbg !72
  store i32 %19, i32* %.dX0002p_353, align 4, !dbg !72
  %20 = load i32, i32* %.dX0002p_353, align 4, !dbg !72
  store i32 %20, i32* %i_320, align 4, !dbg !72
  %21 = load i32, i32* %.ds0002p_352, align 4, !dbg !72
  %22 = load i32, i32* %.de0002p_350, align 4, !dbg !72
  %23 = load i32, i32* %.dX0002p_353, align 4, !dbg !72
  %24 = sub nsw i32 %22, %23, !dbg !72
  %25 = add nsw i32 %21, %24, !dbg !72
  %26 = load i32, i32* %.ds0002p_352, align 4, !dbg !72
  %27 = sdiv i32 %25, %26, !dbg !72
  store i32 %27, i32* %.dY0002p_344, align 4, !dbg !72
  %28 = load i32, i32* %.dY0002p_344, align 4, !dbg !72
  %29 = icmp sle i32 %28, 0, !dbg !72
  br i1 %29, label %L.LB4_358, label %L.LB4_357, !dbg !72

L.LB4_357:                                        ; preds = %L.LB4_357, %L.LB4_492
  %30 = load i32, i32* %__gtid___nv_MAIN_F1L23_2__454, align 4, !dbg !74
  call void @__kmpc_ordered(i64* null, i32 %30), !dbg !74
  %31 = load i32, i32* %i_320, align 4, !dbg !75
  call void @llvm.dbg.value(metadata i32 %31, metadata !73, metadata !DIExpression()), !dbg !71
  %32 = sext i32 %31 to i64, !dbg !75
  %33 = bitcast %struct.BSS1* @.BSS1 to i8*, !dbg !75
  %34 = getelementptr i8, i8* %33, i64 -8, !dbg !75
  %35 = bitcast i8* %34 to i32*, !dbg !75
  %36 = getelementptr i32, i32* %35, i64 %32, !dbg !75
  %37 = load i32, i32* %36, align 4, !dbg !75
  %38 = add nsw i32 %37, 1, !dbg !75
  %39 = load i32, i32* %i_320, align 4, !dbg !75
  call void @llvm.dbg.value(metadata i32 %39, metadata !73, metadata !DIExpression()), !dbg !71
  %40 = sext i32 %39 to i64, !dbg !75
  %41 = bitcast %struct.BSS1* @.BSS1 to i8*, !dbg !75
  %42 = getelementptr i8, i8* %41, i64 -4, !dbg !75
  %43 = bitcast i8* %42 to i32*, !dbg !75
  %44 = getelementptr i32, i32* %43, i64 %40, !dbg !75
  store i32 %38, i32* %44, align 4, !dbg !75
  %45 = load i32, i32* %__gtid___nv_MAIN_F1L23_2__454, align 4, !dbg !76
  call void @__kmpc_end_ordered(i64* null, i32 %45), !dbg !76
  %46 = load i32, i32* %__gtid___nv_MAIN_F1L23_2__454, align 4, !dbg !71
  call void @__kmpc_dispatch_fini_4(i64* null, i32 %46), !dbg !71
  %47 = load i32, i32* %.ds0002p_352, align 4, !dbg !71
  %48 = load i32, i32* %i_320, align 4, !dbg !71
  call void @llvm.dbg.value(metadata i32 %48, metadata !73, metadata !DIExpression()), !dbg !71
  %49 = add nsw i32 %47, %48, !dbg !71
  store i32 %49, i32* %i_320, align 4, !dbg !71
  %50 = load i32, i32* %.dY0002p_344, align 4, !dbg !71
  %51 = sub nsw i32 %50, 1, !dbg !71
  store i32 %51, i32* %.dY0002p_344, align 4, !dbg !71
  %52 = load i32, i32* %.dY0002p_344, align 4, !dbg !71
  %53 = icmp sgt i32 %52, 0, !dbg !71
  br i1 %53, label %L.LB4_357, label %L.LB4_358, !dbg !71

L.LB4_358:                                        ; preds = %L.LB4_357, %L.LB4_492
  br label %L.LB4_342, !dbg !71

L.LB4_343:                                        ; preds = %L.LB4_342
  br label %L.LB4_322

L.LB4_322:                                        ; preds = %L.LB4_343
  ret void, !dbg !71
}

declare void @__kmpc_dispatch_fini_4(i64*, i32) #0

declare void @__kmpc_end_ordered(i64*, i32) #0

declare void @__kmpc_ordered(i64*, i32) #0

declare signext i32 @__kmpc_dispatch_next_4(i64*, i32, i64*, i64*, i64*, i64*) #0

declare void @__kmpc_dispatch_init_4(i64*, i32, i32, i32, i32, i32, i32) #0

declare void @__kmpc_fork_call(i64*, i32, i64*, i64*, ...) #0

declare void @__kmpc_end_serialized_parallel(i64*, i32) #0

declare void @__kmpc_serialized_parallel(i64*, i32) #0

declare void @_mp_ecs_nest(...) #0

declare signext i32 @f90io_ldw_end(...) #0

declare signext i32 @f90io_sc_ch_ldw(...) #0

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

!llvm.module.flags = !{!21, !22}
!llvm.dbg.cu = !{!4}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "var", scope: !2, file: !3, type: !9, isLocal: true, isDefinition: true)
!2 = distinct !DISubprogram(name: "drb155_missingordered_orig_gpu_no", scope: !4, file: !3, line: 11, type: !19, scopeLine: 11, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !4)
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB155-missingordered-orig-gpu-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !6, imports: !5)
!5 = !{}
!6 = !{!0, !7, !13}
!7 = !DIGlobalVariableExpression(var: !8, expr: !DIExpression())
!8 = distinct !DIGlobalVariable(name: "var", scope: !4, file: !3, type: !9, isLocal: true, isDefinition: true)
!9 = !DICompositeType(tag: DW_TAG_array_type, baseType: !10, size: 3200, align: 32, elements: !11)
!10 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!11 = !{!12}
!12 = !DISubrange(count: 100, lowerBound: 1)
!13 = !DIGlobalVariableExpression(var: !14, expr: !DIExpression())
!14 = distinct !DIGlobalVariable(name: "var", scope: !15, file: !3, type: !9, isLocal: true, isDefinition: true)
!15 = distinct !DISubprogram(name: "__nv_MAIN_F1L23_2", scope: !4, file: !3, line: 23, type: !16, scopeLine: 23, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !4)
!16 = !DISubroutineType(types: !17)
!17 = !{null, !10, !18, !18}
!18 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!19 = !DISubroutineType(cc: DW_CC_program, types: !20)
!20 = !{null}
!21 = !{i32 2, !"Dwarf Version", i32 4}
!22 = !{i32 2, !"Debug Info Version", i32 3}
!23 = !DILocalVariable(name: "omp_sched_static", scope: !2, file: !3, type: !10)
!24 = !DILocation(line: 0, scope: !2)
!25 = !DILocalVariable(name: "omp_sched_dynamic", scope: !2, file: !3, type: !10)
!26 = !DILocalVariable(name: "omp_proc_bind_false", scope: !2, file: !3, type: !10)
!27 = !DILocalVariable(name: "omp_proc_bind_true", scope: !2, file: !3, type: !10)
!28 = !DILocalVariable(name: "omp_proc_bind_master", scope: !2, file: !3, type: !10)
!29 = !DILocalVariable(name: "omp_lock_hint_none", scope: !2, file: !3, type: !10)
!30 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !2, file: !3, type: !10)
!31 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !2, file: !3, type: !10)
!32 = !DILocation(line: 38, column: 1, scope: !2)
!33 = !DILocation(line: 11, column: 1, scope: !2)
!34 = !DILocation(line: 18, column: 1, scope: !2)
!35 = !DILocalVariable(name: "i", scope: !2, file: !3, type: !10)
!36 = !DILocation(line: 19, column: 1, scope: !2)
!37 = !DILocation(line: 20, column: 1, scope: !2)
!38 = !DILocation(line: 30, column: 1, scope: !2)
!39 = !DILocation(line: 32, column: 1, scope: !2)
!40 = !DILocation(line: 33, column: 1, scope: !2)
!41 = !DILocation(line: 34, column: 1, scope: !2)
!42 = !DILocalVariable(scope: !2, file: !3, type: !10, flags: DIFlagArtificial)
!43 = !DILocation(line: 36, column: 1, scope: !2)
!44 = distinct !DISubprogram(name: "__nv_MAIN__F1L22_1", scope: !4, file: !3, line: 22, type: !16, scopeLine: 22, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !4)
!45 = !DILocalVariable(name: "__nv_MAIN__F1L22_1Arg0", arg: 1, scope: !44, file: !3, type: !10)
!46 = !DILocation(line: 0, scope: !44)
!47 = !DILocalVariable(name: "__nv_MAIN__F1L22_1Arg1", arg: 2, scope: !44, file: !3, type: !18)
!48 = !DILocalVariable(name: "__nv_MAIN__F1L22_1Arg2", arg: 3, scope: !44, file: !3, type: !18)
!49 = !DILocalVariable(name: "omp_sched_static", scope: !44, file: !3, type: !10)
!50 = !DILocalVariable(name: "omp_sched_dynamic", scope: !44, file: !3, type: !10)
!51 = !DILocalVariable(name: "omp_proc_bind_false", scope: !44, file: !3, type: !10)
!52 = !DILocalVariable(name: "omp_proc_bind_true", scope: !44, file: !3, type: !10)
!53 = !DILocalVariable(name: "omp_proc_bind_master", scope: !44, file: !3, type: !10)
!54 = !DILocalVariable(name: "omp_lock_hint_none", scope: !44, file: !3, type: !10)
!55 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !44, file: !3, type: !10)
!56 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !44, file: !3, type: !10)
!57 = !DILocation(line: 30, column: 1, scope: !44)
!58 = !DILocation(line: 23, column: 1, scope: !44)
!59 = !DILocalVariable(name: "__nv_MAIN_F1L23_2Arg0", arg: 1, scope: !15, file: !3, type: !10)
!60 = !DILocation(line: 0, scope: !15)
!61 = !DILocalVariable(name: "__nv_MAIN_F1L23_2Arg1", arg: 2, scope: !15, file: !3, type: !18)
!62 = !DILocalVariable(name: "__nv_MAIN_F1L23_2Arg2", arg: 3, scope: !15, file: !3, type: !18)
!63 = !DILocalVariable(name: "omp_sched_static", scope: !15, file: !3, type: !10)
!64 = !DILocalVariable(name: "omp_sched_dynamic", scope: !15, file: !3, type: !10)
!65 = !DILocalVariable(name: "omp_proc_bind_false", scope: !15, file: !3, type: !10)
!66 = !DILocalVariable(name: "omp_proc_bind_true", scope: !15, file: !3, type: !10)
!67 = !DILocalVariable(name: "omp_proc_bind_master", scope: !15, file: !3, type: !10)
!68 = !DILocalVariable(name: "omp_lock_hint_none", scope: !15, file: !3, type: !10)
!69 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !15, file: !3, type: !10)
!70 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !15, file: !3, type: !10)
!71 = !DILocation(line: 28, column: 1, scope: !15)
!72 = !DILocation(line: 24, column: 1, scope: !15)
!73 = !DILocalVariable(name: "i", scope: !15, file: !3, type: !10)
!74 = !DILocation(line: 25, column: 1, scope: !15)
!75 = !DILocation(line: 26, column: 1, scope: !15)
!76 = !DILocation(line: 27, column: 1, scope: !15)

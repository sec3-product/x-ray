; ModuleID = '/tmp/DRB011-minusminus-orig-yes-15d76c.ll'
source_filename = "/tmp/DRB011-minusminus-orig-yes-15d76c.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.BSS1 = type <{ [400 x i8] }>
%astruct.dt63 = type <{ i8*, i8*, i8* }>

@.BSS1 = internal global %struct.BSS1 zeroinitializer, align 32, !dbg !0
@.C310_MAIN_ = internal constant i32 25
@.C309_MAIN_ = internal constant i32 14
@.C334_MAIN_ = internal constant [11 x i8] c"numNodes2 ="
@.C284_MAIN_ = internal constant i64 0
@.C331_MAIN_ = internal constant i32 6
@.C328_MAIN_ = internal constant [55 x i8] c"micro-benchmarks-fortran/DRB011-minusminus-orig-yes.f95"
@.C330_MAIN_ = internal constant i32 37
@.C323_MAIN_ = internal constant i32 -1
@.C319_MAIN_ = internal constant i32 -5
@.C318_MAIN_ = internal constant i32 5
@.C300_MAIN_ = internal constant i32 2
@.C285_MAIN_ = internal constant i32 1
@.C315_MAIN_ = internal constant i32 100
@.C283_MAIN_ = internal constant i32 0
@.C323___nv_MAIN__F1L29_1 = internal constant i32 -1
@.C285___nv_MAIN__F1L29_1 = internal constant i32 1
@.C283___nv_MAIN__F1L29_1 = internal constant i32 0

define void @MAIN_() #0 !dbg !2 {
L.entry:
  %__gtid_MAIN__388 = alloca i32, align 4
  %len_317 = alloca i32, align 4
  %numnodes_312 = alloca i32, align 4
  %numnodes2_313 = alloca i32, align 4
  %.dY0001_345 = alloca i32, align 4
  %i_311 = alloca i32, align 4
  %.uplevelArgPack0001_379 = alloca %astruct.dt63, align 16
  %z__io_333 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !21, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i32 2, metadata !23, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i32 0, metadata !24, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i32 1, metadata !25, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i32 2, metadata !26, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i32 0, metadata !27, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i32 1, metadata !28, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i32 2, metadata !29, metadata !DIExpression()), !dbg !22
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !30
  store i32 %0, i32* %__gtid_MAIN__388, align 4, !dbg !30
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !31
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !31
  call void (i8*, ...) %2(i8* %1), !dbg !31
  br label %L.LB1_362

L.LB1_362:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %len_317, metadata !32, metadata !DIExpression()), !dbg !22
  store i32 100, i32* %len_317, align 4, !dbg !33
  %3 = load i32, i32* %len_317, align 4, !dbg !34
  call void @llvm.dbg.value(metadata i32 %3, metadata !32, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.declare(metadata i32* %numnodes_312, metadata !35, metadata !DIExpression()), !dbg !22
  store i32 %3, i32* %numnodes_312, align 4, !dbg !34
  call void @llvm.dbg.declare(metadata i32* %numnodes2_313, metadata !36, metadata !DIExpression()), !dbg !22
  store i32 0, i32* %numnodes2_313, align 4, !dbg !37
  %4 = load i32, i32* %len_317, align 4, !dbg !38
  call void @llvm.dbg.value(metadata i32 %4, metadata !32, metadata !DIExpression()), !dbg !22
  store i32 %4, i32* %.dY0001_345, align 4, !dbg !38
  call void @llvm.dbg.declare(metadata i32* %i_311, metadata !39, metadata !DIExpression()), !dbg !22
  store i32 1, i32* %i_311, align 4, !dbg !38
  %5 = load i32, i32* %.dY0001_345, align 4, !dbg !38
  %6 = icmp sle i32 %5, 0, !dbg !38
  br i1 %6, label %L.LB1_344, label %L.LB1_343, !dbg !38

L.LB1_343:                                        ; preds = %L.LB1_347, %L.LB1_362
  %7 = load i32, i32* %i_311, align 4, !dbg !40
  call void @llvm.dbg.value(metadata i32 %7, metadata !39, metadata !DIExpression()), !dbg !22
  %8 = load i32, i32* %i_311, align 4, !dbg !40
  call void @llvm.dbg.value(metadata i32 %8, metadata !39, metadata !DIExpression()), !dbg !22
  %9 = load i32, i32* %i_311, align 4, !dbg !40
  call void @llvm.dbg.value(metadata i32 %9, metadata !39, metadata !DIExpression()), !dbg !22
  %10 = lshr i32 %9, 31, !dbg !40
  %11 = add nsw i32 %8, %10, !dbg !40
  %12 = ashr i32 %11, 1, !dbg !40
  %13 = mul nsw i32 %12, 2, !dbg !40
  %14 = icmp ne i32 %7, %13, !dbg !40
  br i1 %14, label %L.LB1_346, label %L.LB1_415, !dbg !40

L.LB1_415:                                        ; preds = %L.LB1_343
  %15 = load i32, i32* %i_311, align 4, !dbg !41
  call void @llvm.dbg.value(metadata i32 %15, metadata !39, metadata !DIExpression()), !dbg !22
  %16 = sext i32 %15 to i64, !dbg !41
  %17 = bitcast %struct.BSS1* @.BSS1 to i8*, !dbg !41
  %18 = getelementptr i8, i8* %17, i64 -4, !dbg !41
  %19 = bitcast i8* %18 to i32*, !dbg !41
  %20 = getelementptr i32, i32* %19, i64 %16, !dbg !41
  store i32 5, i32* %20, align 4, !dbg !41
  br label %L.LB1_347, !dbg !42

L.LB1_346:                                        ; preds = %L.LB1_343
  %21 = load i32, i32* %i_311, align 4, !dbg !43
  call void @llvm.dbg.value(metadata i32 %21, metadata !39, metadata !DIExpression()), !dbg !22
  %22 = sext i32 %21 to i64, !dbg !43
  %23 = bitcast %struct.BSS1* @.BSS1 to i8*, !dbg !43
  %24 = getelementptr i8, i8* %23, i64 -4, !dbg !43
  %25 = bitcast i8* %24 to i32*, !dbg !43
  %26 = getelementptr i32, i32* %25, i64 %22, !dbg !43
  store i32 -5, i32* %26, align 4, !dbg !43
  br label %L.LB1_347

L.LB1_347:                                        ; preds = %L.LB1_346, %L.LB1_415
  %27 = load i32, i32* %i_311, align 4, !dbg !44
  call void @llvm.dbg.value(metadata i32 %27, metadata !39, metadata !DIExpression()), !dbg !22
  %28 = add nsw i32 %27, 1, !dbg !44
  store i32 %28, i32* %i_311, align 4, !dbg !44
  %29 = load i32, i32* %.dY0001_345, align 4, !dbg !44
  %30 = sub nsw i32 %29, 1, !dbg !44
  store i32 %30, i32* %.dY0001_345, align 4, !dbg !44
  %31 = load i32, i32* %.dY0001_345, align 4, !dbg !44
  %32 = icmp sgt i32 %31, 0, !dbg !44
  br i1 %32, label %L.LB1_343, label %L.LB1_344, !dbg !44

L.LB1_344:                                        ; preds = %L.LB1_347, %L.LB1_362
  %33 = bitcast i32* %numnodes_312 to i8*, !dbg !45
  %34 = bitcast %astruct.dt63* %.uplevelArgPack0001_379 to i8**, !dbg !45
  store i8* %33, i8** %34, align 8, !dbg !45
  %35 = bitcast i32* %numnodes2_313 to i8*, !dbg !45
  %36 = bitcast %astruct.dt63* %.uplevelArgPack0001_379 to i8*, !dbg !45
  %37 = getelementptr i8, i8* %36, i64 16, !dbg !45
  %38 = bitcast i8* %37 to i8**, !dbg !45
  store i8* %35, i8** %38, align 8, !dbg !45
  br label %L.LB1_386, !dbg !45

L.LB1_386:                                        ; preds = %L.LB1_344
  %39 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L29_1_ to i64*, !dbg !45
  %40 = bitcast %astruct.dt63* %.uplevelArgPack0001_379 to i64*, !dbg !45
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %39, i64* %40), !dbg !45
  call void (...) @_mp_bcs_nest(), !dbg !46
  %41 = bitcast i32* @.C330_MAIN_ to i8*, !dbg !46
  %42 = bitcast [55 x i8]* @.C328_MAIN_ to i8*, !dbg !46
  %43 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !46
  call void (i8*, i8*, i64, ...) %43(i8* %41, i8* %42, i64 55), !dbg !46
  %44 = bitcast i32* @.C331_MAIN_ to i8*, !dbg !46
  %45 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !46
  %46 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !46
  %47 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !46
  %48 = call i32 (i8*, i8*, i8*, i8*, ...) %47(i8* %44, i8* null, i8* %45, i8* %46), !dbg !46
  call void @llvm.dbg.declare(metadata i32* %z__io_333, metadata !47, metadata !DIExpression()), !dbg !22
  store i32 %48, i32* %z__io_333, align 4, !dbg !46
  %49 = bitcast [11 x i8]* @.C334_MAIN_ to i8*, !dbg !46
  %50 = bitcast i32 (...)* @f90io_sc_ch_ldw to i32 (i8*, i32, i64, ...)*, !dbg !46
  %51 = call i32 (i8*, i32, i64, ...) %50(i8* %49, i32 14, i64 11), !dbg !46
  store i32 %51, i32* %z__io_333, align 4, !dbg !46
  %52 = load i32, i32* %numnodes2_313, align 4, !dbg !46
  call void @llvm.dbg.value(metadata i32 %52, metadata !36, metadata !DIExpression()), !dbg !22
  %53 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !46
  %54 = call i32 (i32, i32, ...) %53(i32 %52, i32 25), !dbg !46
  store i32 %54, i32* %z__io_333, align 4, !dbg !46
  %55 = call i32 (...) @f90io_ldw_end(), !dbg !46
  store i32 %55, i32* %z__io_333, align 4, !dbg !46
  call void (...) @_mp_ecs_nest(), !dbg !46
  ret void, !dbg !30
}

define internal void @__nv_MAIN__F1L29_1_(i32* %__nv_MAIN__F1L29_1Arg0, i64* %__nv_MAIN__F1L29_1Arg1, i64* %__nv_MAIN__F1L29_1Arg2) #0 !dbg !9 {
L.entry:
  %__gtid___nv_MAIN__F1L29_1__435 = alloca i32, align 4
  %.i0000p_325 = alloca i32, align 4
  %i_324 = alloca i32, align 4
  %.du0002p_351 = alloca i32, align 4
  %.de0002p_352 = alloca i32, align 4
  %.di0002p_353 = alloca i32, align 4
  %.ds0002p_354 = alloca i32, align 4
  %.dl0002p_356 = alloca i32, align 4
  %.dl0002p.copy_429 = alloca i32, align 4
  %.de0002p.copy_430 = alloca i32, align 4
  %.ds0002p.copy_431 = alloca i32, align 4
  %.dX0002p_355 = alloca i32, align 4
  %.dY0002p_350 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L29_1Arg0, metadata !48, metadata !DIExpression()), !dbg !49
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L29_1Arg1, metadata !50, metadata !DIExpression()), !dbg !49
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L29_1Arg2, metadata !51, metadata !DIExpression()), !dbg !49
  call void @llvm.dbg.value(metadata i32 1, metadata !52, metadata !DIExpression()), !dbg !49
  call void @llvm.dbg.value(metadata i32 2, metadata !53, metadata !DIExpression()), !dbg !49
  call void @llvm.dbg.value(metadata i32 0, metadata !54, metadata !DIExpression()), !dbg !49
  call void @llvm.dbg.value(metadata i32 1, metadata !55, metadata !DIExpression()), !dbg !49
  call void @llvm.dbg.value(metadata i32 2, metadata !56, metadata !DIExpression()), !dbg !49
  call void @llvm.dbg.value(metadata i32 0, metadata !57, metadata !DIExpression()), !dbg !49
  call void @llvm.dbg.value(metadata i32 1, metadata !58, metadata !DIExpression()), !dbg !49
  call void @llvm.dbg.value(metadata i32 2, metadata !59, metadata !DIExpression()), !dbg !49
  %0 = load i32, i32* %__nv_MAIN__F1L29_1Arg0, align 4, !dbg !60
  store i32 %0, i32* %__gtid___nv_MAIN__F1L29_1__435, align 4, !dbg !60
  br label %L.LB2_419

L.LB2_419:                                        ; preds = %L.entry
  br label %L.LB2_322

L.LB2_322:                                        ; preds = %L.LB2_419
  store i32 0, i32* %.i0000p_325, align 4, !dbg !61
  %1 = bitcast i64* %__nv_MAIN__F1L29_1Arg2 to i32**, !dbg !61
  %2 = load i32*, i32** %1, align 8, !dbg !61
  %3 = load i32, i32* %2, align 4, !dbg !61
  call void @llvm.dbg.declare(metadata i32* %i_324, metadata !62, metadata !DIExpression()), !dbg !60
  store i32 %3, i32* %i_324, align 4, !dbg !61
  store i32 1, i32* %.du0002p_351, align 4, !dbg !61
  store i32 1, i32* %.de0002p_352, align 4, !dbg !61
  store i32 -1, i32* %.di0002p_353, align 4, !dbg !61
  %4 = load i32, i32* %.di0002p_353, align 4, !dbg !61
  store i32 %4, i32* %.ds0002p_354, align 4, !dbg !61
  %5 = bitcast i64* %__nv_MAIN__F1L29_1Arg2 to i32**, !dbg !61
  %6 = load i32*, i32** %5, align 8, !dbg !61
  %7 = load i32, i32* %6, align 4, !dbg !61
  store i32 %7, i32* %.dl0002p_356, align 4, !dbg !61
  %8 = load i32, i32* %.dl0002p_356, align 4, !dbg !61
  store i32 %8, i32* %.dl0002p.copy_429, align 4, !dbg !61
  %9 = load i32, i32* %.de0002p_352, align 4, !dbg !61
  store i32 %9, i32* %.de0002p.copy_430, align 4, !dbg !61
  %10 = load i32, i32* %.ds0002p_354, align 4, !dbg !61
  store i32 %10, i32* %.ds0002p.copy_431, align 4, !dbg !61
  %11 = load i32, i32* %__gtid___nv_MAIN__F1L29_1__435, align 4, !dbg !61
  %12 = bitcast i32* %.i0000p_325 to i64*, !dbg !61
  %13 = bitcast i32* %.dl0002p.copy_429 to i64*, !dbg !61
  %14 = bitcast i32* %.de0002p.copy_430 to i64*, !dbg !61
  %15 = bitcast i32* %.ds0002p.copy_431 to i64*, !dbg !61
  %16 = load i32, i32* %.ds0002p.copy_431, align 4, !dbg !61
  call void @__kmpc_for_static_init_4(i64* null, i32 %11, i32 34, i64* %12, i64* %13, i64* %14, i64* %15, i32 %16, i32 1), !dbg !61
  %17 = load i32, i32* %.dl0002p.copy_429, align 4, !dbg !61
  store i32 %17, i32* %.dl0002p_356, align 4, !dbg !61
  %18 = load i32, i32* %.de0002p.copy_430, align 4, !dbg !61
  store i32 %18, i32* %.de0002p_352, align 4, !dbg !61
  %19 = load i32, i32* %.ds0002p.copy_431, align 4, !dbg !61
  store i32 %19, i32* %.ds0002p_354, align 4, !dbg !61
  %20 = load i32, i32* %.dl0002p_356, align 4, !dbg !61
  store i32 %20, i32* %i_324, align 4, !dbg !61
  %21 = load i32, i32* %i_324, align 4, !dbg !61
  call void @llvm.dbg.value(metadata i32 %21, metadata !62, metadata !DIExpression()), !dbg !60
  store i32 %21, i32* %.dX0002p_355, align 4, !dbg !61
  %22 = load i32, i32* %.dX0002p_355, align 4, !dbg !61
  %23 = load i32, i32* %.du0002p_351, align 4, !dbg !61
  %24 = icmp slt i32 %22, %23, !dbg !61
  br i1 %24, label %L.LB2_349, label %L.LB2_458, !dbg !61

L.LB2_458:                                        ; preds = %L.LB2_322
  %25 = load i32, i32* %.dX0002p_355, align 4, !dbg !61
  store i32 %25, i32* %i_324, align 4, !dbg !61
  %26 = load i32, i32* %.di0002p_353, align 4, !dbg !61
  %27 = load i32, i32* %.de0002p_352, align 4, !dbg !61
  %28 = load i32, i32* %.dX0002p_355, align 4, !dbg !61
  %29 = sub nsw i32 %27, %28, !dbg !61
  %30 = add nsw i32 %26, %29, !dbg !61
  %31 = load i32, i32* %.di0002p_353, align 4, !dbg !61
  %32 = sdiv i32 %30, %31, !dbg !61
  store i32 %32, i32* %.dY0002p_350, align 4, !dbg !61
  %33 = load i32, i32* %.dY0002p_350, align 4, !dbg !61
  %34 = icmp sle i32 %33, 0, !dbg !61
  br i1 %34, label %L.LB2_359, label %L.LB2_358, !dbg !61

L.LB2_358:                                        ; preds = %L.LB2_360, %L.LB2_458
  %35 = load i32, i32* %i_324, align 4, !dbg !63
  call void @llvm.dbg.value(metadata i32 %35, metadata !62, metadata !DIExpression()), !dbg !60
  %36 = sext i32 %35 to i64, !dbg !63
  %37 = bitcast %struct.BSS1* @.BSS1 to i8*, !dbg !63
  %38 = getelementptr i8, i8* %37, i64 -4, !dbg !63
  %39 = bitcast i8* %38 to i32*, !dbg !63
  %40 = getelementptr i32, i32* %39, i64 %36, !dbg !63
  %41 = load i32, i32* %40, align 4, !dbg !63
  %42 = icmp sgt i32 %41, 0, !dbg !63
  br i1 %42, label %L.LB2_360, label %L.LB2_459, !dbg !63

L.LB2_459:                                        ; preds = %L.LB2_358
  %43 = bitcast i64* %__nv_MAIN__F1L29_1Arg2 to i8*, !dbg !64
  %44 = getelementptr i8, i8* %43, i64 16, !dbg !64
  %45 = bitcast i8* %44 to i32**, !dbg !64
  %46 = load i32*, i32** %45, align 8, !dbg !64
  %47 = load i32, i32* %46, align 4, !dbg !64
  %48 = sub nsw i32 %47, 1, !dbg !64
  %49 = bitcast i64* %__nv_MAIN__F1L29_1Arg2 to i8*, !dbg !64
  %50 = getelementptr i8, i8* %49, i64 16, !dbg !64
  %51 = bitcast i8* %50 to i32**, !dbg !64
  %52 = load i32*, i32** %51, align 8, !dbg !64
  store i32 %48, i32* %52, align 4, !dbg !64
  br label %L.LB2_360

L.LB2_360:                                        ; preds = %L.LB2_459, %L.LB2_358
  %53 = load i32, i32* %.di0002p_353, align 4, !dbg !60
  %54 = load i32, i32* %i_324, align 4, !dbg !60
  call void @llvm.dbg.value(metadata i32 %54, metadata !62, metadata !DIExpression()), !dbg !60
  %55 = add nsw i32 %53, %54, !dbg !60
  store i32 %55, i32* %i_324, align 4, !dbg !60
  %56 = load i32, i32* %.dY0002p_350, align 4, !dbg !60
  %57 = sub nsw i32 %56, 1, !dbg !60
  store i32 %57, i32* %.dY0002p_350, align 4, !dbg !60
  %58 = load i32, i32* %.dY0002p_350, align 4, !dbg !60
  %59 = icmp sgt i32 %58, 0, !dbg !60
  br i1 %59, label %L.LB2_358, label %L.LB2_359, !dbg !60

L.LB2_359:                                        ; preds = %L.LB2_360, %L.LB2_458
  br label %L.LB2_349

L.LB2_349:                                        ; preds = %L.LB2_359, %L.LB2_322
  %60 = load i32, i32* %__gtid___nv_MAIN__F1L29_1__435, align 4, !dbg !60
  call void @__kmpc_for_static_fini(i64* null, i32 %60), !dbg !60
  br label %L.LB2_326

L.LB2_326:                                        ; preds = %L.LB2_349
  ret void, !dbg !60
}

declare void @__kmpc_for_static_fini(i64*, i32) #0

declare void @__kmpc_for_static_init_4(i64*, i32, i32, i64*, i64*, i64*, i64*, i32, i32) #0

declare void @_mp_ecs_nest(...) #0

declare signext i32 @f90io_ldw_end(...) #0

declare signext i32 @f90io_sc_i_ldw(...) #0

declare signext i32 @f90io_sc_ch_ldw(...) #0

declare signext i32 @f90io_print_init(...) #0

declare void @f90io_src_info03a(...) #0

declare void @_mp_bcs_nest(...) #0

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare void @fort_init(...) #0

declare signext i32 @__kmpc_global_thread_num(i64*) #0

declare void @__kmpc_fork_call(i64*, i32, i64*, i64*, ...) #0

declare void @__kmpc_end_serialized_parallel(i64*, i32) #0

declare void @__kmpc_serialized_parallel(i64*, i32) #0

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { "no-frame-pointer-elim-non-leaf" }
attributes #1 = { nounwind readnone speculatable }

!llvm.module.flags = !{!19, !20}
!llvm.dbg.cu = !{!4}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "x", scope: !2, file: !3, type: !14, isLocal: true, isDefinition: true)
!2 = distinct !DISubprogram(name: "drb011_minusminus_orig_yes", scope: !4, file: !3, line: 11, type: !17, scopeLine: 11, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !4)
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB011-minusminus-orig-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !6, imports: !5)
!5 = !{}
!6 = !{!0, !7}
!7 = !DIGlobalVariableExpression(var: !8, expr: !DIExpression())
!8 = distinct !DIGlobalVariable(name: "x", scope: !9, file: !3, type: !14, isLocal: true, isDefinition: true)
!9 = distinct !DISubprogram(name: "__nv_MAIN__F1L29_1", scope: !4, file: !3, line: 29, type: !10, scopeLine: 29, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !4)
!10 = !DISubroutineType(types: !11)
!11 = !{null, !12, !13, !13}
!12 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!13 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!14 = !DICompositeType(tag: DW_TAG_array_type, baseType: !12, size: 3200, align: 32, elements: !15)
!15 = !{!16}
!16 = !DISubrange(count: 100, lowerBound: 1)
!17 = !DISubroutineType(cc: DW_CC_program, types: !18)
!18 = !{null}
!19 = !{i32 2, !"Dwarf Version", i32 4}
!20 = !{i32 2, !"Debug Info Version", i32 3}
!21 = !DILocalVariable(name: "omp_sched_static", scope: !2, file: !3, type: !12)
!22 = !DILocation(line: 0, scope: !2)
!23 = !DILocalVariable(name: "omp_sched_dynamic", scope: !2, file: !3, type: !12)
!24 = !DILocalVariable(name: "omp_proc_bind_false", scope: !2, file: !3, type: !12)
!25 = !DILocalVariable(name: "omp_proc_bind_true", scope: !2, file: !3, type: !12)
!26 = !DILocalVariable(name: "omp_proc_bind_master", scope: !2, file: !3, type: !12)
!27 = !DILocalVariable(name: "omp_lock_hint_none", scope: !2, file: !3, type: !12)
!28 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !2, file: !3, type: !12)
!29 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !2, file: !3, type: !12)
!30 = !DILocation(line: 38, column: 1, scope: !2)
!31 = !DILocation(line: 11, column: 1, scope: !2)
!32 = !DILocalVariable(name: "len", scope: !2, file: !3, type: !12)
!33 = !DILocation(line: 17, column: 1, scope: !2)
!34 = !DILocation(line: 18, column: 1, scope: !2)
!35 = !DILocalVariable(name: "numnodes", scope: !2, file: !3, type: !12)
!36 = !DILocalVariable(name: "numnodes2", scope: !2, file: !3, type: !12)
!37 = !DILocation(line: 19, column: 1, scope: !2)
!38 = !DILocation(line: 21, column: 1, scope: !2)
!39 = !DILocalVariable(name: "i", scope: !2, file: !3, type: !12)
!40 = !DILocation(line: 22, column: 1, scope: !2)
!41 = !DILocation(line: 23, column: 1, scope: !2)
!42 = !DILocation(line: 24, column: 1, scope: !2)
!43 = !DILocation(line: 25, column: 1, scope: !2)
!44 = !DILocation(line: 27, column: 1, scope: !2)
!45 = !DILocation(line: 29, column: 1, scope: !2)
!46 = !DILocation(line: 37, column: 1, scope: !2)
!47 = !DILocalVariable(scope: !2, file: !3, type: !12, flags: DIFlagArtificial)
!48 = !DILocalVariable(name: "__nv_MAIN__F1L29_1Arg0", arg: 1, scope: !9, file: !3, type: !12)
!49 = !DILocation(line: 0, scope: !9)
!50 = !DILocalVariable(name: "__nv_MAIN__F1L29_1Arg1", arg: 2, scope: !9, file: !3, type: !13)
!51 = !DILocalVariable(name: "__nv_MAIN__F1L29_1Arg2", arg: 3, scope: !9, file: !3, type: !13)
!52 = !DILocalVariable(name: "omp_sched_static", scope: !9, file: !3, type: !12)
!53 = !DILocalVariable(name: "omp_sched_dynamic", scope: !9, file: !3, type: !12)
!54 = !DILocalVariable(name: "omp_proc_bind_false", scope: !9, file: !3, type: !12)
!55 = !DILocalVariable(name: "omp_proc_bind_true", scope: !9, file: !3, type: !12)
!56 = !DILocalVariable(name: "omp_proc_bind_master", scope: !9, file: !3, type: !12)
!57 = !DILocalVariable(name: "omp_lock_hint_none", scope: !9, file: !3, type: !12)
!58 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !9, file: !3, type: !12)
!59 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !9, file: !3, type: !12)
!60 = !DILocation(line: 34, column: 1, scope: !9)
!61 = !DILocation(line: 30, column: 1, scope: !9)
!62 = !DILocalVariable(name: "i", scope: !9, file: !3, type: !12)
!63 = !DILocation(line: 31, column: 1, scope: !9)
!64 = !DILocation(line: 32, column: 1, scope: !9)

; ModuleID = '/tmp/DRB003-antidep2-orig-yes-eeffe9.ll'
source_filename = "/tmp/DRB003-antidep2-orig-yes-eeffe9.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.BSS1 = type <{ [1600 x i8] }>
%astruct.dt63 = type <{ i8*, i8* }>

@.BSS1 = internal global %struct.BSS1 zeroinitializer, align 32, !dbg !0
@.C306_MAIN_ = internal constant i32 27
@.C330_MAIN_ = internal constant i64 10
@.C305_MAIN_ = internal constant i32 14
@.C327_MAIN_ = internal constant [10 x i8] c"a(10,10) ="
@.C284_MAIN_ = internal constant i64 0
@.C326_MAIN_ = internal constant i32 6
@.C323_MAIN_ = internal constant [53 x i8] c"micro-benchmarks-fortran/DRB003-antidep2-orig-yes.f95"
@.C325_MAIN_ = internal constant i32 34
@.C290_MAIN_ = internal constant float 5.000000e-01
@.C285_MAIN_ = internal constant i32 1
@.C307_MAIN_ = internal constant i32 20
@.C283_MAIN_ = internal constant i32 0
@.C285___nv_MAIN__F1L26_1 = internal constant i32 1
@.C283___nv_MAIN__F1L26_1 = internal constant i32 0

define void @MAIN_() #0 !dbg !2 {
L.entry:
  %__gtid_MAIN__385 = alloca i32, align 4
  %len_314 = alloca i32, align 4
  %.dY0001_341 = alloca i32, align 4
  %i_308 = alloca i32, align 4
  %.dY0002_344 = alloca i32, align 4
  %j_309 = alloca i32, align 4
  %.uplevelArgPack0001_378 = alloca %astruct.dt63, align 16
  %z__io_329 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !22, metadata !DIExpression()), !dbg !23
  call void @llvm.dbg.value(metadata i32 0, metadata !24, metadata !DIExpression()), !dbg !23
  call void @llvm.dbg.value(metadata i32 1, metadata !25, metadata !DIExpression()), !dbg !23
  call void @llvm.dbg.value(metadata i32 0, metadata !26, metadata !DIExpression()), !dbg !23
  call void @llvm.dbg.value(metadata i32 1, metadata !27, metadata !DIExpression()), !dbg !23
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !28
  store i32 %0, i32* %__gtid_MAIN__385, align 4, !dbg !28
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !29
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !29
  call void (i8*, ...) %2(i8* %1), !dbg !29
  br label %L.LB1_361

L.LB1_361:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %len_314, metadata !30, metadata !DIExpression()), !dbg !23
  store i32 20, i32* %len_314, align 4, !dbg !31
  %3 = load i32, i32* %len_314, align 4, !dbg !32
  call void @llvm.dbg.value(metadata i32 %3, metadata !30, metadata !DIExpression()), !dbg !23
  store i32 %3, i32* %.dY0001_341, align 4, !dbg !32
  call void @llvm.dbg.declare(metadata i32* %i_308, metadata !33, metadata !DIExpression()), !dbg !23
  store i32 1, i32* %i_308, align 4, !dbg !32
  %4 = load i32, i32* %.dY0001_341, align 4, !dbg !32
  %5 = icmp sle i32 %4, 0, !dbg !32
  br i1 %5, label %L.LB1_340, label %L.LB1_339, !dbg !32

L.LB1_339:                                        ; preds = %L.LB1_343, %L.LB1_361
  %6 = load i32, i32* %len_314, align 4, !dbg !34
  call void @llvm.dbg.value(metadata i32 %6, metadata !30, metadata !DIExpression()), !dbg !23
  store i32 %6, i32* %.dY0002_344, align 4, !dbg !34
  call void @llvm.dbg.declare(metadata i32* %j_309, metadata !35, metadata !DIExpression()), !dbg !23
  store i32 1, i32* %j_309, align 4, !dbg !34
  %7 = load i32, i32* %.dY0002_344, align 4, !dbg !34
  %8 = icmp sle i32 %7, 0, !dbg !34
  br i1 %8, label %L.LB1_343, label %L.LB1_342, !dbg !34

L.LB1_342:                                        ; preds = %L.LB1_342, %L.LB1_339
  %9 = load i32, i32* %i_308, align 4, !dbg !36
  call void @llvm.dbg.value(metadata i32 %9, metadata !33, metadata !DIExpression()), !dbg !23
  %10 = sext i32 %9 to i64, !dbg !36
  %11 = load i32, i32* %j_309, align 4, !dbg !36
  call void @llvm.dbg.value(metadata i32 %11, metadata !35, metadata !DIExpression()), !dbg !23
  %12 = sext i32 %11 to i64, !dbg !36
  %13 = mul nsw i64 %12, 20, !dbg !36
  %14 = add nsw i64 %10, %13, !dbg !36
  %15 = bitcast %struct.BSS1* @.BSS1 to i8*, !dbg !36
  %16 = getelementptr i8, i8* %15, i64 -84, !dbg !36
  %17 = bitcast i8* %16 to float*, !dbg !36
  %18 = getelementptr float, float* %17, i64 %14, !dbg !36
  store float 5.000000e-01, float* %18, align 4, !dbg !36
  %19 = load i32, i32* %j_309, align 4, !dbg !37
  call void @llvm.dbg.value(metadata i32 %19, metadata !35, metadata !DIExpression()), !dbg !23
  %20 = add nsw i32 %19, 1, !dbg !37
  store i32 %20, i32* %j_309, align 4, !dbg !37
  %21 = load i32, i32* %.dY0002_344, align 4, !dbg !37
  %22 = sub nsw i32 %21, 1, !dbg !37
  store i32 %22, i32* %.dY0002_344, align 4, !dbg !37
  %23 = load i32, i32* %.dY0002_344, align 4, !dbg !37
  %24 = icmp sgt i32 %23, 0, !dbg !37
  br i1 %24, label %L.LB1_342, label %L.LB1_343, !dbg !37

L.LB1_343:                                        ; preds = %L.LB1_342, %L.LB1_339
  %25 = load i32, i32* %i_308, align 4, !dbg !38
  call void @llvm.dbg.value(metadata i32 %25, metadata !33, metadata !DIExpression()), !dbg !23
  %26 = add nsw i32 %25, 1, !dbg !38
  store i32 %26, i32* %i_308, align 4, !dbg !38
  %27 = load i32, i32* %.dY0001_341, align 4, !dbg !38
  %28 = sub nsw i32 %27, 1, !dbg !38
  store i32 %28, i32* %.dY0001_341, align 4, !dbg !38
  %29 = load i32, i32* %.dY0001_341, align 4, !dbg !38
  %30 = icmp sgt i32 %29, 0, !dbg !38
  br i1 %30, label %L.LB1_339, label %L.LB1_340, !dbg !38

L.LB1_340:                                        ; preds = %L.LB1_343, %L.LB1_361
  %31 = bitcast i32* %len_314 to i8*, !dbg !39
  %32 = bitcast %astruct.dt63* %.uplevelArgPack0001_378 to i8**, !dbg !39
  store i8* %31, i8** %32, align 8, !dbg !39
  br label %L.LB1_383, !dbg !39

L.LB1_383:                                        ; preds = %L.LB1_340
  %33 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L26_1_ to i64*, !dbg !39
  %34 = bitcast %astruct.dt63* %.uplevelArgPack0001_378 to i64*, !dbg !39
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %33, i64* %34), !dbg !39
  call void (...) @_mp_bcs_nest(), !dbg !40
  %35 = bitcast i32* @.C325_MAIN_ to i8*, !dbg !40
  %36 = bitcast [53 x i8]* @.C323_MAIN_ to i8*, !dbg !40
  %37 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !40
  call void (i8*, i8*, i64, ...) %37(i8* %35, i8* %36, i64 53), !dbg !40
  %38 = bitcast i32* @.C326_MAIN_ to i8*, !dbg !40
  %39 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !40
  %40 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !40
  %41 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !40
  %42 = call i32 (i8*, i8*, i8*, i8*, ...) %41(i8* %38, i8* null, i8* %39, i8* %40), !dbg !40
  call void @llvm.dbg.declare(metadata i32* %z__io_329, metadata !41, metadata !DIExpression()), !dbg !23
  store i32 %42, i32* %z__io_329, align 4, !dbg !40
  %43 = bitcast [10 x i8]* @.C327_MAIN_ to i8*, !dbg !40
  %44 = bitcast i32 (...)* @f90io_sc_ch_ldw to i32 (i8*, i32, i64, ...)*, !dbg !40
  %45 = call i32 (i8*, i32, i64, ...) %44(i8* %43, i32 14, i64 10), !dbg !40
  store i32 %45, i32* %z__io_329, align 4, !dbg !40
  %46 = bitcast %struct.BSS1* @.BSS1 to i8*, !dbg !40
  %47 = getelementptr i8, i8* %46, i64 756, !dbg !40
  %48 = bitcast i8* %47 to float*, !dbg !40
  %49 = load float, float* %48, align 4, !dbg !40
  %50 = bitcast i32 (...)* @f90io_sc_f_ldw to i32 (float, i32, ...)*, !dbg !40
  %51 = call i32 (float, i32, ...) %50(float %49, i32 27), !dbg !40
  store i32 %51, i32* %z__io_329, align 4, !dbg !40
  %52 = call i32 (...) @f90io_ldw_end(), !dbg !40
  store i32 %52, i32* %z__io_329, align 4, !dbg !40
  call void (...) @_mp_ecs_nest(), !dbg !40
  ret void, !dbg !28
}

define internal void @__nv_MAIN__F1L26_1_(i32* %__nv_MAIN__F1L26_1Arg0, i64* %__nv_MAIN__F1L26_1Arg1, i64* %__nv_MAIN__F1L26_1Arg2) #0 !dbg !9 {
L.entry:
  %__gtid___nv_MAIN__F1L26_1__439 = alloca i32, align 4
  %.i0000p_320 = alloca i32, align 4
  %i_319 = alloca i32, align 4
  %.du0003p_348 = alloca i32, align 4
  %.de0003p_349 = alloca i32, align 4
  %.di0003p_350 = alloca i32, align 4
  %.ds0003p_351 = alloca i32, align 4
  %.dl0003p_353 = alloca i32, align 4
  %.dl0003p.copy_433 = alloca i32, align 4
  %.de0003p.copy_434 = alloca i32, align 4
  %.ds0003p.copy_435 = alloca i32, align 4
  %.dX0003p_352 = alloca i32, align 4
  %.dY0003p_347 = alloca i32, align 4
  %.dY0004p_359 = alloca i32, align 4
  %j_318 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L26_1Arg0, metadata !42, metadata !DIExpression()), !dbg !43
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L26_1Arg1, metadata !44, metadata !DIExpression()), !dbg !43
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L26_1Arg2, metadata !45, metadata !DIExpression()), !dbg !43
  call void @llvm.dbg.value(metadata i32 1, metadata !46, metadata !DIExpression()), !dbg !43
  call void @llvm.dbg.value(metadata i32 0, metadata !47, metadata !DIExpression()), !dbg !43
  call void @llvm.dbg.value(metadata i32 1, metadata !48, metadata !DIExpression()), !dbg !43
  call void @llvm.dbg.value(metadata i32 0, metadata !49, metadata !DIExpression()), !dbg !43
  call void @llvm.dbg.value(metadata i32 1, metadata !50, metadata !DIExpression()), !dbg !43
  %0 = load i32, i32* %__nv_MAIN__F1L26_1Arg0, align 4, !dbg !51
  store i32 %0, i32* %__gtid___nv_MAIN__F1L26_1__439, align 4, !dbg !51
  br label %L.LB2_423

L.LB2_423:                                        ; preds = %L.entry
  br label %L.LB2_317

L.LB2_317:                                        ; preds = %L.LB2_423
  store i32 0, i32* %.i0000p_320, align 4, !dbg !52
  call void @llvm.dbg.declare(metadata i32* %i_319, metadata !53, metadata !DIExpression()), !dbg !51
  store i32 1, i32* %i_319, align 4, !dbg !52
  %1 = bitcast i64* %__nv_MAIN__F1L26_1Arg2 to i32**, !dbg !52
  %2 = load i32*, i32** %1, align 8, !dbg !52
  %3 = load i32, i32* %2, align 4, !dbg !52
  %4 = sub nsw i32 %3, 1, !dbg !52
  store i32 %4, i32* %.du0003p_348, align 4, !dbg !52
  %5 = bitcast i64* %__nv_MAIN__F1L26_1Arg2 to i32**, !dbg !52
  %6 = load i32*, i32** %5, align 8, !dbg !52
  %7 = load i32, i32* %6, align 4, !dbg !52
  %8 = sub nsw i32 %7, 1, !dbg !52
  store i32 %8, i32* %.de0003p_349, align 4, !dbg !52
  store i32 1, i32* %.di0003p_350, align 4, !dbg !52
  %9 = load i32, i32* %.di0003p_350, align 4, !dbg !52
  store i32 %9, i32* %.ds0003p_351, align 4, !dbg !52
  store i32 1, i32* %.dl0003p_353, align 4, !dbg !52
  %10 = load i32, i32* %.dl0003p_353, align 4, !dbg !52
  store i32 %10, i32* %.dl0003p.copy_433, align 4, !dbg !52
  %11 = load i32, i32* %.de0003p_349, align 4, !dbg !52
  store i32 %11, i32* %.de0003p.copy_434, align 4, !dbg !52
  %12 = load i32, i32* %.ds0003p_351, align 4, !dbg !52
  store i32 %12, i32* %.ds0003p.copy_435, align 4, !dbg !52
  %13 = load i32, i32* %__gtid___nv_MAIN__F1L26_1__439, align 4, !dbg !52
  %14 = bitcast i32* %.i0000p_320 to i64*, !dbg !52
  %15 = bitcast i32* %.dl0003p.copy_433 to i64*, !dbg !52
  %16 = bitcast i32* %.de0003p.copy_434 to i64*, !dbg !52
  %17 = bitcast i32* %.ds0003p.copy_435 to i64*, !dbg !52
  %18 = load i32, i32* %.ds0003p.copy_435, align 4, !dbg !52
  call void @__kmpc_for_static_init_4(i64* null, i32 %13, i32 34, i64* %14, i64* %15, i64* %16, i64* %17, i32 %18, i32 1), !dbg !52
  %19 = load i32, i32* %.dl0003p.copy_433, align 4, !dbg !52
  store i32 %19, i32* %.dl0003p_353, align 4, !dbg !52
  %20 = load i32, i32* %.de0003p.copy_434, align 4, !dbg !52
  store i32 %20, i32* %.de0003p_349, align 4, !dbg !52
  %21 = load i32, i32* %.ds0003p.copy_435, align 4, !dbg !52
  store i32 %21, i32* %.ds0003p_351, align 4, !dbg !52
  %22 = load i32, i32* %.dl0003p_353, align 4, !dbg !52
  store i32 %22, i32* %i_319, align 4, !dbg !52
  %23 = load i32, i32* %i_319, align 4, !dbg !52
  call void @llvm.dbg.value(metadata i32 %23, metadata !53, metadata !DIExpression()), !dbg !51
  store i32 %23, i32* %.dX0003p_352, align 4, !dbg !52
  %24 = load i32, i32* %.dX0003p_352, align 4, !dbg !52
  %25 = load i32, i32* %.du0003p_348, align 4, !dbg !52
  %26 = icmp sgt i32 %24, %25, !dbg !52
  br i1 %26, label %L.LB2_346, label %L.LB2_466, !dbg !52

L.LB2_466:                                        ; preds = %L.LB2_317
  %27 = load i32, i32* %.dX0003p_352, align 4, !dbg !52
  store i32 %27, i32* %i_319, align 4, !dbg !52
  %28 = load i32, i32* %.di0003p_350, align 4, !dbg !52
  %29 = load i32, i32* %.de0003p_349, align 4, !dbg !52
  %30 = load i32, i32* %.dX0003p_352, align 4, !dbg !52
  %31 = sub nsw i32 %29, %30, !dbg !52
  %32 = add nsw i32 %28, %31, !dbg !52
  %33 = load i32, i32* %.di0003p_350, align 4, !dbg !52
  %34 = sdiv i32 %32, %33, !dbg !52
  store i32 %34, i32* %.dY0003p_347, align 4, !dbg !52
  %35 = load i32, i32* %.dY0003p_347, align 4, !dbg !52
  %36 = icmp sle i32 %35, 0, !dbg !52
  br i1 %36, label %L.LB2_356, label %L.LB2_355, !dbg !52

L.LB2_355:                                        ; preds = %L.LB2_358, %L.LB2_466
  %37 = bitcast i64* %__nv_MAIN__F1L26_1Arg2 to i32**, !dbg !54
  %38 = load i32*, i32** %37, align 8, !dbg !54
  %39 = load i32, i32* %38, align 4, !dbg !54
  store i32 %39, i32* %.dY0004p_359, align 4, !dbg !54
  call void @llvm.dbg.declare(metadata i32* %j_318, metadata !55, metadata !DIExpression()), !dbg !51
  store i32 1, i32* %j_318, align 4, !dbg !54
  %40 = load i32, i32* %.dY0004p_359, align 4, !dbg !54
  %41 = icmp sle i32 %40, 0, !dbg !54
  br i1 %41, label %L.LB2_358, label %L.LB2_357, !dbg !54

L.LB2_357:                                        ; preds = %L.LB2_357, %L.LB2_355
  %42 = load i32, i32* %i_319, align 4, !dbg !56
  call void @llvm.dbg.value(metadata i32 %42, metadata !53, metadata !DIExpression()), !dbg !51
  %43 = sext i32 %42 to i64, !dbg !56
  %44 = load i32, i32* %j_318, align 4, !dbg !56
  call void @llvm.dbg.value(metadata i32 %44, metadata !55, metadata !DIExpression()), !dbg !51
  %45 = sext i32 %44 to i64, !dbg !56
  %46 = mul nsw i64 %45, 20, !dbg !56
  %47 = add nsw i64 %43, %46, !dbg !56
  %48 = bitcast %struct.BSS1* @.BSS1 to i8*, !dbg !56
  %49 = getelementptr i8, i8* %48, i64 -80, !dbg !56
  %50 = bitcast i8* %49 to float*, !dbg !56
  %51 = getelementptr float, float* %50, i64 %47, !dbg !56
  %52 = load float, float* %51, align 4, !dbg !56
  %53 = load i32, i32* %i_319, align 4, !dbg !56
  call void @llvm.dbg.value(metadata i32 %53, metadata !53, metadata !DIExpression()), !dbg !51
  %54 = sext i32 %53 to i64, !dbg !56
  %55 = load i32, i32* %j_318, align 4, !dbg !56
  call void @llvm.dbg.value(metadata i32 %55, metadata !55, metadata !DIExpression()), !dbg !51
  %56 = sext i32 %55 to i64, !dbg !56
  %57 = mul nsw i64 %56, 20, !dbg !56
  %58 = add nsw i64 %54, %57, !dbg !56
  %59 = bitcast %struct.BSS1* @.BSS1 to i8*, !dbg !56
  %60 = getelementptr i8, i8* %59, i64 -84, !dbg !56
  %61 = bitcast i8* %60 to float*, !dbg !56
  %62 = getelementptr float, float* %61, i64 %58, !dbg !56
  %63 = load float, float* %62, align 4, !dbg !56
  %64 = fadd fast float %52, %63, !dbg !56
  %65 = load i32, i32* %i_319, align 4, !dbg !56
  call void @llvm.dbg.value(metadata i32 %65, metadata !53, metadata !DIExpression()), !dbg !51
  %66 = sext i32 %65 to i64, !dbg !56
  %67 = load i32, i32* %j_318, align 4, !dbg !56
  call void @llvm.dbg.value(metadata i32 %67, metadata !55, metadata !DIExpression()), !dbg !51
  %68 = sext i32 %67 to i64, !dbg !56
  %69 = mul nsw i64 %68, 20, !dbg !56
  %70 = add nsw i64 %66, %69, !dbg !56
  %71 = bitcast %struct.BSS1* @.BSS1 to i8*, !dbg !56
  %72 = getelementptr i8, i8* %71, i64 -84, !dbg !56
  %73 = bitcast i8* %72 to float*, !dbg !56
  %74 = getelementptr float, float* %73, i64 %70, !dbg !56
  store float %64, float* %74, align 4, !dbg !56
  %75 = load i32, i32* %j_318, align 4, !dbg !57
  call void @llvm.dbg.value(metadata i32 %75, metadata !55, metadata !DIExpression()), !dbg !51
  %76 = add nsw i32 %75, 1, !dbg !57
  store i32 %76, i32* %j_318, align 4, !dbg !57
  %77 = load i32, i32* %.dY0004p_359, align 4, !dbg !57
  %78 = sub nsw i32 %77, 1, !dbg !57
  store i32 %78, i32* %.dY0004p_359, align 4, !dbg !57
  %79 = load i32, i32* %.dY0004p_359, align 4, !dbg !57
  %80 = icmp sgt i32 %79, 0, !dbg !57
  br i1 %80, label %L.LB2_357, label %L.LB2_358, !dbg !57

L.LB2_358:                                        ; preds = %L.LB2_357, %L.LB2_355
  %81 = load i32, i32* %.di0003p_350, align 4, !dbg !51
  %82 = load i32, i32* %i_319, align 4, !dbg !51
  call void @llvm.dbg.value(metadata i32 %82, metadata !53, metadata !DIExpression()), !dbg !51
  %83 = add nsw i32 %81, %82, !dbg !51
  store i32 %83, i32* %i_319, align 4, !dbg !51
  %84 = load i32, i32* %.dY0003p_347, align 4, !dbg !51
  %85 = sub nsw i32 %84, 1, !dbg !51
  store i32 %85, i32* %.dY0003p_347, align 4, !dbg !51
  %86 = load i32, i32* %.dY0003p_347, align 4, !dbg !51
  %87 = icmp sgt i32 %86, 0, !dbg !51
  br i1 %87, label %L.LB2_355, label %L.LB2_356, !dbg !51

L.LB2_356:                                        ; preds = %L.LB2_358, %L.LB2_466
  br label %L.LB2_346

L.LB2_346:                                        ; preds = %L.LB2_356, %L.LB2_317
  %88 = load i32, i32* %__gtid___nv_MAIN__F1L26_1__439, align 4, !dbg !51
  call void @__kmpc_for_static_fini(i64* null, i32 %88), !dbg !51
  br label %L.LB2_321

L.LB2_321:                                        ; preds = %L.LB2_346
  ret void, !dbg !51
}

declare void @__kmpc_for_static_fini(i64*, i32) #0

declare void @__kmpc_for_static_init_4(i64*, i32, i32, i64*, i64*, i64*, i64*, i32, i32) #0

declare void @_mp_ecs_nest(...) #0

declare signext i32 @f90io_ldw_end(...) #0

declare signext i32 @f90io_sc_f_ldw(...) #0

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

!llvm.module.flags = !{!20, !21}
!llvm.dbg.cu = !{!4}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !3, type: !14, isLocal: true, isDefinition: true)
!2 = distinct !DISubprogram(name: "drb003_antidep2_orig_yes", scope: !4, file: !3, line: 11, type: !18, scopeLine: 11, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !4)
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB003-antidep2-orig-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !6, imports: !5)
!5 = !{}
!6 = !{!0, !7}
!7 = !DIGlobalVariableExpression(var: !8, expr: !DIExpression())
!8 = distinct !DIGlobalVariable(name: "a", scope: !9, file: !3, type: !14, isLocal: true, isDefinition: true)
!9 = distinct !DISubprogram(name: "__nv_MAIN__F1L26_1", scope: !4, file: !3, line: 26, type: !10, scopeLine: 26, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !4)
!10 = !DISubroutineType(types: !11)
!11 = !{null, !12, !13, !13}
!12 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!13 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!14 = !DICompositeType(tag: DW_TAG_array_type, baseType: !15, size: 12800, align: 32, elements: !16)
!15 = !DIBasicType(name: "real", size: 32, align: 32, encoding: DW_ATE_float)
!16 = !{!17, !17}
!17 = !DISubrange(count: 20, lowerBound: 1)
!18 = !DISubroutineType(cc: DW_CC_program, types: !19)
!19 = !{null}
!20 = !{i32 2, !"Dwarf Version", i32 4}
!21 = !{i32 2, !"Debug Info Version", i32 3}
!22 = !DILocalVariable(name: "omp_sched_static", scope: !2, file: !3, type: !12)
!23 = !DILocation(line: 0, scope: !2)
!24 = !DILocalVariable(name: "omp_proc_bind_false", scope: !2, file: !3, type: !12)
!25 = !DILocalVariable(name: "omp_proc_bind_true", scope: !2, file: !3, type: !12)
!26 = !DILocalVariable(name: "omp_lock_hint_none", scope: !2, file: !3, type: !12)
!27 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !2, file: !3, type: !12)
!28 = !DILocation(line: 36, column: 1, scope: !2)
!29 = !DILocation(line: 11, column: 1, scope: !2)
!30 = !DILocalVariable(name: "len", scope: !2, file: !3, type: !12)
!31 = !DILocation(line: 18, column: 1, scope: !2)
!32 = !DILocation(line: 20, column: 1, scope: !2)
!33 = !DILocalVariable(name: "i", scope: !2, file: !3, type: !12)
!34 = !DILocation(line: 21, column: 1, scope: !2)
!35 = !DILocalVariable(name: "j", scope: !2, file: !3, type: !12)
!36 = !DILocation(line: 22, column: 1, scope: !2)
!37 = !DILocation(line: 23, column: 1, scope: !2)
!38 = !DILocation(line: 24, column: 1, scope: !2)
!39 = !DILocation(line: 26, column: 1, scope: !2)
!40 = !DILocation(line: 34, column: 1, scope: !2)
!41 = !DILocalVariable(scope: !2, file: !3, type: !12, flags: DIFlagArtificial)
!42 = !DILocalVariable(name: "__nv_MAIN__F1L26_1Arg0", arg: 1, scope: !9, file: !3, type: !12)
!43 = !DILocation(line: 0, scope: !9)
!44 = !DILocalVariable(name: "__nv_MAIN__F1L26_1Arg1", arg: 2, scope: !9, file: !3, type: !13)
!45 = !DILocalVariable(name: "__nv_MAIN__F1L26_1Arg2", arg: 3, scope: !9, file: !3, type: !13)
!46 = !DILocalVariable(name: "omp_sched_static", scope: !9, file: !3, type: !12)
!47 = !DILocalVariable(name: "omp_proc_bind_false", scope: !9, file: !3, type: !12)
!48 = !DILocalVariable(name: "omp_proc_bind_true", scope: !9, file: !3, type: !12)
!49 = !DILocalVariable(name: "omp_lock_hint_none", scope: !9, file: !3, type: !12)
!50 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !9, file: !3, type: !12)
!51 = !DILocation(line: 31, column: 1, scope: !9)
!52 = !DILocation(line: 27, column: 1, scope: !9)
!53 = !DILocalVariable(name: "i", scope: !9, file: !3, type: !12)
!54 = !DILocation(line: 28, column: 1, scope: !9)
!55 = !DILocalVariable(name: "j", scope: !9, file: !3, type: !12)
!56 = !DILocation(line: 29, column: 1, scope: !9)
!57 = !DILocation(line: 30, column: 1, scope: !9)

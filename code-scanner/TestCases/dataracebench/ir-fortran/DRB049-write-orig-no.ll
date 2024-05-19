; ModuleID = '/tmp/DRB049-write-orig-no-6428ae.ll'
source_filename = "/tmp/DRB049-write-orig-no-6428ae.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.BSS1 = type <{ [4000 x i8] }>
%astruct.dt69 = type <{ i8*, i8* }>

@.BSS1 = internal global %struct.BSS1 zeroinitializer, align 32, !dbg !0
@.C342_MAIN_ = internal constant [6 x i8] c"delete"
@.C341_MAIN_ = internal constant i32 38
@.C305_MAIN_ = internal constant i32 25
@.C335_MAIN_ = internal constant i32 34
@.C329_MAIN_ = internal constant [3 x i8] c"new"
@.C328_MAIN_ = internal constant i32 29
@.C324_MAIN_ = internal constant [3 x i8] c"old"
@.C325_MAIN_ = internal constant [6 x i8] c"append"
@.C326_MAIN_ = internal constant [5 x i8] c"write"
@.C323_MAIN_ = internal constant i32 6
@.C306_MAIN_ = internal constant i32 27
@.C284_MAIN_ = internal constant i64 0
@.C319_MAIN_ = internal constant [14 x i8] c"mytempfile.txt"
@.C316_MAIN_ = internal constant [49 x i8] c"micro-benchmarks-fortran/DRB049-write-orig-no.f95"
@.C318_MAIN_ = internal constant i32 24
@.C285_MAIN_ = internal constant i32 1
@.C311_MAIN_ = internal constant i32 1000
@.C283_MAIN_ = internal constant i32 0
@.C305___nv_MAIN__F1L32_1 = internal constant i32 25
@.C284___nv_MAIN__F1L32_1 = internal constant i64 0
@.C323___nv_MAIN__F1L32_1 = internal constant i32 6
@.C316___nv_MAIN__F1L32_1 = internal constant [49 x i8] c"micro-benchmarks-fortran/DRB049-write-orig-no.f95"
@.C335___nv_MAIN__F1L32_1 = internal constant i32 34
@.C285___nv_MAIN__F1L32_1 = internal constant i32 1
@.C283___nv_MAIN__F1L32_1 = internal constant i32 0

define void @MAIN_() #0 !dbg !2 {
L.entry:
  %__gtid_MAIN__411 = alloca i32, align 4
  %len_314 = alloca i32, align 4
  %.dY0001_349 = alloca i32, align 4
  %i_307 = alloca i32, align 4
  %exist_313 = alloca i32, align 4
  %z__io_321 = alloca i32, align 4
  %stat_309 = alloca i32, align 4
  %.uplevelArgPack0001_404 = alloca %astruct.dt69, align 16
  call void @llvm.dbg.value(metadata i32 1, metadata !21, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i32 0, metadata !23, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i32 1, metadata !24, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i32 0, metadata !25, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i32 1, metadata !26, metadata !DIExpression()), !dbg !22
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !27
  store i32 %0, i32* %__gtid_MAIN__411, align 4, !dbg !27
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !28
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !28
  call void (i8*, ...) %2(i8* %1), !dbg !28
  br label %L.LB1_366

L.LB1_366:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %len_314, metadata !29, metadata !DIExpression()), !dbg !22
  store i32 1000, i32* %len_314, align 4, !dbg !30
  %3 = load i32, i32* %len_314, align 4, !dbg !31
  call void @llvm.dbg.value(metadata i32 %3, metadata !29, metadata !DIExpression()), !dbg !22
  store i32 %3, i32* %.dY0001_349, align 4, !dbg !31
  call void @llvm.dbg.declare(metadata i32* %i_307, metadata !32, metadata !DIExpression()), !dbg !22
  store i32 1, i32* %i_307, align 4, !dbg !31
  %4 = load i32, i32* %.dY0001_349, align 4, !dbg !31
  %5 = icmp sle i32 %4, 0, !dbg !31
  br i1 %5, label %L.LB1_348, label %L.LB1_347, !dbg !31

L.LB1_347:                                        ; preds = %L.LB1_347, %L.LB1_366
  %6 = load i32, i32* %i_307, align 4, !dbg !33
  call void @llvm.dbg.value(metadata i32 %6, metadata !32, metadata !DIExpression()), !dbg !22
  %7 = load i32, i32* %i_307, align 4, !dbg !33
  call void @llvm.dbg.value(metadata i32 %7, metadata !32, metadata !DIExpression()), !dbg !22
  %8 = sext i32 %7 to i64, !dbg !33
  %9 = bitcast %struct.BSS1* @.BSS1 to i8*, !dbg !33
  %10 = getelementptr i8, i8* %9, i64 -4, !dbg !33
  %11 = bitcast i8* %10 to i32*, !dbg !33
  %12 = getelementptr i32, i32* %11, i64 %8, !dbg !33
  store i32 %6, i32* %12, align 4, !dbg !33
  %13 = load i32, i32* %i_307, align 4, !dbg !34
  call void @llvm.dbg.value(metadata i32 %13, metadata !32, metadata !DIExpression()), !dbg !22
  %14 = add nsw i32 %13, 1, !dbg !34
  store i32 %14, i32* %i_307, align 4, !dbg !34
  %15 = load i32, i32* %.dY0001_349, align 4, !dbg !34
  %16 = sub nsw i32 %15, 1, !dbg !34
  store i32 %16, i32* %.dY0001_349, align 4, !dbg !34
  %17 = load i32, i32* %.dY0001_349, align 4, !dbg !34
  %18 = icmp sgt i32 %17, 0, !dbg !34
  br i1 %18, label %L.LB1_347, label %L.LB1_348, !dbg !34

L.LB1_348:                                        ; preds = %L.LB1_347, %L.LB1_366
  call void (...) @_mp_bcs_nest(), !dbg !35
  %19 = bitcast i32* @.C318_MAIN_ to i8*, !dbg !35
  %20 = bitcast [49 x i8]* @.C316_MAIN_ to i8*, !dbg !35
  %21 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !35
  call void (i8*, i8*, i64, ...) %21(i8* %19, i8* %20, i64 49), !dbg !35
  %22 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !35
  %23 = bitcast [14 x i8]* @.C319_MAIN_ to i8*, !dbg !35
  %24 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !35
  call void @llvm.dbg.declare(metadata i32* %exist_313, metadata !36, metadata !DIExpression()), !dbg !22
  %25 = bitcast i32* %exist_313 to i8*, !dbg !35
  %26 = bitcast i32 (...)* @f90io_inquire2003a to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, ...)*, !dbg !35
  %27 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, ...) %26(i8* %22, i8* %23, i8* %24, i8* null, i8* %25, i8* null, i8* null, i8* null, i8* null, i8* null, i8* null, i8* null, i8* null, i8* null, i8* null, i8* null, i8* null, i8* null, i8* null, i8* null, i8* null, i8* null, i8* null, i8* null, i8* null, i64 14, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0), !dbg !35
  call void @llvm.dbg.declare(metadata i32* %z__io_321, metadata !38, metadata !DIExpression()), !dbg !22
  store i32 %27, i32* %z__io_321, align 4, !dbg !35
  call void (...) @_mp_ecs_nest(), !dbg !35
  %28 = load i32, i32* %exist_313, align 4, !dbg !39
  call void @llvm.dbg.value(metadata i32 %28, metadata !36, metadata !DIExpression()), !dbg !22
  %29 = and i32 %28, 1, !dbg !39
  %30 = icmp eq i32 %29, 0, !dbg !39
  br i1 %30, label %L.LB1_350, label %L.LB1_431, !dbg !39

L.LB1_431:                                        ; preds = %L.LB1_348
  call void (...) @_mp_bcs_nest(), !dbg !40
  %31 = bitcast i32* @.C306_MAIN_ to i8*, !dbg !40
  %32 = bitcast [49 x i8]* @.C316_MAIN_ to i8*, !dbg !40
  %33 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !40
  call void (i8*, i8*, i64, ...) %33(i8* %31, i8* %32, i64 49), !dbg !40
  %34 = bitcast i32* @.C323_MAIN_ to i8*, !dbg !40
  %35 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !40
  %36 = bitcast [5 x i8]* @.C326_MAIN_ to i8*, !dbg !40
  %37 = bitcast [14 x i8]* @.C319_MAIN_ to i8*, !dbg !40
  call void @llvm.dbg.declare(metadata i32* %stat_309, metadata !41, metadata !DIExpression()), !dbg !22
  %38 = bitcast i32* %stat_309 to i8*, !dbg !40
  %39 = bitcast [6 x i8]* @.C325_MAIN_ to i8*, !dbg !40
  %40 = bitcast [3 x i8]* @.C324_MAIN_ to i8*, !dbg !40
  %41 = bitcast i32 (...)* @f90io_open2003a to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, ...)*, !dbg !40
  %42 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, ...) %41(i8* %34, i8* %35, i8* null, i8* %36, i8* null, i8* null, i8* %37, i8* null, i8* %38, i8* null, i8* %39, i8* null, i8* %40, i8* null, i64 0, i64 5, i64 0, i64 0, i64 14, i64 0, i64 0, i64 6, i64 3, i64 0), !dbg !40
  store i32 %42, i32* %z__io_321, align 4, !dbg !40
  call void (...) @_mp_ecs_nest(), !dbg !40
  br label %L.LB1_351, !dbg !42

L.LB1_350:                                        ; preds = %L.LB1_348
  call void (...) @_mp_bcs_nest(), !dbg !43
  %43 = bitcast i32* @.C328_MAIN_ to i8*, !dbg !43
  %44 = bitcast [49 x i8]* @.C316_MAIN_ to i8*, !dbg !43
  %45 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !43
  call void (i8*, i8*, i64, ...) %45(i8* %43, i8* %44, i64 49), !dbg !43
  %46 = bitcast i32* @.C323_MAIN_ to i8*, !dbg !43
  %47 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !43
  %48 = bitcast [5 x i8]* @.C326_MAIN_ to i8*, !dbg !43
  %49 = bitcast [14 x i8]* @.C319_MAIN_ to i8*, !dbg !43
  %50 = bitcast i32* %stat_309 to i8*, !dbg !43
  %51 = bitcast [3 x i8]* @.C329_MAIN_ to i8*, !dbg !43
  %52 = bitcast i32 (...)* @f90io_open2003a to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, ...)*, !dbg !43
  %53 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, ...) %52(i8* %46, i8* %47, i8* null, i8* %48, i8* null, i8* null, i8* %49, i8* null, i8* %50, i8* null, i8* null, i8* null, i8* %51, i8* null, i64 0, i64 5, i64 0, i64 0, i64 14, i64 0, i64 0, i64 0, i64 3, i64 0), !dbg !43
  store i32 %53, i32* %z__io_321, align 4, !dbg !43
  call void (...) @_mp_ecs_nest(), !dbg !43
  br label %L.LB1_351

L.LB1_351:                                        ; preds = %L.LB1_350, %L.LB1_431
  %54 = bitcast i32* %len_314 to i8*, !dbg !44
  %55 = bitcast %astruct.dt69* %.uplevelArgPack0001_404 to i8**, !dbg !44
  store i8* %54, i8** %55, align 8, !dbg !44
  br label %L.LB1_409, !dbg !44

L.LB1_409:                                        ; preds = %L.LB1_351
  %56 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L32_1_ to i64*, !dbg !44
  %57 = bitcast %astruct.dt69* %.uplevelArgPack0001_404 to i64*, !dbg !44
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %56, i64* %57), !dbg !44
  %58 = load i32, i32* %stat_309, align 4, !dbg !45
  call void @llvm.dbg.value(metadata i32 %58, metadata !41, metadata !DIExpression()), !dbg !22
  %59 = icmp ne i32 %58, 0, !dbg !45
  br i1 %59, label %L.LB1_364, label %L.LB1_432, !dbg !45

L.LB1_432:                                        ; preds = %L.LB1_409
  call void (...) @_mp_bcs_nest(), !dbg !45
  %60 = bitcast i32* @.C341_MAIN_ to i8*, !dbg !45
  %61 = bitcast [49 x i8]* @.C316_MAIN_ to i8*, !dbg !45
  %62 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !45
  call void (i8*, i8*, i64, ...) %62(i8* %60, i8* %61, i64 49), !dbg !45
  %63 = bitcast i32* @.C323_MAIN_ to i8*, !dbg !45
  %64 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !45
  %65 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !45
  %66 = bitcast [6 x i8]* @.C342_MAIN_ to i8*, !dbg !45
  %67 = bitcast i32 (...)* @f90io_closea to i32 (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !45
  %68 = call i32 (i8*, i8*, i8*, i8*, i64, ...) %67(i8* %63, i8* %64, i8* %65, i8* %66, i64 6), !dbg !45
  store i32 %68, i32* %z__io_321, align 4, !dbg !45
  call void (...) @_mp_ecs_nest(), !dbg !45
  br label %L.LB1_364

L.LB1_364:                                        ; preds = %L.LB1_432, %L.LB1_409
  ret void, !dbg !27
}

define internal void @__nv_MAIN__F1L32_1_(i32* %__nv_MAIN__F1L32_1Arg0, i64* %__nv_MAIN__F1L32_1Arg1, i64* %__nv_MAIN__F1L32_1Arg2) #0 !dbg !9 {
L.entry:
  %__gtid___nv_MAIN__F1L32_1__451 = alloca i32, align 4
  %.i0000p_334 = alloca i32, align 4
  %i_333 = alloca i32, align 4
  %.du0002p_355 = alloca i32, align 4
  %.de0002p_356 = alloca i32, align 4
  %.di0002p_357 = alloca i32, align 4
  %.ds0002p_358 = alloca i32, align 4
  %.dl0002p_360 = alloca i32, align 4
  %.dl0002p.copy_445 = alloca i32, align 4
  %.de0002p.copy_446 = alloca i32, align 4
  %.ds0002p.copy_447 = alloca i32, align 4
  %.dX0002p_359 = alloca i32, align 4
  %.dY0002p_354 = alloca i32, align 4
  %z__io_321 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L32_1Arg0, metadata !46, metadata !DIExpression()), !dbg !47
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L32_1Arg1, metadata !48, metadata !DIExpression()), !dbg !47
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L32_1Arg2, metadata !49, metadata !DIExpression()), !dbg !47
  call void @llvm.dbg.value(metadata i32 1, metadata !50, metadata !DIExpression()), !dbg !47
  call void @llvm.dbg.value(metadata i32 0, metadata !51, metadata !DIExpression()), !dbg !47
  call void @llvm.dbg.value(metadata i32 1, metadata !52, metadata !DIExpression()), !dbg !47
  call void @llvm.dbg.value(metadata i32 0, metadata !53, metadata !DIExpression()), !dbg !47
  call void @llvm.dbg.value(metadata i32 1, metadata !54, metadata !DIExpression()), !dbg !47
  %0 = load i32, i32* %__nv_MAIN__F1L32_1Arg0, align 4, !dbg !55
  store i32 %0, i32* %__gtid___nv_MAIN__F1L32_1__451, align 4, !dbg !55
  br label %L.LB2_436

L.LB2_436:                                        ; preds = %L.entry
  br label %L.LB2_332

L.LB2_332:                                        ; preds = %L.LB2_436
  store i32 0, i32* %.i0000p_334, align 4, !dbg !56
  call void @llvm.dbg.declare(metadata i32* %i_333, metadata !57, metadata !DIExpression()), !dbg !55
  store i32 1, i32* %i_333, align 4, !dbg !56
  %1 = bitcast i64* %__nv_MAIN__F1L32_1Arg2 to i32**, !dbg !56
  %2 = load i32*, i32** %1, align 8, !dbg !56
  %3 = load i32, i32* %2, align 4, !dbg !56
  store i32 %3, i32* %.du0002p_355, align 4, !dbg !56
  %4 = bitcast i64* %__nv_MAIN__F1L32_1Arg2 to i32**, !dbg !56
  %5 = load i32*, i32** %4, align 8, !dbg !56
  %6 = load i32, i32* %5, align 4, !dbg !56
  store i32 %6, i32* %.de0002p_356, align 4, !dbg !56
  store i32 1, i32* %.di0002p_357, align 4, !dbg !56
  %7 = load i32, i32* %.di0002p_357, align 4, !dbg !56
  store i32 %7, i32* %.ds0002p_358, align 4, !dbg !56
  store i32 1, i32* %.dl0002p_360, align 4, !dbg !56
  %8 = load i32, i32* %.dl0002p_360, align 4, !dbg !56
  store i32 %8, i32* %.dl0002p.copy_445, align 4, !dbg !56
  %9 = load i32, i32* %.de0002p_356, align 4, !dbg !56
  store i32 %9, i32* %.de0002p.copy_446, align 4, !dbg !56
  %10 = load i32, i32* %.ds0002p_358, align 4, !dbg !56
  store i32 %10, i32* %.ds0002p.copy_447, align 4, !dbg !56
  %11 = load i32, i32* %__gtid___nv_MAIN__F1L32_1__451, align 4, !dbg !56
  %12 = bitcast i32* %.i0000p_334 to i64*, !dbg !56
  %13 = bitcast i32* %.dl0002p.copy_445 to i64*, !dbg !56
  %14 = bitcast i32* %.de0002p.copy_446 to i64*, !dbg !56
  %15 = bitcast i32* %.ds0002p.copy_447 to i64*, !dbg !56
  %16 = load i32, i32* %.ds0002p.copy_447, align 4, !dbg !56
  call void @__kmpc_for_static_init_4(i64* null, i32 %11, i32 34, i64* %12, i64* %13, i64* %14, i64* %15, i32 %16, i32 1), !dbg !56
  %17 = load i32, i32* %.dl0002p.copy_445, align 4, !dbg !56
  store i32 %17, i32* %.dl0002p_360, align 4, !dbg !56
  %18 = load i32, i32* %.de0002p.copy_446, align 4, !dbg !56
  store i32 %18, i32* %.de0002p_356, align 4, !dbg !56
  %19 = load i32, i32* %.ds0002p.copy_447, align 4, !dbg !56
  store i32 %19, i32* %.ds0002p_358, align 4, !dbg !56
  %20 = load i32, i32* %.dl0002p_360, align 4, !dbg !56
  store i32 %20, i32* %i_333, align 4, !dbg !56
  %21 = load i32, i32* %i_333, align 4, !dbg !56
  call void @llvm.dbg.value(metadata i32 %21, metadata !57, metadata !DIExpression()), !dbg !55
  store i32 %21, i32* %.dX0002p_359, align 4, !dbg !56
  %22 = load i32, i32* %.dX0002p_359, align 4, !dbg !56
  %23 = load i32, i32* %.du0002p_355, align 4, !dbg !56
  %24 = icmp sgt i32 %22, %23, !dbg !56
  br i1 %24, label %L.LB2_353, label %L.LB2_473, !dbg !56

L.LB2_473:                                        ; preds = %L.LB2_332
  %25 = load i32, i32* %.dX0002p_359, align 4, !dbg !56
  store i32 %25, i32* %i_333, align 4, !dbg !56
  %26 = load i32, i32* %.di0002p_357, align 4, !dbg !56
  %27 = load i32, i32* %.de0002p_356, align 4, !dbg !56
  %28 = load i32, i32* %.dX0002p_359, align 4, !dbg !56
  %29 = sub nsw i32 %27, %28, !dbg !56
  %30 = add nsw i32 %26, %29, !dbg !56
  %31 = load i32, i32* %.di0002p_357, align 4, !dbg !56
  %32 = sdiv i32 %30, %31, !dbg !56
  store i32 %32, i32* %.dY0002p_354, align 4, !dbg !56
  %33 = load i32, i32* %.dY0002p_354, align 4, !dbg !56
  %34 = icmp sle i32 %33, 0, !dbg !56
  br i1 %34, label %L.LB2_363, label %L.LB2_362, !dbg !56

L.LB2_362:                                        ; preds = %L.LB2_362, %L.LB2_473
  call void (...) @_mp_bcs_nest(), !dbg !58
  %35 = bitcast i32* @.C335___nv_MAIN__F1L32_1 to i8*, !dbg !58
  %36 = bitcast [49 x i8]* @.C316___nv_MAIN__F1L32_1 to i8*, !dbg !58
  %37 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !58
  call void (i8*, i8*, i64, ...) %37(i8* %35, i8* %36, i64 49), !dbg !58
  %38 = bitcast i32* @.C323___nv_MAIN__F1L32_1 to i8*, !dbg !58
  %39 = bitcast i32* @.C283___nv_MAIN__F1L32_1 to i8*, !dbg !58
  %40 = bitcast i32* @.C283___nv_MAIN__F1L32_1 to i8*, !dbg !58
  %41 = bitcast i32 (...)* @f90io_ldw_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !58
  %42 = call i32 (i8*, i8*, i8*, i8*, ...) %41(i8* %38, i8* null, i8* %39, i8* %40), !dbg !58
  call void @llvm.dbg.declare(metadata i32* %z__io_321, metadata !59, metadata !DIExpression()), !dbg !47
  store i32 %42, i32* %z__io_321, align 4, !dbg !58
  %43 = load i32, i32* %i_333, align 4, !dbg !58
  call void @llvm.dbg.value(metadata i32 %43, metadata !57, metadata !DIExpression()), !dbg !55
  %44 = sext i32 %43 to i64, !dbg !58
  %45 = bitcast %struct.BSS1* @.BSS1 to i8*, !dbg !58
  %46 = getelementptr i8, i8* %45, i64 -4, !dbg !58
  %47 = bitcast i8* %46 to i32*, !dbg !58
  %48 = getelementptr i32, i32* %47, i64 %44, !dbg !58
  %49 = load i32, i32* %48, align 4, !dbg !58
  %50 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !58
  %51 = call i32 (i32, i32, ...) %50(i32 %49, i32 25), !dbg !58
  store i32 %51, i32* %z__io_321, align 4, !dbg !58
  %52 = call i32 (...) @f90io_ldw_end(), !dbg !58
  store i32 %52, i32* %z__io_321, align 4, !dbg !58
  call void (...) @_mp_ecs_nest(), !dbg !58
  %53 = load i32, i32* %.di0002p_357, align 4, !dbg !55
  %54 = load i32, i32* %i_333, align 4, !dbg !55
  call void @llvm.dbg.value(metadata i32 %54, metadata !57, metadata !DIExpression()), !dbg !55
  %55 = add nsw i32 %53, %54, !dbg !55
  store i32 %55, i32* %i_333, align 4, !dbg !55
  %56 = load i32, i32* %.dY0002p_354, align 4, !dbg !55
  %57 = sub nsw i32 %56, 1, !dbg !55
  store i32 %57, i32* %.dY0002p_354, align 4, !dbg !55
  %58 = load i32, i32* %.dY0002p_354, align 4, !dbg !55
  %59 = icmp sgt i32 %58, 0, !dbg !55
  br i1 %59, label %L.LB2_362, label %L.LB2_363, !dbg !55

L.LB2_363:                                        ; preds = %L.LB2_362, %L.LB2_473
  br label %L.LB2_353

L.LB2_353:                                        ; preds = %L.LB2_363, %L.LB2_332
  %60 = load i32, i32* %__gtid___nv_MAIN__F1L32_1__451, align 4, !dbg !55
  call void @__kmpc_for_static_fini(i64* null, i32 %60), !dbg !55
  br label %L.LB2_340

L.LB2_340:                                        ; preds = %L.LB2_353
  ret void, !dbg !55
}

declare signext i32 @f90io_ldw_end(...) #0

declare signext i32 @f90io_sc_i_ldw(...) #0

declare signext i32 @f90io_ldw_init(...) #0

declare void @__kmpc_for_static_fini(i64*, i32) #0

declare void @__kmpc_for_static_init_4(i64*, i32, i32, i64*, i64*, i64*, i64*, i32, i32) #0

declare signext i32 @f90io_closea(...) #0

declare signext i32 @f90io_open2003a(...) #0

declare void @_mp_ecs_nest(...) #0

declare signext i32 @f90io_inquire2003a(...) #0

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
!1 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !3, type: !14, isLocal: true, isDefinition: true)
!2 = distinct !DISubprogram(name: "drb049_fprintf_orig_no", scope: !4, file: !3, line: 10, type: !17, scopeLine: 10, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !4)
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB049-write-orig-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !6, imports: !5)
!5 = !{}
!6 = !{!0, !7}
!7 = !DIGlobalVariableExpression(var: !8, expr: !DIExpression())
!8 = distinct !DIGlobalVariable(name: "a", scope: !9, file: !3, type: !14, isLocal: true, isDefinition: true)
!9 = distinct !DISubprogram(name: "__nv_MAIN__F1L32_1", scope: !4, file: !3, line: 32, type: !10, scopeLine: 32, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !4)
!10 = !DISubroutineType(types: !11)
!11 = !{null, !12, !13, !13}
!12 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!13 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!14 = !DICompositeType(tag: DW_TAG_array_type, baseType: !12, size: 32000, align: 32, elements: !15)
!15 = !{!16}
!16 = !DISubrange(count: 1000, lowerBound: 1)
!17 = !DISubroutineType(cc: DW_CC_program, types: !18)
!18 = !{null}
!19 = !{i32 2, !"Dwarf Version", i32 4}
!20 = !{i32 2, !"Debug Info Version", i32 3}
!21 = !DILocalVariable(name: "omp_sched_static", scope: !2, file: !3, type: !12)
!22 = !DILocation(line: 0, scope: !2)
!23 = !DILocalVariable(name: "omp_proc_bind_false", scope: !2, file: !3, type: !12)
!24 = !DILocalVariable(name: "omp_proc_bind_true", scope: !2, file: !3, type: !12)
!25 = !DILocalVariable(name: "omp_lock_hint_none", scope: !2, file: !3, type: !12)
!26 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !2, file: !3, type: !12)
!27 = !DILocation(line: 40, column: 1, scope: !2)
!28 = !DILocation(line: 10, column: 1, scope: !2)
!29 = !DILocalVariable(name: "len", scope: !2, file: !3, type: !12)
!30 = !DILocation(line: 18, column: 1, scope: !2)
!31 = !DILocation(line: 20, column: 1, scope: !2)
!32 = !DILocalVariable(name: "i", scope: !2, file: !3, type: !12)
!33 = !DILocation(line: 21, column: 1, scope: !2)
!34 = !DILocation(line: 22, column: 1, scope: !2)
!35 = !DILocation(line: 24, column: 1, scope: !2)
!36 = !DILocalVariable(name: "exist", scope: !2, file: !3, type: !37)
!37 = !DIBasicType(name: "logical", size: 32, align: 32, encoding: DW_ATE_boolean)
!38 = !DILocalVariable(scope: !2, file: !3, type: !12, flags: DIFlagArtificial)
!39 = !DILocation(line: 26, column: 1, scope: !2)
!40 = !DILocation(line: 27, column: 1, scope: !2)
!41 = !DILocalVariable(name: "stat", scope: !2, file: !3, type: !12)
!42 = !DILocation(line: 28, column: 1, scope: !2)
!43 = !DILocation(line: 29, column: 1, scope: !2)
!44 = !DILocation(line: 32, column: 1, scope: !2)
!45 = !DILocation(line: 38, column: 1, scope: !2)
!46 = !DILocalVariable(name: "__nv_MAIN__F1L32_1Arg0", arg: 1, scope: !9, file: !3, type: !12)
!47 = !DILocation(line: 0, scope: !9)
!48 = !DILocalVariable(name: "__nv_MAIN__F1L32_1Arg1", arg: 2, scope: !9, file: !3, type: !13)
!49 = !DILocalVariable(name: "__nv_MAIN__F1L32_1Arg2", arg: 3, scope: !9, file: !3, type: !13)
!50 = !DILocalVariable(name: "omp_sched_static", scope: !9, file: !3, type: !12)
!51 = !DILocalVariable(name: "omp_proc_bind_false", scope: !9, file: !3, type: !12)
!52 = !DILocalVariable(name: "omp_proc_bind_true", scope: !9, file: !3, type: !12)
!53 = !DILocalVariable(name: "omp_lock_hint_none", scope: !9, file: !3, type: !12)
!54 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !9, file: !3, type: !12)
!55 = !DILocation(line: 35, column: 1, scope: !9)
!56 = !DILocation(line: 33, column: 1, scope: !9)
!57 = !DILocalVariable(name: "i", scope: !9, file: !3, type: !12)
!58 = !DILocation(line: 34, column: 1, scope: !9)
!59 = !DILocalVariable(scope: !9, file: !3, type: !12, flags: DIFlagArtificial)

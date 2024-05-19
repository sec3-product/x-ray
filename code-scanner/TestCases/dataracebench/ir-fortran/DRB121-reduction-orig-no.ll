; ModuleID = '/tmp/DRB121-reduction-orig-no-63329f.ll'
source_filename = "/tmp/DRB121-reduction-orig-no-63329f.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.STATICS1 = type <{ [52 x i8] }>
%astruct.dt60 = type <{ i8*, i8*, i8* }>

@.STATICS1 = internal global %struct.STATICS1 <{ [52 x i8] c"\FB\FF\FF\FF\05\00\00\00var =\00\00\00\00\00\00\00\03\00\00\00\EB\FF\FF\FF\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\FF\FF\FF\FF\00\00\00\00" }>, align 16
@.C305_MAIN_ = internal constant i32 25
@.C284_MAIN_ = internal constant i64 0
@.C331_MAIN_ = internal constant i32 6
@.C327_MAIN_ = internal constant [53 x i8] c"micro-benchmarks-fortran/DRB121-reduction-orig-no.f95"
@.C329_MAIN_ = internal constant i32 38
@.C316_MAIN_ = internal constant i32 5
@.C285_MAIN_ = internal constant i32 1
@.C283_MAIN_ = internal constant i32 0
@.C316___nv_MAIN__F1L22_1 = internal constant i32 5
@.C285___nv_MAIN__F1L22_1 = internal constant i32 1
@.C283___nv_MAIN__F1L22_1 = internal constant i32 0

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__384 = alloca i32, align 4
  %var_306 = alloca i32, align 4
  %sum1_308 = alloca i32, align 4
  %sum2_309 = alloca i32, align 4
  %.uplevelArgPack0001_375 = alloca %astruct.dt60, align 16
  %z__io_333 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !14, metadata !DIExpression()), !dbg !10
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !15
  store i32 %0, i32* %__gtid_MAIN__384, align 4, !dbg !15
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !16
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !16
  call void (i8*, ...) %2(i8* %1), !dbg !16
  br label %L.LB1_367

L.LB1_367:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %var_306, metadata !17, metadata !DIExpression()), !dbg !10
  store i32 0, i32* %var_306, align 4, !dbg !18
  call void @llvm.dbg.declare(metadata i32* %sum1_308, metadata !19, metadata !DIExpression()), !dbg !10
  store i32 0, i32* %sum1_308, align 4, !dbg !20
  call void @llvm.dbg.declare(metadata i32* %sum2_309, metadata !21, metadata !DIExpression()), !dbg !10
  store i32 0, i32* %sum2_309, align 4, !dbg !22
  %3 = bitcast i32* %var_306 to i8*, !dbg !23
  %4 = bitcast %astruct.dt60* %.uplevelArgPack0001_375 to i8**, !dbg !23
  store i8* %3, i8** %4, align 8, !dbg !23
  %5 = bitcast i32* %sum1_308 to i8*, !dbg !23
  %6 = bitcast %astruct.dt60* %.uplevelArgPack0001_375 to i8*, !dbg !23
  %7 = getelementptr i8, i8* %6, i64 8, !dbg !23
  %8 = bitcast i8* %7 to i8**, !dbg !23
  store i8* %5, i8** %8, align 8, !dbg !23
  %9 = bitcast i32* %sum2_309 to i8*, !dbg !23
  %10 = bitcast %astruct.dt60* %.uplevelArgPack0001_375 to i8*, !dbg !23
  %11 = getelementptr i8, i8* %10, i64 16, !dbg !23
  %12 = bitcast i8* %11 to i8**, !dbg !23
  store i8* %9, i8** %12, align 8, !dbg !23
  br label %L.LB1_382, !dbg !23

L.LB1_382:                                        ; preds = %L.LB1_367
  %13 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L22_1_ to i64*, !dbg !23
  %14 = bitcast %astruct.dt60* %.uplevelArgPack0001_375 to i64*, !dbg !23
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %13, i64* %14), !dbg !23
  call void (...) @_mp_bcs_nest(), !dbg !24
  %15 = bitcast i32* @.C329_MAIN_ to i8*, !dbg !24
  %16 = bitcast [53 x i8]* @.C327_MAIN_ to i8*, !dbg !24
  %17 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !24
  call void (i8*, i8*, i64, ...) %17(i8* %15, i8* %16, i64 53), !dbg !24
  %18 = bitcast i32* @.C331_MAIN_ to i8*, !dbg !24
  %19 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !24
  %20 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !24
  %21 = bitcast %struct.STATICS1* @.STATICS1 to i8*, !dbg !24
  %22 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !24
  %23 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %22(i8* %18, i8* null, i8* %19, i8* %20, i8* %21, i8* null, i64 0), !dbg !24
  call void @llvm.dbg.declare(metadata i32* %z__io_333, metadata !25, metadata !DIExpression()), !dbg !10
  store i32 %23, i32* %z__io_333, align 4, !dbg !24
  %24 = load i32, i32* %var_306, align 4, !dbg !24
  call void @llvm.dbg.value(metadata i32 %24, metadata !17, metadata !DIExpression()), !dbg !10
  %25 = bitcast i32 (...)* @f90io_sc_i_fmt_write to i32 (i32, i32, ...)*, !dbg !24
  %26 = call i32 (i32, i32, ...) %25(i32 %24, i32 25), !dbg !24
  store i32 %26, i32* %z__io_333, align 4, !dbg !24
  %27 = call i32 (...) @f90io_fmtw_end(), !dbg !24
  store i32 %27, i32* %z__io_333, align 4, !dbg !24
  call void (...) @_mp_ecs_nest(), !dbg !24
  ret void, !dbg !15
}

define internal void @__nv_MAIN__F1L22_1_(i32* %__nv_MAIN__F1L22_1Arg0, i64* %__nv_MAIN__F1L22_1Arg1, i64* %__nv_MAIN__F1L22_1Arg2) #0 !dbg !26 {
L.entry:
  %__gtid___nv_MAIN__F1L22_1__429 = alloca i32, align 4
  %var_313 = alloca i32, align 4
  %sum1_315 = alloca i32, align 4
  %.i0000p_318 = alloca i32, align 4
  %i_317 = alloca i32, align 4
  %.du0001p_345 = alloca i32, align 4
  %.de0001p_346 = alloca i32, align 4
  %.di0001p_347 = alloca i32, align 4
  %.ds0001p_348 = alloca i32, align 4
  %.dl0001p_350 = alloca i32, align 4
  %.dl0001p.copy_423 = alloca i32, align 4
  %.de0001p.copy_424 = alloca i32, align 4
  %.ds0001p.copy_425 = alloca i32, align 4
  %.dX0001p_349 = alloca i32, align 4
  %.dY0001p_344 = alloca i32, align 4
  %sum2_321 = alloca i32, align 4
  %.i0001p_323 = alloca i32, align 4
  %i_322 = alloca i32, align 4
  %.du0002p_357 = alloca i32, align 4
  %.de0002p_358 = alloca i32, align 4
  %.di0002p_359 = alloca i32, align 4
  %.ds0002p_360 = alloca i32, align 4
  %.dl0002p_362 = alloca i32, align 4
  %.dl0002p.copy_462 = alloca i32, align 4
  %.de0002p.copy_463 = alloca i32, align 4
  %.ds0002p.copy_464 = alloca i32, align 4
  %.dX0002p_361 = alloca i32, align 4
  %.dY0002p_356 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L22_1Arg0, metadata !30, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L22_1Arg1, metadata !32, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L22_1Arg2, metadata !33, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.value(metadata i32 1, metadata !34, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.value(metadata i32 0, metadata !35, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.value(metadata i32 1, metadata !36, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.value(metadata i32 0, metadata !37, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.value(metadata i32 1, metadata !38, metadata !DIExpression()), !dbg !31
  %0 = load i32, i32* %__nv_MAIN__F1L22_1Arg0, align 4, !dbg !39
  store i32 %0, i32* %__gtid___nv_MAIN__F1L22_1__429, align 4, !dbg !39
  br label %L.LB2_413

L.LB2_413:                                        ; preds = %L.entry
  br label %L.LB2_312

L.LB2_312:                                        ; preds = %L.LB2_413
  call void @llvm.dbg.declare(metadata i32* %var_313, metadata !40, metadata !DIExpression()), !dbg !39
  store i32 0, i32* %var_313, align 4, !dbg !41
  br label %L.LB2_314

L.LB2_314:                                        ; preds = %L.LB2_312
  call void @llvm.dbg.declare(metadata i32* %sum1_315, metadata !42, metadata !DIExpression()), !dbg !39
  store i32 0, i32* %sum1_315, align 4, !dbg !43
  store i32 0, i32* %.i0000p_318, align 4, !dbg !44
  call void @llvm.dbg.declare(metadata i32* %i_317, metadata !45, metadata !DIExpression()), !dbg !39
  store i32 1, i32* %i_317, align 4, !dbg !44
  store i32 5, i32* %.du0001p_345, align 4, !dbg !44
  store i32 5, i32* %.de0001p_346, align 4, !dbg !44
  store i32 1, i32* %.di0001p_347, align 4, !dbg !44
  %1 = load i32, i32* %.di0001p_347, align 4, !dbg !44
  store i32 %1, i32* %.ds0001p_348, align 4, !dbg !44
  store i32 1, i32* %.dl0001p_350, align 4, !dbg !44
  %2 = load i32, i32* %.dl0001p_350, align 4, !dbg !44
  store i32 %2, i32* %.dl0001p.copy_423, align 4, !dbg !44
  %3 = load i32, i32* %.de0001p_346, align 4, !dbg !44
  store i32 %3, i32* %.de0001p.copy_424, align 4, !dbg !44
  %4 = load i32, i32* %.ds0001p_348, align 4, !dbg !44
  store i32 %4, i32* %.ds0001p.copy_425, align 4, !dbg !44
  %5 = load i32, i32* %__gtid___nv_MAIN__F1L22_1__429, align 4, !dbg !44
  %6 = bitcast i32* %.i0000p_318 to i64*, !dbg !44
  %7 = bitcast i32* %.dl0001p.copy_423 to i64*, !dbg !44
  %8 = bitcast i32* %.de0001p.copy_424 to i64*, !dbg !44
  %9 = bitcast i32* %.ds0001p.copy_425 to i64*, !dbg !44
  %10 = load i32, i32* %.ds0001p.copy_425, align 4, !dbg !44
  call void @__kmpc_for_static_init_4(i64* null, i32 %5, i32 34, i64* %6, i64* %7, i64* %8, i64* %9, i32 %10, i32 1), !dbg !44
  %11 = load i32, i32* %.dl0001p.copy_423, align 4, !dbg !44
  store i32 %11, i32* %.dl0001p_350, align 4, !dbg !44
  %12 = load i32, i32* %.de0001p.copy_424, align 4, !dbg !44
  store i32 %12, i32* %.de0001p_346, align 4, !dbg !44
  %13 = load i32, i32* %.ds0001p.copy_425, align 4, !dbg !44
  store i32 %13, i32* %.ds0001p_348, align 4, !dbg !44
  %14 = load i32, i32* %.dl0001p_350, align 4, !dbg !44
  store i32 %14, i32* %i_317, align 4, !dbg !44
  %15 = load i32, i32* %i_317, align 4, !dbg !44
  call void @llvm.dbg.value(metadata i32 %15, metadata !45, metadata !DIExpression()), !dbg !39
  store i32 %15, i32* %.dX0001p_349, align 4, !dbg !44
  %16 = load i32, i32* %.dX0001p_349, align 4, !dbg !44
  %17 = load i32, i32* %.du0001p_345, align 4, !dbg !44
  %18 = icmp sgt i32 %16, %17, !dbg !44
  br i1 %18, label %L.LB2_343, label %L.LB2_475, !dbg !44

L.LB2_475:                                        ; preds = %L.LB2_314
  %19 = load i32, i32* %.dX0001p_349, align 4, !dbg !44
  store i32 %19, i32* %i_317, align 4, !dbg !44
  %20 = load i32, i32* %.di0001p_347, align 4, !dbg !44
  %21 = load i32, i32* %.de0001p_346, align 4, !dbg !44
  %22 = load i32, i32* %.dX0001p_349, align 4, !dbg !44
  %23 = sub nsw i32 %21, %22, !dbg !44
  %24 = add nsw i32 %20, %23, !dbg !44
  %25 = load i32, i32* %.di0001p_347, align 4, !dbg !44
  %26 = sdiv i32 %24, %25, !dbg !44
  store i32 %26, i32* %.dY0001p_344, align 4, !dbg !44
  %27 = load i32, i32* %.dY0001p_344, align 4, !dbg !44
  %28 = icmp sle i32 %27, 0, !dbg !44
  br i1 %28, label %L.LB2_353, label %L.LB2_352, !dbg !44

L.LB2_352:                                        ; preds = %L.LB2_352, %L.LB2_475
  %29 = load i32, i32* %i_317, align 4, !dbg !46
  call void @llvm.dbg.value(metadata i32 %29, metadata !45, metadata !DIExpression()), !dbg !39
  %30 = load i32, i32* %sum1_315, align 4, !dbg !46
  call void @llvm.dbg.value(metadata i32 %30, metadata !42, metadata !DIExpression()), !dbg !39
  %31 = add nsw i32 %29, %30, !dbg !46
  store i32 %31, i32* %sum1_315, align 4, !dbg !46
  %32 = load i32, i32* %.di0001p_347, align 4, !dbg !47
  %33 = load i32, i32* %i_317, align 4, !dbg !47
  call void @llvm.dbg.value(metadata i32 %33, metadata !45, metadata !DIExpression()), !dbg !39
  %34 = add nsw i32 %32, %33, !dbg !47
  store i32 %34, i32* %i_317, align 4, !dbg !47
  %35 = load i32, i32* %.dY0001p_344, align 4, !dbg !47
  %36 = sub nsw i32 %35, 1, !dbg !47
  store i32 %36, i32* %.dY0001p_344, align 4, !dbg !47
  %37 = load i32, i32* %.dY0001p_344, align 4, !dbg !47
  %38 = icmp sgt i32 %37, 0, !dbg !47
  br i1 %38, label %L.LB2_352, label %L.LB2_353, !dbg !47

L.LB2_353:                                        ; preds = %L.LB2_352, %L.LB2_475
  br label %L.LB2_343

L.LB2_343:                                        ; preds = %L.LB2_353, %L.LB2_314
  %39 = load i32, i32* %__gtid___nv_MAIN__F1L22_1__429, align 4, !dbg !47
  call void @__kmpc_for_static_fini(i64* null, i32 %39), !dbg !47
  %40 = call i32 (...) @_mp_bcs_nest_red(), !dbg !47
  %41 = call i32 (...) @_mp_bcs_nest_red(), !dbg !47
  %42 = load i32, i32* %sum1_315, align 4, !dbg !47
  call void @llvm.dbg.value(metadata i32 %42, metadata !42, metadata !DIExpression()), !dbg !39
  %43 = bitcast i64* %__nv_MAIN__F1L22_1Arg2 to i8*, !dbg !47
  %44 = getelementptr i8, i8* %43, i64 8, !dbg !47
  %45 = bitcast i8* %44 to i32**, !dbg !47
  %46 = load i32*, i32** %45, align 8, !dbg !47
  %47 = load i32, i32* %46, align 4, !dbg !47
  %48 = add nsw i32 %42, %47, !dbg !47
  %49 = bitcast i64* %__nv_MAIN__F1L22_1Arg2 to i8*, !dbg !47
  %50 = getelementptr i8, i8* %49, i64 8, !dbg !47
  %51 = bitcast i8* %50 to i32**, !dbg !47
  %52 = load i32*, i32** %51, align 8, !dbg !47
  store i32 %48, i32* %52, align 4, !dbg !47
  %53 = call i32 (...) @_mp_ecs_nest_red(), !dbg !47
  %54 = call i32 (...) @_mp_ecs_nest_red(), !dbg !47
  br label %L.LB2_319

L.LB2_319:                                        ; preds = %L.LB2_343
  %55 = load i32, i32* %__gtid___nv_MAIN__F1L22_1__429, align 4, !dbg !48
  call void @__kmpc_barrier(i64* null, i32 %55), !dbg !48
  br label %L.LB2_320

L.LB2_320:                                        ; preds = %L.LB2_319
  call void @llvm.dbg.declare(metadata i32* %sum2_321, metadata !49, metadata !DIExpression()), !dbg !39
  store i32 0, i32* %sum2_321, align 4, !dbg !50
  store i32 0, i32* %.i0001p_323, align 4, !dbg !51
  call void @llvm.dbg.declare(metadata i32* %i_322, metadata !45, metadata !DIExpression()), !dbg !39
  store i32 1, i32* %i_322, align 4, !dbg !51
  store i32 5, i32* %.du0002p_357, align 4, !dbg !51
  store i32 5, i32* %.de0002p_358, align 4, !dbg !51
  store i32 1, i32* %.di0002p_359, align 4, !dbg !51
  %56 = load i32, i32* %.di0002p_359, align 4, !dbg !51
  store i32 %56, i32* %.ds0002p_360, align 4, !dbg !51
  store i32 1, i32* %.dl0002p_362, align 4, !dbg !51
  %57 = load i32, i32* %.dl0002p_362, align 4, !dbg !51
  store i32 %57, i32* %.dl0002p.copy_462, align 4, !dbg !51
  %58 = load i32, i32* %.de0002p_358, align 4, !dbg !51
  store i32 %58, i32* %.de0002p.copy_463, align 4, !dbg !51
  %59 = load i32, i32* %.ds0002p_360, align 4, !dbg !51
  store i32 %59, i32* %.ds0002p.copy_464, align 4, !dbg !51
  %60 = load i32, i32* %__gtid___nv_MAIN__F1L22_1__429, align 4, !dbg !51
  %61 = bitcast i32* %.i0001p_323 to i64*, !dbg !51
  %62 = bitcast i32* %.dl0002p.copy_462 to i64*, !dbg !51
  %63 = bitcast i32* %.de0002p.copy_463 to i64*, !dbg !51
  %64 = bitcast i32* %.ds0002p.copy_464 to i64*, !dbg !51
  %65 = load i32, i32* %.ds0002p.copy_464, align 4, !dbg !51
  call void @__kmpc_for_static_init_4(i64* null, i32 %60, i32 34, i64* %61, i64* %62, i64* %63, i64* %64, i32 %65, i32 1), !dbg !51
  %66 = load i32, i32* %.dl0002p.copy_462, align 4, !dbg !51
  store i32 %66, i32* %.dl0002p_362, align 4, !dbg !51
  %67 = load i32, i32* %.de0002p.copy_463, align 4, !dbg !51
  store i32 %67, i32* %.de0002p_358, align 4, !dbg !51
  %68 = load i32, i32* %.ds0002p.copy_464, align 4, !dbg !51
  store i32 %68, i32* %.ds0002p_360, align 4, !dbg !51
  %69 = load i32, i32* %.dl0002p_362, align 4, !dbg !51
  store i32 %69, i32* %i_322, align 4, !dbg !51
  %70 = load i32, i32* %i_322, align 4, !dbg !51
  call void @llvm.dbg.value(metadata i32 %70, metadata !45, metadata !DIExpression()), !dbg !39
  store i32 %70, i32* %.dX0002p_361, align 4, !dbg !51
  %71 = load i32, i32* %.dX0002p_361, align 4, !dbg !51
  %72 = load i32, i32* %.du0002p_357, align 4, !dbg !51
  %73 = icmp sgt i32 %71, %72, !dbg !51
  br i1 %73, label %L.LB2_355, label %L.LB2_476, !dbg !51

L.LB2_476:                                        ; preds = %L.LB2_320
  %74 = load i32, i32* %.dX0002p_361, align 4, !dbg !51
  store i32 %74, i32* %i_322, align 4, !dbg !51
  %75 = load i32, i32* %.di0002p_359, align 4, !dbg !51
  %76 = load i32, i32* %.de0002p_358, align 4, !dbg !51
  %77 = load i32, i32* %.dX0002p_361, align 4, !dbg !51
  %78 = sub nsw i32 %76, %77, !dbg !51
  %79 = add nsw i32 %75, %78, !dbg !51
  %80 = load i32, i32* %.di0002p_359, align 4, !dbg !51
  %81 = sdiv i32 %79, %80, !dbg !51
  store i32 %81, i32* %.dY0002p_356, align 4, !dbg !51
  %82 = load i32, i32* %.dY0002p_356, align 4, !dbg !51
  %83 = icmp sle i32 %82, 0, !dbg !51
  br i1 %83, label %L.LB2_365, label %L.LB2_364, !dbg !51

L.LB2_364:                                        ; preds = %L.LB2_364, %L.LB2_476
  %84 = load i32, i32* %i_322, align 4, !dbg !52
  call void @llvm.dbg.value(metadata i32 %84, metadata !45, metadata !DIExpression()), !dbg !39
  %85 = load i32, i32* %sum2_321, align 4, !dbg !52
  call void @llvm.dbg.value(metadata i32 %85, metadata !49, metadata !DIExpression()), !dbg !39
  %86 = add nsw i32 %84, %85, !dbg !52
  store i32 %86, i32* %sum2_321, align 4, !dbg !52
  %87 = load i32, i32* %.di0002p_359, align 4, !dbg !53
  %88 = load i32, i32* %i_322, align 4, !dbg !53
  call void @llvm.dbg.value(metadata i32 %88, metadata !45, metadata !DIExpression()), !dbg !39
  %89 = add nsw i32 %87, %88, !dbg !53
  store i32 %89, i32* %i_322, align 4, !dbg !53
  %90 = load i32, i32* %.dY0002p_356, align 4, !dbg !53
  %91 = sub nsw i32 %90, 1, !dbg !53
  store i32 %91, i32* %.dY0002p_356, align 4, !dbg !53
  %92 = load i32, i32* %.dY0002p_356, align 4, !dbg !53
  %93 = icmp sgt i32 %92, 0, !dbg !53
  br i1 %93, label %L.LB2_364, label %L.LB2_365, !dbg !53

L.LB2_365:                                        ; preds = %L.LB2_364, %L.LB2_476
  br label %L.LB2_355

L.LB2_355:                                        ; preds = %L.LB2_365, %L.LB2_320
  %94 = load i32, i32* %__gtid___nv_MAIN__F1L22_1__429, align 4, !dbg !53
  call void @__kmpc_for_static_fini(i64* null, i32 %94), !dbg !53
  %95 = call i32 (...) @_mp_bcs_nest_red(), !dbg !53
  %96 = call i32 (...) @_mp_bcs_nest_red(), !dbg !53
  %97 = load i32, i32* %sum2_321, align 4, !dbg !53
  call void @llvm.dbg.value(metadata i32 %97, metadata !49, metadata !DIExpression()), !dbg !39
  %98 = bitcast i64* %__nv_MAIN__F1L22_1Arg2 to i8*, !dbg !53
  %99 = getelementptr i8, i8* %98, i64 16, !dbg !53
  %100 = bitcast i8* %99 to i32**, !dbg !53
  %101 = load i32*, i32** %100, align 8, !dbg !53
  %102 = load i32, i32* %101, align 4, !dbg !53
  %103 = add nsw i32 %97, %102, !dbg !53
  %104 = bitcast i64* %__nv_MAIN__F1L22_1Arg2 to i8*, !dbg !53
  %105 = getelementptr i8, i8* %104, i64 16, !dbg !53
  %106 = bitcast i8* %105 to i32**, !dbg !53
  %107 = load i32*, i32** %106, align 8, !dbg !53
  store i32 %103, i32* %107, align 4, !dbg !53
  %108 = call i32 (...) @_mp_ecs_nest_red(), !dbg !53
  %109 = call i32 (...) @_mp_ecs_nest_red(), !dbg !53
  br label %L.LB2_324

L.LB2_324:                                        ; preds = %L.LB2_355
  %110 = load i32, i32* %__gtid___nv_MAIN__F1L22_1__429, align 4, !dbg !54
  call void @__kmpc_barrier(i64* null, i32 %110), !dbg !54
  %111 = bitcast i64* %__nv_MAIN__F1L22_1Arg2 to i8*, !dbg !55
  %112 = getelementptr i8, i8* %111, i64 8, !dbg !55
  %113 = bitcast i8* %112 to i32**, !dbg !55
  %114 = load i32*, i32** %113, align 8, !dbg !55
  %115 = load i32, i32* %114, align 4, !dbg !55
  %116 = bitcast i64* %__nv_MAIN__F1L22_1Arg2 to i8*, !dbg !55
  %117 = getelementptr i8, i8* %116, i64 16, !dbg !55
  %118 = bitcast i8* %117 to i32**, !dbg !55
  %119 = load i32*, i32** %118, align 8, !dbg !55
  %120 = load i32, i32* %119, align 4, !dbg !55
  %121 = add nsw i32 %115, %120, !dbg !55
  store i32 %121, i32* %var_313, align 4, !dbg !55
  %122 = call i32 (...) @_mp_bcs_nest_red(), !dbg !39
  %123 = call i32 (...) @_mp_bcs_nest_red(), !dbg !39
  %124 = load i32, i32* %var_313, align 4, !dbg !39
  call void @llvm.dbg.value(metadata i32 %124, metadata !40, metadata !DIExpression()), !dbg !39
  %125 = bitcast i64* %__nv_MAIN__F1L22_1Arg2 to i32**, !dbg !39
  %126 = load i32*, i32** %125, align 8, !dbg !39
  %127 = load i32, i32* %126, align 4, !dbg !39
  %128 = add nsw i32 %124, %127, !dbg !39
  %129 = bitcast i64* %__nv_MAIN__F1L22_1Arg2 to i32**, !dbg !39
  %130 = load i32*, i32** %129, align 8, !dbg !39
  store i32 %128, i32* %130, align 4, !dbg !39
  %131 = call i32 (...) @_mp_ecs_nest_red(), !dbg !39
  %132 = call i32 (...) @_mp_ecs_nest_red(), !dbg !39
  br label %L.LB2_325

L.LB2_325:                                        ; preds = %L.LB2_324
  ret void, !dbg !39
}

declare signext i32 @_mp_ecs_nest_red(...) #0

declare signext i32 @_mp_bcs_nest_red(...) #0

declare void @__kmpc_barrier(i64*, i32) #0

declare void @__kmpc_for_static_fini(i64*, i32) #0

declare void @__kmpc_for_static_init_4(i64*, i32, i32, i64*, i64*, i64*, i64*, i32, i32) #0

declare void @_mp_ecs_nest(...) #0

declare signext i32 @f90io_fmtw_end(...) #0

declare signext i32 @f90io_sc_i_fmt_write(...) #0

declare signext i32 @f90io_fmtw_inita(...) #0

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

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !4, globals: !4, imports: !4)
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB121-reduction-orig-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb121_reduction_orig_no", scope: !2, file: !3, line: 12, type: !6, scopeLine: 12, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!6 = !DISubroutineType(cc: DW_CC_program, types: !7)
!7 = !{null}
!8 = !DILocalVariable(name: "omp_sched_static", scope: !5, file: !3, type: !9)
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 0, scope: !5)
!11 = !DILocalVariable(name: "omp_proc_bind_false", scope: !5, file: !3, type: !9)
!12 = !DILocalVariable(name: "omp_proc_bind_true", scope: !5, file: !3, type: !9)
!13 = !DILocalVariable(name: "omp_lock_hint_none", scope: !5, file: !3, type: !9)
!14 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !5, file: !3, type: !9)
!15 = !DILocation(line: 40, column: 1, scope: !5)
!16 = !DILocation(line: 12, column: 1, scope: !5)
!17 = !DILocalVariable(name: "var", scope: !5, file: !3, type: !9)
!18 = !DILocation(line: 18, column: 1, scope: !5)
!19 = !DILocalVariable(name: "sum1", scope: !5, file: !3, type: !9)
!20 = !DILocation(line: 19, column: 1, scope: !5)
!21 = !DILocalVariable(name: "sum2", scope: !5, file: !3, type: !9)
!22 = !DILocation(line: 20, column: 1, scope: !5)
!23 = !DILocation(line: 22, column: 1, scope: !5)
!24 = !DILocation(line: 38, column: 1, scope: !5)
!25 = !DILocalVariable(scope: !5, file: !3, type: !9, flags: DIFlagArtificial)
!26 = distinct !DISubprogram(name: "__nv_MAIN__F1L22_1", scope: !2, file: !3, line: 22, type: !27, scopeLine: 22, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!27 = !DISubroutineType(types: !28)
!28 = !{null, !9, !29, !29}
!29 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!30 = !DILocalVariable(name: "__nv_MAIN__F1L22_1Arg0", arg: 1, scope: !26, file: !3, type: !9)
!31 = !DILocation(line: 0, scope: !26)
!32 = !DILocalVariable(name: "__nv_MAIN__F1L22_1Arg1", arg: 2, scope: !26, file: !3, type: !29)
!33 = !DILocalVariable(name: "__nv_MAIN__F1L22_1Arg2", arg: 3, scope: !26, file: !3, type: !29)
!34 = !DILocalVariable(name: "omp_sched_static", scope: !26, file: !3, type: !9)
!35 = !DILocalVariable(name: "omp_proc_bind_false", scope: !26, file: !3, type: !9)
!36 = !DILocalVariable(name: "omp_proc_bind_true", scope: !26, file: !3, type: !9)
!37 = !DILocalVariable(name: "omp_lock_hint_none", scope: !26, file: !3, type: !9)
!38 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !26, file: !3, type: !9)
!39 = !DILocation(line: 36, column: 1, scope: !26)
!40 = !DILocalVariable(name: "var", scope: !26, file: !3, type: !9)
!41 = !DILocation(line: 22, column: 1, scope: !26)
!42 = !DILocalVariable(name: "sum1", scope: !26, file: !3, type: !9)
!43 = !DILocation(line: 23, column: 1, scope: !26)
!44 = !DILocation(line: 24, column: 1, scope: !26)
!45 = !DILocalVariable(name: "i", scope: !26, file: !3, type: !9)
!46 = !DILocation(line: 25, column: 1, scope: !26)
!47 = !DILocation(line: 26, column: 1, scope: !26)
!48 = !DILocation(line: 27, column: 1, scope: !26)
!49 = !DILocalVariable(name: "sum2", scope: !26, file: !3, type: !9)
!50 = !DILocation(line: 29, column: 1, scope: !26)
!51 = !DILocation(line: 30, column: 1, scope: !26)
!52 = !DILocation(line: 31, column: 1, scope: !26)
!53 = !DILocation(line: 32, column: 1, scope: !26)
!54 = !DILocation(line: 33, column: 1, scope: !26)
!55 = !DILocation(line: 35, column: 1, scope: !26)

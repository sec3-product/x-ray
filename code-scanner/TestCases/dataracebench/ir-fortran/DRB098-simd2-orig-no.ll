; ModuleID = '/tmp/DRB098-simd2-orig-no-3646c9.ll'
source_filename = "/tmp/DRB098-simd2-orig-no-3646c9.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.C371_MAIN_ = internal constant i64 50
@.C308_MAIN_ = internal constant i32 14
@.C370_MAIN_ = internal constant [10 x i8] c"c(50,50) ="
@.C367_MAIN_ = internal constant i32 6
@.C364_MAIN_ = internal constant [49 x i8] c"micro-benchmarks-fortran/DRB098-simd2-orig-no.f95"
@.C366_MAIN_ = internal constant i32 41
@.C349_MAIN_ = internal constant double 7.000000e+00
@.C348_MAIN_ = internal constant double 3.000000e+00
@.C293_MAIN_ = internal constant double 2.000000e+00
@.C300_MAIN_ = internal constant i32 8
@.C285_MAIN_ = internal constant i32 1
@.C309_MAIN_ = internal constant i32 28
@.C381_MAIN_ = internal constant i64 8
@.C380_MAIN_ = internal constant i64 28
@.C343_MAIN_ = internal constant i32 100
@.C286_MAIN_ = internal constant i64 1
@.C284_MAIN_ = internal constant i64 0
@.C283_MAIN_ = internal constant i32 0

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %.Z0980_347 = alloca double*, align 8
  %"c$sd3_384" = alloca [22 x i64], align 8
  %.Z0970_346 = alloca double*, align 8
  %"b$sd2_383" = alloca [22 x i64], align 8
  %.Z0969_345 = alloca double*, align 8
  %"a$sd1_379" = alloca [22 x i64], align 8
  %len_344 = alloca i32, align 4
  %z_b_0_311 = alloca i64, align 8
  %z_b_1_312 = alloca i64, align 8
  %z_e_65_318 = alloca i64, align 8
  %z_b_3_314 = alloca i64, align 8
  %z_b_4_315 = alloca i64, align 8
  %z_e_68_319 = alloca i64, align 8
  %z_b_2_313 = alloca i64, align 8
  %z_b_5_316 = alloca i64, align 8
  %z_b_6_317 = alloca i64, align 8
  %z_b_7_322 = alloca i64, align 8
  %z_b_8_323 = alloca i64, align 8
  %z_e_78_329 = alloca i64, align 8
  %z_b_10_325 = alloca i64, align 8
  %z_b_11_326 = alloca i64, align 8
  %z_e_81_330 = alloca i64, align 8
  %z_b_9_324 = alloca i64, align 8
  %z_b_12_327 = alloca i64, align 8
  %z_b_13_328 = alloca i64, align 8
  %z_b_14_332 = alloca i64, align 8
  %z_b_15_333 = alloca i64, align 8
  %z_e_91_339 = alloca i64, align 8
  %z_b_17_335 = alloca i64, align 8
  %z_b_18_336 = alloca i64, align 8
  %z_e_94_340 = alloca i64, align 8
  %z_b_16_334 = alloca i64, align 8
  %z_b_19_337 = alloca i64, align 8
  %z_b_20_338 = alloca i64, align 8
  %.dY0001_392 = alloca i32, align 4
  %i_341 = alloca i32, align 4
  %.dY0002_395 = alloca i32, align 4
  %j_342 = alloca i32, align 4
  %.i0000_353 = alloca i32, align 4
  %.Xc0000_354 = alloca i64, align 8
  %.Xd0000_355 = alloca i64, align 8
  %.Xc0001_360 = alloca i64, align 8
  %.i0001_361 = alloca i32, align 4
  %.dY0003_398 = alloca i64, align 8
  %.id0000_356 = alloca i64, align 8
  %.Xg0000_359 = alloca i64, align 8
  %.Xe0000_357 = alloca i64, align 8
  %.Xf0000_358 = alloca i64, align 8
  %i_352 = alloca i32, align 4
  %z__io_369 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 8, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !14, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !15, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 8, metadata !16, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 8, metadata !17, metadata !DIExpression()), !dbg !10
  %0 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !18
  %1 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !18
  call void (i8*, ...) %1(i8* %0), !dbg !18
  call void @llvm.dbg.declare(metadata double** %.Z0980_347, metadata !19, metadata !DIExpression(DW_OP_deref)), !dbg !10
  %2 = bitcast double** %.Z0980_347 to i8**, !dbg !18
  store i8* null, i8** %2, align 8, !dbg !18
  call void @llvm.dbg.declare(metadata [22 x i64]* %"c$sd3_384", metadata !24, metadata !DIExpression()), !dbg !10
  %3 = bitcast [22 x i64]* %"c$sd3_384" to i64*, !dbg !18
  store i64 0, i64* %3, align 8, !dbg !18
  call void @llvm.dbg.declare(metadata double** %.Z0970_346, metadata !29, metadata !DIExpression(DW_OP_deref)), !dbg !10
  %4 = bitcast double** %.Z0970_346 to i8**, !dbg !18
  store i8* null, i8** %4, align 8, !dbg !18
  call void @llvm.dbg.declare(metadata [22 x i64]* %"b$sd2_383", metadata !24, metadata !DIExpression()), !dbg !10
  %5 = bitcast [22 x i64]* %"b$sd2_383" to i64*, !dbg !18
  store i64 0, i64* %5, align 8, !dbg !18
  call void @llvm.dbg.declare(metadata double** %.Z0969_345, metadata !30, metadata !DIExpression(DW_OP_deref)), !dbg !10
  %6 = bitcast double** %.Z0969_345 to i8**, !dbg !18
  store i8* null, i8** %6, align 8, !dbg !18
  call void @llvm.dbg.declare(metadata [22 x i64]* %"a$sd1_379", metadata !24, metadata !DIExpression()), !dbg !10
  %7 = bitcast [22 x i64]* %"a$sd1_379" to i64*, !dbg !18
  store i64 0, i64* %7, align 8, !dbg !18
  br label %L.LB1_408

L.LB1_408:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %len_344, metadata !31, metadata !DIExpression()), !dbg !10
  store i32 100, i32* %len_344, align 4, !dbg !32
  call void @llvm.dbg.declare(metadata i64* %z_b_0_311, metadata !33, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_0_311, align 8, !dbg !34
  %8 = load i32, i32* %len_344, align 4, !dbg !34
  call void @llvm.dbg.value(metadata i32 %8, metadata !31, metadata !DIExpression()), !dbg !10
  %9 = sext i32 %8 to i64, !dbg !34
  call void @llvm.dbg.declare(metadata i64* %z_b_1_312, metadata !33, metadata !DIExpression()), !dbg !10
  store i64 %9, i64* %z_b_1_312, align 8, !dbg !34
  %10 = load i64, i64* %z_b_1_312, align 8, !dbg !34
  call void @llvm.dbg.value(metadata i64 %10, metadata !33, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_65_318, metadata !33, metadata !DIExpression()), !dbg !10
  store i64 %10, i64* %z_e_65_318, align 8, !dbg !34
  call void @llvm.dbg.declare(metadata i64* %z_b_3_314, metadata !33, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_3_314, align 8, !dbg !34
  %11 = load i32, i32* %len_344, align 4, !dbg !34
  call void @llvm.dbg.value(metadata i32 %11, metadata !31, metadata !DIExpression()), !dbg !10
  %12 = sext i32 %11 to i64, !dbg !34
  call void @llvm.dbg.declare(metadata i64* %z_b_4_315, metadata !33, metadata !DIExpression()), !dbg !10
  store i64 %12, i64* %z_b_4_315, align 8, !dbg !34
  %13 = load i64, i64* %z_b_4_315, align 8, !dbg !34
  call void @llvm.dbg.value(metadata i64 %13, metadata !33, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_68_319, metadata !33, metadata !DIExpression()), !dbg !10
  store i64 %13, i64* %z_e_68_319, align 8, !dbg !34
  %14 = bitcast [22 x i64]* %"a$sd1_379" to i8*, !dbg !34
  %15 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !34
  %16 = bitcast i64* @.C380_MAIN_ to i8*, !dbg !34
  %17 = bitcast i64* @.C381_MAIN_ to i8*, !dbg !34
  %18 = bitcast i64* %z_b_0_311 to i8*, !dbg !34
  %19 = bitcast i64* %z_b_1_312 to i8*, !dbg !34
  %20 = bitcast i64* %z_b_3_314 to i8*, !dbg !34
  %21 = bitcast i64* %z_b_4_315 to i8*, !dbg !34
  %22 = bitcast void (...)* @f90_template2_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !34
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, ...) %22(i8* %14, i8* %15, i8* %16, i8* %17, i8* %18, i8* %19, i8* %20, i8* %21), !dbg !34
  %23 = bitcast [22 x i64]* %"a$sd1_379" to i8*, !dbg !34
  %24 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !34
  call void (i8*, i32, ...) %24(i8* %23, i32 28), !dbg !34
  %25 = load i64, i64* %z_b_1_312, align 8, !dbg !34
  call void @llvm.dbg.value(metadata i64 %25, metadata !33, metadata !DIExpression()), !dbg !10
  %26 = load i64, i64* %z_b_0_311, align 8, !dbg !34
  call void @llvm.dbg.value(metadata i64 %26, metadata !33, metadata !DIExpression()), !dbg !10
  %27 = sub nsw i64 %26, 1, !dbg !34
  %28 = sub nsw i64 %25, %27, !dbg !34
  call void @llvm.dbg.declare(metadata i64* %z_b_2_313, metadata !33, metadata !DIExpression()), !dbg !10
  store i64 %28, i64* %z_b_2_313, align 8, !dbg !34
  %29 = load i64, i64* %z_b_1_312, align 8, !dbg !34
  call void @llvm.dbg.value(metadata i64 %29, metadata !33, metadata !DIExpression()), !dbg !10
  %30 = load i64, i64* %z_b_0_311, align 8, !dbg !34
  call void @llvm.dbg.value(metadata i64 %30, metadata !33, metadata !DIExpression()), !dbg !10
  %31 = sub nsw i64 %30, 1, !dbg !34
  %32 = sub nsw i64 %29, %31, !dbg !34
  %33 = load i64, i64* %z_b_4_315, align 8, !dbg !34
  call void @llvm.dbg.value(metadata i64 %33, metadata !33, metadata !DIExpression()), !dbg !10
  %34 = load i64, i64* %z_b_3_314, align 8, !dbg !34
  call void @llvm.dbg.value(metadata i64 %34, metadata !33, metadata !DIExpression()), !dbg !10
  %35 = sub nsw i64 %34, 1, !dbg !34
  %36 = sub nsw i64 %33, %35, !dbg !34
  %37 = mul nsw i64 %32, %36, !dbg !34
  call void @llvm.dbg.declare(metadata i64* %z_b_5_316, metadata !33, metadata !DIExpression()), !dbg !10
  store i64 %37, i64* %z_b_5_316, align 8, !dbg !34
  %38 = load i64, i64* %z_b_0_311, align 8, !dbg !34
  call void @llvm.dbg.value(metadata i64 %38, metadata !33, metadata !DIExpression()), !dbg !10
  %39 = load i64, i64* %z_b_1_312, align 8, !dbg !34
  call void @llvm.dbg.value(metadata i64 %39, metadata !33, metadata !DIExpression()), !dbg !10
  %40 = load i64, i64* %z_b_0_311, align 8, !dbg !34
  call void @llvm.dbg.value(metadata i64 %40, metadata !33, metadata !DIExpression()), !dbg !10
  %41 = sub nsw i64 %40, 1, !dbg !34
  %42 = sub nsw i64 %39, %41, !dbg !34
  %43 = load i64, i64* %z_b_3_314, align 8, !dbg !34
  call void @llvm.dbg.value(metadata i64 %43, metadata !33, metadata !DIExpression()), !dbg !10
  %44 = mul nsw i64 %42, %43, !dbg !34
  %45 = add nsw i64 %38, %44, !dbg !34
  call void @llvm.dbg.declare(metadata i64* %z_b_6_317, metadata !33, metadata !DIExpression()), !dbg !10
  store i64 %45, i64* %z_b_6_317, align 8, !dbg !34
  %46 = bitcast i64* %z_b_5_316 to i8*, !dbg !34
  %47 = bitcast i64* @.C380_MAIN_ to i8*, !dbg !34
  %48 = bitcast i64* @.C381_MAIN_ to i8*, !dbg !34
  %49 = bitcast double** %.Z0969_345 to i8*, !dbg !34
  %50 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !34
  %51 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !34
  %52 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !34
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %52(i8* %46, i8* %47, i8* %48, i8* null, i8* %49, i8* null, i8* %50, i8* %51, i8* null, i64 0), !dbg !34
  call void @llvm.dbg.declare(metadata i64* %z_b_7_322, metadata !33, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_7_322, align 8, !dbg !35
  %53 = load i32, i32* %len_344, align 4, !dbg !35
  call void @llvm.dbg.value(metadata i32 %53, metadata !31, metadata !DIExpression()), !dbg !10
  %54 = sext i32 %53 to i64, !dbg !35
  call void @llvm.dbg.declare(metadata i64* %z_b_8_323, metadata !33, metadata !DIExpression()), !dbg !10
  store i64 %54, i64* %z_b_8_323, align 8, !dbg !35
  %55 = load i64, i64* %z_b_8_323, align 8, !dbg !35
  call void @llvm.dbg.value(metadata i64 %55, metadata !33, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_78_329, metadata !33, metadata !DIExpression()), !dbg !10
  store i64 %55, i64* %z_e_78_329, align 8, !dbg !35
  call void @llvm.dbg.declare(metadata i64* %z_b_10_325, metadata !33, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_10_325, align 8, !dbg !35
  %56 = load i32, i32* %len_344, align 4, !dbg !35
  call void @llvm.dbg.value(metadata i32 %56, metadata !31, metadata !DIExpression()), !dbg !10
  %57 = sext i32 %56 to i64, !dbg !35
  call void @llvm.dbg.declare(metadata i64* %z_b_11_326, metadata !33, metadata !DIExpression()), !dbg !10
  store i64 %57, i64* %z_b_11_326, align 8, !dbg !35
  %58 = load i64, i64* %z_b_11_326, align 8, !dbg !35
  call void @llvm.dbg.value(metadata i64 %58, metadata !33, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_81_330, metadata !33, metadata !DIExpression()), !dbg !10
  store i64 %58, i64* %z_e_81_330, align 8, !dbg !35
  %59 = bitcast [22 x i64]* %"b$sd2_383" to i8*, !dbg !35
  %60 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !35
  %61 = bitcast i64* @.C380_MAIN_ to i8*, !dbg !35
  %62 = bitcast i64* @.C381_MAIN_ to i8*, !dbg !35
  %63 = bitcast i64* %z_b_7_322 to i8*, !dbg !35
  %64 = bitcast i64* %z_b_8_323 to i8*, !dbg !35
  %65 = bitcast i64* %z_b_10_325 to i8*, !dbg !35
  %66 = bitcast i64* %z_b_11_326 to i8*, !dbg !35
  %67 = bitcast void (...)* @f90_template2_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !35
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, ...) %67(i8* %59, i8* %60, i8* %61, i8* %62, i8* %63, i8* %64, i8* %65, i8* %66), !dbg !35
  %68 = bitcast [22 x i64]* %"b$sd2_383" to i8*, !dbg !35
  %69 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !35
  call void (i8*, i32, ...) %69(i8* %68, i32 28), !dbg !35
  %70 = load i64, i64* %z_b_8_323, align 8, !dbg !35
  call void @llvm.dbg.value(metadata i64 %70, metadata !33, metadata !DIExpression()), !dbg !10
  %71 = load i64, i64* %z_b_7_322, align 8, !dbg !35
  call void @llvm.dbg.value(metadata i64 %71, metadata !33, metadata !DIExpression()), !dbg !10
  %72 = sub nsw i64 %71, 1, !dbg !35
  %73 = sub nsw i64 %70, %72, !dbg !35
  call void @llvm.dbg.declare(metadata i64* %z_b_9_324, metadata !33, metadata !DIExpression()), !dbg !10
  store i64 %73, i64* %z_b_9_324, align 8, !dbg !35
  %74 = load i64, i64* %z_b_8_323, align 8, !dbg !35
  call void @llvm.dbg.value(metadata i64 %74, metadata !33, metadata !DIExpression()), !dbg !10
  %75 = load i64, i64* %z_b_7_322, align 8, !dbg !35
  call void @llvm.dbg.value(metadata i64 %75, metadata !33, metadata !DIExpression()), !dbg !10
  %76 = sub nsw i64 %75, 1, !dbg !35
  %77 = sub nsw i64 %74, %76, !dbg !35
  %78 = load i64, i64* %z_b_11_326, align 8, !dbg !35
  call void @llvm.dbg.value(metadata i64 %78, metadata !33, metadata !DIExpression()), !dbg !10
  %79 = load i64, i64* %z_b_10_325, align 8, !dbg !35
  call void @llvm.dbg.value(metadata i64 %79, metadata !33, metadata !DIExpression()), !dbg !10
  %80 = sub nsw i64 %79, 1, !dbg !35
  %81 = sub nsw i64 %78, %80, !dbg !35
  %82 = mul nsw i64 %77, %81, !dbg !35
  call void @llvm.dbg.declare(metadata i64* %z_b_12_327, metadata !33, metadata !DIExpression()), !dbg !10
  store i64 %82, i64* %z_b_12_327, align 8, !dbg !35
  %83 = load i64, i64* %z_b_7_322, align 8, !dbg !35
  call void @llvm.dbg.value(metadata i64 %83, metadata !33, metadata !DIExpression()), !dbg !10
  %84 = load i64, i64* %z_b_8_323, align 8, !dbg !35
  call void @llvm.dbg.value(metadata i64 %84, metadata !33, metadata !DIExpression()), !dbg !10
  %85 = load i64, i64* %z_b_7_322, align 8, !dbg !35
  call void @llvm.dbg.value(metadata i64 %85, metadata !33, metadata !DIExpression()), !dbg !10
  %86 = sub nsw i64 %85, 1, !dbg !35
  %87 = sub nsw i64 %84, %86, !dbg !35
  %88 = load i64, i64* %z_b_10_325, align 8, !dbg !35
  call void @llvm.dbg.value(metadata i64 %88, metadata !33, metadata !DIExpression()), !dbg !10
  %89 = mul nsw i64 %87, %88, !dbg !35
  %90 = add nsw i64 %83, %89, !dbg !35
  call void @llvm.dbg.declare(metadata i64* %z_b_13_328, metadata !33, metadata !DIExpression()), !dbg !10
  store i64 %90, i64* %z_b_13_328, align 8, !dbg !35
  %91 = bitcast i64* %z_b_12_327 to i8*, !dbg !35
  %92 = bitcast i64* @.C380_MAIN_ to i8*, !dbg !35
  %93 = bitcast i64* @.C381_MAIN_ to i8*, !dbg !35
  %94 = bitcast double** %.Z0970_346 to i8*, !dbg !35
  %95 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !35
  %96 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !35
  %97 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !35
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %97(i8* %91, i8* %92, i8* %93, i8* null, i8* %94, i8* null, i8* %95, i8* %96, i8* null, i64 0), !dbg !35
  call void @llvm.dbg.declare(metadata i64* %z_b_14_332, metadata !33, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_14_332, align 8, !dbg !36
  %98 = load i32, i32* %len_344, align 4, !dbg !36
  call void @llvm.dbg.value(metadata i32 %98, metadata !31, metadata !DIExpression()), !dbg !10
  %99 = sext i32 %98 to i64, !dbg !36
  call void @llvm.dbg.declare(metadata i64* %z_b_15_333, metadata !33, metadata !DIExpression()), !dbg !10
  store i64 %99, i64* %z_b_15_333, align 8, !dbg !36
  %100 = load i64, i64* %z_b_15_333, align 8, !dbg !36
  call void @llvm.dbg.value(metadata i64 %100, metadata !33, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_91_339, metadata !33, metadata !DIExpression()), !dbg !10
  store i64 %100, i64* %z_e_91_339, align 8, !dbg !36
  call void @llvm.dbg.declare(metadata i64* %z_b_17_335, metadata !33, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_17_335, align 8, !dbg !36
  %101 = load i32, i32* %len_344, align 4, !dbg !36
  call void @llvm.dbg.value(metadata i32 %101, metadata !31, metadata !DIExpression()), !dbg !10
  %102 = sext i32 %101 to i64, !dbg !36
  call void @llvm.dbg.declare(metadata i64* %z_b_18_336, metadata !33, metadata !DIExpression()), !dbg !10
  store i64 %102, i64* %z_b_18_336, align 8, !dbg !36
  %103 = load i64, i64* %z_b_18_336, align 8, !dbg !36
  call void @llvm.dbg.value(metadata i64 %103, metadata !33, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_94_340, metadata !33, metadata !DIExpression()), !dbg !10
  store i64 %103, i64* %z_e_94_340, align 8, !dbg !36
  %104 = bitcast [22 x i64]* %"c$sd3_384" to i8*, !dbg !36
  %105 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !36
  %106 = bitcast i64* @.C380_MAIN_ to i8*, !dbg !36
  %107 = bitcast i64* @.C381_MAIN_ to i8*, !dbg !36
  %108 = bitcast i64* %z_b_14_332 to i8*, !dbg !36
  %109 = bitcast i64* %z_b_15_333 to i8*, !dbg !36
  %110 = bitcast i64* %z_b_17_335 to i8*, !dbg !36
  %111 = bitcast i64* %z_b_18_336 to i8*, !dbg !36
  %112 = bitcast void (...)* @f90_template2_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !36
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, ...) %112(i8* %104, i8* %105, i8* %106, i8* %107, i8* %108, i8* %109, i8* %110, i8* %111), !dbg !36
  %113 = bitcast [22 x i64]* %"c$sd3_384" to i8*, !dbg !36
  %114 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !36
  call void (i8*, i32, ...) %114(i8* %113, i32 28), !dbg !36
  %115 = load i64, i64* %z_b_15_333, align 8, !dbg !36
  call void @llvm.dbg.value(metadata i64 %115, metadata !33, metadata !DIExpression()), !dbg !10
  %116 = load i64, i64* %z_b_14_332, align 8, !dbg !36
  call void @llvm.dbg.value(metadata i64 %116, metadata !33, metadata !DIExpression()), !dbg !10
  %117 = sub nsw i64 %116, 1, !dbg !36
  %118 = sub nsw i64 %115, %117, !dbg !36
  call void @llvm.dbg.declare(metadata i64* %z_b_16_334, metadata !33, metadata !DIExpression()), !dbg !10
  store i64 %118, i64* %z_b_16_334, align 8, !dbg !36
  %119 = load i64, i64* %z_b_15_333, align 8, !dbg !36
  call void @llvm.dbg.value(metadata i64 %119, metadata !33, metadata !DIExpression()), !dbg !10
  %120 = load i64, i64* %z_b_14_332, align 8, !dbg !36
  call void @llvm.dbg.value(metadata i64 %120, metadata !33, metadata !DIExpression()), !dbg !10
  %121 = sub nsw i64 %120, 1, !dbg !36
  %122 = sub nsw i64 %119, %121, !dbg !36
  %123 = load i64, i64* %z_b_18_336, align 8, !dbg !36
  call void @llvm.dbg.value(metadata i64 %123, metadata !33, metadata !DIExpression()), !dbg !10
  %124 = load i64, i64* %z_b_17_335, align 8, !dbg !36
  call void @llvm.dbg.value(metadata i64 %124, metadata !33, metadata !DIExpression()), !dbg !10
  %125 = sub nsw i64 %124, 1, !dbg !36
  %126 = sub nsw i64 %123, %125, !dbg !36
  %127 = mul nsw i64 %122, %126, !dbg !36
  call void @llvm.dbg.declare(metadata i64* %z_b_19_337, metadata !33, metadata !DIExpression()), !dbg !10
  store i64 %127, i64* %z_b_19_337, align 8, !dbg !36
  %128 = load i64, i64* %z_b_14_332, align 8, !dbg !36
  call void @llvm.dbg.value(metadata i64 %128, metadata !33, metadata !DIExpression()), !dbg !10
  %129 = load i64, i64* %z_b_15_333, align 8, !dbg !36
  call void @llvm.dbg.value(metadata i64 %129, metadata !33, metadata !DIExpression()), !dbg !10
  %130 = load i64, i64* %z_b_14_332, align 8, !dbg !36
  call void @llvm.dbg.value(metadata i64 %130, metadata !33, metadata !DIExpression()), !dbg !10
  %131 = sub nsw i64 %130, 1, !dbg !36
  %132 = sub nsw i64 %129, %131, !dbg !36
  %133 = load i64, i64* %z_b_17_335, align 8, !dbg !36
  call void @llvm.dbg.value(metadata i64 %133, metadata !33, metadata !DIExpression()), !dbg !10
  %134 = mul nsw i64 %132, %133, !dbg !36
  %135 = add nsw i64 %128, %134, !dbg !36
  call void @llvm.dbg.declare(metadata i64* %z_b_20_338, metadata !33, metadata !DIExpression()), !dbg !10
  store i64 %135, i64* %z_b_20_338, align 8, !dbg !36
  %136 = bitcast i64* %z_b_19_337 to i8*, !dbg !36
  %137 = bitcast i64* @.C380_MAIN_ to i8*, !dbg !36
  %138 = bitcast i64* @.C381_MAIN_ to i8*, !dbg !36
  %139 = bitcast double** %.Z0980_347 to i8*, !dbg !36
  %140 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !36
  %141 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !36
  %142 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !36
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %142(i8* %136, i8* %137, i8* %138, i8* null, i8* %139, i8* null, i8* %140, i8* %141, i8* null, i64 0), !dbg !36
  %143 = load i32, i32* %len_344, align 4, !dbg !37
  call void @llvm.dbg.value(metadata i32 %143, metadata !31, metadata !DIExpression()), !dbg !10
  store i32 %143, i32* %.dY0001_392, align 4, !dbg !37
  call void @llvm.dbg.declare(metadata i32* %i_341, metadata !38, metadata !DIExpression()), !dbg !10
  store i32 1, i32* %i_341, align 4, !dbg !37
  %144 = load i32, i32* %.dY0001_392, align 4, !dbg !37
  %145 = icmp sle i32 %144, 0, !dbg !37
  br i1 %145, label %L.LB1_391, label %L.LB1_390, !dbg !37

L.LB1_390:                                        ; preds = %L.LB1_394, %L.LB1_408
  %146 = load i32, i32* %len_344, align 4, !dbg !39
  call void @llvm.dbg.value(metadata i32 %146, metadata !31, metadata !DIExpression()), !dbg !10
  store i32 %146, i32* %.dY0002_395, align 4, !dbg !39
  call void @llvm.dbg.declare(metadata i32* %j_342, metadata !40, metadata !DIExpression()), !dbg !10
  store i32 1, i32* %j_342, align 4, !dbg !39
  %147 = load i32, i32* %.dY0002_395, align 4, !dbg !39
  %148 = icmp sle i32 %147, 0, !dbg !39
  br i1 %148, label %L.LB1_394, label %L.LB1_393, !dbg !39

L.LB1_393:                                        ; preds = %L.LB1_393, %L.LB1_390
  %149 = load i32, i32* %i_341, align 4, !dbg !41
  call void @llvm.dbg.value(metadata i32 %149, metadata !38, metadata !DIExpression()), !dbg !10
  %150 = sitofp i32 %149 to double, !dbg !41
  %151 = fdiv fast double %150, 2.000000e+00, !dbg !41
  %152 = load i32, i32* %i_341, align 4, !dbg !41
  call void @llvm.dbg.value(metadata i32 %152, metadata !38, metadata !DIExpression()), !dbg !10
  %153 = sext i32 %152 to i64, !dbg !41
  %154 = load i32, i32* %j_342, align 4, !dbg !41
  call void @llvm.dbg.value(metadata i32 %154, metadata !40, metadata !DIExpression()), !dbg !10
  %155 = sext i32 %154 to i64, !dbg !41
  %156 = bitcast [22 x i64]* %"a$sd1_379" to i8*, !dbg !41
  %157 = getelementptr i8, i8* %156, i64 160, !dbg !41
  %158 = bitcast i8* %157 to i64*, !dbg !41
  %159 = load i64, i64* %158, align 8, !dbg !41
  %160 = mul nsw i64 %155, %159, !dbg !41
  %161 = add nsw i64 %153, %160, !dbg !41
  %162 = bitcast [22 x i64]* %"a$sd1_379" to i8*, !dbg !41
  %163 = getelementptr i8, i8* %162, i64 56, !dbg !41
  %164 = bitcast i8* %163 to i64*, !dbg !41
  %165 = load i64, i64* %164, align 8, !dbg !41
  %166 = add nsw i64 %161, %165, !dbg !41
  %167 = load double*, double** %.Z0969_345, align 8, !dbg !41
  call void @llvm.dbg.value(metadata double* %167, metadata !30, metadata !DIExpression()), !dbg !10
  %168 = bitcast double* %167 to i8*, !dbg !41
  %169 = getelementptr i8, i8* %168, i64 -8, !dbg !41
  %170 = bitcast i8* %169 to double*, !dbg !41
  %171 = getelementptr double, double* %170, i64 %166, !dbg !41
  store double %151, double* %171, align 8, !dbg !41
  %172 = load i32, i32* %i_341, align 4, !dbg !42
  call void @llvm.dbg.value(metadata i32 %172, metadata !38, metadata !DIExpression()), !dbg !10
  %173 = sitofp i32 %172 to double, !dbg !42
  %174 = fdiv fast double %173, 3.000000e+00, !dbg !42
  %175 = load i32, i32* %i_341, align 4, !dbg !42
  call void @llvm.dbg.value(metadata i32 %175, metadata !38, metadata !DIExpression()), !dbg !10
  %176 = sext i32 %175 to i64, !dbg !42
  %177 = load i32, i32* %j_342, align 4, !dbg !42
  call void @llvm.dbg.value(metadata i32 %177, metadata !40, metadata !DIExpression()), !dbg !10
  %178 = sext i32 %177 to i64, !dbg !42
  %179 = bitcast [22 x i64]* %"b$sd2_383" to i8*, !dbg !42
  %180 = getelementptr i8, i8* %179, i64 160, !dbg !42
  %181 = bitcast i8* %180 to i64*, !dbg !42
  %182 = load i64, i64* %181, align 8, !dbg !42
  %183 = mul nsw i64 %178, %182, !dbg !42
  %184 = add nsw i64 %176, %183, !dbg !42
  %185 = bitcast [22 x i64]* %"b$sd2_383" to i8*, !dbg !42
  %186 = getelementptr i8, i8* %185, i64 56, !dbg !42
  %187 = bitcast i8* %186 to i64*, !dbg !42
  %188 = load i64, i64* %187, align 8, !dbg !42
  %189 = add nsw i64 %184, %188, !dbg !42
  %190 = load double*, double** %.Z0970_346, align 8, !dbg !42
  call void @llvm.dbg.value(metadata double* %190, metadata !29, metadata !DIExpression()), !dbg !10
  %191 = bitcast double* %190 to i8*, !dbg !42
  %192 = getelementptr i8, i8* %191, i64 -8, !dbg !42
  %193 = bitcast i8* %192 to double*, !dbg !42
  %194 = getelementptr double, double* %193, i64 %189, !dbg !42
  store double %174, double* %194, align 8, !dbg !42
  %195 = load i32, i32* %i_341, align 4, !dbg !43
  call void @llvm.dbg.value(metadata i32 %195, metadata !38, metadata !DIExpression()), !dbg !10
  %196 = sitofp i32 %195 to double, !dbg !43
  %197 = fdiv fast double %196, 7.000000e+00, !dbg !43
  %198 = load i32, i32* %i_341, align 4, !dbg !43
  call void @llvm.dbg.value(metadata i32 %198, metadata !38, metadata !DIExpression()), !dbg !10
  %199 = sext i32 %198 to i64, !dbg !43
  %200 = load i32, i32* %j_342, align 4, !dbg !43
  call void @llvm.dbg.value(metadata i32 %200, metadata !40, metadata !DIExpression()), !dbg !10
  %201 = sext i32 %200 to i64, !dbg !43
  %202 = bitcast [22 x i64]* %"c$sd3_384" to i8*, !dbg !43
  %203 = getelementptr i8, i8* %202, i64 160, !dbg !43
  %204 = bitcast i8* %203 to i64*, !dbg !43
  %205 = load i64, i64* %204, align 8, !dbg !43
  %206 = mul nsw i64 %201, %205, !dbg !43
  %207 = add nsw i64 %199, %206, !dbg !43
  %208 = bitcast [22 x i64]* %"c$sd3_384" to i8*, !dbg !43
  %209 = getelementptr i8, i8* %208, i64 56, !dbg !43
  %210 = bitcast i8* %209 to i64*, !dbg !43
  %211 = load i64, i64* %210, align 8, !dbg !43
  %212 = add nsw i64 %207, %211, !dbg !43
  %213 = load double*, double** %.Z0980_347, align 8, !dbg !43
  call void @llvm.dbg.value(metadata double* %213, metadata !19, metadata !DIExpression()), !dbg !10
  %214 = bitcast double* %213 to i8*, !dbg !43
  %215 = getelementptr i8, i8* %214, i64 -8, !dbg !43
  %216 = bitcast i8* %215 to double*, !dbg !43
  %217 = getelementptr double, double* %216, i64 %212, !dbg !43
  store double %197, double* %217, align 8, !dbg !43
  %218 = load i32, i32* %j_342, align 4, !dbg !44
  call void @llvm.dbg.value(metadata i32 %218, metadata !40, metadata !DIExpression()), !dbg !10
  %219 = add nsw i32 %218, 1, !dbg !44
  store i32 %219, i32* %j_342, align 4, !dbg !44
  %220 = load i32, i32* %.dY0002_395, align 4, !dbg !44
  %221 = sub nsw i32 %220, 1, !dbg !44
  store i32 %221, i32* %.dY0002_395, align 4, !dbg !44
  %222 = load i32, i32* %.dY0002_395, align 4, !dbg !44
  %223 = icmp sgt i32 %222, 0, !dbg !44
  br i1 %223, label %L.LB1_393, label %L.LB1_394, !dbg !44

L.LB1_394:                                        ; preds = %L.LB1_393, %L.LB1_390
  %224 = load i32, i32* %i_341, align 4, !dbg !45
  call void @llvm.dbg.value(metadata i32 %224, metadata !38, metadata !DIExpression()), !dbg !10
  %225 = add nsw i32 %224, 1, !dbg !45
  store i32 %225, i32* %i_341, align 4, !dbg !45
  %226 = load i32, i32* %.dY0001_392, align 4, !dbg !45
  %227 = sub nsw i32 %226, 1, !dbg !45
  store i32 %227, i32* %.dY0001_392, align 4, !dbg !45
  %228 = load i32, i32* %.dY0001_392, align 4, !dbg !45
  %229 = icmp sgt i32 %228, 0, !dbg !45
  br i1 %229, label %L.LB1_390, label %L.LB1_391, !dbg !45

L.LB1_391:                                        ; preds = %L.LB1_394, %L.LB1_408
  br label %L.LB1_351

L.LB1_351:                                        ; preds = %L.LB1_391
  %230 = load i32, i32* %len_344, align 4, !dbg !46
  call void @llvm.dbg.value(metadata i32 %230, metadata !31, metadata !DIExpression()), !dbg !10
  %231 = add nsw i32 %230, 1, !dbg !46
  store i32 %231, i32* %.i0000_353, align 4, !dbg !46
  %232 = load i32, i32* %len_344, align 4, !dbg !46
  call void @llvm.dbg.value(metadata i32 %232, metadata !31, metadata !DIExpression()), !dbg !10
  %233 = sext i32 %232 to i64, !dbg !46
  store i64 %233, i64* %.Xc0000_354, align 8, !dbg !46
  %234 = load i64, i64* %.Xc0000_354, align 8, !dbg !46
  store i64 %234, i64* %.Xd0000_355, align 8, !dbg !46
  %235 = load i32, i32* %len_344, align 4, !dbg !47
  call void @llvm.dbg.value(metadata i32 %235, metadata !31, metadata !DIExpression()), !dbg !10
  %236 = sext i32 %235 to i64, !dbg !47
  store i64 %236, i64* %.Xc0001_360, align 8, !dbg !47
  %237 = load i64, i64* %.Xc0001_360, align 8, !dbg !47
  %238 = load i64, i64* %.Xd0000_355, align 8, !dbg !47
  %239 = mul nsw i64 %237, %238, !dbg !47
  store i64 %239, i64* %.Xd0000_355, align 8, !dbg !47
  store i32 0, i32* %.i0001_361, align 4, !dbg !47
  %240 = load i64, i64* %.Xd0000_355, align 8, !dbg !47
  store i64 %240, i64* %.dY0003_398, align 8, !dbg !47
  store i64 1, i64* %.id0000_356, align 8, !dbg !47
  %241 = load i64, i64* %.dY0003_398, align 8, !dbg !47
  %242 = icmp sle i64 %241, 0, !dbg !47
  br i1 %242, label %L.LB1_397, label %L.LB1_396, !dbg !47

L.LB1_396:                                        ; preds = %L.LB1_396, %L.LB1_351
  %243 = load i64, i64* %.id0000_356, align 8, !dbg !47
  %244 = sub nsw i64 %243, 1, !dbg !47
  store i64 %244, i64* %.Xg0000_359, align 8, !dbg !47
  %245 = load i64, i64* %.Xg0000_359, align 8, !dbg !47
  %246 = load i64, i64* %.Xc0001_360, align 8, !dbg !47
  %247 = sdiv i64 %245, %246, !dbg !47
  store i64 %247, i64* %.Xe0000_357, align 8, !dbg !47
  %248 = load i64, i64* %.Xg0000_359, align 8, !dbg !47
  %249 = load i64, i64* %.Xc0001_360, align 8, !dbg !47
  %250 = load i64, i64* %.Xe0000_357, align 8, !dbg !47
  %251 = mul nsw i64 %249, %250, !dbg !47
  %252 = sub nsw i64 %248, %251, !dbg !47
  store i64 %252, i64* %.Xf0000_358, align 8, !dbg !47
  %253 = load i64, i64* %.Xf0000_358, align 8, !dbg !47
  %254 = trunc i64 %253 to i32, !dbg !47
  %255 = add nsw i32 %254, 1, !dbg !47
  store i32 %255, i32* %j_342, align 4, !dbg !47
  %256 = load i64, i64* %.Xe0000_357, align 8, !dbg !47
  store i64 %256, i64* %.Xg0000_359, align 8, !dbg !47
  %257 = load i64, i64* %.Xg0000_359, align 8, !dbg !47
  %258 = load i64, i64* %.Xc0000_354, align 8, !dbg !47
  %259 = sdiv i64 %257, %258, !dbg !47
  store i64 %259, i64* %.Xe0000_357, align 8, !dbg !47
  %260 = load i64, i64* %.Xg0000_359, align 8, !dbg !47
  %261 = load i64, i64* %.Xc0000_354, align 8, !dbg !47
  %262 = load i64, i64* %.Xe0000_357, align 8, !dbg !47
  %263 = mul nsw i64 %261, %262, !dbg !47
  %264 = sub nsw i64 %260, %263, !dbg !47
  store i64 %264, i64* %.Xf0000_358, align 8, !dbg !47
  %265 = load i64, i64* %.Xf0000_358, align 8, !dbg !47
  %266 = trunc i64 %265 to i32, !dbg !47
  %267 = add nsw i32 %266, 1, !dbg !47
  call void @llvm.dbg.declare(metadata i32* %i_352, metadata !38, metadata !DIExpression()), !dbg !48
  store i32 %267, i32* %i_352, align 4, !dbg !47
  %268 = bitcast [22 x i64]* %"b$sd2_383" to i8*, !dbg !49
  %269 = getelementptr i8, i8* %268, i64 56, !dbg !49
  %270 = bitcast i8* %269 to i64*, !dbg !49
  %271 = load i64, i64* %270, align 8, !dbg !49
  %272 = load i32, i32* %j_342, align 4, !dbg !49
  call void @llvm.dbg.value(metadata i32 %272, metadata !40, metadata !DIExpression()), !dbg !10
  %273 = sext i32 %272 to i64, !dbg !49
  %274 = bitcast [22 x i64]* %"b$sd2_383" to i8*, !dbg !49
  %275 = getelementptr i8, i8* %274, i64 160, !dbg !49
  %276 = bitcast i8* %275 to i64*, !dbg !49
  %277 = load i64, i64* %276, align 8, !dbg !49
  %278 = mul nsw i64 %273, %277, !dbg !49
  %279 = load i32, i32* %i_352, align 4, !dbg !49
  call void @llvm.dbg.value(metadata i32 %279, metadata !38, metadata !DIExpression()), !dbg !48
  %280 = sext i32 %279 to i64, !dbg !49
  %281 = add nsw i64 %278, %280, !dbg !49
  %282 = add nsw i64 %271, %281, !dbg !49
  %283 = load double*, double** %.Z0970_346, align 8, !dbg !49
  call void @llvm.dbg.value(metadata double* %283, metadata !29, metadata !DIExpression()), !dbg !10
  %284 = bitcast double* %283 to i8*, !dbg !49
  %285 = getelementptr i8, i8* %284, i64 -8, !dbg !49
  %286 = bitcast i8* %285 to double*, !dbg !49
  %287 = getelementptr double, double* %286, i64 %282, !dbg !49
  %288 = load double, double* %287, align 8, !dbg !49
  %289 = bitcast [22 x i64]* %"a$sd1_379" to i8*, !dbg !49
  %290 = getelementptr i8, i8* %289, i64 56, !dbg !49
  %291 = bitcast i8* %290 to i64*, !dbg !49
  %292 = load i64, i64* %291, align 8, !dbg !49
  %293 = load i32, i32* %j_342, align 4, !dbg !49
  call void @llvm.dbg.value(metadata i32 %293, metadata !40, metadata !DIExpression()), !dbg !10
  %294 = sext i32 %293 to i64, !dbg !49
  %295 = bitcast [22 x i64]* %"a$sd1_379" to i8*, !dbg !49
  %296 = getelementptr i8, i8* %295, i64 160, !dbg !49
  %297 = bitcast i8* %296 to i64*, !dbg !49
  %298 = load i64, i64* %297, align 8, !dbg !49
  %299 = mul nsw i64 %294, %298, !dbg !49
  %300 = load i32, i32* %i_352, align 4, !dbg !49
  call void @llvm.dbg.value(metadata i32 %300, metadata !38, metadata !DIExpression()), !dbg !48
  %301 = sext i32 %300 to i64, !dbg !49
  %302 = add nsw i64 %299, %301, !dbg !49
  %303 = add nsw i64 %292, %302, !dbg !49
  %304 = load double*, double** %.Z0969_345, align 8, !dbg !49
  call void @llvm.dbg.value(metadata double* %304, metadata !30, metadata !DIExpression()), !dbg !10
  %305 = bitcast double* %304 to i8*, !dbg !49
  %306 = getelementptr i8, i8* %305, i64 -8, !dbg !49
  %307 = bitcast i8* %306 to double*, !dbg !49
  %308 = getelementptr double, double* %307, i64 %303, !dbg !49
  %309 = load double, double* %308, align 8, !dbg !49
  %310 = fmul fast double %288, %309, !dbg !49
  %311 = bitcast [22 x i64]* %"c$sd3_384" to i8*, !dbg !49
  %312 = getelementptr i8, i8* %311, i64 56, !dbg !49
  %313 = bitcast i8* %312 to i64*, !dbg !49
  %314 = load i64, i64* %313, align 8, !dbg !49
  %315 = load i32, i32* %j_342, align 4, !dbg !49
  call void @llvm.dbg.value(metadata i32 %315, metadata !40, metadata !DIExpression()), !dbg !10
  %316 = sext i32 %315 to i64, !dbg !49
  %317 = bitcast [22 x i64]* %"c$sd3_384" to i8*, !dbg !49
  %318 = getelementptr i8, i8* %317, i64 160, !dbg !49
  %319 = bitcast i8* %318 to i64*, !dbg !49
  %320 = load i64, i64* %319, align 8, !dbg !49
  %321 = mul nsw i64 %316, %320, !dbg !49
  %322 = load i32, i32* %i_352, align 4, !dbg !49
  call void @llvm.dbg.value(metadata i32 %322, metadata !38, metadata !DIExpression()), !dbg !48
  %323 = sext i32 %322 to i64, !dbg !49
  %324 = add nsw i64 %321, %323, !dbg !49
  %325 = add nsw i64 %314, %324, !dbg !49
  %326 = load double*, double** %.Z0980_347, align 8, !dbg !49
  call void @llvm.dbg.value(metadata double* %326, metadata !19, metadata !DIExpression()), !dbg !10
  %327 = bitcast double* %326 to i8*, !dbg !49
  %328 = getelementptr i8, i8* %327, i64 -8, !dbg !49
  %329 = bitcast i8* %328 to double*, !dbg !49
  %330 = getelementptr double, double* %329, i64 %325, !dbg !49
  store double %310, double* %330, align 8, !dbg !49
  %331 = load i64, i64* %.id0000_356, align 8, !dbg !50
  %332 = add nsw i64 %331, 1, !dbg !50
  store i64 %332, i64* %.id0000_356, align 8, !dbg !50
  %333 = load i64, i64* %.dY0003_398, align 8, !dbg !50
  %334 = sub nsw i64 %333, 1, !dbg !50
  store i64 %334, i64* %.dY0003_398, align 8, !dbg !50
  %335 = load i64, i64* %.dY0003_398, align 8, !dbg !50
  %336 = icmp sgt i64 %335, 0, !dbg !50
  br i1 %336, label %L.LB1_396, label %L.LB1_397, !dbg !50

L.LB1_397:                                        ; preds = %L.LB1_396, %L.LB1_351
  br label %L.LB1_362

L.LB1_362:                                        ; preds = %L.LB1_397
  call void (...) @_mp_bcs_nest(), !dbg !51
  %337 = bitcast i32* @.C366_MAIN_ to i8*, !dbg !51
  %338 = bitcast [49 x i8]* @.C364_MAIN_ to i8*, !dbg !51
  %339 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !51
  call void (i8*, i8*, i64, ...) %339(i8* %337, i8* %338, i64 49), !dbg !51
  %340 = bitcast i32* @.C367_MAIN_ to i8*, !dbg !51
  %341 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !51
  %342 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !51
  %343 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !51
  %344 = call i32 (i8*, i8*, i8*, i8*, ...) %343(i8* %340, i8* null, i8* %341, i8* %342), !dbg !51
  call void @llvm.dbg.declare(metadata i32* %z__io_369, metadata !52, metadata !DIExpression()), !dbg !10
  store i32 %344, i32* %z__io_369, align 4, !dbg !51
  %345 = bitcast [10 x i8]* @.C370_MAIN_ to i8*, !dbg !51
  %346 = bitcast i32 (...)* @f90io_sc_ch_ldw to i32 (i8*, i32, i64, ...)*, !dbg !51
  %347 = call i32 (i8*, i32, i64, ...) %346(i8* %345, i32 14, i64 10), !dbg !51
  store i32 %347, i32* %z__io_369, align 4, !dbg !51
  %348 = bitcast [22 x i64]* %"c$sd3_384" to i8*, !dbg !51
  %349 = getelementptr i8, i8* %348, i64 56, !dbg !51
  %350 = bitcast i8* %349 to i64*, !dbg !51
  %351 = load i64, i64* %350, align 8, !dbg !51
  %352 = bitcast [22 x i64]* %"c$sd3_384" to i8*, !dbg !51
  %353 = getelementptr i8, i8* %352, i64 160, !dbg !51
  %354 = bitcast i8* %353 to i64*, !dbg !51
  %355 = load i64, i64* %354, align 8, !dbg !51
  %356 = mul nsw i64 %355, 50, !dbg !51
  %357 = add nsw i64 %351, %356, !dbg !51
  %358 = load double*, double** %.Z0980_347, align 8, !dbg !51
  call void @llvm.dbg.value(metadata double* %358, metadata !19, metadata !DIExpression()), !dbg !10
  %359 = bitcast double* %358 to i8*, !dbg !51
  %360 = getelementptr i8, i8* %359, i64 392, !dbg !51
  %361 = bitcast i8* %360 to double*, !dbg !51
  %362 = getelementptr double, double* %361, i64 %357, !dbg !51
  %363 = load double, double* %362, align 8, !dbg !51
  %364 = bitcast i32 (...)* @f90io_sc_d_ldw to i32 (double, i32, ...)*, !dbg !51
  %365 = call i32 (double, i32, ...) %364(double %363, i32 28), !dbg !51
  store i32 %365, i32* %z__io_369, align 4, !dbg !51
  %366 = call i32 (...) @f90io_ldw_end(), !dbg !51
  store i32 %366, i32* %z__io_369, align 4, !dbg !51
  call void (...) @_mp_ecs_nest(), !dbg !51
  %367 = load double*, double** %.Z0969_345, align 8, !dbg !53
  call void @llvm.dbg.value(metadata double* %367, metadata !30, metadata !DIExpression()), !dbg !10
  %368 = bitcast double* %367 to i8*, !dbg !53
  %369 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !53
  %370 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !53
  call void (i8*, i8*, i8*, i8*, i64, ...) %370(i8* null, i8* %368, i8* %369, i8* null, i64 0), !dbg !53
  %371 = bitcast double** %.Z0969_345 to i8**, !dbg !53
  store i8* null, i8** %371, align 8, !dbg !53
  %372 = bitcast [22 x i64]* %"a$sd1_379" to i64*, !dbg !53
  store i64 0, i64* %372, align 8, !dbg !53
  %373 = load double*, double** %.Z0970_346, align 8, !dbg !53
  call void @llvm.dbg.value(metadata double* %373, metadata !29, metadata !DIExpression()), !dbg !10
  %374 = bitcast double* %373 to i8*, !dbg !53
  %375 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !53
  %376 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !53
  call void (i8*, i8*, i8*, i8*, i64, ...) %376(i8* null, i8* %374, i8* %375, i8* null, i64 0), !dbg !53
  %377 = bitcast double** %.Z0970_346 to i8**, !dbg !53
  store i8* null, i8** %377, align 8, !dbg !53
  %378 = bitcast [22 x i64]* %"b$sd2_383" to i64*, !dbg !53
  store i64 0, i64* %378, align 8, !dbg !53
  %379 = load double*, double** %.Z0980_347, align 8, !dbg !53
  call void @llvm.dbg.value(metadata double* %379, metadata !19, metadata !DIExpression()), !dbg !10
  %380 = bitcast double* %379 to i8*, !dbg !53
  %381 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !53
  %382 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !53
  call void (i8*, i8*, i8*, i8*, i64, ...) %382(i8* null, i8* %380, i8* %381, i8* null, i64 0), !dbg !53
  %383 = bitcast double** %.Z0980_347 to i8**, !dbg !53
  store i8* null, i8** %383, align 8, !dbg !53
  %384 = bitcast [22 x i64]* %"c$sd3_384" to i64*, !dbg !53
  store i64 0, i64* %384, align 8, !dbg !53
  ret void, !dbg !48
}

declare void @f90_dealloc03a_i8(...) #0

declare void @_mp_ecs_nest(...) #0

declare signext i32 @f90io_ldw_end(...) #0

declare signext i32 @f90io_sc_d_ldw(...) #0

declare signext i32 @f90io_sc_ch_ldw(...) #0

declare signext i32 @f90io_print_init(...) #0

declare void @f90io_src_info03a(...) #0

declare void @_mp_bcs_nest(...) #0

declare void @f90_alloc04_chka_i8(...) #0

declare void @f90_set_intrin_type_i8(...) #0

declare void @f90_template2_i8(...) #0

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare void @fort_init(...) #0

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { "no-frame-pointer-elim-non-leaf" }
attributes #1 = { nounwind readnone speculatable }

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !4, globals: !4, imports: !4)
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB098-simd2-orig-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb098_simd2_orig_no", scope: !2, file: !3, line: 12, type: !6, scopeLine: 12, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!6 = !DISubroutineType(cc: DW_CC_program, types: !7)
!7 = !{null}
!8 = !DILocalVariable(name: "omp_nest_lock_kind", scope: !5, file: !3, type: !9)
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 0, scope: !5)
!11 = !DILocalVariable(name: "omp_sched_static", scope: !5, file: !3, type: !9)
!12 = !DILocalVariable(name: "omp_proc_bind_false", scope: !5, file: !3, type: !9)
!13 = !DILocalVariable(name: "omp_proc_bind_true", scope: !5, file: !3, type: !9)
!14 = !DILocalVariable(name: "omp_lock_hint_none", scope: !5, file: !3, type: !9)
!15 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !5, file: !3, type: !9)
!16 = !DILocalVariable(name: "omp_lock_hint_speculative", scope: !5, file: !3, type: !9)
!17 = !DILocalVariable(name: "dp", scope: !5, file: !3, type: !9)
!18 = !DILocation(line: 12, column: 1, scope: !5)
!19 = !DILocalVariable(name: "c", scope: !5, file: !3, type: !20)
!20 = !DICompositeType(tag: DW_TAG_array_type, baseType: !21, size: 64, align: 64, elements: !22)
!21 = !DIBasicType(name: "double precision", size: 64, align: 64, encoding: DW_ATE_float)
!22 = !{!23, !23}
!23 = !DISubrange(count: 0, lowerBound: 1)
!24 = !DILocalVariable(scope: !5, file: !3, type: !25, flags: DIFlagArtificial)
!25 = !DICompositeType(tag: DW_TAG_array_type, baseType: !26, size: 1408, align: 64, elements: !27)
!26 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!27 = !{!28}
!28 = !DISubrange(count: 22, lowerBound: 1)
!29 = !DILocalVariable(name: "b", scope: !5, file: !3, type: !20)
!30 = !DILocalVariable(name: "a", scope: !5, file: !3, type: !20)
!31 = !DILocalVariable(name: "len", scope: !5, file: !3, type: !9)
!32 = !DILocation(line: 20, column: 1, scope: !5)
!33 = !DILocalVariable(scope: !5, file: !3, type: !26, flags: DIFlagArtificial)
!34 = !DILocation(line: 21, column: 1, scope: !5)
!35 = !DILocation(line: 22, column: 1, scope: !5)
!36 = !DILocation(line: 23, column: 1, scope: !5)
!37 = !DILocation(line: 25, column: 1, scope: !5)
!38 = !DILocalVariable(name: "i", scope: !5, file: !3, type: !9)
!39 = !DILocation(line: 26, column: 1, scope: !5)
!40 = !DILocalVariable(name: "j", scope: !5, file: !3, type: !9)
!41 = !DILocation(line: 27, column: 1, scope: !5)
!42 = !DILocation(line: 28, column: 1, scope: !5)
!43 = !DILocation(line: 29, column: 1, scope: !5)
!44 = !DILocation(line: 30, column: 1, scope: !5)
!45 = !DILocation(line: 31, column: 1, scope: !5)
!46 = !DILocation(line: 34, column: 1, scope: !5)
!47 = !DILocation(line: 35, column: 1, scope: !5)
!48 = !DILocation(line: 44, column: 1, scope: !5)
!49 = !DILocation(line: 36, column: 1, scope: !5)
!50 = !DILocation(line: 38, column: 1, scope: !5)
!51 = !DILocation(line: 41, column: 1, scope: !5)
!52 = !DILocalVariable(scope: !5, file: !3, type: !9, flags: DIFlagArtificial)
!53 = !DILocation(line: 43, column: 1, scope: !5)

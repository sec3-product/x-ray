; ModuleID = '/tmp/DRB112-linear-orig-no-c225df.ll'
source_filename = "/tmp/DRB112-linear-orig-no-c225df.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%astruct.dt80 = type <{ i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8* }>

@.C336_MAIN_ = internal constant double 7.000000e+00
@.C335_MAIN_ = internal constant double 3.000000e+00
@.C293_MAIN_ = internal constant double 2.000000e+00
@.C300_MAIN_ = internal constant i32 8
@.C285_MAIN_ = internal constant i32 1
@.C308_MAIN_ = internal constant i32 28
@.C347_MAIN_ = internal constant i64 8
@.C346_MAIN_ = internal constant i64 28
@.C330_MAIN_ = internal constant i32 100
@.C286_MAIN_ = internal constant i64 1
@.C284_MAIN_ = internal constant i64 0
@.C283_MAIN_ = internal constant i32 0
@.C285___nv_MAIN__F1L32_1 = internal constant i32 1
@.C283___nv_MAIN__F1L32_1 = internal constant i32 0

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__472 = alloca i32, align 4
  %.Z0974_334 = alloca double*, align 8
  %"c$sd3_350" = alloca [16 x i64], align 8
  %.Z0968_333 = alloca double*, align 8
  %"b$sd2_349" = alloca [16 x i64], align 8
  %.Z0967_332 = alloca double*, align 8
  %"a$sd1_345" = alloca [16 x i64], align 8
  %len_331 = alloca i32, align 4
  %i_309 = alloca i32, align 4
  %j_310 = alloca i32, align 4
  %z_b_0_312 = alloca i64, align 8
  %z_b_1_313 = alloca i64, align 8
  %z_e_62_316 = alloca i64, align 8
  %z_b_2_314 = alloca i64, align 8
  %z_b_3_315 = alloca i64, align 8
  %z_b_4_319 = alloca i64, align 8
  %z_b_5_320 = alloca i64, align 8
  %z_e_69_323 = alloca i64, align 8
  %z_b_6_321 = alloca i64, align 8
  %z_b_7_322 = alloca i64, align 8
  %z_b_8_325 = alloca i64, align 8
  %z_b_9_326 = alloca i64, align 8
  %z_e_76_329 = alloca i64, align 8
  %z_b_10_327 = alloca i64, align 8
  %z_b_11_328 = alloca i64, align 8
  %.dY0001_358 = alloca i32, align 4
  %.uplevelArgPack0001_417 = alloca %astruct.dt80, align 16
  call void @llvm.dbg.value(metadata i32 8, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !14, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !15, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 8, metadata !16, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 8, metadata !17, metadata !DIExpression()), !dbg !10
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !18
  store i32 %0, i32* %__gtid_MAIN__472, align 4, !dbg !18
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !19
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !19
  call void (i8*, ...) %2(i8* %1), !dbg !19
  call void @llvm.dbg.declare(metadata double** %.Z0974_334, metadata !20, metadata !DIExpression(DW_OP_deref)), !dbg !10
  %3 = bitcast double** %.Z0974_334 to i8**, !dbg !19
  store i8* null, i8** %3, align 8, !dbg !19
  call void @llvm.dbg.declare(metadata [16 x i64]* %"c$sd3_350", metadata !25, metadata !DIExpression()), !dbg !10
  %4 = bitcast [16 x i64]* %"c$sd3_350" to i64*, !dbg !19
  store i64 0, i64* %4, align 8, !dbg !19
  call void @llvm.dbg.declare(metadata double** %.Z0968_333, metadata !30, metadata !DIExpression(DW_OP_deref)), !dbg !10
  %5 = bitcast double** %.Z0968_333 to i8**, !dbg !19
  store i8* null, i8** %5, align 8, !dbg !19
  call void @llvm.dbg.declare(metadata [16 x i64]* %"b$sd2_349", metadata !25, metadata !DIExpression()), !dbg !10
  %6 = bitcast [16 x i64]* %"b$sd2_349" to i64*, !dbg !19
  store i64 0, i64* %6, align 8, !dbg !19
  call void @llvm.dbg.declare(metadata double** %.Z0967_332, metadata !31, metadata !DIExpression(DW_OP_deref)), !dbg !10
  %7 = bitcast double** %.Z0967_332 to i8**, !dbg !19
  store i8* null, i8** %7, align 8, !dbg !19
  call void @llvm.dbg.declare(metadata [16 x i64]* %"a$sd1_345", metadata !25, metadata !DIExpression()), !dbg !10
  %8 = bitcast [16 x i64]* %"a$sd1_345" to i64*, !dbg !19
  store i64 0, i64* %8, align 8, !dbg !19
  br label %L.LB1_383

L.LB1_383:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %len_331, metadata !32, metadata !DIExpression()), !dbg !10
  store i32 100, i32* %len_331, align 4, !dbg !33
  call void @llvm.dbg.declare(metadata i32* %i_309, metadata !34, metadata !DIExpression()), !dbg !10
  store i32 0, i32* %i_309, align 4, !dbg !35
  call void @llvm.dbg.declare(metadata i32* %j_310, metadata !36, metadata !DIExpression()), !dbg !10
  store i32 0, i32* %j_310, align 4, !dbg !37
  call void @llvm.dbg.declare(metadata i64* %z_b_0_312, metadata !38, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_0_312, align 8, !dbg !39
  %9 = load i32, i32* %len_331, align 4, !dbg !39
  call void @llvm.dbg.value(metadata i32 %9, metadata !32, metadata !DIExpression()), !dbg !10
  %10 = sext i32 %9 to i64, !dbg !39
  call void @llvm.dbg.declare(metadata i64* %z_b_1_313, metadata !38, metadata !DIExpression()), !dbg !10
  store i64 %10, i64* %z_b_1_313, align 8, !dbg !39
  %11 = load i64, i64* %z_b_1_313, align 8, !dbg !39
  call void @llvm.dbg.value(metadata i64 %11, metadata !38, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_62_316, metadata !38, metadata !DIExpression()), !dbg !10
  store i64 %11, i64* %z_e_62_316, align 8, !dbg !39
  %12 = bitcast [16 x i64]* %"a$sd1_345" to i8*, !dbg !39
  %13 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !39
  %14 = bitcast i64* @.C346_MAIN_ to i8*, !dbg !39
  %15 = bitcast i64* @.C347_MAIN_ to i8*, !dbg !39
  %16 = bitcast i64* %z_b_0_312 to i8*, !dbg !39
  %17 = bitcast i64* %z_b_1_313 to i8*, !dbg !39
  %18 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !39
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %18(i8* %12, i8* %13, i8* %14, i8* %15, i8* %16, i8* %17), !dbg !39
  %19 = bitcast [16 x i64]* %"a$sd1_345" to i8*, !dbg !39
  %20 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !39
  call void (i8*, i32, ...) %20(i8* %19, i32 28), !dbg !39
  %21 = load i64, i64* %z_b_1_313, align 8, !dbg !39
  call void @llvm.dbg.value(metadata i64 %21, metadata !38, metadata !DIExpression()), !dbg !10
  %22 = load i64, i64* %z_b_0_312, align 8, !dbg !39
  call void @llvm.dbg.value(metadata i64 %22, metadata !38, metadata !DIExpression()), !dbg !10
  %23 = sub nsw i64 %22, 1, !dbg !39
  %24 = sub nsw i64 %21, %23, !dbg !39
  call void @llvm.dbg.declare(metadata i64* %z_b_2_314, metadata !38, metadata !DIExpression()), !dbg !10
  store i64 %24, i64* %z_b_2_314, align 8, !dbg !39
  %25 = load i64, i64* %z_b_0_312, align 8, !dbg !39
  call void @llvm.dbg.value(metadata i64 %25, metadata !38, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_b_3_315, metadata !38, metadata !DIExpression()), !dbg !10
  store i64 %25, i64* %z_b_3_315, align 8, !dbg !39
  %26 = bitcast i64* %z_b_2_314 to i8*, !dbg !39
  %27 = bitcast i64* @.C346_MAIN_ to i8*, !dbg !39
  %28 = bitcast i64* @.C347_MAIN_ to i8*, !dbg !39
  %29 = bitcast double** %.Z0967_332 to i8*, !dbg !39
  %30 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !39
  %31 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !39
  %32 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !39
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %32(i8* %26, i8* %27, i8* %28, i8* null, i8* %29, i8* null, i8* %30, i8* %31, i8* null, i64 0), !dbg !39
  call void @llvm.dbg.declare(metadata i64* %z_b_4_319, metadata !38, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_4_319, align 8, !dbg !40
  %33 = load i32, i32* %len_331, align 4, !dbg !40
  call void @llvm.dbg.value(metadata i32 %33, metadata !32, metadata !DIExpression()), !dbg !10
  %34 = sext i32 %33 to i64, !dbg !40
  call void @llvm.dbg.declare(metadata i64* %z_b_5_320, metadata !38, metadata !DIExpression()), !dbg !10
  store i64 %34, i64* %z_b_5_320, align 8, !dbg !40
  %35 = load i64, i64* %z_b_5_320, align 8, !dbg !40
  call void @llvm.dbg.value(metadata i64 %35, metadata !38, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_69_323, metadata !38, metadata !DIExpression()), !dbg !10
  store i64 %35, i64* %z_e_69_323, align 8, !dbg !40
  %36 = bitcast [16 x i64]* %"b$sd2_349" to i8*, !dbg !40
  %37 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !40
  %38 = bitcast i64* @.C346_MAIN_ to i8*, !dbg !40
  %39 = bitcast i64* @.C347_MAIN_ to i8*, !dbg !40
  %40 = bitcast i64* %z_b_4_319 to i8*, !dbg !40
  %41 = bitcast i64* %z_b_5_320 to i8*, !dbg !40
  %42 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !40
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %42(i8* %36, i8* %37, i8* %38, i8* %39, i8* %40, i8* %41), !dbg !40
  %43 = bitcast [16 x i64]* %"b$sd2_349" to i8*, !dbg !40
  %44 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !40
  call void (i8*, i32, ...) %44(i8* %43, i32 28), !dbg !40
  %45 = load i64, i64* %z_b_5_320, align 8, !dbg !40
  call void @llvm.dbg.value(metadata i64 %45, metadata !38, metadata !DIExpression()), !dbg !10
  %46 = load i64, i64* %z_b_4_319, align 8, !dbg !40
  call void @llvm.dbg.value(metadata i64 %46, metadata !38, metadata !DIExpression()), !dbg !10
  %47 = sub nsw i64 %46, 1, !dbg !40
  %48 = sub nsw i64 %45, %47, !dbg !40
  call void @llvm.dbg.declare(metadata i64* %z_b_6_321, metadata !38, metadata !DIExpression()), !dbg !10
  store i64 %48, i64* %z_b_6_321, align 8, !dbg !40
  %49 = load i64, i64* %z_b_4_319, align 8, !dbg !40
  call void @llvm.dbg.value(metadata i64 %49, metadata !38, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_b_7_322, metadata !38, metadata !DIExpression()), !dbg !10
  store i64 %49, i64* %z_b_7_322, align 8, !dbg !40
  %50 = bitcast i64* %z_b_6_321 to i8*, !dbg !40
  %51 = bitcast i64* @.C346_MAIN_ to i8*, !dbg !40
  %52 = bitcast i64* @.C347_MAIN_ to i8*, !dbg !40
  %53 = bitcast double** %.Z0968_333 to i8*, !dbg !40
  %54 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !40
  %55 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !40
  %56 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !40
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %56(i8* %50, i8* %51, i8* %52, i8* null, i8* %53, i8* null, i8* %54, i8* %55, i8* null, i64 0), !dbg !40
  call void @llvm.dbg.declare(metadata i64* %z_b_8_325, metadata !38, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_8_325, align 8, !dbg !41
  %57 = load i32, i32* %len_331, align 4, !dbg !41
  call void @llvm.dbg.value(metadata i32 %57, metadata !32, metadata !DIExpression()), !dbg !10
  %58 = sext i32 %57 to i64, !dbg !41
  call void @llvm.dbg.declare(metadata i64* %z_b_9_326, metadata !38, metadata !DIExpression()), !dbg !10
  store i64 %58, i64* %z_b_9_326, align 8, !dbg !41
  %59 = load i64, i64* %z_b_9_326, align 8, !dbg !41
  call void @llvm.dbg.value(metadata i64 %59, metadata !38, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_76_329, metadata !38, metadata !DIExpression()), !dbg !10
  store i64 %59, i64* %z_e_76_329, align 8, !dbg !41
  %60 = bitcast [16 x i64]* %"c$sd3_350" to i8*, !dbg !41
  %61 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !41
  %62 = bitcast i64* @.C346_MAIN_ to i8*, !dbg !41
  %63 = bitcast i64* @.C347_MAIN_ to i8*, !dbg !41
  %64 = bitcast i64* %z_b_8_325 to i8*, !dbg !41
  %65 = bitcast i64* %z_b_9_326 to i8*, !dbg !41
  %66 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !41
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %66(i8* %60, i8* %61, i8* %62, i8* %63, i8* %64, i8* %65), !dbg !41
  %67 = bitcast [16 x i64]* %"c$sd3_350" to i8*, !dbg !41
  %68 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !41
  call void (i8*, i32, ...) %68(i8* %67, i32 28), !dbg !41
  %69 = load i64, i64* %z_b_9_326, align 8, !dbg !41
  call void @llvm.dbg.value(metadata i64 %69, metadata !38, metadata !DIExpression()), !dbg !10
  %70 = load i64, i64* %z_b_8_325, align 8, !dbg !41
  call void @llvm.dbg.value(metadata i64 %70, metadata !38, metadata !DIExpression()), !dbg !10
  %71 = sub nsw i64 %70, 1, !dbg !41
  %72 = sub nsw i64 %69, %71, !dbg !41
  call void @llvm.dbg.declare(metadata i64* %z_b_10_327, metadata !38, metadata !DIExpression()), !dbg !10
  store i64 %72, i64* %z_b_10_327, align 8, !dbg !41
  %73 = load i64, i64* %z_b_8_325, align 8, !dbg !41
  call void @llvm.dbg.value(metadata i64 %73, metadata !38, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_b_11_328, metadata !38, metadata !DIExpression()), !dbg !10
  store i64 %73, i64* %z_b_11_328, align 8, !dbg !41
  %74 = bitcast i64* %z_b_10_327 to i8*, !dbg !41
  %75 = bitcast i64* @.C346_MAIN_ to i8*, !dbg !41
  %76 = bitcast i64* @.C347_MAIN_ to i8*, !dbg !41
  %77 = bitcast double** %.Z0974_334 to i8*, !dbg !41
  %78 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !41
  %79 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !41
  %80 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !41
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %80(i8* %74, i8* %75, i8* %76, i8* null, i8* %77, i8* null, i8* %78, i8* %79, i8* null, i64 0), !dbg !41
  %81 = load i32, i32* %len_331, align 4, !dbg !42
  call void @llvm.dbg.value(metadata i32 %81, metadata !32, metadata !DIExpression()), !dbg !10
  store i32 %81, i32* %.dY0001_358, align 4, !dbg !42
  store i32 1, i32* %i_309, align 4, !dbg !42
  %82 = load i32, i32* %.dY0001_358, align 4, !dbg !42
  %83 = icmp sle i32 %82, 0, !dbg !42
  br i1 %83, label %L.LB1_357, label %L.LB1_356, !dbg !42

L.LB1_356:                                        ; preds = %L.LB1_356, %L.LB1_383
  %84 = load i32, i32* %i_309, align 4, !dbg !43
  call void @llvm.dbg.value(metadata i32 %84, metadata !34, metadata !DIExpression()), !dbg !10
  %85 = sitofp i32 %84 to double, !dbg !43
  %86 = fdiv fast double %85, 2.000000e+00, !dbg !43
  %87 = load i32, i32* %i_309, align 4, !dbg !43
  call void @llvm.dbg.value(metadata i32 %87, metadata !34, metadata !DIExpression()), !dbg !10
  %88 = sext i32 %87 to i64, !dbg !43
  %89 = bitcast [16 x i64]* %"a$sd1_345" to i8*, !dbg !43
  %90 = getelementptr i8, i8* %89, i64 56, !dbg !43
  %91 = bitcast i8* %90 to i64*, !dbg !43
  %92 = load i64, i64* %91, align 8, !dbg !43
  %93 = add nsw i64 %88, %92, !dbg !43
  %94 = load double*, double** %.Z0967_332, align 8, !dbg !43
  call void @llvm.dbg.value(metadata double* %94, metadata !31, metadata !DIExpression()), !dbg !10
  %95 = bitcast double* %94 to i8*, !dbg !43
  %96 = getelementptr i8, i8* %95, i64 -8, !dbg !43
  %97 = bitcast i8* %96 to double*, !dbg !43
  %98 = getelementptr double, double* %97, i64 %93, !dbg !43
  store double %86, double* %98, align 8, !dbg !43
  %99 = load i32, i32* %i_309, align 4, !dbg !44
  call void @llvm.dbg.value(metadata i32 %99, metadata !34, metadata !DIExpression()), !dbg !10
  %100 = sitofp i32 %99 to double, !dbg !44
  %101 = fdiv fast double %100, 3.000000e+00, !dbg !44
  %102 = load i32, i32* %i_309, align 4, !dbg !44
  call void @llvm.dbg.value(metadata i32 %102, metadata !34, metadata !DIExpression()), !dbg !10
  %103 = sext i32 %102 to i64, !dbg !44
  %104 = bitcast [16 x i64]* %"b$sd2_349" to i8*, !dbg !44
  %105 = getelementptr i8, i8* %104, i64 56, !dbg !44
  %106 = bitcast i8* %105 to i64*, !dbg !44
  %107 = load i64, i64* %106, align 8, !dbg !44
  %108 = add nsw i64 %103, %107, !dbg !44
  %109 = load double*, double** %.Z0968_333, align 8, !dbg !44
  call void @llvm.dbg.value(metadata double* %109, metadata !30, metadata !DIExpression()), !dbg !10
  %110 = bitcast double* %109 to i8*, !dbg !44
  %111 = getelementptr i8, i8* %110, i64 -8, !dbg !44
  %112 = bitcast i8* %111 to double*, !dbg !44
  %113 = getelementptr double, double* %112, i64 %108, !dbg !44
  store double %101, double* %113, align 8, !dbg !44
  %114 = load i32, i32* %i_309, align 4, !dbg !45
  call void @llvm.dbg.value(metadata i32 %114, metadata !34, metadata !DIExpression()), !dbg !10
  %115 = sitofp i32 %114 to double, !dbg !45
  %116 = fdiv fast double %115, 7.000000e+00, !dbg !45
  %117 = load i32, i32* %i_309, align 4, !dbg !45
  call void @llvm.dbg.value(metadata i32 %117, metadata !34, metadata !DIExpression()), !dbg !10
  %118 = sext i32 %117 to i64, !dbg !45
  %119 = bitcast [16 x i64]* %"c$sd3_350" to i8*, !dbg !45
  %120 = getelementptr i8, i8* %119, i64 56, !dbg !45
  %121 = bitcast i8* %120 to i64*, !dbg !45
  %122 = load i64, i64* %121, align 8, !dbg !45
  %123 = add nsw i64 %118, %122, !dbg !45
  %124 = load double*, double** %.Z0974_334, align 8, !dbg !45
  call void @llvm.dbg.value(metadata double* %124, metadata !20, metadata !DIExpression()), !dbg !10
  %125 = bitcast double* %124 to i8*, !dbg !45
  %126 = getelementptr i8, i8* %125, i64 -8, !dbg !45
  %127 = bitcast i8* %126 to double*, !dbg !45
  %128 = getelementptr double, double* %127, i64 %123, !dbg !45
  store double %116, double* %128, align 8, !dbg !45
  %129 = load i32, i32* %i_309, align 4, !dbg !46
  call void @llvm.dbg.value(metadata i32 %129, metadata !34, metadata !DIExpression()), !dbg !10
  %130 = add nsw i32 %129, 1, !dbg !46
  store i32 %130, i32* %i_309, align 4, !dbg !46
  %131 = load i32, i32* %.dY0001_358, align 4, !dbg !46
  %132 = sub nsw i32 %131, 1, !dbg !46
  store i32 %132, i32* %.dY0001_358, align 4, !dbg !46
  %133 = load i32, i32* %.dY0001_358, align 4, !dbg !46
  %134 = icmp sgt i32 %133, 0, !dbg !46
  br i1 %134, label %L.LB1_356, label %L.LB1_357, !dbg !46

L.LB1_357:                                        ; preds = %L.LB1_356, %L.LB1_383
  %135 = bitcast i32* %len_331 to i8*, !dbg !47
  %136 = bitcast %astruct.dt80* %.uplevelArgPack0001_417 to i8**, !dbg !47
  store i8* %135, i8** %136, align 8, !dbg !47
  %137 = bitcast double** %.Z0974_334 to i8*, !dbg !47
  %138 = bitcast %astruct.dt80* %.uplevelArgPack0001_417 to i8*, !dbg !47
  %139 = getelementptr i8, i8* %138, i64 8, !dbg !47
  %140 = bitcast i8* %139 to i8**, !dbg !47
  store i8* %137, i8** %140, align 8, !dbg !47
  %141 = bitcast double** %.Z0974_334 to i8*, !dbg !47
  %142 = bitcast %astruct.dt80* %.uplevelArgPack0001_417 to i8*, !dbg !47
  %143 = getelementptr i8, i8* %142, i64 16, !dbg !47
  %144 = bitcast i8* %143 to i8**, !dbg !47
  store i8* %141, i8** %144, align 8, !dbg !47
  %145 = bitcast i64* %z_b_8_325 to i8*, !dbg !47
  %146 = bitcast %astruct.dt80* %.uplevelArgPack0001_417 to i8*, !dbg !47
  %147 = getelementptr i8, i8* %146, i64 24, !dbg !47
  %148 = bitcast i8* %147 to i8**, !dbg !47
  store i8* %145, i8** %148, align 8, !dbg !47
  %149 = bitcast i64* %z_b_9_326 to i8*, !dbg !47
  %150 = bitcast %astruct.dt80* %.uplevelArgPack0001_417 to i8*, !dbg !47
  %151 = getelementptr i8, i8* %150, i64 32, !dbg !47
  %152 = bitcast i8* %151 to i8**, !dbg !47
  store i8* %149, i8** %152, align 8, !dbg !47
  %153 = bitcast i64* %z_e_76_329 to i8*, !dbg !47
  %154 = bitcast %astruct.dt80* %.uplevelArgPack0001_417 to i8*, !dbg !47
  %155 = getelementptr i8, i8* %154, i64 40, !dbg !47
  %156 = bitcast i8* %155 to i8**, !dbg !47
  store i8* %153, i8** %156, align 8, !dbg !47
  %157 = bitcast i64* %z_b_10_327 to i8*, !dbg !47
  %158 = bitcast %astruct.dt80* %.uplevelArgPack0001_417 to i8*, !dbg !47
  %159 = getelementptr i8, i8* %158, i64 48, !dbg !47
  %160 = bitcast i8* %159 to i8**, !dbg !47
  store i8* %157, i8** %160, align 8, !dbg !47
  %161 = bitcast i64* %z_b_11_328 to i8*, !dbg !47
  %162 = bitcast %astruct.dt80* %.uplevelArgPack0001_417 to i8*, !dbg !47
  %163 = getelementptr i8, i8* %162, i64 56, !dbg !47
  %164 = bitcast i8* %163 to i8**, !dbg !47
  store i8* %161, i8** %164, align 8, !dbg !47
  %165 = bitcast i32* %j_310 to i8*, !dbg !47
  %166 = bitcast %astruct.dt80* %.uplevelArgPack0001_417 to i8*, !dbg !47
  %167 = getelementptr i8, i8* %166, i64 64, !dbg !47
  %168 = bitcast i8* %167 to i8**, !dbg !47
  store i8* %165, i8** %168, align 8, !dbg !47
  %169 = bitcast double** %.Z0967_332 to i8*, !dbg !47
  %170 = bitcast %astruct.dt80* %.uplevelArgPack0001_417 to i8*, !dbg !47
  %171 = getelementptr i8, i8* %170, i64 72, !dbg !47
  %172 = bitcast i8* %171 to i8**, !dbg !47
  store i8* %169, i8** %172, align 8, !dbg !47
  %173 = bitcast double** %.Z0967_332 to i8*, !dbg !47
  %174 = bitcast %astruct.dt80* %.uplevelArgPack0001_417 to i8*, !dbg !47
  %175 = getelementptr i8, i8* %174, i64 80, !dbg !47
  %176 = bitcast i8* %175 to i8**, !dbg !47
  store i8* %173, i8** %176, align 8, !dbg !47
  %177 = bitcast i64* %z_b_0_312 to i8*, !dbg !47
  %178 = bitcast %astruct.dt80* %.uplevelArgPack0001_417 to i8*, !dbg !47
  %179 = getelementptr i8, i8* %178, i64 88, !dbg !47
  %180 = bitcast i8* %179 to i8**, !dbg !47
  store i8* %177, i8** %180, align 8, !dbg !47
  %181 = bitcast i64* %z_b_1_313 to i8*, !dbg !47
  %182 = bitcast %astruct.dt80* %.uplevelArgPack0001_417 to i8*, !dbg !47
  %183 = getelementptr i8, i8* %182, i64 96, !dbg !47
  %184 = bitcast i8* %183 to i8**, !dbg !47
  store i8* %181, i8** %184, align 8, !dbg !47
  %185 = bitcast i64* %z_e_62_316 to i8*, !dbg !47
  %186 = bitcast %astruct.dt80* %.uplevelArgPack0001_417 to i8*, !dbg !47
  %187 = getelementptr i8, i8* %186, i64 104, !dbg !47
  %188 = bitcast i8* %187 to i8**, !dbg !47
  store i8* %185, i8** %188, align 8, !dbg !47
  %189 = bitcast i64* %z_b_2_314 to i8*, !dbg !47
  %190 = bitcast %astruct.dt80* %.uplevelArgPack0001_417 to i8*, !dbg !47
  %191 = getelementptr i8, i8* %190, i64 112, !dbg !47
  %192 = bitcast i8* %191 to i8**, !dbg !47
  store i8* %189, i8** %192, align 8, !dbg !47
  %193 = bitcast i64* %z_b_3_315 to i8*, !dbg !47
  %194 = bitcast %astruct.dt80* %.uplevelArgPack0001_417 to i8*, !dbg !47
  %195 = getelementptr i8, i8* %194, i64 120, !dbg !47
  %196 = bitcast i8* %195 to i8**, !dbg !47
  store i8* %193, i8** %196, align 8, !dbg !47
  %197 = bitcast double** %.Z0968_333 to i8*, !dbg !47
  %198 = bitcast %astruct.dt80* %.uplevelArgPack0001_417 to i8*, !dbg !47
  %199 = getelementptr i8, i8* %198, i64 128, !dbg !47
  %200 = bitcast i8* %199 to i8**, !dbg !47
  store i8* %197, i8** %200, align 8, !dbg !47
  %201 = bitcast double** %.Z0968_333 to i8*, !dbg !47
  %202 = bitcast %astruct.dt80* %.uplevelArgPack0001_417 to i8*, !dbg !47
  %203 = getelementptr i8, i8* %202, i64 136, !dbg !47
  %204 = bitcast i8* %203 to i8**, !dbg !47
  store i8* %201, i8** %204, align 8, !dbg !47
  %205 = bitcast i64* %z_b_4_319 to i8*, !dbg !47
  %206 = bitcast %astruct.dt80* %.uplevelArgPack0001_417 to i8*, !dbg !47
  %207 = getelementptr i8, i8* %206, i64 144, !dbg !47
  %208 = bitcast i8* %207 to i8**, !dbg !47
  store i8* %205, i8** %208, align 8, !dbg !47
  %209 = bitcast i64* %z_b_5_320 to i8*, !dbg !47
  %210 = bitcast %astruct.dt80* %.uplevelArgPack0001_417 to i8*, !dbg !47
  %211 = getelementptr i8, i8* %210, i64 152, !dbg !47
  %212 = bitcast i8* %211 to i8**, !dbg !47
  store i8* %209, i8** %212, align 8, !dbg !47
  %213 = bitcast i64* %z_e_69_323 to i8*, !dbg !47
  %214 = bitcast %astruct.dt80* %.uplevelArgPack0001_417 to i8*, !dbg !47
  %215 = getelementptr i8, i8* %214, i64 160, !dbg !47
  %216 = bitcast i8* %215 to i8**, !dbg !47
  store i8* %213, i8** %216, align 8, !dbg !47
  %217 = bitcast i64* %z_b_6_321 to i8*, !dbg !47
  %218 = bitcast %astruct.dt80* %.uplevelArgPack0001_417 to i8*, !dbg !47
  %219 = getelementptr i8, i8* %218, i64 168, !dbg !47
  %220 = bitcast i8* %219 to i8**, !dbg !47
  store i8* %217, i8** %220, align 8, !dbg !47
  %221 = bitcast i64* %z_b_7_322 to i8*, !dbg !47
  %222 = bitcast %astruct.dt80* %.uplevelArgPack0001_417 to i8*, !dbg !47
  %223 = getelementptr i8, i8* %222, i64 176, !dbg !47
  %224 = bitcast i8* %223 to i8**, !dbg !47
  store i8* %221, i8** %224, align 8, !dbg !47
  %225 = bitcast [16 x i64]* %"a$sd1_345" to i8*, !dbg !47
  %226 = bitcast %astruct.dt80* %.uplevelArgPack0001_417 to i8*, !dbg !47
  %227 = getelementptr i8, i8* %226, i64 184, !dbg !47
  %228 = bitcast i8* %227 to i8**, !dbg !47
  store i8* %225, i8** %228, align 8, !dbg !47
  %229 = bitcast [16 x i64]* %"b$sd2_349" to i8*, !dbg !47
  %230 = bitcast %astruct.dt80* %.uplevelArgPack0001_417 to i8*, !dbg !47
  %231 = getelementptr i8, i8* %230, i64 192, !dbg !47
  %232 = bitcast i8* %231 to i8**, !dbg !47
  store i8* %229, i8** %232, align 8, !dbg !47
  %233 = bitcast [16 x i64]* %"c$sd3_350" to i8*, !dbg !47
  %234 = bitcast %astruct.dt80* %.uplevelArgPack0001_417 to i8*, !dbg !47
  %235 = getelementptr i8, i8* %234, i64 200, !dbg !47
  %236 = bitcast i8* %235 to i8**, !dbg !47
  store i8* %233, i8** %236, align 8, !dbg !47
  br label %L.LB1_470, !dbg !47

L.LB1_470:                                        ; preds = %L.LB1_357
  %237 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L32_1_ to i64*, !dbg !47
  %238 = bitcast %astruct.dt80* %.uplevelArgPack0001_417 to i64*, !dbg !47
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %237, i64* %238), !dbg !47
  %239 = load double*, double** %.Z0967_332, align 8, !dbg !48
  call void @llvm.dbg.value(metadata double* %239, metadata !31, metadata !DIExpression()), !dbg !10
  %240 = bitcast double* %239 to i8*, !dbg !48
  %241 = bitcast i32 (...)* @f90_allocated_i8 to i32 (i8*, ...)*, !dbg !48
  %242 = call i32 (i8*, ...) %241(i8* %240), !dbg !48
  %243 = and i32 %242, 1, !dbg !48
  %244 = icmp eq i32 %243, 0, !dbg !48
  br i1 %244, label %L.LB1_371, label %L.LB1_490, !dbg !48

L.LB1_490:                                        ; preds = %L.LB1_470
  %245 = load double*, double** %.Z0967_332, align 8, !dbg !48
  call void @llvm.dbg.value(metadata double* %245, metadata !31, metadata !DIExpression()), !dbg !10
  %246 = bitcast double* %245 to i8*, !dbg !48
  %247 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !48
  %248 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !48
  call void (i8*, i8*, i8*, i8*, i64, ...) %248(i8* null, i8* %246, i8* %247, i8* null, i64 0), !dbg !48
  %249 = bitcast double** %.Z0967_332 to i8**, !dbg !48
  store i8* null, i8** %249, align 8, !dbg !48
  %250 = bitcast [16 x i64]* %"a$sd1_345" to i64*, !dbg !48
  store i64 0, i64* %250, align 8, !dbg !48
  br label %L.LB1_371

L.LB1_371:                                        ; preds = %L.LB1_490, %L.LB1_470
  %251 = load double*, double** %.Z0968_333, align 8, !dbg !49
  call void @llvm.dbg.value(metadata double* %251, metadata !30, metadata !DIExpression()), !dbg !10
  %252 = bitcast double* %251 to i8*, !dbg !49
  %253 = bitcast i32 (...)* @f90_allocated_i8 to i32 (i8*, ...)*, !dbg !49
  %254 = call i32 (i8*, ...) %253(i8* %252), !dbg !49
  %255 = and i32 %254, 1, !dbg !49
  %256 = icmp eq i32 %255, 0, !dbg !49
  br i1 %256, label %L.LB1_374, label %L.LB1_491, !dbg !49

L.LB1_491:                                        ; preds = %L.LB1_371
  %257 = load double*, double** %.Z0968_333, align 8, !dbg !49
  call void @llvm.dbg.value(metadata double* %257, metadata !30, metadata !DIExpression()), !dbg !10
  %258 = bitcast double* %257 to i8*, !dbg !49
  %259 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !49
  %260 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !49
  call void (i8*, i8*, i8*, i8*, i64, ...) %260(i8* null, i8* %258, i8* %259, i8* null, i64 0), !dbg !49
  %261 = bitcast double** %.Z0968_333 to i8**, !dbg !49
  store i8* null, i8** %261, align 8, !dbg !49
  %262 = bitcast [16 x i64]* %"b$sd2_349" to i64*, !dbg !49
  store i64 0, i64* %262, align 8, !dbg !49
  br label %L.LB1_374

L.LB1_374:                                        ; preds = %L.LB1_491, %L.LB1_371
  ret void, !dbg !18
}

define internal void @__nv_MAIN__F1L32_1_(i32* %__nv_MAIN__F1L32_1Arg0, i64* %__nv_MAIN__F1L32_1Arg1, i64* %__nv_MAIN__F1L32_1Arg2) #0 !dbg !50 {
L.entry:
  %__gtid___nv_MAIN__F1L32_1__510 = alloca i32, align 4
  %.i0000p_341 = alloca i32, align 4
  %i_340 = alloca i32, align 4
  %.du0002p_362 = alloca i32, align 4
  %.de0002p_363 = alloca i32, align 4
  %.di0002p_364 = alloca i32, align 4
  %.ds0002p_365 = alloca i32, align 4
  %.dl0002p_367 = alloca i32, align 4
  %.dl0002p.copy_504 = alloca i32, align 4
  %.de0002p.copy_505 = alloca i32, align 4
  %.ds0002p.copy_506 = alloca i32, align 4
  %.dX0002p_366 = alloca i32, align 4
  %.dY0002p_361 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L32_1Arg0, metadata !53, metadata !DIExpression()), !dbg !54
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L32_1Arg1, metadata !55, metadata !DIExpression()), !dbg !54
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L32_1Arg2, metadata !56, metadata !DIExpression()), !dbg !54
  call void @llvm.dbg.value(metadata i32 8, metadata !57, metadata !DIExpression()), !dbg !54
  call void @llvm.dbg.value(metadata i32 1, metadata !58, metadata !DIExpression()), !dbg !54
  call void @llvm.dbg.value(metadata i32 0, metadata !59, metadata !DIExpression()), !dbg !54
  call void @llvm.dbg.value(metadata i32 1, metadata !60, metadata !DIExpression()), !dbg !54
  call void @llvm.dbg.value(metadata i32 0, metadata !61, metadata !DIExpression()), !dbg !54
  call void @llvm.dbg.value(metadata i32 1, metadata !62, metadata !DIExpression()), !dbg !54
  call void @llvm.dbg.value(metadata i32 8, metadata !63, metadata !DIExpression()), !dbg !54
  call void @llvm.dbg.value(metadata i32 8, metadata !64, metadata !DIExpression()), !dbg !54
  %0 = load i32, i32* %__nv_MAIN__F1L32_1Arg0, align 4, !dbg !65
  store i32 %0, i32* %__gtid___nv_MAIN__F1L32_1__510, align 4, !dbg !65
  br label %L.LB2_495

L.LB2_495:                                        ; preds = %L.entry
  br label %L.LB2_339

L.LB2_339:                                        ; preds = %L.LB2_495
  store i32 0, i32* %.i0000p_341, align 4, !dbg !66
  call void @llvm.dbg.declare(metadata i32* %i_340, metadata !67, metadata !DIExpression()), !dbg !65
  store i32 1, i32* %i_340, align 4, !dbg !66
  %1 = bitcast i64* %__nv_MAIN__F1L32_1Arg2 to i32**, !dbg !66
  %2 = load i32*, i32** %1, align 8, !dbg !66
  %3 = load i32, i32* %2, align 4, !dbg !66
  store i32 %3, i32* %.du0002p_362, align 4, !dbg !66
  %4 = bitcast i64* %__nv_MAIN__F1L32_1Arg2 to i32**, !dbg !66
  %5 = load i32*, i32** %4, align 8, !dbg !66
  %6 = load i32, i32* %5, align 4, !dbg !66
  store i32 %6, i32* %.de0002p_363, align 4, !dbg !66
  store i32 1, i32* %.di0002p_364, align 4, !dbg !66
  %7 = load i32, i32* %.di0002p_364, align 4, !dbg !66
  store i32 %7, i32* %.ds0002p_365, align 4, !dbg !66
  store i32 1, i32* %.dl0002p_367, align 4, !dbg !66
  %8 = load i32, i32* %.dl0002p_367, align 4, !dbg !66
  store i32 %8, i32* %.dl0002p.copy_504, align 4, !dbg !66
  %9 = load i32, i32* %.de0002p_363, align 4, !dbg !66
  store i32 %9, i32* %.de0002p.copy_505, align 4, !dbg !66
  %10 = load i32, i32* %.ds0002p_365, align 4, !dbg !66
  store i32 %10, i32* %.ds0002p.copy_506, align 4, !dbg !66
  %11 = load i32, i32* %__gtid___nv_MAIN__F1L32_1__510, align 4, !dbg !66
  %12 = bitcast i32* %.i0000p_341 to i64*, !dbg !66
  %13 = bitcast i32* %.dl0002p.copy_504 to i64*, !dbg !66
  %14 = bitcast i32* %.de0002p.copy_505 to i64*, !dbg !66
  %15 = bitcast i32* %.ds0002p.copy_506 to i64*, !dbg !66
  %16 = load i32, i32* %.ds0002p.copy_506, align 4, !dbg !66
  call void @__kmpc_for_static_init_4(i64* null, i32 %11, i32 34, i64* %12, i64* %13, i64* %14, i64* %15, i32 %16, i32 1), !dbg !66
  %17 = load i32, i32* %.dl0002p.copy_504, align 4, !dbg !66
  store i32 %17, i32* %.dl0002p_367, align 4, !dbg !66
  %18 = load i32, i32* %.de0002p.copy_505, align 4, !dbg !66
  store i32 %18, i32* %.de0002p_363, align 4, !dbg !66
  %19 = load i32, i32* %.ds0002p.copy_506, align 4, !dbg !66
  store i32 %19, i32* %.ds0002p_365, align 4, !dbg !66
  %20 = load i32, i32* %.dl0002p_367, align 4, !dbg !66
  store i32 %20, i32* %i_340, align 4, !dbg !66
  %21 = load i32, i32* %i_340, align 4, !dbg !66
  call void @llvm.dbg.value(metadata i32 %21, metadata !67, metadata !DIExpression()), !dbg !65
  store i32 %21, i32* %.dX0002p_366, align 4, !dbg !66
  %22 = load i32, i32* %.dX0002p_366, align 4, !dbg !66
  %23 = load i32, i32* %.du0002p_362, align 4, !dbg !66
  %24 = icmp sgt i32 %22, %23, !dbg !66
  br i1 %24, label %L.LB2_360, label %L.LB2_539, !dbg !66

L.LB2_539:                                        ; preds = %L.LB2_339
  %25 = load i32, i32* %.dX0002p_366, align 4, !dbg !66
  store i32 %25, i32* %i_340, align 4, !dbg !66
  %26 = load i32, i32* %.di0002p_364, align 4, !dbg !66
  %27 = load i32, i32* %.de0002p_363, align 4, !dbg !66
  %28 = load i32, i32* %.dX0002p_366, align 4, !dbg !66
  %29 = sub nsw i32 %27, %28, !dbg !66
  %30 = add nsw i32 %26, %29, !dbg !66
  %31 = load i32, i32* %.di0002p_364, align 4, !dbg !66
  %32 = sdiv i32 %30, %31, !dbg !66
  store i32 %32, i32* %.dY0002p_361, align 4, !dbg !66
  %33 = load i32, i32* %.dY0002p_361, align 4, !dbg !66
  %34 = icmp sle i32 %33, 0, !dbg !66
  br i1 %34, label %L.LB2_370, label %L.LB2_369, !dbg !66

L.LB2_369:                                        ; preds = %L.LB2_369, %L.LB2_539
  %35 = load i32, i32* %i_340, align 4, !dbg !68
  call void @llvm.dbg.value(metadata i32 %35, metadata !67, metadata !DIExpression()), !dbg !65
  %36 = sext i32 %35 to i64, !dbg !68
  %37 = bitcast i64* %__nv_MAIN__F1L32_1Arg2 to i8*, !dbg !68
  %38 = getelementptr i8, i8* %37, i64 192, !dbg !68
  %39 = bitcast i8* %38 to i8**, !dbg !68
  %40 = load i8*, i8** %39, align 8, !dbg !68
  %41 = getelementptr i8, i8* %40, i64 56, !dbg !68
  %42 = bitcast i8* %41 to i64*, !dbg !68
  %43 = load i64, i64* %42, align 8, !dbg !68
  %44 = add nsw i64 %36, %43, !dbg !68
  %45 = bitcast i64* %__nv_MAIN__F1L32_1Arg2 to i8*, !dbg !68
  %46 = getelementptr i8, i8* %45, i64 136, !dbg !68
  %47 = bitcast i8* %46 to i8***, !dbg !68
  %48 = load i8**, i8*** %47, align 8, !dbg !68
  %49 = load i8*, i8** %48, align 8, !dbg !68
  %50 = getelementptr i8, i8* %49, i64 -8, !dbg !68
  %51 = bitcast i8* %50 to double*, !dbg !68
  %52 = getelementptr double, double* %51, i64 %44, !dbg !68
  %53 = load double, double* %52, align 8, !dbg !68
  %54 = load i32, i32* %i_340, align 4, !dbg !68
  call void @llvm.dbg.value(metadata i32 %54, metadata !67, metadata !DIExpression()), !dbg !65
  %55 = sext i32 %54 to i64, !dbg !68
  %56 = bitcast i64* %__nv_MAIN__F1L32_1Arg2 to i8*, !dbg !68
  %57 = getelementptr i8, i8* %56, i64 184, !dbg !68
  %58 = bitcast i8* %57 to i8**, !dbg !68
  %59 = load i8*, i8** %58, align 8, !dbg !68
  %60 = getelementptr i8, i8* %59, i64 56, !dbg !68
  %61 = bitcast i8* %60 to i64*, !dbg !68
  %62 = load i64, i64* %61, align 8, !dbg !68
  %63 = add nsw i64 %55, %62, !dbg !68
  %64 = bitcast i64* %__nv_MAIN__F1L32_1Arg2 to i8*, !dbg !68
  %65 = getelementptr i8, i8* %64, i64 80, !dbg !68
  %66 = bitcast i8* %65 to i8***, !dbg !68
  %67 = load i8**, i8*** %66, align 8, !dbg !68
  %68 = load i8*, i8** %67, align 8, !dbg !68
  %69 = getelementptr i8, i8* %68, i64 -8, !dbg !68
  %70 = bitcast i8* %69 to double*, !dbg !68
  %71 = getelementptr double, double* %70, i64 %63, !dbg !68
  %72 = load double, double* %71, align 8, !dbg !68
  %73 = fmul fast double %53, %72, !dbg !68
  %74 = bitcast i64* %__nv_MAIN__F1L32_1Arg2 to i8*, !dbg !68
  %75 = getelementptr i8, i8* %74, i64 64, !dbg !68
  %76 = bitcast i8* %75 to i32**, !dbg !68
  %77 = load i32*, i32** %76, align 8, !dbg !68
  %78 = load i32, i32* %77, align 4, !dbg !68
  %79 = sext i32 %78 to i64, !dbg !68
  %80 = bitcast i64* %__nv_MAIN__F1L32_1Arg2 to i8*, !dbg !68
  %81 = getelementptr i8, i8* %80, i64 200, !dbg !68
  %82 = bitcast i8* %81 to i8**, !dbg !68
  %83 = load i8*, i8** %82, align 8, !dbg !68
  %84 = getelementptr i8, i8* %83, i64 56, !dbg !68
  %85 = bitcast i8* %84 to i64*, !dbg !68
  %86 = load i64, i64* %85, align 8, !dbg !68
  %87 = add nsw i64 %79, %86, !dbg !68
  %88 = bitcast i64* %__nv_MAIN__F1L32_1Arg2 to i8*, !dbg !68
  %89 = getelementptr i8, i8* %88, i64 16, !dbg !68
  %90 = bitcast i8* %89 to i8***, !dbg !68
  %91 = load i8**, i8*** %90, align 8, !dbg !68
  %92 = load i8*, i8** %91, align 8, !dbg !68
  %93 = getelementptr i8, i8* %92, i64 -8, !dbg !68
  %94 = bitcast i8* %93 to double*, !dbg !68
  %95 = getelementptr double, double* %94, i64 %87, !dbg !68
  %96 = load double, double* %95, align 8, !dbg !68
  %97 = fadd fast double %73, %96, !dbg !68
  %98 = bitcast i64* %__nv_MAIN__F1L32_1Arg2 to i8*, !dbg !68
  %99 = getelementptr i8, i8* %98, i64 64, !dbg !68
  %100 = bitcast i8* %99 to i32**, !dbg !68
  %101 = load i32*, i32** %100, align 8, !dbg !68
  %102 = load i32, i32* %101, align 4, !dbg !68
  %103 = sext i32 %102 to i64, !dbg !68
  %104 = bitcast i64* %__nv_MAIN__F1L32_1Arg2 to i8*, !dbg !68
  %105 = getelementptr i8, i8* %104, i64 200, !dbg !68
  %106 = bitcast i8* %105 to i8**, !dbg !68
  %107 = load i8*, i8** %106, align 8, !dbg !68
  %108 = getelementptr i8, i8* %107, i64 56, !dbg !68
  %109 = bitcast i8* %108 to i64*, !dbg !68
  %110 = load i64, i64* %109, align 8, !dbg !68
  %111 = add nsw i64 %103, %110, !dbg !68
  %112 = bitcast i64* %__nv_MAIN__F1L32_1Arg2 to i8*, !dbg !68
  %113 = getelementptr i8, i8* %112, i64 16, !dbg !68
  %114 = bitcast i8* %113 to i8***, !dbg !68
  %115 = load i8**, i8*** %114, align 8, !dbg !68
  %116 = load i8*, i8** %115, align 8, !dbg !68
  %117 = getelementptr i8, i8* %116, i64 -8, !dbg !68
  %118 = bitcast i8* %117 to double*, !dbg !68
  %119 = getelementptr double, double* %118, i64 %111, !dbg !68
  store double %97, double* %119, align 8, !dbg !68
  %120 = bitcast i64* %__nv_MAIN__F1L32_1Arg2 to i8*, !dbg !69
  %121 = getelementptr i8, i8* %120, i64 64, !dbg !69
  %122 = bitcast i8* %121 to i32**, !dbg !69
  %123 = load i32*, i32** %122, align 8, !dbg !69
  %124 = load i32, i32* %123, align 4, !dbg !69
  %125 = add nsw i32 %124, 1, !dbg !69
  %126 = bitcast i64* %__nv_MAIN__F1L32_1Arg2 to i8*, !dbg !69
  %127 = getelementptr i8, i8* %126, i64 64, !dbg !69
  %128 = bitcast i8* %127 to i32**, !dbg !69
  %129 = load i32*, i32** %128, align 8, !dbg !69
  store i32 %125, i32* %129, align 4, !dbg !69
  %130 = load i32, i32* %.di0002p_364, align 4, !dbg !65
  %131 = load i32, i32* %i_340, align 4, !dbg !65
  call void @llvm.dbg.value(metadata i32 %131, metadata !67, metadata !DIExpression()), !dbg !65
  %132 = add nsw i32 %130, %131, !dbg !65
  store i32 %132, i32* %i_340, align 4, !dbg !65
  %133 = load i32, i32* %.dY0002p_361, align 4, !dbg !65
  %134 = sub nsw i32 %133, 1, !dbg !65
  store i32 %134, i32* %.dY0002p_361, align 4, !dbg !65
  %135 = load i32, i32* %.dY0002p_361, align 4, !dbg !65
  %136 = icmp sgt i32 %135, 0, !dbg !65
  br i1 %136, label %L.LB2_369, label %L.LB2_370, !dbg !65

L.LB2_370:                                        ; preds = %L.LB2_369, %L.LB2_539
  br label %L.LB2_360

L.LB2_360:                                        ; preds = %L.LB2_370, %L.LB2_339
  %137 = load i32, i32* %__gtid___nv_MAIN__F1L32_1__510, align 4, !dbg !65
  call void @__kmpc_for_static_fini(i64* null, i32 %137), !dbg !65
  br label %L.LB2_342

L.LB2_342:                                        ; preds = %L.LB2_360
  ret void, !dbg !65
}

declare void @__kmpc_for_static_fini(i64*, i32) #0

declare void @__kmpc_for_static_init_4(i64*, i32, i32, i64*, i64*, i64*, i64*, i32, i32) #0

declare void @f90_dealloc03a_i8(...) #0

declare signext i32 @f90_allocated_i8(...) #0

declare void @f90_alloc04_chka_i8(...) #0

declare void @f90_set_intrin_type_i8(...) #0

declare void @f90_template1_i8(...) #0

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
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB112-linear-orig-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb112_linear_orig_no", scope: !2, file: !3, line: 10, type: !6, scopeLine: 10, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
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
!18 = !DILocation(line: 43, column: 1, scope: !5)
!19 = !DILocation(line: 10, column: 1, scope: !5)
!20 = !DILocalVariable(name: "c", scope: !5, file: !3, type: !21)
!21 = !DICompositeType(tag: DW_TAG_array_type, baseType: !22, size: 64, align: 64, elements: !23)
!22 = !DIBasicType(name: "double precision", size: 64, align: 64, encoding: DW_ATE_float)
!23 = !{!24}
!24 = !DISubrange(count: 0, lowerBound: 1)
!25 = !DILocalVariable(scope: !5, file: !3, type: !26, flags: DIFlagArtificial)
!26 = !DICompositeType(tag: DW_TAG_array_type, baseType: !27, size: 1024, align: 64, elements: !28)
!27 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!28 = !{!29}
!29 = !DISubrange(count: 16, lowerBound: 1)
!30 = !DILocalVariable(name: "b", scope: !5, file: !3, type: !21)
!31 = !DILocalVariable(name: "a", scope: !5, file: !3, type: !21)
!32 = !DILocalVariable(name: "len", scope: !5, file: !3, type: !9)
!33 = !DILocation(line: 18, column: 1, scope: !5)
!34 = !DILocalVariable(name: "i", scope: !5, file: !3, type: !9)
!35 = !DILocation(line: 19, column: 1, scope: !5)
!36 = !DILocalVariable(name: "j", scope: !5, file: !3, type: !9)
!37 = !DILocation(line: 20, column: 1, scope: !5)
!38 = !DILocalVariable(scope: !5, file: !3, type: !27, flags: DIFlagArtificial)
!39 = !DILocation(line: 22, column: 1, scope: !5)
!40 = !DILocation(line: 23, column: 1, scope: !5)
!41 = !DILocation(line: 24, column: 1, scope: !5)
!42 = !DILocation(line: 26, column: 1, scope: !5)
!43 = !DILocation(line: 27, column: 1, scope: !5)
!44 = !DILocation(line: 28, column: 1, scope: !5)
!45 = !DILocation(line: 29, column: 1, scope: !5)
!46 = !DILocation(line: 30, column: 1, scope: !5)
!47 = !DILocation(line: 32, column: 1, scope: !5)
!48 = !DILocation(line: 41, column: 1, scope: !5)
!49 = !DILocation(line: 42, column: 1, scope: !5)
!50 = distinct !DISubprogram(name: "__nv_MAIN__F1L32_1", scope: !2, file: !3, line: 32, type: !51, scopeLine: 32, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!51 = !DISubroutineType(types: !52)
!52 = !{null, !9, !27, !27}
!53 = !DILocalVariable(name: "__nv_MAIN__F1L32_1Arg0", arg: 1, scope: !50, file: !3, type: !9)
!54 = !DILocation(line: 0, scope: !50)
!55 = !DILocalVariable(name: "__nv_MAIN__F1L32_1Arg1", arg: 2, scope: !50, file: !3, type: !27)
!56 = !DILocalVariable(name: "__nv_MAIN__F1L32_1Arg2", arg: 3, scope: !50, file: !3, type: !27)
!57 = !DILocalVariable(name: "omp_nest_lock_kind", scope: !50, file: !3, type: !9)
!58 = !DILocalVariable(name: "omp_sched_static", scope: !50, file: !3, type: !9)
!59 = !DILocalVariable(name: "omp_proc_bind_false", scope: !50, file: !3, type: !9)
!60 = !DILocalVariable(name: "omp_proc_bind_true", scope: !50, file: !3, type: !9)
!61 = !DILocalVariable(name: "omp_lock_hint_none", scope: !50, file: !3, type: !9)
!62 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !50, file: !3, type: !9)
!63 = !DILocalVariable(name: "omp_lock_hint_speculative", scope: !50, file: !3, type: !9)
!64 = !DILocalVariable(name: "dp", scope: !50, file: !3, type: !9)
!65 = !DILocation(line: 36, column: 1, scope: !50)
!66 = !DILocation(line: 33, column: 1, scope: !50)
!67 = !DILocalVariable(name: "i", scope: !50, file: !3, type: !9)
!68 = !DILocation(line: 34, column: 1, scope: !50)
!69 = !DILocation(line: 35, column: 1, scope: !50)

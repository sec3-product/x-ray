; ModuleID = '/tmp/DRB111-linearmissing-orig-yes-b30e63.ll'
source_filename = "/tmp/DRB111-linearmissing-orig-yes-b30e63.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%astruct.dt84 = type <{ i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8* }>

@.C352_MAIN_ = internal constant i64 50
@.C308_MAIN_ = internal constant i32 14
@.C351_MAIN_ = internal constant [7 x i8] c"c(50) ="
@.C348_MAIN_ = internal constant i32 6
@.C345_MAIN_ = internal constant [58 x i8] c"micro-benchmarks-fortran/DRB111-linearmissing-orig-yes.f95"
@.C347_MAIN_ = internal constant i32 41
@.C337_MAIN_ = internal constant double 7.000000e+00
@.C336_MAIN_ = internal constant double 3.000000e+00
@.C293_MAIN_ = internal constant double 2.000000e+00
@.C300_MAIN_ = internal constant i32 8
@.C285_MAIN_ = internal constant i32 1
@.C309_MAIN_ = internal constant i32 28
@.C362_MAIN_ = internal constant i64 8
@.C361_MAIN_ = internal constant i64 28
@.C331_MAIN_ = internal constant i32 100
@.C286_MAIN_ = internal constant i64 1
@.C284_MAIN_ = internal constant i64 0
@.C283_MAIN_ = internal constant i32 0
@.C285___nv_MAIN__F1L34_1 = internal constant i32 1
@.C283___nv_MAIN__F1L34_1 = internal constant i32 0

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__487 = alloca i32, align 4
  %.Z0974_335 = alloca double*, align 8
  %"c$sd3_365" = alloca [16 x i64], align 8
  %.Z0968_334 = alloca double*, align 8
  %"b$sd2_364" = alloca [16 x i64], align 8
  %.Z0967_333 = alloca double*, align 8
  %"a$sd1_360" = alloca [16 x i64], align 8
  %len_332 = alloca i32, align 4
  %i_310 = alloca i32, align 4
  %j_311 = alloca i32, align 4
  %z_b_0_313 = alloca i64, align 8
  %z_b_1_314 = alloca i64, align 8
  %z_e_62_317 = alloca i64, align 8
  %z_b_2_315 = alloca i64, align 8
  %z_b_3_316 = alloca i64, align 8
  %z_b_4_320 = alloca i64, align 8
  %z_b_5_321 = alloca i64, align 8
  %z_e_69_324 = alloca i64, align 8
  %z_b_6_322 = alloca i64, align 8
  %z_b_7_323 = alloca i64, align 8
  %z_b_8_326 = alloca i64, align 8
  %z_b_9_327 = alloca i64, align 8
  %z_e_76_330 = alloca i64, align 8
  %z_b_10_328 = alloca i64, align 8
  %z_b_11_329 = alloca i64, align 8
  %.dY0001_373 = alloca i32, align 4
  %.uplevelArgPack0001_432 = alloca %astruct.dt84, align 16
  %z__io_350 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 8, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !14, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !15, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 8, metadata !16, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 8, metadata !17, metadata !DIExpression()), !dbg !10
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !18
  store i32 %0, i32* %__gtid_MAIN__487, align 4, !dbg !18
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !19
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !19
  call void (i8*, ...) %2(i8* %1), !dbg !19
  call void @llvm.dbg.declare(metadata double** %.Z0974_335, metadata !20, metadata !DIExpression(DW_OP_deref)), !dbg !10
  %3 = bitcast double** %.Z0974_335 to i8**, !dbg !19
  store i8* null, i8** %3, align 8, !dbg !19
  call void @llvm.dbg.declare(metadata [16 x i64]* %"c$sd3_365", metadata !25, metadata !DIExpression()), !dbg !10
  %4 = bitcast [16 x i64]* %"c$sd3_365" to i64*, !dbg !19
  store i64 0, i64* %4, align 8, !dbg !19
  call void @llvm.dbg.declare(metadata double** %.Z0968_334, metadata !30, metadata !DIExpression(DW_OP_deref)), !dbg !10
  %5 = bitcast double** %.Z0968_334 to i8**, !dbg !19
  store i8* null, i8** %5, align 8, !dbg !19
  call void @llvm.dbg.declare(metadata [16 x i64]* %"b$sd2_364", metadata !25, metadata !DIExpression()), !dbg !10
  %6 = bitcast [16 x i64]* %"b$sd2_364" to i64*, !dbg !19
  store i64 0, i64* %6, align 8, !dbg !19
  call void @llvm.dbg.declare(metadata double** %.Z0967_333, metadata !31, metadata !DIExpression(DW_OP_deref)), !dbg !10
  %7 = bitcast double** %.Z0967_333 to i8**, !dbg !19
  store i8* null, i8** %7, align 8, !dbg !19
  call void @llvm.dbg.declare(metadata [16 x i64]* %"a$sd1_360", metadata !25, metadata !DIExpression()), !dbg !10
  %8 = bitcast [16 x i64]* %"a$sd1_360" to i64*, !dbg !19
  store i64 0, i64* %8, align 8, !dbg !19
  br label %L.LB1_398

L.LB1_398:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %len_332, metadata !32, metadata !DIExpression()), !dbg !10
  store i32 100, i32* %len_332, align 4, !dbg !33
  call void @llvm.dbg.declare(metadata i32* %i_310, metadata !34, metadata !DIExpression()), !dbg !10
  store i32 0, i32* %i_310, align 4, !dbg !35
  call void @llvm.dbg.declare(metadata i32* %j_311, metadata !36, metadata !DIExpression()), !dbg !10
  store i32 0, i32* %j_311, align 4, !dbg !37
  call void @llvm.dbg.declare(metadata i64* %z_b_0_313, metadata !38, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_0_313, align 8, !dbg !39
  %9 = load i32, i32* %len_332, align 4, !dbg !39
  call void @llvm.dbg.value(metadata i32 %9, metadata !32, metadata !DIExpression()), !dbg !10
  %10 = sext i32 %9 to i64, !dbg !39
  call void @llvm.dbg.declare(metadata i64* %z_b_1_314, metadata !38, metadata !DIExpression()), !dbg !10
  store i64 %10, i64* %z_b_1_314, align 8, !dbg !39
  %11 = load i64, i64* %z_b_1_314, align 8, !dbg !39
  call void @llvm.dbg.value(metadata i64 %11, metadata !38, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_62_317, metadata !38, metadata !DIExpression()), !dbg !10
  store i64 %11, i64* %z_e_62_317, align 8, !dbg !39
  %12 = bitcast [16 x i64]* %"a$sd1_360" to i8*, !dbg !39
  %13 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !39
  %14 = bitcast i64* @.C361_MAIN_ to i8*, !dbg !39
  %15 = bitcast i64* @.C362_MAIN_ to i8*, !dbg !39
  %16 = bitcast i64* %z_b_0_313 to i8*, !dbg !39
  %17 = bitcast i64* %z_b_1_314 to i8*, !dbg !39
  %18 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !39
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %18(i8* %12, i8* %13, i8* %14, i8* %15, i8* %16, i8* %17), !dbg !39
  %19 = bitcast [16 x i64]* %"a$sd1_360" to i8*, !dbg !39
  %20 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !39
  call void (i8*, i32, ...) %20(i8* %19, i32 28), !dbg !39
  %21 = load i64, i64* %z_b_1_314, align 8, !dbg !39
  call void @llvm.dbg.value(metadata i64 %21, metadata !38, metadata !DIExpression()), !dbg !10
  %22 = load i64, i64* %z_b_0_313, align 8, !dbg !39
  call void @llvm.dbg.value(metadata i64 %22, metadata !38, metadata !DIExpression()), !dbg !10
  %23 = sub nsw i64 %22, 1, !dbg !39
  %24 = sub nsw i64 %21, %23, !dbg !39
  call void @llvm.dbg.declare(metadata i64* %z_b_2_315, metadata !38, metadata !DIExpression()), !dbg !10
  store i64 %24, i64* %z_b_2_315, align 8, !dbg !39
  %25 = load i64, i64* %z_b_0_313, align 8, !dbg !39
  call void @llvm.dbg.value(metadata i64 %25, metadata !38, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_b_3_316, metadata !38, metadata !DIExpression()), !dbg !10
  store i64 %25, i64* %z_b_3_316, align 8, !dbg !39
  %26 = bitcast i64* %z_b_2_315 to i8*, !dbg !39
  %27 = bitcast i64* @.C361_MAIN_ to i8*, !dbg !39
  %28 = bitcast i64* @.C362_MAIN_ to i8*, !dbg !39
  %29 = bitcast double** %.Z0967_333 to i8*, !dbg !39
  %30 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !39
  %31 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !39
  %32 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !39
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %32(i8* %26, i8* %27, i8* %28, i8* null, i8* %29, i8* null, i8* %30, i8* %31, i8* null, i64 0), !dbg !39
  call void @llvm.dbg.declare(metadata i64* %z_b_4_320, metadata !38, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_4_320, align 8, !dbg !40
  %33 = load i32, i32* %len_332, align 4, !dbg !40
  call void @llvm.dbg.value(metadata i32 %33, metadata !32, metadata !DIExpression()), !dbg !10
  %34 = sext i32 %33 to i64, !dbg !40
  call void @llvm.dbg.declare(metadata i64* %z_b_5_321, metadata !38, metadata !DIExpression()), !dbg !10
  store i64 %34, i64* %z_b_5_321, align 8, !dbg !40
  %35 = load i64, i64* %z_b_5_321, align 8, !dbg !40
  call void @llvm.dbg.value(metadata i64 %35, metadata !38, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_69_324, metadata !38, metadata !DIExpression()), !dbg !10
  store i64 %35, i64* %z_e_69_324, align 8, !dbg !40
  %36 = bitcast [16 x i64]* %"b$sd2_364" to i8*, !dbg !40
  %37 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !40
  %38 = bitcast i64* @.C361_MAIN_ to i8*, !dbg !40
  %39 = bitcast i64* @.C362_MAIN_ to i8*, !dbg !40
  %40 = bitcast i64* %z_b_4_320 to i8*, !dbg !40
  %41 = bitcast i64* %z_b_5_321 to i8*, !dbg !40
  %42 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !40
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %42(i8* %36, i8* %37, i8* %38, i8* %39, i8* %40, i8* %41), !dbg !40
  %43 = bitcast [16 x i64]* %"b$sd2_364" to i8*, !dbg !40
  %44 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !40
  call void (i8*, i32, ...) %44(i8* %43, i32 28), !dbg !40
  %45 = load i64, i64* %z_b_5_321, align 8, !dbg !40
  call void @llvm.dbg.value(metadata i64 %45, metadata !38, metadata !DIExpression()), !dbg !10
  %46 = load i64, i64* %z_b_4_320, align 8, !dbg !40
  call void @llvm.dbg.value(metadata i64 %46, metadata !38, metadata !DIExpression()), !dbg !10
  %47 = sub nsw i64 %46, 1, !dbg !40
  %48 = sub nsw i64 %45, %47, !dbg !40
  call void @llvm.dbg.declare(metadata i64* %z_b_6_322, metadata !38, metadata !DIExpression()), !dbg !10
  store i64 %48, i64* %z_b_6_322, align 8, !dbg !40
  %49 = load i64, i64* %z_b_4_320, align 8, !dbg !40
  call void @llvm.dbg.value(metadata i64 %49, metadata !38, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_b_7_323, metadata !38, metadata !DIExpression()), !dbg !10
  store i64 %49, i64* %z_b_7_323, align 8, !dbg !40
  %50 = bitcast i64* %z_b_6_322 to i8*, !dbg !40
  %51 = bitcast i64* @.C361_MAIN_ to i8*, !dbg !40
  %52 = bitcast i64* @.C362_MAIN_ to i8*, !dbg !40
  %53 = bitcast double** %.Z0968_334 to i8*, !dbg !40
  %54 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !40
  %55 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !40
  %56 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !40
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %56(i8* %50, i8* %51, i8* %52, i8* null, i8* %53, i8* null, i8* %54, i8* %55, i8* null, i64 0), !dbg !40
  call void @llvm.dbg.declare(metadata i64* %z_b_8_326, metadata !38, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_8_326, align 8, !dbg !41
  %57 = load i32, i32* %len_332, align 4, !dbg !41
  call void @llvm.dbg.value(metadata i32 %57, metadata !32, metadata !DIExpression()), !dbg !10
  %58 = sext i32 %57 to i64, !dbg !41
  call void @llvm.dbg.declare(metadata i64* %z_b_9_327, metadata !38, metadata !DIExpression()), !dbg !10
  store i64 %58, i64* %z_b_9_327, align 8, !dbg !41
  %59 = load i64, i64* %z_b_9_327, align 8, !dbg !41
  call void @llvm.dbg.value(metadata i64 %59, metadata !38, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_76_330, metadata !38, metadata !DIExpression()), !dbg !10
  store i64 %59, i64* %z_e_76_330, align 8, !dbg !41
  %60 = bitcast [16 x i64]* %"c$sd3_365" to i8*, !dbg !41
  %61 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !41
  %62 = bitcast i64* @.C361_MAIN_ to i8*, !dbg !41
  %63 = bitcast i64* @.C362_MAIN_ to i8*, !dbg !41
  %64 = bitcast i64* %z_b_8_326 to i8*, !dbg !41
  %65 = bitcast i64* %z_b_9_327 to i8*, !dbg !41
  %66 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !41
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %66(i8* %60, i8* %61, i8* %62, i8* %63, i8* %64, i8* %65), !dbg !41
  %67 = bitcast [16 x i64]* %"c$sd3_365" to i8*, !dbg !41
  %68 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !41
  call void (i8*, i32, ...) %68(i8* %67, i32 28), !dbg !41
  %69 = load i64, i64* %z_b_9_327, align 8, !dbg !41
  call void @llvm.dbg.value(metadata i64 %69, metadata !38, metadata !DIExpression()), !dbg !10
  %70 = load i64, i64* %z_b_8_326, align 8, !dbg !41
  call void @llvm.dbg.value(metadata i64 %70, metadata !38, metadata !DIExpression()), !dbg !10
  %71 = sub nsw i64 %70, 1, !dbg !41
  %72 = sub nsw i64 %69, %71, !dbg !41
  call void @llvm.dbg.declare(metadata i64* %z_b_10_328, metadata !38, metadata !DIExpression()), !dbg !10
  store i64 %72, i64* %z_b_10_328, align 8, !dbg !41
  %73 = load i64, i64* %z_b_8_326, align 8, !dbg !41
  call void @llvm.dbg.value(metadata i64 %73, metadata !38, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_b_11_329, metadata !38, metadata !DIExpression()), !dbg !10
  store i64 %73, i64* %z_b_11_329, align 8, !dbg !41
  %74 = bitcast i64* %z_b_10_328 to i8*, !dbg !41
  %75 = bitcast i64* @.C361_MAIN_ to i8*, !dbg !41
  %76 = bitcast i64* @.C362_MAIN_ to i8*, !dbg !41
  %77 = bitcast double** %.Z0974_335 to i8*, !dbg !41
  %78 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !41
  %79 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !41
  %80 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !41
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %80(i8* %74, i8* %75, i8* %76, i8* null, i8* %77, i8* null, i8* %78, i8* %79, i8* null, i64 0), !dbg !41
  %81 = load i32, i32* %len_332, align 4, !dbg !42
  call void @llvm.dbg.value(metadata i32 %81, metadata !32, metadata !DIExpression()), !dbg !10
  store i32 %81, i32* %.dY0001_373, align 4, !dbg !42
  store i32 1, i32* %i_310, align 4, !dbg !42
  %82 = load i32, i32* %.dY0001_373, align 4, !dbg !42
  %83 = icmp sle i32 %82, 0, !dbg !42
  br i1 %83, label %L.LB1_372, label %L.LB1_371, !dbg !42

L.LB1_371:                                        ; preds = %L.LB1_371, %L.LB1_398
  %84 = load i32, i32* %i_310, align 4, !dbg !43
  call void @llvm.dbg.value(metadata i32 %84, metadata !34, metadata !DIExpression()), !dbg !10
  %85 = sitofp i32 %84 to double, !dbg !43
  %86 = fdiv fast double %85, 2.000000e+00, !dbg !43
  %87 = load i32, i32* %i_310, align 4, !dbg !43
  call void @llvm.dbg.value(metadata i32 %87, metadata !34, metadata !DIExpression()), !dbg !10
  %88 = sext i32 %87 to i64, !dbg !43
  %89 = bitcast [16 x i64]* %"a$sd1_360" to i8*, !dbg !43
  %90 = getelementptr i8, i8* %89, i64 56, !dbg !43
  %91 = bitcast i8* %90 to i64*, !dbg !43
  %92 = load i64, i64* %91, align 8, !dbg !43
  %93 = add nsw i64 %88, %92, !dbg !43
  %94 = load double*, double** %.Z0967_333, align 8, !dbg !43
  call void @llvm.dbg.value(metadata double* %94, metadata !31, metadata !DIExpression()), !dbg !10
  %95 = bitcast double* %94 to i8*, !dbg !43
  %96 = getelementptr i8, i8* %95, i64 -8, !dbg !43
  %97 = bitcast i8* %96 to double*, !dbg !43
  %98 = getelementptr double, double* %97, i64 %93, !dbg !43
  store double %86, double* %98, align 8, !dbg !43
  %99 = load i32, i32* %i_310, align 4, !dbg !44
  call void @llvm.dbg.value(metadata i32 %99, metadata !34, metadata !DIExpression()), !dbg !10
  %100 = sitofp i32 %99 to double, !dbg !44
  %101 = fdiv fast double %100, 3.000000e+00, !dbg !44
  %102 = load i32, i32* %i_310, align 4, !dbg !44
  call void @llvm.dbg.value(metadata i32 %102, metadata !34, metadata !DIExpression()), !dbg !10
  %103 = sext i32 %102 to i64, !dbg !44
  %104 = bitcast [16 x i64]* %"b$sd2_364" to i8*, !dbg !44
  %105 = getelementptr i8, i8* %104, i64 56, !dbg !44
  %106 = bitcast i8* %105 to i64*, !dbg !44
  %107 = load i64, i64* %106, align 8, !dbg !44
  %108 = add nsw i64 %103, %107, !dbg !44
  %109 = load double*, double** %.Z0968_334, align 8, !dbg !44
  call void @llvm.dbg.value(metadata double* %109, metadata !30, metadata !DIExpression()), !dbg !10
  %110 = bitcast double* %109 to i8*, !dbg !44
  %111 = getelementptr i8, i8* %110, i64 -8, !dbg !44
  %112 = bitcast i8* %111 to double*, !dbg !44
  %113 = getelementptr double, double* %112, i64 %108, !dbg !44
  store double %101, double* %113, align 8, !dbg !44
  %114 = load i32, i32* %i_310, align 4, !dbg !45
  call void @llvm.dbg.value(metadata i32 %114, metadata !34, metadata !DIExpression()), !dbg !10
  %115 = sitofp i32 %114 to double, !dbg !45
  %116 = fdiv fast double %115, 7.000000e+00, !dbg !45
  %117 = load i32, i32* %i_310, align 4, !dbg !45
  call void @llvm.dbg.value(metadata i32 %117, metadata !34, metadata !DIExpression()), !dbg !10
  %118 = sext i32 %117 to i64, !dbg !45
  %119 = bitcast [16 x i64]* %"c$sd3_365" to i8*, !dbg !45
  %120 = getelementptr i8, i8* %119, i64 56, !dbg !45
  %121 = bitcast i8* %120 to i64*, !dbg !45
  %122 = load i64, i64* %121, align 8, !dbg !45
  %123 = add nsw i64 %118, %122, !dbg !45
  %124 = load double*, double** %.Z0974_335, align 8, !dbg !45
  call void @llvm.dbg.value(metadata double* %124, metadata !20, metadata !DIExpression()), !dbg !10
  %125 = bitcast double* %124 to i8*, !dbg !45
  %126 = getelementptr i8, i8* %125, i64 -8, !dbg !45
  %127 = bitcast i8* %126 to double*, !dbg !45
  %128 = getelementptr double, double* %127, i64 %123, !dbg !45
  store double %116, double* %128, align 8, !dbg !45
  %129 = load i32, i32* %i_310, align 4, !dbg !46
  call void @llvm.dbg.value(metadata i32 %129, metadata !34, metadata !DIExpression()), !dbg !10
  %130 = add nsw i32 %129, 1, !dbg !46
  store i32 %130, i32* %i_310, align 4, !dbg !46
  %131 = load i32, i32* %.dY0001_373, align 4, !dbg !46
  %132 = sub nsw i32 %131, 1, !dbg !46
  store i32 %132, i32* %.dY0001_373, align 4, !dbg !46
  %133 = load i32, i32* %.dY0001_373, align 4, !dbg !46
  %134 = icmp sgt i32 %133, 0, !dbg !46
  br i1 %134, label %L.LB1_371, label %L.LB1_372, !dbg !46

L.LB1_372:                                        ; preds = %L.LB1_371, %L.LB1_398
  %135 = bitcast i32* %len_332 to i8*, !dbg !47
  %136 = bitcast %astruct.dt84* %.uplevelArgPack0001_432 to i8**, !dbg !47
  store i8* %135, i8** %136, align 8, !dbg !47
  %137 = bitcast double** %.Z0974_335 to i8*, !dbg !47
  %138 = bitcast %astruct.dt84* %.uplevelArgPack0001_432 to i8*, !dbg !47
  %139 = getelementptr i8, i8* %138, i64 8, !dbg !47
  %140 = bitcast i8* %139 to i8**, !dbg !47
  store i8* %137, i8** %140, align 8, !dbg !47
  %141 = bitcast double** %.Z0974_335 to i8*, !dbg !47
  %142 = bitcast %astruct.dt84* %.uplevelArgPack0001_432 to i8*, !dbg !47
  %143 = getelementptr i8, i8* %142, i64 16, !dbg !47
  %144 = bitcast i8* %143 to i8**, !dbg !47
  store i8* %141, i8** %144, align 8, !dbg !47
  %145 = bitcast i64* %z_b_8_326 to i8*, !dbg !47
  %146 = bitcast %astruct.dt84* %.uplevelArgPack0001_432 to i8*, !dbg !47
  %147 = getelementptr i8, i8* %146, i64 24, !dbg !47
  %148 = bitcast i8* %147 to i8**, !dbg !47
  store i8* %145, i8** %148, align 8, !dbg !47
  %149 = bitcast i64* %z_b_9_327 to i8*, !dbg !47
  %150 = bitcast %astruct.dt84* %.uplevelArgPack0001_432 to i8*, !dbg !47
  %151 = getelementptr i8, i8* %150, i64 32, !dbg !47
  %152 = bitcast i8* %151 to i8**, !dbg !47
  store i8* %149, i8** %152, align 8, !dbg !47
  %153 = bitcast i64* %z_e_76_330 to i8*, !dbg !47
  %154 = bitcast %astruct.dt84* %.uplevelArgPack0001_432 to i8*, !dbg !47
  %155 = getelementptr i8, i8* %154, i64 40, !dbg !47
  %156 = bitcast i8* %155 to i8**, !dbg !47
  store i8* %153, i8** %156, align 8, !dbg !47
  %157 = bitcast i64* %z_b_10_328 to i8*, !dbg !47
  %158 = bitcast %astruct.dt84* %.uplevelArgPack0001_432 to i8*, !dbg !47
  %159 = getelementptr i8, i8* %158, i64 48, !dbg !47
  %160 = bitcast i8* %159 to i8**, !dbg !47
  store i8* %157, i8** %160, align 8, !dbg !47
  %161 = bitcast i64* %z_b_11_329 to i8*, !dbg !47
  %162 = bitcast %astruct.dt84* %.uplevelArgPack0001_432 to i8*, !dbg !47
  %163 = getelementptr i8, i8* %162, i64 56, !dbg !47
  %164 = bitcast i8* %163 to i8**, !dbg !47
  store i8* %161, i8** %164, align 8, !dbg !47
  %165 = bitcast i32* %j_311 to i8*, !dbg !47
  %166 = bitcast %astruct.dt84* %.uplevelArgPack0001_432 to i8*, !dbg !47
  %167 = getelementptr i8, i8* %166, i64 64, !dbg !47
  %168 = bitcast i8* %167 to i8**, !dbg !47
  store i8* %165, i8** %168, align 8, !dbg !47
  %169 = bitcast double** %.Z0967_333 to i8*, !dbg !47
  %170 = bitcast %astruct.dt84* %.uplevelArgPack0001_432 to i8*, !dbg !47
  %171 = getelementptr i8, i8* %170, i64 72, !dbg !47
  %172 = bitcast i8* %171 to i8**, !dbg !47
  store i8* %169, i8** %172, align 8, !dbg !47
  %173 = bitcast double** %.Z0967_333 to i8*, !dbg !47
  %174 = bitcast %astruct.dt84* %.uplevelArgPack0001_432 to i8*, !dbg !47
  %175 = getelementptr i8, i8* %174, i64 80, !dbg !47
  %176 = bitcast i8* %175 to i8**, !dbg !47
  store i8* %173, i8** %176, align 8, !dbg !47
  %177 = bitcast i64* %z_b_0_313 to i8*, !dbg !47
  %178 = bitcast %astruct.dt84* %.uplevelArgPack0001_432 to i8*, !dbg !47
  %179 = getelementptr i8, i8* %178, i64 88, !dbg !47
  %180 = bitcast i8* %179 to i8**, !dbg !47
  store i8* %177, i8** %180, align 8, !dbg !47
  %181 = bitcast i64* %z_b_1_314 to i8*, !dbg !47
  %182 = bitcast %astruct.dt84* %.uplevelArgPack0001_432 to i8*, !dbg !47
  %183 = getelementptr i8, i8* %182, i64 96, !dbg !47
  %184 = bitcast i8* %183 to i8**, !dbg !47
  store i8* %181, i8** %184, align 8, !dbg !47
  %185 = bitcast i64* %z_e_62_317 to i8*, !dbg !47
  %186 = bitcast %astruct.dt84* %.uplevelArgPack0001_432 to i8*, !dbg !47
  %187 = getelementptr i8, i8* %186, i64 104, !dbg !47
  %188 = bitcast i8* %187 to i8**, !dbg !47
  store i8* %185, i8** %188, align 8, !dbg !47
  %189 = bitcast i64* %z_b_2_315 to i8*, !dbg !47
  %190 = bitcast %astruct.dt84* %.uplevelArgPack0001_432 to i8*, !dbg !47
  %191 = getelementptr i8, i8* %190, i64 112, !dbg !47
  %192 = bitcast i8* %191 to i8**, !dbg !47
  store i8* %189, i8** %192, align 8, !dbg !47
  %193 = bitcast i64* %z_b_3_316 to i8*, !dbg !47
  %194 = bitcast %astruct.dt84* %.uplevelArgPack0001_432 to i8*, !dbg !47
  %195 = getelementptr i8, i8* %194, i64 120, !dbg !47
  %196 = bitcast i8* %195 to i8**, !dbg !47
  store i8* %193, i8** %196, align 8, !dbg !47
  %197 = bitcast double** %.Z0968_334 to i8*, !dbg !47
  %198 = bitcast %astruct.dt84* %.uplevelArgPack0001_432 to i8*, !dbg !47
  %199 = getelementptr i8, i8* %198, i64 128, !dbg !47
  %200 = bitcast i8* %199 to i8**, !dbg !47
  store i8* %197, i8** %200, align 8, !dbg !47
  %201 = bitcast double** %.Z0968_334 to i8*, !dbg !47
  %202 = bitcast %astruct.dt84* %.uplevelArgPack0001_432 to i8*, !dbg !47
  %203 = getelementptr i8, i8* %202, i64 136, !dbg !47
  %204 = bitcast i8* %203 to i8**, !dbg !47
  store i8* %201, i8** %204, align 8, !dbg !47
  %205 = bitcast i64* %z_b_4_320 to i8*, !dbg !47
  %206 = bitcast %astruct.dt84* %.uplevelArgPack0001_432 to i8*, !dbg !47
  %207 = getelementptr i8, i8* %206, i64 144, !dbg !47
  %208 = bitcast i8* %207 to i8**, !dbg !47
  store i8* %205, i8** %208, align 8, !dbg !47
  %209 = bitcast i64* %z_b_5_321 to i8*, !dbg !47
  %210 = bitcast %astruct.dt84* %.uplevelArgPack0001_432 to i8*, !dbg !47
  %211 = getelementptr i8, i8* %210, i64 152, !dbg !47
  %212 = bitcast i8* %211 to i8**, !dbg !47
  store i8* %209, i8** %212, align 8, !dbg !47
  %213 = bitcast i64* %z_e_69_324 to i8*, !dbg !47
  %214 = bitcast %astruct.dt84* %.uplevelArgPack0001_432 to i8*, !dbg !47
  %215 = getelementptr i8, i8* %214, i64 160, !dbg !47
  %216 = bitcast i8* %215 to i8**, !dbg !47
  store i8* %213, i8** %216, align 8, !dbg !47
  %217 = bitcast i64* %z_b_6_322 to i8*, !dbg !47
  %218 = bitcast %astruct.dt84* %.uplevelArgPack0001_432 to i8*, !dbg !47
  %219 = getelementptr i8, i8* %218, i64 168, !dbg !47
  %220 = bitcast i8* %219 to i8**, !dbg !47
  store i8* %217, i8** %220, align 8, !dbg !47
  %221 = bitcast i64* %z_b_7_323 to i8*, !dbg !47
  %222 = bitcast %astruct.dt84* %.uplevelArgPack0001_432 to i8*, !dbg !47
  %223 = getelementptr i8, i8* %222, i64 176, !dbg !47
  %224 = bitcast i8* %223 to i8**, !dbg !47
  store i8* %221, i8** %224, align 8, !dbg !47
  %225 = bitcast [16 x i64]* %"a$sd1_360" to i8*, !dbg !47
  %226 = bitcast %astruct.dt84* %.uplevelArgPack0001_432 to i8*, !dbg !47
  %227 = getelementptr i8, i8* %226, i64 184, !dbg !47
  %228 = bitcast i8* %227 to i8**, !dbg !47
  store i8* %225, i8** %228, align 8, !dbg !47
  %229 = bitcast [16 x i64]* %"b$sd2_364" to i8*, !dbg !47
  %230 = bitcast %astruct.dt84* %.uplevelArgPack0001_432 to i8*, !dbg !47
  %231 = getelementptr i8, i8* %230, i64 192, !dbg !47
  %232 = bitcast i8* %231 to i8**, !dbg !47
  store i8* %229, i8** %232, align 8, !dbg !47
  %233 = bitcast [16 x i64]* %"c$sd3_365" to i8*, !dbg !47
  %234 = bitcast %astruct.dt84* %.uplevelArgPack0001_432 to i8*, !dbg !47
  %235 = getelementptr i8, i8* %234, i64 200, !dbg !47
  %236 = bitcast i8* %235 to i8**, !dbg !47
  store i8* %233, i8** %236, align 8, !dbg !47
  br label %L.LB1_485, !dbg !47

L.LB1_485:                                        ; preds = %L.LB1_372
  %237 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L34_1_ to i64*, !dbg !47
  %238 = bitcast %astruct.dt84* %.uplevelArgPack0001_432 to i64*, !dbg !47
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %237, i64* %238), !dbg !47
  call void (...) @_mp_bcs_nest(), !dbg !48
  %239 = bitcast i32* @.C347_MAIN_ to i8*, !dbg !48
  %240 = bitcast [58 x i8]* @.C345_MAIN_ to i8*, !dbg !48
  %241 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !48
  call void (i8*, i8*, i64, ...) %241(i8* %239, i8* %240, i64 58), !dbg !48
  %242 = bitcast i32* @.C348_MAIN_ to i8*, !dbg !48
  %243 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !48
  %244 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !48
  %245 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !48
  %246 = call i32 (i8*, i8*, i8*, i8*, ...) %245(i8* %242, i8* null, i8* %243, i8* %244), !dbg !48
  call void @llvm.dbg.declare(metadata i32* %z__io_350, metadata !49, metadata !DIExpression()), !dbg !10
  store i32 %246, i32* %z__io_350, align 4, !dbg !48
  %247 = bitcast [7 x i8]* @.C351_MAIN_ to i8*, !dbg !48
  %248 = bitcast i32 (...)* @f90io_sc_ch_ldw to i32 (i8*, i32, i64, ...)*, !dbg !48
  %249 = call i32 (i8*, i32, i64, ...) %248(i8* %247, i32 14, i64 7), !dbg !48
  store i32 %249, i32* %z__io_350, align 4, !dbg !48
  %250 = bitcast [16 x i64]* %"c$sd3_365" to i8*, !dbg !48
  %251 = getelementptr i8, i8* %250, i64 56, !dbg !48
  %252 = bitcast i8* %251 to i64*, !dbg !48
  %253 = load i64, i64* %252, align 8, !dbg !48
  %254 = load double*, double** %.Z0974_335, align 8, !dbg !48
  call void @llvm.dbg.value(metadata double* %254, metadata !20, metadata !DIExpression()), !dbg !10
  %255 = bitcast double* %254 to i8*, !dbg !48
  %256 = getelementptr i8, i8* %255, i64 392, !dbg !48
  %257 = bitcast i8* %256 to double*, !dbg !48
  %258 = getelementptr double, double* %257, i64 %253, !dbg !48
  %259 = load double, double* %258, align 8, !dbg !48
  %260 = bitcast i32 (...)* @f90io_sc_d_ldw to i32 (double, i32, ...)*, !dbg !48
  %261 = call i32 (double, i32, ...) %260(double %259, i32 28), !dbg !48
  store i32 %261, i32* %z__io_350, align 4, !dbg !48
  %262 = call i32 (...) @f90io_ldw_end(), !dbg !48
  store i32 %262, i32* %z__io_350, align 4, !dbg !48
  call void (...) @_mp_ecs_nest(), !dbg !48
  %263 = load double*, double** %.Z0967_333, align 8, !dbg !50
  call void @llvm.dbg.value(metadata double* %263, metadata !31, metadata !DIExpression()), !dbg !10
  %264 = bitcast double* %263 to i8*, !dbg !50
  %265 = bitcast i32 (...)* @f90_allocated_i8 to i32 (i8*, ...)*, !dbg !50
  %266 = call i32 (i8*, ...) %265(i8* %264), !dbg !50
  %267 = and i32 %266, 1, !dbg !50
  %268 = icmp eq i32 %267, 0, !dbg !50
  br i1 %268, label %L.LB1_386, label %L.LB1_519, !dbg !50

L.LB1_519:                                        ; preds = %L.LB1_485
  %269 = load double*, double** %.Z0967_333, align 8, !dbg !50
  call void @llvm.dbg.value(metadata double* %269, metadata !31, metadata !DIExpression()), !dbg !10
  %270 = bitcast double* %269 to i8*, !dbg !50
  %271 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !50
  %272 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !50
  call void (i8*, i8*, i8*, i8*, i64, ...) %272(i8* null, i8* %270, i8* %271, i8* null, i64 0), !dbg !50
  %273 = bitcast double** %.Z0967_333 to i8**, !dbg !50
  store i8* null, i8** %273, align 8, !dbg !50
  %274 = bitcast [16 x i64]* %"a$sd1_360" to i64*, !dbg !50
  store i64 0, i64* %274, align 8, !dbg !50
  br label %L.LB1_386

L.LB1_386:                                        ; preds = %L.LB1_519, %L.LB1_485
  %275 = load double*, double** %.Z0968_334, align 8, !dbg !51
  call void @llvm.dbg.value(metadata double* %275, metadata !30, metadata !DIExpression()), !dbg !10
  %276 = bitcast double* %275 to i8*, !dbg !51
  %277 = bitcast i32 (...)* @f90_allocated_i8 to i32 (i8*, ...)*, !dbg !51
  %278 = call i32 (i8*, ...) %277(i8* %276), !dbg !51
  %279 = and i32 %278, 1, !dbg !51
  %280 = icmp eq i32 %279, 0, !dbg !51
  br i1 %280, label %L.LB1_389, label %L.LB1_520, !dbg !51

L.LB1_520:                                        ; preds = %L.LB1_386
  %281 = load double*, double** %.Z0968_334, align 8, !dbg !51
  call void @llvm.dbg.value(metadata double* %281, metadata !30, metadata !DIExpression()), !dbg !10
  %282 = bitcast double* %281 to i8*, !dbg !51
  %283 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !51
  %284 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !51
  call void (i8*, i8*, i8*, i8*, i64, ...) %284(i8* null, i8* %282, i8* %283, i8* null, i64 0), !dbg !51
  %285 = bitcast double** %.Z0968_334 to i8**, !dbg !51
  store i8* null, i8** %285, align 8, !dbg !51
  %286 = bitcast [16 x i64]* %"b$sd2_364" to i64*, !dbg !51
  store i64 0, i64* %286, align 8, !dbg !51
  br label %L.LB1_389

L.LB1_389:                                        ; preds = %L.LB1_520, %L.LB1_386
  ret void, !dbg !18
}

define internal void @__nv_MAIN__F1L34_1_(i32* %__nv_MAIN__F1L34_1Arg0, i64* %__nv_MAIN__F1L34_1Arg1, i64* %__nv_MAIN__F1L34_1Arg2) #0 !dbg !52 {
L.entry:
  %__gtid___nv_MAIN__F1L34_1__539 = alloca i32, align 4
  %.i0000p_342 = alloca i32, align 4
  %i_341 = alloca i32, align 4
  %.du0002p_377 = alloca i32, align 4
  %.de0002p_378 = alloca i32, align 4
  %.di0002p_379 = alloca i32, align 4
  %.ds0002p_380 = alloca i32, align 4
  %.dl0002p_382 = alloca i32, align 4
  %.dl0002p.copy_533 = alloca i32, align 4
  %.de0002p.copy_534 = alloca i32, align 4
  %.ds0002p.copy_535 = alloca i32, align 4
  %.dX0002p_381 = alloca i32, align 4
  %.dY0002p_376 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L34_1Arg0, metadata !55, metadata !DIExpression()), !dbg !56
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L34_1Arg1, metadata !57, metadata !DIExpression()), !dbg !56
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L34_1Arg2, metadata !58, metadata !DIExpression()), !dbg !56
  call void @llvm.dbg.value(metadata i32 8, metadata !59, metadata !DIExpression()), !dbg !56
  call void @llvm.dbg.value(metadata i32 1, metadata !60, metadata !DIExpression()), !dbg !56
  call void @llvm.dbg.value(metadata i32 0, metadata !61, metadata !DIExpression()), !dbg !56
  call void @llvm.dbg.value(metadata i32 1, metadata !62, metadata !DIExpression()), !dbg !56
  call void @llvm.dbg.value(metadata i32 0, metadata !63, metadata !DIExpression()), !dbg !56
  call void @llvm.dbg.value(metadata i32 1, metadata !64, metadata !DIExpression()), !dbg !56
  call void @llvm.dbg.value(metadata i32 8, metadata !65, metadata !DIExpression()), !dbg !56
  call void @llvm.dbg.value(metadata i32 8, metadata !66, metadata !DIExpression()), !dbg !56
  %0 = load i32, i32* %__nv_MAIN__F1L34_1Arg0, align 4, !dbg !67
  store i32 %0, i32* %__gtid___nv_MAIN__F1L34_1__539, align 4, !dbg !67
  br label %L.LB2_524

L.LB2_524:                                        ; preds = %L.entry
  br label %L.LB2_340

L.LB2_340:                                        ; preds = %L.LB2_524
  store i32 0, i32* %.i0000p_342, align 4, !dbg !68
  call void @llvm.dbg.declare(metadata i32* %i_341, metadata !69, metadata !DIExpression()), !dbg !67
  store i32 1, i32* %i_341, align 4, !dbg !68
  %1 = bitcast i64* %__nv_MAIN__F1L34_1Arg2 to i32**, !dbg !68
  %2 = load i32*, i32** %1, align 8, !dbg !68
  %3 = load i32, i32* %2, align 4, !dbg !68
  store i32 %3, i32* %.du0002p_377, align 4, !dbg !68
  %4 = bitcast i64* %__nv_MAIN__F1L34_1Arg2 to i32**, !dbg !68
  %5 = load i32*, i32** %4, align 8, !dbg !68
  %6 = load i32, i32* %5, align 4, !dbg !68
  store i32 %6, i32* %.de0002p_378, align 4, !dbg !68
  store i32 1, i32* %.di0002p_379, align 4, !dbg !68
  %7 = load i32, i32* %.di0002p_379, align 4, !dbg !68
  store i32 %7, i32* %.ds0002p_380, align 4, !dbg !68
  store i32 1, i32* %.dl0002p_382, align 4, !dbg !68
  %8 = load i32, i32* %.dl0002p_382, align 4, !dbg !68
  store i32 %8, i32* %.dl0002p.copy_533, align 4, !dbg !68
  %9 = load i32, i32* %.de0002p_378, align 4, !dbg !68
  store i32 %9, i32* %.de0002p.copy_534, align 4, !dbg !68
  %10 = load i32, i32* %.ds0002p_380, align 4, !dbg !68
  store i32 %10, i32* %.ds0002p.copy_535, align 4, !dbg !68
  %11 = load i32, i32* %__gtid___nv_MAIN__F1L34_1__539, align 4, !dbg !68
  %12 = bitcast i32* %.i0000p_342 to i64*, !dbg !68
  %13 = bitcast i32* %.dl0002p.copy_533 to i64*, !dbg !68
  %14 = bitcast i32* %.de0002p.copy_534 to i64*, !dbg !68
  %15 = bitcast i32* %.ds0002p.copy_535 to i64*, !dbg !68
  %16 = load i32, i32* %.ds0002p.copy_535, align 4, !dbg !68
  call void @__kmpc_for_static_init_4(i64* null, i32 %11, i32 34, i64* %12, i64* %13, i64* %14, i64* %15, i32 %16, i32 1), !dbg !68
  %17 = load i32, i32* %.dl0002p.copy_533, align 4, !dbg !68
  store i32 %17, i32* %.dl0002p_382, align 4, !dbg !68
  %18 = load i32, i32* %.de0002p.copy_534, align 4, !dbg !68
  store i32 %18, i32* %.de0002p_378, align 4, !dbg !68
  %19 = load i32, i32* %.ds0002p.copy_535, align 4, !dbg !68
  store i32 %19, i32* %.ds0002p_380, align 4, !dbg !68
  %20 = load i32, i32* %.dl0002p_382, align 4, !dbg !68
  store i32 %20, i32* %i_341, align 4, !dbg !68
  %21 = load i32, i32* %i_341, align 4, !dbg !68
  call void @llvm.dbg.value(metadata i32 %21, metadata !69, metadata !DIExpression()), !dbg !67
  store i32 %21, i32* %.dX0002p_381, align 4, !dbg !68
  %22 = load i32, i32* %.dX0002p_381, align 4, !dbg !68
  %23 = load i32, i32* %.du0002p_377, align 4, !dbg !68
  %24 = icmp sgt i32 %22, %23, !dbg !68
  br i1 %24, label %L.LB2_375, label %L.LB2_568, !dbg !68

L.LB2_568:                                        ; preds = %L.LB2_340
  %25 = load i32, i32* %.dX0002p_381, align 4, !dbg !68
  store i32 %25, i32* %i_341, align 4, !dbg !68
  %26 = load i32, i32* %.di0002p_379, align 4, !dbg !68
  %27 = load i32, i32* %.de0002p_378, align 4, !dbg !68
  %28 = load i32, i32* %.dX0002p_381, align 4, !dbg !68
  %29 = sub nsw i32 %27, %28, !dbg !68
  %30 = add nsw i32 %26, %29, !dbg !68
  %31 = load i32, i32* %.di0002p_379, align 4, !dbg !68
  %32 = sdiv i32 %30, %31, !dbg !68
  store i32 %32, i32* %.dY0002p_376, align 4, !dbg !68
  %33 = load i32, i32* %.dY0002p_376, align 4, !dbg !68
  %34 = icmp sle i32 %33, 0, !dbg !68
  br i1 %34, label %L.LB2_385, label %L.LB2_384, !dbg !68

L.LB2_384:                                        ; preds = %L.LB2_384, %L.LB2_568
  %35 = load i32, i32* %i_341, align 4, !dbg !70
  call void @llvm.dbg.value(metadata i32 %35, metadata !69, metadata !DIExpression()), !dbg !67
  %36 = sext i32 %35 to i64, !dbg !70
  %37 = bitcast i64* %__nv_MAIN__F1L34_1Arg2 to i8*, !dbg !70
  %38 = getelementptr i8, i8* %37, i64 192, !dbg !70
  %39 = bitcast i8* %38 to i8**, !dbg !70
  %40 = load i8*, i8** %39, align 8, !dbg !70
  %41 = getelementptr i8, i8* %40, i64 56, !dbg !70
  %42 = bitcast i8* %41 to i64*, !dbg !70
  %43 = load i64, i64* %42, align 8, !dbg !70
  %44 = add nsw i64 %36, %43, !dbg !70
  %45 = bitcast i64* %__nv_MAIN__F1L34_1Arg2 to i8*, !dbg !70
  %46 = getelementptr i8, i8* %45, i64 136, !dbg !70
  %47 = bitcast i8* %46 to i8***, !dbg !70
  %48 = load i8**, i8*** %47, align 8, !dbg !70
  %49 = load i8*, i8** %48, align 8, !dbg !70
  %50 = getelementptr i8, i8* %49, i64 -8, !dbg !70
  %51 = bitcast i8* %50 to double*, !dbg !70
  %52 = getelementptr double, double* %51, i64 %44, !dbg !70
  %53 = load double, double* %52, align 8, !dbg !70
  %54 = load i32, i32* %i_341, align 4, !dbg !70
  call void @llvm.dbg.value(metadata i32 %54, metadata !69, metadata !DIExpression()), !dbg !67
  %55 = sext i32 %54 to i64, !dbg !70
  %56 = bitcast i64* %__nv_MAIN__F1L34_1Arg2 to i8*, !dbg !70
  %57 = getelementptr i8, i8* %56, i64 184, !dbg !70
  %58 = bitcast i8* %57 to i8**, !dbg !70
  %59 = load i8*, i8** %58, align 8, !dbg !70
  %60 = getelementptr i8, i8* %59, i64 56, !dbg !70
  %61 = bitcast i8* %60 to i64*, !dbg !70
  %62 = load i64, i64* %61, align 8, !dbg !70
  %63 = add nsw i64 %55, %62, !dbg !70
  %64 = bitcast i64* %__nv_MAIN__F1L34_1Arg2 to i8*, !dbg !70
  %65 = getelementptr i8, i8* %64, i64 80, !dbg !70
  %66 = bitcast i8* %65 to i8***, !dbg !70
  %67 = load i8**, i8*** %66, align 8, !dbg !70
  %68 = load i8*, i8** %67, align 8, !dbg !70
  %69 = getelementptr i8, i8* %68, i64 -8, !dbg !70
  %70 = bitcast i8* %69 to double*, !dbg !70
  %71 = getelementptr double, double* %70, i64 %63, !dbg !70
  %72 = load double, double* %71, align 8, !dbg !70
  %73 = fmul fast double %53, %72, !dbg !70
  %74 = bitcast i64* %__nv_MAIN__F1L34_1Arg2 to i8*, !dbg !70
  %75 = getelementptr i8, i8* %74, i64 64, !dbg !70
  %76 = bitcast i8* %75 to i32**, !dbg !70
  %77 = load i32*, i32** %76, align 8, !dbg !70
  %78 = load i32, i32* %77, align 4, !dbg !70
  %79 = sext i32 %78 to i64, !dbg !70
  %80 = bitcast i64* %__nv_MAIN__F1L34_1Arg2 to i8*, !dbg !70
  %81 = getelementptr i8, i8* %80, i64 200, !dbg !70
  %82 = bitcast i8* %81 to i8**, !dbg !70
  %83 = load i8*, i8** %82, align 8, !dbg !70
  %84 = getelementptr i8, i8* %83, i64 56, !dbg !70
  %85 = bitcast i8* %84 to i64*, !dbg !70
  %86 = load i64, i64* %85, align 8, !dbg !70
  %87 = add nsw i64 %79, %86, !dbg !70
  %88 = bitcast i64* %__nv_MAIN__F1L34_1Arg2 to i8*, !dbg !70
  %89 = getelementptr i8, i8* %88, i64 16, !dbg !70
  %90 = bitcast i8* %89 to i8***, !dbg !70
  %91 = load i8**, i8*** %90, align 8, !dbg !70
  %92 = load i8*, i8** %91, align 8, !dbg !70
  %93 = getelementptr i8, i8* %92, i64 -8, !dbg !70
  %94 = bitcast i8* %93 to double*, !dbg !70
  %95 = getelementptr double, double* %94, i64 %87, !dbg !70
  %96 = load double, double* %95, align 8, !dbg !70
  %97 = fadd fast double %73, %96, !dbg !70
  %98 = bitcast i64* %__nv_MAIN__F1L34_1Arg2 to i8*, !dbg !70
  %99 = getelementptr i8, i8* %98, i64 64, !dbg !70
  %100 = bitcast i8* %99 to i32**, !dbg !70
  %101 = load i32*, i32** %100, align 8, !dbg !70
  %102 = load i32, i32* %101, align 4, !dbg !70
  %103 = sext i32 %102 to i64, !dbg !70
  %104 = bitcast i64* %__nv_MAIN__F1L34_1Arg2 to i8*, !dbg !70
  %105 = getelementptr i8, i8* %104, i64 200, !dbg !70
  %106 = bitcast i8* %105 to i8**, !dbg !70
  %107 = load i8*, i8** %106, align 8, !dbg !70
  %108 = getelementptr i8, i8* %107, i64 56, !dbg !70
  %109 = bitcast i8* %108 to i64*, !dbg !70
  %110 = load i64, i64* %109, align 8, !dbg !70
  %111 = add nsw i64 %103, %110, !dbg !70
  %112 = bitcast i64* %__nv_MAIN__F1L34_1Arg2 to i8*, !dbg !70
  %113 = getelementptr i8, i8* %112, i64 16, !dbg !70
  %114 = bitcast i8* %113 to i8***, !dbg !70
  %115 = load i8**, i8*** %114, align 8, !dbg !70
  %116 = load i8*, i8** %115, align 8, !dbg !70
  %117 = getelementptr i8, i8* %116, i64 -8, !dbg !70
  %118 = bitcast i8* %117 to double*, !dbg !70
  %119 = getelementptr double, double* %118, i64 %111, !dbg !70
  store double %97, double* %119, align 8, !dbg !70
  %120 = bitcast i64* %__nv_MAIN__F1L34_1Arg2 to i8*, !dbg !71
  %121 = getelementptr i8, i8* %120, i64 64, !dbg !71
  %122 = bitcast i8* %121 to i32**, !dbg !71
  %123 = load i32*, i32** %122, align 8, !dbg !71
  %124 = load i32, i32* %123, align 4, !dbg !71
  %125 = add nsw i32 %124, 1, !dbg !71
  %126 = bitcast i64* %__nv_MAIN__F1L34_1Arg2 to i8*, !dbg !71
  %127 = getelementptr i8, i8* %126, i64 64, !dbg !71
  %128 = bitcast i8* %127 to i32**, !dbg !71
  %129 = load i32*, i32** %128, align 8, !dbg !71
  store i32 %125, i32* %129, align 4, !dbg !71
  %130 = load i32, i32* %.di0002p_379, align 4, !dbg !67
  %131 = load i32, i32* %i_341, align 4, !dbg !67
  call void @llvm.dbg.value(metadata i32 %131, metadata !69, metadata !DIExpression()), !dbg !67
  %132 = add nsw i32 %130, %131, !dbg !67
  store i32 %132, i32* %i_341, align 4, !dbg !67
  %133 = load i32, i32* %.dY0002p_376, align 4, !dbg !67
  %134 = sub nsw i32 %133, 1, !dbg !67
  store i32 %134, i32* %.dY0002p_376, align 4, !dbg !67
  %135 = load i32, i32* %.dY0002p_376, align 4, !dbg !67
  %136 = icmp sgt i32 %135, 0, !dbg !67
  br i1 %136, label %L.LB2_384, label %L.LB2_385, !dbg !67

L.LB2_385:                                        ; preds = %L.LB2_384, %L.LB2_568
  br label %L.LB2_375

L.LB2_375:                                        ; preds = %L.LB2_385, %L.LB2_340
  %137 = load i32, i32* %__gtid___nv_MAIN__F1L34_1__539, align 4, !dbg !67
  call void @__kmpc_for_static_fini(i64* null, i32 %137), !dbg !67
  br label %L.LB2_343

L.LB2_343:                                        ; preds = %L.LB2_375
  ret void, !dbg !67
}

declare void @__kmpc_for_static_fini(i64*, i32) #0

declare void @__kmpc_for_static_init_4(i64*, i32, i32, i64*, i64*, i64*, i64*, i32, i32) #0

declare void @f90_dealloc03a_i8(...) #0

declare signext i32 @f90_allocated_i8(...) #0

declare void @_mp_ecs_nest(...) #0

declare signext i32 @f90io_ldw_end(...) #0

declare signext i32 @f90io_sc_d_ldw(...) #0

declare signext i32 @f90io_sc_ch_ldw(...) #0

declare signext i32 @f90io_print_init(...) #0

declare void @f90io_src_info03a(...) #0

declare void @_mp_bcs_nest(...) #0

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
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB111-linearmissing-orig-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb111_linearmissing_orig_yes", scope: !2, file: !3, line: 12, type: !6, scopeLine: 12, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
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
!18 = !DILocation(line: 46, column: 1, scope: !5)
!19 = !DILocation(line: 12, column: 1, scope: !5)
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
!33 = !DILocation(line: 20, column: 1, scope: !5)
!34 = !DILocalVariable(name: "i", scope: !5, file: !3, type: !9)
!35 = !DILocation(line: 21, column: 1, scope: !5)
!36 = !DILocalVariable(name: "j", scope: !5, file: !3, type: !9)
!37 = !DILocation(line: 22, column: 1, scope: !5)
!38 = !DILocalVariable(scope: !5, file: !3, type: !27, flags: DIFlagArtificial)
!39 = !DILocation(line: 24, column: 1, scope: !5)
!40 = !DILocation(line: 25, column: 1, scope: !5)
!41 = !DILocation(line: 26, column: 1, scope: !5)
!42 = !DILocation(line: 28, column: 1, scope: !5)
!43 = !DILocation(line: 29, column: 1, scope: !5)
!44 = !DILocation(line: 30, column: 1, scope: !5)
!45 = !DILocation(line: 31, column: 1, scope: !5)
!46 = !DILocation(line: 32, column: 1, scope: !5)
!47 = !DILocation(line: 34, column: 1, scope: !5)
!48 = !DILocation(line: 41, column: 1, scope: !5)
!49 = !DILocalVariable(scope: !5, file: !3, type: !9, flags: DIFlagArtificial)
!50 = !DILocation(line: 43, column: 1, scope: !5)
!51 = !DILocation(line: 44, column: 1, scope: !5)
!52 = distinct !DISubprogram(name: "__nv_MAIN__F1L34_1", scope: !2, file: !3, line: 34, type: !53, scopeLine: 34, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!53 = !DISubroutineType(types: !54)
!54 = !{null, !9, !27, !27}
!55 = !DILocalVariable(name: "__nv_MAIN__F1L34_1Arg0", arg: 1, scope: !52, file: !3, type: !9)
!56 = !DILocation(line: 0, scope: !52)
!57 = !DILocalVariable(name: "__nv_MAIN__F1L34_1Arg1", arg: 2, scope: !52, file: !3, type: !27)
!58 = !DILocalVariable(name: "__nv_MAIN__F1L34_1Arg2", arg: 3, scope: !52, file: !3, type: !27)
!59 = !DILocalVariable(name: "omp_nest_lock_kind", scope: !52, file: !3, type: !9)
!60 = !DILocalVariable(name: "omp_sched_static", scope: !52, file: !3, type: !9)
!61 = !DILocalVariable(name: "omp_proc_bind_false", scope: !52, file: !3, type: !9)
!62 = !DILocalVariable(name: "omp_proc_bind_true", scope: !52, file: !3, type: !9)
!63 = !DILocalVariable(name: "omp_lock_hint_none", scope: !52, file: !3, type: !9)
!64 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !52, file: !3, type: !9)
!65 = !DILocalVariable(name: "omp_lock_hint_speculative", scope: !52, file: !3, type: !9)
!66 = !DILocalVariable(name: "dp", scope: !52, file: !3, type: !9)
!67 = !DILocation(line: 38, column: 1, scope: !52)
!68 = !DILocation(line: 35, column: 1, scope: !52)
!69 = !DILocalVariable(name: "i", scope: !52, file: !3, type: !9)
!70 = !DILocation(line: 36, column: 1, scope: !52)
!71 = !DILocation(line: 37, column: 1, scope: !52)

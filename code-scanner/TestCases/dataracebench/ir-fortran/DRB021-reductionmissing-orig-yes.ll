; ModuleID = '/tmp/DRB021-reductionmissing-orig-yes-6be2d0.ll'
source_filename = "/tmp/DRB021-reductionmissing-orig-yes-6be2d0.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%astruct.dt68 = type <{ i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8* }>

@.C305_MAIN_ = internal constant i32 14
@.C339_MAIN_ = internal constant [5 x i8] c"sum ="
@.C336_MAIN_ = internal constant i32 6
@.C333_MAIN_ = internal constant [61 x i8] c"micro-benchmarks-fortran/DRB021-reductionmissing-orig-yes.f95"
@.C335_MAIN_ = internal constant i32 42
@.C290_MAIN_ = internal constant float 5.000000e-01
@.C285_MAIN_ = internal constant i32 1
@.C306_MAIN_ = internal constant i32 27
@.C349_MAIN_ = internal constant i64 4
@.C348_MAIN_ = internal constant i64 27
@.C287_MAIN_ = internal constant float 0.000000e+00
@.C321_MAIN_ = internal constant i32 100
@.C286_MAIN_ = internal constant i64 1
@.C284_MAIN_ = internal constant i64 0
@.C283_MAIN_ = internal constant i32 0
@.C285___nv_MAIN__F1L33_1 = internal constant i32 1
@.C283___nv_MAIN__F1L33_1 = internal constant i32 0

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__446 = alloca i32, align 4
  %.Z0972_323 = alloca float*, align 8
  %"u$sd1_347" = alloca [22 x i64], align 8
  %len_322 = alloca i32, align 4
  %getsum_310 = alloca float, align 4
  %z_b_0_311 = alloca i64, align 8
  %z_b_1_312 = alloca i64, align 8
  %z_e_63_318 = alloca i64, align 8
  %z_b_3_314 = alloca i64, align 8
  %z_b_4_315 = alloca i64, align 8
  %z_e_66_319 = alloca i64, align 8
  %z_b_2_313 = alloca i64, align 8
  %z_b_5_316 = alloca i64, align 8
  %z_b_6_317 = alloca i64, align 8
  %.dY0001_358 = alloca i32, align 4
  %i_307 = alloca i32, align 4
  %.dY0002_361 = alloca i32, align 4
  %j_308 = alloca i32, align 4
  %.uplevelArgPack0001_415 = alloca %astruct.dt68, align 16
  %z__io_338 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !14, metadata !DIExpression()), !dbg !10
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !15
  store i32 %0, i32* %__gtid_MAIN__446, align 4, !dbg !15
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !16
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !16
  call void (i8*, ...) %2(i8* %1), !dbg !16
  call void @llvm.dbg.declare(metadata float** %.Z0972_323, metadata !17, metadata !DIExpression(DW_OP_deref)), !dbg !10
  %3 = bitcast float** %.Z0972_323 to i8**, !dbg !16
  store i8* null, i8** %3, align 8, !dbg !16
  call void @llvm.dbg.declare(metadata [22 x i64]* %"u$sd1_347", metadata !22, metadata !DIExpression()), !dbg !10
  %4 = bitcast [22 x i64]* %"u$sd1_347" to i64*, !dbg !16
  store i64 0, i64* %4, align 8, !dbg !16
  br label %L.LB1_384

L.LB1_384:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %len_322, metadata !27, metadata !DIExpression()), !dbg !10
  store i32 100, i32* %len_322, align 4, !dbg !28
  call void @llvm.dbg.declare(metadata float* %getsum_310, metadata !29, metadata !DIExpression()), !dbg !10
  store float 0.000000e+00, float* %getsum_310, align 4, !dbg !30
  call void @llvm.dbg.declare(metadata i64* %z_b_0_311, metadata !31, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_0_311, align 8, !dbg !32
  %5 = load i32, i32* %len_322, align 4, !dbg !32
  call void @llvm.dbg.value(metadata i32 %5, metadata !27, metadata !DIExpression()), !dbg !10
  %6 = sext i32 %5 to i64, !dbg !32
  call void @llvm.dbg.declare(metadata i64* %z_b_1_312, metadata !31, metadata !DIExpression()), !dbg !10
  store i64 %6, i64* %z_b_1_312, align 8, !dbg !32
  %7 = load i64, i64* %z_b_1_312, align 8, !dbg !32
  call void @llvm.dbg.value(metadata i64 %7, metadata !31, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_63_318, metadata !31, metadata !DIExpression()), !dbg !10
  store i64 %7, i64* %z_e_63_318, align 8, !dbg !32
  call void @llvm.dbg.declare(metadata i64* %z_b_3_314, metadata !31, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_3_314, align 8, !dbg !32
  %8 = load i32, i32* %len_322, align 4, !dbg !32
  call void @llvm.dbg.value(metadata i32 %8, metadata !27, metadata !DIExpression()), !dbg !10
  %9 = sext i32 %8 to i64, !dbg !32
  call void @llvm.dbg.declare(metadata i64* %z_b_4_315, metadata !31, metadata !DIExpression()), !dbg !10
  store i64 %9, i64* %z_b_4_315, align 8, !dbg !32
  %10 = load i64, i64* %z_b_4_315, align 8, !dbg !32
  call void @llvm.dbg.value(metadata i64 %10, metadata !31, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_66_319, metadata !31, metadata !DIExpression()), !dbg !10
  store i64 %10, i64* %z_e_66_319, align 8, !dbg !32
  %11 = bitcast [22 x i64]* %"u$sd1_347" to i8*, !dbg !32
  %12 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !32
  %13 = bitcast i64* @.C348_MAIN_ to i8*, !dbg !32
  %14 = bitcast i64* @.C349_MAIN_ to i8*, !dbg !32
  %15 = bitcast i64* %z_b_0_311 to i8*, !dbg !32
  %16 = bitcast i64* %z_b_1_312 to i8*, !dbg !32
  %17 = bitcast i64* %z_b_3_314 to i8*, !dbg !32
  %18 = bitcast i64* %z_b_4_315 to i8*, !dbg !32
  %19 = bitcast void (...)* @f90_template2_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !32
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, ...) %19(i8* %11, i8* %12, i8* %13, i8* %14, i8* %15, i8* %16, i8* %17, i8* %18), !dbg !32
  %20 = bitcast [22 x i64]* %"u$sd1_347" to i8*, !dbg !32
  %21 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !32
  call void (i8*, i32, ...) %21(i8* %20, i32 27), !dbg !32
  %22 = load i64, i64* %z_b_1_312, align 8, !dbg !32
  call void @llvm.dbg.value(metadata i64 %22, metadata !31, metadata !DIExpression()), !dbg !10
  %23 = load i64, i64* %z_b_0_311, align 8, !dbg !32
  call void @llvm.dbg.value(metadata i64 %23, metadata !31, metadata !DIExpression()), !dbg !10
  %24 = sub nsw i64 %23, 1, !dbg !32
  %25 = sub nsw i64 %22, %24, !dbg !32
  call void @llvm.dbg.declare(metadata i64* %z_b_2_313, metadata !31, metadata !DIExpression()), !dbg !10
  store i64 %25, i64* %z_b_2_313, align 8, !dbg !32
  %26 = load i64, i64* %z_b_1_312, align 8, !dbg !32
  call void @llvm.dbg.value(metadata i64 %26, metadata !31, metadata !DIExpression()), !dbg !10
  %27 = load i64, i64* %z_b_0_311, align 8, !dbg !32
  call void @llvm.dbg.value(metadata i64 %27, metadata !31, metadata !DIExpression()), !dbg !10
  %28 = sub nsw i64 %27, 1, !dbg !32
  %29 = sub nsw i64 %26, %28, !dbg !32
  %30 = load i64, i64* %z_b_4_315, align 8, !dbg !32
  call void @llvm.dbg.value(metadata i64 %30, metadata !31, metadata !DIExpression()), !dbg !10
  %31 = load i64, i64* %z_b_3_314, align 8, !dbg !32
  call void @llvm.dbg.value(metadata i64 %31, metadata !31, metadata !DIExpression()), !dbg !10
  %32 = sub nsw i64 %31, 1, !dbg !32
  %33 = sub nsw i64 %30, %32, !dbg !32
  %34 = mul nsw i64 %29, %33, !dbg !32
  call void @llvm.dbg.declare(metadata i64* %z_b_5_316, metadata !31, metadata !DIExpression()), !dbg !10
  store i64 %34, i64* %z_b_5_316, align 8, !dbg !32
  %35 = load i64, i64* %z_b_0_311, align 8, !dbg !32
  call void @llvm.dbg.value(metadata i64 %35, metadata !31, metadata !DIExpression()), !dbg !10
  %36 = load i64, i64* %z_b_1_312, align 8, !dbg !32
  call void @llvm.dbg.value(metadata i64 %36, metadata !31, metadata !DIExpression()), !dbg !10
  %37 = load i64, i64* %z_b_0_311, align 8, !dbg !32
  call void @llvm.dbg.value(metadata i64 %37, metadata !31, metadata !DIExpression()), !dbg !10
  %38 = sub nsw i64 %37, 1, !dbg !32
  %39 = sub nsw i64 %36, %38, !dbg !32
  %40 = load i64, i64* %z_b_3_314, align 8, !dbg !32
  call void @llvm.dbg.value(metadata i64 %40, metadata !31, metadata !DIExpression()), !dbg !10
  %41 = mul nsw i64 %39, %40, !dbg !32
  %42 = add nsw i64 %35, %41, !dbg !32
  call void @llvm.dbg.declare(metadata i64* %z_b_6_317, metadata !31, metadata !DIExpression()), !dbg !10
  store i64 %42, i64* %z_b_6_317, align 8, !dbg !32
  %43 = bitcast i64* %z_b_5_316 to i8*, !dbg !32
  %44 = bitcast i64* @.C348_MAIN_ to i8*, !dbg !32
  %45 = bitcast i64* @.C349_MAIN_ to i8*, !dbg !32
  %46 = bitcast float** %.Z0972_323 to i8*, !dbg !32
  %47 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !32
  %48 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !32
  %49 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !32
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %49(i8* %43, i8* %44, i8* %45, i8* null, i8* %46, i8* null, i8* %47, i8* %48, i8* null, i64 0), !dbg !32
  %50 = load i32, i32* %len_322, align 4, !dbg !33
  call void @llvm.dbg.value(metadata i32 %50, metadata !27, metadata !DIExpression()), !dbg !10
  store i32 %50, i32* %.dY0001_358, align 4, !dbg !33
  call void @llvm.dbg.declare(metadata i32* %i_307, metadata !34, metadata !DIExpression()), !dbg !10
  store i32 1, i32* %i_307, align 4, !dbg !33
  %51 = load i32, i32* %.dY0001_358, align 4, !dbg !33
  %52 = icmp sle i32 %51, 0, !dbg !33
  br i1 %52, label %L.LB1_357, label %L.LB1_356, !dbg !33

L.LB1_356:                                        ; preds = %L.LB1_360, %L.LB1_384
  %53 = load i32, i32* %len_322, align 4, !dbg !35
  call void @llvm.dbg.value(metadata i32 %53, metadata !27, metadata !DIExpression()), !dbg !10
  store i32 %53, i32* %.dY0002_361, align 4, !dbg !35
  call void @llvm.dbg.declare(metadata i32* %j_308, metadata !36, metadata !DIExpression()), !dbg !10
  store i32 1, i32* %j_308, align 4, !dbg !35
  %54 = load i32, i32* %.dY0002_361, align 4, !dbg !35
  %55 = icmp sle i32 %54, 0, !dbg !35
  br i1 %55, label %L.LB1_360, label %L.LB1_359, !dbg !35

L.LB1_359:                                        ; preds = %L.LB1_359, %L.LB1_356
  %56 = load i32, i32* %i_307, align 4, !dbg !37
  call void @llvm.dbg.value(metadata i32 %56, metadata !34, metadata !DIExpression()), !dbg !10
  %57 = sext i32 %56 to i64, !dbg !37
  %58 = load i32, i32* %j_308, align 4, !dbg !37
  call void @llvm.dbg.value(metadata i32 %58, metadata !36, metadata !DIExpression()), !dbg !10
  %59 = sext i32 %58 to i64, !dbg !37
  %60 = bitcast [22 x i64]* %"u$sd1_347" to i8*, !dbg !37
  %61 = getelementptr i8, i8* %60, i64 160, !dbg !37
  %62 = bitcast i8* %61 to i64*, !dbg !37
  %63 = load i64, i64* %62, align 8, !dbg !37
  %64 = mul nsw i64 %59, %63, !dbg !37
  %65 = add nsw i64 %57, %64, !dbg !37
  %66 = bitcast [22 x i64]* %"u$sd1_347" to i8*, !dbg !37
  %67 = getelementptr i8, i8* %66, i64 56, !dbg !37
  %68 = bitcast i8* %67 to i64*, !dbg !37
  %69 = load i64, i64* %68, align 8, !dbg !37
  %70 = add nsw i64 %65, %69, !dbg !37
  %71 = load float*, float** %.Z0972_323, align 8, !dbg !37
  call void @llvm.dbg.value(metadata float* %71, metadata !17, metadata !DIExpression()), !dbg !10
  %72 = bitcast float* %71 to i8*, !dbg !37
  %73 = getelementptr i8, i8* %72, i64 -4, !dbg !37
  %74 = bitcast i8* %73 to float*, !dbg !37
  %75 = getelementptr float, float* %74, i64 %70, !dbg !37
  store float 5.000000e-01, float* %75, align 4, !dbg !37
  %76 = load i32, i32* %j_308, align 4, !dbg !38
  call void @llvm.dbg.value(metadata i32 %76, metadata !36, metadata !DIExpression()), !dbg !10
  %77 = add nsw i32 %76, 1, !dbg !38
  store i32 %77, i32* %j_308, align 4, !dbg !38
  %78 = load i32, i32* %.dY0002_361, align 4, !dbg !38
  %79 = sub nsw i32 %78, 1, !dbg !38
  store i32 %79, i32* %.dY0002_361, align 4, !dbg !38
  %80 = load i32, i32* %.dY0002_361, align 4, !dbg !38
  %81 = icmp sgt i32 %80, 0, !dbg !38
  br i1 %81, label %L.LB1_359, label %L.LB1_360, !dbg !38

L.LB1_360:                                        ; preds = %L.LB1_359, %L.LB1_356
  %82 = load i32, i32* %i_307, align 4, !dbg !39
  call void @llvm.dbg.value(metadata i32 %82, metadata !34, metadata !DIExpression()), !dbg !10
  %83 = add nsw i32 %82, 1, !dbg !39
  store i32 %83, i32* %i_307, align 4, !dbg !39
  %84 = load i32, i32* %.dY0001_358, align 4, !dbg !39
  %85 = sub nsw i32 %84, 1, !dbg !39
  store i32 %85, i32* %.dY0001_358, align 4, !dbg !39
  %86 = load i32, i32* %.dY0001_358, align 4, !dbg !39
  %87 = icmp sgt i32 %86, 0, !dbg !39
  br i1 %87, label %L.LB1_356, label %L.LB1_357, !dbg !39

L.LB1_357:                                        ; preds = %L.LB1_360, %L.LB1_384
  %88 = bitcast i32* %len_322 to i8*, !dbg !40
  %89 = bitcast %astruct.dt68* %.uplevelArgPack0001_415 to i8**, !dbg !40
  store i8* %88, i8** %89, align 8, !dbg !40
  %90 = bitcast float** %.Z0972_323 to i8*, !dbg !40
  %91 = bitcast %astruct.dt68* %.uplevelArgPack0001_415 to i8*, !dbg !40
  %92 = getelementptr i8, i8* %91, i64 8, !dbg !40
  %93 = bitcast i8* %92 to i8**, !dbg !40
  store i8* %90, i8** %93, align 8, !dbg !40
  %94 = bitcast float** %.Z0972_323 to i8*, !dbg !40
  %95 = bitcast %astruct.dt68* %.uplevelArgPack0001_415 to i8*, !dbg !40
  %96 = getelementptr i8, i8* %95, i64 16, !dbg !40
  %97 = bitcast i8* %96 to i8**, !dbg !40
  store i8* %94, i8** %97, align 8, !dbg !40
  %98 = bitcast i64* %z_b_0_311 to i8*, !dbg !40
  %99 = bitcast %astruct.dt68* %.uplevelArgPack0001_415 to i8*, !dbg !40
  %100 = getelementptr i8, i8* %99, i64 24, !dbg !40
  %101 = bitcast i8* %100 to i8**, !dbg !40
  store i8* %98, i8** %101, align 8, !dbg !40
  %102 = bitcast i64* %z_b_1_312 to i8*, !dbg !40
  %103 = bitcast %astruct.dt68* %.uplevelArgPack0001_415 to i8*, !dbg !40
  %104 = getelementptr i8, i8* %103, i64 32, !dbg !40
  %105 = bitcast i8* %104 to i8**, !dbg !40
  store i8* %102, i8** %105, align 8, !dbg !40
  %106 = bitcast i64* %z_e_63_318 to i8*, !dbg !40
  %107 = bitcast %astruct.dt68* %.uplevelArgPack0001_415 to i8*, !dbg !40
  %108 = getelementptr i8, i8* %107, i64 40, !dbg !40
  %109 = bitcast i8* %108 to i8**, !dbg !40
  store i8* %106, i8** %109, align 8, !dbg !40
  %110 = bitcast i64* %z_b_3_314 to i8*, !dbg !40
  %111 = bitcast %astruct.dt68* %.uplevelArgPack0001_415 to i8*, !dbg !40
  %112 = getelementptr i8, i8* %111, i64 48, !dbg !40
  %113 = bitcast i8* %112 to i8**, !dbg !40
  store i8* %110, i8** %113, align 8, !dbg !40
  %114 = bitcast i64* %z_b_4_315 to i8*, !dbg !40
  %115 = bitcast %astruct.dt68* %.uplevelArgPack0001_415 to i8*, !dbg !40
  %116 = getelementptr i8, i8* %115, i64 56, !dbg !40
  %117 = bitcast i8* %116 to i8**, !dbg !40
  store i8* %114, i8** %117, align 8, !dbg !40
  %118 = bitcast i64* %z_b_2_313 to i8*, !dbg !40
  %119 = bitcast %astruct.dt68* %.uplevelArgPack0001_415 to i8*, !dbg !40
  %120 = getelementptr i8, i8* %119, i64 64, !dbg !40
  %121 = bitcast i8* %120 to i8**, !dbg !40
  store i8* %118, i8** %121, align 8, !dbg !40
  %122 = bitcast i64* %z_e_66_319 to i8*, !dbg !40
  %123 = bitcast %astruct.dt68* %.uplevelArgPack0001_415 to i8*, !dbg !40
  %124 = getelementptr i8, i8* %123, i64 72, !dbg !40
  %125 = bitcast i8* %124 to i8**, !dbg !40
  store i8* %122, i8** %125, align 8, !dbg !40
  %126 = bitcast i64* %z_b_5_316 to i8*, !dbg !40
  %127 = bitcast %astruct.dt68* %.uplevelArgPack0001_415 to i8*, !dbg !40
  %128 = getelementptr i8, i8* %127, i64 80, !dbg !40
  %129 = bitcast i8* %128 to i8**, !dbg !40
  store i8* %126, i8** %129, align 8, !dbg !40
  %130 = bitcast i64* %z_b_6_317 to i8*, !dbg !40
  %131 = bitcast %astruct.dt68* %.uplevelArgPack0001_415 to i8*, !dbg !40
  %132 = getelementptr i8, i8* %131, i64 88, !dbg !40
  %133 = bitcast i8* %132 to i8**, !dbg !40
  store i8* %130, i8** %133, align 8, !dbg !40
  %134 = bitcast float* %getsum_310 to i8*, !dbg !40
  %135 = bitcast %astruct.dt68* %.uplevelArgPack0001_415 to i8*, !dbg !40
  %136 = getelementptr i8, i8* %135, i64 96, !dbg !40
  %137 = bitcast i8* %136 to i8**, !dbg !40
  store i8* %134, i8** %137, align 8, !dbg !40
  %138 = bitcast [22 x i64]* %"u$sd1_347" to i8*, !dbg !40
  %139 = bitcast %astruct.dt68* %.uplevelArgPack0001_415 to i8*, !dbg !40
  %140 = getelementptr i8, i8* %139, i64 104, !dbg !40
  %141 = bitcast i8* %140 to i8**, !dbg !40
  store i8* %138, i8** %141, align 8, !dbg !40
  br label %L.LB1_444, !dbg !40

L.LB1_444:                                        ; preds = %L.LB1_357
  %142 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L33_1_ to i64*, !dbg !40
  %143 = bitcast %astruct.dt68* %.uplevelArgPack0001_415 to i64*, !dbg !40
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %142, i64* %143), !dbg !40
  call void (...) @_mp_bcs_nest(), !dbg !41
  %144 = bitcast i32* @.C335_MAIN_ to i8*, !dbg !41
  %145 = bitcast [61 x i8]* @.C333_MAIN_ to i8*, !dbg !41
  %146 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !41
  call void (i8*, i8*, i64, ...) %146(i8* %144, i8* %145, i64 61), !dbg !41
  %147 = bitcast i32* @.C336_MAIN_ to i8*, !dbg !41
  %148 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !41
  %149 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !41
  %150 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !41
  %151 = call i32 (i8*, i8*, i8*, i8*, ...) %150(i8* %147, i8* null, i8* %148, i8* %149), !dbg !41
  call void @llvm.dbg.declare(metadata i32* %z__io_338, metadata !42, metadata !DIExpression()), !dbg !10
  store i32 %151, i32* %z__io_338, align 4, !dbg !41
  %152 = bitcast [5 x i8]* @.C339_MAIN_ to i8*, !dbg !41
  %153 = bitcast i32 (...)* @f90io_sc_ch_ldw to i32 (i8*, i32, i64, ...)*, !dbg !41
  %154 = call i32 (i8*, i32, i64, ...) %153(i8* %152, i32 14, i64 5), !dbg !41
  store i32 %154, i32* %z__io_338, align 4, !dbg !41
  %155 = load float, float* %getsum_310, align 4, !dbg !41
  call void @llvm.dbg.value(metadata float %155, metadata !29, metadata !DIExpression()), !dbg !10
  %156 = bitcast i32 (...)* @f90io_sc_f_ldw to i32 (float, i32, ...)*, !dbg !41
  %157 = call i32 (float, i32, ...) %156(float %155, i32 27), !dbg !41
  store i32 %157, i32* %z__io_338, align 4, !dbg !41
  %158 = call i32 (...) @f90io_ldw_end(), !dbg !41
  store i32 %158, i32* %z__io_338, align 4, !dbg !41
  call void (...) @_mp_ecs_nest(), !dbg !41
  %159 = load float*, float** %.Z0972_323, align 8, !dbg !43
  call void @llvm.dbg.value(metadata float* %159, metadata !17, metadata !DIExpression()), !dbg !10
  %160 = bitcast float* %159 to i8*, !dbg !43
  %161 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !43
  %162 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !43
  call void (i8*, i8*, i8*, i8*, i64, ...) %162(i8* null, i8* %160, i8* %161, i8* null, i64 0), !dbg !43
  %163 = bitcast float** %.Z0972_323 to i8**, !dbg !43
  store i8* null, i8** %163, align 8, !dbg !43
  %164 = bitcast [22 x i64]* %"u$sd1_347" to i64*, !dbg !43
  store i64 0, i64* %164, align 8, !dbg !43
  ret void, !dbg !15
}

define internal void @__nv_MAIN__F1L33_1_(i32* %__nv_MAIN__F1L33_1Arg0, i64* %__nv_MAIN__F1L33_1Arg1, i64* %__nv_MAIN__F1L33_1Arg2) #0 !dbg !44 {
L.entry:
  %__gtid___nv_MAIN__F1L33_1__491 = alloca i32, align 4
  %.i0000p_330 = alloca i32, align 4
  %i_328 = alloca i32, align 4
  %.du0003p_365 = alloca i32, align 4
  %.de0003p_366 = alloca i32, align 4
  %.di0003p_367 = alloca i32, align 4
  %.ds0003p_368 = alloca i32, align 4
  %.dl0003p_370 = alloca i32, align 4
  %.dl0003p.copy_485 = alloca i32, align 4
  %.de0003p.copy_486 = alloca i32, align 4
  %.ds0003p.copy_487 = alloca i32, align 4
  %.dX0003p_369 = alloca i32, align 4
  %.dY0003p_364 = alloca i32, align 4
  %.dY0004p_376 = alloca i32, align 4
  %j_329 = alloca i32, align 4
  %temp_327 = alloca float, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L33_1Arg0, metadata !47, metadata !DIExpression()), !dbg !48
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L33_1Arg1, metadata !49, metadata !DIExpression()), !dbg !48
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L33_1Arg2, metadata !50, metadata !DIExpression()), !dbg !48
  call void @llvm.dbg.value(metadata i32 1, metadata !51, metadata !DIExpression()), !dbg !48
  call void @llvm.dbg.value(metadata i32 0, metadata !52, metadata !DIExpression()), !dbg !48
  call void @llvm.dbg.value(metadata i32 1, metadata !53, metadata !DIExpression()), !dbg !48
  call void @llvm.dbg.value(metadata i32 0, metadata !54, metadata !DIExpression()), !dbg !48
  call void @llvm.dbg.value(metadata i32 1, metadata !55, metadata !DIExpression()), !dbg !48
  %0 = load i32, i32* %__nv_MAIN__F1L33_1Arg0, align 4, !dbg !56
  store i32 %0, i32* %__gtid___nv_MAIN__F1L33_1__491, align 4, !dbg !56
  br label %L.LB2_476

L.LB2_476:                                        ; preds = %L.entry
  br label %L.LB2_326

L.LB2_326:                                        ; preds = %L.LB2_476
  store i32 0, i32* %.i0000p_330, align 4, !dbg !57
  call void @llvm.dbg.declare(metadata i32* %i_328, metadata !58, metadata !DIExpression()), !dbg !56
  store i32 1, i32* %i_328, align 4, !dbg !57
  %1 = bitcast i64* %__nv_MAIN__F1L33_1Arg2 to i32**, !dbg !57
  %2 = load i32*, i32** %1, align 8, !dbg !57
  %3 = load i32, i32* %2, align 4, !dbg !57
  store i32 %3, i32* %.du0003p_365, align 4, !dbg !57
  %4 = bitcast i64* %__nv_MAIN__F1L33_1Arg2 to i32**, !dbg !57
  %5 = load i32*, i32** %4, align 8, !dbg !57
  %6 = load i32, i32* %5, align 4, !dbg !57
  store i32 %6, i32* %.de0003p_366, align 4, !dbg !57
  store i32 1, i32* %.di0003p_367, align 4, !dbg !57
  %7 = load i32, i32* %.di0003p_367, align 4, !dbg !57
  store i32 %7, i32* %.ds0003p_368, align 4, !dbg !57
  store i32 1, i32* %.dl0003p_370, align 4, !dbg !57
  %8 = load i32, i32* %.dl0003p_370, align 4, !dbg !57
  store i32 %8, i32* %.dl0003p.copy_485, align 4, !dbg !57
  %9 = load i32, i32* %.de0003p_366, align 4, !dbg !57
  store i32 %9, i32* %.de0003p.copy_486, align 4, !dbg !57
  %10 = load i32, i32* %.ds0003p_368, align 4, !dbg !57
  store i32 %10, i32* %.ds0003p.copy_487, align 4, !dbg !57
  %11 = load i32, i32* %__gtid___nv_MAIN__F1L33_1__491, align 4, !dbg !57
  %12 = bitcast i32* %.i0000p_330 to i64*, !dbg !57
  %13 = bitcast i32* %.dl0003p.copy_485 to i64*, !dbg !57
  %14 = bitcast i32* %.de0003p.copy_486 to i64*, !dbg !57
  %15 = bitcast i32* %.ds0003p.copy_487 to i64*, !dbg !57
  %16 = load i32, i32* %.ds0003p.copy_487, align 4, !dbg !57
  call void @__kmpc_for_static_init_4(i64* null, i32 %11, i32 34, i64* %12, i64* %13, i64* %14, i64* %15, i32 %16, i32 1), !dbg !57
  %17 = load i32, i32* %.dl0003p.copy_485, align 4, !dbg !57
  store i32 %17, i32* %.dl0003p_370, align 4, !dbg !57
  %18 = load i32, i32* %.de0003p.copy_486, align 4, !dbg !57
  store i32 %18, i32* %.de0003p_366, align 4, !dbg !57
  %19 = load i32, i32* %.ds0003p.copy_487, align 4, !dbg !57
  store i32 %19, i32* %.ds0003p_368, align 4, !dbg !57
  %20 = load i32, i32* %.dl0003p_370, align 4, !dbg !57
  store i32 %20, i32* %i_328, align 4, !dbg !57
  %21 = load i32, i32* %i_328, align 4, !dbg !57
  call void @llvm.dbg.value(metadata i32 %21, metadata !58, metadata !DIExpression()), !dbg !56
  store i32 %21, i32* %.dX0003p_369, align 4, !dbg !57
  %22 = load i32, i32* %.dX0003p_369, align 4, !dbg !57
  %23 = load i32, i32* %.du0003p_365, align 4, !dbg !57
  %24 = icmp sgt i32 %22, %23, !dbg !57
  br i1 %24, label %L.LB2_363, label %L.LB2_519, !dbg !57

L.LB2_519:                                        ; preds = %L.LB2_326
  %25 = load i32, i32* %.dX0003p_369, align 4, !dbg !57
  store i32 %25, i32* %i_328, align 4, !dbg !57
  %26 = load i32, i32* %.di0003p_367, align 4, !dbg !57
  %27 = load i32, i32* %.de0003p_366, align 4, !dbg !57
  %28 = load i32, i32* %.dX0003p_369, align 4, !dbg !57
  %29 = sub nsw i32 %27, %28, !dbg !57
  %30 = add nsw i32 %26, %29, !dbg !57
  %31 = load i32, i32* %.di0003p_367, align 4, !dbg !57
  %32 = sdiv i32 %30, %31, !dbg !57
  store i32 %32, i32* %.dY0003p_364, align 4, !dbg !57
  %33 = load i32, i32* %.dY0003p_364, align 4, !dbg !57
  %34 = icmp sle i32 %33, 0, !dbg !57
  br i1 %34, label %L.LB2_373, label %L.LB2_372, !dbg !57

L.LB2_372:                                        ; preds = %L.LB2_375, %L.LB2_519
  %35 = bitcast i64* %__nv_MAIN__F1L33_1Arg2 to i32**, !dbg !59
  %36 = load i32*, i32** %35, align 8, !dbg !59
  %37 = load i32, i32* %36, align 4, !dbg !59
  store i32 %37, i32* %.dY0004p_376, align 4, !dbg !59
  call void @llvm.dbg.declare(metadata i32* %j_329, metadata !60, metadata !DIExpression()), !dbg !56
  store i32 1, i32* %j_329, align 4, !dbg !59
  %38 = load i32, i32* %.dY0004p_376, align 4, !dbg !59
  %39 = icmp sle i32 %38, 0, !dbg !59
  br i1 %39, label %L.LB2_375, label %L.LB2_374, !dbg !59

L.LB2_374:                                        ; preds = %L.LB2_374, %L.LB2_372
  %40 = load i32, i32* %i_328, align 4, !dbg !61
  call void @llvm.dbg.value(metadata i32 %40, metadata !58, metadata !DIExpression()), !dbg !56
  %41 = sext i32 %40 to i64, !dbg !61
  %42 = load i32, i32* %j_329, align 4, !dbg !61
  call void @llvm.dbg.value(metadata i32 %42, metadata !60, metadata !DIExpression()), !dbg !56
  %43 = sext i32 %42 to i64, !dbg !61
  %44 = bitcast i64* %__nv_MAIN__F1L33_1Arg2 to i8*, !dbg !61
  %45 = getelementptr i8, i8* %44, i64 104, !dbg !61
  %46 = bitcast i8* %45 to i8**, !dbg !61
  %47 = load i8*, i8** %46, align 8, !dbg !61
  %48 = getelementptr i8, i8* %47, i64 160, !dbg !61
  %49 = bitcast i8* %48 to i64*, !dbg !61
  %50 = load i64, i64* %49, align 8, !dbg !61
  %51 = mul nsw i64 %43, %50, !dbg !61
  %52 = add nsw i64 %41, %51, !dbg !61
  %53 = bitcast i64* %__nv_MAIN__F1L33_1Arg2 to i8*, !dbg !61
  %54 = getelementptr i8, i8* %53, i64 104, !dbg !61
  %55 = bitcast i8* %54 to i8**, !dbg !61
  %56 = load i8*, i8** %55, align 8, !dbg !61
  %57 = getelementptr i8, i8* %56, i64 56, !dbg !61
  %58 = bitcast i8* %57 to i64*, !dbg !61
  %59 = load i64, i64* %58, align 8, !dbg !61
  %60 = add nsw i64 %52, %59, !dbg !61
  %61 = bitcast i64* %__nv_MAIN__F1L33_1Arg2 to i8*, !dbg !61
  %62 = getelementptr i8, i8* %61, i64 16, !dbg !61
  %63 = bitcast i8* %62 to i8***, !dbg !61
  %64 = load i8**, i8*** %63, align 8, !dbg !61
  %65 = load i8*, i8** %64, align 8, !dbg !61
  %66 = getelementptr i8, i8* %65, i64 -4, !dbg !61
  %67 = bitcast i8* %66 to float*, !dbg !61
  %68 = getelementptr float, float* %67, i64 %60, !dbg !61
  %69 = load float, float* %68, align 4, !dbg !61
  call void @llvm.dbg.declare(metadata float* %temp_327, metadata !62, metadata !DIExpression()), !dbg !56
  store float %69, float* %temp_327, align 4, !dbg !61
  %70 = load float, float* %temp_327, align 4, !dbg !63
  call void @llvm.dbg.value(metadata float %70, metadata !62, metadata !DIExpression()), !dbg !56
  %71 = load float, float* %temp_327, align 4, !dbg !63
  call void @llvm.dbg.value(metadata float %71, metadata !62, metadata !DIExpression()), !dbg !56
  %72 = fmul fast float %70, %71, !dbg !63
  %73 = bitcast i64* %__nv_MAIN__F1L33_1Arg2 to i8*, !dbg !63
  %74 = getelementptr i8, i8* %73, i64 96, !dbg !63
  %75 = bitcast i8* %74 to float**, !dbg !63
  %76 = load float*, float** %75, align 8, !dbg !63
  %77 = load float, float* %76, align 4, !dbg !63
  %78 = fadd fast float %72, %77, !dbg !63
  %79 = bitcast i64* %__nv_MAIN__F1L33_1Arg2 to i8*, !dbg !63
  %80 = getelementptr i8, i8* %79, i64 96, !dbg !63
  %81 = bitcast i8* %80 to float**, !dbg !63
  %82 = load float*, float** %81, align 8, !dbg !63
  store float %78, float* %82, align 4, !dbg !63
  %83 = load i32, i32* %j_329, align 4, !dbg !64
  call void @llvm.dbg.value(metadata i32 %83, metadata !60, metadata !DIExpression()), !dbg !56
  %84 = add nsw i32 %83, 1, !dbg !64
  store i32 %84, i32* %j_329, align 4, !dbg !64
  %85 = load i32, i32* %.dY0004p_376, align 4, !dbg !64
  %86 = sub nsw i32 %85, 1, !dbg !64
  store i32 %86, i32* %.dY0004p_376, align 4, !dbg !64
  %87 = load i32, i32* %.dY0004p_376, align 4, !dbg !64
  %88 = icmp sgt i32 %87, 0, !dbg !64
  br i1 %88, label %L.LB2_374, label %L.LB2_375, !dbg !64

L.LB2_375:                                        ; preds = %L.LB2_374, %L.LB2_372
  %89 = load i32, i32* %.di0003p_367, align 4, !dbg !56
  %90 = load i32, i32* %i_328, align 4, !dbg !56
  call void @llvm.dbg.value(metadata i32 %90, metadata !58, metadata !DIExpression()), !dbg !56
  %91 = add nsw i32 %89, %90, !dbg !56
  store i32 %91, i32* %i_328, align 4, !dbg !56
  %92 = load i32, i32* %.dY0003p_364, align 4, !dbg !56
  %93 = sub nsw i32 %92, 1, !dbg !56
  store i32 %93, i32* %.dY0003p_364, align 4, !dbg !56
  %94 = load i32, i32* %.dY0003p_364, align 4, !dbg !56
  %95 = icmp sgt i32 %94, 0, !dbg !56
  br i1 %95, label %L.LB2_372, label %L.LB2_373, !dbg !56

L.LB2_373:                                        ; preds = %L.LB2_375, %L.LB2_519
  br label %L.LB2_363

L.LB2_363:                                        ; preds = %L.LB2_373, %L.LB2_326
  %96 = load i32, i32* %__gtid___nv_MAIN__F1L33_1__491, align 4, !dbg !56
  call void @__kmpc_for_static_fini(i64* null, i32 %96), !dbg !56
  br label %L.LB2_331

L.LB2_331:                                        ; preds = %L.LB2_363
  ret void, !dbg !56
}

declare void @__kmpc_for_static_fini(i64*, i32) #0

declare void @__kmpc_for_static_init_4(i64*, i32, i32, i64*, i64*, i64*, i64*, i32, i32) #0

declare void @f90_dealloc03a_i8(...) #0

declare void @_mp_ecs_nest(...) #0

declare signext i32 @f90io_ldw_end(...) #0

declare signext i32 @f90io_sc_f_ldw(...) #0

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
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB021-reductionmissing-orig-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb021_reductionmissing_orig_yes", scope: !2, file: !3, line: 14, type: !6, scopeLine: 14, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!6 = !DISubroutineType(cc: DW_CC_program, types: !7)
!7 = !{null}
!8 = !DILocalVariable(name: "omp_sched_static", scope: !5, file: !3, type: !9)
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 0, scope: !5)
!11 = !DILocalVariable(name: "omp_proc_bind_false", scope: !5, file: !3, type: !9)
!12 = !DILocalVariable(name: "omp_proc_bind_true", scope: !5, file: !3, type: !9)
!13 = !DILocalVariable(name: "omp_lock_hint_none", scope: !5, file: !3, type: !9)
!14 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !5, file: !3, type: !9)
!15 = !DILocation(line: 44, column: 1, scope: !5)
!16 = !DILocation(line: 14, column: 1, scope: !5)
!17 = !DILocalVariable(name: "u", scope: !5, file: !3, type: !18)
!18 = !DICompositeType(tag: DW_TAG_array_type, baseType: !19, size: 32, align: 32, elements: !20)
!19 = !DIBasicType(name: "real", size: 32, align: 32, encoding: DW_ATE_float)
!20 = !{!21, !21}
!21 = !DISubrange(count: 0, lowerBound: 1)
!22 = !DILocalVariable(scope: !5, file: !3, type: !23, flags: DIFlagArtificial)
!23 = !DICompositeType(tag: DW_TAG_array_type, baseType: !24, size: 1408, align: 64, elements: !25)
!24 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!25 = !{!26}
!26 = !DISubrange(count: 22, lowerBound: 1)
!27 = !DILocalVariable(name: "len", scope: !5, file: !3, type: !9)
!28 = !DILocation(line: 22, column: 1, scope: !5)
!29 = !DILocalVariable(name: "getsum", scope: !5, file: !3, type: !19)
!30 = !DILocation(line: 23, column: 1, scope: !5)
!31 = !DILocalVariable(scope: !5, file: !3, type: !24, flags: DIFlagArtificial)
!32 = !DILocation(line: 25, column: 1, scope: !5)
!33 = !DILocation(line: 27, column: 1, scope: !5)
!34 = !DILocalVariable(name: "i", scope: !5, file: !3, type: !9)
!35 = !DILocation(line: 28, column: 1, scope: !5)
!36 = !DILocalVariable(name: "j", scope: !5, file: !3, type: !9)
!37 = !DILocation(line: 29, column: 1, scope: !5)
!38 = !DILocation(line: 30, column: 1, scope: !5)
!39 = !DILocation(line: 31, column: 1, scope: !5)
!40 = !DILocation(line: 33, column: 1, scope: !5)
!41 = !DILocation(line: 42, column: 1, scope: !5)
!42 = !DILocalVariable(scope: !5, file: !3, type: !9, flags: DIFlagArtificial)
!43 = !DILocation(line: 43, column: 1, scope: !5)
!44 = distinct !DISubprogram(name: "__nv_MAIN__F1L33_1", scope: !2, file: !3, line: 33, type: !45, scopeLine: 33, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!45 = !DISubroutineType(types: !46)
!46 = !{null, !9, !24, !24}
!47 = !DILocalVariable(name: "__nv_MAIN__F1L33_1Arg0", arg: 1, scope: !44, file: !3, type: !9)
!48 = !DILocation(line: 0, scope: !44)
!49 = !DILocalVariable(name: "__nv_MAIN__F1L33_1Arg1", arg: 2, scope: !44, file: !3, type: !24)
!50 = !DILocalVariable(name: "__nv_MAIN__F1L33_1Arg2", arg: 3, scope: !44, file: !3, type: !24)
!51 = !DILocalVariable(name: "omp_sched_static", scope: !44, file: !3, type: !9)
!52 = !DILocalVariable(name: "omp_proc_bind_false", scope: !44, file: !3, type: !9)
!53 = !DILocalVariable(name: "omp_proc_bind_true", scope: !44, file: !3, type: !9)
!54 = !DILocalVariable(name: "omp_lock_hint_none", scope: !44, file: !3, type: !9)
!55 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !44, file: !3, type: !9)
!56 = !DILocation(line: 39, column: 1, scope: !44)
!57 = !DILocation(line: 34, column: 1, scope: !44)
!58 = !DILocalVariable(name: "i", scope: !44, file: !3, type: !9)
!59 = !DILocation(line: 35, column: 1, scope: !44)
!60 = !DILocalVariable(name: "j", scope: !44, file: !3, type: !9)
!61 = !DILocation(line: 36, column: 1, scope: !44)
!62 = !DILocalVariable(name: "temp", scope: !44, file: !3, type: !19)
!63 = !DILocation(line: 37, column: 1, scope: !44)
!64 = !DILocation(line: 38, column: 1, scope: !44)

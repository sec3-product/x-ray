; ModuleID = '/tmp/DRB054-inneronly2-orig-no-b968ce.ll'
source_filename = "/tmp/DRB054-inneronly2-orig-no-b968ce.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%astruct.dt64 = type <{ i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8* }>

@.C300_MAIN_ = internal constant i32 2
@.C285_MAIN_ = internal constant i32 1
@.C309_MAIN_ = internal constant i32 27
@.C336_MAIN_ = internal constant i64 4
@.C335_MAIN_ = internal constant i64 27
@.C324_MAIN_ = internal constant i32 100
@.C286_MAIN_ = internal constant i64 1
@.C284_MAIN_ = internal constant i64 0
@.C283_MAIN_ = internal constant i32 0
@.C285___nv_MAIN__F1L30_1 = internal constant i32 1
@.C300___nv_MAIN__F1L30_1 = internal constant i32 2
@.C283___nv_MAIN__F1L30_1 = internal constant i32 0

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__434 = alloca i32, align 4
  %.Z0972_325 = alloca float*, align 8
  %"b$sd1_334" = alloca [22 x i64], align 8
  %n_312 = alloca i32, align 4
  %m_313 = alloca i32, align 4
  %z_b_0_314 = alloca i64, align 8
  %z_b_1_315 = alloca i64, align 8
  %z_e_63_321 = alloca i64, align 8
  %z_b_3_317 = alloca i64, align 8
  %z_b_4_318 = alloca i64, align 8
  %z_e_66_322 = alloca i64, align 8
  %z_b_2_316 = alloca i64, align 8
  %z_b_5_319 = alloca i64, align 8
  %z_b_6_320 = alloca i64, align 8
  %.dY0001_345 = alloca i32, align 4
  %i_310 = alloca i32, align 4
  %.dY0002_348 = alloca i32, align 4
  %j_311 = alloca i32, align 4
  %.dY0003_351 = alloca i32, align 4
  %.uplevelArgPack0001_403 = alloca %astruct.dt64, align 16
  call void @llvm.dbg.value(metadata i32 1, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 2, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 2, metadata !14, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !15, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !16, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 2, metadata !17, metadata !DIExpression()), !dbg !10
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !18
  store i32 %0, i32* %__gtid_MAIN__434, align 4, !dbg !18
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !19
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !19
  call void (i8*, ...) %2(i8* %1), !dbg !19
  call void @llvm.dbg.declare(metadata float** %.Z0972_325, metadata !20, metadata !DIExpression(DW_OP_deref)), !dbg !10
  %3 = bitcast float** %.Z0972_325 to i8**, !dbg !19
  store i8* null, i8** %3, align 8, !dbg !19
  call void @llvm.dbg.declare(metadata [22 x i64]* %"b$sd1_334", metadata !25, metadata !DIExpression()), !dbg !10
  %4 = bitcast [22 x i64]* %"b$sd1_334" to i64*, !dbg !19
  store i64 0, i64* %4, align 8, !dbg !19
  br label %L.LB1_371

L.LB1_371:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %n_312, metadata !30, metadata !DIExpression()), !dbg !10
  store i32 100, i32* %n_312, align 4, !dbg !31
  call void @llvm.dbg.declare(metadata i32* %m_313, metadata !32, metadata !DIExpression()), !dbg !10
  store i32 100, i32* %m_313, align 4, !dbg !33
  call void @llvm.dbg.declare(metadata i64* %z_b_0_314, metadata !34, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_0_314, align 8, !dbg !35
  %5 = load i32, i32* %n_312, align 4, !dbg !35
  call void @llvm.dbg.value(metadata i32 %5, metadata !30, metadata !DIExpression()), !dbg !10
  %6 = sext i32 %5 to i64, !dbg !35
  call void @llvm.dbg.declare(metadata i64* %z_b_1_315, metadata !34, metadata !DIExpression()), !dbg !10
  store i64 %6, i64* %z_b_1_315, align 8, !dbg !35
  %7 = load i64, i64* %z_b_1_315, align 8, !dbg !35
  call void @llvm.dbg.value(metadata i64 %7, metadata !34, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_63_321, metadata !34, metadata !DIExpression()), !dbg !10
  store i64 %7, i64* %z_e_63_321, align 8, !dbg !35
  call void @llvm.dbg.declare(metadata i64* %z_b_3_317, metadata !34, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_3_317, align 8, !dbg !35
  %8 = load i32, i32* %m_313, align 4, !dbg !35
  call void @llvm.dbg.value(metadata i32 %8, metadata !32, metadata !DIExpression()), !dbg !10
  %9 = sext i32 %8 to i64, !dbg !35
  call void @llvm.dbg.declare(metadata i64* %z_b_4_318, metadata !34, metadata !DIExpression()), !dbg !10
  store i64 %9, i64* %z_b_4_318, align 8, !dbg !35
  %10 = load i64, i64* %z_b_4_318, align 8, !dbg !35
  call void @llvm.dbg.value(metadata i64 %10, metadata !34, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_66_322, metadata !34, metadata !DIExpression()), !dbg !10
  store i64 %10, i64* %z_e_66_322, align 8, !dbg !35
  %11 = bitcast [22 x i64]* %"b$sd1_334" to i8*, !dbg !35
  %12 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !35
  %13 = bitcast i64* @.C335_MAIN_ to i8*, !dbg !35
  %14 = bitcast i64* @.C336_MAIN_ to i8*, !dbg !35
  %15 = bitcast i64* %z_b_0_314 to i8*, !dbg !35
  %16 = bitcast i64* %z_b_1_315 to i8*, !dbg !35
  %17 = bitcast i64* %z_b_3_317 to i8*, !dbg !35
  %18 = bitcast i64* %z_b_4_318 to i8*, !dbg !35
  %19 = bitcast void (...)* @f90_template2_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !35
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, ...) %19(i8* %11, i8* %12, i8* %13, i8* %14, i8* %15, i8* %16, i8* %17, i8* %18), !dbg !35
  %20 = bitcast [22 x i64]* %"b$sd1_334" to i8*, !dbg !35
  %21 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !35
  call void (i8*, i32, ...) %21(i8* %20, i32 27), !dbg !35
  %22 = load i64, i64* %z_b_1_315, align 8, !dbg !35
  call void @llvm.dbg.value(metadata i64 %22, metadata !34, metadata !DIExpression()), !dbg !10
  %23 = load i64, i64* %z_b_0_314, align 8, !dbg !35
  call void @llvm.dbg.value(metadata i64 %23, metadata !34, metadata !DIExpression()), !dbg !10
  %24 = sub nsw i64 %23, 1, !dbg !35
  %25 = sub nsw i64 %22, %24, !dbg !35
  call void @llvm.dbg.declare(metadata i64* %z_b_2_316, metadata !34, metadata !DIExpression()), !dbg !10
  store i64 %25, i64* %z_b_2_316, align 8, !dbg !35
  %26 = load i64, i64* %z_b_1_315, align 8, !dbg !35
  call void @llvm.dbg.value(metadata i64 %26, metadata !34, metadata !DIExpression()), !dbg !10
  %27 = load i64, i64* %z_b_0_314, align 8, !dbg !35
  call void @llvm.dbg.value(metadata i64 %27, metadata !34, metadata !DIExpression()), !dbg !10
  %28 = sub nsw i64 %27, 1, !dbg !35
  %29 = sub nsw i64 %26, %28, !dbg !35
  %30 = load i64, i64* %z_b_4_318, align 8, !dbg !35
  call void @llvm.dbg.value(metadata i64 %30, metadata !34, metadata !DIExpression()), !dbg !10
  %31 = load i64, i64* %z_b_3_317, align 8, !dbg !35
  call void @llvm.dbg.value(metadata i64 %31, metadata !34, metadata !DIExpression()), !dbg !10
  %32 = sub nsw i64 %31, 1, !dbg !35
  %33 = sub nsw i64 %30, %32, !dbg !35
  %34 = mul nsw i64 %29, %33, !dbg !35
  call void @llvm.dbg.declare(metadata i64* %z_b_5_319, metadata !34, metadata !DIExpression()), !dbg !10
  store i64 %34, i64* %z_b_5_319, align 8, !dbg !35
  %35 = load i64, i64* %z_b_0_314, align 8, !dbg !35
  call void @llvm.dbg.value(metadata i64 %35, metadata !34, metadata !DIExpression()), !dbg !10
  %36 = load i64, i64* %z_b_1_315, align 8, !dbg !35
  call void @llvm.dbg.value(metadata i64 %36, metadata !34, metadata !DIExpression()), !dbg !10
  %37 = load i64, i64* %z_b_0_314, align 8, !dbg !35
  call void @llvm.dbg.value(metadata i64 %37, metadata !34, metadata !DIExpression()), !dbg !10
  %38 = sub nsw i64 %37, 1, !dbg !35
  %39 = sub nsw i64 %36, %38, !dbg !35
  %40 = load i64, i64* %z_b_3_317, align 8, !dbg !35
  call void @llvm.dbg.value(metadata i64 %40, metadata !34, metadata !DIExpression()), !dbg !10
  %41 = mul nsw i64 %39, %40, !dbg !35
  %42 = add nsw i64 %35, %41, !dbg !35
  call void @llvm.dbg.declare(metadata i64* %z_b_6_320, metadata !34, metadata !DIExpression()), !dbg !10
  store i64 %42, i64* %z_b_6_320, align 8, !dbg !35
  %43 = bitcast i64* %z_b_5_319 to i8*, !dbg !35
  %44 = bitcast i64* @.C335_MAIN_ to i8*, !dbg !35
  %45 = bitcast i64* @.C336_MAIN_ to i8*, !dbg !35
  %46 = bitcast float** %.Z0972_325 to i8*, !dbg !35
  %47 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !35
  %48 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !35
  %49 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !35
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %49(i8* %43, i8* %44, i8* %45, i8* null, i8* %46, i8* null, i8* %47, i8* %48, i8* null, i64 0), !dbg !35
  %50 = load i32, i32* %n_312, align 4, !dbg !36
  call void @llvm.dbg.value(metadata i32 %50, metadata !30, metadata !DIExpression()), !dbg !10
  store i32 %50, i32* %.dY0001_345, align 4, !dbg !36
  call void @llvm.dbg.declare(metadata i32* %i_310, metadata !37, metadata !DIExpression()), !dbg !10
  store i32 1, i32* %i_310, align 4, !dbg !36
  %51 = load i32, i32* %.dY0001_345, align 4, !dbg !36
  %52 = icmp sle i32 %51, 0, !dbg !36
  br i1 %52, label %L.LB1_344, label %L.LB1_343, !dbg !36

L.LB1_343:                                        ; preds = %L.LB1_347, %L.LB1_371
  %53 = load i32, i32* %m_313, align 4, !dbg !38
  call void @llvm.dbg.value(metadata i32 %53, metadata !32, metadata !DIExpression()), !dbg !10
  store i32 %53, i32* %.dY0002_348, align 4, !dbg !38
  call void @llvm.dbg.declare(metadata i32* %j_311, metadata !39, metadata !DIExpression()), !dbg !10
  store i32 1, i32* %j_311, align 4, !dbg !38
  %54 = load i32, i32* %.dY0002_348, align 4, !dbg !38
  %55 = icmp sle i32 %54, 0, !dbg !38
  br i1 %55, label %L.LB1_347, label %L.LB1_346, !dbg !38

L.LB1_346:                                        ; preds = %L.LB1_346, %L.LB1_343
  %56 = load i32, i32* %j_311, align 4, !dbg !40
  call void @llvm.dbg.value(metadata i32 %56, metadata !39, metadata !DIExpression()), !dbg !10
  %57 = load i32, i32* %i_310, align 4, !dbg !40
  call void @llvm.dbg.value(metadata i32 %57, metadata !37, metadata !DIExpression()), !dbg !10
  %58 = mul nsw i32 %56, %57, !dbg !40
  %59 = sitofp i32 %58 to float, !dbg !40
  %60 = load i32, i32* %i_310, align 4, !dbg !40
  call void @llvm.dbg.value(metadata i32 %60, metadata !37, metadata !DIExpression()), !dbg !10
  %61 = sext i32 %60 to i64, !dbg !40
  %62 = load i32, i32* %j_311, align 4, !dbg !40
  call void @llvm.dbg.value(metadata i32 %62, metadata !39, metadata !DIExpression()), !dbg !10
  %63 = sext i32 %62 to i64, !dbg !40
  %64 = bitcast [22 x i64]* %"b$sd1_334" to i8*, !dbg !40
  %65 = getelementptr i8, i8* %64, i64 160, !dbg !40
  %66 = bitcast i8* %65 to i64*, !dbg !40
  %67 = load i64, i64* %66, align 8, !dbg !40
  %68 = mul nsw i64 %63, %67, !dbg !40
  %69 = add nsw i64 %61, %68, !dbg !40
  %70 = bitcast [22 x i64]* %"b$sd1_334" to i8*, !dbg !40
  %71 = getelementptr i8, i8* %70, i64 56, !dbg !40
  %72 = bitcast i8* %71 to i64*, !dbg !40
  %73 = load i64, i64* %72, align 8, !dbg !40
  %74 = add nsw i64 %69, %73, !dbg !40
  %75 = load float*, float** %.Z0972_325, align 8, !dbg !40
  call void @llvm.dbg.value(metadata float* %75, metadata !20, metadata !DIExpression()), !dbg !10
  %76 = bitcast float* %75 to i8*, !dbg !40
  %77 = getelementptr i8, i8* %76, i64 -4, !dbg !40
  %78 = bitcast i8* %77 to float*, !dbg !40
  %79 = getelementptr float, float* %78, i64 %74, !dbg !40
  store float %59, float* %79, align 4, !dbg !40
  %80 = load i32, i32* %j_311, align 4, !dbg !41
  call void @llvm.dbg.value(metadata i32 %80, metadata !39, metadata !DIExpression()), !dbg !10
  %81 = add nsw i32 %80, 1, !dbg !41
  store i32 %81, i32* %j_311, align 4, !dbg !41
  %82 = load i32, i32* %.dY0002_348, align 4, !dbg !41
  %83 = sub nsw i32 %82, 1, !dbg !41
  store i32 %83, i32* %.dY0002_348, align 4, !dbg !41
  %84 = load i32, i32* %.dY0002_348, align 4, !dbg !41
  %85 = icmp sgt i32 %84, 0, !dbg !41
  br i1 %85, label %L.LB1_346, label %L.LB1_347, !dbg !41

L.LB1_347:                                        ; preds = %L.LB1_346, %L.LB1_343
  %86 = load i32, i32* %i_310, align 4, !dbg !42
  call void @llvm.dbg.value(metadata i32 %86, metadata !37, metadata !DIExpression()), !dbg !10
  %87 = add nsw i32 %86, 1, !dbg !42
  store i32 %87, i32* %i_310, align 4, !dbg !42
  %88 = load i32, i32* %.dY0001_345, align 4, !dbg !42
  %89 = sub nsw i32 %88, 1, !dbg !42
  store i32 %89, i32* %.dY0001_345, align 4, !dbg !42
  %90 = load i32, i32* %.dY0001_345, align 4, !dbg !42
  %91 = icmp sgt i32 %90, 0, !dbg !42
  br i1 %91, label %L.LB1_343, label %L.LB1_344, !dbg !42

L.LB1_344:                                        ; preds = %L.LB1_347, %L.LB1_371
  %92 = load i32, i32* %n_312, align 4, !dbg !43
  call void @llvm.dbg.value(metadata i32 %92, metadata !30, metadata !DIExpression()), !dbg !10
  %93 = sub nsw i32 %92, 1, !dbg !43
  store i32 %93, i32* %.dY0003_351, align 4, !dbg !43
  store i32 2, i32* %i_310, align 4, !dbg !43
  %94 = load i32, i32* %.dY0003_351, align 4, !dbg !43
  %95 = icmp sle i32 %94, 0, !dbg !43
  br i1 %95, label %L.LB1_350, label %L.LB1_349, !dbg !43

L.LB1_349:                                        ; preds = %L.LB1_432, %L.LB1_344
  %96 = bitcast i32* %m_313 to i8*, !dbg !44
  %97 = bitcast %astruct.dt64* %.uplevelArgPack0001_403 to i8**, !dbg !44
  store i8* %96, i8** %97, align 8, !dbg !44
  %98 = bitcast float** %.Z0972_325 to i8*, !dbg !44
  %99 = bitcast %astruct.dt64* %.uplevelArgPack0001_403 to i8*, !dbg !44
  %100 = getelementptr i8, i8* %99, i64 8, !dbg !44
  %101 = bitcast i8* %100 to i8**, !dbg !44
  store i8* %98, i8** %101, align 8, !dbg !44
  %102 = bitcast float** %.Z0972_325 to i8*, !dbg !44
  %103 = bitcast %astruct.dt64* %.uplevelArgPack0001_403 to i8*, !dbg !44
  %104 = getelementptr i8, i8* %103, i64 16, !dbg !44
  %105 = bitcast i8* %104 to i8**, !dbg !44
  store i8* %102, i8** %105, align 8, !dbg !44
  %106 = bitcast i64* %z_b_0_314 to i8*, !dbg !44
  %107 = bitcast %astruct.dt64* %.uplevelArgPack0001_403 to i8*, !dbg !44
  %108 = getelementptr i8, i8* %107, i64 24, !dbg !44
  %109 = bitcast i8* %108 to i8**, !dbg !44
  store i8* %106, i8** %109, align 8, !dbg !44
  %110 = bitcast i64* %z_b_1_315 to i8*, !dbg !44
  %111 = bitcast %astruct.dt64* %.uplevelArgPack0001_403 to i8*, !dbg !44
  %112 = getelementptr i8, i8* %111, i64 32, !dbg !44
  %113 = bitcast i8* %112 to i8**, !dbg !44
  store i8* %110, i8** %113, align 8, !dbg !44
  %114 = bitcast i64* %z_e_63_321 to i8*, !dbg !44
  %115 = bitcast %astruct.dt64* %.uplevelArgPack0001_403 to i8*, !dbg !44
  %116 = getelementptr i8, i8* %115, i64 40, !dbg !44
  %117 = bitcast i8* %116 to i8**, !dbg !44
  store i8* %114, i8** %117, align 8, !dbg !44
  %118 = bitcast i64* %z_b_3_317 to i8*, !dbg !44
  %119 = bitcast %astruct.dt64* %.uplevelArgPack0001_403 to i8*, !dbg !44
  %120 = getelementptr i8, i8* %119, i64 48, !dbg !44
  %121 = bitcast i8* %120 to i8**, !dbg !44
  store i8* %118, i8** %121, align 8, !dbg !44
  %122 = bitcast i64* %z_b_4_318 to i8*, !dbg !44
  %123 = bitcast %astruct.dt64* %.uplevelArgPack0001_403 to i8*, !dbg !44
  %124 = getelementptr i8, i8* %123, i64 56, !dbg !44
  %125 = bitcast i8* %124 to i8**, !dbg !44
  store i8* %122, i8** %125, align 8, !dbg !44
  %126 = bitcast i64* %z_b_2_316 to i8*, !dbg !44
  %127 = bitcast %astruct.dt64* %.uplevelArgPack0001_403 to i8*, !dbg !44
  %128 = getelementptr i8, i8* %127, i64 64, !dbg !44
  %129 = bitcast i8* %128 to i8**, !dbg !44
  store i8* %126, i8** %129, align 8, !dbg !44
  %130 = bitcast i64* %z_e_66_322 to i8*, !dbg !44
  %131 = bitcast %astruct.dt64* %.uplevelArgPack0001_403 to i8*, !dbg !44
  %132 = getelementptr i8, i8* %131, i64 72, !dbg !44
  %133 = bitcast i8* %132 to i8**, !dbg !44
  store i8* %130, i8** %133, align 8, !dbg !44
  %134 = bitcast i64* %z_b_5_319 to i8*, !dbg !44
  %135 = bitcast %astruct.dt64* %.uplevelArgPack0001_403 to i8*, !dbg !44
  %136 = getelementptr i8, i8* %135, i64 80, !dbg !44
  %137 = bitcast i8* %136 to i8**, !dbg !44
  store i8* %134, i8** %137, align 8, !dbg !44
  %138 = bitcast i64* %z_b_6_320 to i8*, !dbg !44
  %139 = bitcast %astruct.dt64* %.uplevelArgPack0001_403 to i8*, !dbg !44
  %140 = getelementptr i8, i8* %139, i64 88, !dbg !44
  %141 = bitcast i8* %140 to i8**, !dbg !44
  store i8* %138, i8** %141, align 8, !dbg !44
  %142 = bitcast i32* %i_310 to i8*, !dbg !44
  %143 = bitcast %astruct.dt64* %.uplevelArgPack0001_403 to i8*, !dbg !44
  %144 = getelementptr i8, i8* %143, i64 96, !dbg !44
  %145 = bitcast i8* %144 to i8**, !dbg !44
  store i8* %142, i8** %145, align 8, !dbg !44
  %146 = bitcast [22 x i64]* %"b$sd1_334" to i8*, !dbg !44
  %147 = bitcast %astruct.dt64* %.uplevelArgPack0001_403 to i8*, !dbg !44
  %148 = getelementptr i8, i8* %147, i64 104, !dbg !44
  %149 = bitcast i8* %148 to i8**, !dbg !44
  store i8* %146, i8** %149, align 8, !dbg !44
  br label %L.LB1_432, !dbg !44

L.LB1_432:                                        ; preds = %L.LB1_349
  %150 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L30_1_ to i64*, !dbg !44
  %151 = bitcast %astruct.dt64* %.uplevelArgPack0001_403 to i64*, !dbg !44
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %150, i64* %151), !dbg !44
  %152 = load i32, i32* %i_310, align 4, !dbg !45
  call void @llvm.dbg.value(metadata i32 %152, metadata !37, metadata !DIExpression()), !dbg !10
  %153 = add nsw i32 %152, 1, !dbg !45
  store i32 %153, i32* %i_310, align 4, !dbg !45
  %154 = load i32, i32* %.dY0003_351, align 4, !dbg !45
  %155 = sub nsw i32 %154, 1, !dbg !45
  store i32 %155, i32* %.dY0003_351, align 4, !dbg !45
  %156 = load i32, i32* %.dY0003_351, align 4, !dbg !45
  %157 = icmp sgt i32 %156, 0, !dbg !45
  br i1 %157, label %L.LB1_349, label %L.LB1_350, !dbg !45

L.LB1_350:                                        ; preds = %L.LB1_432, %L.LB1_344
  %158 = load float*, float** %.Z0972_325, align 8, !dbg !46
  call void @llvm.dbg.value(metadata float* %158, metadata !20, metadata !DIExpression()), !dbg !10
  %159 = bitcast float* %158 to i8*, !dbg !46
  %160 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !46
  %161 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !46
  call void (i8*, i8*, i8*, i8*, i64, ...) %161(i8* null, i8* %159, i8* %160, i8* null, i64 0), !dbg !46
  %162 = bitcast float** %.Z0972_325 to i8**, !dbg !46
  store i8* null, i8** %162, align 8, !dbg !46
  %163 = bitcast [22 x i64]* %"b$sd1_334" to i64*, !dbg !46
  store i64 0, i64* %163, align 8, !dbg !46
  ret void, !dbg !18
}

define internal void @__nv_MAIN__F1L30_1_(i32* %__nv_MAIN__F1L30_1Arg0, i64* %__nv_MAIN__F1L30_1Arg1, i64* %__nv_MAIN__F1L30_1Arg2) #0 !dbg !47 {
L.entry:
  %__gtid___nv_MAIN__F1L30_1__470 = alloca i32, align 4
  %.i0000p_330 = alloca i32, align 4
  %j_329 = alloca i32, align 4
  %.du0004p_355 = alloca i32, align 4
  %.de0004p_356 = alloca i32, align 4
  %.di0004p_357 = alloca i32, align 4
  %.ds0004p_358 = alloca i32, align 4
  %.dl0004p_360 = alloca i32, align 4
  %.dl0004p.copy_464 = alloca i32, align 4
  %.de0004p.copy_465 = alloca i32, align 4
  %.ds0004p.copy_466 = alloca i32, align 4
  %.dX0004p_359 = alloca i32, align 4
  %.dY0004p_354 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L30_1Arg0, metadata !50, metadata !DIExpression()), !dbg !51
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L30_1Arg1, metadata !52, metadata !DIExpression()), !dbg !51
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L30_1Arg2, metadata !53, metadata !DIExpression()), !dbg !51
  call void @llvm.dbg.value(metadata i32 1, metadata !54, metadata !DIExpression()), !dbg !51
  call void @llvm.dbg.value(metadata i32 2, metadata !55, metadata !DIExpression()), !dbg !51
  call void @llvm.dbg.value(metadata i32 0, metadata !56, metadata !DIExpression()), !dbg !51
  call void @llvm.dbg.value(metadata i32 1, metadata !57, metadata !DIExpression()), !dbg !51
  call void @llvm.dbg.value(metadata i32 2, metadata !58, metadata !DIExpression()), !dbg !51
  call void @llvm.dbg.value(metadata i32 0, metadata !59, metadata !DIExpression()), !dbg !51
  call void @llvm.dbg.value(metadata i32 1, metadata !60, metadata !DIExpression()), !dbg !51
  call void @llvm.dbg.value(metadata i32 2, metadata !61, metadata !DIExpression()), !dbg !51
  %0 = load i32, i32* %__nv_MAIN__F1L30_1Arg0, align 4, !dbg !62
  store i32 %0, i32* %__gtid___nv_MAIN__F1L30_1__470, align 4, !dbg !62
  br label %L.LB2_455

L.LB2_455:                                        ; preds = %L.entry
  br label %L.LB2_328

L.LB2_328:                                        ; preds = %L.LB2_455
  store i32 0, i32* %.i0000p_330, align 4, !dbg !63
  call void @llvm.dbg.declare(metadata i32* %j_329, metadata !64, metadata !DIExpression()), !dbg !62
  store i32 2, i32* %j_329, align 4, !dbg !63
  %1 = bitcast i64* %__nv_MAIN__F1L30_1Arg2 to i32**, !dbg !63
  %2 = load i32*, i32** %1, align 8, !dbg !63
  %3 = load i32, i32* %2, align 4, !dbg !63
  store i32 %3, i32* %.du0004p_355, align 4, !dbg !63
  %4 = bitcast i64* %__nv_MAIN__F1L30_1Arg2 to i32**, !dbg !63
  %5 = load i32*, i32** %4, align 8, !dbg !63
  %6 = load i32, i32* %5, align 4, !dbg !63
  store i32 %6, i32* %.de0004p_356, align 4, !dbg !63
  store i32 1, i32* %.di0004p_357, align 4, !dbg !63
  %7 = load i32, i32* %.di0004p_357, align 4, !dbg !63
  store i32 %7, i32* %.ds0004p_358, align 4, !dbg !63
  store i32 2, i32* %.dl0004p_360, align 4, !dbg !63
  %8 = load i32, i32* %.dl0004p_360, align 4, !dbg !63
  store i32 %8, i32* %.dl0004p.copy_464, align 4, !dbg !63
  %9 = load i32, i32* %.de0004p_356, align 4, !dbg !63
  store i32 %9, i32* %.de0004p.copy_465, align 4, !dbg !63
  %10 = load i32, i32* %.ds0004p_358, align 4, !dbg !63
  store i32 %10, i32* %.ds0004p.copy_466, align 4, !dbg !63
  %11 = load i32, i32* %__gtid___nv_MAIN__F1L30_1__470, align 4, !dbg !63
  %12 = bitcast i32* %.i0000p_330 to i64*, !dbg !63
  %13 = bitcast i32* %.dl0004p.copy_464 to i64*, !dbg !63
  %14 = bitcast i32* %.de0004p.copy_465 to i64*, !dbg !63
  %15 = bitcast i32* %.ds0004p.copy_466 to i64*, !dbg !63
  %16 = load i32, i32* %.ds0004p.copy_466, align 4, !dbg !63
  call void @__kmpc_for_static_init_4(i64* null, i32 %11, i32 34, i64* %12, i64* %13, i64* %14, i64* %15, i32 %16, i32 1), !dbg !63
  %17 = load i32, i32* %.dl0004p.copy_464, align 4, !dbg !63
  store i32 %17, i32* %.dl0004p_360, align 4, !dbg !63
  %18 = load i32, i32* %.de0004p.copy_465, align 4, !dbg !63
  store i32 %18, i32* %.de0004p_356, align 4, !dbg !63
  %19 = load i32, i32* %.ds0004p.copy_466, align 4, !dbg !63
  store i32 %19, i32* %.ds0004p_358, align 4, !dbg !63
  %20 = load i32, i32* %.dl0004p_360, align 4, !dbg !63
  store i32 %20, i32* %j_329, align 4, !dbg !63
  %21 = load i32, i32* %j_329, align 4, !dbg !63
  call void @llvm.dbg.value(metadata i32 %21, metadata !64, metadata !DIExpression()), !dbg !62
  store i32 %21, i32* %.dX0004p_359, align 4, !dbg !63
  %22 = load i32, i32* %.dX0004p_359, align 4, !dbg !63
  %23 = load i32, i32* %.du0004p_355, align 4, !dbg !63
  %24 = icmp sgt i32 %22, %23, !dbg !63
  br i1 %24, label %L.LB2_353, label %L.LB2_499, !dbg !63

L.LB2_499:                                        ; preds = %L.LB2_328
  %25 = load i32, i32* %.dX0004p_359, align 4, !dbg !63
  store i32 %25, i32* %j_329, align 4, !dbg !63
  %26 = load i32, i32* %.di0004p_357, align 4, !dbg !63
  %27 = load i32, i32* %.de0004p_356, align 4, !dbg !63
  %28 = load i32, i32* %.dX0004p_359, align 4, !dbg !63
  %29 = sub nsw i32 %27, %28, !dbg !63
  %30 = add nsw i32 %26, %29, !dbg !63
  %31 = load i32, i32* %.di0004p_357, align 4, !dbg !63
  %32 = sdiv i32 %30, %31, !dbg !63
  store i32 %32, i32* %.dY0004p_354, align 4, !dbg !63
  %33 = load i32, i32* %.dY0004p_354, align 4, !dbg !63
  %34 = icmp sle i32 %33, 0, !dbg !63
  br i1 %34, label %L.LB2_363, label %L.LB2_362, !dbg !63

L.LB2_362:                                        ; preds = %L.LB2_362, %L.LB2_499
  %35 = bitcast i64* %__nv_MAIN__F1L30_1Arg2 to i8*, !dbg !65
  %36 = getelementptr i8, i8* %35, i64 96, !dbg !65
  %37 = bitcast i8* %36 to i32**, !dbg !65
  %38 = load i32*, i32** %37, align 8, !dbg !65
  %39 = load i32, i32* %38, align 4, !dbg !65
  %40 = sext i32 %39 to i64, !dbg !65
  %41 = load i32, i32* %j_329, align 4, !dbg !65
  call void @llvm.dbg.value(metadata i32 %41, metadata !64, metadata !DIExpression()), !dbg !62
  %42 = sext i32 %41 to i64, !dbg !65
  %43 = sub nsw i64 %42, 1, !dbg !65
  %44 = bitcast i64* %__nv_MAIN__F1L30_1Arg2 to i8*, !dbg !65
  %45 = getelementptr i8, i8* %44, i64 104, !dbg !65
  %46 = bitcast i8* %45 to i8**, !dbg !65
  %47 = load i8*, i8** %46, align 8, !dbg !65
  %48 = getelementptr i8, i8* %47, i64 160, !dbg !65
  %49 = bitcast i8* %48 to i64*, !dbg !65
  %50 = load i64, i64* %49, align 8, !dbg !65
  %51 = mul nsw i64 %43, %50, !dbg !65
  %52 = add nsw i64 %40, %51, !dbg !65
  %53 = bitcast i64* %__nv_MAIN__F1L30_1Arg2 to i8*, !dbg !65
  %54 = getelementptr i8, i8* %53, i64 104, !dbg !65
  %55 = bitcast i8* %54 to i8**, !dbg !65
  %56 = load i8*, i8** %55, align 8, !dbg !65
  %57 = getelementptr i8, i8* %56, i64 56, !dbg !65
  %58 = bitcast i8* %57 to i64*, !dbg !65
  %59 = load i64, i64* %58, align 8, !dbg !65
  %60 = add nsw i64 %52, %59, !dbg !65
  %61 = bitcast i64* %__nv_MAIN__F1L30_1Arg2 to i8*, !dbg !65
  %62 = getelementptr i8, i8* %61, i64 16, !dbg !65
  %63 = bitcast i8* %62 to i8***, !dbg !65
  %64 = load i8**, i8*** %63, align 8, !dbg !65
  %65 = load i8*, i8** %64, align 8, !dbg !65
  %66 = getelementptr i8, i8* %65, i64 -8, !dbg !65
  %67 = bitcast i8* %66 to float*, !dbg !65
  %68 = getelementptr float, float* %67, i64 %60, !dbg !65
  %69 = load float, float* %68, align 4, !dbg !65
  %70 = bitcast i64* %__nv_MAIN__F1L30_1Arg2 to i8*, !dbg !65
  %71 = getelementptr i8, i8* %70, i64 104, !dbg !65
  %72 = bitcast i8* %71 to i8**, !dbg !65
  %73 = load i8*, i8** %72, align 8, !dbg !65
  %74 = getelementptr i8, i8* %73, i64 56, !dbg !65
  %75 = bitcast i8* %74 to i64*, !dbg !65
  %76 = load i64, i64* %75, align 8, !dbg !65
  %77 = bitcast i64* %__nv_MAIN__F1L30_1Arg2 to i8*, !dbg !65
  %78 = getelementptr i8, i8* %77, i64 96, !dbg !65
  %79 = bitcast i8* %78 to i32**, !dbg !65
  %80 = load i32*, i32** %79, align 8, !dbg !65
  %81 = load i32, i32* %80, align 4, !dbg !65
  %82 = sext i32 %81 to i64, !dbg !65
  %83 = load i32, i32* %j_329, align 4, !dbg !65
  call void @llvm.dbg.value(metadata i32 %83, metadata !64, metadata !DIExpression()), !dbg !62
  %84 = sext i32 %83 to i64, !dbg !65
  %85 = bitcast i64* %__nv_MAIN__F1L30_1Arg2 to i8*, !dbg !65
  %86 = getelementptr i8, i8* %85, i64 104, !dbg !65
  %87 = bitcast i8* %86 to i8**, !dbg !65
  %88 = load i8*, i8** %87, align 8, !dbg !65
  %89 = getelementptr i8, i8* %88, i64 160, !dbg !65
  %90 = bitcast i8* %89 to i64*, !dbg !65
  %91 = load i64, i64* %90, align 8, !dbg !65
  %92 = mul nsw i64 %84, %91, !dbg !65
  %93 = add nsw i64 %82, %92, !dbg !65
  %94 = add nsw i64 %76, %93, !dbg !65
  %95 = bitcast i64* %__nv_MAIN__F1L30_1Arg2 to i8*, !dbg !65
  %96 = getelementptr i8, i8* %95, i64 16, !dbg !65
  %97 = bitcast i8* %96 to i8***, !dbg !65
  %98 = load i8**, i8*** %97, align 8, !dbg !65
  %99 = load i8*, i8** %98, align 8, !dbg !65
  %100 = getelementptr i8, i8* %99, i64 -4, !dbg !65
  %101 = bitcast i8* %100 to float*, !dbg !65
  %102 = getelementptr float, float* %101, i64 %94, !dbg !65
  store float %69, float* %102, align 4, !dbg !65
  %103 = load i32, i32* %.di0004p_357, align 4, !dbg !62
  %104 = load i32, i32* %j_329, align 4, !dbg !62
  call void @llvm.dbg.value(metadata i32 %104, metadata !64, metadata !DIExpression()), !dbg !62
  %105 = add nsw i32 %103, %104, !dbg !62
  store i32 %105, i32* %j_329, align 4, !dbg !62
  %106 = load i32, i32* %.dY0004p_354, align 4, !dbg !62
  %107 = sub nsw i32 %106, 1, !dbg !62
  store i32 %107, i32* %.dY0004p_354, align 4, !dbg !62
  %108 = load i32, i32* %.dY0004p_354, align 4, !dbg !62
  %109 = icmp sgt i32 %108, 0, !dbg !62
  br i1 %109, label %L.LB2_362, label %L.LB2_363, !dbg !62

L.LB2_363:                                        ; preds = %L.LB2_362, %L.LB2_499
  br label %L.LB2_353

L.LB2_353:                                        ; preds = %L.LB2_363, %L.LB2_328
  %110 = load i32, i32* %__gtid___nv_MAIN__F1L30_1__470, align 4, !dbg !62
  call void @__kmpc_for_static_fini(i64* null, i32 %110), !dbg !62
  br label %L.LB2_331

L.LB2_331:                                        ; preds = %L.LB2_353
  ret void, !dbg !62
}

declare void @__kmpc_for_static_fini(i64*, i32) #0

declare void @__kmpc_for_static_init_4(i64*, i32, i32, i64*, i64*, i64*, i64*, i32, i32) #0

declare void @f90_dealloc03a_i8(...) #0

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
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB054-inneronly2-orig-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb054_inneronly2_orig_no", scope: !2, file: !3, line: 11, type: !6, scopeLine: 11, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!6 = !DISubroutineType(cc: DW_CC_program, types: !7)
!7 = !{null}
!8 = !DILocalVariable(name: "omp_sched_static", scope: !5, file: !3, type: !9)
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 0, scope: !5)
!11 = !DILocalVariable(name: "omp_sched_dynamic", scope: !5, file: !3, type: !9)
!12 = !DILocalVariable(name: "omp_proc_bind_false", scope: !5, file: !3, type: !9)
!13 = !DILocalVariable(name: "omp_proc_bind_true", scope: !5, file: !3, type: !9)
!14 = !DILocalVariable(name: "omp_proc_bind_master", scope: !5, file: !3, type: !9)
!15 = !DILocalVariable(name: "omp_lock_hint_none", scope: !5, file: !3, type: !9)
!16 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !5, file: !3, type: !9)
!17 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !5, file: !3, type: !9)
!18 = !DILocation(line: 38, column: 1, scope: !5)
!19 = !DILocation(line: 11, column: 1, scope: !5)
!20 = !DILocalVariable(name: "b", scope: !5, file: !3, type: !21)
!21 = !DICompositeType(tag: DW_TAG_array_type, baseType: !22, size: 32, align: 32, elements: !23)
!22 = !DIBasicType(name: "real", size: 32, align: 32, encoding: DW_ATE_float)
!23 = !{!24, !24}
!24 = !DISubrange(count: 0, lowerBound: 1)
!25 = !DILocalVariable(scope: !5, file: !3, type: !26, flags: DIFlagArtificial)
!26 = !DICompositeType(tag: DW_TAG_array_type, baseType: !27, size: 1408, align: 64, elements: !28)
!27 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!28 = !{!29}
!29 = !DISubrange(count: 22, lowerBound: 1)
!30 = !DILocalVariable(name: "n", scope: !5, file: !3, type: !9)
!31 = !DILocation(line: 18, column: 1, scope: !5)
!32 = !DILocalVariable(name: "m", scope: !5, file: !3, type: !9)
!33 = !DILocation(line: 19, column: 1, scope: !5)
!34 = !DILocalVariable(scope: !5, file: !3, type: !27, flags: DIFlagArtificial)
!35 = !DILocation(line: 21, column: 1, scope: !5)
!36 = !DILocation(line: 23, column: 1, scope: !5)
!37 = !DILocalVariable(name: "i", scope: !5, file: !3, type: !9)
!38 = !DILocation(line: 24, column: 1, scope: !5)
!39 = !DILocalVariable(name: "j", scope: !5, file: !3, type: !9)
!40 = !DILocation(line: 25, column: 1, scope: !5)
!41 = !DILocation(line: 26, column: 1, scope: !5)
!42 = !DILocation(line: 27, column: 1, scope: !5)
!43 = !DILocation(line: 29, column: 1, scope: !5)
!44 = !DILocation(line: 30, column: 1, scope: !5)
!45 = !DILocation(line: 35, column: 1, scope: !5)
!46 = !DILocation(line: 37, column: 1, scope: !5)
!47 = distinct !DISubprogram(name: "__nv_MAIN__F1L30_1", scope: !2, file: !3, line: 30, type: !48, scopeLine: 30, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!48 = !DISubroutineType(types: !49)
!49 = !{null, !9, !27, !27}
!50 = !DILocalVariable(name: "__nv_MAIN__F1L30_1Arg0", arg: 1, scope: !47, file: !3, type: !9)
!51 = !DILocation(line: 0, scope: !47)
!52 = !DILocalVariable(name: "__nv_MAIN__F1L30_1Arg1", arg: 2, scope: !47, file: !3, type: !27)
!53 = !DILocalVariable(name: "__nv_MAIN__F1L30_1Arg2", arg: 3, scope: !47, file: !3, type: !27)
!54 = !DILocalVariable(name: "omp_sched_static", scope: !47, file: !3, type: !9)
!55 = !DILocalVariable(name: "omp_sched_dynamic", scope: !47, file: !3, type: !9)
!56 = !DILocalVariable(name: "omp_proc_bind_false", scope: !47, file: !3, type: !9)
!57 = !DILocalVariable(name: "omp_proc_bind_true", scope: !47, file: !3, type: !9)
!58 = !DILocalVariable(name: "omp_proc_bind_master", scope: !47, file: !3, type: !9)
!59 = !DILocalVariable(name: "omp_lock_hint_none", scope: !47, file: !3, type: !9)
!60 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !47, file: !3, type: !9)
!61 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !47, file: !3, type: !9)
!62 = !DILocation(line: 33, column: 1, scope: !47)
!63 = !DILocation(line: 31, column: 1, scope: !47)
!64 = !DILocalVariable(name: "j", scope: !47, file: !3, type: !9)
!65 = !DILocation(line: 32, column: 1, scope: !47)

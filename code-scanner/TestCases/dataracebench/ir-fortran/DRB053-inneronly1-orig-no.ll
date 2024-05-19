; ModuleID = '/tmp/DRB053-inneronly1-orig-no-42952f.ll'
source_filename = "/tmp/DRB053-inneronly1-orig-no-42952f.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%astruct.dt64 = type <{ i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8* }>

@.C305_MAIN_ = internal constant i32 19
@.C287_MAIN_ = internal constant float 0.000000e+00
@.C307_MAIN_ = internal constant i32 20
@.C285_MAIN_ = internal constant i32 1
@.C306_MAIN_ = internal constant i32 27
@.C332_MAIN_ = internal constant i64 4
@.C331_MAIN_ = internal constant i64 27
@.C320_MAIN_ = internal constant i64 20
@.C286_MAIN_ = internal constant i64 1
@.C284_MAIN_ = internal constant i64 0
@.C283_MAIN_ = internal constant i32 0
@.C307___nv_MAIN__F1L28_1 = internal constant i32 20
@.C285___nv_MAIN__F1L28_1 = internal constant i32 1
@.C283___nv_MAIN__F1L28_1 = internal constant i32 0

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__426 = alloca i32, align 4
  %.Z0970_321 = alloca float*, align 8
  %"a$sd1_330" = alloca [22 x i64], align 8
  %z_b_0_310 = alloca i64, align 8
  %z_b_1_311 = alloca i64, align 8
  %z_e_63_317 = alloca i64, align 8
  %z_b_3_313 = alloca i64, align 8
  %z_b_4_314 = alloca i64, align 8
  %z_e_66_318 = alloca i64, align 8
  %z_b_2_312 = alloca i64, align 8
  %z_b_5_315 = alloca i64, align 8
  %z_b_6_316 = alloca i64, align 8
  %.dY0001_341 = alloca i32, align 4
  %i_308 = alloca i32, align 4
  %.dY0002_344 = alloca i32, align 4
  %j_309 = alloca i32, align 4
  %.dY0003_347 = alloca i32, align 4
  %.uplevelArgPack0001_397 = alloca %astruct.dt64, align 16
  call void @llvm.dbg.value(metadata i32 1, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !14, metadata !DIExpression()), !dbg !10
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !15
  store i32 %0, i32* %__gtid_MAIN__426, align 4, !dbg !15
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !16
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !16
  call void (i8*, ...) %2(i8* %1), !dbg !16
  call void @llvm.dbg.declare(metadata float** %.Z0970_321, metadata !17, metadata !DIExpression(DW_OP_deref)), !dbg !10
  %3 = bitcast float** %.Z0970_321 to i8**, !dbg !16
  store i8* null, i8** %3, align 8, !dbg !16
  call void @llvm.dbg.declare(metadata [22 x i64]* %"a$sd1_330", metadata !22, metadata !DIExpression()), !dbg !10
  %4 = bitcast [22 x i64]* %"a$sd1_330" to i64*, !dbg !16
  store i64 0, i64* %4, align 8, !dbg !16
  br label %L.LB1_367

L.LB1_367:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i64* %z_b_0_310, metadata !27, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_0_310, align 8, !dbg !28
  call void @llvm.dbg.declare(metadata i64* %z_b_1_311, metadata !27, metadata !DIExpression()), !dbg !10
  store i64 20, i64* %z_b_1_311, align 8, !dbg !28
  %5 = load i64, i64* %z_b_1_311, align 8, !dbg !28
  call void @llvm.dbg.value(metadata i64 %5, metadata !27, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_63_317, metadata !27, metadata !DIExpression()), !dbg !10
  store i64 %5, i64* %z_e_63_317, align 8, !dbg !28
  call void @llvm.dbg.declare(metadata i64* %z_b_3_313, metadata !27, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_3_313, align 8, !dbg !28
  call void @llvm.dbg.declare(metadata i64* %z_b_4_314, metadata !27, metadata !DIExpression()), !dbg !10
  store i64 20, i64* %z_b_4_314, align 8, !dbg !28
  %6 = load i64, i64* %z_b_4_314, align 8, !dbg !28
  call void @llvm.dbg.value(metadata i64 %6, metadata !27, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_66_318, metadata !27, metadata !DIExpression()), !dbg !10
  store i64 %6, i64* %z_e_66_318, align 8, !dbg !28
  %7 = bitcast [22 x i64]* %"a$sd1_330" to i8*, !dbg !28
  %8 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !28
  %9 = bitcast i64* @.C331_MAIN_ to i8*, !dbg !28
  %10 = bitcast i64* @.C332_MAIN_ to i8*, !dbg !28
  %11 = bitcast i64* %z_b_0_310 to i8*, !dbg !28
  %12 = bitcast i64* %z_b_1_311 to i8*, !dbg !28
  %13 = bitcast i64* %z_b_3_313 to i8*, !dbg !28
  %14 = bitcast i64* %z_b_4_314 to i8*, !dbg !28
  %15 = bitcast void (...)* @f90_template2_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !28
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, ...) %15(i8* %7, i8* %8, i8* %9, i8* %10, i8* %11, i8* %12, i8* %13, i8* %14), !dbg !28
  %16 = bitcast [22 x i64]* %"a$sd1_330" to i8*, !dbg !28
  %17 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !28
  call void (i8*, i32, ...) %17(i8* %16, i32 27), !dbg !28
  %18 = load i64, i64* %z_b_1_311, align 8, !dbg !28
  call void @llvm.dbg.value(metadata i64 %18, metadata !27, metadata !DIExpression()), !dbg !10
  %19 = load i64, i64* %z_b_0_310, align 8, !dbg !28
  call void @llvm.dbg.value(metadata i64 %19, metadata !27, metadata !DIExpression()), !dbg !10
  %20 = sub nsw i64 %19, 1, !dbg !28
  %21 = sub nsw i64 %18, %20, !dbg !28
  call void @llvm.dbg.declare(metadata i64* %z_b_2_312, metadata !27, metadata !DIExpression()), !dbg !10
  store i64 %21, i64* %z_b_2_312, align 8, !dbg !28
  %22 = load i64, i64* %z_b_1_311, align 8, !dbg !28
  call void @llvm.dbg.value(metadata i64 %22, metadata !27, metadata !DIExpression()), !dbg !10
  %23 = load i64, i64* %z_b_0_310, align 8, !dbg !28
  call void @llvm.dbg.value(metadata i64 %23, metadata !27, metadata !DIExpression()), !dbg !10
  %24 = sub nsw i64 %23, 1, !dbg !28
  %25 = sub nsw i64 %22, %24, !dbg !28
  %26 = load i64, i64* %z_b_4_314, align 8, !dbg !28
  call void @llvm.dbg.value(metadata i64 %26, metadata !27, metadata !DIExpression()), !dbg !10
  %27 = load i64, i64* %z_b_3_313, align 8, !dbg !28
  call void @llvm.dbg.value(metadata i64 %27, metadata !27, metadata !DIExpression()), !dbg !10
  %28 = sub nsw i64 %27, 1, !dbg !28
  %29 = sub nsw i64 %26, %28, !dbg !28
  %30 = mul nsw i64 %25, %29, !dbg !28
  call void @llvm.dbg.declare(metadata i64* %z_b_5_315, metadata !27, metadata !DIExpression()), !dbg !10
  store i64 %30, i64* %z_b_5_315, align 8, !dbg !28
  %31 = load i64, i64* %z_b_0_310, align 8, !dbg !28
  call void @llvm.dbg.value(metadata i64 %31, metadata !27, metadata !DIExpression()), !dbg !10
  %32 = load i64, i64* %z_b_1_311, align 8, !dbg !28
  call void @llvm.dbg.value(metadata i64 %32, metadata !27, metadata !DIExpression()), !dbg !10
  %33 = load i64, i64* %z_b_0_310, align 8, !dbg !28
  call void @llvm.dbg.value(metadata i64 %33, metadata !27, metadata !DIExpression()), !dbg !10
  %34 = sub nsw i64 %33, 1, !dbg !28
  %35 = sub nsw i64 %32, %34, !dbg !28
  %36 = load i64, i64* %z_b_3_313, align 8, !dbg !28
  call void @llvm.dbg.value(metadata i64 %36, metadata !27, metadata !DIExpression()), !dbg !10
  %37 = mul nsw i64 %35, %36, !dbg !28
  %38 = add nsw i64 %31, %37, !dbg !28
  call void @llvm.dbg.declare(metadata i64* %z_b_6_316, metadata !27, metadata !DIExpression()), !dbg !10
  store i64 %38, i64* %z_b_6_316, align 8, !dbg !28
  %39 = bitcast i64* %z_b_5_315 to i8*, !dbg !28
  %40 = bitcast i64* @.C331_MAIN_ to i8*, !dbg !28
  %41 = bitcast i64* @.C332_MAIN_ to i8*, !dbg !28
  %42 = bitcast float** %.Z0970_321 to i8*, !dbg !28
  %43 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !28
  %44 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !28
  %45 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !28
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %45(i8* %39, i8* %40, i8* %41, i8* null, i8* %42, i8* null, i8* %43, i8* %44, i8* null, i64 0), !dbg !28
  store i32 20, i32* %.dY0001_341, align 4, !dbg !29
  call void @llvm.dbg.declare(metadata i32* %i_308, metadata !30, metadata !DIExpression()), !dbg !10
  store i32 1, i32* %i_308, align 4, !dbg !29
  br label %L.LB1_339

L.LB1_339:                                        ; preds = %L.LB1_444, %L.LB1_367
  store i32 20, i32* %.dY0002_344, align 4, !dbg !31
  call void @llvm.dbg.declare(metadata i32* %j_309, metadata !32, metadata !DIExpression()), !dbg !10
  store i32 1, i32* %j_309, align 4, !dbg !31
  br label %L.LB1_342

L.LB1_342:                                        ; preds = %L.LB1_342, %L.LB1_339
  %46 = load i32, i32* %i_308, align 4, !dbg !33
  call void @llvm.dbg.value(metadata i32 %46, metadata !30, metadata !DIExpression()), !dbg !10
  %47 = sext i32 %46 to i64, !dbg !33
  %48 = load i32, i32* %j_309, align 4, !dbg !33
  call void @llvm.dbg.value(metadata i32 %48, metadata !32, metadata !DIExpression()), !dbg !10
  %49 = sext i32 %48 to i64, !dbg !33
  %50 = bitcast [22 x i64]* %"a$sd1_330" to i8*, !dbg !33
  %51 = getelementptr i8, i8* %50, i64 160, !dbg !33
  %52 = bitcast i8* %51 to i64*, !dbg !33
  %53 = load i64, i64* %52, align 8, !dbg !33
  %54 = mul nsw i64 %49, %53, !dbg !33
  %55 = add nsw i64 %47, %54, !dbg !33
  %56 = bitcast [22 x i64]* %"a$sd1_330" to i8*, !dbg !33
  %57 = getelementptr i8, i8* %56, i64 56, !dbg !33
  %58 = bitcast i8* %57 to i64*, !dbg !33
  %59 = load i64, i64* %58, align 8, !dbg !33
  %60 = add nsw i64 %55, %59, !dbg !33
  %61 = load float*, float** %.Z0970_321, align 8, !dbg !33
  call void @llvm.dbg.value(metadata float* %61, metadata !17, metadata !DIExpression()), !dbg !10
  %62 = bitcast float* %61 to i8*, !dbg !33
  %63 = getelementptr i8, i8* %62, i64 -4, !dbg !33
  %64 = bitcast i8* %63 to float*, !dbg !33
  %65 = getelementptr float, float* %64, i64 %60, !dbg !33
  store float 0.000000e+00, float* %65, align 4, !dbg !33
  %66 = load i32, i32* %j_309, align 4, !dbg !34
  call void @llvm.dbg.value(metadata i32 %66, metadata !32, metadata !DIExpression()), !dbg !10
  %67 = add nsw i32 %66, 1, !dbg !34
  store i32 %67, i32* %j_309, align 4, !dbg !34
  %68 = load i32, i32* %.dY0002_344, align 4, !dbg !34
  %69 = sub nsw i32 %68, 1, !dbg !34
  store i32 %69, i32* %.dY0002_344, align 4, !dbg !34
  %70 = load i32, i32* %.dY0002_344, align 4, !dbg !34
  %71 = icmp sgt i32 %70, 0, !dbg !34
  br i1 %71, label %L.LB1_342, label %L.LB1_444, !dbg !34

L.LB1_444:                                        ; preds = %L.LB1_342
  %72 = load i32, i32* %i_308, align 4, !dbg !35
  call void @llvm.dbg.value(metadata i32 %72, metadata !30, metadata !DIExpression()), !dbg !10
  %73 = add nsw i32 %72, 1, !dbg !35
  store i32 %73, i32* %i_308, align 4, !dbg !35
  %74 = load i32, i32* %.dY0001_341, align 4, !dbg !35
  %75 = sub nsw i32 %74, 1, !dbg !35
  store i32 %75, i32* %.dY0001_341, align 4, !dbg !35
  %76 = load i32, i32* %.dY0001_341, align 4, !dbg !35
  %77 = icmp sgt i32 %76, 0, !dbg !35
  br i1 %77, label %L.LB1_339, label %L.LB1_445, !dbg !35

L.LB1_445:                                        ; preds = %L.LB1_444
  store i32 19, i32* %.dY0003_347, align 4, !dbg !36
  store i32 1, i32* %i_308, align 4, !dbg !36
  br label %L.LB1_345

L.LB1_345:                                        ; preds = %L.LB1_424, %L.LB1_445
  %78 = bitcast float** %.Z0970_321 to i8*, !dbg !37
  %79 = bitcast %astruct.dt64* %.uplevelArgPack0001_397 to i8**, !dbg !37
  store i8* %78, i8** %79, align 8, !dbg !37
  %80 = bitcast float** %.Z0970_321 to i8*, !dbg !37
  %81 = bitcast %astruct.dt64* %.uplevelArgPack0001_397 to i8*, !dbg !37
  %82 = getelementptr i8, i8* %81, i64 8, !dbg !37
  %83 = bitcast i8* %82 to i8**, !dbg !37
  store i8* %80, i8** %83, align 8, !dbg !37
  %84 = bitcast i64* %z_b_0_310 to i8*, !dbg !37
  %85 = bitcast %astruct.dt64* %.uplevelArgPack0001_397 to i8*, !dbg !37
  %86 = getelementptr i8, i8* %85, i64 16, !dbg !37
  %87 = bitcast i8* %86 to i8**, !dbg !37
  store i8* %84, i8** %87, align 8, !dbg !37
  %88 = bitcast i64* %z_b_1_311 to i8*, !dbg !37
  %89 = bitcast %astruct.dt64* %.uplevelArgPack0001_397 to i8*, !dbg !37
  %90 = getelementptr i8, i8* %89, i64 24, !dbg !37
  %91 = bitcast i8* %90 to i8**, !dbg !37
  store i8* %88, i8** %91, align 8, !dbg !37
  %92 = bitcast i64* %z_e_63_317 to i8*, !dbg !37
  %93 = bitcast %astruct.dt64* %.uplevelArgPack0001_397 to i8*, !dbg !37
  %94 = getelementptr i8, i8* %93, i64 32, !dbg !37
  %95 = bitcast i8* %94 to i8**, !dbg !37
  store i8* %92, i8** %95, align 8, !dbg !37
  %96 = bitcast i64* %z_b_3_313 to i8*, !dbg !37
  %97 = bitcast %astruct.dt64* %.uplevelArgPack0001_397 to i8*, !dbg !37
  %98 = getelementptr i8, i8* %97, i64 40, !dbg !37
  %99 = bitcast i8* %98 to i8**, !dbg !37
  store i8* %96, i8** %99, align 8, !dbg !37
  %100 = bitcast i64* %z_b_4_314 to i8*, !dbg !37
  %101 = bitcast %astruct.dt64* %.uplevelArgPack0001_397 to i8*, !dbg !37
  %102 = getelementptr i8, i8* %101, i64 48, !dbg !37
  %103 = bitcast i8* %102 to i8**, !dbg !37
  store i8* %100, i8** %103, align 8, !dbg !37
  %104 = bitcast i64* %z_b_2_312 to i8*, !dbg !37
  %105 = bitcast %astruct.dt64* %.uplevelArgPack0001_397 to i8*, !dbg !37
  %106 = getelementptr i8, i8* %105, i64 56, !dbg !37
  %107 = bitcast i8* %106 to i8**, !dbg !37
  store i8* %104, i8** %107, align 8, !dbg !37
  %108 = bitcast i64* %z_e_66_318 to i8*, !dbg !37
  %109 = bitcast %astruct.dt64* %.uplevelArgPack0001_397 to i8*, !dbg !37
  %110 = getelementptr i8, i8* %109, i64 64, !dbg !37
  %111 = bitcast i8* %110 to i8**, !dbg !37
  store i8* %108, i8** %111, align 8, !dbg !37
  %112 = bitcast i64* %z_b_5_315 to i8*, !dbg !37
  %113 = bitcast %astruct.dt64* %.uplevelArgPack0001_397 to i8*, !dbg !37
  %114 = getelementptr i8, i8* %113, i64 72, !dbg !37
  %115 = bitcast i8* %114 to i8**, !dbg !37
  store i8* %112, i8** %115, align 8, !dbg !37
  %116 = bitcast i64* %z_b_6_316 to i8*, !dbg !37
  %117 = bitcast %astruct.dt64* %.uplevelArgPack0001_397 to i8*, !dbg !37
  %118 = getelementptr i8, i8* %117, i64 80, !dbg !37
  %119 = bitcast i8* %118 to i8**, !dbg !37
  store i8* %116, i8** %119, align 8, !dbg !37
  %120 = bitcast i32* %i_308 to i8*, !dbg !37
  %121 = bitcast %astruct.dt64* %.uplevelArgPack0001_397 to i8*, !dbg !37
  %122 = getelementptr i8, i8* %121, i64 88, !dbg !37
  %123 = bitcast i8* %122 to i8**, !dbg !37
  store i8* %120, i8** %123, align 8, !dbg !37
  %124 = bitcast [22 x i64]* %"a$sd1_330" to i8*, !dbg !37
  %125 = bitcast %astruct.dt64* %.uplevelArgPack0001_397 to i8*, !dbg !37
  %126 = getelementptr i8, i8* %125, i64 96, !dbg !37
  %127 = bitcast i8* %126 to i8**, !dbg !37
  store i8* %124, i8** %127, align 8, !dbg !37
  br label %L.LB1_424, !dbg !37

L.LB1_424:                                        ; preds = %L.LB1_345
  %128 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L28_1_ to i64*, !dbg !37
  %129 = bitcast %astruct.dt64* %.uplevelArgPack0001_397 to i64*, !dbg !37
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %128, i64* %129), !dbg !37
  %130 = load i32, i32* %i_308, align 4, !dbg !38
  call void @llvm.dbg.value(metadata i32 %130, metadata !30, metadata !DIExpression()), !dbg !10
  %131 = add nsw i32 %130, 1, !dbg !38
  store i32 %131, i32* %i_308, align 4, !dbg !38
  %132 = load i32, i32* %.dY0003_347, align 4, !dbg !38
  %133 = sub nsw i32 %132, 1, !dbg !38
  store i32 %133, i32* %.dY0003_347, align 4, !dbg !38
  %134 = load i32, i32* %.dY0003_347, align 4, !dbg !38
  %135 = icmp sgt i32 %134, 0, !dbg !38
  br i1 %135, label %L.LB1_345, label %L.LB1_446, !dbg !38

L.LB1_446:                                        ; preds = %L.LB1_424
  %136 = load float*, float** %.Z0970_321, align 8, !dbg !39
  call void @llvm.dbg.value(metadata float* %136, metadata !17, metadata !DIExpression()), !dbg !10
  %137 = bitcast float* %136 to i8*, !dbg !39
  %138 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !39
  %139 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !39
  call void (i8*, i8*, i8*, i8*, i64, ...) %139(i8* null, i8* %137, i8* %138, i8* null, i64 0), !dbg !39
  %140 = bitcast float** %.Z0970_321 to i8**, !dbg !39
  store i8* null, i8** %140, align 8, !dbg !39
  %141 = bitcast [22 x i64]* %"a$sd1_330" to i64*, !dbg !39
  store i64 0, i64* %141, align 8, !dbg !39
  ret void, !dbg !15
}

define internal void @__nv_MAIN__F1L28_1_(i32* %__nv_MAIN__F1L28_1Arg0, i64* %__nv_MAIN__F1L28_1Arg1, i64* %__nv_MAIN__F1L28_1Arg2) #0 !dbg !40 {
L.entry:
  %__gtid___nv_MAIN__F1L28_1__464 = alloca i32, align 4
  %.i0000p_326 = alloca i32, align 4
  %j_325 = alloca i32, align 4
  %.du0004p_351 = alloca i32, align 4
  %.de0004p_352 = alloca i32, align 4
  %.di0004p_353 = alloca i32, align 4
  %.ds0004p_354 = alloca i32, align 4
  %.dl0004p_356 = alloca i32, align 4
  %.dl0004p.copy_458 = alloca i32, align 4
  %.de0004p.copy_459 = alloca i32, align 4
  %.ds0004p.copy_460 = alloca i32, align 4
  %.dX0004p_355 = alloca i32, align 4
  %.dY0004p_350 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L28_1Arg0, metadata !43, metadata !DIExpression()), !dbg !44
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L28_1Arg1, metadata !45, metadata !DIExpression()), !dbg !44
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L28_1Arg2, metadata !46, metadata !DIExpression()), !dbg !44
  call void @llvm.dbg.value(metadata i32 1, metadata !47, metadata !DIExpression()), !dbg !44
  call void @llvm.dbg.value(metadata i32 0, metadata !48, metadata !DIExpression()), !dbg !44
  call void @llvm.dbg.value(metadata i32 1, metadata !49, metadata !DIExpression()), !dbg !44
  call void @llvm.dbg.value(metadata i32 0, metadata !50, metadata !DIExpression()), !dbg !44
  call void @llvm.dbg.value(metadata i32 1, metadata !51, metadata !DIExpression()), !dbg !44
  %0 = load i32, i32* %__nv_MAIN__F1L28_1Arg0, align 4, !dbg !52
  store i32 %0, i32* %__gtid___nv_MAIN__F1L28_1__464, align 4, !dbg !52
  br label %L.LB2_450

L.LB2_450:                                        ; preds = %L.entry
  br label %L.LB2_324

L.LB2_324:                                        ; preds = %L.LB2_450
  store i32 0, i32* %.i0000p_326, align 4, !dbg !53
  call void @llvm.dbg.declare(metadata i32* %j_325, metadata !54, metadata !DIExpression()), !dbg !52
  store i32 1, i32* %j_325, align 4, !dbg !53
  store i32 20, i32* %.du0004p_351, align 4, !dbg !53
  store i32 20, i32* %.de0004p_352, align 4, !dbg !53
  store i32 1, i32* %.di0004p_353, align 4, !dbg !53
  %1 = load i32, i32* %.di0004p_353, align 4, !dbg !53
  store i32 %1, i32* %.ds0004p_354, align 4, !dbg !53
  store i32 1, i32* %.dl0004p_356, align 4, !dbg !53
  %2 = load i32, i32* %.dl0004p_356, align 4, !dbg !53
  store i32 %2, i32* %.dl0004p.copy_458, align 4, !dbg !53
  %3 = load i32, i32* %.de0004p_352, align 4, !dbg !53
  store i32 %3, i32* %.de0004p.copy_459, align 4, !dbg !53
  %4 = load i32, i32* %.ds0004p_354, align 4, !dbg !53
  store i32 %4, i32* %.ds0004p.copy_460, align 4, !dbg !53
  %5 = load i32, i32* %__gtid___nv_MAIN__F1L28_1__464, align 4, !dbg !53
  %6 = bitcast i32* %.i0000p_326 to i64*, !dbg !53
  %7 = bitcast i32* %.dl0004p.copy_458 to i64*, !dbg !53
  %8 = bitcast i32* %.de0004p.copy_459 to i64*, !dbg !53
  %9 = bitcast i32* %.ds0004p.copy_460 to i64*, !dbg !53
  %10 = load i32, i32* %.ds0004p.copy_460, align 4, !dbg !53
  call void @__kmpc_for_static_init_4(i64* null, i32 %5, i32 34, i64* %6, i64* %7, i64* %8, i64* %9, i32 %10, i32 1), !dbg !53
  %11 = load i32, i32* %.dl0004p.copy_458, align 4, !dbg !53
  store i32 %11, i32* %.dl0004p_356, align 4, !dbg !53
  %12 = load i32, i32* %.de0004p.copy_459, align 4, !dbg !53
  store i32 %12, i32* %.de0004p_352, align 4, !dbg !53
  %13 = load i32, i32* %.ds0004p.copy_460, align 4, !dbg !53
  store i32 %13, i32* %.ds0004p_354, align 4, !dbg !53
  %14 = load i32, i32* %.dl0004p_356, align 4, !dbg !53
  store i32 %14, i32* %j_325, align 4, !dbg !53
  %15 = load i32, i32* %j_325, align 4, !dbg !53
  call void @llvm.dbg.value(metadata i32 %15, metadata !54, metadata !DIExpression()), !dbg !52
  store i32 %15, i32* %.dX0004p_355, align 4, !dbg !53
  %16 = load i32, i32* %.dX0004p_355, align 4, !dbg !53
  %17 = load i32, i32* %.du0004p_351, align 4, !dbg !53
  %18 = icmp sgt i32 %16, %17, !dbg !53
  br i1 %18, label %L.LB2_349, label %L.LB2_490, !dbg !53

L.LB2_490:                                        ; preds = %L.LB2_324
  %19 = load i32, i32* %.dX0004p_355, align 4, !dbg !53
  store i32 %19, i32* %j_325, align 4, !dbg !53
  %20 = load i32, i32* %.di0004p_353, align 4, !dbg !53
  %21 = load i32, i32* %.de0004p_352, align 4, !dbg !53
  %22 = load i32, i32* %.dX0004p_355, align 4, !dbg !53
  %23 = sub nsw i32 %21, %22, !dbg !53
  %24 = add nsw i32 %20, %23, !dbg !53
  %25 = load i32, i32* %.di0004p_353, align 4, !dbg !53
  %26 = sdiv i32 %24, %25, !dbg !53
  store i32 %26, i32* %.dY0004p_350, align 4, !dbg !53
  %27 = load i32, i32* %.dY0004p_350, align 4, !dbg !53
  %28 = icmp sle i32 %27, 0, !dbg !53
  br i1 %28, label %L.LB2_359, label %L.LB2_358, !dbg !53

L.LB2_358:                                        ; preds = %L.LB2_358, %L.LB2_490
  %29 = bitcast i64* %__nv_MAIN__F1L28_1Arg2 to i8*, !dbg !55
  %30 = getelementptr i8, i8* %29, i64 88, !dbg !55
  %31 = bitcast i8* %30 to i32**, !dbg !55
  %32 = load i32*, i32** %31, align 8, !dbg !55
  %33 = load i32, i32* %32, align 4, !dbg !55
  %34 = sext i32 %33 to i64, !dbg !55
  %35 = load i32, i32* %j_325, align 4, !dbg !55
  call void @llvm.dbg.value(metadata i32 %35, metadata !54, metadata !DIExpression()), !dbg !52
  %36 = sext i32 %35 to i64, !dbg !55
  %37 = bitcast i64* %__nv_MAIN__F1L28_1Arg2 to i8*, !dbg !55
  %38 = getelementptr i8, i8* %37, i64 96, !dbg !55
  %39 = bitcast i8* %38 to i8**, !dbg !55
  %40 = load i8*, i8** %39, align 8, !dbg !55
  %41 = getelementptr i8, i8* %40, i64 160, !dbg !55
  %42 = bitcast i8* %41 to i64*, !dbg !55
  %43 = load i64, i64* %42, align 8, !dbg !55
  %44 = mul nsw i64 %36, %43, !dbg !55
  %45 = add nsw i64 %34, %44, !dbg !55
  %46 = bitcast i64* %__nv_MAIN__F1L28_1Arg2 to i8*, !dbg !55
  %47 = getelementptr i8, i8* %46, i64 96, !dbg !55
  %48 = bitcast i8* %47 to i8**, !dbg !55
  %49 = load i8*, i8** %48, align 8, !dbg !55
  %50 = getelementptr i8, i8* %49, i64 56, !dbg !55
  %51 = bitcast i8* %50 to i64*, !dbg !55
  %52 = load i64, i64* %51, align 8, !dbg !55
  %53 = add nsw i64 %45, %52, !dbg !55
  %54 = bitcast i64* %__nv_MAIN__F1L28_1Arg2 to i8*, !dbg !55
  %55 = getelementptr i8, i8* %54, i64 8, !dbg !55
  %56 = bitcast i8* %55 to float***, !dbg !55
  %57 = load float**, float*** %56, align 8, !dbg !55
  %58 = load float*, float** %57, align 8, !dbg !55
  %59 = getelementptr float, float* %58, i64 %53, !dbg !55
  %60 = load float, float* %59, align 4, !dbg !55
  %61 = bitcast i64* %__nv_MAIN__F1L28_1Arg2 to i8*, !dbg !55
  %62 = getelementptr i8, i8* %61, i64 88, !dbg !55
  %63 = bitcast i8* %62 to i32**, !dbg !55
  %64 = load i32*, i32** %63, align 8, !dbg !55
  %65 = load i32, i32* %64, align 4, !dbg !55
  %66 = sext i32 %65 to i64, !dbg !55
  %67 = load i32, i32* %j_325, align 4, !dbg !55
  call void @llvm.dbg.value(metadata i32 %67, metadata !54, metadata !DIExpression()), !dbg !52
  %68 = sext i32 %67 to i64, !dbg !55
  %69 = bitcast i64* %__nv_MAIN__F1L28_1Arg2 to i8*, !dbg !55
  %70 = getelementptr i8, i8* %69, i64 96, !dbg !55
  %71 = bitcast i8* %70 to i8**, !dbg !55
  %72 = load i8*, i8** %71, align 8, !dbg !55
  %73 = getelementptr i8, i8* %72, i64 160, !dbg !55
  %74 = bitcast i8* %73 to i64*, !dbg !55
  %75 = load i64, i64* %74, align 8, !dbg !55
  %76 = mul nsw i64 %68, %75, !dbg !55
  %77 = add nsw i64 %66, %76, !dbg !55
  %78 = bitcast i64* %__nv_MAIN__F1L28_1Arg2 to i8*, !dbg !55
  %79 = getelementptr i8, i8* %78, i64 96, !dbg !55
  %80 = bitcast i8* %79 to i8**, !dbg !55
  %81 = load i8*, i8** %80, align 8, !dbg !55
  %82 = getelementptr i8, i8* %81, i64 56, !dbg !55
  %83 = bitcast i8* %82 to i64*, !dbg !55
  %84 = load i64, i64* %83, align 8, !dbg !55
  %85 = add nsw i64 %77, %84, !dbg !55
  %86 = bitcast i64* %__nv_MAIN__F1L28_1Arg2 to i8*, !dbg !55
  %87 = getelementptr i8, i8* %86, i64 8, !dbg !55
  %88 = bitcast i8* %87 to i8***, !dbg !55
  %89 = load i8**, i8*** %88, align 8, !dbg !55
  %90 = load i8*, i8** %89, align 8, !dbg !55
  %91 = getelementptr i8, i8* %90, i64 -4, !dbg !55
  %92 = bitcast i8* %91 to float*, !dbg !55
  %93 = getelementptr float, float* %92, i64 %85, !dbg !55
  %94 = load float, float* %93, align 4, !dbg !55
  %95 = fadd fast float %60, %94, !dbg !55
  %96 = bitcast i64* %__nv_MAIN__F1L28_1Arg2 to i8*, !dbg !55
  %97 = getelementptr i8, i8* %96, i64 88, !dbg !55
  %98 = bitcast i8* %97 to i32**, !dbg !55
  %99 = load i32*, i32** %98, align 8, !dbg !55
  %100 = load i32, i32* %99, align 4, !dbg !55
  %101 = sext i32 %100 to i64, !dbg !55
  %102 = load i32, i32* %j_325, align 4, !dbg !55
  call void @llvm.dbg.value(metadata i32 %102, metadata !54, metadata !DIExpression()), !dbg !52
  %103 = sext i32 %102 to i64, !dbg !55
  %104 = bitcast i64* %__nv_MAIN__F1L28_1Arg2 to i8*, !dbg !55
  %105 = getelementptr i8, i8* %104, i64 96, !dbg !55
  %106 = bitcast i8* %105 to i8**, !dbg !55
  %107 = load i8*, i8** %106, align 8, !dbg !55
  %108 = getelementptr i8, i8* %107, i64 160, !dbg !55
  %109 = bitcast i8* %108 to i64*, !dbg !55
  %110 = load i64, i64* %109, align 8, !dbg !55
  %111 = mul nsw i64 %103, %110, !dbg !55
  %112 = add nsw i64 %101, %111, !dbg !55
  %113 = bitcast i64* %__nv_MAIN__F1L28_1Arg2 to i8*, !dbg !55
  %114 = getelementptr i8, i8* %113, i64 96, !dbg !55
  %115 = bitcast i8* %114 to i8**, !dbg !55
  %116 = load i8*, i8** %115, align 8, !dbg !55
  %117 = getelementptr i8, i8* %116, i64 56, !dbg !55
  %118 = bitcast i8* %117 to i64*, !dbg !55
  %119 = load i64, i64* %118, align 8, !dbg !55
  %120 = add nsw i64 %112, %119, !dbg !55
  %121 = bitcast i64* %__nv_MAIN__F1L28_1Arg2 to i8*, !dbg !55
  %122 = getelementptr i8, i8* %121, i64 8, !dbg !55
  %123 = bitcast i8* %122 to i8***, !dbg !55
  %124 = load i8**, i8*** %123, align 8, !dbg !55
  %125 = load i8*, i8** %124, align 8, !dbg !55
  %126 = getelementptr i8, i8* %125, i64 -4, !dbg !55
  %127 = bitcast i8* %126 to float*, !dbg !55
  %128 = getelementptr float, float* %127, i64 %120, !dbg !55
  store float %95, float* %128, align 4, !dbg !55
  %129 = load i32, i32* %.di0004p_353, align 4, !dbg !52
  %130 = load i32, i32* %j_325, align 4, !dbg !52
  call void @llvm.dbg.value(metadata i32 %130, metadata !54, metadata !DIExpression()), !dbg !52
  %131 = add nsw i32 %129, %130, !dbg !52
  store i32 %131, i32* %j_325, align 4, !dbg !52
  %132 = load i32, i32* %.dY0004p_350, align 4, !dbg !52
  %133 = sub nsw i32 %132, 1, !dbg !52
  store i32 %133, i32* %.dY0004p_350, align 4, !dbg !52
  %134 = load i32, i32* %.dY0004p_350, align 4, !dbg !52
  %135 = icmp sgt i32 %134, 0, !dbg !52
  br i1 %135, label %L.LB2_358, label %L.LB2_359, !dbg !52

L.LB2_359:                                        ; preds = %L.LB2_358, %L.LB2_490
  br label %L.LB2_349

L.LB2_349:                                        ; preds = %L.LB2_359, %L.LB2_324
  %136 = load i32, i32* %__gtid___nv_MAIN__F1L28_1__464, align 4, !dbg !52
  call void @__kmpc_for_static_fini(i64* null, i32 %136), !dbg !52
  br label %L.LB2_327

L.LB2_327:                                        ; preds = %L.LB2_349
  ret void, !dbg !52
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
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB053-inneronly1-orig-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb053_inneronly1_orig_no", scope: !2, file: !3, line: 12, type: !6, scopeLine: 12, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!6 = !DISubroutineType(cc: DW_CC_program, types: !7)
!7 = !{null}
!8 = !DILocalVariable(name: "omp_sched_static", scope: !5, file: !3, type: !9)
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 0, scope: !5)
!11 = !DILocalVariable(name: "omp_proc_bind_false", scope: !5, file: !3, type: !9)
!12 = !DILocalVariable(name: "omp_proc_bind_true", scope: !5, file: !3, type: !9)
!13 = !DILocalVariable(name: "omp_lock_hint_none", scope: !5, file: !3, type: !9)
!14 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !5, file: !3, type: !9)
!15 = !DILocation(line: 36, column: 1, scope: !5)
!16 = !DILocation(line: 12, column: 1, scope: !5)
!17 = !DILocalVariable(name: "a", scope: !5, file: !3, type: !18)
!18 = !DICompositeType(tag: DW_TAG_array_type, baseType: !19, size: 32, align: 32, elements: !20)
!19 = !DIBasicType(name: "real", size: 32, align: 32, encoding: DW_ATE_float)
!20 = !{!21, !21}
!21 = !DISubrange(count: 0, lowerBound: 1)
!22 = !DILocalVariable(scope: !5, file: !3, type: !23, flags: DIFlagArtificial)
!23 = !DICompositeType(tag: DW_TAG_array_type, baseType: !24, size: 1408, align: 64, elements: !25)
!24 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!25 = !{!26}
!26 = !DISubrange(count: 22, lowerBound: 1)
!27 = !DILocalVariable(scope: !5, file: !3, type: !24, flags: DIFlagArtificial)
!28 = !DILocation(line: 19, column: 1, scope: !5)
!29 = !DILocation(line: 21, column: 1, scope: !5)
!30 = !DILocalVariable(name: "i", scope: !5, file: !3, type: !9)
!31 = !DILocation(line: 22, column: 1, scope: !5)
!32 = !DILocalVariable(name: "j", scope: !5, file: !3, type: !9)
!33 = !DILocation(line: 23, column: 1, scope: !5)
!34 = !DILocation(line: 24, column: 1, scope: !5)
!35 = !DILocation(line: 25, column: 1, scope: !5)
!36 = !DILocation(line: 27, column: 1, scope: !5)
!37 = !DILocation(line: 28, column: 1, scope: !5)
!38 = !DILocation(line: 33, column: 1, scope: !5)
!39 = !DILocation(line: 35, column: 1, scope: !5)
!40 = distinct !DISubprogram(name: "__nv_MAIN__F1L28_1", scope: !2, file: !3, line: 28, type: !41, scopeLine: 28, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!41 = !DISubroutineType(types: !42)
!42 = !{null, !9, !24, !24}
!43 = !DILocalVariable(name: "__nv_MAIN__F1L28_1Arg0", arg: 1, scope: !40, file: !3, type: !9)
!44 = !DILocation(line: 0, scope: !40)
!45 = !DILocalVariable(name: "__nv_MAIN__F1L28_1Arg1", arg: 2, scope: !40, file: !3, type: !24)
!46 = !DILocalVariable(name: "__nv_MAIN__F1L28_1Arg2", arg: 3, scope: !40, file: !3, type: !24)
!47 = !DILocalVariable(name: "omp_sched_static", scope: !40, file: !3, type: !9)
!48 = !DILocalVariable(name: "omp_proc_bind_false", scope: !40, file: !3, type: !9)
!49 = !DILocalVariable(name: "omp_proc_bind_true", scope: !40, file: !3, type: !9)
!50 = !DILocalVariable(name: "omp_lock_hint_none", scope: !40, file: !3, type: !9)
!51 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !40, file: !3, type: !9)
!52 = !DILocation(line: 31, column: 1, scope: !40)
!53 = !DILocation(line: 29, column: 1, scope: !40)
!54 = !DILocalVariable(name: "j", scope: !40, file: !3, type: !9)
!55 = !DILocation(line: 30, column: 1, scope: !40)

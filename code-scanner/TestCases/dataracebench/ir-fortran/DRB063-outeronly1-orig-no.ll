; ModuleID = '/tmp/DRB063-outeronly1-orig-no-fe8d55.ll'
source_filename = "/tmp/DRB063-outeronly1-orig-no-fe8d55.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.struct_ul_MAIN__297 = type <{ i8* }>
%astruct.dt70 = type <{ i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8* }>

@.C283_MAIN_ = internal constant i32 0
@.C285_drb063_outeronly1_orig_no_foo = internal constant i32 1
@.C303_drb063_outeronly1_orig_no_foo = internal constant i32 27
@.C334_drb063_outeronly1_orig_no_foo = internal constant i64 4
@.C333_drb063_outeronly1_orig_no_foo = internal constant i64 27
@.C320_drb063_outeronly1_orig_no_foo = internal constant i32 100
@.C283_drb063_outeronly1_orig_no_foo = internal constant i32 0
@.C286_drb063_outeronly1_orig_no_foo = internal constant i64 1
@.C284_drb063_outeronly1_orig_no_foo = internal constant i64 0
@.C285___nv_drb063_outeronly1_orig_no_foo_F1L24_1 = internal constant i32 1
@.C283___nv_drb063_outeronly1_orig_no_foo_F1L24_1 = internal constant i32 0

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %.S0000_309 = alloca %struct.struct_ul_MAIN__297, align 8
  call void @llvm.dbg.value(metadata i32 1, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !14, metadata !DIExpression()), !dbg !10
  %0 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !15
  %1 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !15
  call void (i8*, ...) %1(i8* %0), !dbg !15
  br label %L.LB1_313

L.LB1_313:                                        ; preds = %L.entry
  %2 = bitcast %struct.struct_ul_MAIN__297* %.S0000_309 to i64*, !dbg !16
  call void @drb063_outeronly1_orig_no_foo(i64* %2), !dbg !16
  ret void, !dbg !17
}

define internal void @drb063_outeronly1_orig_no_foo(i64* %.S0000) #0 !dbg !18 {
L.entry:
  %__gtid_drb063_outeronly1_orig_no_foo_421 = alloca i32, align 4
  %.Z0978_322 = alloca float*, align 8
  %"b$sd1_332" = alloca [22 x i64], align 8
  %len_321 = alloca i32, align 4
  %z_b_0_310 = alloca i64, align 8
  %z_b_1_311 = alloca i64, align 8
  %z_e_67_317 = alloca i64, align 8
  %z_b_3_313 = alloca i64, align 8
  %z_b_4_314 = alloca i64, align 8
  %z_e_70_318 = alloca i64, align 8
  %z_b_2_312 = alloca i64, align 8
  %z_b_5_315 = alloca i64, align 8
  %z_b_6_316 = alloca i64, align 8
  %n_308 = alloca i32, align 4
  %m_309 = alloca i32, align 4
  %.uplevelArgPack0001_387 = alloca %astruct.dt70, align 16
  call void @llvm.dbg.declare(metadata i64* %.S0000, metadata !20, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i32 1, metadata !23, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i32 0, metadata !24, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i32 1, metadata !25, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i32 0, metadata !26, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i32 1, metadata !27, metadata !DIExpression()), !dbg !22
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !28
  store i32 %0, i32* %__gtid_drb063_outeronly1_orig_no_foo_421, align 4, !dbg !28
  call void @llvm.dbg.declare(metadata float** %.Z0978_322, metadata !29, metadata !DIExpression(DW_OP_deref)), !dbg !22
  %1 = bitcast float** %.Z0978_322 to i8**, !dbg !34
  store i8* null, i8** %1, align 8, !dbg !34
  call void @llvm.dbg.declare(metadata [22 x i64]* %"b$sd1_332", metadata !35, metadata !DIExpression()), !dbg !22
  %2 = bitcast [22 x i64]* %"b$sd1_332" to i64*, !dbg !34
  store i64 0, i64* %2, align 8, !dbg !34
  br label %L.LB2_366

L.LB2_366:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %len_321, metadata !40, metadata !DIExpression()), !dbg !22
  store i32 100, i32* %len_321, align 4, !dbg !41
  call void @llvm.dbg.declare(metadata i64* %z_b_0_310, metadata !42, metadata !DIExpression()), !dbg !22
  store i64 1, i64* %z_b_0_310, align 8, !dbg !43
  %3 = load i32, i32* %len_321, align 4, !dbg !43
  call void @llvm.dbg.value(metadata i32 %3, metadata !40, metadata !DIExpression()), !dbg !22
  %4 = sext i32 %3 to i64, !dbg !43
  call void @llvm.dbg.declare(metadata i64* %z_b_1_311, metadata !42, metadata !DIExpression()), !dbg !22
  store i64 %4, i64* %z_b_1_311, align 8, !dbg !43
  %5 = load i64, i64* %z_b_1_311, align 8, !dbg !43
  call void @llvm.dbg.value(metadata i64 %5, metadata !42, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.declare(metadata i64* %z_e_67_317, metadata !42, metadata !DIExpression()), !dbg !22
  store i64 %5, i64* %z_e_67_317, align 8, !dbg !43
  call void @llvm.dbg.declare(metadata i64* %z_b_3_313, metadata !42, metadata !DIExpression()), !dbg !22
  store i64 1, i64* %z_b_3_313, align 8, !dbg !43
  %6 = load i32, i32* %len_321, align 4, !dbg !43
  call void @llvm.dbg.value(metadata i32 %6, metadata !40, metadata !DIExpression()), !dbg !22
  %7 = sext i32 %6 to i64, !dbg !43
  call void @llvm.dbg.declare(metadata i64* %z_b_4_314, metadata !42, metadata !DIExpression()), !dbg !22
  store i64 %7, i64* %z_b_4_314, align 8, !dbg !43
  %8 = load i64, i64* %z_b_4_314, align 8, !dbg !43
  call void @llvm.dbg.value(metadata i64 %8, metadata !42, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.declare(metadata i64* %z_e_70_318, metadata !42, metadata !DIExpression()), !dbg !22
  store i64 %8, i64* %z_e_70_318, align 8, !dbg !43
  %9 = bitcast [22 x i64]* %"b$sd1_332" to i8*, !dbg !43
  %10 = bitcast i64* @.C284_drb063_outeronly1_orig_no_foo to i8*, !dbg !43
  %11 = bitcast i64* @.C333_drb063_outeronly1_orig_no_foo to i8*, !dbg !43
  %12 = bitcast i64* @.C334_drb063_outeronly1_orig_no_foo to i8*, !dbg !43
  %13 = bitcast i64* %z_b_0_310 to i8*, !dbg !43
  %14 = bitcast i64* %z_b_1_311 to i8*, !dbg !43
  %15 = bitcast i64* %z_b_3_313 to i8*, !dbg !43
  %16 = bitcast i64* %z_b_4_314 to i8*, !dbg !43
  %17 = bitcast void (...)* @f90_template2_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !43
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, ...) %17(i8* %9, i8* %10, i8* %11, i8* %12, i8* %13, i8* %14, i8* %15, i8* %16), !dbg !43
  %18 = bitcast [22 x i64]* %"b$sd1_332" to i8*, !dbg !43
  %19 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !43
  call void (i8*, i32, ...) %19(i8* %18, i32 27), !dbg !43
  %20 = load i64, i64* %z_b_1_311, align 8, !dbg !43
  call void @llvm.dbg.value(metadata i64 %20, metadata !42, metadata !DIExpression()), !dbg !22
  %21 = load i64, i64* %z_b_0_310, align 8, !dbg !43
  call void @llvm.dbg.value(metadata i64 %21, metadata !42, metadata !DIExpression()), !dbg !22
  %22 = sub nsw i64 %21, 1, !dbg !43
  %23 = sub nsw i64 %20, %22, !dbg !43
  call void @llvm.dbg.declare(metadata i64* %z_b_2_312, metadata !42, metadata !DIExpression()), !dbg !22
  store i64 %23, i64* %z_b_2_312, align 8, !dbg !43
  %24 = load i64, i64* %z_b_1_311, align 8, !dbg !43
  call void @llvm.dbg.value(metadata i64 %24, metadata !42, metadata !DIExpression()), !dbg !22
  %25 = load i64, i64* %z_b_0_310, align 8, !dbg !43
  call void @llvm.dbg.value(metadata i64 %25, metadata !42, metadata !DIExpression()), !dbg !22
  %26 = sub nsw i64 %25, 1, !dbg !43
  %27 = sub nsw i64 %24, %26, !dbg !43
  %28 = load i64, i64* %z_b_4_314, align 8, !dbg !43
  call void @llvm.dbg.value(metadata i64 %28, metadata !42, metadata !DIExpression()), !dbg !22
  %29 = load i64, i64* %z_b_3_313, align 8, !dbg !43
  call void @llvm.dbg.value(metadata i64 %29, metadata !42, metadata !DIExpression()), !dbg !22
  %30 = sub nsw i64 %29, 1, !dbg !43
  %31 = sub nsw i64 %28, %30, !dbg !43
  %32 = mul nsw i64 %27, %31, !dbg !43
  call void @llvm.dbg.declare(metadata i64* %z_b_5_315, metadata !42, metadata !DIExpression()), !dbg !22
  store i64 %32, i64* %z_b_5_315, align 8, !dbg !43
  %33 = load i64, i64* %z_b_0_310, align 8, !dbg !43
  call void @llvm.dbg.value(metadata i64 %33, metadata !42, metadata !DIExpression()), !dbg !22
  %34 = load i64, i64* %z_b_1_311, align 8, !dbg !43
  call void @llvm.dbg.value(metadata i64 %34, metadata !42, metadata !DIExpression()), !dbg !22
  %35 = load i64, i64* %z_b_0_310, align 8, !dbg !43
  call void @llvm.dbg.value(metadata i64 %35, metadata !42, metadata !DIExpression()), !dbg !22
  %36 = sub nsw i64 %35, 1, !dbg !43
  %37 = sub nsw i64 %34, %36, !dbg !43
  %38 = load i64, i64* %z_b_3_313, align 8, !dbg !43
  call void @llvm.dbg.value(metadata i64 %38, metadata !42, metadata !DIExpression()), !dbg !22
  %39 = mul nsw i64 %37, %38, !dbg !43
  %40 = add nsw i64 %33, %39, !dbg !43
  call void @llvm.dbg.declare(metadata i64* %z_b_6_316, metadata !42, metadata !DIExpression()), !dbg !22
  store i64 %40, i64* %z_b_6_316, align 8, !dbg !43
  %41 = bitcast i64* %z_b_5_315 to i8*, !dbg !43
  %42 = bitcast i64* @.C333_drb063_outeronly1_orig_no_foo to i8*, !dbg !43
  %43 = bitcast i64* @.C334_drb063_outeronly1_orig_no_foo to i8*, !dbg !43
  %44 = bitcast float** %.Z0978_322 to i8*, !dbg !43
  %45 = bitcast i64* @.C286_drb063_outeronly1_orig_no_foo to i8*, !dbg !43
  %46 = bitcast i64* @.C284_drb063_outeronly1_orig_no_foo to i8*, !dbg !43
  %47 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !43
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %47(i8* %41, i8* %42, i8* %43, i8* null, i8* %44, i8* null, i8* %45, i8* %46, i8* null, i64 0), !dbg !43
  %48 = load i32, i32* %len_321, align 4, !dbg !44
  call void @llvm.dbg.value(metadata i32 %48, metadata !40, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.declare(metadata i32* %n_308, metadata !45, metadata !DIExpression()), !dbg !22
  store i32 %48, i32* %n_308, align 4, !dbg !44
  %49 = load i32, i32* %len_321, align 4, !dbg !46
  call void @llvm.dbg.value(metadata i32 %49, metadata !40, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.declare(metadata i32* %m_309, metadata !47, metadata !DIExpression()), !dbg !22
  store i32 %49, i32* %m_309, align 4, !dbg !46
  %50 = bitcast i64* %.S0000 to i8*, !dbg !48
  %51 = bitcast %astruct.dt70* %.uplevelArgPack0001_387 to i8**, !dbg !48
  store i8* %50, i8** %51, align 8, !dbg !48
  %52 = bitcast i32* %n_308 to i8*, !dbg !48
  %53 = bitcast %astruct.dt70* %.uplevelArgPack0001_387 to i8*, !dbg !48
  %54 = getelementptr i8, i8* %53, i64 8, !dbg !48
  %55 = bitcast i8* %54 to i8**, !dbg !48
  store i8* %52, i8** %55, align 8, !dbg !48
  %56 = bitcast i32* %m_309 to i8*, !dbg !48
  %57 = bitcast %astruct.dt70* %.uplevelArgPack0001_387 to i8*, !dbg !48
  %58 = getelementptr i8, i8* %57, i64 16, !dbg !48
  %59 = bitcast i8* %58 to i8**, !dbg !48
  store i8* %56, i8** %59, align 8, !dbg !48
  %60 = bitcast float** %.Z0978_322 to i8*, !dbg !48
  %61 = bitcast %astruct.dt70* %.uplevelArgPack0001_387 to i8*, !dbg !48
  %62 = getelementptr i8, i8* %61, i64 24, !dbg !48
  %63 = bitcast i8* %62 to i8**, !dbg !48
  store i8* %60, i8** %63, align 8, !dbg !48
  %64 = bitcast float** %.Z0978_322 to i8*, !dbg !48
  %65 = bitcast %astruct.dt70* %.uplevelArgPack0001_387 to i8*, !dbg !48
  %66 = getelementptr i8, i8* %65, i64 32, !dbg !48
  %67 = bitcast i8* %66 to i8**, !dbg !48
  store i8* %64, i8** %67, align 8, !dbg !48
  %68 = bitcast i64* %z_b_0_310 to i8*, !dbg !48
  %69 = bitcast %astruct.dt70* %.uplevelArgPack0001_387 to i8*, !dbg !48
  %70 = getelementptr i8, i8* %69, i64 40, !dbg !48
  %71 = bitcast i8* %70 to i8**, !dbg !48
  store i8* %68, i8** %71, align 8, !dbg !48
  %72 = bitcast i64* %z_b_1_311 to i8*, !dbg !48
  %73 = bitcast %astruct.dt70* %.uplevelArgPack0001_387 to i8*, !dbg !48
  %74 = getelementptr i8, i8* %73, i64 48, !dbg !48
  %75 = bitcast i8* %74 to i8**, !dbg !48
  store i8* %72, i8** %75, align 8, !dbg !48
  %76 = bitcast i64* %z_e_67_317 to i8*, !dbg !48
  %77 = bitcast %astruct.dt70* %.uplevelArgPack0001_387 to i8*, !dbg !48
  %78 = getelementptr i8, i8* %77, i64 56, !dbg !48
  %79 = bitcast i8* %78 to i8**, !dbg !48
  store i8* %76, i8** %79, align 8, !dbg !48
  %80 = bitcast i64* %z_b_3_313 to i8*, !dbg !48
  %81 = bitcast %astruct.dt70* %.uplevelArgPack0001_387 to i8*, !dbg !48
  %82 = getelementptr i8, i8* %81, i64 64, !dbg !48
  %83 = bitcast i8* %82 to i8**, !dbg !48
  store i8* %80, i8** %83, align 8, !dbg !48
  %84 = bitcast i64* %z_b_4_314 to i8*, !dbg !48
  %85 = bitcast %astruct.dt70* %.uplevelArgPack0001_387 to i8*, !dbg !48
  %86 = getelementptr i8, i8* %85, i64 72, !dbg !48
  %87 = bitcast i8* %86 to i8**, !dbg !48
  store i8* %84, i8** %87, align 8, !dbg !48
  %88 = bitcast i64* %z_b_2_312 to i8*, !dbg !48
  %89 = bitcast %astruct.dt70* %.uplevelArgPack0001_387 to i8*, !dbg !48
  %90 = getelementptr i8, i8* %89, i64 80, !dbg !48
  %91 = bitcast i8* %90 to i8**, !dbg !48
  store i8* %88, i8** %91, align 8, !dbg !48
  %92 = bitcast i64* %z_e_70_318 to i8*, !dbg !48
  %93 = bitcast %astruct.dt70* %.uplevelArgPack0001_387 to i8*, !dbg !48
  %94 = getelementptr i8, i8* %93, i64 88, !dbg !48
  %95 = bitcast i8* %94 to i8**, !dbg !48
  store i8* %92, i8** %95, align 8, !dbg !48
  %96 = bitcast i64* %z_b_5_315 to i8*, !dbg !48
  %97 = bitcast %astruct.dt70* %.uplevelArgPack0001_387 to i8*, !dbg !48
  %98 = getelementptr i8, i8* %97, i64 96, !dbg !48
  %99 = bitcast i8* %98 to i8**, !dbg !48
  store i8* %96, i8** %99, align 8, !dbg !48
  %100 = bitcast i64* %z_b_6_316 to i8*, !dbg !48
  %101 = bitcast %astruct.dt70* %.uplevelArgPack0001_387 to i8*, !dbg !48
  %102 = getelementptr i8, i8* %101, i64 104, !dbg !48
  %103 = bitcast i8* %102 to i8**, !dbg !48
  store i8* %100, i8** %103, align 8, !dbg !48
  %104 = bitcast [22 x i64]* %"b$sd1_332" to i8*, !dbg !48
  %105 = bitcast %astruct.dt70* %.uplevelArgPack0001_387 to i8*, !dbg !48
  %106 = getelementptr i8, i8* %105, i64 112, !dbg !48
  %107 = bitcast i8* %106 to i8**, !dbg !48
  store i8* %104, i8** %107, align 8, !dbg !48
  br label %L.LB2_419, !dbg !48

L.LB2_419:                                        ; preds = %L.LB2_366
  %108 = bitcast void (i32*, i64*, i64*)* @__nv_drb063_outeronly1_orig_no_foo_F1L24_1_ to i64*, !dbg !48
  %109 = bitcast %astruct.dt70* %.uplevelArgPack0001_387 to i64*, !dbg !48
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %108, i64* %109), !dbg !48
  %110 = load float*, float** %.Z0978_322, align 8, !dbg !28
  call void @llvm.dbg.value(metadata float* %110, metadata !29, metadata !DIExpression()), !dbg !22
  %111 = bitcast float* %110 to i8*, !dbg !28
  %112 = bitcast i32 (...)* @f90_allocated_i8 to i32 (i8*, ...)*, !dbg !28
  %113 = call i32 (i8*, ...) %112(i8* %111), !dbg !28
  %114 = and i32 %113, 1, !dbg !28
  %115 = icmp eq i32 %114, 0, !dbg !28
  br i1 %115, label %L.LB2_354, label %L.LB2_441, !dbg !28

L.LB2_441:                                        ; preds = %L.LB2_419
  %116 = load float*, float** %.Z0978_322, align 8, !dbg !28
  call void @llvm.dbg.value(metadata float* %116, metadata !29, metadata !DIExpression()), !dbg !22
  %117 = bitcast float* %116 to i8*, !dbg !28
  %118 = bitcast i64* @.C284_drb063_outeronly1_orig_no_foo to i8*, !dbg !28
  %119 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !28
  call void (i8*, i8*, i8*, i8*, i64, ...) %119(i8* null, i8* %117, i8* %118, i8* null, i64 0), !dbg !28
  %120 = bitcast float** %.Z0978_322 to i8**, !dbg !28
  store i8* null, i8** %120, align 8, !dbg !28
  %121 = bitcast [22 x i64]* %"b$sd1_332" to i64*, !dbg !28
  store i64 0, i64* %121, align 8, !dbg !28
  br label %L.LB2_354

L.LB2_354:                                        ; preds = %L.LB2_441, %L.LB2_419
  ret void, !dbg !28
}

define internal void @__nv_drb063_outeronly1_orig_no_foo_F1L24_1_(i32* %__nv_drb063_outeronly1_orig_no_foo_F1L24_1Arg0, i64* %__nv_drb063_outeronly1_orig_no_foo_F1L24_1Arg1, i64* %__nv_drb063_outeronly1_orig_no_foo_F1L24_1Arg2) #0 !dbg !49 {
L.entry:
  %.S0000_357 = alloca i8*, align 8
  %__gtid___nv_drb063_outeronly1_orig_no_foo_F1L24_1__464 = alloca i32, align 4
  %.i0000p_328 = alloca i32, align 4
  %i_327 = alloca i32, align 4
  %.du0001p_342 = alloca i32, align 4
  %.de0001p_343 = alloca i32, align 4
  %.di0001p_344 = alloca i32, align 4
  %.ds0001p_345 = alloca i32, align 4
  %.dl0001p_347 = alloca i32, align 4
  %.dl0001p.copy_458 = alloca i32, align 4
  %.de0001p.copy_459 = alloca i32, align 4
  %.ds0001p.copy_460 = alloca i32, align 4
  %.dX0001p_346 = alloca i32, align 4
  %.dY0001p_341 = alloca i32, align 4
  %.dY0002p_353 = alloca i32, align 4
  %j_326 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_drb063_outeronly1_orig_no_foo_F1L24_1Arg0, metadata !52, metadata !DIExpression()), !dbg !53
  call void @llvm.dbg.declare(metadata i64* %__nv_drb063_outeronly1_orig_no_foo_F1L24_1Arg1, metadata !54, metadata !DIExpression()), !dbg !53
  call void @llvm.dbg.declare(metadata i64* %__nv_drb063_outeronly1_orig_no_foo_F1L24_1Arg2, metadata !55, metadata !DIExpression()), !dbg !53
  call void @llvm.dbg.value(metadata i32 1, metadata !56, metadata !DIExpression()), !dbg !53
  call void @llvm.dbg.value(metadata i32 0, metadata !57, metadata !DIExpression()), !dbg !53
  call void @llvm.dbg.value(metadata i32 1, metadata !58, metadata !DIExpression()), !dbg !53
  call void @llvm.dbg.value(metadata i32 0, metadata !59, metadata !DIExpression()), !dbg !53
  call void @llvm.dbg.value(metadata i32 1, metadata !60, metadata !DIExpression()), !dbg !53
  %0 = bitcast i64* %__nv_drb063_outeronly1_orig_no_foo_F1L24_1Arg2 to i8**, !dbg !61
  %1 = load i8*, i8** %0, align 8, !dbg !61
  %2 = bitcast i8** %.S0000_357 to i64*, !dbg !61
  store i8* %1, i8** %.S0000_357, align 8, !dbg !61
  %3 = load i32, i32* %__nv_drb063_outeronly1_orig_no_foo_F1L24_1Arg0, align 4, !dbg !62
  store i32 %3, i32* %__gtid___nv_drb063_outeronly1_orig_no_foo_F1L24_1__464, align 4, !dbg !62
  br label %L.LB3_449

L.LB3_449:                                        ; preds = %L.entry
  br label %L.LB3_325

L.LB3_325:                                        ; preds = %L.LB3_449
  store i32 0, i32* %.i0000p_328, align 4, !dbg !63
  call void @llvm.dbg.declare(metadata i32* %i_327, metadata !64, metadata !DIExpression()), !dbg !62
  store i32 1, i32* %i_327, align 4, !dbg !63
  %4 = bitcast i64* %__nv_drb063_outeronly1_orig_no_foo_F1L24_1Arg2 to i8*, !dbg !63
  %5 = getelementptr i8, i8* %4, i64 8, !dbg !63
  %6 = bitcast i8* %5 to i32**, !dbg !63
  %7 = load i32*, i32** %6, align 8, !dbg !63
  %8 = load i32, i32* %7, align 4, !dbg !63
  store i32 %8, i32* %.du0001p_342, align 4, !dbg !63
  %9 = bitcast i64* %__nv_drb063_outeronly1_orig_no_foo_F1L24_1Arg2 to i8*, !dbg !63
  %10 = getelementptr i8, i8* %9, i64 8, !dbg !63
  %11 = bitcast i8* %10 to i32**, !dbg !63
  %12 = load i32*, i32** %11, align 8, !dbg !63
  %13 = load i32, i32* %12, align 4, !dbg !63
  store i32 %13, i32* %.de0001p_343, align 4, !dbg !63
  store i32 1, i32* %.di0001p_344, align 4, !dbg !63
  %14 = load i32, i32* %.di0001p_344, align 4, !dbg !63
  store i32 %14, i32* %.ds0001p_345, align 4, !dbg !63
  store i32 1, i32* %.dl0001p_347, align 4, !dbg !63
  %15 = load i32, i32* %.dl0001p_347, align 4, !dbg !63
  store i32 %15, i32* %.dl0001p.copy_458, align 4, !dbg !63
  %16 = load i32, i32* %.de0001p_343, align 4, !dbg !63
  store i32 %16, i32* %.de0001p.copy_459, align 4, !dbg !63
  %17 = load i32, i32* %.ds0001p_345, align 4, !dbg !63
  store i32 %17, i32* %.ds0001p.copy_460, align 4, !dbg !63
  %18 = load i32, i32* %__gtid___nv_drb063_outeronly1_orig_no_foo_F1L24_1__464, align 4, !dbg !63
  %19 = bitcast i32* %.i0000p_328 to i64*, !dbg !63
  %20 = bitcast i32* %.dl0001p.copy_458 to i64*, !dbg !63
  %21 = bitcast i32* %.de0001p.copy_459 to i64*, !dbg !63
  %22 = bitcast i32* %.ds0001p.copy_460 to i64*, !dbg !63
  %23 = load i32, i32* %.ds0001p.copy_460, align 4, !dbg !63
  call void @__kmpc_for_static_init_4(i64* null, i32 %18, i32 34, i64* %19, i64* %20, i64* %21, i64* %22, i32 %23, i32 1), !dbg !63
  %24 = load i32, i32* %.dl0001p.copy_458, align 4, !dbg !63
  store i32 %24, i32* %.dl0001p_347, align 4, !dbg !63
  %25 = load i32, i32* %.de0001p.copy_459, align 4, !dbg !63
  store i32 %25, i32* %.de0001p_343, align 4, !dbg !63
  %26 = load i32, i32* %.ds0001p.copy_460, align 4, !dbg !63
  store i32 %26, i32* %.ds0001p_345, align 4, !dbg !63
  %27 = load i32, i32* %.dl0001p_347, align 4, !dbg !63
  store i32 %27, i32* %i_327, align 4, !dbg !63
  %28 = load i32, i32* %i_327, align 4, !dbg !63
  call void @llvm.dbg.value(metadata i32 %28, metadata !64, metadata !DIExpression()), !dbg !62
  store i32 %28, i32* %.dX0001p_346, align 4, !dbg !63
  %29 = load i32, i32* %.dX0001p_346, align 4, !dbg !63
  %30 = load i32, i32* %.du0001p_342, align 4, !dbg !63
  %31 = icmp sgt i32 %29, %30, !dbg !63
  br i1 %31, label %L.LB3_340, label %L.LB3_496, !dbg !63

L.LB3_496:                                        ; preds = %L.LB3_325
  %32 = load i32, i32* %.dX0001p_346, align 4, !dbg !63
  store i32 %32, i32* %i_327, align 4, !dbg !63
  %33 = load i32, i32* %.di0001p_344, align 4, !dbg !63
  %34 = load i32, i32* %.de0001p_343, align 4, !dbg !63
  %35 = load i32, i32* %.dX0001p_346, align 4, !dbg !63
  %36 = sub nsw i32 %34, %35, !dbg !63
  %37 = add nsw i32 %33, %36, !dbg !63
  %38 = load i32, i32* %.di0001p_344, align 4, !dbg !63
  %39 = sdiv i32 %37, %38, !dbg !63
  store i32 %39, i32* %.dY0001p_341, align 4, !dbg !63
  %40 = load i32, i32* %.dY0001p_341, align 4, !dbg !63
  %41 = icmp sle i32 %40, 0, !dbg !63
  br i1 %41, label %L.LB3_350, label %L.LB3_349, !dbg !63

L.LB3_349:                                        ; preds = %L.LB3_352, %L.LB3_496
  %42 = bitcast i64* %__nv_drb063_outeronly1_orig_no_foo_F1L24_1Arg2 to i8*, !dbg !65
  %43 = getelementptr i8, i8* %42, i64 16, !dbg !65
  %44 = bitcast i8* %43 to i32**, !dbg !65
  %45 = load i32*, i32** %44, align 8, !dbg !65
  %46 = load i32, i32* %45, align 4, !dbg !65
  %47 = sub nsw i32 %46, 1, !dbg !65
  store i32 %47, i32* %.dY0002p_353, align 4, !dbg !65
  call void @llvm.dbg.declare(metadata i32* %j_326, metadata !66, metadata !DIExpression()), !dbg !62
  store i32 1, i32* %j_326, align 4, !dbg !65
  %48 = load i32, i32* %.dY0002p_353, align 4, !dbg !65
  %49 = icmp sle i32 %48, 0, !dbg !65
  br i1 %49, label %L.LB3_352, label %L.LB3_351, !dbg !65

L.LB3_351:                                        ; preds = %L.LB3_351, %L.LB3_349
  %50 = load i32, i32* %i_327, align 4, !dbg !67
  call void @llvm.dbg.value(metadata i32 %50, metadata !64, metadata !DIExpression()), !dbg !62
  %51 = sext i32 %50 to i64, !dbg !67
  %52 = load i32, i32* %j_326, align 4, !dbg !67
  call void @llvm.dbg.value(metadata i32 %52, metadata !66, metadata !DIExpression()), !dbg !62
  %53 = sext i32 %52 to i64, !dbg !67
  %54 = add nsw i64 %53, 1, !dbg !67
  %55 = bitcast i64* %__nv_drb063_outeronly1_orig_no_foo_F1L24_1Arg2 to i8*, !dbg !67
  %56 = getelementptr i8, i8* %55, i64 112, !dbg !67
  %57 = bitcast i8* %56 to i8**, !dbg !67
  %58 = load i8*, i8** %57, align 8, !dbg !67
  %59 = getelementptr i8, i8* %58, i64 160, !dbg !67
  %60 = bitcast i8* %59 to i64*, !dbg !67
  %61 = load i64, i64* %60, align 8, !dbg !67
  %62 = mul nsw i64 %54, %61, !dbg !67
  %63 = add nsw i64 %51, %62, !dbg !67
  %64 = bitcast i64* %__nv_drb063_outeronly1_orig_no_foo_F1L24_1Arg2 to i8*, !dbg !67
  %65 = getelementptr i8, i8* %64, i64 112, !dbg !67
  %66 = bitcast i8* %65 to i8**, !dbg !67
  %67 = load i8*, i8** %66, align 8, !dbg !67
  %68 = getelementptr i8, i8* %67, i64 56, !dbg !67
  %69 = bitcast i8* %68 to i64*, !dbg !67
  %70 = load i64, i64* %69, align 8, !dbg !67
  %71 = add nsw i64 %63, %70, !dbg !67
  %72 = bitcast i64* %__nv_drb063_outeronly1_orig_no_foo_F1L24_1Arg2 to i8*, !dbg !67
  %73 = getelementptr i8, i8* %72, i64 32, !dbg !67
  %74 = bitcast i8* %73 to i8***, !dbg !67
  %75 = load i8**, i8*** %74, align 8, !dbg !67
  %76 = load i8*, i8** %75, align 8, !dbg !67
  %77 = getelementptr i8, i8* %76, i64 -4, !dbg !67
  %78 = bitcast i8* %77 to float*, !dbg !67
  %79 = getelementptr float, float* %78, i64 %71, !dbg !67
  %80 = load float, float* %79, align 4, !dbg !67
  %81 = bitcast i64* %__nv_drb063_outeronly1_orig_no_foo_F1L24_1Arg2 to i8*, !dbg !67
  %82 = getelementptr i8, i8* %81, i64 112, !dbg !67
  %83 = bitcast i8* %82 to i8**, !dbg !67
  %84 = load i8*, i8** %83, align 8, !dbg !67
  %85 = getelementptr i8, i8* %84, i64 56, !dbg !67
  %86 = bitcast i8* %85 to i64*, !dbg !67
  %87 = load i64, i64* %86, align 8, !dbg !67
  %88 = load i32, i32* %i_327, align 4, !dbg !67
  call void @llvm.dbg.value(metadata i32 %88, metadata !64, metadata !DIExpression()), !dbg !62
  %89 = sext i32 %88 to i64, !dbg !67
  %90 = load i32, i32* %j_326, align 4, !dbg !67
  call void @llvm.dbg.value(metadata i32 %90, metadata !66, metadata !DIExpression()), !dbg !62
  %91 = sext i32 %90 to i64, !dbg !67
  %92 = bitcast i64* %__nv_drb063_outeronly1_orig_no_foo_F1L24_1Arg2 to i8*, !dbg !67
  %93 = getelementptr i8, i8* %92, i64 112, !dbg !67
  %94 = bitcast i8* %93 to i8**, !dbg !67
  %95 = load i8*, i8** %94, align 8, !dbg !67
  %96 = getelementptr i8, i8* %95, i64 160, !dbg !67
  %97 = bitcast i8* %96 to i64*, !dbg !67
  %98 = load i64, i64* %97, align 8, !dbg !67
  %99 = mul nsw i64 %91, %98, !dbg !67
  %100 = add nsw i64 %89, %99, !dbg !67
  %101 = add nsw i64 %87, %100, !dbg !67
  %102 = bitcast i64* %__nv_drb063_outeronly1_orig_no_foo_F1L24_1Arg2 to i8*, !dbg !67
  %103 = getelementptr i8, i8* %102, i64 32, !dbg !67
  %104 = bitcast i8* %103 to i8***, !dbg !67
  %105 = load i8**, i8*** %104, align 8, !dbg !67
  %106 = load i8*, i8** %105, align 8, !dbg !67
  %107 = getelementptr i8, i8* %106, i64 -4, !dbg !67
  %108 = bitcast i8* %107 to float*, !dbg !67
  %109 = getelementptr float, float* %108, i64 %101, !dbg !67
  store float %80, float* %109, align 4, !dbg !67
  %110 = load i32, i32* %j_326, align 4, !dbg !68
  call void @llvm.dbg.value(metadata i32 %110, metadata !66, metadata !DIExpression()), !dbg !62
  %111 = add nsw i32 %110, 1, !dbg !68
  store i32 %111, i32* %j_326, align 4, !dbg !68
  %112 = load i32, i32* %.dY0002p_353, align 4, !dbg !68
  %113 = sub nsw i32 %112, 1, !dbg !68
  store i32 %113, i32* %.dY0002p_353, align 4, !dbg !68
  %114 = load i32, i32* %.dY0002p_353, align 4, !dbg !68
  %115 = icmp sgt i32 %114, 0, !dbg !68
  br i1 %115, label %L.LB3_351, label %L.LB3_352, !dbg !68

L.LB3_352:                                        ; preds = %L.LB3_351, %L.LB3_349
  %116 = load i32, i32* %.di0001p_344, align 4, !dbg !62
  %117 = load i32, i32* %i_327, align 4, !dbg !62
  call void @llvm.dbg.value(metadata i32 %117, metadata !64, metadata !DIExpression()), !dbg !62
  %118 = add nsw i32 %116, %117, !dbg !62
  store i32 %118, i32* %i_327, align 4, !dbg !62
  %119 = load i32, i32* %.dY0001p_341, align 4, !dbg !62
  %120 = sub nsw i32 %119, 1, !dbg !62
  store i32 %120, i32* %.dY0001p_341, align 4, !dbg !62
  %121 = load i32, i32* %.dY0001p_341, align 4, !dbg !62
  %122 = icmp sgt i32 %121, 0, !dbg !62
  br i1 %122, label %L.LB3_349, label %L.LB3_350, !dbg !62

L.LB3_350:                                        ; preds = %L.LB3_352, %L.LB3_496
  br label %L.LB3_340

L.LB3_340:                                        ; preds = %L.LB3_350, %L.LB3_325
  %123 = load i32, i32* %__gtid___nv_drb063_outeronly1_orig_no_foo_F1L24_1__464, align 4, !dbg !62
  call void @__kmpc_for_static_fini(i64* null, i32 %123), !dbg !62
  br label %L.LB3_329

L.LB3_329:                                        ; preds = %L.LB3_340
  ret void, !dbg !62
}

declare void @__kmpc_for_static_fini(i64*, i32) #0

declare void @__kmpc_for_static_init_4(i64*, i32, i32, i64*, i64*, i64*, i64*, i32, i32) #0

declare void @f90_dealloc03a_i8(...) #0

declare signext i32 @f90_allocated_i8(...) #0

declare void @f90_alloc04_chka_i8(...) #0

declare void @f90_set_intrin_type_i8(...) #0

declare void @f90_template2_i8(...) #0

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare signext i32 @__kmpc_global_thread_num(i64*) #0

declare void @__kmpc_fork_call(i64*, i32, i64*, i64*, ...) #0

declare void @__kmpc_end_serialized_parallel(i64*, i32) #0

declare void @__kmpc_serialized_parallel(i64*, i32) #0

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
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB063-outeronly1-orig-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb063_outeronly1_orig_no", scope: !2, file: !3, line: 10, type: !6, scopeLine: 10, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!6 = !DISubroutineType(cc: DW_CC_program, types: !7)
!7 = !{null}
!8 = !DILocalVariable(name: "omp_sched_static", scope: !5, file: !3, type: !9)
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 0, scope: !5)
!11 = !DILocalVariable(name: "omp_proc_bind_false", scope: !5, file: !3, type: !9)
!12 = !DILocalVariable(name: "omp_proc_bind_true", scope: !5, file: !3, type: !9)
!13 = !DILocalVariable(name: "omp_lock_hint_none", scope: !5, file: !3, type: !9)
!14 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !5, file: !3, type: !9)
!15 = !DILocation(line: 10, column: 1, scope: !5)
!16 = !DILocation(line: 14, column: 1, scope: !5)
!17 = !DILocation(line: 15, column: 1, scope: !5)
!18 = distinct !DISubprogram(name: "foo", scope: !5, file: !3, line: 16, type: !19, scopeLine: 16, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!19 = !DISubroutineType(types: !7)
!20 = !DILocalVariable(arg: 1, scope: !18, file: !3, type: !21, flags: DIFlagArtificial)
!21 = !DIBasicType(name: "uinteger*8", size: 64, align: 64, encoding: DW_ATE_unsigned)
!22 = !DILocation(line: 0, scope: !18)
!23 = !DILocalVariable(name: "omp_sched_static", scope: !18, file: !3, type: !9)
!24 = !DILocalVariable(name: "omp_proc_bind_false", scope: !18, file: !3, type: !9)
!25 = !DILocalVariable(name: "omp_proc_bind_true", scope: !18, file: !3, type: !9)
!26 = !DILocalVariable(name: "omp_lock_hint_none", scope: !18, file: !3, type: !9)
!27 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !18, file: !3, type: !9)
!28 = !DILocation(line: 32, column: 1, scope: !18)
!29 = !DILocalVariable(name: "b", scope: !18, file: !3, type: !30)
!30 = !DICompositeType(tag: DW_TAG_array_type, baseType: !31, size: 32, align: 32, elements: !32)
!31 = !DIBasicType(name: "real", size: 32, align: 32, encoding: DW_ATE_float)
!32 = !{!33, !33}
!33 = !DISubrange(count: 0, lowerBound: 1)
!34 = !DILocation(line: 16, column: 1, scope: !18)
!35 = !DILocalVariable(scope: !18, file: !3, type: !36, flags: DIFlagArtificial)
!36 = !DICompositeType(tag: DW_TAG_array_type, baseType: !37, size: 1408, align: 64, elements: !38)
!37 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!38 = !{!39}
!39 = !DISubrange(count: 22, lowerBound: 1)
!40 = !DILocalVariable(name: "len", scope: !18, file: !3, type: !9)
!41 = !DILocation(line: 20, column: 1, scope: !18)
!42 = !DILocalVariable(scope: !18, file: !3, type: !37, flags: DIFlagArtificial)
!43 = !DILocation(line: 21, column: 1, scope: !18)
!44 = !DILocation(line: 22, column: 1, scope: !18)
!45 = !DILocalVariable(name: "n", scope: !18, file: !3, type: !9)
!46 = !DILocation(line: 23, column: 1, scope: !18)
!47 = !DILocalVariable(name: "m", scope: !18, file: !3, type: !9)
!48 = !DILocation(line: 24, column: 1, scope: !18)
!49 = distinct !DISubprogram(name: "__nv_drb063_outeronly1_orig_no_foo_F1L24_1", scope: !2, file: !3, line: 24, type: !50, scopeLine: 24, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!50 = !DISubroutineType(types: !51)
!51 = !{null, !9, !37, !37}
!52 = !DILocalVariable(name: "__nv_drb063_outeronly1_orig_no_foo_F1L24_1Arg0", arg: 1, scope: !49, file: !3, type: !9)
!53 = !DILocation(line: 0, scope: !49)
!54 = !DILocalVariable(name: "__nv_drb063_outeronly1_orig_no_foo_F1L24_1Arg1", arg: 2, scope: !49, file: !3, type: !37)
!55 = !DILocalVariable(name: "__nv_drb063_outeronly1_orig_no_foo_F1L24_1Arg2", arg: 3, scope: !49, file: !3, type: !37)
!56 = !DILocalVariable(name: "omp_sched_static", scope: !49, file: !3, type: !9)
!57 = !DILocalVariable(name: "omp_proc_bind_false", scope: !49, file: !3, type: !9)
!58 = !DILocalVariable(name: "omp_proc_bind_true", scope: !49, file: !3, type: !9)
!59 = !DILocalVariable(name: "omp_lock_hint_none", scope: !49, file: !3, type: !9)
!60 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !49, file: !3, type: !9)
!61 = !DILocation(line: 24, column: 1, scope: !49)
!62 = !DILocation(line: 29, column: 1, scope: !49)
!63 = !DILocation(line: 25, column: 1, scope: !49)
!64 = !DILocalVariable(name: "i", scope: !49, file: !3, type: !9)
!65 = !DILocation(line: 26, column: 1, scope: !49)
!66 = !DILocalVariable(name: "j", scope: !49, file: !3, type: !9)
!67 = !DILocation(line: 27, column: 1, scope: !49)
!68 = !DILocation(line: 28, column: 1, scope: !49)

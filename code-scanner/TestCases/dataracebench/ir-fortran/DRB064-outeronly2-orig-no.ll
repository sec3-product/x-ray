; ModuleID = '/tmp/DRB064-outeronly2-orig-no-fbeb1e.ll'
source_filename = "/tmp/DRB064-outeronly2-orig-no-fbeb1e.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.struct_ul_MAIN__297 = type <{ i8* }>
%astruct.dt70 = type <{ i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8* }>

@.C283_MAIN_ = internal constant i32 0
@.C298_drb064_outeronly2_orig_no_foo = internal constant i32 2
@.C285_drb064_outeronly2_orig_no_foo = internal constant i32 1
@.C307_drb064_outeronly2_orig_no_foo = internal constant i32 27
@.C338_drb064_outeronly2_orig_no_foo = internal constant i64 4
@.C337_drb064_outeronly2_orig_no_foo = internal constant i64 27
@.C324_drb064_outeronly2_orig_no_foo = internal constant i32 100
@.C283_drb064_outeronly2_orig_no_foo = internal constant i32 0
@.C286_drb064_outeronly2_orig_no_foo = internal constant i64 1
@.C284_drb064_outeronly2_orig_no_foo = internal constant i64 0
@.C298___nv_drb064_outeronly2_orig_no_foo_F1L27_1 = internal constant i32 2
@.C285___nv_drb064_outeronly2_orig_no_foo_F1L27_1 = internal constant i32 1
@.C283___nv_drb064_outeronly2_orig_no_foo_F1L27_1 = internal constant i32 0

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
  call void @drb064_outeronly2_orig_no_foo(i64* %2), !dbg !16
  ret void, !dbg !17
}

define internal void @drb064_outeronly2_orig_no_foo(i64* %.S0000) #0 !dbg !18 {
L.entry:
  %__gtid_drb064_outeronly2_orig_no_foo_425 = alloca i32, align 4
  %.Z0978_326 = alloca float*, align 8
  %"b$sd1_336" = alloca [22 x i64], align 8
  %len_325 = alloca i32, align 4
  %z_b_0_314 = alloca i64, align 8
  %z_b_1_315 = alloca i64, align 8
  %z_e_67_321 = alloca i64, align 8
  %z_b_3_317 = alloca i64, align 8
  %z_b_4_318 = alloca i64, align 8
  %z_e_70_322 = alloca i64, align 8
  %z_b_2_316 = alloca i64, align 8
  %z_b_5_319 = alloca i64, align 8
  %z_b_6_320 = alloca i64, align 8
  %n_312 = alloca i32, align 4
  %m_313 = alloca i32, align 4
  %.uplevelArgPack0001_391 = alloca %astruct.dt70, align 16
  call void @llvm.dbg.declare(metadata i64* %.S0000, metadata !20, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i32 1, metadata !23, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i32 2, metadata !24, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i32 0, metadata !25, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i32 1, metadata !26, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i32 2, metadata !27, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i32 0, metadata !28, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i32 1, metadata !29, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i32 2, metadata !30, metadata !DIExpression()), !dbg !22
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !31
  store i32 %0, i32* %__gtid_drb064_outeronly2_orig_no_foo_425, align 4, !dbg !31
  call void @llvm.dbg.declare(metadata float** %.Z0978_326, metadata !32, metadata !DIExpression(DW_OP_deref)), !dbg !22
  %1 = bitcast float** %.Z0978_326 to i8**, !dbg !37
  store i8* null, i8** %1, align 8, !dbg !37
  call void @llvm.dbg.declare(metadata [22 x i64]* %"b$sd1_336", metadata !38, metadata !DIExpression()), !dbg !22
  %2 = bitcast [22 x i64]* %"b$sd1_336" to i64*, !dbg !37
  store i64 0, i64* %2, align 8, !dbg !37
  br label %L.LB2_370

L.LB2_370:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %len_325, metadata !43, metadata !DIExpression()), !dbg !22
  store i32 100, i32* %len_325, align 4, !dbg !44
  call void @llvm.dbg.declare(metadata i64* %z_b_0_314, metadata !45, metadata !DIExpression()), !dbg !22
  store i64 1, i64* %z_b_0_314, align 8, !dbg !46
  %3 = load i32, i32* %len_325, align 4, !dbg !46
  call void @llvm.dbg.value(metadata i32 %3, metadata !43, metadata !DIExpression()), !dbg !22
  %4 = sext i32 %3 to i64, !dbg !46
  call void @llvm.dbg.declare(metadata i64* %z_b_1_315, metadata !45, metadata !DIExpression()), !dbg !22
  store i64 %4, i64* %z_b_1_315, align 8, !dbg !46
  %5 = load i64, i64* %z_b_1_315, align 8, !dbg !46
  call void @llvm.dbg.value(metadata i64 %5, metadata !45, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.declare(metadata i64* %z_e_67_321, metadata !45, metadata !DIExpression()), !dbg !22
  store i64 %5, i64* %z_e_67_321, align 8, !dbg !46
  call void @llvm.dbg.declare(metadata i64* %z_b_3_317, metadata !45, metadata !DIExpression()), !dbg !22
  store i64 1, i64* %z_b_3_317, align 8, !dbg !46
  %6 = load i32, i32* %len_325, align 4, !dbg !46
  call void @llvm.dbg.value(metadata i32 %6, metadata !43, metadata !DIExpression()), !dbg !22
  %7 = sext i32 %6 to i64, !dbg !46
  call void @llvm.dbg.declare(metadata i64* %z_b_4_318, metadata !45, metadata !DIExpression()), !dbg !22
  store i64 %7, i64* %z_b_4_318, align 8, !dbg !46
  %8 = load i64, i64* %z_b_4_318, align 8, !dbg !46
  call void @llvm.dbg.value(metadata i64 %8, metadata !45, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.declare(metadata i64* %z_e_70_322, metadata !45, metadata !DIExpression()), !dbg !22
  store i64 %8, i64* %z_e_70_322, align 8, !dbg !46
  %9 = bitcast [22 x i64]* %"b$sd1_336" to i8*, !dbg !46
  %10 = bitcast i64* @.C284_drb064_outeronly2_orig_no_foo to i8*, !dbg !46
  %11 = bitcast i64* @.C337_drb064_outeronly2_orig_no_foo to i8*, !dbg !46
  %12 = bitcast i64* @.C338_drb064_outeronly2_orig_no_foo to i8*, !dbg !46
  %13 = bitcast i64* %z_b_0_314 to i8*, !dbg !46
  %14 = bitcast i64* %z_b_1_315 to i8*, !dbg !46
  %15 = bitcast i64* %z_b_3_317 to i8*, !dbg !46
  %16 = bitcast i64* %z_b_4_318 to i8*, !dbg !46
  %17 = bitcast void (...)* @f90_template2_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !46
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, ...) %17(i8* %9, i8* %10, i8* %11, i8* %12, i8* %13, i8* %14, i8* %15, i8* %16), !dbg !46
  %18 = bitcast [22 x i64]* %"b$sd1_336" to i8*, !dbg !46
  %19 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !46
  call void (i8*, i32, ...) %19(i8* %18, i32 27), !dbg !46
  %20 = load i64, i64* %z_b_1_315, align 8, !dbg !46
  call void @llvm.dbg.value(metadata i64 %20, metadata !45, metadata !DIExpression()), !dbg !22
  %21 = load i64, i64* %z_b_0_314, align 8, !dbg !46
  call void @llvm.dbg.value(metadata i64 %21, metadata !45, metadata !DIExpression()), !dbg !22
  %22 = sub nsw i64 %21, 1, !dbg !46
  %23 = sub nsw i64 %20, %22, !dbg !46
  call void @llvm.dbg.declare(metadata i64* %z_b_2_316, metadata !45, metadata !DIExpression()), !dbg !22
  store i64 %23, i64* %z_b_2_316, align 8, !dbg !46
  %24 = load i64, i64* %z_b_1_315, align 8, !dbg !46
  call void @llvm.dbg.value(metadata i64 %24, metadata !45, metadata !DIExpression()), !dbg !22
  %25 = load i64, i64* %z_b_0_314, align 8, !dbg !46
  call void @llvm.dbg.value(metadata i64 %25, metadata !45, metadata !DIExpression()), !dbg !22
  %26 = sub nsw i64 %25, 1, !dbg !46
  %27 = sub nsw i64 %24, %26, !dbg !46
  %28 = load i64, i64* %z_b_4_318, align 8, !dbg !46
  call void @llvm.dbg.value(metadata i64 %28, metadata !45, metadata !DIExpression()), !dbg !22
  %29 = load i64, i64* %z_b_3_317, align 8, !dbg !46
  call void @llvm.dbg.value(metadata i64 %29, metadata !45, metadata !DIExpression()), !dbg !22
  %30 = sub nsw i64 %29, 1, !dbg !46
  %31 = sub nsw i64 %28, %30, !dbg !46
  %32 = mul nsw i64 %27, %31, !dbg !46
  call void @llvm.dbg.declare(metadata i64* %z_b_5_319, metadata !45, metadata !DIExpression()), !dbg !22
  store i64 %32, i64* %z_b_5_319, align 8, !dbg !46
  %33 = load i64, i64* %z_b_0_314, align 8, !dbg !46
  call void @llvm.dbg.value(metadata i64 %33, metadata !45, metadata !DIExpression()), !dbg !22
  %34 = load i64, i64* %z_b_1_315, align 8, !dbg !46
  call void @llvm.dbg.value(metadata i64 %34, metadata !45, metadata !DIExpression()), !dbg !22
  %35 = load i64, i64* %z_b_0_314, align 8, !dbg !46
  call void @llvm.dbg.value(metadata i64 %35, metadata !45, metadata !DIExpression()), !dbg !22
  %36 = sub nsw i64 %35, 1, !dbg !46
  %37 = sub nsw i64 %34, %36, !dbg !46
  %38 = load i64, i64* %z_b_3_317, align 8, !dbg !46
  call void @llvm.dbg.value(metadata i64 %38, metadata !45, metadata !DIExpression()), !dbg !22
  %39 = mul nsw i64 %37, %38, !dbg !46
  %40 = add nsw i64 %33, %39, !dbg !46
  call void @llvm.dbg.declare(metadata i64* %z_b_6_320, metadata !45, metadata !DIExpression()), !dbg !22
  store i64 %40, i64* %z_b_6_320, align 8, !dbg !46
  %41 = bitcast i64* %z_b_5_319 to i8*, !dbg !46
  %42 = bitcast i64* @.C337_drb064_outeronly2_orig_no_foo to i8*, !dbg !46
  %43 = bitcast i64* @.C338_drb064_outeronly2_orig_no_foo to i8*, !dbg !46
  %44 = bitcast float** %.Z0978_326 to i8*, !dbg !46
  %45 = bitcast i64* @.C286_drb064_outeronly2_orig_no_foo to i8*, !dbg !46
  %46 = bitcast i64* @.C284_drb064_outeronly2_orig_no_foo to i8*, !dbg !46
  %47 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !46
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %47(i8* %41, i8* %42, i8* %43, i8* null, i8* %44, i8* null, i8* %45, i8* %46, i8* null, i64 0), !dbg !46
  %48 = load i32, i32* %len_325, align 4, !dbg !47
  call void @llvm.dbg.value(metadata i32 %48, metadata !43, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.declare(metadata i32* %n_312, metadata !48, metadata !DIExpression()), !dbg !22
  store i32 %48, i32* %n_312, align 4, !dbg !47
  %49 = load i32, i32* %len_325, align 4, !dbg !49
  call void @llvm.dbg.value(metadata i32 %49, metadata !43, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.declare(metadata i32* %m_313, metadata !50, metadata !DIExpression()), !dbg !22
  store i32 %49, i32* %m_313, align 4, !dbg !49
  %50 = bitcast i64* %.S0000 to i8*, !dbg !51
  %51 = bitcast %astruct.dt70* %.uplevelArgPack0001_391 to i8**, !dbg !51
  store i8* %50, i8** %51, align 8, !dbg !51
  %52 = bitcast i32* %n_312 to i8*, !dbg !51
  %53 = bitcast %astruct.dt70* %.uplevelArgPack0001_391 to i8*, !dbg !51
  %54 = getelementptr i8, i8* %53, i64 8, !dbg !51
  %55 = bitcast i8* %54 to i8**, !dbg !51
  store i8* %52, i8** %55, align 8, !dbg !51
  %56 = bitcast i32* %m_313 to i8*, !dbg !51
  %57 = bitcast %astruct.dt70* %.uplevelArgPack0001_391 to i8*, !dbg !51
  %58 = getelementptr i8, i8* %57, i64 16, !dbg !51
  %59 = bitcast i8* %58 to i8**, !dbg !51
  store i8* %56, i8** %59, align 8, !dbg !51
  %60 = bitcast float** %.Z0978_326 to i8*, !dbg !51
  %61 = bitcast %astruct.dt70* %.uplevelArgPack0001_391 to i8*, !dbg !51
  %62 = getelementptr i8, i8* %61, i64 24, !dbg !51
  %63 = bitcast i8* %62 to i8**, !dbg !51
  store i8* %60, i8** %63, align 8, !dbg !51
  %64 = bitcast float** %.Z0978_326 to i8*, !dbg !51
  %65 = bitcast %astruct.dt70* %.uplevelArgPack0001_391 to i8*, !dbg !51
  %66 = getelementptr i8, i8* %65, i64 32, !dbg !51
  %67 = bitcast i8* %66 to i8**, !dbg !51
  store i8* %64, i8** %67, align 8, !dbg !51
  %68 = bitcast i64* %z_b_0_314 to i8*, !dbg !51
  %69 = bitcast %astruct.dt70* %.uplevelArgPack0001_391 to i8*, !dbg !51
  %70 = getelementptr i8, i8* %69, i64 40, !dbg !51
  %71 = bitcast i8* %70 to i8**, !dbg !51
  store i8* %68, i8** %71, align 8, !dbg !51
  %72 = bitcast i64* %z_b_1_315 to i8*, !dbg !51
  %73 = bitcast %astruct.dt70* %.uplevelArgPack0001_391 to i8*, !dbg !51
  %74 = getelementptr i8, i8* %73, i64 48, !dbg !51
  %75 = bitcast i8* %74 to i8**, !dbg !51
  store i8* %72, i8** %75, align 8, !dbg !51
  %76 = bitcast i64* %z_e_67_321 to i8*, !dbg !51
  %77 = bitcast %astruct.dt70* %.uplevelArgPack0001_391 to i8*, !dbg !51
  %78 = getelementptr i8, i8* %77, i64 56, !dbg !51
  %79 = bitcast i8* %78 to i8**, !dbg !51
  store i8* %76, i8** %79, align 8, !dbg !51
  %80 = bitcast i64* %z_b_3_317 to i8*, !dbg !51
  %81 = bitcast %astruct.dt70* %.uplevelArgPack0001_391 to i8*, !dbg !51
  %82 = getelementptr i8, i8* %81, i64 64, !dbg !51
  %83 = bitcast i8* %82 to i8**, !dbg !51
  store i8* %80, i8** %83, align 8, !dbg !51
  %84 = bitcast i64* %z_b_4_318 to i8*, !dbg !51
  %85 = bitcast %astruct.dt70* %.uplevelArgPack0001_391 to i8*, !dbg !51
  %86 = getelementptr i8, i8* %85, i64 72, !dbg !51
  %87 = bitcast i8* %86 to i8**, !dbg !51
  store i8* %84, i8** %87, align 8, !dbg !51
  %88 = bitcast i64* %z_b_2_316 to i8*, !dbg !51
  %89 = bitcast %astruct.dt70* %.uplevelArgPack0001_391 to i8*, !dbg !51
  %90 = getelementptr i8, i8* %89, i64 80, !dbg !51
  %91 = bitcast i8* %90 to i8**, !dbg !51
  store i8* %88, i8** %91, align 8, !dbg !51
  %92 = bitcast i64* %z_e_70_322 to i8*, !dbg !51
  %93 = bitcast %astruct.dt70* %.uplevelArgPack0001_391 to i8*, !dbg !51
  %94 = getelementptr i8, i8* %93, i64 88, !dbg !51
  %95 = bitcast i8* %94 to i8**, !dbg !51
  store i8* %92, i8** %95, align 8, !dbg !51
  %96 = bitcast i64* %z_b_5_319 to i8*, !dbg !51
  %97 = bitcast %astruct.dt70* %.uplevelArgPack0001_391 to i8*, !dbg !51
  %98 = getelementptr i8, i8* %97, i64 96, !dbg !51
  %99 = bitcast i8* %98 to i8**, !dbg !51
  store i8* %96, i8** %99, align 8, !dbg !51
  %100 = bitcast i64* %z_b_6_320 to i8*, !dbg !51
  %101 = bitcast %astruct.dt70* %.uplevelArgPack0001_391 to i8*, !dbg !51
  %102 = getelementptr i8, i8* %101, i64 104, !dbg !51
  %103 = bitcast i8* %102 to i8**, !dbg !51
  store i8* %100, i8** %103, align 8, !dbg !51
  %104 = bitcast [22 x i64]* %"b$sd1_336" to i8*, !dbg !51
  %105 = bitcast %astruct.dt70* %.uplevelArgPack0001_391 to i8*, !dbg !51
  %106 = getelementptr i8, i8* %105, i64 112, !dbg !51
  %107 = bitcast i8* %106 to i8**, !dbg !51
  store i8* %104, i8** %107, align 8, !dbg !51
  br label %L.LB2_423, !dbg !51

L.LB2_423:                                        ; preds = %L.LB2_370
  %108 = bitcast void (i32*, i64*, i64*)* @__nv_drb064_outeronly2_orig_no_foo_F1L27_1_ to i64*, !dbg !51
  %109 = bitcast %astruct.dt70* %.uplevelArgPack0001_391 to i64*, !dbg !51
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %108, i64* %109), !dbg !51
  %110 = load float*, float** %.Z0978_326, align 8, !dbg !31
  call void @llvm.dbg.value(metadata float* %110, metadata !32, metadata !DIExpression()), !dbg !22
  %111 = bitcast float* %110 to i8*, !dbg !31
  %112 = bitcast i32 (...)* @f90_allocated_i8 to i32 (i8*, ...)*, !dbg !31
  %113 = call i32 (i8*, ...) %112(i8* %111), !dbg !31
  %114 = and i32 %113, 1, !dbg !31
  %115 = icmp eq i32 %114, 0, !dbg !31
  br i1 %115, label %L.LB2_358, label %L.LB2_445, !dbg !31

L.LB2_445:                                        ; preds = %L.LB2_423
  %116 = load float*, float** %.Z0978_326, align 8, !dbg !31
  call void @llvm.dbg.value(metadata float* %116, metadata !32, metadata !DIExpression()), !dbg !22
  %117 = bitcast float* %116 to i8*, !dbg !31
  %118 = bitcast i64* @.C284_drb064_outeronly2_orig_no_foo to i8*, !dbg !31
  %119 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !31
  call void (i8*, i8*, i8*, i8*, i64, ...) %119(i8* null, i8* %117, i8* %118, i8* null, i64 0), !dbg !31
  %120 = bitcast float** %.Z0978_326 to i8**, !dbg !31
  store i8* null, i8** %120, align 8, !dbg !31
  %121 = bitcast [22 x i64]* %"b$sd1_336" to i64*, !dbg !31
  store i64 0, i64* %121, align 8, !dbg !31
  br label %L.LB2_358

L.LB2_358:                                        ; preds = %L.LB2_445, %L.LB2_423
  ret void, !dbg !31
}

define internal void @__nv_drb064_outeronly2_orig_no_foo_F1L27_1_(i32* %__nv_drb064_outeronly2_orig_no_foo_F1L27_1Arg0, i64* %__nv_drb064_outeronly2_orig_no_foo_F1L27_1Arg1, i64* %__nv_drb064_outeronly2_orig_no_foo_F1L27_1Arg2) #0 !dbg !52 {
L.entry:
  %.S0000_361 = alloca i8*, align 8
  %__gtid___nv_drb064_outeronly2_orig_no_foo_F1L27_1__468 = alloca i32, align 4
  %.i0000p_332 = alloca i32, align 4
  %i_331 = alloca i32, align 4
  %.du0001p_346 = alloca i32, align 4
  %.de0001p_347 = alloca i32, align 4
  %.di0001p_348 = alloca i32, align 4
  %.ds0001p_349 = alloca i32, align 4
  %.dl0001p_351 = alloca i32, align 4
  %.dl0001p.copy_462 = alloca i32, align 4
  %.de0001p.copy_463 = alloca i32, align 4
  %.ds0001p.copy_464 = alloca i32, align 4
  %.dX0001p_350 = alloca i32, align 4
  %.dY0001p_345 = alloca i32, align 4
  %.dY0002p_357 = alloca i32, align 4
  %j_330 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_drb064_outeronly2_orig_no_foo_F1L27_1Arg0, metadata !55, metadata !DIExpression()), !dbg !56
  call void @llvm.dbg.declare(metadata i64* %__nv_drb064_outeronly2_orig_no_foo_F1L27_1Arg1, metadata !57, metadata !DIExpression()), !dbg !56
  call void @llvm.dbg.declare(metadata i64* %__nv_drb064_outeronly2_orig_no_foo_F1L27_1Arg2, metadata !58, metadata !DIExpression()), !dbg !56
  call void @llvm.dbg.value(metadata i32 1, metadata !59, metadata !DIExpression()), !dbg !56
  call void @llvm.dbg.value(metadata i32 2, metadata !60, metadata !DIExpression()), !dbg !56
  call void @llvm.dbg.value(metadata i32 0, metadata !61, metadata !DIExpression()), !dbg !56
  call void @llvm.dbg.value(metadata i32 1, metadata !62, metadata !DIExpression()), !dbg !56
  call void @llvm.dbg.value(metadata i32 2, metadata !63, metadata !DIExpression()), !dbg !56
  call void @llvm.dbg.value(metadata i32 0, metadata !64, metadata !DIExpression()), !dbg !56
  call void @llvm.dbg.value(metadata i32 1, metadata !65, metadata !DIExpression()), !dbg !56
  call void @llvm.dbg.value(metadata i32 2, metadata !66, metadata !DIExpression()), !dbg !56
  %0 = bitcast i64* %__nv_drb064_outeronly2_orig_no_foo_F1L27_1Arg2 to i8**, !dbg !67
  %1 = load i8*, i8** %0, align 8, !dbg !67
  %2 = bitcast i8** %.S0000_361 to i64*, !dbg !67
  store i8* %1, i8** %.S0000_361, align 8, !dbg !67
  %3 = load i32, i32* %__nv_drb064_outeronly2_orig_no_foo_F1L27_1Arg0, align 4, !dbg !68
  store i32 %3, i32* %__gtid___nv_drb064_outeronly2_orig_no_foo_F1L27_1__468, align 4, !dbg !68
  br label %L.LB3_453

L.LB3_453:                                        ; preds = %L.entry
  br label %L.LB3_329

L.LB3_329:                                        ; preds = %L.LB3_453
  store i32 0, i32* %.i0000p_332, align 4, !dbg !69
  call void @llvm.dbg.declare(metadata i32* %i_331, metadata !70, metadata !DIExpression()), !dbg !68
  store i32 1, i32* %i_331, align 4, !dbg !69
  %4 = bitcast i64* %__nv_drb064_outeronly2_orig_no_foo_F1L27_1Arg2 to i8*, !dbg !69
  %5 = getelementptr i8, i8* %4, i64 8, !dbg !69
  %6 = bitcast i8* %5 to i32**, !dbg !69
  %7 = load i32*, i32** %6, align 8, !dbg !69
  %8 = load i32, i32* %7, align 4, !dbg !69
  store i32 %8, i32* %.du0001p_346, align 4, !dbg !69
  %9 = bitcast i64* %__nv_drb064_outeronly2_orig_no_foo_F1L27_1Arg2 to i8*, !dbg !69
  %10 = getelementptr i8, i8* %9, i64 8, !dbg !69
  %11 = bitcast i8* %10 to i32**, !dbg !69
  %12 = load i32*, i32** %11, align 8, !dbg !69
  %13 = load i32, i32* %12, align 4, !dbg !69
  store i32 %13, i32* %.de0001p_347, align 4, !dbg !69
  store i32 1, i32* %.di0001p_348, align 4, !dbg !69
  %14 = load i32, i32* %.di0001p_348, align 4, !dbg !69
  store i32 %14, i32* %.ds0001p_349, align 4, !dbg !69
  store i32 1, i32* %.dl0001p_351, align 4, !dbg !69
  %15 = load i32, i32* %.dl0001p_351, align 4, !dbg !69
  store i32 %15, i32* %.dl0001p.copy_462, align 4, !dbg !69
  %16 = load i32, i32* %.de0001p_347, align 4, !dbg !69
  store i32 %16, i32* %.de0001p.copy_463, align 4, !dbg !69
  %17 = load i32, i32* %.ds0001p_349, align 4, !dbg !69
  store i32 %17, i32* %.ds0001p.copy_464, align 4, !dbg !69
  %18 = load i32, i32* %__gtid___nv_drb064_outeronly2_orig_no_foo_F1L27_1__468, align 4, !dbg !69
  %19 = bitcast i32* %.i0000p_332 to i64*, !dbg !69
  %20 = bitcast i32* %.dl0001p.copy_462 to i64*, !dbg !69
  %21 = bitcast i32* %.de0001p.copy_463 to i64*, !dbg !69
  %22 = bitcast i32* %.ds0001p.copy_464 to i64*, !dbg !69
  %23 = load i32, i32* %.ds0001p.copy_464, align 4, !dbg !69
  call void @__kmpc_for_static_init_4(i64* null, i32 %18, i32 34, i64* %19, i64* %20, i64* %21, i64* %22, i32 %23, i32 1), !dbg !69
  %24 = load i32, i32* %.dl0001p.copy_462, align 4, !dbg !69
  store i32 %24, i32* %.dl0001p_351, align 4, !dbg !69
  %25 = load i32, i32* %.de0001p.copy_463, align 4, !dbg !69
  store i32 %25, i32* %.de0001p_347, align 4, !dbg !69
  %26 = load i32, i32* %.ds0001p.copy_464, align 4, !dbg !69
  store i32 %26, i32* %.ds0001p_349, align 4, !dbg !69
  %27 = load i32, i32* %.dl0001p_351, align 4, !dbg !69
  store i32 %27, i32* %i_331, align 4, !dbg !69
  %28 = load i32, i32* %i_331, align 4, !dbg !69
  call void @llvm.dbg.value(metadata i32 %28, metadata !70, metadata !DIExpression()), !dbg !68
  store i32 %28, i32* %.dX0001p_350, align 4, !dbg !69
  %29 = load i32, i32* %.dX0001p_350, align 4, !dbg !69
  %30 = load i32, i32* %.du0001p_346, align 4, !dbg !69
  %31 = icmp sgt i32 %29, %30, !dbg !69
  br i1 %31, label %L.LB3_344, label %L.LB3_500, !dbg !69

L.LB3_500:                                        ; preds = %L.LB3_329
  %32 = load i32, i32* %.dX0001p_350, align 4, !dbg !69
  store i32 %32, i32* %i_331, align 4, !dbg !69
  %33 = load i32, i32* %.di0001p_348, align 4, !dbg !69
  %34 = load i32, i32* %.de0001p_347, align 4, !dbg !69
  %35 = load i32, i32* %.dX0001p_350, align 4, !dbg !69
  %36 = sub nsw i32 %34, %35, !dbg !69
  %37 = add nsw i32 %33, %36, !dbg !69
  %38 = load i32, i32* %.di0001p_348, align 4, !dbg !69
  %39 = sdiv i32 %37, %38, !dbg !69
  store i32 %39, i32* %.dY0001p_345, align 4, !dbg !69
  %40 = load i32, i32* %.dY0001p_345, align 4, !dbg !69
  %41 = icmp sle i32 %40, 0, !dbg !69
  br i1 %41, label %L.LB3_354, label %L.LB3_353, !dbg !69

L.LB3_353:                                        ; preds = %L.LB3_356, %L.LB3_500
  %42 = bitcast i64* %__nv_drb064_outeronly2_orig_no_foo_F1L27_1Arg2 to i8*, !dbg !71
  %43 = getelementptr i8, i8* %42, i64 16, !dbg !71
  %44 = bitcast i8* %43 to i32**, !dbg !71
  %45 = load i32*, i32** %44, align 8, !dbg !71
  %46 = load i32, i32* %45, align 4, !dbg !71
  %47 = sub nsw i32 %46, 1, !dbg !71
  store i32 %47, i32* %.dY0002p_357, align 4, !dbg !71
  call void @llvm.dbg.declare(metadata i32* %j_330, metadata !72, metadata !DIExpression()), !dbg !68
  store i32 2, i32* %j_330, align 4, !dbg !71
  %48 = load i32, i32* %.dY0002p_357, align 4, !dbg !71
  %49 = icmp sle i32 %48, 0, !dbg !71
  br i1 %49, label %L.LB3_356, label %L.LB3_355, !dbg !71

L.LB3_355:                                        ; preds = %L.LB3_355, %L.LB3_353
  %50 = load i32, i32* %i_331, align 4, !dbg !73
  call void @llvm.dbg.value(metadata i32 %50, metadata !70, metadata !DIExpression()), !dbg !68
  %51 = sext i32 %50 to i64, !dbg !73
  %52 = load i32, i32* %j_330, align 4, !dbg !73
  call void @llvm.dbg.value(metadata i32 %52, metadata !72, metadata !DIExpression()), !dbg !68
  %53 = sext i32 %52 to i64, !dbg !73
  %54 = sub nsw i64 %53, 1, !dbg !73
  %55 = bitcast i64* %__nv_drb064_outeronly2_orig_no_foo_F1L27_1Arg2 to i8*, !dbg !73
  %56 = getelementptr i8, i8* %55, i64 112, !dbg !73
  %57 = bitcast i8* %56 to i8**, !dbg !73
  %58 = load i8*, i8** %57, align 8, !dbg !73
  %59 = getelementptr i8, i8* %58, i64 160, !dbg !73
  %60 = bitcast i8* %59 to i64*, !dbg !73
  %61 = load i64, i64* %60, align 8, !dbg !73
  %62 = mul nsw i64 %54, %61, !dbg !73
  %63 = add nsw i64 %51, %62, !dbg !73
  %64 = bitcast i64* %__nv_drb064_outeronly2_orig_no_foo_F1L27_1Arg2 to i8*, !dbg !73
  %65 = getelementptr i8, i8* %64, i64 112, !dbg !73
  %66 = bitcast i8* %65 to i8**, !dbg !73
  %67 = load i8*, i8** %66, align 8, !dbg !73
  %68 = getelementptr i8, i8* %67, i64 56, !dbg !73
  %69 = bitcast i8* %68 to i64*, !dbg !73
  %70 = load i64, i64* %69, align 8, !dbg !73
  %71 = add nsw i64 %63, %70, !dbg !73
  %72 = bitcast i64* %__nv_drb064_outeronly2_orig_no_foo_F1L27_1Arg2 to i8*, !dbg !73
  %73 = getelementptr i8, i8* %72, i64 32, !dbg !73
  %74 = bitcast i8* %73 to i8***, !dbg !73
  %75 = load i8**, i8*** %74, align 8, !dbg !73
  %76 = load i8*, i8** %75, align 8, !dbg !73
  %77 = getelementptr i8, i8* %76, i64 -4, !dbg !73
  %78 = bitcast i8* %77 to float*, !dbg !73
  %79 = getelementptr float, float* %78, i64 %71, !dbg !73
  %80 = load float, float* %79, align 4, !dbg !73
  %81 = bitcast i64* %__nv_drb064_outeronly2_orig_no_foo_F1L27_1Arg2 to i8*, !dbg !73
  %82 = getelementptr i8, i8* %81, i64 112, !dbg !73
  %83 = bitcast i8* %82 to i8**, !dbg !73
  %84 = load i8*, i8** %83, align 8, !dbg !73
  %85 = getelementptr i8, i8* %84, i64 56, !dbg !73
  %86 = bitcast i8* %85 to i64*, !dbg !73
  %87 = load i64, i64* %86, align 8, !dbg !73
  %88 = load i32, i32* %i_331, align 4, !dbg !73
  call void @llvm.dbg.value(metadata i32 %88, metadata !70, metadata !DIExpression()), !dbg !68
  %89 = sext i32 %88 to i64, !dbg !73
  %90 = load i32, i32* %j_330, align 4, !dbg !73
  call void @llvm.dbg.value(metadata i32 %90, metadata !72, metadata !DIExpression()), !dbg !68
  %91 = sext i32 %90 to i64, !dbg !73
  %92 = bitcast i64* %__nv_drb064_outeronly2_orig_no_foo_F1L27_1Arg2 to i8*, !dbg !73
  %93 = getelementptr i8, i8* %92, i64 112, !dbg !73
  %94 = bitcast i8* %93 to i8**, !dbg !73
  %95 = load i8*, i8** %94, align 8, !dbg !73
  %96 = getelementptr i8, i8* %95, i64 160, !dbg !73
  %97 = bitcast i8* %96 to i64*, !dbg !73
  %98 = load i64, i64* %97, align 8, !dbg !73
  %99 = mul nsw i64 %91, %98, !dbg !73
  %100 = add nsw i64 %89, %99, !dbg !73
  %101 = add nsw i64 %87, %100, !dbg !73
  %102 = bitcast i64* %__nv_drb064_outeronly2_orig_no_foo_F1L27_1Arg2 to i8*, !dbg !73
  %103 = getelementptr i8, i8* %102, i64 32, !dbg !73
  %104 = bitcast i8* %103 to i8***, !dbg !73
  %105 = load i8**, i8*** %104, align 8, !dbg !73
  %106 = load i8*, i8** %105, align 8, !dbg !73
  %107 = getelementptr i8, i8* %106, i64 -4, !dbg !73
  %108 = bitcast i8* %107 to float*, !dbg !73
  %109 = getelementptr float, float* %108, i64 %101, !dbg !73
  store float %80, float* %109, align 4, !dbg !73
  %110 = load i32, i32* %j_330, align 4, !dbg !74
  call void @llvm.dbg.value(metadata i32 %110, metadata !72, metadata !DIExpression()), !dbg !68
  %111 = add nsw i32 %110, 1, !dbg !74
  store i32 %111, i32* %j_330, align 4, !dbg !74
  %112 = load i32, i32* %.dY0002p_357, align 4, !dbg !74
  %113 = sub nsw i32 %112, 1, !dbg !74
  store i32 %113, i32* %.dY0002p_357, align 4, !dbg !74
  %114 = load i32, i32* %.dY0002p_357, align 4, !dbg !74
  %115 = icmp sgt i32 %114, 0, !dbg !74
  br i1 %115, label %L.LB3_355, label %L.LB3_356, !dbg !74

L.LB3_356:                                        ; preds = %L.LB3_355, %L.LB3_353
  %116 = load i32, i32* %.di0001p_348, align 4, !dbg !68
  %117 = load i32, i32* %i_331, align 4, !dbg !68
  call void @llvm.dbg.value(metadata i32 %117, metadata !70, metadata !DIExpression()), !dbg !68
  %118 = add nsw i32 %116, %117, !dbg !68
  store i32 %118, i32* %i_331, align 4, !dbg !68
  %119 = load i32, i32* %.dY0001p_345, align 4, !dbg !68
  %120 = sub nsw i32 %119, 1, !dbg !68
  store i32 %120, i32* %.dY0001p_345, align 4, !dbg !68
  %121 = load i32, i32* %.dY0001p_345, align 4, !dbg !68
  %122 = icmp sgt i32 %121, 0, !dbg !68
  br i1 %122, label %L.LB3_353, label %L.LB3_354, !dbg !68

L.LB3_354:                                        ; preds = %L.LB3_356, %L.LB3_500
  br label %L.LB3_344

L.LB3_344:                                        ; preds = %L.LB3_354, %L.LB3_329
  %123 = load i32, i32* %__gtid___nv_drb064_outeronly2_orig_no_foo_F1L27_1__468, align 4, !dbg !68
  call void @__kmpc_for_static_fini(i64* null, i32 %123), !dbg !68
  br label %L.LB3_333

L.LB3_333:                                        ; preds = %L.LB3_344
  ret void, !dbg !68
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
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB064-outeronly2-orig-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb064_outeronly2_orig_no", scope: !2, file: !3, line: 12, type: !6, scopeLine: 12, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!6 = !DISubroutineType(cc: DW_CC_program, types: !7)
!7 = !{null}
!8 = !DILocalVariable(name: "omp_sched_static", scope: !5, file: !3, type: !9)
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 0, scope: !5)
!11 = !DILocalVariable(name: "omp_proc_bind_false", scope: !5, file: !3, type: !9)
!12 = !DILocalVariable(name: "omp_proc_bind_true", scope: !5, file: !3, type: !9)
!13 = !DILocalVariable(name: "omp_lock_hint_none", scope: !5, file: !3, type: !9)
!14 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !5, file: !3, type: !9)
!15 = !DILocation(line: 12, column: 1, scope: !5)
!16 = !DILocation(line: 16, column: 1, scope: !5)
!17 = !DILocation(line: 17, column: 1, scope: !5)
!18 = distinct !DISubprogram(name: "foo", scope: !5, file: !3, line: 18, type: !19, scopeLine: 18, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!19 = !DISubroutineType(types: !7)
!20 = !DILocalVariable(arg: 1, scope: !18, file: !3, type: !21, flags: DIFlagArtificial)
!21 = !DIBasicType(name: "uinteger*8", size: 64, align: 64, encoding: DW_ATE_unsigned)
!22 = !DILocation(line: 0, scope: !18)
!23 = !DILocalVariable(name: "omp_sched_static", scope: !18, file: !3, type: !9)
!24 = !DILocalVariable(name: "omp_sched_dynamic", scope: !18, file: !3, type: !9)
!25 = !DILocalVariable(name: "omp_proc_bind_false", scope: !18, file: !3, type: !9)
!26 = !DILocalVariable(name: "omp_proc_bind_true", scope: !18, file: !3, type: !9)
!27 = !DILocalVariable(name: "omp_proc_bind_master", scope: !18, file: !3, type: !9)
!28 = !DILocalVariable(name: "omp_lock_hint_none", scope: !18, file: !3, type: !9)
!29 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !18, file: !3, type: !9)
!30 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !18, file: !3, type: !9)
!31 = !DILocation(line: 34, column: 1, scope: !18)
!32 = !DILocalVariable(name: "b", scope: !18, file: !3, type: !33)
!33 = !DICompositeType(tag: DW_TAG_array_type, baseType: !34, size: 32, align: 32, elements: !35)
!34 = !DIBasicType(name: "real", size: 32, align: 32, encoding: DW_ATE_float)
!35 = !{!36, !36}
!36 = !DISubrange(count: 0, lowerBound: 1)
!37 = !DILocation(line: 18, column: 1, scope: !18)
!38 = !DILocalVariable(scope: !18, file: !3, type: !39, flags: DIFlagArtificial)
!39 = !DICompositeType(tag: DW_TAG_array_type, baseType: !40, size: 1408, align: 64, elements: !41)
!40 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!41 = !{!42}
!42 = !DISubrange(count: 22, lowerBound: 1)
!43 = !DILocalVariable(name: "len", scope: !18, file: !3, type: !9)
!44 = !DILocation(line: 22, column: 1, scope: !18)
!45 = !DILocalVariable(scope: !18, file: !3, type: !40, flags: DIFlagArtificial)
!46 = !DILocation(line: 23, column: 1, scope: !18)
!47 = !DILocation(line: 24, column: 1, scope: !18)
!48 = !DILocalVariable(name: "n", scope: !18, file: !3, type: !9)
!49 = !DILocation(line: 25, column: 1, scope: !18)
!50 = !DILocalVariable(name: "m", scope: !18, file: !3, type: !9)
!51 = !DILocation(line: 27, column: 1, scope: !18)
!52 = distinct !DISubprogram(name: "__nv_drb064_outeronly2_orig_no_foo_F1L27_1", scope: !2, file: !3, line: 27, type: !53, scopeLine: 27, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!53 = !DISubroutineType(types: !54)
!54 = !{null, !9, !40, !40}
!55 = !DILocalVariable(name: "__nv_drb064_outeronly2_orig_no_foo_F1L27_1Arg0", arg: 1, scope: !52, file: !3, type: !9)
!56 = !DILocation(line: 0, scope: !52)
!57 = !DILocalVariable(name: "__nv_drb064_outeronly2_orig_no_foo_F1L27_1Arg1", arg: 2, scope: !52, file: !3, type: !40)
!58 = !DILocalVariable(name: "__nv_drb064_outeronly2_orig_no_foo_F1L27_1Arg2", arg: 3, scope: !52, file: !3, type: !40)
!59 = !DILocalVariable(name: "omp_sched_static", scope: !52, file: !3, type: !9)
!60 = !DILocalVariable(name: "omp_sched_dynamic", scope: !52, file: !3, type: !9)
!61 = !DILocalVariable(name: "omp_proc_bind_false", scope: !52, file: !3, type: !9)
!62 = !DILocalVariable(name: "omp_proc_bind_true", scope: !52, file: !3, type: !9)
!63 = !DILocalVariable(name: "omp_proc_bind_master", scope: !52, file: !3, type: !9)
!64 = !DILocalVariable(name: "omp_lock_hint_none", scope: !52, file: !3, type: !9)
!65 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !52, file: !3, type: !9)
!66 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !52, file: !3, type: !9)
!67 = !DILocation(line: 27, column: 1, scope: !52)
!68 = !DILocation(line: 32, column: 1, scope: !52)
!69 = !DILocation(line: 28, column: 1, scope: !52)
!70 = !DILocalVariable(name: "i", scope: !52, file: !3, type: !9)
!71 = !DILocation(line: 29, column: 1, scope: !52)
!72 = !DILocalVariable(name: "j", scope: !52, file: !3, type: !9)
!73 = !DILocation(line: 30, column: 1, scope: !52)
!74 = !DILocation(line: 31, column: 1, scope: !52)

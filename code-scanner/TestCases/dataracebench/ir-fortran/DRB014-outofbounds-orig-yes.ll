; ModuleID = '/tmp/DRB014-outofbounds-orig-yes-dd359a.ll'
source_filename = "/tmp/DRB014-outofbounds-orig-yes-dd359a.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%astruct.dt68 = type <{ i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8* }>

@.C342_MAIN_ = internal constant i64 50
@.C309_MAIN_ = internal constant i32 14
@.C341_MAIN_ = internal constant [9 x i8] c"b(50,50)="
@.C338_MAIN_ = internal constant i32 6
@.C335_MAIN_ = internal constant [56 x i8] c"micro-benchmarks-fortran/DRB014-outofbounds-orig-yes.f95"
@.C337_MAIN_ = internal constant i32 45
@.C285_MAIN_ = internal constant i32 1
@.C300_MAIN_ = internal constant i32 2
@.C310_MAIN_ = internal constant i32 27
@.C352_MAIN_ = internal constant i64 4
@.C351_MAIN_ = internal constant i64 27
@.C325_MAIN_ = internal constant i32 100
@.C286_MAIN_ = internal constant i64 1
@.C284_MAIN_ = internal constant i64 0
@.C283_MAIN_ = internal constant i32 0
@.C285___nv_MAIN__F1L38_1 = internal constant i32 1
@.C300___nv_MAIN__F1L38_1 = internal constant i32 2
@.C283___nv_MAIN__F1L38_1 = internal constant i32 0

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__432 = alloca i32, align 4
  %.Z0972_326 = alloca float*, align 8
  %"b$sd1_350" = alloca [22 x i64], align 8
  %n_313 = alloca i32, align 4
  %m_314 = alloca i32, align 4
  %z_b_0_315 = alloca i64, align 8
  %z_b_1_316 = alloca i64, align 8
  %z_e_63_322 = alloca i64, align 8
  %z_b_3_318 = alloca i64, align 8
  %z_b_4_319 = alloca i64, align 8
  %z_e_66_323 = alloca i64, align 8
  %z_b_2_317 = alloca i64, align 8
  %z_b_5_320 = alloca i64, align 8
  %z_b_6_321 = alloca i64, align 8
  %.uplevelArgPack0001_401 = alloca %astruct.dt68, align 16
  %z__io_340 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 2, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 2, metadata !14, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !15, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !16, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 2, metadata !17, metadata !DIExpression()), !dbg !10
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !18
  store i32 %0, i32* %__gtid_MAIN__432, align 4, !dbg !18
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !19
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !19
  call void (i8*, ...) %2(i8* %1), !dbg !19
  call void @llvm.dbg.declare(metadata float** %.Z0972_326, metadata !20, metadata !DIExpression(DW_OP_deref)), !dbg !10
  %3 = bitcast float** %.Z0972_326 to i8**, !dbg !19
  store i8* null, i8** %3, align 8, !dbg !19
  call void @llvm.dbg.declare(metadata [22 x i64]* %"b$sd1_350", metadata !25, metadata !DIExpression()), !dbg !10
  %4 = bitcast [22 x i64]* %"b$sd1_350" to i64*, !dbg !19
  store i64 0, i64* %4, align 8, !dbg !19
  br label %L.LB1_381

L.LB1_381:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %n_313, metadata !30, metadata !DIExpression()), !dbg !10
  store i32 100, i32* %n_313, align 4, !dbg !31
  call void @llvm.dbg.declare(metadata i32* %m_314, metadata !32, metadata !DIExpression()), !dbg !10
  store i32 100, i32* %m_314, align 4, !dbg !33
  call void @llvm.dbg.declare(metadata i64* %z_b_0_315, metadata !34, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_0_315, align 8, !dbg !35
  %5 = load i32, i32* %n_313, align 4, !dbg !35
  call void @llvm.dbg.value(metadata i32 %5, metadata !30, metadata !DIExpression()), !dbg !10
  %6 = sext i32 %5 to i64, !dbg !35
  call void @llvm.dbg.declare(metadata i64* %z_b_1_316, metadata !34, metadata !DIExpression()), !dbg !10
  store i64 %6, i64* %z_b_1_316, align 8, !dbg !35
  %7 = load i64, i64* %z_b_1_316, align 8, !dbg !35
  call void @llvm.dbg.value(metadata i64 %7, metadata !34, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_63_322, metadata !34, metadata !DIExpression()), !dbg !10
  store i64 %7, i64* %z_e_63_322, align 8, !dbg !35
  call void @llvm.dbg.declare(metadata i64* %z_b_3_318, metadata !34, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_3_318, align 8, !dbg !35
  %8 = load i32, i32* %m_314, align 4, !dbg !35
  call void @llvm.dbg.value(metadata i32 %8, metadata !32, metadata !DIExpression()), !dbg !10
  %9 = sext i32 %8 to i64, !dbg !35
  call void @llvm.dbg.declare(metadata i64* %z_b_4_319, metadata !34, metadata !DIExpression()), !dbg !10
  store i64 %9, i64* %z_b_4_319, align 8, !dbg !35
  %10 = load i64, i64* %z_b_4_319, align 8, !dbg !35
  call void @llvm.dbg.value(metadata i64 %10, metadata !34, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_66_323, metadata !34, metadata !DIExpression()), !dbg !10
  store i64 %10, i64* %z_e_66_323, align 8, !dbg !35
  %11 = bitcast [22 x i64]* %"b$sd1_350" to i8*, !dbg !35
  %12 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !35
  %13 = bitcast i64* @.C351_MAIN_ to i8*, !dbg !35
  %14 = bitcast i64* @.C352_MAIN_ to i8*, !dbg !35
  %15 = bitcast i64* %z_b_0_315 to i8*, !dbg !35
  %16 = bitcast i64* %z_b_1_316 to i8*, !dbg !35
  %17 = bitcast i64* %z_b_3_318 to i8*, !dbg !35
  %18 = bitcast i64* %z_b_4_319 to i8*, !dbg !35
  %19 = bitcast void (...)* @f90_template2_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !35
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, ...) %19(i8* %11, i8* %12, i8* %13, i8* %14, i8* %15, i8* %16, i8* %17, i8* %18), !dbg !35
  %20 = bitcast [22 x i64]* %"b$sd1_350" to i8*, !dbg !35
  %21 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !35
  call void (i8*, i32, ...) %21(i8* %20, i32 27), !dbg !35
  %22 = load i64, i64* %z_b_1_316, align 8, !dbg !35
  call void @llvm.dbg.value(metadata i64 %22, metadata !34, metadata !DIExpression()), !dbg !10
  %23 = load i64, i64* %z_b_0_315, align 8, !dbg !35
  call void @llvm.dbg.value(metadata i64 %23, metadata !34, metadata !DIExpression()), !dbg !10
  %24 = sub nsw i64 %23, 1, !dbg !35
  %25 = sub nsw i64 %22, %24, !dbg !35
  call void @llvm.dbg.declare(metadata i64* %z_b_2_317, metadata !34, metadata !DIExpression()), !dbg !10
  store i64 %25, i64* %z_b_2_317, align 8, !dbg !35
  %26 = load i64, i64* %z_b_1_316, align 8, !dbg !35
  call void @llvm.dbg.value(metadata i64 %26, metadata !34, metadata !DIExpression()), !dbg !10
  %27 = load i64, i64* %z_b_0_315, align 8, !dbg !35
  call void @llvm.dbg.value(metadata i64 %27, metadata !34, metadata !DIExpression()), !dbg !10
  %28 = sub nsw i64 %27, 1, !dbg !35
  %29 = sub nsw i64 %26, %28, !dbg !35
  %30 = load i64, i64* %z_b_4_319, align 8, !dbg !35
  call void @llvm.dbg.value(metadata i64 %30, metadata !34, metadata !DIExpression()), !dbg !10
  %31 = load i64, i64* %z_b_3_318, align 8, !dbg !35
  call void @llvm.dbg.value(metadata i64 %31, metadata !34, metadata !DIExpression()), !dbg !10
  %32 = sub nsw i64 %31, 1, !dbg !35
  %33 = sub nsw i64 %30, %32, !dbg !35
  %34 = mul nsw i64 %29, %33, !dbg !35
  call void @llvm.dbg.declare(metadata i64* %z_b_5_320, metadata !34, metadata !DIExpression()), !dbg !10
  store i64 %34, i64* %z_b_5_320, align 8, !dbg !35
  %35 = load i64, i64* %z_b_0_315, align 8, !dbg !35
  call void @llvm.dbg.value(metadata i64 %35, metadata !34, metadata !DIExpression()), !dbg !10
  %36 = load i64, i64* %z_b_1_316, align 8, !dbg !35
  call void @llvm.dbg.value(metadata i64 %36, metadata !34, metadata !DIExpression()), !dbg !10
  %37 = load i64, i64* %z_b_0_315, align 8, !dbg !35
  call void @llvm.dbg.value(metadata i64 %37, metadata !34, metadata !DIExpression()), !dbg !10
  %38 = sub nsw i64 %37, 1, !dbg !35
  %39 = sub nsw i64 %36, %38, !dbg !35
  %40 = load i64, i64* %z_b_3_318, align 8, !dbg !35
  call void @llvm.dbg.value(metadata i64 %40, metadata !34, metadata !DIExpression()), !dbg !10
  %41 = mul nsw i64 %39, %40, !dbg !35
  %42 = add nsw i64 %35, %41, !dbg !35
  call void @llvm.dbg.declare(metadata i64* %z_b_6_321, metadata !34, metadata !DIExpression()), !dbg !10
  store i64 %42, i64* %z_b_6_321, align 8, !dbg !35
  %43 = bitcast i64* %z_b_5_320 to i8*, !dbg !35
  %44 = bitcast i64* @.C351_MAIN_ to i8*, !dbg !35
  %45 = bitcast i64* @.C352_MAIN_ to i8*, !dbg !35
  %46 = bitcast float** %.Z0972_326 to i8*, !dbg !35
  %47 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !35
  %48 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !35
  %49 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !35
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %49(i8* %43, i8* %44, i8* %45, i8* null, i8* %46, i8* null, i8* %47, i8* %48, i8* null, i64 0), !dbg !35
  %50 = bitcast i32* %n_313 to i8*, !dbg !36
  %51 = bitcast %astruct.dt68* %.uplevelArgPack0001_401 to i8**, !dbg !36
  store i8* %50, i8** %51, align 8, !dbg !36
  %52 = bitcast i32* %m_314 to i8*, !dbg !36
  %53 = bitcast %astruct.dt68* %.uplevelArgPack0001_401 to i8*, !dbg !36
  %54 = getelementptr i8, i8* %53, i64 8, !dbg !36
  %55 = bitcast i8* %54 to i8**, !dbg !36
  store i8* %52, i8** %55, align 8, !dbg !36
  %56 = bitcast float** %.Z0972_326 to i8*, !dbg !36
  %57 = bitcast %astruct.dt68* %.uplevelArgPack0001_401 to i8*, !dbg !36
  %58 = getelementptr i8, i8* %57, i64 16, !dbg !36
  %59 = bitcast i8* %58 to i8**, !dbg !36
  store i8* %56, i8** %59, align 8, !dbg !36
  %60 = bitcast float** %.Z0972_326 to i8*, !dbg !36
  %61 = bitcast %astruct.dt68* %.uplevelArgPack0001_401 to i8*, !dbg !36
  %62 = getelementptr i8, i8* %61, i64 24, !dbg !36
  %63 = bitcast i8* %62 to i8**, !dbg !36
  store i8* %60, i8** %63, align 8, !dbg !36
  %64 = bitcast i64* %z_b_0_315 to i8*, !dbg !36
  %65 = bitcast %astruct.dt68* %.uplevelArgPack0001_401 to i8*, !dbg !36
  %66 = getelementptr i8, i8* %65, i64 32, !dbg !36
  %67 = bitcast i8* %66 to i8**, !dbg !36
  store i8* %64, i8** %67, align 8, !dbg !36
  %68 = bitcast i64* %z_b_1_316 to i8*, !dbg !36
  %69 = bitcast %astruct.dt68* %.uplevelArgPack0001_401 to i8*, !dbg !36
  %70 = getelementptr i8, i8* %69, i64 40, !dbg !36
  %71 = bitcast i8* %70 to i8**, !dbg !36
  store i8* %68, i8** %71, align 8, !dbg !36
  %72 = bitcast i64* %z_e_63_322 to i8*, !dbg !36
  %73 = bitcast %astruct.dt68* %.uplevelArgPack0001_401 to i8*, !dbg !36
  %74 = getelementptr i8, i8* %73, i64 48, !dbg !36
  %75 = bitcast i8* %74 to i8**, !dbg !36
  store i8* %72, i8** %75, align 8, !dbg !36
  %76 = bitcast i64* %z_b_3_318 to i8*, !dbg !36
  %77 = bitcast %astruct.dt68* %.uplevelArgPack0001_401 to i8*, !dbg !36
  %78 = getelementptr i8, i8* %77, i64 56, !dbg !36
  %79 = bitcast i8* %78 to i8**, !dbg !36
  store i8* %76, i8** %79, align 8, !dbg !36
  %80 = bitcast i64* %z_b_4_319 to i8*, !dbg !36
  %81 = bitcast %astruct.dt68* %.uplevelArgPack0001_401 to i8*, !dbg !36
  %82 = getelementptr i8, i8* %81, i64 64, !dbg !36
  %83 = bitcast i8* %82 to i8**, !dbg !36
  store i8* %80, i8** %83, align 8, !dbg !36
  %84 = bitcast i64* %z_b_2_317 to i8*, !dbg !36
  %85 = bitcast %astruct.dt68* %.uplevelArgPack0001_401 to i8*, !dbg !36
  %86 = getelementptr i8, i8* %85, i64 72, !dbg !36
  %87 = bitcast i8* %86 to i8**, !dbg !36
  store i8* %84, i8** %87, align 8, !dbg !36
  %88 = bitcast i64* %z_e_66_323 to i8*, !dbg !36
  %89 = bitcast %astruct.dt68* %.uplevelArgPack0001_401 to i8*, !dbg !36
  %90 = getelementptr i8, i8* %89, i64 80, !dbg !36
  %91 = bitcast i8* %90 to i8**, !dbg !36
  store i8* %88, i8** %91, align 8, !dbg !36
  %92 = bitcast i64* %z_b_5_320 to i8*, !dbg !36
  %93 = bitcast %astruct.dt68* %.uplevelArgPack0001_401 to i8*, !dbg !36
  %94 = getelementptr i8, i8* %93, i64 88, !dbg !36
  %95 = bitcast i8* %94 to i8**, !dbg !36
  store i8* %92, i8** %95, align 8, !dbg !36
  %96 = bitcast i64* %z_b_6_321 to i8*, !dbg !36
  %97 = bitcast %astruct.dt68* %.uplevelArgPack0001_401 to i8*, !dbg !36
  %98 = getelementptr i8, i8* %97, i64 96, !dbg !36
  %99 = bitcast i8* %98 to i8**, !dbg !36
  store i8* %96, i8** %99, align 8, !dbg !36
  %100 = bitcast [22 x i64]* %"b$sd1_350" to i8*, !dbg !36
  %101 = bitcast %astruct.dt68* %.uplevelArgPack0001_401 to i8*, !dbg !36
  %102 = getelementptr i8, i8* %101, i64 104, !dbg !36
  %103 = bitcast i8* %102 to i8**, !dbg !36
  store i8* %100, i8** %103, align 8, !dbg !36
  br label %L.LB1_430, !dbg !36

L.LB1_430:                                        ; preds = %L.LB1_381
  %104 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L38_1_ to i64*, !dbg !36
  %105 = bitcast %astruct.dt68* %.uplevelArgPack0001_401 to i64*, !dbg !36
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %104, i64* %105), !dbg !36
  call void (...) @_mp_bcs_nest(), !dbg !37
  %106 = bitcast i32* @.C337_MAIN_ to i8*, !dbg !37
  %107 = bitcast [56 x i8]* @.C335_MAIN_ to i8*, !dbg !37
  %108 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !37
  call void (i8*, i8*, i64, ...) %108(i8* %106, i8* %107, i64 56), !dbg !37
  %109 = bitcast i32* @.C338_MAIN_ to i8*, !dbg !37
  %110 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !37
  %111 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !37
  %112 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !37
  %113 = call i32 (i8*, i8*, i8*, i8*, ...) %112(i8* %109, i8* null, i8* %110, i8* %111), !dbg !37
  call void @llvm.dbg.declare(metadata i32* %z__io_340, metadata !38, metadata !DIExpression()), !dbg !10
  store i32 %113, i32* %z__io_340, align 4, !dbg !37
  %114 = bitcast [9 x i8]* @.C341_MAIN_ to i8*, !dbg !37
  %115 = bitcast i32 (...)* @f90io_sc_ch_ldw to i32 (i8*, i32, i64, ...)*, !dbg !37
  %116 = call i32 (i8*, i32, i64, ...) %115(i8* %114, i32 14, i64 9), !dbg !37
  store i32 %116, i32* %z__io_340, align 4, !dbg !37
  %117 = bitcast [22 x i64]* %"b$sd1_350" to i8*, !dbg !37
  %118 = getelementptr i8, i8* %117, i64 160, !dbg !37
  %119 = bitcast i8* %118 to i64*, !dbg !37
  %120 = load i64, i64* %119, align 8, !dbg !37
  %121 = mul nsw i64 %120, 50, !dbg !37
  %122 = bitcast [22 x i64]* %"b$sd1_350" to i8*, !dbg !37
  %123 = getelementptr i8, i8* %122, i64 56, !dbg !37
  %124 = bitcast i8* %123 to i64*, !dbg !37
  %125 = load i64, i64* %124, align 8, !dbg !37
  %126 = add nsw i64 %121, %125, !dbg !37
  %127 = load float*, float** %.Z0972_326, align 8, !dbg !37
  call void @llvm.dbg.value(metadata float* %127, metadata !20, metadata !DIExpression()), !dbg !10
  %128 = bitcast float* %127 to i8*, !dbg !37
  %129 = getelementptr i8, i8* %128, i64 196, !dbg !37
  %130 = bitcast i8* %129 to float*, !dbg !37
  %131 = getelementptr float, float* %130, i64 %126, !dbg !37
  %132 = load float, float* %131, align 4, !dbg !37
  %133 = bitcast i32 (...)* @f90io_sc_f_ldw to i32 (float, i32, ...)*, !dbg !37
  %134 = call i32 (float, i32, ...) %133(float %132, i32 27), !dbg !37
  store i32 %134, i32* %z__io_340, align 4, !dbg !37
  %135 = call i32 (...) @f90io_ldw_end(), !dbg !37
  store i32 %135, i32* %z__io_340, align 4, !dbg !37
  call void (...) @_mp_ecs_nest(), !dbg !37
  %136 = load float*, float** %.Z0972_326, align 8, !dbg !39
  call void @llvm.dbg.value(metadata float* %136, metadata !20, metadata !DIExpression()), !dbg !10
  %137 = bitcast float* %136 to i8*, !dbg !39
  %138 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !39
  %139 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !39
  call void (i8*, i8*, i8*, i8*, i64, ...) %139(i8* null, i8* %137, i8* %138, i8* null, i64 0), !dbg !39
  %140 = bitcast float** %.Z0972_326 to i8**, !dbg !39
  store i8* null, i8** %140, align 8, !dbg !39
  %141 = bitcast [22 x i64]* %"b$sd1_350" to i64*, !dbg !39
  store i64 0, i64* %141, align 8, !dbg !39
  ret void, !dbg !18
}

define internal void @__nv_MAIN__F1L38_1_(i32* %__nv_MAIN__F1L38_1Arg0, i64* %__nv_MAIN__F1L38_1Arg1, i64* %__nv_MAIN__F1L38_1Arg2) #0 !dbg !40 {
L.entry:
  %__gtid___nv_MAIN__F1L38_1__487 = alloca i32, align 4
  %.i0000p_332 = alloca i32, align 4
  %j_331 = alloca i32, align 4
  %.du0001p_362 = alloca i32, align 4
  %.de0001p_363 = alloca i32, align 4
  %.di0001p_364 = alloca i32, align 4
  %.ds0001p_365 = alloca i32, align 4
  %.dl0001p_367 = alloca i32, align 4
  %.dl0001p.copy_481 = alloca i32, align 4
  %.de0001p.copy_482 = alloca i32, align 4
  %.ds0001p.copy_483 = alloca i32, align 4
  %.dX0001p_366 = alloca i32, align 4
  %.dY0001p_361 = alloca i32, align 4
  %.dY0002p_373 = alloca i32, align 4
  %i_330 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L38_1Arg0, metadata !43, metadata !DIExpression()), !dbg !44
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L38_1Arg1, metadata !45, metadata !DIExpression()), !dbg !44
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L38_1Arg2, metadata !46, metadata !DIExpression()), !dbg !44
  call void @llvm.dbg.value(metadata i32 1, metadata !47, metadata !DIExpression()), !dbg !44
  call void @llvm.dbg.value(metadata i32 2, metadata !48, metadata !DIExpression()), !dbg !44
  call void @llvm.dbg.value(metadata i32 0, metadata !49, metadata !DIExpression()), !dbg !44
  call void @llvm.dbg.value(metadata i32 1, metadata !50, metadata !DIExpression()), !dbg !44
  call void @llvm.dbg.value(metadata i32 2, metadata !51, metadata !DIExpression()), !dbg !44
  call void @llvm.dbg.value(metadata i32 0, metadata !52, metadata !DIExpression()), !dbg !44
  call void @llvm.dbg.value(metadata i32 1, metadata !53, metadata !DIExpression()), !dbg !44
  call void @llvm.dbg.value(metadata i32 2, metadata !54, metadata !DIExpression()), !dbg !44
  %0 = load i32, i32* %__nv_MAIN__F1L38_1Arg0, align 4, !dbg !55
  store i32 %0, i32* %__gtid___nv_MAIN__F1L38_1__487, align 4, !dbg !55
  br label %L.LB2_472

L.LB2_472:                                        ; preds = %L.entry
  br label %L.LB2_329

L.LB2_329:                                        ; preds = %L.LB2_472
  store i32 0, i32* %.i0000p_332, align 4, !dbg !56
  call void @llvm.dbg.declare(metadata i32* %j_331, metadata !57, metadata !DIExpression()), !dbg !55
  store i32 2, i32* %j_331, align 4, !dbg !56
  %1 = bitcast i64* %__nv_MAIN__F1L38_1Arg2 to i32**, !dbg !56
  %2 = load i32*, i32** %1, align 8, !dbg !56
  %3 = load i32, i32* %2, align 4, !dbg !56
  store i32 %3, i32* %.du0001p_362, align 4, !dbg !56
  %4 = bitcast i64* %__nv_MAIN__F1L38_1Arg2 to i32**, !dbg !56
  %5 = load i32*, i32** %4, align 8, !dbg !56
  %6 = load i32, i32* %5, align 4, !dbg !56
  store i32 %6, i32* %.de0001p_363, align 4, !dbg !56
  store i32 1, i32* %.di0001p_364, align 4, !dbg !56
  %7 = load i32, i32* %.di0001p_364, align 4, !dbg !56
  store i32 %7, i32* %.ds0001p_365, align 4, !dbg !56
  store i32 2, i32* %.dl0001p_367, align 4, !dbg !56
  %8 = load i32, i32* %.dl0001p_367, align 4, !dbg !56
  store i32 %8, i32* %.dl0001p.copy_481, align 4, !dbg !56
  %9 = load i32, i32* %.de0001p_363, align 4, !dbg !56
  store i32 %9, i32* %.de0001p.copy_482, align 4, !dbg !56
  %10 = load i32, i32* %.ds0001p_365, align 4, !dbg !56
  store i32 %10, i32* %.ds0001p.copy_483, align 4, !dbg !56
  %11 = load i32, i32* %__gtid___nv_MAIN__F1L38_1__487, align 4, !dbg !56
  %12 = bitcast i32* %.i0000p_332 to i64*, !dbg !56
  %13 = bitcast i32* %.dl0001p.copy_481 to i64*, !dbg !56
  %14 = bitcast i32* %.de0001p.copy_482 to i64*, !dbg !56
  %15 = bitcast i32* %.ds0001p.copy_483 to i64*, !dbg !56
  %16 = load i32, i32* %.ds0001p.copy_483, align 4, !dbg !56
  call void @__kmpc_for_static_init_4(i64* null, i32 %11, i32 34, i64* %12, i64* %13, i64* %14, i64* %15, i32 %16, i32 1), !dbg !56
  %17 = load i32, i32* %.dl0001p.copy_481, align 4, !dbg !56
  store i32 %17, i32* %.dl0001p_367, align 4, !dbg !56
  %18 = load i32, i32* %.de0001p.copy_482, align 4, !dbg !56
  store i32 %18, i32* %.de0001p_363, align 4, !dbg !56
  %19 = load i32, i32* %.ds0001p.copy_483, align 4, !dbg !56
  store i32 %19, i32* %.ds0001p_365, align 4, !dbg !56
  %20 = load i32, i32* %.dl0001p_367, align 4, !dbg !56
  store i32 %20, i32* %j_331, align 4, !dbg !56
  %21 = load i32, i32* %j_331, align 4, !dbg !56
  call void @llvm.dbg.value(metadata i32 %21, metadata !57, metadata !DIExpression()), !dbg !55
  store i32 %21, i32* %.dX0001p_366, align 4, !dbg !56
  %22 = load i32, i32* %.dX0001p_366, align 4, !dbg !56
  %23 = load i32, i32* %.du0001p_362, align 4, !dbg !56
  %24 = icmp sgt i32 %22, %23, !dbg !56
  br i1 %24, label %L.LB2_360, label %L.LB2_519, !dbg !56

L.LB2_519:                                        ; preds = %L.LB2_329
  %25 = load i32, i32* %.dX0001p_366, align 4, !dbg !56
  store i32 %25, i32* %j_331, align 4, !dbg !56
  %26 = load i32, i32* %.di0001p_364, align 4, !dbg !56
  %27 = load i32, i32* %.de0001p_363, align 4, !dbg !56
  %28 = load i32, i32* %.dX0001p_366, align 4, !dbg !56
  %29 = sub nsw i32 %27, %28, !dbg !56
  %30 = add nsw i32 %26, %29, !dbg !56
  %31 = load i32, i32* %.di0001p_364, align 4, !dbg !56
  %32 = sdiv i32 %30, %31, !dbg !56
  store i32 %32, i32* %.dY0001p_361, align 4, !dbg !56
  %33 = load i32, i32* %.dY0001p_361, align 4, !dbg !56
  %34 = icmp sle i32 %33, 0, !dbg !56
  br i1 %34, label %L.LB2_370, label %L.LB2_369, !dbg !56

L.LB2_369:                                        ; preds = %L.LB2_372, %L.LB2_519
  %35 = bitcast i64* %__nv_MAIN__F1L38_1Arg2 to i8*, !dbg !58
  %36 = getelementptr i8, i8* %35, i64 8, !dbg !58
  %37 = bitcast i8* %36 to i32**, !dbg !58
  %38 = load i32*, i32** %37, align 8, !dbg !58
  %39 = load i32, i32* %38, align 4, !dbg !58
  store i32 %39, i32* %.dY0002p_373, align 4, !dbg !58
  call void @llvm.dbg.declare(metadata i32* %i_330, metadata !59, metadata !DIExpression()), !dbg !55
  store i32 1, i32* %i_330, align 4, !dbg !58
  %40 = load i32, i32* %.dY0002p_373, align 4, !dbg !58
  %41 = icmp sle i32 %40, 0, !dbg !58
  br i1 %41, label %L.LB2_372, label %L.LB2_371, !dbg !58

L.LB2_371:                                        ; preds = %L.LB2_371, %L.LB2_369
  %42 = load i32, i32* %i_330, align 4, !dbg !60
  call void @llvm.dbg.value(metadata i32 %42, metadata !59, metadata !DIExpression()), !dbg !55
  %43 = sext i32 %42 to i64, !dbg !60
  %44 = load i32, i32* %j_331, align 4, !dbg !60
  call void @llvm.dbg.value(metadata i32 %44, metadata !57, metadata !DIExpression()), !dbg !55
  %45 = sext i32 %44 to i64, !dbg !60
  %46 = bitcast i64* %__nv_MAIN__F1L38_1Arg2 to i8*, !dbg !60
  %47 = getelementptr i8, i8* %46, i64 104, !dbg !60
  %48 = bitcast i8* %47 to i8**, !dbg !60
  %49 = load i8*, i8** %48, align 8, !dbg !60
  %50 = getelementptr i8, i8* %49, i64 160, !dbg !60
  %51 = bitcast i8* %50 to i64*, !dbg !60
  %52 = load i64, i64* %51, align 8, !dbg !60
  %53 = mul nsw i64 %45, %52, !dbg !60
  %54 = add nsw i64 %43, %53, !dbg !60
  %55 = bitcast i64* %__nv_MAIN__F1L38_1Arg2 to i8*, !dbg !60
  %56 = getelementptr i8, i8* %55, i64 104, !dbg !60
  %57 = bitcast i8* %56 to i8**, !dbg !60
  %58 = load i8*, i8** %57, align 8, !dbg !60
  %59 = getelementptr i8, i8* %58, i64 56, !dbg !60
  %60 = bitcast i8* %59 to i64*, !dbg !60
  %61 = load i64, i64* %60, align 8, !dbg !60
  %62 = add nsw i64 %54, %61, !dbg !60
  %63 = bitcast i64* %__nv_MAIN__F1L38_1Arg2 to i8*, !dbg !60
  %64 = getelementptr i8, i8* %63, i64 24, !dbg !60
  %65 = bitcast i8* %64 to i8***, !dbg !60
  %66 = load i8**, i8*** %65, align 8, !dbg !60
  %67 = load i8*, i8** %66, align 8, !dbg !60
  %68 = getelementptr i8, i8* %67, i64 -8, !dbg !60
  %69 = bitcast i8* %68 to float*, !dbg !60
  %70 = getelementptr float, float* %69, i64 %62, !dbg !60
  %71 = load float, float* %70, align 4, !dbg !60
  %72 = load i32, i32* %i_330, align 4, !dbg !60
  call void @llvm.dbg.value(metadata i32 %72, metadata !59, metadata !DIExpression()), !dbg !55
  %73 = sext i32 %72 to i64, !dbg !60
  %74 = load i32, i32* %j_331, align 4, !dbg !60
  call void @llvm.dbg.value(metadata i32 %74, metadata !57, metadata !DIExpression()), !dbg !55
  %75 = sext i32 %74 to i64, !dbg !60
  %76 = bitcast i64* %__nv_MAIN__F1L38_1Arg2 to i8*, !dbg !60
  %77 = getelementptr i8, i8* %76, i64 104, !dbg !60
  %78 = bitcast i8* %77 to i8**, !dbg !60
  %79 = load i8*, i8** %78, align 8, !dbg !60
  %80 = getelementptr i8, i8* %79, i64 160, !dbg !60
  %81 = bitcast i8* %80 to i64*, !dbg !60
  %82 = load i64, i64* %81, align 8, !dbg !60
  %83 = mul nsw i64 %75, %82, !dbg !60
  %84 = add nsw i64 %73, %83, !dbg !60
  %85 = bitcast i64* %__nv_MAIN__F1L38_1Arg2 to i8*, !dbg !60
  %86 = getelementptr i8, i8* %85, i64 104, !dbg !60
  %87 = bitcast i8* %86 to i8**, !dbg !60
  %88 = load i8*, i8** %87, align 8, !dbg !60
  %89 = getelementptr i8, i8* %88, i64 56, !dbg !60
  %90 = bitcast i8* %89 to i64*, !dbg !60
  %91 = load i64, i64* %90, align 8, !dbg !60
  %92 = add nsw i64 %84, %91, !dbg !60
  %93 = bitcast i64* %__nv_MAIN__F1L38_1Arg2 to i8*, !dbg !60
  %94 = getelementptr i8, i8* %93, i64 24, !dbg !60
  %95 = bitcast i8* %94 to i8***, !dbg !60
  %96 = load i8**, i8*** %95, align 8, !dbg !60
  %97 = load i8*, i8** %96, align 8, !dbg !60
  %98 = getelementptr i8, i8* %97, i64 -4, !dbg !60
  %99 = bitcast i8* %98 to float*, !dbg !60
  %100 = getelementptr float, float* %99, i64 %92, !dbg !60
  store float %71, float* %100, align 4, !dbg !60
  %101 = load i32, i32* %i_330, align 4, !dbg !61
  call void @llvm.dbg.value(metadata i32 %101, metadata !59, metadata !DIExpression()), !dbg !55
  %102 = add nsw i32 %101, 1, !dbg !61
  store i32 %102, i32* %i_330, align 4, !dbg !61
  %103 = load i32, i32* %.dY0002p_373, align 4, !dbg !61
  %104 = sub nsw i32 %103, 1, !dbg !61
  store i32 %104, i32* %.dY0002p_373, align 4, !dbg !61
  %105 = load i32, i32* %.dY0002p_373, align 4, !dbg !61
  %106 = icmp sgt i32 %105, 0, !dbg !61
  br i1 %106, label %L.LB2_371, label %L.LB2_372, !dbg !61

L.LB2_372:                                        ; preds = %L.LB2_371, %L.LB2_369
  %107 = load i32, i32* %.di0001p_364, align 4, !dbg !55
  %108 = load i32, i32* %j_331, align 4, !dbg !55
  call void @llvm.dbg.value(metadata i32 %108, metadata !57, metadata !DIExpression()), !dbg !55
  %109 = add nsw i32 %107, %108, !dbg !55
  store i32 %109, i32* %j_331, align 4, !dbg !55
  %110 = load i32, i32* %.dY0001p_361, align 4, !dbg !55
  %111 = sub nsw i32 %110, 1, !dbg !55
  store i32 %111, i32* %.dY0001p_361, align 4, !dbg !55
  %112 = load i32, i32* %.dY0001p_361, align 4, !dbg !55
  %113 = icmp sgt i32 %112, 0, !dbg !55
  br i1 %113, label %L.LB2_369, label %L.LB2_370, !dbg !55

L.LB2_370:                                        ; preds = %L.LB2_372, %L.LB2_519
  br label %L.LB2_360

L.LB2_360:                                        ; preds = %L.LB2_370, %L.LB2_329
  %114 = load i32, i32* %__gtid___nv_MAIN__F1L38_1__487, align 4, !dbg !55
  call void @__kmpc_for_static_fini(i64* null, i32 %114), !dbg !55
  br label %L.LB2_333

L.LB2_333:                                        ; preds = %L.LB2_360
  ret void, !dbg !55
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
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB014-outofbounds-orig-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb014_outofbounds_orig_yes", scope: !2, file: !3, line: 26, type: !6, scopeLine: 26, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
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
!18 = !DILocation(line: 48, column: 1, scope: !5)
!19 = !DILocation(line: 26, column: 1, scope: !5)
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
!31 = !DILocation(line: 33, column: 1, scope: !5)
!32 = !DILocalVariable(name: "m", scope: !5, file: !3, type: !9)
!33 = !DILocation(line: 34, column: 1, scope: !5)
!34 = !DILocalVariable(scope: !5, file: !3, type: !27, flags: DIFlagArtificial)
!35 = !DILocation(line: 36, column: 1, scope: !5)
!36 = !DILocation(line: 38, column: 1, scope: !5)
!37 = !DILocation(line: 45, column: 1, scope: !5)
!38 = !DILocalVariable(scope: !5, file: !3, type: !9, flags: DIFlagArtificial)
!39 = !DILocation(line: 47, column: 1, scope: !5)
!40 = distinct !DISubprogram(name: "__nv_MAIN__F1L38_1", scope: !2, file: !3, line: 38, type: !41, scopeLine: 38, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!41 = !DISubroutineType(types: !42)
!42 = !{null, !9, !27, !27}
!43 = !DILocalVariable(name: "__nv_MAIN__F1L38_1Arg0", arg: 1, scope: !40, file: !3, type: !9)
!44 = !DILocation(line: 0, scope: !40)
!45 = !DILocalVariable(name: "__nv_MAIN__F1L38_1Arg1", arg: 2, scope: !40, file: !3, type: !27)
!46 = !DILocalVariable(name: "__nv_MAIN__F1L38_1Arg2", arg: 3, scope: !40, file: !3, type: !27)
!47 = !DILocalVariable(name: "omp_sched_static", scope: !40, file: !3, type: !9)
!48 = !DILocalVariable(name: "omp_sched_dynamic", scope: !40, file: !3, type: !9)
!49 = !DILocalVariable(name: "omp_proc_bind_false", scope: !40, file: !3, type: !9)
!50 = !DILocalVariable(name: "omp_proc_bind_true", scope: !40, file: !3, type: !9)
!51 = !DILocalVariable(name: "omp_proc_bind_master", scope: !40, file: !3, type: !9)
!52 = !DILocalVariable(name: "omp_lock_hint_none", scope: !40, file: !3, type: !9)
!53 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !40, file: !3, type: !9)
!54 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !40, file: !3, type: !9)
!55 = !DILocation(line: 43, column: 1, scope: !40)
!56 = !DILocation(line: 39, column: 1, scope: !40)
!57 = !DILocalVariable(name: "j", scope: !40, file: !3, type: !9)
!58 = !DILocation(line: 40, column: 1, scope: !40)
!59 = !DILocalVariable(name: "i", scope: !40, file: !3, type: !9)
!60 = !DILocation(line: 41, column: 1, scope: !40)
!61 = !DILocation(line: 42, column: 1, scope: !40)

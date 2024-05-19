; ModuleID = '/tmp/DRB066-pointernoaliasing-orig-no-38b7b8.ll'
source_filename = "/tmp/DRB066-pointernoaliasing-orig-no-38b7b8.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%astruct.dt88 = type <{ i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8* }>

@.C342_drb066_setup_ = internal constant float 2.500000e+00
@.C291_drb066_setup_ = internal constant double 0.000000e+00
@.C285_drb066_setup_ = internal constant i32 1
@.C347_drb066_setup_ = internal constant i64 9
@.C305_drb066_setup_ = internal constant i32 28
@.C317_drb066_setup_ = internal constant i64 8
@.C345_drb066_setup_ = internal constant i64 28
@.C316_drb066_setup_ = internal constant i64 12
@.C315_drb066_setup_ = internal constant i64 11
@.C283_drb066_setup_ = internal constant i32 0
@.C286_drb066_setup_ = internal constant i64 1
@.C284_drb066_setup_ = internal constant i64 0
@.C342___nv_drb066_setup__F1L29_1 = internal constant float 2.500000e+00
@.C291___nv_drb066_setup__F1L29_1 = internal constant double 0.000000e+00
@.C285___nv_drb066_setup__F1L29_1 = internal constant i32 1
@.C283___nv_drb066_setup__F1L29_1 = internal constant i32 0
@.C309_MAIN_ = internal constant i32 1000
@.C283_MAIN_ = internal constant i32 0

; Function Attrs: noinline
define float @drb066_() #0 {
.L.entry:
  ret float undef
}

define void @drb066_setup_(i32 %_V_n.arg) #1 !dbg !5 {
L.entry:
  %_V_n.addr = alloca i32, align 4
  %n_306 = alloca i32, align 4
  %__gtid_drb066_setup__466 = alloca i32, align 4
  %.Z0998_336 = alloca double*, align 8
  %"tar2$sd2_350" = alloca [16 x i64], align 8
  %.Z0997_335 = alloca double*, align 8
  %"tar1$sd1_349" = alloca [16 x i64], align 8
  %"m_nvol$p_320" = alloca double*, align 8
  %"m_nvol$sd_319" = alloca [16 x i64], align 8
  %"m_pdv_sum$p_313" = alloca double*, align 8
  %"m_pdv_sum$sd_312" = alloca [16 x i64], align 8
  %.g0000_403 = alloca i64, align 8
  %z_b_8_322 = alloca i64, align 8
  %z_b_9_323 = alloca i64, align 8
  %z_e_98_326 = alloca i64, align 8
  %z_b_10_324 = alloca i64, align 8
  %z_b_11_325 = alloca i64, align 8
  %z_b_12_329 = alloca i64, align 8
  %z_b_13_330 = alloca i64, align 8
  %z_e_105_333 = alloca i64, align 8
  %z_b_14_331 = alloca i64, align 8
  %z_b_15_332 = alloca i64, align 8
  %.uplevelArgPack0001_429 = alloca %astruct.dt88, align 16
  %"drb066_setup___$eq_307" = alloca [304 x i8], align 4
  call void @llvm.dbg.declare(metadata i32* %_V_n.addr, metadata !10, metadata !DIExpression()), !dbg !11
  store i32 %_V_n.arg, i32* %_V_n.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %_V_n.addr, metadata !12, metadata !DIExpression()), !dbg !11
  call void @llvm.dbg.value(metadata i32 1, metadata !13, metadata !DIExpression()), !dbg !11
  call void @llvm.dbg.value(metadata i32 0, metadata !14, metadata !DIExpression()), !dbg !11
  call void @llvm.dbg.value(metadata i32 1, metadata !15, metadata !DIExpression()), !dbg !11
  call void @llvm.dbg.value(metadata i32 0, metadata !16, metadata !DIExpression()), !dbg !11
  call void @llvm.dbg.value(metadata i32 1, metadata !17, metadata !DIExpression()), !dbg !11
  call void @llvm.dbg.value(metadata i32 8, metadata !18, metadata !DIExpression()), !dbg !11
  %0 = load i32, i32* %_V_n.addr, align 4, !dbg !19
  call void @llvm.dbg.value(metadata i32 %0, metadata !10, metadata !DIExpression()), !dbg !11
  call void @llvm.dbg.declare(metadata i32* %n_306, metadata !20, metadata !DIExpression()), !dbg !11
  store i32 %0, i32* %n_306, align 4, !dbg !19
  %1 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !21
  store i32 %1, i32* %__gtid_drb066_setup__466, align 4, !dbg !21
  call void @llvm.dbg.declare(metadata double** %.Z0998_336, metadata !22, metadata !DIExpression(DW_OP_deref)), !dbg !11
  %2 = bitcast double** %.Z0998_336 to i8**, !dbg !19
  store i8* null, i8** %2, align 8, !dbg !19
  call void @llvm.dbg.declare(metadata [16 x i64]* %"tar2$sd2_350", metadata !27, metadata !DIExpression()), !dbg !11
  %3 = bitcast [16 x i64]* %"tar2$sd2_350" to i64*, !dbg !19
  store i64 0, i64* %3, align 8, !dbg !19
  call void @llvm.dbg.declare(metadata double** %.Z0997_335, metadata !32, metadata !DIExpression(DW_OP_deref)), !dbg !11
  %4 = bitcast double** %.Z0997_335 to i8**, !dbg !19
  store i8* null, i8** %4, align 8, !dbg !19
  call void @llvm.dbg.declare(metadata [16 x i64]* %"tar1$sd1_349", metadata !27, metadata !DIExpression()), !dbg !11
  %5 = bitcast [16 x i64]* %"tar1$sd1_349" to i64*, !dbg !19
  store i64 0, i64* %5, align 8, !dbg !19
  call void @llvm.dbg.declare(metadata double** %"m_nvol$p_320", metadata !33, metadata !DIExpression(DW_OP_deref)), !dbg !11
  %6 = bitcast double** %"m_nvol$p_320" to i8**, !dbg !19
  store i8* null, i8** %6, align 8, !dbg !19
  call void @llvm.dbg.declare(metadata [16 x i64]* %"m_nvol$sd_319", metadata !27, metadata !DIExpression()), !dbg !11
  %7 = bitcast [16 x i64]* %"m_nvol$sd_319" to i64*, !dbg !19
  store i64 0, i64* %7, align 8, !dbg !19
  call void @llvm.dbg.declare(metadata double** %"m_pdv_sum$p_313", metadata !34, metadata !DIExpression(DW_OP_deref)), !dbg !11
  %8 = bitcast double** %"m_pdv_sum$p_313" to i8**, !dbg !19
  store i8* null, i8** %8, align 8, !dbg !19
  call void @llvm.dbg.declare(metadata [16 x i64]* %"m_pdv_sum$sd_312", metadata !27, metadata !DIExpression()), !dbg !11
  %9 = bitcast [16 x i64]* %"m_pdv_sum$sd_312" to i64*, !dbg !19
  store i64 0, i64* %9, align 8, !dbg !19
  br label %L.LB2_387

L.LB2_387:                                        ; preds = %L.entry
  %10 = bitcast [16 x i64]* %"m_pdv_sum$sd_312" to i8*, !dbg !35
  %11 = getelementptr i8, i8* %10, i64 80, !dbg !35
  %12 = bitcast i8* %11 to i64*, !dbg !35
  store i64 1, i64* %12, align 8, !dbg !35
  %13 = load i32, i32* %n_306, align 4, !dbg !35
  call void @llvm.dbg.value(metadata i32 %13, metadata !20, metadata !DIExpression()), !dbg !11
  %14 = sext i32 %13 to i64, !dbg !35
  %15 = bitcast [16 x i64]* %"m_pdv_sum$sd_312" to i8*, !dbg !35
  %16 = getelementptr i8, i8* %15, i64 88, !dbg !35
  %17 = bitcast i8* %16 to i64*, !dbg !35
  store i64 %14, i64* %17, align 8, !dbg !35
  %18 = bitcast [16 x i64]* %"m_pdv_sum$sd_312" to i8*, !dbg !35
  %19 = getelementptr i8, i8* %18, i64 88, !dbg !35
  %20 = bitcast i8* %19 to i64*, !dbg !35
  %21 = load i64, i64* %20, align 8, !dbg !35
  %22 = sub nsw i64 %21, 1, !dbg !35
  %23 = bitcast [16 x i64]* %"m_pdv_sum$sd_312" to i8*, !dbg !35
  %24 = getelementptr i8, i8* %23, i64 80, !dbg !35
  %25 = bitcast i8* %24 to i64*, !dbg !35
  %26 = load i64, i64* %25, align 8, !dbg !35
  %27 = add nsw i64 %22, %26, !dbg !35
  store i64 %27, i64* %.g0000_403, align 8, !dbg !35
  %28 = bitcast [16 x i64]* %"m_pdv_sum$sd_312" to i8*, !dbg !35
  %29 = bitcast i64* @.C284_drb066_setup_ to i8*, !dbg !35
  %30 = bitcast i64* @.C345_drb066_setup_ to i8*, !dbg !35
  %31 = bitcast i64* @.C317_drb066_setup_ to i8*, !dbg !35
  %32 = bitcast [16 x i64]* %"m_pdv_sum$sd_312" to i8*, !dbg !35
  %33 = getelementptr i8, i8* %32, i64 80, !dbg !35
  %34 = bitcast i64* %.g0000_403 to i8*, !dbg !35
  %35 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !35
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %35(i8* %28, i8* %29, i8* %30, i8* %31, i8* %33, i8* %34), !dbg !35
  %36 = bitcast [16 x i64]* %"m_pdv_sum$sd_312" to i8*, !dbg !35
  %37 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !35
  call void (i8*, i32, ...) %37(i8* %36, i32 28), !dbg !35
  %38 = bitcast [16 x i64]* %"m_pdv_sum$sd_312" to i8*, !dbg !35
  %39 = getelementptr i8, i8* %38, i64 88, !dbg !35
  %40 = bitcast i8* %39 to i64*, !dbg !35
  %41 = load i64, i64* %40, align 8, !dbg !35
  %42 = sub nsw i64 %41, 1, !dbg !35
  %43 = bitcast [16 x i64]* %"m_pdv_sum$sd_312" to i8*, !dbg !35
  %44 = getelementptr i8, i8* %43, i64 80, !dbg !35
  %45 = bitcast i8* %44 to i64*, !dbg !35
  %46 = load i64, i64* %45, align 8, !dbg !35
  %47 = add nsw i64 %42, %46, !dbg !35
  %48 = bitcast [16 x i64]* %"m_pdv_sum$sd_312" to i8*, !dbg !35
  %49 = getelementptr i8, i8* %48, i64 80, !dbg !35
  %50 = bitcast i8* %49 to i64*, !dbg !35
  %51 = load i64, i64* %50, align 8, !dbg !35
  %52 = sub nsw i64 %51, 1, !dbg !35
  %53 = sub nsw i64 %47, %52, !dbg !35
  store i64 %53, i64* %.g0000_403, align 8, !dbg !35
  %54 = bitcast i64* %.g0000_403 to i8*, !dbg !35
  %55 = bitcast i64* @.C345_drb066_setup_ to i8*, !dbg !35
  %56 = bitcast i64* @.C317_drb066_setup_ to i8*, !dbg !35
  %57 = bitcast double** %"m_pdv_sum$p_313" to i8*, !dbg !35
  %58 = bitcast i64* @.C286_drb066_setup_ to i8*, !dbg !35
  %59 = bitcast i64* @.C284_drb066_setup_ to i8*, !dbg !35
  %60 = bitcast void (...)* @f90_alloc04a_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !35
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %60(i8* %54, i8* %55, i8* %56, i8* null, i8* %57, i8* null, i8* %58, i8* %59, i8* null, i64 0), !dbg !35
  %61 = bitcast [16 x i64]* %"m_pdv_sum$sd_312" to i8*, !dbg !35
  %62 = getelementptr i8, i8* %61, i64 64, !dbg !35
  %63 = bitcast double** %"m_pdv_sum$p_313" to i8*, !dbg !35
  %64 = bitcast void (...)* @f90_ptrcp to void (i8*, i8*, ...)*, !dbg !35
  call void (i8*, i8*, ...) %64(i8* %62, i8* %63), !dbg !35
  %65 = bitcast [16 x i64]* %"m_nvol$sd_319" to i8*, !dbg !36
  %66 = getelementptr i8, i8* %65, i64 80, !dbg !36
  %67 = bitcast i8* %66 to i64*, !dbg !36
  store i64 1, i64* %67, align 8, !dbg !36
  %68 = load i32, i32* %n_306, align 4, !dbg !36
  call void @llvm.dbg.value(metadata i32 %68, metadata !20, metadata !DIExpression()), !dbg !11
  %69 = sext i32 %68 to i64, !dbg !36
  %70 = bitcast [16 x i64]* %"m_nvol$sd_319" to i8*, !dbg !36
  %71 = getelementptr i8, i8* %70, i64 88, !dbg !36
  %72 = bitcast i8* %71 to i64*, !dbg !36
  store i64 %69, i64* %72, align 8, !dbg !36
  %73 = bitcast [16 x i64]* %"m_nvol$sd_319" to i8*, !dbg !36
  %74 = getelementptr i8, i8* %73, i64 88, !dbg !36
  %75 = bitcast i8* %74 to i64*, !dbg !36
  %76 = load i64, i64* %75, align 8, !dbg !36
  %77 = sub nsw i64 %76, 1, !dbg !36
  %78 = bitcast [16 x i64]* %"m_nvol$sd_319" to i8*, !dbg !36
  %79 = getelementptr i8, i8* %78, i64 80, !dbg !36
  %80 = bitcast i8* %79 to i64*, !dbg !36
  %81 = load i64, i64* %80, align 8, !dbg !36
  %82 = add nsw i64 %77, %81, !dbg !36
  store i64 %82, i64* %.g0000_403, align 8, !dbg !36
  %83 = bitcast [16 x i64]* %"m_nvol$sd_319" to i8*, !dbg !36
  %84 = bitcast i64* @.C284_drb066_setup_ to i8*, !dbg !36
  %85 = bitcast i64* @.C345_drb066_setup_ to i8*, !dbg !36
  %86 = bitcast i64* @.C317_drb066_setup_ to i8*, !dbg !36
  %87 = bitcast [16 x i64]* %"m_nvol$sd_319" to i8*, !dbg !36
  %88 = getelementptr i8, i8* %87, i64 80, !dbg !36
  %89 = bitcast i64* %.g0000_403 to i8*, !dbg !36
  %90 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !36
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %90(i8* %83, i8* %84, i8* %85, i8* %86, i8* %88, i8* %89), !dbg !36
  %91 = bitcast [16 x i64]* %"m_nvol$sd_319" to i8*, !dbg !36
  %92 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !36
  call void (i8*, i32, ...) %92(i8* %91, i32 28), !dbg !36
  %93 = bitcast [16 x i64]* %"m_nvol$sd_319" to i8*, !dbg !36
  %94 = getelementptr i8, i8* %93, i64 88, !dbg !36
  %95 = bitcast i8* %94 to i64*, !dbg !36
  %96 = load i64, i64* %95, align 8, !dbg !36
  %97 = sub nsw i64 %96, 1, !dbg !36
  %98 = bitcast [16 x i64]* %"m_nvol$sd_319" to i8*, !dbg !36
  %99 = getelementptr i8, i8* %98, i64 80, !dbg !36
  %100 = bitcast i8* %99 to i64*, !dbg !36
  %101 = load i64, i64* %100, align 8, !dbg !36
  %102 = add nsw i64 %97, %101, !dbg !36
  %103 = bitcast [16 x i64]* %"m_nvol$sd_319" to i8*, !dbg !36
  %104 = getelementptr i8, i8* %103, i64 80, !dbg !36
  %105 = bitcast i8* %104 to i64*, !dbg !36
  %106 = load i64, i64* %105, align 8, !dbg !36
  %107 = sub nsw i64 %106, 1, !dbg !36
  %108 = sub nsw i64 %102, %107, !dbg !36
  store i64 %108, i64* %.g0000_403, align 8, !dbg !36
  %109 = bitcast i64* %.g0000_403 to i8*, !dbg !36
  %110 = bitcast i64* @.C345_drb066_setup_ to i8*, !dbg !36
  %111 = bitcast i64* @.C317_drb066_setup_ to i8*, !dbg !36
  %112 = bitcast double** %"m_nvol$p_320" to i8*, !dbg !36
  %113 = bitcast i64* @.C286_drb066_setup_ to i8*, !dbg !36
  %114 = bitcast i64* @.C284_drb066_setup_ to i8*, !dbg !36
  %115 = bitcast void (...)* @f90_alloc04a_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !36
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %115(i8* %109, i8* %110, i8* %111, i8* null, i8* %112, i8* null, i8* %113, i8* %114, i8* null, i64 0), !dbg !36
  %116 = bitcast [16 x i64]* %"m_nvol$sd_319" to i8*, !dbg !36
  %117 = getelementptr i8, i8* %116, i64 64, !dbg !36
  %118 = bitcast double** %"m_nvol$p_320" to i8*, !dbg !36
  %119 = bitcast void (...)* @f90_ptrcp to void (i8*, i8*, ...)*, !dbg !36
  call void (i8*, i8*, ...) %119(i8* %117, i8* %118), !dbg !36
  call void @llvm.dbg.declare(metadata i64* %z_b_8_322, metadata !37, metadata !DIExpression()), !dbg !11
  store i64 1, i64* %z_b_8_322, align 8, !dbg !38
  %120 = load i32, i32* %n_306, align 4, !dbg !38
  call void @llvm.dbg.value(metadata i32 %120, metadata !20, metadata !DIExpression()), !dbg !11
  %121 = sext i32 %120 to i64, !dbg !38
  call void @llvm.dbg.declare(metadata i64* %z_b_9_323, metadata !37, metadata !DIExpression()), !dbg !11
  store i64 %121, i64* %z_b_9_323, align 8, !dbg !38
  %122 = load i64, i64* %z_b_9_323, align 8, !dbg !38
  call void @llvm.dbg.value(metadata i64 %122, metadata !37, metadata !DIExpression()), !dbg !11
  call void @llvm.dbg.declare(metadata i64* %z_e_98_326, metadata !37, metadata !DIExpression()), !dbg !11
  store i64 %122, i64* %z_e_98_326, align 8, !dbg !38
  %123 = bitcast [16 x i64]* %"tar1$sd1_349" to i8*, !dbg !38
  %124 = bitcast i64* @.C284_drb066_setup_ to i8*, !dbg !38
  %125 = bitcast i64* @.C345_drb066_setup_ to i8*, !dbg !38
  %126 = bitcast i64* @.C317_drb066_setup_ to i8*, !dbg !38
  %127 = bitcast i64* %z_b_8_322 to i8*, !dbg !38
  %128 = bitcast i64* %z_b_9_323 to i8*, !dbg !38
  %129 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !38
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %129(i8* %123, i8* %124, i8* %125, i8* %126, i8* %127, i8* %128), !dbg !38
  %130 = bitcast [16 x i64]* %"tar1$sd1_349" to i8*, !dbg !38
  %131 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !38
  call void (i8*, i32, ...) %131(i8* %130, i32 28), !dbg !38
  %132 = load i64, i64* %z_b_9_323, align 8, !dbg !38
  call void @llvm.dbg.value(metadata i64 %132, metadata !37, metadata !DIExpression()), !dbg !11
  %133 = load i64, i64* %z_b_8_322, align 8, !dbg !38
  call void @llvm.dbg.value(metadata i64 %133, metadata !37, metadata !DIExpression()), !dbg !11
  %134 = sub nsw i64 %133, 1, !dbg !38
  %135 = sub nsw i64 %132, %134, !dbg !38
  call void @llvm.dbg.declare(metadata i64* %z_b_10_324, metadata !37, metadata !DIExpression()), !dbg !11
  store i64 %135, i64* %z_b_10_324, align 8, !dbg !38
  %136 = load i64, i64* %z_b_8_322, align 8, !dbg !38
  call void @llvm.dbg.value(metadata i64 %136, metadata !37, metadata !DIExpression()), !dbg !11
  call void @llvm.dbg.declare(metadata i64* %z_b_11_325, metadata !37, metadata !DIExpression()), !dbg !11
  store i64 %136, i64* %z_b_11_325, align 8, !dbg !38
  %137 = bitcast i64* %z_b_10_324 to i8*, !dbg !38
  %138 = bitcast i64* @.C345_drb066_setup_ to i8*, !dbg !38
  %139 = bitcast i64* @.C317_drb066_setup_ to i8*, !dbg !38
  %140 = bitcast double** %.Z0997_335 to i8*, !dbg !38
  %141 = bitcast i64* @.C286_drb066_setup_ to i8*, !dbg !38
  %142 = bitcast i64* @.C284_drb066_setup_ to i8*, !dbg !38
  %143 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !38
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %143(i8* %137, i8* %138, i8* %139, i8* null, i8* %140, i8* null, i8* %141, i8* %142, i8* null, i64 0), !dbg !38
  call void @llvm.dbg.declare(metadata i64* %z_b_12_329, metadata !37, metadata !DIExpression()), !dbg !11
  store i64 1, i64* %z_b_12_329, align 8, !dbg !39
  %144 = load i32, i32* %n_306, align 4, !dbg !39
  call void @llvm.dbg.value(metadata i32 %144, metadata !20, metadata !DIExpression()), !dbg !11
  %145 = sext i32 %144 to i64, !dbg !39
  call void @llvm.dbg.declare(metadata i64* %z_b_13_330, metadata !37, metadata !DIExpression()), !dbg !11
  store i64 %145, i64* %z_b_13_330, align 8, !dbg !39
  %146 = load i64, i64* %z_b_13_330, align 8, !dbg !39
  call void @llvm.dbg.value(metadata i64 %146, metadata !37, metadata !DIExpression()), !dbg !11
  call void @llvm.dbg.declare(metadata i64* %z_e_105_333, metadata !37, metadata !DIExpression()), !dbg !11
  store i64 %146, i64* %z_e_105_333, align 8, !dbg !39
  %147 = bitcast [16 x i64]* %"tar2$sd2_350" to i8*, !dbg !39
  %148 = bitcast i64* @.C284_drb066_setup_ to i8*, !dbg !39
  %149 = bitcast i64* @.C345_drb066_setup_ to i8*, !dbg !39
  %150 = bitcast i64* @.C317_drb066_setup_ to i8*, !dbg !39
  %151 = bitcast i64* %z_b_12_329 to i8*, !dbg !39
  %152 = bitcast i64* %z_b_13_330 to i8*, !dbg !39
  %153 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !39
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %153(i8* %147, i8* %148, i8* %149, i8* %150, i8* %151, i8* %152), !dbg !39
  %154 = bitcast [16 x i64]* %"tar2$sd2_350" to i8*, !dbg !39
  %155 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !39
  call void (i8*, i32, ...) %155(i8* %154, i32 28), !dbg !39
  %156 = load i64, i64* %z_b_13_330, align 8, !dbg !39
  call void @llvm.dbg.value(metadata i64 %156, metadata !37, metadata !DIExpression()), !dbg !11
  %157 = load i64, i64* %z_b_12_329, align 8, !dbg !39
  call void @llvm.dbg.value(metadata i64 %157, metadata !37, metadata !DIExpression()), !dbg !11
  %158 = sub nsw i64 %157, 1, !dbg !39
  %159 = sub nsw i64 %156, %158, !dbg !39
  call void @llvm.dbg.declare(metadata i64* %z_b_14_331, metadata !37, metadata !DIExpression()), !dbg !11
  store i64 %159, i64* %z_b_14_331, align 8, !dbg !39
  %160 = load i64, i64* %z_b_12_329, align 8, !dbg !39
  call void @llvm.dbg.value(metadata i64 %160, metadata !37, metadata !DIExpression()), !dbg !11
  call void @llvm.dbg.declare(metadata i64* %z_b_15_332, metadata !37, metadata !DIExpression()), !dbg !11
  store i64 %160, i64* %z_b_15_332, align 8, !dbg !39
  %161 = bitcast i64* %z_b_14_331 to i8*, !dbg !39
  %162 = bitcast i64* @.C345_drb066_setup_ to i8*, !dbg !39
  %163 = bitcast i64* @.C317_drb066_setup_ to i8*, !dbg !39
  %164 = bitcast double** %.Z0998_336 to i8*, !dbg !39
  %165 = bitcast i64* @.C286_drb066_setup_ to i8*, !dbg !39
  %166 = bitcast i64* @.C284_drb066_setup_ to i8*, !dbg !39
  %167 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !39
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %167(i8* %161, i8* %162, i8* %163, i8* null, i8* %164, i8* null, i8* %165, i8* %166, i8* null, i64 0), !dbg !39
  %168 = load double*, double** %"m_pdv_sum$p_313", align 8, !dbg !40
  call void @llvm.dbg.value(metadata double* %168, metadata !34, metadata !DIExpression()), !dbg !11
  %169 = bitcast double* %168 to i8*, !dbg !40
  %170 = bitcast [16 x i64]* %"m_pdv_sum$sd_312" to i8*, !dbg !40
  %171 = load double*, double** %.Z0997_335, align 8, !dbg !40
  call void @llvm.dbg.value(metadata double* %171, metadata !32, metadata !DIExpression()), !dbg !11
  %172 = bitcast double* %171 to i8*, !dbg !40
  %173 = bitcast [16 x i64]* %"tar1$sd1_349" to i8*, !dbg !40
  %174 = bitcast i64* @.C284_drb066_setup_ to i8*, !dbg !40
  %175 = bitcast i64 (...)* @fort_ptr_assn_i8 to i64 (i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !40
  %176 = call i64 (i8*, i8*, i8*, i8*, i8*, ...) %175(i8* %169, i8* %170, i8* %172, i8* %173, i8* %174), !dbg !40
  %177 = inttoptr i64 %176 to i8*, !dbg !40
  %178 = bitcast double** %"m_pdv_sum$p_313" to i8**, !dbg !40
  store i8* %177, i8** %178, align 8, !dbg !40
  %179 = load double*, double** %"m_nvol$p_320", align 8, !dbg !41
  call void @llvm.dbg.value(metadata double* %179, metadata !33, metadata !DIExpression()), !dbg !11
  %180 = bitcast double* %179 to i8*, !dbg !41
  %181 = bitcast [16 x i64]* %"m_nvol$sd_319" to i8*, !dbg !41
  %182 = load double*, double** %.Z0998_336, align 8, !dbg !41
  call void @llvm.dbg.value(metadata double* %182, metadata !22, metadata !DIExpression()), !dbg !11
  %183 = bitcast double* %182 to i8*, !dbg !41
  %184 = bitcast [16 x i64]* %"tar2$sd2_350" to i8*, !dbg !41
  %185 = bitcast i64* @.C284_drb066_setup_ to i8*, !dbg !41
  %186 = bitcast i64 (...)* @fort_ptr_assn_i8 to i64 (i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !41
  %187 = call i64 (i8*, i8*, i8*, i8*, i8*, ...) %186(i8* %180, i8* %181, i8* %183, i8* %184, i8* %185), !dbg !41
  %188 = inttoptr i64 %187 to i8*, !dbg !41
  %189 = bitcast double** %"m_nvol$p_320" to i8**, !dbg !41
  store i8* %188, i8** %189, align 8, !dbg !41
  %190 = bitcast i32* %n_306 to i8*, !dbg !42
  %191 = bitcast %astruct.dt88* %.uplevelArgPack0001_429 to i8**, !dbg !42
  store i8* %190, i8** %191, align 8, !dbg !42
  %192 = bitcast double** %.Z0997_335 to i8*, !dbg !42
  %193 = bitcast %astruct.dt88* %.uplevelArgPack0001_429 to i8*, !dbg !42
  %194 = getelementptr i8, i8* %193, i64 8, !dbg !42
  %195 = bitcast i8* %194 to i8**, !dbg !42
  store i8* %192, i8** %195, align 8, !dbg !42
  %196 = bitcast double** %.Z0997_335 to i8*, !dbg !42
  %197 = bitcast %astruct.dt88* %.uplevelArgPack0001_429 to i8*, !dbg !42
  %198 = getelementptr i8, i8* %197, i64 16, !dbg !42
  %199 = bitcast i8* %198 to i8**, !dbg !42
  store i8* %196, i8** %199, align 8, !dbg !42
  %200 = bitcast i64* %z_b_8_322 to i8*, !dbg !42
  %201 = bitcast %astruct.dt88* %.uplevelArgPack0001_429 to i8*, !dbg !42
  %202 = getelementptr i8, i8* %201, i64 24, !dbg !42
  %203 = bitcast i8* %202 to i8**, !dbg !42
  store i8* %200, i8** %203, align 8, !dbg !42
  %204 = bitcast i64* %z_b_9_323 to i8*, !dbg !42
  %205 = bitcast %astruct.dt88* %.uplevelArgPack0001_429 to i8*, !dbg !42
  %206 = getelementptr i8, i8* %205, i64 32, !dbg !42
  %207 = bitcast i8* %206 to i8**, !dbg !42
  store i8* %204, i8** %207, align 8, !dbg !42
  %208 = bitcast i64* %z_e_98_326 to i8*, !dbg !42
  %209 = bitcast %astruct.dt88* %.uplevelArgPack0001_429 to i8*, !dbg !42
  %210 = getelementptr i8, i8* %209, i64 40, !dbg !42
  %211 = bitcast i8* %210 to i8**, !dbg !42
  store i8* %208, i8** %211, align 8, !dbg !42
  %212 = bitcast i64* %z_b_10_324 to i8*, !dbg !42
  %213 = bitcast %astruct.dt88* %.uplevelArgPack0001_429 to i8*, !dbg !42
  %214 = getelementptr i8, i8* %213, i64 48, !dbg !42
  %215 = bitcast i8* %214 to i8**, !dbg !42
  store i8* %212, i8** %215, align 8, !dbg !42
  %216 = bitcast i64* %z_b_11_325 to i8*, !dbg !42
  %217 = bitcast %astruct.dt88* %.uplevelArgPack0001_429 to i8*, !dbg !42
  %218 = getelementptr i8, i8* %217, i64 56, !dbg !42
  %219 = bitcast i8* %218 to i8**, !dbg !42
  store i8* %216, i8** %219, align 8, !dbg !42
  %220 = bitcast double** %.Z0998_336 to i8*, !dbg !42
  %221 = bitcast %astruct.dt88* %.uplevelArgPack0001_429 to i8*, !dbg !42
  %222 = getelementptr i8, i8* %221, i64 64, !dbg !42
  %223 = bitcast i8* %222 to i8**, !dbg !42
  store i8* %220, i8** %223, align 8, !dbg !42
  %224 = bitcast double** %.Z0998_336 to i8*, !dbg !42
  %225 = bitcast %astruct.dt88* %.uplevelArgPack0001_429 to i8*, !dbg !42
  %226 = getelementptr i8, i8* %225, i64 72, !dbg !42
  %227 = bitcast i8* %226 to i8**, !dbg !42
  store i8* %224, i8** %227, align 8, !dbg !42
  %228 = bitcast i64* %z_b_12_329 to i8*, !dbg !42
  %229 = bitcast %astruct.dt88* %.uplevelArgPack0001_429 to i8*, !dbg !42
  %230 = getelementptr i8, i8* %229, i64 80, !dbg !42
  %231 = bitcast i8* %230 to i8**, !dbg !42
  store i8* %228, i8** %231, align 8, !dbg !42
  %232 = bitcast i64* %z_b_13_330 to i8*, !dbg !42
  %233 = bitcast %astruct.dt88* %.uplevelArgPack0001_429 to i8*, !dbg !42
  %234 = getelementptr i8, i8* %233, i64 88, !dbg !42
  %235 = bitcast i8* %234 to i8**, !dbg !42
  store i8* %232, i8** %235, align 8, !dbg !42
  %236 = bitcast i64* %z_e_105_333 to i8*, !dbg !42
  %237 = bitcast %astruct.dt88* %.uplevelArgPack0001_429 to i8*, !dbg !42
  %238 = getelementptr i8, i8* %237, i64 96, !dbg !42
  %239 = bitcast i8* %238 to i8**, !dbg !42
  store i8* %236, i8** %239, align 8, !dbg !42
  %240 = bitcast i64* %z_b_14_331 to i8*, !dbg !42
  %241 = bitcast %astruct.dt88* %.uplevelArgPack0001_429 to i8*, !dbg !42
  %242 = getelementptr i8, i8* %241, i64 104, !dbg !42
  %243 = bitcast i8* %242 to i8**, !dbg !42
  store i8* %240, i8** %243, align 8, !dbg !42
  %244 = bitcast i64* %z_b_15_332 to i8*, !dbg !42
  %245 = bitcast %astruct.dt88* %.uplevelArgPack0001_429 to i8*, !dbg !42
  %246 = getelementptr i8, i8* %245, i64 112, !dbg !42
  %247 = bitcast i8* %246 to i8**, !dbg !42
  store i8* %244, i8** %247, align 8, !dbg !42
  %248 = bitcast [16 x i64]* %"tar1$sd1_349" to i8*, !dbg !42
  %249 = bitcast %astruct.dt88* %.uplevelArgPack0001_429 to i8*, !dbg !42
  %250 = getelementptr i8, i8* %249, i64 120, !dbg !42
  %251 = bitcast i8* %250 to i8**, !dbg !42
  store i8* %248, i8** %251, align 8, !dbg !42
  %252 = bitcast [16 x i64]* %"tar2$sd2_350" to i8*, !dbg !42
  %253 = bitcast %astruct.dt88* %.uplevelArgPack0001_429 to i8*, !dbg !42
  %254 = getelementptr i8, i8* %253, i64 128, !dbg !42
  %255 = bitcast i8* %254 to i8**, !dbg !42
  store i8* %252, i8** %255, align 8, !dbg !42
  br label %L.LB2_464, !dbg !42

L.LB2_464:                                        ; preds = %L.LB2_387
  %256 = bitcast void (i32*, i64*, i64*)* @__nv_drb066_setup__F1L29_1_ to i64*, !dbg !42
  %257 = bitcast %astruct.dt88* %.uplevelArgPack0001_429 to i64*, !dbg !42
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %256, i64* %257), !dbg !42
  %258 = load double*, double** %"m_pdv_sum$p_313", align 8, !dbg !43
  call void @llvm.dbg.value(metadata double* %258, metadata !34, metadata !DIExpression()), !dbg !11
  %259 = bitcast double* %258 to i8*, !dbg !43
  %260 = icmp eq i8* %259, null, !dbg !43
  br i1 %260, label %L.LB2_368, label %L.LB2_484, !dbg !43

L.LB2_484:                                        ; preds = %L.LB2_464
  %261 = bitcast double** %"m_pdv_sum$p_313" to i8**, !dbg !43
  store i8* null, i8** %261, align 8, !dbg !43
  %262 = bitcast [16 x i64]* %"m_pdv_sum$sd_312" to i64*, !dbg !43
  store i64 0, i64* %262, align 8, !dbg !43
  br label %L.LB2_368

L.LB2_368:                                        ; preds = %L.LB2_484, %L.LB2_464
  %263 = load double*, double** %"m_nvol$p_320", align 8, !dbg !44
  call void @llvm.dbg.value(metadata double* %263, metadata !33, metadata !DIExpression()), !dbg !11
  %264 = bitcast double* %263 to i8*, !dbg !44
  %265 = icmp eq i8* %264, null, !dbg !44
  br i1 %265, label %L.LB2_369, label %L.LB2_485, !dbg !44

L.LB2_485:                                        ; preds = %L.LB2_368
  %266 = bitcast double** %"m_nvol$p_320" to i8**, !dbg !44
  store i8* null, i8** %266, align 8, !dbg !44
  %267 = bitcast [16 x i64]* %"m_nvol$sd_319" to i64*, !dbg !44
  store i64 0, i64* %267, align 8, !dbg !44
  br label %L.LB2_369

L.LB2_369:                                        ; preds = %L.LB2_485, %L.LB2_368
  %268 = load double*, double** %.Z0997_335, align 8, !dbg !45
  call void @llvm.dbg.value(metadata double* %268, metadata !32, metadata !DIExpression()), !dbg !11
  %269 = bitcast double* %268 to i8*, !dbg !45
  %270 = bitcast i64* @.C286_drb066_setup_ to i8*, !dbg !45
  %271 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !45
  call void (i8*, i8*, i8*, i8*, i64, ...) %271(i8* null, i8* %269, i8* %270, i8* null, i64 0), !dbg !45
  %272 = bitcast double** %.Z0997_335 to i8**, !dbg !45
  store i8* null, i8** %272, align 8, !dbg !45
  %273 = bitcast [16 x i64]* %"tar1$sd1_349" to i64*, !dbg !45
  store i64 0, i64* %273, align 8, !dbg !45
  %274 = load double*, double** %.Z0998_336, align 8, !dbg !45
  call void @llvm.dbg.value(metadata double* %274, metadata !22, metadata !DIExpression()), !dbg !11
  %275 = bitcast double* %274 to i8*, !dbg !45
  %276 = bitcast i64* @.C284_drb066_setup_ to i8*, !dbg !45
  %277 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !45
  call void (i8*, i8*, i8*, i8*, i64, ...) %277(i8* null, i8* %275, i8* %276, i8* null, i64 0), !dbg !45
  %278 = bitcast double** %.Z0998_336 to i8**, !dbg !45
  store i8* null, i8** %278, align 8, !dbg !45
  %279 = bitcast [16 x i64]* %"tar2$sd2_350" to i64*, !dbg !45
  store i64 0, i64* %279, align 8, !dbg !45
  %280 = load double*, double** %.Z0997_335, align 8, !dbg !21
  call void @llvm.dbg.value(metadata double* %280, metadata !32, metadata !DIExpression()), !dbg !11
  %281 = bitcast double* %280 to i8*, !dbg !21
  %282 = bitcast i32 (...)* @f90_allocated_i8 to i32 (i8*, ...)*, !dbg !21
  %283 = call i32 (i8*, ...) %282(i8* %281), !dbg !21
  %284 = and i32 %283, 1, !dbg !21
  %285 = icmp eq i32 %284, 0, !dbg !21
  br i1 %285, label %L.LB2_371, label %L.LB2_486, !dbg !21

L.LB2_486:                                        ; preds = %L.LB2_369
  %286 = load double*, double** %.Z0997_335, align 8, !dbg !21
  call void @llvm.dbg.value(metadata double* %286, metadata !32, metadata !DIExpression()), !dbg !11
  %287 = bitcast double* %286 to i8*, !dbg !21
  %288 = bitcast i64* @.C284_drb066_setup_ to i8*, !dbg !21
  %289 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !21
  call void (i8*, i8*, i8*, i8*, i64, ...) %289(i8* null, i8* %287, i8* %288, i8* null, i64 0), !dbg !21
  %290 = bitcast double** %.Z0997_335 to i8**, !dbg !21
  store i8* null, i8** %290, align 8, !dbg !21
  %291 = bitcast [16 x i64]* %"tar1$sd1_349" to i64*, !dbg !21
  store i64 0, i64* %291, align 8, !dbg !21
  br label %L.LB2_371

L.LB2_371:                                        ; preds = %L.LB2_486, %L.LB2_369
  %292 = load double*, double** %.Z0998_336, align 8, !dbg !21
  call void @llvm.dbg.value(metadata double* %292, metadata !22, metadata !DIExpression()), !dbg !11
  %293 = bitcast double* %292 to i8*, !dbg !21
  %294 = bitcast i32 (...)* @f90_allocated_i8 to i32 (i8*, ...)*, !dbg !21
  %295 = call i32 (i8*, ...) %294(i8* %293), !dbg !21
  %296 = and i32 %295, 1, !dbg !21
  %297 = icmp eq i32 %296, 0, !dbg !21
  br i1 %297, label %L.LB2_373, label %L.LB2_487, !dbg !21

L.LB2_487:                                        ; preds = %L.LB2_371
  %298 = load double*, double** %.Z0998_336, align 8, !dbg !21
  call void @llvm.dbg.value(metadata double* %298, metadata !22, metadata !DIExpression()), !dbg !11
  %299 = bitcast double* %298 to i8*, !dbg !21
  %300 = bitcast i64* @.C284_drb066_setup_ to i8*, !dbg !21
  %301 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !21
  call void (i8*, i8*, i8*, i8*, i64, ...) %301(i8* null, i8* %299, i8* %300, i8* null, i64 0), !dbg !21
  %302 = bitcast double** %.Z0998_336 to i8**, !dbg !21
  store i8* null, i8** %302, align 8, !dbg !21
  %303 = bitcast [16 x i64]* %"tar2$sd2_350" to i64*, !dbg !21
  store i64 0, i64* %303, align 8, !dbg !21
  br label %L.LB2_373

L.LB2_373:                                        ; preds = %L.LB2_487, %L.LB2_371
  ret void, !dbg !21
}

define internal void @__nv_drb066_setup__F1L29_1_(i32* %__nv_drb066_setup__F1L29_1Arg0, i64* %__nv_drb066_setup__F1L29_1Arg1, i64* %__nv_drb066_setup__F1L29_1Arg2) #1 !dbg !46 {
L.entry:
  %__gtid___nv_drb066_setup__F1L29_1__506 = alloca i32, align 4
  %.i0000p_341 = alloca i32, align 4
  %i_340 = alloca i32, align 4
  %.du0001p_359 = alloca i32, align 4
  %.de0001p_360 = alloca i32, align 4
  %.di0001p_361 = alloca i32, align 4
  %.ds0001p_362 = alloca i32, align 4
  %.dl0001p_364 = alloca i32, align 4
  %.dl0001p.copy_500 = alloca i32, align 4
  %.de0001p.copy_501 = alloca i32, align 4
  %.ds0001p.copy_502 = alloca i32, align 4
  %.dX0001p_363 = alloca i32, align 4
  %.dY0001p_358 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_drb066_setup__F1L29_1Arg0, metadata !49, metadata !DIExpression()), !dbg !50
  call void @llvm.dbg.declare(metadata i64* %__nv_drb066_setup__F1L29_1Arg1, metadata !51, metadata !DIExpression()), !dbg !50
  call void @llvm.dbg.declare(metadata i64* %__nv_drb066_setup__F1L29_1Arg2, metadata !52, metadata !DIExpression()), !dbg !50
  call void @llvm.dbg.value(metadata i32 1, metadata !53, metadata !DIExpression()), !dbg !50
  call void @llvm.dbg.value(metadata i32 0, metadata !54, metadata !DIExpression()), !dbg !50
  call void @llvm.dbg.value(metadata i32 1, metadata !55, metadata !DIExpression()), !dbg !50
  call void @llvm.dbg.value(metadata i32 0, metadata !56, metadata !DIExpression()), !dbg !50
  call void @llvm.dbg.value(metadata i32 1, metadata !57, metadata !DIExpression()), !dbg !50
  call void @llvm.dbg.value(metadata i32 8, metadata !58, metadata !DIExpression()), !dbg !50
  %0 = load i32, i32* %__nv_drb066_setup__F1L29_1Arg0, align 4, !dbg !59
  store i32 %0, i32* %__gtid___nv_drb066_setup__F1L29_1__506, align 4, !dbg !59
  br label %L.LB3_491

L.LB3_491:                                        ; preds = %L.entry
  br label %L.LB3_339

L.LB3_339:                                        ; preds = %L.LB3_491
  store i32 0, i32* %.i0000p_341, align 4, !dbg !60
  call void @llvm.dbg.declare(metadata i32* %i_340, metadata !61, metadata !DIExpression()), !dbg !59
  store i32 1, i32* %i_340, align 4, !dbg !60
  %1 = bitcast i64* %__nv_drb066_setup__F1L29_1Arg2 to i32**, !dbg !60
  %2 = load i32*, i32** %1, align 8, !dbg !60
  %3 = load i32, i32* %2, align 4, !dbg !60
  store i32 %3, i32* %.du0001p_359, align 4, !dbg !60
  %4 = bitcast i64* %__nv_drb066_setup__F1L29_1Arg2 to i32**, !dbg !60
  %5 = load i32*, i32** %4, align 8, !dbg !60
  %6 = load i32, i32* %5, align 4, !dbg !60
  store i32 %6, i32* %.de0001p_360, align 4, !dbg !60
  store i32 1, i32* %.di0001p_361, align 4, !dbg !60
  %7 = load i32, i32* %.di0001p_361, align 4, !dbg !60
  store i32 %7, i32* %.ds0001p_362, align 4, !dbg !60
  store i32 1, i32* %.dl0001p_364, align 4, !dbg !60
  %8 = load i32, i32* %.dl0001p_364, align 4, !dbg !60
  store i32 %8, i32* %.dl0001p.copy_500, align 4, !dbg !60
  %9 = load i32, i32* %.de0001p_360, align 4, !dbg !60
  store i32 %9, i32* %.de0001p.copy_501, align 4, !dbg !60
  %10 = load i32, i32* %.ds0001p_362, align 4, !dbg !60
  store i32 %10, i32* %.ds0001p.copy_502, align 4, !dbg !60
  %11 = load i32, i32* %__gtid___nv_drb066_setup__F1L29_1__506, align 4, !dbg !60
  %12 = bitcast i32* %.i0000p_341 to i64*, !dbg !60
  %13 = bitcast i32* %.dl0001p.copy_500 to i64*, !dbg !60
  %14 = bitcast i32* %.de0001p.copy_501 to i64*, !dbg !60
  %15 = bitcast i32* %.ds0001p.copy_502 to i64*, !dbg !60
  %16 = load i32, i32* %.ds0001p.copy_502, align 4, !dbg !60
  call void @__kmpc_for_static_init_4(i64* null, i32 %11, i32 34, i64* %12, i64* %13, i64* %14, i64* %15, i32 %16, i32 1), !dbg !60
  %17 = load i32, i32* %.dl0001p.copy_500, align 4, !dbg !60
  store i32 %17, i32* %.dl0001p_364, align 4, !dbg !60
  %18 = load i32, i32* %.de0001p.copy_501, align 4, !dbg !60
  store i32 %18, i32* %.de0001p_360, align 4, !dbg !60
  %19 = load i32, i32* %.ds0001p.copy_502, align 4, !dbg !60
  store i32 %19, i32* %.ds0001p_362, align 4, !dbg !60
  %20 = load i32, i32* %.dl0001p_364, align 4, !dbg !60
  store i32 %20, i32* %i_340, align 4, !dbg !60
  %21 = load i32, i32* %i_340, align 4, !dbg !60
  call void @llvm.dbg.value(metadata i32 %21, metadata !61, metadata !DIExpression()), !dbg !59
  store i32 %21, i32* %.dX0001p_363, align 4, !dbg !60
  %22 = load i32, i32* %.dX0001p_363, align 4, !dbg !60
  %23 = load i32, i32* %.du0001p_359, align 4, !dbg !60
  %24 = icmp sgt i32 %22, %23, !dbg !60
  br i1 %24, label %L.LB3_357, label %L.LB3_535, !dbg !60

L.LB3_535:                                        ; preds = %L.LB3_339
  %25 = load i32, i32* %.dX0001p_363, align 4, !dbg !60
  store i32 %25, i32* %i_340, align 4, !dbg !60
  %26 = load i32, i32* %.di0001p_361, align 4, !dbg !60
  %27 = load i32, i32* %.de0001p_360, align 4, !dbg !60
  %28 = load i32, i32* %.dX0001p_363, align 4, !dbg !60
  %29 = sub nsw i32 %27, %28, !dbg !60
  %30 = add nsw i32 %26, %29, !dbg !60
  %31 = load i32, i32* %.di0001p_361, align 4, !dbg !60
  %32 = sdiv i32 %30, %31, !dbg !60
  store i32 %32, i32* %.dY0001p_358, align 4, !dbg !60
  %33 = load i32, i32* %.dY0001p_358, align 4, !dbg !60
  %34 = icmp sle i32 %33, 0, !dbg !60
  br i1 %34, label %L.LB3_367, label %L.LB3_366, !dbg !60

L.LB3_366:                                        ; preds = %L.LB3_366, %L.LB3_535
  %35 = load i32, i32* %i_340, align 4, !dbg !62
  call void @llvm.dbg.value(metadata i32 %35, metadata !61, metadata !DIExpression()), !dbg !59
  %36 = sext i32 %35 to i64, !dbg !62
  %37 = bitcast i64* %__nv_drb066_setup__F1L29_1Arg2 to i8*, !dbg !62
  %38 = getelementptr i8, i8* %37, i64 120, !dbg !62
  %39 = bitcast i8* %38 to i8**, !dbg !62
  %40 = load i8*, i8** %39, align 8, !dbg !62
  %41 = getelementptr i8, i8* %40, i64 56, !dbg !62
  %42 = bitcast i8* %41 to i64*, !dbg !62
  %43 = load i64, i64* %42, align 8, !dbg !62
  %44 = add nsw i64 %36, %43, !dbg !62
  %45 = bitcast i64* %__nv_drb066_setup__F1L29_1Arg2 to i8*, !dbg !62
  %46 = getelementptr i8, i8* %45, i64 16, !dbg !62
  %47 = bitcast i8* %46 to i8***, !dbg !62
  %48 = load i8**, i8*** %47, align 8, !dbg !62
  %49 = load i8*, i8** %48, align 8, !dbg !62
  %50 = getelementptr i8, i8* %49, i64 -8, !dbg !62
  %51 = bitcast i8* %50 to double*, !dbg !62
  %52 = getelementptr double, double* %51, i64 %44, !dbg !62
  store double 0.000000e+00, double* %52, align 8, !dbg !62
  %53 = load i32, i32* %i_340, align 4, !dbg !63
  call void @llvm.dbg.value(metadata i32 %53, metadata !61, metadata !DIExpression()), !dbg !59
  %54 = sitofp i32 %53 to float, !dbg !63
  %55 = fmul fast float %54, 2.500000e+00, !dbg !63
  %56 = fpext float %55 to double, !dbg !63
  %57 = load i32, i32* %i_340, align 4, !dbg !63
  call void @llvm.dbg.value(metadata i32 %57, metadata !61, metadata !DIExpression()), !dbg !59
  %58 = sext i32 %57 to i64, !dbg !63
  %59 = bitcast i64* %__nv_drb066_setup__F1L29_1Arg2 to i8*, !dbg !63
  %60 = getelementptr i8, i8* %59, i64 128, !dbg !63
  %61 = bitcast i8* %60 to i8**, !dbg !63
  %62 = load i8*, i8** %61, align 8, !dbg !63
  %63 = getelementptr i8, i8* %62, i64 56, !dbg !63
  %64 = bitcast i8* %63 to i64*, !dbg !63
  %65 = load i64, i64* %64, align 8, !dbg !63
  %66 = add nsw i64 %58, %65, !dbg !63
  %67 = bitcast i64* %__nv_drb066_setup__F1L29_1Arg2 to i8*, !dbg !63
  %68 = getelementptr i8, i8* %67, i64 72, !dbg !63
  %69 = bitcast i8* %68 to i8***, !dbg !63
  %70 = load i8**, i8*** %69, align 8, !dbg !63
  %71 = load i8*, i8** %70, align 8, !dbg !63
  %72 = getelementptr i8, i8* %71, i64 -8, !dbg !63
  %73 = bitcast i8* %72 to double*, !dbg !63
  %74 = getelementptr double, double* %73, i64 %66, !dbg !63
  store double %56, double* %74, align 8, !dbg !63
  %75 = load i32, i32* %.di0001p_361, align 4, !dbg !59
  %76 = load i32, i32* %i_340, align 4, !dbg !59
  call void @llvm.dbg.value(metadata i32 %76, metadata !61, metadata !DIExpression()), !dbg !59
  %77 = add nsw i32 %75, %76, !dbg !59
  store i32 %77, i32* %i_340, align 4, !dbg !59
  %78 = load i32, i32* %.dY0001p_358, align 4, !dbg !59
  %79 = sub nsw i32 %78, 1, !dbg !59
  store i32 %79, i32* %.dY0001p_358, align 4, !dbg !59
  %80 = load i32, i32* %.dY0001p_358, align 4, !dbg !59
  %81 = icmp sgt i32 %80, 0, !dbg !59
  br i1 %81, label %L.LB3_366, label %L.LB3_367, !dbg !59

L.LB3_367:                                        ; preds = %L.LB3_366, %L.LB3_535
  br label %L.LB3_357

L.LB3_357:                                        ; preds = %L.LB3_367, %L.LB3_339
  %82 = load i32, i32* %__gtid___nv_drb066_setup__F1L29_1__506, align 4, !dbg !59
  call void @__kmpc_for_static_fini(i64* null, i32 %82), !dbg !59
  br label %L.LB3_343

L.LB3_343:                                        ; preds = %L.LB3_357
  ret void, !dbg !59
}

define void @MAIN_() #1 !dbg !64 {
L.entry:
  %n_308 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !67, metadata !DIExpression()), !dbg !68
  call void @llvm.dbg.value(metadata i32 0, metadata !69, metadata !DIExpression()), !dbg !68
  call void @llvm.dbg.value(metadata i32 1, metadata !70, metadata !DIExpression()), !dbg !68
  call void @llvm.dbg.value(metadata i32 0, metadata !71, metadata !DIExpression()), !dbg !68
  call void @llvm.dbg.value(metadata i32 1, metadata !72, metadata !DIExpression()), !dbg !68
  %0 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !73
  %1 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !73
  call void (i8*, ...) %1(i8* %0), !dbg !73
  br label %L.LB5_314

L.LB5_314:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %n_308, metadata !74, metadata !DIExpression()), !dbg !68
  store i32 1000, i32* %n_308, align 4, !dbg !75
  %2 = load i32, i32* %n_308, align 4, !dbg !76
  call void @llvm.dbg.value(metadata i32 %2, metadata !74, metadata !DIExpression()), !dbg !68
  call void @drb066_setup_(i32 %2), !dbg !76
  ret void, !dbg !77
}

declare void @fort_init(...) #1

declare void @__kmpc_for_static_fini(i64*, i32) #1

declare void @__kmpc_for_static_init_4(i64*, i32, i32, i64*, i64*, i64*, i64*, i32, i32) #1

declare signext i32 @f90_allocated_i8(...) #1

declare void @f90_dealloc03a_i8(...) #1

declare i64 @fort_ptr_assn_i8(...) #1

declare void @f90_alloc04_chka_i8(...) #1

declare void @f90_ptrcp(...) #1

declare void @f90_alloc04a_i8(...) #1

declare void @f90_set_intrin_type_i8(...) #1

declare void @f90_template1_i8(...) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

declare signext i32 @__kmpc_global_thread_num(i64*) #1

declare void @__kmpc_fork_call(i64*, i32, i64*, i64*, ...) #1

declare void @__kmpc_end_serialized_parallel(i64*, i32) #1

declare void @__kmpc_serialized_parallel(i64*, i32) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { noinline }
attributes #1 = { "no-frame-pointer-elim-non-leaf" }
attributes #2 = { nounwind readnone speculatable }

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !4, globals: !4, imports: !4)
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB066-pointernoaliasing-orig-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "setup", scope: !6, file: !3, line: 14, type: !7, scopeLine: 14, spFlags: DISPFlagDefinition, unit: !2)
!6 = !DIModule(scope: !2, name: "drb066")
!7 = !DISubroutineType(types: !8)
!8 = !{null, !9}
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DILocalVariable(name: "_V_n", scope: !5, file: !3, type: !9)
!11 = !DILocation(line: 0, scope: !5)
!12 = !DILocalVariable(name: "_V_n", arg: 1, scope: !5, file: !3, type: !9)
!13 = !DILocalVariable(name: "omp_sched_static", scope: !5, file: !3, type: !9)
!14 = !DILocalVariable(name: "omp_proc_bind_false", scope: !5, file: !3, type: !9)
!15 = !DILocalVariable(name: "omp_proc_bind_true", scope: !5, file: !3, type: !9)
!16 = !DILocalVariable(name: "omp_lock_hint_none", scope: !5, file: !3, type: !9)
!17 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !5, file: !3, type: !9)
!18 = !DILocalVariable(name: "dp", scope: !5, file: !3, type: !9)
!19 = !DILocation(line: 14, column: 1, scope: !5)
!20 = !DILocalVariable(name: "n", scope: !5, file: !3, type: !9)
!21 = !DILocation(line: 40, column: 1, scope: !5)
!22 = !DILocalVariable(name: "tar2", scope: !5, file: !3, type: !23)
!23 = !DICompositeType(tag: DW_TAG_array_type, baseType: !24, size: 64, align: 64, elements: !25)
!24 = !DIBasicType(name: "double precision", size: 64, align: 64, encoding: DW_ATE_float)
!25 = !{!26}
!26 = !DISubrange(count: 0, lowerBound: 1)
!27 = !DILocalVariable(scope: !5, file: !3, type: !28, flags: DIFlagArtificial)
!28 = !DICompositeType(tag: DW_TAG_array_type, baseType: !29, size: 1024, align: 64, elements: !30)
!29 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!30 = !{!31}
!31 = !DISubrange(count: 16, lowerBound: 1)
!32 = !DILocalVariable(name: "tar1", scope: !5, file: !3, type: !23)
!33 = !DILocalVariable(name: "m_nvol", scope: !5, file: !3, type: !23)
!34 = !DILocalVariable(name: "m_pdv_sum", scope: !5, file: !3, type: !23)
!35 = !DILocation(line: 21, column: 1, scope: !5)
!36 = !DILocation(line: 22, column: 1, scope: !5)
!37 = !DILocalVariable(scope: !5, file: !3, type: !29, flags: DIFlagArtificial)
!38 = !DILocation(line: 23, column: 1, scope: !5)
!39 = !DILocation(line: 24, column: 1, scope: !5)
!40 = !DILocation(line: 26, column: 1, scope: !5)
!41 = !DILocation(line: 27, column: 1, scope: !5)
!42 = !DILocation(line: 29, column: 1, scope: !5)
!43 = !DILocation(line: 37, column: 1, scope: !5)
!44 = !DILocation(line: 38, column: 1, scope: !5)
!45 = !DILocation(line: 39, column: 1, scope: !5)
!46 = distinct !DISubprogram(name: "__nv_drb066_setup__F1L29_1", scope: !2, file: !3, line: 29, type: !47, scopeLine: 29, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!47 = !DISubroutineType(types: !48)
!48 = !{null, !9, !29, !29}
!49 = !DILocalVariable(name: "__nv_drb066_setup__F1L29_1Arg0", arg: 1, scope: !46, file: !3, type: !9)
!50 = !DILocation(line: 0, scope: !46)
!51 = !DILocalVariable(name: "__nv_drb066_setup__F1L29_1Arg1", arg: 2, scope: !46, file: !3, type: !29)
!52 = !DILocalVariable(name: "__nv_drb066_setup__F1L29_1Arg2", arg: 3, scope: !46, file: !3, type: !29)
!53 = !DILocalVariable(name: "omp_sched_static", scope: !46, file: !3, type: !9)
!54 = !DILocalVariable(name: "omp_proc_bind_false", scope: !46, file: !3, type: !9)
!55 = !DILocalVariable(name: "omp_proc_bind_true", scope: !46, file: !3, type: !9)
!56 = !DILocalVariable(name: "omp_lock_hint_none", scope: !46, file: !3, type: !9)
!57 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !46, file: !3, type: !9)
!58 = !DILocalVariable(name: "dp", scope: !46, file: !3, type: !9)
!59 = !DILocation(line: 33, column: 1, scope: !46)
!60 = !DILocation(line: 30, column: 1, scope: !46)
!61 = !DILocalVariable(name: "i", scope: !46, file: !3, type: !9)
!62 = !DILocation(line: 31, column: 1, scope: !46)
!63 = !DILocation(line: 32, column: 1, scope: !46)
!64 = distinct !DISubprogram(name: "drb066_pointernoaliasing_orig_no", scope: !2, file: !3, line: 43, type: !65, scopeLine: 43, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!65 = !DISubroutineType(cc: DW_CC_program, types: !66)
!66 = !{null}
!67 = !DILocalVariable(name: "omp_sched_static", scope: !64, file: !3, type: !9)
!68 = !DILocation(line: 0, scope: !64)
!69 = !DILocalVariable(name: "omp_proc_bind_false", scope: !64, file: !3, type: !9)
!70 = !DILocalVariable(name: "omp_proc_bind_true", scope: !64, file: !3, type: !9)
!71 = !DILocalVariable(name: "omp_lock_hint_none", scope: !64, file: !3, type: !9)
!72 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !64, file: !3, type: !9)
!73 = !DILocation(line: 43, column: 1, scope: !64)
!74 = !DILocalVariable(name: "n", scope: !64, file: !3, type: !9)
!75 = !DILocation(line: 49, column: 1, scope: !64)
!76 = !DILocation(line: 51, column: 1, scope: !64)
!77 = !DILocation(line: 53, column: 1, scope: !64)

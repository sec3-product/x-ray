; ModuleID = '/tmp/DRB037-truedepseconddimension-orig-yes-5c2afe.ll'
source_filename = "/tmp/DRB037-truedepseconddimension-orig-yes-5c2afe.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.STATICS1 = type <{ [48 x i8] }>
%astruct.dt68 = type <{ i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8* }>

@.STATICS1 = internal global %struct.STATICS1 <{ [48 x i8] c"\FB\FF\FF\FF\0C\00\00\00b(500,500) =\EA\FF\FF\FF\00\00\00\00\14\00\00\00\00\00\00\00\06\00\00\00\FF\FF\FF\FF\00\00\00\00" }>, align 16
@.C341_MAIN_ = internal constant i64 500
@.C338_MAIN_ = internal constant i32 6
@.C334_MAIN_ = internal constant [67 x i8] c"micro-benchmarks-fortran/DRB037-truedepseconddimension-orig-yes.f95"
@.C336_MAIN_ = internal constant i32 34
@.C300_MAIN_ = internal constant i32 2
@.C285_MAIN_ = internal constant i32 1
@.C309_MAIN_ = internal constant i32 27
@.C351_MAIN_ = internal constant i64 4
@.C350_MAIN_ = internal constant i64 27
@.C324_MAIN_ = internal constant i32 1000
@.C286_MAIN_ = internal constant i64 1
@.C284_MAIN_ = internal constant i64 0
@.C283_MAIN_ = internal constant i32 0
@.C285___nv_MAIN__F1L27_1 = internal constant i32 1
@.C300___nv_MAIN__F1L27_1 = internal constant i32 2
@.C283___nv_MAIN__F1L27_1 = internal constant i32 0

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__434 = alloca i32, align 4
  %.Z0972_326 = alloca float*, align 8
  %"b$sd1_349" = alloca [22 x i64], align 8
  %len_325 = alloca i32, align 4
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
  %.dY0001_360 = alloca i32, align 4
  %i_310 = alloca i32, align 4
  %.uplevelArgPack0001_403 = alloca %astruct.dt68, align 16
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
  store i32 %0, i32* %__gtid_MAIN__434, align 4, !dbg !18
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !19
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !19
  call void (i8*, ...) %2(i8* %1), !dbg !19
  call void @llvm.dbg.declare(metadata float** %.Z0972_326, metadata !20, metadata !DIExpression(DW_OP_deref)), !dbg !10
  %3 = bitcast float** %.Z0972_326 to i8**, !dbg !19
  store i8* null, i8** %3, align 8, !dbg !19
  call void @llvm.dbg.declare(metadata [22 x i64]* %"b$sd1_349", metadata !25, metadata !DIExpression()), !dbg !10
  %4 = bitcast [22 x i64]* %"b$sd1_349" to i64*, !dbg !19
  store i64 0, i64* %4, align 8, !dbg !19
  br label %L.LB1_380

L.LB1_380:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %len_325, metadata !30, metadata !DIExpression()), !dbg !10
  store i32 1000, i32* %len_325, align 4, !dbg !31
  %5 = load i32, i32* %len_325, align 4, !dbg !32
  call void @llvm.dbg.value(metadata i32 %5, metadata !30, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i32* %n_312, metadata !33, metadata !DIExpression()), !dbg !10
  store i32 %5, i32* %n_312, align 4, !dbg !32
  %6 = load i32, i32* %len_325, align 4, !dbg !34
  call void @llvm.dbg.value(metadata i32 %6, metadata !30, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i32* %m_313, metadata !35, metadata !DIExpression()), !dbg !10
  store i32 %6, i32* %m_313, align 4, !dbg !34
  call void @llvm.dbg.declare(metadata i64* %z_b_0_314, metadata !36, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_0_314, align 8, !dbg !37
  %7 = load i32, i32* %len_325, align 4, !dbg !37
  call void @llvm.dbg.value(metadata i32 %7, metadata !30, metadata !DIExpression()), !dbg !10
  %8 = sext i32 %7 to i64, !dbg !37
  call void @llvm.dbg.declare(metadata i64* %z_b_1_315, metadata !36, metadata !DIExpression()), !dbg !10
  store i64 %8, i64* %z_b_1_315, align 8, !dbg !37
  %9 = load i64, i64* %z_b_1_315, align 8, !dbg !37
  call void @llvm.dbg.value(metadata i64 %9, metadata !36, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_63_321, metadata !36, metadata !DIExpression()), !dbg !10
  store i64 %9, i64* %z_e_63_321, align 8, !dbg !37
  call void @llvm.dbg.declare(metadata i64* %z_b_3_317, metadata !36, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_3_317, align 8, !dbg !37
  %10 = load i32, i32* %len_325, align 4, !dbg !37
  call void @llvm.dbg.value(metadata i32 %10, metadata !30, metadata !DIExpression()), !dbg !10
  %11 = sext i32 %10 to i64, !dbg !37
  call void @llvm.dbg.declare(metadata i64* %z_b_4_318, metadata !36, metadata !DIExpression()), !dbg !10
  store i64 %11, i64* %z_b_4_318, align 8, !dbg !37
  %12 = load i64, i64* %z_b_4_318, align 8, !dbg !37
  call void @llvm.dbg.value(metadata i64 %12, metadata !36, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_66_322, metadata !36, metadata !DIExpression()), !dbg !10
  store i64 %12, i64* %z_e_66_322, align 8, !dbg !37
  %13 = bitcast [22 x i64]* %"b$sd1_349" to i8*, !dbg !37
  %14 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !37
  %15 = bitcast i64* @.C350_MAIN_ to i8*, !dbg !37
  %16 = bitcast i64* @.C351_MAIN_ to i8*, !dbg !37
  %17 = bitcast i64* %z_b_0_314 to i8*, !dbg !37
  %18 = bitcast i64* %z_b_1_315 to i8*, !dbg !37
  %19 = bitcast i64* %z_b_3_317 to i8*, !dbg !37
  %20 = bitcast i64* %z_b_4_318 to i8*, !dbg !37
  %21 = bitcast void (...)* @f90_template2_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !37
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, ...) %21(i8* %13, i8* %14, i8* %15, i8* %16, i8* %17, i8* %18, i8* %19, i8* %20), !dbg !37
  %22 = bitcast [22 x i64]* %"b$sd1_349" to i8*, !dbg !37
  %23 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !37
  call void (i8*, i32, ...) %23(i8* %22, i32 27), !dbg !37
  %24 = load i64, i64* %z_b_1_315, align 8, !dbg !37
  call void @llvm.dbg.value(metadata i64 %24, metadata !36, metadata !DIExpression()), !dbg !10
  %25 = load i64, i64* %z_b_0_314, align 8, !dbg !37
  call void @llvm.dbg.value(metadata i64 %25, metadata !36, metadata !DIExpression()), !dbg !10
  %26 = sub nsw i64 %25, 1, !dbg !37
  %27 = sub nsw i64 %24, %26, !dbg !37
  call void @llvm.dbg.declare(metadata i64* %z_b_2_316, metadata !36, metadata !DIExpression()), !dbg !10
  store i64 %27, i64* %z_b_2_316, align 8, !dbg !37
  %28 = load i64, i64* %z_b_1_315, align 8, !dbg !37
  call void @llvm.dbg.value(metadata i64 %28, metadata !36, metadata !DIExpression()), !dbg !10
  %29 = load i64, i64* %z_b_0_314, align 8, !dbg !37
  call void @llvm.dbg.value(metadata i64 %29, metadata !36, metadata !DIExpression()), !dbg !10
  %30 = sub nsw i64 %29, 1, !dbg !37
  %31 = sub nsw i64 %28, %30, !dbg !37
  %32 = load i64, i64* %z_b_4_318, align 8, !dbg !37
  call void @llvm.dbg.value(metadata i64 %32, metadata !36, metadata !DIExpression()), !dbg !10
  %33 = load i64, i64* %z_b_3_317, align 8, !dbg !37
  call void @llvm.dbg.value(metadata i64 %33, metadata !36, metadata !DIExpression()), !dbg !10
  %34 = sub nsw i64 %33, 1, !dbg !37
  %35 = sub nsw i64 %32, %34, !dbg !37
  %36 = mul nsw i64 %31, %35, !dbg !37
  call void @llvm.dbg.declare(metadata i64* %z_b_5_319, metadata !36, metadata !DIExpression()), !dbg !10
  store i64 %36, i64* %z_b_5_319, align 8, !dbg !37
  %37 = load i64, i64* %z_b_0_314, align 8, !dbg !37
  call void @llvm.dbg.value(metadata i64 %37, metadata !36, metadata !DIExpression()), !dbg !10
  %38 = load i64, i64* %z_b_1_315, align 8, !dbg !37
  call void @llvm.dbg.value(metadata i64 %38, metadata !36, metadata !DIExpression()), !dbg !10
  %39 = load i64, i64* %z_b_0_314, align 8, !dbg !37
  call void @llvm.dbg.value(metadata i64 %39, metadata !36, metadata !DIExpression()), !dbg !10
  %40 = sub nsw i64 %39, 1, !dbg !37
  %41 = sub nsw i64 %38, %40, !dbg !37
  %42 = load i64, i64* %z_b_3_317, align 8, !dbg !37
  call void @llvm.dbg.value(metadata i64 %42, metadata !36, metadata !DIExpression()), !dbg !10
  %43 = mul nsw i64 %41, %42, !dbg !37
  %44 = add nsw i64 %37, %43, !dbg !37
  call void @llvm.dbg.declare(metadata i64* %z_b_6_320, metadata !36, metadata !DIExpression()), !dbg !10
  store i64 %44, i64* %z_b_6_320, align 8, !dbg !37
  %45 = bitcast i64* %z_b_5_319 to i8*, !dbg !37
  %46 = bitcast i64* @.C350_MAIN_ to i8*, !dbg !37
  %47 = bitcast i64* @.C351_MAIN_ to i8*, !dbg !37
  %48 = bitcast float** %.Z0972_326 to i8*, !dbg !37
  %49 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !37
  %50 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !37
  %51 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !37
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %51(i8* %45, i8* %46, i8* %47, i8* null, i8* %48, i8* null, i8* %49, i8* %50, i8* null, i64 0), !dbg !37
  %52 = load i32, i32* %n_312, align 4, !dbg !38
  call void @llvm.dbg.value(metadata i32 %52, metadata !33, metadata !DIExpression()), !dbg !10
  store i32 %52, i32* %.dY0001_360, align 4, !dbg !38
  call void @llvm.dbg.declare(metadata i32* %i_310, metadata !39, metadata !DIExpression()), !dbg !10
  store i32 1, i32* %i_310, align 4, !dbg !38
  %53 = load i32, i32* %.dY0001_360, align 4, !dbg !38
  %54 = icmp sle i32 %53, 0, !dbg !38
  br i1 %54, label %L.LB1_359, label %L.LB1_358, !dbg !38

L.LB1_358:                                        ; preds = %L.LB1_432, %L.LB1_380
  %55 = bitcast i32* %m_313 to i8*, !dbg !40
  %56 = bitcast %astruct.dt68* %.uplevelArgPack0001_403 to i8**, !dbg !40
  store i8* %55, i8** %56, align 8, !dbg !40
  %57 = bitcast float** %.Z0972_326 to i8*, !dbg !40
  %58 = bitcast %astruct.dt68* %.uplevelArgPack0001_403 to i8*, !dbg !40
  %59 = getelementptr i8, i8* %58, i64 8, !dbg !40
  %60 = bitcast i8* %59 to i8**, !dbg !40
  store i8* %57, i8** %60, align 8, !dbg !40
  %61 = bitcast float** %.Z0972_326 to i8*, !dbg !40
  %62 = bitcast %astruct.dt68* %.uplevelArgPack0001_403 to i8*, !dbg !40
  %63 = getelementptr i8, i8* %62, i64 16, !dbg !40
  %64 = bitcast i8* %63 to i8**, !dbg !40
  store i8* %61, i8** %64, align 8, !dbg !40
  %65 = bitcast i64* %z_b_0_314 to i8*, !dbg !40
  %66 = bitcast %astruct.dt68* %.uplevelArgPack0001_403 to i8*, !dbg !40
  %67 = getelementptr i8, i8* %66, i64 24, !dbg !40
  %68 = bitcast i8* %67 to i8**, !dbg !40
  store i8* %65, i8** %68, align 8, !dbg !40
  %69 = bitcast i64* %z_b_1_315 to i8*, !dbg !40
  %70 = bitcast %astruct.dt68* %.uplevelArgPack0001_403 to i8*, !dbg !40
  %71 = getelementptr i8, i8* %70, i64 32, !dbg !40
  %72 = bitcast i8* %71 to i8**, !dbg !40
  store i8* %69, i8** %72, align 8, !dbg !40
  %73 = bitcast i64* %z_e_63_321 to i8*, !dbg !40
  %74 = bitcast %astruct.dt68* %.uplevelArgPack0001_403 to i8*, !dbg !40
  %75 = getelementptr i8, i8* %74, i64 40, !dbg !40
  %76 = bitcast i8* %75 to i8**, !dbg !40
  store i8* %73, i8** %76, align 8, !dbg !40
  %77 = bitcast i64* %z_b_3_317 to i8*, !dbg !40
  %78 = bitcast %astruct.dt68* %.uplevelArgPack0001_403 to i8*, !dbg !40
  %79 = getelementptr i8, i8* %78, i64 48, !dbg !40
  %80 = bitcast i8* %79 to i8**, !dbg !40
  store i8* %77, i8** %80, align 8, !dbg !40
  %81 = bitcast i64* %z_b_4_318 to i8*, !dbg !40
  %82 = bitcast %astruct.dt68* %.uplevelArgPack0001_403 to i8*, !dbg !40
  %83 = getelementptr i8, i8* %82, i64 56, !dbg !40
  %84 = bitcast i8* %83 to i8**, !dbg !40
  store i8* %81, i8** %84, align 8, !dbg !40
  %85 = bitcast i64* %z_b_2_316 to i8*, !dbg !40
  %86 = bitcast %astruct.dt68* %.uplevelArgPack0001_403 to i8*, !dbg !40
  %87 = getelementptr i8, i8* %86, i64 64, !dbg !40
  %88 = bitcast i8* %87 to i8**, !dbg !40
  store i8* %85, i8** %88, align 8, !dbg !40
  %89 = bitcast i64* %z_e_66_322 to i8*, !dbg !40
  %90 = bitcast %astruct.dt68* %.uplevelArgPack0001_403 to i8*, !dbg !40
  %91 = getelementptr i8, i8* %90, i64 72, !dbg !40
  %92 = bitcast i8* %91 to i8**, !dbg !40
  store i8* %89, i8** %92, align 8, !dbg !40
  %93 = bitcast i64* %z_b_5_319 to i8*, !dbg !40
  %94 = bitcast %astruct.dt68* %.uplevelArgPack0001_403 to i8*, !dbg !40
  %95 = getelementptr i8, i8* %94, i64 80, !dbg !40
  %96 = bitcast i8* %95 to i8**, !dbg !40
  store i8* %93, i8** %96, align 8, !dbg !40
  %97 = bitcast i64* %z_b_6_320 to i8*, !dbg !40
  %98 = bitcast %astruct.dt68* %.uplevelArgPack0001_403 to i8*, !dbg !40
  %99 = getelementptr i8, i8* %98, i64 88, !dbg !40
  %100 = bitcast i8* %99 to i8**, !dbg !40
  store i8* %97, i8** %100, align 8, !dbg !40
  %101 = bitcast i32* %i_310 to i8*, !dbg !40
  %102 = bitcast %astruct.dt68* %.uplevelArgPack0001_403 to i8*, !dbg !40
  %103 = getelementptr i8, i8* %102, i64 96, !dbg !40
  %104 = bitcast i8* %103 to i8**, !dbg !40
  store i8* %101, i8** %104, align 8, !dbg !40
  %105 = bitcast [22 x i64]* %"b$sd1_349" to i8*, !dbg !40
  %106 = bitcast %astruct.dt68* %.uplevelArgPack0001_403 to i8*, !dbg !40
  %107 = getelementptr i8, i8* %106, i64 104, !dbg !40
  %108 = bitcast i8* %107 to i8**, !dbg !40
  store i8* %105, i8** %108, align 8, !dbg !40
  br label %L.LB1_432, !dbg !40

L.LB1_432:                                        ; preds = %L.LB1_358
  %109 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L27_1_ to i64*, !dbg !40
  %110 = bitcast %astruct.dt68* %.uplevelArgPack0001_403 to i64*, !dbg !40
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %109, i64* %110), !dbg !40
  %111 = load i32, i32* %i_310, align 4, !dbg !41
  call void @llvm.dbg.value(metadata i32 %111, metadata !39, metadata !DIExpression()), !dbg !10
  %112 = add nsw i32 %111, 1, !dbg !41
  store i32 %112, i32* %i_310, align 4, !dbg !41
  %113 = load i32, i32* %.dY0001_360, align 4, !dbg !41
  %114 = sub nsw i32 %113, 1, !dbg !41
  store i32 %114, i32* %.dY0001_360, align 4, !dbg !41
  %115 = load i32, i32* %.dY0001_360, align 4, !dbg !41
  %116 = icmp sgt i32 %115, 0, !dbg !41
  br i1 %116, label %L.LB1_358, label %L.LB1_359, !dbg !41

L.LB1_359:                                        ; preds = %L.LB1_432, %L.LB1_380
  call void (...) @_mp_bcs_nest(), !dbg !42
  %117 = bitcast i32* @.C336_MAIN_ to i8*, !dbg !42
  %118 = bitcast [67 x i8]* @.C334_MAIN_ to i8*, !dbg !42
  %119 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !42
  call void (i8*, i8*, i64, ...) %119(i8* %117, i8* %118, i64 67), !dbg !42
  %120 = bitcast i32* @.C338_MAIN_ to i8*, !dbg !42
  %121 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !42
  %122 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !42
  %123 = bitcast %struct.STATICS1* @.STATICS1 to i8*, !dbg !42
  %124 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !42
  %125 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %124(i8* %120, i8* null, i8* %121, i8* %122, i8* %123, i8* null, i64 0), !dbg !42
  call void @llvm.dbg.declare(metadata i32* %z__io_340, metadata !43, metadata !DIExpression()), !dbg !10
  store i32 %125, i32* %z__io_340, align 4, !dbg !42
  %126 = bitcast [22 x i64]* %"b$sd1_349" to i8*, !dbg !42
  %127 = getelementptr i8, i8* %126, i64 160, !dbg !42
  %128 = bitcast i8* %127 to i64*, !dbg !42
  %129 = load i64, i64* %128, align 8, !dbg !42
  %130 = mul nsw i64 %129, 500, !dbg !42
  %131 = bitcast [22 x i64]* %"b$sd1_349" to i8*, !dbg !42
  %132 = getelementptr i8, i8* %131, i64 56, !dbg !42
  %133 = bitcast i8* %132 to i64*, !dbg !42
  %134 = load i64, i64* %133, align 8, !dbg !42
  %135 = add nsw i64 %130, %134, !dbg !42
  %136 = load float*, float** %.Z0972_326, align 8, !dbg !42
  call void @llvm.dbg.value(metadata float* %136, metadata !20, metadata !DIExpression()), !dbg !10
  %137 = bitcast float* %136 to i8*, !dbg !42
  %138 = getelementptr i8, i8* %137, i64 1996, !dbg !42
  %139 = bitcast i8* %138 to float*, !dbg !42
  %140 = getelementptr float, float* %139, i64 %135, !dbg !42
  %141 = load float, float* %140, align 4, !dbg !42
  %142 = bitcast i32 (...)* @f90io_sc_f_fmt_write to i32 (float, i32, ...)*, !dbg !42
  %143 = call i32 (float, i32, ...) %142(float %141, i32 27), !dbg !42
  store i32 %143, i32* %z__io_340, align 4, !dbg !42
  %144 = call i32 (...) @f90io_fmtw_end(), !dbg !42
  store i32 %144, i32* %z__io_340, align 4, !dbg !42
  call void (...) @_mp_ecs_nest(), !dbg !42
  %145 = load float*, float** %.Z0972_326, align 8, !dbg !44
  call void @llvm.dbg.value(metadata float* %145, metadata !20, metadata !DIExpression()), !dbg !10
  %146 = bitcast float* %145 to i8*, !dbg !44
  %147 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !44
  %148 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !44
  call void (i8*, i8*, i8*, i8*, i64, ...) %148(i8* null, i8* %146, i8* %147, i8* null, i64 0), !dbg !44
  %149 = bitcast float** %.Z0972_326 to i8**, !dbg !44
  store i8* null, i8** %149, align 8, !dbg !44
  %150 = bitcast [22 x i64]* %"b$sd1_349" to i64*, !dbg !44
  store i64 0, i64* %150, align 8, !dbg !44
  ret void, !dbg !18
}

define internal void @__nv_MAIN__F1L27_1_(i32* %__nv_MAIN__F1L27_1Arg0, i64* %__nv_MAIN__F1L27_1Arg1, i64* %__nv_MAIN__F1L27_1Arg2) #0 !dbg !45 {
L.entry:
  %__gtid___nv_MAIN__F1L27_1__487 = alloca i32, align 4
  %.i0000p_331 = alloca i32, align 4
  %j_330 = alloca i32, align 4
  %.du0002p_364 = alloca i32, align 4
  %.de0002p_365 = alloca i32, align 4
  %.di0002p_366 = alloca i32, align 4
  %.ds0002p_367 = alloca i32, align 4
  %.dl0002p_369 = alloca i32, align 4
  %.dl0002p.copy_481 = alloca i32, align 4
  %.de0002p.copy_482 = alloca i32, align 4
  %.ds0002p.copy_483 = alloca i32, align 4
  %.dX0002p_368 = alloca i32, align 4
  %.dY0002p_363 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L27_1Arg0, metadata !48, metadata !DIExpression()), !dbg !49
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L27_1Arg1, metadata !50, metadata !DIExpression()), !dbg !49
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L27_1Arg2, metadata !51, metadata !DIExpression()), !dbg !49
  call void @llvm.dbg.value(metadata i32 1, metadata !52, metadata !DIExpression()), !dbg !49
  call void @llvm.dbg.value(metadata i32 2, metadata !53, metadata !DIExpression()), !dbg !49
  call void @llvm.dbg.value(metadata i32 0, metadata !54, metadata !DIExpression()), !dbg !49
  call void @llvm.dbg.value(metadata i32 1, metadata !55, metadata !DIExpression()), !dbg !49
  call void @llvm.dbg.value(metadata i32 2, metadata !56, metadata !DIExpression()), !dbg !49
  call void @llvm.dbg.value(metadata i32 0, metadata !57, metadata !DIExpression()), !dbg !49
  call void @llvm.dbg.value(metadata i32 1, metadata !58, metadata !DIExpression()), !dbg !49
  call void @llvm.dbg.value(metadata i32 2, metadata !59, metadata !DIExpression()), !dbg !49
  %0 = load i32, i32* %__nv_MAIN__F1L27_1Arg0, align 4, !dbg !60
  store i32 %0, i32* %__gtid___nv_MAIN__F1L27_1__487, align 4, !dbg !60
  br label %L.LB2_472

L.LB2_472:                                        ; preds = %L.entry
  br label %L.LB2_329

L.LB2_329:                                        ; preds = %L.LB2_472
  store i32 0, i32* %.i0000p_331, align 4, !dbg !61
  call void @llvm.dbg.declare(metadata i32* %j_330, metadata !62, metadata !DIExpression()), !dbg !60
  store i32 2, i32* %j_330, align 4, !dbg !61
  %1 = bitcast i64* %__nv_MAIN__F1L27_1Arg2 to i32**, !dbg !61
  %2 = load i32*, i32** %1, align 8, !dbg !61
  %3 = load i32, i32* %2, align 4, !dbg !61
  store i32 %3, i32* %.du0002p_364, align 4, !dbg !61
  %4 = bitcast i64* %__nv_MAIN__F1L27_1Arg2 to i32**, !dbg !61
  %5 = load i32*, i32** %4, align 8, !dbg !61
  %6 = load i32, i32* %5, align 4, !dbg !61
  store i32 %6, i32* %.de0002p_365, align 4, !dbg !61
  store i32 1, i32* %.di0002p_366, align 4, !dbg !61
  %7 = load i32, i32* %.di0002p_366, align 4, !dbg !61
  store i32 %7, i32* %.ds0002p_367, align 4, !dbg !61
  store i32 2, i32* %.dl0002p_369, align 4, !dbg !61
  %8 = load i32, i32* %.dl0002p_369, align 4, !dbg !61
  store i32 %8, i32* %.dl0002p.copy_481, align 4, !dbg !61
  %9 = load i32, i32* %.de0002p_365, align 4, !dbg !61
  store i32 %9, i32* %.de0002p.copy_482, align 4, !dbg !61
  %10 = load i32, i32* %.ds0002p_367, align 4, !dbg !61
  store i32 %10, i32* %.ds0002p.copy_483, align 4, !dbg !61
  %11 = load i32, i32* %__gtid___nv_MAIN__F1L27_1__487, align 4, !dbg !61
  %12 = bitcast i32* %.i0000p_331 to i64*, !dbg !61
  %13 = bitcast i32* %.dl0002p.copy_481 to i64*, !dbg !61
  %14 = bitcast i32* %.de0002p.copy_482 to i64*, !dbg !61
  %15 = bitcast i32* %.ds0002p.copy_483 to i64*, !dbg !61
  %16 = load i32, i32* %.ds0002p.copy_483, align 4, !dbg !61
  call void @__kmpc_for_static_init_4(i64* null, i32 %11, i32 34, i64* %12, i64* %13, i64* %14, i64* %15, i32 %16, i32 1), !dbg !61
  %17 = load i32, i32* %.dl0002p.copy_481, align 4, !dbg !61
  store i32 %17, i32* %.dl0002p_369, align 4, !dbg !61
  %18 = load i32, i32* %.de0002p.copy_482, align 4, !dbg !61
  store i32 %18, i32* %.de0002p_365, align 4, !dbg !61
  %19 = load i32, i32* %.ds0002p.copy_483, align 4, !dbg !61
  store i32 %19, i32* %.ds0002p_367, align 4, !dbg !61
  %20 = load i32, i32* %.dl0002p_369, align 4, !dbg !61
  store i32 %20, i32* %j_330, align 4, !dbg !61
  %21 = load i32, i32* %j_330, align 4, !dbg !61
  call void @llvm.dbg.value(metadata i32 %21, metadata !62, metadata !DIExpression()), !dbg !60
  store i32 %21, i32* %.dX0002p_368, align 4, !dbg !61
  %22 = load i32, i32* %.dX0002p_368, align 4, !dbg !61
  %23 = load i32, i32* %.du0002p_364, align 4, !dbg !61
  %24 = icmp sgt i32 %22, %23, !dbg !61
  br i1 %24, label %L.LB2_362, label %L.LB2_513, !dbg !61

L.LB2_513:                                        ; preds = %L.LB2_329
  %25 = load i32, i32* %.dX0002p_368, align 4, !dbg !61
  store i32 %25, i32* %j_330, align 4, !dbg !61
  %26 = load i32, i32* %.di0002p_366, align 4, !dbg !61
  %27 = load i32, i32* %.de0002p_365, align 4, !dbg !61
  %28 = load i32, i32* %.dX0002p_368, align 4, !dbg !61
  %29 = sub nsw i32 %27, %28, !dbg !61
  %30 = add nsw i32 %26, %29, !dbg !61
  %31 = load i32, i32* %.di0002p_366, align 4, !dbg !61
  %32 = sdiv i32 %30, %31, !dbg !61
  store i32 %32, i32* %.dY0002p_363, align 4, !dbg !61
  %33 = load i32, i32* %.dY0002p_363, align 4, !dbg !61
  %34 = icmp sle i32 %33, 0, !dbg !61
  br i1 %34, label %L.LB2_372, label %L.LB2_371, !dbg !61

L.LB2_371:                                        ; preds = %L.LB2_371, %L.LB2_513
  %35 = bitcast i64* %__nv_MAIN__F1L27_1Arg2 to i8*, !dbg !63
  %36 = getelementptr i8, i8* %35, i64 96, !dbg !63
  %37 = bitcast i8* %36 to i32**, !dbg !63
  %38 = load i32*, i32** %37, align 8, !dbg !63
  %39 = load i32, i32* %38, align 4, !dbg !63
  %40 = sext i32 %39 to i64, !dbg !63
  %41 = load i32, i32* %j_330, align 4, !dbg !63
  call void @llvm.dbg.value(metadata i32 %41, metadata !62, metadata !DIExpression()), !dbg !60
  %42 = sext i32 %41 to i64, !dbg !63
  %43 = sub nsw i64 %42, 1, !dbg !63
  %44 = bitcast i64* %__nv_MAIN__F1L27_1Arg2 to i8*, !dbg !63
  %45 = getelementptr i8, i8* %44, i64 104, !dbg !63
  %46 = bitcast i8* %45 to i8**, !dbg !63
  %47 = load i8*, i8** %46, align 8, !dbg !63
  %48 = getelementptr i8, i8* %47, i64 160, !dbg !63
  %49 = bitcast i8* %48 to i64*, !dbg !63
  %50 = load i64, i64* %49, align 8, !dbg !63
  %51 = mul nsw i64 %43, %50, !dbg !63
  %52 = add nsw i64 %40, %51, !dbg !63
  %53 = bitcast i64* %__nv_MAIN__F1L27_1Arg2 to i8*, !dbg !63
  %54 = getelementptr i8, i8* %53, i64 104, !dbg !63
  %55 = bitcast i8* %54 to i8**, !dbg !63
  %56 = load i8*, i8** %55, align 8, !dbg !63
  %57 = getelementptr i8, i8* %56, i64 56, !dbg !63
  %58 = bitcast i8* %57 to i64*, !dbg !63
  %59 = load i64, i64* %58, align 8, !dbg !63
  %60 = add nsw i64 %52, %59, !dbg !63
  %61 = bitcast i64* %__nv_MAIN__F1L27_1Arg2 to i8*, !dbg !63
  %62 = getelementptr i8, i8* %61, i64 16, !dbg !63
  %63 = bitcast i8* %62 to i8***, !dbg !63
  %64 = load i8**, i8*** %63, align 8, !dbg !63
  %65 = load i8*, i8** %64, align 8, !dbg !63
  %66 = getelementptr i8, i8* %65, i64 -4, !dbg !63
  %67 = bitcast i8* %66 to float*, !dbg !63
  %68 = getelementptr float, float* %67, i64 %60, !dbg !63
  %69 = load float, float* %68, align 4, !dbg !63
  %70 = bitcast i64* %__nv_MAIN__F1L27_1Arg2 to i8*, !dbg !63
  %71 = getelementptr i8, i8* %70, i64 104, !dbg !63
  %72 = bitcast i8* %71 to i8**, !dbg !63
  %73 = load i8*, i8** %72, align 8, !dbg !63
  %74 = getelementptr i8, i8* %73, i64 56, !dbg !63
  %75 = bitcast i8* %74 to i64*, !dbg !63
  %76 = load i64, i64* %75, align 8, !dbg !63
  %77 = bitcast i64* %__nv_MAIN__F1L27_1Arg2 to i8*, !dbg !63
  %78 = getelementptr i8, i8* %77, i64 96, !dbg !63
  %79 = bitcast i8* %78 to i32**, !dbg !63
  %80 = load i32*, i32** %79, align 8, !dbg !63
  %81 = load i32, i32* %80, align 4, !dbg !63
  %82 = sext i32 %81 to i64, !dbg !63
  %83 = load i32, i32* %j_330, align 4, !dbg !63
  call void @llvm.dbg.value(metadata i32 %83, metadata !62, metadata !DIExpression()), !dbg !60
  %84 = sext i32 %83 to i64, !dbg !63
  %85 = bitcast i64* %__nv_MAIN__F1L27_1Arg2 to i8*, !dbg !63
  %86 = getelementptr i8, i8* %85, i64 104, !dbg !63
  %87 = bitcast i8* %86 to i8**, !dbg !63
  %88 = load i8*, i8** %87, align 8, !dbg !63
  %89 = getelementptr i8, i8* %88, i64 160, !dbg !63
  %90 = bitcast i8* %89 to i64*, !dbg !63
  %91 = load i64, i64* %90, align 8, !dbg !63
  %92 = mul nsw i64 %84, %91, !dbg !63
  %93 = add nsw i64 %82, %92, !dbg !63
  %94 = add nsw i64 %76, %93, !dbg !63
  %95 = bitcast i64* %__nv_MAIN__F1L27_1Arg2 to i8*, !dbg !63
  %96 = getelementptr i8, i8* %95, i64 16, !dbg !63
  %97 = bitcast i8* %96 to i8***, !dbg !63
  %98 = load i8**, i8*** %97, align 8, !dbg !63
  %99 = load i8*, i8** %98, align 8, !dbg !63
  %100 = getelementptr i8, i8* %99, i64 -4, !dbg !63
  %101 = bitcast i8* %100 to float*, !dbg !63
  %102 = getelementptr float, float* %101, i64 %94, !dbg !63
  store float %69, float* %102, align 4, !dbg !63
  %103 = load i32, i32* %.di0002p_366, align 4, !dbg !60
  %104 = load i32, i32* %j_330, align 4, !dbg !60
  call void @llvm.dbg.value(metadata i32 %104, metadata !62, metadata !DIExpression()), !dbg !60
  %105 = add nsw i32 %103, %104, !dbg !60
  store i32 %105, i32* %j_330, align 4, !dbg !60
  %106 = load i32, i32* %.dY0002p_363, align 4, !dbg !60
  %107 = sub nsw i32 %106, 1, !dbg !60
  store i32 %107, i32* %.dY0002p_363, align 4, !dbg !60
  %108 = load i32, i32* %.dY0002p_363, align 4, !dbg !60
  %109 = icmp sgt i32 %108, 0, !dbg !60
  br i1 %109, label %L.LB2_371, label %L.LB2_372, !dbg !60

L.LB2_372:                                        ; preds = %L.LB2_371, %L.LB2_513
  br label %L.LB2_362

L.LB2_362:                                        ; preds = %L.LB2_372, %L.LB2_329
  %110 = load i32, i32* %__gtid___nv_MAIN__F1L27_1__487, align 4, !dbg !60
  call void @__kmpc_for_static_fini(i64* null, i32 %110), !dbg !60
  br label %L.LB2_332

L.LB2_332:                                        ; preds = %L.LB2_362
  ret void, !dbg !60
}

declare void @__kmpc_for_static_fini(i64*, i32) #0

declare void @__kmpc_for_static_init_4(i64*, i32, i32, i64*, i64*, i64*, i64*, i32, i32) #0

declare void @f90_dealloc03a_i8(...) #0

declare void @_mp_ecs_nest(...) #0

declare signext i32 @f90io_fmtw_end(...) #0

declare signext i32 @f90io_sc_f_fmt_write(...) #0

declare signext i32 @f90io_fmtw_inita(...) #0

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
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB037-truedepseconddimension-orig-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb037_truedepseconddimension_orig_yes", scope: !2, file: !3, line: 13, type: !6, scopeLine: 13, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
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
!19 = !DILocation(line: 13, column: 1, scope: !5)
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
!30 = !DILocalVariable(name: "len", scope: !5, file: !3, type: !9)
!31 = !DILocation(line: 20, column: 1, scope: !5)
!32 = !DILocation(line: 21, column: 1, scope: !5)
!33 = !DILocalVariable(name: "n", scope: !5, file: !3, type: !9)
!34 = !DILocation(line: 22, column: 1, scope: !5)
!35 = !DILocalVariable(name: "m", scope: !5, file: !3, type: !9)
!36 = !DILocalVariable(scope: !5, file: !3, type: !27, flags: DIFlagArtificial)
!37 = !DILocation(line: 24, column: 1, scope: !5)
!38 = !DILocation(line: 26, column: 1, scope: !5)
!39 = !DILocalVariable(name: "i", scope: !5, file: !3, type: !9)
!40 = !DILocation(line: 27, column: 1, scope: !5)
!41 = !DILocation(line: 32, column: 1, scope: !5)
!42 = !DILocation(line: 34, column: 1, scope: !5)
!43 = !DILocalVariable(scope: !5, file: !3, type: !9, flags: DIFlagArtificial)
!44 = !DILocation(line: 37, column: 1, scope: !5)
!45 = distinct !DISubprogram(name: "__nv_MAIN__F1L27_1", scope: !2, file: !3, line: 27, type: !46, scopeLine: 27, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!46 = !DISubroutineType(types: !47)
!47 = !{null, !9, !27, !27}
!48 = !DILocalVariable(name: "__nv_MAIN__F1L27_1Arg0", arg: 1, scope: !45, file: !3, type: !9)
!49 = !DILocation(line: 0, scope: !45)
!50 = !DILocalVariable(name: "__nv_MAIN__F1L27_1Arg1", arg: 2, scope: !45, file: !3, type: !27)
!51 = !DILocalVariable(name: "__nv_MAIN__F1L27_1Arg2", arg: 3, scope: !45, file: !3, type: !27)
!52 = !DILocalVariable(name: "omp_sched_static", scope: !45, file: !3, type: !9)
!53 = !DILocalVariable(name: "omp_sched_dynamic", scope: !45, file: !3, type: !9)
!54 = !DILocalVariable(name: "omp_proc_bind_false", scope: !45, file: !3, type: !9)
!55 = !DILocalVariable(name: "omp_proc_bind_true", scope: !45, file: !3, type: !9)
!56 = !DILocalVariable(name: "omp_proc_bind_master", scope: !45, file: !3, type: !9)
!57 = !DILocalVariable(name: "omp_lock_hint_none", scope: !45, file: !3, type: !9)
!58 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !45, file: !3, type: !9)
!59 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !45, file: !3, type: !9)
!60 = !DILocation(line: 30, column: 1, scope: !45)
!61 = !DILocation(line: 28, column: 1, scope: !45)
!62 = !DILocalVariable(name: "j", scope: !45, file: !3, type: !9)
!63 = !DILocation(line: 29, column: 1, scope: !45)

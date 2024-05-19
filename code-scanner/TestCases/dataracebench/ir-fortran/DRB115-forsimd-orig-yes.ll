; ModuleID = '/tmp/DRB115-forsimd-orig-yes-73c4f9.ll'
source_filename = "/tmp/DRB115-forsimd-orig-yes-73c4f9.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.C337_MAIN_ = internal constant i64 50
@.C305_MAIN_ = internal constant i32 14
@.C336_MAIN_ = internal constant [7 x i8] c"a(50) ="
@.C333_MAIN_ = internal constant i32 6
@.C330_MAIN_ = internal constant [52 x i8] c"micro-benchmarks-fortran/DRB115-forsimd-orig-yes.f95"
@.C332_MAIN_ = internal constant i32 33
@.C285_MAIN_ = internal constant i32 1
@.C306_MAIN_ = internal constant i32 25
@.C347_MAIN_ = internal constant i64 4
@.C346_MAIN_ = internal constant i64 25
@.C320_MAIN_ = internal constant i32 100
@.C286_MAIN_ = internal constant i64 1
@.C284_MAIN_ = internal constant i64 0
@.C283_MAIN_ = internal constant i32 0

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %.Z0966_323 = alloca i32*, align 8
  %"b$sd2_349" = alloca [16 x i64], align 8
  %.Z0965_322 = alloca i32*, align 8
  %"a$sd1_345" = alloca [16 x i64], align 8
  %len_321 = alloca i32, align 4
  %z_b_0_308 = alloca i64, align 8
  %z_b_1_309 = alloca i64, align 8
  %z_e_60_312 = alloca i64, align 8
  %z_b_2_310 = alloca i64, align 8
  %z_b_3_311 = alloca i64, align 8
  %z_b_4_315 = alloca i64, align 8
  %z_b_5_316 = alloca i64, align 8
  %z_e_67_319 = alloca i64, align 8
  %z_b_6_317 = alloca i64, align 8
  %z_b_7_318 = alloca i64, align 8
  %.dY0001_357 = alloca i32, align 4
  %i_307 = alloca i32, align 4
  %.i0000_327 = alloca i32, align 4
  %.dY0002_360 = alloca i32, align 4
  %i_326 = alloca i32, align 4
  %z__io_335 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !14, metadata !DIExpression()), !dbg !10
  %0 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !15
  %1 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !15
  call void (i8*, ...) %1(i8* %0), !dbg !15
  call void @llvm.dbg.declare(metadata i32** %.Z0966_323, metadata !16, metadata !DIExpression(DW_OP_deref)), !dbg !10
  %2 = bitcast i32** %.Z0966_323 to i8**, !dbg !15
  store i8* null, i8** %2, align 8, !dbg !15
  call void @llvm.dbg.declare(metadata [16 x i64]* %"b$sd2_349", metadata !20, metadata !DIExpression()), !dbg !10
  %3 = bitcast [16 x i64]* %"b$sd2_349" to i64*, !dbg !15
  store i64 0, i64* %3, align 8, !dbg !15
  call void @llvm.dbg.declare(metadata i32** %.Z0965_322, metadata !25, metadata !DIExpression(DW_OP_deref)), !dbg !10
  %4 = bitcast i32** %.Z0965_322 to i8**, !dbg !15
  store i8* null, i8** %4, align 8, !dbg !15
  call void @llvm.dbg.declare(metadata [16 x i64]* %"a$sd1_345", metadata !20, metadata !DIExpression()), !dbg !10
  %5 = bitcast [16 x i64]* %"a$sd1_345" to i64*, !dbg !15
  store i64 0, i64* %5, align 8, !dbg !15
  br label %L.LB1_370

L.LB1_370:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %len_321, metadata !26, metadata !DIExpression()), !dbg !10
  store i32 100, i32* %len_321, align 4, !dbg !27
  call void @llvm.dbg.declare(metadata i64* %z_b_0_308, metadata !28, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_0_308, align 8, !dbg !29
  %6 = load i32, i32* %len_321, align 4, !dbg !29
  call void @llvm.dbg.value(metadata i32 %6, metadata !26, metadata !DIExpression()), !dbg !10
  %7 = sext i32 %6 to i64, !dbg !29
  call void @llvm.dbg.declare(metadata i64* %z_b_1_309, metadata !28, metadata !DIExpression()), !dbg !10
  store i64 %7, i64* %z_b_1_309, align 8, !dbg !29
  %8 = load i64, i64* %z_b_1_309, align 8, !dbg !29
  call void @llvm.dbg.value(metadata i64 %8, metadata !28, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_60_312, metadata !28, metadata !DIExpression()), !dbg !10
  store i64 %8, i64* %z_e_60_312, align 8, !dbg !29
  %9 = bitcast [16 x i64]* %"a$sd1_345" to i8*, !dbg !29
  %10 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !29
  %11 = bitcast i64* @.C346_MAIN_ to i8*, !dbg !29
  %12 = bitcast i64* @.C347_MAIN_ to i8*, !dbg !29
  %13 = bitcast i64* %z_b_0_308 to i8*, !dbg !29
  %14 = bitcast i64* %z_b_1_309 to i8*, !dbg !29
  %15 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !29
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %15(i8* %9, i8* %10, i8* %11, i8* %12, i8* %13, i8* %14), !dbg !29
  %16 = bitcast [16 x i64]* %"a$sd1_345" to i8*, !dbg !29
  %17 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !29
  call void (i8*, i32, ...) %17(i8* %16, i32 25), !dbg !29
  %18 = load i64, i64* %z_b_1_309, align 8, !dbg !29
  call void @llvm.dbg.value(metadata i64 %18, metadata !28, metadata !DIExpression()), !dbg !10
  %19 = load i64, i64* %z_b_0_308, align 8, !dbg !29
  call void @llvm.dbg.value(metadata i64 %19, metadata !28, metadata !DIExpression()), !dbg !10
  %20 = sub nsw i64 %19, 1, !dbg !29
  %21 = sub nsw i64 %18, %20, !dbg !29
  call void @llvm.dbg.declare(metadata i64* %z_b_2_310, metadata !28, metadata !DIExpression()), !dbg !10
  store i64 %21, i64* %z_b_2_310, align 8, !dbg !29
  %22 = load i64, i64* %z_b_0_308, align 8, !dbg !29
  call void @llvm.dbg.value(metadata i64 %22, metadata !28, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_b_3_311, metadata !28, metadata !DIExpression()), !dbg !10
  store i64 %22, i64* %z_b_3_311, align 8, !dbg !29
  %23 = bitcast i64* %z_b_2_310 to i8*, !dbg !29
  %24 = bitcast i64* @.C346_MAIN_ to i8*, !dbg !29
  %25 = bitcast i64* @.C347_MAIN_ to i8*, !dbg !29
  %26 = bitcast i32** %.Z0965_322 to i8*, !dbg !29
  %27 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !29
  %28 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !29
  %29 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !29
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %29(i8* %23, i8* %24, i8* %25, i8* null, i8* %26, i8* null, i8* %27, i8* %28, i8* null, i64 0), !dbg !29
  call void @llvm.dbg.declare(metadata i64* %z_b_4_315, metadata !28, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_4_315, align 8, !dbg !30
  %30 = load i32, i32* %len_321, align 4, !dbg !30
  call void @llvm.dbg.value(metadata i32 %30, metadata !26, metadata !DIExpression()), !dbg !10
  %31 = sext i32 %30 to i64, !dbg !30
  call void @llvm.dbg.declare(metadata i64* %z_b_5_316, metadata !28, metadata !DIExpression()), !dbg !10
  store i64 %31, i64* %z_b_5_316, align 8, !dbg !30
  %32 = load i64, i64* %z_b_5_316, align 8, !dbg !30
  call void @llvm.dbg.value(metadata i64 %32, metadata !28, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_67_319, metadata !28, metadata !DIExpression()), !dbg !10
  store i64 %32, i64* %z_e_67_319, align 8, !dbg !30
  %33 = bitcast [16 x i64]* %"b$sd2_349" to i8*, !dbg !30
  %34 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !30
  %35 = bitcast i64* @.C346_MAIN_ to i8*, !dbg !30
  %36 = bitcast i64* @.C347_MAIN_ to i8*, !dbg !30
  %37 = bitcast i64* %z_b_4_315 to i8*, !dbg !30
  %38 = bitcast i64* %z_b_5_316 to i8*, !dbg !30
  %39 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !30
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %39(i8* %33, i8* %34, i8* %35, i8* %36, i8* %37, i8* %38), !dbg !30
  %40 = bitcast [16 x i64]* %"b$sd2_349" to i8*, !dbg !30
  %41 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !30
  call void (i8*, i32, ...) %41(i8* %40, i32 25), !dbg !30
  %42 = load i64, i64* %z_b_5_316, align 8, !dbg !30
  call void @llvm.dbg.value(metadata i64 %42, metadata !28, metadata !DIExpression()), !dbg !10
  %43 = load i64, i64* %z_b_4_315, align 8, !dbg !30
  call void @llvm.dbg.value(metadata i64 %43, metadata !28, metadata !DIExpression()), !dbg !10
  %44 = sub nsw i64 %43, 1, !dbg !30
  %45 = sub nsw i64 %42, %44, !dbg !30
  call void @llvm.dbg.declare(metadata i64* %z_b_6_317, metadata !28, metadata !DIExpression()), !dbg !10
  store i64 %45, i64* %z_b_6_317, align 8, !dbg !30
  %46 = load i64, i64* %z_b_4_315, align 8, !dbg !30
  call void @llvm.dbg.value(metadata i64 %46, metadata !28, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_b_7_318, metadata !28, metadata !DIExpression()), !dbg !10
  store i64 %46, i64* %z_b_7_318, align 8, !dbg !30
  %47 = bitcast i64* %z_b_6_317 to i8*, !dbg !30
  %48 = bitcast i64* @.C346_MAIN_ to i8*, !dbg !30
  %49 = bitcast i64* @.C347_MAIN_ to i8*, !dbg !30
  %50 = bitcast i32** %.Z0966_323 to i8*, !dbg !30
  %51 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !30
  %52 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !30
  %53 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !30
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %53(i8* %47, i8* %48, i8* %49, i8* null, i8* %50, i8* null, i8* %51, i8* %52, i8* null, i64 0), !dbg !30
  %54 = load i32, i32* %len_321, align 4, !dbg !31
  call void @llvm.dbg.value(metadata i32 %54, metadata !26, metadata !DIExpression()), !dbg !10
  store i32 %54, i32* %.dY0001_357, align 4, !dbg !31
  call void @llvm.dbg.declare(metadata i32* %i_307, metadata !32, metadata !DIExpression()), !dbg !10
  store i32 1, i32* %i_307, align 4, !dbg !31
  %55 = load i32, i32* %.dY0001_357, align 4, !dbg !31
  %56 = icmp sle i32 %55, 0, !dbg !31
  br i1 %56, label %L.LB1_356, label %L.LB1_355, !dbg !31

L.LB1_355:                                        ; preds = %L.LB1_355, %L.LB1_370
  %57 = load i32, i32* %i_307, align 4, !dbg !33
  call void @llvm.dbg.value(metadata i32 %57, metadata !32, metadata !DIExpression()), !dbg !10
  %58 = load i32, i32* %i_307, align 4, !dbg !33
  call void @llvm.dbg.value(metadata i32 %58, metadata !32, metadata !DIExpression()), !dbg !10
  %59 = sext i32 %58 to i64, !dbg !33
  %60 = bitcast [16 x i64]* %"a$sd1_345" to i8*, !dbg !33
  %61 = getelementptr i8, i8* %60, i64 56, !dbg !33
  %62 = bitcast i8* %61 to i64*, !dbg !33
  %63 = load i64, i64* %62, align 8, !dbg !33
  %64 = add nsw i64 %59, %63, !dbg !33
  %65 = load i32*, i32** %.Z0965_322, align 8, !dbg !33
  call void @llvm.dbg.value(metadata i32* %65, metadata !25, metadata !DIExpression()), !dbg !10
  %66 = bitcast i32* %65 to i8*, !dbg !33
  %67 = getelementptr i8, i8* %66, i64 -4, !dbg !33
  %68 = bitcast i8* %67 to i32*, !dbg !33
  %69 = getelementptr i32, i32* %68, i64 %64, !dbg !33
  store i32 %57, i32* %69, align 4, !dbg !33
  %70 = load i32, i32* %i_307, align 4, !dbg !34
  call void @llvm.dbg.value(metadata i32 %70, metadata !32, metadata !DIExpression()), !dbg !10
  %71 = add nsw i32 %70, 1, !dbg !34
  %72 = load i32, i32* %i_307, align 4, !dbg !34
  call void @llvm.dbg.value(metadata i32 %72, metadata !32, metadata !DIExpression()), !dbg !10
  %73 = sext i32 %72 to i64, !dbg !34
  %74 = bitcast [16 x i64]* %"b$sd2_349" to i8*, !dbg !34
  %75 = getelementptr i8, i8* %74, i64 56, !dbg !34
  %76 = bitcast i8* %75 to i64*, !dbg !34
  %77 = load i64, i64* %76, align 8, !dbg !34
  %78 = add nsw i64 %73, %77, !dbg !34
  %79 = load i32*, i32** %.Z0966_323, align 8, !dbg !34
  call void @llvm.dbg.value(metadata i32* %79, metadata !16, metadata !DIExpression()), !dbg !10
  %80 = bitcast i32* %79 to i8*, !dbg !34
  %81 = getelementptr i8, i8* %80, i64 -4, !dbg !34
  %82 = bitcast i8* %81 to i32*, !dbg !34
  %83 = getelementptr i32, i32* %82, i64 %78, !dbg !34
  store i32 %71, i32* %83, align 4, !dbg !34
  %84 = load i32, i32* %i_307, align 4, !dbg !35
  call void @llvm.dbg.value(metadata i32 %84, metadata !32, metadata !DIExpression()), !dbg !10
  %85 = add nsw i32 %84, 1, !dbg !35
  store i32 %85, i32* %i_307, align 4, !dbg !35
  %86 = load i32, i32* %.dY0001_357, align 4, !dbg !35
  %87 = sub nsw i32 %86, 1, !dbg !35
  store i32 %87, i32* %.dY0001_357, align 4, !dbg !35
  %88 = load i32, i32* %.dY0001_357, align 4, !dbg !35
  %89 = icmp sgt i32 %88, 0, !dbg !35
  br i1 %89, label %L.LB1_355, label %L.LB1_356, !dbg !35

L.LB1_356:                                        ; preds = %L.LB1_355, %L.LB1_370
  br label %L.LB1_325

L.LB1_325:                                        ; preds = %L.LB1_356
  %90 = load i32, i32* %len_321, align 4, !dbg !36
  call void @llvm.dbg.value(metadata i32 %90, metadata !26, metadata !DIExpression()), !dbg !10
  store i32 %90, i32* %.i0000_327, align 4, !dbg !36
  %91 = load i32, i32* %len_321, align 4, !dbg !36
  call void @llvm.dbg.value(metadata i32 %91, metadata !26, metadata !DIExpression()), !dbg !10
  %92 = sub nsw i32 %91, 1, !dbg !36
  store i32 %92, i32* %.dY0002_360, align 4, !dbg !36
  call void @llvm.dbg.declare(metadata i32* %i_326, metadata !32, metadata !DIExpression()), !dbg !37
  store i32 1, i32* %i_326, align 4, !dbg !36
  %93 = load i32, i32* %.dY0002_360, align 4, !dbg !36
  %94 = icmp sle i32 %93, 0, !dbg !36
  br i1 %94, label %L.LB1_359, label %L.LB1_358, !dbg !36

L.LB1_358:                                        ; preds = %L.LB1_358, %L.LB1_325
  %95 = bitcast [16 x i64]* %"b$sd2_349" to i8*, !dbg !38
  %96 = getelementptr i8, i8* %95, i64 56, !dbg !38
  %97 = bitcast i8* %96 to i64*, !dbg !38
  %98 = load i64, i64* %97, align 8, !dbg !38
  %99 = load i32, i32* %i_326, align 4, !dbg !38
  call void @llvm.dbg.value(metadata i32 %99, metadata !32, metadata !DIExpression()), !dbg !37
  %100 = sext i32 %99 to i64, !dbg !38
  %101 = add nsw i64 %98, %100, !dbg !38
  %102 = load i32*, i32** %.Z0966_323, align 8, !dbg !38
  call void @llvm.dbg.value(metadata i32* %102, metadata !16, metadata !DIExpression()), !dbg !10
  %103 = bitcast i32* %102 to i8*, !dbg !38
  %104 = getelementptr i8, i8* %103, i64 -4, !dbg !38
  %105 = bitcast i8* %104 to i32*, !dbg !38
  %106 = getelementptr i32, i32* %105, i64 %101, !dbg !38
  %107 = load i32, i32* %106, align 4, !dbg !38
  %108 = bitcast [16 x i64]* %"a$sd1_345" to i8*, !dbg !38
  %109 = getelementptr i8, i8* %108, i64 56, !dbg !38
  %110 = bitcast i8* %109 to i64*, !dbg !38
  %111 = load i64, i64* %110, align 8, !dbg !38
  %112 = load i32, i32* %i_326, align 4, !dbg !38
  call void @llvm.dbg.value(metadata i32 %112, metadata !32, metadata !DIExpression()), !dbg !37
  %113 = sext i32 %112 to i64, !dbg !38
  %114 = add nsw i64 %111, %113, !dbg !38
  %115 = load i32*, i32** %.Z0965_322, align 8, !dbg !38
  call void @llvm.dbg.value(metadata i32* %115, metadata !25, metadata !DIExpression()), !dbg !10
  %116 = bitcast i32* %115 to i8*, !dbg !38
  %117 = getelementptr i8, i8* %116, i64 -4, !dbg !38
  %118 = bitcast i8* %117 to i32*, !dbg !38
  %119 = getelementptr i32, i32* %118, i64 %114, !dbg !38
  %120 = load i32, i32* %119, align 4, !dbg !38
  %121 = add nsw i32 %107, %120, !dbg !38
  %122 = bitcast [16 x i64]* %"a$sd1_345" to i8*, !dbg !38
  %123 = getelementptr i8, i8* %122, i64 56, !dbg !38
  %124 = bitcast i8* %123 to i64*, !dbg !38
  %125 = load i64, i64* %124, align 8, !dbg !38
  %126 = load i32, i32* %i_326, align 4, !dbg !38
  call void @llvm.dbg.value(metadata i32 %126, metadata !32, metadata !DIExpression()), !dbg !37
  %127 = sext i32 %126 to i64, !dbg !38
  %128 = add nsw i64 %125, %127, !dbg !38
  %129 = load i32*, i32** %.Z0965_322, align 8, !dbg !38
  call void @llvm.dbg.value(metadata i32* %129, metadata !25, metadata !DIExpression()), !dbg !10
  %130 = getelementptr i32, i32* %129, i64 %128, !dbg !38
  store i32 %121, i32* %130, align 4, !dbg !38
  %131 = load i32, i32* %i_326, align 4, !dbg !39
  call void @llvm.dbg.value(metadata i32 %131, metadata !32, metadata !DIExpression()), !dbg !37
  %132 = add nsw i32 %131, 1, !dbg !39
  store i32 %132, i32* %i_326, align 4, !dbg !39
  %133 = load i32, i32* %.dY0002_360, align 4, !dbg !39
  %134 = sub nsw i32 %133, 1, !dbg !39
  store i32 %134, i32* %.dY0002_360, align 4, !dbg !39
  %135 = load i32, i32* %.dY0002_360, align 4, !dbg !39
  %136 = icmp sgt i32 %135, 0, !dbg !39
  br i1 %136, label %L.LB1_358, label %L.LB1_359, !dbg !39

L.LB1_359:                                        ; preds = %L.LB1_358, %L.LB1_325
  br label %L.LB1_328

L.LB1_328:                                        ; preds = %L.LB1_359
  call void (...) @_mp_bcs_nest(), !dbg !40
  %137 = bitcast i32* @.C332_MAIN_ to i8*, !dbg !40
  %138 = bitcast [52 x i8]* @.C330_MAIN_ to i8*, !dbg !40
  %139 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !40
  call void (i8*, i8*, i64, ...) %139(i8* %137, i8* %138, i64 52), !dbg !40
  %140 = bitcast i32* @.C333_MAIN_ to i8*, !dbg !40
  %141 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !40
  %142 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !40
  %143 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !40
  %144 = call i32 (i8*, i8*, i8*, i8*, ...) %143(i8* %140, i8* null, i8* %141, i8* %142), !dbg !40
  call void @llvm.dbg.declare(metadata i32* %z__io_335, metadata !41, metadata !DIExpression()), !dbg !10
  store i32 %144, i32* %z__io_335, align 4, !dbg !40
  %145 = bitcast [7 x i8]* @.C336_MAIN_ to i8*, !dbg !40
  %146 = bitcast i32 (...)* @f90io_sc_ch_ldw to i32 (i8*, i32, i64, ...)*, !dbg !40
  %147 = call i32 (i8*, i32, i64, ...) %146(i8* %145, i32 14, i64 7), !dbg !40
  store i32 %147, i32* %z__io_335, align 4, !dbg !40
  %148 = bitcast [16 x i64]* %"a$sd1_345" to i8*, !dbg !40
  %149 = getelementptr i8, i8* %148, i64 56, !dbg !40
  %150 = bitcast i8* %149 to i64*, !dbg !40
  %151 = load i64, i64* %150, align 8, !dbg !40
  %152 = load i32*, i32** %.Z0965_322, align 8, !dbg !40
  call void @llvm.dbg.value(metadata i32* %152, metadata !25, metadata !DIExpression()), !dbg !10
  %153 = bitcast i32* %152 to i8*, !dbg !40
  %154 = getelementptr i8, i8* %153, i64 196, !dbg !40
  %155 = bitcast i8* %154 to i32*, !dbg !40
  %156 = getelementptr i32, i32* %155, i64 %151, !dbg !40
  %157 = load i32, i32* %156, align 4, !dbg !40
  %158 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !40
  %159 = call i32 (i32, i32, ...) %158(i32 %157, i32 25), !dbg !40
  store i32 %159, i32* %z__io_335, align 4, !dbg !40
  %160 = call i32 (...) @f90io_ldw_end(), !dbg !40
  store i32 %160, i32* %z__io_335, align 4, !dbg !40
  call void (...) @_mp_ecs_nest(), !dbg !40
  %161 = load i32*, i32** %.Z0965_322, align 8, !dbg !42
  call void @llvm.dbg.value(metadata i32* %161, metadata !25, metadata !DIExpression()), !dbg !10
  %162 = bitcast i32* %161 to i8*, !dbg !42
  %163 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !42
  %164 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !42
  call void (i8*, i8*, i8*, i8*, i64, ...) %164(i8* null, i8* %162, i8* %163, i8* null, i64 0), !dbg !42
  %165 = bitcast i32** %.Z0965_322 to i8**, !dbg !42
  store i8* null, i8** %165, align 8, !dbg !42
  %166 = bitcast [16 x i64]* %"a$sd1_345" to i64*, !dbg !42
  store i64 0, i64* %166, align 8, !dbg !42
  %167 = load i32*, i32** %.Z0966_323, align 8, !dbg !42
  call void @llvm.dbg.value(metadata i32* %167, metadata !16, metadata !DIExpression()), !dbg !10
  %168 = bitcast i32* %167 to i8*, !dbg !42
  %169 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !42
  %170 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !42
  call void (i8*, i8*, i8*, i8*, i64, ...) %170(i8* null, i8* %168, i8* %169, i8* null, i64 0), !dbg !42
  %171 = bitcast i32** %.Z0966_323 to i8**, !dbg !42
  store i8* null, i8** %171, align 8, !dbg !42
  %172 = bitcast [16 x i64]* %"b$sd2_349" to i64*, !dbg !42
  store i64 0, i64* %172, align 8, !dbg !42
  ret void, !dbg !37
}

declare void @f90_dealloc03a_i8(...) #0

declare void @_mp_ecs_nest(...) #0

declare signext i32 @f90io_ldw_end(...) #0

declare signext i32 @f90io_sc_i_ldw(...) #0

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

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { "no-frame-pointer-elim-non-leaf" }
attributes #1 = { nounwind readnone speculatable }

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !4, globals: !4, imports: !4)
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB115-forsimd-orig-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb115_forsimd_orig_yes", scope: !2, file: !3, line: 12, type: !6, scopeLine: 12, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
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
!16 = !DILocalVariable(name: "b", scope: !5, file: !3, type: !17)
!17 = !DICompositeType(tag: DW_TAG_array_type, baseType: !9, size: 32, align: 32, elements: !18)
!18 = !{!19}
!19 = !DISubrange(count: 0, lowerBound: 1)
!20 = !DILocalVariable(scope: !5, file: !3, type: !21, flags: DIFlagArtificial)
!21 = !DICompositeType(tag: DW_TAG_array_type, baseType: !22, size: 1024, align: 64, elements: !23)
!22 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!23 = !{!24}
!24 = !DISubrange(count: 16, lowerBound: 1)
!25 = !DILocalVariable(name: "a", scope: !5, file: !3, type: !17)
!26 = !DILocalVariable(name: "len", scope: !5, file: !3, type: !9)
!27 = !DILocation(line: 19, column: 1, scope: !5)
!28 = !DILocalVariable(scope: !5, file: !3, type: !22, flags: DIFlagArtificial)
!29 = !DILocation(line: 20, column: 1, scope: !5)
!30 = !DILocation(line: 21, column: 1, scope: !5)
!31 = !DILocation(line: 23, column: 1, scope: !5)
!32 = !DILocalVariable(name: "i", scope: !5, file: !3, type: !9)
!33 = !DILocation(line: 24, column: 1, scope: !5)
!34 = !DILocation(line: 25, column: 1, scope: !5)
!35 = !DILocation(line: 26, column: 1, scope: !5)
!36 = !DILocation(line: 29, column: 1, scope: !5)
!37 = !DILocation(line: 36, column: 1, scope: !5)
!38 = !DILocation(line: 30, column: 1, scope: !5)
!39 = !DILocation(line: 31, column: 1, scope: !5)
!40 = !DILocation(line: 33, column: 1, scope: !5)
!41 = !DILocalVariable(scope: !5, file: !3, type: !9, flags: DIFlagArtificial)
!42 = !DILocation(line: 35, column: 1, scope: !5)

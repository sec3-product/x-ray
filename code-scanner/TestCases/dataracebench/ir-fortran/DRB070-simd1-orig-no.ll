; ModuleID = '/tmp/DRB070-simd1-orig-no-b4fcc6.ll'
source_filename = "/tmp/DRB070-simd1-orig-no-b4fcc6.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.C285_MAIN_ = internal constant i32 1
@.C305_MAIN_ = internal constant i32 25
@.C339_MAIN_ = internal constant i64 4
@.C338_MAIN_ = internal constant i64 25
@.C325_MAIN_ = internal constant i32 100
@.C286_MAIN_ = internal constant i64 1
@.C284_MAIN_ = internal constant i64 0
@.C283_MAIN_ = internal constant i32 0

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %.Z0972_329 = alloca i32*, align 8
  %"c$sd3_342" = alloca [16 x i64], align 8
  %.Z0966_328 = alloca i32*, align 8
  %"b$sd2_341" = alloca [16 x i64], align 8
  %.Z0965_327 = alloca i32*, align 8
  %"a$sd1_337" = alloca [16 x i64], align 8
  %len_326 = alloca i32, align 4
  %z_b_0_307 = alloca i64, align 8
  %z_b_1_308 = alloca i64, align 8
  %z_e_60_311 = alloca i64, align 8
  %z_b_2_309 = alloca i64, align 8
  %z_b_3_310 = alloca i64, align 8
  %z_b_4_314 = alloca i64, align 8
  %z_b_5_315 = alloca i64, align 8
  %z_e_67_318 = alloca i64, align 8
  %z_b_6_316 = alloca i64, align 8
  %z_b_7_317 = alloca i64, align 8
  %z_b_8_320 = alloca i64, align 8
  %z_b_9_321 = alloca i64, align 8
  %z_e_74_324 = alloca i64, align 8
  %z_b_10_322 = alloca i64, align 8
  %z_b_11_323 = alloca i64, align 8
  %.i0000_333 = alloca i32, align 4
  %.dY0001_350 = alloca i32, align 4
  %i_332 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !14, metadata !DIExpression()), !dbg !10
  %0 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !15
  %1 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !15
  call void (i8*, ...) %1(i8* %0), !dbg !15
  call void @llvm.dbg.declare(metadata i32** %.Z0972_329, metadata !16, metadata !DIExpression(DW_OP_deref)), !dbg !10
  %2 = bitcast i32** %.Z0972_329 to i8**, !dbg !15
  store i8* null, i8** %2, align 8, !dbg !15
  call void @llvm.dbg.declare(metadata [16 x i64]* %"c$sd3_342", metadata !20, metadata !DIExpression()), !dbg !10
  %3 = bitcast [16 x i64]* %"c$sd3_342" to i64*, !dbg !15
  store i64 0, i64* %3, align 8, !dbg !15
  call void @llvm.dbg.declare(metadata i32** %.Z0966_328, metadata !25, metadata !DIExpression(DW_OP_deref)), !dbg !10
  %4 = bitcast i32** %.Z0966_328 to i8**, !dbg !15
  store i8* null, i8** %4, align 8, !dbg !15
  call void @llvm.dbg.declare(metadata [16 x i64]* %"b$sd2_341", metadata !20, metadata !DIExpression()), !dbg !10
  %5 = bitcast [16 x i64]* %"b$sd2_341" to i64*, !dbg !15
  store i64 0, i64* %5, align 8, !dbg !15
  call void @llvm.dbg.declare(metadata i32** %.Z0965_327, metadata !26, metadata !DIExpression(DW_OP_deref)), !dbg !10
  %6 = bitcast i32** %.Z0965_327 to i8**, !dbg !15
  store i8* null, i8** %6, align 8, !dbg !15
  call void @llvm.dbg.declare(metadata [16 x i64]* %"a$sd1_337", metadata !20, metadata !DIExpression()), !dbg !10
  %7 = bitcast [16 x i64]* %"a$sd1_337" to i64*, !dbg !15
  store i64 0, i64* %7, align 8, !dbg !15
  br label %L.LB1_362

L.LB1_362:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %len_326, metadata !27, metadata !DIExpression()), !dbg !10
  store i32 100, i32* %len_326, align 4, !dbg !28
  call void @llvm.dbg.declare(metadata i64* %z_b_0_307, metadata !29, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_0_307, align 8, !dbg !30
  %8 = load i32, i32* %len_326, align 4, !dbg !30
  call void @llvm.dbg.value(metadata i32 %8, metadata !27, metadata !DIExpression()), !dbg !10
  %9 = sext i32 %8 to i64, !dbg !30
  call void @llvm.dbg.declare(metadata i64* %z_b_1_308, metadata !29, metadata !DIExpression()), !dbg !10
  store i64 %9, i64* %z_b_1_308, align 8, !dbg !30
  %10 = load i64, i64* %z_b_1_308, align 8, !dbg !30
  call void @llvm.dbg.value(metadata i64 %10, metadata !29, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_60_311, metadata !29, metadata !DIExpression()), !dbg !10
  store i64 %10, i64* %z_e_60_311, align 8, !dbg !30
  %11 = bitcast [16 x i64]* %"a$sd1_337" to i8*, !dbg !30
  %12 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !30
  %13 = bitcast i64* @.C338_MAIN_ to i8*, !dbg !30
  %14 = bitcast i64* @.C339_MAIN_ to i8*, !dbg !30
  %15 = bitcast i64* %z_b_0_307 to i8*, !dbg !30
  %16 = bitcast i64* %z_b_1_308 to i8*, !dbg !30
  %17 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !30
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %17(i8* %11, i8* %12, i8* %13, i8* %14, i8* %15, i8* %16), !dbg !30
  %18 = bitcast [16 x i64]* %"a$sd1_337" to i8*, !dbg !30
  %19 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !30
  call void (i8*, i32, ...) %19(i8* %18, i32 25), !dbg !30
  %20 = load i64, i64* %z_b_1_308, align 8, !dbg !30
  call void @llvm.dbg.value(metadata i64 %20, metadata !29, metadata !DIExpression()), !dbg !10
  %21 = load i64, i64* %z_b_0_307, align 8, !dbg !30
  call void @llvm.dbg.value(metadata i64 %21, metadata !29, metadata !DIExpression()), !dbg !10
  %22 = sub nsw i64 %21, 1, !dbg !30
  %23 = sub nsw i64 %20, %22, !dbg !30
  call void @llvm.dbg.declare(metadata i64* %z_b_2_309, metadata !29, metadata !DIExpression()), !dbg !10
  store i64 %23, i64* %z_b_2_309, align 8, !dbg !30
  %24 = load i64, i64* %z_b_0_307, align 8, !dbg !30
  call void @llvm.dbg.value(metadata i64 %24, metadata !29, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_b_3_310, metadata !29, metadata !DIExpression()), !dbg !10
  store i64 %24, i64* %z_b_3_310, align 8, !dbg !30
  %25 = bitcast i64* %z_b_2_309 to i8*, !dbg !30
  %26 = bitcast i64* @.C338_MAIN_ to i8*, !dbg !30
  %27 = bitcast i64* @.C339_MAIN_ to i8*, !dbg !30
  %28 = bitcast i32** %.Z0965_327 to i8*, !dbg !30
  %29 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !30
  %30 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !30
  %31 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !30
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %31(i8* %25, i8* %26, i8* %27, i8* null, i8* %28, i8* null, i8* %29, i8* %30, i8* null, i64 0), !dbg !30
  call void @llvm.dbg.declare(metadata i64* %z_b_4_314, metadata !29, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_4_314, align 8, !dbg !31
  %32 = load i32, i32* %len_326, align 4, !dbg !31
  call void @llvm.dbg.value(metadata i32 %32, metadata !27, metadata !DIExpression()), !dbg !10
  %33 = sext i32 %32 to i64, !dbg !31
  call void @llvm.dbg.declare(metadata i64* %z_b_5_315, metadata !29, metadata !DIExpression()), !dbg !10
  store i64 %33, i64* %z_b_5_315, align 8, !dbg !31
  %34 = load i64, i64* %z_b_5_315, align 8, !dbg !31
  call void @llvm.dbg.value(metadata i64 %34, metadata !29, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_67_318, metadata !29, metadata !DIExpression()), !dbg !10
  store i64 %34, i64* %z_e_67_318, align 8, !dbg !31
  %35 = bitcast [16 x i64]* %"b$sd2_341" to i8*, !dbg !31
  %36 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !31
  %37 = bitcast i64* @.C338_MAIN_ to i8*, !dbg !31
  %38 = bitcast i64* @.C339_MAIN_ to i8*, !dbg !31
  %39 = bitcast i64* %z_b_4_314 to i8*, !dbg !31
  %40 = bitcast i64* %z_b_5_315 to i8*, !dbg !31
  %41 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !31
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %41(i8* %35, i8* %36, i8* %37, i8* %38, i8* %39, i8* %40), !dbg !31
  %42 = bitcast [16 x i64]* %"b$sd2_341" to i8*, !dbg !31
  %43 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !31
  call void (i8*, i32, ...) %43(i8* %42, i32 25), !dbg !31
  %44 = load i64, i64* %z_b_5_315, align 8, !dbg !31
  call void @llvm.dbg.value(metadata i64 %44, metadata !29, metadata !DIExpression()), !dbg !10
  %45 = load i64, i64* %z_b_4_314, align 8, !dbg !31
  call void @llvm.dbg.value(metadata i64 %45, metadata !29, metadata !DIExpression()), !dbg !10
  %46 = sub nsw i64 %45, 1, !dbg !31
  %47 = sub nsw i64 %44, %46, !dbg !31
  call void @llvm.dbg.declare(metadata i64* %z_b_6_316, metadata !29, metadata !DIExpression()), !dbg !10
  store i64 %47, i64* %z_b_6_316, align 8, !dbg !31
  %48 = load i64, i64* %z_b_4_314, align 8, !dbg !31
  call void @llvm.dbg.value(metadata i64 %48, metadata !29, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_b_7_317, metadata !29, metadata !DIExpression()), !dbg !10
  store i64 %48, i64* %z_b_7_317, align 8, !dbg !31
  %49 = bitcast i64* %z_b_6_316 to i8*, !dbg !31
  %50 = bitcast i64* @.C338_MAIN_ to i8*, !dbg !31
  %51 = bitcast i64* @.C339_MAIN_ to i8*, !dbg !31
  %52 = bitcast i32** %.Z0966_328 to i8*, !dbg !31
  %53 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !31
  %54 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !31
  %55 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !31
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %55(i8* %49, i8* %50, i8* %51, i8* null, i8* %52, i8* null, i8* %53, i8* %54, i8* null, i64 0), !dbg !31
  call void @llvm.dbg.declare(metadata i64* %z_b_8_320, metadata !29, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_8_320, align 8, !dbg !32
  %56 = load i32, i32* %len_326, align 4, !dbg !32
  call void @llvm.dbg.value(metadata i32 %56, metadata !27, metadata !DIExpression()), !dbg !10
  %57 = sext i32 %56 to i64, !dbg !32
  call void @llvm.dbg.declare(metadata i64* %z_b_9_321, metadata !29, metadata !DIExpression()), !dbg !10
  store i64 %57, i64* %z_b_9_321, align 8, !dbg !32
  %58 = load i64, i64* %z_b_9_321, align 8, !dbg !32
  call void @llvm.dbg.value(metadata i64 %58, metadata !29, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_74_324, metadata !29, metadata !DIExpression()), !dbg !10
  store i64 %58, i64* %z_e_74_324, align 8, !dbg !32
  %59 = bitcast [16 x i64]* %"c$sd3_342" to i8*, !dbg !32
  %60 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !32
  %61 = bitcast i64* @.C338_MAIN_ to i8*, !dbg !32
  %62 = bitcast i64* @.C339_MAIN_ to i8*, !dbg !32
  %63 = bitcast i64* %z_b_8_320 to i8*, !dbg !32
  %64 = bitcast i64* %z_b_9_321 to i8*, !dbg !32
  %65 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !32
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %65(i8* %59, i8* %60, i8* %61, i8* %62, i8* %63, i8* %64), !dbg !32
  %66 = bitcast [16 x i64]* %"c$sd3_342" to i8*, !dbg !32
  %67 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !32
  call void (i8*, i32, ...) %67(i8* %66, i32 25), !dbg !32
  %68 = load i64, i64* %z_b_9_321, align 8, !dbg !32
  call void @llvm.dbg.value(metadata i64 %68, metadata !29, metadata !DIExpression()), !dbg !10
  %69 = load i64, i64* %z_b_8_320, align 8, !dbg !32
  call void @llvm.dbg.value(metadata i64 %69, metadata !29, metadata !DIExpression()), !dbg !10
  %70 = sub nsw i64 %69, 1, !dbg !32
  %71 = sub nsw i64 %68, %70, !dbg !32
  call void @llvm.dbg.declare(metadata i64* %z_b_10_322, metadata !29, metadata !DIExpression()), !dbg !10
  store i64 %71, i64* %z_b_10_322, align 8, !dbg !32
  %72 = load i64, i64* %z_b_8_320, align 8, !dbg !32
  call void @llvm.dbg.value(metadata i64 %72, metadata !29, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_b_11_323, metadata !29, metadata !DIExpression()), !dbg !10
  store i64 %72, i64* %z_b_11_323, align 8, !dbg !32
  %73 = bitcast i64* %z_b_10_322 to i8*, !dbg !32
  %74 = bitcast i64* @.C338_MAIN_ to i8*, !dbg !32
  %75 = bitcast i64* @.C339_MAIN_ to i8*, !dbg !32
  %76 = bitcast i32** %.Z0972_329 to i8*, !dbg !32
  %77 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !32
  %78 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !32
  %79 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !32
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %79(i8* %73, i8* %74, i8* %75, i8* null, i8* %76, i8* null, i8* %77, i8* %78, i8* null, i64 0), !dbg !32
  br label %L.LB1_331

L.LB1_331:                                        ; preds = %L.LB1_362
  %80 = load i32, i32* %len_326, align 4, !dbg !33
  call void @llvm.dbg.value(metadata i32 %80, metadata !27, metadata !DIExpression()), !dbg !10
  %81 = add nsw i32 %80, 1, !dbg !33
  store i32 %81, i32* %.i0000_333, align 4, !dbg !33
  %82 = load i32, i32* %len_326, align 4, !dbg !33
  call void @llvm.dbg.value(metadata i32 %82, metadata !27, metadata !DIExpression()), !dbg !10
  store i32 %82, i32* %.dY0001_350, align 4, !dbg !33
  call void @llvm.dbg.declare(metadata i32* %i_332, metadata !34, metadata !DIExpression()), !dbg !35
  store i32 1, i32* %i_332, align 4, !dbg !33
  %83 = load i32, i32* %.dY0001_350, align 4, !dbg !33
  %84 = icmp sle i32 %83, 0, !dbg !33
  br i1 %84, label %L.LB1_349, label %L.LB1_348, !dbg !33

L.LB1_348:                                        ; preds = %L.LB1_348, %L.LB1_331
  %85 = load i32, i32* %i_332, align 4, !dbg !36
  call void @llvm.dbg.value(metadata i32 %85, metadata !34, metadata !DIExpression()), !dbg !35
  %86 = sext i32 %85 to i64, !dbg !36
  %87 = bitcast [16 x i64]* %"c$sd3_342" to i8*, !dbg !36
  %88 = getelementptr i8, i8* %87, i64 56, !dbg !36
  %89 = bitcast i8* %88 to i64*, !dbg !36
  %90 = load i64, i64* %89, align 8, !dbg !36
  %91 = add nsw i64 %86, %90, !dbg !36
  %92 = load i32*, i32** %.Z0972_329, align 8, !dbg !36
  call void @llvm.dbg.value(metadata i32* %92, metadata !16, metadata !DIExpression()), !dbg !10
  %93 = bitcast i32* %92 to i8*, !dbg !36
  %94 = getelementptr i8, i8* %93, i64 -4, !dbg !36
  %95 = bitcast i8* %94 to i32*, !dbg !36
  %96 = getelementptr i32, i32* %95, i64 %91, !dbg !36
  %97 = load i32, i32* %96, align 4, !dbg !36
  %98 = load i32, i32* %i_332, align 4, !dbg !36
  call void @llvm.dbg.value(metadata i32 %98, metadata !34, metadata !DIExpression()), !dbg !35
  %99 = sext i32 %98 to i64, !dbg !36
  %100 = bitcast [16 x i64]* %"b$sd2_341" to i8*, !dbg !36
  %101 = getelementptr i8, i8* %100, i64 56, !dbg !36
  %102 = bitcast i8* %101 to i64*, !dbg !36
  %103 = load i64, i64* %102, align 8, !dbg !36
  %104 = add nsw i64 %99, %103, !dbg !36
  %105 = load i32*, i32** %.Z0966_328, align 8, !dbg !36
  call void @llvm.dbg.value(metadata i32* %105, metadata !25, metadata !DIExpression()), !dbg !10
  %106 = bitcast i32* %105 to i8*, !dbg !36
  %107 = getelementptr i8, i8* %106, i64 -4, !dbg !36
  %108 = bitcast i8* %107 to i32*, !dbg !36
  %109 = getelementptr i32, i32* %108, i64 %104, !dbg !36
  %110 = load i32, i32* %109, align 4, !dbg !36
  %111 = add nsw i32 %97, %110, !dbg !36
  %112 = load i32, i32* %i_332, align 4, !dbg !36
  call void @llvm.dbg.value(metadata i32 %112, metadata !34, metadata !DIExpression()), !dbg !35
  %113 = sext i32 %112 to i64, !dbg !36
  %114 = bitcast [16 x i64]* %"a$sd1_337" to i8*, !dbg !36
  %115 = getelementptr i8, i8* %114, i64 56, !dbg !36
  %116 = bitcast i8* %115 to i64*, !dbg !36
  %117 = load i64, i64* %116, align 8, !dbg !36
  %118 = add nsw i64 %113, %117, !dbg !36
  %119 = load i32*, i32** %.Z0965_327, align 8, !dbg !36
  call void @llvm.dbg.value(metadata i32* %119, metadata !26, metadata !DIExpression()), !dbg !10
  %120 = bitcast i32* %119 to i8*, !dbg !36
  %121 = getelementptr i8, i8* %120, i64 -4, !dbg !36
  %122 = bitcast i8* %121 to i32*, !dbg !36
  %123 = getelementptr i32, i32* %122, i64 %118, !dbg !36
  store i32 %111, i32* %123, align 4, !dbg !36
  %124 = load i32, i32* %i_332, align 4, !dbg !37
  call void @llvm.dbg.value(metadata i32 %124, metadata !34, metadata !DIExpression()), !dbg !35
  %125 = add nsw i32 %124, 1, !dbg !37
  store i32 %125, i32* %i_332, align 4, !dbg !37
  %126 = load i32, i32* %.dY0001_350, align 4, !dbg !37
  %127 = sub nsw i32 %126, 1, !dbg !37
  store i32 %127, i32* %.dY0001_350, align 4, !dbg !37
  %128 = load i32, i32* %.dY0001_350, align 4, !dbg !37
  %129 = icmp sgt i32 %128, 0, !dbg !37
  br i1 %129, label %L.LB1_348, label %L.LB1_349, !dbg !37

L.LB1_349:                                        ; preds = %L.LB1_348, %L.LB1_331
  br label %L.LB1_334

L.LB1_334:                                        ; preds = %L.LB1_349
  %130 = load i32*, i32** %.Z0965_327, align 8, !dbg !38
  call void @llvm.dbg.value(metadata i32* %130, metadata !26, metadata !DIExpression()), !dbg !10
  %131 = bitcast i32* %130 to i8*, !dbg !38
  %132 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !38
  %133 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !38
  call void (i8*, i8*, i8*, i8*, i64, ...) %133(i8* null, i8* %131, i8* %132, i8* null, i64 0), !dbg !38
  %134 = bitcast i32** %.Z0965_327 to i8**, !dbg !38
  store i8* null, i8** %134, align 8, !dbg !38
  %135 = bitcast [16 x i64]* %"a$sd1_337" to i64*, !dbg !38
  store i64 0, i64* %135, align 8, !dbg !38
  %136 = load i32*, i32** %.Z0966_328, align 8, !dbg !38
  call void @llvm.dbg.value(metadata i32* %136, metadata !25, metadata !DIExpression()), !dbg !10
  %137 = bitcast i32* %136 to i8*, !dbg !38
  %138 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !38
  %139 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !38
  call void (i8*, i8*, i8*, i8*, i64, ...) %139(i8* null, i8* %137, i8* %138, i8* null, i64 0), !dbg !38
  %140 = bitcast i32** %.Z0966_328 to i8**, !dbg !38
  store i8* null, i8** %140, align 8, !dbg !38
  %141 = bitcast [16 x i64]* %"b$sd2_341" to i64*, !dbg !38
  store i64 0, i64* %141, align 8, !dbg !38
  %142 = load i32*, i32** %.Z0972_329, align 8, !dbg !38
  call void @llvm.dbg.value(metadata i32* %142, metadata !16, metadata !DIExpression()), !dbg !10
  %143 = bitcast i32* %142 to i8*, !dbg !38
  %144 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !38
  %145 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !38
  call void (i8*, i8*, i8*, i8*, i64, ...) %145(i8* null, i8* %143, i8* %144, i8* null, i64 0), !dbg !38
  %146 = bitcast i32** %.Z0972_329 to i8**, !dbg !38
  store i8* null, i8** %146, align 8, !dbg !38
  %147 = bitcast [16 x i64]* %"c$sd3_342" to i64*, !dbg !38
  store i64 0, i64* %147, align 8, !dbg !38
  ret void, !dbg !35
}

declare void @f90_dealloc03a_i8(...) #0

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
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB070-simd1-orig-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb070_simd1_orig_no", scope: !2, file: !3, line: 10, type: !6, scopeLine: 10, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
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
!16 = !DILocalVariable(name: "c", scope: !5, file: !3, type: !17)
!17 = !DICompositeType(tag: DW_TAG_array_type, baseType: !9, size: 32, align: 32, elements: !18)
!18 = !{!19}
!19 = !DISubrange(count: 0, lowerBound: 1)
!20 = !DILocalVariable(scope: !5, file: !3, type: !21, flags: DIFlagArtificial)
!21 = !DICompositeType(tag: DW_TAG_array_type, baseType: !22, size: 1024, align: 64, elements: !23)
!22 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!23 = !{!24}
!24 = !DISubrange(count: 16, lowerBound: 1)
!25 = !DILocalVariable(name: "b", scope: !5, file: !3, type: !17)
!26 = !DILocalVariable(name: "a", scope: !5, file: !3, type: !17)
!27 = !DILocalVariable(name: "len", scope: !5, file: !3, type: !9)
!28 = !DILocation(line: 16, column: 1, scope: !5)
!29 = !DILocalVariable(scope: !5, file: !3, type: !22, flags: DIFlagArtificial)
!30 = !DILocation(line: 17, column: 1, scope: !5)
!31 = !DILocation(line: 18, column: 1, scope: !5)
!32 = !DILocation(line: 19, column: 1, scope: !5)
!33 = !DILocation(line: 22, column: 1, scope: !5)
!34 = !DILocalVariable(name: "i", scope: !5, file: !3, type: !9)
!35 = !DILocation(line: 28, column: 1, scope: !5)
!36 = !DILocation(line: 23, column: 1, scope: !5)
!37 = !DILocation(line: 24, column: 1, scope: !5)
!38 = !DILocation(line: 27, column: 1, scope: !5)

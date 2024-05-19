; ModuleID = '/tmp/DRB073-doall2-orig-yes-707331.ll'
source_filename = "/tmp/DRB073-doall2-orig-yes-707331.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%astruct.dt64 = type <{ i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8* }>

@.C285_MAIN_ = internal constant i32 1
@.C305_MAIN_ = internal constant i32 25
@.C332_MAIN_ = internal constant i64 4
@.C331_MAIN_ = internal constant i64 25
@.C318_MAIN_ = internal constant i32 100
@.C286_MAIN_ = internal constant i64 1
@.C284_MAIN_ = internal constant i64 0
@.C283_MAIN_ = internal constant i32 0
@.C318___nv_MAIN__F1L26_1 = internal constant i32 100
@.C285___nv_MAIN__F1L26_1 = internal constant i32 1
@.C283___nv_MAIN__F1L26_1 = internal constant i32 0

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__410 = alloca i32, align 4
  %.Z0970_320 = alloca i32*, align 8
  %"a$sd1_330" = alloca [22 x i64], align 8
  %len_319 = alloca i32, align 4
  %z_b_0_308 = alloca i64, align 8
  %z_b_1_309 = alloca i64, align 8
  %z_e_63_315 = alloca i64, align 8
  %z_b_3_311 = alloca i64, align 8
  %z_b_4_312 = alloca i64, align 8
  %z_e_66_316 = alloca i64, align 8
  %z_b_2_310 = alloca i64, align 8
  %z_b_5_313 = alloca i64, align 8
  %z_b_6_314 = alloca i64, align 8
  %j_307 = alloca i32, align 4
  %.uplevelArgPack0001_380 = alloca %astruct.dt64, align 16
  call void @llvm.dbg.value(metadata i32 1, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !14, metadata !DIExpression()), !dbg !10
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !15
  store i32 %0, i32* %__gtid_MAIN__410, align 4, !dbg !15
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !16
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !16
  call void (i8*, ...) %2(i8* %1), !dbg !16
  call void @llvm.dbg.declare(metadata i32** %.Z0970_320, metadata !17, metadata !DIExpression(DW_OP_deref)), !dbg !10
  %3 = bitcast i32** %.Z0970_320 to i8**, !dbg !16
  store i8* null, i8** %3, align 8, !dbg !16
  call void @llvm.dbg.declare(metadata [22 x i64]* %"a$sd1_330", metadata !21, metadata !DIExpression()), !dbg !10
  %4 = bitcast [22 x i64]* %"a$sd1_330" to i64*, !dbg !16
  store i64 0, i64* %4, align 8, !dbg !16
  br label %L.LB1_361

L.LB1_361:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %len_319, metadata !26, metadata !DIExpression()), !dbg !10
  store i32 100, i32* %len_319, align 4, !dbg !27
  call void @llvm.dbg.declare(metadata i64* %z_b_0_308, metadata !28, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_0_308, align 8, !dbg !29
  %5 = load i32, i32* %len_319, align 4, !dbg !29
  call void @llvm.dbg.value(metadata i32 %5, metadata !26, metadata !DIExpression()), !dbg !10
  %6 = sext i32 %5 to i64, !dbg !29
  call void @llvm.dbg.declare(metadata i64* %z_b_1_309, metadata !28, metadata !DIExpression()), !dbg !10
  store i64 %6, i64* %z_b_1_309, align 8, !dbg !29
  %7 = load i64, i64* %z_b_1_309, align 8, !dbg !29
  call void @llvm.dbg.value(metadata i64 %7, metadata !28, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_63_315, metadata !28, metadata !DIExpression()), !dbg !10
  store i64 %7, i64* %z_e_63_315, align 8, !dbg !29
  call void @llvm.dbg.declare(metadata i64* %z_b_3_311, metadata !28, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_3_311, align 8, !dbg !29
  %8 = load i32, i32* %len_319, align 4, !dbg !29
  call void @llvm.dbg.value(metadata i32 %8, metadata !26, metadata !DIExpression()), !dbg !10
  %9 = sext i32 %8 to i64, !dbg !29
  call void @llvm.dbg.declare(metadata i64* %z_b_4_312, metadata !28, metadata !DIExpression()), !dbg !10
  store i64 %9, i64* %z_b_4_312, align 8, !dbg !29
  %10 = load i64, i64* %z_b_4_312, align 8, !dbg !29
  call void @llvm.dbg.value(metadata i64 %10, metadata !28, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_66_316, metadata !28, metadata !DIExpression()), !dbg !10
  store i64 %10, i64* %z_e_66_316, align 8, !dbg !29
  %11 = bitcast [22 x i64]* %"a$sd1_330" to i8*, !dbg !29
  %12 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !29
  %13 = bitcast i64* @.C331_MAIN_ to i8*, !dbg !29
  %14 = bitcast i64* @.C332_MAIN_ to i8*, !dbg !29
  %15 = bitcast i64* %z_b_0_308 to i8*, !dbg !29
  %16 = bitcast i64* %z_b_1_309 to i8*, !dbg !29
  %17 = bitcast i64* %z_b_3_311 to i8*, !dbg !29
  %18 = bitcast i64* %z_b_4_312 to i8*, !dbg !29
  %19 = bitcast void (...)* @f90_template2_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !29
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, ...) %19(i8* %11, i8* %12, i8* %13, i8* %14, i8* %15, i8* %16, i8* %17, i8* %18), !dbg !29
  %20 = bitcast [22 x i64]* %"a$sd1_330" to i8*, !dbg !29
  %21 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !29
  call void (i8*, i32, ...) %21(i8* %20, i32 25), !dbg !29
  %22 = load i64, i64* %z_b_1_309, align 8, !dbg !29
  call void @llvm.dbg.value(metadata i64 %22, metadata !28, metadata !DIExpression()), !dbg !10
  %23 = load i64, i64* %z_b_0_308, align 8, !dbg !29
  call void @llvm.dbg.value(metadata i64 %23, metadata !28, metadata !DIExpression()), !dbg !10
  %24 = sub nsw i64 %23, 1, !dbg !29
  %25 = sub nsw i64 %22, %24, !dbg !29
  call void @llvm.dbg.declare(metadata i64* %z_b_2_310, metadata !28, metadata !DIExpression()), !dbg !10
  store i64 %25, i64* %z_b_2_310, align 8, !dbg !29
  %26 = load i64, i64* %z_b_1_309, align 8, !dbg !29
  call void @llvm.dbg.value(metadata i64 %26, metadata !28, metadata !DIExpression()), !dbg !10
  %27 = load i64, i64* %z_b_0_308, align 8, !dbg !29
  call void @llvm.dbg.value(metadata i64 %27, metadata !28, metadata !DIExpression()), !dbg !10
  %28 = sub nsw i64 %27, 1, !dbg !29
  %29 = sub nsw i64 %26, %28, !dbg !29
  %30 = load i64, i64* %z_b_4_312, align 8, !dbg !29
  call void @llvm.dbg.value(metadata i64 %30, metadata !28, metadata !DIExpression()), !dbg !10
  %31 = load i64, i64* %z_b_3_311, align 8, !dbg !29
  call void @llvm.dbg.value(metadata i64 %31, metadata !28, metadata !DIExpression()), !dbg !10
  %32 = sub nsw i64 %31, 1, !dbg !29
  %33 = sub nsw i64 %30, %32, !dbg !29
  %34 = mul nsw i64 %29, %33, !dbg !29
  call void @llvm.dbg.declare(metadata i64* %z_b_5_313, metadata !28, metadata !DIExpression()), !dbg !10
  store i64 %34, i64* %z_b_5_313, align 8, !dbg !29
  %35 = load i64, i64* %z_b_0_308, align 8, !dbg !29
  call void @llvm.dbg.value(metadata i64 %35, metadata !28, metadata !DIExpression()), !dbg !10
  %36 = load i64, i64* %z_b_1_309, align 8, !dbg !29
  call void @llvm.dbg.value(metadata i64 %36, metadata !28, metadata !DIExpression()), !dbg !10
  %37 = load i64, i64* %z_b_0_308, align 8, !dbg !29
  call void @llvm.dbg.value(metadata i64 %37, metadata !28, metadata !DIExpression()), !dbg !10
  %38 = sub nsw i64 %37, 1, !dbg !29
  %39 = sub nsw i64 %36, %38, !dbg !29
  %40 = load i64, i64* %z_b_3_311, align 8, !dbg !29
  call void @llvm.dbg.value(metadata i64 %40, metadata !28, metadata !DIExpression()), !dbg !10
  %41 = mul nsw i64 %39, %40, !dbg !29
  %42 = add nsw i64 %35, %41, !dbg !29
  call void @llvm.dbg.declare(metadata i64* %z_b_6_314, metadata !28, metadata !DIExpression()), !dbg !10
  store i64 %42, i64* %z_b_6_314, align 8, !dbg !29
  %43 = bitcast i64* %z_b_5_313 to i8*, !dbg !29
  %44 = bitcast i64* @.C331_MAIN_ to i8*, !dbg !29
  %45 = bitcast i64* @.C332_MAIN_ to i8*, !dbg !29
  %46 = bitcast i32** %.Z0970_320 to i8*, !dbg !29
  %47 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !29
  %48 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !29
  %49 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !29
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %49(i8* %43, i8* %44, i8* %45, i8* null, i8* %46, i8* null, i8* %47, i8* %48, i8* null, i64 0), !dbg !29
  call void @llvm.dbg.declare(metadata i32* %j_307, metadata !30, metadata !DIExpression()), !dbg !10
  %50 = bitcast i32* %j_307 to i8*, !dbg !31
  %51 = bitcast %astruct.dt64* %.uplevelArgPack0001_380 to i8**, !dbg !31
  store i8* %50, i8** %51, align 8, !dbg !31
  %52 = bitcast i32** %.Z0970_320 to i8*, !dbg !31
  %53 = bitcast %astruct.dt64* %.uplevelArgPack0001_380 to i8*, !dbg !31
  %54 = getelementptr i8, i8* %53, i64 8, !dbg !31
  %55 = bitcast i8* %54 to i8**, !dbg !31
  store i8* %52, i8** %55, align 8, !dbg !31
  %56 = bitcast i32** %.Z0970_320 to i8*, !dbg !31
  %57 = bitcast %astruct.dt64* %.uplevelArgPack0001_380 to i8*, !dbg !31
  %58 = getelementptr i8, i8* %57, i64 16, !dbg !31
  %59 = bitcast i8* %58 to i8**, !dbg !31
  store i8* %56, i8** %59, align 8, !dbg !31
  %60 = bitcast i64* %z_b_0_308 to i8*, !dbg !31
  %61 = bitcast %astruct.dt64* %.uplevelArgPack0001_380 to i8*, !dbg !31
  %62 = getelementptr i8, i8* %61, i64 24, !dbg !31
  %63 = bitcast i8* %62 to i8**, !dbg !31
  store i8* %60, i8** %63, align 8, !dbg !31
  %64 = bitcast i64* %z_b_1_309 to i8*, !dbg !31
  %65 = bitcast %astruct.dt64* %.uplevelArgPack0001_380 to i8*, !dbg !31
  %66 = getelementptr i8, i8* %65, i64 32, !dbg !31
  %67 = bitcast i8* %66 to i8**, !dbg !31
  store i8* %64, i8** %67, align 8, !dbg !31
  %68 = bitcast i64* %z_e_63_315 to i8*, !dbg !31
  %69 = bitcast %astruct.dt64* %.uplevelArgPack0001_380 to i8*, !dbg !31
  %70 = getelementptr i8, i8* %69, i64 40, !dbg !31
  %71 = bitcast i8* %70 to i8**, !dbg !31
  store i8* %68, i8** %71, align 8, !dbg !31
  %72 = bitcast i64* %z_b_3_311 to i8*, !dbg !31
  %73 = bitcast %astruct.dt64* %.uplevelArgPack0001_380 to i8*, !dbg !31
  %74 = getelementptr i8, i8* %73, i64 48, !dbg !31
  %75 = bitcast i8* %74 to i8**, !dbg !31
  store i8* %72, i8** %75, align 8, !dbg !31
  %76 = bitcast i64* %z_b_4_312 to i8*, !dbg !31
  %77 = bitcast %astruct.dt64* %.uplevelArgPack0001_380 to i8*, !dbg !31
  %78 = getelementptr i8, i8* %77, i64 56, !dbg !31
  %79 = bitcast i8* %78 to i8**, !dbg !31
  store i8* %76, i8** %79, align 8, !dbg !31
  %80 = bitcast i64* %z_b_2_310 to i8*, !dbg !31
  %81 = bitcast %astruct.dt64* %.uplevelArgPack0001_380 to i8*, !dbg !31
  %82 = getelementptr i8, i8* %81, i64 64, !dbg !31
  %83 = bitcast i8* %82 to i8**, !dbg !31
  store i8* %80, i8** %83, align 8, !dbg !31
  %84 = bitcast i64* %z_e_66_316 to i8*, !dbg !31
  %85 = bitcast %astruct.dt64* %.uplevelArgPack0001_380 to i8*, !dbg !31
  %86 = getelementptr i8, i8* %85, i64 72, !dbg !31
  %87 = bitcast i8* %86 to i8**, !dbg !31
  store i8* %84, i8** %87, align 8, !dbg !31
  %88 = bitcast i64* %z_b_5_313 to i8*, !dbg !31
  %89 = bitcast %astruct.dt64* %.uplevelArgPack0001_380 to i8*, !dbg !31
  %90 = getelementptr i8, i8* %89, i64 80, !dbg !31
  %91 = bitcast i8* %90 to i8**, !dbg !31
  store i8* %88, i8** %91, align 8, !dbg !31
  %92 = bitcast i64* %z_b_6_314 to i8*, !dbg !31
  %93 = bitcast %astruct.dt64* %.uplevelArgPack0001_380 to i8*, !dbg !31
  %94 = getelementptr i8, i8* %93, i64 88, !dbg !31
  %95 = bitcast i8* %94 to i8**, !dbg !31
  store i8* %92, i8** %95, align 8, !dbg !31
  %96 = bitcast [22 x i64]* %"a$sd1_330" to i8*, !dbg !31
  %97 = bitcast %astruct.dt64* %.uplevelArgPack0001_380 to i8*, !dbg !31
  %98 = getelementptr i8, i8* %97, i64 96, !dbg !31
  %99 = bitcast i8* %98 to i8**, !dbg !31
  store i8* %96, i8** %99, align 8, !dbg !31
  br label %L.LB1_408, !dbg !31

L.LB1_408:                                        ; preds = %L.LB1_361
  %100 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L26_1_ to i64*, !dbg !31
  %101 = bitcast %astruct.dt64* %.uplevelArgPack0001_380 to i64*, !dbg !31
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %100, i64* %101), !dbg !31
  %102 = load i32*, i32** %.Z0970_320, align 8, !dbg !32
  call void @llvm.dbg.value(metadata i32* %102, metadata !17, metadata !DIExpression()), !dbg !10
  %103 = bitcast i32* %102 to i8*, !dbg !32
  %104 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !32
  %105 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !32
  call void (i8*, i8*, i8*, i8*, i64, ...) %105(i8* null, i8* %103, i8* %104, i8* null, i64 0), !dbg !32
  %106 = bitcast i32** %.Z0970_320 to i8**, !dbg !32
  store i8* null, i8** %106, align 8, !dbg !32
  %107 = bitcast [22 x i64]* %"a$sd1_330" to i64*, !dbg !32
  store i64 0, i64* %107, align 8, !dbg !32
  ret void, !dbg !15
}

define internal void @__nv_MAIN__F1L26_1_(i32* %__nv_MAIN__F1L26_1Arg0, i64* %__nv_MAIN__F1L26_1Arg1, i64* %__nv_MAIN__F1L26_1Arg2) #0 !dbg !33 {
L.entry:
  %__gtid___nv_MAIN__F1L26_1__445 = alloca i32, align 4
  %.i0000p_325 = alloca i32, align 4
  %i_324 = alloca i32, align 4
  %.du0001p_342 = alloca i32, align 4
  %.de0001p_343 = alloca i32, align 4
  %.di0001p_344 = alloca i32, align 4
  %.ds0001p_345 = alloca i32, align 4
  %.dl0001p_347 = alloca i32, align 4
  %.dl0001p.copy_439 = alloca i32, align 4
  %.de0001p.copy_440 = alloca i32, align 4
  %.ds0001p.copy_441 = alloca i32, align 4
  %.dX0001p_346 = alloca i32, align 4
  %.dY0001p_341 = alloca i32, align 4
  %.dY0002p_353 = alloca i32, align 4
  %j_326 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L26_1Arg0, metadata !36, metadata !DIExpression()), !dbg !37
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L26_1Arg1, metadata !38, metadata !DIExpression()), !dbg !37
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L26_1Arg2, metadata !39, metadata !DIExpression()), !dbg !37
  call void @llvm.dbg.value(metadata i32 1, metadata !40, metadata !DIExpression()), !dbg !37
  call void @llvm.dbg.value(metadata i32 0, metadata !41, metadata !DIExpression()), !dbg !37
  call void @llvm.dbg.value(metadata i32 1, metadata !42, metadata !DIExpression()), !dbg !37
  call void @llvm.dbg.value(metadata i32 0, metadata !43, metadata !DIExpression()), !dbg !37
  call void @llvm.dbg.value(metadata i32 1, metadata !44, metadata !DIExpression()), !dbg !37
  %0 = load i32, i32* %__nv_MAIN__F1L26_1Arg0, align 4, !dbg !45
  store i32 %0, i32* %__gtid___nv_MAIN__F1L26_1__445, align 4, !dbg !45
  br label %L.LB2_431

L.LB2_431:                                        ; preds = %L.entry
  br label %L.LB2_323

L.LB2_323:                                        ; preds = %L.LB2_431
  store i32 0, i32* %.i0000p_325, align 4, !dbg !46
  call void @llvm.dbg.declare(metadata i32* %i_324, metadata !47, metadata !DIExpression()), !dbg !45
  store i32 1, i32* %i_324, align 4, !dbg !46
  store i32 100, i32* %.du0001p_342, align 4, !dbg !46
  store i32 100, i32* %.de0001p_343, align 4, !dbg !46
  store i32 1, i32* %.di0001p_344, align 4, !dbg !46
  %1 = load i32, i32* %.di0001p_344, align 4, !dbg !46
  store i32 %1, i32* %.ds0001p_345, align 4, !dbg !46
  store i32 1, i32* %.dl0001p_347, align 4, !dbg !46
  %2 = load i32, i32* %.dl0001p_347, align 4, !dbg !46
  store i32 %2, i32* %.dl0001p.copy_439, align 4, !dbg !46
  %3 = load i32, i32* %.de0001p_343, align 4, !dbg !46
  store i32 %3, i32* %.de0001p.copy_440, align 4, !dbg !46
  %4 = load i32, i32* %.ds0001p_345, align 4, !dbg !46
  store i32 %4, i32* %.ds0001p.copy_441, align 4, !dbg !46
  %5 = load i32, i32* %__gtid___nv_MAIN__F1L26_1__445, align 4, !dbg !46
  %6 = bitcast i32* %.i0000p_325 to i64*, !dbg !46
  %7 = bitcast i32* %.dl0001p.copy_439 to i64*, !dbg !46
  %8 = bitcast i32* %.de0001p.copy_440 to i64*, !dbg !46
  %9 = bitcast i32* %.ds0001p.copy_441 to i64*, !dbg !46
  %10 = load i32, i32* %.ds0001p.copy_441, align 4, !dbg !46
  call void @__kmpc_for_static_init_4(i64* null, i32 %5, i32 34, i64* %6, i64* %7, i64* %8, i64* %9, i32 %10, i32 1), !dbg !46
  %11 = load i32, i32* %.dl0001p.copy_439, align 4, !dbg !46
  store i32 %11, i32* %.dl0001p_347, align 4, !dbg !46
  %12 = load i32, i32* %.de0001p.copy_440, align 4, !dbg !46
  store i32 %12, i32* %.de0001p_343, align 4, !dbg !46
  %13 = load i32, i32* %.ds0001p.copy_441, align 4, !dbg !46
  store i32 %13, i32* %.ds0001p_345, align 4, !dbg !46
  %14 = load i32, i32* %.dl0001p_347, align 4, !dbg !46
  store i32 %14, i32* %i_324, align 4, !dbg !46
  %15 = load i32, i32* %i_324, align 4, !dbg !46
  call void @llvm.dbg.value(metadata i32 %15, metadata !47, metadata !DIExpression()), !dbg !45
  store i32 %15, i32* %.dX0001p_346, align 4, !dbg !46
  %16 = load i32, i32* %.dX0001p_346, align 4, !dbg !46
  %17 = load i32, i32* %.du0001p_342, align 4, !dbg !46
  %18 = icmp sgt i32 %16, %17, !dbg !46
  br i1 %18, label %L.LB2_340, label %L.LB2_477, !dbg !46

L.LB2_477:                                        ; preds = %L.LB2_323
  %19 = load i32, i32* %.dX0001p_346, align 4, !dbg !46
  store i32 %19, i32* %i_324, align 4, !dbg !46
  %20 = load i32, i32* %.di0001p_344, align 4, !dbg !46
  %21 = load i32, i32* %.de0001p_343, align 4, !dbg !46
  %22 = load i32, i32* %.dX0001p_346, align 4, !dbg !46
  %23 = sub nsw i32 %21, %22, !dbg !46
  %24 = add nsw i32 %20, %23, !dbg !46
  %25 = load i32, i32* %.di0001p_344, align 4, !dbg !46
  %26 = sdiv i32 %24, %25, !dbg !46
  store i32 %26, i32* %.dY0001p_341, align 4, !dbg !46
  %27 = load i32, i32* %.dY0001p_341, align 4, !dbg !46
  %28 = icmp sle i32 %27, 0, !dbg !46
  br i1 %28, label %L.LB2_350, label %L.LB2_349, !dbg !46

L.LB2_349:                                        ; preds = %L.LB2_478, %L.LB2_477
  store i32 100, i32* %.dY0002p_353, align 4, !dbg !48
  call void @llvm.dbg.declare(metadata i32* %j_326, metadata !49, metadata !DIExpression()), !dbg !45
  store i32 1, i32* %j_326, align 4, !dbg !48
  br label %L.LB2_351

L.LB2_351:                                        ; preds = %L.LB2_351, %L.LB2_349
  %29 = load i32, i32* %i_324, align 4, !dbg !50
  call void @llvm.dbg.value(metadata i32 %29, metadata !47, metadata !DIExpression()), !dbg !45
  %30 = sext i32 %29 to i64, !dbg !50
  %31 = load i32, i32* %j_326, align 4, !dbg !50
  call void @llvm.dbg.value(metadata i32 %31, metadata !49, metadata !DIExpression()), !dbg !45
  %32 = sext i32 %31 to i64, !dbg !50
  %33 = bitcast i64* %__nv_MAIN__F1L26_1Arg2 to i8*, !dbg !50
  %34 = getelementptr i8, i8* %33, i64 96, !dbg !50
  %35 = bitcast i8* %34 to i8**, !dbg !50
  %36 = load i8*, i8** %35, align 8, !dbg !50
  %37 = getelementptr i8, i8* %36, i64 160, !dbg !50
  %38 = bitcast i8* %37 to i64*, !dbg !50
  %39 = load i64, i64* %38, align 8, !dbg !50
  %40 = mul nsw i64 %32, %39, !dbg !50
  %41 = add nsw i64 %30, %40, !dbg !50
  %42 = bitcast i64* %__nv_MAIN__F1L26_1Arg2 to i8*, !dbg !50
  %43 = getelementptr i8, i8* %42, i64 96, !dbg !50
  %44 = bitcast i8* %43 to i8**, !dbg !50
  %45 = load i8*, i8** %44, align 8, !dbg !50
  %46 = getelementptr i8, i8* %45, i64 56, !dbg !50
  %47 = bitcast i8* %46 to i64*, !dbg !50
  %48 = load i64, i64* %47, align 8, !dbg !50
  %49 = add nsw i64 %41, %48, !dbg !50
  %50 = bitcast i64* %__nv_MAIN__F1L26_1Arg2 to i8*, !dbg !50
  %51 = getelementptr i8, i8* %50, i64 16, !dbg !50
  %52 = bitcast i8* %51 to i8***, !dbg !50
  %53 = load i8**, i8*** %52, align 8, !dbg !50
  %54 = load i8*, i8** %53, align 8, !dbg !50
  %55 = getelementptr i8, i8* %54, i64 -4, !dbg !50
  %56 = bitcast i8* %55 to i32*, !dbg !50
  %57 = getelementptr i32, i32* %56, i64 %49, !dbg !50
  %58 = load i32, i32* %57, align 4, !dbg !50
  %59 = add nsw i32 %58, 1, !dbg !50
  %60 = load i32, i32* %i_324, align 4, !dbg !50
  call void @llvm.dbg.value(metadata i32 %60, metadata !47, metadata !DIExpression()), !dbg !45
  %61 = sext i32 %60 to i64, !dbg !50
  %62 = load i32, i32* %j_326, align 4, !dbg !50
  call void @llvm.dbg.value(metadata i32 %62, metadata !49, metadata !DIExpression()), !dbg !45
  %63 = sext i32 %62 to i64, !dbg !50
  %64 = bitcast i64* %__nv_MAIN__F1L26_1Arg2 to i8*, !dbg !50
  %65 = getelementptr i8, i8* %64, i64 96, !dbg !50
  %66 = bitcast i8* %65 to i8**, !dbg !50
  %67 = load i8*, i8** %66, align 8, !dbg !50
  %68 = getelementptr i8, i8* %67, i64 160, !dbg !50
  %69 = bitcast i8* %68 to i64*, !dbg !50
  %70 = load i64, i64* %69, align 8, !dbg !50
  %71 = mul nsw i64 %63, %70, !dbg !50
  %72 = add nsw i64 %61, %71, !dbg !50
  %73 = bitcast i64* %__nv_MAIN__F1L26_1Arg2 to i8*, !dbg !50
  %74 = getelementptr i8, i8* %73, i64 96, !dbg !50
  %75 = bitcast i8* %74 to i8**, !dbg !50
  %76 = load i8*, i8** %75, align 8, !dbg !50
  %77 = getelementptr i8, i8* %76, i64 56, !dbg !50
  %78 = bitcast i8* %77 to i64*, !dbg !50
  %79 = load i64, i64* %78, align 8, !dbg !50
  %80 = add nsw i64 %72, %79, !dbg !50
  %81 = bitcast i64* %__nv_MAIN__F1L26_1Arg2 to i8*, !dbg !50
  %82 = getelementptr i8, i8* %81, i64 16, !dbg !50
  %83 = bitcast i8* %82 to i8***, !dbg !50
  %84 = load i8**, i8*** %83, align 8, !dbg !50
  %85 = load i8*, i8** %84, align 8, !dbg !50
  %86 = getelementptr i8, i8* %85, i64 -4, !dbg !50
  %87 = bitcast i8* %86 to i32*, !dbg !50
  %88 = getelementptr i32, i32* %87, i64 %80, !dbg !50
  store i32 %59, i32* %88, align 4, !dbg !50
  %89 = load i32, i32* %j_326, align 4, !dbg !51
  call void @llvm.dbg.value(metadata i32 %89, metadata !49, metadata !DIExpression()), !dbg !45
  %90 = add nsw i32 %89, 1, !dbg !51
  store i32 %90, i32* %j_326, align 4, !dbg !51
  %91 = load i32, i32* %.dY0002p_353, align 4, !dbg !51
  %92 = sub nsw i32 %91, 1, !dbg !51
  store i32 %92, i32* %.dY0002p_353, align 4, !dbg !51
  %93 = load i32, i32* %.dY0002p_353, align 4, !dbg !51
  %94 = icmp sgt i32 %93, 0, !dbg !51
  br i1 %94, label %L.LB2_351, label %L.LB2_478, !dbg !51

L.LB2_478:                                        ; preds = %L.LB2_351
  %95 = load i32, i32* %.di0001p_344, align 4, !dbg !45
  %96 = load i32, i32* %i_324, align 4, !dbg !45
  call void @llvm.dbg.value(metadata i32 %96, metadata !47, metadata !DIExpression()), !dbg !45
  %97 = add nsw i32 %95, %96, !dbg !45
  store i32 %97, i32* %i_324, align 4, !dbg !45
  %98 = load i32, i32* %.dY0001p_341, align 4, !dbg !45
  %99 = sub nsw i32 %98, 1, !dbg !45
  store i32 %99, i32* %.dY0001p_341, align 4, !dbg !45
  %100 = load i32, i32* %.dY0001p_341, align 4, !dbg !45
  %101 = icmp sgt i32 %100, 0, !dbg !45
  br i1 %101, label %L.LB2_349, label %L.LB2_350, !dbg !45

L.LB2_350:                                        ; preds = %L.LB2_478, %L.LB2_477
  br label %L.LB2_340

L.LB2_340:                                        ; preds = %L.LB2_350, %L.LB2_323
  %102 = load i32, i32* %__gtid___nv_MAIN__F1L26_1__445, align 4, !dbg !45
  call void @__kmpc_for_static_fini(i64* null, i32 %102), !dbg !45
  br label %L.LB2_327

L.LB2_327:                                        ; preds = %L.LB2_340
  ret void, !dbg !45
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
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB073-doall2-orig-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb073_doall2_orig_yes", scope: !2, file: !3, line: 16, type: !6, scopeLine: 16, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
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
!16 = !DILocation(line: 16, column: 1, scope: !5)
!17 = !DILocalVariable(name: "a", scope: !5, file: !3, type: !18)
!18 = !DICompositeType(tag: DW_TAG_array_type, baseType: !9, size: 32, align: 32, elements: !19)
!19 = !{!20, !20}
!20 = !DISubrange(count: 0, lowerBound: 1)
!21 = !DILocalVariable(scope: !5, file: !3, type: !22, flags: DIFlagArtificial)
!22 = !DICompositeType(tag: DW_TAG_array_type, baseType: !23, size: 1408, align: 64, elements: !24)
!23 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!24 = !{!25}
!25 = !DISubrange(count: 22, lowerBound: 1)
!26 = !DILocalVariable(name: "len", scope: !5, file: !3, type: !9)
!27 = !DILocation(line: 22, column: 1, scope: !5)
!28 = !DILocalVariable(scope: !5, file: !3, type: !23, flags: DIFlagArtificial)
!29 = !DILocation(line: 24, column: 1, scope: !5)
!30 = !DILocalVariable(name: "j", scope: !5, file: !3, type: !9)
!31 = !DILocation(line: 26, column: 1, scope: !5)
!32 = !DILocation(line: 35, column: 1, scope: !5)
!33 = distinct !DISubprogram(name: "__nv_MAIN__F1L26_1", scope: !2, file: !3, line: 26, type: !34, scopeLine: 26, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!34 = !DISubroutineType(types: !35)
!35 = !{null, !9, !23, !23}
!36 = !DILocalVariable(name: "__nv_MAIN__F1L26_1Arg0", arg: 1, scope: !33, file: !3, type: !9)
!37 = !DILocation(line: 0, scope: !33)
!38 = !DILocalVariable(name: "__nv_MAIN__F1L26_1Arg1", arg: 2, scope: !33, file: !3, type: !23)
!39 = !DILocalVariable(name: "__nv_MAIN__F1L26_1Arg2", arg: 3, scope: !33, file: !3, type: !23)
!40 = !DILocalVariable(name: "omp_sched_static", scope: !33, file: !3, type: !9)
!41 = !DILocalVariable(name: "omp_proc_bind_false", scope: !33, file: !3, type: !9)
!42 = !DILocalVariable(name: "omp_proc_bind_true", scope: !33, file: !3, type: !9)
!43 = !DILocalVariable(name: "omp_lock_hint_none", scope: !33, file: !3, type: !9)
!44 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !33, file: !3, type: !9)
!45 = !DILocation(line: 31, column: 1, scope: !33)
!46 = !DILocation(line: 27, column: 1, scope: !33)
!47 = !DILocalVariable(name: "i", scope: !33, file: !3, type: !9)
!48 = !DILocation(line: 28, column: 1, scope: !33)
!49 = !DILocalVariable(name: "j", scope: !33, file: !3, type: !9)
!50 = !DILocation(line: 29, column: 1, scope: !33)
!51 = !DILocation(line: 30, column: 1, scope: !33)

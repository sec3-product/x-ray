; ModuleID = '/tmp/DRB062-matrixvector2-orig-no-5d8fd5.ll'
source_filename = "/tmp/DRB062-matrixvector2-orig-no-5d8fd5.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.struct_ul_MAIN__297 = type <{ i8* }>
%astruct.dt88 = type <{ i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8* }>

@.C283_MAIN_ = internal constant i32 0
@.C346_drb062_matrixvector2_orig_no_foo = internal constant i32 6
@.C343_drb062_matrixvector2_orig_no_foo = internal constant [57 x i8] c"micro-benchmarks-fortran/DRB062-matrixvector2-orig-no.f95"
@.C345_drb062_matrixvector2_orig_no_foo = internal constant i32 32
@.C287_drb062_matrixvector2_orig_no_foo = internal constant float 0.000000e+00
@.C285_drb062_matrixvector2_orig_no_foo = internal constant i32 1
@.C303_drb062_matrixvector2_orig_no_foo = internal constant i32 27
@.C358_drb062_matrixvector2_orig_no_foo = internal constant i64 4
@.C357_drb062_matrixvector2_orig_no_foo = internal constant i64 27
@.C331_drb062_matrixvector2_orig_no_foo = internal constant i32 1000
@.C283_drb062_matrixvector2_orig_no_foo = internal constant i32 0
@.C286_drb062_matrixvector2_orig_no_foo = internal constant i64 1
@.C284_drb062_matrixvector2_orig_no_foo = internal constant i64 0
@.C303___nv_drb062_matrixvector2_orig_no_foo_F1L29_1 = internal constant i32 27
@.C284___nv_drb062_matrixvector2_orig_no_foo_F1L29_1 = internal constant i64 0
@.C346___nv_drb062_matrixvector2_orig_no_foo_F1L29_1 = internal constant i32 6
@.C343___nv_drb062_matrixvector2_orig_no_foo_F1L29_1 = internal constant [57 x i8] c"micro-benchmarks-fortran/DRB062-matrixvector2-orig-no.f95"
@.C345___nv_drb062_matrixvector2_orig_no_foo_F1L29_1 = internal constant i32 32
@.C285___nv_drb062_matrixvector2_orig_no_foo_F1L29_1 = internal constant i32 1
@.C283___nv_drb062_matrixvector2_orig_no_foo_F1L29_1 = internal constant i32 0
@.C287___nv_drb062_matrixvector2_orig_no_foo_F1L29_1 = internal constant float 0.000000e+00

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
  call void @drb062_matrixvector2_orig_no_foo(i64* %2), !dbg !16
  ret void, !dbg !17
}

define internal void @drb062_matrixvector2_orig_no_foo(i64* %.S0000) #0 !dbg !18 {
L.entry:
  %__gtid_drb062_matrixvector2_orig_no_foo_484 = alloca i32, align 4
  %.Z0984_334 = alloca float*, align 8
  %"v_out$sd3_362" = alloca [16 x i64], align 8
  %.Z0983_333 = alloca float*, align 8
  %"v$sd2_361" = alloca [16 x i64], align 8
  %.Z0977_332 = alloca float*, align 8
  %"a$sd1_356" = alloca [22 x i64], align 8
  %n_308 = alloca i32, align 4
  %z_b_0_309 = alloca i64, align 8
  %z_b_1_310 = alloca i64, align 8
  %z_e_67_316 = alloca i64, align 8
  %z_b_3_312 = alloca i64, align 8
  %z_b_4_313 = alloca i64, align 8
  %z_e_70_317 = alloca i64, align 8
  %z_b_2_311 = alloca i64, align 8
  %z_b_5_314 = alloca i64, align 8
  %z_b_6_315 = alloca i64, align 8
  %z_b_7_319 = alloca i64, align 8
  %z_b_8_320 = alloca i64, align 8
  %z_e_77_323 = alloca i64, align 8
  %z_b_9_321 = alloca i64, align 8
  %z_b_10_322 = alloca i64, align 8
  %z_b_11_326 = alloca i64, align 8
  %z_b_12_327 = alloca i64, align 8
  %z_e_84_330 = alloca i64, align 8
  %z_b_13_328 = alloca i64, align 8
  %z_b_14_329 = alloca i64, align 8
  %.dY0001_369 = alloca i32, align 4
  %i_306 = alloca i32, align 4
  %sum_335 = alloca float, align 4
  %.uplevelArgPack0001_432 = alloca %astruct.dt88, align 16
  call void @llvm.dbg.declare(metadata i64* %.S0000, metadata !20, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i32 1, metadata !23, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i32 0, metadata !24, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i32 1, metadata !25, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i32 0, metadata !26, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i32 1, metadata !27, metadata !DIExpression()), !dbg !22
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !28
  store i32 %0, i32* %__gtid_drb062_matrixvector2_orig_no_foo_484, align 4, !dbg !28
  call void @llvm.dbg.declare(metadata float** %.Z0984_334, metadata !29, metadata !DIExpression(DW_OP_deref)), !dbg !22
  %1 = bitcast float** %.Z0984_334 to i8**, !dbg !34
  store i8* null, i8** %1, align 8, !dbg !34
  call void @llvm.dbg.declare(metadata [16 x i64]* %"v_out$sd3_362", metadata !35, metadata !DIExpression()), !dbg !22
  %2 = bitcast [16 x i64]* %"v_out$sd3_362" to i64*, !dbg !34
  store i64 0, i64* %2, align 8, !dbg !34
  call void @llvm.dbg.declare(metadata float** %.Z0983_333, metadata !40, metadata !DIExpression(DW_OP_deref)), !dbg !22
  %3 = bitcast float** %.Z0983_333 to i8**, !dbg !34
  store i8* null, i8** %3, align 8, !dbg !34
  call void @llvm.dbg.declare(metadata [16 x i64]* %"v$sd2_361", metadata !35, metadata !DIExpression()), !dbg !22
  %4 = bitcast [16 x i64]* %"v$sd2_361" to i64*, !dbg !34
  store i64 0, i64* %4, align 8, !dbg !34
  call void @llvm.dbg.declare(metadata float** %.Z0977_332, metadata !41, metadata !DIExpression(DW_OP_deref)), !dbg !22
  %5 = bitcast float** %.Z0977_332 to i8**, !dbg !34
  store i8* null, i8** %5, align 8, !dbg !34
  call void @llvm.dbg.declare(metadata [22 x i64]* %"a$sd1_356", metadata !44, metadata !DIExpression()), !dbg !22
  %6 = bitcast [22 x i64]* %"a$sd1_356" to i64*, !dbg !34
  store i64 0, i64* %6, align 8, !dbg !34
  br label %L.LB2_400

L.LB2_400:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %n_308, metadata !48, metadata !DIExpression()), !dbg !22
  store i32 1000, i32* %n_308, align 4, !dbg !49
  call void @llvm.dbg.declare(metadata i64* %z_b_0_309, metadata !50, metadata !DIExpression()), !dbg !22
  store i64 1, i64* %z_b_0_309, align 8, !dbg !51
  %7 = load i32, i32* %n_308, align 4, !dbg !51
  call void @llvm.dbg.value(metadata i32 %7, metadata !48, metadata !DIExpression()), !dbg !22
  %8 = sext i32 %7 to i64, !dbg !51
  call void @llvm.dbg.declare(metadata i64* %z_b_1_310, metadata !50, metadata !DIExpression()), !dbg !22
  store i64 %8, i64* %z_b_1_310, align 8, !dbg !51
  %9 = load i64, i64* %z_b_1_310, align 8, !dbg !51
  call void @llvm.dbg.value(metadata i64 %9, metadata !50, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.declare(metadata i64* %z_e_67_316, metadata !50, metadata !DIExpression()), !dbg !22
  store i64 %9, i64* %z_e_67_316, align 8, !dbg !51
  call void @llvm.dbg.declare(metadata i64* %z_b_3_312, metadata !50, metadata !DIExpression()), !dbg !22
  store i64 1, i64* %z_b_3_312, align 8, !dbg !51
  %10 = load i32, i32* %n_308, align 4, !dbg !51
  call void @llvm.dbg.value(metadata i32 %10, metadata !48, metadata !DIExpression()), !dbg !22
  %11 = sext i32 %10 to i64, !dbg !51
  call void @llvm.dbg.declare(metadata i64* %z_b_4_313, metadata !50, metadata !DIExpression()), !dbg !22
  store i64 %11, i64* %z_b_4_313, align 8, !dbg !51
  %12 = load i64, i64* %z_b_4_313, align 8, !dbg !51
  call void @llvm.dbg.value(metadata i64 %12, metadata !50, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.declare(metadata i64* %z_e_70_317, metadata !50, metadata !DIExpression()), !dbg !22
  store i64 %12, i64* %z_e_70_317, align 8, !dbg !51
  %13 = bitcast [22 x i64]* %"a$sd1_356" to i8*, !dbg !51
  %14 = bitcast i64* @.C284_drb062_matrixvector2_orig_no_foo to i8*, !dbg !51
  %15 = bitcast i64* @.C357_drb062_matrixvector2_orig_no_foo to i8*, !dbg !51
  %16 = bitcast i64* @.C358_drb062_matrixvector2_orig_no_foo to i8*, !dbg !51
  %17 = bitcast i64* %z_b_0_309 to i8*, !dbg !51
  %18 = bitcast i64* %z_b_1_310 to i8*, !dbg !51
  %19 = bitcast i64* %z_b_3_312 to i8*, !dbg !51
  %20 = bitcast i64* %z_b_4_313 to i8*, !dbg !51
  %21 = bitcast void (...)* @f90_template2_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !51
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, ...) %21(i8* %13, i8* %14, i8* %15, i8* %16, i8* %17, i8* %18, i8* %19, i8* %20), !dbg !51
  %22 = bitcast [22 x i64]* %"a$sd1_356" to i8*, !dbg !51
  %23 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !51
  call void (i8*, i32, ...) %23(i8* %22, i32 27), !dbg !51
  %24 = load i64, i64* %z_b_1_310, align 8, !dbg !51
  call void @llvm.dbg.value(metadata i64 %24, metadata !50, metadata !DIExpression()), !dbg !22
  %25 = load i64, i64* %z_b_0_309, align 8, !dbg !51
  call void @llvm.dbg.value(metadata i64 %25, metadata !50, metadata !DIExpression()), !dbg !22
  %26 = sub nsw i64 %25, 1, !dbg !51
  %27 = sub nsw i64 %24, %26, !dbg !51
  call void @llvm.dbg.declare(metadata i64* %z_b_2_311, metadata !50, metadata !DIExpression()), !dbg !22
  store i64 %27, i64* %z_b_2_311, align 8, !dbg !51
  %28 = load i64, i64* %z_b_1_310, align 8, !dbg !51
  call void @llvm.dbg.value(metadata i64 %28, metadata !50, metadata !DIExpression()), !dbg !22
  %29 = load i64, i64* %z_b_0_309, align 8, !dbg !51
  call void @llvm.dbg.value(metadata i64 %29, metadata !50, metadata !DIExpression()), !dbg !22
  %30 = sub nsw i64 %29, 1, !dbg !51
  %31 = sub nsw i64 %28, %30, !dbg !51
  %32 = load i64, i64* %z_b_4_313, align 8, !dbg !51
  call void @llvm.dbg.value(metadata i64 %32, metadata !50, metadata !DIExpression()), !dbg !22
  %33 = load i64, i64* %z_b_3_312, align 8, !dbg !51
  call void @llvm.dbg.value(metadata i64 %33, metadata !50, metadata !DIExpression()), !dbg !22
  %34 = sub nsw i64 %33, 1, !dbg !51
  %35 = sub nsw i64 %32, %34, !dbg !51
  %36 = mul nsw i64 %31, %35, !dbg !51
  call void @llvm.dbg.declare(metadata i64* %z_b_5_314, metadata !50, metadata !DIExpression()), !dbg !22
  store i64 %36, i64* %z_b_5_314, align 8, !dbg !51
  %37 = load i64, i64* %z_b_0_309, align 8, !dbg !51
  call void @llvm.dbg.value(metadata i64 %37, metadata !50, metadata !DIExpression()), !dbg !22
  %38 = load i64, i64* %z_b_1_310, align 8, !dbg !51
  call void @llvm.dbg.value(metadata i64 %38, metadata !50, metadata !DIExpression()), !dbg !22
  %39 = load i64, i64* %z_b_0_309, align 8, !dbg !51
  call void @llvm.dbg.value(metadata i64 %39, metadata !50, metadata !DIExpression()), !dbg !22
  %40 = sub nsw i64 %39, 1, !dbg !51
  %41 = sub nsw i64 %38, %40, !dbg !51
  %42 = load i64, i64* %z_b_3_312, align 8, !dbg !51
  call void @llvm.dbg.value(metadata i64 %42, metadata !50, metadata !DIExpression()), !dbg !22
  %43 = mul nsw i64 %41, %42, !dbg !51
  %44 = add nsw i64 %37, %43, !dbg !51
  call void @llvm.dbg.declare(metadata i64* %z_b_6_315, metadata !50, metadata !DIExpression()), !dbg !22
  store i64 %44, i64* %z_b_6_315, align 8, !dbg !51
  %45 = bitcast i64* %z_b_5_314 to i8*, !dbg !51
  %46 = bitcast i64* @.C357_drb062_matrixvector2_orig_no_foo to i8*, !dbg !51
  %47 = bitcast i64* @.C358_drb062_matrixvector2_orig_no_foo to i8*, !dbg !51
  %48 = bitcast float** %.Z0977_332 to i8*, !dbg !51
  %49 = bitcast i64* @.C286_drb062_matrixvector2_orig_no_foo to i8*, !dbg !51
  %50 = bitcast i64* @.C284_drb062_matrixvector2_orig_no_foo to i8*, !dbg !51
  %51 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !51
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %51(i8* %45, i8* %46, i8* %47, i8* null, i8* %48, i8* null, i8* %49, i8* %50, i8* null, i64 0), !dbg !51
  call void @llvm.dbg.declare(metadata i64* %z_b_7_319, metadata !50, metadata !DIExpression()), !dbg !22
  store i64 1, i64* %z_b_7_319, align 8, !dbg !52
  %52 = load i32, i32* %n_308, align 4, !dbg !52
  call void @llvm.dbg.value(metadata i32 %52, metadata !48, metadata !DIExpression()), !dbg !22
  %53 = sext i32 %52 to i64, !dbg !52
  call void @llvm.dbg.declare(metadata i64* %z_b_8_320, metadata !50, metadata !DIExpression()), !dbg !22
  store i64 %53, i64* %z_b_8_320, align 8, !dbg !52
  %54 = load i64, i64* %z_b_8_320, align 8, !dbg !52
  call void @llvm.dbg.value(metadata i64 %54, metadata !50, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.declare(metadata i64* %z_e_77_323, metadata !50, metadata !DIExpression()), !dbg !22
  store i64 %54, i64* %z_e_77_323, align 8, !dbg !52
  %55 = bitcast [16 x i64]* %"v$sd2_361" to i8*, !dbg !52
  %56 = bitcast i64* @.C284_drb062_matrixvector2_orig_no_foo to i8*, !dbg !52
  %57 = bitcast i64* @.C357_drb062_matrixvector2_orig_no_foo to i8*, !dbg !52
  %58 = bitcast i64* @.C358_drb062_matrixvector2_orig_no_foo to i8*, !dbg !52
  %59 = bitcast i64* %z_b_7_319 to i8*, !dbg !52
  %60 = bitcast i64* %z_b_8_320 to i8*, !dbg !52
  %61 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !52
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %61(i8* %55, i8* %56, i8* %57, i8* %58, i8* %59, i8* %60), !dbg !52
  %62 = bitcast [16 x i64]* %"v$sd2_361" to i8*, !dbg !52
  %63 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !52
  call void (i8*, i32, ...) %63(i8* %62, i32 27), !dbg !52
  %64 = load i64, i64* %z_b_8_320, align 8, !dbg !52
  call void @llvm.dbg.value(metadata i64 %64, metadata !50, metadata !DIExpression()), !dbg !22
  %65 = load i64, i64* %z_b_7_319, align 8, !dbg !52
  call void @llvm.dbg.value(metadata i64 %65, metadata !50, metadata !DIExpression()), !dbg !22
  %66 = sub nsw i64 %65, 1, !dbg !52
  %67 = sub nsw i64 %64, %66, !dbg !52
  call void @llvm.dbg.declare(metadata i64* %z_b_9_321, metadata !50, metadata !DIExpression()), !dbg !22
  store i64 %67, i64* %z_b_9_321, align 8, !dbg !52
  %68 = load i64, i64* %z_b_7_319, align 8, !dbg !52
  call void @llvm.dbg.value(metadata i64 %68, metadata !50, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.declare(metadata i64* %z_b_10_322, metadata !50, metadata !DIExpression()), !dbg !22
  store i64 %68, i64* %z_b_10_322, align 8, !dbg !52
  %69 = bitcast i64* %z_b_9_321 to i8*, !dbg !52
  %70 = bitcast i64* @.C357_drb062_matrixvector2_orig_no_foo to i8*, !dbg !52
  %71 = bitcast i64* @.C358_drb062_matrixvector2_orig_no_foo to i8*, !dbg !52
  %72 = bitcast float** %.Z0983_333 to i8*, !dbg !52
  %73 = bitcast i64* @.C286_drb062_matrixvector2_orig_no_foo to i8*, !dbg !52
  %74 = bitcast i64* @.C284_drb062_matrixvector2_orig_no_foo to i8*, !dbg !52
  %75 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !52
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %75(i8* %69, i8* %70, i8* %71, i8* null, i8* %72, i8* null, i8* %73, i8* %74, i8* null, i64 0), !dbg !52
  call void @llvm.dbg.declare(metadata i64* %z_b_11_326, metadata !50, metadata !DIExpression()), !dbg !22
  store i64 1, i64* %z_b_11_326, align 8, !dbg !53
  %76 = load i32, i32* %n_308, align 4, !dbg !53
  call void @llvm.dbg.value(metadata i32 %76, metadata !48, metadata !DIExpression()), !dbg !22
  %77 = sext i32 %76 to i64, !dbg !53
  call void @llvm.dbg.declare(metadata i64* %z_b_12_327, metadata !50, metadata !DIExpression()), !dbg !22
  store i64 %77, i64* %z_b_12_327, align 8, !dbg !53
  %78 = load i64, i64* %z_b_12_327, align 8, !dbg !53
  call void @llvm.dbg.value(metadata i64 %78, metadata !50, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.declare(metadata i64* %z_e_84_330, metadata !50, metadata !DIExpression()), !dbg !22
  store i64 %78, i64* %z_e_84_330, align 8, !dbg !53
  %79 = bitcast [16 x i64]* %"v_out$sd3_362" to i8*, !dbg !53
  %80 = bitcast i64* @.C284_drb062_matrixvector2_orig_no_foo to i8*, !dbg !53
  %81 = bitcast i64* @.C357_drb062_matrixvector2_orig_no_foo to i8*, !dbg !53
  %82 = bitcast i64* @.C358_drb062_matrixvector2_orig_no_foo to i8*, !dbg !53
  %83 = bitcast i64* %z_b_11_326 to i8*, !dbg !53
  %84 = bitcast i64* %z_b_12_327 to i8*, !dbg !53
  %85 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !53
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %85(i8* %79, i8* %80, i8* %81, i8* %82, i8* %83, i8* %84), !dbg !53
  %86 = bitcast [16 x i64]* %"v_out$sd3_362" to i8*, !dbg !53
  %87 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !53
  call void (i8*, i32, ...) %87(i8* %86, i32 27), !dbg !53
  %88 = load i64, i64* %z_b_12_327, align 8, !dbg !53
  call void @llvm.dbg.value(metadata i64 %88, metadata !50, metadata !DIExpression()), !dbg !22
  %89 = load i64, i64* %z_b_11_326, align 8, !dbg !53
  call void @llvm.dbg.value(metadata i64 %89, metadata !50, metadata !DIExpression()), !dbg !22
  %90 = sub nsw i64 %89, 1, !dbg !53
  %91 = sub nsw i64 %88, %90, !dbg !53
  call void @llvm.dbg.declare(metadata i64* %z_b_13_328, metadata !50, metadata !DIExpression()), !dbg !22
  store i64 %91, i64* %z_b_13_328, align 8, !dbg !53
  %92 = load i64, i64* %z_b_11_326, align 8, !dbg !53
  call void @llvm.dbg.value(metadata i64 %92, metadata !50, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.declare(metadata i64* %z_b_14_329, metadata !50, metadata !DIExpression()), !dbg !22
  store i64 %92, i64* %z_b_14_329, align 8, !dbg !53
  %93 = bitcast i64* %z_b_13_328 to i8*, !dbg !53
  %94 = bitcast i64* @.C357_drb062_matrixvector2_orig_no_foo to i8*, !dbg !53
  %95 = bitcast i64* @.C358_drb062_matrixvector2_orig_no_foo to i8*, !dbg !53
  %96 = bitcast float** %.Z0984_334 to i8*, !dbg !53
  %97 = bitcast i64* @.C286_drb062_matrixvector2_orig_no_foo to i8*, !dbg !53
  %98 = bitcast i64* @.C284_drb062_matrixvector2_orig_no_foo to i8*, !dbg !53
  %99 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !53
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %99(i8* %93, i8* %94, i8* %95, i8* null, i8* %96, i8* null, i8* %97, i8* %98, i8* null, i64 0), !dbg !53
  %100 = load i32, i32* %n_308, align 4, !dbg !54
  call void @llvm.dbg.value(metadata i32 %100, metadata !48, metadata !DIExpression()), !dbg !22
  store i32 %100, i32* %.dY0001_369, align 4, !dbg !54
  call void @llvm.dbg.declare(metadata i32* %i_306, metadata !55, metadata !DIExpression()), !dbg !22
  store i32 1, i32* %i_306, align 4, !dbg !54
  %101 = load i32, i32* %.dY0001_369, align 4, !dbg !54
  %102 = icmp sle i32 %101, 0, !dbg !54
  br i1 %102, label %L.LB2_368, label %L.LB2_367, !dbg !54

L.LB2_367:                                        ; preds = %L.LB2_482, %L.LB2_400
  call void @llvm.dbg.declare(metadata float* %sum_335, metadata !56, metadata !DIExpression()), !dbg !22
  store float 0.000000e+00, float* %sum_335, align 4, !dbg !57
  %103 = bitcast i64* %.S0000 to i8*, !dbg !58
  %104 = bitcast %astruct.dt88* %.uplevelArgPack0001_432 to i8**, !dbg !58
  store i8* %103, i8** %104, align 8, !dbg !58
  %105 = bitcast float* %sum_335 to i8*, !dbg !58
  %106 = bitcast %astruct.dt88* %.uplevelArgPack0001_432 to i8*, !dbg !58
  %107 = getelementptr i8, i8* %106, i64 8, !dbg !58
  %108 = bitcast i8* %107 to i8**, !dbg !58
  store i8* %105, i8** %108, align 8, !dbg !58
  %109 = bitcast i32* %n_308 to i8*, !dbg !58
  %110 = bitcast %astruct.dt88* %.uplevelArgPack0001_432 to i8*, !dbg !58
  %111 = getelementptr i8, i8* %110, i64 16, !dbg !58
  %112 = bitcast i8* %111 to i8**, !dbg !58
  store i8* %109, i8** %112, align 8, !dbg !58
  %113 = bitcast float** %.Z0977_332 to i8*, !dbg !58
  %114 = bitcast %astruct.dt88* %.uplevelArgPack0001_432 to i8*, !dbg !58
  %115 = getelementptr i8, i8* %114, i64 24, !dbg !58
  %116 = bitcast i8* %115 to i8**, !dbg !58
  store i8* %113, i8** %116, align 8, !dbg !58
  %117 = bitcast float** %.Z0977_332 to i8*, !dbg !58
  %118 = bitcast %astruct.dt88* %.uplevelArgPack0001_432 to i8*, !dbg !58
  %119 = getelementptr i8, i8* %118, i64 32, !dbg !58
  %120 = bitcast i8* %119 to i8**, !dbg !58
  store i8* %117, i8** %120, align 8, !dbg !58
  %121 = bitcast i64* %z_b_0_309 to i8*, !dbg !58
  %122 = bitcast %astruct.dt88* %.uplevelArgPack0001_432 to i8*, !dbg !58
  %123 = getelementptr i8, i8* %122, i64 40, !dbg !58
  %124 = bitcast i8* %123 to i8**, !dbg !58
  store i8* %121, i8** %124, align 8, !dbg !58
  %125 = bitcast i64* %z_b_1_310 to i8*, !dbg !58
  %126 = bitcast %astruct.dt88* %.uplevelArgPack0001_432 to i8*, !dbg !58
  %127 = getelementptr i8, i8* %126, i64 48, !dbg !58
  %128 = bitcast i8* %127 to i8**, !dbg !58
  store i8* %125, i8** %128, align 8, !dbg !58
  %129 = bitcast i64* %z_e_67_316 to i8*, !dbg !58
  %130 = bitcast %astruct.dt88* %.uplevelArgPack0001_432 to i8*, !dbg !58
  %131 = getelementptr i8, i8* %130, i64 56, !dbg !58
  %132 = bitcast i8* %131 to i8**, !dbg !58
  store i8* %129, i8** %132, align 8, !dbg !58
  %133 = bitcast i64* %z_b_3_312 to i8*, !dbg !58
  %134 = bitcast %astruct.dt88* %.uplevelArgPack0001_432 to i8*, !dbg !58
  %135 = getelementptr i8, i8* %134, i64 64, !dbg !58
  %136 = bitcast i8* %135 to i8**, !dbg !58
  store i8* %133, i8** %136, align 8, !dbg !58
  %137 = bitcast i64* %z_b_4_313 to i8*, !dbg !58
  %138 = bitcast %astruct.dt88* %.uplevelArgPack0001_432 to i8*, !dbg !58
  %139 = getelementptr i8, i8* %138, i64 72, !dbg !58
  %140 = bitcast i8* %139 to i8**, !dbg !58
  store i8* %137, i8** %140, align 8, !dbg !58
  %141 = bitcast i64* %z_b_2_311 to i8*, !dbg !58
  %142 = bitcast %astruct.dt88* %.uplevelArgPack0001_432 to i8*, !dbg !58
  %143 = getelementptr i8, i8* %142, i64 80, !dbg !58
  %144 = bitcast i8* %143 to i8**, !dbg !58
  store i8* %141, i8** %144, align 8, !dbg !58
  %145 = bitcast i64* %z_e_70_317 to i8*, !dbg !58
  %146 = bitcast %astruct.dt88* %.uplevelArgPack0001_432 to i8*, !dbg !58
  %147 = getelementptr i8, i8* %146, i64 88, !dbg !58
  %148 = bitcast i8* %147 to i8**, !dbg !58
  store i8* %145, i8** %148, align 8, !dbg !58
  %149 = bitcast i64* %z_b_5_314 to i8*, !dbg !58
  %150 = bitcast %astruct.dt88* %.uplevelArgPack0001_432 to i8*, !dbg !58
  %151 = getelementptr i8, i8* %150, i64 96, !dbg !58
  %152 = bitcast i8* %151 to i8**, !dbg !58
  store i8* %149, i8** %152, align 8, !dbg !58
  %153 = bitcast i64* %z_b_6_315 to i8*, !dbg !58
  %154 = bitcast %astruct.dt88* %.uplevelArgPack0001_432 to i8*, !dbg !58
  %155 = getelementptr i8, i8* %154, i64 104, !dbg !58
  %156 = bitcast i8* %155 to i8**, !dbg !58
  store i8* %153, i8** %156, align 8, !dbg !58
  %157 = bitcast i32* %i_306 to i8*, !dbg !58
  %158 = bitcast %astruct.dt88* %.uplevelArgPack0001_432 to i8*, !dbg !58
  %159 = getelementptr i8, i8* %158, i64 112, !dbg !58
  %160 = bitcast i8* %159 to i8**, !dbg !58
  store i8* %157, i8** %160, align 8, !dbg !58
  %161 = bitcast float** %.Z0983_333 to i8*, !dbg !58
  %162 = bitcast %astruct.dt88* %.uplevelArgPack0001_432 to i8*, !dbg !58
  %163 = getelementptr i8, i8* %162, i64 120, !dbg !58
  %164 = bitcast i8* %163 to i8**, !dbg !58
  store i8* %161, i8** %164, align 8, !dbg !58
  %165 = bitcast float** %.Z0983_333 to i8*, !dbg !58
  %166 = bitcast %astruct.dt88* %.uplevelArgPack0001_432 to i8*, !dbg !58
  %167 = getelementptr i8, i8* %166, i64 128, !dbg !58
  %168 = bitcast i8* %167 to i8**, !dbg !58
  store i8* %165, i8** %168, align 8, !dbg !58
  %169 = bitcast i64* %z_b_7_319 to i8*, !dbg !58
  %170 = bitcast %astruct.dt88* %.uplevelArgPack0001_432 to i8*, !dbg !58
  %171 = getelementptr i8, i8* %170, i64 136, !dbg !58
  %172 = bitcast i8* %171 to i8**, !dbg !58
  store i8* %169, i8** %172, align 8, !dbg !58
  %173 = bitcast i64* %z_b_8_320 to i8*, !dbg !58
  %174 = bitcast %astruct.dt88* %.uplevelArgPack0001_432 to i8*, !dbg !58
  %175 = getelementptr i8, i8* %174, i64 144, !dbg !58
  %176 = bitcast i8* %175 to i8**, !dbg !58
  store i8* %173, i8** %176, align 8, !dbg !58
  %177 = bitcast i64* %z_e_77_323 to i8*, !dbg !58
  %178 = bitcast %astruct.dt88* %.uplevelArgPack0001_432 to i8*, !dbg !58
  %179 = getelementptr i8, i8* %178, i64 152, !dbg !58
  %180 = bitcast i8* %179 to i8**, !dbg !58
  store i8* %177, i8** %180, align 8, !dbg !58
  %181 = bitcast i64* %z_b_9_321 to i8*, !dbg !58
  %182 = bitcast %astruct.dt88* %.uplevelArgPack0001_432 to i8*, !dbg !58
  %183 = getelementptr i8, i8* %182, i64 160, !dbg !58
  %184 = bitcast i8* %183 to i8**, !dbg !58
  store i8* %181, i8** %184, align 8, !dbg !58
  %185 = bitcast i64* %z_b_10_322 to i8*, !dbg !58
  %186 = bitcast %astruct.dt88* %.uplevelArgPack0001_432 to i8*, !dbg !58
  %187 = getelementptr i8, i8* %186, i64 168, !dbg !58
  %188 = bitcast i8* %187 to i8**, !dbg !58
  store i8* %185, i8** %188, align 8, !dbg !58
  %189 = bitcast [22 x i64]* %"a$sd1_356" to i8*, !dbg !58
  %190 = bitcast %astruct.dt88* %.uplevelArgPack0001_432 to i8*, !dbg !58
  %191 = getelementptr i8, i8* %190, i64 176, !dbg !58
  %192 = bitcast i8* %191 to i8**, !dbg !58
  store i8* %189, i8** %192, align 8, !dbg !58
  %193 = bitcast [16 x i64]* %"v$sd2_361" to i8*, !dbg !58
  %194 = bitcast %astruct.dt88* %.uplevelArgPack0001_432 to i8*, !dbg !58
  %195 = getelementptr i8, i8* %194, i64 184, !dbg !58
  %196 = bitcast i8* %195 to i8**, !dbg !58
  store i8* %193, i8** %196, align 8, !dbg !58
  br label %L.LB2_482, !dbg !58

L.LB2_482:                                        ; preds = %L.LB2_367
  %197 = bitcast void (i32*, i64*, i64*)* @__nv_drb062_matrixvector2_orig_no_foo_F1L29_1_ to i64*, !dbg !58
  %198 = bitcast %astruct.dt88* %.uplevelArgPack0001_432 to i64*, !dbg !58
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %197, i64* %198), !dbg !58
  %199 = load float, float* %sum_335, align 4, !dbg !59
  call void @llvm.dbg.value(metadata float %199, metadata !56, metadata !DIExpression()), !dbg !22
  %200 = load i32, i32* %i_306, align 4, !dbg !59
  call void @llvm.dbg.value(metadata i32 %200, metadata !55, metadata !DIExpression()), !dbg !22
  %201 = sext i32 %200 to i64, !dbg !59
  %202 = bitcast [16 x i64]* %"v_out$sd3_362" to i8*, !dbg !59
  %203 = getelementptr i8, i8* %202, i64 56, !dbg !59
  %204 = bitcast i8* %203 to i64*, !dbg !59
  %205 = load i64, i64* %204, align 8, !dbg !59
  %206 = add nsw i64 %201, %205, !dbg !59
  %207 = load float*, float** %.Z0984_334, align 8, !dbg !59
  call void @llvm.dbg.value(metadata float* %207, metadata !29, metadata !DIExpression()), !dbg !22
  %208 = bitcast float* %207 to i8*, !dbg !59
  %209 = getelementptr i8, i8* %208, i64 -4, !dbg !59
  %210 = bitcast i8* %209 to float*, !dbg !59
  %211 = getelementptr float, float* %210, i64 %206, !dbg !59
  store float %199, float* %211, align 4, !dbg !59
  %212 = load i32, i32* %i_306, align 4, !dbg !60
  call void @llvm.dbg.value(metadata i32 %212, metadata !55, metadata !DIExpression()), !dbg !22
  %213 = add nsw i32 %212, 1, !dbg !60
  store i32 %213, i32* %i_306, align 4, !dbg !60
  %214 = load i32, i32* %.dY0001_369, align 4, !dbg !60
  %215 = sub nsw i32 %214, 1, !dbg !60
  store i32 %215, i32* %.dY0001_369, align 4, !dbg !60
  %216 = load i32, i32* %.dY0001_369, align 4, !dbg !60
  %217 = icmp sgt i32 %216, 0, !dbg !60
  br i1 %217, label %L.LB2_367, label %L.LB2_368, !dbg !60

L.LB2_368:                                        ; preds = %L.LB2_482, %L.LB2_400
  %218 = load float*, float** %.Z0977_332, align 8, !dbg !28
  call void @llvm.dbg.value(metadata float* %218, metadata !41, metadata !DIExpression()), !dbg !22
  %219 = bitcast float* %218 to i8*, !dbg !28
  %220 = bitcast i32 (...)* @f90_allocated_i8 to i32 (i8*, ...)*, !dbg !28
  %221 = call i32 (i8*, ...) %220(i8* %219), !dbg !28
  %222 = and i32 %221, 1, !dbg !28
  %223 = icmp eq i32 %222, 0, !dbg !28
  br i1 %223, label %L.LB2_382, label %L.LB2_508, !dbg !28

L.LB2_508:                                        ; preds = %L.LB2_368
  %224 = load float*, float** %.Z0977_332, align 8, !dbg !28
  call void @llvm.dbg.value(metadata float* %224, metadata !41, metadata !DIExpression()), !dbg !22
  %225 = bitcast float* %224 to i8*, !dbg !28
  %226 = bitcast i64* @.C284_drb062_matrixvector2_orig_no_foo to i8*, !dbg !28
  %227 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !28
  call void (i8*, i8*, i8*, i8*, i64, ...) %227(i8* null, i8* %225, i8* %226, i8* null, i64 0), !dbg !28
  %228 = bitcast float** %.Z0977_332 to i8**, !dbg !28
  store i8* null, i8** %228, align 8, !dbg !28
  %229 = bitcast [22 x i64]* %"a$sd1_356" to i64*, !dbg !28
  store i64 0, i64* %229, align 8, !dbg !28
  br label %L.LB2_382

L.LB2_382:                                        ; preds = %L.LB2_508, %L.LB2_368
  %230 = load float*, float** %.Z0983_333, align 8, !dbg !28
  call void @llvm.dbg.value(metadata float* %230, metadata !40, metadata !DIExpression()), !dbg !22
  %231 = bitcast float* %230 to i8*, !dbg !28
  %232 = bitcast i32 (...)* @f90_allocated_i8 to i32 (i8*, ...)*, !dbg !28
  %233 = call i32 (i8*, ...) %232(i8* %231), !dbg !28
  %234 = and i32 %233, 1, !dbg !28
  %235 = icmp eq i32 %234, 0, !dbg !28
  br i1 %235, label %L.LB2_385, label %L.LB2_509, !dbg !28

L.LB2_509:                                        ; preds = %L.LB2_382
  %236 = load float*, float** %.Z0983_333, align 8, !dbg !28
  call void @llvm.dbg.value(metadata float* %236, metadata !40, metadata !DIExpression()), !dbg !22
  %237 = bitcast float* %236 to i8*, !dbg !28
  %238 = bitcast i64* @.C284_drb062_matrixvector2_orig_no_foo to i8*, !dbg !28
  %239 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !28
  call void (i8*, i8*, i8*, i8*, i64, ...) %239(i8* null, i8* %237, i8* %238, i8* null, i64 0), !dbg !28
  %240 = bitcast float** %.Z0983_333 to i8**, !dbg !28
  store i8* null, i8** %240, align 8, !dbg !28
  %241 = bitcast [16 x i64]* %"v$sd2_361" to i64*, !dbg !28
  store i64 0, i64* %241, align 8, !dbg !28
  br label %L.LB2_385

L.LB2_385:                                        ; preds = %L.LB2_509, %L.LB2_382
  %242 = load float*, float** %.Z0984_334, align 8, !dbg !28
  call void @llvm.dbg.value(metadata float* %242, metadata !29, metadata !DIExpression()), !dbg !22
  %243 = bitcast float* %242 to i8*, !dbg !28
  %244 = bitcast i32 (...)* @f90_allocated_i8 to i32 (i8*, ...)*, !dbg !28
  %245 = call i32 (i8*, ...) %244(i8* %243), !dbg !28
  %246 = and i32 %245, 1, !dbg !28
  %247 = icmp eq i32 %246, 0, !dbg !28
  br i1 %247, label %L.LB2_386, label %L.LB2_510, !dbg !28

L.LB2_510:                                        ; preds = %L.LB2_385
  %248 = load float*, float** %.Z0984_334, align 8, !dbg !28
  call void @llvm.dbg.value(metadata float* %248, metadata !29, metadata !DIExpression()), !dbg !22
  %249 = bitcast float* %248 to i8*, !dbg !28
  %250 = bitcast i64* @.C284_drb062_matrixvector2_orig_no_foo to i8*, !dbg !28
  %251 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !28
  call void (i8*, i8*, i8*, i8*, i64, ...) %251(i8* null, i8* %249, i8* %250, i8* null, i64 0), !dbg !28
  %252 = bitcast float** %.Z0984_334 to i8**, !dbg !28
  store i8* null, i8** %252, align 8, !dbg !28
  %253 = bitcast [16 x i64]* %"v_out$sd3_362" to i64*, !dbg !28
  store i64 0, i64* %253, align 8, !dbg !28
  br label %L.LB2_386

L.LB2_386:                                        ; preds = %L.LB2_510, %L.LB2_385
  ret void, !dbg !28
}

define internal void @__nv_drb062_matrixvector2_orig_no_foo_F1L29_1_(i32* %__nv_drb062_matrixvector2_orig_no_foo_F1L29_1Arg0, i64* %__nv_drb062_matrixvector2_orig_no_foo_F1L29_1Arg1, i64* %__nv_drb062_matrixvector2_orig_no_foo_F1L29_1Arg2) #0 !dbg !61 {
L.entry:
  %.S0000_387 = alloca i8*, align 8
  %__gtid___nv_drb062_matrixvector2_orig_no_foo_F1L29_1__534 = alloca i32, align 4
  %sum_339 = alloca float, align 4
  %.i0000p_341 = alloca i32, align 4
  %j_340 = alloca i32, align 4
  %.du0002p_373 = alloca i32, align 4
  %.de0002p_374 = alloca i32, align 4
  %.di0002p_375 = alloca i32, align 4
  %.ds0002p_376 = alloca i32, align 4
  %.dl0002p_378 = alloca i32, align 4
  %.dl0002p.copy_528 = alloca i32, align 4
  %.de0002p.copy_529 = alloca i32, align 4
  %.ds0002p.copy_530 = alloca i32, align 4
  %.dX0002p_377 = alloca i32, align 4
  %.dY0002p_372 = alloca i32, align 4
  %z__io_348 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_drb062_matrixvector2_orig_no_foo_F1L29_1Arg0, metadata !64, metadata !DIExpression()), !dbg !65
  call void @llvm.dbg.declare(metadata i64* %__nv_drb062_matrixvector2_orig_no_foo_F1L29_1Arg1, metadata !66, metadata !DIExpression()), !dbg !65
  call void @llvm.dbg.declare(metadata i64* %__nv_drb062_matrixvector2_orig_no_foo_F1L29_1Arg2, metadata !67, metadata !DIExpression()), !dbg !65
  call void @llvm.dbg.value(metadata i32 1, metadata !68, metadata !DIExpression()), !dbg !65
  call void @llvm.dbg.value(metadata i32 0, metadata !69, metadata !DIExpression()), !dbg !65
  call void @llvm.dbg.value(metadata i32 1, metadata !70, metadata !DIExpression()), !dbg !65
  call void @llvm.dbg.value(metadata i32 0, metadata !71, metadata !DIExpression()), !dbg !65
  call void @llvm.dbg.value(metadata i32 1, metadata !72, metadata !DIExpression()), !dbg !65
  %0 = bitcast i64* %__nv_drb062_matrixvector2_orig_no_foo_F1L29_1Arg2 to i8**, !dbg !73
  %1 = load i8*, i8** %0, align 8, !dbg !73
  %2 = bitcast i8** %.S0000_387 to i64*, !dbg !73
  store i8* %1, i8** %.S0000_387, align 8, !dbg !73
  %3 = load i32, i32* %__nv_drb062_matrixvector2_orig_no_foo_F1L29_1Arg0, align 4, !dbg !74
  store i32 %3, i32* %__gtid___nv_drb062_matrixvector2_orig_no_foo_F1L29_1__534, align 4, !dbg !74
  br label %L.LB3_518

L.LB3_518:                                        ; preds = %L.entry
  br label %L.LB3_338

L.LB3_338:                                        ; preds = %L.LB3_518
  call void @llvm.dbg.declare(metadata float* %sum_339, metadata !75, metadata !DIExpression()), !dbg !74
  store float 0.000000e+00, float* %sum_339, align 4, !dbg !73
  store i32 0, i32* %.i0000p_341, align 4, !dbg !76
  call void @llvm.dbg.declare(metadata i32* %j_340, metadata !77, metadata !DIExpression()), !dbg !74
  store i32 1, i32* %j_340, align 4, !dbg !76
  %4 = bitcast i64* %__nv_drb062_matrixvector2_orig_no_foo_F1L29_1Arg2 to i8*, !dbg !76
  %5 = getelementptr i8, i8* %4, i64 16, !dbg !76
  %6 = bitcast i8* %5 to i32**, !dbg !76
  %7 = load i32*, i32** %6, align 8, !dbg !76
  %8 = load i32, i32* %7, align 4, !dbg !76
  store i32 %8, i32* %.du0002p_373, align 4, !dbg !76
  %9 = bitcast i64* %__nv_drb062_matrixvector2_orig_no_foo_F1L29_1Arg2 to i8*, !dbg !76
  %10 = getelementptr i8, i8* %9, i64 16, !dbg !76
  %11 = bitcast i8* %10 to i32**, !dbg !76
  %12 = load i32*, i32** %11, align 8, !dbg !76
  %13 = load i32, i32* %12, align 4, !dbg !76
  store i32 %13, i32* %.de0002p_374, align 4, !dbg !76
  store i32 1, i32* %.di0002p_375, align 4, !dbg !76
  %14 = load i32, i32* %.di0002p_375, align 4, !dbg !76
  store i32 %14, i32* %.ds0002p_376, align 4, !dbg !76
  store i32 1, i32* %.dl0002p_378, align 4, !dbg !76
  %15 = load i32, i32* %.dl0002p_378, align 4, !dbg !76
  store i32 %15, i32* %.dl0002p.copy_528, align 4, !dbg !76
  %16 = load i32, i32* %.de0002p_374, align 4, !dbg !76
  store i32 %16, i32* %.de0002p.copy_529, align 4, !dbg !76
  %17 = load i32, i32* %.ds0002p_376, align 4, !dbg !76
  store i32 %17, i32* %.ds0002p.copy_530, align 4, !dbg !76
  %18 = load i32, i32* %__gtid___nv_drb062_matrixvector2_orig_no_foo_F1L29_1__534, align 4, !dbg !76
  %19 = bitcast i32* %.i0000p_341 to i64*, !dbg !76
  %20 = bitcast i32* %.dl0002p.copy_528 to i64*, !dbg !76
  %21 = bitcast i32* %.de0002p.copy_529 to i64*, !dbg !76
  %22 = bitcast i32* %.ds0002p.copy_530 to i64*, !dbg !76
  %23 = load i32, i32* %.ds0002p.copy_530, align 4, !dbg !76
  call void @__kmpc_for_static_init_4(i64* null, i32 %18, i32 34, i64* %19, i64* %20, i64* %21, i64* %22, i32 %23, i32 1), !dbg !76
  %24 = load i32, i32* %.dl0002p.copy_528, align 4, !dbg !76
  store i32 %24, i32* %.dl0002p_378, align 4, !dbg !76
  %25 = load i32, i32* %.de0002p.copy_529, align 4, !dbg !76
  store i32 %25, i32* %.de0002p_374, align 4, !dbg !76
  %26 = load i32, i32* %.ds0002p.copy_530, align 4, !dbg !76
  store i32 %26, i32* %.ds0002p_376, align 4, !dbg !76
  %27 = load i32, i32* %.dl0002p_378, align 4, !dbg !76
  store i32 %27, i32* %j_340, align 4, !dbg !76
  %28 = load i32, i32* %j_340, align 4, !dbg !76
  call void @llvm.dbg.value(metadata i32 %28, metadata !77, metadata !DIExpression()), !dbg !74
  store i32 %28, i32* %.dX0002p_377, align 4, !dbg !76
  %29 = load i32, i32* %.dX0002p_377, align 4, !dbg !76
  %30 = load i32, i32* %.du0002p_373, align 4, !dbg !76
  %31 = icmp sgt i32 %29, %30, !dbg !76
  br i1 %31, label %L.LB3_371, label %L.LB3_573, !dbg !76

L.LB3_573:                                        ; preds = %L.LB3_338
  %32 = load i32, i32* %.dX0002p_377, align 4, !dbg !76
  store i32 %32, i32* %j_340, align 4, !dbg !76
  %33 = load i32, i32* %.di0002p_375, align 4, !dbg !76
  %34 = load i32, i32* %.de0002p_374, align 4, !dbg !76
  %35 = load i32, i32* %.dX0002p_377, align 4, !dbg !76
  %36 = sub nsw i32 %34, %35, !dbg !76
  %37 = add nsw i32 %33, %36, !dbg !76
  %38 = load i32, i32* %.di0002p_375, align 4, !dbg !76
  %39 = sdiv i32 %37, %38, !dbg !76
  store i32 %39, i32* %.dY0002p_372, align 4, !dbg !76
  %40 = load i32, i32* %.dY0002p_372, align 4, !dbg !76
  %41 = icmp sle i32 %40, 0, !dbg !76
  br i1 %41, label %L.LB3_381, label %L.LB3_380, !dbg !76

L.LB3_380:                                        ; preds = %L.LB3_380, %L.LB3_573
  %42 = load i32, i32* %j_340, align 4, !dbg !78
  call void @llvm.dbg.value(metadata i32 %42, metadata !77, metadata !DIExpression()), !dbg !74
  %43 = sext i32 %42 to i64, !dbg !78
  %44 = bitcast i64* %__nv_drb062_matrixvector2_orig_no_foo_F1L29_1Arg2 to i8*, !dbg !78
  %45 = getelementptr i8, i8* %44, i64 184, !dbg !78
  %46 = bitcast i8* %45 to i8**, !dbg !78
  %47 = load i8*, i8** %46, align 8, !dbg !78
  %48 = getelementptr i8, i8* %47, i64 56, !dbg !78
  %49 = bitcast i8* %48 to i64*, !dbg !78
  %50 = load i64, i64* %49, align 8, !dbg !78
  %51 = add nsw i64 %43, %50, !dbg !78
  %52 = bitcast i64* %__nv_drb062_matrixvector2_orig_no_foo_F1L29_1Arg2 to i8*, !dbg !78
  %53 = getelementptr i8, i8* %52, i64 128, !dbg !78
  %54 = bitcast i8* %53 to i8***, !dbg !78
  %55 = load i8**, i8*** %54, align 8, !dbg !78
  %56 = load i8*, i8** %55, align 8, !dbg !78
  %57 = getelementptr i8, i8* %56, i64 -4, !dbg !78
  %58 = bitcast i8* %57 to float*, !dbg !78
  %59 = getelementptr float, float* %58, i64 %51, !dbg !78
  %60 = load float, float* %59, align 4, !dbg !78
  %61 = bitcast i64* %__nv_drb062_matrixvector2_orig_no_foo_F1L29_1Arg2 to i8*, !dbg !78
  %62 = getelementptr i8, i8* %61, i64 112, !dbg !78
  %63 = bitcast i8* %62 to i32**, !dbg !78
  %64 = load i32*, i32** %63, align 8, !dbg !78
  %65 = load i32, i32* %64, align 4, !dbg !78
  %66 = sext i32 %65 to i64, !dbg !78
  %67 = load i32, i32* %j_340, align 4, !dbg !78
  call void @llvm.dbg.value(metadata i32 %67, metadata !77, metadata !DIExpression()), !dbg !74
  %68 = sext i32 %67 to i64, !dbg !78
  %69 = bitcast i64* %__nv_drb062_matrixvector2_orig_no_foo_F1L29_1Arg2 to i8*, !dbg !78
  %70 = getelementptr i8, i8* %69, i64 176, !dbg !78
  %71 = bitcast i8* %70 to i8**, !dbg !78
  %72 = load i8*, i8** %71, align 8, !dbg !78
  %73 = getelementptr i8, i8* %72, i64 160, !dbg !78
  %74 = bitcast i8* %73 to i64*, !dbg !78
  %75 = load i64, i64* %74, align 8, !dbg !78
  %76 = mul nsw i64 %68, %75, !dbg !78
  %77 = add nsw i64 %66, %76, !dbg !78
  %78 = bitcast i64* %__nv_drb062_matrixvector2_orig_no_foo_F1L29_1Arg2 to i8*, !dbg !78
  %79 = getelementptr i8, i8* %78, i64 176, !dbg !78
  %80 = bitcast i8* %79 to i8**, !dbg !78
  %81 = load i8*, i8** %80, align 8, !dbg !78
  %82 = getelementptr i8, i8* %81, i64 56, !dbg !78
  %83 = bitcast i8* %82 to i64*, !dbg !78
  %84 = load i64, i64* %83, align 8, !dbg !78
  %85 = add nsw i64 %77, %84, !dbg !78
  %86 = bitcast i64* %__nv_drb062_matrixvector2_orig_no_foo_F1L29_1Arg2 to i8*, !dbg !78
  %87 = getelementptr i8, i8* %86, i64 32, !dbg !78
  %88 = bitcast i8* %87 to i8***, !dbg !78
  %89 = load i8**, i8*** %88, align 8, !dbg !78
  %90 = load i8*, i8** %89, align 8, !dbg !78
  %91 = getelementptr i8, i8* %90, i64 -4, !dbg !78
  %92 = bitcast i8* %91 to float*, !dbg !78
  %93 = getelementptr float, float* %92, i64 %85, !dbg !78
  %94 = load float, float* %93, align 4, !dbg !78
  %95 = fmul fast float %60, %94, !dbg !78
  %96 = load float, float* %sum_339, align 4, !dbg !78
  call void @llvm.dbg.value(metadata float %96, metadata !75, metadata !DIExpression()), !dbg !74
  %97 = fadd fast float %95, %96, !dbg !78
  store float %97, float* %sum_339, align 4, !dbg !78
  call void (...) @_mp_bcs_nest(), !dbg !79
  %98 = bitcast i32* @.C345___nv_drb062_matrixvector2_orig_no_foo_F1L29_1 to i8*, !dbg !79
  %99 = bitcast [57 x i8]* @.C343___nv_drb062_matrixvector2_orig_no_foo_F1L29_1 to i8*, !dbg !79
  %100 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !79
  call void (i8*, i8*, i64, ...) %100(i8* %98, i8* %99, i64 57), !dbg !79
  %101 = bitcast i32* @.C346___nv_drb062_matrixvector2_orig_no_foo_F1L29_1 to i8*, !dbg !79
  %102 = bitcast i32* @.C283___nv_drb062_matrixvector2_orig_no_foo_F1L29_1 to i8*, !dbg !79
  %103 = bitcast i32* @.C283___nv_drb062_matrixvector2_orig_no_foo_F1L29_1 to i8*, !dbg !79
  %104 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !79
  %105 = call i32 (i8*, i8*, i8*, i8*, ...) %104(i8* %101, i8* null, i8* %102, i8* %103), !dbg !79
  call void @llvm.dbg.declare(metadata i32* %z__io_348, metadata !80, metadata !DIExpression()), !dbg !65
  store i32 %105, i32* %z__io_348, align 4, !dbg !79
  %106 = load float, float* %sum_339, align 4, !dbg !79
  call void @llvm.dbg.value(metadata float %106, metadata !75, metadata !DIExpression()), !dbg !74
  %107 = bitcast i32 (...)* @f90io_sc_f_ldw to i32 (float, i32, ...)*, !dbg !79
  %108 = call i32 (float, i32, ...) %107(float %106, i32 27), !dbg !79
  store i32 %108, i32* %z__io_348, align 4, !dbg !79
  %109 = call i32 (...) @f90io_ldw_end(), !dbg !79
  store i32 %109, i32* %z__io_348, align 4, !dbg !79
  call void (...) @_mp_ecs_nest(), !dbg !79
  %110 = load i32, i32* %.di0002p_375, align 4, !dbg !74
  %111 = load i32, i32* %j_340, align 4, !dbg !74
  call void @llvm.dbg.value(metadata i32 %111, metadata !77, metadata !DIExpression()), !dbg !74
  %112 = add nsw i32 %110, %111, !dbg !74
  store i32 %112, i32* %j_340, align 4, !dbg !74
  %113 = load i32, i32* %.dY0002p_372, align 4, !dbg !74
  %114 = sub nsw i32 %113, 1, !dbg !74
  store i32 %114, i32* %.dY0002p_372, align 4, !dbg !74
  %115 = load i32, i32* %.dY0002p_372, align 4, !dbg !74
  %116 = icmp sgt i32 %115, 0, !dbg !74
  br i1 %116, label %L.LB3_380, label %L.LB3_381, !dbg !74

L.LB3_381:                                        ; preds = %L.LB3_380, %L.LB3_573
  br label %L.LB3_371

L.LB3_371:                                        ; preds = %L.LB3_381, %L.LB3_338
  %117 = load i32, i32* %__gtid___nv_drb062_matrixvector2_orig_no_foo_F1L29_1__534, align 4, !dbg !74
  call void @__kmpc_for_static_fini(i64* null, i32 %117), !dbg !74
  %118 = call i32 (...) @_mp_bcs_nest_red(), !dbg !74
  %119 = call i32 (...) @_mp_bcs_nest_red(), !dbg !74
  %120 = load float, float* %sum_339, align 4, !dbg !74
  call void @llvm.dbg.value(metadata float %120, metadata !75, metadata !DIExpression()), !dbg !74
  %121 = bitcast i64* %__nv_drb062_matrixvector2_orig_no_foo_F1L29_1Arg2 to i8*, !dbg !74
  %122 = getelementptr i8, i8* %121, i64 8, !dbg !74
  %123 = bitcast i8* %122 to float**, !dbg !74
  %124 = load float*, float** %123, align 8, !dbg !74
  %125 = load float, float* %124, align 4, !dbg !74
  %126 = fadd fast float %120, %125, !dbg !74
  %127 = bitcast i64* %__nv_drb062_matrixvector2_orig_no_foo_F1L29_1Arg2 to i8*, !dbg !74
  %128 = getelementptr i8, i8* %127, i64 8, !dbg !74
  %129 = bitcast i8* %128 to float**, !dbg !74
  %130 = load float*, float** %129, align 8, !dbg !74
  store float %126, float* %130, align 4, !dbg !74
  %131 = call i32 (...) @_mp_ecs_nest_red(), !dbg !74
  %132 = call i32 (...) @_mp_ecs_nest_red(), !dbg !74
  br label %L.LB3_353

L.LB3_353:                                        ; preds = %L.LB3_371
  ret void, !dbg !74
}

declare signext i32 @_mp_ecs_nest_red(...) #0

declare signext i32 @_mp_bcs_nest_red(...) #0

declare void @_mp_ecs_nest(...) #0

declare signext i32 @f90io_ldw_end(...) #0

declare signext i32 @f90io_sc_f_ldw(...) #0

declare signext i32 @f90io_print_init(...) #0

declare void @f90io_src_info03a(...) #0

declare void @_mp_bcs_nest(...) #0

declare void @__kmpc_for_static_fini(i64*, i32) #0

declare void @__kmpc_for_static_init_4(i64*, i32, i32, i64*, i64*, i64*, i64*, i32, i32) #0

declare void @f90_dealloc03a_i8(...) #0

declare signext i32 @f90_allocated_i8(...) #0

declare void @f90_template1_i8(...) #0

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
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB062-matrixvector2-orig-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb062_matrixvector2_orig_no", scope: !2, file: !3, line: 10, type: !6, scopeLine: 10, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
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
!28 = !DILocation(line: 38, column: 1, scope: !18)
!29 = !DILocalVariable(name: "v_out", scope: !18, file: !3, type: !30)
!30 = !DICompositeType(tag: DW_TAG_array_type, baseType: !31, size: 32, align: 32, elements: !32)
!31 = !DIBasicType(name: "real", size: 32, align: 32, encoding: DW_ATE_float)
!32 = !{!33}
!33 = !DISubrange(count: 0, lowerBound: 1)
!34 = !DILocation(line: 16, column: 1, scope: !18)
!35 = !DILocalVariable(scope: !18, file: !3, type: !36, flags: DIFlagArtificial)
!36 = !DICompositeType(tag: DW_TAG_array_type, baseType: !37, size: 1024, align: 64, elements: !38)
!37 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!38 = !{!39}
!39 = !DISubrange(count: 16, lowerBound: 1)
!40 = !DILocalVariable(name: "v", scope: !18, file: !3, type: !30)
!41 = !DILocalVariable(name: "a", scope: !18, file: !3, type: !42)
!42 = !DICompositeType(tag: DW_TAG_array_type, baseType: !31, size: 32, align: 32, elements: !43)
!43 = !{!33, !33}
!44 = !DILocalVariable(scope: !18, file: !3, type: !45, flags: DIFlagArtificial)
!45 = !DICompositeType(tag: DW_TAG_array_type, baseType: !37, size: 1408, align: 64, elements: !46)
!46 = !{!47}
!47 = !DISubrange(count: 22, lowerBound: 1)
!48 = !DILocalVariable(name: "n", scope: !18, file: !3, type: !9)
!49 = !DILocation(line: 22, column: 1, scope: !18)
!50 = !DILocalVariable(scope: !18, file: !3, type: !37, flags: DIFlagArtificial)
!51 = !DILocation(line: 23, column: 1, scope: !18)
!52 = !DILocation(line: 24, column: 1, scope: !18)
!53 = !DILocation(line: 25, column: 1, scope: !18)
!54 = !DILocation(line: 27, column: 1, scope: !18)
!55 = !DILocalVariable(name: "i", scope: !18, file: !3, type: !9)
!56 = !DILocalVariable(name: "sum", scope: !18, file: !3, type: !31)
!57 = !DILocation(line: 28, column: 1, scope: !18)
!58 = !DILocation(line: 29, column: 1, scope: !18)
!59 = !DILocation(line: 35, column: 1, scope: !18)
!60 = !DILocation(line: 36, column: 1, scope: !18)
!61 = distinct !DISubprogram(name: "__nv_drb062_matrixvector2_orig_no_foo_F1L29_1", scope: !2, file: !3, line: 29, type: !62, scopeLine: 29, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!62 = !DISubroutineType(types: !63)
!63 = !{null, !9, !37, !37}
!64 = !DILocalVariable(name: "__nv_drb062_matrixvector2_orig_no_foo_F1L29_1Arg0", arg: 1, scope: !61, file: !3, type: !9)
!65 = !DILocation(line: 0, scope: !61)
!66 = !DILocalVariable(name: "__nv_drb062_matrixvector2_orig_no_foo_F1L29_1Arg1", arg: 2, scope: !61, file: !3, type: !37)
!67 = !DILocalVariable(name: "__nv_drb062_matrixvector2_orig_no_foo_F1L29_1Arg2", arg: 3, scope: !61, file: !3, type: !37)
!68 = !DILocalVariable(name: "omp_sched_static", scope: !61, file: !3, type: !9)
!69 = !DILocalVariable(name: "omp_proc_bind_false", scope: !61, file: !3, type: !9)
!70 = !DILocalVariable(name: "omp_proc_bind_true", scope: !61, file: !3, type: !9)
!71 = !DILocalVariable(name: "omp_lock_hint_none", scope: !61, file: !3, type: !9)
!72 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !61, file: !3, type: !9)
!73 = !DILocation(line: 29, column: 1, scope: !61)
!74 = !DILocation(line: 33, column: 1, scope: !61)
!75 = !DILocalVariable(name: "sum", scope: !61, file: !3, type: !31)
!76 = !DILocation(line: 30, column: 1, scope: !61)
!77 = !DILocalVariable(name: "j", scope: !61, file: !3, type: !9)
!78 = !DILocation(line: 31, column: 1, scope: !61)
!79 = !DILocation(line: 32, column: 1, scope: !61)
!80 = !DILocalVariable(scope: !61, file: !3, type: !9, flags: DIFlagArtificial)

; ModuleID = '/tmp/DRB061-matrixvector1-orig-no-bcd240.ll'
source_filename = "/tmp/DRB061-matrixvector1-orig-no-bcd240.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.struct_ul_MAIN__297 = type <{ i8* }>
%astruct.dt86 = type <{ i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8* }>

@.C283_MAIN_ = internal constant i32 0
@.C285_drb061_matrixvector1_orig_no_foo = internal constant i32 1
@.C303_drb061_matrixvector1_orig_no_foo = internal constant i32 27
@.C348_drb061_matrixvector1_orig_no_foo = internal constant i64 4
@.C347_drb061_matrixvector1_orig_no_foo = internal constant i64 27
@.C331_drb061_matrixvector1_orig_no_foo = internal constant i32 100
@.C283_drb061_matrixvector1_orig_no_foo = internal constant i32 0
@.C286_drb061_matrixvector1_orig_no_foo = internal constant i64 1
@.C284_drb061_matrixvector1_orig_no_foo = internal constant i64 0
@.C285___nv_drb061_matrixvector1_orig_no_foo_F1L26_1 = internal constant i32 1
@.C283___nv_drb061_matrixvector1_orig_no_foo_F1L26_1 = internal constant i32 0

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
  call void @drb061_matrixvector1_orig_no_foo(i64* %2), !dbg !16
  ret void, !dbg !17
}

define internal void @drb061_matrixvector1_orig_no_foo(i64* %.S0000) #0 !dbg !18 {
L.entry:
  %__gtid_drb061_matrixvector1_orig_no_foo_483 = alloca i32, align 4
  %.Z0984_334 = alloca float*, align 8
  %"v_out$sd3_352" = alloca [16 x i64], align 8
  %.Z0983_333 = alloca float*, align 8
  %"v$sd2_351" = alloca [16 x i64], align 8
  %.Z0977_332 = alloca float*, align 8
  %"a$sd1_346" = alloca [22 x i64], align 8
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
  %.uplevelArgPack0001_419 = alloca %astruct.dt86, align 16
  call void @llvm.dbg.declare(metadata i64* %.S0000, metadata !20, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i32 1, metadata !23, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i32 0, metadata !24, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i32 1, metadata !25, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i32 0, metadata !26, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i32 1, metadata !27, metadata !DIExpression()), !dbg !22
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !28
  store i32 %0, i32* %__gtid_drb061_matrixvector1_orig_no_foo_483, align 4, !dbg !28
  call void @llvm.dbg.declare(metadata float** %.Z0984_334, metadata !29, metadata !DIExpression(DW_OP_deref)), !dbg !22
  %1 = bitcast float** %.Z0984_334 to i8**, !dbg !34
  store i8* null, i8** %1, align 8, !dbg !34
  call void @llvm.dbg.declare(metadata [16 x i64]* %"v_out$sd3_352", metadata !35, metadata !DIExpression()), !dbg !22
  %2 = bitcast [16 x i64]* %"v_out$sd3_352" to i64*, !dbg !34
  store i64 0, i64* %2, align 8, !dbg !34
  call void @llvm.dbg.declare(metadata float** %.Z0983_333, metadata !40, metadata !DIExpression(DW_OP_deref)), !dbg !22
  %3 = bitcast float** %.Z0983_333 to i8**, !dbg !34
  store i8* null, i8** %3, align 8, !dbg !34
  call void @llvm.dbg.declare(metadata [16 x i64]* %"v$sd2_351", metadata !35, metadata !DIExpression()), !dbg !22
  %4 = bitcast [16 x i64]* %"v$sd2_351" to i64*, !dbg !34
  store i64 0, i64* %4, align 8, !dbg !34
  call void @llvm.dbg.declare(metadata float** %.Z0977_332, metadata !41, metadata !DIExpression(DW_OP_deref)), !dbg !22
  %5 = bitcast float** %.Z0977_332 to i8**, !dbg !34
  store i8* null, i8** %5, align 8, !dbg !34
  call void @llvm.dbg.declare(metadata [22 x i64]* %"a$sd1_346", metadata !44, metadata !DIExpression()), !dbg !22
  %6 = bitcast [22 x i64]* %"a$sd1_346" to i64*, !dbg !34
  store i64 0, i64* %6, align 8, !dbg !34
  br label %L.LB2_390

L.LB2_390:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %n_308, metadata !48, metadata !DIExpression()), !dbg !22
  store i32 100, i32* %n_308, align 4, !dbg !49
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
  %13 = bitcast [22 x i64]* %"a$sd1_346" to i8*, !dbg !51
  %14 = bitcast i64* @.C284_drb061_matrixvector1_orig_no_foo to i8*, !dbg !51
  %15 = bitcast i64* @.C347_drb061_matrixvector1_orig_no_foo to i8*, !dbg !51
  %16 = bitcast i64* @.C348_drb061_matrixvector1_orig_no_foo to i8*, !dbg !51
  %17 = bitcast i64* %z_b_0_309 to i8*, !dbg !51
  %18 = bitcast i64* %z_b_1_310 to i8*, !dbg !51
  %19 = bitcast i64* %z_b_3_312 to i8*, !dbg !51
  %20 = bitcast i64* %z_b_4_313 to i8*, !dbg !51
  %21 = bitcast void (...)* @f90_template2_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !51
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, ...) %21(i8* %13, i8* %14, i8* %15, i8* %16, i8* %17, i8* %18, i8* %19, i8* %20), !dbg !51
  %22 = bitcast [22 x i64]* %"a$sd1_346" to i8*, !dbg !51
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
  %46 = bitcast i64* @.C347_drb061_matrixvector1_orig_no_foo to i8*, !dbg !51
  %47 = bitcast i64* @.C348_drb061_matrixvector1_orig_no_foo to i8*, !dbg !51
  %48 = bitcast float** %.Z0977_332 to i8*, !dbg !51
  %49 = bitcast i64* @.C286_drb061_matrixvector1_orig_no_foo to i8*, !dbg !51
  %50 = bitcast i64* @.C284_drb061_matrixvector1_orig_no_foo to i8*, !dbg !51
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
  %55 = bitcast [16 x i64]* %"v$sd2_351" to i8*, !dbg !52
  %56 = bitcast i64* @.C284_drb061_matrixvector1_orig_no_foo to i8*, !dbg !52
  %57 = bitcast i64* @.C347_drb061_matrixvector1_orig_no_foo to i8*, !dbg !52
  %58 = bitcast i64* @.C348_drb061_matrixvector1_orig_no_foo to i8*, !dbg !52
  %59 = bitcast i64* %z_b_7_319 to i8*, !dbg !52
  %60 = bitcast i64* %z_b_8_320 to i8*, !dbg !52
  %61 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !52
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %61(i8* %55, i8* %56, i8* %57, i8* %58, i8* %59, i8* %60), !dbg !52
  %62 = bitcast [16 x i64]* %"v$sd2_351" to i8*, !dbg !52
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
  %70 = bitcast i64* @.C347_drb061_matrixvector1_orig_no_foo to i8*, !dbg !52
  %71 = bitcast i64* @.C348_drb061_matrixvector1_orig_no_foo to i8*, !dbg !52
  %72 = bitcast float** %.Z0983_333 to i8*, !dbg !52
  %73 = bitcast i64* @.C286_drb061_matrixvector1_orig_no_foo to i8*, !dbg !52
  %74 = bitcast i64* @.C284_drb061_matrixvector1_orig_no_foo to i8*, !dbg !52
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
  %79 = bitcast [16 x i64]* %"v_out$sd3_352" to i8*, !dbg !53
  %80 = bitcast i64* @.C284_drb061_matrixvector1_orig_no_foo to i8*, !dbg !53
  %81 = bitcast i64* @.C347_drb061_matrixvector1_orig_no_foo to i8*, !dbg !53
  %82 = bitcast i64* @.C348_drb061_matrixvector1_orig_no_foo to i8*, !dbg !53
  %83 = bitcast i64* %z_b_11_326 to i8*, !dbg !53
  %84 = bitcast i64* %z_b_12_327 to i8*, !dbg !53
  %85 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !53
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %85(i8* %79, i8* %80, i8* %81, i8* %82, i8* %83, i8* %84), !dbg !53
  %86 = bitcast [16 x i64]* %"v_out$sd3_352" to i8*, !dbg !53
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
  %94 = bitcast i64* @.C347_drb061_matrixvector1_orig_no_foo to i8*, !dbg !53
  %95 = bitcast i64* @.C348_drb061_matrixvector1_orig_no_foo to i8*, !dbg !53
  %96 = bitcast float** %.Z0984_334 to i8*, !dbg !53
  %97 = bitcast i64* @.C286_drb061_matrixvector1_orig_no_foo to i8*, !dbg !53
  %98 = bitcast i64* @.C284_drb061_matrixvector1_orig_no_foo to i8*, !dbg !53
  %99 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !53
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %99(i8* %93, i8* %94, i8* %95, i8* null, i8* %96, i8* null, i8* %97, i8* %98, i8* null, i64 0), !dbg !53
  %100 = bitcast i64* %.S0000 to i8*, !dbg !54
  %101 = bitcast %astruct.dt86* %.uplevelArgPack0001_419 to i8**, !dbg !54
  store i8* %100, i8** %101, align 8, !dbg !54
  %102 = bitcast i32* %n_308 to i8*, !dbg !54
  %103 = bitcast %astruct.dt86* %.uplevelArgPack0001_419 to i8*, !dbg !54
  %104 = getelementptr i8, i8* %103, i64 8, !dbg !54
  %105 = bitcast i8* %104 to i8**, !dbg !54
  store i8* %102, i8** %105, align 8, !dbg !54
  %106 = bitcast float** %.Z0977_332 to i8*, !dbg !54
  %107 = bitcast %astruct.dt86* %.uplevelArgPack0001_419 to i8*, !dbg !54
  %108 = getelementptr i8, i8* %107, i64 16, !dbg !54
  %109 = bitcast i8* %108 to i8**, !dbg !54
  store i8* %106, i8** %109, align 8, !dbg !54
  %110 = bitcast float** %.Z0977_332 to i8*, !dbg !54
  %111 = bitcast %astruct.dt86* %.uplevelArgPack0001_419 to i8*, !dbg !54
  %112 = getelementptr i8, i8* %111, i64 24, !dbg !54
  %113 = bitcast i8* %112 to i8**, !dbg !54
  store i8* %110, i8** %113, align 8, !dbg !54
  %114 = bitcast i64* %z_b_0_309 to i8*, !dbg !54
  %115 = bitcast %astruct.dt86* %.uplevelArgPack0001_419 to i8*, !dbg !54
  %116 = getelementptr i8, i8* %115, i64 32, !dbg !54
  %117 = bitcast i8* %116 to i8**, !dbg !54
  store i8* %114, i8** %117, align 8, !dbg !54
  %118 = bitcast i64* %z_b_1_310 to i8*, !dbg !54
  %119 = bitcast %astruct.dt86* %.uplevelArgPack0001_419 to i8*, !dbg !54
  %120 = getelementptr i8, i8* %119, i64 40, !dbg !54
  %121 = bitcast i8* %120 to i8**, !dbg !54
  store i8* %118, i8** %121, align 8, !dbg !54
  %122 = bitcast i64* %z_e_67_316 to i8*, !dbg !54
  %123 = bitcast %astruct.dt86* %.uplevelArgPack0001_419 to i8*, !dbg !54
  %124 = getelementptr i8, i8* %123, i64 48, !dbg !54
  %125 = bitcast i8* %124 to i8**, !dbg !54
  store i8* %122, i8** %125, align 8, !dbg !54
  %126 = bitcast i64* %z_b_3_312 to i8*, !dbg !54
  %127 = bitcast %astruct.dt86* %.uplevelArgPack0001_419 to i8*, !dbg !54
  %128 = getelementptr i8, i8* %127, i64 56, !dbg !54
  %129 = bitcast i8* %128 to i8**, !dbg !54
  store i8* %126, i8** %129, align 8, !dbg !54
  %130 = bitcast i64* %z_b_4_313 to i8*, !dbg !54
  %131 = bitcast %astruct.dt86* %.uplevelArgPack0001_419 to i8*, !dbg !54
  %132 = getelementptr i8, i8* %131, i64 64, !dbg !54
  %133 = bitcast i8* %132 to i8**, !dbg !54
  store i8* %130, i8** %133, align 8, !dbg !54
  %134 = bitcast i64* %z_b_2_311 to i8*, !dbg !54
  %135 = bitcast %astruct.dt86* %.uplevelArgPack0001_419 to i8*, !dbg !54
  %136 = getelementptr i8, i8* %135, i64 72, !dbg !54
  %137 = bitcast i8* %136 to i8**, !dbg !54
  store i8* %134, i8** %137, align 8, !dbg !54
  %138 = bitcast i64* %z_e_70_317 to i8*, !dbg !54
  %139 = bitcast %astruct.dt86* %.uplevelArgPack0001_419 to i8*, !dbg !54
  %140 = getelementptr i8, i8* %139, i64 80, !dbg !54
  %141 = bitcast i8* %140 to i8**, !dbg !54
  store i8* %138, i8** %141, align 8, !dbg !54
  %142 = bitcast i64* %z_b_5_314 to i8*, !dbg !54
  %143 = bitcast %astruct.dt86* %.uplevelArgPack0001_419 to i8*, !dbg !54
  %144 = getelementptr i8, i8* %143, i64 88, !dbg !54
  %145 = bitcast i8* %144 to i8**, !dbg !54
  store i8* %142, i8** %145, align 8, !dbg !54
  %146 = bitcast i64* %z_b_6_315 to i8*, !dbg !54
  %147 = bitcast %astruct.dt86* %.uplevelArgPack0001_419 to i8*, !dbg !54
  %148 = getelementptr i8, i8* %147, i64 96, !dbg !54
  %149 = bitcast i8* %148 to i8**, !dbg !54
  store i8* %146, i8** %149, align 8, !dbg !54
  %150 = bitcast float** %.Z0983_333 to i8*, !dbg !54
  %151 = bitcast %astruct.dt86* %.uplevelArgPack0001_419 to i8*, !dbg !54
  %152 = getelementptr i8, i8* %151, i64 104, !dbg !54
  %153 = bitcast i8* %152 to i8**, !dbg !54
  store i8* %150, i8** %153, align 8, !dbg !54
  %154 = bitcast float** %.Z0983_333 to i8*, !dbg !54
  %155 = bitcast %astruct.dt86* %.uplevelArgPack0001_419 to i8*, !dbg !54
  %156 = getelementptr i8, i8* %155, i64 112, !dbg !54
  %157 = bitcast i8* %156 to i8**, !dbg !54
  store i8* %154, i8** %157, align 8, !dbg !54
  %158 = bitcast i64* %z_b_7_319 to i8*, !dbg !54
  %159 = bitcast %astruct.dt86* %.uplevelArgPack0001_419 to i8*, !dbg !54
  %160 = getelementptr i8, i8* %159, i64 120, !dbg !54
  %161 = bitcast i8* %160 to i8**, !dbg !54
  store i8* %158, i8** %161, align 8, !dbg !54
  %162 = bitcast i64* %z_b_8_320 to i8*, !dbg !54
  %163 = bitcast %astruct.dt86* %.uplevelArgPack0001_419 to i8*, !dbg !54
  %164 = getelementptr i8, i8* %163, i64 128, !dbg !54
  %165 = bitcast i8* %164 to i8**, !dbg !54
  store i8* %162, i8** %165, align 8, !dbg !54
  %166 = bitcast i64* %z_e_77_323 to i8*, !dbg !54
  %167 = bitcast %astruct.dt86* %.uplevelArgPack0001_419 to i8*, !dbg !54
  %168 = getelementptr i8, i8* %167, i64 136, !dbg !54
  %169 = bitcast i8* %168 to i8**, !dbg !54
  store i8* %166, i8** %169, align 8, !dbg !54
  %170 = bitcast i64* %z_b_9_321 to i8*, !dbg !54
  %171 = bitcast %astruct.dt86* %.uplevelArgPack0001_419 to i8*, !dbg !54
  %172 = getelementptr i8, i8* %171, i64 144, !dbg !54
  %173 = bitcast i8* %172 to i8**, !dbg !54
  store i8* %170, i8** %173, align 8, !dbg !54
  %174 = bitcast i64* %z_b_10_322 to i8*, !dbg !54
  %175 = bitcast %astruct.dt86* %.uplevelArgPack0001_419 to i8*, !dbg !54
  %176 = getelementptr i8, i8* %175, i64 152, !dbg !54
  %177 = bitcast i8* %176 to i8**, !dbg !54
  store i8* %174, i8** %177, align 8, !dbg !54
  %178 = bitcast float** %.Z0984_334 to i8*, !dbg !54
  %179 = bitcast %astruct.dt86* %.uplevelArgPack0001_419 to i8*, !dbg !54
  %180 = getelementptr i8, i8* %179, i64 160, !dbg !54
  %181 = bitcast i8* %180 to i8**, !dbg !54
  store i8* %178, i8** %181, align 8, !dbg !54
  %182 = bitcast float** %.Z0984_334 to i8*, !dbg !54
  %183 = bitcast %astruct.dt86* %.uplevelArgPack0001_419 to i8*, !dbg !54
  %184 = getelementptr i8, i8* %183, i64 168, !dbg !54
  %185 = bitcast i8* %184 to i8**, !dbg !54
  store i8* %182, i8** %185, align 8, !dbg !54
  %186 = bitcast i64* %z_b_11_326 to i8*, !dbg !54
  %187 = bitcast %astruct.dt86* %.uplevelArgPack0001_419 to i8*, !dbg !54
  %188 = getelementptr i8, i8* %187, i64 176, !dbg !54
  %189 = bitcast i8* %188 to i8**, !dbg !54
  store i8* %186, i8** %189, align 8, !dbg !54
  %190 = bitcast i64* %z_b_12_327 to i8*, !dbg !54
  %191 = bitcast %astruct.dt86* %.uplevelArgPack0001_419 to i8*, !dbg !54
  %192 = getelementptr i8, i8* %191, i64 184, !dbg !54
  %193 = bitcast i8* %192 to i8**, !dbg !54
  store i8* %190, i8** %193, align 8, !dbg !54
  %194 = bitcast i64* %z_e_84_330 to i8*, !dbg !54
  %195 = bitcast %astruct.dt86* %.uplevelArgPack0001_419 to i8*, !dbg !54
  %196 = getelementptr i8, i8* %195, i64 192, !dbg !54
  %197 = bitcast i8* %196 to i8**, !dbg !54
  store i8* %194, i8** %197, align 8, !dbg !54
  %198 = bitcast i64* %z_b_13_328 to i8*, !dbg !54
  %199 = bitcast %astruct.dt86* %.uplevelArgPack0001_419 to i8*, !dbg !54
  %200 = getelementptr i8, i8* %199, i64 200, !dbg !54
  %201 = bitcast i8* %200 to i8**, !dbg !54
  store i8* %198, i8** %201, align 8, !dbg !54
  %202 = bitcast i64* %z_b_14_329 to i8*, !dbg !54
  %203 = bitcast %astruct.dt86* %.uplevelArgPack0001_419 to i8*, !dbg !54
  %204 = getelementptr i8, i8* %203, i64 208, !dbg !54
  %205 = bitcast i8* %204 to i8**, !dbg !54
  store i8* %202, i8** %205, align 8, !dbg !54
  %206 = bitcast [22 x i64]* %"a$sd1_346" to i8*, !dbg !54
  %207 = bitcast %astruct.dt86* %.uplevelArgPack0001_419 to i8*, !dbg !54
  %208 = getelementptr i8, i8* %207, i64 216, !dbg !54
  %209 = bitcast i8* %208 to i8**, !dbg !54
  store i8* %206, i8** %209, align 8, !dbg !54
  %210 = bitcast [16 x i64]* %"v$sd2_351" to i8*, !dbg !54
  %211 = bitcast %astruct.dt86* %.uplevelArgPack0001_419 to i8*, !dbg !54
  %212 = getelementptr i8, i8* %211, i64 224, !dbg !54
  %213 = bitcast i8* %212 to i8**, !dbg !54
  store i8* %210, i8** %213, align 8, !dbg !54
  %214 = bitcast [16 x i64]* %"v_out$sd3_352" to i8*, !dbg !54
  %215 = bitcast %astruct.dt86* %.uplevelArgPack0001_419 to i8*, !dbg !54
  %216 = getelementptr i8, i8* %215, i64 232, !dbg !54
  %217 = bitcast i8* %216 to i8**, !dbg !54
  store i8* %214, i8** %217, align 8, !dbg !54
  br label %L.LB2_481, !dbg !54

L.LB2_481:                                        ; preds = %L.LB2_390
  %218 = bitcast void (i32*, i64*, i64*)* @__nv_drb061_matrixvector1_orig_no_foo_F1L26_1_ to i64*, !dbg !54
  %219 = bitcast %astruct.dt86* %.uplevelArgPack0001_419 to i64*, !dbg !54
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %218, i64* %219), !dbg !54
  %220 = load float*, float** %.Z0977_332, align 8, !dbg !28
  call void @llvm.dbg.value(metadata float* %220, metadata !41, metadata !DIExpression()), !dbg !22
  %221 = bitcast float* %220 to i8*, !dbg !28
  %222 = bitcast i32 (...)* @f90_allocated_i8 to i32 (i8*, ...)*, !dbg !28
  %223 = call i32 (i8*, ...) %222(i8* %221), !dbg !28
  %224 = and i32 %223, 1, !dbg !28
  %225 = icmp eq i32 %224, 0, !dbg !28
  br i1 %225, label %L.LB2_372, label %L.LB2_503, !dbg !28

L.LB2_503:                                        ; preds = %L.LB2_481
  %226 = load float*, float** %.Z0977_332, align 8, !dbg !28
  call void @llvm.dbg.value(metadata float* %226, metadata !41, metadata !DIExpression()), !dbg !22
  %227 = bitcast float* %226 to i8*, !dbg !28
  %228 = bitcast i64* @.C284_drb061_matrixvector1_orig_no_foo to i8*, !dbg !28
  %229 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !28
  call void (i8*, i8*, i8*, i8*, i64, ...) %229(i8* null, i8* %227, i8* %228, i8* null, i64 0), !dbg !28
  %230 = bitcast float** %.Z0977_332 to i8**, !dbg !28
  store i8* null, i8** %230, align 8, !dbg !28
  %231 = bitcast [22 x i64]* %"a$sd1_346" to i64*, !dbg !28
  store i64 0, i64* %231, align 8, !dbg !28
  br label %L.LB2_372

L.LB2_372:                                        ; preds = %L.LB2_503, %L.LB2_481
  %232 = load float*, float** %.Z0983_333, align 8, !dbg !28
  call void @llvm.dbg.value(metadata float* %232, metadata !40, metadata !DIExpression()), !dbg !22
  %233 = bitcast float* %232 to i8*, !dbg !28
  %234 = bitcast i32 (...)* @f90_allocated_i8 to i32 (i8*, ...)*, !dbg !28
  %235 = call i32 (i8*, ...) %234(i8* %233), !dbg !28
  %236 = and i32 %235, 1, !dbg !28
  %237 = icmp eq i32 %236, 0, !dbg !28
  br i1 %237, label %L.LB2_375, label %L.LB2_504, !dbg !28

L.LB2_504:                                        ; preds = %L.LB2_372
  %238 = load float*, float** %.Z0983_333, align 8, !dbg !28
  call void @llvm.dbg.value(metadata float* %238, metadata !40, metadata !DIExpression()), !dbg !22
  %239 = bitcast float* %238 to i8*, !dbg !28
  %240 = bitcast i64* @.C284_drb061_matrixvector1_orig_no_foo to i8*, !dbg !28
  %241 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !28
  call void (i8*, i8*, i8*, i8*, i64, ...) %241(i8* null, i8* %239, i8* %240, i8* null, i64 0), !dbg !28
  %242 = bitcast float** %.Z0983_333 to i8**, !dbg !28
  store i8* null, i8** %242, align 8, !dbg !28
  %243 = bitcast [16 x i64]* %"v$sd2_351" to i64*, !dbg !28
  store i64 0, i64* %243, align 8, !dbg !28
  br label %L.LB2_375

L.LB2_375:                                        ; preds = %L.LB2_504, %L.LB2_372
  %244 = load float*, float** %.Z0984_334, align 8, !dbg !28
  call void @llvm.dbg.value(metadata float* %244, metadata !29, metadata !DIExpression()), !dbg !22
  %245 = bitcast float* %244 to i8*, !dbg !28
  %246 = bitcast i32 (...)* @f90_allocated_i8 to i32 (i8*, ...)*, !dbg !28
  %247 = call i32 (i8*, ...) %246(i8* %245), !dbg !28
  %248 = and i32 %247, 1, !dbg !28
  %249 = icmp eq i32 %248, 0, !dbg !28
  br i1 %249, label %L.LB2_376, label %L.LB2_505, !dbg !28

L.LB2_505:                                        ; preds = %L.LB2_375
  %250 = load float*, float** %.Z0984_334, align 8, !dbg !28
  call void @llvm.dbg.value(metadata float* %250, metadata !29, metadata !DIExpression()), !dbg !22
  %251 = bitcast float* %250 to i8*, !dbg !28
  %252 = bitcast i64* @.C284_drb061_matrixvector1_orig_no_foo to i8*, !dbg !28
  %253 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !28
  call void (i8*, i8*, i8*, i8*, i64, ...) %253(i8* null, i8* %251, i8* %252, i8* null, i64 0), !dbg !28
  %254 = bitcast float** %.Z0984_334 to i8**, !dbg !28
  store i8* null, i8** %254, align 8, !dbg !28
  %255 = bitcast [16 x i64]* %"v_out$sd3_352" to i64*, !dbg !28
  store i64 0, i64* %255, align 8, !dbg !28
  br label %L.LB2_376

L.LB2_376:                                        ; preds = %L.LB2_505, %L.LB2_375
  ret void, !dbg !28
}

define internal void @__nv_drb061_matrixvector1_orig_no_foo_F1L26_1_(i32* %__nv_drb061_matrixvector1_orig_no_foo_F1L26_1Arg0, i64* %__nv_drb061_matrixvector1_orig_no_foo_F1L26_1Arg1, i64* %__nv_drb061_matrixvector1_orig_no_foo_F1L26_1Arg2) #0 !dbg !55 {
L.entry:
  %.S0000_377 = alloca i8*, align 8
  %__gtid___nv_drb061_matrixvector1_orig_no_foo_F1L26_1__528 = alloca i32, align 4
  %.i0000p_342 = alloca i32, align 4
  %i_339 = alloca i32, align 4
  %.du0001p_360 = alloca i32, align 4
  %.de0001p_361 = alloca i32, align 4
  %.di0001p_362 = alloca i32, align 4
  %.ds0001p_363 = alloca i32, align 4
  %.dl0001p_365 = alloca i32, align 4
  %.dl0001p.copy_522 = alloca i32, align 4
  %.de0001p.copy_523 = alloca i32, align 4
  %.ds0001p.copy_524 = alloca i32, align 4
  %.dX0001p_364 = alloca i32, align 4
  %.dY0001p_359 = alloca i32, align 4
  %.dY0002p_371 = alloca i32, align 4
  %j_340 = alloca i32, align 4
  %sum_341 = alloca float, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_drb061_matrixvector1_orig_no_foo_F1L26_1Arg0, metadata !58, metadata !DIExpression()), !dbg !59
  call void @llvm.dbg.declare(metadata i64* %__nv_drb061_matrixvector1_orig_no_foo_F1L26_1Arg1, metadata !60, metadata !DIExpression()), !dbg !59
  call void @llvm.dbg.declare(metadata i64* %__nv_drb061_matrixvector1_orig_no_foo_F1L26_1Arg2, metadata !61, metadata !DIExpression()), !dbg !59
  call void @llvm.dbg.value(metadata i32 1, metadata !62, metadata !DIExpression()), !dbg !59
  call void @llvm.dbg.value(metadata i32 0, metadata !63, metadata !DIExpression()), !dbg !59
  call void @llvm.dbg.value(metadata i32 1, metadata !64, metadata !DIExpression()), !dbg !59
  call void @llvm.dbg.value(metadata i32 0, metadata !65, metadata !DIExpression()), !dbg !59
  call void @llvm.dbg.value(metadata i32 1, metadata !66, metadata !DIExpression()), !dbg !59
  %0 = bitcast i64* %__nv_drb061_matrixvector1_orig_no_foo_F1L26_1Arg2 to i8**, !dbg !67
  %1 = load i8*, i8** %0, align 8, !dbg !67
  %2 = bitcast i8** %.S0000_377 to i64*, !dbg !67
  store i8* %1, i8** %.S0000_377, align 8, !dbg !67
  %3 = load i32, i32* %__nv_drb061_matrixvector1_orig_no_foo_F1L26_1Arg0, align 4, !dbg !68
  store i32 %3, i32* %__gtid___nv_drb061_matrixvector1_orig_no_foo_F1L26_1__528, align 4, !dbg !68
  br label %L.LB3_513

L.LB3_513:                                        ; preds = %L.entry
  br label %L.LB3_338

L.LB3_338:                                        ; preds = %L.LB3_513
  store i32 0, i32* %.i0000p_342, align 4, !dbg !69
  call void @llvm.dbg.declare(metadata i32* %i_339, metadata !70, metadata !DIExpression()), !dbg !68
  store i32 1, i32* %i_339, align 4, !dbg !69
  %4 = bitcast i64* %__nv_drb061_matrixvector1_orig_no_foo_F1L26_1Arg2 to i8*, !dbg !69
  %5 = getelementptr i8, i8* %4, i64 8, !dbg !69
  %6 = bitcast i8* %5 to i32**, !dbg !69
  %7 = load i32*, i32** %6, align 8, !dbg !69
  %8 = load i32, i32* %7, align 4, !dbg !69
  store i32 %8, i32* %.du0001p_360, align 4, !dbg !69
  %9 = bitcast i64* %__nv_drb061_matrixvector1_orig_no_foo_F1L26_1Arg2 to i8*, !dbg !69
  %10 = getelementptr i8, i8* %9, i64 8, !dbg !69
  %11 = bitcast i8* %10 to i32**, !dbg !69
  %12 = load i32*, i32** %11, align 8, !dbg !69
  %13 = load i32, i32* %12, align 4, !dbg !69
  store i32 %13, i32* %.de0001p_361, align 4, !dbg !69
  store i32 1, i32* %.di0001p_362, align 4, !dbg !69
  %14 = load i32, i32* %.di0001p_362, align 4, !dbg !69
  store i32 %14, i32* %.ds0001p_363, align 4, !dbg !69
  store i32 1, i32* %.dl0001p_365, align 4, !dbg !69
  %15 = load i32, i32* %.dl0001p_365, align 4, !dbg !69
  store i32 %15, i32* %.dl0001p.copy_522, align 4, !dbg !69
  %16 = load i32, i32* %.de0001p_361, align 4, !dbg !69
  store i32 %16, i32* %.de0001p.copy_523, align 4, !dbg !69
  %17 = load i32, i32* %.ds0001p_363, align 4, !dbg !69
  store i32 %17, i32* %.ds0001p.copy_524, align 4, !dbg !69
  %18 = load i32, i32* %__gtid___nv_drb061_matrixvector1_orig_no_foo_F1L26_1__528, align 4, !dbg !69
  %19 = bitcast i32* %.i0000p_342 to i64*, !dbg !69
  %20 = bitcast i32* %.dl0001p.copy_522 to i64*, !dbg !69
  %21 = bitcast i32* %.de0001p.copy_523 to i64*, !dbg !69
  %22 = bitcast i32* %.ds0001p.copy_524 to i64*, !dbg !69
  %23 = load i32, i32* %.ds0001p.copy_524, align 4, !dbg !69
  call void @__kmpc_for_static_init_4(i64* null, i32 %18, i32 34, i64* %19, i64* %20, i64* %21, i64* %22, i32 %23, i32 1), !dbg !69
  %24 = load i32, i32* %.dl0001p.copy_522, align 4, !dbg !69
  store i32 %24, i32* %.dl0001p_365, align 4, !dbg !69
  %25 = load i32, i32* %.de0001p.copy_523, align 4, !dbg !69
  store i32 %25, i32* %.de0001p_361, align 4, !dbg !69
  %26 = load i32, i32* %.ds0001p.copy_524, align 4, !dbg !69
  store i32 %26, i32* %.ds0001p_363, align 4, !dbg !69
  %27 = load i32, i32* %.dl0001p_365, align 4, !dbg !69
  store i32 %27, i32* %i_339, align 4, !dbg !69
  %28 = load i32, i32* %i_339, align 4, !dbg !69
  call void @llvm.dbg.value(metadata i32 %28, metadata !70, metadata !DIExpression()), !dbg !68
  store i32 %28, i32* %.dX0001p_364, align 4, !dbg !69
  %29 = load i32, i32* %.dX0001p_364, align 4, !dbg !69
  %30 = load i32, i32* %.du0001p_360, align 4, !dbg !69
  %31 = icmp sgt i32 %29, %30, !dbg !69
  br i1 %31, label %L.LB3_358, label %L.LB3_564, !dbg !69

L.LB3_564:                                        ; preds = %L.LB3_338
  %32 = load i32, i32* %.dX0001p_364, align 4, !dbg !69
  store i32 %32, i32* %i_339, align 4, !dbg !69
  %33 = load i32, i32* %.di0001p_362, align 4, !dbg !69
  %34 = load i32, i32* %.de0001p_361, align 4, !dbg !69
  %35 = load i32, i32* %.dX0001p_364, align 4, !dbg !69
  %36 = sub nsw i32 %34, %35, !dbg !69
  %37 = add nsw i32 %33, %36, !dbg !69
  %38 = load i32, i32* %.di0001p_362, align 4, !dbg !69
  %39 = sdiv i32 %37, %38, !dbg !69
  store i32 %39, i32* %.dY0001p_359, align 4, !dbg !69
  %40 = load i32, i32* %.dY0001p_359, align 4, !dbg !69
  %41 = icmp sle i32 %40, 0, !dbg !69
  br i1 %41, label %L.LB3_368, label %L.LB3_367, !dbg !69

L.LB3_367:                                        ; preds = %L.LB3_370, %L.LB3_564
  %42 = bitcast i64* %__nv_drb061_matrixvector1_orig_no_foo_F1L26_1Arg2 to i8*, !dbg !71
  %43 = getelementptr i8, i8* %42, i64 8, !dbg !71
  %44 = bitcast i8* %43 to i32**, !dbg !71
  %45 = load i32*, i32** %44, align 8, !dbg !71
  %46 = load i32, i32* %45, align 4, !dbg !71
  store i32 %46, i32* %.dY0002p_371, align 4, !dbg !71
  call void @llvm.dbg.declare(metadata i32* %j_340, metadata !72, metadata !DIExpression()), !dbg !68
  store i32 1, i32* %j_340, align 4, !dbg !71
  %47 = load i32, i32* %.dY0002p_371, align 4, !dbg !71
  %48 = icmp sle i32 %47, 0, !dbg !71
  br i1 %48, label %L.LB3_370, label %L.LB3_369, !dbg !71

L.LB3_369:                                        ; preds = %L.LB3_369, %L.LB3_367
  call void @llvm.dbg.declare(metadata float* %sum_341, metadata !73, metadata !DIExpression()), !dbg !68
  %49 = load float, float* %sum_341, align 4, !dbg !74
  call void @llvm.dbg.value(metadata float %49, metadata !73, metadata !DIExpression()), !dbg !68
  %50 = load i32, i32* %j_340, align 4, !dbg !74
  call void @llvm.dbg.value(metadata i32 %50, metadata !72, metadata !DIExpression()), !dbg !68
  %51 = sext i32 %50 to i64, !dbg !74
  %52 = bitcast i64* %__nv_drb061_matrixvector1_orig_no_foo_F1L26_1Arg2 to i8*, !dbg !74
  %53 = getelementptr i8, i8* %52, i64 224, !dbg !74
  %54 = bitcast i8* %53 to i8**, !dbg !74
  %55 = load i8*, i8** %54, align 8, !dbg !74
  %56 = getelementptr i8, i8* %55, i64 56, !dbg !74
  %57 = bitcast i8* %56 to i64*, !dbg !74
  %58 = load i64, i64* %57, align 8, !dbg !74
  %59 = add nsw i64 %51, %58, !dbg !74
  %60 = bitcast i64* %__nv_drb061_matrixvector1_orig_no_foo_F1L26_1Arg2 to i8*, !dbg !74
  %61 = getelementptr i8, i8* %60, i64 112, !dbg !74
  %62 = bitcast i8* %61 to i8***, !dbg !74
  %63 = load i8**, i8*** %62, align 8, !dbg !74
  %64 = load i8*, i8** %63, align 8, !dbg !74
  %65 = getelementptr i8, i8* %64, i64 -4, !dbg !74
  %66 = bitcast i8* %65 to float*, !dbg !74
  %67 = getelementptr float, float* %66, i64 %59, !dbg !74
  %68 = load float, float* %67, align 4, !dbg !74
  %69 = load i32, i32* %i_339, align 4, !dbg !74
  call void @llvm.dbg.value(metadata i32 %69, metadata !70, metadata !DIExpression()), !dbg !68
  %70 = sext i32 %69 to i64, !dbg !74
  %71 = load i32, i32* %j_340, align 4, !dbg !74
  call void @llvm.dbg.value(metadata i32 %71, metadata !72, metadata !DIExpression()), !dbg !68
  %72 = sext i32 %71 to i64, !dbg !74
  %73 = bitcast i64* %__nv_drb061_matrixvector1_orig_no_foo_F1L26_1Arg2 to i8*, !dbg !74
  %74 = getelementptr i8, i8* %73, i64 216, !dbg !74
  %75 = bitcast i8* %74 to i8**, !dbg !74
  %76 = load i8*, i8** %75, align 8, !dbg !74
  %77 = getelementptr i8, i8* %76, i64 160, !dbg !74
  %78 = bitcast i8* %77 to i64*, !dbg !74
  %79 = load i64, i64* %78, align 8, !dbg !74
  %80 = mul nsw i64 %72, %79, !dbg !74
  %81 = add nsw i64 %70, %80, !dbg !74
  %82 = bitcast i64* %__nv_drb061_matrixvector1_orig_no_foo_F1L26_1Arg2 to i8*, !dbg !74
  %83 = getelementptr i8, i8* %82, i64 216, !dbg !74
  %84 = bitcast i8* %83 to i8**, !dbg !74
  %85 = load i8*, i8** %84, align 8, !dbg !74
  %86 = getelementptr i8, i8* %85, i64 56, !dbg !74
  %87 = bitcast i8* %86 to i64*, !dbg !74
  %88 = load i64, i64* %87, align 8, !dbg !74
  %89 = add nsw i64 %81, %88, !dbg !74
  %90 = bitcast i64* %__nv_drb061_matrixvector1_orig_no_foo_F1L26_1Arg2 to i8*, !dbg !74
  %91 = getelementptr i8, i8* %90, i64 24, !dbg !74
  %92 = bitcast i8* %91 to i8***, !dbg !74
  %93 = load i8**, i8*** %92, align 8, !dbg !74
  %94 = load i8*, i8** %93, align 8, !dbg !74
  %95 = getelementptr i8, i8* %94, i64 -4, !dbg !74
  %96 = bitcast i8* %95 to float*, !dbg !74
  %97 = getelementptr float, float* %96, i64 %89, !dbg !74
  %98 = load float, float* %97, align 4, !dbg !74
  %99 = fmul fast float %68, %98, !dbg !74
  %100 = fadd fast float %49, %99, !dbg !74
  store float %100, float* %sum_341, align 4, !dbg !74
  %101 = load i32, i32* %j_340, align 4, !dbg !75
  call void @llvm.dbg.value(metadata i32 %101, metadata !72, metadata !DIExpression()), !dbg !68
  %102 = add nsw i32 %101, 1, !dbg !75
  store i32 %102, i32* %j_340, align 4, !dbg !75
  %103 = load i32, i32* %.dY0002p_371, align 4, !dbg !75
  %104 = sub nsw i32 %103, 1, !dbg !75
  store i32 %104, i32* %.dY0002p_371, align 4, !dbg !75
  %105 = load i32, i32* %.dY0002p_371, align 4, !dbg !75
  %106 = icmp sgt i32 %105, 0, !dbg !75
  br i1 %106, label %L.LB3_369, label %L.LB3_370, !dbg !75

L.LB3_370:                                        ; preds = %L.LB3_369, %L.LB3_367
  %107 = load float, float* %sum_341, align 4, !dbg !76
  call void @llvm.dbg.value(metadata float %107, metadata !73, metadata !DIExpression()), !dbg !68
  %108 = load i32, i32* %i_339, align 4, !dbg !76
  call void @llvm.dbg.value(metadata i32 %108, metadata !70, metadata !DIExpression()), !dbg !68
  %109 = sext i32 %108 to i64, !dbg !76
  %110 = bitcast i64* %__nv_drb061_matrixvector1_orig_no_foo_F1L26_1Arg2 to i8*, !dbg !76
  %111 = getelementptr i8, i8* %110, i64 232, !dbg !76
  %112 = bitcast i8* %111 to i8**, !dbg !76
  %113 = load i8*, i8** %112, align 8, !dbg !76
  %114 = getelementptr i8, i8* %113, i64 56, !dbg !76
  %115 = bitcast i8* %114 to i64*, !dbg !76
  %116 = load i64, i64* %115, align 8, !dbg !76
  %117 = add nsw i64 %109, %116, !dbg !76
  %118 = bitcast i64* %__nv_drb061_matrixvector1_orig_no_foo_F1L26_1Arg2 to i8*, !dbg !76
  %119 = getelementptr i8, i8* %118, i64 168, !dbg !76
  %120 = bitcast i8* %119 to i8***, !dbg !76
  %121 = load i8**, i8*** %120, align 8, !dbg !76
  %122 = load i8*, i8** %121, align 8, !dbg !76
  %123 = getelementptr i8, i8* %122, i64 -4, !dbg !76
  %124 = bitcast i8* %123 to float*, !dbg !76
  %125 = getelementptr float, float* %124, i64 %117, !dbg !76
  store float %107, float* %125, align 4, !dbg !76
  %126 = load i32, i32* %.di0001p_362, align 4, !dbg !68
  %127 = load i32, i32* %i_339, align 4, !dbg !68
  call void @llvm.dbg.value(metadata i32 %127, metadata !70, metadata !DIExpression()), !dbg !68
  %128 = add nsw i32 %126, %127, !dbg !68
  store i32 %128, i32* %i_339, align 4, !dbg !68
  %129 = load i32, i32* %.dY0001p_359, align 4, !dbg !68
  %130 = sub nsw i32 %129, 1, !dbg !68
  store i32 %130, i32* %.dY0001p_359, align 4, !dbg !68
  %131 = load i32, i32* %.dY0001p_359, align 4, !dbg !68
  %132 = icmp sgt i32 %131, 0, !dbg !68
  br i1 %132, label %L.LB3_367, label %L.LB3_368, !dbg !68

L.LB3_368:                                        ; preds = %L.LB3_370, %L.LB3_564
  br label %L.LB3_358

L.LB3_358:                                        ; preds = %L.LB3_368, %L.LB3_338
  %133 = load i32, i32* %__gtid___nv_drb061_matrixvector1_orig_no_foo_F1L26_1__528, align 4, !dbg !68
  call void @__kmpc_for_static_fini(i64* null, i32 %133), !dbg !68
  br label %L.LB3_343

L.LB3_343:                                        ; preds = %L.LB3_358
  ret void, !dbg !68
}

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
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB061-matrixvector1-orig-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb061_matrixvector1_orig_no", scope: !2, file: !3, line: 10, type: !6, scopeLine: 10, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
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
!16 = !DILocation(line: 13, column: 1, scope: !5)
!17 = !DILocation(line: 14, column: 1, scope: !5)
!18 = distinct !DISubprogram(name: "foo", scope: !5, file: !3, line: 15, type: !19, scopeLine: 15, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!19 = !DISubroutineType(types: !7)
!20 = !DILocalVariable(arg: 1, scope: !18, file: !3, type: !21, flags: DIFlagArtificial)
!21 = !DIBasicType(name: "uinteger*8", size: 64, align: 64, encoding: DW_ATE_unsigned)
!22 = !DILocation(line: 0, scope: !18)
!23 = !DILocalVariable(name: "omp_sched_static", scope: !18, file: !3, type: !9)
!24 = !DILocalVariable(name: "omp_proc_bind_false", scope: !18, file: !3, type: !9)
!25 = !DILocalVariable(name: "omp_proc_bind_true", scope: !18, file: !3, type: !9)
!26 = !DILocalVariable(name: "omp_lock_hint_none", scope: !18, file: !3, type: !9)
!27 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !18, file: !3, type: !9)
!28 = !DILocation(line: 35, column: 1, scope: !18)
!29 = !DILocalVariable(name: "v_out", scope: !18, file: !3, type: !30)
!30 = !DICompositeType(tag: DW_TAG_array_type, baseType: !31, size: 32, align: 32, elements: !32)
!31 = !DIBasicType(name: "real", size: 32, align: 32, encoding: DW_ATE_float)
!32 = !{!33}
!33 = !DISubrange(count: 0, lowerBound: 1)
!34 = !DILocation(line: 15, column: 1, scope: !18)
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
!49 = !DILocation(line: 21, column: 1, scope: !18)
!50 = !DILocalVariable(scope: !18, file: !3, type: !37, flags: DIFlagArtificial)
!51 = !DILocation(line: 22, column: 1, scope: !18)
!52 = !DILocation(line: 23, column: 1, scope: !18)
!53 = !DILocation(line: 24, column: 1, scope: !18)
!54 = !DILocation(line: 26, column: 1, scope: !18)
!55 = distinct !DISubprogram(name: "__nv_drb061_matrixvector1_orig_no_foo_F1L26_1", scope: !2, file: !3, line: 26, type: !56, scopeLine: 26, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!56 = !DISubroutineType(types: !57)
!57 = !{null, !9, !37, !37}
!58 = !DILocalVariable(name: "__nv_drb061_matrixvector1_orig_no_foo_F1L26_1Arg0", arg: 1, scope: !55, file: !3, type: !9)
!59 = !DILocation(line: 0, scope: !55)
!60 = !DILocalVariable(name: "__nv_drb061_matrixvector1_orig_no_foo_F1L26_1Arg1", arg: 2, scope: !55, file: !3, type: !37)
!61 = !DILocalVariable(name: "__nv_drb061_matrixvector1_orig_no_foo_F1L26_1Arg2", arg: 3, scope: !55, file: !3, type: !37)
!62 = !DILocalVariable(name: "omp_sched_static", scope: !55, file: !3, type: !9)
!63 = !DILocalVariable(name: "omp_proc_bind_false", scope: !55, file: !3, type: !9)
!64 = !DILocalVariable(name: "omp_proc_bind_true", scope: !55, file: !3, type: !9)
!65 = !DILocalVariable(name: "omp_lock_hint_none", scope: !55, file: !3, type: !9)
!66 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !55, file: !3, type: !9)
!67 = !DILocation(line: 26, column: 1, scope: !55)
!68 = !DILocation(line: 32, column: 1, scope: !55)
!69 = !DILocation(line: 27, column: 1, scope: !55)
!70 = !DILocalVariable(name: "i", scope: !55, file: !3, type: !9)
!71 = !DILocation(line: 28, column: 1, scope: !55)
!72 = !DILocalVariable(name: "j", scope: !55, file: !3, type: !9)
!73 = !DILocalVariable(name: "sum", scope: !55, file: !3, type: !31)
!74 = !DILocation(line: 29, column: 1, scope: !55)
!75 = !DILocation(line: 30, column: 1, scope: !55)
!76 = !DILocation(line: 31, column: 1, scope: !55)

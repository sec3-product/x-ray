; ModuleID = '/tmp/DRB097-target-teams-distribute-orig-no-f83729.ll'
source_filename = "/tmp/DRB097-target-teams-distribute-orig-no-f83729.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct_iso_c_binding_10_ = type <{ [16 x i8] }>
%astruct.dt108 = type <{ i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8* }>
%astruct.dt114 = type <{ i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8* }>
%astruct.dt171 = type <{ [144 x i8] }>
%astruct.dt273 = type <{ [144 x i8], i8*, i8* }>

@.C422_MAIN_ = internal constant [8 x i8] c"; sum2 ="
@.C357_MAIN_ = internal constant i32 25
@.C356_MAIN_ = internal constant i32 14
@.C421_MAIN_ = internal constant [5 x i8] c"sum ="
@.C418_MAIN_ = internal constant i32 6
@.C415_MAIN_ = internal constant [67 x i8] c"micro-benchmarks-fortran/DRB097-target-teams-distribute-orig-no.f95"
@.C417_MAIN_ = internal constant i32 52
@.C300_MAIN_ = internal constant i32 4
@.C395_MAIN_ = internal constant i64 256
@.C388_MAIN_ = internal constant i32 256
@.C387_MAIN_ = internal constant i32 10
@.C285_MAIN_ = internal constant i32 1
@.C383_MAIN_ = internal constant double 3.000000e+00
@.C293_MAIN_ = internal constant double 2.000000e+00
@.C301_MAIN_ = internal constant i32 8
@.C358_MAIN_ = internal constant i32 28
@.C432_MAIN_ = internal constant i64 8
@.C431_MAIN_ = internal constant i64 28
@.C291_MAIN_ = internal constant double 0.000000e+00
@.C379_MAIN_ = internal constant i64 2560
@.C286_MAIN_ = internal constant i64 1
@.C284_MAIN_ = internal constant i64 0
@.C283_MAIN_ = internal constant i32 0
@.C300___nv_MAIN__F1L32_1 = internal constant i32 4
@.C395___nv_MAIN__F1L32_1 = internal constant i64 256
@.C286___nv_MAIN__F1L32_1 = internal constant i64 1
@.C283___nv_MAIN__F1L32_1 = internal constant i32 0
@.C291___nv_MAIN__F1L32_1 = internal constant double 0.000000e+00
@.C388___nv_MAIN__F1L32_1 = internal constant i32 256
@.C387___nv_MAIN__F1L32_1 = internal constant i32 10
@.C286___nv_MAIN__F1L46_2 = internal constant i64 1
@.C283___nv_MAIN__F1L46_2 = internal constant i32 0
@.C291___nv_MAIN__F1L46_2 = internal constant double 0.000000e+00
@.C300___nv_MAIN_F1L33_3 = internal constant i32 4
@.C395___nv_MAIN_F1L33_3 = internal constant i64 256
@.C286___nv_MAIN_F1L33_3 = internal constant i64 1
@.C283___nv_MAIN_F1L33_3 = internal constant i32 0
@.C291___nv_MAIN_F1L33_3 = internal constant double 0.000000e+00
@.C300___nv_MAIN_F1L36_4 = internal constant i32 4
@.C395___nv_MAIN_F1L36_4 = internal constant i64 256
@.C286___nv_MAIN_F1L36_4 = internal constant i64 1
@.C283___nv_MAIN_F1L36_4 = internal constant i32 0
@.C291___nv_MAIN_F1L36_4 = internal constant double 0.000000e+00
@_iso_c_binding_10_ = external global %struct_iso_c_binding_10_, align 64, !dbg !0, !dbg !7

define void @MAIN_() #0 !dbg !77 {
L.entry:
  %__gtid_MAIN__552 = alloca i32, align 4
  %.Z1054_382 = alloca double*, align 8
  %"b$sd2_434" = alloca [16 x i64], align 8
  %.Z1053_381 = alloca double*, align 8
  %"a$sd1_430" = alloca [16 x i64], align 8
  %len_378 = alloca i64, align 8
  %sum_380 = alloca double, align 8
  %sum2_365 = alloca double, align 8
  %z_b_0_366 = alloca i64, align 8
  %z_b_1_367 = alloca i64, align 8
  %z_e_119_370 = alloca i64, align 8
  %z_b_2_368 = alloca i64, align 8
  %z_b_3_369 = alloca i64, align 8
  %z_b_4_373 = alloca i64, align 8
  %z_b_5_374 = alloca i64, align 8
  %z_e_126_377 = alloca i64, align 8
  %z_b_6_375 = alloca i64, align 8
  %z_b_7_376 = alloca i64, align 8
  %.dY0001_442 = alloca i64, align 8
  %i_361 = alloca i64, align 8
  %.uplevelArgPack0001_515 = alloca %astruct.dt108, align 16
  %.uplevelArgPack0002_558 = alloca %astruct.dt114, align 16
  %z__io_420 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 4, metadata !85, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 8, metadata !88, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 8, metadata !89, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 8, metadata !90, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 8, metadata !91, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 1, metadata !92, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 1, metadata !93, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 4, metadata !94, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 8, metadata !95, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 1, metadata !96, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 4, metadata !97, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 8, metadata !98, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 1, metadata !99, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 8, metadata !100, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 8, metadata !101, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 8, metadata !102, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 8, metadata !103, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 4, metadata !104, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 8, metadata !105, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 8, metadata !106, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 4, metadata !107, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 8, metadata !108, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 8, metadata !109, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 1, metadata !110, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 1, metadata !111, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 4, metadata !112, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 4, metadata !113, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 4, metadata !114, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 8, metadata !115, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 4, metadata !116, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 4, metadata !117, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 4, metadata !118, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 4, metadata !119, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 1, metadata !120, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 4, metadata !121, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 0, metadata !122, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 1, metadata !123, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 4, metadata !124, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 0, metadata !125, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 1, metadata !126, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 4, metadata !127, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 8, metadata !128, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 8, metadata !129, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 8, metadata !130, metadata !DIExpression()), !dbg !87
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !131
  store i32 %0, i32* %__gtid_MAIN__552, align 4, !dbg !131
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !132
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !132
  call void (i8*, ...) %2(i8* %1), !dbg !132
  call void @llvm.dbg.declare(metadata double** %.Z1054_382, metadata !133, metadata !DIExpression(DW_OP_deref)), !dbg !87
  %3 = bitcast double** %.Z1054_382 to i8**, !dbg !132
  store i8* null, i8** %3, align 8, !dbg !132
  call void @llvm.dbg.declare(metadata [16 x i64]* %"b$sd2_434", metadata !136, metadata !DIExpression()), !dbg !87
  %4 = bitcast [16 x i64]* %"b$sd2_434" to i64*, !dbg !132
  store i64 0, i64* %4, align 8, !dbg !132
  call void @llvm.dbg.declare(metadata double** %.Z1053_381, metadata !140, metadata !DIExpression(DW_OP_deref)), !dbg !87
  %5 = bitcast double** %.Z1053_381 to i8**, !dbg !132
  store i8* null, i8** %5, align 8, !dbg !132
  call void @llvm.dbg.declare(metadata [16 x i64]* %"a$sd1_430", metadata !136, metadata !DIExpression()), !dbg !87
  %6 = bitcast [16 x i64]* %"a$sd1_430" to i64*, !dbg !132
  store i64 0, i64* %6, align 8, !dbg !132
  br label %L.LB1_486

L.LB1_486:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i64* %len_378, metadata !141, metadata !DIExpression()), !dbg !87
  store i64 2560, i64* %len_378, align 8, !dbg !142
  call void @llvm.dbg.declare(metadata double* %sum_380, metadata !143, metadata !DIExpression()), !dbg !87
  store double 0.000000e+00, double* %sum_380, align 8, !dbg !144
  call void @llvm.dbg.declare(metadata double* %sum2_365, metadata !145, metadata !DIExpression()), !dbg !87
  store double 0.000000e+00, double* %sum2_365, align 8, !dbg !146
  call void @llvm.dbg.declare(metadata i64* %z_b_0_366, metadata !147, metadata !DIExpression()), !dbg !87
  store i64 1, i64* %z_b_0_366, align 8, !dbg !148
  %7 = load i64, i64* %len_378, align 8, !dbg !148
  call void @llvm.dbg.value(metadata i64 %7, metadata !141, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.declare(metadata i64* %z_b_1_367, metadata !147, metadata !DIExpression()), !dbg !87
  store i64 %7, i64* %z_b_1_367, align 8, !dbg !148
  %8 = load i64, i64* %z_b_1_367, align 8, !dbg !148
  call void @llvm.dbg.value(metadata i64 %8, metadata !147, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.declare(metadata i64* %z_e_119_370, metadata !147, metadata !DIExpression()), !dbg !87
  store i64 %8, i64* %z_e_119_370, align 8, !dbg !148
  %9 = bitcast [16 x i64]* %"a$sd1_430" to i8*, !dbg !148
  %10 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !148
  %11 = bitcast i64* @.C431_MAIN_ to i8*, !dbg !148
  %12 = bitcast i64* @.C432_MAIN_ to i8*, !dbg !148
  %13 = bitcast i64* %z_b_0_366 to i8*, !dbg !148
  %14 = bitcast i64* %z_b_1_367 to i8*, !dbg !148
  %15 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !148
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %15(i8* %9, i8* %10, i8* %11, i8* %12, i8* %13, i8* %14), !dbg !148
  %16 = bitcast [16 x i64]* %"a$sd1_430" to i8*, !dbg !148
  %17 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !148
  call void (i8*, i32, ...) %17(i8* %16, i32 28), !dbg !148
  %18 = load i64, i64* %z_b_1_367, align 8, !dbg !148
  call void @llvm.dbg.value(metadata i64 %18, metadata !147, metadata !DIExpression()), !dbg !87
  %19 = load i64, i64* %z_b_0_366, align 8, !dbg !148
  call void @llvm.dbg.value(metadata i64 %19, metadata !147, metadata !DIExpression()), !dbg !87
  %20 = sub nsw i64 %19, 1, !dbg !148
  %21 = sub nsw i64 %18, %20, !dbg !148
  call void @llvm.dbg.declare(metadata i64* %z_b_2_368, metadata !147, metadata !DIExpression()), !dbg !87
  store i64 %21, i64* %z_b_2_368, align 8, !dbg !148
  %22 = load i64, i64* %z_b_0_366, align 8, !dbg !148
  call void @llvm.dbg.value(metadata i64 %22, metadata !147, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.declare(metadata i64* %z_b_3_369, metadata !147, metadata !DIExpression()), !dbg !87
  store i64 %22, i64* %z_b_3_369, align 8, !dbg !148
  %23 = bitcast i64* %z_b_2_368 to i8*, !dbg !148
  %24 = bitcast i64* @.C431_MAIN_ to i8*, !dbg !148
  %25 = bitcast i64* @.C432_MAIN_ to i8*, !dbg !148
  %26 = bitcast double** %.Z1053_381 to i8*, !dbg !148
  %27 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !148
  %28 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !148
  %29 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !148
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %29(i8* %23, i8* %24, i8* %25, i8* null, i8* %26, i8* null, i8* %27, i8* %28, i8* null, i64 0), !dbg !148
  call void @llvm.dbg.declare(metadata i64* %z_b_4_373, metadata !147, metadata !DIExpression()), !dbg !87
  store i64 1, i64* %z_b_4_373, align 8, !dbg !149
  %30 = load i64, i64* %len_378, align 8, !dbg !149
  call void @llvm.dbg.value(metadata i64 %30, metadata !141, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.declare(metadata i64* %z_b_5_374, metadata !147, metadata !DIExpression()), !dbg !87
  store i64 %30, i64* %z_b_5_374, align 8, !dbg !149
  %31 = load i64, i64* %z_b_5_374, align 8, !dbg !149
  call void @llvm.dbg.value(metadata i64 %31, metadata !147, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.declare(metadata i64* %z_e_126_377, metadata !147, metadata !DIExpression()), !dbg !87
  store i64 %31, i64* %z_e_126_377, align 8, !dbg !149
  %32 = bitcast [16 x i64]* %"b$sd2_434" to i8*, !dbg !149
  %33 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !149
  %34 = bitcast i64* @.C431_MAIN_ to i8*, !dbg !149
  %35 = bitcast i64* @.C432_MAIN_ to i8*, !dbg !149
  %36 = bitcast i64* %z_b_4_373 to i8*, !dbg !149
  %37 = bitcast i64* %z_b_5_374 to i8*, !dbg !149
  %38 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !149
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %38(i8* %32, i8* %33, i8* %34, i8* %35, i8* %36, i8* %37), !dbg !149
  %39 = bitcast [16 x i64]* %"b$sd2_434" to i8*, !dbg !149
  %40 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !149
  call void (i8*, i32, ...) %40(i8* %39, i32 28), !dbg !149
  %41 = load i64, i64* %z_b_5_374, align 8, !dbg !149
  call void @llvm.dbg.value(metadata i64 %41, metadata !147, metadata !DIExpression()), !dbg !87
  %42 = load i64, i64* %z_b_4_373, align 8, !dbg !149
  call void @llvm.dbg.value(metadata i64 %42, metadata !147, metadata !DIExpression()), !dbg !87
  %43 = sub nsw i64 %42, 1, !dbg !149
  %44 = sub nsw i64 %41, %43, !dbg !149
  call void @llvm.dbg.declare(metadata i64* %z_b_6_375, metadata !147, metadata !DIExpression()), !dbg !87
  store i64 %44, i64* %z_b_6_375, align 8, !dbg !149
  %45 = load i64, i64* %z_b_4_373, align 8, !dbg !149
  call void @llvm.dbg.value(metadata i64 %45, metadata !147, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.declare(metadata i64* %z_b_7_376, metadata !147, metadata !DIExpression()), !dbg !87
  store i64 %45, i64* %z_b_7_376, align 8, !dbg !149
  %46 = bitcast i64* %z_b_6_375 to i8*, !dbg !149
  %47 = bitcast i64* @.C431_MAIN_ to i8*, !dbg !149
  %48 = bitcast i64* @.C432_MAIN_ to i8*, !dbg !149
  %49 = bitcast double** %.Z1054_382 to i8*, !dbg !149
  %50 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !149
  %51 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !149
  %52 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !149
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %52(i8* %46, i8* %47, i8* %48, i8* null, i8* %49, i8* null, i8* %50, i8* %51, i8* null, i64 0), !dbg !149
  %53 = load i64, i64* %len_378, align 8, !dbg !150
  call void @llvm.dbg.value(metadata i64 %53, metadata !141, metadata !DIExpression()), !dbg !87
  store i64 %53, i64* %.dY0001_442, align 8, !dbg !150
  call void @llvm.dbg.declare(metadata i64* %i_361, metadata !151, metadata !DIExpression()), !dbg !87
  store i64 1, i64* %i_361, align 8, !dbg !150
  %54 = load i64, i64* %.dY0001_442, align 8, !dbg !150
  %55 = icmp sle i64 %54, 0, !dbg !150
  br i1 %55, label %L.LB1_441, label %L.LB1_440, !dbg !150

L.LB1_440:                                        ; preds = %L.LB1_440, %L.LB1_486
  %56 = load i64, i64* %i_361, align 8, !dbg !152
  call void @llvm.dbg.value(metadata i64 %56, metadata !151, metadata !DIExpression()), !dbg !87
  %57 = sitofp i64 %56 to double, !dbg !152
  %58 = fdiv fast double %57, 2.000000e+00, !dbg !152
  %59 = load i64, i64* %i_361, align 8, !dbg !152
  call void @llvm.dbg.value(metadata i64 %59, metadata !151, metadata !DIExpression()), !dbg !87
  %60 = bitcast [16 x i64]* %"a$sd1_430" to i8*, !dbg !152
  %61 = getelementptr i8, i8* %60, i64 56, !dbg !152
  %62 = bitcast i8* %61 to i64*, !dbg !152
  %63 = load i64, i64* %62, align 8, !dbg !152
  %64 = add nsw i64 %59, %63, !dbg !152
  %65 = load double*, double** %.Z1053_381, align 8, !dbg !152
  call void @llvm.dbg.value(metadata double* %65, metadata !140, metadata !DIExpression()), !dbg !87
  %66 = bitcast double* %65 to i8*, !dbg !152
  %67 = getelementptr i8, i8* %66, i64 -8, !dbg !152
  %68 = bitcast i8* %67 to double*, !dbg !152
  %69 = getelementptr double, double* %68, i64 %64, !dbg !152
  store double %58, double* %69, align 8, !dbg !152
  %70 = load i64, i64* %i_361, align 8, !dbg !153
  call void @llvm.dbg.value(metadata i64 %70, metadata !151, metadata !DIExpression()), !dbg !87
  %71 = sitofp i64 %70 to double, !dbg !153
  %72 = fdiv fast double %71, 3.000000e+00, !dbg !153
  %73 = load i64, i64* %i_361, align 8, !dbg !153
  call void @llvm.dbg.value(metadata i64 %73, metadata !151, metadata !DIExpression()), !dbg !87
  %74 = bitcast [16 x i64]* %"b$sd2_434" to i8*, !dbg !153
  %75 = getelementptr i8, i8* %74, i64 56, !dbg !153
  %76 = bitcast i8* %75 to i64*, !dbg !153
  %77 = load i64, i64* %76, align 8, !dbg !153
  %78 = add nsw i64 %73, %77, !dbg !153
  %79 = load double*, double** %.Z1054_382, align 8, !dbg !153
  call void @llvm.dbg.value(metadata double* %79, metadata !133, metadata !DIExpression()), !dbg !87
  %80 = bitcast double* %79 to i8*, !dbg !153
  %81 = getelementptr i8, i8* %80, i64 -8, !dbg !153
  %82 = bitcast i8* %81 to double*, !dbg !153
  %83 = getelementptr double, double* %82, i64 %78, !dbg !153
  store double %72, double* %83, align 8, !dbg !153
  %84 = load i64, i64* %i_361, align 8, !dbg !154
  call void @llvm.dbg.value(metadata i64 %84, metadata !151, metadata !DIExpression()), !dbg !87
  %85 = add nsw i64 %84, 1, !dbg !154
  store i64 %85, i64* %i_361, align 8, !dbg !154
  %86 = load i64, i64* %.dY0001_442, align 8, !dbg !154
  %87 = sub nsw i64 %86, 1, !dbg !154
  store i64 %87, i64* %.dY0001_442, align 8, !dbg !154
  %88 = load i64, i64* %.dY0001_442, align 8, !dbg !154
  %89 = icmp sgt i64 %88, 0, !dbg !154
  br i1 %89, label %L.LB1_440, label %L.LB1_441, !dbg !154

L.LB1_441:                                        ; preds = %L.LB1_440, %L.LB1_486
  %90 = bitcast double* %sum_380 to i8*, !dbg !155
  %91 = bitcast %astruct.dt108* %.uplevelArgPack0001_515 to i8**, !dbg !155
  store i8* %90, i8** %91, align 8, !dbg !155
  %92 = bitcast i64* %len_378 to i8*, !dbg !155
  %93 = bitcast %astruct.dt108* %.uplevelArgPack0001_515 to i8*, !dbg !155
  %94 = getelementptr i8, i8* %93, i64 8, !dbg !155
  %95 = bitcast i8* %94 to i8**, !dbg !155
  store i8* %92, i8** %95, align 8, !dbg !155
  %96 = bitcast double** %.Z1053_381 to i8*, !dbg !155
  %97 = bitcast %astruct.dt108* %.uplevelArgPack0001_515 to i8*, !dbg !155
  %98 = getelementptr i8, i8* %97, i64 16, !dbg !155
  %99 = bitcast i8* %98 to i8**, !dbg !155
  store i8* %96, i8** %99, align 8, !dbg !155
  %100 = bitcast double** %.Z1053_381 to i8*, !dbg !155
  %101 = bitcast %astruct.dt108* %.uplevelArgPack0001_515 to i8*, !dbg !155
  %102 = getelementptr i8, i8* %101, i64 24, !dbg !155
  %103 = bitcast i8* %102 to i8**, !dbg !155
  store i8* %100, i8** %103, align 8, !dbg !155
  %104 = bitcast i64* %z_b_0_366 to i8*, !dbg !155
  %105 = bitcast %astruct.dt108* %.uplevelArgPack0001_515 to i8*, !dbg !155
  %106 = getelementptr i8, i8* %105, i64 32, !dbg !155
  %107 = bitcast i8* %106 to i8**, !dbg !155
  store i8* %104, i8** %107, align 8, !dbg !155
  %108 = bitcast i64* %z_b_1_367 to i8*, !dbg !155
  %109 = bitcast %astruct.dt108* %.uplevelArgPack0001_515 to i8*, !dbg !155
  %110 = getelementptr i8, i8* %109, i64 40, !dbg !155
  %111 = bitcast i8* %110 to i8**, !dbg !155
  store i8* %108, i8** %111, align 8, !dbg !155
  %112 = bitcast i64* %z_e_119_370 to i8*, !dbg !155
  %113 = bitcast %astruct.dt108* %.uplevelArgPack0001_515 to i8*, !dbg !155
  %114 = getelementptr i8, i8* %113, i64 48, !dbg !155
  %115 = bitcast i8* %114 to i8**, !dbg !155
  store i8* %112, i8** %115, align 8, !dbg !155
  %116 = bitcast i64* %z_b_2_368 to i8*, !dbg !155
  %117 = bitcast %astruct.dt108* %.uplevelArgPack0001_515 to i8*, !dbg !155
  %118 = getelementptr i8, i8* %117, i64 56, !dbg !155
  %119 = bitcast i8* %118 to i8**, !dbg !155
  store i8* %116, i8** %119, align 8, !dbg !155
  %120 = bitcast i64* %z_b_3_369 to i8*, !dbg !155
  %121 = bitcast %astruct.dt108* %.uplevelArgPack0001_515 to i8*, !dbg !155
  %122 = getelementptr i8, i8* %121, i64 64, !dbg !155
  %123 = bitcast i8* %122 to i8**, !dbg !155
  store i8* %120, i8** %123, align 8, !dbg !155
  %124 = bitcast double** %.Z1054_382 to i8*, !dbg !155
  %125 = bitcast %astruct.dt108* %.uplevelArgPack0001_515 to i8*, !dbg !155
  %126 = getelementptr i8, i8* %125, i64 72, !dbg !155
  %127 = bitcast i8* %126 to i8**, !dbg !155
  store i8* %124, i8** %127, align 8, !dbg !155
  %128 = bitcast double** %.Z1054_382 to i8*, !dbg !155
  %129 = bitcast %astruct.dt108* %.uplevelArgPack0001_515 to i8*, !dbg !155
  %130 = getelementptr i8, i8* %129, i64 80, !dbg !155
  %131 = bitcast i8* %130 to i8**, !dbg !155
  store i8* %128, i8** %131, align 8, !dbg !155
  %132 = bitcast i64* %z_b_4_373 to i8*, !dbg !155
  %133 = bitcast %astruct.dt108* %.uplevelArgPack0001_515 to i8*, !dbg !155
  %134 = getelementptr i8, i8* %133, i64 88, !dbg !155
  %135 = bitcast i8* %134 to i8**, !dbg !155
  store i8* %132, i8** %135, align 8, !dbg !155
  %136 = bitcast i64* %z_b_5_374 to i8*, !dbg !155
  %137 = bitcast %astruct.dt108* %.uplevelArgPack0001_515 to i8*, !dbg !155
  %138 = getelementptr i8, i8* %137, i64 96, !dbg !155
  %139 = bitcast i8* %138 to i8**, !dbg !155
  store i8* %136, i8** %139, align 8, !dbg !155
  %140 = bitcast i64* %z_e_126_377 to i8*, !dbg !155
  %141 = bitcast %astruct.dt108* %.uplevelArgPack0001_515 to i8*, !dbg !155
  %142 = getelementptr i8, i8* %141, i64 104, !dbg !155
  %143 = bitcast i8* %142 to i8**, !dbg !155
  store i8* %140, i8** %143, align 8, !dbg !155
  %144 = bitcast i64* %z_b_6_375 to i8*, !dbg !155
  %145 = bitcast %astruct.dt108* %.uplevelArgPack0001_515 to i8*, !dbg !155
  %146 = getelementptr i8, i8* %145, i64 112, !dbg !155
  %147 = bitcast i8* %146 to i8**, !dbg !155
  store i8* %144, i8** %147, align 8, !dbg !155
  %148 = bitcast i64* %z_b_7_376 to i8*, !dbg !155
  %149 = bitcast %astruct.dt108* %.uplevelArgPack0001_515 to i8*, !dbg !155
  %150 = getelementptr i8, i8* %149, i64 120, !dbg !155
  %151 = bitcast i8* %150 to i8**, !dbg !155
  store i8* %148, i8** %151, align 8, !dbg !155
  %152 = bitcast [16 x i64]* %"a$sd1_430" to i8*, !dbg !155
  %153 = bitcast %astruct.dt108* %.uplevelArgPack0001_515 to i8*, !dbg !155
  %154 = getelementptr i8, i8* %153, i64 128, !dbg !155
  %155 = bitcast i8* %154 to i8**, !dbg !155
  store i8* %152, i8** %155, align 8, !dbg !155
  %156 = bitcast [16 x i64]* %"b$sd2_434" to i8*, !dbg !155
  %157 = bitcast %astruct.dt108* %.uplevelArgPack0001_515 to i8*, !dbg !155
  %158 = getelementptr i8, i8* %157, i64 136, !dbg !155
  %159 = bitcast i8* %158 to i8**, !dbg !155
  store i8* %156, i8** %159, align 8, !dbg !155
  %160 = bitcast %astruct.dt108* %.uplevelArgPack0001_515 to i64*, !dbg !155
  call void @__nv_MAIN__F1L32_1_(i32* %__gtid_MAIN__552, i64* null, i64* %160), !dbg !155
  %161 = bitcast double* %sum2_365 to i8*, !dbg !156
  %162 = bitcast %astruct.dt114* %.uplevelArgPack0002_558 to i8**, !dbg !156
  store i8* %161, i8** %162, align 8, !dbg !156
  %163 = bitcast i64* %len_378 to i8*, !dbg !156
  %164 = bitcast %astruct.dt114* %.uplevelArgPack0002_558 to i8*, !dbg !156
  %165 = getelementptr i8, i8* %164, i64 8, !dbg !156
  %166 = bitcast i8* %165 to i8**, !dbg !156
  store i8* %163, i8** %166, align 8, !dbg !156
  %167 = bitcast double** %.Z1053_381 to i8*, !dbg !156
  %168 = bitcast %astruct.dt114* %.uplevelArgPack0002_558 to i8*, !dbg !156
  %169 = getelementptr i8, i8* %168, i64 16, !dbg !156
  %170 = bitcast i8* %169 to i8**, !dbg !156
  store i8* %167, i8** %170, align 8, !dbg !156
  %171 = bitcast double** %.Z1053_381 to i8*, !dbg !156
  %172 = bitcast %astruct.dt114* %.uplevelArgPack0002_558 to i8*, !dbg !156
  %173 = getelementptr i8, i8* %172, i64 24, !dbg !156
  %174 = bitcast i8* %173 to i8**, !dbg !156
  store i8* %171, i8** %174, align 8, !dbg !156
  %175 = bitcast i64* %z_b_0_366 to i8*, !dbg !156
  %176 = bitcast %astruct.dt114* %.uplevelArgPack0002_558 to i8*, !dbg !156
  %177 = getelementptr i8, i8* %176, i64 32, !dbg !156
  %178 = bitcast i8* %177 to i8**, !dbg !156
  store i8* %175, i8** %178, align 8, !dbg !156
  %179 = bitcast i64* %z_b_1_367 to i8*, !dbg !156
  %180 = bitcast %astruct.dt114* %.uplevelArgPack0002_558 to i8*, !dbg !156
  %181 = getelementptr i8, i8* %180, i64 40, !dbg !156
  %182 = bitcast i8* %181 to i8**, !dbg !156
  store i8* %179, i8** %182, align 8, !dbg !156
  %183 = bitcast i64* %z_e_119_370 to i8*, !dbg !156
  %184 = bitcast %astruct.dt114* %.uplevelArgPack0002_558 to i8*, !dbg !156
  %185 = getelementptr i8, i8* %184, i64 48, !dbg !156
  %186 = bitcast i8* %185 to i8**, !dbg !156
  store i8* %183, i8** %186, align 8, !dbg !156
  %187 = bitcast i64* %z_b_2_368 to i8*, !dbg !156
  %188 = bitcast %astruct.dt114* %.uplevelArgPack0002_558 to i8*, !dbg !156
  %189 = getelementptr i8, i8* %188, i64 56, !dbg !156
  %190 = bitcast i8* %189 to i8**, !dbg !156
  store i8* %187, i8** %190, align 8, !dbg !156
  %191 = bitcast i64* %z_b_3_369 to i8*, !dbg !156
  %192 = bitcast %astruct.dt114* %.uplevelArgPack0002_558 to i8*, !dbg !156
  %193 = getelementptr i8, i8* %192, i64 64, !dbg !156
  %194 = bitcast i8* %193 to i8**, !dbg !156
  store i8* %191, i8** %194, align 8, !dbg !156
  %195 = bitcast double** %.Z1054_382 to i8*, !dbg !156
  %196 = bitcast %astruct.dt114* %.uplevelArgPack0002_558 to i8*, !dbg !156
  %197 = getelementptr i8, i8* %196, i64 72, !dbg !156
  %198 = bitcast i8* %197 to i8**, !dbg !156
  store i8* %195, i8** %198, align 8, !dbg !156
  %199 = bitcast double** %.Z1054_382 to i8*, !dbg !156
  %200 = bitcast %astruct.dt114* %.uplevelArgPack0002_558 to i8*, !dbg !156
  %201 = getelementptr i8, i8* %200, i64 80, !dbg !156
  %202 = bitcast i8* %201 to i8**, !dbg !156
  store i8* %199, i8** %202, align 8, !dbg !156
  %203 = bitcast i64* %z_b_4_373 to i8*, !dbg !156
  %204 = bitcast %astruct.dt114* %.uplevelArgPack0002_558 to i8*, !dbg !156
  %205 = getelementptr i8, i8* %204, i64 88, !dbg !156
  %206 = bitcast i8* %205 to i8**, !dbg !156
  store i8* %203, i8** %206, align 8, !dbg !156
  %207 = bitcast i64* %z_b_5_374 to i8*, !dbg !156
  %208 = bitcast %astruct.dt114* %.uplevelArgPack0002_558 to i8*, !dbg !156
  %209 = getelementptr i8, i8* %208, i64 96, !dbg !156
  %210 = bitcast i8* %209 to i8**, !dbg !156
  store i8* %207, i8** %210, align 8, !dbg !156
  %211 = bitcast i64* %z_e_126_377 to i8*, !dbg !156
  %212 = bitcast %astruct.dt114* %.uplevelArgPack0002_558 to i8*, !dbg !156
  %213 = getelementptr i8, i8* %212, i64 104, !dbg !156
  %214 = bitcast i8* %213 to i8**, !dbg !156
  store i8* %211, i8** %214, align 8, !dbg !156
  %215 = bitcast i64* %z_b_6_375 to i8*, !dbg !156
  %216 = bitcast %astruct.dt114* %.uplevelArgPack0002_558 to i8*, !dbg !156
  %217 = getelementptr i8, i8* %216, i64 112, !dbg !156
  %218 = bitcast i8* %217 to i8**, !dbg !156
  store i8* %215, i8** %218, align 8, !dbg !156
  %219 = bitcast i64* %z_b_7_376 to i8*, !dbg !156
  %220 = bitcast %astruct.dt114* %.uplevelArgPack0002_558 to i8*, !dbg !156
  %221 = getelementptr i8, i8* %220, i64 120, !dbg !156
  %222 = bitcast i8* %221 to i8**, !dbg !156
  store i8* %219, i8** %222, align 8, !dbg !156
  %223 = bitcast [16 x i64]* %"a$sd1_430" to i8*, !dbg !156
  %224 = bitcast %astruct.dt114* %.uplevelArgPack0002_558 to i8*, !dbg !156
  %225 = getelementptr i8, i8* %224, i64 128, !dbg !156
  %226 = bitcast i8* %225 to i8**, !dbg !156
  store i8* %223, i8** %226, align 8, !dbg !156
  %227 = bitcast [16 x i64]* %"b$sd2_434" to i8*, !dbg !156
  %228 = bitcast %astruct.dt114* %.uplevelArgPack0002_558 to i8*, !dbg !156
  %229 = getelementptr i8, i8* %228, i64 136, !dbg !156
  %230 = bitcast i8* %229 to i8**, !dbg !156
  store i8* %227, i8** %230, align 8, !dbg !156
  br label %L.LB1_595, !dbg !156

L.LB1_595:                                        ; preds = %L.LB1_441
  %231 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L46_2_ to i64*, !dbg !156
  %232 = bitcast %astruct.dt114* %.uplevelArgPack0002_558 to i64*, !dbg !156
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %231, i64* %232), !dbg !156
  call void (...) @_mp_bcs_nest(), !dbg !157
  %233 = bitcast i32* @.C417_MAIN_ to i8*, !dbg !157
  %234 = bitcast [67 x i8]* @.C415_MAIN_ to i8*, !dbg !157
  %235 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !157
  call void (i8*, i8*, i64, ...) %235(i8* %233, i8* %234, i64 67), !dbg !157
  %236 = bitcast i32* @.C418_MAIN_ to i8*, !dbg !157
  %237 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !157
  %238 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !157
  %239 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !157
  %240 = call i32 (i8*, i8*, i8*, i8*, ...) %239(i8* %236, i8* null, i8* %237, i8* %238), !dbg !157
  call void @llvm.dbg.declare(metadata i32* %z__io_420, metadata !158, metadata !DIExpression()), !dbg !87
  store i32 %240, i32* %z__io_420, align 4, !dbg !157
  %241 = bitcast [5 x i8]* @.C421_MAIN_ to i8*, !dbg !157
  %242 = bitcast i32 (...)* @f90io_sc_ch_ldw to i32 (i8*, i32, i64, ...)*, !dbg !157
  %243 = call i32 (i8*, i32, i64, ...) %242(i8* %241, i32 14, i64 5), !dbg !157
  store i32 %243, i32* %z__io_420, align 4, !dbg !157
  %244 = load double, double* %sum_380, align 8, !dbg !157
  call void @llvm.dbg.value(metadata double %244, metadata !143, metadata !DIExpression()), !dbg !87
  %245 = fptosi double %244 to i32, !dbg !157
  %246 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !157
  %247 = call i32 (i32, i32, ...) %246(i32 %245, i32 25), !dbg !157
  store i32 %247, i32* %z__io_420, align 4, !dbg !157
  %248 = bitcast [8 x i8]* @.C422_MAIN_ to i8*, !dbg !157
  %249 = bitcast i32 (...)* @f90io_sc_ch_ldw to i32 (i8*, i32, i64, ...)*, !dbg !157
  %250 = call i32 (i8*, i32, i64, ...) %249(i8* %248, i32 14, i64 8), !dbg !157
  store i32 %250, i32* %z__io_420, align 4, !dbg !157
  %251 = load double, double* %sum2_365, align 8, !dbg !157
  call void @llvm.dbg.value(metadata double %251, metadata !145, metadata !DIExpression()), !dbg !87
  %252 = fptosi double %251 to i32, !dbg !157
  %253 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !157
  %254 = call i32 (i32, i32, ...) %253(i32 %252, i32 25), !dbg !157
  store i32 %254, i32* %z__io_420, align 4, !dbg !157
  %255 = call i32 (...) @f90io_ldw_end(), !dbg !157
  store i32 %255, i32* %z__io_420, align 4, !dbg !157
  call void (...) @_mp_ecs_nest(), !dbg !157
  %256 = load double*, double** %.Z1053_381, align 8, !dbg !159
  call void @llvm.dbg.value(metadata double* %256, metadata !140, metadata !DIExpression()), !dbg !87
  %257 = bitcast double* %256 to i8*, !dbg !159
  %258 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !159
  %259 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !159
  call void (i8*, i8*, i8*, i8*, i64, ...) %259(i8* null, i8* %257, i8* %258, i8* null, i64 0), !dbg !159
  %260 = bitcast double** %.Z1053_381 to i8**, !dbg !159
  store i8* null, i8** %260, align 8, !dbg !159
  %261 = bitcast [16 x i64]* %"a$sd1_430" to i64*, !dbg !159
  store i64 0, i64* %261, align 8, !dbg !159
  %262 = load double*, double** %.Z1054_382, align 8, !dbg !159
  call void @llvm.dbg.value(metadata double* %262, metadata !133, metadata !DIExpression()), !dbg !87
  %263 = bitcast double* %262 to i8*, !dbg !159
  %264 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !159
  %265 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !159
  call void (i8*, i8*, i8*, i8*, i64, ...) %265(i8* null, i8* %263, i8* %264, i8* null, i64 0), !dbg !159
  %266 = bitcast double** %.Z1054_382 to i8**, !dbg !159
  store i8* null, i8** %266, align 8, !dbg !159
  %267 = bitcast [16 x i64]* %"b$sd2_434" to i64*, !dbg !159
  store i64 0, i64* %267, align 8, !dbg !159
  ret void, !dbg !131
}

define internal void @__nv_MAIN__F1L32_1_(i32* %__nv_MAIN__F1L32_1Arg0, i64* %__nv_MAIN__F1L32_1Arg1, i64* %__nv_MAIN__F1L32_1Arg2) #0 !dbg !160 {
L.entry:
  %__gtid___nv_MAIN__F1L32_1__636 = alloca i32, align 4
  %.uplevelArgPack0003_632 = alloca %astruct.dt171, align 16
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L32_1Arg0, metadata !163, metadata !DIExpression()), !dbg !164
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L32_1Arg1, metadata !165, metadata !DIExpression()), !dbg !164
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L32_1Arg2, metadata !166, metadata !DIExpression()), !dbg !164
  call void @llvm.dbg.value(metadata i32 4, metadata !167, metadata !DIExpression()), !dbg !164
  call void @llvm.dbg.value(metadata i32 8, metadata !168, metadata !DIExpression()), !dbg !164
  call void @llvm.dbg.value(metadata i32 8, metadata !169, metadata !DIExpression()), !dbg !164
  call void @llvm.dbg.value(metadata i32 8, metadata !170, metadata !DIExpression()), !dbg !164
  call void @llvm.dbg.value(metadata i32 8, metadata !171, metadata !DIExpression()), !dbg !164
  call void @llvm.dbg.value(metadata i32 1, metadata !172, metadata !DIExpression()), !dbg !164
  call void @llvm.dbg.value(metadata i32 1, metadata !173, metadata !DIExpression()), !dbg !164
  call void @llvm.dbg.value(metadata i32 4, metadata !174, metadata !DIExpression()), !dbg !164
  call void @llvm.dbg.value(metadata i32 8, metadata !175, metadata !DIExpression()), !dbg !164
  call void @llvm.dbg.value(metadata i32 1, metadata !176, metadata !DIExpression()), !dbg !164
  call void @llvm.dbg.value(metadata i32 4, metadata !177, metadata !DIExpression()), !dbg !164
  call void @llvm.dbg.value(metadata i32 8, metadata !178, metadata !DIExpression()), !dbg !164
  call void @llvm.dbg.value(metadata i32 1, metadata !179, metadata !DIExpression()), !dbg !164
  call void @llvm.dbg.value(metadata i32 8, metadata !180, metadata !DIExpression()), !dbg !164
  call void @llvm.dbg.value(metadata i32 8, metadata !181, metadata !DIExpression()), !dbg !164
  call void @llvm.dbg.value(metadata i32 8, metadata !182, metadata !DIExpression()), !dbg !164
  call void @llvm.dbg.value(metadata i32 8, metadata !183, metadata !DIExpression()), !dbg !164
  call void @llvm.dbg.value(metadata i32 4, metadata !184, metadata !DIExpression()), !dbg !164
  call void @llvm.dbg.value(metadata i32 8, metadata !185, metadata !DIExpression()), !dbg !164
  call void @llvm.dbg.value(metadata i32 8, metadata !186, metadata !DIExpression()), !dbg !164
  call void @llvm.dbg.value(metadata i32 4, metadata !187, metadata !DIExpression()), !dbg !164
  call void @llvm.dbg.value(metadata i32 8, metadata !188, metadata !DIExpression()), !dbg !164
  call void @llvm.dbg.value(metadata i32 8, metadata !189, metadata !DIExpression()), !dbg !164
  call void @llvm.dbg.value(metadata i32 1, metadata !190, metadata !DIExpression()), !dbg !164
  call void @llvm.dbg.value(metadata i32 1, metadata !191, metadata !DIExpression()), !dbg !164
  call void @llvm.dbg.value(metadata i32 4, metadata !192, metadata !DIExpression()), !dbg !164
  call void @llvm.dbg.value(metadata i32 4, metadata !193, metadata !DIExpression()), !dbg !164
  call void @llvm.dbg.value(metadata i32 4, metadata !194, metadata !DIExpression()), !dbg !164
  call void @llvm.dbg.value(metadata i32 8, metadata !195, metadata !DIExpression()), !dbg !164
  call void @llvm.dbg.value(metadata i32 4, metadata !196, metadata !DIExpression()), !dbg !164
  call void @llvm.dbg.value(metadata i32 4, metadata !197, metadata !DIExpression()), !dbg !164
  call void @llvm.dbg.value(metadata i32 4, metadata !198, metadata !DIExpression()), !dbg !164
  call void @llvm.dbg.value(metadata i32 4, metadata !199, metadata !DIExpression()), !dbg !164
  call void @llvm.dbg.value(metadata i32 1, metadata !200, metadata !DIExpression()), !dbg !164
  call void @llvm.dbg.value(metadata i32 4, metadata !201, metadata !DIExpression()), !dbg !164
  call void @llvm.dbg.value(metadata i32 0, metadata !202, metadata !DIExpression()), !dbg !164
  call void @llvm.dbg.value(metadata i32 1, metadata !203, metadata !DIExpression()), !dbg !164
  call void @llvm.dbg.value(metadata i32 4, metadata !204, metadata !DIExpression()), !dbg !164
  call void @llvm.dbg.value(metadata i32 0, metadata !205, metadata !DIExpression()), !dbg !164
  call void @llvm.dbg.value(metadata i32 1, metadata !206, metadata !DIExpression()), !dbg !164
  call void @llvm.dbg.value(metadata i32 4, metadata !207, metadata !DIExpression()), !dbg !164
  call void @llvm.dbg.value(metadata i32 8, metadata !208, metadata !DIExpression()), !dbg !164
  call void @llvm.dbg.value(metadata i32 8, metadata !209, metadata !DIExpression()), !dbg !164
  call void @llvm.dbg.value(metadata i32 8, metadata !210, metadata !DIExpression()), !dbg !164
  %0 = load i32, i32* %__nv_MAIN__F1L32_1Arg0, align 4, !dbg !211
  store i32 %0, i32* %__gtid___nv_MAIN__F1L32_1__636, align 4, !dbg !211
  br label %L.LB2_627

L.LB2_627:                                        ; preds = %L.entry
  br label %L.LB2_386

L.LB2_386:                                        ; preds = %L.LB2_627
  %1 = bitcast i64* %__nv_MAIN__F1L32_1Arg2 to i8*, !dbg !212
  %2 = bitcast %astruct.dt171* %.uplevelArgPack0003_632 to i8*, !dbg !212
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %2, i8* align 8 %1, i64 144, i1 false), !dbg !212
  %3 = load i32, i32* %__gtid___nv_MAIN__F1L32_1__636, align 4, !dbg !212
  call void @__kmpc_push_num_teams(i64* null, i32 %3, i32 10, i32 256), !dbg !212
  %4 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN_F1L33_3_ to i64*, !dbg !212
  %5 = bitcast %astruct.dt171* %.uplevelArgPack0003_632 to i64*, !dbg !212
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_teams(i64* null, i32 1, i64* %4, i64* %5), !dbg !212
  br label %L.LB2_406

L.LB2_406:                                        ; preds = %L.LB2_386
  ret void, !dbg !211
}

define internal void @__nv_MAIN__F1L46_2_(i32* %__nv_MAIN__F1L46_2Arg0, i64* %__nv_MAIN__F1L46_2Arg1, i64* %__nv_MAIN__F1L46_2Arg2) #0 !dbg !213 {
L.entry:
  %__gtid___nv_MAIN__F1L46_2__673 = alloca i32, align 4
  %sum2_410 = alloca double, align 8
  %.i0002p_412 = alloca i32, align 4
  %i_411 = alloca i64, align 8
  %.du0004p_470 = alloca i64, align 8
  %.de0004p_471 = alloca i64, align 8
  %.di0004p_472 = alloca i64, align 8
  %.ds0004p_473 = alloca i64, align 8
  %.dl0004p_475 = alloca i64, align 8
  %.dl0004p.copy_667 = alloca i64, align 8
  %.de0004p.copy_668 = alloca i64, align 8
  %.ds0004p.copy_669 = alloca i64, align 8
  %.dX0004p_474 = alloca i64, align 8
  %.dY0004p_469 = alloca i64, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L46_2Arg0, metadata !214, metadata !DIExpression()), !dbg !215
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L46_2Arg1, metadata !216, metadata !DIExpression()), !dbg !215
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L46_2Arg2, metadata !217, metadata !DIExpression()), !dbg !215
  call void @llvm.dbg.value(metadata i32 4, metadata !218, metadata !DIExpression()), !dbg !215
  call void @llvm.dbg.value(metadata i32 8, metadata !219, metadata !DIExpression()), !dbg !215
  call void @llvm.dbg.value(metadata i32 8, metadata !220, metadata !DIExpression()), !dbg !215
  call void @llvm.dbg.value(metadata i32 8, metadata !221, metadata !DIExpression()), !dbg !215
  call void @llvm.dbg.value(metadata i32 8, metadata !222, metadata !DIExpression()), !dbg !215
  call void @llvm.dbg.value(metadata i32 1, metadata !223, metadata !DIExpression()), !dbg !215
  call void @llvm.dbg.value(metadata i32 1, metadata !224, metadata !DIExpression()), !dbg !215
  call void @llvm.dbg.value(metadata i32 4, metadata !225, metadata !DIExpression()), !dbg !215
  call void @llvm.dbg.value(metadata i32 8, metadata !226, metadata !DIExpression()), !dbg !215
  call void @llvm.dbg.value(metadata i32 1, metadata !227, metadata !DIExpression()), !dbg !215
  call void @llvm.dbg.value(metadata i32 4, metadata !228, metadata !DIExpression()), !dbg !215
  call void @llvm.dbg.value(metadata i32 8, metadata !229, metadata !DIExpression()), !dbg !215
  call void @llvm.dbg.value(metadata i32 1, metadata !230, metadata !DIExpression()), !dbg !215
  call void @llvm.dbg.value(metadata i32 8, metadata !231, metadata !DIExpression()), !dbg !215
  call void @llvm.dbg.value(metadata i32 8, metadata !232, metadata !DIExpression()), !dbg !215
  call void @llvm.dbg.value(metadata i32 8, metadata !233, metadata !DIExpression()), !dbg !215
  call void @llvm.dbg.value(metadata i32 8, metadata !234, metadata !DIExpression()), !dbg !215
  call void @llvm.dbg.value(metadata i32 4, metadata !235, metadata !DIExpression()), !dbg !215
  call void @llvm.dbg.value(metadata i32 8, metadata !236, metadata !DIExpression()), !dbg !215
  call void @llvm.dbg.value(metadata i32 8, metadata !237, metadata !DIExpression()), !dbg !215
  call void @llvm.dbg.value(metadata i32 4, metadata !238, metadata !DIExpression()), !dbg !215
  call void @llvm.dbg.value(metadata i32 8, metadata !239, metadata !DIExpression()), !dbg !215
  call void @llvm.dbg.value(metadata i32 8, metadata !240, metadata !DIExpression()), !dbg !215
  call void @llvm.dbg.value(metadata i32 1, metadata !241, metadata !DIExpression()), !dbg !215
  call void @llvm.dbg.value(metadata i32 1, metadata !242, metadata !DIExpression()), !dbg !215
  call void @llvm.dbg.value(metadata i32 4, metadata !243, metadata !DIExpression()), !dbg !215
  call void @llvm.dbg.value(metadata i32 4, metadata !244, metadata !DIExpression()), !dbg !215
  call void @llvm.dbg.value(metadata i32 4, metadata !245, metadata !DIExpression()), !dbg !215
  call void @llvm.dbg.value(metadata i32 8, metadata !246, metadata !DIExpression()), !dbg !215
  call void @llvm.dbg.value(metadata i32 4, metadata !247, metadata !DIExpression()), !dbg !215
  call void @llvm.dbg.value(metadata i32 4, metadata !248, metadata !DIExpression()), !dbg !215
  call void @llvm.dbg.value(metadata i32 4, metadata !249, metadata !DIExpression()), !dbg !215
  call void @llvm.dbg.value(metadata i32 4, metadata !250, metadata !DIExpression()), !dbg !215
  call void @llvm.dbg.value(metadata i32 1, metadata !251, metadata !DIExpression()), !dbg !215
  call void @llvm.dbg.value(metadata i32 4, metadata !252, metadata !DIExpression()), !dbg !215
  call void @llvm.dbg.value(metadata i32 0, metadata !253, metadata !DIExpression()), !dbg !215
  call void @llvm.dbg.value(metadata i32 1, metadata !254, metadata !DIExpression()), !dbg !215
  call void @llvm.dbg.value(metadata i32 4, metadata !255, metadata !DIExpression()), !dbg !215
  call void @llvm.dbg.value(metadata i32 0, metadata !256, metadata !DIExpression()), !dbg !215
  call void @llvm.dbg.value(metadata i32 1, metadata !257, metadata !DIExpression()), !dbg !215
  call void @llvm.dbg.value(metadata i32 4, metadata !258, metadata !DIExpression()), !dbg !215
  call void @llvm.dbg.value(metadata i32 8, metadata !259, metadata !DIExpression()), !dbg !215
  call void @llvm.dbg.value(metadata i32 8, metadata !260, metadata !DIExpression()), !dbg !215
  call void @llvm.dbg.value(metadata i32 8, metadata !261, metadata !DIExpression()), !dbg !215
  %0 = load i32, i32* %__nv_MAIN__F1L46_2Arg0, align 4, !dbg !262
  store i32 %0, i32* %__gtid___nv_MAIN__F1L46_2__673, align 4, !dbg !262
  br label %L.LB3_657

L.LB3_657:                                        ; preds = %L.entry
  br label %L.LB3_409

L.LB3_409:                                        ; preds = %L.LB3_657
  call void @llvm.dbg.declare(metadata double* %sum2_410, metadata !263, metadata !DIExpression()), !dbg !262
  store double 0.000000e+00, double* %sum2_410, align 8, !dbg !264
  store i32 0, i32* %.i0002p_412, align 4, !dbg !265
  call void @llvm.dbg.declare(metadata i64* %i_411, metadata !266, metadata !DIExpression()), !dbg !262
  store i64 1, i64* %i_411, align 8, !dbg !265
  %1 = bitcast i64* %__nv_MAIN__F1L46_2Arg2 to i8*, !dbg !265
  %2 = getelementptr i8, i8* %1, i64 8, !dbg !265
  %3 = bitcast i8* %2 to i64**, !dbg !265
  %4 = load i64*, i64** %3, align 8, !dbg !265
  %5 = load i64, i64* %4, align 8, !dbg !265
  store i64 %5, i64* %.du0004p_470, align 8, !dbg !265
  %6 = bitcast i64* %__nv_MAIN__F1L46_2Arg2 to i8*, !dbg !265
  %7 = getelementptr i8, i8* %6, i64 8, !dbg !265
  %8 = bitcast i8* %7 to i64**, !dbg !265
  %9 = load i64*, i64** %8, align 8, !dbg !265
  %10 = load i64, i64* %9, align 8, !dbg !265
  store i64 %10, i64* %.de0004p_471, align 8, !dbg !265
  store i64 1, i64* %.di0004p_472, align 8, !dbg !265
  %11 = load i64, i64* %.di0004p_472, align 8, !dbg !265
  store i64 %11, i64* %.ds0004p_473, align 8, !dbg !265
  store i64 1, i64* %.dl0004p_475, align 8, !dbg !265
  %12 = load i64, i64* %.dl0004p_475, align 8, !dbg !265
  store i64 %12, i64* %.dl0004p.copy_667, align 8, !dbg !265
  %13 = load i64, i64* %.de0004p_471, align 8, !dbg !265
  store i64 %13, i64* %.de0004p.copy_668, align 8, !dbg !265
  %14 = load i64, i64* %.ds0004p_473, align 8, !dbg !265
  store i64 %14, i64* %.ds0004p.copy_669, align 8, !dbg !265
  %15 = load i32, i32* %__gtid___nv_MAIN__F1L46_2__673, align 4, !dbg !265
  %16 = bitcast i32* %.i0002p_412 to i64*, !dbg !265
  %17 = load i64, i64* %.ds0004p.copy_669, align 8, !dbg !265
  call void @__kmpc_for_static_init_8(i64* null, i32 %15, i32 34, i64* %16, i64* %.dl0004p.copy_667, i64* %.de0004p.copy_668, i64* %.ds0004p.copy_669, i64 %17, i64 1), !dbg !265
  %18 = load i64, i64* %.dl0004p.copy_667, align 8, !dbg !265
  store i64 %18, i64* %.dl0004p_475, align 8, !dbg !265
  %19 = load i64, i64* %.de0004p.copy_668, align 8, !dbg !265
  store i64 %19, i64* %.de0004p_471, align 8, !dbg !265
  %20 = load i64, i64* %.ds0004p.copy_669, align 8, !dbg !265
  store i64 %20, i64* %.ds0004p_473, align 8, !dbg !265
  %21 = load i64, i64* %.dl0004p_475, align 8, !dbg !265
  store i64 %21, i64* %i_411, align 8, !dbg !265
  %22 = load i64, i64* %i_411, align 8, !dbg !265
  call void @llvm.dbg.value(metadata i64 %22, metadata !266, metadata !DIExpression()), !dbg !262
  store i64 %22, i64* %.dX0004p_474, align 8, !dbg !265
  %23 = load i64, i64* %.dX0004p_474, align 8, !dbg !265
  %24 = load i64, i64* %.du0004p_470, align 8, !dbg !265
  %25 = icmp sgt i64 %23, %24, !dbg !265
  br i1 %25, label %L.LB3_468, label %L.LB3_701, !dbg !265

L.LB3_701:                                        ; preds = %L.LB3_409
  %26 = load i64, i64* %.dX0004p_474, align 8, !dbg !265
  store i64 %26, i64* %i_411, align 8, !dbg !265
  %27 = load i64, i64* %.di0004p_472, align 8, !dbg !265
  %28 = load i64, i64* %.de0004p_471, align 8, !dbg !265
  %29 = load i64, i64* %.dX0004p_474, align 8, !dbg !265
  %30 = sub nsw i64 %28, %29, !dbg !265
  %31 = add nsw i64 %27, %30, !dbg !265
  %32 = load i64, i64* %.di0004p_472, align 8, !dbg !265
  %33 = sdiv i64 %31, %32, !dbg !265
  store i64 %33, i64* %.dY0004p_469, align 8, !dbg !265
  %34 = load i64, i64* %.dY0004p_469, align 8, !dbg !265
  %35 = icmp sle i64 %34, 0, !dbg !265
  br i1 %35, label %L.LB3_478, label %L.LB3_477, !dbg !265

L.LB3_477:                                        ; preds = %L.LB3_477, %L.LB3_701
  %36 = load i64, i64* %i_411, align 8, !dbg !267
  call void @llvm.dbg.value(metadata i64 %36, metadata !266, metadata !DIExpression()), !dbg !262
  %37 = bitcast i64* %__nv_MAIN__F1L46_2Arg2 to i8*, !dbg !267
  %38 = getelementptr i8, i8* %37, i64 136, !dbg !267
  %39 = bitcast i8* %38 to i8**, !dbg !267
  %40 = load i8*, i8** %39, align 8, !dbg !267
  %41 = getelementptr i8, i8* %40, i64 56, !dbg !267
  %42 = bitcast i8* %41 to i64*, !dbg !267
  %43 = load i64, i64* %42, align 8, !dbg !267
  %44 = add nsw i64 %36, %43, !dbg !267
  %45 = bitcast i64* %__nv_MAIN__F1L46_2Arg2 to i8*, !dbg !267
  %46 = getelementptr i8, i8* %45, i64 80, !dbg !267
  %47 = bitcast i8* %46 to i8***, !dbg !267
  %48 = load i8**, i8*** %47, align 8, !dbg !267
  %49 = load i8*, i8** %48, align 8, !dbg !267
  %50 = getelementptr i8, i8* %49, i64 -8, !dbg !267
  %51 = bitcast i8* %50 to double*, !dbg !267
  %52 = getelementptr double, double* %51, i64 %44, !dbg !267
  %53 = load double, double* %52, align 8, !dbg !267
  %54 = load i64, i64* %i_411, align 8, !dbg !267
  call void @llvm.dbg.value(metadata i64 %54, metadata !266, metadata !DIExpression()), !dbg !262
  %55 = bitcast i64* %__nv_MAIN__F1L46_2Arg2 to i8*, !dbg !267
  %56 = getelementptr i8, i8* %55, i64 128, !dbg !267
  %57 = bitcast i8* %56 to i8**, !dbg !267
  %58 = load i8*, i8** %57, align 8, !dbg !267
  %59 = getelementptr i8, i8* %58, i64 56, !dbg !267
  %60 = bitcast i8* %59 to i64*, !dbg !267
  %61 = load i64, i64* %60, align 8, !dbg !267
  %62 = add nsw i64 %54, %61, !dbg !267
  %63 = bitcast i64* %__nv_MAIN__F1L46_2Arg2 to i8*, !dbg !267
  %64 = getelementptr i8, i8* %63, i64 24, !dbg !267
  %65 = bitcast i8* %64 to i8***, !dbg !267
  %66 = load i8**, i8*** %65, align 8, !dbg !267
  %67 = load i8*, i8** %66, align 8, !dbg !267
  %68 = getelementptr i8, i8* %67, i64 -8, !dbg !267
  %69 = bitcast i8* %68 to double*, !dbg !267
  %70 = getelementptr double, double* %69, i64 %62, !dbg !267
  %71 = load double, double* %70, align 8, !dbg !267
  %72 = fmul fast double %53, %71, !dbg !267
  %73 = load double, double* %sum2_410, align 8, !dbg !267
  call void @llvm.dbg.value(metadata double %73, metadata !263, metadata !DIExpression()), !dbg !262
  %74 = fadd fast double %72, %73, !dbg !267
  store double %74, double* %sum2_410, align 8, !dbg !267
  %75 = load i64, i64* %.di0004p_472, align 8, !dbg !262
  %76 = load i64, i64* %i_411, align 8, !dbg !262
  call void @llvm.dbg.value(metadata i64 %76, metadata !266, metadata !DIExpression()), !dbg !262
  %77 = add nsw i64 %75, %76, !dbg !262
  store i64 %77, i64* %i_411, align 8, !dbg !262
  %78 = load i64, i64* %.dY0004p_469, align 8, !dbg !262
  %79 = sub nsw i64 %78, 1, !dbg !262
  store i64 %79, i64* %.dY0004p_469, align 8, !dbg !262
  %80 = load i64, i64* %.dY0004p_469, align 8, !dbg !262
  %81 = icmp sgt i64 %80, 0, !dbg !262
  br i1 %81, label %L.LB3_477, label %L.LB3_478, !dbg !262

L.LB3_478:                                        ; preds = %L.LB3_477, %L.LB3_701
  br label %L.LB3_468

L.LB3_468:                                        ; preds = %L.LB3_478, %L.LB3_409
  %82 = load i32, i32* %__gtid___nv_MAIN__F1L46_2__673, align 4, !dbg !262
  call void @__kmpc_for_static_fini(i64* null, i32 %82), !dbg !262
  %83 = call i32 (...) @_mp_bcs_nest_red(), !dbg !262
  %84 = call i32 (...) @_mp_bcs_nest_red(), !dbg !262
  %85 = load double, double* %sum2_410, align 8, !dbg !262
  call void @llvm.dbg.value(metadata double %85, metadata !263, metadata !DIExpression()), !dbg !262
  %86 = bitcast i64* %__nv_MAIN__F1L46_2Arg2 to double**, !dbg !262
  %87 = load double*, double** %86, align 8, !dbg !262
  %88 = load double, double* %87, align 8, !dbg !262
  %89 = fadd fast double %85, %88, !dbg !262
  %90 = bitcast i64* %__nv_MAIN__F1L46_2Arg2 to double**, !dbg !262
  %91 = load double*, double** %90, align 8, !dbg !262
  store double %89, double* %91, align 8, !dbg !262
  %92 = call i32 (...) @_mp_ecs_nest_red(), !dbg !262
  %93 = call i32 (...) @_mp_ecs_nest_red(), !dbg !262
  br label %L.LB3_413

L.LB3_413:                                        ; preds = %L.LB3_468
  ret void, !dbg !262
}

define internal void @__nv_MAIN_F1L33_3_(i32* %__nv_MAIN_F1L33_3Arg0, i64* %__nv_MAIN_F1L33_3Arg1, i64* %__nv_MAIN_F1L33_3Arg2) #0 !dbg !268 {
L.entry:
  %__gtid___nv_MAIN_F1L33_3__721 = alloca i32, align 4
  %sum_392 = alloca double, align 8
  %.i0000p_396 = alloca i32, align 4
  %i2_394 = alloca i64, align 8
  %.du0002_446 = alloca i64, align 8
  %.de0002_447 = alloca i64, align 8
  %.di0002_448 = alloca i64, align 8
  %.ds0002_449 = alloca i64, align 8
  %.dl0002_451 = alloca i64, align 8
  %.dl0002.copy_715 = alloca i64, align 8
  %.de0002.copy_716 = alloca i64, align 8
  %.ds0002.copy_717 = alloca i64, align 8
  %.dX0002_450 = alloca i64, align 8
  %.dY0002_445 = alloca i64, align 8
  %.uplevelArgPack0004_730 = alloca %astruct.dt273, align 16
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L33_3Arg0, metadata !269, metadata !DIExpression()), !dbg !270
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L33_3Arg1, metadata !271, metadata !DIExpression()), !dbg !270
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L33_3Arg2, metadata !272, metadata !DIExpression()), !dbg !270
  call void @llvm.dbg.value(metadata i32 4, metadata !273, metadata !DIExpression()), !dbg !270
  call void @llvm.dbg.value(metadata i32 8, metadata !274, metadata !DIExpression()), !dbg !270
  call void @llvm.dbg.value(metadata i32 8, metadata !275, metadata !DIExpression()), !dbg !270
  call void @llvm.dbg.value(metadata i32 8, metadata !276, metadata !DIExpression()), !dbg !270
  call void @llvm.dbg.value(metadata i32 8, metadata !277, metadata !DIExpression()), !dbg !270
  call void @llvm.dbg.value(metadata i32 1, metadata !278, metadata !DIExpression()), !dbg !270
  call void @llvm.dbg.value(metadata i32 1, metadata !279, metadata !DIExpression()), !dbg !270
  call void @llvm.dbg.value(metadata i32 4, metadata !280, metadata !DIExpression()), !dbg !270
  call void @llvm.dbg.value(metadata i32 8, metadata !281, metadata !DIExpression()), !dbg !270
  call void @llvm.dbg.value(metadata i32 1, metadata !282, metadata !DIExpression()), !dbg !270
  call void @llvm.dbg.value(metadata i32 4, metadata !283, metadata !DIExpression()), !dbg !270
  call void @llvm.dbg.value(metadata i32 8, metadata !284, metadata !DIExpression()), !dbg !270
  call void @llvm.dbg.value(metadata i32 1, metadata !285, metadata !DIExpression()), !dbg !270
  call void @llvm.dbg.value(metadata i32 8, metadata !286, metadata !DIExpression()), !dbg !270
  call void @llvm.dbg.value(metadata i32 8, metadata !287, metadata !DIExpression()), !dbg !270
  call void @llvm.dbg.value(metadata i32 8, metadata !288, metadata !DIExpression()), !dbg !270
  call void @llvm.dbg.value(metadata i32 8, metadata !289, metadata !DIExpression()), !dbg !270
  call void @llvm.dbg.value(metadata i32 4, metadata !290, metadata !DIExpression()), !dbg !270
  call void @llvm.dbg.value(metadata i32 8, metadata !291, metadata !DIExpression()), !dbg !270
  call void @llvm.dbg.value(metadata i32 8, metadata !292, metadata !DIExpression()), !dbg !270
  call void @llvm.dbg.value(metadata i32 4, metadata !293, metadata !DIExpression()), !dbg !270
  call void @llvm.dbg.value(metadata i32 8, metadata !294, metadata !DIExpression()), !dbg !270
  call void @llvm.dbg.value(metadata i32 8, metadata !295, metadata !DIExpression()), !dbg !270
  call void @llvm.dbg.value(metadata i32 1, metadata !296, metadata !DIExpression()), !dbg !270
  call void @llvm.dbg.value(metadata i32 1, metadata !297, metadata !DIExpression()), !dbg !270
  call void @llvm.dbg.value(metadata i32 4, metadata !298, metadata !DIExpression()), !dbg !270
  call void @llvm.dbg.value(metadata i32 4, metadata !299, metadata !DIExpression()), !dbg !270
  call void @llvm.dbg.value(metadata i32 4, metadata !300, metadata !DIExpression()), !dbg !270
  call void @llvm.dbg.value(metadata i32 8, metadata !301, metadata !DIExpression()), !dbg !270
  call void @llvm.dbg.value(metadata i32 4, metadata !302, metadata !DIExpression()), !dbg !270
  call void @llvm.dbg.value(metadata i32 4, metadata !303, metadata !DIExpression()), !dbg !270
  call void @llvm.dbg.value(metadata i32 4, metadata !304, metadata !DIExpression()), !dbg !270
  call void @llvm.dbg.value(metadata i32 4, metadata !305, metadata !DIExpression()), !dbg !270
  call void @llvm.dbg.value(metadata i32 1, metadata !306, metadata !DIExpression()), !dbg !270
  call void @llvm.dbg.value(metadata i32 4, metadata !307, metadata !DIExpression()), !dbg !270
  call void @llvm.dbg.value(metadata i32 0, metadata !308, metadata !DIExpression()), !dbg !270
  call void @llvm.dbg.value(metadata i32 1, metadata !309, metadata !DIExpression()), !dbg !270
  call void @llvm.dbg.value(metadata i32 4, metadata !310, metadata !DIExpression()), !dbg !270
  call void @llvm.dbg.value(metadata i32 0, metadata !311, metadata !DIExpression()), !dbg !270
  call void @llvm.dbg.value(metadata i32 1, metadata !312, metadata !DIExpression()), !dbg !270
  call void @llvm.dbg.value(metadata i32 4, metadata !313, metadata !DIExpression()), !dbg !270
  call void @llvm.dbg.value(metadata i32 8, metadata !314, metadata !DIExpression()), !dbg !270
  call void @llvm.dbg.value(metadata i32 8, metadata !315, metadata !DIExpression()), !dbg !270
  call void @llvm.dbg.value(metadata i32 8, metadata !316, metadata !DIExpression()), !dbg !270
  %0 = load i32, i32* %__nv_MAIN_F1L33_3Arg0, align 4, !dbg !317
  store i32 %0, i32* %__gtid___nv_MAIN_F1L33_3__721, align 4, !dbg !317
  br label %L.LB5_705

L.LB5_705:                                        ; preds = %L.entry
  br label %L.LB5_391

L.LB5_391:                                        ; preds = %L.LB5_705
  call void @llvm.dbg.declare(metadata double* %sum_392, metadata !318, metadata !DIExpression()), !dbg !317
  store double 0.000000e+00, double* %sum_392, align 8, !dbg !319
  br label %L.LB5_393

L.LB5_393:                                        ; preds = %L.LB5_391
  store i32 0, i32* %.i0000p_396, align 4, !dbg !320
  call void @llvm.dbg.declare(metadata i64* %i2_394, metadata !321, metadata !DIExpression()), !dbg !317
  store i64 1, i64* %i2_394, align 8, !dbg !320
  %1 = bitcast i64* %__nv_MAIN_F1L33_3Arg2 to i8*, !dbg !320
  %2 = getelementptr i8, i8* %1, i64 8, !dbg !320
  %3 = bitcast i8* %2 to i64**, !dbg !320
  %4 = load i64*, i64** %3, align 8, !dbg !320
  %5 = load i64, i64* %4, align 8, !dbg !320
  store i64 %5, i64* %.du0002_446, align 8, !dbg !320
  %6 = bitcast i64* %__nv_MAIN_F1L33_3Arg2 to i8*, !dbg !320
  %7 = getelementptr i8, i8* %6, i64 8, !dbg !320
  %8 = bitcast i8* %7 to i64**, !dbg !320
  %9 = load i64*, i64** %8, align 8, !dbg !320
  %10 = load i64, i64* %9, align 8, !dbg !320
  store i64 %10, i64* %.de0002_447, align 8, !dbg !320
  store i64 256, i64* %.di0002_448, align 8, !dbg !320
  %11 = load i64, i64* %.di0002_448, align 8, !dbg !320
  store i64 %11, i64* %.ds0002_449, align 8, !dbg !320
  store i64 1, i64* %.dl0002_451, align 8, !dbg !320
  %12 = load i64, i64* %.dl0002_451, align 8, !dbg !320
  store i64 %12, i64* %.dl0002.copy_715, align 8, !dbg !320
  %13 = load i64, i64* %.de0002_447, align 8, !dbg !320
  store i64 %13, i64* %.de0002.copy_716, align 8, !dbg !320
  %14 = load i64, i64* %.ds0002_449, align 8, !dbg !320
  store i64 %14, i64* %.ds0002.copy_717, align 8, !dbg !320
  %15 = load i32, i32* %__gtid___nv_MAIN_F1L33_3__721, align 4, !dbg !320
  %16 = bitcast i32* %.i0000p_396 to i64*, !dbg !320
  %17 = load i64, i64* %.ds0002.copy_717, align 8, !dbg !320
  call void @__kmpc_for_static_init_8(i64* null, i32 %15, i32 92, i64* %16, i64* %.dl0002.copy_715, i64* %.de0002.copy_716, i64* %.ds0002.copy_717, i64 %17, i64 1), !dbg !320
  %18 = load i64, i64* %.dl0002.copy_715, align 8, !dbg !320
  store i64 %18, i64* %.dl0002_451, align 8, !dbg !320
  %19 = load i64, i64* %.de0002.copy_716, align 8, !dbg !320
  store i64 %19, i64* %.de0002_447, align 8, !dbg !320
  %20 = load i64, i64* %.ds0002.copy_717, align 8, !dbg !320
  store i64 %20, i64* %.ds0002_449, align 8, !dbg !320
  %21 = load i64, i64* %.dl0002_451, align 8, !dbg !320
  store i64 %21, i64* %i2_394, align 8, !dbg !320
  %22 = load i64, i64* %i2_394, align 8, !dbg !320
  call void @llvm.dbg.value(metadata i64 %22, metadata !321, metadata !DIExpression()), !dbg !317
  store i64 %22, i64* %.dX0002_450, align 8, !dbg !320
  %23 = load i64, i64* %.dX0002_450, align 8, !dbg !320
  %24 = load i64, i64* %.du0002_446, align 8, !dbg !320
  %25 = icmp sgt i64 %23, %24, !dbg !320
  br i1 %25, label %L.LB5_444, label %L.LB5_744, !dbg !320

L.LB5_744:                                        ; preds = %L.LB5_393
  %26 = load i64, i64* %.du0002_446, align 8, !dbg !320
  %27 = load i64, i64* %.de0002_447, align 8, !dbg !320
  %28 = icmp slt i64 %26, %27, !dbg !320
  %29 = select i1 %28, i64 %26, i64 %27, !dbg !320
  store i64 %29, i64* %.de0002_447, align 8, !dbg !320
  %30 = load i64, i64* %.dX0002_450, align 8, !dbg !320
  store i64 %30, i64* %i2_394, align 8, !dbg !320
  %31 = load i64, i64* %.di0002_448, align 8, !dbg !320
  %32 = load i64, i64* %.de0002_447, align 8, !dbg !320
  %33 = load i64, i64* %.dX0002_450, align 8, !dbg !320
  %34 = sub nsw i64 %32, %33, !dbg !320
  %35 = add nsw i64 %31, %34, !dbg !320
  %36 = load i64, i64* %.di0002_448, align 8, !dbg !320
  %37 = sdiv i64 %35, %36, !dbg !320
  store i64 %37, i64* %.dY0002_445, align 8, !dbg !320
  %38 = load i64, i64* %.dY0002_445, align 8, !dbg !320
  %39 = icmp sle i64 %38, 0, !dbg !320
  br i1 %39, label %L.LB5_454, label %L.LB5_453, !dbg !320

L.LB5_453:                                        ; preds = %L.LB5_736, %L.LB5_744
  %40 = bitcast i64* %__nv_MAIN_F1L33_3Arg2 to i8*, !dbg !322
  %41 = bitcast %astruct.dt273* %.uplevelArgPack0004_730 to i8*, !dbg !322
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %41, i8* align 8 %40, i64 144, i1 false), !dbg !322
  %42 = bitcast double* %sum_392 to i8*, !dbg !322
  %43 = bitcast %astruct.dt273* %.uplevelArgPack0004_730 to i8*, !dbg !322
  %44 = getelementptr i8, i8* %43, i64 144, !dbg !322
  %45 = bitcast i8* %44 to i8**, !dbg !322
  store i8* %42, i8** %45, align 8, !dbg !322
  %46 = bitcast i64* %i2_394 to i8*, !dbg !322
  %47 = bitcast %astruct.dt273* %.uplevelArgPack0004_730 to i8*, !dbg !322
  %48 = getelementptr i8, i8* %47, i64 152, !dbg !322
  %49 = bitcast i8* %48 to i8**, !dbg !322
  store i8* %46, i8** %49, align 8, !dbg !322
  br label %L.LB5_736, !dbg !322

L.LB5_736:                                        ; preds = %L.LB5_453
  %50 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN_F1L36_4_ to i64*, !dbg !322
  %51 = bitcast %astruct.dt273* %.uplevelArgPack0004_730 to i64*, !dbg !322
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %50, i64* %51), !dbg !322
  %52 = load i64, i64* %.di0002_448, align 8, !dbg !323
  %53 = load i64, i64* %i2_394, align 8, !dbg !323
  call void @llvm.dbg.value(metadata i64 %53, metadata !321, metadata !DIExpression()), !dbg !317
  %54 = add nsw i64 %52, %53, !dbg !323
  store i64 %54, i64* %i2_394, align 8, !dbg !323
  %55 = load i64, i64* %.dY0002_445, align 8, !dbg !323
  %56 = sub nsw i64 %55, 1, !dbg !323
  store i64 %56, i64* %.dY0002_445, align 8, !dbg !323
  %57 = load i64, i64* %.dY0002_445, align 8, !dbg !323
  %58 = icmp sgt i64 %57, 0, !dbg !323
  br i1 %58, label %L.LB5_453, label %L.LB5_454, !dbg !323

L.LB5_454:                                        ; preds = %L.LB5_736, %L.LB5_744
  br label %L.LB5_444

L.LB5_444:                                        ; preds = %L.LB5_454, %L.LB5_393
  %59 = load i32, i32* %__gtid___nv_MAIN_F1L33_3__721, align 4, !dbg !323
  call void @__kmpc_for_static_fini(i64* null, i32 %59), !dbg !323
  br label %L.LB5_404

L.LB5_404:                                        ; preds = %L.LB5_444
  %60 = call i32 (...) @_mp_bcs_nest_red(), !dbg !317
  %61 = call i32 (...) @_mp_bcs_nest_red(), !dbg !317
  %62 = load double, double* %sum_392, align 8, !dbg !317
  call void @llvm.dbg.value(metadata double %62, metadata !318, metadata !DIExpression()), !dbg !317
  %63 = bitcast i64* %__nv_MAIN_F1L33_3Arg2 to double**, !dbg !317
  %64 = load double*, double** %63, align 8, !dbg !317
  %65 = load double, double* %64, align 8, !dbg !317
  %66 = fadd fast double %62, %65, !dbg !317
  %67 = bitcast i64* %__nv_MAIN_F1L33_3Arg2 to double**, !dbg !317
  %68 = load double*, double** %67, align 8, !dbg !317
  store double %66, double* %68, align 8, !dbg !317
  %69 = call i32 (...) @_mp_ecs_nest_red(), !dbg !317
  %70 = call i32 (...) @_mp_ecs_nest_red(), !dbg !317
  br label %L.LB5_405

L.LB5_405:                                        ; preds = %L.LB5_404
  ret void, !dbg !317
}

define internal void @__nv_MAIN_F1L36_4_(i32* %__nv_MAIN_F1L36_4Arg0, i64* %__nv_MAIN_F1L36_4Arg1, i64* %__nv_MAIN_F1L36_4Arg2) #0 !dbg !324 {
L.entry:
  %__gtid___nv_MAIN_F1L36_4__765 = alloca i32, align 4
  %sum_400 = alloca double, align 8
  %.i0001p_402 = alloca i32, align 4
  %i_401 = alloca i64, align 8
  %.du0003p_458 = alloca i64, align 8
  %.de0003p_459 = alloca i64, align 8
  %.di0003p_460 = alloca i64, align 8
  %.ds0003p_461 = alloca i64, align 8
  %.dl0003p_463 = alloca i64, align 8
  %.dl0003p.copy_759 = alloca i64, align 8
  %.de0003p.copy_760 = alloca i64, align 8
  %.ds0003p.copy_761 = alloca i64, align 8
  %.dX0003p_462 = alloca i64, align 8
  %.dY0003p_457 = alloca i64, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L36_4Arg0, metadata !325, metadata !DIExpression()), !dbg !326
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L36_4Arg1, metadata !327, metadata !DIExpression()), !dbg !326
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L36_4Arg2, metadata !328, metadata !DIExpression()), !dbg !326
  call void @llvm.dbg.value(metadata i32 4, metadata !329, metadata !DIExpression()), !dbg !326
  call void @llvm.dbg.value(metadata i32 8, metadata !330, metadata !DIExpression()), !dbg !326
  call void @llvm.dbg.value(metadata i32 8, metadata !331, metadata !DIExpression()), !dbg !326
  call void @llvm.dbg.value(metadata i32 8, metadata !332, metadata !DIExpression()), !dbg !326
  call void @llvm.dbg.value(metadata i32 8, metadata !333, metadata !DIExpression()), !dbg !326
  call void @llvm.dbg.value(metadata i32 1, metadata !334, metadata !DIExpression()), !dbg !326
  call void @llvm.dbg.value(metadata i32 1, metadata !335, metadata !DIExpression()), !dbg !326
  call void @llvm.dbg.value(metadata i32 4, metadata !336, metadata !DIExpression()), !dbg !326
  call void @llvm.dbg.value(metadata i32 8, metadata !337, metadata !DIExpression()), !dbg !326
  call void @llvm.dbg.value(metadata i32 1, metadata !338, metadata !DIExpression()), !dbg !326
  call void @llvm.dbg.value(metadata i32 4, metadata !339, metadata !DIExpression()), !dbg !326
  call void @llvm.dbg.value(metadata i32 8, metadata !340, metadata !DIExpression()), !dbg !326
  call void @llvm.dbg.value(metadata i32 1, metadata !341, metadata !DIExpression()), !dbg !326
  call void @llvm.dbg.value(metadata i32 8, metadata !342, metadata !DIExpression()), !dbg !326
  call void @llvm.dbg.value(metadata i32 8, metadata !343, metadata !DIExpression()), !dbg !326
  call void @llvm.dbg.value(metadata i32 8, metadata !344, metadata !DIExpression()), !dbg !326
  call void @llvm.dbg.value(metadata i32 8, metadata !345, metadata !DIExpression()), !dbg !326
  call void @llvm.dbg.value(metadata i32 4, metadata !346, metadata !DIExpression()), !dbg !326
  call void @llvm.dbg.value(metadata i32 8, metadata !347, metadata !DIExpression()), !dbg !326
  call void @llvm.dbg.value(metadata i32 8, metadata !348, metadata !DIExpression()), !dbg !326
  call void @llvm.dbg.value(metadata i32 4, metadata !349, metadata !DIExpression()), !dbg !326
  call void @llvm.dbg.value(metadata i32 8, metadata !350, metadata !DIExpression()), !dbg !326
  call void @llvm.dbg.value(metadata i32 8, metadata !351, metadata !DIExpression()), !dbg !326
  call void @llvm.dbg.value(metadata i32 1, metadata !352, metadata !DIExpression()), !dbg !326
  call void @llvm.dbg.value(metadata i32 1, metadata !353, metadata !DIExpression()), !dbg !326
  call void @llvm.dbg.value(metadata i32 4, metadata !354, metadata !DIExpression()), !dbg !326
  call void @llvm.dbg.value(metadata i32 4, metadata !355, metadata !DIExpression()), !dbg !326
  call void @llvm.dbg.value(metadata i32 4, metadata !356, metadata !DIExpression()), !dbg !326
  call void @llvm.dbg.value(metadata i32 8, metadata !357, metadata !DIExpression()), !dbg !326
  call void @llvm.dbg.value(metadata i32 4, metadata !358, metadata !DIExpression()), !dbg !326
  call void @llvm.dbg.value(metadata i32 4, metadata !359, metadata !DIExpression()), !dbg !326
  call void @llvm.dbg.value(metadata i32 4, metadata !360, metadata !DIExpression()), !dbg !326
  call void @llvm.dbg.value(metadata i32 4, metadata !361, metadata !DIExpression()), !dbg !326
  call void @llvm.dbg.value(metadata i32 1, metadata !362, metadata !DIExpression()), !dbg !326
  call void @llvm.dbg.value(metadata i32 4, metadata !363, metadata !DIExpression()), !dbg !326
  call void @llvm.dbg.value(metadata i32 0, metadata !364, metadata !DIExpression()), !dbg !326
  call void @llvm.dbg.value(metadata i32 1, metadata !365, metadata !DIExpression()), !dbg !326
  call void @llvm.dbg.value(metadata i32 4, metadata !366, metadata !DIExpression()), !dbg !326
  call void @llvm.dbg.value(metadata i32 0, metadata !367, metadata !DIExpression()), !dbg !326
  call void @llvm.dbg.value(metadata i32 1, metadata !368, metadata !DIExpression()), !dbg !326
  call void @llvm.dbg.value(metadata i32 4, metadata !369, metadata !DIExpression()), !dbg !326
  call void @llvm.dbg.value(metadata i32 8, metadata !370, metadata !DIExpression()), !dbg !326
  call void @llvm.dbg.value(metadata i32 8, metadata !371, metadata !DIExpression()), !dbg !326
  call void @llvm.dbg.value(metadata i32 8, metadata !372, metadata !DIExpression()), !dbg !326
  %0 = load i32, i32* %__nv_MAIN_F1L36_4Arg0, align 4, !dbg !373
  store i32 %0, i32* %__gtid___nv_MAIN_F1L36_4__765, align 4, !dbg !373
  br label %L.LB7_748

L.LB7_748:                                        ; preds = %L.entry
  br label %L.LB7_399

L.LB7_399:                                        ; preds = %L.LB7_748
  call void @llvm.dbg.declare(metadata double* %sum_400, metadata !374, metadata !DIExpression()), !dbg !373
  store double 0.000000e+00, double* %sum_400, align 8, !dbg !375
  store i32 0, i32* %.i0001p_402, align 4, !dbg !376
  %1 = bitcast i64* %__nv_MAIN_F1L36_4Arg2 to i8*, !dbg !376
  %2 = getelementptr i8, i8* %1, i64 152, !dbg !376
  %3 = bitcast i8* %2 to i64**, !dbg !376
  %4 = load i64*, i64** %3, align 8, !dbg !376
  %5 = load i64, i64* %4, align 8, !dbg !376
  %6 = add nsw i64 %5, 1, !dbg !376
  call void @llvm.dbg.declare(metadata i64* %i_401, metadata !377, metadata !DIExpression()), !dbg !373
  store i64 %6, i64* %i_401, align 8, !dbg !376
  %7 = bitcast i64* %__nv_MAIN_F1L36_4Arg2 to i8*, !dbg !376
  %8 = getelementptr i8, i8* %7, i64 152, !dbg !376
  %9 = bitcast i8* %8 to i64**, !dbg !376
  %10 = load i64*, i64** %9, align 8, !dbg !376
  %11 = load i64, i64* %10, align 8, !dbg !376
  %12 = add nsw i64 %11, 256, !dbg !376
  %13 = bitcast i64* %__nv_MAIN_F1L36_4Arg2 to i8*, !dbg !376
  %14 = getelementptr i8, i8* %13, i64 8, !dbg !376
  %15 = bitcast i8* %14 to i64**, !dbg !376
  %16 = load i64*, i64** %15, align 8, !dbg !376
  %17 = load i64, i64* %16, align 8, !dbg !376
  %18 = icmp slt i64 %12, %17, !dbg !376
  %19 = sext i1 %18 to i32, !dbg !376
  %20 = trunc i32 %19 to i1, !dbg !376
  %21 = bitcast i64* %__nv_MAIN_F1L36_4Arg2 to i8*, !dbg !376
  %22 = getelementptr i8, i8* %21, i64 152, !dbg !376
  %23 = bitcast i8* %22 to i64**, !dbg !376
  %24 = load i64*, i64** %23, align 8, !dbg !376
  %25 = load i64, i64* %24, align 8, !dbg !376
  %26 = add nsw i64 %25, 256, !dbg !376
  %27 = bitcast i64* %__nv_MAIN_F1L36_4Arg2 to i8*, !dbg !376
  %28 = getelementptr i8, i8* %27, i64 8, !dbg !376
  %29 = bitcast i8* %28 to i64**, !dbg !376
  %30 = load i64*, i64** %29, align 8, !dbg !376
  %31 = load i64, i64* %30, align 8, !dbg !376
  %32 = select i1 %20, i64 %26, i64 %31, !dbg !376
  store i64 %32, i64* %.du0003p_458, align 8, !dbg !376
  %33 = bitcast i64* %__nv_MAIN_F1L36_4Arg2 to i8*, !dbg !376
  %34 = getelementptr i8, i8* %33, i64 152, !dbg !376
  %35 = bitcast i8* %34 to i64**, !dbg !376
  %36 = load i64*, i64** %35, align 8, !dbg !376
  %37 = load i64, i64* %36, align 8, !dbg !376
  %38 = add nsw i64 %37, 256, !dbg !376
  %39 = bitcast i64* %__nv_MAIN_F1L36_4Arg2 to i8*, !dbg !376
  %40 = getelementptr i8, i8* %39, i64 8, !dbg !376
  %41 = bitcast i8* %40 to i64**, !dbg !376
  %42 = load i64*, i64** %41, align 8, !dbg !376
  %43 = load i64, i64* %42, align 8, !dbg !376
  %44 = icmp slt i64 %38, %43, !dbg !376
  %45 = sext i1 %44 to i32, !dbg !376
  %46 = trunc i32 %45 to i1, !dbg !376
  %47 = bitcast i64* %__nv_MAIN_F1L36_4Arg2 to i8*, !dbg !376
  %48 = getelementptr i8, i8* %47, i64 152, !dbg !376
  %49 = bitcast i8* %48 to i64**, !dbg !376
  %50 = load i64*, i64** %49, align 8, !dbg !376
  %51 = load i64, i64* %50, align 8, !dbg !376
  %52 = add nsw i64 %51, 256, !dbg !376
  %53 = bitcast i64* %__nv_MAIN_F1L36_4Arg2 to i8*, !dbg !376
  %54 = getelementptr i8, i8* %53, i64 8, !dbg !376
  %55 = bitcast i8* %54 to i64**, !dbg !376
  %56 = load i64*, i64** %55, align 8, !dbg !376
  %57 = load i64, i64* %56, align 8, !dbg !376
  %58 = select i1 %46, i64 %52, i64 %57, !dbg !376
  store i64 %58, i64* %.de0003p_459, align 8, !dbg !376
  store i64 1, i64* %.di0003p_460, align 8, !dbg !376
  %59 = load i64, i64* %.di0003p_460, align 8, !dbg !376
  store i64 %59, i64* %.ds0003p_461, align 8, !dbg !376
  %60 = bitcast i64* %__nv_MAIN_F1L36_4Arg2 to i8*, !dbg !376
  %61 = getelementptr i8, i8* %60, i64 152, !dbg !376
  %62 = bitcast i8* %61 to i64**, !dbg !376
  %63 = load i64*, i64** %62, align 8, !dbg !376
  %64 = load i64, i64* %63, align 8, !dbg !376
  %65 = add nsw i64 %64, 1, !dbg !376
  store i64 %65, i64* %.dl0003p_463, align 8, !dbg !376
  %66 = load i64, i64* %.dl0003p_463, align 8, !dbg !376
  store i64 %66, i64* %.dl0003p.copy_759, align 8, !dbg !376
  %67 = load i64, i64* %.de0003p_459, align 8, !dbg !376
  store i64 %67, i64* %.de0003p.copy_760, align 8, !dbg !376
  %68 = load i64, i64* %.ds0003p_461, align 8, !dbg !376
  store i64 %68, i64* %.ds0003p.copy_761, align 8, !dbg !376
  %69 = load i32, i32* %__gtid___nv_MAIN_F1L36_4__765, align 4, !dbg !376
  %70 = bitcast i32* %.i0001p_402 to i64*, !dbg !376
  %71 = load i64, i64* %.ds0003p.copy_761, align 8, !dbg !376
  call void @__kmpc_for_static_init_8(i64* null, i32 %69, i32 34, i64* %70, i64* %.dl0003p.copy_759, i64* %.de0003p.copy_760, i64* %.ds0003p.copy_761, i64 %71, i64 1), !dbg !376
  %72 = load i64, i64* %.dl0003p.copy_759, align 8, !dbg !376
  store i64 %72, i64* %.dl0003p_463, align 8, !dbg !376
  %73 = load i64, i64* %.de0003p.copy_760, align 8, !dbg !376
  store i64 %73, i64* %.de0003p_459, align 8, !dbg !376
  %74 = load i64, i64* %.ds0003p.copy_761, align 8, !dbg !376
  store i64 %74, i64* %.ds0003p_461, align 8, !dbg !376
  %75 = load i64, i64* %.dl0003p_463, align 8, !dbg !376
  store i64 %75, i64* %i_401, align 8, !dbg !376
  %76 = load i64, i64* %i_401, align 8, !dbg !376
  call void @llvm.dbg.value(metadata i64 %76, metadata !377, metadata !DIExpression()), !dbg !373
  store i64 %76, i64* %.dX0003p_462, align 8, !dbg !376
  %77 = load i64, i64* %.dX0003p_462, align 8, !dbg !376
  %78 = load i64, i64* %.du0003p_458, align 8, !dbg !376
  %79 = icmp sgt i64 %77, %78, !dbg !376
  br i1 %79, label %L.LB7_456, label %L.LB7_774, !dbg !376

L.LB7_774:                                        ; preds = %L.LB7_399
  %80 = load i64, i64* %.dX0003p_462, align 8, !dbg !376
  store i64 %80, i64* %i_401, align 8, !dbg !376
  %81 = load i64, i64* %.di0003p_460, align 8, !dbg !376
  %82 = load i64, i64* %.de0003p_459, align 8, !dbg !376
  %83 = load i64, i64* %.dX0003p_462, align 8, !dbg !376
  %84 = sub nsw i64 %82, %83, !dbg !376
  %85 = add nsw i64 %81, %84, !dbg !376
  %86 = load i64, i64* %.di0003p_460, align 8, !dbg !376
  %87 = sdiv i64 %85, %86, !dbg !376
  store i64 %87, i64* %.dY0003p_457, align 8, !dbg !376
  %88 = load i64, i64* %.dY0003p_457, align 8, !dbg !376
  %89 = icmp sle i64 %88, 0, !dbg !376
  br i1 %89, label %L.LB7_466, label %L.LB7_465, !dbg !376

L.LB7_465:                                        ; preds = %L.LB7_465, %L.LB7_774
  %90 = load i64, i64* %i_401, align 8, !dbg !378
  call void @llvm.dbg.value(metadata i64 %90, metadata !377, metadata !DIExpression()), !dbg !373
  %91 = bitcast i64* %__nv_MAIN_F1L36_4Arg2 to i8*, !dbg !378
  %92 = getelementptr i8, i8* %91, i64 136, !dbg !378
  %93 = bitcast i8* %92 to i8**, !dbg !378
  %94 = load i8*, i8** %93, align 8, !dbg !378
  %95 = getelementptr i8, i8* %94, i64 56, !dbg !378
  %96 = bitcast i8* %95 to i64*, !dbg !378
  %97 = load i64, i64* %96, align 8, !dbg !378
  %98 = add nsw i64 %90, %97, !dbg !378
  %99 = bitcast i64* %__nv_MAIN_F1L36_4Arg2 to i8*, !dbg !378
  %100 = getelementptr i8, i8* %99, i64 80, !dbg !378
  %101 = bitcast i8* %100 to i8***, !dbg !378
  %102 = load i8**, i8*** %101, align 8, !dbg !378
  %103 = load i8*, i8** %102, align 8, !dbg !378
  %104 = getelementptr i8, i8* %103, i64 -8, !dbg !378
  %105 = bitcast i8* %104 to double*, !dbg !378
  %106 = getelementptr double, double* %105, i64 %98, !dbg !378
  %107 = load double, double* %106, align 8, !dbg !378
  %108 = load i64, i64* %i_401, align 8, !dbg !378
  call void @llvm.dbg.value(metadata i64 %108, metadata !377, metadata !DIExpression()), !dbg !373
  %109 = bitcast i64* %__nv_MAIN_F1L36_4Arg2 to i8*, !dbg !378
  %110 = getelementptr i8, i8* %109, i64 128, !dbg !378
  %111 = bitcast i8* %110 to i8**, !dbg !378
  %112 = load i8*, i8** %111, align 8, !dbg !378
  %113 = getelementptr i8, i8* %112, i64 56, !dbg !378
  %114 = bitcast i8* %113 to i64*, !dbg !378
  %115 = load i64, i64* %114, align 8, !dbg !378
  %116 = add nsw i64 %108, %115, !dbg !378
  %117 = bitcast i64* %__nv_MAIN_F1L36_4Arg2 to i8*, !dbg !378
  %118 = getelementptr i8, i8* %117, i64 24, !dbg !378
  %119 = bitcast i8* %118 to i8***, !dbg !378
  %120 = load i8**, i8*** %119, align 8, !dbg !378
  %121 = load i8*, i8** %120, align 8, !dbg !378
  %122 = getelementptr i8, i8* %121, i64 -8, !dbg !378
  %123 = bitcast i8* %122 to double*, !dbg !378
  %124 = getelementptr double, double* %123, i64 %116, !dbg !378
  %125 = load double, double* %124, align 8, !dbg !378
  %126 = fmul fast double %107, %125, !dbg !378
  %127 = load double, double* %sum_400, align 8, !dbg !378
  call void @llvm.dbg.value(metadata double %127, metadata !374, metadata !DIExpression()), !dbg !373
  %128 = fadd fast double %126, %127, !dbg !378
  store double %128, double* %sum_400, align 8, !dbg !378
  %129 = load i64, i64* %.di0003p_460, align 8, !dbg !373
  %130 = load i64, i64* %i_401, align 8, !dbg !373
  call void @llvm.dbg.value(metadata i64 %130, metadata !377, metadata !DIExpression()), !dbg !373
  %131 = add nsw i64 %129, %130, !dbg !373
  store i64 %131, i64* %i_401, align 8, !dbg !373
  %132 = load i64, i64* %.dY0003p_457, align 8, !dbg !373
  %133 = sub nsw i64 %132, 1, !dbg !373
  store i64 %133, i64* %.dY0003p_457, align 8, !dbg !373
  %134 = load i64, i64* %.dY0003p_457, align 8, !dbg !373
  %135 = icmp sgt i64 %134, 0, !dbg !373
  br i1 %135, label %L.LB7_465, label %L.LB7_466, !dbg !373

L.LB7_466:                                        ; preds = %L.LB7_465, %L.LB7_774
  br label %L.LB7_456

L.LB7_456:                                        ; preds = %L.LB7_466, %L.LB7_399
  %136 = load i32, i32* %__gtid___nv_MAIN_F1L36_4__765, align 4, !dbg !373
  call void @__kmpc_for_static_fini(i64* null, i32 %136), !dbg !373
  %137 = call i32 (...) @_mp_bcs_nest_red(), !dbg !373
  %138 = call i32 (...) @_mp_bcs_nest_red(), !dbg !373
  %139 = load double, double* %sum_400, align 8, !dbg !373
  call void @llvm.dbg.value(metadata double %139, metadata !374, metadata !DIExpression()), !dbg !373
  %140 = bitcast i64* %__nv_MAIN_F1L36_4Arg2 to i8*, !dbg !373
  %141 = getelementptr i8, i8* %140, i64 144, !dbg !373
  %142 = bitcast i8* %141 to double**, !dbg !373
  %143 = load double*, double** %142, align 8, !dbg !373
  %144 = load double, double* %143, align 8, !dbg !373
  %145 = fadd fast double %139, %144, !dbg !373
  %146 = bitcast i64* %__nv_MAIN_F1L36_4Arg2 to i8*, !dbg !373
  %147 = getelementptr i8, i8* %146, i64 144, !dbg !373
  %148 = bitcast i8* %147 to double**, !dbg !373
  %149 = load double*, double** %148, align 8, !dbg !373
  store double %145, double* %149, align 8, !dbg !373
  %150 = call i32 (...) @_mp_ecs_nest_red(), !dbg !373
  %151 = call i32 (...) @_mp_ecs_nest_red(), !dbg !373
  br label %L.LB7_403

L.LB7_403:                                        ; preds = %L.LB7_456
  ret void, !dbg !373
}

declare signext i32 @_mp_ecs_nest_red(...) #0

declare signext i32 @_mp_bcs_nest_red(...) #0

declare void @__kmpc_for_static_fini(i64*, i32) #0

declare void @__kmpc_for_static_init_8(i64*, i32, i32, i64*, i64*, i64*, i64*, i64, i64) #0

declare void @__kmpc_fork_teams(i64*, i32, i64*, i64*, ...) #0

declare void @__kmpc_push_num_teams(i64*, i32, i32, i32) #0

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

declare signext i32 @__kmpc_global_thread_num(i64*) #0

declare void @__kmpc_fork_call(i64*, i32, i64*, i64*, ...) #0

declare void @__kmpc_end_serialized_parallel(i64*, i32) #0

declare void @__kmpc_serialized_parallel(i64*, i32) #0

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1 immarg) #2

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { "no-frame-pointer-elim-non-leaf" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { argmemonly nounwind }

!llvm.module.flags = !{!83, !84}
!llvm.dbg.cu = !{!3}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "c_null_ptr$ac", scope: !2, file: !4, type: !80, isLocal: false, isDefinition: true)
!2 = !DIModule(scope: !3, name: "iso_c_binding")
!3 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !4, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !6, imports: !75)
!4 = !DIFile(filename: "micro-benchmarks-fortran/DRB097-target-teams-distribute-orig-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!5 = !{}
!6 = !{!0, !7, !13, !21, !27, !33, !39, !45, !51, !57, !63, !69}
!7 = !DIGlobalVariableExpression(var: !8, expr: !DIExpression(DW_OP_plus_uconst, 8))
!8 = distinct !DIGlobalVariable(name: "c_null_funptr$ac", scope: !2, file: !4, type: !9, isLocal: false, isDefinition: true)
!9 = !DICompositeType(tag: DW_TAG_structure_type, name: "c_funptr", file: !4, size: 64, align: 64, elements: !10)
!10 = !{!11}
!11 = !DIDerivedType(tag: DW_TAG_member, name: "val", scope: !9, file: !4, baseType: !12, size: 64, align: 64)
!12 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!13 = !DIGlobalVariableExpression(var: !14, expr: !DIExpression())
!14 = distinct !DIGlobalVariable(name: "iso_c_binding$$$c_ptr$$td", scope: !2, file: !4, type: !15, isLocal: false, isDefinition: true)
!15 = !DICompositeType(tag: DW_TAG_array_type, baseType: !16, size: 64, align: 64, elements: !19)
!16 = !DICompositeType(tag: DW_TAG_structure_type, name: "c_ptr", file: !4, size: 64, align: 64, elements: !17)
!17 = !{!18}
!18 = !DIDerivedType(tag: DW_TAG_member, name: "val", scope: !16, file: !4, baseType: !12, size: 64, align: 64)
!19 = !{!20}
!20 = !DISubrange(count: 0, lowerBound: 1)
!21 = !DIGlobalVariableExpression(var: !22, expr: !DIExpression())
!22 = distinct !DIGlobalVariable(name: "iso_c_binding$$$c_funptr$$td", scope: !2, file: !4, type: !23, isLocal: false, isDefinition: true)
!23 = !DICompositeType(tag: DW_TAG_array_type, baseType: !24, size: 64, align: 64, elements: !19)
!24 = !DICompositeType(tag: DW_TAG_structure_type, name: "c_funptr", file: !4, size: 64, align: 64, elements: !25)
!25 = !{!26}
!26 = !DIDerivedType(tag: DW_TAG_member, name: "val", scope: !24, file: !4, baseType: !12, size: 64, align: 64)
!27 = !DIGlobalVariableExpression(var: !28, expr: !DIExpression())
!28 = distinct !DIGlobalVariable(name: "iso_c_binding$$$c_ptr$$td", scope: !3, file: !4, type: !29, isLocal: false, isDefinition: true)
!29 = !DICompositeType(tag: DW_TAG_array_type, baseType: !30, size: 64, align: 64, elements: !19)
!30 = !DICompositeType(tag: DW_TAG_structure_type, name: "c_ptr", file: !4, size: 64, align: 64, elements: !31)
!31 = !{!32}
!32 = !DIDerivedType(tag: DW_TAG_member, name: "val", scope: !30, file: !4, baseType: !12, size: 64, align: 64)
!33 = !DIGlobalVariableExpression(var: !34, expr: !DIExpression())
!34 = distinct !DIGlobalVariable(name: "iso_c_binding$$$c_funptr$$td", scope: !3, file: !4, type: !35, isLocal: false, isDefinition: true)
!35 = !DICompositeType(tag: DW_TAG_array_type, baseType: !36, size: 64, align: 64, elements: !19)
!36 = !DICompositeType(tag: DW_TAG_structure_type, name: "c_funptr", file: !4, size: 64, align: 64, elements: !37)
!37 = !{!38}
!38 = !DIDerivedType(tag: DW_TAG_member, name: "val", scope: !36, file: !4, baseType: !12, size: 64, align: 64)
!39 = !DIGlobalVariableExpression(var: !40, expr: !DIExpression())
!40 = distinct !DIGlobalVariable(name: "iso_c_binding$$$c_ptr$$td", scope: !3, file: !4, type: !41, isLocal: false, isDefinition: true)
!41 = !DICompositeType(tag: DW_TAG_array_type, baseType: !42, size: 64, align: 64, elements: !19)
!42 = !DICompositeType(tag: DW_TAG_structure_type, name: "c_ptr", file: !4, size: 64, align: 64, elements: !43)
!43 = !{!44}
!44 = !DIDerivedType(tag: DW_TAG_member, name: "val", scope: !42, file: !4, baseType: !12, size: 64, align: 64)
!45 = !DIGlobalVariableExpression(var: !46, expr: !DIExpression())
!46 = distinct !DIGlobalVariable(name: "iso_c_binding$$$c_funptr$$td", scope: !3, file: !4, type: !47, isLocal: false, isDefinition: true)
!47 = !DICompositeType(tag: DW_TAG_array_type, baseType: !48, size: 64, align: 64, elements: !19)
!48 = !DICompositeType(tag: DW_TAG_structure_type, name: "c_funptr", file: !4, size: 64, align: 64, elements: !49)
!49 = !{!50}
!50 = !DIDerivedType(tag: DW_TAG_member, name: "val", scope: !48, file: !4, baseType: !12, size: 64, align: 64)
!51 = !DIGlobalVariableExpression(var: !52, expr: !DIExpression())
!52 = distinct !DIGlobalVariable(name: "iso_c_binding$$$c_ptr$$td", scope: !3, file: !4, type: !53, isLocal: false, isDefinition: true)
!53 = !DICompositeType(tag: DW_TAG_array_type, baseType: !54, size: 64, align: 64, elements: !19)
!54 = !DICompositeType(tag: DW_TAG_structure_type, name: "c_ptr", file: !4, size: 64, align: 64, elements: !55)
!55 = !{!56}
!56 = !DIDerivedType(tag: DW_TAG_member, name: "val", scope: !54, file: !4, baseType: !12, size: 64, align: 64)
!57 = !DIGlobalVariableExpression(var: !58, expr: !DIExpression())
!58 = distinct !DIGlobalVariable(name: "iso_c_binding$$$c_funptr$$td", scope: !3, file: !4, type: !59, isLocal: false, isDefinition: true)
!59 = !DICompositeType(tag: DW_TAG_array_type, baseType: !60, size: 64, align: 64, elements: !19)
!60 = !DICompositeType(tag: DW_TAG_structure_type, name: "c_funptr", file: !4, size: 64, align: 64, elements: !61)
!61 = !{!62}
!62 = !DIDerivedType(tag: DW_TAG_member, name: "val", scope: !60, file: !4, baseType: !12, size: 64, align: 64)
!63 = !DIGlobalVariableExpression(var: !64, expr: !DIExpression())
!64 = distinct !DIGlobalVariable(name: "iso_c_binding$$$c_ptr$$td", scope: !3, file: !4, type: !65, isLocal: false, isDefinition: true)
!65 = !DICompositeType(tag: DW_TAG_array_type, baseType: !66, size: 64, align: 64, elements: !19)
!66 = !DICompositeType(tag: DW_TAG_structure_type, name: "c_ptr", file: !4, size: 64, align: 64, elements: !67)
!67 = !{!68}
!68 = !DIDerivedType(tag: DW_TAG_member, name: "val", scope: !66, file: !4, baseType: !12, size: 64, align: 64)
!69 = !DIGlobalVariableExpression(var: !70, expr: !DIExpression())
!70 = distinct !DIGlobalVariable(name: "iso_c_binding$$$c_funptr$$td", scope: !3, file: !4, type: !71, isLocal: false, isDefinition: true)
!71 = !DICompositeType(tag: DW_TAG_array_type, baseType: !72, size: 64, align: 64, elements: !19)
!72 = !DICompositeType(tag: DW_TAG_structure_type, name: "c_funptr", file: !4, size: 64, align: 64, elements: !73)
!73 = !{!74}
!74 = !DIDerivedType(tag: DW_TAG_member, name: "val", scope: !72, file: !4, baseType: !12, size: 64, align: 64)
!75 = !{!76}
!76 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !77, entity: !2, file: !4, line: 10)
!77 = distinct !DISubprogram(name: "drb097_target_teams_distribute_orig_no", scope: !3, file: !4, line: 10, type: !78, scopeLine: 10, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !3)
!78 = !DISubroutineType(cc: DW_CC_program, types: !79)
!79 = !{null}
!80 = !DICompositeType(tag: DW_TAG_structure_type, name: "c_ptr", file: !4, size: 64, align: 64, elements: !81)
!81 = !{!82}
!82 = !DIDerivedType(tag: DW_TAG_member, name: "val", scope: !80, file: !4, baseType: !12, size: 64, align: 64)
!83 = !{i32 2, !"Dwarf Version", i32 4}
!84 = !{i32 2, !"Debug Info Version", i32 3}
!85 = !DILocalVariable(name: "c_int", scope: !77, file: !4, type: !86)
!86 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!87 = !DILocation(line: 0, scope: !77)
!88 = !DILocalVariable(name: "c_long", scope: !77, file: !4, type: !86)
!89 = !DILocalVariable(name: "c_intptr_t", scope: !77, file: !4, type: !86)
!90 = !DILocalVariable(name: "c_size_t", scope: !77, file: !4, type: !86)
!91 = !DILocalVariable(name: "c_long_long", scope: !77, file: !4, type: !86)
!92 = !DILocalVariable(name: "c_signed_char", scope: !77, file: !4, type: !86)
!93 = !DILocalVariable(name: "c_int8_t", scope: !77, file: !4, type: !86)
!94 = !DILocalVariable(name: "c_int32_t", scope: !77, file: !4, type: !86)
!95 = !DILocalVariable(name: "c_int64_t", scope: !77, file: !4, type: !86)
!96 = !DILocalVariable(name: "c_int_least8_t", scope: !77, file: !4, type: !86)
!97 = !DILocalVariable(name: "c_int_least32_t", scope: !77, file: !4, type: !86)
!98 = !DILocalVariable(name: "c_int_least64_t", scope: !77, file: !4, type: !86)
!99 = !DILocalVariable(name: "c_int_fast8_t", scope: !77, file: !4, type: !86)
!100 = !DILocalVariable(name: "c_int_fast16_t", scope: !77, file: !4, type: !86)
!101 = !DILocalVariable(name: "c_int_fast32_t", scope: !77, file: !4, type: !86)
!102 = !DILocalVariable(name: "c_int_fast64_t", scope: !77, file: !4, type: !86)
!103 = !DILocalVariable(name: "c_intmax_t", scope: !77, file: !4, type: !86)
!104 = !DILocalVariable(name: "c_float", scope: !77, file: !4, type: !86)
!105 = !DILocalVariable(name: "c_double", scope: !77, file: !4, type: !86)
!106 = !DILocalVariable(name: "c_long_double", scope: !77, file: !4, type: !86)
!107 = !DILocalVariable(name: "c_float_complex", scope: !77, file: !4, type: !86)
!108 = !DILocalVariable(name: "c_double_complex", scope: !77, file: !4, type: !86)
!109 = !DILocalVariable(name: "c_long_double_complex", scope: !77, file: !4, type: !86)
!110 = !DILocalVariable(name: "c_bool", scope: !77, file: !4, type: !86)
!111 = !DILocalVariable(name: "c_char", scope: !77, file: !4, type: !86)
!112 = !DILocalVariable(name: "omp_integer_kind", scope: !77, file: !4, type: !86)
!113 = !DILocalVariable(name: "omp_logical_kind", scope: !77, file: !4, type: !86)
!114 = !DILocalVariable(name: "omp_lock_kind", scope: !77, file: !4, type: !86)
!115 = !DILocalVariable(name: "omp_nest_lock_kind", scope: !77, file: !4, type: !86)
!116 = !DILocalVariable(name: "omp_sched_kind", scope: !77, file: !4, type: !86)
!117 = !DILocalVariable(name: "omp_real_kind", scope: !77, file: !4, type: !86)
!118 = !DILocalVariable(name: "omp_proc_bind_kind", scope: !77, file: !4, type: !86)
!119 = !DILocalVariable(name: "omp_lock_hint_kind", scope: !77, file: !4, type: !86)
!120 = !DILocalVariable(name: "omp_sched_static", scope: !77, file: !4, type: !86)
!121 = !DILocalVariable(name: "omp_sched_auto", scope: !77, file: !4, type: !86)
!122 = !DILocalVariable(name: "omp_proc_bind_false", scope: !77, file: !4, type: !86)
!123 = !DILocalVariable(name: "omp_proc_bind_true", scope: !77, file: !4, type: !86)
!124 = !DILocalVariable(name: "omp_proc_bind_spread", scope: !77, file: !4, type: !86)
!125 = !DILocalVariable(name: "omp_lock_hint_none", scope: !77, file: !4, type: !86)
!126 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !77, file: !4, type: !86)
!127 = !DILocalVariable(name: "omp_lock_hint_nonspeculative", scope: !77, file: !4, type: !86)
!128 = !DILocalVariable(name: "omp_lock_hint_speculative", scope: !77, file: !4, type: !86)
!129 = !DILocalVariable(name: "dp", scope: !77, file: !4, type: !86)
!130 = !DILocalVariable(name: "double_kind", scope: !77, file: !4, type: !86)
!131 = !DILocation(line: 55, column: 1, scope: !77)
!132 = !DILocation(line: 10, column: 1, scope: !77)
!133 = !DILocalVariable(name: "b", scope: !77, file: !4, type: !134)
!134 = !DICompositeType(tag: DW_TAG_array_type, baseType: !135, size: 64, align: 64, elements: !19)
!135 = !DIBasicType(name: "double precision", size: 64, align: 64, encoding: DW_ATE_float)
!136 = !DILocalVariable(scope: !77, file: !4, type: !137, flags: DIFlagArtificial)
!137 = !DICompositeType(tag: DW_TAG_array_type, baseType: !12, size: 1024, align: 64, elements: !138)
!138 = !{!139}
!139 = !DISubrange(count: 16, lowerBound: 1)
!140 = !DILocalVariable(name: "a", scope: !77, file: !4, type: !134)
!141 = !DILocalVariable(name: "len", scope: !77, file: !4, type: !12)
!142 = !DILocation(line: 20, column: 1, scope: !77)
!143 = !DILocalVariable(name: "sum", scope: !77, file: !4, type: !135)
!144 = !DILocation(line: 21, column: 1, scope: !77)
!145 = !DILocalVariable(name: "sum2", scope: !77, file: !4, type: !135)
!146 = !DILocation(line: 22, column: 1, scope: !77)
!147 = !DILocalVariable(scope: !77, file: !4, type: !12, flags: DIFlagArtificial)
!148 = !DILocation(line: 24, column: 1, scope: !77)
!149 = !DILocation(line: 25, column: 1, scope: !77)
!150 = !DILocation(line: 27, column: 1, scope: !77)
!151 = !DILocalVariable(name: "i", scope: !77, file: !4, type: !12)
!152 = !DILocation(line: 28, column: 1, scope: !77)
!153 = !DILocation(line: 29, column: 1, scope: !77)
!154 = !DILocation(line: 30, column: 1, scope: !77)
!155 = !DILocation(line: 44, column: 1, scope: !77)
!156 = !DILocation(line: 46, column: 1, scope: !77)
!157 = !DILocation(line: 52, column: 1, scope: !77)
!158 = !DILocalVariable(scope: !77, file: !4, type: !86, flags: DIFlagArtificial)
!159 = !DILocation(line: 54, column: 1, scope: !77)
!160 = distinct !DISubprogram(name: "__nv_MAIN__F1L32_1", scope: !3, file: !4, line: 32, type: !161, scopeLine: 32, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !3)
!161 = !DISubroutineType(types: !162)
!162 = !{null, !86, !12, !12}
!163 = !DILocalVariable(name: "__nv_MAIN__F1L32_1Arg0", arg: 1, scope: !160, file: !4, type: !86)
!164 = !DILocation(line: 0, scope: !160)
!165 = !DILocalVariable(name: "__nv_MAIN__F1L32_1Arg1", arg: 2, scope: !160, file: !4, type: !12)
!166 = !DILocalVariable(name: "__nv_MAIN__F1L32_1Arg2", arg: 3, scope: !160, file: !4, type: !12)
!167 = !DILocalVariable(name: "c_int", scope: !160, file: !4, type: !86)
!168 = !DILocalVariable(name: "c_long", scope: !160, file: !4, type: !86)
!169 = !DILocalVariable(name: "c_intptr_t", scope: !160, file: !4, type: !86)
!170 = !DILocalVariable(name: "c_size_t", scope: !160, file: !4, type: !86)
!171 = !DILocalVariable(name: "c_long_long", scope: !160, file: !4, type: !86)
!172 = !DILocalVariable(name: "c_signed_char", scope: !160, file: !4, type: !86)
!173 = !DILocalVariable(name: "c_int8_t", scope: !160, file: !4, type: !86)
!174 = !DILocalVariable(name: "c_int32_t", scope: !160, file: !4, type: !86)
!175 = !DILocalVariable(name: "c_int64_t", scope: !160, file: !4, type: !86)
!176 = !DILocalVariable(name: "c_int_least8_t", scope: !160, file: !4, type: !86)
!177 = !DILocalVariable(name: "c_int_least32_t", scope: !160, file: !4, type: !86)
!178 = !DILocalVariable(name: "c_int_least64_t", scope: !160, file: !4, type: !86)
!179 = !DILocalVariable(name: "c_int_fast8_t", scope: !160, file: !4, type: !86)
!180 = !DILocalVariable(name: "c_int_fast16_t", scope: !160, file: !4, type: !86)
!181 = !DILocalVariable(name: "c_int_fast32_t", scope: !160, file: !4, type: !86)
!182 = !DILocalVariable(name: "c_int_fast64_t", scope: !160, file: !4, type: !86)
!183 = !DILocalVariable(name: "c_intmax_t", scope: !160, file: !4, type: !86)
!184 = !DILocalVariable(name: "c_float", scope: !160, file: !4, type: !86)
!185 = !DILocalVariable(name: "c_double", scope: !160, file: !4, type: !86)
!186 = !DILocalVariable(name: "c_long_double", scope: !160, file: !4, type: !86)
!187 = !DILocalVariable(name: "c_float_complex", scope: !160, file: !4, type: !86)
!188 = !DILocalVariable(name: "c_double_complex", scope: !160, file: !4, type: !86)
!189 = !DILocalVariable(name: "c_long_double_complex", scope: !160, file: !4, type: !86)
!190 = !DILocalVariable(name: "c_bool", scope: !160, file: !4, type: !86)
!191 = !DILocalVariable(name: "c_char", scope: !160, file: !4, type: !86)
!192 = !DILocalVariable(name: "omp_integer_kind", scope: !160, file: !4, type: !86)
!193 = !DILocalVariable(name: "omp_logical_kind", scope: !160, file: !4, type: !86)
!194 = !DILocalVariable(name: "omp_lock_kind", scope: !160, file: !4, type: !86)
!195 = !DILocalVariable(name: "omp_nest_lock_kind", scope: !160, file: !4, type: !86)
!196 = !DILocalVariable(name: "omp_sched_kind", scope: !160, file: !4, type: !86)
!197 = !DILocalVariable(name: "omp_real_kind", scope: !160, file: !4, type: !86)
!198 = !DILocalVariable(name: "omp_proc_bind_kind", scope: !160, file: !4, type: !86)
!199 = !DILocalVariable(name: "omp_lock_hint_kind", scope: !160, file: !4, type: !86)
!200 = !DILocalVariable(name: "omp_sched_static", scope: !160, file: !4, type: !86)
!201 = !DILocalVariable(name: "omp_sched_auto", scope: !160, file: !4, type: !86)
!202 = !DILocalVariable(name: "omp_proc_bind_false", scope: !160, file: !4, type: !86)
!203 = !DILocalVariable(name: "omp_proc_bind_true", scope: !160, file: !4, type: !86)
!204 = !DILocalVariable(name: "omp_proc_bind_spread", scope: !160, file: !4, type: !86)
!205 = !DILocalVariable(name: "omp_lock_hint_none", scope: !160, file: !4, type: !86)
!206 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !160, file: !4, type: !86)
!207 = !DILocalVariable(name: "omp_lock_hint_nonspeculative", scope: !160, file: !4, type: !86)
!208 = !DILocalVariable(name: "omp_lock_hint_speculative", scope: !160, file: !4, type: !86)
!209 = !DILocalVariable(name: "dp", scope: !160, file: !4, type: !86)
!210 = !DILocalVariable(name: "double_kind", scope: !160, file: !4, type: !86)
!211 = !DILocation(line: 44, column: 1, scope: !160)
!212 = !DILocation(line: 33, column: 1, scope: !160)
!213 = distinct !DISubprogram(name: "__nv_MAIN__F1L46_2", scope: !3, file: !4, line: 46, type: !161, scopeLine: 46, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !3)
!214 = !DILocalVariable(name: "__nv_MAIN__F1L46_2Arg0", arg: 1, scope: !213, file: !4, type: !86)
!215 = !DILocation(line: 0, scope: !213)
!216 = !DILocalVariable(name: "__nv_MAIN__F1L46_2Arg1", arg: 2, scope: !213, file: !4, type: !12)
!217 = !DILocalVariable(name: "__nv_MAIN__F1L46_2Arg2", arg: 3, scope: !213, file: !4, type: !12)
!218 = !DILocalVariable(name: "c_int", scope: !213, file: !4, type: !86)
!219 = !DILocalVariable(name: "c_long", scope: !213, file: !4, type: !86)
!220 = !DILocalVariable(name: "c_intptr_t", scope: !213, file: !4, type: !86)
!221 = !DILocalVariable(name: "c_size_t", scope: !213, file: !4, type: !86)
!222 = !DILocalVariable(name: "c_long_long", scope: !213, file: !4, type: !86)
!223 = !DILocalVariable(name: "c_signed_char", scope: !213, file: !4, type: !86)
!224 = !DILocalVariable(name: "c_int8_t", scope: !213, file: !4, type: !86)
!225 = !DILocalVariable(name: "c_int32_t", scope: !213, file: !4, type: !86)
!226 = !DILocalVariable(name: "c_int64_t", scope: !213, file: !4, type: !86)
!227 = !DILocalVariable(name: "c_int_least8_t", scope: !213, file: !4, type: !86)
!228 = !DILocalVariable(name: "c_int_least32_t", scope: !213, file: !4, type: !86)
!229 = !DILocalVariable(name: "c_int_least64_t", scope: !213, file: !4, type: !86)
!230 = !DILocalVariable(name: "c_int_fast8_t", scope: !213, file: !4, type: !86)
!231 = !DILocalVariable(name: "c_int_fast16_t", scope: !213, file: !4, type: !86)
!232 = !DILocalVariable(name: "c_int_fast32_t", scope: !213, file: !4, type: !86)
!233 = !DILocalVariable(name: "c_int_fast64_t", scope: !213, file: !4, type: !86)
!234 = !DILocalVariable(name: "c_intmax_t", scope: !213, file: !4, type: !86)
!235 = !DILocalVariable(name: "c_float", scope: !213, file: !4, type: !86)
!236 = !DILocalVariable(name: "c_double", scope: !213, file: !4, type: !86)
!237 = !DILocalVariable(name: "c_long_double", scope: !213, file: !4, type: !86)
!238 = !DILocalVariable(name: "c_float_complex", scope: !213, file: !4, type: !86)
!239 = !DILocalVariable(name: "c_double_complex", scope: !213, file: !4, type: !86)
!240 = !DILocalVariable(name: "c_long_double_complex", scope: !213, file: !4, type: !86)
!241 = !DILocalVariable(name: "c_bool", scope: !213, file: !4, type: !86)
!242 = !DILocalVariable(name: "c_char", scope: !213, file: !4, type: !86)
!243 = !DILocalVariable(name: "omp_integer_kind", scope: !213, file: !4, type: !86)
!244 = !DILocalVariable(name: "omp_logical_kind", scope: !213, file: !4, type: !86)
!245 = !DILocalVariable(name: "omp_lock_kind", scope: !213, file: !4, type: !86)
!246 = !DILocalVariable(name: "omp_nest_lock_kind", scope: !213, file: !4, type: !86)
!247 = !DILocalVariable(name: "omp_sched_kind", scope: !213, file: !4, type: !86)
!248 = !DILocalVariable(name: "omp_real_kind", scope: !213, file: !4, type: !86)
!249 = !DILocalVariable(name: "omp_proc_bind_kind", scope: !213, file: !4, type: !86)
!250 = !DILocalVariable(name: "omp_lock_hint_kind", scope: !213, file: !4, type: !86)
!251 = !DILocalVariable(name: "omp_sched_static", scope: !213, file: !4, type: !86)
!252 = !DILocalVariable(name: "omp_sched_auto", scope: !213, file: !4, type: !86)
!253 = !DILocalVariable(name: "omp_proc_bind_false", scope: !213, file: !4, type: !86)
!254 = !DILocalVariable(name: "omp_proc_bind_true", scope: !213, file: !4, type: !86)
!255 = !DILocalVariable(name: "omp_proc_bind_spread", scope: !213, file: !4, type: !86)
!256 = !DILocalVariable(name: "omp_lock_hint_none", scope: !213, file: !4, type: !86)
!257 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !213, file: !4, type: !86)
!258 = !DILocalVariable(name: "omp_lock_hint_nonspeculative", scope: !213, file: !4, type: !86)
!259 = !DILocalVariable(name: "omp_lock_hint_speculative", scope: !213, file: !4, type: !86)
!260 = !DILocalVariable(name: "dp", scope: !213, file: !4, type: !86)
!261 = !DILocalVariable(name: "double_kind", scope: !213, file: !4, type: !86)
!262 = !DILocation(line: 49, column: 1, scope: !213)
!263 = !DILocalVariable(name: "sum2", scope: !213, file: !4, type: !135)
!264 = !DILocation(line: 46, column: 1, scope: !213)
!265 = !DILocation(line: 47, column: 1, scope: !213)
!266 = !DILocalVariable(name: "i", scope: !213, file: !4, type: !12)
!267 = !DILocation(line: 48, column: 1, scope: !213)
!268 = distinct !DISubprogram(name: "__nv_MAIN_F1L33_3", scope: !3, file: !4, line: 33, type: !161, scopeLine: 33, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !3)
!269 = !DILocalVariable(name: "__nv_MAIN_F1L33_3Arg0", arg: 1, scope: !268, file: !4, type: !86)
!270 = !DILocation(line: 0, scope: !268)
!271 = !DILocalVariable(name: "__nv_MAIN_F1L33_3Arg1", arg: 2, scope: !268, file: !4, type: !12)
!272 = !DILocalVariable(name: "__nv_MAIN_F1L33_3Arg2", arg: 3, scope: !268, file: !4, type: !12)
!273 = !DILocalVariable(name: "c_int", scope: !268, file: !4, type: !86)
!274 = !DILocalVariable(name: "c_long", scope: !268, file: !4, type: !86)
!275 = !DILocalVariable(name: "c_intptr_t", scope: !268, file: !4, type: !86)
!276 = !DILocalVariable(name: "c_size_t", scope: !268, file: !4, type: !86)
!277 = !DILocalVariable(name: "c_long_long", scope: !268, file: !4, type: !86)
!278 = !DILocalVariable(name: "c_signed_char", scope: !268, file: !4, type: !86)
!279 = !DILocalVariable(name: "c_int8_t", scope: !268, file: !4, type: !86)
!280 = !DILocalVariable(name: "c_int32_t", scope: !268, file: !4, type: !86)
!281 = !DILocalVariable(name: "c_int64_t", scope: !268, file: !4, type: !86)
!282 = !DILocalVariable(name: "c_int_least8_t", scope: !268, file: !4, type: !86)
!283 = !DILocalVariable(name: "c_int_least32_t", scope: !268, file: !4, type: !86)
!284 = !DILocalVariable(name: "c_int_least64_t", scope: !268, file: !4, type: !86)
!285 = !DILocalVariable(name: "c_int_fast8_t", scope: !268, file: !4, type: !86)
!286 = !DILocalVariable(name: "c_int_fast16_t", scope: !268, file: !4, type: !86)
!287 = !DILocalVariable(name: "c_int_fast32_t", scope: !268, file: !4, type: !86)
!288 = !DILocalVariable(name: "c_int_fast64_t", scope: !268, file: !4, type: !86)
!289 = !DILocalVariable(name: "c_intmax_t", scope: !268, file: !4, type: !86)
!290 = !DILocalVariable(name: "c_float", scope: !268, file: !4, type: !86)
!291 = !DILocalVariable(name: "c_double", scope: !268, file: !4, type: !86)
!292 = !DILocalVariable(name: "c_long_double", scope: !268, file: !4, type: !86)
!293 = !DILocalVariable(name: "c_float_complex", scope: !268, file: !4, type: !86)
!294 = !DILocalVariable(name: "c_double_complex", scope: !268, file: !4, type: !86)
!295 = !DILocalVariable(name: "c_long_double_complex", scope: !268, file: !4, type: !86)
!296 = !DILocalVariable(name: "c_bool", scope: !268, file: !4, type: !86)
!297 = !DILocalVariable(name: "c_char", scope: !268, file: !4, type: !86)
!298 = !DILocalVariable(name: "omp_integer_kind", scope: !268, file: !4, type: !86)
!299 = !DILocalVariable(name: "omp_logical_kind", scope: !268, file: !4, type: !86)
!300 = !DILocalVariable(name: "omp_lock_kind", scope: !268, file: !4, type: !86)
!301 = !DILocalVariable(name: "omp_nest_lock_kind", scope: !268, file: !4, type: !86)
!302 = !DILocalVariable(name: "omp_sched_kind", scope: !268, file: !4, type: !86)
!303 = !DILocalVariable(name: "omp_real_kind", scope: !268, file: !4, type: !86)
!304 = !DILocalVariable(name: "omp_proc_bind_kind", scope: !268, file: !4, type: !86)
!305 = !DILocalVariable(name: "omp_lock_hint_kind", scope: !268, file: !4, type: !86)
!306 = !DILocalVariable(name: "omp_sched_static", scope: !268, file: !4, type: !86)
!307 = !DILocalVariable(name: "omp_sched_auto", scope: !268, file: !4, type: !86)
!308 = !DILocalVariable(name: "omp_proc_bind_false", scope: !268, file: !4, type: !86)
!309 = !DILocalVariable(name: "omp_proc_bind_true", scope: !268, file: !4, type: !86)
!310 = !DILocalVariable(name: "omp_proc_bind_spread", scope: !268, file: !4, type: !86)
!311 = !DILocalVariable(name: "omp_lock_hint_none", scope: !268, file: !4, type: !86)
!312 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !268, file: !4, type: !86)
!313 = !DILocalVariable(name: "omp_lock_hint_nonspeculative", scope: !268, file: !4, type: !86)
!314 = !DILocalVariable(name: "omp_lock_hint_speculative", scope: !268, file: !4, type: !86)
!315 = !DILocalVariable(name: "dp", scope: !268, file: !4, type: !86)
!316 = !DILocalVariable(name: "double_kind", scope: !268, file: !4, type: !86)
!317 = !DILocation(line: 43, column: 1, scope: !268)
!318 = !DILocalVariable(name: "sum", scope: !268, file: !4, type: !135)
!319 = !DILocation(line: 33, column: 1, scope: !268)
!320 = !DILocation(line: 35, column: 1, scope: !268)
!321 = !DILocalVariable(name: "i2", scope: !268, file: !4, type: !12)
!322 = !DILocation(line: 36, column: 1, scope: !268)
!323 = !DILocation(line: 41, column: 1, scope: !268)
!324 = distinct !DISubprogram(name: "__nv_MAIN_F1L36_4", scope: !3, file: !4, line: 36, type: !161, scopeLine: 36, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !3)
!325 = !DILocalVariable(name: "__nv_MAIN_F1L36_4Arg0", arg: 1, scope: !324, file: !4, type: !86)
!326 = !DILocation(line: 0, scope: !324)
!327 = !DILocalVariable(name: "__nv_MAIN_F1L36_4Arg1", arg: 2, scope: !324, file: !4, type: !12)
!328 = !DILocalVariable(name: "__nv_MAIN_F1L36_4Arg2", arg: 3, scope: !324, file: !4, type: !12)
!329 = !DILocalVariable(name: "c_int", scope: !324, file: !4, type: !86)
!330 = !DILocalVariable(name: "c_long", scope: !324, file: !4, type: !86)
!331 = !DILocalVariable(name: "c_intptr_t", scope: !324, file: !4, type: !86)
!332 = !DILocalVariable(name: "c_size_t", scope: !324, file: !4, type: !86)
!333 = !DILocalVariable(name: "c_long_long", scope: !324, file: !4, type: !86)
!334 = !DILocalVariable(name: "c_signed_char", scope: !324, file: !4, type: !86)
!335 = !DILocalVariable(name: "c_int8_t", scope: !324, file: !4, type: !86)
!336 = !DILocalVariable(name: "c_int32_t", scope: !324, file: !4, type: !86)
!337 = !DILocalVariable(name: "c_int64_t", scope: !324, file: !4, type: !86)
!338 = !DILocalVariable(name: "c_int_least8_t", scope: !324, file: !4, type: !86)
!339 = !DILocalVariable(name: "c_int_least32_t", scope: !324, file: !4, type: !86)
!340 = !DILocalVariable(name: "c_int_least64_t", scope: !324, file: !4, type: !86)
!341 = !DILocalVariable(name: "c_int_fast8_t", scope: !324, file: !4, type: !86)
!342 = !DILocalVariable(name: "c_int_fast16_t", scope: !324, file: !4, type: !86)
!343 = !DILocalVariable(name: "c_int_fast32_t", scope: !324, file: !4, type: !86)
!344 = !DILocalVariable(name: "c_int_fast64_t", scope: !324, file: !4, type: !86)
!345 = !DILocalVariable(name: "c_intmax_t", scope: !324, file: !4, type: !86)
!346 = !DILocalVariable(name: "c_float", scope: !324, file: !4, type: !86)
!347 = !DILocalVariable(name: "c_double", scope: !324, file: !4, type: !86)
!348 = !DILocalVariable(name: "c_long_double", scope: !324, file: !4, type: !86)
!349 = !DILocalVariable(name: "c_float_complex", scope: !324, file: !4, type: !86)
!350 = !DILocalVariable(name: "c_double_complex", scope: !324, file: !4, type: !86)
!351 = !DILocalVariable(name: "c_long_double_complex", scope: !324, file: !4, type: !86)
!352 = !DILocalVariable(name: "c_bool", scope: !324, file: !4, type: !86)
!353 = !DILocalVariable(name: "c_char", scope: !324, file: !4, type: !86)
!354 = !DILocalVariable(name: "omp_integer_kind", scope: !324, file: !4, type: !86)
!355 = !DILocalVariable(name: "omp_logical_kind", scope: !324, file: !4, type: !86)
!356 = !DILocalVariable(name: "omp_lock_kind", scope: !324, file: !4, type: !86)
!357 = !DILocalVariable(name: "omp_nest_lock_kind", scope: !324, file: !4, type: !86)
!358 = !DILocalVariable(name: "omp_sched_kind", scope: !324, file: !4, type: !86)
!359 = !DILocalVariable(name: "omp_real_kind", scope: !324, file: !4, type: !86)
!360 = !DILocalVariable(name: "omp_proc_bind_kind", scope: !324, file: !4, type: !86)
!361 = !DILocalVariable(name: "omp_lock_hint_kind", scope: !324, file: !4, type: !86)
!362 = !DILocalVariable(name: "omp_sched_static", scope: !324, file: !4, type: !86)
!363 = !DILocalVariable(name: "omp_sched_auto", scope: !324, file: !4, type: !86)
!364 = !DILocalVariable(name: "omp_proc_bind_false", scope: !324, file: !4, type: !86)
!365 = !DILocalVariable(name: "omp_proc_bind_true", scope: !324, file: !4, type: !86)
!366 = !DILocalVariable(name: "omp_proc_bind_spread", scope: !324, file: !4, type: !86)
!367 = !DILocalVariable(name: "omp_lock_hint_none", scope: !324, file: !4, type: !86)
!368 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !324, file: !4, type: !86)
!369 = !DILocalVariable(name: "omp_lock_hint_nonspeculative", scope: !324, file: !4, type: !86)
!370 = !DILocalVariable(name: "omp_lock_hint_speculative", scope: !324, file: !4, type: !86)
!371 = !DILocalVariable(name: "dp", scope: !324, file: !4, type: !86)
!372 = !DILocalVariable(name: "double_kind", scope: !324, file: !4, type: !86)
!373 = !DILocation(line: 39, column: 1, scope: !324)
!374 = !DILocalVariable(name: "sum", scope: !324, file: !4, type: !135)
!375 = !DILocation(line: 36, column: 1, scope: !324)
!376 = !DILocation(line: 37, column: 1, scope: !324)
!377 = !DILocalVariable(name: "i", scope: !324, file: !4, type: !12)
!378 = !DILocation(line: 38, column: 1, scope: !324)

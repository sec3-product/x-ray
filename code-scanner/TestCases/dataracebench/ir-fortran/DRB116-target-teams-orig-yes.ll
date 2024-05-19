; ModuleID = '/tmp/DRB116-target-teams-orig-yes-a5acb5.ll'
source_filename = "/tmp/DRB116-target-teams-orig-yes-a5acb5.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%astruct.dt68 = type <{ i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8* }>
%astruct.dt113 = type <{ [64 x i8] }>

@.C312_MAIN_ = internal constant i32 14
@.C341_MAIN_ = internal constant [6 x i8] c"a(50)="
@.C338_MAIN_ = internal constant i32 6
@.C335_MAIN_ = internal constant [57 x i8] c"micro-benchmarks-fortran/DRB116-target-teams-orig-yes.f95"
@.C337_MAIN_ = internal constant i32 33
@.C331_MAIN_ = internal constant i64 50
@.C301_MAIN_ = internal constant i32 2
@.C293_MAIN_ = internal constant double 2.000000e+00
@.C300_MAIN_ = internal constant i32 8
@.C285_MAIN_ = internal constant i32 1
@.C313_MAIN_ = internal constant i32 28
@.C351_MAIN_ = internal constant i64 8
@.C350_MAIN_ = internal constant i64 28
@.C322_MAIN_ = internal constant i32 100
@.C286_MAIN_ = internal constant i64 1
@.C284_MAIN_ = internal constant i64 0
@.C283_MAIN_ = internal constant i32 0
@.C293___nv_MAIN__F1L27_1 = internal constant double 2.000000e+00
@.C331___nv_MAIN__F1L27_1 = internal constant i64 50
@.C283___nv_MAIN__F1L27_1 = internal constant i32 0
@.C301___nv_MAIN__F1L27_1 = internal constant i32 2
@.C293___nv_MAIN_F1L28_2 = internal constant double 2.000000e+00
@.C331___nv_MAIN_F1L28_2 = internal constant i64 50

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__404 = alloca i32, align 4
  %.Z0966_324 = alloca double*, align 8
  %"a$sd1_349" = alloca [16 x i64], align 8
  %len_323 = alloca i32, align 4
  %z_b_0_316 = alloca i64, align 8
  %z_b_1_317 = alloca i64, align 8
  %z_e_62_320 = alloca i64, align 8
  %z_b_2_318 = alloca i64, align 8
  %z_b_3_319 = alloca i64, align 8
  %.dY0001_360 = alloca i32, align 4
  %i_314 = alloca i32, align 4
  %.uplevelArgPack0001_387 = alloca %astruct.dt68, align 16
  %z__io_340 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 8, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 2, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !14, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 2, metadata !15, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !16, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !17, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 2, metadata !18, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 8, metadata !19, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 8, metadata !20, metadata !DIExpression()), !dbg !10
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !21
  store i32 %0, i32* %__gtid_MAIN__404, align 4, !dbg !21
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !22
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !22
  call void (i8*, ...) %2(i8* %1), !dbg !22
  call void @llvm.dbg.declare(metadata double** %.Z0966_324, metadata !23, metadata !DIExpression(DW_OP_deref)), !dbg !10
  %3 = bitcast double** %.Z0966_324 to i8**, !dbg !22
  store i8* null, i8** %3, align 8, !dbg !22
  call void @llvm.dbg.declare(metadata [16 x i64]* %"a$sd1_349", metadata !28, metadata !DIExpression()), !dbg !10
  %4 = bitcast [16 x i64]* %"a$sd1_349" to i64*, !dbg !22
  store i64 0, i64* %4, align 8, !dbg !22
  br label %L.LB1_366

L.LB1_366:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %len_323, metadata !33, metadata !DIExpression()), !dbg !10
  store i32 100, i32* %len_323, align 4, !dbg !34
  call void @llvm.dbg.declare(metadata i64* %z_b_0_316, metadata !35, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_0_316, align 8, !dbg !36
  %5 = load i32, i32* %len_323, align 4, !dbg !36
  call void @llvm.dbg.value(metadata i32 %5, metadata !33, metadata !DIExpression()), !dbg !10
  %6 = sext i32 %5 to i64, !dbg !36
  call void @llvm.dbg.declare(metadata i64* %z_b_1_317, metadata !35, metadata !DIExpression()), !dbg !10
  store i64 %6, i64* %z_b_1_317, align 8, !dbg !36
  %7 = load i64, i64* %z_b_1_317, align 8, !dbg !36
  call void @llvm.dbg.value(metadata i64 %7, metadata !35, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_62_320, metadata !35, metadata !DIExpression()), !dbg !10
  store i64 %7, i64* %z_e_62_320, align 8, !dbg !36
  %8 = bitcast [16 x i64]* %"a$sd1_349" to i8*, !dbg !36
  %9 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !36
  %10 = bitcast i64* @.C350_MAIN_ to i8*, !dbg !36
  %11 = bitcast i64* @.C351_MAIN_ to i8*, !dbg !36
  %12 = bitcast i64* %z_b_0_316 to i8*, !dbg !36
  %13 = bitcast i64* %z_b_1_317 to i8*, !dbg !36
  %14 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !36
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %14(i8* %8, i8* %9, i8* %10, i8* %11, i8* %12, i8* %13), !dbg !36
  %15 = bitcast [16 x i64]* %"a$sd1_349" to i8*, !dbg !36
  %16 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !36
  call void (i8*, i32, ...) %16(i8* %15, i32 28), !dbg !36
  %17 = load i64, i64* %z_b_1_317, align 8, !dbg !36
  call void @llvm.dbg.value(metadata i64 %17, metadata !35, metadata !DIExpression()), !dbg !10
  %18 = load i64, i64* %z_b_0_316, align 8, !dbg !36
  call void @llvm.dbg.value(metadata i64 %18, metadata !35, metadata !DIExpression()), !dbg !10
  %19 = sub nsw i64 %18, 1, !dbg !36
  %20 = sub nsw i64 %17, %19, !dbg !36
  call void @llvm.dbg.declare(metadata i64* %z_b_2_318, metadata !35, metadata !DIExpression()), !dbg !10
  store i64 %20, i64* %z_b_2_318, align 8, !dbg !36
  %21 = load i64, i64* %z_b_0_316, align 8, !dbg !36
  call void @llvm.dbg.value(metadata i64 %21, metadata !35, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_b_3_319, metadata !35, metadata !DIExpression()), !dbg !10
  store i64 %21, i64* %z_b_3_319, align 8, !dbg !36
  %22 = bitcast i64* %z_b_2_318 to i8*, !dbg !36
  %23 = bitcast i64* @.C350_MAIN_ to i8*, !dbg !36
  %24 = bitcast i64* @.C351_MAIN_ to i8*, !dbg !36
  %25 = bitcast double** %.Z0966_324 to i8*, !dbg !36
  %26 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !36
  %27 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !36
  %28 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !36
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %28(i8* %22, i8* %23, i8* %24, i8* null, i8* %25, i8* null, i8* %26, i8* %27, i8* null, i64 0), !dbg !36
  %29 = load i32, i32* %len_323, align 4, !dbg !37
  call void @llvm.dbg.value(metadata i32 %29, metadata !33, metadata !DIExpression()), !dbg !10
  store i32 %29, i32* %.dY0001_360, align 4, !dbg !37
  call void @llvm.dbg.declare(metadata i32* %i_314, metadata !38, metadata !DIExpression()), !dbg !10
  store i32 1, i32* %i_314, align 4, !dbg !37
  %30 = load i32, i32* %.dY0001_360, align 4, !dbg !37
  %31 = icmp sle i32 %30, 0, !dbg !37
  br i1 %31, label %L.LB1_359, label %L.LB1_358, !dbg !37

L.LB1_358:                                        ; preds = %L.LB1_358, %L.LB1_366
  %32 = load i32, i32* %i_314, align 4, !dbg !39
  call void @llvm.dbg.value(metadata i32 %32, metadata !38, metadata !DIExpression()), !dbg !10
  %33 = sitofp i32 %32 to double, !dbg !39
  %34 = fdiv fast double %33, 2.000000e+00, !dbg !39
  %35 = load i32, i32* %i_314, align 4, !dbg !39
  call void @llvm.dbg.value(metadata i32 %35, metadata !38, metadata !DIExpression()), !dbg !10
  %36 = sext i32 %35 to i64, !dbg !39
  %37 = bitcast [16 x i64]* %"a$sd1_349" to i8*, !dbg !39
  %38 = getelementptr i8, i8* %37, i64 56, !dbg !39
  %39 = bitcast i8* %38 to i64*, !dbg !39
  %40 = load i64, i64* %39, align 8, !dbg !39
  %41 = add nsw i64 %36, %40, !dbg !39
  %42 = load double*, double** %.Z0966_324, align 8, !dbg !39
  call void @llvm.dbg.value(metadata double* %42, metadata !23, metadata !DIExpression()), !dbg !10
  %43 = bitcast double* %42 to i8*, !dbg !39
  %44 = getelementptr i8, i8* %43, i64 -8, !dbg !39
  %45 = bitcast i8* %44 to double*, !dbg !39
  %46 = getelementptr double, double* %45, i64 %41, !dbg !39
  store double %34, double* %46, align 8, !dbg !39
  %47 = load i32, i32* %i_314, align 4, !dbg !40
  call void @llvm.dbg.value(metadata i32 %47, metadata !38, metadata !DIExpression()), !dbg !10
  %48 = add nsw i32 %47, 1, !dbg !40
  store i32 %48, i32* %i_314, align 4, !dbg !40
  %49 = load i32, i32* %.dY0001_360, align 4, !dbg !40
  %50 = sub nsw i32 %49, 1, !dbg !40
  store i32 %50, i32* %.dY0001_360, align 4, !dbg !40
  %51 = load i32, i32* %.dY0001_360, align 4, !dbg !40
  %52 = icmp sgt i32 %51, 0, !dbg !40
  br i1 %52, label %L.LB1_358, label %L.LB1_359, !dbg !40

L.LB1_359:                                        ; preds = %L.LB1_358, %L.LB1_366
  %53 = bitcast double** %.Z0966_324 to i8*, !dbg !41
  %54 = bitcast %astruct.dt68* %.uplevelArgPack0001_387 to i8**, !dbg !41
  store i8* %53, i8** %54, align 8, !dbg !41
  %55 = bitcast double** %.Z0966_324 to i8*, !dbg !41
  %56 = bitcast %astruct.dt68* %.uplevelArgPack0001_387 to i8*, !dbg !41
  %57 = getelementptr i8, i8* %56, i64 8, !dbg !41
  %58 = bitcast i8* %57 to i8**, !dbg !41
  store i8* %55, i8** %58, align 8, !dbg !41
  %59 = bitcast i64* %z_b_0_316 to i8*, !dbg !41
  %60 = bitcast %astruct.dt68* %.uplevelArgPack0001_387 to i8*, !dbg !41
  %61 = getelementptr i8, i8* %60, i64 16, !dbg !41
  %62 = bitcast i8* %61 to i8**, !dbg !41
  store i8* %59, i8** %62, align 8, !dbg !41
  %63 = bitcast i64* %z_b_1_317 to i8*, !dbg !41
  %64 = bitcast %astruct.dt68* %.uplevelArgPack0001_387 to i8*, !dbg !41
  %65 = getelementptr i8, i8* %64, i64 24, !dbg !41
  %66 = bitcast i8* %65 to i8**, !dbg !41
  store i8* %63, i8** %66, align 8, !dbg !41
  %67 = bitcast i64* %z_e_62_320 to i8*, !dbg !41
  %68 = bitcast %astruct.dt68* %.uplevelArgPack0001_387 to i8*, !dbg !41
  %69 = getelementptr i8, i8* %68, i64 32, !dbg !41
  %70 = bitcast i8* %69 to i8**, !dbg !41
  store i8* %67, i8** %70, align 8, !dbg !41
  %71 = bitcast i64* %z_b_2_318 to i8*, !dbg !41
  %72 = bitcast %astruct.dt68* %.uplevelArgPack0001_387 to i8*, !dbg !41
  %73 = getelementptr i8, i8* %72, i64 40, !dbg !41
  %74 = bitcast i8* %73 to i8**, !dbg !41
  store i8* %71, i8** %74, align 8, !dbg !41
  %75 = bitcast i64* %z_b_3_319 to i8*, !dbg !41
  %76 = bitcast %astruct.dt68* %.uplevelArgPack0001_387 to i8*, !dbg !41
  %77 = getelementptr i8, i8* %76, i64 48, !dbg !41
  %78 = bitcast i8* %77 to i8**, !dbg !41
  store i8* %75, i8** %78, align 8, !dbg !41
  %79 = bitcast [16 x i64]* %"a$sd1_349" to i8*, !dbg !41
  %80 = bitcast %astruct.dt68* %.uplevelArgPack0001_387 to i8*, !dbg !41
  %81 = getelementptr i8, i8* %80, i64 56, !dbg !41
  %82 = bitcast i8* %81 to i8**, !dbg !41
  store i8* %79, i8** %82, align 8, !dbg !41
  %83 = bitcast %astruct.dt68* %.uplevelArgPack0001_387 to i64*, !dbg !41
  call void @__nv_MAIN__F1L27_1_(i32* %__gtid_MAIN__404, i64* null, i64* %83), !dbg !41
  call void (...) @_mp_bcs_nest(), !dbg !42
  %84 = bitcast i32* @.C337_MAIN_ to i8*, !dbg !42
  %85 = bitcast [57 x i8]* @.C335_MAIN_ to i8*, !dbg !42
  %86 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !42
  call void (i8*, i8*, i64, ...) %86(i8* %84, i8* %85, i64 57), !dbg !42
  %87 = bitcast i32* @.C338_MAIN_ to i8*, !dbg !42
  %88 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !42
  %89 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !42
  %90 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !42
  %91 = call i32 (i8*, i8*, i8*, i8*, ...) %90(i8* %87, i8* null, i8* %88, i8* %89), !dbg !42
  call void @llvm.dbg.declare(metadata i32* %z__io_340, metadata !43, metadata !DIExpression()), !dbg !10
  store i32 %91, i32* %z__io_340, align 4, !dbg !42
  %92 = bitcast [6 x i8]* @.C341_MAIN_ to i8*, !dbg !42
  %93 = bitcast i32 (...)* @f90io_sc_ch_ldw to i32 (i8*, i32, i64, ...)*, !dbg !42
  %94 = call i32 (i8*, i32, i64, ...) %93(i8* %92, i32 14, i64 6), !dbg !42
  store i32 %94, i32* %z__io_340, align 4, !dbg !42
  %95 = bitcast [16 x i64]* %"a$sd1_349" to i8*, !dbg !42
  %96 = getelementptr i8, i8* %95, i64 56, !dbg !42
  %97 = bitcast i8* %96 to i64*, !dbg !42
  %98 = load i64, i64* %97, align 8, !dbg !42
  %99 = load double*, double** %.Z0966_324, align 8, !dbg !42
  call void @llvm.dbg.value(metadata double* %99, metadata !23, metadata !DIExpression()), !dbg !10
  %100 = bitcast double* %99 to i8*, !dbg !42
  %101 = getelementptr i8, i8* %100, i64 392, !dbg !42
  %102 = bitcast i8* %101 to double*, !dbg !42
  %103 = getelementptr double, double* %102, i64 %98, !dbg !42
  %104 = load double, double* %103, align 8, !dbg !42
  %105 = bitcast i32 (...)* @f90io_sc_d_ldw to i32 (double, i32, ...)*, !dbg !42
  %106 = call i32 (double, i32, ...) %105(double %104, i32 28), !dbg !42
  store i32 %106, i32* %z__io_340, align 4, !dbg !42
  %107 = call i32 (...) @f90io_ldw_end(), !dbg !42
  store i32 %107, i32* %z__io_340, align 4, !dbg !42
  call void (...) @_mp_ecs_nest(), !dbg !42
  %108 = load double*, double** %.Z0966_324, align 8, !dbg !44
  call void @llvm.dbg.value(metadata double* %108, metadata !23, metadata !DIExpression()), !dbg !10
  %109 = bitcast double* %108 to i8*, !dbg !44
  %110 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !44
  %111 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !44
  call void (i8*, i8*, i8*, i8*, i64, ...) %111(i8* null, i8* %109, i8* %110, i8* null, i64 0), !dbg !44
  %112 = bitcast double** %.Z0966_324 to i8**, !dbg !44
  store i8* null, i8** %112, align 8, !dbg !44
  %113 = bitcast [16 x i64]* %"a$sd1_349" to i64*, !dbg !44
  store i64 0, i64* %113, align 8, !dbg !44
  ret void, !dbg !21
}

define internal void @__nv_MAIN__F1L27_1_(i32* %__nv_MAIN__F1L27_1Arg0, i64* %__nv_MAIN__F1L27_1Arg1, i64* %__nv_MAIN__F1L27_1Arg2) #0 !dbg !45 {
L.entry:
  %__gtid___nv_MAIN__F1L27_1__435 = alloca i32, align 4
  %.uplevelArgPack0002_431 = alloca %astruct.dt113, align 16
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L27_1Arg0, metadata !48, metadata !DIExpression()), !dbg !49
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L27_1Arg1, metadata !50, metadata !DIExpression()), !dbg !49
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L27_1Arg2, metadata !51, metadata !DIExpression()), !dbg !49
  call void @llvm.dbg.value(metadata i32 8, metadata !52, metadata !DIExpression()), !dbg !49
  call void @llvm.dbg.value(metadata i32 1, metadata !53, metadata !DIExpression()), !dbg !49
  call void @llvm.dbg.value(metadata i32 2, metadata !54, metadata !DIExpression()), !dbg !49
  call void @llvm.dbg.value(metadata i32 0, metadata !55, metadata !DIExpression()), !dbg !49
  call void @llvm.dbg.value(metadata i32 1, metadata !56, metadata !DIExpression()), !dbg !49
  call void @llvm.dbg.value(metadata i32 2, metadata !57, metadata !DIExpression()), !dbg !49
  call void @llvm.dbg.value(metadata i32 0, metadata !58, metadata !DIExpression()), !dbg !49
  call void @llvm.dbg.value(metadata i32 1, metadata !59, metadata !DIExpression()), !dbg !49
  call void @llvm.dbg.value(metadata i32 2, metadata !60, metadata !DIExpression()), !dbg !49
  call void @llvm.dbg.value(metadata i32 8, metadata !61, metadata !DIExpression()), !dbg !49
  call void @llvm.dbg.value(metadata i32 8, metadata !62, metadata !DIExpression()), !dbg !49
  %0 = load i32, i32* %__nv_MAIN__F1L27_1Arg0, align 4, !dbg !63
  store i32 %0, i32* %__gtid___nv_MAIN__F1L27_1__435, align 4, !dbg !63
  br label %L.LB2_426

L.LB2_426:                                        ; preds = %L.entry
  br label %L.LB2_327

L.LB2_327:                                        ; preds = %L.LB2_426
  %1 = load i64, i64* %__nv_MAIN__F1L27_1Arg2, align 8, !dbg !64
  %2 = bitcast %astruct.dt113* %.uplevelArgPack0002_431 to i64*, !dbg !64
  store i64 %1, i64* %2, align 8, !dbg !64
  %3 = bitcast i64* %__nv_MAIN__F1L27_1Arg2 to i8*, !dbg !63
  %4 = getelementptr i8, i8* %3, i64 8, !dbg !63
  %5 = bitcast i8* %4 to i64*, !dbg !63
  %6 = load i64, i64* %5, align 8, !dbg !63
  %7 = bitcast %astruct.dt113* %.uplevelArgPack0002_431 to i8*, !dbg !63
  %8 = getelementptr i8, i8* %7, i64 8, !dbg !63
  %9 = bitcast i8* %8 to i64*, !dbg !63
  store i64 %6, i64* %9, align 8, !dbg !63
  %10 = bitcast i64* %__nv_MAIN__F1L27_1Arg2 to i8*, !dbg !63
  %11 = getelementptr i8, i8* %10, i64 16, !dbg !63
  %12 = bitcast i8* %11 to i64*, !dbg !63
  %13 = load i64, i64* %12, align 8, !dbg !63
  %14 = bitcast %astruct.dt113* %.uplevelArgPack0002_431 to i8*, !dbg !63
  %15 = getelementptr i8, i8* %14, i64 16, !dbg !63
  %16 = bitcast i8* %15 to i64*, !dbg !63
  store i64 %13, i64* %16, align 8, !dbg !63
  %17 = bitcast i64* %__nv_MAIN__F1L27_1Arg2 to i8*, !dbg !63
  %18 = getelementptr i8, i8* %17, i64 24, !dbg !63
  %19 = bitcast i8* %18 to i64*, !dbg !63
  %20 = load i64, i64* %19, align 8, !dbg !63
  %21 = bitcast %astruct.dt113* %.uplevelArgPack0002_431 to i8*, !dbg !63
  %22 = getelementptr i8, i8* %21, i64 24, !dbg !63
  %23 = bitcast i8* %22 to i64*, !dbg !63
  store i64 %20, i64* %23, align 8, !dbg !63
  %24 = bitcast i64* %__nv_MAIN__F1L27_1Arg2 to i8*, !dbg !63
  %25 = getelementptr i8, i8* %24, i64 32, !dbg !63
  %26 = bitcast i8* %25 to i64*, !dbg !63
  %27 = load i64, i64* %26, align 8, !dbg !63
  %28 = bitcast %astruct.dt113* %.uplevelArgPack0002_431 to i8*, !dbg !63
  %29 = getelementptr i8, i8* %28, i64 32, !dbg !63
  %30 = bitcast i8* %29 to i64*, !dbg !63
  store i64 %27, i64* %30, align 8, !dbg !63
  %31 = bitcast i64* %__nv_MAIN__F1L27_1Arg2 to i8*, !dbg !63
  %32 = getelementptr i8, i8* %31, i64 40, !dbg !63
  %33 = bitcast i8* %32 to i64*, !dbg !63
  %34 = load i64, i64* %33, align 8, !dbg !63
  %35 = bitcast %astruct.dt113* %.uplevelArgPack0002_431 to i8*, !dbg !63
  %36 = getelementptr i8, i8* %35, i64 40, !dbg !63
  %37 = bitcast i8* %36 to i64*, !dbg !63
  store i64 %34, i64* %37, align 8, !dbg !63
  %38 = bitcast i64* %__nv_MAIN__F1L27_1Arg2 to i8*, !dbg !63
  %39 = getelementptr i8, i8* %38, i64 48, !dbg !63
  %40 = bitcast i8* %39 to i64*, !dbg !63
  %41 = load i64, i64* %40, align 8, !dbg !63
  %42 = bitcast %astruct.dt113* %.uplevelArgPack0002_431 to i8*, !dbg !63
  %43 = getelementptr i8, i8* %42, i64 48, !dbg !63
  %44 = bitcast i8* %43 to i64*, !dbg !63
  store i64 %41, i64* %44, align 8, !dbg !63
  %45 = bitcast i64* %__nv_MAIN__F1L27_1Arg2 to i8*, !dbg !63
  %46 = getelementptr i8, i8* %45, i64 56, !dbg !63
  %47 = bitcast i8* %46 to i64*, !dbg !63
  %48 = load i64, i64* %47, align 8, !dbg !63
  %49 = bitcast %astruct.dt113* %.uplevelArgPack0002_431 to i8*, !dbg !63
  %50 = getelementptr i8, i8* %49, i64 56, !dbg !63
  %51 = bitcast i8* %50 to i64*, !dbg !63
  store i64 %48, i64* %51, align 8, !dbg !63
  %52 = load i32, i32* %__gtid___nv_MAIN__F1L27_1__435, align 4, !dbg !64
  call void @__kmpc_push_num_teams(i64* null, i32 %52, i32 2, i32 0), !dbg !64
  %53 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN_F1L28_2_ to i64*, !dbg !64
  %54 = bitcast %astruct.dt113* %.uplevelArgPack0002_431 to i64*, !dbg !64
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_teams(i64* null, i32 1, i64* %53, i64* %54), !dbg !64
  br label %L.LB2_333

L.LB2_333:                                        ; preds = %L.LB2_327
  ret void, !dbg !63
}

define internal void @__nv_MAIN_F1L28_2_(i32* %__nv_MAIN_F1L28_2Arg0, i64* %__nv_MAIN_F1L28_2Arg1, i64* %__nv_MAIN_F1L28_2Arg2) #0 !dbg !65 {
L.entry:
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L28_2Arg0, metadata !66, metadata !DIExpression()), !dbg !67
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L28_2Arg1, metadata !68, metadata !DIExpression()), !dbg !67
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L28_2Arg2, metadata !69, metadata !DIExpression()), !dbg !67
  call void @llvm.dbg.value(metadata i32 8, metadata !70, metadata !DIExpression()), !dbg !67
  call void @llvm.dbg.value(metadata i32 1, metadata !71, metadata !DIExpression()), !dbg !67
  call void @llvm.dbg.value(metadata i32 2, metadata !72, metadata !DIExpression()), !dbg !67
  call void @llvm.dbg.value(metadata i32 0, metadata !73, metadata !DIExpression()), !dbg !67
  call void @llvm.dbg.value(metadata i32 1, metadata !74, metadata !DIExpression()), !dbg !67
  call void @llvm.dbg.value(metadata i32 2, metadata !75, metadata !DIExpression()), !dbg !67
  call void @llvm.dbg.value(metadata i32 0, metadata !76, metadata !DIExpression()), !dbg !67
  call void @llvm.dbg.value(metadata i32 1, metadata !77, metadata !DIExpression()), !dbg !67
  call void @llvm.dbg.value(metadata i32 2, metadata !78, metadata !DIExpression()), !dbg !67
  call void @llvm.dbg.value(metadata i32 8, metadata !79, metadata !DIExpression()), !dbg !67
  call void @llvm.dbg.value(metadata i32 8, metadata !80, metadata !DIExpression()), !dbg !67
  br label %L.LB4_469

L.LB4_469:                                        ; preds = %L.entry
  br label %L.LB4_330

L.LB4_330:                                        ; preds = %L.LB4_469
  %0 = bitcast i64* %__nv_MAIN_F1L28_2Arg2 to i8*, !dbg !81
  %1 = getelementptr i8, i8* %0, i64 56, !dbg !81
  %2 = bitcast i8* %1 to i8**, !dbg !81
  %3 = load i8*, i8** %2, align 8, !dbg !81
  %4 = getelementptr i8, i8* %3, i64 56, !dbg !81
  %5 = bitcast i8* %4 to i64*, !dbg !81
  %6 = load i64, i64* %5, align 8, !dbg !81
  %7 = bitcast i64* %__nv_MAIN_F1L28_2Arg2 to i8*, !dbg !81
  %8 = getelementptr i8, i8* %7, i64 8, !dbg !81
  %9 = bitcast i8* %8 to i8***, !dbg !81
  %10 = load i8**, i8*** %9, align 8, !dbg !81
  %11 = load i8*, i8** %10, align 8, !dbg !81
  %12 = getelementptr i8, i8* %11, i64 392, !dbg !81
  %13 = bitcast i8* %12 to double*, !dbg !81
  %14 = getelementptr double, double* %13, i64 %6, !dbg !81
  %15 = load double, double* %14, align 8, !dbg !81
  %16 = bitcast i64* %__nv_MAIN_F1L28_2Arg2 to i8*, !dbg !81
  %17 = getelementptr i8, i8* %16, i64 56, !dbg !81
  %18 = bitcast i8* %17 to i8**, !dbg !81
  %19 = load i8*, i8** %18, align 8, !dbg !81
  %20 = getelementptr i8, i8* %19, i64 56, !dbg !81
  %21 = bitcast i8* %20 to i64*, !dbg !81
  %22 = load i64, i64* %21, align 8, !dbg !81
  %23 = bitcast i64* %__nv_MAIN_F1L28_2Arg2 to i8*, !dbg !81
  %24 = getelementptr i8, i8* %23, i64 8, !dbg !81
  %25 = bitcast i8* %24 to i8***, !dbg !81
  %26 = load i8**, i8*** %25, align 8, !dbg !81
  %27 = load i8*, i8** %26, align 8, !dbg !81
  %28 = getelementptr i8, i8* %27, i64 392, !dbg !81
  %29 = bitcast i8* %28 to double*, !dbg !81
  %30 = getelementptr double, double* %29, i64 %22, !dbg !81
  %31 = load double, double* %30, align 8, !dbg !81
  %32 = fadd fast double %15, %31, !dbg !81
  %33 = bitcast i64* %__nv_MAIN_F1L28_2Arg2 to i8*, !dbg !81
  %34 = getelementptr i8, i8* %33, i64 56, !dbg !81
  %35 = bitcast i8* %34 to i8**, !dbg !81
  %36 = load i8*, i8** %35, align 8, !dbg !81
  %37 = getelementptr i8, i8* %36, i64 56, !dbg !81
  %38 = bitcast i8* %37 to i64*, !dbg !81
  %39 = load i64, i64* %38, align 8, !dbg !81
  %40 = bitcast i64* %__nv_MAIN_F1L28_2Arg2 to i8*, !dbg !81
  %41 = getelementptr i8, i8* %40, i64 8, !dbg !81
  %42 = bitcast i8* %41 to i8***, !dbg !81
  %43 = load i8**, i8*** %42, align 8, !dbg !81
  %44 = load i8*, i8** %43, align 8, !dbg !81
  %45 = getelementptr i8, i8* %44, i64 392, !dbg !81
  %46 = bitcast i8* %45 to double*, !dbg !81
  %47 = getelementptr double, double* %46, i64 %39, !dbg !81
  store double %32, double* %47, align 8, !dbg !81
  br label %L.LB4_332

L.LB4_332:                                        ; preds = %L.LB4_330
  ret void, !dbg !82
}

declare void @__kmpc_fork_teams(i64*, i32, i64*, i64*, ...) #0

declare void @__kmpc_push_num_teams(i64*, i32, i32, i32) #0

declare void @f90_dealloc03a_i8(...) #0

declare void @_mp_ecs_nest(...) #0

declare signext i32 @f90io_ldw_end(...) #0

declare signext i32 @f90io_sc_d_ldw(...) #0

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

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { "no-frame-pointer-elim-non-leaf" }
attributes #1 = { nounwind readnone speculatable }

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !4, globals: !4, imports: !4)
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB116-target-teams-orig-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb116_target_teams_orig_yes", scope: !2, file: !3, line: 12, type: !6, scopeLine: 12, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!6 = !DISubroutineType(cc: DW_CC_program, types: !7)
!7 = !{null}
!8 = !DILocalVariable(name: "omp_nest_lock_kind", scope: !5, file: !3, type: !9)
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 0, scope: !5)
!11 = !DILocalVariable(name: "omp_sched_static", scope: !5, file: !3, type: !9)
!12 = !DILocalVariable(name: "omp_sched_dynamic", scope: !5, file: !3, type: !9)
!13 = !DILocalVariable(name: "omp_proc_bind_false", scope: !5, file: !3, type: !9)
!14 = !DILocalVariable(name: "omp_proc_bind_true", scope: !5, file: !3, type: !9)
!15 = !DILocalVariable(name: "omp_proc_bind_master", scope: !5, file: !3, type: !9)
!16 = !DILocalVariable(name: "omp_lock_hint_none", scope: !5, file: !3, type: !9)
!17 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !5, file: !3, type: !9)
!18 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !5, file: !3, type: !9)
!19 = !DILocalVariable(name: "omp_lock_hint_speculative", scope: !5, file: !3, type: !9)
!20 = !DILocalVariable(name: "dp", scope: !5, file: !3, type: !9)
!21 = !DILocation(line: 36, column: 1, scope: !5)
!22 = !DILocation(line: 12, column: 1, scope: !5)
!23 = !DILocalVariable(name: "a", scope: !5, file: !3, type: !24)
!24 = !DICompositeType(tag: DW_TAG_array_type, baseType: !25, size: 64, align: 64, elements: !26)
!25 = !DIBasicType(name: "double precision", size: 64, align: 64, encoding: DW_ATE_float)
!26 = !{!27}
!27 = !DISubrange(count: 0, lowerBound: 1)
!28 = !DILocalVariable(scope: !5, file: !3, type: !29, flags: DIFlagArtificial)
!29 = !DICompositeType(tag: DW_TAG_array_type, baseType: !30, size: 1024, align: 64, elements: !31)
!30 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!31 = !{!32}
!32 = !DISubrange(count: 16, lowerBound: 1)
!33 = !DILocalVariable(name: "len", scope: !5, file: !3, type: !9)
!34 = !DILocation(line: 20, column: 1, scope: !5)
!35 = !DILocalVariable(scope: !5, file: !3, type: !30, flags: DIFlagArtificial)
!36 = !DILocation(line: 21, column: 1, scope: !5)
!37 = !DILocation(line: 23, column: 1, scope: !5)
!38 = !DILocalVariable(name: "i", scope: !5, file: !3, type: !9)
!39 = !DILocation(line: 24, column: 1, scope: !5)
!40 = !DILocation(line: 25, column: 1, scope: !5)
!41 = !DILocation(line: 31, column: 1, scope: !5)
!42 = !DILocation(line: 33, column: 1, scope: !5)
!43 = !DILocalVariable(scope: !5, file: !3, type: !9, flags: DIFlagArtificial)
!44 = !DILocation(line: 35, column: 1, scope: !5)
!45 = distinct !DISubprogram(name: "__nv_MAIN__F1L27_1", scope: !2, file: !3, line: 27, type: !46, scopeLine: 27, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!46 = !DISubroutineType(types: !47)
!47 = !{null, !9, !30, !30}
!48 = !DILocalVariable(name: "__nv_MAIN__F1L27_1Arg0", arg: 1, scope: !45, file: !3, type: !9)
!49 = !DILocation(line: 0, scope: !45)
!50 = !DILocalVariable(name: "__nv_MAIN__F1L27_1Arg1", arg: 2, scope: !45, file: !3, type: !30)
!51 = !DILocalVariable(name: "__nv_MAIN__F1L27_1Arg2", arg: 3, scope: !45, file: !3, type: !30)
!52 = !DILocalVariable(name: "omp_nest_lock_kind", scope: !45, file: !3, type: !9)
!53 = !DILocalVariable(name: "omp_sched_static", scope: !45, file: !3, type: !9)
!54 = !DILocalVariable(name: "omp_sched_dynamic", scope: !45, file: !3, type: !9)
!55 = !DILocalVariable(name: "omp_proc_bind_false", scope: !45, file: !3, type: !9)
!56 = !DILocalVariable(name: "omp_proc_bind_true", scope: !45, file: !3, type: !9)
!57 = !DILocalVariable(name: "omp_proc_bind_master", scope: !45, file: !3, type: !9)
!58 = !DILocalVariable(name: "omp_lock_hint_none", scope: !45, file: !3, type: !9)
!59 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !45, file: !3, type: !9)
!60 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !45, file: !3, type: !9)
!61 = !DILocalVariable(name: "omp_lock_hint_speculative", scope: !45, file: !3, type: !9)
!62 = !DILocalVariable(name: "dp", scope: !45, file: !3, type: !9)
!63 = !DILocation(line: 31, column: 1, scope: !45)
!64 = !DILocation(line: 28, column: 1, scope: !45)
!65 = distinct !DISubprogram(name: "__nv_MAIN_F1L28_2", scope: !2, file: !3, line: 28, type: !46, scopeLine: 28, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!66 = !DILocalVariable(name: "__nv_MAIN_F1L28_2Arg0", arg: 1, scope: !65, file: !3, type: !9)
!67 = !DILocation(line: 0, scope: !65)
!68 = !DILocalVariable(name: "__nv_MAIN_F1L28_2Arg1", arg: 2, scope: !65, file: !3, type: !30)
!69 = !DILocalVariable(name: "__nv_MAIN_F1L28_2Arg2", arg: 3, scope: !65, file: !3, type: !30)
!70 = !DILocalVariable(name: "omp_nest_lock_kind", scope: !65, file: !3, type: !9)
!71 = !DILocalVariable(name: "omp_sched_static", scope: !65, file: !3, type: !9)
!72 = !DILocalVariable(name: "omp_sched_dynamic", scope: !65, file: !3, type: !9)
!73 = !DILocalVariable(name: "omp_proc_bind_false", scope: !65, file: !3, type: !9)
!74 = !DILocalVariable(name: "omp_proc_bind_true", scope: !65, file: !3, type: !9)
!75 = !DILocalVariable(name: "omp_proc_bind_master", scope: !65, file: !3, type: !9)
!76 = !DILocalVariable(name: "omp_lock_hint_none", scope: !65, file: !3, type: !9)
!77 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !65, file: !3, type: !9)
!78 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !65, file: !3, type: !9)
!79 = !DILocalVariable(name: "omp_lock_hint_speculative", scope: !65, file: !3, type: !9)
!80 = !DILocalVariable(name: "dp", scope: !65, file: !3, type: !9)
!81 = !DILocation(line: 29, column: 1, scope: !65)
!82 = !DILocation(line: 30, column: 1, scope: !65)

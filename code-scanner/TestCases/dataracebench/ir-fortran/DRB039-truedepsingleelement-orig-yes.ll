; ModuleID = '/tmp/DRB039-truedepsingleelement-orig-yes-e2fe4d.ll'
source_filename = "/tmp/DRB039-truedepsingleelement-orig-yes-e2fe4d.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.STATICS1 = type <{ [44 x i8] }>
%astruct.dt68 = type <{ i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8* }>

@.STATICS1 = internal global %struct.STATICS1 <{ [44 x i8] c"\FB\FF\FF\FF\08\00\00\00a(500) =\EB\FF\FF\FF\00\00\00\00\03\00\00\00\00\00\00\00\01\00\00\00\FF\FF\FF\FF\00\00\00\00" }>, align 16
@.C334_MAIN_ = internal constant i64 500
@.C331_MAIN_ = internal constant i32 6
@.C328_MAIN_ = internal constant [65 x i8] c"micro-benchmarks-fortran/DRB039-truedepsingleelement-orig-yes.f95"
@.C310_MAIN_ = internal constant i32 28
@.C285_MAIN_ = internal constant i32 1
@.C300_MAIN_ = internal constant i32 2
@.C309_MAIN_ = internal constant i32 25
@.C344_MAIN_ = internal constant i64 4
@.C343_MAIN_ = internal constant i64 25
@.C318_MAIN_ = internal constant i32 1000
@.C286_MAIN_ = internal constant i64 1
@.C284_MAIN_ = internal constant i64 0
@.C283_MAIN_ = internal constant i32 0
@.C286___nv_MAIN__F1L22_1 = internal constant i64 1
@.C285___nv_MAIN__F1L22_1 = internal constant i32 1
@.C283___nv_MAIN__F1L22_1 = internal constant i32 0

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__408 = alloca i32, align 4
  %.Z0965_320 = alloca i32*, align 8
  %"a$sd1_342" = alloca [16 x i64], align 8
  %len_319 = alloca i32, align 4
  %z_b_0_312 = alloca i64, align 8
  %z_b_1_313 = alloca i64, align 8
  %z_e_60_316 = alloca i64, align 8
  %z_b_2_314 = alloca i64, align 8
  %z_b_3_315 = alloca i64, align 8
  %.uplevelArgPack0001_387 = alloca %astruct.dt68, align 16
  %z__io_333 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 2, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 2, metadata !14, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !15, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !16, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 2, metadata !17, metadata !DIExpression()), !dbg !10
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !18
  store i32 %0, i32* %__gtid_MAIN__408, align 4, !dbg !18
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !19
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !19
  call void (i8*, ...) %2(i8* %1), !dbg !19
  call void @llvm.dbg.declare(metadata i32** %.Z0965_320, metadata !20, metadata !DIExpression(DW_OP_deref)), !dbg !10
  %3 = bitcast i32** %.Z0965_320 to i8**, !dbg !19
  store i8* null, i8** %3, align 8, !dbg !19
  call void @llvm.dbg.declare(metadata [16 x i64]* %"a$sd1_342", metadata !24, metadata !DIExpression()), !dbg !10
  %4 = bitcast [16 x i64]* %"a$sd1_342" to i64*, !dbg !19
  store i64 0, i64* %4, align 8, !dbg !19
  br label %L.LB1_370

L.LB1_370:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %len_319, metadata !29, metadata !DIExpression()), !dbg !10
  store i32 1000, i32* %len_319, align 4, !dbg !30
  call void @llvm.dbg.declare(metadata i64* %z_b_0_312, metadata !31, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_0_312, align 8, !dbg !32
  %5 = load i32, i32* %len_319, align 4, !dbg !32
  call void @llvm.dbg.value(metadata i32 %5, metadata !29, metadata !DIExpression()), !dbg !10
  %6 = sext i32 %5 to i64, !dbg !32
  call void @llvm.dbg.declare(metadata i64* %z_b_1_313, metadata !31, metadata !DIExpression()), !dbg !10
  store i64 %6, i64* %z_b_1_313, align 8, !dbg !32
  %7 = load i64, i64* %z_b_1_313, align 8, !dbg !32
  call void @llvm.dbg.value(metadata i64 %7, metadata !31, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_60_316, metadata !31, metadata !DIExpression()), !dbg !10
  store i64 %7, i64* %z_e_60_316, align 8, !dbg !32
  %8 = bitcast [16 x i64]* %"a$sd1_342" to i8*, !dbg !32
  %9 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !32
  %10 = bitcast i64* @.C343_MAIN_ to i8*, !dbg !32
  %11 = bitcast i64* @.C344_MAIN_ to i8*, !dbg !32
  %12 = bitcast i64* %z_b_0_312 to i8*, !dbg !32
  %13 = bitcast i64* %z_b_1_313 to i8*, !dbg !32
  %14 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !32
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %14(i8* %8, i8* %9, i8* %10, i8* %11, i8* %12, i8* %13), !dbg !32
  %15 = bitcast [16 x i64]* %"a$sd1_342" to i8*, !dbg !32
  %16 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !32
  call void (i8*, i32, ...) %16(i8* %15, i32 25), !dbg !32
  %17 = load i64, i64* %z_b_1_313, align 8, !dbg !32
  call void @llvm.dbg.value(metadata i64 %17, metadata !31, metadata !DIExpression()), !dbg !10
  %18 = load i64, i64* %z_b_0_312, align 8, !dbg !32
  call void @llvm.dbg.value(metadata i64 %18, metadata !31, metadata !DIExpression()), !dbg !10
  %19 = sub nsw i64 %18, 1, !dbg !32
  %20 = sub nsw i64 %17, %19, !dbg !32
  call void @llvm.dbg.declare(metadata i64* %z_b_2_314, metadata !31, metadata !DIExpression()), !dbg !10
  store i64 %20, i64* %z_b_2_314, align 8, !dbg !32
  %21 = load i64, i64* %z_b_0_312, align 8, !dbg !32
  call void @llvm.dbg.value(metadata i64 %21, metadata !31, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_b_3_315, metadata !31, metadata !DIExpression()), !dbg !10
  store i64 %21, i64* %z_b_3_315, align 8, !dbg !32
  %22 = bitcast i64* %z_b_2_314 to i8*, !dbg !32
  %23 = bitcast i64* @.C343_MAIN_ to i8*, !dbg !32
  %24 = bitcast i64* @.C344_MAIN_ to i8*, !dbg !32
  %25 = bitcast i32** %.Z0965_320 to i8*, !dbg !32
  %26 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !32
  %27 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !32
  %28 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !32
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %28(i8* %22, i8* %23, i8* %24, i8* null, i8* %25, i8* null, i8* %26, i8* %27, i8* null, i64 0), !dbg !32
  %29 = bitcast [16 x i64]* %"a$sd1_342" to i8*, !dbg !33
  %30 = getelementptr i8, i8* %29, i64 56, !dbg !33
  %31 = bitcast i8* %30 to i64*, !dbg !33
  %32 = load i64, i64* %31, align 8, !dbg !33
  %33 = load i32*, i32** %.Z0965_320, align 8, !dbg !33
  call void @llvm.dbg.value(metadata i32* %33, metadata !20, metadata !DIExpression()), !dbg !10
  %34 = getelementptr i32, i32* %33, i64 %32, !dbg !33
  store i32 2, i32* %34, align 4, !dbg !33
  %35 = bitcast i32* %len_319 to i8*, !dbg !34
  %36 = bitcast %astruct.dt68* %.uplevelArgPack0001_387 to i8**, !dbg !34
  store i8* %35, i8** %36, align 8, !dbg !34
  %37 = bitcast i32** %.Z0965_320 to i8*, !dbg !34
  %38 = bitcast %astruct.dt68* %.uplevelArgPack0001_387 to i8*, !dbg !34
  %39 = getelementptr i8, i8* %38, i64 8, !dbg !34
  %40 = bitcast i8* %39 to i8**, !dbg !34
  store i8* %37, i8** %40, align 8, !dbg !34
  %41 = bitcast i32** %.Z0965_320 to i8*, !dbg !34
  %42 = bitcast %astruct.dt68* %.uplevelArgPack0001_387 to i8*, !dbg !34
  %43 = getelementptr i8, i8* %42, i64 16, !dbg !34
  %44 = bitcast i8* %43 to i8**, !dbg !34
  store i8* %41, i8** %44, align 8, !dbg !34
  %45 = bitcast i64* %z_b_0_312 to i8*, !dbg !34
  %46 = bitcast %astruct.dt68* %.uplevelArgPack0001_387 to i8*, !dbg !34
  %47 = getelementptr i8, i8* %46, i64 24, !dbg !34
  %48 = bitcast i8* %47 to i8**, !dbg !34
  store i8* %45, i8** %48, align 8, !dbg !34
  %49 = bitcast i64* %z_b_1_313 to i8*, !dbg !34
  %50 = bitcast %astruct.dt68* %.uplevelArgPack0001_387 to i8*, !dbg !34
  %51 = getelementptr i8, i8* %50, i64 32, !dbg !34
  %52 = bitcast i8* %51 to i8**, !dbg !34
  store i8* %49, i8** %52, align 8, !dbg !34
  %53 = bitcast i64* %z_e_60_316 to i8*, !dbg !34
  %54 = bitcast %astruct.dt68* %.uplevelArgPack0001_387 to i8*, !dbg !34
  %55 = getelementptr i8, i8* %54, i64 40, !dbg !34
  %56 = bitcast i8* %55 to i8**, !dbg !34
  store i8* %53, i8** %56, align 8, !dbg !34
  %57 = bitcast i64* %z_b_2_314 to i8*, !dbg !34
  %58 = bitcast %astruct.dt68* %.uplevelArgPack0001_387 to i8*, !dbg !34
  %59 = getelementptr i8, i8* %58, i64 48, !dbg !34
  %60 = bitcast i8* %59 to i8**, !dbg !34
  store i8* %57, i8** %60, align 8, !dbg !34
  %61 = bitcast i64* %z_b_3_315 to i8*, !dbg !34
  %62 = bitcast %astruct.dt68* %.uplevelArgPack0001_387 to i8*, !dbg !34
  %63 = getelementptr i8, i8* %62, i64 56, !dbg !34
  %64 = bitcast i8* %63 to i8**, !dbg !34
  store i8* %61, i8** %64, align 8, !dbg !34
  %65 = bitcast [16 x i64]* %"a$sd1_342" to i8*, !dbg !34
  %66 = bitcast %astruct.dt68* %.uplevelArgPack0001_387 to i8*, !dbg !34
  %67 = getelementptr i8, i8* %66, i64 64, !dbg !34
  %68 = bitcast i8* %67 to i8**, !dbg !34
  store i8* %65, i8** %68, align 8, !dbg !34
  br label %L.LB1_406, !dbg !34

L.LB1_406:                                        ; preds = %L.LB1_370
  %69 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L22_1_ to i64*, !dbg !34
  %70 = bitcast %astruct.dt68* %.uplevelArgPack0001_387 to i64*, !dbg !34
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %69, i64* %70), !dbg !34
  call void (...) @_mp_bcs_nest(), !dbg !35
  %71 = bitcast i32* @.C310_MAIN_ to i8*, !dbg !35
  %72 = bitcast [65 x i8]* @.C328_MAIN_ to i8*, !dbg !35
  %73 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !35
  call void (i8*, i8*, i64, ...) %73(i8* %71, i8* %72, i64 65), !dbg !35
  %74 = bitcast i32* @.C331_MAIN_ to i8*, !dbg !35
  %75 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !35
  %76 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !35
  %77 = bitcast %struct.STATICS1* @.STATICS1 to i8*, !dbg !35
  %78 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !35
  %79 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %78(i8* %74, i8* null, i8* %75, i8* %76, i8* %77, i8* null, i64 0), !dbg !35
  call void @llvm.dbg.declare(metadata i32* %z__io_333, metadata !36, metadata !DIExpression()), !dbg !10
  store i32 %79, i32* %z__io_333, align 4, !dbg !35
  %80 = bitcast [16 x i64]* %"a$sd1_342" to i8*, !dbg !35
  %81 = getelementptr i8, i8* %80, i64 56, !dbg !35
  %82 = bitcast i8* %81 to i64*, !dbg !35
  %83 = load i64, i64* %82, align 8, !dbg !35
  %84 = load i32*, i32** %.Z0965_320, align 8, !dbg !35
  call void @llvm.dbg.value(metadata i32* %84, metadata !20, metadata !DIExpression()), !dbg !10
  %85 = bitcast i32* %84 to i8*, !dbg !35
  %86 = getelementptr i8, i8* %85, i64 1996, !dbg !35
  %87 = bitcast i8* %86 to i32*, !dbg !35
  %88 = getelementptr i32, i32* %87, i64 %83, !dbg !35
  %89 = load i32, i32* %88, align 4, !dbg !35
  %90 = bitcast i32 (...)* @f90io_sc_i_fmt_write to i32 (i32, i32, ...)*, !dbg !35
  %91 = call i32 (i32, i32, ...) %90(i32 %89, i32 25), !dbg !35
  store i32 %91, i32* %z__io_333, align 4, !dbg !35
  %92 = call i32 (...) @f90io_fmtw_end(), !dbg !35
  store i32 %92, i32* %z__io_333, align 4, !dbg !35
  call void (...) @_mp_ecs_nest(), !dbg !35
  %93 = load i32*, i32** %.Z0965_320, align 8, !dbg !37
  call void @llvm.dbg.value(metadata i32* %93, metadata !20, metadata !DIExpression()), !dbg !10
  %94 = bitcast i32* %93 to i8*, !dbg !37
  %95 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !37
  %96 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !37
  call void (i8*, i8*, i8*, i8*, i64, ...) %96(i8* null, i8* %94, i8* %95, i8* null, i64 0), !dbg !37
  %97 = bitcast i32** %.Z0965_320 to i8**, !dbg !37
  store i8* null, i8** %97, align 8, !dbg !37
  %98 = bitcast [16 x i64]* %"a$sd1_342" to i64*, !dbg !37
  store i64 0, i64* %98, align 8, !dbg !37
  ret void, !dbg !18
}

define internal void @__nv_MAIN__F1L22_1_(i32* %__nv_MAIN__F1L22_1Arg0, i64* %__nv_MAIN__F1L22_1Arg1, i64* %__nv_MAIN__F1L22_1Arg2) #0 !dbg !38 {
L.entry:
  %__gtid___nv_MAIN__F1L22_1__456 = alloca i32, align 4
  %.i0000p_325 = alloca i32, align 4
  %i_324 = alloca i32, align 4
  %.du0001p_354 = alloca i32, align 4
  %.de0001p_355 = alloca i32, align 4
  %.di0001p_356 = alloca i32, align 4
  %.ds0001p_357 = alloca i32, align 4
  %.dl0001p_359 = alloca i32, align 4
  %.dl0001p.copy_450 = alloca i32, align 4
  %.de0001p.copy_451 = alloca i32, align 4
  %.ds0001p.copy_452 = alloca i32, align 4
  %.dX0001p_358 = alloca i32, align 4
  %.dY0001p_353 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L22_1Arg0, metadata !41, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L22_1Arg1, metadata !43, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L22_1Arg2, metadata !44, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.value(metadata i32 1, metadata !45, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.value(metadata i32 2, metadata !46, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.value(metadata i32 0, metadata !47, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.value(metadata i32 1, metadata !48, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.value(metadata i32 2, metadata !49, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.value(metadata i32 0, metadata !50, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.value(metadata i32 1, metadata !51, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.value(metadata i32 2, metadata !52, metadata !DIExpression()), !dbg !42
  %0 = load i32, i32* %__nv_MAIN__F1L22_1Arg0, align 4, !dbg !53
  store i32 %0, i32* %__gtid___nv_MAIN__F1L22_1__456, align 4, !dbg !53
  br label %L.LB2_441

L.LB2_441:                                        ; preds = %L.entry
  br label %L.LB2_323

L.LB2_323:                                        ; preds = %L.LB2_441
  store i32 0, i32* %.i0000p_325, align 4, !dbg !54
  call void @llvm.dbg.declare(metadata i32* %i_324, metadata !55, metadata !DIExpression()), !dbg !53
  store i32 1, i32* %i_324, align 4, !dbg !54
  %1 = bitcast i64* %__nv_MAIN__F1L22_1Arg2 to i32**, !dbg !54
  %2 = load i32*, i32** %1, align 8, !dbg !54
  %3 = load i32, i32* %2, align 4, !dbg !54
  store i32 %3, i32* %.du0001p_354, align 4, !dbg !54
  %4 = bitcast i64* %__nv_MAIN__F1L22_1Arg2 to i32**, !dbg !54
  %5 = load i32*, i32** %4, align 8, !dbg !54
  %6 = load i32, i32* %5, align 4, !dbg !54
  store i32 %6, i32* %.de0001p_355, align 4, !dbg !54
  store i32 1, i32* %.di0001p_356, align 4, !dbg !54
  %7 = load i32, i32* %.di0001p_356, align 4, !dbg !54
  store i32 %7, i32* %.ds0001p_357, align 4, !dbg !54
  store i32 1, i32* %.dl0001p_359, align 4, !dbg !54
  %8 = load i32, i32* %.dl0001p_359, align 4, !dbg !54
  store i32 %8, i32* %.dl0001p.copy_450, align 4, !dbg !54
  %9 = load i32, i32* %.de0001p_355, align 4, !dbg !54
  store i32 %9, i32* %.de0001p.copy_451, align 4, !dbg !54
  %10 = load i32, i32* %.ds0001p_357, align 4, !dbg !54
  store i32 %10, i32* %.ds0001p.copy_452, align 4, !dbg !54
  %11 = load i32, i32* %__gtid___nv_MAIN__F1L22_1__456, align 4, !dbg !54
  %12 = bitcast i32* %.i0000p_325 to i64*, !dbg !54
  %13 = bitcast i32* %.dl0001p.copy_450 to i64*, !dbg !54
  %14 = bitcast i32* %.de0001p.copy_451 to i64*, !dbg !54
  %15 = bitcast i32* %.ds0001p.copy_452 to i64*, !dbg !54
  %16 = load i32, i32* %.ds0001p.copy_452, align 4, !dbg !54
  call void @__kmpc_for_static_init_4(i64* null, i32 %11, i32 34, i64* %12, i64* %13, i64* %14, i64* %15, i32 %16, i32 1), !dbg !54
  %17 = load i32, i32* %.dl0001p.copy_450, align 4, !dbg !54
  store i32 %17, i32* %.dl0001p_359, align 4, !dbg !54
  %18 = load i32, i32* %.de0001p.copy_451, align 4, !dbg !54
  store i32 %18, i32* %.de0001p_355, align 4, !dbg !54
  %19 = load i32, i32* %.ds0001p.copy_452, align 4, !dbg !54
  store i32 %19, i32* %.ds0001p_357, align 4, !dbg !54
  %20 = load i32, i32* %.dl0001p_359, align 4, !dbg !54
  store i32 %20, i32* %i_324, align 4, !dbg !54
  %21 = load i32, i32* %i_324, align 4, !dbg !54
  call void @llvm.dbg.value(metadata i32 %21, metadata !55, metadata !DIExpression()), !dbg !53
  store i32 %21, i32* %.dX0001p_358, align 4, !dbg !54
  %22 = load i32, i32* %.dX0001p_358, align 4, !dbg !54
  %23 = load i32, i32* %.du0001p_354, align 4, !dbg !54
  %24 = icmp sgt i32 %22, %23, !dbg !54
  br i1 %24, label %L.LB2_352, label %L.LB2_482, !dbg !54

L.LB2_482:                                        ; preds = %L.LB2_323
  %25 = load i32, i32* %.dX0001p_358, align 4, !dbg !54
  store i32 %25, i32* %i_324, align 4, !dbg !54
  %26 = load i32, i32* %.di0001p_356, align 4, !dbg !54
  %27 = load i32, i32* %.de0001p_355, align 4, !dbg !54
  %28 = load i32, i32* %.dX0001p_358, align 4, !dbg !54
  %29 = sub nsw i32 %27, %28, !dbg !54
  %30 = add nsw i32 %26, %29, !dbg !54
  %31 = load i32, i32* %.di0001p_356, align 4, !dbg !54
  %32 = sdiv i32 %30, %31, !dbg !54
  store i32 %32, i32* %.dY0001p_353, align 4, !dbg !54
  %33 = load i32, i32* %.dY0001p_353, align 4, !dbg !54
  %34 = icmp sle i32 %33, 0, !dbg !54
  br i1 %34, label %L.LB2_362, label %L.LB2_361, !dbg !54

L.LB2_361:                                        ; preds = %L.LB2_361, %L.LB2_482
  %35 = load i32, i32* %i_324, align 4, !dbg !56
  call void @llvm.dbg.value(metadata i32 %35, metadata !55, metadata !DIExpression()), !dbg !53
  %36 = sext i32 %35 to i64, !dbg !56
  %37 = bitcast i64* %__nv_MAIN__F1L22_1Arg2 to i8*, !dbg !56
  %38 = getelementptr i8, i8* %37, i64 64, !dbg !56
  %39 = bitcast i8* %38 to i8**, !dbg !56
  %40 = load i8*, i8** %39, align 8, !dbg !56
  %41 = getelementptr i8, i8* %40, i64 56, !dbg !56
  %42 = bitcast i8* %41 to i64*, !dbg !56
  %43 = load i64, i64* %42, align 8, !dbg !56
  %44 = add nsw i64 %36, %43, !dbg !56
  %45 = bitcast i64* %__nv_MAIN__F1L22_1Arg2 to i8*, !dbg !56
  %46 = getelementptr i8, i8* %45, i64 16, !dbg !56
  %47 = bitcast i8* %46 to i8***, !dbg !56
  %48 = load i8**, i8*** %47, align 8, !dbg !56
  %49 = load i8*, i8** %48, align 8, !dbg !56
  %50 = getelementptr i8, i8* %49, i64 -4, !dbg !56
  %51 = bitcast i8* %50 to i32*, !dbg !56
  %52 = getelementptr i32, i32* %51, i64 %44, !dbg !56
  %53 = load i32, i32* %52, align 4, !dbg !56
  %54 = bitcast i64* %__nv_MAIN__F1L22_1Arg2 to i8*, !dbg !56
  %55 = getelementptr i8, i8* %54, i64 64, !dbg !56
  %56 = bitcast i8* %55 to i8**, !dbg !56
  %57 = load i8*, i8** %56, align 8, !dbg !56
  %58 = getelementptr i8, i8* %57, i64 56, !dbg !56
  %59 = bitcast i8* %58 to i64*, !dbg !56
  %60 = load i64, i64* %59, align 8, !dbg !56
  %61 = bitcast i64* %__nv_MAIN__F1L22_1Arg2 to i8*, !dbg !56
  %62 = getelementptr i8, i8* %61, i64 16, !dbg !56
  %63 = bitcast i8* %62 to i32***, !dbg !56
  %64 = load i32**, i32*** %63, align 8, !dbg !56
  %65 = load i32*, i32** %64, align 8, !dbg !56
  %66 = getelementptr i32, i32* %65, i64 %60, !dbg !56
  %67 = load i32, i32* %66, align 4, !dbg !56
  %68 = add nsw i32 %53, %67, !dbg !56
  %69 = load i32, i32* %i_324, align 4, !dbg !56
  call void @llvm.dbg.value(metadata i32 %69, metadata !55, metadata !DIExpression()), !dbg !53
  %70 = sext i32 %69 to i64, !dbg !56
  %71 = bitcast i64* %__nv_MAIN__F1L22_1Arg2 to i8*, !dbg !56
  %72 = getelementptr i8, i8* %71, i64 64, !dbg !56
  %73 = bitcast i8* %72 to i8**, !dbg !56
  %74 = load i8*, i8** %73, align 8, !dbg !56
  %75 = getelementptr i8, i8* %74, i64 56, !dbg !56
  %76 = bitcast i8* %75 to i64*, !dbg !56
  %77 = load i64, i64* %76, align 8, !dbg !56
  %78 = add nsw i64 %70, %77, !dbg !56
  %79 = bitcast i64* %__nv_MAIN__F1L22_1Arg2 to i8*, !dbg !56
  %80 = getelementptr i8, i8* %79, i64 16, !dbg !56
  %81 = bitcast i8* %80 to i8***, !dbg !56
  %82 = load i8**, i8*** %81, align 8, !dbg !56
  %83 = load i8*, i8** %82, align 8, !dbg !56
  %84 = getelementptr i8, i8* %83, i64 -4, !dbg !56
  %85 = bitcast i8* %84 to i32*, !dbg !56
  %86 = getelementptr i32, i32* %85, i64 %78, !dbg !56
  store i32 %68, i32* %86, align 4, !dbg !56
  %87 = load i32, i32* %.di0001p_356, align 4, !dbg !53
  %88 = load i32, i32* %i_324, align 4, !dbg !53
  call void @llvm.dbg.value(metadata i32 %88, metadata !55, metadata !DIExpression()), !dbg !53
  %89 = add nsw i32 %87, %88, !dbg !53
  store i32 %89, i32* %i_324, align 4, !dbg !53
  %90 = load i32, i32* %.dY0001p_353, align 4, !dbg !53
  %91 = sub nsw i32 %90, 1, !dbg !53
  store i32 %91, i32* %.dY0001p_353, align 4, !dbg !53
  %92 = load i32, i32* %.dY0001p_353, align 4, !dbg !53
  %93 = icmp sgt i32 %92, 0, !dbg !53
  br i1 %93, label %L.LB2_361, label %L.LB2_362, !dbg !53

L.LB2_362:                                        ; preds = %L.LB2_361, %L.LB2_482
  br label %L.LB2_352

L.LB2_352:                                        ; preds = %L.LB2_362, %L.LB2_323
  %94 = load i32, i32* %__gtid___nv_MAIN__F1L22_1__456, align 4, !dbg !53
  call void @__kmpc_for_static_fini(i64* null, i32 %94), !dbg !53
  br label %L.LB2_326

L.LB2_326:                                        ; preds = %L.LB2_352
  ret void, !dbg !53
}

declare void @__kmpc_for_static_fini(i64*, i32) #0

declare void @__kmpc_for_static_init_4(i64*, i32, i32, i64*, i64*, i64*, i64*, i32, i32) #0

declare void @f90_dealloc03a_i8(...) #0

declare void @_mp_ecs_nest(...) #0

declare signext i32 @f90io_fmtw_end(...) #0

declare signext i32 @f90io_sc_i_fmt_write(...) #0

declare signext i32 @f90io_fmtw_inita(...) #0

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

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { "no-frame-pointer-elim-non-leaf" }
attributes #1 = { nounwind readnone speculatable }

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !4, globals: !4, imports: !4)
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB039-truedepsingleelement-orig-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb039_truedepsingleelement_orig_yes", scope: !2, file: !3, line: 10, type: !6, scopeLine: 10, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
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
!18 = !DILocation(line: 32, column: 1, scope: !5)
!19 = !DILocation(line: 10, column: 1, scope: !5)
!20 = !DILocalVariable(name: "a", scope: !5, file: !3, type: !21)
!21 = !DICompositeType(tag: DW_TAG_array_type, baseType: !9, size: 32, align: 32, elements: !22)
!22 = !{!23}
!23 = !DISubrange(count: 0, lowerBound: 1)
!24 = !DILocalVariable(scope: !5, file: !3, type: !25, flags: DIFlagArtificial)
!25 = !DICompositeType(tag: DW_TAG_array_type, baseType: !26, size: 1024, align: 64, elements: !27)
!26 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!27 = !{!28}
!28 = !DISubrange(count: 16, lowerBound: 1)
!29 = !DILocalVariable(name: "len", scope: !5, file: !3, type: !9)
!30 = !DILocation(line: 17, column: 1, scope: !5)
!31 = !DILocalVariable(scope: !5, file: !3, type: !26, flags: DIFlagArtificial)
!32 = !DILocation(line: 18, column: 1, scope: !5)
!33 = !DILocation(line: 20, column: 1, scope: !5)
!34 = !DILocation(line: 22, column: 1, scope: !5)
!35 = !DILocation(line: 28, column: 1, scope: !5)
!36 = !DILocalVariable(scope: !5, file: !3, type: !9, flags: DIFlagArtificial)
!37 = !DILocation(line: 31, column: 1, scope: !5)
!38 = distinct !DISubprogram(name: "__nv_MAIN__F1L22_1", scope: !2, file: !3, line: 22, type: !39, scopeLine: 22, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!39 = !DISubroutineType(types: !40)
!40 = !{null, !9, !26, !26}
!41 = !DILocalVariable(name: "__nv_MAIN__F1L22_1Arg0", arg: 1, scope: !38, file: !3, type: !9)
!42 = !DILocation(line: 0, scope: !38)
!43 = !DILocalVariable(name: "__nv_MAIN__F1L22_1Arg1", arg: 2, scope: !38, file: !3, type: !26)
!44 = !DILocalVariable(name: "__nv_MAIN__F1L22_1Arg2", arg: 3, scope: !38, file: !3, type: !26)
!45 = !DILocalVariable(name: "omp_sched_static", scope: !38, file: !3, type: !9)
!46 = !DILocalVariable(name: "omp_sched_dynamic", scope: !38, file: !3, type: !9)
!47 = !DILocalVariable(name: "omp_proc_bind_false", scope: !38, file: !3, type: !9)
!48 = !DILocalVariable(name: "omp_proc_bind_true", scope: !38, file: !3, type: !9)
!49 = !DILocalVariable(name: "omp_proc_bind_master", scope: !38, file: !3, type: !9)
!50 = !DILocalVariable(name: "omp_lock_hint_none", scope: !38, file: !3, type: !9)
!51 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !38, file: !3, type: !9)
!52 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !38, file: !3, type: !9)
!53 = !DILocation(line: 25, column: 1, scope: !38)
!54 = !DILocation(line: 23, column: 1, scope: !38)
!55 = !DILocalVariable(name: "i", scope: !38, file: !3, type: !9)
!56 = !DILocation(line: 24, column: 1, scope: !38)

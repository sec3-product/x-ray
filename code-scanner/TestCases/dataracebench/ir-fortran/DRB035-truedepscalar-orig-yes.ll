; ModuleID = '/tmp/DRB035-truedepscalar-orig-yes-ad7f91.ll'
source_filename = "/tmp/DRB035-truedepscalar-orig-yes-ad7f91.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.STATICS1 = type <{ [44 x i8] }>
%astruct.dt68 = type <{ i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8* }>

@.STATICS1 = internal global %struct.STATICS1 <{ [44 x i8] c"\FB\FF\FF\FF\07\00\00\00a(50) =\00\EB\FF\FF\FF\00\00\00\00\03\00\00\00\00\00\00\00\01\00\00\00\FF\FF\FF\FF\00\00\00\00" }>, align 16
@.C332_MAIN_ = internal constant i64 50
@.C329_MAIN_ = internal constant i32 6
@.C325_MAIN_ = internal constant [58 x i8] c"micro-benchmarks-fortran/DRB035-truedepscalar-orig-yes.f95"
@.C327_MAIN_ = internal constant i32 29
@.C285_MAIN_ = internal constant i32 1
@.C305_MAIN_ = internal constant i32 25
@.C342_MAIN_ = internal constant i64 4
@.C341_MAIN_ = internal constant i64 25
@.C316_MAIN_ = internal constant i32 10
@.C314_MAIN_ = internal constant i32 100
@.C286_MAIN_ = internal constant i64 1
@.C284_MAIN_ = internal constant i64 0
@.C283_MAIN_ = internal constant i32 0
@.C285___nv_MAIN__F1L22_1 = internal constant i32 1
@.C283___nv_MAIN__F1L22_1 = internal constant i32 0

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__407 = alloca i32, align 4
  %.Z0966_317 = alloca i32*, align 8
  %"a$sd1_340" = alloca [16 x i64], align 8
  %len_315 = alloca i32, align 4
  %tmp_307 = alloca i32, align 4
  %z_b_0_308 = alloca i64, align 8
  %z_b_1_309 = alloca i64, align 8
  %z_e_60_312 = alloca i64, align 8
  %z_b_2_310 = alloca i64, align 8
  %z_b_3_311 = alloca i64, align 8
  %.uplevelArgPack0001_384 = alloca %astruct.dt68, align 16
  %z__io_331 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !14, metadata !DIExpression()), !dbg !10
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !15
  store i32 %0, i32* %__gtid_MAIN__407, align 4, !dbg !15
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !16
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !16
  call void (i8*, ...) %2(i8* %1), !dbg !16
  call void @llvm.dbg.declare(metadata i32** %.Z0966_317, metadata !17, metadata !DIExpression(DW_OP_deref)), !dbg !10
  %3 = bitcast i32** %.Z0966_317 to i8**, !dbg !16
  store i8* null, i8** %3, align 8, !dbg !16
  call void @llvm.dbg.declare(metadata [16 x i64]* %"a$sd1_340", metadata !21, metadata !DIExpression()), !dbg !10
  %4 = bitcast [16 x i64]* %"a$sd1_340" to i64*, !dbg !16
  store i64 0, i64* %4, align 8, !dbg !16
  br label %L.LB1_368

L.LB1_368:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %len_315, metadata !26, metadata !DIExpression()), !dbg !10
  store i32 100, i32* %len_315, align 4, !dbg !27
  call void @llvm.dbg.declare(metadata i32* %tmp_307, metadata !28, metadata !DIExpression()), !dbg !10
  store i32 10, i32* %tmp_307, align 4, !dbg !29
  call void @llvm.dbg.declare(metadata i64* %z_b_0_308, metadata !30, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_0_308, align 8, !dbg !31
  %5 = load i32, i32* %len_315, align 4, !dbg !31
  call void @llvm.dbg.value(metadata i32 %5, metadata !26, metadata !DIExpression()), !dbg !10
  %6 = sext i32 %5 to i64, !dbg !31
  call void @llvm.dbg.declare(metadata i64* %z_b_1_309, metadata !30, metadata !DIExpression()), !dbg !10
  store i64 %6, i64* %z_b_1_309, align 8, !dbg !31
  %7 = load i64, i64* %z_b_1_309, align 8, !dbg !31
  call void @llvm.dbg.value(metadata i64 %7, metadata !30, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_60_312, metadata !30, metadata !DIExpression()), !dbg !10
  store i64 %7, i64* %z_e_60_312, align 8, !dbg !31
  %8 = bitcast [16 x i64]* %"a$sd1_340" to i8*, !dbg !31
  %9 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !31
  %10 = bitcast i64* @.C341_MAIN_ to i8*, !dbg !31
  %11 = bitcast i64* @.C342_MAIN_ to i8*, !dbg !31
  %12 = bitcast i64* %z_b_0_308 to i8*, !dbg !31
  %13 = bitcast i64* %z_b_1_309 to i8*, !dbg !31
  %14 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !31
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %14(i8* %8, i8* %9, i8* %10, i8* %11, i8* %12, i8* %13), !dbg !31
  %15 = bitcast [16 x i64]* %"a$sd1_340" to i8*, !dbg !31
  %16 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !31
  call void (i8*, i32, ...) %16(i8* %15, i32 25), !dbg !31
  %17 = load i64, i64* %z_b_1_309, align 8, !dbg !31
  call void @llvm.dbg.value(metadata i64 %17, metadata !30, metadata !DIExpression()), !dbg !10
  %18 = load i64, i64* %z_b_0_308, align 8, !dbg !31
  call void @llvm.dbg.value(metadata i64 %18, metadata !30, metadata !DIExpression()), !dbg !10
  %19 = sub nsw i64 %18, 1, !dbg !31
  %20 = sub nsw i64 %17, %19, !dbg !31
  call void @llvm.dbg.declare(metadata i64* %z_b_2_310, metadata !30, metadata !DIExpression()), !dbg !10
  store i64 %20, i64* %z_b_2_310, align 8, !dbg !31
  %21 = load i64, i64* %z_b_0_308, align 8, !dbg !31
  call void @llvm.dbg.value(metadata i64 %21, metadata !30, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_b_3_311, metadata !30, metadata !DIExpression()), !dbg !10
  store i64 %21, i64* %z_b_3_311, align 8, !dbg !31
  %22 = bitcast i64* %z_b_2_310 to i8*, !dbg !31
  %23 = bitcast i64* @.C341_MAIN_ to i8*, !dbg !31
  %24 = bitcast i64* @.C342_MAIN_ to i8*, !dbg !31
  %25 = bitcast i32** %.Z0966_317 to i8*, !dbg !31
  %26 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !31
  %27 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !31
  %28 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !31
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %28(i8* %22, i8* %23, i8* %24, i8* null, i8* %25, i8* null, i8* %26, i8* %27, i8* null, i64 0), !dbg !31
  %29 = bitcast i32* %len_315 to i8*, !dbg !32
  %30 = bitcast %astruct.dt68* %.uplevelArgPack0001_384 to i8**, !dbg !32
  store i8* %29, i8** %30, align 8, !dbg !32
  %31 = bitcast i32** %.Z0966_317 to i8*, !dbg !32
  %32 = bitcast %astruct.dt68* %.uplevelArgPack0001_384 to i8*, !dbg !32
  %33 = getelementptr i8, i8* %32, i64 8, !dbg !32
  %34 = bitcast i8* %33 to i8**, !dbg !32
  store i8* %31, i8** %34, align 8, !dbg !32
  %35 = bitcast i32** %.Z0966_317 to i8*, !dbg !32
  %36 = bitcast %astruct.dt68* %.uplevelArgPack0001_384 to i8*, !dbg !32
  %37 = getelementptr i8, i8* %36, i64 16, !dbg !32
  %38 = bitcast i8* %37 to i8**, !dbg !32
  store i8* %35, i8** %38, align 8, !dbg !32
  %39 = bitcast i64* %z_b_0_308 to i8*, !dbg !32
  %40 = bitcast %astruct.dt68* %.uplevelArgPack0001_384 to i8*, !dbg !32
  %41 = getelementptr i8, i8* %40, i64 24, !dbg !32
  %42 = bitcast i8* %41 to i8**, !dbg !32
  store i8* %39, i8** %42, align 8, !dbg !32
  %43 = bitcast i64* %z_b_1_309 to i8*, !dbg !32
  %44 = bitcast %astruct.dt68* %.uplevelArgPack0001_384 to i8*, !dbg !32
  %45 = getelementptr i8, i8* %44, i64 32, !dbg !32
  %46 = bitcast i8* %45 to i8**, !dbg !32
  store i8* %43, i8** %46, align 8, !dbg !32
  %47 = bitcast i64* %z_e_60_312 to i8*, !dbg !32
  %48 = bitcast %astruct.dt68* %.uplevelArgPack0001_384 to i8*, !dbg !32
  %49 = getelementptr i8, i8* %48, i64 40, !dbg !32
  %50 = bitcast i8* %49 to i8**, !dbg !32
  store i8* %47, i8** %50, align 8, !dbg !32
  %51 = bitcast i64* %z_b_2_310 to i8*, !dbg !32
  %52 = bitcast %astruct.dt68* %.uplevelArgPack0001_384 to i8*, !dbg !32
  %53 = getelementptr i8, i8* %52, i64 48, !dbg !32
  %54 = bitcast i8* %53 to i8**, !dbg !32
  store i8* %51, i8** %54, align 8, !dbg !32
  %55 = bitcast i64* %z_b_3_311 to i8*, !dbg !32
  %56 = bitcast %astruct.dt68* %.uplevelArgPack0001_384 to i8*, !dbg !32
  %57 = getelementptr i8, i8* %56, i64 56, !dbg !32
  %58 = bitcast i8* %57 to i8**, !dbg !32
  store i8* %55, i8** %58, align 8, !dbg !32
  %59 = bitcast i32* %tmp_307 to i8*, !dbg !32
  %60 = bitcast %astruct.dt68* %.uplevelArgPack0001_384 to i8*, !dbg !32
  %61 = getelementptr i8, i8* %60, i64 64, !dbg !32
  %62 = bitcast i8* %61 to i8**, !dbg !32
  store i8* %59, i8** %62, align 8, !dbg !32
  %63 = bitcast [16 x i64]* %"a$sd1_340" to i8*, !dbg !32
  %64 = bitcast %astruct.dt68* %.uplevelArgPack0001_384 to i8*, !dbg !32
  %65 = getelementptr i8, i8* %64, i64 72, !dbg !32
  %66 = bitcast i8* %65 to i8**, !dbg !32
  store i8* %63, i8** %66, align 8, !dbg !32
  br label %L.LB1_405, !dbg !32

L.LB1_405:                                        ; preds = %L.LB1_368
  %67 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L22_1_ to i64*, !dbg !32
  %68 = bitcast %astruct.dt68* %.uplevelArgPack0001_384 to i64*, !dbg !32
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %67, i64* %68), !dbg !32
  call void (...) @_mp_bcs_nest(), !dbg !33
  %69 = bitcast i32* @.C327_MAIN_ to i8*, !dbg !33
  %70 = bitcast [58 x i8]* @.C325_MAIN_ to i8*, !dbg !33
  %71 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !33
  call void (i8*, i8*, i64, ...) %71(i8* %69, i8* %70, i64 58), !dbg !33
  %72 = bitcast i32* @.C329_MAIN_ to i8*, !dbg !33
  %73 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !33
  %74 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !33
  %75 = bitcast %struct.STATICS1* @.STATICS1 to i8*, !dbg !33
  %76 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !33
  %77 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %76(i8* %72, i8* null, i8* %73, i8* %74, i8* %75, i8* null, i64 0), !dbg !33
  call void @llvm.dbg.declare(metadata i32* %z__io_331, metadata !34, metadata !DIExpression()), !dbg !10
  store i32 %77, i32* %z__io_331, align 4, !dbg !33
  %78 = bitcast [16 x i64]* %"a$sd1_340" to i8*, !dbg !33
  %79 = getelementptr i8, i8* %78, i64 56, !dbg !33
  %80 = bitcast i8* %79 to i64*, !dbg !33
  %81 = load i64, i64* %80, align 8, !dbg !33
  %82 = load i32*, i32** %.Z0966_317, align 8, !dbg !33
  call void @llvm.dbg.value(metadata i32* %82, metadata !17, metadata !DIExpression()), !dbg !10
  %83 = bitcast i32* %82 to i8*, !dbg !33
  %84 = getelementptr i8, i8* %83, i64 196, !dbg !33
  %85 = bitcast i8* %84 to i32*, !dbg !33
  %86 = getelementptr i32, i32* %85, i64 %81, !dbg !33
  %87 = load i32, i32* %86, align 4, !dbg !33
  %88 = bitcast i32 (...)* @f90io_sc_i_fmt_write to i32 (i32, i32, ...)*, !dbg !33
  %89 = call i32 (i32, i32, ...) %88(i32 %87, i32 25), !dbg !33
  store i32 %89, i32* %z__io_331, align 4, !dbg !33
  %90 = call i32 (...) @f90io_fmtw_end(), !dbg !33
  store i32 %90, i32* %z__io_331, align 4, !dbg !33
  call void (...) @_mp_ecs_nest(), !dbg !33
  %91 = load i32*, i32** %.Z0966_317, align 8, !dbg !35
  call void @llvm.dbg.value(metadata i32* %91, metadata !17, metadata !DIExpression()), !dbg !10
  %92 = bitcast i32* %91 to i8*, !dbg !35
  %93 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !35
  %94 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !35
  call void (i8*, i8*, i8*, i8*, i64, ...) %94(i8* null, i8* %92, i8* %93, i8* null, i64 0), !dbg !35
  %95 = bitcast i32** %.Z0966_317 to i8**, !dbg !35
  store i8* null, i8** %95, align 8, !dbg !35
  %96 = bitcast [16 x i64]* %"a$sd1_340" to i64*, !dbg !35
  store i64 0, i64* %96, align 8, !dbg !35
  ret void, !dbg !15
}

define internal void @__nv_MAIN__F1L22_1_(i32* %__nv_MAIN__F1L22_1Arg0, i64* %__nv_MAIN__F1L22_1Arg1, i64* %__nv_MAIN__F1L22_1Arg2) #0 !dbg !36 {
L.entry:
  %__gtid___nv_MAIN__F1L22_1__457 = alloca i32, align 4
  %.i0000p_322 = alloca i32, align 4
  %i_321 = alloca i32, align 4
  %.du0001p_352 = alloca i32, align 4
  %.de0001p_353 = alloca i32, align 4
  %.di0001p_354 = alloca i32, align 4
  %.ds0001p_355 = alloca i32, align 4
  %.dl0001p_357 = alloca i32, align 4
  %.dl0001p.copy_451 = alloca i32, align 4
  %.de0001p.copy_452 = alloca i32, align 4
  %.ds0001p.copy_453 = alloca i32, align 4
  %.dX0001p_356 = alloca i32, align 4
  %.dY0001p_351 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L22_1Arg0, metadata !39, metadata !DIExpression()), !dbg !40
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L22_1Arg1, metadata !41, metadata !DIExpression()), !dbg !40
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L22_1Arg2, metadata !42, metadata !DIExpression()), !dbg !40
  call void @llvm.dbg.value(metadata i32 1, metadata !43, metadata !DIExpression()), !dbg !40
  call void @llvm.dbg.value(metadata i32 0, metadata !44, metadata !DIExpression()), !dbg !40
  call void @llvm.dbg.value(metadata i32 1, metadata !45, metadata !DIExpression()), !dbg !40
  call void @llvm.dbg.value(metadata i32 0, metadata !46, metadata !DIExpression()), !dbg !40
  call void @llvm.dbg.value(metadata i32 1, metadata !47, metadata !DIExpression()), !dbg !40
  %0 = load i32, i32* %__nv_MAIN__F1L22_1Arg0, align 4, !dbg !48
  store i32 %0, i32* %__gtid___nv_MAIN__F1L22_1__457, align 4, !dbg !48
  br label %L.LB2_442

L.LB2_442:                                        ; preds = %L.entry
  br label %L.LB2_320

L.LB2_320:                                        ; preds = %L.LB2_442
  store i32 0, i32* %.i0000p_322, align 4, !dbg !49
  call void @llvm.dbg.declare(metadata i32* %i_321, metadata !50, metadata !DIExpression()), !dbg !48
  store i32 1, i32* %i_321, align 4, !dbg !49
  %1 = bitcast i64* %__nv_MAIN__F1L22_1Arg2 to i32**, !dbg !49
  %2 = load i32*, i32** %1, align 8, !dbg !49
  %3 = load i32, i32* %2, align 4, !dbg !49
  store i32 %3, i32* %.du0001p_352, align 4, !dbg !49
  %4 = bitcast i64* %__nv_MAIN__F1L22_1Arg2 to i32**, !dbg !49
  %5 = load i32*, i32** %4, align 8, !dbg !49
  %6 = load i32, i32* %5, align 4, !dbg !49
  store i32 %6, i32* %.de0001p_353, align 4, !dbg !49
  store i32 1, i32* %.di0001p_354, align 4, !dbg !49
  %7 = load i32, i32* %.di0001p_354, align 4, !dbg !49
  store i32 %7, i32* %.ds0001p_355, align 4, !dbg !49
  store i32 1, i32* %.dl0001p_357, align 4, !dbg !49
  %8 = load i32, i32* %.dl0001p_357, align 4, !dbg !49
  store i32 %8, i32* %.dl0001p.copy_451, align 4, !dbg !49
  %9 = load i32, i32* %.de0001p_353, align 4, !dbg !49
  store i32 %9, i32* %.de0001p.copy_452, align 4, !dbg !49
  %10 = load i32, i32* %.ds0001p_355, align 4, !dbg !49
  store i32 %10, i32* %.ds0001p.copy_453, align 4, !dbg !49
  %11 = load i32, i32* %__gtid___nv_MAIN__F1L22_1__457, align 4, !dbg !49
  %12 = bitcast i32* %.i0000p_322 to i64*, !dbg !49
  %13 = bitcast i32* %.dl0001p.copy_451 to i64*, !dbg !49
  %14 = bitcast i32* %.de0001p.copy_452 to i64*, !dbg !49
  %15 = bitcast i32* %.ds0001p.copy_453 to i64*, !dbg !49
  %16 = load i32, i32* %.ds0001p.copy_453, align 4, !dbg !49
  call void @__kmpc_for_static_init_4(i64* null, i32 %11, i32 34, i64* %12, i64* %13, i64* %14, i64* %15, i32 %16, i32 1), !dbg !49
  %17 = load i32, i32* %.dl0001p.copy_451, align 4, !dbg !49
  store i32 %17, i32* %.dl0001p_357, align 4, !dbg !49
  %18 = load i32, i32* %.de0001p.copy_452, align 4, !dbg !49
  store i32 %18, i32* %.de0001p_353, align 4, !dbg !49
  %19 = load i32, i32* %.ds0001p.copy_453, align 4, !dbg !49
  store i32 %19, i32* %.ds0001p_355, align 4, !dbg !49
  %20 = load i32, i32* %.dl0001p_357, align 4, !dbg !49
  store i32 %20, i32* %i_321, align 4, !dbg !49
  %21 = load i32, i32* %i_321, align 4, !dbg !49
  call void @llvm.dbg.value(metadata i32 %21, metadata !50, metadata !DIExpression()), !dbg !48
  store i32 %21, i32* %.dX0001p_356, align 4, !dbg !49
  %22 = load i32, i32* %.dX0001p_356, align 4, !dbg !49
  %23 = load i32, i32* %.du0001p_352, align 4, !dbg !49
  %24 = icmp sgt i32 %22, %23, !dbg !49
  br i1 %24, label %L.LB2_350, label %L.LB2_484, !dbg !49

L.LB2_484:                                        ; preds = %L.LB2_320
  %25 = load i32, i32* %.dX0001p_356, align 4, !dbg !49
  store i32 %25, i32* %i_321, align 4, !dbg !49
  %26 = load i32, i32* %.di0001p_354, align 4, !dbg !49
  %27 = load i32, i32* %.de0001p_353, align 4, !dbg !49
  %28 = load i32, i32* %.dX0001p_356, align 4, !dbg !49
  %29 = sub nsw i32 %27, %28, !dbg !49
  %30 = add nsw i32 %26, %29, !dbg !49
  %31 = load i32, i32* %.di0001p_354, align 4, !dbg !49
  %32 = sdiv i32 %30, %31, !dbg !49
  store i32 %32, i32* %.dY0001p_351, align 4, !dbg !49
  %33 = load i32, i32* %.dY0001p_351, align 4, !dbg !49
  %34 = icmp sle i32 %33, 0, !dbg !49
  br i1 %34, label %L.LB2_360, label %L.LB2_359, !dbg !49

L.LB2_359:                                        ; preds = %L.LB2_359, %L.LB2_484
  %35 = bitcast i64* %__nv_MAIN__F1L22_1Arg2 to i8*, !dbg !51
  %36 = getelementptr i8, i8* %35, i64 64, !dbg !51
  %37 = bitcast i8* %36 to i32**, !dbg !51
  %38 = load i32*, i32** %37, align 8, !dbg !51
  %39 = load i32, i32* %38, align 4, !dbg !51
  %40 = load i32, i32* %i_321, align 4, !dbg !51
  call void @llvm.dbg.value(metadata i32 %40, metadata !50, metadata !DIExpression()), !dbg !48
  %41 = sext i32 %40 to i64, !dbg !51
  %42 = bitcast i64* %__nv_MAIN__F1L22_1Arg2 to i8*, !dbg !51
  %43 = getelementptr i8, i8* %42, i64 72, !dbg !51
  %44 = bitcast i8* %43 to i8**, !dbg !51
  %45 = load i8*, i8** %44, align 8, !dbg !51
  %46 = getelementptr i8, i8* %45, i64 56, !dbg !51
  %47 = bitcast i8* %46 to i64*, !dbg !51
  %48 = load i64, i64* %47, align 8, !dbg !51
  %49 = add nsw i64 %41, %48, !dbg !51
  %50 = bitcast i64* %__nv_MAIN__F1L22_1Arg2 to i8*, !dbg !51
  %51 = getelementptr i8, i8* %50, i64 16, !dbg !51
  %52 = bitcast i8* %51 to i8***, !dbg !51
  %53 = load i8**, i8*** %52, align 8, !dbg !51
  %54 = load i8*, i8** %53, align 8, !dbg !51
  %55 = getelementptr i8, i8* %54, i64 -4, !dbg !51
  %56 = bitcast i8* %55 to i32*, !dbg !51
  %57 = getelementptr i32, i32* %56, i64 %49, !dbg !51
  store i32 %39, i32* %57, align 4, !dbg !51
  %58 = load i32, i32* %i_321, align 4, !dbg !52
  call void @llvm.dbg.value(metadata i32 %58, metadata !50, metadata !DIExpression()), !dbg !48
  %59 = load i32, i32* %i_321, align 4, !dbg !52
  call void @llvm.dbg.value(metadata i32 %59, metadata !50, metadata !DIExpression()), !dbg !48
  %60 = sext i32 %59 to i64, !dbg !52
  %61 = bitcast i64* %__nv_MAIN__F1L22_1Arg2 to i8*, !dbg !52
  %62 = getelementptr i8, i8* %61, i64 72, !dbg !52
  %63 = bitcast i8* %62 to i8**, !dbg !52
  %64 = load i8*, i8** %63, align 8, !dbg !52
  %65 = getelementptr i8, i8* %64, i64 56, !dbg !52
  %66 = bitcast i8* %65 to i64*, !dbg !52
  %67 = load i64, i64* %66, align 8, !dbg !52
  %68 = add nsw i64 %60, %67, !dbg !52
  %69 = bitcast i64* %__nv_MAIN__F1L22_1Arg2 to i8*, !dbg !52
  %70 = getelementptr i8, i8* %69, i64 16, !dbg !52
  %71 = bitcast i8* %70 to i8***, !dbg !52
  %72 = load i8**, i8*** %71, align 8, !dbg !52
  %73 = load i8*, i8** %72, align 8, !dbg !52
  %74 = getelementptr i8, i8* %73, i64 -4, !dbg !52
  %75 = bitcast i8* %74 to i32*, !dbg !52
  %76 = getelementptr i32, i32* %75, i64 %68, !dbg !52
  %77 = load i32, i32* %76, align 4, !dbg !52
  %78 = add nsw i32 %58, %77, !dbg !52
  %79 = bitcast i64* %__nv_MAIN__F1L22_1Arg2 to i8*, !dbg !52
  %80 = getelementptr i8, i8* %79, i64 64, !dbg !52
  %81 = bitcast i8* %80 to i32**, !dbg !52
  %82 = load i32*, i32** %81, align 8, !dbg !52
  store i32 %78, i32* %82, align 4, !dbg !52
  %83 = load i32, i32* %.di0001p_354, align 4, !dbg !48
  %84 = load i32, i32* %i_321, align 4, !dbg !48
  call void @llvm.dbg.value(metadata i32 %84, metadata !50, metadata !DIExpression()), !dbg !48
  %85 = add nsw i32 %83, %84, !dbg !48
  store i32 %85, i32* %i_321, align 4, !dbg !48
  %86 = load i32, i32* %.dY0001p_351, align 4, !dbg !48
  %87 = sub nsw i32 %86, 1, !dbg !48
  store i32 %87, i32* %.dY0001p_351, align 4, !dbg !48
  %88 = load i32, i32* %.dY0001p_351, align 4, !dbg !48
  %89 = icmp sgt i32 %88, 0, !dbg !48
  br i1 %89, label %L.LB2_359, label %L.LB2_360, !dbg !48

L.LB2_360:                                        ; preds = %L.LB2_359, %L.LB2_484
  br label %L.LB2_350

L.LB2_350:                                        ; preds = %L.LB2_360, %L.LB2_320
  %90 = load i32, i32* %__gtid___nv_MAIN__F1L22_1__457, align 4, !dbg !48
  call void @__kmpc_for_static_fini(i64* null, i32 %90), !dbg !48
  br label %L.LB2_323

L.LB2_323:                                        ; preds = %L.LB2_350
  ret void, !dbg !48
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
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB035-truedepscalar-orig-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb035_truedepscalar_orig_yes", scope: !2, file: !3, line: 11, type: !6, scopeLine: 11, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!6 = !DISubroutineType(cc: DW_CC_program, types: !7)
!7 = !{null}
!8 = !DILocalVariable(name: "omp_sched_static", scope: !5, file: !3, type: !9)
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 0, scope: !5)
!11 = !DILocalVariable(name: "omp_proc_bind_false", scope: !5, file: !3, type: !9)
!12 = !DILocalVariable(name: "omp_proc_bind_true", scope: !5, file: !3, type: !9)
!13 = !DILocalVariable(name: "omp_lock_hint_none", scope: !5, file: !3, type: !9)
!14 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !5, file: !3, type: !9)
!15 = !DILocation(line: 33, column: 1, scope: !5)
!16 = !DILocation(line: 11, column: 1, scope: !5)
!17 = !DILocalVariable(name: "a", scope: !5, file: !3, type: !18)
!18 = !DICompositeType(tag: DW_TAG_array_type, baseType: !9, size: 32, align: 32, elements: !19)
!19 = !{!20}
!20 = !DISubrange(count: 0, lowerBound: 1)
!21 = !DILocalVariable(scope: !5, file: !3, type: !22, flags: DIFlagArtificial)
!22 = !DICompositeType(tag: DW_TAG_array_type, baseType: !23, size: 1024, align: 64, elements: !24)
!23 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!24 = !{!25}
!25 = !DISubrange(count: 16, lowerBound: 1)
!26 = !DILocalVariable(name: "len", scope: !5, file: !3, type: !9)
!27 = !DILocation(line: 18, column: 1, scope: !5)
!28 = !DILocalVariable(name: "tmp", scope: !5, file: !3, type: !9)
!29 = !DILocation(line: 19, column: 1, scope: !5)
!30 = !DILocalVariable(scope: !5, file: !3, type: !23, flags: DIFlagArtificial)
!31 = !DILocation(line: 20, column: 1, scope: !5)
!32 = !DILocation(line: 22, column: 1, scope: !5)
!33 = !DILocation(line: 29, column: 1, scope: !5)
!34 = !DILocalVariable(scope: !5, file: !3, type: !9, flags: DIFlagArtificial)
!35 = !DILocation(line: 32, column: 1, scope: !5)
!36 = distinct !DISubprogram(name: "__nv_MAIN__F1L22_1", scope: !2, file: !3, line: 22, type: !37, scopeLine: 22, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!37 = !DISubroutineType(types: !38)
!38 = !{null, !9, !23, !23}
!39 = !DILocalVariable(name: "__nv_MAIN__F1L22_1Arg0", arg: 1, scope: !36, file: !3, type: !9)
!40 = !DILocation(line: 0, scope: !36)
!41 = !DILocalVariable(name: "__nv_MAIN__F1L22_1Arg1", arg: 2, scope: !36, file: !3, type: !23)
!42 = !DILocalVariable(name: "__nv_MAIN__F1L22_1Arg2", arg: 3, scope: !36, file: !3, type: !23)
!43 = !DILocalVariable(name: "omp_sched_static", scope: !36, file: !3, type: !9)
!44 = !DILocalVariable(name: "omp_proc_bind_false", scope: !36, file: !3, type: !9)
!45 = !DILocalVariable(name: "omp_proc_bind_true", scope: !36, file: !3, type: !9)
!46 = !DILocalVariable(name: "omp_lock_hint_none", scope: !36, file: !3, type: !9)
!47 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !36, file: !3, type: !9)
!48 = !DILocation(line: 26, column: 1, scope: !36)
!49 = !DILocation(line: 23, column: 1, scope: !36)
!50 = !DILocalVariable(name: "i", scope: !36, file: !3, type: !9)
!51 = !DILocation(line: 24, column: 1, scope: !36)
!52 = !DILocation(line: 25, column: 1, scope: !36)

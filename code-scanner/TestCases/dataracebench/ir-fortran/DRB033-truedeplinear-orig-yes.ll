; ModuleID = '/tmp/DRB033-truedeplinear-orig-yes-fae683.ll'
source_filename = "/tmp/DRB033-truedeplinear-orig-yes-fae683.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.STATICS1 = type <{ [48 x i8] }>
%astruct.dt68 = type <{ i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8* }>

@.STATICS1 = internal global %struct.STATICS1 <{ [48 x i8] c"\FB\FF\FF\FF\09\00\00\00a(1002) =\00\00\00\EB\FF\FF\FF\00\00\00\00\03\00\00\00\00\00\00\00\01\00\00\00\FF\FF\FF\FF\00\00\00\00" }>, align 16
@.C335_MAIN_ = internal constant i64 1002
@.C332_MAIN_ = internal constant i32 6
@.C328_MAIN_ = internal constant [58 x i8] c"micro-benchmarks-fortran/DRB033-truedeplinear-orig-yes.f95"
@.C330_MAIN_ = internal constant i32 31
@.C300_MAIN_ = internal constant i32 2
@.C323_MAIN_ = internal constant i32 1000
@.C285_MAIN_ = internal constant i32 1
@.C309_MAIN_ = internal constant i32 25
@.C345_MAIN_ = internal constant i64 4
@.C344_MAIN_ = internal constant i64 25
@.C317_MAIN_ = internal constant i32 2000
@.C286_MAIN_ = internal constant i64 1
@.C284_MAIN_ = internal constant i64 0
@.C283_MAIN_ = internal constant i32 0
@.C300___nv_MAIN__F1L25_1 = internal constant i32 2
@.C323___nv_MAIN__F1L25_1 = internal constant i32 1000
@.C285___nv_MAIN__F1L25_1 = internal constant i32 1
@.C283___nv_MAIN__F1L25_1 = internal constant i32 0

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__414 = alloca i32, align 4
  %.Z0965_319 = alloca i32*, align 8
  %"a$sd1_343" = alloca [16 x i64], align 8
  %len_318 = alloca i32, align 4
  %z_b_0_311 = alloca i64, align 8
  %z_b_1_312 = alloca i64, align 8
  %z_e_60_315 = alloca i64, align 8
  %z_b_2_313 = alloca i64, align 8
  %z_b_3_314 = alloca i64, align 8
  %.dY0001_354 = alloca i32, align 4
  %i_310 = alloca i32, align 4
  %.uplevelArgPack0001_395 = alloca %astruct.dt68, align 16
  %z__io_334 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 2, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 2, metadata !14, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !15, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !16, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 2, metadata !17, metadata !DIExpression()), !dbg !10
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !18
  store i32 %0, i32* %__gtid_MAIN__414, align 4, !dbg !18
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !19
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !19
  call void (i8*, ...) %2(i8* %1), !dbg !19
  call void @llvm.dbg.declare(metadata i32** %.Z0965_319, metadata !20, metadata !DIExpression(DW_OP_deref)), !dbg !10
  %3 = bitcast i32** %.Z0965_319 to i8**, !dbg !19
  store i8* null, i8** %3, align 8, !dbg !19
  call void @llvm.dbg.declare(metadata [16 x i64]* %"a$sd1_343", metadata !24, metadata !DIExpression()), !dbg !10
  %4 = bitcast [16 x i64]* %"a$sd1_343" to i64*, !dbg !19
  store i64 0, i64* %4, align 8, !dbg !19
  br label %L.LB1_374

L.LB1_374:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %len_318, metadata !29, metadata !DIExpression()), !dbg !10
  store i32 2000, i32* %len_318, align 4, !dbg !30
  call void @llvm.dbg.declare(metadata i64* %z_b_0_311, metadata !31, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_0_311, align 8, !dbg !32
  %5 = load i32, i32* %len_318, align 4, !dbg !32
  call void @llvm.dbg.value(metadata i32 %5, metadata !29, metadata !DIExpression()), !dbg !10
  %6 = sext i32 %5 to i64, !dbg !32
  call void @llvm.dbg.declare(metadata i64* %z_b_1_312, metadata !31, metadata !DIExpression()), !dbg !10
  store i64 %6, i64* %z_b_1_312, align 8, !dbg !32
  %7 = load i64, i64* %z_b_1_312, align 8, !dbg !32
  call void @llvm.dbg.value(metadata i64 %7, metadata !31, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_60_315, metadata !31, metadata !DIExpression()), !dbg !10
  store i64 %7, i64* %z_e_60_315, align 8, !dbg !32
  %8 = bitcast [16 x i64]* %"a$sd1_343" to i8*, !dbg !32
  %9 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !32
  %10 = bitcast i64* @.C344_MAIN_ to i8*, !dbg !32
  %11 = bitcast i64* @.C345_MAIN_ to i8*, !dbg !32
  %12 = bitcast i64* %z_b_0_311 to i8*, !dbg !32
  %13 = bitcast i64* %z_b_1_312 to i8*, !dbg !32
  %14 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !32
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %14(i8* %8, i8* %9, i8* %10, i8* %11, i8* %12, i8* %13), !dbg !32
  %15 = bitcast [16 x i64]* %"a$sd1_343" to i8*, !dbg !32
  %16 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !32
  call void (i8*, i32, ...) %16(i8* %15, i32 25), !dbg !32
  %17 = load i64, i64* %z_b_1_312, align 8, !dbg !32
  call void @llvm.dbg.value(metadata i64 %17, metadata !31, metadata !DIExpression()), !dbg !10
  %18 = load i64, i64* %z_b_0_311, align 8, !dbg !32
  call void @llvm.dbg.value(metadata i64 %18, metadata !31, metadata !DIExpression()), !dbg !10
  %19 = sub nsw i64 %18, 1, !dbg !32
  %20 = sub nsw i64 %17, %19, !dbg !32
  call void @llvm.dbg.declare(metadata i64* %z_b_2_313, metadata !31, metadata !DIExpression()), !dbg !10
  store i64 %20, i64* %z_b_2_313, align 8, !dbg !32
  %21 = load i64, i64* %z_b_0_311, align 8, !dbg !32
  call void @llvm.dbg.value(metadata i64 %21, metadata !31, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_b_3_314, metadata !31, metadata !DIExpression()), !dbg !10
  store i64 %21, i64* %z_b_3_314, align 8, !dbg !32
  %22 = bitcast i64* %z_b_2_313 to i8*, !dbg !32
  %23 = bitcast i64* @.C344_MAIN_ to i8*, !dbg !32
  %24 = bitcast i64* @.C345_MAIN_ to i8*, !dbg !32
  %25 = bitcast i32** %.Z0965_319 to i8*, !dbg !32
  %26 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !32
  %27 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !32
  %28 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !32
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %28(i8* %22, i8* %23, i8* %24, i8* null, i8* %25, i8* null, i8* %26, i8* %27, i8* null, i64 0), !dbg !32
  %29 = load i32, i32* %len_318, align 4, !dbg !33
  call void @llvm.dbg.value(metadata i32 %29, metadata !29, metadata !DIExpression()), !dbg !10
  store i32 %29, i32* %.dY0001_354, align 4, !dbg !33
  call void @llvm.dbg.declare(metadata i32* %i_310, metadata !34, metadata !DIExpression()), !dbg !10
  store i32 1, i32* %i_310, align 4, !dbg !33
  %30 = load i32, i32* %.dY0001_354, align 4, !dbg !33
  %31 = icmp sle i32 %30, 0, !dbg !33
  br i1 %31, label %L.LB1_353, label %L.LB1_352, !dbg !33

L.LB1_352:                                        ; preds = %L.LB1_352, %L.LB1_374
  %32 = load i32, i32* %i_310, align 4, !dbg !35
  call void @llvm.dbg.value(metadata i32 %32, metadata !34, metadata !DIExpression()), !dbg !10
  %33 = load i32, i32* %i_310, align 4, !dbg !35
  call void @llvm.dbg.value(metadata i32 %33, metadata !34, metadata !DIExpression()), !dbg !10
  %34 = sext i32 %33 to i64, !dbg !35
  %35 = bitcast [16 x i64]* %"a$sd1_343" to i8*, !dbg !35
  %36 = getelementptr i8, i8* %35, i64 56, !dbg !35
  %37 = bitcast i8* %36 to i64*, !dbg !35
  %38 = load i64, i64* %37, align 8, !dbg !35
  %39 = add nsw i64 %34, %38, !dbg !35
  %40 = load i32*, i32** %.Z0965_319, align 8, !dbg !35
  call void @llvm.dbg.value(metadata i32* %40, metadata !20, metadata !DIExpression()), !dbg !10
  %41 = bitcast i32* %40 to i8*, !dbg !35
  %42 = getelementptr i8, i8* %41, i64 -4, !dbg !35
  %43 = bitcast i8* %42 to i32*, !dbg !35
  %44 = getelementptr i32, i32* %43, i64 %39, !dbg !35
  store i32 %32, i32* %44, align 4, !dbg !35
  %45 = load i32, i32* %i_310, align 4, !dbg !36
  call void @llvm.dbg.value(metadata i32 %45, metadata !34, metadata !DIExpression()), !dbg !10
  %46 = add nsw i32 %45, 1, !dbg !36
  store i32 %46, i32* %i_310, align 4, !dbg !36
  %47 = load i32, i32* %.dY0001_354, align 4, !dbg !36
  %48 = sub nsw i32 %47, 1, !dbg !36
  store i32 %48, i32* %.dY0001_354, align 4, !dbg !36
  %49 = load i32, i32* %.dY0001_354, align 4, !dbg !36
  %50 = icmp sgt i32 %49, 0, !dbg !36
  br i1 %50, label %L.LB1_352, label %L.LB1_353, !dbg !36

L.LB1_353:                                        ; preds = %L.LB1_352, %L.LB1_374
  %51 = bitcast i32** %.Z0965_319 to i8*, !dbg !37
  %52 = bitcast %astruct.dt68* %.uplevelArgPack0001_395 to i8**, !dbg !37
  store i8* %51, i8** %52, align 8, !dbg !37
  %53 = bitcast i32** %.Z0965_319 to i8*, !dbg !37
  %54 = bitcast %astruct.dt68* %.uplevelArgPack0001_395 to i8*, !dbg !37
  %55 = getelementptr i8, i8* %54, i64 8, !dbg !37
  %56 = bitcast i8* %55 to i8**, !dbg !37
  store i8* %53, i8** %56, align 8, !dbg !37
  %57 = bitcast i64* %z_b_0_311 to i8*, !dbg !37
  %58 = bitcast %astruct.dt68* %.uplevelArgPack0001_395 to i8*, !dbg !37
  %59 = getelementptr i8, i8* %58, i64 16, !dbg !37
  %60 = bitcast i8* %59 to i8**, !dbg !37
  store i8* %57, i8** %60, align 8, !dbg !37
  %61 = bitcast i64* %z_b_1_312 to i8*, !dbg !37
  %62 = bitcast %astruct.dt68* %.uplevelArgPack0001_395 to i8*, !dbg !37
  %63 = getelementptr i8, i8* %62, i64 24, !dbg !37
  %64 = bitcast i8* %63 to i8**, !dbg !37
  store i8* %61, i8** %64, align 8, !dbg !37
  %65 = bitcast i64* %z_e_60_315 to i8*, !dbg !37
  %66 = bitcast %astruct.dt68* %.uplevelArgPack0001_395 to i8*, !dbg !37
  %67 = getelementptr i8, i8* %66, i64 32, !dbg !37
  %68 = bitcast i8* %67 to i8**, !dbg !37
  store i8* %65, i8** %68, align 8, !dbg !37
  %69 = bitcast i64* %z_b_2_313 to i8*, !dbg !37
  %70 = bitcast %astruct.dt68* %.uplevelArgPack0001_395 to i8*, !dbg !37
  %71 = getelementptr i8, i8* %70, i64 40, !dbg !37
  %72 = bitcast i8* %71 to i8**, !dbg !37
  store i8* %69, i8** %72, align 8, !dbg !37
  %73 = bitcast i64* %z_b_3_314 to i8*, !dbg !37
  %74 = bitcast %astruct.dt68* %.uplevelArgPack0001_395 to i8*, !dbg !37
  %75 = getelementptr i8, i8* %74, i64 48, !dbg !37
  %76 = bitcast i8* %75 to i8**, !dbg !37
  store i8* %73, i8** %76, align 8, !dbg !37
  %77 = bitcast [16 x i64]* %"a$sd1_343" to i8*, !dbg !37
  %78 = bitcast %astruct.dt68* %.uplevelArgPack0001_395 to i8*, !dbg !37
  %79 = getelementptr i8, i8* %78, i64 56, !dbg !37
  %80 = bitcast i8* %79 to i8**, !dbg !37
  store i8* %77, i8** %80, align 8, !dbg !37
  br label %L.LB1_412, !dbg !37

L.LB1_412:                                        ; preds = %L.LB1_353
  %81 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L25_1_ to i64*, !dbg !37
  %82 = bitcast %astruct.dt68* %.uplevelArgPack0001_395 to i64*, !dbg !37
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %81, i64* %82), !dbg !37
  call void (...) @_mp_bcs_nest(), !dbg !38
  %83 = bitcast i32* @.C330_MAIN_ to i8*, !dbg !38
  %84 = bitcast [58 x i8]* @.C328_MAIN_ to i8*, !dbg !38
  %85 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !38
  call void (i8*, i8*, i64, ...) %85(i8* %83, i8* %84, i64 58), !dbg !38
  %86 = bitcast i32* @.C332_MAIN_ to i8*, !dbg !38
  %87 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !38
  %88 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !38
  %89 = bitcast %struct.STATICS1* @.STATICS1 to i8*, !dbg !38
  %90 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !38
  %91 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %90(i8* %86, i8* null, i8* %87, i8* %88, i8* %89, i8* null, i64 0), !dbg !38
  call void @llvm.dbg.declare(metadata i32* %z__io_334, metadata !39, metadata !DIExpression()), !dbg !10
  store i32 %91, i32* %z__io_334, align 4, !dbg !38
  %92 = bitcast [16 x i64]* %"a$sd1_343" to i8*, !dbg !38
  %93 = getelementptr i8, i8* %92, i64 56, !dbg !38
  %94 = bitcast i8* %93 to i64*, !dbg !38
  %95 = load i64, i64* %94, align 8, !dbg !38
  %96 = load i32*, i32** %.Z0965_319, align 8, !dbg !38
  call void @llvm.dbg.value(metadata i32* %96, metadata !20, metadata !DIExpression()), !dbg !10
  %97 = bitcast i32* %96 to i8*, !dbg !38
  %98 = getelementptr i8, i8* %97, i64 4004, !dbg !38
  %99 = bitcast i8* %98 to i32*, !dbg !38
  %100 = getelementptr i32, i32* %99, i64 %95, !dbg !38
  %101 = load i32, i32* %100, align 4, !dbg !38
  %102 = bitcast i32 (...)* @f90io_sc_i_fmt_write to i32 (i32, i32, ...)*, !dbg !38
  %103 = call i32 (i32, i32, ...) %102(i32 %101, i32 25), !dbg !38
  store i32 %103, i32* %z__io_334, align 4, !dbg !38
  %104 = call i32 (...) @f90io_fmtw_end(), !dbg !38
  store i32 %104, i32* %z__io_334, align 4, !dbg !38
  call void (...) @_mp_ecs_nest(), !dbg !38
  %105 = load i32*, i32** %.Z0965_319, align 8, !dbg !40
  call void @llvm.dbg.value(metadata i32* %105, metadata !20, metadata !DIExpression()), !dbg !10
  %106 = bitcast i32* %105 to i8*, !dbg !40
  %107 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !40
  %108 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !40
  call void (i8*, i8*, i8*, i8*, i64, ...) %108(i8* null, i8* %106, i8* %107, i8* null, i64 0), !dbg !40
  %109 = bitcast i32** %.Z0965_319 to i8**, !dbg !40
  store i8* null, i8** %109, align 8, !dbg !40
  %110 = bitcast [16 x i64]* %"a$sd1_343" to i64*, !dbg !40
  store i64 0, i64* %110, align 8, !dbg !40
  ret void, !dbg !18
}

define internal void @__nv_MAIN__F1L25_1_(i32* %__nv_MAIN__F1L25_1Arg0, i64* %__nv_MAIN__F1L25_1Arg1, i64* %__nv_MAIN__F1L25_1Arg2) #0 !dbg !41 {
L.entry:
  %__gtid___nv_MAIN__F1L25_1__461 = alloca i32, align 4
  %.i0000p_325 = alloca i32, align 4
  %i_324 = alloca i32, align 4
  %.du0002p_358 = alloca i32, align 4
  %.de0002p_359 = alloca i32, align 4
  %.di0002p_360 = alloca i32, align 4
  %.ds0002p_361 = alloca i32, align 4
  %.dl0002p_363 = alloca i32, align 4
  %.dl0002p.copy_455 = alloca i32, align 4
  %.de0002p.copy_456 = alloca i32, align 4
  %.ds0002p.copy_457 = alloca i32, align 4
  %.dX0002p_362 = alloca i32, align 4
  %.dY0002p_357 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L25_1Arg0, metadata !44, metadata !DIExpression()), !dbg !45
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L25_1Arg1, metadata !46, metadata !DIExpression()), !dbg !45
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L25_1Arg2, metadata !47, metadata !DIExpression()), !dbg !45
  call void @llvm.dbg.value(metadata i32 1, metadata !48, metadata !DIExpression()), !dbg !45
  call void @llvm.dbg.value(metadata i32 2, metadata !49, metadata !DIExpression()), !dbg !45
  call void @llvm.dbg.value(metadata i32 0, metadata !50, metadata !DIExpression()), !dbg !45
  call void @llvm.dbg.value(metadata i32 1, metadata !51, metadata !DIExpression()), !dbg !45
  call void @llvm.dbg.value(metadata i32 2, metadata !52, metadata !DIExpression()), !dbg !45
  call void @llvm.dbg.value(metadata i32 0, metadata !53, metadata !DIExpression()), !dbg !45
  call void @llvm.dbg.value(metadata i32 1, metadata !54, metadata !DIExpression()), !dbg !45
  call void @llvm.dbg.value(metadata i32 2, metadata !55, metadata !DIExpression()), !dbg !45
  %0 = load i32, i32* %__nv_MAIN__F1L25_1Arg0, align 4, !dbg !56
  store i32 %0, i32* %__gtid___nv_MAIN__F1L25_1__461, align 4, !dbg !56
  br label %L.LB2_447

L.LB2_447:                                        ; preds = %L.entry
  br label %L.LB2_322

L.LB2_322:                                        ; preds = %L.LB2_447
  store i32 0, i32* %.i0000p_325, align 4, !dbg !57
  call void @llvm.dbg.declare(metadata i32* %i_324, metadata !58, metadata !DIExpression()), !dbg !56
  store i32 1, i32* %i_324, align 4, !dbg !57
  store i32 1000, i32* %.du0002p_358, align 4, !dbg !57
  store i32 1000, i32* %.de0002p_359, align 4, !dbg !57
  store i32 1, i32* %.di0002p_360, align 4, !dbg !57
  %1 = load i32, i32* %.di0002p_360, align 4, !dbg !57
  store i32 %1, i32* %.ds0002p_361, align 4, !dbg !57
  store i32 1, i32* %.dl0002p_363, align 4, !dbg !57
  %2 = load i32, i32* %.dl0002p_363, align 4, !dbg !57
  store i32 %2, i32* %.dl0002p.copy_455, align 4, !dbg !57
  %3 = load i32, i32* %.de0002p_359, align 4, !dbg !57
  store i32 %3, i32* %.de0002p.copy_456, align 4, !dbg !57
  %4 = load i32, i32* %.ds0002p_361, align 4, !dbg !57
  store i32 %4, i32* %.ds0002p.copy_457, align 4, !dbg !57
  %5 = load i32, i32* %__gtid___nv_MAIN__F1L25_1__461, align 4, !dbg !57
  %6 = bitcast i32* %.i0000p_325 to i64*, !dbg !57
  %7 = bitcast i32* %.dl0002p.copy_455 to i64*, !dbg !57
  %8 = bitcast i32* %.de0002p.copy_456 to i64*, !dbg !57
  %9 = bitcast i32* %.ds0002p.copy_457 to i64*, !dbg !57
  %10 = load i32, i32* %.ds0002p.copy_457, align 4, !dbg !57
  call void @__kmpc_for_static_init_4(i64* null, i32 %5, i32 34, i64* %6, i64* %7, i64* %8, i64* %9, i32 %10, i32 1), !dbg !57
  %11 = load i32, i32* %.dl0002p.copy_455, align 4, !dbg !57
  store i32 %11, i32* %.dl0002p_363, align 4, !dbg !57
  %12 = load i32, i32* %.de0002p.copy_456, align 4, !dbg !57
  store i32 %12, i32* %.de0002p_359, align 4, !dbg !57
  %13 = load i32, i32* %.ds0002p.copy_457, align 4, !dbg !57
  store i32 %13, i32* %.ds0002p_361, align 4, !dbg !57
  %14 = load i32, i32* %.dl0002p_363, align 4, !dbg !57
  store i32 %14, i32* %i_324, align 4, !dbg !57
  %15 = load i32, i32* %i_324, align 4, !dbg !57
  call void @llvm.dbg.value(metadata i32 %15, metadata !58, metadata !DIExpression()), !dbg !56
  store i32 %15, i32* %.dX0002p_362, align 4, !dbg !57
  %16 = load i32, i32* %.dX0002p_362, align 4, !dbg !57
  %17 = load i32, i32* %.du0002p_358, align 4, !dbg !57
  %18 = icmp sgt i32 %16, %17, !dbg !57
  br i1 %18, label %L.LB2_356, label %L.LB2_485, !dbg !57

L.LB2_485:                                        ; preds = %L.LB2_322
  %19 = load i32, i32* %.dX0002p_362, align 4, !dbg !57
  store i32 %19, i32* %i_324, align 4, !dbg !57
  %20 = load i32, i32* %.di0002p_360, align 4, !dbg !57
  %21 = load i32, i32* %.de0002p_359, align 4, !dbg !57
  %22 = load i32, i32* %.dX0002p_362, align 4, !dbg !57
  %23 = sub nsw i32 %21, %22, !dbg !57
  %24 = add nsw i32 %20, %23, !dbg !57
  %25 = load i32, i32* %.di0002p_360, align 4, !dbg !57
  %26 = sdiv i32 %24, %25, !dbg !57
  store i32 %26, i32* %.dY0002p_357, align 4, !dbg !57
  %27 = load i32, i32* %.dY0002p_357, align 4, !dbg !57
  %28 = icmp sle i32 %27, 0, !dbg !57
  br i1 %28, label %L.LB2_366, label %L.LB2_365, !dbg !57

L.LB2_365:                                        ; preds = %L.LB2_365, %L.LB2_485
  %29 = load i32, i32* %i_324, align 4, !dbg !59
  call void @llvm.dbg.value(metadata i32 %29, metadata !58, metadata !DIExpression()), !dbg !56
  %30 = sext i32 %29 to i64, !dbg !59
  %31 = bitcast i64* %__nv_MAIN__F1L25_1Arg2 to i8*, !dbg !59
  %32 = getelementptr i8, i8* %31, i64 56, !dbg !59
  %33 = bitcast i8* %32 to i8**, !dbg !59
  %34 = load i8*, i8** %33, align 8, !dbg !59
  %35 = getelementptr i8, i8* %34, i64 56, !dbg !59
  %36 = bitcast i8* %35 to i64*, !dbg !59
  %37 = load i64, i64* %36, align 8, !dbg !59
  %38 = add nsw i64 %30, %37, !dbg !59
  %39 = bitcast i64* %__nv_MAIN__F1L25_1Arg2 to i8*, !dbg !59
  %40 = getelementptr i8, i8* %39, i64 8, !dbg !59
  %41 = bitcast i8* %40 to i8***, !dbg !59
  %42 = load i8**, i8*** %41, align 8, !dbg !59
  %43 = load i8*, i8** %42, align 8, !dbg !59
  %44 = getelementptr i8, i8* %43, i64 -4, !dbg !59
  %45 = bitcast i8* %44 to i32*, !dbg !59
  %46 = getelementptr i32, i32* %45, i64 %38, !dbg !59
  %47 = load i32, i32* %46, align 4, !dbg !59
  %48 = add nsw i32 %47, 1, !dbg !59
  %49 = bitcast i64* %__nv_MAIN__F1L25_1Arg2 to i8*, !dbg !59
  %50 = getelementptr i8, i8* %49, i64 56, !dbg !59
  %51 = bitcast i8* %50 to i8**, !dbg !59
  %52 = load i8*, i8** %51, align 8, !dbg !59
  %53 = getelementptr i8, i8* %52, i64 56, !dbg !59
  %54 = bitcast i8* %53 to i64*, !dbg !59
  %55 = load i64, i64* %54, align 8, !dbg !59
  %56 = load i32, i32* %i_324, align 4, !dbg !59
  call void @llvm.dbg.value(metadata i32 %56, metadata !58, metadata !DIExpression()), !dbg !56
  %57 = mul nsw i32 %56, 2, !dbg !59
  %58 = sext i32 %57 to i64, !dbg !59
  %59 = add nsw i64 %55, %58, !dbg !59
  %60 = bitcast i64* %__nv_MAIN__F1L25_1Arg2 to i8*, !dbg !59
  %61 = getelementptr i8, i8* %60, i64 8, !dbg !59
  %62 = bitcast i8* %61 to i8***, !dbg !59
  %63 = load i8**, i8*** %62, align 8, !dbg !59
  %64 = load i8*, i8** %63, align 8, !dbg !59
  %65 = getelementptr i8, i8* %64, i64 -4, !dbg !59
  %66 = bitcast i8* %65 to i32*, !dbg !59
  %67 = getelementptr i32, i32* %66, i64 %59, !dbg !59
  store i32 %48, i32* %67, align 4, !dbg !59
  %68 = load i32, i32* %.di0002p_360, align 4, !dbg !56
  %69 = load i32, i32* %i_324, align 4, !dbg !56
  call void @llvm.dbg.value(metadata i32 %69, metadata !58, metadata !DIExpression()), !dbg !56
  %70 = add nsw i32 %68, %69, !dbg !56
  store i32 %70, i32* %i_324, align 4, !dbg !56
  %71 = load i32, i32* %.dY0002p_357, align 4, !dbg !56
  %72 = sub nsw i32 %71, 1, !dbg !56
  store i32 %72, i32* %.dY0002p_357, align 4, !dbg !56
  %73 = load i32, i32* %.dY0002p_357, align 4, !dbg !56
  %74 = icmp sgt i32 %73, 0, !dbg !56
  br i1 %74, label %L.LB2_365, label %L.LB2_366, !dbg !56

L.LB2_366:                                        ; preds = %L.LB2_365, %L.LB2_485
  br label %L.LB2_356

L.LB2_356:                                        ; preds = %L.LB2_366, %L.LB2_322
  %75 = load i32, i32* %__gtid___nv_MAIN__F1L25_1__461, align 4, !dbg !56
  call void @__kmpc_for_static_fini(i64* null, i32 %75), !dbg !56
  br label %L.LB2_326

L.LB2_326:                                        ; preds = %L.LB2_356
  ret void, !dbg !56
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
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB033-truedeplinear-orig-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb033_truedeplinear_orig_yes", scope: !2, file: !3, line: 11, type: !6, scopeLine: 11, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
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
!18 = !DILocation(line: 35, column: 1, scope: !5)
!19 = !DILocation(line: 11, column: 1, scope: !5)
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
!30 = !DILocation(line: 18, column: 1, scope: !5)
!31 = !DILocalVariable(scope: !5, file: !3, type: !26, flags: DIFlagArtificial)
!32 = !DILocation(line: 19, column: 1, scope: !5)
!33 = !DILocation(line: 21, column: 1, scope: !5)
!34 = !DILocalVariable(name: "i", scope: !5, file: !3, type: !9)
!35 = !DILocation(line: 22, column: 1, scope: !5)
!36 = !DILocation(line: 23, column: 1, scope: !5)
!37 = !DILocation(line: 25, column: 1, scope: !5)
!38 = !DILocation(line: 31, column: 1, scope: !5)
!39 = !DILocalVariable(scope: !5, file: !3, type: !9, flags: DIFlagArtificial)
!40 = !DILocation(line: 34, column: 1, scope: !5)
!41 = distinct !DISubprogram(name: "__nv_MAIN__F1L25_1", scope: !2, file: !3, line: 25, type: !42, scopeLine: 25, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!42 = !DISubroutineType(types: !43)
!43 = !{null, !9, !26, !26}
!44 = !DILocalVariable(name: "__nv_MAIN__F1L25_1Arg0", arg: 1, scope: !41, file: !3, type: !9)
!45 = !DILocation(line: 0, scope: !41)
!46 = !DILocalVariable(name: "__nv_MAIN__F1L25_1Arg1", arg: 2, scope: !41, file: !3, type: !26)
!47 = !DILocalVariable(name: "__nv_MAIN__F1L25_1Arg2", arg: 3, scope: !41, file: !3, type: !26)
!48 = !DILocalVariable(name: "omp_sched_static", scope: !41, file: !3, type: !9)
!49 = !DILocalVariable(name: "omp_sched_dynamic", scope: !41, file: !3, type: !9)
!50 = !DILocalVariable(name: "omp_proc_bind_false", scope: !41, file: !3, type: !9)
!51 = !DILocalVariable(name: "omp_proc_bind_true", scope: !41, file: !3, type: !9)
!52 = !DILocalVariable(name: "omp_proc_bind_master", scope: !41, file: !3, type: !9)
!53 = !DILocalVariable(name: "omp_lock_hint_none", scope: !41, file: !3, type: !9)
!54 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !41, file: !3, type: !9)
!55 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !41, file: !3, type: !9)
!56 = !DILocation(line: 28, column: 1, scope: !41)
!57 = !DILocation(line: 26, column: 1, scope: !41)
!58 = !DILocalVariable(name: "i", scope: !41, file: !3, type: !9)
!59 = !DILocation(line: 27, column: 1, scope: !41)

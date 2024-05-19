; ModuleID = '/tmp/DRB104-nowait-barrier-orig-no-cb2820.ll'
source_filename = "/tmp/DRB104-nowait-barrier-orig-no-cb2820.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.STATICS1 = type <{ [52 x i8] }>
%astruct.dt68 = type <{ i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8* }>

@.STATICS1 = internal global %struct.STATICS1 <{ [52 x i8] c"\FB\FF\FF\FF\07\00\00\00error =\00\00\00\00\00\03\00\00\00\EB\FF\FF\FF\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\FF\FF\FF\FF\00\00\00\00" }>, align 16
@.C335_MAIN_ = internal constant i32 6
@.C331_MAIN_ = internal constant [58 x i8] c"micro-benchmarks-fortran/DRB104-nowait-barrier-orig-no.f95"
@.C333_MAIN_ = internal constant i32 42
@.C328_MAIN_ = internal constant i64 9
@.C285_MAIN_ = internal constant i32 1
@.C305_MAIN_ = internal constant i32 25
@.C347_MAIN_ = internal constant i64 4
@.C346_MAIN_ = internal constant i64 25
@.C317_MAIN_ = internal constant i32 5
@.C315_MAIN_ = internal constant i32 1000
@.C286_MAIN_ = internal constant i64 1
@.C284_MAIN_ = internal constant i64 0
@.C283_MAIN_ = internal constant i32 0
@.C317___nv_MAIN__F1L29_1 = internal constant i32 5
@.C285___nv_MAIN__F1L29_1 = internal constant i32 1
@.C283___nv_MAIN__F1L29_1 = internal constant i32 0

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__422 = alloca i32, align 4
  %.Z0967_318 = alloca i32*, align 8
  %"a$sd1_345" = alloca [16 x i64], align 8
  %len_316 = alloca i32, align 4
  %b_308 = alloca i32, align 4
  %z_b_0_309 = alloca i64, align 8
  %z_b_1_310 = alloca i64, align 8
  %z_e_60_313 = alloca i64, align 8
  %z_b_2_311 = alloca i64, align 8
  %z_b_3_312 = alloca i64, align 8
  %.dY0001_356 = alloca i32, align 4
  %i_306 = alloca i32, align 4
  %.uplevelArgPack0001_399 = alloca %astruct.dt68, align 16
  %.s0000_439 = alloca i32, align 4
  %.s0001_440 = alloca i32, align 4
  %error_307 = alloca i32, align 4
  %z__io_337 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !14, metadata !DIExpression()), !dbg !10
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !15
  store i32 %0, i32* %__gtid_MAIN__422, align 4, !dbg !15
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !16
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !16
  call void (i8*, ...) %2(i8* %1), !dbg !16
  call void @llvm.dbg.declare(metadata i32** %.Z0967_318, metadata !17, metadata !DIExpression(DW_OP_deref)), !dbg !10
  %3 = bitcast i32** %.Z0967_318 to i8**, !dbg !16
  store i8* null, i8** %3, align 8, !dbg !16
  call void @llvm.dbg.declare(metadata [16 x i64]* %"a$sd1_345", metadata !21, metadata !DIExpression()), !dbg !10
  %4 = bitcast [16 x i64]* %"a$sd1_345" to i64*, !dbg !16
  store i64 0, i64* %4, align 8, !dbg !16
  br label %L.LB1_377

L.LB1_377:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %len_316, metadata !26, metadata !DIExpression()), !dbg !10
  store i32 1000, i32* %len_316, align 4, !dbg !27
  call void @llvm.dbg.declare(metadata i32* %b_308, metadata !28, metadata !DIExpression()), !dbg !10
  store i32 5, i32* %b_308, align 4, !dbg !29
  call void @llvm.dbg.declare(metadata i64* %z_b_0_309, metadata !30, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_0_309, align 8, !dbg !31
  %5 = load i32, i32* %len_316, align 4, !dbg !31
  call void @llvm.dbg.value(metadata i32 %5, metadata !26, metadata !DIExpression()), !dbg !10
  %6 = sext i32 %5 to i64, !dbg !31
  call void @llvm.dbg.declare(metadata i64* %z_b_1_310, metadata !30, metadata !DIExpression()), !dbg !10
  store i64 %6, i64* %z_b_1_310, align 8, !dbg !31
  %7 = load i64, i64* %z_b_1_310, align 8, !dbg !31
  call void @llvm.dbg.value(metadata i64 %7, metadata !30, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_60_313, metadata !30, metadata !DIExpression()), !dbg !10
  store i64 %7, i64* %z_e_60_313, align 8, !dbg !31
  %8 = bitcast [16 x i64]* %"a$sd1_345" to i8*, !dbg !31
  %9 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !31
  %10 = bitcast i64* @.C346_MAIN_ to i8*, !dbg !31
  %11 = bitcast i64* @.C347_MAIN_ to i8*, !dbg !31
  %12 = bitcast i64* %z_b_0_309 to i8*, !dbg !31
  %13 = bitcast i64* %z_b_1_310 to i8*, !dbg !31
  %14 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !31
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %14(i8* %8, i8* %9, i8* %10, i8* %11, i8* %12, i8* %13), !dbg !31
  %15 = bitcast [16 x i64]* %"a$sd1_345" to i8*, !dbg !31
  %16 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !31
  call void (i8*, i32, ...) %16(i8* %15, i32 25), !dbg !31
  %17 = load i64, i64* %z_b_1_310, align 8, !dbg !31
  call void @llvm.dbg.value(metadata i64 %17, metadata !30, metadata !DIExpression()), !dbg !10
  %18 = load i64, i64* %z_b_0_309, align 8, !dbg !31
  call void @llvm.dbg.value(metadata i64 %18, metadata !30, metadata !DIExpression()), !dbg !10
  %19 = sub nsw i64 %18, 1, !dbg !31
  %20 = sub nsw i64 %17, %19, !dbg !31
  call void @llvm.dbg.declare(metadata i64* %z_b_2_311, metadata !30, metadata !DIExpression()), !dbg !10
  store i64 %20, i64* %z_b_2_311, align 8, !dbg !31
  %21 = load i64, i64* %z_b_0_309, align 8, !dbg !31
  call void @llvm.dbg.value(metadata i64 %21, metadata !30, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_b_3_312, metadata !30, metadata !DIExpression()), !dbg !10
  store i64 %21, i64* %z_b_3_312, align 8, !dbg !31
  %22 = bitcast i64* %z_b_2_311 to i8*, !dbg !31
  %23 = bitcast i64* @.C346_MAIN_ to i8*, !dbg !31
  %24 = bitcast i64* @.C347_MAIN_ to i8*, !dbg !31
  %25 = bitcast i32** %.Z0967_318 to i8*, !dbg !31
  %26 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !31
  %27 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !31
  %28 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !31
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %28(i8* %22, i8* %23, i8* %24, i8* null, i8* %25, i8* null, i8* %26, i8* %27, i8* null, i64 0), !dbg !31
  %29 = load i32, i32* %len_316, align 4, !dbg !32
  call void @llvm.dbg.value(metadata i32 %29, metadata !26, metadata !DIExpression()), !dbg !10
  store i32 %29, i32* %.dY0001_356, align 4, !dbg !32
  call void @llvm.dbg.declare(metadata i32* %i_306, metadata !33, metadata !DIExpression()), !dbg !10
  store i32 1, i32* %i_306, align 4, !dbg !32
  %30 = load i32, i32* %.dY0001_356, align 4, !dbg !32
  %31 = icmp sle i32 %30, 0, !dbg !32
  br i1 %31, label %L.LB1_355, label %L.LB1_354, !dbg !32

L.LB1_354:                                        ; preds = %L.LB1_354, %L.LB1_377
  %32 = load i32, i32* %i_306, align 4, !dbg !34
  call void @llvm.dbg.value(metadata i32 %32, metadata !33, metadata !DIExpression()), !dbg !10
  %33 = load i32, i32* %i_306, align 4, !dbg !34
  call void @llvm.dbg.value(metadata i32 %33, metadata !33, metadata !DIExpression()), !dbg !10
  %34 = sext i32 %33 to i64, !dbg !34
  %35 = bitcast [16 x i64]* %"a$sd1_345" to i8*, !dbg !34
  %36 = getelementptr i8, i8* %35, i64 56, !dbg !34
  %37 = bitcast i8* %36 to i64*, !dbg !34
  %38 = load i64, i64* %37, align 8, !dbg !34
  %39 = add nsw i64 %34, %38, !dbg !34
  %40 = load i32*, i32** %.Z0967_318, align 8, !dbg !34
  call void @llvm.dbg.value(metadata i32* %40, metadata !17, metadata !DIExpression()), !dbg !10
  %41 = bitcast i32* %40 to i8*, !dbg !34
  %42 = getelementptr i8, i8* %41, i64 -4, !dbg !34
  %43 = bitcast i8* %42 to i32*, !dbg !34
  %44 = getelementptr i32, i32* %43, i64 %39, !dbg !34
  store i32 %32, i32* %44, align 4, !dbg !34
  %45 = load i32, i32* %i_306, align 4, !dbg !35
  call void @llvm.dbg.value(metadata i32 %45, metadata !33, metadata !DIExpression()), !dbg !10
  %46 = add nsw i32 %45, 1, !dbg !35
  store i32 %46, i32* %i_306, align 4, !dbg !35
  %47 = load i32, i32* %.dY0001_356, align 4, !dbg !35
  %48 = sub nsw i32 %47, 1, !dbg !35
  store i32 %48, i32* %.dY0001_356, align 4, !dbg !35
  %49 = load i32, i32* %.dY0001_356, align 4, !dbg !35
  %50 = icmp sgt i32 %49, 0, !dbg !35
  br i1 %50, label %L.LB1_354, label %L.LB1_355, !dbg !35

L.LB1_355:                                        ; preds = %L.LB1_354, %L.LB1_377
  %51 = bitcast i32* %len_316 to i8*, !dbg !36
  %52 = bitcast %astruct.dt68* %.uplevelArgPack0001_399 to i8**, !dbg !36
  store i8* %51, i8** %52, align 8, !dbg !36
  %53 = bitcast i32** %.Z0967_318 to i8*, !dbg !36
  %54 = bitcast %astruct.dt68* %.uplevelArgPack0001_399 to i8*, !dbg !36
  %55 = getelementptr i8, i8* %54, i64 8, !dbg !36
  %56 = bitcast i8* %55 to i8**, !dbg !36
  store i8* %53, i8** %56, align 8, !dbg !36
  %57 = bitcast i32** %.Z0967_318 to i8*, !dbg !36
  %58 = bitcast %astruct.dt68* %.uplevelArgPack0001_399 to i8*, !dbg !36
  %59 = getelementptr i8, i8* %58, i64 16, !dbg !36
  %60 = bitcast i8* %59 to i8**, !dbg !36
  store i8* %57, i8** %60, align 8, !dbg !36
  %61 = bitcast i64* %z_b_0_309 to i8*, !dbg !36
  %62 = bitcast %astruct.dt68* %.uplevelArgPack0001_399 to i8*, !dbg !36
  %63 = getelementptr i8, i8* %62, i64 24, !dbg !36
  %64 = bitcast i8* %63 to i8**, !dbg !36
  store i8* %61, i8** %64, align 8, !dbg !36
  %65 = bitcast i64* %z_b_1_310 to i8*, !dbg !36
  %66 = bitcast %astruct.dt68* %.uplevelArgPack0001_399 to i8*, !dbg !36
  %67 = getelementptr i8, i8* %66, i64 32, !dbg !36
  %68 = bitcast i8* %67 to i8**, !dbg !36
  store i8* %65, i8** %68, align 8, !dbg !36
  %69 = bitcast i64* %z_e_60_313 to i8*, !dbg !36
  %70 = bitcast %astruct.dt68* %.uplevelArgPack0001_399 to i8*, !dbg !36
  %71 = getelementptr i8, i8* %70, i64 40, !dbg !36
  %72 = bitcast i8* %71 to i8**, !dbg !36
  store i8* %69, i8** %72, align 8, !dbg !36
  %73 = bitcast i64* %z_b_2_311 to i8*, !dbg !36
  %74 = bitcast %astruct.dt68* %.uplevelArgPack0001_399 to i8*, !dbg !36
  %75 = getelementptr i8, i8* %74, i64 48, !dbg !36
  %76 = bitcast i8* %75 to i8**, !dbg !36
  store i8* %73, i8** %76, align 8, !dbg !36
  %77 = bitcast i64* %z_b_3_312 to i8*, !dbg !36
  %78 = bitcast %astruct.dt68* %.uplevelArgPack0001_399 to i8*, !dbg !36
  %79 = getelementptr i8, i8* %78, i64 56, !dbg !36
  %80 = bitcast i8* %79 to i8**, !dbg !36
  store i8* %77, i8** %80, align 8, !dbg !36
  %81 = bitcast i32* %b_308 to i8*, !dbg !36
  %82 = bitcast %astruct.dt68* %.uplevelArgPack0001_399 to i8*, !dbg !36
  %83 = getelementptr i8, i8* %82, i64 64, !dbg !36
  %84 = bitcast i8* %83 to i8**, !dbg !36
  store i8* %81, i8** %84, align 8, !dbg !36
  %85 = bitcast [16 x i64]* %"a$sd1_345" to i8*, !dbg !36
  %86 = bitcast %astruct.dt68* %.uplevelArgPack0001_399 to i8*, !dbg !36
  %87 = getelementptr i8, i8* %86, i64 72, !dbg !36
  %88 = bitcast i8* %87 to i8**, !dbg !36
  store i8* %85, i8** %88, align 8, !dbg !36
  br label %L.LB1_420, !dbg !36

L.LB1_420:                                        ; preds = %L.LB1_355
  %89 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L29_1_ to i64*, !dbg !36
  %90 = bitcast %astruct.dt68* %.uplevelArgPack0001_399 to i64*, !dbg !36
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %89, i64* %90), !dbg !36
  %91 = load i32, i32* %__gtid_MAIN__422, align 4, !dbg !37
  call void @__kmpc_barrier(i64* null, i32 %91), !dbg !37
  store i32 -1, i32* %.s0000_439, align 4, !dbg !38
  store i32 0, i32* %.s0001_440, align 4, !dbg !38
  %92 = load i32, i32* %__gtid_MAIN__422, align 4, !dbg !38
  %93 = call i32 @__kmpc_single(i64* null, i32 %92), !dbg !38
  %94 = icmp eq i32 %93, 0, !dbg !38
  br i1 %94, label %L.LB1_369, label %L.LB1_327, !dbg !38

L.LB1_327:                                        ; preds = %L.LB1_420
  %95 = bitcast [16 x i64]* %"a$sd1_345" to i8*, !dbg !39
  %96 = getelementptr i8, i8* %95, i64 56, !dbg !39
  %97 = bitcast i8* %96 to i64*, !dbg !39
  %98 = load i64, i64* %97, align 8, !dbg !39
  %99 = load i32*, i32** %.Z0967_318, align 8, !dbg !39
  call void @llvm.dbg.value(metadata i32* %99, metadata !17, metadata !DIExpression()), !dbg !10
  %100 = bitcast i32* %99 to i8*, !dbg !39
  %101 = getelementptr i8, i8* %100, i64 32, !dbg !39
  %102 = bitcast i8* %101 to i32*, !dbg !39
  %103 = getelementptr i32, i32* %102, i64 %98, !dbg !39
  %104 = load i32, i32* %103, align 4, !dbg !39
  %105 = add nsw i32 %104, 1, !dbg !39
  call void @llvm.dbg.declare(metadata i32* %error_307, metadata !40, metadata !DIExpression()), !dbg !10
  store i32 %105, i32* %error_307, align 4, !dbg !39
  %106 = load i32, i32* %__gtid_MAIN__422, align 4, !dbg !41
  store i32 %106, i32* %.s0000_439, align 4, !dbg !41
  store i32 1, i32* %.s0001_440, align 4, !dbg !41
  %107 = load i32, i32* %__gtid_MAIN__422, align 4, !dbg !41
  call void @__kmpc_end_single(i64* null, i32 %107), !dbg !41
  br label %L.LB1_369

L.LB1_369:                                        ; preds = %L.LB1_327, %L.LB1_420
  br label %L.LB1_329

L.LB1_329:                                        ; preds = %L.LB1_369
  %108 = load i32, i32* %__gtid_MAIN__422, align 4, !dbg !41
  call void @__kmpc_barrier(i64* null, i32 %108), !dbg !41
  call void (...) @_mp_bcs_nest(), !dbg !42
  %109 = bitcast i32* @.C333_MAIN_ to i8*, !dbg !42
  %110 = bitcast [58 x i8]* @.C331_MAIN_ to i8*, !dbg !42
  %111 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !42
  call void (i8*, i8*, i64, ...) %111(i8* %109, i8* %110, i64 58), !dbg !42
  %112 = bitcast i32* @.C335_MAIN_ to i8*, !dbg !42
  %113 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !42
  %114 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !42
  %115 = bitcast %struct.STATICS1* @.STATICS1 to i8*, !dbg !42
  %116 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !42
  %117 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %116(i8* %112, i8* null, i8* %113, i8* %114, i8* %115, i8* null, i64 0), !dbg !42
  call void @llvm.dbg.declare(metadata i32* %z__io_337, metadata !43, metadata !DIExpression()), !dbg !10
  store i32 %117, i32* %z__io_337, align 4, !dbg !42
  %118 = load i32, i32* %error_307, align 4, !dbg !42
  call void @llvm.dbg.value(metadata i32 %118, metadata !40, metadata !DIExpression()), !dbg !10
  %119 = bitcast i32 (...)* @f90io_sc_i_fmt_write to i32 (i32, i32, ...)*, !dbg !42
  %120 = call i32 (i32, i32, ...) %119(i32 %118, i32 25), !dbg !42
  store i32 %120, i32* %z__io_337, align 4, !dbg !42
  %121 = call i32 (...) @f90io_fmtw_end(), !dbg !42
  store i32 %121, i32* %z__io_337, align 4, !dbg !42
  call void (...) @_mp_ecs_nest(), !dbg !42
  %122 = load i32*, i32** %.Z0967_318, align 8, !dbg !44
  call void @llvm.dbg.value(metadata i32* %122, metadata !17, metadata !DIExpression()), !dbg !10
  %123 = bitcast i32* %122 to i8*, !dbg !44
  %124 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !44
  %125 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !44
  call void (i8*, i8*, i8*, i8*, i64, ...) %125(i8* null, i8* %123, i8* %124, i8* null, i64 0), !dbg !44
  %126 = bitcast i32** %.Z0967_318 to i8**, !dbg !44
  store i8* null, i8** %126, align 8, !dbg !44
  %127 = bitcast [16 x i64]* %"a$sd1_345" to i64*, !dbg !44
  store i64 0, i64* %127, align 8, !dbg !44
  ret void, !dbg !15
}

define internal void @__nv_MAIN__F1L29_1_(i32* %__nv_MAIN__F1L29_1Arg0, i64* %__nv_MAIN__F1L29_1Arg1, i64* %__nv_MAIN__F1L29_1Arg2) #0 !dbg !45 {
L.entry:
  %__gtid___nv_MAIN__F1L29_1__485 = alloca i32, align 4
  %.i0000p_324 = alloca i32, align 4
  %i_323 = alloca i32, align 4
  %.du0002p_360 = alloca i32, align 4
  %.de0002p_361 = alloca i32, align 4
  %.di0002p_362 = alloca i32, align 4
  %.ds0002p_363 = alloca i32, align 4
  %.dl0002p_365 = alloca i32, align 4
  %.dl0002p.copy_479 = alloca i32, align 4
  %.de0002p.copy_480 = alloca i32, align 4
  %.ds0002p.copy_481 = alloca i32, align 4
  %.dX0002p_364 = alloca i32, align 4
  %.dY0002p_359 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L29_1Arg0, metadata !48, metadata !DIExpression()), !dbg !49
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L29_1Arg1, metadata !50, metadata !DIExpression()), !dbg !49
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L29_1Arg2, metadata !51, metadata !DIExpression()), !dbg !49
  call void @llvm.dbg.value(metadata i32 1, metadata !52, metadata !DIExpression()), !dbg !49
  call void @llvm.dbg.value(metadata i32 0, metadata !53, metadata !DIExpression()), !dbg !49
  call void @llvm.dbg.value(metadata i32 1, metadata !54, metadata !DIExpression()), !dbg !49
  call void @llvm.dbg.value(metadata i32 0, metadata !55, metadata !DIExpression()), !dbg !49
  call void @llvm.dbg.value(metadata i32 1, metadata !56, metadata !DIExpression()), !dbg !49
  %0 = load i32, i32* %__nv_MAIN__F1L29_1Arg0, align 4, !dbg !57
  store i32 %0, i32* %__gtid___nv_MAIN__F1L29_1__485, align 4, !dbg !57
  br label %L.LB2_470

L.LB2_470:                                        ; preds = %L.entry
  br label %L.LB2_321

L.LB2_321:                                        ; preds = %L.LB2_470
  br label %L.LB2_322

L.LB2_322:                                        ; preds = %L.LB2_321
  store i32 0, i32* %.i0000p_324, align 4, !dbg !58
  call void @llvm.dbg.declare(metadata i32* %i_323, metadata !59, metadata !DIExpression()), !dbg !57
  store i32 1, i32* %i_323, align 4, !dbg !58
  %1 = bitcast i64* %__nv_MAIN__F1L29_1Arg2 to i32**, !dbg !58
  %2 = load i32*, i32** %1, align 8, !dbg !58
  %3 = load i32, i32* %2, align 4, !dbg !58
  store i32 %3, i32* %.du0002p_360, align 4, !dbg !58
  %4 = bitcast i64* %__nv_MAIN__F1L29_1Arg2 to i32**, !dbg !58
  %5 = load i32*, i32** %4, align 8, !dbg !58
  %6 = load i32, i32* %5, align 4, !dbg !58
  store i32 %6, i32* %.de0002p_361, align 4, !dbg !58
  store i32 1, i32* %.di0002p_362, align 4, !dbg !58
  %7 = load i32, i32* %.di0002p_362, align 4, !dbg !58
  store i32 %7, i32* %.ds0002p_363, align 4, !dbg !58
  store i32 1, i32* %.dl0002p_365, align 4, !dbg !58
  %8 = load i32, i32* %.dl0002p_365, align 4, !dbg !58
  store i32 %8, i32* %.dl0002p.copy_479, align 4, !dbg !58
  %9 = load i32, i32* %.de0002p_361, align 4, !dbg !58
  store i32 %9, i32* %.de0002p.copy_480, align 4, !dbg !58
  %10 = load i32, i32* %.ds0002p_363, align 4, !dbg !58
  store i32 %10, i32* %.ds0002p.copy_481, align 4, !dbg !58
  %11 = load i32, i32* %__gtid___nv_MAIN__F1L29_1__485, align 4, !dbg !58
  %12 = bitcast i32* %.i0000p_324 to i64*, !dbg !58
  %13 = bitcast i32* %.dl0002p.copy_479 to i64*, !dbg !58
  %14 = bitcast i32* %.de0002p.copy_480 to i64*, !dbg !58
  %15 = bitcast i32* %.ds0002p.copy_481 to i64*, !dbg !58
  %16 = load i32, i32* %.ds0002p.copy_481, align 4, !dbg !58
  call void @__kmpc_for_static_init_4(i64* null, i32 %11, i32 34, i64* %12, i64* %13, i64* %14, i64* %15, i32 %16, i32 1), !dbg !58
  %17 = load i32, i32* %.dl0002p.copy_479, align 4, !dbg !58
  store i32 %17, i32* %.dl0002p_365, align 4, !dbg !58
  %18 = load i32, i32* %.de0002p.copy_480, align 4, !dbg !58
  store i32 %18, i32* %.de0002p_361, align 4, !dbg !58
  %19 = load i32, i32* %.ds0002p.copy_481, align 4, !dbg !58
  store i32 %19, i32* %.ds0002p_363, align 4, !dbg !58
  %20 = load i32, i32* %.dl0002p_365, align 4, !dbg !58
  store i32 %20, i32* %i_323, align 4, !dbg !58
  %21 = load i32, i32* %i_323, align 4, !dbg !58
  call void @llvm.dbg.value(metadata i32 %21, metadata !59, metadata !DIExpression()), !dbg !57
  store i32 %21, i32* %.dX0002p_364, align 4, !dbg !58
  %22 = load i32, i32* %.dX0002p_364, align 4, !dbg !58
  %23 = load i32, i32* %.du0002p_360, align 4, !dbg !58
  %24 = icmp sgt i32 %22, %23, !dbg !58
  br i1 %24, label %L.LB2_358, label %L.LB2_510, !dbg !58

L.LB2_510:                                        ; preds = %L.LB2_322
  %25 = load i32, i32* %.dX0002p_364, align 4, !dbg !58
  store i32 %25, i32* %i_323, align 4, !dbg !58
  %26 = load i32, i32* %.di0002p_362, align 4, !dbg !58
  %27 = load i32, i32* %.de0002p_361, align 4, !dbg !58
  %28 = load i32, i32* %.dX0002p_364, align 4, !dbg !58
  %29 = sub nsw i32 %27, %28, !dbg !58
  %30 = add nsw i32 %26, %29, !dbg !58
  %31 = load i32, i32* %.di0002p_362, align 4, !dbg !58
  %32 = sdiv i32 %30, %31, !dbg !58
  store i32 %32, i32* %.dY0002p_359, align 4, !dbg !58
  %33 = load i32, i32* %.dY0002p_359, align 4, !dbg !58
  %34 = icmp sle i32 %33, 0, !dbg !58
  br i1 %34, label %L.LB2_368, label %L.LB2_367, !dbg !58

L.LB2_367:                                        ; preds = %L.LB2_367, %L.LB2_510
  %35 = load i32, i32* %i_323, align 4, !dbg !60
  call void @llvm.dbg.value(metadata i32 %35, metadata !59, metadata !DIExpression()), !dbg !57
  %36 = sext i32 %35 to i64, !dbg !60
  %37 = bitcast i64* %__nv_MAIN__F1L29_1Arg2 to i8*, !dbg !60
  %38 = getelementptr i8, i8* %37, i64 72, !dbg !60
  %39 = bitcast i8* %38 to i8**, !dbg !60
  %40 = load i8*, i8** %39, align 8, !dbg !60
  %41 = getelementptr i8, i8* %40, i64 56, !dbg !60
  %42 = bitcast i8* %41 to i64*, !dbg !60
  %43 = load i64, i64* %42, align 8, !dbg !60
  %44 = add nsw i64 %36, %43, !dbg !60
  %45 = bitcast i64* %__nv_MAIN__F1L29_1Arg2 to i8*, !dbg !60
  %46 = getelementptr i8, i8* %45, i64 16, !dbg !60
  %47 = bitcast i8* %46 to i8***, !dbg !60
  %48 = load i8**, i8*** %47, align 8, !dbg !60
  %49 = load i8*, i8** %48, align 8, !dbg !60
  %50 = getelementptr i8, i8* %49, i64 -4, !dbg !60
  %51 = bitcast i8* %50 to i32*, !dbg !60
  %52 = getelementptr i32, i32* %51, i64 %44, !dbg !60
  %53 = load i32, i32* %52, align 4, !dbg !60
  %54 = mul nsw i32 %53, 5, !dbg !60
  %55 = bitcast i64* %__nv_MAIN__F1L29_1Arg2 to i8*, !dbg !60
  %56 = getelementptr i8, i8* %55, i64 64, !dbg !60
  %57 = bitcast i8* %56 to i32**, !dbg !60
  %58 = load i32*, i32** %57, align 8, !dbg !60
  %59 = load i32, i32* %58, align 4, !dbg !60
  %60 = add nsw i32 %54, %59, !dbg !60
  %61 = load i32, i32* %i_323, align 4, !dbg !60
  call void @llvm.dbg.value(metadata i32 %61, metadata !59, metadata !DIExpression()), !dbg !57
  %62 = sext i32 %61 to i64, !dbg !60
  %63 = bitcast i64* %__nv_MAIN__F1L29_1Arg2 to i8*, !dbg !60
  %64 = getelementptr i8, i8* %63, i64 72, !dbg !60
  %65 = bitcast i8* %64 to i8**, !dbg !60
  %66 = load i8*, i8** %65, align 8, !dbg !60
  %67 = getelementptr i8, i8* %66, i64 56, !dbg !60
  %68 = bitcast i8* %67 to i64*, !dbg !60
  %69 = load i64, i64* %68, align 8, !dbg !60
  %70 = add nsw i64 %62, %69, !dbg !60
  %71 = bitcast i64* %__nv_MAIN__F1L29_1Arg2 to i8*, !dbg !60
  %72 = getelementptr i8, i8* %71, i64 16, !dbg !60
  %73 = bitcast i8* %72 to i8***, !dbg !60
  %74 = load i8**, i8*** %73, align 8, !dbg !60
  %75 = load i8*, i8** %74, align 8, !dbg !60
  %76 = getelementptr i8, i8* %75, i64 -4, !dbg !60
  %77 = bitcast i8* %76 to i32*, !dbg !60
  %78 = getelementptr i32, i32* %77, i64 %70, !dbg !60
  store i32 %60, i32* %78, align 4, !dbg !60
  %79 = load i32, i32* %.di0002p_362, align 4, !dbg !61
  %80 = load i32, i32* %i_323, align 4, !dbg !61
  call void @llvm.dbg.value(metadata i32 %80, metadata !59, metadata !DIExpression()), !dbg !57
  %81 = add nsw i32 %79, %80, !dbg !61
  store i32 %81, i32* %i_323, align 4, !dbg !61
  %82 = load i32, i32* %.dY0002p_359, align 4, !dbg !61
  %83 = sub nsw i32 %82, 1, !dbg !61
  store i32 %83, i32* %.dY0002p_359, align 4, !dbg !61
  %84 = load i32, i32* %.dY0002p_359, align 4, !dbg !61
  %85 = icmp sgt i32 %84, 0, !dbg !61
  br i1 %85, label %L.LB2_367, label %L.LB2_368, !dbg !61

L.LB2_368:                                        ; preds = %L.LB2_367, %L.LB2_510
  br label %L.LB2_358

L.LB2_358:                                        ; preds = %L.LB2_368, %L.LB2_322
  %86 = load i32, i32* %__gtid___nv_MAIN__F1L29_1__485, align 4, !dbg !61
  call void @__kmpc_for_static_fini(i64* null, i32 %86), !dbg !61
  br label %L.LB2_325

L.LB2_325:                                        ; preds = %L.LB2_358
  br label %L.LB2_326

L.LB2_326:                                        ; preds = %L.LB2_325
  ret void, !dbg !57
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

declare void @__kmpc_end_single(i64*, i32) #0

declare signext i32 @__kmpc_single(i64*, i32) #0

declare void @__kmpc_barrier(i64*, i32) #0

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
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB104-nowait-barrier-orig-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb104_nowait_barrier_orig_no", scope: !2, file: !3, line: 14, type: !6, scopeLine: 14, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!6 = !DISubroutineType(cc: DW_CC_program, types: !7)
!7 = !{null}
!8 = !DILocalVariable(name: "omp_sched_static", scope: !5, file: !3, type: !9)
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 0, scope: !5)
!11 = !DILocalVariable(name: "omp_proc_bind_false", scope: !5, file: !3, type: !9)
!12 = !DILocalVariable(name: "omp_proc_bind_true", scope: !5, file: !3, type: !9)
!13 = !DILocalVariable(name: "omp_lock_hint_none", scope: !5, file: !3, type: !9)
!14 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !5, file: !3, type: !9)
!15 = !DILocation(line: 46, column: 1, scope: !5)
!16 = !DILocation(line: 14, column: 1, scope: !5)
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
!27 = !DILocation(line: 21, column: 1, scope: !5)
!28 = !DILocalVariable(name: "b", scope: !5, file: !3, type: !9)
!29 = !DILocation(line: 22, column: 1, scope: !5)
!30 = !DILocalVariable(scope: !5, file: !3, type: !23, flags: DIFlagArtificial)
!31 = !DILocation(line: 23, column: 1, scope: !5)
!32 = !DILocation(line: 25, column: 1, scope: !5)
!33 = !DILocalVariable(name: "i", scope: !5, file: !3, type: !9)
!34 = !DILocation(line: 26, column: 1, scope: !5)
!35 = !DILocation(line: 27, column: 1, scope: !5)
!36 = !DILocation(line: 29, column: 1, scope: !5)
!37 = !DILocation(line: 37, column: 1, scope: !5)
!38 = !DILocation(line: 38, column: 1, scope: !5)
!39 = !DILocation(line: 39, column: 1, scope: !5)
!40 = !DILocalVariable(name: "error", scope: !5, file: !3, type: !9)
!41 = !DILocation(line: 40, column: 1, scope: !5)
!42 = !DILocation(line: 42, column: 1, scope: !5)
!43 = !DILocalVariable(scope: !5, file: !3, type: !9, flags: DIFlagArtificial)
!44 = !DILocation(line: 45, column: 1, scope: !5)
!45 = distinct !DISubprogram(name: "__nv_MAIN__F1L29_1", scope: !2, file: !3, line: 29, type: !46, scopeLine: 29, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!46 = !DISubroutineType(types: !47)
!47 = !{null, !9, !23, !23}
!48 = !DILocalVariable(name: "__nv_MAIN__F1L29_1Arg0", arg: 1, scope: !45, file: !3, type: !9)
!49 = !DILocation(line: 0, scope: !45)
!50 = !DILocalVariable(name: "__nv_MAIN__F1L29_1Arg1", arg: 2, scope: !45, file: !3, type: !23)
!51 = !DILocalVariable(name: "__nv_MAIN__F1L29_1Arg2", arg: 3, scope: !45, file: !3, type: !23)
!52 = !DILocalVariable(name: "omp_sched_static", scope: !45, file: !3, type: !9)
!53 = !DILocalVariable(name: "omp_proc_bind_false", scope: !45, file: !3, type: !9)
!54 = !DILocalVariable(name: "omp_proc_bind_true", scope: !45, file: !3, type: !9)
!55 = !DILocalVariable(name: "omp_lock_hint_none", scope: !45, file: !3, type: !9)
!56 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !45, file: !3, type: !9)
!57 = !DILocation(line: 35, column: 1, scope: !45)
!58 = !DILocation(line: 31, column: 1, scope: !45)
!59 = !DILocalVariable(name: "i", scope: !45, file: !3, type: !9)
!60 = !DILocation(line: 32, column: 1, scope: !45)
!61 = !DILocation(line: 33, column: 1, scope: !45)

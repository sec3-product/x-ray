; ModuleID = '/tmp/DRB028-privatemissing-orig-yes-8f8669.ll'
source_filename = "/tmp/DRB028-privatemissing-orig-yes-8f8669.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.STATICS1 = type <{ [44 x i8] }>
%astruct.dt68 = type <{ i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8* }>

@.STATICS1 = internal global %struct.STATICS1 <{ [44 x i8] c"\FB\FF\FF\FF\06\00\00\00a(50)=\00\00\EB\FF\FF\FF\00\00\00\00\03\00\00\00\00\00\00\00\01\00\00\00\FF\FF\FF\FF\00\00\00\00" }>, align 16
@.C331_MAIN_ = internal constant i64 50
@.C328_MAIN_ = internal constant i32 6
@.C324_MAIN_ = internal constant [59 x i8] c"micro-benchmarks-fortran/DRB028-privatemissing-orig-yes.f95"
@.C326_MAIN_ = internal constant i32 33
@.C285_MAIN_ = internal constant i32 1
@.C305_MAIN_ = internal constant i32 25
@.C341_MAIN_ = internal constant i64 4
@.C340_MAIN_ = internal constant i64 25
@.C314_MAIN_ = internal constant i32 100
@.C286_MAIN_ = internal constant i64 1
@.C284_MAIN_ = internal constant i64 0
@.C283_MAIN_ = internal constant i32 0
@.C285___nv_MAIN__F1L26_1 = internal constant i32 1
@.C283___nv_MAIN__F1L26_1 = internal constant i32 0

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__415 = alloca i32, align 4
  %.Z0966_316 = alloca i32*, align 8
  %"a$sd1_339" = alloca [16 x i64], align 8
  %len_315 = alloca i32, align 4
  %z_b_0_308 = alloca i64, align 8
  %z_b_1_309 = alloca i64, align 8
  %z_e_60_312 = alloca i64, align 8
  %z_b_2_310 = alloca i64, align 8
  %z_b_3_311 = alloca i64, align 8
  %.dY0001_350 = alloca i32, align 4
  %i_306 = alloca i32, align 4
  %.uplevelArgPack0001_391 = alloca %astruct.dt68, align 16
  %tmp_307 = alloca i32, align 4
  %z__io_330 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !14, metadata !DIExpression()), !dbg !10
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !15
  store i32 %0, i32* %__gtid_MAIN__415, align 4, !dbg !15
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !16
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !16
  call void (i8*, ...) %2(i8* %1), !dbg !16
  call void @llvm.dbg.declare(metadata i32** %.Z0966_316, metadata !17, metadata !DIExpression(DW_OP_deref)), !dbg !10
  %3 = bitcast i32** %.Z0966_316 to i8**, !dbg !16
  store i8* null, i8** %3, align 8, !dbg !16
  call void @llvm.dbg.declare(metadata [16 x i64]* %"a$sd1_339", metadata !21, metadata !DIExpression()), !dbg !10
  %4 = bitcast [16 x i64]* %"a$sd1_339" to i64*, !dbg !16
  store i64 0, i64* %4, align 8, !dbg !16
  br label %L.LB1_370

L.LB1_370:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %len_315, metadata !26, metadata !DIExpression()), !dbg !10
  store i32 100, i32* %len_315, align 4, !dbg !27
  call void @llvm.dbg.declare(metadata i64* %z_b_0_308, metadata !28, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_0_308, align 8, !dbg !29
  %5 = load i32, i32* %len_315, align 4, !dbg !29
  call void @llvm.dbg.value(metadata i32 %5, metadata !26, metadata !DIExpression()), !dbg !10
  %6 = sext i32 %5 to i64, !dbg !29
  call void @llvm.dbg.declare(metadata i64* %z_b_1_309, metadata !28, metadata !DIExpression()), !dbg !10
  store i64 %6, i64* %z_b_1_309, align 8, !dbg !29
  %7 = load i64, i64* %z_b_1_309, align 8, !dbg !29
  call void @llvm.dbg.value(metadata i64 %7, metadata !28, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_60_312, metadata !28, metadata !DIExpression()), !dbg !10
  store i64 %7, i64* %z_e_60_312, align 8, !dbg !29
  %8 = bitcast [16 x i64]* %"a$sd1_339" to i8*, !dbg !29
  %9 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !29
  %10 = bitcast i64* @.C340_MAIN_ to i8*, !dbg !29
  %11 = bitcast i64* @.C341_MAIN_ to i8*, !dbg !29
  %12 = bitcast i64* %z_b_0_308 to i8*, !dbg !29
  %13 = bitcast i64* %z_b_1_309 to i8*, !dbg !29
  %14 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !29
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %14(i8* %8, i8* %9, i8* %10, i8* %11, i8* %12, i8* %13), !dbg !29
  %15 = bitcast [16 x i64]* %"a$sd1_339" to i8*, !dbg !29
  %16 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !29
  call void (i8*, i32, ...) %16(i8* %15, i32 25), !dbg !29
  %17 = load i64, i64* %z_b_1_309, align 8, !dbg !29
  call void @llvm.dbg.value(metadata i64 %17, metadata !28, metadata !DIExpression()), !dbg !10
  %18 = load i64, i64* %z_b_0_308, align 8, !dbg !29
  call void @llvm.dbg.value(metadata i64 %18, metadata !28, metadata !DIExpression()), !dbg !10
  %19 = sub nsw i64 %18, 1, !dbg !29
  %20 = sub nsw i64 %17, %19, !dbg !29
  call void @llvm.dbg.declare(metadata i64* %z_b_2_310, metadata !28, metadata !DIExpression()), !dbg !10
  store i64 %20, i64* %z_b_2_310, align 8, !dbg !29
  %21 = load i64, i64* %z_b_0_308, align 8, !dbg !29
  call void @llvm.dbg.value(metadata i64 %21, metadata !28, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_b_3_311, metadata !28, metadata !DIExpression()), !dbg !10
  store i64 %21, i64* %z_b_3_311, align 8, !dbg !29
  %22 = bitcast i64* %z_b_2_310 to i8*, !dbg !29
  %23 = bitcast i64* @.C340_MAIN_ to i8*, !dbg !29
  %24 = bitcast i64* @.C341_MAIN_ to i8*, !dbg !29
  %25 = bitcast i32** %.Z0966_316 to i8*, !dbg !29
  %26 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !29
  %27 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !29
  %28 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !29
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %28(i8* %22, i8* %23, i8* %24, i8* null, i8* %25, i8* null, i8* %26, i8* %27, i8* null, i64 0), !dbg !29
  %29 = load i32, i32* %len_315, align 4, !dbg !30
  call void @llvm.dbg.value(metadata i32 %29, metadata !26, metadata !DIExpression()), !dbg !10
  store i32 %29, i32* %.dY0001_350, align 4, !dbg !30
  call void @llvm.dbg.declare(metadata i32* %i_306, metadata !31, metadata !DIExpression()), !dbg !10
  store i32 1, i32* %i_306, align 4, !dbg !30
  %30 = load i32, i32* %.dY0001_350, align 4, !dbg !30
  %31 = icmp sle i32 %30, 0, !dbg !30
  br i1 %31, label %L.LB1_349, label %L.LB1_348, !dbg !30

L.LB1_348:                                        ; preds = %L.LB1_348, %L.LB1_370
  %32 = load i32, i32* %i_306, align 4, !dbg !32
  call void @llvm.dbg.value(metadata i32 %32, metadata !31, metadata !DIExpression()), !dbg !10
  %33 = load i32, i32* %i_306, align 4, !dbg !32
  call void @llvm.dbg.value(metadata i32 %33, metadata !31, metadata !DIExpression()), !dbg !10
  %34 = sext i32 %33 to i64, !dbg !32
  %35 = bitcast [16 x i64]* %"a$sd1_339" to i8*, !dbg !32
  %36 = getelementptr i8, i8* %35, i64 56, !dbg !32
  %37 = bitcast i8* %36 to i64*, !dbg !32
  %38 = load i64, i64* %37, align 8, !dbg !32
  %39 = add nsw i64 %34, %38, !dbg !32
  %40 = load i32*, i32** %.Z0966_316, align 8, !dbg !32
  call void @llvm.dbg.value(metadata i32* %40, metadata !17, metadata !DIExpression()), !dbg !10
  %41 = bitcast i32* %40 to i8*, !dbg !32
  %42 = getelementptr i8, i8* %41, i64 -4, !dbg !32
  %43 = bitcast i8* %42 to i32*, !dbg !32
  %44 = getelementptr i32, i32* %43, i64 %39, !dbg !32
  store i32 %32, i32* %44, align 4, !dbg !32
  %45 = load i32, i32* %i_306, align 4, !dbg !33
  call void @llvm.dbg.value(metadata i32 %45, metadata !31, metadata !DIExpression()), !dbg !10
  %46 = add nsw i32 %45, 1, !dbg !33
  store i32 %46, i32* %i_306, align 4, !dbg !33
  %47 = load i32, i32* %.dY0001_350, align 4, !dbg !33
  %48 = sub nsw i32 %47, 1, !dbg !33
  store i32 %48, i32* %.dY0001_350, align 4, !dbg !33
  %49 = load i32, i32* %.dY0001_350, align 4, !dbg !33
  %50 = icmp sgt i32 %49, 0, !dbg !33
  br i1 %50, label %L.LB1_348, label %L.LB1_349, !dbg !33

L.LB1_349:                                        ; preds = %L.LB1_348, %L.LB1_370
  %51 = bitcast i32* %len_315 to i8*, !dbg !34
  %52 = bitcast %astruct.dt68* %.uplevelArgPack0001_391 to i8**, !dbg !34
  store i8* %51, i8** %52, align 8, !dbg !34
  %53 = bitcast i32** %.Z0966_316 to i8*, !dbg !34
  %54 = bitcast %astruct.dt68* %.uplevelArgPack0001_391 to i8*, !dbg !34
  %55 = getelementptr i8, i8* %54, i64 8, !dbg !34
  %56 = bitcast i8* %55 to i8**, !dbg !34
  store i8* %53, i8** %56, align 8, !dbg !34
  %57 = bitcast i32** %.Z0966_316 to i8*, !dbg !34
  %58 = bitcast %astruct.dt68* %.uplevelArgPack0001_391 to i8*, !dbg !34
  %59 = getelementptr i8, i8* %58, i64 16, !dbg !34
  %60 = bitcast i8* %59 to i8**, !dbg !34
  store i8* %57, i8** %60, align 8, !dbg !34
  %61 = bitcast i64* %z_b_0_308 to i8*, !dbg !34
  %62 = bitcast %astruct.dt68* %.uplevelArgPack0001_391 to i8*, !dbg !34
  %63 = getelementptr i8, i8* %62, i64 24, !dbg !34
  %64 = bitcast i8* %63 to i8**, !dbg !34
  store i8* %61, i8** %64, align 8, !dbg !34
  %65 = bitcast i64* %z_b_1_309 to i8*, !dbg !34
  %66 = bitcast %astruct.dt68* %.uplevelArgPack0001_391 to i8*, !dbg !34
  %67 = getelementptr i8, i8* %66, i64 32, !dbg !34
  %68 = bitcast i8* %67 to i8**, !dbg !34
  store i8* %65, i8** %68, align 8, !dbg !34
  %69 = bitcast i64* %z_e_60_312 to i8*, !dbg !34
  %70 = bitcast %astruct.dt68* %.uplevelArgPack0001_391 to i8*, !dbg !34
  %71 = getelementptr i8, i8* %70, i64 40, !dbg !34
  %72 = bitcast i8* %71 to i8**, !dbg !34
  store i8* %69, i8** %72, align 8, !dbg !34
  %73 = bitcast i64* %z_b_2_310 to i8*, !dbg !34
  %74 = bitcast %astruct.dt68* %.uplevelArgPack0001_391 to i8*, !dbg !34
  %75 = getelementptr i8, i8* %74, i64 48, !dbg !34
  %76 = bitcast i8* %75 to i8**, !dbg !34
  store i8* %73, i8** %76, align 8, !dbg !34
  %77 = bitcast i64* %z_b_3_311 to i8*, !dbg !34
  %78 = bitcast %astruct.dt68* %.uplevelArgPack0001_391 to i8*, !dbg !34
  %79 = getelementptr i8, i8* %78, i64 56, !dbg !34
  %80 = bitcast i8* %79 to i8**, !dbg !34
  store i8* %77, i8** %80, align 8, !dbg !34
  call void @llvm.dbg.declare(metadata i32* %tmp_307, metadata !35, metadata !DIExpression()), !dbg !10
  %81 = bitcast i32* %tmp_307 to i8*, !dbg !34
  %82 = bitcast %astruct.dt68* %.uplevelArgPack0001_391 to i8*, !dbg !34
  %83 = getelementptr i8, i8* %82, i64 64, !dbg !34
  %84 = bitcast i8* %83 to i8**, !dbg !34
  store i8* %81, i8** %84, align 8, !dbg !34
  %85 = bitcast [16 x i64]* %"a$sd1_339" to i8*, !dbg !34
  %86 = bitcast %astruct.dt68* %.uplevelArgPack0001_391 to i8*, !dbg !34
  %87 = getelementptr i8, i8* %86, i64 72, !dbg !34
  %88 = bitcast i8* %87 to i8**, !dbg !34
  store i8* %85, i8** %88, align 8, !dbg !34
  br label %L.LB1_413, !dbg !34

L.LB1_413:                                        ; preds = %L.LB1_349
  %89 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L26_1_ to i64*, !dbg !34
  %90 = bitcast %astruct.dt68* %.uplevelArgPack0001_391 to i64*, !dbg !34
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %89, i64* %90), !dbg !34
  call void (...) @_mp_bcs_nest(), !dbg !36
  %91 = bitcast i32* @.C326_MAIN_ to i8*, !dbg !36
  %92 = bitcast [59 x i8]* @.C324_MAIN_ to i8*, !dbg !36
  %93 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !36
  call void (i8*, i8*, i64, ...) %93(i8* %91, i8* %92, i64 59), !dbg !36
  %94 = bitcast i32* @.C328_MAIN_ to i8*, !dbg !36
  %95 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !36
  %96 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !36
  %97 = bitcast %struct.STATICS1* @.STATICS1 to i8*, !dbg !36
  %98 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !36
  %99 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %98(i8* %94, i8* null, i8* %95, i8* %96, i8* %97, i8* null, i64 0), !dbg !36
  call void @llvm.dbg.declare(metadata i32* %z__io_330, metadata !37, metadata !DIExpression()), !dbg !10
  store i32 %99, i32* %z__io_330, align 4, !dbg !36
  %100 = bitcast [16 x i64]* %"a$sd1_339" to i8*, !dbg !36
  %101 = getelementptr i8, i8* %100, i64 56, !dbg !36
  %102 = bitcast i8* %101 to i64*, !dbg !36
  %103 = load i64, i64* %102, align 8, !dbg !36
  %104 = load i32*, i32** %.Z0966_316, align 8, !dbg !36
  call void @llvm.dbg.value(metadata i32* %104, metadata !17, metadata !DIExpression()), !dbg !10
  %105 = bitcast i32* %104 to i8*, !dbg !36
  %106 = getelementptr i8, i8* %105, i64 196, !dbg !36
  %107 = bitcast i8* %106 to i32*, !dbg !36
  %108 = getelementptr i32, i32* %107, i64 %103, !dbg !36
  %109 = load i32, i32* %108, align 4, !dbg !36
  %110 = bitcast i32 (...)* @f90io_sc_i_fmt_write to i32 (i32, i32, ...)*, !dbg !36
  %111 = call i32 (i32, i32, ...) %110(i32 %109, i32 25), !dbg !36
  store i32 %111, i32* %z__io_330, align 4, !dbg !36
  %112 = call i32 (...) @f90io_fmtw_end(), !dbg !36
  store i32 %112, i32* %z__io_330, align 4, !dbg !36
  call void (...) @_mp_ecs_nest(), !dbg !36
  %113 = load i32*, i32** %.Z0966_316, align 8, !dbg !38
  call void @llvm.dbg.value(metadata i32* %113, metadata !17, metadata !DIExpression()), !dbg !10
  %114 = bitcast i32* %113 to i8*, !dbg !38
  %115 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !38
  %116 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !38
  call void (i8*, i8*, i8*, i8*, i64, ...) %116(i8* null, i8* %114, i8* %115, i8* null, i64 0), !dbg !38
  %117 = bitcast i32** %.Z0966_316 to i8**, !dbg !38
  store i8* null, i8** %117, align 8, !dbg !38
  %118 = bitcast [16 x i64]* %"a$sd1_339" to i64*, !dbg !38
  store i64 0, i64* %118, align 8, !dbg !38
  ret void, !dbg !15
}

define internal void @__nv_MAIN__F1L26_1_(i32* %__nv_MAIN__F1L26_1Arg0, i64* %__nv_MAIN__F1L26_1Arg1, i64* %__nv_MAIN__F1L26_1Arg2) #0 !dbg !39 {
L.entry:
  %__gtid___nv_MAIN__F1L26_1__463 = alloca i32, align 4
  %.i0000p_321 = alloca i32, align 4
  %i_320 = alloca i32, align 4
  %.du0002p_354 = alloca i32, align 4
  %.de0002p_355 = alloca i32, align 4
  %.di0002p_356 = alloca i32, align 4
  %.ds0002p_357 = alloca i32, align 4
  %.dl0002p_359 = alloca i32, align 4
  %.dl0002p.copy_457 = alloca i32, align 4
  %.de0002p.copy_458 = alloca i32, align 4
  %.ds0002p.copy_459 = alloca i32, align 4
  %.dX0002p_358 = alloca i32, align 4
  %.dY0002p_353 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L26_1Arg0, metadata !42, metadata !DIExpression()), !dbg !43
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L26_1Arg1, metadata !44, metadata !DIExpression()), !dbg !43
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L26_1Arg2, metadata !45, metadata !DIExpression()), !dbg !43
  call void @llvm.dbg.value(metadata i32 1, metadata !46, metadata !DIExpression()), !dbg !43
  call void @llvm.dbg.value(metadata i32 0, metadata !47, metadata !DIExpression()), !dbg !43
  call void @llvm.dbg.value(metadata i32 1, metadata !48, metadata !DIExpression()), !dbg !43
  call void @llvm.dbg.value(metadata i32 0, metadata !49, metadata !DIExpression()), !dbg !43
  call void @llvm.dbg.value(metadata i32 1, metadata !50, metadata !DIExpression()), !dbg !43
  %0 = load i32, i32* %__nv_MAIN__F1L26_1Arg0, align 4, !dbg !51
  store i32 %0, i32* %__gtid___nv_MAIN__F1L26_1__463, align 4, !dbg !51
  br label %L.LB2_448

L.LB2_448:                                        ; preds = %L.entry
  br label %L.LB2_319

L.LB2_319:                                        ; preds = %L.LB2_448
  store i32 0, i32* %.i0000p_321, align 4, !dbg !52
  call void @llvm.dbg.declare(metadata i32* %i_320, metadata !53, metadata !DIExpression()), !dbg !51
  store i32 1, i32* %i_320, align 4, !dbg !52
  %1 = bitcast i64* %__nv_MAIN__F1L26_1Arg2 to i32**, !dbg !52
  %2 = load i32*, i32** %1, align 8, !dbg !52
  %3 = load i32, i32* %2, align 4, !dbg !52
  store i32 %3, i32* %.du0002p_354, align 4, !dbg !52
  %4 = bitcast i64* %__nv_MAIN__F1L26_1Arg2 to i32**, !dbg !52
  %5 = load i32*, i32** %4, align 8, !dbg !52
  %6 = load i32, i32* %5, align 4, !dbg !52
  store i32 %6, i32* %.de0002p_355, align 4, !dbg !52
  store i32 1, i32* %.di0002p_356, align 4, !dbg !52
  %7 = load i32, i32* %.di0002p_356, align 4, !dbg !52
  store i32 %7, i32* %.ds0002p_357, align 4, !dbg !52
  store i32 1, i32* %.dl0002p_359, align 4, !dbg !52
  %8 = load i32, i32* %.dl0002p_359, align 4, !dbg !52
  store i32 %8, i32* %.dl0002p.copy_457, align 4, !dbg !52
  %9 = load i32, i32* %.de0002p_355, align 4, !dbg !52
  store i32 %9, i32* %.de0002p.copy_458, align 4, !dbg !52
  %10 = load i32, i32* %.ds0002p_357, align 4, !dbg !52
  store i32 %10, i32* %.ds0002p.copy_459, align 4, !dbg !52
  %11 = load i32, i32* %__gtid___nv_MAIN__F1L26_1__463, align 4, !dbg !52
  %12 = bitcast i32* %.i0000p_321 to i64*, !dbg !52
  %13 = bitcast i32* %.dl0002p.copy_457 to i64*, !dbg !52
  %14 = bitcast i32* %.de0002p.copy_458 to i64*, !dbg !52
  %15 = bitcast i32* %.ds0002p.copy_459 to i64*, !dbg !52
  %16 = load i32, i32* %.ds0002p.copy_459, align 4, !dbg !52
  call void @__kmpc_for_static_init_4(i64* null, i32 %11, i32 34, i64* %12, i64* %13, i64* %14, i64* %15, i32 %16, i32 1), !dbg !52
  %17 = load i32, i32* %.dl0002p.copy_457, align 4, !dbg !52
  store i32 %17, i32* %.dl0002p_359, align 4, !dbg !52
  %18 = load i32, i32* %.de0002p.copy_458, align 4, !dbg !52
  store i32 %18, i32* %.de0002p_355, align 4, !dbg !52
  %19 = load i32, i32* %.ds0002p.copy_459, align 4, !dbg !52
  store i32 %19, i32* %.ds0002p_357, align 4, !dbg !52
  %20 = load i32, i32* %.dl0002p_359, align 4, !dbg !52
  store i32 %20, i32* %i_320, align 4, !dbg !52
  %21 = load i32, i32* %i_320, align 4, !dbg !52
  call void @llvm.dbg.value(metadata i32 %21, metadata !53, metadata !DIExpression()), !dbg !51
  store i32 %21, i32* %.dX0002p_358, align 4, !dbg !52
  %22 = load i32, i32* %.dX0002p_358, align 4, !dbg !52
  %23 = load i32, i32* %.du0002p_354, align 4, !dbg !52
  %24 = icmp sgt i32 %22, %23, !dbg !52
  br i1 %24, label %L.LB2_352, label %L.LB2_488, !dbg !52

L.LB2_488:                                        ; preds = %L.LB2_319
  %25 = load i32, i32* %.dX0002p_358, align 4, !dbg !52
  store i32 %25, i32* %i_320, align 4, !dbg !52
  %26 = load i32, i32* %.di0002p_356, align 4, !dbg !52
  %27 = load i32, i32* %.de0002p_355, align 4, !dbg !52
  %28 = load i32, i32* %.dX0002p_358, align 4, !dbg !52
  %29 = sub nsw i32 %27, %28, !dbg !52
  %30 = add nsw i32 %26, %29, !dbg !52
  %31 = load i32, i32* %.di0002p_356, align 4, !dbg !52
  %32 = sdiv i32 %30, %31, !dbg !52
  store i32 %32, i32* %.dY0002p_353, align 4, !dbg !52
  %33 = load i32, i32* %.dY0002p_353, align 4, !dbg !52
  %34 = icmp sle i32 %33, 0, !dbg !52
  br i1 %34, label %L.LB2_362, label %L.LB2_361, !dbg !52

L.LB2_361:                                        ; preds = %L.LB2_361, %L.LB2_488
  %35 = load i32, i32* %i_320, align 4, !dbg !54
  call void @llvm.dbg.value(metadata i32 %35, metadata !53, metadata !DIExpression()), !dbg !51
  %36 = load i32, i32* %i_320, align 4, !dbg !54
  call void @llvm.dbg.value(metadata i32 %36, metadata !53, metadata !DIExpression()), !dbg !51
  %37 = sext i32 %36 to i64, !dbg !54
  %38 = bitcast i64* %__nv_MAIN__F1L26_1Arg2 to i8*, !dbg !54
  %39 = getelementptr i8, i8* %38, i64 72, !dbg !54
  %40 = bitcast i8* %39 to i8**, !dbg !54
  %41 = load i8*, i8** %40, align 8, !dbg !54
  %42 = getelementptr i8, i8* %41, i64 56, !dbg !54
  %43 = bitcast i8* %42 to i64*, !dbg !54
  %44 = load i64, i64* %43, align 8, !dbg !54
  %45 = add nsw i64 %37, %44, !dbg !54
  %46 = bitcast i64* %__nv_MAIN__F1L26_1Arg2 to i8*, !dbg !54
  %47 = getelementptr i8, i8* %46, i64 16, !dbg !54
  %48 = bitcast i8* %47 to i8***, !dbg !54
  %49 = load i8**, i8*** %48, align 8, !dbg !54
  %50 = load i8*, i8** %49, align 8, !dbg !54
  %51 = getelementptr i8, i8* %50, i64 -4, !dbg !54
  %52 = bitcast i8* %51 to i32*, !dbg !54
  %53 = getelementptr i32, i32* %52, i64 %45, !dbg !54
  %54 = load i32, i32* %53, align 4, !dbg !54
  %55 = add nsw i32 %35, %54, !dbg !54
  %56 = bitcast i64* %__nv_MAIN__F1L26_1Arg2 to i8*, !dbg !54
  %57 = getelementptr i8, i8* %56, i64 64, !dbg !54
  %58 = bitcast i8* %57 to i32**, !dbg !54
  %59 = load i32*, i32** %58, align 8, !dbg !54
  store i32 %55, i32* %59, align 4, !dbg !54
  %60 = bitcast i64* %__nv_MAIN__F1L26_1Arg2 to i8*, !dbg !55
  %61 = getelementptr i8, i8* %60, i64 64, !dbg !55
  %62 = bitcast i8* %61 to i32**, !dbg !55
  %63 = load i32*, i32** %62, align 8, !dbg !55
  %64 = load i32, i32* %63, align 4, !dbg !55
  %65 = load i32, i32* %i_320, align 4, !dbg !55
  call void @llvm.dbg.value(metadata i32 %65, metadata !53, metadata !DIExpression()), !dbg !51
  %66 = sext i32 %65 to i64, !dbg !55
  %67 = bitcast i64* %__nv_MAIN__F1L26_1Arg2 to i8*, !dbg !55
  %68 = getelementptr i8, i8* %67, i64 72, !dbg !55
  %69 = bitcast i8* %68 to i8**, !dbg !55
  %70 = load i8*, i8** %69, align 8, !dbg !55
  %71 = getelementptr i8, i8* %70, i64 56, !dbg !55
  %72 = bitcast i8* %71 to i64*, !dbg !55
  %73 = load i64, i64* %72, align 8, !dbg !55
  %74 = add nsw i64 %66, %73, !dbg !55
  %75 = bitcast i64* %__nv_MAIN__F1L26_1Arg2 to i8*, !dbg !55
  %76 = getelementptr i8, i8* %75, i64 16, !dbg !55
  %77 = bitcast i8* %76 to i8***, !dbg !55
  %78 = load i8**, i8*** %77, align 8, !dbg !55
  %79 = load i8*, i8** %78, align 8, !dbg !55
  %80 = getelementptr i8, i8* %79, i64 -4, !dbg !55
  %81 = bitcast i8* %80 to i32*, !dbg !55
  %82 = getelementptr i32, i32* %81, i64 %74, !dbg !55
  store i32 %64, i32* %82, align 4, !dbg !55
  %83 = load i32, i32* %.di0002p_356, align 4, !dbg !51
  %84 = load i32, i32* %i_320, align 4, !dbg !51
  call void @llvm.dbg.value(metadata i32 %84, metadata !53, metadata !DIExpression()), !dbg !51
  %85 = add nsw i32 %83, %84, !dbg !51
  store i32 %85, i32* %i_320, align 4, !dbg !51
  %86 = load i32, i32* %.dY0002p_353, align 4, !dbg !51
  %87 = sub nsw i32 %86, 1, !dbg !51
  store i32 %87, i32* %.dY0002p_353, align 4, !dbg !51
  %88 = load i32, i32* %.dY0002p_353, align 4, !dbg !51
  %89 = icmp sgt i32 %88, 0, !dbg !51
  br i1 %89, label %L.LB2_361, label %L.LB2_362, !dbg !51

L.LB2_362:                                        ; preds = %L.LB2_361, %L.LB2_488
  br label %L.LB2_352

L.LB2_352:                                        ; preds = %L.LB2_362, %L.LB2_319
  %90 = load i32, i32* %__gtid___nv_MAIN__F1L26_1__463, align 4, !dbg !51
  call void @__kmpc_for_static_fini(i64* null, i32 %90), !dbg !51
  br label %L.LB2_322

L.LB2_322:                                        ; preds = %L.LB2_352
  ret void, !dbg !51
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
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB028-privatemissing-orig-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb028_privatemissing_orig_yes", scope: !2, file: !3, line: 12, type: !6, scopeLine: 12, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!6 = !DISubroutineType(cc: DW_CC_program, types: !7)
!7 = !{null}
!8 = !DILocalVariable(name: "omp_sched_static", scope: !5, file: !3, type: !9)
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 0, scope: !5)
!11 = !DILocalVariable(name: "omp_proc_bind_false", scope: !5, file: !3, type: !9)
!12 = !DILocalVariable(name: "omp_proc_bind_true", scope: !5, file: !3, type: !9)
!13 = !DILocalVariable(name: "omp_lock_hint_none", scope: !5, file: !3, type: !9)
!14 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !5, file: !3, type: !9)
!15 = !DILocation(line: 37, column: 1, scope: !5)
!16 = !DILocation(line: 12, column: 1, scope: !5)
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
!27 = !DILocation(line: 19, column: 1, scope: !5)
!28 = !DILocalVariable(scope: !5, file: !3, type: !23, flags: DIFlagArtificial)
!29 = !DILocation(line: 20, column: 1, scope: !5)
!30 = !DILocation(line: 22, column: 1, scope: !5)
!31 = !DILocalVariable(name: "i", scope: !5, file: !3, type: !9)
!32 = !DILocation(line: 23, column: 1, scope: !5)
!33 = !DILocation(line: 24, column: 1, scope: !5)
!34 = !DILocation(line: 26, column: 1, scope: !5)
!35 = !DILocalVariable(name: "tmp", scope: !5, file: !3, type: !9)
!36 = !DILocation(line: 33, column: 1, scope: !5)
!37 = !DILocalVariable(scope: !5, file: !3, type: !9, flags: DIFlagArtificial)
!38 = !DILocation(line: 36, column: 1, scope: !5)
!39 = distinct !DISubprogram(name: "__nv_MAIN__F1L26_1", scope: !2, file: !3, line: 26, type: !40, scopeLine: 26, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!40 = !DISubroutineType(types: !41)
!41 = !{null, !9, !23, !23}
!42 = !DILocalVariable(name: "__nv_MAIN__F1L26_1Arg0", arg: 1, scope: !39, file: !3, type: !9)
!43 = !DILocation(line: 0, scope: !39)
!44 = !DILocalVariable(name: "__nv_MAIN__F1L26_1Arg1", arg: 2, scope: !39, file: !3, type: !23)
!45 = !DILocalVariable(name: "__nv_MAIN__F1L26_1Arg2", arg: 3, scope: !39, file: !3, type: !23)
!46 = !DILocalVariable(name: "omp_sched_static", scope: !39, file: !3, type: !9)
!47 = !DILocalVariable(name: "omp_proc_bind_false", scope: !39, file: !3, type: !9)
!48 = !DILocalVariable(name: "omp_proc_bind_true", scope: !39, file: !3, type: !9)
!49 = !DILocalVariable(name: "omp_lock_hint_none", scope: !39, file: !3, type: !9)
!50 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !39, file: !3, type: !9)
!51 = !DILocation(line: 30, column: 1, scope: !39)
!52 = !DILocation(line: 27, column: 1, scope: !39)
!53 = !DILocalVariable(name: "i", scope: !39, file: !3, type: !9)
!54 = !DILocation(line: 28, column: 1, scope: !39)
!55 = !DILocation(line: 29, column: 1, scope: !39)

; ModuleID = '/tmp/DRB026-targetparallelfor-orig-yes-d33368.ll'
source_filename = "/tmp/DRB026-targetparallelfor-orig-yes-d33368.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%astruct.dt68 = type <{ i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8* }>
%astruct.dt113 = type <{ [72 x i8] }>

@.C305_MAIN_ = internal constant i32 14
@.C332_MAIN_ = internal constant [26 x i8] c"Values for i and a(i) are:"
@.C331_MAIN_ = internal constant i32 6
@.C328_MAIN_ = internal constant [62 x i8] c"micro-benchmarks-fortran/DRB026-targetparallelfor-orig-yes.f95"
@.C330_MAIN_ = internal constant i32 35
@.C285_MAIN_ = internal constant i32 1
@.C306_MAIN_ = internal constant i32 25
@.C344_MAIN_ = internal constant i64 4
@.C343_MAIN_ = internal constant i64 25
@.C314_MAIN_ = internal constant i32 1000
@.C286_MAIN_ = internal constant i64 1
@.C284_MAIN_ = internal constant i64 0
@.C283_MAIN_ = internal constant i32 0
@.C285___nv_MAIN__F1L26_1 = internal constant i32 1
@.C283___nv_MAIN__F1L26_1 = internal constant i32 0
@.C285___nv_MAIN_F1L27_2 = internal constant i32 1
@.C283___nv_MAIN_F1L27_2 = internal constant i32 0

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__416 = alloca i32, align 4
  %.Z0965_316 = alloca i32*, align 8
  %"a$sd1_342" = alloca [16 x i64], align 8
  %len_315 = alloca i32, align 4
  %z_b_0_308 = alloca i64, align 8
  %z_b_1_309 = alloca i64, align 8
  %z_e_60_312 = alloca i64, align 8
  %z_b_2_310 = alloca i64, align 8
  %z_b_3_311 = alloca i64, align 8
  %.dY0001_353 = alloca i32, align 4
  %i_307 = alloca i32, align 4
  %.uplevelArgPack0001_397 = alloca %astruct.dt68, align 16
  %.dY0003_368 = alloca i32, align 4
  %z__io_334 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !14, metadata !DIExpression()), !dbg !10
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !15
  store i32 %0, i32* %__gtid_MAIN__416, align 4, !dbg !15
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !16
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !16
  call void (i8*, ...) %2(i8* %1), !dbg !16
  call void @llvm.dbg.declare(metadata i32** %.Z0965_316, metadata !17, metadata !DIExpression(DW_OP_deref)), !dbg !10
  %3 = bitcast i32** %.Z0965_316 to i8**, !dbg !16
  store i8* null, i8** %3, align 8, !dbg !16
  call void @llvm.dbg.declare(metadata [16 x i64]* %"a$sd1_342", metadata !21, metadata !DIExpression()), !dbg !10
  %4 = bitcast [16 x i64]* %"a$sd1_342" to i64*, !dbg !16
  store i64 0, i64* %4, align 8, !dbg !16
  br label %L.LB1_376

L.LB1_376:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %len_315, metadata !26, metadata !DIExpression()), !dbg !10
  store i32 1000, i32* %len_315, align 4, !dbg !27
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
  %8 = bitcast [16 x i64]* %"a$sd1_342" to i8*, !dbg !29
  %9 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !29
  %10 = bitcast i64* @.C343_MAIN_ to i8*, !dbg !29
  %11 = bitcast i64* @.C344_MAIN_ to i8*, !dbg !29
  %12 = bitcast i64* %z_b_0_308 to i8*, !dbg !29
  %13 = bitcast i64* %z_b_1_309 to i8*, !dbg !29
  %14 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !29
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %14(i8* %8, i8* %9, i8* %10, i8* %11, i8* %12, i8* %13), !dbg !29
  %15 = bitcast [16 x i64]* %"a$sd1_342" to i8*, !dbg !29
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
  %23 = bitcast i64* @.C343_MAIN_ to i8*, !dbg !29
  %24 = bitcast i64* @.C344_MAIN_ to i8*, !dbg !29
  %25 = bitcast i32** %.Z0965_316 to i8*, !dbg !29
  %26 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !29
  %27 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !29
  %28 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !29
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %28(i8* %22, i8* %23, i8* %24, i8* null, i8* %25, i8* null, i8* %26, i8* %27, i8* null, i64 0), !dbg !29
  %29 = load i32, i32* %len_315, align 4, !dbg !30
  call void @llvm.dbg.value(metadata i32 %29, metadata !26, metadata !DIExpression()), !dbg !10
  store i32 %29, i32* %.dY0001_353, align 4, !dbg !30
  call void @llvm.dbg.declare(metadata i32* %i_307, metadata !31, metadata !DIExpression()), !dbg !10
  store i32 1, i32* %i_307, align 4, !dbg !30
  %30 = load i32, i32* %.dY0001_353, align 4, !dbg !30
  %31 = icmp sle i32 %30, 0, !dbg !30
  br i1 %31, label %L.LB1_352, label %L.LB1_351, !dbg !30

L.LB1_351:                                        ; preds = %L.LB1_351, %L.LB1_376
  %32 = load i32, i32* %i_307, align 4, !dbg !32
  call void @llvm.dbg.value(metadata i32 %32, metadata !31, metadata !DIExpression()), !dbg !10
  %33 = load i32, i32* %i_307, align 4, !dbg !32
  call void @llvm.dbg.value(metadata i32 %33, metadata !31, metadata !DIExpression()), !dbg !10
  %34 = sext i32 %33 to i64, !dbg !32
  %35 = bitcast [16 x i64]* %"a$sd1_342" to i8*, !dbg !32
  %36 = getelementptr i8, i8* %35, i64 56, !dbg !32
  %37 = bitcast i8* %36 to i64*, !dbg !32
  %38 = load i64, i64* %37, align 8, !dbg !32
  %39 = add nsw i64 %34, %38, !dbg !32
  %40 = load i32*, i32** %.Z0965_316, align 8, !dbg !32
  call void @llvm.dbg.value(metadata i32* %40, metadata !17, metadata !DIExpression()), !dbg !10
  %41 = bitcast i32* %40 to i8*, !dbg !32
  %42 = getelementptr i8, i8* %41, i64 -4, !dbg !32
  %43 = bitcast i8* %42 to i32*, !dbg !32
  %44 = getelementptr i32, i32* %43, i64 %39, !dbg !32
  store i32 %32, i32* %44, align 4, !dbg !32
  %45 = load i32, i32* %i_307, align 4, !dbg !33
  call void @llvm.dbg.value(metadata i32 %45, metadata !31, metadata !DIExpression()), !dbg !10
  %46 = add nsw i32 %45, 1, !dbg !33
  store i32 %46, i32* %i_307, align 4, !dbg !33
  %47 = load i32, i32* %.dY0001_353, align 4, !dbg !33
  %48 = sub nsw i32 %47, 1, !dbg !33
  store i32 %48, i32* %.dY0001_353, align 4, !dbg !33
  %49 = load i32, i32* %.dY0001_353, align 4, !dbg !33
  %50 = icmp sgt i32 %49, 0, !dbg !33
  br i1 %50, label %L.LB1_351, label %L.LB1_352, !dbg !33

L.LB1_352:                                        ; preds = %L.LB1_351, %L.LB1_376
  %51 = bitcast i32* %len_315 to i8*, !dbg !34
  %52 = bitcast %astruct.dt68* %.uplevelArgPack0001_397 to i8**, !dbg !34
  store i8* %51, i8** %52, align 8, !dbg !34
  %53 = bitcast i32** %.Z0965_316 to i8*, !dbg !34
  %54 = bitcast %astruct.dt68* %.uplevelArgPack0001_397 to i8*, !dbg !34
  %55 = getelementptr i8, i8* %54, i64 8, !dbg !34
  %56 = bitcast i8* %55 to i8**, !dbg !34
  store i8* %53, i8** %56, align 8, !dbg !34
  %57 = bitcast i32** %.Z0965_316 to i8*, !dbg !34
  %58 = bitcast %astruct.dt68* %.uplevelArgPack0001_397 to i8*, !dbg !34
  %59 = getelementptr i8, i8* %58, i64 16, !dbg !34
  %60 = bitcast i8* %59 to i8**, !dbg !34
  store i8* %57, i8** %60, align 8, !dbg !34
  %61 = bitcast i64* %z_b_0_308 to i8*, !dbg !34
  %62 = bitcast %astruct.dt68* %.uplevelArgPack0001_397 to i8*, !dbg !34
  %63 = getelementptr i8, i8* %62, i64 24, !dbg !34
  %64 = bitcast i8* %63 to i8**, !dbg !34
  store i8* %61, i8** %64, align 8, !dbg !34
  %65 = bitcast i64* %z_b_1_309 to i8*, !dbg !34
  %66 = bitcast %astruct.dt68* %.uplevelArgPack0001_397 to i8*, !dbg !34
  %67 = getelementptr i8, i8* %66, i64 32, !dbg !34
  %68 = bitcast i8* %67 to i8**, !dbg !34
  store i8* %65, i8** %68, align 8, !dbg !34
  %69 = bitcast i64* %z_e_60_312 to i8*, !dbg !34
  %70 = bitcast %astruct.dt68* %.uplevelArgPack0001_397 to i8*, !dbg !34
  %71 = getelementptr i8, i8* %70, i64 40, !dbg !34
  %72 = bitcast i8* %71 to i8**, !dbg !34
  store i8* %69, i8** %72, align 8, !dbg !34
  %73 = bitcast i64* %z_b_2_310 to i8*, !dbg !34
  %74 = bitcast %astruct.dt68* %.uplevelArgPack0001_397 to i8*, !dbg !34
  %75 = getelementptr i8, i8* %74, i64 48, !dbg !34
  %76 = bitcast i8* %75 to i8**, !dbg !34
  store i8* %73, i8** %76, align 8, !dbg !34
  %77 = bitcast i64* %z_b_3_311 to i8*, !dbg !34
  %78 = bitcast %astruct.dt68* %.uplevelArgPack0001_397 to i8*, !dbg !34
  %79 = getelementptr i8, i8* %78, i64 56, !dbg !34
  %80 = bitcast i8* %79 to i8**, !dbg !34
  store i8* %77, i8** %80, align 8, !dbg !34
  %81 = bitcast [16 x i64]* %"a$sd1_342" to i8*, !dbg !34
  %82 = bitcast %astruct.dt68* %.uplevelArgPack0001_397 to i8*, !dbg !34
  %83 = getelementptr i8, i8* %82, i64 64, !dbg !34
  %84 = bitcast i8* %83 to i8**, !dbg !34
  store i8* %81, i8** %84, align 8, !dbg !34
  %85 = bitcast %astruct.dt68* %.uplevelArgPack0001_397 to i64*, !dbg !34
  call void @__nv_MAIN__F1L26_1_(i32* %__gtid_MAIN__416, i64* null, i64* %85), !dbg !34
  %86 = load i32, i32* %len_315, align 4, !dbg !35
  call void @llvm.dbg.value(metadata i32 %86, metadata !26, metadata !DIExpression()), !dbg !10
  store i32 %86, i32* %.dY0003_368, align 4, !dbg !35
  store i32 1, i32* %i_307, align 4, !dbg !35
  %87 = load i32, i32* %.dY0003_368, align 4, !dbg !35
  %88 = icmp sle i32 %87, 0, !dbg !35
  br i1 %88, label %L.LB1_367, label %L.LB1_366, !dbg !35

L.LB1_366:                                        ; preds = %L.LB1_366, %L.LB1_352
  call void (...) @_mp_bcs_nest(), !dbg !36
  %89 = bitcast i32* @.C330_MAIN_ to i8*, !dbg !36
  %90 = bitcast [62 x i8]* @.C328_MAIN_ to i8*, !dbg !36
  %91 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !36
  call void (i8*, i8*, i64, ...) %91(i8* %89, i8* %90, i64 62), !dbg !36
  %92 = bitcast i32* @.C331_MAIN_ to i8*, !dbg !36
  %93 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !36
  %94 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !36
  %95 = bitcast i32 (...)* @f90io_ldw_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !36
  %96 = call i32 (i8*, i8*, i8*, i8*, ...) %95(i8* %92, i8* null, i8* %93, i8* %94), !dbg !36
  call void @llvm.dbg.declare(metadata i32* %z__io_334, metadata !37, metadata !DIExpression()), !dbg !10
  store i32 %96, i32* %z__io_334, align 4, !dbg !36
  %97 = bitcast [26 x i8]* @.C332_MAIN_ to i8*, !dbg !36
  %98 = bitcast i32 (...)* @f90io_sc_ch_ldw to i32 (i8*, i32, i64, ...)*, !dbg !36
  %99 = call i32 (i8*, i32, i64, ...) %98(i8* %97, i32 14, i64 26), !dbg !36
  store i32 %99, i32* %z__io_334, align 4, !dbg !36
  %100 = load i32, i32* %i_307, align 4, !dbg !36
  call void @llvm.dbg.value(metadata i32 %100, metadata !31, metadata !DIExpression()), !dbg !10
  %101 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !36
  %102 = call i32 (i32, i32, ...) %101(i32 %100, i32 25), !dbg !36
  store i32 %102, i32* %z__io_334, align 4, !dbg !36
  %103 = load i32, i32* %i_307, align 4, !dbg !36
  call void @llvm.dbg.value(metadata i32 %103, metadata !31, metadata !DIExpression()), !dbg !10
  %104 = sext i32 %103 to i64, !dbg !36
  %105 = bitcast [16 x i64]* %"a$sd1_342" to i8*, !dbg !36
  %106 = getelementptr i8, i8* %105, i64 56, !dbg !36
  %107 = bitcast i8* %106 to i64*, !dbg !36
  %108 = load i64, i64* %107, align 8, !dbg !36
  %109 = add nsw i64 %104, %108, !dbg !36
  %110 = load i32*, i32** %.Z0965_316, align 8, !dbg !36
  call void @llvm.dbg.value(metadata i32* %110, metadata !17, metadata !DIExpression()), !dbg !10
  %111 = bitcast i32* %110 to i8*, !dbg !36
  %112 = getelementptr i8, i8* %111, i64 -4, !dbg !36
  %113 = bitcast i8* %112 to i32*, !dbg !36
  %114 = getelementptr i32, i32* %113, i64 %109, !dbg !36
  %115 = load i32, i32* %114, align 4, !dbg !36
  %116 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !36
  %117 = call i32 (i32, i32, ...) %116(i32 %115, i32 25), !dbg !36
  store i32 %117, i32* %z__io_334, align 4, !dbg !36
  %118 = call i32 (...) @f90io_ldw_end(), !dbg !36
  store i32 %118, i32* %z__io_334, align 4, !dbg !36
  call void (...) @_mp_ecs_nest(), !dbg !36
  %119 = load i32, i32* %i_307, align 4, !dbg !38
  call void @llvm.dbg.value(metadata i32 %119, metadata !31, metadata !DIExpression()), !dbg !10
  %120 = add nsw i32 %119, 1, !dbg !38
  store i32 %120, i32* %i_307, align 4, !dbg !38
  %121 = load i32, i32* %.dY0003_368, align 4, !dbg !38
  %122 = sub nsw i32 %121, 1, !dbg !38
  store i32 %122, i32* %.dY0003_368, align 4, !dbg !38
  %123 = load i32, i32* %.dY0003_368, align 4, !dbg !38
  %124 = icmp sgt i32 %123, 0, !dbg !38
  br i1 %124, label %L.LB1_366, label %L.LB1_367, !dbg !38

L.LB1_367:                                        ; preds = %L.LB1_366, %L.LB1_352
  %125 = load i32*, i32** %.Z0965_316, align 8, !dbg !39
  call void @llvm.dbg.value(metadata i32* %125, metadata !17, metadata !DIExpression()), !dbg !10
  %126 = bitcast i32* %125 to i8*, !dbg !39
  %127 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !39
  %128 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !39
  call void (i8*, i8*, i8*, i8*, i64, ...) %128(i8* null, i8* %126, i8* %127, i8* null, i64 0), !dbg !39
  %129 = bitcast i32** %.Z0965_316 to i8**, !dbg !39
  store i8* null, i8** %129, align 8, !dbg !39
  %130 = bitcast [16 x i64]* %"a$sd1_342" to i64*, !dbg !39
  store i64 0, i64* %130, align 8, !dbg !39
  ret void, !dbg !15
}

define internal void @__nv_MAIN__F1L26_1_(i32* %__nv_MAIN__F1L26_1Arg0, i64* %__nv_MAIN__F1L26_1Arg1, i64* %__nv_MAIN__F1L26_1Arg2) #0 !dbg !40 {
L.entry:
  %__gtid___nv_MAIN__F1L26_1__445 = alloca i32, align 4
  %.uplevelArgPack0002_440 = alloca %astruct.dt113, align 16
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L26_1Arg0, metadata !43, metadata !DIExpression()), !dbg !44
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L26_1Arg1, metadata !45, metadata !DIExpression()), !dbg !44
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L26_1Arg2, metadata !46, metadata !DIExpression()), !dbg !44
  call void @llvm.dbg.value(metadata i32 1, metadata !47, metadata !DIExpression()), !dbg !44
  call void @llvm.dbg.value(metadata i32 0, metadata !48, metadata !DIExpression()), !dbg !44
  call void @llvm.dbg.value(metadata i32 1, metadata !49, metadata !DIExpression()), !dbg !44
  call void @llvm.dbg.value(metadata i32 0, metadata !50, metadata !DIExpression()), !dbg !44
  call void @llvm.dbg.value(metadata i32 1, metadata !51, metadata !DIExpression()), !dbg !44
  %0 = load i32, i32* %__nv_MAIN__F1L26_1Arg0, align 4, !dbg !52
  store i32 %0, i32* %__gtid___nv_MAIN__F1L26_1__445, align 4, !dbg !52
  br label %L.LB2_435

L.LB2_435:                                        ; preds = %L.entry
  br label %L.LB2_319

L.LB2_319:                                        ; preds = %L.LB2_435
  %1 = bitcast i64* %__nv_MAIN__F1L26_1Arg2 to i8*, !dbg !53
  %2 = bitcast %astruct.dt113* %.uplevelArgPack0002_440 to i8*, !dbg !53
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %2, i8* align 8 %1, i64 72, i1 false), !dbg !53
  br label %L.LB2_443, !dbg !53

L.LB2_443:                                        ; preds = %L.LB2_319
  %3 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN_F1L27_2_ to i64*, !dbg !53
  %4 = bitcast %astruct.dt113* %.uplevelArgPack0002_440 to i64*, !dbg !53
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %3, i64* %4), !dbg !53
  br label %L.LB2_326

L.LB2_326:                                        ; preds = %L.LB2_443
  ret void, !dbg !52
}

define internal void @__nv_MAIN_F1L27_2_(i32* %__nv_MAIN_F1L27_2Arg0, i64* %__nv_MAIN_F1L27_2Arg1, i64* %__nv_MAIN_F1L27_2Arg2) #0 !dbg !54 {
L.entry:
  %__gtid___nv_MAIN_F1L27_2__483 = alloca i32, align 4
  %.i0000p_324 = alloca i32, align 4
  %i_323 = alloca i32, align 4
  %.du0002p_357 = alloca i32, align 4
  %.de0002p_358 = alloca i32, align 4
  %.di0002p_359 = alloca i32, align 4
  %.ds0002p_360 = alloca i32, align 4
  %.dl0002p_362 = alloca i32, align 4
  %.dl0002p.copy_477 = alloca i32, align 4
  %.de0002p.copy_478 = alloca i32, align 4
  %.ds0002p.copy_479 = alloca i32, align 4
  %.dX0002p_361 = alloca i32, align 4
  %.dY0002p_356 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L27_2Arg0, metadata !55, metadata !DIExpression()), !dbg !56
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L27_2Arg1, metadata !57, metadata !DIExpression()), !dbg !56
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L27_2Arg2, metadata !58, metadata !DIExpression()), !dbg !56
  call void @llvm.dbg.value(metadata i32 1, metadata !59, metadata !DIExpression()), !dbg !56
  call void @llvm.dbg.value(metadata i32 0, metadata !60, metadata !DIExpression()), !dbg !56
  call void @llvm.dbg.value(metadata i32 1, metadata !61, metadata !DIExpression()), !dbg !56
  call void @llvm.dbg.value(metadata i32 0, metadata !62, metadata !DIExpression()), !dbg !56
  call void @llvm.dbg.value(metadata i32 1, metadata !63, metadata !DIExpression()), !dbg !56
  %0 = load i32, i32* %__nv_MAIN_F1L27_2Arg0, align 4, !dbg !64
  store i32 %0, i32* %__gtid___nv_MAIN_F1L27_2__483, align 4, !dbg !64
  br label %L.LB4_468

L.LB4_468:                                        ; preds = %L.entry
  br label %L.LB4_322

L.LB4_322:                                        ; preds = %L.LB4_468
  store i32 0, i32* %.i0000p_324, align 4, !dbg !65
  call void @llvm.dbg.declare(metadata i32* %i_323, metadata !66, metadata !DIExpression()), !dbg !64
  store i32 1, i32* %i_323, align 4, !dbg !65
  %1 = bitcast i64* %__nv_MAIN_F1L27_2Arg2 to i32**, !dbg !65
  %2 = load i32*, i32** %1, align 8, !dbg !65
  %3 = load i32, i32* %2, align 4, !dbg !65
  %4 = sub nsw i32 %3, 1, !dbg !65
  store i32 %4, i32* %.du0002p_357, align 4, !dbg !65
  %5 = bitcast i64* %__nv_MAIN_F1L27_2Arg2 to i32**, !dbg !65
  %6 = load i32*, i32** %5, align 8, !dbg !65
  %7 = load i32, i32* %6, align 4, !dbg !65
  %8 = sub nsw i32 %7, 1, !dbg !65
  store i32 %8, i32* %.de0002p_358, align 4, !dbg !65
  store i32 1, i32* %.di0002p_359, align 4, !dbg !65
  %9 = load i32, i32* %.di0002p_359, align 4, !dbg !65
  store i32 %9, i32* %.ds0002p_360, align 4, !dbg !65
  store i32 1, i32* %.dl0002p_362, align 4, !dbg !65
  %10 = load i32, i32* %.dl0002p_362, align 4, !dbg !65
  store i32 %10, i32* %.dl0002p.copy_477, align 4, !dbg !65
  %11 = load i32, i32* %.de0002p_358, align 4, !dbg !65
  store i32 %11, i32* %.de0002p.copy_478, align 4, !dbg !65
  %12 = load i32, i32* %.ds0002p_360, align 4, !dbg !65
  store i32 %12, i32* %.ds0002p.copy_479, align 4, !dbg !65
  %13 = load i32, i32* %__gtid___nv_MAIN_F1L27_2__483, align 4, !dbg !65
  %14 = bitcast i32* %.i0000p_324 to i64*, !dbg !65
  %15 = bitcast i32* %.dl0002p.copy_477 to i64*, !dbg !65
  %16 = bitcast i32* %.de0002p.copy_478 to i64*, !dbg !65
  %17 = bitcast i32* %.ds0002p.copy_479 to i64*, !dbg !65
  %18 = load i32, i32* %.ds0002p.copy_479, align 4, !dbg !65
  call void @__kmpc_for_static_init_4(i64* null, i32 %13, i32 34, i64* %14, i64* %15, i64* %16, i64* %17, i32 %18, i32 1), !dbg !65
  %19 = load i32, i32* %.dl0002p.copy_477, align 4, !dbg !65
  store i32 %19, i32* %.dl0002p_362, align 4, !dbg !65
  %20 = load i32, i32* %.de0002p.copy_478, align 4, !dbg !65
  store i32 %20, i32* %.de0002p_358, align 4, !dbg !65
  %21 = load i32, i32* %.ds0002p.copy_479, align 4, !dbg !65
  store i32 %21, i32* %.ds0002p_360, align 4, !dbg !65
  %22 = load i32, i32* %.dl0002p_362, align 4, !dbg !65
  store i32 %22, i32* %i_323, align 4, !dbg !65
  %23 = load i32, i32* %i_323, align 4, !dbg !65
  call void @llvm.dbg.value(metadata i32 %23, metadata !66, metadata !DIExpression()), !dbg !64
  store i32 %23, i32* %.dX0002p_361, align 4, !dbg !65
  %24 = load i32, i32* %.dX0002p_361, align 4, !dbg !65
  %25 = load i32, i32* %.du0002p_357, align 4, !dbg !65
  %26 = icmp sgt i32 %24, %25, !dbg !65
  br i1 %26, label %L.LB4_355, label %L.LB4_507, !dbg !65

L.LB4_507:                                        ; preds = %L.LB4_322
  %27 = load i32, i32* %.dX0002p_361, align 4, !dbg !65
  store i32 %27, i32* %i_323, align 4, !dbg !65
  %28 = load i32, i32* %.di0002p_359, align 4, !dbg !65
  %29 = load i32, i32* %.de0002p_358, align 4, !dbg !65
  %30 = load i32, i32* %.dX0002p_361, align 4, !dbg !65
  %31 = sub nsw i32 %29, %30, !dbg !65
  %32 = add nsw i32 %28, %31, !dbg !65
  %33 = load i32, i32* %.di0002p_359, align 4, !dbg !65
  %34 = sdiv i32 %32, %33, !dbg !65
  store i32 %34, i32* %.dY0002p_356, align 4, !dbg !65
  %35 = load i32, i32* %.dY0002p_356, align 4, !dbg !65
  %36 = icmp sle i32 %35, 0, !dbg !65
  br i1 %36, label %L.LB4_365, label %L.LB4_364, !dbg !65

L.LB4_364:                                        ; preds = %L.LB4_364, %L.LB4_507
  %37 = load i32, i32* %i_323, align 4, !dbg !67
  call void @llvm.dbg.value(metadata i32 %37, metadata !66, metadata !DIExpression()), !dbg !64
  %38 = sext i32 %37 to i64, !dbg !67
  %39 = bitcast i64* %__nv_MAIN_F1L27_2Arg2 to i8*, !dbg !67
  %40 = getelementptr i8, i8* %39, i64 64, !dbg !67
  %41 = bitcast i8* %40 to i8**, !dbg !67
  %42 = load i8*, i8** %41, align 8, !dbg !67
  %43 = getelementptr i8, i8* %42, i64 56, !dbg !67
  %44 = bitcast i8* %43 to i64*, !dbg !67
  %45 = load i64, i64* %44, align 8, !dbg !67
  %46 = add nsw i64 %38, %45, !dbg !67
  %47 = bitcast i64* %__nv_MAIN_F1L27_2Arg2 to i8*, !dbg !67
  %48 = getelementptr i8, i8* %47, i64 16, !dbg !67
  %49 = bitcast i8* %48 to i32***, !dbg !67
  %50 = load i32**, i32*** %49, align 8, !dbg !67
  %51 = load i32*, i32** %50, align 8, !dbg !67
  %52 = getelementptr i32, i32* %51, i64 %46, !dbg !67
  %53 = load i32, i32* %52, align 4, !dbg !67
  %54 = add nsw i32 %53, 1, !dbg !67
  %55 = load i32, i32* %i_323, align 4, !dbg !67
  call void @llvm.dbg.value(metadata i32 %55, metadata !66, metadata !DIExpression()), !dbg !64
  %56 = sext i32 %55 to i64, !dbg !67
  %57 = bitcast i64* %__nv_MAIN_F1L27_2Arg2 to i8*, !dbg !67
  %58 = getelementptr i8, i8* %57, i64 64, !dbg !67
  %59 = bitcast i8* %58 to i8**, !dbg !67
  %60 = load i8*, i8** %59, align 8, !dbg !67
  %61 = getelementptr i8, i8* %60, i64 56, !dbg !67
  %62 = bitcast i8* %61 to i64*, !dbg !67
  %63 = load i64, i64* %62, align 8, !dbg !67
  %64 = add nsw i64 %56, %63, !dbg !67
  %65 = bitcast i64* %__nv_MAIN_F1L27_2Arg2 to i8*, !dbg !67
  %66 = getelementptr i8, i8* %65, i64 16, !dbg !67
  %67 = bitcast i8* %66 to i8***, !dbg !67
  %68 = load i8**, i8*** %67, align 8, !dbg !67
  %69 = load i8*, i8** %68, align 8, !dbg !67
  %70 = getelementptr i8, i8* %69, i64 -4, !dbg !67
  %71 = bitcast i8* %70 to i32*, !dbg !67
  %72 = getelementptr i32, i32* %71, i64 %64, !dbg !67
  store i32 %54, i32* %72, align 4, !dbg !67
  %73 = load i32, i32* %.di0002p_359, align 4, !dbg !64
  %74 = load i32, i32* %i_323, align 4, !dbg !64
  call void @llvm.dbg.value(metadata i32 %74, metadata !66, metadata !DIExpression()), !dbg !64
  %75 = add nsw i32 %73, %74, !dbg !64
  store i32 %75, i32* %i_323, align 4, !dbg !64
  %76 = load i32, i32* %.dY0002p_356, align 4, !dbg !64
  %77 = sub nsw i32 %76, 1, !dbg !64
  store i32 %77, i32* %.dY0002p_356, align 4, !dbg !64
  %78 = load i32, i32* %.dY0002p_356, align 4, !dbg !64
  %79 = icmp sgt i32 %78, 0, !dbg !64
  br i1 %79, label %L.LB4_364, label %L.LB4_365, !dbg !64

L.LB4_365:                                        ; preds = %L.LB4_364, %L.LB4_507
  br label %L.LB4_355

L.LB4_355:                                        ; preds = %L.LB4_365, %L.LB4_322
  %80 = load i32, i32* %__gtid___nv_MAIN_F1L27_2__483, align 4, !dbg !64
  call void @__kmpc_for_static_fini(i64* null, i32 %80), !dbg !64
  br label %L.LB4_325

L.LB4_325:                                        ; preds = %L.LB4_355
  ret void, !dbg !64
}

declare void @__kmpc_for_static_fini(i64*, i32) #0

declare void @__kmpc_for_static_init_4(i64*, i32, i32, i64*, i64*, i64*, i64*, i32, i32) #0

declare void @__kmpc_fork_call(i64*, i32, i64*, i64*, ...) #0

declare void @__kmpc_end_serialized_parallel(i64*, i32) #0

declare void @__kmpc_serialized_parallel(i64*, i32) #0

declare void @f90_dealloc03a_i8(...) #0

declare void @_mp_ecs_nest(...) #0

declare signext i32 @f90io_ldw_end(...) #0

declare signext i32 @f90io_sc_i_ldw(...) #0

declare signext i32 @f90io_sc_ch_ldw(...) #0

declare signext i32 @f90io_ldw_init(...) #0

declare void @f90io_src_info03a(...) #0

declare void @_mp_bcs_nest(...) #0

declare void @f90_alloc04_chka_i8(...) #0

declare void @f90_set_intrin_type_i8(...) #0

declare void @f90_template1_i8(...) #0

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare void @fort_init(...) #0

declare signext i32 @__kmpc_global_thread_num(i64*) #0

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1 immarg) #2

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { "no-frame-pointer-elim-non-leaf" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { argmemonly nounwind }

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !4, globals: !4, imports: !4)
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB026-targetparallelfor-orig-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb026_targetparallelfor_orig_yes", scope: !2, file: !3, line: 11, type: !6, scopeLine: 11, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!6 = !DISubroutineType(cc: DW_CC_program, types: !7)
!7 = !{null}
!8 = !DILocalVariable(name: "omp_sched_static", scope: !5, file: !3, type: !9)
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 0, scope: !5)
!11 = !DILocalVariable(name: "omp_proc_bind_false", scope: !5, file: !3, type: !9)
!12 = !DILocalVariable(name: "omp_proc_bind_true", scope: !5, file: !3, type: !9)
!13 = !DILocalVariable(name: "omp_lock_hint_none", scope: !5, file: !3, type: !9)
!14 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !5, file: !3, type: !9)
!15 = !DILocation(line: 39, column: 1, scope: !5)
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
!28 = !DILocalVariable(scope: !5, file: !3, type: !23, flags: DIFlagArtificial)
!29 = !DILocation(line: 20, column: 1, scope: !5)
!30 = !DILocation(line: 22, column: 1, scope: !5)
!31 = !DILocalVariable(name: "i", scope: !5, file: !3, type: !9)
!32 = !DILocation(line: 23, column: 1, scope: !5)
!33 = !DILocation(line: 24, column: 1, scope: !5)
!34 = !DILocation(line: 32, column: 1, scope: !5)
!35 = !DILocation(line: 34, column: 1, scope: !5)
!36 = !DILocation(line: 35, column: 1, scope: !5)
!37 = !DILocalVariable(scope: !5, file: !3, type: !9, flags: DIFlagArtificial)
!38 = !DILocation(line: 36, column: 1, scope: !5)
!39 = !DILocation(line: 38, column: 1, scope: !5)
!40 = distinct !DISubprogram(name: "__nv_MAIN__F1L26_1", scope: !2, file: !3, line: 26, type: !41, scopeLine: 26, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!41 = !DISubroutineType(types: !42)
!42 = !{null, !9, !23, !23}
!43 = !DILocalVariable(name: "__nv_MAIN__F1L26_1Arg0", arg: 1, scope: !40, file: !3, type: !9)
!44 = !DILocation(line: 0, scope: !40)
!45 = !DILocalVariable(name: "__nv_MAIN__F1L26_1Arg1", arg: 2, scope: !40, file: !3, type: !23)
!46 = !DILocalVariable(name: "__nv_MAIN__F1L26_1Arg2", arg: 3, scope: !40, file: !3, type: !23)
!47 = !DILocalVariable(name: "omp_sched_static", scope: !40, file: !3, type: !9)
!48 = !DILocalVariable(name: "omp_proc_bind_false", scope: !40, file: !3, type: !9)
!49 = !DILocalVariable(name: "omp_proc_bind_true", scope: !40, file: !3, type: !9)
!50 = !DILocalVariable(name: "omp_lock_hint_none", scope: !40, file: !3, type: !9)
!51 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !40, file: !3, type: !9)
!52 = !DILocation(line: 32, column: 1, scope: !40)
!53 = !DILocation(line: 27, column: 1, scope: !40)
!54 = distinct !DISubprogram(name: "__nv_MAIN_F1L27_2", scope: !2, file: !3, line: 27, type: !41, scopeLine: 27, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!55 = !DILocalVariable(name: "__nv_MAIN_F1L27_2Arg0", arg: 1, scope: !54, file: !3, type: !9)
!56 = !DILocation(line: 0, scope: !54)
!57 = !DILocalVariable(name: "__nv_MAIN_F1L27_2Arg1", arg: 2, scope: !54, file: !3, type: !23)
!58 = !DILocalVariable(name: "__nv_MAIN_F1L27_2Arg2", arg: 3, scope: !54, file: !3, type: !23)
!59 = !DILocalVariable(name: "omp_sched_static", scope: !54, file: !3, type: !9)
!60 = !DILocalVariable(name: "omp_proc_bind_false", scope: !54, file: !3, type: !9)
!61 = !DILocalVariable(name: "omp_proc_bind_true", scope: !54, file: !3, type: !9)
!62 = !DILocalVariable(name: "omp_lock_hint_none", scope: !54, file: !3, type: !9)
!63 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !54, file: !3, type: !9)
!64 = !DILocation(line: 30, column: 1, scope: !54)
!65 = !DILocation(line: 28, column: 1, scope: !54)
!66 = !DILocalVariable(name: "i", scope: !54, file: !3, type: !9)
!67 = !DILocation(line: 29, column: 1, scope: !54)

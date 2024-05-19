; ModuleID = '/tmp/DRB013-nowait-orig-yes-42d3c8.ll'
source_filename = "/tmp/DRB013-nowait-orig-yes-42d3c8.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%astruct.dt68 = type <{ i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8* }>
%astruct.dt113 = type <{ [88 x i8] }>

@.C305_MAIN_ = internal constant i32 14
@.C342_MAIN_ = internal constant [7 x i8] c"error ="
@.C339_MAIN_ = internal constant i32 6
@.C336_MAIN_ = internal constant [51 x i8] c"micro-benchmarks-fortran/DRB013-nowait-orig-yes.f95"
@.C338_MAIN_ = internal constant i32 46
@.C331_MAIN_ = internal constant i64 10
@.C285_MAIN_ = internal constant i32 1
@.C306_MAIN_ = internal constant i32 25
@.C352_MAIN_ = internal constant i64 4
@.C351_MAIN_ = internal constant i64 25
@.C317_MAIN_ = internal constant i32 1000
@.C316_MAIN_ = internal constant i32 5
@.C286_MAIN_ = internal constant i64 1
@.C284_MAIN_ = internal constant i64 0
@.C283_MAIN_ = internal constant i32 0
@.C331___nv_MAIN__F1L33_1 = internal constant i64 10
@.C316___nv_MAIN__F1L33_1 = internal constant i32 5
@.C285___nv_MAIN__F1L33_1 = internal constant i32 1
@.C283___nv_MAIN__F1L33_1 = internal constant i32 0
@.C331___nv_MAIN_F1L34_2 = internal constant i64 10
@.C316___nv_MAIN_F1L34_2 = internal constant i32 5
@.C285___nv_MAIN_F1L34_2 = internal constant i32 1
@.C283___nv_MAIN_F1L34_2 = internal constant i32 0

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__430 = alloca i32, align 4
  %.Z0967_319 = alloca i32*, align 8
  %"a$sd1_350" = alloca [16 x i64], align 8
  %b_309 = alloca i32, align 4
  %len_318 = alloca i32, align 4
  %z_b_0_310 = alloca i64, align 8
  %z_b_1_311 = alloca i64, align 8
  %z_e_60_314 = alloca i64, align 8
  %z_b_2_312 = alloca i64, align 8
  %z_b_3_313 = alloca i64, align 8
  %.dY0001_361 = alloca i32, align 4
  %i_307 = alloca i32, align 4
  %.uplevelArgPack0001_404 = alloca %astruct.dt68, align 16
  %error_308 = alloca i32, align 4
  %z__io_341 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !14, metadata !DIExpression()), !dbg !10
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !15
  store i32 %0, i32* %__gtid_MAIN__430, align 4, !dbg !15
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !16
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !16
  call void (i8*, ...) %2(i8* %1), !dbg !16
  call void @llvm.dbg.declare(metadata i32** %.Z0967_319, metadata !17, metadata !DIExpression(DW_OP_deref)), !dbg !10
  %3 = bitcast i32** %.Z0967_319 to i8**, !dbg !16
  store i8* null, i8** %3, align 8, !dbg !16
  call void @llvm.dbg.declare(metadata [16 x i64]* %"a$sd1_350", metadata !21, metadata !DIExpression()), !dbg !10
  %4 = bitcast [16 x i64]* %"a$sd1_350" to i64*, !dbg !16
  store i64 0, i64* %4, align 8, !dbg !16
  br label %L.LB1_382

L.LB1_382:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %b_309, metadata !26, metadata !DIExpression()), !dbg !10
  store i32 5, i32* %b_309, align 4, !dbg !27
  call void @llvm.dbg.declare(metadata i32* %len_318, metadata !28, metadata !DIExpression()), !dbg !10
  store i32 1000, i32* %len_318, align 4, !dbg !29
  call void @llvm.dbg.declare(metadata i64* %z_b_0_310, metadata !30, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_0_310, align 8, !dbg !31
  %5 = load i32, i32* %len_318, align 4, !dbg !31
  call void @llvm.dbg.value(metadata i32 %5, metadata !28, metadata !DIExpression()), !dbg !10
  %6 = sext i32 %5 to i64, !dbg !31
  call void @llvm.dbg.declare(metadata i64* %z_b_1_311, metadata !30, metadata !DIExpression()), !dbg !10
  store i64 %6, i64* %z_b_1_311, align 8, !dbg !31
  %7 = load i64, i64* %z_b_1_311, align 8, !dbg !31
  call void @llvm.dbg.value(metadata i64 %7, metadata !30, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_60_314, metadata !30, metadata !DIExpression()), !dbg !10
  store i64 %7, i64* %z_e_60_314, align 8, !dbg !31
  %8 = bitcast [16 x i64]* %"a$sd1_350" to i8*, !dbg !31
  %9 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !31
  %10 = bitcast i64* @.C351_MAIN_ to i8*, !dbg !31
  %11 = bitcast i64* @.C352_MAIN_ to i8*, !dbg !31
  %12 = bitcast i64* %z_b_0_310 to i8*, !dbg !31
  %13 = bitcast i64* %z_b_1_311 to i8*, !dbg !31
  %14 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !31
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %14(i8* %8, i8* %9, i8* %10, i8* %11, i8* %12, i8* %13), !dbg !31
  %15 = bitcast [16 x i64]* %"a$sd1_350" to i8*, !dbg !31
  %16 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !31
  call void (i8*, i32, ...) %16(i8* %15, i32 25), !dbg !31
  %17 = load i64, i64* %z_b_1_311, align 8, !dbg !31
  call void @llvm.dbg.value(metadata i64 %17, metadata !30, metadata !DIExpression()), !dbg !10
  %18 = load i64, i64* %z_b_0_310, align 8, !dbg !31
  call void @llvm.dbg.value(metadata i64 %18, metadata !30, metadata !DIExpression()), !dbg !10
  %19 = sub nsw i64 %18, 1, !dbg !31
  %20 = sub nsw i64 %17, %19, !dbg !31
  call void @llvm.dbg.declare(metadata i64* %z_b_2_312, metadata !30, metadata !DIExpression()), !dbg !10
  store i64 %20, i64* %z_b_2_312, align 8, !dbg !31
  %21 = load i64, i64* %z_b_0_310, align 8, !dbg !31
  call void @llvm.dbg.value(metadata i64 %21, metadata !30, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_b_3_313, metadata !30, metadata !DIExpression()), !dbg !10
  store i64 %21, i64* %z_b_3_313, align 8, !dbg !31
  %22 = bitcast i64* %z_b_2_312 to i8*, !dbg !31
  %23 = bitcast i64* @.C351_MAIN_ to i8*, !dbg !31
  %24 = bitcast i64* @.C352_MAIN_ to i8*, !dbg !31
  %25 = bitcast i32** %.Z0967_319 to i8*, !dbg !31
  %26 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !31
  %27 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !31
  %28 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !31
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %28(i8* %22, i8* %23, i8* %24, i8* null, i8* %25, i8* null, i8* %26, i8* %27, i8* null, i64 0), !dbg !31
  %29 = load i32, i32* %len_318, align 4, !dbg !32
  call void @llvm.dbg.value(metadata i32 %29, metadata !28, metadata !DIExpression()), !dbg !10
  store i32 %29, i32* %.dY0001_361, align 4, !dbg !32
  call void @llvm.dbg.declare(metadata i32* %i_307, metadata !33, metadata !DIExpression()), !dbg !10
  store i32 1, i32* %i_307, align 4, !dbg !32
  %30 = load i32, i32* %.dY0001_361, align 4, !dbg !32
  %31 = icmp sle i32 %30, 0, !dbg !32
  br i1 %31, label %L.LB1_360, label %L.LB1_359, !dbg !32

L.LB1_359:                                        ; preds = %L.LB1_359, %L.LB1_382
  %32 = load i32, i32* %i_307, align 4, !dbg !34
  call void @llvm.dbg.value(metadata i32 %32, metadata !33, metadata !DIExpression()), !dbg !10
  %33 = load i32, i32* %i_307, align 4, !dbg !34
  call void @llvm.dbg.value(metadata i32 %33, metadata !33, metadata !DIExpression()), !dbg !10
  %34 = sext i32 %33 to i64, !dbg !34
  %35 = bitcast [16 x i64]* %"a$sd1_350" to i8*, !dbg !34
  %36 = getelementptr i8, i8* %35, i64 56, !dbg !34
  %37 = bitcast i8* %36 to i64*, !dbg !34
  %38 = load i64, i64* %37, align 8, !dbg !34
  %39 = add nsw i64 %34, %38, !dbg !34
  %40 = load i32*, i32** %.Z0967_319, align 8, !dbg !34
  call void @llvm.dbg.value(metadata i32* %40, metadata !17, metadata !DIExpression()), !dbg !10
  %41 = bitcast i32* %40 to i8*, !dbg !34
  %42 = getelementptr i8, i8* %41, i64 -4, !dbg !34
  %43 = bitcast i8* %42 to i32*, !dbg !34
  %44 = getelementptr i32, i32* %43, i64 %39, !dbg !34
  store i32 %32, i32* %44, align 4, !dbg !34
  %45 = load i32, i32* %i_307, align 4, !dbg !35
  call void @llvm.dbg.value(metadata i32 %45, metadata !33, metadata !DIExpression()), !dbg !10
  %46 = add nsw i32 %45, 1, !dbg !35
  store i32 %46, i32* %i_307, align 4, !dbg !35
  %47 = load i32, i32* %.dY0001_361, align 4, !dbg !35
  %48 = sub nsw i32 %47, 1, !dbg !35
  store i32 %48, i32* %.dY0001_361, align 4, !dbg !35
  %49 = load i32, i32* %.dY0001_361, align 4, !dbg !35
  %50 = icmp sgt i32 %49, 0, !dbg !35
  br i1 %50, label %L.LB1_359, label %L.LB1_360, !dbg !35

L.LB1_360:                                        ; preds = %L.LB1_359, %L.LB1_382
  %51 = bitcast i32* %len_318 to i8*, !dbg !36
  %52 = bitcast %astruct.dt68* %.uplevelArgPack0001_404 to i8**, !dbg !36
  store i8* %51, i8** %52, align 8, !dbg !36
  %53 = bitcast i32** %.Z0967_319 to i8*, !dbg !36
  %54 = bitcast %astruct.dt68* %.uplevelArgPack0001_404 to i8*, !dbg !36
  %55 = getelementptr i8, i8* %54, i64 8, !dbg !36
  %56 = bitcast i8* %55 to i8**, !dbg !36
  store i8* %53, i8** %56, align 8, !dbg !36
  %57 = bitcast i32** %.Z0967_319 to i8*, !dbg !36
  %58 = bitcast %astruct.dt68* %.uplevelArgPack0001_404 to i8*, !dbg !36
  %59 = getelementptr i8, i8* %58, i64 16, !dbg !36
  %60 = bitcast i8* %59 to i8**, !dbg !36
  store i8* %57, i8** %60, align 8, !dbg !36
  %61 = bitcast i64* %z_b_0_310 to i8*, !dbg !36
  %62 = bitcast %astruct.dt68* %.uplevelArgPack0001_404 to i8*, !dbg !36
  %63 = getelementptr i8, i8* %62, i64 24, !dbg !36
  %64 = bitcast i8* %63 to i8**, !dbg !36
  store i8* %61, i8** %64, align 8, !dbg !36
  %65 = bitcast i64* %z_b_1_311 to i8*, !dbg !36
  %66 = bitcast %astruct.dt68* %.uplevelArgPack0001_404 to i8*, !dbg !36
  %67 = getelementptr i8, i8* %66, i64 32, !dbg !36
  %68 = bitcast i8* %67 to i8**, !dbg !36
  store i8* %65, i8** %68, align 8, !dbg !36
  %69 = bitcast i64* %z_e_60_314 to i8*, !dbg !36
  %70 = bitcast %astruct.dt68* %.uplevelArgPack0001_404 to i8*, !dbg !36
  %71 = getelementptr i8, i8* %70, i64 40, !dbg !36
  %72 = bitcast i8* %71 to i8**, !dbg !36
  store i8* %69, i8** %72, align 8, !dbg !36
  %73 = bitcast i64* %z_b_2_312 to i8*, !dbg !36
  %74 = bitcast %astruct.dt68* %.uplevelArgPack0001_404 to i8*, !dbg !36
  %75 = getelementptr i8, i8* %74, i64 48, !dbg !36
  %76 = bitcast i8* %75 to i8**, !dbg !36
  store i8* %73, i8** %76, align 8, !dbg !36
  %77 = bitcast i64* %z_b_3_313 to i8*, !dbg !36
  %78 = bitcast %astruct.dt68* %.uplevelArgPack0001_404 to i8*, !dbg !36
  %79 = getelementptr i8, i8* %78, i64 56, !dbg !36
  %80 = bitcast i8* %79 to i8**, !dbg !36
  store i8* %77, i8** %80, align 8, !dbg !36
  %81 = bitcast i32* %b_309 to i8*, !dbg !36
  %82 = bitcast %astruct.dt68* %.uplevelArgPack0001_404 to i8*, !dbg !36
  %83 = getelementptr i8, i8* %82, i64 64, !dbg !36
  %84 = bitcast i8* %83 to i8**, !dbg !36
  store i8* %81, i8** %84, align 8, !dbg !36
  call void @llvm.dbg.declare(metadata i32* %error_308, metadata !37, metadata !DIExpression()), !dbg !10
  %85 = bitcast i32* %error_308 to i8*, !dbg !36
  %86 = bitcast %astruct.dt68* %.uplevelArgPack0001_404 to i8*, !dbg !36
  %87 = getelementptr i8, i8* %86, i64 72, !dbg !36
  %88 = bitcast i8* %87 to i8**, !dbg !36
  store i8* %85, i8** %88, align 8, !dbg !36
  %89 = bitcast [16 x i64]* %"a$sd1_350" to i8*, !dbg !36
  %90 = bitcast %astruct.dt68* %.uplevelArgPack0001_404 to i8*, !dbg !36
  %91 = getelementptr i8, i8* %90, i64 80, !dbg !36
  %92 = bitcast i8* %91 to i8**, !dbg !36
  store i8* %89, i8** %92, align 8, !dbg !36
  br label %L.LB1_428, !dbg !36

L.LB1_428:                                        ; preds = %L.LB1_360
  %93 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L33_1_ to i64*, !dbg !36
  %94 = bitcast %astruct.dt68* %.uplevelArgPack0001_404 to i64*, !dbg !36
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %93, i64* %94), !dbg !36
  call void (...) @_mp_bcs_nest(), !dbg !38
  %95 = bitcast i32* @.C338_MAIN_ to i8*, !dbg !38
  %96 = bitcast [51 x i8]* @.C336_MAIN_ to i8*, !dbg !38
  %97 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !38
  call void (i8*, i8*, i64, ...) %97(i8* %95, i8* %96, i64 51), !dbg !38
  %98 = bitcast i32* @.C339_MAIN_ to i8*, !dbg !38
  %99 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !38
  %100 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !38
  %101 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !38
  %102 = call i32 (i8*, i8*, i8*, i8*, ...) %101(i8* %98, i8* null, i8* %99, i8* %100), !dbg !38
  call void @llvm.dbg.declare(metadata i32* %z__io_341, metadata !39, metadata !DIExpression()), !dbg !10
  store i32 %102, i32* %z__io_341, align 4, !dbg !38
  %103 = bitcast [7 x i8]* @.C342_MAIN_ to i8*, !dbg !38
  %104 = bitcast i32 (...)* @f90io_sc_ch_ldw to i32 (i8*, i32, i64, ...)*, !dbg !38
  %105 = call i32 (i8*, i32, i64, ...) %104(i8* %103, i32 14, i64 7), !dbg !38
  store i32 %105, i32* %z__io_341, align 4, !dbg !38
  %106 = load i32, i32* %error_308, align 4, !dbg !38
  call void @llvm.dbg.value(metadata i32 %106, metadata !37, metadata !DIExpression()), !dbg !10
  %107 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !38
  %108 = call i32 (i32, i32, ...) %107(i32 %106, i32 25), !dbg !38
  store i32 %108, i32* %z__io_341, align 4, !dbg !38
  %109 = call i32 (...) @f90io_ldw_end(), !dbg !38
  store i32 %109, i32* %z__io_341, align 4, !dbg !38
  call void (...) @_mp_ecs_nest(), !dbg !38
  %110 = load i32*, i32** %.Z0967_319, align 8, !dbg !40
  call void @llvm.dbg.value(metadata i32* %110, metadata !17, metadata !DIExpression()), !dbg !10
  %111 = bitcast i32* %110 to i8*, !dbg !40
  %112 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !40
  %113 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !40
  call void (i8*, i8*, i8*, i8*, i64, ...) %113(i8* null, i8* %111, i8* %112, i8* null, i64 0), !dbg !40
  %114 = bitcast i32** %.Z0967_319 to i8**, !dbg !40
  store i8* null, i8** %114, align 8, !dbg !40
  %115 = bitcast [16 x i64]* %"a$sd1_350" to i64*, !dbg !40
  store i64 0, i64* %115, align 8, !dbg !40
  ret void, !dbg !15
}

define internal void @__nv_MAIN__F1L33_1_(i32* %__nv_MAIN__F1L33_1Arg0, i64* %__nv_MAIN__F1L33_1Arg1, i64* %__nv_MAIN__F1L33_1Arg2) #0 !dbg !41 {
L.entry:
  %__gtid___nv_MAIN__F1L33_1__470 = alloca i32, align 4
  %.uplevelArgPack0002_465 = alloca %astruct.dt113, align 16
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L33_1Arg0, metadata !44, metadata !DIExpression()), !dbg !45
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L33_1Arg1, metadata !46, metadata !DIExpression()), !dbg !45
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L33_1Arg2, metadata !47, metadata !DIExpression()), !dbg !45
  call void @llvm.dbg.value(metadata i32 1, metadata !48, metadata !DIExpression()), !dbg !45
  call void @llvm.dbg.value(metadata i32 0, metadata !49, metadata !DIExpression()), !dbg !45
  call void @llvm.dbg.value(metadata i32 1, metadata !50, metadata !DIExpression()), !dbg !45
  call void @llvm.dbg.value(metadata i32 0, metadata !51, metadata !DIExpression()), !dbg !45
  call void @llvm.dbg.value(metadata i32 1, metadata !52, metadata !DIExpression()), !dbg !45
  %0 = load i32, i32* %__nv_MAIN__F1L33_1Arg0, align 4, !dbg !53
  store i32 %0, i32* %__gtid___nv_MAIN__F1L33_1__470, align 4, !dbg !53
  br label %L.LB2_460

L.LB2_460:                                        ; preds = %L.entry
  br label %L.LB2_322

L.LB2_322:                                        ; preds = %L.LB2_460
  %1 = bitcast i64* %__nv_MAIN__F1L33_1Arg2 to i8*, !dbg !54
  %2 = bitcast %astruct.dt113* %.uplevelArgPack0002_465 to i8*, !dbg !54
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %2, i8* align 8 %1, i64 88, i1 false), !dbg !54
  br label %L.LB2_468, !dbg !54

L.LB2_468:                                        ; preds = %L.LB2_322
  %3 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN_F1L34_2_ to i64*, !dbg !54
  %4 = bitcast %astruct.dt113* %.uplevelArgPack0002_465 to i64*, !dbg !54
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %3, i64* %4), !dbg !54
  br label %L.LB2_334

L.LB2_334:                                        ; preds = %L.LB2_468
  ret void, !dbg !53
}

define internal void @__nv_MAIN_F1L34_2_(i32* %__nv_MAIN_F1L34_2Arg0, i64* %__nv_MAIN_F1L34_2Arg1, i64* %__nv_MAIN_F1L34_2Arg2) #0 !dbg !55 {
L.entry:
  %__gtid___nv_MAIN_F1L34_2__497 = alloca i32, align 4
  %.i0000p_328 = alloca i32, align 4
  %i_327 = alloca i32, align 4
  %.du0002p_365 = alloca i32, align 4
  %.de0002p_366 = alloca i32, align 4
  %.di0002p_367 = alloca i32, align 4
  %.ds0002p_368 = alloca i32, align 4
  %.dl0002p_370 = alloca i32, align 4
  %.dl0002p.copy_491 = alloca i32, align 4
  %.de0002p.copy_492 = alloca i32, align 4
  %.ds0002p.copy_493 = alloca i32, align 4
  %.dX0002p_369 = alloca i32, align 4
  %.dY0002p_364 = alloca i32, align 4
  %.s0000_518 = alloca i32, align 4
  %.s0001_519 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L34_2Arg0, metadata !56, metadata !DIExpression()), !dbg !57
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L34_2Arg1, metadata !58, metadata !DIExpression()), !dbg !57
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L34_2Arg2, metadata !59, metadata !DIExpression()), !dbg !57
  call void @llvm.dbg.value(metadata i32 1, metadata !60, metadata !DIExpression()), !dbg !57
  call void @llvm.dbg.value(metadata i32 0, metadata !61, metadata !DIExpression()), !dbg !57
  call void @llvm.dbg.value(metadata i32 1, metadata !62, metadata !DIExpression()), !dbg !57
  call void @llvm.dbg.value(metadata i32 0, metadata !63, metadata !DIExpression()), !dbg !57
  call void @llvm.dbg.value(metadata i32 1, metadata !64, metadata !DIExpression()), !dbg !57
  %0 = load i32, i32* %__nv_MAIN_F1L34_2Arg0, align 4, !dbg !65
  store i32 %0, i32* %__gtid___nv_MAIN_F1L34_2__497, align 4, !dbg !65
  br label %L.LB4_482

L.LB4_482:                                        ; preds = %L.entry
  br label %L.LB4_325

L.LB4_325:                                        ; preds = %L.LB4_482
  br label %L.LB4_326

L.LB4_326:                                        ; preds = %L.LB4_325
  store i32 0, i32* %.i0000p_328, align 4, !dbg !66
  call void @llvm.dbg.declare(metadata i32* %i_327, metadata !67, metadata !DIExpression()), !dbg !65
  store i32 1, i32* %i_327, align 4, !dbg !66
  %1 = bitcast i64* %__nv_MAIN_F1L34_2Arg2 to i32**, !dbg !66
  %2 = load i32*, i32** %1, align 8, !dbg !66
  %3 = load i32, i32* %2, align 4, !dbg !66
  store i32 %3, i32* %.du0002p_365, align 4, !dbg !66
  %4 = bitcast i64* %__nv_MAIN_F1L34_2Arg2 to i32**, !dbg !66
  %5 = load i32*, i32** %4, align 8, !dbg !66
  %6 = load i32, i32* %5, align 4, !dbg !66
  store i32 %6, i32* %.de0002p_366, align 4, !dbg !66
  store i32 1, i32* %.di0002p_367, align 4, !dbg !66
  %7 = load i32, i32* %.di0002p_367, align 4, !dbg !66
  store i32 %7, i32* %.ds0002p_368, align 4, !dbg !66
  store i32 1, i32* %.dl0002p_370, align 4, !dbg !66
  %8 = load i32, i32* %.dl0002p_370, align 4, !dbg !66
  store i32 %8, i32* %.dl0002p.copy_491, align 4, !dbg !66
  %9 = load i32, i32* %.de0002p_366, align 4, !dbg !66
  store i32 %9, i32* %.de0002p.copy_492, align 4, !dbg !66
  %10 = load i32, i32* %.ds0002p_368, align 4, !dbg !66
  store i32 %10, i32* %.ds0002p.copy_493, align 4, !dbg !66
  %11 = load i32, i32* %__gtid___nv_MAIN_F1L34_2__497, align 4, !dbg !66
  %12 = bitcast i32* %.i0000p_328 to i64*, !dbg !66
  %13 = bitcast i32* %.dl0002p.copy_491 to i64*, !dbg !66
  %14 = bitcast i32* %.de0002p.copy_492 to i64*, !dbg !66
  %15 = bitcast i32* %.ds0002p.copy_493 to i64*, !dbg !66
  %16 = load i32, i32* %.ds0002p.copy_493, align 4, !dbg !66
  call void @__kmpc_for_static_init_4(i64* null, i32 %11, i32 34, i64* %12, i64* %13, i64* %14, i64* %15, i32 %16, i32 1), !dbg !66
  %17 = load i32, i32* %.dl0002p.copy_491, align 4, !dbg !66
  store i32 %17, i32* %.dl0002p_370, align 4, !dbg !66
  %18 = load i32, i32* %.de0002p.copy_492, align 4, !dbg !66
  store i32 %18, i32* %.de0002p_366, align 4, !dbg !66
  %19 = load i32, i32* %.ds0002p.copy_493, align 4, !dbg !66
  store i32 %19, i32* %.ds0002p_368, align 4, !dbg !66
  %20 = load i32, i32* %.dl0002p_370, align 4, !dbg !66
  store i32 %20, i32* %i_327, align 4, !dbg !66
  %21 = load i32, i32* %i_327, align 4, !dbg !66
  call void @llvm.dbg.value(metadata i32 %21, metadata !67, metadata !DIExpression()), !dbg !65
  store i32 %21, i32* %.dX0002p_369, align 4, !dbg !66
  %22 = load i32, i32* %.dX0002p_369, align 4, !dbg !66
  %23 = load i32, i32* %.du0002p_365, align 4, !dbg !66
  %24 = icmp sgt i32 %22, %23, !dbg !66
  br i1 %24, label %L.LB4_363, label %L.LB4_542, !dbg !66

L.LB4_542:                                        ; preds = %L.LB4_326
  %25 = load i32, i32* %.dX0002p_369, align 4, !dbg !66
  store i32 %25, i32* %i_327, align 4, !dbg !66
  %26 = load i32, i32* %.di0002p_367, align 4, !dbg !66
  %27 = load i32, i32* %.de0002p_366, align 4, !dbg !66
  %28 = load i32, i32* %.dX0002p_369, align 4, !dbg !66
  %29 = sub nsw i32 %27, %28, !dbg !66
  %30 = add nsw i32 %26, %29, !dbg !66
  %31 = load i32, i32* %.di0002p_367, align 4, !dbg !66
  %32 = sdiv i32 %30, %31, !dbg !66
  store i32 %32, i32* %.dY0002p_364, align 4, !dbg !66
  %33 = load i32, i32* %.dY0002p_364, align 4, !dbg !66
  %34 = icmp sle i32 %33, 0, !dbg !66
  br i1 %34, label %L.LB4_373, label %L.LB4_372, !dbg !66

L.LB4_372:                                        ; preds = %L.LB4_372, %L.LB4_542
  %35 = load i32, i32* %i_327, align 4, !dbg !68
  call void @llvm.dbg.value(metadata i32 %35, metadata !67, metadata !DIExpression()), !dbg !65
  %36 = sext i32 %35 to i64, !dbg !68
  %37 = bitcast i64* %__nv_MAIN_F1L34_2Arg2 to i8*, !dbg !68
  %38 = getelementptr i8, i8* %37, i64 80, !dbg !68
  %39 = bitcast i8* %38 to i8**, !dbg !68
  %40 = load i8*, i8** %39, align 8, !dbg !68
  %41 = getelementptr i8, i8* %40, i64 56, !dbg !68
  %42 = bitcast i8* %41 to i64*, !dbg !68
  %43 = load i64, i64* %42, align 8, !dbg !68
  %44 = add nsw i64 %36, %43, !dbg !68
  %45 = bitcast i64* %__nv_MAIN_F1L34_2Arg2 to i8*, !dbg !68
  %46 = getelementptr i8, i8* %45, i64 16, !dbg !68
  %47 = bitcast i8* %46 to i8***, !dbg !68
  %48 = load i8**, i8*** %47, align 8, !dbg !68
  %49 = load i8*, i8** %48, align 8, !dbg !68
  %50 = getelementptr i8, i8* %49, i64 -4, !dbg !68
  %51 = bitcast i8* %50 to i32*, !dbg !68
  %52 = getelementptr i32, i32* %51, i64 %44, !dbg !68
  %53 = load i32, i32* %52, align 4, !dbg !68
  %54 = mul nsw i32 %53, 5, !dbg !68
  %55 = bitcast i64* %__nv_MAIN_F1L34_2Arg2 to i8*, !dbg !68
  %56 = getelementptr i8, i8* %55, i64 64, !dbg !68
  %57 = bitcast i8* %56 to i32**, !dbg !68
  %58 = load i32*, i32** %57, align 8, !dbg !68
  %59 = load i32, i32* %58, align 4, !dbg !68
  %60 = add nsw i32 %54, %59, !dbg !68
  %61 = load i32, i32* %i_327, align 4, !dbg !68
  call void @llvm.dbg.value(metadata i32 %61, metadata !67, metadata !DIExpression()), !dbg !65
  %62 = sext i32 %61 to i64, !dbg !68
  %63 = bitcast i64* %__nv_MAIN_F1L34_2Arg2 to i8*, !dbg !68
  %64 = getelementptr i8, i8* %63, i64 80, !dbg !68
  %65 = bitcast i8* %64 to i8**, !dbg !68
  %66 = load i8*, i8** %65, align 8, !dbg !68
  %67 = getelementptr i8, i8* %66, i64 56, !dbg !68
  %68 = bitcast i8* %67 to i64*, !dbg !68
  %69 = load i64, i64* %68, align 8, !dbg !68
  %70 = add nsw i64 %62, %69, !dbg !68
  %71 = bitcast i64* %__nv_MAIN_F1L34_2Arg2 to i8*, !dbg !68
  %72 = getelementptr i8, i8* %71, i64 16, !dbg !68
  %73 = bitcast i8* %72 to i8***, !dbg !68
  %74 = load i8**, i8*** %73, align 8, !dbg !68
  %75 = load i8*, i8** %74, align 8, !dbg !68
  %76 = getelementptr i8, i8* %75, i64 -4, !dbg !68
  %77 = bitcast i8* %76 to i32*, !dbg !68
  %78 = getelementptr i32, i32* %77, i64 %70, !dbg !68
  store i32 %60, i32* %78, align 4, !dbg !68
  %79 = load i32, i32* %.di0002p_367, align 4, !dbg !69
  %80 = load i32, i32* %i_327, align 4, !dbg !69
  call void @llvm.dbg.value(metadata i32 %80, metadata !67, metadata !DIExpression()), !dbg !65
  %81 = add nsw i32 %79, %80, !dbg !69
  store i32 %81, i32* %i_327, align 4, !dbg !69
  %82 = load i32, i32* %.dY0002p_364, align 4, !dbg !69
  %83 = sub nsw i32 %82, 1, !dbg !69
  store i32 %83, i32* %.dY0002p_364, align 4, !dbg !69
  %84 = load i32, i32* %.dY0002p_364, align 4, !dbg !69
  %85 = icmp sgt i32 %84, 0, !dbg !69
  br i1 %85, label %L.LB4_372, label %L.LB4_373, !dbg !69

L.LB4_373:                                        ; preds = %L.LB4_372, %L.LB4_542
  br label %L.LB4_363

L.LB4_363:                                        ; preds = %L.LB4_373, %L.LB4_326
  %86 = load i32, i32* %__gtid___nv_MAIN_F1L34_2__497, align 4, !dbg !69
  call void @__kmpc_for_static_fini(i64* null, i32 %86), !dbg !69
  br label %L.LB4_329

L.LB4_329:                                        ; preds = %L.LB4_363
  store i32 -1, i32* %.s0000_518, align 4, !dbg !70
  store i32 0, i32* %.s0001_519, align 4, !dbg !70
  %87 = load i32, i32* %__gtid___nv_MAIN_F1L34_2__497, align 4, !dbg !70
  %88 = call i32 @__kmpc_single(i64* null, i32 %87), !dbg !70
  %89 = icmp eq i32 %88, 0, !dbg !70
  br i1 %89, label %L.LB4_374, label %L.LB4_330, !dbg !70

L.LB4_330:                                        ; preds = %L.LB4_329
  %90 = bitcast i64* %__nv_MAIN_F1L34_2Arg2 to i8*, !dbg !71
  %91 = getelementptr i8, i8* %90, i64 80, !dbg !71
  %92 = bitcast i8* %91 to i8**, !dbg !71
  %93 = load i8*, i8** %92, align 8, !dbg !71
  %94 = getelementptr i8, i8* %93, i64 56, !dbg !71
  %95 = bitcast i8* %94 to i64*, !dbg !71
  %96 = load i64, i64* %95, align 8, !dbg !71
  %97 = bitcast i64* %__nv_MAIN_F1L34_2Arg2 to i8*, !dbg !71
  %98 = getelementptr i8, i8* %97, i64 16, !dbg !71
  %99 = bitcast i8* %98 to i8***, !dbg !71
  %100 = load i8**, i8*** %99, align 8, !dbg !71
  %101 = load i8*, i8** %100, align 8, !dbg !71
  %102 = getelementptr i8, i8* %101, i64 36, !dbg !71
  %103 = bitcast i8* %102 to i32*, !dbg !71
  %104 = getelementptr i32, i32* %103, i64 %96, !dbg !71
  %105 = load i32, i32* %104, align 4, !dbg !71
  %106 = add nsw i32 %105, 1, !dbg !71
  %107 = bitcast i64* %__nv_MAIN_F1L34_2Arg2 to i8*, !dbg !71
  %108 = getelementptr i8, i8* %107, i64 72, !dbg !71
  %109 = bitcast i8* %108 to i32**, !dbg !71
  %110 = load i32*, i32** %109, align 8, !dbg !71
  store i32 %106, i32* %110, align 4, !dbg !71
  %111 = load i32, i32* %__gtid___nv_MAIN_F1L34_2__497, align 4, !dbg !72
  store i32 %111, i32* %.s0000_518, align 4, !dbg !72
  store i32 1, i32* %.s0001_519, align 4, !dbg !72
  %112 = load i32, i32* %__gtid___nv_MAIN_F1L34_2__497, align 4, !dbg !72
  call void @__kmpc_end_single(i64* null, i32 %112), !dbg !72
  br label %L.LB4_374

L.LB4_374:                                        ; preds = %L.LB4_330, %L.LB4_329
  br label %L.LB4_332

L.LB4_332:                                        ; preds = %L.LB4_374
  %113 = load i32, i32* %__gtid___nv_MAIN_F1L34_2__497, align 4, !dbg !72
  call void @__kmpc_barrier(i64* null, i32 %113), !dbg !72
  br label %L.LB4_333

L.LB4_333:                                        ; preds = %L.LB4_332
  ret void, !dbg !65
}

declare void @__kmpc_barrier(i64*, i32) #0

declare void @__kmpc_end_single(i64*, i32) #0

declare signext i32 @__kmpc_single(i64*, i32) #0

declare void @__kmpc_for_static_fini(i64*, i32) #0

declare void @__kmpc_for_static_init_4(i64*, i32, i32, i64*, i64*, i64*, i64*, i32, i32) #0

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

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !4, globals: !4, imports: !4)
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB013-nowait-orig-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb013_nowait_orig_yes", scope: !2, file: !3, line: 17, type: !6, scopeLine: 17, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!6 = !DISubroutineType(cc: DW_CC_program, types: !7)
!7 = !{null}
!8 = !DILocalVariable(name: "omp_sched_static", scope: !5, file: !3, type: !9)
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 0, scope: !5)
!11 = !DILocalVariable(name: "omp_proc_bind_false", scope: !5, file: !3, type: !9)
!12 = !DILocalVariable(name: "omp_proc_bind_true", scope: !5, file: !3, type: !9)
!13 = !DILocalVariable(name: "omp_lock_hint_none", scope: !5, file: !3, type: !9)
!14 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !5, file: !3, type: !9)
!15 = !DILocation(line: 49, column: 1, scope: !5)
!16 = !DILocation(line: 17, column: 1, scope: !5)
!17 = !DILocalVariable(name: "a", scope: !5, file: !3, type: !18)
!18 = !DICompositeType(tag: DW_TAG_array_type, baseType: !9, size: 32, align: 32, elements: !19)
!19 = !{!20}
!20 = !DISubrange(count: 0, lowerBound: 1)
!21 = !DILocalVariable(scope: !5, file: !3, type: !22, flags: DIFlagArtificial)
!22 = !DICompositeType(tag: DW_TAG_array_type, baseType: !23, size: 1024, align: 64, elements: !24)
!23 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!24 = !{!25}
!25 = !DISubrange(count: 16, lowerBound: 1)
!26 = !DILocalVariable(name: "b", scope: !5, file: !3, type: !9)
!27 = !DILocation(line: 24, column: 1, scope: !5)
!28 = !DILocalVariable(name: "len", scope: !5, file: !3, type: !9)
!29 = !DILocation(line: 25, column: 1, scope: !5)
!30 = !DILocalVariable(scope: !5, file: !3, type: !23, flags: DIFlagArtificial)
!31 = !DILocation(line: 27, column: 1, scope: !5)
!32 = !DILocation(line: 29, column: 1, scope: !5)
!33 = !DILocalVariable(name: "i", scope: !5, file: !3, type: !9)
!34 = !DILocation(line: 30, column: 1, scope: !5)
!35 = !DILocation(line: 31, column: 1, scope: !5)
!36 = !DILocation(line: 33, column: 1, scope: !5)
!37 = !DILocalVariable(name: "error", scope: !5, file: !3, type: !9)
!38 = !DILocation(line: 46, column: 1, scope: !5)
!39 = !DILocalVariable(scope: !5, file: !3, type: !9, flags: DIFlagArtificial)
!40 = !DILocation(line: 48, column: 1, scope: !5)
!41 = distinct !DISubprogram(name: "__nv_MAIN__F1L33_1", scope: !2, file: !3, line: 33, type: !42, scopeLine: 33, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!42 = !DISubroutineType(types: !43)
!43 = !{null, !9, !23, !23}
!44 = !DILocalVariable(name: "__nv_MAIN__F1L33_1Arg0", arg: 1, scope: !41, file: !3, type: !9)
!45 = !DILocation(line: 0, scope: !41)
!46 = !DILocalVariable(name: "__nv_MAIN__F1L33_1Arg1", arg: 2, scope: !41, file: !3, type: !23)
!47 = !DILocalVariable(name: "__nv_MAIN__F1L33_1Arg2", arg: 3, scope: !41, file: !3, type: !23)
!48 = !DILocalVariable(name: "omp_sched_static", scope: !41, file: !3, type: !9)
!49 = !DILocalVariable(name: "omp_proc_bind_false", scope: !41, file: !3, type: !9)
!50 = !DILocalVariable(name: "omp_proc_bind_true", scope: !41, file: !3, type: !9)
!51 = !DILocalVariable(name: "omp_lock_hint_none", scope: !41, file: !3, type: !9)
!52 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !41, file: !3, type: !9)
!53 = !DILocation(line: 44, column: 1, scope: !41)
!54 = !DILocation(line: 34, column: 1, scope: !41)
!55 = distinct !DISubprogram(name: "__nv_MAIN_F1L34_2", scope: !2, file: !3, line: 34, type: !42, scopeLine: 34, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!56 = !DILocalVariable(name: "__nv_MAIN_F1L34_2Arg0", arg: 1, scope: !55, file: !3, type: !9)
!57 = !DILocation(line: 0, scope: !55)
!58 = !DILocalVariable(name: "__nv_MAIN_F1L34_2Arg1", arg: 2, scope: !55, file: !3, type: !23)
!59 = !DILocalVariable(name: "__nv_MAIN_F1L34_2Arg2", arg: 3, scope: !55, file: !3, type: !23)
!60 = !DILocalVariable(name: "omp_sched_static", scope: !55, file: !3, type: !9)
!61 = !DILocalVariable(name: "omp_proc_bind_false", scope: !55, file: !3, type: !9)
!62 = !DILocalVariable(name: "omp_proc_bind_true", scope: !55, file: !3, type: !9)
!63 = !DILocalVariable(name: "omp_lock_hint_none", scope: !55, file: !3, type: !9)
!64 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !55, file: !3, type: !9)
!65 = !DILocation(line: 43, column: 1, scope: !55)
!66 = !DILocation(line: 36, column: 1, scope: !55)
!67 = !DILocalVariable(name: "i", scope: !55, file: !3, type: !9)
!68 = !DILocation(line: 37, column: 1, scope: !55)
!69 = !DILocation(line: 38, column: 1, scope: !55)
!70 = !DILocation(line: 40, column: 1, scope: !55)
!71 = !DILocation(line: 41, column: 1, scope: !55)
!72 = !DILocation(line: 42, column: 1, scope: !55)

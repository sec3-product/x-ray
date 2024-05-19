; ModuleID = '/tmp/DRB117-taskwait-waitonlychild-orig-yes-69561b.ll'
source_filename = "/tmp/DRB117-taskwait-waitonlychild-orig-yes-69561b.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%astruct.dt82 = type <{ i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8* }>

@.C320_MAIN_ = internal constant i32 14
@.C366_MAIN_ = internal constant [5 x i8] c"sum ="
@.C363_MAIN_ = internal constant i32 6
@.C360_MAIN_ = internal constant [67 x i8] c"micro-benchmarks-fortran/DRB117-taskwait-waitonlychild-orig-yes.f95"
@.C362_MAIN_ = internal constant i32 40
@.C353_MAIN_ = internal constant i64 3
@.C352_MAIN_ = internal constant i64 2
@.C300_MAIN_ = internal constant i32 4
@.C285_MAIN_ = internal constant i32 1
@.C383_MAIN_ = internal constant i32 2
@.C301_MAIN_ = internal constant i32 2
@.C321_MAIN_ = internal constant i32 25
@.C375_MAIN_ = internal constant i64 25
@.C335_MAIN_ = internal constant i64 4
@.C286_MAIN_ = internal constant i64 1
@.C284_MAIN_ = internal constant i64 0
@.C283_MAIN_ = internal constant i32 0
@.C286___nv_MAIN__F1L21_1 = internal constant i64 1
@.C335___nv_MAIN__F1L21_1 = internal constant i64 4
@.C353___nv_MAIN__F1L21_1 = internal constant i64 3
@.C352___nv_MAIN__F1L21_1 = internal constant i64 2
@.C300___nv_MAIN__F1L21_1 = internal constant i32 4
@.C285___nv_MAIN__F1L21_1 = internal constant i32 1
@.C283___nv_MAIN__F1L21_1 = internal constant i32 0
@.C286___nv_MAIN_F1L29_2 = internal constant i64 1
@.C335___nv_MAIN_F1L29_2 = internal constant i64 4
@.C353___nv_MAIN_F1L29_2 = internal constant i64 3
@.C352___nv_MAIN_F1L29_2 = internal constant i64 2
@.C283___nv_MAIN_F1L29_2 = internal constant i32 0
@.C285___nv_MAIN_F1L29_2 = internal constant i32 1
@.C335___nv_MAIN_F1L30_3 = internal constant i64 4
@.C353___nv_MAIN_F1L30_3 = internal constant i64 3
@.C352___nv_MAIN_F1L30_3 = internal constant i64 2

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__470 = alloca i32, align 4
  %.Z0965_337 = alloca i32*, align 8
  %"psum$sd2_377" = alloca [16 x i64], align 8
  %.Z0964_336 = alloca i32*, align 8
  %"a$sd1_374" = alloca [16 x i64], align 8
  %z_b_0_322 = alloca i64, align 8
  %z_b_1_323 = alloca i64, align 8
  %z_e_60_326 = alloca i64, align 8
  %z_b_2_324 = alloca i64, align 8
  %z_b_3_325 = alloca i64, align 8
  %z_b_4_329 = alloca i64, align 8
  %z_b_5_330 = alloca i64, align 8
  %z_e_67_333 = alloca i64, align 8
  %z_b_6_331 = alloca i64, align 8
  %z_b_7_332 = alloca i64, align 8
  %.uplevelArgPack0001_432 = alloca %astruct.dt82, align 16
  %sum_356 = alloca i32, align 4
  %z__io_365 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 4, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 4, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 4, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 4, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 4, metadata !14, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 4, metadata !15, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 4, metadata !16, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !17, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 2, metadata !18, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 4, metadata !19, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !20, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !21, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 2, metadata !22, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 4, metadata !23, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !24, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !25, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 2, metadata !26, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 4, metadata !27, metadata !DIExpression()), !dbg !10
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !28
  store i32 %0, i32* %__gtid_MAIN__470, align 4, !dbg !28
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !29
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !29
  call void (i8*, ...) %2(i8* %1), !dbg !29
  call void @llvm.dbg.declare(metadata i32** %.Z0965_337, metadata !30, metadata !DIExpression(DW_OP_deref)), !dbg !10
  %3 = bitcast i32** %.Z0965_337 to i8**, !dbg !29
  store i8* null, i8** %3, align 8, !dbg !29
  call void @llvm.dbg.declare(metadata [16 x i64]* %"psum$sd2_377", metadata !34, metadata !DIExpression()), !dbg !10
  %4 = bitcast [16 x i64]* %"psum$sd2_377" to i64*, !dbg !29
  store i64 0, i64* %4, align 8, !dbg !29
  call void @llvm.dbg.declare(metadata i32** %.Z0964_336, metadata !39, metadata !DIExpression(DW_OP_deref)), !dbg !10
  %5 = bitcast i32** %.Z0964_336 to i8**, !dbg !29
  store i8* null, i8** %5, align 8, !dbg !29
  call void @llvm.dbg.declare(metadata [16 x i64]* %"a$sd1_374", metadata !34, metadata !DIExpression()), !dbg !10
  %6 = bitcast [16 x i64]* %"a$sd1_374" to i64*, !dbg !29
  store i64 0, i64* %6, align 8, !dbg !29
  br label %L.LB1_413

L.LB1_413:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i64* %z_b_0_322, metadata !40, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_0_322, align 8, !dbg !41
  call void @llvm.dbg.declare(metadata i64* %z_b_1_323, metadata !40, metadata !DIExpression()), !dbg !10
  store i64 4, i64* %z_b_1_323, align 8, !dbg !41
  %7 = load i64, i64* %z_b_1_323, align 8, !dbg !41
  call void @llvm.dbg.value(metadata i64 %7, metadata !40, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_60_326, metadata !40, metadata !DIExpression()), !dbg !10
  store i64 %7, i64* %z_e_60_326, align 8, !dbg !41
  %8 = bitcast [16 x i64]* %"a$sd1_374" to i8*, !dbg !41
  %9 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !41
  %10 = bitcast i64* @.C375_MAIN_ to i8*, !dbg !41
  %11 = bitcast i64* @.C335_MAIN_ to i8*, !dbg !41
  %12 = bitcast i64* %z_b_0_322 to i8*, !dbg !41
  %13 = bitcast i64* %z_b_1_323 to i8*, !dbg !41
  %14 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !41
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %14(i8* %8, i8* %9, i8* %10, i8* %11, i8* %12, i8* %13), !dbg !41
  %15 = bitcast [16 x i64]* %"a$sd1_374" to i8*, !dbg !41
  %16 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !41
  call void (i8*, i32, ...) %16(i8* %15, i32 25), !dbg !41
  %17 = load i64, i64* %z_b_1_323, align 8, !dbg !41
  call void @llvm.dbg.value(metadata i64 %17, metadata !40, metadata !DIExpression()), !dbg !10
  %18 = load i64, i64* %z_b_0_322, align 8, !dbg !41
  call void @llvm.dbg.value(metadata i64 %18, metadata !40, metadata !DIExpression()), !dbg !10
  %19 = sub nsw i64 %18, 1, !dbg !41
  %20 = sub nsw i64 %17, %19, !dbg !41
  call void @llvm.dbg.declare(metadata i64* %z_b_2_324, metadata !40, metadata !DIExpression()), !dbg !10
  store i64 %20, i64* %z_b_2_324, align 8, !dbg !41
  %21 = load i64, i64* %z_b_0_322, align 8, !dbg !41
  call void @llvm.dbg.value(metadata i64 %21, metadata !40, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_b_3_325, metadata !40, metadata !DIExpression()), !dbg !10
  store i64 %21, i64* %z_b_3_325, align 8, !dbg !41
  %22 = bitcast i64* %z_b_2_324 to i8*, !dbg !41
  %23 = bitcast i64* @.C375_MAIN_ to i8*, !dbg !41
  %24 = bitcast i64* @.C335_MAIN_ to i8*, !dbg !41
  %25 = bitcast i32** %.Z0964_336 to i8*, !dbg !41
  %26 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !41
  %27 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !41
  %28 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !41
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %28(i8* %22, i8* %23, i8* %24, i8* null, i8* %25, i8* null, i8* %26, i8* %27, i8* null, i64 0), !dbg !41
  call void @llvm.dbg.declare(metadata i64* %z_b_4_329, metadata !40, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_4_329, align 8, !dbg !42
  call void @llvm.dbg.declare(metadata i64* %z_b_5_330, metadata !40, metadata !DIExpression()), !dbg !10
  store i64 4, i64* %z_b_5_330, align 8, !dbg !42
  %29 = load i64, i64* %z_b_5_330, align 8, !dbg !42
  call void @llvm.dbg.value(metadata i64 %29, metadata !40, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_67_333, metadata !40, metadata !DIExpression()), !dbg !10
  store i64 %29, i64* %z_e_67_333, align 8, !dbg !42
  %30 = bitcast [16 x i64]* %"psum$sd2_377" to i8*, !dbg !42
  %31 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !42
  %32 = bitcast i64* @.C375_MAIN_ to i8*, !dbg !42
  %33 = bitcast i64* @.C335_MAIN_ to i8*, !dbg !42
  %34 = bitcast i64* %z_b_4_329 to i8*, !dbg !42
  %35 = bitcast i64* %z_b_5_330 to i8*, !dbg !42
  %36 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !42
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %36(i8* %30, i8* %31, i8* %32, i8* %33, i8* %34, i8* %35), !dbg !42
  %37 = bitcast [16 x i64]* %"psum$sd2_377" to i8*, !dbg !42
  %38 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !42
  call void (i8*, i32, ...) %38(i8* %37, i32 25), !dbg !42
  %39 = load i64, i64* %z_b_5_330, align 8, !dbg !42
  call void @llvm.dbg.value(metadata i64 %39, metadata !40, metadata !DIExpression()), !dbg !10
  %40 = load i64, i64* %z_b_4_329, align 8, !dbg !42
  call void @llvm.dbg.value(metadata i64 %40, metadata !40, metadata !DIExpression()), !dbg !10
  %41 = sub nsw i64 %40, 1, !dbg !42
  %42 = sub nsw i64 %39, %41, !dbg !42
  call void @llvm.dbg.declare(metadata i64* %z_b_6_331, metadata !40, metadata !DIExpression()), !dbg !10
  store i64 %42, i64* %z_b_6_331, align 8, !dbg !42
  %43 = load i64, i64* %z_b_4_329, align 8, !dbg !42
  call void @llvm.dbg.value(metadata i64 %43, metadata !40, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_b_7_332, metadata !40, metadata !DIExpression()), !dbg !10
  store i64 %43, i64* %z_b_7_332, align 8, !dbg !42
  %44 = bitcast i64* %z_b_6_331 to i8*, !dbg !42
  %45 = bitcast i64* @.C375_MAIN_ to i8*, !dbg !42
  %46 = bitcast i64* @.C335_MAIN_ to i8*, !dbg !42
  %47 = bitcast i32** %.Z0965_337 to i8*, !dbg !42
  %48 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !42
  %49 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !42
  %50 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !42
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %50(i8* %44, i8* %45, i8* %46, i8* null, i8* %47, i8* null, i8* %48, i8* %49, i8* null, i64 0), !dbg !42
  %51 = bitcast i32** %.Z0964_336 to i8*, !dbg !43
  %52 = bitcast %astruct.dt82* %.uplevelArgPack0001_432 to i8**, !dbg !43
  store i8* %51, i8** %52, align 8, !dbg !43
  %53 = bitcast i32** %.Z0964_336 to i8*, !dbg !43
  %54 = bitcast %astruct.dt82* %.uplevelArgPack0001_432 to i8*, !dbg !43
  %55 = getelementptr i8, i8* %54, i64 8, !dbg !43
  %56 = bitcast i8* %55 to i8**, !dbg !43
  store i8* %53, i8** %56, align 8, !dbg !43
  %57 = bitcast i64* %z_b_0_322 to i8*, !dbg !43
  %58 = bitcast %astruct.dt82* %.uplevelArgPack0001_432 to i8*, !dbg !43
  %59 = getelementptr i8, i8* %58, i64 16, !dbg !43
  %60 = bitcast i8* %59 to i8**, !dbg !43
  store i8* %57, i8** %60, align 8, !dbg !43
  %61 = bitcast i64* %z_b_1_323 to i8*, !dbg !43
  %62 = bitcast %astruct.dt82* %.uplevelArgPack0001_432 to i8*, !dbg !43
  %63 = getelementptr i8, i8* %62, i64 24, !dbg !43
  %64 = bitcast i8* %63 to i8**, !dbg !43
  store i8* %61, i8** %64, align 8, !dbg !43
  %65 = bitcast i64* %z_e_60_326 to i8*, !dbg !43
  %66 = bitcast %astruct.dt82* %.uplevelArgPack0001_432 to i8*, !dbg !43
  %67 = getelementptr i8, i8* %66, i64 32, !dbg !43
  %68 = bitcast i8* %67 to i8**, !dbg !43
  store i8* %65, i8** %68, align 8, !dbg !43
  %69 = bitcast i64* %z_b_2_324 to i8*, !dbg !43
  %70 = bitcast %astruct.dt82* %.uplevelArgPack0001_432 to i8*, !dbg !43
  %71 = getelementptr i8, i8* %70, i64 40, !dbg !43
  %72 = bitcast i8* %71 to i8**, !dbg !43
  store i8* %69, i8** %72, align 8, !dbg !43
  %73 = bitcast i64* %z_b_3_325 to i8*, !dbg !43
  %74 = bitcast %astruct.dt82* %.uplevelArgPack0001_432 to i8*, !dbg !43
  %75 = getelementptr i8, i8* %74, i64 48, !dbg !43
  %76 = bitcast i8* %75 to i8**, !dbg !43
  store i8* %73, i8** %76, align 8, !dbg !43
  %77 = bitcast i32** %.Z0965_337 to i8*, !dbg !43
  %78 = bitcast %astruct.dt82* %.uplevelArgPack0001_432 to i8*, !dbg !43
  %79 = getelementptr i8, i8* %78, i64 56, !dbg !43
  %80 = bitcast i8* %79 to i8**, !dbg !43
  store i8* %77, i8** %80, align 8, !dbg !43
  %81 = bitcast i32** %.Z0965_337 to i8*, !dbg !43
  %82 = bitcast %astruct.dt82* %.uplevelArgPack0001_432 to i8*, !dbg !43
  %83 = getelementptr i8, i8* %82, i64 64, !dbg !43
  %84 = bitcast i8* %83 to i8**, !dbg !43
  store i8* %81, i8** %84, align 8, !dbg !43
  %85 = bitcast i64* %z_b_4_329 to i8*, !dbg !43
  %86 = bitcast %astruct.dt82* %.uplevelArgPack0001_432 to i8*, !dbg !43
  %87 = getelementptr i8, i8* %86, i64 72, !dbg !43
  %88 = bitcast i8* %87 to i8**, !dbg !43
  store i8* %85, i8** %88, align 8, !dbg !43
  %89 = bitcast i64* %z_b_5_330 to i8*, !dbg !43
  %90 = bitcast %astruct.dt82* %.uplevelArgPack0001_432 to i8*, !dbg !43
  %91 = getelementptr i8, i8* %90, i64 80, !dbg !43
  %92 = bitcast i8* %91 to i8**, !dbg !43
  store i8* %89, i8** %92, align 8, !dbg !43
  %93 = bitcast i64* %z_e_67_333 to i8*, !dbg !43
  %94 = bitcast %astruct.dt82* %.uplevelArgPack0001_432 to i8*, !dbg !43
  %95 = getelementptr i8, i8* %94, i64 88, !dbg !43
  %96 = bitcast i8* %95 to i8**, !dbg !43
  store i8* %93, i8** %96, align 8, !dbg !43
  %97 = bitcast i64* %z_b_6_331 to i8*, !dbg !43
  %98 = bitcast %astruct.dt82* %.uplevelArgPack0001_432 to i8*, !dbg !43
  %99 = getelementptr i8, i8* %98, i64 96, !dbg !43
  %100 = bitcast i8* %99 to i8**, !dbg !43
  store i8* %97, i8** %100, align 8, !dbg !43
  %101 = bitcast i64* %z_b_7_332 to i8*, !dbg !43
  %102 = bitcast %astruct.dt82* %.uplevelArgPack0001_432 to i8*, !dbg !43
  %103 = getelementptr i8, i8* %102, i64 104, !dbg !43
  %104 = bitcast i8* %103 to i8**, !dbg !43
  store i8* %101, i8** %104, align 8, !dbg !43
  call void @llvm.dbg.declare(metadata i32* %sum_356, metadata !44, metadata !DIExpression()), !dbg !10
  %105 = bitcast i32* %sum_356 to i8*, !dbg !43
  %106 = bitcast %astruct.dt82* %.uplevelArgPack0001_432 to i8*, !dbg !43
  %107 = getelementptr i8, i8* %106, i64 112, !dbg !43
  %108 = bitcast i8* %107 to i8**, !dbg !43
  store i8* %105, i8** %108, align 8, !dbg !43
  %109 = bitcast [16 x i64]* %"a$sd1_374" to i8*, !dbg !43
  %110 = bitcast %astruct.dt82* %.uplevelArgPack0001_432 to i8*, !dbg !43
  %111 = getelementptr i8, i8* %110, i64 120, !dbg !43
  %112 = bitcast i8* %111 to i8**, !dbg !43
  store i8* %109, i8** %112, align 8, !dbg !43
  %113 = bitcast [16 x i64]* %"psum$sd2_377" to i8*, !dbg !43
  %114 = bitcast %astruct.dt82* %.uplevelArgPack0001_432 to i8*, !dbg !43
  %115 = getelementptr i8, i8* %114, i64 128, !dbg !43
  %116 = bitcast i8* %115 to i8**, !dbg !43
  store i8* %113, i8** %116, align 8, !dbg !43
  br label %L.LB1_468, !dbg !43

L.LB1_468:                                        ; preds = %L.LB1_413
  %117 = load i32, i32* %__gtid_MAIN__470, align 4, !dbg !43
  call void @__kmpc_push_num_threads(i64* null, i32 %117, i32 2), !dbg !43
  %118 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L21_1_ to i64*, !dbg !43
  %119 = bitcast %astruct.dt82* %.uplevelArgPack0001_432 to i64*, !dbg !43
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %118, i64* %119), !dbg !43
  call void (...) @_mp_bcs_nest(), !dbg !45
  %120 = bitcast i32* @.C362_MAIN_ to i8*, !dbg !45
  %121 = bitcast [67 x i8]* @.C360_MAIN_ to i8*, !dbg !45
  %122 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !45
  call void (i8*, i8*, i64, ...) %122(i8* %120, i8* %121, i64 67), !dbg !45
  %123 = bitcast i32* @.C363_MAIN_ to i8*, !dbg !45
  %124 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !45
  %125 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !45
  %126 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !45
  %127 = call i32 (i8*, i8*, i8*, i8*, ...) %126(i8* %123, i8* null, i8* %124, i8* %125), !dbg !45
  call void @llvm.dbg.declare(metadata i32* %z__io_365, metadata !46, metadata !DIExpression()), !dbg !10
  store i32 %127, i32* %z__io_365, align 4, !dbg !45
  %128 = bitcast [5 x i8]* @.C366_MAIN_ to i8*, !dbg !45
  %129 = bitcast i32 (...)* @f90io_sc_ch_ldw to i32 (i8*, i32, i64, ...)*, !dbg !45
  %130 = call i32 (i8*, i32, i64, ...) %129(i8* %128, i32 14, i64 5), !dbg !45
  store i32 %130, i32* %z__io_365, align 4, !dbg !45
  %131 = load i32, i32* %sum_356, align 4, !dbg !45
  call void @llvm.dbg.value(metadata i32 %131, metadata !44, metadata !DIExpression()), !dbg !10
  %132 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !45
  %133 = call i32 (i32, i32, ...) %132(i32 %131, i32 25), !dbg !45
  store i32 %133, i32* %z__io_365, align 4, !dbg !45
  %134 = call i32 (...) @f90io_ldw_end(), !dbg !45
  store i32 %134, i32* %z__io_365, align 4, !dbg !45
  call void (...) @_mp_ecs_nest(), !dbg !45
  %135 = load i32*, i32** %.Z0964_336, align 8, !dbg !47
  call void @llvm.dbg.value(metadata i32* %135, metadata !39, metadata !DIExpression()), !dbg !10
  %136 = bitcast i32* %135 to i8*, !dbg !47
  %137 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !47
  %138 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !47
  call void (i8*, i8*, i8*, i8*, i64, ...) %138(i8* null, i8* %136, i8* %137, i8* null, i64 0), !dbg !47
  %139 = bitcast i32** %.Z0964_336 to i8**, !dbg !47
  store i8* null, i8** %139, align 8, !dbg !47
  %140 = bitcast [16 x i64]* %"a$sd1_374" to i64*, !dbg !47
  store i64 0, i64* %140, align 8, !dbg !47
  %141 = load i32*, i32** %.Z0965_337, align 8, !dbg !47
  call void @llvm.dbg.value(metadata i32* %141, metadata !30, metadata !DIExpression()), !dbg !10
  %142 = bitcast i32* %141 to i8*, !dbg !47
  %143 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !47
  %144 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !47
  call void (i8*, i8*, i8*, i8*, i64, ...) %144(i8* null, i8* %142, i8* %143, i8* null, i64 0), !dbg !47
  %145 = bitcast i32** %.Z0965_337 to i8**, !dbg !47
  store i8* null, i8** %145, align 8, !dbg !47
  %146 = bitcast [16 x i64]* %"psum$sd2_377" to i64*, !dbg !47
  store i64 0, i64* %146, align 8, !dbg !47
  ret void, !dbg !28
}

define internal void @__nv_MAIN__F1L21_1_(i32* %__nv_MAIN__F1L21_1Arg0, i64* %__nv_MAIN__F1L21_1Arg1, i64* %__nv_MAIN__F1L21_1Arg2) #0 !dbg !48 {
L.entry:
  %__gtid___nv_MAIN__F1L21_1__521 = alloca i32, align 4
  %.i0000p_343 = alloca i32, align 4
  %i_342 = alloca i32, align 4
  %.dY0001p_386 = alloca i32, align 4
  %.du0001p_391 = alloca i32, align 4
  %.de0001p_392 = alloca i32, align 4
  %.di0001p_393 = alloca i32, align 4
  %.ds0001p_394 = alloca i32, align 4
  %.dx0001p_396 = alloca i32, align 4
  %.dl0001p_397 = alloca i32, align 4
  %.dU0001p_398 = alloca i32, align 4
  %.dl0001p.copy_515 = alloca i32, align 4
  %.dU0001p.copy_516 = alloca i32, align 4
  %.ds0001p.copy_517 = alloca i32, align 4
  %.dX0001p_395 = alloca i32, align 4
  %.s0000_549 = alloca i32, align 4
  %.s0001_550 = alloca i32, align 4
  %.s0002_558 = alloca i32, align 4
  %.z0428_557 = alloca i8*, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L21_1Arg0, metadata !51, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L21_1Arg1, metadata !53, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L21_1Arg2, metadata !54, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.value(metadata i32 4, metadata !55, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.value(metadata i32 4, metadata !56, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.value(metadata i32 4, metadata !57, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.value(metadata i32 4, metadata !58, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.value(metadata i32 4, metadata !59, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.value(metadata i32 4, metadata !60, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.value(metadata i32 4, metadata !61, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.value(metadata i32 1, metadata !62, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.value(metadata i32 2, metadata !63, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.value(metadata i32 4, metadata !64, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.value(metadata i32 0, metadata !65, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.value(metadata i32 1, metadata !66, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.value(metadata i32 2, metadata !67, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.value(metadata i32 4, metadata !68, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.value(metadata i32 0, metadata !69, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.value(metadata i32 1, metadata !70, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.value(metadata i32 2, metadata !71, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.value(metadata i32 4, metadata !72, metadata !DIExpression()), !dbg !52
  %0 = load i32, i32* %__nv_MAIN__F1L21_1Arg0, align 4, !dbg !73
  store i32 %0, i32* %__gtid___nv_MAIN__F1L21_1__521, align 4, !dbg !73
  br label %L.LB2_504

L.LB2_504:                                        ; preds = %L.entry
  br label %L.LB2_340

L.LB2_340:                                        ; preds = %L.LB2_504
  br label %L.LB2_341

L.LB2_341:                                        ; preds = %L.LB2_340
  store i32 0, i32* %.i0000p_343, align 4, !dbg !74
  call void @llvm.dbg.declare(metadata i32* %i_342, metadata !75, metadata !DIExpression()), !dbg !73
  store i32 1, i32* %i_342, align 4, !dbg !74
  store i32 4, i32* %.dY0001p_386, align 4, !dbg !74
  store i32 1, i32* %i_342, align 4, !dbg !74
  store i32 4, i32* %.du0001p_391, align 4, !dbg !74
  store i32 4, i32* %.de0001p_392, align 4, !dbg !74
  store i32 1, i32* %.di0001p_393, align 4, !dbg !74
  %1 = load i32, i32* %.di0001p_393, align 4, !dbg !74
  store i32 %1, i32* %.ds0001p_394, align 4, !dbg !74
  store i32 1, i32* %.dx0001p_396, align 4, !dbg !74
  store i32 1, i32* %.dl0001p_397, align 4, !dbg !74
  store i32 4, i32* %.dU0001p_398, align 4, !dbg !74
  %2 = load i32, i32* %.dl0001p_397, align 4, !dbg !74
  store i32 %2, i32* %.dl0001p.copy_515, align 4, !dbg !74
  %3 = load i32, i32* %.dU0001p_398, align 4, !dbg !74
  store i32 %3, i32* %.dU0001p.copy_516, align 4, !dbg !74
  %4 = load i32, i32* %.ds0001p_394, align 4, !dbg !74
  store i32 %4, i32* %.ds0001p.copy_517, align 4, !dbg !74
  %5 = load i32, i32* %__gtid___nv_MAIN__F1L21_1__521, align 4, !dbg !74
  %6 = load i32, i32* %.dl0001p.copy_515, align 4, !dbg !74
  %7 = load i32, i32* %.dU0001p.copy_516, align 4, !dbg !74
  %8 = load i32, i32* %.ds0001p.copy_517, align 4, !dbg !74
  call void @__kmpc_dispatch_init_4(i64* null, i32 %5, i32 35, i32 %6, i32 %7, i32 %8, i32 1), !dbg !74
  %9 = load i32, i32* %.dl0001p.copy_515, align 4, !dbg !74
  store i32 %9, i32* %.dl0001p_397, align 4, !dbg !74
  %10 = load i32, i32* %.dU0001p.copy_516, align 4, !dbg !74
  store i32 %10, i32* %.dU0001p_398, align 4, !dbg !74
  %11 = load i32, i32* %.ds0001p.copy_517, align 4, !dbg !74
  store i32 %11, i32* %.ds0001p_394, align 4, !dbg !74
  br label %L.LB2_384

L.LB2_384:                                        ; preds = %L.LB2_400, %L.LB2_341
  %12 = load i32, i32* %__gtid___nv_MAIN__F1L21_1__521, align 4, !dbg !74
  %13 = bitcast i32* %.i0000p_343 to i64*, !dbg !74
  %14 = bitcast i32* %.dx0001p_396 to i64*, !dbg !74
  %15 = bitcast i32* %.de0001p_392 to i64*, !dbg !74
  %16 = bitcast i32* %.ds0001p_394 to i64*, !dbg !74
  %17 = call i32 @__kmpc_dispatch_next_4(i64* null, i32 %12, i64* %13, i64* %14, i64* %15, i64* %16), !dbg !74
  %18 = icmp eq i32 %17, 0, !dbg !74
  br i1 %18, label %L.LB2_385, label %L.LB2_595, !dbg !74

L.LB2_595:                                        ; preds = %L.LB2_384
  %19 = load i32, i32* %.dx0001p_396, align 4, !dbg !74
  store i32 %19, i32* %.dX0001p_395, align 4, !dbg !74
  %20 = load i32, i32* %.dX0001p_395, align 4, !dbg !74
  store i32 %20, i32* %i_342, align 4, !dbg !74
  %21 = load i32, i32* %.ds0001p_394, align 4, !dbg !74
  %22 = load i32, i32* %.de0001p_392, align 4, !dbg !74
  %23 = load i32, i32* %.dX0001p_395, align 4, !dbg !74
  %24 = sub nsw i32 %22, %23, !dbg !74
  %25 = add nsw i32 %21, %24, !dbg !74
  %26 = load i32, i32* %.ds0001p_394, align 4, !dbg !74
  %27 = sdiv i32 %25, %26, !dbg !74
  store i32 %27, i32* %.dY0001p_386, align 4, !dbg !74
  %28 = load i32, i32* %.dY0001p_386, align 4, !dbg !74
  %29 = icmp sle i32 %28, 0, !dbg !74
  br i1 %29, label %L.LB2_400, label %L.LB2_399, !dbg !74

L.LB2_399:                                        ; preds = %L.LB2_399, %L.LB2_595
  %30 = load i32, i32* %i_342, align 4, !dbg !76
  call void @llvm.dbg.value(metadata i32 %30, metadata !75, metadata !DIExpression()), !dbg !73
  %31 = load i32, i32* %i_342, align 4, !dbg !76
  call void @llvm.dbg.value(metadata i32 %31, metadata !75, metadata !DIExpression()), !dbg !73
  %32 = sext i32 %31 to i64, !dbg !76
  %33 = bitcast i64* %__nv_MAIN__F1L21_1Arg2 to i8*, !dbg !76
  %34 = getelementptr i8, i8* %33, i64 120, !dbg !76
  %35 = bitcast i8* %34 to i8**, !dbg !76
  %36 = load i8*, i8** %35, align 8, !dbg !76
  %37 = getelementptr i8, i8* %36, i64 56, !dbg !76
  %38 = bitcast i8* %37 to i64*, !dbg !76
  %39 = load i64, i64* %38, align 8, !dbg !76
  %40 = add nsw i64 %32, %39, !dbg !76
  %41 = bitcast i64* %__nv_MAIN__F1L21_1Arg2 to i8*, !dbg !76
  %42 = getelementptr i8, i8* %41, i64 8, !dbg !76
  %43 = bitcast i8* %42 to i8***, !dbg !76
  %44 = load i8**, i8*** %43, align 8, !dbg !76
  %45 = load i8*, i8** %44, align 8, !dbg !76
  %46 = getelementptr i8, i8* %45, i64 -4, !dbg !76
  %47 = bitcast i8* %46 to i32*, !dbg !76
  %48 = getelementptr i32, i32* %47, i64 %40, !dbg !76
  store i32 %30, i32* %48, align 4, !dbg !76
  %49 = load i32, i32* %.ds0001p_394, align 4, !dbg !77
  %50 = load i32, i32* %i_342, align 4, !dbg !77
  call void @llvm.dbg.value(metadata i32 %50, metadata !75, metadata !DIExpression()), !dbg !73
  %51 = add nsw i32 %49, %50, !dbg !77
  store i32 %51, i32* %i_342, align 4, !dbg !77
  %52 = load i32, i32* %.dY0001p_386, align 4, !dbg !77
  %53 = sub nsw i32 %52, 1, !dbg !77
  store i32 %53, i32* %.dY0001p_386, align 4, !dbg !77
  %54 = load i32, i32* %.dY0001p_386, align 4, !dbg !77
  %55 = icmp sgt i32 %54, 0, !dbg !77
  br i1 %55, label %L.LB2_399, label %L.LB2_400, !dbg !77

L.LB2_400:                                        ; preds = %L.LB2_399, %L.LB2_595
  br label %L.LB2_384, !dbg !77

L.LB2_385:                                        ; preds = %L.LB2_384
  br label %L.LB2_344

L.LB2_344:                                        ; preds = %L.LB2_385
  %56 = load i32, i32* %__gtid___nv_MAIN__F1L21_1__521, align 4, !dbg !78
  call void @__kmpc_barrier(i64* null, i32 %56), !dbg !78
  store i32 -1, i32* %.s0000_549, align 4, !dbg !79
  store i32 0, i32* %.s0001_550, align 4, !dbg !79
  %57 = load i32, i32* %__gtid___nv_MAIN__F1L21_1__521, align 4, !dbg !79
  %58 = call i32 @__kmpc_single(i64* null, i32 %57), !dbg !79
  %59 = icmp eq i32 %58, 0, !dbg !79
  br i1 %59, label %L.LB2_401, label %L.LB2_345, !dbg !79

L.LB2_345:                                        ; preds = %L.LB2_344
  store i32 1, i32* %.s0002_558, align 4, !dbg !80
  %60 = load i32, i32* %__gtid___nv_MAIN__F1L21_1__521, align 4, !dbg !81
  %61 = load i32, i32* %.s0002_558, align 4, !dbg !81
  %62 = bitcast void (i32, i64*)* @__nv_MAIN_F1L29_2_ to i64*, !dbg !81
  %63 = call i8* @__kmpc_omp_task_alloc(i64* null, i32 %60, i32 %61, i32 40, i32 136, i64* %62), !dbg !81
  store i8* %63, i8** %.z0428_557, align 8, !dbg !81
  %64 = bitcast i64* %__nv_MAIN__F1L21_1Arg2 to i8*, !dbg !81
  %65 = load i8*, i8** %.z0428_557, align 8, !dbg !81
  %66 = bitcast i8* %65 to i8**, !dbg !81
  %67 = load i8*, i8** %66, align 8, !dbg !81
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %67, i8* align 8 %64, i64 136, i1 false), !dbg !81
  %68 = load i32, i32* %__gtid___nv_MAIN__F1L21_1__521, align 4, !dbg !81
  %69 = load i8*, i8** %.z0428_557, align 8, !dbg !81
  %70 = bitcast i8* %69 to i64*, !dbg !81
  call void @__kmpc_omp_task(i64* null, i32 %68, i64* %70), !dbg !81
  br label %L.LB2_402

L.LB2_402:                                        ; preds = %L.LB2_345
  %71 = load i32, i32* %__gtid___nv_MAIN__F1L21_1__521, align 4, !dbg !82
  %72 = call i32 @__kmpc_omp_taskwait(i64* null, i32 %71), !dbg !82
  %73 = bitcast i64* %__nv_MAIN__F1L21_1Arg2 to i8*, !dbg !83
  %74 = getelementptr i8, i8* %73, i64 128, !dbg !83
  %75 = bitcast i8* %74 to i8**, !dbg !83
  %76 = load i8*, i8** %75, align 8, !dbg !83
  %77 = getelementptr i8, i8* %76, i64 56, !dbg !83
  %78 = bitcast i8* %77 to i64*, !dbg !83
  %79 = load i64, i64* %78, align 8, !dbg !83
  %80 = bitcast i64* %__nv_MAIN__F1L21_1Arg2 to i8*, !dbg !83
  %81 = getelementptr i8, i8* %80, i64 64, !dbg !83
  %82 = bitcast i8* %81 to i32***, !dbg !83
  %83 = load i32**, i32*** %82, align 8, !dbg !83
  %84 = load i32*, i32** %83, align 8, !dbg !83
  %85 = getelementptr i32, i32* %84, i64 %79, !dbg !83
  %86 = load i32, i32* %85, align 4, !dbg !83
  %87 = bitcast i64* %__nv_MAIN__F1L21_1Arg2 to i8*, !dbg !83
  %88 = getelementptr i8, i8* %87, i64 128, !dbg !83
  %89 = bitcast i8* %88 to i8**, !dbg !83
  %90 = load i8*, i8** %89, align 8, !dbg !83
  %91 = getelementptr i8, i8* %90, i64 56, !dbg !83
  %92 = bitcast i8* %91 to i64*, !dbg !83
  %93 = load i64, i64* %92, align 8, !dbg !83
  %94 = bitcast i64* %__nv_MAIN__F1L21_1Arg2 to i8*, !dbg !83
  %95 = getelementptr i8, i8* %94, i64 64, !dbg !83
  %96 = bitcast i8* %95 to i8***, !dbg !83
  %97 = load i8**, i8*** %96, align 8, !dbg !83
  %98 = load i8*, i8** %97, align 8, !dbg !83
  %99 = getelementptr i8, i8* %98, i64 4, !dbg !83
  %100 = bitcast i8* %99 to i32*, !dbg !83
  %101 = getelementptr i32, i32* %100, i64 %93, !dbg !83
  %102 = load i32, i32* %101, align 4, !dbg !83
  %103 = add nsw i32 %86, %102, !dbg !83
  %104 = bitcast i64* %__nv_MAIN__F1L21_1Arg2 to i8*, !dbg !83
  %105 = getelementptr i8, i8* %104, i64 112, !dbg !83
  %106 = bitcast i8* %105 to i32**, !dbg !83
  %107 = load i32*, i32** %106, align 8, !dbg !83
  store i32 %103, i32* %107, align 4, !dbg !83
  %108 = load i32, i32* %__gtid___nv_MAIN__F1L21_1__521, align 4, !dbg !84
  store i32 %108, i32* %.s0000_549, align 4, !dbg !84
  store i32 1, i32* %.s0001_550, align 4, !dbg !84
  %109 = load i32, i32* %__gtid___nv_MAIN__F1L21_1__521, align 4, !dbg !84
  call void @__kmpc_end_single(i64* null, i32 %109), !dbg !84
  br label %L.LB2_401

L.LB2_401:                                        ; preds = %L.LB2_402, %L.LB2_344
  br label %L.LB2_357

L.LB2_357:                                        ; preds = %L.LB2_401
  %110 = load i32, i32* %__gtid___nv_MAIN__F1L21_1__521, align 4, !dbg !84
  call void @__kmpc_barrier(i64* null, i32 %110), !dbg !84
  br label %L.LB2_358

L.LB2_358:                                        ; preds = %L.LB2_357
  ret void, !dbg !73
}

define internal void @__nv_MAIN_F1L29_2_(i32 %__nv_MAIN_F1L29_2Arg0.arg, i64* %__nv_MAIN_F1L29_2Arg1) #0 !dbg !85 {
L.entry:
  %__nv_MAIN_F1L29_2Arg0.addr = alloca i32, align 4
  %.S0000_597 = alloca i8*, align 8
  %__gtid___nv_MAIN_F1L29_2__608 = alloca i32, align 4
  %.s0003_603 = alloca i32, align 4
  %.z0560_602 = alloca i8*, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L29_2Arg0.addr, metadata !88, metadata !DIExpression()), !dbg !89
  store i32 %__nv_MAIN_F1L29_2Arg0.arg, i32* %__nv_MAIN_F1L29_2Arg0.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L29_2Arg0.addr, metadata !90, metadata !DIExpression()), !dbg !89
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L29_2Arg1, metadata !91, metadata !DIExpression()), !dbg !89
  call void @llvm.dbg.value(metadata i32 4, metadata !92, metadata !DIExpression()), !dbg !89
  call void @llvm.dbg.value(metadata i32 4, metadata !93, metadata !DIExpression()), !dbg !89
  call void @llvm.dbg.value(metadata i32 4, metadata !94, metadata !DIExpression()), !dbg !89
  call void @llvm.dbg.value(metadata i32 4, metadata !95, metadata !DIExpression()), !dbg !89
  call void @llvm.dbg.value(metadata i32 4, metadata !96, metadata !DIExpression()), !dbg !89
  call void @llvm.dbg.value(metadata i32 4, metadata !97, metadata !DIExpression()), !dbg !89
  call void @llvm.dbg.value(metadata i32 4, metadata !98, metadata !DIExpression()), !dbg !89
  call void @llvm.dbg.value(metadata i32 1, metadata !99, metadata !DIExpression()), !dbg !89
  call void @llvm.dbg.value(metadata i32 2, metadata !100, metadata !DIExpression()), !dbg !89
  call void @llvm.dbg.value(metadata i32 4, metadata !101, metadata !DIExpression()), !dbg !89
  call void @llvm.dbg.value(metadata i32 0, metadata !102, metadata !DIExpression()), !dbg !89
  call void @llvm.dbg.value(metadata i32 1, metadata !103, metadata !DIExpression()), !dbg !89
  call void @llvm.dbg.value(metadata i32 2, metadata !104, metadata !DIExpression()), !dbg !89
  call void @llvm.dbg.value(metadata i32 4, metadata !105, metadata !DIExpression()), !dbg !89
  call void @llvm.dbg.value(metadata i32 0, metadata !106, metadata !DIExpression()), !dbg !89
  call void @llvm.dbg.value(metadata i32 1, metadata !107, metadata !DIExpression()), !dbg !89
  call void @llvm.dbg.value(metadata i32 2, metadata !108, metadata !DIExpression()), !dbg !89
  call void @llvm.dbg.value(metadata i32 4, metadata !109, metadata !DIExpression()), !dbg !89
  %0 = bitcast i64* %__nv_MAIN_F1L29_2Arg1 to i8**, !dbg !110
  %1 = load i8*, i8** %0, align 8, !dbg !110
  store i8* %1, i8** %.S0000_597, align 8, !dbg !110
  %2 = load i32, i32* %__nv_MAIN_F1L29_2Arg0.addr, align 4, !dbg !111
  call void @llvm.dbg.value(metadata i32 %2, metadata !88, metadata !DIExpression()), !dbg !89
  store i32 %2, i32* %__gtid___nv_MAIN_F1L29_2__608, align 4, !dbg !111
  br label %L.LB4_601

L.LB4_601:                                        ; preds = %L.entry
  br label %L.LB4_348

L.LB4_348:                                        ; preds = %L.LB4_601
  store i32 1, i32* %.s0003_603, align 4, !dbg !112
  %3 = load i32, i32* %__gtid___nv_MAIN_F1L29_2__608, align 4, !dbg !113
  %4 = load i32, i32* %.s0003_603, align 4, !dbg !113
  %5 = bitcast void (i32, i64*)* @__nv_MAIN_F1L30_3_ to i64*, !dbg !113
  %6 = call i8* @__kmpc_omp_task_alloc(i64* null, i32 %3, i32 %4, i32 40, i32 136, i64* %5), !dbg !113
  store i8* %6, i8** %.z0560_602, align 8, !dbg !113
  %7 = load i8*, i8** %.S0000_597, align 8, !dbg !113
  %8 = load i8*, i8** %.z0560_602, align 8, !dbg !113
  %9 = bitcast i8* %8 to i8**, !dbg !113
  %10 = load i8*, i8** %9, align 8, !dbg !113
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %10, i8* align 8 %7, i64 136, i1 false), !dbg !113
  %11 = load i32, i32* %__gtid___nv_MAIN_F1L29_2__608, align 4, !dbg !113
  %12 = load i8*, i8** %.z0560_602, align 8, !dbg !113
  %13 = bitcast i8* %12 to i64*, !dbg !113
  call void @__kmpc_omp_task(i64* null, i32 %11, i64* %13), !dbg !113
  br label %L.LB4_403

L.LB4_403:                                        ; preds = %L.LB4_348
  %14 = load i8*, i8** %.S0000_597, align 8, !dbg !114
  %15 = getelementptr i8, i8* %14, i64 120, !dbg !114
  %16 = bitcast i8* %15 to i8**, !dbg !114
  %17 = load i8*, i8** %16, align 8, !dbg !114
  %18 = getelementptr i8, i8* %17, i64 56, !dbg !114
  %19 = bitcast i8* %18 to i64*, !dbg !114
  %20 = load i64, i64* %19, align 8, !dbg !114
  %21 = load i8*, i8** %.S0000_597, align 8, !dbg !114
  %22 = getelementptr i8, i8* %21, i64 8, !dbg !114
  %23 = bitcast i8* %22 to i8***, !dbg !114
  %24 = load i8**, i8*** %23, align 8, !dbg !114
  %25 = load i8*, i8** %24, align 8, !dbg !114
  %26 = getelementptr i8, i8* %25, i64 4, !dbg !114
  %27 = bitcast i8* %26 to i32*, !dbg !114
  %28 = getelementptr i32, i32* %27, i64 %20, !dbg !114
  %29 = load i32, i32* %28, align 4, !dbg !114
  %30 = load i8*, i8** %.S0000_597, align 8, !dbg !114
  %31 = getelementptr i8, i8* %30, i64 120, !dbg !114
  %32 = bitcast i8* %31 to i8**, !dbg !114
  %33 = load i8*, i8** %32, align 8, !dbg !114
  %34 = getelementptr i8, i8* %33, i64 56, !dbg !114
  %35 = bitcast i8* %34 to i64*, !dbg !114
  %36 = load i64, i64* %35, align 8, !dbg !114
  %37 = load i8*, i8** %.S0000_597, align 8, !dbg !114
  %38 = getelementptr i8, i8* %37, i64 8, !dbg !114
  %39 = bitcast i8* %38 to i32***, !dbg !114
  %40 = load i32**, i32*** %39, align 8, !dbg !114
  %41 = load i32*, i32** %40, align 8, !dbg !114
  %42 = getelementptr i32, i32* %41, i64 %36, !dbg !114
  %43 = load i32, i32* %42, align 4, !dbg !114
  %44 = add nsw i32 %29, %43, !dbg !114
  %45 = load i8*, i8** %.S0000_597, align 8, !dbg !114
  %46 = getelementptr i8, i8* %45, i64 128, !dbg !114
  %47 = bitcast i8* %46 to i8**, !dbg !114
  %48 = load i8*, i8** %47, align 8, !dbg !114
  %49 = getelementptr i8, i8* %48, i64 56, !dbg !114
  %50 = bitcast i8* %49 to i64*, !dbg !114
  %51 = load i64, i64* %50, align 8, !dbg !114
  %52 = load i8*, i8** %.S0000_597, align 8, !dbg !114
  %53 = getelementptr i8, i8* %52, i64 64, !dbg !114
  %54 = bitcast i8* %53 to i32***, !dbg !114
  %55 = load i32**, i32*** %54, align 8, !dbg !114
  %56 = load i32*, i32** %55, align 8, !dbg !114
  %57 = getelementptr i32, i32* %56, i64 %51, !dbg !114
  store i32 %44, i32* %57, align 4, !dbg !114
  br label %L.LB4_355

L.LB4_355:                                        ; preds = %L.LB4_403
  ret void, !dbg !111
}

define internal void @__nv_MAIN_F1L30_3_(i32 %__nv_MAIN_F1L30_3Arg0.arg, i64* %__nv_MAIN_F1L30_3Arg1) #0 !dbg !115 {
L.entry:
  %__nv_MAIN_F1L30_3Arg0.addr = alloca i32, align 4
  %.S0000_597 = alloca i8*, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L30_3Arg0.addr, metadata !116, metadata !DIExpression()), !dbg !117
  store i32 %__nv_MAIN_F1L30_3Arg0.arg, i32* %__nv_MAIN_F1L30_3Arg0.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L30_3Arg0.addr, metadata !118, metadata !DIExpression()), !dbg !117
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L30_3Arg1, metadata !119, metadata !DIExpression()), !dbg !117
  call void @llvm.dbg.value(metadata i32 4, metadata !120, metadata !DIExpression()), !dbg !117
  call void @llvm.dbg.value(metadata i32 4, metadata !121, metadata !DIExpression()), !dbg !117
  call void @llvm.dbg.value(metadata i32 4, metadata !122, metadata !DIExpression()), !dbg !117
  call void @llvm.dbg.value(metadata i32 4, metadata !123, metadata !DIExpression()), !dbg !117
  call void @llvm.dbg.value(metadata i32 4, metadata !124, metadata !DIExpression()), !dbg !117
  call void @llvm.dbg.value(metadata i32 4, metadata !125, metadata !DIExpression()), !dbg !117
  call void @llvm.dbg.value(metadata i32 4, metadata !126, metadata !DIExpression()), !dbg !117
  call void @llvm.dbg.value(metadata i32 1, metadata !127, metadata !DIExpression()), !dbg !117
  call void @llvm.dbg.value(metadata i32 2, metadata !128, metadata !DIExpression()), !dbg !117
  call void @llvm.dbg.value(metadata i32 4, metadata !129, metadata !DIExpression()), !dbg !117
  call void @llvm.dbg.value(metadata i32 0, metadata !130, metadata !DIExpression()), !dbg !117
  call void @llvm.dbg.value(metadata i32 1, metadata !131, metadata !DIExpression()), !dbg !117
  call void @llvm.dbg.value(metadata i32 2, metadata !132, metadata !DIExpression()), !dbg !117
  call void @llvm.dbg.value(metadata i32 4, metadata !133, metadata !DIExpression()), !dbg !117
  call void @llvm.dbg.value(metadata i32 0, metadata !134, metadata !DIExpression()), !dbg !117
  call void @llvm.dbg.value(metadata i32 1, metadata !135, metadata !DIExpression()), !dbg !117
  call void @llvm.dbg.value(metadata i32 2, metadata !136, metadata !DIExpression()), !dbg !117
  call void @llvm.dbg.value(metadata i32 4, metadata !137, metadata !DIExpression()), !dbg !117
  %0 = bitcast i64* %__nv_MAIN_F1L30_3Arg1 to i8**, !dbg !138
  %1 = load i8*, i8** %0, align 8, !dbg !138
  store i8* %1, i8** %.S0000_597, align 8, !dbg !138
  br label %L.LB6_620

L.LB6_620:                                        ; preds = %L.entry
  br label %L.LB6_351

L.LB6_351:                                        ; preds = %L.LB6_620
  %2 = load i8*, i8** %.S0000_597, align 8, !dbg !139
  %3 = getelementptr i8, i8* %2, i64 120, !dbg !139
  %4 = bitcast i8* %3 to i8**, !dbg !139
  %5 = load i8*, i8** %4, align 8, !dbg !139
  %6 = getelementptr i8, i8* %5, i64 56, !dbg !139
  %7 = bitcast i8* %6 to i64*, !dbg !139
  %8 = load i64, i64* %7, align 8, !dbg !139
  %9 = load i8*, i8** %.S0000_597, align 8, !dbg !139
  %10 = getelementptr i8, i8* %9, i64 8, !dbg !139
  %11 = bitcast i8* %10 to i8***, !dbg !139
  %12 = load i8**, i8*** %11, align 8, !dbg !139
  %13 = load i8*, i8** %12, align 8, !dbg !139
  %14 = getelementptr i8, i8* %13, i64 12, !dbg !139
  %15 = bitcast i8* %14 to i32*, !dbg !139
  %16 = getelementptr i32, i32* %15, i64 %8, !dbg !139
  %17 = load i32, i32* %16, align 4, !dbg !139
  %18 = load i8*, i8** %.S0000_597, align 8, !dbg !139
  %19 = getelementptr i8, i8* %18, i64 120, !dbg !139
  %20 = bitcast i8* %19 to i8**, !dbg !139
  %21 = load i8*, i8** %20, align 8, !dbg !139
  %22 = getelementptr i8, i8* %21, i64 56, !dbg !139
  %23 = bitcast i8* %22 to i64*, !dbg !139
  %24 = load i64, i64* %23, align 8, !dbg !139
  %25 = load i8*, i8** %.S0000_597, align 8, !dbg !139
  %26 = getelementptr i8, i8* %25, i64 8, !dbg !139
  %27 = bitcast i8* %26 to i8***, !dbg !139
  %28 = load i8**, i8*** %27, align 8, !dbg !139
  %29 = load i8*, i8** %28, align 8, !dbg !139
  %30 = getelementptr i8, i8* %29, i64 8, !dbg !139
  %31 = bitcast i8* %30 to i32*, !dbg !139
  %32 = getelementptr i32, i32* %31, i64 %24, !dbg !139
  %33 = load i32, i32* %32, align 4, !dbg !139
  %34 = add nsw i32 %17, %33, !dbg !139
  %35 = load i8*, i8** %.S0000_597, align 8, !dbg !139
  %36 = getelementptr i8, i8* %35, i64 128, !dbg !139
  %37 = bitcast i8* %36 to i8**, !dbg !139
  %38 = load i8*, i8** %37, align 8, !dbg !139
  %39 = getelementptr i8, i8* %38, i64 56, !dbg !139
  %40 = bitcast i8* %39 to i64*, !dbg !139
  %41 = load i64, i64* %40, align 8, !dbg !139
  %42 = load i8*, i8** %.S0000_597, align 8, !dbg !139
  %43 = getelementptr i8, i8* %42, i64 64, !dbg !139
  %44 = bitcast i8* %43 to i8***, !dbg !139
  %45 = load i8**, i8*** %44, align 8, !dbg !139
  %46 = load i8*, i8** %45, align 8, !dbg !139
  %47 = getelementptr i8, i8* %46, i64 4, !dbg !139
  %48 = bitcast i8* %47 to i32*, !dbg !139
  %49 = getelementptr i32, i32* %48, i64 %41, !dbg !139
  store i32 %34, i32* %49, align 4, !dbg !139
  br label %L.LB6_354

L.LB6_354:                                        ; preds = %L.LB6_351
  ret void, !dbg !140
}

declare void @__kmpc_end_single(i64*, i32) #0

declare signext i32 @__kmpc_omp_taskwait(i64*, i32) #0

declare void @__kmpc_omp_task(i64*, i32, i64*) #0

declare i8* @__kmpc_omp_task_alloc(i64*, i32, i32, i32, i32, i64*) #0

declare signext i32 @__kmpc_single(i64*, i32) #0

declare void @__kmpc_barrier(i64*, i32) #0

declare signext i32 @__kmpc_dispatch_next_4(i64*, i32, i64*, i64*, i64*, i64*) #0

declare void @__kmpc_dispatch_init_4(i64*, i32, i32, i32, i32, i32, i32) #0

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

declare void @__kmpc_push_num_threads(i64*, i32, i32) #0

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
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB117-taskwait-waitonlychild-orig-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb117_taskwait_waitonlychild_orig_yes", scope: !2, file: !3, line: 11, type: !6, scopeLine: 11, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!6 = !DISubroutineType(cc: DW_CC_program, types: !7)
!7 = !{null}
!8 = !DILocalVariable(name: "omp_integer_kind", scope: !5, file: !3, type: !9)
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 0, scope: !5)
!11 = !DILocalVariable(name: "omp_logical_kind", scope: !5, file: !3, type: !9)
!12 = !DILocalVariable(name: "omp_lock_kind", scope: !5, file: !3, type: !9)
!13 = !DILocalVariable(name: "omp_sched_kind", scope: !5, file: !3, type: !9)
!14 = !DILocalVariable(name: "omp_real_kind", scope: !5, file: !3, type: !9)
!15 = !DILocalVariable(name: "omp_proc_bind_kind", scope: !5, file: !3, type: !9)
!16 = !DILocalVariable(name: "omp_lock_hint_kind", scope: !5, file: !3, type: !9)
!17 = !DILocalVariable(name: "omp_sched_static", scope: !5, file: !3, type: !9)
!18 = !DILocalVariable(name: "omp_sched_dynamic", scope: !5, file: !3, type: !9)
!19 = !DILocalVariable(name: "omp_sched_auto", scope: !5, file: !3, type: !9)
!20 = !DILocalVariable(name: "omp_proc_bind_false", scope: !5, file: !3, type: !9)
!21 = !DILocalVariable(name: "omp_proc_bind_true", scope: !5, file: !3, type: !9)
!22 = !DILocalVariable(name: "omp_proc_bind_master", scope: !5, file: !3, type: !9)
!23 = !DILocalVariable(name: "omp_proc_bind_spread", scope: !5, file: !3, type: !9)
!24 = !DILocalVariable(name: "omp_lock_hint_none", scope: !5, file: !3, type: !9)
!25 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !5, file: !3, type: !9)
!26 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !5, file: !3, type: !9)
!27 = !DILocalVariable(name: "omp_lock_hint_nonspeculative", scope: !5, file: !3, type: !9)
!28 = !DILocation(line: 43, column: 1, scope: !5)
!29 = !DILocation(line: 11, column: 1, scope: !5)
!30 = !DILocalVariable(name: "psum", scope: !5, file: !3, type: !31)
!31 = !DICompositeType(tag: DW_TAG_array_type, baseType: !9, size: 32, align: 32, elements: !32)
!32 = !{!33}
!33 = !DISubrange(count: 0, lowerBound: 1)
!34 = !DILocalVariable(scope: !5, file: !3, type: !35, flags: DIFlagArtificial)
!35 = !DICompositeType(tag: DW_TAG_array_type, baseType: !36, size: 1024, align: 64, elements: !37)
!36 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!37 = !{!38}
!38 = !DISubrange(count: 16, lowerBound: 1)
!39 = !DILocalVariable(name: "a", scope: !5, file: !3, type: !31)
!40 = !DILocalVariable(scope: !5, file: !3, type: !36, flags: DIFlagArtificial)
!41 = !DILocation(line: 18, column: 1, scope: !5)
!42 = !DILocation(line: 19, column: 1, scope: !5)
!43 = !DILocation(line: 21, column: 1, scope: !5)
!44 = !DILocalVariable(name: "sum", scope: !5, file: !3, type: !9)
!45 = !DILocation(line: 40, column: 1, scope: !5)
!46 = !DILocalVariable(scope: !5, file: !3, type: !9, flags: DIFlagArtificial)
!47 = !DILocation(line: 42, column: 1, scope: !5)
!48 = distinct !DISubprogram(name: "__nv_MAIN__F1L21_1", scope: !2, file: !3, line: 21, type: !49, scopeLine: 21, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!49 = !DISubroutineType(types: !50)
!50 = !{null, !9, !36, !36}
!51 = !DILocalVariable(name: "__nv_MAIN__F1L21_1Arg0", arg: 1, scope: !48, file: !3, type: !9)
!52 = !DILocation(line: 0, scope: !48)
!53 = !DILocalVariable(name: "__nv_MAIN__F1L21_1Arg1", arg: 2, scope: !48, file: !3, type: !36)
!54 = !DILocalVariable(name: "__nv_MAIN__F1L21_1Arg2", arg: 3, scope: !48, file: !3, type: !36)
!55 = !DILocalVariable(name: "omp_integer_kind", scope: !48, file: !3, type: !9)
!56 = !DILocalVariable(name: "omp_logical_kind", scope: !48, file: !3, type: !9)
!57 = !DILocalVariable(name: "omp_lock_kind", scope: !48, file: !3, type: !9)
!58 = !DILocalVariable(name: "omp_sched_kind", scope: !48, file: !3, type: !9)
!59 = !DILocalVariable(name: "omp_real_kind", scope: !48, file: !3, type: !9)
!60 = !DILocalVariable(name: "omp_proc_bind_kind", scope: !48, file: !3, type: !9)
!61 = !DILocalVariable(name: "omp_lock_hint_kind", scope: !48, file: !3, type: !9)
!62 = !DILocalVariable(name: "omp_sched_static", scope: !48, file: !3, type: !9)
!63 = !DILocalVariable(name: "omp_sched_dynamic", scope: !48, file: !3, type: !9)
!64 = !DILocalVariable(name: "omp_sched_auto", scope: !48, file: !3, type: !9)
!65 = !DILocalVariable(name: "omp_proc_bind_false", scope: !48, file: !3, type: !9)
!66 = !DILocalVariable(name: "omp_proc_bind_true", scope: !48, file: !3, type: !9)
!67 = !DILocalVariable(name: "omp_proc_bind_master", scope: !48, file: !3, type: !9)
!68 = !DILocalVariable(name: "omp_proc_bind_spread", scope: !48, file: !3, type: !9)
!69 = !DILocalVariable(name: "omp_lock_hint_none", scope: !48, file: !3, type: !9)
!70 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !48, file: !3, type: !9)
!71 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !48, file: !3, type: !9)
!72 = !DILocalVariable(name: "omp_lock_hint_nonspeculative", scope: !48, file: !3, type: !9)
!73 = !DILocation(line: 38, column: 1, scope: !48)
!74 = !DILocation(line: 23, column: 1, scope: !48)
!75 = !DILocalVariable(name: "i", scope: !48, file: !3, type: !9)
!76 = !DILocation(line: 24, column: 1, scope: !48)
!77 = !DILocation(line: 25, column: 1, scope: !48)
!78 = !DILocation(line: 26, column: 1, scope: !48)
!79 = !DILocation(line: 28, column: 1, scope: !48)
!80 = !DILocation(line: 29, column: 1, scope: !48)
!81 = !DILocation(line: 34, column: 1, scope: !48)
!82 = !DILocation(line: 35, column: 1, scope: !48)
!83 = !DILocation(line: 36, column: 1, scope: !48)
!84 = !DILocation(line: 37, column: 1, scope: !48)
!85 = distinct !DISubprogram(name: "__nv_MAIN_F1L29_2", scope: !2, file: !3, line: 29, type: !86, scopeLine: 29, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!86 = !DISubroutineType(types: !87)
!87 = !{null, !9, !36}
!88 = !DILocalVariable(name: "__nv_MAIN_F1L29_2Arg0", scope: !85, file: !3, type: !9)
!89 = !DILocation(line: 0, scope: !85)
!90 = !DILocalVariable(name: "__nv_MAIN_F1L29_2Arg0", arg: 1, scope: !85, file: !3, type: !9)
!91 = !DILocalVariable(name: "__nv_MAIN_F1L29_2Arg1", arg: 2, scope: !85, file: !3, type: !36)
!92 = !DILocalVariable(name: "omp_integer_kind", scope: !85, file: !3, type: !9)
!93 = !DILocalVariable(name: "omp_logical_kind", scope: !85, file: !3, type: !9)
!94 = !DILocalVariable(name: "omp_lock_kind", scope: !85, file: !3, type: !9)
!95 = !DILocalVariable(name: "omp_sched_kind", scope: !85, file: !3, type: !9)
!96 = !DILocalVariable(name: "omp_real_kind", scope: !85, file: !3, type: !9)
!97 = !DILocalVariable(name: "omp_proc_bind_kind", scope: !85, file: !3, type: !9)
!98 = !DILocalVariable(name: "omp_lock_hint_kind", scope: !85, file: !3, type: !9)
!99 = !DILocalVariable(name: "omp_sched_static", scope: !85, file: !3, type: !9)
!100 = !DILocalVariable(name: "omp_sched_dynamic", scope: !85, file: !3, type: !9)
!101 = !DILocalVariable(name: "omp_sched_auto", scope: !85, file: !3, type: !9)
!102 = !DILocalVariable(name: "omp_proc_bind_false", scope: !85, file: !3, type: !9)
!103 = !DILocalVariable(name: "omp_proc_bind_true", scope: !85, file: !3, type: !9)
!104 = !DILocalVariable(name: "omp_proc_bind_master", scope: !85, file: !3, type: !9)
!105 = !DILocalVariable(name: "omp_proc_bind_spread", scope: !85, file: !3, type: !9)
!106 = !DILocalVariable(name: "omp_lock_hint_none", scope: !85, file: !3, type: !9)
!107 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !85, file: !3, type: !9)
!108 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !85, file: !3, type: !9)
!109 = !DILocalVariable(name: "omp_lock_hint_nonspeculative", scope: !85, file: !3, type: !9)
!110 = !DILocation(line: 29, column: 1, scope: !85)
!111 = !DILocation(line: 34, column: 1, scope: !85)
!112 = !DILocation(line: 30, column: 1, scope: !85)
!113 = !DILocation(line: 32, column: 1, scope: !85)
!114 = !DILocation(line: 33, column: 1, scope: !85)
!115 = distinct !DISubprogram(name: "__nv_MAIN_F1L30_3", scope: !2, file: !3, line: 30, type: !86, scopeLine: 30, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!116 = !DILocalVariable(name: "__nv_MAIN_F1L30_3Arg0", scope: !115, file: !3, type: !9)
!117 = !DILocation(line: 0, scope: !115)
!118 = !DILocalVariable(name: "__nv_MAIN_F1L30_3Arg0", arg: 1, scope: !115, file: !3, type: !9)
!119 = !DILocalVariable(name: "__nv_MAIN_F1L30_3Arg1", arg: 2, scope: !115, file: !3, type: !36)
!120 = !DILocalVariable(name: "omp_integer_kind", scope: !115, file: !3, type: !9)
!121 = !DILocalVariable(name: "omp_logical_kind", scope: !115, file: !3, type: !9)
!122 = !DILocalVariable(name: "omp_lock_kind", scope: !115, file: !3, type: !9)
!123 = !DILocalVariable(name: "omp_sched_kind", scope: !115, file: !3, type: !9)
!124 = !DILocalVariable(name: "omp_real_kind", scope: !115, file: !3, type: !9)
!125 = !DILocalVariable(name: "omp_proc_bind_kind", scope: !115, file: !3, type: !9)
!126 = !DILocalVariable(name: "omp_lock_hint_kind", scope: !115, file: !3, type: !9)
!127 = !DILocalVariable(name: "omp_sched_static", scope: !115, file: !3, type: !9)
!128 = !DILocalVariable(name: "omp_sched_dynamic", scope: !115, file: !3, type: !9)
!129 = !DILocalVariable(name: "omp_sched_auto", scope: !115, file: !3, type: !9)
!130 = !DILocalVariable(name: "omp_proc_bind_false", scope: !115, file: !3, type: !9)
!131 = !DILocalVariable(name: "omp_proc_bind_true", scope: !115, file: !3, type: !9)
!132 = !DILocalVariable(name: "omp_proc_bind_master", scope: !115, file: !3, type: !9)
!133 = !DILocalVariable(name: "omp_proc_bind_spread", scope: !115, file: !3, type: !9)
!134 = !DILocalVariable(name: "omp_lock_hint_none", scope: !115, file: !3, type: !9)
!135 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !115, file: !3, type: !9)
!136 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !115, file: !3, type: !9)
!137 = !DILocalVariable(name: "omp_lock_hint_nonspeculative", scope: !115, file: !3, type: !9)
!138 = !DILocation(line: 30, column: 1, scope: !115)
!139 = !DILocation(line: 31, column: 1, scope: !115)
!140 = !DILocation(line: 32, column: 1, scope: !115)

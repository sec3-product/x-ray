; ModuleID = '/tmp/DRB114-if-orig-yes-4ad078.ll'
source_filename = "/tmp/DRB114-if-orig-yes-4ad078.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%astruct.dt68 = type <{ i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8* }>

@.C340_MAIN_ = internal constant i64 50
@.C309_MAIN_ = internal constant i32 14
@.C339_MAIN_ = internal constant [7 x i8] c"a(50) ="
@.C336_MAIN_ = internal constant i32 6
@.C333_MAIN_ = internal constant [47 x i8] c"micro-benchmarks-fortran/DRB114-if-orig-yes.f95"
@.C335_MAIN_ = internal constant i32 36
@.C292_MAIN_ = internal constant double 1.000000e+00
@.C300_MAIN_ = internal constant i32 2
@.C325_MAIN_ = internal constant float 1.000000e+02
@.C348_MAIN_ = internal constant i64 27
@.C285_MAIN_ = internal constant i32 1
@.C310_MAIN_ = internal constant i32 28
@.C352_MAIN_ = internal constant i64 8
@.C351_MAIN_ = internal constant i64 28
@.C322_MAIN_ = internal constant i32 100
@.C286_MAIN_ = internal constant i64 1
@.C284_MAIN_ = internal constant i64 0
@.C283_MAIN_ = internal constant i32 0
@.C292___nv_MAIN__F1L30_1 = internal constant double 1.000000e+00
@.C285___nv_MAIN__F1L30_1 = internal constant i32 1
@.C283___nv_MAIN__F1L30_1 = internal constant i32 0

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__427 = alloca i32, align 4
  %.Z0969_324 = alloca double*, align 8
  %"a$sd1_350" = alloca [16 x i64], align 8
  %len_323 = alloca i32, align 4
  %z_b_0_316 = alloca i64, align 8
  %z_b_1_317 = alloca i64, align 8
  %z_e_62_320 = alloca i64, align 8
  %z_b_2_318 = alloca i64, align 8
  %z_b_3_319 = alloca i64, align 8
  %.dY0001_361 = alloca i32, align 4
  %i_311 = alloca i32, align 4
  %u_314 = alloca float, align 4
  %j_313 = alloca i32, align 4
  %.uplevelArgPack0001_406 = alloca %astruct.dt68, align 16
  %z__io_338 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 2, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 2, metadata !14, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !15, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !16, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 2, metadata !17, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 8, metadata !18, metadata !DIExpression()), !dbg !10
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !19
  store i32 %0, i32* %__gtid_MAIN__427, align 4, !dbg !19
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !20
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !20
  call void (i8*, ...) %2(i8* %1), !dbg !20
  call void @llvm.dbg.declare(metadata double** %.Z0969_324, metadata !21, metadata !DIExpression(DW_OP_deref)), !dbg !10
  %3 = bitcast double** %.Z0969_324 to i8**, !dbg !20
  store i8* null, i8** %3, align 8, !dbg !20
  call void @llvm.dbg.declare(metadata [16 x i64]* %"a$sd1_350", metadata !26, metadata !DIExpression()), !dbg !10
  %4 = bitcast [16 x i64]* %"a$sd1_350" to i64*, !dbg !20
  store i64 0, i64* %4, align 8, !dbg !20
  br label %L.LB1_380

L.LB1_380:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %len_323, metadata !31, metadata !DIExpression()), !dbg !10
  store i32 100, i32* %len_323, align 4, !dbg !32
  call void @llvm.dbg.declare(metadata i64* %z_b_0_316, metadata !33, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_0_316, align 8, !dbg !34
  %5 = load i32, i32* %len_323, align 4, !dbg !34
  call void @llvm.dbg.value(metadata i32 %5, metadata !31, metadata !DIExpression()), !dbg !10
  %6 = sext i32 %5 to i64, !dbg !34
  call void @llvm.dbg.declare(metadata i64* %z_b_1_317, metadata !33, metadata !DIExpression()), !dbg !10
  store i64 %6, i64* %z_b_1_317, align 8, !dbg !34
  %7 = load i64, i64* %z_b_1_317, align 8, !dbg !34
  call void @llvm.dbg.value(metadata i64 %7, metadata !33, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_62_320, metadata !33, metadata !DIExpression()), !dbg !10
  store i64 %7, i64* %z_e_62_320, align 8, !dbg !34
  %8 = bitcast [16 x i64]* %"a$sd1_350" to i8*, !dbg !34
  %9 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !34
  %10 = bitcast i64* @.C351_MAIN_ to i8*, !dbg !34
  %11 = bitcast i64* @.C352_MAIN_ to i8*, !dbg !34
  %12 = bitcast i64* %z_b_0_316 to i8*, !dbg !34
  %13 = bitcast i64* %z_b_1_317 to i8*, !dbg !34
  %14 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !34
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %14(i8* %8, i8* %9, i8* %10, i8* %11, i8* %12, i8* %13), !dbg !34
  %15 = bitcast [16 x i64]* %"a$sd1_350" to i8*, !dbg !34
  %16 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !34
  call void (i8*, i32, ...) %16(i8* %15, i32 28), !dbg !34
  %17 = load i64, i64* %z_b_1_317, align 8, !dbg !34
  call void @llvm.dbg.value(metadata i64 %17, metadata !33, metadata !DIExpression()), !dbg !10
  %18 = load i64, i64* %z_b_0_316, align 8, !dbg !34
  call void @llvm.dbg.value(metadata i64 %18, metadata !33, metadata !DIExpression()), !dbg !10
  %19 = sub nsw i64 %18, 1, !dbg !34
  %20 = sub nsw i64 %17, %19, !dbg !34
  call void @llvm.dbg.declare(metadata i64* %z_b_2_318, metadata !33, metadata !DIExpression()), !dbg !10
  store i64 %20, i64* %z_b_2_318, align 8, !dbg !34
  %21 = load i64, i64* %z_b_0_316, align 8, !dbg !34
  call void @llvm.dbg.value(metadata i64 %21, metadata !33, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_b_3_319, metadata !33, metadata !DIExpression()), !dbg !10
  store i64 %21, i64* %z_b_3_319, align 8, !dbg !34
  %22 = bitcast i64* %z_b_2_318 to i8*, !dbg !34
  %23 = bitcast i64* @.C351_MAIN_ to i8*, !dbg !34
  %24 = bitcast i64* @.C352_MAIN_ to i8*, !dbg !34
  %25 = bitcast double** %.Z0969_324 to i8*, !dbg !34
  %26 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !34
  %27 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !34
  %28 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !34
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %28(i8* %22, i8* %23, i8* %24, i8* null, i8* %25, i8* null, i8* %26, i8* %27, i8* null, i64 0), !dbg !34
  %29 = load i32, i32* %len_323, align 4, !dbg !35
  call void @llvm.dbg.value(metadata i32 %29, metadata !31, metadata !DIExpression()), !dbg !10
  store i32 %29, i32* %.dY0001_361, align 4, !dbg !35
  call void @llvm.dbg.declare(metadata i32* %i_311, metadata !36, metadata !DIExpression()), !dbg !10
  store i32 1, i32* %i_311, align 4, !dbg !35
  %30 = load i32, i32* %.dY0001_361, align 4, !dbg !35
  %31 = icmp sle i32 %30, 0, !dbg !35
  br i1 %31, label %L.LB1_360, label %L.LB1_359, !dbg !35

L.LB1_359:                                        ; preds = %L.LB1_359, %L.LB1_380
  %32 = load i32, i32* %i_311, align 4, !dbg !37
  call void @llvm.dbg.value(metadata i32 %32, metadata !36, metadata !DIExpression()), !dbg !10
  %33 = sitofp i32 %32 to double, !dbg !37
  %34 = load i32, i32* %i_311, align 4, !dbg !37
  call void @llvm.dbg.value(metadata i32 %34, metadata !36, metadata !DIExpression()), !dbg !10
  %35 = sext i32 %34 to i64, !dbg !37
  %36 = bitcast [16 x i64]* %"a$sd1_350" to i8*, !dbg !37
  %37 = getelementptr i8, i8* %36, i64 56, !dbg !37
  %38 = bitcast i8* %37 to i64*, !dbg !37
  %39 = load i64, i64* %38, align 8, !dbg !37
  %40 = add nsw i64 %35, %39, !dbg !37
  %41 = load double*, double** %.Z0969_324, align 8, !dbg !37
  call void @llvm.dbg.value(metadata double* %41, metadata !21, metadata !DIExpression()), !dbg !10
  %42 = bitcast double* %41 to i8*, !dbg !37
  %43 = getelementptr i8, i8* %42, i64 -8, !dbg !37
  %44 = bitcast i8* %43 to double*, !dbg !37
  %45 = getelementptr double, double* %44, i64 %40, !dbg !37
  store double %33, double* %45, align 8, !dbg !37
  %46 = load i32, i32* %i_311, align 4, !dbg !38
  call void @llvm.dbg.value(metadata i32 %46, metadata !36, metadata !DIExpression()), !dbg !10
  %47 = add nsw i32 %46, 1, !dbg !38
  store i32 %47, i32* %i_311, align 4, !dbg !38
  %48 = load i32, i32* %.dY0001_361, align 4, !dbg !38
  %49 = sub nsw i32 %48, 1, !dbg !38
  store i32 %49, i32* %.dY0001_361, align 4, !dbg !38
  %50 = load i32, i32* %.dY0001_361, align 4, !dbg !38
  %51 = icmp sgt i32 %50, 0, !dbg !38
  br i1 %51, label %L.LB1_359, label %L.LB1_360, !dbg !38

L.LB1_360:                                        ; preds = %L.LB1_359, %L.LB1_380
  call void @llvm.dbg.declare(metadata float* %u_314, metadata !39, metadata !DIExpression()), !dbg !10
  %52 = bitcast float* %u_314 to i8*, !dbg !41
  %53 = bitcast i64* @.C348_MAIN_ to i8*, !dbg !41
  %54 = bitcast void (...)* @fort_rnum_i8 to void (i8*, i8*, ...)*, !dbg !41
  call void (i8*, i8*, ...) %54(i8* %52, i8* %53), !dbg !41
  %55 = load float, float* %u_314, align 4, !dbg !42
  call void @llvm.dbg.value(metadata float %55, metadata !39, metadata !DIExpression()), !dbg !10
  %56 = fmul fast float %55, 1.000000e+02, !dbg !42
  %57 = call float @llvm.floor.f32(float %56), !dbg !42
  %58 = fptosi float %57 to i32, !dbg !42
  call void @llvm.dbg.declare(metadata i32* %j_313, metadata !43, metadata !DIExpression()), !dbg !10
  store i32 %58, i32* %j_313, align 4, !dbg !42
  %59 = bitcast i32* %len_323 to i8*, !dbg !44
  %60 = bitcast %astruct.dt68* %.uplevelArgPack0001_406 to i8**, !dbg !44
  store i8* %59, i8** %60, align 8, !dbg !44
  %61 = bitcast double** %.Z0969_324 to i8*, !dbg !44
  %62 = bitcast %astruct.dt68* %.uplevelArgPack0001_406 to i8*, !dbg !44
  %63 = getelementptr i8, i8* %62, i64 8, !dbg !44
  %64 = bitcast i8* %63 to i8**, !dbg !44
  store i8* %61, i8** %64, align 8, !dbg !44
  %65 = bitcast double** %.Z0969_324 to i8*, !dbg !44
  %66 = bitcast %astruct.dt68* %.uplevelArgPack0001_406 to i8*, !dbg !44
  %67 = getelementptr i8, i8* %66, i64 16, !dbg !44
  %68 = bitcast i8* %67 to i8**, !dbg !44
  store i8* %65, i8** %68, align 8, !dbg !44
  %69 = bitcast i64* %z_b_0_316 to i8*, !dbg !44
  %70 = bitcast %astruct.dt68* %.uplevelArgPack0001_406 to i8*, !dbg !44
  %71 = getelementptr i8, i8* %70, i64 24, !dbg !44
  %72 = bitcast i8* %71 to i8**, !dbg !44
  store i8* %69, i8** %72, align 8, !dbg !44
  %73 = bitcast i64* %z_b_1_317 to i8*, !dbg !44
  %74 = bitcast %astruct.dt68* %.uplevelArgPack0001_406 to i8*, !dbg !44
  %75 = getelementptr i8, i8* %74, i64 32, !dbg !44
  %76 = bitcast i8* %75 to i8**, !dbg !44
  store i8* %73, i8** %76, align 8, !dbg !44
  %77 = bitcast i64* %z_e_62_320 to i8*, !dbg !44
  %78 = bitcast %astruct.dt68* %.uplevelArgPack0001_406 to i8*, !dbg !44
  %79 = getelementptr i8, i8* %78, i64 40, !dbg !44
  %80 = bitcast i8* %79 to i8**, !dbg !44
  store i8* %77, i8** %80, align 8, !dbg !44
  %81 = bitcast i64* %z_b_2_318 to i8*, !dbg !44
  %82 = bitcast %astruct.dt68* %.uplevelArgPack0001_406 to i8*, !dbg !44
  %83 = getelementptr i8, i8* %82, i64 48, !dbg !44
  %84 = bitcast i8* %83 to i8**, !dbg !44
  store i8* %81, i8** %84, align 8, !dbg !44
  %85 = bitcast i64* %z_b_3_319 to i8*, !dbg !44
  %86 = bitcast %astruct.dt68* %.uplevelArgPack0001_406 to i8*, !dbg !44
  %87 = getelementptr i8, i8* %86, i64 56, !dbg !44
  %88 = bitcast i8* %87 to i8**, !dbg !44
  store i8* %85, i8** %88, align 8, !dbg !44
  %89 = bitcast [16 x i64]* %"a$sd1_350" to i8*, !dbg !44
  %90 = bitcast %astruct.dt68* %.uplevelArgPack0001_406 to i8*, !dbg !44
  %91 = getelementptr i8, i8* %90, i64 64, !dbg !44
  %92 = bitcast i8* %91 to i8**, !dbg !44
  store i8* %89, i8** %92, align 8, !dbg !44
  %93 = load i32, i32* %j_313, align 4, !dbg !44
  call void @llvm.dbg.value(metadata i32 %93, metadata !43, metadata !DIExpression()), !dbg !10
  %94 = load i32, i32* %j_313, align 4, !dbg !44
  call void @llvm.dbg.value(metadata i32 %94, metadata !43, metadata !DIExpression()), !dbg !10
  %95 = load i32, i32* %j_313, align 4, !dbg !44
  call void @llvm.dbg.value(metadata i32 %95, metadata !43, metadata !DIExpression()), !dbg !10
  %96 = lshr i32 %95, 31, !dbg !44
  %97 = add nsw i32 %94, %96, !dbg !44
  %98 = ashr i32 %97, 1, !dbg !44
  %99 = mul nsw i32 %98, 2, !dbg !44
  %100 = icmp eq i32 %93, %99, !dbg !44
  %101 = sext i1 %100 to i32, !dbg !44
  %102 = xor i32 -1, %101, !dbg !44
  %103 = icmp eq i32 %102, 0, !dbg !44
  br i1 %103, label %L.LB1_425, label %L.LB1_460, !dbg !44

L.LB1_460:                                        ; preds = %L.LB1_360
  %104 = load i32, i32* %__gtid_MAIN__427, align 4, !dbg !44
  call void @__kmpc_serialized_parallel(i64* null, i32 %104), !dbg !44
  %105 = bitcast %astruct.dt68* %.uplevelArgPack0001_406 to i64*, !dbg !44
  call void @__nv_MAIN__F1L30_1_(i32* %__gtid_MAIN__427, i64* null, i64* %105), !dbg !44
  %106 = load i32, i32* %__gtid_MAIN__427, align 4, !dbg !44
  call void @__kmpc_end_serialized_parallel(i64* null, i32 %106), !dbg !44
  br label %L.LB1_426, !dbg !44

L.LB1_425:                                        ; preds = %L.LB1_360
  %107 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L30_1_ to i64*, !dbg !44
  %108 = bitcast %astruct.dt68* %.uplevelArgPack0001_406 to i64*, !dbg !44
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %107, i64* %108), !dbg !44
  br label %L.LB1_426

L.LB1_426:                                        ; preds = %L.LB1_425, %L.LB1_460
  call void (...) @_mp_bcs_nest(), !dbg !45
  %109 = bitcast i32* @.C335_MAIN_ to i8*, !dbg !45
  %110 = bitcast [47 x i8]* @.C333_MAIN_ to i8*, !dbg !45
  %111 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !45
  call void (i8*, i8*, i64, ...) %111(i8* %109, i8* %110, i64 47), !dbg !45
  %112 = bitcast i32* @.C336_MAIN_ to i8*, !dbg !45
  %113 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !45
  %114 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !45
  %115 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !45
  %116 = call i32 (i8*, i8*, i8*, i8*, ...) %115(i8* %112, i8* null, i8* %113, i8* %114), !dbg !45
  call void @llvm.dbg.declare(metadata i32* %z__io_338, metadata !46, metadata !DIExpression()), !dbg !10
  store i32 %116, i32* %z__io_338, align 4, !dbg !45
  %117 = bitcast [7 x i8]* @.C339_MAIN_ to i8*, !dbg !45
  %118 = bitcast i32 (...)* @f90io_sc_ch_ldw to i32 (i8*, i32, i64, ...)*, !dbg !45
  %119 = call i32 (i8*, i32, i64, ...) %118(i8* %117, i32 14, i64 7), !dbg !45
  store i32 %119, i32* %z__io_338, align 4, !dbg !45
  %120 = bitcast [16 x i64]* %"a$sd1_350" to i8*, !dbg !45
  %121 = getelementptr i8, i8* %120, i64 56, !dbg !45
  %122 = bitcast i8* %121 to i64*, !dbg !45
  %123 = load i64, i64* %122, align 8, !dbg !45
  %124 = load double*, double** %.Z0969_324, align 8, !dbg !45
  call void @llvm.dbg.value(metadata double* %124, metadata !21, metadata !DIExpression()), !dbg !10
  %125 = bitcast double* %124 to i8*, !dbg !45
  %126 = getelementptr i8, i8* %125, i64 392, !dbg !45
  %127 = bitcast i8* %126 to double*, !dbg !45
  %128 = getelementptr double, double* %127, i64 %123, !dbg !45
  %129 = load double, double* %128, align 8, !dbg !45
  %130 = bitcast i32 (...)* @f90io_sc_d_ldw to i32 (double, i32, ...)*, !dbg !45
  %131 = call i32 (double, i32, ...) %130(double %129, i32 28), !dbg !45
  store i32 %131, i32* %z__io_338, align 4, !dbg !45
  %132 = call i32 (...) @f90io_ldw_end(), !dbg !45
  store i32 %132, i32* %z__io_338, align 4, !dbg !45
  call void (...) @_mp_ecs_nest(), !dbg !45
  %133 = load double*, double** %.Z0969_324, align 8, !dbg !47
  call void @llvm.dbg.value(metadata double* %133, metadata !21, metadata !DIExpression()), !dbg !10
  %134 = bitcast double* %133 to i8*, !dbg !47
  %135 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !47
  %136 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !47
  call void (i8*, i8*, i8*, i8*, i64, ...) %136(i8* null, i8* %134, i8* %135, i8* null, i64 0), !dbg !47
  %137 = bitcast double** %.Z0969_324 to i8**, !dbg !47
  store i8* null, i8** %137, align 8, !dbg !47
  %138 = bitcast [16 x i64]* %"a$sd1_350" to i64*, !dbg !47
  store i64 0, i64* %138, align 8, !dbg !47
  ret void, !dbg !19
}

define internal void @__nv_MAIN__F1L30_1_(i32* %__nv_MAIN__F1L30_1Arg0, i64* %__nv_MAIN__F1L30_1Arg1, i64* %__nv_MAIN__F1L30_1Arg2) #0 !dbg !48 {
L.entry:
  %__gtid___nv_MAIN__F1L30_1__479 = alloca i32, align 4
  %.i0000p_330 = alloca i32, align 4
  %i_329 = alloca i32, align 4
  %.du0002p_365 = alloca i32, align 4
  %.de0002p_366 = alloca i32, align 4
  %.di0002p_367 = alloca i32, align 4
  %.ds0002p_368 = alloca i32, align 4
  %.dl0002p_370 = alloca i32, align 4
  %.dl0002p.copy_473 = alloca i32, align 4
  %.de0002p.copy_474 = alloca i32, align 4
  %.ds0002p.copy_475 = alloca i32, align 4
  %.dX0002p_369 = alloca i32, align 4
  %.dY0002p_364 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L30_1Arg0, metadata !51, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L30_1Arg1, metadata !53, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L30_1Arg2, metadata !54, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.value(metadata i32 1, metadata !55, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.value(metadata i32 2, metadata !56, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.value(metadata i32 0, metadata !57, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.value(metadata i32 1, metadata !58, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.value(metadata i32 2, metadata !59, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.value(metadata i32 0, metadata !60, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.value(metadata i32 1, metadata !61, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.value(metadata i32 2, metadata !62, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.value(metadata i32 8, metadata !63, metadata !DIExpression()), !dbg !52
  %0 = load i32, i32* %__nv_MAIN__F1L30_1Arg0, align 4, !dbg !64
  store i32 %0, i32* %__gtid___nv_MAIN__F1L30_1__479, align 4, !dbg !64
  br label %L.LB2_464

L.LB2_464:                                        ; preds = %L.entry
  br label %L.LB2_328

L.LB2_328:                                        ; preds = %L.LB2_464
  store i32 0, i32* %.i0000p_330, align 4, !dbg !65
  call void @llvm.dbg.declare(metadata i32* %i_329, metadata !66, metadata !DIExpression()), !dbg !64
  store i32 1, i32* %i_329, align 4, !dbg !65
  %1 = bitcast i64* %__nv_MAIN__F1L30_1Arg2 to i32**, !dbg !65
  %2 = load i32*, i32** %1, align 8, !dbg !65
  %3 = load i32, i32* %2, align 4, !dbg !65
  %4 = sub nsw i32 %3, 1, !dbg !65
  store i32 %4, i32* %.du0002p_365, align 4, !dbg !65
  %5 = bitcast i64* %__nv_MAIN__F1L30_1Arg2 to i32**, !dbg !65
  %6 = load i32*, i32** %5, align 8, !dbg !65
  %7 = load i32, i32* %6, align 4, !dbg !65
  %8 = sub nsw i32 %7, 1, !dbg !65
  store i32 %8, i32* %.de0002p_366, align 4, !dbg !65
  store i32 1, i32* %.di0002p_367, align 4, !dbg !65
  %9 = load i32, i32* %.di0002p_367, align 4, !dbg !65
  store i32 %9, i32* %.ds0002p_368, align 4, !dbg !65
  store i32 1, i32* %.dl0002p_370, align 4, !dbg !65
  %10 = load i32, i32* %.dl0002p_370, align 4, !dbg !65
  store i32 %10, i32* %.dl0002p.copy_473, align 4, !dbg !65
  %11 = load i32, i32* %.de0002p_366, align 4, !dbg !65
  store i32 %11, i32* %.de0002p.copy_474, align 4, !dbg !65
  %12 = load i32, i32* %.ds0002p_368, align 4, !dbg !65
  store i32 %12, i32* %.ds0002p.copy_475, align 4, !dbg !65
  %13 = load i32, i32* %__gtid___nv_MAIN__F1L30_1__479, align 4, !dbg !65
  %14 = bitcast i32* %.i0000p_330 to i64*, !dbg !65
  %15 = bitcast i32* %.dl0002p.copy_473 to i64*, !dbg !65
  %16 = bitcast i32* %.de0002p.copy_474 to i64*, !dbg !65
  %17 = bitcast i32* %.ds0002p.copy_475 to i64*, !dbg !65
  %18 = load i32, i32* %.ds0002p.copy_475, align 4, !dbg !65
  call void @__kmpc_for_static_init_4(i64* null, i32 %13, i32 34, i64* %14, i64* %15, i64* %16, i64* %17, i32 %18, i32 1), !dbg !65
  %19 = load i32, i32* %.dl0002p.copy_473, align 4, !dbg !65
  store i32 %19, i32* %.dl0002p_370, align 4, !dbg !65
  %20 = load i32, i32* %.de0002p.copy_474, align 4, !dbg !65
  store i32 %20, i32* %.de0002p_366, align 4, !dbg !65
  %21 = load i32, i32* %.ds0002p.copy_475, align 4, !dbg !65
  store i32 %21, i32* %.ds0002p_368, align 4, !dbg !65
  %22 = load i32, i32* %.dl0002p_370, align 4, !dbg !65
  store i32 %22, i32* %i_329, align 4, !dbg !65
  %23 = load i32, i32* %i_329, align 4, !dbg !65
  call void @llvm.dbg.value(metadata i32 %23, metadata !66, metadata !DIExpression()), !dbg !64
  store i32 %23, i32* %.dX0002p_369, align 4, !dbg !65
  %24 = load i32, i32* %.dX0002p_369, align 4, !dbg !65
  %25 = load i32, i32* %.du0002p_365, align 4, !dbg !65
  %26 = icmp sgt i32 %24, %25, !dbg !65
  br i1 %26, label %L.LB2_363, label %L.LB2_503, !dbg !65

L.LB2_503:                                        ; preds = %L.LB2_328
  %27 = load i32, i32* %.dX0002p_369, align 4, !dbg !65
  store i32 %27, i32* %i_329, align 4, !dbg !65
  %28 = load i32, i32* %.di0002p_367, align 4, !dbg !65
  %29 = load i32, i32* %.de0002p_366, align 4, !dbg !65
  %30 = load i32, i32* %.dX0002p_369, align 4, !dbg !65
  %31 = sub nsw i32 %29, %30, !dbg !65
  %32 = add nsw i32 %28, %31, !dbg !65
  %33 = load i32, i32* %.di0002p_367, align 4, !dbg !65
  %34 = sdiv i32 %32, %33, !dbg !65
  store i32 %34, i32* %.dY0002p_364, align 4, !dbg !65
  %35 = load i32, i32* %.dY0002p_364, align 4, !dbg !65
  %36 = icmp sle i32 %35, 0, !dbg !65
  br i1 %36, label %L.LB2_373, label %L.LB2_372, !dbg !65

L.LB2_372:                                        ; preds = %L.LB2_372, %L.LB2_503
  %37 = load i32, i32* %i_329, align 4, !dbg !67
  call void @llvm.dbg.value(metadata i32 %37, metadata !66, metadata !DIExpression()), !dbg !64
  %38 = sext i32 %37 to i64, !dbg !67
  %39 = bitcast i64* %__nv_MAIN__F1L30_1Arg2 to i8*, !dbg !67
  %40 = getelementptr i8, i8* %39, i64 64, !dbg !67
  %41 = bitcast i8* %40 to i8**, !dbg !67
  %42 = load i8*, i8** %41, align 8, !dbg !67
  %43 = getelementptr i8, i8* %42, i64 56, !dbg !67
  %44 = bitcast i8* %43 to i64*, !dbg !67
  %45 = load i64, i64* %44, align 8, !dbg !67
  %46 = add nsw i64 %38, %45, !dbg !67
  %47 = bitcast i64* %__nv_MAIN__F1L30_1Arg2 to i8*, !dbg !67
  %48 = getelementptr i8, i8* %47, i64 16, !dbg !67
  %49 = bitcast i8* %48 to i8***, !dbg !67
  %50 = load i8**, i8*** %49, align 8, !dbg !67
  %51 = load i8*, i8** %50, align 8, !dbg !67
  %52 = getelementptr i8, i8* %51, i64 -8, !dbg !67
  %53 = bitcast i8* %52 to double*, !dbg !67
  %54 = getelementptr double, double* %53, i64 %46, !dbg !67
  %55 = load double, double* %54, align 8, !dbg !67
  %56 = fadd fast double %55, 1.000000e+00, !dbg !67
  %57 = load i32, i32* %i_329, align 4, !dbg !67
  call void @llvm.dbg.value(metadata i32 %57, metadata !66, metadata !DIExpression()), !dbg !64
  %58 = sext i32 %57 to i64, !dbg !67
  %59 = bitcast i64* %__nv_MAIN__F1L30_1Arg2 to i8*, !dbg !67
  %60 = getelementptr i8, i8* %59, i64 64, !dbg !67
  %61 = bitcast i8* %60 to i8**, !dbg !67
  %62 = load i8*, i8** %61, align 8, !dbg !67
  %63 = getelementptr i8, i8* %62, i64 56, !dbg !67
  %64 = bitcast i8* %63 to i64*, !dbg !67
  %65 = load i64, i64* %64, align 8, !dbg !67
  %66 = add nsw i64 %58, %65, !dbg !67
  %67 = bitcast i64* %__nv_MAIN__F1L30_1Arg2 to i8*, !dbg !67
  %68 = getelementptr i8, i8* %67, i64 16, !dbg !67
  %69 = bitcast i8* %68 to double***, !dbg !67
  %70 = load double**, double*** %69, align 8, !dbg !67
  %71 = load double*, double** %70, align 8, !dbg !67
  %72 = getelementptr double, double* %71, i64 %66, !dbg !67
  store double %56, double* %72, align 8, !dbg !67
  %73 = load i32, i32* %.di0002p_367, align 4, !dbg !64
  %74 = load i32, i32* %i_329, align 4, !dbg !64
  call void @llvm.dbg.value(metadata i32 %74, metadata !66, metadata !DIExpression()), !dbg !64
  %75 = add nsw i32 %73, %74, !dbg !64
  store i32 %75, i32* %i_329, align 4, !dbg !64
  %76 = load i32, i32* %.dY0002p_364, align 4, !dbg !64
  %77 = sub nsw i32 %76, 1, !dbg !64
  store i32 %77, i32* %.dY0002p_364, align 4, !dbg !64
  %78 = load i32, i32* %.dY0002p_364, align 4, !dbg !64
  %79 = icmp sgt i32 %78, 0, !dbg !64
  br i1 %79, label %L.LB2_372, label %L.LB2_373, !dbg !64

L.LB2_373:                                        ; preds = %L.LB2_372, %L.LB2_503
  br label %L.LB2_363

L.LB2_363:                                        ; preds = %L.LB2_373, %L.LB2_328
  %80 = load i32, i32* %__gtid___nv_MAIN__F1L30_1__479, align 4, !dbg !64
  call void @__kmpc_for_static_fini(i64* null, i32 %80), !dbg !64
  br label %L.LB2_331

L.LB2_331:                                        ; preds = %L.LB2_363
  ret void, !dbg !64
}

declare void @__kmpc_for_static_fini(i64*, i32) #0

declare void @__kmpc_for_static_init_4(i64*, i32, i32, i64*, i64*, i64*, i64*, i32, i32) #0

declare void @f90_dealloc03a_i8(...) #0

declare void @_mp_ecs_nest(...) #0

declare signext i32 @f90io_ldw_end(...) #0

declare signext i32 @f90io_sc_d_ldw(...) #0

declare signext i32 @f90io_sc_ch_ldw(...) #0

declare signext i32 @f90io_print_init(...) #0

declare void @f90io_src_info03a(...) #0

declare void @_mp_bcs_nest(...) #0

; Function Attrs: nounwind readnone speculatable
declare float @llvm.floor.f32(float) #1

declare void @fort_rnum_i8(...) #0

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

; Function Attrs: nounwind readnone
declare float @__fs_floor_1(float) #2

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { "no-frame-pointer-elim-non-leaf" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { nounwind readnone "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "unsafe-fp-math"="true" "use-soft-float"="false" }

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !4, globals: !4, imports: !4)
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB114-if-orig-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb114_if_orig_yes", scope: !2, file: !3, line: 11, type: !6, scopeLine: 11, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
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
!18 = !DILocalVariable(name: "dp", scope: !5, file: !3, type: !9)
!19 = !DILocation(line: 39, column: 1, scope: !5)
!20 = !DILocation(line: 11, column: 1, scope: !5)
!21 = !DILocalVariable(name: "a", scope: !5, file: !3, type: !22)
!22 = !DICompositeType(tag: DW_TAG_array_type, baseType: !23, size: 64, align: 64, elements: !24)
!23 = !DIBasicType(name: "double precision", size: 64, align: 64, encoding: DW_ATE_float)
!24 = !{!25}
!25 = !DISubrange(count: 0, lowerBound: 1)
!26 = !DILocalVariable(scope: !5, file: !3, type: !27, flags: DIFlagArtificial)
!27 = !DICompositeType(tag: DW_TAG_array_type, baseType: !28, size: 1024, align: 64, elements: !29)
!28 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!29 = !{!30}
!30 = !DISubrange(count: 16, lowerBound: 1)
!31 = !DILocalVariable(name: "len", scope: !5, file: !3, type: !9)
!32 = !DILocation(line: 20, column: 1, scope: !5)
!33 = !DILocalVariable(scope: !5, file: !3, type: !28, flags: DIFlagArtificial)
!34 = !DILocation(line: 21, column: 1, scope: !5)
!35 = !DILocation(line: 23, column: 1, scope: !5)
!36 = !DILocalVariable(name: "i", scope: !5, file: !3, type: !9)
!37 = !DILocation(line: 24, column: 1, scope: !5)
!38 = !DILocation(line: 25, column: 1, scope: !5)
!39 = !DILocalVariable(name: "u", scope: !5, file: !3, type: !40)
!40 = !DIBasicType(name: "real", size: 32, align: 32, encoding: DW_ATE_float)
!41 = !DILocation(line: 27, column: 1, scope: !5)
!42 = !DILocation(line: 28, column: 1, scope: !5)
!43 = !DILocalVariable(name: "j", scope: !5, file: !3, type: !9)
!44 = !DILocation(line: 30, column: 1, scope: !5)
!45 = !DILocation(line: 36, column: 1, scope: !5)
!46 = !DILocalVariable(scope: !5, file: !3, type: !9, flags: DIFlagArtificial)
!47 = !DILocation(line: 38, column: 1, scope: !5)
!48 = distinct !DISubprogram(name: "__nv_MAIN__F1L30_1", scope: !2, file: !3, line: 30, type: !49, scopeLine: 30, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!49 = !DISubroutineType(types: !50)
!50 = !{null, !9, !28, !28}
!51 = !DILocalVariable(name: "__nv_MAIN__F1L30_1Arg0", arg: 1, scope: !48, file: !3, type: !9)
!52 = !DILocation(line: 0, scope: !48)
!53 = !DILocalVariable(name: "__nv_MAIN__F1L30_1Arg1", arg: 2, scope: !48, file: !3, type: !28)
!54 = !DILocalVariable(name: "__nv_MAIN__F1L30_1Arg2", arg: 3, scope: !48, file: !3, type: !28)
!55 = !DILocalVariable(name: "omp_sched_static", scope: !48, file: !3, type: !9)
!56 = !DILocalVariable(name: "omp_sched_dynamic", scope: !48, file: !3, type: !9)
!57 = !DILocalVariable(name: "omp_proc_bind_false", scope: !48, file: !3, type: !9)
!58 = !DILocalVariable(name: "omp_proc_bind_true", scope: !48, file: !3, type: !9)
!59 = !DILocalVariable(name: "omp_proc_bind_master", scope: !48, file: !3, type: !9)
!60 = !DILocalVariable(name: "omp_lock_hint_none", scope: !48, file: !3, type: !9)
!61 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !48, file: !3, type: !9)
!62 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !48, file: !3, type: !9)
!63 = !DILocalVariable(name: "dp", scope: !48, file: !3, type: !9)
!64 = !DILocation(line: 33, column: 1, scope: !48)
!65 = !DILocation(line: 31, column: 1, scope: !48)
!66 = !DILocalVariable(name: "i", scope: !48, file: !3, type: !9)
!67 = !DILocation(line: 32, column: 1, scope: !48)

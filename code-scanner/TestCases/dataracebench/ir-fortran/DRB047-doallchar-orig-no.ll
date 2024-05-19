; ModuleID = '/tmp/DRB047-doallchar-orig-no-b4728d.ll'
source_filename = "/tmp/DRB047-doallchar-orig-no-b4728d.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%astruct.dt74 = type <{ i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8* }>

@.C341_MAIN_ = internal constant i64 23
@.C340_MAIN_ = internal constant [4 x i8] c"a(i)"
@.C338_MAIN_ = internal constant i32 6
@.C337_MAIN_ = internal constant i32 29
@.C306_MAIN_ = internal constant i32 25
@.C328_MAIN_ = internal constant [5 x i8] c"(i10)"
@.C325_MAIN_ = internal constant [53 x i8] c"micro-benchmarks-fortran/DRB047-doallchar-orig-no.f95"
@.C327_MAIN_ = internal constant i32 24
@.C307_MAIN_ = internal constant i32 100
@.C285_MAIN_ = internal constant i32 1
@.C305_MAIN_ = internal constant i32 14
@.C348_MAIN_ = internal constant i64 14
@.C316_MAIN_ = internal constant i64 100
@.C286_MAIN_ = internal constant i64 1
@.C284_MAIN_ = internal constant i64 0
@.C283_MAIN_ = internal constant i32 0
@.C306___nv_MAIN__F1L22_1 = internal constant i32 25
@.C284___nv_MAIN__F1L22_1 = internal constant i64 0
@.C328___nv_MAIN__F1L22_1 = internal constant [5 x i8] c"(i10)"
@.C305___nv_MAIN__F1L22_1 = internal constant i32 14
@.C325___nv_MAIN__F1L22_1 = internal constant [53 x i8] c"micro-benchmarks-fortran/DRB047-doallchar-orig-no.f95"
@.C327___nv_MAIN__F1L22_1 = internal constant i32 24
@.C307___nv_MAIN__F1L22_1 = internal constant i32 100
@.C285___nv_MAIN__F1L22_1 = internal constant i32 1
@.C283___nv_MAIN__F1L22_1 = internal constant i32 0

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__407 = alloca i32, align 4
  %.Z0965_317 = alloca [100 x i8]*, align 8
  %"a$sd1_347" = alloca [16 x i64], align 8
  %z_b_0_308 = alloca i64, align 8
  %z_b_1_309 = alloca i64, align 8
  %z_e_61_312 = alloca i64, align 8
  %z_b_2_310 = alloca i64, align 8
  %z_b_3_311 = alloca i64, align 8
  %.uplevelArgPack0001_387 = alloca %astruct.dt74, align 16
  %z__io_330 = alloca i32, align 4
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
  call void @llvm.dbg.declare(metadata [100 x i8]** %.Z0965_317, metadata !17, metadata !DIExpression(DW_OP_deref)), !dbg !10
  %3 = bitcast [100 x i8]** %.Z0965_317 to i8**, !dbg !16
  store i8* null, i8** %3, align 8, !dbg !16
  call void @llvm.dbg.declare(metadata [16 x i64]* %"a$sd1_347", metadata !22, metadata !DIExpression()), !dbg !10
  %4 = bitcast [16 x i64]* %"a$sd1_347" to i64*, !dbg !16
  store i64 0, i64* %4, align 8, !dbg !16
  br label %L.LB1_373

L.LB1_373:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i64* %z_b_0_308, metadata !27, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_0_308, align 8, !dbg !28
  call void @llvm.dbg.declare(metadata i64* %z_b_1_309, metadata !27, metadata !DIExpression()), !dbg !10
  store i64 100, i64* %z_b_1_309, align 8, !dbg !28
  %5 = load i64, i64* %z_b_1_309, align 8, !dbg !28
  call void @llvm.dbg.value(metadata i64 %5, metadata !27, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_61_312, metadata !27, metadata !DIExpression()), !dbg !10
  store i64 %5, i64* %z_e_61_312, align 8, !dbg !28
  %6 = bitcast [16 x i64]* %"a$sd1_347" to i8*, !dbg !28
  %7 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !28
  %8 = bitcast i64* @.C348_MAIN_ to i8*, !dbg !28
  %9 = bitcast i64* @.C316_MAIN_ to i8*, !dbg !28
  %10 = bitcast i64* %z_b_0_308 to i8*, !dbg !28
  %11 = bitcast i64* %z_b_1_309 to i8*, !dbg !28
  %12 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !28
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %12(i8* %6, i8* %7, i8* %8, i8* %9, i8* %10, i8* %11), !dbg !28
  %13 = bitcast [16 x i64]* %"a$sd1_347" to i8*, !dbg !28
  %14 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !28
  call void (i8*, i32, ...) %14(i8* %13, i32 14), !dbg !28
  %15 = load i64, i64* %z_b_1_309, align 8, !dbg !28
  call void @llvm.dbg.value(metadata i64 %15, metadata !27, metadata !DIExpression()), !dbg !10
  %16 = load i64, i64* %z_b_0_308, align 8, !dbg !28
  call void @llvm.dbg.value(metadata i64 %16, metadata !27, metadata !DIExpression()), !dbg !10
  %17 = sub nsw i64 %16, 1, !dbg !28
  %18 = sub nsw i64 %15, %17, !dbg !28
  call void @llvm.dbg.declare(metadata i64* %z_b_2_310, metadata !27, metadata !DIExpression()), !dbg !10
  store i64 %18, i64* %z_b_2_310, align 8, !dbg !28
  %19 = load i64, i64* %z_b_0_308, align 8, !dbg !28
  call void @llvm.dbg.value(metadata i64 %19, metadata !27, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_b_3_311, metadata !27, metadata !DIExpression()), !dbg !10
  store i64 %19, i64* %z_b_3_311, align 8, !dbg !28
  %20 = bitcast i64* %z_b_2_310 to i8*, !dbg !28
  %21 = bitcast i64* @.C348_MAIN_ to i8*, !dbg !28
  %22 = bitcast i64* @.C316_MAIN_ to i8*, !dbg !28
  %23 = bitcast [100 x i8]** %.Z0965_317 to i8*, !dbg !28
  %24 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !28
  %25 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !28
  %26 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !28
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %26(i8* %20, i8* %21, i8* %22, i8* null, i8* %23, i8* null, i8* %24, i8* %25, i8* null, i64 0), !dbg !28
  %27 = bitcast [100 x i8]** %.Z0965_317 to i8*, !dbg !29
  %28 = bitcast %astruct.dt74* %.uplevelArgPack0001_387 to i8**, !dbg !29
  store i8* %27, i8** %28, align 8, !dbg !29
  %29 = bitcast [100 x i8]** %.Z0965_317 to i8*, !dbg !29
  %30 = bitcast %astruct.dt74* %.uplevelArgPack0001_387 to i8*, !dbg !29
  %31 = getelementptr i8, i8* %30, i64 16, !dbg !29
  %32 = bitcast i8* %31 to i8**, !dbg !29
  store i8* %29, i8** %32, align 8, !dbg !29
  %33 = bitcast i64* %z_b_0_308 to i8*, !dbg !29
  %34 = bitcast %astruct.dt74* %.uplevelArgPack0001_387 to i8*, !dbg !29
  %35 = getelementptr i8, i8* %34, i64 24, !dbg !29
  %36 = bitcast i8* %35 to i8**, !dbg !29
  store i8* %33, i8** %36, align 8, !dbg !29
  %37 = bitcast i64* %z_b_1_309 to i8*, !dbg !29
  %38 = bitcast %astruct.dt74* %.uplevelArgPack0001_387 to i8*, !dbg !29
  %39 = getelementptr i8, i8* %38, i64 32, !dbg !29
  %40 = bitcast i8* %39 to i8**, !dbg !29
  store i8* %37, i8** %40, align 8, !dbg !29
  %41 = bitcast i64* %z_e_61_312 to i8*, !dbg !29
  %42 = bitcast %astruct.dt74* %.uplevelArgPack0001_387 to i8*, !dbg !29
  %43 = getelementptr i8, i8* %42, i64 40, !dbg !29
  %44 = bitcast i8* %43 to i8**, !dbg !29
  store i8* %41, i8** %44, align 8, !dbg !29
  %45 = bitcast i64* %z_b_2_310 to i8*, !dbg !29
  %46 = bitcast %astruct.dt74* %.uplevelArgPack0001_387 to i8*, !dbg !29
  %47 = getelementptr i8, i8* %46, i64 48, !dbg !29
  %48 = bitcast i8* %47 to i8**, !dbg !29
  store i8* %45, i8** %48, align 8, !dbg !29
  %49 = bitcast i64* %z_b_3_311 to i8*, !dbg !29
  %50 = bitcast %astruct.dt74* %.uplevelArgPack0001_387 to i8*, !dbg !29
  %51 = getelementptr i8, i8* %50, i64 56, !dbg !29
  %52 = bitcast i8* %51 to i8**, !dbg !29
  store i8* %49, i8** %52, align 8, !dbg !29
  %53 = bitcast [16 x i64]* %"a$sd1_347" to i8*, !dbg !29
  %54 = bitcast %astruct.dt74* %.uplevelArgPack0001_387 to i8*, !dbg !29
  %55 = getelementptr i8, i8* %54, i64 64, !dbg !29
  %56 = bitcast i8* %55 to i8**, !dbg !29
  store i8* %53, i8** %56, align 8, !dbg !29
  br label %L.LB1_405, !dbg !29

L.LB1_405:                                        ; preds = %L.LB1_373
  %57 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L22_1_ to i64*, !dbg !29
  %58 = bitcast %astruct.dt74* %.uplevelArgPack0001_387 to i64*, !dbg !29
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %57, i64* %58), !dbg !29
  call void (...) @_mp_bcs_nest(), !dbg !30
  %59 = bitcast i32* @.C337_MAIN_ to i8*, !dbg !30
  %60 = bitcast [53 x i8]* @.C325_MAIN_ to i8*, !dbg !30
  %61 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !30
  call void (i8*, i8*, i64, ...) %61(i8* %59, i8* %60, i64 53), !dbg !30
  %62 = bitcast i32* @.C338_MAIN_ to i8*, !dbg !30
  %63 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !30
  %64 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !30
  %65 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !30
  %66 = call i32 (i8*, i8*, i8*, i8*, ...) %65(i8* %62, i8* null, i8* %63, i8* %64), !dbg !30
  call void @llvm.dbg.declare(metadata i32* %z__io_330, metadata !31, metadata !DIExpression()), !dbg !10
  store i32 %66, i32* %z__io_330, align 4, !dbg !30
  %67 = bitcast [4 x i8]* @.C340_MAIN_ to i8*, !dbg !30
  %68 = bitcast i32 (...)* @f90io_sc_ch_ldw to i32 (i8*, i32, i64, ...)*, !dbg !30
  %69 = call i32 (i8*, i32, i64, ...) %68(i8* %67, i32 14, i64 4), !dbg !30
  store i32 %69, i32* %z__io_330, align 4, !dbg !30
  %70 = load [100 x i8]*, [100 x i8]** %.Z0965_317, align 8, !dbg !30
  call void @llvm.dbg.value(metadata [100 x i8]* %70, metadata !17, metadata !DIExpression()), !dbg !10
  %71 = bitcast [100 x i8]* %70 to i8*, !dbg !30
  %72 = getelementptr i8, i8* %71, i64 2200, !dbg !30
  %73 = bitcast [16 x i64]* %"a$sd1_347" to i8*, !dbg !30
  %74 = getelementptr i8, i8* %73, i64 56, !dbg !30
  %75 = bitcast i8* %74 to i64*, !dbg !30
  %76 = load i64, i64* %75, align 8, !dbg !30
  %77 = mul nsw i64 %76, 100, !dbg !30
  %78 = getelementptr i8, i8* %72, i64 %77, !dbg !30
  %79 = bitcast i32 (...)* @f90io_sc_ch_ldw to i32 (i8*, i32, i64, ...)*, !dbg !30
  %80 = call i32 (i8*, i32, i64, ...) %79(i8* %78, i32 14, i64 100), !dbg !30
  store i32 %80, i32* %z__io_330, align 4, !dbg !30
  %81 = call i32 (...) @f90io_ldw_end(), !dbg !30
  store i32 %81, i32* %z__io_330, align 4, !dbg !30
  call void (...) @_mp_ecs_nest(), !dbg !30
  ret void, !dbg !15
}

define internal void @__nv_MAIN__F1L22_1_(i32* %__nv_MAIN__F1L22_1Arg0, i64* %__nv_MAIN__F1L22_1Arg1, i64* %__nv_MAIN__F1L22_1Arg2) #0 !dbg !32 {
L.entry:
  %__gtid___nv_MAIN__F1L22_1__458 = alloca i32, align 4
  %.i0000p_323 = alloca i32, align 4
  %i_322 = alloca i32, align 4
  %.du0001p_358 = alloca i32, align 4
  %.de0001p_359 = alloca i32, align 4
  %.di0001p_360 = alloca i32, align 4
  %.ds0001p_361 = alloca i32, align 4
  %.dl0001p_363 = alloca i32, align 4
  %.dl0001p.copy_452 = alloca i32, align 4
  %.de0001p.copy_453 = alloca i32, align 4
  %.ds0001p.copy_454 = alloca i32, align 4
  %.dX0001p_362 = alloca i32, align 4
  %.dY0001p_357 = alloca i32, align 4
  %z__io_330 = alloca i32, align 4
  %str_321 = alloca [50 x i8], align 16
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L22_1Arg0, metadata !35, metadata !DIExpression()), !dbg !36
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L22_1Arg1, metadata !37, metadata !DIExpression()), !dbg !36
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L22_1Arg2, metadata !38, metadata !DIExpression()), !dbg !36
  call void @llvm.dbg.value(metadata i32 1, metadata !39, metadata !DIExpression()), !dbg !36
  call void @llvm.dbg.value(metadata i32 0, metadata !40, metadata !DIExpression()), !dbg !36
  call void @llvm.dbg.value(metadata i32 1, metadata !41, metadata !DIExpression()), !dbg !36
  call void @llvm.dbg.value(metadata i32 0, metadata !42, metadata !DIExpression()), !dbg !36
  call void @llvm.dbg.value(metadata i32 1, metadata !43, metadata !DIExpression()), !dbg !36
  %0 = load i32, i32* %__nv_MAIN__F1L22_1Arg0, align 4, !dbg !44
  store i32 %0, i32* %__gtid___nv_MAIN__F1L22_1__458, align 4, !dbg !44
  br label %L.LB2_444

L.LB2_444:                                        ; preds = %L.entry
  br label %L.LB2_320

L.LB2_320:                                        ; preds = %L.LB2_444
  store i32 0, i32* %.i0000p_323, align 4, !dbg !45
  call void @llvm.dbg.declare(metadata i32* %i_322, metadata !46, metadata !DIExpression()), !dbg !44
  store i32 1, i32* %i_322, align 4, !dbg !45
  store i32 100, i32* %.du0001p_358, align 4, !dbg !45
  store i32 100, i32* %.de0001p_359, align 4, !dbg !45
  store i32 1, i32* %.di0001p_360, align 4, !dbg !45
  %1 = load i32, i32* %.di0001p_360, align 4, !dbg !45
  store i32 %1, i32* %.ds0001p_361, align 4, !dbg !45
  store i32 1, i32* %.dl0001p_363, align 4, !dbg !45
  %2 = load i32, i32* %.dl0001p_363, align 4, !dbg !45
  store i32 %2, i32* %.dl0001p.copy_452, align 4, !dbg !45
  %3 = load i32, i32* %.de0001p_359, align 4, !dbg !45
  store i32 %3, i32* %.de0001p.copy_453, align 4, !dbg !45
  %4 = load i32, i32* %.ds0001p_361, align 4, !dbg !45
  store i32 %4, i32* %.ds0001p.copy_454, align 4, !dbg !45
  %5 = load i32, i32* %__gtid___nv_MAIN__F1L22_1__458, align 4, !dbg !45
  %6 = bitcast i32* %.i0000p_323 to i64*, !dbg !45
  %7 = bitcast i32* %.dl0001p.copy_452 to i64*, !dbg !45
  %8 = bitcast i32* %.de0001p.copy_453 to i64*, !dbg !45
  %9 = bitcast i32* %.ds0001p.copy_454 to i64*, !dbg !45
  %10 = load i32, i32* %.ds0001p.copy_454, align 4, !dbg !45
  call void @__kmpc_for_static_init_4(i64* null, i32 %5, i32 34, i64* %6, i64* %7, i64* %8, i64* %9, i32 %10, i32 1), !dbg !45
  %11 = load i32, i32* %.dl0001p.copy_452, align 4, !dbg !45
  store i32 %11, i32* %.dl0001p_363, align 4, !dbg !45
  %12 = load i32, i32* %.de0001p.copy_453, align 4, !dbg !45
  store i32 %12, i32* %.de0001p_359, align 4, !dbg !45
  %13 = load i32, i32* %.ds0001p.copy_454, align 4, !dbg !45
  store i32 %13, i32* %.ds0001p_361, align 4, !dbg !45
  %14 = load i32, i32* %.dl0001p_363, align 4, !dbg !45
  store i32 %14, i32* %i_322, align 4, !dbg !45
  %15 = load i32, i32* %i_322, align 4, !dbg !45
  call void @llvm.dbg.value(metadata i32 %15, metadata !46, metadata !DIExpression()), !dbg !44
  store i32 %15, i32* %.dX0001p_362, align 4, !dbg !45
  %16 = load i32, i32* %.dX0001p_362, align 4, !dbg !45
  %17 = load i32, i32* %.du0001p_358, align 4, !dbg !45
  %18 = icmp sgt i32 %16, %17, !dbg !45
  br i1 %18, label %L.LB2_356, label %L.LB2_495, !dbg !45

L.LB2_495:                                        ; preds = %L.LB2_320
  %19 = load i32, i32* %.dX0001p_362, align 4, !dbg !45
  store i32 %19, i32* %i_322, align 4, !dbg !45
  %20 = load i32, i32* %.di0001p_360, align 4, !dbg !45
  %21 = load i32, i32* %.de0001p_359, align 4, !dbg !45
  %22 = load i32, i32* %.dX0001p_362, align 4, !dbg !45
  %23 = sub nsw i32 %21, %22, !dbg !45
  %24 = add nsw i32 %20, %23, !dbg !45
  %25 = load i32, i32* %.di0001p_360, align 4, !dbg !45
  %26 = sdiv i32 %24, %25, !dbg !45
  store i32 %26, i32* %.dY0001p_357, align 4, !dbg !45
  %27 = load i32, i32* %.dY0001p_357, align 4, !dbg !45
  %28 = icmp sle i32 %27, 0, !dbg !45
  br i1 %28, label %L.LB2_366, label %L.LB2_365, !dbg !45

L.LB2_365:                                        ; preds = %L.LB2_365, %L.LB2_495
  call void (...) @_mp_bcs_nest(), !dbg !47
  %29 = bitcast i32* @.C327___nv_MAIN__F1L22_1 to i8*, !dbg !47
  %30 = bitcast [53 x i8]* @.C325___nv_MAIN__F1L22_1 to i8*, !dbg !47
  %31 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !47
  call void (i8*, i8*, i64, ...) %31(i8* %29, i8* %30, i64 53), !dbg !47
  %32 = bitcast i32* @.C305___nv_MAIN__F1L22_1 to i8*, !dbg !47
  %33 = bitcast i32* @.C285___nv_MAIN__F1L22_1 to i8*, !dbg !47
  %34 = bitcast [5 x i8]* @.C328___nv_MAIN__F1L22_1 to i8*, !dbg !47
  %35 = bitcast i32 (...)* @f90io_encode_fmta to i32 (i8*, i8*, i8*, i64, ...)*, !dbg !47
  %36 = call i32 (i8*, i8*, i8*, i64, ...) %35(i8* %32, i8* %33, i8* %34, i64 5), !dbg !47
  call void @llvm.dbg.declare(metadata i32* %z__io_330, metadata !48, metadata !DIExpression()), !dbg !36
  store i32 %36, i32* %z__io_330, align 4, !dbg !47
  call void @llvm.dbg.declare(metadata [50 x i8]* %str_321, metadata !49, metadata !DIExpression()), !dbg !44
  %37 = bitcast [50 x i8]* %str_321 to i8*, !dbg !47
  %38 = bitcast i32* @.C285___nv_MAIN__F1L22_1 to i8*, !dbg !47
  %39 = bitcast i32* @.C283___nv_MAIN__F1L22_1 to i8*, !dbg !47
  %40 = bitcast i32* @.C283___nv_MAIN__F1L22_1 to i8*, !dbg !47
  %41 = bitcast i32 (...)* @f90io_fmtw_intern_inita to i32 (i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !47
  %42 = call i32 (i8*, i8*, i8*, i8*, i8*, i64, ...) %41(i8* %37, i8* %38, i8* %39, i8* %40, i8* null, i64 50), !dbg !47
  store i32 %42, i32* %z__io_330, align 4, !dbg !47
  %43 = load i32, i32* %i_322, align 4, !dbg !47
  call void @llvm.dbg.value(metadata i32 %43, metadata !46, metadata !DIExpression()), !dbg !44
  %44 = bitcast i32 (...)* @f90io_sc_i_fmt_write to i32 (i32, i32, ...)*, !dbg !47
  %45 = call i32 (i32, i32, ...) %44(i32 %43, i32 25), !dbg !47
  store i32 %45, i32* %z__io_330, align 4, !dbg !47
  %46 = call i32 (...) @f90io_fmtw_end(), !dbg !47
  store i32 %46, i32* %z__io_330, align 4, !dbg !47
  call void (...) @_mp_ecs_nest(), !dbg !47
  %47 = bitcast i64* %__nv_MAIN__F1L22_1Arg2 to i8*, !dbg !51
  %48 = getelementptr i8, i8* %47, i64 16, !dbg !51
  %49 = bitcast i8* %48 to i8***, !dbg !51
  %50 = load i8**, i8*** %49, align 8, !dbg !51
  %51 = load i8*, i8** %50, align 8, !dbg !51
  %52 = getelementptr i8, i8* %51, i64 -100, !dbg !51
  %53 = load i32, i32* %i_322, align 4, !dbg !51
  call void @llvm.dbg.value(metadata i32 %53, metadata !46, metadata !DIExpression()), !dbg !44
  %54 = sext i32 %53 to i64, !dbg !51
  %55 = bitcast i64* %__nv_MAIN__F1L22_1Arg2 to i8*, !dbg !51
  %56 = getelementptr i8, i8* %55, i64 64, !dbg !51
  %57 = bitcast i8* %56 to i8**, !dbg !51
  %58 = load i8*, i8** %57, align 8, !dbg !51
  %59 = getelementptr i8, i8* %58, i64 56, !dbg !51
  %60 = bitcast i8* %59 to i64*, !dbg !51
  %61 = load i64, i64* %60, align 8, !dbg !51
  %62 = add nsw i64 %54, %61, !dbg !51
  %63 = mul nsw i64 %62, 100, !dbg !51
  %64 = getelementptr i8, i8* %52, i64 %63, !dbg !51
  %65 = bitcast [50 x i8]* %str_321 to i8*, !dbg !51
  %66 = bitcast i32 (...)* @f90_str_copy_klen to i32 (i32, i8*, i64, i8*, i64, ...)*, !dbg !51
  %67 = call i32 (i32, i8*, i64, i8*, i64, ...) %66(i32 1, i8* %64, i64 100, i8* %65, i64 50), !dbg !51
  %68 = load i32, i32* %.di0001p_360, align 4, !dbg !44
  %69 = load i32, i32* %i_322, align 4, !dbg !44
  call void @llvm.dbg.value(metadata i32 %69, metadata !46, metadata !DIExpression()), !dbg !44
  %70 = add nsw i32 %68, %69, !dbg !44
  store i32 %70, i32* %i_322, align 4, !dbg !44
  %71 = load i32, i32* %.dY0001p_357, align 4, !dbg !44
  %72 = sub nsw i32 %71, 1, !dbg !44
  store i32 %72, i32* %.dY0001p_357, align 4, !dbg !44
  %73 = load i32, i32* %.dY0001p_357, align 4, !dbg !44
  %74 = icmp sgt i32 %73, 0, !dbg !44
  br i1 %74, label %L.LB2_365, label %L.LB2_366, !dbg !44

L.LB2_366:                                        ; preds = %L.LB2_365, %L.LB2_495
  br label %L.LB2_356

L.LB2_356:                                        ; preds = %L.LB2_366, %L.LB2_320
  %75 = load i32, i32* %__gtid___nv_MAIN__F1L22_1__458, align 4, !dbg !44
  call void @__kmpc_for_static_fini(i64* null, i32 %75), !dbg !44
  br label %L.LB2_336

L.LB2_336:                                        ; preds = %L.LB2_356
  ret void, !dbg !44
}

declare signext i32 @f90_str_copy_klen(...) #0

declare signext i32 @f90io_fmtw_end(...) #0

declare signext i32 @f90io_sc_i_fmt_write(...) #0

declare signext i32 @f90io_fmtw_intern_inita(...) #0

declare signext i32 @f90io_encode_fmta(...) #0

declare void @__kmpc_for_static_fini(i64*, i32) #0

declare void @__kmpc_for_static_init_4(i64*, i32, i32, i64*, i64*, i64*, i64*, i32, i32) #0

declare void @_mp_ecs_nest(...) #0

declare signext i32 @f90io_ldw_end(...) #0

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

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { "no-frame-pointer-elim-non-leaf" }
attributes #1 = { nounwind readnone speculatable }

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !4, globals: !4, imports: !4)
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB047-doallchar-orig-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb047_doallchar_orig_no", scope: !2, file: !3, line: 12, type: !6, scopeLine: 12, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!6 = !DISubroutineType(cc: DW_CC_program, types: !7)
!7 = !{null}
!8 = !DILocalVariable(name: "omp_sched_static", scope: !5, file: !3, type: !9)
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 0, scope: !5)
!11 = !DILocalVariable(name: "omp_proc_bind_false", scope: !5, file: !3, type: !9)
!12 = !DILocalVariable(name: "omp_proc_bind_true", scope: !5, file: !3, type: !9)
!13 = !DILocalVariable(name: "omp_lock_hint_none", scope: !5, file: !3, type: !9)
!14 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !5, file: !3, type: !9)
!15 = !DILocation(line: 30, column: 1, scope: !5)
!16 = !DILocation(line: 12, column: 1, scope: !5)
!17 = !DILocalVariable(name: "a", scope: !5, file: !3, type: !18)
!18 = !DICompositeType(tag: DW_TAG_array_type, baseType: !19, size: 800, align: 8, elements: !20)
!19 = !DIBasicType(name: "character", size: 800, align: 8, encoding: DW_ATE_signed)
!20 = !{!21}
!21 = !DISubrange(count: 0, lowerBound: 1)
!22 = !DILocalVariable(scope: !5, file: !3, type: !23, flags: DIFlagArtificial)
!23 = !DICompositeType(tag: DW_TAG_array_type, baseType: !24, size: 1024, align: 64, elements: !25)
!24 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!25 = !{!26}
!26 = !DISubrange(count: 16, lowerBound: 1)
!27 = !DILocalVariable(scope: !5, file: !3, type: !24, flags: DIFlagArtificial)
!28 = !DILocation(line: 20, column: 1, scope: !5)
!29 = !DILocation(line: 22, column: 1, scope: !5)
!30 = !DILocation(line: 29, column: 1, scope: !5)
!31 = !DILocalVariable(scope: !5, file: !3, type: !9, flags: DIFlagArtificial)
!32 = distinct !DISubprogram(name: "__nv_MAIN__F1L22_1", scope: !2, file: !3, line: 22, type: !33, scopeLine: 22, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!33 = !DISubroutineType(types: !34)
!34 = !{null, !9, !24, !24}
!35 = !DILocalVariable(name: "__nv_MAIN__F1L22_1Arg0", arg: 1, scope: !32, file: !3, type: !9)
!36 = !DILocation(line: 0, scope: !32)
!37 = !DILocalVariable(name: "__nv_MAIN__F1L22_1Arg1", arg: 2, scope: !32, file: !3, type: !24)
!38 = !DILocalVariable(name: "__nv_MAIN__F1L22_1Arg2", arg: 3, scope: !32, file: !3, type: !24)
!39 = !DILocalVariable(name: "omp_sched_static", scope: !32, file: !3, type: !9)
!40 = !DILocalVariable(name: "omp_proc_bind_false", scope: !32, file: !3, type: !9)
!41 = !DILocalVariable(name: "omp_proc_bind_true", scope: !32, file: !3, type: !9)
!42 = !DILocalVariable(name: "omp_lock_hint_none", scope: !32, file: !3, type: !9)
!43 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !32, file: !3, type: !9)
!44 = !DILocation(line: 26, column: 1, scope: !32)
!45 = !DILocation(line: 23, column: 1, scope: !32)
!46 = !DILocalVariable(name: "i", scope: !32, file: !3, type: !9)
!47 = !DILocation(line: 24, column: 1, scope: !32)
!48 = !DILocalVariable(scope: !32, file: !3, type: !9, flags: DIFlagArtificial)
!49 = !DILocalVariable(name: "str", scope: !32, file: !3, type: !50)
!50 = !DIBasicType(name: "character", size: 400, align: 8, encoding: DW_ATE_signed)
!51 = !DILocation(line: 25, column: 1, scope: !32)

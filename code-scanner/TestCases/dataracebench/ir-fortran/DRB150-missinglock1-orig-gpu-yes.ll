; ModuleID = '/tmp/DRB150-missinglock1-orig-gpu-yes-fffd5f.ll'
source_filename = "/tmp/DRB150-missinglock1-orig-gpu-yes-fffd5f.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%astruct.dt70 = type <{ i8*, i8* }>
%astruct.dt112 = type <{ [16 x i8] }>
%astruct.dt166 = type <{ [16 x i8], i8*, i8* }>

@.C305_MAIN_ = internal constant i32 25
@.C284_MAIN_ = internal constant i64 0
@.C345_MAIN_ = internal constant i32 6
@.C342_MAIN_ = internal constant [61 x i8] c"micro-benchmarks-fortran/DRB150-missinglock1-orig-gpu-yes.f95"
@.C344_MAIN_ = internal constant i32 33
@.C325_MAIN_ = internal constant i32 10
@.C285_MAIN_ = internal constant i32 1
@.C283_MAIN_ = internal constant i32 0
@.C325___nv_MAIN__F1L21_1 = internal constant i32 10
@.C285___nv_MAIN__F1L21_1 = internal constant i32 1
@.C283___nv_MAIN__F1L21_1 = internal constant i32 0
@.C325___nv_MAIN_F1L22_2 = internal constant i32 10
@.C285___nv_MAIN_F1L22_2 = internal constant i32 1
@.C283___nv_MAIN_F1L22_2 = internal constant i32 0
@.C285___nv_MAIN_F1L23_3 = internal constant i32 1
@.C283___nv_MAIN_F1L23_3 = internal constant i32 0

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__392 = alloca i32, align 4
  %lck_316 = alloca i32, align 4
  %.uplevelArgPack0001_386 = alloca %astruct.dt70, align 16
  %var_314 = alloca i32, align 4
  %z__io_347 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !14, metadata !DIExpression()), !dbg !10
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !15
  store i32 %0, i32* %__gtid_MAIN__392, align 4, !dbg !15
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !16
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !16
  call void (i8*, ...) %2(i8* %1), !dbg !16
  br label %L.LB1_380

L.LB1_380:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %lck_316, metadata !17, metadata !DIExpression()), !dbg !10
  %3 = bitcast i32* %lck_316 to i64*, !dbg !18
  call void @omp_init_lock_(i64* %3), !dbg !18
  %4 = bitcast i32* %lck_316 to i8*, !dbg !19
  %5 = bitcast %astruct.dt70* %.uplevelArgPack0001_386 to i8**, !dbg !19
  store i8* %4, i8** %5, align 8, !dbg !19
  call void @llvm.dbg.declare(metadata i32* %var_314, metadata !20, metadata !DIExpression()), !dbg !10
  %6 = bitcast i32* %var_314 to i8*, !dbg !19
  %7 = bitcast %astruct.dt70* %.uplevelArgPack0001_386 to i8*, !dbg !19
  %8 = getelementptr i8, i8* %7, i64 8, !dbg !19
  %9 = bitcast i8* %8 to i8**, !dbg !19
  store i8* %6, i8** %9, align 8, !dbg !19
  %10 = bitcast %astruct.dt70* %.uplevelArgPack0001_386 to i64*, !dbg !19
  call void @__nv_MAIN__F1L21_1_(i32* %__gtid_MAIN__392, i64* null, i64* %10), !dbg !19
  %11 = bitcast i32* %lck_316 to i64*, !dbg !21
  call void @omp_destroy_lock_(i64* %11), !dbg !21
  call void (...) @_mp_bcs_nest(), !dbg !22
  %12 = bitcast i32* @.C344_MAIN_ to i8*, !dbg !22
  %13 = bitcast [61 x i8]* @.C342_MAIN_ to i8*, !dbg !22
  %14 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !22
  call void (i8*, i8*, i64, ...) %14(i8* %12, i8* %13, i64 61), !dbg !22
  %15 = bitcast i32* @.C345_MAIN_ to i8*, !dbg !22
  %16 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !22
  %17 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !22
  %18 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !22
  %19 = call i32 (i8*, i8*, i8*, i8*, ...) %18(i8* %15, i8* null, i8* %16, i8* %17), !dbg !22
  call void @llvm.dbg.declare(metadata i32* %z__io_347, metadata !23, metadata !DIExpression()), !dbg !10
  store i32 %19, i32* %z__io_347, align 4, !dbg !22
  %20 = load i32, i32* %var_314, align 4, !dbg !22
  call void @llvm.dbg.value(metadata i32 %20, metadata !20, metadata !DIExpression()), !dbg !10
  %21 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !22
  %22 = call i32 (i32, i32, ...) %21(i32 %20, i32 25), !dbg !22
  store i32 %22, i32* %z__io_347, align 4, !dbg !22
  %23 = call i32 (...) @f90io_ldw_end(), !dbg !22
  store i32 %23, i32* %z__io_347, align 4, !dbg !22
  call void (...) @_mp_ecs_nest(), !dbg !22
  ret void, !dbg !15
}

define internal void @__nv_MAIN__F1L21_1_(i32* %__nv_MAIN__F1L21_1Arg0, i64* %__nv_MAIN__F1L21_1Arg1, i64* %__nv_MAIN__F1L21_1Arg2) #0 !dbg !24 {
L.entry:
  %.uplevelArgPack0002_412 = alloca %astruct.dt112, align 16
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L21_1Arg0, metadata !28, metadata !DIExpression()), !dbg !29
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L21_1Arg1, metadata !30, metadata !DIExpression()), !dbg !29
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L21_1Arg2, metadata !31, metadata !DIExpression()), !dbg !29
  call void @llvm.dbg.value(metadata i32 1, metadata !32, metadata !DIExpression()), !dbg !29
  call void @llvm.dbg.value(metadata i32 0, metadata !33, metadata !DIExpression()), !dbg !29
  call void @llvm.dbg.value(metadata i32 1, metadata !34, metadata !DIExpression()), !dbg !29
  call void @llvm.dbg.value(metadata i32 0, metadata !35, metadata !DIExpression()), !dbg !29
  call void @llvm.dbg.value(metadata i32 1, metadata !36, metadata !DIExpression()), !dbg !29
  br label %L.LB2_407

L.LB2_407:                                        ; preds = %L.entry
  br label %L.LB2_319

L.LB2_319:                                        ; preds = %L.LB2_407
  %0 = load i64, i64* %__nv_MAIN__F1L21_1Arg2, align 8, !dbg !37
  %1 = bitcast %astruct.dt112* %.uplevelArgPack0002_412 to i64*, !dbg !37
  store i64 %0, i64* %1, align 8, !dbg !37
  %2 = bitcast i64* %__nv_MAIN__F1L21_1Arg2 to i8*, !dbg !38
  %3 = getelementptr i8, i8* %2, i64 8, !dbg !38
  %4 = bitcast i8* %3 to i64*, !dbg !38
  %5 = load i64, i64* %4, align 8, !dbg !38
  %6 = bitcast %astruct.dt112* %.uplevelArgPack0002_412 to i8*, !dbg !38
  %7 = getelementptr i8, i8* %6, i64 8, !dbg !38
  %8 = bitcast i8* %7 to i64*, !dbg !38
  store i64 %5, i64* %8, align 8, !dbg !38
  %9 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN_F1L22_2_ to i64*, !dbg !37
  %10 = bitcast %astruct.dt112* %.uplevelArgPack0002_412 to i64*, !dbg !37
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_teams(i64* null, i32 1, i64* %9, i64* %10), !dbg !37
  br label %L.LB2_340

L.LB2_340:                                        ; preds = %L.LB2_319
  ret void, !dbg !38
}

define internal void @__nv_MAIN_F1L22_2_(i32* %__nv_MAIN_F1L22_2Arg0, i64* %__nv_MAIN_F1L22_2Arg1, i64* %__nv_MAIN_F1L22_2Arg2) #0 !dbg !39 {
L.entry:
  %__gtid___nv_MAIN_F1L22_2__449 = alloca i32, align 4
  %.i0000p_327 = alloca i32, align 4
  %.i0001p_328 = alloca i32, align 4
  %.i0002p_329 = alloca i32, align 4
  %.i0003p_330 = alloca i32, align 4
  %i_326 = alloca i32, align 4
  %.du0001_358 = alloca i32, align 4
  %.de0001_359 = alloca i32, align 4
  %.di0001_360 = alloca i32, align 4
  %.ds0001_361 = alloca i32, align 4
  %.dl0001_363 = alloca i32, align 4
  %.dl0001.copy_443 = alloca i32, align 4
  %.de0001.copy_444 = alloca i32, align 4
  %.ds0001.copy_445 = alloca i32, align 4
  %.dX0001_362 = alloca i32, align 4
  %.dY0001_357 = alloca i32, align 4
  %.uplevelArgPack0003_468 = alloca %astruct.dt166, align 16
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L22_2Arg0, metadata !40, metadata !DIExpression()), !dbg !41
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L22_2Arg1, metadata !42, metadata !DIExpression()), !dbg !41
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L22_2Arg2, metadata !43, metadata !DIExpression()), !dbg !41
  call void @llvm.dbg.value(metadata i32 1, metadata !44, metadata !DIExpression()), !dbg !41
  call void @llvm.dbg.value(metadata i32 0, metadata !45, metadata !DIExpression()), !dbg !41
  call void @llvm.dbg.value(metadata i32 1, metadata !46, metadata !DIExpression()), !dbg !41
  call void @llvm.dbg.value(metadata i32 0, metadata !47, metadata !DIExpression()), !dbg !41
  call void @llvm.dbg.value(metadata i32 1, metadata !48, metadata !DIExpression()), !dbg !41
  %0 = load i32, i32* %__nv_MAIN_F1L22_2Arg0, align 4, !dbg !49
  store i32 %0, i32* %__gtid___nv_MAIN_F1L22_2__449, align 4, !dbg !49
  br label %L.LB4_432

L.LB4_432:                                        ; preds = %L.entry
  br label %L.LB4_322

L.LB4_322:                                        ; preds = %L.LB4_432
  br label %L.LB4_323

L.LB4_323:                                        ; preds = %L.LB4_322
  br label %L.LB4_324

L.LB4_324:                                        ; preds = %L.LB4_323
  store i32 0, i32* %.i0000p_327, align 4, !dbg !50
  store i32 1, i32* %.i0001p_328, align 4, !dbg !50
  store i32 10, i32* %.i0002p_329, align 4, !dbg !50
  store i32 1, i32* %.i0003p_330, align 4, !dbg !50
  %1 = load i32, i32* %.i0001p_328, align 4, !dbg !50
  call void @llvm.dbg.declare(metadata i32* %i_326, metadata !51, metadata !DIExpression()), !dbg !49
  store i32 %1, i32* %i_326, align 4, !dbg !50
  %2 = load i32, i32* %.i0002p_329, align 4, !dbg !50
  store i32 %2, i32* %.du0001_358, align 4, !dbg !50
  %3 = load i32, i32* %.i0002p_329, align 4, !dbg !50
  store i32 %3, i32* %.de0001_359, align 4, !dbg !50
  store i32 1, i32* %.di0001_360, align 4, !dbg !50
  %4 = load i32, i32* %.di0001_360, align 4, !dbg !50
  store i32 %4, i32* %.ds0001_361, align 4, !dbg !50
  %5 = load i32, i32* %.i0001p_328, align 4, !dbg !50
  store i32 %5, i32* %.dl0001_363, align 4, !dbg !50
  %6 = load i32, i32* %.dl0001_363, align 4, !dbg !50
  store i32 %6, i32* %.dl0001.copy_443, align 4, !dbg !50
  %7 = load i32, i32* %.de0001_359, align 4, !dbg !50
  store i32 %7, i32* %.de0001.copy_444, align 4, !dbg !50
  %8 = load i32, i32* %.ds0001_361, align 4, !dbg !50
  store i32 %8, i32* %.ds0001.copy_445, align 4, !dbg !50
  %9 = load i32, i32* %__gtid___nv_MAIN_F1L22_2__449, align 4, !dbg !50
  %10 = bitcast i32* %.i0000p_327 to i64*, !dbg !50
  %11 = bitcast i32* %.dl0001.copy_443 to i64*, !dbg !50
  %12 = bitcast i32* %.de0001.copy_444 to i64*, !dbg !50
  %13 = bitcast i32* %.ds0001.copy_445 to i64*, !dbg !50
  %14 = load i32, i32* %.ds0001.copy_445, align 4, !dbg !50
  call void @__kmpc_for_static_init_4(i64* null, i32 %9, i32 92, i64* %10, i64* %11, i64* %12, i64* %13, i32 %14, i32 1), !dbg !50
  %15 = load i32, i32* %.dl0001.copy_443, align 4, !dbg !50
  store i32 %15, i32* %.dl0001_363, align 4, !dbg !50
  %16 = load i32, i32* %.de0001.copy_444, align 4, !dbg !50
  store i32 %16, i32* %.de0001_359, align 4, !dbg !50
  %17 = load i32, i32* %.ds0001.copy_445, align 4, !dbg !50
  store i32 %17, i32* %.ds0001_361, align 4, !dbg !50
  %18 = load i32, i32* %.dl0001_363, align 4, !dbg !50
  store i32 %18, i32* %i_326, align 4, !dbg !50
  %19 = load i32, i32* %i_326, align 4, !dbg !50
  call void @llvm.dbg.value(metadata i32 %19, metadata !51, metadata !DIExpression()), !dbg !49
  store i32 %19, i32* %.dX0001_362, align 4, !dbg !50
  %20 = load i32, i32* %.dX0001_362, align 4, !dbg !50
  %21 = load i32, i32* %.du0001_358, align 4, !dbg !50
  %22 = icmp sgt i32 %20, %21, !dbg !50
  br i1 %22, label %L.LB4_356, label %L.LB4_500, !dbg !50

L.LB4_500:                                        ; preds = %L.LB4_324
  %23 = load i32, i32* %.du0001_358, align 4, !dbg !50
  %24 = load i32, i32* %.de0001_359, align 4, !dbg !50
  %25 = icmp slt i32 %23, %24, !dbg !50
  %26 = select i1 %25, i32 %23, i32 %24, !dbg !50
  store i32 %26, i32* %.de0001_359, align 4, !dbg !50
  %27 = load i32, i32* %.dX0001_362, align 4, !dbg !50
  store i32 %27, i32* %i_326, align 4, !dbg !50
  %28 = load i32, i32* %.di0001_360, align 4, !dbg !50
  %29 = load i32, i32* %.de0001_359, align 4, !dbg !50
  %30 = load i32, i32* %.dX0001_362, align 4, !dbg !50
  %31 = sub nsw i32 %29, %30, !dbg !50
  %32 = add nsw i32 %28, %31, !dbg !50
  %33 = load i32, i32* %.di0001_360, align 4, !dbg !50
  %34 = sdiv i32 %32, %33, !dbg !50
  store i32 %34, i32* %.dY0001_357, align 4, !dbg !50
  %35 = load i32, i32* %i_326, align 4, !dbg !50
  call void @llvm.dbg.value(metadata i32 %35, metadata !51, metadata !DIExpression()), !dbg !49
  store i32 %35, i32* %.i0001p_328, align 4, !dbg !50
  %36 = load i32, i32* %.de0001_359, align 4, !dbg !50
  store i32 %36, i32* %.i0002p_329, align 4, !dbg !50
  %37 = load i64, i64* %__nv_MAIN_F1L22_2Arg2, align 8, !dbg !50
  %38 = bitcast %astruct.dt166* %.uplevelArgPack0003_468 to i64*, !dbg !50
  store i64 %37, i64* %38, align 8, !dbg !50
  %39 = bitcast i64* %__nv_MAIN_F1L22_2Arg2 to i8*, !dbg !49
  %40 = getelementptr i8, i8* %39, i64 8, !dbg !49
  %41 = bitcast i8* %40 to i64*, !dbg !49
  %42 = load i64, i64* %41, align 8, !dbg !49
  %43 = bitcast %astruct.dt166* %.uplevelArgPack0003_468 to i8*, !dbg !49
  %44 = getelementptr i8, i8* %43, i64 8, !dbg !49
  %45 = bitcast i8* %44 to i64*, !dbg !49
  store i64 %42, i64* %45, align 8, !dbg !49
  %46 = bitcast i32* %.i0001p_328 to i8*, !dbg !50
  %47 = bitcast %astruct.dt166* %.uplevelArgPack0003_468 to i8*, !dbg !50
  %48 = getelementptr i8, i8* %47, i64 16, !dbg !50
  %49 = bitcast i8* %48 to i8**, !dbg !50
  store i8* %46, i8** %49, align 8, !dbg !50
  %50 = bitcast i32* %.i0002p_329 to i8*, !dbg !50
  %51 = bitcast %astruct.dt166* %.uplevelArgPack0003_468 to i8*, !dbg !50
  %52 = getelementptr i8, i8* %51, i64 24, !dbg !50
  %53 = bitcast i8* %52 to i8**, !dbg !50
  store i8* %50, i8** %53, align 8, !dbg !50
  br label %L.LB4_475, !dbg !50

L.LB4_475:                                        ; preds = %L.LB4_500
  %54 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN_F1L23_3_ to i64*, !dbg !50
  %55 = bitcast %astruct.dt166* %.uplevelArgPack0003_468 to i64*, !dbg !50
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %54, i64* %55), !dbg !50
  br label %L.LB4_356

L.LB4_356:                                        ; preds = %L.LB4_475, %L.LB4_324
  %56 = load i32, i32* %__gtid___nv_MAIN_F1L22_2__449, align 4, !dbg !52
  call void @__kmpc_for_static_fini(i64* null, i32 %56), !dbg !52
  br label %L.LB4_337

L.LB4_337:                                        ; preds = %L.LB4_356
  br label %L.LB4_338

L.LB4_338:                                        ; preds = %L.LB4_337
  br label %L.LB4_339

L.LB4_339:                                        ; preds = %L.LB4_338
  ret void, !dbg !49
}

define internal void @__nv_MAIN_F1L23_3_(i32* %__nv_MAIN_F1L23_3Arg0, i64* %__nv_MAIN_F1L23_3Arg1, i64* %__nv_MAIN_F1L23_3Arg2) #0 !dbg !53 {
L.entry:
  %__gtid___nv_MAIN_F1L23_3__521 = alloca i32, align 4
  %.i0004p_335 = alloca i32, align 4
  %i_334 = alloca i32, align 4
  %.du0002p_370 = alloca i32, align 4
  %.de0002p_371 = alloca i32, align 4
  %.di0002p_372 = alloca i32, align 4
  %.ds0002p_373 = alloca i32, align 4
  %.dl0002p_375 = alloca i32, align 4
  %.dl0002p.copy_515 = alloca i32, align 4
  %.de0002p.copy_516 = alloca i32, align 4
  %.ds0002p.copy_517 = alloca i32, align 4
  %.dX0002p_374 = alloca i32, align 4
  %.dY0002p_369 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L23_3Arg0, metadata !54, metadata !DIExpression()), !dbg !55
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L23_3Arg1, metadata !56, metadata !DIExpression()), !dbg !55
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L23_3Arg2, metadata !57, metadata !DIExpression()), !dbg !55
  call void @llvm.dbg.value(metadata i32 1, metadata !58, metadata !DIExpression()), !dbg !55
  call void @llvm.dbg.value(metadata i32 0, metadata !59, metadata !DIExpression()), !dbg !55
  call void @llvm.dbg.value(metadata i32 1, metadata !60, metadata !DIExpression()), !dbg !55
  call void @llvm.dbg.value(metadata i32 0, metadata !61, metadata !DIExpression()), !dbg !55
  call void @llvm.dbg.value(metadata i32 1, metadata !62, metadata !DIExpression()), !dbg !55
  %0 = load i32, i32* %__nv_MAIN_F1L23_3Arg0, align 4, !dbg !63
  store i32 %0, i32* %__gtid___nv_MAIN_F1L23_3__521, align 4, !dbg !63
  br label %L.LB6_504

L.LB6_504:                                        ; preds = %L.entry
  br label %L.LB6_333

L.LB6_333:                                        ; preds = %L.LB6_504
  store i32 0, i32* %.i0004p_335, align 4, !dbg !64
  %1 = bitcast i64* %__nv_MAIN_F1L23_3Arg2 to i8*, !dbg !64
  %2 = getelementptr i8, i8* %1, i64 16, !dbg !64
  %3 = bitcast i8* %2 to i32**, !dbg !64
  %4 = load i32*, i32** %3, align 8, !dbg !64
  %5 = load i32, i32* %4, align 4, !dbg !64
  call void @llvm.dbg.declare(metadata i32* %i_334, metadata !65, metadata !DIExpression()), !dbg !63
  store i32 %5, i32* %i_334, align 4, !dbg !64
  %6 = bitcast i64* %__nv_MAIN_F1L23_3Arg2 to i8*, !dbg !64
  %7 = getelementptr i8, i8* %6, i64 24, !dbg !64
  %8 = bitcast i8* %7 to i32**, !dbg !64
  %9 = load i32*, i32** %8, align 8, !dbg !64
  %10 = load i32, i32* %9, align 4, !dbg !64
  store i32 %10, i32* %.du0002p_370, align 4, !dbg !64
  %11 = bitcast i64* %__nv_MAIN_F1L23_3Arg2 to i8*, !dbg !64
  %12 = getelementptr i8, i8* %11, i64 24, !dbg !64
  %13 = bitcast i8* %12 to i32**, !dbg !64
  %14 = load i32*, i32** %13, align 8, !dbg !64
  %15 = load i32, i32* %14, align 4, !dbg !64
  store i32 %15, i32* %.de0002p_371, align 4, !dbg !64
  store i32 1, i32* %.di0002p_372, align 4, !dbg !64
  %16 = load i32, i32* %.di0002p_372, align 4, !dbg !64
  store i32 %16, i32* %.ds0002p_373, align 4, !dbg !64
  %17 = bitcast i64* %__nv_MAIN_F1L23_3Arg2 to i8*, !dbg !64
  %18 = getelementptr i8, i8* %17, i64 16, !dbg !64
  %19 = bitcast i8* %18 to i32**, !dbg !64
  %20 = load i32*, i32** %19, align 8, !dbg !64
  %21 = load i32, i32* %20, align 4, !dbg !64
  store i32 %21, i32* %.dl0002p_375, align 4, !dbg !64
  %22 = load i32, i32* %.dl0002p_375, align 4, !dbg !64
  store i32 %22, i32* %.dl0002p.copy_515, align 4, !dbg !64
  %23 = load i32, i32* %.de0002p_371, align 4, !dbg !64
  store i32 %23, i32* %.de0002p.copy_516, align 4, !dbg !64
  %24 = load i32, i32* %.ds0002p_373, align 4, !dbg !64
  store i32 %24, i32* %.ds0002p.copy_517, align 4, !dbg !64
  %25 = load i32, i32* %__gtid___nv_MAIN_F1L23_3__521, align 4, !dbg !64
  %26 = bitcast i32* %.i0004p_335 to i64*, !dbg !64
  %27 = bitcast i32* %.dl0002p.copy_515 to i64*, !dbg !64
  %28 = bitcast i32* %.de0002p.copy_516 to i64*, !dbg !64
  %29 = bitcast i32* %.ds0002p.copy_517 to i64*, !dbg !64
  %30 = load i32, i32* %.ds0002p.copy_517, align 4, !dbg !64
  call void @__kmpc_for_static_init_4(i64* null, i32 %25, i32 34, i64* %26, i64* %27, i64* %28, i64* %29, i32 %30, i32 1), !dbg !64
  %31 = load i32, i32* %.dl0002p.copy_515, align 4, !dbg !64
  store i32 %31, i32* %.dl0002p_375, align 4, !dbg !64
  %32 = load i32, i32* %.de0002p.copy_516, align 4, !dbg !64
  store i32 %32, i32* %.de0002p_371, align 4, !dbg !64
  %33 = load i32, i32* %.ds0002p.copy_517, align 4, !dbg !64
  store i32 %33, i32* %.ds0002p_373, align 4, !dbg !64
  %34 = load i32, i32* %.dl0002p_375, align 4, !dbg !64
  store i32 %34, i32* %i_334, align 4, !dbg !64
  %35 = load i32, i32* %i_334, align 4, !dbg !64
  call void @llvm.dbg.value(metadata i32 %35, metadata !65, metadata !DIExpression()), !dbg !63
  store i32 %35, i32* %.dX0002p_374, align 4, !dbg !64
  %36 = load i32, i32* %.dX0002p_374, align 4, !dbg !64
  %37 = load i32, i32* %.du0002p_370, align 4, !dbg !64
  %38 = icmp sgt i32 %36, %37, !dbg !64
  br i1 %38, label %L.LB6_368, label %L.LB6_530, !dbg !64

L.LB6_530:                                        ; preds = %L.LB6_333
  %39 = load i32, i32* %.dX0002p_374, align 4, !dbg !64
  store i32 %39, i32* %i_334, align 4, !dbg !64
  %40 = load i32, i32* %.di0002p_372, align 4, !dbg !64
  %41 = load i32, i32* %.de0002p_371, align 4, !dbg !64
  %42 = load i32, i32* %.dX0002p_374, align 4, !dbg !64
  %43 = sub nsw i32 %41, %42, !dbg !64
  %44 = add nsw i32 %40, %43, !dbg !64
  %45 = load i32, i32* %.di0002p_372, align 4, !dbg !64
  %46 = sdiv i32 %44, %45, !dbg !64
  store i32 %46, i32* %.dY0002p_369, align 4, !dbg !64
  %47 = load i32, i32* %.dY0002p_369, align 4, !dbg !64
  %48 = icmp sle i32 %47, 0, !dbg !64
  br i1 %48, label %L.LB6_378, label %L.LB6_377, !dbg !64

L.LB6_377:                                        ; preds = %L.LB6_377, %L.LB6_530
  %49 = bitcast i64* %__nv_MAIN_F1L23_3Arg2 to i64**, !dbg !66
  %50 = load i64*, i64** %49, align 8, !dbg !66
  call void @omp_set_lock_(i64* %50), !dbg !66
  %51 = bitcast i64* %__nv_MAIN_F1L23_3Arg2 to i8*, !dbg !67
  %52 = getelementptr i8, i8* %51, i64 8, !dbg !67
  %53 = bitcast i8* %52 to i32**, !dbg !67
  %54 = load i32*, i32** %53, align 8, !dbg !67
  %55 = load i32, i32* %54, align 4, !dbg !67
  %56 = add nsw i32 %55, 1, !dbg !67
  %57 = bitcast i64* %__nv_MAIN_F1L23_3Arg2 to i8*, !dbg !67
  %58 = getelementptr i8, i8* %57, i64 8, !dbg !67
  %59 = bitcast i8* %58 to i32**, !dbg !67
  %60 = load i32*, i32** %59, align 8, !dbg !67
  store i32 %56, i32* %60, align 4, !dbg !67
  %61 = bitcast i64* %__nv_MAIN_F1L23_3Arg2 to i64**, !dbg !68
  %62 = load i64*, i64** %61, align 8, !dbg !68
  call void @omp_unset_lock_(i64* %62), !dbg !68
  %63 = load i32, i32* %.di0002p_372, align 4, !dbg !63
  %64 = load i32, i32* %i_334, align 4, !dbg !63
  call void @llvm.dbg.value(metadata i32 %64, metadata !65, metadata !DIExpression()), !dbg !63
  %65 = add nsw i32 %63, %64, !dbg !63
  store i32 %65, i32* %i_334, align 4, !dbg !63
  %66 = load i32, i32* %.dY0002p_369, align 4, !dbg !63
  %67 = sub nsw i32 %66, 1, !dbg !63
  store i32 %67, i32* %.dY0002p_369, align 4, !dbg !63
  %68 = load i32, i32* %.dY0002p_369, align 4, !dbg !63
  %69 = icmp sgt i32 %68, 0, !dbg !63
  br i1 %69, label %L.LB6_377, label %L.LB6_378, !dbg !63

L.LB6_378:                                        ; preds = %L.LB6_377, %L.LB6_530
  br label %L.LB6_368

L.LB6_368:                                        ; preds = %L.LB6_378, %L.LB6_333
  %70 = load i32, i32* %__gtid___nv_MAIN_F1L23_3__521, align 4, !dbg !63
  call void @__kmpc_for_static_fini(i64* null, i32 %70), !dbg !63
  br label %L.LB6_336

L.LB6_336:                                        ; preds = %L.LB6_368
  ret void, !dbg !63
}

declare void @__kmpc_for_static_fini(i64*, i32) #0

declare void @__kmpc_fork_call(i64*, i32, i64*, i64*, ...) #0

declare void @__kmpc_end_serialized_parallel(i64*, i32) #0

declare void @__kmpc_serialized_parallel(i64*, i32) #0

declare void @__kmpc_for_static_init_4(i64*, i32, i32, i64*, i64*, i64*, i64*, i32, i32) #0

declare void @__kmpc_fork_teams(i64*, i32, i64*, i64*, ...) #0

declare void @_mp_ecs_nest(...) #0

declare signext i32 @f90io_ldw_end(...) #0

declare signext i32 @f90io_sc_i_ldw(...) #0

declare signext i32 @f90io_print_init(...) #0

declare void @f90io_src_info03a(...) #0

declare void @_mp_bcs_nest(...) #0

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare void @fort_init(...) #0

declare signext i32 @__kmpc_global_thread_num(i64*) #0

declare void @omp_unset_lock_(i64*) #0

declare void @omp_set_lock_(i64*) #0

declare void @omp_init_lock_(i64*) #0

declare void @omp_destroy_lock_(i64*) #0

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { "no-frame-pointer-elim-non-leaf" }
attributes #1 = { nounwind readnone speculatable }

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !4, globals: !4, imports: !4)
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB150-missinglock1-orig-gpu-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb150_missinglock1_orig_gpu_yes", scope: !2, file: !3, line: 13, type: !6, scopeLine: 13, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!6 = !DISubroutineType(cc: DW_CC_program, types: !7)
!7 = !{null}
!8 = !DILocalVariable(name: "omp_sched_static", scope: !5, file: !3, type: !9)
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 0, scope: !5)
!11 = !DILocalVariable(name: "omp_proc_bind_false", scope: !5, file: !3, type: !9)
!12 = !DILocalVariable(name: "omp_proc_bind_true", scope: !5, file: !3, type: !9)
!13 = !DILocalVariable(name: "omp_lock_hint_none", scope: !5, file: !3, type: !9)
!14 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !5, file: !3, type: !9)
!15 = !DILocation(line: 34, column: 1, scope: !5)
!16 = !DILocation(line: 13, column: 1, scope: !5)
!17 = !DILocalVariable(name: "lck", scope: !5, file: !3, type: !9)
!18 = !DILocation(line: 19, column: 1, scope: !5)
!19 = !DILocation(line: 29, column: 1, scope: !5)
!20 = !DILocalVariable(name: "var", scope: !5, file: !3, type: !9)
!21 = !DILocation(line: 31, column: 1, scope: !5)
!22 = !DILocation(line: 33, column: 1, scope: !5)
!23 = !DILocalVariable(scope: !5, file: !3, type: !9, flags: DIFlagArtificial)
!24 = distinct !DISubprogram(name: "__nv_MAIN__F1L21_1", scope: !2, file: !3, line: 21, type: !25, scopeLine: 21, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!25 = !DISubroutineType(types: !26)
!26 = !{null, !9, !27, !27}
!27 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!28 = !DILocalVariable(name: "__nv_MAIN__F1L21_1Arg0", arg: 1, scope: !24, file: !3, type: !9)
!29 = !DILocation(line: 0, scope: !24)
!30 = !DILocalVariable(name: "__nv_MAIN__F1L21_1Arg1", arg: 2, scope: !24, file: !3, type: !27)
!31 = !DILocalVariable(name: "__nv_MAIN__F1L21_1Arg2", arg: 3, scope: !24, file: !3, type: !27)
!32 = !DILocalVariable(name: "omp_sched_static", scope: !24, file: !3, type: !9)
!33 = !DILocalVariable(name: "omp_proc_bind_false", scope: !24, file: !3, type: !9)
!34 = !DILocalVariable(name: "omp_proc_bind_true", scope: !24, file: !3, type: !9)
!35 = !DILocalVariable(name: "omp_lock_hint_none", scope: !24, file: !3, type: !9)
!36 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !24, file: !3, type: !9)
!37 = !DILocation(line: 22, column: 1, scope: !24)
!38 = !DILocation(line: 29, column: 1, scope: !24)
!39 = distinct !DISubprogram(name: "__nv_MAIN_F1L22_2", scope: !2, file: !3, line: 22, type: !25, scopeLine: 22, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!40 = !DILocalVariable(name: "__nv_MAIN_F1L22_2Arg0", arg: 1, scope: !39, file: !3, type: !9)
!41 = !DILocation(line: 0, scope: !39)
!42 = !DILocalVariable(name: "__nv_MAIN_F1L22_2Arg1", arg: 2, scope: !39, file: !3, type: !27)
!43 = !DILocalVariable(name: "__nv_MAIN_F1L22_2Arg2", arg: 3, scope: !39, file: !3, type: !27)
!44 = !DILocalVariable(name: "omp_sched_static", scope: !39, file: !3, type: !9)
!45 = !DILocalVariable(name: "omp_proc_bind_false", scope: !39, file: !3, type: !9)
!46 = !DILocalVariable(name: "omp_proc_bind_true", scope: !39, file: !3, type: !9)
!47 = !DILocalVariable(name: "omp_lock_hint_none", scope: !39, file: !3, type: !9)
!48 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !39, file: !3, type: !9)
!49 = !DILocation(line: 28, column: 1, scope: !39)
!50 = !DILocation(line: 23, column: 1, scope: !39)
!51 = !DILocalVariable(name: "i", scope: !39, file: !3, type: !9)
!52 = !DILocation(line: 27, column: 1, scope: !39)
!53 = distinct !DISubprogram(name: "__nv_MAIN_F1L23_3", scope: !2, file: !3, line: 23, type: !25, scopeLine: 23, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!54 = !DILocalVariable(name: "__nv_MAIN_F1L23_3Arg0", arg: 1, scope: !53, file: !3, type: !9)
!55 = !DILocation(line: 0, scope: !53)
!56 = !DILocalVariable(name: "__nv_MAIN_F1L23_3Arg1", arg: 2, scope: !53, file: !3, type: !27)
!57 = !DILocalVariable(name: "__nv_MAIN_F1L23_3Arg2", arg: 3, scope: !53, file: !3, type: !27)
!58 = !DILocalVariable(name: "omp_sched_static", scope: !53, file: !3, type: !9)
!59 = !DILocalVariable(name: "omp_proc_bind_false", scope: !53, file: !3, type: !9)
!60 = !DILocalVariable(name: "omp_proc_bind_true", scope: !53, file: !3, type: !9)
!61 = !DILocalVariable(name: "omp_lock_hint_none", scope: !53, file: !3, type: !9)
!62 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !53, file: !3, type: !9)
!63 = !DILocation(line: 27, column: 1, scope: !53)
!64 = !DILocation(line: 23, column: 1, scope: !53)
!65 = !DILocalVariable(name: "i", scope: !53, file: !3, type: !9)
!66 = !DILocation(line: 24, column: 1, scope: !53)
!67 = !DILocation(line: 25, column: 1, scope: !53)
!68 = !DILocation(line: 26, column: 1, scope: !53)

; ModuleID = '/tmp/DRB099-targetparallelfor2-orig-no-ec7ee6.ll'
source_filename = "/tmp/DRB099-targetparallelfor2-orig-no-ec7ee6.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%astruct.dt72 = type <{ i8*, i8*, i8*, i8*, i8*, i8*, i8* }>
%astruct.dt105 = type <{ [56 x i8] }>

@.C306_drb099_foo_ = internal constant i32 8
@.C283_drb099_foo_ = internal constant i32 0
@.C285_drb099_foo_ = internal constant i32 1
@.C306___nv_drb099_foo__F1L21_1 = internal constant i32 8
@.C285___nv_drb099_foo__F1L21_1 = internal constant i32 1
@.C283___nv_drb099_foo__F1L21_1 = internal constant i32 0
@.C306___nv_drb099_F1L22_2 = internal constant i32 8
@.C285___nv_drb099_F1L22_2 = internal constant i32 1
@.C283___nv_drb099_F1L22_2 = internal constant i32 0
@.C349_MAIN_ = internal constant i64 50
@.C309_MAIN_ = internal constant i32 14
@.C348_MAIN_ = internal constant [7 x i8] c"b(50) ="
@.C345_MAIN_ = internal constant i32 6
@.C342_MAIN_ = internal constant [62 x i8] c"micro-benchmarks-fortran/DRB099-targetparallelfor2-orig-no.f95"
@.C344_MAIN_ = internal constant i32 54
@.C313_MAIN_ = internal constant i64 12
@.C312_MAIN_ = internal constant i64 11
@.C291_MAIN_ = internal constant double 0.000000e+00
@.C293_MAIN_ = internal constant double 2.000000e+00
@.C301_MAIN_ = internal constant i32 8
@.C285_MAIN_ = internal constant i32 1
@.C310_MAIN_ = internal constant i32 28
@.C314_MAIN_ = internal constant i64 8
@.C357_MAIN_ = internal constant i64 28
@.C337_MAIN_ = internal constant i32 1000
@.C286_MAIN_ = internal constant i64 1
@.C284_MAIN_ = internal constant i64 0
@.C283_MAIN_ = internal constant i32 0

; Function Attrs: noinline
define float @drb099_() #0 {
.L.entry:
  ret float undef
}

define float @drb099_foo_(i64* %"a$p", i64* %"b$p", i64* %n, i64* %"a$sd1", i64* %"b$sd3") #1 !dbg !5 {
L.entry:
  %__gtid_drb099_foo__396 = alloca i32, align 4
  %.uplevelArgPack0001_376 = alloca %astruct.dt72, align 16
  %foo_303 = alloca float, align 4
  call void @llvm.dbg.declare(metadata i64* %"a$p", metadata !20, metadata !DIExpression(DW_OP_deref)), !dbg !21
  call void @llvm.dbg.declare(metadata i64* %"b$p", metadata !22, metadata !DIExpression(DW_OP_deref)), !dbg !21
  call void @llvm.dbg.declare(metadata i64* %n, metadata !23, metadata !DIExpression()), !dbg !21
  call void @llvm.dbg.declare(metadata i64* %"a$sd1", metadata !24, metadata !DIExpression()), !dbg !21
  call void @llvm.dbg.declare(metadata i64* %"b$sd3", metadata !25, metadata !DIExpression()), !dbg !21
  call void @llvm.dbg.value(metadata i32 8, metadata !26, metadata !DIExpression()), !dbg !21
  call void @llvm.dbg.value(metadata i32 1, metadata !27, metadata !DIExpression()), !dbg !21
  call void @llvm.dbg.value(metadata i32 0, metadata !28, metadata !DIExpression()), !dbg !21
  call void @llvm.dbg.value(metadata i32 1, metadata !29, metadata !DIExpression()), !dbg !21
  call void @llvm.dbg.value(metadata i32 0, metadata !30, metadata !DIExpression()), !dbg !21
  call void @llvm.dbg.value(metadata i32 1, metadata !31, metadata !DIExpression()), !dbg !21
  call void @llvm.dbg.value(metadata i32 8, metadata !32, metadata !DIExpression()), !dbg !21
  call void @llvm.dbg.value(metadata i32 8, metadata !33, metadata !DIExpression()), !dbg !21
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !34
  store i32 %0, i32* %__gtid_drb099_foo__396, align 4, !dbg !34
  br label %L.LB2_371

L.LB2_371:                                        ; preds = %L.entry
  %1 = bitcast i64* %n to i8*, !dbg !35
  %2 = bitcast %astruct.dt72* %.uplevelArgPack0001_376 to i8**, !dbg !35
  store i8* %1, i8** %2, align 8, !dbg !35
  %3 = bitcast i64* %"b$p" to i8*, !dbg !35
  %4 = bitcast %astruct.dt72* %.uplevelArgPack0001_376 to i8*, !dbg !35
  %5 = getelementptr i8, i8* %4, i64 8, !dbg !35
  %6 = bitcast i8* %5 to i8**, !dbg !35
  store i8* %3, i8** %6, align 8, !dbg !35
  %7 = bitcast i64* %"b$sd3" to i8*, !dbg !35
  %8 = bitcast %astruct.dt72* %.uplevelArgPack0001_376 to i8*, !dbg !35
  %9 = getelementptr i8, i8* %8, i64 16, !dbg !35
  %10 = bitcast i8* %9 to i8**, !dbg !35
  store i8* %7, i8** %10, align 8, !dbg !35
  %11 = bitcast i64* %"b$p" to i8*, !dbg !35
  %12 = bitcast %astruct.dt72* %.uplevelArgPack0001_376 to i8*, !dbg !35
  %13 = getelementptr i8, i8* %12, i64 24, !dbg !35
  %14 = bitcast i8* %13 to i8**, !dbg !35
  store i8* %11, i8** %14, align 8, !dbg !35
  %15 = bitcast i64* %"a$p" to i8*, !dbg !35
  %16 = bitcast %astruct.dt72* %.uplevelArgPack0001_376 to i8*, !dbg !35
  %17 = getelementptr i8, i8* %16, i64 32, !dbg !35
  %18 = bitcast i8* %17 to i8**, !dbg !35
  store i8* %15, i8** %18, align 8, !dbg !35
  %19 = bitcast i64* %"a$sd1" to i8*, !dbg !35
  %20 = bitcast %astruct.dt72* %.uplevelArgPack0001_376 to i8*, !dbg !35
  %21 = getelementptr i8, i8* %20, i64 40, !dbg !35
  %22 = bitcast i8* %21 to i8**, !dbg !35
  store i8* %19, i8** %22, align 8, !dbg !35
  %23 = bitcast i64* %"a$p" to i8*, !dbg !35
  %24 = bitcast %astruct.dt72* %.uplevelArgPack0001_376 to i8*, !dbg !35
  %25 = getelementptr i8, i8* %24, i64 48, !dbg !35
  %26 = bitcast i8* %25 to i8**, !dbg !35
  store i8* %23, i8** %26, align 8, !dbg !35
  %27 = bitcast %astruct.dt72* %.uplevelArgPack0001_376 to i64*, !dbg !35
  call void @__nv_drb099_foo__F1L21_1_(i32* %__gtid_drb099_foo__396, i64* null, i64* %27), !dbg !35
  br label %L.LB2_332, !dbg !36

L.LB2_332:                                        ; preds = %L.LB2_371
  call void @llvm.dbg.declare(metadata float* %foo_303, metadata !37, metadata !DIExpression()), !dbg !21
  %28 = load float, float* %foo_303, align 4, !dbg !34
  call void @llvm.dbg.value(metadata float %28, metadata !37, metadata !DIExpression()), !dbg !21
  ret float %28, !dbg !34
}

define internal void @__nv_drb099_foo__F1L21_1_(i32* %__nv_drb099_foo__F1L21_1Arg0, i64* %__nv_drb099_foo__F1L21_1Arg1, i64* %__nv_drb099_foo__F1L21_1Arg2) #1 !dbg !38 {
L.entry:
  %__gtid___nv_drb099_foo__F1L21_1__416 = alloca i32, align 4
  %.uplevelArgPack0002_411 = alloca %astruct.dt105, align 16
  call void @llvm.dbg.declare(metadata i32* %__nv_drb099_foo__F1L21_1Arg0, metadata !41, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.declare(metadata i64* %__nv_drb099_foo__F1L21_1Arg1, metadata !43, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.declare(metadata i64* %__nv_drb099_foo__F1L21_1Arg2, metadata !44, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.value(metadata i32 8, metadata !45, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.value(metadata i32 1, metadata !46, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.value(metadata i32 0, metadata !47, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.value(metadata i32 1, metadata !48, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.value(metadata i32 0, metadata !49, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.value(metadata i32 1, metadata !50, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.value(metadata i32 8, metadata !51, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.value(metadata i32 8, metadata !52, metadata !DIExpression()), !dbg !42
  %0 = load i32, i32* %__nv_drb099_foo__F1L21_1Arg0, align 4, !dbg !53
  store i32 %0, i32* %__gtid___nv_drb099_foo__F1L21_1__416, align 4, !dbg !53
  br label %L.LB3_406

L.LB3_406:                                        ; preds = %L.entry
  br label %L.LB3_324

L.LB3_324:                                        ; preds = %L.LB3_406
  %1 = load i64, i64* %__nv_drb099_foo__F1L21_1Arg2, align 8, !dbg !54
  %2 = bitcast %astruct.dt105* %.uplevelArgPack0002_411 to i64*, !dbg !54
  store i64 %1, i64* %2, align 8, !dbg !54
  %3 = bitcast i64* %__nv_drb099_foo__F1L21_1Arg2 to i8*, !dbg !53
  %4 = getelementptr i8, i8* %3, i64 8, !dbg !53
  %5 = bitcast i8* %4 to i64*, !dbg !53
  %6 = load i64, i64* %5, align 8, !dbg !53
  %7 = bitcast %astruct.dt105* %.uplevelArgPack0002_411 to i8*, !dbg !53
  %8 = getelementptr i8, i8* %7, i64 8, !dbg !53
  %9 = bitcast i8* %8 to i64*, !dbg !53
  store i64 %6, i64* %9, align 8, !dbg !53
  %10 = bitcast i64* %__nv_drb099_foo__F1L21_1Arg2 to i8*, !dbg !53
  %11 = getelementptr i8, i8* %10, i64 16, !dbg !53
  %12 = bitcast i8* %11 to i64*, !dbg !53
  %13 = load i64, i64* %12, align 8, !dbg !53
  %14 = bitcast %astruct.dt105* %.uplevelArgPack0002_411 to i8*, !dbg !53
  %15 = getelementptr i8, i8* %14, i64 16, !dbg !53
  %16 = bitcast i8* %15 to i64*, !dbg !53
  store i64 %13, i64* %16, align 8, !dbg !53
  %17 = bitcast i64* %__nv_drb099_foo__F1L21_1Arg2 to i8*, !dbg !53
  %18 = getelementptr i8, i8* %17, i64 24, !dbg !53
  %19 = bitcast i8* %18 to i64*, !dbg !53
  %20 = load i64, i64* %19, align 8, !dbg !53
  %21 = bitcast %astruct.dt105* %.uplevelArgPack0002_411 to i8*, !dbg !53
  %22 = getelementptr i8, i8* %21, i64 24, !dbg !53
  %23 = bitcast i8* %22 to i64*, !dbg !53
  store i64 %20, i64* %23, align 8, !dbg !53
  %24 = bitcast i64* %__nv_drb099_foo__F1L21_1Arg2 to i8*, !dbg !53
  %25 = getelementptr i8, i8* %24, i64 32, !dbg !53
  %26 = bitcast i8* %25 to i64*, !dbg !53
  %27 = load i64, i64* %26, align 8, !dbg !53
  %28 = bitcast %astruct.dt105* %.uplevelArgPack0002_411 to i8*, !dbg !53
  %29 = getelementptr i8, i8* %28, i64 32, !dbg !53
  %30 = bitcast i8* %29 to i64*, !dbg !53
  store i64 %27, i64* %30, align 8, !dbg !53
  %31 = bitcast i64* %__nv_drb099_foo__F1L21_1Arg2 to i8*, !dbg !53
  %32 = getelementptr i8, i8* %31, i64 40, !dbg !53
  %33 = bitcast i8* %32 to i64*, !dbg !53
  %34 = load i64, i64* %33, align 8, !dbg !53
  %35 = bitcast %astruct.dt105* %.uplevelArgPack0002_411 to i8*, !dbg !53
  %36 = getelementptr i8, i8* %35, i64 40, !dbg !53
  %37 = bitcast i8* %36 to i64*, !dbg !53
  store i64 %34, i64* %37, align 8, !dbg !53
  %38 = bitcast i64* %__nv_drb099_foo__F1L21_1Arg2 to i8*, !dbg !53
  %39 = getelementptr i8, i8* %38, i64 48, !dbg !53
  %40 = bitcast i8* %39 to i64*, !dbg !53
  %41 = load i64, i64* %40, align 8, !dbg !53
  %42 = bitcast %astruct.dt105* %.uplevelArgPack0002_411 to i8*, !dbg !53
  %43 = getelementptr i8, i8* %42, i64 48, !dbg !53
  %44 = bitcast i8* %43 to i64*, !dbg !53
  store i64 %41, i64* %44, align 8, !dbg !53
  br label %L.LB3_414, !dbg !54

L.LB3_414:                                        ; preds = %L.LB3_324
  %45 = bitcast void (i32*, i64*, i64*)* @__nv_drb099_F1L22_2_ to i64*, !dbg !54
  %46 = bitcast %astruct.dt105* %.uplevelArgPack0002_411 to i64*, !dbg !54
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %45, i64* %46), !dbg !54
  br label %L.LB3_331

L.LB3_331:                                        ; preds = %L.LB3_414
  ret void, !dbg !53
}

define internal void @__nv_drb099_F1L22_2_(i32* %__nv_drb099_F1L22_2Arg0, i64* %__nv_drb099_F1L22_2Arg1, i64* %__nv_drb099_F1L22_2Arg2) #1 !dbg !55 {
L.entry:
  %__gtid___nv_drb099_F1L22_2__468 = alloca i32, align 4
  %.i0000p_329 = alloca i32, align 4
  %i_328 = alloca i32, align 4
  %.du0001p_357 = alloca i32, align 4
  %.de0001p_358 = alloca i32, align 4
  %.di0001p_359 = alloca i32, align 4
  %.ds0001p_360 = alloca i32, align 4
  %.dl0001p_362 = alloca i32, align 4
  %.dl0001p.copy_462 = alloca i32, align 4
  %.de0001p.copy_463 = alloca i32, align 4
  %.ds0001p.copy_464 = alloca i32, align 4
  %.dX0001p_361 = alloca i32, align 4
  %.dY0001p_356 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_drb099_F1L22_2Arg0, metadata !56, metadata !DIExpression()), !dbg !57
  call void @llvm.dbg.declare(metadata i64* %__nv_drb099_F1L22_2Arg1, metadata !58, metadata !DIExpression()), !dbg !57
  call void @llvm.dbg.declare(metadata i64* %__nv_drb099_F1L22_2Arg2, metadata !59, metadata !DIExpression()), !dbg !57
  call void @llvm.dbg.value(metadata i32 8, metadata !60, metadata !DIExpression()), !dbg !57
  call void @llvm.dbg.value(metadata i32 1, metadata !61, metadata !DIExpression()), !dbg !57
  call void @llvm.dbg.value(metadata i32 0, metadata !62, metadata !DIExpression()), !dbg !57
  call void @llvm.dbg.value(metadata i32 1, metadata !63, metadata !DIExpression()), !dbg !57
  call void @llvm.dbg.value(metadata i32 0, metadata !64, metadata !DIExpression()), !dbg !57
  call void @llvm.dbg.value(metadata i32 1, metadata !65, metadata !DIExpression()), !dbg !57
  call void @llvm.dbg.value(metadata i32 8, metadata !66, metadata !DIExpression()), !dbg !57
  call void @llvm.dbg.value(metadata i32 8, metadata !67, metadata !DIExpression()), !dbg !57
  %0 = load i32, i32* %__nv_drb099_F1L22_2Arg0, align 4, !dbg !68
  store i32 %0, i32* %__gtid___nv_drb099_F1L22_2__468, align 4, !dbg !68
  br label %L.LB5_453

L.LB5_453:                                        ; preds = %L.entry
  br label %L.LB5_327

L.LB5_327:                                        ; preds = %L.LB5_453
  store i32 0, i32* %.i0000p_329, align 4, !dbg !69
  call void @llvm.dbg.declare(metadata i32* %i_328, metadata !70, metadata !DIExpression()), !dbg !68
  store i32 1, i32* %i_328, align 4, !dbg !69
  %1 = bitcast i64* %__nv_drb099_F1L22_2Arg2 to i32**, !dbg !69
  %2 = load i32*, i32** %1, align 8, !dbg !69
  %3 = load i32, i32* %2, align 4, !dbg !69
  store i32 %3, i32* %.du0001p_357, align 4, !dbg !69
  %4 = bitcast i64* %__nv_drb099_F1L22_2Arg2 to i32**, !dbg !69
  %5 = load i32*, i32** %4, align 8, !dbg !69
  %6 = load i32, i32* %5, align 4, !dbg !69
  store i32 %6, i32* %.de0001p_358, align 4, !dbg !69
  store i32 1, i32* %.di0001p_359, align 4, !dbg !69
  %7 = load i32, i32* %.di0001p_359, align 4, !dbg !69
  store i32 %7, i32* %.ds0001p_360, align 4, !dbg !69
  store i32 1, i32* %.dl0001p_362, align 4, !dbg !69
  %8 = load i32, i32* %.dl0001p_362, align 4, !dbg !69
  store i32 %8, i32* %.dl0001p.copy_462, align 4, !dbg !69
  %9 = load i32, i32* %.de0001p_358, align 4, !dbg !69
  store i32 %9, i32* %.de0001p.copy_463, align 4, !dbg !69
  %10 = load i32, i32* %.ds0001p_360, align 4, !dbg !69
  store i32 %10, i32* %.ds0001p.copy_464, align 4, !dbg !69
  %11 = load i32, i32* %__gtid___nv_drb099_F1L22_2__468, align 4, !dbg !69
  %12 = bitcast i32* %.i0000p_329 to i64*, !dbg !69
  %13 = bitcast i32* %.dl0001p.copy_462 to i64*, !dbg !69
  %14 = bitcast i32* %.de0001p.copy_463 to i64*, !dbg !69
  %15 = bitcast i32* %.ds0001p.copy_464 to i64*, !dbg !69
  %16 = load i32, i32* %.ds0001p.copy_464, align 4, !dbg !69
  call void @__kmpc_for_static_init_4(i64* null, i32 %11, i32 34, i64* %12, i64* %13, i64* %14, i64* %15, i32 %16, i32 1), !dbg !69
  %17 = load i32, i32* %.dl0001p.copy_462, align 4, !dbg !69
  store i32 %17, i32* %.dl0001p_362, align 4, !dbg !69
  %18 = load i32, i32* %.de0001p.copy_463, align 4, !dbg !69
  store i32 %18, i32* %.de0001p_358, align 4, !dbg !69
  %19 = load i32, i32* %.ds0001p.copy_464, align 4, !dbg !69
  store i32 %19, i32* %.ds0001p_360, align 4, !dbg !69
  %20 = load i32, i32* %.dl0001p_362, align 4, !dbg !69
  store i32 %20, i32* %i_328, align 4, !dbg !69
  %21 = load i32, i32* %i_328, align 4, !dbg !69
  call void @llvm.dbg.value(metadata i32 %21, metadata !70, metadata !DIExpression()), !dbg !68
  store i32 %21, i32* %.dX0001p_361, align 4, !dbg !69
  %22 = load i32, i32* %.dX0001p_361, align 4, !dbg !69
  %23 = load i32, i32* %.du0001p_357, align 4, !dbg !69
  %24 = icmp sgt i32 %22, %23, !dbg !69
  br i1 %24, label %L.LB5_355, label %L.LB5_493, !dbg !69

L.LB5_493:                                        ; preds = %L.LB5_327
  %25 = load i32, i32* %.dX0001p_361, align 4, !dbg !69
  store i32 %25, i32* %i_328, align 4, !dbg !69
  %26 = load i32, i32* %.di0001p_359, align 4, !dbg !69
  %27 = load i32, i32* %.de0001p_358, align 4, !dbg !69
  %28 = load i32, i32* %.dX0001p_361, align 4, !dbg !69
  %29 = sub nsw i32 %27, %28, !dbg !69
  %30 = add nsw i32 %26, %29, !dbg !69
  %31 = load i32, i32* %.di0001p_359, align 4, !dbg !69
  %32 = sdiv i32 %30, %31, !dbg !69
  store i32 %32, i32* %.dY0001p_356, align 4, !dbg !69
  %33 = load i32, i32* %.dY0001p_356, align 4, !dbg !69
  %34 = icmp sle i32 %33, 0, !dbg !69
  br i1 %34, label %L.LB5_365, label %L.LB5_364, !dbg !69

L.LB5_364:                                        ; preds = %L.LB5_364, %L.LB5_493
  %35 = load i32, i32* %i_328, align 4, !dbg !71
  call void @llvm.dbg.value(metadata i32 %35, metadata !70, metadata !DIExpression()), !dbg !68
  %36 = sitofp i32 %35 to double, !dbg !71
  %37 = load i32, i32* %i_328, align 4, !dbg !71
  call void @llvm.dbg.value(metadata i32 %37, metadata !70, metadata !DIExpression()), !dbg !68
  %38 = sext i32 %37 to i64, !dbg !71
  %39 = bitcast i64* %__nv_drb099_F1L22_2Arg2 to i8*, !dbg !71
  %40 = getelementptr i8, i8* %39, i64 40, !dbg !71
  %41 = bitcast i8* %40 to i8**, !dbg !71
  %42 = load i8*, i8** %41, align 8, !dbg !71
  %43 = getelementptr i8, i8* %42, i64 56, !dbg !71
  %44 = bitcast i8* %43 to i64*, !dbg !71
  %45 = load i64, i64* %44, align 8, !dbg !71
  %46 = add nsw i64 %38, %45, !dbg !71
  %47 = bitcast i64* %__nv_drb099_F1L22_2Arg2 to i8*, !dbg !71
  %48 = getelementptr i8, i8* %47, i64 48, !dbg !71
  %49 = bitcast i8* %48 to i8***, !dbg !71
  %50 = load i8**, i8*** %49, align 8, !dbg !71
  %51 = load i8*, i8** %50, align 8, !dbg !71
  %52 = getelementptr i8, i8* %51, i64 -8, !dbg !71
  %53 = bitcast i8* %52 to double*, !dbg !71
  %54 = getelementptr double, double* %53, i64 %46, !dbg !71
  %55 = load double, double* %54, align 8, !dbg !71
  %56 = fmul fast double %36, %55, !dbg !71
  %57 = load i32, i32* %i_328, align 4, !dbg !71
  call void @llvm.dbg.value(metadata i32 %57, metadata !70, metadata !DIExpression()), !dbg !68
  %58 = sext i32 %57 to i64, !dbg !71
  %59 = bitcast i64* %__nv_drb099_F1L22_2Arg2 to i8*, !dbg !71
  %60 = getelementptr i8, i8* %59, i64 16, !dbg !71
  %61 = bitcast i8* %60 to i8**, !dbg !71
  %62 = load i8*, i8** %61, align 8, !dbg !71
  %63 = getelementptr i8, i8* %62, i64 56, !dbg !71
  %64 = bitcast i8* %63 to i64*, !dbg !71
  %65 = load i64, i64* %64, align 8, !dbg !71
  %66 = add nsw i64 %58, %65, !dbg !71
  %67 = bitcast i64* %__nv_drb099_F1L22_2Arg2 to i8*, !dbg !71
  %68 = getelementptr i8, i8* %67, i64 24, !dbg !71
  %69 = bitcast i8* %68 to i8***, !dbg !71
  %70 = load i8**, i8*** %69, align 8, !dbg !71
  %71 = load i8*, i8** %70, align 8, !dbg !71
  %72 = getelementptr i8, i8* %71, i64 -8, !dbg !71
  %73 = bitcast i8* %72 to double*, !dbg !71
  %74 = getelementptr double, double* %73, i64 %66, !dbg !71
  store double %56, double* %74, align 8, !dbg !71
  %75 = load i32, i32* %.di0001p_359, align 4, !dbg !68
  %76 = load i32, i32* %i_328, align 4, !dbg !68
  call void @llvm.dbg.value(metadata i32 %76, metadata !70, metadata !DIExpression()), !dbg !68
  %77 = add nsw i32 %75, %76, !dbg !68
  store i32 %77, i32* %i_328, align 4, !dbg !68
  %78 = load i32, i32* %.dY0001p_356, align 4, !dbg !68
  %79 = sub nsw i32 %78, 1, !dbg !68
  store i32 %79, i32* %.dY0001p_356, align 4, !dbg !68
  %80 = load i32, i32* %.dY0001p_356, align 4, !dbg !68
  %81 = icmp sgt i32 %80, 0, !dbg !68
  br i1 %81, label %L.LB5_364, label %L.LB5_365, !dbg !68

L.LB5_365:                                        ; preds = %L.LB5_364, %L.LB5_493
  br label %L.LB5_355

L.LB5_355:                                        ; preds = %L.LB5_365, %L.LB5_327
  %82 = load i32, i32* %__gtid___nv_drb099_F1L22_2__468, align 4, !dbg !68
  call void @__kmpc_for_static_fini(i64* null, i32 %82), !dbg !68
  br label %L.LB5_330

L.LB5_330:                                        ; preds = %L.LB5_355
  ret void, !dbg !68
}

define void @MAIN_() #1 !dbg !72 {
L.entry:
  %.Z0982_340 = alloca double*, align 8
  %"b$sd8_359" = alloca [16 x i64], align 8
  %.Z0981_339 = alloca double*, align 8
  %"a$sd6_356" = alloca [16 x i64], align 8
  %len_338 = alloca i32, align 4
  %z_b_0_324 = alloca i64, align 8
  %z_b_1_325 = alloca i64, align 8
  %z_e_84_328 = alloca i64, align 8
  %z_b_2_326 = alloca i64, align 8
  %z_b_3_327 = alloca i64, align 8
  %z_b_4_331 = alloca i64, align 8
  %z_b_5_332 = alloca i64, align 8
  %z_e_91_335 = alloca i64, align 8
  %z_b_6_333 = alloca i64, align 8
  %z_b_7_334 = alloca i64, align 8
  %.dY0001_377 = alloca i32, align 4
  %i_322 = alloca i32, align 4
  %x_336 = alloca float, align 4
  %z__io_347 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 8, metadata !75, metadata !DIExpression()), !dbg !76
  call void @llvm.dbg.value(metadata i32 1, metadata !77, metadata !DIExpression()), !dbg !76
  call void @llvm.dbg.value(metadata i32 0, metadata !78, metadata !DIExpression()), !dbg !76
  call void @llvm.dbg.value(metadata i32 1, metadata !79, metadata !DIExpression()), !dbg !76
  call void @llvm.dbg.value(metadata i32 0, metadata !80, metadata !DIExpression()), !dbg !76
  call void @llvm.dbg.value(metadata i32 1, metadata !81, metadata !DIExpression()), !dbg !76
  call void @llvm.dbg.value(metadata i32 8, metadata !82, metadata !DIExpression()), !dbg !76
  call void @llvm.dbg.value(metadata i32 8, metadata !83, metadata !DIExpression()), !dbg !76
  %0 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !84
  %1 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !84
  call void (i8*, ...) %1(i8* %0), !dbg !84
  call void @llvm.dbg.declare(metadata double** %.Z0982_340, metadata !85, metadata !DIExpression(DW_OP_deref)), !dbg !76
  %2 = bitcast double** %.Z0982_340 to i8**, !dbg !84
  store i8* null, i8** %2, align 8, !dbg !84
  call void @llvm.dbg.declare(metadata [16 x i64]* %"b$sd8_359", metadata !86, metadata !DIExpression()), !dbg !76
  %3 = bitcast [16 x i64]* %"b$sd8_359" to i64*, !dbg !84
  store i64 0, i64* %3, align 8, !dbg !84
  call void @llvm.dbg.declare(metadata double** %.Z0981_339, metadata !87, metadata !DIExpression(DW_OP_deref)), !dbg !76
  %4 = bitcast double** %.Z0981_339 to i8**, !dbg !84
  store i8* null, i8** %4, align 8, !dbg !84
  call void @llvm.dbg.declare(metadata [16 x i64]* %"a$sd6_356", metadata !86, metadata !DIExpression()), !dbg !76
  %5 = bitcast [16 x i64]* %"a$sd6_356" to i64*, !dbg !84
  store i64 0, i64* %5, align 8, !dbg !84
  br label %L.LB7_387

L.LB7_387:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %len_338, metadata !88, metadata !DIExpression()), !dbg !76
  store i32 1000, i32* %len_338, align 4, !dbg !89
  call void @llvm.dbg.declare(metadata i64* %z_b_0_324, metadata !90, metadata !DIExpression()), !dbg !76
  store i64 1, i64* %z_b_0_324, align 8, !dbg !91
  %6 = load i32, i32* %len_338, align 4, !dbg !91
  call void @llvm.dbg.value(metadata i32 %6, metadata !88, metadata !DIExpression()), !dbg !76
  %7 = sext i32 %6 to i64, !dbg !91
  call void @llvm.dbg.declare(metadata i64* %z_b_1_325, metadata !90, metadata !DIExpression()), !dbg !76
  store i64 %7, i64* %z_b_1_325, align 8, !dbg !91
  %8 = load i64, i64* %z_b_1_325, align 8, !dbg !91
  call void @llvm.dbg.value(metadata i64 %8, metadata !90, metadata !DIExpression()), !dbg !76
  call void @llvm.dbg.declare(metadata i64* %z_e_84_328, metadata !90, metadata !DIExpression()), !dbg !76
  store i64 %8, i64* %z_e_84_328, align 8, !dbg !91
  %9 = bitcast [16 x i64]* %"a$sd6_356" to i8*, !dbg !91
  %10 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !91
  %11 = bitcast i64* @.C357_MAIN_ to i8*, !dbg !91
  %12 = bitcast i64* @.C314_MAIN_ to i8*, !dbg !91
  %13 = bitcast i64* %z_b_0_324 to i8*, !dbg !91
  %14 = bitcast i64* %z_b_1_325 to i8*, !dbg !91
  %15 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !91
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %15(i8* %9, i8* %10, i8* %11, i8* %12, i8* %13, i8* %14), !dbg !91
  %16 = bitcast [16 x i64]* %"a$sd6_356" to i8*, !dbg !91
  %17 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !91
  call void (i8*, i32, ...) %17(i8* %16, i32 28), !dbg !91
  %18 = load i64, i64* %z_b_1_325, align 8, !dbg !91
  call void @llvm.dbg.value(metadata i64 %18, metadata !90, metadata !DIExpression()), !dbg !76
  %19 = load i64, i64* %z_b_0_324, align 8, !dbg !91
  call void @llvm.dbg.value(metadata i64 %19, metadata !90, metadata !DIExpression()), !dbg !76
  %20 = sub nsw i64 %19, 1, !dbg !91
  %21 = sub nsw i64 %18, %20, !dbg !91
  call void @llvm.dbg.declare(metadata i64* %z_b_2_326, metadata !90, metadata !DIExpression()), !dbg !76
  store i64 %21, i64* %z_b_2_326, align 8, !dbg !91
  %22 = load i64, i64* %z_b_0_324, align 8, !dbg !91
  call void @llvm.dbg.value(metadata i64 %22, metadata !90, metadata !DIExpression()), !dbg !76
  call void @llvm.dbg.declare(metadata i64* %z_b_3_327, metadata !90, metadata !DIExpression()), !dbg !76
  store i64 %22, i64* %z_b_3_327, align 8, !dbg !91
  %23 = bitcast i64* %z_b_2_326 to i8*, !dbg !91
  %24 = bitcast i64* @.C357_MAIN_ to i8*, !dbg !91
  %25 = bitcast i64* @.C314_MAIN_ to i8*, !dbg !91
  %26 = bitcast double** %.Z0981_339 to i8*, !dbg !91
  %27 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !91
  %28 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !91
  %29 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !91
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %29(i8* %23, i8* %24, i8* %25, i8* null, i8* %26, i8* null, i8* %27, i8* %28, i8* null, i64 0), !dbg !91
  call void @llvm.dbg.declare(metadata i64* %z_b_4_331, metadata !90, metadata !DIExpression()), !dbg !76
  store i64 1, i64* %z_b_4_331, align 8, !dbg !92
  %30 = load i32, i32* %len_338, align 4, !dbg !92
  call void @llvm.dbg.value(metadata i32 %30, metadata !88, metadata !DIExpression()), !dbg !76
  %31 = sext i32 %30 to i64, !dbg !92
  call void @llvm.dbg.declare(metadata i64* %z_b_5_332, metadata !90, metadata !DIExpression()), !dbg !76
  store i64 %31, i64* %z_b_5_332, align 8, !dbg !92
  %32 = load i64, i64* %z_b_5_332, align 8, !dbg !92
  call void @llvm.dbg.value(metadata i64 %32, metadata !90, metadata !DIExpression()), !dbg !76
  call void @llvm.dbg.declare(metadata i64* %z_e_91_335, metadata !90, metadata !DIExpression()), !dbg !76
  store i64 %32, i64* %z_e_91_335, align 8, !dbg !92
  %33 = bitcast [16 x i64]* %"b$sd8_359" to i8*, !dbg !92
  %34 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !92
  %35 = bitcast i64* @.C357_MAIN_ to i8*, !dbg !92
  %36 = bitcast i64* @.C314_MAIN_ to i8*, !dbg !92
  %37 = bitcast i64* %z_b_4_331 to i8*, !dbg !92
  %38 = bitcast i64* %z_b_5_332 to i8*, !dbg !92
  %39 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !92
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %39(i8* %33, i8* %34, i8* %35, i8* %36, i8* %37, i8* %38), !dbg !92
  %40 = bitcast [16 x i64]* %"b$sd8_359" to i8*, !dbg !92
  %41 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !92
  call void (i8*, i32, ...) %41(i8* %40, i32 28), !dbg !92
  %42 = load i64, i64* %z_b_5_332, align 8, !dbg !92
  call void @llvm.dbg.value(metadata i64 %42, metadata !90, metadata !DIExpression()), !dbg !76
  %43 = load i64, i64* %z_b_4_331, align 8, !dbg !92
  call void @llvm.dbg.value(metadata i64 %43, metadata !90, metadata !DIExpression()), !dbg !76
  %44 = sub nsw i64 %43, 1, !dbg !92
  %45 = sub nsw i64 %42, %44, !dbg !92
  call void @llvm.dbg.declare(metadata i64* %z_b_6_333, metadata !90, metadata !DIExpression()), !dbg !76
  store i64 %45, i64* %z_b_6_333, align 8, !dbg !92
  %46 = load i64, i64* %z_b_4_331, align 8, !dbg !92
  call void @llvm.dbg.value(metadata i64 %46, metadata !90, metadata !DIExpression()), !dbg !76
  call void @llvm.dbg.declare(metadata i64* %z_b_7_334, metadata !90, metadata !DIExpression()), !dbg !76
  store i64 %46, i64* %z_b_7_334, align 8, !dbg !92
  %47 = bitcast i64* %z_b_6_333 to i8*, !dbg !92
  %48 = bitcast i64* @.C357_MAIN_ to i8*, !dbg !92
  %49 = bitcast i64* @.C314_MAIN_ to i8*, !dbg !92
  %50 = bitcast double** %.Z0982_340 to i8*, !dbg !92
  %51 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !92
  %52 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !92
  %53 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !92
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %53(i8* %47, i8* %48, i8* %49, i8* null, i8* %50, i8* null, i8* %51, i8* %52, i8* null, i64 0), !dbg !92
  %54 = load i32, i32* %len_338, align 4, !dbg !93
  call void @llvm.dbg.value(metadata i32 %54, metadata !88, metadata !DIExpression()), !dbg !76
  store i32 %54, i32* %.dY0001_377, align 4, !dbg !93
  call void @llvm.dbg.declare(metadata i32* %i_322, metadata !94, metadata !DIExpression()), !dbg !76
  store i32 1, i32* %i_322, align 4, !dbg !93
  %55 = load i32, i32* %.dY0001_377, align 4, !dbg !93
  %56 = icmp sle i32 %55, 0, !dbg !93
  br i1 %56, label %L.LB7_376, label %L.LB7_375, !dbg !93

L.LB7_375:                                        ; preds = %L.LB7_375, %L.LB7_387
  %57 = load i32, i32* %i_322, align 4, !dbg !95
  call void @llvm.dbg.value(metadata i32 %57, metadata !94, metadata !DIExpression()), !dbg !76
  %58 = sitofp i32 %57 to double, !dbg !95
  %59 = fdiv fast double %58, 2.000000e+00, !dbg !95
  %60 = load i32, i32* %i_322, align 4, !dbg !95
  call void @llvm.dbg.value(metadata i32 %60, metadata !94, metadata !DIExpression()), !dbg !76
  %61 = sext i32 %60 to i64, !dbg !95
  %62 = bitcast [16 x i64]* %"a$sd6_356" to i8*, !dbg !95
  %63 = getelementptr i8, i8* %62, i64 56, !dbg !95
  %64 = bitcast i8* %63 to i64*, !dbg !95
  %65 = load i64, i64* %64, align 8, !dbg !95
  %66 = add nsw i64 %61, %65, !dbg !95
  %67 = load double*, double** %.Z0981_339, align 8, !dbg !95
  call void @llvm.dbg.value(metadata double* %67, metadata !87, metadata !DIExpression()), !dbg !76
  %68 = bitcast double* %67 to i8*, !dbg !95
  %69 = getelementptr i8, i8* %68, i64 -8, !dbg !95
  %70 = bitcast i8* %69 to double*, !dbg !95
  %71 = getelementptr double, double* %70, i64 %66, !dbg !95
  store double %59, double* %71, align 8, !dbg !95
  %72 = load i32, i32* %i_322, align 4, !dbg !96
  call void @llvm.dbg.value(metadata i32 %72, metadata !94, metadata !DIExpression()), !dbg !76
  %73 = sext i32 %72 to i64, !dbg !96
  %74 = bitcast [16 x i64]* %"b$sd8_359" to i8*, !dbg !96
  %75 = getelementptr i8, i8* %74, i64 56, !dbg !96
  %76 = bitcast i8* %75 to i64*, !dbg !96
  %77 = load i64, i64* %76, align 8, !dbg !96
  %78 = add nsw i64 %73, %77, !dbg !96
  %79 = load double*, double** %.Z0982_340, align 8, !dbg !96
  call void @llvm.dbg.value(metadata double* %79, metadata !85, metadata !DIExpression()), !dbg !76
  %80 = bitcast double* %79 to i8*, !dbg !96
  %81 = getelementptr i8, i8* %80, i64 -8, !dbg !96
  %82 = bitcast i8* %81 to double*, !dbg !96
  %83 = getelementptr double, double* %82, i64 %78, !dbg !96
  store double 0.000000e+00, double* %83, align 8, !dbg !96
  %84 = load i32, i32* %i_322, align 4, !dbg !97
  call void @llvm.dbg.value(metadata i32 %84, metadata !94, metadata !DIExpression()), !dbg !76
  %85 = add nsw i32 %84, 1, !dbg !97
  store i32 %85, i32* %i_322, align 4, !dbg !97
  %86 = load i32, i32* %.dY0001_377, align 4, !dbg !97
  %87 = sub nsw i32 %86, 1, !dbg !97
  store i32 %87, i32* %.dY0001_377, align 4, !dbg !97
  %88 = load i32, i32* %.dY0001_377, align 4, !dbg !97
  %89 = icmp sgt i32 %88, 0, !dbg !97
  br i1 %89, label %L.LB7_375, label %L.LB7_376, !dbg !97

L.LB7_376:                                        ; preds = %L.LB7_375, %L.LB7_387
  %90 = bitcast double** %.Z0981_339 to i64*, !dbg !98
  %91 = bitcast double** %.Z0982_340 to i64*, !dbg !98
  %92 = bitcast i32* %len_338 to i64*, !dbg !98
  %93 = bitcast [16 x i64]* %"a$sd6_356" to i64*, !dbg !98
  %94 = bitcast [16 x i64]* %"b$sd8_359" to i64*, !dbg !98
  %95 = call float @drb099_foo_(i64* %90, i64* %91, i64* %92, i64* %93, i64* %94), !dbg !98
  call void @llvm.dbg.declare(metadata float* %x_336, metadata !99, metadata !DIExpression()), !dbg !76
  store float %95, float* %x_336, align 4, !dbg !98
  %96 = bitcast [16 x i64]* %"b$sd8_359" to i8*, !dbg !100
  %97 = getelementptr i8, i8* %96, i64 80, !dbg !100
  %98 = bitcast i8* %97 to i64*, !dbg !100
  %99 = load i64, i64* %98, align 8, !dbg !100
  store i64 %99, i64* %z_b_4_331, align 8, !dbg !100
  %100 = load i64, i64* %z_b_4_331, align 8, !dbg !100
  call void @llvm.dbg.value(metadata i64 %100, metadata !90, metadata !DIExpression()), !dbg !76
  %101 = bitcast [16 x i64]* %"b$sd8_359" to i8*, !dbg !100
  %102 = getelementptr i8, i8* %101, i64 88, !dbg !100
  %103 = bitcast i8* %102 to i64*, !dbg !100
  %104 = load i64, i64* %103, align 8, !dbg !100
  %105 = sub nsw i64 %104, 1, !dbg !100
  %106 = add nsw i64 %100, %105, !dbg !100
  store i64 %106, i64* %z_b_5_332, align 8, !dbg !100
  %107 = load i64, i64* %z_b_5_332, align 8, !dbg !100
  call void @llvm.dbg.value(metadata i64 %107, metadata !90, metadata !DIExpression()), !dbg !76
  %108 = load i64, i64* %z_b_4_331, align 8, !dbg !100
  call void @llvm.dbg.value(metadata i64 %108, metadata !90, metadata !DIExpression()), !dbg !76
  %109 = sub nsw i64 %107, %108, !dbg !100
  %110 = add nsw i64 %109, 1, !dbg !100
  store i64 %110, i64* %z_e_91_335, align 8, !dbg !100
  %111 = bitcast [16 x i64]* %"a$sd6_356" to i8*, !dbg !100
  %112 = getelementptr i8, i8* %111, i64 80, !dbg !100
  %113 = bitcast i8* %112 to i64*, !dbg !100
  %114 = load i64, i64* %113, align 8, !dbg !100
  store i64 %114, i64* %z_b_0_324, align 8, !dbg !100
  %115 = load i64, i64* %z_b_0_324, align 8, !dbg !100
  call void @llvm.dbg.value(metadata i64 %115, metadata !90, metadata !DIExpression()), !dbg !76
  %116 = bitcast [16 x i64]* %"a$sd6_356" to i8*, !dbg !100
  %117 = getelementptr i8, i8* %116, i64 88, !dbg !100
  %118 = bitcast i8* %117 to i64*, !dbg !100
  %119 = load i64, i64* %118, align 8, !dbg !100
  %120 = sub nsw i64 %119, 1, !dbg !100
  %121 = add nsw i64 %115, %120, !dbg !100
  store i64 %121, i64* %z_b_1_325, align 8, !dbg !100
  %122 = load i64, i64* %z_b_1_325, align 8, !dbg !100
  call void @llvm.dbg.value(metadata i64 %122, metadata !90, metadata !DIExpression()), !dbg !76
  %123 = load i64, i64* %z_b_0_324, align 8, !dbg !100
  call void @llvm.dbg.value(metadata i64 %123, metadata !90, metadata !DIExpression()), !dbg !76
  %124 = sub nsw i64 %122, %123, !dbg !100
  %125 = add nsw i64 %124, 1, !dbg !100
  store i64 %125, i64* %z_e_84_328, align 8, !dbg !100
  call void (...) @_mp_bcs_nest(), !dbg !100
  %126 = bitcast i32* @.C344_MAIN_ to i8*, !dbg !100
  %127 = bitcast [62 x i8]* @.C342_MAIN_ to i8*, !dbg !100
  %128 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !100
  call void (i8*, i8*, i64, ...) %128(i8* %126, i8* %127, i64 62), !dbg !100
  %129 = bitcast i32* @.C345_MAIN_ to i8*, !dbg !100
  %130 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !100
  %131 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !100
  %132 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !100
  %133 = call i32 (i8*, i8*, i8*, i8*, ...) %132(i8* %129, i8* null, i8* %130, i8* %131), !dbg !100
  call void @llvm.dbg.declare(metadata i32* %z__io_347, metadata !101, metadata !DIExpression()), !dbg !76
  store i32 %133, i32* %z__io_347, align 4, !dbg !100
  %134 = bitcast [7 x i8]* @.C348_MAIN_ to i8*, !dbg !100
  %135 = bitcast i32 (...)* @f90io_sc_ch_ldw to i32 (i8*, i32, i64, ...)*, !dbg !100
  %136 = call i32 (i8*, i32, i64, ...) %135(i8* %134, i32 14, i64 7), !dbg !100
  store i32 %136, i32* %z__io_347, align 4, !dbg !100
  %137 = bitcast [16 x i64]* %"b$sd8_359" to i8*, !dbg !100
  %138 = getelementptr i8, i8* %137, i64 56, !dbg !100
  %139 = bitcast i8* %138 to i64*, !dbg !100
  %140 = load i64, i64* %139, align 8, !dbg !100
  %141 = load double*, double** %.Z0982_340, align 8, !dbg !100
  call void @llvm.dbg.value(metadata double* %141, metadata !85, metadata !DIExpression()), !dbg !76
  %142 = bitcast double* %141 to i8*, !dbg !100
  %143 = getelementptr i8, i8* %142, i64 392, !dbg !100
  %144 = bitcast i8* %143 to double*, !dbg !100
  %145 = getelementptr double, double* %144, i64 %140, !dbg !100
  %146 = load double, double* %145, align 8, !dbg !100
  %147 = bitcast i32 (...)* @f90io_sc_d_ldw to i32 (double, i32, ...)*, !dbg !100
  %148 = call i32 (double, i32, ...) %147(double %146, i32 28), !dbg !100
  store i32 %148, i32* %z__io_347, align 4, !dbg !100
  %149 = call i32 (...) @f90io_ldw_end(), !dbg !100
  store i32 %149, i32* %z__io_347, align 4, !dbg !100
  call void (...) @_mp_ecs_nest(), !dbg !100
  %150 = load double*, double** %.Z0981_339, align 8, !dbg !102
  call void @llvm.dbg.value(metadata double* %150, metadata !87, metadata !DIExpression()), !dbg !76
  %151 = bitcast double* %150 to i8*, !dbg !102
  %152 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !102
  %153 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !102
  call void (i8*, i8*, i8*, i8*, i64, ...) %153(i8* null, i8* %151, i8* %152, i8* null, i64 0), !dbg !102
  %154 = bitcast double** %.Z0981_339 to i8**, !dbg !102
  store i8* null, i8** %154, align 8, !dbg !102
  %155 = bitcast [16 x i64]* %"a$sd6_356" to i64*, !dbg !102
  store i64 0, i64* %155, align 8, !dbg !102
  %156 = load double*, double** %.Z0982_340, align 8, !dbg !102
  call void @llvm.dbg.value(metadata double* %156, metadata !85, metadata !DIExpression()), !dbg !76
  %157 = bitcast double* %156 to i8*, !dbg !102
  %158 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !102
  %159 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !102
  call void (i8*, i8*, i8*, i8*, i64, ...) %159(i8* null, i8* %157, i8* %158, i8* null, i64 0), !dbg !102
  %160 = bitcast double** %.Z0982_340 to i8**, !dbg !102
  store i8* null, i8** %160, align 8, !dbg !102
  %161 = bitcast [16 x i64]* %"b$sd8_359" to i64*, !dbg !102
  store i64 0, i64* %161, align 8, !dbg !102
  ret void, !dbg !103
}

declare void @f90_dealloc03a_i8(...) #1

declare void @_mp_ecs_nest(...) #1

declare signext i32 @f90io_ldw_end(...) #1

declare signext i32 @f90io_sc_d_ldw(...) #1

declare signext i32 @f90io_sc_ch_ldw(...) #1

declare signext i32 @f90io_print_init(...) #1

declare void @f90io_src_info03a(...) #1

declare void @_mp_bcs_nest(...) #1

declare void @f90_alloc04_chka_i8(...) #1

declare void @f90_set_intrin_type_i8(...) #1

declare void @f90_template1_i8(...) #1

declare void @fort_init(...) #1

declare void @__kmpc_for_static_fini(i64*, i32) #1

declare void @__kmpc_for_static_init_4(i64*, i32, i32, i64*, i64*, i64*, i64*, i32, i32) #1

declare void @__kmpc_fork_call(i64*, i32, i64*, i64*, ...) #1

declare void @__kmpc_end_serialized_parallel(i64*, i32) #1

declare void @__kmpc_serialized_parallel(i64*, i32) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

declare signext i32 @__kmpc_global_thread_num(i64*) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { noinline }
attributes #1 = { "no-frame-pointer-elim-non-leaf" }
attributes #2 = { nounwind readnone speculatable }

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !4, globals: !4, imports: !4)
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB099-targetparallelfor2-orig-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "foo", scope: !6, file: !3, line: 14, type: !7, scopeLine: 14, spFlags: DISPFlagDefinition, unit: !2)
!6 = !DIModule(scope: !2, name: "drb099")
!7 = !DISubroutineType(types: !8)
!8 = !{!9, !11, !11, !15, !16, !16}
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64, align: 64)
!10 = !DIBasicType(name: "real", size: 32, align: 32, encoding: DW_ATE_float)
!11 = !DICompositeType(tag: DW_TAG_array_type, baseType: !12, size: 64, align: 64, elements: !13)
!12 = !DIBasicType(name: "double precision", size: 64, align: 64, encoding: DW_ATE_float)
!13 = !{!14}
!14 = !DISubrange(count: 0, lowerBound: 1)
!15 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!16 = !DICompositeType(tag: DW_TAG_array_type, baseType: !17, size: 1024, align: 64, elements: !18)
!17 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!18 = !{!19}
!19 = !DISubrange(count: 16, lowerBound: 1)
!20 = !DILocalVariable(arg: 1, scope: !5, file: !3, type: !11, flags: DIFlagArtificial)
!21 = !DILocation(line: 0, scope: !5)
!22 = !DILocalVariable(arg: 2, scope: !5, file: !3, type: !11, flags: DIFlagArtificial)
!23 = !DILocalVariable(name: "n", arg: 3, scope: !5, file: !3, type: !15)
!24 = !DILocalVariable(arg: 4, scope: !5, file: !3, type: !16, flags: DIFlagArtificial)
!25 = !DILocalVariable(arg: 5, scope: !5, file: !3, type: !16, flags: DIFlagArtificial)
!26 = !DILocalVariable(name: "omp_nest_lock_kind", scope: !5, file: !3, type: !15)
!27 = !DILocalVariable(name: "omp_sched_static", scope: !5, file: !3, type: !15)
!28 = !DILocalVariable(name: "omp_proc_bind_false", scope: !5, file: !3, type: !15)
!29 = !DILocalVariable(name: "omp_proc_bind_true", scope: !5, file: !3, type: !15)
!30 = !DILocalVariable(name: "omp_lock_hint_none", scope: !5, file: !3, type: !15)
!31 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !5, file: !3, type: !15)
!32 = !DILocalVariable(name: "omp_lock_hint_speculative", scope: !5, file: !3, type: !15)
!33 = !DILocalVariable(name: "dp", scope: !5, file: !3, type: !15)
!34 = !DILocation(line: 30, column: 1, scope: !5)
!35 = !DILocation(line: 27, column: 1, scope: !5)
!36 = !DILocation(line: 29, column: 1, scope: !5)
!37 = !DILocalVariable(scope: !5, file: !3, type: !10, flags: DIFlagArtificial)
!38 = distinct !DISubprogram(name: "__nv_drb099_foo__F1L21_1", scope: !2, file: !3, line: 21, type: !39, scopeLine: 21, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!39 = !DISubroutineType(types: !40)
!40 = !{null, !15, !17, !17}
!41 = !DILocalVariable(name: "__nv_drb099_foo__F1L21_1Arg0", arg: 1, scope: !38, file: !3, type: !15)
!42 = !DILocation(line: 0, scope: !38)
!43 = !DILocalVariable(name: "__nv_drb099_foo__F1L21_1Arg1", arg: 2, scope: !38, file: !3, type: !17)
!44 = !DILocalVariable(name: "__nv_drb099_foo__F1L21_1Arg2", arg: 3, scope: !38, file: !3, type: !17)
!45 = !DILocalVariable(name: "omp_nest_lock_kind", scope: !38, file: !3, type: !15)
!46 = !DILocalVariable(name: "omp_sched_static", scope: !38, file: !3, type: !15)
!47 = !DILocalVariable(name: "omp_proc_bind_false", scope: !38, file: !3, type: !15)
!48 = !DILocalVariable(name: "omp_proc_bind_true", scope: !38, file: !3, type: !15)
!49 = !DILocalVariable(name: "omp_lock_hint_none", scope: !38, file: !3, type: !15)
!50 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !38, file: !3, type: !15)
!51 = !DILocalVariable(name: "omp_lock_hint_speculative", scope: !38, file: !3, type: !15)
!52 = !DILocalVariable(name: "dp", scope: !38, file: !3, type: !15)
!53 = !DILocation(line: 27, column: 1, scope: !38)
!54 = !DILocation(line: 22, column: 1, scope: !38)
!55 = distinct !DISubprogram(name: "__nv_drb099_F1L22_2", scope: !2, file: !3, line: 22, type: !39, scopeLine: 22, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!56 = !DILocalVariable(name: "__nv_drb099_F1L22_2Arg0", arg: 1, scope: !55, file: !3, type: !15)
!57 = !DILocation(line: 0, scope: !55)
!58 = !DILocalVariable(name: "__nv_drb099_F1L22_2Arg1", arg: 2, scope: !55, file: !3, type: !17)
!59 = !DILocalVariable(name: "__nv_drb099_F1L22_2Arg2", arg: 3, scope: !55, file: !3, type: !17)
!60 = !DILocalVariable(name: "omp_nest_lock_kind", scope: !55, file: !3, type: !15)
!61 = !DILocalVariable(name: "omp_sched_static", scope: !55, file: !3, type: !15)
!62 = !DILocalVariable(name: "omp_proc_bind_false", scope: !55, file: !3, type: !15)
!63 = !DILocalVariable(name: "omp_proc_bind_true", scope: !55, file: !3, type: !15)
!64 = !DILocalVariable(name: "omp_lock_hint_none", scope: !55, file: !3, type: !15)
!65 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !55, file: !3, type: !15)
!66 = !DILocalVariable(name: "omp_lock_hint_speculative", scope: !55, file: !3, type: !15)
!67 = !DILocalVariable(name: "dp", scope: !55, file: !3, type: !15)
!68 = !DILocation(line: 25, column: 1, scope: !55)
!69 = !DILocation(line: 23, column: 1, scope: !55)
!70 = !DILocalVariable(name: "i", scope: !55, file: !3, type: !15)
!71 = !DILocation(line: 24, column: 1, scope: !55)
!72 = distinct !DISubprogram(name: "drb099_targetparallelfor2_orig_no", scope: !2, file: !3, line: 33, type: !73, scopeLine: 33, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!73 = !DISubroutineType(cc: DW_CC_program, types: !74)
!74 = !{null}
!75 = !DILocalVariable(name: "omp_nest_lock_kind", scope: !72, file: !3, type: !15)
!76 = !DILocation(line: 0, scope: !72)
!77 = !DILocalVariable(name: "omp_sched_static", scope: !72, file: !3, type: !15)
!78 = !DILocalVariable(name: "omp_proc_bind_false", scope: !72, file: !3, type: !15)
!79 = !DILocalVariable(name: "omp_proc_bind_true", scope: !72, file: !3, type: !15)
!80 = !DILocalVariable(name: "omp_lock_hint_none", scope: !72, file: !3, type: !15)
!81 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !72, file: !3, type: !15)
!82 = !DILocalVariable(name: "omp_lock_hint_speculative", scope: !72, file: !3, type: !15)
!83 = !DILocalVariable(name: "dp", scope: !72, file: !3, type: !15)
!84 = !DILocation(line: 33, column: 1, scope: !72)
!85 = !DILocalVariable(name: "b", scope: !72, file: !3, type: !11)
!86 = !DILocalVariable(scope: !72, file: !3, type: !16, flags: DIFlagArtificial)
!87 = !DILocalVariable(name: "a", scope: !72, file: !3, type: !11)
!88 = !DILocalVariable(name: "len", scope: !72, file: !3, type: !15)
!89 = !DILocation(line: 43, column: 1, scope: !72)
!90 = !DILocalVariable(scope: !72, file: !3, type: !17, flags: DIFlagArtificial)
!91 = !DILocation(line: 45, column: 1, scope: !72)
!92 = !DILocation(line: 46, column: 1, scope: !72)
!93 = !DILocation(line: 48, column: 1, scope: !72)
!94 = !DILocalVariable(name: "i", scope: !72, file: !3, type: !15)
!95 = !DILocation(line: 49, column: 1, scope: !72)
!96 = !DILocation(line: 50, column: 1, scope: !72)
!97 = !DILocation(line: 51, column: 1, scope: !72)
!98 = !DILocation(line: 53, column: 1, scope: !72)
!99 = !DILocalVariable(name: "x", scope: !72, file: !3, type: !10)
!100 = !DILocation(line: 54, column: 1, scope: !72)
!101 = !DILocalVariable(scope: !72, file: !3, type: !15, flags: DIFlagArtificial)
!102 = !DILocation(line: 56, column: 1, scope: !72)
!103 = !DILocation(line: 57, column: 1, scope: !72)

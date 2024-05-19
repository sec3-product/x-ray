; ModuleID = '/tmp/DRB016-outputdep-orig-yes-95e9b0.ll'
source_filename = "/tmp/DRB016-outputdep-orig-yes-95e9b0.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct_globalarray_2_ = type <{ [8 x i8] }>
%struct_globalarray_0_ = type <{ [144 x i8] }>
%astruct.dt71 = type <{ i8*, i8*, i8*, i8*, i8* }>

@.C315_globalarray_useglobalarray_ = internal constant i32 25
@.C314_globalarray_useglobalarray_ = internal constant i64 4
@.C313_globalarray_useglobalarray_ = internal constant i64 25
@.C284_globalarray_useglobalarray_ = internal constant i64 0
@.C311_globalarray_useglobalarray_ = internal constant i64 100
@.C305_globalarray_useglobalarray_ = internal constant i64 12
@.C286_globalarray_useglobalarray_ = internal constant i64 1
@.C304_globalarray_useglobalarray_ = internal constant i64 11
@.C310_globalarray_useglobalarray_ = internal constant i32 100
@.C307_MAIN_ = internal constant i32 25
@.C306_MAIN_ = internal constant i32 14
@.C334_MAIN_ = internal constant [3 x i8] c"x ="
@.C284_MAIN_ = internal constant i64 0
@.C333_MAIN_ = internal constant i32 6
@.C330_MAIN_ = internal constant [54 x i8] c"micro-benchmarks-fortran/DRB016-outputdep-orig-yes.f95"
@.C332_MAIN_ = internal constant i32 53
@.C285_MAIN_ = internal constant i32 1
@.C322_MAIN_ = internal constant i32 10
@.C320_MAIN_ = internal constant i32 100
@.C283_MAIN_ = internal constant i32 0
@.C285___nv_MAIN__F1L46_1 = internal constant i32 1
@.C283___nv_MAIN__F1L46_1 = internal constant i32 0
@_globalarray_2_ = common global %struct_globalarray_2_ zeroinitializer, align 64, !dbg !0
@_globalarray_0_ = common global %struct_globalarray_0_ zeroinitializer, align 64, !dbg !7, !dbg !13

; Function Attrs: noinline
define float @globalarray_() #0 {
.L.entry:
  ret float undef
}

define void @globalarray_useglobalarray_(i64* %len) #1 !dbg !26 {
L.entry:
  %.g0000_343 = alloca i64, align 8
  call void @llvm.dbg.declare(metadata i64* %len, metadata !29, metadata !DIExpression()), !dbg !30
  br label %L.LB2_322

L.LB2_322:                                        ; preds = %L.entry
  %0 = bitcast i64* %len to i32*, !dbg !31
  store i32 100, i32* %0, align 4, !dbg !31
  %1 = bitcast %struct_globalarray_0_* @_globalarray_0_ to i8*, !dbg !32
  %2 = getelementptr i8, i8* %1, i64 96, !dbg !32
  %3 = bitcast i8* %2 to i64*, !dbg !32
  store i64 1, i64* %3, align 8, !dbg !32
  %4 = bitcast %struct_globalarray_0_* @_globalarray_0_ to i8*, !dbg !32
  %5 = getelementptr i8, i8* %4, i64 104, !dbg !32
  %6 = bitcast i8* %5 to i64*, !dbg !32
  store i64 100, i64* %6, align 8, !dbg !32
  %7 = bitcast %struct_globalarray_0_* @_globalarray_0_ to i8*, !dbg !32
  %8 = getelementptr i8, i8* %7, i64 104, !dbg !32
  %9 = bitcast i8* %8 to i64*, !dbg !32
  %10 = load i64, i64* %9, align 8, !dbg !32
  %11 = sub nsw i64 %10, 1, !dbg !32
  %12 = bitcast %struct_globalarray_0_* @_globalarray_0_ to i8*, !dbg !32
  %13 = getelementptr i8, i8* %12, i64 96, !dbg !32
  %14 = bitcast i8* %13 to i64*, !dbg !32
  %15 = load i64, i64* %14, align 8, !dbg !32
  %16 = add nsw i64 %11, %15, !dbg !32
  store i64 %16, i64* %.g0000_343, align 8, !dbg !32
  %17 = bitcast %struct_globalarray_0_* @_globalarray_0_ to i8*, !dbg !32
  %18 = getelementptr i8, i8* %17, i64 16, !dbg !32
  %19 = bitcast i64* @.C284_globalarray_useglobalarray_ to i8*, !dbg !32
  %20 = bitcast i64* @.C313_globalarray_useglobalarray_ to i8*, !dbg !32
  %21 = bitcast i64* @.C314_globalarray_useglobalarray_ to i8*, !dbg !32
  %22 = bitcast %struct_globalarray_0_* @_globalarray_0_ to i8*, !dbg !32
  %23 = getelementptr i8, i8* %22, i64 96, !dbg !32
  %24 = bitcast i64* %.g0000_343 to i8*, !dbg !32
  %25 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !32
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %25(i8* %18, i8* %19, i8* %20, i8* %21, i8* %23, i8* %24), !dbg !32
  %26 = bitcast %struct_globalarray_0_* @_globalarray_0_ to i8*, !dbg !32
  %27 = getelementptr i8, i8* %26, i64 16, !dbg !32
  %28 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !32
  call void (i8*, i32, ...) %28(i8* %27, i32 25), !dbg !32
  %29 = bitcast %struct_globalarray_0_* @_globalarray_0_ to i8*, !dbg !32
  %30 = getelementptr i8, i8* %29, i64 104, !dbg !32
  %31 = bitcast i8* %30 to i64*, !dbg !32
  %32 = load i64, i64* %31, align 8, !dbg !32
  %33 = sub nsw i64 %32, 1, !dbg !32
  %34 = bitcast %struct_globalarray_0_* @_globalarray_0_ to i8*, !dbg !32
  %35 = getelementptr i8, i8* %34, i64 96, !dbg !32
  %36 = bitcast i8* %35 to i64*, !dbg !32
  %37 = load i64, i64* %36, align 8, !dbg !32
  %38 = add nsw i64 %33, %37, !dbg !32
  %39 = bitcast %struct_globalarray_0_* @_globalarray_0_ to i8*, !dbg !32
  %40 = getelementptr i8, i8* %39, i64 96, !dbg !32
  %41 = bitcast i8* %40 to i64*, !dbg !32
  %42 = load i64, i64* %41, align 8, !dbg !32
  %43 = sub nsw i64 %42, 1, !dbg !32
  %44 = sub nsw i64 %38, %43, !dbg !32
  store i64 %44, i64* %.g0000_343, align 8, !dbg !32
  %45 = bitcast i64* %.g0000_343 to i8*, !dbg !32
  %46 = bitcast i64* @.C313_globalarray_useglobalarray_ to i8*, !dbg !32
  %47 = bitcast i64* @.C314_globalarray_useglobalarray_ to i8*, !dbg !32
  %48 = bitcast %struct_globalarray_0_* @_globalarray_0_ to i8*, !dbg !32
  %49 = bitcast i64* @.C286_globalarray_useglobalarray_ to i8*, !dbg !32
  %50 = bitcast i64* @.C284_globalarray_useglobalarray_ to i8*, !dbg !32
  %51 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !32
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %51(i8* %45, i8* %46, i8* %47, i8* null, i8* %48, i8* null, i8* %49, i8* %50, i8* null, i64 0), !dbg !32
  ret void, !dbg !33
}

define void @MAIN_() #1 !dbg !21 {
L.entry:
  %__gtid_MAIN__380 = alloca i32, align 4
  %len_321 = alloca i32, align 4
  %x_319 = alloca i32, align 4
  %.uplevelArgPack0001_365 = alloca %astruct.dt71, align 16
  %z__io_336 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !34, metadata !DIExpression()), !dbg !35
  call void @llvm.dbg.value(metadata i32 0, metadata !36, metadata !DIExpression()), !dbg !35
  call void @llvm.dbg.value(metadata i32 1, metadata !37, metadata !DIExpression()), !dbg !35
  call void @llvm.dbg.value(metadata i32 0, metadata !38, metadata !DIExpression()), !dbg !35
  call void @llvm.dbg.value(metadata i32 1, metadata !39, metadata !DIExpression()), !dbg !35
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !40
  store i32 %0, i32* %__gtid_MAIN__380, align 4, !dbg !40
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !41
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !41
  call void (i8*, ...) %2(i8* %1), !dbg !41
  br label %L.LB3_358

L.LB3_358:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %len_321, metadata !42, metadata !DIExpression()), !dbg !35
  store i32 100, i32* %len_321, align 4, !dbg !43
  call void @llvm.dbg.declare(metadata i32* %x_319, metadata !44, metadata !DIExpression()), !dbg !35
  store i32 10, i32* %x_319, align 4, !dbg !45
  %3 = bitcast i32* %len_321 to i64*, !dbg !46
  call void @globalarray_useglobalarray_(i64* %3), !dbg !46
  %4 = bitcast i32* %len_321 to i8*, !dbg !47
  %5 = bitcast %astruct.dt71* %.uplevelArgPack0001_365 to i8**, !dbg !47
  store i8* %4, i8** %5, align 8, !dbg !47
  %6 = bitcast %struct_globalarray_0_* @_globalarray_0_ to i8*, !dbg !47
  %7 = bitcast %astruct.dt71* %.uplevelArgPack0001_365 to i8*, !dbg !47
  %8 = getelementptr i8, i8* %7, i64 8, !dbg !47
  %9 = bitcast i8* %8 to i8**, !dbg !47
  store i8* %6, i8** %9, align 8, !dbg !47
  %10 = bitcast %struct_globalarray_0_* @_globalarray_0_ to i8*, !dbg !47
  %11 = getelementptr i8, i8* %10, i64 16, !dbg !47
  %12 = bitcast %astruct.dt71* %.uplevelArgPack0001_365 to i8*, !dbg !47
  %13 = getelementptr i8, i8* %12, i64 16, !dbg !47
  %14 = bitcast i8* %13 to i8**, !dbg !47
  store i8* %11, i8** %14, align 8, !dbg !47
  %15 = bitcast %struct_globalarray_0_* @_globalarray_0_ to i8*, !dbg !47
  %16 = bitcast %astruct.dt71* %.uplevelArgPack0001_365 to i8*, !dbg !47
  %17 = getelementptr i8, i8* %16, i64 24, !dbg !47
  %18 = bitcast i8* %17 to i8**, !dbg !47
  store i8* %15, i8** %18, align 8, !dbg !47
  %19 = bitcast i32* %x_319 to i8*, !dbg !47
  %20 = bitcast %astruct.dt71* %.uplevelArgPack0001_365 to i8*, !dbg !47
  %21 = getelementptr i8, i8* %20, i64 32, !dbg !47
  %22 = bitcast i8* %21 to i8**, !dbg !47
  store i8* %19, i8** %22, align 8, !dbg !47
  br label %L.LB3_378, !dbg !47

L.LB3_378:                                        ; preds = %L.LB3_358
  %23 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L46_1_ to i64*, !dbg !47
  %24 = bitcast %astruct.dt71* %.uplevelArgPack0001_365 to i64*, !dbg !47
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %23, i64* %24), !dbg !47
  call void (...) @_mp_bcs_nest(), !dbg !48
  %25 = bitcast i32* @.C332_MAIN_ to i8*, !dbg !48
  %26 = bitcast [54 x i8]* @.C330_MAIN_ to i8*, !dbg !48
  %27 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !48
  call void (i8*, i8*, i64, ...) %27(i8* %25, i8* %26, i64 54), !dbg !48
  %28 = bitcast i32* @.C333_MAIN_ to i8*, !dbg !48
  %29 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !48
  %30 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !48
  %31 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !48
  %32 = call i32 (i8*, i8*, i8*, i8*, ...) %31(i8* %28, i8* null, i8* %29, i8* %30), !dbg !48
  call void @llvm.dbg.declare(metadata i32* %z__io_336, metadata !49, metadata !DIExpression()), !dbg !35
  store i32 %32, i32* %z__io_336, align 4, !dbg !48
  %33 = bitcast [3 x i8]* @.C334_MAIN_ to i8*, !dbg !48
  %34 = bitcast i32 (...)* @f90io_sc_ch_ldw to i32 (i8*, i32, i64, ...)*, !dbg !48
  %35 = call i32 (i8*, i32, i64, ...) %34(i8* %33, i32 14, i64 3), !dbg !48
  store i32 %35, i32* %z__io_336, align 4, !dbg !48
  %36 = load i32, i32* %x_319, align 4, !dbg !48
  call void @llvm.dbg.value(metadata i32 %36, metadata !44, metadata !DIExpression()), !dbg !35
  %37 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !48
  %38 = call i32 (i32, i32, ...) %37(i32 %36, i32 25), !dbg !48
  store i32 %38, i32* %z__io_336, align 4, !dbg !48
  %39 = call i32 (...) @f90io_ldw_end(), !dbg !48
  store i32 %39, i32* %z__io_336, align 4, !dbg !48
  call void (...) @_mp_ecs_nest(), !dbg !48
  ret void, !dbg !40
}

define internal void @__nv_MAIN__F1L46_1_(i32* %__nv_MAIN__F1L46_1Arg0, i64* %__nv_MAIN__F1L46_1Arg1, i64* %__nv_MAIN__F1L46_1Arg2) #1 !dbg !50 {
L.entry:
  %__gtid___nv_MAIN__F1L46_1__428 = alloca i32, align 4
  %.i0000p_327 = alloca i32, align 4
  %i_326 = alloca i32, align 4
  %.du0001p_348 = alloca i32, align 4
  %.de0001p_349 = alloca i32, align 4
  %.di0001p_350 = alloca i32, align 4
  %.ds0001p_351 = alloca i32, align 4
  %.dl0001p_353 = alloca i32, align 4
  %.dl0001p.copy_422 = alloca i32, align 4
  %.de0001p.copy_423 = alloca i32, align 4
  %.ds0001p.copy_424 = alloca i32, align 4
  %.dX0001p_352 = alloca i32, align 4
  %.dY0001p_347 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L46_1Arg0, metadata !53, metadata !DIExpression()), !dbg !54
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L46_1Arg1, metadata !55, metadata !DIExpression()), !dbg !54
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L46_1Arg2, metadata !56, metadata !DIExpression()), !dbg !54
  call void @llvm.dbg.value(metadata i32 1, metadata !57, metadata !DIExpression()), !dbg !54
  call void @llvm.dbg.value(metadata i32 0, metadata !58, metadata !DIExpression()), !dbg !54
  call void @llvm.dbg.value(metadata i32 1, metadata !59, metadata !DIExpression()), !dbg !54
  call void @llvm.dbg.value(metadata i32 0, metadata !60, metadata !DIExpression()), !dbg !54
  call void @llvm.dbg.value(metadata i32 1, metadata !61, metadata !DIExpression()), !dbg !54
  %0 = load i32, i32* %__nv_MAIN__F1L46_1Arg0, align 4, !dbg !62
  store i32 %0, i32* %__gtid___nv_MAIN__F1L46_1__428, align 4, !dbg !62
  br label %L.LB4_412

L.LB4_412:                                        ; preds = %L.entry
  br label %L.LB4_325

L.LB4_325:                                        ; preds = %L.LB4_412
  store i32 0, i32* %.i0000p_327, align 4, !dbg !63
  call void @llvm.dbg.declare(metadata i32* %i_326, metadata !64, metadata !DIExpression()), !dbg !62
  store i32 1, i32* %i_326, align 4, !dbg !63
  %1 = bitcast i64* %__nv_MAIN__F1L46_1Arg2 to i32**, !dbg !63
  %2 = load i32*, i32** %1, align 8, !dbg !63
  %3 = load i32, i32* %2, align 4, !dbg !63
  store i32 %3, i32* %.du0001p_348, align 4, !dbg !63
  %4 = bitcast i64* %__nv_MAIN__F1L46_1Arg2 to i32**, !dbg !63
  %5 = load i32*, i32** %4, align 8, !dbg !63
  %6 = load i32, i32* %5, align 4, !dbg !63
  store i32 %6, i32* %.de0001p_349, align 4, !dbg !63
  store i32 1, i32* %.di0001p_350, align 4, !dbg !63
  %7 = load i32, i32* %.di0001p_350, align 4, !dbg !63
  store i32 %7, i32* %.ds0001p_351, align 4, !dbg !63
  store i32 1, i32* %.dl0001p_353, align 4, !dbg !63
  %8 = load i32, i32* %.dl0001p_353, align 4, !dbg !63
  store i32 %8, i32* %.dl0001p.copy_422, align 4, !dbg !63
  %9 = load i32, i32* %.de0001p_349, align 4, !dbg !63
  store i32 %9, i32* %.de0001p.copy_423, align 4, !dbg !63
  %10 = load i32, i32* %.ds0001p_351, align 4, !dbg !63
  store i32 %10, i32* %.ds0001p.copy_424, align 4, !dbg !63
  %11 = load i32, i32* %__gtid___nv_MAIN__F1L46_1__428, align 4, !dbg !63
  %12 = bitcast i32* %.i0000p_327 to i64*, !dbg !63
  %13 = bitcast i32* %.dl0001p.copy_422 to i64*, !dbg !63
  %14 = bitcast i32* %.de0001p.copy_423 to i64*, !dbg !63
  %15 = bitcast i32* %.ds0001p.copy_424 to i64*, !dbg !63
  %16 = load i32, i32* %.ds0001p.copy_424, align 4, !dbg !63
  call void @__kmpc_for_static_init_4(i64* null, i32 %11, i32 34, i64* %12, i64* %13, i64* %14, i64* %15, i32 %16, i32 1), !dbg !63
  %17 = load i32, i32* %.dl0001p.copy_422, align 4, !dbg !63
  store i32 %17, i32* %.dl0001p_353, align 4, !dbg !63
  %18 = load i32, i32* %.de0001p.copy_423, align 4, !dbg !63
  store i32 %18, i32* %.de0001p_349, align 4, !dbg !63
  %19 = load i32, i32* %.ds0001p.copy_424, align 4, !dbg !63
  store i32 %19, i32* %.ds0001p_351, align 4, !dbg !63
  %20 = load i32, i32* %.dl0001p_353, align 4, !dbg !63
  store i32 %20, i32* %i_326, align 4, !dbg !63
  %21 = load i32, i32* %i_326, align 4, !dbg !63
  call void @llvm.dbg.value(metadata i32 %21, metadata !64, metadata !DIExpression()), !dbg !62
  store i32 %21, i32* %.dX0001p_352, align 4, !dbg !63
  %22 = load i32, i32* %.dX0001p_352, align 4, !dbg !63
  %23 = load i32, i32* %.du0001p_348, align 4, !dbg !63
  %24 = icmp sgt i32 %22, %23, !dbg !63
  br i1 %24, label %L.LB4_346, label %L.LB4_458, !dbg !63

L.LB4_458:                                        ; preds = %L.LB4_325
  %25 = load i32, i32* %.dX0001p_352, align 4, !dbg !63
  store i32 %25, i32* %i_326, align 4, !dbg !63
  %26 = load i32, i32* %.di0001p_350, align 4, !dbg !63
  %27 = load i32, i32* %.de0001p_349, align 4, !dbg !63
  %28 = load i32, i32* %.dX0001p_352, align 4, !dbg !63
  %29 = sub nsw i32 %27, %28, !dbg !63
  %30 = add nsw i32 %26, %29, !dbg !63
  %31 = load i32, i32* %.di0001p_350, align 4, !dbg !63
  %32 = sdiv i32 %30, %31, !dbg !63
  store i32 %32, i32* %.dY0001p_347, align 4, !dbg !63
  %33 = load i32, i32* %.dY0001p_347, align 4, !dbg !63
  %34 = icmp sle i32 %33, 0, !dbg !63
  br i1 %34, label %L.LB4_356, label %L.LB4_355, !dbg !63

L.LB4_355:                                        ; preds = %L.LB4_355, %L.LB4_458
  %35 = bitcast i64* %__nv_MAIN__F1L46_1Arg2 to i8*, !dbg !65
  %36 = getelementptr i8, i8* %35, i64 32, !dbg !65
  %37 = bitcast i8* %36 to i32**, !dbg !65
  %38 = load i32*, i32** %37, align 8, !dbg !65
  %39 = load i32, i32* %38, align 4, !dbg !65
  %40 = load i32, i32* %i_326, align 4, !dbg !65
  call void @llvm.dbg.value(metadata i32 %40, metadata !64, metadata !DIExpression()), !dbg !62
  %41 = sext i32 %40 to i64, !dbg !65
  %42 = bitcast i64* %__nv_MAIN__F1L46_1Arg2 to i8*, !dbg !65
  %43 = getelementptr i8, i8* %42, i64 16, !dbg !65
  %44 = bitcast i8* %43 to i8**, !dbg !65
  %45 = load i8*, i8** %44, align 8, !dbg !65
  %46 = getelementptr i8, i8* %45, i64 56, !dbg !65
  %47 = bitcast i8* %46 to i64*, !dbg !65
  %48 = load i64, i64* %47, align 8, !dbg !65
  %49 = add nsw i64 %41, %48, !dbg !65
  %50 = bitcast i64* %__nv_MAIN__F1L46_1Arg2 to i8*, !dbg !65
  %51 = getelementptr i8, i8* %50, i64 24, !dbg !65
  %52 = bitcast i8* %51 to i8***, !dbg !65
  %53 = load i8**, i8*** %52, align 8, !dbg !65
  %54 = load i8*, i8** %53, align 8, !dbg !65
  %55 = getelementptr i8, i8* %54, i64 -4, !dbg !65
  %56 = bitcast i8* %55 to i32*, !dbg !65
  %57 = getelementptr i32, i32* %56, i64 %49, !dbg !65
  store i32 %39, i32* %57, align 4, !dbg !65
  %58 = load i32, i32* %i_326, align 4, !dbg !66
  call void @llvm.dbg.value(metadata i32 %58, metadata !64, metadata !DIExpression()), !dbg !62
  %59 = bitcast i64* %__nv_MAIN__F1L46_1Arg2 to i8*, !dbg !66
  %60 = getelementptr i8, i8* %59, i64 32, !dbg !66
  %61 = bitcast i8* %60 to i32**, !dbg !66
  %62 = load i32*, i32** %61, align 8, !dbg !66
  store i32 %58, i32* %62, align 4, !dbg !66
  %63 = load i32, i32* %.di0001p_350, align 4, !dbg !62
  %64 = load i32, i32* %i_326, align 4, !dbg !62
  call void @llvm.dbg.value(metadata i32 %64, metadata !64, metadata !DIExpression()), !dbg !62
  %65 = add nsw i32 %63, %64, !dbg !62
  store i32 %65, i32* %i_326, align 4, !dbg !62
  %66 = load i32, i32* %.dY0001p_347, align 4, !dbg !62
  %67 = sub nsw i32 %66, 1, !dbg !62
  store i32 %67, i32* %.dY0001p_347, align 4, !dbg !62
  %68 = load i32, i32* %.dY0001p_347, align 4, !dbg !62
  %69 = icmp sgt i32 %68, 0, !dbg !62
  br i1 %69, label %L.LB4_355, label %L.LB4_356, !dbg !62

L.LB4_356:                                        ; preds = %L.LB4_355, %L.LB4_458
  br label %L.LB4_346

L.LB4_346:                                        ; preds = %L.LB4_356, %L.LB4_325
  %70 = load i32, i32* %__gtid___nv_MAIN__F1L46_1__428, align 4, !dbg !62
  call void @__kmpc_for_static_fini(i64* null, i32 %70), !dbg !62
  br label %L.LB4_328

L.LB4_328:                                        ; preds = %L.LB4_346
  ret void, !dbg !62
}

declare void @__kmpc_for_static_fini(i64*, i32) #1

declare void @__kmpc_for_static_init_4(i64*, i32, i32, i64*, i64*, i64*, i64*, i32, i32) #1

declare void @_mp_ecs_nest(...) #1

declare signext i32 @f90io_ldw_end(...) #1

declare signext i32 @f90io_sc_i_ldw(...) #1

declare signext i32 @f90io_sc_ch_ldw(...) #1

declare signext i32 @f90io_print_init(...) #1

declare void @f90io_src_info03a(...) #1

declare void @_mp_bcs_nest(...) #1

declare void @fort_init(...) #1

declare signext i32 @__kmpc_global_thread_num(i64*) #1

declare void @__kmpc_fork_call(i64*, i32, i64*, i64*, ...) #1

declare void @__kmpc_end_serialized_parallel(i64*, i32) #1

declare void @__kmpc_serialized_parallel(i64*, i32) #1

declare void @f90_alloc04_chka_i8(...) #1

declare void @f90_set_intrin_type_i8(...) #1

declare void @f90_template1_i8(...) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { noinline }
attributes #1 = { "no-frame-pointer-elim-non-leaf" }
attributes #2 = { nounwind readnone speculatable }

!llvm.module.flags = !{!24, !25}
!llvm.dbg.cu = !{!3}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "z_b_0", scope: !2, file: !4, type: !16, isLocal: false, isDefinition: true)
!2 = !DIModule(scope: !3, name: "globalarray")
!3 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !4, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !6, imports: !19)
!4 = !DIFile(filename: "micro-benchmarks-fortran/DRB016-outputdep-orig-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!5 = !{}
!6 = !{!7, !13, !0}
!7 = !DIGlobalVariableExpression(var: !8, expr: !DIExpression(DW_OP_deref))
!8 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !4, type: !9, isLocal: false, isDefinition: true)
!9 = !DICompositeType(tag: DW_TAG_array_type, baseType: !10, size: 32, align: 32, elements: !11)
!10 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!11 = !{!12}
!12 = !DISubrange(count: 0, lowerBound: 1)
!13 = !DIGlobalVariableExpression(var: !14, expr: !DIExpression(DW_OP_plus_uconst, 16))
!14 = distinct !DIGlobalVariable(name: "a$sd", scope: !2, file: !4, type: !15, isLocal: false, isDefinition: true)
!15 = !DICompositeType(tag: DW_TAG_array_type, baseType: !16, size: 1024, align: 64, elements: !17)
!16 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!17 = !{!18}
!18 = !DISubrange(count: 16, lowerBound: 1)
!19 = !{!20}
!20 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !21, entity: !2, file: !4, line: 33)
!21 = distinct !DISubprogram(name: "drb016_outputdep_orig_yes", scope: !3, file: !4, line: 33, type: !22, scopeLine: 33, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !3)
!22 = !DISubroutineType(cc: DW_CC_program, types: !23)
!23 = !{null}
!24 = !{i32 2, !"Dwarf Version", i32 4}
!25 = !{i32 2, !"Debug Info Version", i32 3}
!26 = distinct !DISubprogram(name: "useglobalarray", scope: !2, file: !4, line: 26, type: !27, scopeLine: 26, spFlags: DISPFlagDefinition, unit: !3)
!27 = !DISubroutineType(types: !28)
!28 = !{null, !10}
!29 = !DILocalVariable(name: "len", arg: 1, scope: !26, file: !4, type: !10)
!30 = !DILocation(line: 0, scope: !26)
!31 = !DILocation(line: 28, column: 1, scope: !26)
!32 = !DILocation(line: 29, column: 1, scope: !26)
!33 = !DILocation(line: 30, column: 1, scope: !26)
!34 = !DILocalVariable(name: "omp_sched_static", scope: !21, file: !4, type: !10)
!35 = !DILocation(line: 0, scope: !21)
!36 = !DILocalVariable(name: "omp_proc_bind_false", scope: !21, file: !4, type: !10)
!37 = !DILocalVariable(name: "omp_proc_bind_true", scope: !21, file: !4, type: !10)
!38 = !DILocalVariable(name: "omp_lock_hint_none", scope: !21, file: !4, type: !10)
!39 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !21, file: !4, type: !10)
!40 = !DILocation(line: 54, column: 1, scope: !21)
!41 = !DILocation(line: 33, column: 1, scope: !21)
!42 = !DILocalVariable(name: "len", scope: !21, file: !4, type: !10)
!43 = !DILocation(line: 41, column: 1, scope: !21)
!44 = !DILocalVariable(name: "x", scope: !21, file: !4, type: !10)
!45 = !DILocation(line: 42, column: 1, scope: !21)
!46 = !DILocation(line: 44, column: 1, scope: !21)
!47 = !DILocation(line: 46, column: 1, scope: !21)
!48 = !DILocation(line: 53, column: 1, scope: !21)
!49 = !DILocalVariable(scope: !21, file: !4, type: !10, flags: DIFlagArtificial)
!50 = distinct !DISubprogram(name: "__nv_MAIN__F1L46_1", scope: !3, file: !4, line: 46, type: !51, scopeLine: 46, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !3)
!51 = !DISubroutineType(types: !52)
!52 = !{null, !10, !16, !16}
!53 = !DILocalVariable(name: "__nv_MAIN__F1L46_1Arg0", arg: 1, scope: !50, file: !4, type: !10)
!54 = !DILocation(line: 0, scope: !50)
!55 = !DILocalVariable(name: "__nv_MAIN__F1L46_1Arg1", arg: 2, scope: !50, file: !4, type: !16)
!56 = !DILocalVariable(name: "__nv_MAIN__F1L46_1Arg2", arg: 3, scope: !50, file: !4, type: !16)
!57 = !DILocalVariable(name: "omp_sched_static", scope: !50, file: !4, type: !10)
!58 = !DILocalVariable(name: "omp_proc_bind_false", scope: !50, file: !4, type: !10)
!59 = !DILocalVariable(name: "omp_proc_bind_true", scope: !50, file: !4, type: !10)
!60 = !DILocalVariable(name: "omp_lock_hint_none", scope: !50, file: !4, type: !10)
!61 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !50, file: !4, type: !10)
!62 = !DILocation(line: 50, column: 1, scope: !50)
!63 = !DILocation(line: 47, column: 1, scope: !50)
!64 = !DILocalVariable(name: "i", scope: !50, file: !4, type: !10)
!65 = !DILocation(line: 48, column: 1, scope: !50)
!66 = !DILocation(line: 49, column: 1, scope: !50)

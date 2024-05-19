; ModuleID = '/tmp/DRB163-simdmissinglock1-orig-gpu-no-59a9c0.ll'
source_filename = "/tmp/DRB163-simdmissinglock1-orig-gpu-no-59a9c0.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct_drb163_0_ = type <{ [72 x i8] }>
%astruct.dt157 = type <{ i8*, i8*, i8* }>

@.C307_MAIN_ = internal constant i32 25
@.C284_MAIN_ = internal constant i64 0
@.C348_MAIN_ = internal constant i32 6
@.C345_MAIN_ = internal constant [64 x i8] c"micro-benchmarks-fortran/DRB163-simdmissinglock1-orig-gpu-no.f95"
@.C347_MAIN_ = internal constant i32 38
@.C308_MAIN_ = internal constant i32 17
@.C309_MAIN_ = internal constant i32 20
@.C310_MAIN_ = internal constant i64 16
@.C286_MAIN_ = internal constant i64 1
@.C306_MAIN_ = internal constant i32 16
@.C285_MAIN_ = internal constant i32 1
@.C283_MAIN_ = internal constant i32 0
@.C306___nv_MAIN__F1L24_1 = internal constant i32 16
@.C308___nv_MAIN__F1L24_1 = internal constant i32 17
@.C309___nv_MAIN__F1L24_1 = internal constant i32 20
@.C285___nv_MAIN__F1L24_1 = internal constant i32 1
@.C283___nv_MAIN__F1L24_1 = internal constant i32 0
@.C310___nv_MAIN__F1L24_1 = internal constant i64 16
@.C286___nv_MAIN__F1L24_1 = internal constant i64 1
@.C306___nv_MAIN_F1L25_2 = internal constant i32 16
@.C308___nv_MAIN_F1L25_2 = internal constant i32 17
@.C309___nv_MAIN_F1L25_2 = internal constant i32 20
@.C285___nv_MAIN_F1L25_2 = internal constant i32 1
@.C283___nv_MAIN_F1L25_2 = internal constant i32 0
@.C310___nv_MAIN_F1L25_2 = internal constant i64 16
@.C286___nv_MAIN_F1L25_2 = internal constant i64 1
@.C306___nv_MAIN_F1L26_3 = internal constant i32 16
@.C308___nv_MAIN_F1L26_3 = internal constant i32 17
@.C285___nv_MAIN_F1L26_3 = internal constant i32 1
@.C283___nv_MAIN_F1L26_3 = internal constant i32 0
@.C310___nv_MAIN_F1L26_3 = internal constant i64 16
@.C286___nv_MAIN_F1L26_3 = internal constant i64 1
@_drb163_0_ = common global %struct_drb163_0_ zeroinitializer, align 64, !dbg !0, !dbg !7, !dbg !10

; Function Attrs: noinline
define float @drb163_() #0 {
.L.entry:
  ret float undef
}

define void @MAIN_() #1 !dbg !14 {
L.entry:
  %__gtid_MAIN__422 = alloca i32, align 4
  %.dY0001_363 = alloca i32, align 4
  %.dY0009_405 = alloca i32, align 4
  %z__io_350 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !22, metadata !DIExpression()), !dbg !23
  call void @llvm.dbg.value(metadata i32 0, metadata !24, metadata !DIExpression()), !dbg !23
  call void @llvm.dbg.value(metadata i32 1, metadata !25, metadata !DIExpression()), !dbg !23
  call void @llvm.dbg.value(metadata i32 0, metadata !26, metadata !DIExpression()), !dbg !23
  call void @llvm.dbg.value(metadata i32 1, metadata !27, metadata !DIExpression()), !dbg !23
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !28
  store i32 %0, i32* %__gtid_MAIN__422, align 4, !dbg !28
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !29
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !29
  call void (i8*, ...) %2(i8* %1), !dbg !29
  br label %L.LB2_408

L.LB2_408:                                        ; preds = %L.entry
  store i32 16, i32* %.dY0001_363, align 4, !dbg !30
  %3 = bitcast %struct_drb163_0_* @_drb163_0_ to i8*, !dbg !30
  %4 = getelementptr i8, i8* %3, i64 64, !dbg !30
  %5 = bitcast i8* %4 to i32*, !dbg !30
  store i32 1, i32* %5, align 4, !dbg !30
  br label %L.LB2_361

L.LB2_361:                                        ; preds = %L.LB2_361, %L.LB2_408
  %6 = bitcast %struct_drb163_0_* @_drb163_0_ to i8*, !dbg !31
  %7 = getelementptr i8, i8* %6, i64 64, !dbg !31
  %8 = bitcast i8* %7 to i32*, !dbg !31
  %9 = load i32, i32* %8, align 4, !dbg !31
  %10 = sext i32 %9 to i64, !dbg !31
  %11 = bitcast %struct_drb163_0_* @_drb163_0_ to i8*, !dbg !31
  %12 = getelementptr i8, i8* %11, i64 -4, !dbg !31
  %13 = bitcast i8* %12 to i32*, !dbg !31
  %14 = getelementptr i32, i32* %13, i64 %10, !dbg !31
  store i32 0, i32* %14, align 4, !dbg !31
  %15 = bitcast %struct_drb163_0_* @_drb163_0_ to i8*, !dbg !32
  %16 = getelementptr i8, i8* %15, i64 64, !dbg !32
  %17 = bitcast i8* %16 to i32*, !dbg !32
  %18 = load i32, i32* %17, align 4, !dbg !32
  %19 = add nsw i32 %18, 1, !dbg !32
  %20 = bitcast %struct_drb163_0_* @_drb163_0_ to i8*, !dbg !32
  %21 = getelementptr i8, i8* %20, i64 64, !dbg !32
  %22 = bitcast i8* %21 to i32*, !dbg !32
  store i32 %19, i32* %22, align 4, !dbg !32
  %23 = load i32, i32* %.dY0001_363, align 4, !dbg !32
  %24 = sub nsw i32 %23, 1, !dbg !32
  store i32 %24, i32* %.dY0001_363, align 4, !dbg !32
  %25 = load i32, i32* %.dY0001_363, align 4, !dbg !32
  %26 = icmp sgt i32 %25, 0, !dbg !32
  br i1 %26, label %L.LB2_361, label %L.LB2_436, !dbg !32

L.LB2_436:                                        ; preds = %L.LB2_361
  call void @__nv_MAIN__F1L24_1_(i32* %__gtid_MAIN__422, i64* null, i64* null), !dbg !33
  store i32 16, i32* %.dY0009_405, align 4, !dbg !34
  %27 = bitcast %struct_drb163_0_* @_drb163_0_ to i8*, !dbg !34
  %28 = getelementptr i8, i8* %27, i64 64, !dbg !34
  %29 = bitcast i8* %28 to i32*, !dbg !34
  store i32 1, i32* %29, align 4, !dbg !34
  br label %L.LB2_403

L.LB2_403:                                        ; preds = %L.LB2_406, %L.LB2_436
  %30 = bitcast %struct_drb163_0_* @_drb163_0_ to i8*, !dbg !35
  %31 = getelementptr i8, i8* %30, i64 64, !dbg !35
  %32 = bitcast i8* %31 to i32*, !dbg !35
  %33 = load i32, i32* %32, align 4, !dbg !35
  %34 = sext i32 %33 to i64, !dbg !35
  %35 = bitcast %struct_drb163_0_* @_drb163_0_ to i8*, !dbg !35
  %36 = getelementptr i8, i8* %35, i64 -4, !dbg !35
  %37 = bitcast i8* %36 to i32*, !dbg !35
  %38 = getelementptr i32, i32* %37, i64 %34, !dbg !35
  %39 = load i32, i32* %38, align 4, !dbg !35
  %40 = icmp eq i32 %39, 20, !dbg !35
  br i1 %40, label %L.LB2_406, label %L.LB2_437, !dbg !35

L.LB2_437:                                        ; preds = %L.LB2_403
  call void (...) @_mp_bcs_nest(), !dbg !36
  %41 = bitcast i32* @.C347_MAIN_ to i8*, !dbg !36
  %42 = bitcast [64 x i8]* @.C345_MAIN_ to i8*, !dbg !36
  %43 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !36
  call void (i8*, i8*, i64, ...) %43(i8* %41, i8* %42, i64 64), !dbg !36
  %44 = bitcast i32* @.C348_MAIN_ to i8*, !dbg !36
  %45 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !36
  %46 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !36
  %47 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !36
  %48 = call i32 (i8*, i8*, i8*, i8*, ...) %47(i8* %44, i8* null, i8* %45, i8* %46), !dbg !36
  call void @llvm.dbg.declare(metadata i32* %z__io_350, metadata !37, metadata !DIExpression()), !dbg !23
  store i32 %48, i32* %z__io_350, align 4, !dbg !36
  %49 = bitcast %struct_drb163_0_* @_drb163_0_ to i8*, !dbg !36
  %50 = getelementptr i8, i8* %49, i64 64, !dbg !36
  %51 = bitcast i8* %50 to i32*, !dbg !36
  %52 = load i32, i32* %51, align 4, !dbg !36
  %53 = sext i32 %52 to i64, !dbg !36
  %54 = bitcast %struct_drb163_0_* @_drb163_0_ to i8*, !dbg !36
  %55 = getelementptr i8, i8* %54, i64 -4, !dbg !36
  %56 = bitcast i8* %55 to i32*, !dbg !36
  %57 = getelementptr i32, i32* %56, i64 %53, !dbg !36
  %58 = load i32, i32* %57, align 4, !dbg !36
  %59 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !36
  %60 = call i32 (i32, i32, ...) %59(i32 %58, i32 25), !dbg !36
  store i32 %60, i32* %z__io_350, align 4, !dbg !36
  %61 = bitcast %struct_drb163_0_* @_drb163_0_ to i8*, !dbg !36
  %62 = getelementptr i8, i8* %61, i64 64, !dbg !36
  %63 = bitcast i8* %62 to i32*, !dbg !36
  %64 = load i32, i32* %63, align 4, !dbg !36
  %65 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !36
  %66 = call i32 (i32, i32, ...) %65(i32 %64, i32 25), !dbg !36
  store i32 %66, i32* %z__io_350, align 4, !dbg !36
  %67 = call i32 (...) @f90io_ldw_end(), !dbg !36
  store i32 %67, i32* %z__io_350, align 4, !dbg !36
  call void (...) @_mp_ecs_nest(), !dbg !36
  br label %L.LB2_406

L.LB2_406:                                        ; preds = %L.LB2_437, %L.LB2_403
  %68 = bitcast %struct_drb163_0_* @_drb163_0_ to i8*, !dbg !38
  %69 = getelementptr i8, i8* %68, i64 64, !dbg !38
  %70 = bitcast i8* %69 to i32*, !dbg !38
  %71 = load i32, i32* %70, align 4, !dbg !38
  %72 = add nsw i32 %71, 1, !dbg !38
  %73 = bitcast %struct_drb163_0_* @_drb163_0_ to i8*, !dbg !38
  %74 = getelementptr i8, i8* %73, i64 64, !dbg !38
  %75 = bitcast i8* %74 to i32*, !dbg !38
  store i32 %72, i32* %75, align 4, !dbg !38
  %76 = load i32, i32* %.dY0009_405, align 4, !dbg !38
  %77 = sub nsw i32 %76, 1, !dbg !38
  store i32 %77, i32* %.dY0009_405, align 4, !dbg !38
  %78 = load i32, i32* %.dY0009_405, align 4, !dbg !38
  %79 = icmp sgt i32 %78, 0, !dbg !38
  br i1 %79, label %L.LB2_403, label %L.LB2_438, !dbg !38

L.LB2_438:                                        ; preds = %L.LB2_406
  ret void, !dbg !28
}

define internal void @__nv_MAIN__F1L24_1_(i32* %__nv_MAIN__F1L24_1Arg0, i64* %__nv_MAIN__F1L24_1Arg1, i64* %__nv_MAIN__F1L24_1Arg2) #1 !dbg !39 {
L.entry:
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L24_1Arg0, metadata !43, metadata !DIExpression()), !dbg !44
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L24_1Arg1, metadata !45, metadata !DIExpression()), !dbg !44
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L24_1Arg2, metadata !46, metadata !DIExpression()), !dbg !44
  call void @llvm.dbg.value(metadata i32 1, metadata !47, metadata !DIExpression()), !dbg !44
  call void @llvm.dbg.value(metadata i32 0, metadata !48, metadata !DIExpression()), !dbg !44
  call void @llvm.dbg.value(metadata i32 1, metadata !49, metadata !DIExpression()), !dbg !44
  call void @llvm.dbg.value(metadata i32 0, metadata !50, metadata !DIExpression()), !dbg !44
  call void @llvm.dbg.value(metadata i32 1, metadata !51, metadata !DIExpression()), !dbg !44
  br label %L.LB3_442

L.LB3_442:                                        ; preds = %L.entry
  br label %L.LB3_317

L.LB3_317:                                        ; preds = %L.LB3_442
  %0 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN_F1L25_2_ to i64*, !dbg !52
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_teams(i64* null, i32 1, i64* %0, i64* %__nv_MAIN__F1L24_1Arg2), !dbg !52
  br label %L.LB3_343

L.LB3_343:                                        ; preds = %L.LB3_317
  ret void, !dbg !53
}

define internal void @__nv_MAIN_F1L25_2_(i32* %__nv_MAIN_F1L25_2Arg0, i64* %__nv_MAIN_F1L25_2Arg1, i64* %__nv_MAIN_F1L25_2Arg2) #1 !dbg !54 {
L.entry:
  %__gtid___nv_MAIN_F1L25_2__481 = alloca i32, align 4
  %.dY0002_366 = alloca i64, align 8
  %"i$c_356" = alloca i64, align 8
  %var_321 = alloca [16 x i32], align 16
  %.i0000p_325 = alloca i32, align 4
  %.i0001p_326 = alloca i32, align 4
  %.i0002p_327 = alloca i32, align 4
  %.i0003p_328 = alloca i32, align 4
  %i_324 = alloca i32, align 4
  %.du0003_370 = alloca i32, align 4
  %.de0003_371 = alloca i32, align 4
  %.di0003_372 = alloca i32, align 4
  %.ds0003_373 = alloca i32, align 4
  %.dl0003_375 = alloca i32, align 4
  %.dl0003.copy_475 = alloca i32, align 4
  %.de0003.copy_476 = alloca i32, align 4
  %.ds0003.copy_477 = alloca i32, align 4
  %.dX0003_374 = alloca i32, align 4
  %.dY0003_369 = alloca i32, align 4
  %.uplevelArgPack0001_500 = alloca %astruct.dt157, align 16
  %.dY0008_402 = alloca i64, align 8
  %"i$e_358" = alloca i64, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L25_2Arg0, metadata !55, metadata !DIExpression()), !dbg !56
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L25_2Arg1, metadata !57, metadata !DIExpression()), !dbg !56
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L25_2Arg2, metadata !58, metadata !DIExpression()), !dbg !56
  call void @llvm.dbg.value(metadata i32 1, metadata !59, metadata !DIExpression()), !dbg !56
  call void @llvm.dbg.value(metadata i32 0, metadata !60, metadata !DIExpression()), !dbg !56
  call void @llvm.dbg.value(metadata i32 1, metadata !61, metadata !DIExpression()), !dbg !56
  call void @llvm.dbg.value(metadata i32 0, metadata !62, metadata !DIExpression()), !dbg !56
  call void @llvm.dbg.value(metadata i32 1, metadata !63, metadata !DIExpression()), !dbg !56
  %0 = load i32, i32* %__nv_MAIN_F1L25_2Arg0, align 4, !dbg !64
  store i32 %0, i32* %__gtid___nv_MAIN_F1L25_2__481, align 4, !dbg !64
  br label %L.LB5_460

L.LB5_460:                                        ; preds = %L.entry
  br label %L.LB5_320

L.LB5_320:                                        ; preds = %L.LB5_460
  store i64 16, i64* %.dY0002_366, align 8, !dbg !65
  call void @llvm.dbg.declare(metadata i64* %"i$c_356", metadata !66, metadata !DIExpression()), !dbg !56
  store i64 1, i64* %"i$c_356", align 8, !dbg !65
  br label %L.LB5_364

L.LB5_364:                                        ; preds = %L.LB5_364, %L.LB5_320
  %1 = load i64, i64* %"i$c_356", align 8, !dbg !65
  call void @llvm.dbg.value(metadata i64 %1, metadata !66, metadata !DIExpression()), !dbg !56
  call void @llvm.dbg.declare(metadata [16 x i32]* %var_321, metadata !67, metadata !DIExpression()), !dbg !64
  %2 = bitcast [16 x i32]* %var_321 to i8*, !dbg !65
  %3 = getelementptr i8, i8* %2, i64 -4, !dbg !65
  %4 = bitcast i8* %3 to i32*, !dbg !65
  %5 = getelementptr i32, i32* %4, i64 %1, !dbg !65
  store i32 0, i32* %5, align 4, !dbg !65
  %6 = load i64, i64* %"i$c_356", align 8, !dbg !65
  call void @llvm.dbg.value(metadata i64 %6, metadata !66, metadata !DIExpression()), !dbg !56
  %7 = add nsw i64 %6, 1, !dbg !65
  store i64 %7, i64* %"i$c_356", align 8, !dbg !65
  %8 = load i64, i64* %.dY0002_366, align 8, !dbg !65
  %9 = sub nsw i64 %8, 1, !dbg !65
  store i64 %9, i64* %.dY0002_366, align 8, !dbg !65
  %10 = load i64, i64* %.dY0002_366, align 8, !dbg !65
  %11 = icmp sgt i64 %10, 0, !dbg !65
  br i1 %11, label %L.LB5_364, label %L.LB5_322, !dbg !65

L.LB5_322:                                        ; preds = %L.LB5_364
  br label %L.LB5_323

L.LB5_323:                                        ; preds = %L.LB5_322
  store i32 0, i32* %.i0000p_325, align 4, !dbg !68
  store i32 1, i32* %.i0001p_326, align 4, !dbg !68
  store i32 20, i32* %.i0002p_327, align 4, !dbg !68
  store i32 1, i32* %.i0003p_328, align 4, !dbg !68
  %12 = load i32, i32* %.i0001p_326, align 4, !dbg !68
  call void @llvm.dbg.declare(metadata i32* %i_324, metadata !69, metadata !DIExpression()), !dbg !64
  store i32 %12, i32* %i_324, align 4, !dbg !68
  %13 = load i32, i32* %.i0002p_327, align 4, !dbg !68
  store i32 %13, i32* %.du0003_370, align 4, !dbg !68
  %14 = load i32, i32* %.i0002p_327, align 4, !dbg !68
  store i32 %14, i32* %.de0003_371, align 4, !dbg !68
  store i32 1, i32* %.di0003_372, align 4, !dbg !68
  %15 = load i32, i32* %.di0003_372, align 4, !dbg !68
  store i32 %15, i32* %.ds0003_373, align 4, !dbg !68
  %16 = load i32, i32* %.i0001p_326, align 4, !dbg !68
  store i32 %16, i32* %.dl0003_375, align 4, !dbg !68
  %17 = load i32, i32* %.dl0003_375, align 4, !dbg !68
  store i32 %17, i32* %.dl0003.copy_475, align 4, !dbg !68
  %18 = load i32, i32* %.de0003_371, align 4, !dbg !68
  store i32 %18, i32* %.de0003.copy_476, align 4, !dbg !68
  %19 = load i32, i32* %.ds0003_373, align 4, !dbg !68
  store i32 %19, i32* %.ds0003.copy_477, align 4, !dbg !68
  %20 = load i32, i32* %__gtid___nv_MAIN_F1L25_2__481, align 4, !dbg !68
  %21 = bitcast i32* %.i0000p_325 to i64*, !dbg !68
  %22 = bitcast i32* %.dl0003.copy_475 to i64*, !dbg !68
  %23 = bitcast i32* %.de0003.copy_476 to i64*, !dbg !68
  %24 = bitcast i32* %.ds0003.copy_477 to i64*, !dbg !68
  %25 = load i32, i32* %.ds0003.copy_477, align 4, !dbg !68
  call void @__kmpc_for_static_init_4(i64* null, i32 %20, i32 92, i64* %21, i64* %22, i64* %23, i64* %24, i32 %25, i32 1), !dbg !68
  %26 = load i32, i32* %.dl0003.copy_475, align 4, !dbg !68
  store i32 %26, i32* %.dl0003_375, align 4, !dbg !68
  %27 = load i32, i32* %.de0003.copy_476, align 4, !dbg !68
  store i32 %27, i32* %.de0003_371, align 4, !dbg !68
  %28 = load i32, i32* %.ds0003.copy_477, align 4, !dbg !68
  store i32 %28, i32* %.ds0003_373, align 4, !dbg !68
  %29 = load i32, i32* %.dl0003_375, align 4, !dbg !68
  store i32 %29, i32* %i_324, align 4, !dbg !68
  %30 = load i32, i32* %i_324, align 4, !dbg !68
  call void @llvm.dbg.value(metadata i32 %30, metadata !69, metadata !DIExpression()), !dbg !64
  store i32 %30, i32* %.dX0003_374, align 4, !dbg !68
  %31 = load i32, i32* %.dX0003_374, align 4, !dbg !68
  %32 = load i32, i32* %.du0003_370, align 4, !dbg !68
  %33 = icmp sgt i32 %31, %32, !dbg !68
  br i1 %33, label %L.LB5_368, label %L.LB5_533, !dbg !68

L.LB5_533:                                        ; preds = %L.LB5_323
  %34 = load i32, i32* %.du0003_370, align 4, !dbg !68
  %35 = load i32, i32* %.de0003_371, align 4, !dbg !68
  %36 = icmp slt i32 %34, %35, !dbg !68
  %37 = select i1 %36, i32 %34, i32 %35, !dbg !68
  store i32 %37, i32* %.de0003_371, align 4, !dbg !68
  %38 = load i32, i32* %.dX0003_374, align 4, !dbg !68
  store i32 %38, i32* %i_324, align 4, !dbg !68
  %39 = load i32, i32* %.di0003_372, align 4, !dbg !68
  %40 = load i32, i32* %.de0003_371, align 4, !dbg !68
  %41 = load i32, i32* %.dX0003_374, align 4, !dbg !68
  %42 = sub nsw i32 %40, %41, !dbg !68
  %43 = add nsw i32 %39, %42, !dbg !68
  %44 = load i32, i32* %.di0003_372, align 4, !dbg !68
  %45 = sdiv i32 %43, %44, !dbg !68
  store i32 %45, i32* %.dY0003_369, align 4, !dbg !68
  %46 = load i32, i32* %i_324, align 4, !dbg !68
  call void @llvm.dbg.value(metadata i32 %46, metadata !69, metadata !DIExpression()), !dbg !64
  store i32 %46, i32* %.i0001p_326, align 4, !dbg !68
  %47 = load i32, i32* %.de0003_371, align 4, !dbg !68
  store i32 %47, i32* %.i0002p_327, align 4, !dbg !68
  %48 = bitcast [16 x i32]* %var_321 to i8*, !dbg !68
  %49 = bitcast %astruct.dt157* %.uplevelArgPack0001_500 to i8**, !dbg !68
  store i8* %48, i8** %49, align 8, !dbg !68
  %50 = bitcast i32* %.i0001p_326 to i8*, !dbg !68
  %51 = bitcast %astruct.dt157* %.uplevelArgPack0001_500 to i8*, !dbg !68
  %52 = getelementptr i8, i8* %51, i64 8, !dbg !68
  %53 = bitcast i8* %52 to i8**, !dbg !68
  store i8* %50, i8** %53, align 8, !dbg !68
  %54 = bitcast i32* %.i0002p_327 to i8*, !dbg !68
  %55 = bitcast %astruct.dt157* %.uplevelArgPack0001_500 to i8*, !dbg !68
  %56 = getelementptr i8, i8* %55, i64 16, !dbg !68
  %57 = bitcast i8* %56 to i8**, !dbg !68
  store i8* %54, i8** %57, align 8, !dbg !68
  br label %L.LB5_507, !dbg !68

L.LB5_507:                                        ; preds = %L.LB5_533
  %58 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN_F1L26_3_ to i64*, !dbg !68
  %59 = bitcast %astruct.dt157* %.uplevelArgPack0001_500 to i64*, !dbg !68
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %58, i64* %59), !dbg !68
  br label %L.LB5_368

L.LB5_368:                                        ; preds = %L.LB5_507, %L.LB5_323
  %60 = load i32, i32* %__gtid___nv_MAIN_F1L25_2__481, align 4, !dbg !70
  call void @__kmpc_for_static_fini(i64* null, i32 %60), !dbg !70
  br label %L.LB5_340

L.LB5_340:                                        ; preds = %L.LB5_368
  br label %L.LB5_341

L.LB5_341:                                        ; preds = %L.LB5_340
  %61 = call i32 (...) @_mp_bcs_nest_red(), !dbg !64
  %62 = call i32 (...) @_mp_bcs_nest_red(), !dbg !64
  store i64 16, i64* %.dY0008_402, align 8, !dbg !64
  call void @llvm.dbg.declare(metadata i64* %"i$e_358", metadata !66, metadata !DIExpression()), !dbg !56
  store i64 1, i64* %"i$e_358", align 8, !dbg !64
  br label %L.LB5_400

L.LB5_400:                                        ; preds = %L.LB5_400, %L.LB5_341
  %63 = load i64, i64* %"i$e_358", align 8, !dbg !64
  call void @llvm.dbg.value(metadata i64 %63, metadata !66, metadata !DIExpression()), !dbg !56
  %64 = bitcast [16 x i32]* %var_321 to i8*, !dbg !64
  %65 = getelementptr i8, i8* %64, i64 -4, !dbg !64
  %66 = bitcast i8* %65 to i32*, !dbg !64
  %67 = getelementptr i32, i32* %66, i64 %63, !dbg !64
  %68 = load i32, i32* %67, align 4, !dbg !64
  %69 = load i64, i64* %"i$e_358", align 8, !dbg !64
  call void @llvm.dbg.value(metadata i64 %69, metadata !66, metadata !DIExpression()), !dbg !56
  %70 = bitcast %struct_drb163_0_* @_drb163_0_ to i8*, !dbg !64
  %71 = getelementptr i8, i8* %70, i64 -4, !dbg !64
  %72 = bitcast i8* %71 to i32*, !dbg !64
  %73 = getelementptr i32, i32* %72, i64 %69, !dbg !64
  %74 = load i32, i32* %73, align 4, !dbg !64
  %75 = add nsw i32 %68, %74, !dbg !64
  %76 = load i64, i64* %"i$e_358", align 8, !dbg !64
  call void @llvm.dbg.value(metadata i64 %76, metadata !66, metadata !DIExpression()), !dbg !56
  %77 = bitcast %struct_drb163_0_* @_drb163_0_ to i8*, !dbg !64
  %78 = getelementptr i8, i8* %77, i64 -4, !dbg !64
  %79 = bitcast i8* %78 to i32*, !dbg !64
  %80 = getelementptr i32, i32* %79, i64 %76, !dbg !64
  store i32 %75, i32* %80, align 4, !dbg !64
  %81 = load i64, i64* %"i$e_358", align 8, !dbg !64
  call void @llvm.dbg.value(metadata i64 %81, metadata !66, metadata !DIExpression()), !dbg !56
  %82 = add nsw i64 %81, 1, !dbg !64
  store i64 %82, i64* %"i$e_358", align 8, !dbg !64
  %83 = load i64, i64* %.dY0008_402, align 8, !dbg !64
  %84 = sub nsw i64 %83, 1, !dbg !64
  store i64 %84, i64* %.dY0008_402, align 8, !dbg !64
  %85 = load i64, i64* %.dY0008_402, align 8, !dbg !64
  %86 = icmp sgt i64 %85, 0, !dbg !64
  br i1 %86, label %L.LB5_400, label %L.LB5_534, !dbg !64

L.LB5_534:                                        ; preds = %L.LB5_400
  %87 = call i32 (...) @_mp_ecs_nest_red(), !dbg !64
  %88 = call i32 (...) @_mp_ecs_nest_red(), !dbg !64
  br label %L.LB5_342

L.LB5_342:                                        ; preds = %L.LB5_534
  ret void, !dbg !64
}

define internal void @__nv_MAIN_F1L26_3_(i32* %__nv_MAIN_F1L26_3Arg0, i64* %__nv_MAIN_F1L26_3Arg1, i64* %__nv_MAIN_F1L26_3Arg2) #1 !dbg !71 {
L.entry:
  %__gtid___nv_MAIN_F1L26_3__559 = alloca i32, align 4
  %.dY0004p_381 = alloca i64, align 8
  %"i$d_357" = alloca i64, align 8
  %var_332 = alloca [16 x i32], align 16
  %.i0004p_334 = alloca i32, align 4
  %i_333 = alloca i32, align 4
  %.du0005p_385 = alloca i32, align 4
  %.de0005p_386 = alloca i32, align 4
  %.di0005p_387 = alloca i32, align 4
  %.ds0005p_388 = alloca i32, align 4
  %.dl0005p_390 = alloca i32, align 4
  %.dl0005p.copy_553 = alloca i32, align 4
  %.de0005p.copy_554 = alloca i32, align 4
  %.ds0005p.copy_555 = alloca i32, align 4
  %.dX0005p_389 = alloca i32, align 4
  %.dY0005p_384 = alloca i32, align 4
  %.i0005p_337 = alloca i32, align 4
  %.dY0006p_396 = alloca i32, align 4
  %j_336 = alloca i32, align 4
  %.dY0007p_399 = alloca i64, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L26_3Arg0, metadata !72, metadata !DIExpression()), !dbg !73
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L26_3Arg1, metadata !74, metadata !DIExpression()), !dbg !73
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L26_3Arg2, metadata !75, metadata !DIExpression()), !dbg !73
  call void @llvm.dbg.value(metadata i32 1, metadata !76, metadata !DIExpression()), !dbg !73
  call void @llvm.dbg.value(metadata i32 0, metadata !77, metadata !DIExpression()), !dbg !73
  call void @llvm.dbg.value(metadata i32 1, metadata !78, metadata !DIExpression()), !dbg !73
  call void @llvm.dbg.value(metadata i32 0, metadata !79, metadata !DIExpression()), !dbg !73
  call void @llvm.dbg.value(metadata i32 1, metadata !80, metadata !DIExpression()), !dbg !73
  %0 = load i32, i32* %__nv_MAIN_F1L26_3Arg0, align 4, !dbg !81
  store i32 %0, i32* %__gtid___nv_MAIN_F1L26_3__559, align 4, !dbg !81
  br label %L.LB7_538

L.LB7_538:                                        ; preds = %L.entry
  br label %L.LB7_331

L.LB7_331:                                        ; preds = %L.LB7_538
  store i64 16, i64* %.dY0004p_381, align 8, !dbg !82
  call void @llvm.dbg.declare(metadata i64* %"i$d_357", metadata !83, metadata !DIExpression()), !dbg !73
  store i64 1, i64* %"i$d_357", align 8, !dbg !82
  br label %L.LB7_379

L.LB7_379:                                        ; preds = %L.LB7_379, %L.LB7_331
  %1 = load i64, i64* %"i$d_357", align 8, !dbg !82
  call void @llvm.dbg.value(metadata i64 %1, metadata !83, metadata !DIExpression()), !dbg !73
  call void @llvm.dbg.declare(metadata [16 x i32]* %var_332, metadata !84, metadata !DIExpression()), !dbg !81
  %2 = bitcast [16 x i32]* %var_332 to i8*, !dbg !82
  %3 = getelementptr i8, i8* %2, i64 -4, !dbg !82
  %4 = bitcast i8* %3 to i32*, !dbg !82
  %5 = getelementptr i32, i32* %4, i64 %1, !dbg !82
  store i32 0, i32* %5, align 4, !dbg !82
  %6 = load i64, i64* %"i$d_357", align 8, !dbg !82
  call void @llvm.dbg.value(metadata i64 %6, metadata !83, metadata !DIExpression()), !dbg !73
  %7 = add nsw i64 %6, 1, !dbg !82
  store i64 %7, i64* %"i$d_357", align 8, !dbg !82
  %8 = load i64, i64* %.dY0004p_381, align 8, !dbg !82
  %9 = sub nsw i64 %8, 1, !dbg !82
  store i64 %9, i64* %.dY0004p_381, align 8, !dbg !82
  %10 = load i64, i64* %.dY0004p_381, align 8, !dbg !82
  %11 = icmp sgt i64 %10, 0, !dbg !82
  br i1 %11, label %L.LB7_379, label %L.LB7_572, !dbg !82

L.LB7_572:                                        ; preds = %L.LB7_379
  store i32 0, i32* %.i0004p_334, align 4, !dbg !82
  %12 = bitcast i64* %__nv_MAIN_F1L26_3Arg2 to i8*, !dbg !82
  %13 = getelementptr i8, i8* %12, i64 8, !dbg !82
  %14 = bitcast i8* %13 to i32**, !dbg !82
  %15 = load i32*, i32** %14, align 8, !dbg !82
  %16 = load i32, i32* %15, align 4, !dbg !82
  call void @llvm.dbg.declare(metadata i32* %i_333, metadata !85, metadata !DIExpression()), !dbg !81
  store i32 %16, i32* %i_333, align 4, !dbg !82
  %17 = bitcast i64* %__nv_MAIN_F1L26_3Arg2 to i8*, !dbg !82
  %18 = getelementptr i8, i8* %17, i64 16, !dbg !82
  %19 = bitcast i8* %18 to i32**, !dbg !82
  %20 = load i32*, i32** %19, align 8, !dbg !82
  %21 = load i32, i32* %20, align 4, !dbg !82
  store i32 %21, i32* %.du0005p_385, align 4, !dbg !82
  %22 = bitcast i64* %__nv_MAIN_F1L26_3Arg2 to i8*, !dbg !82
  %23 = getelementptr i8, i8* %22, i64 16, !dbg !82
  %24 = bitcast i8* %23 to i32**, !dbg !82
  %25 = load i32*, i32** %24, align 8, !dbg !82
  %26 = load i32, i32* %25, align 4, !dbg !82
  store i32 %26, i32* %.de0005p_386, align 4, !dbg !82
  store i32 1, i32* %.di0005p_387, align 4, !dbg !82
  %27 = load i32, i32* %.di0005p_387, align 4, !dbg !82
  store i32 %27, i32* %.ds0005p_388, align 4, !dbg !82
  %28 = bitcast i64* %__nv_MAIN_F1L26_3Arg2 to i8*, !dbg !82
  %29 = getelementptr i8, i8* %28, i64 8, !dbg !82
  %30 = bitcast i8* %29 to i32**, !dbg !82
  %31 = load i32*, i32** %30, align 8, !dbg !82
  %32 = load i32, i32* %31, align 4, !dbg !82
  store i32 %32, i32* %.dl0005p_390, align 4, !dbg !82
  %33 = load i32, i32* %.dl0005p_390, align 4, !dbg !82
  store i32 %33, i32* %.dl0005p.copy_553, align 4, !dbg !82
  %34 = load i32, i32* %.de0005p_386, align 4, !dbg !82
  store i32 %34, i32* %.de0005p.copy_554, align 4, !dbg !82
  %35 = load i32, i32* %.ds0005p_388, align 4, !dbg !82
  store i32 %35, i32* %.ds0005p.copy_555, align 4, !dbg !82
  %36 = load i32, i32* %__gtid___nv_MAIN_F1L26_3__559, align 4, !dbg !82
  %37 = bitcast i32* %.i0004p_334 to i64*, !dbg !82
  %38 = bitcast i32* %.dl0005p.copy_553 to i64*, !dbg !82
  %39 = bitcast i32* %.de0005p.copy_554 to i64*, !dbg !82
  %40 = bitcast i32* %.ds0005p.copy_555 to i64*, !dbg !82
  %41 = load i32, i32* %.ds0005p.copy_555, align 4, !dbg !82
  call void @__kmpc_for_static_init_4(i64* null, i32 %36, i32 34, i64* %37, i64* %38, i64* %39, i64* %40, i32 %41, i32 1), !dbg !82
  %42 = load i32, i32* %.dl0005p.copy_553, align 4, !dbg !82
  store i32 %42, i32* %.dl0005p_390, align 4, !dbg !82
  %43 = load i32, i32* %.de0005p.copy_554, align 4, !dbg !82
  store i32 %43, i32* %.de0005p_386, align 4, !dbg !82
  %44 = load i32, i32* %.ds0005p.copy_555, align 4, !dbg !82
  store i32 %44, i32* %.ds0005p_388, align 4, !dbg !82
  %45 = load i32, i32* %.dl0005p_390, align 4, !dbg !82
  store i32 %45, i32* %i_333, align 4, !dbg !82
  %46 = load i32, i32* %i_333, align 4, !dbg !82
  call void @llvm.dbg.value(metadata i32 %46, metadata !85, metadata !DIExpression()), !dbg !81
  store i32 %46, i32* %.dX0005p_389, align 4, !dbg !82
  %47 = load i32, i32* %.dX0005p_389, align 4, !dbg !82
  %48 = load i32, i32* %.du0005p_385, align 4, !dbg !82
  %49 = icmp sgt i32 %47, %48, !dbg !82
  br i1 %49, label %L.LB7_383, label %L.LB7_573, !dbg !82

L.LB7_573:                                        ; preds = %L.LB7_572
  %50 = load i32, i32* %.dX0005p_389, align 4, !dbg !82
  store i32 %50, i32* %i_333, align 4, !dbg !82
  %51 = load i32, i32* %.di0005p_387, align 4, !dbg !82
  %52 = load i32, i32* %.de0005p_386, align 4, !dbg !82
  %53 = load i32, i32* %.dX0005p_389, align 4, !dbg !82
  %54 = sub nsw i32 %52, %53, !dbg !82
  %55 = add nsw i32 %51, %54, !dbg !82
  %56 = load i32, i32* %.di0005p_387, align 4, !dbg !82
  %57 = sdiv i32 %55, %56, !dbg !82
  store i32 %57, i32* %.dY0005p_384, align 4, !dbg !82
  %58 = load i32, i32* %.dY0005p_384, align 4, !dbg !82
  %59 = icmp sle i32 %58, 0, !dbg !82
  br i1 %59, label %L.LB7_393, label %L.LB7_392, !dbg !82

L.LB7_392:                                        ; preds = %L.LB7_338, %L.LB7_573
  br label %L.LB7_335

L.LB7_335:                                        ; preds = %L.LB7_392
  store i32 17, i32* %.i0005p_337, align 4, !dbg !86
  store i32 16, i32* %.dY0006p_396, align 4, !dbg !86
  call void @llvm.dbg.declare(metadata i32* %j_336, metadata !87, metadata !DIExpression()), !dbg !81
  store i32 1, i32* %j_336, align 4, !dbg !86
  br label %L.LB7_394

L.LB7_394:                                        ; preds = %L.LB7_394, %L.LB7_335
  %60 = load i32, i32* %j_336, align 4, !dbg !88
  call void @llvm.dbg.value(metadata i32 %60, metadata !87, metadata !DIExpression()), !dbg !81
  %61 = sext i32 %60 to i64, !dbg !88
  %62 = bitcast [16 x i32]* %var_332 to i8*, !dbg !88
  %63 = getelementptr i8, i8* %62, i64 -4, !dbg !88
  %64 = bitcast i8* %63 to i32*, !dbg !88
  %65 = getelementptr i32, i32* %64, i64 %61, !dbg !88
  %66 = load i32, i32* %65, align 4, !dbg !88
  %67 = add nsw i32 %66, 1, !dbg !88
  %68 = load i32, i32* %j_336, align 4, !dbg !88
  call void @llvm.dbg.value(metadata i32 %68, metadata !87, metadata !DIExpression()), !dbg !81
  %69 = sext i32 %68 to i64, !dbg !88
  %70 = bitcast [16 x i32]* %var_332 to i8*, !dbg !88
  %71 = getelementptr i8, i8* %70, i64 -4, !dbg !88
  %72 = bitcast i8* %71 to i32*, !dbg !88
  %73 = getelementptr i32, i32* %72, i64 %69, !dbg !88
  store i32 %67, i32* %73, align 4, !dbg !88
  %74 = load i32, i32* %j_336, align 4, !dbg !89
  call void @llvm.dbg.value(metadata i32 %74, metadata !87, metadata !DIExpression()), !dbg !81
  %75 = add nsw i32 %74, 1, !dbg !89
  store i32 %75, i32* %j_336, align 4, !dbg !89
  %76 = load i32, i32* %.dY0006p_396, align 4, !dbg !89
  %77 = sub nsw i32 %76, 1, !dbg !89
  store i32 %77, i32* %.dY0006p_396, align 4, !dbg !89
  %78 = load i32, i32* %.dY0006p_396, align 4, !dbg !89
  %79 = icmp sgt i32 %78, 0, !dbg !89
  br i1 %79, label %L.LB7_394, label %L.LB7_338, !dbg !89

L.LB7_338:                                        ; preds = %L.LB7_394
  %80 = load i32, i32* %.di0005p_387, align 4, !dbg !81
  %81 = load i32, i32* %i_333, align 4, !dbg !81
  call void @llvm.dbg.value(metadata i32 %81, metadata !85, metadata !DIExpression()), !dbg !81
  %82 = add nsw i32 %80, %81, !dbg !81
  store i32 %82, i32* %i_333, align 4, !dbg !81
  %83 = load i32, i32* %.dY0005p_384, align 4, !dbg !81
  %84 = sub nsw i32 %83, 1, !dbg !81
  store i32 %84, i32* %.dY0005p_384, align 4, !dbg !81
  %85 = load i32, i32* %.dY0005p_384, align 4, !dbg !81
  %86 = icmp sgt i32 %85, 0, !dbg !81
  br i1 %86, label %L.LB7_392, label %L.LB7_393, !dbg !81

L.LB7_393:                                        ; preds = %L.LB7_338, %L.LB7_573
  br label %L.LB7_383

L.LB7_383:                                        ; preds = %L.LB7_393, %L.LB7_572
  %87 = load i32, i32* %__gtid___nv_MAIN_F1L26_3__559, align 4, !dbg !81
  call void @__kmpc_for_static_fini(i64* null, i32 %87), !dbg !81
  %88 = call i32 (...) @_mp_bcs_nest_red(), !dbg !81
  %89 = call i32 (...) @_mp_bcs_nest_red(), !dbg !81
  store i64 16, i64* %.dY0007p_399, align 8, !dbg !81
  store i64 1, i64* %"i$d_357", align 8, !dbg !81
  br label %L.LB7_397

L.LB7_397:                                        ; preds = %L.LB7_397, %L.LB7_383
  %90 = load i64, i64* %"i$d_357", align 8, !dbg !81
  call void @llvm.dbg.value(metadata i64 %90, metadata !83, metadata !DIExpression()), !dbg !73
  %91 = bitcast i64* %__nv_MAIN_F1L26_3Arg2 to i8**, !dbg !81
  %92 = load i8*, i8** %91, align 8, !dbg !81
  %93 = getelementptr i8, i8* %92, i64 -4, !dbg !81
  %94 = bitcast i8* %93 to i32*, !dbg !81
  %95 = getelementptr i32, i32* %94, i64 %90, !dbg !81
  %96 = load i32, i32* %95, align 4, !dbg !81
  %97 = load i64, i64* %"i$d_357", align 8, !dbg !81
  call void @llvm.dbg.value(metadata i64 %97, metadata !83, metadata !DIExpression()), !dbg !73
  %98 = bitcast [16 x i32]* %var_332 to i8*, !dbg !81
  %99 = getelementptr i8, i8* %98, i64 -4, !dbg !81
  %100 = bitcast i8* %99 to i32*, !dbg !81
  %101 = getelementptr i32, i32* %100, i64 %97, !dbg !81
  %102 = load i32, i32* %101, align 4, !dbg !81
  %103 = add nsw i32 %96, %102, !dbg !81
  %104 = load i64, i64* %"i$d_357", align 8, !dbg !81
  call void @llvm.dbg.value(metadata i64 %104, metadata !83, metadata !DIExpression()), !dbg !73
  %105 = bitcast i64* %__nv_MAIN_F1L26_3Arg2 to i8**, !dbg !81
  %106 = load i8*, i8** %105, align 8, !dbg !81
  %107 = getelementptr i8, i8* %106, i64 -4, !dbg !81
  %108 = bitcast i8* %107 to i32*, !dbg !81
  %109 = getelementptr i32, i32* %108, i64 %104, !dbg !81
  store i32 %103, i32* %109, align 4, !dbg !81
  %110 = load i64, i64* %"i$d_357", align 8, !dbg !81
  call void @llvm.dbg.value(metadata i64 %110, metadata !83, metadata !DIExpression()), !dbg !73
  %111 = add nsw i64 %110, 1, !dbg !81
  store i64 %111, i64* %"i$d_357", align 8, !dbg !81
  %112 = load i64, i64* %.dY0007p_399, align 8, !dbg !81
  %113 = sub nsw i64 %112, 1, !dbg !81
  store i64 %113, i64* %.dY0007p_399, align 8, !dbg !81
  %114 = load i64, i64* %.dY0007p_399, align 8, !dbg !81
  %115 = icmp sgt i64 %114, 0, !dbg !81
  br i1 %115, label %L.LB7_397, label %L.LB7_574, !dbg !81

L.LB7_574:                                        ; preds = %L.LB7_397
  %116 = call i32 (...) @_mp_ecs_nest_red(), !dbg !81
  %117 = call i32 (...) @_mp_ecs_nest_red(), !dbg !81
  br label %L.LB7_339

L.LB7_339:                                        ; preds = %L.LB7_574
  ret void, !dbg !81
}

declare signext i32 @_mp_ecs_nest_red(...) #1

declare signext i32 @_mp_bcs_nest_red(...) #1

declare void @__kmpc_for_static_fini(i64*, i32) #1

declare void @__kmpc_fork_call(i64*, i32, i64*, i64*, ...) #1

declare void @__kmpc_end_serialized_parallel(i64*, i32) #1

declare void @__kmpc_serialized_parallel(i64*, i32) #1

declare void @__kmpc_for_static_init_4(i64*, i32, i32, i64*, i64*, i64*, i64*, i32, i32) #1

declare void @__kmpc_fork_teams(i64*, i32, i64*, i64*, ...) #1

declare void @_mp_ecs_nest(...) #1

declare signext i32 @f90io_ldw_end(...) #1

declare signext i32 @f90io_sc_i_ldw(...) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

declare signext i32 @f90io_print_init(...) #1

declare void @f90io_src_info03a(...) #1

declare void @_mp_bcs_nest(...) #1

declare void @fort_init(...) #1

declare signext i32 @__kmpc_global_thread_num(i64*) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { noinline }
attributes #1 = { "no-frame-pointer-elim-non-leaf" }
attributes #2 = { nounwind readnone speculatable }

!llvm.module.flags = !{!20, !21}
!llvm.dbg.cu = !{!3}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "var", scope: !2, file: !4, type: !17, isLocal: false, isDefinition: true)
!2 = !DIModule(scope: !3, name: "drb163")
!3 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !4, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !6, imports: !12)
!4 = !DIFile(filename: "micro-benchmarks-fortran/DRB163-simdmissinglock1-orig-gpu-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!5 = !{}
!6 = !{!0, !7, !10}
!7 = !DIGlobalVariableExpression(var: !8, expr: !DIExpression(DW_OP_plus_uconst, 64))
!8 = distinct !DIGlobalVariable(name: "i", scope: !2, file: !4, type: !9, isLocal: false, isDefinition: true)
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DIGlobalVariableExpression(var: !11, expr: !DIExpression(DW_OP_plus_uconst, 68))
!11 = distinct !DIGlobalVariable(name: "j", scope: !2, file: !4, type: !9, isLocal: false, isDefinition: true)
!12 = !{!13}
!13 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !14, entity: !2, file: !4, line: 15)
!14 = distinct !DISubprogram(name: "drb163_simdmissinglock1_orig_gpu_no", scope: !3, file: !4, line: 15, type: !15, scopeLine: 15, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !3)
!15 = !DISubroutineType(cc: DW_CC_program, types: !16)
!16 = !{null}
!17 = !DICompositeType(tag: DW_TAG_array_type, baseType: !9, size: 512, align: 32, elements: !18)
!18 = !{!19}
!19 = !DISubrange(count: 16, lowerBound: 1)
!20 = !{i32 2, !"Dwarf Version", i32 4}
!21 = !{i32 2, !"Debug Info Version", i32 3}
!22 = !DILocalVariable(name: "omp_sched_static", scope: !14, file: !4, type: !9)
!23 = !DILocation(line: 0, scope: !14)
!24 = !DILocalVariable(name: "omp_proc_bind_false", scope: !14, file: !4, type: !9)
!25 = !DILocalVariable(name: "omp_proc_bind_true", scope: !14, file: !4, type: !9)
!26 = !DILocalVariable(name: "omp_lock_hint_none", scope: !14, file: !4, type: !9)
!27 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !14, file: !4, type: !9)
!28 = !DILocation(line: 42, column: 1, scope: !14)
!29 = !DILocation(line: 15, column: 1, scope: !14)
!30 = !DILocation(line: 20, column: 1, scope: !14)
!31 = !DILocation(line: 21, column: 1, scope: !14)
!32 = !DILocation(line: 22, column: 1, scope: !14)
!33 = !DILocation(line: 34, column: 1, scope: !14)
!34 = !DILocation(line: 36, column: 1, scope: !14)
!35 = !DILocation(line: 37, column: 1, scope: !14)
!36 = !DILocation(line: 38, column: 1, scope: !14)
!37 = !DILocalVariable(scope: !14, file: !4, type: !9, flags: DIFlagArtificial)
!38 = !DILocation(line: 40, column: 1, scope: !14)
!39 = distinct !DISubprogram(name: "__nv_MAIN__F1L24_1", scope: !3, file: !4, line: 24, type: !40, scopeLine: 24, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !3)
!40 = !DISubroutineType(types: !41)
!41 = !{null, !9, !42, !42}
!42 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!43 = !DILocalVariable(name: "__nv_MAIN__F1L24_1Arg0", arg: 1, scope: !39, file: !4, type: !9)
!44 = !DILocation(line: 0, scope: !39)
!45 = !DILocalVariable(name: "__nv_MAIN__F1L24_1Arg1", arg: 2, scope: !39, file: !4, type: !42)
!46 = !DILocalVariable(name: "__nv_MAIN__F1L24_1Arg2", arg: 3, scope: !39, file: !4, type: !42)
!47 = !DILocalVariable(name: "omp_sched_static", scope: !39, file: !4, type: !9)
!48 = !DILocalVariable(name: "omp_proc_bind_false", scope: !39, file: !4, type: !9)
!49 = !DILocalVariable(name: "omp_proc_bind_true", scope: !39, file: !4, type: !9)
!50 = !DILocalVariable(name: "omp_lock_hint_none", scope: !39, file: !4, type: !9)
!51 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !39, file: !4, type: !9)
!52 = !DILocation(line: 25, column: 1, scope: !39)
!53 = !DILocation(line: 34, column: 1, scope: !39)
!54 = distinct !DISubprogram(name: "__nv_MAIN_F1L25_2", scope: !3, file: !4, line: 25, type: !40, scopeLine: 25, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !3)
!55 = !DILocalVariable(name: "__nv_MAIN_F1L25_2Arg0", arg: 1, scope: !54, file: !4, type: !9)
!56 = !DILocation(line: 0, scope: !54)
!57 = !DILocalVariable(name: "__nv_MAIN_F1L25_2Arg1", arg: 2, scope: !54, file: !4, type: !42)
!58 = !DILocalVariable(name: "__nv_MAIN_F1L25_2Arg2", arg: 3, scope: !54, file: !4, type: !42)
!59 = !DILocalVariable(name: "omp_sched_static", scope: !54, file: !4, type: !9)
!60 = !DILocalVariable(name: "omp_proc_bind_false", scope: !54, file: !4, type: !9)
!61 = !DILocalVariable(name: "omp_proc_bind_true", scope: !54, file: !4, type: !9)
!62 = !DILocalVariable(name: "omp_lock_hint_none", scope: !54, file: !4, type: !9)
!63 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !54, file: !4, type: !9)
!64 = !DILocation(line: 33, column: 1, scope: !54)
!65 = !DILocation(line: 25, column: 1, scope: !54)
!66 = !DILocalVariable(scope: !54, file: !4, type: !42, flags: DIFlagArtificial)
!67 = !DILocalVariable(name: "var", scope: !54, file: !4, type: !17)
!68 = !DILocation(line: 26, column: 1, scope: !54)
!69 = !DILocalVariable(name: "i", scope: !54, file: !4, type: !9)
!70 = !DILocation(line: 32, column: 1, scope: !54)
!71 = distinct !DISubprogram(name: "__nv_MAIN_F1L26_3", scope: !3, file: !4, line: 26, type: !40, scopeLine: 26, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !3)
!72 = !DILocalVariable(name: "__nv_MAIN_F1L26_3Arg0", arg: 1, scope: !71, file: !4, type: !9)
!73 = !DILocation(line: 0, scope: !71)
!74 = !DILocalVariable(name: "__nv_MAIN_F1L26_3Arg1", arg: 2, scope: !71, file: !4, type: !42)
!75 = !DILocalVariable(name: "__nv_MAIN_F1L26_3Arg2", arg: 3, scope: !71, file: !4, type: !42)
!76 = !DILocalVariable(name: "omp_sched_static", scope: !71, file: !4, type: !9)
!77 = !DILocalVariable(name: "omp_proc_bind_false", scope: !71, file: !4, type: !9)
!78 = !DILocalVariable(name: "omp_proc_bind_true", scope: !71, file: !4, type: !9)
!79 = !DILocalVariable(name: "omp_lock_hint_none", scope: !71, file: !4, type: !9)
!80 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !71, file: !4, type: !9)
!81 = !DILocation(line: 32, column: 1, scope: !71)
!82 = !DILocation(line: 26, column: 1, scope: !71)
!83 = !DILocalVariable(scope: !71, file: !4, type: !42, flags: DIFlagArtificial)
!84 = !DILocalVariable(name: "var", scope: !71, file: !4, type: !17)
!85 = !DILocalVariable(name: "i", scope: !71, file: !4, type: !9)
!86 = !DILocation(line: 28, column: 1, scope: !71)
!87 = !DILocalVariable(name: "j", scope: !71, file: !4, type: !9)
!88 = !DILocation(line: 29, column: 1, scope: !71)
!89 = !DILocation(line: 30, column: 1, scope: !71)

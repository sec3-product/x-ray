; ModuleID = '/tmp/DRB059-lastprivate-orig-no-a54a81.ll'
source_filename = "/tmp/DRB059-lastprivate-orig-no-a54a81.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.STATICS4 = type <{ [40 x i8] }>
%struct.struct_ul_MAIN__297 = type <{ i8* }>
%astruct.dt62 = type <{ i8* }>
%astruct.dt66 = type <{ i8*, i8* }>

@.C283_MAIN_ = internal constant i32 0
@.STATICS4 = internal global %struct.STATICS4 <{ [40 x i8] c"\FB\FF\FF\FF\03\00\00\00x =\00\EB\FF\FF\FF\00\00\00\00\03\00\00\00\00\00\00\00\01\00\00\00\FF\FF\FF\FF\00\00\00\00" }>, align 16
@.C303_drb059_lastprivate_orig_no_foo = internal constant i32 25
@.C321_drb059_lastprivate_orig_no_foo = internal constant i32 6
@.C317_drb059_lastprivate_orig_no_foo = internal constant [55 x i8] c"micro-benchmarks-fortran/DRB059-lastprivate-orig-no.f95"
@.C319_drb059_lastprivate_orig_no_foo = internal constant i32 29
@.C284_drb059_lastprivate_orig_no_foo = internal constant i64 0
@.C313_drb059_lastprivate_orig_no_foo = internal constant i32 100
@.C285_drb059_lastprivate_orig_no_foo = internal constant i32 1
@.C283_drb059_lastprivate_orig_no_foo = internal constant i32 0
@.C284___nv_drb059_lastprivate_orig_no_foo_F1L24_2 = internal constant i64 0
@.C313___nv_drb059_lastprivate_orig_no_foo_F1L24_2 = internal constant i32 100
@.C285___nv_drb059_lastprivate_orig_no_foo_F1L24_2 = internal constant i32 1
@.C283___nv_drb059_lastprivate_orig_no_foo_F1L24_2 = internal constant i32 0

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__328 = alloca i32, align 4
  %.S0000_313 = alloca %struct.struct_ul_MAIN__297, align 8
  %.uplevelArgPack0001_322 = alloca %astruct.dt62, align 8
  call void @llvm.dbg.value(metadata i32 1, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !14, metadata !DIExpression()), !dbg !10
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !15
  store i32 %0, i32* %__gtid_MAIN__328, align 4, !dbg !15
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !16
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !16
  call void (i8*, ...) %2(i8* %1), !dbg !16
  br label %L.LB1_317

L.LB1_317:                                        ; preds = %L.entry
  %3 = bitcast %struct.struct_ul_MAIN__297* %.S0000_313 to i8*, !dbg !17
  %4 = bitcast %astruct.dt62* %.uplevelArgPack0001_322 to i8**, !dbg !17
  store i8* %3, i8** %4, align 8, !dbg !17
  br label %L.LB1_326, !dbg !17

L.LB1_326:                                        ; preds = %L.LB1_317
  %5 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L18_1_ to i64*, !dbg !17
  %6 = bitcast %astruct.dt62* %.uplevelArgPack0001_322 to i64*, !dbg !17
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %5, i64* %6), !dbg !17
  ret void, !dbg !15
}

define internal void @__nv_MAIN__F1L18_1_(i32* %__nv_MAIN__F1L18_1Arg0, i64* %__nv_MAIN__F1L18_1Arg1, i64* %__nv_MAIN__F1L18_1Arg2) #0 !dbg !18 {
L.entry:
  %.S0000_313 = alloca i8*, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L18_1Arg0, metadata !22, metadata !DIExpression()), !dbg !23
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L18_1Arg1, metadata !24, metadata !DIExpression()), !dbg !23
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L18_1Arg2, metadata !25, metadata !DIExpression()), !dbg !23
  call void @llvm.dbg.value(metadata i32 1, metadata !26, metadata !DIExpression()), !dbg !23
  call void @llvm.dbg.value(metadata i32 0, metadata !27, metadata !DIExpression()), !dbg !23
  call void @llvm.dbg.value(metadata i32 1, metadata !28, metadata !DIExpression()), !dbg !23
  call void @llvm.dbg.value(metadata i32 0, metadata !29, metadata !DIExpression()), !dbg !23
  call void @llvm.dbg.value(metadata i32 1, metadata !30, metadata !DIExpression()), !dbg !23
  %0 = bitcast i64* %__nv_MAIN__F1L18_1Arg2 to i8**, !dbg !31
  %1 = load i8*, i8** %0, align 8, !dbg !31
  %2 = bitcast i8** %.S0000_313 to i64*, !dbg !31
  store i8* %1, i8** %.S0000_313, align 8, !dbg !31
  br label %L.LB2_350

L.LB2_350:                                        ; preds = %L.entry
  br label %L.LB2_308

L.LB2_308:                                        ; preds = %L.LB2_350
  %3 = load i8*, i8** %.S0000_313, align 8, !dbg !32
  %4 = bitcast i8* %3 to i64*, !dbg !32
  call void @drb059_lastprivate_orig_no_foo(i64* %4), !dbg !32
  br label %L.LB2_309

L.LB2_309:                                        ; preds = %L.LB2_308
  ret void, !dbg !33
}

define internal void @drb059_lastprivate_orig_no_foo(i64* %.S0000) #0 !dbg !34 {
L.entry:
  %__gtid_drb059_lastprivate_orig_no_foo_361 = alloca i32, align 4
  %.uplevelArgPack0002_352 = alloca %astruct.dt66, align 16
  %x_307 = alloca i32, align 4
  %z__io_323 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i64* %.S0000, metadata !36, metadata !DIExpression()), !dbg !38
  call void @llvm.dbg.value(metadata i32 1, metadata !39, metadata !DIExpression()), !dbg !38
  call void @llvm.dbg.value(metadata i32 0, metadata !40, metadata !DIExpression()), !dbg !38
  call void @llvm.dbg.value(metadata i32 1, metadata !41, metadata !DIExpression()), !dbg !38
  call void @llvm.dbg.value(metadata i32 0, metadata !42, metadata !DIExpression()), !dbg !38
  call void @llvm.dbg.value(metadata i32 1, metadata !43, metadata !DIExpression()), !dbg !38
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !44
  store i32 %0, i32* %__gtid_drb059_lastprivate_orig_no_foo_361, align 4, !dbg !44
  br label %L.LB4_347

L.LB4_347:                                        ; preds = %L.entry
  %1 = bitcast i64* %.S0000 to i8*, !dbg !45
  %2 = bitcast %astruct.dt66* %.uplevelArgPack0002_352 to i8**, !dbg !45
  store i8* %1, i8** %2, align 8, !dbg !45
  call void @llvm.dbg.declare(metadata i32* %x_307, metadata !46, metadata !DIExpression()), !dbg !38
  %3 = bitcast i32* %x_307 to i8*, !dbg !45
  %4 = bitcast %astruct.dt66* %.uplevelArgPack0002_352 to i8*, !dbg !45
  %5 = getelementptr i8, i8* %4, i64 8, !dbg !45
  %6 = bitcast i8* %5 to i8**, !dbg !45
  store i8* %3, i8** %6, align 8, !dbg !45
  br label %L.LB4_359, !dbg !45

L.LB4_359:                                        ; preds = %L.LB4_347
  %7 = bitcast void (i32*, i64*, i64*)* @__nv_drb059_lastprivate_orig_no_foo_F1L24_2_ to i64*, !dbg !45
  %8 = bitcast %astruct.dt66* %.uplevelArgPack0002_352 to i64*, !dbg !45
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %7, i64* %8), !dbg !45
  call void (...) @_mp_bcs_nest(), !dbg !47
  %9 = bitcast i32* @.C319_drb059_lastprivate_orig_no_foo to i8*, !dbg !47
  %10 = bitcast [55 x i8]* @.C317_drb059_lastprivate_orig_no_foo to i8*, !dbg !47
  %11 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !47
  call void (i8*, i8*, i64, ...) %11(i8* %9, i8* %10, i64 55), !dbg !47
  %12 = bitcast i32* @.C321_drb059_lastprivate_orig_no_foo to i8*, !dbg !47
  %13 = bitcast i32* @.C283_drb059_lastprivate_orig_no_foo to i8*, !dbg !47
  %14 = bitcast i32* @.C283_drb059_lastprivate_orig_no_foo to i8*, !dbg !47
  %15 = bitcast %struct.STATICS4* @.STATICS4 to i8*, !dbg !47
  %16 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !47
  %17 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %16(i8* %12, i8* null, i8* %13, i8* %14, i8* %15, i8* null, i64 0), !dbg !47
  call void @llvm.dbg.declare(metadata i32* %z__io_323, metadata !48, metadata !DIExpression()), !dbg !38
  store i32 %17, i32* %z__io_323, align 4, !dbg !47
  %18 = load i32, i32* %x_307, align 4, !dbg !47
  call void @llvm.dbg.value(metadata i32 %18, metadata !46, metadata !DIExpression()), !dbg !38
  %19 = bitcast i32 (...)* @f90io_sc_i_fmt_write to i32 (i32, i32, ...)*, !dbg !47
  %20 = call i32 (i32, i32, ...) %19(i32 %18, i32 25), !dbg !47
  store i32 %20, i32* %z__io_323, align 4, !dbg !47
  %21 = call i32 (...) @f90io_fmtw_end(), !dbg !47
  store i32 %21, i32* %z__io_323, align 4, !dbg !47
  call void (...) @_mp_ecs_nest(), !dbg !47
  ret void, !dbg !44
}

define internal void @__nv_drb059_lastprivate_orig_no_foo_F1L24_2_(i32* %__nv_drb059_lastprivate_orig_no_foo_F1L24_2Arg0, i64* %__nv_drb059_lastprivate_orig_no_foo_F1L24_2Arg1, i64* %__nv_drb059_lastprivate_orig_no_foo_F1L24_2Arg2) #0 !dbg !49 {
L.entry:
  %.S0000_343 = alloca i8*, align 8
  %__gtid___nv_drb059_lastprivate_orig_no_foo_F1L24_2__411 = alloca i32, align 4
  %.i0000p_314 = alloca i32, align 4
  %i_311 = alloca i32, align 4
  %.du0001p_333 = alloca i32, align 4
  %.de0001p_334 = alloca i32, align 4
  %.di0001p_335 = alloca i32, align 4
  %.ds0001p_336 = alloca i32, align 4
  %.dl0001p_338 = alloca i32, align 4
  %.dl0001p.copy_405 = alloca i32, align 4
  %.de0001p.copy_406 = alloca i32, align 4
  %.ds0001p.copy_407 = alloca i32, align 4
  %.dX0001p_337 = alloca i32, align 4
  %.dY0001p_332 = alloca i32, align 4
  %x_312 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_drb059_lastprivate_orig_no_foo_F1L24_2Arg0, metadata !50, metadata !DIExpression()), !dbg !51
  call void @llvm.dbg.declare(metadata i64* %__nv_drb059_lastprivate_orig_no_foo_F1L24_2Arg1, metadata !52, metadata !DIExpression()), !dbg !51
  call void @llvm.dbg.declare(metadata i64* %__nv_drb059_lastprivate_orig_no_foo_F1L24_2Arg2, metadata !53, metadata !DIExpression()), !dbg !51
  call void @llvm.dbg.value(metadata i32 1, metadata !54, metadata !DIExpression()), !dbg !51
  call void @llvm.dbg.value(metadata i32 0, metadata !55, metadata !DIExpression()), !dbg !51
  call void @llvm.dbg.value(metadata i32 1, metadata !56, metadata !DIExpression()), !dbg !51
  call void @llvm.dbg.value(metadata i32 0, metadata !57, metadata !DIExpression()), !dbg !51
  call void @llvm.dbg.value(metadata i32 1, metadata !58, metadata !DIExpression()), !dbg !51
  %0 = bitcast i64* %__nv_drb059_lastprivate_orig_no_foo_F1L24_2Arg2 to i8**, !dbg !59
  %1 = load i8*, i8** %0, align 8, !dbg !59
  %2 = bitcast i8** %.S0000_343 to i64*, !dbg !59
  store i8* %1, i8** %.S0000_343, align 8, !dbg !59
  %3 = load i32, i32* %__nv_drb059_lastprivate_orig_no_foo_F1L24_2Arg0, align 4, !dbg !60
  store i32 %3, i32* %__gtid___nv_drb059_lastprivate_orig_no_foo_F1L24_2__411, align 4, !dbg !60
  br label %L.LB5_397

L.LB5_397:                                        ; preds = %L.entry
  br label %L.LB5_310

L.LB5_310:                                        ; preds = %L.LB5_397
  store i32 0, i32* %.i0000p_314, align 4, !dbg !61
  call void @llvm.dbg.declare(metadata i32* %i_311, metadata !62, metadata !DIExpression()), !dbg !60
  store i32 1, i32* %i_311, align 4, !dbg !61
  store i32 100, i32* %.du0001p_333, align 4, !dbg !61
  store i32 100, i32* %.de0001p_334, align 4, !dbg !61
  store i32 1, i32* %.di0001p_335, align 4, !dbg !61
  %4 = load i32, i32* %.di0001p_335, align 4, !dbg !61
  store i32 %4, i32* %.ds0001p_336, align 4, !dbg !61
  store i32 1, i32* %.dl0001p_338, align 4, !dbg !61
  %5 = load i32, i32* %.dl0001p_338, align 4, !dbg !61
  store i32 %5, i32* %.dl0001p.copy_405, align 4, !dbg !61
  %6 = load i32, i32* %.de0001p_334, align 4, !dbg !61
  store i32 %6, i32* %.de0001p.copy_406, align 4, !dbg !61
  %7 = load i32, i32* %.ds0001p_336, align 4, !dbg !61
  store i32 %7, i32* %.ds0001p.copy_407, align 4, !dbg !61
  %8 = load i32, i32* %__gtid___nv_drb059_lastprivate_orig_no_foo_F1L24_2__411, align 4, !dbg !61
  %9 = bitcast i32* %.i0000p_314 to i64*, !dbg !61
  %10 = bitcast i32* %.dl0001p.copy_405 to i64*, !dbg !61
  %11 = bitcast i32* %.de0001p.copy_406 to i64*, !dbg !61
  %12 = bitcast i32* %.ds0001p.copy_407 to i64*, !dbg !61
  %13 = load i32, i32* %.ds0001p.copy_407, align 4, !dbg !61
  call void @__kmpc_for_static_init_4(i64* null, i32 %8, i32 34, i64* %9, i64* %10, i64* %11, i64* %12, i32 %13, i32 1), !dbg !61
  %14 = load i32, i32* %.dl0001p.copy_405, align 4, !dbg !61
  store i32 %14, i32* %.dl0001p_338, align 4, !dbg !61
  %15 = load i32, i32* %.de0001p.copy_406, align 4, !dbg !61
  store i32 %15, i32* %.de0001p_334, align 4, !dbg !61
  %16 = load i32, i32* %.ds0001p.copy_407, align 4, !dbg !61
  store i32 %16, i32* %.ds0001p_336, align 4, !dbg !61
  %17 = load i32, i32* %.dl0001p_338, align 4, !dbg !61
  store i32 %17, i32* %i_311, align 4, !dbg !61
  %18 = load i32, i32* %i_311, align 4, !dbg !61
  call void @llvm.dbg.value(metadata i32 %18, metadata !62, metadata !DIExpression()), !dbg !60
  store i32 %18, i32* %.dX0001p_337, align 4, !dbg !61
  %19 = load i32, i32* %.dX0001p_337, align 4, !dbg !61
  %20 = load i32, i32* %.du0001p_333, align 4, !dbg !61
  %21 = icmp sgt i32 %19, %20, !dbg !61
  br i1 %21, label %L.LB5_331, label %L.LB5_435, !dbg !61

L.LB5_435:                                        ; preds = %L.LB5_310
  %22 = load i32, i32* %.dX0001p_337, align 4, !dbg !61
  store i32 %22, i32* %i_311, align 4, !dbg !61
  %23 = load i32, i32* %.di0001p_335, align 4, !dbg !61
  %24 = load i32, i32* %.de0001p_334, align 4, !dbg !61
  %25 = load i32, i32* %.dX0001p_337, align 4, !dbg !61
  %26 = sub nsw i32 %24, %25, !dbg !61
  %27 = add nsw i32 %23, %26, !dbg !61
  %28 = load i32, i32* %.di0001p_335, align 4, !dbg !61
  %29 = sdiv i32 %27, %28, !dbg !61
  store i32 %29, i32* %.dY0001p_332, align 4, !dbg !61
  %30 = load i32, i32* %.dY0001p_332, align 4, !dbg !61
  %31 = icmp sle i32 %30, 0, !dbg !61
  br i1 %31, label %L.LB5_341, label %L.LB5_340, !dbg !61

L.LB5_340:                                        ; preds = %L.LB5_340, %L.LB5_435
  %32 = load i32, i32* %i_311, align 4, !dbg !63
  call void @llvm.dbg.value(metadata i32 %32, metadata !62, metadata !DIExpression()), !dbg !60
  call void @llvm.dbg.declare(metadata i32* %x_312, metadata !64, metadata !DIExpression()), !dbg !60
  store i32 %32, i32* %x_312, align 4, !dbg !63
  %33 = load i32, i32* %.di0001p_335, align 4, !dbg !60
  %34 = load i32, i32* %i_311, align 4, !dbg !60
  call void @llvm.dbg.value(metadata i32 %34, metadata !62, metadata !DIExpression()), !dbg !60
  %35 = add nsw i32 %33, %34, !dbg !60
  store i32 %35, i32* %i_311, align 4, !dbg !60
  %36 = load i32, i32* %.dY0001p_332, align 4, !dbg !60
  %37 = sub nsw i32 %36, 1, !dbg !60
  store i32 %37, i32* %.dY0001p_332, align 4, !dbg !60
  %38 = load i32, i32* %.dY0001p_332, align 4, !dbg !60
  %39 = icmp sgt i32 %38, 0, !dbg !60
  br i1 %39, label %L.LB5_340, label %L.LB5_341, !dbg !60

L.LB5_341:                                        ; preds = %L.LB5_340, %L.LB5_435
  br label %L.LB5_331

L.LB5_331:                                        ; preds = %L.LB5_341, %L.LB5_310
  %40 = load i32, i32* %__gtid___nv_drb059_lastprivate_orig_no_foo_F1L24_2__411, align 4, !dbg !60
  call void @__kmpc_for_static_fini(i64* null, i32 %40), !dbg !60
  %41 = load i32, i32* %.i0000p_314, align 4, !dbg !60
  %42 = sext i32 %41 to i64, !dbg !60
  %43 = icmp eq i64 %42, 0, !dbg !60
  br i1 %43, label %L.LB5_342, label %L.LB5_436, !dbg !60

L.LB5_436:                                        ; preds = %L.LB5_331
  %44 = load i32, i32* %x_312, align 4, !dbg !60
  call void @llvm.dbg.value(metadata i32 %44, metadata !64, metadata !DIExpression()), !dbg !60
  %45 = bitcast i64* %__nv_drb059_lastprivate_orig_no_foo_F1L24_2Arg2 to i8*, !dbg !60
  %46 = getelementptr i8, i8* %45, i64 8, !dbg !60
  %47 = bitcast i8* %46 to i32**, !dbg !60
  %48 = load i32*, i32** %47, align 8, !dbg !60
  store i32 %44, i32* %48, align 4, !dbg !60
  br label %L.LB5_342

L.LB5_342:                                        ; preds = %L.LB5_436, %L.LB5_331
  br label %L.LB5_315

L.LB5_315:                                        ; preds = %L.LB5_342
  ret void, !dbg !60
}

declare void @__kmpc_for_static_fini(i64*, i32) #0

declare void @__kmpc_for_static_init_4(i64*, i32, i32, i64*, i64*, i64*, i64*, i32, i32) #0

declare void @_mp_ecs_nest(...) #0

declare signext i32 @f90io_fmtw_end(...) #0

declare signext i32 @f90io_sc_i_fmt_write(...) #0

declare signext i32 @f90io_fmtw_inita(...) #0

declare void @f90io_src_info03a(...) #0

declare void @_mp_bcs_nest(...) #0

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
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB059-lastprivate-orig-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb059_lastprivate_orig_no", scope: !2, file: !3, line: 15, type: !6, scopeLine: 15, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!6 = !DISubroutineType(cc: DW_CC_program, types: !7)
!7 = !{null}
!8 = !DILocalVariable(name: "omp_sched_static", scope: !5, file: !3, type: !9)
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 0, scope: !5)
!11 = !DILocalVariable(name: "omp_proc_bind_false", scope: !5, file: !3, type: !9)
!12 = !DILocalVariable(name: "omp_proc_bind_true", scope: !5, file: !3, type: !9)
!13 = !DILocalVariable(name: "omp_lock_hint_none", scope: !5, file: !3, type: !9)
!14 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !5, file: !3, type: !9)
!15 = !DILocation(line: 21, column: 1, scope: !5)
!16 = !DILocation(line: 15, column: 1, scope: !5)
!17 = !DILocation(line: 18, column: 1, scope: !5)
!18 = distinct !DISubprogram(name: "__nv_MAIN__F1L18_1", scope: !2, file: !3, line: 18, type: !19, scopeLine: 18, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!19 = !DISubroutineType(types: !20)
!20 = !{null, !9, !21, !21}
!21 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!22 = !DILocalVariable(name: "__nv_MAIN__F1L18_1Arg0", arg: 1, scope: !18, file: !3, type: !9)
!23 = !DILocation(line: 0, scope: !18)
!24 = !DILocalVariable(name: "__nv_MAIN__F1L18_1Arg1", arg: 2, scope: !18, file: !3, type: !21)
!25 = !DILocalVariable(name: "__nv_MAIN__F1L18_1Arg2", arg: 3, scope: !18, file: !3, type: !21)
!26 = !DILocalVariable(name: "omp_sched_static", scope: !18, file: !3, type: !9)
!27 = !DILocalVariable(name: "omp_proc_bind_false", scope: !18, file: !3, type: !9)
!28 = !DILocalVariable(name: "omp_proc_bind_true", scope: !18, file: !3, type: !9)
!29 = !DILocalVariable(name: "omp_lock_hint_none", scope: !18, file: !3, type: !9)
!30 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !18, file: !3, type: !9)
!31 = !DILocation(line: 18, column: 1, scope: !18)
!32 = !DILocation(line: 19, column: 1, scope: !18)
!33 = !DILocation(line: 20, column: 1, scope: !18)
!34 = distinct !DISubprogram(name: "foo", scope: !5, file: !3, line: 22, type: !35, scopeLine: 22, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!35 = !DISubroutineType(types: !7)
!36 = !DILocalVariable(arg: 1, scope: !34, file: !3, type: !37, flags: DIFlagArtificial)
!37 = !DIBasicType(name: "uinteger*8", size: 64, align: 64, encoding: DW_ATE_unsigned)
!38 = !DILocation(line: 0, scope: !34)
!39 = !DILocalVariable(name: "omp_sched_static", scope: !34, file: !3, type: !9)
!40 = !DILocalVariable(name: "omp_proc_bind_false", scope: !34, file: !3, type: !9)
!41 = !DILocalVariable(name: "omp_proc_bind_true", scope: !34, file: !3, type: !9)
!42 = !DILocalVariable(name: "omp_lock_hint_none", scope: !34, file: !3, type: !9)
!43 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !34, file: !3, type: !9)
!44 = !DILocation(line: 31, column: 1, scope: !34)
!45 = !DILocation(line: 24, column: 1, scope: !34)
!46 = !DILocalVariable(name: "x", scope: !34, file: !3, type: !9)
!47 = !DILocation(line: 29, column: 1, scope: !34)
!48 = !DILocalVariable(scope: !34, file: !3, type: !9, flags: DIFlagArtificial)
!49 = distinct !DISubprogram(name: "__nv_drb059_lastprivate_orig_no_foo_F1L24_2", scope: !2, file: !3, line: 24, type: !19, scopeLine: 24, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!50 = !DILocalVariable(name: "__nv_drb059_lastprivate_orig_no_foo_F1L24_2Arg0", arg: 1, scope: !49, file: !3, type: !9)
!51 = !DILocation(line: 0, scope: !49)
!52 = !DILocalVariable(name: "__nv_drb059_lastprivate_orig_no_foo_F1L24_2Arg1", arg: 2, scope: !49, file: !3, type: !21)
!53 = !DILocalVariable(name: "__nv_drb059_lastprivate_orig_no_foo_F1L24_2Arg2", arg: 3, scope: !49, file: !3, type: !21)
!54 = !DILocalVariable(name: "omp_sched_static", scope: !49, file: !3, type: !9)
!55 = !DILocalVariable(name: "omp_proc_bind_false", scope: !49, file: !3, type: !9)
!56 = !DILocalVariable(name: "omp_proc_bind_true", scope: !49, file: !3, type: !9)
!57 = !DILocalVariable(name: "omp_lock_hint_none", scope: !49, file: !3, type: !9)
!58 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !49, file: !3, type: !9)
!59 = !DILocation(line: 24, column: 1, scope: !49)
!60 = !DILocation(line: 27, column: 1, scope: !49)
!61 = !DILocation(line: 25, column: 1, scope: !49)
!62 = !DILocalVariable(name: "i", scope: !49, file: !3, type: !9)
!63 = !DILocation(line: 26, column: 1, scope: !49)
!64 = !DILocalVariable(name: "x", scope: !49, file: !3, type: !9)

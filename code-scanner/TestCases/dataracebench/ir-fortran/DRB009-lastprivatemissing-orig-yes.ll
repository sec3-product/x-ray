; ModuleID = '/tmp/DRB009-lastprivatemissing-orig-yes-acadbb.ll'
source_filename = "/tmp/DRB009-lastprivatemissing-orig-yes-acadbb.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%astruct.dt60 = type <{ i8*, i8* }>

@.C305_MAIN_ = internal constant i32 14
@.C321_MAIN_ = internal constant [3 x i8] c"x ="
@.C284_MAIN_ = internal constant i64 0
@.C320_MAIN_ = internal constant i32 6
@.C318_MAIN_ = internal constant [63 x i8] c"micro-benchmarks-fortran/DRB009-lastprivatemissing-orig-yes.f95"
@.C306_MAIN_ = internal constant i32 25
@.C285_MAIN_ = internal constant i32 1
@.C309_MAIN_ = internal constant i32 10000
@.C283_MAIN_ = internal constant i32 0
@.C285___nv_MAIN__F1L19_1 = internal constant i32 1
@.C283___nv_MAIN__F1L19_1 = internal constant i32 0

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__359 = alloca i32, align 4
  %len_310 = alloca i32, align 4
  %.uplevelArgPack0001_351 = alloca %astruct.dt60, align 16
  %x_308 = alloca i32, align 4
  %z__io_323 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !14, metadata !DIExpression()), !dbg !10
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !15
  store i32 %0, i32* %__gtid_MAIN__359, align 4, !dbg !15
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !16
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !16
  call void (i8*, ...) %2(i8* %1), !dbg !16
  br label %L.LB1_345

L.LB1_345:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %len_310, metadata !17, metadata !DIExpression()), !dbg !10
  store i32 10000, i32* %len_310, align 4, !dbg !18
  %3 = bitcast i32* %len_310 to i8*, !dbg !19
  %4 = bitcast %astruct.dt60* %.uplevelArgPack0001_351 to i8**, !dbg !19
  store i8* %3, i8** %4, align 8, !dbg !19
  call void @llvm.dbg.declare(metadata i32* %x_308, metadata !20, metadata !DIExpression()), !dbg !10
  %5 = bitcast i32* %x_308 to i8*, !dbg !19
  %6 = bitcast %astruct.dt60* %.uplevelArgPack0001_351 to i8*, !dbg !19
  %7 = getelementptr i8, i8* %6, i64 8, !dbg !19
  %8 = bitcast i8* %7 to i8**, !dbg !19
  store i8* %5, i8** %8, align 8, !dbg !19
  br label %L.LB1_357, !dbg !19

L.LB1_357:                                        ; preds = %L.LB1_345
  %9 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L19_1_ to i64*, !dbg !19
  %10 = bitcast %astruct.dt60* %.uplevelArgPack0001_351 to i64*, !dbg !19
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %9, i64* %10), !dbg !19
  call void (...) @_mp_bcs_nest(), !dbg !21
  %11 = bitcast i32* @.C306_MAIN_ to i8*, !dbg !21
  %12 = bitcast [63 x i8]* @.C318_MAIN_ to i8*, !dbg !21
  %13 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !21
  call void (i8*, i8*, i64, ...) %13(i8* %11, i8* %12, i64 63), !dbg !21
  %14 = bitcast i32* @.C320_MAIN_ to i8*, !dbg !21
  %15 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !21
  %16 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !21
  %17 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !21
  %18 = call i32 (i8*, i8*, i8*, i8*, ...) %17(i8* %14, i8* null, i8* %15, i8* %16), !dbg !21
  call void @llvm.dbg.declare(metadata i32* %z__io_323, metadata !22, metadata !DIExpression()), !dbg !10
  store i32 %18, i32* %z__io_323, align 4, !dbg !21
  %19 = bitcast [3 x i8]* @.C321_MAIN_ to i8*, !dbg !21
  %20 = bitcast i32 (...)* @f90io_sc_ch_ldw to i32 (i8*, i32, i64, ...)*, !dbg !21
  %21 = call i32 (i8*, i32, i64, ...) %20(i8* %19, i32 14, i64 3), !dbg !21
  store i32 %21, i32* %z__io_323, align 4, !dbg !21
  %22 = load i32, i32* %x_308, align 4, !dbg !21
  call void @llvm.dbg.value(metadata i32 %22, metadata !20, metadata !DIExpression()), !dbg !10
  %23 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !21
  %24 = call i32 (i32, i32, ...) %23(i32 %22, i32 25), !dbg !21
  store i32 %24, i32* %z__io_323, align 4, !dbg !21
  %25 = call i32 (...) @f90io_ldw_end(), !dbg !21
  store i32 %25, i32* %z__io_323, align 4, !dbg !21
  call void (...) @_mp_ecs_nest(), !dbg !21
  ret void, !dbg !15
}

define internal void @__nv_MAIN__F1L19_1_(i32* %__nv_MAIN__F1L19_1Arg0, i64* %__nv_MAIN__F1L19_1Arg1, i64* %__nv_MAIN__F1L19_1Arg2) #0 !dbg !23 {
L.entry:
  %__gtid___nv_MAIN__F1L19_1__405 = alloca i32, align 4
  %.i0000p_315 = alloca i32, align 4
  %i_314 = alloca i32, align 4
  %.du0001p_335 = alloca i32, align 4
  %.de0001p_336 = alloca i32, align 4
  %.di0001p_337 = alloca i32, align 4
  %.ds0001p_338 = alloca i32, align 4
  %.dl0001p_340 = alloca i32, align 4
  %.dl0001p.copy_399 = alloca i32, align 4
  %.de0001p.copy_400 = alloca i32, align 4
  %.ds0001p.copy_401 = alloca i32, align 4
  %.dX0001p_339 = alloca i32, align 4
  %.dY0001p_334 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L19_1Arg0, metadata !27, metadata !DIExpression()), !dbg !28
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L19_1Arg1, metadata !29, metadata !DIExpression()), !dbg !28
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L19_1Arg2, metadata !30, metadata !DIExpression()), !dbg !28
  call void @llvm.dbg.value(metadata i32 1, metadata !31, metadata !DIExpression()), !dbg !28
  call void @llvm.dbg.value(metadata i32 0, metadata !32, metadata !DIExpression()), !dbg !28
  call void @llvm.dbg.value(metadata i32 1, metadata !33, metadata !DIExpression()), !dbg !28
  call void @llvm.dbg.value(metadata i32 0, metadata !34, metadata !DIExpression()), !dbg !28
  call void @llvm.dbg.value(metadata i32 1, metadata !35, metadata !DIExpression()), !dbg !28
  %0 = load i32, i32* %__nv_MAIN__F1L19_1Arg0, align 4, !dbg !36
  store i32 %0, i32* %__gtid___nv_MAIN__F1L19_1__405, align 4, !dbg !36
  br label %L.LB2_389

L.LB2_389:                                        ; preds = %L.entry
  br label %L.LB2_313

L.LB2_313:                                        ; preds = %L.LB2_389
  store i32 0, i32* %.i0000p_315, align 4, !dbg !37
  call void @llvm.dbg.declare(metadata i32* %i_314, metadata !38, metadata !DIExpression()), !dbg !36
  store i32 0, i32* %i_314, align 4, !dbg !37
  %1 = bitcast i64* %__nv_MAIN__F1L19_1Arg2 to i32**, !dbg !37
  %2 = load i32*, i32** %1, align 8, !dbg !37
  %3 = load i32, i32* %2, align 4, !dbg !37
  store i32 %3, i32* %.du0001p_335, align 4, !dbg !37
  %4 = bitcast i64* %__nv_MAIN__F1L19_1Arg2 to i32**, !dbg !37
  %5 = load i32*, i32** %4, align 8, !dbg !37
  %6 = load i32, i32* %5, align 4, !dbg !37
  store i32 %6, i32* %.de0001p_336, align 4, !dbg !37
  store i32 1, i32* %.di0001p_337, align 4, !dbg !37
  %7 = load i32, i32* %.di0001p_337, align 4, !dbg !37
  store i32 %7, i32* %.ds0001p_338, align 4, !dbg !37
  store i32 0, i32* %.dl0001p_340, align 4, !dbg !37
  %8 = load i32, i32* %.dl0001p_340, align 4, !dbg !37
  store i32 %8, i32* %.dl0001p.copy_399, align 4, !dbg !37
  %9 = load i32, i32* %.de0001p_336, align 4, !dbg !37
  store i32 %9, i32* %.de0001p.copy_400, align 4, !dbg !37
  %10 = load i32, i32* %.ds0001p_338, align 4, !dbg !37
  store i32 %10, i32* %.ds0001p.copy_401, align 4, !dbg !37
  %11 = load i32, i32* %__gtid___nv_MAIN__F1L19_1__405, align 4, !dbg !37
  %12 = bitcast i32* %.i0000p_315 to i64*, !dbg !37
  %13 = bitcast i32* %.dl0001p.copy_399 to i64*, !dbg !37
  %14 = bitcast i32* %.de0001p.copy_400 to i64*, !dbg !37
  %15 = bitcast i32* %.ds0001p.copy_401 to i64*, !dbg !37
  %16 = load i32, i32* %.ds0001p.copy_401, align 4, !dbg !37
  call void @__kmpc_for_static_init_4(i64* null, i32 %11, i32 34, i64* %12, i64* %13, i64* %14, i64* %15, i32 %16, i32 1), !dbg !37
  %17 = load i32, i32* %.dl0001p.copy_399, align 4, !dbg !37
  store i32 %17, i32* %.dl0001p_340, align 4, !dbg !37
  %18 = load i32, i32* %.de0001p.copy_400, align 4, !dbg !37
  store i32 %18, i32* %.de0001p_336, align 4, !dbg !37
  %19 = load i32, i32* %.ds0001p.copy_401, align 4, !dbg !37
  store i32 %19, i32* %.ds0001p_338, align 4, !dbg !37
  %20 = load i32, i32* %.dl0001p_340, align 4, !dbg !37
  store i32 %20, i32* %i_314, align 4, !dbg !37
  %21 = load i32, i32* %i_314, align 4, !dbg !37
  call void @llvm.dbg.value(metadata i32 %21, metadata !38, metadata !DIExpression()), !dbg !36
  store i32 %21, i32* %.dX0001p_339, align 4, !dbg !37
  %22 = load i32, i32* %.dX0001p_339, align 4, !dbg !37
  %23 = load i32, i32* %.du0001p_335, align 4, !dbg !37
  %24 = icmp sgt i32 %22, %23, !dbg !37
  br i1 %24, label %L.LB2_333, label %L.LB2_428, !dbg !37

L.LB2_428:                                        ; preds = %L.LB2_313
  %25 = load i32, i32* %.dX0001p_339, align 4, !dbg !37
  store i32 %25, i32* %i_314, align 4, !dbg !37
  %26 = load i32, i32* %.di0001p_337, align 4, !dbg !37
  %27 = load i32, i32* %.de0001p_336, align 4, !dbg !37
  %28 = load i32, i32* %.dX0001p_339, align 4, !dbg !37
  %29 = sub nsw i32 %27, %28, !dbg !37
  %30 = add nsw i32 %26, %29, !dbg !37
  %31 = load i32, i32* %.di0001p_337, align 4, !dbg !37
  %32 = sdiv i32 %30, %31, !dbg !37
  store i32 %32, i32* %.dY0001p_334, align 4, !dbg !37
  %33 = load i32, i32* %.dY0001p_334, align 4, !dbg !37
  %34 = icmp sle i32 %33, 0, !dbg !37
  br i1 %34, label %L.LB2_343, label %L.LB2_342, !dbg !37

L.LB2_342:                                        ; preds = %L.LB2_342, %L.LB2_428
  %35 = load i32, i32* %i_314, align 4, !dbg !39
  call void @llvm.dbg.value(metadata i32 %35, metadata !38, metadata !DIExpression()), !dbg !36
  %36 = bitcast i64* %__nv_MAIN__F1L19_1Arg2 to i8*, !dbg !39
  %37 = getelementptr i8, i8* %36, i64 8, !dbg !39
  %38 = bitcast i8* %37 to i32**, !dbg !39
  %39 = load i32*, i32** %38, align 8, !dbg !39
  store i32 %35, i32* %39, align 4, !dbg !39
  %40 = load i32, i32* %.di0001p_337, align 4, !dbg !36
  %41 = load i32, i32* %i_314, align 4, !dbg !36
  call void @llvm.dbg.value(metadata i32 %41, metadata !38, metadata !DIExpression()), !dbg !36
  %42 = add nsw i32 %40, %41, !dbg !36
  store i32 %42, i32* %i_314, align 4, !dbg !36
  %43 = load i32, i32* %.dY0001p_334, align 4, !dbg !36
  %44 = sub nsw i32 %43, 1, !dbg !36
  store i32 %44, i32* %.dY0001p_334, align 4, !dbg !36
  %45 = load i32, i32* %.dY0001p_334, align 4, !dbg !36
  %46 = icmp sgt i32 %45, 0, !dbg !36
  br i1 %46, label %L.LB2_342, label %L.LB2_343, !dbg !36

L.LB2_343:                                        ; preds = %L.LB2_342, %L.LB2_428
  br label %L.LB2_333

L.LB2_333:                                        ; preds = %L.LB2_343, %L.LB2_313
  %47 = load i32, i32* %__gtid___nv_MAIN__F1L19_1__405, align 4, !dbg !36
  call void @__kmpc_for_static_fini(i64* null, i32 %47), !dbg !36
  br label %L.LB2_316

L.LB2_316:                                        ; preds = %L.LB2_333
  ret void, !dbg !36
}

declare void @__kmpc_for_static_fini(i64*, i32) #0

declare void @__kmpc_for_static_init_4(i64*, i32, i32, i64*, i64*, i64*, i64*, i32, i32) #0

declare void @_mp_ecs_nest(...) #0

declare signext i32 @f90io_ldw_end(...) #0

declare signext i32 @f90io_sc_i_ldw(...) #0

declare signext i32 @f90io_sc_ch_ldw(...) #0

declare signext i32 @f90io_print_init(...) #0

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
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB009-lastprivatemissing-orig-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb009_lastprivatemissing_orig_yes", scope: !2, file: !3, line: 12, type: !6, scopeLine: 12, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!6 = !DISubroutineType(cc: DW_CC_program, types: !7)
!7 = !{null}
!8 = !DILocalVariable(name: "omp_sched_static", scope: !5, file: !3, type: !9)
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 0, scope: !5)
!11 = !DILocalVariable(name: "omp_proc_bind_false", scope: !5, file: !3, type: !9)
!12 = !DILocalVariable(name: "omp_proc_bind_true", scope: !5, file: !3, type: !9)
!13 = !DILocalVariable(name: "omp_lock_hint_none", scope: !5, file: !3, type: !9)
!14 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !5, file: !3, type: !9)
!15 = !DILocation(line: 26, column: 1, scope: !5)
!16 = !DILocation(line: 12, column: 1, scope: !5)
!17 = !DILocalVariable(name: "len", scope: !5, file: !3, type: !9)
!18 = !DILocation(line: 17, column: 1, scope: !5)
!19 = !DILocation(line: 19, column: 1, scope: !5)
!20 = !DILocalVariable(name: "x", scope: !5, file: !3, type: !9)
!21 = !DILocation(line: 25, column: 1, scope: !5)
!22 = !DILocalVariable(scope: !5, file: !3, type: !9, flags: DIFlagArtificial)
!23 = distinct !DISubprogram(name: "__nv_MAIN__F1L19_1", scope: !2, file: !3, line: 19, type: !24, scopeLine: 19, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!24 = !DISubroutineType(types: !25)
!25 = !{null, !9, !26, !26}
!26 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!27 = !DILocalVariable(name: "__nv_MAIN__F1L19_1Arg0", arg: 1, scope: !23, file: !3, type: !9)
!28 = !DILocation(line: 0, scope: !23)
!29 = !DILocalVariable(name: "__nv_MAIN__F1L19_1Arg1", arg: 2, scope: !23, file: !3, type: !26)
!30 = !DILocalVariable(name: "__nv_MAIN__F1L19_1Arg2", arg: 3, scope: !23, file: !3, type: !26)
!31 = !DILocalVariable(name: "omp_sched_static", scope: !23, file: !3, type: !9)
!32 = !DILocalVariable(name: "omp_proc_bind_false", scope: !23, file: !3, type: !9)
!33 = !DILocalVariable(name: "omp_proc_bind_true", scope: !23, file: !3, type: !9)
!34 = !DILocalVariable(name: "omp_lock_hint_none", scope: !23, file: !3, type: !9)
!35 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !23, file: !3, type: !9)
!36 = !DILocation(line: 22, column: 1, scope: !23)
!37 = !DILocation(line: 20, column: 1, scope: !23)
!38 = !DILocalVariable(name: "i", scope: !23, file: !3, type: !9)
!39 = !DILocation(line: 21, column: 1, scope: !23)

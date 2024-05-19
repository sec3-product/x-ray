; ModuleID = '/tmp/DRB110-ordered-orig-no-a1dbbc.ll'
source_filename = "/tmp/DRB110-ordered-orig-no-a1dbbc.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%astruct.dt66 = type <{ i8* }>

@.C306_MAIN_ = internal constant i32 25
@.C305_MAIN_ = internal constant i32 14
@.C323_MAIN_ = internal constant [3 x i8] c"x ="
@.C284_MAIN_ = internal constant i64 0
@.C320_MAIN_ = internal constant i32 6
@.C318_MAIN_ = internal constant [51 x i8] c"micro-benchmarks-fortran/DRB110-ordered-orig-no.f95"
@.C307_MAIN_ = internal constant i32 27
@.C313_MAIN_ = internal constant i32 100
@.C285_MAIN_ = internal constant i32 1
@.C283_MAIN_ = internal constant i32 0
@.C313___nv_MAIN__F1L19_1 = internal constant i32 100
@.C285___nv_MAIN__F1L19_1 = internal constant i32 1
@.C283___nv_MAIN__F1L19_1 = internal constant i32 0

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__361 = alloca i32, align 4
  %x_308 = alloca i32, align 4
  %.uplevelArgPack0001_356 = alloca %astruct.dt66, align 8
  %z__io_322 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !14, metadata !DIExpression()), !dbg !10
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !15
  store i32 %0, i32* %__gtid_MAIN__361, align 4, !dbg !15
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !16
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !16
  call void (i8*, ...) %2(i8* %1), !dbg !16
  br label %L.LB1_350

L.LB1_350:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %x_308, metadata !17, metadata !DIExpression()), !dbg !10
  store i32 0, i32* %x_308, align 4, !dbg !18
  %3 = bitcast i32* %x_308 to i8*, !dbg !19
  %4 = bitcast %astruct.dt66* %.uplevelArgPack0001_356 to i8**, !dbg !19
  store i8* %3, i8** %4, align 8, !dbg !19
  br label %L.LB1_359, !dbg !19

L.LB1_359:                                        ; preds = %L.LB1_350
  %5 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L19_1_ to i64*, !dbg !19
  %6 = bitcast %astruct.dt66* %.uplevelArgPack0001_356 to i64*, !dbg !19
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %5, i64* %6), !dbg !19
  call void (...) @_mp_bcs_nest(), !dbg !20
  %7 = bitcast i32* @.C307_MAIN_ to i8*, !dbg !20
  %8 = bitcast [51 x i8]* @.C318_MAIN_ to i8*, !dbg !20
  %9 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !20
  call void (i8*, i8*, i64, ...) %9(i8* %7, i8* %8, i64 51), !dbg !20
  %10 = bitcast i32* @.C320_MAIN_ to i8*, !dbg !20
  %11 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !20
  %12 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !20
  %13 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !20
  %14 = call i32 (i8*, i8*, i8*, i8*, ...) %13(i8* %10, i8* null, i8* %11, i8* %12), !dbg !20
  call void @llvm.dbg.declare(metadata i32* %z__io_322, metadata !21, metadata !DIExpression()), !dbg !10
  store i32 %14, i32* %z__io_322, align 4, !dbg !20
  %15 = bitcast [3 x i8]* @.C323_MAIN_ to i8*, !dbg !20
  %16 = bitcast i32 (...)* @f90io_sc_ch_ldw to i32 (i8*, i32, i64, ...)*, !dbg !20
  %17 = call i32 (i8*, i32, i64, ...) %16(i8* %15, i32 14, i64 3), !dbg !20
  store i32 %17, i32* %z__io_322, align 4, !dbg !20
  %18 = load i32, i32* %x_308, align 4, !dbg !20
  call void @llvm.dbg.value(metadata i32 %18, metadata !17, metadata !DIExpression()), !dbg !10
  %19 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !20
  %20 = call i32 (i32, i32, ...) %19(i32 %18, i32 25), !dbg !20
  store i32 %20, i32* %z__io_322, align 4, !dbg !20
  %21 = call i32 (...) @f90io_ldw_end(), !dbg !20
  store i32 %21, i32* %z__io_322, align 4, !dbg !20
  call void (...) @_mp_ecs_nest(), !dbg !20
  ret void, !dbg !15
}

define internal void @__nv_MAIN__F1L19_1_(i32* %__nv_MAIN__F1L19_1Arg0, i64* %__nv_MAIN__F1L19_1Arg1, i64* %__nv_MAIN__F1L19_1Arg2) #0 !dbg !22 {
L.entry:
  %__gtid___nv_MAIN__F1L19_1__408 = alloca i32, align 4
  %.i0000p_315 = alloca i32, align 4
  %i_314 = alloca i32, align 4
  %.dY0001p_334 = alloca i32, align 4
  %.du0001p_339 = alloca i32, align 4
  %.de0001p_340 = alloca i32, align 4
  %.di0001p_341 = alloca i32, align 4
  %.ds0001p_342 = alloca i32, align 4
  %.dx0001p_344 = alloca i32, align 4
  %.dl0001p_345 = alloca i32, align 4
  %.dU0001p_346 = alloca i32, align 4
  %.dl0001p.copy_402 = alloca i32, align 4
  %.dU0001p.copy_403 = alloca i32, align 4
  %.ds0001p.copy_404 = alloca i32, align 4
  %.dX0001p_343 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L19_1Arg0, metadata !26, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L19_1Arg1, metadata !28, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L19_1Arg2, metadata !29, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.value(metadata i32 1, metadata !30, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.value(metadata i32 0, metadata !31, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.value(metadata i32 1, metadata !32, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.value(metadata i32 0, metadata !33, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.value(metadata i32 1, metadata !34, metadata !DIExpression()), !dbg !27
  %0 = load i32, i32* %__nv_MAIN__F1L19_1Arg0, align 4, !dbg !35
  store i32 %0, i32* %__gtid___nv_MAIN__F1L19_1__408, align 4, !dbg !35
  br label %L.LB2_391

L.LB2_391:                                        ; preds = %L.entry
  br label %L.LB2_312

L.LB2_312:                                        ; preds = %L.LB2_391
  store i32 0, i32* %.i0000p_315, align 4, !dbg !36
  call void @llvm.dbg.declare(metadata i32* %i_314, metadata !37, metadata !DIExpression()), !dbg !35
  store i32 1, i32* %i_314, align 4, !dbg !36
  store i32 100, i32* %.dY0001p_334, align 4, !dbg !36
  store i32 1, i32* %i_314, align 4, !dbg !36
  store i32 100, i32* %.du0001p_339, align 4, !dbg !36
  store i32 100, i32* %.de0001p_340, align 4, !dbg !36
  store i32 1, i32* %.di0001p_341, align 4, !dbg !36
  %1 = load i32, i32* %.di0001p_341, align 4, !dbg !36
  store i32 %1, i32* %.ds0001p_342, align 4, !dbg !36
  store i32 1, i32* %.dx0001p_344, align 4, !dbg !36
  store i32 1, i32* %.dl0001p_345, align 4, !dbg !36
  store i32 100, i32* %.dU0001p_346, align 4, !dbg !36
  %2 = load i32, i32* %.dl0001p_345, align 4, !dbg !36
  store i32 %2, i32* %.dl0001p.copy_402, align 4, !dbg !36
  %3 = load i32, i32* %.dU0001p_346, align 4, !dbg !36
  store i32 %3, i32* %.dU0001p.copy_403, align 4, !dbg !36
  %4 = load i32, i32* %.ds0001p_342, align 4, !dbg !36
  store i32 %4, i32* %.ds0001p.copy_404, align 4, !dbg !36
  %5 = load i32, i32* %__gtid___nv_MAIN__F1L19_1__408, align 4, !dbg !36
  %6 = load i32, i32* %.dl0001p.copy_402, align 4, !dbg !36
  %7 = load i32, i32* %.dU0001p.copy_403, align 4, !dbg !36
  %8 = load i32, i32* %.ds0001p.copy_404, align 4, !dbg !36
  call void @__kmpc_dispatch_init_4(i64* null, i32 %5, i32 66, i32 %6, i32 %7, i32 %8, i32 0), !dbg !36
  %9 = load i32, i32* %.dl0001p.copy_402, align 4, !dbg !36
  store i32 %9, i32* %.dl0001p_345, align 4, !dbg !36
  %10 = load i32, i32* %.dU0001p.copy_403, align 4, !dbg !36
  store i32 %10, i32* %.dU0001p_346, align 4, !dbg !36
  %11 = load i32, i32* %.ds0001p.copy_404, align 4, !dbg !36
  store i32 %11, i32* %.ds0001p_342, align 4, !dbg !36
  br label %L.LB2_332

L.LB2_332:                                        ; preds = %L.LB2_348, %L.LB2_312
  %12 = load i32, i32* %__gtid___nv_MAIN__F1L19_1__408, align 4, !dbg !36
  %13 = bitcast i32* %.i0000p_315 to i64*, !dbg !36
  %14 = bitcast i32* %.dx0001p_344 to i64*, !dbg !36
  %15 = bitcast i32* %.de0001p_340 to i64*, !dbg !36
  %16 = bitcast i32* %.ds0001p_342 to i64*, !dbg !36
  %17 = call i32 @__kmpc_dispatch_next_4(i64* null, i32 %12, i64* %13, i64* %14, i64* %15, i64* %16), !dbg !36
  %18 = icmp eq i32 %17, 0, !dbg !36
  br i1 %18, label %L.LB2_333, label %L.LB2_442, !dbg !36

L.LB2_442:                                        ; preds = %L.LB2_332
  %19 = load i32, i32* %.dx0001p_344, align 4, !dbg !36
  store i32 %19, i32* %.dX0001p_343, align 4, !dbg !36
  %20 = load i32, i32* %.dX0001p_343, align 4, !dbg !36
  store i32 %20, i32* %i_314, align 4, !dbg !36
  %21 = load i32, i32* %.ds0001p_342, align 4, !dbg !36
  %22 = load i32, i32* %.de0001p_340, align 4, !dbg !36
  %23 = load i32, i32* %.dX0001p_343, align 4, !dbg !36
  %24 = sub nsw i32 %22, %23, !dbg !36
  %25 = add nsw i32 %21, %24, !dbg !36
  %26 = load i32, i32* %.ds0001p_342, align 4, !dbg !36
  %27 = sdiv i32 %25, %26, !dbg !36
  store i32 %27, i32* %.dY0001p_334, align 4, !dbg !36
  %28 = load i32, i32* %.dY0001p_334, align 4, !dbg !36
  %29 = icmp sle i32 %28, 0, !dbg !36
  br i1 %29, label %L.LB2_348, label %L.LB2_347, !dbg !36

L.LB2_347:                                        ; preds = %L.LB2_347, %L.LB2_442
  %30 = load i32, i32* %__gtid___nv_MAIN__F1L19_1__408, align 4, !dbg !38
  call void @__kmpc_ordered(i64* null, i32 %30), !dbg !38
  %31 = bitcast i64* %__nv_MAIN__F1L19_1Arg2 to i32**, !dbg !39
  %32 = load i32*, i32** %31, align 8, !dbg !39
  %33 = load i32, i32* %32, align 4, !dbg !39
  %34 = add nsw i32 %33, 1, !dbg !39
  %35 = bitcast i64* %__nv_MAIN__F1L19_1Arg2 to i32**, !dbg !39
  %36 = load i32*, i32** %35, align 8, !dbg !39
  store i32 %34, i32* %36, align 4, !dbg !39
  %37 = load i32, i32* %__gtid___nv_MAIN__F1L19_1__408, align 4, !dbg !40
  call void @__kmpc_end_ordered(i64* null, i32 %37), !dbg !40
  %38 = load i32, i32* %__gtid___nv_MAIN__F1L19_1__408, align 4, !dbg !35
  call void @__kmpc_dispatch_fini_4(i64* null, i32 %38), !dbg !35
  %39 = load i32, i32* %.ds0001p_342, align 4, !dbg !35
  %40 = load i32, i32* %i_314, align 4, !dbg !35
  call void @llvm.dbg.value(metadata i32 %40, metadata !37, metadata !DIExpression()), !dbg !35
  %41 = add nsw i32 %39, %40, !dbg !35
  store i32 %41, i32* %i_314, align 4, !dbg !35
  %42 = load i32, i32* %.dY0001p_334, align 4, !dbg !35
  %43 = sub nsw i32 %42, 1, !dbg !35
  store i32 %43, i32* %.dY0001p_334, align 4, !dbg !35
  %44 = load i32, i32* %.dY0001p_334, align 4, !dbg !35
  %45 = icmp sgt i32 %44, 0, !dbg !35
  br i1 %45, label %L.LB2_347, label %L.LB2_348, !dbg !35

L.LB2_348:                                        ; preds = %L.LB2_347, %L.LB2_442
  br label %L.LB2_332, !dbg !35

L.LB2_333:                                        ; preds = %L.LB2_332
  br label %L.LB2_316

L.LB2_316:                                        ; preds = %L.LB2_333
  ret void, !dbg !35
}

declare void @__kmpc_dispatch_fini_4(i64*, i32) #0

declare void @__kmpc_end_ordered(i64*, i32) #0

declare void @__kmpc_ordered(i64*, i32) #0

declare signext i32 @__kmpc_dispatch_next_4(i64*, i32, i64*, i64*, i64*, i64*) #0

declare void @__kmpc_dispatch_init_4(i64*, i32, i32, i32, i32, i32, i32) #0

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
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB110-ordered-orig-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb110_ordered_orig_no", scope: !2, file: !3, line: 12, type: !6, scopeLine: 12, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!6 = !DISubroutineType(cc: DW_CC_program, types: !7)
!7 = !{null}
!8 = !DILocalVariable(name: "omp_sched_static", scope: !5, file: !3, type: !9)
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 0, scope: !5)
!11 = !DILocalVariable(name: "omp_proc_bind_false", scope: !5, file: !3, type: !9)
!12 = !DILocalVariable(name: "omp_proc_bind_true", scope: !5, file: !3, type: !9)
!13 = !DILocalVariable(name: "omp_lock_hint_none", scope: !5, file: !3, type: !9)
!14 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !5, file: !3, type: !9)
!15 = !DILocation(line: 28, column: 1, scope: !5)
!16 = !DILocation(line: 12, column: 1, scope: !5)
!17 = !DILocalVariable(name: "x", scope: !5, file: !3, type: !9)
!18 = !DILocation(line: 17, column: 1, scope: !5)
!19 = !DILocation(line: 19, column: 1, scope: !5)
!20 = !DILocation(line: 27, column: 1, scope: !5)
!21 = !DILocalVariable(scope: !5, file: !3, type: !9, flags: DIFlagArtificial)
!22 = distinct !DISubprogram(name: "__nv_MAIN__F1L19_1", scope: !2, file: !3, line: 19, type: !23, scopeLine: 19, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!23 = !DISubroutineType(types: !24)
!24 = !{null, !9, !25, !25}
!25 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!26 = !DILocalVariable(name: "__nv_MAIN__F1L19_1Arg0", arg: 1, scope: !22, file: !3, type: !9)
!27 = !DILocation(line: 0, scope: !22)
!28 = !DILocalVariable(name: "__nv_MAIN__F1L19_1Arg1", arg: 2, scope: !22, file: !3, type: !25)
!29 = !DILocalVariable(name: "__nv_MAIN__F1L19_1Arg2", arg: 3, scope: !22, file: !3, type: !25)
!30 = !DILocalVariable(name: "omp_sched_static", scope: !22, file: !3, type: !9)
!31 = !DILocalVariable(name: "omp_proc_bind_false", scope: !22, file: !3, type: !9)
!32 = !DILocalVariable(name: "omp_proc_bind_true", scope: !22, file: !3, type: !9)
!33 = !DILocalVariable(name: "omp_lock_hint_none", scope: !22, file: !3, type: !9)
!34 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !22, file: !3, type: !9)
!35 = !DILocation(line: 24, column: 1, scope: !22)
!36 = !DILocation(line: 20, column: 1, scope: !22)
!37 = !DILocalVariable(name: "i", scope: !22, file: !3, type: !9)
!38 = !DILocation(line: 21, column: 1, scope: !22)
!39 = !DILocation(line: 22, column: 1, scope: !22)
!40 = !DILocation(line: 23, column: 1, scope: !22)

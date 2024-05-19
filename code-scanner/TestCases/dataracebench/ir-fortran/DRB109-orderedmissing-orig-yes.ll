; ModuleID = '/tmp/DRB109-orderedmissing-orig-yes-871053.ll'
source_filename = "/tmp/DRB109-orderedmissing-orig-yes-871053.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%astruct.dt66 = type <{ i8* }>

@.C305_MAIN_ = internal constant i32 14
@.C322_MAIN_ = internal constant [3 x i8] c"x ="
@.C284_MAIN_ = internal constant i64 0
@.C319_MAIN_ = internal constant i32 6
@.C317_MAIN_ = internal constant [59 x i8] c"micro-benchmarks-fortran/DRB109-orderedmissing-orig-yes.f95"
@.C306_MAIN_ = internal constant i32 25
@.C312_MAIN_ = internal constant i32 100
@.C285_MAIN_ = internal constant i32 1
@.C283_MAIN_ = internal constant i32 0
@.C312___nv_MAIN__F1L19_1 = internal constant i32 100
@.C285___nv_MAIN__F1L19_1 = internal constant i32 1
@.C283___nv_MAIN__F1L19_1 = internal constant i32 0

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__360 = alloca i32, align 4
  %x_307 = alloca i32, align 4
  %.uplevelArgPack0001_354 = alloca %astruct.dt66, align 8
  %z__io_321 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !14, metadata !DIExpression()), !dbg !10
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !15
  store i32 %0, i32* %__gtid_MAIN__360, align 4, !dbg !15
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !16
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !16
  call void (i8*, ...) %2(i8* %1), !dbg !16
  br label %L.LB1_349

L.LB1_349:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %x_307, metadata !17, metadata !DIExpression()), !dbg !10
  %3 = bitcast i32* %x_307 to i8*, !dbg !18
  %4 = bitcast %astruct.dt66* %.uplevelArgPack0001_354 to i8**, !dbg !18
  store i8* %3, i8** %4, align 8, !dbg !18
  br label %L.LB1_358, !dbg !18

L.LB1_358:                                        ; preds = %L.LB1_349
  %5 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L19_1_ to i64*, !dbg !18
  %6 = bitcast %astruct.dt66* %.uplevelArgPack0001_354 to i64*, !dbg !18
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %5, i64* %6), !dbg !18
  call void (...) @_mp_bcs_nest(), !dbg !19
  %7 = bitcast i32* @.C306_MAIN_ to i8*, !dbg !19
  %8 = bitcast [59 x i8]* @.C317_MAIN_ to i8*, !dbg !19
  %9 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !19
  call void (i8*, i8*, i64, ...) %9(i8* %7, i8* %8, i64 59), !dbg !19
  %10 = bitcast i32* @.C319_MAIN_ to i8*, !dbg !19
  %11 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !19
  %12 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !19
  %13 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !19
  %14 = call i32 (i8*, i8*, i8*, i8*, ...) %13(i8* %10, i8* null, i8* %11, i8* %12), !dbg !19
  call void @llvm.dbg.declare(metadata i32* %z__io_321, metadata !20, metadata !DIExpression()), !dbg !10
  store i32 %14, i32* %z__io_321, align 4, !dbg !19
  %15 = bitcast [3 x i8]* @.C322_MAIN_ to i8*, !dbg !19
  %16 = bitcast i32 (...)* @f90io_sc_ch_ldw to i32 (i8*, i32, i64, ...)*, !dbg !19
  %17 = call i32 (i8*, i32, i64, ...) %16(i8* %15, i32 14, i64 3), !dbg !19
  store i32 %17, i32* %z__io_321, align 4, !dbg !19
  %18 = load i32, i32* %x_307, align 4, !dbg !19
  call void @llvm.dbg.value(metadata i32 %18, metadata !17, metadata !DIExpression()), !dbg !10
  %19 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !19
  %20 = call i32 (i32, i32, ...) %19(i32 %18, i32 25), !dbg !19
  store i32 %20, i32* %z__io_321, align 4, !dbg !19
  %21 = call i32 (...) @f90io_ldw_end(), !dbg !19
  store i32 %21, i32* %z__io_321, align 4, !dbg !19
  call void (...) @_mp_ecs_nest(), !dbg !19
  ret void, !dbg !15
}

define internal void @__nv_MAIN__F1L19_1_(i32* %__nv_MAIN__F1L19_1Arg0, i64* %__nv_MAIN__F1L19_1Arg1, i64* %__nv_MAIN__F1L19_1Arg2) #0 !dbg !21 {
L.entry:
  %__gtid___nv_MAIN__F1L19_1__407 = alloca i32, align 4
  %.i0000p_314 = alloca i32, align 4
  %i_313 = alloca i32, align 4
  %.dY0001p_333 = alloca i32, align 4
  %.du0001p_338 = alloca i32, align 4
  %.de0001p_339 = alloca i32, align 4
  %.di0001p_340 = alloca i32, align 4
  %.ds0001p_341 = alloca i32, align 4
  %.dx0001p_343 = alloca i32, align 4
  %.dl0001p_344 = alloca i32, align 4
  %.dU0001p_345 = alloca i32, align 4
  %.dl0001p.copy_401 = alloca i32, align 4
  %.dU0001p.copy_402 = alloca i32, align 4
  %.ds0001p.copy_403 = alloca i32, align 4
  %.dX0001p_342 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L19_1Arg0, metadata !25, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L19_1Arg1, metadata !27, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L19_1Arg2, metadata !28, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.value(metadata i32 1, metadata !29, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.value(metadata i32 0, metadata !30, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.value(metadata i32 1, metadata !31, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.value(metadata i32 0, metadata !32, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.value(metadata i32 1, metadata !33, metadata !DIExpression()), !dbg !26
  %0 = load i32, i32* %__nv_MAIN__F1L19_1Arg0, align 4, !dbg !34
  store i32 %0, i32* %__gtid___nv_MAIN__F1L19_1__407, align 4, !dbg !34
  br label %L.LB2_390

L.LB2_390:                                        ; preds = %L.entry
  br label %L.LB2_311

L.LB2_311:                                        ; preds = %L.LB2_390
  store i32 0, i32* %.i0000p_314, align 4, !dbg !35
  call void @llvm.dbg.declare(metadata i32* %i_313, metadata !36, metadata !DIExpression()), !dbg !34
  store i32 1, i32* %i_313, align 4, !dbg !35
  store i32 100, i32* %.dY0001p_333, align 4, !dbg !35
  store i32 1, i32* %i_313, align 4, !dbg !35
  store i32 100, i32* %.du0001p_338, align 4, !dbg !35
  store i32 100, i32* %.de0001p_339, align 4, !dbg !35
  store i32 1, i32* %.di0001p_340, align 4, !dbg !35
  %1 = load i32, i32* %.di0001p_340, align 4, !dbg !35
  store i32 %1, i32* %.ds0001p_341, align 4, !dbg !35
  store i32 1, i32* %.dx0001p_343, align 4, !dbg !35
  store i32 1, i32* %.dl0001p_344, align 4, !dbg !35
  store i32 100, i32* %.dU0001p_345, align 4, !dbg !35
  %2 = load i32, i32* %.dl0001p_344, align 4, !dbg !35
  store i32 %2, i32* %.dl0001p.copy_401, align 4, !dbg !35
  %3 = load i32, i32* %.dU0001p_345, align 4, !dbg !35
  store i32 %3, i32* %.dU0001p.copy_402, align 4, !dbg !35
  %4 = load i32, i32* %.ds0001p_341, align 4, !dbg !35
  store i32 %4, i32* %.ds0001p.copy_403, align 4, !dbg !35
  %5 = load i32, i32* %__gtid___nv_MAIN__F1L19_1__407, align 4, !dbg !35
  %6 = load i32, i32* %.dl0001p.copy_401, align 4, !dbg !35
  %7 = load i32, i32* %.dU0001p.copy_402, align 4, !dbg !35
  %8 = load i32, i32* %.ds0001p.copy_403, align 4, !dbg !35
  call void @__kmpc_dispatch_init_4(i64* null, i32 %5, i32 66, i32 %6, i32 %7, i32 %8, i32 0), !dbg !35
  %9 = load i32, i32* %.dl0001p.copy_401, align 4, !dbg !35
  store i32 %9, i32* %.dl0001p_344, align 4, !dbg !35
  %10 = load i32, i32* %.dU0001p.copy_402, align 4, !dbg !35
  store i32 %10, i32* %.dU0001p_345, align 4, !dbg !35
  %11 = load i32, i32* %.ds0001p.copy_403, align 4, !dbg !35
  store i32 %11, i32* %.ds0001p_341, align 4, !dbg !35
  br label %L.LB2_331

L.LB2_331:                                        ; preds = %L.LB2_347, %L.LB2_311
  %12 = load i32, i32* %__gtid___nv_MAIN__F1L19_1__407, align 4, !dbg !35
  %13 = bitcast i32* %.i0000p_314 to i64*, !dbg !35
  %14 = bitcast i32* %.dx0001p_343 to i64*, !dbg !35
  %15 = bitcast i32* %.de0001p_339 to i64*, !dbg !35
  %16 = bitcast i32* %.ds0001p_341 to i64*, !dbg !35
  %17 = call i32 @__kmpc_dispatch_next_4(i64* null, i32 %12, i64* %13, i64* %14, i64* %15, i64* %16), !dbg !35
  %18 = icmp eq i32 %17, 0, !dbg !35
  br i1 %18, label %L.LB2_332, label %L.LB2_435, !dbg !35

L.LB2_435:                                        ; preds = %L.LB2_331
  %19 = load i32, i32* %.dx0001p_343, align 4, !dbg !35
  store i32 %19, i32* %.dX0001p_342, align 4, !dbg !35
  %20 = load i32, i32* %.dX0001p_342, align 4, !dbg !35
  store i32 %20, i32* %i_313, align 4, !dbg !35
  %21 = load i32, i32* %.ds0001p_341, align 4, !dbg !35
  %22 = load i32, i32* %.de0001p_339, align 4, !dbg !35
  %23 = load i32, i32* %.dX0001p_342, align 4, !dbg !35
  %24 = sub nsw i32 %22, %23, !dbg !35
  %25 = add nsw i32 %21, %24, !dbg !35
  %26 = load i32, i32* %.ds0001p_341, align 4, !dbg !35
  %27 = sdiv i32 %25, %26, !dbg !35
  store i32 %27, i32* %.dY0001p_333, align 4, !dbg !35
  %28 = load i32, i32* %.dY0001p_333, align 4, !dbg !35
  %29 = icmp sle i32 %28, 0, !dbg !35
  br i1 %29, label %L.LB2_347, label %L.LB2_346, !dbg !35

L.LB2_346:                                        ; preds = %L.LB2_346, %L.LB2_435
  %30 = bitcast i64* %__nv_MAIN__F1L19_1Arg2 to i32**, !dbg !37
  %31 = load i32*, i32** %30, align 8, !dbg !37
  %32 = load i32, i32* %31, align 4, !dbg !37
  %33 = add nsw i32 %32, 1, !dbg !37
  %34 = bitcast i64* %__nv_MAIN__F1L19_1Arg2 to i32**, !dbg !37
  %35 = load i32*, i32** %34, align 8, !dbg !37
  store i32 %33, i32* %35, align 4, !dbg !37
  %36 = load i32, i32* %__gtid___nv_MAIN__F1L19_1__407, align 4, !dbg !34
  call void @__kmpc_dispatch_fini_4(i64* null, i32 %36), !dbg !34
  %37 = load i32, i32* %.ds0001p_341, align 4, !dbg !34
  %38 = load i32, i32* %i_313, align 4, !dbg !34
  call void @llvm.dbg.value(metadata i32 %38, metadata !36, metadata !DIExpression()), !dbg !34
  %39 = add nsw i32 %37, %38, !dbg !34
  store i32 %39, i32* %i_313, align 4, !dbg !34
  %40 = load i32, i32* %.dY0001p_333, align 4, !dbg !34
  %41 = sub nsw i32 %40, 1, !dbg !34
  store i32 %41, i32* %.dY0001p_333, align 4, !dbg !34
  %42 = load i32, i32* %.dY0001p_333, align 4, !dbg !34
  %43 = icmp sgt i32 %42, 0, !dbg !34
  br i1 %43, label %L.LB2_346, label %L.LB2_347, !dbg !34

L.LB2_347:                                        ; preds = %L.LB2_346, %L.LB2_435
  br label %L.LB2_331, !dbg !34

L.LB2_332:                                        ; preds = %L.LB2_331
  br label %L.LB2_315

L.LB2_315:                                        ; preds = %L.LB2_332
  ret void, !dbg !34
}

declare void @__kmpc_dispatch_fini_4(i64*, i32) #0

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
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB109-orderedmissing-orig-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb109_orderedmissing_orig_yes", scope: !2, file: !3, line: 13, type: !6, scopeLine: 13, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!6 = !DISubroutineType(cc: DW_CC_program, types: !7)
!7 = !{null}
!8 = !DILocalVariable(name: "omp_sched_static", scope: !5, file: !3, type: !9)
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 0, scope: !5)
!11 = !DILocalVariable(name: "omp_proc_bind_false", scope: !5, file: !3, type: !9)
!12 = !DILocalVariable(name: "omp_proc_bind_true", scope: !5, file: !3, type: !9)
!13 = !DILocalVariable(name: "omp_lock_hint_none", scope: !5, file: !3, type: !9)
!14 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !5, file: !3, type: !9)
!15 = !DILocation(line: 27, column: 1, scope: !5)
!16 = !DILocation(line: 13, column: 1, scope: !5)
!17 = !DILocalVariable(name: "x", scope: !5, file: !3, type: !9)
!18 = !DILocation(line: 19, column: 1, scope: !5)
!19 = !DILocation(line: 25, column: 1, scope: !5)
!20 = !DILocalVariable(scope: !5, file: !3, type: !9, flags: DIFlagArtificial)
!21 = distinct !DISubprogram(name: "__nv_MAIN__F1L19_1", scope: !2, file: !3, line: 19, type: !22, scopeLine: 19, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!22 = !DISubroutineType(types: !23)
!23 = !{null, !9, !24, !24}
!24 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!25 = !DILocalVariable(name: "__nv_MAIN__F1L19_1Arg0", arg: 1, scope: !21, file: !3, type: !9)
!26 = !DILocation(line: 0, scope: !21)
!27 = !DILocalVariable(name: "__nv_MAIN__F1L19_1Arg1", arg: 2, scope: !21, file: !3, type: !24)
!28 = !DILocalVariable(name: "__nv_MAIN__F1L19_1Arg2", arg: 3, scope: !21, file: !3, type: !24)
!29 = !DILocalVariable(name: "omp_sched_static", scope: !21, file: !3, type: !9)
!30 = !DILocalVariable(name: "omp_proc_bind_false", scope: !21, file: !3, type: !9)
!31 = !DILocalVariable(name: "omp_proc_bind_true", scope: !21, file: !3, type: !9)
!32 = !DILocalVariable(name: "omp_lock_hint_none", scope: !21, file: !3, type: !9)
!33 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !21, file: !3, type: !9)
!34 = !DILocation(line: 22, column: 1, scope: !21)
!35 = !DILocation(line: 20, column: 1, scope: !21)
!36 = !DILocalVariable(name: "i", scope: !21, file: !3, type: !9)
!37 = !DILocation(line: 21, column: 1, scope: !21)

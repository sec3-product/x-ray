; ModuleID = '/tmp/DRB170-workshare-orig-no-548e02.ll'
source_filename = "/tmp/DRB170-workshare-orig-no-548e02.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%astruct.dt58 = type <{ i8*, i8*, i8*, i8* }>

@.C309_MAIN_ = internal constant i32 25
@.C284_MAIN_ = internal constant i64 0
@.C328_MAIN_ = internal constant [53 x i8] c"micro-benchmarks-fortran/DRB170-workshare-orig-no.f95"
@.C330_MAIN_ = internal constant i32 33
@.C326_MAIN_ = internal constant i32 6
@.C300_MAIN_ = internal constant i32 2
@.C285_MAIN_ = internal constant i32 1
@.C283_MAIN_ = internal constant i32 0
@.C300___nv_MAIN__F1L21_1 = internal constant i32 2
@.C285___nv_MAIN__F1L21_1 = internal constant i32 1

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__364 = alloca i32, align 4
  %bb_311 = alloca i32, align 4
  %cc_312 = alloca i32, align 4
  %aa_310 = alloca i32, align 4
  %.uplevelArgPack0001_351 = alloca %astruct.dt58, align 16
  %res_313 = alloca i32, align 4
  %z__io_332 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 2, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 2, metadata !14, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !15, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !16, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 2, metadata !17, metadata !DIExpression()), !dbg !10
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !18
  store i32 %0, i32* %__gtid_MAIN__364, align 4, !dbg !18
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !19
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !19
  call void (i8*, ...) %2(i8* %1), !dbg !19
  br label %L.LB1_344

L.LB1_344:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %bb_311, metadata !20, metadata !DIExpression()), !dbg !10
  store i32 1, i32* %bb_311, align 4, !dbg !21
  call void @llvm.dbg.declare(metadata i32* %cc_312, metadata !22, metadata !DIExpression()), !dbg !10
  store i32 2, i32* %cc_312, align 4, !dbg !23
  call void @llvm.dbg.declare(metadata i32* %aa_310, metadata !24, metadata !DIExpression()), !dbg !10
  %3 = bitcast i32* %aa_310 to i8*, !dbg !25
  %4 = bitcast %astruct.dt58* %.uplevelArgPack0001_351 to i8**, !dbg !25
  store i8* %3, i8** %4, align 8, !dbg !25
  %5 = bitcast i32* %bb_311 to i8*, !dbg !25
  %6 = bitcast %astruct.dt58* %.uplevelArgPack0001_351 to i8*, !dbg !25
  %7 = getelementptr i8, i8* %6, i64 8, !dbg !25
  %8 = bitcast i8* %7 to i8**, !dbg !25
  store i8* %5, i8** %8, align 8, !dbg !25
  %9 = bitcast i32* %cc_312 to i8*, !dbg !25
  %10 = bitcast %astruct.dt58* %.uplevelArgPack0001_351 to i8*, !dbg !25
  %11 = getelementptr i8, i8* %10, i64 16, !dbg !25
  %12 = bitcast i8* %11 to i8**, !dbg !25
  store i8* %9, i8** %12, align 8, !dbg !25
  call void @llvm.dbg.declare(metadata i32* %res_313, metadata !26, metadata !DIExpression()), !dbg !10
  %13 = bitcast i32* %res_313 to i8*, !dbg !25
  %14 = bitcast %astruct.dt58* %.uplevelArgPack0001_351 to i8*, !dbg !25
  %15 = getelementptr i8, i8* %14, i64 24, !dbg !25
  %16 = bitcast i8* %15 to i8**, !dbg !25
  store i8* %13, i8** %16, align 8, !dbg !25
  br label %L.LB1_362, !dbg !25

L.LB1_362:                                        ; preds = %L.LB1_344
  %17 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L21_1_ to i64*, !dbg !25
  %18 = bitcast %astruct.dt58* %.uplevelArgPack0001_351 to i64*, !dbg !25
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %17, i64* %18), !dbg !25
  %19 = load i32, i32* %res_313, align 4, !dbg !27
  call void @llvm.dbg.value(metadata i32 %19, metadata !26, metadata !DIExpression()), !dbg !10
  %20 = icmp eq i32 %19, 6, !dbg !27
  br i1 %20, label %L.LB1_342, label %L.LB1_388, !dbg !27

L.LB1_388:                                        ; preds = %L.LB1_362
  call void (...) @_mp_bcs_nest(), !dbg !28
  %21 = bitcast i32* @.C330_MAIN_ to i8*, !dbg !28
  %22 = bitcast [53 x i8]* @.C328_MAIN_ to i8*, !dbg !28
  %23 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !28
  call void (i8*, i8*, i64, ...) %23(i8* %21, i8* %22, i64 53), !dbg !28
  %24 = bitcast i32* @.C326_MAIN_ to i8*, !dbg !28
  %25 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !28
  %26 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !28
  %27 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !28
  %28 = call i32 (i8*, i8*, i8*, i8*, ...) %27(i8* %24, i8* null, i8* %25, i8* %26), !dbg !28
  call void @llvm.dbg.declare(metadata i32* %z__io_332, metadata !29, metadata !DIExpression()), !dbg !10
  store i32 %28, i32* %z__io_332, align 4, !dbg !28
  %29 = load i32, i32* %res_313, align 4, !dbg !28
  call void @llvm.dbg.value(metadata i32 %29, metadata !26, metadata !DIExpression()), !dbg !10
  %30 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !28
  %31 = call i32 (i32, i32, ...) %30(i32 %29, i32 25), !dbg !28
  store i32 %31, i32* %z__io_332, align 4, !dbg !28
  %32 = call i32 (...) @f90io_ldw_end(), !dbg !28
  store i32 %32, i32* %z__io_332, align 4, !dbg !28
  call void (...) @_mp_ecs_nest(), !dbg !28
  br label %L.LB1_342

L.LB1_342:                                        ; preds = %L.LB1_388, %L.LB1_362
  ret void, !dbg !18
}

define internal void @__nv_MAIN__F1L21_1_(i32* %__nv_MAIN__F1L21_1Arg0, i64* %__nv_MAIN__F1L21_1Arg1, i64* %__nv_MAIN__F1L21_1Arg2) #0 !dbg !30 {
L.entry:
  %__gtid___nv_MAIN__F1L21_1__398 = alloca i32, align 4
  %.s0000_393 = alloca i32, align 4
  %.s0001_394 = alloca i32, align 4
  %.s0002_413 = alloca i32, align 4
  %.s0003_414 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L21_1Arg0, metadata !34, metadata !DIExpression()), !dbg !35
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L21_1Arg1, metadata !36, metadata !DIExpression()), !dbg !35
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L21_1Arg2, metadata !37, metadata !DIExpression()), !dbg !35
  call void @llvm.dbg.value(metadata i32 1, metadata !38, metadata !DIExpression()), !dbg !35
  call void @llvm.dbg.value(metadata i32 2, metadata !39, metadata !DIExpression()), !dbg !35
  call void @llvm.dbg.value(metadata i32 0, metadata !40, metadata !DIExpression()), !dbg !35
  call void @llvm.dbg.value(metadata i32 1, metadata !41, metadata !DIExpression()), !dbg !35
  call void @llvm.dbg.value(metadata i32 2, metadata !42, metadata !DIExpression()), !dbg !35
  call void @llvm.dbg.value(metadata i32 0, metadata !43, metadata !DIExpression()), !dbg !35
  call void @llvm.dbg.value(metadata i32 1, metadata !44, metadata !DIExpression()), !dbg !35
  call void @llvm.dbg.value(metadata i32 2, metadata !45, metadata !DIExpression()), !dbg !35
  %0 = load i32, i32* %__nv_MAIN__F1L21_1Arg0, align 4, !dbg !46
  store i32 %0, i32* %__gtid___nv_MAIN__F1L21_1__398, align 4, !dbg !46
  br label %L.LB2_392

L.LB2_392:                                        ; preds = %L.entry
  br label %L.LB2_316

L.LB2_316:                                        ; preds = %L.LB2_392
  br label %L.LB2_319

L.LB2_319:                                        ; preds = %L.LB2_316
  store i32 -1, i32* %.s0000_393, align 4, !dbg !47
  store i32 0, i32* %.s0001_394, align 4, !dbg !47
  %1 = load i32, i32* %__gtid___nv_MAIN__F1L21_1__398, align 4, !dbg !47
  %2 = call i32 @__kmpc_single(i64* null, i32 %1), !dbg !47
  %3 = icmp eq i32 %2, 0, !dbg !47
  br i1 %3, label %L.LB2_340, label %L.LB2_422, !dbg !47

L.LB2_422:                                        ; preds = %L.LB2_319
  %4 = bitcast i64* %__nv_MAIN__F1L21_1Arg2 to i8*, !dbg !47
  %5 = getelementptr i8, i8* %4, i64 8, !dbg !47
  %6 = bitcast i8* %5 to i32**, !dbg !47
  %7 = load i32*, i32** %6, align 8, !dbg !47
  %8 = load i32, i32* %7, align 4, !dbg !47
  %9 = bitcast i64* %__nv_MAIN__F1L21_1Arg2 to i32**, !dbg !47
  %10 = load i32*, i32** %9, align 8, !dbg !47
  store i32 %8, i32* %10, align 4, !dbg !47
  %11 = bitcast i64* %__nv_MAIN__F1L21_1Arg2 to i32**, !dbg !48
  %12 = load i32*, i32** %11, align 8, !dbg !48
  %13 = load i32, i32* %12, align 4, !dbg !48
  %14 = bitcast i64* %__nv_MAIN__F1L21_1Arg2 to i8*, !dbg !48
  %15 = getelementptr i8, i8* %14, i64 16, !dbg !48
  %16 = bitcast i8* %15 to i32**, !dbg !48
  %17 = load i32*, i32** %16, align 8, !dbg !48
  %18 = load i32, i32* %17, align 4, !dbg !48
  %19 = add nsw i32 %13, %18, !dbg !48
  %20 = bitcast i64* %__nv_MAIN__F1L21_1Arg2 to i32**, !dbg !48
  %21 = load i32*, i32** %20, align 8, !dbg !48
  store i32 %19, i32* %21, align 4, !dbg !48
  %22 = load i32, i32* %__gtid___nv_MAIN__F1L21_1__398, align 4, !dbg !49
  store i32 %22, i32* %.s0000_393, align 4, !dbg !49
  store i32 1, i32* %.s0001_394, align 4, !dbg !49
  %23 = load i32, i32* %__gtid___nv_MAIN__F1L21_1__398, align 4, !dbg !49
  call void @__kmpc_end_single(i64* null, i32 %23), !dbg !49
  br label %L.LB2_340

L.LB2_340:                                        ; preds = %L.LB2_422, %L.LB2_319
  %24 = load i32, i32* %__gtid___nv_MAIN__F1L21_1__398, align 4, !dbg !49
  call void @__kmpc_barrier(i64* null, i32 %24), !dbg !49
  br label %L.LB2_320

L.LB2_320:                                        ; preds = %L.LB2_340
  br label %L.LB2_323

L.LB2_323:                                        ; preds = %L.LB2_320
  store i32 -1, i32* %.s0002_413, align 4, !dbg !50
  store i32 0, i32* %.s0003_414, align 4, !dbg !50
  %25 = load i32, i32* %__gtid___nv_MAIN__F1L21_1__398, align 4, !dbg !50
  %26 = call i32 @__kmpc_single(i64* null, i32 %25), !dbg !50
  %27 = icmp eq i32 %26, 0, !dbg !50
  br i1 %27, label %L.LB2_341, label %L.LB2_423, !dbg !50

L.LB2_423:                                        ; preds = %L.LB2_323
  %28 = bitcast i64* %__nv_MAIN__F1L21_1Arg2 to i32**, !dbg !50
  %29 = load i32*, i32** %28, align 8, !dbg !50
  %30 = load i32, i32* %29, align 4, !dbg !50
  %31 = mul nsw i32 %30, 2, !dbg !50
  %32 = bitcast i64* %__nv_MAIN__F1L21_1Arg2 to i8*, !dbg !50
  %33 = getelementptr i8, i8* %32, i64 24, !dbg !50
  %34 = bitcast i8* %33 to i32**, !dbg !50
  %35 = load i32*, i32** %34, align 8, !dbg !50
  store i32 %31, i32* %35, align 4, !dbg !50
  %36 = load i32, i32* %__gtid___nv_MAIN__F1L21_1__398, align 4, !dbg !51
  store i32 %36, i32* %.s0002_413, align 4, !dbg !51
  store i32 1, i32* %.s0003_414, align 4, !dbg !51
  %37 = load i32, i32* %__gtid___nv_MAIN__F1L21_1__398, align 4, !dbg !51
  call void @__kmpc_end_single(i64* null, i32 %37), !dbg !51
  br label %L.LB2_341

L.LB2_341:                                        ; preds = %L.LB2_423, %L.LB2_323
  %38 = load i32, i32* %__gtid___nv_MAIN__F1L21_1__398, align 4, !dbg !51
  call void @__kmpc_barrier(i64* null, i32 %38), !dbg !51
  br label %L.LB2_324

L.LB2_324:                                        ; preds = %L.LB2_341
  br label %L.LB2_325

L.LB2_325:                                        ; preds = %L.LB2_324
  ret void, !dbg !46
}

declare void @__kmpc_barrier(i64*, i32) #0

declare void @__kmpc_end_single(i64*, i32) #0

declare signext i32 @__kmpc_single(i64*, i32) #0

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
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB170-workshare-orig-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb170_workshare_orig_no", scope: !2, file: !3, line: 12, type: !6, scopeLine: 12, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
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
!18 = !DILocation(line: 35, column: 1, scope: !5)
!19 = !DILocation(line: 12, column: 1, scope: !5)
!20 = !DILocalVariable(name: "bb", scope: !5, file: !3, type: !9)
!21 = !DILocation(line: 18, column: 1, scope: !5)
!22 = !DILocalVariable(name: "cc", scope: !5, file: !3, type: !9)
!23 = !DILocation(line: 19, column: 1, scope: !5)
!24 = !DILocalVariable(name: "aa", scope: !5, file: !3, type: !9)
!25 = !DILocation(line: 21, column: 1, scope: !5)
!26 = !DILocalVariable(name: "res", scope: !5, file: !3, type: !9)
!27 = !DILocation(line: 32, column: 1, scope: !5)
!28 = !DILocation(line: 33, column: 1, scope: !5)
!29 = !DILocalVariable(scope: !5, file: !3, type: !9, flags: DIFlagArtificial)
!30 = distinct !DISubprogram(name: "__nv_MAIN__F1L21_1", scope: !2, file: !3, line: 21, type: !31, scopeLine: 21, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!31 = !DISubroutineType(types: !32)
!32 = !{null, !9, !33, !33}
!33 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!34 = !DILocalVariable(name: "__nv_MAIN__F1L21_1Arg0", arg: 1, scope: !30, file: !3, type: !9)
!35 = !DILocation(line: 0, scope: !30)
!36 = !DILocalVariable(name: "__nv_MAIN__F1L21_1Arg1", arg: 2, scope: !30, file: !3, type: !33)
!37 = !DILocalVariable(name: "__nv_MAIN__F1L21_1Arg2", arg: 3, scope: !30, file: !3, type: !33)
!38 = !DILocalVariable(name: "omp_sched_static", scope: !30, file: !3, type: !9)
!39 = !DILocalVariable(name: "omp_sched_dynamic", scope: !30, file: !3, type: !9)
!40 = !DILocalVariable(name: "omp_proc_bind_false", scope: !30, file: !3, type: !9)
!41 = !DILocalVariable(name: "omp_proc_bind_true", scope: !30, file: !3, type: !9)
!42 = !DILocalVariable(name: "omp_proc_bind_master", scope: !30, file: !3, type: !9)
!43 = !DILocalVariable(name: "omp_lock_hint_none", scope: !30, file: !3, type: !9)
!44 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !30, file: !3, type: !9)
!45 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !30, file: !3, type: !9)
!46 = !DILocation(line: 30, column: 1, scope: !30)
!47 = !DILocation(line: 23, column: 1, scope: !30)
!48 = !DILocation(line: 24, column: 1, scope: !30)
!49 = !DILocation(line: 25, column: 1, scope: !30)
!50 = !DILocation(line: 28, column: 1, scope: !30)
!51 = !DILocation(line: 29, column: 1, scope: !30)

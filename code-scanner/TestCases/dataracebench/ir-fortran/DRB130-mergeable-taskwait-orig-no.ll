; ModuleID = '/tmp/DRB130-mergeable-taskwait-orig-no-b68e83.ll'
source_filename = "/tmp/DRB130-mergeable-taskwait-orig-no-b68e83.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.STATICS1 = type <{ [48 x i8] }>

@.STATICS1 = internal global %struct.STATICS1 <{ [48 x i8] c"\FB\FF\FF\FF\03\00\00\00x =\00\00\00\00\00\03\00\00\00\EB\FF\FF\FF\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\FF\FF\FF\FF\00\00\00\00" }>, align 16
@.C309_MAIN_ = internal constant i32 25
@.C284_MAIN_ = internal constant i64 0
@.C320_MAIN_ = internal constant i32 6
@.C316_MAIN_ = internal constant [62 x i8] c"micro-benchmarks-fortran/DRB130-mergeable-taskwait-orig-no.f95"
@.C318_MAIN_ = internal constant i32 24
@.C285_MAIN_ = internal constant i32 1
@.C300_MAIN_ = internal constant i32 2
@.C283_MAIN_ = internal constant i32 0
@.C285___nv_MAIN__F1L20_1 = internal constant i32 1

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__341 = alloca i32, align 4
  %x_310 = alloca i32, align 4
  %.s0000_336 = alloca i32, align 4
  %.z0297_335 = alloca i8*, align 8
  %z__io_322 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 2, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 2, metadata !14, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !15, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !16, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 2, metadata !17, metadata !DIExpression()), !dbg !10
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !18
  store i32 %0, i32* %__gtid_MAIN__341, align 4, !dbg !18
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !19
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !19
  call void (i8*, ...) %2(i8* %1), !dbg !19
  br label %L.LB1_333

L.LB1_333:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %x_310, metadata !20, metadata !DIExpression()), !dbg !10
  store i32 2, i32* %x_310, align 4, !dbg !21
  store i32 1, i32* %.s0000_336, align 4, !dbg !22
  %3 = load i32, i32* %__gtid_MAIN__341, align 4, !dbg !23
  %4 = load i32, i32* %.s0000_336, align 4, !dbg !23
  %5 = bitcast void (i32, i64*)* @__nv_MAIN__F1L20_1_ to i64*, !dbg !23
  %6 = call i8* @__kmpc_omp_task_alloc(i64* null, i32 %3, i32 %4, i32 40, i32 8, i64* %5), !dbg !23
  store i8* %6, i8** %.z0297_335, align 8, !dbg !23
  %7 = bitcast i32* %x_310 to i8*, !dbg !23
  %8 = load i8*, i8** %.z0297_335, align 8, !dbg !23
  %9 = bitcast i8* %8 to i8***, !dbg !23
  %10 = load i8**, i8*** %9, align 8, !dbg !23
  store i8* %7, i8** %10, align 8, !dbg !23
  %11 = load i32, i32* %__gtid_MAIN__341, align 4, !dbg !23
  %12 = load i8*, i8** %.z0297_335, align 8, !dbg !23
  %13 = bitcast i8* %12 to i64*, !dbg !23
  call void @__kmpc_omp_task(i64* null, i32 %11, i64* %13), !dbg !23
  br label %L.LB1_331

L.LB1_331:                                        ; preds = %L.LB1_333
  call void (...) @_mp_bcs_nest(), !dbg !24
  %14 = bitcast i32* @.C318_MAIN_ to i8*, !dbg !24
  %15 = bitcast [62 x i8]* @.C316_MAIN_ to i8*, !dbg !24
  %16 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !24
  call void (i8*, i8*, i64, ...) %16(i8* %14, i8* %15, i64 62), !dbg !24
  %17 = bitcast i32* @.C320_MAIN_ to i8*, !dbg !24
  %18 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !24
  %19 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !24
  %20 = bitcast %struct.STATICS1* @.STATICS1 to i8*, !dbg !24
  %21 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !24
  %22 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %21(i8* %17, i8* null, i8* %18, i8* %19, i8* %20, i8* null, i64 0), !dbg !24
  call void @llvm.dbg.declare(metadata i32* %z__io_322, metadata !25, metadata !DIExpression()), !dbg !10
  store i32 %22, i32* %z__io_322, align 4, !dbg !24
  %23 = load i32, i32* %x_310, align 4, !dbg !24
  call void @llvm.dbg.value(metadata i32 %23, metadata !20, metadata !DIExpression()), !dbg !10
  %24 = bitcast i32 (...)* @f90io_sc_i_fmt_write to i32 (i32, i32, ...)*, !dbg !24
  %25 = call i32 (i32, i32, ...) %24(i32 %23, i32 25), !dbg !24
  store i32 %25, i32* %z__io_322, align 4, !dbg !24
  %26 = call i32 (...) @f90io_fmtw_end(), !dbg !24
  store i32 %26, i32* %z__io_322, align 4, !dbg !24
  call void (...) @_mp_ecs_nest(), !dbg !24
  ret void, !dbg !18
}

define internal void @__nv_MAIN__F1L20_1_(i32 %__nv_MAIN__F1L20_1Arg0.arg, i64* %__nv_MAIN__F1L20_1Arg1) #0 !dbg !26 {
L.entry:
  %__nv_MAIN__F1L20_1Arg0.addr = alloca i32, align 4
  %.S0000_374 = alloca i8*, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L20_1Arg0.addr, metadata !30, metadata !DIExpression()), !dbg !31
  store i32 %__nv_MAIN__F1L20_1Arg0.arg, i32* %__nv_MAIN__F1L20_1Arg0.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L20_1Arg0.addr, metadata !32, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L20_1Arg1, metadata !33, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.value(metadata i32 1, metadata !34, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.value(metadata i32 2, metadata !35, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.value(metadata i32 0, metadata !36, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.value(metadata i32 1, metadata !37, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.value(metadata i32 2, metadata !38, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.value(metadata i32 0, metadata !39, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.value(metadata i32 1, metadata !40, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.value(metadata i32 2, metadata !41, metadata !DIExpression()), !dbg !31
  %0 = bitcast i64* %__nv_MAIN__F1L20_1Arg1 to i8**, !dbg !42
  %1 = load i8*, i8** %0, align 8, !dbg !42
  store i8* %1, i8** %.S0000_374, align 8, !dbg !42
  br label %L.LB2_378

L.LB2_378:                                        ; preds = %L.entry
  br label %L.LB2_313

L.LB2_313:                                        ; preds = %L.LB2_378
  %2 = load i8*, i8** %.S0000_374, align 8, !dbg !43
  %3 = bitcast i8* %2 to i32**, !dbg !43
  %4 = load i32*, i32** %3, align 8, !dbg !43
  %5 = load i32, i32* %4, align 4, !dbg !43
  %6 = add nsw i32 %5, 1, !dbg !43
  %7 = load i8*, i8** %.S0000_374, align 8, !dbg !43
  %8 = bitcast i8* %7 to i32**, !dbg !43
  %9 = load i32*, i32** %8, align 8, !dbg !43
  store i32 %6, i32* %9, align 4, !dbg !43
  br label %L.LB2_314

L.LB2_314:                                        ; preds = %L.LB2_313
  ret void, !dbg !44
}

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

declare void @__kmpc_omp_task(i64*, i32, i64*) #0

declare i8* @__kmpc_omp_task_alloc(i64*, i32, i32, i32, i32, i64*) #0

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { "no-frame-pointer-elim-non-leaf" }
attributes #1 = { nounwind readnone speculatable }

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !4, globals: !4, imports: !4)
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB130-mergeable-taskwait-orig-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb130_mergeable_taskwait_orig_no", scope: !2, file: !3, line: 13, type: !6, scopeLine: 13, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
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
!18 = !DILocation(line: 26, column: 1, scope: !5)
!19 = !DILocation(line: 13, column: 1, scope: !5)
!20 = !DILocalVariable(name: "x", scope: !5, file: !3, type: !9)
!21 = !DILocation(line: 18, column: 1, scope: !5)
!22 = !DILocation(line: 20, column: 1, scope: !5)
!23 = !DILocation(line: 22, column: 1, scope: !5)
!24 = !DILocation(line: 24, column: 1, scope: !5)
!25 = !DILocalVariable(scope: !5, file: !3, type: !9, flags: DIFlagArtificial)
!26 = distinct !DISubprogram(name: "__nv_MAIN__F1L20_1", scope: !2, file: !3, line: 20, type: !27, scopeLine: 20, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!27 = !DISubroutineType(types: !28)
!28 = !{null, !9, !29}
!29 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!30 = !DILocalVariable(name: "__nv_MAIN__F1L20_1Arg0", scope: !26, file: !3, type: !9)
!31 = !DILocation(line: 0, scope: !26)
!32 = !DILocalVariable(name: "__nv_MAIN__F1L20_1Arg0", arg: 1, scope: !26, file: !3, type: !9)
!33 = !DILocalVariable(name: "__nv_MAIN__F1L20_1Arg1", arg: 2, scope: !26, file: !3, type: !29)
!34 = !DILocalVariable(name: "omp_sched_static", scope: !26, file: !3, type: !9)
!35 = !DILocalVariable(name: "omp_sched_dynamic", scope: !26, file: !3, type: !9)
!36 = !DILocalVariable(name: "omp_proc_bind_false", scope: !26, file: !3, type: !9)
!37 = !DILocalVariable(name: "omp_proc_bind_true", scope: !26, file: !3, type: !9)
!38 = !DILocalVariable(name: "omp_proc_bind_master", scope: !26, file: !3, type: !9)
!39 = !DILocalVariable(name: "omp_lock_hint_none", scope: !26, file: !3, type: !9)
!40 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !26, file: !3, type: !9)
!41 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !26, file: !3, type: !9)
!42 = !DILocation(line: 20, column: 1, scope: !26)
!43 = !DILocation(line: 21, column: 1, scope: !26)
!44 = !DILocation(line: 22, column: 1, scope: !26)

; ModuleID = '/tmp/DRB124-master-orig-yes-53de1c.ll'
source_filename = "/tmp/DRB124-master-orig-yes-53de1c.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%astruct.dt56 = type <{ i8* }>

@.C311_MAIN_ = internal constant i32 10
@.C283_MAIN_ = internal constant i32 0
@.C311___nv_MAIN__F1L22_1 = internal constant i32 10

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__329 = alloca i32, align 4
  %init_305 = alloca i32, align 4
  %.uplevelArgPack0001_323 = alloca %astruct.dt56, align 8
  call void @llvm.dbg.value(metadata i32 1, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !14, metadata !DIExpression()), !dbg !10
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !15
  store i32 %0, i32* %__gtid_MAIN__329, align 4, !dbg !15
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !16
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !16
  call void (i8*, ...) %2(i8* %1), !dbg !16
  br label %L.LB1_318

L.LB1_318:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %init_305, metadata !17, metadata !DIExpression()), !dbg !10
  %3 = bitcast i32* %init_305 to i8*, !dbg !18
  %4 = bitcast %astruct.dt56* %.uplevelArgPack0001_323 to i8**, !dbg !18
  store i8* %3, i8** %4, align 8, !dbg !18
  br label %L.LB1_327, !dbg !18

L.LB1_327:                                        ; preds = %L.LB1_318
  %5 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L22_1_ to i64*, !dbg !18
  %6 = bitcast %astruct.dt56* %.uplevelArgPack0001_323 to i64*, !dbg !18
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %5, i64* %6), !dbg !18
  ret void, !dbg !15
}

define internal void @__nv_MAIN__F1L22_1_(i32* %__nv_MAIN__F1L22_1Arg0, i64* %__nv_MAIN__F1L22_1Arg1, i64* %__nv_MAIN__F1L22_1Arg2) #0 !dbg !19 {
L.entry:
  %__gtid___nv_MAIN__F1L22_1__351 = alloca i32, align 4
  %local_310 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L22_1Arg0, metadata !23, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L22_1Arg1, metadata !25, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L22_1Arg2, metadata !26, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.value(metadata i32 1, metadata !27, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.value(metadata i32 0, metadata !28, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.value(metadata i32 1, metadata !29, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.value(metadata i32 0, metadata !30, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.value(metadata i32 1, metadata !31, metadata !DIExpression()), !dbg !24
  %0 = load i32, i32* %__nv_MAIN__F1L22_1Arg0, align 4, !dbg !32
  store i32 %0, i32* %__gtid___nv_MAIN__F1L22_1__351, align 4, !dbg !32
  br label %L.LB2_350

L.LB2_350:                                        ; preds = %L.entry
  br label %L.LB2_309

L.LB2_309:                                        ; preds = %L.LB2_350
  %1 = load i32, i32* %__gtid___nv_MAIN__F1L22_1__351, align 4, !dbg !33
  %2 = call i32 @__kmpc_master(i64* null, i32 %1), !dbg !33
  %3 = icmp eq i32 %2, 0, !dbg !33
  br i1 %3, label %L.LB2_316, label %L.LB2_366, !dbg !33

L.LB2_366:                                        ; preds = %L.LB2_309
  %4 = bitcast i64* %__nv_MAIN__F1L22_1Arg2 to i32**, !dbg !34
  %5 = load i32*, i32** %4, align 8, !dbg !34
  store i32 10, i32* %5, align 4, !dbg !34
  %6 = load i32, i32* %__gtid___nv_MAIN__F1L22_1__351, align 4, !dbg !35
  call void @__kmpc_end_master(i64* null, i32 %6), !dbg !35
  br label %L.LB2_316

L.LB2_316:                                        ; preds = %L.LB2_366, %L.LB2_309
  %7 = bitcast i64* %__nv_MAIN__F1L22_1Arg2 to i32**, !dbg !36
  %8 = load i32*, i32** %7, align 8, !dbg !36
  %9 = load i32, i32* %8, align 4, !dbg !36
  call void @llvm.dbg.declare(metadata i32* %local_310, metadata !37, metadata !DIExpression()), !dbg !32
  store i32 %9, i32* %local_310, align 4, !dbg !36
  br label %L.LB2_312

L.LB2_312:                                        ; preds = %L.LB2_316
  ret void, !dbg !32
}

declare void @__kmpc_end_master(i64*, i32) #0

declare signext i32 @__kmpc_master(i64*, i32) #0

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
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB124-master-orig-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb124_master_orig_yes", scope: !2, file: !3, line: 16, type: !6, scopeLine: 16, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!6 = !DISubroutineType(cc: DW_CC_program, types: !7)
!7 = !{null}
!8 = !DILocalVariable(name: "omp_sched_static", scope: !5, file: !3, type: !9)
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 0, scope: !5)
!11 = !DILocalVariable(name: "omp_proc_bind_false", scope: !5, file: !3, type: !9)
!12 = !DILocalVariable(name: "omp_proc_bind_true", scope: !5, file: !3, type: !9)
!13 = !DILocalVariable(name: "omp_lock_hint_none", scope: !5, file: !3, type: !9)
!14 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !5, file: !3, type: !9)
!15 = !DILocation(line: 29, column: 1, scope: !5)
!16 = !DILocation(line: 16, column: 1, scope: !5)
!17 = !DILocalVariable(name: "init", scope: !5, file: !3, type: !9)
!18 = !DILocation(line: 22, column: 1, scope: !5)
!19 = distinct !DISubprogram(name: "__nv_MAIN__F1L22_1", scope: !2, file: !3, line: 22, type: !20, scopeLine: 22, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!20 = !DISubroutineType(types: !21)
!21 = !{null, !9, !22, !22}
!22 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!23 = !DILocalVariable(name: "__nv_MAIN__F1L22_1Arg0", arg: 1, scope: !19, file: !3, type: !9)
!24 = !DILocation(line: 0, scope: !19)
!25 = !DILocalVariable(name: "__nv_MAIN__F1L22_1Arg1", arg: 2, scope: !19, file: !3, type: !22)
!26 = !DILocalVariable(name: "__nv_MAIN__F1L22_1Arg2", arg: 3, scope: !19, file: !3, type: !22)
!27 = !DILocalVariable(name: "omp_sched_static", scope: !19, file: !3, type: !9)
!28 = !DILocalVariable(name: "omp_proc_bind_false", scope: !19, file: !3, type: !9)
!29 = !DILocalVariable(name: "omp_proc_bind_true", scope: !19, file: !3, type: !9)
!30 = !DILocalVariable(name: "omp_lock_hint_none", scope: !19, file: !3, type: !9)
!31 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !19, file: !3, type: !9)
!32 = !DILocation(line: 27, column: 1, scope: !19)
!33 = !DILocation(line: 23, column: 1, scope: !19)
!34 = !DILocation(line: 24, column: 1, scope: !19)
!35 = !DILocation(line: 25, column: 1, scope: !19)
!36 = !DILocation(line: 26, column: 1, scope: !19)
!37 = !DILocalVariable(name: "local", scope: !19, file: !3, type: !9)

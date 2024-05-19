; ModuleID = '/tmp/DRB083-declared-in-func-orig-no-0a3268.ll'
source_filename = "/tmp/DRB083-declared-in-func-orig-no-0a3268.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.C285_drb083_foo_ = internal constant i32 1
@.C283_drb083_foo_ = internal constant i32 0
@.C283_MAIN_ = internal constant i32 0

; Function Attrs: noinline
define float @drb083_() #0 {
.L.entry:
  ret float undef
}

define void @drb083_foo_() #1 !dbg !5 {
L.entry:
  %q_306 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !9, metadata !DIExpression()), !dbg !11
  call void @llvm.dbg.value(metadata i32 0, metadata !12, metadata !DIExpression()), !dbg !11
  call void @llvm.dbg.value(metadata i32 1, metadata !13, metadata !DIExpression()), !dbg !11
  call void @llvm.dbg.value(metadata i32 0, metadata !14, metadata !DIExpression()), !dbg !11
  call void @llvm.dbg.value(metadata i32 1, metadata !15, metadata !DIExpression()), !dbg !11
  br label %L.LB2_309

L.LB2_309:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %q_306, metadata !16, metadata !DIExpression()), !dbg !11
  store i32 0, i32* %q_306, align 4, !dbg !17
  %0 = load i32, i32* %q_306, align 4, !dbg !18
  call void @llvm.dbg.value(metadata i32 %0, metadata !16, metadata !DIExpression()), !dbg !11
  %1 = add nsw i32 %0, 1, !dbg !18
  store i32 %1, i32* %q_306, align 4, !dbg !18
  ret void, !dbg !19
}

define void @MAIN_() #1 !dbg !20 {
L.entry:
  %__gtid_MAIN__323 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !22, metadata !DIExpression()), !dbg !23
  call void @llvm.dbg.value(metadata i32 0, metadata !24, metadata !DIExpression()), !dbg !23
  call void @llvm.dbg.value(metadata i32 1, metadata !25, metadata !DIExpression()), !dbg !23
  call void @llvm.dbg.value(metadata i32 0, metadata !26, metadata !DIExpression()), !dbg !23
  call void @llvm.dbg.value(metadata i32 1, metadata !27, metadata !DIExpression()), !dbg !23
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !28
  store i32 %0, i32* %__gtid_MAIN__323, align 4, !dbg !28
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !29
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !29
  call void (i8*, ...) %2(i8* %1), !dbg !29
  br label %L.LB3_315

L.LB3_315:                                        ; preds = %L.entry
  br label %L.LB3_321, !dbg !30

L.LB3_321:                                        ; preds = %L.LB3_315
  %3 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L27_1_ to i64*, !dbg !30
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %3, i64* null), !dbg !30
  ret void, !dbg !28
}

define internal void @__nv_MAIN__F1L27_1_(i32* %__nv_MAIN__F1L27_1Arg0, i64* %__nv_MAIN__F1L27_1Arg1, i64* %__nv_MAIN__F1L27_1Arg2) #1 !dbg !31 {
L.entry:
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L27_1Arg0, metadata !35, metadata !DIExpression()), !dbg !36
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L27_1Arg1, metadata !37, metadata !DIExpression()), !dbg !36
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L27_1Arg2, metadata !38, metadata !DIExpression()), !dbg !36
  call void @llvm.dbg.value(metadata i32 1, metadata !39, metadata !DIExpression()), !dbg !36
  call void @llvm.dbg.value(metadata i32 0, metadata !40, metadata !DIExpression()), !dbg !36
  call void @llvm.dbg.value(metadata i32 1, metadata !41, metadata !DIExpression()), !dbg !36
  call void @llvm.dbg.value(metadata i32 0, metadata !42, metadata !DIExpression()), !dbg !36
  call void @llvm.dbg.value(metadata i32 1, metadata !43, metadata !DIExpression()), !dbg !36
  br label %L.LB4_344

L.LB4_344:                                        ; preds = %L.entry
  br label %L.LB4_309

L.LB4_309:                                        ; preds = %L.LB4_344
  call void @drb083_foo_(), !dbg !44
  br label %L.LB4_310

L.LB4_310:                                        ; preds = %L.LB4_309
  ret void, !dbg !45
}

declare void @fort_init(...) #1

declare signext i32 @__kmpc_global_thread_num(i64*) #1

declare void @__kmpc_fork_call(i64*, i32, i64*, i64*, ...) #1

declare void @__kmpc_end_serialized_parallel(i64*, i32) #1

declare void @__kmpc_serialized_parallel(i64*, i32) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

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
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB083-declared-in-func-orig-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "foo", scope: !6, file: !3, line: 15, type: !7, scopeLine: 15, spFlags: DISPFlagDefinition, unit: !2)
!6 = !DIModule(scope: !2, name: "drb083")
!7 = !DISubroutineType(types: !8)
!8 = !{null}
!9 = !DILocalVariable(name: "omp_sched_static", scope: !5, file: !3, type: !10)
!10 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!11 = !DILocation(line: 0, scope: !5)
!12 = !DILocalVariable(name: "omp_proc_bind_false", scope: !5, file: !3, type: !10)
!13 = !DILocalVariable(name: "omp_proc_bind_true", scope: !5, file: !3, type: !10)
!14 = !DILocalVariable(name: "omp_lock_hint_none", scope: !5, file: !3, type: !10)
!15 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !5, file: !3, type: !10)
!16 = !DILocalVariable(name: "q", scope: !5, file: !3, type: !10)
!17 = !DILocation(line: 17, column: 1, scope: !5)
!18 = !DILocation(line: 18, column: 1, scope: !5)
!19 = !DILocation(line: 19, column: 1, scope: !5)
!20 = distinct !DISubprogram(name: "drb083_declared_in_func_orig_no", scope: !2, file: !3, line: 22, type: !21, scopeLine: 22, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!21 = !DISubroutineType(cc: DW_CC_program, types: !8)
!22 = !DILocalVariable(name: "omp_sched_static", scope: !20, file: !3, type: !10)
!23 = !DILocation(line: 0, scope: !20)
!24 = !DILocalVariable(name: "omp_proc_bind_false", scope: !20, file: !3, type: !10)
!25 = !DILocalVariable(name: "omp_proc_bind_true", scope: !20, file: !3, type: !10)
!26 = !DILocalVariable(name: "omp_lock_hint_none", scope: !20, file: !3, type: !10)
!27 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !20, file: !3, type: !10)
!28 = !DILocation(line: 30, column: 1, scope: !20)
!29 = !DILocation(line: 22, column: 1, scope: !20)
!30 = !DILocation(line: 27, column: 1, scope: !20)
!31 = distinct !DISubprogram(name: "__nv_MAIN__F1L27_1", scope: !2, file: !3, line: 27, type: !32, scopeLine: 27, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!32 = !DISubroutineType(types: !33)
!33 = !{null, !10, !34, !34}
!34 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!35 = !DILocalVariable(name: "__nv_MAIN__F1L27_1Arg0", arg: 1, scope: !31, file: !3, type: !10)
!36 = !DILocation(line: 0, scope: !31)
!37 = !DILocalVariable(name: "__nv_MAIN__F1L27_1Arg1", arg: 2, scope: !31, file: !3, type: !34)
!38 = !DILocalVariable(name: "__nv_MAIN__F1L27_1Arg2", arg: 3, scope: !31, file: !3, type: !34)
!39 = !DILocalVariable(name: "omp_sched_static", scope: !31, file: !3, type: !10)
!40 = !DILocalVariable(name: "omp_proc_bind_false", scope: !31, file: !3, type: !10)
!41 = !DILocalVariable(name: "omp_proc_bind_true", scope: !31, file: !3, type: !10)
!42 = !DILocalVariable(name: "omp_lock_hint_none", scope: !31, file: !3, type: !10)
!43 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !31, file: !3, type: !10)
!44 = !DILocation(line: 28, column: 1, scope: !31)
!45 = !DILocation(line: 29, column: 1, scope: !31)

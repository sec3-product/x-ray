; ModuleID = '/tmp/DRB082-declared-in-func-orig-yes-ecddb6.ll'
source_filename = "/tmp/DRB082-declared-in-func-orig-yes-ecddb6.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.BSS2 = type <{ [4 x i8] }>

@.BSS2 = internal global %struct.BSS2 zeroinitializer, align 32, !dbg !0
@.C305_global_foo_foo_ = internal constant i32 25
@.C283_global_foo_foo_ = internal constant i32 0
@.C284_global_foo_foo_ = internal constant i64 0
@.C312_global_foo_foo_ = internal constant i32 6
@.C310_global_foo_foo_ = internal constant [61 x i8] c"micro-benchmarks-fortran/DRB082-declared-in-func-orig-yes.f95"
@.C306_global_foo_foo_ = internal constant i32 20
@.C285_global_foo_foo_ = internal constant i32 1
@.C283_MAIN_ = internal constant i32 0

; Function Attrs: noinline
define float @global_foo_() #0 {
.L.entry:
  ret float undef
}

define void @global_foo_foo_() #1 !dbg !2 {
L.entry:
  %z__io_314 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !13, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.value(metadata i32 0, metadata !15, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.value(metadata i32 1, metadata !16, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.value(metadata i32 0, metadata !17, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.value(metadata i32 1, metadata !18, metadata !DIExpression()), !dbg !14
  br label %L.LB2_321

L.LB2_321:                                        ; preds = %L.entry
  %0 = bitcast %struct.BSS2* @.BSS2 to i32*, !dbg !19
  %1 = load i32, i32* %0, align 4, !dbg !19
  %2 = add nsw i32 %1, 1, !dbg !19
  %3 = bitcast %struct.BSS2* @.BSS2 to i32*, !dbg !19
  store i32 %2, i32* %3, align 4, !dbg !19
  call void (...) @_mp_bcs_nest(), !dbg !20
  %4 = bitcast i32* @.C306_global_foo_foo_ to i8*, !dbg !20
  %5 = bitcast [61 x i8]* @.C310_global_foo_foo_ to i8*, !dbg !20
  %6 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !20
  call void (i8*, i8*, i64, ...) %6(i8* %4, i8* %5, i64 61), !dbg !20
  %7 = bitcast i32* @.C312_global_foo_foo_ to i8*, !dbg !20
  %8 = bitcast i32* @.C283_global_foo_foo_ to i8*, !dbg !20
  %9 = bitcast i32* @.C283_global_foo_foo_ to i8*, !dbg !20
  %10 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !20
  %11 = call i32 (i8*, i8*, i8*, i8*, ...) %10(i8* %7, i8* null, i8* %8, i8* %9), !dbg !20
  call void @llvm.dbg.declare(metadata i32* %z__io_314, metadata !21, metadata !DIExpression()), !dbg !14
  store i32 %11, i32* %z__io_314, align 4, !dbg !20
  %12 = bitcast %struct.BSS2* @.BSS2 to i32*, !dbg !20
  %13 = load i32, i32* %12, align 4, !dbg !20
  %14 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !20
  %15 = call i32 (i32, i32, ...) %14(i32 %13, i32 25), !dbg !20
  store i32 %15, i32* %z__io_314, align 4, !dbg !20
  %16 = call i32 (...) @f90io_ldw_end(), !dbg !20
  store i32 %16, i32* %z__io_314, align 4, !dbg !20
  call void (...) @_mp_ecs_nest(), !dbg !20
  ret void, !dbg !22
}

define void @MAIN_() #1 !dbg !23 {
L.entry:
  %__gtid_MAIN__323 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !25, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.value(metadata i32 0, metadata !27, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.value(metadata i32 1, metadata !28, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.value(metadata i32 0, metadata !29, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.value(metadata i32 1, metadata !30, metadata !DIExpression()), !dbg !26
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !31
  store i32 %0, i32* %__gtid_MAIN__323, align 4, !dbg !31
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !32
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !32
  call void (i8*, ...) %2(i8* %1), !dbg !32
  br label %L.LB3_315

L.LB3_315:                                        ; preds = %L.entry
  br label %L.LB3_321, !dbg !33

L.LB3_321:                                        ; preds = %L.LB3_315
  %3 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L29_1_ to i64*, !dbg !33
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %3, i64* null), !dbg !33
  ret void, !dbg !31
}

define internal void @__nv_MAIN__F1L29_1_(i32* %__nv_MAIN__F1L29_1Arg0, i64* %__nv_MAIN__F1L29_1Arg1, i64* %__nv_MAIN__F1L29_1Arg2) #1 !dbg !34 {
L.entry:
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L29_1Arg0, metadata !38, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L29_1Arg1, metadata !40, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L29_1Arg2, metadata !41, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.value(metadata i32 1, metadata !42, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.value(metadata i32 0, metadata !43, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.value(metadata i32 1, metadata !44, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.value(metadata i32 0, metadata !45, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.value(metadata i32 1, metadata !46, metadata !DIExpression()), !dbg !39
  br label %L.LB4_344

L.LB4_344:                                        ; preds = %L.entry
  br label %L.LB4_309

L.LB4_309:                                        ; preds = %L.LB4_344
  call void @global_foo_foo_(), !dbg !47
  br label %L.LB4_310

L.LB4_310:                                        ; preds = %L.LB4_309
  ret void, !dbg !48
}

declare void @fort_init(...) #1

declare signext i32 @__kmpc_global_thread_num(i64*) #1

declare void @__kmpc_fork_call(i64*, i32, i64*, i64*, ...) #1

declare void @__kmpc_end_serialized_parallel(i64*, i32) #1

declare void @__kmpc_serialized_parallel(i64*, i32) #1

declare void @_mp_ecs_nest(...) #1

declare signext i32 @f90io_ldw_end(...) #1

declare signext i32 @f90io_sc_i_ldw(...) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

declare signext i32 @f90io_print_init(...) #1

declare void @f90io_src_info03a(...) #1

declare void @_mp_bcs_nest(...) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { noinline }
attributes #1 = { "no-frame-pointer-elim-non-leaf" }
attributes #2 = { nounwind readnone speculatable }

!llvm.module.flags = !{!11, !12}
!llvm.dbg.cu = !{!5}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "i", scope: !2, file: !3, type: !10, isLocal: true, isDefinition: true)
!2 = distinct !DISubprogram(name: "foo", scope: !4, file: !3, line: 17, type: !8, scopeLine: 17, spFlags: DISPFlagDefinition, unit: !5)
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB082-declared-in-func-orig-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !DIModule(scope: !5, name: "global_foo")
!5 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !6, retainedTypes: !6, globals: !7, imports: !6)
!6 = !{}
!7 = !{!0}
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!11 = !{i32 2, !"Dwarf Version", i32 4}
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !DILocalVariable(name: "omp_sched_static", scope: !2, file: !3, type: !10)
!14 = !DILocation(line: 0, scope: !2)
!15 = !DILocalVariable(name: "omp_proc_bind_false", scope: !2, file: !3, type: !10)
!16 = !DILocalVariable(name: "omp_proc_bind_true", scope: !2, file: !3, type: !10)
!17 = !DILocalVariable(name: "omp_lock_hint_none", scope: !2, file: !3, type: !10)
!18 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !2, file: !3, type: !10)
!19 = !DILocation(line: 19, column: 1, scope: !2)
!20 = !DILocation(line: 20, column: 1, scope: !2)
!21 = !DILocalVariable(scope: !2, file: !3, type: !10, flags: DIFlagArtificial)
!22 = !DILocation(line: 21, column: 1, scope: !2)
!23 = distinct !DISubprogram(name: "drb082_declared_in_func_orig_yes", scope: !5, file: !3, line: 24, type: !24, scopeLine: 24, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !5)
!24 = !DISubroutineType(cc: DW_CC_program, types: !9)
!25 = !DILocalVariable(name: "omp_sched_static", scope: !23, file: !3, type: !10)
!26 = !DILocation(line: 0, scope: !23)
!27 = !DILocalVariable(name: "omp_proc_bind_false", scope: !23, file: !3, type: !10)
!28 = !DILocalVariable(name: "omp_proc_bind_true", scope: !23, file: !3, type: !10)
!29 = !DILocalVariable(name: "omp_lock_hint_none", scope: !23, file: !3, type: !10)
!30 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !23, file: !3, type: !10)
!31 = !DILocation(line: 32, column: 1, scope: !23)
!32 = !DILocation(line: 24, column: 1, scope: !23)
!33 = !DILocation(line: 29, column: 1, scope: !23)
!34 = distinct !DISubprogram(name: "__nv_MAIN__F1L29_1", scope: !5, file: !3, line: 29, type: !35, scopeLine: 29, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !5)
!35 = !DISubroutineType(types: !36)
!36 = !{null, !10, !37, !37}
!37 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!38 = !DILocalVariable(name: "__nv_MAIN__F1L29_1Arg0", arg: 1, scope: !34, file: !3, type: !10)
!39 = !DILocation(line: 0, scope: !34)
!40 = !DILocalVariable(name: "__nv_MAIN__F1L29_1Arg1", arg: 2, scope: !34, file: !3, type: !37)
!41 = !DILocalVariable(name: "__nv_MAIN__F1L29_1Arg2", arg: 3, scope: !34, file: !3, type: !37)
!42 = !DILocalVariable(name: "omp_sched_static", scope: !34, file: !3, type: !10)
!43 = !DILocalVariable(name: "omp_proc_bind_false", scope: !34, file: !3, type: !10)
!44 = !DILocalVariable(name: "omp_proc_bind_true", scope: !34, file: !3, type: !10)
!45 = !DILocalVariable(name: "omp_lock_hint_none", scope: !34, file: !3, type: !10)
!46 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !34, file: !3, type: !10)
!47 = !DILocation(line: 30, column: 1, scope: !34)
!48 = !DILocation(line: 31, column: 1, scope: !34)

; ModuleID = '/tmp/DRB077-single-orig-no-c90325.ll'
source_filename = "/tmp/DRB077-single-orig-no-c90325.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.STATICS1 = type <{ [44 x i8] }>
%astruct.dt60 = type <{ i8* }>

@.STATICS1 = internal global %struct.STATICS1 <{ [44 x i8] c"\FB\FF\FF\FF\07\00\00\00count =\00\EB\FF\FF\FF\00\00\00\00\03\00\00\00\00\00\00\00\01\00\00\00\FF\FF\FF\FF\00\00\00\00" }>, align 16
@.C305_MAIN_ = internal constant i32 25
@.C284_MAIN_ = internal constant i64 0
@.C319_MAIN_ = internal constant i32 6
@.C316_MAIN_ = internal constant [50 x i8] c"micro-benchmarks-fortran/DRB077-single-orig-no.f95"
@.C306_MAIN_ = internal constant i32 23
@.C285_MAIN_ = internal constant i32 1
@.C283_MAIN_ = internal constant i32 0
@.C285___nv_MAIN__F1L17_1 = internal constant i32 1

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__343 = alloca i32, align 4
  %count_307 = alloca i32, align 4
  %.uplevelArgPack0001_338 = alloca %astruct.dt60, align 8
  %z__io_321 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !14, metadata !DIExpression()), !dbg !10
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !15
  store i32 %0, i32* %__gtid_MAIN__343, align 4, !dbg !15
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !16
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !16
  call void (i8*, ...) %2(i8* %1), !dbg !16
  br label %L.LB1_332

L.LB1_332:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %count_307, metadata !17, metadata !DIExpression()), !dbg !10
  store i32 0, i32* %count_307, align 4, !dbg !18
  %3 = bitcast i32* %count_307 to i8*, !dbg !19
  %4 = bitcast %astruct.dt60* %.uplevelArgPack0001_338 to i8**, !dbg !19
  store i8* %3, i8** %4, align 8, !dbg !19
  br label %L.LB1_341, !dbg !19

L.LB1_341:                                        ; preds = %L.LB1_332
  %5 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L17_1_ to i64*, !dbg !19
  %6 = bitcast %astruct.dt60* %.uplevelArgPack0001_338 to i64*, !dbg !19
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %5, i64* %6), !dbg !19
  call void (...) @_mp_bcs_nest(), !dbg !20
  %7 = bitcast i32* @.C306_MAIN_ to i8*, !dbg !20
  %8 = bitcast [50 x i8]* @.C316_MAIN_ to i8*, !dbg !20
  %9 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !20
  call void (i8*, i8*, i64, ...) %9(i8* %7, i8* %8, i64 50), !dbg !20
  %10 = bitcast i32* @.C319_MAIN_ to i8*, !dbg !20
  %11 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !20
  %12 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !20
  %13 = bitcast %struct.STATICS1* @.STATICS1 to i8*, !dbg !20
  %14 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !20
  %15 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %14(i8* %10, i8* null, i8* %11, i8* %12, i8* %13, i8* null, i64 0), !dbg !20
  call void @llvm.dbg.declare(metadata i32* %z__io_321, metadata !21, metadata !DIExpression()), !dbg !10
  store i32 %15, i32* %z__io_321, align 4, !dbg !20
  %16 = load i32, i32* %count_307, align 4, !dbg !20
  call void @llvm.dbg.value(metadata i32 %16, metadata !17, metadata !DIExpression()), !dbg !10
  %17 = bitcast i32 (...)* @f90io_sc_i_fmt_write to i32 (i32, i32, ...)*, !dbg !20
  %18 = call i32 (i32, i32, ...) %17(i32 %16, i32 25), !dbg !20
  store i32 %18, i32* %z__io_321, align 4, !dbg !20
  %19 = call i32 (...) @f90io_fmtw_end(), !dbg !20
  store i32 %19, i32* %z__io_321, align 4, !dbg !20
  call void (...) @_mp_ecs_nest(), !dbg !20
  ret void, !dbg !15
}

define internal void @__nv_MAIN__F1L17_1_(i32* %__nv_MAIN__F1L17_1Arg0, i64* %__nv_MAIN__F1L17_1Arg1, i64* %__nv_MAIN__F1L17_1Arg2) #0 !dbg !22 {
L.entry:
  %__gtid___nv_MAIN__F1L17_1__378 = alloca i32, align 4
  %.s0000_373 = alloca i32, align 4
  %.s0001_374 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L17_1Arg0, metadata !26, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L17_1Arg1, metadata !28, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L17_1Arg2, metadata !29, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.value(metadata i32 1, metadata !30, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.value(metadata i32 0, metadata !31, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.value(metadata i32 1, metadata !32, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.value(metadata i32 0, metadata !33, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.value(metadata i32 1, metadata !34, metadata !DIExpression()), !dbg !27
  %0 = load i32, i32* %__nv_MAIN__F1L17_1Arg0, align 4, !dbg !35
  store i32 %0, i32* %__gtid___nv_MAIN__F1L17_1__378, align 4, !dbg !35
  br label %L.LB2_372

L.LB2_372:                                        ; preds = %L.entry
  br label %L.LB2_310

L.LB2_310:                                        ; preds = %L.LB2_372
  store i32 -1, i32* %.s0000_373, align 4, !dbg !36
  store i32 0, i32* %.s0001_374, align 4, !dbg !36
  %1 = load i32, i32* %__gtid___nv_MAIN__F1L17_1__378, align 4, !dbg !36
  %2 = call i32 @__kmpc_single(i64* null, i32 %1), !dbg !36
  %3 = icmp eq i32 %2, 0, !dbg !36
  br i1 %3, label %L.LB2_330, label %L.LB2_312, !dbg !36

L.LB2_312:                                        ; preds = %L.LB2_310
  %4 = bitcast i64* %__nv_MAIN__F1L17_1Arg2 to i32**, !dbg !37
  %5 = load i32*, i32** %4, align 8, !dbg !37
  %6 = load i32, i32* %5, align 4, !dbg !37
  %7 = add nsw i32 %6, 1, !dbg !37
  %8 = bitcast i64* %__nv_MAIN__F1L17_1Arg2 to i32**, !dbg !37
  %9 = load i32*, i32** %8, align 8, !dbg !37
  store i32 %7, i32* %9, align 4, !dbg !37
  %10 = load i32, i32* %__gtid___nv_MAIN__F1L17_1__378, align 4, !dbg !38
  store i32 %10, i32* %.s0000_373, align 4, !dbg !38
  store i32 1, i32* %.s0001_374, align 4, !dbg !38
  %11 = load i32, i32* %__gtid___nv_MAIN__F1L17_1__378, align 4, !dbg !38
  call void @__kmpc_end_single(i64* null, i32 %11), !dbg !38
  br label %L.LB2_330

L.LB2_330:                                        ; preds = %L.LB2_312, %L.LB2_310
  br label %L.LB2_313

L.LB2_313:                                        ; preds = %L.LB2_330
  %12 = load i32, i32* %__gtid___nv_MAIN__F1L17_1__378, align 4, !dbg !38
  call void @__kmpc_barrier(i64* null, i32 %12), !dbg !38
  br label %L.LB2_314

L.LB2_314:                                        ; preds = %L.LB2_313
  ret void, !dbg !35
}

declare void @__kmpc_barrier(i64*, i32) #0

declare void @__kmpc_end_single(i64*, i32) #0

declare signext i32 @__kmpc_single(i64*, i32) #0

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
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB077-single-orig-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb077_single_orig_no", scope: !2, file: !3, line: 10, type: !6, scopeLine: 10, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!6 = !DISubroutineType(cc: DW_CC_program, types: !7)
!7 = !{null}
!8 = !DILocalVariable(name: "omp_sched_static", scope: !5, file: !3, type: !9)
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 0, scope: !5)
!11 = !DILocalVariable(name: "omp_proc_bind_false", scope: !5, file: !3, type: !9)
!12 = !DILocalVariable(name: "omp_proc_bind_true", scope: !5, file: !3, type: !9)
!13 = !DILocalVariable(name: "omp_lock_hint_none", scope: !5, file: !3, type: !9)
!14 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !5, file: !3, type: !9)
!15 = !DILocation(line: 25, column: 1, scope: !5)
!16 = !DILocation(line: 10, column: 1, scope: !5)
!17 = !DILocalVariable(name: "count", scope: !5, file: !3, type: !9)
!18 = !DILocation(line: 15, column: 1, scope: !5)
!19 = !DILocation(line: 17, column: 1, scope: !5)
!20 = !DILocation(line: 23, column: 1, scope: !5)
!21 = !DILocalVariable(scope: !5, file: !3, type: !9, flags: DIFlagArtificial)
!22 = distinct !DISubprogram(name: "__nv_MAIN__F1L17_1", scope: !2, file: !3, line: 17, type: !23, scopeLine: 17, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!23 = !DISubroutineType(types: !24)
!24 = !{null, !9, !25, !25}
!25 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!26 = !DILocalVariable(name: "__nv_MAIN__F1L17_1Arg0", arg: 1, scope: !22, file: !3, type: !9)
!27 = !DILocation(line: 0, scope: !22)
!28 = !DILocalVariable(name: "__nv_MAIN__F1L17_1Arg1", arg: 2, scope: !22, file: !3, type: !25)
!29 = !DILocalVariable(name: "__nv_MAIN__F1L17_1Arg2", arg: 3, scope: !22, file: !3, type: !25)
!30 = !DILocalVariable(name: "omp_sched_static", scope: !22, file: !3, type: !9)
!31 = !DILocalVariable(name: "omp_proc_bind_false", scope: !22, file: !3, type: !9)
!32 = !DILocalVariable(name: "omp_proc_bind_true", scope: !22, file: !3, type: !9)
!33 = !DILocalVariable(name: "omp_lock_hint_none", scope: !22, file: !3, type: !9)
!34 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !22, file: !3, type: !9)
!35 = !DILocation(line: 21, column: 1, scope: !22)
!36 = !DILocation(line: 18, column: 1, scope: !22)
!37 = !DILocation(line: 19, column: 1, scope: !22)
!38 = !DILocation(line: 20, column: 1, scope: !22)

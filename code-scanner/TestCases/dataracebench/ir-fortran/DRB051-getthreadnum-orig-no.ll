; ModuleID = '/tmp/DRB051-getthreadnum-orig-no-9d2bff.ll'
source_filename = "/tmp/DRB051-getthreadnum-orig-no-9d2bff.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.STATICS1 = type <{ [48 x i8] }>
%astruct.dt60 = type <{ i8* }>

@.STATICS1 = internal global %struct.STATICS1 <{ [48 x i8] c"\FB\FF\FF\FF\0C\00\00\00numThreads =\EB\FF\FF\FF\00\00\00\00\03\00\00\00\00\00\00\00\01\00\00\00\FF\FF\FF\FF\00\00\00\00" }>, align 16
@.C305_MAIN_ = internal constant i32 25
@.C284_MAIN_ = internal constant i64 0
@.C320_MAIN_ = internal constant i32 6
@.C317_MAIN_ = internal constant [56 x i8] c"micro-benchmarks-fortran/DRB051-getthreadnum-orig-no.f95"
@.C306_MAIN_ = internal constant i32 22
@.C283_MAIN_ = internal constant i32 0
@.C283___nv_MAIN__F1L16_1 = internal constant i32 0

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__344 = alloca i32, align 4
  %numthreads_311 = alloca i32, align 4
  %.uplevelArgPack0001_338 = alloca %astruct.dt60, align 8
  %z__io_322 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !14, metadata !DIExpression()), !dbg !10
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !15
  store i32 %0, i32* %__gtid_MAIN__344, align 4, !dbg !15
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !16
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !16
  call void (i8*, ...) %2(i8* %1), !dbg !16
  br label %L.LB1_333

L.LB1_333:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %numthreads_311, metadata !17, metadata !DIExpression()), !dbg !10
  %3 = bitcast i32* %numthreads_311 to i8*, !dbg !18
  %4 = bitcast %astruct.dt60* %.uplevelArgPack0001_338 to i8**, !dbg !18
  store i8* %3, i8** %4, align 8, !dbg !18
  br label %L.LB1_342, !dbg !18

L.LB1_342:                                        ; preds = %L.LB1_333
  %5 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L16_1_ to i64*, !dbg !18
  %6 = bitcast %astruct.dt60* %.uplevelArgPack0001_338 to i64*, !dbg !18
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %5, i64* %6), !dbg !18
  call void (...) @_mp_bcs_nest(), !dbg !19
  %7 = bitcast i32* @.C306_MAIN_ to i8*, !dbg !19
  %8 = bitcast [56 x i8]* @.C317_MAIN_ to i8*, !dbg !19
  %9 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !19
  call void (i8*, i8*, i64, ...) %9(i8* %7, i8* %8, i64 56), !dbg !19
  %10 = bitcast i32* @.C320_MAIN_ to i8*, !dbg !19
  %11 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !19
  %12 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !19
  %13 = bitcast %struct.STATICS1* @.STATICS1 to i8*, !dbg !19
  %14 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !19
  %15 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %14(i8* %10, i8* null, i8* %11, i8* %12, i8* %13, i8* null, i64 0), !dbg !19
  call void @llvm.dbg.declare(metadata i32* %z__io_322, metadata !20, metadata !DIExpression()), !dbg !10
  store i32 %15, i32* %z__io_322, align 4, !dbg !19
  %16 = load i32, i32* %numthreads_311, align 4, !dbg !19
  call void @llvm.dbg.value(metadata i32 %16, metadata !17, metadata !DIExpression()), !dbg !10
  %17 = bitcast i32 (...)* @f90io_sc_i_fmt_write to i32 (i32, i32, ...)*, !dbg !19
  %18 = call i32 (i32, i32, ...) %17(i32 %16, i32 25), !dbg !19
  store i32 %18, i32* %z__io_322, align 4, !dbg !19
  %19 = call i32 (...) @f90io_fmtw_end(), !dbg !19
  store i32 %19, i32* %z__io_322, align 4, !dbg !19
  call void (...) @_mp_ecs_nest(), !dbg !19
  ret void, !dbg !15
}

define internal void @__nv_MAIN__F1L16_1_(i32* %__nv_MAIN__F1L16_1Arg0, i64* %__nv_MAIN__F1L16_1Arg1, i64* %__nv_MAIN__F1L16_1Arg2) #0 !dbg !21 {
L.entry:
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L16_1Arg0, metadata !25, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L16_1Arg1, metadata !27, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L16_1Arg2, metadata !28, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.value(metadata i32 1, metadata !29, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.value(metadata i32 0, metadata !30, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.value(metadata i32 1, metadata !31, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.value(metadata i32 0, metadata !32, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.value(metadata i32 1, metadata !33, metadata !DIExpression()), !dbg !26
  br label %L.LB2_373

L.LB2_373:                                        ; preds = %L.entry
  br label %L.LB2_314

L.LB2_314:                                        ; preds = %L.LB2_373
  %0 = call i32 (...) @omp_get_thread_num_(), !dbg !34
  %1 = icmp ne i32 %0, 0, !dbg !34
  br i1 %1, label %L.LB2_331, label %L.LB2_377, !dbg !34

L.LB2_377:                                        ; preds = %L.LB2_314
  %2 = call i32 (...) @omp_get_num_threads_(), !dbg !35
  %3 = bitcast i64* %__nv_MAIN__F1L16_1Arg2 to i32**, !dbg !35
  %4 = load i32*, i32** %3, align 8, !dbg !35
  store i32 %2, i32* %4, align 4, !dbg !35
  br label %L.LB2_331

L.LB2_331:                                        ; preds = %L.LB2_377, %L.LB2_314
  br label %L.LB2_315

L.LB2_315:                                        ; preds = %L.LB2_331
  ret void, !dbg !36
}

declare signext i32 @omp_get_num_threads_(...) #0

declare signext i32 @omp_get_thread_num_(...) #0

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
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB051-getthreadnum-orig-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb051_getthreadnum_orig_no", scope: !2, file: !3, line: 10, type: !6, scopeLine: 10, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!6 = !DISubroutineType(cc: DW_CC_program, types: !7)
!7 = !{null}
!8 = !DILocalVariable(name: "omp_sched_static", scope: !5, file: !3, type: !9)
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 0, scope: !5)
!11 = !DILocalVariable(name: "omp_proc_bind_false", scope: !5, file: !3, type: !9)
!12 = !DILocalVariable(name: "omp_proc_bind_true", scope: !5, file: !3, type: !9)
!13 = !DILocalVariable(name: "omp_lock_hint_none", scope: !5, file: !3, type: !9)
!14 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !5, file: !3, type: !9)
!15 = !DILocation(line: 24, column: 1, scope: !5)
!16 = !DILocation(line: 10, column: 1, scope: !5)
!17 = !DILocalVariable(name: "numthreads", scope: !5, file: !3, type: !9)
!18 = !DILocation(line: 16, column: 1, scope: !5)
!19 = !DILocation(line: 22, column: 1, scope: !5)
!20 = !DILocalVariable(scope: !5, file: !3, type: !9, flags: DIFlagArtificial)
!21 = distinct !DISubprogram(name: "__nv_MAIN__F1L16_1", scope: !2, file: !3, line: 16, type: !22, scopeLine: 16, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!22 = !DISubroutineType(types: !23)
!23 = !{null, !9, !24, !24}
!24 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!25 = !DILocalVariable(name: "__nv_MAIN__F1L16_1Arg0", arg: 1, scope: !21, file: !3, type: !9)
!26 = !DILocation(line: 0, scope: !21)
!27 = !DILocalVariable(name: "__nv_MAIN__F1L16_1Arg1", arg: 2, scope: !21, file: !3, type: !24)
!28 = !DILocalVariable(name: "__nv_MAIN__F1L16_1Arg2", arg: 3, scope: !21, file: !3, type: !24)
!29 = !DILocalVariable(name: "omp_sched_static", scope: !21, file: !3, type: !9)
!30 = !DILocalVariable(name: "omp_proc_bind_false", scope: !21, file: !3, type: !9)
!31 = !DILocalVariable(name: "omp_proc_bind_true", scope: !21, file: !3, type: !9)
!32 = !DILocalVariable(name: "omp_lock_hint_none", scope: !21, file: !3, type: !9)
!33 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !21, file: !3, type: !9)
!34 = !DILocation(line: 17, column: 1, scope: !21)
!35 = !DILocation(line: 18, column: 1, scope: !21)
!36 = !DILocation(line: 20, column: 1, scope: !21)

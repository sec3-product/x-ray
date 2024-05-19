; ModuleID = '/tmp/DRB103-master-orig-no-718478.ll'
source_filename = "/tmp/DRB103-master-orig-no-718478.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.STATICS1 = type <{ [76 x i8] }>
%astruct.dt60 = type <{ i8* }>

@.STATICS1 = internal global %struct.STATICS1 <{ [76 x i8] c"\FB\FF\FF\FF\1D\00\00\00Number of threads requested =\00\00\00\00\00\00\00\03\00\00\00\EB\FF\FF\FF\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\FF\FF\FF\FF\00\00\00\00" }>, align 16
@.C306_MAIN_ = internal constant i32 25
@.C284_MAIN_ = internal constant i64 0
@.C317_MAIN_ = internal constant i32 6
@.C314_MAIN_ = internal constant [50 x i8] c"micro-benchmarks-fortran/DRB103-master-orig-no.f95"
@.C305_MAIN_ = internal constant i32 19
@.C283_MAIN_ = internal constant i32 0
@.C306___nv_MAIN__F1L16_1 = internal constant i32 25
@.C283___nv_MAIN__F1L16_1 = internal constant i32 0
@.C284___nv_MAIN__F1L16_1 = internal constant i64 0
@.C317___nv_MAIN__F1L16_1 = internal constant i32 6
@.C314___nv_MAIN__F1L16_1 = internal constant [50 x i8] c"micro-benchmarks-fortran/DRB103-master-orig-no.f95"
@.C305___nv_MAIN__F1L16_1 = internal constant i32 19

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__342 = alloca i32, align 4
  %k_309 = alloca i32, align 4
  %.uplevelArgPack0001_336 = alloca %astruct.dt60, align 8
  call void @llvm.dbg.value(metadata i32 1, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !14, metadata !DIExpression()), !dbg !10
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !15
  store i32 %0, i32* %__gtid_MAIN__342, align 4, !dbg !15
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !16
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !16
  call void (i8*, ...) %2(i8* %1), !dbg !16
  br label %L.LB1_331

L.LB1_331:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %k_309, metadata !17, metadata !DIExpression()), !dbg !10
  %3 = bitcast i32* %k_309 to i8*, !dbg !18
  %4 = bitcast %astruct.dt60* %.uplevelArgPack0001_336 to i8**, !dbg !18
  store i8* %3, i8** %4, align 8, !dbg !18
  br label %L.LB1_340, !dbg !18

L.LB1_340:                                        ; preds = %L.LB1_331
  %5 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L16_1_ to i64*, !dbg !18
  %6 = bitcast %astruct.dt60* %.uplevelArgPack0001_336 to i64*, !dbg !18
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %5, i64* %6), !dbg !18
  ret void, !dbg !15
}

define internal void @__nv_MAIN__F1L16_1_(i32* %__nv_MAIN__F1L16_1Arg0, i64* %__nv_MAIN__F1L16_1Arg1, i64* %__nv_MAIN__F1L16_1Arg2) #0 !dbg !19 {
L.entry:
  %__gtid___nv_MAIN__F1L16_1__364 = alloca i32, align 4
  %z__io_319 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L16_1Arg0, metadata !23, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L16_1Arg1, metadata !25, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L16_1Arg2, metadata !26, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.value(metadata i32 1, metadata !27, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.value(metadata i32 0, metadata !28, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.value(metadata i32 1, metadata !29, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.value(metadata i32 0, metadata !30, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.value(metadata i32 1, metadata !31, metadata !DIExpression()), !dbg !24
  %0 = load i32, i32* %__nv_MAIN__F1L16_1Arg0, align 4, !dbg !32
  store i32 %0, i32* %__gtid___nv_MAIN__F1L16_1__364, align 4, !dbg !32
  br label %L.LB2_363

L.LB2_363:                                        ; preds = %L.entry
  br label %L.LB2_312

L.LB2_312:                                        ; preds = %L.LB2_363
  %1 = load i32, i32* %__gtid___nv_MAIN__F1L16_1__364, align 4, !dbg !33
  %2 = call i32 @__kmpc_master(i64* null, i32 %1), !dbg !33
  %3 = icmp eq i32 %2, 0, !dbg !33
  br i1 %3, label %L.LB2_329, label %L.LB2_385, !dbg !33

L.LB2_385:                                        ; preds = %L.LB2_312
  %4 = call i32 (...) @omp_get_num_threads_(), !dbg !34
  %5 = bitcast i64* %__nv_MAIN__F1L16_1Arg2 to i32**, !dbg !34
  %6 = load i32*, i32** %5, align 8, !dbg !34
  store i32 %4, i32* %6, align 4, !dbg !34
  call void (...) @_mp_bcs_nest(), !dbg !35
  %7 = bitcast i32* @.C305___nv_MAIN__F1L16_1 to i8*, !dbg !35
  %8 = bitcast [50 x i8]* @.C314___nv_MAIN__F1L16_1 to i8*, !dbg !35
  %9 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !35
  call void (i8*, i8*, i64, ...) %9(i8* %7, i8* %8, i64 50), !dbg !35
  %10 = bitcast i32* @.C317___nv_MAIN__F1L16_1 to i8*, !dbg !35
  %11 = bitcast i32* @.C283___nv_MAIN__F1L16_1 to i8*, !dbg !35
  %12 = bitcast i32* @.C283___nv_MAIN__F1L16_1 to i8*, !dbg !35
  %13 = bitcast %struct.STATICS1* @.STATICS1 to i8*, !dbg !35
  %14 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !35
  %15 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %14(i8* %10, i8* null, i8* %11, i8* %12, i8* %13, i8* null, i64 0), !dbg !35
  call void @llvm.dbg.declare(metadata i32* %z__io_319, metadata !36, metadata !DIExpression()), !dbg !24
  store i32 %15, i32* %z__io_319, align 4, !dbg !35
  %16 = bitcast i64* %__nv_MAIN__F1L16_1Arg2 to i32**, !dbg !35
  %17 = load i32*, i32** %16, align 8, !dbg !35
  %18 = load i32, i32* %17, align 4, !dbg !35
  %19 = bitcast i32 (...)* @f90io_sc_i_fmt_write to i32 (i32, i32, ...)*, !dbg !35
  %20 = call i32 (i32, i32, ...) %19(i32 %18, i32 25), !dbg !35
  store i32 %20, i32* %z__io_319, align 4, !dbg !35
  %21 = call i32 (...) @f90io_fmtw_end(), !dbg !35
  store i32 %21, i32* %z__io_319, align 4, !dbg !35
  call void (...) @_mp_ecs_nest(), !dbg !35
  %22 = load i32, i32* %__gtid___nv_MAIN__F1L16_1__364, align 4, !dbg !37
  call void @__kmpc_end_master(i64* null, i32 %22), !dbg !37
  br label %L.LB2_329

L.LB2_329:                                        ; preds = %L.LB2_385, %L.LB2_312
  br label %L.LB2_325

L.LB2_325:                                        ; preds = %L.LB2_329
  ret void, !dbg !32
}

declare void @_mp_ecs_nest(...) #0

declare signext i32 @f90io_fmtw_end(...) #0

declare signext i32 @f90io_sc_i_fmt_write(...) #0

declare signext i32 @f90io_fmtw_inita(...) #0

declare void @f90io_src_info03a(...) #0

declare void @_mp_bcs_nest(...) #0

declare signext i32 @omp_get_num_threads_(...) #0

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
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB103-master-orig-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb103_master_orig_no", scope: !2, file: !3, line: 10, type: !6, scopeLine: 10, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
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
!17 = !DILocalVariable(name: "k", scope: !5, file: !3, type: !9)
!18 = !DILocation(line: 16, column: 1, scope: !5)
!19 = distinct !DISubprogram(name: "__nv_MAIN__F1L16_1", scope: !2, file: !3, line: 16, type: !20, scopeLine: 16, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!20 = !DISubroutineType(types: !21)
!21 = !{null, !9, !22, !22}
!22 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!23 = !DILocalVariable(name: "__nv_MAIN__F1L16_1Arg0", arg: 1, scope: !19, file: !3, type: !9)
!24 = !DILocation(line: 0, scope: !19)
!25 = !DILocalVariable(name: "__nv_MAIN__F1L16_1Arg1", arg: 2, scope: !19, file: !3, type: !22)
!26 = !DILocalVariable(name: "__nv_MAIN__F1L16_1Arg2", arg: 3, scope: !19, file: !3, type: !22)
!27 = !DILocalVariable(name: "omp_sched_static", scope: !19, file: !3, type: !9)
!28 = !DILocalVariable(name: "omp_proc_bind_false", scope: !19, file: !3, type: !9)
!29 = !DILocalVariable(name: "omp_proc_bind_true", scope: !19, file: !3, type: !9)
!30 = !DILocalVariable(name: "omp_lock_hint_none", scope: !19, file: !3, type: !9)
!31 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !19, file: !3, type: !9)
!32 = !DILocation(line: 22, column: 1, scope: !19)
!33 = !DILocation(line: 17, column: 1, scope: !19)
!34 = !DILocation(line: 18, column: 1, scope: !19)
!35 = !DILocation(line: 19, column: 1, scope: !19)
!36 = !DILocalVariable(scope: !19, file: !3, type: !9, flags: DIFlagArtificial)
!37 = !DILocation(line: 21, column: 1, scope: !19)

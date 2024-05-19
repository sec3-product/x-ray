; ModuleID = '/tmp/DRB075-getthreadnum-orig-yes-6d7fc1.ll'
source_filename = "/tmp/DRB075-getthreadnum-orig-yes-6d7fc1.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%astruct.dt60 = type <{ i8* }>

@.C306_MAIN_ = internal constant i32 25
@.C305_MAIN_ = internal constant i32 14
@.C322_MAIN_ = internal constant [12 x i8] c"numThreads ="
@.C284_MAIN_ = internal constant i64 0
@.C319_MAIN_ = internal constant i32 6
@.C316_MAIN_ = internal constant [57 x i8] c"micro-benchmarks-fortran/DRB075-getthreadnum-orig-yes.f95"
@.C318_MAIN_ = internal constant i32 24
@.C283_MAIN_ = internal constant i32 0
@.C306___nv_MAIN__F1L20_1 = internal constant i32 25
@.C305___nv_MAIN__F1L20_1 = internal constant i32 14
@.C322___nv_MAIN__F1L20_1 = internal constant [12 x i8] c"numThreads ="
@.C284___nv_MAIN__F1L20_1 = internal constant i64 0
@.C319___nv_MAIN__F1L20_1 = internal constant i32 6
@.C316___nv_MAIN__F1L20_1 = internal constant [57 x i8] c"micro-benchmarks-fortran/DRB075-getthreadnum-orig-yes.f95"
@.C318___nv_MAIN__F1L20_1 = internal constant i32 24
@.C283___nv_MAIN__F1L20_1 = internal constant i32 0

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__346 = alloca i32, align 4
  %numthreads_311 = alloca i32, align 4
  %.uplevelArgPack0001_341 = alloca %astruct.dt60, align 8
  call void @llvm.dbg.value(metadata i32 1, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !14, metadata !DIExpression()), !dbg !10
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !15
  store i32 %0, i32* %__gtid_MAIN__346, align 4, !dbg !15
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !16
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !16
  call void (i8*, ...) %2(i8* %1), !dbg !16
  br label %L.LB1_335

L.LB1_335:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %numthreads_311, metadata !17, metadata !DIExpression()), !dbg !10
  store i32 0, i32* %numthreads_311, align 4, !dbg !18
  %3 = bitcast i32* %numthreads_311 to i8*, !dbg !19
  %4 = bitcast %astruct.dt60* %.uplevelArgPack0001_341 to i8**, !dbg !19
  store i8* %3, i8** %4, align 8, !dbg !19
  br label %L.LB1_344, !dbg !19

L.LB1_344:                                        ; preds = %L.LB1_335
  %5 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L20_1_ to i64*, !dbg !19
  %6 = bitcast %astruct.dt60* %.uplevelArgPack0001_341 to i64*, !dbg !19
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %5, i64* %6), !dbg !19
  ret void, !dbg !15
}

define internal void @__nv_MAIN__F1L20_1_(i32* %__nv_MAIN__F1L20_1Arg0, i64* %__nv_MAIN__F1L20_1Arg1, i64* %__nv_MAIN__F1L20_1Arg2) #0 !dbg !20 {
L.entry:
  %z__io_321 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L20_1Arg0, metadata !24, metadata !DIExpression()), !dbg !25
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L20_1Arg1, metadata !26, metadata !DIExpression()), !dbg !25
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L20_1Arg2, metadata !27, metadata !DIExpression()), !dbg !25
  call void @llvm.dbg.value(metadata i32 1, metadata !28, metadata !DIExpression()), !dbg !25
  call void @llvm.dbg.value(metadata i32 0, metadata !29, metadata !DIExpression()), !dbg !25
  call void @llvm.dbg.value(metadata i32 1, metadata !30, metadata !DIExpression()), !dbg !25
  call void @llvm.dbg.value(metadata i32 0, metadata !31, metadata !DIExpression()), !dbg !25
  call void @llvm.dbg.value(metadata i32 1, metadata !32, metadata !DIExpression()), !dbg !25
  br label %L.LB2_367

L.LB2_367:                                        ; preds = %L.entry
  br label %L.LB2_314

L.LB2_314:                                        ; preds = %L.LB2_367
  %0 = call i32 (...) @omp_get_thread_num_(), !dbg !33
  %1 = icmp ne i32 %0, 0, !dbg !33
  br i1 %1, label %L.LB2_332, label %L.LB2_381, !dbg !33

L.LB2_381:                                        ; preds = %L.LB2_314
  %2 = call i32 (...) @omp_get_num_threads_(), !dbg !34
  %3 = bitcast i64* %__nv_MAIN__F1L20_1Arg2 to i32**, !dbg !34
  %4 = load i32*, i32** %3, align 8, !dbg !34
  store i32 %2, i32* %4, align 4, !dbg !34
  br label %L.LB2_333, !dbg !35

L.LB2_332:                                        ; preds = %L.LB2_314
  call void (...) @_mp_bcs_nest(), !dbg !36
  %5 = bitcast i32* @.C318___nv_MAIN__F1L20_1 to i8*, !dbg !36
  %6 = bitcast [57 x i8]* @.C316___nv_MAIN__F1L20_1 to i8*, !dbg !36
  %7 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !36
  call void (i8*, i8*, i64, ...) %7(i8* %5, i8* %6, i64 57), !dbg !36
  %8 = bitcast i32* @.C319___nv_MAIN__F1L20_1 to i8*, !dbg !36
  %9 = bitcast i32* @.C283___nv_MAIN__F1L20_1 to i8*, !dbg !36
  %10 = bitcast i32* @.C283___nv_MAIN__F1L20_1 to i8*, !dbg !36
  %11 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !36
  %12 = call i32 (i8*, i8*, i8*, i8*, ...) %11(i8* %8, i8* null, i8* %9, i8* %10), !dbg !36
  call void @llvm.dbg.declare(metadata i32* %z__io_321, metadata !37, metadata !DIExpression()), !dbg !25
  store i32 %12, i32* %z__io_321, align 4, !dbg !36
  %13 = bitcast [12 x i8]* @.C322___nv_MAIN__F1L20_1 to i8*, !dbg !36
  %14 = bitcast i32 (...)* @f90io_sc_ch_ldw to i32 (i8*, i32, i64, ...)*, !dbg !36
  %15 = call i32 (i8*, i32, i64, ...) %14(i8* %13, i32 14, i64 12), !dbg !36
  store i32 %15, i32* %z__io_321, align 4, !dbg !36
  %16 = bitcast i64* %__nv_MAIN__F1L20_1Arg2 to i32**, !dbg !36
  %17 = load i32*, i32** %16, align 8, !dbg !36
  %18 = load i32, i32* %17, align 4, !dbg !36
  %19 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !36
  %20 = call i32 (i32, i32, ...) %19(i32 %18, i32 25), !dbg !36
  store i32 %20, i32* %z__io_321, align 4, !dbg !36
  %21 = call i32 (...) @f90io_ldw_end(), !dbg !36
  store i32 %21, i32* %z__io_321, align 4, !dbg !36
  call void (...) @_mp_ecs_nest(), !dbg !36
  br label %L.LB2_333

L.LB2_333:                                        ; preds = %L.LB2_332, %L.LB2_381
  br label %L.LB2_328

L.LB2_328:                                        ; preds = %L.LB2_333
  ret void, !dbg !38
}

declare void @_mp_ecs_nest(...) #0

declare signext i32 @f90io_ldw_end(...) #0

declare signext i32 @f90io_sc_i_ldw(...) #0

declare signext i32 @f90io_sc_ch_ldw(...) #0

declare signext i32 @f90io_print_init(...) #0

declare void @f90io_src_info03a(...) #0

declare void @_mp_bcs_nest(...) #0

declare signext i32 @omp_get_num_threads_(...) #0

declare signext i32 @omp_get_thread_num_(...) #0

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
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB075-getthreadnum-orig-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb075_getthreadnum_orig_yes", scope: !2, file: !3, line: 13, type: !6, scopeLine: 13, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
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
!17 = !DILocalVariable(name: "numthreads", scope: !5, file: !3, type: !9)
!18 = !DILocation(line: 18, column: 1, scope: !5)
!19 = !DILocation(line: 20, column: 1, scope: !5)
!20 = distinct !DISubprogram(name: "__nv_MAIN__F1L20_1", scope: !2, file: !3, line: 20, type: !21, scopeLine: 20, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!21 = !DISubroutineType(types: !22)
!22 = !{null, !9, !23, !23}
!23 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!24 = !DILocalVariable(name: "__nv_MAIN__F1L20_1Arg0", arg: 1, scope: !20, file: !3, type: !9)
!25 = !DILocation(line: 0, scope: !20)
!26 = !DILocalVariable(name: "__nv_MAIN__F1L20_1Arg1", arg: 2, scope: !20, file: !3, type: !23)
!27 = !DILocalVariable(name: "__nv_MAIN__F1L20_1Arg2", arg: 3, scope: !20, file: !3, type: !23)
!28 = !DILocalVariable(name: "omp_sched_static", scope: !20, file: !3, type: !9)
!29 = !DILocalVariable(name: "omp_proc_bind_false", scope: !20, file: !3, type: !9)
!30 = !DILocalVariable(name: "omp_proc_bind_true", scope: !20, file: !3, type: !9)
!31 = !DILocalVariable(name: "omp_lock_hint_none", scope: !20, file: !3, type: !9)
!32 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !20, file: !3, type: !9)
!33 = !DILocation(line: 21, column: 1, scope: !20)
!34 = !DILocation(line: 22, column: 1, scope: !20)
!35 = !DILocation(line: 23, column: 1, scope: !20)
!36 = !DILocation(line: 24, column: 1, scope: !20)
!37 = !DILocalVariable(scope: !20, file: !3, type: !9, flags: DIFlagArtificial)
!38 = !DILocation(line: 26, column: 1, scope: !20)

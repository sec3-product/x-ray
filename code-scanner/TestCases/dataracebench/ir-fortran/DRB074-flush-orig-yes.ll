; ModuleID = '/tmp/DRB074-flush-orig-yes-2e8a45.ll'
source_filename = "/tmp/DRB074-flush-orig-yes-2e8a45.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct__cs_unspc_ = type <{ [32 x i8] }>
%astruct.dt63 = type <{ i8*, i8* }>

@.C285_drb074_f1_ = internal constant i32 1
@.C307_MAIN_ = internal constant i32 25
@.C306_MAIN_ = internal constant i32 14
@.C325_MAIN_ = internal constant [5 x i8] c"sum ="
@.C284_MAIN_ = internal constant i64 0
@.C322_MAIN_ = internal constant i32 6
@.C319_MAIN_ = internal constant [50 x i8] c"micro-benchmarks-fortran/DRB074-flush-orig-yes.f95"
@.C321_MAIN_ = internal constant i32 42
@.C334_MAIN_ = internal constant i32 10
@.C312_MAIN_ = internal constant i32 10
@.C283_MAIN_ = internal constant i32 0
@.C283___nv_MAIN__F1L36_1 = internal constant i32 0
@__cs_unspc_ = common global %struct__cs_unspc_ zeroinitializer, align 64

; Function Attrs: noinline
define float @drb074_() #0 {
.L.entry:
  ret float undef
}

define void @drb074_f1_(i64* %q) #1 !dbg !5 {
L.entry:
  %__gtid_drb074_f1__315 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i64* %q, metadata !10, metadata !DIExpression()), !dbg !11
  call void @llvm.dbg.value(metadata i32 1, metadata !12, metadata !DIExpression()), !dbg !11
  call void @llvm.dbg.value(metadata i32 0, metadata !13, metadata !DIExpression()), !dbg !11
  call void @llvm.dbg.value(metadata i32 1, metadata !14, metadata !DIExpression()), !dbg !11
  call void @llvm.dbg.value(metadata i32 0, metadata !15, metadata !DIExpression()), !dbg !11
  call void @llvm.dbg.value(metadata i32 1, metadata !16, metadata !DIExpression()), !dbg !11
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !17
  store i32 %0, i32* %__gtid_drb074_f1__315, align 4, !dbg !17
  br label %L.LB2_314

L.LB2_314:                                        ; preds = %L.entry
  %1 = load i32, i32* %__gtid_drb074_f1__315, align 4, !dbg !18
  %2 = bitcast %struct__cs_unspc_* @__cs_unspc_ to i64*, !dbg !18
  call void @__kmpc_critical(i64* null, i32 %1, i64* %2), !dbg !18
  %3 = bitcast i64* %q to i32*, !dbg !18
  store i32 1, i32* %3, align 4, !dbg !18
  %4 = load i32, i32* %__gtid_drb074_f1__315, align 4, !dbg !18
  %5 = bitcast %struct__cs_unspc_* @__cs_unspc_ to i64*, !dbg !18
  call void @__kmpc_end_critical(i64* null, i32 %4, i64* %5), !dbg !18
  call void @__kmpc_flush(i64* null), !dbg !19
  ret void, !dbg !17
}

define void @MAIN_() #1 !dbg !20 {
L.entry:
  %__gtid_MAIN__351 = alloca i32, align 4
  %i_310 = alloca i32, align 4
  %sum_311 = alloca i32, align 4
  %.uplevelArgPack0001_344 = alloca %astruct.dt63, align 16
  %z__io_324 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !23, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.value(metadata i32 0, metadata !25, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.value(metadata i32 1, metadata !26, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.value(metadata i32 0, metadata !27, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.value(metadata i32 1, metadata !28, metadata !DIExpression()), !dbg !24
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !29
  store i32 %0, i32* %__gtid_MAIN__351, align 4, !dbg !29
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !30
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !30
  call void (i8*, ...) %2(i8* %1), !dbg !30
  br label %L.LB3_337

L.LB3_337:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %i_310, metadata !31, metadata !DIExpression()), !dbg !24
  store i32 0, i32* %i_310, align 4, !dbg !32
  call void @llvm.dbg.declare(metadata i32* %sum_311, metadata !33, metadata !DIExpression()), !dbg !24
  store i32 0, i32* %sum_311, align 4, !dbg !34
  %3 = bitcast i32* %sum_311 to i8*, !dbg !35
  %4 = bitcast %astruct.dt63* %.uplevelArgPack0001_344 to i8**, !dbg !35
  store i8* %3, i8** %4, align 8, !dbg !35
  %5 = bitcast i32* %i_310 to i8*, !dbg !35
  %6 = bitcast %astruct.dt63* %.uplevelArgPack0001_344 to i8*, !dbg !35
  %7 = getelementptr i8, i8* %6, i64 8, !dbg !35
  %8 = bitcast i8* %7 to i8**, !dbg !35
  store i8* %5, i8** %8, align 8, !dbg !35
  br label %L.LB3_349, !dbg !35

L.LB3_349:                                        ; preds = %L.LB3_337
  %9 = load i32, i32* %__gtid_MAIN__351, align 4, !dbg !35
  call void @__kmpc_push_num_threads(i64* null, i32 %9, i32 10), !dbg !35
  %10 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L36_1_ to i64*, !dbg !35
  %11 = bitcast %astruct.dt63* %.uplevelArgPack0001_344 to i64*, !dbg !35
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %10, i64* %11), !dbg !35
  %12 = load i32, i32* %sum_311, align 4, !dbg !36
  call void @llvm.dbg.value(metadata i32 %12, metadata !33, metadata !DIExpression()), !dbg !24
  %13 = icmp eq i32 %12, 10, !dbg !36
  br i1 %13, label %L.LB3_335, label %L.LB3_382, !dbg !36

L.LB3_382:                                        ; preds = %L.LB3_349
  call void (...) @_mp_bcs_nest(), !dbg !37
  %14 = bitcast i32* @.C321_MAIN_ to i8*, !dbg !37
  %15 = bitcast [50 x i8]* @.C319_MAIN_ to i8*, !dbg !37
  %16 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !37
  call void (i8*, i8*, i64, ...) %16(i8* %14, i8* %15, i64 50), !dbg !37
  %17 = bitcast i32* @.C322_MAIN_ to i8*, !dbg !37
  %18 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !37
  %19 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !37
  %20 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !37
  %21 = call i32 (i8*, i8*, i8*, i8*, ...) %20(i8* %17, i8* null, i8* %18, i8* %19), !dbg !37
  call void @llvm.dbg.declare(metadata i32* %z__io_324, metadata !38, metadata !DIExpression()), !dbg !24
  store i32 %21, i32* %z__io_324, align 4, !dbg !37
  %22 = bitcast [5 x i8]* @.C325_MAIN_ to i8*, !dbg !37
  %23 = bitcast i32 (...)* @f90io_sc_ch_ldw to i32 (i8*, i32, i64, ...)*, !dbg !37
  %24 = call i32 (i8*, i32, i64, ...) %23(i8* %22, i32 14, i64 5), !dbg !37
  store i32 %24, i32* %z__io_324, align 4, !dbg !37
  %25 = load i32, i32* %sum_311, align 4, !dbg !37
  call void @llvm.dbg.value(metadata i32 %25, metadata !33, metadata !DIExpression()), !dbg !24
  %26 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !37
  %27 = call i32 (i32, i32, ...) %26(i32 %25, i32 25), !dbg !37
  store i32 %27, i32* %z__io_324, align 4, !dbg !37
  %28 = call i32 (...) @f90io_ldw_end(), !dbg !37
  store i32 %28, i32* %z__io_324, align 4, !dbg !37
  call void (...) @_mp_ecs_nest(), !dbg !37
  br label %L.LB3_335

L.LB3_335:                                        ; preds = %L.LB3_382, %L.LB3_349
  ret void, !dbg !29
}

define internal void @__nv_MAIN__F1L36_1_(i32* %__nv_MAIN__F1L36_1Arg0, i64* %__nv_MAIN__F1L36_1Arg1, i64* %__nv_MAIN__F1L36_1Arg2) #1 !dbg !39 {
L.entry:
  %sum_316 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L36_1Arg0, metadata !43, metadata !DIExpression()), !dbg !44
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L36_1Arg1, metadata !45, metadata !DIExpression()), !dbg !44
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L36_1Arg2, metadata !46, metadata !DIExpression()), !dbg !44
  call void @llvm.dbg.value(metadata i32 1, metadata !47, metadata !DIExpression()), !dbg !44
  call void @llvm.dbg.value(metadata i32 0, metadata !48, metadata !DIExpression()), !dbg !44
  call void @llvm.dbg.value(metadata i32 1, metadata !49, metadata !DIExpression()), !dbg !44
  call void @llvm.dbg.value(metadata i32 0, metadata !50, metadata !DIExpression()), !dbg !44
  call void @llvm.dbg.value(metadata i32 1, metadata !51, metadata !DIExpression()), !dbg !44
  br label %L.LB4_386

L.LB4_386:                                        ; preds = %L.entry
  br label %L.LB4_315

L.LB4_315:                                        ; preds = %L.LB4_386
  call void @llvm.dbg.declare(metadata i32* %sum_316, metadata !52, metadata !DIExpression()), !dbg !53
  store i32 0, i32* %sum_316, align 4, !dbg !54
  %0 = bitcast i64* %__nv_MAIN__F1L36_1Arg2 to i8*, !dbg !55
  %1 = getelementptr i8, i8* %0, i64 8, !dbg !55
  %2 = bitcast i8* %1 to i64**, !dbg !55
  %3 = load i64*, i64** %2, align 8, !dbg !55
  call void @drb074_f1_(i64* %3), !dbg !55
  %4 = load i32, i32* %sum_316, align 4, !dbg !56
  call void @llvm.dbg.value(metadata i32 %4, metadata !52, metadata !DIExpression()), !dbg !53
  %5 = bitcast i64* %__nv_MAIN__F1L36_1Arg2 to i8*, !dbg !56
  %6 = getelementptr i8, i8* %5, i64 8, !dbg !56
  %7 = bitcast i8* %6 to i32**, !dbg !56
  %8 = load i32*, i32** %7, align 8, !dbg !56
  %9 = load i32, i32* %8, align 4, !dbg !56
  %10 = add nsw i32 %4, %9, !dbg !56
  store i32 %10, i32* %sum_316, align 4, !dbg !56
  %11 = call i32 (...) @_mp_bcs_nest_red(), !dbg !53
  %12 = call i32 (...) @_mp_bcs_nest_red(), !dbg !53
  %13 = load i32, i32* %sum_316, align 4, !dbg !53
  call void @llvm.dbg.value(metadata i32 %13, metadata !52, metadata !DIExpression()), !dbg !53
  %14 = bitcast i64* %__nv_MAIN__F1L36_1Arg2 to i32**, !dbg !53
  %15 = load i32*, i32** %14, align 8, !dbg !53
  %16 = load i32, i32* %15, align 4, !dbg !53
  %17 = add nsw i32 %13, %16, !dbg !53
  %18 = bitcast i64* %__nv_MAIN__F1L36_1Arg2 to i32**, !dbg !53
  %19 = load i32*, i32** %18, align 8, !dbg !53
  store i32 %17, i32* %19, align 4, !dbg !53
  %20 = call i32 (...) @_mp_ecs_nest_red(), !dbg !53
  %21 = call i32 (...) @_mp_ecs_nest_red(), !dbg !53
  br label %L.LB4_317

L.LB4_317:                                        ; preds = %L.LB4_315
  ret void, !dbg !53
}

declare signext i32 @_mp_ecs_nest_red(...) #1

declare signext i32 @_mp_bcs_nest_red(...) #1

declare void @_mp_ecs_nest(...) #1

declare signext i32 @f90io_ldw_end(...) #1

declare signext i32 @f90io_sc_i_ldw(...) #1

declare signext i32 @f90io_sc_ch_ldw(...) #1

declare signext i32 @f90io_print_init(...) #1

declare void @f90io_src_info03a(...) #1

declare void @_mp_bcs_nest(...) #1

declare void @fort_init(...) #1

declare void @__kmpc_fork_call(i64*, i32, i64*, i64*, ...) #1

declare void @__kmpc_push_num_threads(i64*, i32, i32) #1

declare void @__kmpc_end_serialized_parallel(i64*, i32) #1

declare void @__kmpc_serialized_parallel(i64*, i32) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

declare signext i32 @__kmpc_global_thread_num(i64*) #1

declare void @__kmpc_flush(i64*) #1

declare void @__kmpc_end_critical(i64*, i32, i64*) #1

declare void @__kmpc_critical(i64*, i32, i64*) #1

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
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB074-flush-orig-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "f1", scope: !6, file: !3, line: 18, type: !7, scopeLine: 18, spFlags: DISPFlagDefinition, unit: !2)
!6 = !DIModule(scope: !2, name: "drb074")
!7 = !DISubroutineType(types: !8)
!8 = !{null, !9}
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DILocalVariable(name: "q", arg: 1, scope: !5, file: !3, type: !9)
!11 = !DILocation(line: 0, scope: !5)
!12 = !DILocalVariable(name: "omp_sched_static", scope: !5, file: !3, type: !9)
!13 = !DILocalVariable(name: "omp_proc_bind_false", scope: !5, file: !3, type: !9)
!14 = !DILocalVariable(name: "omp_proc_bind_true", scope: !5, file: !3, type: !9)
!15 = !DILocalVariable(name: "omp_lock_hint_none", scope: !5, file: !3, type: !9)
!16 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !5, file: !3, type: !9)
!17 = !DILocation(line: 24, column: 1, scope: !5)
!18 = !DILocation(line: 21, column: 1, scope: !5)
!19 = !DILocation(line: 23, column: 1, scope: !5)
!20 = distinct !DISubprogram(name: "drb074_flush_orig_yes", scope: !2, file: !3, line: 27, type: !21, scopeLine: 27, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!21 = !DISubroutineType(cc: DW_CC_program, types: !22)
!22 = !{null}
!23 = !DILocalVariable(name: "omp_sched_static", scope: !20, file: !3, type: !9)
!24 = !DILocation(line: 0, scope: !20)
!25 = !DILocalVariable(name: "omp_proc_bind_false", scope: !20, file: !3, type: !9)
!26 = !DILocalVariable(name: "omp_proc_bind_true", scope: !20, file: !3, type: !9)
!27 = !DILocalVariable(name: "omp_lock_hint_none", scope: !20, file: !3, type: !9)
!28 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !20, file: !3, type: !9)
!29 = !DILocation(line: 44, column: 1, scope: !20)
!30 = !DILocation(line: 27, column: 1, scope: !20)
!31 = !DILocalVariable(name: "i", scope: !20, file: !3, type: !9)
!32 = !DILocation(line: 33, column: 1, scope: !20)
!33 = !DILocalVariable(name: "sum", scope: !20, file: !3, type: !9)
!34 = !DILocation(line: 34, column: 1, scope: !20)
!35 = !DILocation(line: 36, column: 1, scope: !20)
!36 = !DILocation(line: 41, column: 1, scope: !20)
!37 = !DILocation(line: 42, column: 1, scope: !20)
!38 = !DILocalVariable(scope: !20, file: !3, type: !9, flags: DIFlagArtificial)
!39 = distinct !DISubprogram(name: "__nv_MAIN__F1L36_1", scope: !2, file: !3, line: 36, type: !40, scopeLine: 36, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!40 = !DISubroutineType(types: !41)
!41 = !{null, !9, !42, !42}
!42 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!43 = !DILocalVariable(name: "__nv_MAIN__F1L36_1Arg0", arg: 1, scope: !39, file: !3, type: !9)
!44 = !DILocation(line: 0, scope: !39)
!45 = !DILocalVariable(name: "__nv_MAIN__F1L36_1Arg1", arg: 2, scope: !39, file: !3, type: !42)
!46 = !DILocalVariable(name: "__nv_MAIN__F1L36_1Arg2", arg: 3, scope: !39, file: !3, type: !42)
!47 = !DILocalVariable(name: "omp_sched_static", scope: !39, file: !3, type: !9)
!48 = !DILocalVariable(name: "omp_proc_bind_false", scope: !39, file: !3, type: !9)
!49 = !DILocalVariable(name: "omp_proc_bind_true", scope: !39, file: !3, type: !9)
!50 = !DILocalVariable(name: "omp_lock_hint_none", scope: !39, file: !3, type: !9)
!51 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !39, file: !3, type: !9)
!52 = !DILocalVariable(name: "sum", scope: !39, file: !3, type: !9)
!53 = !DILocation(line: 39, column: 1, scope: !39)
!54 = !DILocation(line: 36, column: 1, scope: !39)
!55 = !DILocation(line: 37, column: 1, scope: !39)
!56 = !DILocation(line: 38, column: 1, scope: !39)

; ModuleID = '/tmp/DRB069-sectionslock1-orig-no-e8471b.ll'
source_filename = "/tmp/DRB069-sectionslock1-orig-no-e8471b.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.STATICS1 = type <{ [40 x i8] }>
%astruct.dt72 = type <{ i8*, i8* }>

@.STATICS1 = internal global %struct.STATICS1 <{ [40 x i8] c"\FB\FF\FF\FF\03\00\00\00I =\00\EB\FF\FF\FF\00\00\00\00\03\00\00\00\00\00\00\00\01\00\00\00\FF\FF\FF\FF\00\00\00\00" }>, align 16
@.C312_MAIN_ = internal constant i32 25
@.C284_MAIN_ = internal constant i64 0
@.C332_MAIN_ = internal constant i32 6
@.C328_MAIN_ = internal constant [57 x i8] c"micro-benchmarks-fortran/DRB069-sectionslock1-orig-no.f95"
@.C330_MAIN_ = internal constant i32 32
@.C301_MAIN_ = internal constant i32 3
@.C300_MAIN_ = internal constant i32 2
@.C285_MAIN_ = internal constant i32 1
@.C283_MAIN_ = internal constant i32 0
@.C301___nv_MAIN__F1L19_1 = internal constant i32 3
@.C300___nv_MAIN__F1L19_1 = internal constant i32 2
@.C285___nv_MAIN__F1L19_1 = internal constant i32 1
@.C283___nv_MAIN__F1L19_1 = internal constant i32 0

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__363 = alloca i32, align 4
  %i_322 = alloca i32, align 4
  %lock_321 = alloca i32, align 4
  %.uplevelArgPack0001_356 = alloca %astruct.dt72, align 16
  %z__io_334 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 2, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 3, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !14, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 2, metadata !15, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 3, metadata !16, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !17, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !18, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 2, metadata !19, metadata !DIExpression()), !dbg !10
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !20
  store i32 %0, i32* %__gtid_MAIN__363, align 4, !dbg !20
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !21
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !21
  call void (i8*, ...) %2(i8* %1), !dbg !21
  br label %L.LB1_349

L.LB1_349:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %i_322, metadata !22, metadata !DIExpression()), !dbg !10
  store i32 0, i32* %i_322, align 4, !dbg !23
  call void @llvm.dbg.declare(metadata i32* %lock_321, metadata !24, metadata !DIExpression()), !dbg !10
  %3 = bitcast i32* %lock_321 to i64*, !dbg !25
  call void @omp_init_lock_(i64* %3), !dbg !25
  %4 = bitcast i32* %lock_321 to i8*, !dbg !26
  %5 = bitcast %astruct.dt72* %.uplevelArgPack0001_356 to i8**, !dbg !26
  store i8* %4, i8** %5, align 8, !dbg !26
  %6 = bitcast i32* %i_322 to i8*, !dbg !26
  %7 = bitcast %astruct.dt72* %.uplevelArgPack0001_356 to i8*, !dbg !26
  %8 = getelementptr i8, i8* %7, i64 8, !dbg !26
  %9 = bitcast i8* %8 to i8**, !dbg !26
  store i8* %6, i8** %9, align 8, !dbg !26
  br label %L.LB1_361, !dbg !26

L.LB1_361:                                        ; preds = %L.LB1_349
  %10 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L19_1_ to i64*, !dbg !26
  %11 = bitcast %astruct.dt72* %.uplevelArgPack0001_356 to i64*, !dbg !26
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %10, i64* %11), !dbg !26
  %12 = bitcast i32* %lock_321 to i64*, !dbg !27
  call void @omp_destroy_lock_(i64* %12), !dbg !27
  call void (...) @_mp_bcs_nest(), !dbg !28
  %13 = bitcast i32* @.C330_MAIN_ to i8*, !dbg !28
  %14 = bitcast [57 x i8]* @.C328_MAIN_ to i8*, !dbg !28
  %15 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !28
  call void (i8*, i8*, i64, ...) %15(i8* %13, i8* %14, i64 57), !dbg !28
  %16 = bitcast i32* @.C332_MAIN_ to i8*, !dbg !28
  %17 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !28
  %18 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !28
  %19 = bitcast %struct.STATICS1* @.STATICS1 to i8*, !dbg !28
  %20 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !28
  %21 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %20(i8* %16, i8* null, i8* %17, i8* %18, i8* %19, i8* null, i64 0), !dbg !28
  call void @llvm.dbg.declare(metadata i32* %z__io_334, metadata !29, metadata !DIExpression()), !dbg !10
  store i32 %21, i32* %z__io_334, align 4, !dbg !28
  %22 = load i32, i32* %i_322, align 4, !dbg !28
  call void @llvm.dbg.value(metadata i32 %22, metadata !22, metadata !DIExpression()), !dbg !10
  %23 = bitcast i32 (...)* @f90io_sc_i_fmt_write to i32 (i32, i32, ...)*, !dbg !28
  %24 = call i32 (i32, i32, ...) %23(i32 %22, i32 25), !dbg !28
  store i32 %24, i32* %z__io_334, align 4, !dbg !28
  %25 = call i32 (...) @f90io_fmtw_end(), !dbg !28
  store i32 %25, i32* %z__io_334, align 4, !dbg !28
  call void (...) @_mp_ecs_nest(), !dbg !28
  ret void, !dbg !20
}

define internal void @__nv_MAIN__F1L19_1_(i32* %__nv_MAIN__F1L19_1Arg0, i64* %__nv_MAIN__F1L19_1Arg1, i64* %__nv_MAIN__F1L19_1Arg2) #0 !dbg !30 {
L.entry:
  %__gtid___nv_MAIN__F1L19_1__399 = alloca i32, align 4
  %.s0001_394 = alloca i32, align 4
  %.s0000_393 = alloca i32, align 4
  %.s0003_396 = alloca i32, align 4
  %.s0002_395 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L19_1Arg0, metadata !34, metadata !DIExpression()), !dbg !35
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L19_1Arg1, metadata !36, metadata !DIExpression()), !dbg !35
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L19_1Arg2, metadata !37, metadata !DIExpression()), !dbg !35
  call void @llvm.dbg.value(metadata i32 1, metadata !38, metadata !DIExpression()), !dbg !35
  call void @llvm.dbg.value(metadata i32 2, metadata !39, metadata !DIExpression()), !dbg !35
  call void @llvm.dbg.value(metadata i32 3, metadata !40, metadata !DIExpression()), !dbg !35
  call void @llvm.dbg.value(metadata i32 0, metadata !41, metadata !DIExpression()), !dbg !35
  call void @llvm.dbg.value(metadata i32 1, metadata !42, metadata !DIExpression()), !dbg !35
  call void @llvm.dbg.value(metadata i32 2, metadata !43, metadata !DIExpression()), !dbg !35
  call void @llvm.dbg.value(metadata i32 3, metadata !44, metadata !DIExpression()), !dbg !35
  call void @llvm.dbg.value(metadata i32 0, metadata !45, metadata !DIExpression()), !dbg !35
  call void @llvm.dbg.value(metadata i32 1, metadata !46, metadata !DIExpression()), !dbg !35
  call void @llvm.dbg.value(metadata i32 2, metadata !47, metadata !DIExpression()), !dbg !35
  %0 = load i32, i32* %__nv_MAIN__F1L19_1Arg0, align 4, !dbg !48
  store i32 %0, i32* %__gtid___nv_MAIN__F1L19_1__399, align 4, !dbg !48
  br label %L.LB2_392

L.LB2_392:                                        ; preds = %L.entry
  br label %L.LB2_325

L.LB2_325:                                        ; preds = %L.LB2_392
  store i32 2, i32* %.s0001_394, align 4, !dbg !48
  store i32 0, i32* %.s0000_393, align 4, !dbg !49
  store i32 1, i32* %.s0003_396, align 4, !dbg !49
  %1 = load i32, i32* %__gtid___nv_MAIN__F1L19_1__399, align 4, !dbg !49
  %2 = bitcast i32* %.s0002_395 to i64*, !dbg !49
  %3 = bitcast i32* %.s0000_393 to i64*, !dbg !49
  %4 = bitcast i32* %.s0001_394 to i64*, !dbg !49
  %5 = bitcast i32* %.s0003_396 to i64*, !dbg !49
  call void @__kmpc_for_static_init_4(i64* null, i32 %1, i32 34, i64* %2, i64* %3, i64* %4, i64* %5, i32 1, i32 0), !dbg !49
  br label %L.LB2_343

L.LB2_343:                                        ; preds = %L.LB2_325
  %6 = load i32, i32* %.s0000_393, align 4, !dbg !49
  %7 = icmp ne i32 %6, 0, !dbg !49
  br i1 %7, label %L.LB2_344, label %L.LB2_424, !dbg !49

L.LB2_424:                                        ; preds = %L.LB2_343
  br label %L.LB2_344

L.LB2_344:                                        ; preds = %L.LB2_424, %L.LB2_343
  %8 = load i32, i32* %.s0001_394, align 4, !dbg !50
  %9 = icmp ugt i32 1, %8, !dbg !50
  br i1 %9, label %L.LB2_345, label %L.LB2_425, !dbg !50

L.LB2_425:                                        ; preds = %L.LB2_344
  %10 = load i32, i32* %.s0000_393, align 4, !dbg !50
  %11 = icmp ult i32 1, %10, !dbg !50
  br i1 %11, label %L.LB2_345, label %L.LB2_426, !dbg !50

L.LB2_426:                                        ; preds = %L.LB2_425
  %12 = bitcast i64* %__nv_MAIN__F1L19_1Arg2 to i64**, !dbg !51
  %13 = load i64*, i64** %12, align 8, !dbg !51
  call void @omp_set_lock_(i64* %13), !dbg !51
  %14 = bitcast i64* %__nv_MAIN__F1L19_1Arg2 to i8*, !dbg !52
  %15 = getelementptr i8, i8* %14, i64 8, !dbg !52
  %16 = bitcast i8* %15 to i32**, !dbg !52
  %17 = load i32*, i32** %16, align 8, !dbg !52
  %18 = load i32, i32* %17, align 4, !dbg !52
  %19 = add nsw i32 %18, 1, !dbg !52
  %20 = bitcast i64* %__nv_MAIN__F1L19_1Arg2 to i8*, !dbg !52
  %21 = getelementptr i8, i8* %20, i64 8, !dbg !52
  %22 = bitcast i8* %21 to i32**, !dbg !52
  %23 = load i32*, i32** %22, align 8, !dbg !52
  store i32 %19, i32* %23, align 4, !dbg !52
  %24 = bitcast i64* %__nv_MAIN__F1L19_1Arg2 to i64**, !dbg !53
  %25 = load i64*, i64** %24, align 8, !dbg !53
  call void @omp_unset_lock_(i64* %25), !dbg !53
  br label %L.LB2_345

L.LB2_345:                                        ; preds = %L.LB2_426, %L.LB2_425, %L.LB2_344
  %26 = load i32, i32* %.s0001_394, align 4, !dbg !54
  %27 = icmp ugt i32 2, %26, !dbg !54
  br i1 %27, label %L.LB2_346, label %L.LB2_427, !dbg !54

L.LB2_427:                                        ; preds = %L.LB2_345
  %28 = load i32, i32* %.s0000_393, align 4, !dbg !54
  %29 = icmp ult i32 2, %28, !dbg !54
  br i1 %29, label %L.LB2_346, label %L.LB2_428, !dbg !54

L.LB2_428:                                        ; preds = %L.LB2_427
  %30 = bitcast i64* %__nv_MAIN__F1L19_1Arg2 to i64**, !dbg !55
  %31 = load i64*, i64** %30, align 8, !dbg !55
  call void @omp_set_lock_(i64* %31), !dbg !55
  %32 = bitcast i64* %__nv_MAIN__F1L19_1Arg2 to i8*, !dbg !56
  %33 = getelementptr i8, i8* %32, i64 8, !dbg !56
  %34 = bitcast i8* %33 to i32**, !dbg !56
  %35 = load i32*, i32** %34, align 8, !dbg !56
  %36 = load i32, i32* %35, align 4, !dbg !56
  %37 = add nsw i32 %36, 2, !dbg !56
  %38 = bitcast i64* %__nv_MAIN__F1L19_1Arg2 to i8*, !dbg !56
  %39 = getelementptr i8, i8* %38, i64 8, !dbg !56
  %40 = bitcast i8* %39 to i32**, !dbg !56
  %41 = load i32*, i32** %40, align 8, !dbg !56
  store i32 %37, i32* %41, align 4, !dbg !56
  %42 = bitcast i64* %__nv_MAIN__F1L19_1Arg2 to i64**, !dbg !57
  %43 = load i64*, i64** %42, align 8, !dbg !57
  call void @omp_unset_lock_(i64* %43), !dbg !57
  br label %L.LB2_346

L.LB2_346:                                        ; preds = %L.LB2_428, %L.LB2_427, %L.LB2_345
  br label %L.LB2_347

L.LB2_347:                                        ; preds = %L.LB2_346
  br label %L.LB2_326

L.LB2_326:                                        ; preds = %L.LB2_347
  ret void, !dbg !48
}

declare void @__kmpc_for_static_init_4(i64*, i32, i32, i64*, i64*, i64*, i64*, i32 zeroext, i32 zeroext) #0

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

declare void @omp_unset_lock_(i64*) #0

declare void @omp_set_lock_(i64*) #0

declare void @omp_init_lock_(i64*) #0

declare void @omp_destroy_lock_(i64*) #0

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { "no-frame-pointer-elim-non-leaf" }
attributes #1 = { nounwind readnone speculatable }

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !4, globals: !4, imports: !4)
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB069-sectionslock1-orig-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb069_sectionslock1_orig_no", scope: !2, file: !3, line: 10, type: !6, scopeLine: 10, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!6 = !DISubroutineType(cc: DW_CC_program, types: !7)
!7 = !{null}
!8 = !DILocalVariable(name: "omp_sched_static", scope: !5, file: !3, type: !9)
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 0, scope: !5)
!11 = !DILocalVariable(name: "omp_sched_dynamic", scope: !5, file: !3, type: !9)
!12 = !DILocalVariable(name: "omp_sched_guided", scope: !5, file: !3, type: !9)
!13 = !DILocalVariable(name: "omp_proc_bind_false", scope: !5, file: !3, type: !9)
!14 = !DILocalVariable(name: "omp_proc_bind_true", scope: !5, file: !3, type: !9)
!15 = !DILocalVariable(name: "omp_proc_bind_master", scope: !5, file: !3, type: !9)
!16 = !DILocalVariable(name: "omp_proc_bind_close", scope: !5, file: !3, type: !9)
!17 = !DILocalVariable(name: "omp_lock_hint_none", scope: !5, file: !3, type: !9)
!18 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !5, file: !3, type: !9)
!19 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !5, file: !3, type: !9)
!20 = !DILocation(line: 34, column: 1, scope: !5)
!21 = !DILocation(line: 10, column: 1, scope: !5)
!22 = !DILocalVariable(name: "i", scope: !5, file: !3, type: !9)
!23 = !DILocation(line: 16, column: 1, scope: !5)
!24 = !DILocalVariable(name: "lock", scope: !5, file: !3, type: !9)
!25 = !DILocation(line: 17, column: 1, scope: !5)
!26 = !DILocation(line: 19, column: 1, scope: !5)
!27 = !DILocation(line: 30, column: 1, scope: !5)
!28 = !DILocation(line: 32, column: 1, scope: !5)
!29 = !DILocalVariable(scope: !5, file: !3, type: !9, flags: DIFlagArtificial)
!30 = distinct !DISubprogram(name: "__nv_MAIN__F1L19_1", scope: !2, file: !3, line: 19, type: !31, scopeLine: 19, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!31 = !DISubroutineType(types: !32)
!32 = !{null, !9, !33, !33}
!33 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!34 = !DILocalVariable(name: "__nv_MAIN__F1L19_1Arg0", arg: 1, scope: !30, file: !3, type: !9)
!35 = !DILocation(line: 0, scope: !30)
!36 = !DILocalVariable(name: "__nv_MAIN__F1L19_1Arg1", arg: 2, scope: !30, file: !3, type: !33)
!37 = !DILocalVariable(name: "__nv_MAIN__F1L19_1Arg2", arg: 3, scope: !30, file: !3, type: !33)
!38 = !DILocalVariable(name: "omp_sched_static", scope: !30, file: !3, type: !9)
!39 = !DILocalVariable(name: "omp_sched_dynamic", scope: !30, file: !3, type: !9)
!40 = !DILocalVariable(name: "omp_sched_guided", scope: !30, file: !3, type: !9)
!41 = !DILocalVariable(name: "omp_proc_bind_false", scope: !30, file: !3, type: !9)
!42 = !DILocalVariable(name: "omp_proc_bind_true", scope: !30, file: !3, type: !9)
!43 = !DILocalVariable(name: "omp_proc_bind_master", scope: !30, file: !3, type: !9)
!44 = !DILocalVariable(name: "omp_proc_bind_close", scope: !30, file: !3, type: !9)
!45 = !DILocalVariable(name: "omp_lock_hint_none", scope: !30, file: !3, type: !9)
!46 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !30, file: !3, type: !9)
!47 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !30, file: !3, type: !9)
!48 = !DILocation(line: 28, column: 1, scope: !30)
!49 = !DILocation(line: 19, column: 1, scope: !30)
!50 = !DILocation(line: 20, column: 1, scope: !30)
!51 = !DILocation(line: 21, column: 1, scope: !30)
!52 = !DILocation(line: 22, column: 1, scope: !30)
!53 = !DILocation(line: 23, column: 1, scope: !30)
!54 = !DILocation(line: 24, column: 1, scope: !30)
!55 = !DILocation(line: 25, column: 1, scope: !30)
!56 = !DILocation(line: 26, column: 1, scope: !30)
!57 = !DILocation(line: 27, column: 1, scope: !30)

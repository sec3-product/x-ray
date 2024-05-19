; ModuleID = '/tmp/DRB072-taskdep1-orig-no-3ade5e.ll'
source_filename = "/tmp/DRB072-taskdep1-orig-no-3ade5e.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%astruct.dt60 = type <{ i8* }>

@.C309_MAIN_ = internal constant i32 14
@.C333_MAIN_ = internal constant [19 x i8] c"i is not equal to 2"
@.C284_MAIN_ = internal constant i64 0
@.C330_MAIN_ = internal constant i32 6
@.C327_MAIN_ = internal constant [52 x i8] c"micro-benchmarks-fortran/DRB072-taskdep1-orig-no.f95"
@.C329_MAIN_ = internal constant i32 30
@.C300_MAIN_ = internal constant i32 2
@.C285_MAIN_ = internal constant i32 1
@.C283_MAIN_ = internal constant i32 0
@.C300___nv_MAIN__F1L18_1 = internal constant i32 2
@.C283___nv_MAIN__F1L18_1 = internal constant i32 0
@.C285___nv_MAIN__F1L18_1 = internal constant i32 1
@.C285___nv_MAIN_F1L20_2 = internal constant i32 1
@.C300___nv_MAIN_F1L23_3 = internal constant i32 2

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__357 = alloca i32, align 4
  %i_310 = alloca i32, align 4
  %.uplevelArgPack0001_352 = alloca %astruct.dt60, align 8
  %z__io_332 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 2, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 2, metadata !14, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !15, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !16, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 2, metadata !17, metadata !DIExpression()), !dbg !10
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !18
  store i32 %0, i32* %__gtid_MAIN__357, align 4, !dbg !18
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !19
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !19
  call void (i8*, ...) %2(i8* %1), !dbg !19
  br label %L.LB1_346

L.LB1_346:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %i_310, metadata !20, metadata !DIExpression()), !dbg !10
  store i32 0, i32* %i_310, align 4, !dbg !21
  %3 = bitcast i32* %i_310 to i8*, !dbg !22
  %4 = bitcast %astruct.dt60* %.uplevelArgPack0001_352 to i8**, !dbg !22
  store i8* %3, i8** %4, align 8, !dbg !22
  br label %L.LB1_355, !dbg !22

L.LB1_355:                                        ; preds = %L.LB1_346
  %5 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L18_1_ to i64*, !dbg !22
  %6 = bitcast %astruct.dt60* %.uplevelArgPack0001_352 to i64*, !dbg !22
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %5, i64* %6), !dbg !22
  %7 = load i32, i32* %i_310, align 4, !dbg !23
  call void @llvm.dbg.value(metadata i32 %7, metadata !20, metadata !DIExpression()), !dbg !10
  %8 = icmp eq i32 %7, 2, !dbg !23
  br i1 %8, label %L.LB1_344, label %L.LB1_384, !dbg !23

L.LB1_384:                                        ; preds = %L.LB1_355
  call void (...) @_mp_bcs_nest(), !dbg !24
  %9 = bitcast i32* @.C329_MAIN_ to i8*, !dbg !24
  %10 = bitcast [52 x i8]* @.C327_MAIN_ to i8*, !dbg !24
  %11 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !24
  call void (i8*, i8*, i64, ...) %11(i8* %9, i8* %10, i64 52), !dbg !24
  %12 = bitcast i32* @.C330_MAIN_ to i8*, !dbg !24
  %13 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !24
  %14 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !24
  %15 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !24
  %16 = call i32 (i8*, i8*, i8*, i8*, ...) %15(i8* %12, i8* null, i8* %13, i8* %14), !dbg !24
  call void @llvm.dbg.declare(metadata i32* %z__io_332, metadata !25, metadata !DIExpression()), !dbg !10
  store i32 %16, i32* %z__io_332, align 4, !dbg !24
  %17 = bitcast [19 x i8]* @.C333_MAIN_ to i8*, !dbg !24
  %18 = bitcast i32 (...)* @f90io_sc_ch_ldw to i32 (i8*, i32, i64, ...)*, !dbg !24
  %19 = call i32 (i8*, i32, i64, ...) %18(i8* %17, i32 14, i64 19), !dbg !24
  store i32 %19, i32* %z__io_332, align 4, !dbg !24
  %20 = call i32 (...) @f90io_ldw_end(), !dbg !24
  store i32 %20, i32* %z__io_332, align 4, !dbg !24
  call void (...) @_mp_ecs_nest(), !dbg !24
  br label %L.LB1_344

L.LB1_344:                                        ; preds = %L.LB1_384, %L.LB1_355
  ret void, !dbg !18
}

define internal void @__nv_MAIN__F1L18_1_(i32* %__nv_MAIN__F1L18_1Arg0, i64* %__nv_MAIN__F1L18_1Arg1, i64* %__nv_MAIN__F1L18_1Arg2) #0 !dbg !26 {
L.entry:
  %__gtid___nv_MAIN__F1L18_1__394 = alloca i32, align 4
  %.s0000_389 = alloca i32, align 4
  %.s0001_390 = alloca i32, align 4
  %.s0002_400 = alloca i32, align 4
  %.z0348_399 = alloca i8*, align 8
  %.s0003_424 = alloca i32, align 4
  %.z0348_423 = alloca i8*, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L18_1Arg0, metadata !30, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L18_1Arg1, metadata !32, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L18_1Arg2, metadata !33, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.value(metadata i32 1, metadata !34, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.value(metadata i32 2, metadata !35, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.value(metadata i32 0, metadata !36, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.value(metadata i32 1, metadata !37, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.value(metadata i32 2, metadata !38, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.value(metadata i32 0, metadata !39, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.value(metadata i32 1, metadata !40, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.value(metadata i32 2, metadata !41, metadata !DIExpression()), !dbg !31
  %0 = load i32, i32* %__nv_MAIN__F1L18_1Arg0, align 4, !dbg !42
  store i32 %0, i32* %__gtid___nv_MAIN__F1L18_1__394, align 4, !dbg !42
  br label %L.LB2_388

L.LB2_388:                                        ; preds = %L.entry
  br label %L.LB2_313

L.LB2_313:                                        ; preds = %L.LB2_388
  store i32 -1, i32* %.s0000_389, align 4, !dbg !43
  store i32 0, i32* %.s0001_390, align 4, !dbg !43
  %1 = load i32, i32* %__gtid___nv_MAIN__F1L18_1__394, align 4, !dbg !43
  %2 = call i32 @__kmpc_single(i64* null, i32 %1), !dbg !43
  %3 = icmp eq i32 %2, 0, !dbg !43
  br i1 %3, label %L.LB2_341, label %L.LB2_315, !dbg !43

L.LB2_315:                                        ; preds = %L.LB2_313
  store i32 1, i32* %.s0002_400, align 4, !dbg !44
  %4 = load i32, i32* %__gtid___nv_MAIN__F1L18_1__394, align 4, !dbg !45
  %5 = load i32, i32* %.s0002_400, align 4, !dbg !45
  %6 = bitcast void (i32, i64*)* @__nv_MAIN_F1L20_2_ to i64*, !dbg !45
  %7 = call i8* @__kmpc_omp_task_alloc(i64* null, i32 %4, i32 %5, i32 40, i32 8, i64* %6), !dbg !45
  store i8* %7, i8** %.z0348_399, align 8, !dbg !45
  %8 = load i64, i64* %__nv_MAIN__F1L18_1Arg2, align 8, !dbg !45
  %9 = load i8*, i8** %.z0348_399, align 8, !dbg !45
  %10 = bitcast i8* %9 to i64**, !dbg !45
  %11 = load i64*, i64** %10, align 8, !dbg !45
  store i64 %8, i64* %11, align 8, !dbg !45
  %12 = load i32, i32* %__gtid___nv_MAIN__F1L18_1__394, align 4, !dbg !45
  %13 = load i8*, i8** %.z0348_399, align 8, !dbg !45
  %14 = bitcast i8* %13 to i64*, !dbg !45
  call void @__kmpc_omp_task(i64* null, i32 %12, i64* %14), !dbg !45
  br label %L.LB2_342

L.LB2_342:                                        ; preds = %L.LB2_315
  store i32 1, i32* %.s0003_424, align 4, !dbg !46
  %15 = load i32, i32* %__gtid___nv_MAIN__F1L18_1__394, align 4, !dbg !47
  %16 = load i32, i32* %.s0003_424, align 4, !dbg !47
  %17 = bitcast void (i32, i64*)* @__nv_MAIN_F1L23_3_ to i64*, !dbg !47
  %18 = call i8* @__kmpc_omp_task_alloc(i64* null, i32 %15, i32 %16, i32 40, i32 8, i64* %17), !dbg !47
  store i8* %18, i8** %.z0348_423, align 8, !dbg !47
  %19 = load i64, i64* %__nv_MAIN__F1L18_1Arg2, align 8, !dbg !47
  %20 = load i8*, i8** %.z0348_423, align 8, !dbg !47
  %21 = bitcast i8* %20 to i64**, !dbg !47
  %22 = load i64*, i64** %21, align 8, !dbg !47
  store i64 %19, i64* %22, align 8, !dbg !47
  %23 = load i32, i32* %__gtid___nv_MAIN__F1L18_1__394, align 4, !dbg !47
  %24 = load i8*, i8** %.z0348_423, align 8, !dbg !47
  %25 = bitcast i8* %24 to i64*, !dbg !47
  call void @__kmpc_omp_task(i64* null, i32 %23, i64* %25), !dbg !47
  br label %L.LB2_343

L.LB2_343:                                        ; preds = %L.LB2_342
  %26 = load i32, i32* %__gtid___nv_MAIN__F1L18_1__394, align 4, !dbg !48
  store i32 %26, i32* %.s0000_389, align 4, !dbg !48
  store i32 1, i32* %.s0001_390, align 4, !dbg !48
  %27 = load i32, i32* %__gtid___nv_MAIN__F1L18_1__394, align 4, !dbg !48
  call void @__kmpc_end_single(i64* null, i32 %27), !dbg !48
  br label %L.LB2_341

L.LB2_341:                                        ; preds = %L.LB2_343, %L.LB2_313
  br label %L.LB2_324

L.LB2_324:                                        ; preds = %L.LB2_341
  %28 = load i32, i32* %__gtid___nv_MAIN__F1L18_1__394, align 4, !dbg !48
  call void @__kmpc_barrier(i64* null, i32 %28), !dbg !48
  br label %L.LB2_325

L.LB2_325:                                        ; preds = %L.LB2_324
  ret void, !dbg !42
}

define internal void @__nv_MAIN_F1L20_2_(i32 %__nv_MAIN_F1L20_2Arg0.arg, i64* %__nv_MAIN_F1L20_2Arg1) #0 !dbg !49 {
L.entry:
  %__nv_MAIN_F1L20_2Arg0.addr = alloca i32, align 4
  %.S0000_446 = alloca i8*, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L20_2Arg0.addr, metadata !52, metadata !DIExpression()), !dbg !53
  store i32 %__nv_MAIN_F1L20_2Arg0.arg, i32* %__nv_MAIN_F1L20_2Arg0.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L20_2Arg0.addr, metadata !54, metadata !DIExpression()), !dbg !53
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L20_2Arg1, metadata !55, metadata !DIExpression()), !dbg !53
  call void @llvm.dbg.value(metadata i32 1, metadata !56, metadata !DIExpression()), !dbg !53
  call void @llvm.dbg.value(metadata i32 2, metadata !57, metadata !DIExpression()), !dbg !53
  call void @llvm.dbg.value(metadata i32 0, metadata !58, metadata !DIExpression()), !dbg !53
  call void @llvm.dbg.value(metadata i32 1, metadata !59, metadata !DIExpression()), !dbg !53
  call void @llvm.dbg.value(metadata i32 2, metadata !60, metadata !DIExpression()), !dbg !53
  call void @llvm.dbg.value(metadata i32 0, metadata !61, metadata !DIExpression()), !dbg !53
  call void @llvm.dbg.value(metadata i32 1, metadata !62, metadata !DIExpression()), !dbg !53
  call void @llvm.dbg.value(metadata i32 2, metadata !63, metadata !DIExpression()), !dbg !53
  %0 = bitcast i64* %__nv_MAIN_F1L20_2Arg1 to i8**, !dbg !64
  %1 = load i8*, i8** %0, align 8, !dbg !64
  store i8* %1, i8** %.S0000_446, align 8, !dbg !64
  br label %L.LB4_450

L.LB4_450:                                        ; preds = %L.entry
  br label %L.LB4_318

L.LB4_318:                                        ; preds = %L.LB4_450
  %2 = load i8*, i8** %.S0000_446, align 8, !dbg !65
  %3 = bitcast i8* %2 to i32**, !dbg !65
  %4 = load i32*, i32** %3, align 8, !dbg !65
  store i32 1, i32* %4, align 4, !dbg !65
  br label %L.LB4_319

L.LB4_319:                                        ; preds = %L.LB4_318
  ret void, !dbg !66
}

define internal void @__nv_MAIN_F1L23_3_(i32 %__nv_MAIN_F1L23_3Arg0.arg, i64* %__nv_MAIN_F1L23_3Arg1) #0 !dbg !67 {
L.entry:
  %__nv_MAIN_F1L23_3Arg0.addr = alloca i32, align 4
  %.S0000_446 = alloca i8*, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L23_3Arg0.addr, metadata !68, metadata !DIExpression()), !dbg !69
  store i32 %__nv_MAIN_F1L23_3Arg0.arg, i32* %__nv_MAIN_F1L23_3Arg0.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L23_3Arg0.addr, metadata !70, metadata !DIExpression()), !dbg !69
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L23_3Arg1, metadata !71, metadata !DIExpression()), !dbg !69
  call void @llvm.dbg.value(metadata i32 1, metadata !72, metadata !DIExpression()), !dbg !69
  call void @llvm.dbg.value(metadata i32 2, metadata !73, metadata !DIExpression()), !dbg !69
  call void @llvm.dbg.value(metadata i32 0, metadata !74, metadata !DIExpression()), !dbg !69
  call void @llvm.dbg.value(metadata i32 1, metadata !75, metadata !DIExpression()), !dbg !69
  call void @llvm.dbg.value(metadata i32 2, metadata !76, metadata !DIExpression()), !dbg !69
  call void @llvm.dbg.value(metadata i32 0, metadata !77, metadata !DIExpression()), !dbg !69
  call void @llvm.dbg.value(metadata i32 1, metadata !78, metadata !DIExpression()), !dbg !69
  call void @llvm.dbg.value(metadata i32 2, metadata !79, metadata !DIExpression()), !dbg !69
  %0 = bitcast i64* %__nv_MAIN_F1L23_3Arg1 to i8**, !dbg !80
  %1 = load i8*, i8** %0, align 8, !dbg !80
  store i8* %1, i8** %.S0000_446, align 8, !dbg !80
  br label %L.LB5_456

L.LB5_456:                                        ; preds = %L.entry
  br label %L.LB5_322

L.LB5_322:                                        ; preds = %L.LB5_456
  %2 = load i8*, i8** %.S0000_446, align 8, !dbg !81
  %3 = bitcast i8* %2 to i32**, !dbg !81
  %4 = load i32*, i32** %3, align 8, !dbg !81
  store i32 2, i32* %4, align 4, !dbg !81
  br label %L.LB5_323

L.LB5_323:                                        ; preds = %L.LB5_322
  ret void, !dbg !82
}

declare void @__kmpc_barrier(i64*, i32) #0

declare void @__kmpc_end_single(i64*, i32) #0

declare void @__kmpc_omp_task(i64*, i32, i64*) #0

declare i8* @__kmpc_omp_task_alloc(i64*, i32, i32, i32, i32, i64*) #0

declare signext i32 @__kmpc_single(i64*, i32) #0

declare void @_mp_ecs_nest(...) #0

declare signext i32 @f90io_ldw_end(...) #0

declare signext i32 @f90io_sc_ch_ldw(...) #0

declare signext i32 @f90io_print_init(...) #0

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
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB072-taskdep1-orig-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb072_taskdep1_orig_no", scope: !2, file: !3, line: 11, type: !6, scopeLine: 11, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
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
!18 = !DILocation(line: 32, column: 1, scope: !5)
!19 = !DILocation(line: 11, column: 1, scope: !5)
!20 = !DILocalVariable(name: "i", scope: !5, file: !3, type: !9)
!21 = !DILocation(line: 16, column: 1, scope: !5)
!22 = !DILocation(line: 18, column: 1, scope: !5)
!23 = !DILocation(line: 29, column: 1, scope: !5)
!24 = !DILocation(line: 30, column: 1, scope: !5)
!25 = !DILocalVariable(scope: !5, file: !3, type: !9, flags: DIFlagArtificial)
!26 = distinct !DISubprogram(name: "__nv_MAIN__F1L18_1", scope: !2, file: !3, line: 18, type: !27, scopeLine: 18, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!27 = !DISubroutineType(types: !28)
!28 = !{null, !9, !29, !29}
!29 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!30 = !DILocalVariable(name: "__nv_MAIN__F1L18_1Arg0", arg: 1, scope: !26, file: !3, type: !9)
!31 = !DILocation(line: 0, scope: !26)
!32 = !DILocalVariable(name: "__nv_MAIN__F1L18_1Arg1", arg: 2, scope: !26, file: !3, type: !29)
!33 = !DILocalVariable(name: "__nv_MAIN__F1L18_1Arg2", arg: 3, scope: !26, file: !3, type: !29)
!34 = !DILocalVariable(name: "omp_sched_static", scope: !26, file: !3, type: !9)
!35 = !DILocalVariable(name: "omp_sched_dynamic", scope: !26, file: !3, type: !9)
!36 = !DILocalVariable(name: "omp_proc_bind_false", scope: !26, file: !3, type: !9)
!37 = !DILocalVariable(name: "omp_proc_bind_true", scope: !26, file: !3, type: !9)
!38 = !DILocalVariable(name: "omp_proc_bind_master", scope: !26, file: !3, type: !9)
!39 = !DILocalVariable(name: "omp_lock_hint_none", scope: !26, file: !3, type: !9)
!40 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !26, file: !3, type: !9)
!41 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !26, file: !3, type: !9)
!42 = !DILocation(line: 27, column: 1, scope: !26)
!43 = !DILocation(line: 19, column: 1, scope: !26)
!44 = !DILocation(line: 20, column: 1, scope: !26)
!45 = !DILocation(line: 22, column: 1, scope: !26)
!46 = !DILocation(line: 23, column: 1, scope: !26)
!47 = !DILocation(line: 25, column: 1, scope: !26)
!48 = !DILocation(line: 26, column: 1, scope: !26)
!49 = distinct !DISubprogram(name: "__nv_MAIN_F1L20_2", scope: !2, file: !3, line: 20, type: !50, scopeLine: 20, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!50 = !DISubroutineType(types: !51)
!51 = !{null, !9, !29}
!52 = !DILocalVariable(name: "__nv_MAIN_F1L20_2Arg0", scope: !49, file: !3, type: !9)
!53 = !DILocation(line: 0, scope: !49)
!54 = !DILocalVariable(name: "__nv_MAIN_F1L20_2Arg0", arg: 1, scope: !49, file: !3, type: !9)
!55 = !DILocalVariable(name: "__nv_MAIN_F1L20_2Arg1", arg: 2, scope: !49, file: !3, type: !29)
!56 = !DILocalVariable(name: "omp_sched_static", scope: !49, file: !3, type: !9)
!57 = !DILocalVariable(name: "omp_sched_dynamic", scope: !49, file: !3, type: !9)
!58 = !DILocalVariable(name: "omp_proc_bind_false", scope: !49, file: !3, type: !9)
!59 = !DILocalVariable(name: "omp_proc_bind_true", scope: !49, file: !3, type: !9)
!60 = !DILocalVariable(name: "omp_proc_bind_master", scope: !49, file: !3, type: !9)
!61 = !DILocalVariable(name: "omp_lock_hint_none", scope: !49, file: !3, type: !9)
!62 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !49, file: !3, type: !9)
!63 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !49, file: !3, type: !9)
!64 = !DILocation(line: 20, column: 1, scope: !49)
!65 = !DILocation(line: 21, column: 1, scope: !49)
!66 = !DILocation(line: 22, column: 1, scope: !49)
!67 = distinct !DISubprogram(name: "__nv_MAIN_F1L23_3", scope: !2, file: !3, line: 23, type: !50, scopeLine: 23, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!68 = !DILocalVariable(name: "__nv_MAIN_F1L23_3Arg0", scope: !67, file: !3, type: !9)
!69 = !DILocation(line: 0, scope: !67)
!70 = !DILocalVariable(name: "__nv_MAIN_F1L23_3Arg0", arg: 1, scope: !67, file: !3, type: !9)
!71 = !DILocalVariable(name: "__nv_MAIN_F1L23_3Arg1", arg: 2, scope: !67, file: !3, type: !29)
!72 = !DILocalVariable(name: "omp_sched_static", scope: !67, file: !3, type: !9)
!73 = !DILocalVariable(name: "omp_sched_dynamic", scope: !67, file: !3, type: !9)
!74 = !DILocalVariable(name: "omp_proc_bind_false", scope: !67, file: !3, type: !9)
!75 = !DILocalVariable(name: "omp_proc_bind_true", scope: !67, file: !3, type: !9)
!76 = !DILocalVariable(name: "omp_proc_bind_master", scope: !67, file: !3, type: !9)
!77 = !DILocalVariable(name: "omp_lock_hint_none", scope: !67, file: !3, type: !9)
!78 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !67, file: !3, type: !9)
!79 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !67, file: !3, type: !9)
!80 = !DILocation(line: 23, column: 1, scope: !67)
!81 = !DILocation(line: 24, column: 1, scope: !67)
!82 = !DILocation(line: 25, column: 1, scope: !67)

; ModuleID = '/tmp/DRB027-taskdependmissing-orig-yes-bcc9e0.ll'
source_filename = "/tmp/DRB027-taskdependmissing-orig-yes-bcc9e0.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.STATICS1 = type <{ [40 x i8] }>
%astruct.dt60 = type <{ i8* }>

@.STATICS1 = internal global %struct.STATICS1 <{ [40 x i8] c"\FB\FF\FF\FF\02\00\00\00i=\00\00\EB\FF\FF\FF\00\00\00\00\03\00\00\00\00\00\00\00\01\00\00\00\FF\FF\FF\FF\00\00\00\00" }>, align 16
@.C309_MAIN_ = internal constant i32 25
@.C284_MAIN_ = internal constant i64 0
@.C331_MAIN_ = internal constant i32 6
@.C327_MAIN_ = internal constant [62 x i8] c"micro-benchmarks-fortran/DRB027-taskdependmissing-orig-yes.f95"
@.C329_MAIN_ = internal constant i32 30
@.C300_MAIN_ = internal constant i32 2
@.C285_MAIN_ = internal constant i32 1
@.C283_MAIN_ = internal constant i32 0
@.C300___nv_MAIN__F1L19_1 = internal constant i32 2
@.C283___nv_MAIN__F1L19_1 = internal constant i32 0
@.C285___nv_MAIN__F1L19_1 = internal constant i32 1
@.C285___nv_MAIN_F1L21_2 = internal constant i32 1
@.C300___nv_MAIN_F1L24_3 = internal constant i32 2

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__357 = alloca i32, align 4
  %i_310 = alloca i32, align 4
  %.uplevelArgPack0001_352 = alloca %astruct.dt60, align 8
  %z__io_333 = alloca i32, align 4
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
  %5 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L19_1_ to i64*, !dbg !22
  %6 = bitcast %astruct.dt60* %.uplevelArgPack0001_352 to i64*, !dbg !22
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %5, i64* %6), !dbg !22
  call void (...) @_mp_bcs_nest(), !dbg !23
  %7 = bitcast i32* @.C329_MAIN_ to i8*, !dbg !23
  %8 = bitcast [62 x i8]* @.C327_MAIN_ to i8*, !dbg !23
  %9 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !23
  call void (i8*, i8*, i64, ...) %9(i8* %7, i8* %8, i64 62), !dbg !23
  %10 = bitcast i32* @.C331_MAIN_ to i8*, !dbg !23
  %11 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !23
  %12 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !23
  %13 = bitcast %struct.STATICS1* @.STATICS1 to i8*, !dbg !23
  %14 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !23
  %15 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %14(i8* %10, i8* null, i8* %11, i8* %12, i8* %13, i8* null, i64 0), !dbg !23
  call void @llvm.dbg.declare(metadata i32* %z__io_333, metadata !24, metadata !DIExpression()), !dbg !10
  store i32 %15, i32* %z__io_333, align 4, !dbg !23
  %16 = load i32, i32* %i_310, align 4, !dbg !23
  call void @llvm.dbg.value(metadata i32 %16, metadata !20, metadata !DIExpression()), !dbg !10
  %17 = bitcast i32 (...)* @f90io_sc_i_fmt_write to i32 (i32, i32, ...)*, !dbg !23
  %18 = call i32 (i32, i32, ...) %17(i32 %16, i32 25), !dbg !23
  store i32 %18, i32* %z__io_333, align 4, !dbg !23
  %19 = call i32 (...) @f90io_fmtw_end(), !dbg !23
  store i32 %19, i32* %z__io_333, align 4, !dbg !23
  call void (...) @_mp_ecs_nest(), !dbg !23
  ret void, !dbg !18
}

define internal void @__nv_MAIN__F1L19_1_(i32* %__nv_MAIN__F1L19_1Arg0, i64* %__nv_MAIN__F1L19_1Arg1, i64* %__nv_MAIN__F1L19_1Arg2) #0 !dbg !25 {
L.entry:
  %__gtid___nv_MAIN__F1L19_1__392 = alloca i32, align 4
  %.s0000_387 = alloca i32, align 4
  %.s0001_388 = alloca i32, align 4
  %.s0002_398 = alloca i32, align 4
  %.z0348_397 = alloca i8*, align 8
  %.s0003_422 = alloca i32, align 4
  %.z0348_421 = alloca i8*, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L19_1Arg0, metadata !29, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L19_1Arg1, metadata !31, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L19_1Arg2, metadata !32, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.value(metadata i32 1, metadata !33, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.value(metadata i32 2, metadata !34, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.value(metadata i32 0, metadata !35, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.value(metadata i32 1, metadata !36, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.value(metadata i32 2, metadata !37, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.value(metadata i32 0, metadata !38, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.value(metadata i32 1, metadata !39, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.value(metadata i32 2, metadata !40, metadata !DIExpression()), !dbg !30
  %0 = load i32, i32* %__nv_MAIN__F1L19_1Arg0, align 4, !dbg !41
  store i32 %0, i32* %__gtid___nv_MAIN__F1L19_1__392, align 4, !dbg !41
  br label %L.LB2_386

L.LB2_386:                                        ; preds = %L.entry
  br label %L.LB2_313

L.LB2_313:                                        ; preds = %L.LB2_386
  store i32 -1, i32* %.s0000_387, align 4, !dbg !42
  store i32 0, i32* %.s0001_388, align 4, !dbg !42
  %1 = load i32, i32* %__gtid___nv_MAIN__F1L19_1__392, align 4, !dbg !42
  %2 = call i32 @__kmpc_single(i64* null, i32 %1), !dbg !42
  %3 = icmp eq i32 %2, 0, !dbg !42
  br i1 %3, label %L.LB2_342, label %L.LB2_315, !dbg !42

L.LB2_315:                                        ; preds = %L.LB2_313
  store i32 1, i32* %.s0002_398, align 4, !dbg !43
  %4 = load i32, i32* %__gtid___nv_MAIN__F1L19_1__392, align 4, !dbg !44
  %5 = load i32, i32* %.s0002_398, align 4, !dbg !44
  %6 = bitcast void (i32, i64*)* @__nv_MAIN_F1L21_2_ to i64*, !dbg !44
  %7 = call i8* @__kmpc_omp_task_alloc(i64* null, i32 %4, i32 %5, i32 40, i32 8, i64* %6), !dbg !44
  store i8* %7, i8** %.z0348_397, align 8, !dbg !44
  %8 = load i64, i64* %__nv_MAIN__F1L19_1Arg2, align 8, !dbg !44
  %9 = load i8*, i8** %.z0348_397, align 8, !dbg !44
  %10 = bitcast i8* %9 to i64**, !dbg !44
  %11 = load i64*, i64** %10, align 8, !dbg !44
  store i64 %8, i64* %11, align 8, !dbg !44
  %12 = load i32, i32* %__gtid___nv_MAIN__F1L19_1__392, align 4, !dbg !44
  %13 = load i8*, i8** %.z0348_397, align 8, !dbg !44
  %14 = bitcast i8* %13 to i64*, !dbg !44
  call void @__kmpc_omp_task(i64* null, i32 %12, i64* %14), !dbg !44
  br label %L.LB2_343

L.LB2_343:                                        ; preds = %L.LB2_315
  store i32 1, i32* %.s0003_422, align 4, !dbg !45
  %15 = load i32, i32* %__gtid___nv_MAIN__F1L19_1__392, align 4, !dbg !46
  %16 = load i32, i32* %.s0003_422, align 4, !dbg !46
  %17 = bitcast void (i32, i64*)* @__nv_MAIN_F1L24_3_ to i64*, !dbg !46
  %18 = call i8* @__kmpc_omp_task_alloc(i64* null, i32 %15, i32 %16, i32 40, i32 8, i64* %17), !dbg !46
  store i8* %18, i8** %.z0348_421, align 8, !dbg !46
  %19 = load i64, i64* %__nv_MAIN__F1L19_1Arg2, align 8, !dbg !46
  %20 = load i8*, i8** %.z0348_421, align 8, !dbg !46
  %21 = bitcast i8* %20 to i64**, !dbg !46
  %22 = load i64*, i64** %21, align 8, !dbg !46
  store i64 %19, i64* %22, align 8, !dbg !46
  %23 = load i32, i32* %__gtid___nv_MAIN__F1L19_1__392, align 4, !dbg !46
  %24 = load i8*, i8** %.z0348_421, align 8, !dbg !46
  %25 = bitcast i8* %24 to i64*, !dbg !46
  call void @__kmpc_omp_task(i64* null, i32 %23, i64* %25), !dbg !46
  br label %L.LB2_344

L.LB2_344:                                        ; preds = %L.LB2_343
  %26 = load i32, i32* %__gtid___nv_MAIN__F1L19_1__392, align 4, !dbg !47
  store i32 %26, i32* %.s0000_387, align 4, !dbg !47
  store i32 1, i32* %.s0001_388, align 4, !dbg !47
  %27 = load i32, i32* %__gtid___nv_MAIN__F1L19_1__392, align 4, !dbg !47
  call void @__kmpc_end_single(i64* null, i32 %27), !dbg !47
  br label %L.LB2_342

L.LB2_342:                                        ; preds = %L.LB2_344, %L.LB2_313
  br label %L.LB2_324

L.LB2_324:                                        ; preds = %L.LB2_342
  %28 = load i32, i32* %__gtid___nv_MAIN__F1L19_1__392, align 4, !dbg !47
  call void @__kmpc_barrier(i64* null, i32 %28), !dbg !47
  br label %L.LB2_325

L.LB2_325:                                        ; preds = %L.LB2_324
  ret void, !dbg !41
}

define internal void @__nv_MAIN_F1L21_2_(i32 %__nv_MAIN_F1L21_2Arg0.arg, i64* %__nv_MAIN_F1L21_2Arg1) #0 !dbg !48 {
L.entry:
  %__nv_MAIN_F1L21_2Arg0.addr = alloca i32, align 4
  %.S0000_443 = alloca i8*, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L21_2Arg0.addr, metadata !51, metadata !DIExpression()), !dbg !52
  store i32 %__nv_MAIN_F1L21_2Arg0.arg, i32* %__nv_MAIN_F1L21_2Arg0.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L21_2Arg0.addr, metadata !53, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L21_2Arg1, metadata !54, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.value(metadata i32 1, metadata !55, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.value(metadata i32 2, metadata !56, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.value(metadata i32 0, metadata !57, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.value(metadata i32 1, metadata !58, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.value(metadata i32 2, metadata !59, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.value(metadata i32 0, metadata !60, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.value(metadata i32 1, metadata !61, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.value(metadata i32 2, metadata !62, metadata !DIExpression()), !dbg !52
  %0 = bitcast i64* %__nv_MAIN_F1L21_2Arg1 to i8**, !dbg !63
  %1 = load i8*, i8** %0, align 8, !dbg !63
  store i8* %1, i8** %.S0000_443, align 8, !dbg !63
  br label %L.LB4_447

L.LB4_447:                                        ; preds = %L.entry
  br label %L.LB4_318

L.LB4_318:                                        ; preds = %L.LB4_447
  %2 = load i8*, i8** %.S0000_443, align 8, !dbg !64
  %3 = bitcast i8* %2 to i32**, !dbg !64
  %4 = load i32*, i32** %3, align 8, !dbg !64
  store i32 1, i32* %4, align 4, !dbg !64
  br label %L.LB4_319

L.LB4_319:                                        ; preds = %L.LB4_318
  ret void, !dbg !65
}

define internal void @__nv_MAIN_F1L24_3_(i32 %__nv_MAIN_F1L24_3Arg0.arg, i64* %__nv_MAIN_F1L24_3Arg1) #0 !dbg !66 {
L.entry:
  %__nv_MAIN_F1L24_3Arg0.addr = alloca i32, align 4
  %.S0000_443 = alloca i8*, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L24_3Arg0.addr, metadata !67, metadata !DIExpression()), !dbg !68
  store i32 %__nv_MAIN_F1L24_3Arg0.arg, i32* %__nv_MAIN_F1L24_3Arg0.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L24_3Arg0.addr, metadata !69, metadata !DIExpression()), !dbg !68
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L24_3Arg1, metadata !70, metadata !DIExpression()), !dbg !68
  call void @llvm.dbg.value(metadata i32 1, metadata !71, metadata !DIExpression()), !dbg !68
  call void @llvm.dbg.value(metadata i32 2, metadata !72, metadata !DIExpression()), !dbg !68
  call void @llvm.dbg.value(metadata i32 0, metadata !73, metadata !DIExpression()), !dbg !68
  call void @llvm.dbg.value(metadata i32 1, metadata !74, metadata !DIExpression()), !dbg !68
  call void @llvm.dbg.value(metadata i32 2, metadata !75, metadata !DIExpression()), !dbg !68
  call void @llvm.dbg.value(metadata i32 0, metadata !76, metadata !DIExpression()), !dbg !68
  call void @llvm.dbg.value(metadata i32 1, metadata !77, metadata !DIExpression()), !dbg !68
  call void @llvm.dbg.value(metadata i32 2, metadata !78, metadata !DIExpression()), !dbg !68
  %0 = bitcast i64* %__nv_MAIN_F1L24_3Arg1 to i8**, !dbg !79
  %1 = load i8*, i8** %0, align 8, !dbg !79
  store i8* %1, i8** %.S0000_443, align 8, !dbg !79
  br label %L.LB5_453

L.LB5_453:                                        ; preds = %L.entry
  br label %L.LB5_322

L.LB5_322:                                        ; preds = %L.LB5_453
  %2 = load i8*, i8** %.S0000_443, align 8, !dbg !80
  %3 = bitcast i8* %2 to i32**, !dbg !80
  %4 = load i32*, i32** %3, align 8, !dbg !80
  store i32 2, i32* %4, align 4, !dbg !80
  br label %L.LB5_323

L.LB5_323:                                        ; preds = %L.LB5_322
  ret void, !dbg !81
}

declare void @__kmpc_barrier(i64*, i32) #0

declare void @__kmpc_end_single(i64*, i32) #0

declare void @__kmpc_omp_task(i64*, i32, i64*) #0

declare i8* @__kmpc_omp_task_alloc(i64*, i32, i32, i32, i32, i64*) #0

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
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB027-taskdependmissing-orig-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb027_taskdependmissing_orig_yes", scope: !2, file: !3, line: 12, type: !6, scopeLine: 12, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
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
!18 = !DILocation(line: 33, column: 1, scope: !5)
!19 = !DILocation(line: 12, column: 1, scope: !5)
!20 = !DILocalVariable(name: "i", scope: !5, file: !3, type: !9)
!21 = !DILocation(line: 17, column: 1, scope: !5)
!22 = !DILocation(line: 19, column: 1, scope: !5)
!23 = !DILocation(line: 30, column: 1, scope: !5)
!24 = !DILocalVariable(scope: !5, file: !3, type: !9, flags: DIFlagArtificial)
!25 = distinct !DISubprogram(name: "__nv_MAIN__F1L19_1", scope: !2, file: !3, line: 19, type: !26, scopeLine: 19, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!26 = !DISubroutineType(types: !27)
!27 = !{null, !9, !28, !28}
!28 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!29 = !DILocalVariable(name: "__nv_MAIN__F1L19_1Arg0", arg: 1, scope: !25, file: !3, type: !9)
!30 = !DILocation(line: 0, scope: !25)
!31 = !DILocalVariable(name: "__nv_MAIN__F1L19_1Arg1", arg: 2, scope: !25, file: !3, type: !28)
!32 = !DILocalVariable(name: "__nv_MAIN__F1L19_1Arg2", arg: 3, scope: !25, file: !3, type: !28)
!33 = !DILocalVariable(name: "omp_sched_static", scope: !25, file: !3, type: !9)
!34 = !DILocalVariable(name: "omp_sched_dynamic", scope: !25, file: !3, type: !9)
!35 = !DILocalVariable(name: "omp_proc_bind_false", scope: !25, file: !3, type: !9)
!36 = !DILocalVariable(name: "omp_proc_bind_true", scope: !25, file: !3, type: !9)
!37 = !DILocalVariable(name: "omp_proc_bind_master", scope: !25, file: !3, type: !9)
!38 = !DILocalVariable(name: "omp_lock_hint_none", scope: !25, file: !3, type: !9)
!39 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !25, file: !3, type: !9)
!40 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !25, file: !3, type: !9)
!41 = !DILocation(line: 28, column: 1, scope: !25)
!42 = !DILocation(line: 20, column: 1, scope: !25)
!43 = !DILocation(line: 21, column: 1, scope: !25)
!44 = !DILocation(line: 23, column: 1, scope: !25)
!45 = !DILocation(line: 24, column: 1, scope: !25)
!46 = !DILocation(line: 26, column: 1, scope: !25)
!47 = !DILocation(line: 27, column: 1, scope: !25)
!48 = distinct !DISubprogram(name: "__nv_MAIN_F1L21_2", scope: !2, file: !3, line: 21, type: !49, scopeLine: 21, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!49 = !DISubroutineType(types: !50)
!50 = !{null, !9, !28}
!51 = !DILocalVariable(name: "__nv_MAIN_F1L21_2Arg0", scope: !48, file: !3, type: !9)
!52 = !DILocation(line: 0, scope: !48)
!53 = !DILocalVariable(name: "__nv_MAIN_F1L21_2Arg0", arg: 1, scope: !48, file: !3, type: !9)
!54 = !DILocalVariable(name: "__nv_MAIN_F1L21_2Arg1", arg: 2, scope: !48, file: !3, type: !28)
!55 = !DILocalVariable(name: "omp_sched_static", scope: !48, file: !3, type: !9)
!56 = !DILocalVariable(name: "omp_sched_dynamic", scope: !48, file: !3, type: !9)
!57 = !DILocalVariable(name: "omp_proc_bind_false", scope: !48, file: !3, type: !9)
!58 = !DILocalVariable(name: "omp_proc_bind_true", scope: !48, file: !3, type: !9)
!59 = !DILocalVariable(name: "omp_proc_bind_master", scope: !48, file: !3, type: !9)
!60 = !DILocalVariable(name: "omp_lock_hint_none", scope: !48, file: !3, type: !9)
!61 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !48, file: !3, type: !9)
!62 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !48, file: !3, type: !9)
!63 = !DILocation(line: 21, column: 1, scope: !48)
!64 = !DILocation(line: 22, column: 1, scope: !48)
!65 = !DILocation(line: 23, column: 1, scope: !48)
!66 = distinct !DISubprogram(name: "__nv_MAIN_F1L24_3", scope: !2, file: !3, line: 24, type: !49, scopeLine: 24, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!67 = !DILocalVariable(name: "__nv_MAIN_F1L24_3Arg0", scope: !66, file: !3, type: !9)
!68 = !DILocation(line: 0, scope: !66)
!69 = !DILocalVariable(name: "__nv_MAIN_F1L24_3Arg0", arg: 1, scope: !66, file: !3, type: !9)
!70 = !DILocalVariable(name: "__nv_MAIN_F1L24_3Arg1", arg: 2, scope: !66, file: !3, type: !28)
!71 = !DILocalVariable(name: "omp_sched_static", scope: !66, file: !3, type: !9)
!72 = !DILocalVariable(name: "omp_sched_dynamic", scope: !66, file: !3, type: !9)
!73 = !DILocalVariable(name: "omp_proc_bind_false", scope: !66, file: !3, type: !9)
!74 = !DILocalVariable(name: "omp_proc_bind_true", scope: !66, file: !3, type: !9)
!75 = !DILocalVariable(name: "omp_proc_bind_master", scope: !66, file: !3, type: !9)
!76 = !DILocalVariable(name: "omp_lock_hint_none", scope: !66, file: !3, type: !9)
!77 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !66, file: !3, type: !9)
!78 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !66, file: !3, type: !9)
!79 = !DILocation(line: 24, column: 1, scope: !66)
!80 = !DILocation(line: 25, column: 1, scope: !66)
!81 = !DILocation(line: 26, column: 1, scope: !66)

; ModuleID = '/tmp/DRB107-taskgroup-orig-no-4880e7.ll'
source_filename = "/tmp/DRB107-taskgroup-orig-no-4880e7.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.STATICS1 = type <{ [52 x i8] }>
%astruct.dt60 = type <{ i8* }>

@.STATICS1 = internal global %struct.STATICS1 <{ [52 x i8] c"\FB\FF\FF\FF\08\00\00\00result =\00\00\00\00\03\00\00\00\EB\FF\FF\FF\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\FF\FF\FF\FF\00\00\00\00" }>, align 16
@.C312_MAIN_ = internal constant i32 25
@.C284_MAIN_ = internal constant i64 0
@.C335_MAIN_ = internal constant i32 6
@.C331_MAIN_ = internal constant [53 x i8] c"micro-benchmarks-fortran/DRB107-taskgroup-orig-no.f95"
@.C333_MAIN_ = internal constant i32 32
@.C300_MAIN_ = internal constant i32 2
@.C301_MAIN_ = internal constant i32 3
@.C285_MAIN_ = internal constant i32 1
@.C283_MAIN_ = internal constant i32 0
@.C300___nv_MAIN__F1L18_1 = internal constant i32 2
@.C301___nv_MAIN__F1L18_1 = internal constant i32 3
@.C283___nv_MAIN__F1L18_1 = internal constant i32 0
@.C285___nv_MAIN__F1L18_1 = internal constant i32 1
@.C285___nv_MAIN_F1L21_2 = internal constant i32 1
@.C301___nv_MAIN_F1L21_2 = internal constant i32 3
@.C300___nv_MAIN_F1L26_3 = internal constant i32 2

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__361 = alloca i32, align 4
  %result_313 = alloca i32, align 4
  %.uplevelArgPack0001_356 = alloca %astruct.dt60, align 8
  %z__io_337 = alloca i32, align 4
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
  store i32 %0, i32* %__gtid_MAIN__361, align 4, !dbg !20
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !21
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !21
  call void (i8*, ...) %2(i8* %1), !dbg !21
  br label %L.LB1_350

L.LB1_350:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %result_313, metadata !22, metadata !DIExpression()), !dbg !10
  store i32 0, i32* %result_313, align 4, !dbg !23
  %3 = bitcast i32* %result_313 to i8*, !dbg !24
  %4 = bitcast %astruct.dt60* %.uplevelArgPack0001_356 to i8**, !dbg !24
  store i8* %3, i8** %4, align 8, !dbg !24
  br label %L.LB1_359, !dbg !24

L.LB1_359:                                        ; preds = %L.LB1_350
  %5 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L18_1_ to i64*, !dbg !24
  %6 = bitcast %astruct.dt60* %.uplevelArgPack0001_356 to i64*, !dbg !24
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %5, i64* %6), !dbg !24
  call void (...) @_mp_bcs_nest(), !dbg !25
  %7 = bitcast i32* @.C333_MAIN_ to i8*, !dbg !25
  %8 = bitcast [53 x i8]* @.C331_MAIN_ to i8*, !dbg !25
  %9 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !25
  call void (i8*, i8*, i64, ...) %9(i8* %7, i8* %8, i64 53), !dbg !25
  %10 = bitcast i32* @.C335_MAIN_ to i8*, !dbg !25
  %11 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !25
  %12 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !25
  %13 = bitcast %struct.STATICS1* @.STATICS1 to i8*, !dbg !25
  %14 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !25
  %15 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %14(i8* %10, i8* null, i8* %11, i8* %12, i8* %13, i8* null, i64 0), !dbg !25
  call void @llvm.dbg.declare(metadata i32* %z__io_337, metadata !26, metadata !DIExpression()), !dbg !10
  store i32 %15, i32* %z__io_337, align 4, !dbg !25
  %16 = load i32, i32* %result_313, align 4, !dbg !25
  call void @llvm.dbg.value(metadata i32 %16, metadata !22, metadata !DIExpression()), !dbg !10
  %17 = bitcast i32 (...)* @f90io_sc_i_fmt_write to i32 (i32, i32, ...)*, !dbg !25
  %18 = call i32 (i32, i32, ...) %17(i32 %16, i32 25), !dbg !25
  store i32 %18, i32* %z__io_337, align 4, !dbg !25
  %19 = call i32 (...) @f90io_fmtw_end(), !dbg !25
  store i32 %19, i32* %z__io_337, align 4, !dbg !25
  call void (...) @_mp_ecs_nest(), !dbg !25
  ret void, !dbg !20
}

define internal void @__nv_MAIN__F1L18_1_(i32* %__nv_MAIN__F1L18_1Arg0, i64* %__nv_MAIN__F1L18_1Arg1, i64* %__nv_MAIN__F1L18_1Arg2) #0 !dbg !27 {
L.entry:
  %__gtid___nv_MAIN__F1L18_1__396 = alloca i32, align 4
  %.s0000_391 = alloca i32, align 4
  %.s0001_392 = alloca i32, align 4
  %.s0002_405 = alloca i32, align 4
  %.z0352_404 = alloca i8*, align 8
  %.s0003_432 = alloca i32, align 4
  %.z0352_431 = alloca i8*, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L18_1Arg0, metadata !31, metadata !DIExpression()), !dbg !32
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L18_1Arg1, metadata !33, metadata !DIExpression()), !dbg !32
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L18_1Arg2, metadata !34, metadata !DIExpression()), !dbg !32
  call void @llvm.dbg.value(metadata i32 1, metadata !35, metadata !DIExpression()), !dbg !32
  call void @llvm.dbg.value(metadata i32 2, metadata !36, metadata !DIExpression()), !dbg !32
  call void @llvm.dbg.value(metadata i32 3, metadata !37, metadata !DIExpression()), !dbg !32
  call void @llvm.dbg.value(metadata i32 0, metadata !38, metadata !DIExpression()), !dbg !32
  call void @llvm.dbg.value(metadata i32 1, metadata !39, metadata !DIExpression()), !dbg !32
  call void @llvm.dbg.value(metadata i32 2, metadata !40, metadata !DIExpression()), !dbg !32
  call void @llvm.dbg.value(metadata i32 3, metadata !41, metadata !DIExpression()), !dbg !32
  call void @llvm.dbg.value(metadata i32 0, metadata !42, metadata !DIExpression()), !dbg !32
  call void @llvm.dbg.value(metadata i32 1, metadata !43, metadata !DIExpression()), !dbg !32
  call void @llvm.dbg.value(metadata i32 2, metadata !44, metadata !DIExpression()), !dbg !32
  %0 = load i32, i32* %__nv_MAIN__F1L18_1Arg0, align 4, !dbg !45
  store i32 %0, i32* %__gtid___nv_MAIN__F1L18_1__396, align 4, !dbg !45
  br label %L.LB2_390

L.LB2_390:                                        ; preds = %L.entry
  br label %L.LB2_316

L.LB2_316:                                        ; preds = %L.LB2_390
  store i32 -1, i32* %.s0000_391, align 4, !dbg !46
  store i32 0, i32* %.s0001_392, align 4, !dbg !46
  %1 = load i32, i32* %__gtid___nv_MAIN__F1L18_1__396, align 4, !dbg !46
  %2 = call i32 @__kmpc_single(i64* null, i32 %1), !dbg !46
  %3 = icmp eq i32 %2, 0, !dbg !46
  br i1 %3, label %L.LB2_346, label %L.LB2_318, !dbg !46

L.LB2_318:                                        ; preds = %L.LB2_316
  %4 = load i32, i32* %__gtid___nv_MAIN__F1L18_1__396, align 4, !dbg !47
  call void @__kmpc_taskgroup(i64* null, i32 %4), !dbg !47
  store i32 1, i32* %.s0002_405, align 4, !dbg !48
  %5 = load i32, i32* %__gtid___nv_MAIN__F1L18_1__396, align 4, !dbg !49
  %6 = load i32, i32* %.s0002_405, align 4, !dbg !49
  %7 = bitcast void (i32, i64*)* @__nv_MAIN_F1L21_2_ to i64*, !dbg !49
  %8 = call i8* @__kmpc_omp_task_alloc(i64* null, i32 %5, i32 %6, i32 40, i32 8, i64* %7), !dbg !49
  store i8* %8, i8** %.z0352_404, align 8, !dbg !49
  %9 = load i64, i64* %__nv_MAIN__F1L18_1Arg2, align 8, !dbg !49
  %10 = load i8*, i8** %.z0352_404, align 8, !dbg !49
  %11 = bitcast i8* %10 to i64**, !dbg !49
  %12 = load i64*, i64** %11, align 8, !dbg !49
  store i64 %9, i64* %12, align 8, !dbg !49
  %13 = load i32, i32* %__gtid___nv_MAIN__F1L18_1__396, align 4, !dbg !49
  %14 = load i8*, i8** %.z0352_404, align 8, !dbg !49
  %15 = bitcast i8* %14 to i64*, !dbg !49
  call void @__kmpc_omp_task(i64* null, i32 %13, i64* %15), !dbg !49
  br label %L.LB2_347

L.LB2_347:                                        ; preds = %L.LB2_318
  %16 = load i32, i32* %__gtid___nv_MAIN__F1L18_1__396, align 4, !dbg !50
  call void @__kmpc_end_taskgroup(i64* null, i32 %16), !dbg !50
  store i32 1, i32* %.s0003_432, align 4, !dbg !51
  %17 = load i32, i32* %__gtid___nv_MAIN__F1L18_1__396, align 4, !dbg !52
  %18 = load i32, i32* %.s0003_432, align 4, !dbg !52
  %19 = bitcast void (i32, i64*)* @__nv_MAIN_F1L26_3_ to i64*, !dbg !52
  %20 = call i8* @__kmpc_omp_task_alloc(i64* null, i32 %17, i32 %18, i32 40, i32 8, i64* %19), !dbg !52
  store i8* %20, i8** %.z0352_431, align 8, !dbg !52
  %21 = load i64, i64* %__nv_MAIN__F1L18_1Arg2, align 8, !dbg !52
  %22 = load i8*, i8** %.z0352_431, align 8, !dbg !52
  %23 = bitcast i8* %22 to i64**, !dbg !52
  %24 = load i64*, i64** %23, align 8, !dbg !52
  store i64 %21, i64* %24, align 8, !dbg !52
  %25 = load i32, i32* %__gtid___nv_MAIN__F1L18_1__396, align 4, !dbg !52
  %26 = load i8*, i8** %.z0352_431, align 8, !dbg !52
  %27 = bitcast i8* %26 to i64*, !dbg !52
  call void @__kmpc_omp_task(i64* null, i32 %25, i64* %27), !dbg !52
  br label %L.LB2_348

L.LB2_348:                                        ; preds = %L.LB2_347
  %28 = load i32, i32* %__gtid___nv_MAIN__F1L18_1__396, align 4, !dbg !53
  store i32 %28, i32* %.s0000_391, align 4, !dbg !53
  store i32 1, i32* %.s0001_392, align 4, !dbg !53
  %29 = load i32, i32* %__gtid___nv_MAIN__F1L18_1__396, align 4, !dbg !53
  call void @__kmpc_end_single(i64* null, i32 %29), !dbg !53
  br label %L.LB2_346

L.LB2_346:                                        ; preds = %L.LB2_348, %L.LB2_316
  br label %L.LB2_328

L.LB2_328:                                        ; preds = %L.LB2_346
  %30 = load i32, i32* %__gtid___nv_MAIN__F1L18_1__396, align 4, !dbg !53
  call void @__kmpc_barrier(i64* null, i32 %30), !dbg !53
  br label %L.LB2_329

L.LB2_329:                                        ; preds = %L.LB2_328
  ret void, !dbg !45
}

define internal void @__nv_MAIN_F1L21_2_(i32 %__nv_MAIN_F1L21_2Arg0.arg, i64* %__nv_MAIN_F1L21_2Arg1) #0 !dbg !54 {
L.entry:
  %__nv_MAIN_F1L21_2Arg0.addr = alloca i32, align 4
  %.S0000_453 = alloca i8*, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L21_2Arg0.addr, metadata !57, metadata !DIExpression()), !dbg !58
  store i32 %__nv_MAIN_F1L21_2Arg0.arg, i32* %__nv_MAIN_F1L21_2Arg0.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L21_2Arg0.addr, metadata !59, metadata !DIExpression()), !dbg !58
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L21_2Arg1, metadata !60, metadata !DIExpression()), !dbg !58
  call void @llvm.dbg.value(metadata i32 1, metadata !61, metadata !DIExpression()), !dbg !58
  call void @llvm.dbg.value(metadata i32 2, metadata !62, metadata !DIExpression()), !dbg !58
  call void @llvm.dbg.value(metadata i32 3, metadata !63, metadata !DIExpression()), !dbg !58
  call void @llvm.dbg.value(metadata i32 0, metadata !64, metadata !DIExpression()), !dbg !58
  call void @llvm.dbg.value(metadata i32 1, metadata !65, metadata !DIExpression()), !dbg !58
  call void @llvm.dbg.value(metadata i32 2, metadata !66, metadata !DIExpression()), !dbg !58
  call void @llvm.dbg.value(metadata i32 3, metadata !67, metadata !DIExpression()), !dbg !58
  call void @llvm.dbg.value(metadata i32 0, metadata !68, metadata !DIExpression()), !dbg !58
  call void @llvm.dbg.value(metadata i32 1, metadata !69, metadata !DIExpression()), !dbg !58
  call void @llvm.dbg.value(metadata i32 2, metadata !70, metadata !DIExpression()), !dbg !58
  %0 = bitcast i64* %__nv_MAIN_F1L21_2Arg1 to i8**, !dbg !71
  %1 = load i8*, i8** %0, align 8, !dbg !71
  store i8* %1, i8** %.S0000_453, align 8, !dbg !71
  br label %L.LB4_457

L.LB4_457:                                        ; preds = %L.entry
  br label %L.LB4_321

L.LB4_321:                                        ; preds = %L.LB4_457
  %2 = bitcast i32* @.C301___nv_MAIN_F1L21_2 to i8*, !dbg !72
  %3 = bitcast void (...)* @sleep_ to void (i8*, ...)*, !dbg !72
  call void (i8*, ...) %3(i8* %2), !dbg !72
  %4 = load i8*, i8** %.S0000_453, align 8, !dbg !73
  %5 = bitcast i8* %4 to i32**, !dbg !73
  %6 = load i32*, i32** %5, align 8, !dbg !73
  store i32 1, i32* %6, align 4, !dbg !73
  br label %L.LB4_323

L.LB4_323:                                        ; preds = %L.LB4_321
  ret void, !dbg !74
}

define internal void @__nv_MAIN_F1L26_3_(i32 %__nv_MAIN_F1L26_3Arg0.arg, i64* %__nv_MAIN_F1L26_3Arg1) #0 !dbg !75 {
L.entry:
  %__nv_MAIN_F1L26_3Arg0.addr = alloca i32, align 4
  %.S0000_453 = alloca i8*, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L26_3Arg0.addr, metadata !76, metadata !DIExpression()), !dbg !77
  store i32 %__nv_MAIN_F1L26_3Arg0.arg, i32* %__nv_MAIN_F1L26_3Arg0.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L26_3Arg0.addr, metadata !78, metadata !DIExpression()), !dbg !77
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L26_3Arg1, metadata !79, metadata !DIExpression()), !dbg !77
  call void @llvm.dbg.value(metadata i32 1, metadata !80, metadata !DIExpression()), !dbg !77
  call void @llvm.dbg.value(metadata i32 2, metadata !81, metadata !DIExpression()), !dbg !77
  call void @llvm.dbg.value(metadata i32 3, metadata !82, metadata !DIExpression()), !dbg !77
  call void @llvm.dbg.value(metadata i32 0, metadata !83, metadata !DIExpression()), !dbg !77
  call void @llvm.dbg.value(metadata i32 1, metadata !84, metadata !DIExpression()), !dbg !77
  call void @llvm.dbg.value(metadata i32 2, metadata !85, metadata !DIExpression()), !dbg !77
  call void @llvm.dbg.value(metadata i32 3, metadata !86, metadata !DIExpression()), !dbg !77
  call void @llvm.dbg.value(metadata i32 0, metadata !87, metadata !DIExpression()), !dbg !77
  call void @llvm.dbg.value(metadata i32 1, metadata !88, metadata !DIExpression()), !dbg !77
  call void @llvm.dbg.value(metadata i32 2, metadata !89, metadata !DIExpression()), !dbg !77
  %0 = bitcast i64* %__nv_MAIN_F1L26_3Arg1 to i8**, !dbg !90
  %1 = load i8*, i8** %0, align 8, !dbg !90
  store i8* %1, i8** %.S0000_453, align 8, !dbg !90
  br label %L.LB5_464

L.LB5_464:                                        ; preds = %L.entry
  br label %L.LB5_326

L.LB5_326:                                        ; preds = %L.LB5_464
  %2 = load i8*, i8** %.S0000_453, align 8, !dbg !91
  %3 = bitcast i8* %2 to i32**, !dbg !91
  %4 = load i32*, i32** %3, align 8, !dbg !91
  store i32 2, i32* %4, align 4, !dbg !91
  br label %L.LB5_327

L.LB5_327:                                        ; preds = %L.LB5_326
  ret void, !dbg !92
}

declare void @sleep_(...) #0

declare void @__kmpc_barrier(i64*, i32) #0

declare void @__kmpc_end_single(i64*, i32) #0

declare void @__kmpc_end_taskgroup(i64*, i32) #0

declare void @__kmpc_omp_task(i64*, i32, i64*) #0

declare i8* @__kmpc_omp_task_alloc(i64*, i32, i32, i32, i32, i64*) #0

declare void @__kmpc_taskgroup(i64*, i32) #0

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
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB107-taskgroup-orig-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb107_taskgroup_orig_no", scope: !2, file: !3, line: 11, type: !6, scopeLine: 11, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
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
!20 = !DILocation(line: 35, column: 1, scope: !5)
!21 = !DILocation(line: 11, column: 1, scope: !5)
!22 = !DILocalVariable(name: "result", scope: !5, file: !3, type: !9)
!23 = !DILocation(line: 16, column: 1, scope: !5)
!24 = !DILocation(line: 18, column: 1, scope: !5)
!25 = !DILocation(line: 32, column: 1, scope: !5)
!26 = !DILocalVariable(scope: !5, file: !3, type: !9, flags: DIFlagArtificial)
!27 = distinct !DISubprogram(name: "__nv_MAIN__F1L18_1", scope: !2, file: !3, line: 18, type: !28, scopeLine: 18, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!28 = !DISubroutineType(types: !29)
!29 = !{null, !9, !30, !30}
!30 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!31 = !DILocalVariable(name: "__nv_MAIN__F1L18_1Arg0", arg: 1, scope: !27, file: !3, type: !9)
!32 = !DILocation(line: 0, scope: !27)
!33 = !DILocalVariable(name: "__nv_MAIN__F1L18_1Arg1", arg: 2, scope: !27, file: !3, type: !30)
!34 = !DILocalVariable(name: "__nv_MAIN__F1L18_1Arg2", arg: 3, scope: !27, file: !3, type: !30)
!35 = !DILocalVariable(name: "omp_sched_static", scope: !27, file: !3, type: !9)
!36 = !DILocalVariable(name: "omp_sched_dynamic", scope: !27, file: !3, type: !9)
!37 = !DILocalVariable(name: "omp_sched_guided", scope: !27, file: !3, type: !9)
!38 = !DILocalVariable(name: "omp_proc_bind_false", scope: !27, file: !3, type: !9)
!39 = !DILocalVariable(name: "omp_proc_bind_true", scope: !27, file: !3, type: !9)
!40 = !DILocalVariable(name: "omp_proc_bind_master", scope: !27, file: !3, type: !9)
!41 = !DILocalVariable(name: "omp_proc_bind_close", scope: !27, file: !3, type: !9)
!42 = !DILocalVariable(name: "omp_lock_hint_none", scope: !27, file: !3, type: !9)
!43 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !27, file: !3, type: !9)
!44 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !27, file: !3, type: !9)
!45 = !DILocation(line: 30, column: 1, scope: !27)
!46 = !DILocation(line: 19, column: 1, scope: !27)
!47 = !DILocation(line: 20, column: 1, scope: !27)
!48 = !DILocation(line: 21, column: 1, scope: !27)
!49 = !DILocation(line: 24, column: 1, scope: !27)
!50 = !DILocation(line: 25, column: 1, scope: !27)
!51 = !DILocation(line: 26, column: 1, scope: !27)
!52 = !DILocation(line: 28, column: 1, scope: !27)
!53 = !DILocation(line: 29, column: 1, scope: !27)
!54 = distinct !DISubprogram(name: "__nv_MAIN_F1L21_2", scope: !2, file: !3, line: 21, type: !55, scopeLine: 21, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!55 = !DISubroutineType(types: !56)
!56 = !{null, !9, !30}
!57 = !DILocalVariable(name: "__nv_MAIN_F1L21_2Arg0", scope: !54, file: !3, type: !9)
!58 = !DILocation(line: 0, scope: !54)
!59 = !DILocalVariable(name: "__nv_MAIN_F1L21_2Arg0", arg: 1, scope: !54, file: !3, type: !9)
!60 = !DILocalVariable(name: "__nv_MAIN_F1L21_2Arg1", arg: 2, scope: !54, file: !3, type: !30)
!61 = !DILocalVariable(name: "omp_sched_static", scope: !54, file: !3, type: !9)
!62 = !DILocalVariable(name: "omp_sched_dynamic", scope: !54, file: !3, type: !9)
!63 = !DILocalVariable(name: "omp_sched_guided", scope: !54, file: !3, type: !9)
!64 = !DILocalVariable(name: "omp_proc_bind_false", scope: !54, file: !3, type: !9)
!65 = !DILocalVariable(name: "omp_proc_bind_true", scope: !54, file: !3, type: !9)
!66 = !DILocalVariable(name: "omp_proc_bind_master", scope: !54, file: !3, type: !9)
!67 = !DILocalVariable(name: "omp_proc_bind_close", scope: !54, file: !3, type: !9)
!68 = !DILocalVariable(name: "omp_lock_hint_none", scope: !54, file: !3, type: !9)
!69 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !54, file: !3, type: !9)
!70 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !54, file: !3, type: !9)
!71 = !DILocation(line: 21, column: 1, scope: !54)
!72 = !DILocation(line: 22, column: 1, scope: !54)
!73 = !DILocation(line: 23, column: 1, scope: !54)
!74 = !DILocation(line: 24, column: 1, scope: !54)
!75 = distinct !DISubprogram(name: "__nv_MAIN_F1L26_3", scope: !2, file: !3, line: 26, type: !55, scopeLine: 26, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!76 = !DILocalVariable(name: "__nv_MAIN_F1L26_3Arg0", scope: !75, file: !3, type: !9)
!77 = !DILocation(line: 0, scope: !75)
!78 = !DILocalVariable(name: "__nv_MAIN_F1L26_3Arg0", arg: 1, scope: !75, file: !3, type: !9)
!79 = !DILocalVariable(name: "__nv_MAIN_F1L26_3Arg1", arg: 2, scope: !75, file: !3, type: !30)
!80 = !DILocalVariable(name: "omp_sched_static", scope: !75, file: !3, type: !9)
!81 = !DILocalVariable(name: "omp_sched_dynamic", scope: !75, file: !3, type: !9)
!82 = !DILocalVariable(name: "omp_sched_guided", scope: !75, file: !3, type: !9)
!83 = !DILocalVariable(name: "omp_proc_bind_false", scope: !75, file: !3, type: !9)
!84 = !DILocalVariable(name: "omp_proc_bind_true", scope: !75, file: !3, type: !9)
!85 = !DILocalVariable(name: "omp_proc_bind_master", scope: !75, file: !3, type: !9)
!86 = !DILocalVariable(name: "omp_proc_bind_close", scope: !75, file: !3, type: !9)
!87 = !DILocalVariable(name: "omp_lock_hint_none", scope: !75, file: !3, type: !9)
!88 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !75, file: !3, type: !9)
!89 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !75, file: !3, type: !9)
!90 = !DILocation(line: 26, column: 1, scope: !75)
!91 = !DILocation(line: 27, column: 1, scope: !75)
!92 = !DILocation(line: 28, column: 1, scope: !75)

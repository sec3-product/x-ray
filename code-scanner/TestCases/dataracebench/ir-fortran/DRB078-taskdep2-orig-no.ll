; ModuleID = '/tmp/DRB078-taskdep2-orig-no-39b862.ll'
source_filename = "/tmp/DRB078-taskdep2-orig-no-39b862.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.STATICS1 = type <{ [28 x i8] }>
%astruct.dt58 = type <{ i8* }>

@.STATICS1 = internal global %struct.STATICS1 <{ [28 x i8] c"\EB\FF\FF\FF\00\00\00\00\03\00\00\00\00\00\00\00\01\00\00\00\FF\FF\FF\FF\00\00\00\00" }>, align 16
@.C312_MAIN_ = internal constant i32 25
@.C284_MAIN_ = internal constant i64 0
@.C335_MAIN_ = internal constant i32 6
@.C331_MAIN_ = internal constant [52 x i8] c"micro-benchmarks-fortran/DRB078-taskdep2-orig-no.f95"
@.C333_MAIN_ = internal constant i32 32
@.C300_MAIN_ = internal constant i32 2
@.C301_MAIN_ = internal constant i32 3
@.C285_MAIN_ = internal constant i32 1
@.C283_MAIN_ = internal constant i32 0
@.C300___nv_MAIN__F1L19_1 = internal constant i32 2
@.C301___nv_MAIN__F1L19_1 = internal constant i32 3
@.C283___nv_MAIN__F1L19_1 = internal constant i32 0
@.C285___nv_MAIN__F1L19_1 = internal constant i32 1
@.C301___nv_MAIN_F1L21_2 = internal constant i32 3
@.C300___nv_MAIN_F1L25_3 = internal constant i32 2

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__361 = alloca i32, align 4
  %i_313 = alloca i32, align 4
  %.uplevelArgPack0001_356 = alloca %astruct.dt58, align 8
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
  call void @llvm.dbg.declare(metadata i32* %i_313, metadata !22, metadata !DIExpression()), !dbg !10
  store i32 0, i32* %i_313, align 4, !dbg !23
  %3 = bitcast i32* %i_313 to i8*, !dbg !24
  %4 = bitcast %astruct.dt58* %.uplevelArgPack0001_356 to i8**, !dbg !24
  store i8* %3, i8** %4, align 8, !dbg !24
  br label %L.LB1_359, !dbg !24

L.LB1_359:                                        ; preds = %L.LB1_350
  %5 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L19_1_ to i64*, !dbg !24
  %6 = bitcast %astruct.dt58* %.uplevelArgPack0001_356 to i64*, !dbg !24
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %5, i64* %6), !dbg !24
  %7 = load i32, i32* %i_313, align 4, !dbg !25
  call void @llvm.dbg.value(metadata i32 %7, metadata !22, metadata !DIExpression()), !dbg !10
  %8 = icmp eq i32 %7, 2, !dbg !25
  br i1 %8, label %L.LB1_348, label %L.LB1_387, !dbg !25

L.LB1_387:                                        ; preds = %L.LB1_359
  call void (...) @_mp_bcs_nest(), !dbg !26
  %9 = bitcast i32* @.C333_MAIN_ to i8*, !dbg !26
  %10 = bitcast [52 x i8]* @.C331_MAIN_ to i8*, !dbg !26
  %11 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !26
  call void (i8*, i8*, i64, ...) %11(i8* %9, i8* %10, i64 52), !dbg !26
  %12 = bitcast i32* @.C335_MAIN_ to i8*, !dbg !26
  %13 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !26
  %14 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !26
  %15 = bitcast %struct.STATICS1* @.STATICS1 to i8*, !dbg !26
  %16 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !26
  %17 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %16(i8* %12, i8* null, i8* %13, i8* %14, i8* %15, i8* null, i64 0), !dbg !26
  call void @llvm.dbg.declare(metadata i32* %z__io_337, metadata !27, metadata !DIExpression()), !dbg !10
  store i32 %17, i32* %z__io_337, align 4, !dbg !26
  %18 = load i32, i32* %i_313, align 4, !dbg !26
  call void @llvm.dbg.value(metadata i32 %18, metadata !22, metadata !DIExpression()), !dbg !10
  %19 = bitcast i32 (...)* @f90io_sc_i_fmt_write to i32 (i32, i32, ...)*, !dbg !26
  %20 = call i32 (i32, i32, ...) %19(i32 %18, i32 25), !dbg !26
  store i32 %20, i32* %z__io_337, align 4, !dbg !26
  %21 = call i32 (...) @f90io_fmtw_end(), !dbg !26
  store i32 %21, i32* %z__io_337, align 4, !dbg !26
  call void (...) @_mp_ecs_nest(), !dbg !26
  br label %L.LB1_348

L.LB1_348:                                        ; preds = %L.LB1_387, %L.LB1_359
  ret void, !dbg !20
}

define internal void @__nv_MAIN__F1L19_1_(i32* %__nv_MAIN__F1L19_1Arg0, i64* %__nv_MAIN__F1L19_1Arg1, i64* %__nv_MAIN__F1L19_1Arg2) #0 !dbg !28 {
L.entry:
  %__gtid___nv_MAIN__F1L19_1__397 = alloca i32, align 4
  %.s0000_392 = alloca i32, align 4
  %.s0001_393 = alloca i32, align 4
  %.s0002_403 = alloca i32, align 4
  %.z0352_402 = alloca i8*, align 8
  %.s0003_427 = alloca i32, align 4
  %.z0352_426 = alloca i8*, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L19_1Arg0, metadata !32, metadata !DIExpression()), !dbg !33
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L19_1Arg1, metadata !34, metadata !DIExpression()), !dbg !33
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L19_1Arg2, metadata !35, metadata !DIExpression()), !dbg !33
  call void @llvm.dbg.value(metadata i32 1, metadata !36, metadata !DIExpression()), !dbg !33
  call void @llvm.dbg.value(metadata i32 2, metadata !37, metadata !DIExpression()), !dbg !33
  call void @llvm.dbg.value(metadata i32 3, metadata !38, metadata !DIExpression()), !dbg !33
  call void @llvm.dbg.value(metadata i32 0, metadata !39, metadata !DIExpression()), !dbg !33
  call void @llvm.dbg.value(metadata i32 1, metadata !40, metadata !DIExpression()), !dbg !33
  call void @llvm.dbg.value(metadata i32 2, metadata !41, metadata !DIExpression()), !dbg !33
  call void @llvm.dbg.value(metadata i32 3, metadata !42, metadata !DIExpression()), !dbg !33
  call void @llvm.dbg.value(metadata i32 0, metadata !43, metadata !DIExpression()), !dbg !33
  call void @llvm.dbg.value(metadata i32 1, metadata !44, metadata !DIExpression()), !dbg !33
  call void @llvm.dbg.value(metadata i32 2, metadata !45, metadata !DIExpression()), !dbg !33
  %0 = load i32, i32* %__nv_MAIN__F1L19_1Arg0, align 4, !dbg !46
  store i32 %0, i32* %__gtid___nv_MAIN__F1L19_1__397, align 4, !dbg !46
  br label %L.LB2_391

L.LB2_391:                                        ; preds = %L.entry
  br label %L.LB2_316

L.LB2_316:                                        ; preds = %L.LB2_391
  store i32 -1, i32* %.s0000_392, align 4, !dbg !47
  store i32 0, i32* %.s0001_393, align 4, !dbg !47
  %1 = load i32, i32* %__gtid___nv_MAIN__F1L19_1__397, align 4, !dbg !47
  %2 = call i32 @__kmpc_single(i64* null, i32 %1), !dbg !47
  %3 = icmp eq i32 %2, 0, !dbg !47
  br i1 %3, label %L.LB2_345, label %L.LB2_318, !dbg !47

L.LB2_318:                                        ; preds = %L.LB2_316
  store i32 1, i32* %.s0002_403, align 4, !dbg !48
  %4 = load i32, i32* %__gtid___nv_MAIN__F1L19_1__397, align 4, !dbg !49
  %5 = load i32, i32* %.s0002_403, align 4, !dbg !49
  %6 = bitcast void (i32, i64*)* @__nv_MAIN_F1L21_2_ to i64*, !dbg !49
  %7 = call i8* @__kmpc_omp_task_alloc(i64* null, i32 %4, i32 %5, i32 40, i32 8, i64* %6), !dbg !49
  store i8* %7, i8** %.z0352_402, align 8, !dbg !49
  %8 = load i64, i64* %__nv_MAIN__F1L19_1Arg2, align 8, !dbg !49
  %9 = load i8*, i8** %.z0352_402, align 8, !dbg !49
  %10 = bitcast i8* %9 to i64**, !dbg !49
  %11 = load i64*, i64** %10, align 8, !dbg !49
  store i64 %8, i64* %11, align 8, !dbg !49
  %12 = load i32, i32* %__gtid___nv_MAIN__F1L19_1__397, align 4, !dbg !49
  %13 = load i8*, i8** %.z0352_402, align 8, !dbg !49
  %14 = bitcast i8* %13 to i64*, !dbg !49
  call void @__kmpc_omp_task(i64* null, i32 %12, i64* %14), !dbg !49
  br label %L.LB2_346

L.LB2_346:                                        ; preds = %L.LB2_318
  store i32 1, i32* %.s0003_427, align 4, !dbg !50
  %15 = load i32, i32* %__gtid___nv_MAIN__F1L19_1__397, align 4, !dbg !51
  %16 = load i32, i32* %.s0003_427, align 4, !dbg !51
  %17 = bitcast void (i32, i64*)* @__nv_MAIN_F1L25_3_ to i64*, !dbg !51
  %18 = call i8* @__kmpc_omp_task_alloc(i64* null, i32 %15, i32 %16, i32 40, i32 8, i64* %17), !dbg !51
  store i8* %18, i8** %.z0352_426, align 8, !dbg !51
  %19 = load i64, i64* %__nv_MAIN__F1L19_1Arg2, align 8, !dbg !51
  %20 = load i8*, i8** %.z0352_426, align 8, !dbg !51
  %21 = bitcast i8* %20 to i64**, !dbg !51
  %22 = load i64*, i64** %21, align 8, !dbg !51
  store i64 %19, i64* %22, align 8, !dbg !51
  %23 = load i32, i32* %__gtid___nv_MAIN__F1L19_1__397, align 4, !dbg !51
  %24 = load i8*, i8** %.z0352_426, align 8, !dbg !51
  %25 = bitcast i8* %24 to i64*, !dbg !51
  call void @__kmpc_omp_task(i64* null, i32 %23, i64* %25), !dbg !51
  br label %L.LB2_347

L.LB2_347:                                        ; preds = %L.LB2_346
  %26 = load i32, i32* %__gtid___nv_MAIN__F1L19_1__397, align 4, !dbg !52
  store i32 %26, i32* %.s0000_392, align 4, !dbg !52
  store i32 1, i32* %.s0001_393, align 4, !dbg !52
  %27 = load i32, i32* %__gtid___nv_MAIN__F1L19_1__397, align 4, !dbg !52
  call void @__kmpc_end_single(i64* null, i32 %27), !dbg !52
  br label %L.LB2_345

L.LB2_345:                                        ; preds = %L.LB2_347, %L.LB2_316
  br label %L.LB2_328

L.LB2_328:                                        ; preds = %L.LB2_345
  %28 = load i32, i32* %__gtid___nv_MAIN__F1L19_1__397, align 4, !dbg !52
  call void @__kmpc_barrier(i64* null, i32 %28), !dbg !52
  br label %L.LB2_329

L.LB2_329:                                        ; preds = %L.LB2_328
  ret void, !dbg !46
}

define internal void @__nv_MAIN_F1L21_2_(i32 %__nv_MAIN_F1L21_2Arg0.arg, i64* %__nv_MAIN_F1L21_2Arg1) #0 !dbg !53 {
L.entry:
  %__nv_MAIN_F1L21_2Arg0.addr = alloca i32, align 4
  %.S0000_448 = alloca i8*, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L21_2Arg0.addr, metadata !56, metadata !DIExpression()), !dbg !57
  store i32 %__nv_MAIN_F1L21_2Arg0.arg, i32* %__nv_MAIN_F1L21_2Arg0.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L21_2Arg0.addr, metadata !58, metadata !DIExpression()), !dbg !57
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L21_2Arg1, metadata !59, metadata !DIExpression()), !dbg !57
  call void @llvm.dbg.value(metadata i32 1, metadata !60, metadata !DIExpression()), !dbg !57
  call void @llvm.dbg.value(metadata i32 2, metadata !61, metadata !DIExpression()), !dbg !57
  call void @llvm.dbg.value(metadata i32 3, metadata !62, metadata !DIExpression()), !dbg !57
  call void @llvm.dbg.value(metadata i32 0, metadata !63, metadata !DIExpression()), !dbg !57
  call void @llvm.dbg.value(metadata i32 1, metadata !64, metadata !DIExpression()), !dbg !57
  call void @llvm.dbg.value(metadata i32 2, metadata !65, metadata !DIExpression()), !dbg !57
  call void @llvm.dbg.value(metadata i32 3, metadata !66, metadata !DIExpression()), !dbg !57
  call void @llvm.dbg.value(metadata i32 0, metadata !67, metadata !DIExpression()), !dbg !57
  call void @llvm.dbg.value(metadata i32 1, metadata !68, metadata !DIExpression()), !dbg !57
  call void @llvm.dbg.value(metadata i32 2, metadata !69, metadata !DIExpression()), !dbg !57
  %0 = bitcast i64* %__nv_MAIN_F1L21_2Arg1 to i8**, !dbg !70
  %1 = load i8*, i8** %0, align 8, !dbg !70
  store i8* %1, i8** %.S0000_448, align 8, !dbg !70
  br label %L.LB4_452

L.LB4_452:                                        ; preds = %L.entry
  br label %L.LB4_321

L.LB4_321:                                        ; preds = %L.LB4_452
  %2 = bitcast i32* @.C301___nv_MAIN_F1L21_2 to i8*, !dbg !71
  %3 = bitcast void (...)* @sleep_ to void (i8*, ...)*, !dbg !71
  call void (i8*, ...) %3(i8* %2), !dbg !71
  %4 = load i8*, i8** %.S0000_448, align 8, !dbg !72
  %5 = bitcast i8* %4 to i32**, !dbg !72
  %6 = load i32*, i32** %5, align 8, !dbg !72
  store i32 3, i32* %6, align 4, !dbg !72
  br label %L.LB4_323

L.LB4_323:                                        ; preds = %L.LB4_321
  ret void, !dbg !73
}

define internal void @__nv_MAIN_F1L25_3_(i32 %__nv_MAIN_F1L25_3Arg0.arg, i64* %__nv_MAIN_F1L25_3Arg1) #0 !dbg !74 {
L.entry:
  %__nv_MAIN_F1L25_3Arg0.addr = alloca i32, align 4
  %.S0000_448 = alloca i8*, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L25_3Arg0.addr, metadata !75, metadata !DIExpression()), !dbg !76
  store i32 %__nv_MAIN_F1L25_3Arg0.arg, i32* %__nv_MAIN_F1L25_3Arg0.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L25_3Arg0.addr, metadata !77, metadata !DIExpression()), !dbg !76
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L25_3Arg1, metadata !78, metadata !DIExpression()), !dbg !76
  call void @llvm.dbg.value(metadata i32 1, metadata !79, metadata !DIExpression()), !dbg !76
  call void @llvm.dbg.value(metadata i32 2, metadata !80, metadata !DIExpression()), !dbg !76
  call void @llvm.dbg.value(metadata i32 3, metadata !81, metadata !DIExpression()), !dbg !76
  call void @llvm.dbg.value(metadata i32 0, metadata !82, metadata !DIExpression()), !dbg !76
  call void @llvm.dbg.value(metadata i32 1, metadata !83, metadata !DIExpression()), !dbg !76
  call void @llvm.dbg.value(metadata i32 2, metadata !84, metadata !DIExpression()), !dbg !76
  call void @llvm.dbg.value(metadata i32 3, metadata !85, metadata !DIExpression()), !dbg !76
  call void @llvm.dbg.value(metadata i32 0, metadata !86, metadata !DIExpression()), !dbg !76
  call void @llvm.dbg.value(metadata i32 1, metadata !87, metadata !DIExpression()), !dbg !76
  call void @llvm.dbg.value(metadata i32 2, metadata !88, metadata !DIExpression()), !dbg !76
  %0 = bitcast i64* %__nv_MAIN_F1L25_3Arg1 to i8**, !dbg !89
  %1 = load i8*, i8** %0, align 8, !dbg !89
  store i8* %1, i8** %.S0000_448, align 8, !dbg !89
  br label %L.LB5_459

L.LB5_459:                                        ; preds = %L.entry
  br label %L.LB5_326

L.LB5_326:                                        ; preds = %L.LB5_459
  %2 = load i8*, i8** %.S0000_448, align 8, !dbg !90
  %3 = bitcast i8* %2 to i32**, !dbg !90
  %4 = load i32*, i32** %3, align 8, !dbg !90
  store i32 2, i32* %4, align 4, !dbg !90
  br label %L.LB5_327

L.LB5_327:                                        ; preds = %L.LB5_326
  ret void, !dbg !91
}

declare void @sleep_(...) #0

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
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB078-taskdep2-orig-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb078_taskdep2_orig_no", scope: !2, file: !3, line: 12, type: !6, scopeLine: 12, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
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
!21 = !DILocation(line: 12, column: 1, scope: !5)
!22 = !DILocalVariable(name: "i", scope: !5, file: !3, type: !9)
!23 = !DILocation(line: 17, column: 1, scope: !5)
!24 = !DILocation(line: 19, column: 1, scope: !5)
!25 = !DILocation(line: 31, column: 1, scope: !5)
!26 = !DILocation(line: 32, column: 1, scope: !5)
!27 = !DILocalVariable(scope: !5, file: !3, type: !9, flags: DIFlagArtificial)
!28 = distinct !DISubprogram(name: "__nv_MAIN__F1L19_1", scope: !2, file: !3, line: 19, type: !29, scopeLine: 19, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!29 = !DISubroutineType(types: !30)
!30 = !{null, !9, !31, !31}
!31 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!32 = !DILocalVariable(name: "__nv_MAIN__F1L19_1Arg0", arg: 1, scope: !28, file: !3, type: !9)
!33 = !DILocation(line: 0, scope: !28)
!34 = !DILocalVariable(name: "__nv_MAIN__F1L19_1Arg1", arg: 2, scope: !28, file: !3, type: !31)
!35 = !DILocalVariable(name: "__nv_MAIN__F1L19_1Arg2", arg: 3, scope: !28, file: !3, type: !31)
!36 = !DILocalVariable(name: "omp_sched_static", scope: !28, file: !3, type: !9)
!37 = !DILocalVariable(name: "omp_sched_dynamic", scope: !28, file: !3, type: !9)
!38 = !DILocalVariable(name: "omp_sched_guided", scope: !28, file: !3, type: !9)
!39 = !DILocalVariable(name: "omp_proc_bind_false", scope: !28, file: !3, type: !9)
!40 = !DILocalVariable(name: "omp_proc_bind_true", scope: !28, file: !3, type: !9)
!41 = !DILocalVariable(name: "omp_proc_bind_master", scope: !28, file: !3, type: !9)
!42 = !DILocalVariable(name: "omp_proc_bind_close", scope: !28, file: !3, type: !9)
!43 = !DILocalVariable(name: "omp_lock_hint_none", scope: !28, file: !3, type: !9)
!44 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !28, file: !3, type: !9)
!45 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !28, file: !3, type: !9)
!46 = !DILocation(line: 29, column: 1, scope: !28)
!47 = !DILocation(line: 20, column: 1, scope: !28)
!48 = !DILocation(line: 21, column: 1, scope: !28)
!49 = !DILocation(line: 24, column: 1, scope: !28)
!50 = !DILocation(line: 25, column: 1, scope: !28)
!51 = !DILocation(line: 27, column: 1, scope: !28)
!52 = !DILocation(line: 28, column: 1, scope: !28)
!53 = distinct !DISubprogram(name: "__nv_MAIN_F1L21_2", scope: !2, file: !3, line: 21, type: !54, scopeLine: 21, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!54 = !DISubroutineType(types: !55)
!55 = !{null, !9, !31}
!56 = !DILocalVariable(name: "__nv_MAIN_F1L21_2Arg0", scope: !53, file: !3, type: !9)
!57 = !DILocation(line: 0, scope: !53)
!58 = !DILocalVariable(name: "__nv_MAIN_F1L21_2Arg0", arg: 1, scope: !53, file: !3, type: !9)
!59 = !DILocalVariable(name: "__nv_MAIN_F1L21_2Arg1", arg: 2, scope: !53, file: !3, type: !31)
!60 = !DILocalVariable(name: "omp_sched_static", scope: !53, file: !3, type: !9)
!61 = !DILocalVariable(name: "omp_sched_dynamic", scope: !53, file: !3, type: !9)
!62 = !DILocalVariable(name: "omp_sched_guided", scope: !53, file: !3, type: !9)
!63 = !DILocalVariable(name: "omp_proc_bind_false", scope: !53, file: !3, type: !9)
!64 = !DILocalVariable(name: "omp_proc_bind_true", scope: !53, file: !3, type: !9)
!65 = !DILocalVariable(name: "omp_proc_bind_master", scope: !53, file: !3, type: !9)
!66 = !DILocalVariable(name: "omp_proc_bind_close", scope: !53, file: !3, type: !9)
!67 = !DILocalVariable(name: "omp_lock_hint_none", scope: !53, file: !3, type: !9)
!68 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !53, file: !3, type: !9)
!69 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !53, file: !3, type: !9)
!70 = !DILocation(line: 21, column: 1, scope: !53)
!71 = !DILocation(line: 22, column: 1, scope: !53)
!72 = !DILocation(line: 23, column: 1, scope: !53)
!73 = !DILocation(line: 24, column: 1, scope: !53)
!74 = distinct !DISubprogram(name: "__nv_MAIN_F1L25_3", scope: !2, file: !3, line: 25, type: !54, scopeLine: 25, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!75 = !DILocalVariable(name: "__nv_MAIN_F1L25_3Arg0", scope: !74, file: !3, type: !9)
!76 = !DILocation(line: 0, scope: !74)
!77 = !DILocalVariable(name: "__nv_MAIN_F1L25_3Arg0", arg: 1, scope: !74, file: !3, type: !9)
!78 = !DILocalVariable(name: "__nv_MAIN_F1L25_3Arg1", arg: 2, scope: !74, file: !3, type: !31)
!79 = !DILocalVariable(name: "omp_sched_static", scope: !74, file: !3, type: !9)
!80 = !DILocalVariable(name: "omp_sched_dynamic", scope: !74, file: !3, type: !9)
!81 = !DILocalVariable(name: "omp_sched_guided", scope: !74, file: !3, type: !9)
!82 = !DILocalVariable(name: "omp_proc_bind_false", scope: !74, file: !3, type: !9)
!83 = !DILocalVariable(name: "omp_proc_bind_true", scope: !74, file: !3, type: !9)
!84 = !DILocalVariable(name: "omp_proc_bind_master", scope: !74, file: !3, type: !9)
!85 = !DILocalVariable(name: "omp_proc_bind_close", scope: !74, file: !3, type: !9)
!86 = !DILocalVariable(name: "omp_lock_hint_none", scope: !74, file: !3, type: !9)
!87 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !74, file: !3, type: !9)
!88 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !74, file: !3, type: !9)
!89 = !DILocation(line: 25, column: 1, scope: !74)
!90 = !DILocation(line: 26, column: 1, scope: !74)
!91 = !DILocation(line: 27, column: 1, scope: !74)

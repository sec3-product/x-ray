; ModuleID = '/tmp/DRB139-worksharingcritical-orig-no-059269.ll'
source_filename = "/tmp/DRB139-worksharingcritical-orig-no-059269.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.STATICS1 = type <{ [48 x i8] }>
%struct__cs_name_ = type <{ [32 x i8] }>
%astruct.dt63 = type <{ i8* }>
%astruct.dt117 = type <{ [8 x i8] }>

@.STATICS1 = internal global %struct.STATICS1 <{ [48 x i8] c"\FB\FF\FF\FF\04\00\00\00i = \00\00\00\00\03\00\00\00\EB\FF\FF\FF\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\FF\FF\FF\FF\00\00\00\00" }>, align 16
@.C309_MAIN_ = internal constant i32 25
@.C284_MAIN_ = internal constant i64 0
@.C331_MAIN_ = internal constant i32 6
@.C327_MAIN_ = internal constant [63 x i8] c"micro-benchmarks-fortran/DRB139-worksharingcritical-orig-no.f95"
@.C329_MAIN_ = internal constant i32 32
@.C300_MAIN_ = internal constant i32 2
@.C285_MAIN_ = internal constant i32 1
@.C283_MAIN_ = internal constant i32 0
@.C300___nv_MAIN__F1L21_1 = internal constant i32 2
@.C285___nv_MAIN__F1L21_1 = internal constant i32 1
@.C283___nv_MAIN__F1L21_1 = internal constant i32 0
@.C285___nv_MAIN_F1L24_2 = internal constant i32 1
@__cs_name_ = common global %struct__cs_name_ zeroinitializer, align 64

define void @MAIN_() #0 !dbg !18 {
L.entry:
  %__gtid_MAIN__359 = alloca i32, align 4
  %i_310 = alloca i32, align 4
  %.uplevelArgPack0001_354 = alloca %astruct.dt63, align 8
  %z__io_333 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !21, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i32 2, metadata !23, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i32 0, metadata !24, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i32 1, metadata !25, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i32 2, metadata !26, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i32 0, metadata !27, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i32 1, metadata !28, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i32 2, metadata !29, metadata !DIExpression()), !dbg !22
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !30
  store i32 %0, i32* %__gtid_MAIN__359, align 4, !dbg !30
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !31
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !31
  call void (i8*, ...) %2(i8* %1), !dbg !31
  br label %L.LB1_348

L.LB1_348:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %i_310, metadata !32, metadata !DIExpression()), !dbg !22
  store i32 1, i32* %i_310, align 4, !dbg !33
  %3 = bitcast i32* %i_310 to i8*, !dbg !34
  %4 = bitcast %astruct.dt63* %.uplevelArgPack0001_354 to i8**, !dbg !34
  store i8* %3, i8** %4, align 8, !dbg !34
  br label %L.LB1_357, !dbg !34

L.LB1_357:                                        ; preds = %L.LB1_348
  %5 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L21_1_ to i64*, !dbg !34
  %6 = bitcast %astruct.dt63* %.uplevelArgPack0001_354 to i64*, !dbg !34
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %5, i64* %6), !dbg !34
  call void (...) @_mp_bcs_nest(), !dbg !35
  %7 = bitcast i32* @.C329_MAIN_ to i8*, !dbg !35
  %8 = bitcast [63 x i8]* @.C327_MAIN_ to i8*, !dbg !35
  %9 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !35
  call void (i8*, i8*, i64, ...) %9(i8* %7, i8* %8, i64 63), !dbg !35
  %10 = bitcast i32* @.C331_MAIN_ to i8*, !dbg !35
  %11 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !35
  %12 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !35
  %13 = bitcast %struct.STATICS1* @.STATICS1 to i8*, !dbg !35
  %14 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !35
  %15 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %14(i8* %10, i8* null, i8* %11, i8* %12, i8* %13, i8* null, i64 0), !dbg !35
  call void @llvm.dbg.declare(metadata i32* %z__io_333, metadata !36, metadata !DIExpression()), !dbg !22
  store i32 %15, i32* %z__io_333, align 4, !dbg !35
  %16 = load i32, i32* %i_310, align 4, !dbg !35
  call void @llvm.dbg.value(metadata i32 %16, metadata !32, metadata !DIExpression()), !dbg !22
  %17 = bitcast i32 (...)* @f90io_sc_i_fmt_write to i32 (i32, i32, ...)*, !dbg !35
  %18 = call i32 (i32, i32, ...) %17(i32 %16, i32 25), !dbg !35
  store i32 %18, i32* %z__io_333, align 4, !dbg !35
  %19 = call i32 (...) @f90io_fmtw_end(), !dbg !35
  store i32 %19, i32* %z__io_333, align 4, !dbg !35
  call void (...) @_mp_ecs_nest(), !dbg !35
  ret void, !dbg !30
}

define internal void @__nv_MAIN__F1L21_1_(i32* %__nv_MAIN__F1L21_1Arg0, i64* %__nv_MAIN__F1L21_1Arg1, i64* %__nv_MAIN__F1L21_1Arg2) #0 !dbg !9 {
L.entry:
  %__gtid___nv_MAIN__F1L21_1__396 = alloca i32, align 4
  %.s0001_391 = alloca i32, align 4
  %.s0000_390 = alloca i32, align 4
  %.s0003_393 = alloca i32, align 4
  %.s0002_392 = alloca i32, align 4
  %.uplevelArgPack0002_420 = alloca %astruct.dt117, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L21_1Arg0, metadata !37, metadata !DIExpression()), !dbg !38
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L21_1Arg1, metadata !39, metadata !DIExpression()), !dbg !38
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L21_1Arg2, metadata !40, metadata !DIExpression()), !dbg !38
  call void @llvm.dbg.value(metadata i32 1, metadata !41, metadata !DIExpression()), !dbg !38
  call void @llvm.dbg.value(metadata i32 2, metadata !42, metadata !DIExpression()), !dbg !38
  call void @llvm.dbg.value(metadata i32 0, metadata !43, metadata !DIExpression()), !dbg !38
  call void @llvm.dbg.value(metadata i32 1, metadata !44, metadata !DIExpression()), !dbg !38
  call void @llvm.dbg.value(metadata i32 2, metadata !45, metadata !DIExpression()), !dbg !38
  call void @llvm.dbg.value(metadata i32 0, metadata !46, metadata !DIExpression()), !dbg !38
  call void @llvm.dbg.value(metadata i32 1, metadata !47, metadata !DIExpression()), !dbg !38
  call void @llvm.dbg.value(metadata i32 2, metadata !48, metadata !DIExpression()), !dbg !38
  %0 = load i32, i32* %__nv_MAIN__F1L21_1Arg0, align 4, !dbg !49
  store i32 %0, i32* %__gtid___nv_MAIN__F1L21_1__396, align 4, !dbg !49
  br label %L.LB2_389

L.LB2_389:                                        ; preds = %L.entry
  br label %L.LB2_313

L.LB2_313:                                        ; preds = %L.LB2_389
  store i32 1, i32* %.s0001_391, align 4, !dbg !49
  store i32 0, i32* %.s0000_390, align 4, !dbg !50
  store i32 1, i32* %.s0003_393, align 4, !dbg !50
  %1 = load i32, i32* %__gtid___nv_MAIN__F1L21_1__396, align 4, !dbg !50
  %2 = bitcast i32* %.s0002_392 to i64*, !dbg !50
  %3 = bitcast i32* %.s0000_390 to i64*, !dbg !50
  %4 = bitcast i32* %.s0001_391 to i64*, !dbg !50
  %5 = bitcast i32* %.s0003_393 to i64*, !dbg !50
  call void @__kmpc_for_static_init_4(i64* null, i32 %1, i32 34, i64* %2, i64* %3, i64* %4, i64* %5, i32 1, i32 0), !dbg !50
  br label %L.LB2_342

L.LB2_342:                                        ; preds = %L.LB2_313
  %6 = load i32, i32* %.s0000_390, align 4, !dbg !50
  %7 = icmp ne i32 %6, 0, !dbg !50
  br i1 %7, label %L.LB2_343, label %L.LB2_441, !dbg !50

L.LB2_441:                                        ; preds = %L.LB2_342
  br label %L.LB2_343

L.LB2_343:                                        ; preds = %L.LB2_441, %L.LB2_342
  %8 = load i32, i32* %.s0001_391, align 4, !dbg !51
  %9 = icmp ugt i32 1, %8, !dbg !51
  br i1 %9, label %L.LB2_344, label %L.LB2_442, !dbg !51

L.LB2_442:                                        ; preds = %L.LB2_343
  %10 = load i32, i32* %.s0000_390, align 4, !dbg !51
  %11 = icmp ult i32 1, %10, !dbg !51
  br i1 %11, label %L.LB2_344, label %L.LB2_443, !dbg !51

L.LB2_443:                                        ; preds = %L.LB2_442
  %12 = load i32, i32* %__gtid___nv_MAIN__F1L21_1__396, align 4, !dbg !52
  %13 = bitcast %struct__cs_name_* @__cs_name_ to i64*, !dbg !52
  call void @__kmpc_critical(i64* null, i32 %12, i64* %13), !dbg !52
  %14 = load i64, i64* %__nv_MAIN__F1L21_1Arg2, align 8, !dbg !52
  %15 = bitcast %astruct.dt117* %.uplevelArgPack0002_420 to i64*, !dbg !52
  store i64 %14, i64* %15, align 8, !dbg !52
  br label %L.LB2_423, !dbg !52

L.LB2_423:                                        ; preds = %L.LB2_443
  %16 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN_F1L24_2_ to i64*, !dbg !52
  %17 = bitcast %astruct.dt117* %.uplevelArgPack0002_420 to i64*, !dbg !52
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %16, i64* %17), !dbg !52
  %18 = load i32, i32* %__gtid___nv_MAIN__F1L21_1__396, align 4, !dbg !53
  %19 = bitcast %struct__cs_name_* @__cs_name_ to i64*, !dbg !53
  call void @__kmpc_end_critical(i64* null, i32 %18, i64* %19), !dbg !53
  br label %L.LB2_344

L.LB2_344:                                        ; preds = %L.LB2_423, %L.LB2_442, %L.LB2_343
  br label %L.LB2_346

L.LB2_346:                                        ; preds = %L.LB2_344
  br label %L.LB2_325

L.LB2_325:                                        ; preds = %L.LB2_346
  ret void, !dbg !49
}

define internal void @__nv_MAIN_F1L24_2_(i32* %__nv_MAIN_F1L24_2Arg0, i64* %__nv_MAIN_F1L24_2Arg1, i64* %__nv_MAIN_F1L24_2Arg2) #0 !dbg !54 {
L.entry:
  %__gtid___nv_MAIN_F1L24_2__453 = alloca i32, align 4
  %.s0008_448 = alloca i32, align 4
  %.s0009_449 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L24_2Arg0, metadata !55, metadata !DIExpression()), !dbg !56
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L24_2Arg1, metadata !57, metadata !DIExpression()), !dbg !56
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L24_2Arg2, metadata !58, metadata !DIExpression()), !dbg !56
  call void @llvm.dbg.value(metadata i32 1, metadata !59, metadata !DIExpression()), !dbg !56
  call void @llvm.dbg.value(metadata i32 2, metadata !60, metadata !DIExpression()), !dbg !56
  call void @llvm.dbg.value(metadata i32 0, metadata !61, metadata !DIExpression()), !dbg !56
  call void @llvm.dbg.value(metadata i32 1, metadata !62, metadata !DIExpression()), !dbg !56
  call void @llvm.dbg.value(metadata i32 2, metadata !63, metadata !DIExpression()), !dbg !56
  call void @llvm.dbg.value(metadata i32 0, metadata !64, metadata !DIExpression()), !dbg !56
  call void @llvm.dbg.value(metadata i32 1, metadata !65, metadata !DIExpression()), !dbg !56
  call void @llvm.dbg.value(metadata i32 2, metadata !66, metadata !DIExpression()), !dbg !56
  %0 = load i32, i32* %__nv_MAIN_F1L24_2Arg0, align 4, !dbg !67
  store i32 %0, i32* %__gtid___nv_MAIN_F1L24_2__453, align 4, !dbg !67
  br label %L.LB4_447

L.LB4_447:                                        ; preds = %L.entry
  br label %L.LB4_320

L.LB4_320:                                        ; preds = %L.LB4_447
  store i32 -1, i32* %.s0008_448, align 4, !dbg !68
  store i32 0, i32* %.s0009_449, align 4, !dbg !68
  %1 = load i32, i32* %__gtid___nv_MAIN_F1L24_2__453, align 4, !dbg !68
  %2 = call i32 @__kmpc_single(i64* null, i32 %1), !dbg !68
  %3 = icmp eq i32 %2, 0, !dbg !68
  br i1 %3, label %L.LB4_345, label %L.LB4_322, !dbg !68

L.LB4_322:                                        ; preds = %L.LB4_320
  %4 = bitcast i64* %__nv_MAIN_F1L24_2Arg2 to i32**, !dbg !69
  %5 = load i32*, i32** %4, align 8, !dbg !69
  %6 = load i32, i32* %5, align 4, !dbg !69
  %7 = add nsw i32 %6, 1, !dbg !69
  %8 = bitcast i64* %__nv_MAIN_F1L24_2Arg2 to i32**, !dbg !69
  %9 = load i32*, i32** %8, align 8, !dbg !69
  store i32 %7, i32* %9, align 4, !dbg !69
  %10 = load i32, i32* %__gtid___nv_MAIN_F1L24_2__453, align 4, !dbg !70
  store i32 %10, i32* %.s0008_448, align 4, !dbg !70
  store i32 1, i32* %.s0009_449, align 4, !dbg !70
  %11 = load i32, i32* %__gtid___nv_MAIN_F1L24_2__453, align 4, !dbg !70
  call void @__kmpc_end_single(i64* null, i32 %11), !dbg !70
  br label %L.LB4_345

L.LB4_345:                                        ; preds = %L.LB4_322, %L.LB4_320
  br label %L.LB4_323

L.LB4_323:                                        ; preds = %L.LB4_345
  %12 = load i32, i32* %__gtid___nv_MAIN_F1L24_2__453, align 4, !dbg !70
  call void @__kmpc_barrier(i64* null, i32 %12), !dbg !70
  br label %L.LB4_324

L.LB4_324:                                        ; preds = %L.LB4_323
  ret void, !dbg !67
}

declare void @__kmpc_barrier(i64*, i32) #0

declare void @__kmpc_end_single(i64*, i32) #0

declare signext i32 @__kmpc_single(i64*, i32) #0

declare void @__kmpc_end_critical(i64*, i32, i64*) #0

declare void @__kmpc_critical(i64*, i32, i64*) #0

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

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { "no-frame-pointer-elim-non-leaf" }
attributes #1 = { nounwind readnone speculatable }

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !4, globals: !5, imports: !4)
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB139-worksharingcritical-orig-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = !{!6}
!6 = !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!7 = distinct !DIGlobalVariable(name: "__cs_name", scope: !8, type: !14, isLocal: false, isDefinition: true)
!8 = distinct !DICommonBlock(scope: !9, declaration: !7, name: "__cs_name")
!9 = distinct !DISubprogram(name: "__nv_MAIN__F1L21_1", scope: !2, file: !3, line: 21, type: !10, scopeLine: 21, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!10 = !DISubroutineType(types: !11)
!11 = !{null, !12, !13, !13}
!12 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!13 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!14 = !DICompositeType(tag: DW_TAG_array_type, baseType: !15, size: 256, align: 8, elements: !16)
!15 = !DIBasicType(name: "byte", size: 8, align: 8, encoding: DW_ATE_signed)
!16 = !{!17}
!17 = !DISubrange(count: 32)
!18 = distinct !DISubprogram(name: "drb139_worksharingcritical_orig_no", scope: !2, file: !3, line: 14, type: !19, scopeLine: 14, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!19 = !DISubroutineType(cc: DW_CC_program, types: !20)
!20 = !{null}
!21 = !DILocalVariable(name: "omp_sched_static", scope: !18, file: !3, type: !12)
!22 = !DILocation(line: 0, scope: !18)
!23 = !DILocalVariable(name: "omp_sched_dynamic", scope: !18, file: !3, type: !12)
!24 = !DILocalVariable(name: "omp_proc_bind_false", scope: !18, file: !3, type: !12)
!25 = !DILocalVariable(name: "omp_proc_bind_true", scope: !18, file: !3, type: !12)
!26 = !DILocalVariable(name: "omp_proc_bind_master", scope: !18, file: !3, type: !12)
!27 = !DILocalVariable(name: "omp_lock_hint_none", scope: !18, file: !3, type: !12)
!28 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !18, file: !3, type: !12)
!29 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !18, file: !3, type: !12)
!30 = !DILocation(line: 34, column: 1, scope: !18)
!31 = !DILocation(line: 14, column: 1, scope: !18)
!32 = !DILocalVariable(name: "i", scope: !18, file: !3, type: !12)
!33 = !DILocation(line: 19, column: 1, scope: !18)
!34 = !DILocation(line: 21, column: 1, scope: !18)
!35 = !DILocation(line: 32, column: 1, scope: !18)
!36 = !DILocalVariable(scope: !18, file: !3, type: !12, flags: DIFlagArtificial)
!37 = !DILocalVariable(name: "__nv_MAIN__F1L21_1Arg0", arg: 1, scope: !9, file: !3, type: !12)
!38 = !DILocation(line: 0, scope: !9)
!39 = !DILocalVariable(name: "__nv_MAIN__F1L21_1Arg1", arg: 2, scope: !9, file: !3, type: !13)
!40 = !DILocalVariable(name: "__nv_MAIN__F1L21_1Arg2", arg: 3, scope: !9, file: !3, type: !13)
!41 = !DILocalVariable(name: "omp_sched_static", scope: !9, file: !3, type: !12)
!42 = !DILocalVariable(name: "omp_sched_dynamic", scope: !9, file: !3, type: !12)
!43 = !DILocalVariable(name: "omp_proc_bind_false", scope: !9, file: !3, type: !12)
!44 = !DILocalVariable(name: "omp_proc_bind_true", scope: !9, file: !3, type: !12)
!45 = !DILocalVariable(name: "omp_proc_bind_master", scope: !9, file: !3, type: !12)
!46 = !DILocalVariable(name: "omp_lock_hint_none", scope: !9, file: !3, type: !12)
!47 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !9, file: !3, type: !12)
!48 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !9, file: !3, type: !12)
!49 = !DILocation(line: 30, column: 1, scope: !9)
!50 = !DILocation(line: 21, column: 1, scope: !9)
!51 = !DILocation(line: 22, column: 1, scope: !9)
!52 = !DILocation(line: 24, column: 1, scope: !9)
!53 = !DILocation(line: 28, column: 1, scope: !9)
!54 = distinct !DISubprogram(name: "__nv_MAIN_F1L24_2", scope: !2, file: !3, line: 24, type: !10, scopeLine: 24, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!55 = !DILocalVariable(name: "__nv_MAIN_F1L24_2Arg0", arg: 1, scope: !54, file: !3, type: !12)
!56 = !DILocation(line: 0, scope: !54)
!57 = !DILocalVariable(name: "__nv_MAIN_F1L24_2Arg1", arg: 2, scope: !54, file: !3, type: !13)
!58 = !DILocalVariable(name: "__nv_MAIN_F1L24_2Arg2", arg: 3, scope: !54, file: !3, type: !13)
!59 = !DILocalVariable(name: "omp_sched_static", scope: !54, file: !3, type: !12)
!60 = !DILocalVariable(name: "omp_sched_dynamic", scope: !54, file: !3, type: !12)
!61 = !DILocalVariable(name: "omp_proc_bind_false", scope: !54, file: !3, type: !12)
!62 = !DILocalVariable(name: "omp_proc_bind_true", scope: !54, file: !3, type: !12)
!63 = !DILocalVariable(name: "omp_proc_bind_master", scope: !54, file: !3, type: !12)
!64 = !DILocalVariable(name: "omp_lock_hint_none", scope: !54, file: !3, type: !12)
!65 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !54, file: !3, type: !12)
!66 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !54, file: !3, type: !12)
!67 = !DILocation(line: 28, column: 1, scope: !54)
!68 = !DILocation(line: 25, column: 1, scope: !54)
!69 = !DILocation(line: 26, column: 1, scope: !54)
!70 = !DILocation(line: 27, column: 1, scope: !54)

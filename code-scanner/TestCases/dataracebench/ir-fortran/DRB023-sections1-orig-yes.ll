; ModuleID = '/tmp/DRB023-sections1-orig-yes-e920ad.ll'
source_filename = "/tmp/DRB023-sections1-orig-yes-e920ad.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.STATICS1 = type <{ [40 x i8] }>
%astruct.dt60 = type <{ i8* }>

@.STATICS1 = internal global %struct.STATICS1 <{ [40 x i8] c"\FB\FF\FF\FF\02\00\00\00i=\00\00\EB\FF\FF\FF\00\00\00\00\03\00\00\00\00\00\00\00\01\00\00\00\FF\FF\FF\FF\00\00\00\00" }>, align 16
@.C284_MAIN_ = internal constant i64 0
@.C322_MAIN_ = internal constant i32 6
@.C319_MAIN_ = internal constant [54 x i8] c"micro-benchmarks-fortran/DRB023-sections1-orig-yes.f95"
@.C312_MAIN_ = internal constant i32 25
@.C301_MAIN_ = internal constant i32 3
@.C300_MAIN_ = internal constant i32 2
@.C285_MAIN_ = internal constant i32 1
@.C283_MAIN_ = internal constant i32 0
@.C301___nv_MAIN__F1L18_1 = internal constant i32 3
@.C300___nv_MAIN__F1L18_1 = internal constant i32 2
@.C285___nv_MAIN__F1L18_1 = internal constant i32 1
@.C283___nv_MAIN__F1L18_1 = internal constant i32 0

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__350 = alloca i32, align 4
  %i_313 = alloca i32, align 4
  %.uplevelArgPack0001_345 = alloca %astruct.dt60, align 8
  %z__io_324 = alloca i32, align 4
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
  store i32 %0, i32* %__gtid_MAIN__350, align 4, !dbg !20
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !21
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !21
  call void (i8*, ...) %2(i8* %1), !dbg !21
  br label %L.LB1_339

L.LB1_339:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %i_313, metadata !22, metadata !DIExpression()), !dbg !10
  store i32 0, i32* %i_313, align 4, !dbg !23
  %3 = bitcast i32* %i_313 to i8*, !dbg !24
  %4 = bitcast %astruct.dt60* %.uplevelArgPack0001_345 to i8**, !dbg !24
  store i8* %3, i8** %4, align 8, !dbg !24
  br label %L.LB1_348, !dbg !24

L.LB1_348:                                        ; preds = %L.LB1_339
  %5 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L18_1_ to i64*, !dbg !24
  %6 = bitcast %astruct.dt60* %.uplevelArgPack0001_345 to i64*, !dbg !24
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %5, i64* %6), !dbg !24
  call void (...) @_mp_bcs_nest(), !dbg !25
  %7 = bitcast i32* @.C312_MAIN_ to i8*, !dbg !25
  %8 = bitcast [54 x i8]* @.C319_MAIN_ to i8*, !dbg !25
  %9 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !25
  call void (i8*, i8*, i64, ...) %9(i8* %7, i8* %8, i64 54), !dbg !25
  %10 = bitcast i32* @.C322_MAIN_ to i8*, !dbg !25
  %11 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !25
  %12 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !25
  %13 = bitcast %struct.STATICS1* @.STATICS1 to i8*, !dbg !25
  %14 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !25
  %15 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %14(i8* %10, i8* null, i8* %11, i8* %12, i8* %13, i8* null, i64 0), !dbg !25
  call void @llvm.dbg.declare(metadata i32* %z__io_324, metadata !26, metadata !DIExpression()), !dbg !10
  store i32 %15, i32* %z__io_324, align 4, !dbg !25
  %16 = load i32, i32* %i_313, align 4, !dbg !25
  call void @llvm.dbg.value(metadata i32 %16, metadata !22, metadata !DIExpression()), !dbg !10
  %17 = bitcast i32 (...)* @f90io_sc_i_fmt_write to i32 (i32, i32, ...)*, !dbg !25
  %18 = call i32 (i32, i32, ...) %17(i32 %16, i32 25), !dbg !25
  store i32 %18, i32* %z__io_324, align 4, !dbg !25
  %19 = call i32 (...) @f90io_fmtw_end(), !dbg !25
  store i32 %19, i32* %z__io_324, align 4, !dbg !25
  call void (...) @_mp_ecs_nest(), !dbg !25
  ret void, !dbg !20
}

define internal void @__nv_MAIN__F1L18_1_(i32* %__nv_MAIN__F1L18_1Arg0, i64* %__nv_MAIN__F1L18_1Arg1, i64* %__nv_MAIN__F1L18_1Arg2) #0 !dbg !27 {
L.entry:
  %__gtid___nv_MAIN__F1L18_1__386 = alloca i32, align 4
  %.s0001_381 = alloca i32, align 4
  %.s0000_380 = alloca i32, align 4
  %.s0003_383 = alloca i32, align 4
  %.s0002_382 = alloca i32, align 4
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
  store i32 %0, i32* %__gtid___nv_MAIN__F1L18_1__386, align 4, !dbg !45
  br label %L.LB2_379

L.LB2_379:                                        ; preds = %L.entry
  br label %L.LB2_316

L.LB2_316:                                        ; preds = %L.LB2_379
  store i32 2, i32* %.s0001_381, align 4, !dbg !45
  store i32 0, i32* %.s0000_380, align 4, !dbg !46
  store i32 1, i32* %.s0003_383, align 4, !dbg !46
  %1 = load i32, i32* %__gtid___nv_MAIN__F1L18_1__386, align 4, !dbg !46
  %2 = bitcast i32* %.s0002_382 to i64*, !dbg !46
  %3 = bitcast i32* %.s0000_380 to i64*, !dbg !46
  %4 = bitcast i32* %.s0001_381 to i64*, !dbg !46
  %5 = bitcast i32* %.s0003_383 to i64*, !dbg !46
  call void @__kmpc_for_static_init_4(i64* null, i32 %1, i32 34, i64* %2, i64* %3, i64* %4, i64* %5, i32 1, i32 0), !dbg !46
  br label %L.LB2_333

L.LB2_333:                                        ; preds = %L.LB2_316
  %6 = load i32, i32* %.s0000_380, align 4, !dbg !46
  %7 = icmp ne i32 %6, 0, !dbg !46
  br i1 %7, label %L.LB2_334, label %L.LB2_410, !dbg !46

L.LB2_410:                                        ; preds = %L.LB2_333
  br label %L.LB2_334

L.LB2_334:                                        ; preds = %L.LB2_410, %L.LB2_333
  %8 = load i32, i32* %.s0001_381, align 4, !dbg !47
  %9 = icmp ugt i32 1, %8, !dbg !47
  br i1 %9, label %L.LB2_335, label %L.LB2_411, !dbg !47

L.LB2_411:                                        ; preds = %L.LB2_334
  %10 = load i32, i32* %.s0000_380, align 4, !dbg !47
  %11 = icmp ult i32 1, %10, !dbg !47
  br i1 %11, label %L.LB2_335, label %L.LB2_412, !dbg !47

L.LB2_412:                                        ; preds = %L.LB2_411
  %12 = bitcast i64* %__nv_MAIN__F1L18_1Arg2 to i32**, !dbg !48
  %13 = load i32*, i32** %12, align 8, !dbg !48
  store i32 1, i32* %13, align 4, !dbg !48
  br label %L.LB2_335

L.LB2_335:                                        ; preds = %L.LB2_412, %L.LB2_411, %L.LB2_334
  %14 = load i32, i32* %.s0001_381, align 4, !dbg !49
  %15 = icmp ugt i32 2, %14, !dbg !49
  br i1 %15, label %L.LB2_336, label %L.LB2_413, !dbg !49

L.LB2_413:                                        ; preds = %L.LB2_335
  %16 = load i32, i32* %.s0000_380, align 4, !dbg !49
  %17 = icmp ult i32 2, %16, !dbg !49
  br i1 %17, label %L.LB2_336, label %L.LB2_414, !dbg !49

L.LB2_414:                                        ; preds = %L.LB2_413
  %18 = bitcast i64* %__nv_MAIN__F1L18_1Arg2 to i32**, !dbg !50
  %19 = load i32*, i32** %18, align 8, !dbg !50
  store i32 2, i32* %19, align 4, !dbg !50
  br label %L.LB2_336

L.LB2_336:                                        ; preds = %L.LB2_414, %L.LB2_413, %L.LB2_335
  br label %L.LB2_337

L.LB2_337:                                        ; preds = %L.LB2_336
  br label %L.LB2_317

L.LB2_317:                                        ; preds = %L.LB2_337
  ret void, !dbg !45
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

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { "no-frame-pointer-elim-non-leaf" }
attributes #1 = { nounwind readnone speculatable }

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !4, globals: !4, imports: !4)
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB023-sections1-orig-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb023_sections1_orig_yes", scope: !2, file: !3, line: 11, type: !6, scopeLine: 11, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
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
!20 = !DILocation(line: 28, column: 1, scope: !5)
!21 = !DILocation(line: 11, column: 1, scope: !5)
!22 = !DILocalVariable(name: "i", scope: !5, file: !3, type: !9)
!23 = !DILocation(line: 16, column: 1, scope: !5)
!24 = !DILocation(line: 18, column: 1, scope: !5)
!25 = !DILocation(line: 25, column: 1, scope: !5)
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
!45 = !DILocation(line: 23, column: 1, scope: !27)
!46 = !DILocation(line: 18, column: 1, scope: !27)
!47 = !DILocation(line: 19, column: 1, scope: !27)
!48 = !DILocation(line: 20, column: 1, scope: !27)
!49 = !DILocation(line: 21, column: 1, scope: !27)
!50 = !DILocation(line: 22, column: 1, scope: !27)

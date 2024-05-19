; ModuleID = '/tmp/DRB122-taskundeferred-orig-no-748021.ll'
source_filename = "/tmp/DRB122-taskundeferred-orig-no-748021.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.STATICS1 = type <{ [52 x i8] }>
%astruct.dt60 = type <{ i8*, i8* }>

@.STATICS1 = internal global %struct.STATICS1 <{ [52 x i8] c"\FB\FF\FF\FF\05\00\00\00var =\00\00\00\00\00\00\00\03\00\00\00\EB\FF\FF\FF\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\FF\FF\FF\FF\00\00\00\00" }>, align 16
@.C305_MAIN_ = internal constant i32 25
@.C284_MAIN_ = internal constant i64 0
@.C324_MAIN_ = internal constant i32 6
@.C321_MAIN_ = internal constant [58 x i8] c"micro-benchmarks-fortran/DRB122-taskundeferred-orig-no.f95"
@.C306_MAIN_ = internal constant i32 27
@.C312_MAIN_ = internal constant i32 10
@.C285_MAIN_ = internal constant i32 1
@.C283_MAIN_ = internal constant i32 0
@.C312___nv_MAIN__F1L19_1 = internal constant i32 10
@.C285___nv_MAIN__F1L19_1 = internal constant i32 1
@.C283___nv_MAIN__F1L19_1 = internal constant i32 0
@.C285___nv_MAIN_F1L21_2 = internal constant i32 1

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__357 = alloca i32, align 4
  %var_307 = alloca i32, align 4
  %i_308 = alloca i32, align 4
  %.uplevelArgPack0001_349 = alloca %astruct.dt60, align 16
  %z__io_326 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !14, metadata !DIExpression()), !dbg !10
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !15
  store i32 %0, i32* %__gtid_MAIN__357, align 4, !dbg !15
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !16
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !16
  call void (i8*, ...) %2(i8* %1), !dbg !16
  br label %L.LB1_343

L.LB1_343:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %var_307, metadata !17, metadata !DIExpression()), !dbg !10
  store i32 0, i32* %var_307, align 4, !dbg !18
  call void @llvm.dbg.declare(metadata i32* %i_308, metadata !19, metadata !DIExpression()), !dbg !10
  %3 = bitcast i32* %i_308 to i8*, !dbg !20
  %4 = bitcast %astruct.dt60* %.uplevelArgPack0001_349 to i8**, !dbg !20
  store i8* %3, i8** %4, align 8, !dbg !20
  %5 = bitcast i32* %var_307 to i8*, !dbg !20
  %6 = bitcast %astruct.dt60* %.uplevelArgPack0001_349 to i8*, !dbg !20
  %7 = getelementptr i8, i8* %6, i64 8, !dbg !20
  %8 = bitcast i8* %7 to i8**, !dbg !20
  store i8* %5, i8** %8, align 8, !dbg !20
  br label %L.LB1_355, !dbg !20

L.LB1_355:                                        ; preds = %L.LB1_343
  %9 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L19_1_ to i64*, !dbg !20
  %10 = bitcast %astruct.dt60* %.uplevelArgPack0001_349 to i64*, !dbg !20
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %9, i64* %10), !dbg !20
  call void (...) @_mp_bcs_nest(), !dbg !21
  %11 = bitcast i32* @.C306_MAIN_ to i8*, !dbg !21
  %12 = bitcast [58 x i8]* @.C321_MAIN_ to i8*, !dbg !21
  %13 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !21
  call void (i8*, i8*, i64, ...) %13(i8* %11, i8* %12, i64 58), !dbg !21
  %14 = bitcast i32* @.C324_MAIN_ to i8*, !dbg !21
  %15 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !21
  %16 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !21
  %17 = bitcast %struct.STATICS1* @.STATICS1 to i8*, !dbg !21
  %18 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !21
  %19 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %18(i8* %14, i8* null, i8* %15, i8* %16, i8* %17, i8* null, i64 0), !dbg !21
  call void @llvm.dbg.declare(metadata i32* %z__io_326, metadata !22, metadata !DIExpression()), !dbg !10
  store i32 %19, i32* %z__io_326, align 4, !dbg !21
  %20 = load i32, i32* %var_307, align 4, !dbg !21
  call void @llvm.dbg.value(metadata i32 %20, metadata !17, metadata !DIExpression()), !dbg !10
  %21 = bitcast i32 (...)* @f90io_sc_i_fmt_write to i32 (i32, i32, ...)*, !dbg !21
  %22 = call i32 (i32, i32, ...) %21(i32 %20, i32 25), !dbg !21
  store i32 %22, i32* %z__io_326, align 4, !dbg !21
  %23 = call i32 (...) @f90io_fmtw_end(), !dbg !21
  store i32 %23, i32* %z__io_326, align 4, !dbg !21
  call void (...) @_mp_ecs_nest(), !dbg !21
  ret void, !dbg !15
}

define internal void @__nv_MAIN__F1L19_1_(i32* %__nv_MAIN__F1L19_1Arg0, i64* %__nv_MAIN__F1L19_1Arg1, i64* %__nv_MAIN__F1L19_1Arg2) #0 !dbg !23 {
L.entry:
  %__gtid___nv_MAIN__F1L19_1__393 = alloca i32, align 4
  %.s0001_388 = alloca i32, align 4
  %.s0000_387 = alloca i32, align 4
  %.s0003_390 = alloca i32, align 4
  %.s0002_389 = alloca i32, align 4
  %.dY0001p_339 = alloca i32, align 4
  %i_313 = alloca i32, align 4
  %.s0004_411 = alloca i32, align 4
  %.z0345_410 = alloca i8*, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L19_1Arg0, metadata !27, metadata !DIExpression()), !dbg !28
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L19_1Arg1, metadata !29, metadata !DIExpression()), !dbg !28
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L19_1Arg2, metadata !30, metadata !DIExpression()), !dbg !28
  call void @llvm.dbg.value(metadata i32 1, metadata !31, metadata !DIExpression()), !dbg !28
  call void @llvm.dbg.value(metadata i32 0, metadata !32, metadata !DIExpression()), !dbg !28
  call void @llvm.dbg.value(metadata i32 1, metadata !33, metadata !DIExpression()), !dbg !28
  call void @llvm.dbg.value(metadata i32 0, metadata !34, metadata !DIExpression()), !dbg !28
  call void @llvm.dbg.value(metadata i32 1, metadata !35, metadata !DIExpression()), !dbg !28
  %0 = load i32, i32* %__nv_MAIN__F1L19_1Arg0, align 4, !dbg !36
  store i32 %0, i32* %__gtid___nv_MAIN__F1L19_1__393, align 4, !dbg !36
  br label %L.LB2_386

L.LB2_386:                                        ; preds = %L.entry
  br label %L.LB2_311

L.LB2_311:                                        ; preds = %L.LB2_386
  store i32 0, i32* %.s0001_388, align 4, !dbg !36
  store i32 0, i32* %.s0000_387, align 4, !dbg !37
  store i32 1, i32* %.s0003_390, align 4, !dbg !37
  %1 = load i32, i32* %__gtid___nv_MAIN__F1L19_1__393, align 4, !dbg !37
  %2 = bitcast i32* %.s0002_389 to i64*, !dbg !37
  %3 = bitcast i32* %.s0000_387 to i64*, !dbg !37
  %4 = bitcast i32* %.s0001_388 to i64*, !dbg !37
  %5 = bitcast i32* %.s0003_390 to i64*, !dbg !37
  call void @__kmpc_for_static_init_4(i64* null, i32 %1, i32 34, i64* %2, i64* %3, i64* %4, i64* %5, i32 1, i32 0), !dbg !37
  br label %L.LB2_335

L.LB2_335:                                        ; preds = %L.LB2_311
  %6 = load i32, i32* %.s0000_387, align 4, !dbg !37
  %7 = icmp ne i32 %6, 0, !dbg !37
  br i1 %7, label %L.LB2_336, label %L.LB2_453, !dbg !37

L.LB2_453:                                        ; preds = %L.LB2_335
  store i32 10, i32* %.dY0001p_339, align 4, !dbg !38
  call void @llvm.dbg.declare(metadata i32* %i_313, metadata !39, metadata !DIExpression()), !dbg !36
  store i32 1, i32* %i_313, align 4, !dbg !38
  br label %L.LB2_337

L.LB2_337:                                        ; preds = %L.LB2_340, %L.LB2_453
  store i32 1, i32* %.s0004_411, align 4, !dbg !40
  %8 = load i32, i32* %__gtid___nv_MAIN__F1L19_1__393, align 4, !dbg !41
  %9 = load i32, i32* %.s0004_411, align 4, !dbg !41
  %10 = bitcast void (i32, i64*)* @__nv_MAIN_F1L21_2_ to i64*, !dbg !41
  %11 = call i8* @__kmpc_omp_task_alloc(i64* null, i32 %8, i32 %9, i32 40, i32 16, i64* %10), !dbg !41
  store i8* %11, i8** %.z0345_410, align 8, !dbg !41
  %12 = load i64, i64* %__nv_MAIN__F1L19_1Arg2, align 8, !dbg !41
  %13 = load i8*, i8** %.z0345_410, align 8, !dbg !41
  %14 = bitcast i8* %13 to i64**, !dbg !41
  %15 = load i64*, i64** %14, align 8, !dbg !41
  store i64 %12, i64* %15, align 8, !dbg !41
  %16 = bitcast i64* %__nv_MAIN__F1L19_1Arg2 to i8*, !dbg !36
  %17 = getelementptr i8, i8* %16, i64 8, !dbg !36
  %18 = bitcast i8* %17 to i64*, !dbg !36
  %19 = load i64, i64* %18, align 8, !dbg !36
  %20 = bitcast i64* %15 to i8*, !dbg !36
  %21 = getelementptr i8, i8* %20, i64 8, !dbg !36
  %22 = bitcast i8* %21 to i64*, !dbg !36
  store i64 %19, i64* %22, align 8, !dbg !36
  %23 = load i32, i32* %__gtid___nv_MAIN__F1L19_1__393, align 4, !dbg !41
  %24 = load i8*, i8** %.z0345_410, align 8, !dbg !41
  %25 = bitcast i8* %24 to i64*, !dbg !41
  call void @__kmpc_omp_task_begin_if0(i64* null, i32 %23, i64* %25), !dbg !41
  %26 = load i32, i32* %__gtid___nv_MAIN__F1L19_1__393, align 4, !dbg !41
  %27 = load i8*, i8** %.z0345_410, align 8, !dbg !41
  %28 = bitcast i8* %27 to i64*, !dbg !41
  call void @__nv_MAIN_F1L21_2_(i32 %26, i64* %28), !dbg !41
  %29 = load i32, i32* %__gtid___nv_MAIN__F1L19_1__393, align 4, !dbg !41
  %30 = load i8*, i8** %.z0345_410, align 8, !dbg !41
  %31 = bitcast i8* %30 to i64*, !dbg !41
  call void @__kmpc_omp_task_complete_if0(i64* null, i32 %29, i64* %31), !dbg !41
  br label %L.LB2_340, !dbg !41

L.LB2_340:                                        ; preds = %L.LB2_337
  %32 = load i32, i32* %i_313, align 4, !dbg !42
  call void @llvm.dbg.value(metadata i32 %32, metadata !39, metadata !DIExpression()), !dbg !36
  %33 = add nsw i32 %32, 1, !dbg !42
  store i32 %33, i32* %i_313, align 4, !dbg !42
  %34 = load i32, i32* %.dY0001p_339, align 4, !dbg !42
  %35 = sub nsw i32 %34, 1, !dbg !42
  store i32 %35, i32* %.dY0001p_339, align 4, !dbg !42
  %36 = load i32, i32* %.dY0001p_339, align 4, !dbg !42
  %37 = icmp sgt i32 %36, 0, !dbg !42
  br i1 %37, label %L.LB2_337, label %L.LB2_454, !dbg !42

L.LB2_454:                                        ; preds = %L.LB2_340
  br label %L.LB2_336

L.LB2_336:                                        ; preds = %L.LB2_454, %L.LB2_335
  br label %L.LB2_341

L.LB2_341:                                        ; preds = %L.LB2_336
  br label %L.LB2_319

L.LB2_319:                                        ; preds = %L.LB2_341
  ret void, !dbg !36
}

define internal void @__nv_MAIN_F1L21_2_(i32 %__nv_MAIN_F1L21_2Arg0.arg, i64* %__nv_MAIN_F1L21_2Arg1) #0 !dbg !43 {
L.entry:
  %__nv_MAIN_F1L21_2Arg0.addr = alloca i32, align 4
  %.S0000_456 = alloca i8*, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L21_2Arg0.addr, metadata !46, metadata !DIExpression()), !dbg !47
  store i32 %__nv_MAIN_F1L21_2Arg0.arg, i32* %__nv_MAIN_F1L21_2Arg0.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L21_2Arg0.addr, metadata !48, metadata !DIExpression()), !dbg !47
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L21_2Arg1, metadata !49, metadata !DIExpression()), !dbg !47
  call void @llvm.dbg.value(metadata i32 1, metadata !50, metadata !DIExpression()), !dbg !47
  call void @llvm.dbg.value(metadata i32 0, metadata !51, metadata !DIExpression()), !dbg !47
  call void @llvm.dbg.value(metadata i32 1, metadata !52, metadata !DIExpression()), !dbg !47
  call void @llvm.dbg.value(metadata i32 0, metadata !53, metadata !DIExpression()), !dbg !47
  call void @llvm.dbg.value(metadata i32 1, metadata !54, metadata !DIExpression()), !dbg !47
  %0 = bitcast i64* %__nv_MAIN_F1L21_2Arg1 to i8**, !dbg !55
  %1 = load i8*, i8** %0, align 8, !dbg !55
  store i8* %1, i8** %.S0000_456, align 8, !dbg !55
  br label %L.LB4_460

L.LB4_460:                                        ; preds = %L.entry
  br label %L.LB4_317

L.LB4_317:                                        ; preds = %L.LB4_460
  %2 = load i8*, i8** %.S0000_456, align 8, !dbg !56
  %3 = getelementptr i8, i8* %2, i64 8, !dbg !56
  %4 = bitcast i8* %3 to i32**, !dbg !56
  %5 = load i32*, i32** %4, align 8, !dbg !56
  %6 = load i32, i32* %5, align 4, !dbg !56
  %7 = add nsw i32 %6, 1, !dbg !56
  %8 = load i8*, i8** %.S0000_456, align 8, !dbg !56
  %9 = getelementptr i8, i8* %8, i64 8, !dbg !56
  %10 = bitcast i8* %9 to i32**, !dbg !56
  %11 = load i32*, i32** %10, align 8, !dbg !56
  store i32 %7, i32* %11, align 4, !dbg !56
  br label %L.LB4_318

L.LB4_318:                                        ; preds = %L.LB4_317
  ret void, !dbg !57
}

declare void @__kmpc_omp_task(i64*, i32, i64*) #0

declare void @__kmpc_omp_task_complete_if0(i64*, i32, i64*) #0

declare void @__kmpc_omp_task_begin_if0(i64*, i32, i64*) #0

declare i8* @__kmpc_omp_task_alloc(i64*, i32, i32, i32, i32, i64*) #0

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
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB122-taskundeferred-orig-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb122_taskundeferred_orig_no", scope: !2, file: !3, line: 12, type: !6, scopeLine: 12, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!6 = !DISubroutineType(cc: DW_CC_program, types: !7)
!7 = !{null}
!8 = !DILocalVariable(name: "omp_sched_static", scope: !5, file: !3, type: !9)
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 0, scope: !5)
!11 = !DILocalVariable(name: "omp_proc_bind_false", scope: !5, file: !3, type: !9)
!12 = !DILocalVariable(name: "omp_proc_bind_true", scope: !5, file: !3, type: !9)
!13 = !DILocalVariable(name: "omp_lock_hint_none", scope: !5, file: !3, type: !9)
!14 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !5, file: !3, type: !9)
!15 = !DILocation(line: 29, column: 1, scope: !5)
!16 = !DILocation(line: 12, column: 1, scope: !5)
!17 = !DILocalVariable(name: "var", scope: !5, file: !3, type: !9)
!18 = !DILocation(line: 17, column: 1, scope: !5)
!19 = !DILocalVariable(name: "i", scope: !5, file: !3, type: !9)
!20 = !DILocation(line: 19, column: 1, scope: !5)
!21 = !DILocation(line: 27, column: 1, scope: !5)
!22 = !DILocalVariable(scope: !5, file: !3, type: !9, flags: DIFlagArtificial)
!23 = distinct !DISubprogram(name: "__nv_MAIN__F1L19_1", scope: !2, file: !3, line: 19, type: !24, scopeLine: 19, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!24 = !DISubroutineType(types: !25)
!25 = !{null, !9, !26, !26}
!26 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!27 = !DILocalVariable(name: "__nv_MAIN__F1L19_1Arg0", arg: 1, scope: !23, file: !3, type: !9)
!28 = !DILocation(line: 0, scope: !23)
!29 = !DILocalVariable(name: "__nv_MAIN__F1L19_1Arg1", arg: 2, scope: !23, file: !3, type: !26)
!30 = !DILocalVariable(name: "__nv_MAIN__F1L19_1Arg2", arg: 3, scope: !23, file: !3, type: !26)
!31 = !DILocalVariable(name: "omp_sched_static", scope: !23, file: !3, type: !9)
!32 = !DILocalVariable(name: "omp_proc_bind_false", scope: !23, file: !3, type: !9)
!33 = !DILocalVariable(name: "omp_proc_bind_true", scope: !23, file: !3, type: !9)
!34 = !DILocalVariable(name: "omp_lock_hint_none", scope: !23, file: !3, type: !9)
!35 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !23, file: !3, type: !9)
!36 = !DILocation(line: 25, column: 1, scope: !23)
!37 = !DILocation(line: 19, column: 1, scope: !23)
!38 = !DILocation(line: 20, column: 1, scope: !23)
!39 = !DILocalVariable(name: "i", scope: !23, file: !3, type: !9)
!40 = !DILocation(line: 21, column: 1, scope: !23)
!41 = !DILocation(line: 23, column: 1, scope: !23)
!42 = !DILocation(line: 24, column: 1, scope: !23)
!43 = distinct !DISubprogram(name: "__nv_MAIN_F1L21_2", scope: !2, file: !3, line: 21, type: !44, scopeLine: 21, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!44 = !DISubroutineType(types: !45)
!45 = !{null, !9, !26}
!46 = !DILocalVariable(name: "__nv_MAIN_F1L21_2Arg0", scope: !43, file: !3, type: !9)
!47 = !DILocation(line: 0, scope: !43)
!48 = !DILocalVariable(name: "__nv_MAIN_F1L21_2Arg0", arg: 1, scope: !43, file: !3, type: !9)
!49 = !DILocalVariable(name: "__nv_MAIN_F1L21_2Arg1", arg: 2, scope: !43, file: !3, type: !26)
!50 = !DILocalVariable(name: "omp_sched_static", scope: !43, file: !3, type: !9)
!51 = !DILocalVariable(name: "omp_proc_bind_false", scope: !43, file: !3, type: !9)
!52 = !DILocalVariable(name: "omp_proc_bind_true", scope: !43, file: !3, type: !9)
!53 = !DILocalVariable(name: "omp_lock_hint_none", scope: !43, file: !3, type: !9)
!54 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !43, file: !3, type: !9)
!55 = !DILocation(line: 21, column: 1, scope: !43)
!56 = !DILocation(line: 22, column: 1, scope: !43)
!57 = !DILocation(line: 23, column: 1, scope: !43)

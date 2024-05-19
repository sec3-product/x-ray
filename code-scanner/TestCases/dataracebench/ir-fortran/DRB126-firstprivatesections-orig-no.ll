; ModuleID = '/tmp/DRB126-firstprivatesections-orig-no-f15aac.ll'
source_filename = "/tmp/DRB126-firstprivatesections-orig-no-f15aac.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.STATICS1 = type <{ [124 x i8] }>
%astruct.dt66 = type <{ i8* }>

@.STATICS1 = internal global %struct.STATICS1 <{ [124 x i8] c"\FB\FF\FF\FF\0F\00\00\00section_count =\00\00\00\00\00\03\00\00\00\EB\FF\FF\FF\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\FF\FF\FF\FF\00\00\00\00\00\00\00\00\FB\FF\FF\FF\0F\00\00\00section_count =\00\00\00\00\00\03\00\00\00\EB\FF\FF\FF\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\FF\FF\FF\FF\00\00\00\00" }>, align 16
@.C300_MAIN_ = internal constant i32 4
@.C351_MAIN_ = internal constant i32 33
@.C302_MAIN_ = internal constant i32 3
@.C323_MAIN_ = internal constant i32 25
@.C284_MAIN_ = internal constant i64 0
@.C343_MAIN_ = internal constant i32 6
@.C340_MAIN_ = internal constant [64 x i8] c"micro-benchmarks-fortran/DRB126-firstprivatesections-orig-no.f95"
@.C324_MAIN_ = internal constant i32 28
@.C301_MAIN_ = internal constant i32 2
@.C285_MAIN_ = internal constant i32 1
@.C283_MAIN_ = internal constant i32 0
@.C300___nv_MAIN__F1L24_1 = internal constant i32 4
@.C351___nv_MAIN__F1L24_1 = internal constant i32 33
@.C302___nv_MAIN__F1L24_1 = internal constant i32 3
@.C323___nv_MAIN__F1L24_1 = internal constant i32 25
@.C284___nv_MAIN__F1L24_1 = internal constant i64 0
@.C343___nv_MAIN__F1L24_1 = internal constant i32 6
@.C340___nv_MAIN__F1L24_1 = internal constant [64 x i8] c"micro-benchmarks-fortran/DRB126-firstprivatesections-orig-no.f95"
@.C324___nv_MAIN__F1L24_1 = internal constant i32 28
@.C301___nv_MAIN__F1L24_1 = internal constant i32 2
@.C285___nv_MAIN__F1L24_1 = internal constant i32 1
@.C283___nv_MAIN__F1L24_1 = internal constant i32 0

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__377 = alloca i32, align 4
  %section_count_331 = alloca i32, align 4
  %.uplevelArgPack0001_372 = alloca %astruct.dt66, align 8
  call void @llvm.dbg.value(metadata i32 4, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 4, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 4, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 4, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 4, metadata !14, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 4, metadata !15, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 4, metadata !16, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !17, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 2, metadata !18, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 3, metadata !19, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 4, metadata !20, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !21, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !22, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 2, metadata !23, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 3, metadata !24, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 4, metadata !25, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !26, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !27, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 2, metadata !28, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 4, metadata !29, metadata !DIExpression()), !dbg !10
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !30
  store i32 %0, i32* %__gtid_MAIN__377, align 4, !dbg !30
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !31
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !31
  call void (i8*, ...) %2(i8* %1), !dbg !31
  br label %L.LB1_365

L.LB1_365:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %section_count_331, metadata !32, metadata !DIExpression()), !dbg !10
  store i32 0, i32* %section_count_331, align 4, !dbg !33
  %3 = bitcast i32* @.C283_MAIN_ to i64*, !dbg !34
  call void @omp_lib_osd4_(i64* %3), !dbg !34
  %4 = bitcast i32* @.C285_MAIN_ to i64*, !dbg !35
  call void @omp_lib_osnt4_(i64* %4), !dbg !35
  %5 = bitcast i32* %section_count_331 to i8*, !dbg !36
  %6 = bitcast %astruct.dt66* %.uplevelArgPack0001_372 to i8**, !dbg !36
  store i8* %5, i8** %6, align 8, !dbg !36
  br label %L.LB1_375, !dbg !36

L.LB1_375:                                        ; preds = %L.LB1_365
  %7 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L24_1_ to i64*, !dbg !36
  %8 = bitcast %astruct.dt66* %.uplevelArgPack0001_372 to i64*, !dbg !36
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %7, i64* %8), !dbg !36
  ret void, !dbg !30
}

define internal void @__nv_MAIN__F1L24_1_(i32* %__nv_MAIN__F1L24_1Arg0, i64* %__nv_MAIN__F1L24_1Arg1, i64* %__nv_MAIN__F1L24_1Arg2) #0 !dbg !37 {
L.entry:
  %__gtid___nv_MAIN__F1L24_1__408 = alloca i32, align 4
  %.s0001_400 = alloca i32, align 4
  %section_count_338 = alloca i32, align 4
  %.s0000_399 = alloca i32, align 4
  %.s0003_402 = alloca i32, align 4
  %.s0002_401 = alloca i32, align 4
  %z__io_345 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L24_1Arg0, metadata !41, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L24_1Arg1, metadata !43, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L24_1Arg2, metadata !44, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.value(metadata i32 4, metadata !45, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.value(metadata i32 4, metadata !46, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.value(metadata i32 4, metadata !47, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.value(metadata i32 4, metadata !48, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.value(metadata i32 4, metadata !49, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.value(metadata i32 4, metadata !50, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.value(metadata i32 4, metadata !51, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.value(metadata i32 1, metadata !52, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.value(metadata i32 2, metadata !53, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.value(metadata i32 3, metadata !54, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.value(metadata i32 4, metadata !55, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.value(metadata i32 0, metadata !56, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.value(metadata i32 1, metadata !57, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.value(metadata i32 2, metadata !58, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.value(metadata i32 3, metadata !59, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.value(metadata i32 4, metadata !60, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.value(metadata i32 0, metadata !61, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.value(metadata i32 1, metadata !62, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.value(metadata i32 2, metadata !63, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.value(metadata i32 4, metadata !64, metadata !DIExpression()), !dbg !42
  %0 = load i32, i32* %__nv_MAIN__F1L24_1Arg0, align 4, !dbg !65
  store i32 %0, i32* %__gtid___nv_MAIN__F1L24_1__408, align 4, !dbg !65
  br label %L.LB2_398

L.LB2_398:                                        ; preds = %L.entry
  br label %L.LB2_335

L.LB2_335:                                        ; preds = %L.LB2_398
  br label %L.LB2_337

L.LB2_337:                                        ; preds = %L.LB2_335
  store i32 3, i32* %.s0001_400, align 4, !dbg !66
  %1 = bitcast i64* %__nv_MAIN__F1L24_1Arg2 to i32**, !dbg !67
  %2 = load i32*, i32** %1, align 8, !dbg !67
  %3 = load i32, i32* %2, align 4, !dbg !67
  call void @llvm.dbg.declare(metadata i32* %section_count_338, metadata !68, metadata !DIExpression()), !dbg !65
  store i32 %3, i32* %section_count_338, align 4, !dbg !67
  store i32 0, i32* %.s0000_399, align 4, !dbg !67
  store i32 1, i32* %.s0003_402, align 4, !dbg !67
  %4 = load i32, i32* %__gtid___nv_MAIN__F1L24_1__408, align 4, !dbg !67
  %5 = bitcast i32* %.s0002_401 to i64*, !dbg !67
  %6 = bitcast i32* %.s0000_399 to i64*, !dbg !67
  %7 = bitcast i32* %.s0001_400 to i64*, !dbg !67
  %8 = bitcast i32* %.s0003_402 to i64*, !dbg !67
  call void @__kmpc_for_static_init_4(i64* null, i32 %4, i32 34, i64* %5, i64* %6, i64* %7, i64* %8, i32 1, i32 0), !dbg !67
  br label %L.LB2_358

L.LB2_358:                                        ; preds = %L.LB2_337
  %9 = load i32, i32* %.s0000_399, align 4, !dbg !67
  %10 = icmp ne i32 %9, 0, !dbg !67
  br i1 %10, label %L.LB2_359, label %L.LB2_443, !dbg !67

L.LB2_443:                                        ; preds = %L.LB2_358
  br label %L.LB2_359

L.LB2_359:                                        ; preds = %L.LB2_443, %L.LB2_358
  %11 = load i32, i32* %.s0001_400, align 4, !dbg !67
  %12 = icmp ugt i32 1, %11, !dbg !67
  br i1 %12, label %L.LB2_360, label %L.LB2_444, !dbg !67

L.LB2_444:                                        ; preds = %L.LB2_359
  %13 = load i32, i32* %.s0000_399, align 4, !dbg !67
  %14 = icmp ult i32 1, %13, !dbg !67
  br i1 %14, label %L.LB2_360, label %L.LB2_445, !dbg !67

L.LB2_445:                                        ; preds = %L.LB2_444
  br label %L.LB2_360

L.LB2_360:                                        ; preds = %L.LB2_445, %L.LB2_444, %L.LB2_359
  %15 = load i32, i32* %.s0001_400, align 4, !dbg !69
  %16 = icmp ugt i32 2, %15, !dbg !69
  br i1 %16, label %L.LB2_361, label %L.LB2_446, !dbg !69

L.LB2_446:                                        ; preds = %L.LB2_360
  %17 = load i32, i32* %.s0000_399, align 4, !dbg !69
  %18 = icmp ult i32 2, %17, !dbg !69
  br i1 %18, label %L.LB2_361, label %L.LB2_447, !dbg !69

L.LB2_447:                                        ; preds = %L.LB2_446
  %19 = load i32, i32* %section_count_338, align 4, !dbg !70
  call void @llvm.dbg.value(metadata i32 %19, metadata !68, metadata !DIExpression()), !dbg !65
  %20 = add nsw i32 %19, 1, !dbg !70
  store i32 %20, i32* %section_count_338, align 4, !dbg !70
  call void (...) @_mp_bcs_nest(), !dbg !71
  %21 = bitcast i32* @.C324___nv_MAIN__F1L24_1 to i8*, !dbg !71
  %22 = bitcast [64 x i8]* @.C340___nv_MAIN__F1L24_1 to i8*, !dbg !71
  %23 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !71
  call void (i8*, i8*, i64, ...) %23(i8* %21, i8* %22, i64 64), !dbg !71
  %24 = bitcast i32* @.C343___nv_MAIN__F1L24_1 to i8*, !dbg !71
  %25 = bitcast i32* @.C283___nv_MAIN__F1L24_1 to i8*, !dbg !71
  %26 = bitcast i32* @.C283___nv_MAIN__F1L24_1 to i8*, !dbg !71
  %27 = bitcast %struct.STATICS1* @.STATICS1 to i8*, !dbg !71
  %28 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !71
  %29 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %28(i8* %24, i8* null, i8* %25, i8* %26, i8* %27, i8* null, i64 0), !dbg !71
  call void @llvm.dbg.declare(metadata i32* %z__io_345, metadata !72, metadata !DIExpression()), !dbg !42
  store i32 %29, i32* %z__io_345, align 4, !dbg !71
  %30 = load i32, i32* %section_count_338, align 4, !dbg !71
  call void @llvm.dbg.value(metadata i32 %30, metadata !68, metadata !DIExpression()), !dbg !65
  %31 = bitcast i32 (...)* @f90io_sc_i_fmt_write to i32 (i32, i32, ...)*, !dbg !71
  %32 = call i32 (i32, i32, ...) %31(i32 %30, i32 25), !dbg !71
  store i32 %32, i32* %z__io_345, align 4, !dbg !71
  %33 = call i32 (...) @f90io_fmtw_end(), !dbg !71
  store i32 %33, i32* %z__io_345, align 4, !dbg !71
  call void (...) @_mp_ecs_nest(), !dbg !71
  br label %L.LB2_361

L.LB2_361:                                        ; preds = %L.LB2_447, %L.LB2_446, %L.LB2_360
  %34 = load i32, i32* %.s0001_400, align 4, !dbg !73
  %35 = icmp ugt i32 3, %34, !dbg !73
  br i1 %35, label %L.LB2_362, label %L.LB2_448, !dbg !73

L.LB2_448:                                        ; preds = %L.LB2_361
  %36 = load i32, i32* %.s0000_399, align 4, !dbg !73
  %37 = icmp ult i32 3, %36, !dbg !73
  br i1 %37, label %L.LB2_362, label %L.LB2_449, !dbg !73

L.LB2_449:                                        ; preds = %L.LB2_448
  %38 = load i32, i32* %section_count_338, align 4, !dbg !74
  call void @llvm.dbg.value(metadata i32 %38, metadata !68, metadata !DIExpression()), !dbg !65
  %39 = add nsw i32 %38, 1, !dbg !74
  store i32 %39, i32* %section_count_338, align 4, !dbg !74
  call void (...) @_mp_bcs_nest(), !dbg !75
  %40 = bitcast i32* @.C351___nv_MAIN__F1L24_1 to i8*, !dbg !75
  %41 = bitcast [64 x i8]* @.C340___nv_MAIN__F1L24_1 to i8*, !dbg !75
  %42 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !75
  call void (i8*, i8*, i64, ...) %42(i8* %40, i8* %41, i64 64), !dbg !75
  %43 = bitcast i32* @.C343___nv_MAIN__F1L24_1 to i8*, !dbg !75
  %44 = bitcast i32* @.C283___nv_MAIN__F1L24_1 to i8*, !dbg !75
  %45 = bitcast i32* @.C283___nv_MAIN__F1L24_1 to i8*, !dbg !75
  %46 = bitcast %struct.STATICS1* @.STATICS1 to i8*, !dbg !75
  %47 = getelementptr i8, i8* %46, i64 64, !dbg !75
  %48 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !75
  %49 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %48(i8* %43, i8* null, i8* %44, i8* %45, i8* %47, i8* null, i64 0), !dbg !75
  store i32 %49, i32* %z__io_345, align 4, !dbg !75
  %50 = load i32, i32* %section_count_338, align 4, !dbg !75
  call void @llvm.dbg.value(metadata i32 %50, metadata !68, metadata !DIExpression()), !dbg !65
  %51 = bitcast i32 (...)* @f90io_sc_i_fmt_write to i32 (i32, i32, ...)*, !dbg !75
  %52 = call i32 (i32, i32, ...) %51(i32 %50, i32 25), !dbg !75
  store i32 %52, i32* %z__io_345, align 4, !dbg !75
  %53 = call i32 (...) @f90io_fmtw_end(), !dbg !75
  store i32 %53, i32* %z__io_345, align 4, !dbg !75
  call void (...) @_mp_ecs_nest(), !dbg !75
  br label %L.LB2_362

L.LB2_362:                                        ; preds = %L.LB2_449, %L.LB2_448, %L.LB2_361
  br label %L.LB2_363

L.LB2_363:                                        ; preds = %L.LB2_362
  %54 = load i32, i32* %__gtid___nv_MAIN__F1L24_1__408, align 4, !dbg !66
  call void @__kmpc_barrier(i64* null, i32 %54), !dbg !66
  br label %L.LB2_353

L.LB2_353:                                        ; preds = %L.LB2_363
  br label %L.LB2_354

L.LB2_354:                                        ; preds = %L.LB2_353
  ret void, !dbg !65
}

declare void @_mp_ecs_nest(...) #0

declare signext i32 @f90io_fmtw_end(...) #0

declare signext i32 @f90io_sc_i_fmt_write(...) #0

declare signext i32 @f90io_fmtw_inita(...) #0

declare void @f90io_src_info03a(...) #0

declare void @_mp_bcs_nest(...) #0

declare void @__kmpc_barrier(i64*, i32) #0

declare void @__kmpc_for_static_init_4(i64*, i32, i32, i64*, i64*, i64*, i64*, i32 zeroext, i32 zeroext) #0

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare void @fort_init(...) #0

declare signext i32 @__kmpc_global_thread_num(i64*) #0

declare void @__kmpc_fork_call(i64*, i32, i64*, i64*, ...) #0

declare void @__kmpc_end_serialized_parallel(i64*, i32) #0

declare void @__kmpc_serialized_parallel(i64*, i32) #0

declare void @omp_lib_osnt4_(i64*) #0

declare void @omp_lib_osd4_(i64*) #0

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { "no-frame-pointer-elim-non-leaf" }
attributes #1 = { nounwind readnone speculatable }

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !4, globals: !4, imports: !4)
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB126-firstprivatesections-orig-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb126_firstprivatesections_orig_no", scope: !2, file: !3, line: 13, type: !6, scopeLine: 13, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!6 = !DISubroutineType(cc: DW_CC_program, types: !7)
!7 = !{null}
!8 = !DILocalVariable(name: "omp_integer_kind", scope: !5, file: !3, type: !9)
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 0, scope: !5)
!11 = !DILocalVariable(name: "omp_logical_kind", scope: !5, file: !3, type: !9)
!12 = !DILocalVariable(name: "omp_lock_kind", scope: !5, file: !3, type: !9)
!13 = !DILocalVariable(name: "omp_sched_kind", scope: !5, file: !3, type: !9)
!14 = !DILocalVariable(name: "omp_real_kind", scope: !5, file: !3, type: !9)
!15 = !DILocalVariable(name: "omp_proc_bind_kind", scope: !5, file: !3, type: !9)
!16 = !DILocalVariable(name: "omp_lock_hint_kind", scope: !5, file: !3, type: !9)
!17 = !DILocalVariable(name: "omp_sched_static", scope: !5, file: !3, type: !9)
!18 = !DILocalVariable(name: "omp_sched_dynamic", scope: !5, file: !3, type: !9)
!19 = !DILocalVariable(name: "omp_sched_guided", scope: !5, file: !3, type: !9)
!20 = !DILocalVariable(name: "omp_sched_auto", scope: !5, file: !3, type: !9)
!21 = !DILocalVariable(name: "omp_proc_bind_false", scope: !5, file: !3, type: !9)
!22 = !DILocalVariable(name: "omp_proc_bind_true", scope: !5, file: !3, type: !9)
!23 = !DILocalVariable(name: "omp_proc_bind_master", scope: !5, file: !3, type: !9)
!24 = !DILocalVariable(name: "omp_proc_bind_close", scope: !5, file: !3, type: !9)
!25 = !DILocalVariable(name: "omp_proc_bind_spread", scope: !5, file: !3, type: !9)
!26 = !DILocalVariable(name: "omp_lock_hint_none", scope: !5, file: !3, type: !9)
!27 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !5, file: !3, type: !9)
!28 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !5, file: !3, type: !9)
!29 = !DILocalVariable(name: "omp_lock_hint_nonspeculative", scope: !5, file: !3, type: !9)
!30 = !DILocation(line: 37, column: 1, scope: !5)
!31 = !DILocation(line: 13, column: 1, scope: !5)
!32 = !DILocalVariable(name: "section_count", scope: !5, file: !3, type: !9)
!33 = !DILocation(line: 19, column: 1, scope: !5)
!34 = !DILocation(line: 21, column: 1, scope: !5)
!35 = !DILocation(line: 22, column: 1, scope: !5)
!36 = !DILocation(line: 24, column: 1, scope: !5)
!37 = distinct !DISubprogram(name: "__nv_MAIN__F1L24_1", scope: !2, file: !3, line: 24, type: !38, scopeLine: 24, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!38 = !DISubroutineType(types: !39)
!39 = !{null, !9, !40, !40}
!40 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!41 = !DILocalVariable(name: "__nv_MAIN__F1L24_1Arg0", arg: 1, scope: !37, file: !3, type: !9)
!42 = !DILocation(line: 0, scope: !37)
!43 = !DILocalVariable(name: "__nv_MAIN__F1L24_1Arg1", arg: 2, scope: !37, file: !3, type: !40)
!44 = !DILocalVariable(name: "__nv_MAIN__F1L24_1Arg2", arg: 3, scope: !37, file: !3, type: !40)
!45 = !DILocalVariable(name: "omp_integer_kind", scope: !37, file: !3, type: !9)
!46 = !DILocalVariable(name: "omp_logical_kind", scope: !37, file: !3, type: !9)
!47 = !DILocalVariable(name: "omp_lock_kind", scope: !37, file: !3, type: !9)
!48 = !DILocalVariable(name: "omp_sched_kind", scope: !37, file: !3, type: !9)
!49 = !DILocalVariable(name: "omp_real_kind", scope: !37, file: !3, type: !9)
!50 = !DILocalVariable(name: "omp_proc_bind_kind", scope: !37, file: !3, type: !9)
!51 = !DILocalVariable(name: "omp_lock_hint_kind", scope: !37, file: !3, type: !9)
!52 = !DILocalVariable(name: "omp_sched_static", scope: !37, file: !3, type: !9)
!53 = !DILocalVariable(name: "omp_sched_dynamic", scope: !37, file: !3, type: !9)
!54 = !DILocalVariable(name: "omp_sched_guided", scope: !37, file: !3, type: !9)
!55 = !DILocalVariable(name: "omp_sched_auto", scope: !37, file: !3, type: !9)
!56 = !DILocalVariable(name: "omp_proc_bind_false", scope: !37, file: !3, type: !9)
!57 = !DILocalVariable(name: "omp_proc_bind_true", scope: !37, file: !3, type: !9)
!58 = !DILocalVariable(name: "omp_proc_bind_master", scope: !37, file: !3, type: !9)
!59 = !DILocalVariable(name: "omp_proc_bind_close", scope: !37, file: !3, type: !9)
!60 = !DILocalVariable(name: "omp_proc_bind_spread", scope: !37, file: !3, type: !9)
!61 = !DILocalVariable(name: "omp_lock_hint_none", scope: !37, file: !3, type: !9)
!62 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !37, file: !3, type: !9)
!63 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !37, file: !3, type: !9)
!64 = !DILocalVariable(name: "omp_lock_hint_nonspeculative", scope: !37, file: !3, type: !9)
!65 = !DILocation(line: 36, column: 1, scope: !37)
!66 = !DILocation(line: 35, column: 1, scope: !37)
!67 = !DILocation(line: 25, column: 1, scope: !37)
!68 = !DILocalVariable(name: "section_count", scope: !37, file: !3, type: !9)
!69 = !DILocation(line: 26, column: 1, scope: !37)
!70 = !DILocation(line: 27, column: 1, scope: !37)
!71 = !DILocation(line: 28, column: 1, scope: !37)
!72 = !DILocalVariable(scope: !37, file: !3, type: !9, flags: DIFlagArtificial)
!73 = !DILocation(line: 31, column: 1, scope: !37)
!74 = !DILocation(line: 32, column: 1, scope: !37)
!75 = !DILocation(line: 33, column: 1, scope: !37)

; ModuleID = '/tmp/DRB137-simdsafelen-orig-no-c987a9.ll'
source_filename = "/tmp/DRB137-simdsafelen-orig-no-c987a9.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.BSS1 = type <{ [16 x i8] }>

@.BSS1 = internal global %struct.BSS1 zeroinitializer, align 32, !dbg !0
@.C337_MAIN_ = internal constant i64 3
@.C284_MAIN_ = internal constant i64 0
@.C334_MAIN_ = internal constant i32 6
@.C332_MAIN_ = internal constant [55 x i8] c"micro-benchmarks-fortran/DRB137-simdsafelen-orig-no.f95"
@.C320_MAIN_ = internal constant i32 27
@.C288_MAIN_ = internal constant float 1.000000e+00
@.C285_MAIN_ = internal constant i32 1
@.C300_MAIN_ = internal constant i32 4
@.C301_MAIN_ = internal constant i32 2
@.C283_MAIN_ = internal constant i32 0

define void @MAIN_() #0 !dbg !2 {
L.entry:
  %m_322 = alloca i32, align 4
  %n_323 = alloca i32, align 4
  %.i0000_329 = alloca i32, align 4
  %.dY0001_347 = alloca i32, align 4
  %i_328 = alloca i32, align 4
  %z__io_336 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 4, metadata !15, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.value(metadata i32 4, metadata !18, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.value(metadata i32 4, metadata !19, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.value(metadata i32 4, metadata !20, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.value(metadata i32 4, metadata !21, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.value(metadata i32 4, metadata !22, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.value(metadata i32 4, metadata !23, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.value(metadata i32 1, metadata !24, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.value(metadata i32 2, metadata !25, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.value(metadata i32 4, metadata !26, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.value(metadata i32 0, metadata !27, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.value(metadata i32 1, metadata !28, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.value(metadata i32 2, metadata !29, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.value(metadata i32 4, metadata !30, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.value(metadata i32 0, metadata !31, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.value(metadata i32 1, metadata !32, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.value(metadata i32 2, metadata !33, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.value(metadata i32 4, metadata !34, metadata !DIExpression()), !dbg !17
  %0 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !35
  %1 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !35
  call void (i8*, ...) %1(i8* %0), !dbg !35
  br label %L.LB1_349

L.LB1_349:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %m_322, metadata !36, metadata !DIExpression()), !dbg !17
  store i32 2, i32* %m_322, align 4, !dbg !37
  call void @llvm.dbg.declare(metadata i32* %n_323, metadata !38, metadata !DIExpression()), !dbg !17
  store i32 4, i32* %n_323, align 4, !dbg !39
  br label %L.LB1_327

L.LB1_327:                                        ; preds = %L.LB1_349
  %2 = load i32, i32* %n_323, align 4, !dbg !40
  call void @llvm.dbg.value(metadata i32 %2, metadata !38, metadata !DIExpression()), !dbg !17
  %3 = load i32, i32* %m_322, align 4, !dbg !40
  call void @llvm.dbg.value(metadata i32 %3, metadata !36, metadata !DIExpression()), !dbg !17
  %4 = sub nsw i32 %2, %3, !dbg !40
  %5 = load i32, i32* %m_322, align 4, !dbg !40
  call void @llvm.dbg.value(metadata i32 %5, metadata !36, metadata !DIExpression()), !dbg !17
  %6 = add nsw i32 %5, 1, !dbg !40
  %7 = add nsw i32 %4, %6, !dbg !40
  store i32 %7, i32* %.i0000_329, align 4, !dbg !40
  %8 = load i32, i32* %n_323, align 4, !dbg !40
  call void @llvm.dbg.value(metadata i32 %8, metadata !38, metadata !DIExpression()), !dbg !17
  %9 = load i32, i32* %m_322, align 4, !dbg !40
  call void @llvm.dbg.value(metadata i32 %9, metadata !36, metadata !DIExpression()), !dbg !17
  %10 = sub nsw i32 %8, %9, !dbg !40
  store i32 %10, i32* %.dY0001_347, align 4, !dbg !40
  %11 = load i32, i32* %m_322, align 4, !dbg !40
  call void @llvm.dbg.value(metadata i32 %11, metadata !36, metadata !DIExpression()), !dbg !17
  %12 = add nsw i32 %11, 1, !dbg !40
  call void @llvm.dbg.declare(metadata i32* %i_328, metadata !41, metadata !DIExpression()), !dbg !42
  store i32 %12, i32* %i_328, align 4, !dbg !40
  %13 = load i32, i32* %.dY0001_347, align 4, !dbg !40
  %14 = icmp sle i32 %13, 0, !dbg !40
  br i1 %14, label %L.LB1_346, label %L.LB1_345, !dbg !40

L.LB1_345:                                        ; preds = %L.LB1_345, %L.LB1_327
  %15 = load i32, i32* %i_328, align 4, !dbg !43
  call void @llvm.dbg.value(metadata i32 %15, metadata !41, metadata !DIExpression()), !dbg !42
  %16 = load i32, i32* %m_322, align 4, !dbg !43
  call void @llvm.dbg.value(metadata i32 %16, metadata !36, metadata !DIExpression()), !dbg !17
  %17 = sub nsw i32 %15, %16, !dbg !43
  %18 = sext i32 %17 to i64, !dbg !43
  %19 = bitcast %struct.BSS1* @.BSS1 to i8*, !dbg !43
  %20 = getelementptr i8, i8* %19, i64 -4, !dbg !43
  %21 = bitcast i8* %20 to float*, !dbg !43
  %22 = getelementptr float, float* %21, i64 %18, !dbg !43
  %23 = load float, float* %22, align 4, !dbg !43
  %24 = fsub fast float %23, 1.000000e+00, !dbg !43
  %25 = load i32, i32* %i_328, align 4, !dbg !43
  call void @llvm.dbg.value(metadata i32 %25, metadata !41, metadata !DIExpression()), !dbg !42
  %26 = sext i32 %25 to i64, !dbg !43
  %27 = bitcast %struct.BSS1* @.BSS1 to i8*, !dbg !43
  %28 = getelementptr i8, i8* %27, i64 -4, !dbg !43
  %29 = bitcast i8* %28 to float*, !dbg !43
  %30 = getelementptr float, float* %29, i64 %26, !dbg !43
  store float %24, float* %30, align 4, !dbg !43
  %31 = load i32, i32* %i_328, align 4, !dbg !44
  call void @llvm.dbg.value(metadata i32 %31, metadata !41, metadata !DIExpression()), !dbg !42
  %32 = add nsw i32 %31, 1, !dbg !44
  store i32 %32, i32* %i_328, align 4, !dbg !44
  %33 = load i32, i32* %.dY0001_347, align 4, !dbg !44
  %34 = sub nsw i32 %33, 1, !dbg !44
  store i32 %34, i32* %.dY0001_347, align 4, !dbg !44
  %35 = load i32, i32* %.dY0001_347, align 4, !dbg !44
  %36 = icmp sgt i32 %35, 0, !dbg !44
  br i1 %36, label %L.LB1_345, label %L.LB1_346, !dbg !44

L.LB1_346:                                        ; preds = %L.LB1_345, %L.LB1_327
  br label %L.LB1_330

L.LB1_330:                                        ; preds = %L.LB1_346
  call void (...) @_mp_bcs_nest(), !dbg !45
  %37 = bitcast i32* @.C320_MAIN_ to i8*, !dbg !45
  %38 = bitcast [55 x i8]* @.C332_MAIN_ to i8*, !dbg !45
  %39 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !45
  call void (i8*, i8*, i64, ...) %39(i8* %37, i8* %38, i64 55), !dbg !45
  %40 = bitcast i32* @.C334_MAIN_ to i8*, !dbg !45
  %41 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !45
  %42 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !45
  %43 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !45
  %44 = call i32 (i8*, i8*, i8*, i8*, ...) %43(i8* %40, i8* null, i8* %41, i8* %42), !dbg !45
  call void @llvm.dbg.declare(metadata i32* %z__io_336, metadata !46, metadata !DIExpression()), !dbg !17
  store i32 %44, i32* %z__io_336, align 4, !dbg !45
  %45 = bitcast %struct.BSS1* @.BSS1 to i8*, !dbg !45
  %46 = getelementptr i8, i8* %45, i64 8, !dbg !45
  %47 = bitcast i8* %46 to float*, !dbg !45
  %48 = load float, float* %47, align 4, !dbg !45
  %49 = bitcast i32 (...)* @f90io_sc_f_ldw to i32 (float, i32, ...)*, !dbg !45
  %50 = call i32 (float, i32, ...) %49(float %48, i32 27), !dbg !45
  store i32 %50, i32* %z__io_336, align 4, !dbg !45
  %51 = call i32 (...) @f90io_ldw_end(), !dbg !45
  store i32 %51, i32* %z__io_336, align 4, !dbg !45
  call void (...) @_mp_ecs_nest(), !dbg !45
  ret void, !dbg !42
}

declare void @_mp_ecs_nest(...) #0

declare signext i32 @f90io_ldw_end(...) #0

declare signext i32 @f90io_sc_f_ldw(...) #0

declare signext i32 @f90io_print_init(...) #0

declare void @f90io_src_info03a(...) #0

declare void @_mp_bcs_nest(...) #0

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare void @fort_init(...) #0

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { "no-frame-pointer-elim-non-leaf" }
attributes #1 = { nounwind readnone speculatable }

!llvm.module.flags = !{!13, !14}
!llvm.dbg.cu = !{!4}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "b", scope: !2, file: !3, type: !9, isLocal: true, isDefinition: true)
!2 = distinct !DISubprogram(name: "drb137_simdsafelen_orig_no", scope: !4, file: !3, line: 12, type: !7, scopeLine: 12, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !4)
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB137-simdsafelen-orig-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !6, imports: !5)
!5 = !{}
!6 = !{!0}
!7 = !DISubroutineType(cc: DW_CC_program, types: !8)
!8 = !{null}
!9 = !DICompositeType(tag: DW_TAG_array_type, baseType: !10, size: 128, align: 32, elements: !11)
!10 = !DIBasicType(name: "real", size: 32, align: 32, encoding: DW_ATE_float)
!11 = !{!12}
!12 = !DISubrange(count: 4, lowerBound: 1)
!13 = !{i32 2, !"Dwarf Version", i32 4}
!14 = !{i32 2, !"Debug Info Version", i32 3}
!15 = !DILocalVariable(name: "omp_integer_kind", scope: !2, file: !3, type: !16)
!16 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!17 = !DILocation(line: 0, scope: !2)
!18 = !DILocalVariable(name: "omp_logical_kind", scope: !2, file: !3, type: !16)
!19 = !DILocalVariable(name: "omp_lock_kind", scope: !2, file: !3, type: !16)
!20 = !DILocalVariable(name: "omp_sched_kind", scope: !2, file: !3, type: !16)
!21 = !DILocalVariable(name: "omp_real_kind", scope: !2, file: !3, type: !16)
!22 = !DILocalVariable(name: "omp_proc_bind_kind", scope: !2, file: !3, type: !16)
!23 = !DILocalVariable(name: "omp_lock_hint_kind", scope: !2, file: !3, type: !16)
!24 = !DILocalVariable(name: "omp_sched_static", scope: !2, file: !3, type: !16)
!25 = !DILocalVariable(name: "omp_sched_dynamic", scope: !2, file: !3, type: !16)
!26 = !DILocalVariable(name: "omp_sched_auto", scope: !2, file: !3, type: !16)
!27 = !DILocalVariable(name: "omp_proc_bind_false", scope: !2, file: !3, type: !16)
!28 = !DILocalVariable(name: "omp_proc_bind_true", scope: !2, file: !3, type: !16)
!29 = !DILocalVariable(name: "omp_proc_bind_master", scope: !2, file: !3, type: !16)
!30 = !DILocalVariable(name: "omp_proc_bind_spread", scope: !2, file: !3, type: !16)
!31 = !DILocalVariable(name: "omp_lock_hint_none", scope: !2, file: !3, type: !16)
!32 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !2, file: !3, type: !16)
!33 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !2, file: !3, type: !16)
!34 = !DILocalVariable(name: "omp_lock_hint_nonspeculative", scope: !2, file: !3, type: !16)
!35 = !DILocation(line: 12, column: 1, scope: !2)
!36 = !DILocalVariable(name: "m", scope: !2, file: !3, type: !16)
!37 = !DILocation(line: 19, column: 1, scope: !2)
!38 = !DILocalVariable(name: "n", scope: !2, file: !3, type: !16)
!39 = !DILocation(line: 20, column: 1, scope: !2)
!40 = !DILocation(line: 23, column: 1, scope: !2)
!41 = !DILocalVariable(name: "i", scope: !2, file: !3, type: !16)
!42 = !DILocation(line: 28, column: 1, scope: !2)
!43 = !DILocation(line: 24, column: 1, scope: !2)
!44 = !DILocation(line: 25, column: 1, scope: !2)
!45 = !DILocation(line: 27, column: 1, scope: !2)
!46 = !DILocalVariable(scope: !2, file: !3, type: !16, flags: DIFlagArtificial)

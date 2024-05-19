; ModuleID = '/tmp/DRB138-simdsafelen-orig-yes-a28cda.ll'
source_filename = "/tmp/DRB138-simdsafelen-orig-yes-a28cda.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.BSS1 = type <{ [16 x i8] }>

@.BSS1 = internal global %struct.BSS1 zeroinitializer, align 32, !dbg !0
@.C333_MAIN_ = internal constant i64 3
@.C284_MAIN_ = internal constant i64 0
@.C330_MAIN_ = internal constant i32 6
@.C328_MAIN_ = internal constant [56 x i8] c"micro-benchmarks-fortran/DRB138-simdsafelen-orig-yes.f95"
@.C316_MAIN_ = internal constant i32 27
@.C288_MAIN_ = internal constant float 1.000000e+00
@.C300_MAIN_ = internal constant i32 4
@.C285_MAIN_ = internal constant i32 1
@.C283_MAIN_ = internal constant i32 0

define void @MAIN_() #0 !dbg !2 {
L.entry:
  %m_318 = alloca i32, align 4
  %n_319 = alloca i32, align 4
  %.i0000_325 = alloca i32, align 4
  %.dY0001_343 = alloca i32, align 4
  %i_324 = alloca i32, align 4
  %z__io_332 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 4, metadata !15, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.value(metadata i32 4, metadata !18, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.value(metadata i32 4, metadata !19, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.value(metadata i32 4, metadata !20, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.value(metadata i32 4, metadata !21, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.value(metadata i32 4, metadata !22, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.value(metadata i32 4, metadata !23, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.value(metadata i32 1, metadata !24, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.value(metadata i32 4, metadata !25, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.value(metadata i32 0, metadata !26, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.value(metadata i32 1, metadata !27, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.value(metadata i32 4, metadata !28, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.value(metadata i32 0, metadata !29, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.value(metadata i32 1, metadata !30, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.value(metadata i32 4, metadata !31, metadata !DIExpression()), !dbg !17
  %0 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !32
  %1 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !32
  call void (i8*, ...) %1(i8* %0), !dbg !32
  br label %L.LB1_345

L.LB1_345:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %m_318, metadata !33, metadata !DIExpression()), !dbg !17
  store i32 1, i32* %m_318, align 4, !dbg !34
  call void @llvm.dbg.declare(metadata i32* %n_319, metadata !35, metadata !DIExpression()), !dbg !17
  store i32 4, i32* %n_319, align 4, !dbg !36
  br label %L.LB1_323

L.LB1_323:                                        ; preds = %L.LB1_345
  %2 = load i32, i32* %n_319, align 4, !dbg !37
  call void @llvm.dbg.value(metadata i32 %2, metadata !35, metadata !DIExpression()), !dbg !17
  %3 = load i32, i32* %m_318, align 4, !dbg !37
  call void @llvm.dbg.value(metadata i32 %3, metadata !33, metadata !DIExpression()), !dbg !17
  %4 = sub nsw i32 %2, %3, !dbg !37
  %5 = load i32, i32* %m_318, align 4, !dbg !37
  call void @llvm.dbg.value(metadata i32 %5, metadata !33, metadata !DIExpression()), !dbg !17
  %6 = add nsw i32 %5, 1, !dbg !37
  %7 = add nsw i32 %4, %6, !dbg !37
  store i32 %7, i32* %.i0000_325, align 4, !dbg !37
  %8 = load i32, i32* %n_319, align 4, !dbg !37
  call void @llvm.dbg.value(metadata i32 %8, metadata !35, metadata !DIExpression()), !dbg !17
  %9 = load i32, i32* %m_318, align 4, !dbg !37
  call void @llvm.dbg.value(metadata i32 %9, metadata !33, metadata !DIExpression()), !dbg !17
  %10 = sub nsw i32 %8, %9, !dbg !37
  store i32 %10, i32* %.dY0001_343, align 4, !dbg !37
  %11 = load i32, i32* %m_318, align 4, !dbg !37
  call void @llvm.dbg.value(metadata i32 %11, metadata !33, metadata !DIExpression()), !dbg !17
  %12 = add nsw i32 %11, 1, !dbg !37
  call void @llvm.dbg.declare(metadata i32* %i_324, metadata !38, metadata !DIExpression()), !dbg !39
  store i32 %12, i32* %i_324, align 4, !dbg !37
  %13 = load i32, i32* %.dY0001_343, align 4, !dbg !37
  %14 = icmp sle i32 %13, 0, !dbg !37
  br i1 %14, label %L.LB1_342, label %L.LB1_341, !dbg !37

L.LB1_341:                                        ; preds = %L.LB1_341, %L.LB1_323
  %15 = load i32, i32* %i_324, align 4, !dbg !40
  call void @llvm.dbg.value(metadata i32 %15, metadata !38, metadata !DIExpression()), !dbg !39
  %16 = load i32, i32* %m_318, align 4, !dbg !40
  call void @llvm.dbg.value(metadata i32 %16, metadata !33, metadata !DIExpression()), !dbg !17
  %17 = sub nsw i32 %15, %16, !dbg !40
  %18 = sext i32 %17 to i64, !dbg !40
  %19 = bitcast %struct.BSS1* @.BSS1 to i8*, !dbg !40
  %20 = getelementptr i8, i8* %19, i64 -4, !dbg !40
  %21 = bitcast i8* %20 to float*, !dbg !40
  %22 = getelementptr float, float* %21, i64 %18, !dbg !40
  %23 = load float, float* %22, align 4, !dbg !40
  %24 = fsub fast float %23, 1.000000e+00, !dbg !40
  %25 = load i32, i32* %i_324, align 4, !dbg !40
  call void @llvm.dbg.value(metadata i32 %25, metadata !38, metadata !DIExpression()), !dbg !39
  %26 = sext i32 %25 to i64, !dbg !40
  %27 = bitcast %struct.BSS1* @.BSS1 to i8*, !dbg !40
  %28 = getelementptr i8, i8* %27, i64 -4, !dbg !40
  %29 = bitcast i8* %28 to float*, !dbg !40
  %30 = getelementptr float, float* %29, i64 %26, !dbg !40
  store float %24, float* %30, align 4, !dbg !40
  %31 = load i32, i32* %i_324, align 4, !dbg !41
  call void @llvm.dbg.value(metadata i32 %31, metadata !38, metadata !DIExpression()), !dbg !39
  %32 = add nsw i32 %31, 1, !dbg !41
  store i32 %32, i32* %i_324, align 4, !dbg !41
  %33 = load i32, i32* %.dY0001_343, align 4, !dbg !41
  %34 = sub nsw i32 %33, 1, !dbg !41
  store i32 %34, i32* %.dY0001_343, align 4, !dbg !41
  %35 = load i32, i32* %.dY0001_343, align 4, !dbg !41
  %36 = icmp sgt i32 %35, 0, !dbg !41
  br i1 %36, label %L.LB1_341, label %L.LB1_342, !dbg !41

L.LB1_342:                                        ; preds = %L.LB1_341, %L.LB1_323
  br label %L.LB1_326

L.LB1_326:                                        ; preds = %L.LB1_342
  call void (...) @_mp_bcs_nest(), !dbg !42
  %37 = bitcast i32* @.C316_MAIN_ to i8*, !dbg !42
  %38 = bitcast [56 x i8]* @.C328_MAIN_ to i8*, !dbg !42
  %39 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !42
  call void (i8*, i8*, i64, ...) %39(i8* %37, i8* %38, i64 56), !dbg !42
  %40 = bitcast i32* @.C330_MAIN_ to i8*, !dbg !42
  %41 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !42
  %42 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !42
  %43 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !42
  %44 = call i32 (i8*, i8*, i8*, i8*, ...) %43(i8* %40, i8* null, i8* %41, i8* %42), !dbg !42
  call void @llvm.dbg.declare(metadata i32* %z__io_332, metadata !43, metadata !DIExpression()), !dbg !17
  store i32 %44, i32* %z__io_332, align 4, !dbg !42
  %45 = bitcast %struct.BSS1* @.BSS1 to i8*, !dbg !42
  %46 = getelementptr i8, i8* %45, i64 8, !dbg !42
  %47 = bitcast i8* %46 to float*, !dbg !42
  %48 = load float, float* %47, align 4, !dbg !42
  %49 = bitcast i32 (...)* @f90io_sc_f_ldw to i32 (float, i32, ...)*, !dbg !42
  %50 = call i32 (float, i32, ...) %49(float %48, i32 27), !dbg !42
  store i32 %50, i32* %z__io_332, align 4, !dbg !42
  %51 = call i32 (...) @f90io_ldw_end(), !dbg !42
  store i32 %51, i32* %z__io_332, align 4, !dbg !42
  call void (...) @_mp_ecs_nest(), !dbg !42
  ret void, !dbg !39
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
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB138-simdsafelen-orig-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
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
!25 = !DILocalVariable(name: "omp_sched_auto", scope: !2, file: !3, type: !16)
!26 = !DILocalVariable(name: "omp_proc_bind_false", scope: !2, file: !3, type: !16)
!27 = !DILocalVariable(name: "omp_proc_bind_true", scope: !2, file: !3, type: !16)
!28 = !DILocalVariable(name: "omp_proc_bind_spread", scope: !2, file: !3, type: !16)
!29 = !DILocalVariable(name: "omp_lock_hint_none", scope: !2, file: !3, type: !16)
!30 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !2, file: !3, type: !16)
!31 = !DILocalVariable(name: "omp_lock_hint_nonspeculative", scope: !2, file: !3, type: !16)
!32 = !DILocation(line: 12, column: 1, scope: !2)
!33 = !DILocalVariable(name: "m", scope: !2, file: !3, type: !16)
!34 = !DILocation(line: 19, column: 1, scope: !2)
!35 = !DILocalVariable(name: "n", scope: !2, file: !3, type: !16)
!36 = !DILocation(line: 20, column: 1, scope: !2)
!37 = !DILocation(line: 23, column: 1, scope: !2)
!38 = !DILocalVariable(name: "i", scope: !2, file: !3, type: !16)
!39 = !DILocation(line: 28, column: 1, scope: !2)
!40 = !DILocation(line: 24, column: 1, scope: !2)
!41 = !DILocation(line: 25, column: 1, scope: !2)
!42 = !DILocation(line: 27, column: 1, scope: !2)
!43 = !DILocalVariable(scope: !2, file: !3, type: !16, flags: DIFlagArtificial)

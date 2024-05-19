; ModuleID = '/tmp/DRB089-dynamic-storage2-orig-yes-035864.ll'
source_filename = "/tmp/DRB089-dynamic-storage2-orig-yes-035864.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%astruct.dt63 = type <{ i8*, i8*, i8* }>

@.C305_MAIN_ = internal constant i32 25
@.C318_MAIN_ = internal constant i32 6
@.C316_MAIN_ = internal constant [61 x i8] c"micro-benchmarks-fortran/DRB089-dynamic-storage2-orig-yes.f95"
@.C306_MAIN_ = internal constant i32 28
@.C285_MAIN_ = internal constant i32 1
@.C330_MAIN_ = internal constant i64 25
@.C329_MAIN_ = internal constant i64 4
@.C286_MAIN_ = internal constant i64 1
@.C284_MAIN_ = internal constant i64 0
@.C283_MAIN_ = internal constant i32 0
@.C285___nv_MAIN__F1L24_1 = internal constant i32 1

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__356 = alloca i32, align 4
  %"counter$p_309" = alloca i32*, align 8
  %"counter$sd_308" = alloca [1 x i64], align 8
  %.uplevelArgPack0001_347 = alloca %astruct.dt63, align 16
  %z__io_320 = alloca i32, align 4
  %"MAIN___$eq_297" = alloca [32 x i8], align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !14, metadata !DIExpression()), !dbg !10
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !15
  store i32 %0, i32* %__gtid_MAIN__356, align 4, !dbg !15
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !16
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !16
  call void (i8*, ...) %2(i8* %1), !dbg !16
  %3 = bitcast i32** %"counter$p_309" to i8**, !dbg !16
  store i8* null, i8** %3, align 8, !dbg !16
  call void @llvm.dbg.declare(metadata [1 x i64]* %"counter$sd_308", metadata !17, metadata !DIExpression()), !dbg !10
  %4 = bitcast [1 x i64]* %"counter$sd_308" to i64*, !dbg !16
  store i64 0, i64* %4, align 8, !dbg !16
  br label %L.LB1_338

L.LB1_338:                                        ; preds = %L.entry
  %5 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !22
  %6 = bitcast i64* @.C330_MAIN_ to i8*, !dbg !22
  %7 = bitcast i64* @.C329_MAIN_ to i8*, !dbg !22
  %8 = bitcast i32** %"counter$p_309" to i8*, !dbg !22
  %9 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !22
  %10 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !22
  %11 = bitcast void (...)* @f90_ptr_alloc04a_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !22
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %11(i8* %5, i8* %6, i8* %7, i8* null, i8* %8, i8* null, i8* %9, i8* %10, i8* null, i64 0), !dbg !22
  %12 = load i32*, i32** %"counter$p_309", align 8, !dbg !23
  store i32 0, i32* %12, align 4, !dbg !23
  %13 = bitcast i32** %"counter$p_309" to i8*, !dbg !24
  %14 = bitcast %astruct.dt63* %.uplevelArgPack0001_347 to i8**, !dbg !24
  store i8* %13, i8** %14, align 8, !dbg !24
  %15 = bitcast [1 x i64]* %"counter$sd_308" to i8*, !dbg !24
  %16 = bitcast %astruct.dt63* %.uplevelArgPack0001_347 to i8*, !dbg !24
  %17 = getelementptr i8, i8* %16, i64 8, !dbg !24
  %18 = bitcast i8* %17 to i8**, !dbg !24
  store i8* %15, i8** %18, align 8, !dbg !24
  %19 = bitcast i32** %"counter$p_309" to i8*, !dbg !24
  %20 = bitcast %astruct.dt63* %.uplevelArgPack0001_347 to i8*, !dbg !24
  %21 = getelementptr i8, i8* %20, i64 16, !dbg !24
  %22 = bitcast i8* %21 to i8**, !dbg !24
  store i8* %19, i8** %22, align 8, !dbg !24
  br label %L.LB1_354, !dbg !24

L.LB1_354:                                        ; preds = %L.LB1_338
  %23 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L24_1_ to i64*, !dbg !24
  %24 = bitcast %astruct.dt63* %.uplevelArgPack0001_347 to i64*, !dbg !24
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %23, i64* %24), !dbg !24
  call void (...) @_mp_bcs_nest(), !dbg !25
  %25 = bitcast i32* @.C306_MAIN_ to i8*, !dbg !25
  %26 = bitcast [61 x i8]* @.C316_MAIN_ to i8*, !dbg !25
  %27 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !25
  call void (i8*, i8*, i64, ...) %27(i8* %25, i8* %26, i64 61), !dbg !25
  %28 = bitcast i32* @.C318_MAIN_ to i8*, !dbg !25
  %29 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !25
  %30 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !25
  %31 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !25
  %32 = call i32 (i8*, i8*, i8*, i8*, ...) %31(i8* %28, i8* null, i8* %29, i8* %30), !dbg !25
  call void @llvm.dbg.declare(metadata i32* %z__io_320, metadata !26, metadata !DIExpression()), !dbg !10
  store i32 %32, i32* %z__io_320, align 4, !dbg !25
  %33 = load i32*, i32** %"counter$p_309", align 8, !dbg !25
  %34 = load i32, i32* %33, align 4, !dbg !25
  %35 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !25
  %36 = call i32 (i32, i32, ...) %35(i32 %34, i32 25), !dbg !25
  store i32 %36, i32* %z__io_320, align 4, !dbg !25
  %37 = call i32 (...) @f90io_ldw_end(), !dbg !25
  store i32 %37, i32* %z__io_320, align 4, !dbg !25
  call void (...) @_mp_ecs_nest(), !dbg !25
  %38 = load i32*, i32** %"counter$p_309", align 8, !dbg !27
  %39 = bitcast i32* %38 to i8*, !dbg !27
  %40 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !27
  %41 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !27
  call void (i8*, i8*, i8*, i8*, i64, ...) %41(i8* null, i8* %39, i8* %40, i8* null, i64 0), !dbg !27
  %42 = bitcast i32** %"counter$p_309" to i8**, !dbg !27
  store i8* null, i8** %42, align 8, !dbg !27
  %43 = bitcast [1 x i64]* %"counter$sd_308" to i64*, !dbg !27
  store i64 0, i64* %43, align 8, !dbg !27
  ret void, !dbg !15
}

define internal void @__nv_MAIN__F1L24_1_(i32* %__nv_MAIN__F1L24_1Arg0, i64* %__nv_MAIN__F1L24_1Arg1, i64* %__nv_MAIN__F1L24_1Arg2) #0 !dbg !28 {
L.entry:
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L24_1Arg0, metadata !31, metadata !DIExpression()), !dbg !32
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L24_1Arg1, metadata !33, metadata !DIExpression()), !dbg !32
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L24_1Arg2, metadata !34, metadata !DIExpression()), !dbg !32
  call void @llvm.dbg.value(metadata i32 1, metadata !35, metadata !DIExpression()), !dbg !32
  call void @llvm.dbg.value(metadata i32 0, metadata !36, metadata !DIExpression()), !dbg !32
  call void @llvm.dbg.value(metadata i32 1, metadata !37, metadata !DIExpression()), !dbg !32
  call void @llvm.dbg.value(metadata i32 0, metadata !38, metadata !DIExpression()), !dbg !32
  call void @llvm.dbg.value(metadata i32 1, metadata !39, metadata !DIExpression()), !dbg !32
  br label %L.LB2_383

L.LB2_383:                                        ; preds = %L.entry
  br label %L.LB2_313

L.LB2_313:                                        ; preds = %L.LB2_383
  %0 = bitcast i64* %__nv_MAIN__F1L24_1Arg2 to i8*, !dbg !40
  %1 = getelementptr i8, i8* %0, i64 16, !dbg !40
  %2 = bitcast i8* %1 to i32***, !dbg !40
  %3 = load i32**, i32*** %2, align 8, !dbg !40
  %4 = load i32*, i32** %3, align 8, !dbg !40
  %5 = load i32, i32* %4, align 4, !dbg !40
  %6 = add nsw i32 %5, 1, !dbg !40
  %7 = bitcast i64* %__nv_MAIN__F1L24_1Arg2 to i8*, !dbg !40
  %8 = getelementptr i8, i8* %7, i64 16, !dbg !40
  %9 = bitcast i8* %8 to i32***, !dbg !40
  %10 = load i32**, i32*** %9, align 8, !dbg !40
  %11 = load i32*, i32** %10, align 8, !dbg !40
  store i32 %6, i32* %11, align 4, !dbg !40
  br label %L.LB2_314

L.LB2_314:                                        ; preds = %L.LB2_313
  ret void, !dbg !41
}

declare void @f90_dealloc03a_i8(...) #0

declare void @_mp_ecs_nest(...) #0

declare signext i32 @f90io_ldw_end(...) #0

declare signext i32 @f90io_sc_i_ldw(...) #0

declare signext i32 @f90io_print_init(...) #0

declare void @f90io_src_info03a(...) #0

declare void @_mp_bcs_nest(...) #0

declare void @f90_ptr_alloc04a_i8(...) #0

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
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB089-dynamic-storage2-orig-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb088_dynamic_storage_orig_yes", scope: !2, file: !3, line: 15, type: !6, scopeLine: 15, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!6 = !DISubroutineType(cc: DW_CC_program, types: !7)
!7 = !{null}
!8 = !DILocalVariable(name: "omp_sched_static", scope: !5, file: !3, type: !9)
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 0, scope: !5)
!11 = !DILocalVariable(name: "omp_proc_bind_false", scope: !5, file: !3, type: !9)
!12 = !DILocalVariable(name: "omp_proc_bind_true", scope: !5, file: !3, type: !9)
!13 = !DILocalVariable(name: "omp_lock_hint_none", scope: !5, file: !3, type: !9)
!14 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !5, file: !3, type: !9)
!15 = !DILocation(line: 32, column: 1, scope: !5)
!16 = !DILocation(line: 15, column: 1, scope: !5)
!17 = !DILocalVariable(scope: !5, file: !3, type: !18, flags: DIFlagArtificial)
!18 = !DICompositeType(tag: DW_TAG_array_type, baseType: !19, size: 64, align: 64, elements: !20)
!19 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!20 = !{!21}
!21 = !DISubrange(count: 0, lowerBound: 1)
!22 = !DILocation(line: 20, column: 1, scope: !5)
!23 = !DILocation(line: 22, column: 1, scope: !5)
!24 = !DILocation(line: 24, column: 1, scope: !5)
!25 = !DILocation(line: 28, column: 1, scope: !5)
!26 = !DILocalVariable(scope: !5, file: !3, type: !9, flags: DIFlagArtificial)
!27 = !DILocation(line: 30, column: 1, scope: !5)
!28 = distinct !DISubprogram(name: "__nv_MAIN__F1L24_1", scope: !2, file: !3, line: 24, type: !29, scopeLine: 24, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!29 = !DISubroutineType(types: !30)
!30 = !{null, !9, !19, !19}
!31 = !DILocalVariable(name: "__nv_MAIN__F1L24_1Arg0", arg: 1, scope: !28, file: !3, type: !9)
!32 = !DILocation(line: 0, scope: !28)
!33 = !DILocalVariable(name: "__nv_MAIN__F1L24_1Arg1", arg: 2, scope: !28, file: !3, type: !19)
!34 = !DILocalVariable(name: "__nv_MAIN__F1L24_1Arg2", arg: 3, scope: !28, file: !3, type: !19)
!35 = !DILocalVariable(name: "omp_sched_static", scope: !28, file: !3, type: !9)
!36 = !DILocalVariable(name: "omp_proc_bind_false", scope: !28, file: !3, type: !9)
!37 = !DILocalVariable(name: "omp_proc_bind_true", scope: !28, file: !3, type: !9)
!38 = !DILocalVariable(name: "omp_lock_hint_none", scope: !28, file: !3, type: !9)
!39 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !28, file: !3, type: !9)
!40 = !DILocation(line: 25, column: 1, scope: !28)
!41 = !DILocation(line: 26, column: 1, scope: !28)

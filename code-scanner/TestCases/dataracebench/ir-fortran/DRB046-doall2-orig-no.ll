; ModuleID = '/tmp/DRB046-doall2-orig-no-d4f2bd.ll'
source_filename = "/tmp/DRB046-doall2-orig-no-d4f2bd.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.BSS1 = type <{ [40000 x i8] }>
%astruct.dt59 = type <{ i8* }>

@.BSS1 = internal global %struct.BSS1 zeroinitializer, align 32
@.C308_MAIN_ = internal constant i32 100
@.C285_MAIN_ = internal constant i32 1
@.C283_MAIN_ = internal constant i32 0
@.C308___nv_MAIN__F1L20_1 = internal constant i32 100
@.C285___nv_MAIN__F1L20_1 = internal constant i32 1
@.C283___nv_MAIN__F1L20_1 = internal constant i32 0

define void @MAIN_() #0 !dbg !18 {
L.entry:
  %__gtid_MAIN__349 = alloca i32, align 4
  %.uplevelArgPack0001_343 = alloca %astruct.dt59, align 8
  call void @llvm.dbg.value(metadata i32 1, metadata !21, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i32 0, metadata !23, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i32 1, metadata !24, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i32 0, metadata !25, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i32 1, metadata !26, metadata !DIExpression()), !dbg !22
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !27
  store i32 %0, i32* %__gtid_MAIN__349, align 4, !dbg !27
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !28
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !28
  call void (i8*, ...) %2(i8* %1), !dbg !28
  br label %L.LB1_338

L.LB1_338:                                        ; preds = %L.entry
  br label %L.LB1_347, !dbg !29

L.LB1_347:                                        ; preds = %L.LB1_338
  %3 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L20_1_ to i64*, !dbg !29
  %4 = bitcast %astruct.dt59* %.uplevelArgPack0001_343 to i64*, !dbg !29
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %3, i64* %4), !dbg !29
  ret void, !dbg !27
}

define internal void @__nv_MAIN__F1L20_1_(i32* %__nv_MAIN__F1L20_1Arg0, i64* %__nv_MAIN__F1L20_1Arg1, i64* %__nv_MAIN__F1L20_1Arg2) #0 !dbg !14 {
L.entry:
  %__gtid___nv_MAIN__F1L20_1__384 = alloca i32, align 4
  %.i0000p_317 = alloca i32, align 4
  %i_316 = alloca i32, align 4
  %.du0001p_325 = alloca i32, align 4
  %.de0001p_326 = alloca i32, align 4
  %.di0001p_327 = alloca i32, align 4
  %.ds0001p_328 = alloca i32, align 4
  %.dl0001p_330 = alloca i32, align 4
  %.dl0001p.copy_378 = alloca i32, align 4
  %.de0001p.copy_379 = alloca i32, align 4
  %.ds0001p.copy_380 = alloca i32, align 4
  %.dX0001p_329 = alloca i32, align 4
  %.dY0001p_324 = alloca i32, align 4
  %.dY0002p_336 = alloca i32, align 4
  %j_315 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L20_1Arg0, metadata !30, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L20_1Arg1, metadata !32, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L20_1Arg2, metadata !33, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.value(metadata i32 1, metadata !34, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.value(metadata i32 0, metadata !35, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.value(metadata i32 1, metadata !36, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.value(metadata i32 0, metadata !37, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.value(metadata i32 1, metadata !38, metadata !DIExpression()), !dbg !31
  %0 = load i32, i32* %__nv_MAIN__F1L20_1Arg0, align 4, !dbg !39
  store i32 %0, i32* %__gtid___nv_MAIN__F1L20_1__384, align 4, !dbg !39
  br label %L.LB2_370

L.LB2_370:                                        ; preds = %L.entry
  br label %L.LB2_314

L.LB2_314:                                        ; preds = %L.LB2_370
  store i32 0, i32* %.i0000p_317, align 4, !dbg !40
  call void @llvm.dbg.declare(metadata i32* %i_316, metadata !41, metadata !DIExpression()), !dbg !39
  store i32 1, i32* %i_316, align 4, !dbg !40
  store i32 100, i32* %.du0001p_325, align 4, !dbg !40
  store i32 100, i32* %.de0001p_326, align 4, !dbg !40
  store i32 1, i32* %.di0001p_327, align 4, !dbg !40
  %1 = load i32, i32* %.di0001p_327, align 4, !dbg !40
  store i32 %1, i32* %.ds0001p_328, align 4, !dbg !40
  store i32 1, i32* %.dl0001p_330, align 4, !dbg !40
  %2 = load i32, i32* %.dl0001p_330, align 4, !dbg !40
  store i32 %2, i32* %.dl0001p.copy_378, align 4, !dbg !40
  %3 = load i32, i32* %.de0001p_326, align 4, !dbg !40
  store i32 %3, i32* %.de0001p.copy_379, align 4, !dbg !40
  %4 = load i32, i32* %.ds0001p_328, align 4, !dbg !40
  store i32 %4, i32* %.ds0001p.copy_380, align 4, !dbg !40
  %5 = load i32, i32* %__gtid___nv_MAIN__F1L20_1__384, align 4, !dbg !40
  %6 = bitcast i32* %.i0000p_317 to i64*, !dbg !40
  %7 = bitcast i32* %.dl0001p.copy_378 to i64*, !dbg !40
  %8 = bitcast i32* %.de0001p.copy_379 to i64*, !dbg !40
  %9 = bitcast i32* %.ds0001p.copy_380 to i64*, !dbg !40
  %10 = load i32, i32* %.ds0001p.copy_380, align 4, !dbg !40
  call void @__kmpc_for_static_init_4(i64* null, i32 %5, i32 34, i64* %6, i64* %7, i64* %8, i64* %9, i32 %10, i32 1), !dbg !40
  %11 = load i32, i32* %.dl0001p.copy_378, align 4, !dbg !40
  store i32 %11, i32* %.dl0001p_330, align 4, !dbg !40
  %12 = load i32, i32* %.de0001p.copy_379, align 4, !dbg !40
  store i32 %12, i32* %.de0001p_326, align 4, !dbg !40
  %13 = load i32, i32* %.ds0001p.copy_380, align 4, !dbg !40
  store i32 %13, i32* %.ds0001p_328, align 4, !dbg !40
  %14 = load i32, i32* %.dl0001p_330, align 4, !dbg !40
  store i32 %14, i32* %i_316, align 4, !dbg !40
  %15 = load i32, i32* %i_316, align 4, !dbg !40
  call void @llvm.dbg.value(metadata i32 %15, metadata !41, metadata !DIExpression()), !dbg !39
  store i32 %15, i32* %.dX0001p_329, align 4, !dbg !40
  %16 = load i32, i32* %.dX0001p_329, align 4, !dbg !40
  %17 = load i32, i32* %.du0001p_325, align 4, !dbg !40
  %18 = icmp sgt i32 %16, %17, !dbg !40
  br i1 %18, label %L.LB2_323, label %L.LB2_414, !dbg !40

L.LB2_414:                                        ; preds = %L.LB2_314
  %19 = load i32, i32* %.dX0001p_329, align 4, !dbg !40
  store i32 %19, i32* %i_316, align 4, !dbg !40
  %20 = load i32, i32* %.di0001p_327, align 4, !dbg !40
  %21 = load i32, i32* %.de0001p_326, align 4, !dbg !40
  %22 = load i32, i32* %.dX0001p_329, align 4, !dbg !40
  %23 = sub nsw i32 %21, %22, !dbg !40
  %24 = add nsw i32 %20, %23, !dbg !40
  %25 = load i32, i32* %.di0001p_327, align 4, !dbg !40
  %26 = sdiv i32 %24, %25, !dbg !40
  store i32 %26, i32* %.dY0001p_324, align 4, !dbg !40
  %27 = load i32, i32* %.dY0001p_324, align 4, !dbg !40
  %28 = icmp sle i32 %27, 0, !dbg !40
  br i1 %28, label %L.LB2_333, label %L.LB2_332, !dbg !40

L.LB2_332:                                        ; preds = %L.LB2_415, %L.LB2_414
  store i32 100, i32* %.dY0002p_336, align 4, !dbg !42
  call void @llvm.dbg.declare(metadata i32* %j_315, metadata !43, metadata !DIExpression()), !dbg !39
  store i32 1, i32* %j_315, align 4, !dbg !42
  br label %L.LB2_334

L.LB2_334:                                        ; preds = %L.LB2_334, %L.LB2_332
  %29 = load i32, i32* %i_316, align 4, !dbg !44
  call void @llvm.dbg.value(metadata i32 %29, metadata !41, metadata !DIExpression()), !dbg !39
  %30 = sext i32 %29 to i64, !dbg !44
  %31 = load i32, i32* %j_315, align 4, !dbg !44
  call void @llvm.dbg.value(metadata i32 %31, metadata !43, metadata !DIExpression()), !dbg !39
  %32 = sext i32 %31 to i64, !dbg !44
  %33 = mul nsw i64 %32, 100, !dbg !44
  %34 = add nsw i64 %30, %33, !dbg !44
  %35 = bitcast %struct.BSS1* @.BSS1 to i8*, !dbg !44
  %36 = getelementptr i8, i8* %35, i64 -404, !dbg !44
  %37 = bitcast i8* %36 to i32*, !dbg !44
  %38 = getelementptr i32, i32* %37, i64 %34, !dbg !44
  %39 = load i32, i32* %38, align 4, !dbg !44
  %40 = add nsw i32 %39, 1, !dbg !44
  %41 = load i32, i32* %i_316, align 4, !dbg !44
  call void @llvm.dbg.value(metadata i32 %41, metadata !41, metadata !DIExpression()), !dbg !39
  %42 = sext i32 %41 to i64, !dbg !44
  %43 = load i32, i32* %j_315, align 4, !dbg !44
  call void @llvm.dbg.value(metadata i32 %43, metadata !43, metadata !DIExpression()), !dbg !39
  %44 = sext i32 %43 to i64, !dbg !44
  %45 = mul nsw i64 %44, 100, !dbg !44
  %46 = add nsw i64 %42, %45, !dbg !44
  %47 = bitcast %struct.BSS1* @.BSS1 to i8*, !dbg !44
  %48 = getelementptr i8, i8* %47, i64 -404, !dbg !44
  %49 = bitcast i8* %48 to i32*, !dbg !44
  %50 = getelementptr i32, i32* %49, i64 %46, !dbg !44
  store i32 %40, i32* %50, align 4, !dbg !44
  %51 = load i32, i32* %j_315, align 4, !dbg !45
  call void @llvm.dbg.value(metadata i32 %51, metadata !43, metadata !DIExpression()), !dbg !39
  %52 = add nsw i32 %51, 1, !dbg !45
  store i32 %52, i32* %j_315, align 4, !dbg !45
  %53 = load i32, i32* %.dY0002p_336, align 4, !dbg !45
  %54 = sub nsw i32 %53, 1, !dbg !45
  store i32 %54, i32* %.dY0002p_336, align 4, !dbg !45
  %55 = load i32, i32* %.dY0002p_336, align 4, !dbg !45
  %56 = icmp sgt i32 %55, 0, !dbg !45
  br i1 %56, label %L.LB2_334, label %L.LB2_415, !dbg !45

L.LB2_415:                                        ; preds = %L.LB2_334
  %57 = load i32, i32* %.di0001p_327, align 4, !dbg !39
  %58 = load i32, i32* %i_316, align 4, !dbg !39
  call void @llvm.dbg.value(metadata i32 %58, metadata !41, metadata !DIExpression()), !dbg !39
  %59 = add nsw i32 %57, %58, !dbg !39
  store i32 %59, i32* %i_316, align 4, !dbg !39
  %60 = load i32, i32* %.dY0001p_324, align 4, !dbg !39
  %61 = sub nsw i32 %60, 1, !dbg !39
  store i32 %61, i32* %.dY0001p_324, align 4, !dbg !39
  %62 = load i32, i32* %.dY0001p_324, align 4, !dbg !39
  %63 = icmp sgt i32 %62, 0, !dbg !39
  br i1 %63, label %L.LB2_332, label %L.LB2_333, !dbg !39

L.LB2_333:                                        ; preds = %L.LB2_415, %L.LB2_414
  br label %L.LB2_323

L.LB2_323:                                        ; preds = %L.LB2_333, %L.LB2_314
  %64 = load i32, i32* %__gtid___nv_MAIN__F1L20_1__384, align 4, !dbg !39
  call void @__kmpc_for_static_fini(i64* null, i32 %64), !dbg !39
  br label %L.LB2_318

L.LB2_318:                                        ; preds = %L.LB2_323
  ret void, !dbg !39
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare void @__kmpc_for_static_fini(i64*, i32) #0

declare void @__kmpc_for_static_init_4(i64*, i32, i32, i64*, i64*, i64*, i64*, i32, i32) #0

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
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB046-doall2-orig-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = !{!6, !12}
!6 = !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!7 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !3, type: !8, isLocal: true, isDefinition: true)
!8 = !DICompositeType(tag: DW_TAG_array_type, baseType: !9, size: 320000, align: 32, elements: !10)
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !{!11, !11}
!11 = !DISubrange(count: 100, lowerBound: 1)
!12 = !DIGlobalVariableExpression(var: !13, expr: !DIExpression())
!13 = distinct !DIGlobalVariable(name: "a", scope: !14, file: !3, type: !8, isLocal: true, isDefinition: true)
!14 = distinct !DISubprogram(name: "__nv_MAIN__F1L20_1", scope: !2, file: !3, line: 20, type: !15, scopeLine: 20, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!15 = !DISubroutineType(types: !16)
!16 = !{null, !9, !17, !17}
!17 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!18 = distinct !DISubprogram(name: "drb046_doall2_orig_no", scope: !2, file: !3, line: 13, type: !19, scopeLine: 13, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!19 = !DISubroutineType(cc: DW_CC_program, types: !20)
!20 = !{null}
!21 = !DILocalVariable(name: "omp_sched_static", scope: !18, file: !3, type: !9)
!22 = !DILocation(line: 0, scope: !18)
!23 = !DILocalVariable(name: "omp_proc_bind_false", scope: !18, file: !3, type: !9)
!24 = !DILocalVariable(name: "omp_proc_bind_true", scope: !18, file: !3, type: !9)
!25 = !DILocalVariable(name: "omp_lock_hint_none", scope: !18, file: !3, type: !9)
!26 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !18, file: !3, type: !9)
!27 = !DILocation(line: 27, column: 1, scope: !18)
!28 = !DILocation(line: 13, column: 1, scope: !18)
!29 = !DILocation(line: 20, column: 1, scope: !18)
!30 = !DILocalVariable(name: "__nv_MAIN__F1L20_1Arg0", arg: 1, scope: !14, file: !3, type: !9)
!31 = !DILocation(line: 0, scope: !14)
!32 = !DILocalVariable(name: "__nv_MAIN__F1L20_1Arg1", arg: 2, scope: !14, file: !3, type: !17)
!33 = !DILocalVariable(name: "__nv_MAIN__F1L20_1Arg2", arg: 3, scope: !14, file: !3, type: !17)
!34 = !DILocalVariable(name: "omp_sched_static", scope: !14, file: !3, type: !9)
!35 = !DILocalVariable(name: "omp_proc_bind_false", scope: !14, file: !3, type: !9)
!36 = !DILocalVariable(name: "omp_proc_bind_true", scope: !14, file: !3, type: !9)
!37 = !DILocalVariable(name: "omp_lock_hint_none", scope: !14, file: !3, type: !9)
!38 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !14, file: !3, type: !9)
!39 = !DILocation(line: 25, column: 1, scope: !14)
!40 = !DILocation(line: 21, column: 1, scope: !14)
!41 = !DILocalVariable(name: "i", scope: !14, file: !3, type: !9)
!42 = !DILocation(line: 22, column: 1, scope: !14)
!43 = !DILocalVariable(name: "j", scope: !14, file: !3, type: !9)
!44 = !DILocation(line: 23, column: 1, scope: !14)
!45 = !DILocation(line: 24, column: 1, scope: !14)

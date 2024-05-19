; ModuleID = '/tmp/DRB001-antidep1-orig-yes-dc2576.ll'
source_filename = "/tmp/DRB001-antidep1-orig-yes-dc2576.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.BSS1 = type <{ [4000 x i8] }>
%struct.STATICS1 = type <{ [44 x i8] }>
%astruct.dt63 = type <{ i8*, i8* }>

@.BSS1 = internal global %struct.BSS1 zeroinitializer, align 32, !dbg !0
@.STATICS1 = internal global %struct.STATICS1 <{ [44 x i8] c"\FB\FF\FF\FF\07\00\00\00a(500)=\00\EB\FF\FF\FF\00\00\00\00\03\00\00\00\00\00\00\00\01\00\00\00\FF\FF\FF\FF\00\00\00\00" }>, align 16
@.C305_MAIN_ = internal constant i32 25
@.C325_MAIN_ = internal constant i64 500
@.C284_MAIN_ = internal constant i64 0
@.C322_MAIN_ = internal constant i32 6
@.C318_MAIN_ = internal constant [53 x i8] c"micro-benchmarks-fortran/DRB001-antidep1-orig-yes.f95"
@.C320_MAIN_ = internal constant i32 29
@.C285_MAIN_ = internal constant i32 1
@.C308_MAIN_ = internal constant i32 1000
@.C283_MAIN_ = internal constant i32 0
@.C285___nv_MAIN__F1L23_1 = internal constant i32 1
@.C283___nv_MAIN__F1L23_1 = internal constant i32 0

define void @MAIN_() #0 !dbg !2 {
L.entry:
  %__gtid_MAIN__371 = alloca i32, align 4
  %len_310 = alloca i32, align 4
  %.dY0001_336 = alloca i32, align 4
  %i_306 = alloca i32, align 4
  %.uplevelArgPack0001_364 = alloca %astruct.dt63, align 16
  %z__io_324 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !21, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i32 0, metadata !23, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i32 1, metadata !24, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i32 0, metadata !25, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i32 1, metadata !26, metadata !DIExpression()), !dbg !22
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !27
  store i32 %0, i32* %__gtid_MAIN__371, align 4, !dbg !27
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !28
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !28
  call void (i8*, ...) %2(i8* %1), !dbg !28
  br label %L.LB1_350

L.LB1_350:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %len_310, metadata !29, metadata !DIExpression()), !dbg !22
  store i32 1000, i32* %len_310, align 4, !dbg !30
  %3 = load i32, i32* %len_310, align 4, !dbg !31
  call void @llvm.dbg.value(metadata i32 %3, metadata !29, metadata !DIExpression()), !dbg !22
  store i32 %3, i32* %.dY0001_336, align 4, !dbg !31
  call void @llvm.dbg.declare(metadata i32* %i_306, metadata !32, metadata !DIExpression()), !dbg !22
  store i32 1, i32* %i_306, align 4, !dbg !31
  %4 = load i32, i32* %.dY0001_336, align 4, !dbg !31
  %5 = icmp sle i32 %4, 0, !dbg !31
  br i1 %5, label %L.LB1_335, label %L.LB1_334, !dbg !31

L.LB1_334:                                        ; preds = %L.LB1_334, %L.LB1_350
  %6 = load i32, i32* %i_306, align 4, !dbg !33
  call void @llvm.dbg.value(metadata i32 %6, metadata !32, metadata !DIExpression()), !dbg !22
  %7 = load i32, i32* %i_306, align 4, !dbg !33
  call void @llvm.dbg.value(metadata i32 %7, metadata !32, metadata !DIExpression()), !dbg !22
  %8 = sext i32 %7 to i64, !dbg !33
  %9 = bitcast %struct.BSS1* @.BSS1 to i8*, !dbg !33
  %10 = getelementptr i8, i8* %9, i64 -4, !dbg !33
  %11 = bitcast i8* %10 to i32*, !dbg !33
  %12 = getelementptr i32, i32* %11, i64 %8, !dbg !33
  store i32 %6, i32* %12, align 4, !dbg !33
  %13 = load i32, i32* %i_306, align 4, !dbg !34
  call void @llvm.dbg.value(metadata i32 %13, metadata !32, metadata !DIExpression()), !dbg !22
  %14 = add nsw i32 %13, 1, !dbg !34
  store i32 %14, i32* %i_306, align 4, !dbg !34
  %15 = load i32, i32* %.dY0001_336, align 4, !dbg !34
  %16 = sub nsw i32 %15, 1, !dbg !34
  store i32 %16, i32* %.dY0001_336, align 4, !dbg !34
  %17 = load i32, i32* %.dY0001_336, align 4, !dbg !34
  %18 = icmp sgt i32 %17, 0, !dbg !34
  br i1 %18, label %L.LB1_334, label %L.LB1_335, !dbg !34

L.LB1_335:                                        ; preds = %L.LB1_334, %L.LB1_350
  %19 = bitcast i32* %len_310 to i8*, !dbg !35
  %20 = bitcast %astruct.dt63* %.uplevelArgPack0001_364 to i8**, !dbg !35
  store i8* %19, i8** %20, align 8, !dbg !35
  br label %L.LB1_369, !dbg !35

L.LB1_369:                                        ; preds = %L.LB1_335
  %21 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L23_1_ to i64*, !dbg !35
  %22 = bitcast %astruct.dt63* %.uplevelArgPack0001_364 to i64*, !dbg !35
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %21, i64* %22), !dbg !35
  call void (...) @_mp_bcs_nest(), !dbg !36
  %23 = bitcast i32* @.C320_MAIN_ to i8*, !dbg !36
  %24 = bitcast [53 x i8]* @.C318_MAIN_ to i8*, !dbg !36
  %25 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !36
  call void (i8*, i8*, i64, ...) %25(i8* %23, i8* %24, i64 53), !dbg !36
  %26 = bitcast i32* @.C322_MAIN_ to i8*, !dbg !36
  %27 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !36
  %28 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !36
  %29 = bitcast %struct.STATICS1* @.STATICS1 to i8*, !dbg !36
  %30 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !36
  %31 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %30(i8* %26, i8* null, i8* %27, i8* %28, i8* %29, i8* null, i64 0), !dbg !36
  call void @llvm.dbg.declare(metadata i32* %z__io_324, metadata !37, metadata !DIExpression()), !dbg !22
  store i32 %31, i32* %z__io_324, align 4, !dbg !36
  %32 = bitcast %struct.BSS1* @.BSS1 to i8*, !dbg !36
  %33 = getelementptr i8, i8* %32, i64 1996, !dbg !36
  %34 = bitcast i8* %33 to i32*, !dbg !36
  %35 = load i32, i32* %34, align 4, !dbg !36
  %36 = bitcast i32 (...)* @f90io_sc_i_fmt_write to i32 (i32, i32, ...)*, !dbg !36
  %37 = call i32 (i32, i32, ...) %36(i32 %35, i32 25), !dbg !36
  store i32 %37, i32* %z__io_324, align 4, !dbg !36
  %38 = call i32 (...) @f90io_fmtw_end(), !dbg !36
  store i32 %38, i32* %z__io_324, align 4, !dbg !36
  call void (...) @_mp_ecs_nest(), !dbg !36
  ret void, !dbg !27
}

define internal void @__nv_MAIN__F1L23_1_(i32* %__nv_MAIN__F1L23_1Arg0, i64* %__nv_MAIN__F1L23_1Arg1, i64* %__nv_MAIN__F1L23_1Arg2) #0 !dbg !9 {
L.entry:
  %__gtid___nv_MAIN__F1L23_1__421 = alloca i32, align 4
  %.i0000p_315 = alloca i32, align 4
  %i_314 = alloca i32, align 4
  %.du0002p_340 = alloca i32, align 4
  %.de0002p_341 = alloca i32, align 4
  %.di0002p_342 = alloca i32, align 4
  %.ds0002p_343 = alloca i32, align 4
  %.dl0002p_345 = alloca i32, align 4
  %.dl0002p.copy_415 = alloca i32, align 4
  %.de0002p.copy_416 = alloca i32, align 4
  %.ds0002p.copy_417 = alloca i32, align 4
  %.dX0002p_344 = alloca i32, align 4
  %.dY0002p_339 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L23_1Arg0, metadata !38, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L23_1Arg1, metadata !40, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L23_1Arg2, metadata !41, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.value(metadata i32 1, metadata !42, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.value(metadata i32 0, metadata !43, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.value(metadata i32 1, metadata !44, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.value(metadata i32 0, metadata !45, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.value(metadata i32 1, metadata !46, metadata !DIExpression()), !dbg !39
  %0 = load i32, i32* %__nv_MAIN__F1L23_1Arg0, align 4, !dbg !47
  store i32 %0, i32* %__gtid___nv_MAIN__F1L23_1__421, align 4, !dbg !47
  br label %L.LB2_406

L.LB2_406:                                        ; preds = %L.entry
  br label %L.LB2_313

L.LB2_313:                                        ; preds = %L.LB2_406
  store i32 0, i32* %.i0000p_315, align 4, !dbg !48
  call void @llvm.dbg.declare(metadata i32* %i_314, metadata !49, metadata !DIExpression()), !dbg !47
  store i32 1, i32* %i_314, align 4, !dbg !48
  %1 = bitcast i64* %__nv_MAIN__F1L23_1Arg2 to i32**, !dbg !48
  %2 = load i32*, i32** %1, align 8, !dbg !48
  %3 = load i32, i32* %2, align 4, !dbg !48
  %4 = sub nsw i32 %3, 1, !dbg !48
  store i32 %4, i32* %.du0002p_340, align 4, !dbg !48
  %5 = bitcast i64* %__nv_MAIN__F1L23_1Arg2 to i32**, !dbg !48
  %6 = load i32*, i32** %5, align 8, !dbg !48
  %7 = load i32, i32* %6, align 4, !dbg !48
  %8 = sub nsw i32 %7, 1, !dbg !48
  store i32 %8, i32* %.de0002p_341, align 4, !dbg !48
  store i32 1, i32* %.di0002p_342, align 4, !dbg !48
  %9 = load i32, i32* %.di0002p_342, align 4, !dbg !48
  store i32 %9, i32* %.ds0002p_343, align 4, !dbg !48
  store i32 1, i32* %.dl0002p_345, align 4, !dbg !48
  %10 = load i32, i32* %.dl0002p_345, align 4, !dbg !48
  store i32 %10, i32* %.dl0002p.copy_415, align 4, !dbg !48
  %11 = load i32, i32* %.de0002p_341, align 4, !dbg !48
  store i32 %11, i32* %.de0002p.copy_416, align 4, !dbg !48
  %12 = load i32, i32* %.ds0002p_343, align 4, !dbg !48
  store i32 %12, i32* %.ds0002p.copy_417, align 4, !dbg !48
  %13 = load i32, i32* %__gtid___nv_MAIN__F1L23_1__421, align 4, !dbg !48
  %14 = bitcast i32* %.i0000p_315 to i64*, !dbg !48
  %15 = bitcast i32* %.dl0002p.copy_415 to i64*, !dbg !48
  %16 = bitcast i32* %.de0002p.copy_416 to i64*, !dbg !48
  %17 = bitcast i32* %.ds0002p.copy_417 to i64*, !dbg !48
  %18 = load i32, i32* %.ds0002p.copy_417, align 4, !dbg !48
  call void @__kmpc_for_static_init_4(i64* null, i32 %13, i32 34, i64* %14, i64* %15, i64* %16, i64* %17, i32 %18, i32 1), !dbg !48
  %19 = load i32, i32* %.dl0002p.copy_415, align 4, !dbg !48
  store i32 %19, i32* %.dl0002p_345, align 4, !dbg !48
  %20 = load i32, i32* %.de0002p.copy_416, align 4, !dbg !48
  store i32 %20, i32* %.de0002p_341, align 4, !dbg !48
  %21 = load i32, i32* %.ds0002p.copy_417, align 4, !dbg !48
  store i32 %21, i32* %.ds0002p_343, align 4, !dbg !48
  %22 = load i32, i32* %.dl0002p_345, align 4, !dbg !48
  store i32 %22, i32* %i_314, align 4, !dbg !48
  %23 = load i32, i32* %i_314, align 4, !dbg !48
  call void @llvm.dbg.value(metadata i32 %23, metadata !49, metadata !DIExpression()), !dbg !47
  store i32 %23, i32* %.dX0002p_344, align 4, !dbg !48
  %24 = load i32, i32* %.dX0002p_344, align 4, !dbg !48
  %25 = load i32, i32* %.du0002p_340, align 4, !dbg !48
  %26 = icmp sgt i32 %24, %25, !dbg !48
  br i1 %26, label %L.LB2_338, label %L.LB2_443, !dbg !48

L.LB2_443:                                        ; preds = %L.LB2_313
  %27 = load i32, i32* %.dX0002p_344, align 4, !dbg !48
  store i32 %27, i32* %i_314, align 4, !dbg !48
  %28 = load i32, i32* %.di0002p_342, align 4, !dbg !48
  %29 = load i32, i32* %.de0002p_341, align 4, !dbg !48
  %30 = load i32, i32* %.dX0002p_344, align 4, !dbg !48
  %31 = sub nsw i32 %29, %30, !dbg !48
  %32 = add nsw i32 %28, %31, !dbg !48
  %33 = load i32, i32* %.di0002p_342, align 4, !dbg !48
  %34 = sdiv i32 %32, %33, !dbg !48
  store i32 %34, i32* %.dY0002p_339, align 4, !dbg !48
  %35 = load i32, i32* %.dY0002p_339, align 4, !dbg !48
  %36 = icmp sle i32 %35, 0, !dbg !48
  br i1 %36, label %L.LB2_348, label %L.LB2_347, !dbg !48

L.LB2_347:                                        ; preds = %L.LB2_347, %L.LB2_443
  %37 = load i32, i32* %i_314, align 4, !dbg !50
  call void @llvm.dbg.value(metadata i32 %37, metadata !49, metadata !DIExpression()), !dbg !47
  %38 = sext i32 %37 to i64, !dbg !50
  %39 = bitcast %struct.BSS1* @.BSS1 to i32*, !dbg !50
  %40 = getelementptr i32, i32* %39, i64 %38, !dbg !50
  %41 = load i32, i32* %40, align 4, !dbg !50
  %42 = add nsw i32 %41, 1, !dbg !50
  %43 = load i32, i32* %i_314, align 4, !dbg !50
  call void @llvm.dbg.value(metadata i32 %43, metadata !49, metadata !DIExpression()), !dbg !47
  %44 = sext i32 %43 to i64, !dbg !50
  %45 = bitcast %struct.BSS1* @.BSS1 to i8*, !dbg !50
  %46 = getelementptr i8, i8* %45, i64 -4, !dbg !50
  %47 = bitcast i8* %46 to i32*, !dbg !50
  %48 = getelementptr i32, i32* %47, i64 %44, !dbg !50
  store i32 %42, i32* %48, align 4, !dbg !50
  %49 = load i32, i32* %.di0002p_342, align 4, !dbg !47
  %50 = load i32, i32* %i_314, align 4, !dbg !47
  call void @llvm.dbg.value(metadata i32 %50, metadata !49, metadata !DIExpression()), !dbg !47
  %51 = add nsw i32 %49, %50, !dbg !47
  store i32 %51, i32* %i_314, align 4, !dbg !47
  %52 = load i32, i32* %.dY0002p_339, align 4, !dbg !47
  %53 = sub nsw i32 %52, 1, !dbg !47
  store i32 %53, i32* %.dY0002p_339, align 4, !dbg !47
  %54 = load i32, i32* %.dY0002p_339, align 4, !dbg !47
  %55 = icmp sgt i32 %54, 0, !dbg !47
  br i1 %55, label %L.LB2_347, label %L.LB2_348, !dbg !47

L.LB2_348:                                        ; preds = %L.LB2_347, %L.LB2_443
  br label %L.LB2_338

L.LB2_338:                                        ; preds = %L.LB2_348, %L.LB2_313
  %56 = load i32, i32* %__gtid___nv_MAIN__F1L23_1__421, align 4, !dbg !47
  call void @__kmpc_for_static_fini(i64* null, i32 %56), !dbg !47
  br label %L.LB2_316

L.LB2_316:                                        ; preds = %L.LB2_338
  ret void, !dbg !47
}

declare void @__kmpc_for_static_fini(i64*, i32) #0

declare void @__kmpc_for_static_init_4(i64*, i32, i32, i64*, i64*, i64*, i64*, i32, i32) #0

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

!llvm.module.flags = !{!19, !20}
!llvm.dbg.cu = !{!4}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !3, type: !14, isLocal: true, isDefinition: true)
!2 = distinct !DISubprogram(name: "drb001_antidep1_orig_yes", scope: !4, file: !3, line: 11, type: !17, scopeLine: 11, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !4)
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB001-antidep1-orig-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !6, imports: !5)
!5 = !{}
!6 = !{!0, !7}
!7 = !DIGlobalVariableExpression(var: !8, expr: !DIExpression())
!8 = distinct !DIGlobalVariable(name: "a", scope: !9, file: !3, type: !14, isLocal: true, isDefinition: true)
!9 = distinct !DISubprogram(name: "__nv_MAIN__F1L23_1", scope: !4, file: !3, line: 23, type: !10, scopeLine: 23, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !4)
!10 = !DISubroutineType(types: !11)
!11 = !{null, !12, !13, !13}
!12 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!13 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!14 = !DICompositeType(tag: DW_TAG_array_type, baseType: !12, size: 32000, align: 32, elements: !15)
!15 = !{!16}
!16 = !DISubrange(count: 1000, lowerBound: 1)
!17 = !DISubroutineType(cc: DW_CC_program, types: !18)
!18 = !{null}
!19 = !{i32 2, !"Dwarf Version", i32 4}
!20 = !{i32 2, !"Debug Info Version", i32 3}
!21 = !DILocalVariable(name: "omp_sched_static", scope: !2, file: !3, type: !12)
!22 = !DILocation(line: 0, scope: !2)
!23 = !DILocalVariable(name: "omp_proc_bind_false", scope: !2, file: !3, type: !12)
!24 = !DILocalVariable(name: "omp_proc_bind_true", scope: !2, file: !3, type: !12)
!25 = !DILocalVariable(name: "omp_lock_hint_none", scope: !2, file: !3, type: !12)
!26 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !2, file: !3, type: !12)
!27 = !DILocation(line: 31, column: 1, scope: !2)
!28 = !DILocation(line: 11, column: 1, scope: !2)
!29 = !DILocalVariable(name: "len", scope: !2, file: !3, type: !12)
!30 = !DILocation(line: 17, column: 1, scope: !2)
!31 = !DILocation(line: 19, column: 1, scope: !2)
!32 = !DILocalVariable(name: "i", scope: !2, file: !3, type: !12)
!33 = !DILocation(line: 20, column: 1, scope: !2)
!34 = !DILocation(line: 21, column: 1, scope: !2)
!35 = !DILocation(line: 23, column: 1, scope: !2)
!36 = !DILocation(line: 29, column: 1, scope: !2)
!37 = !DILocalVariable(scope: !2, file: !3, type: !12, flags: DIFlagArtificial)
!38 = !DILocalVariable(name: "__nv_MAIN__F1L23_1Arg0", arg: 1, scope: !9, file: !3, type: !12)
!39 = !DILocation(line: 0, scope: !9)
!40 = !DILocalVariable(name: "__nv_MAIN__F1L23_1Arg1", arg: 2, scope: !9, file: !3, type: !13)
!41 = !DILocalVariable(name: "__nv_MAIN__F1L23_1Arg2", arg: 3, scope: !9, file: !3, type: !13)
!42 = !DILocalVariable(name: "omp_sched_static", scope: !9, file: !3, type: !12)
!43 = !DILocalVariable(name: "omp_proc_bind_false", scope: !9, file: !3, type: !12)
!44 = !DILocalVariable(name: "omp_proc_bind_true", scope: !9, file: !3, type: !12)
!45 = !DILocalVariable(name: "omp_lock_hint_none", scope: !9, file: !3, type: !12)
!46 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !9, file: !3, type: !12)
!47 = !DILocation(line: 26, column: 1, scope: !9)
!48 = !DILocation(line: 24, column: 1, scope: !9)
!49 = !DILocalVariable(name: "i", scope: !9, file: !3, type: !12)
!50 = !DILocation(line: 25, column: 1, scope: !9)

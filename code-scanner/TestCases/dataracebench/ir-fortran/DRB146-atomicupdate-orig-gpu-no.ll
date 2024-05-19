; ModuleID = '/tmp/DRB146-atomicupdate-orig-gpu-no-21941a.ll'
source_filename = "/tmp/DRB146-atomicupdate-orig-gpu-no-21941a.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%astruct.dt58 = type <{ i8* }>
%astruct.dt100 = type <{ [8 x i8] }>

@.C305_MAIN_ = internal constant i32 25
@.C284_MAIN_ = internal constant i64 0
@.C325_MAIN_ = internal constant i32 6
@.C323_MAIN_ = internal constant [60 x i8] c"micro-benchmarks-fortran/DRB146-atomicupdate-orig-gpu-no.f95"
@.C306_MAIN_ = internal constant i32 28
@.C316_MAIN_ = internal constant i32 100
@.C285_MAIN_ = internal constant i32 1
@.C283_MAIN_ = internal constant i32 0
@.C316___nv_MAIN__F1L18_1 = internal constant i32 100
@.C285___nv_MAIN__F1L18_1 = internal constant i32 1
@.C283___nv_MAIN__F1L18_1 = internal constant i32 0
@.C316___nv_MAIN_F1L19_2 = internal constant i32 100
@.C285___nv_MAIN_F1L19_2 = internal constant i32 1
@.C283___nv_MAIN_F1L19_2 = internal constant i32 0

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__357 = alloca i32, align 4
  %var_307 = alloca i32, align 4
  %.uplevelArgPack0001_354 = alloca %astruct.dt58, align 8
  %z__io_327 = alloca i32, align 4
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
  br label %L.LB1_348

L.LB1_348:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %var_307, metadata !17, metadata !DIExpression()), !dbg !10
  store i32 0, i32* %var_307, align 4, !dbg !18
  %3 = bitcast i32* %var_307 to i8*, !dbg !19
  %4 = bitcast %astruct.dt58* %.uplevelArgPack0001_354 to i8**, !dbg !19
  store i8* %3, i8** %4, align 8, !dbg !19
  %5 = bitcast %astruct.dt58* %.uplevelArgPack0001_354 to i64*, !dbg !19
  call void @__nv_MAIN__F1L18_1_(i32* %__gtid_MAIN__357, i64* null, i64* %5), !dbg !19
  call void (...) @_mp_bcs_nest(), !dbg !20
  %6 = bitcast i32* @.C306_MAIN_ to i8*, !dbg !20
  %7 = bitcast [60 x i8]* @.C323_MAIN_ to i8*, !dbg !20
  %8 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !20
  call void (i8*, i8*, i64, ...) %8(i8* %6, i8* %7, i64 60), !dbg !20
  %9 = bitcast i32* @.C325_MAIN_ to i8*, !dbg !20
  %10 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !20
  %11 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !20
  %12 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !20
  %13 = call i32 (i8*, i8*, i8*, i8*, ...) %12(i8* %9, i8* null, i8* %10, i8* %11), !dbg !20
  call void @llvm.dbg.declare(metadata i32* %z__io_327, metadata !21, metadata !DIExpression()), !dbg !10
  store i32 %13, i32* %z__io_327, align 4, !dbg !20
  %14 = load i32, i32* %var_307, align 4, !dbg !20
  call void @llvm.dbg.value(metadata i32 %14, metadata !17, metadata !DIExpression()), !dbg !10
  %15 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !20
  %16 = call i32 (i32, i32, ...) %15(i32 %14, i32 25), !dbg !20
  store i32 %16, i32* %z__io_327, align 4, !dbg !20
  %17 = call i32 (...) @f90io_ldw_end(), !dbg !20
  store i32 %17, i32* %z__io_327, align 4, !dbg !20
  call void (...) @_mp_ecs_nest(), !dbg !20
  ret void, !dbg !15
}

define internal void @__nv_MAIN__F1L18_1_(i32* %__nv_MAIN__F1L18_1Arg0, i64* %__nv_MAIN__F1L18_1Arg1, i64* %__nv_MAIN__F1L18_1Arg2) #0 !dbg !22 {
L.entry:
  %.uplevelArgPack0002_377 = alloca %astruct.dt100, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L18_1Arg0, metadata !26, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L18_1Arg1, metadata !28, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L18_1Arg2, metadata !29, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.value(metadata i32 1, metadata !30, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.value(metadata i32 0, metadata !31, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.value(metadata i32 1, metadata !32, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.value(metadata i32 0, metadata !33, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.value(metadata i32 1, metadata !34, metadata !DIExpression()), !dbg !27
  br label %L.LB2_372

L.LB2_372:                                        ; preds = %L.entry
  br label %L.LB2_311

L.LB2_311:                                        ; preds = %L.LB2_372
  %0 = load i64, i64* %__nv_MAIN__F1L18_1Arg2, align 8, !dbg !35
  %1 = bitcast %astruct.dt100* %.uplevelArgPack0002_377 to i64*, !dbg !35
  store i64 %0, i64* %1, align 8, !dbg !35
  %2 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN_F1L19_2_ to i64*, !dbg !35
  %3 = bitcast %astruct.dt100* %.uplevelArgPack0002_377 to i64*, !dbg !35
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_teams(i64* null, i32 1, i64* %2, i64* %3), !dbg !35
  br label %L.LB2_321

L.LB2_321:                                        ; preds = %L.LB2_311
  ret void, !dbg !36
}

define internal void @__nv_MAIN_F1L19_2_(i32* %__nv_MAIN_F1L19_2Arg0, i64* %__nv_MAIN_F1L19_2Arg1, i64* %__nv_MAIN_F1L19_2Arg2) #0 !dbg !37 {
L.entry:
  %__gtid___nv_MAIN_F1L19_2__409 = alloca i32, align 4
  %.i0000p_318 = alloca i32, align 4
  %i_317 = alloca i32, align 4
  %.du0001_338 = alloca i32, align 4
  %.de0001_339 = alloca i32, align 4
  %.di0001_340 = alloca i32, align 4
  %.ds0001_341 = alloca i32, align 4
  %.dl0001_343 = alloca i32, align 4
  %.dl0001.copy_403 = alloca i32, align 4
  %.de0001.copy_404 = alloca i32, align 4
  %.ds0001.copy_405 = alloca i32, align 4
  %.dX0001_342 = alloca i32, align 4
  %.dY0001_337 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L19_2Arg0, metadata !38, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L19_2Arg1, metadata !40, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L19_2Arg2, metadata !41, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.value(metadata i32 1, metadata !42, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.value(metadata i32 0, metadata !43, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.value(metadata i32 1, metadata !44, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.value(metadata i32 0, metadata !45, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.value(metadata i32 1, metadata !46, metadata !DIExpression()), !dbg !39
  %0 = load i32, i32* %__nv_MAIN_F1L19_2Arg0, align 4, !dbg !47
  store i32 %0, i32* %__gtid___nv_MAIN_F1L19_2__409, align 4, !dbg !47
  br label %L.LB4_395

L.LB4_395:                                        ; preds = %L.entry
  br label %L.LB4_314

L.LB4_314:                                        ; preds = %L.LB4_395
  br label %L.LB4_315

L.LB4_315:                                        ; preds = %L.LB4_314
  store i32 0, i32* %.i0000p_318, align 4, !dbg !48
  call void @llvm.dbg.declare(metadata i32* %i_317, metadata !49, metadata !DIExpression()), !dbg !47
  store i32 1, i32* %i_317, align 4, !dbg !48
  store i32 100, i32* %.du0001_338, align 4, !dbg !48
  store i32 100, i32* %.de0001_339, align 4, !dbg !48
  store i32 1, i32* %.di0001_340, align 4, !dbg !48
  %1 = load i32, i32* %.di0001_340, align 4, !dbg !48
  store i32 %1, i32* %.ds0001_341, align 4, !dbg !48
  store i32 1, i32* %.dl0001_343, align 4, !dbg !48
  %2 = load i32, i32* %.dl0001_343, align 4, !dbg !48
  store i32 %2, i32* %.dl0001.copy_403, align 4, !dbg !48
  %3 = load i32, i32* %.de0001_339, align 4, !dbg !48
  store i32 %3, i32* %.de0001.copy_404, align 4, !dbg !48
  %4 = load i32, i32* %.ds0001_341, align 4, !dbg !48
  store i32 %4, i32* %.ds0001.copy_405, align 4, !dbg !48
  %5 = load i32, i32* %__gtid___nv_MAIN_F1L19_2__409, align 4, !dbg !48
  %6 = bitcast i32* %.i0000p_318 to i64*, !dbg !48
  %7 = bitcast i32* %.dl0001.copy_403 to i64*, !dbg !48
  %8 = bitcast i32* %.de0001.copy_404 to i64*, !dbg !48
  %9 = bitcast i32* %.ds0001.copy_405 to i64*, !dbg !48
  %10 = load i32, i32* %.ds0001.copy_405, align 4, !dbg !48
  call void @__kmpc_for_static_init_4(i64* null, i32 %5, i32 92, i64* %6, i64* %7, i64* %8, i64* %9, i32 %10, i32 1), !dbg !48
  %11 = load i32, i32* %.dl0001.copy_403, align 4, !dbg !48
  store i32 %11, i32* %.dl0001_343, align 4, !dbg !48
  %12 = load i32, i32* %.de0001.copy_404, align 4, !dbg !48
  store i32 %12, i32* %.de0001_339, align 4, !dbg !48
  %13 = load i32, i32* %.ds0001.copy_405, align 4, !dbg !48
  store i32 %13, i32* %.ds0001_341, align 4, !dbg !48
  %14 = load i32, i32* %.dl0001_343, align 4, !dbg !48
  store i32 %14, i32* %i_317, align 4, !dbg !48
  %15 = load i32, i32* %i_317, align 4, !dbg !48
  call void @llvm.dbg.value(metadata i32 %15, metadata !49, metadata !DIExpression()), !dbg !47
  store i32 %15, i32* %.dX0001_342, align 4, !dbg !48
  %16 = load i32, i32* %.dX0001_342, align 4, !dbg !48
  %17 = load i32, i32* %.du0001_338, align 4, !dbg !48
  %18 = icmp sgt i32 %16, %17, !dbg !48
  br i1 %18, label %L.LB4_336, label %L.LB4_434, !dbg !48

L.LB4_434:                                        ; preds = %L.LB4_315
  %19 = load i32, i32* %.du0001_338, align 4, !dbg !48
  %20 = load i32, i32* %.de0001_339, align 4, !dbg !48
  %21 = icmp slt i32 %19, %20, !dbg !48
  %22 = select i1 %21, i32 %19, i32 %20, !dbg !48
  store i32 %22, i32* %.de0001_339, align 4, !dbg !48
  %23 = load i32, i32* %.dX0001_342, align 4, !dbg !48
  store i32 %23, i32* %i_317, align 4, !dbg !48
  %24 = load i32, i32* %.di0001_340, align 4, !dbg !48
  %25 = load i32, i32* %.de0001_339, align 4, !dbg !48
  %26 = load i32, i32* %.dX0001_342, align 4, !dbg !48
  %27 = sub nsw i32 %25, %26, !dbg !48
  %28 = add nsw i32 %24, %27, !dbg !48
  %29 = load i32, i32* %.di0001_340, align 4, !dbg !48
  %30 = sdiv i32 %28, %29, !dbg !48
  store i32 %30, i32* %.dY0001_337, align 4, !dbg !48
  %31 = load i32, i32* %.dY0001_337, align 4, !dbg !48
  %32 = icmp sle i32 %31, 0, !dbg !48
  br i1 %32, label %L.LB4_346, label %L.LB4_345, !dbg !48

L.LB4_345:                                        ; preds = %L.LB4_345, %L.LB4_434
  %33 = call i32 (...) @_mp_bcs_nest_red(), !dbg !50
  %34 = bitcast i64* %__nv_MAIN_F1L19_2Arg2 to i32**, !dbg !50
  %35 = load i32*, i32** %34, align 8, !dbg !50
  %36 = load i32, i32* %35, align 4, !dbg !50
  %37 = add nsw i32 %36, 1, !dbg !50
  %38 = bitcast i64* %__nv_MAIN_F1L19_2Arg2 to i32**, !dbg !50
  %39 = load i32*, i32** %38, align 8, !dbg !50
  store i32 %37, i32* %39, align 4, !dbg !50
  %40 = call i32 (...) @_mp_ecs_nest_red(), !dbg !50
  %41 = load i32, i32* %.di0001_340, align 4, !dbg !51
  %42 = load i32, i32* %i_317, align 4, !dbg !51
  call void @llvm.dbg.value(metadata i32 %42, metadata !49, metadata !DIExpression()), !dbg !47
  %43 = add nsw i32 %41, %42, !dbg !51
  store i32 %43, i32* %i_317, align 4, !dbg !51
  %44 = load i32, i32* %.dY0001_337, align 4, !dbg !51
  %45 = sub nsw i32 %44, 1, !dbg !51
  store i32 %45, i32* %.dY0001_337, align 4, !dbg !51
  %46 = load i32, i32* %.dY0001_337, align 4, !dbg !51
  %47 = icmp sgt i32 %46, 0, !dbg !51
  br i1 %47, label %L.LB4_345, label %L.LB4_346, !dbg !51

L.LB4_346:                                        ; preds = %L.LB4_345, %L.LB4_434
  br label %L.LB4_336

L.LB4_336:                                        ; preds = %L.LB4_346, %L.LB4_315
  %48 = load i32, i32* %__gtid___nv_MAIN_F1L19_2__409, align 4, !dbg !51
  call void @__kmpc_for_static_fini(i64* null, i32 %48), !dbg !51
  br label %L.LB4_319

L.LB4_319:                                        ; preds = %L.LB4_336
  br label %L.LB4_320

L.LB4_320:                                        ; preds = %L.LB4_319
  ret void, !dbg !47
}

declare signext i32 @_mp_ecs_nest_red(...) #0

declare signext i32 @_mp_bcs_nest_red(...) #0

declare void @__kmpc_for_static_fini(i64*, i32) #0

declare void @__kmpc_for_static_init_4(i64*, i32, i32, i64*, i64*, i64*, i64*, i32, i32) #0

declare void @__kmpc_fork_teams(i64*, i32, i64*, i64*, ...) #0

declare void @_mp_ecs_nest(...) #0

declare signext i32 @f90io_ldw_end(...) #0

declare signext i32 @f90io_sc_i_ldw(...) #0

declare signext i32 @f90io_print_init(...) #0

declare void @f90io_src_info03a(...) #0

declare void @_mp_bcs_nest(...) #0

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare void @fort_init(...) #0

declare signext i32 @__kmpc_global_thread_num(i64*) #0

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { "no-frame-pointer-elim-non-leaf" }
attributes #1 = { nounwind readnone speculatable }

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !4, globals: !4, imports: !4)
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB146-atomicupdate-orig-gpu-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb146_atomicupdate_orig_gpu_no", scope: !2, file: !3, line: 11, type: !6, scopeLine: 11, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
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
!16 = !DILocation(line: 11, column: 1, scope: !5)
!17 = !DILocalVariable(name: "var", scope: !5, file: !3, type: !9)
!18 = !DILocation(line: 16, column: 1, scope: !5)
!19 = !DILocation(line: 26, column: 1, scope: !5)
!20 = !DILocation(line: 28, column: 1, scope: !5)
!21 = !DILocalVariable(scope: !5, file: !3, type: !9, flags: DIFlagArtificial)
!22 = distinct !DISubprogram(name: "__nv_MAIN__F1L18_1", scope: !2, file: !3, line: 18, type: !23, scopeLine: 18, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!23 = !DISubroutineType(types: !24)
!24 = !{null, !9, !25, !25}
!25 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!26 = !DILocalVariable(name: "__nv_MAIN__F1L18_1Arg0", arg: 1, scope: !22, file: !3, type: !9)
!27 = !DILocation(line: 0, scope: !22)
!28 = !DILocalVariable(name: "__nv_MAIN__F1L18_1Arg1", arg: 2, scope: !22, file: !3, type: !25)
!29 = !DILocalVariable(name: "__nv_MAIN__F1L18_1Arg2", arg: 3, scope: !22, file: !3, type: !25)
!30 = !DILocalVariable(name: "omp_sched_static", scope: !22, file: !3, type: !9)
!31 = !DILocalVariable(name: "omp_proc_bind_false", scope: !22, file: !3, type: !9)
!32 = !DILocalVariable(name: "omp_proc_bind_true", scope: !22, file: !3, type: !9)
!33 = !DILocalVariable(name: "omp_lock_hint_none", scope: !22, file: !3, type: !9)
!34 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !22, file: !3, type: !9)
!35 = !DILocation(line: 19, column: 1, scope: !22)
!36 = !DILocation(line: 26, column: 1, scope: !22)
!37 = distinct !DISubprogram(name: "__nv_MAIN_F1L19_2", scope: !2, file: !3, line: 19, type: !23, scopeLine: 19, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!38 = !DILocalVariable(name: "__nv_MAIN_F1L19_2Arg0", arg: 1, scope: !37, file: !3, type: !9)
!39 = !DILocation(line: 0, scope: !37)
!40 = !DILocalVariable(name: "__nv_MAIN_F1L19_2Arg1", arg: 2, scope: !37, file: !3, type: !25)
!41 = !DILocalVariable(name: "__nv_MAIN_F1L19_2Arg2", arg: 3, scope: !37, file: !3, type: !25)
!42 = !DILocalVariable(name: "omp_sched_static", scope: !37, file: !3, type: !9)
!43 = !DILocalVariable(name: "omp_proc_bind_false", scope: !37, file: !3, type: !9)
!44 = !DILocalVariable(name: "omp_proc_bind_true", scope: !37, file: !3, type: !9)
!45 = !DILocalVariable(name: "omp_lock_hint_none", scope: !37, file: !3, type: !9)
!46 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !37, file: !3, type: !9)
!47 = !DILocation(line: 25, column: 1, scope: !37)
!48 = !DILocation(line: 20, column: 1, scope: !37)
!49 = !DILocalVariable(name: "i", scope: !37, file: !3, type: !9)
!50 = !DILocation(line: 22, column: 1, scope: !37)
!51 = !DILocation(line: 24, column: 1, scope: !37)

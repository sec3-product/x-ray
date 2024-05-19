; ModuleID = '/tmp/DRB154-missinglock3-orig-gpu-no-55ea55.ll'
source_filename = "/tmp/DRB154-missinglock3-orig-gpu-no-55ea55.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%astruct.dt70 = type <{ i8*, i8* }>
%astruct.dt112 = type <{ [16 x i8] }>

@.C305_MAIN_ = internal constant i32 25
@.C284_MAIN_ = internal constant i64 0
@.C335_MAIN_ = internal constant i32 6
@.C332_MAIN_ = internal constant [60 x i8] c"micro-benchmarks-fortran/DRB154-missinglock3-orig-gpu-no.f95"
@.C334_MAIN_ = internal constant i32 33
@.C325_MAIN_ = internal constant i32 100
@.C285_MAIN_ = internal constant i32 1
@.C283_MAIN_ = internal constant i32 0
@.C325___nv_MAIN__F1L21_1 = internal constant i32 100
@.C285___nv_MAIN__F1L21_1 = internal constant i32 1
@.C283___nv_MAIN__F1L21_1 = internal constant i32 0
@.C325___nv_MAIN_F1L22_2 = internal constant i32 100
@.C285___nv_MAIN_F1L22_2 = internal constant i32 1
@.C283___nv_MAIN_F1L22_2 = internal constant i32 0

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__370 = alloca i32, align 4
  %var_315 = alloca i32, align 4
  %lck_314 = alloca i32, align 4
  %.uplevelArgPack0001_365 = alloca %astruct.dt70, align 16
  %z__io_337 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !14, metadata !DIExpression()), !dbg !10
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !15
  store i32 %0, i32* %__gtid_MAIN__370, align 4, !dbg !15
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !16
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !16
  call void (i8*, ...) %2(i8* %1), !dbg !16
  br label %L.LB1_358

L.LB1_358:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %var_315, metadata !17, metadata !DIExpression()), !dbg !10
  store i32 0, i32* %var_315, align 4, !dbg !18
  call void @llvm.dbg.declare(metadata i32* %lck_314, metadata !19, metadata !DIExpression()), !dbg !10
  %3 = bitcast i32* %lck_314 to i64*, !dbg !20
  call void @omp_init_lock_(i64* %3), !dbg !20
  %4 = bitcast i32* %var_315 to i8*, !dbg !21
  %5 = bitcast %astruct.dt70* %.uplevelArgPack0001_365 to i8**, !dbg !21
  store i8* %4, i8** %5, align 8, !dbg !21
  %6 = bitcast i32* %lck_314 to i8*, !dbg !21
  %7 = bitcast %astruct.dt70* %.uplevelArgPack0001_365 to i8*, !dbg !21
  %8 = getelementptr i8, i8* %7, i64 8, !dbg !21
  %9 = bitcast i8* %8 to i8**, !dbg !21
  store i8* %6, i8** %9, align 8, !dbg !21
  %10 = bitcast %astruct.dt70* %.uplevelArgPack0001_365 to i64*, !dbg !21
  call void @__nv_MAIN__F1L21_1_(i32* %__gtid_MAIN__370, i64* null, i64* %10), !dbg !21
  %11 = bitcast i32* %lck_314 to i64*, !dbg !22
  call void @omp_destroy_lock_(i64* %11), !dbg !22
  call void (...) @_mp_bcs_nest(), !dbg !23
  %12 = bitcast i32* @.C334_MAIN_ to i8*, !dbg !23
  %13 = bitcast [60 x i8]* @.C332_MAIN_ to i8*, !dbg !23
  %14 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !23
  call void (i8*, i8*, i64, ...) %14(i8* %12, i8* %13, i64 60), !dbg !23
  %15 = bitcast i32* @.C335_MAIN_ to i8*, !dbg !23
  %16 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !23
  %17 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !23
  %18 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !23
  %19 = call i32 (i8*, i8*, i8*, i8*, ...) %18(i8* %15, i8* null, i8* %16, i8* %17), !dbg !23
  call void @llvm.dbg.declare(metadata i32* %z__io_337, metadata !24, metadata !DIExpression()), !dbg !10
  store i32 %19, i32* %z__io_337, align 4, !dbg !23
  %20 = load i32, i32* %var_315, align 4, !dbg !23
  call void @llvm.dbg.value(metadata i32 %20, metadata !17, metadata !DIExpression()), !dbg !10
  %21 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !23
  %22 = call i32 (i32, i32, ...) %21(i32 %20, i32 25), !dbg !23
  store i32 %22, i32* %z__io_337, align 4, !dbg !23
  %23 = call i32 (...) @f90io_ldw_end(), !dbg !23
  store i32 %23, i32* %z__io_337, align 4, !dbg !23
  call void (...) @_mp_ecs_nest(), !dbg !23
  ret void, !dbg !15
}

define internal void @__nv_MAIN__F1L21_1_(i32* %__nv_MAIN__F1L21_1Arg0, i64* %__nv_MAIN__F1L21_1Arg1, i64* %__nv_MAIN__F1L21_1Arg2) #0 !dbg !25 {
L.entry:
  %.uplevelArgPack0002_390 = alloca %astruct.dt112, align 16
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L21_1Arg0, metadata !29, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L21_1Arg1, metadata !31, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L21_1Arg2, metadata !32, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.value(metadata i32 1, metadata !33, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.value(metadata i32 0, metadata !34, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.value(metadata i32 1, metadata !35, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.value(metadata i32 0, metadata !36, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.value(metadata i32 1, metadata !37, metadata !DIExpression()), !dbg !30
  br label %L.LB2_385

L.LB2_385:                                        ; preds = %L.entry
  br label %L.LB2_319

L.LB2_319:                                        ; preds = %L.LB2_385
  %0 = load i64, i64* %__nv_MAIN__F1L21_1Arg2, align 8, !dbg !38
  %1 = bitcast %astruct.dt112* %.uplevelArgPack0002_390 to i64*, !dbg !38
  store i64 %0, i64* %1, align 8, !dbg !38
  %2 = bitcast i64* %__nv_MAIN__F1L21_1Arg2 to i8*, !dbg !39
  %3 = getelementptr i8, i8* %2, i64 8, !dbg !39
  %4 = bitcast i8* %3 to i64*, !dbg !39
  %5 = load i64, i64* %4, align 8, !dbg !39
  %6 = bitcast %astruct.dt112* %.uplevelArgPack0002_390 to i8*, !dbg !39
  %7 = getelementptr i8, i8* %6, i64 8, !dbg !39
  %8 = bitcast i8* %7 to i64*, !dbg !39
  store i64 %5, i64* %8, align 8, !dbg !39
  %9 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN_F1L22_2_ to i64*, !dbg !38
  %10 = bitcast %astruct.dt112* %.uplevelArgPack0002_390 to i64*, !dbg !38
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_teams(i64* null, i32 1, i64* %9, i64* %10), !dbg !38
  br label %L.LB2_330

L.LB2_330:                                        ; preds = %L.LB2_319
  ret void, !dbg !39
}

define internal void @__nv_MAIN_F1L22_2_(i32* %__nv_MAIN_F1L22_2Arg0, i64* %__nv_MAIN_F1L22_2Arg1, i64* %__nv_MAIN_F1L22_2Arg2) #0 !dbg !40 {
L.entry:
  %__gtid___nv_MAIN_F1L22_2__425 = alloca i32, align 4
  %var_323 = alloca i32, align 4
  %.i0000p_327 = alloca i32, align 4
  %i_326 = alloca i32, align 4
  %.du0001_348 = alloca i32, align 4
  %.de0001_349 = alloca i32, align 4
  %.di0001_350 = alloca i32, align 4
  %.ds0001_351 = alloca i32, align 4
  %.dl0001_353 = alloca i32, align 4
  %.dl0001.copy_419 = alloca i32, align 4
  %.de0001.copy_420 = alloca i32, align 4
  %.ds0001.copy_421 = alloca i32, align 4
  %.dX0001_352 = alloca i32, align 4
  %.dY0001_347 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L22_2Arg0, metadata !41, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L22_2Arg1, metadata !43, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L22_2Arg2, metadata !44, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.value(metadata i32 1, metadata !45, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.value(metadata i32 0, metadata !46, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.value(metadata i32 1, metadata !47, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.value(metadata i32 0, metadata !48, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.value(metadata i32 1, metadata !49, metadata !DIExpression()), !dbg !42
  %0 = load i32, i32* %__nv_MAIN_F1L22_2Arg0, align 4, !dbg !50
  store i32 %0, i32* %__gtid___nv_MAIN_F1L22_2__425, align 4, !dbg !50
  br label %L.LB4_410

L.LB4_410:                                        ; preds = %L.entry
  br label %L.LB4_322

L.LB4_322:                                        ; preds = %L.LB4_410
  call void @llvm.dbg.declare(metadata i32* %var_323, metadata !51, metadata !DIExpression()), !dbg !50
  store i32 0, i32* %var_323, align 4, !dbg !52
  br label %L.LB4_324

L.LB4_324:                                        ; preds = %L.LB4_322
  store i32 0, i32* %.i0000p_327, align 4, !dbg !53
  call void @llvm.dbg.declare(metadata i32* %i_326, metadata !54, metadata !DIExpression()), !dbg !50
  store i32 1, i32* %i_326, align 4, !dbg !53
  store i32 100, i32* %.du0001_348, align 4, !dbg !53
  store i32 100, i32* %.de0001_349, align 4, !dbg !53
  store i32 1, i32* %.di0001_350, align 4, !dbg !53
  %1 = load i32, i32* %.di0001_350, align 4, !dbg !53
  store i32 %1, i32* %.ds0001_351, align 4, !dbg !53
  store i32 1, i32* %.dl0001_353, align 4, !dbg !53
  %2 = load i32, i32* %.dl0001_353, align 4, !dbg !53
  store i32 %2, i32* %.dl0001.copy_419, align 4, !dbg !53
  %3 = load i32, i32* %.de0001_349, align 4, !dbg !53
  store i32 %3, i32* %.de0001.copy_420, align 4, !dbg !53
  %4 = load i32, i32* %.ds0001_351, align 4, !dbg !53
  store i32 %4, i32* %.ds0001.copy_421, align 4, !dbg !53
  %5 = load i32, i32* %__gtid___nv_MAIN_F1L22_2__425, align 4, !dbg !53
  %6 = bitcast i32* %.i0000p_327 to i64*, !dbg !53
  %7 = bitcast i32* %.dl0001.copy_419 to i64*, !dbg !53
  %8 = bitcast i32* %.de0001.copy_420 to i64*, !dbg !53
  %9 = bitcast i32* %.ds0001.copy_421 to i64*, !dbg !53
  %10 = load i32, i32* %.ds0001.copy_421, align 4, !dbg !53
  call void @__kmpc_for_static_init_4(i64* null, i32 %5, i32 92, i64* %6, i64* %7, i64* %8, i64* %9, i32 %10, i32 1), !dbg !53
  %11 = load i32, i32* %.dl0001.copy_419, align 4, !dbg !53
  store i32 %11, i32* %.dl0001_353, align 4, !dbg !53
  %12 = load i32, i32* %.de0001.copy_420, align 4, !dbg !53
  store i32 %12, i32* %.de0001_349, align 4, !dbg !53
  %13 = load i32, i32* %.ds0001.copy_421, align 4, !dbg !53
  store i32 %13, i32* %.ds0001_351, align 4, !dbg !53
  %14 = load i32, i32* %.dl0001_353, align 4, !dbg !53
  store i32 %14, i32* %i_326, align 4, !dbg !53
  %15 = load i32, i32* %i_326, align 4, !dbg !53
  call void @llvm.dbg.value(metadata i32 %15, metadata !54, metadata !DIExpression()), !dbg !50
  store i32 %15, i32* %.dX0001_352, align 4, !dbg !53
  %16 = load i32, i32* %.dX0001_352, align 4, !dbg !53
  %17 = load i32, i32* %.du0001_348, align 4, !dbg !53
  %18 = icmp sgt i32 %16, %17, !dbg !53
  br i1 %18, label %L.LB4_346, label %L.LB4_450, !dbg !53

L.LB4_450:                                        ; preds = %L.LB4_324
  %19 = load i32, i32* %.du0001_348, align 4, !dbg !53
  %20 = load i32, i32* %.de0001_349, align 4, !dbg !53
  %21 = icmp slt i32 %19, %20, !dbg !53
  %22 = select i1 %21, i32 %19, i32 %20, !dbg !53
  store i32 %22, i32* %.de0001_349, align 4, !dbg !53
  %23 = load i32, i32* %.dX0001_352, align 4, !dbg !53
  store i32 %23, i32* %i_326, align 4, !dbg !53
  %24 = load i32, i32* %.di0001_350, align 4, !dbg !53
  %25 = load i32, i32* %.de0001_349, align 4, !dbg !53
  %26 = load i32, i32* %.dX0001_352, align 4, !dbg !53
  %27 = sub nsw i32 %25, %26, !dbg !53
  %28 = add nsw i32 %24, %27, !dbg !53
  %29 = load i32, i32* %.di0001_350, align 4, !dbg !53
  %30 = sdiv i32 %28, %29, !dbg !53
  store i32 %30, i32* %.dY0001_347, align 4, !dbg !53
  %31 = load i32, i32* %.dY0001_347, align 4, !dbg !53
  %32 = icmp sle i32 %31, 0, !dbg !53
  br i1 %32, label %L.LB4_356, label %L.LB4_355, !dbg !53

L.LB4_355:                                        ; preds = %L.LB4_355, %L.LB4_450
  %33 = bitcast i64* %__nv_MAIN_F1L22_2Arg2 to i8*, !dbg !55
  %34 = getelementptr i8, i8* %33, i64 8, !dbg !55
  %35 = bitcast i8* %34 to i64**, !dbg !55
  %36 = load i64*, i64** %35, align 8, !dbg !55
  call void @omp_set_lock_(i64* %36), !dbg !55
  %37 = load i32, i32* %var_323, align 4, !dbg !56
  call void @llvm.dbg.value(metadata i32 %37, metadata !51, metadata !DIExpression()), !dbg !50
  %38 = add nsw i32 %37, 1, !dbg !56
  store i32 %38, i32* %var_323, align 4, !dbg !56
  %39 = bitcast i64* %__nv_MAIN_F1L22_2Arg2 to i8*, !dbg !57
  %40 = getelementptr i8, i8* %39, i64 8, !dbg !57
  %41 = bitcast i8* %40 to i64**, !dbg !57
  %42 = load i64*, i64** %41, align 8, !dbg !57
  call void @omp_unset_lock_(i64* %42), !dbg !57
  %43 = load i32, i32* %.di0001_350, align 4, !dbg !58
  %44 = load i32, i32* %i_326, align 4, !dbg !58
  call void @llvm.dbg.value(metadata i32 %44, metadata !54, metadata !DIExpression()), !dbg !50
  %45 = add nsw i32 %43, %44, !dbg !58
  store i32 %45, i32* %i_326, align 4, !dbg !58
  %46 = load i32, i32* %.dY0001_347, align 4, !dbg !58
  %47 = sub nsw i32 %46, 1, !dbg !58
  store i32 %47, i32* %.dY0001_347, align 4, !dbg !58
  %48 = load i32, i32* %.dY0001_347, align 4, !dbg !58
  %49 = icmp sgt i32 %48, 0, !dbg !58
  br i1 %49, label %L.LB4_355, label %L.LB4_356, !dbg !58

L.LB4_356:                                        ; preds = %L.LB4_355, %L.LB4_450
  br label %L.LB4_346

L.LB4_346:                                        ; preds = %L.LB4_356, %L.LB4_324
  %50 = load i32, i32* %__gtid___nv_MAIN_F1L22_2__425, align 4, !dbg !58
  call void @__kmpc_for_static_fini(i64* null, i32 %50), !dbg !58
  br label %L.LB4_328

L.LB4_328:                                        ; preds = %L.LB4_346
  %51 = call i32 (...) @_mp_bcs_nest_red(), !dbg !50
  %52 = call i32 (...) @_mp_bcs_nest_red(), !dbg !50
  %53 = load i32, i32* %var_323, align 4, !dbg !50
  call void @llvm.dbg.value(metadata i32 %53, metadata !51, metadata !DIExpression()), !dbg !50
  %54 = bitcast i64* %__nv_MAIN_F1L22_2Arg2 to i32**, !dbg !50
  %55 = load i32*, i32** %54, align 8, !dbg !50
  %56 = load i32, i32* %55, align 4, !dbg !50
  %57 = add nsw i32 %53, %56, !dbg !50
  %58 = bitcast i64* %__nv_MAIN_F1L22_2Arg2 to i32**, !dbg !50
  %59 = load i32*, i32** %58, align 8, !dbg !50
  store i32 %57, i32* %59, align 4, !dbg !50
  %60 = call i32 (...) @_mp_ecs_nest_red(), !dbg !50
  %61 = call i32 (...) @_mp_ecs_nest_red(), !dbg !50
  br label %L.LB4_329

L.LB4_329:                                        ; preds = %L.LB4_328
  ret void, !dbg !50
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

declare void @omp_unset_lock_(i64*) #0

declare void @omp_set_lock_(i64*) #0

declare void @omp_init_lock_(i64*) #0

declare void @omp_destroy_lock_(i64*) #0

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { "no-frame-pointer-elim-non-leaf" }
attributes #1 = { nounwind readnone speculatable }

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !4, globals: !4, imports: !4)
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB154-missinglock3-orig-gpu-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb154_missinglock3_orig_gpu_no", scope: !2, file: !3, line: 11, type: !6, scopeLine: 11, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!6 = !DISubroutineType(cc: DW_CC_program, types: !7)
!7 = !{null}
!8 = !DILocalVariable(name: "omp_sched_static", scope: !5, file: !3, type: !9)
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 0, scope: !5)
!11 = !DILocalVariable(name: "omp_proc_bind_false", scope: !5, file: !3, type: !9)
!12 = !DILocalVariable(name: "omp_proc_bind_true", scope: !5, file: !3, type: !9)
!13 = !DILocalVariable(name: "omp_lock_hint_none", scope: !5, file: !3, type: !9)
!14 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !5, file: !3, type: !9)
!15 = !DILocation(line: 34, column: 1, scope: !5)
!16 = !DILocation(line: 11, column: 1, scope: !5)
!17 = !DILocalVariable(name: "var", scope: !5, file: !3, type: !9)
!18 = !DILocation(line: 17, column: 1, scope: !5)
!19 = !DILocalVariable(name: "lck", scope: !5, file: !3, type: !9)
!20 = !DILocation(line: 19, column: 1, scope: !5)
!21 = !DILocation(line: 29, column: 1, scope: !5)
!22 = !DILocation(line: 31, column: 1, scope: !5)
!23 = !DILocation(line: 33, column: 1, scope: !5)
!24 = !DILocalVariable(scope: !5, file: !3, type: !9, flags: DIFlagArtificial)
!25 = distinct !DISubprogram(name: "__nv_MAIN__F1L21_1", scope: !2, file: !3, line: 21, type: !26, scopeLine: 21, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!26 = !DISubroutineType(types: !27)
!27 = !{null, !9, !28, !28}
!28 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!29 = !DILocalVariable(name: "__nv_MAIN__F1L21_1Arg0", arg: 1, scope: !25, file: !3, type: !9)
!30 = !DILocation(line: 0, scope: !25)
!31 = !DILocalVariable(name: "__nv_MAIN__F1L21_1Arg1", arg: 2, scope: !25, file: !3, type: !28)
!32 = !DILocalVariable(name: "__nv_MAIN__F1L21_1Arg2", arg: 3, scope: !25, file: !3, type: !28)
!33 = !DILocalVariable(name: "omp_sched_static", scope: !25, file: !3, type: !9)
!34 = !DILocalVariable(name: "omp_proc_bind_false", scope: !25, file: !3, type: !9)
!35 = !DILocalVariable(name: "omp_proc_bind_true", scope: !25, file: !3, type: !9)
!36 = !DILocalVariable(name: "omp_lock_hint_none", scope: !25, file: !3, type: !9)
!37 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !25, file: !3, type: !9)
!38 = !DILocation(line: 22, column: 1, scope: !25)
!39 = !DILocation(line: 29, column: 1, scope: !25)
!40 = distinct !DISubprogram(name: "__nv_MAIN_F1L22_2", scope: !2, file: !3, line: 22, type: !26, scopeLine: 22, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!41 = !DILocalVariable(name: "__nv_MAIN_F1L22_2Arg0", arg: 1, scope: !40, file: !3, type: !9)
!42 = !DILocation(line: 0, scope: !40)
!43 = !DILocalVariable(name: "__nv_MAIN_F1L22_2Arg1", arg: 2, scope: !40, file: !3, type: !28)
!44 = !DILocalVariable(name: "__nv_MAIN_F1L22_2Arg2", arg: 3, scope: !40, file: !3, type: !28)
!45 = !DILocalVariable(name: "omp_sched_static", scope: !40, file: !3, type: !9)
!46 = !DILocalVariable(name: "omp_proc_bind_false", scope: !40, file: !3, type: !9)
!47 = !DILocalVariable(name: "omp_proc_bind_true", scope: !40, file: !3, type: !9)
!48 = !DILocalVariable(name: "omp_lock_hint_none", scope: !40, file: !3, type: !9)
!49 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !40, file: !3, type: !9)
!50 = !DILocation(line: 28, column: 1, scope: !40)
!51 = !DILocalVariable(name: "var", scope: !40, file: !3, type: !9)
!52 = !DILocation(line: 22, column: 1, scope: !40)
!53 = !DILocation(line: 23, column: 1, scope: !40)
!54 = !DILocalVariable(name: "i", scope: !40, file: !3, type: !9)
!55 = !DILocation(line: 24, column: 1, scope: !40)
!56 = !DILocation(line: 25, column: 1, scope: !40)
!57 = !DILocation(line: 26, column: 1, scope: !40)
!58 = !DILocation(line: 27, column: 1, scope: !40)

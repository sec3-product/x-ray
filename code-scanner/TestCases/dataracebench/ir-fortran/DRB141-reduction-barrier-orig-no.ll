; ModuleID = '/tmp/DRB141-reduction-barrier-orig-no-e7d07c.ll'
source_filename = "/tmp/DRB141-reduction-barrier-orig-no-e7d07c.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%astruct.dt60 = type <{ i8* }>

@.C306_MAIN_ = internal constant i32 25
@.C305_MAIN_ = internal constant i32 14
@.C327_MAIN_ = internal constant [7 x i8] c"Sum is "
@.C284_MAIN_ = internal constant i64 0
@.C324_MAIN_ = internal constant i32 6
@.C321_MAIN_ = internal constant [61 x i8] c"micro-benchmarks-fortran/DRB141-reduction-barrier-orig-no.f95"
@.C323_MAIN_ = internal constant i32 34
@.C315_MAIN_ = internal constant i32 10
@.C285_MAIN_ = internal constant i32 1
@.C283_MAIN_ = internal constant i32 0
@.C306___nv_MAIN__F1L20_1 = internal constant i32 25
@.C305___nv_MAIN__F1L20_1 = internal constant i32 14
@.C327___nv_MAIN__F1L20_1 = internal constant [7 x i8] c"Sum is "
@.C284___nv_MAIN__F1L20_1 = internal constant i64 0
@.C324___nv_MAIN__F1L20_1 = internal constant i32 6
@.C321___nv_MAIN__F1L20_1 = internal constant [61 x i8] c"micro-benchmarks-fortran/DRB141-reduction-barrier-orig-no.f95"
@.C323___nv_MAIN__F1L20_1 = internal constant i32 34
@.C315___nv_MAIN__F1L20_1 = internal constant i32 10
@.C285___nv_MAIN__F1L20_1 = internal constant i32 1
@.C283___nv_MAIN__F1L20_1 = internal constant i32 0

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__364 = alloca i32, align 4
  %a_307 = alloca i32, align 4
  %.uplevelArgPack0001_358 = alloca %astruct.dt60, align 8
  call void @llvm.dbg.value(metadata i32 1, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !14, metadata !DIExpression()), !dbg !10
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !15
  store i32 %0, i32* %__gtid_MAIN__364, align 4, !dbg !15
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !16
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !16
  call void (i8*, ...) %2(i8* %1), !dbg !16
  br label %L.LB1_353

L.LB1_353:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %a_307, metadata !17, metadata !DIExpression()), !dbg !10
  %3 = bitcast i32* %a_307 to i8*, !dbg !18
  %4 = bitcast %astruct.dt60* %.uplevelArgPack0001_358 to i8**, !dbg !18
  store i8* %3, i8** %4, align 8, !dbg !18
  br label %L.LB1_362, !dbg !18

L.LB1_362:                                        ; preds = %L.LB1_353
  %5 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L20_1_ to i64*, !dbg !18
  %6 = bitcast %astruct.dt60* %.uplevelArgPack0001_358 to i64*, !dbg !18
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %5, i64* %6), !dbg !18
  ret void, !dbg !15
}

define internal void @__nv_MAIN__F1L20_1_(i32* %__nv_MAIN__F1L20_1Arg0, i64* %__nv_MAIN__F1L20_1Arg1, i64* %__nv_MAIN__F1L20_1Arg2) #0 !dbg !19 {
L.entry:
  %__gtid___nv_MAIN__F1L20_1__386 = alloca i32, align 4
  %a_314 = alloca i32, align 4
  %.i0000p_317 = alloca i32, align 4
  %i_316 = alloca i32, align 4
  %.du0001p_342 = alloca i32, align 4
  %.de0001p_343 = alloca i32, align 4
  %.di0001p_344 = alloca i32, align 4
  %.ds0001p_345 = alloca i32, align 4
  %.dl0001p_347 = alloca i32, align 4
  %.dl0001p.copy_407 = alloca i32, align 4
  %.de0001p.copy_408 = alloca i32, align 4
  %.ds0001p.copy_409 = alloca i32, align 4
  %.dX0001p_346 = alloca i32, align 4
  %.dY0001p_341 = alloca i32, align 4
  %.s0000_430 = alloca i32, align 4
  %.s0001_431 = alloca i32, align 4
  %z__io_326 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L20_1Arg0, metadata !23, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L20_1Arg1, metadata !25, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L20_1Arg2, metadata !26, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.value(metadata i32 1, metadata !27, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.value(metadata i32 0, metadata !28, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.value(metadata i32 1, metadata !29, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.value(metadata i32 0, metadata !30, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.value(metadata i32 1, metadata !31, metadata !DIExpression()), !dbg !24
  %0 = load i32, i32* %__nv_MAIN__F1L20_1Arg0, align 4, !dbg !32
  store i32 %0, i32* %__gtid___nv_MAIN__F1L20_1__386, align 4, !dbg !32
  br label %L.LB2_385

L.LB2_385:                                        ; preds = %L.entry
  br label %L.LB2_311

L.LB2_311:                                        ; preds = %L.LB2_385
  %1 = load i32, i32* %__gtid___nv_MAIN__F1L20_1__386, align 4, !dbg !33
  %2 = call i32 @__kmpc_master(i64* null, i32 %1), !dbg !33
  %3 = icmp eq i32 %2, 0, !dbg !33
  br i1 %3, label %L.LB2_338, label %L.LB2_454, !dbg !33

L.LB2_454:                                        ; preds = %L.LB2_311
  %4 = bitcast i64* %__nv_MAIN__F1L20_1Arg2 to i32**, !dbg !34
  %5 = load i32*, i32** %4, align 8, !dbg !34
  store i32 0, i32* %5, align 4, !dbg !34
  %6 = load i32, i32* %__gtid___nv_MAIN__F1L20_1__386, align 4, !dbg !35
  call void @__kmpc_end_master(i64* null, i32 %6), !dbg !35
  br label %L.LB2_338

L.LB2_338:                                        ; preds = %L.LB2_454, %L.LB2_311
  %7 = load i32, i32* %__gtid___nv_MAIN__F1L20_1__386, align 4, !dbg !36
  call void @__kmpc_barrier(i64* null, i32 %7), !dbg !36
  br label %L.LB2_313

L.LB2_313:                                        ; preds = %L.LB2_338
  call void @llvm.dbg.declare(metadata i32* %a_314, metadata !37, metadata !DIExpression()), !dbg !32
  store i32 0, i32* %a_314, align 4, !dbg !38
  store i32 0, i32* %.i0000p_317, align 4, !dbg !39
  call void @llvm.dbg.declare(metadata i32* %i_316, metadata !40, metadata !DIExpression()), !dbg !32
  store i32 1, i32* %i_316, align 4, !dbg !39
  store i32 10, i32* %.du0001p_342, align 4, !dbg !39
  store i32 10, i32* %.de0001p_343, align 4, !dbg !39
  store i32 1, i32* %.di0001p_344, align 4, !dbg !39
  %8 = load i32, i32* %.di0001p_344, align 4, !dbg !39
  store i32 %8, i32* %.ds0001p_345, align 4, !dbg !39
  store i32 1, i32* %.dl0001p_347, align 4, !dbg !39
  %9 = load i32, i32* %.dl0001p_347, align 4, !dbg !39
  store i32 %9, i32* %.dl0001p.copy_407, align 4, !dbg !39
  %10 = load i32, i32* %.de0001p_343, align 4, !dbg !39
  store i32 %10, i32* %.de0001p.copy_408, align 4, !dbg !39
  %11 = load i32, i32* %.ds0001p_345, align 4, !dbg !39
  store i32 %11, i32* %.ds0001p.copy_409, align 4, !dbg !39
  %12 = load i32, i32* %__gtid___nv_MAIN__F1L20_1__386, align 4, !dbg !39
  %13 = bitcast i32* %.i0000p_317 to i64*, !dbg !39
  %14 = bitcast i32* %.dl0001p.copy_407 to i64*, !dbg !39
  %15 = bitcast i32* %.de0001p.copy_408 to i64*, !dbg !39
  %16 = bitcast i32* %.ds0001p.copy_409 to i64*, !dbg !39
  %17 = load i32, i32* %.ds0001p.copy_409, align 4, !dbg !39
  call void @__kmpc_for_static_init_4(i64* null, i32 %12, i32 34, i64* %13, i64* %14, i64* %15, i64* %16, i32 %17, i32 1), !dbg !39
  %18 = load i32, i32* %.dl0001p.copy_407, align 4, !dbg !39
  store i32 %18, i32* %.dl0001p_347, align 4, !dbg !39
  %19 = load i32, i32* %.de0001p.copy_408, align 4, !dbg !39
  store i32 %19, i32* %.de0001p_343, align 4, !dbg !39
  %20 = load i32, i32* %.ds0001p.copy_409, align 4, !dbg !39
  store i32 %20, i32* %.ds0001p_345, align 4, !dbg !39
  %21 = load i32, i32* %.dl0001p_347, align 4, !dbg !39
  store i32 %21, i32* %i_316, align 4, !dbg !39
  %22 = load i32, i32* %i_316, align 4, !dbg !39
  call void @llvm.dbg.value(metadata i32 %22, metadata !40, metadata !DIExpression()), !dbg !32
  store i32 %22, i32* %.dX0001p_346, align 4, !dbg !39
  %23 = load i32, i32* %.dX0001p_346, align 4, !dbg !39
  %24 = load i32, i32* %.du0001p_342, align 4, !dbg !39
  %25 = icmp sgt i32 %23, %24, !dbg !39
  br i1 %25, label %L.LB2_340, label %L.LB2_455, !dbg !39

L.LB2_455:                                        ; preds = %L.LB2_313
  %26 = load i32, i32* %.dX0001p_346, align 4, !dbg !39
  store i32 %26, i32* %i_316, align 4, !dbg !39
  %27 = load i32, i32* %.di0001p_344, align 4, !dbg !39
  %28 = load i32, i32* %.de0001p_343, align 4, !dbg !39
  %29 = load i32, i32* %.dX0001p_346, align 4, !dbg !39
  %30 = sub nsw i32 %28, %29, !dbg !39
  %31 = add nsw i32 %27, %30, !dbg !39
  %32 = load i32, i32* %.di0001p_344, align 4, !dbg !39
  %33 = sdiv i32 %31, %32, !dbg !39
  store i32 %33, i32* %.dY0001p_341, align 4, !dbg !39
  %34 = load i32, i32* %.dY0001p_341, align 4, !dbg !39
  %35 = icmp sle i32 %34, 0, !dbg !39
  br i1 %35, label %L.LB2_350, label %L.LB2_349, !dbg !39

L.LB2_349:                                        ; preds = %L.LB2_349, %L.LB2_455
  %36 = load i32, i32* %i_316, align 4, !dbg !41
  call void @llvm.dbg.value(metadata i32 %36, metadata !40, metadata !DIExpression()), !dbg !32
  %37 = load i32, i32* %a_314, align 4, !dbg !41
  call void @llvm.dbg.value(metadata i32 %37, metadata !37, metadata !DIExpression()), !dbg !32
  %38 = add nsw i32 %36, %37, !dbg !41
  store i32 %38, i32* %a_314, align 4, !dbg !41
  %39 = load i32, i32* %.di0001p_344, align 4, !dbg !42
  %40 = load i32, i32* %i_316, align 4, !dbg !42
  call void @llvm.dbg.value(metadata i32 %40, metadata !40, metadata !DIExpression()), !dbg !32
  %41 = add nsw i32 %39, %40, !dbg !42
  store i32 %41, i32* %i_316, align 4, !dbg !42
  %42 = load i32, i32* %.dY0001p_341, align 4, !dbg !42
  %43 = sub nsw i32 %42, 1, !dbg !42
  store i32 %43, i32* %.dY0001p_341, align 4, !dbg !42
  %44 = load i32, i32* %.dY0001p_341, align 4, !dbg !42
  %45 = icmp sgt i32 %44, 0, !dbg !42
  br i1 %45, label %L.LB2_349, label %L.LB2_350, !dbg !42

L.LB2_350:                                        ; preds = %L.LB2_349, %L.LB2_455
  br label %L.LB2_340

L.LB2_340:                                        ; preds = %L.LB2_350, %L.LB2_313
  %46 = load i32, i32* %__gtid___nv_MAIN__F1L20_1__386, align 4, !dbg !42
  call void @__kmpc_for_static_fini(i64* null, i32 %46), !dbg !42
  %47 = call i32 (...) @_mp_bcs_nest_red(), !dbg !42
  %48 = call i32 (...) @_mp_bcs_nest_red(), !dbg !42
  %49 = load i32, i32* %a_314, align 4, !dbg !42
  call void @llvm.dbg.value(metadata i32 %49, metadata !37, metadata !DIExpression()), !dbg !32
  %50 = bitcast i64* %__nv_MAIN__F1L20_1Arg2 to i32**, !dbg !42
  %51 = load i32*, i32** %50, align 8, !dbg !42
  %52 = load i32, i32* %51, align 4, !dbg !42
  %53 = add nsw i32 %49, %52, !dbg !42
  %54 = bitcast i64* %__nv_MAIN__F1L20_1Arg2 to i32**, !dbg !42
  %55 = load i32*, i32** %54, align 8, !dbg !42
  store i32 %53, i32* %55, align 4, !dbg !42
  %56 = call i32 (...) @_mp_ecs_nest_red(), !dbg !42
  %57 = call i32 (...) @_mp_ecs_nest_red(), !dbg !42
  br label %L.LB2_318

L.LB2_318:                                        ; preds = %L.LB2_340
  %58 = load i32, i32* %__gtid___nv_MAIN__F1L20_1__386, align 4, !dbg !43
  call void @__kmpc_barrier(i64* null, i32 %58), !dbg !43
  store i32 -1, i32* %.s0000_430, align 4, !dbg !44
  store i32 0, i32* %.s0001_431, align 4, !dbg !44
  %59 = load i32, i32* %__gtid___nv_MAIN__F1L20_1__386, align 4, !dbg !44
  %60 = call i32 @__kmpc_single(i64* null, i32 %59), !dbg !44
  %61 = icmp eq i32 %60, 0, !dbg !44
  br i1 %61, label %L.LB2_351, label %L.LB2_319, !dbg !44

L.LB2_319:                                        ; preds = %L.LB2_318
  call void (...) @_mp_bcs_nest(), !dbg !45
  %62 = bitcast i32* @.C323___nv_MAIN__F1L20_1 to i8*, !dbg !45
  %63 = bitcast [61 x i8]* @.C321___nv_MAIN__F1L20_1 to i8*, !dbg !45
  %64 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !45
  call void (i8*, i8*, i64, ...) %64(i8* %62, i8* %63, i64 61), !dbg !45
  %65 = bitcast i32* @.C324___nv_MAIN__F1L20_1 to i8*, !dbg !45
  %66 = bitcast i32* @.C283___nv_MAIN__F1L20_1 to i8*, !dbg !45
  %67 = bitcast i32* @.C283___nv_MAIN__F1L20_1 to i8*, !dbg !45
  %68 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !45
  %69 = call i32 (i8*, i8*, i8*, i8*, ...) %68(i8* %65, i8* null, i8* %66, i8* %67), !dbg !45
  call void @llvm.dbg.declare(metadata i32* %z__io_326, metadata !46, metadata !DIExpression()), !dbg !24
  store i32 %69, i32* %z__io_326, align 4, !dbg !45
  %70 = bitcast [7 x i8]* @.C327___nv_MAIN__F1L20_1 to i8*, !dbg !45
  %71 = bitcast i32 (...)* @f90io_sc_ch_ldw to i32 (i8*, i32, i64, ...)*, !dbg !45
  %72 = call i32 (i8*, i32, i64, ...) %71(i8* %70, i32 14, i64 7), !dbg !45
  store i32 %72, i32* %z__io_326, align 4, !dbg !45
  %73 = bitcast i64* %__nv_MAIN__F1L20_1Arg2 to i32**, !dbg !45
  %74 = load i32*, i32** %73, align 8, !dbg !45
  %75 = load i32, i32* %74, align 4, !dbg !45
  %76 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !45
  %77 = call i32 (i32, i32, ...) %76(i32 %75, i32 25), !dbg !45
  store i32 %77, i32* %z__io_326, align 4, !dbg !45
  %78 = call i32 (...) @f90io_ldw_end(), !dbg !45
  store i32 %78, i32* %z__io_326, align 4, !dbg !45
  call void (...) @_mp_ecs_nest(), !dbg !45
  %79 = load i32, i32* %__gtid___nv_MAIN__F1L20_1__386, align 4, !dbg !47
  store i32 %79, i32* %.s0000_430, align 4, !dbg !47
  store i32 1, i32* %.s0001_431, align 4, !dbg !47
  %80 = load i32, i32* %__gtid___nv_MAIN__F1L20_1__386, align 4, !dbg !47
  call void @__kmpc_end_single(i64* null, i32 %80), !dbg !47
  br label %L.LB2_351

L.LB2_351:                                        ; preds = %L.LB2_319, %L.LB2_318
  br label %L.LB2_333

L.LB2_333:                                        ; preds = %L.LB2_351
  %81 = load i32, i32* %__gtid___nv_MAIN__F1L20_1__386, align 4, !dbg !47
  call void @__kmpc_barrier(i64* null, i32 %81), !dbg !47
  br label %L.LB2_334

L.LB2_334:                                        ; preds = %L.LB2_333
  ret void, !dbg !32
}

declare void @_mp_ecs_nest(...) #0

declare signext i32 @f90io_ldw_end(...) #0

declare signext i32 @f90io_sc_i_ldw(...) #0

declare signext i32 @f90io_sc_ch_ldw(...) #0

declare signext i32 @f90io_print_init(...) #0

declare void @f90io_src_info03a(...) #0

declare void @_mp_bcs_nest(...) #0

declare signext i32 @_mp_ecs_nest_red(...) #0

declare signext i32 @_mp_bcs_nest_red(...) #0

declare void @__kmpc_end_single(i64*, i32) #0

declare signext i32 @__kmpc_single(i64*, i32) #0

declare void @__kmpc_for_static_fini(i64*, i32) #0

declare void @__kmpc_for_static_init_4(i64*, i32, i32, i64*, i64*, i64*, i64*, i32, i32) #0

declare void @__kmpc_barrier(i64*, i32) #0

declare void @__kmpc_end_master(i64*, i32) #0

declare signext i32 @__kmpc_master(i64*, i32) #0

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
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB141-reduction-barrier-orig-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb141_reduction_barrier_orig_no", scope: !2, file: !3, line: 14, type: !6, scopeLine: 14, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!6 = !DISubroutineType(cc: DW_CC_program, types: !7)
!7 = !{null}
!8 = !DILocalVariable(name: "omp_sched_static", scope: !5, file: !3, type: !9)
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 0, scope: !5)
!11 = !DILocalVariable(name: "omp_proc_bind_false", scope: !5, file: !3, type: !9)
!12 = !DILocalVariable(name: "omp_proc_bind_true", scope: !5, file: !3, type: !9)
!13 = !DILocalVariable(name: "omp_lock_hint_none", scope: !5, file: !3, type: !9)
!14 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !5, file: !3, type: !9)
!15 = !DILocation(line: 38, column: 1, scope: !5)
!16 = !DILocation(line: 14, column: 1, scope: !5)
!17 = !DILocalVariable(name: "a", scope: !5, file: !3, type: !9)
!18 = !DILocation(line: 20, column: 1, scope: !5)
!19 = distinct !DISubprogram(name: "__nv_MAIN__F1L20_1", scope: !2, file: !3, line: 20, type: !20, scopeLine: 20, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!20 = !DISubroutineType(types: !21)
!21 = !{null, !9, !22, !22}
!22 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!23 = !DILocalVariable(name: "__nv_MAIN__F1L20_1Arg0", arg: 1, scope: !19, file: !3, type: !9)
!24 = !DILocation(line: 0, scope: !19)
!25 = !DILocalVariable(name: "__nv_MAIN__F1L20_1Arg1", arg: 2, scope: !19, file: !3, type: !22)
!26 = !DILocalVariable(name: "__nv_MAIN__F1L20_1Arg2", arg: 3, scope: !19, file: !3, type: !22)
!27 = !DILocalVariable(name: "omp_sched_static", scope: !19, file: !3, type: !9)
!28 = !DILocalVariable(name: "omp_proc_bind_false", scope: !19, file: !3, type: !9)
!29 = !DILocalVariable(name: "omp_proc_bind_true", scope: !19, file: !3, type: !9)
!30 = !DILocalVariable(name: "omp_lock_hint_none", scope: !19, file: !3, type: !9)
!31 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !19, file: !3, type: !9)
!32 = !DILocation(line: 37, column: 1, scope: !19)
!33 = !DILocation(line: 21, column: 1, scope: !19)
!34 = !DILocation(line: 22, column: 1, scope: !19)
!35 = !DILocation(line: 23, column: 1, scope: !19)
!36 = !DILocation(line: 25, column: 1, scope: !19)
!37 = !DILocalVariable(name: "a", scope: !19, file: !3, type: !9)
!38 = !DILocation(line: 27, column: 1, scope: !19)
!39 = !DILocation(line: 28, column: 1, scope: !19)
!40 = !DILocalVariable(name: "i", scope: !19, file: !3, type: !9)
!41 = !DILocation(line: 29, column: 1, scope: !19)
!42 = !DILocation(line: 30, column: 1, scope: !19)
!43 = !DILocation(line: 31, column: 1, scope: !19)
!44 = !DILocation(line: 33, column: 1, scope: !19)
!45 = !DILocation(line: 34, column: 1, scope: !19)
!46 = !DILocalVariable(scope: !19, file: !3, type: !9, flags: DIFlagArtificial)
!47 = !DILocation(line: 35, column: 1, scope: !19)

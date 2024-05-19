; ModuleID = '/tmp/DRB140-reduction-barrier-orig-yes-009c34.ll'
source_filename = "/tmp/DRB140-reduction-barrier-orig-yes-009c34.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%astruct.dt60 = type <{ i8* }>

@.C306_MAIN_ = internal constant i32 25
@.C305_MAIN_ = internal constant i32 14
@.C327_MAIN_ = internal constant [7 x i8] c"Sum is "
@.C284_MAIN_ = internal constant i64 0
@.C324_MAIN_ = internal constant i32 6
@.C321_MAIN_ = internal constant [62 x i8] c"micro-benchmarks-fortran/DRB140-reduction-barrier-orig-yes.f95"
@.C323_MAIN_ = internal constant i32 31
@.C315_MAIN_ = internal constant i32 10
@.C285_MAIN_ = internal constant i32 1
@.C283_MAIN_ = internal constant i32 0
@.C306___nv_MAIN__F1L19_1 = internal constant i32 25
@.C305___nv_MAIN__F1L19_1 = internal constant i32 14
@.C327___nv_MAIN__F1L19_1 = internal constant [7 x i8] c"Sum is "
@.C284___nv_MAIN__F1L19_1 = internal constant i64 0
@.C324___nv_MAIN__F1L19_1 = internal constant i32 6
@.C321___nv_MAIN__F1L19_1 = internal constant [62 x i8] c"micro-benchmarks-fortran/DRB140-reduction-barrier-orig-yes.f95"
@.C323___nv_MAIN__F1L19_1 = internal constant i32 31
@.C315___nv_MAIN__F1L19_1 = internal constant i32 10
@.C285___nv_MAIN__F1L19_1 = internal constant i32 1
@.C283___nv_MAIN__F1L19_1 = internal constant i32 0

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
  %5 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L19_1_ to i64*, !dbg !18
  %6 = bitcast %astruct.dt60* %.uplevelArgPack0001_358 to i64*, !dbg !18
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %5, i64* %6), !dbg !18
  ret void, !dbg !15
}

define internal void @__nv_MAIN__F1L19_1_(i32* %__nv_MAIN__F1L19_1Arg0, i64* %__nv_MAIN__F1L19_1Arg1, i64* %__nv_MAIN__F1L19_1Arg2) #0 !dbg !19 {
L.entry:
  %__gtid___nv_MAIN__F1L19_1__386 = alloca i32, align 4
  %a_314 = alloca i32, align 4
  %.i0000p_317 = alloca i32, align 4
  %i_316 = alloca i32, align 4
  %.du0001p_342 = alloca i32, align 4
  %.de0001p_343 = alloca i32, align 4
  %.di0001p_344 = alloca i32, align 4
  %.ds0001p_345 = alloca i32, align 4
  %.dl0001p_347 = alloca i32, align 4
  %.dl0001p.copy_404 = alloca i32, align 4
  %.de0001p.copy_405 = alloca i32, align 4
  %.ds0001p.copy_406 = alloca i32, align 4
  %.dX0001p_346 = alloca i32, align 4
  %.dY0001p_341 = alloca i32, align 4
  %.s0000_431 = alloca i32, align 4
  %.s0001_432 = alloca i32, align 4
  %z__io_326 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L19_1Arg0, metadata !23, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L19_1Arg1, metadata !25, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L19_1Arg2, metadata !26, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.value(metadata i32 1, metadata !27, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.value(metadata i32 0, metadata !28, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.value(metadata i32 1, metadata !29, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.value(metadata i32 0, metadata !30, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.value(metadata i32 1, metadata !31, metadata !DIExpression()), !dbg !24
  %0 = load i32, i32* %__nv_MAIN__F1L19_1Arg0, align 4, !dbg !32
  store i32 %0, i32* %__gtid___nv_MAIN__F1L19_1__386, align 4, !dbg !32
  br label %L.LB2_385

L.LB2_385:                                        ; preds = %L.entry
  br label %L.LB2_311

L.LB2_311:                                        ; preds = %L.LB2_385
  %1 = load i32, i32* %__gtid___nv_MAIN__F1L19_1__386, align 4, !dbg !33
  %2 = call i32 @__kmpc_master(i64* null, i32 %1), !dbg !33
  %3 = icmp eq i32 %2, 0, !dbg !33
  br i1 %3, label %L.LB2_338, label %L.LB2_455, !dbg !33

L.LB2_455:                                        ; preds = %L.LB2_311
  %4 = bitcast i64* %__nv_MAIN__F1L19_1Arg2 to i32**, !dbg !34
  %5 = load i32*, i32** %4, align 8, !dbg !34
  store i32 0, i32* %5, align 4, !dbg !34
  %6 = load i32, i32* %__gtid___nv_MAIN__F1L19_1__386, align 4, !dbg !35
  call void @__kmpc_end_master(i64* null, i32 %6), !dbg !35
  br label %L.LB2_338

L.LB2_338:                                        ; preds = %L.LB2_455, %L.LB2_311
  br label %L.LB2_313

L.LB2_313:                                        ; preds = %L.LB2_338
  call void @llvm.dbg.declare(metadata i32* %a_314, metadata !36, metadata !DIExpression()), !dbg !32
  store i32 0, i32* %a_314, align 4, !dbg !37
  store i32 0, i32* %.i0000p_317, align 4, !dbg !38
  call void @llvm.dbg.declare(metadata i32* %i_316, metadata !39, metadata !DIExpression()), !dbg !32
  store i32 1, i32* %i_316, align 4, !dbg !38
  store i32 10, i32* %.du0001p_342, align 4, !dbg !38
  store i32 10, i32* %.de0001p_343, align 4, !dbg !38
  store i32 1, i32* %.di0001p_344, align 4, !dbg !38
  %7 = load i32, i32* %.di0001p_344, align 4, !dbg !38
  store i32 %7, i32* %.ds0001p_345, align 4, !dbg !38
  store i32 1, i32* %.dl0001p_347, align 4, !dbg !38
  %8 = load i32, i32* %.dl0001p_347, align 4, !dbg !38
  store i32 %8, i32* %.dl0001p.copy_404, align 4, !dbg !38
  %9 = load i32, i32* %.de0001p_343, align 4, !dbg !38
  store i32 %9, i32* %.de0001p.copy_405, align 4, !dbg !38
  %10 = load i32, i32* %.ds0001p_345, align 4, !dbg !38
  store i32 %10, i32* %.ds0001p.copy_406, align 4, !dbg !38
  %11 = load i32, i32* %__gtid___nv_MAIN__F1L19_1__386, align 4, !dbg !38
  %12 = bitcast i32* %.i0000p_317 to i64*, !dbg !38
  %13 = bitcast i32* %.dl0001p.copy_404 to i64*, !dbg !38
  %14 = bitcast i32* %.de0001p.copy_405 to i64*, !dbg !38
  %15 = bitcast i32* %.ds0001p.copy_406 to i64*, !dbg !38
  %16 = load i32, i32* %.ds0001p.copy_406, align 4, !dbg !38
  call void @__kmpc_for_static_init_4(i64* null, i32 %11, i32 34, i64* %12, i64* %13, i64* %14, i64* %15, i32 %16, i32 1), !dbg !38
  %17 = load i32, i32* %.dl0001p.copy_404, align 4, !dbg !38
  store i32 %17, i32* %.dl0001p_347, align 4, !dbg !38
  %18 = load i32, i32* %.de0001p.copy_405, align 4, !dbg !38
  store i32 %18, i32* %.de0001p_343, align 4, !dbg !38
  %19 = load i32, i32* %.ds0001p.copy_406, align 4, !dbg !38
  store i32 %19, i32* %.ds0001p_345, align 4, !dbg !38
  %20 = load i32, i32* %.dl0001p_347, align 4, !dbg !38
  store i32 %20, i32* %i_316, align 4, !dbg !38
  %21 = load i32, i32* %i_316, align 4, !dbg !38
  call void @llvm.dbg.value(metadata i32 %21, metadata !39, metadata !DIExpression()), !dbg !32
  store i32 %21, i32* %.dX0001p_346, align 4, !dbg !38
  %22 = load i32, i32* %.dX0001p_346, align 4, !dbg !38
  %23 = load i32, i32* %.du0001p_342, align 4, !dbg !38
  %24 = icmp sgt i32 %22, %23, !dbg !38
  br i1 %24, label %L.LB2_340, label %L.LB2_456, !dbg !38

L.LB2_456:                                        ; preds = %L.LB2_313
  %25 = load i32, i32* %.dX0001p_346, align 4, !dbg !38
  store i32 %25, i32* %i_316, align 4, !dbg !38
  %26 = load i32, i32* %.di0001p_344, align 4, !dbg !38
  %27 = load i32, i32* %.de0001p_343, align 4, !dbg !38
  %28 = load i32, i32* %.dX0001p_346, align 4, !dbg !38
  %29 = sub nsw i32 %27, %28, !dbg !38
  %30 = add nsw i32 %26, %29, !dbg !38
  %31 = load i32, i32* %.di0001p_344, align 4, !dbg !38
  %32 = sdiv i32 %30, %31, !dbg !38
  store i32 %32, i32* %.dY0001p_341, align 4, !dbg !38
  %33 = load i32, i32* %.dY0001p_341, align 4, !dbg !38
  %34 = icmp sle i32 %33, 0, !dbg !38
  br i1 %34, label %L.LB2_350, label %L.LB2_349, !dbg !38

L.LB2_349:                                        ; preds = %L.LB2_349, %L.LB2_456
  %35 = load i32, i32* %i_316, align 4, !dbg !40
  call void @llvm.dbg.value(metadata i32 %35, metadata !39, metadata !DIExpression()), !dbg !32
  %36 = load i32, i32* %a_314, align 4, !dbg !40
  call void @llvm.dbg.value(metadata i32 %36, metadata !36, metadata !DIExpression()), !dbg !32
  %37 = add nsw i32 %35, %36, !dbg !40
  store i32 %37, i32* %a_314, align 4, !dbg !40
  %38 = load i32, i32* %.di0001p_344, align 4, !dbg !41
  %39 = load i32, i32* %i_316, align 4, !dbg !41
  call void @llvm.dbg.value(metadata i32 %39, metadata !39, metadata !DIExpression()), !dbg !32
  %40 = add nsw i32 %38, %39, !dbg !41
  store i32 %40, i32* %i_316, align 4, !dbg !41
  %41 = load i32, i32* %.dY0001p_341, align 4, !dbg !41
  %42 = sub nsw i32 %41, 1, !dbg !41
  store i32 %42, i32* %.dY0001p_341, align 4, !dbg !41
  %43 = load i32, i32* %.dY0001p_341, align 4, !dbg !41
  %44 = icmp sgt i32 %43, 0, !dbg !41
  br i1 %44, label %L.LB2_349, label %L.LB2_350, !dbg !41

L.LB2_350:                                        ; preds = %L.LB2_349, %L.LB2_456
  br label %L.LB2_340

L.LB2_340:                                        ; preds = %L.LB2_350, %L.LB2_313
  %45 = load i32, i32* %__gtid___nv_MAIN__F1L19_1__386, align 4, !dbg !41
  call void @__kmpc_for_static_fini(i64* null, i32 %45), !dbg !41
  %46 = call i32 (...) @_mp_bcs_nest_red(), !dbg !41
  %47 = call i32 (...) @_mp_bcs_nest_red(), !dbg !41
  %48 = load i32, i32* %a_314, align 4, !dbg !41
  call void @llvm.dbg.value(metadata i32 %48, metadata !36, metadata !DIExpression()), !dbg !32
  %49 = bitcast i64* %__nv_MAIN__F1L19_1Arg2 to i32**, !dbg !41
  %50 = load i32*, i32** %49, align 8, !dbg !41
  %51 = load i32, i32* %50, align 4, !dbg !41
  %52 = add nsw i32 %48, %51, !dbg !41
  %53 = bitcast i64* %__nv_MAIN__F1L19_1Arg2 to i32**, !dbg !41
  %54 = load i32*, i32** %53, align 8, !dbg !41
  store i32 %52, i32* %54, align 4, !dbg !41
  %55 = call i32 (...) @_mp_ecs_nest_red(), !dbg !41
  %56 = call i32 (...) @_mp_ecs_nest_red(), !dbg !41
  br label %L.LB2_318

L.LB2_318:                                        ; preds = %L.LB2_340
  %57 = load i32, i32* %__gtid___nv_MAIN__F1L19_1__386, align 4, !dbg !42
  call void @__kmpc_barrier(i64* null, i32 %57), !dbg !42
  store i32 -1, i32* %.s0000_431, align 4, !dbg !43
  store i32 0, i32* %.s0001_432, align 4, !dbg !43
  %58 = load i32, i32* %__gtid___nv_MAIN__F1L19_1__386, align 4, !dbg !43
  %59 = call i32 @__kmpc_single(i64* null, i32 %58), !dbg !43
  %60 = icmp eq i32 %59, 0, !dbg !43
  br i1 %60, label %L.LB2_351, label %L.LB2_319, !dbg !43

L.LB2_319:                                        ; preds = %L.LB2_318
  call void (...) @_mp_bcs_nest(), !dbg !44
  %61 = bitcast i32* @.C323___nv_MAIN__F1L19_1 to i8*, !dbg !44
  %62 = bitcast [62 x i8]* @.C321___nv_MAIN__F1L19_1 to i8*, !dbg !44
  %63 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !44
  call void (i8*, i8*, i64, ...) %63(i8* %61, i8* %62, i64 62), !dbg !44
  %64 = bitcast i32* @.C324___nv_MAIN__F1L19_1 to i8*, !dbg !44
  %65 = bitcast i32* @.C283___nv_MAIN__F1L19_1 to i8*, !dbg !44
  %66 = bitcast i32* @.C283___nv_MAIN__F1L19_1 to i8*, !dbg !44
  %67 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !44
  %68 = call i32 (i8*, i8*, i8*, i8*, ...) %67(i8* %64, i8* null, i8* %65, i8* %66), !dbg !44
  call void @llvm.dbg.declare(metadata i32* %z__io_326, metadata !45, metadata !DIExpression()), !dbg !24
  store i32 %68, i32* %z__io_326, align 4, !dbg !44
  %69 = bitcast [7 x i8]* @.C327___nv_MAIN__F1L19_1 to i8*, !dbg !44
  %70 = bitcast i32 (...)* @f90io_sc_ch_ldw to i32 (i8*, i32, i64, ...)*, !dbg !44
  %71 = call i32 (i8*, i32, i64, ...) %70(i8* %69, i32 14, i64 7), !dbg !44
  store i32 %71, i32* %z__io_326, align 4, !dbg !44
  %72 = bitcast i64* %__nv_MAIN__F1L19_1Arg2 to i32**, !dbg !44
  %73 = load i32*, i32** %72, align 8, !dbg !44
  %74 = load i32, i32* %73, align 4, !dbg !44
  %75 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !44
  %76 = call i32 (i32, i32, ...) %75(i32 %74, i32 25), !dbg !44
  store i32 %76, i32* %z__io_326, align 4, !dbg !44
  %77 = call i32 (...) @f90io_ldw_end(), !dbg !44
  store i32 %77, i32* %z__io_326, align 4, !dbg !44
  call void (...) @_mp_ecs_nest(), !dbg !44
  %78 = load i32, i32* %__gtid___nv_MAIN__F1L19_1__386, align 4, !dbg !46
  store i32 %78, i32* %.s0000_431, align 4, !dbg !46
  store i32 1, i32* %.s0001_432, align 4, !dbg !46
  %79 = load i32, i32* %__gtid___nv_MAIN__F1L19_1__386, align 4, !dbg !46
  call void @__kmpc_end_single(i64* null, i32 %79), !dbg !46
  br label %L.LB2_351

L.LB2_351:                                        ; preds = %L.LB2_319, %L.LB2_318
  br label %L.LB2_333

L.LB2_333:                                        ; preds = %L.LB2_351
  %80 = load i32, i32* %__gtid___nv_MAIN__F1L19_1__386, align 4, !dbg !46
  call void @__kmpc_barrier(i64* null, i32 %80), !dbg !46
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

declare void @__kmpc_barrier(i64*, i32) #0

declare void @__kmpc_for_static_fini(i64*, i32) #0

declare void @__kmpc_for_static_init_4(i64*, i32, i32, i64*, i64*, i64*, i64*, i32, i32) #0

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
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB140-reduction-barrier-orig-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb140_reduction_barrier_orig_yes", scope: !2, file: !3, line: 13, type: !6, scopeLine: 13, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!6 = !DISubroutineType(cc: DW_CC_program, types: !7)
!7 = !{null}
!8 = !DILocalVariable(name: "omp_sched_static", scope: !5, file: !3, type: !9)
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 0, scope: !5)
!11 = !DILocalVariable(name: "omp_proc_bind_false", scope: !5, file: !3, type: !9)
!12 = !DILocalVariable(name: "omp_proc_bind_true", scope: !5, file: !3, type: !9)
!13 = !DILocalVariable(name: "omp_lock_hint_none", scope: !5, file: !3, type: !9)
!14 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !5, file: !3, type: !9)
!15 = !DILocation(line: 35, column: 1, scope: !5)
!16 = !DILocation(line: 13, column: 1, scope: !5)
!17 = !DILocalVariable(name: "a", scope: !5, file: !3, type: !9)
!18 = !DILocation(line: 19, column: 1, scope: !5)
!19 = distinct !DISubprogram(name: "__nv_MAIN__F1L19_1", scope: !2, file: !3, line: 19, type: !20, scopeLine: 19, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!20 = !DISubroutineType(types: !21)
!21 = !{null, !9, !22, !22}
!22 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!23 = !DILocalVariable(name: "__nv_MAIN__F1L19_1Arg0", arg: 1, scope: !19, file: !3, type: !9)
!24 = !DILocation(line: 0, scope: !19)
!25 = !DILocalVariable(name: "__nv_MAIN__F1L19_1Arg1", arg: 2, scope: !19, file: !3, type: !22)
!26 = !DILocalVariable(name: "__nv_MAIN__F1L19_1Arg2", arg: 3, scope: !19, file: !3, type: !22)
!27 = !DILocalVariable(name: "omp_sched_static", scope: !19, file: !3, type: !9)
!28 = !DILocalVariable(name: "omp_proc_bind_false", scope: !19, file: !3, type: !9)
!29 = !DILocalVariable(name: "omp_proc_bind_true", scope: !19, file: !3, type: !9)
!30 = !DILocalVariable(name: "omp_lock_hint_none", scope: !19, file: !3, type: !9)
!31 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !19, file: !3, type: !9)
!32 = !DILocation(line: 34, column: 1, scope: !19)
!33 = !DILocation(line: 20, column: 1, scope: !19)
!34 = !DILocation(line: 21, column: 1, scope: !19)
!35 = !DILocation(line: 22, column: 1, scope: !19)
!36 = !DILocalVariable(name: "a", scope: !19, file: !3, type: !9)
!37 = !DILocation(line: 24, column: 1, scope: !19)
!38 = !DILocation(line: 25, column: 1, scope: !19)
!39 = !DILocalVariable(name: "i", scope: !19, file: !3, type: !9)
!40 = !DILocation(line: 26, column: 1, scope: !19)
!41 = !DILocation(line: 27, column: 1, scope: !19)
!42 = !DILocation(line: 28, column: 1, scope: !19)
!43 = !DILocation(line: 30, column: 1, scope: !19)
!44 = !DILocation(line: 31, column: 1, scope: !19)
!45 = !DILocalVariable(scope: !19, file: !3, type: !9, flags: DIFlagArtificial)
!46 = !DILocation(line: 32, column: 1, scope: !19)

; ModuleID = '/tmp/DRB076-flush-orig-no-7fd9d6.ll'
source_filename = "/tmp/DRB076-flush-orig-no-7fd9d6.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%astruct.dt63 = type <{ i8* }>

@.C285_drb076_f1_ = internal constant i32 1
@.C307_MAIN_ = internal constant i32 25
@.C306_MAIN_ = internal constant i32 14
@.C326_MAIN_ = internal constant [5 x i8] c"sum ="
@.C284_MAIN_ = internal constant i64 0
@.C323_MAIN_ = internal constant i32 6
@.C320_MAIN_ = internal constant [49 x i8] c"micro-benchmarks-fortran/DRB076-flush-orig-no.f95"
@.C322_MAIN_ = internal constant i32 39
@.C335_MAIN_ = internal constant i32 10
@.C312_MAIN_ = internal constant i32 10
@.C283_MAIN_ = internal constant i32 0
@.C283___nv_MAIN__F1L33_1 = internal constant i32 0

; Function Attrs: noinline
define float @drb076_() #0 {
.L.entry:
  ret float undef
}

define void @drb076_f1_(i64* %q) #1 !dbg !5 {
L.entry:
  call void @llvm.dbg.declare(metadata i64* %q, metadata !10, metadata !DIExpression()), !dbg !11
  call void @llvm.dbg.value(metadata i32 1, metadata !12, metadata !DIExpression()), !dbg !11
  call void @llvm.dbg.value(metadata i32 0, metadata !13, metadata !DIExpression()), !dbg !11
  call void @llvm.dbg.value(metadata i32 1, metadata !14, metadata !DIExpression()), !dbg !11
  call void @llvm.dbg.value(metadata i32 0, metadata !15, metadata !DIExpression()), !dbg !11
  call void @llvm.dbg.value(metadata i32 1, metadata !16, metadata !DIExpression()), !dbg !11
  br label %L.LB2_310

L.LB2_310:                                        ; preds = %L.entry
  %0 = bitcast i64* %q to i32*, !dbg !17
  store i32 1, i32* %0, align 4, !dbg !17
  ret void, !dbg !18
}

define void @MAIN_() #1 !dbg !19 {
L.entry:
  %__gtid_MAIN__350 = alloca i32, align 4
  %i_310 = alloca i32, align 4
  %sum_311 = alloca i32, align 4
  %.uplevelArgPack0001_345 = alloca %astruct.dt63, align 8
  %z__io_325 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !22, metadata !DIExpression()), !dbg !23
  call void @llvm.dbg.value(metadata i32 0, metadata !24, metadata !DIExpression()), !dbg !23
  call void @llvm.dbg.value(metadata i32 1, metadata !25, metadata !DIExpression()), !dbg !23
  call void @llvm.dbg.value(metadata i32 0, metadata !26, metadata !DIExpression()), !dbg !23
  call void @llvm.dbg.value(metadata i32 1, metadata !27, metadata !DIExpression()), !dbg !23
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !28
  store i32 %0, i32* %__gtid_MAIN__350, align 4, !dbg !28
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !29
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !29
  call void (i8*, ...) %2(i8* %1), !dbg !29
  br label %L.LB3_338

L.LB3_338:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %i_310, metadata !30, metadata !DIExpression()), !dbg !23
  store i32 0, i32* %i_310, align 4, !dbg !31
  call void @llvm.dbg.declare(metadata i32* %sum_311, metadata !32, metadata !DIExpression()), !dbg !23
  store i32 0, i32* %sum_311, align 4, !dbg !33
  %3 = bitcast i32* %sum_311 to i8*, !dbg !34
  %4 = bitcast %astruct.dt63* %.uplevelArgPack0001_345 to i8**, !dbg !34
  store i8* %3, i8** %4, align 8, !dbg !34
  br label %L.LB3_348, !dbg !34

L.LB3_348:                                        ; preds = %L.LB3_338
  %5 = load i32, i32* %__gtid_MAIN__350, align 4, !dbg !34
  call void @__kmpc_push_num_threads(i64* null, i32 %5, i32 10), !dbg !34
  %6 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L33_1_ to i64*, !dbg !34
  %7 = bitcast %astruct.dt63* %.uplevelArgPack0001_345 to i64*, !dbg !34
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %6, i64* %7), !dbg !34
  %8 = load i32, i32* %sum_311, align 4, !dbg !35
  call void @llvm.dbg.value(metadata i32 %8, metadata !32, metadata !DIExpression()), !dbg !23
  %9 = icmp eq i32 %8, 10, !dbg !35
  br i1 %9, label %L.LB3_336, label %L.LB3_381, !dbg !35

L.LB3_381:                                        ; preds = %L.LB3_348
  call void (...) @_mp_bcs_nest(), !dbg !36
  %10 = bitcast i32* @.C322_MAIN_ to i8*, !dbg !36
  %11 = bitcast [49 x i8]* @.C320_MAIN_ to i8*, !dbg !36
  %12 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !36
  call void (i8*, i8*, i64, ...) %12(i8* %10, i8* %11, i64 49), !dbg !36
  %13 = bitcast i32* @.C323_MAIN_ to i8*, !dbg !36
  %14 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !36
  %15 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !36
  %16 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !36
  %17 = call i32 (i8*, i8*, i8*, i8*, ...) %16(i8* %13, i8* null, i8* %14, i8* %15), !dbg !36
  call void @llvm.dbg.declare(metadata i32* %z__io_325, metadata !37, metadata !DIExpression()), !dbg !23
  store i32 %17, i32* %z__io_325, align 4, !dbg !36
  %18 = bitcast [5 x i8]* @.C326_MAIN_ to i8*, !dbg !36
  %19 = bitcast i32 (...)* @f90io_sc_ch_ldw to i32 (i8*, i32, i64, ...)*, !dbg !36
  %20 = call i32 (i8*, i32, i64, ...) %19(i8* %18, i32 14, i64 5), !dbg !36
  store i32 %20, i32* %z__io_325, align 4, !dbg !36
  %21 = load i32, i32* %sum_311, align 4, !dbg !36
  call void @llvm.dbg.value(metadata i32 %21, metadata !32, metadata !DIExpression()), !dbg !23
  %22 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !36
  %23 = call i32 (i32, i32, ...) %22(i32 %21, i32 25), !dbg !36
  store i32 %23, i32* %z__io_325, align 4, !dbg !36
  %24 = call i32 (...) @f90io_ldw_end(), !dbg !36
  store i32 %24, i32* %z__io_325, align 4, !dbg !36
  call void (...) @_mp_ecs_nest(), !dbg !36
  br label %L.LB3_336

L.LB3_336:                                        ; preds = %L.LB3_381, %L.LB3_348
  ret void, !dbg !28
}

define internal void @__nv_MAIN__F1L33_1_(i32* %__nv_MAIN__F1L33_1Arg0, i64* %__nv_MAIN__F1L33_1Arg1, i64* %__nv_MAIN__F1L33_1Arg2) #1 !dbg !38 {
L.entry:
  %sum_317 = alloca i32, align 4
  %i_316 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L33_1Arg0, metadata !42, metadata !DIExpression()), !dbg !43
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L33_1Arg1, metadata !44, metadata !DIExpression()), !dbg !43
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L33_1Arg2, metadata !45, metadata !DIExpression()), !dbg !43
  call void @llvm.dbg.value(metadata i32 1, metadata !46, metadata !DIExpression()), !dbg !43
  call void @llvm.dbg.value(metadata i32 0, metadata !47, metadata !DIExpression()), !dbg !43
  call void @llvm.dbg.value(metadata i32 1, metadata !48, metadata !DIExpression()), !dbg !43
  call void @llvm.dbg.value(metadata i32 0, metadata !49, metadata !DIExpression()), !dbg !43
  call void @llvm.dbg.value(metadata i32 1, metadata !50, metadata !DIExpression()), !dbg !43
  br label %L.LB4_385

L.LB4_385:                                        ; preds = %L.entry
  br label %L.LB4_315

L.LB4_315:                                        ; preds = %L.LB4_385
  call void @llvm.dbg.declare(metadata i32* %sum_317, metadata !51, metadata !DIExpression()), !dbg !52
  store i32 0, i32* %sum_317, align 4, !dbg !53
  call void @llvm.dbg.declare(metadata i32* %i_316, metadata !54, metadata !DIExpression()), !dbg !52
  %0 = bitcast i32* %i_316 to i64*, !dbg !55
  call void @drb076_f1_(i64* %0), !dbg !55
  %1 = load i32, i32* %i_316, align 4, !dbg !56
  call void @llvm.dbg.value(metadata i32 %1, metadata !54, metadata !DIExpression()), !dbg !52
  %2 = load i32, i32* %sum_317, align 4, !dbg !56
  call void @llvm.dbg.value(metadata i32 %2, metadata !51, metadata !DIExpression()), !dbg !52
  %3 = add nsw i32 %1, %2, !dbg !56
  store i32 %3, i32* %sum_317, align 4, !dbg !56
  %4 = call i32 (...) @_mp_bcs_nest_red(), !dbg !52
  %5 = call i32 (...) @_mp_bcs_nest_red(), !dbg !52
  %6 = load i32, i32* %sum_317, align 4, !dbg !52
  call void @llvm.dbg.value(metadata i32 %6, metadata !51, metadata !DIExpression()), !dbg !52
  %7 = bitcast i64* %__nv_MAIN__F1L33_1Arg2 to i32**, !dbg !52
  %8 = load i32*, i32** %7, align 8, !dbg !52
  %9 = load i32, i32* %8, align 4, !dbg !52
  %10 = add nsw i32 %6, %9, !dbg !52
  %11 = bitcast i64* %__nv_MAIN__F1L33_1Arg2 to i32**, !dbg !52
  %12 = load i32*, i32** %11, align 8, !dbg !52
  store i32 %10, i32* %12, align 4, !dbg !52
  %13 = call i32 (...) @_mp_ecs_nest_red(), !dbg !52
  %14 = call i32 (...) @_mp_ecs_nest_red(), !dbg !52
  br label %L.LB4_318

L.LB4_318:                                        ; preds = %L.LB4_315
  ret void, !dbg !52
}

declare signext i32 @_mp_ecs_nest_red(...) #1

declare signext i32 @_mp_bcs_nest_red(...) #1

declare void @_mp_ecs_nest(...) #1

declare signext i32 @f90io_ldw_end(...) #1

declare signext i32 @f90io_sc_i_ldw(...) #1

declare signext i32 @f90io_sc_ch_ldw(...) #1

declare signext i32 @f90io_print_init(...) #1

declare void @f90io_src_info03a(...) #1

declare void @_mp_bcs_nest(...) #1

declare void @fort_init(...) #1

declare signext i32 @__kmpc_global_thread_num(i64*) #1

declare void @__kmpc_fork_call(i64*, i32, i64*, i64*, ...) #1

declare void @__kmpc_push_num_threads(i64*, i32, i32) #1

declare void @__kmpc_end_serialized_parallel(i64*, i32) #1

declare void @__kmpc_serialized_parallel(i64*, i32) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { noinline }
attributes #1 = { "no-frame-pointer-elim-non-leaf" }
attributes #2 = { nounwind readnone speculatable }

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !4, globals: !4, imports: !4)
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB076-flush-orig-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "f1", scope: !6, file: !3, line: 18, type: !7, scopeLine: 18, spFlags: DISPFlagDefinition, unit: !2)
!6 = !DIModule(scope: !2, name: "drb076")
!7 = !DISubroutineType(types: !8)
!8 = !{null, !9}
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DILocalVariable(name: "q", arg: 1, scope: !5, file: !3, type: !9)
!11 = !DILocation(line: 0, scope: !5)
!12 = !DILocalVariable(name: "omp_sched_static", scope: !5, file: !3, type: !9)
!13 = !DILocalVariable(name: "omp_proc_bind_false", scope: !5, file: !3, type: !9)
!14 = !DILocalVariable(name: "omp_proc_bind_true", scope: !5, file: !3, type: !9)
!15 = !DILocalVariable(name: "omp_lock_hint_none", scope: !5, file: !3, type: !9)
!16 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !5, file: !3, type: !9)
!17 = !DILocation(line: 20, column: 1, scope: !5)
!18 = !DILocation(line: 21, column: 1, scope: !5)
!19 = distinct !DISubprogram(name: "drb076_flush_orig_no", scope: !2, file: !3, line: 24, type: !20, scopeLine: 24, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!20 = !DISubroutineType(cc: DW_CC_program, types: !21)
!21 = !{null}
!22 = !DILocalVariable(name: "omp_sched_static", scope: !19, file: !3, type: !9)
!23 = !DILocation(line: 0, scope: !19)
!24 = !DILocalVariable(name: "omp_proc_bind_false", scope: !19, file: !3, type: !9)
!25 = !DILocalVariable(name: "omp_proc_bind_true", scope: !19, file: !3, type: !9)
!26 = !DILocalVariable(name: "omp_lock_hint_none", scope: !19, file: !3, type: !9)
!27 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !19, file: !3, type: !9)
!28 = !DILocation(line: 41, column: 1, scope: !19)
!29 = !DILocation(line: 24, column: 1, scope: !19)
!30 = !DILocalVariable(name: "i", scope: !19, file: !3, type: !9)
!31 = !DILocation(line: 30, column: 1, scope: !19)
!32 = !DILocalVariable(name: "sum", scope: !19, file: !3, type: !9)
!33 = !DILocation(line: 31, column: 1, scope: !19)
!34 = !DILocation(line: 33, column: 1, scope: !19)
!35 = !DILocation(line: 38, column: 1, scope: !19)
!36 = !DILocation(line: 39, column: 1, scope: !19)
!37 = !DILocalVariable(scope: !19, file: !3, type: !9, flags: DIFlagArtificial)
!38 = distinct !DISubprogram(name: "__nv_MAIN__F1L33_1", scope: !2, file: !3, line: 33, type: !39, scopeLine: 33, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!39 = !DISubroutineType(types: !40)
!40 = !{null, !9, !41, !41}
!41 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!42 = !DILocalVariable(name: "__nv_MAIN__F1L33_1Arg0", arg: 1, scope: !38, file: !3, type: !9)
!43 = !DILocation(line: 0, scope: !38)
!44 = !DILocalVariable(name: "__nv_MAIN__F1L33_1Arg1", arg: 2, scope: !38, file: !3, type: !41)
!45 = !DILocalVariable(name: "__nv_MAIN__F1L33_1Arg2", arg: 3, scope: !38, file: !3, type: !41)
!46 = !DILocalVariable(name: "omp_sched_static", scope: !38, file: !3, type: !9)
!47 = !DILocalVariable(name: "omp_proc_bind_false", scope: !38, file: !3, type: !9)
!48 = !DILocalVariable(name: "omp_proc_bind_true", scope: !38, file: !3, type: !9)
!49 = !DILocalVariable(name: "omp_lock_hint_none", scope: !38, file: !3, type: !9)
!50 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !38, file: !3, type: !9)
!51 = !DILocalVariable(name: "sum", scope: !38, file: !3, type: !9)
!52 = !DILocation(line: 36, column: 1, scope: !38)
!53 = !DILocation(line: 33, column: 1, scope: !38)
!54 = !DILocalVariable(name: "i", scope: !38, file: !3, type: !9)
!55 = !DILocation(line: 34, column: 1, scope: !38)
!56 = !DILocation(line: 35, column: 1, scope: !38)

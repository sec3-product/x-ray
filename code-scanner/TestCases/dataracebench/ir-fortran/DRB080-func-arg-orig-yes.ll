; ModuleID = '/tmp/DRB080-func-arg-orig-yes-525a23.ll'
source_filename = "/tmp/DRB080-func-arg-orig-yes-525a23.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%astruct.dt60 = type <{ i8* }>

@.C285_f1_ = internal constant i32 1
@.C306_MAIN_ = internal constant i32 25
@.C305_MAIN_ = internal constant i32 14
@.C320_MAIN_ = internal constant [4 x i8] c"i = "
@.C284_MAIN_ = internal constant i64 0
@.C317_MAIN_ = internal constant i32 6
@.C314_MAIN_ = internal constant [53 x i8] c"micro-benchmarks-fortran/DRB080-func-arg-orig-yes.f95"
@.C316_MAIN_ = internal constant i32 30
@.C283_MAIN_ = internal constant i32 0

define void @f1_(i64* %q) #0 !dbg !5 {
L.entry:
  call void @llvm.dbg.declare(metadata i64* %q, metadata !9, metadata !DIExpression()), !dbg !10
  br label %L.LB1_302

L.LB1_302:                                        ; preds = %L.entry
  %0 = bitcast i64* %q to i32*, !dbg !11
  %1 = load i32, i32* %0, align 4, !dbg !11
  %2 = add nsw i32 %1, 1, !dbg !11
  %3 = bitcast i64* %q to i32*, !dbg !11
  store i32 %2, i32* %3, align 4, !dbg !11
  ret void, !dbg !12
}

define void @MAIN_() #0 !dbg !13 {
L.entry:
  %__gtid_MAIN__341 = alloca i32, align 4
  %i_307 = alloca i32, align 4
  %.uplevelArgPack0001_336 = alloca %astruct.dt60, align 8
  %z__io_319 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !16, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.value(metadata i32 0, metadata !18, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.value(metadata i32 1, metadata !19, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.value(metadata i32 0, metadata !20, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.value(metadata i32 1, metadata !21, metadata !DIExpression()), !dbg !17
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !22
  store i32 %0, i32* %__gtid_MAIN__341, align 4, !dbg !22
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !23
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !23
  call void (i8*, ...) %2(i8* %1), !dbg !23
  br label %L.LB2_330

L.LB2_330:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %i_307, metadata !24, metadata !DIExpression()), !dbg !17
  store i32 0, i32* %i_307, align 4, !dbg !25
  %3 = bitcast i32* %i_307 to i8*, !dbg !26
  %4 = bitcast %astruct.dt60* %.uplevelArgPack0001_336 to i8**, !dbg !26
  store i8* %3, i8** %4, align 8, !dbg !26
  br label %L.LB2_339, !dbg !26

L.LB2_339:                                        ; preds = %L.LB2_330
  %5 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L26_1_ to i64*, !dbg !26
  %6 = bitcast %astruct.dt60* %.uplevelArgPack0001_336 to i64*, !dbg !26
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %5, i64* %6), !dbg !26
  call void (...) @_mp_bcs_nest(), !dbg !27
  %7 = bitcast i32* @.C316_MAIN_ to i8*, !dbg !27
  %8 = bitcast [53 x i8]* @.C314_MAIN_ to i8*, !dbg !27
  %9 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !27
  call void (i8*, i8*, i64, ...) %9(i8* %7, i8* %8, i64 53), !dbg !27
  %10 = bitcast i32* @.C317_MAIN_ to i8*, !dbg !27
  %11 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !27
  %12 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !27
  %13 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !27
  %14 = call i32 (i8*, i8*, i8*, i8*, ...) %13(i8* %10, i8* null, i8* %11, i8* %12), !dbg !27
  call void @llvm.dbg.declare(metadata i32* %z__io_319, metadata !28, metadata !DIExpression()), !dbg !17
  store i32 %14, i32* %z__io_319, align 4, !dbg !27
  %15 = bitcast [4 x i8]* @.C320_MAIN_ to i8*, !dbg !27
  %16 = bitcast i32 (...)* @f90io_sc_ch_ldw to i32 (i8*, i32, i64, ...)*, !dbg !27
  %17 = call i32 (i8*, i32, i64, ...) %16(i8* %15, i32 14, i64 4), !dbg !27
  store i32 %17, i32* %z__io_319, align 4, !dbg !27
  %18 = load i32, i32* %i_307, align 4, !dbg !27
  call void @llvm.dbg.value(metadata i32 %18, metadata !24, metadata !DIExpression()), !dbg !17
  %19 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !27
  %20 = call i32 (i32, i32, ...) %19(i32 %18, i32 25), !dbg !27
  store i32 %20, i32* %z__io_319, align 4, !dbg !27
  %21 = call i32 (...) @f90io_ldw_end(), !dbg !27
  store i32 %21, i32* %z__io_319, align 4, !dbg !27
  call void (...) @_mp_ecs_nest(), !dbg !27
  ret void, !dbg !22
}

define internal void @__nv_MAIN__F1L26_1_(i32* %__nv_MAIN__F1L26_1Arg0, i64* %__nv_MAIN__F1L26_1Arg1, i64* %__nv_MAIN__F1L26_1Arg2) #0 !dbg !29 {
L.entry:
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L26_1Arg0, metadata !33, metadata !DIExpression()), !dbg !34
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L26_1Arg1, metadata !35, metadata !DIExpression()), !dbg !34
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L26_1Arg2, metadata !36, metadata !DIExpression()), !dbg !34
  call void @llvm.dbg.value(metadata i32 1, metadata !37, metadata !DIExpression()), !dbg !34
  call void @llvm.dbg.value(metadata i32 0, metadata !38, metadata !DIExpression()), !dbg !34
  call void @llvm.dbg.value(metadata i32 1, metadata !39, metadata !DIExpression()), !dbg !34
  call void @llvm.dbg.value(metadata i32 0, metadata !40, metadata !DIExpression()), !dbg !34
  call void @llvm.dbg.value(metadata i32 1, metadata !41, metadata !DIExpression()), !dbg !34
  br label %L.LB3_371

L.LB3_371:                                        ; preds = %L.entry
  br label %L.LB3_310

L.LB3_310:                                        ; preds = %L.LB3_371
  %0 = bitcast i64* %__nv_MAIN__F1L26_1Arg2 to i64**, !dbg !42
  %1 = load i64*, i64** %0, align 8, !dbg !42
  call void @f1_(i64* %1), !dbg !42
  br label %L.LB3_312

L.LB3_312:                                        ; preds = %L.LB3_310
  ret void, !dbg !43
}

declare void @_mp_ecs_nest(...) #0

declare signext i32 @f90io_ldw_end(...) #0

declare signext i32 @f90io_sc_i_ldw(...) #0

declare signext i32 @f90io_sc_ch_ldw(...) #0

declare signext i32 @f90io_print_init(...) #0

declare void @f90io_src_info03a(...) #0

declare void @_mp_bcs_nest(...) #0

declare void @fort_init(...) #0

declare signext i32 @__kmpc_global_thread_num(i64*) #0

declare void @__kmpc_fork_call(i64*, i32, i64*, i64*, ...) #0

declare void @__kmpc_end_serialized_parallel(i64*, i32) #0

declare void @__kmpc_serialized_parallel(i64*, i32) #0

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { "no-frame-pointer-elim-non-leaf" }
attributes #1 = { nounwind readnone speculatable }

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !4, globals: !4, imports: !4)
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB080-func-arg-orig-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "f1", scope: !2, file: !3, line: 12, type: !6, scopeLine: 12, spFlags: DISPFlagDefinition, unit: !2)
!6 = !DISubroutineType(types: !7)
!7 = !{null, !8}
!8 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !DILocalVariable(name: "q", arg: 1, scope: !5, file: !3, type: !8)
!10 = !DILocation(line: 0, scope: !5)
!11 = !DILocation(line: 15, column: 1, scope: !5)
!12 = !DILocation(line: 16, column: 1, scope: !5)
!13 = distinct !DISubprogram(name: "drb080_func_arg_orig_yes", scope: !2, file: !3, line: 18, type: !14, scopeLine: 18, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!14 = !DISubroutineType(cc: DW_CC_program, types: !15)
!15 = !{null}
!16 = !DILocalVariable(name: "omp_sched_static", scope: !13, file: !3, type: !8)
!17 = !DILocation(line: 0, scope: !13)
!18 = !DILocalVariable(name: "omp_proc_bind_false", scope: !13, file: !3, type: !8)
!19 = !DILocalVariable(name: "omp_proc_bind_true", scope: !13, file: !3, type: !8)
!20 = !DILocalVariable(name: "omp_lock_hint_none", scope: !13, file: !3, type: !8)
!21 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !13, file: !3, type: !8)
!22 = !DILocation(line: 31, column: 1, scope: !13)
!23 = !DILocation(line: 18, column: 1, scope: !13)
!24 = !DILocalVariable(name: "i", scope: !13, file: !3, type: !8)
!25 = !DILocation(line: 24, column: 1, scope: !13)
!26 = !DILocation(line: 26, column: 1, scope: !13)
!27 = !DILocation(line: 30, column: 1, scope: !13)
!28 = !DILocalVariable(scope: !13, file: !3, type: !8, flags: DIFlagArtificial)
!29 = distinct !DISubprogram(name: "__nv_MAIN__F1L26_1", scope: !2, file: !3, line: 26, type: !30, scopeLine: 26, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!30 = !DISubroutineType(types: !31)
!31 = !{null, !8, !32, !32}
!32 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!33 = !DILocalVariable(name: "__nv_MAIN__F1L26_1Arg0", arg: 1, scope: !29, file: !3, type: !8)
!34 = !DILocation(line: 0, scope: !29)
!35 = !DILocalVariable(name: "__nv_MAIN__F1L26_1Arg1", arg: 2, scope: !29, file: !3, type: !32)
!36 = !DILocalVariable(name: "__nv_MAIN__F1L26_1Arg2", arg: 3, scope: !29, file: !3, type: !32)
!37 = !DILocalVariable(name: "omp_sched_static", scope: !29, file: !3, type: !8)
!38 = !DILocalVariable(name: "omp_proc_bind_false", scope: !29, file: !3, type: !8)
!39 = !DILocalVariable(name: "omp_proc_bind_true", scope: !29, file: !3, type: !8)
!40 = !DILocalVariable(name: "omp_lock_hint_none", scope: !29, file: !3, type: !8)
!41 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !29, file: !3, type: !8)
!42 = !DILocation(line: 27, column: 1, scope: !29)
!43 = !DILocation(line: 28, column: 1, scope: !29)

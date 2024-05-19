; ModuleID = '/tmp/DRB081-func-arg-orig-no-40238a.ll'
source_filename = "/tmp/DRB081-func-arg-orig-no-40238a.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.STATICS3 = type <{ [40 x i8] }>
%astruct.dt63 = type <{ i8* }>

@.C285_global_f1_ = internal constant i32 1
@.STATICS3 = internal global %struct.STATICS3 <{ [40 x i8] c"\FB\FF\FF\FF\03\00\00\00i =\00\EB\FF\FF\FF\00\00\00\00\03\00\00\00\00\00\00\00\01\00\00\00\FF\FF\FF\FF\00\00\00\00" }>, align 16
@.C306_MAIN_ = internal constant i32 25
@.C284_MAIN_ = internal constant i64 0
@.C319_MAIN_ = internal constant i32 6
@.C315_MAIN_ = internal constant [52 x i8] c"micro-benchmarks-fortran/DRB081-func-arg-orig-no.f95"
@.C317_MAIN_ = internal constant i32 33
@.C283_MAIN_ = internal constant i32 0

; Function Attrs: noinline
define float @global_() #0 {
.L.entry:
  ret float undef
}

define void @global_f1_(i32 %_V_i.arg) #1 !dbg !5 {
L.entry:
  %_V_i.addr = alloca i32, align 4
  %i_298 = alloca i32, align 4
  %"global_f1___$eq_299" = alloca [16 x i8], align 4
  call void @llvm.dbg.declare(metadata i32* %_V_i.addr, metadata !10, metadata !DIExpression()), !dbg !11
  store i32 %_V_i.arg, i32* %_V_i.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %_V_i.addr, metadata !12, metadata !DIExpression()), !dbg !11
  %0 = load i32, i32* %_V_i.addr, align 4, !dbg !13
  call void @llvm.dbg.value(metadata i32 %0, metadata !10, metadata !DIExpression()), !dbg !11
  call void @llvm.dbg.declare(metadata i32* %i_298, metadata !14, metadata !DIExpression()), !dbg !11
  store i32 %0, i32* %i_298, align 4, !dbg !13
  br label %L.LB2_306

L.LB2_306:                                        ; preds = %L.entry
  %1 = load i32, i32* %i_298, align 4, !dbg !15
  call void @llvm.dbg.value(metadata i32 %1, metadata !14, metadata !DIExpression()), !dbg !11
  %2 = add nsw i32 %1, 1, !dbg !15
  store i32 %2, i32* %i_298, align 4, !dbg !15
  ret void, !dbg !16
}

define void @MAIN_() #1 !dbg !17 {
L.entry:
  %__gtid_MAIN__343 = alloca i32, align 4
  %i_309 = alloca i32, align 4
  %.uplevelArgPack0001_338 = alloca %astruct.dt63, align 8
  %z__io_321 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !20, metadata !DIExpression()), !dbg !21
  call void @llvm.dbg.value(metadata i32 0, metadata !22, metadata !DIExpression()), !dbg !21
  call void @llvm.dbg.value(metadata i32 1, metadata !23, metadata !DIExpression()), !dbg !21
  call void @llvm.dbg.value(metadata i32 0, metadata !24, metadata !DIExpression()), !dbg !21
  call void @llvm.dbg.value(metadata i32 1, metadata !25, metadata !DIExpression()), !dbg !21
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !26
  store i32 %0, i32* %__gtid_MAIN__343, align 4, !dbg !26
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !27
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !27
  call void (i8*, ...) %2(i8* %1), !dbg !27
  br label %L.LB3_332

L.LB3_332:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %i_309, metadata !28, metadata !DIExpression()), !dbg !21
  store i32 0, i32* %i_309, align 4, !dbg !29
  %3 = bitcast i32* %i_309 to i8*, !dbg !30
  %4 = bitcast %astruct.dt63* %.uplevelArgPack0001_338 to i8**, !dbg !30
  store i8* %3, i8** %4, align 8, !dbg !30
  br label %L.LB3_341, !dbg !30

L.LB3_341:                                        ; preds = %L.LB3_332
  %5 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L28_1_ to i64*, !dbg !30
  %6 = bitcast %astruct.dt63* %.uplevelArgPack0001_338 to i64*, !dbg !30
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %5, i64* %6), !dbg !30
  %7 = load i32, i32* %i_309, align 4, !dbg !31
  call void @llvm.dbg.value(metadata i32 %7, metadata !28, metadata !DIExpression()), !dbg !21
  %8 = icmp eq i32 %7, 0, !dbg !31
  br i1 %8, label %L.LB3_330, label %L.LB3_369, !dbg !31

L.LB3_369:                                        ; preds = %L.LB3_341
  call void (...) @_mp_bcs_nest(), !dbg !32
  %9 = bitcast i32* @.C317_MAIN_ to i8*, !dbg !32
  %10 = bitcast [52 x i8]* @.C315_MAIN_ to i8*, !dbg !32
  %11 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !32
  call void (i8*, i8*, i64, ...) %11(i8* %9, i8* %10, i64 52), !dbg !32
  %12 = bitcast i32* @.C319_MAIN_ to i8*, !dbg !32
  %13 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !32
  %14 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !32
  %15 = bitcast %struct.STATICS3* @.STATICS3 to i8*, !dbg !32
  %16 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !32
  %17 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %16(i8* %12, i8* null, i8* %13, i8* %14, i8* %15, i8* null, i64 0), !dbg !32
  call void @llvm.dbg.declare(metadata i32* %z__io_321, metadata !33, metadata !DIExpression()), !dbg !21
  store i32 %17, i32* %z__io_321, align 4, !dbg !32
  %18 = load i32, i32* %i_309, align 4, !dbg !32
  call void @llvm.dbg.value(metadata i32 %18, metadata !28, metadata !DIExpression()), !dbg !21
  %19 = bitcast i32 (...)* @f90io_sc_i_fmt_write to i32 (i32, i32, ...)*, !dbg !32
  %20 = call i32 (i32, i32, ...) %19(i32 %18, i32 25), !dbg !32
  store i32 %20, i32* %z__io_321, align 4, !dbg !32
  %21 = call i32 (...) @f90io_fmtw_end(), !dbg !32
  store i32 %21, i32* %z__io_321, align 4, !dbg !32
  call void (...) @_mp_ecs_nest(), !dbg !32
  br label %L.LB3_330

L.LB3_330:                                        ; preds = %L.LB3_369, %L.LB3_341
  ret void, !dbg !26
}

define internal void @__nv_MAIN__F1L28_1_(i32* %__nv_MAIN__F1L28_1Arg0, i64* %__nv_MAIN__F1L28_1Arg1, i64* %__nv_MAIN__F1L28_1Arg2) #1 !dbg !34 {
L.entry:
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L28_1Arg0, metadata !38, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L28_1Arg1, metadata !40, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L28_1Arg2, metadata !41, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.value(metadata i32 1, metadata !42, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.value(metadata i32 0, metadata !43, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.value(metadata i32 1, metadata !44, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.value(metadata i32 0, metadata !45, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.value(metadata i32 1, metadata !46, metadata !DIExpression()), !dbg !39
  br label %L.LB4_373

L.LB4_373:                                        ; preds = %L.entry
  br label %L.LB4_312

L.LB4_312:                                        ; preds = %L.LB4_373
  %0 = bitcast i64* %__nv_MAIN__F1L28_1Arg2 to i32**, !dbg !47
  %1 = load i32*, i32** %0, align 8, !dbg !47
  %2 = load i32, i32* %1, align 4, !dbg !47
  call void @global_f1_(i32 %2), !dbg !47
  br label %L.LB4_313

L.LB4_313:                                        ; preds = %L.LB4_312
  ret void, !dbg !48
}

declare void @_mp_ecs_nest(...) #1

declare signext i32 @f90io_fmtw_end(...) #1

declare signext i32 @f90io_sc_i_fmt_write(...) #1

declare signext i32 @f90io_fmtw_inita(...) #1

declare void @f90io_src_info03a(...) #1

declare void @_mp_bcs_nest(...) #1

declare void @fort_init(...) #1

declare signext i32 @__kmpc_global_thread_num(i64*) #1

declare void @__kmpc_fork_call(i64*, i32, i64*, i64*, ...) #1

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
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB081-func-arg-orig-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "f1", scope: !6, file: !3, line: 14, type: !7, scopeLine: 14, spFlags: DISPFlagDefinition, unit: !2)
!6 = !DIModule(scope: !2, name: "global")
!7 = !DISubroutineType(types: !8)
!8 = !{null, !9}
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DILocalVariable(name: "_V_i", scope: !5, file: !3, type: !9)
!11 = !DILocation(line: 0, scope: !5)
!12 = !DILocalVariable(name: "_V_i", arg: 1, scope: !5, file: !3, type: !9)
!13 = !DILocation(line: 14, column: 1, scope: !5)
!14 = !DILocalVariable(name: "i", scope: !5, file: !3, type: !9)
!15 = !DILocation(line: 16, column: 1, scope: !5)
!16 = !DILocation(line: 17, column: 1, scope: !5)
!17 = distinct !DISubprogram(name: "drb080_func_arg_orig_yes", scope: !2, file: !3, line: 20, type: !18, scopeLine: 20, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!18 = !DISubroutineType(cc: DW_CC_program, types: !19)
!19 = !{null}
!20 = !DILocalVariable(name: "omp_sched_static", scope: !17, file: !3, type: !9)
!21 = !DILocation(line: 0, scope: !17)
!22 = !DILocalVariable(name: "omp_proc_bind_false", scope: !17, file: !3, type: !9)
!23 = !DILocalVariable(name: "omp_proc_bind_true", scope: !17, file: !3, type: !9)
!24 = !DILocalVariable(name: "omp_lock_hint_none", scope: !17, file: !3, type: !9)
!25 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !17, file: !3, type: !9)
!26 = !DILocation(line: 36, column: 1, scope: !17)
!27 = !DILocation(line: 20, column: 1, scope: !17)
!28 = !DILocalVariable(name: "i", scope: !17, file: !3, type: !9)
!29 = !DILocation(line: 26, column: 1, scope: !17)
!30 = !DILocation(line: 28, column: 1, scope: !17)
!31 = !DILocation(line: 32, column: 1, scope: !17)
!32 = !DILocation(line: 33, column: 1, scope: !17)
!33 = !DILocalVariable(scope: !17, file: !3, type: !9, flags: DIFlagArtificial)
!34 = distinct !DISubprogram(name: "__nv_MAIN__F1L28_1", scope: !2, file: !3, line: 28, type: !35, scopeLine: 28, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!35 = !DISubroutineType(types: !36)
!36 = !{null, !9, !37, !37}
!37 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!38 = !DILocalVariable(name: "__nv_MAIN__F1L28_1Arg0", arg: 1, scope: !34, file: !3, type: !9)
!39 = !DILocation(line: 0, scope: !34)
!40 = !DILocalVariable(name: "__nv_MAIN__F1L28_1Arg1", arg: 2, scope: !34, file: !3, type: !37)
!41 = !DILocalVariable(name: "__nv_MAIN__F1L28_1Arg2", arg: 3, scope: !34, file: !3, type: !37)
!42 = !DILocalVariable(name: "omp_sched_static", scope: !34, file: !3, type: !9)
!43 = !DILocalVariable(name: "omp_proc_bind_false", scope: !34, file: !3, type: !9)
!44 = !DILocalVariable(name: "omp_proc_bind_true", scope: !34, file: !3, type: !9)
!45 = !DILocalVariable(name: "omp_lock_hint_none", scope: !34, file: !3, type: !9)
!46 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !34, file: !3, type: !9)
!47 = !DILocation(line: 29, column: 1, scope: !34)
!48 = !DILocation(line: 30, column: 1, scope: !34)

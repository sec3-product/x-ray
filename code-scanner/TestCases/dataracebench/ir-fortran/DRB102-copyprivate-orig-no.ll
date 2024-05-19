; ModuleID = '/tmp/DRB102-copyprivate-orig-no-545753.ll'
source_filename = "/tmp/DRB102-copyprivate-orig-no-545753.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.STATICS2 = type <{ [84 x i8] }>
%struct_drb102_3_ = type <{ [16 x i8] }>

@.STATICS2 = internal global %struct.STATICS2 <{ [84 x i8] c"\FB\FF\FF\FF\03\00\00\00x =\00\EA\FF\FF\FF\00\00\00\00\03\00\00\00\00\00\00\00\01\00\00\00\F7\FF\FF\FF\00\00\00\00\02\00\00\00\FB\FF\FF\FF\03\00\00\00y =\00\EB\FF\FF\FF\00\00\00\00\03\00\00\00\00\00\00\00\01\00\00\00\FF\FF\FF\FF\00\00\00\00" }>, align 16
@.C306_MAIN_ = internal constant i32 25
@.C307_MAIN_ = internal constant i32 28
@.C284_MAIN_ = internal constant i64 0
@.C323_MAIN_ = internal constant i32 6
@.C319_MAIN_ = internal constant [55 x i8] c"micro-benchmarks-fortran/DRB102-copyprivate-orig-no.f95"
@.C321_MAIN_ = internal constant i32 30
@.C292_MAIN_ = internal constant double 1.000000e+00
@.C285_MAIN_ = internal constant i32 1
@.C283_MAIN_ = internal constant i32 0
@.C292___nv_MAIN__F1L23_1 = internal constant double 1.000000e+00
@.C285___nv_MAIN__F1L23_1 = internal constant i32 1
@_drb102_3_ = common global %struct_drb102_3_ zeroinitializer, align 64, !dbg !0, !dbg !7
@TPp_drb102_3_ = common global i8* null, align 64

; Function Attrs: noinline
define float @drb102_() #0 {
.L.entry:
  ret float undef
}

define void @MAIN_() #1 !dbg !12 {
L.entry:
  %__gtid_MAIN__347 = alloca i32, align 4
  %.T0370_370 = alloca i8*, align 8
  %z__io_325 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !28, metadata !DIExpression()), !dbg !29
  call void @llvm.dbg.value(metadata i32 0, metadata !30, metadata !DIExpression()), !dbg !29
  call void @llvm.dbg.value(metadata i32 1, metadata !31, metadata !DIExpression()), !dbg !29
  call void @llvm.dbg.value(metadata i32 0, metadata !32, metadata !DIExpression()), !dbg !29
  call void @llvm.dbg.value(metadata i32 1, metadata !33, metadata !DIExpression()), !dbg !29
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !34
  store i32 %0, i32* %__gtid_MAIN__347, align 4, !dbg !34
  %1 = load i32, i32* %__gtid_MAIN__347, align 4, !dbg !34
  %2 = bitcast %struct_drb102_3_* @_drb102_3_ to i64*, !dbg !34
  %3 = bitcast i8** @TPp_drb102_3_ to i64*, !dbg !34
  %4 = call i8* @__kmpc_threadprivate_cached(i64* null, i32 %1, i64* %2, i64 16, i64* %3), !dbg !34
  store i8* %4, i8** %.T0370_370, align 8, !dbg !34
  %5 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !35
  %6 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !35
  call void (i8*, ...) %6(i8* %5), !dbg !35
  br label %L.LB2_339

L.LB2_339:                                        ; preds = %L.entry
  br label %L.LB2_345, !dbg !36

L.LB2_345:                                        ; preds = %L.LB2_339
  %7 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L23_1_ to i64*, !dbg !36
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %7, i64* null), !dbg !36
  call void (...) @_mp_bcs_nest(), !dbg !37
  %8 = bitcast i32* @.C321_MAIN_ to i8*, !dbg !37
  %9 = bitcast [55 x i8]* @.C319_MAIN_ to i8*, !dbg !37
  %10 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !37
  call void (i8*, i8*, i64, ...) %10(i8* %8, i8* %9, i64 55), !dbg !37
  %11 = bitcast i32* @.C323_MAIN_ to i8*, !dbg !37
  %12 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !37
  %13 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !37
  %14 = bitcast %struct.STATICS2* @.STATICS2 to i8*, !dbg !37
  %15 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !37
  %16 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %15(i8* %11, i8* null, i8* %12, i8* %13, i8* %14, i8* null, i64 0), !dbg !37
  call void @llvm.dbg.declare(metadata i32* %z__io_325, metadata !38, metadata !DIExpression()), !dbg !29
  store i32 %16, i32* %z__io_325, align 4, !dbg !37
  %17 = load i8*, i8** %.T0370_370, align 8, !dbg !37
  %18 = getelementptr i8, i8* %17, i64 8, !dbg !37
  %19 = bitcast i8* %18 to double*, !dbg !37
  %20 = load double, double* %19, align 8, !dbg !37
  %21 = bitcast i32 (...)* @f90io_sc_d_fmt_write to i32 (double, i32, ...)*, !dbg !37
  %22 = call i32 (double, i32, ...) %21(double %20, i32 28), !dbg !37
  store i32 %22, i32* %z__io_325, align 4, !dbg !37
  %23 = load i8*, i8** %.T0370_370, align 8, !dbg !37
  %24 = bitcast i8* %23 to i32*, !dbg !37
  %25 = load i32, i32* %24, align 4, !dbg !37
  %26 = bitcast i32 (...)* @f90io_sc_i_fmt_write to i32 (i32, i32, ...)*, !dbg !37
  %27 = call i32 (i32, i32, ...) %26(i32 %25, i32 25), !dbg !37
  store i32 %27, i32* %z__io_325, align 4, !dbg !37
  %28 = call i32 (...) @f90io_fmtw_end(), !dbg !37
  store i32 %28, i32* %z__io_325, align 4, !dbg !37
  call void (...) @_mp_ecs_nest(), !dbg !37
  ret void, !dbg !34
}

define internal void @__nv_MAIN__F1L23_1_(i32* %__nv_MAIN__F1L23_1Arg0, i64* %__nv_MAIN__F1L23_1Arg1, i64* %__nv_MAIN__F1L23_1Arg2) #1 !dbg !19 {
L.entry:
  %__gtid___nv_MAIN__F1L23_1__396 = alloca i32, align 4
  %.T0401_401 = alloca i8*, align 8
  %.s0000_391 = alloca i32, align 4
  %.s0001_392 = alloca i32, align 4
  %.s0002_409 = alloca i64, align 8
  %.a0001_408 = alloca [6 x i8*], align 16
  %.s0003_414 = alloca i64, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L23_1Arg0, metadata !39, metadata !DIExpression()), !dbg !40
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L23_1Arg1, metadata !41, metadata !DIExpression()), !dbg !40
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L23_1Arg2, metadata !42, metadata !DIExpression()), !dbg !40
  call void @llvm.dbg.value(metadata i32 1, metadata !43, metadata !DIExpression()), !dbg !40
  call void @llvm.dbg.value(metadata i32 0, metadata !44, metadata !DIExpression()), !dbg !40
  call void @llvm.dbg.value(metadata i32 1, metadata !45, metadata !DIExpression()), !dbg !40
  call void @llvm.dbg.value(metadata i32 0, metadata !46, metadata !DIExpression()), !dbg !40
  call void @llvm.dbg.value(metadata i32 1, metadata !47, metadata !DIExpression()), !dbg !40
  %0 = load i32, i32* %__nv_MAIN__F1L23_1Arg0, align 4, !dbg !48
  store i32 %0, i32* %__gtid___nv_MAIN__F1L23_1__396, align 4, !dbg !48
  %1 = load i32, i32* %__gtid___nv_MAIN__F1L23_1__396, align 4, !dbg !48
  %2 = bitcast %struct_drb102_3_* @_drb102_3_ to i64*, !dbg !48
  %3 = bitcast i8** @TPp_drb102_3_ to i64*, !dbg !48
  %4 = call i8* @__kmpc_threadprivate_cached(i64* null, i32 %1, i64* %2, i64 16, i64* %3), !dbg !48
  store i8* %4, i8** %.T0401_401, align 8, !dbg !48
  br label %L.LB3_390

L.LB3_390:                                        ; preds = %L.entry
  br label %L.LB3_313

L.LB3_313:                                        ; preds = %L.LB3_390
  store i32 -1, i32* %.s0000_391, align 4, !dbg !49
  store i32 0, i32* %.s0001_392, align 4, !dbg !49
  %5 = load i32, i32* %__gtid___nv_MAIN__F1L23_1__396, align 4, !dbg !49
  %6 = call i32 @__kmpc_single(i64* null, i32 %5), !dbg !49
  %7 = icmp eq i32 %6, 0, !dbg !49
  br i1 %7, label %L.LB3_336, label %L.LB3_315, !dbg !49

L.LB3_315:                                        ; preds = %L.LB3_313
  %8 = load i8*, i8** %.T0401_401, align 8, !dbg !50
  %9 = getelementptr i8, i8* %8, i64 8, !dbg !50
  %10 = bitcast i8* %9 to double*, !dbg !50
  store double 1.000000e+00, double* %10, align 8, !dbg !50
  %11 = load i8*, i8** %.T0401_401, align 8, !dbg !51
  %12 = bitcast i8* %11 to i32*, !dbg !51
  store i32 1, i32* %12, align 4, !dbg !51
  %13 = load i32, i32* %__gtid___nv_MAIN__F1L23_1__396, align 4, !dbg !52
  store i32 %13, i32* %.s0000_391, align 4, !dbg !52
  store i32 1, i32* %.s0001_392, align 4, !dbg !52
  %14 = load i32, i32* %__gtid___nv_MAIN__F1L23_1__396, align 4, !dbg !52
  call void @__kmpc_end_single(i64* null, i32 %14), !dbg !52
  br label %L.LB3_336

L.LB3_336:                                        ; preds = %L.LB3_315, %L.LB3_313
  store i64 4, i64* %.s0002_409, align 8, !dbg !52
  %15 = bitcast i64* %.s0002_409 to i8*, !dbg !52
  %16 = bitcast [6 x i8*]* %.a0001_408 to i8**, !dbg !52
  store i8* %15, i8** %16, align 8, !dbg !52
  %17 = load i8*, i8** %.T0401_401, align 8, !dbg !52
  %18 = bitcast [6 x i8*]* %.a0001_408 to i8*, !dbg !52
  %19 = getelementptr i8, i8* %18, i64 8, !dbg !52
  %20 = bitcast i8* %19 to i8**, !dbg !52
  store i8* %17, i8** %20, align 8, !dbg !52
  store i64 8, i64* %.s0003_414, align 8, !dbg !52
  %21 = bitcast i64* %.s0003_414 to i8*, !dbg !52
  %22 = bitcast [6 x i8*]* %.a0001_408 to i8*, !dbg !52
  %23 = getelementptr i8, i8* %22, i64 16, !dbg !52
  %24 = bitcast i8* %23 to i8**, !dbg !52
  store i8* %21, i8** %24, align 8, !dbg !52
  %25 = load i8*, i8** %.T0401_401, align 8, !dbg !52
  %26 = bitcast [6 x i8*]* %.a0001_408 to i8*, !dbg !52
  %27 = getelementptr i8, i8* %26, i64 24, !dbg !52
  %28 = bitcast i8* %27 to i8**, !dbg !52
  store i8* %25, i8** %28, align 8, !dbg !52
  %29 = bitcast [6 x i8*]* %.a0001_408 to i8*, !dbg !52
  %30 = getelementptr i8, i8* %29, i64 32, !dbg !52
  %31 = bitcast i8* %30 to i8**, !dbg !52
  store i8* null, i8** %31, align 8, !dbg !52
  %32 = bitcast [6 x i8*]* %.a0001_408 to i8*, !dbg !52
  %33 = getelementptr i8, i8* %32, i64 40, !dbg !52
  %34 = bitcast i8* %33 to i8**, !dbg !52
  store i8* null, i8** %34, align 8, !dbg !52
  %35 = load i32, i32* %__gtid___nv_MAIN__F1L23_1__396, align 4, !dbg !52
  %36 = bitcast [6 x i8*]* %.a0001_408 to i64*, !dbg !52
  %37 = bitcast i32 (...)* @_mp_copypriv_kmpc to i64*, !dbg !52
  %38 = load i32, i32* %.s0001_392, align 4, !dbg !52
  call void @__kmpc_copyprivate(i64* null, i32 %35, i64 0, i64* %36, i64* %37, i32 %38), !dbg !52
  br label %L.LB3_316

L.LB3_316:                                        ; preds = %L.LB3_336
  br label %L.LB3_317

L.LB3_317:                                        ; preds = %L.LB3_316
  ret void, !dbg !48
}

declare signext i32 @_mp_copypriv_kmpc(...) #1

declare void @__kmpc_copyprivate(i64*, i32, i64, i64*, i64*, i32) #1

declare void @__kmpc_end_single(i64*, i32) #1

declare signext i32 @__kmpc_single(i64*, i32) #1

declare void @_mp_ecs_nest(...) #1

declare signext i32 @f90io_fmtw_end(...) #1

declare signext i32 @f90io_sc_i_fmt_write(...) #1

declare signext i32 @f90io_sc_d_fmt_write(...) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

declare signext i32 @f90io_fmtw_inita(...) #1

declare void @f90io_src_info03a(...) #1

declare void @_mp_bcs_nest(...) #1

declare void @fort_init(...) #1

declare signext i32 @__kmpc_global_thread_num(i64*) #1

declare i8* @__kmpc_threadprivate_cached(i64*, i32, i64*, i64, i64*) #1

declare void @__kmpc_fork_call(i64*, i32, i64*, i64*, ...) #1

declare void @__kmpc_end_serialized_parallel(i64*, i32) #1

declare void @__kmpc_serialized_parallel(i64*, i32) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { noinline }
attributes #1 = { "no-frame-pointer-elim-non-leaf" }
attributes #2 = { nounwind readnone speculatable }

!llvm.module.flags = !{!26, !27}
!llvm.dbg.cu = !{!3}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "y", scope: !2, file: !4, type: !22, isLocal: false, isDefinition: true)
!2 = !DIModule(scope: !3, name: "drb102")
!3 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !4, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !6, imports: !24)
!4 = !DIFile(filename: "micro-benchmarks-fortran/DRB102-copyprivate-orig-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!5 = !{}
!6 = !{!0, !7, !10, !17}
!7 = !DIGlobalVariableExpression(var: !8, expr: !DIExpression(DW_OP_plus_uconst, 8))
!8 = distinct !DIGlobalVariable(name: "x", scope: !2, file: !4, type: !9, isLocal: false, isDefinition: true)
!9 = !DIBasicType(name: "double precision", size: 64, align: 64, encoding: DW_ATE_float)
!10 = !DIGlobalVariableExpression(var: !11, expr: !DIExpression())
!11 = distinct !DIGlobalVariable(name: "TPp_drb102$3", scope: !12, file: !4, type: !15, isLocal: false, isDefinition: true)
!12 = distinct !DISubprogram(name: "drb102_copyprivate_orig_no", scope: !3, file: !4, line: 18, type: !13, scopeLine: 18, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !3)
!13 = !DISubroutineType(cc: DW_CC_program, types: !14)
!14 = !{null}
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !16, size: 64, align: 64)
!16 = !DIBasicType(name: "any", encoding: DW_ATE_signed)
!17 = !DIGlobalVariableExpression(var: !18, expr: !DIExpression())
!18 = distinct !DIGlobalVariable(name: "TPp_drb102$3", scope: !19, file: !4, type: !15, isLocal: false, isDefinition: true)
!19 = distinct !DISubprogram(name: "__nv_MAIN__F1L23_1", scope: !3, file: !4, line: 23, type: !20, scopeLine: 23, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !3)
!20 = !DISubroutineType(types: !21)
!21 = !{null, !22, !23, !23}
!22 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!23 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!24 = !{!25}
!25 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !12, entity: !2, file: !4, line: 18)
!26 = !{i32 2, !"Dwarf Version", i32 4}
!27 = !{i32 2, !"Debug Info Version", i32 3}
!28 = !DILocalVariable(name: "omp_sched_static", scope: !12, file: !4, type: !22)
!29 = !DILocation(line: 0, scope: !12)
!30 = !DILocalVariable(name: "omp_proc_bind_false", scope: !12, file: !4, type: !22)
!31 = !DILocalVariable(name: "omp_proc_bind_true", scope: !12, file: !4, type: !22)
!32 = !DILocalVariable(name: "omp_lock_hint_none", scope: !12, file: !4, type: !22)
!33 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !12, file: !4, type: !22)
!34 = !DILocation(line: 33, column: 1, scope: !12)
!35 = !DILocation(line: 18, column: 1, scope: !12)
!36 = !DILocation(line: 23, column: 1, scope: !12)
!37 = !DILocation(line: 30, column: 1, scope: !12)
!38 = !DILocalVariable(scope: !12, file: !4, type: !22, flags: DIFlagArtificial)
!39 = !DILocalVariable(name: "__nv_MAIN__F1L23_1Arg0", arg: 1, scope: !19, file: !4, type: !22)
!40 = !DILocation(line: 0, scope: !19)
!41 = !DILocalVariable(name: "__nv_MAIN__F1L23_1Arg1", arg: 2, scope: !19, file: !4, type: !23)
!42 = !DILocalVariable(name: "__nv_MAIN__F1L23_1Arg2", arg: 3, scope: !19, file: !4, type: !23)
!43 = !DILocalVariable(name: "omp_sched_static", scope: !19, file: !4, type: !22)
!44 = !DILocalVariable(name: "omp_proc_bind_false", scope: !19, file: !4, type: !22)
!45 = !DILocalVariable(name: "omp_proc_bind_true", scope: !19, file: !4, type: !22)
!46 = !DILocalVariable(name: "omp_lock_hint_none", scope: !19, file: !4, type: !22)
!47 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !19, file: !4, type: !22)
!48 = !DILocation(line: 28, column: 1, scope: !19)
!49 = !DILocation(line: 24, column: 1, scope: !19)
!50 = !DILocation(line: 25, column: 1, scope: !19)
!51 = !DILocation(line: 26, column: 1, scope: !19)
!52 = !DILocation(line: 27, column: 1, scope: !19)

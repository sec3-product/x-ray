; ModuleID = '/tmp/DRB088-dynamic-storage-orig-yes-fcabdb.ll'
source_filename = "/tmp/DRB088-dynamic-storage-orig-yes-fcabdb.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct_drb088_0_ = type <{ [24 x i8] }>

@.C285_drb088_foo_ = internal constant i32 1
@.C306_MAIN_ = internal constant i32 25
@.C321_MAIN_ = internal constant i32 6
@.C318_MAIN_ = internal constant [60 x i8] c"micro-benchmarks-fortran/DRB088-dynamic-storage-orig-yes.f95"
@.C320_MAIN_ = internal constant i32 38
@.C284_MAIN_ = internal constant i64 0
@.C333_MAIN_ = internal constant i64 25
@.C332_MAIN_ = internal constant i64 4
@.C286_MAIN_ = internal constant i64 1
@.C283_MAIN_ = internal constant i32 0
@_drb088_0_ = common global %struct_drb088_0_ zeroinitializer, align 64, !dbg !0

; Function Attrs: noinline
define float @drb088_() #0 {
.L.entry:
  ret float undef
}

define void @drb088_foo_() #1 !dbg !18 {
L.entry:
  br label %L.LB2_306

L.LB2_306:                                        ; preds = %L.entry
  %0 = bitcast %struct_drb088_0_* @_drb088_0_ to i32**, !dbg !20
  %1 = load i32*, i32** %0, align 8, !dbg !20
  %2 = load i32, i32* %1, align 4, !dbg !20
  %3 = add nsw i32 %2, 1, !dbg !20
  %4 = bitcast %struct_drb088_0_* @_drb088_0_ to i32**, !dbg !20
  %5 = load i32*, i32** %4, align 8, !dbg !20
  store i32 %3, i32* %5, align 4, !dbg !20
  ret void, !dbg !21
}

define void @MAIN_() #1 !dbg !9 {
L.entry:
  %__gtid_MAIN__349 = alloca i32, align 4
  %z__io_323 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !22, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.value(metadata i32 0, metadata !25, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.value(metadata i32 1, metadata !26, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.value(metadata i32 0, metadata !27, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.value(metadata i32 1, metadata !28, metadata !DIExpression()), !dbg !24
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !29
  store i32 %0, i32* %__gtid_MAIN__349, align 4, !dbg !29
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !30
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !30
  call void (i8*, ...) %2(i8* %1), !dbg !30
  br label %L.LB3_336

L.LB3_336:                                        ; preds = %L.entry
  %3 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !31
  %4 = bitcast i64* @.C333_MAIN_ to i8*, !dbg !31
  %5 = bitcast i64* @.C332_MAIN_ to i8*, !dbg !31
  %6 = bitcast %struct_drb088_0_* @_drb088_0_ to i8*, !dbg !31
  %7 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !31
  %8 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !31
  %9 = bitcast void (...)* @f90_ptr_alloc04a_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !31
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %9(i8* %3, i8* %4, i8* %5, i8* null, i8* %6, i8* null, i8* %7, i8* %8, i8* null, i64 0), !dbg !31
  %10 = bitcast %struct_drb088_0_* @_drb088_0_ to i32**, !dbg !32
  %11 = load i32*, i32** %10, align 8, !dbg !32
  store i32 0, i32* %11, align 4, !dbg !32
  br label %L.LB3_347, !dbg !33

L.LB3_347:                                        ; preds = %L.LB3_336
  %12 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L34_1_ to i64*, !dbg !33
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %12, i64* null), !dbg !33
  call void (...) @_mp_bcs_nest(), !dbg !34
  %13 = bitcast i32* @.C320_MAIN_ to i8*, !dbg !34
  %14 = bitcast [60 x i8]* @.C318_MAIN_ to i8*, !dbg !34
  %15 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !34
  call void (i8*, i8*, i64, ...) %15(i8* %13, i8* %14, i64 60), !dbg !34
  %16 = bitcast i32* @.C321_MAIN_ to i8*, !dbg !34
  %17 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !34
  %18 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !34
  %19 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !34
  %20 = call i32 (i8*, i8*, i8*, i8*, ...) %19(i8* %16, i8* null, i8* %17, i8* %18), !dbg !34
  call void @llvm.dbg.declare(metadata i32* %z__io_323, metadata !35, metadata !DIExpression()), !dbg !24
  store i32 %20, i32* %z__io_323, align 4, !dbg !34
  %21 = bitcast %struct_drb088_0_* @_drb088_0_ to i32**, !dbg !34
  %22 = load i32*, i32** %21, align 8, !dbg !34
  %23 = load i32, i32* %22, align 4, !dbg !34
  %24 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !34
  %25 = call i32 (i32, i32, ...) %24(i32 %23, i32 25), !dbg !34
  store i32 %25, i32* %z__io_323, align 4, !dbg !34
  %26 = call i32 (...) @f90io_ldw_end(), !dbg !34
  store i32 %26, i32* %z__io_323, align 4, !dbg !34
  call void (...) @_mp_ecs_nest(), !dbg !34
  %27 = bitcast %struct_drb088_0_* @_drb088_0_ to i8**, !dbg !36
  %28 = load i8*, i8** %27, align 8, !dbg !36
  %29 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !36
  %30 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !36
  call void (i8*, i8*, i8*, i8*, i64, ...) %30(i8* null, i8* %28, i8* %29, i8* null, i64 0), !dbg !36
  %31 = bitcast %struct_drb088_0_* @_drb088_0_ to i8**, !dbg !36
  store i8* null, i8** %31, align 8, !dbg !36
  %32 = bitcast %struct_drb088_0_* @_drb088_0_ to i8*, !dbg !36
  %33 = getelementptr i8, i8* %32, i64 16, !dbg !36
  %34 = bitcast i8* %33 to i64*, !dbg !36
  store i64 0, i64* %34, align 8, !dbg !36
  ret void, !dbg !29
}

define internal void @__nv_MAIN__F1L34_1_(i32* %__nv_MAIN__F1L34_1Arg0, i64* %__nv_MAIN__F1L34_1Arg1, i64* %__nv_MAIN__F1L34_1Arg2) #1 !dbg !37 {
L.entry:
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L34_1Arg0, metadata !40, metadata !DIExpression()), !dbg !41
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L34_1Arg1, metadata !42, metadata !DIExpression()), !dbg !41
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L34_1Arg2, metadata !43, metadata !DIExpression()), !dbg !41
  call void @llvm.dbg.value(metadata i32 1, metadata !44, metadata !DIExpression()), !dbg !41
  call void @llvm.dbg.value(metadata i32 0, metadata !45, metadata !DIExpression()), !dbg !41
  call void @llvm.dbg.value(metadata i32 1, metadata !46, metadata !DIExpression()), !dbg !41
  call void @llvm.dbg.value(metadata i32 0, metadata !47, metadata !DIExpression()), !dbg !41
  call void @llvm.dbg.value(metadata i32 1, metadata !48, metadata !DIExpression()), !dbg !41
  br label %L.LB4_380

L.LB4_380:                                        ; preds = %L.entry
  br label %L.LB4_315

L.LB4_315:                                        ; preds = %L.LB4_380
  call void @drb088_foo_(), !dbg !49
  br label %L.LB4_316

L.LB4_316:                                        ; preds = %L.LB4_315
  ret void, !dbg !50
}

declare void @f90_dealloc03a_i8(...) #1

declare void @_mp_ecs_nest(...) #1

declare signext i32 @f90io_ldw_end(...) #1

declare signext i32 @f90io_sc_i_ldw(...) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

declare signext i32 @f90io_print_init(...) #1

declare void @f90io_src_info03a(...) #1

declare void @_mp_bcs_nest(...) #1

declare void @f90_ptr_alloc04a_i8(...) #1

declare void @fort_init(...) #1

declare signext i32 @__kmpc_global_thread_num(i64*) #1

declare void @__kmpc_fork_call(i64*, i32, i64*, i64*, ...) #1

declare void @__kmpc_end_serialized_parallel(i64*, i32) #1

declare void @__kmpc_serialized_parallel(i64*, i32) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { noinline }
attributes #1 = { "no-frame-pointer-elim-non-leaf" }
attributes #2 = { nounwind readnone speculatable }

!llvm.module.flags = !{!16, !17}
!llvm.dbg.cu = !{!3}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression(DW_OP_plus_uconst, 16))
!1 = distinct !DIGlobalVariable(name: "counter$sd", scope: !2, file: !4, type: !12, isLocal: false, isDefinition: true)
!2 = !DIModule(scope: !3, name: "drb088")
!3 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !4, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !6, imports: !7)
!4 = !DIFile(filename: "micro-benchmarks-fortran/DRB088-dynamic-storage-orig-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!5 = !{}
!6 = !{!0}
!7 = !{!8}
!8 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !9, entity: !2, file: !4, line: 25)
!9 = distinct !DISubprogram(name: "drb088_dynamic_storage_orig_yes", scope: !3, file: !4, line: 25, type: !10, scopeLine: 25, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !3)
!10 = !DISubroutineType(cc: DW_CC_program, types: !11)
!11 = !{null}
!12 = !DICompositeType(tag: DW_TAG_array_type, baseType: !13, size: 64, align: 64, elements: !14)
!13 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!14 = !{!15}
!15 = !DISubrange(count: 0, lowerBound: 1)
!16 = !{i32 2, !"Dwarf Version", i32 4}
!17 = !{i32 2, !"Debug Info Version", i32 3}
!18 = distinct !DISubprogram(name: "foo", scope: !2, file: !4, line: 21, type: !19, scopeLine: 21, spFlags: DISPFlagDefinition, unit: !3)
!19 = !DISubroutineType(types: !11)
!20 = !DILocation(line: 22, column: 1, scope: !18)
!21 = !DILocation(line: 23, column: 1, scope: !18)
!22 = !DILocalVariable(name: "omp_sched_static", scope: !9, file: !4, type: !23)
!23 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!24 = !DILocation(line: 0, scope: !9)
!25 = !DILocalVariable(name: "omp_proc_bind_false", scope: !9, file: !4, type: !23)
!26 = !DILocalVariable(name: "omp_proc_bind_true", scope: !9, file: !4, type: !23)
!27 = !DILocalVariable(name: "omp_lock_hint_none", scope: !9, file: !4, type: !23)
!28 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !9, file: !4, type: !23)
!29 = !DILocation(line: 42, column: 1, scope: !9)
!30 = !DILocation(line: 25, column: 1, scope: !9)
!31 = !DILocation(line: 30, column: 1, scope: !9)
!32 = !DILocation(line: 32, column: 1, scope: !9)
!33 = !DILocation(line: 34, column: 1, scope: !9)
!34 = !DILocation(line: 38, column: 1, scope: !9)
!35 = !DILocalVariable(scope: !9, file: !4, type: !23, flags: DIFlagArtificial)
!36 = !DILocation(line: 40, column: 1, scope: !9)
!37 = distinct !DISubprogram(name: "__nv_MAIN__F1L34_1", scope: !3, file: !4, line: 34, type: !38, scopeLine: 34, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !3)
!38 = !DISubroutineType(types: !39)
!39 = !{null, !23, !13, !13}
!40 = !DILocalVariable(name: "__nv_MAIN__F1L34_1Arg0", arg: 1, scope: !37, file: !4, type: !23)
!41 = !DILocation(line: 0, scope: !37)
!42 = !DILocalVariable(name: "__nv_MAIN__F1L34_1Arg1", arg: 2, scope: !37, file: !4, type: !13)
!43 = !DILocalVariable(name: "__nv_MAIN__F1L34_1Arg2", arg: 3, scope: !37, file: !4, type: !13)
!44 = !DILocalVariable(name: "omp_sched_static", scope: !37, file: !4, type: !23)
!45 = !DILocalVariable(name: "omp_proc_bind_false", scope: !37, file: !4, type: !23)
!46 = !DILocalVariable(name: "omp_proc_bind_true", scope: !37, file: !4, type: !23)
!47 = !DILocalVariable(name: "omp_lock_hint_none", scope: !37, file: !4, type: !23)
!48 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !37, file: !4, type: !23)
!49 = !DILocation(line: 35, column: 1, scope: !37)
!50 = !DILocation(line: 36, column: 1, scope: !37)

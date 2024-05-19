; ModuleID = '/tmp/DRB086-static-data-member-orig-yes-972bc9.ll'
source_filename = "/tmp/DRB086-static-data-member-orig-yes-972bc9.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%structdrb086_a_td_ = type <{ [8 x i64], [6 x i8*], [11 x i8] }>
%struct.BSS3 = type <{ [8 x i8] }>
%struct_drb086_0_ = type <{ [4 x i8] }>
%struct_drb086_3_ = type <{ [4 x i8] }>

@drb086_a_td_ = global %structdrb086_a_td_ <{ [8 x i64] [i64 43, i64 33, i64 0, i64 8, i64 0, i64 0, i64 0, i64 0], [6 x i8*] [i8* null, i8* bitcast (%structdrb086_a_td_* @drb086_a_td_ to i8*), i8* null, i8* null, i8* null, i8* null], [11 x i8] c"drb086$a$td" }>
@.C285_drb086_foo_ = internal constant i32 1
@.BSS3 = internal global %struct.BSS3 zeroinitializer, align 32, !dbg !0
@.C306_MAIN_ = internal constant i32 25
@.C284_MAIN_ = internal constant i64 0
@.C325_MAIN_ = internal constant i32 6
@.C322_MAIN_ = internal constant [63 x i8] c"micro-benchmarks-fortran/DRB086-static-data-member-orig-yes.f95"
@.C324_MAIN_ = internal constant i32 44
@.C283_MAIN_ = internal constant i32 0
@_drb086_0_ = common global %struct_drb086_0_ zeroinitializer, align 64, !dbg !11
@_drb086_3_ = common global %struct_drb086_3_ zeroinitializer, align 64, !dbg !7
@TPp_drb086_3_ = common global i8* null, align 64

; Function Attrs: noinline
define float @drb086_() #0 {
.L.entry:
  ret float undef
}

define void @drb086_foo_() #1 !dbg !24 {
L.entry:
  %__gtid_drb086_foo__318 = alloca i32, align 4
  %.T0314_314 = alloca i8*, align 8
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !65
  store i32 %0, i32* %__gtid_drb086_foo__318, align 4, !dbg !65
  %1 = load i32, i32* %__gtid_drb086_foo__318, align 4, !dbg !65
  %2 = bitcast %struct_drb086_3_* @_drb086_3_ to i64*, !dbg !65
  %3 = bitcast i8** @TPp_drb086_3_ to i64*, !dbg !65
  %4 = call i8* @__kmpc_threadprivate_cached(i64* null, i32 %1, i64* %2, i64 4, i64* %3), !dbg !65
  store i8* %4, i8** %.T0314_314, align 8, !dbg !65
  br label %L.LB2_310

L.LB2_310:                                        ; preds = %L.entry
  %5 = bitcast %struct_drb086_0_* @_drb086_0_ to i32*, !dbg !66
  %6 = load i32, i32* %5, align 4, !dbg !66
  %7 = add nsw i32 %6, 1, !dbg !66
  %8 = bitcast %struct_drb086_0_* @_drb086_0_ to i32*, !dbg !66
  store i32 %7, i32* %8, align 4, !dbg !66
  %9 = load i8*, i8** %.T0314_314, align 8, !dbg !67
  %10 = bitcast i8* %9 to i32*, !dbg !67
  %11 = load i32, i32* %10, align 4, !dbg !67
  %12 = add nsw i32 %11, 1, !dbg !67
  %13 = load i8*, i8** %.T0314_314, align 8, !dbg !67
  %14 = bitcast i8* %13 to i32*, !dbg !67
  store i32 %12, i32* %14, align 4, !dbg !67
  ret void, !dbg !65
}

define void @MAIN_() #1 !dbg !2 {
L.entry:
  %__gtid_MAIN__348 = alloca i32, align 4
  %.T0371_371 = alloca i8*, align 8
  %z__io_327 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !68, metadata !DIExpression()), !dbg !69
  call void @llvm.dbg.value(metadata i32 0, metadata !70, metadata !DIExpression()), !dbg !69
  call void @llvm.dbg.value(metadata i32 1, metadata !71, metadata !DIExpression()), !dbg !69
  call void @llvm.dbg.value(metadata i32 0, metadata !72, metadata !DIExpression()), !dbg !69
  call void @llvm.dbg.value(metadata i32 1, metadata !73, metadata !DIExpression()), !dbg !69
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !74
  store i32 %0, i32* %__gtid_MAIN__348, align 4, !dbg !74
  %1 = load i32, i32* %__gtid_MAIN__348, align 4, !dbg !74
  %2 = bitcast %struct_drb086_3_* @_drb086_3_ to i64*, !dbg !74
  %3 = bitcast i8** @TPp_drb086_3_ to i64*, !dbg !74
  %4 = call i8* @__kmpc_threadprivate_cached(i64* null, i32 %1, i64* %2, i64 4, i64* %3), !dbg !74
  store i8* %4, i8** %.T0371_371, align 8, !dbg !74
  %5 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !75
  %6 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !75
  call void (i8*, ...) %6(i8* %5), !dbg !75
  br label %L.LB3_337

L.LB3_337:                                        ; preds = %L.entry
  %7 = bitcast %struct.BSS3* @.BSS3 to i32*, !dbg !76
  store i32 0, i32* %7, align 4, !dbg !76
  %8 = bitcast %struct.BSS3* @.BSS3 to i8*, !dbg !76
  %9 = getelementptr i8, i8* %8, i64 4, !dbg !76
  %10 = bitcast i8* %9 to i32*, !dbg !76
  store i32 0, i32* %10, align 4, !dbg !76
  br label %L.LB3_346, !dbg !77

L.LB3_346:                                        ; preds = %L.LB3_337
  %11 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L40_1_ to i64*, !dbg !77
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %11, i64* null), !dbg !77
  call void (...) @_mp_bcs_nest(), !dbg !78
  %12 = bitcast i32* @.C324_MAIN_ to i8*, !dbg !78
  %13 = bitcast [63 x i8]* @.C322_MAIN_ to i8*, !dbg !78
  %14 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !78
  call void (i8*, i8*, i64, ...) %14(i8* %12, i8* %13, i64 63), !dbg !78
  %15 = bitcast i32* @.C325_MAIN_ to i8*, !dbg !78
  %16 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !78
  %17 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !78
  %18 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !78
  %19 = call i32 (i8*, i8*, i8*, i8*, ...) %18(i8* %15, i8* null, i8* %16, i8* %17), !dbg !78
  call void @llvm.dbg.declare(metadata i32* %z__io_327, metadata !79, metadata !DIExpression()), !dbg !69
  store i32 %19, i32* %z__io_327, align 4, !dbg !78
  %20 = bitcast %struct_drb086_0_* @_drb086_0_ to i32*, !dbg !78
  %21 = load i32, i32* %20, align 4, !dbg !78
  %22 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !78
  %23 = call i32 (i32, i32, ...) %22(i32 %21, i32 25), !dbg !78
  store i32 %23, i32* %z__io_327, align 4, !dbg !78
  %24 = load i8*, i8** %.T0371_371, align 8, !dbg !78
  %25 = bitcast i8* %24 to i32*, !dbg !78
  %26 = load i32, i32* %25, align 4, !dbg !78
  %27 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !78
  %28 = call i32 (i32, i32, ...) %27(i32 %26, i32 25), !dbg !78
  store i32 %28, i32* %z__io_327, align 4, !dbg !78
  %29 = call i32 (...) @f90io_ldw_end(), !dbg !78
  store i32 %29, i32* %z__io_327, align 4, !dbg !78
  call void (...) @_mp_ecs_nest(), !dbg !78
  ret void, !dbg !74
}

define internal void @__nv_MAIN__F1L40_1_(i32* %__nv_MAIN__F1L40_1Arg0, i64* %__nv_MAIN__F1L40_1Arg1, i64* %__nv_MAIN__F1L40_1Arg2) #1 !dbg !47 {
L.entry:
  %__gtid___nv_MAIN__F1L40_1__392 = alloca i32, align 4
  %.T0391_391 = alloca i8*, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L40_1Arg0, metadata !80, metadata !DIExpression()), !dbg !81
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L40_1Arg1, metadata !82, metadata !DIExpression()), !dbg !81
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L40_1Arg2, metadata !83, metadata !DIExpression()), !dbg !81
  call void @llvm.dbg.value(metadata i32 1, metadata !84, metadata !DIExpression()), !dbg !81
  call void @llvm.dbg.value(metadata i32 0, metadata !85, metadata !DIExpression()), !dbg !81
  call void @llvm.dbg.value(metadata i32 1, metadata !86, metadata !DIExpression()), !dbg !81
  call void @llvm.dbg.value(metadata i32 0, metadata !87, metadata !DIExpression()), !dbg !81
  call void @llvm.dbg.value(metadata i32 1, metadata !88, metadata !DIExpression()), !dbg !81
  %0 = load i32, i32* %__nv_MAIN__F1L40_1Arg0, align 4, !dbg !89
  store i32 %0, i32* %__gtid___nv_MAIN__F1L40_1__392, align 4, !dbg !89
  %1 = load i32, i32* %__gtid___nv_MAIN__F1L40_1__392, align 4, !dbg !89
  %2 = bitcast %struct_drb086_3_* @_drb086_3_ to i64*, !dbg !89
  %3 = bitcast i8** @TPp_drb086_3_ to i64*, !dbg !89
  %4 = call i8* @__kmpc_threadprivate_cached(i64* null, i32 %1, i64* %2, i64 4, i64* %3), !dbg !89
  store i8* %4, i8** %.T0391_391, align 8, !dbg !89
  br label %L.LB4_390

L.LB4_390:                                        ; preds = %L.entry
  br label %L.LB4_319

L.LB4_319:                                        ; preds = %L.LB4_390
  call void @drb086_foo_(), !dbg !90
  br label %L.LB4_320

L.LB4_320:                                        ; preds = %L.LB4_319
  ret void, !dbg !89
}

declare void @_mp_ecs_nest(...) #1

declare signext i32 @f90io_ldw_end(...) #1

declare signext i32 @f90io_sc_i_ldw(...) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

declare signext i32 @f90io_print_init(...) #1

declare void @f90io_src_info03a(...) #1

declare void @_mp_bcs_nest(...) #1

declare void @fort_init(...) #1

declare void @__kmpc_fork_call(i64*, i32, i64*, i64*, ...) #1

declare void @__kmpc_end_serialized_parallel(i64*, i32) #1

declare void @__kmpc_serialized_parallel(i64*, i32) #1

declare signext i32 @__kmpc_global_thread_num(i64*) #1

declare i8* @__kmpc_threadprivate_cached(i64*, i32, i64*, i64, i64*) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { noinline }
attributes #1 = { "no-frame-pointer-elim-non-leaf" }
attributes #2 = { nounwind readnone speculatable }

!llvm.module.flags = !{!63, !64}
!llvm.dbg.cu = !{!4}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "c", scope: !2, file: !3, type: !41, isLocal: true, isDefinition: true)
!2 = distinct !DISubprogram(name: "drb086_static_data_member_orig_yes", scope: !4, file: !3, line: 32, type: !62, scopeLine: 32, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !4)
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB086-static-data-member-orig-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !6, imports: !60)
!5 = !{}
!6 = !{!7, !11, !13, !22, !29, !36, !0, !38, !45, !51, !58}
!7 = !DIGlobalVariableExpression(var: !8, expr: !DIExpression())
!8 = distinct !DIGlobalVariable(name: "pcounter", scope: !9, file: !3, type: !10, isLocal: false, isDefinition: true)
!9 = !DIModule(scope: !4, name: "drb086")
!10 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!11 = !DIGlobalVariableExpression(var: !12, expr: !DIExpression())
!12 = distinct !DIGlobalVariable(name: "counter", scope: !9, file: !3, type: !10, isLocal: false, isDefinition: true)
!13 = !DIGlobalVariableExpression(var: !14, expr: !DIExpression())
!14 = distinct !DIGlobalVariable(name: "drb086$a$td", scope: !9, file: !3, type: !15, isLocal: false, isDefinition: true)
!15 = !DICompositeType(tag: DW_TAG_array_type, baseType: !16, size: 64, align: 32, elements: !20)
!16 = !DICompositeType(tag: DW_TAG_structure_type, name: "a", file: !3, size: 64, align: 32, elements: !17)
!17 = !{!18, !19}
!18 = !DIDerivedType(tag: DW_TAG_member, name: "counter", scope: !16, file: !3, baseType: !10, size: 32, align: 32)
!19 = !DIDerivedType(tag: DW_TAG_member, name: "pcounter", scope: !16, file: !3, baseType: !10, size: 32, align: 32, offset: 32)
!20 = !{!21}
!21 = !DISubrange(count: 0, lowerBound: 1)
!22 = !DIGlobalVariableExpression(var: !23, expr: !DIExpression())
!23 = distinct !DIGlobalVariable(name: "TPp_drb086$3", scope: !24, file: !3, type: !27, isLocal: false, isDefinition: true)
!24 = distinct !DISubprogram(name: "foo", scope: !9, file: !3, line: 26, type: !25, scopeLine: 26, spFlags: DISPFlagDefinition, unit: !4)
!25 = !DISubroutineType(types: !26)
!26 = !{null}
!27 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !28, size: 64, align: 64)
!28 = !DIBasicType(name: "any", encoding: DW_ATE_signed)
!29 = !DIGlobalVariableExpression(var: !30, expr: !DIExpression())
!30 = distinct !DIGlobalVariable(name: "drb086$a$td", scope: !4, file: !3, type: !31, isLocal: false, isDefinition: true)
!31 = !DICompositeType(tag: DW_TAG_array_type, baseType: !32, size: 64, align: 32, elements: !20)
!32 = !DICompositeType(tag: DW_TAG_structure_type, name: "a", file: !3, size: 64, align: 32, elements: !33)
!33 = !{!34, !35}
!34 = !DIDerivedType(tag: DW_TAG_member, name: "counter", scope: !32, file: !3, baseType: !10, size: 32, align: 32)
!35 = !DIDerivedType(tag: DW_TAG_member, name: "pcounter", scope: !32, file: !3, baseType: !10, size: 32, align: 32, offset: 32)
!36 = !DIGlobalVariableExpression(var: !37, expr: !DIExpression())
!37 = distinct !DIGlobalVariable(name: "TPp_drb086$3", scope: !2, file: !3, type: !27, isLocal: false, isDefinition: true)
!38 = !DIGlobalVariableExpression(var: !39, expr: !DIExpression())
!39 = distinct !DIGlobalVariable(name: "drb086$a$td", scope: !4, file: !3, type: !40, isLocal: false, isDefinition: true)
!40 = !DICompositeType(tag: DW_TAG_array_type, baseType: !41, size: 64, align: 32, elements: !20)
!41 = !DICompositeType(tag: DW_TAG_structure_type, name: "a", file: !3, size: 64, align: 32, elements: !42)
!42 = !{!43, !44}
!43 = !DIDerivedType(tag: DW_TAG_member, name: "counter", scope: !41, file: !3, baseType: !10, size: 32, align: 32)
!44 = !DIDerivedType(tag: DW_TAG_member, name: "pcounter", scope: !41, file: !3, baseType: !10, size: 32, align: 32, offset: 32)
!45 = !DIGlobalVariableExpression(var: !46, expr: !DIExpression())
!46 = distinct !DIGlobalVariable(name: "TPp_drb086$3", scope: !47, file: !3, type: !27, isLocal: false, isDefinition: true)
!47 = distinct !DISubprogram(name: "__nv_MAIN__F1L40_1", scope: !4, file: !3, line: 40, type: !48, scopeLine: 40, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !4)
!48 = !DISubroutineType(types: !49)
!49 = !{null, !10, !50, !50}
!50 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!51 = !DIGlobalVariableExpression(var: !52, expr: !DIExpression())
!52 = distinct !DIGlobalVariable(name: "drb086$a$td", scope: !4, file: !3, type: !53, isLocal: false, isDefinition: true)
!53 = !DICompositeType(tag: DW_TAG_array_type, baseType: !54, size: 64, align: 32, elements: !20)
!54 = !DICompositeType(tag: DW_TAG_structure_type, name: "a", file: !3, size: 64, align: 32, elements: !55)
!55 = !{!56, !57}
!56 = !DIDerivedType(tag: DW_TAG_member, name: "counter", scope: !54, file: !3, baseType: !10, size: 32, align: 32)
!57 = !DIDerivedType(tag: DW_TAG_member, name: "pcounter", scope: !54, file: !3, baseType: !10, size: 32, align: 32, offset: 32)
!58 = !DIGlobalVariableExpression(var: !59, expr: !DIExpression())
!59 = distinct !DIGlobalVariable(name: "c", scope: !4, file: !3, type: !54, isLocal: true, isDefinition: true)
!60 = !{!61}
!61 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !2, entity: !9, file: !3, line: 32)
!62 = !DISubroutineType(cc: DW_CC_program, types: !26)
!63 = !{i32 2, !"Dwarf Version", i32 4}
!64 = !{i32 2, !"Debug Info Version", i32 3}
!65 = !DILocation(line: 29, column: 1, scope: !24)
!66 = !DILocation(line: 27, column: 1, scope: !24)
!67 = !DILocation(line: 28, column: 1, scope: !24)
!68 = !DILocalVariable(name: "omp_sched_static", scope: !2, file: !3, type: !10)
!69 = !DILocation(line: 0, scope: !2)
!70 = !DILocalVariable(name: "omp_proc_bind_false", scope: !2, file: !3, type: !10)
!71 = !DILocalVariable(name: "omp_proc_bind_true", scope: !2, file: !3, type: !10)
!72 = !DILocalVariable(name: "omp_lock_hint_none", scope: !2, file: !3, type: !10)
!73 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !2, file: !3, type: !10)
!74 = !DILocation(line: 45, column: 1, scope: !2)
!75 = !DILocation(line: 32, column: 1, scope: !2)
!76 = !DILocation(line: 38, column: 1, scope: !2)
!77 = !DILocation(line: 40, column: 1, scope: !2)
!78 = !DILocation(line: 44, column: 1, scope: !2)
!79 = !DILocalVariable(scope: !2, file: !3, type: !10, flags: DIFlagArtificial)
!80 = !DILocalVariable(name: "__nv_MAIN__F1L40_1Arg0", arg: 1, scope: !47, file: !3, type: !10)
!81 = !DILocation(line: 0, scope: !47)
!82 = !DILocalVariable(name: "__nv_MAIN__F1L40_1Arg1", arg: 2, scope: !47, file: !3, type: !50)
!83 = !DILocalVariable(name: "__nv_MAIN__F1L40_1Arg2", arg: 3, scope: !47, file: !3, type: !50)
!84 = !DILocalVariable(name: "omp_sched_static", scope: !47, file: !3, type: !10)
!85 = !DILocalVariable(name: "omp_proc_bind_false", scope: !47, file: !3, type: !10)
!86 = !DILocalVariable(name: "omp_proc_bind_true", scope: !47, file: !3, type: !10)
!87 = !DILocalVariable(name: "omp_lock_hint_none", scope: !47, file: !3, type: !10)
!88 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !47, file: !3, type: !10)
!89 = !DILocation(line: 42, column: 1, scope: !47)
!90 = !DILocation(line: 41, column: 1, scope: !47)

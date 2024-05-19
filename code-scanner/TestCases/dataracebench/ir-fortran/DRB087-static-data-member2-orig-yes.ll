; ModuleID = '/tmp/DRB087-static-data-member2-orig-yes-4b468f.ll'
source_filename = "/tmp/DRB087-static-data-member2-orig-yes-4b468f.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct_drb087_10_ = type <{ [8 x i8] }>
%struct.STATICS1 = type <{ [8 x i8] }>
%structdrb087_a_td_ = type <{ [8 x i64], [6 x i8*], [11 x i8] }>
%struct.STATICS2 = type <{ [8 x i8] }>
%struct_drb087_0_ = type <{ [4 x i8] }>
%struct_drb087_3_ = type <{ [4 x i8] }>

@_drb087_10_ = global %struct_drb087_10_ zeroinitializer, align 64, !dbg !0
@.STATICS1 = internal global %struct.STATICS1 zeroinitializer, align 16
@drb087_a_td_ = global %structdrb087_a_td_ <{ [8 x i64] [i64 43, i64 33, i64 0, i64 8, i64 0, i64 0, i64 0, i64 0], [6 x i8*] [i8* getelementptr inbounds (%struct_drb087_10_, %struct_drb087_10_* @_drb087_10_, i32 0, i32 0, i32 0), i8* bitcast (%structdrb087_a_td_* @drb087_a_td_ to i8*), i8* null, i8* null, i8* null, i8* null], [11 x i8] c"drb087$a$td" }>
@.STATICS2 = internal global %struct.STATICS2 zeroinitializer, align 16, !dbg !28
@.C306_MAIN_ = internal constant i32 25
@.C284_MAIN_ = internal constant i64 0
@.C326_MAIN_ = internal constant i32 6
@.C323_MAIN_ = internal constant [64 x i8] c"micro-benchmarks-fortran/DRB087-static-data-member2-orig-yes.f95"
@.C325_MAIN_ = internal constant i32 41
@.C285_MAIN_ = internal constant i32 1
@.C283_MAIN_ = internal constant i32 0
@.C285___nv_MAIN__F1L36_1 = internal constant i32 1
@_drb087_0_ = common global %struct_drb087_0_ zeroinitializer, align 64, !dbg !10
@_drb087_3_ = common global %struct_drb087_3_ zeroinitializer, align 64, !dbg !7
@TPp_drb087_3_ = common global i8* null, align 64

; Function Attrs: noinline
define float @drb087_() #0 {
.L.entry:
  ret float undef
}

define void @MAIN_() #1 !dbg !23 {
L.entry:
  %__gtid_MAIN__349 = alloca i32, align 4
  %.T0372_372 = alloca i8*, align 8
  %z__io_328 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !56, metadata !DIExpression()), !dbg !57
  call void @llvm.dbg.value(metadata i32 0, metadata !58, metadata !DIExpression()), !dbg !57
  call void @llvm.dbg.value(metadata i32 1, metadata !59, metadata !DIExpression()), !dbg !57
  call void @llvm.dbg.value(metadata i32 0, metadata !60, metadata !DIExpression()), !dbg !57
  call void @llvm.dbg.value(metadata i32 1, metadata !61, metadata !DIExpression()), !dbg !57
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !62
  store i32 %0, i32* %__gtid_MAIN__349, align 4, !dbg !62
  %1 = load i32, i32* %__gtid_MAIN__349, align 4, !dbg !62
  %2 = bitcast %struct_drb087_3_* @_drb087_3_ to i64*, !dbg !62
  %3 = bitcast i8** @TPp_drb087_3_ to i64*, !dbg !62
  %4 = call i8* @__kmpc_threadprivate_cached(i64* null, i32 %1, i64* %2, i64 4, i64* %3), !dbg !62
  store i8* %4, i8** %.T0372_372, align 8, !dbg !62
  %5 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !63
  %6 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !63
  call void (i8*, ...) %6(i8* %5), !dbg !63
  br label %L.LB2_338

L.LB2_338:                                        ; preds = %L.entry
  %7 = bitcast %struct.STATICS2* @.STATICS2 to i32*, !dbg !64
  store i32 0, i32* %7, align 4, !dbg !64
  %8 = bitcast %struct.STATICS2* @.STATICS2 to i8*, !dbg !64
  %9 = getelementptr i8, i8* %8, i64 4, !dbg !64
  %10 = bitcast i8* %9 to i32*, !dbg !64
  store i32 0, i32* %10, align 4, !dbg !64
  br label %L.LB2_347, !dbg !65

L.LB2_347:                                        ; preds = %L.LB2_338
  %11 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L36_1_ to i64*, !dbg !65
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %11, i64* null), !dbg !65
  call void (...) @_mp_bcs_nest(), !dbg !66
  %12 = bitcast i32* @.C325_MAIN_ to i8*, !dbg !66
  %13 = bitcast [64 x i8]* @.C323_MAIN_ to i8*, !dbg !66
  %14 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !66
  call void (i8*, i8*, i64, ...) %14(i8* %12, i8* %13, i64 64), !dbg !66
  %15 = bitcast i32* @.C326_MAIN_ to i8*, !dbg !66
  %16 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !66
  %17 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !66
  %18 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !66
  %19 = call i32 (i8*, i8*, i8*, i8*, ...) %18(i8* %15, i8* null, i8* %16, i8* %17), !dbg !66
  call void @llvm.dbg.declare(metadata i32* %z__io_328, metadata !67, metadata !DIExpression()), !dbg !57
  store i32 %19, i32* %z__io_328, align 4, !dbg !66
  %20 = bitcast %struct_drb087_0_* @_drb087_0_ to i32*, !dbg !66
  %21 = load i32, i32* %20, align 4, !dbg !66
  %22 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !66
  %23 = call i32 (i32, i32, ...) %22(i32 %21, i32 25), !dbg !66
  store i32 %23, i32* %z__io_328, align 4, !dbg !66
  %24 = load i8*, i8** %.T0372_372, align 8, !dbg !66
  %25 = bitcast i8* %24 to i32*, !dbg !66
  %26 = load i32, i32* %25, align 4, !dbg !66
  %27 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !66
  %28 = call i32 (i32, i32, ...) %27(i32 %26, i32 25), !dbg !66
  store i32 %28, i32* %z__io_328, align 4, !dbg !66
  %29 = call i32 (...) @f90io_ldw_end(), !dbg !66
  store i32 %29, i32* %z__io_328, align 4, !dbg !66
  call void (...) @_mp_ecs_nest(), !dbg !66
  ret void, !dbg !62
}

define internal void @__nv_MAIN__F1L36_1_(i32* %__nv_MAIN__F1L36_1Arg0, i64* %__nv_MAIN__F1L36_1Arg1, i64* %__nv_MAIN__F1L36_1Arg2) #1 !dbg !39 {
L.entry:
  %__gtid___nv_MAIN__F1L36_1__395 = alloca i32, align 4
  %.T0393_393 = alloca i8*, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L36_1Arg0, metadata !68, metadata !DIExpression()), !dbg !69
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L36_1Arg1, metadata !70, metadata !DIExpression()), !dbg !69
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L36_1Arg2, metadata !71, metadata !DIExpression()), !dbg !69
  call void @llvm.dbg.value(metadata i32 1, metadata !72, metadata !DIExpression()), !dbg !69
  call void @llvm.dbg.value(metadata i32 0, metadata !73, metadata !DIExpression()), !dbg !69
  call void @llvm.dbg.value(metadata i32 1, metadata !74, metadata !DIExpression()), !dbg !69
  call void @llvm.dbg.value(metadata i32 0, metadata !75, metadata !DIExpression()), !dbg !69
  call void @llvm.dbg.value(metadata i32 1, metadata !76, metadata !DIExpression()), !dbg !69
  %0 = load i32, i32* %__nv_MAIN__F1L36_1Arg0, align 4, !dbg !77
  store i32 %0, i32* %__gtid___nv_MAIN__F1L36_1__395, align 4, !dbg !77
  %1 = load i32, i32* %__gtid___nv_MAIN__F1L36_1__395, align 4, !dbg !77
  %2 = bitcast %struct_drb087_3_* @_drb087_3_ to i64*, !dbg !77
  %3 = bitcast i8** @TPp_drb087_3_ to i64*, !dbg !77
  %4 = call i8* @__kmpc_threadprivate_cached(i64* null, i32 %1, i64* %2, i64 4, i64* %3), !dbg !77
  store i8* %4, i8** %.T0393_393, align 8, !dbg !77
  br label %L.LB3_392

L.LB3_392:                                        ; preds = %L.entry
  br label %L.LB3_320

L.LB3_320:                                        ; preds = %L.LB3_392
  %5 = bitcast %struct_drb087_0_* @_drb087_0_ to i32*, !dbg !78
  %6 = load i32, i32* %5, align 4, !dbg !78
  %7 = add nsw i32 %6, 1, !dbg !78
  %8 = bitcast %struct_drb087_0_* @_drb087_0_ to i32*, !dbg !78
  store i32 %7, i32* %8, align 4, !dbg !78
  %9 = load i8*, i8** %.T0393_393, align 8, !dbg !79
  %10 = bitcast i8* %9 to i32*, !dbg !79
  %11 = load i32, i32* %10, align 4, !dbg !79
  %12 = add nsw i32 %11, 1, !dbg !79
  %13 = load i8*, i8** %.T0393_393, align 8, !dbg !79
  %14 = bitcast i8* %13 to i32*, !dbg !79
  store i32 %12, i32* %14, align 4, !dbg !79
  br label %L.LB3_321

L.LB3_321:                                        ; preds = %L.LB3_320
  ret void, !dbg !77
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

!llvm.module.flags = !{!54, !55}
!llvm.dbg.cu = !{!3}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "._dtInit0058", scope: !2, file: !4, type: !15, isLocal: false, isDefinition: true)
!2 = !DIModule(scope: !3, name: "drb087")
!3 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !4, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !6, imports: !52)
!4 = !DIFile(filename: "micro-benchmarks-fortran/DRB087-static-data-member2-orig-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!5 = !{}
!6 = !{!0, !7, !10, !12, !21, !28, !34, !37, !43, !50}
!7 = !DIGlobalVariableExpression(var: !8, expr: !DIExpression())
!8 = distinct !DIGlobalVariable(name: "pcounter", scope: !2, file: !4, type: !9, isLocal: false, isDefinition: true)
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DIGlobalVariableExpression(var: !11, expr: !DIExpression())
!11 = distinct !DIGlobalVariable(name: "counter", scope: !2, file: !4, type: !9, isLocal: false, isDefinition: true)
!12 = !DIGlobalVariableExpression(var: !13, expr: !DIExpression())
!13 = distinct !DIGlobalVariable(name: "drb087$a$td", scope: !2, file: !4, type: !14, isLocal: false, isDefinition: true)
!14 = !DICompositeType(tag: DW_TAG_array_type, baseType: !15, size: 64, align: 32, elements: !19)
!15 = !DICompositeType(tag: DW_TAG_structure_type, name: "a", file: !4, size: 64, align: 32, elements: !16)
!16 = !{!17, !18}
!17 = !DIDerivedType(tag: DW_TAG_member, name: "counter", scope: !15, file: !4, baseType: !9, size: 32, align: 32)
!18 = !DIDerivedType(tag: DW_TAG_member, name: "pcounter", scope: !15, file: !4, baseType: !9, size: 32, align: 32, offset: 32)
!19 = !{!20}
!20 = !DISubrange(count: 0, lowerBound: 1)
!21 = !DIGlobalVariableExpression(var: !22, expr: !DIExpression())
!22 = distinct !DIGlobalVariable(name: "TPp_drb087$3", scope: !23, file: !4, type: !26, isLocal: false, isDefinition: true)
!23 = distinct !DISubprogram(name: "drb087_static_data_member2_orig_yes", scope: !3, file: !4, line: 28, type: !24, scopeLine: 28, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !3)
!24 = !DISubroutineType(cc: DW_CC_program, types: !25)
!25 = !{null}
!26 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !27, size: 64, align: 64)
!27 = !DIBasicType(name: "any", encoding: DW_ATE_signed)
!28 = !DIGlobalVariableExpression(var: !29, expr: !DIExpression())
!29 = distinct !DIGlobalVariable(name: "c", scope: !23, file: !4, type: !30, isLocal: true, isDefinition: true)
!30 = !DICompositeType(tag: DW_TAG_structure_type, name: "a", file: !4, size: 64, align: 32, elements: !31)
!31 = !{!32, !33}
!32 = !DIDerivedType(tag: DW_TAG_member, name: "counter", scope: !30, file: !4, baseType: !9, size: 32, align: 32)
!33 = !DIDerivedType(tag: DW_TAG_member, name: "pcounter", scope: !30, file: !4, baseType: !9, size: 32, align: 32, offset: 32)
!34 = !DIGlobalVariableExpression(var: !35, expr: !DIExpression())
!35 = distinct !DIGlobalVariable(name: "drb087$a$td", scope: !3, file: !4, type: !36, isLocal: false, isDefinition: true)
!36 = !DICompositeType(tag: DW_TAG_array_type, baseType: !30, size: 64, align: 32, elements: !19)
!37 = !DIGlobalVariableExpression(var: !38, expr: !DIExpression())
!38 = distinct !DIGlobalVariable(name: "TPp_drb087$3", scope: !39, file: !4, type: !26, isLocal: false, isDefinition: true)
!39 = distinct !DISubprogram(name: "__nv_MAIN__F1L36_1", scope: !3, file: !4, line: 36, type: !40, scopeLine: 36, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !3)
!40 = !DISubroutineType(types: !41)
!41 = !{null, !9, !42, !42}
!42 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!43 = !DIGlobalVariableExpression(var: !44, expr: !DIExpression())
!44 = distinct !DIGlobalVariable(name: "drb087$a$td", scope: !3, file: !4, type: !45, isLocal: false, isDefinition: true)
!45 = !DICompositeType(tag: DW_TAG_array_type, baseType: !46, size: 64, align: 32, elements: !19)
!46 = !DICompositeType(tag: DW_TAG_structure_type, name: "a", file: !4, size: 64, align: 32, elements: !47)
!47 = !{!48, !49}
!48 = !DIDerivedType(tag: DW_TAG_member, name: "counter", scope: !46, file: !4, baseType: !9, size: 32, align: 32)
!49 = !DIDerivedType(tag: DW_TAG_member, name: "pcounter", scope: !46, file: !4, baseType: !9, size: 32, align: 32, offset: 32)
!50 = !DIGlobalVariableExpression(var: !51, expr: !DIExpression())
!51 = distinct !DIGlobalVariable(name: "c", scope: !3, file: !4, type: !46, isLocal: true, isDefinition: true)
!52 = !{!53}
!53 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !23, entity: !2, file: !4, line: 28)
!54 = !{i32 2, !"Dwarf Version", i32 4}
!55 = !{i32 2, !"Debug Info Version", i32 3}
!56 = !DILocalVariable(name: "omp_sched_static", scope: !23, file: !4, type: !9)
!57 = !DILocation(line: 0, scope: !23)
!58 = !DILocalVariable(name: "omp_proc_bind_false", scope: !23, file: !4, type: !9)
!59 = !DILocalVariable(name: "omp_proc_bind_true", scope: !23, file: !4, type: !9)
!60 = !DILocalVariable(name: "omp_lock_hint_none", scope: !23, file: !4, type: !9)
!61 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !23, file: !4, type: !9)
!62 = !DILocation(line: 42, column: 1, scope: !23)
!63 = !DILocation(line: 28, column: 1, scope: !23)
!64 = !DILocation(line: 34, column: 1, scope: !23)
!65 = !DILocation(line: 36, column: 1, scope: !23)
!66 = !DILocation(line: 41, column: 1, scope: !23)
!67 = !DILocalVariable(scope: !23, file: !4, type: !9, flags: DIFlagArtificial)
!68 = !DILocalVariable(name: "__nv_MAIN__F1L36_1Arg0", arg: 1, scope: !39, file: !4, type: !9)
!69 = !DILocation(line: 0, scope: !39)
!70 = !DILocalVariable(name: "__nv_MAIN__F1L36_1Arg1", arg: 2, scope: !39, file: !4, type: !42)
!71 = !DILocalVariable(name: "__nv_MAIN__F1L36_1Arg2", arg: 3, scope: !39, file: !4, type: !42)
!72 = !DILocalVariable(name: "omp_sched_static", scope: !39, file: !4, type: !9)
!73 = !DILocalVariable(name: "omp_proc_bind_false", scope: !39, file: !4, type: !9)
!74 = !DILocalVariable(name: "omp_proc_bind_true", scope: !39, file: !4, type: !9)
!75 = !DILocalVariable(name: "omp_lock_hint_none", scope: !39, file: !4, type: !9)
!76 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !39, file: !4, type: !9)
!77 = !DILocation(line: 39, column: 1, scope: !39)
!78 = !DILocation(line: 37, column: 1, scope: !39)
!79 = !DILocation(line: 38, column: 1, scope: !39)

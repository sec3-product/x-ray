; ModuleID = '/tmp/DRB127-tasking-threadprivate1-orig-no-7e4ff5.ll'
source_filename = "/tmp/DRB127-tasking-threadprivate1-orig-no-7e4ff5.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct_drb127_3_ = type <{ [4 x i8] }>
%struct_drb127_0_ = type <{ [4 x i8] }>

@.C314_drb127_foo_ = internal constant i32 2
@.C283_drb127_foo_ = internal constant i32 0
@.C285_drb127_foo_ = internal constant i32 1
@.C314___nv_drb127_foo__F1L19_1 = internal constant i32 2
@.C283___nv_drb127_foo__F1L19_1 = internal constant i32 0
@.C285___nv_drb127_foo__F1L19_1 = internal constant i32 1
@.C283___nv_drb127_F1L20_2 = internal constant i32 0
@.C285___nv_drb127_F1L20_2 = internal constant i32 1
@.C283_MAIN_ = internal constant i32 0
@_drb127_3_ = common global %struct_drb127_3_ zeroinitializer, align 64, !dbg !0
@_drb127_0_ = common global %struct_drb127_0_ zeroinitializer, align 64, !dbg !7
@TPp_drb127_3_ = common global i8* null, align 64

; Function Attrs: noinline
define float @drb127_() #0 {
.L.entry:
  ret float undef
}

define void @drb127_foo_() #1 !dbg !12 {
L.entry:
  %__gtid_drb127_foo__329 = alloca i32, align 4
  %.T0349_349 = alloca i8*, align 8
  %.s0000_324 = alloca i32, align 4
  %.z0302_323 = alloca i8*, align 8
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !37
  store i32 %0, i32* %__gtid_drb127_foo__329, align 4, !dbg !37
  %1 = load i32, i32* %__gtid_drb127_foo__329, align 4, !dbg !37
  %2 = bitcast %struct_drb127_3_* @_drb127_3_ to i64*, !dbg !37
  %3 = bitcast i8** @TPp_drb127_3_ to i64*, !dbg !37
  %4 = call i8* @__kmpc_threadprivate_cached(i64* null, i32 %1, i64* %2, i64 4, i64* %3), !dbg !37
  store i8* %4, i8** %.T0349_349, align 8, !dbg !37
  br label %L.LB2_322

L.LB2_322:                                        ; preds = %L.entry
  store i32 1, i32* %.s0000_324, align 4, !dbg !38
  %5 = load i32, i32* %__gtid_drb127_foo__329, align 4, !dbg !39
  %6 = load i32, i32* %.s0000_324, align 4, !dbg !39
  %7 = bitcast void (i32, i64*)* @__nv_drb127_foo__F1L19_1_ to i64*, !dbg !39
  %8 = call i8* @__kmpc_omp_task_alloc(i64* null, i32 %5, i32 %6, i32 40, i32 0, i64* %7), !dbg !39
  store i8* %8, i8** %.z0302_323, align 8, !dbg !39
  %9 = load i32, i32* %__gtid_drb127_foo__329, align 4, !dbg !39
  %10 = load i8*, i8** %.z0302_323, align 8, !dbg !39
  %11 = bitcast i8* %10 to i64*, !dbg !39
  call void @__kmpc_omp_task(i64* null, i32 %9, i64* %11), !dbg !39
  br label %L.LB2_318

L.LB2_318:                                        ; preds = %L.LB2_322
  ret void, !dbg !37
}

define internal void @__nv_drb127_foo__F1L19_1_(i32 %__nv_drb127_foo__F1L19_1Arg0.arg, i64* %__nv_drb127_foo__F1L19_1Arg1) #1 !dbg !19 {
L.entry:
  %__nv_drb127_foo__F1L19_1Arg0.addr = alloca i32, align 4
  %.S0000_365 = alloca i8*, align 8
  %__gtid___nv_drb127_foo__F1L19_1__376 = alloca i32, align 4
  %.T0380_380 = alloca i8*, align 8
  %.s0001_371 = alloca i32, align 4
  %.z0326_370 = alloca i8*, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_drb127_foo__F1L19_1Arg0.addr, metadata !40, metadata !DIExpression()), !dbg !41
  store i32 %__nv_drb127_foo__F1L19_1Arg0.arg, i32* %__nv_drb127_foo__F1L19_1Arg0.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_drb127_foo__F1L19_1Arg0.addr, metadata !42, metadata !DIExpression()), !dbg !41
  call void @llvm.dbg.declare(metadata i64* %__nv_drb127_foo__F1L19_1Arg1, metadata !43, metadata !DIExpression()), !dbg !41
  %0 = bitcast i64* %__nv_drb127_foo__F1L19_1Arg1 to i8**, !dbg !44
  %1 = load i8*, i8** %0, align 8, !dbg !44
  store i8* %1, i8** %.S0000_365, align 8, !dbg !44
  %2 = load i32, i32* %__nv_drb127_foo__F1L19_1Arg0.addr, align 4, !dbg !45
  call void @llvm.dbg.value(metadata i32 %2, metadata !40, metadata !DIExpression()), !dbg !41
  store i32 %2, i32* %__gtid___nv_drb127_foo__F1L19_1__376, align 4, !dbg !45
  %3 = load i32, i32* %__gtid___nv_drb127_foo__F1L19_1__376, align 4, !dbg !45
  %4 = bitcast %struct_drb127_3_* @_drb127_3_ to i64*, !dbg !45
  %5 = bitcast i8** @TPp_drb127_3_ to i64*, !dbg !45
  %6 = call i8* @__kmpc_threadprivate_cached(i64* null, i32 %3, i64* %4, i64 4, i64* %5), !dbg !45
  store i8* %6, i8** %.T0380_380, align 8, !dbg !45
  br label %L.LB3_369

L.LB3_369:                                        ; preds = %L.entry
  br label %L.LB3_305

L.LB3_305:                                        ; preds = %L.LB3_369
  store i32 1, i32* %.s0001_371, align 4, !dbg !46
  %7 = load i32, i32* %__gtid___nv_drb127_foo__F1L19_1__376, align 4, !dbg !47
  %8 = load i32, i32* %.s0001_371, align 4, !dbg !47
  %9 = bitcast void (i32, i64*)* @__nv_drb127_F1L20_2_ to i64*, !dbg !47
  %10 = call i8* @__kmpc_omp_task_alloc(i64* null, i32 %7, i32 %8, i32 40, i32 0, i64* %9), !dbg !47
  store i8* %10, i8** %.z0326_370, align 8, !dbg !47
  %11 = load i32, i32* %__gtid___nv_drb127_foo__F1L19_1__376, align 4, !dbg !47
  %12 = load i8*, i8** %.z0326_370, align 8, !dbg !47
  %13 = bitcast i8* %12 to i64*, !dbg !47
  call void @__kmpc_omp_task(i64* null, i32 %11, i64* %13), !dbg !47
  br label %L.LB3_319

L.LB3_319:                                        ; preds = %L.LB3_305
  %14 = load i8*, i8** %.T0380_380, align 8, !dbg !48
  %15 = bitcast i8* %14 to i32*, !dbg !48
  store i32 2, i32* %15, align 4, !dbg !48
  br label %L.LB3_315

L.LB3_315:                                        ; preds = %L.LB3_319
  ret void, !dbg !45
}

define internal void @__nv_drb127_F1L20_2_(i32 %__nv_drb127_F1L20_2Arg0.arg, i64* %__nv_drb127_F1L20_2Arg1) #1 !dbg !25 {
L.entry:
  %__nv_drb127_F1L20_2Arg0.addr = alloca i32, align 4
  %.S0000_365 = alloca i8*, align 8
  %__gtid___nv_drb127_F1L20_2__397 = alloca i32, align 4
  %.T0389_389 = alloca i8*, align 8
  %.s0002_392 = alloca i32, align 4
  %.z0373_391 = alloca i8*, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_drb127_F1L20_2Arg0.addr, metadata !49, metadata !DIExpression()), !dbg !50
  store i32 %__nv_drb127_F1L20_2Arg0.arg, i32* %__nv_drb127_F1L20_2Arg0.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_drb127_F1L20_2Arg0.addr, metadata !51, metadata !DIExpression()), !dbg !50
  call void @llvm.dbg.declare(metadata i64* %__nv_drb127_F1L20_2Arg1, metadata !52, metadata !DIExpression()), !dbg !50
  %0 = bitcast i64* %__nv_drb127_F1L20_2Arg1 to i8**, !dbg !53
  %1 = load i8*, i8** %0, align 8, !dbg !53
  store i8* %1, i8** %.S0000_365, align 8, !dbg !53
  %2 = load i32, i32* %__nv_drb127_F1L20_2Arg0.addr, align 4, !dbg !54
  call void @llvm.dbg.value(metadata i32 %2, metadata !49, metadata !DIExpression()), !dbg !50
  store i32 %2, i32* %__gtid___nv_drb127_F1L20_2__397, align 4, !dbg !54
  %3 = load i32, i32* %__gtid___nv_drb127_F1L20_2__397, align 4, !dbg !54
  %4 = bitcast %struct_drb127_3_* @_drb127_3_ to i64*, !dbg !54
  %5 = bitcast i8** @TPp_drb127_3_ to i64*, !dbg !54
  %6 = call i8* @__kmpc_threadprivate_cached(i64* null, i32 %3, i64* %4, i64 4, i64* %5), !dbg !54
  store i8* %6, i8** %.T0389_389, align 8, !dbg !54
  br label %L.LB5_388

L.LB5_388:                                        ; preds = %L.entry
  br label %L.LB5_308

L.LB5_308:                                        ; preds = %L.LB5_388
  %7 = load i8*, i8** %.T0389_389, align 8, !dbg !55
  %8 = bitcast i8* %7 to i32*, !dbg !55
  store i32 1, i32* %8, align 4, !dbg !55
  store i32 1, i32* %.s0002_392, align 4, !dbg !56
  %9 = load i32, i32* %__gtid___nv_drb127_F1L20_2__397, align 4, !dbg !57
  %10 = load i32, i32* %.s0002_392, align 4, !dbg !57
  %11 = bitcast void (i32, i64*)* @__nv_drb127_F1L22_3_ to i64*, !dbg !57
  %12 = call i8* @__kmpc_omp_task_alloc(i64* null, i32 %9, i32 %10, i32 40, i32 0, i64* %11), !dbg !57
  store i8* %12, i8** %.z0373_391, align 8, !dbg !57
  %13 = load i32, i32* %__gtid___nv_drb127_F1L20_2__397, align 4, !dbg !57
  %14 = load i8*, i8** %.z0373_391, align 8, !dbg !57
  %15 = bitcast i8* %14 to i64*, !dbg !57
  call void @__kmpc_omp_task(i64* null, i32 %13, i64* %15), !dbg !57
  br label %L.LB5_320

L.LB5_320:                                        ; preds = %L.LB5_308
  %16 = load i8*, i8** %.T0389_389, align 8, !dbg !58
  %17 = bitcast i8* %16 to i32*, !dbg !58
  %18 = load i32, i32* %17, align 4, !dbg !58
  %19 = bitcast %struct_drb127_0_* @_drb127_0_ to i32*, !dbg !58
  store i32 %18, i32* %19, align 4, !dbg !58
  br label %L.LB5_313

L.LB5_313:                                        ; preds = %L.LB5_320
  ret void, !dbg !54
}

define internal void @__nv_drb127_F1L22_3_(i32 %__nv_drb127_F1L22_3Arg0.arg, i64* %__nv_drb127_F1L22_3Arg1) #1 !dbg !28 {
L.entry:
  %__nv_drb127_F1L22_3Arg0.addr = alloca i32, align 4
  %.S0000_365 = alloca i8*, align 8
  %__gtid___nv_drb127_F1L22_3__410 = alloca i32, align 4
  %.T0409_409 = alloca i8*, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_drb127_F1L22_3Arg0.addr, metadata !59, metadata !DIExpression()), !dbg !60
  store i32 %__nv_drb127_F1L22_3Arg0.arg, i32* %__nv_drb127_F1L22_3Arg0.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_drb127_F1L22_3Arg0.addr, metadata !61, metadata !DIExpression()), !dbg !60
  call void @llvm.dbg.declare(metadata i64* %__nv_drb127_F1L22_3Arg1, metadata !62, metadata !DIExpression()), !dbg !60
  %0 = bitcast i64* %__nv_drb127_F1L22_3Arg1 to i8**, !dbg !63
  %1 = load i8*, i8** %0, align 8, !dbg !63
  store i8* %1, i8** %.S0000_365, align 8, !dbg !63
  %2 = load i32, i32* %__nv_drb127_F1L22_3Arg0.addr, align 4, !dbg !64
  call void @llvm.dbg.value(metadata i32 %2, metadata !59, metadata !DIExpression()), !dbg !60
  store i32 %2, i32* %__gtid___nv_drb127_F1L22_3__410, align 4, !dbg !64
  %3 = load i32, i32* %__gtid___nv_drb127_F1L22_3__410, align 4, !dbg !64
  %4 = bitcast %struct_drb127_3_* @_drb127_3_ to i64*, !dbg !64
  %5 = bitcast i8** @TPp_drb127_3_ to i64*, !dbg !64
  %6 = call i8* @__kmpc_threadprivate_cached(i64* null, i32 %3, i64* %4, i64 4, i64* %5), !dbg !64
  store i8* %6, i8** %.T0409_409, align 8, !dbg !64
  br label %L.LB7_408

L.LB7_408:                                        ; preds = %L.entry
  br label %L.LB7_311

L.LB7_311:                                        ; preds = %L.LB7_408
  br label %L.LB7_312

L.LB7_312:                                        ; preds = %L.LB7_311
  ret void, !dbg !64
}

define void @MAIN_() #1 !dbg !31 {
L.entry:
  %__gtid_MAIN__321 = alloca i32, align 4
  %.T0320_320 = alloca i8*, align 8
  call void @llvm.dbg.value(metadata i32 1, metadata !65, metadata !DIExpression()), !dbg !66
  call void @llvm.dbg.value(metadata i32 0, metadata !67, metadata !DIExpression()), !dbg !66
  call void @llvm.dbg.value(metadata i32 1, metadata !68, metadata !DIExpression()), !dbg !66
  call void @llvm.dbg.value(metadata i32 0, metadata !69, metadata !DIExpression()), !dbg !66
  call void @llvm.dbg.value(metadata i32 1, metadata !70, metadata !DIExpression()), !dbg !66
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !71
  store i32 %0, i32* %__gtid_MAIN__321, align 4, !dbg !71
  %1 = load i32, i32* %__gtid_MAIN__321, align 4, !dbg !71
  %2 = bitcast %struct_drb127_3_* @_drb127_3_ to i64*, !dbg !71
  %3 = bitcast i8** @TPp_drb127_3_ to i64*, !dbg !71
  %4 = call i8* @__kmpc_threadprivate_cached(i64* null, i32 %1, i64* %2, i64 4, i64* %3), !dbg !71
  store i8* %4, i8** %.T0320_320, align 8, !dbg !71
  %5 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !72
  %6 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !72
  call void (i8*, ...) %6(i8* %5), !dbg !72
  br label %L.LB9_316

L.LB9_316:                                        ; preds = %L.entry
  call void @drb127_foo_(), !dbg !73
  ret void, !dbg !71
}

declare void @fort_init(...) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

declare signext i32 @__kmpc_global_thread_num(i64*) #1

declare i8* @__kmpc_threadprivate_cached(i64*, i32, i64*, i64, i64*) #1

declare void @__kmpc_omp_task(i64*, i32, i64*) #1

declare i8* @__kmpc_omp_task_alloc(i64*, i32, i32, i32, i32, i64*) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { noinline }
attributes #1 = { "no-frame-pointer-elim-non-leaf" }
attributes #2 = { nounwind readnone speculatable }

!llvm.module.flags = !{!35, !36}
!llvm.dbg.cu = !{!3}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "tp", scope: !2, file: !4, type: !9, isLocal: false, isDefinition: true)
!2 = !DIModule(scope: !3, name: "drb127")
!3 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !4, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !6, imports: !33)
!4 = !DIFile(filename: "micro-benchmarks-fortran/DRB127-tasking-threadprivate1-orig-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!5 = !{}
!6 = !{!7, !0, !10, !17, !23, !26, !29}
!7 = !DIGlobalVariableExpression(var: !8, expr: !DIExpression())
!8 = distinct !DIGlobalVariable(name: "var", scope: !2, file: !4, type: !9, isLocal: false, isDefinition: true)
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DIGlobalVariableExpression(var: !11, expr: !DIExpression())
!11 = distinct !DIGlobalVariable(name: "TPp_drb127$3", scope: !12, file: !4, type: !15, isLocal: false, isDefinition: true)
!12 = distinct !DISubprogram(name: "foo", scope: !2, file: !4, line: 18, type: !13, scopeLine: 18, spFlags: DISPFlagDefinition, unit: !3)
!13 = !DISubroutineType(types: !14)
!14 = !{null}
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !16, size: 64, align: 64)
!16 = !DIBasicType(name: "any", encoding: DW_ATE_signed)
!17 = !DIGlobalVariableExpression(var: !18, expr: !DIExpression())
!18 = distinct !DIGlobalVariable(name: "TPp_drb127$3", scope: !19, file: !4, type: !15, isLocal: false, isDefinition: true)
!19 = distinct !DISubprogram(name: "__nv_drb127_foo__F1L19_1", scope: !3, file: !4, line: 19, type: !20, scopeLine: 19, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !3)
!20 = !DISubroutineType(types: !21)
!21 = !{null, !9, !22}
!22 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!23 = !DIGlobalVariableExpression(var: !24, expr: !DIExpression())
!24 = distinct !DIGlobalVariable(name: "TPp_drb127$3", scope: !25, file: !4, type: !15, isLocal: false, isDefinition: true)
!25 = distinct !DISubprogram(name: "__nv_drb127_F1L20_2", scope: !3, file: !4, line: 20, type: !20, scopeLine: 20, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !3)
!26 = !DIGlobalVariableExpression(var: !27, expr: !DIExpression())
!27 = distinct !DIGlobalVariable(name: "TPp_drb127$3", scope: !28, file: !4, type: !15, isLocal: false, isDefinition: true)
!28 = distinct !DISubprogram(name: "__nv_drb127_F1L22_3", scope: !3, file: !4, line: 22, type: !20, scopeLine: 22, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !3)
!29 = !DIGlobalVariableExpression(var: !30, expr: !DIExpression())
!30 = distinct !DIGlobalVariable(name: "TPp_drb127$3", scope: !31, file: !4, type: !15, isLocal: false, isDefinition: true)
!31 = distinct !DISubprogram(name: "drb127_tasking_threadprivate1_orig_no", scope: !3, file: !4, line: 31, type: !32, scopeLine: 31, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !3)
!32 = !DISubroutineType(cc: DW_CC_program, types: !14)
!33 = !{!34}
!34 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !31, entity: !2, file: !4, line: 31)
!35 = !{i32 2, !"Dwarf Version", i32 4}
!36 = !{i32 2, !"Debug Info Version", i32 3}
!37 = !DILocation(line: 28, column: 1, scope: !12)
!38 = !DILocation(line: 19, column: 1, scope: !12)
!39 = !DILocation(line: 27, column: 1, scope: !12)
!40 = !DILocalVariable(name: "__nv_drb127_foo__F1L19_1Arg0", scope: !19, file: !4, type: !9)
!41 = !DILocation(line: 0, scope: !19)
!42 = !DILocalVariable(name: "__nv_drb127_foo__F1L19_1Arg0", arg: 1, scope: !19, file: !4, type: !9)
!43 = !DILocalVariable(name: "__nv_drb127_foo__F1L19_1Arg1", arg: 2, scope: !19, file: !4, type: !22)
!44 = !DILocation(line: 19, column: 1, scope: !19)
!45 = !DILocation(line: 27, column: 1, scope: !19)
!46 = !DILocation(line: 20, column: 1, scope: !19)
!47 = !DILocation(line: 25, column: 1, scope: !19)
!48 = !DILocation(line: 26, column: 1, scope: !19)
!49 = !DILocalVariable(name: "__nv_drb127_F1L20_2Arg0", scope: !25, file: !4, type: !9)
!50 = !DILocation(line: 0, scope: !25)
!51 = !DILocalVariable(name: "__nv_drb127_F1L20_2Arg0", arg: 1, scope: !25, file: !4, type: !9)
!52 = !DILocalVariable(name: "__nv_drb127_F1L20_2Arg1", arg: 2, scope: !25, file: !4, type: !22)
!53 = !DILocation(line: 20, column: 1, scope: !25)
!54 = !DILocation(line: 25, column: 1, scope: !25)
!55 = !DILocation(line: 21, column: 1, scope: !25)
!56 = !DILocation(line: 22, column: 1, scope: !25)
!57 = !DILocation(line: 23, column: 1, scope: !25)
!58 = !DILocation(line: 24, column: 1, scope: !25)
!59 = !DILocalVariable(name: "__nv_drb127_F1L22_3Arg0", scope: !28, file: !4, type: !9)
!60 = !DILocation(line: 0, scope: !28)
!61 = !DILocalVariable(name: "__nv_drb127_F1L22_3Arg0", arg: 1, scope: !28, file: !4, type: !9)
!62 = !DILocalVariable(name: "__nv_drb127_F1L22_3Arg1", arg: 2, scope: !28, file: !4, type: !22)
!63 = !DILocation(line: 22, column: 1, scope: !28)
!64 = !DILocation(line: 23, column: 1, scope: !28)
!65 = !DILocalVariable(name: "omp_sched_static", scope: !31, file: !4, type: !9)
!66 = !DILocation(line: 0, scope: !31)
!67 = !DILocalVariable(name: "omp_proc_bind_false", scope: !31, file: !4, type: !9)
!68 = !DILocalVariable(name: "omp_proc_bind_true", scope: !31, file: !4, type: !9)
!69 = !DILocalVariable(name: "omp_lock_hint_none", scope: !31, file: !4, type: !9)
!70 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !31, file: !4, type: !9)
!71 = !DILocation(line: 38, column: 1, scope: !31)
!72 = !DILocation(line: 31, column: 1, scope: !31)
!73 = !DILocation(line: 36, column: 1, scope: !31)

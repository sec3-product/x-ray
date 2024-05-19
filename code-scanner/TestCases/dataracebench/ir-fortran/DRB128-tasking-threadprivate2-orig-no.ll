; ModuleID = '/tmp/DRB128-tasking-threadprivate2-orig-no-e5b5c4.ll'
source_filename = "/tmp/DRB128-tasking-threadprivate2-orig-no-e5b5c4.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct_drb128_3_ = type <{ [4 x i8] }>
%struct_drb128_0_ = type <{ [4 x i8] }>

@.C283_drb128_foo_ = internal constant i32 0
@.C285_drb128_foo_ = internal constant i32 1
@.C283___nv_drb128_foo__F1L19_1 = internal constant i32 0
@.C285___nv_drb128_foo__F1L19_1 = internal constant i32 1
@.C283___nv_drb128_F1L20_2 = internal constant i32 0
@.C285___nv_drb128_F1L20_2 = internal constant i32 1
@.C306_MAIN_ = internal constant i32 25
@.C284_MAIN_ = internal constant i64 0
@.C316_MAIN_ = internal constant i32 6
@.C313_MAIN_ = internal constant [66 x i8] c"micro-benchmarks-fortran/DRB128-tasking-threadprivate2-orig-no.f95"
@.C315_MAIN_ = internal constant i32 36
@.C283_MAIN_ = internal constant i32 0
@_drb128_3_ = common global %struct_drb128_3_ zeroinitializer, align 64, !dbg !0
@_drb128_0_ = common global %struct_drb128_0_ zeroinitializer, align 64, !dbg !7
@TPp_drb128_3_ = common global i8* null, align 64

; Function Attrs: noinline
define float @drb128_() #0 {
.L.entry:
  ret float undef
}

define void @drb128_foo_() #1 !dbg !12 {
L.entry:
  %__gtid_drb128_foo__328 = alloca i32, align 4
  %.T0348_348 = alloca i8*, align 8
  %.s0000_323 = alloca i32, align 4
  %.z0302_322 = alloca i8*, align 8
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !37
  store i32 %0, i32* %__gtid_drb128_foo__328, align 4, !dbg !37
  %1 = load i32, i32* %__gtid_drb128_foo__328, align 4, !dbg !37
  %2 = bitcast %struct_drb128_3_* @_drb128_3_ to i64*, !dbg !37
  %3 = bitcast i8** @TPp_drb128_3_ to i64*, !dbg !37
  %4 = call i8* @__kmpc_threadprivate_cached(i64* null, i32 %1, i64* %2, i64 4, i64* %3), !dbg !37
  store i8* %4, i8** %.T0348_348, align 8, !dbg !37
  br label %L.LB2_321

L.LB2_321:                                        ; preds = %L.entry
  store i32 1, i32* %.s0000_323, align 4, !dbg !38
  %5 = load i32, i32* %__gtid_drb128_foo__328, align 4, !dbg !39
  %6 = load i32, i32* %.s0000_323, align 4, !dbg !39
  %7 = bitcast void (i32, i64*)* @__nv_drb128_foo__F1L19_1_ to i64*, !dbg !39
  %8 = call i8* @__kmpc_omp_task_alloc(i64* null, i32 %5, i32 %6, i32 40, i32 0, i64* %7), !dbg !39
  store i8* %8, i8** %.z0302_322, align 8, !dbg !39
  %9 = load i32, i32* %__gtid_drb128_foo__328, align 4, !dbg !39
  %10 = load i8*, i8** %.z0302_322, align 8, !dbg !39
  %11 = bitcast i8* %10 to i64*, !dbg !39
  call void @__kmpc_omp_task(i64* null, i32 %9, i64* %11), !dbg !39
  br label %L.LB2_317

L.LB2_317:                                        ; preds = %L.LB2_321
  ret void, !dbg !37
}

define internal void @__nv_drb128_foo__F1L19_1_(i32 %__nv_drb128_foo__F1L19_1Arg0.arg, i64* %__nv_drb128_foo__F1L19_1Arg1) #1 !dbg !19 {
L.entry:
  %__nv_drb128_foo__F1L19_1Arg0.addr = alloca i32, align 4
  %.S0000_364 = alloca i8*, align 8
  %__gtid___nv_drb128_foo__F1L19_1__375 = alloca i32, align 4
  %.T0379_379 = alloca i8*, align 8
  %.s0001_370 = alloca i32, align 4
  %.z0325_369 = alloca i8*, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_drb128_foo__F1L19_1Arg0.addr, metadata !40, metadata !DIExpression()), !dbg !41
  store i32 %__nv_drb128_foo__F1L19_1Arg0.arg, i32* %__nv_drb128_foo__F1L19_1Arg0.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_drb128_foo__F1L19_1Arg0.addr, metadata !42, metadata !DIExpression()), !dbg !41
  call void @llvm.dbg.declare(metadata i64* %__nv_drb128_foo__F1L19_1Arg1, metadata !43, metadata !DIExpression()), !dbg !41
  %0 = bitcast i64* %__nv_drb128_foo__F1L19_1Arg1 to i8**, !dbg !44
  %1 = load i8*, i8** %0, align 8, !dbg !44
  store i8* %1, i8** %.S0000_364, align 8, !dbg !44
  %2 = load i32, i32* %__nv_drb128_foo__F1L19_1Arg0.addr, align 4, !dbg !45
  call void @llvm.dbg.value(metadata i32 %2, metadata !40, metadata !DIExpression()), !dbg !41
  store i32 %2, i32* %__gtid___nv_drb128_foo__F1L19_1__375, align 4, !dbg !45
  %3 = load i32, i32* %__gtid___nv_drb128_foo__F1L19_1__375, align 4, !dbg !45
  %4 = bitcast %struct_drb128_3_* @_drb128_3_ to i64*, !dbg !45
  %5 = bitcast i8** @TPp_drb128_3_ to i64*, !dbg !45
  %6 = call i8* @__kmpc_threadprivate_cached(i64* null, i32 %3, i64* %4, i64 4, i64* %5), !dbg !45
  store i8* %6, i8** %.T0379_379, align 8, !dbg !45
  br label %L.LB3_368

L.LB3_368:                                        ; preds = %L.entry
  br label %L.LB3_305

L.LB3_305:                                        ; preds = %L.LB3_368
  store i32 1, i32* %.s0001_370, align 4, !dbg !46
  %7 = load i32, i32* %__gtid___nv_drb128_foo__F1L19_1__375, align 4, !dbg !47
  %8 = load i32, i32* %.s0001_370, align 4, !dbg !47
  %9 = bitcast void (i32, i64*)* @__nv_drb128_F1L20_2_ to i64*, !dbg !47
  %10 = call i8* @__kmpc_omp_task_alloc(i64* null, i32 %7, i32 %8, i32 40, i32 0, i64* %9), !dbg !47
  store i8* %10, i8** %.z0325_369, align 8, !dbg !47
  %11 = load i32, i32* %__gtid___nv_drb128_foo__F1L19_1__375, align 4, !dbg !47
  %12 = load i8*, i8** %.z0325_369, align 8, !dbg !47
  %13 = bitcast i8* %12 to i64*, !dbg !47
  call void @__kmpc_omp_task(i64* null, i32 %11, i64* %13), !dbg !47
  br label %L.LB3_318

L.LB3_318:                                        ; preds = %L.LB3_305
  br label %L.LB3_314

L.LB3_314:                                        ; preds = %L.LB3_318
  ret void, !dbg !45
}

define internal void @__nv_drb128_F1L20_2_(i32 %__nv_drb128_F1L20_2Arg0.arg, i64* %__nv_drb128_F1L20_2Arg1) #1 !dbg !25 {
L.entry:
  %__nv_drb128_F1L20_2Arg0.addr = alloca i32, align 4
  %.S0000_364 = alloca i8*, align 8
  %__gtid___nv_drb128_F1L20_2__396 = alloca i32, align 4
  %.T0388_388 = alloca i8*, align 8
  %.s0002_391 = alloca i32, align 4
  %.z0372_390 = alloca i8*, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_drb128_F1L20_2Arg0.addr, metadata !48, metadata !DIExpression()), !dbg !49
  store i32 %__nv_drb128_F1L20_2Arg0.arg, i32* %__nv_drb128_F1L20_2Arg0.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_drb128_F1L20_2Arg0.addr, metadata !50, metadata !DIExpression()), !dbg !49
  call void @llvm.dbg.declare(metadata i64* %__nv_drb128_F1L20_2Arg1, metadata !51, metadata !DIExpression()), !dbg !49
  %0 = bitcast i64* %__nv_drb128_F1L20_2Arg1 to i8**, !dbg !52
  %1 = load i8*, i8** %0, align 8, !dbg !52
  store i8* %1, i8** %.S0000_364, align 8, !dbg !52
  %2 = load i32, i32* %__nv_drb128_F1L20_2Arg0.addr, align 4, !dbg !53
  call void @llvm.dbg.value(metadata i32 %2, metadata !48, metadata !DIExpression()), !dbg !49
  store i32 %2, i32* %__gtid___nv_drb128_F1L20_2__396, align 4, !dbg !53
  %3 = load i32, i32* %__gtid___nv_drb128_F1L20_2__396, align 4, !dbg !53
  %4 = bitcast %struct_drb128_3_* @_drb128_3_ to i64*, !dbg !53
  %5 = bitcast i8** @TPp_drb128_3_ to i64*, !dbg !53
  %6 = call i8* @__kmpc_threadprivate_cached(i64* null, i32 %3, i64* %4, i64 4, i64* %5), !dbg !53
  store i8* %6, i8** %.T0388_388, align 8, !dbg !53
  br label %L.LB5_387

L.LB5_387:                                        ; preds = %L.entry
  br label %L.LB5_308

L.LB5_308:                                        ; preds = %L.LB5_387
  %7 = load i8*, i8** %.T0388_388, align 8, !dbg !54
  %8 = bitcast i8* %7 to i32*, !dbg !54
  store i32 1, i32* %8, align 4, !dbg !54
  store i32 1, i32* %.s0002_391, align 4, !dbg !55
  %9 = load i32, i32* %__gtid___nv_drb128_F1L20_2__396, align 4, !dbg !56
  %10 = load i32, i32* %.s0002_391, align 4, !dbg !56
  %11 = bitcast void (i32, i64*)* @__nv_drb128_F1L22_3_ to i64*, !dbg !56
  %12 = call i8* @__kmpc_omp_task_alloc(i64* null, i32 %9, i32 %10, i32 40, i32 0, i64* %11), !dbg !56
  store i8* %12, i8** %.z0372_390, align 8, !dbg !56
  %13 = load i32, i32* %__gtid___nv_drb128_F1L20_2__396, align 4, !dbg !56
  %14 = load i8*, i8** %.z0372_390, align 8, !dbg !56
  %15 = bitcast i8* %14 to i64*, !dbg !56
  call void @__kmpc_omp_task(i64* null, i32 %13, i64* %15), !dbg !56
  br label %L.LB5_319

L.LB5_319:                                        ; preds = %L.LB5_308
  %16 = load i8*, i8** %.T0388_388, align 8, !dbg !57
  %17 = bitcast i8* %16 to i32*, !dbg !57
  %18 = load i32, i32* %17, align 4, !dbg !57
  %19 = bitcast %struct_drb128_0_* @_drb128_0_ to i32*, !dbg !57
  store i32 %18, i32* %19, align 4, !dbg !57
  br label %L.LB5_313

L.LB5_313:                                        ; preds = %L.LB5_319
  ret void, !dbg !53
}

define internal void @__nv_drb128_F1L22_3_(i32 %__nv_drb128_F1L22_3Arg0.arg, i64* %__nv_drb128_F1L22_3Arg1) #1 !dbg !28 {
L.entry:
  %__nv_drb128_F1L22_3Arg0.addr = alloca i32, align 4
  %.S0000_364 = alloca i8*, align 8
  %__gtid___nv_drb128_F1L22_3__409 = alloca i32, align 4
  %.T0408_408 = alloca i8*, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_drb128_F1L22_3Arg0.addr, metadata !58, metadata !DIExpression()), !dbg !59
  store i32 %__nv_drb128_F1L22_3Arg0.arg, i32* %__nv_drb128_F1L22_3Arg0.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_drb128_F1L22_3Arg0.addr, metadata !60, metadata !DIExpression()), !dbg !59
  call void @llvm.dbg.declare(metadata i64* %__nv_drb128_F1L22_3Arg1, metadata !61, metadata !DIExpression()), !dbg !59
  %0 = bitcast i64* %__nv_drb128_F1L22_3Arg1 to i8**, !dbg !62
  %1 = load i8*, i8** %0, align 8, !dbg !62
  store i8* %1, i8** %.S0000_364, align 8, !dbg !62
  %2 = load i32, i32* %__nv_drb128_F1L22_3Arg0.addr, align 4, !dbg !63
  call void @llvm.dbg.value(metadata i32 %2, metadata !58, metadata !DIExpression()), !dbg !59
  store i32 %2, i32* %__gtid___nv_drb128_F1L22_3__409, align 4, !dbg !63
  %3 = load i32, i32* %__gtid___nv_drb128_F1L22_3__409, align 4, !dbg !63
  %4 = bitcast %struct_drb128_3_* @_drb128_3_ to i64*, !dbg !63
  %5 = bitcast i8** @TPp_drb128_3_ to i64*, !dbg !63
  %6 = call i8* @__kmpc_threadprivate_cached(i64* null, i32 %3, i64* %4, i64 4, i64* %5), !dbg !63
  store i8* %6, i8** %.T0408_408, align 8, !dbg !63
  br label %L.LB7_407

L.LB7_407:                                        ; preds = %L.entry
  br label %L.LB7_311

L.LB7_311:                                        ; preds = %L.LB7_407
  br label %L.LB7_312

L.LB7_312:                                        ; preds = %L.LB7_311
  ret void, !dbg !63
}

define void @MAIN_() #1 !dbg !31 {
L.entry:
  %__gtid_MAIN__340 = alloca i32, align 4
  %.T0339_339 = alloca i8*, align 8
  %z__io_318 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !64, metadata !DIExpression()), !dbg !65
  call void @llvm.dbg.value(metadata i32 0, metadata !66, metadata !DIExpression()), !dbg !65
  call void @llvm.dbg.value(metadata i32 1, metadata !67, metadata !DIExpression()), !dbg !65
  call void @llvm.dbg.value(metadata i32 0, metadata !68, metadata !DIExpression()), !dbg !65
  call void @llvm.dbg.value(metadata i32 1, metadata !69, metadata !DIExpression()), !dbg !65
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !70
  store i32 %0, i32* %__gtid_MAIN__340, align 4, !dbg !70
  %1 = load i32, i32* %__gtid_MAIN__340, align 4, !dbg !70
  %2 = bitcast %struct_drb128_3_* @_drb128_3_ to i64*, !dbg !70
  %3 = bitcast i8** @TPp_drb128_3_ to i64*, !dbg !70
  %4 = call i8* @__kmpc_threadprivate_cached(i64* null, i32 %1, i64* %2, i64 4, i64* %3), !dbg !70
  store i8* %4, i8** %.T0339_339, align 8, !dbg !70
  %5 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !71
  %6 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !71
  call void (i8*, ...) %6(i8* %5), !dbg !71
  br label %L.LB9_328

L.LB9_328:                                        ; preds = %L.entry
  call void @drb128_foo_(), !dbg !72
  call void (...) @_mp_bcs_nest(), !dbg !73
  %7 = bitcast i32* @.C315_MAIN_ to i8*, !dbg !73
  %8 = bitcast [66 x i8]* @.C313_MAIN_ to i8*, !dbg !73
  %9 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !73
  call void (i8*, i8*, i64, ...) %9(i8* %7, i8* %8, i64 66), !dbg !73
  %10 = bitcast i32* @.C316_MAIN_ to i8*, !dbg !73
  %11 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !73
  %12 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !73
  %13 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !73
  %14 = call i32 (i8*, i8*, i8*, i8*, ...) %13(i8* %10, i8* null, i8* %11, i8* %12), !dbg !73
  call void @llvm.dbg.declare(metadata i32* %z__io_318, metadata !74, metadata !DIExpression()), !dbg !65
  store i32 %14, i32* %z__io_318, align 4, !dbg !73
  %15 = bitcast %struct_drb128_0_* @_drb128_0_ to i32*, !dbg !73
  %16 = load i32, i32* %15, align 4, !dbg !73
  %17 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !73
  %18 = call i32 (i32, i32, ...) %17(i32 %16, i32 25), !dbg !73
  store i32 %18, i32* %z__io_318, align 4, !dbg !73
  %19 = call i32 (...) @f90io_ldw_end(), !dbg !73
  store i32 %19, i32* %z__io_318, align 4, !dbg !73
  call void (...) @_mp_ecs_nest(), !dbg !73
  ret void, !dbg !70
}

declare void @_mp_ecs_nest(...) #1

declare signext i32 @f90io_ldw_end(...) #1

declare signext i32 @f90io_sc_i_ldw(...) #1

declare signext i32 @f90io_print_init(...) #1

declare void @f90io_src_info03a(...) #1

declare void @_mp_bcs_nest(...) #1

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
!2 = !DIModule(scope: !3, name: "drb128")
!3 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !4, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !6, imports: !33)
!4 = !DIFile(filename: "micro-benchmarks-fortran/DRB128-tasking-threadprivate2-orig-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!5 = !{}
!6 = !{!7, !0, !10, !17, !23, !26, !29}
!7 = !DIGlobalVariableExpression(var: !8, expr: !DIExpression())
!8 = distinct !DIGlobalVariable(name: "var", scope: !2, file: !4, type: !9, isLocal: false, isDefinition: true)
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DIGlobalVariableExpression(var: !11, expr: !DIExpression())
!11 = distinct !DIGlobalVariable(name: "TPp_drb128$3", scope: !12, file: !4, type: !15, isLocal: false, isDefinition: true)
!12 = distinct !DISubprogram(name: "foo", scope: !2, file: !4, line: 18, type: !13, scopeLine: 18, spFlags: DISPFlagDefinition, unit: !3)
!13 = !DISubroutineType(types: !14)
!14 = !{null}
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !16, size: 64, align: 64)
!16 = !DIBasicType(name: "any", encoding: DW_ATE_signed)
!17 = !DIGlobalVariableExpression(var: !18, expr: !DIExpression())
!18 = distinct !DIGlobalVariable(name: "TPp_drb128$3", scope: !19, file: !4, type: !15, isLocal: false, isDefinition: true)
!19 = distinct !DISubprogram(name: "__nv_drb128_foo__F1L19_1", scope: !3, file: !4, line: 19, type: !20, scopeLine: 19, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !3)
!20 = !DISubroutineType(types: !21)
!21 = !{null, !9, !22}
!22 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!23 = !DIGlobalVariableExpression(var: !24, expr: !DIExpression())
!24 = distinct !DIGlobalVariable(name: "TPp_drb128$3", scope: !25, file: !4, type: !15, isLocal: false, isDefinition: true)
!25 = distinct !DISubprogram(name: "__nv_drb128_F1L20_2", scope: !3, file: !4, line: 20, type: !20, scopeLine: 20, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !3)
!26 = !DIGlobalVariableExpression(var: !27, expr: !DIExpression())
!27 = distinct !DIGlobalVariable(name: "TPp_drb128$3", scope: !28, file: !4, type: !15, isLocal: false, isDefinition: true)
!28 = distinct !DISubprogram(name: "__nv_drb128_F1L22_3", scope: !3, file: !4, line: 22, type: !20, scopeLine: 22, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !3)
!29 = !DIGlobalVariableExpression(var: !30, expr: !DIExpression())
!30 = distinct !DIGlobalVariable(name: "TPp_drb128$3", scope: !31, file: !4, type: !15, isLocal: false, isDefinition: true)
!31 = distinct !DISubprogram(name: "drb128_tasking_threadprivate2_orig_no", scope: !3, file: !4, line: 30, type: !32, scopeLine: 30, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !3)
!32 = !DISubroutineType(cc: DW_CC_program, types: !14)
!33 = !{!34}
!34 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !31, entity: !2, file: !4, line: 30)
!35 = !{i32 2, !"Dwarf Version", i32 4}
!36 = !{i32 2, !"Debug Info Version", i32 3}
!37 = !DILocation(line: 27, column: 1, scope: !12)
!38 = !DILocation(line: 19, column: 1, scope: !12)
!39 = !DILocation(line: 26, column: 1, scope: !12)
!40 = !DILocalVariable(name: "__nv_drb128_foo__F1L19_1Arg0", scope: !19, file: !4, type: !9)
!41 = !DILocation(line: 0, scope: !19)
!42 = !DILocalVariable(name: "__nv_drb128_foo__F1L19_1Arg0", arg: 1, scope: !19, file: !4, type: !9)
!43 = !DILocalVariable(name: "__nv_drb128_foo__F1L19_1Arg1", arg: 2, scope: !19, file: !4, type: !22)
!44 = !DILocation(line: 19, column: 1, scope: !19)
!45 = !DILocation(line: 26, column: 1, scope: !19)
!46 = !DILocation(line: 20, column: 1, scope: !19)
!47 = !DILocation(line: 25, column: 1, scope: !19)
!48 = !DILocalVariable(name: "__nv_drb128_F1L20_2Arg0", scope: !25, file: !4, type: !9)
!49 = !DILocation(line: 0, scope: !25)
!50 = !DILocalVariable(name: "__nv_drb128_F1L20_2Arg0", arg: 1, scope: !25, file: !4, type: !9)
!51 = !DILocalVariable(name: "__nv_drb128_F1L20_2Arg1", arg: 2, scope: !25, file: !4, type: !22)
!52 = !DILocation(line: 20, column: 1, scope: !25)
!53 = !DILocation(line: 25, column: 1, scope: !25)
!54 = !DILocation(line: 21, column: 1, scope: !25)
!55 = !DILocation(line: 22, column: 1, scope: !25)
!56 = !DILocation(line: 23, column: 1, scope: !25)
!57 = !DILocation(line: 24, column: 1, scope: !25)
!58 = !DILocalVariable(name: "__nv_drb128_F1L22_3Arg0", scope: !28, file: !4, type: !9)
!59 = !DILocation(line: 0, scope: !28)
!60 = !DILocalVariable(name: "__nv_drb128_F1L22_3Arg0", arg: 1, scope: !28, file: !4, type: !9)
!61 = !DILocalVariable(name: "__nv_drb128_F1L22_3Arg1", arg: 2, scope: !28, file: !4, type: !22)
!62 = !DILocation(line: 22, column: 1, scope: !28)
!63 = !DILocation(line: 23, column: 1, scope: !28)
!64 = !DILocalVariable(name: "omp_sched_static", scope: !31, file: !4, type: !9)
!65 = !DILocation(line: 0, scope: !31)
!66 = !DILocalVariable(name: "omp_proc_bind_false", scope: !31, file: !4, type: !9)
!67 = !DILocalVariable(name: "omp_proc_bind_true", scope: !31, file: !4, type: !9)
!68 = !DILocalVariable(name: "omp_lock_hint_none", scope: !31, file: !4, type: !9)
!69 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !31, file: !4, type: !9)
!70 = !DILocation(line: 37, column: 1, scope: !31)
!71 = !DILocation(line: 30, column: 1, scope: !31)
!72 = !DILocation(line: 35, column: 1, scope: !31)
!73 = !DILocation(line: 36, column: 1, scope: !31)
!74 = !DILocalVariable(scope: !31, file: !4, type: !9, flags: DIFlagArtificial)

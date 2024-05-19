; ModuleID = '/tmp/DRB084-threadprivatemissing-orig-yes-0d3bb9.ll'
source_filename = "/tmp/DRB084-threadprivatemissing-orig-yes-0d3bb9.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct__cs_unspc_ = type <{ [32 x i8] }>
%struct_drb084_2_ = type <{ [16 x i8] }>
%astruct.dt66 = type <{ i8* }>

@.C335_MAIN_ = internal constant [6 x i8] c"sum1 ="
@.C339_MAIN_ = internal constant i32 26
@.C306_MAIN_ = internal constant i32 14
@.C334_MAIN_ = internal constant [6 x i8] c"sum = "
@.C331_MAIN_ = internal constant i32 6
@.C328_MAIN_ = internal constant [65 x i8] c"micro-benchmarks-fortran/DRB084-threadprivatemissing-orig-yes.f95"
@.C330_MAIN_ = internal constant i32 47
@.C319_MAIN_ = internal constant i64 1001
@.C286_MAIN_ = internal constant i64 1
@.C284_MAIN_ = internal constant i64 0
@.C283_MAIN_ = internal constant i32 0
@.C319___nv_MAIN__F1L32_1 = internal constant i64 1001
@.C286___nv_MAIN__F1L32_1 = internal constant i64 1
@.C283___nv_MAIN__F1L32_1 = internal constant i32 0
@__cs_unspc_ = common global %struct__cs_unspc_ zeroinitializer, align 64
@_drb084_2_ = common global %struct_drb084_2_ zeroinitializer, align 64, !dbg !0, !dbg !7

; Function Attrs: noinline
define float @drb084_() #0 {
.L.entry:
  ret float undef
}

define void @drb084_foo_(i64* %i) #1 !dbg !28 {
L.entry:
  call void @llvm.dbg.declare(metadata i64* %i, metadata !31, metadata !DIExpression()), !dbg !32
  br label %L.LB2_306

L.LB2_306:                                        ; preds = %L.entry
  %0 = load i64, i64* %i, align 8, !dbg !33
  %1 = bitcast %struct_drb084_2_* @_drb084_2_ to i64*, !dbg !33
  %2 = load i64, i64* %1, align 8, !dbg !33
  %3 = add nsw i64 %0, %2, !dbg !33
  %4 = bitcast %struct_drb084_2_* @_drb084_2_ to i64*, !dbg !33
  store i64 %3, i64* %4, align 8, !dbg !33
  ret void, !dbg !34
}

define void @MAIN_() #1 !dbg !23 {
L.entry:
  %__gtid_MAIN__372 = alloca i32, align 4
  %sum_313 = alloca i64, align 8
  %.uplevelArgPack0001_367 = alloca %astruct.dt66, align 8
  %.dY0002_359 = alloca i64, align 8
  %i_312 = alloca i64, align 8
  %z__io_333 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !35, metadata !DIExpression()), !dbg !36
  call void @llvm.dbg.value(metadata i32 0, metadata !37, metadata !DIExpression()), !dbg !36
  call void @llvm.dbg.value(metadata i32 1, metadata !38, metadata !DIExpression()), !dbg !36
  call void @llvm.dbg.value(metadata i32 0, metadata !39, metadata !DIExpression()), !dbg !36
  call void @llvm.dbg.value(metadata i32 1, metadata !40, metadata !DIExpression()), !dbg !36
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !41
  store i32 %0, i32* %__gtid_MAIN__372, align 4, !dbg !41
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !42
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !42
  call void (i8*, ...) %2(i8* %1), !dbg !42
  br label %L.LB3_361

L.LB3_361:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i64* %sum_313, metadata !43, metadata !DIExpression()), !dbg !36
  store i64 0, i64* %sum_313, align 8, !dbg !44
  %3 = bitcast i64* %sum_313 to i8*, !dbg !45
  %4 = bitcast %astruct.dt66* %.uplevelArgPack0001_367 to i8**, !dbg !45
  store i8* %3, i8** %4, align 8, !dbg !45
  br label %L.LB3_370, !dbg !45

L.LB3_370:                                        ; preds = %L.LB3_361
  %5 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L32_1_ to i64*, !dbg !45
  %6 = bitcast %astruct.dt66* %.uplevelArgPack0001_367 to i64*, !dbg !45
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %5, i64* %6), !dbg !45
  store i64 1001, i64* %.dY0002_359, align 8, !dbg !46
  call void @llvm.dbg.declare(metadata i64* %i_312, metadata !47, metadata !DIExpression()), !dbg !36
  store i64 1, i64* %i_312, align 8, !dbg !46
  br label %L.LB3_357

L.LB3_357:                                        ; preds = %L.LB3_357, %L.LB3_370
  %7 = bitcast %struct_drb084_2_* @_drb084_2_ to i8*, !dbg !48
  %8 = getelementptr i8, i8* %7, i64 8, !dbg !48
  %9 = bitcast i8* %8 to i64*, !dbg !48
  %10 = load i64, i64* %9, align 8, !dbg !48
  %11 = load i64, i64* %i_312, align 8, !dbg !48
  call void @llvm.dbg.value(metadata i64 %11, metadata !47, metadata !DIExpression()), !dbg !36
  %12 = add nsw i64 %10, %11, !dbg !48
  %13 = bitcast %struct_drb084_2_* @_drb084_2_ to i8*, !dbg !48
  %14 = getelementptr i8, i8* %13, i64 8, !dbg !48
  %15 = bitcast i8* %14 to i64*, !dbg !48
  store i64 %12, i64* %15, align 8, !dbg !48
  %16 = load i64, i64* %i_312, align 8, !dbg !49
  call void @llvm.dbg.value(metadata i64 %16, metadata !47, metadata !DIExpression()), !dbg !36
  %17 = add nsw i64 %16, 1, !dbg !49
  store i64 %17, i64* %i_312, align 8, !dbg !49
  %18 = load i64, i64* %.dY0002_359, align 8, !dbg !49
  %19 = sub nsw i64 %18, 1, !dbg !49
  store i64 %19, i64* %.dY0002_359, align 8, !dbg !49
  %20 = load i64, i64* %.dY0002_359, align 8, !dbg !49
  %21 = icmp sgt i64 %20, 0, !dbg !49
  br i1 %21, label %L.LB3_357, label %L.LB3_404, !dbg !49

L.LB3_404:                                        ; preds = %L.LB3_357
  call void (...) @_mp_bcs_nest(), !dbg !50
  %22 = bitcast i32* @.C330_MAIN_ to i8*, !dbg !50
  %23 = bitcast [65 x i8]* @.C328_MAIN_ to i8*, !dbg !50
  %24 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !50
  call void (i8*, i8*, i64, ...) %24(i8* %22, i8* %23, i64 65), !dbg !50
  %25 = bitcast i32* @.C331_MAIN_ to i8*, !dbg !50
  %26 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !50
  %27 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !50
  %28 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !50
  %29 = call i32 (i8*, i8*, i8*, i8*, ...) %28(i8* %25, i8* null, i8* %26, i8* %27), !dbg !50
  call void @llvm.dbg.declare(metadata i32* %z__io_333, metadata !51, metadata !DIExpression()), !dbg !36
  store i32 %29, i32* %z__io_333, align 4, !dbg !50
  %30 = bitcast [6 x i8]* @.C334_MAIN_ to i8*, !dbg !50
  %31 = bitcast i32 (...)* @f90io_sc_ch_ldw to i32 (i8*, i32, i64, ...)*, !dbg !50
  %32 = call i32 (i8*, i32, i64, ...) %31(i8* %30, i32 14, i64 6), !dbg !50
  store i32 %32, i32* %z__io_333, align 4, !dbg !50
  %33 = load i64, i64* %sum_313, align 8, !dbg !50
  call void @llvm.dbg.value(metadata i64 %33, metadata !43, metadata !DIExpression()), !dbg !36
  %34 = bitcast i32 (...)* @f90io_sc_l_ldw to i32 (i64, i32, ...)*, !dbg !50
  %35 = call i32 (i64, i32, ...) %34(i64 %33, i32 26), !dbg !50
  store i32 %35, i32* %z__io_333, align 4, !dbg !50
  %36 = bitcast [6 x i8]* @.C335_MAIN_ to i8*, !dbg !50
  %37 = bitcast i32 (...)* @f90io_sc_ch_ldw to i32 (i8*, i32, i64, ...)*, !dbg !50
  %38 = call i32 (i8*, i32, i64, ...) %37(i8* %36, i32 14, i64 6), !dbg !50
  store i32 %38, i32* %z__io_333, align 4, !dbg !50
  %39 = bitcast %struct_drb084_2_* @_drb084_2_ to i8*, !dbg !50
  %40 = getelementptr i8, i8* %39, i64 8, !dbg !50
  %41 = bitcast i8* %40 to i64*, !dbg !50
  %42 = load i64, i64* %41, align 8, !dbg !50
  %43 = bitcast i32 (...)* @f90io_sc_l_ldw to i32 (i64, i32, ...)*, !dbg !50
  %44 = call i32 (i64, i32, ...) %43(i64 %42, i32 26), !dbg !50
  store i32 %44, i32* %z__io_333, align 4, !dbg !50
  %45 = call i32 (...) @f90io_ldw_end(), !dbg !50
  store i32 %45, i32* %z__io_333, align 4, !dbg !50
  call void (...) @_mp_ecs_nest(), !dbg !50
  ret void, !dbg !41
}

define internal void @__nv_MAIN__F1L32_1_(i32* %__nv_MAIN__F1L32_1Arg0, i64* %__nv_MAIN__F1L32_1Arg1, i64* %__nv_MAIN__F1L32_1Arg2) #1 !dbg !13 {
L.entry:
  %__gtid___nv_MAIN__F1L32_1__422 = alloca i32, align 4
  %.i0000p_320 = alloca i32, align 4
  %i_318 = alloca i64, align 8
  %.du0001p_348 = alloca i64, align 8
  %.de0001p_349 = alloca i64, align 8
  %.di0001p_350 = alloca i64, align 8
  %.ds0001p_351 = alloca i64, align 8
  %.dl0001p_353 = alloca i64, align 8
  %.dl0001p.copy_416 = alloca i64, align 8
  %.de0001p.copy_417 = alloca i64, align 8
  %.ds0001p.copy_418 = alloca i64, align 8
  %.dX0001p_352 = alloca i64, align 8
  %.dY0001p_347 = alloca i64, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L32_1Arg0, metadata !52, metadata !DIExpression()), !dbg !53
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L32_1Arg1, metadata !54, metadata !DIExpression()), !dbg !53
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L32_1Arg2, metadata !55, metadata !DIExpression()), !dbg !53
  call void @llvm.dbg.value(metadata i32 1, metadata !56, metadata !DIExpression()), !dbg !53
  call void @llvm.dbg.value(metadata i32 0, metadata !57, metadata !DIExpression()), !dbg !53
  call void @llvm.dbg.value(metadata i32 1, metadata !58, metadata !DIExpression()), !dbg !53
  call void @llvm.dbg.value(metadata i32 0, metadata !59, metadata !DIExpression()), !dbg !53
  call void @llvm.dbg.value(metadata i32 1, metadata !60, metadata !DIExpression()), !dbg !53
  %0 = load i32, i32* %__nv_MAIN__F1L32_1Arg0, align 4, !dbg !61
  store i32 %0, i32* %__gtid___nv_MAIN__F1L32_1__422, align 4, !dbg !61
  br label %L.LB4_408

L.LB4_408:                                        ; preds = %L.entry
  br label %L.LB4_316

L.LB4_316:                                        ; preds = %L.LB4_408
  br label %L.LB4_317

L.LB4_317:                                        ; preds = %L.LB4_316
  store i32 0, i32* %.i0000p_320, align 4, !dbg !62
  call void @llvm.dbg.declare(metadata i64* %i_318, metadata !63, metadata !DIExpression()), !dbg !61
  store i64 1, i64* %i_318, align 8, !dbg !62
  store i64 1001, i64* %.du0001p_348, align 8, !dbg !62
  store i64 1001, i64* %.de0001p_349, align 8, !dbg !62
  store i64 1, i64* %.di0001p_350, align 8, !dbg !62
  %1 = load i64, i64* %.di0001p_350, align 8, !dbg !62
  store i64 %1, i64* %.ds0001p_351, align 8, !dbg !62
  store i64 1, i64* %.dl0001p_353, align 8, !dbg !62
  %2 = load i64, i64* %.dl0001p_353, align 8, !dbg !62
  store i64 %2, i64* %.dl0001p.copy_416, align 8, !dbg !62
  %3 = load i64, i64* %.de0001p_349, align 8, !dbg !62
  store i64 %3, i64* %.de0001p.copy_417, align 8, !dbg !62
  %4 = load i64, i64* %.ds0001p_351, align 8, !dbg !62
  store i64 %4, i64* %.ds0001p.copy_418, align 8, !dbg !62
  %5 = load i32, i32* %__gtid___nv_MAIN__F1L32_1__422, align 4, !dbg !62
  %6 = bitcast i32* %.i0000p_320 to i64*, !dbg !62
  %7 = load i64, i64* %.ds0001p.copy_418, align 8, !dbg !62
  call void @__kmpc_for_static_init_8(i64* null, i32 %5, i32 34, i64* %6, i64* %.dl0001p.copy_416, i64* %.de0001p.copy_417, i64* %.ds0001p.copy_418, i64 %7, i64 1), !dbg !62
  %8 = load i64, i64* %.dl0001p.copy_416, align 8, !dbg !62
  store i64 %8, i64* %.dl0001p_353, align 8, !dbg !62
  %9 = load i64, i64* %.de0001p.copy_417, align 8, !dbg !62
  store i64 %9, i64* %.de0001p_349, align 8, !dbg !62
  %10 = load i64, i64* %.ds0001p.copy_418, align 8, !dbg !62
  store i64 %10, i64* %.ds0001p_351, align 8, !dbg !62
  %11 = load i64, i64* %.dl0001p_353, align 8, !dbg !62
  store i64 %11, i64* %i_318, align 8, !dbg !62
  %12 = load i64, i64* %i_318, align 8, !dbg !62
  call void @llvm.dbg.value(metadata i64 %12, metadata !63, metadata !DIExpression()), !dbg !61
  store i64 %12, i64* %.dX0001p_352, align 8, !dbg !62
  %13 = load i64, i64* %.dX0001p_352, align 8, !dbg !62
  %14 = load i64, i64* %.du0001p_348, align 8, !dbg !62
  %15 = icmp sgt i64 %13, %14, !dbg !62
  br i1 %15, label %L.LB4_346, label %L.LB4_459, !dbg !62

L.LB4_459:                                        ; preds = %L.LB4_317
  %16 = load i64, i64* %.dX0001p_352, align 8, !dbg !62
  store i64 %16, i64* %i_318, align 8, !dbg !62
  %17 = load i64, i64* %.di0001p_350, align 8, !dbg !62
  %18 = load i64, i64* %.de0001p_349, align 8, !dbg !62
  %19 = load i64, i64* %.dX0001p_352, align 8, !dbg !62
  %20 = sub nsw i64 %18, %19, !dbg !62
  %21 = add nsw i64 %17, %20, !dbg !62
  %22 = load i64, i64* %.di0001p_350, align 8, !dbg !62
  %23 = sdiv i64 %21, %22, !dbg !62
  store i64 %23, i64* %.dY0001p_347, align 8, !dbg !62
  %24 = load i64, i64* %.dY0001p_347, align 8, !dbg !62
  %25 = icmp sle i64 %24, 0, !dbg !62
  br i1 %25, label %L.LB4_356, label %L.LB4_355, !dbg !62

L.LB4_355:                                        ; preds = %L.LB4_355, %L.LB4_459
  call void @drb084_foo_(i64* %i_318), !dbg !64
  %26 = load i64, i64* %.di0001p_350, align 8, !dbg !65
  %27 = load i64, i64* %i_318, align 8, !dbg !65
  call void @llvm.dbg.value(metadata i64 %27, metadata !63, metadata !DIExpression()), !dbg !61
  %28 = add nsw i64 %26, %27, !dbg !65
  store i64 %28, i64* %i_318, align 8, !dbg !65
  %29 = load i64, i64* %.dY0001p_347, align 8, !dbg !65
  %30 = sub nsw i64 %29, 1, !dbg !65
  store i64 %30, i64* %.dY0001p_347, align 8, !dbg !65
  %31 = load i64, i64* %.dY0001p_347, align 8, !dbg !65
  %32 = icmp sgt i64 %31, 0, !dbg !65
  br i1 %32, label %L.LB4_355, label %L.LB4_356, !dbg !65

L.LB4_356:                                        ; preds = %L.LB4_355, %L.LB4_459
  br label %L.LB4_346

L.LB4_346:                                        ; preds = %L.LB4_356, %L.LB4_317
  %33 = load i32, i32* %__gtid___nv_MAIN__F1L32_1__422, align 4, !dbg !65
  call void @__kmpc_for_static_fini(i64* null, i32 %33), !dbg !65
  br label %L.LB4_321

L.LB4_321:                                        ; preds = %L.LB4_346
  %34 = load i32, i32* %__gtid___nv_MAIN__F1L32_1__422, align 4, !dbg !66
  call void @__kmpc_barrier(i64* null, i32 %34), !dbg !66
  %35 = load i32, i32* %__gtid___nv_MAIN__F1L32_1__422, align 4, !dbg !67
  %36 = bitcast %struct__cs_unspc_* @__cs_unspc_ to i64*, !dbg !67
  call void @__kmpc_critical(i64* null, i32 %35, i64* %36), !dbg !67
  %37 = bitcast %struct_drb084_2_* @_drb084_2_ to i64*, !dbg !67
  %38 = load i64, i64* %37, align 8, !dbg !67
  %39 = bitcast i64* %__nv_MAIN__F1L32_1Arg2 to i64**, !dbg !67
  %40 = load i64*, i64** %39, align 8, !dbg !67
  %41 = load i64, i64* %40, align 8, !dbg !67
  %42 = add nsw i64 %38, %41, !dbg !67
  %43 = bitcast i64* %__nv_MAIN__F1L32_1Arg2 to i64**, !dbg !67
  %44 = load i64*, i64** %43, align 8, !dbg !67
  store i64 %42, i64* %44, align 8, !dbg !67
  %45 = load i32, i32* %__gtid___nv_MAIN__F1L32_1__422, align 4, !dbg !67
  %46 = bitcast %struct__cs_unspc_* @__cs_unspc_ to i64*, !dbg !67
  call void @__kmpc_end_critical(i64* null, i32 %45, i64* %46), !dbg !67
  br label %L.LB4_326

L.LB4_326:                                        ; preds = %L.LB4_321
  ret void, !dbg !61
}

declare void @__kmpc_end_critical(i64*, i32, i64*) #1

declare void @__kmpc_critical(i64*, i32, i64*) #1

declare void @__kmpc_barrier(i64*, i32) #1

declare void @__kmpc_for_static_fini(i64*, i32) #1

declare void @__kmpc_for_static_init_8(i64*, i32, i32, i64*, i64*, i64*, i64*, i64, i64) #1

declare void @_mp_ecs_nest(...) #1

declare signext i32 @f90io_ldw_end(...) #1

declare signext i32 @f90io_sc_l_ldw(...) #1

declare signext i32 @f90io_sc_ch_ldw(...) #1

declare signext i32 @f90io_print_init(...) #1

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

!llvm.module.flags = !{!26, !27}
!llvm.dbg.cu = !{!3}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "sum0", scope: !2, file: !4, type: !9, isLocal: false, isDefinition: true)
!2 = !DIModule(scope: !3, name: "drb084")
!3 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !4, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !6, imports: !21)
!4 = !DIFile(filename: "micro-benchmarks-fortran/DRB084-threadprivatemissing-orig-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!5 = !{}
!6 = !{!0, !7, !10}
!7 = !DIGlobalVariableExpression(var: !8, expr: !DIExpression(DW_OP_plus_uconst, 8))
!8 = distinct !DIGlobalVariable(name: "sum1", scope: !2, file: !4, type: !9, isLocal: false, isDefinition: true)
!9 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!10 = !DIGlobalVariableExpression(var: !11, expr: !DIExpression())
!11 = distinct !DIGlobalVariable(name: "__cs_unspc", scope: !12, type: !17, isLocal: false, isDefinition: true)
!12 = distinct !DICommonBlock(scope: !13, declaration: !11, name: "__cs_unspc")
!13 = distinct !DISubprogram(name: "__nv_MAIN__F1L32_1", scope: !3, file: !4, line: 32, type: !14, scopeLine: 32, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !3)
!14 = !DISubroutineType(types: !15)
!15 = !{null, !16, !9, !9}
!16 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!17 = !DICompositeType(tag: DW_TAG_array_type, baseType: !18, size: 256, align: 8, elements: !19)
!18 = !DIBasicType(name: "byte", size: 8, align: 8, encoding: DW_ATE_signed)
!19 = !{!20}
!20 = !DISubrange(count: 32)
!21 = !{!22}
!22 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !23, entity: !2, file: !4, line: 24)
!23 = distinct !DISubprogram(name: "drb084_threadprivatemissing_orig_yes", scope: !3, file: !4, line: 24, type: !24, scopeLine: 24, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !3)
!24 = !DISubroutineType(cc: DW_CC_program, types: !25)
!25 = !{null}
!26 = !{i32 2, !"Dwarf Version", i32 4}
!27 = !{i32 2, !"Debug Info Version", i32 3}
!28 = distinct !DISubprogram(name: "foo", scope: !2, file: !4, line: 18, type: !29, scopeLine: 18, spFlags: DISPFlagDefinition, unit: !3)
!29 = !DISubroutineType(types: !30)
!30 = !{null, !9}
!31 = !DILocalVariable(name: "i", arg: 1, scope: !28, file: !4, type: !9)
!32 = !DILocation(line: 0, scope: !28)
!33 = !DILocation(line: 20, column: 1, scope: !28)
!34 = !DILocation(line: 21, column: 1, scope: !28)
!35 = !DILocalVariable(name: "omp_sched_static", scope: !23, file: !4, type: !16)
!36 = !DILocation(line: 0, scope: !23)
!37 = !DILocalVariable(name: "omp_proc_bind_false", scope: !23, file: !4, type: !16)
!38 = !DILocalVariable(name: "omp_proc_bind_true", scope: !23, file: !4, type: !16)
!39 = !DILocalVariable(name: "omp_lock_hint_none", scope: !23, file: !4, type: !16)
!40 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !23, file: !4, type: !16)
!41 = !DILocation(line: 48, column: 1, scope: !23)
!42 = !DILocation(line: 24, column: 1, scope: !23)
!43 = !DILocalVariable(name: "sum", scope: !23, file: !4, type: !9)
!44 = !DILocation(line: 30, column: 1, scope: !23)
!45 = !DILocation(line: 32, column: 1, scope: !23)
!46 = !DILocation(line: 43, column: 1, scope: !23)
!47 = !DILocalVariable(name: "i", scope: !23, file: !4, type: !9)
!48 = !DILocation(line: 44, column: 1, scope: !23)
!49 = !DILocation(line: 45, column: 1, scope: !23)
!50 = !DILocation(line: 47, column: 1, scope: !23)
!51 = !DILocalVariable(scope: !23, file: !4, type: !16, flags: DIFlagArtificial)
!52 = !DILocalVariable(name: "__nv_MAIN__F1L32_1Arg0", arg: 1, scope: !13, file: !4, type: !16)
!53 = !DILocation(line: 0, scope: !13)
!54 = !DILocalVariable(name: "__nv_MAIN__F1L32_1Arg1", arg: 2, scope: !13, file: !4, type: !9)
!55 = !DILocalVariable(name: "__nv_MAIN__F1L32_1Arg2", arg: 3, scope: !13, file: !4, type: !9)
!56 = !DILocalVariable(name: "omp_sched_static", scope: !13, file: !4, type: !16)
!57 = !DILocalVariable(name: "omp_proc_bind_false", scope: !13, file: !4, type: !16)
!58 = !DILocalVariable(name: "omp_proc_bind_true", scope: !13, file: !4, type: !16)
!59 = !DILocalVariable(name: "omp_lock_hint_none", scope: !13, file: !4, type: !16)
!60 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !13, file: !4, type: !16)
!61 = !DILocation(line: 41, column: 1, scope: !13)
!62 = !DILocation(line: 34, column: 1, scope: !13)
!63 = !DILocalVariable(name: "i", scope: !13, file: !4, type: !9)
!64 = !DILocation(line: 35, column: 1, scope: !13)
!65 = !DILocation(line: 36, column: 1, scope: !13)
!66 = !DILocation(line: 37, column: 1, scope: !13)
!67 = !DILocation(line: 39, column: 1, scope: !13)

; ModuleID = '/tmp/DRB092-threadprivatemissing2-orig-yes-01c259.ll'
source_filename = "/tmp/DRB092-threadprivatemissing2-orig-yes-01c259.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct__cs_unspc_ = type <{ [32 x i8] }>
%struct_drb092_0_ = type <{ [8 x i8] }>
%astruct.dt65 = type <{ i8* }>

@.C334_MAIN_ = internal constant [6 x i8] c"sum1 ="
@.C307_MAIN_ = internal constant i32 25
@.C306_MAIN_ = internal constant i32 14
@.C333_MAIN_ = internal constant [5 x i8] c"sum ="
@.C284_MAIN_ = internal constant i64 0
@.C330_MAIN_ = internal constant i32 6
@.C327_MAIN_ = internal constant [66 x i8] c"micro-benchmarks-fortran/DRB092-threadprivatemissing2-orig-yes.f95"
@.C329_MAIN_ = internal constant i32 46
@.C317_MAIN_ = internal constant i32 1001
@.C285_MAIN_ = internal constant i32 1
@.C283_MAIN_ = internal constant i32 0
@.C317___nv_MAIN__F1L31_1 = internal constant i32 1001
@.C285___nv_MAIN__F1L31_1 = internal constant i32 1
@.C283___nv_MAIN__F1L31_1 = internal constant i32 0
@__cs_unspc_ = common global %struct__cs_unspc_ zeroinitializer, align 64
@_drb092_0_ = common global %struct_drb092_0_ zeroinitializer, align 64, !dbg !0, !dbg !7

; Function Attrs: noinline
define float @drb092_() #0 {
.L.entry:
  ret float undef
}

define void @MAIN_() #1 !dbg !23 {
L.entry:
  %__gtid_MAIN__372 = alloca i32, align 4
  %sum_312 = alloca i32, align 4
  %.uplevelArgPack0001_367 = alloca %astruct.dt65, align 8
  %.dY0002_357 = alloca i32, align 4
  %i_311 = alloca i32, align 4
  %z__io_332 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !28, metadata !DIExpression()), !dbg !29
  call void @llvm.dbg.value(metadata i32 0, metadata !30, metadata !DIExpression()), !dbg !29
  call void @llvm.dbg.value(metadata i32 1, metadata !31, metadata !DIExpression()), !dbg !29
  call void @llvm.dbg.value(metadata i32 0, metadata !32, metadata !DIExpression()), !dbg !29
  call void @llvm.dbg.value(metadata i32 1, metadata !33, metadata !DIExpression()), !dbg !29
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !34
  store i32 %0, i32* %__gtid_MAIN__372, align 4, !dbg !34
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !35
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !35
  call void (i8*, ...) %2(i8* %1), !dbg !35
  br label %L.LB2_359

L.LB2_359:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %sum_312, metadata !36, metadata !DIExpression()), !dbg !29
  store i32 0, i32* %sum_312, align 4, !dbg !37
  %3 = bitcast %struct_drb092_0_* @_drb092_0_ to i32*, !dbg !38
  store i32 0, i32* %3, align 4, !dbg !38
  %4 = bitcast %struct_drb092_0_* @_drb092_0_ to i8*, !dbg !39
  %5 = getelementptr i8, i8* %4, i64 4, !dbg !39
  %6 = bitcast i8* %5 to i32*, !dbg !39
  store i32 0, i32* %6, align 4, !dbg !39
  %7 = bitcast i32* %sum_312 to i8*, !dbg !40
  %8 = bitcast %astruct.dt65* %.uplevelArgPack0001_367 to i8**, !dbg !40
  store i8* %7, i8** %8, align 8, !dbg !40
  br label %L.LB2_370, !dbg !40

L.LB2_370:                                        ; preds = %L.LB2_359
  %9 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L31_1_ to i64*, !dbg !40
  %10 = bitcast %astruct.dt65* %.uplevelArgPack0001_367 to i64*, !dbg !40
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %9, i64* %10), !dbg !40
  store i32 1001, i32* %.dY0002_357, align 4, !dbg !41
  call void @llvm.dbg.declare(metadata i32* %i_311, metadata !42, metadata !DIExpression()), !dbg !29
  store i32 1, i32* %i_311, align 4, !dbg !41
  br label %L.LB2_355

L.LB2_355:                                        ; preds = %L.LB2_355, %L.LB2_370
  %11 = load i32, i32* %i_311, align 4, !dbg !43
  call void @llvm.dbg.value(metadata i32 %11, metadata !42, metadata !DIExpression()), !dbg !29
  %12 = bitcast %struct_drb092_0_* @_drb092_0_ to i8*, !dbg !43
  %13 = getelementptr i8, i8* %12, i64 4, !dbg !43
  %14 = bitcast i8* %13 to i32*, !dbg !43
  %15 = load i32, i32* %14, align 4, !dbg !43
  %16 = add nsw i32 %11, %15, !dbg !43
  %17 = bitcast %struct_drb092_0_* @_drb092_0_ to i8*, !dbg !43
  %18 = getelementptr i8, i8* %17, i64 4, !dbg !43
  %19 = bitcast i8* %18 to i32*, !dbg !43
  store i32 %16, i32* %19, align 4, !dbg !43
  %20 = load i32, i32* %i_311, align 4, !dbg !44
  call void @llvm.dbg.value(metadata i32 %20, metadata !42, metadata !DIExpression()), !dbg !29
  %21 = add nsw i32 %20, 1, !dbg !44
  store i32 %21, i32* %i_311, align 4, !dbg !44
  %22 = load i32, i32* %.dY0002_357, align 4, !dbg !44
  %23 = sub nsw i32 %22, 1, !dbg !44
  store i32 %23, i32* %.dY0002_357, align 4, !dbg !44
  %24 = load i32, i32* %.dY0002_357, align 4, !dbg !44
  %25 = icmp sgt i32 %24, 0, !dbg !44
  br i1 %25, label %L.LB2_355, label %L.LB2_405, !dbg !44

L.LB2_405:                                        ; preds = %L.LB2_355
  call void (...) @_mp_bcs_nest(), !dbg !45
  %26 = bitcast i32* @.C329_MAIN_ to i8*, !dbg !45
  %27 = bitcast [66 x i8]* @.C327_MAIN_ to i8*, !dbg !45
  %28 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !45
  call void (i8*, i8*, i64, ...) %28(i8* %26, i8* %27, i64 66), !dbg !45
  %29 = bitcast i32* @.C330_MAIN_ to i8*, !dbg !45
  %30 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !45
  %31 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !45
  %32 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !45
  %33 = call i32 (i8*, i8*, i8*, i8*, ...) %32(i8* %29, i8* null, i8* %30, i8* %31), !dbg !45
  call void @llvm.dbg.declare(metadata i32* %z__io_332, metadata !46, metadata !DIExpression()), !dbg !29
  store i32 %33, i32* %z__io_332, align 4, !dbg !45
  %34 = bitcast [5 x i8]* @.C333_MAIN_ to i8*, !dbg !45
  %35 = bitcast i32 (...)* @f90io_sc_ch_ldw to i32 (i8*, i32, i64, ...)*, !dbg !45
  %36 = call i32 (i8*, i32, i64, ...) %35(i8* %34, i32 14, i64 5), !dbg !45
  store i32 %36, i32* %z__io_332, align 4, !dbg !45
  %37 = load i32, i32* %sum_312, align 4, !dbg !45
  call void @llvm.dbg.value(metadata i32 %37, metadata !36, metadata !DIExpression()), !dbg !29
  %38 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !45
  %39 = call i32 (i32, i32, ...) %38(i32 %37, i32 25), !dbg !45
  store i32 %39, i32* %z__io_332, align 4, !dbg !45
  %40 = bitcast [6 x i8]* @.C334_MAIN_ to i8*, !dbg !45
  %41 = bitcast i32 (...)* @f90io_sc_ch_ldw to i32 (i8*, i32, i64, ...)*, !dbg !45
  %42 = call i32 (i8*, i32, i64, ...) %41(i8* %40, i32 14, i64 6), !dbg !45
  store i32 %42, i32* %z__io_332, align 4, !dbg !45
  %43 = bitcast %struct_drb092_0_* @_drb092_0_ to i8*, !dbg !45
  %44 = getelementptr i8, i8* %43, i64 4, !dbg !45
  %45 = bitcast i8* %44 to i32*, !dbg !45
  %46 = load i32, i32* %45, align 4, !dbg !45
  %47 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !45
  %48 = call i32 (i32, i32, ...) %47(i32 %46, i32 25), !dbg !45
  store i32 %48, i32* %z__io_332, align 4, !dbg !45
  %49 = call i32 (...) @f90io_ldw_end(), !dbg !45
  store i32 %49, i32* %z__io_332, align 4, !dbg !45
  call void (...) @_mp_ecs_nest(), !dbg !45
  ret void, !dbg !34
}

define internal void @__nv_MAIN__F1L31_1_(i32* %__nv_MAIN__F1L31_1Arg0, i64* %__nv_MAIN__F1L31_1Arg1, i64* %__nv_MAIN__F1L31_1Arg2) #1 !dbg !13 {
L.entry:
  %__gtid___nv_MAIN__F1L31_1__423 = alloca i32, align 4
  %.i0000p_319 = alloca i32, align 4
  %i_318 = alloca i32, align 4
  %.du0001p_346 = alloca i32, align 4
  %.de0001p_347 = alloca i32, align 4
  %.di0001p_348 = alloca i32, align 4
  %.ds0001p_349 = alloca i32, align 4
  %.dl0001p_351 = alloca i32, align 4
  %.dl0001p.copy_417 = alloca i32, align 4
  %.de0001p.copy_418 = alloca i32, align 4
  %.ds0001p.copy_419 = alloca i32, align 4
  %.dX0001p_350 = alloca i32, align 4
  %.dY0001p_345 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L31_1Arg0, metadata !47, metadata !DIExpression()), !dbg !48
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L31_1Arg1, metadata !49, metadata !DIExpression()), !dbg !48
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L31_1Arg2, metadata !50, metadata !DIExpression()), !dbg !48
  call void @llvm.dbg.value(metadata i32 1, metadata !51, metadata !DIExpression()), !dbg !48
  call void @llvm.dbg.value(metadata i32 0, metadata !52, metadata !DIExpression()), !dbg !48
  call void @llvm.dbg.value(metadata i32 1, metadata !53, metadata !DIExpression()), !dbg !48
  call void @llvm.dbg.value(metadata i32 0, metadata !54, metadata !DIExpression()), !dbg !48
  call void @llvm.dbg.value(metadata i32 1, metadata !55, metadata !DIExpression()), !dbg !48
  %0 = load i32, i32* %__nv_MAIN__F1L31_1Arg0, align 4, !dbg !56
  store i32 %0, i32* %__gtid___nv_MAIN__F1L31_1__423, align 4, !dbg !56
  br label %L.LB3_409

L.LB3_409:                                        ; preds = %L.entry
  br label %L.LB3_315

L.LB3_315:                                        ; preds = %L.LB3_409
  br label %L.LB3_316

L.LB3_316:                                        ; preds = %L.LB3_315
  store i32 0, i32* %.i0000p_319, align 4, !dbg !57
  call void @llvm.dbg.declare(metadata i32* %i_318, metadata !58, metadata !DIExpression()), !dbg !56
  store i32 1, i32* %i_318, align 4, !dbg !57
  store i32 1001, i32* %.du0001p_346, align 4, !dbg !57
  store i32 1001, i32* %.de0001p_347, align 4, !dbg !57
  store i32 1, i32* %.di0001p_348, align 4, !dbg !57
  %1 = load i32, i32* %.di0001p_348, align 4, !dbg !57
  store i32 %1, i32* %.ds0001p_349, align 4, !dbg !57
  store i32 1, i32* %.dl0001p_351, align 4, !dbg !57
  %2 = load i32, i32* %.dl0001p_351, align 4, !dbg !57
  store i32 %2, i32* %.dl0001p.copy_417, align 4, !dbg !57
  %3 = load i32, i32* %.de0001p_347, align 4, !dbg !57
  store i32 %3, i32* %.de0001p.copy_418, align 4, !dbg !57
  %4 = load i32, i32* %.ds0001p_349, align 4, !dbg !57
  store i32 %4, i32* %.ds0001p.copy_419, align 4, !dbg !57
  %5 = load i32, i32* %__gtid___nv_MAIN__F1L31_1__423, align 4, !dbg !57
  %6 = bitcast i32* %.i0000p_319 to i64*, !dbg !57
  %7 = bitcast i32* %.dl0001p.copy_417 to i64*, !dbg !57
  %8 = bitcast i32* %.de0001p.copy_418 to i64*, !dbg !57
  %9 = bitcast i32* %.ds0001p.copy_419 to i64*, !dbg !57
  %10 = load i32, i32* %.ds0001p.copy_419, align 4, !dbg !57
  call void @__kmpc_for_static_init_4(i64* null, i32 %5, i32 34, i64* %6, i64* %7, i64* %8, i64* %9, i32 %10, i32 1), !dbg !57
  %11 = load i32, i32* %.dl0001p.copy_417, align 4, !dbg !57
  store i32 %11, i32* %.dl0001p_351, align 4, !dbg !57
  %12 = load i32, i32* %.de0001p.copy_418, align 4, !dbg !57
  store i32 %12, i32* %.de0001p_347, align 4, !dbg !57
  %13 = load i32, i32* %.ds0001p.copy_419, align 4, !dbg !57
  store i32 %13, i32* %.ds0001p_349, align 4, !dbg !57
  %14 = load i32, i32* %.dl0001p_351, align 4, !dbg !57
  store i32 %14, i32* %i_318, align 4, !dbg !57
  %15 = load i32, i32* %i_318, align 4, !dbg !57
  call void @llvm.dbg.value(metadata i32 %15, metadata !58, metadata !DIExpression()), !dbg !56
  store i32 %15, i32* %.dX0001p_350, align 4, !dbg !57
  %16 = load i32, i32* %.dX0001p_350, align 4, !dbg !57
  %17 = load i32, i32* %.du0001p_346, align 4, !dbg !57
  %18 = icmp sgt i32 %16, %17, !dbg !57
  br i1 %18, label %L.LB3_344, label %L.LB3_459, !dbg !57

L.LB3_459:                                        ; preds = %L.LB3_316
  %19 = load i32, i32* %.dX0001p_350, align 4, !dbg !57
  store i32 %19, i32* %i_318, align 4, !dbg !57
  %20 = load i32, i32* %.di0001p_348, align 4, !dbg !57
  %21 = load i32, i32* %.de0001p_347, align 4, !dbg !57
  %22 = load i32, i32* %.dX0001p_350, align 4, !dbg !57
  %23 = sub nsw i32 %21, %22, !dbg !57
  %24 = add nsw i32 %20, %23, !dbg !57
  %25 = load i32, i32* %.di0001p_348, align 4, !dbg !57
  %26 = sdiv i32 %24, %25, !dbg !57
  store i32 %26, i32* %.dY0001p_345, align 4, !dbg !57
  %27 = load i32, i32* %.dY0001p_345, align 4, !dbg !57
  %28 = icmp sle i32 %27, 0, !dbg !57
  br i1 %28, label %L.LB3_354, label %L.LB3_353, !dbg !57

L.LB3_353:                                        ; preds = %L.LB3_353, %L.LB3_459
  %29 = load i32, i32* %i_318, align 4, !dbg !59
  call void @llvm.dbg.value(metadata i32 %29, metadata !58, metadata !DIExpression()), !dbg !56
  %30 = bitcast %struct_drb092_0_* @_drb092_0_ to i32*, !dbg !59
  %31 = load i32, i32* %30, align 4, !dbg !59
  %32 = add nsw i32 %29, %31, !dbg !59
  %33 = bitcast %struct_drb092_0_* @_drb092_0_ to i32*, !dbg !59
  store i32 %32, i32* %33, align 4, !dbg !59
  %34 = load i32, i32* %.di0001p_348, align 4, !dbg !60
  %35 = load i32, i32* %i_318, align 4, !dbg !60
  call void @llvm.dbg.value(metadata i32 %35, metadata !58, metadata !DIExpression()), !dbg !56
  %36 = add nsw i32 %34, %35, !dbg !60
  store i32 %36, i32* %i_318, align 4, !dbg !60
  %37 = load i32, i32* %.dY0001p_345, align 4, !dbg !60
  %38 = sub nsw i32 %37, 1, !dbg !60
  store i32 %38, i32* %.dY0001p_345, align 4, !dbg !60
  %39 = load i32, i32* %.dY0001p_345, align 4, !dbg !60
  %40 = icmp sgt i32 %39, 0, !dbg !60
  br i1 %40, label %L.LB3_353, label %L.LB3_354, !dbg !60

L.LB3_354:                                        ; preds = %L.LB3_353, %L.LB3_459
  br label %L.LB3_344

L.LB3_344:                                        ; preds = %L.LB3_354, %L.LB3_316
  %41 = load i32, i32* %__gtid___nv_MAIN__F1L31_1__423, align 4, !dbg !60
  call void @__kmpc_for_static_fini(i64* null, i32 %41), !dbg !60
  br label %L.LB3_320

L.LB3_320:                                        ; preds = %L.LB3_344
  %42 = load i32, i32* %__gtid___nv_MAIN__F1L31_1__423, align 4, !dbg !61
  call void @__kmpc_barrier(i64* null, i32 %42), !dbg !61
  %43 = load i32, i32* %__gtid___nv_MAIN__F1L31_1__423, align 4, !dbg !62
  %44 = bitcast %struct__cs_unspc_* @__cs_unspc_ to i64*, !dbg !62
  call void @__kmpc_critical(i64* null, i32 %43, i64* %44), !dbg !62
  %45 = bitcast %struct_drb092_0_* @_drb092_0_ to i32*, !dbg !62
  %46 = load i32, i32* %45, align 4, !dbg !62
  %47 = bitcast i64* %__nv_MAIN__F1L31_1Arg2 to i32**, !dbg !62
  %48 = load i32*, i32** %47, align 8, !dbg !62
  %49 = load i32, i32* %48, align 4, !dbg !62
  %50 = add nsw i32 %46, %49, !dbg !62
  %51 = bitcast i64* %__nv_MAIN__F1L31_1Arg2 to i32**, !dbg !62
  %52 = load i32*, i32** %51, align 8, !dbg !62
  store i32 %50, i32* %52, align 4, !dbg !62
  %53 = load i32, i32* %__gtid___nv_MAIN__F1L31_1__423, align 4, !dbg !62
  %54 = bitcast %struct__cs_unspc_* @__cs_unspc_ to i64*, !dbg !62
  call void @__kmpc_end_critical(i64* null, i32 %53, i64* %54), !dbg !62
  br label %L.LB3_325

L.LB3_325:                                        ; preds = %L.LB3_320
  ret void, !dbg !56
}

declare void @__kmpc_end_critical(i64*, i32, i64*) #1

declare void @__kmpc_critical(i64*, i32, i64*) #1

declare void @__kmpc_barrier(i64*, i32) #1

declare void @__kmpc_for_static_fini(i64*, i32) #1

declare void @__kmpc_for_static_init_4(i64*, i32, i32, i64*, i64*, i64*, i64*, i32, i32) #1

declare void @_mp_ecs_nest(...) #1

declare signext i32 @f90io_ldw_end(...) #1

declare signext i32 @f90io_sc_i_ldw(...) #1

declare signext i32 @f90io_sc_ch_ldw(...) #1

declare signext i32 @f90io_print_init(...) #1

declare void @f90io_src_info03a(...) #1

declare void @_mp_bcs_nest(...) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

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

!llvm.module.flags = !{!26, !27}
!llvm.dbg.cu = !{!3}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "sum0", scope: !2, file: !4, type: !9, isLocal: false, isDefinition: true)
!2 = !DIModule(scope: !3, name: "drb092")
!3 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !4, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !6, imports: !21)
!4 = !DIFile(filename: "micro-benchmarks-fortran/DRB092-threadprivatemissing2-orig-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!5 = !{}
!6 = !{!0, !7, !10}
!7 = !DIGlobalVariableExpression(var: !8, expr: !DIExpression(DW_OP_plus_uconst, 4))
!8 = distinct !DIGlobalVariable(name: "sum1", scope: !2, file: !4, type: !9, isLocal: false, isDefinition: true)
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DIGlobalVariableExpression(var: !11, expr: !DIExpression())
!11 = distinct !DIGlobalVariable(name: "__cs_unspc", scope: !12, type: !17, isLocal: false, isDefinition: true)
!12 = distinct !DICommonBlock(scope: !13, declaration: !11, name: "__cs_unspc")
!13 = distinct !DISubprogram(name: "__nv_MAIN__F1L31_1", scope: !3, file: !4, line: 31, type: !14, scopeLine: 31, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !3)
!14 = !DISubroutineType(types: !15)
!15 = !{null, !9, !16, !16}
!16 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!17 = !DICompositeType(tag: DW_TAG_array_type, baseType: !18, size: 256, align: 8, elements: !19)
!18 = !DIBasicType(name: "byte", size: 8, align: 8, encoding: DW_ATE_signed)
!19 = !{!20}
!20 = !DISubrange(count: 32)
!21 = !{!22}
!22 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !23, entity: !2, file: !4, line: 21)
!23 = distinct !DISubprogram(name: "drb092_threadprivatemissing2_orig_yes", scope: !3, file: !4, line: 21, type: !24, scopeLine: 21, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !3)
!24 = !DISubroutineType(cc: DW_CC_program, types: !25)
!25 = !{null}
!26 = !{i32 2, !"Dwarf Version", i32 4}
!27 = !{i32 2, !"Debug Info Version", i32 3}
!28 = !DILocalVariable(name: "omp_sched_static", scope: !23, file: !4, type: !9)
!29 = !DILocation(line: 0, scope: !23)
!30 = !DILocalVariable(name: "omp_proc_bind_false", scope: !23, file: !4, type: !9)
!31 = !DILocalVariable(name: "omp_proc_bind_true", scope: !23, file: !4, type: !9)
!32 = !DILocalVariable(name: "omp_lock_hint_none", scope: !23, file: !4, type: !9)
!33 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !23, file: !4, type: !9)
!34 = !DILocation(line: 48, column: 1, scope: !23)
!35 = !DILocation(line: 21, column: 1, scope: !23)
!36 = !DILocalVariable(name: "sum", scope: !23, file: !4, type: !9)
!37 = !DILocation(line: 27, column: 1, scope: !23)
!38 = !DILocation(line: 28, column: 1, scope: !23)
!39 = !DILocation(line: 29, column: 1, scope: !23)
!40 = !DILocation(line: 31, column: 1, scope: !23)
!41 = !DILocation(line: 42, column: 1, scope: !23)
!42 = !DILocalVariable(name: "i", scope: !23, file: !4, type: !9)
!43 = !DILocation(line: 43, column: 1, scope: !23)
!44 = !DILocation(line: 44, column: 1, scope: !23)
!45 = !DILocation(line: 46, column: 1, scope: !23)
!46 = !DILocalVariable(scope: !23, file: !4, type: !9, flags: DIFlagArtificial)
!47 = !DILocalVariable(name: "__nv_MAIN__F1L31_1Arg0", arg: 1, scope: !13, file: !4, type: !9)
!48 = !DILocation(line: 0, scope: !13)
!49 = !DILocalVariable(name: "__nv_MAIN__F1L31_1Arg1", arg: 2, scope: !13, file: !4, type: !16)
!50 = !DILocalVariable(name: "__nv_MAIN__F1L31_1Arg2", arg: 3, scope: !13, file: !4, type: !16)
!51 = !DILocalVariable(name: "omp_sched_static", scope: !13, file: !4, type: !9)
!52 = !DILocalVariable(name: "omp_proc_bind_false", scope: !13, file: !4, type: !9)
!53 = !DILocalVariable(name: "omp_proc_bind_true", scope: !13, file: !4, type: !9)
!54 = !DILocalVariable(name: "omp_lock_hint_none", scope: !13, file: !4, type: !9)
!55 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !13, file: !4, type: !9)
!56 = !DILocation(line: 40, column: 1, scope: !13)
!57 = !DILocation(line: 33, column: 1, scope: !13)
!58 = !DILocalVariable(name: "i", scope: !13, file: !4, type: !9)
!59 = !DILocation(line: 34, column: 1, scope: !13)
!60 = !DILocation(line: 35, column: 1, scope: !13)
!61 = !DILocation(line: 36, column: 1, scope: !13)
!62 = !DILocation(line: 38, column: 1, scope: !13)

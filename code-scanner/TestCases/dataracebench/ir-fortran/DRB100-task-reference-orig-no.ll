; ModuleID = '/tmp/DRB100-task-reference-orig-no-ef12ee.ll'
source_filename = "/tmp/DRB100-task-reference-orig-no-ef12ee.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct_drb100_2_ = type <{ [8 x i8] }>
%struct_drb100_0_ = type <{ [144 x i8] }>
%astruct.dt75 = type <{ i8* }>

@.C283_drb100_gen_task_ = internal constant i32 0
@.C285_drb100_gen_task_ = internal constant i32 1
@.C285___nv_drb100_gen_task__F1L23_1 = internal constant i32 1
@.C340_MAIN_ = internal constant [13 x i8] c" not expected"
@.C339_MAIN_ = internal constant [3 x i8] c") ="
@.C306_MAIN_ = internal constant i32 14
@.C338_MAIN_ = internal constant [11 x i8] c"warning: a("
@.C335_MAIN_ = internal constant i32 6
@.C332_MAIN_ = internal constant [58 x i8] c"micro-benchmarks-fortran/DRB100-task-reference-orig-no.f95"
@.C334_MAIN_ = internal constant i32 47
@.C321_MAIN_ = internal constant i32 100
@.C285_MAIN_ = internal constant i32 1
@.C307_MAIN_ = internal constant i32 25
@.C348_MAIN_ = internal constant i64 4
@.C347_MAIN_ = internal constant i64 25
@.C284_MAIN_ = internal constant i64 0
@.C322_MAIN_ = internal constant i64 100
@.C310_MAIN_ = internal constant i64 12
@.C286_MAIN_ = internal constant i64 1
@.C309_MAIN_ = internal constant i64 11
@.C283_MAIN_ = internal constant i32 0
@.C321___nv_MAIN__F1L37_2 = internal constant i32 100
@.C285___nv_MAIN__F1L37_2 = internal constant i32 1
@_drb100_2_ = common global %struct_drb100_2_ zeroinitializer, align 64, !dbg !0
@_drb100_0_ = common global %struct_drb100_0_ zeroinitializer, align 64, !dbg !7, !dbg !13

; Function Attrs: noinline
define float @drb100_() #0 {
.L.entry:
  ret float undef
}

define void @drb100_gen_task_(i64* %i) #1 !dbg !26 {
L.entry:
  %__gtid_drb100_gen_task__334 = alloca i32, align 4
  %.s0000_326 = alloca i32, align 4
  %.z0307_325 = alloca i8*, align 8
  call void @llvm.dbg.declare(metadata i64* %i, metadata !29, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.value(metadata i32 1, metadata !31, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.value(metadata i32 0, metadata !32, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.value(metadata i32 1, metadata !33, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.value(metadata i32 0, metadata !34, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.value(metadata i32 1, metadata !35, metadata !DIExpression()), !dbg !30
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !36
  store i32 %0, i32* %__gtid_drb100_gen_task__334, align 4, !dbg !36
  br label %L.LB2_324

L.LB2_324:                                        ; preds = %L.entry
  store i32 1, i32* %.s0000_326, align 4, !dbg !37
  %1 = load i32, i32* %__gtid_drb100_gen_task__334, align 4, !dbg !38
  %2 = load i32, i32* %.s0000_326, align 4, !dbg !38
  %3 = bitcast void (i32, i64*)* @__nv_drb100_gen_task__F1L23_1_ to i64*, !dbg !38
  %4 = call i8* @__kmpc_omp_task_alloc(i64* null, i32 %1, i32 %2, i32 44, i32 32, i64* %3), !dbg !38
  store i8* %4, i8** %.z0307_325, align 8, !dbg !38
  %5 = bitcast i64* %i to i8*, !dbg !38
  %6 = load i8*, i8** %.z0307_325, align 8, !dbg !38
  %7 = bitcast i8* %6 to i8***, !dbg !38
  %8 = load i8**, i8*** %7, align 8, !dbg !38
  store i8* %5, i8** %8, align 8, !dbg !38
  %9 = bitcast %struct_drb100_0_* @_drb100_0_ to i8*, !dbg !38
  %10 = load i8*, i8** %.z0307_325, align 8, !dbg !38
  %11 = bitcast i8* %10 to i8**, !dbg !38
  %12 = load i8*, i8** %11, align 8, !dbg !38
  %13 = getelementptr i8, i8* %12, i64 8, !dbg !38
  %14 = bitcast i8* %13 to i8**, !dbg !38
  store i8* %9, i8** %14, align 8, !dbg !38
  %15 = bitcast %struct_drb100_0_* @_drb100_0_ to i8*, !dbg !38
  %16 = getelementptr i8, i8* %15, i64 16, !dbg !38
  %17 = load i8*, i8** %.z0307_325, align 8, !dbg !38
  %18 = bitcast i8* %17 to i8**, !dbg !38
  %19 = load i8*, i8** %18, align 8, !dbg !38
  %20 = getelementptr i8, i8* %19, i64 16, !dbg !38
  %21 = bitcast i8* %20 to i8**, !dbg !38
  store i8* %16, i8** %21, align 8, !dbg !38
  %22 = bitcast %struct_drb100_0_* @_drb100_0_ to i8*, !dbg !38
  %23 = load i8*, i8** %.z0307_325, align 8, !dbg !38
  %24 = bitcast i8* %23 to i8**, !dbg !38
  %25 = load i8*, i8** %24, align 8, !dbg !38
  %26 = getelementptr i8, i8* %25, i64 24, !dbg !38
  %27 = bitcast i8* %26 to i8**, !dbg !38
  store i8* %22, i8** %27, align 8, !dbg !38
  %28 = bitcast i64* %i to i32*, !dbg !37
  %29 = load i32, i32* %28, align 4, !dbg !37
  %30 = load i8*, i8** %.z0307_325, align 8, !dbg !37
  %31 = getelementptr i8, i8* %30, i64 40, !dbg !37
  %32 = bitcast i8* %31 to i32*, !dbg !37
  store i32 %29, i32* %32, align 4, !dbg !37
  %33 = load i32, i32* %__gtid_drb100_gen_task__334, align 4, !dbg !38
  %34 = load i8*, i8** %.z0307_325, align 8, !dbg !38
  %35 = bitcast i8* %34 to i64*, !dbg !38
  call void @__kmpc_omp_task(i64* null, i32 %33, i64* %35), !dbg !38
  br label %L.LB2_322

L.LB2_322:                                        ; preds = %L.LB2_324
  ret void, !dbg !36
}

define internal void @__nv_drb100_gen_task__F1L23_1_(i32 %__nv_drb100_gen_task__F1L23_1Arg0.arg, i64* %__nv_drb100_gen_task__F1L23_1Arg1) #1 !dbg !39 {
L.entry:
  %__nv_drb100_gen_task__F1L23_1Arg0.addr = alloca i32, align 4
  %.S0000_368 = alloca i8*, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_drb100_gen_task__F1L23_1Arg0.addr, metadata !42, metadata !DIExpression()), !dbg !43
  store i32 %__nv_drb100_gen_task__F1L23_1Arg0.arg, i32* %__nv_drb100_gen_task__F1L23_1Arg0.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_drb100_gen_task__F1L23_1Arg0.addr, metadata !44, metadata !DIExpression()), !dbg !43
  call void @llvm.dbg.declare(metadata i64* %__nv_drb100_gen_task__F1L23_1Arg1, metadata !45, metadata !DIExpression()), !dbg !43
  call void @llvm.dbg.value(metadata i32 1, metadata !46, metadata !DIExpression()), !dbg !43
  call void @llvm.dbg.value(metadata i32 0, metadata !47, metadata !DIExpression()), !dbg !43
  call void @llvm.dbg.value(metadata i32 1, metadata !48, metadata !DIExpression()), !dbg !43
  call void @llvm.dbg.value(metadata i32 0, metadata !49, metadata !DIExpression()), !dbg !43
  call void @llvm.dbg.value(metadata i32 1, metadata !50, metadata !DIExpression()), !dbg !43
  %0 = bitcast i64* %__nv_drb100_gen_task__F1L23_1Arg1 to i8**, !dbg !51
  %1 = load i8*, i8** %0, align 8, !dbg !51
  store i8* %1, i8** %.S0000_368, align 8, !dbg !51
  br label %L.LB3_372

L.LB3_372:                                        ; preds = %L.entry
  br label %L.LB3_317

L.LB3_317:                                        ; preds = %L.LB3_372
  %2 = bitcast i64* %__nv_drb100_gen_task__F1L23_1Arg1 to i8*, !dbg !52
  %3 = getelementptr i8, i8* %2, i64 40, !dbg !52
  %4 = bitcast i8* %3 to i64*, !dbg !52
  %5 = bitcast i64* %4 to i32*, !dbg !52
  %6 = load i32, i32* %5, align 4, !dbg !52
  %7 = add nsw i32 %6, 1, !dbg !52
  %8 = bitcast i64* %__nv_drb100_gen_task__F1L23_1Arg1 to i8*, !dbg !52
  %9 = getelementptr i8, i8* %8, i64 40, !dbg !52
  %10 = bitcast i8* %9 to i64*, !dbg !52
  %11 = bitcast i64* %10 to i32*, !dbg !52
  %12 = load i32, i32* %11, align 4, !dbg !52
  %13 = sext i32 %12 to i64, !dbg !52
  %14 = load i8*, i8** %.S0000_368, align 8, !dbg !52
  %15 = getelementptr i8, i8* %14, i64 16, !dbg !52
  %16 = bitcast i8* %15 to i8**, !dbg !52
  %17 = load i8*, i8** %16, align 8, !dbg !52
  %18 = getelementptr i8, i8* %17, i64 56, !dbg !52
  %19 = bitcast i8* %18 to i64*, !dbg !52
  %20 = load i64, i64* %19, align 8, !dbg !52
  %21 = add nsw i64 %13, %20, !dbg !52
  %22 = load i8*, i8** %.S0000_368, align 8, !dbg !52
  %23 = getelementptr i8, i8* %22, i64 24, !dbg !52
  %24 = bitcast i8* %23 to i8***, !dbg !52
  %25 = load i8**, i8*** %24, align 8, !dbg !52
  %26 = load i8*, i8** %25, align 8, !dbg !52
  %27 = getelementptr i8, i8* %26, i64 -4, !dbg !52
  %28 = bitcast i8* %27 to i32*, !dbg !52
  %29 = getelementptr i32, i32* %28, i64 %21, !dbg !52
  store i32 %7, i32* %29, align 4, !dbg !52
  br label %L.LB3_319

L.LB3_319:                                        ; preds = %L.LB3_317
  ret void, !dbg !53
}

define void @MAIN_() #1 !dbg !21 {
L.entry:
  %__gtid_MAIN__398 = alloca i32, align 4
  %.g0000_384 = alloca i64, align 8
  %i_320 = alloca i32, align 4
  %.uplevelArgPack0002_392 = alloca %astruct.dt75, align 8
  %.dY0002_361 = alloca i32, align 4
  %z__io_337 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !54, metadata !DIExpression()), !dbg !55
  call void @llvm.dbg.value(metadata i32 0, metadata !56, metadata !DIExpression()), !dbg !55
  call void @llvm.dbg.value(metadata i32 1, metadata !57, metadata !DIExpression()), !dbg !55
  call void @llvm.dbg.value(metadata i32 0, metadata !58, metadata !DIExpression()), !dbg !55
  call void @llvm.dbg.value(metadata i32 1, metadata !59, metadata !DIExpression()), !dbg !55
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !60
  store i32 %0, i32* %__gtid_MAIN__398, align 4, !dbg !60
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !61
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !61
  call void (i8*, ...) %2(i8* %1), !dbg !61
  br label %L.LB5_364

L.LB5_364:                                        ; preds = %L.entry
  %3 = bitcast %struct_drb100_0_* @_drb100_0_ to i8*, !dbg !62
  %4 = getelementptr i8, i8* %3, i64 96, !dbg !62
  %5 = bitcast i8* %4 to i64*, !dbg !62
  store i64 1, i64* %5, align 8, !dbg !62
  %6 = bitcast %struct_drb100_0_* @_drb100_0_ to i8*, !dbg !62
  %7 = getelementptr i8, i8* %6, i64 104, !dbg !62
  %8 = bitcast i8* %7 to i64*, !dbg !62
  store i64 100, i64* %8, align 8, !dbg !62
  %9 = bitcast %struct_drb100_0_* @_drb100_0_ to i8*, !dbg !62
  %10 = getelementptr i8, i8* %9, i64 104, !dbg !62
  %11 = bitcast i8* %10 to i64*, !dbg !62
  %12 = load i64, i64* %11, align 8, !dbg !62
  %13 = sub nsw i64 %12, 1, !dbg !62
  %14 = bitcast %struct_drb100_0_* @_drb100_0_ to i8*, !dbg !62
  %15 = getelementptr i8, i8* %14, i64 96, !dbg !62
  %16 = bitcast i8* %15 to i64*, !dbg !62
  %17 = load i64, i64* %16, align 8, !dbg !62
  %18 = add nsw i64 %13, %17, !dbg !62
  store i64 %18, i64* %.g0000_384, align 8, !dbg !62
  %19 = bitcast %struct_drb100_0_* @_drb100_0_ to i8*, !dbg !62
  %20 = getelementptr i8, i8* %19, i64 16, !dbg !62
  %21 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !62
  %22 = bitcast i64* @.C347_MAIN_ to i8*, !dbg !62
  %23 = bitcast i64* @.C348_MAIN_ to i8*, !dbg !62
  %24 = bitcast %struct_drb100_0_* @_drb100_0_ to i8*, !dbg !62
  %25 = getelementptr i8, i8* %24, i64 96, !dbg !62
  %26 = bitcast i64* %.g0000_384 to i8*, !dbg !62
  %27 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !62
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %27(i8* %20, i8* %21, i8* %22, i8* %23, i8* %25, i8* %26), !dbg !62
  %28 = bitcast %struct_drb100_0_* @_drb100_0_ to i8*, !dbg !62
  %29 = getelementptr i8, i8* %28, i64 16, !dbg !62
  %30 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !62
  call void (i8*, i32, ...) %30(i8* %29, i32 25), !dbg !62
  %31 = bitcast %struct_drb100_0_* @_drb100_0_ to i8*, !dbg !62
  %32 = getelementptr i8, i8* %31, i64 104, !dbg !62
  %33 = bitcast i8* %32 to i64*, !dbg !62
  %34 = load i64, i64* %33, align 8, !dbg !62
  %35 = sub nsw i64 %34, 1, !dbg !62
  %36 = bitcast %struct_drb100_0_* @_drb100_0_ to i8*, !dbg !62
  %37 = getelementptr i8, i8* %36, i64 96, !dbg !62
  %38 = bitcast i8* %37 to i64*, !dbg !62
  %39 = load i64, i64* %38, align 8, !dbg !62
  %40 = add nsw i64 %35, %39, !dbg !62
  %41 = bitcast %struct_drb100_0_* @_drb100_0_ to i8*, !dbg !62
  %42 = getelementptr i8, i8* %41, i64 96, !dbg !62
  %43 = bitcast i8* %42 to i64*, !dbg !62
  %44 = load i64, i64* %43, align 8, !dbg !62
  %45 = sub nsw i64 %44, 1, !dbg !62
  %46 = sub nsw i64 %40, %45, !dbg !62
  store i64 %46, i64* %.g0000_384, align 8, !dbg !62
  %47 = bitcast i64* %.g0000_384 to i8*, !dbg !62
  %48 = bitcast i64* @.C347_MAIN_ to i8*, !dbg !62
  %49 = bitcast i64* @.C348_MAIN_ to i8*, !dbg !62
  %50 = bitcast %struct_drb100_0_* @_drb100_0_ to i8*, !dbg !62
  %51 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !62
  %52 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !62
  %53 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !62
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %53(i8* %47, i8* %48, i8* %49, i8* null, i8* %50, i8* null, i8* %51, i8* %52, i8* null, i64 0), !dbg !62
  call void @llvm.dbg.declare(metadata i32* %i_320, metadata !63, metadata !DIExpression()), !dbg !55
  %54 = bitcast i32* %i_320 to i8*, !dbg !64
  %55 = bitcast %astruct.dt75* %.uplevelArgPack0002_392 to i8**, !dbg !64
  store i8* %54, i8** %55, align 8, !dbg !64
  br label %L.LB5_396, !dbg !64

L.LB5_396:                                        ; preds = %L.LB5_364
  %56 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L37_2_ to i64*, !dbg !64
  %57 = bitcast %astruct.dt75* %.uplevelArgPack0002_392 to i64*, !dbg !64
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %56, i64* %57), !dbg !64
  store i32 100, i32* %.dY0002_361, align 4, !dbg !65
  store i32 1, i32* %i_320, align 4, !dbg !65
  br label %L.LB5_359

L.LB5_359:                                        ; preds = %L.LB5_362, %L.LB5_396
  %58 = load i32, i32* %i_320, align 4, !dbg !66
  call void @llvm.dbg.value(metadata i32 %58, metadata !63, metadata !DIExpression()), !dbg !55
  %59 = sext i32 %58 to i64, !dbg !66
  %60 = bitcast %struct_drb100_0_* @_drb100_0_ to i8*, !dbg !66
  %61 = getelementptr i8, i8* %60, i64 72, !dbg !66
  %62 = bitcast i8* %61 to i64*, !dbg !66
  %63 = load i64, i64* %62, align 8, !dbg !66
  %64 = add nsw i64 %59, %63, !dbg !66
  %65 = bitcast %struct_drb100_0_* @_drb100_0_ to i8**, !dbg !66
  %66 = load i8*, i8** %65, align 8, !dbg !66
  %67 = getelementptr i8, i8* %66, i64 -4, !dbg !66
  %68 = bitcast i8* %67 to i32*, !dbg !66
  %69 = getelementptr i32, i32* %68, i64 %64, !dbg !66
  %70 = load i32, i32* %69, align 4, !dbg !66
  %71 = load i32, i32* %i_320, align 4, !dbg !66
  call void @llvm.dbg.value(metadata i32 %71, metadata !63, metadata !DIExpression()), !dbg !55
  %72 = add nsw i32 %71, 1, !dbg !66
  %73 = icmp eq i32 %70, %72, !dbg !66
  br i1 %73, label %L.LB5_362, label %L.LB5_436, !dbg !66

L.LB5_436:                                        ; preds = %L.LB5_359
  call void (...) @_mp_bcs_nest(), !dbg !67
  %74 = bitcast i32* @.C334_MAIN_ to i8*, !dbg !67
  %75 = bitcast [58 x i8]* @.C332_MAIN_ to i8*, !dbg !67
  %76 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !67
  call void (i8*, i8*, i64, ...) %76(i8* %74, i8* %75, i64 58), !dbg !67
  %77 = bitcast i32* @.C335_MAIN_ to i8*, !dbg !67
  %78 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !67
  %79 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !67
  %80 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !67
  %81 = call i32 (i8*, i8*, i8*, i8*, ...) %80(i8* %77, i8* null, i8* %78, i8* %79), !dbg !67
  call void @llvm.dbg.declare(metadata i32* %z__io_337, metadata !68, metadata !DIExpression()), !dbg !55
  store i32 %81, i32* %z__io_337, align 4, !dbg !67
  %82 = bitcast [11 x i8]* @.C338_MAIN_ to i8*, !dbg !67
  %83 = bitcast i32 (...)* @f90io_sc_ch_ldw to i32 (i8*, i32, i64, ...)*, !dbg !67
  %84 = call i32 (i8*, i32, i64, ...) %83(i8* %82, i32 14, i64 11), !dbg !67
  store i32 %84, i32* %z__io_337, align 4, !dbg !67
  %85 = load i32, i32* %i_320, align 4, !dbg !67
  call void @llvm.dbg.value(metadata i32 %85, metadata !63, metadata !DIExpression()), !dbg !55
  %86 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !67
  %87 = call i32 (i32, i32, ...) %86(i32 %85, i32 25), !dbg !67
  store i32 %87, i32* %z__io_337, align 4, !dbg !67
  %88 = bitcast [3 x i8]* @.C339_MAIN_ to i8*, !dbg !67
  %89 = bitcast i32 (...)* @f90io_sc_ch_ldw to i32 (i8*, i32, i64, ...)*, !dbg !67
  %90 = call i32 (i8*, i32, i64, ...) %89(i8* %88, i32 14, i64 3), !dbg !67
  store i32 %90, i32* %z__io_337, align 4, !dbg !67
  %91 = load i32, i32* %i_320, align 4, !dbg !67
  call void @llvm.dbg.value(metadata i32 %91, metadata !63, metadata !DIExpression()), !dbg !55
  %92 = sext i32 %91 to i64, !dbg !67
  %93 = bitcast %struct_drb100_0_* @_drb100_0_ to i8*, !dbg !67
  %94 = getelementptr i8, i8* %93, i64 72, !dbg !67
  %95 = bitcast i8* %94 to i64*, !dbg !67
  %96 = load i64, i64* %95, align 8, !dbg !67
  %97 = add nsw i64 %92, %96, !dbg !67
  %98 = bitcast %struct_drb100_0_* @_drb100_0_ to i8**, !dbg !67
  %99 = load i8*, i8** %98, align 8, !dbg !67
  %100 = getelementptr i8, i8* %99, i64 -4, !dbg !67
  %101 = bitcast i8* %100 to i32*, !dbg !67
  %102 = getelementptr i32, i32* %101, i64 %97, !dbg !67
  %103 = load i32, i32* %102, align 4, !dbg !67
  %104 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !67
  %105 = call i32 (i32, i32, ...) %104(i32 %103, i32 25), !dbg !67
  store i32 %105, i32* %z__io_337, align 4, !dbg !67
  %106 = bitcast [13 x i8]* @.C340_MAIN_ to i8*, !dbg !67
  %107 = bitcast i32 (...)* @f90io_sc_ch_ldw to i32 (i8*, i32, i64, ...)*, !dbg !67
  %108 = call i32 (i8*, i32, i64, ...) %107(i8* %106, i32 14, i64 13), !dbg !67
  store i32 %108, i32* %z__io_337, align 4, !dbg !67
  %109 = load i32, i32* %i_320, align 4, !dbg !67
  call void @llvm.dbg.value(metadata i32 %109, metadata !63, metadata !DIExpression()), !dbg !55
  %110 = add nsw i32 %109, 1, !dbg !67
  %111 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !67
  %112 = call i32 (i32, i32, ...) %111(i32 %110, i32 25), !dbg !67
  store i32 %112, i32* %z__io_337, align 4, !dbg !67
  %113 = call i32 (...) @f90io_ldw_end(), !dbg !67
  store i32 %113, i32* %z__io_337, align 4, !dbg !67
  call void (...) @_mp_ecs_nest(), !dbg !67
  br label %L.LB5_362

L.LB5_362:                                        ; preds = %L.LB5_436, %L.LB5_359
  %114 = load i32, i32* %i_320, align 4, !dbg !69
  call void @llvm.dbg.value(metadata i32 %114, metadata !63, metadata !DIExpression()), !dbg !55
  %115 = add nsw i32 %114, 1, !dbg !69
  store i32 %115, i32* %i_320, align 4, !dbg !69
  %116 = load i32, i32* %.dY0002_361, align 4, !dbg !69
  %117 = sub nsw i32 %116, 1, !dbg !69
  store i32 %117, i32* %.dY0002_361, align 4, !dbg !69
  %118 = load i32, i32* %.dY0002_361, align 4, !dbg !69
  %119 = icmp sgt i32 %118, 0, !dbg !69
  br i1 %119, label %L.LB5_359, label %L.LB5_437, !dbg !69

L.LB5_437:                                        ; preds = %L.LB5_362
  ret void, !dbg !60
}

define internal void @__nv_MAIN__F1L37_2_(i32* %__nv_MAIN__F1L37_2Arg0, i64* %__nv_MAIN__F1L37_2Arg1, i64* %__nv_MAIN__F1L37_2Arg2) #1 !dbg !70 {
L.entry:
  %__gtid___nv_MAIN__F1L37_2__447 = alloca i32, align 4
  %.s0001_442 = alloca i32, align 4
  %.s0002_443 = alloca i32, align 4
  %.dY0001p_358 = alloca i32, align 4
  %i_328 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L37_2Arg0, metadata !73, metadata !DIExpression()), !dbg !74
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L37_2Arg1, metadata !75, metadata !DIExpression()), !dbg !74
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L37_2Arg2, metadata !76, metadata !DIExpression()), !dbg !74
  call void @llvm.dbg.value(metadata i32 1, metadata !77, metadata !DIExpression()), !dbg !74
  call void @llvm.dbg.value(metadata i32 0, metadata !78, metadata !DIExpression()), !dbg !74
  call void @llvm.dbg.value(metadata i32 1, metadata !79, metadata !DIExpression()), !dbg !74
  call void @llvm.dbg.value(metadata i32 0, metadata !80, metadata !DIExpression()), !dbg !74
  call void @llvm.dbg.value(metadata i32 1, metadata !81, metadata !DIExpression()), !dbg !74
  %0 = load i32, i32* %__nv_MAIN__F1L37_2Arg0, align 4, !dbg !82
  store i32 %0, i32* %__gtid___nv_MAIN__F1L37_2__447, align 4, !dbg !82
  br label %L.LB6_441

L.LB6_441:                                        ; preds = %L.entry
  br label %L.LB6_325

L.LB6_325:                                        ; preds = %L.LB6_441
  store i32 -1, i32* %.s0001_442, align 4, !dbg !83
  store i32 0, i32* %.s0002_443, align 4, !dbg !83
  %1 = load i32, i32* %__gtid___nv_MAIN__F1L37_2__447, align 4, !dbg !83
  %2 = call i32 @__kmpc_single(i64* null, i32 %1), !dbg !83
  %3 = icmp eq i32 %2, 0, !dbg !83
  br i1 %3, label %L.LB6_355, label %L.LB6_327, !dbg !83

L.LB6_327:                                        ; preds = %L.LB6_325
  store i32 100, i32* %.dY0001p_358, align 4, !dbg !84
  call void @llvm.dbg.declare(metadata i32* %i_328, metadata !85, metadata !DIExpression()), !dbg !82
  store i32 1, i32* %i_328, align 4, !dbg !84
  br label %L.LB6_356

L.LB6_356:                                        ; preds = %L.LB6_356, %L.LB6_327
  %4 = bitcast i32* %i_328 to i64*, !dbg !86
  call void @drb100_gen_task_(i64* %4), !dbg !86
  %5 = load i32, i32* %i_328, align 4, !dbg !87
  call void @llvm.dbg.value(metadata i32 %5, metadata !85, metadata !DIExpression()), !dbg !82
  %6 = add nsw i32 %5, 1, !dbg !87
  store i32 %6, i32* %i_328, align 4, !dbg !87
  %7 = load i32, i32* %.dY0001p_358, align 4, !dbg !87
  %8 = sub nsw i32 %7, 1, !dbg !87
  store i32 %8, i32* %.dY0001p_358, align 4, !dbg !87
  %9 = load i32, i32* %.dY0001p_358, align 4, !dbg !87
  %10 = icmp sgt i32 %9, 0, !dbg !87
  br i1 %10, label %L.LB6_356, label %L.LB6_464, !dbg !87

L.LB6_464:                                        ; preds = %L.LB6_356
  %11 = load i32, i32* %__gtid___nv_MAIN__F1L37_2__447, align 4, !dbg !88
  store i32 %11, i32* %.s0001_442, align 4, !dbg !88
  store i32 1, i32* %.s0002_443, align 4, !dbg !88
  %12 = load i32, i32* %__gtid___nv_MAIN__F1L37_2__447, align 4, !dbg !88
  call void @__kmpc_end_single(i64* null, i32 %12), !dbg !88
  br label %L.LB6_355

L.LB6_355:                                        ; preds = %L.LB6_464, %L.LB6_325
  br label %L.LB6_329

L.LB6_329:                                        ; preds = %L.LB6_355
  %13 = load i32, i32* %__gtid___nv_MAIN__F1L37_2__447, align 4, !dbg !88
  call void @__kmpc_barrier(i64* null, i32 %13), !dbg !88
  br label %L.LB6_330

L.LB6_330:                                        ; preds = %L.LB6_329
  ret void, !dbg !82
}

declare void @__kmpc_barrier(i64*, i32) #1

declare void @__kmpc_end_single(i64*, i32) #1

declare signext i32 @__kmpc_single(i64*, i32) #1

declare void @_mp_ecs_nest(...) #1

declare signext i32 @f90io_ldw_end(...) #1

declare signext i32 @f90io_sc_i_ldw(...) #1

declare signext i32 @f90io_sc_ch_ldw(...) #1

declare signext i32 @f90io_print_init(...) #1

declare void @f90io_src_info03a(...) #1

declare void @_mp_bcs_nest(...) #1

declare void @f90_alloc04_chka_i8(...) #1

declare void @f90_set_intrin_type_i8(...) #1

declare void @f90_template1_i8(...) #1

declare void @fort_init(...) #1

declare void @__kmpc_fork_call(i64*, i32, i64*, i64*, ...) #1

declare void @__kmpc_end_serialized_parallel(i64*, i32) #1

declare void @__kmpc_serialized_parallel(i64*, i32) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

declare signext i32 @__kmpc_global_thread_num(i64*) #1

declare void @__kmpc_omp_task(i64*, i32, i64*) #1

declare i8* @__kmpc_omp_task_alloc(i64*, i32, i32, i32, i32, i64*) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { noinline }
attributes #1 = { "no-frame-pointer-elim-non-leaf" }
attributes #2 = { nounwind readnone speculatable }

!llvm.module.flags = !{!24, !25}
!llvm.dbg.cu = !{!3}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "z_b_0", scope: !2, file: !4, type: !16, isLocal: false, isDefinition: true)
!2 = !DIModule(scope: !3, name: "drb100")
!3 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !4, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !6, imports: !19)
!4 = !DIFile(filename: "micro-benchmarks-fortran/DRB100-task-reference-orig-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!5 = !{}
!6 = !{!7, !13, !0}
!7 = !DIGlobalVariableExpression(var: !8, expr: !DIExpression(DW_OP_deref))
!8 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !4, type: !9, isLocal: false, isDefinition: true)
!9 = !DICompositeType(tag: DW_TAG_array_type, baseType: !10, size: 32, align: 32, elements: !11)
!10 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!11 = !{!12}
!12 = !DISubrange(count: 0, lowerBound: 1)
!13 = !DIGlobalVariableExpression(var: !14, expr: !DIExpression(DW_OP_plus_uconst, 16))
!14 = distinct !DIGlobalVariable(name: "a$sd", scope: !2, file: !4, type: !15, isLocal: false, isDefinition: true)
!15 = !DICompositeType(tag: DW_TAG_array_type, baseType: !16, size: 1024, align: 64, elements: !17)
!16 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!17 = !{!18}
!18 = !DISubrange(count: 16, lowerBound: 1)
!19 = !{!20}
!20 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !21, entity: !2, file: !4, line: 29)
!21 = distinct !DISubprogram(name: "drb100_task_reference_orig_no", scope: !3, file: !4, line: 29, type: !22, scopeLine: 29, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !3)
!22 = !DISubroutineType(cc: DW_CC_program, types: !23)
!23 = !{null}
!24 = !{i32 2, !"Dwarf Version", i32 4}
!25 = !{i32 2, !"Debug Info Version", i32 3}
!26 = distinct !DISubprogram(name: "gen_task", scope: !2, file: !4, line: 19, type: !27, scopeLine: 19, spFlags: DISPFlagDefinition, unit: !3)
!27 = !DISubroutineType(types: !28)
!28 = !{null, !10}
!29 = !DILocalVariable(name: "i", arg: 1, scope: !26, file: !4, type: !10)
!30 = !DILocation(line: 0, scope: !26)
!31 = !DILocalVariable(name: "omp_sched_static", scope: !26, file: !4, type: !10)
!32 = !DILocalVariable(name: "omp_proc_bind_false", scope: !26, file: !4, type: !10)
!33 = !DILocalVariable(name: "omp_proc_bind_true", scope: !26, file: !4, type: !10)
!34 = !DILocalVariable(name: "omp_lock_hint_none", scope: !26, file: !4, type: !10)
!35 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !26, file: !4, type: !10)
!36 = !DILocation(line: 26, column: 1, scope: !26)
!37 = !DILocation(line: 23, column: 1, scope: !26)
!38 = !DILocation(line: 25, column: 1, scope: !26)
!39 = distinct !DISubprogram(name: "__nv_drb100_gen_task__F1L23_1", scope: !3, file: !4, line: 23, type: !40, scopeLine: 23, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !3)
!40 = !DISubroutineType(types: !41)
!41 = !{null, !10, !16}
!42 = !DILocalVariable(name: "__nv_drb100_gen_task__F1L23_1Arg0", scope: !39, file: !4, type: !10)
!43 = !DILocation(line: 0, scope: !39)
!44 = !DILocalVariable(name: "__nv_drb100_gen_task__F1L23_1Arg0", arg: 1, scope: !39, file: !4, type: !10)
!45 = !DILocalVariable(name: "__nv_drb100_gen_task__F1L23_1Arg1", arg: 2, scope: !39, file: !4, type: !16)
!46 = !DILocalVariable(name: "omp_sched_static", scope: !39, file: !4, type: !10)
!47 = !DILocalVariable(name: "omp_proc_bind_false", scope: !39, file: !4, type: !10)
!48 = !DILocalVariable(name: "omp_proc_bind_true", scope: !39, file: !4, type: !10)
!49 = !DILocalVariable(name: "omp_lock_hint_none", scope: !39, file: !4, type: !10)
!50 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !39, file: !4, type: !10)
!51 = !DILocation(line: 23, column: 1, scope: !39)
!52 = !DILocation(line: 24, column: 1, scope: !39)
!53 = !DILocation(line: 25, column: 1, scope: !39)
!54 = !DILocalVariable(name: "omp_sched_static", scope: !21, file: !4, type: !10)
!55 = !DILocation(line: 0, scope: !21)
!56 = !DILocalVariable(name: "omp_proc_bind_false", scope: !21, file: !4, type: !10)
!57 = !DILocalVariable(name: "omp_proc_bind_true", scope: !21, file: !4, type: !10)
!58 = !DILocalVariable(name: "omp_lock_hint_none", scope: !21, file: !4, type: !10)
!59 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !21, file: !4, type: !10)
!60 = !DILocation(line: 51, column: 1, scope: !21)
!61 = !DILocation(line: 29, column: 1, scope: !21)
!62 = !DILocation(line: 35, column: 1, scope: !21)
!63 = !DILocalVariable(name: "i", scope: !21, file: !4, type: !10)
!64 = !DILocation(line: 37, column: 1, scope: !21)
!65 = !DILocation(line: 45, column: 1, scope: !21)
!66 = !DILocation(line: 46, column: 1, scope: !21)
!67 = !DILocation(line: 47, column: 1, scope: !21)
!68 = !DILocalVariable(scope: !21, file: !4, type: !10, flags: DIFlagArtificial)
!69 = !DILocation(line: 50, column: 1, scope: !21)
!70 = distinct !DISubprogram(name: "__nv_MAIN__F1L37_2", scope: !3, file: !4, line: 37, type: !71, scopeLine: 37, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !3)
!71 = !DISubroutineType(types: !72)
!72 = !{null, !10, !16, !16}
!73 = !DILocalVariable(name: "__nv_MAIN__F1L37_2Arg0", arg: 1, scope: !70, file: !4, type: !10)
!74 = !DILocation(line: 0, scope: !70)
!75 = !DILocalVariable(name: "__nv_MAIN__F1L37_2Arg1", arg: 2, scope: !70, file: !4, type: !16)
!76 = !DILocalVariable(name: "__nv_MAIN__F1L37_2Arg2", arg: 3, scope: !70, file: !4, type: !16)
!77 = !DILocalVariable(name: "omp_sched_static", scope: !70, file: !4, type: !10)
!78 = !DILocalVariable(name: "omp_proc_bind_false", scope: !70, file: !4, type: !10)
!79 = !DILocalVariable(name: "omp_proc_bind_true", scope: !70, file: !4, type: !10)
!80 = !DILocalVariable(name: "omp_lock_hint_none", scope: !70, file: !4, type: !10)
!81 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !70, file: !4, type: !10)
!82 = !DILocation(line: 43, column: 1, scope: !70)
!83 = !DILocation(line: 38, column: 1, scope: !70)
!84 = !DILocation(line: 39, column: 1, scope: !70)
!85 = !DILocalVariable(name: "i", scope: !70, file: !4, type: !10)
!86 = !DILocation(line: 40, column: 1, scope: !70)
!87 = !DILocation(line: 41, column: 1, scope: !70)
!88 = !DILocation(line: 42, column: 1, scope: !70)

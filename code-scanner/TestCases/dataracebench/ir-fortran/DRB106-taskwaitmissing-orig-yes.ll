; ModuleID = '/tmp/DRB106-taskwaitmissing-orig-yes-3fe2cd.ll'
source_filename = "/tmp/DRB106-taskwaitmissing-orig-yes-3fe2cd.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct_drb106_0_ = type <{ [4 x i8] }>
%astruct.dt65 = type <{ i8* }>

@.C283_drb106_fib_ = internal constant i32 0
@.C285_drb106_fib_ = internal constant i32 1
@.C305_drb106_fib_ = internal constant i32 2
@.C285___nv_drb106_fib__F1L28_1 = internal constant i32 1
@.C305___nv_drb106_fib__F1L31_2 = internal constant i32 2
@.C330_MAIN_ = internal constant [2 x i8] c" ="
@.C307_MAIN_ = internal constant i32 25
@.C306_MAIN_ = internal constant i32 14
@.C329_MAIN_ = internal constant [8 x i8] c"Fib for "
@.C284_MAIN_ = internal constant i64 0
@.C326_MAIN_ = internal constant i32 6
@.C323_MAIN_ = internal constant [60 x i8] c"micro-benchmarks-fortran/DRB106-taskwaitmissing-orig-yes.f95"
@.C325_MAIN_ = internal constant i32 54
@.C285_MAIN_ = internal constant i32 1
@.C314_MAIN_ = internal constant i32 30
@.C283_MAIN_ = internal constant i32 0
@.C285___nv_MAIN__F1L48_3 = internal constant i32 1
@_drb106_0_ = common global %struct_drb106_0_ zeroinitializer, align 64, !dbg !0

; Function Attrs: noinline
define float @drb106_() #0 {
.L.entry:
  ret float undef
}

define signext i32 @drb106_fib_(i64* %n) #1 !dbg !15 {
L.entry:
  %__gtid_drb106_fib__344 = alloca i32, align 4
  %r_301 = alloca i32, align 4
  %.s0000_337 = alloca i32, align 4
  %.z0302_336 = alloca i8*, align 8
  %i_314 = alloca i32, align 4
  %.s0001_367 = alloca i32, align 4
  %.z0302_366 = alloca i8*, align 8
  %j_315 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i64* %n, metadata !19, metadata !DIExpression()), !dbg !20
  call void @llvm.dbg.value(metadata i32 1, metadata !21, metadata !DIExpression()), !dbg !20
  call void @llvm.dbg.value(metadata i32 2, metadata !22, metadata !DIExpression()), !dbg !20
  call void @llvm.dbg.value(metadata i32 0, metadata !23, metadata !DIExpression()), !dbg !20
  call void @llvm.dbg.value(metadata i32 1, metadata !24, metadata !DIExpression()), !dbg !20
  call void @llvm.dbg.value(metadata i32 2, metadata !25, metadata !DIExpression()), !dbg !20
  call void @llvm.dbg.value(metadata i32 0, metadata !26, metadata !DIExpression()), !dbg !20
  call void @llvm.dbg.value(metadata i32 1, metadata !27, metadata !DIExpression()), !dbg !20
  call void @llvm.dbg.value(metadata i32 2, metadata !28, metadata !DIExpression()), !dbg !20
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !29
  store i32 %0, i32* %__gtid_drb106_fib__344, align 4, !dbg !29
  br label %L.LB2_333

L.LB2_333:                                        ; preds = %L.entry
  %1 = bitcast i64* %n to i32*, !dbg !30
  %2 = load i32, i32* %1, align 4, !dbg !30
  %3 = icmp sge i32 %2, 2, !dbg !30
  br i1 %3, label %L.LB2_328, label %L.LB2_387, !dbg !30

L.LB2_387:                                        ; preds = %L.LB2_333
  %4 = bitcast i64* %n to i32*, !dbg !31
  %5 = load i32, i32* %4, align 4, !dbg !31
  call void @llvm.dbg.declare(metadata i32* %r_301, metadata !32, metadata !DIExpression()), !dbg !20
  store i32 %5, i32* %r_301, align 4, !dbg !31
  br label %L.LB2_329, !dbg !33

L.LB2_328:                                        ; preds = %L.LB2_333
  store i32 1, i32* %.s0000_337, align 4, !dbg !34
  %6 = load i32, i32* %__gtid_drb106_fib__344, align 4, !dbg !35
  %7 = load i32, i32* %.s0000_337, align 4, !dbg !35
  %8 = bitcast void (i32, i64*)* @__nv_drb106_fib__F1L28_1_ to i64*, !dbg !35
  %9 = call i8* @__kmpc_omp_task_alloc(i64* null, i32 %6, i32 %7, i32 44, i32 16, i64* %8), !dbg !35
  store i8* %9, i8** %.z0302_336, align 8, !dbg !35
  %10 = bitcast i64* %n to i8*, !dbg !35
  %11 = load i8*, i8** %.z0302_336, align 8, !dbg !35
  %12 = bitcast i8* %11 to i8***, !dbg !35
  %13 = load i8**, i8*** %12, align 8, !dbg !35
  store i8* %10, i8** %13, align 8, !dbg !35
  call void @llvm.dbg.declare(metadata i32* %i_314, metadata !36, metadata !DIExpression()), !dbg !20
  %14 = bitcast i32* %i_314 to i8*, !dbg !35
  %15 = load i8*, i8** %.z0302_336, align 8, !dbg !35
  %16 = bitcast i8* %15 to i8**, !dbg !35
  %17 = load i8*, i8** %16, align 8, !dbg !35
  %18 = getelementptr i8, i8* %17, i64 8, !dbg !35
  %19 = bitcast i8* %18 to i8**, !dbg !35
  store i8* %14, i8** %19, align 8, !dbg !35
  %20 = bitcast i64* %n to i32*, !dbg !34
  %21 = load i32, i32* %20, align 4, !dbg !34
  %22 = load i8*, i8** %.z0302_336, align 8, !dbg !34
  %23 = getelementptr i8, i8* %22, i64 40, !dbg !34
  %24 = bitcast i8* %23 to i32*, !dbg !34
  store i32 %21, i32* %24, align 4, !dbg !34
  %25 = load i32, i32* %__gtid_drb106_fib__344, align 4, !dbg !35
  %26 = load i8*, i8** %.z0302_336, align 8, !dbg !35
  %27 = bitcast i8* %26 to i64*, !dbg !35
  call void @__kmpc_omp_task(i64* null, i32 %25, i64* %27), !dbg !35
  br label %L.LB2_330

L.LB2_330:                                        ; preds = %L.LB2_328
  store i32 1, i32* %.s0001_367, align 4, !dbg !37
  %28 = load i32, i32* %__gtid_drb106_fib__344, align 4, !dbg !38
  %29 = load i32, i32* %.s0001_367, align 4, !dbg !38
  %30 = bitcast void (i32, i64*)* @__nv_drb106_fib__F1L31_2_ to i64*, !dbg !38
  %31 = call i8* @__kmpc_omp_task_alloc(i64* null, i32 %28, i32 %29, i32 44, i32 16, i64* %30), !dbg !38
  store i8* %31, i8** %.z0302_366, align 8, !dbg !38
  %32 = bitcast i64* %n to i8*, !dbg !38
  %33 = load i8*, i8** %.z0302_366, align 8, !dbg !38
  %34 = bitcast i8* %33 to i8***, !dbg !38
  %35 = load i8**, i8*** %34, align 8, !dbg !38
  store i8* %32, i8** %35, align 8, !dbg !38
  call void @llvm.dbg.declare(metadata i32* %j_315, metadata !39, metadata !DIExpression()), !dbg !20
  %36 = bitcast i32* %j_315 to i8*, !dbg !38
  %37 = load i8*, i8** %.z0302_366, align 8, !dbg !38
  %38 = bitcast i8* %37 to i8**, !dbg !38
  %39 = load i8*, i8** %38, align 8, !dbg !38
  %40 = getelementptr i8, i8* %39, i64 8, !dbg !38
  %41 = bitcast i8* %40 to i8**, !dbg !38
  store i8* %36, i8** %41, align 8, !dbg !38
  %42 = bitcast i64* %n to i32*, !dbg !37
  %43 = load i32, i32* %42, align 4, !dbg !37
  %44 = load i8*, i8** %.z0302_366, align 8, !dbg !37
  %45 = getelementptr i8, i8* %44, i64 40, !dbg !37
  %46 = bitcast i8* %45 to i32*, !dbg !37
  store i32 %43, i32* %46, align 4, !dbg !37
  %47 = load i32, i32* %__gtid_drb106_fib__344, align 4, !dbg !38
  %48 = load i8*, i8** %.z0302_366, align 8, !dbg !38
  %49 = bitcast i8* %48 to i64*, !dbg !38
  call void @__kmpc_omp_task(i64* null, i32 %47, i64* %49), !dbg !38
  br label %L.LB2_331

L.LB2_331:                                        ; preds = %L.LB2_330
  %50 = load i32, i32* %j_315, align 4, !dbg !40
  call void @llvm.dbg.value(metadata i32 %50, metadata !39, metadata !DIExpression()), !dbg !20
  %51 = load i32, i32* %i_314, align 4, !dbg !40
  call void @llvm.dbg.value(metadata i32 %51, metadata !36, metadata !DIExpression()), !dbg !20
  %52 = add nsw i32 %50, %51, !dbg !40
  store i32 %52, i32* %r_301, align 4, !dbg !40
  br label %L.LB2_329

L.LB2_329:                                        ; preds = %L.LB2_331, %L.LB2_387
  %53 = load i32, i32* %__gtid_drb106_fib__344, align 4, !dbg !41
  %54 = call i32 @__kmpc_omp_taskwait(i64* null, i32 %53), !dbg !41
  %55 = load i32, i32* %r_301, align 4, !dbg !29
  call void @llvm.dbg.value(metadata i32 %55, metadata !32, metadata !DIExpression()), !dbg !20
  ret i32 %55, !dbg !29
}

define internal void @__nv_drb106_fib__F1L28_1_(i32 %__nv_drb106_fib__F1L28_1Arg0.arg, i64* %__nv_drb106_fib__F1L28_1Arg1) #1 !dbg !42 {
L.entry:
  %__nv_drb106_fib__F1L28_1Arg0.addr = alloca i32, align 4
  %.S0000_389 = alloca i8*, align 8
  %.D0000_396 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_drb106_fib__F1L28_1Arg0.addr, metadata !46, metadata !DIExpression()), !dbg !47
  store i32 %__nv_drb106_fib__F1L28_1Arg0.arg, i32* %__nv_drb106_fib__F1L28_1Arg0.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_drb106_fib__F1L28_1Arg0.addr, metadata !48, metadata !DIExpression()), !dbg !47
  call void @llvm.dbg.declare(metadata i64* %__nv_drb106_fib__F1L28_1Arg1, metadata !49, metadata !DIExpression()), !dbg !47
  call void @llvm.dbg.value(metadata i32 1, metadata !50, metadata !DIExpression()), !dbg !47
  call void @llvm.dbg.value(metadata i32 2, metadata !51, metadata !DIExpression()), !dbg !47
  call void @llvm.dbg.value(metadata i32 0, metadata !52, metadata !DIExpression()), !dbg !47
  call void @llvm.dbg.value(metadata i32 1, metadata !53, metadata !DIExpression()), !dbg !47
  call void @llvm.dbg.value(metadata i32 2, metadata !54, metadata !DIExpression()), !dbg !47
  call void @llvm.dbg.value(metadata i32 0, metadata !55, metadata !DIExpression()), !dbg !47
  call void @llvm.dbg.value(metadata i32 1, metadata !56, metadata !DIExpression()), !dbg !47
  call void @llvm.dbg.value(metadata i32 2, metadata !57, metadata !DIExpression()), !dbg !47
  %0 = bitcast i64* %__nv_drb106_fib__F1L28_1Arg1 to i8**, !dbg !58
  %1 = load i8*, i8** %0, align 8, !dbg !58
  store i8* %1, i8** %.S0000_389, align 8, !dbg !58
  br label %L.LB3_393

L.LB3_393:                                        ; preds = %L.entry
  br label %L.LB3_318

L.LB3_318:                                        ; preds = %L.LB3_393
  %2 = bitcast i64* %__nv_drb106_fib__F1L28_1Arg1 to i8*, !dbg !59
  %3 = getelementptr i8, i8* %2, i64 40, !dbg !59
  %4 = bitcast i8* %3 to i64*, !dbg !59
  %5 = bitcast i64* %4 to i32*, !dbg !59
  %6 = load i32, i32* %5, align 4, !dbg !59
  %7 = sub nsw i32 %6, 1, !dbg !59
  store i32 %7, i32* %.D0000_396, align 4, !dbg !59
  %8 = bitcast i32* %.D0000_396 to i64*, !dbg !59
  %9 = call i32 @drb106_fib_(i64* %8), !dbg !59
  %10 = load i8*, i8** %.S0000_389, align 8, !dbg !59
  %11 = getelementptr i8, i8* %10, i64 8, !dbg !59
  %12 = bitcast i8* %11 to i32**, !dbg !59
  %13 = load i32*, i32** %12, align 8, !dbg !59
  store i32 %9, i32* %13, align 4, !dbg !59
  br label %L.LB3_320

L.LB3_320:                                        ; preds = %L.LB3_318
  ret void, !dbg !60
}

define internal void @__nv_drb106_fib__F1L31_2_(i32 %__nv_drb106_fib__F1L31_2Arg0.arg, i64* %__nv_drb106_fib__F1L31_2Arg1) #1 !dbg !61 {
L.entry:
  %__nv_drb106_fib__F1L31_2Arg0.addr = alloca i32, align 4
  %.S0000_389 = alloca i8*, align 8
  %.D0001_406 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_drb106_fib__F1L31_2Arg0.addr, metadata !62, metadata !DIExpression()), !dbg !63
  store i32 %__nv_drb106_fib__F1L31_2Arg0.arg, i32* %__nv_drb106_fib__F1L31_2Arg0.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_drb106_fib__F1L31_2Arg0.addr, metadata !64, metadata !DIExpression()), !dbg !63
  call void @llvm.dbg.declare(metadata i64* %__nv_drb106_fib__F1L31_2Arg1, metadata !65, metadata !DIExpression()), !dbg !63
  call void @llvm.dbg.value(metadata i32 1, metadata !66, metadata !DIExpression()), !dbg !63
  call void @llvm.dbg.value(metadata i32 2, metadata !67, metadata !DIExpression()), !dbg !63
  call void @llvm.dbg.value(metadata i32 0, metadata !68, metadata !DIExpression()), !dbg !63
  call void @llvm.dbg.value(metadata i32 1, metadata !69, metadata !DIExpression()), !dbg !63
  call void @llvm.dbg.value(metadata i32 2, metadata !70, metadata !DIExpression()), !dbg !63
  call void @llvm.dbg.value(metadata i32 0, metadata !71, metadata !DIExpression()), !dbg !63
  call void @llvm.dbg.value(metadata i32 1, metadata !72, metadata !DIExpression()), !dbg !63
  call void @llvm.dbg.value(metadata i32 2, metadata !73, metadata !DIExpression()), !dbg !63
  %0 = bitcast i64* %__nv_drb106_fib__F1L31_2Arg1 to i8**, !dbg !74
  %1 = load i8*, i8** %0, align 8, !dbg !74
  store i8* %1, i8** %.S0000_389, align 8, !dbg !74
  br label %L.LB4_403

L.LB4_403:                                        ; preds = %L.entry
  br label %L.LB4_323

L.LB4_323:                                        ; preds = %L.LB4_403
  %2 = bitcast i64* %__nv_drb106_fib__F1L31_2Arg1 to i8*, !dbg !75
  %3 = getelementptr i8, i8* %2, i64 40, !dbg !75
  %4 = bitcast i8* %3 to i64*, !dbg !75
  %5 = bitcast i64* %4 to i32*, !dbg !75
  %6 = load i32, i32* %5, align 4, !dbg !75
  %7 = sub nsw i32 %6, 2, !dbg !75
  store i32 %7, i32* %.D0001_406, align 4, !dbg !75
  %8 = bitcast i32* %.D0001_406 to i64*, !dbg !75
  %9 = call i32 @drb106_fib_(i64* %8), !dbg !75
  %10 = load i8*, i8** %.S0000_389, align 8, !dbg !75
  %11 = getelementptr i8, i8* %10, i64 8, !dbg !75
  %12 = bitcast i8* %11 to i32**, !dbg !75
  %13 = load i32*, i32** %12, align 8, !dbg !75
  store i32 %9, i32* %13, align 4, !dbg !75
  br label %L.LB4_325

L.LB4_325:                                        ; preds = %L.LB4_323
  ret void, !dbg !76
}

define void @MAIN_() #1 !dbg !9 {
L.entry:
  %__gtid_MAIN__353 = alloca i32, align 4
  %result_313 = alloca i32, align 4
  %.uplevelArgPack0003_347 = alloca %astruct.dt65, align 8
  %z__io_328 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !77, metadata !DIExpression()), !dbg !78
  call void @llvm.dbg.value(metadata i32 0, metadata !79, metadata !DIExpression()), !dbg !78
  call void @llvm.dbg.value(metadata i32 1, metadata !80, metadata !DIExpression()), !dbg !78
  call void @llvm.dbg.value(metadata i32 0, metadata !81, metadata !DIExpression()), !dbg !78
  call void @llvm.dbg.value(metadata i32 1, metadata !82, metadata !DIExpression()), !dbg !78
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !83
  store i32 %0, i32* %__gtid_MAIN__353, align 4, !dbg !83
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !84
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !84
  call void (i8*, ...) %2(i8* %1), !dbg !84
  br label %L.LB6_341

L.LB6_341:                                        ; preds = %L.entry
  %3 = bitcast %struct_drb106_0_* @_drb106_0_ to i32*, !dbg !85
  store i32 30, i32* %3, align 4, !dbg !85
  call void @llvm.dbg.declare(metadata i32* %result_313, metadata !86, metadata !DIExpression()), !dbg !78
  %4 = bitcast i32* %result_313 to i8*, !dbg !87
  %5 = bitcast %astruct.dt65* %.uplevelArgPack0003_347 to i8**, !dbg !87
  store i8* %4, i8** %5, align 8, !dbg !87
  br label %L.LB6_351, !dbg !87

L.LB6_351:                                        ; preds = %L.LB6_341
  %6 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L48_3_ to i64*, !dbg !87
  %7 = bitcast %astruct.dt65* %.uplevelArgPack0003_347 to i64*, !dbg !87
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %6, i64* %7), !dbg !87
  call void (...) @_mp_bcs_nest(), !dbg !88
  %8 = bitcast i32* @.C325_MAIN_ to i8*, !dbg !88
  %9 = bitcast [60 x i8]* @.C323_MAIN_ to i8*, !dbg !88
  %10 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !88
  call void (i8*, i8*, i64, ...) %10(i8* %8, i8* %9, i64 60), !dbg !88
  %11 = bitcast i32* @.C326_MAIN_ to i8*, !dbg !88
  %12 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !88
  %13 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !88
  %14 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !88
  %15 = call i32 (i8*, i8*, i8*, i8*, ...) %14(i8* %11, i8* null, i8* %12, i8* %13), !dbg !88
  call void @llvm.dbg.declare(metadata i32* %z__io_328, metadata !89, metadata !DIExpression()), !dbg !78
  store i32 %15, i32* %z__io_328, align 4, !dbg !88
  %16 = bitcast [8 x i8]* @.C329_MAIN_ to i8*, !dbg !88
  %17 = bitcast i32 (...)* @f90io_sc_ch_ldw to i32 (i8*, i32, i64, ...)*, !dbg !88
  %18 = call i32 (i8*, i32, i64, ...) %17(i8* %16, i32 14, i64 8), !dbg !88
  store i32 %18, i32* %z__io_328, align 4, !dbg !88
  %19 = bitcast %struct_drb106_0_* @_drb106_0_ to i32*, !dbg !88
  %20 = load i32, i32* %19, align 4, !dbg !88
  %21 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !88
  %22 = call i32 (i32, i32, ...) %21(i32 %20, i32 25), !dbg !88
  store i32 %22, i32* %z__io_328, align 4, !dbg !88
  %23 = bitcast [2 x i8]* @.C330_MAIN_ to i8*, !dbg !88
  %24 = bitcast i32 (...)* @f90io_sc_ch_ldw to i32 (i8*, i32, i64, ...)*, !dbg !88
  %25 = call i32 (i8*, i32, i64, ...) %24(i8* %23, i32 14, i64 2), !dbg !88
  store i32 %25, i32* %z__io_328, align 4, !dbg !88
  %26 = load i32, i32* %result_313, align 4, !dbg !88
  call void @llvm.dbg.value(metadata i32 %26, metadata !86, metadata !DIExpression()), !dbg !78
  %27 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !88
  %28 = call i32 (i32, i32, ...) %27(i32 %26, i32 25), !dbg !88
  store i32 %28, i32* %z__io_328, align 4, !dbg !88
  %29 = call i32 (...) @f90io_ldw_end(), !dbg !88
  store i32 %29, i32* %z__io_328, align 4, !dbg !88
  call void (...) @_mp_ecs_nest(), !dbg !88
  ret void, !dbg !83
}

define internal void @__nv_MAIN__F1L48_3_(i32* %__nv_MAIN__F1L48_3Arg0, i64* %__nv_MAIN__F1L48_3Arg1, i64* %__nv_MAIN__F1L48_3Arg2) #1 !dbg !90 {
L.entry:
  %__gtid___nv_MAIN__F1L48_3__393 = alloca i32, align 4
  %.s0002_388 = alloca i32, align 4
  %.s0003_389 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L48_3Arg0, metadata !93, metadata !DIExpression()), !dbg !94
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L48_3Arg1, metadata !95, metadata !DIExpression()), !dbg !94
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L48_3Arg2, metadata !96, metadata !DIExpression()), !dbg !94
  call void @llvm.dbg.value(metadata i32 1, metadata !97, metadata !DIExpression()), !dbg !94
  call void @llvm.dbg.value(metadata i32 0, metadata !98, metadata !DIExpression()), !dbg !94
  call void @llvm.dbg.value(metadata i32 1, metadata !99, metadata !DIExpression()), !dbg !94
  call void @llvm.dbg.value(metadata i32 0, metadata !100, metadata !DIExpression()), !dbg !94
  call void @llvm.dbg.value(metadata i32 1, metadata !101, metadata !DIExpression()), !dbg !94
  %0 = load i32, i32* %__nv_MAIN__F1L48_3Arg0, align 4, !dbg !102
  store i32 %0, i32* %__gtid___nv_MAIN__F1L48_3__393, align 4, !dbg !102
  br label %L.LB7_387

L.LB7_387:                                        ; preds = %L.entry
  br label %L.LB7_317

L.LB7_317:                                        ; preds = %L.LB7_387
  store i32 -1, i32* %.s0002_388, align 4, !dbg !103
  store i32 0, i32* %.s0003_389, align 4, !dbg !103
  %1 = load i32, i32* %__gtid___nv_MAIN__F1L48_3__393, align 4, !dbg !103
  %2 = call i32 @__kmpc_single(i64* null, i32 %1), !dbg !103
  %3 = icmp eq i32 %2, 0, !dbg !103
  br i1 %3, label %L.LB7_339, label %L.LB7_319, !dbg !103

L.LB7_319:                                        ; preds = %L.LB7_317
  %4 = bitcast %struct_drb106_0_* @_drb106_0_ to i64*, !dbg !104
  %5 = call i32 @drb106_fib_(i64* %4), !dbg !104
  %6 = bitcast i64* %__nv_MAIN__F1L48_3Arg2 to i32**, !dbg !104
  %7 = load i32*, i32** %6, align 8, !dbg !104
  store i32 %5, i32* %7, align 4, !dbg !104
  %8 = load i32, i32* %__gtid___nv_MAIN__F1L48_3__393, align 4, !dbg !105
  store i32 %8, i32* %.s0002_388, align 4, !dbg !105
  store i32 1, i32* %.s0003_389, align 4, !dbg !105
  %9 = load i32, i32* %__gtid___nv_MAIN__F1L48_3__393, align 4, !dbg !105
  call void @__kmpc_end_single(i64* null, i32 %9), !dbg !105
  br label %L.LB7_339

L.LB7_339:                                        ; preds = %L.LB7_319, %L.LB7_317
  br label %L.LB7_320

L.LB7_320:                                        ; preds = %L.LB7_339
  %10 = load i32, i32* %__gtid___nv_MAIN__F1L48_3__393, align 4, !dbg !105
  call void @__kmpc_barrier(i64* null, i32 %10), !dbg !105
  br label %L.LB7_321

L.LB7_321:                                        ; preds = %L.LB7_320
  ret void, !dbg !102
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

declare void @fort_init(...) #1

declare void @__kmpc_fork_call(i64*, i32, i64*, i64*, ...) #1

declare void @__kmpc_end_serialized_parallel(i64*, i32) #1

declare void @__kmpc_serialized_parallel(i64*, i32) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

declare signext i32 @__kmpc_global_thread_num(i64*) #1

declare signext i32 @__kmpc_omp_taskwait(i64*, i32) #1

declare void @__kmpc_omp_task(i64*, i32, i64*) #1

declare i8* @__kmpc_omp_task_alloc(i64*, i32, i32, i32, i32, i64*) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { noinline }
attributes #1 = { "no-frame-pointer-elim-non-leaf" }
attributes #2 = { nounwind readnone speculatable }

!llvm.module.flags = !{!13, !14}
!llvm.dbg.cu = !{!3}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "input", scope: !2, file: !4, type: !12, isLocal: false, isDefinition: true)
!2 = !DIModule(scope: !3, name: "drb106")
!3 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !4, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !6, imports: !7)
!4 = !DIFile(filename: "micro-benchmarks-fortran/DRB106-taskwaitmissing-orig-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!5 = !{}
!6 = !{!0}
!7 = !{!8}
!8 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !9, entity: !2, file: !4, line: 40)
!9 = distinct !DISubprogram(name: "drb106_taskwaitmissing_orig_yes", scope: !3, file: !4, line: 40, type: !10, scopeLine: 40, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !3)
!10 = !DISubroutineType(cc: DW_CC_program, types: !11)
!11 = !{null}
!12 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!13 = !{i32 2, !"Dwarf Version", i32 4}
!14 = !{i32 2, !"Debug Info Version", i32 3}
!15 = distinct !DISubprogram(name: "fib", scope: !2, file: !4, line: 20, type: !16, scopeLine: 20, spFlags: DISPFlagDefinition | DISPFlagRecursive, unit: !3)
!16 = !DISubroutineType(types: !17)
!17 = !{!18, !12}
!18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64, align: 64)
!19 = !DILocalVariable(name: "n", arg: 1, scope: !15, file: !4, type: !12)
!20 = !DILocation(line: 0, scope: !15)
!21 = !DILocalVariable(name: "omp_sched_static", scope: !15, file: !4, type: !12)
!22 = !DILocalVariable(name: "omp_sched_dynamic", scope: !15, file: !4, type: !12)
!23 = !DILocalVariable(name: "omp_proc_bind_false", scope: !15, file: !4, type: !12)
!24 = !DILocalVariable(name: "omp_proc_bind_true", scope: !15, file: !4, type: !12)
!25 = !DILocalVariable(name: "omp_proc_bind_master", scope: !15, file: !4, type: !12)
!26 = !DILocalVariable(name: "omp_lock_hint_none", scope: !15, file: !4, type: !12)
!27 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !15, file: !4, type: !12)
!28 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !15, file: !4, type: !12)
!29 = !DILocation(line: 37, column: 1, scope: !15)
!30 = !DILocation(line: 25, column: 1, scope: !15)
!31 = !DILocation(line: 26, column: 1, scope: !15)
!32 = !DILocalVariable(scope: !15, file: !4, type: !12, flags: DIFlagArtificial)
!33 = !DILocation(line: 27, column: 1, scope: !15)
!34 = !DILocation(line: 28, column: 1, scope: !15)
!35 = !DILocation(line: 30, column: 1, scope: !15)
!36 = !DILocalVariable(name: "i", scope: !15, file: !4, type: !12)
!37 = !DILocation(line: 31, column: 1, scope: !15)
!38 = !DILocation(line: 33, column: 1, scope: !15)
!39 = !DILocalVariable(name: "j", scope: !15, file: !4, type: !12)
!40 = !DILocation(line: 34, column: 1, scope: !15)
!41 = !DILocation(line: 36, column: 1, scope: !15)
!42 = distinct !DISubprogram(name: "__nv_drb106_fib__F1L28_1", scope: !3, file: !4, line: 28, type: !43, scopeLine: 28, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !3)
!43 = !DISubroutineType(types: !44)
!44 = !{null, !12, !45}
!45 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!46 = !DILocalVariable(name: "__nv_drb106_fib__F1L28_1Arg0", scope: !42, file: !4, type: !12)
!47 = !DILocation(line: 0, scope: !42)
!48 = !DILocalVariable(name: "__nv_drb106_fib__F1L28_1Arg0", arg: 1, scope: !42, file: !4, type: !12)
!49 = !DILocalVariable(name: "__nv_drb106_fib__F1L28_1Arg1", arg: 2, scope: !42, file: !4, type: !45)
!50 = !DILocalVariable(name: "omp_sched_static", scope: !42, file: !4, type: !12)
!51 = !DILocalVariable(name: "omp_sched_dynamic", scope: !42, file: !4, type: !12)
!52 = !DILocalVariable(name: "omp_proc_bind_false", scope: !42, file: !4, type: !12)
!53 = !DILocalVariable(name: "omp_proc_bind_true", scope: !42, file: !4, type: !12)
!54 = !DILocalVariable(name: "omp_proc_bind_master", scope: !42, file: !4, type: !12)
!55 = !DILocalVariable(name: "omp_lock_hint_none", scope: !42, file: !4, type: !12)
!56 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !42, file: !4, type: !12)
!57 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !42, file: !4, type: !12)
!58 = !DILocation(line: 28, column: 1, scope: !42)
!59 = !DILocation(line: 29, column: 1, scope: !42)
!60 = !DILocation(line: 30, column: 1, scope: !42)
!61 = distinct !DISubprogram(name: "__nv_drb106_fib__F1L31_2", scope: !3, file: !4, line: 31, type: !43, scopeLine: 31, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !3)
!62 = !DILocalVariable(name: "__nv_drb106_fib__F1L31_2Arg0", scope: !61, file: !4, type: !12)
!63 = !DILocation(line: 0, scope: !61)
!64 = !DILocalVariable(name: "__nv_drb106_fib__F1L31_2Arg0", arg: 1, scope: !61, file: !4, type: !12)
!65 = !DILocalVariable(name: "__nv_drb106_fib__F1L31_2Arg1", arg: 2, scope: !61, file: !4, type: !45)
!66 = !DILocalVariable(name: "omp_sched_static", scope: !61, file: !4, type: !12)
!67 = !DILocalVariable(name: "omp_sched_dynamic", scope: !61, file: !4, type: !12)
!68 = !DILocalVariable(name: "omp_proc_bind_false", scope: !61, file: !4, type: !12)
!69 = !DILocalVariable(name: "omp_proc_bind_true", scope: !61, file: !4, type: !12)
!70 = !DILocalVariable(name: "omp_proc_bind_master", scope: !61, file: !4, type: !12)
!71 = !DILocalVariable(name: "omp_lock_hint_none", scope: !61, file: !4, type: !12)
!72 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !61, file: !4, type: !12)
!73 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !61, file: !4, type: !12)
!74 = !DILocation(line: 31, column: 1, scope: !61)
!75 = !DILocation(line: 32, column: 1, scope: !61)
!76 = !DILocation(line: 33, column: 1, scope: !61)
!77 = !DILocalVariable(name: "omp_sched_static", scope: !9, file: !4, type: !12)
!78 = !DILocation(line: 0, scope: !9)
!79 = !DILocalVariable(name: "omp_proc_bind_false", scope: !9, file: !4, type: !12)
!80 = !DILocalVariable(name: "omp_proc_bind_true", scope: !9, file: !4, type: !12)
!81 = !DILocalVariable(name: "omp_lock_hint_none", scope: !9, file: !4, type: !12)
!82 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !9, file: !4, type: !12)
!83 = !DILocation(line: 56, column: 1, scope: !9)
!84 = !DILocation(line: 40, column: 1, scope: !9)
!85 = !DILocation(line: 46, column: 1, scope: !9)
!86 = !DILocalVariable(name: "result", scope: !9, file: !4, type: !12)
!87 = !DILocation(line: 48, column: 1, scope: !9)
!88 = !DILocation(line: 54, column: 1, scope: !9)
!89 = !DILocalVariable(scope: !9, file: !4, type: !12, flags: DIFlagArtificial)
!90 = distinct !DISubprogram(name: "__nv_MAIN__F1L48_3", scope: !3, file: !4, line: 48, type: !91, scopeLine: 48, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !3)
!91 = !DISubroutineType(types: !92)
!92 = !{null, !12, !45, !45}
!93 = !DILocalVariable(name: "__nv_MAIN__F1L48_3Arg0", arg: 1, scope: !90, file: !4, type: !12)
!94 = !DILocation(line: 0, scope: !90)
!95 = !DILocalVariable(name: "__nv_MAIN__F1L48_3Arg1", arg: 2, scope: !90, file: !4, type: !45)
!96 = !DILocalVariable(name: "__nv_MAIN__F1L48_3Arg2", arg: 3, scope: !90, file: !4, type: !45)
!97 = !DILocalVariable(name: "omp_sched_static", scope: !90, file: !4, type: !12)
!98 = !DILocalVariable(name: "omp_proc_bind_false", scope: !90, file: !4, type: !12)
!99 = !DILocalVariable(name: "omp_proc_bind_true", scope: !90, file: !4, type: !12)
!100 = !DILocalVariable(name: "omp_lock_hint_none", scope: !90, file: !4, type: !12)
!101 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !90, file: !4, type: !12)
!102 = !DILocation(line: 52, column: 1, scope: !90)
!103 = !DILocation(line: 49, column: 1, scope: !90)
!104 = !DILocation(line: 50, column: 1, scope: !90)
!105 = !DILocation(line: 51, column: 1, scope: !90)

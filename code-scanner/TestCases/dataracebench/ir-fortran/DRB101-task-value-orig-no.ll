; ModuleID = '/tmp/DRB101-task-value-orig-no-e53db0.ll'
source_filename = "/tmp/DRB101-task-value-orig-no-e53db0.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct_drb101_2_ = type <{ [8 x i8] }>
%struct_drb101_0_ = type <{ [144 x i8] }>
%astruct.dt75 = type <{ i8* }>

@.C283_drb101_gen_task_ = internal constant i32 0
@.C285_drb101_gen_task_ = internal constant i32 1
@.C285___nv_drb101_gen_task__F1L20_1 = internal constant i32 1
@.C340_MAIN_ = internal constant [13 x i8] c" not expected"
@.C339_MAIN_ = internal constant [3 x i8] c") ="
@.C306_MAIN_ = internal constant i32 14
@.C338_MAIN_ = internal constant [11 x i8] c"warning: a("
@.C335_MAIN_ = internal constant i32 6
@.C332_MAIN_ = internal constant [54 x i8] c"micro-benchmarks-fortran/DRB101-task-value-orig-no.f95"
@.C334_MAIN_ = internal constant i32 44
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
@.C321___nv_MAIN__F1L34_2 = internal constant i32 100
@.C285___nv_MAIN__F1L34_2 = internal constant i32 1
@_drb101_2_ = common global %struct_drb101_2_ zeroinitializer, align 64, !dbg !0
@_drb101_0_ = common global %struct_drb101_0_ zeroinitializer, align 64, !dbg !7, !dbg !13

; Function Attrs: noinline
define float @drb101_() #0 {
.L.entry:
  ret float undef
}

define void @drb101_gen_task_(i32 %_V_i.arg) #1 !dbg !26 {
L.entry:
  %_V_i.addr = alloca i32, align 4
  %i_306 = alloca i32, align 4
  %__gtid_drb101_gen_task__336 = alloca i32, align 4
  %.s0000_329 = alloca i32, align 4
  %.z0307_328 = alloca i8*, align 8
  %"drb101_gen_task___$eq_307" = alloca [16 x i8], align 4
  call void @llvm.dbg.declare(metadata i32* %_V_i.addr, metadata !29, metadata !DIExpression()), !dbg !30
  store i32 %_V_i.arg, i32* %_V_i.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %_V_i.addr, metadata !31, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.value(metadata i32 1, metadata !32, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.value(metadata i32 0, metadata !33, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.value(metadata i32 1, metadata !34, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.value(metadata i32 0, metadata !35, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.value(metadata i32 1, metadata !36, metadata !DIExpression()), !dbg !30
  %0 = load i32, i32* %_V_i.addr, align 4, !dbg !37
  call void @llvm.dbg.value(metadata i32 %0, metadata !29, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.declare(metadata i32* %i_306, metadata !38, metadata !DIExpression()), !dbg !30
  store i32 %0, i32* %i_306, align 4, !dbg !37
  %1 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !39
  store i32 %1, i32* %__gtid_drb101_gen_task__336, align 4, !dbg !39
  br label %L.LB2_327

L.LB2_327:                                        ; preds = %L.entry
  store i32 1, i32* %.s0000_329, align 4, !dbg !40
  %2 = load i32, i32* %__gtid_drb101_gen_task__336, align 4, !dbg !41
  %3 = load i32, i32* %.s0000_329, align 4, !dbg !41
  %4 = bitcast void (i32, i64*)* @__nv_drb101_gen_task__F1L20_1_ to i64*, !dbg !41
  %5 = call i8* @__kmpc_omp_task_alloc(i64* null, i32 %2, i32 %3, i32 44, i32 32, i64* %4), !dbg !41
  store i8* %5, i8** %.z0307_328, align 8, !dbg !41
  %6 = bitcast %struct_drb101_0_* @_drb101_0_ to i8*, !dbg !41
  %7 = load i8*, i8** %.z0307_328, align 8, !dbg !41
  %8 = bitcast i8* %7 to i8***, !dbg !41
  %9 = load i8**, i8*** %8, align 8, !dbg !41
  store i8* %6, i8** %9, align 8, !dbg !41
  %10 = bitcast %struct_drb101_0_* @_drb101_0_ to i8*, !dbg !41
  %11 = getelementptr i8, i8* %10, i64 16, !dbg !41
  %12 = load i8*, i8** %.z0307_328, align 8, !dbg !41
  %13 = bitcast i8* %12 to i8**, !dbg !41
  %14 = load i8*, i8** %13, align 8, !dbg !41
  %15 = getelementptr i8, i8* %14, i64 8, !dbg !41
  %16 = bitcast i8* %15 to i8**, !dbg !41
  store i8* %11, i8** %16, align 8, !dbg !41
  %17 = bitcast %struct_drb101_0_* @_drb101_0_ to i8*, !dbg !41
  %18 = load i8*, i8** %.z0307_328, align 8, !dbg !41
  %19 = bitcast i8* %18 to i8**, !dbg !41
  %20 = load i8*, i8** %19, align 8, !dbg !41
  %21 = getelementptr i8, i8* %20, i64 16, !dbg !41
  %22 = bitcast i8* %21 to i8**, !dbg !41
  store i8* %17, i8** %22, align 8, !dbg !41
  %23 = bitcast i32* %i_306 to i8*, !dbg !41
  %24 = load i8*, i8** %.z0307_328, align 8, !dbg !41
  %25 = bitcast i8* %24 to i8**, !dbg !41
  %26 = load i8*, i8** %25, align 8, !dbg !41
  %27 = getelementptr i8, i8* %26, i64 24, !dbg !41
  %28 = bitcast i8* %27 to i8**, !dbg !41
  store i8* %23, i8** %28, align 8, !dbg !41
  %29 = load i32, i32* %i_306, align 4, !dbg !40
  call void @llvm.dbg.value(metadata i32 %29, metadata !38, metadata !DIExpression()), !dbg !30
  %30 = load i8*, i8** %.z0307_328, align 8, !dbg !40
  %31 = getelementptr i8, i8* %30, i64 40, !dbg !40
  %32 = bitcast i8* %31 to i32*, !dbg !40
  store i32 %29, i32* %32, align 4, !dbg !40
  %33 = load i32, i32* %__gtid_drb101_gen_task__336, align 4, !dbg !41
  %34 = load i8*, i8** %.z0307_328, align 8, !dbg !41
  %35 = bitcast i8* %34 to i64*, !dbg !41
  call void @__kmpc_omp_task(i64* null, i32 %33, i64* %35), !dbg !41
  br label %L.LB2_323

L.LB2_323:                                        ; preds = %L.LB2_327
  ret void, !dbg !39
}

define internal void @__nv_drb101_gen_task__F1L20_1_(i32 %__nv_drb101_gen_task__F1L20_1Arg0.arg, i64* %__nv_drb101_gen_task__F1L20_1Arg1) #1 !dbg !42 {
L.entry:
  %__nv_drb101_gen_task__F1L20_1Arg0.addr = alloca i32, align 4
  %.S0000_370 = alloca i8*, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_drb101_gen_task__F1L20_1Arg0.addr, metadata !45, metadata !DIExpression()), !dbg !46
  store i32 %__nv_drb101_gen_task__F1L20_1Arg0.arg, i32* %__nv_drb101_gen_task__F1L20_1Arg0.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_drb101_gen_task__F1L20_1Arg0.addr, metadata !47, metadata !DIExpression()), !dbg !46
  call void @llvm.dbg.declare(metadata i64* %__nv_drb101_gen_task__F1L20_1Arg1, metadata !48, metadata !DIExpression()), !dbg !46
  call void @llvm.dbg.value(metadata i32 1, metadata !49, metadata !DIExpression()), !dbg !46
  call void @llvm.dbg.value(metadata i32 0, metadata !50, metadata !DIExpression()), !dbg !46
  call void @llvm.dbg.value(metadata i32 1, metadata !51, metadata !DIExpression()), !dbg !46
  call void @llvm.dbg.value(metadata i32 0, metadata !52, metadata !DIExpression()), !dbg !46
  call void @llvm.dbg.value(metadata i32 1, metadata !53, metadata !DIExpression()), !dbg !46
  %0 = bitcast i64* %__nv_drb101_gen_task__F1L20_1Arg1 to i8**, !dbg !54
  %1 = load i8*, i8** %0, align 8, !dbg !54
  store i8* %1, i8** %.S0000_370, align 8, !dbg !54
  br label %L.LB3_374

L.LB3_374:                                        ; preds = %L.entry
  br label %L.LB3_318

L.LB3_318:                                        ; preds = %L.LB3_374
  %2 = bitcast i64* %__nv_drb101_gen_task__F1L20_1Arg1 to i8*, !dbg !55
  %3 = getelementptr i8, i8* %2, i64 40, !dbg !55
  %4 = bitcast i8* %3 to i64*, !dbg !55
  %5 = bitcast i64* %4 to i32*, !dbg !55
  %6 = load i32, i32* %5, align 4, !dbg !55
  %7 = add nsw i32 %6, 1, !dbg !55
  %8 = bitcast i64* %__nv_drb101_gen_task__F1L20_1Arg1 to i8*, !dbg !55
  %9 = getelementptr i8, i8* %8, i64 40, !dbg !55
  %10 = bitcast i8* %9 to i64*, !dbg !55
  %11 = bitcast i64* %10 to i32*, !dbg !55
  %12 = load i32, i32* %11, align 4, !dbg !55
  %13 = sext i32 %12 to i64, !dbg !55
  %14 = load i8*, i8** %.S0000_370, align 8, !dbg !55
  %15 = getelementptr i8, i8* %14, i64 8, !dbg !55
  %16 = bitcast i8* %15 to i8**, !dbg !55
  %17 = load i8*, i8** %16, align 8, !dbg !55
  %18 = getelementptr i8, i8* %17, i64 56, !dbg !55
  %19 = bitcast i8* %18 to i64*, !dbg !55
  %20 = load i64, i64* %19, align 8, !dbg !55
  %21 = add nsw i64 %13, %20, !dbg !55
  %22 = load i8*, i8** %.S0000_370, align 8, !dbg !55
  %23 = getelementptr i8, i8* %22, i64 16, !dbg !55
  %24 = bitcast i8* %23 to i8***, !dbg !55
  %25 = load i8**, i8*** %24, align 8, !dbg !55
  %26 = load i8*, i8** %25, align 8, !dbg !55
  %27 = getelementptr i8, i8* %26, i64 -4, !dbg !55
  %28 = bitcast i8* %27 to i32*, !dbg !55
  %29 = getelementptr i32, i32* %28, i64 %21, !dbg !55
  store i32 %7, i32* %29, align 4, !dbg !55
  br label %L.LB3_320

L.LB3_320:                                        ; preds = %L.LB3_318
  ret void, !dbg !56
}

define void @MAIN_() #1 !dbg !21 {
L.entry:
  %__gtid_MAIN__398 = alloca i32, align 4
  %.g0000_384 = alloca i64, align 8
  %i_320 = alloca i32, align 4
  %.uplevelArgPack0002_392 = alloca %astruct.dt75, align 8
  %.dY0002_361 = alloca i32, align 4
  %z__io_337 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !57, metadata !DIExpression()), !dbg !58
  call void @llvm.dbg.value(metadata i32 0, metadata !59, metadata !DIExpression()), !dbg !58
  call void @llvm.dbg.value(metadata i32 1, metadata !60, metadata !DIExpression()), !dbg !58
  call void @llvm.dbg.value(metadata i32 0, metadata !61, metadata !DIExpression()), !dbg !58
  call void @llvm.dbg.value(metadata i32 1, metadata !62, metadata !DIExpression()), !dbg !58
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !63
  store i32 %0, i32* %__gtid_MAIN__398, align 4, !dbg !63
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !64
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !64
  call void (i8*, ...) %2(i8* %1), !dbg !64
  br label %L.LB5_364

L.LB5_364:                                        ; preds = %L.entry
  %3 = bitcast %struct_drb101_0_* @_drb101_0_ to i8*, !dbg !65
  %4 = getelementptr i8, i8* %3, i64 96, !dbg !65
  %5 = bitcast i8* %4 to i64*, !dbg !65
  store i64 1, i64* %5, align 8, !dbg !65
  %6 = bitcast %struct_drb101_0_* @_drb101_0_ to i8*, !dbg !65
  %7 = getelementptr i8, i8* %6, i64 104, !dbg !65
  %8 = bitcast i8* %7 to i64*, !dbg !65
  store i64 100, i64* %8, align 8, !dbg !65
  %9 = bitcast %struct_drb101_0_* @_drb101_0_ to i8*, !dbg !65
  %10 = getelementptr i8, i8* %9, i64 104, !dbg !65
  %11 = bitcast i8* %10 to i64*, !dbg !65
  %12 = load i64, i64* %11, align 8, !dbg !65
  %13 = sub nsw i64 %12, 1, !dbg !65
  %14 = bitcast %struct_drb101_0_* @_drb101_0_ to i8*, !dbg !65
  %15 = getelementptr i8, i8* %14, i64 96, !dbg !65
  %16 = bitcast i8* %15 to i64*, !dbg !65
  %17 = load i64, i64* %16, align 8, !dbg !65
  %18 = add nsw i64 %13, %17, !dbg !65
  store i64 %18, i64* %.g0000_384, align 8, !dbg !65
  %19 = bitcast %struct_drb101_0_* @_drb101_0_ to i8*, !dbg !65
  %20 = getelementptr i8, i8* %19, i64 16, !dbg !65
  %21 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !65
  %22 = bitcast i64* @.C347_MAIN_ to i8*, !dbg !65
  %23 = bitcast i64* @.C348_MAIN_ to i8*, !dbg !65
  %24 = bitcast %struct_drb101_0_* @_drb101_0_ to i8*, !dbg !65
  %25 = getelementptr i8, i8* %24, i64 96, !dbg !65
  %26 = bitcast i64* %.g0000_384 to i8*, !dbg !65
  %27 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !65
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %27(i8* %20, i8* %21, i8* %22, i8* %23, i8* %25, i8* %26), !dbg !65
  %28 = bitcast %struct_drb101_0_* @_drb101_0_ to i8*, !dbg !65
  %29 = getelementptr i8, i8* %28, i64 16, !dbg !65
  %30 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !65
  call void (i8*, i32, ...) %30(i8* %29, i32 25), !dbg !65
  %31 = bitcast %struct_drb101_0_* @_drb101_0_ to i8*, !dbg !65
  %32 = getelementptr i8, i8* %31, i64 104, !dbg !65
  %33 = bitcast i8* %32 to i64*, !dbg !65
  %34 = load i64, i64* %33, align 8, !dbg !65
  %35 = sub nsw i64 %34, 1, !dbg !65
  %36 = bitcast %struct_drb101_0_* @_drb101_0_ to i8*, !dbg !65
  %37 = getelementptr i8, i8* %36, i64 96, !dbg !65
  %38 = bitcast i8* %37 to i64*, !dbg !65
  %39 = load i64, i64* %38, align 8, !dbg !65
  %40 = add nsw i64 %35, %39, !dbg !65
  %41 = bitcast %struct_drb101_0_* @_drb101_0_ to i8*, !dbg !65
  %42 = getelementptr i8, i8* %41, i64 96, !dbg !65
  %43 = bitcast i8* %42 to i64*, !dbg !65
  %44 = load i64, i64* %43, align 8, !dbg !65
  %45 = sub nsw i64 %44, 1, !dbg !65
  %46 = sub nsw i64 %40, %45, !dbg !65
  store i64 %46, i64* %.g0000_384, align 8, !dbg !65
  %47 = bitcast i64* %.g0000_384 to i8*, !dbg !65
  %48 = bitcast i64* @.C347_MAIN_ to i8*, !dbg !65
  %49 = bitcast i64* @.C348_MAIN_ to i8*, !dbg !65
  %50 = bitcast %struct_drb101_0_* @_drb101_0_ to i8*, !dbg !65
  %51 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !65
  %52 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !65
  %53 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !65
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %53(i8* %47, i8* %48, i8* %49, i8* null, i8* %50, i8* null, i8* %51, i8* %52, i8* null, i64 0), !dbg !65
  call void @llvm.dbg.declare(metadata i32* %i_320, metadata !66, metadata !DIExpression()), !dbg !58
  %54 = bitcast i32* %i_320 to i8*, !dbg !67
  %55 = bitcast %astruct.dt75* %.uplevelArgPack0002_392 to i8**, !dbg !67
  store i8* %54, i8** %55, align 8, !dbg !67
  br label %L.LB5_396, !dbg !67

L.LB5_396:                                        ; preds = %L.LB5_364
  %56 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L34_2_ to i64*, !dbg !67
  %57 = bitcast %astruct.dt75* %.uplevelArgPack0002_392 to i64*, !dbg !67
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %56, i64* %57), !dbg !67
  store i32 100, i32* %.dY0002_361, align 4, !dbg !68
  store i32 1, i32* %i_320, align 4, !dbg !68
  br label %L.LB5_359

L.LB5_359:                                        ; preds = %L.LB5_362, %L.LB5_396
  %58 = load i32, i32* %i_320, align 4, !dbg !69
  call void @llvm.dbg.value(metadata i32 %58, metadata !66, metadata !DIExpression()), !dbg !58
  %59 = sext i32 %58 to i64, !dbg !69
  %60 = bitcast %struct_drb101_0_* @_drb101_0_ to i8*, !dbg !69
  %61 = getelementptr i8, i8* %60, i64 72, !dbg !69
  %62 = bitcast i8* %61 to i64*, !dbg !69
  %63 = load i64, i64* %62, align 8, !dbg !69
  %64 = add nsw i64 %59, %63, !dbg !69
  %65 = bitcast %struct_drb101_0_* @_drb101_0_ to i8**, !dbg !69
  %66 = load i8*, i8** %65, align 8, !dbg !69
  %67 = getelementptr i8, i8* %66, i64 -4, !dbg !69
  %68 = bitcast i8* %67 to i32*, !dbg !69
  %69 = getelementptr i32, i32* %68, i64 %64, !dbg !69
  %70 = load i32, i32* %69, align 4, !dbg !69
  %71 = load i32, i32* %i_320, align 4, !dbg !69
  call void @llvm.dbg.value(metadata i32 %71, metadata !66, metadata !DIExpression()), !dbg !58
  %72 = add nsw i32 %71, 1, !dbg !69
  %73 = icmp eq i32 %70, %72, !dbg !69
  br i1 %73, label %L.LB5_362, label %L.LB5_436, !dbg !69

L.LB5_436:                                        ; preds = %L.LB5_359
  call void (...) @_mp_bcs_nest(), !dbg !70
  %74 = bitcast i32* @.C334_MAIN_ to i8*, !dbg !70
  %75 = bitcast [54 x i8]* @.C332_MAIN_ to i8*, !dbg !70
  %76 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !70
  call void (i8*, i8*, i64, ...) %76(i8* %74, i8* %75, i64 54), !dbg !70
  %77 = bitcast i32* @.C335_MAIN_ to i8*, !dbg !70
  %78 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !70
  %79 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !70
  %80 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !70
  %81 = call i32 (i8*, i8*, i8*, i8*, ...) %80(i8* %77, i8* null, i8* %78, i8* %79), !dbg !70
  call void @llvm.dbg.declare(metadata i32* %z__io_337, metadata !71, metadata !DIExpression()), !dbg !58
  store i32 %81, i32* %z__io_337, align 4, !dbg !70
  %82 = bitcast [11 x i8]* @.C338_MAIN_ to i8*, !dbg !70
  %83 = bitcast i32 (...)* @f90io_sc_ch_ldw to i32 (i8*, i32, i64, ...)*, !dbg !70
  %84 = call i32 (i8*, i32, i64, ...) %83(i8* %82, i32 14, i64 11), !dbg !70
  store i32 %84, i32* %z__io_337, align 4, !dbg !70
  %85 = load i32, i32* %i_320, align 4, !dbg !70
  call void @llvm.dbg.value(metadata i32 %85, metadata !66, metadata !DIExpression()), !dbg !58
  %86 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !70
  %87 = call i32 (i32, i32, ...) %86(i32 %85, i32 25), !dbg !70
  store i32 %87, i32* %z__io_337, align 4, !dbg !70
  %88 = bitcast [3 x i8]* @.C339_MAIN_ to i8*, !dbg !70
  %89 = bitcast i32 (...)* @f90io_sc_ch_ldw to i32 (i8*, i32, i64, ...)*, !dbg !70
  %90 = call i32 (i8*, i32, i64, ...) %89(i8* %88, i32 14, i64 3), !dbg !70
  store i32 %90, i32* %z__io_337, align 4, !dbg !70
  %91 = load i32, i32* %i_320, align 4, !dbg !70
  call void @llvm.dbg.value(metadata i32 %91, metadata !66, metadata !DIExpression()), !dbg !58
  %92 = sext i32 %91 to i64, !dbg !70
  %93 = bitcast %struct_drb101_0_* @_drb101_0_ to i8*, !dbg !70
  %94 = getelementptr i8, i8* %93, i64 72, !dbg !70
  %95 = bitcast i8* %94 to i64*, !dbg !70
  %96 = load i64, i64* %95, align 8, !dbg !70
  %97 = add nsw i64 %92, %96, !dbg !70
  %98 = bitcast %struct_drb101_0_* @_drb101_0_ to i8**, !dbg !70
  %99 = load i8*, i8** %98, align 8, !dbg !70
  %100 = getelementptr i8, i8* %99, i64 -4, !dbg !70
  %101 = bitcast i8* %100 to i32*, !dbg !70
  %102 = getelementptr i32, i32* %101, i64 %97, !dbg !70
  %103 = load i32, i32* %102, align 4, !dbg !70
  %104 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !70
  %105 = call i32 (i32, i32, ...) %104(i32 %103, i32 25), !dbg !70
  store i32 %105, i32* %z__io_337, align 4, !dbg !70
  %106 = bitcast [13 x i8]* @.C340_MAIN_ to i8*, !dbg !70
  %107 = bitcast i32 (...)* @f90io_sc_ch_ldw to i32 (i8*, i32, i64, ...)*, !dbg !70
  %108 = call i32 (i8*, i32, i64, ...) %107(i8* %106, i32 14, i64 13), !dbg !70
  store i32 %108, i32* %z__io_337, align 4, !dbg !70
  %109 = load i32, i32* %i_320, align 4, !dbg !70
  call void @llvm.dbg.value(metadata i32 %109, metadata !66, metadata !DIExpression()), !dbg !58
  %110 = add nsw i32 %109, 1, !dbg !70
  %111 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !70
  %112 = call i32 (i32, i32, ...) %111(i32 %110, i32 25), !dbg !70
  store i32 %112, i32* %z__io_337, align 4, !dbg !70
  %113 = call i32 (...) @f90io_ldw_end(), !dbg !70
  store i32 %113, i32* %z__io_337, align 4, !dbg !70
  call void (...) @_mp_ecs_nest(), !dbg !70
  br label %L.LB5_362

L.LB5_362:                                        ; preds = %L.LB5_436, %L.LB5_359
  %114 = load i32, i32* %i_320, align 4, !dbg !72
  call void @llvm.dbg.value(metadata i32 %114, metadata !66, metadata !DIExpression()), !dbg !58
  %115 = add nsw i32 %114, 1, !dbg !72
  store i32 %115, i32* %i_320, align 4, !dbg !72
  %116 = load i32, i32* %.dY0002_361, align 4, !dbg !72
  %117 = sub nsw i32 %116, 1, !dbg !72
  store i32 %117, i32* %.dY0002_361, align 4, !dbg !72
  %118 = load i32, i32* %.dY0002_361, align 4, !dbg !72
  %119 = icmp sgt i32 %118, 0, !dbg !72
  br i1 %119, label %L.LB5_359, label %L.LB5_437, !dbg !72

L.LB5_437:                                        ; preds = %L.LB5_362
  ret void, !dbg !63
}

define internal void @__nv_MAIN__F1L34_2_(i32* %__nv_MAIN__F1L34_2Arg0, i64* %__nv_MAIN__F1L34_2Arg1, i64* %__nv_MAIN__F1L34_2Arg2) #1 !dbg !73 {
L.entry:
  %__gtid___nv_MAIN__F1L34_2__447 = alloca i32, align 4
  %.s0001_442 = alloca i32, align 4
  %.s0002_443 = alloca i32, align 4
  %.dY0001p_358 = alloca i32, align 4
  %i_328 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L34_2Arg0, metadata !76, metadata !DIExpression()), !dbg !77
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L34_2Arg1, metadata !78, metadata !DIExpression()), !dbg !77
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L34_2Arg2, metadata !79, metadata !DIExpression()), !dbg !77
  call void @llvm.dbg.value(metadata i32 1, metadata !80, metadata !DIExpression()), !dbg !77
  call void @llvm.dbg.value(metadata i32 0, metadata !81, metadata !DIExpression()), !dbg !77
  call void @llvm.dbg.value(metadata i32 1, metadata !82, metadata !DIExpression()), !dbg !77
  call void @llvm.dbg.value(metadata i32 0, metadata !83, metadata !DIExpression()), !dbg !77
  call void @llvm.dbg.value(metadata i32 1, metadata !84, metadata !DIExpression()), !dbg !77
  %0 = load i32, i32* %__nv_MAIN__F1L34_2Arg0, align 4, !dbg !85
  store i32 %0, i32* %__gtid___nv_MAIN__F1L34_2__447, align 4, !dbg !85
  br label %L.LB6_441

L.LB6_441:                                        ; preds = %L.entry
  br label %L.LB6_325

L.LB6_325:                                        ; preds = %L.LB6_441
  store i32 -1, i32* %.s0001_442, align 4, !dbg !86
  store i32 0, i32* %.s0002_443, align 4, !dbg !86
  %1 = load i32, i32* %__gtid___nv_MAIN__F1L34_2__447, align 4, !dbg !86
  %2 = call i32 @__kmpc_single(i64* null, i32 %1), !dbg !86
  %3 = icmp eq i32 %2, 0, !dbg !86
  br i1 %3, label %L.LB6_355, label %L.LB6_327, !dbg !86

L.LB6_327:                                        ; preds = %L.LB6_325
  store i32 100, i32* %.dY0001p_358, align 4, !dbg !87
  call void @llvm.dbg.declare(metadata i32* %i_328, metadata !88, metadata !DIExpression()), !dbg !85
  store i32 1, i32* %i_328, align 4, !dbg !87
  br label %L.LB6_356

L.LB6_356:                                        ; preds = %L.LB6_356, %L.LB6_327
  %4 = load i32, i32* %i_328, align 4, !dbg !89
  call void @llvm.dbg.value(metadata i32 %4, metadata !88, metadata !DIExpression()), !dbg !85
  call void @drb101_gen_task_(i32 %4), !dbg !89
  %5 = load i32, i32* %i_328, align 4, !dbg !90
  call void @llvm.dbg.value(metadata i32 %5, metadata !88, metadata !DIExpression()), !dbg !85
  %6 = add nsw i32 %5, 1, !dbg !90
  store i32 %6, i32* %i_328, align 4, !dbg !90
  %7 = load i32, i32* %.dY0001p_358, align 4, !dbg !90
  %8 = sub nsw i32 %7, 1, !dbg !90
  store i32 %8, i32* %.dY0001p_358, align 4, !dbg !90
  %9 = load i32, i32* %.dY0001p_358, align 4, !dbg !90
  %10 = icmp sgt i32 %9, 0, !dbg !90
  br i1 %10, label %L.LB6_356, label %L.LB6_464, !dbg !90

L.LB6_464:                                        ; preds = %L.LB6_356
  %11 = load i32, i32* %__gtid___nv_MAIN__F1L34_2__447, align 4, !dbg !91
  store i32 %11, i32* %.s0001_442, align 4, !dbg !91
  store i32 1, i32* %.s0002_443, align 4, !dbg !91
  %12 = load i32, i32* %__gtid___nv_MAIN__F1L34_2__447, align 4, !dbg !91
  call void @__kmpc_end_single(i64* null, i32 %12), !dbg !91
  br label %L.LB6_355

L.LB6_355:                                        ; preds = %L.LB6_464, %L.LB6_325
  br label %L.LB6_329

L.LB6_329:                                        ; preds = %L.LB6_355
  %13 = load i32, i32* %__gtid___nv_MAIN__F1L34_2__447, align 4, !dbg !91
  call void @__kmpc_barrier(i64* null, i32 %13), !dbg !91
  br label %L.LB6_330

L.LB6_330:                                        ; preds = %L.LB6_329
  ret void, !dbg !85
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
!2 = !DIModule(scope: !3, name: "drb101")
!3 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !4, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !6, imports: !19)
!4 = !DIFile(filename: "micro-benchmarks-fortran/DRB101-task-value-orig-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
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
!20 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !21, entity: !2, file: !4, line: 26)
!21 = distinct !DISubprogram(name: "drb101_task_value_orig_no", scope: !3, file: !4, line: 26, type: !22, scopeLine: 26, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !3)
!22 = !DISubroutineType(cc: DW_CC_program, types: !23)
!23 = !{null}
!24 = !{i32 2, !"Dwarf Version", i32 4}
!25 = !{i32 2, !"Debug Info Version", i32 3}
!26 = distinct !DISubprogram(name: "gen_task", scope: !2, file: !4, line: 16, type: !27, scopeLine: 16, spFlags: DISPFlagDefinition, unit: !3)
!27 = !DISubroutineType(types: !28)
!28 = !{null, !10}
!29 = !DILocalVariable(name: "_V_i", scope: !26, file: !4, type: !10)
!30 = !DILocation(line: 0, scope: !26)
!31 = !DILocalVariable(name: "_V_i", arg: 1, scope: !26, file: !4, type: !10)
!32 = !DILocalVariable(name: "omp_sched_static", scope: !26, file: !4, type: !10)
!33 = !DILocalVariable(name: "omp_proc_bind_false", scope: !26, file: !4, type: !10)
!34 = !DILocalVariable(name: "omp_proc_bind_true", scope: !26, file: !4, type: !10)
!35 = !DILocalVariable(name: "omp_lock_hint_none", scope: !26, file: !4, type: !10)
!36 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !26, file: !4, type: !10)
!37 = !DILocation(line: 16, column: 1, scope: !26)
!38 = !DILocalVariable(name: "i", scope: !26, file: !4, type: !10)
!39 = !DILocation(line: 23, column: 1, scope: !26)
!40 = !DILocation(line: 20, column: 1, scope: !26)
!41 = !DILocation(line: 22, column: 1, scope: !26)
!42 = distinct !DISubprogram(name: "__nv_drb101_gen_task__F1L20_1", scope: !3, file: !4, line: 20, type: !43, scopeLine: 20, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !3)
!43 = !DISubroutineType(types: !44)
!44 = !{null, !10, !16}
!45 = !DILocalVariable(name: "__nv_drb101_gen_task__F1L20_1Arg0", scope: !42, file: !4, type: !10)
!46 = !DILocation(line: 0, scope: !42)
!47 = !DILocalVariable(name: "__nv_drb101_gen_task__F1L20_1Arg0", arg: 1, scope: !42, file: !4, type: !10)
!48 = !DILocalVariable(name: "__nv_drb101_gen_task__F1L20_1Arg1", arg: 2, scope: !42, file: !4, type: !16)
!49 = !DILocalVariable(name: "omp_sched_static", scope: !42, file: !4, type: !10)
!50 = !DILocalVariable(name: "omp_proc_bind_false", scope: !42, file: !4, type: !10)
!51 = !DILocalVariable(name: "omp_proc_bind_true", scope: !42, file: !4, type: !10)
!52 = !DILocalVariable(name: "omp_lock_hint_none", scope: !42, file: !4, type: !10)
!53 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !42, file: !4, type: !10)
!54 = !DILocation(line: 20, column: 1, scope: !42)
!55 = !DILocation(line: 21, column: 1, scope: !42)
!56 = !DILocation(line: 22, column: 1, scope: !42)
!57 = !DILocalVariable(name: "omp_sched_static", scope: !21, file: !4, type: !10)
!58 = !DILocation(line: 0, scope: !21)
!59 = !DILocalVariable(name: "omp_proc_bind_false", scope: !21, file: !4, type: !10)
!60 = !DILocalVariable(name: "omp_proc_bind_true", scope: !21, file: !4, type: !10)
!61 = !DILocalVariable(name: "omp_lock_hint_none", scope: !21, file: !4, type: !10)
!62 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !21, file: !4, type: !10)
!63 = !DILocation(line: 48, column: 1, scope: !21)
!64 = !DILocation(line: 26, column: 1, scope: !21)
!65 = !DILocation(line: 32, column: 1, scope: !21)
!66 = !DILocalVariable(name: "i", scope: !21, file: !4, type: !10)
!67 = !DILocation(line: 34, column: 1, scope: !21)
!68 = !DILocation(line: 42, column: 1, scope: !21)
!69 = !DILocation(line: 43, column: 1, scope: !21)
!70 = !DILocation(line: 44, column: 1, scope: !21)
!71 = !DILocalVariable(scope: !21, file: !4, type: !10, flags: DIFlagArtificial)
!72 = !DILocation(line: 47, column: 1, scope: !21)
!73 = distinct !DISubprogram(name: "__nv_MAIN__F1L34_2", scope: !3, file: !4, line: 34, type: !74, scopeLine: 34, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !3)
!74 = !DISubroutineType(types: !75)
!75 = !{null, !10, !16, !16}
!76 = !DILocalVariable(name: "__nv_MAIN__F1L34_2Arg0", arg: 1, scope: !73, file: !4, type: !10)
!77 = !DILocation(line: 0, scope: !73)
!78 = !DILocalVariable(name: "__nv_MAIN__F1L34_2Arg1", arg: 2, scope: !73, file: !4, type: !16)
!79 = !DILocalVariable(name: "__nv_MAIN__F1L34_2Arg2", arg: 3, scope: !73, file: !4, type: !16)
!80 = !DILocalVariable(name: "omp_sched_static", scope: !73, file: !4, type: !10)
!81 = !DILocalVariable(name: "omp_proc_bind_false", scope: !73, file: !4, type: !10)
!82 = !DILocalVariable(name: "omp_proc_bind_true", scope: !73, file: !4, type: !10)
!83 = !DILocalVariable(name: "omp_lock_hint_none", scope: !73, file: !4, type: !10)
!84 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !73, file: !4, type: !10)
!85 = !DILocation(line: 40, column: 1, scope: !73)
!86 = !DILocation(line: 35, column: 1, scope: !73)
!87 = !DILocation(line: 36, column: 1, scope: !73)
!88 = !DILocalVariable(name: "i", scope: !73, file: !4, type: !10)
!89 = !DILocation(line: 37, column: 1, scope: !73)
!90 = !DILocation(line: 38, column: 1, scope: !73)
!91 = !DILocation(line: 39, column: 1, scope: !73)

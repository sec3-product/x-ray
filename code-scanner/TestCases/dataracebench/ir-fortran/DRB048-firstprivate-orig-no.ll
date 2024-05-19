; ModuleID = '/tmp/DRB048-firstprivate-orig-no-2352d6.ll'
source_filename = "/tmp/DRB048-firstprivate-orig-no-2352d6.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct_drb048_2_ = type <{ [8 x i8] }>
%struct_drb048_0_ = type <{ [144 x i8] }>
%astruct.dt72 = type <{ i8*, i8*, i8*, i8*, i8* }>

@.C285_drb048_foo_ = internal constant i32 1
@.C283_drb048_foo_ = internal constant i32 0
@.C285___nv_drb048_foo__F1L23_1 = internal constant i32 1
@.C283___nv_drb048_foo__F1L23_1 = internal constant i32 0
@.C332_MAIN_ = internal constant i64 50
@.C329_MAIN_ = internal constant i32 6
@.C326_MAIN_ = internal constant [56 x i8] c"micro-benchmarks-fortran/DRB048-firstprivate-orig-no.f95"
@.C328_MAIN_ = internal constant i32 38
@.C324_MAIN_ = internal constant i32 7
@.C322_MAIN_ = internal constant i32 100
@.C306_MAIN_ = internal constant i32 25
@.C339_MAIN_ = internal constant i64 4
@.C338_MAIN_ = internal constant i64 25
@.C284_MAIN_ = internal constant i64 0
@.C323_MAIN_ = internal constant i64 100
@.C309_MAIN_ = internal constant i64 12
@.C286_MAIN_ = internal constant i64 1
@.C308_MAIN_ = internal constant i64 11
@.C283_MAIN_ = internal constant i32 0
@_drb048_2_ = common global %struct_drb048_2_ zeroinitializer, align 64, !dbg !0
@_drb048_0_ = common global %struct_drb048_0_ zeroinitializer, align 64, !dbg !7, !dbg !13

; Function Attrs: noinline
define float @drb048_() #0 {
.L.entry:
  ret float undef
}

define void @drb048_foo_(i64* %"a$p6", i32 %_V_n.arg, i32 %_V_g.arg, i64* %"a$sd5") #1 !dbg !26 {
L.entry:
  %_V_n.addr = alloca i32, align 4
  %_V_g.addr = alloca i32, align 4
  %n_313 = alloca i32, align 4
  %g_314 = alloca i32, align 4
  %__gtid_drb048_foo__382 = alloca i32, align 4
  %.uplevelArgPack0001_367 = alloca %astruct.dt72, align 16
  %"drb048_foo___$eq_316" = alloca [16 x i8], align 4
  call void @llvm.dbg.declare(metadata i64* %"a$p6", metadata !29, metadata !DIExpression(DW_OP_deref)), !dbg !30
  call void @llvm.dbg.declare(metadata i32* %_V_n.addr, metadata !31, metadata !DIExpression()), !dbg !30
  store i32 %_V_n.arg, i32* %_V_n.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %_V_n.addr, metadata !32, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.declare(metadata i32* %_V_g.addr, metadata !33, metadata !DIExpression()), !dbg !30
  store i32 %_V_g.arg, i32* %_V_g.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %_V_g.addr, metadata !34, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.declare(metadata i64* %"a$sd5", metadata !35, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.value(metadata i32 1, metadata !36, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.value(metadata i32 0, metadata !37, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.value(metadata i32 1, metadata !38, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.value(metadata i32 0, metadata !39, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.value(metadata i32 1, metadata !40, metadata !DIExpression()), !dbg !30
  %0 = load i32, i32* %_V_n.addr, align 4, !dbg !41
  call void @llvm.dbg.value(metadata i32 %0, metadata !31, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.declare(metadata i32* %n_313, metadata !42, metadata !DIExpression()), !dbg !30
  store i32 %0, i32* %n_313, align 4, !dbg !41
  %1 = load i32, i32* %_V_g.addr, align 4, !dbg !41
  call void @llvm.dbg.value(metadata i32 %1, metadata !33, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.declare(metadata i32* %g_314, metadata !43, metadata !DIExpression()), !dbg !30
  store i32 %1, i32* %g_314, align 4, !dbg !41
  %2 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !44
  store i32 %2, i32* %__gtid_drb048_foo__382, align 4, !dbg !44
  br label %L.LB2_362

L.LB2_362:                                        ; preds = %L.entry
  %3 = bitcast i32* %g_314 to i8*, !dbg !45
  %4 = bitcast %astruct.dt72* %.uplevelArgPack0001_367 to i8**, !dbg !45
  store i8* %3, i8** %4, align 8, !dbg !45
  %5 = bitcast i32* %n_313 to i8*, !dbg !45
  %6 = bitcast %astruct.dt72* %.uplevelArgPack0001_367 to i8*, !dbg !45
  %7 = getelementptr i8, i8* %6, i64 8, !dbg !45
  %8 = bitcast i8* %7 to i8**, !dbg !45
  store i8* %5, i8** %8, align 8, !dbg !45
  %9 = bitcast i64* %"a$p6" to i8*, !dbg !45
  %10 = bitcast %astruct.dt72* %.uplevelArgPack0001_367 to i8*, !dbg !45
  %11 = getelementptr i8, i8* %10, i64 16, !dbg !45
  %12 = bitcast i8* %11 to i8**, !dbg !45
  store i8* %9, i8** %12, align 8, !dbg !45
  %13 = bitcast i64* %"a$sd5" to i8*, !dbg !45
  %14 = bitcast %astruct.dt72* %.uplevelArgPack0001_367 to i8*, !dbg !45
  %15 = getelementptr i8, i8* %14, i64 24, !dbg !45
  %16 = bitcast i8* %15 to i8**, !dbg !45
  store i8* %13, i8** %16, align 8, !dbg !45
  %17 = bitcast i64* %"a$p6" to i8*, !dbg !45
  %18 = bitcast %astruct.dt72* %.uplevelArgPack0001_367 to i8*, !dbg !45
  %19 = getelementptr i8, i8* %18, i64 32, !dbg !45
  %20 = bitcast i8* %19 to i8**, !dbg !45
  store i8* %17, i8** %20, align 8, !dbg !45
  br label %L.LB2_380, !dbg !45

L.LB2_380:                                        ; preds = %L.LB2_362
  %21 = bitcast void (i32*, i64*, i64*)* @__nv_drb048_foo__F1L23_1_ to i64*, !dbg !45
  %22 = bitcast %astruct.dt72* %.uplevelArgPack0001_367 to i64*, !dbg !45
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %21, i64* %22), !dbg !45
  ret void, !dbg !44
}

define internal void @__nv_drb048_foo__F1L23_1_(i32* %__nv_drb048_foo__F1L23_1Arg0, i64* %__nv_drb048_foo__F1L23_1Arg1, i64* %__nv_drb048_foo__F1L23_1Arg2) #1 !dbg !46 {
L.entry:
  %__gtid___nv_drb048_foo__F1L23_1__423 = alloca i32, align 4
  %g_326 = alloca i32, align 4
  %.i0000p_328 = alloca i32, align 4
  %i_327 = alloca i32, align 4
  %.du0001p_345 = alloca i32, align 4
  %.de0001p_346 = alloca i32, align 4
  %.di0001p_347 = alloca i32, align 4
  %.ds0001p_348 = alloca i32, align 4
  %.dl0001p_350 = alloca i32, align 4
  %.dl0001p.copy_417 = alloca i32, align 4
  %.de0001p.copy_418 = alloca i32, align 4
  %.ds0001p.copy_419 = alloca i32, align 4
  %.dX0001p_349 = alloca i32, align 4
  %.dY0001p_344 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_drb048_foo__F1L23_1Arg0, metadata !49, metadata !DIExpression()), !dbg !50
  call void @llvm.dbg.declare(metadata i64* %__nv_drb048_foo__F1L23_1Arg1, metadata !51, metadata !DIExpression()), !dbg !50
  call void @llvm.dbg.declare(metadata i64* %__nv_drb048_foo__F1L23_1Arg2, metadata !52, metadata !DIExpression()), !dbg !50
  call void @llvm.dbg.value(metadata i32 1, metadata !53, metadata !DIExpression()), !dbg !50
  call void @llvm.dbg.value(metadata i32 0, metadata !54, metadata !DIExpression()), !dbg !50
  call void @llvm.dbg.value(metadata i32 1, metadata !55, metadata !DIExpression()), !dbg !50
  call void @llvm.dbg.value(metadata i32 0, metadata !56, metadata !DIExpression()), !dbg !50
  call void @llvm.dbg.value(metadata i32 1, metadata !57, metadata !DIExpression()), !dbg !50
  %0 = load i32, i32* %__nv_drb048_foo__F1L23_1Arg0, align 4, !dbg !58
  store i32 %0, i32* %__gtid___nv_drb048_foo__F1L23_1__423, align 4, !dbg !58
  br label %L.LB3_405

L.LB3_405:                                        ; preds = %L.entry
  br label %L.LB3_325

L.LB3_325:                                        ; preds = %L.LB3_405
  %1 = bitcast i64* %__nv_drb048_foo__F1L23_1Arg2 to i32**, !dbg !59
  %2 = load i32*, i32** %1, align 8, !dbg !59
  %3 = load i32, i32* %2, align 4, !dbg !59
  call void @llvm.dbg.declare(metadata i32* %g_326, metadata !60, metadata !DIExpression()), !dbg !58
  store i32 %3, i32* %g_326, align 4, !dbg !59
  store i32 0, i32* %.i0000p_328, align 4, !dbg !61
  call void @llvm.dbg.declare(metadata i32* %i_327, metadata !62, metadata !DIExpression()), !dbg !58
  store i32 1, i32* %i_327, align 4, !dbg !61
  %4 = bitcast i64* %__nv_drb048_foo__F1L23_1Arg2 to i8*, !dbg !61
  %5 = getelementptr i8, i8* %4, i64 8, !dbg !61
  %6 = bitcast i8* %5 to i32**, !dbg !61
  %7 = load i32*, i32** %6, align 8, !dbg !61
  %8 = load i32, i32* %7, align 4, !dbg !61
  store i32 %8, i32* %.du0001p_345, align 4, !dbg !61
  %9 = bitcast i64* %__nv_drb048_foo__F1L23_1Arg2 to i8*, !dbg !61
  %10 = getelementptr i8, i8* %9, i64 8, !dbg !61
  %11 = bitcast i8* %10 to i32**, !dbg !61
  %12 = load i32*, i32** %11, align 8, !dbg !61
  %13 = load i32, i32* %12, align 4, !dbg !61
  store i32 %13, i32* %.de0001p_346, align 4, !dbg !61
  store i32 1, i32* %.di0001p_347, align 4, !dbg !61
  %14 = load i32, i32* %.di0001p_347, align 4, !dbg !61
  store i32 %14, i32* %.ds0001p_348, align 4, !dbg !61
  store i32 1, i32* %.dl0001p_350, align 4, !dbg !61
  %15 = load i32, i32* %.dl0001p_350, align 4, !dbg !61
  store i32 %15, i32* %.dl0001p.copy_417, align 4, !dbg !61
  %16 = load i32, i32* %.de0001p_346, align 4, !dbg !61
  store i32 %16, i32* %.de0001p.copy_418, align 4, !dbg !61
  %17 = load i32, i32* %.ds0001p_348, align 4, !dbg !61
  store i32 %17, i32* %.ds0001p.copy_419, align 4, !dbg !61
  %18 = load i32, i32* %__gtid___nv_drb048_foo__F1L23_1__423, align 4, !dbg !61
  %19 = bitcast i32* %.i0000p_328 to i64*, !dbg !61
  %20 = bitcast i32* %.dl0001p.copy_417 to i64*, !dbg !61
  %21 = bitcast i32* %.de0001p.copy_418 to i64*, !dbg !61
  %22 = bitcast i32* %.ds0001p.copy_419 to i64*, !dbg !61
  %23 = load i32, i32* %.ds0001p.copy_419, align 4, !dbg !61
  call void @__kmpc_for_static_init_4(i64* null, i32 %18, i32 34, i64* %19, i64* %20, i64* %21, i64* %22, i32 %23, i32 1), !dbg !61
  %24 = load i32, i32* %.dl0001p.copy_417, align 4, !dbg !61
  store i32 %24, i32* %.dl0001p_350, align 4, !dbg !61
  %25 = load i32, i32* %.de0001p.copy_418, align 4, !dbg !61
  store i32 %25, i32* %.de0001p_346, align 4, !dbg !61
  %26 = load i32, i32* %.ds0001p.copy_419, align 4, !dbg !61
  store i32 %26, i32* %.ds0001p_348, align 4, !dbg !61
  %27 = load i32, i32* %.dl0001p_350, align 4, !dbg !61
  store i32 %27, i32* %i_327, align 4, !dbg !61
  %28 = load i32, i32* %i_327, align 4, !dbg !61
  call void @llvm.dbg.value(metadata i32 %28, metadata !62, metadata !DIExpression()), !dbg !58
  store i32 %28, i32* %.dX0001p_349, align 4, !dbg !61
  %29 = load i32, i32* %.dX0001p_349, align 4, !dbg !61
  %30 = load i32, i32* %.du0001p_345, align 4, !dbg !61
  %31 = icmp sgt i32 %29, %30, !dbg !61
  br i1 %31, label %L.LB3_343, label %L.LB3_452, !dbg !61

L.LB3_452:                                        ; preds = %L.LB3_325
  %32 = load i32, i32* %.dX0001p_349, align 4, !dbg !61
  store i32 %32, i32* %i_327, align 4, !dbg !61
  %33 = load i32, i32* %.di0001p_347, align 4, !dbg !61
  %34 = load i32, i32* %.de0001p_346, align 4, !dbg !61
  %35 = load i32, i32* %.dX0001p_349, align 4, !dbg !61
  %36 = sub nsw i32 %34, %35, !dbg !61
  %37 = add nsw i32 %33, %36, !dbg !61
  %38 = load i32, i32* %.di0001p_347, align 4, !dbg !61
  %39 = sdiv i32 %37, %38, !dbg !61
  store i32 %39, i32* %.dY0001p_344, align 4, !dbg !61
  %40 = load i32, i32* %.dY0001p_344, align 4, !dbg !61
  %41 = icmp sle i32 %40, 0, !dbg !61
  br i1 %41, label %L.LB3_353, label %L.LB3_352, !dbg !61

L.LB3_352:                                        ; preds = %L.LB3_352, %L.LB3_452
  %42 = load i32, i32* %i_327, align 4, !dbg !63
  call void @llvm.dbg.value(metadata i32 %42, metadata !62, metadata !DIExpression()), !dbg !58
  %43 = sext i32 %42 to i64, !dbg !63
  %44 = bitcast i64* %__nv_drb048_foo__F1L23_1Arg2 to i8*, !dbg !63
  %45 = getelementptr i8, i8* %44, i64 24, !dbg !63
  %46 = bitcast i8* %45 to i8**, !dbg !63
  %47 = load i8*, i8** %46, align 8, !dbg !63
  %48 = getelementptr i8, i8* %47, i64 56, !dbg !63
  %49 = bitcast i8* %48 to i64*, !dbg !63
  %50 = load i64, i64* %49, align 8, !dbg !63
  %51 = add nsw i64 %43, %50, !dbg !63
  %52 = bitcast i64* %__nv_drb048_foo__F1L23_1Arg2 to i8*, !dbg !63
  %53 = getelementptr i8, i8* %52, i64 32, !dbg !63
  %54 = bitcast i8* %53 to i8***, !dbg !63
  %55 = load i8**, i8*** %54, align 8, !dbg !63
  %56 = load i8*, i8** %55, align 8, !dbg !63
  %57 = getelementptr i8, i8* %56, i64 -4, !dbg !63
  %58 = bitcast i8* %57 to i32*, !dbg !63
  %59 = getelementptr i32, i32* %58, i64 %51, !dbg !63
  %60 = load i32, i32* %59, align 4, !dbg !63
  %61 = load i32, i32* %g_326, align 4, !dbg !63
  call void @llvm.dbg.value(metadata i32 %61, metadata !60, metadata !DIExpression()), !dbg !58
  %62 = add nsw i32 %60, %61, !dbg !63
  %63 = load i32, i32* %i_327, align 4, !dbg !63
  call void @llvm.dbg.value(metadata i32 %63, metadata !62, metadata !DIExpression()), !dbg !58
  %64 = sext i32 %63 to i64, !dbg !63
  %65 = bitcast i64* %__nv_drb048_foo__F1L23_1Arg2 to i8*, !dbg !63
  %66 = getelementptr i8, i8* %65, i64 24, !dbg !63
  %67 = bitcast i8* %66 to i8**, !dbg !63
  %68 = load i8*, i8** %67, align 8, !dbg !63
  %69 = getelementptr i8, i8* %68, i64 56, !dbg !63
  %70 = bitcast i8* %69 to i64*, !dbg !63
  %71 = load i64, i64* %70, align 8, !dbg !63
  %72 = add nsw i64 %64, %71, !dbg !63
  %73 = bitcast i64* %__nv_drb048_foo__F1L23_1Arg2 to i8*, !dbg !63
  %74 = getelementptr i8, i8* %73, i64 32, !dbg !63
  %75 = bitcast i8* %74 to i8***, !dbg !63
  %76 = load i8**, i8*** %75, align 8, !dbg !63
  %77 = load i8*, i8** %76, align 8, !dbg !63
  %78 = getelementptr i8, i8* %77, i64 -4, !dbg !63
  %79 = bitcast i8* %78 to i32*, !dbg !63
  %80 = getelementptr i32, i32* %79, i64 %72, !dbg !63
  store i32 %62, i32* %80, align 4, !dbg !63
  %81 = load i32, i32* %.di0001p_347, align 4, !dbg !58
  %82 = load i32, i32* %i_327, align 4, !dbg !58
  call void @llvm.dbg.value(metadata i32 %82, metadata !62, metadata !DIExpression()), !dbg !58
  %83 = add nsw i32 %81, %82, !dbg !58
  store i32 %83, i32* %i_327, align 4, !dbg !58
  %84 = load i32, i32* %.dY0001p_344, align 4, !dbg !58
  %85 = sub nsw i32 %84, 1, !dbg !58
  store i32 %85, i32* %.dY0001p_344, align 4, !dbg !58
  %86 = load i32, i32* %.dY0001p_344, align 4, !dbg !58
  %87 = icmp sgt i32 %86, 0, !dbg !58
  br i1 %87, label %L.LB3_352, label %L.LB3_353, !dbg !58

L.LB3_353:                                        ; preds = %L.LB3_352, %L.LB3_452
  br label %L.LB3_343

L.LB3_343:                                        ; preds = %L.LB3_353, %L.LB3_325
  %88 = load i32, i32* %__gtid___nv_drb048_foo__F1L23_1__423, align 4, !dbg !58
  call void @__kmpc_for_static_fini(i64* null, i32 %88), !dbg !58
  br label %L.LB3_329

L.LB3_329:                                        ; preds = %L.LB3_343
  ret void, !dbg !58
}

define void @MAIN_() #1 !dbg !21 {
L.entry:
  %.g0000_373 = alloca i64, align 8
  %z__io_331 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !64, metadata !DIExpression()), !dbg !65
  call void @llvm.dbg.value(metadata i32 0, metadata !66, metadata !DIExpression()), !dbg !65
  call void @llvm.dbg.value(metadata i32 1, metadata !67, metadata !DIExpression()), !dbg !65
  call void @llvm.dbg.value(metadata i32 0, metadata !68, metadata !DIExpression()), !dbg !65
  call void @llvm.dbg.value(metadata i32 1, metadata !69, metadata !DIExpression()), !dbg !65
  %0 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !70
  %1 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !70
  call void (i8*, ...) %1(i8* %0), !dbg !70
  br label %L.LB5_353

L.LB5_353:                                        ; preds = %L.entry
  %2 = bitcast %struct_drb048_0_* @_drb048_0_ to i8*, !dbg !71
  %3 = getelementptr i8, i8* %2, i64 96, !dbg !71
  %4 = bitcast i8* %3 to i64*, !dbg !71
  store i64 1, i64* %4, align 8, !dbg !71
  %5 = bitcast %struct_drb048_0_* @_drb048_0_ to i8*, !dbg !71
  %6 = getelementptr i8, i8* %5, i64 104, !dbg !71
  %7 = bitcast i8* %6 to i64*, !dbg !71
  store i64 100, i64* %7, align 8, !dbg !71
  %8 = bitcast %struct_drb048_0_* @_drb048_0_ to i8*, !dbg !71
  %9 = getelementptr i8, i8* %8, i64 104, !dbg !71
  %10 = bitcast i8* %9 to i64*, !dbg !71
  %11 = load i64, i64* %10, align 8, !dbg !71
  %12 = sub nsw i64 %11, 1, !dbg !71
  %13 = bitcast %struct_drb048_0_* @_drb048_0_ to i8*, !dbg !71
  %14 = getelementptr i8, i8* %13, i64 96, !dbg !71
  %15 = bitcast i8* %14 to i64*, !dbg !71
  %16 = load i64, i64* %15, align 8, !dbg !71
  %17 = add nsw i64 %12, %16, !dbg !71
  store i64 %17, i64* %.g0000_373, align 8, !dbg !71
  %18 = bitcast %struct_drb048_0_* @_drb048_0_ to i8*, !dbg !71
  %19 = getelementptr i8, i8* %18, i64 16, !dbg !71
  %20 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !71
  %21 = bitcast i64* @.C338_MAIN_ to i8*, !dbg !71
  %22 = bitcast i64* @.C339_MAIN_ to i8*, !dbg !71
  %23 = bitcast %struct_drb048_0_* @_drb048_0_ to i8*, !dbg !71
  %24 = getelementptr i8, i8* %23, i64 96, !dbg !71
  %25 = bitcast i64* %.g0000_373 to i8*, !dbg !71
  %26 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !71
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %26(i8* %19, i8* %20, i8* %21, i8* %22, i8* %24, i8* %25), !dbg !71
  %27 = bitcast %struct_drb048_0_* @_drb048_0_ to i8*, !dbg !71
  %28 = getelementptr i8, i8* %27, i64 16, !dbg !71
  %29 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !71
  call void (i8*, i32, ...) %29(i8* %28, i32 25), !dbg !71
  %30 = bitcast %struct_drb048_0_* @_drb048_0_ to i8*, !dbg !71
  %31 = getelementptr i8, i8* %30, i64 104, !dbg !71
  %32 = bitcast i8* %31 to i64*, !dbg !71
  %33 = load i64, i64* %32, align 8, !dbg !71
  %34 = sub nsw i64 %33, 1, !dbg !71
  %35 = bitcast %struct_drb048_0_* @_drb048_0_ to i8*, !dbg !71
  %36 = getelementptr i8, i8* %35, i64 96, !dbg !71
  %37 = bitcast i8* %36 to i64*, !dbg !71
  %38 = load i64, i64* %37, align 8, !dbg !71
  %39 = add nsw i64 %34, %38, !dbg !71
  %40 = bitcast %struct_drb048_0_* @_drb048_0_ to i8*, !dbg !71
  %41 = getelementptr i8, i8* %40, i64 96, !dbg !71
  %42 = bitcast i8* %41 to i64*, !dbg !71
  %43 = load i64, i64* %42, align 8, !dbg !71
  %44 = sub nsw i64 %43, 1, !dbg !71
  %45 = sub nsw i64 %39, %44, !dbg !71
  store i64 %45, i64* %.g0000_373, align 8, !dbg !71
  %46 = bitcast i64* %.g0000_373 to i8*, !dbg !71
  %47 = bitcast i64* @.C338_MAIN_ to i8*, !dbg !71
  %48 = bitcast i64* @.C339_MAIN_ to i8*, !dbg !71
  %49 = bitcast %struct_drb048_0_* @_drb048_0_ to i8*, !dbg !71
  %50 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !71
  %51 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !71
  %52 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !71
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %52(i8* %46, i8* %47, i8* %48, i8* null, i8* %49, i8* null, i8* %50, i8* %51, i8* null, i64 0), !dbg !71
  %53 = bitcast %struct_drb048_0_* @_drb048_0_ to i64*, !dbg !72
  %54 = bitcast %struct_drb048_0_* @_drb048_0_ to i8*, !dbg !72
  %55 = getelementptr i8, i8* %54, i64 16, !dbg !72
  %56 = bitcast i8* %55 to i64*, !dbg !72
  call void @drb048_foo_(i64* %53, i32 100, i32 7, i64* %56), !dbg !72
  call void (...) @_mp_bcs_nest(), !dbg !73
  %57 = bitcast i32* @.C328_MAIN_ to i8*, !dbg !73
  %58 = bitcast [56 x i8]* @.C326_MAIN_ to i8*, !dbg !73
  %59 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !73
  call void (i8*, i8*, i64, ...) %59(i8* %57, i8* %58, i64 56), !dbg !73
  %60 = bitcast i32* @.C329_MAIN_ to i8*, !dbg !73
  %61 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !73
  %62 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !73
  %63 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !73
  %64 = call i32 (i8*, i8*, i8*, i8*, ...) %63(i8* %60, i8* null, i8* %61, i8* %62), !dbg !73
  call void @llvm.dbg.declare(metadata i32* %z__io_331, metadata !74, metadata !DIExpression()), !dbg !65
  store i32 %64, i32* %z__io_331, align 4, !dbg !73
  %65 = bitcast %struct_drb048_0_* @_drb048_0_ to i8*, !dbg !73
  %66 = getelementptr i8, i8* %65, i64 72, !dbg !73
  %67 = bitcast i8* %66 to i64*, !dbg !73
  %68 = load i64, i64* %67, align 8, !dbg !73
  %69 = bitcast %struct_drb048_0_* @_drb048_0_ to i8**, !dbg !73
  %70 = load i8*, i8** %69, align 8, !dbg !73
  %71 = getelementptr i8, i8* %70, i64 196, !dbg !73
  %72 = bitcast i8* %71 to i32*, !dbg !73
  %73 = getelementptr i32, i32* %72, i64 %68, !dbg !73
  %74 = load i32, i32* %73, align 4, !dbg !73
  %75 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !73
  %76 = call i32 (i32, i32, ...) %75(i32 %74, i32 25), !dbg !73
  store i32 %76, i32* %z__io_331, align 4, !dbg !73
  %77 = call i32 (...) @f90io_ldw_end(), !dbg !73
  store i32 %77, i32* %z__io_331, align 4, !dbg !73
  call void (...) @_mp_ecs_nest(), !dbg !73
  ret void, !dbg !75
}

declare void @_mp_ecs_nest(...) #1

declare signext i32 @f90io_ldw_end(...) #1

declare signext i32 @f90io_sc_i_ldw(...) #1

declare signext i32 @f90io_print_init(...) #1

declare void @f90io_src_info03a(...) #1

declare void @_mp_bcs_nest(...) #1

declare void @f90_alloc04_chka_i8(...) #1

declare void @f90_set_intrin_type_i8(...) #1

declare void @f90_template1_i8(...) #1

declare void @fort_init(...) #1

declare void @__kmpc_for_static_fini(i64*, i32) #1

declare void @__kmpc_for_static_init_4(i64*, i32, i32, i64*, i64*, i64*, i64*, i32, i32) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

declare signext i32 @__kmpc_global_thread_num(i64*) #1

declare void @__kmpc_fork_call(i64*, i32, i64*, i64*, ...) #1

declare void @__kmpc_end_serialized_parallel(i64*, i32) #1

declare void @__kmpc_serialized_parallel(i64*, i32) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { noinline }
attributes #1 = { "no-frame-pointer-elim-non-leaf" }
attributes #2 = { nounwind readnone speculatable }

!llvm.module.flags = !{!24, !25}
!llvm.dbg.cu = !{!3}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "z_b_0", scope: !2, file: !4, type: !16, isLocal: false, isDefinition: true)
!2 = !DIModule(scope: !3, name: "drb048")
!3 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !4, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !6, imports: !19)
!4 = !DIFile(filename: "micro-benchmarks-fortran/DRB048-firstprivate-orig-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
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
!20 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !21, entity: !2, file: !4, line: 31)
!21 = distinct !DISubprogram(name: "drb048_firstprivate_orig_no", scope: !3, file: !4, line: 31, type: !22, scopeLine: 31, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !3)
!22 = !DISubroutineType(cc: DW_CC_program, types: !23)
!23 = !{null}
!24 = !{i32 2, !"Dwarf Version", i32 4}
!25 = !{i32 2, !"Debug Info Version", i32 3}
!26 = distinct !DISubprogram(name: "foo", scope: !2, file: !4, line: 16, type: !27, scopeLine: 16, spFlags: DISPFlagDefinition, unit: !3)
!27 = !DISubroutineType(types: !28)
!28 = !{null, !9, !10, !10, !15}
!29 = !DILocalVariable(arg: 1, scope: !26, file: !4, type: !9, flags: DIFlagArtificial)
!30 = !DILocation(line: 0, scope: !26)
!31 = !DILocalVariable(name: "_V_n", scope: !26, file: !4, type: !10)
!32 = !DILocalVariable(name: "_V_n", arg: 2, scope: !26, file: !4, type: !10)
!33 = !DILocalVariable(name: "_V_g", scope: !26, file: !4, type: !10)
!34 = !DILocalVariable(name: "_V_g", arg: 3, scope: !26, file: !4, type: !10)
!35 = !DILocalVariable(arg: 4, scope: !26, file: !4, type: !15, flags: DIFlagArtificial)
!36 = !DILocalVariable(name: "omp_sched_static", scope: !26, file: !4, type: !10)
!37 = !DILocalVariable(name: "omp_proc_bind_false", scope: !26, file: !4, type: !10)
!38 = !DILocalVariable(name: "omp_proc_bind_true", scope: !26, file: !4, type: !10)
!39 = !DILocalVariable(name: "omp_lock_hint_none", scope: !26, file: !4, type: !10)
!40 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !26, file: !4, type: !10)
!41 = !DILocation(line: 16, column: 1, scope: !26)
!42 = !DILocalVariable(name: "n", scope: !26, file: !4, type: !10)
!43 = !DILocalVariable(name: "g", scope: !26, file: !4, type: !10)
!44 = !DILocation(line: 28, column: 1, scope: !26)
!45 = !DILocation(line: 23, column: 1, scope: !26)
!46 = distinct !DISubprogram(name: "__nv_drb048_foo__F1L23_1", scope: !3, file: !4, line: 23, type: !47, scopeLine: 23, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !3)
!47 = !DISubroutineType(types: !48)
!48 = !{null, !10, !16, !16}
!49 = !DILocalVariable(name: "__nv_drb048_foo__F1L23_1Arg0", arg: 1, scope: !46, file: !4, type: !10)
!50 = !DILocation(line: 0, scope: !46)
!51 = !DILocalVariable(name: "__nv_drb048_foo__F1L23_1Arg1", arg: 2, scope: !46, file: !4, type: !16)
!52 = !DILocalVariable(name: "__nv_drb048_foo__F1L23_1Arg2", arg: 3, scope: !46, file: !4, type: !16)
!53 = !DILocalVariable(name: "omp_sched_static", scope: !46, file: !4, type: !10)
!54 = !DILocalVariable(name: "omp_proc_bind_false", scope: !46, file: !4, type: !10)
!55 = !DILocalVariable(name: "omp_proc_bind_true", scope: !46, file: !4, type: !10)
!56 = !DILocalVariable(name: "omp_lock_hint_none", scope: !46, file: !4, type: !10)
!57 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !46, file: !4, type: !10)
!58 = !DILocation(line: 26, column: 1, scope: !46)
!59 = !DILocation(line: 23, column: 1, scope: !46)
!60 = !DILocalVariable(name: "g", scope: !46, file: !4, type: !10)
!61 = !DILocation(line: 24, column: 1, scope: !46)
!62 = !DILocalVariable(name: "i", scope: !46, file: !4, type: !10)
!63 = !DILocation(line: 25, column: 1, scope: !46)
!64 = !DILocalVariable(name: "omp_sched_static", scope: !21, file: !4, type: !10)
!65 = !DILocation(line: 0, scope: !21)
!66 = !DILocalVariable(name: "omp_proc_bind_false", scope: !21, file: !4, type: !10)
!67 = !DILocalVariable(name: "omp_proc_bind_true", scope: !21, file: !4, type: !10)
!68 = !DILocalVariable(name: "omp_lock_hint_none", scope: !21, file: !4, type: !10)
!69 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !21, file: !4, type: !10)
!70 = !DILocation(line: 31, column: 1, scope: !21)
!71 = !DILocation(line: 36, column: 1, scope: !21)
!72 = !DILocation(line: 37, column: 1, scope: !21)
!73 = !DILocation(line: 38, column: 1, scope: !21)
!74 = !DILocalVariable(scope: !21, file: !4, type: !10, flags: DIFlagArtificial)
!75 = !DILocation(line: 39, column: 1, scope: !21)

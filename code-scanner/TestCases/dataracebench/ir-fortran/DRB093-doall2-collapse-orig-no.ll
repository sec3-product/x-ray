; ModuleID = '/tmp/DRB093-doall2-collapse-orig-no-15a787.ll'
source_filename = "/tmp/DRB093-doall2-collapse-orig-no-15a787.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct_drb093_2_ = type <{ [16 x i8] }>
%struct_drb093_0_ = type <{ [192 x i8] }>
%astruct.dt64 = type <{ i8*, i8*, i8*, i8*, i8* }>

@.C285_MAIN_ = internal constant i32 1
@.C306_MAIN_ = internal constant i32 25
@.C341_MAIN_ = internal constant i64 4
@.C340_MAIN_ = internal constant i64 25
@.C284_MAIN_ = internal constant i64 0
@.C311_MAIN_ = internal constant i64 18
@.C310_MAIN_ = internal constant i64 17
@.C309_MAIN_ = internal constant i64 12
@.C286_MAIN_ = internal constant i64 1
@.C308_MAIN_ = internal constant i64 11
@.C322_MAIN_ = internal constant i32 100
@.C283_MAIN_ = internal constant i32 0
@.C285___nv_MAIN__F1L28_1 = internal constant i32 1
@.C286___nv_MAIN__F1L28_1 = internal constant i64 1
@.C283___nv_MAIN__F1L28_1 = internal constant i32 0
@_drb093_2_ = common global %struct_drb093_2_ zeroinitializer, align 64, !dbg !0, !dbg !19
@_drb093_0_ = common global %struct_drb093_0_ zeroinitializer, align 64, !dbg !7, !dbg !13

; Function Attrs: noinline
define float @drb093_() #0 {
.L.entry:
  ret float undef
}

define void @MAIN_() #1 !dbg !23 {
L.entry:
  %__gtid_MAIN__418 = alloca i32, align 4
  %len_323 = alloca i32, align 4
  %.g0000_394 = alloca i64, align 8
  %.g0001_396 = alloca i64, align 8
  %.uplevelArgPack0001_404 = alloca %astruct.dt64, align 16
  %j_321 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !28, metadata !DIExpression()), !dbg !29
  call void @llvm.dbg.value(metadata i32 0, metadata !30, metadata !DIExpression()), !dbg !29
  call void @llvm.dbg.value(metadata i32 1, metadata !31, metadata !DIExpression()), !dbg !29
  call void @llvm.dbg.value(metadata i32 0, metadata !32, metadata !DIExpression()), !dbg !29
  call void @llvm.dbg.value(metadata i32 1, metadata !33, metadata !DIExpression()), !dbg !29
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !34
  store i32 %0, i32* %__gtid_MAIN__418, align 4, !dbg !34
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !35
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !35
  call void (i8*, ...) %2(i8* %1), !dbg !35
  br label %L.LB2_361

L.LB2_361:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %len_323, metadata !36, metadata !DIExpression()), !dbg !29
  store i32 100, i32* %len_323, align 4, !dbg !37
  %3 = bitcast %struct_drb093_0_* @_drb093_0_ to i8*, !dbg !38
  %4 = getelementptr i8, i8* %3, i64 96, !dbg !38
  %5 = bitcast i8* %4 to i64*, !dbg !38
  store i64 1, i64* %5, align 8, !dbg !38
  %6 = load i32, i32* %len_323, align 4, !dbg !38
  call void @llvm.dbg.value(metadata i32 %6, metadata !36, metadata !DIExpression()), !dbg !29
  %7 = sext i32 %6 to i64, !dbg !38
  %8 = bitcast %struct_drb093_0_* @_drb093_0_ to i8*, !dbg !38
  %9 = getelementptr i8, i8* %8, i64 104, !dbg !38
  %10 = bitcast i8* %9 to i64*, !dbg !38
  store i64 %7, i64* %10, align 8, !dbg !38
  %11 = bitcast %struct_drb093_0_* @_drb093_0_ to i8*, !dbg !38
  %12 = getelementptr i8, i8* %11, i64 144, !dbg !38
  %13 = bitcast i8* %12 to i64*, !dbg !38
  store i64 1, i64* %13, align 8, !dbg !38
  %14 = load i32, i32* %len_323, align 4, !dbg !38
  call void @llvm.dbg.value(metadata i32 %14, metadata !36, metadata !DIExpression()), !dbg !29
  %15 = sext i32 %14 to i64, !dbg !38
  %16 = bitcast %struct_drb093_0_* @_drb093_0_ to i8*, !dbg !38
  %17 = getelementptr i8, i8* %16, i64 152, !dbg !38
  %18 = bitcast i8* %17 to i64*, !dbg !38
  store i64 %15, i64* %18, align 8, !dbg !38
  %19 = bitcast %struct_drb093_0_* @_drb093_0_ to i8*, !dbg !38
  %20 = getelementptr i8, i8* %19, i64 104, !dbg !38
  %21 = bitcast i8* %20 to i64*, !dbg !38
  %22 = load i64, i64* %21, align 8, !dbg !38
  %23 = sub nsw i64 %22, 1, !dbg !38
  %24 = bitcast %struct_drb093_0_* @_drb093_0_ to i8*, !dbg !38
  %25 = getelementptr i8, i8* %24, i64 96, !dbg !38
  %26 = bitcast i8* %25 to i64*, !dbg !38
  %27 = load i64, i64* %26, align 8, !dbg !38
  %28 = add nsw i64 %23, %27, !dbg !38
  store i64 %28, i64* %.g0000_394, align 8, !dbg !38
  %29 = bitcast %struct_drb093_0_* @_drb093_0_ to i8*, !dbg !38
  %30 = getelementptr i8, i8* %29, i64 152, !dbg !38
  %31 = bitcast i8* %30 to i64*, !dbg !38
  %32 = load i64, i64* %31, align 8, !dbg !38
  %33 = sub nsw i64 %32, 1, !dbg !38
  %34 = bitcast %struct_drb093_0_* @_drb093_0_ to i8*, !dbg !38
  %35 = getelementptr i8, i8* %34, i64 144, !dbg !38
  %36 = bitcast i8* %35 to i64*, !dbg !38
  %37 = load i64, i64* %36, align 8, !dbg !38
  %38 = add nsw i64 %33, %37, !dbg !38
  store i64 %38, i64* %.g0001_396, align 8, !dbg !38
  %39 = bitcast %struct_drb093_0_* @_drb093_0_ to i8*, !dbg !38
  %40 = getelementptr i8, i8* %39, i64 16, !dbg !38
  %41 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !38
  %42 = bitcast i64* @.C340_MAIN_ to i8*, !dbg !38
  %43 = bitcast i64* @.C341_MAIN_ to i8*, !dbg !38
  %44 = bitcast %struct_drb093_0_* @_drb093_0_ to i8*, !dbg !38
  %45 = getelementptr i8, i8* %44, i64 96, !dbg !38
  %46 = bitcast i64* %.g0000_394 to i8*, !dbg !38
  %47 = bitcast %struct_drb093_0_* @_drb093_0_ to i8*, !dbg !38
  %48 = getelementptr i8, i8* %47, i64 144, !dbg !38
  %49 = bitcast i64* %.g0001_396 to i8*, !dbg !38
  %50 = bitcast void (...)* @f90_template2_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !38
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, ...) %50(i8* %40, i8* %41, i8* %42, i8* %43, i8* %45, i8* %46, i8* %48, i8* %49), !dbg !38
  %51 = bitcast %struct_drb093_0_* @_drb093_0_ to i8*, !dbg !38
  %52 = getelementptr i8, i8* %51, i64 16, !dbg !38
  %53 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !38
  call void (i8*, i32, ...) %53(i8* %52, i32 25), !dbg !38
  %54 = bitcast %struct_drb093_0_* @_drb093_0_ to i8*, !dbg !38
  %55 = getelementptr i8, i8* %54, i64 104, !dbg !38
  %56 = bitcast i8* %55 to i64*, !dbg !38
  %57 = load i64, i64* %56, align 8, !dbg !38
  %58 = sub nsw i64 %57, 1, !dbg !38
  %59 = bitcast %struct_drb093_0_* @_drb093_0_ to i8*, !dbg !38
  %60 = getelementptr i8, i8* %59, i64 96, !dbg !38
  %61 = bitcast i8* %60 to i64*, !dbg !38
  %62 = load i64, i64* %61, align 8, !dbg !38
  %63 = add nsw i64 %58, %62, !dbg !38
  %64 = bitcast %struct_drb093_0_* @_drb093_0_ to i8*, !dbg !38
  %65 = getelementptr i8, i8* %64, i64 96, !dbg !38
  %66 = bitcast i8* %65 to i64*, !dbg !38
  %67 = load i64, i64* %66, align 8, !dbg !38
  %68 = sub nsw i64 %67, 1, !dbg !38
  %69 = sub nsw i64 %63, %68, !dbg !38
  %70 = bitcast %struct_drb093_0_* @_drb093_0_ to i8*, !dbg !38
  %71 = getelementptr i8, i8* %70, i64 152, !dbg !38
  %72 = bitcast i8* %71 to i64*, !dbg !38
  %73 = load i64, i64* %72, align 8, !dbg !38
  %74 = sub nsw i64 %73, 1, !dbg !38
  %75 = bitcast %struct_drb093_0_* @_drb093_0_ to i8*, !dbg !38
  %76 = getelementptr i8, i8* %75, i64 144, !dbg !38
  %77 = bitcast i8* %76 to i64*, !dbg !38
  %78 = load i64, i64* %77, align 8, !dbg !38
  %79 = add nsw i64 %74, %78, !dbg !38
  %80 = bitcast %struct_drb093_0_* @_drb093_0_ to i8*, !dbg !38
  %81 = getelementptr i8, i8* %80, i64 144, !dbg !38
  %82 = bitcast i8* %81 to i64*, !dbg !38
  %83 = load i64, i64* %82, align 8, !dbg !38
  %84 = sub nsw i64 %83, 1, !dbg !38
  %85 = sub nsw i64 %79, %84, !dbg !38
  %86 = mul nsw i64 %69, %85, !dbg !38
  store i64 %86, i64* %.g0000_394, align 8, !dbg !38
  %87 = bitcast i64* %.g0000_394 to i8*, !dbg !38
  %88 = bitcast i64* @.C340_MAIN_ to i8*, !dbg !38
  %89 = bitcast i64* @.C341_MAIN_ to i8*, !dbg !38
  %90 = bitcast %struct_drb093_0_* @_drb093_0_ to i8*, !dbg !38
  %91 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !38
  %92 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !38
  %93 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !38
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %93(i8* %87, i8* %88, i8* %89, i8* null, i8* %90, i8* null, i8* %91, i8* %92, i8* null, i64 0), !dbg !38
  %94 = bitcast i32* %len_323 to i8*, !dbg !39
  %95 = bitcast %astruct.dt64* %.uplevelArgPack0001_404 to i8**, !dbg !39
  store i8* %94, i8** %95, align 8, !dbg !39
  call void @llvm.dbg.declare(metadata i32* %j_321, metadata !40, metadata !DIExpression()), !dbg !29
  %96 = bitcast i32* %j_321 to i8*, !dbg !39
  %97 = bitcast %astruct.dt64* %.uplevelArgPack0001_404 to i8*, !dbg !39
  %98 = getelementptr i8, i8* %97, i64 8, !dbg !39
  %99 = bitcast i8* %98 to i8**, !dbg !39
  store i8* %96, i8** %99, align 8, !dbg !39
  %100 = bitcast %struct_drb093_0_* @_drb093_0_ to i8*, !dbg !39
  %101 = bitcast %astruct.dt64* %.uplevelArgPack0001_404 to i8*, !dbg !39
  %102 = getelementptr i8, i8* %101, i64 16, !dbg !39
  %103 = bitcast i8* %102 to i8**, !dbg !39
  store i8* %100, i8** %103, align 8, !dbg !39
  %104 = bitcast %struct_drb093_0_* @_drb093_0_ to i8*, !dbg !39
  %105 = getelementptr i8, i8* %104, i64 16, !dbg !39
  %106 = bitcast %astruct.dt64* %.uplevelArgPack0001_404 to i8*, !dbg !39
  %107 = getelementptr i8, i8* %106, i64 24, !dbg !39
  %108 = bitcast i8* %107 to i8**, !dbg !39
  store i8* %105, i8** %108, align 8, !dbg !39
  %109 = bitcast %struct_drb093_0_* @_drb093_0_ to i8*, !dbg !39
  %110 = bitcast %astruct.dt64* %.uplevelArgPack0001_404 to i8*, !dbg !39
  %111 = getelementptr i8, i8* %110, i64 32, !dbg !39
  %112 = bitcast i8* %111 to i8**, !dbg !39
  store i8* %109, i8** %112, align 8, !dbg !39
  br label %L.LB2_416, !dbg !39

L.LB2_416:                                        ; preds = %L.LB2_361
  %113 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L28_1_ to i64*, !dbg !39
  %114 = bitcast %astruct.dt64* %.uplevelArgPack0001_404 to i64*, !dbg !39
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %113, i64* %114), !dbg !39
  ret void, !dbg !34
}

define internal void @__nv_MAIN__F1L28_1_(i32* %__nv_MAIN__F1L28_1Arg0, i64* %__nv_MAIN__F1L28_1Arg1, i64* %__nv_MAIN__F1L28_1Arg2) #1 !dbg !41 {
L.entry:
  %__gtid___nv_MAIN__F1L28_1__460 = alloca i32, align 4
  %.i0000p_328 = alloca i32, align 4
  %.Xc0000p_329 = alloca i64, align 8
  %.Xd0000p_330 = alloca i64, align 8
  %.Xc0001p_336 = alloca i64, align 8
  %.i0001p_337 = alloca i32, align 4
  %.id0000p_331 = alloca i64, align 8
  %.du0001p_351 = alloca i64, align 8
  %.de0001p_352 = alloca i64, align 8
  %.di0001p_353 = alloca i64, align 8
  %.ds0001p_354 = alloca i64, align 8
  %.dl0001p_356 = alloca i64, align 8
  %.dl0001p.copy_454 = alloca i64, align 8
  %.de0001p.copy_455 = alloca i64, align 8
  %.ds0001p.copy_456 = alloca i64, align 8
  %.dX0001p_355 = alloca i64, align 8
  %.dY0001p_350 = alloca i64, align 8
  %.Xg0000p_334 = alloca i64, align 8
  %.Xe0000p_332 = alloca i64, align 8
  %.Xf0000p_333 = alloca i64, align 8
  %j_335 = alloca i32, align 4
  %i_327 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L28_1Arg0, metadata !44, metadata !DIExpression()), !dbg !45
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L28_1Arg1, metadata !46, metadata !DIExpression()), !dbg !45
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L28_1Arg2, metadata !47, metadata !DIExpression()), !dbg !45
  call void @llvm.dbg.value(metadata i32 1, metadata !48, metadata !DIExpression()), !dbg !45
  call void @llvm.dbg.value(metadata i32 0, metadata !49, metadata !DIExpression()), !dbg !45
  call void @llvm.dbg.value(metadata i32 1, metadata !50, metadata !DIExpression()), !dbg !45
  call void @llvm.dbg.value(metadata i32 0, metadata !51, metadata !DIExpression()), !dbg !45
  call void @llvm.dbg.value(metadata i32 1, metadata !52, metadata !DIExpression()), !dbg !45
  %0 = load i32, i32* %__nv_MAIN__F1L28_1Arg0, align 4, !dbg !53
  store i32 %0, i32* %__gtid___nv_MAIN__F1L28_1__460, align 4, !dbg !53
  br label %L.LB3_441

L.LB3_441:                                        ; preds = %L.entry
  br label %L.LB3_326

L.LB3_326:                                        ; preds = %L.LB3_441
  store i32 0, i32* %.i0000p_328, align 4, !dbg !54
  %1 = bitcast i64* %__nv_MAIN__F1L28_1Arg2 to i32**, !dbg !54
  %2 = load i32*, i32** %1, align 8, !dbg !54
  %3 = load i32, i32* %2, align 4, !dbg !54
  %4 = sext i32 %3 to i64, !dbg !54
  store i64 %4, i64* %.Xc0000p_329, align 8, !dbg !54
  %5 = load i64, i64* %.Xc0000p_329, align 8, !dbg !54
  store i64 %5, i64* %.Xd0000p_330, align 8, !dbg !54
  %6 = bitcast i64* %__nv_MAIN__F1L28_1Arg2 to i32**, !dbg !55
  %7 = load i32*, i32** %6, align 8, !dbg !55
  %8 = load i32, i32* %7, align 4, !dbg !55
  %9 = sext i32 %8 to i64, !dbg !55
  store i64 %9, i64* %.Xc0001p_336, align 8, !dbg !55
  %10 = load i64, i64* %.Xc0001p_336, align 8, !dbg !55
  %11 = load i64, i64* %.Xd0000p_330, align 8, !dbg !55
  %12 = mul nsw i64 %10, %11, !dbg !55
  store i64 %12, i64* %.Xd0000p_330, align 8, !dbg !55
  store i32 0, i32* %.i0001p_337, align 4, !dbg !55
  store i64 1, i64* %.id0000p_331, align 8, !dbg !55
  %13 = load i64, i64* %.Xd0000p_330, align 8, !dbg !55
  store i64 %13, i64* %.du0001p_351, align 8, !dbg !55
  %14 = load i64, i64* %.Xd0000p_330, align 8, !dbg !55
  store i64 %14, i64* %.de0001p_352, align 8, !dbg !55
  store i64 1, i64* %.di0001p_353, align 8, !dbg !55
  %15 = load i64, i64* %.di0001p_353, align 8, !dbg !55
  store i64 %15, i64* %.ds0001p_354, align 8, !dbg !55
  store i64 1, i64* %.dl0001p_356, align 8, !dbg !55
  %16 = load i64, i64* %.dl0001p_356, align 8, !dbg !55
  store i64 %16, i64* %.dl0001p.copy_454, align 8, !dbg !55
  %17 = load i64, i64* %.de0001p_352, align 8, !dbg !55
  store i64 %17, i64* %.de0001p.copy_455, align 8, !dbg !55
  %18 = load i64, i64* %.ds0001p_354, align 8, !dbg !55
  store i64 %18, i64* %.ds0001p.copy_456, align 8, !dbg !55
  %19 = load i32, i32* %__gtid___nv_MAIN__F1L28_1__460, align 4, !dbg !55
  %20 = bitcast i32* %.i0001p_337 to i64*, !dbg !55
  %21 = load i64, i64* %.ds0001p.copy_456, align 8, !dbg !55
  call void @__kmpc_for_static_init_8(i64* null, i32 %19, i32 34, i64* %20, i64* %.dl0001p.copy_454, i64* %.de0001p.copy_455, i64* %.ds0001p.copy_456, i64 %21, i64 1), !dbg !55
  %22 = load i64, i64* %.dl0001p.copy_454, align 8, !dbg !55
  store i64 %22, i64* %.dl0001p_356, align 8, !dbg !55
  %23 = load i64, i64* %.de0001p.copy_455, align 8, !dbg !55
  store i64 %23, i64* %.de0001p_352, align 8, !dbg !55
  %24 = load i64, i64* %.ds0001p.copy_456, align 8, !dbg !55
  store i64 %24, i64* %.ds0001p_354, align 8, !dbg !55
  %25 = load i64, i64* %.dl0001p_356, align 8, !dbg !55
  store i64 %25, i64* %.id0000p_331, align 8, !dbg !55
  %26 = load i64, i64* %.id0000p_331, align 8, !dbg !55
  store i64 %26, i64* %.dX0001p_355, align 8, !dbg !55
  %27 = load i64, i64* %.dX0001p_355, align 8, !dbg !55
  %28 = load i64, i64* %.du0001p_351, align 8, !dbg !55
  %29 = icmp sgt i64 %27, %28, !dbg !55
  br i1 %29, label %L.LB3_349, label %L.LB3_494, !dbg !55

L.LB3_494:                                        ; preds = %L.LB3_326
  %30 = load i64, i64* %.dX0001p_355, align 8, !dbg !55
  store i64 %30, i64* %.id0000p_331, align 8, !dbg !55
  %31 = load i64, i64* %.di0001p_353, align 8, !dbg !55
  %32 = load i64, i64* %.de0001p_352, align 8, !dbg !55
  %33 = load i64, i64* %.dX0001p_355, align 8, !dbg !55
  %34 = sub nsw i64 %32, %33, !dbg !55
  %35 = add nsw i64 %31, %34, !dbg !55
  %36 = load i64, i64* %.di0001p_353, align 8, !dbg !55
  %37 = sdiv i64 %35, %36, !dbg !55
  store i64 %37, i64* %.dY0001p_350, align 8, !dbg !55
  %38 = load i64, i64* %.dY0001p_350, align 8, !dbg !55
  %39 = icmp sle i64 %38, 0, !dbg !55
  br i1 %39, label %L.LB3_359, label %L.LB3_358, !dbg !55

L.LB3_358:                                        ; preds = %L.LB3_358, %L.LB3_494
  %40 = load i64, i64* %.id0000p_331, align 8, !dbg !55
  %41 = sub nsw i64 %40, 1, !dbg !55
  store i64 %41, i64* %.Xg0000p_334, align 8, !dbg !55
  %42 = load i64, i64* %.Xg0000p_334, align 8, !dbg !55
  %43 = load i64, i64* %.Xc0001p_336, align 8, !dbg !55
  %44 = sdiv i64 %42, %43, !dbg !55
  store i64 %44, i64* %.Xe0000p_332, align 8, !dbg !55
  %45 = load i64, i64* %.Xg0000p_334, align 8, !dbg !55
  %46 = load i64, i64* %.Xc0001p_336, align 8, !dbg !55
  %47 = load i64, i64* %.Xe0000p_332, align 8, !dbg !55
  %48 = mul nsw i64 %46, %47, !dbg !55
  %49 = sub nsw i64 %45, %48, !dbg !55
  store i64 %49, i64* %.Xf0000p_333, align 8, !dbg !55
  %50 = load i64, i64* %.Xf0000p_333, align 8, !dbg !55
  %51 = trunc i64 %50 to i32, !dbg !55
  %52 = add nsw i32 %51, 1, !dbg !55
  call void @llvm.dbg.declare(metadata i32* %j_335, metadata !56, metadata !DIExpression()), !dbg !53
  store i32 %52, i32* %j_335, align 4, !dbg !55
  %53 = load i64, i64* %.Xe0000p_332, align 8, !dbg !55
  store i64 %53, i64* %.Xg0000p_334, align 8, !dbg !55
  %54 = load i64, i64* %.Xg0000p_334, align 8, !dbg !55
  %55 = load i64, i64* %.Xc0000p_329, align 8, !dbg !55
  %56 = sdiv i64 %54, %55, !dbg !55
  store i64 %56, i64* %.Xe0000p_332, align 8, !dbg !55
  %57 = load i64, i64* %.Xg0000p_334, align 8, !dbg !55
  %58 = load i64, i64* %.Xc0000p_329, align 8, !dbg !55
  %59 = load i64, i64* %.Xe0000p_332, align 8, !dbg !55
  %60 = mul nsw i64 %58, %59, !dbg !55
  %61 = sub nsw i64 %57, %60, !dbg !55
  store i64 %61, i64* %.Xf0000p_333, align 8, !dbg !55
  %62 = load i64, i64* %.Xf0000p_333, align 8, !dbg !55
  %63 = trunc i64 %62 to i32, !dbg !55
  %64 = add nsw i32 %63, 1, !dbg !55
  call void @llvm.dbg.declare(metadata i32* %i_327, metadata !57, metadata !DIExpression()), !dbg !53
  store i32 %64, i32* %i_327, align 4, !dbg !55
  %65 = load i32, i32* %i_327, align 4, !dbg !58
  call void @llvm.dbg.value(metadata i32 %65, metadata !57, metadata !DIExpression()), !dbg !53
  %66 = sext i32 %65 to i64, !dbg !58
  %67 = load i32, i32* %j_335, align 4, !dbg !58
  call void @llvm.dbg.value(metadata i32 %67, metadata !56, metadata !DIExpression()), !dbg !53
  %68 = sext i32 %67 to i64, !dbg !58
  %69 = bitcast i64* %__nv_MAIN__F1L28_1Arg2 to i8*, !dbg !58
  %70 = getelementptr i8, i8* %69, i64 24, !dbg !58
  %71 = bitcast i8* %70 to i8**, !dbg !58
  %72 = load i8*, i8** %71, align 8, !dbg !58
  %73 = getelementptr i8, i8* %72, i64 160, !dbg !58
  %74 = bitcast i8* %73 to i64*, !dbg !58
  %75 = load i64, i64* %74, align 8, !dbg !58
  %76 = mul nsw i64 %68, %75, !dbg !58
  %77 = add nsw i64 %66, %76, !dbg !58
  %78 = bitcast i64* %__nv_MAIN__F1L28_1Arg2 to i8*, !dbg !58
  %79 = getelementptr i8, i8* %78, i64 24, !dbg !58
  %80 = bitcast i8* %79 to i8**, !dbg !58
  %81 = load i8*, i8** %80, align 8, !dbg !58
  %82 = getelementptr i8, i8* %81, i64 56, !dbg !58
  %83 = bitcast i8* %82 to i64*, !dbg !58
  %84 = load i64, i64* %83, align 8, !dbg !58
  %85 = add nsw i64 %77, %84, !dbg !58
  %86 = bitcast i64* %__nv_MAIN__F1L28_1Arg2 to i8*, !dbg !58
  %87 = getelementptr i8, i8* %86, i64 32, !dbg !58
  %88 = bitcast i8* %87 to i8***, !dbg !58
  %89 = load i8**, i8*** %88, align 8, !dbg !58
  %90 = load i8*, i8** %89, align 8, !dbg !58
  %91 = getelementptr i8, i8* %90, i64 -4, !dbg !58
  %92 = bitcast i8* %91 to i32*, !dbg !58
  %93 = getelementptr i32, i32* %92, i64 %85, !dbg !58
  %94 = load i32, i32* %93, align 4, !dbg !58
  %95 = add nsw i32 %94, 1, !dbg !58
  %96 = load i32, i32* %i_327, align 4, !dbg !58
  call void @llvm.dbg.value(metadata i32 %96, metadata !57, metadata !DIExpression()), !dbg !53
  %97 = sext i32 %96 to i64, !dbg !58
  %98 = load i32, i32* %j_335, align 4, !dbg !58
  call void @llvm.dbg.value(metadata i32 %98, metadata !56, metadata !DIExpression()), !dbg !53
  %99 = sext i32 %98 to i64, !dbg !58
  %100 = bitcast i64* %__nv_MAIN__F1L28_1Arg2 to i8*, !dbg !58
  %101 = getelementptr i8, i8* %100, i64 24, !dbg !58
  %102 = bitcast i8* %101 to i8**, !dbg !58
  %103 = load i8*, i8** %102, align 8, !dbg !58
  %104 = getelementptr i8, i8* %103, i64 160, !dbg !58
  %105 = bitcast i8* %104 to i64*, !dbg !58
  %106 = load i64, i64* %105, align 8, !dbg !58
  %107 = mul nsw i64 %99, %106, !dbg !58
  %108 = add nsw i64 %97, %107, !dbg !58
  %109 = bitcast i64* %__nv_MAIN__F1L28_1Arg2 to i8*, !dbg !58
  %110 = getelementptr i8, i8* %109, i64 24, !dbg !58
  %111 = bitcast i8* %110 to i8**, !dbg !58
  %112 = load i8*, i8** %111, align 8, !dbg !58
  %113 = getelementptr i8, i8* %112, i64 56, !dbg !58
  %114 = bitcast i8* %113 to i64*, !dbg !58
  %115 = load i64, i64* %114, align 8, !dbg !58
  %116 = add nsw i64 %108, %115, !dbg !58
  %117 = bitcast i64* %__nv_MAIN__F1L28_1Arg2 to i8*, !dbg !58
  %118 = getelementptr i8, i8* %117, i64 32, !dbg !58
  %119 = bitcast i8* %118 to i8***, !dbg !58
  %120 = load i8**, i8*** %119, align 8, !dbg !58
  %121 = load i8*, i8** %120, align 8, !dbg !58
  %122 = getelementptr i8, i8* %121, i64 -4, !dbg !58
  %123 = bitcast i8* %122 to i32*, !dbg !58
  %124 = getelementptr i32, i32* %123, i64 %116, !dbg !58
  store i32 %95, i32* %124, align 4, !dbg !58
  %125 = load i64, i64* %.di0001p_353, align 8, !dbg !53
  %126 = load i64, i64* %.id0000p_331, align 8, !dbg !53
  %127 = add nsw i64 %125, %126, !dbg !53
  store i64 %127, i64* %.id0000p_331, align 8, !dbg !53
  %128 = load i64, i64* %.dY0001p_350, align 8, !dbg !53
  %129 = sub nsw i64 %128, 1, !dbg !53
  store i64 %129, i64* %.dY0001p_350, align 8, !dbg !53
  %130 = load i64, i64* %.dY0001p_350, align 8, !dbg !53
  %131 = icmp sgt i64 %130, 0, !dbg !53
  br i1 %131, label %L.LB3_358, label %L.LB3_359, !dbg !53

L.LB3_359:                                        ; preds = %L.LB3_358, %L.LB3_494
  br label %L.LB3_349

L.LB3_349:                                        ; preds = %L.LB3_359, %L.LB3_326
  %132 = load i32, i32* %__gtid___nv_MAIN__F1L28_1__460, align 4, !dbg !53
  call void @__kmpc_for_static_fini(i64* null, i32 %132), !dbg !53
  br label %L.LB3_338

L.LB3_338:                                        ; preds = %L.LB3_349
  ret void, !dbg !53
}

declare void @__kmpc_for_static_fini(i64*, i32) #1

declare void @__kmpc_for_static_init_8(i64*, i32, i32, i64*, i64*, i64*, i64*, i64, i64) #1

declare void @f90_alloc04_chka_i8(...) #1

declare void @f90_set_intrin_type_i8(...) #1

declare void @f90_template2_i8(...) #1

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
!1 = distinct !DIGlobalVariable(name: "z_b_0", scope: !2, file: !4, type: !16, isLocal: false, isDefinition: true)
!2 = !DIModule(scope: !3, name: "drb093")
!3 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !4, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !6, imports: !21)
!4 = !DIFile(filename: "micro-benchmarks-fortran/DRB093-doall2-collapse-orig-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!5 = !{}
!6 = !{!7, !13, !0, !19}
!7 = !DIGlobalVariableExpression(var: !8, expr: !DIExpression(DW_OP_deref))
!8 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !4, type: !9, isLocal: false, isDefinition: true)
!9 = !DICompositeType(tag: DW_TAG_array_type, baseType: !10, size: 32, align: 32, elements: !11)
!10 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!11 = !{!12, !12}
!12 = !DISubrange(count: 0, lowerBound: 1)
!13 = !DIGlobalVariableExpression(var: !14, expr: !DIExpression(DW_OP_plus_uconst, 16))
!14 = distinct !DIGlobalVariable(name: "a$sd", scope: !2, file: !4, type: !15, isLocal: false, isDefinition: true)
!15 = !DICompositeType(tag: DW_TAG_array_type, baseType: !16, size: 1408, align: 64, elements: !17)
!16 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!17 = !{!18}
!18 = !DISubrange(count: 22, lowerBound: 1)
!19 = !DIGlobalVariableExpression(var: !20, expr: !DIExpression(DW_OP_plus_uconst, 8))
!20 = distinct !DIGlobalVariable(name: "z_b_1", scope: !2, file: !4, type: !16, isLocal: false, isDefinition: true)
!21 = !{!22}
!22 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !23, entity: !2, file: !4, line: 18)
!23 = distinct !DISubprogram(name: "drb093_doall2_collapse_orig_no", scope: !3, file: !4, line: 18, type: !24, scopeLine: 18, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !3)
!24 = !DISubroutineType(cc: DW_CC_program, types: !25)
!25 = !{null}
!26 = !{i32 2, !"Dwarf Version", i32 4}
!27 = !{i32 2, !"Debug Info Version", i32 3}
!28 = !DILocalVariable(name: "omp_sched_static", scope: !23, file: !4, type: !10)
!29 = !DILocation(line: 0, scope: !23)
!30 = !DILocalVariable(name: "omp_proc_bind_false", scope: !23, file: !4, type: !10)
!31 = !DILocalVariable(name: "omp_proc_bind_true", scope: !23, file: !4, type: !10)
!32 = !DILocalVariable(name: "omp_lock_hint_none", scope: !23, file: !4, type: !10)
!33 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !23, file: !4, type: !10)
!34 = !DILocation(line: 35, column: 1, scope: !23)
!35 = !DILocation(line: 18, column: 1, scope: !23)
!36 = !DILocalVariable(name: "len", scope: !23, file: !4, type: !10)
!37 = !DILocation(line: 24, column: 1, scope: !23)
!38 = !DILocation(line: 26, column: 1, scope: !23)
!39 = !DILocation(line: 28, column: 1, scope: !23)
!40 = !DILocalVariable(name: "j", scope: !23, file: !4, type: !10)
!41 = distinct !DISubprogram(name: "__nv_MAIN__F1L28_1", scope: !3, file: !4, line: 28, type: !42, scopeLine: 28, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !3)
!42 = !DISubroutineType(types: !43)
!43 = !{null, !10, !16, !16}
!44 = !DILocalVariable(name: "__nv_MAIN__F1L28_1Arg0", arg: 1, scope: !41, file: !4, type: !10)
!45 = !DILocation(line: 0, scope: !41)
!46 = !DILocalVariable(name: "__nv_MAIN__F1L28_1Arg1", arg: 2, scope: !41, file: !4, type: !16)
!47 = !DILocalVariable(name: "__nv_MAIN__F1L28_1Arg2", arg: 3, scope: !41, file: !4, type: !16)
!48 = !DILocalVariable(name: "omp_sched_static", scope: !41, file: !4, type: !10)
!49 = !DILocalVariable(name: "omp_proc_bind_false", scope: !41, file: !4, type: !10)
!50 = !DILocalVariable(name: "omp_proc_bind_true", scope: !41, file: !4, type: !10)
!51 = !DILocalVariable(name: "omp_lock_hint_none", scope: !41, file: !4, type: !10)
!52 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !41, file: !4, type: !10)
!53 = !DILocation(line: 33, column: 1, scope: !41)
!54 = !DILocation(line: 29, column: 1, scope: !41)
!55 = !DILocation(line: 30, column: 1, scope: !41)
!56 = !DILocalVariable(name: "j", scope: !41, file: !4, type: !10)
!57 = !DILocalVariable(name: "i", scope: !41, file: !4, type: !10)
!58 = !DILocation(line: 31, column: 1, scope: !41)

; ModuleID = '/tmp/DRB159-nobarrier-orig-gpu-no-c24f35.ll'
source_filename = "/tmp/DRB159-nobarrier-orig-gpu-no-c24f35.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct_drb159_0_ = type <{ [128 x i8] }>

@.C313_MAIN_ = internal constant i32 25
@.C284_MAIN_ = internal constant i64 0
@.C346_MAIN_ = internal constant i32 6
@.C343_MAIN_ = internal constant [57 x i8] c"micro-benchmarks-fortran/DRB159-nobarrier-orig-gpu-no.f95"
@.C345_MAIN_ = internal constant i32 54
@.C330_MAIN_ = internal constant i32 100
@.C302_MAIN_ = internal constant i32 2
@.C301_MAIN_ = internal constant i32 8
@.C285_MAIN_ = internal constant i32 1
@.C283_MAIN_ = internal constant i32 0
@.C301___nv_MAIN__F1L29_1 = internal constant i32 8
@.C330___nv_MAIN__F1L29_1 = internal constant i32 100
@.C285___nv_MAIN__F1L29_1 = internal constant i32 1
@.C283___nv_MAIN__F1L29_1 = internal constant i32 0
@.C301___nv_MAIN_F1L30_2 = internal constant i32 8
@.C283___nv_MAIN_F1L30_2 = internal constant i32 0
@.C330___nv_MAIN_F1L30_2 = internal constant i32 100
@.C285___nv_MAIN_F1L30_2 = internal constant i32 1
@_drb159_0_ = common global %struct_drb159_0_ zeroinitializer, align 64, !dbg !0, !dbg !7, !dbg !10, !dbg !12, !dbg !14, !dbg !16, !dbg !21, !dbg !23

; Function Attrs: noinline
define float @drb159_() #0 {
.L.entry:
  ret float undef
}

define void @MAIN_() #1 !dbg !27 {
L.entry:
  %__gtid_MAIN__414 = alloca i32, align 4
  %.dY0001_358 = alloca i32, align 4
  %.dY0005_389 = alloca i32, align 4
  %.dY0006_392 = alloca i32, align 4
  %z__io_348 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 8, metadata !32, metadata !DIExpression()), !dbg !33
  call void @llvm.dbg.value(metadata i32 1, metadata !34, metadata !DIExpression()), !dbg !33
  call void @llvm.dbg.value(metadata i32 2, metadata !35, metadata !DIExpression()), !dbg !33
  call void @llvm.dbg.value(metadata i32 0, metadata !36, metadata !DIExpression()), !dbg !33
  call void @llvm.dbg.value(metadata i32 1, metadata !37, metadata !DIExpression()), !dbg !33
  call void @llvm.dbg.value(metadata i32 2, metadata !38, metadata !DIExpression()), !dbg !33
  call void @llvm.dbg.value(metadata i32 0, metadata !39, metadata !DIExpression()), !dbg !33
  call void @llvm.dbg.value(metadata i32 1, metadata !40, metadata !DIExpression()), !dbg !33
  call void @llvm.dbg.value(metadata i32 2, metadata !41, metadata !DIExpression()), !dbg !33
  call void @llvm.dbg.value(metadata i32 8, metadata !42, metadata !DIExpression()), !dbg !33
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !43
  store i32 %0, i32* %__gtid_MAIN__414, align 4, !dbg !43
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !44
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !44
  call void (i8*, ...) %2(i8* %1), !dbg !44
  br label %L.LB2_395

L.LB2_395:                                        ; preds = %L.entry
  store i32 8, i32* %.dY0001_358, align 4, !dbg !45
  %3 = bitcast %struct_drb159_0_* @_drb159_0_ to i8*, !dbg !45
  %4 = getelementptr i8, i8* %3, i64 4, !dbg !45
  %5 = bitcast i8* %4 to i32*, !dbg !45
  store i32 1, i32* %5, align 4, !dbg !45
  br label %L.LB2_356

L.LB2_356:                                        ; preds = %L.LB2_356, %L.LB2_395
  %6 = bitcast %struct_drb159_0_* @_drb159_0_ to i8*, !dbg !46
  %7 = getelementptr i8, i8* %6, i64 4, !dbg !46
  %8 = bitcast i8* %7 to i32*, !dbg !46
  %9 = load i32, i32* %8, align 4, !dbg !46
  %10 = sext i32 %9 to i64, !dbg !46
  %11 = bitcast %struct_drb159_0_* @_drb159_0_ to i8*, !dbg !46
  %12 = getelementptr i8, i8* %11, i64 28, !dbg !46
  %13 = bitcast i8* %12 to i32*, !dbg !46
  %14 = getelementptr i32, i32* %13, i64 %10, !dbg !46
  store i32 0, i32* %14, align 4, !dbg !46
  %15 = bitcast %struct_drb159_0_* @_drb159_0_ to i8*, !dbg !47
  %16 = getelementptr i8, i8* %15, i64 4, !dbg !47
  %17 = bitcast i8* %16 to i32*, !dbg !47
  %18 = load i32, i32* %17, align 4, !dbg !47
  %19 = sext i32 %18 to i64, !dbg !47
  %20 = bitcast %struct_drb159_0_* @_drb159_0_ to i8*, !dbg !47
  %21 = getelementptr i8, i8* %20, i64 60, !dbg !47
  %22 = bitcast i8* %21 to i32*, !dbg !47
  %23 = getelementptr i32, i32* %22, i64 %19, !dbg !47
  store i32 2, i32* %23, align 4, !dbg !47
  %24 = bitcast %struct_drb159_0_* @_drb159_0_ to i8*, !dbg !48
  %25 = getelementptr i8, i8* %24, i64 4, !dbg !48
  %26 = bitcast i8* %25 to i32*, !dbg !48
  %27 = load i32, i32* %26, align 4, !dbg !48
  %28 = sext i32 %27 to i64, !dbg !48
  %29 = bitcast %struct_drb159_0_* @_drb159_0_ to i8*, !dbg !48
  %30 = getelementptr i8, i8* %29, i64 92, !dbg !48
  %31 = bitcast i8* %30 to i32*, !dbg !48
  %32 = getelementptr i32, i32* %31, i64 %28, !dbg !48
  store i32 0, i32* %32, align 4, !dbg !48
  %33 = bitcast %struct_drb159_0_* @_drb159_0_ to i8*, !dbg !49
  %34 = getelementptr i8, i8* %33, i64 4, !dbg !49
  %35 = bitcast i8* %34 to i32*, !dbg !49
  %36 = load i32, i32* %35, align 4, !dbg !49
  %37 = add nsw i32 %36, 1, !dbg !49
  %38 = bitcast %struct_drb159_0_* @_drb159_0_ to i8*, !dbg !49
  %39 = getelementptr i8, i8* %38, i64 4, !dbg !49
  %40 = bitcast i8* %39 to i32*, !dbg !49
  store i32 %37, i32* %40, align 4, !dbg !49
  %41 = load i32, i32* %.dY0001_358, align 4, !dbg !49
  %42 = sub nsw i32 %41, 1, !dbg !49
  store i32 %42, i32* %.dY0001_358, align 4, !dbg !49
  %43 = load i32, i32* %.dY0001_358, align 4, !dbg !49
  %44 = icmp sgt i32 %43, 0, !dbg !49
  br i1 %44, label %L.LB2_356, label %L.LB2_430, !dbg !49

L.LB2_430:                                        ; preds = %L.LB2_356
  %45 = bitcast %struct_drb159_0_* @_drb159_0_ to i32*, !dbg !50
  store i32 2, i32* %45, align 4, !dbg !50
  call void @__nv_MAIN__F1L29_1_(i32* %__gtid_MAIN__414, i64* null, i64* null), !dbg !51
  store i32 100, i32* %.dY0005_389, align 4, !dbg !52
  %46 = bitcast %struct_drb159_0_* @_drb159_0_ to i8*, !dbg !52
  %47 = getelementptr i8, i8* %46, i64 4, !dbg !52
  %48 = bitcast i8* %47 to i32*, !dbg !52
  store i32 1, i32* %48, align 4, !dbg !52
  br label %L.LB2_387

L.LB2_387:                                        ; preds = %L.LB2_387, %L.LB2_430
  %49 = bitcast %struct_drb159_0_* @_drb159_0_ to i8*, !dbg !53
  %50 = getelementptr i8, i8* %49, i64 16, !dbg !53
  %51 = bitcast i8* %50 to i32*, !dbg !53
  %52 = load i32, i32* %51, align 4, !dbg !53
  %53 = add nsw i32 %52, 2, !dbg !53
  %54 = bitcast %struct_drb159_0_* @_drb159_0_ to i8*, !dbg !53
  %55 = getelementptr i8, i8* %54, i64 16, !dbg !53
  %56 = bitcast i8* %55 to i32*, !dbg !53
  store i32 %53, i32* %56, align 4, !dbg !53
  %57 = bitcast %struct_drb159_0_* @_drb159_0_ to i8*, !dbg !54
  %58 = getelementptr i8, i8* %57, i64 16, !dbg !54
  %59 = bitcast i8* %58 to i32*, !dbg !54
  %60 = load i32, i32* %59, align 4, !dbg !54
  %61 = mul nsw i32 %60, 2, !dbg !54
  %62 = bitcast %struct_drb159_0_* @_drb159_0_ to i8*, !dbg !54
  %63 = getelementptr i8, i8* %62, i64 16, !dbg !54
  %64 = bitcast i8* %63 to i32*, !dbg !54
  store i32 %61, i32* %64, align 4, !dbg !54
  %65 = bitcast %struct_drb159_0_* @_drb159_0_ to i8*, !dbg !55
  %66 = getelementptr i8, i8* %65, i64 4, !dbg !55
  %67 = bitcast i8* %66 to i32*, !dbg !55
  %68 = load i32, i32* %67, align 4, !dbg !55
  %69 = add nsw i32 %68, 1, !dbg !55
  %70 = bitcast %struct_drb159_0_* @_drb159_0_ to i8*, !dbg !55
  %71 = getelementptr i8, i8* %70, i64 4, !dbg !55
  %72 = bitcast i8* %71 to i32*, !dbg !55
  store i32 %69, i32* %72, align 4, !dbg !55
  %73 = load i32, i32* %.dY0005_389, align 4, !dbg !55
  %74 = sub nsw i32 %73, 1, !dbg !55
  store i32 %74, i32* %.dY0005_389, align 4, !dbg !55
  %75 = load i32, i32* %.dY0005_389, align 4, !dbg !55
  %76 = icmp sgt i32 %75, 0, !dbg !55
  br i1 %76, label %L.LB2_387, label %L.LB2_431, !dbg !55

L.LB2_431:                                        ; preds = %L.LB2_387
  store i32 8, i32* %.dY0006_392, align 4, !dbg !56
  %77 = bitcast %struct_drb159_0_* @_drb159_0_ to i8*, !dbg !56
  %78 = getelementptr i8, i8* %77, i64 4, !dbg !56
  %79 = bitcast i8* %78 to i32*, !dbg !56
  store i32 1, i32* %79, align 4, !dbg !56
  br label %L.LB2_390

L.LB2_390:                                        ; preds = %L.LB2_393, %L.LB2_431
  %80 = bitcast %struct_drb159_0_* @_drb159_0_ to i8*, !dbg !57
  %81 = getelementptr i8, i8* %80, i64 16, !dbg !57
  %82 = bitcast i8* %81 to i32*, !dbg !57
  %83 = load i32, i32* %82, align 4, !dbg !57
  %84 = bitcast %struct_drb159_0_* @_drb159_0_ to i8*, !dbg !57
  %85 = getelementptr i8, i8* %84, i64 4, !dbg !57
  %86 = bitcast i8* %85 to i32*, !dbg !57
  %87 = load i32, i32* %86, align 4, !dbg !57
  %88 = sext i32 %87 to i64, !dbg !57
  %89 = bitcast %struct_drb159_0_* @_drb159_0_ to i8*, !dbg !57
  %90 = getelementptr i8, i8* %89, i64 28, !dbg !57
  %91 = bitcast i8* %90 to i32*, !dbg !57
  %92 = getelementptr i32, i32* %91, i64 %88, !dbg !57
  %93 = load i32, i32* %92, align 4, !dbg !57
  %94 = icmp eq i32 %83, %93, !dbg !57
  br i1 %94, label %L.LB2_393, label %L.LB2_432, !dbg !57

L.LB2_432:                                        ; preds = %L.LB2_390
  call void (...) @_mp_bcs_nest(), !dbg !58
  %95 = bitcast i32* @.C345_MAIN_ to i8*, !dbg !58
  %96 = bitcast [57 x i8]* @.C343_MAIN_ to i8*, !dbg !58
  %97 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !58
  call void (i8*, i8*, i64, ...) %97(i8* %95, i8* %96, i64 57), !dbg !58
  %98 = bitcast i32* @.C346_MAIN_ to i8*, !dbg !58
  %99 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !58
  %100 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !58
  %101 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !58
  %102 = call i32 (i8*, i8*, i8*, i8*, ...) %101(i8* %98, i8* null, i8* %99, i8* %100), !dbg !58
  call void @llvm.dbg.declare(metadata i32* %z__io_348, metadata !59, metadata !DIExpression()), !dbg !33
  store i32 %102, i32* %z__io_348, align 4, !dbg !58
  %103 = bitcast %struct_drb159_0_* @_drb159_0_ to i8*, !dbg !58
  %104 = getelementptr i8, i8* %103, i64 4, !dbg !58
  %105 = bitcast i8* %104 to i32*, !dbg !58
  %106 = load i32, i32* %105, align 4, !dbg !58
  %107 = sext i32 %106 to i64, !dbg !58
  %108 = bitcast %struct_drb159_0_* @_drb159_0_ to i8*, !dbg !58
  %109 = getelementptr i8, i8* %108, i64 28, !dbg !58
  %110 = bitcast i8* %109 to i32*, !dbg !58
  %111 = getelementptr i32, i32* %110, i64 %107, !dbg !58
  %112 = load i32, i32* %111, align 4, !dbg !58
  %113 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !58
  %114 = call i32 (i32, i32, ...) %113(i32 %112, i32 25), !dbg !58
  store i32 %114, i32* %z__io_348, align 4, !dbg !58
  %115 = bitcast %struct_drb159_0_* @_drb159_0_ to i8*, !dbg !58
  %116 = getelementptr i8, i8* %115, i64 16, !dbg !58
  %117 = bitcast i8* %116 to i32*, !dbg !58
  %118 = load i32, i32* %117, align 4, !dbg !58
  %119 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !58
  %120 = call i32 (i32, i32, ...) %119(i32 %118, i32 25), !dbg !58
  store i32 %120, i32* %z__io_348, align 4, !dbg !58
  %121 = call i32 (...) @f90io_ldw_end(), !dbg !58
  store i32 %121, i32* %z__io_348, align 4, !dbg !58
  call void (...) @_mp_ecs_nest(), !dbg !58
  br label %L.LB2_393

L.LB2_393:                                        ; preds = %L.LB2_432, %L.LB2_390
  %122 = bitcast %struct_drb159_0_* @_drb159_0_ to i8*, !dbg !60
  %123 = getelementptr i8, i8* %122, i64 4, !dbg !60
  %124 = bitcast i8* %123 to i32*, !dbg !60
  %125 = load i32, i32* %124, align 4, !dbg !60
  %126 = add nsw i32 %125, 1, !dbg !60
  %127 = bitcast %struct_drb159_0_* @_drb159_0_ to i8*, !dbg !60
  %128 = getelementptr i8, i8* %127, i64 4, !dbg !60
  %129 = bitcast i8* %128 to i32*, !dbg !60
  store i32 %126, i32* %129, align 4, !dbg !60
  %130 = load i32, i32* %.dY0006_392, align 4, !dbg !60
  %131 = sub nsw i32 %130, 1, !dbg !60
  store i32 %131, i32* %.dY0006_392, align 4, !dbg !60
  %132 = load i32, i32* %.dY0006_392, align 4, !dbg !60
  %133 = icmp sgt i32 %132, 0, !dbg !60
  br i1 %133, label %L.LB2_390, label %L.LB2_433, !dbg !60

L.LB2_433:                                        ; preds = %L.LB2_393
  ret void, !dbg !43
}

define internal void @__nv_MAIN__F1L29_1_(i32* %__nv_MAIN__F1L29_1Arg0, i64* %__nv_MAIN__F1L29_1Arg1, i64* %__nv_MAIN__F1L29_1Arg2) #1 !dbg !61 {
L.entry:
  %__gtid___nv_MAIN__F1L29_1__445 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L29_1Arg0, metadata !65, metadata !DIExpression()), !dbg !66
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L29_1Arg1, metadata !67, metadata !DIExpression()), !dbg !66
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L29_1Arg2, metadata !68, metadata !DIExpression()), !dbg !66
  call void @llvm.dbg.value(metadata i32 8, metadata !69, metadata !DIExpression()), !dbg !66
  call void @llvm.dbg.value(metadata i32 1, metadata !70, metadata !DIExpression()), !dbg !66
  call void @llvm.dbg.value(metadata i32 2, metadata !71, metadata !DIExpression()), !dbg !66
  call void @llvm.dbg.value(metadata i32 0, metadata !72, metadata !DIExpression()), !dbg !66
  call void @llvm.dbg.value(metadata i32 1, metadata !73, metadata !DIExpression()), !dbg !66
  call void @llvm.dbg.value(metadata i32 2, metadata !74, metadata !DIExpression()), !dbg !66
  call void @llvm.dbg.value(metadata i32 0, metadata !75, metadata !DIExpression()), !dbg !66
  call void @llvm.dbg.value(metadata i32 1, metadata !76, metadata !DIExpression()), !dbg !66
  call void @llvm.dbg.value(metadata i32 2, metadata !77, metadata !DIExpression()), !dbg !66
  call void @llvm.dbg.value(metadata i32 8, metadata !78, metadata !DIExpression()), !dbg !66
  %0 = load i32, i32* %__nv_MAIN__F1L29_1Arg0, align 4, !dbg !79
  store i32 %0, i32* %__gtid___nv_MAIN__F1L29_1__445, align 4, !dbg !79
  br label %L.LB3_437

L.LB3_437:                                        ; preds = %L.entry
  br label %L.LB3_326

L.LB3_326:                                        ; preds = %L.LB3_437
  br label %L.LB3_443, !dbg !80

L.LB3_443:                                        ; preds = %L.LB3_326
  %1 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN_F1L30_2_ to i64*, !dbg !80
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %1, i64* %__nv_MAIN__F1L29_1Arg2), !dbg !80
  br label %L.LB3_341

L.LB3_341:                                        ; preds = %L.LB3_443
  ret void, !dbg !79
}

define internal void @__nv_MAIN_F1L30_2_(i32* %__nv_MAIN_F1L30_2Arg0, i64* %__nv_MAIN_F1L30_2Arg1, i64* %__nv_MAIN_F1L30_2Arg2) #1 !dbg !81 {
L.entry:
  %__gtid___nv_MAIN_F1L30_2__483 = alloca i32, align 4
  %.dY0002p_361 = alloca i32, align 4
  %i_331 = alloca i32, align 4
  %.i0000p_334 = alloca i32, align 4
  %j_333 = alloca i32, align 4
  %.du0003p_365 = alloca i32, align 4
  %.de0003p_366 = alloca i32, align 4
  %.di0003p_367 = alloca i32, align 4
  %.ds0003p_368 = alloca i32, align 4
  %.dl0003p_370 = alloca i32, align 4
  %.dl0003p.copy_477 = alloca i32, align 4
  %.de0003p.copy_478 = alloca i32, align 4
  %.ds0003p.copy_479 = alloca i32, align 4
  %.dX0003p_369 = alloca i32, align 4
  %.dY0003p_364 = alloca i32, align 4
  %.i0001p_338 = alloca i32, align 4
  %j_337 = alloca i32, align 4
  %.du0004p_377 = alloca i32, align 4
  %.de0004p_378 = alloca i32, align 4
  %.di0004p_379 = alloca i32, align 4
  %.ds0004p_380 = alloca i32, align 4
  %.dl0004p_382 = alloca i32, align 4
  %.dl0004p.copy_512 = alloca i32, align 4
  %.de0004p.copy_513 = alloca i32, align 4
  %.ds0004p.copy_514 = alloca i32, align 4
  %.dX0004p_381 = alloca i32, align 4
  %.dY0004p_376 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L30_2Arg0, metadata !82, metadata !DIExpression()), !dbg !83
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L30_2Arg1, metadata !84, metadata !DIExpression()), !dbg !83
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L30_2Arg2, metadata !85, metadata !DIExpression()), !dbg !83
  call void @llvm.dbg.value(metadata i32 8, metadata !86, metadata !DIExpression()), !dbg !83
  call void @llvm.dbg.value(metadata i32 1, metadata !87, metadata !DIExpression()), !dbg !83
  call void @llvm.dbg.value(metadata i32 2, metadata !88, metadata !DIExpression()), !dbg !83
  call void @llvm.dbg.value(metadata i32 0, metadata !89, metadata !DIExpression()), !dbg !83
  call void @llvm.dbg.value(metadata i32 1, metadata !90, metadata !DIExpression()), !dbg !83
  call void @llvm.dbg.value(metadata i32 2, metadata !91, metadata !DIExpression()), !dbg !83
  call void @llvm.dbg.value(metadata i32 0, metadata !92, metadata !DIExpression()), !dbg !83
  call void @llvm.dbg.value(metadata i32 1, metadata !93, metadata !DIExpression()), !dbg !83
  call void @llvm.dbg.value(metadata i32 2, metadata !94, metadata !DIExpression()), !dbg !83
  call void @llvm.dbg.value(metadata i32 8, metadata !95, metadata !DIExpression()), !dbg !83
  %0 = load i32, i32* %__nv_MAIN_F1L30_2Arg0, align 4, !dbg !96
  store i32 %0, i32* %__gtid___nv_MAIN_F1L30_2__483, align 4, !dbg !96
  br label %L.LB5_467

L.LB5_467:                                        ; preds = %L.entry
  br label %L.LB5_329

L.LB5_329:                                        ; preds = %L.LB5_467
  store i32 100, i32* %.dY0002p_361, align 4, !dbg !97
  call void @llvm.dbg.declare(metadata i32* %i_331, metadata !98, metadata !DIExpression()), !dbg !96
  store i32 1, i32* %i_331, align 4, !dbg !97
  br label %L.LB5_359

L.LB5_359:                                        ; preds = %L.LB5_339, %L.LB5_329
  br label %L.LB5_332

L.LB5_332:                                        ; preds = %L.LB5_359
  store i32 0, i32* %.i0000p_334, align 4, !dbg !99
  call void @llvm.dbg.declare(metadata i32* %j_333, metadata !100, metadata !DIExpression()), !dbg !96
  store i32 1, i32* %j_333, align 4, !dbg !99
  store i32 8, i32* %.du0003p_365, align 4, !dbg !99
  store i32 8, i32* %.de0003p_366, align 4, !dbg !99
  store i32 1, i32* %.di0003p_367, align 4, !dbg !99
  %1 = load i32, i32* %.di0003p_367, align 4, !dbg !99
  store i32 %1, i32* %.ds0003p_368, align 4, !dbg !99
  store i32 1, i32* %.dl0003p_370, align 4, !dbg !99
  %2 = load i32, i32* %.dl0003p_370, align 4, !dbg !99
  store i32 %2, i32* %.dl0003p.copy_477, align 4, !dbg !99
  %3 = load i32, i32* %.de0003p_366, align 4, !dbg !99
  store i32 %3, i32* %.de0003p.copy_478, align 4, !dbg !99
  %4 = load i32, i32* %.ds0003p_368, align 4, !dbg !99
  store i32 %4, i32* %.ds0003p.copy_479, align 4, !dbg !99
  %5 = load i32, i32* %__gtid___nv_MAIN_F1L30_2__483, align 4, !dbg !99
  %6 = bitcast i32* %.i0000p_334 to i64*, !dbg !99
  %7 = bitcast i32* %.dl0003p.copy_477 to i64*, !dbg !99
  %8 = bitcast i32* %.de0003p.copy_478 to i64*, !dbg !99
  %9 = bitcast i32* %.ds0003p.copy_479 to i64*, !dbg !99
  %10 = load i32, i32* %.ds0003p.copy_479, align 4, !dbg !99
  call void @__kmpc_for_static_init_4(i64* null, i32 %5, i32 34, i64* %6, i64* %7, i64* %8, i64* %9, i32 %10, i32 1), !dbg !99
  %11 = load i32, i32* %.dl0003p.copy_477, align 4, !dbg !99
  store i32 %11, i32* %.dl0003p_370, align 4, !dbg !99
  %12 = load i32, i32* %.de0003p.copy_478, align 4, !dbg !99
  store i32 %12, i32* %.de0003p_366, align 4, !dbg !99
  %13 = load i32, i32* %.ds0003p.copy_479, align 4, !dbg !99
  store i32 %13, i32* %.ds0003p_368, align 4, !dbg !99
  %14 = load i32, i32* %.dl0003p_370, align 4, !dbg !99
  store i32 %14, i32* %j_333, align 4, !dbg !99
  %15 = load i32, i32* %j_333, align 4, !dbg !99
  call void @llvm.dbg.value(metadata i32 %15, metadata !100, metadata !DIExpression()), !dbg !96
  store i32 %15, i32* %.dX0003p_369, align 4, !dbg !99
  %16 = load i32, i32* %.dX0003p_369, align 4, !dbg !99
  %17 = load i32, i32* %.du0003p_365, align 4, !dbg !99
  %18 = icmp sgt i32 %16, %17, !dbg !99
  br i1 %18, label %L.LB5_363, label %L.LB5_524, !dbg !99

L.LB5_524:                                        ; preds = %L.LB5_332
  %19 = load i32, i32* %.dX0003p_369, align 4, !dbg !99
  store i32 %19, i32* %j_333, align 4, !dbg !99
  %20 = load i32, i32* %.di0003p_367, align 4, !dbg !99
  %21 = load i32, i32* %.de0003p_366, align 4, !dbg !99
  %22 = load i32, i32* %.dX0003p_369, align 4, !dbg !99
  %23 = sub nsw i32 %21, %22, !dbg !99
  %24 = add nsw i32 %20, %23, !dbg !99
  %25 = load i32, i32* %.di0003p_367, align 4, !dbg !99
  %26 = sdiv i32 %24, %25, !dbg !99
  store i32 %26, i32* %.dY0003p_364, align 4, !dbg !99
  %27 = load i32, i32* %.dY0003p_364, align 4, !dbg !99
  %28 = icmp sle i32 %27, 0, !dbg !99
  br i1 %28, label %L.LB5_373, label %L.LB5_372, !dbg !99

L.LB5_372:                                        ; preds = %L.LB5_372, %L.LB5_524
  %29 = load i32, i32* %j_333, align 4, !dbg !101
  call void @llvm.dbg.value(metadata i32 %29, metadata !100, metadata !DIExpression()), !dbg !96
  %30 = sext i32 %29 to i64, !dbg !101
  %31 = bitcast %struct_drb159_0_* @_drb159_0_ to i8*, !dbg !101
  %32 = getelementptr i8, i8* %31, i64 60, !dbg !101
  %33 = bitcast i8* %32 to i32*, !dbg !101
  %34 = getelementptr i32, i32* %33, i64 %30, !dbg !101
  %35 = load i32, i32* %34, align 4, !dbg !101
  %36 = load i32, i32* %j_333, align 4, !dbg !101
  call void @llvm.dbg.value(metadata i32 %36, metadata !100, metadata !DIExpression()), !dbg !96
  %37 = sext i32 %36 to i64, !dbg !101
  %38 = bitcast %struct_drb159_0_* @_drb159_0_ to i8*, !dbg !101
  %39 = getelementptr i8, i8* %38, i64 28, !dbg !101
  %40 = bitcast i8* %39 to i32*, !dbg !101
  %41 = getelementptr i32, i32* %40, i64 %37, !dbg !101
  %42 = load i32, i32* %41, align 4, !dbg !101
  %43 = add nsw i32 %35, %42, !dbg !101
  %44 = load i32, i32* %j_333, align 4, !dbg !101
  call void @llvm.dbg.value(metadata i32 %44, metadata !100, metadata !DIExpression()), !dbg !96
  %45 = sext i32 %44 to i64, !dbg !101
  %46 = bitcast %struct_drb159_0_* @_drb159_0_ to i8*, !dbg !101
  %47 = getelementptr i8, i8* %46, i64 92, !dbg !101
  %48 = bitcast i8* %47 to i32*, !dbg !101
  %49 = getelementptr i32, i32* %48, i64 %45, !dbg !101
  store i32 %43, i32* %49, align 4, !dbg !101
  %50 = load i32, i32* %.di0003p_367, align 4, !dbg !102
  %51 = load i32, i32* %j_333, align 4, !dbg !102
  call void @llvm.dbg.value(metadata i32 %51, metadata !100, metadata !DIExpression()), !dbg !96
  %52 = add nsw i32 %50, %51, !dbg !102
  store i32 %52, i32* %j_333, align 4, !dbg !102
  %53 = load i32, i32* %.dY0003p_364, align 4, !dbg !102
  %54 = sub nsw i32 %53, 1, !dbg !102
  store i32 %54, i32* %.dY0003p_364, align 4, !dbg !102
  %55 = load i32, i32* %.dY0003p_364, align 4, !dbg !102
  %56 = icmp sgt i32 %55, 0, !dbg !102
  br i1 %56, label %L.LB5_372, label %L.LB5_373, !dbg !102

L.LB5_373:                                        ; preds = %L.LB5_372, %L.LB5_524
  br label %L.LB5_363

L.LB5_363:                                        ; preds = %L.LB5_373, %L.LB5_332
  %57 = load i32, i32* %__gtid___nv_MAIN_F1L30_2__483, align 4, !dbg !102
  call void @__kmpc_for_static_fini(i64* null, i32 %57), !dbg !102
  br label %L.LB5_335

L.LB5_335:                                        ; preds = %L.LB5_363
  %58 = load i32, i32* %__gtid___nv_MAIN_F1L30_2__483, align 4, !dbg !103
  call void @__kmpc_barrier(i64* null, i32 %58), !dbg !103
  br label %L.LB5_336

L.LB5_336:                                        ; preds = %L.LB5_335
  store i32 0, i32* %.i0001p_338, align 4, !dbg !104
  call void @llvm.dbg.declare(metadata i32* %j_337, metadata !100, metadata !DIExpression()), !dbg !96
  store i32 8, i32* %j_337, align 4, !dbg !104
  store i32 1, i32* %.du0004p_377, align 4, !dbg !104
  store i32 1, i32* %.de0004p_378, align 4, !dbg !104
  %59 = bitcast %struct_drb159_0_* @_drb159_0_ to i8*, !dbg !104
  %60 = getelementptr i8, i8* %59, i64 12, !dbg !104
  %61 = bitcast i8* %60 to i32*, !dbg !104
  %62 = load i32, i32* %61, align 4, !dbg !104
  %63 = sub nsw i32 %62, 1, !dbg !104
  store i32 %63, i32* %.di0004p_379, align 4, !dbg !104
  %64 = load i32, i32* %.di0004p_379, align 4, !dbg !104
  store i32 %64, i32* %.ds0004p_380, align 4, !dbg !104
  store i32 8, i32* %.dl0004p_382, align 4, !dbg !104
  %65 = load i32, i32* %.dl0004p_382, align 4, !dbg !104
  store i32 %65, i32* %.dl0004p.copy_512, align 4, !dbg !104
  %66 = load i32, i32* %.de0004p_378, align 4, !dbg !104
  store i32 %66, i32* %.de0004p.copy_513, align 4, !dbg !104
  %67 = load i32, i32* %.ds0004p_380, align 4, !dbg !104
  store i32 %67, i32* %.ds0004p.copy_514, align 4, !dbg !104
  %68 = load i32, i32* %__gtid___nv_MAIN_F1L30_2__483, align 4, !dbg !104
  %69 = bitcast i32* %.i0001p_338 to i64*, !dbg !104
  %70 = bitcast i32* %.dl0004p.copy_512 to i64*, !dbg !104
  %71 = bitcast i32* %.de0004p.copy_513 to i64*, !dbg !104
  %72 = bitcast i32* %.ds0004p.copy_514 to i64*, !dbg !104
  %73 = load i32, i32* %.ds0004p.copy_514, align 4, !dbg !104
  call void @__kmpc_for_static_init_4(i64* null, i32 %68, i32 34, i64* %69, i64* %70, i64* %71, i64* %72, i32 %73, i32 1), !dbg !104
  %74 = load i32, i32* %.dl0004p.copy_512, align 4, !dbg !104
  store i32 %74, i32* %.dl0004p_382, align 4, !dbg !104
  %75 = load i32, i32* %.de0004p.copy_513, align 4, !dbg !104
  store i32 %75, i32* %.de0004p_378, align 4, !dbg !104
  %76 = load i32, i32* %.ds0004p.copy_514, align 4, !dbg !104
  store i32 %76, i32* %.ds0004p_380, align 4, !dbg !104
  %77 = load i32, i32* %.dl0004p_382, align 4, !dbg !104
  store i32 %77, i32* %j_337, align 4, !dbg !104
  %78 = load i32, i32* %j_337, align 4, !dbg !104
  call void @llvm.dbg.value(metadata i32 %78, metadata !100, metadata !DIExpression()), !dbg !96
  store i32 %78, i32* %.dX0004p_381, align 4, !dbg !104
  %79 = load i32, i32* %.ds0004p_380, align 4, !dbg !104
  %80 = icmp slt i32 %79, 0, !dbg !104
  br i1 %80, label %L.LB5_383, label %L.LB5_525, !dbg !104

L.LB5_525:                                        ; preds = %L.LB5_336
  %81 = load i32, i32* %.dX0004p_381, align 4, !dbg !104
  %82 = load i32, i32* %.du0004p_377, align 4, !dbg !104
  %83 = icmp sgt i32 %81, %82, !dbg !104
  br i1 %83, label %L.LB5_375, label %L.LB5_526, !dbg !104

L.LB5_526:                                        ; preds = %L.LB5_525
  br label %L.LB5_384, !dbg !104

L.LB5_383:                                        ; preds = %L.LB5_336
  %84 = load i32, i32* %.dX0004p_381, align 4, !dbg !104
  %85 = load i32, i32* %.du0004p_377, align 4, !dbg !104
  %86 = icmp slt i32 %84, %85, !dbg !104
  br i1 %86, label %L.LB5_375, label %L.LB5_384, !dbg !104

L.LB5_384:                                        ; preds = %L.LB5_383, %L.LB5_526
  %87 = load i32, i32* %.dX0004p_381, align 4, !dbg !104
  store i32 %87, i32* %j_337, align 4, !dbg !104
  %88 = load i32, i32* %.di0004p_379, align 4, !dbg !104
  %89 = load i32, i32* %.de0004p_378, align 4, !dbg !104
  %90 = load i32, i32* %.dX0004p_381, align 4, !dbg !104
  %91 = sub nsw i32 %89, %90, !dbg !104
  %92 = add nsw i32 %88, %91, !dbg !104
  %93 = load i32, i32* %.di0004p_379, align 4, !dbg !104
  %94 = sdiv i32 %92, %93, !dbg !104
  store i32 %94, i32* %.dY0004p_376, align 4, !dbg !104
  %95 = load i32, i32* %.dY0004p_376, align 4, !dbg !104
  %96 = icmp sle i32 %95, 0, !dbg !104
  br i1 %96, label %L.LB5_386, label %L.LB5_385, !dbg !104

L.LB5_385:                                        ; preds = %L.LB5_385, %L.LB5_384
  %97 = load i32, i32* %j_337, align 4, !dbg !105
  call void @llvm.dbg.value(metadata i32 %97, metadata !100, metadata !DIExpression()), !dbg !96
  %98 = sext i32 %97 to i64, !dbg !105
  %99 = bitcast %struct_drb159_0_* @_drb159_0_ to i8*, !dbg !105
  %100 = getelementptr i8, i8* %99, i64 92, !dbg !105
  %101 = bitcast i8* %100 to i32*, !dbg !105
  %102 = getelementptr i32, i32* %101, i64 %98, !dbg !105
  %103 = load i32, i32* %102, align 4, !dbg !105
  %104 = bitcast %struct_drb159_0_* @_drb159_0_ to i32*, !dbg !105
  %105 = load i32, i32* %104, align 4, !dbg !105
  %106 = mul nsw i32 %103, %105, !dbg !105
  %107 = load i32, i32* %j_337, align 4, !dbg !105
  call void @llvm.dbg.value(metadata i32 %107, metadata !100, metadata !DIExpression()), !dbg !96
  %108 = sext i32 %107 to i64, !dbg !105
  %109 = bitcast %struct_drb159_0_* @_drb159_0_ to i8*, !dbg !105
  %110 = getelementptr i8, i8* %109, i64 28, !dbg !105
  %111 = bitcast i8* %110 to i32*, !dbg !105
  %112 = getelementptr i32, i32* %111, i64 %108, !dbg !105
  store i32 %106, i32* %112, align 4, !dbg !105
  %113 = load i32, i32* %.di0004p_379, align 4, !dbg !106
  %114 = load i32, i32* %j_337, align 4, !dbg !106
  call void @llvm.dbg.value(metadata i32 %114, metadata !100, metadata !DIExpression()), !dbg !96
  %115 = add nsw i32 %113, %114, !dbg !106
  store i32 %115, i32* %j_337, align 4, !dbg !106
  %116 = load i32, i32* %.dY0004p_376, align 4, !dbg !106
  %117 = sub nsw i32 %116, 1, !dbg !106
  store i32 %117, i32* %.dY0004p_376, align 4, !dbg !106
  %118 = load i32, i32* %.dY0004p_376, align 4, !dbg !106
  %119 = icmp sgt i32 %118, 0, !dbg !106
  br i1 %119, label %L.LB5_385, label %L.LB5_386, !dbg !106

L.LB5_386:                                        ; preds = %L.LB5_385, %L.LB5_384
  br label %L.LB5_375

L.LB5_375:                                        ; preds = %L.LB5_386, %L.LB5_383, %L.LB5_525
  %120 = load i32, i32* %__gtid___nv_MAIN_F1L30_2__483, align 4, !dbg !106
  call void @__kmpc_for_static_fini(i64* null, i32 %120), !dbg !106
  br label %L.LB5_339

L.LB5_339:                                        ; preds = %L.LB5_375
  %121 = load i32, i32* %__gtid___nv_MAIN_F1L30_2__483, align 4, !dbg !107
  call void @__kmpc_barrier(i64* null, i32 %121), !dbg !107
  %122 = load i32, i32* %i_331, align 4, !dbg !108
  call void @llvm.dbg.value(metadata i32 %122, metadata !98, metadata !DIExpression()), !dbg !96
  %123 = add nsw i32 %122, 1, !dbg !108
  store i32 %123, i32* %i_331, align 4, !dbg !108
  %124 = load i32, i32* %.dY0002p_361, align 4, !dbg !108
  %125 = sub nsw i32 %124, 1, !dbg !108
  store i32 %125, i32* %.dY0002p_361, align 4, !dbg !108
  %126 = load i32, i32* %.dY0002p_361, align 4, !dbg !108
  %127 = icmp sgt i32 %126, 0, !dbg !108
  br i1 %127, label %L.LB5_359, label %L.LB5_340, !dbg !108

L.LB5_340:                                        ; preds = %L.LB5_339
  ret void, !dbg !96
}

declare void @__kmpc_barrier(i64*, i32) #1

declare void @__kmpc_for_static_fini(i64*, i32) #1

declare void @__kmpc_for_static_init_4(i64*, i32, i32, i64*, i64*, i64*, i64*, i32, i32) #1

declare void @__kmpc_fork_call(i64*, i32, i64*, i64*, ...) #1

declare void @__kmpc_end_serialized_parallel(i64*, i32) #1

declare void @__kmpc_serialized_parallel(i64*, i32) #1

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

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { noinline }
attributes #1 = { "no-frame-pointer-elim-non-leaf" }
attributes #2 = { nounwind readnone speculatable }

!llvm.module.flags = !{!30, !31}
!llvm.dbg.cu = !{!3}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !4, type: !9, isLocal: false, isDefinition: true)
!2 = !DIModule(scope: !3, name: "drb159")
!3 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !4, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !6, imports: !25)
!4 = !DIFile(filename: "micro-benchmarks-fortran/DRB159-nobarrier-orig-gpu-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!5 = !{}
!6 = !{!0, !7, !10, !12, !14, !16, !21, !23}
!7 = !DIGlobalVariableExpression(var: !8, expr: !DIExpression(DW_OP_plus_uconst, 4))
!8 = distinct !DIGlobalVariable(name: "i", scope: !2, file: !4, type: !9, isLocal: false, isDefinition: true)
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DIGlobalVariableExpression(var: !11, expr: !DIExpression(DW_OP_plus_uconst, 8))
!11 = distinct !DIGlobalVariable(name: "j", scope: !2, file: !4, type: !9, isLocal: false, isDefinition: true)
!12 = !DIGlobalVariableExpression(var: !13, expr: !DIExpression(DW_OP_plus_uconst, 12))
!13 = distinct !DIGlobalVariable(name: "k", scope: !2, file: !4, type: !9, isLocal: false, isDefinition: true)
!14 = !DIGlobalVariableExpression(var: !15, expr: !DIExpression(DW_OP_plus_uconst, 16))
!15 = distinct !DIGlobalVariable(name: "val", scope: !2, file: !4, type: !9, isLocal: false, isDefinition: true)
!16 = !DIGlobalVariableExpression(var: !17, expr: !DIExpression(DW_OP_plus_uconst, 32))
!17 = distinct !DIGlobalVariable(name: "b", scope: !2, file: !4, type: !18, isLocal: false, isDefinition: true)
!18 = !DICompositeType(tag: DW_TAG_array_type, baseType: !9, size: 256, align: 32, elements: !19)
!19 = !{!20}
!20 = !DISubrange(count: 8, lowerBound: 1)
!21 = !DIGlobalVariableExpression(var: !22, expr: !DIExpression(DW_OP_plus_uconst, 64))
!22 = distinct !DIGlobalVariable(name: "c", scope: !2, file: !4, type: !18, isLocal: false, isDefinition: true)
!23 = !DIGlobalVariableExpression(var: !24, expr: !DIExpression(DW_OP_plus_uconst, 96))
!24 = distinct !DIGlobalVariable(name: "temp", scope: !2, file: !4, type: !18, isLocal: false, isDefinition: true)
!25 = !{!26}
!26 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !27, entity: !2, file: !4, line: 16)
!27 = distinct !DISubprogram(name: "drb159_nobarrier_orig_gpu_no", scope: !3, file: !4, line: 16, type: !28, scopeLine: 16, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !3)
!28 = !DISubroutineType(cc: DW_CC_program, types: !29)
!29 = !{null}
!30 = !{i32 2, !"Dwarf Version", i32 4}
!31 = !{i32 2, !"Debug Info Version", i32 3}
!32 = !DILocalVariable(name: "omp_nest_lock_kind", scope: !27, file: !4, type: !9)
!33 = !DILocation(line: 0, scope: !27)
!34 = !DILocalVariable(name: "omp_sched_static", scope: !27, file: !4, type: !9)
!35 = !DILocalVariable(name: "omp_sched_dynamic", scope: !27, file: !4, type: !9)
!36 = !DILocalVariable(name: "omp_proc_bind_false", scope: !27, file: !4, type: !9)
!37 = !DILocalVariable(name: "omp_proc_bind_true", scope: !27, file: !4, type: !9)
!38 = !DILocalVariable(name: "omp_proc_bind_master", scope: !27, file: !4, type: !9)
!39 = !DILocalVariable(name: "omp_lock_hint_none", scope: !27, file: !4, type: !9)
!40 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !27, file: !4, type: !9)
!41 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !27, file: !4, type: !9)
!42 = !DILocalVariable(name: "omp_lock_hint_speculative", scope: !27, file: !4, type: !9)
!43 = !DILocation(line: 58, column: 1, scope: !27)
!44 = !DILocation(line: 16, column: 1, scope: !27)
!45 = !DILocation(line: 21, column: 1, scope: !27)
!46 = !DILocation(line: 22, column: 1, scope: !27)
!47 = !DILocation(line: 23, column: 1, scope: !27)
!48 = !DILocation(line: 24, column: 1, scope: !27)
!49 = !DILocation(line: 25, column: 1, scope: !27)
!50 = !DILocation(line: 27, column: 1, scope: !27)
!51 = !DILocation(line: 45, column: 1, scope: !27)
!52 = !DILocation(line: 47, column: 1, scope: !27)
!53 = !DILocation(line: 48, column: 1, scope: !27)
!54 = !DILocation(line: 49, column: 1, scope: !27)
!55 = !DILocation(line: 50, column: 1, scope: !27)
!56 = !DILocation(line: 52, column: 1, scope: !27)
!57 = !DILocation(line: 53, column: 1, scope: !27)
!58 = !DILocation(line: 54, column: 1, scope: !27)
!59 = !DILocalVariable(scope: !27, file: !4, type: !9, flags: DIFlagArtificial)
!60 = !DILocation(line: 56, column: 1, scope: !27)
!61 = distinct !DISubprogram(name: "__nv_MAIN__F1L29_1", scope: !3, file: !4, line: 29, type: !62, scopeLine: 29, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !3)
!62 = !DISubroutineType(types: !63)
!63 = !{null, !9, !64, !64}
!64 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!65 = !DILocalVariable(name: "__nv_MAIN__F1L29_1Arg0", arg: 1, scope: !61, file: !4, type: !9)
!66 = !DILocation(line: 0, scope: !61)
!67 = !DILocalVariable(name: "__nv_MAIN__F1L29_1Arg1", arg: 2, scope: !61, file: !4, type: !64)
!68 = !DILocalVariable(name: "__nv_MAIN__F1L29_1Arg2", arg: 3, scope: !61, file: !4, type: !64)
!69 = !DILocalVariable(name: "omp_nest_lock_kind", scope: !61, file: !4, type: !9)
!70 = !DILocalVariable(name: "omp_sched_static", scope: !61, file: !4, type: !9)
!71 = !DILocalVariable(name: "omp_sched_dynamic", scope: !61, file: !4, type: !9)
!72 = !DILocalVariable(name: "omp_proc_bind_false", scope: !61, file: !4, type: !9)
!73 = !DILocalVariable(name: "omp_proc_bind_true", scope: !61, file: !4, type: !9)
!74 = !DILocalVariable(name: "omp_proc_bind_master", scope: !61, file: !4, type: !9)
!75 = !DILocalVariable(name: "omp_lock_hint_none", scope: !61, file: !4, type: !9)
!76 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !61, file: !4, type: !9)
!77 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !61, file: !4, type: !9)
!78 = !DILocalVariable(name: "omp_lock_hint_speculative", scope: !61, file: !4, type: !9)
!79 = !DILocation(line: 45, column: 1, scope: !61)
!80 = !DILocation(line: 30, column: 1, scope: !61)
!81 = distinct !DISubprogram(name: "__nv_MAIN_F1L30_2", scope: !3, file: !4, line: 30, type: !62, scopeLine: 30, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !3)
!82 = !DILocalVariable(name: "__nv_MAIN_F1L30_2Arg0", arg: 1, scope: !81, file: !4, type: !9)
!83 = !DILocation(line: 0, scope: !81)
!84 = !DILocalVariable(name: "__nv_MAIN_F1L30_2Arg1", arg: 2, scope: !81, file: !4, type: !64)
!85 = !DILocalVariable(name: "__nv_MAIN_F1L30_2Arg2", arg: 3, scope: !81, file: !4, type: !64)
!86 = !DILocalVariable(name: "omp_nest_lock_kind", scope: !81, file: !4, type: !9)
!87 = !DILocalVariable(name: "omp_sched_static", scope: !81, file: !4, type: !9)
!88 = !DILocalVariable(name: "omp_sched_dynamic", scope: !81, file: !4, type: !9)
!89 = !DILocalVariable(name: "omp_proc_bind_false", scope: !81, file: !4, type: !9)
!90 = !DILocalVariable(name: "omp_proc_bind_true", scope: !81, file: !4, type: !9)
!91 = !DILocalVariable(name: "omp_proc_bind_master", scope: !81, file: !4, type: !9)
!92 = !DILocalVariable(name: "omp_lock_hint_none", scope: !81, file: !4, type: !9)
!93 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !81, file: !4, type: !9)
!94 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !81, file: !4, type: !9)
!95 = !DILocalVariable(name: "omp_lock_hint_speculative", scope: !81, file: !4, type: !9)
!96 = !DILocation(line: 44, column: 1, scope: !81)
!97 = !DILocation(line: 31, column: 1, scope: !81)
!98 = !DILocalVariable(name: "i", scope: !81, file: !4, type: !9)
!99 = !DILocation(line: 33, column: 1, scope: !81)
!100 = !DILocalVariable(name: "j", scope: !81, file: !4, type: !9)
!101 = !DILocation(line: 34, column: 1, scope: !81)
!102 = !DILocation(line: 35, column: 1, scope: !81)
!103 = !DILocation(line: 36, column: 1, scope: !81)
!104 = !DILocation(line: 39, column: 1, scope: !81)
!105 = !DILocation(line: 40, column: 1, scope: !81)
!106 = !DILocation(line: 41, column: 1, scope: !81)
!107 = !DILocation(line: 42, column: 1, scope: !81)
!108 = !DILocation(line: 43, column: 1, scope: !81)

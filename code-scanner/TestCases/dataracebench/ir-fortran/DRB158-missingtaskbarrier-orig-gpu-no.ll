; ModuleID = '/tmp/DRB158-missingtaskbarrier-orig-gpu-no-ad8787.ll'
source_filename = "/tmp/DRB158-missingtaskbarrier-orig-gpu-no-ad8787.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct_drb158_0_ = type <{ [528 x i8] }>

@.C309_MAIN_ = internal constant i32 25
@.C284_MAIN_ = internal constant i64 0
@.C334_MAIN_ = internal constant i32 6
@.C331_MAIN_ = internal constant [66 x i8] c"micro-benchmarks-fortran/DRB158-missingtaskbarrier-orig-gpu-no.f95"
@.C333_MAIN_ = internal constant i32 41
@.C317_MAIN_ = internal constant i32 5
@.C301_MAIN_ = internal constant i32 3
@.C316_MAIN_ = internal constant i32 64
@.C285_MAIN_ = internal constant i32 1
@.C283_MAIN_ = internal constant i32 0
@.C283___nv_MAIN__F1L27_1 = internal constant i32 0
@.C316___nv_MAIN__F1L27_1 = internal constant i32 64
@.C285___nv_MAIN__F1L27_1 = internal constant i32 1
@_drb158_0_ = common global %struct_drb158_0_ zeroinitializer, align 64, !dbg !0, !dbg !7, !dbg !10, !dbg !15

; Function Attrs: noinline
define float @drb158_() #0 {
.L.entry:
  ret float undef
}

define void @MAIN_() #1 !dbg !19 {
L.entry:
  %__gtid_MAIN__374 = alloca i32, align 4
  %.dY0001_346 = alloca i32, align 4
  %.dY0003_354 = alloca i32, align 4
  %z__io_336 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !24, metadata !DIExpression()), !dbg !25
  call void @llvm.dbg.value(metadata i32 3, metadata !26, metadata !DIExpression()), !dbg !25
  call void @llvm.dbg.value(metadata i32 0, metadata !27, metadata !DIExpression()), !dbg !25
  call void @llvm.dbg.value(metadata i32 1, metadata !28, metadata !DIExpression()), !dbg !25
  call void @llvm.dbg.value(metadata i32 3, metadata !29, metadata !DIExpression()), !dbg !25
  call void @llvm.dbg.value(metadata i32 0, metadata !30, metadata !DIExpression()), !dbg !25
  call void @llvm.dbg.value(metadata i32 1, metadata !31, metadata !DIExpression()), !dbg !25
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !32
  store i32 %0, i32* %__gtid_MAIN__374, align 4, !dbg !32
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !33
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !33
  call void (i8*, ...) %2(i8* %1), !dbg !33
  br label %L.LB2_357

L.LB2_357:                                        ; preds = %L.entry
  store i32 64, i32* %.dY0001_346, align 4, !dbg !34
  %3 = bitcast %struct_drb158_0_* @_drb158_0_ to i8*, !dbg !34
  %4 = getelementptr i8, i8* %3, i64 4, !dbg !34
  %5 = bitcast i8* %4 to i32*, !dbg !34
  store i32 1, i32* %5, align 4, !dbg !34
  br label %L.LB2_344

L.LB2_344:                                        ; preds = %L.LB2_344, %L.LB2_357
  %6 = bitcast %struct_drb158_0_* @_drb158_0_ to i8*, !dbg !35
  %7 = getelementptr i8, i8* %6, i64 4, !dbg !35
  %8 = bitcast i8* %7 to i32*, !dbg !35
  %9 = load i32, i32* %8, align 4, !dbg !35
  %10 = sext i32 %9 to i64, !dbg !35
  %11 = bitcast %struct_drb158_0_* @_drb158_0_ to i8*, !dbg !35
  %12 = getelementptr i8, i8* %11, i64 12, !dbg !35
  %13 = bitcast i8* %12 to i32*, !dbg !35
  %14 = getelementptr i32, i32* %13, i64 %10, !dbg !35
  store i32 0, i32* %14, align 4, !dbg !35
  %15 = bitcast %struct_drb158_0_* @_drb158_0_ to i8*, !dbg !36
  %16 = getelementptr i8, i8* %15, i64 4, !dbg !36
  %17 = bitcast i8* %16 to i32*, !dbg !36
  %18 = load i32, i32* %17, align 4, !dbg !36
  %19 = sext i32 %18 to i64, !dbg !36
  %20 = bitcast %struct_drb158_0_* @_drb158_0_ to i8*, !dbg !36
  %21 = getelementptr i8, i8* %20, i64 268, !dbg !36
  %22 = bitcast i8* %21 to i32*, !dbg !36
  %23 = getelementptr i32, i32* %22, i64 %19, !dbg !36
  store i32 3, i32* %23, align 4, !dbg !36
  %24 = bitcast %struct_drb158_0_* @_drb158_0_ to i8*, !dbg !37
  %25 = getelementptr i8, i8* %24, i64 4, !dbg !37
  %26 = bitcast i8* %25 to i32*, !dbg !37
  %27 = load i32, i32* %26, align 4, !dbg !37
  %28 = add nsw i32 %27, 1, !dbg !37
  %29 = bitcast %struct_drb158_0_* @_drb158_0_ to i8*, !dbg !37
  %30 = getelementptr i8, i8* %29, i64 4, !dbg !37
  %31 = bitcast i8* %30 to i32*, !dbg !37
  store i32 %28, i32* %31, align 4, !dbg !37
  %32 = load i32, i32* %.dY0001_346, align 4, !dbg !37
  %33 = sub nsw i32 %32, 1, !dbg !37
  store i32 %33, i32* %.dY0001_346, align 4, !dbg !37
  %34 = load i32, i32* %.dY0001_346, align 4, !dbg !37
  %35 = icmp sgt i32 %34, 0, !dbg !37
  br i1 %35, label %L.LB2_344, label %L.LB2_391, !dbg !37

L.LB2_391:                                        ; preds = %L.LB2_344
  %36 = bitcast %struct_drb158_0_* @_drb158_0_ to i32*, !dbg !38
  store i32 5, i32* %36, align 4, !dbg !38
  call void @__nv_MAIN__F1L27_1_(i32* %__gtid_MAIN__374, i64* null, i64* null), !dbg !39
  store i32 64, i32* %.dY0003_354, align 4, !dbg !40
  %37 = bitcast %struct_drb158_0_* @_drb158_0_ to i8*, !dbg !40
  %38 = getelementptr i8, i8* %37, i64 4, !dbg !40
  %39 = bitcast i8* %38 to i32*, !dbg !40
  store i32 1, i32* %39, align 4, !dbg !40
  br label %L.LB2_352

L.LB2_352:                                        ; preds = %L.LB2_355, %L.LB2_391
  %40 = bitcast %struct_drb158_0_* @_drb158_0_ to i8*, !dbg !41
  %41 = getelementptr i8, i8* %40, i64 4, !dbg !41
  %42 = bitcast i8* %41 to i32*, !dbg !41
  %43 = load i32, i32* %42, align 4, !dbg !41
  %44 = sext i32 %43 to i64, !dbg !41
  %45 = bitcast %struct_drb158_0_* @_drb158_0_ to i8*, !dbg !41
  %46 = getelementptr i8, i8* %45, i64 12, !dbg !41
  %47 = bitcast i8* %46 to i32*, !dbg !41
  %48 = getelementptr i32, i32* %47, i64 %44, !dbg !41
  %49 = load i32, i32* %48, align 4, !dbg !41
  %50 = icmp eq i32 %49, 3, !dbg !41
  br i1 %50, label %L.LB2_355, label %L.LB2_392, !dbg !41

L.LB2_392:                                        ; preds = %L.LB2_352
  call void (...) @_mp_bcs_nest(), !dbg !42
  %51 = bitcast i32* @.C333_MAIN_ to i8*, !dbg !42
  %52 = bitcast [66 x i8]* @.C331_MAIN_ to i8*, !dbg !42
  %53 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !42
  call void (i8*, i8*, i64, ...) %53(i8* %51, i8* %52, i64 66), !dbg !42
  %54 = bitcast i32* @.C334_MAIN_ to i8*, !dbg !42
  %55 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !42
  %56 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !42
  %57 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !42
  %58 = call i32 (i8*, i8*, i8*, i8*, ...) %57(i8* %54, i8* null, i8* %55, i8* %56), !dbg !42
  call void @llvm.dbg.declare(metadata i32* %z__io_336, metadata !43, metadata !DIExpression()), !dbg !25
  store i32 %58, i32* %z__io_336, align 4, !dbg !42
  %59 = bitcast %struct_drb158_0_* @_drb158_0_ to i8*, !dbg !42
  %60 = getelementptr i8, i8* %59, i64 4, !dbg !42
  %61 = bitcast i8* %60 to i32*, !dbg !42
  %62 = load i32, i32* %61, align 4, !dbg !42
  %63 = sext i32 %62 to i64, !dbg !42
  %64 = bitcast %struct_drb158_0_* @_drb158_0_ to i8*, !dbg !42
  %65 = getelementptr i8, i8* %64, i64 12, !dbg !42
  %66 = bitcast i8* %65 to i32*, !dbg !42
  %67 = getelementptr i32, i32* %66, i64 %63, !dbg !42
  %68 = load i32, i32* %67, align 4, !dbg !42
  %69 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !42
  %70 = call i32 (i32, i32, ...) %69(i32 %68, i32 25), !dbg !42
  store i32 %70, i32* %z__io_336, align 4, !dbg !42
  %71 = call i32 (...) @f90io_ldw_end(), !dbg !42
  store i32 %71, i32* %z__io_336, align 4, !dbg !42
  call void (...) @_mp_ecs_nest(), !dbg !42
  br label %L.LB2_355

L.LB2_355:                                        ; preds = %L.LB2_392, %L.LB2_352
  %72 = bitcast %struct_drb158_0_* @_drb158_0_ to i8*, !dbg !44
  %73 = getelementptr i8, i8* %72, i64 4, !dbg !44
  %74 = bitcast i8* %73 to i32*, !dbg !44
  %75 = load i32, i32* %74, align 4, !dbg !44
  %76 = add nsw i32 %75, 1, !dbg !44
  %77 = bitcast %struct_drb158_0_* @_drb158_0_ to i8*, !dbg !44
  %78 = getelementptr i8, i8* %77, i64 4, !dbg !44
  %79 = bitcast i8* %78 to i32*, !dbg !44
  store i32 %76, i32* %79, align 4, !dbg !44
  %80 = load i32, i32* %.dY0003_354, align 4, !dbg !44
  %81 = sub nsw i32 %80, 1, !dbg !44
  store i32 %81, i32* %.dY0003_354, align 4, !dbg !44
  %82 = load i32, i32* %.dY0003_354, align 4, !dbg !44
  %83 = icmp sgt i32 %82, 0, !dbg !44
  br i1 %83, label %L.LB2_352, label %L.LB2_393, !dbg !44

L.LB2_393:                                        ; preds = %L.LB2_355
  %84 = load i32, i32* %__gtid_MAIN__374, align 4, !dbg !45
  %85 = call i32 @__kmpc_omp_taskwait(i64* null, i32 %84), !dbg !45
  ret void, !dbg !32
}

define internal void @__nv_MAIN__F1L27_1_(i32* %__nv_MAIN__F1L27_1Arg0, i64* %__nv_MAIN__F1L27_1Arg1, i64* %__nv_MAIN__F1L27_1Arg2) #1 !dbg !46 {
L.entry:
  %__gtid___nv_MAIN__F1L27_1__405 = alloca i32, align 4
  %.dY0002_349 = alloca i32, align 4
  %.s0000_400 = alloca i32, align 4
  %.z0369_399 = alloca i8*, align 8
  %.s0001_423 = alloca i32, align 4
  %.z0369_422 = alloca i8*, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L27_1Arg0, metadata !50, metadata !DIExpression()), !dbg !51
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L27_1Arg1, metadata !52, metadata !DIExpression()), !dbg !51
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L27_1Arg2, metadata !53, metadata !DIExpression()), !dbg !51
  call void @llvm.dbg.value(metadata i32 1, metadata !54, metadata !DIExpression()), !dbg !51
  call void @llvm.dbg.value(metadata i32 3, metadata !55, metadata !DIExpression()), !dbg !51
  call void @llvm.dbg.value(metadata i32 0, metadata !56, metadata !DIExpression()), !dbg !51
  call void @llvm.dbg.value(metadata i32 1, metadata !57, metadata !DIExpression()), !dbg !51
  call void @llvm.dbg.value(metadata i32 3, metadata !58, metadata !DIExpression()), !dbg !51
  call void @llvm.dbg.value(metadata i32 0, metadata !59, metadata !DIExpression()), !dbg !51
  call void @llvm.dbg.value(metadata i32 1, metadata !60, metadata !DIExpression()), !dbg !51
  %0 = load i32, i32* %__nv_MAIN__F1L27_1Arg0, align 4, !dbg !61
  store i32 %0, i32* %__gtid___nv_MAIN__F1L27_1__405, align 4, !dbg !61
  br label %L.LB3_397

L.LB3_397:                                        ; preds = %L.entry
  br label %L.LB3_320

L.LB3_320:                                        ; preds = %L.LB3_397
  store i32 64, i32* %.dY0002_349, align 4, !dbg !62
  %1 = bitcast %struct_drb158_0_* @_drb158_0_ to i8*, !dbg !62
  %2 = getelementptr i8, i8* %1, i64 4, !dbg !62
  %3 = bitcast i8* %2 to i32*, !dbg !62
  store i32 1, i32* %3, align 4, !dbg !62
  br label %L.LB3_347

L.LB3_347:                                        ; preds = %L.LB3_351, %L.LB3_320
  store i32 1, i32* %.s0000_400, align 4, !dbg !63
  %4 = load i32, i32* %__gtid___nv_MAIN__F1L27_1__405, align 4, !dbg !64
  %5 = load i32, i32* %.s0000_400, align 4, !dbg !64
  %6 = bitcast void (i32, i64*)* @__nv_MAIN_F1L29_2_ to i64*, !dbg !64
  %7 = call i8* @__kmpc_omp_task_alloc(i64* null, i32 %4, i32 %5, i32 40, i32 0, i64* %6), !dbg !64
  store i8* %7, i8** %.z0369_399, align 8, !dbg !64
  %8 = load i32, i32* %__gtid___nv_MAIN__F1L27_1__405, align 4, !dbg !64
  %9 = load i8*, i8** %.z0369_399, align 8, !dbg !64
  %10 = bitcast i8* %9 to i64*, !dbg !64
  call void @__kmpc_omp_task(i64* null, i32 %8, i64* %10), !dbg !64
  br label %L.LB3_350

L.LB3_350:                                        ; preds = %L.LB3_347
  store i32 1, i32* %.s0001_423, align 4, !dbg !65
  %11 = load i32, i32* %__gtid___nv_MAIN__F1L27_1__405, align 4, !dbg !66
  %12 = load i32, i32* %.s0001_423, align 4, !dbg !66
  %13 = bitcast void (i32, i64*)* @__nv_MAIN_F1L33_3_ to i64*, !dbg !66
  %14 = call i8* @__kmpc_omp_task_alloc(i64* null, i32 %11, i32 %12, i32 40, i32 8, i64* %13), !dbg !66
  store i8* %14, i8** %.z0369_422, align 8, !dbg !66
  %15 = load i32, i32* %__gtid___nv_MAIN__F1L27_1__405, align 4, !dbg !66
  %16 = load i8*, i8** %.z0369_422, align 8, !dbg !66
  %17 = bitcast i8* %16 to i64*, !dbg !66
  call void @__kmpc_omp_task(i64* null, i32 %15, i64* %17), !dbg !66
  br label %L.LB3_351

L.LB3_351:                                        ; preds = %L.LB3_350
  %18 = bitcast %struct_drb158_0_* @_drb158_0_ to i8*, !dbg !67
  %19 = getelementptr i8, i8* %18, i64 4, !dbg !67
  %20 = bitcast i8* %19 to i32*, !dbg !67
  %21 = load i32, i32* %20, align 4, !dbg !67
  %22 = add nsw i32 %21, 1, !dbg !67
  %23 = bitcast %struct_drb158_0_* @_drb158_0_ to i8*, !dbg !67
  %24 = getelementptr i8, i8* %23, i64 4, !dbg !67
  %25 = bitcast i8* %24 to i32*, !dbg !67
  store i32 %22, i32* %25, align 4, !dbg !67
  %26 = load i32, i32* %.dY0002_349, align 4, !dbg !67
  %27 = sub nsw i32 %26, 1, !dbg !67
  store i32 %27, i32* %.dY0002_349, align 4, !dbg !67
  %28 = load i32, i32* %.dY0002_349, align 4, !dbg !67
  %29 = icmp sgt i32 %28, 0, !dbg !67
  br i1 %29, label %L.LB3_347, label %L.LB3_329, !dbg !67

L.LB3_329:                                        ; preds = %L.LB3_351
  ret void, !dbg !61
}

define internal void @__nv_MAIN_F1L29_2_(i32 %__nv_MAIN_F1L29_2Arg0.arg, i64* %__nv_MAIN_F1L29_2Arg1) #1 !dbg !68 {
L.entry:
  %__nv_MAIN_F1L29_2Arg0.addr = alloca i32, align 4
  %.S0000_436 = alloca i8*, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L29_2Arg0.addr, metadata !71, metadata !DIExpression()), !dbg !72
  store i32 %__nv_MAIN_F1L29_2Arg0.arg, i32* %__nv_MAIN_F1L29_2Arg0.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L29_2Arg0.addr, metadata !73, metadata !DIExpression()), !dbg !72
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L29_2Arg1, metadata !74, metadata !DIExpression()), !dbg !72
  call void @llvm.dbg.value(metadata i32 1, metadata !75, metadata !DIExpression()), !dbg !72
  call void @llvm.dbg.value(metadata i32 3, metadata !76, metadata !DIExpression()), !dbg !72
  call void @llvm.dbg.value(metadata i32 0, metadata !77, metadata !DIExpression()), !dbg !72
  call void @llvm.dbg.value(metadata i32 1, metadata !78, metadata !DIExpression()), !dbg !72
  call void @llvm.dbg.value(metadata i32 3, metadata !79, metadata !DIExpression()), !dbg !72
  call void @llvm.dbg.value(metadata i32 0, metadata !80, metadata !DIExpression()), !dbg !72
  call void @llvm.dbg.value(metadata i32 1, metadata !81, metadata !DIExpression()), !dbg !72
  %0 = bitcast i64* %__nv_MAIN_F1L29_2Arg1 to i8**, !dbg !82
  %1 = load i8*, i8** %0, align 8, !dbg !82
  store i8* %1, i8** %.S0000_436, align 8, !dbg !82
  br label %L.LB5_440

L.LB5_440:                                        ; preds = %L.entry
  br label %L.LB5_323

L.LB5_323:                                        ; preds = %L.LB5_440
  %2 = bitcast %struct_drb158_0_* @_drb158_0_ to i32*, !dbg !83
  %3 = load i32, i32* %2, align 4, !dbg !83
  %4 = bitcast %struct_drb158_0_* @_drb158_0_ to i8*, !dbg !83
  %5 = getelementptr i8, i8* %4, i64 4, !dbg !83
  %6 = bitcast i8* %5 to i32*, !dbg !83
  %7 = load i32, i32* %6, align 4, !dbg !83
  %8 = sext i32 %7 to i64, !dbg !83
  %9 = bitcast %struct_drb158_0_* @_drb158_0_ to i8*, !dbg !83
  %10 = getelementptr i8, i8* %9, i64 12, !dbg !83
  %11 = bitcast i8* %10 to i32*, !dbg !83
  %12 = getelementptr i32, i32* %11, i64 %8, !dbg !83
  %13 = load i32, i32* %12, align 4, !dbg !83
  %14 = mul nsw i32 %3, %13, !dbg !83
  %15 = bitcast %struct_drb158_0_* @_drb158_0_ to i8*, !dbg !83
  %16 = getelementptr i8, i8* %15, i64 4, !dbg !83
  %17 = bitcast i8* %16 to i32*, !dbg !83
  %18 = load i32, i32* %17, align 4, !dbg !83
  %19 = sext i32 %18 to i64, !dbg !83
  %20 = bitcast %struct_drb158_0_* @_drb158_0_ to i8*, !dbg !83
  %21 = getelementptr i8, i8* %20, i64 12, !dbg !83
  %22 = bitcast i8* %21 to i32*, !dbg !83
  %23 = getelementptr i32, i32* %22, i64 %19, !dbg !83
  store i32 %14, i32* %23, align 4, !dbg !83
  br label %L.LB5_324

L.LB5_324:                                        ; preds = %L.LB5_323
  ret void, !dbg !84
}

define internal void @__nv_MAIN_F1L33_3_(i32 %__nv_MAIN_F1L33_3Arg0.arg, i64* %__nv_MAIN_F1L33_3Arg1) #1 !dbg !85 {
L.entry:
  %__nv_MAIN_F1L33_3Arg0.addr = alloca i32, align 4
  %.S0000_436 = alloca i8*, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L33_3Arg0.addr, metadata !86, metadata !DIExpression()), !dbg !87
  store i32 %__nv_MAIN_F1L33_3Arg0.arg, i32* %__nv_MAIN_F1L33_3Arg0.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L33_3Arg0.addr, metadata !88, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L33_3Arg1, metadata !89, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 1, metadata !90, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 3, metadata !91, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 0, metadata !92, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 1, metadata !93, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 3, metadata !94, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 0, metadata !95, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 1, metadata !96, metadata !DIExpression()), !dbg !87
  %0 = bitcast i64* %__nv_MAIN_F1L33_3Arg1 to i8**, !dbg !97
  %1 = load i8*, i8** %0, align 8, !dbg !97
  store i8* %1, i8** %.S0000_436, align 8, !dbg !97
  br label %L.LB6_446

L.LB6_446:                                        ; preds = %L.entry
  br label %L.LB6_327

L.LB6_327:                                        ; preds = %L.LB6_446
  %2 = bitcast %struct_drb158_0_* @_drb158_0_ to i8*, !dbg !98
  %3 = getelementptr i8, i8* %2, i64 4, !dbg !98
  %4 = bitcast i8* %3 to i32*, !dbg !98
  %5 = load i32, i32* %4, align 4, !dbg !98
  %6 = sext i32 %5 to i64, !dbg !98
  %7 = bitcast %struct_drb158_0_* @_drb158_0_ to i8*, !dbg !98
  %8 = getelementptr i8, i8* %7, i64 268, !dbg !98
  %9 = bitcast i8* %8 to i32*, !dbg !98
  %10 = getelementptr i32, i32* %9, i64 %6, !dbg !98
  %11 = load i32, i32* %10, align 4, !dbg !98
  %12 = bitcast %struct_drb158_0_* @_drb158_0_ to i8*, !dbg !98
  %13 = getelementptr i8, i8* %12, i64 4, !dbg !98
  %14 = bitcast i8* %13 to i32*, !dbg !98
  %15 = load i32, i32* %14, align 4, !dbg !98
  %16 = sext i32 %15 to i64, !dbg !98
  %17 = bitcast %struct_drb158_0_* @_drb158_0_ to i8*, !dbg !98
  %18 = getelementptr i8, i8* %17, i64 12, !dbg !98
  %19 = bitcast i8* %18 to i32*, !dbg !98
  %20 = getelementptr i32, i32* %19, i64 %16, !dbg !98
  %21 = load i32, i32* %20, align 4, !dbg !98
  %22 = add nsw i32 %11, %21, !dbg !98
  %23 = bitcast %struct_drb158_0_* @_drb158_0_ to i8*, !dbg !98
  %24 = getelementptr i8, i8* %23, i64 4, !dbg !98
  %25 = bitcast i8* %24 to i32*, !dbg !98
  %26 = load i32, i32* %25, align 4, !dbg !98
  %27 = sext i32 %26 to i64, !dbg !98
  %28 = bitcast %struct_drb158_0_* @_drb158_0_ to i8*, !dbg !98
  %29 = getelementptr i8, i8* %28, i64 12, !dbg !98
  %30 = bitcast i8* %29 to i32*, !dbg !98
  %31 = getelementptr i32, i32* %30, i64 %27, !dbg !98
  store i32 %22, i32* %31, align 4, !dbg !98
  br label %L.LB6_328

L.LB6_328:                                        ; preds = %L.LB6_327
  ret void, !dbg !99
}

declare void @__kmpc_omp_task(i64*, i32, i64*) #1

declare i8* @__kmpc_omp_task_alloc(i64*, i32, i32, i32, i32, i64*) #1

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

declare signext i32 @__kmpc_omp_taskwait(i64*, i32) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { noinline }
attributes #1 = { "no-frame-pointer-elim-non-leaf" }
attributes #2 = { nounwind readnone speculatable }

!llvm.module.flags = !{!22, !23}
!llvm.dbg.cu = !{!3}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !4, type: !9, isLocal: false, isDefinition: true)
!2 = !DIModule(scope: !3, name: "drb158")
!3 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !4, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !6, imports: !17)
!4 = !DIFile(filename: "micro-benchmarks-fortran/DRB158-missingtaskbarrier-orig-gpu-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!5 = !{}
!6 = !{!0, !7, !10, !15}
!7 = !DIGlobalVariableExpression(var: !8, expr: !DIExpression(DW_OP_plus_uconst, 4))
!8 = distinct !DIGlobalVariable(name: "i", scope: !2, file: !4, type: !9, isLocal: false, isDefinition: true)
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DIGlobalVariableExpression(var: !11, expr: !DIExpression(DW_OP_plus_uconst, 16))
!11 = distinct !DIGlobalVariable(name: "x", scope: !2, file: !4, type: !12, isLocal: false, isDefinition: true)
!12 = !DICompositeType(tag: DW_TAG_array_type, baseType: !9, size: 2048, align: 32, elements: !13)
!13 = !{!14}
!14 = !DISubrange(count: 64, lowerBound: 1)
!15 = !DIGlobalVariableExpression(var: !16, expr: !DIExpression(DW_OP_plus_uconst, 272))
!16 = distinct !DIGlobalVariable(name: "y", scope: !2, file: !4, type: !12, isLocal: false, isDefinition: true)
!17 = !{!18}
!18 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !19, entity: !2, file: !4, line: 15)
!19 = distinct !DISubprogram(name: "drb158_missingtaskbarrier_orig_gpu_no", scope: !3, file: !4, line: 15, type: !20, scopeLine: 15, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !3)
!20 = !DISubroutineType(cc: DW_CC_program, types: !21)
!21 = !{null}
!22 = !{i32 2, !"Dwarf Version", i32 4}
!23 = !{i32 2, !"Debug Info Version", i32 3}
!24 = !DILocalVariable(name: "omp_sched_static", scope: !19, file: !4, type: !9)
!25 = !DILocation(line: 0, scope: !19)
!26 = !DILocalVariable(name: "omp_sched_guided", scope: !19, file: !4, type: !9)
!27 = !DILocalVariable(name: "omp_proc_bind_false", scope: !19, file: !4, type: !9)
!28 = !DILocalVariable(name: "omp_proc_bind_true", scope: !19, file: !4, type: !9)
!29 = !DILocalVariable(name: "omp_proc_bind_close", scope: !19, file: !4, type: !9)
!30 = !DILocalVariable(name: "omp_lock_hint_none", scope: !19, file: !4, type: !9)
!31 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !19, file: !4, type: !9)
!32 = !DILocation(line: 47, column: 1, scope: !19)
!33 = !DILocation(line: 15, column: 1, scope: !19)
!34 = !DILocation(line: 20, column: 1, scope: !19)
!35 = !DILocation(line: 21, column: 1, scope: !19)
!36 = !DILocation(line: 22, column: 1, scope: !19)
!37 = !DILocation(line: 23, column: 1, scope: !19)
!38 = !DILocation(line: 25, column: 1, scope: !19)
!39 = !DILocation(line: 37, column: 1, scope: !19)
!40 = !DILocation(line: 39, column: 1, scope: !19)
!41 = !DILocation(line: 40, column: 1, scope: !19)
!42 = !DILocation(line: 41, column: 1, scope: !19)
!43 = !DILocalVariable(scope: !19, file: !4, type: !9, flags: DIFlagArtificial)
!44 = !DILocation(line: 43, column: 1, scope: !19)
!45 = !DILocation(line: 45, column: 1, scope: !19)
!46 = distinct !DISubprogram(name: "__nv_MAIN__F1L27_1", scope: !3, file: !4, line: 27, type: !47, scopeLine: 27, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !3)
!47 = !DISubroutineType(types: !48)
!48 = !{null, !9, !49, !49}
!49 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!50 = !DILocalVariable(name: "__nv_MAIN__F1L27_1Arg0", arg: 1, scope: !46, file: !4, type: !9)
!51 = !DILocation(line: 0, scope: !46)
!52 = !DILocalVariable(name: "__nv_MAIN__F1L27_1Arg1", arg: 2, scope: !46, file: !4, type: !49)
!53 = !DILocalVariable(name: "__nv_MAIN__F1L27_1Arg2", arg: 3, scope: !46, file: !4, type: !49)
!54 = !DILocalVariable(name: "omp_sched_static", scope: !46, file: !4, type: !9)
!55 = !DILocalVariable(name: "omp_sched_guided", scope: !46, file: !4, type: !9)
!56 = !DILocalVariable(name: "omp_proc_bind_false", scope: !46, file: !4, type: !9)
!57 = !DILocalVariable(name: "omp_proc_bind_true", scope: !46, file: !4, type: !9)
!58 = !DILocalVariable(name: "omp_proc_bind_close", scope: !46, file: !4, type: !9)
!59 = !DILocalVariable(name: "omp_lock_hint_none", scope: !46, file: !4, type: !9)
!60 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !46, file: !4, type: !9)
!61 = !DILocation(line: 37, column: 1, scope: !46)
!62 = !DILocation(line: 28, column: 1, scope: !46)
!63 = !DILocation(line: 29, column: 1, scope: !46)
!64 = !DILocation(line: 31, column: 1, scope: !46)
!65 = !DILocation(line: 33, column: 1, scope: !46)
!66 = !DILocation(line: 35, column: 1, scope: !46)
!67 = !DILocation(line: 36, column: 1, scope: !46)
!68 = distinct !DISubprogram(name: "__nv_MAIN_F1L29_2", scope: !3, file: !4, line: 29, type: !69, scopeLine: 29, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !3)
!69 = !DISubroutineType(types: !70)
!70 = !{null, !9, !49}
!71 = !DILocalVariable(name: "__nv_MAIN_F1L29_2Arg0", scope: !68, file: !4, type: !9)
!72 = !DILocation(line: 0, scope: !68)
!73 = !DILocalVariable(name: "__nv_MAIN_F1L29_2Arg0", arg: 1, scope: !68, file: !4, type: !9)
!74 = !DILocalVariable(name: "__nv_MAIN_F1L29_2Arg1", arg: 2, scope: !68, file: !4, type: !49)
!75 = !DILocalVariable(name: "omp_sched_static", scope: !68, file: !4, type: !9)
!76 = !DILocalVariable(name: "omp_sched_guided", scope: !68, file: !4, type: !9)
!77 = !DILocalVariable(name: "omp_proc_bind_false", scope: !68, file: !4, type: !9)
!78 = !DILocalVariable(name: "omp_proc_bind_true", scope: !68, file: !4, type: !9)
!79 = !DILocalVariable(name: "omp_proc_bind_close", scope: !68, file: !4, type: !9)
!80 = !DILocalVariable(name: "omp_lock_hint_none", scope: !68, file: !4, type: !9)
!81 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !68, file: !4, type: !9)
!82 = !DILocation(line: 29, column: 1, scope: !68)
!83 = !DILocation(line: 30, column: 1, scope: !68)
!84 = !DILocation(line: 31, column: 1, scope: !68)
!85 = distinct !DISubprogram(name: "__nv_MAIN_F1L33_3", scope: !3, file: !4, line: 33, type: !69, scopeLine: 33, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !3)
!86 = !DILocalVariable(name: "__nv_MAIN_F1L33_3Arg0", scope: !85, file: !4, type: !9)
!87 = !DILocation(line: 0, scope: !85)
!88 = !DILocalVariable(name: "__nv_MAIN_F1L33_3Arg0", arg: 1, scope: !85, file: !4, type: !9)
!89 = !DILocalVariable(name: "__nv_MAIN_F1L33_3Arg1", arg: 2, scope: !85, file: !4, type: !49)
!90 = !DILocalVariable(name: "omp_sched_static", scope: !85, file: !4, type: !9)
!91 = !DILocalVariable(name: "omp_sched_guided", scope: !85, file: !4, type: !9)
!92 = !DILocalVariable(name: "omp_proc_bind_false", scope: !85, file: !4, type: !9)
!93 = !DILocalVariable(name: "omp_proc_bind_true", scope: !85, file: !4, type: !9)
!94 = !DILocalVariable(name: "omp_proc_bind_close", scope: !85, file: !4, type: !9)
!95 = !DILocalVariable(name: "omp_lock_hint_none", scope: !85, file: !4, type: !9)
!96 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !85, file: !4, type: !9)
!97 = !DILocation(line: 33, column: 1, scope: !85)
!98 = !DILocation(line: 34, column: 1, scope: !85)
!99 = !DILocation(line: 35, column: 1, scope: !85)

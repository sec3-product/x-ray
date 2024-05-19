; ModuleID = '/tmp/DRB160-nobarrier-orig-gpu-yes-de5698.ll'
source_filename = "/tmp/DRB160-nobarrier-orig-gpu-yes-de5698.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct_drb160_0_ = type <{ [128 x i8] }>

@.C313_MAIN_ = internal constant i32 25
@.C284_MAIN_ = internal constant i64 0
@.C345_MAIN_ = internal constant i32 6
@.C342_MAIN_ = internal constant [58 x i8] c"micro-benchmarks-fortran/DRB160-nobarrier-orig-gpu-yes.f95"
@.C344_MAIN_ = internal constant i32 56
@.C330_MAIN_ = internal constant i32 100
@.C302_MAIN_ = internal constant i32 2
@.C301_MAIN_ = internal constant i32 8
@.C285_MAIN_ = internal constant i32 1
@.C283_MAIN_ = internal constant i32 0
@.C301___nv_MAIN__F1L31_1 = internal constant i32 8
@.C283___nv_MAIN__F1L31_1 = internal constant i32 0
@.C330___nv_MAIN__F1L31_1 = internal constant i32 100
@.C285___nv_MAIN__F1L31_1 = internal constant i32 1
@.C301___nv_MAIN_F1L32_2 = internal constant i32 8
@.C283___nv_MAIN_F1L32_2 = internal constant i32 0
@.C330___nv_MAIN_F1L32_2 = internal constant i32 100
@.C285___nv_MAIN_F1L32_2 = internal constant i32 1
@_drb160_0_ = common global %struct_drb160_0_ zeroinitializer, align 64, !dbg !0, !dbg !7, !dbg !10, !dbg !12, !dbg !14, !dbg !16, !dbg !21, !dbg !23

; Function Attrs: noinline
define float @drb160_() #0 {
.L.entry:
  ret float undef
}

define void @MAIN_() #1 !dbg !27 {
L.entry:
  %__gtid_MAIN__414 = alloca i32, align 4
  %.dY0001_357 = alloca i32, align 4
  %.dY0005_388 = alloca i32, align 4
  %.dY0006_391 = alloca i32, align 4
  %z__io_347 = alloca i32, align 4
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
  br label %L.LB2_394

L.LB2_394:                                        ; preds = %L.entry
  store i32 8, i32* %.dY0001_357, align 4, !dbg !45
  %3 = bitcast %struct_drb160_0_* @_drb160_0_ to i8*, !dbg !45
  %4 = getelementptr i8, i8* %3, i64 4, !dbg !45
  %5 = bitcast i8* %4 to i32*, !dbg !45
  store i32 1, i32* %5, align 4, !dbg !45
  br label %L.LB2_355

L.LB2_355:                                        ; preds = %L.LB2_355, %L.LB2_394
  %6 = bitcast %struct_drb160_0_* @_drb160_0_ to i8*, !dbg !46
  %7 = getelementptr i8, i8* %6, i64 4, !dbg !46
  %8 = bitcast i8* %7 to i32*, !dbg !46
  %9 = load i32, i32* %8, align 4, !dbg !46
  %10 = sext i32 %9 to i64, !dbg !46
  %11 = bitcast %struct_drb160_0_* @_drb160_0_ to i8*, !dbg !46
  %12 = getelementptr i8, i8* %11, i64 28, !dbg !46
  %13 = bitcast i8* %12 to i32*, !dbg !46
  %14 = getelementptr i32, i32* %13, i64 %10, !dbg !46
  store i32 0, i32* %14, align 4, !dbg !46
  %15 = bitcast %struct_drb160_0_* @_drb160_0_ to i8*, !dbg !47
  %16 = getelementptr i8, i8* %15, i64 4, !dbg !47
  %17 = bitcast i8* %16 to i32*, !dbg !47
  %18 = load i32, i32* %17, align 4, !dbg !47
  %19 = sext i32 %18 to i64, !dbg !47
  %20 = bitcast %struct_drb160_0_* @_drb160_0_ to i8*, !dbg !47
  %21 = getelementptr i8, i8* %20, i64 60, !dbg !47
  %22 = bitcast i8* %21 to i32*, !dbg !47
  %23 = getelementptr i32, i32* %22, i64 %19, !dbg !47
  store i32 2, i32* %23, align 4, !dbg !47
  %24 = bitcast %struct_drb160_0_* @_drb160_0_ to i8*, !dbg !48
  %25 = getelementptr i8, i8* %24, i64 4, !dbg !48
  %26 = bitcast i8* %25 to i32*, !dbg !48
  %27 = load i32, i32* %26, align 4, !dbg !48
  %28 = sext i32 %27 to i64, !dbg !48
  %29 = bitcast %struct_drb160_0_* @_drb160_0_ to i8*, !dbg !48
  %30 = getelementptr i8, i8* %29, i64 92, !dbg !48
  %31 = bitcast i8* %30 to i32*, !dbg !48
  %32 = getelementptr i32, i32* %31, i64 %28, !dbg !48
  store i32 0, i32* %32, align 4, !dbg !48
  %33 = bitcast %struct_drb160_0_* @_drb160_0_ to i8*, !dbg !49
  %34 = getelementptr i8, i8* %33, i64 4, !dbg !49
  %35 = bitcast i8* %34 to i32*, !dbg !49
  %36 = load i32, i32* %35, align 4, !dbg !49
  %37 = add nsw i32 %36, 1, !dbg !49
  %38 = bitcast %struct_drb160_0_* @_drb160_0_ to i8*, !dbg !49
  %39 = getelementptr i8, i8* %38, i64 4, !dbg !49
  %40 = bitcast i8* %39 to i32*, !dbg !49
  store i32 %37, i32* %40, align 4, !dbg !49
  %41 = load i32, i32* %.dY0001_357, align 4, !dbg !49
  %42 = sub nsw i32 %41, 1, !dbg !49
  store i32 %42, i32* %.dY0001_357, align 4, !dbg !49
  %43 = load i32, i32* %.dY0001_357, align 4, !dbg !49
  %44 = icmp sgt i32 %43, 0, !dbg !49
  br i1 %44, label %L.LB2_355, label %L.LB2_429, !dbg !49

L.LB2_429:                                        ; preds = %L.LB2_355
  %45 = bitcast %struct_drb160_0_* @_drb160_0_ to i32*, !dbg !50
  store i32 2, i32* %45, align 4, !dbg !50
  %46 = bitcast %struct_drb160_0_* @_drb160_0_ to i8*, !dbg !51
  %47 = getelementptr i8, i8* %46, i64 16, !dbg !51
  %48 = bitcast i8* %47 to i32*, !dbg !51
  store i32 0, i32* %48, align 4, !dbg !51
  call void @__nv_MAIN__F1L31_1_(i32* %__gtid_MAIN__414, i64* null, i64* null), !dbg !52
  store i32 100, i32* %.dY0005_388, align 4, !dbg !53
  %49 = bitcast %struct_drb160_0_* @_drb160_0_ to i8*, !dbg !53
  %50 = getelementptr i8, i8* %49, i64 4, !dbg !53
  %51 = bitcast i8* %50 to i32*, !dbg !53
  store i32 1, i32* %51, align 4, !dbg !53
  br label %L.LB2_386

L.LB2_386:                                        ; preds = %L.LB2_386, %L.LB2_429
  %52 = bitcast %struct_drb160_0_* @_drb160_0_ to i8*, !dbg !54
  %53 = getelementptr i8, i8* %52, i64 16, !dbg !54
  %54 = bitcast i8* %53 to i32*, !dbg !54
  %55 = load i32, i32* %54, align 4, !dbg !54
  %56 = add nsw i32 %55, 2, !dbg !54
  %57 = bitcast %struct_drb160_0_* @_drb160_0_ to i8*, !dbg !54
  %58 = getelementptr i8, i8* %57, i64 16, !dbg !54
  %59 = bitcast i8* %58 to i32*, !dbg !54
  store i32 %56, i32* %59, align 4, !dbg !54
  %60 = bitcast %struct_drb160_0_* @_drb160_0_ to i8*, !dbg !55
  %61 = getelementptr i8, i8* %60, i64 16, !dbg !55
  %62 = bitcast i8* %61 to i32*, !dbg !55
  %63 = load i32, i32* %62, align 4, !dbg !55
  %64 = mul nsw i32 %63, 2, !dbg !55
  %65 = bitcast %struct_drb160_0_* @_drb160_0_ to i8*, !dbg !55
  %66 = getelementptr i8, i8* %65, i64 16, !dbg !55
  %67 = bitcast i8* %66 to i32*, !dbg !55
  store i32 %64, i32* %67, align 4, !dbg !55
  %68 = bitcast %struct_drb160_0_* @_drb160_0_ to i8*, !dbg !56
  %69 = getelementptr i8, i8* %68, i64 4, !dbg !56
  %70 = bitcast i8* %69 to i32*, !dbg !56
  %71 = load i32, i32* %70, align 4, !dbg !56
  %72 = add nsw i32 %71, 1, !dbg !56
  %73 = bitcast %struct_drb160_0_* @_drb160_0_ to i8*, !dbg !56
  %74 = getelementptr i8, i8* %73, i64 4, !dbg !56
  %75 = bitcast i8* %74 to i32*, !dbg !56
  store i32 %72, i32* %75, align 4, !dbg !56
  %76 = load i32, i32* %.dY0005_388, align 4, !dbg !56
  %77 = sub nsw i32 %76, 1, !dbg !56
  store i32 %77, i32* %.dY0005_388, align 4, !dbg !56
  %78 = load i32, i32* %.dY0005_388, align 4, !dbg !56
  %79 = icmp sgt i32 %78, 0, !dbg !56
  br i1 %79, label %L.LB2_386, label %L.LB2_430, !dbg !56

L.LB2_430:                                        ; preds = %L.LB2_386
  store i32 8, i32* %.dY0006_391, align 4, !dbg !57
  %80 = bitcast %struct_drb160_0_* @_drb160_0_ to i8*, !dbg !57
  %81 = getelementptr i8, i8* %80, i64 4, !dbg !57
  %82 = bitcast i8* %81 to i32*, !dbg !57
  store i32 1, i32* %82, align 4, !dbg !57
  br label %L.LB2_389

L.LB2_389:                                        ; preds = %L.LB2_392, %L.LB2_430
  %83 = bitcast %struct_drb160_0_* @_drb160_0_ to i8*, !dbg !58
  %84 = getelementptr i8, i8* %83, i64 16, !dbg !58
  %85 = bitcast i8* %84 to i32*, !dbg !58
  %86 = load i32, i32* %85, align 4, !dbg !58
  %87 = bitcast %struct_drb160_0_* @_drb160_0_ to i8*, !dbg !58
  %88 = getelementptr i8, i8* %87, i64 4, !dbg !58
  %89 = bitcast i8* %88 to i32*, !dbg !58
  %90 = load i32, i32* %89, align 4, !dbg !58
  %91 = sext i32 %90 to i64, !dbg !58
  %92 = bitcast %struct_drb160_0_* @_drb160_0_ to i8*, !dbg !58
  %93 = getelementptr i8, i8* %92, i64 28, !dbg !58
  %94 = bitcast i8* %93 to i32*, !dbg !58
  %95 = getelementptr i32, i32* %94, i64 %91, !dbg !58
  %96 = load i32, i32* %95, align 4, !dbg !58
  %97 = icmp eq i32 %86, %96, !dbg !58
  br i1 %97, label %L.LB2_392, label %L.LB2_431, !dbg !58

L.LB2_431:                                        ; preds = %L.LB2_389
  call void (...) @_mp_bcs_nest(), !dbg !59
  %98 = bitcast i32* @.C344_MAIN_ to i8*, !dbg !59
  %99 = bitcast [58 x i8]* @.C342_MAIN_ to i8*, !dbg !59
  %100 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !59
  call void (i8*, i8*, i64, ...) %100(i8* %98, i8* %99, i64 58), !dbg !59
  %101 = bitcast i32* @.C345_MAIN_ to i8*, !dbg !59
  %102 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !59
  %103 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !59
  %104 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !59
  %105 = call i32 (i8*, i8*, i8*, i8*, ...) %104(i8* %101, i8* null, i8* %102, i8* %103), !dbg !59
  call void @llvm.dbg.declare(metadata i32* %z__io_347, metadata !60, metadata !DIExpression()), !dbg !33
  store i32 %105, i32* %z__io_347, align 4, !dbg !59
  %106 = bitcast %struct_drb160_0_* @_drb160_0_ to i8*, !dbg !59
  %107 = getelementptr i8, i8* %106, i64 4, !dbg !59
  %108 = bitcast i8* %107 to i32*, !dbg !59
  %109 = load i32, i32* %108, align 4, !dbg !59
  %110 = sext i32 %109 to i64, !dbg !59
  %111 = bitcast %struct_drb160_0_* @_drb160_0_ to i8*, !dbg !59
  %112 = getelementptr i8, i8* %111, i64 28, !dbg !59
  %113 = bitcast i8* %112 to i32*, !dbg !59
  %114 = getelementptr i32, i32* %113, i64 %110, !dbg !59
  %115 = load i32, i32* %114, align 4, !dbg !59
  %116 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !59
  %117 = call i32 (i32, i32, ...) %116(i32 %115, i32 25), !dbg !59
  store i32 %117, i32* %z__io_347, align 4, !dbg !59
  %118 = bitcast %struct_drb160_0_* @_drb160_0_ to i8*, !dbg !59
  %119 = getelementptr i8, i8* %118, i64 16, !dbg !59
  %120 = bitcast i8* %119 to i32*, !dbg !59
  %121 = load i32, i32* %120, align 4, !dbg !59
  %122 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !59
  %123 = call i32 (i32, i32, ...) %122(i32 %121, i32 25), !dbg !59
  store i32 %123, i32* %z__io_347, align 4, !dbg !59
  %124 = call i32 (...) @f90io_ldw_end(), !dbg !59
  store i32 %124, i32* %z__io_347, align 4, !dbg !59
  call void (...) @_mp_ecs_nest(), !dbg !59
  br label %L.LB2_392

L.LB2_392:                                        ; preds = %L.LB2_431, %L.LB2_389
  %125 = bitcast %struct_drb160_0_* @_drb160_0_ to i8*, !dbg !61
  %126 = getelementptr i8, i8* %125, i64 4, !dbg !61
  %127 = bitcast i8* %126 to i32*, !dbg !61
  %128 = load i32, i32* %127, align 4, !dbg !61
  %129 = add nsw i32 %128, 1, !dbg !61
  %130 = bitcast %struct_drb160_0_* @_drb160_0_ to i8*, !dbg !61
  %131 = getelementptr i8, i8* %130, i64 4, !dbg !61
  %132 = bitcast i8* %131 to i32*, !dbg !61
  store i32 %129, i32* %132, align 4, !dbg !61
  %133 = load i32, i32* %.dY0006_391, align 4, !dbg !61
  %134 = sub nsw i32 %133, 1, !dbg !61
  store i32 %134, i32* %.dY0006_391, align 4, !dbg !61
  %135 = load i32, i32* %.dY0006_391, align 4, !dbg !61
  %136 = icmp sgt i32 %135, 0, !dbg !61
  br i1 %136, label %L.LB2_389, label %L.LB2_432, !dbg !61

L.LB2_432:                                        ; preds = %L.LB2_392
  ret void, !dbg !43
}

define internal void @__nv_MAIN__F1L31_1_(i32* %__nv_MAIN__F1L31_1Arg0, i64* %__nv_MAIN__F1L31_1Arg1, i64* %__nv_MAIN__F1L31_1Arg2) #1 !dbg !62 {
L.entry:
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L31_1Arg0, metadata !66, metadata !DIExpression()), !dbg !67
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L31_1Arg1, metadata !68, metadata !DIExpression()), !dbg !67
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L31_1Arg2, metadata !69, metadata !DIExpression()), !dbg !67
  call void @llvm.dbg.value(metadata i32 8, metadata !70, metadata !DIExpression()), !dbg !67
  call void @llvm.dbg.value(metadata i32 1, metadata !71, metadata !DIExpression()), !dbg !67
  call void @llvm.dbg.value(metadata i32 2, metadata !72, metadata !DIExpression()), !dbg !67
  call void @llvm.dbg.value(metadata i32 0, metadata !73, metadata !DIExpression()), !dbg !67
  call void @llvm.dbg.value(metadata i32 1, metadata !74, metadata !DIExpression()), !dbg !67
  call void @llvm.dbg.value(metadata i32 2, metadata !75, metadata !DIExpression()), !dbg !67
  call void @llvm.dbg.value(metadata i32 0, metadata !76, metadata !DIExpression()), !dbg !67
  call void @llvm.dbg.value(metadata i32 1, metadata !77, metadata !DIExpression()), !dbg !67
  call void @llvm.dbg.value(metadata i32 2, metadata !78, metadata !DIExpression()), !dbg !67
  call void @llvm.dbg.value(metadata i32 8, metadata !79, metadata !DIExpression()), !dbg !67
  br label %L.LB3_436

L.LB3_436:                                        ; preds = %L.entry
  br label %L.LB3_326

L.LB3_326:                                        ; preds = %L.LB3_436
  %0 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN_F1L32_2_ to i64*, !dbg !80
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_teams(i64* null, i32 1, i64* %0, i64* %__nv_MAIN__F1L31_1Arg2), !dbg !80
  br label %L.LB3_340

L.LB3_340:                                        ; preds = %L.LB3_326
  ret void, !dbg !81
}

define internal void @__nv_MAIN_F1L32_2_(i32* %__nv_MAIN_F1L32_2Arg0, i64* %__nv_MAIN_F1L32_2Arg1, i64* %__nv_MAIN_F1L32_2Arg2) #1 !dbg !82 {
L.entry:
  %__gtid___nv_MAIN_F1L32_2__469 = alloca i32, align 4
  %.dY0002_360 = alloca i32, align 4
  %.i0000p_333 = alloca i32, align 4
  %j_332 = alloca i32, align 4
  %.du0003_364 = alloca i32, align 4
  %.de0003_365 = alloca i32, align 4
  %.di0003_366 = alloca i32, align 4
  %.ds0003_367 = alloca i32, align 4
  %.dl0003_369 = alloca i32, align 4
  %.dl0003.copy_463 = alloca i32, align 4
  %.de0003.copy_464 = alloca i32, align 4
  %.ds0003.copy_465 = alloca i32, align 4
  %.dX0003_368 = alloca i32, align 4
  %.dY0003_363 = alloca i32, align 4
  %.i0001p_337 = alloca i32, align 4
  %j_336 = alloca i32, align 4
  %.du0004_376 = alloca i32, align 4
  %.de0004_377 = alloca i32, align 4
  %.di0004_378 = alloca i32, align 4
  %.ds0004_379 = alloca i32, align 4
  %.dl0004_381 = alloca i32, align 4
  %.dl0004.copy_495 = alloca i32, align 4
  %.de0004.copy_496 = alloca i32, align 4
  %.ds0004.copy_497 = alloca i32, align 4
  %.dX0004_380 = alloca i32, align 4
  %.dY0004_375 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L32_2Arg0, metadata !83, metadata !DIExpression()), !dbg !84
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L32_2Arg1, metadata !85, metadata !DIExpression()), !dbg !84
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L32_2Arg2, metadata !86, metadata !DIExpression()), !dbg !84
  call void @llvm.dbg.value(metadata i32 8, metadata !87, metadata !DIExpression()), !dbg !84
  call void @llvm.dbg.value(metadata i32 1, metadata !88, metadata !DIExpression()), !dbg !84
  call void @llvm.dbg.value(metadata i32 2, metadata !89, metadata !DIExpression()), !dbg !84
  call void @llvm.dbg.value(metadata i32 0, metadata !90, metadata !DIExpression()), !dbg !84
  call void @llvm.dbg.value(metadata i32 1, metadata !91, metadata !DIExpression()), !dbg !84
  call void @llvm.dbg.value(metadata i32 2, metadata !92, metadata !DIExpression()), !dbg !84
  call void @llvm.dbg.value(metadata i32 0, metadata !93, metadata !DIExpression()), !dbg !84
  call void @llvm.dbg.value(metadata i32 1, metadata !94, metadata !DIExpression()), !dbg !84
  call void @llvm.dbg.value(metadata i32 2, metadata !95, metadata !DIExpression()), !dbg !84
  call void @llvm.dbg.value(metadata i32 8, metadata !96, metadata !DIExpression()), !dbg !84
  %0 = load i32, i32* %__nv_MAIN_F1L32_2Arg0, align 4, !dbg !97
  store i32 %0, i32* %__gtid___nv_MAIN_F1L32_2__469, align 4, !dbg !97
  br label %L.LB5_454

L.LB5_454:                                        ; preds = %L.entry
  br label %L.LB5_329

L.LB5_329:                                        ; preds = %L.LB5_454
  store i32 100, i32* %.dY0002_360, align 4, !dbg !98
  %1 = bitcast %struct_drb160_0_* @_drb160_0_ to i8*, !dbg !98
  %2 = getelementptr i8, i8* %1, i64 4, !dbg !98
  %3 = bitcast i8* %2 to i32*, !dbg !98
  store i32 1, i32* %3, align 4, !dbg !98
  br label %L.LB5_358

L.LB5_358:                                        ; preds = %L.LB5_338, %L.LB5_329
  br label %L.LB5_331

L.LB5_331:                                        ; preds = %L.LB5_358
  store i32 0, i32* %.i0000p_333, align 4, !dbg !99
  call void @llvm.dbg.declare(metadata i32* %j_332, metadata !100, metadata !DIExpression()), !dbg !97
  store i32 1, i32* %j_332, align 4, !dbg !99
  store i32 8, i32* %.du0003_364, align 4, !dbg !99
  store i32 8, i32* %.de0003_365, align 4, !dbg !99
  store i32 1, i32* %.di0003_366, align 4, !dbg !99
  %4 = load i32, i32* %.di0003_366, align 4, !dbg !99
  store i32 %4, i32* %.ds0003_367, align 4, !dbg !99
  store i32 1, i32* %.dl0003_369, align 4, !dbg !99
  %5 = load i32, i32* %.dl0003_369, align 4, !dbg !99
  store i32 %5, i32* %.dl0003.copy_463, align 4, !dbg !99
  %6 = load i32, i32* %.de0003_365, align 4, !dbg !99
  store i32 %6, i32* %.de0003.copy_464, align 4, !dbg !99
  %7 = load i32, i32* %.ds0003_367, align 4, !dbg !99
  store i32 %7, i32* %.ds0003.copy_465, align 4, !dbg !99
  %8 = load i32, i32* %__gtid___nv_MAIN_F1L32_2__469, align 4, !dbg !99
  %9 = bitcast i32* %.i0000p_333 to i64*, !dbg !99
  %10 = bitcast i32* %.dl0003.copy_463 to i64*, !dbg !99
  %11 = bitcast i32* %.de0003.copy_464 to i64*, !dbg !99
  %12 = bitcast i32* %.ds0003.copy_465 to i64*, !dbg !99
  %13 = load i32, i32* %.ds0003.copy_465, align 4, !dbg !99
  call void @__kmpc_for_static_init_4(i64* null, i32 %8, i32 92, i64* %9, i64* %10, i64* %11, i64* %12, i32 %13, i32 1), !dbg !99
  %14 = load i32, i32* %.dl0003.copy_463, align 4, !dbg !99
  store i32 %14, i32* %.dl0003_369, align 4, !dbg !99
  %15 = load i32, i32* %.de0003.copy_464, align 4, !dbg !99
  store i32 %15, i32* %.de0003_365, align 4, !dbg !99
  %16 = load i32, i32* %.ds0003.copy_465, align 4, !dbg !99
  store i32 %16, i32* %.ds0003_367, align 4, !dbg !99
  %17 = load i32, i32* %.dl0003_369, align 4, !dbg !99
  store i32 %17, i32* %j_332, align 4, !dbg !99
  %18 = load i32, i32* %j_332, align 4, !dbg !99
  call void @llvm.dbg.value(metadata i32 %18, metadata !100, metadata !DIExpression()), !dbg !97
  store i32 %18, i32* %.dX0003_368, align 4, !dbg !99
  %19 = load i32, i32* %.dX0003_368, align 4, !dbg !99
  %20 = load i32, i32* %.du0003_364, align 4, !dbg !99
  %21 = icmp sgt i32 %19, %20, !dbg !99
  br i1 %21, label %L.LB5_362, label %L.LB5_507, !dbg !99

L.LB5_507:                                        ; preds = %L.LB5_331
  %22 = load i32, i32* %.du0003_364, align 4, !dbg !99
  %23 = load i32, i32* %.de0003_365, align 4, !dbg !99
  %24 = icmp slt i32 %22, %23, !dbg !99
  %25 = select i1 %24, i32 %22, i32 %23, !dbg !99
  store i32 %25, i32* %.de0003_365, align 4, !dbg !99
  %26 = load i32, i32* %.dX0003_368, align 4, !dbg !99
  store i32 %26, i32* %j_332, align 4, !dbg !99
  %27 = load i32, i32* %.di0003_366, align 4, !dbg !99
  %28 = load i32, i32* %.de0003_365, align 4, !dbg !99
  %29 = load i32, i32* %.dX0003_368, align 4, !dbg !99
  %30 = sub nsw i32 %28, %29, !dbg !99
  %31 = add nsw i32 %27, %30, !dbg !99
  %32 = load i32, i32* %.di0003_366, align 4, !dbg !99
  %33 = sdiv i32 %31, %32, !dbg !99
  store i32 %33, i32* %.dY0003_363, align 4, !dbg !99
  %34 = load i32, i32* %.dY0003_363, align 4, !dbg !99
  %35 = icmp sle i32 %34, 0, !dbg !99
  br i1 %35, label %L.LB5_372, label %L.LB5_371, !dbg !99

L.LB5_371:                                        ; preds = %L.LB5_371, %L.LB5_507
  %36 = load i32, i32* %j_332, align 4, !dbg !101
  call void @llvm.dbg.value(metadata i32 %36, metadata !100, metadata !DIExpression()), !dbg !97
  %37 = sext i32 %36 to i64, !dbg !101
  %38 = bitcast %struct_drb160_0_* @_drb160_0_ to i8*, !dbg !101
  %39 = getelementptr i8, i8* %38, i64 60, !dbg !101
  %40 = bitcast i8* %39 to i32*, !dbg !101
  %41 = getelementptr i32, i32* %40, i64 %37, !dbg !101
  %42 = load i32, i32* %41, align 4, !dbg !101
  %43 = load i32, i32* %j_332, align 4, !dbg !101
  call void @llvm.dbg.value(metadata i32 %43, metadata !100, metadata !DIExpression()), !dbg !97
  %44 = sext i32 %43 to i64, !dbg !101
  %45 = bitcast %struct_drb160_0_* @_drb160_0_ to i8*, !dbg !101
  %46 = getelementptr i8, i8* %45, i64 28, !dbg !101
  %47 = bitcast i8* %46 to i32*, !dbg !101
  %48 = getelementptr i32, i32* %47, i64 %44, !dbg !101
  %49 = load i32, i32* %48, align 4, !dbg !101
  %50 = add nsw i32 %42, %49, !dbg !101
  %51 = load i32, i32* %j_332, align 4, !dbg !101
  call void @llvm.dbg.value(metadata i32 %51, metadata !100, metadata !DIExpression()), !dbg !97
  %52 = sext i32 %51 to i64, !dbg !101
  %53 = bitcast %struct_drb160_0_* @_drb160_0_ to i8*, !dbg !101
  %54 = getelementptr i8, i8* %53, i64 92, !dbg !101
  %55 = bitcast i8* %54 to i32*, !dbg !101
  %56 = getelementptr i32, i32* %55, i64 %52, !dbg !101
  store i32 %50, i32* %56, align 4, !dbg !101
  %57 = load i32, i32* %.di0003_366, align 4, !dbg !102
  %58 = load i32, i32* %j_332, align 4, !dbg !102
  call void @llvm.dbg.value(metadata i32 %58, metadata !100, metadata !DIExpression()), !dbg !97
  %59 = add nsw i32 %57, %58, !dbg !102
  store i32 %59, i32* %j_332, align 4, !dbg !102
  %60 = load i32, i32* %.dY0003_363, align 4, !dbg !102
  %61 = sub nsw i32 %60, 1, !dbg !102
  store i32 %61, i32* %.dY0003_363, align 4, !dbg !102
  %62 = load i32, i32* %.dY0003_363, align 4, !dbg !102
  %63 = icmp sgt i32 %62, 0, !dbg !102
  br i1 %63, label %L.LB5_371, label %L.LB5_372, !dbg !102

L.LB5_372:                                        ; preds = %L.LB5_371, %L.LB5_507
  br label %L.LB5_362

L.LB5_362:                                        ; preds = %L.LB5_372, %L.LB5_331
  %64 = load i32, i32* %__gtid___nv_MAIN_F1L32_2__469, align 4, !dbg !102
  call void @__kmpc_for_static_fini(i64* null, i32 %64), !dbg !102
  br label %L.LB5_334

L.LB5_334:                                        ; preds = %L.LB5_362
  br label %L.LB5_335

L.LB5_335:                                        ; preds = %L.LB5_334
  store i32 0, i32* %.i0001p_337, align 4, !dbg !103
  call void @llvm.dbg.declare(metadata i32* %j_336, metadata !100, metadata !DIExpression()), !dbg !97
  store i32 8, i32* %j_336, align 4, !dbg !103
  store i32 1, i32* %.du0004_376, align 4, !dbg !103
  store i32 1, i32* %.de0004_377, align 4, !dbg !103
  %65 = bitcast %struct_drb160_0_* @_drb160_0_ to i8*, !dbg !103
  %66 = getelementptr i8, i8* %65, i64 12, !dbg !103
  %67 = bitcast i8* %66 to i32*, !dbg !103
  %68 = load i32, i32* %67, align 4, !dbg !103
  %69 = sub nsw i32 %68, 1, !dbg !103
  store i32 %69, i32* %.di0004_378, align 4, !dbg !103
  %70 = load i32, i32* %.di0004_378, align 4, !dbg !103
  store i32 %70, i32* %.ds0004_379, align 4, !dbg !103
  store i32 8, i32* %.dl0004_381, align 4, !dbg !103
  %71 = load i32, i32* %.dl0004_381, align 4, !dbg !103
  store i32 %71, i32* %.dl0004.copy_495, align 4, !dbg !103
  %72 = load i32, i32* %.de0004_377, align 4, !dbg !103
  store i32 %72, i32* %.de0004.copy_496, align 4, !dbg !103
  %73 = load i32, i32* %.ds0004_379, align 4, !dbg !103
  store i32 %73, i32* %.ds0004.copy_497, align 4, !dbg !103
  %74 = load i32, i32* %__gtid___nv_MAIN_F1L32_2__469, align 4, !dbg !103
  %75 = bitcast i32* %.i0001p_337 to i64*, !dbg !103
  %76 = bitcast i32* %.dl0004.copy_495 to i64*, !dbg !103
  %77 = bitcast i32* %.de0004.copy_496 to i64*, !dbg !103
  %78 = bitcast i32* %.ds0004.copy_497 to i64*, !dbg !103
  %79 = load i32, i32* %.ds0004.copy_497, align 4, !dbg !103
  call void @__kmpc_for_static_init_4(i64* null, i32 %74, i32 92, i64* %75, i64* %76, i64* %77, i64* %78, i32 %79, i32 1), !dbg !103
  %80 = load i32, i32* %.dl0004.copy_495, align 4, !dbg !103
  store i32 %80, i32* %.dl0004_381, align 4, !dbg !103
  %81 = load i32, i32* %.de0004.copy_496, align 4, !dbg !103
  store i32 %81, i32* %.de0004_377, align 4, !dbg !103
  %82 = load i32, i32* %.ds0004.copy_497, align 4, !dbg !103
  store i32 %82, i32* %.ds0004_379, align 4, !dbg !103
  %83 = load i32, i32* %.dl0004_381, align 4, !dbg !103
  store i32 %83, i32* %j_336, align 4, !dbg !103
  %84 = load i32, i32* %j_336, align 4, !dbg !103
  call void @llvm.dbg.value(metadata i32 %84, metadata !100, metadata !DIExpression()), !dbg !97
  store i32 %84, i32* %.dX0004_380, align 4, !dbg !103
  %85 = load i32, i32* %.ds0004_379, align 4, !dbg !103
  %86 = icmp slt i32 %85, 0, !dbg !103
  br i1 %86, label %L.LB5_382, label %L.LB5_508, !dbg !103

L.LB5_508:                                        ; preds = %L.LB5_335
  %87 = load i32, i32* %.dX0004_380, align 4, !dbg !103
  %88 = load i32, i32* %.du0004_376, align 4, !dbg !103
  %89 = icmp sgt i32 %87, %88, !dbg !103
  br i1 %89, label %L.LB5_374, label %L.LB5_509, !dbg !103

L.LB5_509:                                        ; preds = %L.LB5_508
  %90 = load i32, i32* %.du0004_376, align 4, !dbg !103
  %91 = load i32, i32* %.de0004_377, align 4, !dbg !103
  %92 = icmp slt i32 %90, %91, !dbg !103
  %93 = select i1 %92, i32 %90, i32 %91, !dbg !103
  store i32 %93, i32* %.de0004_377, align 4, !dbg !103
  br label %L.LB5_383, !dbg !103

L.LB5_382:                                        ; preds = %L.LB5_335
  %94 = load i32, i32* %.dX0004_380, align 4, !dbg !103
  %95 = load i32, i32* %.du0004_376, align 4, !dbg !103
  %96 = icmp slt i32 %94, %95, !dbg !103
  br i1 %96, label %L.LB5_374, label %L.LB5_510, !dbg !103

L.LB5_510:                                        ; preds = %L.LB5_382
  %97 = load i32, i32* %.du0004_376, align 4, !dbg !103
  %98 = load i32, i32* %.de0004_377, align 4, !dbg !103
  %99 = icmp sgt i32 %97, %98, !dbg !103
  %100 = select i1 %99, i32 %97, i32 %98, !dbg !103
  store i32 %100, i32* %.de0004_377, align 4, !dbg !103
  br label %L.LB5_383, !dbg !103

L.LB5_383:                                        ; preds = %L.LB5_510, %L.LB5_509
  %101 = load i32, i32* %.dX0004_380, align 4, !dbg !103
  store i32 %101, i32* %j_336, align 4, !dbg !103
  %102 = load i32, i32* %.di0004_378, align 4, !dbg !103
  %103 = load i32, i32* %.de0004_377, align 4, !dbg !103
  %104 = load i32, i32* %.dX0004_380, align 4, !dbg !103
  %105 = sub nsw i32 %103, %104, !dbg !103
  %106 = add nsw i32 %102, %105, !dbg !103
  %107 = load i32, i32* %.di0004_378, align 4, !dbg !103
  %108 = sdiv i32 %106, %107, !dbg !103
  store i32 %108, i32* %.dY0004_375, align 4, !dbg !103
  %109 = load i32, i32* %.dY0004_375, align 4, !dbg !103
  %110 = icmp sle i32 %109, 0, !dbg !103
  br i1 %110, label %L.LB5_385, label %L.LB5_384, !dbg !103

L.LB5_384:                                        ; preds = %L.LB5_384, %L.LB5_383
  %111 = load i32, i32* %j_336, align 4, !dbg !104
  call void @llvm.dbg.value(metadata i32 %111, metadata !100, metadata !DIExpression()), !dbg !97
  %112 = sext i32 %111 to i64, !dbg !104
  %113 = bitcast %struct_drb160_0_* @_drb160_0_ to i8*, !dbg !104
  %114 = getelementptr i8, i8* %113, i64 92, !dbg !104
  %115 = bitcast i8* %114 to i32*, !dbg !104
  %116 = getelementptr i32, i32* %115, i64 %112, !dbg !104
  %117 = load i32, i32* %116, align 4, !dbg !104
  %118 = bitcast %struct_drb160_0_* @_drb160_0_ to i32*, !dbg !104
  %119 = load i32, i32* %118, align 4, !dbg !104
  %120 = mul nsw i32 %117, %119, !dbg !104
  %121 = load i32, i32* %j_336, align 4, !dbg !104
  call void @llvm.dbg.value(metadata i32 %121, metadata !100, metadata !DIExpression()), !dbg !97
  %122 = sext i32 %121 to i64, !dbg !104
  %123 = bitcast %struct_drb160_0_* @_drb160_0_ to i8*, !dbg !104
  %124 = getelementptr i8, i8* %123, i64 28, !dbg !104
  %125 = bitcast i8* %124 to i32*, !dbg !104
  %126 = getelementptr i32, i32* %125, i64 %122, !dbg !104
  store i32 %120, i32* %126, align 4, !dbg !104
  %127 = load i32, i32* %.di0004_378, align 4, !dbg !105
  %128 = load i32, i32* %j_336, align 4, !dbg !105
  call void @llvm.dbg.value(metadata i32 %128, metadata !100, metadata !DIExpression()), !dbg !97
  %129 = add nsw i32 %127, %128, !dbg !105
  store i32 %129, i32* %j_336, align 4, !dbg !105
  %130 = load i32, i32* %.dY0004_375, align 4, !dbg !105
  %131 = sub nsw i32 %130, 1, !dbg !105
  store i32 %131, i32* %.dY0004_375, align 4, !dbg !105
  %132 = load i32, i32* %.dY0004_375, align 4, !dbg !105
  %133 = icmp sgt i32 %132, 0, !dbg !105
  br i1 %133, label %L.LB5_384, label %L.LB5_385, !dbg !105

L.LB5_385:                                        ; preds = %L.LB5_384, %L.LB5_383
  br label %L.LB5_374

L.LB5_374:                                        ; preds = %L.LB5_385, %L.LB5_382, %L.LB5_508
  %134 = load i32, i32* %__gtid___nv_MAIN_F1L32_2__469, align 4, !dbg !105
  call void @__kmpc_for_static_fini(i64* null, i32 %134), !dbg !105
  br label %L.LB5_338

L.LB5_338:                                        ; preds = %L.LB5_374
  %135 = bitcast %struct_drb160_0_* @_drb160_0_ to i8*, !dbg !106
  %136 = getelementptr i8, i8* %135, i64 4, !dbg !106
  %137 = bitcast i8* %136 to i32*, !dbg !106
  %138 = load i32, i32* %137, align 4, !dbg !106
  %139 = add nsw i32 %138, 1, !dbg !106
  %140 = bitcast %struct_drb160_0_* @_drb160_0_ to i8*, !dbg !106
  %141 = getelementptr i8, i8* %140, i64 4, !dbg !106
  %142 = bitcast i8* %141 to i32*, !dbg !106
  store i32 %139, i32* %142, align 4, !dbg !106
  %143 = load i32, i32* %.dY0002_360, align 4, !dbg !106
  %144 = sub nsw i32 %143, 1, !dbg !106
  store i32 %144, i32* %.dY0002_360, align 4, !dbg !106
  %145 = load i32, i32* %.dY0002_360, align 4, !dbg !106
  %146 = icmp sgt i32 %145, 0, !dbg !106
  br i1 %146, label %L.LB5_358, label %L.LB5_339, !dbg !106

L.LB5_339:                                        ; preds = %L.LB5_338
  ret void, !dbg !97
}

declare void @__kmpc_for_static_fini(i64*, i32) #1

declare void @__kmpc_for_static_init_4(i64*, i32, i32, i64*, i64*, i64*, i64*, i32, i32) #1

declare void @__kmpc_fork_teams(i64*, i32, i64*, i64*, ...) #1

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
!2 = !DIModule(scope: !3, name: "drb160")
!3 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !4, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !6, imports: !25)
!4 = !DIFile(filename: "micro-benchmarks-fortran/DRB160-nobarrier-orig-gpu-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
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
!26 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !27, entity: !2, file: !4, line: 17)
!27 = distinct !DISubprogram(name: "drb160_nobarrier_orig_gpu_yes", scope: !3, file: !4, line: 17, type: !28, scopeLine: 17, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !3)
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
!43 = !DILocation(line: 60, column: 1, scope: !27)
!44 = !DILocation(line: 17, column: 1, scope: !27)
!45 = !DILocation(line: 22, column: 1, scope: !27)
!46 = !DILocation(line: 23, column: 1, scope: !27)
!47 = !DILocation(line: 24, column: 1, scope: !27)
!48 = !DILocation(line: 25, column: 1, scope: !27)
!49 = !DILocation(line: 26, column: 1, scope: !27)
!50 = !DILocation(line: 28, column: 1, scope: !27)
!51 = !DILocation(line: 29, column: 1, scope: !27)
!52 = !DILocation(line: 47, column: 1, scope: !27)
!53 = !DILocation(line: 49, column: 1, scope: !27)
!54 = !DILocation(line: 50, column: 1, scope: !27)
!55 = !DILocation(line: 51, column: 1, scope: !27)
!56 = !DILocation(line: 52, column: 1, scope: !27)
!57 = !DILocation(line: 54, column: 1, scope: !27)
!58 = !DILocation(line: 55, column: 1, scope: !27)
!59 = !DILocation(line: 56, column: 1, scope: !27)
!60 = !DILocalVariable(scope: !27, file: !4, type: !9, flags: DIFlagArtificial)
!61 = !DILocation(line: 58, column: 1, scope: !27)
!62 = distinct !DISubprogram(name: "__nv_MAIN__F1L31_1", scope: !3, file: !4, line: 31, type: !63, scopeLine: 31, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !3)
!63 = !DISubroutineType(types: !64)
!64 = !{null, !9, !65, !65}
!65 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!66 = !DILocalVariable(name: "__nv_MAIN__F1L31_1Arg0", arg: 1, scope: !62, file: !4, type: !9)
!67 = !DILocation(line: 0, scope: !62)
!68 = !DILocalVariable(name: "__nv_MAIN__F1L31_1Arg1", arg: 2, scope: !62, file: !4, type: !65)
!69 = !DILocalVariable(name: "__nv_MAIN__F1L31_1Arg2", arg: 3, scope: !62, file: !4, type: !65)
!70 = !DILocalVariable(name: "omp_nest_lock_kind", scope: !62, file: !4, type: !9)
!71 = !DILocalVariable(name: "omp_sched_static", scope: !62, file: !4, type: !9)
!72 = !DILocalVariable(name: "omp_sched_dynamic", scope: !62, file: !4, type: !9)
!73 = !DILocalVariable(name: "omp_proc_bind_false", scope: !62, file: !4, type: !9)
!74 = !DILocalVariable(name: "omp_proc_bind_true", scope: !62, file: !4, type: !9)
!75 = !DILocalVariable(name: "omp_proc_bind_master", scope: !62, file: !4, type: !9)
!76 = !DILocalVariable(name: "omp_lock_hint_none", scope: !62, file: !4, type: !9)
!77 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !62, file: !4, type: !9)
!78 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !62, file: !4, type: !9)
!79 = !DILocalVariable(name: "omp_lock_hint_speculative", scope: !62, file: !4, type: !9)
!80 = !DILocation(line: 32, column: 1, scope: !62)
!81 = !DILocation(line: 47, column: 1, scope: !62)
!82 = distinct !DISubprogram(name: "__nv_MAIN_F1L32_2", scope: !3, file: !4, line: 32, type: !63, scopeLine: 32, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !3)
!83 = !DILocalVariable(name: "__nv_MAIN_F1L32_2Arg0", arg: 1, scope: !82, file: !4, type: !9)
!84 = !DILocation(line: 0, scope: !82)
!85 = !DILocalVariable(name: "__nv_MAIN_F1L32_2Arg1", arg: 2, scope: !82, file: !4, type: !65)
!86 = !DILocalVariable(name: "__nv_MAIN_F1L32_2Arg2", arg: 3, scope: !82, file: !4, type: !65)
!87 = !DILocalVariable(name: "omp_nest_lock_kind", scope: !82, file: !4, type: !9)
!88 = !DILocalVariable(name: "omp_sched_static", scope: !82, file: !4, type: !9)
!89 = !DILocalVariable(name: "omp_sched_dynamic", scope: !82, file: !4, type: !9)
!90 = !DILocalVariable(name: "omp_proc_bind_false", scope: !82, file: !4, type: !9)
!91 = !DILocalVariable(name: "omp_proc_bind_true", scope: !82, file: !4, type: !9)
!92 = !DILocalVariable(name: "omp_proc_bind_master", scope: !82, file: !4, type: !9)
!93 = !DILocalVariable(name: "omp_lock_hint_none", scope: !82, file: !4, type: !9)
!94 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !82, file: !4, type: !9)
!95 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !82, file: !4, type: !9)
!96 = !DILocalVariable(name: "omp_lock_hint_speculative", scope: !82, file: !4, type: !9)
!97 = !DILocation(line: 46, column: 1, scope: !82)
!98 = !DILocation(line: 33, column: 1, scope: !82)
!99 = !DILocation(line: 35, column: 1, scope: !82)
!100 = !DILocalVariable(name: "j", scope: !82, file: !4, type: !9)
!101 = !DILocation(line: 36, column: 1, scope: !82)
!102 = !DILocation(line: 37, column: 1, scope: !82)
!103 = !DILocation(line: 41, column: 1, scope: !82)
!104 = !DILocation(line: 42, column: 1, scope: !82)
!105 = !DILocation(line: 43, column: 1, scope: !82)
!106 = !DILocation(line: 45, column: 1, scope: !82)

; ModuleID = '/tmp/DRB050-functionparameter-orig-no-c8ad17.ll'
source_filename = "/tmp/DRB050-functionparameter-orig-no-c8ad17.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct_drb050_2_ = type <{ [16 x i8] }>
%struct_drb050_0_ = type <{ [288 x i8] }>
%astruct.dt88 = type <{ i8*, i8*, i8*, i8*, i8*, i8*, i8* }>

@.C294_drb050_foo1_ = internal constant double 5.000000e-01
@.C285_drb050_foo1_ = internal constant i32 1
@.C283_drb050_foo1_ = internal constant i32 0
@.C294___nv_drb050_foo1__F1L23_1 = internal constant double 5.000000e-01
@.C285___nv_drb050_foo1__F1L23_1 = internal constant i32 1
@.C283___nv_drb050_foo1__F1L23_1 = internal constant i32 0
@.C329_MAIN_ = internal constant i32 100
@.C306_MAIN_ = internal constant i32 28
@.C310_MAIN_ = internal constant i64 8
@.C332_MAIN_ = internal constant i64 28
@.C284_MAIN_ = internal constant i64 0
@.C330_MAIN_ = internal constant i64 100
@.C309_MAIN_ = internal constant i64 12
@.C286_MAIN_ = internal constant i64 1
@.C308_MAIN_ = internal constant i64 11
@.C283_MAIN_ = internal constant i32 0
@_drb050_2_ = common global %struct_drb050_2_ zeroinitializer, align 64, !dbg !0, !dbg !23
@_drb050_0_ = common global %struct_drb050_0_ zeroinitializer, align 64, !dbg !7, !dbg !13, !dbg !19, !dbg !21

; Function Attrs: noinline
define float @drb050_() #0 {
.L.entry:
  ret float undef
}

define void @drb050_foo1_(i64* %"o1$p10", i64* %"c$p14", i64* %len, i64* %"o1$sd9", i64* %"c$sd13") #1 !dbg !32 {
L.entry:
  %__gtid_drb050_foo1__404 = alloca i32, align 4
  %.uplevelArgPack0001_382 = alloca %astruct.dt88, align 16
  call void @llvm.dbg.declare(metadata i64* %"o1$p10", metadata !36, metadata !DIExpression(DW_OP_deref)), !dbg !37
  call void @llvm.dbg.declare(metadata i64* %"c$p14", metadata !38, metadata !DIExpression(DW_OP_deref)), !dbg !37
  call void @llvm.dbg.declare(metadata i64* %len, metadata !39, metadata !DIExpression()), !dbg !37
  call void @llvm.dbg.declare(metadata i64* %"o1$sd9", metadata !40, metadata !DIExpression()), !dbg !37
  call void @llvm.dbg.declare(metadata i64* %"c$sd13", metadata !41, metadata !DIExpression()), !dbg !37
  call void @llvm.dbg.value(metadata i32 1, metadata !42, metadata !DIExpression()), !dbg !37
  call void @llvm.dbg.value(metadata i32 0, metadata !43, metadata !DIExpression()), !dbg !37
  call void @llvm.dbg.value(metadata i32 1, metadata !44, metadata !DIExpression()), !dbg !37
  call void @llvm.dbg.value(metadata i32 0, metadata !45, metadata !DIExpression()), !dbg !37
  call void @llvm.dbg.value(metadata i32 1, metadata !46, metadata !DIExpression()), !dbg !37
  call void @llvm.dbg.value(metadata i32 8, metadata !47, metadata !DIExpression()), !dbg !37
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !48
  store i32 %0, i32* %__gtid_drb050_foo1__404, align 4, !dbg !48
  br label %L.LB2_377

L.LB2_377:                                        ; preds = %L.entry
  %1 = bitcast i64* %len to i8*, !dbg !49
  %2 = bitcast %astruct.dt88* %.uplevelArgPack0001_382 to i8**, !dbg !49
  store i8* %1, i8** %2, align 8, !dbg !49
  %3 = bitcast i64* %"c$p14" to i8*, !dbg !49
  %4 = bitcast %astruct.dt88* %.uplevelArgPack0001_382 to i8*, !dbg !49
  %5 = getelementptr i8, i8* %4, i64 8, !dbg !49
  %6 = bitcast i8* %5 to i8**, !dbg !49
  store i8* %3, i8** %6, align 8, !dbg !49
  %7 = bitcast i64* %"c$sd13" to i8*, !dbg !49
  %8 = bitcast %astruct.dt88* %.uplevelArgPack0001_382 to i8*, !dbg !49
  %9 = getelementptr i8, i8* %8, i64 16, !dbg !49
  %10 = bitcast i8* %9 to i8**, !dbg !49
  store i8* %7, i8** %10, align 8, !dbg !49
  %11 = bitcast i64* %"c$p14" to i8*, !dbg !49
  %12 = bitcast %astruct.dt88* %.uplevelArgPack0001_382 to i8*, !dbg !49
  %13 = getelementptr i8, i8* %12, i64 24, !dbg !49
  %14 = bitcast i8* %13 to i8**, !dbg !49
  store i8* %11, i8** %14, align 8, !dbg !49
  %15 = bitcast i64* %"o1$p10" to i8*, !dbg !49
  %16 = bitcast %astruct.dt88* %.uplevelArgPack0001_382 to i8*, !dbg !49
  %17 = getelementptr i8, i8* %16, i64 32, !dbg !49
  %18 = bitcast i8* %17 to i8**, !dbg !49
  store i8* %15, i8** %18, align 8, !dbg !49
  %19 = bitcast i64* %"o1$sd9" to i8*, !dbg !49
  %20 = bitcast %astruct.dt88* %.uplevelArgPack0001_382 to i8*, !dbg !49
  %21 = getelementptr i8, i8* %20, i64 40, !dbg !49
  %22 = bitcast i8* %21 to i8**, !dbg !49
  store i8* %19, i8** %22, align 8, !dbg !49
  %23 = bitcast i64* %"o1$p10" to i8*, !dbg !49
  %24 = bitcast %astruct.dt88* %.uplevelArgPack0001_382 to i8*, !dbg !49
  %25 = getelementptr i8, i8* %24, i64 48, !dbg !49
  %26 = bitcast i8* %25 to i8**, !dbg !49
  store i8* %23, i8** %26, align 8, !dbg !49
  br label %L.LB2_402, !dbg !49

L.LB2_402:                                        ; preds = %L.LB2_377
  %27 = bitcast void (i32*, i64*, i64*)* @__nv_drb050_foo1__F1L23_1_ to i64*, !dbg !49
  %28 = bitcast %astruct.dt88* %.uplevelArgPack0001_382 to i64*, !dbg !49
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %27, i64* %28), !dbg !49
  ret void, !dbg !48
}

define internal void @__nv_drb050_foo1__F1L23_1_(i32* %__nv_drb050_foo1__F1L23_1Arg0, i64* %__nv_drb050_foo1__F1L23_1Arg1, i64* %__nv_drb050_foo1__F1L23_1Arg2) #1 !dbg !50 {
L.entry:
  %__gtid___nv_drb050_foo1__F1L23_1__443 = alloca i32, align 4
  %.i0000p_336 = alloca i32, align 4
  %i_335 = alloca i32, align 4
  %.du0001p_363 = alloca i32, align 4
  %.de0001p_364 = alloca i32, align 4
  %.di0001p_365 = alloca i32, align 4
  %.ds0001p_366 = alloca i32, align 4
  %.dl0001p_368 = alloca i32, align 4
  %.dl0001p.copy_437 = alloca i32, align 4
  %.de0001p.copy_438 = alloca i32, align 4
  %.ds0001p.copy_439 = alloca i32, align 4
  %.dX0001p_367 = alloca i32, align 4
  %.dY0001p_362 = alloca i32, align 4
  %volnew_o8_334 = alloca double, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_drb050_foo1__F1L23_1Arg0, metadata !53, metadata !DIExpression()), !dbg !54
  call void @llvm.dbg.declare(metadata i64* %__nv_drb050_foo1__F1L23_1Arg1, metadata !55, metadata !DIExpression()), !dbg !54
  call void @llvm.dbg.declare(metadata i64* %__nv_drb050_foo1__F1L23_1Arg2, metadata !56, metadata !DIExpression()), !dbg !54
  call void @llvm.dbg.value(metadata i32 1, metadata !57, metadata !DIExpression()), !dbg !54
  call void @llvm.dbg.value(metadata i32 0, metadata !58, metadata !DIExpression()), !dbg !54
  call void @llvm.dbg.value(metadata i32 1, metadata !59, metadata !DIExpression()), !dbg !54
  call void @llvm.dbg.value(metadata i32 0, metadata !60, metadata !DIExpression()), !dbg !54
  call void @llvm.dbg.value(metadata i32 1, metadata !61, metadata !DIExpression()), !dbg !54
  call void @llvm.dbg.value(metadata i32 8, metadata !62, metadata !DIExpression()), !dbg !54
  %0 = load i32, i32* %__nv_drb050_foo1__F1L23_1Arg0, align 4, !dbg !63
  store i32 %0, i32* %__gtid___nv_drb050_foo1__F1L23_1__443, align 4, !dbg !63
  br label %L.LB3_427

L.LB3_427:                                        ; preds = %L.entry
  br label %L.LB3_333

L.LB3_333:                                        ; preds = %L.LB3_427
  store i32 0, i32* %.i0000p_336, align 4, !dbg !64
  call void @llvm.dbg.declare(metadata i32* %i_335, metadata !65, metadata !DIExpression()), !dbg !63
  store i32 1, i32* %i_335, align 4, !dbg !64
  %1 = bitcast i64* %__nv_drb050_foo1__F1L23_1Arg2 to i32**, !dbg !64
  %2 = load i32*, i32** %1, align 8, !dbg !64
  %3 = load i32, i32* %2, align 4, !dbg !64
  store i32 %3, i32* %.du0001p_363, align 4, !dbg !64
  %4 = bitcast i64* %__nv_drb050_foo1__F1L23_1Arg2 to i32**, !dbg !64
  %5 = load i32*, i32** %4, align 8, !dbg !64
  %6 = load i32, i32* %5, align 4, !dbg !64
  store i32 %6, i32* %.de0001p_364, align 4, !dbg !64
  store i32 1, i32* %.di0001p_365, align 4, !dbg !64
  %7 = load i32, i32* %.di0001p_365, align 4, !dbg !64
  store i32 %7, i32* %.ds0001p_366, align 4, !dbg !64
  store i32 1, i32* %.dl0001p_368, align 4, !dbg !64
  %8 = load i32, i32* %.dl0001p_368, align 4, !dbg !64
  store i32 %8, i32* %.dl0001p.copy_437, align 4, !dbg !64
  %9 = load i32, i32* %.de0001p_364, align 4, !dbg !64
  store i32 %9, i32* %.de0001p.copy_438, align 4, !dbg !64
  %10 = load i32, i32* %.ds0001p_366, align 4, !dbg !64
  store i32 %10, i32* %.ds0001p.copy_439, align 4, !dbg !64
  %11 = load i32, i32* %__gtid___nv_drb050_foo1__F1L23_1__443, align 4, !dbg !64
  %12 = bitcast i32* %.i0000p_336 to i64*, !dbg !64
  %13 = bitcast i32* %.dl0001p.copy_437 to i64*, !dbg !64
  %14 = bitcast i32* %.de0001p.copy_438 to i64*, !dbg !64
  %15 = bitcast i32* %.ds0001p.copy_439 to i64*, !dbg !64
  %16 = load i32, i32* %.ds0001p.copy_439, align 4, !dbg !64
  call void @__kmpc_for_static_init_4(i64* null, i32 %11, i32 34, i64* %12, i64* %13, i64* %14, i64* %15, i32 %16, i32 1), !dbg !64
  %17 = load i32, i32* %.dl0001p.copy_437, align 4, !dbg !64
  store i32 %17, i32* %.dl0001p_368, align 4, !dbg !64
  %18 = load i32, i32* %.de0001p.copy_438, align 4, !dbg !64
  store i32 %18, i32* %.de0001p_364, align 4, !dbg !64
  %19 = load i32, i32* %.ds0001p.copy_439, align 4, !dbg !64
  store i32 %19, i32* %.ds0001p_366, align 4, !dbg !64
  %20 = load i32, i32* %.dl0001p_368, align 4, !dbg !64
  store i32 %20, i32* %i_335, align 4, !dbg !64
  %21 = load i32, i32* %i_335, align 4, !dbg !64
  call void @llvm.dbg.value(metadata i32 %21, metadata !65, metadata !DIExpression()), !dbg !63
  store i32 %21, i32* %.dX0001p_367, align 4, !dbg !64
  %22 = load i32, i32* %.dX0001p_367, align 4, !dbg !64
  %23 = load i32, i32* %.du0001p_363, align 4, !dbg !64
  %24 = icmp sgt i32 %22, %23, !dbg !64
  br i1 %24, label %L.LB3_361, label %L.LB3_475, !dbg !64

L.LB3_475:                                        ; preds = %L.LB3_333
  %25 = load i32, i32* %.dX0001p_367, align 4, !dbg !64
  store i32 %25, i32* %i_335, align 4, !dbg !64
  %26 = load i32, i32* %.di0001p_365, align 4, !dbg !64
  %27 = load i32, i32* %.de0001p_364, align 4, !dbg !64
  %28 = load i32, i32* %.dX0001p_367, align 4, !dbg !64
  %29 = sub nsw i32 %27, %28, !dbg !64
  %30 = add nsw i32 %26, %29, !dbg !64
  %31 = load i32, i32* %.di0001p_365, align 4, !dbg !64
  %32 = sdiv i32 %30, %31, !dbg !64
  store i32 %32, i32* %.dY0001p_362, align 4, !dbg !64
  %33 = load i32, i32* %.dY0001p_362, align 4, !dbg !64
  %34 = icmp sle i32 %33, 0, !dbg !64
  br i1 %34, label %L.LB3_371, label %L.LB3_370, !dbg !64

L.LB3_370:                                        ; preds = %L.LB3_370, %L.LB3_475
  %35 = load i32, i32* %i_335, align 4, !dbg !66
  call void @llvm.dbg.value(metadata i32 %35, metadata !65, metadata !DIExpression()), !dbg !63
  %36 = sext i32 %35 to i64, !dbg !66
  %37 = bitcast i64* %__nv_drb050_foo1__F1L23_1Arg2 to i8*, !dbg !66
  %38 = getelementptr i8, i8* %37, i64 16, !dbg !66
  %39 = bitcast i8* %38 to i8**, !dbg !66
  %40 = load i8*, i8** %39, align 8, !dbg !66
  %41 = getelementptr i8, i8* %40, i64 56, !dbg !66
  %42 = bitcast i8* %41 to i64*, !dbg !66
  %43 = load i64, i64* %42, align 8, !dbg !66
  %44 = add nsw i64 %36, %43, !dbg !66
  %45 = bitcast i64* %__nv_drb050_foo1__F1L23_1Arg2 to i8*, !dbg !66
  %46 = getelementptr i8, i8* %45, i64 24, !dbg !66
  %47 = bitcast i8* %46 to i8***, !dbg !66
  %48 = load i8**, i8*** %47, align 8, !dbg !66
  %49 = load i8*, i8** %48, align 8, !dbg !66
  %50 = getelementptr i8, i8* %49, i64 -8, !dbg !66
  %51 = bitcast i8* %50 to double*, !dbg !66
  %52 = getelementptr double, double* %51, i64 %44, !dbg !66
  %53 = load double, double* %52, align 8, !dbg !66
  %54 = fmul fast double %53, 5.000000e-01, !dbg !66
  call void @llvm.dbg.declare(metadata double* %volnew_o8_334, metadata !67, metadata !DIExpression()), !dbg !63
  store double %54, double* %volnew_o8_334, align 8, !dbg !66
  %55 = load double, double* %volnew_o8_334, align 8, !dbg !68
  call void @llvm.dbg.value(metadata double %55, metadata !67, metadata !DIExpression()), !dbg !63
  %56 = load i32, i32* %i_335, align 4, !dbg !68
  call void @llvm.dbg.value(metadata i32 %56, metadata !65, metadata !DIExpression()), !dbg !63
  %57 = sext i32 %56 to i64, !dbg !68
  %58 = bitcast i64* %__nv_drb050_foo1__F1L23_1Arg2 to i8*, !dbg !68
  %59 = getelementptr i8, i8* %58, i64 40, !dbg !68
  %60 = bitcast i8* %59 to i8**, !dbg !68
  %61 = load i8*, i8** %60, align 8, !dbg !68
  %62 = getelementptr i8, i8* %61, i64 56, !dbg !68
  %63 = bitcast i8* %62 to i64*, !dbg !68
  %64 = load i64, i64* %63, align 8, !dbg !68
  %65 = add nsw i64 %57, %64, !dbg !68
  %66 = bitcast i64* %__nv_drb050_foo1__F1L23_1Arg2 to i8*, !dbg !68
  %67 = getelementptr i8, i8* %66, i64 48, !dbg !68
  %68 = bitcast i8* %67 to i8***, !dbg !68
  %69 = load i8**, i8*** %68, align 8, !dbg !68
  %70 = load i8*, i8** %69, align 8, !dbg !68
  %71 = getelementptr i8, i8* %70, i64 -8, !dbg !68
  %72 = bitcast i8* %71 to double*, !dbg !68
  %73 = getelementptr double, double* %72, i64 %65, !dbg !68
  store double %55, double* %73, align 8, !dbg !68
  %74 = load i32, i32* %.di0001p_365, align 4, !dbg !63
  %75 = load i32, i32* %i_335, align 4, !dbg !63
  call void @llvm.dbg.value(metadata i32 %75, metadata !65, metadata !DIExpression()), !dbg !63
  %76 = add nsw i32 %74, %75, !dbg !63
  store i32 %76, i32* %i_335, align 4, !dbg !63
  %77 = load i32, i32* %.dY0001p_362, align 4, !dbg !63
  %78 = sub nsw i32 %77, 1, !dbg !63
  store i32 %78, i32* %.dY0001p_362, align 4, !dbg !63
  %79 = load i32, i32* %.dY0001p_362, align 4, !dbg !63
  %80 = icmp sgt i32 %79, 0, !dbg !63
  br i1 %80, label %L.LB3_370, label %L.LB3_371, !dbg !63

L.LB3_371:                                        ; preds = %L.LB3_370, %L.LB3_475
  br label %L.LB3_361

L.LB3_361:                                        ; preds = %L.LB3_371, %L.LB3_333
  %81 = load i32, i32* %__gtid___nv_drb050_foo1__F1L23_1__443, align 4, !dbg !63
  call void @__kmpc_for_static_fini(i64* null, i32 %81), !dbg !63
  br label %L.LB3_337

L.LB3_337:                                        ; preds = %L.LB3_361
  ret void, !dbg !63
}

define void @MAIN_() #1 !dbg !27 {
L.entry:
  %.g0000_371 = alloca i64, align 8
  call void @llvm.dbg.value(metadata i32 1, metadata !69, metadata !DIExpression()), !dbg !70
  call void @llvm.dbg.value(metadata i32 0, metadata !71, metadata !DIExpression()), !dbg !70
  call void @llvm.dbg.value(metadata i32 1, metadata !72, metadata !DIExpression()), !dbg !70
  call void @llvm.dbg.value(metadata i32 0, metadata !73, metadata !DIExpression()), !dbg !70
  call void @llvm.dbg.value(metadata i32 1, metadata !74, metadata !DIExpression()), !dbg !70
  %0 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !75
  %1 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !75
  call void (i8*, ...) %1(i8* %0), !dbg !75
  br label %L.LB5_352

L.LB5_352:                                        ; preds = %L.entry
  %2 = bitcast %struct_drb050_0_* @_drb050_0_ to i8*, !dbg !76
  %3 = getelementptr i8, i8* %2, i64 96, !dbg !76
  %4 = bitcast i8* %3 to i64*, !dbg !76
  store i64 1, i64* %4, align 8, !dbg !76
  %5 = bitcast %struct_drb050_0_* @_drb050_0_ to i8*, !dbg !76
  %6 = getelementptr i8, i8* %5, i64 104, !dbg !76
  %7 = bitcast i8* %6 to i64*, !dbg !76
  store i64 100, i64* %7, align 8, !dbg !76
  %8 = bitcast %struct_drb050_0_* @_drb050_0_ to i8*, !dbg !76
  %9 = getelementptr i8, i8* %8, i64 104, !dbg !76
  %10 = bitcast i8* %9 to i64*, !dbg !76
  %11 = load i64, i64* %10, align 8, !dbg !76
  %12 = sub nsw i64 %11, 1, !dbg !76
  %13 = bitcast %struct_drb050_0_* @_drb050_0_ to i8*, !dbg !76
  %14 = getelementptr i8, i8* %13, i64 96, !dbg !76
  %15 = bitcast i8* %14 to i64*, !dbg !76
  %16 = load i64, i64* %15, align 8, !dbg !76
  %17 = add nsw i64 %12, %16, !dbg !76
  store i64 %17, i64* %.g0000_371, align 8, !dbg !76
  %18 = bitcast %struct_drb050_0_* @_drb050_0_ to i8*, !dbg !76
  %19 = getelementptr i8, i8* %18, i64 16, !dbg !76
  %20 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !76
  %21 = bitcast i64* @.C332_MAIN_ to i8*, !dbg !76
  %22 = bitcast i64* @.C310_MAIN_ to i8*, !dbg !76
  %23 = bitcast %struct_drb050_0_* @_drb050_0_ to i8*, !dbg !76
  %24 = getelementptr i8, i8* %23, i64 96, !dbg !76
  %25 = bitcast i64* %.g0000_371 to i8*, !dbg !76
  %26 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !76
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %26(i8* %19, i8* %20, i8* %21, i8* %22, i8* %24, i8* %25), !dbg !76
  %27 = bitcast %struct_drb050_0_* @_drb050_0_ to i8*, !dbg !76
  %28 = getelementptr i8, i8* %27, i64 16, !dbg !76
  %29 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !76
  call void (i8*, i32, ...) %29(i8* %28, i32 28), !dbg !76
  %30 = bitcast %struct_drb050_0_* @_drb050_0_ to i8*, !dbg !76
  %31 = getelementptr i8, i8* %30, i64 104, !dbg !76
  %32 = bitcast i8* %31 to i64*, !dbg !76
  %33 = load i64, i64* %32, align 8, !dbg !76
  %34 = sub nsw i64 %33, 1, !dbg !76
  %35 = bitcast %struct_drb050_0_* @_drb050_0_ to i8*, !dbg !76
  %36 = getelementptr i8, i8* %35, i64 96, !dbg !76
  %37 = bitcast i8* %36 to i64*, !dbg !76
  %38 = load i64, i64* %37, align 8, !dbg !76
  %39 = add nsw i64 %34, %38, !dbg !76
  %40 = bitcast %struct_drb050_0_* @_drb050_0_ to i8*, !dbg !76
  %41 = getelementptr i8, i8* %40, i64 96, !dbg !76
  %42 = bitcast i8* %41 to i64*, !dbg !76
  %43 = load i64, i64* %42, align 8, !dbg !76
  %44 = sub nsw i64 %43, 1, !dbg !76
  %45 = sub nsw i64 %39, %44, !dbg !76
  store i64 %45, i64* %.g0000_371, align 8, !dbg !76
  %46 = bitcast i64* %.g0000_371 to i8*, !dbg !76
  %47 = bitcast i64* @.C332_MAIN_ to i8*, !dbg !76
  %48 = bitcast i64* @.C310_MAIN_ to i8*, !dbg !76
  %49 = bitcast %struct_drb050_0_* @_drb050_0_ to i8*, !dbg !76
  %50 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !76
  %51 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !76
  %52 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !76
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %52(i8* %46, i8* %47, i8* %48, i8* null, i8* %49, i8* null, i8* %50, i8* %51, i8* null, i64 0), !dbg !76
  %53 = bitcast %struct_drb050_0_* @_drb050_0_ to i8*, !dbg !77
  %54 = getelementptr i8, i8* %53, i64 240, !dbg !77
  %55 = bitcast i8* %54 to i64*, !dbg !77
  store i64 1, i64* %55, align 8, !dbg !77
  %56 = bitcast %struct_drb050_0_* @_drb050_0_ to i8*, !dbg !77
  %57 = getelementptr i8, i8* %56, i64 248, !dbg !77
  %58 = bitcast i8* %57 to i64*, !dbg !77
  store i64 100, i64* %58, align 8, !dbg !77
  %59 = bitcast %struct_drb050_0_* @_drb050_0_ to i8*, !dbg !77
  %60 = getelementptr i8, i8* %59, i64 248, !dbg !77
  %61 = bitcast i8* %60 to i64*, !dbg !77
  %62 = load i64, i64* %61, align 8, !dbg !77
  %63 = sub nsw i64 %62, 1, !dbg !77
  %64 = bitcast %struct_drb050_0_* @_drb050_0_ to i8*, !dbg !77
  %65 = getelementptr i8, i8* %64, i64 240, !dbg !77
  %66 = bitcast i8* %65 to i64*, !dbg !77
  %67 = load i64, i64* %66, align 8, !dbg !77
  %68 = add nsw i64 %63, %67, !dbg !77
  store i64 %68, i64* %.g0000_371, align 8, !dbg !77
  %69 = bitcast %struct_drb050_0_* @_drb050_0_ to i8*, !dbg !77
  %70 = getelementptr i8, i8* %69, i64 160, !dbg !77
  %71 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !77
  %72 = bitcast i64* @.C332_MAIN_ to i8*, !dbg !77
  %73 = bitcast i64* @.C310_MAIN_ to i8*, !dbg !77
  %74 = bitcast %struct_drb050_0_* @_drb050_0_ to i8*, !dbg !77
  %75 = getelementptr i8, i8* %74, i64 240, !dbg !77
  %76 = bitcast i64* %.g0000_371 to i8*, !dbg !77
  %77 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !77
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %77(i8* %70, i8* %71, i8* %72, i8* %73, i8* %75, i8* %76), !dbg !77
  %78 = bitcast %struct_drb050_0_* @_drb050_0_ to i8*, !dbg !77
  %79 = getelementptr i8, i8* %78, i64 160, !dbg !77
  %80 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !77
  call void (i8*, i32, ...) %80(i8* %79, i32 28), !dbg !77
  %81 = bitcast %struct_drb050_0_* @_drb050_0_ to i8*, !dbg !77
  %82 = getelementptr i8, i8* %81, i64 248, !dbg !77
  %83 = bitcast i8* %82 to i64*, !dbg !77
  %84 = load i64, i64* %83, align 8, !dbg !77
  %85 = sub nsw i64 %84, 1, !dbg !77
  %86 = bitcast %struct_drb050_0_* @_drb050_0_ to i8*, !dbg !77
  %87 = getelementptr i8, i8* %86, i64 240, !dbg !77
  %88 = bitcast i8* %87 to i64*, !dbg !77
  %89 = load i64, i64* %88, align 8, !dbg !77
  %90 = add nsw i64 %85, %89, !dbg !77
  %91 = bitcast %struct_drb050_0_* @_drb050_0_ to i8*, !dbg !77
  %92 = getelementptr i8, i8* %91, i64 240, !dbg !77
  %93 = bitcast i8* %92 to i64*, !dbg !77
  %94 = load i64, i64* %93, align 8, !dbg !77
  %95 = sub nsw i64 %94, 1, !dbg !77
  %96 = sub nsw i64 %90, %95, !dbg !77
  store i64 %96, i64* %.g0000_371, align 8, !dbg !77
  %97 = bitcast i64* %.g0000_371 to i8*, !dbg !77
  %98 = bitcast i64* @.C332_MAIN_ to i8*, !dbg !77
  %99 = bitcast i64* @.C310_MAIN_ to i8*, !dbg !77
  %100 = bitcast %struct_drb050_0_* @_drb050_0_ to i8*, !dbg !77
  %101 = getelementptr i8, i8* %100, i64 144, !dbg !77
  %102 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !77
  %103 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !77
  %104 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !77
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %104(i8* %97, i8* %98, i8* %99, i8* null, i8* %101, i8* null, i8* %102, i8* %103, i8* null, i64 0), !dbg !77
  %105 = bitcast %struct_drb050_0_* @_drb050_0_ to i64*, !dbg !78
  %106 = bitcast %struct_drb050_0_* @_drb050_0_ to i8*, !dbg !78
  %107 = getelementptr i8, i8* %106, i64 144, !dbg !78
  %108 = bitcast i8* %107 to i64*, !dbg !78
  %109 = bitcast i32* @.C329_MAIN_ to i64*, !dbg !78
  %110 = bitcast %struct_drb050_0_* @_drb050_0_ to i8*, !dbg !78
  %111 = getelementptr i8, i8* %110, i64 16, !dbg !78
  %112 = bitcast i8* %111 to i64*, !dbg !78
  %113 = bitcast %struct_drb050_0_* @_drb050_0_ to i8*, !dbg !78
  %114 = getelementptr i8, i8* %113, i64 160, !dbg !78
  %115 = bitcast i8* %114 to i64*, !dbg !78
  call void @drb050_foo1_(i64* %105, i64* %108, i64* %109, i64* %112, i64* %115), !dbg !78
  ret void, !dbg !79
}

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

!llvm.module.flags = !{!30, !31}
!llvm.dbg.cu = !{!3}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "z_b_0", scope: !2, file: !4, type: !16, isLocal: false, isDefinition: true)
!2 = !DIModule(scope: !3, name: "drb050")
!3 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !4, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !6, imports: !25)
!4 = !DIFile(filename: "micro-benchmarks-fortran/DRB050-functionparameter-orig-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!5 = !{}
!6 = !{!7, !13, !19, !21, !0, !23}
!7 = !DIGlobalVariableExpression(var: !8, expr: !DIExpression(DW_OP_deref))
!8 = distinct !DIGlobalVariable(name: "o1", scope: !2, file: !4, type: !9, isLocal: false, isDefinition: true)
!9 = !DICompositeType(tag: DW_TAG_array_type, baseType: !10, size: 64, align: 64, elements: !11)
!10 = !DIBasicType(name: "double precision", size: 64, align: 64, encoding: DW_ATE_float)
!11 = !{!12}
!12 = !DISubrange(count: 0, lowerBound: 1)
!13 = !DIGlobalVariableExpression(var: !14, expr: !DIExpression(DW_OP_plus_uconst, 16))
!14 = distinct !DIGlobalVariable(name: "o1$sd", scope: !2, file: !4, type: !15, isLocal: false, isDefinition: true)
!15 = !DICompositeType(tag: DW_TAG_array_type, baseType: !16, size: 1024, align: 64, elements: !17)
!16 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!17 = !{!18}
!18 = !DISubrange(count: 16, lowerBound: 1)
!19 = !DIGlobalVariableExpression(var: !20, expr: !DIExpression(DW_OP_deref))
!20 = distinct !DIGlobalVariable(name: "c", scope: !2, file: !4, type: !9, isLocal: false, isDefinition: true)
!21 = !DIGlobalVariableExpression(var: !22, expr: !DIExpression(DW_OP_plus_uconst, 160))
!22 = distinct !DIGlobalVariable(name: "c$sd", scope: !2, file: !4, type: !15, isLocal: false, isDefinition: true)
!23 = !DIGlobalVariableExpression(var: !24, expr: !DIExpression(DW_OP_plus_uconst, 8))
!24 = distinct !DIGlobalVariable(name: "z_b_1", scope: !2, file: !4, type: !16, isLocal: false, isDefinition: true)
!25 = !{!26}
!26 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !27, entity: !2, file: !4, line: 33)
!27 = distinct !DISubprogram(name: "drb050_functionparameter_orig_no", scope: !3, file: !4, line: 33, type: !28, scopeLine: 33, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !3)
!28 = !DISubroutineType(cc: DW_CC_program, types: !29)
!29 = !{null}
!30 = !{i32 2, !"Dwarf Version", i32 4}
!31 = !{i32 2, !"Debug Info Version", i32 3}
!32 = distinct !DISubprogram(name: "foo1", scope: !2, file: !4, line: 17, type: !33, scopeLine: 17, spFlags: DISPFlagDefinition, unit: !3)
!33 = !DISubroutineType(types: !34)
!34 = !{null, !9, !9, !35, !15, !15}
!35 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!36 = !DILocalVariable(arg: 1, scope: !32, file: !4, type: !9, flags: DIFlagArtificial)
!37 = !DILocation(line: 0, scope: !32)
!38 = !DILocalVariable(arg: 2, scope: !32, file: !4, type: !9, flags: DIFlagArtificial)
!39 = !DILocalVariable(name: "len", arg: 3, scope: !32, file: !4, type: !35)
!40 = !DILocalVariable(arg: 4, scope: !32, file: !4, type: !15, flags: DIFlagArtificial)
!41 = !DILocalVariable(arg: 5, scope: !32, file: !4, type: !15, flags: DIFlagArtificial)
!42 = !DILocalVariable(name: "omp_sched_static", scope: !32, file: !4, type: !35)
!43 = !DILocalVariable(name: "omp_proc_bind_false", scope: !32, file: !4, type: !35)
!44 = !DILocalVariable(name: "omp_proc_bind_true", scope: !32, file: !4, type: !35)
!45 = !DILocalVariable(name: "omp_lock_hint_none", scope: !32, file: !4, type: !35)
!46 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !32, file: !4, type: !35)
!47 = !DILocalVariable(name: "dp", scope: !32, file: !4, type: !35)
!48 = !DILocation(line: 30, column: 1, scope: !32)
!49 = !DILocation(line: 23, column: 1, scope: !32)
!50 = distinct !DISubprogram(name: "__nv_drb050_foo1__F1L23_1", scope: !3, file: !4, line: 23, type: !51, scopeLine: 23, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !3)
!51 = !DISubroutineType(types: !52)
!52 = !{null, !35, !16, !16}
!53 = !DILocalVariable(name: "__nv_drb050_foo1__F1L23_1Arg0", arg: 1, scope: !50, file: !4, type: !35)
!54 = !DILocation(line: 0, scope: !50)
!55 = !DILocalVariable(name: "__nv_drb050_foo1__F1L23_1Arg1", arg: 2, scope: !50, file: !4, type: !16)
!56 = !DILocalVariable(name: "__nv_drb050_foo1__F1L23_1Arg2", arg: 3, scope: !50, file: !4, type: !16)
!57 = !DILocalVariable(name: "omp_sched_static", scope: !50, file: !4, type: !35)
!58 = !DILocalVariable(name: "omp_proc_bind_false", scope: !50, file: !4, type: !35)
!59 = !DILocalVariable(name: "omp_proc_bind_true", scope: !50, file: !4, type: !35)
!60 = !DILocalVariable(name: "omp_lock_hint_none", scope: !50, file: !4, type: !35)
!61 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !50, file: !4, type: !35)
!62 = !DILocalVariable(name: "dp", scope: !50, file: !4, type: !35)
!63 = !DILocation(line: 27, column: 1, scope: !50)
!64 = !DILocation(line: 24, column: 1, scope: !50)
!65 = !DILocalVariable(name: "i", scope: !50, file: !4, type: !35)
!66 = !DILocation(line: 25, column: 1, scope: !50)
!67 = !DILocalVariable(name: "volnew_o8", scope: !50, file: !4, type: !10)
!68 = !DILocation(line: 26, column: 1, scope: !50)
!69 = !DILocalVariable(name: "omp_sched_static", scope: !27, file: !4, type: !35)
!70 = !DILocation(line: 0, scope: !27)
!71 = !DILocalVariable(name: "omp_proc_bind_false", scope: !27, file: !4, type: !35)
!72 = !DILocalVariable(name: "omp_proc_bind_true", scope: !27, file: !4, type: !35)
!73 = !DILocalVariable(name: "omp_lock_hint_none", scope: !27, file: !4, type: !35)
!74 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !27, file: !4, type: !35)
!75 = !DILocation(line: 33, column: 1, scope: !27)
!76 = !DILocation(line: 38, column: 1, scope: !27)
!77 = !DILocation(line: 39, column: 1, scope: !27)
!78 = !DILocation(line: 41, column: 1, scope: !27)
!79 = !DILocation(line: 42, column: 1, scope: !27)

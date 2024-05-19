; ModuleID = '/tmp/DRB071-targetparallelfor-orig-no-e2d5e2.ll'
source_filename = "/tmp/DRB071-targetparallelfor-orig-no-e2d5e2.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%astruct.dt64 = type <{ i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8* }>
%astruct.dt97 = type <{ [72 x i8] }>

@.C285_MAIN_ = internal constant i32 1
@.C305_MAIN_ = internal constant i32 25
@.C329_MAIN_ = internal constant i64 4
@.C328_MAIN_ = internal constant i64 25
@.C286_MAIN_ = internal constant i64 1
@.C284_MAIN_ = internal constant i64 0
@.C283_MAIN_ = internal constant i32 0
@.C285___nv_MAIN__F1L23_1 = internal constant i32 1
@.C283___nv_MAIN__F1L23_1 = internal constant i32 0
@.C285___nv_MAIN_F1L24_2 = internal constant i32 1
@.C283___nv_MAIN_F1L24_2 = internal constant i32 0

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__398 = alloca i32, align 4
  %.Z0965_314 = alloca i32*, align 8
  %"a$sd1_327" = alloca [16 x i64], align 8
  %z_b_0_307 = alloca i64, align 8
  %len_313 = alloca i32, align 4
  %z_b_1_308 = alloca i64, align 8
  %z_e_60_311 = alloca i64, align 8
  %z_b_2_309 = alloca i64, align 8
  %z_b_3_310 = alloca i64, align 8
  %.dY0001_338 = alloca i32, align 4
  %i_306 = alloca i32, align 4
  %.uplevelArgPack0001_379 = alloca %astruct.dt64, align 16
  call void @llvm.dbg.value(metadata i32 1, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !14, metadata !DIExpression()), !dbg !10
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !15
  store i32 %0, i32* %__gtid_MAIN__398, align 4, !dbg !15
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !16
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !16
  call void (i8*, ...) %2(i8* %1), !dbg !16
  call void @llvm.dbg.declare(metadata i32** %.Z0965_314, metadata !17, metadata !DIExpression(DW_OP_deref)), !dbg !10
  %3 = bitcast i32** %.Z0965_314 to i8**, !dbg !16
  store i8* null, i8** %3, align 8, !dbg !16
  call void @llvm.dbg.declare(metadata [16 x i64]* %"a$sd1_327", metadata !21, metadata !DIExpression()), !dbg !10
  %4 = bitcast [16 x i64]* %"a$sd1_327" to i64*, !dbg !16
  store i64 0, i64* %4, align 8, !dbg !16
  br label %L.LB1_358

L.LB1_358:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i64* %z_b_0_307, metadata !26, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_0_307, align 8, !dbg !27
  call void @llvm.dbg.declare(metadata i32* %len_313, metadata !28, metadata !DIExpression()), !dbg !10
  %5 = load i32, i32* %len_313, align 4, !dbg !27
  call void @llvm.dbg.value(metadata i32 %5, metadata !28, metadata !DIExpression()), !dbg !10
  %6 = sext i32 %5 to i64, !dbg !27
  call void @llvm.dbg.declare(metadata i64* %z_b_1_308, metadata !26, metadata !DIExpression()), !dbg !10
  store i64 %6, i64* %z_b_1_308, align 8, !dbg !27
  %7 = load i64, i64* %z_b_1_308, align 8, !dbg !27
  call void @llvm.dbg.value(metadata i64 %7, metadata !26, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_60_311, metadata !26, metadata !DIExpression()), !dbg !10
  store i64 %7, i64* %z_e_60_311, align 8, !dbg !27
  %8 = bitcast [16 x i64]* %"a$sd1_327" to i8*, !dbg !27
  %9 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !27
  %10 = bitcast i64* @.C328_MAIN_ to i8*, !dbg !27
  %11 = bitcast i64* @.C329_MAIN_ to i8*, !dbg !27
  %12 = bitcast i64* %z_b_0_307 to i8*, !dbg !27
  %13 = bitcast i64* %z_b_1_308 to i8*, !dbg !27
  %14 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !27
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %14(i8* %8, i8* %9, i8* %10, i8* %11, i8* %12, i8* %13), !dbg !27
  %15 = bitcast [16 x i64]* %"a$sd1_327" to i8*, !dbg !27
  %16 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !27
  call void (i8*, i32, ...) %16(i8* %15, i32 25), !dbg !27
  %17 = load i64, i64* %z_b_1_308, align 8, !dbg !27
  call void @llvm.dbg.value(metadata i64 %17, metadata !26, metadata !DIExpression()), !dbg !10
  %18 = load i64, i64* %z_b_0_307, align 8, !dbg !27
  call void @llvm.dbg.value(metadata i64 %18, metadata !26, metadata !DIExpression()), !dbg !10
  %19 = sub nsw i64 %18, 1, !dbg !27
  %20 = sub nsw i64 %17, %19, !dbg !27
  call void @llvm.dbg.declare(metadata i64* %z_b_2_309, metadata !26, metadata !DIExpression()), !dbg !10
  store i64 %20, i64* %z_b_2_309, align 8, !dbg !27
  %21 = load i64, i64* %z_b_0_307, align 8, !dbg !27
  call void @llvm.dbg.value(metadata i64 %21, metadata !26, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_b_3_310, metadata !26, metadata !DIExpression()), !dbg !10
  store i64 %21, i64* %z_b_3_310, align 8, !dbg !27
  %22 = bitcast i64* %z_b_2_309 to i8*, !dbg !27
  %23 = bitcast i64* @.C328_MAIN_ to i8*, !dbg !27
  %24 = bitcast i64* @.C329_MAIN_ to i8*, !dbg !27
  %25 = bitcast i32** %.Z0965_314 to i8*, !dbg !27
  %26 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !27
  %27 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !27
  %28 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !27
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %28(i8* %22, i8* %23, i8* %24, i8* null, i8* %25, i8* null, i8* %26, i8* %27, i8* null, i64 0), !dbg !27
  %29 = load i32, i32* %len_313, align 4, !dbg !29
  call void @llvm.dbg.value(metadata i32 %29, metadata !28, metadata !DIExpression()), !dbg !10
  store i32 %29, i32* %.dY0001_338, align 4, !dbg !29
  call void @llvm.dbg.declare(metadata i32* %i_306, metadata !30, metadata !DIExpression()), !dbg !10
  store i32 1, i32* %i_306, align 4, !dbg !29
  %30 = load i32, i32* %.dY0001_338, align 4, !dbg !29
  %31 = icmp sle i32 %30, 0, !dbg !29
  br i1 %31, label %L.LB1_337, label %L.LB1_336, !dbg !29

L.LB1_336:                                        ; preds = %L.LB1_336, %L.LB1_358
  %32 = load i32, i32* %i_306, align 4, !dbg !31
  call void @llvm.dbg.value(metadata i32 %32, metadata !30, metadata !DIExpression()), !dbg !10
  %33 = load i32, i32* %i_306, align 4, !dbg !31
  call void @llvm.dbg.value(metadata i32 %33, metadata !30, metadata !DIExpression()), !dbg !10
  %34 = sext i32 %33 to i64, !dbg !31
  %35 = bitcast [16 x i64]* %"a$sd1_327" to i8*, !dbg !31
  %36 = getelementptr i8, i8* %35, i64 56, !dbg !31
  %37 = bitcast i8* %36 to i64*, !dbg !31
  %38 = load i64, i64* %37, align 8, !dbg !31
  %39 = add nsw i64 %34, %38, !dbg !31
  %40 = load i32*, i32** %.Z0965_314, align 8, !dbg !31
  call void @llvm.dbg.value(metadata i32* %40, metadata !17, metadata !DIExpression()), !dbg !10
  %41 = bitcast i32* %40 to i8*, !dbg !31
  %42 = getelementptr i8, i8* %41, i64 -4, !dbg !31
  %43 = bitcast i8* %42 to i32*, !dbg !31
  %44 = getelementptr i32, i32* %43, i64 %39, !dbg !31
  store i32 %32, i32* %44, align 4, !dbg !31
  %45 = load i32, i32* %i_306, align 4, !dbg !32
  call void @llvm.dbg.value(metadata i32 %45, metadata !30, metadata !DIExpression()), !dbg !10
  %46 = add nsw i32 %45, 1, !dbg !32
  store i32 %46, i32* %i_306, align 4, !dbg !32
  %47 = load i32, i32* %.dY0001_338, align 4, !dbg !32
  %48 = sub nsw i32 %47, 1, !dbg !32
  store i32 %48, i32* %.dY0001_338, align 4, !dbg !32
  %49 = load i32, i32* %.dY0001_338, align 4, !dbg !32
  %50 = icmp sgt i32 %49, 0, !dbg !32
  br i1 %50, label %L.LB1_336, label %L.LB1_337, !dbg !32

L.LB1_337:                                        ; preds = %L.LB1_336, %L.LB1_358
  %51 = bitcast i32* %len_313 to i8*, !dbg !33
  %52 = bitcast %astruct.dt64* %.uplevelArgPack0001_379 to i8**, !dbg !33
  store i8* %51, i8** %52, align 8, !dbg !33
  %53 = bitcast i32** %.Z0965_314 to i8*, !dbg !33
  %54 = bitcast %astruct.dt64* %.uplevelArgPack0001_379 to i8*, !dbg !33
  %55 = getelementptr i8, i8* %54, i64 8, !dbg !33
  %56 = bitcast i8* %55 to i8**, !dbg !33
  store i8* %53, i8** %56, align 8, !dbg !33
  %57 = bitcast i32** %.Z0965_314 to i8*, !dbg !33
  %58 = bitcast %astruct.dt64* %.uplevelArgPack0001_379 to i8*, !dbg !33
  %59 = getelementptr i8, i8* %58, i64 16, !dbg !33
  %60 = bitcast i8* %59 to i8**, !dbg !33
  store i8* %57, i8** %60, align 8, !dbg !33
  %61 = bitcast i64* %z_b_0_307 to i8*, !dbg !33
  %62 = bitcast %astruct.dt64* %.uplevelArgPack0001_379 to i8*, !dbg !33
  %63 = getelementptr i8, i8* %62, i64 24, !dbg !33
  %64 = bitcast i8* %63 to i8**, !dbg !33
  store i8* %61, i8** %64, align 8, !dbg !33
  %65 = bitcast i64* %z_b_1_308 to i8*, !dbg !33
  %66 = bitcast %astruct.dt64* %.uplevelArgPack0001_379 to i8*, !dbg !33
  %67 = getelementptr i8, i8* %66, i64 32, !dbg !33
  %68 = bitcast i8* %67 to i8**, !dbg !33
  store i8* %65, i8** %68, align 8, !dbg !33
  %69 = bitcast i64* %z_e_60_311 to i8*, !dbg !33
  %70 = bitcast %astruct.dt64* %.uplevelArgPack0001_379 to i8*, !dbg !33
  %71 = getelementptr i8, i8* %70, i64 40, !dbg !33
  %72 = bitcast i8* %71 to i8**, !dbg !33
  store i8* %69, i8** %72, align 8, !dbg !33
  %73 = bitcast i64* %z_b_2_309 to i8*, !dbg !33
  %74 = bitcast %astruct.dt64* %.uplevelArgPack0001_379 to i8*, !dbg !33
  %75 = getelementptr i8, i8* %74, i64 48, !dbg !33
  %76 = bitcast i8* %75 to i8**, !dbg !33
  store i8* %73, i8** %76, align 8, !dbg !33
  %77 = bitcast i64* %z_b_3_310 to i8*, !dbg !33
  %78 = bitcast %astruct.dt64* %.uplevelArgPack0001_379 to i8*, !dbg !33
  %79 = getelementptr i8, i8* %78, i64 56, !dbg !33
  %80 = bitcast i8* %79 to i8**, !dbg !33
  store i8* %77, i8** %80, align 8, !dbg !33
  %81 = bitcast [16 x i64]* %"a$sd1_327" to i8*, !dbg !33
  %82 = bitcast %astruct.dt64* %.uplevelArgPack0001_379 to i8*, !dbg !33
  %83 = getelementptr i8, i8* %82, i64 64, !dbg !33
  %84 = bitcast i8* %83 to i8**, !dbg !33
  store i8* %81, i8** %84, align 8, !dbg !33
  %85 = bitcast %astruct.dt64* %.uplevelArgPack0001_379 to i64*, !dbg !33
  call void @__nv_MAIN__F1L23_1_(i32* %__gtid_MAIN__398, i64* null, i64* %85), !dbg !33
  %86 = load i32*, i32** %.Z0965_314, align 8, !dbg !34
  call void @llvm.dbg.value(metadata i32* %86, metadata !17, metadata !DIExpression()), !dbg !10
  %87 = bitcast i32* %86 to i8*, !dbg !34
  %88 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !34
  %89 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !34
  call void (i8*, i8*, i8*, i8*, i64, ...) %89(i8* null, i8* %87, i8* %88, i8* null, i64 0), !dbg !34
  %90 = bitcast i32** %.Z0965_314 to i8**, !dbg !34
  store i8* null, i8** %90, align 8, !dbg !34
  %91 = bitcast [16 x i64]* %"a$sd1_327" to i64*, !dbg !34
  store i64 0, i64* %91, align 8, !dbg !34
  ret void, !dbg !15
}

define internal void @__nv_MAIN__F1L23_1_(i32* %__nv_MAIN__F1L23_1Arg0, i64* %__nv_MAIN__F1L23_1Arg1, i64* %__nv_MAIN__F1L23_1Arg2) #0 !dbg !35 {
L.entry:
  %__gtid___nv_MAIN__F1L23_1__417 = alloca i32, align 4
  %.uplevelArgPack0002_412 = alloca %astruct.dt97, align 16
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L23_1Arg0, metadata !38, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L23_1Arg1, metadata !40, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L23_1Arg2, metadata !41, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.value(metadata i32 1, metadata !42, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.value(metadata i32 0, metadata !43, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.value(metadata i32 1, metadata !44, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.value(metadata i32 0, metadata !45, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.value(metadata i32 1, metadata !46, metadata !DIExpression()), !dbg !39
  %0 = load i32, i32* %__nv_MAIN__F1L23_1Arg0, align 4, !dbg !47
  store i32 %0, i32* %__gtid___nv_MAIN__F1L23_1__417, align 4, !dbg !47
  br label %L.LB2_407

L.LB2_407:                                        ; preds = %L.entry
  br label %L.LB2_317

L.LB2_317:                                        ; preds = %L.LB2_407
  %1 = bitcast i64* %__nv_MAIN__F1L23_1Arg2 to i8*, !dbg !48
  %2 = bitcast %astruct.dt97* %.uplevelArgPack0002_412 to i8*, !dbg !48
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %2, i8* align 8 %1, i64 72, i1 false), !dbg !48
  br label %L.LB2_415, !dbg !48

L.LB2_415:                                        ; preds = %L.LB2_317
  %3 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN_F1L24_2_ to i64*, !dbg !48
  %4 = bitcast %astruct.dt97* %.uplevelArgPack0002_412 to i64*, !dbg !48
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %3, i64* %4), !dbg !48
  br label %L.LB2_324

L.LB2_324:                                        ; preds = %L.LB2_415
  ret void, !dbg !47
}

define internal void @__nv_MAIN_F1L24_2_(i32* %__nv_MAIN_F1L24_2Arg0, i64* %__nv_MAIN_F1L24_2Arg1, i64* %__nv_MAIN_F1L24_2Arg2) #0 !dbg !49 {
L.entry:
  %__gtid___nv_MAIN_F1L24_2__455 = alloca i32, align 4
  %.i0000p_322 = alloca i32, align 4
  %i_321 = alloca i32, align 4
  %.du0002p_342 = alloca i32, align 4
  %.de0002p_343 = alloca i32, align 4
  %.di0002p_344 = alloca i32, align 4
  %.ds0002p_345 = alloca i32, align 4
  %.dl0002p_347 = alloca i32, align 4
  %.dl0002p.copy_449 = alloca i32, align 4
  %.de0002p.copy_450 = alloca i32, align 4
  %.ds0002p.copy_451 = alloca i32, align 4
  %.dX0002p_346 = alloca i32, align 4
  %.dY0002p_341 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L24_2Arg0, metadata !50, metadata !DIExpression()), !dbg !51
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L24_2Arg1, metadata !52, metadata !DIExpression()), !dbg !51
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L24_2Arg2, metadata !53, metadata !DIExpression()), !dbg !51
  call void @llvm.dbg.value(metadata i32 1, metadata !54, metadata !DIExpression()), !dbg !51
  call void @llvm.dbg.value(metadata i32 0, metadata !55, metadata !DIExpression()), !dbg !51
  call void @llvm.dbg.value(metadata i32 1, metadata !56, metadata !DIExpression()), !dbg !51
  call void @llvm.dbg.value(metadata i32 0, metadata !57, metadata !DIExpression()), !dbg !51
  call void @llvm.dbg.value(metadata i32 1, metadata !58, metadata !DIExpression()), !dbg !51
  %0 = load i32, i32* %__nv_MAIN_F1L24_2Arg0, align 4, !dbg !59
  store i32 %0, i32* %__gtid___nv_MAIN_F1L24_2__455, align 4, !dbg !59
  br label %L.LB4_440

L.LB4_440:                                        ; preds = %L.entry
  br label %L.LB4_320

L.LB4_320:                                        ; preds = %L.LB4_440
  store i32 0, i32* %.i0000p_322, align 4, !dbg !60
  call void @llvm.dbg.declare(metadata i32* %i_321, metadata !61, metadata !DIExpression()), !dbg !59
  store i32 1, i32* %i_321, align 4, !dbg !60
  %1 = bitcast i64* %__nv_MAIN_F1L24_2Arg2 to i32**, !dbg !60
  %2 = load i32*, i32** %1, align 8, !dbg !60
  %3 = load i32, i32* %2, align 4, !dbg !60
  store i32 %3, i32* %.du0002p_342, align 4, !dbg !60
  %4 = bitcast i64* %__nv_MAIN_F1L24_2Arg2 to i32**, !dbg !60
  %5 = load i32*, i32** %4, align 8, !dbg !60
  %6 = load i32, i32* %5, align 4, !dbg !60
  store i32 %6, i32* %.de0002p_343, align 4, !dbg !60
  store i32 1, i32* %.di0002p_344, align 4, !dbg !60
  %7 = load i32, i32* %.di0002p_344, align 4, !dbg !60
  store i32 %7, i32* %.ds0002p_345, align 4, !dbg !60
  store i32 1, i32* %.dl0002p_347, align 4, !dbg !60
  %8 = load i32, i32* %.dl0002p_347, align 4, !dbg !60
  store i32 %8, i32* %.dl0002p.copy_449, align 4, !dbg !60
  %9 = load i32, i32* %.de0002p_343, align 4, !dbg !60
  store i32 %9, i32* %.de0002p.copy_450, align 4, !dbg !60
  %10 = load i32, i32* %.ds0002p_345, align 4, !dbg !60
  store i32 %10, i32* %.ds0002p.copy_451, align 4, !dbg !60
  %11 = load i32, i32* %__gtid___nv_MAIN_F1L24_2__455, align 4, !dbg !60
  %12 = bitcast i32* %.i0000p_322 to i64*, !dbg !60
  %13 = bitcast i32* %.dl0002p.copy_449 to i64*, !dbg !60
  %14 = bitcast i32* %.de0002p.copy_450 to i64*, !dbg !60
  %15 = bitcast i32* %.ds0002p.copy_451 to i64*, !dbg !60
  %16 = load i32, i32* %.ds0002p.copy_451, align 4, !dbg !60
  call void @__kmpc_for_static_init_4(i64* null, i32 %11, i32 34, i64* %12, i64* %13, i64* %14, i64* %15, i32 %16, i32 1), !dbg !60
  %17 = load i32, i32* %.dl0002p.copy_449, align 4, !dbg !60
  store i32 %17, i32* %.dl0002p_347, align 4, !dbg !60
  %18 = load i32, i32* %.de0002p.copy_450, align 4, !dbg !60
  store i32 %18, i32* %.de0002p_343, align 4, !dbg !60
  %19 = load i32, i32* %.ds0002p.copy_451, align 4, !dbg !60
  store i32 %19, i32* %.ds0002p_345, align 4, !dbg !60
  %20 = load i32, i32* %.dl0002p_347, align 4, !dbg !60
  store i32 %20, i32* %i_321, align 4, !dbg !60
  %21 = load i32, i32* %i_321, align 4, !dbg !60
  call void @llvm.dbg.value(metadata i32 %21, metadata !61, metadata !DIExpression()), !dbg !59
  store i32 %21, i32* %.dX0002p_346, align 4, !dbg !60
  %22 = load i32, i32* %.dX0002p_346, align 4, !dbg !60
  %23 = load i32, i32* %.du0002p_342, align 4, !dbg !60
  %24 = icmp sgt i32 %22, %23, !dbg !60
  br i1 %24, label %L.LB4_340, label %L.LB4_479, !dbg !60

L.LB4_479:                                        ; preds = %L.LB4_320
  %25 = load i32, i32* %.dX0002p_346, align 4, !dbg !60
  store i32 %25, i32* %i_321, align 4, !dbg !60
  %26 = load i32, i32* %.di0002p_344, align 4, !dbg !60
  %27 = load i32, i32* %.de0002p_343, align 4, !dbg !60
  %28 = load i32, i32* %.dX0002p_346, align 4, !dbg !60
  %29 = sub nsw i32 %27, %28, !dbg !60
  %30 = add nsw i32 %26, %29, !dbg !60
  %31 = load i32, i32* %.di0002p_344, align 4, !dbg !60
  %32 = sdiv i32 %30, %31, !dbg !60
  store i32 %32, i32* %.dY0002p_341, align 4, !dbg !60
  %33 = load i32, i32* %.dY0002p_341, align 4, !dbg !60
  %34 = icmp sle i32 %33, 0, !dbg !60
  br i1 %34, label %L.LB4_350, label %L.LB4_349, !dbg !60

L.LB4_349:                                        ; preds = %L.LB4_349, %L.LB4_479
  %35 = load i32, i32* %i_321, align 4, !dbg !62
  call void @llvm.dbg.value(metadata i32 %35, metadata !61, metadata !DIExpression()), !dbg !59
  %36 = sext i32 %35 to i64, !dbg !62
  %37 = bitcast i64* %__nv_MAIN_F1L24_2Arg2 to i8*, !dbg !62
  %38 = getelementptr i8, i8* %37, i64 64, !dbg !62
  %39 = bitcast i8* %38 to i8**, !dbg !62
  %40 = load i8*, i8** %39, align 8, !dbg !62
  %41 = getelementptr i8, i8* %40, i64 56, !dbg !62
  %42 = bitcast i8* %41 to i64*, !dbg !62
  %43 = load i64, i64* %42, align 8, !dbg !62
  %44 = add nsw i64 %36, %43, !dbg !62
  %45 = bitcast i64* %__nv_MAIN_F1L24_2Arg2 to i8*, !dbg !62
  %46 = getelementptr i8, i8* %45, i64 16, !dbg !62
  %47 = bitcast i8* %46 to i8***, !dbg !62
  %48 = load i8**, i8*** %47, align 8, !dbg !62
  %49 = load i8*, i8** %48, align 8, !dbg !62
  %50 = getelementptr i8, i8* %49, i64 -4, !dbg !62
  %51 = bitcast i8* %50 to i32*, !dbg !62
  %52 = getelementptr i32, i32* %51, i64 %44, !dbg !62
  %53 = load i32, i32* %52, align 4, !dbg !62
  %54 = add nsw i32 %53, 1, !dbg !62
  %55 = load i32, i32* %i_321, align 4, !dbg !62
  call void @llvm.dbg.value(metadata i32 %55, metadata !61, metadata !DIExpression()), !dbg !59
  %56 = sext i32 %55 to i64, !dbg !62
  %57 = bitcast i64* %__nv_MAIN_F1L24_2Arg2 to i8*, !dbg !62
  %58 = getelementptr i8, i8* %57, i64 64, !dbg !62
  %59 = bitcast i8* %58 to i8**, !dbg !62
  %60 = load i8*, i8** %59, align 8, !dbg !62
  %61 = getelementptr i8, i8* %60, i64 56, !dbg !62
  %62 = bitcast i8* %61 to i64*, !dbg !62
  %63 = load i64, i64* %62, align 8, !dbg !62
  %64 = add nsw i64 %56, %63, !dbg !62
  %65 = bitcast i64* %__nv_MAIN_F1L24_2Arg2 to i8*, !dbg !62
  %66 = getelementptr i8, i8* %65, i64 16, !dbg !62
  %67 = bitcast i8* %66 to i8***, !dbg !62
  %68 = load i8**, i8*** %67, align 8, !dbg !62
  %69 = load i8*, i8** %68, align 8, !dbg !62
  %70 = getelementptr i8, i8* %69, i64 -4, !dbg !62
  %71 = bitcast i8* %70 to i32*, !dbg !62
  %72 = getelementptr i32, i32* %71, i64 %64, !dbg !62
  store i32 %54, i32* %72, align 4, !dbg !62
  %73 = load i32, i32* %.di0002p_344, align 4, !dbg !59
  %74 = load i32, i32* %i_321, align 4, !dbg !59
  call void @llvm.dbg.value(metadata i32 %74, metadata !61, metadata !DIExpression()), !dbg !59
  %75 = add nsw i32 %73, %74, !dbg !59
  store i32 %75, i32* %i_321, align 4, !dbg !59
  %76 = load i32, i32* %.dY0002p_341, align 4, !dbg !59
  %77 = sub nsw i32 %76, 1, !dbg !59
  store i32 %77, i32* %.dY0002p_341, align 4, !dbg !59
  %78 = load i32, i32* %.dY0002p_341, align 4, !dbg !59
  %79 = icmp sgt i32 %78, 0, !dbg !59
  br i1 %79, label %L.LB4_349, label %L.LB4_350, !dbg !59

L.LB4_350:                                        ; preds = %L.LB4_349, %L.LB4_479
  br label %L.LB4_340

L.LB4_340:                                        ; preds = %L.LB4_350, %L.LB4_320
  %80 = load i32, i32* %__gtid___nv_MAIN_F1L24_2__455, align 4, !dbg !59
  call void @__kmpc_for_static_fini(i64* null, i32 %80), !dbg !59
  br label %L.LB4_323

L.LB4_323:                                        ; preds = %L.LB4_340
  ret void, !dbg !59
}

declare void @__kmpc_for_static_fini(i64*, i32) #0

declare void @__kmpc_for_static_init_4(i64*, i32, i32, i64*, i64*, i64*, i64*, i32, i32) #0

declare void @__kmpc_fork_call(i64*, i32, i64*, i64*, ...) #0

declare void @__kmpc_end_serialized_parallel(i64*, i32) #0

declare void @__kmpc_serialized_parallel(i64*, i32) #0

declare void @f90_dealloc03a_i8(...) #0

declare void @f90_alloc04_chka_i8(...) #0

declare void @f90_set_intrin_type_i8(...) #0

declare void @f90_template1_i8(...) #0

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare void @fort_init(...) #0

declare signext i32 @__kmpc_global_thread_num(i64*) #0

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1 immarg) #2

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { "no-frame-pointer-elim-non-leaf" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { argmemonly nounwind }

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !4, globals: !4, imports: !4)
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB071-targetparallelfor-orig-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb071_targetparallelfor_orig_no", scope: !2, file: !3, line: 10, type: !6, scopeLine: 10, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!6 = !DISubroutineType(cc: DW_CC_program, types: !7)
!7 = !{null}
!8 = !DILocalVariable(name: "omp_sched_static", scope: !5, file: !3, type: !9)
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 0, scope: !5)
!11 = !DILocalVariable(name: "omp_proc_bind_false", scope: !5, file: !3, type: !9)
!12 = !DILocalVariable(name: "omp_proc_bind_true", scope: !5, file: !3, type: !9)
!13 = !DILocalVariable(name: "omp_lock_hint_none", scope: !5, file: !3, type: !9)
!14 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !5, file: !3, type: !9)
!15 = !DILocation(line: 31, column: 1, scope: !5)
!16 = !DILocation(line: 10, column: 1, scope: !5)
!17 = !DILocalVariable(name: "a", scope: !5, file: !3, type: !18)
!18 = !DICompositeType(tag: DW_TAG_array_type, baseType: !9, size: 32, align: 32, elements: !19)
!19 = !{!20}
!20 = !DISubrange(count: 0, lowerBound: 1)
!21 = !DILocalVariable(scope: !5, file: !3, type: !22, flags: DIFlagArtificial)
!22 = !DICompositeType(tag: DW_TAG_array_type, baseType: !23, size: 1024, align: 64, elements: !24)
!23 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!24 = !{!25}
!25 = !DISubrange(count: 16, lowerBound: 1)
!26 = !DILocalVariable(scope: !5, file: !3, type: !23, flags: DIFlagArtificial)
!27 = !DILocation(line: 17, column: 1, scope: !5)
!28 = !DILocalVariable(name: "len", scope: !5, file: !3, type: !9)
!29 = !DILocation(line: 19, column: 1, scope: !5)
!30 = !DILocalVariable(name: "i", scope: !5, file: !3, type: !9)
!31 = !DILocation(line: 20, column: 1, scope: !5)
!32 = !DILocation(line: 21, column: 1, scope: !5)
!33 = !DILocation(line: 28, column: 1, scope: !5)
!34 = !DILocation(line: 30, column: 1, scope: !5)
!35 = distinct !DISubprogram(name: "__nv_MAIN__F1L23_1", scope: !2, file: !3, line: 23, type: !36, scopeLine: 23, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!36 = !DISubroutineType(types: !37)
!37 = !{null, !9, !23, !23}
!38 = !DILocalVariable(name: "__nv_MAIN__F1L23_1Arg0", arg: 1, scope: !35, file: !3, type: !9)
!39 = !DILocation(line: 0, scope: !35)
!40 = !DILocalVariable(name: "__nv_MAIN__F1L23_1Arg1", arg: 2, scope: !35, file: !3, type: !23)
!41 = !DILocalVariable(name: "__nv_MAIN__F1L23_1Arg2", arg: 3, scope: !35, file: !3, type: !23)
!42 = !DILocalVariable(name: "omp_sched_static", scope: !35, file: !3, type: !9)
!43 = !DILocalVariable(name: "omp_proc_bind_false", scope: !35, file: !3, type: !9)
!44 = !DILocalVariable(name: "omp_proc_bind_true", scope: !35, file: !3, type: !9)
!45 = !DILocalVariable(name: "omp_lock_hint_none", scope: !35, file: !3, type: !9)
!46 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !35, file: !3, type: !9)
!47 = !DILocation(line: 28, column: 1, scope: !35)
!48 = !DILocation(line: 24, column: 1, scope: !35)
!49 = distinct !DISubprogram(name: "__nv_MAIN_F1L24_2", scope: !2, file: !3, line: 24, type: !36, scopeLine: 24, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!50 = !DILocalVariable(name: "__nv_MAIN_F1L24_2Arg0", arg: 1, scope: !49, file: !3, type: !9)
!51 = !DILocation(line: 0, scope: !49)
!52 = !DILocalVariable(name: "__nv_MAIN_F1L24_2Arg1", arg: 2, scope: !49, file: !3, type: !23)
!53 = !DILocalVariable(name: "__nv_MAIN_F1L24_2Arg2", arg: 3, scope: !49, file: !3, type: !23)
!54 = !DILocalVariable(name: "omp_sched_static", scope: !49, file: !3, type: !9)
!55 = !DILocalVariable(name: "omp_proc_bind_false", scope: !49, file: !3, type: !9)
!56 = !DILocalVariable(name: "omp_proc_bind_true", scope: !49, file: !3, type: !9)
!57 = !DILocalVariable(name: "omp_lock_hint_none", scope: !49, file: !3, type: !9)
!58 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !49, file: !3, type: !9)
!59 = !DILocation(line: 27, column: 1, scope: !49)
!60 = !DILocation(line: 25, column: 1, scope: !49)
!61 = !DILocalVariable(name: "i", scope: !49, file: !3, type: !9)
!62 = !DILocation(line: 26, column: 1, scope: !49)

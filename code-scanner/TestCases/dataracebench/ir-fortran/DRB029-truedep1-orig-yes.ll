; ModuleID = '/tmp/DRB029-truedep1-orig-yes-649cbb.ll'
source_filename = "/tmp/DRB029-truedep1-orig-yes-649cbb.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.STATICS1 = type <{ [44 x i8] }>
%astruct.dt68 = type <{ i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8* }>

@.STATICS1 = internal global %struct.STATICS1 <{ [44 x i8] c"\FB\FF\FF\FF\06\00\00\00a(50)=\00\00\EB\FF\FF\FF\00\00\00\00\03\00\00\00\00\00\00\00\01\00\00\00\FF\FF\FF\FF\00\00\00\00" }>, align 16
@.C330_MAIN_ = internal constant i64 50
@.C327_MAIN_ = internal constant i32 6
@.C323_MAIN_ = internal constant [53 x i8] c"micro-benchmarks-fortran/DRB029-truedep1-orig-yes.f95"
@.C325_MAIN_ = internal constant i32 31
@.C285_MAIN_ = internal constant i32 1
@.C305_MAIN_ = internal constant i32 25
@.C340_MAIN_ = internal constant i64 4
@.C339_MAIN_ = internal constant i64 25
@.C313_MAIN_ = internal constant i32 100
@.C286_MAIN_ = internal constant i64 1
@.C284_MAIN_ = internal constant i64 0
@.C283_MAIN_ = internal constant i32 0
@.C285___nv_MAIN__F1L25_1 = internal constant i32 1
@.C283___nv_MAIN__F1L25_1 = internal constant i32 0

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__411 = alloca i32, align 4
  %.Z0965_315 = alloca i32*, align 8
  %"a$sd1_338" = alloca [16 x i64], align 8
  %len_314 = alloca i32, align 4
  %z_b_0_307 = alloca i64, align 8
  %z_b_1_308 = alloca i64, align 8
  %z_e_60_311 = alloca i64, align 8
  %z_b_2_309 = alloca i64, align 8
  %z_b_3_310 = alloca i64, align 8
  %.dY0001_349 = alloca i32, align 4
  %i_306 = alloca i32, align 4
  %.uplevelArgPack0001_390 = alloca %astruct.dt68, align 16
  %z__io_329 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !14, metadata !DIExpression()), !dbg !10
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !15
  store i32 %0, i32* %__gtid_MAIN__411, align 4, !dbg !15
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !16
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !16
  call void (i8*, ...) %2(i8* %1), !dbg !16
  call void @llvm.dbg.declare(metadata i32** %.Z0965_315, metadata !17, metadata !DIExpression(DW_OP_deref)), !dbg !10
  %3 = bitcast i32** %.Z0965_315 to i8**, !dbg !16
  store i8* null, i8** %3, align 8, !dbg !16
  call void @llvm.dbg.declare(metadata [16 x i64]* %"a$sd1_338", metadata !21, metadata !DIExpression()), !dbg !10
  %4 = bitcast [16 x i64]* %"a$sd1_338" to i64*, !dbg !16
  store i64 0, i64* %4, align 8, !dbg !16
  br label %L.LB1_369

L.LB1_369:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %len_314, metadata !26, metadata !DIExpression()), !dbg !10
  store i32 100, i32* %len_314, align 4, !dbg !27
  call void @llvm.dbg.declare(metadata i64* %z_b_0_307, metadata !28, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_0_307, align 8, !dbg !29
  %5 = load i32, i32* %len_314, align 4, !dbg !29
  call void @llvm.dbg.value(metadata i32 %5, metadata !26, metadata !DIExpression()), !dbg !10
  %6 = sext i32 %5 to i64, !dbg !29
  call void @llvm.dbg.declare(metadata i64* %z_b_1_308, metadata !28, metadata !DIExpression()), !dbg !10
  store i64 %6, i64* %z_b_1_308, align 8, !dbg !29
  %7 = load i64, i64* %z_b_1_308, align 8, !dbg !29
  call void @llvm.dbg.value(metadata i64 %7, metadata !28, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_60_311, metadata !28, metadata !DIExpression()), !dbg !10
  store i64 %7, i64* %z_e_60_311, align 8, !dbg !29
  %8 = bitcast [16 x i64]* %"a$sd1_338" to i8*, !dbg !29
  %9 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !29
  %10 = bitcast i64* @.C339_MAIN_ to i8*, !dbg !29
  %11 = bitcast i64* @.C340_MAIN_ to i8*, !dbg !29
  %12 = bitcast i64* %z_b_0_307 to i8*, !dbg !29
  %13 = bitcast i64* %z_b_1_308 to i8*, !dbg !29
  %14 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !29
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %14(i8* %8, i8* %9, i8* %10, i8* %11, i8* %12, i8* %13), !dbg !29
  %15 = bitcast [16 x i64]* %"a$sd1_338" to i8*, !dbg !29
  %16 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !29
  call void (i8*, i32, ...) %16(i8* %15, i32 25), !dbg !29
  %17 = load i64, i64* %z_b_1_308, align 8, !dbg !29
  call void @llvm.dbg.value(metadata i64 %17, metadata !28, metadata !DIExpression()), !dbg !10
  %18 = load i64, i64* %z_b_0_307, align 8, !dbg !29
  call void @llvm.dbg.value(metadata i64 %18, metadata !28, metadata !DIExpression()), !dbg !10
  %19 = sub nsw i64 %18, 1, !dbg !29
  %20 = sub nsw i64 %17, %19, !dbg !29
  call void @llvm.dbg.declare(metadata i64* %z_b_2_309, metadata !28, metadata !DIExpression()), !dbg !10
  store i64 %20, i64* %z_b_2_309, align 8, !dbg !29
  %21 = load i64, i64* %z_b_0_307, align 8, !dbg !29
  call void @llvm.dbg.value(metadata i64 %21, metadata !28, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_b_3_310, metadata !28, metadata !DIExpression()), !dbg !10
  store i64 %21, i64* %z_b_3_310, align 8, !dbg !29
  %22 = bitcast i64* %z_b_2_309 to i8*, !dbg !29
  %23 = bitcast i64* @.C339_MAIN_ to i8*, !dbg !29
  %24 = bitcast i64* @.C340_MAIN_ to i8*, !dbg !29
  %25 = bitcast i32** %.Z0965_315 to i8*, !dbg !29
  %26 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !29
  %27 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !29
  %28 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !29
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %28(i8* %22, i8* %23, i8* %24, i8* null, i8* %25, i8* null, i8* %26, i8* %27, i8* null, i64 0), !dbg !29
  %29 = load i32, i32* %len_314, align 4, !dbg !30
  call void @llvm.dbg.value(metadata i32 %29, metadata !26, metadata !DIExpression()), !dbg !10
  store i32 %29, i32* %.dY0001_349, align 4, !dbg !30
  call void @llvm.dbg.declare(metadata i32* %i_306, metadata !31, metadata !DIExpression()), !dbg !10
  store i32 1, i32* %i_306, align 4, !dbg !30
  %30 = load i32, i32* %.dY0001_349, align 4, !dbg !30
  %31 = icmp sle i32 %30, 0, !dbg !30
  br i1 %31, label %L.LB1_348, label %L.LB1_347, !dbg !30

L.LB1_347:                                        ; preds = %L.LB1_347, %L.LB1_369
  %32 = load i32, i32* %i_306, align 4, !dbg !32
  call void @llvm.dbg.value(metadata i32 %32, metadata !31, metadata !DIExpression()), !dbg !10
  %33 = load i32, i32* %i_306, align 4, !dbg !32
  call void @llvm.dbg.value(metadata i32 %33, metadata !31, metadata !DIExpression()), !dbg !10
  %34 = sext i32 %33 to i64, !dbg !32
  %35 = bitcast [16 x i64]* %"a$sd1_338" to i8*, !dbg !32
  %36 = getelementptr i8, i8* %35, i64 56, !dbg !32
  %37 = bitcast i8* %36 to i64*, !dbg !32
  %38 = load i64, i64* %37, align 8, !dbg !32
  %39 = add nsw i64 %34, %38, !dbg !32
  %40 = load i32*, i32** %.Z0965_315, align 8, !dbg !32
  call void @llvm.dbg.value(metadata i32* %40, metadata !17, metadata !DIExpression()), !dbg !10
  %41 = bitcast i32* %40 to i8*, !dbg !32
  %42 = getelementptr i8, i8* %41, i64 -4, !dbg !32
  %43 = bitcast i8* %42 to i32*, !dbg !32
  %44 = getelementptr i32, i32* %43, i64 %39, !dbg !32
  store i32 %32, i32* %44, align 4, !dbg !32
  %45 = load i32, i32* %i_306, align 4, !dbg !33
  call void @llvm.dbg.value(metadata i32 %45, metadata !31, metadata !DIExpression()), !dbg !10
  %46 = add nsw i32 %45, 1, !dbg !33
  store i32 %46, i32* %i_306, align 4, !dbg !33
  %47 = load i32, i32* %.dY0001_349, align 4, !dbg !33
  %48 = sub nsw i32 %47, 1, !dbg !33
  store i32 %48, i32* %.dY0001_349, align 4, !dbg !33
  %49 = load i32, i32* %.dY0001_349, align 4, !dbg !33
  %50 = icmp sgt i32 %49, 0, !dbg !33
  br i1 %50, label %L.LB1_347, label %L.LB1_348, !dbg !33

L.LB1_348:                                        ; preds = %L.LB1_347, %L.LB1_369
  %51 = bitcast i32* %len_314 to i8*, !dbg !34
  %52 = bitcast %astruct.dt68* %.uplevelArgPack0001_390 to i8**, !dbg !34
  store i8* %51, i8** %52, align 8, !dbg !34
  %53 = bitcast i32** %.Z0965_315 to i8*, !dbg !34
  %54 = bitcast %astruct.dt68* %.uplevelArgPack0001_390 to i8*, !dbg !34
  %55 = getelementptr i8, i8* %54, i64 8, !dbg !34
  %56 = bitcast i8* %55 to i8**, !dbg !34
  store i8* %53, i8** %56, align 8, !dbg !34
  %57 = bitcast i32** %.Z0965_315 to i8*, !dbg !34
  %58 = bitcast %astruct.dt68* %.uplevelArgPack0001_390 to i8*, !dbg !34
  %59 = getelementptr i8, i8* %58, i64 16, !dbg !34
  %60 = bitcast i8* %59 to i8**, !dbg !34
  store i8* %57, i8** %60, align 8, !dbg !34
  %61 = bitcast i64* %z_b_0_307 to i8*, !dbg !34
  %62 = bitcast %astruct.dt68* %.uplevelArgPack0001_390 to i8*, !dbg !34
  %63 = getelementptr i8, i8* %62, i64 24, !dbg !34
  %64 = bitcast i8* %63 to i8**, !dbg !34
  store i8* %61, i8** %64, align 8, !dbg !34
  %65 = bitcast i64* %z_b_1_308 to i8*, !dbg !34
  %66 = bitcast %astruct.dt68* %.uplevelArgPack0001_390 to i8*, !dbg !34
  %67 = getelementptr i8, i8* %66, i64 32, !dbg !34
  %68 = bitcast i8* %67 to i8**, !dbg !34
  store i8* %65, i8** %68, align 8, !dbg !34
  %69 = bitcast i64* %z_e_60_311 to i8*, !dbg !34
  %70 = bitcast %astruct.dt68* %.uplevelArgPack0001_390 to i8*, !dbg !34
  %71 = getelementptr i8, i8* %70, i64 40, !dbg !34
  %72 = bitcast i8* %71 to i8**, !dbg !34
  store i8* %69, i8** %72, align 8, !dbg !34
  %73 = bitcast i64* %z_b_2_309 to i8*, !dbg !34
  %74 = bitcast %astruct.dt68* %.uplevelArgPack0001_390 to i8*, !dbg !34
  %75 = getelementptr i8, i8* %74, i64 48, !dbg !34
  %76 = bitcast i8* %75 to i8**, !dbg !34
  store i8* %73, i8** %76, align 8, !dbg !34
  %77 = bitcast i64* %z_b_3_310 to i8*, !dbg !34
  %78 = bitcast %astruct.dt68* %.uplevelArgPack0001_390 to i8*, !dbg !34
  %79 = getelementptr i8, i8* %78, i64 56, !dbg !34
  %80 = bitcast i8* %79 to i8**, !dbg !34
  store i8* %77, i8** %80, align 8, !dbg !34
  %81 = bitcast [16 x i64]* %"a$sd1_338" to i8*, !dbg !34
  %82 = bitcast %astruct.dt68* %.uplevelArgPack0001_390 to i8*, !dbg !34
  %83 = getelementptr i8, i8* %82, i64 64, !dbg !34
  %84 = bitcast i8* %83 to i8**, !dbg !34
  store i8* %81, i8** %84, align 8, !dbg !34
  br label %L.LB1_409, !dbg !34

L.LB1_409:                                        ; preds = %L.LB1_348
  %85 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L25_1_ to i64*, !dbg !34
  %86 = bitcast %astruct.dt68* %.uplevelArgPack0001_390 to i64*, !dbg !34
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %85, i64* %86), !dbg !34
  call void (...) @_mp_bcs_nest(), !dbg !35
  %87 = bitcast i32* @.C325_MAIN_ to i8*, !dbg !35
  %88 = bitcast [53 x i8]* @.C323_MAIN_ to i8*, !dbg !35
  %89 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !35
  call void (i8*, i8*, i64, ...) %89(i8* %87, i8* %88, i64 53), !dbg !35
  %90 = bitcast i32* @.C327_MAIN_ to i8*, !dbg !35
  %91 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !35
  %92 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !35
  %93 = bitcast %struct.STATICS1* @.STATICS1 to i8*, !dbg !35
  %94 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !35
  %95 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %94(i8* %90, i8* null, i8* %91, i8* %92, i8* %93, i8* null, i64 0), !dbg !35
  call void @llvm.dbg.declare(metadata i32* %z__io_329, metadata !36, metadata !DIExpression()), !dbg !10
  store i32 %95, i32* %z__io_329, align 4, !dbg !35
  %96 = bitcast [16 x i64]* %"a$sd1_338" to i8*, !dbg !35
  %97 = getelementptr i8, i8* %96, i64 56, !dbg !35
  %98 = bitcast i8* %97 to i64*, !dbg !35
  %99 = load i64, i64* %98, align 8, !dbg !35
  %100 = load i32*, i32** %.Z0965_315, align 8, !dbg !35
  call void @llvm.dbg.value(metadata i32* %100, metadata !17, metadata !DIExpression()), !dbg !10
  %101 = bitcast i32* %100 to i8*, !dbg !35
  %102 = getelementptr i8, i8* %101, i64 196, !dbg !35
  %103 = bitcast i8* %102 to i32*, !dbg !35
  %104 = getelementptr i32, i32* %103, i64 %99, !dbg !35
  %105 = load i32, i32* %104, align 4, !dbg !35
  %106 = bitcast i32 (...)* @f90io_sc_i_fmt_write to i32 (i32, i32, ...)*, !dbg !35
  %107 = call i32 (i32, i32, ...) %106(i32 %105, i32 25), !dbg !35
  store i32 %107, i32* %z__io_329, align 4, !dbg !35
  %108 = call i32 (...) @f90io_fmtw_end(), !dbg !35
  store i32 %108, i32* %z__io_329, align 4, !dbg !35
  call void (...) @_mp_ecs_nest(), !dbg !35
  %109 = load i32*, i32** %.Z0965_315, align 8, !dbg !37
  call void @llvm.dbg.value(metadata i32* %109, metadata !17, metadata !DIExpression()), !dbg !10
  %110 = bitcast i32* %109 to i8*, !dbg !37
  %111 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !37
  %112 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !37
  call void (i8*, i8*, i8*, i8*, i64, ...) %112(i8* null, i8* %110, i8* %111, i8* null, i64 0), !dbg !37
  %113 = bitcast i32** %.Z0965_315 to i8**, !dbg !37
  store i8* null, i8** %113, align 8, !dbg !37
  %114 = bitcast [16 x i64]* %"a$sd1_338" to i64*, !dbg !37
  store i64 0, i64* %114, align 8, !dbg !37
  ret void, !dbg !15
}

define internal void @__nv_MAIN__F1L25_1_(i32* %__nv_MAIN__F1L25_1Arg0, i64* %__nv_MAIN__F1L25_1Arg1, i64* %__nv_MAIN__F1L25_1Arg2) #0 !dbg !38 {
L.entry:
  %__gtid___nv_MAIN__F1L25_1__459 = alloca i32, align 4
  %.i0000p_320 = alloca i32, align 4
  %i_319 = alloca i32, align 4
  %.du0002p_353 = alloca i32, align 4
  %.de0002p_354 = alloca i32, align 4
  %.di0002p_355 = alloca i32, align 4
  %.ds0002p_356 = alloca i32, align 4
  %.dl0002p_358 = alloca i32, align 4
  %.dl0002p.copy_453 = alloca i32, align 4
  %.de0002p.copy_454 = alloca i32, align 4
  %.ds0002p.copy_455 = alloca i32, align 4
  %.dX0002p_357 = alloca i32, align 4
  %.dY0002p_352 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L25_1Arg0, metadata !41, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L25_1Arg1, metadata !43, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L25_1Arg2, metadata !44, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.value(metadata i32 1, metadata !45, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.value(metadata i32 0, metadata !46, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.value(metadata i32 1, metadata !47, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.value(metadata i32 0, metadata !48, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.value(metadata i32 1, metadata !49, metadata !DIExpression()), !dbg !42
  %0 = load i32, i32* %__nv_MAIN__F1L25_1Arg0, align 4, !dbg !50
  store i32 %0, i32* %__gtid___nv_MAIN__F1L25_1__459, align 4, !dbg !50
  br label %L.LB2_444

L.LB2_444:                                        ; preds = %L.entry
  br label %L.LB2_318

L.LB2_318:                                        ; preds = %L.LB2_444
  store i32 0, i32* %.i0000p_320, align 4, !dbg !51
  call void @llvm.dbg.declare(metadata i32* %i_319, metadata !52, metadata !DIExpression()), !dbg !50
  store i32 1, i32* %i_319, align 4, !dbg !51
  %1 = bitcast i64* %__nv_MAIN__F1L25_1Arg2 to i32**, !dbg !51
  %2 = load i32*, i32** %1, align 8, !dbg !51
  %3 = load i32, i32* %2, align 4, !dbg !51
  %4 = sub nsw i32 %3, 1, !dbg !51
  store i32 %4, i32* %.du0002p_353, align 4, !dbg !51
  %5 = bitcast i64* %__nv_MAIN__F1L25_1Arg2 to i32**, !dbg !51
  %6 = load i32*, i32** %5, align 8, !dbg !51
  %7 = load i32, i32* %6, align 4, !dbg !51
  %8 = sub nsw i32 %7, 1, !dbg !51
  store i32 %8, i32* %.de0002p_354, align 4, !dbg !51
  store i32 1, i32* %.di0002p_355, align 4, !dbg !51
  %9 = load i32, i32* %.di0002p_355, align 4, !dbg !51
  store i32 %9, i32* %.ds0002p_356, align 4, !dbg !51
  store i32 1, i32* %.dl0002p_358, align 4, !dbg !51
  %10 = load i32, i32* %.dl0002p_358, align 4, !dbg !51
  store i32 %10, i32* %.dl0002p.copy_453, align 4, !dbg !51
  %11 = load i32, i32* %.de0002p_354, align 4, !dbg !51
  store i32 %11, i32* %.de0002p.copy_454, align 4, !dbg !51
  %12 = load i32, i32* %.ds0002p_356, align 4, !dbg !51
  store i32 %12, i32* %.ds0002p.copy_455, align 4, !dbg !51
  %13 = load i32, i32* %__gtid___nv_MAIN__F1L25_1__459, align 4, !dbg !51
  %14 = bitcast i32* %.i0000p_320 to i64*, !dbg !51
  %15 = bitcast i32* %.dl0002p.copy_453 to i64*, !dbg !51
  %16 = bitcast i32* %.de0002p.copy_454 to i64*, !dbg !51
  %17 = bitcast i32* %.ds0002p.copy_455 to i64*, !dbg !51
  %18 = load i32, i32* %.ds0002p.copy_455, align 4, !dbg !51
  call void @__kmpc_for_static_init_4(i64* null, i32 %13, i32 34, i64* %14, i64* %15, i64* %16, i64* %17, i32 %18, i32 1), !dbg !51
  %19 = load i32, i32* %.dl0002p.copy_453, align 4, !dbg !51
  store i32 %19, i32* %.dl0002p_358, align 4, !dbg !51
  %20 = load i32, i32* %.de0002p.copy_454, align 4, !dbg !51
  store i32 %20, i32* %.de0002p_354, align 4, !dbg !51
  %21 = load i32, i32* %.ds0002p.copy_455, align 4, !dbg !51
  store i32 %21, i32* %.ds0002p_356, align 4, !dbg !51
  %22 = load i32, i32* %.dl0002p_358, align 4, !dbg !51
  store i32 %22, i32* %i_319, align 4, !dbg !51
  %23 = load i32, i32* %i_319, align 4, !dbg !51
  call void @llvm.dbg.value(metadata i32 %23, metadata !52, metadata !DIExpression()), !dbg !50
  store i32 %23, i32* %.dX0002p_357, align 4, !dbg !51
  %24 = load i32, i32* %.dX0002p_357, align 4, !dbg !51
  %25 = load i32, i32* %.du0002p_353, align 4, !dbg !51
  %26 = icmp sgt i32 %24, %25, !dbg !51
  br i1 %26, label %L.LB2_351, label %L.LB2_483, !dbg !51

L.LB2_483:                                        ; preds = %L.LB2_318
  %27 = load i32, i32* %.dX0002p_357, align 4, !dbg !51
  store i32 %27, i32* %i_319, align 4, !dbg !51
  %28 = load i32, i32* %.di0002p_355, align 4, !dbg !51
  %29 = load i32, i32* %.de0002p_354, align 4, !dbg !51
  %30 = load i32, i32* %.dX0002p_357, align 4, !dbg !51
  %31 = sub nsw i32 %29, %30, !dbg !51
  %32 = add nsw i32 %28, %31, !dbg !51
  %33 = load i32, i32* %.di0002p_355, align 4, !dbg !51
  %34 = sdiv i32 %32, %33, !dbg !51
  store i32 %34, i32* %.dY0002p_352, align 4, !dbg !51
  %35 = load i32, i32* %.dY0002p_352, align 4, !dbg !51
  %36 = icmp sle i32 %35, 0, !dbg !51
  br i1 %36, label %L.LB2_361, label %L.LB2_360, !dbg !51

L.LB2_360:                                        ; preds = %L.LB2_360, %L.LB2_483
  %37 = load i32, i32* %i_319, align 4, !dbg !53
  call void @llvm.dbg.value(metadata i32 %37, metadata !52, metadata !DIExpression()), !dbg !50
  %38 = sext i32 %37 to i64, !dbg !53
  %39 = bitcast i64* %__nv_MAIN__F1L25_1Arg2 to i8*, !dbg !53
  %40 = getelementptr i8, i8* %39, i64 64, !dbg !53
  %41 = bitcast i8* %40 to i8**, !dbg !53
  %42 = load i8*, i8** %41, align 8, !dbg !53
  %43 = getelementptr i8, i8* %42, i64 56, !dbg !53
  %44 = bitcast i8* %43 to i64*, !dbg !53
  %45 = load i64, i64* %44, align 8, !dbg !53
  %46 = add nsw i64 %38, %45, !dbg !53
  %47 = bitcast i64* %__nv_MAIN__F1L25_1Arg2 to i8*, !dbg !53
  %48 = getelementptr i8, i8* %47, i64 16, !dbg !53
  %49 = bitcast i8* %48 to i8***, !dbg !53
  %50 = load i8**, i8*** %49, align 8, !dbg !53
  %51 = load i8*, i8** %50, align 8, !dbg !53
  %52 = getelementptr i8, i8* %51, i64 -4, !dbg !53
  %53 = bitcast i8* %52 to i32*, !dbg !53
  %54 = getelementptr i32, i32* %53, i64 %46, !dbg !53
  %55 = load i32, i32* %54, align 4, !dbg !53
  %56 = add nsw i32 %55, 1, !dbg !53
  %57 = load i32, i32* %i_319, align 4, !dbg !53
  call void @llvm.dbg.value(metadata i32 %57, metadata !52, metadata !DIExpression()), !dbg !50
  %58 = sext i32 %57 to i64, !dbg !53
  %59 = bitcast i64* %__nv_MAIN__F1L25_1Arg2 to i8*, !dbg !53
  %60 = getelementptr i8, i8* %59, i64 64, !dbg !53
  %61 = bitcast i8* %60 to i8**, !dbg !53
  %62 = load i8*, i8** %61, align 8, !dbg !53
  %63 = getelementptr i8, i8* %62, i64 56, !dbg !53
  %64 = bitcast i8* %63 to i64*, !dbg !53
  %65 = load i64, i64* %64, align 8, !dbg !53
  %66 = add nsw i64 %58, %65, !dbg !53
  %67 = bitcast i64* %__nv_MAIN__F1L25_1Arg2 to i8*, !dbg !53
  %68 = getelementptr i8, i8* %67, i64 16, !dbg !53
  %69 = bitcast i8* %68 to i32***, !dbg !53
  %70 = load i32**, i32*** %69, align 8, !dbg !53
  %71 = load i32*, i32** %70, align 8, !dbg !53
  %72 = getelementptr i32, i32* %71, i64 %66, !dbg !53
  store i32 %56, i32* %72, align 4, !dbg !53
  %73 = load i32, i32* %.di0002p_355, align 4, !dbg !50
  %74 = load i32, i32* %i_319, align 4, !dbg !50
  call void @llvm.dbg.value(metadata i32 %74, metadata !52, metadata !DIExpression()), !dbg !50
  %75 = add nsw i32 %73, %74, !dbg !50
  store i32 %75, i32* %i_319, align 4, !dbg !50
  %76 = load i32, i32* %.dY0002p_352, align 4, !dbg !50
  %77 = sub nsw i32 %76, 1, !dbg !50
  store i32 %77, i32* %.dY0002p_352, align 4, !dbg !50
  %78 = load i32, i32* %.dY0002p_352, align 4, !dbg !50
  %79 = icmp sgt i32 %78, 0, !dbg !50
  br i1 %79, label %L.LB2_360, label %L.LB2_361, !dbg !50

L.LB2_361:                                        ; preds = %L.LB2_360, %L.LB2_483
  br label %L.LB2_351

L.LB2_351:                                        ; preds = %L.LB2_361, %L.LB2_318
  %80 = load i32, i32* %__gtid___nv_MAIN__F1L25_1__459, align 4, !dbg !50
  call void @__kmpc_for_static_fini(i64* null, i32 %80), !dbg !50
  br label %L.LB2_321

L.LB2_321:                                        ; preds = %L.LB2_351
  ret void, !dbg !50
}

declare void @__kmpc_for_static_fini(i64*, i32) #0

declare void @__kmpc_for_static_init_4(i64*, i32, i32, i64*, i64*, i64*, i64*, i32, i32) #0

declare void @f90_dealloc03a_i8(...) #0

declare void @_mp_ecs_nest(...) #0

declare signext i32 @f90io_fmtw_end(...) #0

declare signext i32 @f90io_sc_i_fmt_write(...) #0

declare signext i32 @f90io_fmtw_inita(...) #0

declare void @f90io_src_info03a(...) #0

declare void @_mp_bcs_nest(...) #0

declare void @f90_alloc04_chka_i8(...) #0

declare void @f90_set_intrin_type_i8(...) #0

declare void @f90_template1_i8(...) #0

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare void @fort_init(...) #0

declare signext i32 @__kmpc_global_thread_num(i64*) #0

declare void @__kmpc_fork_call(i64*, i32, i64*, i64*, ...) #0

declare void @__kmpc_end_serialized_parallel(i64*, i32) #0

declare void @__kmpc_serialized_parallel(i64*, i32) #0

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { "no-frame-pointer-elim-non-leaf" }
attributes #1 = { nounwind readnone speculatable }

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !4, globals: !4, imports: !4)
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB029-truedep1-orig-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb029_truedep1_orig_yes", scope: !2, file: !3, line: 11, type: !6, scopeLine: 11, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!6 = !DISubroutineType(cc: DW_CC_program, types: !7)
!7 = !{null}
!8 = !DILocalVariable(name: "omp_sched_static", scope: !5, file: !3, type: !9)
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 0, scope: !5)
!11 = !DILocalVariable(name: "omp_proc_bind_false", scope: !5, file: !3, type: !9)
!12 = !DILocalVariable(name: "omp_proc_bind_true", scope: !5, file: !3, type: !9)
!13 = !DILocalVariable(name: "omp_lock_hint_none", scope: !5, file: !3, type: !9)
!14 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !5, file: !3, type: !9)
!15 = !DILocation(line: 35, column: 1, scope: !5)
!16 = !DILocation(line: 11, column: 1, scope: !5)
!17 = !DILocalVariable(name: "a", scope: !5, file: !3, type: !18)
!18 = !DICompositeType(tag: DW_TAG_array_type, baseType: !9, size: 32, align: 32, elements: !19)
!19 = !{!20}
!20 = !DISubrange(count: 0, lowerBound: 1)
!21 = !DILocalVariable(scope: !5, file: !3, type: !22, flags: DIFlagArtificial)
!22 = !DICompositeType(tag: DW_TAG_array_type, baseType: !23, size: 1024, align: 64, elements: !24)
!23 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!24 = !{!25}
!25 = !DISubrange(count: 16, lowerBound: 1)
!26 = !DILocalVariable(name: "len", scope: !5, file: !3, type: !9)
!27 = !DILocation(line: 18, column: 1, scope: !5)
!28 = !DILocalVariable(scope: !5, file: !3, type: !23, flags: DIFlagArtificial)
!29 = !DILocation(line: 19, column: 1, scope: !5)
!30 = !DILocation(line: 21, column: 1, scope: !5)
!31 = !DILocalVariable(name: "i", scope: !5, file: !3, type: !9)
!32 = !DILocation(line: 22, column: 1, scope: !5)
!33 = !DILocation(line: 23, column: 1, scope: !5)
!34 = !DILocation(line: 25, column: 1, scope: !5)
!35 = !DILocation(line: 31, column: 1, scope: !5)
!36 = !DILocalVariable(scope: !5, file: !3, type: !9, flags: DIFlagArtificial)
!37 = !DILocation(line: 34, column: 1, scope: !5)
!38 = distinct !DISubprogram(name: "__nv_MAIN__F1L25_1", scope: !2, file: !3, line: 25, type: !39, scopeLine: 25, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!39 = !DISubroutineType(types: !40)
!40 = !{null, !9, !23, !23}
!41 = !DILocalVariable(name: "__nv_MAIN__F1L25_1Arg0", arg: 1, scope: !38, file: !3, type: !9)
!42 = !DILocation(line: 0, scope: !38)
!43 = !DILocalVariable(name: "__nv_MAIN__F1L25_1Arg1", arg: 2, scope: !38, file: !3, type: !23)
!44 = !DILocalVariable(name: "__nv_MAIN__F1L25_1Arg2", arg: 3, scope: !38, file: !3, type: !23)
!45 = !DILocalVariable(name: "omp_sched_static", scope: !38, file: !3, type: !9)
!46 = !DILocalVariable(name: "omp_proc_bind_false", scope: !38, file: !3, type: !9)
!47 = !DILocalVariable(name: "omp_proc_bind_true", scope: !38, file: !3, type: !9)
!48 = !DILocalVariable(name: "omp_lock_hint_none", scope: !38, file: !3, type: !9)
!49 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !38, file: !3, type: !9)
!50 = !DILocation(line: 28, column: 1, scope: !38)
!51 = !DILocation(line: 26, column: 1, scope: !38)
!52 = !DILocalVariable(name: "i", scope: !38, file: !3, type: !9)
!53 = !DILocation(line: 27, column: 1, scope: !38)

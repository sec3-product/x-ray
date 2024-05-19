; ModuleID = '/tmp/DRB012-minusminus-var-yes-f56d2a.ll'
source_filename = "/tmp/DRB012-minusminus-var-yes-f56d2a.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%astruct.dt88 = type <{ i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8* }>

@.C369_MAIN_ = internal constant [11 x i8] c"numNodes2 ="
@.C367_MAIN_ = internal constant i32 64
@.C363_MAIN_ = internal constant i32 -1
@.C359_MAIN_ = internal constant i32 -5
@.C351_MAIN_ = internal constant i32 5
@.C300_MAIN_ = internal constant i32 2
@.C383_MAIN_ = internal constant i64 25
@.C357_MAIN_ = internal constant [29 x i8] c"Error, invalid integer value."
@.C356_MAIN_ = internal constant i32 39
@.C310_MAIN_ = internal constant i32 25
@.C352_MAIN_ = internal constant [5 x i8] c"(i10)"
@.C348_MAIN_ = internal constant i32 37
@.C376_MAIN_ = internal constant i64 4
@.C349_MAIN_ = internal constant [37 x i8] c"Allocation error, program terminated."
@.C312_MAIN_ = internal constant i32 28
@.C380_MAIN_ = internal constant i64 80
@.C379_MAIN_ = internal constant i64 14
@.C340_MAIN_ = internal constant [35 x i8] c"No command line arguments provided."
@.C338_MAIN_ = internal constant i32 6
@.C339_MAIN_ = internal constant [3 x i8] c"(a)"
@.C285_MAIN_ = internal constant i32 1
@.C309_MAIN_ = internal constant i32 14
@.C336_MAIN_ = internal constant [54 x i8] c"micro-benchmarks-fortran/DRB012-minusminus-var-yes.f95"
@.C311_MAIN_ = internal constant i32 23
@.C332_MAIN_ = internal constant i32 100
@.C286_MAIN_ = internal constant i64 1
@.C284_MAIN_ = internal constant i64 0
@.C283_MAIN_ = internal constant i32 0
@.C363___nv_MAIN__F1L56_1 = internal constant i32 -1
@.C285___nv_MAIN__F1L56_1 = internal constant i32 1
@.C283___nv_MAIN__F1L56_1 = internal constant i32 0

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__511 = alloca i32, align 4
  %.Z0978_358 = alloca i32*, align 8
  %"x$sd2_382" = alloca [16 x i64], align 8
  %.Z0972_347 = alloca [80 x i8]*, align 8
  %"args$sd1_378" = alloca [16 x i64], align 8
  %len_333 = alloca i32, align 4
  %argcount_316 = alloca i32, align 4
  %z__io_342 = alloca i32, align 4
  %z_b_0_320 = alloca i64, align 8
  %z_b_1_321 = alloca i64, align 8
  %z_e_61_324 = alloca i64, align 8
  %z_b_2_322 = alloca i64, align 8
  %z_b_3_323 = alloca i64, align 8
  %allocstatus_317 = alloca i32, align 4
  %.dY0001_393 = alloca i32, align 4
  %ix_319 = alloca i32, align 4
  %rderr_318 = alloca i32, align 4
  %z_b_4_326 = alloca i64, align 8
  %z_b_5_327 = alloca i64, align 8
  %z_e_68_330 = alloca i64, align 8
  %z_b_6_328 = alloca i64, align 8
  %z_b_7_329 = alloca i64, align 8
  %numnodes_314 = alloca i32, align 4
  %numnodes2_315 = alloca i32, align 4
  %.dY0002_398 = alloca i32, align 4
  %i_313 = alloca i32, align 4
  %.uplevelArgPack0001_488 = alloca %astruct.dt88, align 16
  call void @llvm.dbg.value(metadata i32 1, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 2, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 2, metadata !14, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !15, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !16, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 2, metadata !17, metadata !DIExpression()), !dbg !10
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !18
  store i32 %0, i32* %__gtid_MAIN__511, align 4, !dbg !18
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !19
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !19
  call void (i8*, ...) %2(i8* %1), !dbg !19
  call void @llvm.dbg.declare(metadata i32** %.Z0978_358, metadata !20, metadata !DIExpression(DW_OP_deref)), !dbg !10
  %3 = bitcast i32** %.Z0978_358 to i8**, !dbg !19
  store i8* null, i8** %3, align 8, !dbg !19
  call void @llvm.dbg.declare(metadata [16 x i64]* %"x$sd2_382", metadata !24, metadata !DIExpression()), !dbg !10
  %4 = bitcast [16 x i64]* %"x$sd2_382" to i64*, !dbg !19
  store i64 0, i64* %4, align 8, !dbg !19
  call void @llvm.dbg.declare(metadata [80 x i8]** %.Z0972_347, metadata !29, metadata !DIExpression(DW_OP_deref)), !dbg !10
  %5 = bitcast [80 x i8]** %.Z0972_347 to i8**, !dbg !19
  store i8* null, i8** %5, align 8, !dbg !19
  call void @llvm.dbg.declare(metadata [16 x i64]* %"args$sd1_378", metadata !24, metadata !DIExpression()), !dbg !10
  %6 = bitcast [16 x i64]* %"args$sd1_378" to i64*, !dbg !19
  store i64 0, i64* %6, align 8, !dbg !19
  br label %L.LB1_423

L.LB1_423:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %len_333, metadata !32, metadata !DIExpression()), !dbg !10
  store i32 100, i32* %len_333, align 4, !dbg !33
  %7 = call i32 (...) @f90_cmd_arg_cnt(), !dbg !34
  call void @llvm.dbg.declare(metadata i32* %argcount_316, metadata !35, metadata !DIExpression()), !dbg !10
  store i32 %7, i32* %argcount_316, align 4, !dbg !34
  %8 = load i32, i32* %argcount_316, align 4, !dbg !36
  call void @llvm.dbg.value(metadata i32 %8, metadata !35, metadata !DIExpression()), !dbg !10
  %9 = icmp ne i32 %8, 0, !dbg !36
  br i1 %9, label %L.LB1_387, label %L.LB1_533, !dbg !36

L.LB1_533:                                        ; preds = %L.LB1_423
  call void (...) @_mp_bcs_nest(), !dbg !37
  %10 = bitcast i32* @.C311_MAIN_ to i8*, !dbg !37
  %11 = bitcast [54 x i8]* @.C336_MAIN_ to i8*, !dbg !37
  %12 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !37
  call void (i8*, i8*, i64, ...) %12(i8* %10, i8* %11, i64 54), !dbg !37
  %13 = bitcast i32* @.C309_MAIN_ to i8*, !dbg !37
  %14 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !37
  %15 = bitcast [3 x i8]* @.C339_MAIN_ to i8*, !dbg !37
  %16 = bitcast i32 (...)* @f90io_encode_fmta to i32 (i8*, i8*, i8*, i64, ...)*, !dbg !37
  %17 = call i32 (i8*, i8*, i8*, i64, ...) %16(i8* %13, i8* %14, i8* %15, i64 3), !dbg !37
  call void @llvm.dbg.declare(metadata i32* %z__io_342, metadata !38, metadata !DIExpression()), !dbg !10
  store i32 %17, i32* %z__io_342, align 4, !dbg !37
  %18 = bitcast i32* @.C338_MAIN_ to i8*, !dbg !37
  %19 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !37
  %20 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !37
  %21 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !37
  %22 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %21(i8* %18, i8* null, i8* %19, i8* %20, i8* null, i8* null, i64 0), !dbg !37
  store i32 %22, i32* %z__io_342, align 4, !dbg !37
  %23 = bitcast i32* @.C309_MAIN_ to i8*, !dbg !37
  %24 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !37
  %25 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !37
  %26 = bitcast [35 x i8]* @.C340_MAIN_ to i8*, !dbg !37
  %27 = bitcast i32 (...)* @f90io_fmt_writea to i32 (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !37
  %28 = call i32 (i8*, i8*, i8*, i8*, i64, ...) %27(i8* %23, i8* %24, i8* %25, i8* %26, i64 35), !dbg !37
  store i32 %28, i32* %z__io_342, align 4, !dbg !37
  %29 = call i32 (...) @f90io_fmtw_end(), !dbg !37
  store i32 %29, i32* %z__io_342, align 4, !dbg !37
  call void (...) @_mp_ecs_nest(), !dbg !37
  br label %L.LB1_387

L.LB1_387:                                        ; preds = %L.LB1_533, %L.LB1_423
  call void @llvm.dbg.declare(metadata i64* %z_b_0_320, metadata !39, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_0_320, align 8, !dbg !40
  %30 = load i32, i32* %argcount_316, align 4, !dbg !40
  call void @llvm.dbg.value(metadata i32 %30, metadata !35, metadata !DIExpression()), !dbg !10
  %31 = sext i32 %30 to i64, !dbg !40
  call void @llvm.dbg.declare(metadata i64* %z_b_1_321, metadata !39, metadata !DIExpression()), !dbg !10
  store i64 %31, i64* %z_b_1_321, align 8, !dbg !40
  %32 = load i64, i64* %z_b_1_321, align 8, !dbg !40
  call void @llvm.dbg.value(metadata i64 %32, metadata !39, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_61_324, metadata !39, metadata !DIExpression()), !dbg !10
  store i64 %32, i64* %z_e_61_324, align 8, !dbg !40
  %33 = bitcast [16 x i64]* %"args$sd1_378" to i8*, !dbg !40
  %34 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !40
  %35 = bitcast i64* @.C379_MAIN_ to i8*, !dbg !40
  %36 = bitcast i64* @.C380_MAIN_ to i8*, !dbg !40
  %37 = bitcast i64* %z_b_0_320 to i8*, !dbg !40
  %38 = bitcast i64* %z_b_1_321 to i8*, !dbg !40
  %39 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !40
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %39(i8* %33, i8* %34, i8* %35, i8* %36, i8* %37, i8* %38), !dbg !40
  %40 = bitcast [16 x i64]* %"args$sd1_378" to i8*, !dbg !40
  %41 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !40
  call void (i8*, i32, ...) %41(i8* %40, i32 14), !dbg !40
  %42 = load i64, i64* %z_b_1_321, align 8, !dbg !40
  call void @llvm.dbg.value(metadata i64 %42, metadata !39, metadata !DIExpression()), !dbg !10
  %43 = load i64, i64* %z_b_0_320, align 8, !dbg !40
  call void @llvm.dbg.value(metadata i64 %43, metadata !39, metadata !DIExpression()), !dbg !10
  %44 = sub nsw i64 %43, 1, !dbg !40
  %45 = sub nsw i64 %42, %44, !dbg !40
  call void @llvm.dbg.declare(metadata i64* %z_b_2_322, metadata !39, metadata !DIExpression()), !dbg !10
  store i64 %45, i64* %z_b_2_322, align 8, !dbg !40
  %46 = load i64, i64* %z_b_0_320, align 8, !dbg !40
  call void @llvm.dbg.value(metadata i64 %46, metadata !39, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_b_3_323, metadata !39, metadata !DIExpression()), !dbg !10
  store i64 %46, i64* %z_b_3_323, align 8, !dbg !40
  %47 = bitcast i64* %z_b_2_322 to i8*, !dbg !40
  %48 = bitcast i64* @.C379_MAIN_ to i8*, !dbg !40
  %49 = bitcast i64* @.C380_MAIN_ to i8*, !dbg !40
  call void @llvm.dbg.declare(metadata i32* %allocstatus_317, metadata !41, metadata !DIExpression()), !dbg !10
  %50 = bitcast i32* %allocstatus_317 to i8*, !dbg !40
  %51 = bitcast [80 x i8]** %.Z0972_347 to i8*, !dbg !40
  %52 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !40
  %53 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !40
  %54 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !40
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %54(i8* %47, i8* %48, i8* %49, i8* %50, i8* %51, i8* null, i8* %52, i8* %53, i8* null, i64 0), !dbg !40
  %55 = load i32, i32* %allocstatus_317, align 4, !dbg !42
  call void @llvm.dbg.value(metadata i32 %55, metadata !41, metadata !DIExpression()), !dbg !10
  %56 = icmp sle i32 %55, 0, !dbg !42
  br i1 %56, label %L.LB1_390, label %L.LB1_534, !dbg !42

L.LB1_534:                                        ; preds = %L.LB1_387
  call void (...) @_mp_bcs_nest(), !dbg !43
  %57 = bitcast i32* @.C312_MAIN_ to i8*, !dbg !43
  %58 = bitcast [54 x i8]* @.C336_MAIN_ to i8*, !dbg !43
  %59 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !43
  call void (i8*, i8*, i64, ...) %59(i8* %57, i8* %58, i64 54), !dbg !43
  %60 = bitcast i32* @.C309_MAIN_ to i8*, !dbg !43
  %61 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !43
  %62 = bitcast [3 x i8]* @.C339_MAIN_ to i8*, !dbg !43
  %63 = bitcast i32 (...)* @f90io_encode_fmta to i32 (i8*, i8*, i8*, i64, ...)*, !dbg !43
  %64 = call i32 (i8*, i8*, i8*, i64, ...) %63(i8* %60, i8* %61, i8* %62, i64 3), !dbg !43
  store i32 %64, i32* %z__io_342, align 4, !dbg !43
  %65 = bitcast i32* @.C338_MAIN_ to i8*, !dbg !43
  %66 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !43
  %67 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !43
  %68 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !43
  %69 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %68(i8* %65, i8* null, i8* %66, i8* %67, i8* null, i8* null, i64 0), !dbg !43
  store i32 %69, i32* %z__io_342, align 4, !dbg !43
  %70 = bitcast i32* @.C309_MAIN_ to i8*, !dbg !43
  %71 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !43
  %72 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !43
  %73 = bitcast [37 x i8]* @.C349_MAIN_ to i8*, !dbg !43
  %74 = bitcast i32 (...)* @f90io_fmt_writea to i32 (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !43
  %75 = call i32 (i8*, i8*, i8*, i8*, i64, ...) %74(i8* %70, i8* %71, i8* %72, i8* %73, i64 37), !dbg !43
  store i32 %75, i32* %z__io_342, align 4, !dbg !43
  %76 = call i32 (...) @f90io_fmtw_end(), !dbg !43
  store i32 %76, i32* %z__io_342, align 4, !dbg !43
  call void (...) @_mp_ecs_nest(), !dbg !43
  %77 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !44
  %78 = bitcast void (...)* @f90_stop08a to void (i8*, i8*, i64, ...)*, !dbg !44
  call void (i8*, i8*, i64, ...) %78(i8* %77, i8* null, i64 0), !dbg !44
  br label %L.LB1_390

L.LB1_390:                                        ; preds = %L.LB1_534, %L.LB1_387
  %79 = load i32, i32* %argcount_316, align 4, !dbg !45
  call void @llvm.dbg.value(metadata i32 %79, metadata !35, metadata !DIExpression()), !dbg !10
  store i32 %79, i32* %.dY0001_393, align 4, !dbg !45
  call void @llvm.dbg.declare(metadata i32* %ix_319, metadata !46, metadata !DIExpression()), !dbg !10
  store i32 1, i32* %ix_319, align 4, !dbg !45
  %80 = load i32, i32* %.dY0001_393, align 4, !dbg !45
  %81 = icmp sle i32 %80, 0, !dbg !45
  br i1 %81, label %L.LB1_392, label %L.LB1_391, !dbg !45

L.LB1_391:                                        ; preds = %L.LB1_391, %L.LB1_390
  %82 = bitcast i32* %ix_319 to i8*, !dbg !47
  %83 = load [80 x i8]*, [80 x i8]** %.Z0972_347, align 8, !dbg !47
  call void @llvm.dbg.value(metadata [80 x i8]* %83, metadata !29, metadata !DIExpression()), !dbg !10
  %84 = bitcast [80 x i8]* %83 to i8*, !dbg !47
  %85 = getelementptr i8, i8* %84, i64 -80, !dbg !47
  %86 = load i32, i32* %ix_319, align 4, !dbg !47
  call void @llvm.dbg.value(metadata i32 %86, metadata !46, metadata !DIExpression()), !dbg !10
  %87 = sext i32 %86 to i64, !dbg !47
  %88 = bitcast [16 x i64]* %"args$sd1_378" to i8*, !dbg !47
  %89 = getelementptr i8, i8* %88, i64 56, !dbg !47
  %90 = bitcast i8* %89 to i64*, !dbg !47
  %91 = load i64, i64* %90, align 8, !dbg !47
  %92 = add nsw i64 %87, %91, !dbg !47
  %93 = mul nsw i64 %92, 80, !dbg !47
  %94 = getelementptr i8, i8* %85, i64 %93, !dbg !47
  %95 = bitcast i64* @.C376_MAIN_ to i8*, !dbg !47
  %96 = bitcast void (...)* @f90_get_cmd_arga to void (i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !47
  call void (i8*, i8*, i8*, i8*, i8*, i64, ...) %96(i8* %82, i8* %94, i8* null, i8* null, i8* %95, i64 80), !dbg !47
  %97 = load i32, i32* %ix_319, align 4, !dbg !48
  call void @llvm.dbg.value(metadata i32 %97, metadata !46, metadata !DIExpression()), !dbg !10
  %98 = add nsw i32 %97, 1, !dbg !48
  store i32 %98, i32* %ix_319, align 4, !dbg !48
  %99 = load i32, i32* %.dY0001_393, align 4, !dbg !48
  %100 = sub nsw i32 %99, 1, !dbg !48
  store i32 %100, i32* %.dY0001_393, align 4, !dbg !48
  %101 = load i32, i32* %.dY0001_393, align 4, !dbg !48
  %102 = icmp sgt i32 %101, 0, !dbg !48
  br i1 %102, label %L.LB1_391, label %L.LB1_392, !dbg !48

L.LB1_392:                                        ; preds = %L.LB1_391, %L.LB1_390
  %103 = load i32, i32* %argcount_316, align 4, !dbg !49
  call void @llvm.dbg.value(metadata i32 %103, metadata !35, metadata !DIExpression()), !dbg !10
  %104 = icmp sle i32 %103, 0, !dbg !49
  br i1 %104, label %L.LB1_394, label %L.LB1_535, !dbg !49

L.LB1_535:                                        ; preds = %L.LB1_392
  call void (...) @_mp_bcs_nest(), !dbg !50
  %105 = bitcast i32* @.C348_MAIN_ to i8*, !dbg !50
  %106 = bitcast [54 x i8]* @.C336_MAIN_ to i8*, !dbg !50
  %107 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !50
  call void (i8*, i8*, i64, ...) %107(i8* %105, i8* %106, i64 54), !dbg !50
  %108 = bitcast i32* @.C309_MAIN_ to i8*, !dbg !50
  %109 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !50
  %110 = bitcast [5 x i8]* @.C352_MAIN_ to i8*, !dbg !50
  %111 = bitcast i32 (...)* @f90io_encode_fmta to i32 (i8*, i8*, i8*, i64, ...)*, !dbg !50
  %112 = call i32 (i8*, i8*, i8*, i64, ...) %111(i8* %108, i8* %109, i8* %110, i64 5), !dbg !50
  store i32 %112, i32* %z__io_342, align 4, !dbg !50
  %113 = load [80 x i8]*, [80 x i8]** %.Z0972_347, align 8, !dbg !50
  call void @llvm.dbg.value(metadata [80 x i8]* %113, metadata !29, metadata !DIExpression()), !dbg !10
  %114 = bitcast [80 x i8]* %113 to i8*, !dbg !50
  %115 = bitcast [16 x i64]* %"args$sd1_378" to i8*, !dbg !50
  %116 = getelementptr i8, i8* %115, i64 56, !dbg !50
  %117 = bitcast i8* %116 to i64*, !dbg !50
  %118 = load i64, i64* %117, align 8, !dbg !50
  %119 = mul nsw i64 %118, 80, !dbg !50
  %120 = getelementptr i8, i8* %114, i64 %119, !dbg !50
  %121 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !50
  %122 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !50
  call void @llvm.dbg.declare(metadata i32* %rderr_318, metadata !51, metadata !DIExpression()), !dbg !10
  %123 = bitcast i32* %rderr_318 to i8*, !dbg !50
  %124 = bitcast i32 (...)* @f90io_fmtr_intern_inita to i32 (i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !50
  %125 = call i32 (i8*, i8*, i8*, i8*, i8*, i64, ...) %124(i8* %120, i8* %121, i8* %122, i8* %123, i8* null, i64 80), !dbg !50
  store i32 %125, i32* %z__io_342, align 4, !dbg !50
  %126 = bitcast i32* @.C310_MAIN_ to i8*, !dbg !50
  %127 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !50
  %128 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !50
  %129 = bitcast i32* %len_333 to i8*, !dbg !50
  %130 = bitcast i32 (...)* @f90io_fmt_reada to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !50
  %131 = call i32 (i8*, i8*, i8*, i8*, ...) %130(i8* %126, i8* %127, i8* %128, i8* %129), !dbg !50
  store i32 %131, i32* %z__io_342, align 4, !dbg !50
  %132 = call i32 (...) @f90io_fmtr_end(), !dbg !50
  store i32 %132, i32* %z__io_342, align 4, !dbg !50
  call void (...) @_mp_ecs_nest(), !dbg !50
  %133 = load i32, i32* %rderr_318, align 4, !dbg !52
  call void @llvm.dbg.value(metadata i32 %133, metadata !51, metadata !DIExpression()), !dbg !10
  %134 = icmp eq i32 %133, 0, !dbg !52
  br i1 %134, label %L.LB1_395, label %L.LB1_536, !dbg !52

L.LB1_536:                                        ; preds = %L.LB1_535
  call void (...) @_mp_bcs_nest(), !dbg !53
  %135 = bitcast i32* @.C356_MAIN_ to i8*, !dbg !53
  %136 = bitcast [54 x i8]* @.C336_MAIN_ to i8*, !dbg !53
  %137 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !53
  call void (i8*, i8*, i64, ...) %137(i8* %135, i8* %136, i64 54), !dbg !53
  %138 = bitcast i32* @.C309_MAIN_ to i8*, !dbg !53
  %139 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !53
  %140 = bitcast [3 x i8]* @.C339_MAIN_ to i8*, !dbg !53
  %141 = bitcast i32 (...)* @f90io_encode_fmta to i32 (i8*, i8*, i8*, i64, ...)*, !dbg !53
  %142 = call i32 (i8*, i8*, i8*, i64, ...) %141(i8* %138, i8* %139, i8* %140, i64 3), !dbg !53
  store i32 %142, i32* %z__io_342, align 4, !dbg !53
  %143 = bitcast i32* @.C338_MAIN_ to i8*, !dbg !53
  %144 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !53
  %145 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !53
  %146 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !53
  %147 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %146(i8* %143, i8* null, i8* %144, i8* %145, i8* null, i8* null, i64 0), !dbg !53
  store i32 %147, i32* %z__io_342, align 4, !dbg !53
  %148 = bitcast i32* @.C309_MAIN_ to i8*, !dbg !53
  %149 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !53
  %150 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !53
  %151 = bitcast [29 x i8]* @.C357_MAIN_ to i8*, !dbg !53
  %152 = bitcast i32 (...)* @f90io_fmt_writea to i32 (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !53
  %153 = call i32 (i8*, i8*, i8*, i8*, i64, ...) %152(i8* %148, i8* %149, i8* %150, i8* %151, i64 29), !dbg !53
  store i32 %153, i32* %z__io_342, align 4, !dbg !53
  %154 = call i32 (...) @f90io_fmtw_end(), !dbg !53
  store i32 %154, i32* %z__io_342, align 4, !dbg !53
  call void (...) @_mp_ecs_nest(), !dbg !53
  br label %L.LB1_395

L.LB1_395:                                        ; preds = %L.LB1_536, %L.LB1_535
  br label %L.LB1_394

L.LB1_394:                                        ; preds = %L.LB1_395, %L.LB1_392
  call void @llvm.dbg.declare(metadata i64* %z_b_4_326, metadata !39, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_4_326, align 8, !dbg !54
  %155 = load i32, i32* %len_333, align 4, !dbg !54
  call void @llvm.dbg.value(metadata i32 %155, metadata !32, metadata !DIExpression()), !dbg !10
  %156 = sext i32 %155 to i64, !dbg !54
  call void @llvm.dbg.declare(metadata i64* %z_b_5_327, metadata !39, metadata !DIExpression()), !dbg !10
  store i64 %156, i64* %z_b_5_327, align 8, !dbg !54
  %157 = load i64, i64* %z_b_5_327, align 8, !dbg !54
  call void @llvm.dbg.value(metadata i64 %157, metadata !39, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_68_330, metadata !39, metadata !DIExpression()), !dbg !10
  store i64 %157, i64* %z_e_68_330, align 8, !dbg !54
  %158 = bitcast [16 x i64]* %"x$sd2_382" to i8*, !dbg !54
  %159 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !54
  %160 = bitcast i64* @.C383_MAIN_ to i8*, !dbg !54
  %161 = bitcast i64* @.C376_MAIN_ to i8*, !dbg !54
  %162 = bitcast i64* %z_b_4_326 to i8*, !dbg !54
  %163 = bitcast i64* %z_b_5_327 to i8*, !dbg !54
  %164 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !54
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %164(i8* %158, i8* %159, i8* %160, i8* %161, i8* %162, i8* %163), !dbg !54
  %165 = bitcast [16 x i64]* %"x$sd2_382" to i8*, !dbg !54
  %166 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !54
  call void (i8*, i32, ...) %166(i8* %165, i32 25), !dbg !54
  %167 = load i64, i64* %z_b_5_327, align 8, !dbg !54
  call void @llvm.dbg.value(metadata i64 %167, metadata !39, metadata !DIExpression()), !dbg !10
  %168 = load i64, i64* %z_b_4_326, align 8, !dbg !54
  call void @llvm.dbg.value(metadata i64 %168, metadata !39, metadata !DIExpression()), !dbg !10
  %169 = sub nsw i64 %168, 1, !dbg !54
  %170 = sub nsw i64 %167, %169, !dbg !54
  call void @llvm.dbg.declare(metadata i64* %z_b_6_328, metadata !39, metadata !DIExpression()), !dbg !10
  store i64 %170, i64* %z_b_6_328, align 8, !dbg !54
  %171 = load i64, i64* %z_b_4_326, align 8, !dbg !54
  call void @llvm.dbg.value(metadata i64 %171, metadata !39, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_b_7_329, metadata !39, metadata !DIExpression()), !dbg !10
  store i64 %171, i64* %z_b_7_329, align 8, !dbg !54
  %172 = bitcast i64* %z_b_6_328 to i8*, !dbg !54
  %173 = bitcast i64* @.C383_MAIN_ to i8*, !dbg !54
  %174 = bitcast i64* @.C376_MAIN_ to i8*, !dbg !54
  %175 = bitcast i32** %.Z0978_358 to i8*, !dbg !54
  %176 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !54
  %177 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !54
  %178 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !54
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %178(i8* %172, i8* %173, i8* %174, i8* null, i8* %175, i8* null, i8* %176, i8* %177, i8* null, i64 0), !dbg !54
  %179 = load i32, i32* %len_333, align 4, !dbg !55
  call void @llvm.dbg.value(metadata i32 %179, metadata !32, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i32* %numnodes_314, metadata !56, metadata !DIExpression()), !dbg !10
  store i32 %179, i32* %numnodes_314, align 4, !dbg !55
  call void @llvm.dbg.declare(metadata i32* %numnodes2_315, metadata !57, metadata !DIExpression()), !dbg !10
  store i32 0, i32* %numnodes2_315, align 4, !dbg !58
  %180 = load i32, i32* %len_333, align 4, !dbg !59
  call void @llvm.dbg.value(metadata i32 %180, metadata !32, metadata !DIExpression()), !dbg !10
  store i32 %180, i32* %.dY0002_398, align 4, !dbg !59
  call void @llvm.dbg.declare(metadata i32* %i_313, metadata !60, metadata !DIExpression()), !dbg !10
  store i32 1, i32* %i_313, align 4, !dbg !59
  %181 = load i32, i32* %.dY0002_398, align 4, !dbg !59
  %182 = icmp sle i32 %181, 0, !dbg !59
  br i1 %182, label %L.LB1_397, label %L.LB1_396, !dbg !59

L.LB1_396:                                        ; preds = %L.LB1_400, %L.LB1_394
  %183 = load i32, i32* %i_313, align 4, !dbg !61
  call void @llvm.dbg.value(metadata i32 %183, metadata !60, metadata !DIExpression()), !dbg !10
  %184 = load i32, i32* %i_313, align 4, !dbg !61
  call void @llvm.dbg.value(metadata i32 %184, metadata !60, metadata !DIExpression()), !dbg !10
  %185 = load i32, i32* %i_313, align 4, !dbg !61
  call void @llvm.dbg.value(metadata i32 %185, metadata !60, metadata !DIExpression()), !dbg !10
  %186 = lshr i32 %185, 31, !dbg !61
  %187 = add nsw i32 %184, %186, !dbg !61
  %188 = ashr i32 %187, 1, !dbg !61
  %189 = mul nsw i32 %188, 2, !dbg !61
  %190 = icmp ne i32 %183, %189, !dbg !61
  br i1 %190, label %L.LB1_399, label %L.LB1_537, !dbg !61

L.LB1_537:                                        ; preds = %L.LB1_396
  %191 = load i32, i32* %i_313, align 4, !dbg !62
  call void @llvm.dbg.value(metadata i32 %191, metadata !60, metadata !DIExpression()), !dbg !10
  %192 = sext i32 %191 to i64, !dbg !62
  %193 = bitcast [16 x i64]* %"x$sd2_382" to i8*, !dbg !62
  %194 = getelementptr i8, i8* %193, i64 56, !dbg !62
  %195 = bitcast i8* %194 to i64*, !dbg !62
  %196 = load i64, i64* %195, align 8, !dbg !62
  %197 = add nsw i64 %192, %196, !dbg !62
  %198 = load i32*, i32** %.Z0978_358, align 8, !dbg !62
  call void @llvm.dbg.value(metadata i32* %198, metadata !20, metadata !DIExpression()), !dbg !10
  %199 = bitcast i32* %198 to i8*, !dbg !62
  %200 = getelementptr i8, i8* %199, i64 -4, !dbg !62
  %201 = bitcast i8* %200 to i32*, !dbg !62
  %202 = getelementptr i32, i32* %201, i64 %197, !dbg !62
  store i32 5, i32* %202, align 4, !dbg !62
  br label %L.LB1_400, !dbg !63

L.LB1_399:                                        ; preds = %L.LB1_396
  %203 = load i32, i32* %i_313, align 4, !dbg !64
  call void @llvm.dbg.value(metadata i32 %203, metadata !60, metadata !DIExpression()), !dbg !10
  %204 = sext i32 %203 to i64, !dbg !64
  %205 = bitcast [16 x i64]* %"x$sd2_382" to i8*, !dbg !64
  %206 = getelementptr i8, i8* %205, i64 56, !dbg !64
  %207 = bitcast i8* %206 to i64*, !dbg !64
  %208 = load i64, i64* %207, align 8, !dbg !64
  %209 = add nsw i64 %204, %208, !dbg !64
  %210 = load i32*, i32** %.Z0978_358, align 8, !dbg !64
  call void @llvm.dbg.value(metadata i32* %210, metadata !20, metadata !DIExpression()), !dbg !10
  %211 = bitcast i32* %210 to i8*, !dbg !64
  %212 = getelementptr i8, i8* %211, i64 -4, !dbg !64
  %213 = bitcast i8* %212 to i32*, !dbg !64
  %214 = getelementptr i32, i32* %213, i64 %209, !dbg !64
  store i32 -5, i32* %214, align 4, !dbg !64
  br label %L.LB1_400

L.LB1_400:                                        ; preds = %L.LB1_399, %L.LB1_537
  %215 = load i32, i32* %i_313, align 4, !dbg !65
  call void @llvm.dbg.value(metadata i32 %215, metadata !60, metadata !DIExpression()), !dbg !10
  %216 = add nsw i32 %215, 1, !dbg !65
  store i32 %216, i32* %i_313, align 4, !dbg !65
  %217 = load i32, i32* %.dY0002_398, align 4, !dbg !65
  %218 = sub nsw i32 %217, 1, !dbg !65
  store i32 %218, i32* %.dY0002_398, align 4, !dbg !65
  %219 = load i32, i32* %.dY0002_398, align 4, !dbg !65
  %220 = icmp sgt i32 %219, 0, !dbg !65
  br i1 %220, label %L.LB1_396, label %L.LB1_397, !dbg !65

L.LB1_397:                                        ; preds = %L.LB1_400, %L.LB1_394
  %221 = bitcast i32* %numnodes_314 to i8*, !dbg !66
  %222 = bitcast %astruct.dt88* %.uplevelArgPack0001_488 to i8**, !dbg !66
  store i8* %221, i8** %222, align 8, !dbg !66
  %223 = bitcast i32** %.Z0978_358 to i8*, !dbg !66
  %224 = bitcast %astruct.dt88* %.uplevelArgPack0001_488 to i8*, !dbg !66
  %225 = getelementptr i8, i8* %224, i64 8, !dbg !66
  %226 = bitcast i8* %225 to i8**, !dbg !66
  store i8* %223, i8** %226, align 8, !dbg !66
  %227 = bitcast i32** %.Z0978_358 to i8*, !dbg !66
  %228 = bitcast %astruct.dt88* %.uplevelArgPack0001_488 to i8*, !dbg !66
  %229 = getelementptr i8, i8* %228, i64 16, !dbg !66
  %230 = bitcast i8* %229 to i8**, !dbg !66
  store i8* %227, i8** %230, align 8, !dbg !66
  %231 = bitcast i64* %z_b_4_326 to i8*, !dbg !66
  %232 = bitcast %astruct.dt88* %.uplevelArgPack0001_488 to i8*, !dbg !66
  %233 = getelementptr i8, i8* %232, i64 24, !dbg !66
  %234 = bitcast i8* %233 to i8**, !dbg !66
  store i8* %231, i8** %234, align 8, !dbg !66
  %235 = bitcast i64* %z_b_5_327 to i8*, !dbg !66
  %236 = bitcast %astruct.dt88* %.uplevelArgPack0001_488 to i8*, !dbg !66
  %237 = getelementptr i8, i8* %236, i64 32, !dbg !66
  %238 = bitcast i8* %237 to i8**, !dbg !66
  store i8* %235, i8** %238, align 8, !dbg !66
  %239 = bitcast i64* %z_e_68_330 to i8*, !dbg !66
  %240 = bitcast %astruct.dt88* %.uplevelArgPack0001_488 to i8*, !dbg !66
  %241 = getelementptr i8, i8* %240, i64 40, !dbg !66
  %242 = bitcast i8* %241 to i8**, !dbg !66
  store i8* %239, i8** %242, align 8, !dbg !66
  %243 = bitcast i64* %z_b_6_328 to i8*, !dbg !66
  %244 = bitcast %astruct.dt88* %.uplevelArgPack0001_488 to i8*, !dbg !66
  %245 = getelementptr i8, i8* %244, i64 48, !dbg !66
  %246 = bitcast i8* %245 to i8**, !dbg !66
  store i8* %243, i8** %246, align 8, !dbg !66
  %247 = bitcast i64* %z_b_7_329 to i8*, !dbg !66
  %248 = bitcast %astruct.dt88* %.uplevelArgPack0001_488 to i8*, !dbg !66
  %249 = getelementptr i8, i8* %248, i64 56, !dbg !66
  %250 = bitcast i8* %249 to i8**, !dbg !66
  store i8* %247, i8** %250, align 8, !dbg !66
  %251 = bitcast i32* %numnodes2_315 to i8*, !dbg !66
  %252 = bitcast %astruct.dt88* %.uplevelArgPack0001_488 to i8*, !dbg !66
  %253 = getelementptr i8, i8* %252, i64 64, !dbg !66
  %254 = bitcast i8* %253 to i8**, !dbg !66
  store i8* %251, i8** %254, align 8, !dbg !66
  %255 = bitcast [16 x i64]* %"x$sd2_382" to i8*, !dbg !66
  %256 = bitcast %astruct.dt88* %.uplevelArgPack0001_488 to i8*, !dbg !66
  %257 = getelementptr i8, i8* %256, i64 72, !dbg !66
  %258 = bitcast i8* %257 to i8**, !dbg !66
  store i8* %255, i8** %258, align 8, !dbg !66
  br label %L.LB1_509, !dbg !66

L.LB1_509:                                        ; preds = %L.LB1_397
  %259 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L56_1_ to i64*, !dbg !66
  %260 = bitcast %astruct.dt88* %.uplevelArgPack0001_488 to i64*, !dbg !66
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %259, i64* %260), !dbg !66
  call void (...) @_mp_bcs_nest(), !dbg !67
  %261 = bitcast i32* @.C367_MAIN_ to i8*, !dbg !67
  %262 = bitcast [54 x i8]* @.C336_MAIN_ to i8*, !dbg !67
  %263 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !67
  call void (i8*, i8*, i64, ...) %263(i8* %261, i8* %262, i64 54), !dbg !67
  %264 = bitcast i32* @.C338_MAIN_ to i8*, !dbg !67
  %265 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !67
  %266 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !67
  %267 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !67
  %268 = call i32 (i8*, i8*, i8*, i8*, ...) %267(i8* %264, i8* null, i8* %265, i8* %266), !dbg !67
  store i32 %268, i32* %z__io_342, align 4, !dbg !67
  %269 = bitcast [11 x i8]* @.C369_MAIN_ to i8*, !dbg !67
  %270 = bitcast i32 (...)* @f90io_sc_ch_ldw to i32 (i8*, i32, i64, ...)*, !dbg !67
  %271 = call i32 (i8*, i32, i64, ...) %270(i8* %269, i32 14, i64 11), !dbg !67
  store i32 %271, i32* %z__io_342, align 4, !dbg !67
  %272 = load i32, i32* %numnodes2_315, align 4, !dbg !67
  call void @llvm.dbg.value(metadata i32 %272, metadata !57, metadata !DIExpression()), !dbg !10
  %273 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !67
  %274 = call i32 (i32, i32, ...) %273(i32 %272, i32 25), !dbg !67
  store i32 %274, i32* %z__io_342, align 4, !dbg !67
  %275 = call i32 (...) @f90io_ldw_end(), !dbg !67
  store i32 %275, i32* %z__io_342, align 4, !dbg !67
  call void (...) @_mp_ecs_nest(), !dbg !67
  %276 = load [80 x i8]*, [80 x i8]** %.Z0972_347, align 8, !dbg !68
  call void @llvm.dbg.value(metadata [80 x i8]* %276, metadata !29, metadata !DIExpression()), !dbg !10
  %277 = bitcast [80 x i8]* %276 to i8*, !dbg !68
  %278 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !68
  %279 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, i64, ...)*, !dbg !68
  call void (i8*, i8*, i8*, i8*, i64, i64, ...) %279(i8* null, i8* %277, i8* %278, i8* null, i64 80, i64 0), !dbg !68
  %280 = bitcast [80 x i8]** %.Z0972_347 to i8**, !dbg !68
  store i8* null, i8** %280, align 8, !dbg !68
  %281 = bitcast [16 x i64]* %"args$sd1_378" to i64*, !dbg !68
  store i64 0, i64* %281, align 8, !dbg !68
  %282 = load i32*, i32** %.Z0978_358, align 8, !dbg !68
  call void @llvm.dbg.value(metadata i32* %282, metadata !20, metadata !DIExpression()), !dbg !10
  %283 = bitcast i32* %282 to i8*, !dbg !68
  %284 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !68
  %285 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !68
  call void (i8*, i8*, i8*, i8*, i64, ...) %285(i8* null, i8* %283, i8* %284, i8* null, i64 0), !dbg !68
  %286 = bitcast i32** %.Z0978_358 to i8**, !dbg !68
  store i8* null, i8** %286, align 8, !dbg !68
  %287 = bitcast [16 x i64]* %"x$sd2_382" to i64*, !dbg !68
  store i64 0, i64* %287, align 8, !dbg !68
  ret void, !dbg !18
}

define internal void @__nv_MAIN__F1L56_1_(i32* %__nv_MAIN__F1L56_1Arg0, i64* %__nv_MAIN__F1L56_1Arg1, i64* %__nv_MAIN__F1L56_1Arg2) #0 !dbg !69 {
L.entry:
  %__gtid___nv_MAIN__F1L56_1__556 = alloca i32, align 4
  %.i0000p_365 = alloca i32, align 4
  %i_364 = alloca i32, align 4
  %.du0003p_404 = alloca i32, align 4
  %.de0003p_405 = alloca i32, align 4
  %.di0003p_406 = alloca i32, align 4
  %.ds0003p_407 = alloca i32, align 4
  %.dl0003p_409 = alloca i32, align 4
  %.dl0003p.copy_550 = alloca i32, align 4
  %.de0003p.copy_551 = alloca i32, align 4
  %.ds0003p.copy_552 = alloca i32, align 4
  %.dX0003p_408 = alloca i32, align 4
  %.dY0003p_403 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L56_1Arg0, metadata !72, metadata !DIExpression()), !dbg !73
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L56_1Arg1, metadata !74, metadata !DIExpression()), !dbg !73
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L56_1Arg2, metadata !75, metadata !DIExpression()), !dbg !73
  call void @llvm.dbg.value(metadata i32 1, metadata !76, metadata !DIExpression()), !dbg !73
  call void @llvm.dbg.value(metadata i32 2, metadata !77, metadata !DIExpression()), !dbg !73
  call void @llvm.dbg.value(metadata i32 0, metadata !78, metadata !DIExpression()), !dbg !73
  call void @llvm.dbg.value(metadata i32 1, metadata !79, metadata !DIExpression()), !dbg !73
  call void @llvm.dbg.value(metadata i32 2, metadata !80, metadata !DIExpression()), !dbg !73
  call void @llvm.dbg.value(metadata i32 0, metadata !81, metadata !DIExpression()), !dbg !73
  call void @llvm.dbg.value(metadata i32 1, metadata !82, metadata !DIExpression()), !dbg !73
  call void @llvm.dbg.value(metadata i32 2, metadata !83, metadata !DIExpression()), !dbg !73
  %0 = load i32, i32* %__nv_MAIN__F1L56_1Arg0, align 4, !dbg !84
  store i32 %0, i32* %__gtid___nv_MAIN__F1L56_1__556, align 4, !dbg !84
  br label %L.LB2_541

L.LB2_541:                                        ; preds = %L.entry
  br label %L.LB2_362

L.LB2_362:                                        ; preds = %L.LB2_541
  store i32 0, i32* %.i0000p_365, align 4, !dbg !85
  %1 = bitcast i64* %__nv_MAIN__F1L56_1Arg2 to i32**, !dbg !85
  %2 = load i32*, i32** %1, align 8, !dbg !85
  %3 = load i32, i32* %2, align 4, !dbg !85
  call void @llvm.dbg.declare(metadata i32* %i_364, metadata !86, metadata !DIExpression()), !dbg !84
  store i32 %3, i32* %i_364, align 4, !dbg !85
  store i32 1, i32* %.du0003p_404, align 4, !dbg !85
  store i32 1, i32* %.de0003p_405, align 4, !dbg !85
  store i32 -1, i32* %.di0003p_406, align 4, !dbg !85
  %4 = load i32, i32* %.di0003p_406, align 4, !dbg !85
  store i32 %4, i32* %.ds0003p_407, align 4, !dbg !85
  %5 = bitcast i64* %__nv_MAIN__F1L56_1Arg2 to i32**, !dbg !85
  %6 = load i32*, i32** %5, align 8, !dbg !85
  %7 = load i32, i32* %6, align 4, !dbg !85
  store i32 %7, i32* %.dl0003p_409, align 4, !dbg !85
  %8 = load i32, i32* %.dl0003p_409, align 4, !dbg !85
  store i32 %8, i32* %.dl0003p.copy_550, align 4, !dbg !85
  %9 = load i32, i32* %.de0003p_405, align 4, !dbg !85
  store i32 %9, i32* %.de0003p.copy_551, align 4, !dbg !85
  %10 = load i32, i32* %.ds0003p_407, align 4, !dbg !85
  store i32 %10, i32* %.ds0003p.copy_552, align 4, !dbg !85
  %11 = load i32, i32* %__gtid___nv_MAIN__F1L56_1__556, align 4, !dbg !85
  %12 = bitcast i32* %.i0000p_365 to i64*, !dbg !85
  %13 = bitcast i32* %.dl0003p.copy_550 to i64*, !dbg !85
  %14 = bitcast i32* %.de0003p.copy_551 to i64*, !dbg !85
  %15 = bitcast i32* %.ds0003p.copy_552 to i64*, !dbg !85
  %16 = load i32, i32* %.ds0003p.copy_552, align 4, !dbg !85
  call void @__kmpc_for_static_init_4(i64* null, i32 %11, i32 34, i64* %12, i64* %13, i64* %14, i64* %15, i32 %16, i32 1), !dbg !85
  %17 = load i32, i32* %.dl0003p.copy_550, align 4, !dbg !85
  store i32 %17, i32* %.dl0003p_409, align 4, !dbg !85
  %18 = load i32, i32* %.de0003p.copy_551, align 4, !dbg !85
  store i32 %18, i32* %.de0003p_405, align 4, !dbg !85
  %19 = load i32, i32* %.ds0003p.copy_552, align 4, !dbg !85
  store i32 %19, i32* %.ds0003p_407, align 4, !dbg !85
  %20 = load i32, i32* %.dl0003p_409, align 4, !dbg !85
  store i32 %20, i32* %i_364, align 4, !dbg !85
  %21 = load i32, i32* %i_364, align 4, !dbg !85
  call void @llvm.dbg.value(metadata i32 %21, metadata !86, metadata !DIExpression()), !dbg !84
  store i32 %21, i32* %.dX0003p_408, align 4, !dbg !85
  %22 = load i32, i32* %.dX0003p_408, align 4, !dbg !85
  %23 = load i32, i32* %.du0003p_404, align 4, !dbg !85
  %24 = icmp slt i32 %22, %23, !dbg !85
  br i1 %24, label %L.LB2_402, label %L.LB2_581, !dbg !85

L.LB2_581:                                        ; preds = %L.LB2_362
  %25 = load i32, i32* %.dX0003p_408, align 4, !dbg !85
  store i32 %25, i32* %i_364, align 4, !dbg !85
  %26 = load i32, i32* %.di0003p_406, align 4, !dbg !85
  %27 = load i32, i32* %.de0003p_405, align 4, !dbg !85
  %28 = load i32, i32* %.dX0003p_408, align 4, !dbg !85
  %29 = sub nsw i32 %27, %28, !dbg !85
  %30 = add nsw i32 %26, %29, !dbg !85
  %31 = load i32, i32* %.di0003p_406, align 4, !dbg !85
  %32 = sdiv i32 %30, %31, !dbg !85
  store i32 %32, i32* %.dY0003p_403, align 4, !dbg !85
  %33 = load i32, i32* %.dY0003p_403, align 4, !dbg !85
  %34 = icmp sle i32 %33, 0, !dbg !85
  br i1 %34, label %L.LB2_412, label %L.LB2_411, !dbg !85

L.LB2_411:                                        ; preds = %L.LB2_413, %L.LB2_581
  %35 = load i32, i32* %i_364, align 4, !dbg !87
  call void @llvm.dbg.value(metadata i32 %35, metadata !86, metadata !DIExpression()), !dbg !84
  %36 = sext i32 %35 to i64, !dbg !87
  %37 = bitcast i64* %__nv_MAIN__F1L56_1Arg2 to i8*, !dbg !87
  %38 = getelementptr i8, i8* %37, i64 72, !dbg !87
  %39 = bitcast i8* %38 to i8**, !dbg !87
  %40 = load i8*, i8** %39, align 8, !dbg !87
  %41 = getelementptr i8, i8* %40, i64 56, !dbg !87
  %42 = bitcast i8* %41 to i64*, !dbg !87
  %43 = load i64, i64* %42, align 8, !dbg !87
  %44 = add nsw i64 %36, %43, !dbg !87
  %45 = bitcast i64* %__nv_MAIN__F1L56_1Arg2 to i8*, !dbg !87
  %46 = getelementptr i8, i8* %45, i64 16, !dbg !87
  %47 = bitcast i8* %46 to i8***, !dbg !87
  %48 = load i8**, i8*** %47, align 8, !dbg !87
  %49 = load i8*, i8** %48, align 8, !dbg !87
  %50 = getelementptr i8, i8* %49, i64 -4, !dbg !87
  %51 = bitcast i8* %50 to i32*, !dbg !87
  %52 = getelementptr i32, i32* %51, i64 %44, !dbg !87
  %53 = load i32, i32* %52, align 4, !dbg !87
  %54 = icmp sgt i32 %53, 0, !dbg !87
  br i1 %54, label %L.LB2_413, label %L.LB2_582, !dbg !87

L.LB2_582:                                        ; preds = %L.LB2_411
  %55 = bitcast i64* %__nv_MAIN__F1L56_1Arg2 to i8*, !dbg !88
  %56 = getelementptr i8, i8* %55, i64 64, !dbg !88
  %57 = bitcast i8* %56 to i32**, !dbg !88
  %58 = load i32*, i32** %57, align 8, !dbg !88
  %59 = load i32, i32* %58, align 4, !dbg !88
  %60 = sub nsw i32 %59, 1, !dbg !88
  %61 = bitcast i64* %__nv_MAIN__F1L56_1Arg2 to i8*, !dbg !88
  %62 = getelementptr i8, i8* %61, i64 64, !dbg !88
  %63 = bitcast i8* %62 to i32**, !dbg !88
  %64 = load i32*, i32** %63, align 8, !dbg !88
  store i32 %60, i32* %64, align 4, !dbg !88
  br label %L.LB2_413

L.LB2_413:                                        ; preds = %L.LB2_582, %L.LB2_411
  %65 = load i32, i32* %.di0003p_406, align 4, !dbg !84
  %66 = load i32, i32* %i_364, align 4, !dbg !84
  call void @llvm.dbg.value(metadata i32 %66, metadata !86, metadata !DIExpression()), !dbg !84
  %67 = add nsw i32 %65, %66, !dbg !84
  store i32 %67, i32* %i_364, align 4, !dbg !84
  %68 = load i32, i32* %.dY0003p_403, align 4, !dbg !84
  %69 = sub nsw i32 %68, 1, !dbg !84
  store i32 %69, i32* %.dY0003p_403, align 4, !dbg !84
  %70 = load i32, i32* %.dY0003p_403, align 4, !dbg !84
  %71 = icmp sgt i32 %70, 0, !dbg !84
  br i1 %71, label %L.LB2_411, label %L.LB2_412, !dbg !84

L.LB2_412:                                        ; preds = %L.LB2_413, %L.LB2_581
  br label %L.LB2_402

L.LB2_402:                                        ; preds = %L.LB2_412, %L.LB2_362
  %72 = load i32, i32* %__gtid___nv_MAIN__F1L56_1__556, align 4, !dbg !84
  call void @__kmpc_for_static_fini(i64* null, i32 %72), !dbg !84
  br label %L.LB2_366

L.LB2_366:                                        ; preds = %L.LB2_402
  ret void, !dbg !84
}

declare void @__kmpc_for_static_fini(i64*, i32) #0

declare void @__kmpc_for_static_init_4(i64*, i32, i32, i64*, i64*, i64*, i64*, i32, i32) #0

declare void @f90_dealloc03a_i8(...) #0

declare signext i32 @f90io_ldw_end(...) #0

declare signext i32 @f90io_sc_i_ldw(...) #0

declare signext i32 @f90io_sc_ch_ldw(...) #0

declare signext i32 @f90io_print_init(...) #0

declare signext i32 @f90io_fmtr_end(...) #0

declare signext i32 @f90io_fmt_reada(...) #0

declare signext i32 @f90io_fmtr_intern_inita(...) #0

declare void @f90_get_cmd_arga(...) #0

declare void @f90_stop08a(...) #0

declare void @f90_alloc04_chka_i8(...) #0

declare void @f90_set_intrin_type_i8(...) #0

declare void @f90_template1_i8(...) #0

declare void @_mp_ecs_nest(...) #0

declare signext i32 @f90io_fmtw_end(...) #0

declare signext i32 @f90io_fmt_writea(...) #0

declare signext i32 @f90io_fmtw_inita(...) #0

declare signext i32 @f90io_encode_fmta(...) #0

declare void @f90io_src_info03a(...) #0

declare void @_mp_bcs_nest(...) #0

declare signext i32 @f90_cmd_arg_cnt(...) #0

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
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB012-minusminus-var-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb012_minusminus_var_yes", scope: !2, file: !3, line: 11, type: !6, scopeLine: 11, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!6 = !DISubroutineType(cc: DW_CC_program, types: !7)
!7 = !{null}
!8 = !DILocalVariable(name: "omp_sched_static", scope: !5, file: !3, type: !9)
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 0, scope: !5)
!11 = !DILocalVariable(name: "omp_sched_dynamic", scope: !5, file: !3, type: !9)
!12 = !DILocalVariable(name: "omp_proc_bind_false", scope: !5, file: !3, type: !9)
!13 = !DILocalVariable(name: "omp_proc_bind_true", scope: !5, file: !3, type: !9)
!14 = !DILocalVariable(name: "omp_proc_bind_master", scope: !5, file: !3, type: !9)
!15 = !DILocalVariable(name: "omp_lock_hint_none", scope: !5, file: !3, type: !9)
!16 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !5, file: !3, type: !9)
!17 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !5, file: !3, type: !9)
!18 = !DILocation(line: 67, column: 1, scope: !5)
!19 = !DILocation(line: 11, column: 1, scope: !5)
!20 = !DILocalVariable(name: "x", scope: !5, file: !3, type: !21)
!21 = !DICompositeType(tag: DW_TAG_array_type, baseType: !9, size: 32, align: 32, elements: !22)
!22 = !{!23}
!23 = !DISubrange(count: 0, lowerBound: 1)
!24 = !DILocalVariable(scope: !5, file: !3, type: !25, flags: DIFlagArtificial)
!25 = !DICompositeType(tag: DW_TAG_array_type, baseType: !26, size: 1024, align: 64, elements: !27)
!26 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!27 = !{!28}
!28 = !DISubrange(count: 16, lowerBound: 1)
!29 = !DILocalVariable(name: "args", scope: !5, file: !3, type: !30)
!30 = !DICompositeType(tag: DW_TAG_array_type, baseType: !31, size: 640, align: 8, elements: !22)
!31 = !DIBasicType(name: "character", size: 640, align: 8, encoding: DW_ATE_signed)
!32 = !DILocalVariable(name: "len", scope: !5, file: !3, type: !9)
!33 = !DILocation(line: 19, column: 1, scope: !5)
!34 = !DILocation(line: 21, column: 1, scope: !5)
!35 = !DILocalVariable(name: "argcount", scope: !5, file: !3, type: !9)
!36 = !DILocation(line: 22, column: 1, scope: !5)
!37 = !DILocation(line: 23, column: 1, scope: !5)
!38 = !DILocalVariable(scope: !5, file: !3, type: !9, flags: DIFlagArtificial)
!39 = !DILocalVariable(scope: !5, file: !3, type: !26, flags: DIFlagArtificial)
!40 = !DILocation(line: 26, column: 1, scope: !5)
!41 = !DILocalVariable(name: "allocstatus", scope: !5, file: !3, type: !9)
!42 = !DILocation(line: 27, column: 1, scope: !5)
!43 = !DILocation(line: 28, column: 1, scope: !5)
!44 = !DILocation(line: 29, column: 1, scope: !5)
!45 = !DILocation(line: 32, column: 1, scope: !5)
!46 = !DILocalVariable(name: "ix", scope: !5, file: !3, type: !9)
!47 = !DILocation(line: 33, column: 1, scope: !5)
!48 = !DILocation(line: 34, column: 1, scope: !5)
!49 = !DILocation(line: 36, column: 1, scope: !5)
!50 = !DILocation(line: 37, column: 1, scope: !5)
!51 = !DILocalVariable(name: "rderr", scope: !5, file: !3, type: !9)
!52 = !DILocation(line: 38, column: 1, scope: !5)
!53 = !DILocation(line: 39, column: 1, scope: !5)
!54 = !DILocation(line: 43, column: 1, scope: !5)
!55 = !DILocation(line: 45, column: 1, scope: !5)
!56 = !DILocalVariable(name: "numnodes", scope: !5, file: !3, type: !9)
!57 = !DILocalVariable(name: "numnodes2", scope: !5, file: !3, type: !9)
!58 = !DILocation(line: 46, column: 1, scope: !5)
!59 = !DILocation(line: 48, column: 1, scope: !5)
!60 = !DILocalVariable(name: "i", scope: !5, file: !3, type: !9)
!61 = !DILocation(line: 49, column: 1, scope: !5)
!62 = !DILocation(line: 50, column: 1, scope: !5)
!63 = !DILocation(line: 51, column: 1, scope: !5)
!64 = !DILocation(line: 52, column: 1, scope: !5)
!65 = !DILocation(line: 54, column: 1, scope: !5)
!66 = !DILocation(line: 56, column: 1, scope: !5)
!67 = !DILocation(line: 64, column: 1, scope: !5)
!68 = !DILocation(line: 66, column: 1, scope: !5)
!69 = distinct !DISubprogram(name: "__nv_MAIN__F1L56_1", scope: !2, file: !3, line: 56, type: !70, scopeLine: 56, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!70 = !DISubroutineType(types: !71)
!71 = !{null, !9, !26, !26}
!72 = !DILocalVariable(name: "__nv_MAIN__F1L56_1Arg0", arg: 1, scope: !69, file: !3, type: !9)
!73 = !DILocation(line: 0, scope: !69)
!74 = !DILocalVariable(name: "__nv_MAIN__F1L56_1Arg1", arg: 2, scope: !69, file: !3, type: !26)
!75 = !DILocalVariable(name: "__nv_MAIN__F1L56_1Arg2", arg: 3, scope: !69, file: !3, type: !26)
!76 = !DILocalVariable(name: "omp_sched_static", scope: !69, file: !3, type: !9)
!77 = !DILocalVariable(name: "omp_sched_dynamic", scope: !69, file: !3, type: !9)
!78 = !DILocalVariable(name: "omp_proc_bind_false", scope: !69, file: !3, type: !9)
!79 = !DILocalVariable(name: "omp_proc_bind_true", scope: !69, file: !3, type: !9)
!80 = !DILocalVariable(name: "omp_proc_bind_master", scope: !69, file: !3, type: !9)
!81 = !DILocalVariable(name: "omp_lock_hint_none", scope: !69, file: !3, type: !9)
!82 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !69, file: !3, type: !9)
!83 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !69, file: !3, type: !9)
!84 = !DILocation(line: 61, column: 1, scope: !69)
!85 = !DILocation(line: 57, column: 1, scope: !69)
!86 = !DILocalVariable(name: "i", scope: !69, file: !3, type: !9)
!87 = !DILocation(line: 58, column: 1, scope: !69)
!88 = !DILocation(line: 59, column: 1, scope: !69)

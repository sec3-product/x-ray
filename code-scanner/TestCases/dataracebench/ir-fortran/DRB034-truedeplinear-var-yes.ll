; ModuleID = '/tmp/DRB034-truedeplinear-var-yes-8382bc.ll'
source_filename = "/tmp/DRB034-truedeplinear-var-yes-8382bc.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%astruct.dt86 = type <{ i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8* }>

@.C300_MAIN_ = internal constant i32 2
@.C372_MAIN_ = internal constant i64 25
@.C355_MAIN_ = internal constant [29 x i8] c"Error, invalid integer value."
@.C354_MAIN_ = internal constant i32 39
@.C310_MAIN_ = internal constant i32 25
@.C350_MAIN_ = internal constant [5 x i8] c"(i10)"
@.C347_MAIN_ = internal constant i32 37
@.C365_MAIN_ = internal constant i64 4
@.C348_MAIN_ = internal constant [37 x i8] c"Allocation error, program terminated."
@.C312_MAIN_ = internal constant i32 28
@.C369_MAIN_ = internal constant i64 80
@.C368_MAIN_ = internal constant i64 14
@.C339_MAIN_ = internal constant [35 x i8] c"No command line arguments provided."
@.C337_MAIN_ = internal constant i32 6
@.C338_MAIN_ = internal constant [3 x i8] c"(a)"
@.C285_MAIN_ = internal constant i32 1
@.C309_MAIN_ = internal constant i32 14
@.C335_MAIN_ = internal constant [57 x i8] c"micro-benchmarks-fortran/DRB034-truedeplinear-var-yes.f95"
@.C311_MAIN_ = internal constant i32 23
@.C331_MAIN_ = internal constant i32 2000
@.C286_MAIN_ = internal constant i64 1
@.C284_MAIN_ = internal constant i64 0
@.C283_MAIN_ = internal constant i32 0
@.C300___nv_MAIN__F1L51_1 = internal constant i32 2
@.C285___nv_MAIN__F1L51_1 = internal constant i32 1
@.C283___nv_MAIN__F1L51_1 = internal constant i32 0

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__495 = alloca i32, align 4
  %.Z0977_356 = alloca i32*, align 8
  %"a$sd2_371" = alloca [16 x i64], align 8
  %.Z0971_346 = alloca [80 x i8]*, align 8
  %"args$sd1_367" = alloca [16 x i64], align 8
  %len_332 = alloca i32, align 4
  %argcount_315 = alloca i32, align 4
  %z__io_341 = alloca i32, align 4
  %z_b_0_319 = alloca i64, align 8
  %z_b_1_320 = alloca i64, align 8
  %z_e_61_323 = alloca i64, align 8
  %z_b_2_321 = alloca i64, align 8
  %z_b_3_322 = alloca i64, align 8
  %allocstatus_316 = alloca i32, align 4
  %.dY0001_382 = alloca i32, align 4
  %ix_318 = alloca i32, align 4
  %rderr_317 = alloca i32, align 4
  %z_b_4_325 = alloca i64, align 8
  %z_b_5_326 = alloca i64, align 8
  %z_e_68_329 = alloca i64, align 8
  %z_b_6_327 = alloca i64, align 8
  %z_b_7_328 = alloca i64, align 8
  %.dY0002_387 = alloca i32, align 4
  %i_313 = alloca i32, align 4
  %ulen_314 = alloca i32, align 4
  %.uplevelArgPack0001_474 = alloca %astruct.dt86, align 16
  call void @llvm.dbg.value(metadata i32 1, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 2, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 2, metadata !14, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !15, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !16, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 2, metadata !17, metadata !DIExpression()), !dbg !10
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !18
  store i32 %0, i32* %__gtid_MAIN__495, align 4, !dbg !18
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !19
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !19
  call void (i8*, ...) %2(i8* %1), !dbg !19
  call void @llvm.dbg.declare(metadata i32** %.Z0977_356, metadata !20, metadata !DIExpression(DW_OP_deref)), !dbg !10
  %3 = bitcast i32** %.Z0977_356 to i8**, !dbg !19
  store i8* null, i8** %3, align 8, !dbg !19
  call void @llvm.dbg.declare(metadata [16 x i64]* %"a$sd2_371", metadata !24, metadata !DIExpression()), !dbg !10
  %4 = bitcast [16 x i64]* %"a$sd2_371" to i64*, !dbg !19
  store i64 0, i64* %4, align 8, !dbg !19
  call void @llvm.dbg.declare(metadata [80 x i8]** %.Z0971_346, metadata !29, metadata !DIExpression(DW_OP_deref)), !dbg !10
  %5 = bitcast [80 x i8]** %.Z0971_346 to i8**, !dbg !19
  store i8* null, i8** %5, align 8, !dbg !19
  call void @llvm.dbg.declare(metadata [16 x i64]* %"args$sd1_367", metadata !24, metadata !DIExpression()), !dbg !10
  %6 = bitcast [16 x i64]* %"args$sd1_367" to i64*, !dbg !19
  store i64 0, i64* %6, align 8, !dbg !19
  br label %L.LB1_409

L.LB1_409:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %len_332, metadata !32, metadata !DIExpression()), !dbg !10
  store i32 2000, i32* %len_332, align 4, !dbg !33
  %7 = call i32 (...) @f90_cmd_arg_cnt(), !dbg !34
  call void @llvm.dbg.declare(metadata i32* %argcount_315, metadata !35, metadata !DIExpression()), !dbg !10
  store i32 %7, i32* %argcount_315, align 4, !dbg !34
  %8 = load i32, i32* %argcount_315, align 4, !dbg !36
  call void @llvm.dbg.value(metadata i32 %8, metadata !35, metadata !DIExpression()), !dbg !10
  %9 = icmp ne i32 %8, 0, !dbg !36
  br i1 %9, label %L.LB1_376, label %L.LB1_513, !dbg !36

L.LB1_513:                                        ; preds = %L.LB1_409
  call void (...) @_mp_bcs_nest(), !dbg !37
  %10 = bitcast i32* @.C311_MAIN_ to i8*, !dbg !37
  %11 = bitcast [57 x i8]* @.C335_MAIN_ to i8*, !dbg !37
  %12 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !37
  call void (i8*, i8*, i64, ...) %12(i8* %10, i8* %11, i64 57), !dbg !37
  %13 = bitcast i32* @.C309_MAIN_ to i8*, !dbg !37
  %14 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !37
  %15 = bitcast [3 x i8]* @.C338_MAIN_ to i8*, !dbg !37
  %16 = bitcast i32 (...)* @f90io_encode_fmta to i32 (i8*, i8*, i8*, i64, ...)*, !dbg !37
  %17 = call i32 (i8*, i8*, i8*, i64, ...) %16(i8* %13, i8* %14, i8* %15, i64 3), !dbg !37
  call void @llvm.dbg.declare(metadata i32* %z__io_341, metadata !38, metadata !DIExpression()), !dbg !10
  store i32 %17, i32* %z__io_341, align 4, !dbg !37
  %18 = bitcast i32* @.C337_MAIN_ to i8*, !dbg !37
  %19 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !37
  %20 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !37
  %21 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !37
  %22 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %21(i8* %18, i8* null, i8* %19, i8* %20, i8* null, i8* null, i64 0), !dbg !37
  store i32 %22, i32* %z__io_341, align 4, !dbg !37
  %23 = bitcast i32* @.C309_MAIN_ to i8*, !dbg !37
  %24 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !37
  %25 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !37
  %26 = bitcast [35 x i8]* @.C339_MAIN_ to i8*, !dbg !37
  %27 = bitcast i32 (...)* @f90io_fmt_writea to i32 (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !37
  %28 = call i32 (i8*, i8*, i8*, i8*, i64, ...) %27(i8* %23, i8* %24, i8* %25, i8* %26, i64 35), !dbg !37
  store i32 %28, i32* %z__io_341, align 4, !dbg !37
  %29 = call i32 (...) @f90io_fmtw_end(), !dbg !37
  store i32 %29, i32* %z__io_341, align 4, !dbg !37
  call void (...) @_mp_ecs_nest(), !dbg !37
  br label %L.LB1_376

L.LB1_376:                                        ; preds = %L.LB1_513, %L.LB1_409
  call void @llvm.dbg.declare(metadata i64* %z_b_0_319, metadata !39, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_0_319, align 8, !dbg !40
  %30 = load i32, i32* %argcount_315, align 4, !dbg !40
  call void @llvm.dbg.value(metadata i32 %30, metadata !35, metadata !DIExpression()), !dbg !10
  %31 = sext i32 %30 to i64, !dbg !40
  call void @llvm.dbg.declare(metadata i64* %z_b_1_320, metadata !39, metadata !DIExpression()), !dbg !10
  store i64 %31, i64* %z_b_1_320, align 8, !dbg !40
  %32 = load i64, i64* %z_b_1_320, align 8, !dbg !40
  call void @llvm.dbg.value(metadata i64 %32, metadata !39, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_61_323, metadata !39, metadata !DIExpression()), !dbg !10
  store i64 %32, i64* %z_e_61_323, align 8, !dbg !40
  %33 = bitcast [16 x i64]* %"args$sd1_367" to i8*, !dbg !40
  %34 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !40
  %35 = bitcast i64* @.C368_MAIN_ to i8*, !dbg !40
  %36 = bitcast i64* @.C369_MAIN_ to i8*, !dbg !40
  %37 = bitcast i64* %z_b_0_319 to i8*, !dbg !40
  %38 = bitcast i64* %z_b_1_320 to i8*, !dbg !40
  %39 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !40
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %39(i8* %33, i8* %34, i8* %35, i8* %36, i8* %37, i8* %38), !dbg !40
  %40 = bitcast [16 x i64]* %"args$sd1_367" to i8*, !dbg !40
  %41 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !40
  call void (i8*, i32, ...) %41(i8* %40, i32 14), !dbg !40
  %42 = load i64, i64* %z_b_1_320, align 8, !dbg !40
  call void @llvm.dbg.value(metadata i64 %42, metadata !39, metadata !DIExpression()), !dbg !10
  %43 = load i64, i64* %z_b_0_319, align 8, !dbg !40
  call void @llvm.dbg.value(metadata i64 %43, metadata !39, metadata !DIExpression()), !dbg !10
  %44 = sub nsw i64 %43, 1, !dbg !40
  %45 = sub nsw i64 %42, %44, !dbg !40
  call void @llvm.dbg.declare(metadata i64* %z_b_2_321, metadata !39, metadata !DIExpression()), !dbg !10
  store i64 %45, i64* %z_b_2_321, align 8, !dbg !40
  %46 = load i64, i64* %z_b_0_319, align 8, !dbg !40
  call void @llvm.dbg.value(metadata i64 %46, metadata !39, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_b_3_322, metadata !39, metadata !DIExpression()), !dbg !10
  store i64 %46, i64* %z_b_3_322, align 8, !dbg !40
  %47 = bitcast i64* %z_b_2_321 to i8*, !dbg !40
  %48 = bitcast i64* @.C368_MAIN_ to i8*, !dbg !40
  %49 = bitcast i64* @.C369_MAIN_ to i8*, !dbg !40
  call void @llvm.dbg.declare(metadata i32* %allocstatus_316, metadata !41, metadata !DIExpression()), !dbg !10
  %50 = bitcast i32* %allocstatus_316 to i8*, !dbg !40
  %51 = bitcast [80 x i8]** %.Z0971_346 to i8*, !dbg !40
  %52 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !40
  %53 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !40
  %54 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !40
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %54(i8* %47, i8* %48, i8* %49, i8* %50, i8* %51, i8* null, i8* %52, i8* %53, i8* null, i64 0), !dbg !40
  %55 = load i32, i32* %allocstatus_316, align 4, !dbg !42
  call void @llvm.dbg.value(metadata i32 %55, metadata !41, metadata !DIExpression()), !dbg !10
  %56 = icmp sle i32 %55, 0, !dbg !42
  br i1 %56, label %L.LB1_379, label %L.LB1_514, !dbg !42

L.LB1_514:                                        ; preds = %L.LB1_376
  call void (...) @_mp_bcs_nest(), !dbg !43
  %57 = bitcast i32* @.C312_MAIN_ to i8*, !dbg !43
  %58 = bitcast [57 x i8]* @.C335_MAIN_ to i8*, !dbg !43
  %59 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !43
  call void (i8*, i8*, i64, ...) %59(i8* %57, i8* %58, i64 57), !dbg !43
  %60 = bitcast i32* @.C309_MAIN_ to i8*, !dbg !43
  %61 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !43
  %62 = bitcast [3 x i8]* @.C338_MAIN_ to i8*, !dbg !43
  %63 = bitcast i32 (...)* @f90io_encode_fmta to i32 (i8*, i8*, i8*, i64, ...)*, !dbg !43
  %64 = call i32 (i8*, i8*, i8*, i64, ...) %63(i8* %60, i8* %61, i8* %62, i64 3), !dbg !43
  store i32 %64, i32* %z__io_341, align 4, !dbg !43
  %65 = bitcast i32* @.C337_MAIN_ to i8*, !dbg !43
  %66 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !43
  %67 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !43
  %68 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !43
  %69 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %68(i8* %65, i8* null, i8* %66, i8* %67, i8* null, i8* null, i64 0), !dbg !43
  store i32 %69, i32* %z__io_341, align 4, !dbg !43
  %70 = bitcast i32* @.C309_MAIN_ to i8*, !dbg !43
  %71 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !43
  %72 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !43
  %73 = bitcast [37 x i8]* @.C348_MAIN_ to i8*, !dbg !43
  %74 = bitcast i32 (...)* @f90io_fmt_writea to i32 (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !43
  %75 = call i32 (i8*, i8*, i8*, i8*, i64, ...) %74(i8* %70, i8* %71, i8* %72, i8* %73, i64 37), !dbg !43
  store i32 %75, i32* %z__io_341, align 4, !dbg !43
  %76 = call i32 (...) @f90io_fmtw_end(), !dbg !43
  store i32 %76, i32* %z__io_341, align 4, !dbg !43
  call void (...) @_mp_ecs_nest(), !dbg !43
  %77 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !44
  %78 = bitcast void (...)* @f90_stop08a to void (i8*, i8*, i64, ...)*, !dbg !44
  call void (i8*, i8*, i64, ...) %78(i8* %77, i8* null, i64 0), !dbg !44
  br label %L.LB1_379

L.LB1_379:                                        ; preds = %L.LB1_514, %L.LB1_376
  %79 = load i32, i32* %argcount_315, align 4, !dbg !45
  call void @llvm.dbg.value(metadata i32 %79, metadata !35, metadata !DIExpression()), !dbg !10
  store i32 %79, i32* %.dY0001_382, align 4, !dbg !45
  call void @llvm.dbg.declare(metadata i32* %ix_318, metadata !46, metadata !DIExpression()), !dbg !10
  store i32 1, i32* %ix_318, align 4, !dbg !45
  %80 = load i32, i32* %.dY0001_382, align 4, !dbg !45
  %81 = icmp sle i32 %80, 0, !dbg !45
  br i1 %81, label %L.LB1_381, label %L.LB1_380, !dbg !45

L.LB1_380:                                        ; preds = %L.LB1_380, %L.LB1_379
  %82 = bitcast i32* %ix_318 to i8*, !dbg !47
  %83 = load [80 x i8]*, [80 x i8]** %.Z0971_346, align 8, !dbg !47
  call void @llvm.dbg.value(metadata [80 x i8]* %83, metadata !29, metadata !DIExpression()), !dbg !10
  %84 = bitcast [80 x i8]* %83 to i8*, !dbg !47
  %85 = getelementptr i8, i8* %84, i64 -80, !dbg !47
  %86 = load i32, i32* %ix_318, align 4, !dbg !47
  call void @llvm.dbg.value(metadata i32 %86, metadata !46, metadata !DIExpression()), !dbg !10
  %87 = sext i32 %86 to i64, !dbg !47
  %88 = bitcast [16 x i64]* %"args$sd1_367" to i8*, !dbg !47
  %89 = getelementptr i8, i8* %88, i64 56, !dbg !47
  %90 = bitcast i8* %89 to i64*, !dbg !47
  %91 = load i64, i64* %90, align 8, !dbg !47
  %92 = add nsw i64 %87, %91, !dbg !47
  %93 = mul nsw i64 %92, 80, !dbg !47
  %94 = getelementptr i8, i8* %85, i64 %93, !dbg !47
  %95 = bitcast i64* @.C365_MAIN_ to i8*, !dbg !47
  %96 = bitcast void (...)* @f90_get_cmd_arga to void (i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !47
  call void (i8*, i8*, i8*, i8*, i8*, i64, ...) %96(i8* %82, i8* %94, i8* null, i8* null, i8* %95, i64 80), !dbg !47
  %97 = load i32, i32* %ix_318, align 4, !dbg !48
  call void @llvm.dbg.value(metadata i32 %97, metadata !46, metadata !DIExpression()), !dbg !10
  %98 = add nsw i32 %97, 1, !dbg !48
  store i32 %98, i32* %ix_318, align 4, !dbg !48
  %99 = load i32, i32* %.dY0001_382, align 4, !dbg !48
  %100 = sub nsw i32 %99, 1, !dbg !48
  store i32 %100, i32* %.dY0001_382, align 4, !dbg !48
  %101 = load i32, i32* %.dY0001_382, align 4, !dbg !48
  %102 = icmp sgt i32 %101, 0, !dbg !48
  br i1 %102, label %L.LB1_380, label %L.LB1_381, !dbg !48

L.LB1_381:                                        ; preds = %L.LB1_380, %L.LB1_379
  %103 = load i32, i32* %argcount_315, align 4, !dbg !49
  call void @llvm.dbg.value(metadata i32 %103, metadata !35, metadata !DIExpression()), !dbg !10
  %104 = icmp sle i32 %103, 0, !dbg !49
  br i1 %104, label %L.LB1_383, label %L.LB1_515, !dbg !49

L.LB1_515:                                        ; preds = %L.LB1_381
  call void (...) @_mp_bcs_nest(), !dbg !50
  %105 = bitcast i32* @.C347_MAIN_ to i8*, !dbg !50
  %106 = bitcast [57 x i8]* @.C335_MAIN_ to i8*, !dbg !50
  %107 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !50
  call void (i8*, i8*, i64, ...) %107(i8* %105, i8* %106, i64 57), !dbg !50
  %108 = bitcast i32* @.C309_MAIN_ to i8*, !dbg !50
  %109 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !50
  %110 = bitcast [5 x i8]* @.C350_MAIN_ to i8*, !dbg !50
  %111 = bitcast i32 (...)* @f90io_encode_fmta to i32 (i8*, i8*, i8*, i64, ...)*, !dbg !50
  %112 = call i32 (i8*, i8*, i8*, i64, ...) %111(i8* %108, i8* %109, i8* %110, i64 5), !dbg !50
  store i32 %112, i32* %z__io_341, align 4, !dbg !50
  %113 = load [80 x i8]*, [80 x i8]** %.Z0971_346, align 8, !dbg !50
  call void @llvm.dbg.value(metadata [80 x i8]* %113, metadata !29, metadata !DIExpression()), !dbg !10
  %114 = bitcast [80 x i8]* %113 to i8*, !dbg !50
  %115 = bitcast [16 x i64]* %"args$sd1_367" to i8*, !dbg !50
  %116 = getelementptr i8, i8* %115, i64 56, !dbg !50
  %117 = bitcast i8* %116 to i64*, !dbg !50
  %118 = load i64, i64* %117, align 8, !dbg !50
  %119 = mul nsw i64 %118, 80, !dbg !50
  %120 = getelementptr i8, i8* %114, i64 %119, !dbg !50
  %121 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !50
  %122 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !50
  call void @llvm.dbg.declare(metadata i32* %rderr_317, metadata !51, metadata !DIExpression()), !dbg !10
  %123 = bitcast i32* %rderr_317 to i8*, !dbg !50
  %124 = bitcast i32 (...)* @f90io_fmtr_intern_inita to i32 (i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !50
  %125 = call i32 (i8*, i8*, i8*, i8*, i8*, i64, ...) %124(i8* %120, i8* %121, i8* %122, i8* %123, i8* null, i64 80), !dbg !50
  store i32 %125, i32* %z__io_341, align 4, !dbg !50
  %126 = bitcast i32* @.C310_MAIN_ to i8*, !dbg !50
  %127 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !50
  %128 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !50
  %129 = bitcast i32* %len_332 to i8*, !dbg !50
  %130 = bitcast i32 (...)* @f90io_fmt_reada to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !50
  %131 = call i32 (i8*, i8*, i8*, i8*, ...) %130(i8* %126, i8* %127, i8* %128, i8* %129), !dbg !50
  store i32 %131, i32* %z__io_341, align 4, !dbg !50
  %132 = call i32 (...) @f90io_fmtr_end(), !dbg !50
  store i32 %132, i32* %z__io_341, align 4, !dbg !50
  call void (...) @_mp_ecs_nest(), !dbg !50
  %133 = load i32, i32* %rderr_317, align 4, !dbg !52
  call void @llvm.dbg.value(metadata i32 %133, metadata !51, metadata !DIExpression()), !dbg !10
  %134 = icmp eq i32 %133, 0, !dbg !52
  br i1 %134, label %L.LB1_384, label %L.LB1_516, !dbg !52

L.LB1_516:                                        ; preds = %L.LB1_515
  call void (...) @_mp_bcs_nest(), !dbg !53
  %135 = bitcast i32* @.C354_MAIN_ to i8*, !dbg !53
  %136 = bitcast [57 x i8]* @.C335_MAIN_ to i8*, !dbg !53
  %137 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !53
  call void (i8*, i8*, i64, ...) %137(i8* %135, i8* %136, i64 57), !dbg !53
  %138 = bitcast i32* @.C309_MAIN_ to i8*, !dbg !53
  %139 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !53
  %140 = bitcast [3 x i8]* @.C338_MAIN_ to i8*, !dbg !53
  %141 = bitcast i32 (...)* @f90io_encode_fmta to i32 (i8*, i8*, i8*, i64, ...)*, !dbg !53
  %142 = call i32 (i8*, i8*, i8*, i64, ...) %141(i8* %138, i8* %139, i8* %140, i64 3), !dbg !53
  store i32 %142, i32* %z__io_341, align 4, !dbg !53
  %143 = bitcast i32* @.C337_MAIN_ to i8*, !dbg !53
  %144 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !53
  %145 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !53
  %146 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !53
  %147 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %146(i8* %143, i8* null, i8* %144, i8* %145, i8* null, i8* null, i64 0), !dbg !53
  store i32 %147, i32* %z__io_341, align 4, !dbg !53
  %148 = bitcast i32* @.C309_MAIN_ to i8*, !dbg !53
  %149 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !53
  %150 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !53
  %151 = bitcast [29 x i8]* @.C355_MAIN_ to i8*, !dbg !53
  %152 = bitcast i32 (...)* @f90io_fmt_writea to i32 (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !53
  %153 = call i32 (i8*, i8*, i8*, i8*, i64, ...) %152(i8* %148, i8* %149, i8* %150, i8* %151, i64 29), !dbg !53
  store i32 %153, i32* %z__io_341, align 4, !dbg !53
  %154 = call i32 (...) @f90io_fmtw_end(), !dbg !53
  store i32 %154, i32* %z__io_341, align 4, !dbg !53
  call void (...) @_mp_ecs_nest(), !dbg !53
  br label %L.LB1_384

L.LB1_384:                                        ; preds = %L.LB1_516, %L.LB1_515
  br label %L.LB1_383

L.LB1_383:                                        ; preds = %L.LB1_384, %L.LB1_381
  call void @llvm.dbg.declare(metadata i64* %z_b_4_325, metadata !39, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_4_325, align 8, !dbg !54
  %155 = load i32, i32* %len_332, align 4, !dbg !54
  call void @llvm.dbg.value(metadata i32 %155, metadata !32, metadata !DIExpression()), !dbg !10
  %156 = sext i32 %155 to i64, !dbg !54
  call void @llvm.dbg.declare(metadata i64* %z_b_5_326, metadata !39, metadata !DIExpression()), !dbg !10
  store i64 %156, i64* %z_b_5_326, align 8, !dbg !54
  %157 = load i64, i64* %z_b_5_326, align 8, !dbg !54
  call void @llvm.dbg.value(metadata i64 %157, metadata !39, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_68_329, metadata !39, metadata !DIExpression()), !dbg !10
  store i64 %157, i64* %z_e_68_329, align 8, !dbg !54
  %158 = bitcast [16 x i64]* %"a$sd2_371" to i8*, !dbg !54
  %159 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !54
  %160 = bitcast i64* @.C372_MAIN_ to i8*, !dbg !54
  %161 = bitcast i64* @.C365_MAIN_ to i8*, !dbg !54
  %162 = bitcast i64* %z_b_4_325 to i8*, !dbg !54
  %163 = bitcast i64* %z_b_5_326 to i8*, !dbg !54
  %164 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !54
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %164(i8* %158, i8* %159, i8* %160, i8* %161, i8* %162, i8* %163), !dbg !54
  %165 = bitcast [16 x i64]* %"a$sd2_371" to i8*, !dbg !54
  %166 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !54
  call void (i8*, i32, ...) %166(i8* %165, i32 25), !dbg !54
  %167 = load i64, i64* %z_b_5_326, align 8, !dbg !54
  call void @llvm.dbg.value(metadata i64 %167, metadata !39, metadata !DIExpression()), !dbg !10
  %168 = load i64, i64* %z_b_4_325, align 8, !dbg !54
  call void @llvm.dbg.value(metadata i64 %168, metadata !39, metadata !DIExpression()), !dbg !10
  %169 = sub nsw i64 %168, 1, !dbg !54
  %170 = sub nsw i64 %167, %169, !dbg !54
  call void @llvm.dbg.declare(metadata i64* %z_b_6_327, metadata !39, metadata !DIExpression()), !dbg !10
  store i64 %170, i64* %z_b_6_327, align 8, !dbg !54
  %171 = load i64, i64* %z_b_4_325, align 8, !dbg !54
  call void @llvm.dbg.value(metadata i64 %171, metadata !39, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_b_7_328, metadata !39, metadata !DIExpression()), !dbg !10
  store i64 %171, i64* %z_b_7_328, align 8, !dbg !54
  %172 = bitcast i64* %z_b_6_327 to i8*, !dbg !54
  %173 = bitcast i64* @.C372_MAIN_ to i8*, !dbg !54
  %174 = bitcast i64* @.C365_MAIN_ to i8*, !dbg !54
  %175 = bitcast i32** %.Z0977_356 to i8*, !dbg !54
  %176 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !54
  %177 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !54
  %178 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !54
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %178(i8* %172, i8* %173, i8* %174, i8* null, i8* %175, i8* null, i8* %176, i8* %177, i8* null, i64 0), !dbg !54
  %179 = load i32, i32* %len_332, align 4, !dbg !55
  call void @llvm.dbg.value(metadata i32 %179, metadata !32, metadata !DIExpression()), !dbg !10
  store i32 %179, i32* %.dY0002_387, align 4, !dbg !55
  call void @llvm.dbg.declare(metadata i32* %i_313, metadata !56, metadata !DIExpression()), !dbg !10
  store i32 1, i32* %i_313, align 4, !dbg !55
  %180 = load i32, i32* %.dY0002_387, align 4, !dbg !55
  %181 = icmp sle i32 %180, 0, !dbg !55
  br i1 %181, label %L.LB1_386, label %L.LB1_385, !dbg !55

L.LB1_385:                                        ; preds = %L.LB1_385, %L.LB1_383
  %182 = load i32, i32* %i_313, align 4, !dbg !57
  call void @llvm.dbg.value(metadata i32 %182, metadata !56, metadata !DIExpression()), !dbg !10
  %183 = load i32, i32* %i_313, align 4, !dbg !57
  call void @llvm.dbg.value(metadata i32 %183, metadata !56, metadata !DIExpression()), !dbg !10
  %184 = sext i32 %183 to i64, !dbg !57
  %185 = bitcast [16 x i64]* %"a$sd2_371" to i8*, !dbg !57
  %186 = getelementptr i8, i8* %185, i64 56, !dbg !57
  %187 = bitcast i8* %186 to i64*, !dbg !57
  %188 = load i64, i64* %187, align 8, !dbg !57
  %189 = add nsw i64 %184, %188, !dbg !57
  %190 = load i32*, i32** %.Z0977_356, align 8, !dbg !57
  call void @llvm.dbg.value(metadata i32* %190, metadata !20, metadata !DIExpression()), !dbg !10
  %191 = bitcast i32* %190 to i8*, !dbg !57
  %192 = getelementptr i8, i8* %191, i64 -4, !dbg !57
  %193 = bitcast i8* %192 to i32*, !dbg !57
  %194 = getelementptr i32, i32* %193, i64 %189, !dbg !57
  store i32 %182, i32* %194, align 4, !dbg !57
  %195 = load i32, i32* %i_313, align 4, !dbg !58
  call void @llvm.dbg.value(metadata i32 %195, metadata !56, metadata !DIExpression()), !dbg !10
  %196 = add nsw i32 %195, 1, !dbg !58
  store i32 %196, i32* %i_313, align 4, !dbg !58
  %197 = load i32, i32* %.dY0002_387, align 4, !dbg !58
  %198 = sub nsw i32 %197, 1, !dbg !58
  store i32 %198, i32* %.dY0002_387, align 4, !dbg !58
  %199 = load i32, i32* %.dY0002_387, align 4, !dbg !58
  %200 = icmp sgt i32 %199, 0, !dbg !58
  br i1 %200, label %L.LB1_385, label %L.LB1_386, !dbg !58

L.LB1_386:                                        ; preds = %L.LB1_385, %L.LB1_383
  %201 = load i32, i32* %len_332, align 4, !dbg !59
  call void @llvm.dbg.value(metadata i32 %201, metadata !32, metadata !DIExpression()), !dbg !10
  %202 = load i32, i32* %len_332, align 4, !dbg !59
  call void @llvm.dbg.value(metadata i32 %202, metadata !32, metadata !DIExpression()), !dbg !10
  %203 = lshr i32 %202, 31, !dbg !59
  %204 = add nsw i32 %201, %203, !dbg !59
  %205 = ashr i32 %204, 1, !dbg !59
  call void @llvm.dbg.declare(metadata i32* %ulen_314, metadata !60, metadata !DIExpression()), !dbg !10
  store i32 %205, i32* %ulen_314, align 4, !dbg !59
  %206 = bitcast i32* %ulen_314 to i8*, !dbg !61
  %207 = bitcast %astruct.dt86* %.uplevelArgPack0001_474 to i8**, !dbg !61
  store i8* %206, i8** %207, align 8, !dbg !61
  %208 = bitcast i32** %.Z0977_356 to i8*, !dbg !61
  %209 = bitcast %astruct.dt86* %.uplevelArgPack0001_474 to i8*, !dbg !61
  %210 = getelementptr i8, i8* %209, i64 8, !dbg !61
  %211 = bitcast i8* %210 to i8**, !dbg !61
  store i8* %208, i8** %211, align 8, !dbg !61
  %212 = bitcast i32** %.Z0977_356 to i8*, !dbg !61
  %213 = bitcast %astruct.dt86* %.uplevelArgPack0001_474 to i8*, !dbg !61
  %214 = getelementptr i8, i8* %213, i64 16, !dbg !61
  %215 = bitcast i8* %214 to i8**, !dbg !61
  store i8* %212, i8** %215, align 8, !dbg !61
  %216 = bitcast i64* %z_b_4_325 to i8*, !dbg !61
  %217 = bitcast %astruct.dt86* %.uplevelArgPack0001_474 to i8*, !dbg !61
  %218 = getelementptr i8, i8* %217, i64 24, !dbg !61
  %219 = bitcast i8* %218 to i8**, !dbg !61
  store i8* %216, i8** %219, align 8, !dbg !61
  %220 = bitcast i64* %z_b_5_326 to i8*, !dbg !61
  %221 = bitcast %astruct.dt86* %.uplevelArgPack0001_474 to i8*, !dbg !61
  %222 = getelementptr i8, i8* %221, i64 32, !dbg !61
  %223 = bitcast i8* %222 to i8**, !dbg !61
  store i8* %220, i8** %223, align 8, !dbg !61
  %224 = bitcast i64* %z_e_68_329 to i8*, !dbg !61
  %225 = bitcast %astruct.dt86* %.uplevelArgPack0001_474 to i8*, !dbg !61
  %226 = getelementptr i8, i8* %225, i64 40, !dbg !61
  %227 = bitcast i8* %226 to i8**, !dbg !61
  store i8* %224, i8** %227, align 8, !dbg !61
  %228 = bitcast i64* %z_b_6_327 to i8*, !dbg !61
  %229 = bitcast %astruct.dt86* %.uplevelArgPack0001_474 to i8*, !dbg !61
  %230 = getelementptr i8, i8* %229, i64 48, !dbg !61
  %231 = bitcast i8* %230 to i8**, !dbg !61
  store i8* %228, i8** %231, align 8, !dbg !61
  %232 = bitcast i64* %z_b_7_328 to i8*, !dbg !61
  %233 = bitcast %astruct.dt86* %.uplevelArgPack0001_474 to i8*, !dbg !61
  %234 = getelementptr i8, i8* %233, i64 56, !dbg !61
  %235 = bitcast i8* %234 to i8**, !dbg !61
  store i8* %232, i8** %235, align 8, !dbg !61
  %236 = bitcast [16 x i64]* %"a$sd2_371" to i8*, !dbg !61
  %237 = bitcast %astruct.dt86* %.uplevelArgPack0001_474 to i8*, !dbg !61
  %238 = getelementptr i8, i8* %237, i64 64, !dbg !61
  %239 = bitcast i8* %238 to i8**, !dbg !61
  store i8* %236, i8** %239, align 8, !dbg !61
  br label %L.LB1_493, !dbg !61

L.LB1_493:                                        ; preds = %L.LB1_386
  %240 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L51_1_ to i64*, !dbg !61
  %241 = bitcast %astruct.dt86* %.uplevelArgPack0001_474 to i64*, !dbg !61
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %240, i64* %241), !dbg !61
  %242 = load [80 x i8]*, [80 x i8]** %.Z0971_346, align 8, !dbg !62
  call void @llvm.dbg.value(metadata [80 x i8]* %242, metadata !29, metadata !DIExpression()), !dbg !10
  %243 = bitcast [80 x i8]* %242 to i8*, !dbg !62
  %244 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !62
  %245 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, i64, ...)*, !dbg !62
  call void (i8*, i8*, i8*, i8*, i64, i64, ...) %245(i8* null, i8* %243, i8* %244, i8* null, i64 80, i64 0), !dbg !62
  %246 = bitcast [80 x i8]** %.Z0971_346 to i8**, !dbg !62
  store i8* null, i8** %246, align 8, !dbg !62
  %247 = bitcast [16 x i64]* %"args$sd1_367" to i64*, !dbg !62
  store i64 0, i64* %247, align 8, !dbg !62
  %248 = load i32*, i32** %.Z0977_356, align 8, !dbg !62
  call void @llvm.dbg.value(metadata i32* %248, metadata !20, metadata !DIExpression()), !dbg !10
  %249 = bitcast i32* %248 to i8*, !dbg !62
  %250 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !62
  %251 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !62
  call void (i8*, i8*, i8*, i8*, i64, ...) %251(i8* null, i8* %249, i8* %250, i8* null, i64 0), !dbg !62
  %252 = bitcast i32** %.Z0977_356 to i8**, !dbg !62
  store i8* null, i8** %252, align 8, !dbg !62
  %253 = bitcast [16 x i64]* %"a$sd2_371" to i64*, !dbg !62
  store i64 0, i64* %253, align 8, !dbg !62
  ret void, !dbg !18
}

define internal void @__nv_MAIN__F1L51_1_(i32* %__nv_MAIN__F1L51_1Arg0, i64* %__nv_MAIN__F1L51_1Arg1, i64* %__nv_MAIN__F1L51_1Arg2) #0 !dbg !63 {
L.entry:
  %__gtid___nv_MAIN__F1L51_1__535 = alloca i32, align 4
  %.i0000p_361 = alloca i32, align 4
  %i_360 = alloca i32, align 4
  %.du0003p_391 = alloca i32, align 4
  %.de0003p_392 = alloca i32, align 4
  %.di0003p_393 = alloca i32, align 4
  %.ds0003p_394 = alloca i32, align 4
  %.dl0003p_396 = alloca i32, align 4
  %.dl0003p.copy_529 = alloca i32, align 4
  %.de0003p.copy_530 = alloca i32, align 4
  %.ds0003p.copy_531 = alloca i32, align 4
  %.dX0003p_395 = alloca i32, align 4
  %.dY0003p_390 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L51_1Arg0, metadata !66, metadata !DIExpression()), !dbg !67
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L51_1Arg1, metadata !68, metadata !DIExpression()), !dbg !67
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L51_1Arg2, metadata !69, metadata !DIExpression()), !dbg !67
  call void @llvm.dbg.value(metadata i32 1, metadata !70, metadata !DIExpression()), !dbg !67
  call void @llvm.dbg.value(metadata i32 2, metadata !71, metadata !DIExpression()), !dbg !67
  call void @llvm.dbg.value(metadata i32 0, metadata !72, metadata !DIExpression()), !dbg !67
  call void @llvm.dbg.value(metadata i32 1, metadata !73, metadata !DIExpression()), !dbg !67
  call void @llvm.dbg.value(metadata i32 2, metadata !74, metadata !DIExpression()), !dbg !67
  call void @llvm.dbg.value(metadata i32 0, metadata !75, metadata !DIExpression()), !dbg !67
  call void @llvm.dbg.value(metadata i32 1, metadata !76, metadata !DIExpression()), !dbg !67
  call void @llvm.dbg.value(metadata i32 2, metadata !77, metadata !DIExpression()), !dbg !67
  %0 = load i32, i32* %__nv_MAIN__F1L51_1Arg0, align 4, !dbg !78
  store i32 %0, i32* %__gtid___nv_MAIN__F1L51_1__535, align 4, !dbg !78
  br label %L.LB2_520

L.LB2_520:                                        ; preds = %L.entry
  br label %L.LB2_359

L.LB2_359:                                        ; preds = %L.LB2_520
  store i32 0, i32* %.i0000p_361, align 4, !dbg !79
  call void @llvm.dbg.declare(metadata i32* %i_360, metadata !80, metadata !DIExpression()), !dbg !78
  store i32 1, i32* %i_360, align 4, !dbg !79
  %1 = bitcast i64* %__nv_MAIN__F1L51_1Arg2 to i32**, !dbg !79
  %2 = load i32*, i32** %1, align 8, !dbg !79
  %3 = load i32, i32* %2, align 4, !dbg !79
  store i32 %3, i32* %.du0003p_391, align 4, !dbg !79
  %4 = bitcast i64* %__nv_MAIN__F1L51_1Arg2 to i32**, !dbg !79
  %5 = load i32*, i32** %4, align 8, !dbg !79
  %6 = load i32, i32* %5, align 4, !dbg !79
  store i32 %6, i32* %.de0003p_392, align 4, !dbg !79
  store i32 1, i32* %.di0003p_393, align 4, !dbg !79
  %7 = load i32, i32* %.di0003p_393, align 4, !dbg !79
  store i32 %7, i32* %.ds0003p_394, align 4, !dbg !79
  store i32 1, i32* %.dl0003p_396, align 4, !dbg !79
  %8 = load i32, i32* %.dl0003p_396, align 4, !dbg !79
  store i32 %8, i32* %.dl0003p.copy_529, align 4, !dbg !79
  %9 = load i32, i32* %.de0003p_392, align 4, !dbg !79
  store i32 %9, i32* %.de0003p.copy_530, align 4, !dbg !79
  %10 = load i32, i32* %.ds0003p_394, align 4, !dbg !79
  store i32 %10, i32* %.ds0003p.copy_531, align 4, !dbg !79
  %11 = load i32, i32* %__gtid___nv_MAIN__F1L51_1__535, align 4, !dbg !79
  %12 = bitcast i32* %.i0000p_361 to i64*, !dbg !79
  %13 = bitcast i32* %.dl0003p.copy_529 to i64*, !dbg !79
  %14 = bitcast i32* %.de0003p.copy_530 to i64*, !dbg !79
  %15 = bitcast i32* %.ds0003p.copy_531 to i64*, !dbg !79
  %16 = load i32, i32* %.ds0003p.copy_531, align 4, !dbg !79
  call void @__kmpc_for_static_init_4(i64* null, i32 %11, i32 34, i64* %12, i64* %13, i64* %14, i64* %15, i32 %16, i32 1), !dbg !79
  %17 = load i32, i32* %.dl0003p.copy_529, align 4, !dbg !79
  store i32 %17, i32* %.dl0003p_396, align 4, !dbg !79
  %18 = load i32, i32* %.de0003p.copy_530, align 4, !dbg !79
  store i32 %18, i32* %.de0003p_392, align 4, !dbg !79
  %19 = load i32, i32* %.ds0003p.copy_531, align 4, !dbg !79
  store i32 %19, i32* %.ds0003p_394, align 4, !dbg !79
  %20 = load i32, i32* %.dl0003p_396, align 4, !dbg !79
  store i32 %20, i32* %i_360, align 4, !dbg !79
  %21 = load i32, i32* %i_360, align 4, !dbg !79
  call void @llvm.dbg.value(metadata i32 %21, metadata !80, metadata !DIExpression()), !dbg !78
  store i32 %21, i32* %.dX0003p_395, align 4, !dbg !79
  %22 = load i32, i32* %.dX0003p_395, align 4, !dbg !79
  %23 = load i32, i32* %.du0003p_391, align 4, !dbg !79
  %24 = icmp sgt i32 %22, %23, !dbg !79
  br i1 %24, label %L.LB2_389, label %L.LB2_559, !dbg !79

L.LB2_559:                                        ; preds = %L.LB2_359
  %25 = load i32, i32* %.dX0003p_395, align 4, !dbg !79
  store i32 %25, i32* %i_360, align 4, !dbg !79
  %26 = load i32, i32* %.di0003p_393, align 4, !dbg !79
  %27 = load i32, i32* %.de0003p_392, align 4, !dbg !79
  %28 = load i32, i32* %.dX0003p_395, align 4, !dbg !79
  %29 = sub nsw i32 %27, %28, !dbg !79
  %30 = add nsw i32 %26, %29, !dbg !79
  %31 = load i32, i32* %.di0003p_393, align 4, !dbg !79
  %32 = sdiv i32 %30, %31, !dbg !79
  store i32 %32, i32* %.dY0003p_390, align 4, !dbg !79
  %33 = load i32, i32* %.dY0003p_390, align 4, !dbg !79
  %34 = icmp sle i32 %33, 0, !dbg !79
  br i1 %34, label %L.LB2_399, label %L.LB2_398, !dbg !79

L.LB2_398:                                        ; preds = %L.LB2_398, %L.LB2_559
  %35 = load i32, i32* %i_360, align 4, !dbg !81
  call void @llvm.dbg.value(metadata i32 %35, metadata !80, metadata !DIExpression()), !dbg !78
  %36 = sext i32 %35 to i64, !dbg !81
  %37 = bitcast i64* %__nv_MAIN__F1L51_1Arg2 to i8*, !dbg !81
  %38 = getelementptr i8, i8* %37, i64 64, !dbg !81
  %39 = bitcast i8* %38 to i8**, !dbg !81
  %40 = load i8*, i8** %39, align 8, !dbg !81
  %41 = getelementptr i8, i8* %40, i64 56, !dbg !81
  %42 = bitcast i8* %41 to i64*, !dbg !81
  %43 = load i64, i64* %42, align 8, !dbg !81
  %44 = add nsw i64 %36, %43, !dbg !81
  %45 = bitcast i64* %__nv_MAIN__F1L51_1Arg2 to i8*, !dbg !81
  %46 = getelementptr i8, i8* %45, i64 16, !dbg !81
  %47 = bitcast i8* %46 to i8***, !dbg !81
  %48 = load i8**, i8*** %47, align 8, !dbg !81
  %49 = load i8*, i8** %48, align 8, !dbg !81
  %50 = getelementptr i8, i8* %49, i64 -4, !dbg !81
  %51 = bitcast i8* %50 to i32*, !dbg !81
  %52 = getelementptr i32, i32* %51, i64 %44, !dbg !81
  %53 = load i32, i32* %52, align 4, !dbg !81
  %54 = add nsw i32 %53, 1, !dbg !81
  %55 = bitcast i64* %__nv_MAIN__F1L51_1Arg2 to i8*, !dbg !81
  %56 = getelementptr i8, i8* %55, i64 64, !dbg !81
  %57 = bitcast i8* %56 to i8**, !dbg !81
  %58 = load i8*, i8** %57, align 8, !dbg !81
  %59 = getelementptr i8, i8* %58, i64 56, !dbg !81
  %60 = bitcast i8* %59 to i64*, !dbg !81
  %61 = load i64, i64* %60, align 8, !dbg !81
  %62 = load i32, i32* %i_360, align 4, !dbg !81
  call void @llvm.dbg.value(metadata i32 %62, metadata !80, metadata !DIExpression()), !dbg !78
  %63 = mul nsw i32 %62, 2, !dbg !81
  %64 = sext i32 %63 to i64, !dbg !81
  %65 = add nsw i64 %61, %64, !dbg !81
  %66 = bitcast i64* %__nv_MAIN__F1L51_1Arg2 to i8*, !dbg !81
  %67 = getelementptr i8, i8* %66, i64 16, !dbg !81
  %68 = bitcast i8* %67 to i8***, !dbg !81
  %69 = load i8**, i8*** %68, align 8, !dbg !81
  %70 = load i8*, i8** %69, align 8, !dbg !81
  %71 = getelementptr i8, i8* %70, i64 -4, !dbg !81
  %72 = bitcast i8* %71 to i32*, !dbg !81
  %73 = getelementptr i32, i32* %72, i64 %65, !dbg !81
  store i32 %54, i32* %73, align 4, !dbg !81
  %74 = load i32, i32* %.di0003p_393, align 4, !dbg !78
  %75 = load i32, i32* %i_360, align 4, !dbg !78
  call void @llvm.dbg.value(metadata i32 %75, metadata !80, metadata !DIExpression()), !dbg !78
  %76 = add nsw i32 %74, %75, !dbg !78
  store i32 %76, i32* %i_360, align 4, !dbg !78
  %77 = load i32, i32* %.dY0003p_390, align 4, !dbg !78
  %78 = sub nsw i32 %77, 1, !dbg !78
  store i32 %78, i32* %.dY0003p_390, align 4, !dbg !78
  %79 = load i32, i32* %.dY0003p_390, align 4, !dbg !78
  %80 = icmp sgt i32 %79, 0, !dbg !78
  br i1 %80, label %L.LB2_398, label %L.LB2_399, !dbg !78

L.LB2_399:                                        ; preds = %L.LB2_398, %L.LB2_559
  br label %L.LB2_389

L.LB2_389:                                        ; preds = %L.LB2_399, %L.LB2_359
  %81 = load i32, i32* %__gtid___nv_MAIN__F1L51_1__535, align 4, !dbg !78
  call void @__kmpc_for_static_fini(i64* null, i32 %81), !dbg !78
  br label %L.LB2_362

L.LB2_362:                                        ; preds = %L.LB2_389
  ret void, !dbg !78
}

declare void @__kmpc_for_static_fini(i64*, i32) #0

declare void @__kmpc_for_static_init_4(i64*, i32, i32, i64*, i64*, i64*, i64*, i32, i32) #0

declare void @f90_dealloc03a_i8(...) #0

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
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB034-truedeplinear-var-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb034_truedeplinear_var_yes", scope: !2, file: !3, line: 11, type: !6, scopeLine: 11, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
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
!18 = !DILocation(line: 58, column: 1, scope: !5)
!19 = !DILocation(line: 11, column: 1, scope: !5)
!20 = !DILocalVariable(name: "a", scope: !5, file: !3, type: !21)
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
!56 = !DILocalVariable(name: "i", scope: !5, file: !3, type: !9)
!57 = !DILocation(line: 46, column: 1, scope: !5)
!58 = !DILocation(line: 47, column: 1, scope: !5)
!59 = !DILocation(line: 49, column: 1, scope: !5)
!60 = !DILocalVariable(name: "ulen", scope: !5, file: !3, type: !9)
!61 = !DILocation(line: 51, column: 1, scope: !5)
!62 = !DILocation(line: 57, column: 1, scope: !5)
!63 = distinct !DISubprogram(name: "__nv_MAIN__F1L51_1", scope: !2, file: !3, line: 51, type: !64, scopeLine: 51, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!64 = !DISubroutineType(types: !65)
!65 = !{null, !9, !26, !26}
!66 = !DILocalVariable(name: "__nv_MAIN__F1L51_1Arg0", arg: 1, scope: !63, file: !3, type: !9)
!67 = !DILocation(line: 0, scope: !63)
!68 = !DILocalVariable(name: "__nv_MAIN__F1L51_1Arg1", arg: 2, scope: !63, file: !3, type: !26)
!69 = !DILocalVariable(name: "__nv_MAIN__F1L51_1Arg2", arg: 3, scope: !63, file: !3, type: !26)
!70 = !DILocalVariable(name: "omp_sched_static", scope: !63, file: !3, type: !9)
!71 = !DILocalVariable(name: "omp_sched_dynamic", scope: !63, file: !3, type: !9)
!72 = !DILocalVariable(name: "omp_proc_bind_false", scope: !63, file: !3, type: !9)
!73 = !DILocalVariable(name: "omp_proc_bind_true", scope: !63, file: !3, type: !9)
!74 = !DILocalVariable(name: "omp_proc_bind_master", scope: !63, file: !3, type: !9)
!75 = !DILocalVariable(name: "omp_lock_hint_none", scope: !63, file: !3, type: !9)
!76 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !63, file: !3, type: !9)
!77 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !63, file: !3, type: !9)
!78 = !DILocation(line: 54, column: 1, scope: !63)
!79 = !DILocation(line: 52, column: 1, scope: !63)
!80 = !DILocalVariable(name: "i", scope: !63, file: !3, type: !9)
!81 = !DILocation(line: 53, column: 1, scope: !63)

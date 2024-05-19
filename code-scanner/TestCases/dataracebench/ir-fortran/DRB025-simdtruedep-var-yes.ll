; ModuleID = '/tmp/DRB025-simdtruedep-var-yes-1beb61.ll'
source_filename = "/tmp/DRB025-simdtruedep-var-yes-1beb61.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.C364_MAIN_ = internal constant [26 x i8] c"Values for i and a(i) are:"
@.C363_MAIN_ = internal constant i32 59
@.C379_MAIN_ = internal constant i64 25
@.C355_MAIN_ = internal constant [29 x i8] c"Error, invalid integer value."
@.C354_MAIN_ = internal constant i32 41
@.C350_MAIN_ = internal constant [5 x i8] c"(i10)"
@.C349_MAIN_ = internal constant i32 39
@.C372_MAIN_ = internal constant i64 4
@.C347_MAIN_ = internal constant [37 x i8] c"Allocation error, program terminated."
@.C346_MAIN_ = internal constant i32 30
@.C376_MAIN_ = internal constant i64 80
@.C375_MAIN_ = internal constant i64 14
@.C338_MAIN_ = internal constant [35 x i8] c"No command line arguments provided."
@.C336_MAIN_ = internal constant i32 6
@.C337_MAIN_ = internal constant [3 x i8] c"(a)"
@.C285_MAIN_ = internal constant i32 1
@.C305_MAIN_ = internal constant i32 14
@.C334_MAIN_ = internal constant [55 x i8] c"micro-benchmarks-fortran/DRB025-simdtruedep-var-yes.f95"
@.C306_MAIN_ = internal constant i32 25
@.C330_MAIN_ = internal constant i32 100
@.C286_MAIN_ = internal constant i64 1
@.C284_MAIN_ = internal constant i64 0
@.C283_MAIN_ = internal constant i32 0

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %.Z0982_357 = alloca i32*, align 8
  %"b$sd3_380" = alloca [16 x i64], align 8
  %.Z0976_356 = alloca i32*, align 8
  %"a$sd2_378" = alloca [16 x i64], align 8
  %.Z0970_345 = alloca [80 x i8]*, align 8
  %"args$sd1_374" = alloca [16 x i64], align 8
  %len_331 = alloca i32, align 4
  %argcount_308 = alloca i32, align 4
  %z__io_340 = alloca i32, align 4
  %z_b_0_312 = alloca i64, align 8
  %z_b_1_313 = alloca i64, align 8
  %z_e_61_316 = alloca i64, align 8
  %z_b_2_314 = alloca i64, align 8
  %z_b_3_315 = alloca i64, align 8
  %allocstatus_309 = alloca i32, align 4
  %.dY0001_390 = alloca i32, align 4
  %ix_311 = alloca i32, align 4
  %rderr_310 = alloca i32, align 4
  %z_b_4_318 = alloca i64, align 8
  %z_b_5_319 = alloca i64, align 8
  %z_e_68_322 = alloca i64, align 8
  %z_b_6_320 = alloca i64, align 8
  %z_b_7_321 = alloca i64, align 8
  %z_b_8_324 = alloca i64, align 8
  %z_b_9_325 = alloca i64, align 8
  %z_e_75_328 = alloca i64, align 8
  %z_b_10_326 = alloca i64, align 8
  %z_b_11_327 = alloca i64, align 8
  %.dY0002_395 = alloca i32, align 4
  %i_307 = alloca i32, align 4
  %.i0000_361 = alloca i32, align 4
  %.dY0003_398 = alloca i32, align 4
  %i_360 = alloca i32, align 4
  %.dY0004_401 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !14, metadata !DIExpression()), !dbg !10
  %0 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !15
  %1 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !15
  call void (i8*, ...) %1(i8* %0), !dbg !15
  call void @llvm.dbg.declare(metadata i32** %.Z0982_357, metadata !16, metadata !DIExpression(DW_OP_deref)), !dbg !10
  %2 = bitcast i32** %.Z0982_357 to i8**, !dbg !15
  store i8* null, i8** %2, align 8, !dbg !15
  call void @llvm.dbg.declare(metadata [16 x i64]* %"b$sd3_380", metadata !20, metadata !DIExpression()), !dbg !10
  %3 = bitcast [16 x i64]* %"b$sd3_380" to i64*, !dbg !15
  store i64 0, i64* %3, align 8, !dbg !15
  call void @llvm.dbg.declare(metadata i32** %.Z0976_356, metadata !25, metadata !DIExpression(DW_OP_deref)), !dbg !10
  %4 = bitcast i32** %.Z0976_356 to i8**, !dbg !15
  store i8* null, i8** %4, align 8, !dbg !15
  call void @llvm.dbg.declare(metadata [16 x i64]* %"a$sd2_378", metadata !20, metadata !DIExpression()), !dbg !10
  %5 = bitcast [16 x i64]* %"a$sd2_378" to i64*, !dbg !15
  store i64 0, i64* %5, align 8, !dbg !15
  call void @llvm.dbg.declare(metadata [80 x i8]** %.Z0970_345, metadata !26, metadata !DIExpression(DW_OP_deref)), !dbg !10
  %6 = bitcast [80 x i8]** %.Z0970_345 to i8**, !dbg !15
  store i8* null, i8** %6, align 8, !dbg !15
  call void @llvm.dbg.declare(metadata [16 x i64]* %"args$sd1_374", metadata !20, metadata !DIExpression()), !dbg !10
  %7 = bitcast [16 x i64]* %"args$sd1_374" to i64*, !dbg !15
  store i64 0, i64* %7, align 8, !dbg !15
  br label %L.LB1_413

L.LB1_413:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %len_331, metadata !29, metadata !DIExpression()), !dbg !10
  store i32 100, i32* %len_331, align 4, !dbg !30
  %8 = call i32 (...) @f90_cmd_arg_cnt(), !dbg !31
  call void @llvm.dbg.declare(metadata i32* %argcount_308, metadata !32, metadata !DIExpression()), !dbg !10
  store i32 %8, i32* %argcount_308, align 4, !dbg !31
  %9 = load i32, i32* %argcount_308, align 4, !dbg !33
  call void @llvm.dbg.value(metadata i32 %9, metadata !32, metadata !DIExpression()), !dbg !10
  %10 = icmp ne i32 %9, 0, !dbg !33
  br i1 %10, label %L.LB1_384, label %L.LB1_488, !dbg !33

L.LB1_488:                                        ; preds = %L.LB1_413
  call void (...) @_mp_bcs_nest(), !dbg !34
  %11 = bitcast i32* @.C306_MAIN_ to i8*, !dbg !34
  %12 = bitcast [55 x i8]* @.C334_MAIN_ to i8*, !dbg !34
  %13 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !34
  call void (i8*, i8*, i64, ...) %13(i8* %11, i8* %12, i64 55), !dbg !34
  %14 = bitcast i32* @.C305_MAIN_ to i8*, !dbg !34
  %15 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !34
  %16 = bitcast [3 x i8]* @.C337_MAIN_ to i8*, !dbg !34
  %17 = bitcast i32 (...)* @f90io_encode_fmta to i32 (i8*, i8*, i8*, i64, ...)*, !dbg !34
  %18 = call i32 (i8*, i8*, i8*, i64, ...) %17(i8* %14, i8* %15, i8* %16, i64 3), !dbg !34
  call void @llvm.dbg.declare(metadata i32* %z__io_340, metadata !35, metadata !DIExpression()), !dbg !10
  store i32 %18, i32* %z__io_340, align 4, !dbg !34
  %19 = bitcast i32* @.C336_MAIN_ to i8*, !dbg !34
  %20 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !34
  %21 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !34
  %22 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !34
  %23 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %22(i8* %19, i8* null, i8* %20, i8* %21, i8* null, i8* null, i64 0), !dbg !34
  store i32 %23, i32* %z__io_340, align 4, !dbg !34
  %24 = bitcast i32* @.C305_MAIN_ to i8*, !dbg !34
  %25 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !34
  %26 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !34
  %27 = bitcast [35 x i8]* @.C338_MAIN_ to i8*, !dbg !34
  %28 = bitcast i32 (...)* @f90io_fmt_writea to i32 (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !34
  %29 = call i32 (i8*, i8*, i8*, i8*, i64, ...) %28(i8* %24, i8* %25, i8* %26, i8* %27, i64 35), !dbg !34
  store i32 %29, i32* %z__io_340, align 4, !dbg !34
  %30 = call i32 (...) @f90io_fmtw_end(), !dbg !34
  store i32 %30, i32* %z__io_340, align 4, !dbg !34
  call void (...) @_mp_ecs_nest(), !dbg !34
  br label %L.LB1_384

L.LB1_384:                                        ; preds = %L.LB1_488, %L.LB1_413
  call void @llvm.dbg.declare(metadata i64* %z_b_0_312, metadata !36, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_0_312, align 8, !dbg !37
  %31 = load i32, i32* %argcount_308, align 4, !dbg !37
  call void @llvm.dbg.value(metadata i32 %31, metadata !32, metadata !DIExpression()), !dbg !10
  %32 = sext i32 %31 to i64, !dbg !37
  call void @llvm.dbg.declare(metadata i64* %z_b_1_313, metadata !36, metadata !DIExpression()), !dbg !10
  store i64 %32, i64* %z_b_1_313, align 8, !dbg !37
  %33 = load i64, i64* %z_b_1_313, align 8, !dbg !37
  call void @llvm.dbg.value(metadata i64 %33, metadata !36, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_61_316, metadata !36, metadata !DIExpression()), !dbg !10
  store i64 %33, i64* %z_e_61_316, align 8, !dbg !37
  %34 = bitcast [16 x i64]* %"args$sd1_374" to i8*, !dbg !37
  %35 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !37
  %36 = bitcast i64* @.C375_MAIN_ to i8*, !dbg !37
  %37 = bitcast i64* @.C376_MAIN_ to i8*, !dbg !37
  %38 = bitcast i64* %z_b_0_312 to i8*, !dbg !37
  %39 = bitcast i64* %z_b_1_313 to i8*, !dbg !37
  %40 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !37
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %40(i8* %34, i8* %35, i8* %36, i8* %37, i8* %38, i8* %39), !dbg !37
  %41 = bitcast [16 x i64]* %"args$sd1_374" to i8*, !dbg !37
  %42 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !37
  call void (i8*, i32, ...) %42(i8* %41, i32 14), !dbg !37
  %43 = load i64, i64* %z_b_1_313, align 8, !dbg !37
  call void @llvm.dbg.value(metadata i64 %43, metadata !36, metadata !DIExpression()), !dbg !10
  %44 = load i64, i64* %z_b_0_312, align 8, !dbg !37
  call void @llvm.dbg.value(metadata i64 %44, metadata !36, metadata !DIExpression()), !dbg !10
  %45 = sub nsw i64 %44, 1, !dbg !37
  %46 = sub nsw i64 %43, %45, !dbg !37
  call void @llvm.dbg.declare(metadata i64* %z_b_2_314, metadata !36, metadata !DIExpression()), !dbg !10
  store i64 %46, i64* %z_b_2_314, align 8, !dbg !37
  %47 = load i64, i64* %z_b_0_312, align 8, !dbg !37
  call void @llvm.dbg.value(metadata i64 %47, metadata !36, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_b_3_315, metadata !36, metadata !DIExpression()), !dbg !10
  store i64 %47, i64* %z_b_3_315, align 8, !dbg !37
  %48 = bitcast i64* %z_b_2_314 to i8*, !dbg !37
  %49 = bitcast i64* @.C375_MAIN_ to i8*, !dbg !37
  %50 = bitcast i64* @.C376_MAIN_ to i8*, !dbg !37
  call void @llvm.dbg.declare(metadata i32* %allocstatus_309, metadata !38, metadata !DIExpression()), !dbg !10
  %51 = bitcast i32* %allocstatus_309 to i8*, !dbg !37
  %52 = bitcast [80 x i8]** %.Z0970_345 to i8*, !dbg !37
  %53 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !37
  %54 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !37
  %55 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !37
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %55(i8* %48, i8* %49, i8* %50, i8* %51, i8* %52, i8* null, i8* %53, i8* %54, i8* null, i64 0), !dbg !37
  %56 = load i32, i32* %allocstatus_309, align 4, !dbg !39
  call void @llvm.dbg.value(metadata i32 %56, metadata !38, metadata !DIExpression()), !dbg !10
  %57 = icmp sle i32 %56, 0, !dbg !39
  br i1 %57, label %L.LB1_387, label %L.LB1_489, !dbg !39

L.LB1_489:                                        ; preds = %L.LB1_384
  call void (...) @_mp_bcs_nest(), !dbg !40
  %58 = bitcast i32* @.C346_MAIN_ to i8*, !dbg !40
  %59 = bitcast [55 x i8]* @.C334_MAIN_ to i8*, !dbg !40
  %60 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !40
  call void (i8*, i8*, i64, ...) %60(i8* %58, i8* %59, i64 55), !dbg !40
  %61 = bitcast i32* @.C305_MAIN_ to i8*, !dbg !40
  %62 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !40
  %63 = bitcast [3 x i8]* @.C337_MAIN_ to i8*, !dbg !40
  %64 = bitcast i32 (...)* @f90io_encode_fmta to i32 (i8*, i8*, i8*, i64, ...)*, !dbg !40
  %65 = call i32 (i8*, i8*, i8*, i64, ...) %64(i8* %61, i8* %62, i8* %63, i64 3), !dbg !40
  store i32 %65, i32* %z__io_340, align 4, !dbg !40
  %66 = bitcast i32* @.C336_MAIN_ to i8*, !dbg !40
  %67 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !40
  %68 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !40
  %69 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !40
  %70 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %69(i8* %66, i8* null, i8* %67, i8* %68, i8* null, i8* null, i64 0), !dbg !40
  store i32 %70, i32* %z__io_340, align 4, !dbg !40
  %71 = bitcast i32* @.C305_MAIN_ to i8*, !dbg !40
  %72 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !40
  %73 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !40
  %74 = bitcast [37 x i8]* @.C347_MAIN_ to i8*, !dbg !40
  %75 = bitcast i32 (...)* @f90io_fmt_writea to i32 (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !40
  %76 = call i32 (i8*, i8*, i8*, i8*, i64, ...) %75(i8* %71, i8* %72, i8* %73, i8* %74, i64 37), !dbg !40
  store i32 %76, i32* %z__io_340, align 4, !dbg !40
  %77 = call i32 (...) @f90io_fmtw_end(), !dbg !40
  store i32 %77, i32* %z__io_340, align 4, !dbg !40
  call void (...) @_mp_ecs_nest(), !dbg !40
  %78 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !41
  %79 = bitcast void (...)* @f90_stop08a to void (i8*, i8*, i64, ...)*, !dbg !41
  call void (i8*, i8*, i64, ...) %79(i8* %78, i8* null, i64 0), !dbg !41
  br label %L.LB1_387

L.LB1_387:                                        ; preds = %L.LB1_489, %L.LB1_384
  %80 = load i32, i32* %argcount_308, align 4, !dbg !42
  call void @llvm.dbg.value(metadata i32 %80, metadata !32, metadata !DIExpression()), !dbg !10
  store i32 %80, i32* %.dY0001_390, align 4, !dbg !42
  call void @llvm.dbg.declare(metadata i32* %ix_311, metadata !43, metadata !DIExpression()), !dbg !10
  store i32 1, i32* %ix_311, align 4, !dbg !42
  %81 = load i32, i32* %.dY0001_390, align 4, !dbg !42
  %82 = icmp sle i32 %81, 0, !dbg !42
  br i1 %82, label %L.LB1_389, label %L.LB1_388, !dbg !42

L.LB1_388:                                        ; preds = %L.LB1_388, %L.LB1_387
  %83 = bitcast i32* %ix_311 to i8*, !dbg !44
  %84 = load [80 x i8]*, [80 x i8]** %.Z0970_345, align 8, !dbg !44
  call void @llvm.dbg.value(metadata [80 x i8]* %84, metadata !26, metadata !DIExpression()), !dbg !10
  %85 = bitcast [80 x i8]* %84 to i8*, !dbg !44
  %86 = getelementptr i8, i8* %85, i64 -80, !dbg !44
  %87 = load i32, i32* %ix_311, align 4, !dbg !44
  call void @llvm.dbg.value(metadata i32 %87, metadata !43, metadata !DIExpression()), !dbg !10
  %88 = sext i32 %87 to i64, !dbg !44
  %89 = bitcast [16 x i64]* %"args$sd1_374" to i8*, !dbg !44
  %90 = getelementptr i8, i8* %89, i64 56, !dbg !44
  %91 = bitcast i8* %90 to i64*, !dbg !44
  %92 = load i64, i64* %91, align 8, !dbg !44
  %93 = add nsw i64 %88, %92, !dbg !44
  %94 = mul nsw i64 %93, 80, !dbg !44
  %95 = getelementptr i8, i8* %86, i64 %94, !dbg !44
  %96 = bitcast i64* @.C372_MAIN_ to i8*, !dbg !44
  %97 = bitcast void (...)* @f90_get_cmd_arga to void (i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !44
  call void (i8*, i8*, i8*, i8*, i8*, i64, ...) %97(i8* %83, i8* %95, i8* null, i8* null, i8* %96, i64 80), !dbg !44
  %98 = load i32, i32* %ix_311, align 4, !dbg !45
  call void @llvm.dbg.value(metadata i32 %98, metadata !43, metadata !DIExpression()), !dbg !10
  %99 = add nsw i32 %98, 1, !dbg !45
  store i32 %99, i32* %ix_311, align 4, !dbg !45
  %100 = load i32, i32* %.dY0001_390, align 4, !dbg !45
  %101 = sub nsw i32 %100, 1, !dbg !45
  store i32 %101, i32* %.dY0001_390, align 4, !dbg !45
  %102 = load i32, i32* %.dY0001_390, align 4, !dbg !45
  %103 = icmp sgt i32 %102, 0, !dbg !45
  br i1 %103, label %L.LB1_388, label %L.LB1_389, !dbg !45

L.LB1_389:                                        ; preds = %L.LB1_388, %L.LB1_387
  %104 = load i32, i32* %argcount_308, align 4, !dbg !46
  call void @llvm.dbg.value(metadata i32 %104, metadata !32, metadata !DIExpression()), !dbg !10
  %105 = icmp sle i32 %104, 0, !dbg !46
  br i1 %105, label %L.LB1_391, label %L.LB1_490, !dbg !46

L.LB1_490:                                        ; preds = %L.LB1_389
  call void (...) @_mp_bcs_nest(), !dbg !47
  %106 = bitcast i32* @.C349_MAIN_ to i8*, !dbg !47
  %107 = bitcast [55 x i8]* @.C334_MAIN_ to i8*, !dbg !47
  %108 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !47
  call void (i8*, i8*, i64, ...) %108(i8* %106, i8* %107, i64 55), !dbg !47
  %109 = bitcast i32* @.C305_MAIN_ to i8*, !dbg !47
  %110 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !47
  %111 = bitcast [5 x i8]* @.C350_MAIN_ to i8*, !dbg !47
  %112 = bitcast i32 (...)* @f90io_encode_fmta to i32 (i8*, i8*, i8*, i64, ...)*, !dbg !47
  %113 = call i32 (i8*, i8*, i8*, i64, ...) %112(i8* %109, i8* %110, i8* %111, i64 5), !dbg !47
  store i32 %113, i32* %z__io_340, align 4, !dbg !47
  %114 = load [80 x i8]*, [80 x i8]** %.Z0970_345, align 8, !dbg !47
  call void @llvm.dbg.value(metadata [80 x i8]* %114, metadata !26, metadata !DIExpression()), !dbg !10
  %115 = bitcast [80 x i8]* %114 to i8*, !dbg !47
  %116 = bitcast [16 x i64]* %"args$sd1_374" to i8*, !dbg !47
  %117 = getelementptr i8, i8* %116, i64 56, !dbg !47
  %118 = bitcast i8* %117 to i64*, !dbg !47
  %119 = load i64, i64* %118, align 8, !dbg !47
  %120 = mul nsw i64 %119, 80, !dbg !47
  %121 = getelementptr i8, i8* %115, i64 %120, !dbg !47
  %122 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !47
  %123 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !47
  call void @llvm.dbg.declare(metadata i32* %rderr_310, metadata !48, metadata !DIExpression()), !dbg !10
  %124 = bitcast i32* %rderr_310 to i8*, !dbg !47
  %125 = bitcast i32 (...)* @f90io_fmtr_intern_inita to i32 (i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !47
  %126 = call i32 (i8*, i8*, i8*, i8*, i8*, i64, ...) %125(i8* %121, i8* %122, i8* %123, i8* %124, i8* null, i64 80), !dbg !47
  store i32 %126, i32* %z__io_340, align 4, !dbg !47
  %127 = bitcast i32* @.C306_MAIN_ to i8*, !dbg !47
  %128 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !47
  %129 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !47
  %130 = bitcast i32* %len_331 to i8*, !dbg !47
  %131 = bitcast i32 (...)* @f90io_fmt_reada to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !47
  %132 = call i32 (i8*, i8*, i8*, i8*, ...) %131(i8* %127, i8* %128, i8* %129, i8* %130), !dbg !47
  store i32 %132, i32* %z__io_340, align 4, !dbg !47
  %133 = call i32 (...) @f90io_fmtr_end(), !dbg !47
  store i32 %133, i32* %z__io_340, align 4, !dbg !47
  call void (...) @_mp_ecs_nest(), !dbg !47
  %134 = load i32, i32* %rderr_310, align 4, !dbg !49
  call void @llvm.dbg.value(metadata i32 %134, metadata !48, metadata !DIExpression()), !dbg !10
  %135 = icmp eq i32 %134, 0, !dbg !49
  br i1 %135, label %L.LB1_392, label %L.LB1_491, !dbg !49

L.LB1_491:                                        ; preds = %L.LB1_490
  call void (...) @_mp_bcs_nest(), !dbg !50
  %136 = bitcast i32* @.C354_MAIN_ to i8*, !dbg !50
  %137 = bitcast [55 x i8]* @.C334_MAIN_ to i8*, !dbg !50
  %138 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !50
  call void (i8*, i8*, i64, ...) %138(i8* %136, i8* %137, i64 55), !dbg !50
  %139 = bitcast i32* @.C305_MAIN_ to i8*, !dbg !50
  %140 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !50
  %141 = bitcast [3 x i8]* @.C337_MAIN_ to i8*, !dbg !50
  %142 = bitcast i32 (...)* @f90io_encode_fmta to i32 (i8*, i8*, i8*, i64, ...)*, !dbg !50
  %143 = call i32 (i8*, i8*, i8*, i64, ...) %142(i8* %139, i8* %140, i8* %141, i64 3), !dbg !50
  store i32 %143, i32* %z__io_340, align 4, !dbg !50
  %144 = bitcast i32* @.C336_MAIN_ to i8*, !dbg !50
  %145 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !50
  %146 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !50
  %147 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !50
  %148 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %147(i8* %144, i8* null, i8* %145, i8* %146, i8* null, i8* null, i64 0), !dbg !50
  store i32 %148, i32* %z__io_340, align 4, !dbg !50
  %149 = bitcast i32* @.C305_MAIN_ to i8*, !dbg !50
  %150 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !50
  %151 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !50
  %152 = bitcast [29 x i8]* @.C355_MAIN_ to i8*, !dbg !50
  %153 = bitcast i32 (...)* @f90io_fmt_writea to i32 (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !50
  %154 = call i32 (i8*, i8*, i8*, i8*, i64, ...) %153(i8* %149, i8* %150, i8* %151, i8* %152, i64 29), !dbg !50
  store i32 %154, i32* %z__io_340, align 4, !dbg !50
  %155 = call i32 (...) @f90io_fmtw_end(), !dbg !50
  store i32 %155, i32* %z__io_340, align 4, !dbg !50
  call void (...) @_mp_ecs_nest(), !dbg !50
  br label %L.LB1_392

L.LB1_392:                                        ; preds = %L.LB1_491, %L.LB1_490
  br label %L.LB1_391

L.LB1_391:                                        ; preds = %L.LB1_392, %L.LB1_389
  call void @llvm.dbg.declare(metadata i64* %z_b_4_318, metadata !36, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_4_318, align 8, !dbg !51
  %156 = load i32, i32* %len_331, align 4, !dbg !51
  call void @llvm.dbg.value(metadata i32 %156, metadata !29, metadata !DIExpression()), !dbg !10
  %157 = sext i32 %156 to i64, !dbg !51
  call void @llvm.dbg.declare(metadata i64* %z_b_5_319, metadata !36, metadata !DIExpression()), !dbg !10
  store i64 %157, i64* %z_b_5_319, align 8, !dbg !51
  %158 = load i64, i64* %z_b_5_319, align 8, !dbg !51
  call void @llvm.dbg.value(metadata i64 %158, metadata !36, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_68_322, metadata !36, metadata !DIExpression()), !dbg !10
  store i64 %158, i64* %z_e_68_322, align 8, !dbg !51
  %159 = bitcast [16 x i64]* %"a$sd2_378" to i8*, !dbg !51
  %160 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !51
  %161 = bitcast i64* @.C379_MAIN_ to i8*, !dbg !51
  %162 = bitcast i64* @.C372_MAIN_ to i8*, !dbg !51
  %163 = bitcast i64* %z_b_4_318 to i8*, !dbg !51
  %164 = bitcast i64* %z_b_5_319 to i8*, !dbg !51
  %165 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !51
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %165(i8* %159, i8* %160, i8* %161, i8* %162, i8* %163, i8* %164), !dbg !51
  %166 = bitcast [16 x i64]* %"a$sd2_378" to i8*, !dbg !51
  %167 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !51
  call void (i8*, i32, ...) %167(i8* %166, i32 25), !dbg !51
  %168 = load i64, i64* %z_b_5_319, align 8, !dbg !51
  call void @llvm.dbg.value(metadata i64 %168, metadata !36, metadata !DIExpression()), !dbg !10
  %169 = load i64, i64* %z_b_4_318, align 8, !dbg !51
  call void @llvm.dbg.value(metadata i64 %169, metadata !36, metadata !DIExpression()), !dbg !10
  %170 = sub nsw i64 %169, 1, !dbg !51
  %171 = sub nsw i64 %168, %170, !dbg !51
  call void @llvm.dbg.declare(metadata i64* %z_b_6_320, metadata !36, metadata !DIExpression()), !dbg !10
  store i64 %171, i64* %z_b_6_320, align 8, !dbg !51
  %172 = load i64, i64* %z_b_4_318, align 8, !dbg !51
  call void @llvm.dbg.value(metadata i64 %172, metadata !36, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_b_7_321, metadata !36, metadata !DIExpression()), !dbg !10
  store i64 %172, i64* %z_b_7_321, align 8, !dbg !51
  %173 = bitcast i64* %z_b_6_320 to i8*, !dbg !51
  %174 = bitcast i64* @.C379_MAIN_ to i8*, !dbg !51
  %175 = bitcast i64* @.C372_MAIN_ to i8*, !dbg !51
  %176 = bitcast i32** %.Z0976_356 to i8*, !dbg !51
  %177 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !51
  %178 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !51
  %179 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !51
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %179(i8* %173, i8* %174, i8* %175, i8* null, i8* %176, i8* null, i8* %177, i8* %178, i8* null, i64 0), !dbg !51
  call void @llvm.dbg.declare(metadata i64* %z_b_8_324, metadata !36, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_8_324, align 8, !dbg !52
  %180 = load i32, i32* %len_331, align 4, !dbg !52
  call void @llvm.dbg.value(metadata i32 %180, metadata !29, metadata !DIExpression()), !dbg !10
  %181 = sext i32 %180 to i64, !dbg !52
  call void @llvm.dbg.declare(metadata i64* %z_b_9_325, metadata !36, metadata !DIExpression()), !dbg !10
  store i64 %181, i64* %z_b_9_325, align 8, !dbg !52
  %182 = load i64, i64* %z_b_9_325, align 8, !dbg !52
  call void @llvm.dbg.value(metadata i64 %182, metadata !36, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_75_328, metadata !36, metadata !DIExpression()), !dbg !10
  store i64 %182, i64* %z_e_75_328, align 8, !dbg !52
  %183 = bitcast [16 x i64]* %"b$sd3_380" to i8*, !dbg !52
  %184 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !52
  %185 = bitcast i64* @.C379_MAIN_ to i8*, !dbg !52
  %186 = bitcast i64* @.C372_MAIN_ to i8*, !dbg !52
  %187 = bitcast i64* %z_b_8_324 to i8*, !dbg !52
  %188 = bitcast i64* %z_b_9_325 to i8*, !dbg !52
  %189 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !52
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %189(i8* %183, i8* %184, i8* %185, i8* %186, i8* %187, i8* %188), !dbg !52
  %190 = bitcast [16 x i64]* %"b$sd3_380" to i8*, !dbg !52
  %191 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !52
  call void (i8*, i32, ...) %191(i8* %190, i32 25), !dbg !52
  %192 = load i64, i64* %z_b_9_325, align 8, !dbg !52
  call void @llvm.dbg.value(metadata i64 %192, metadata !36, metadata !DIExpression()), !dbg !10
  %193 = load i64, i64* %z_b_8_324, align 8, !dbg !52
  call void @llvm.dbg.value(metadata i64 %193, metadata !36, metadata !DIExpression()), !dbg !10
  %194 = sub nsw i64 %193, 1, !dbg !52
  %195 = sub nsw i64 %192, %194, !dbg !52
  call void @llvm.dbg.declare(metadata i64* %z_b_10_326, metadata !36, metadata !DIExpression()), !dbg !10
  store i64 %195, i64* %z_b_10_326, align 8, !dbg !52
  %196 = load i64, i64* %z_b_8_324, align 8, !dbg !52
  call void @llvm.dbg.value(metadata i64 %196, metadata !36, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_b_11_327, metadata !36, metadata !DIExpression()), !dbg !10
  store i64 %196, i64* %z_b_11_327, align 8, !dbg !52
  %197 = bitcast i64* %z_b_10_326 to i8*, !dbg !52
  %198 = bitcast i64* @.C379_MAIN_ to i8*, !dbg !52
  %199 = bitcast i64* @.C372_MAIN_ to i8*, !dbg !52
  %200 = bitcast i32** %.Z0982_357 to i8*, !dbg !52
  %201 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !52
  %202 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !52
  %203 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !52
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %203(i8* %197, i8* %198, i8* %199, i8* null, i8* %200, i8* null, i8* %201, i8* %202, i8* null, i64 0), !dbg !52
  %204 = load i32, i32* %len_331, align 4, !dbg !53
  call void @llvm.dbg.value(metadata i32 %204, metadata !29, metadata !DIExpression()), !dbg !10
  store i32 %204, i32* %.dY0002_395, align 4, !dbg !53
  call void @llvm.dbg.declare(metadata i32* %i_307, metadata !54, metadata !DIExpression()), !dbg !10
  store i32 1, i32* %i_307, align 4, !dbg !53
  %205 = load i32, i32* %.dY0002_395, align 4, !dbg !53
  %206 = icmp sle i32 %205, 0, !dbg !53
  br i1 %206, label %L.LB1_394, label %L.LB1_393, !dbg !53

L.LB1_393:                                        ; preds = %L.LB1_393, %L.LB1_391
  %207 = load i32, i32* %i_307, align 4, !dbg !55
  call void @llvm.dbg.value(metadata i32 %207, metadata !54, metadata !DIExpression()), !dbg !10
  %208 = load i32, i32* %i_307, align 4, !dbg !55
  call void @llvm.dbg.value(metadata i32 %208, metadata !54, metadata !DIExpression()), !dbg !10
  %209 = sext i32 %208 to i64, !dbg !55
  %210 = bitcast [16 x i64]* %"a$sd2_378" to i8*, !dbg !55
  %211 = getelementptr i8, i8* %210, i64 56, !dbg !55
  %212 = bitcast i8* %211 to i64*, !dbg !55
  %213 = load i64, i64* %212, align 8, !dbg !55
  %214 = add nsw i64 %209, %213, !dbg !55
  %215 = load i32*, i32** %.Z0976_356, align 8, !dbg !55
  call void @llvm.dbg.value(metadata i32* %215, metadata !25, metadata !DIExpression()), !dbg !10
  %216 = bitcast i32* %215 to i8*, !dbg !55
  %217 = getelementptr i8, i8* %216, i64 -4, !dbg !55
  %218 = bitcast i8* %217 to i32*, !dbg !55
  %219 = getelementptr i32, i32* %218, i64 %214, !dbg !55
  store i32 %207, i32* %219, align 4, !dbg !55
  %220 = load i32, i32* %i_307, align 4, !dbg !56
  call void @llvm.dbg.value(metadata i32 %220, metadata !54, metadata !DIExpression()), !dbg !10
  %221 = add nsw i32 %220, 1, !dbg !56
  %222 = load i32, i32* %i_307, align 4, !dbg !56
  call void @llvm.dbg.value(metadata i32 %222, metadata !54, metadata !DIExpression()), !dbg !10
  %223 = sext i32 %222 to i64, !dbg !56
  %224 = bitcast [16 x i64]* %"b$sd3_380" to i8*, !dbg !56
  %225 = getelementptr i8, i8* %224, i64 56, !dbg !56
  %226 = bitcast i8* %225 to i64*, !dbg !56
  %227 = load i64, i64* %226, align 8, !dbg !56
  %228 = add nsw i64 %223, %227, !dbg !56
  %229 = load i32*, i32** %.Z0982_357, align 8, !dbg !56
  call void @llvm.dbg.value(metadata i32* %229, metadata !16, metadata !DIExpression()), !dbg !10
  %230 = bitcast i32* %229 to i8*, !dbg !56
  %231 = getelementptr i8, i8* %230, i64 -4, !dbg !56
  %232 = bitcast i8* %231 to i32*, !dbg !56
  %233 = getelementptr i32, i32* %232, i64 %228, !dbg !56
  store i32 %221, i32* %233, align 4, !dbg !56
  %234 = load i32, i32* %i_307, align 4, !dbg !57
  call void @llvm.dbg.value(metadata i32 %234, metadata !54, metadata !DIExpression()), !dbg !10
  %235 = add nsw i32 %234, 1, !dbg !57
  store i32 %235, i32* %i_307, align 4, !dbg !57
  %236 = load i32, i32* %.dY0002_395, align 4, !dbg !57
  %237 = sub nsw i32 %236, 1, !dbg !57
  store i32 %237, i32* %.dY0002_395, align 4, !dbg !57
  %238 = load i32, i32* %.dY0002_395, align 4, !dbg !57
  %239 = icmp sgt i32 %238, 0, !dbg !57
  br i1 %239, label %L.LB1_393, label %L.LB1_394, !dbg !57

L.LB1_394:                                        ; preds = %L.LB1_393, %L.LB1_391
  br label %L.LB1_359

L.LB1_359:                                        ; preds = %L.LB1_394
  %240 = load i32, i32* %len_331, align 4, !dbg !58
  call void @llvm.dbg.value(metadata i32 %240, metadata !29, metadata !DIExpression()), !dbg !10
  store i32 %240, i32* %.i0000_361, align 4, !dbg !58
  %241 = load i32, i32* %len_331, align 4, !dbg !58
  call void @llvm.dbg.value(metadata i32 %241, metadata !29, metadata !DIExpression()), !dbg !10
  %242 = sub nsw i32 %241, 1, !dbg !58
  store i32 %242, i32* %.dY0003_398, align 4, !dbg !58
  call void @llvm.dbg.declare(metadata i32* %i_360, metadata !54, metadata !DIExpression()), !dbg !59
  store i32 1, i32* %i_360, align 4, !dbg !58
  %243 = load i32, i32* %.dY0003_398, align 4, !dbg !58
  %244 = icmp sle i32 %243, 0, !dbg !58
  br i1 %244, label %L.LB1_397, label %L.LB1_396, !dbg !58

L.LB1_396:                                        ; preds = %L.LB1_396, %L.LB1_359
  %245 = bitcast [16 x i64]* %"b$sd3_380" to i8*, !dbg !60
  %246 = getelementptr i8, i8* %245, i64 56, !dbg !60
  %247 = bitcast i8* %246 to i64*, !dbg !60
  %248 = load i64, i64* %247, align 8, !dbg !60
  %249 = load i32, i32* %i_360, align 4, !dbg !60
  call void @llvm.dbg.value(metadata i32 %249, metadata !54, metadata !DIExpression()), !dbg !59
  %250 = sext i32 %249 to i64, !dbg !60
  %251 = add nsw i64 %248, %250, !dbg !60
  %252 = load i32*, i32** %.Z0982_357, align 8, !dbg !60
  call void @llvm.dbg.value(metadata i32* %252, metadata !16, metadata !DIExpression()), !dbg !10
  %253 = bitcast i32* %252 to i8*, !dbg !60
  %254 = getelementptr i8, i8* %253, i64 -4, !dbg !60
  %255 = bitcast i8* %254 to i32*, !dbg !60
  %256 = getelementptr i32, i32* %255, i64 %251, !dbg !60
  %257 = load i32, i32* %256, align 4, !dbg !60
  %258 = bitcast [16 x i64]* %"a$sd2_378" to i8*, !dbg !60
  %259 = getelementptr i8, i8* %258, i64 56, !dbg !60
  %260 = bitcast i8* %259 to i64*, !dbg !60
  %261 = load i64, i64* %260, align 8, !dbg !60
  %262 = load i32, i32* %i_360, align 4, !dbg !60
  call void @llvm.dbg.value(metadata i32 %262, metadata !54, metadata !DIExpression()), !dbg !59
  %263 = sext i32 %262 to i64, !dbg !60
  %264 = add nsw i64 %261, %263, !dbg !60
  %265 = load i32*, i32** %.Z0976_356, align 8, !dbg !60
  call void @llvm.dbg.value(metadata i32* %265, metadata !25, metadata !DIExpression()), !dbg !10
  %266 = bitcast i32* %265 to i8*, !dbg !60
  %267 = getelementptr i8, i8* %266, i64 -4, !dbg !60
  %268 = bitcast i8* %267 to i32*, !dbg !60
  %269 = getelementptr i32, i32* %268, i64 %264, !dbg !60
  %270 = load i32, i32* %269, align 4, !dbg !60
  %271 = add nsw i32 %257, %270, !dbg !60
  %272 = bitcast [16 x i64]* %"a$sd2_378" to i8*, !dbg !60
  %273 = getelementptr i8, i8* %272, i64 56, !dbg !60
  %274 = bitcast i8* %273 to i64*, !dbg !60
  %275 = load i64, i64* %274, align 8, !dbg !60
  %276 = load i32, i32* %i_360, align 4, !dbg !60
  call void @llvm.dbg.value(metadata i32 %276, metadata !54, metadata !DIExpression()), !dbg !59
  %277 = sext i32 %276 to i64, !dbg !60
  %278 = add nsw i64 %275, %277, !dbg !60
  %279 = load i32*, i32** %.Z0976_356, align 8, !dbg !60
  call void @llvm.dbg.value(metadata i32* %279, metadata !25, metadata !DIExpression()), !dbg !10
  %280 = getelementptr i32, i32* %279, i64 %278, !dbg !60
  store i32 %271, i32* %280, align 4, !dbg !60
  %281 = load i32, i32* %i_360, align 4, !dbg !61
  call void @llvm.dbg.value(metadata i32 %281, metadata !54, metadata !DIExpression()), !dbg !59
  %282 = add nsw i32 %281, 1, !dbg !61
  store i32 %282, i32* %i_360, align 4, !dbg !61
  %283 = load i32, i32* %.dY0003_398, align 4, !dbg !61
  %284 = sub nsw i32 %283, 1, !dbg !61
  store i32 %284, i32* %.dY0003_398, align 4, !dbg !61
  %285 = load i32, i32* %.dY0003_398, align 4, !dbg !61
  %286 = icmp sgt i32 %285, 0, !dbg !61
  br i1 %286, label %L.LB1_396, label %L.LB1_397, !dbg !61

L.LB1_397:                                        ; preds = %L.LB1_396, %L.LB1_359
  br label %L.LB1_362

L.LB1_362:                                        ; preds = %L.LB1_397
  %287 = load i32, i32* %len_331, align 4, !dbg !62
  call void @llvm.dbg.value(metadata i32 %287, metadata !29, metadata !DIExpression()), !dbg !10
  store i32 %287, i32* %.dY0004_401, align 4, !dbg !62
  store i32 1, i32* %i_307, align 4, !dbg !62
  %288 = load i32, i32* %.dY0004_401, align 4, !dbg !62
  %289 = icmp sle i32 %288, 0, !dbg !62
  br i1 %289, label %L.LB1_400, label %L.LB1_399, !dbg !62

L.LB1_399:                                        ; preds = %L.LB1_399, %L.LB1_362
  call void (...) @_mp_bcs_nest(), !dbg !63
  %290 = bitcast i32* @.C363_MAIN_ to i8*, !dbg !63
  %291 = bitcast [55 x i8]* @.C334_MAIN_ to i8*, !dbg !63
  %292 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !63
  call void (i8*, i8*, i64, ...) %292(i8* %290, i8* %291, i64 55), !dbg !63
  %293 = bitcast i32* @.C336_MAIN_ to i8*, !dbg !63
  %294 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !63
  %295 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !63
  %296 = bitcast i32 (...)* @f90io_ldw_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !63
  %297 = call i32 (i8*, i8*, i8*, i8*, ...) %296(i8* %293, i8* null, i8* %294, i8* %295), !dbg !63
  store i32 %297, i32* %z__io_340, align 4, !dbg !63
  %298 = bitcast [26 x i8]* @.C364_MAIN_ to i8*, !dbg !63
  %299 = bitcast i32 (...)* @f90io_sc_ch_ldw to i32 (i8*, i32, i64, ...)*, !dbg !63
  %300 = call i32 (i8*, i32, i64, ...) %299(i8* %298, i32 14, i64 26), !dbg !63
  store i32 %300, i32* %z__io_340, align 4, !dbg !63
  %301 = load i32, i32* %i_307, align 4, !dbg !63
  call void @llvm.dbg.value(metadata i32 %301, metadata !54, metadata !DIExpression()), !dbg !10
  %302 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !63
  %303 = call i32 (i32, i32, ...) %302(i32 %301, i32 25), !dbg !63
  store i32 %303, i32* %z__io_340, align 4, !dbg !63
  %304 = load i32, i32* %i_307, align 4, !dbg !63
  call void @llvm.dbg.value(metadata i32 %304, metadata !54, metadata !DIExpression()), !dbg !10
  %305 = sext i32 %304 to i64, !dbg !63
  %306 = bitcast [16 x i64]* %"a$sd2_378" to i8*, !dbg !63
  %307 = getelementptr i8, i8* %306, i64 56, !dbg !63
  %308 = bitcast i8* %307 to i64*, !dbg !63
  %309 = load i64, i64* %308, align 8, !dbg !63
  %310 = add nsw i64 %305, %309, !dbg !63
  %311 = load i32*, i32** %.Z0976_356, align 8, !dbg !63
  call void @llvm.dbg.value(metadata i32* %311, metadata !25, metadata !DIExpression()), !dbg !10
  %312 = bitcast i32* %311 to i8*, !dbg !63
  %313 = getelementptr i8, i8* %312, i64 -4, !dbg !63
  %314 = bitcast i8* %313 to i32*, !dbg !63
  %315 = getelementptr i32, i32* %314, i64 %310, !dbg !63
  %316 = load i32, i32* %315, align 4, !dbg !63
  %317 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !63
  %318 = call i32 (i32, i32, ...) %317(i32 %316, i32 25), !dbg !63
  store i32 %318, i32* %z__io_340, align 4, !dbg !63
  %319 = call i32 (...) @f90io_ldw_end(), !dbg !63
  store i32 %319, i32* %z__io_340, align 4, !dbg !63
  call void (...) @_mp_ecs_nest(), !dbg !63
  %320 = load i32, i32* %i_307, align 4, !dbg !64
  call void @llvm.dbg.value(metadata i32 %320, metadata !54, metadata !DIExpression()), !dbg !10
  %321 = add nsw i32 %320, 1, !dbg !64
  store i32 %321, i32* %i_307, align 4, !dbg !64
  %322 = load i32, i32* %.dY0004_401, align 4, !dbg !64
  %323 = sub nsw i32 %322, 1, !dbg !64
  store i32 %323, i32* %.dY0004_401, align 4, !dbg !64
  %324 = load i32, i32* %.dY0004_401, align 4, !dbg !64
  %325 = icmp sgt i32 %324, 0, !dbg !64
  br i1 %325, label %L.LB1_399, label %L.LB1_400, !dbg !64

L.LB1_400:                                        ; preds = %L.LB1_399, %L.LB1_362
  %326 = load [80 x i8]*, [80 x i8]** %.Z0970_345, align 8, !dbg !65
  call void @llvm.dbg.value(metadata [80 x i8]* %326, metadata !26, metadata !DIExpression()), !dbg !10
  %327 = bitcast [80 x i8]* %326 to i8*, !dbg !65
  %328 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !65
  %329 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, i64, ...)*, !dbg !65
  call void (i8*, i8*, i8*, i8*, i64, i64, ...) %329(i8* null, i8* %327, i8* %328, i8* null, i64 80, i64 0), !dbg !65
  %330 = bitcast [80 x i8]** %.Z0970_345 to i8**, !dbg !65
  store i8* null, i8** %330, align 8, !dbg !65
  %331 = bitcast [16 x i64]* %"args$sd1_374" to i64*, !dbg !65
  store i64 0, i64* %331, align 8, !dbg !65
  %332 = load i32*, i32** %.Z0976_356, align 8, !dbg !65
  call void @llvm.dbg.value(metadata i32* %332, metadata !25, metadata !DIExpression()), !dbg !10
  %333 = bitcast i32* %332 to i8*, !dbg !65
  %334 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !65
  %335 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !65
  call void (i8*, i8*, i8*, i8*, i64, ...) %335(i8* null, i8* %333, i8* %334, i8* null, i64 0), !dbg !65
  %336 = bitcast i32** %.Z0976_356 to i8**, !dbg !65
  store i8* null, i8** %336, align 8, !dbg !65
  %337 = bitcast [16 x i64]* %"a$sd2_378" to i64*, !dbg !65
  store i64 0, i64* %337, align 8, !dbg !65
  %338 = load i32*, i32** %.Z0982_357, align 8, !dbg !65
  call void @llvm.dbg.value(metadata i32* %338, metadata !16, metadata !DIExpression()), !dbg !10
  %339 = bitcast i32* %338 to i8*, !dbg !65
  %340 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !65
  %341 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !65
  call void (i8*, i8*, i8*, i8*, i64, ...) %341(i8* null, i8* %339, i8* %340, i8* null, i64 0), !dbg !65
  %342 = bitcast i32** %.Z0982_357 to i8**, !dbg !65
  store i8* null, i8** %342, align 8, !dbg !65
  %343 = bitcast [16 x i64]* %"b$sd3_380" to i64*, !dbg !65
  store i64 0, i64* %343, align 8, !dbg !65
  ret void, !dbg !59
}

declare void @f90_dealloc03a_i8(...) #0

declare signext i32 @f90io_ldw_end(...) #0

declare signext i32 @f90io_sc_i_ldw(...) #0

declare signext i32 @f90io_sc_ch_ldw(...) #0

declare signext i32 @f90io_ldw_init(...) #0

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

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { "no-frame-pointer-elim-non-leaf" }
attributes #1 = { nounwind readnone speculatable }

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !4, globals: !4, imports: !4)
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB025-simdtruedep-var-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb025_simdtruedep_var_yes", scope: !2, file: !3, line: 12, type: !6, scopeLine: 12, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!6 = !DISubroutineType(cc: DW_CC_program, types: !7)
!7 = !{null}
!8 = !DILocalVariable(name: "omp_sched_static", scope: !5, file: !3, type: !9)
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 0, scope: !5)
!11 = !DILocalVariable(name: "omp_proc_bind_false", scope: !5, file: !3, type: !9)
!12 = !DILocalVariable(name: "omp_proc_bind_true", scope: !5, file: !3, type: !9)
!13 = !DILocalVariable(name: "omp_lock_hint_none", scope: !5, file: !3, type: !9)
!14 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !5, file: !3, type: !9)
!15 = !DILocation(line: 12, column: 1, scope: !5)
!16 = !DILocalVariable(name: "b", scope: !5, file: !3, type: !17)
!17 = !DICompositeType(tag: DW_TAG_array_type, baseType: !9, size: 32, align: 32, elements: !18)
!18 = !{!19}
!19 = !DISubrange(count: 0, lowerBound: 1)
!20 = !DILocalVariable(scope: !5, file: !3, type: !21, flags: DIFlagArtificial)
!21 = !DICompositeType(tag: DW_TAG_array_type, baseType: !22, size: 1024, align: 64, elements: !23)
!22 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!23 = !{!24}
!24 = !DISubrange(count: 16, lowerBound: 1)
!25 = !DILocalVariable(name: "a", scope: !5, file: !3, type: !17)
!26 = !DILocalVariable(name: "args", scope: !5, file: !3, type: !27)
!27 = !DICompositeType(tag: DW_TAG_array_type, baseType: !28, size: 640, align: 8, elements: !18)
!28 = !DIBasicType(name: "character", size: 640, align: 8, encoding: DW_ATE_signed)
!29 = !DILocalVariable(name: "len", scope: !5, file: !3, type: !9)
!30 = !DILocation(line: 21, column: 1, scope: !5)
!31 = !DILocation(line: 23, column: 1, scope: !5)
!32 = !DILocalVariable(name: "argcount", scope: !5, file: !3, type: !9)
!33 = !DILocation(line: 24, column: 1, scope: !5)
!34 = !DILocation(line: 25, column: 1, scope: !5)
!35 = !DILocalVariable(scope: !5, file: !3, type: !9, flags: DIFlagArtificial)
!36 = !DILocalVariable(scope: !5, file: !3, type: !22, flags: DIFlagArtificial)
!37 = !DILocation(line: 28, column: 1, scope: !5)
!38 = !DILocalVariable(name: "allocstatus", scope: !5, file: !3, type: !9)
!39 = !DILocation(line: 29, column: 1, scope: !5)
!40 = !DILocation(line: 30, column: 1, scope: !5)
!41 = !DILocation(line: 31, column: 1, scope: !5)
!42 = !DILocation(line: 34, column: 1, scope: !5)
!43 = !DILocalVariable(name: "ix", scope: !5, file: !3, type: !9)
!44 = !DILocation(line: 35, column: 1, scope: !5)
!45 = !DILocation(line: 36, column: 1, scope: !5)
!46 = !DILocation(line: 38, column: 1, scope: !5)
!47 = !DILocation(line: 39, column: 1, scope: !5)
!48 = !DILocalVariable(name: "rderr", scope: !5, file: !3, type: !9)
!49 = !DILocation(line: 40, column: 1, scope: !5)
!50 = !DILocation(line: 41, column: 1, scope: !5)
!51 = !DILocation(line: 45, column: 1, scope: !5)
!52 = !DILocation(line: 46, column: 1, scope: !5)
!53 = !DILocation(line: 48, column: 1, scope: !5)
!54 = !DILocalVariable(name: "i", scope: !5, file: !3, type: !9)
!55 = !DILocation(line: 49, column: 1, scope: !5)
!56 = !DILocation(line: 50, column: 1, scope: !5)
!57 = !DILocation(line: 51, column: 1, scope: !5)
!58 = !DILocation(line: 54, column: 1, scope: !5)
!59 = !DILocation(line: 64, column: 1, scope: !5)
!60 = !DILocation(line: 55, column: 1, scope: !5)
!61 = !DILocation(line: 56, column: 1, scope: !5)
!62 = !DILocation(line: 58, column: 1, scope: !5)
!63 = !DILocation(line: 59, column: 1, scope: !5)
!64 = !DILocation(line: 60, column: 1, scope: !5)
!65 = !DILocation(line: 62, column: 1, scope: !5)

; ModuleID = '/tmp/DRB036-truedepscalar-var-yes-d5f2c8.ll'
source_filename = "/tmp/DRB036-truedepscalar-var-yes-d5f2c8.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%astruct.dt86 = type <{ i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8* }>

@.C369_MAIN_ = internal constant i64 25
@.C352_MAIN_ = internal constant [29 x i8] c"Error, invalid integer value."
@.C351_MAIN_ = internal constant i32 40
@.C306_MAIN_ = internal constant i32 25
@.C347_MAIN_ = internal constant [5 x i8] c"(i10)"
@.C346_MAIN_ = internal constant i32 38
@.C362_MAIN_ = internal constant i64 4
@.C344_MAIN_ = internal constant [37 x i8] c"Allocation error, program terminated."
@.C343_MAIN_ = internal constant i32 29
@.C366_MAIN_ = internal constant i64 80
@.C365_MAIN_ = internal constant i64 14
@.C335_MAIN_ = internal constant [35 x i8] c"No command line arguments provided."
@.C333_MAIN_ = internal constant i32 6
@.C334_MAIN_ = internal constant [3 x i8] c"(a)"
@.C285_MAIN_ = internal constant i32 1
@.C305_MAIN_ = internal constant i32 14
@.C330_MAIN_ = internal constant [57 x i8] c"micro-benchmarks-fortran/DRB036-truedepscalar-var-yes.f95"
@.C332_MAIN_ = internal constant i32 24
@.C327_MAIN_ = internal constant i32 10
@.C325_MAIN_ = internal constant i32 100
@.C286_MAIN_ = internal constant i64 1
@.C284_MAIN_ = internal constant i64 0
@.C283_MAIN_ = internal constant i32 0
@.C285___nv_MAIN__F1L46_1 = internal constant i32 1
@.C283___nv_MAIN__F1L46_1 = internal constant i32 0

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__485 = alloca i32, align 4
  %.Z0977_353 = alloca i32*, align 8
  %"a$sd2_368" = alloca [16 x i64], align 8
  %.Z0971_342 = alloca [80 x i8]*, align 8
  %"args$sd1_364" = alloca [16 x i64], align 8
  %len_326 = alloca i32, align 4
  %tmp_308 = alloca i32, align 4
  %argcount_309 = alloca i32, align 4
  %z__io_337 = alloca i32, align 4
  %z_b_0_313 = alloca i64, align 8
  %z_b_1_314 = alloca i64, align 8
  %z_e_61_317 = alloca i64, align 8
  %z_b_2_315 = alloca i64, align 8
  %z_b_3_316 = alloca i64, align 8
  %allocstatus_310 = alloca i32, align 4
  %.dY0001_379 = alloca i32, align 4
  %ix_312 = alloca i32, align 4
  %rderr_311 = alloca i32, align 4
  %z_b_4_319 = alloca i64, align 8
  %z_b_5_320 = alloca i64, align 8
  %z_e_68_323 = alloca i64, align 8
  %z_b_6_321 = alloca i64, align 8
  %z_b_7_322 = alloca i64, align 8
  %.uplevelArgPack0001_462 = alloca %astruct.dt86, align 16
  call void @llvm.dbg.value(metadata i32 1, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !14, metadata !DIExpression()), !dbg !10
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !15
  store i32 %0, i32* %__gtid_MAIN__485, align 4, !dbg !15
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !16
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !16
  call void (i8*, ...) %2(i8* %1), !dbg !16
  call void @llvm.dbg.declare(metadata i32** %.Z0977_353, metadata !17, metadata !DIExpression(DW_OP_deref)), !dbg !10
  %3 = bitcast i32** %.Z0977_353 to i8**, !dbg !16
  store i8* null, i8** %3, align 8, !dbg !16
  call void @llvm.dbg.declare(metadata [16 x i64]* %"a$sd2_368", metadata !21, metadata !DIExpression()), !dbg !10
  %4 = bitcast [16 x i64]* %"a$sd2_368" to i64*, !dbg !16
  store i64 0, i64* %4, align 8, !dbg !16
  call void @llvm.dbg.declare(metadata [80 x i8]** %.Z0971_342, metadata !26, metadata !DIExpression(DW_OP_deref)), !dbg !10
  %5 = bitcast [80 x i8]** %.Z0971_342 to i8**, !dbg !16
  store i8* null, i8** %5, align 8, !dbg !16
  call void @llvm.dbg.declare(metadata [16 x i64]* %"args$sd1_364", metadata !21, metadata !DIExpression()), !dbg !10
  %6 = bitcast [16 x i64]* %"args$sd1_364" to i64*, !dbg !16
  store i64 0, i64* %6, align 8, !dbg !16
  br label %L.LB1_403

L.LB1_403:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %len_326, metadata !29, metadata !DIExpression()), !dbg !10
  store i32 100, i32* %len_326, align 4, !dbg !30
  call void @llvm.dbg.declare(metadata i32* %tmp_308, metadata !31, metadata !DIExpression()), !dbg !10
  store i32 10, i32* %tmp_308, align 4, !dbg !32
  %7 = call i32 (...) @f90_cmd_arg_cnt(), !dbg !33
  call void @llvm.dbg.declare(metadata i32* %argcount_309, metadata !34, metadata !DIExpression()), !dbg !10
  store i32 %7, i32* %argcount_309, align 4, !dbg !33
  %8 = load i32, i32* %argcount_309, align 4, !dbg !35
  call void @llvm.dbg.value(metadata i32 %8, metadata !34, metadata !DIExpression()), !dbg !10
  %9 = icmp ne i32 %8, 0, !dbg !35
  br i1 %9, label %L.LB1_373, label %L.LB1_503, !dbg !35

L.LB1_503:                                        ; preds = %L.LB1_403
  call void (...) @_mp_bcs_nest(), !dbg !36
  %10 = bitcast i32* @.C332_MAIN_ to i8*, !dbg !36
  %11 = bitcast [57 x i8]* @.C330_MAIN_ to i8*, !dbg !36
  %12 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !36
  call void (i8*, i8*, i64, ...) %12(i8* %10, i8* %11, i64 57), !dbg !36
  %13 = bitcast i32* @.C305_MAIN_ to i8*, !dbg !36
  %14 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !36
  %15 = bitcast [3 x i8]* @.C334_MAIN_ to i8*, !dbg !36
  %16 = bitcast i32 (...)* @f90io_encode_fmta to i32 (i8*, i8*, i8*, i64, ...)*, !dbg !36
  %17 = call i32 (i8*, i8*, i8*, i64, ...) %16(i8* %13, i8* %14, i8* %15, i64 3), !dbg !36
  call void @llvm.dbg.declare(metadata i32* %z__io_337, metadata !37, metadata !DIExpression()), !dbg !10
  store i32 %17, i32* %z__io_337, align 4, !dbg !36
  %18 = bitcast i32* @.C333_MAIN_ to i8*, !dbg !36
  %19 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !36
  %20 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !36
  %21 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !36
  %22 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %21(i8* %18, i8* null, i8* %19, i8* %20, i8* null, i8* null, i64 0), !dbg !36
  store i32 %22, i32* %z__io_337, align 4, !dbg !36
  %23 = bitcast i32* @.C305_MAIN_ to i8*, !dbg !36
  %24 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !36
  %25 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !36
  %26 = bitcast [35 x i8]* @.C335_MAIN_ to i8*, !dbg !36
  %27 = bitcast i32 (...)* @f90io_fmt_writea to i32 (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !36
  %28 = call i32 (i8*, i8*, i8*, i8*, i64, ...) %27(i8* %23, i8* %24, i8* %25, i8* %26, i64 35), !dbg !36
  store i32 %28, i32* %z__io_337, align 4, !dbg !36
  %29 = call i32 (...) @f90io_fmtw_end(), !dbg !36
  store i32 %29, i32* %z__io_337, align 4, !dbg !36
  call void (...) @_mp_ecs_nest(), !dbg !36
  br label %L.LB1_373

L.LB1_373:                                        ; preds = %L.LB1_503, %L.LB1_403
  call void @llvm.dbg.declare(metadata i64* %z_b_0_313, metadata !38, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_0_313, align 8, !dbg !39
  %30 = load i32, i32* %argcount_309, align 4, !dbg !39
  call void @llvm.dbg.value(metadata i32 %30, metadata !34, metadata !DIExpression()), !dbg !10
  %31 = sext i32 %30 to i64, !dbg !39
  call void @llvm.dbg.declare(metadata i64* %z_b_1_314, metadata !38, metadata !DIExpression()), !dbg !10
  store i64 %31, i64* %z_b_1_314, align 8, !dbg !39
  %32 = load i64, i64* %z_b_1_314, align 8, !dbg !39
  call void @llvm.dbg.value(metadata i64 %32, metadata !38, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_61_317, metadata !38, metadata !DIExpression()), !dbg !10
  store i64 %32, i64* %z_e_61_317, align 8, !dbg !39
  %33 = bitcast [16 x i64]* %"args$sd1_364" to i8*, !dbg !39
  %34 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !39
  %35 = bitcast i64* @.C365_MAIN_ to i8*, !dbg !39
  %36 = bitcast i64* @.C366_MAIN_ to i8*, !dbg !39
  %37 = bitcast i64* %z_b_0_313 to i8*, !dbg !39
  %38 = bitcast i64* %z_b_1_314 to i8*, !dbg !39
  %39 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !39
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %39(i8* %33, i8* %34, i8* %35, i8* %36, i8* %37, i8* %38), !dbg !39
  %40 = bitcast [16 x i64]* %"args$sd1_364" to i8*, !dbg !39
  %41 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !39
  call void (i8*, i32, ...) %41(i8* %40, i32 14), !dbg !39
  %42 = load i64, i64* %z_b_1_314, align 8, !dbg !39
  call void @llvm.dbg.value(metadata i64 %42, metadata !38, metadata !DIExpression()), !dbg !10
  %43 = load i64, i64* %z_b_0_313, align 8, !dbg !39
  call void @llvm.dbg.value(metadata i64 %43, metadata !38, metadata !DIExpression()), !dbg !10
  %44 = sub nsw i64 %43, 1, !dbg !39
  %45 = sub nsw i64 %42, %44, !dbg !39
  call void @llvm.dbg.declare(metadata i64* %z_b_2_315, metadata !38, metadata !DIExpression()), !dbg !10
  store i64 %45, i64* %z_b_2_315, align 8, !dbg !39
  %46 = load i64, i64* %z_b_0_313, align 8, !dbg !39
  call void @llvm.dbg.value(metadata i64 %46, metadata !38, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_b_3_316, metadata !38, metadata !DIExpression()), !dbg !10
  store i64 %46, i64* %z_b_3_316, align 8, !dbg !39
  %47 = bitcast i64* %z_b_2_315 to i8*, !dbg !39
  %48 = bitcast i64* @.C365_MAIN_ to i8*, !dbg !39
  %49 = bitcast i64* @.C366_MAIN_ to i8*, !dbg !39
  call void @llvm.dbg.declare(metadata i32* %allocstatus_310, metadata !40, metadata !DIExpression()), !dbg !10
  %50 = bitcast i32* %allocstatus_310 to i8*, !dbg !39
  %51 = bitcast [80 x i8]** %.Z0971_342 to i8*, !dbg !39
  %52 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !39
  %53 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !39
  %54 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !39
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %54(i8* %47, i8* %48, i8* %49, i8* %50, i8* %51, i8* null, i8* %52, i8* %53, i8* null, i64 0), !dbg !39
  %55 = load i32, i32* %allocstatus_310, align 4, !dbg !41
  call void @llvm.dbg.value(metadata i32 %55, metadata !40, metadata !DIExpression()), !dbg !10
  %56 = icmp sle i32 %55, 0, !dbg !41
  br i1 %56, label %L.LB1_376, label %L.LB1_504, !dbg !41

L.LB1_504:                                        ; preds = %L.LB1_373
  call void (...) @_mp_bcs_nest(), !dbg !42
  %57 = bitcast i32* @.C343_MAIN_ to i8*, !dbg !42
  %58 = bitcast [57 x i8]* @.C330_MAIN_ to i8*, !dbg !42
  %59 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !42
  call void (i8*, i8*, i64, ...) %59(i8* %57, i8* %58, i64 57), !dbg !42
  %60 = bitcast i32* @.C305_MAIN_ to i8*, !dbg !42
  %61 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !42
  %62 = bitcast [3 x i8]* @.C334_MAIN_ to i8*, !dbg !42
  %63 = bitcast i32 (...)* @f90io_encode_fmta to i32 (i8*, i8*, i8*, i64, ...)*, !dbg !42
  %64 = call i32 (i8*, i8*, i8*, i64, ...) %63(i8* %60, i8* %61, i8* %62, i64 3), !dbg !42
  store i32 %64, i32* %z__io_337, align 4, !dbg !42
  %65 = bitcast i32* @.C333_MAIN_ to i8*, !dbg !42
  %66 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !42
  %67 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !42
  %68 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !42
  %69 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %68(i8* %65, i8* null, i8* %66, i8* %67, i8* null, i8* null, i64 0), !dbg !42
  store i32 %69, i32* %z__io_337, align 4, !dbg !42
  %70 = bitcast i32* @.C305_MAIN_ to i8*, !dbg !42
  %71 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !42
  %72 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !42
  %73 = bitcast [37 x i8]* @.C344_MAIN_ to i8*, !dbg !42
  %74 = bitcast i32 (...)* @f90io_fmt_writea to i32 (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !42
  %75 = call i32 (i8*, i8*, i8*, i8*, i64, ...) %74(i8* %70, i8* %71, i8* %72, i8* %73, i64 37), !dbg !42
  store i32 %75, i32* %z__io_337, align 4, !dbg !42
  %76 = call i32 (...) @f90io_fmtw_end(), !dbg !42
  store i32 %76, i32* %z__io_337, align 4, !dbg !42
  call void (...) @_mp_ecs_nest(), !dbg !42
  %77 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !43
  %78 = bitcast void (...)* @f90_stop08a to void (i8*, i8*, i64, ...)*, !dbg !43
  call void (i8*, i8*, i64, ...) %78(i8* %77, i8* null, i64 0), !dbg !43
  br label %L.LB1_376

L.LB1_376:                                        ; preds = %L.LB1_504, %L.LB1_373
  %79 = load i32, i32* %argcount_309, align 4, !dbg !44
  call void @llvm.dbg.value(metadata i32 %79, metadata !34, metadata !DIExpression()), !dbg !10
  store i32 %79, i32* %.dY0001_379, align 4, !dbg !44
  call void @llvm.dbg.declare(metadata i32* %ix_312, metadata !45, metadata !DIExpression()), !dbg !10
  store i32 1, i32* %ix_312, align 4, !dbg !44
  %80 = load i32, i32* %.dY0001_379, align 4, !dbg !44
  %81 = icmp sle i32 %80, 0, !dbg !44
  br i1 %81, label %L.LB1_378, label %L.LB1_377, !dbg !44

L.LB1_377:                                        ; preds = %L.LB1_377, %L.LB1_376
  %82 = bitcast i32* %ix_312 to i8*, !dbg !46
  %83 = load [80 x i8]*, [80 x i8]** %.Z0971_342, align 8, !dbg !46
  call void @llvm.dbg.value(metadata [80 x i8]* %83, metadata !26, metadata !DIExpression()), !dbg !10
  %84 = bitcast [80 x i8]* %83 to i8*, !dbg !46
  %85 = getelementptr i8, i8* %84, i64 -80, !dbg !46
  %86 = load i32, i32* %ix_312, align 4, !dbg !46
  call void @llvm.dbg.value(metadata i32 %86, metadata !45, metadata !DIExpression()), !dbg !10
  %87 = sext i32 %86 to i64, !dbg !46
  %88 = bitcast [16 x i64]* %"args$sd1_364" to i8*, !dbg !46
  %89 = getelementptr i8, i8* %88, i64 56, !dbg !46
  %90 = bitcast i8* %89 to i64*, !dbg !46
  %91 = load i64, i64* %90, align 8, !dbg !46
  %92 = add nsw i64 %87, %91, !dbg !46
  %93 = mul nsw i64 %92, 80, !dbg !46
  %94 = getelementptr i8, i8* %85, i64 %93, !dbg !46
  %95 = bitcast i64* @.C362_MAIN_ to i8*, !dbg !46
  %96 = bitcast void (...)* @f90_get_cmd_arga to void (i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !46
  call void (i8*, i8*, i8*, i8*, i8*, i64, ...) %96(i8* %82, i8* %94, i8* null, i8* null, i8* %95, i64 80), !dbg !46
  %97 = load i32, i32* %ix_312, align 4, !dbg !47
  call void @llvm.dbg.value(metadata i32 %97, metadata !45, metadata !DIExpression()), !dbg !10
  %98 = add nsw i32 %97, 1, !dbg !47
  store i32 %98, i32* %ix_312, align 4, !dbg !47
  %99 = load i32, i32* %.dY0001_379, align 4, !dbg !47
  %100 = sub nsw i32 %99, 1, !dbg !47
  store i32 %100, i32* %.dY0001_379, align 4, !dbg !47
  %101 = load i32, i32* %.dY0001_379, align 4, !dbg !47
  %102 = icmp sgt i32 %101, 0, !dbg !47
  br i1 %102, label %L.LB1_377, label %L.LB1_378, !dbg !47

L.LB1_378:                                        ; preds = %L.LB1_377, %L.LB1_376
  %103 = load i32, i32* %argcount_309, align 4, !dbg !48
  call void @llvm.dbg.value(metadata i32 %103, metadata !34, metadata !DIExpression()), !dbg !10
  %104 = icmp sle i32 %103, 0, !dbg !48
  br i1 %104, label %L.LB1_380, label %L.LB1_505, !dbg !48

L.LB1_505:                                        ; preds = %L.LB1_378
  call void (...) @_mp_bcs_nest(), !dbg !49
  %105 = bitcast i32* @.C346_MAIN_ to i8*, !dbg !49
  %106 = bitcast [57 x i8]* @.C330_MAIN_ to i8*, !dbg !49
  %107 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !49
  call void (i8*, i8*, i64, ...) %107(i8* %105, i8* %106, i64 57), !dbg !49
  %108 = bitcast i32* @.C305_MAIN_ to i8*, !dbg !49
  %109 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !49
  %110 = bitcast [5 x i8]* @.C347_MAIN_ to i8*, !dbg !49
  %111 = bitcast i32 (...)* @f90io_encode_fmta to i32 (i8*, i8*, i8*, i64, ...)*, !dbg !49
  %112 = call i32 (i8*, i8*, i8*, i64, ...) %111(i8* %108, i8* %109, i8* %110, i64 5), !dbg !49
  store i32 %112, i32* %z__io_337, align 4, !dbg !49
  %113 = load [80 x i8]*, [80 x i8]** %.Z0971_342, align 8, !dbg !49
  call void @llvm.dbg.value(metadata [80 x i8]* %113, metadata !26, metadata !DIExpression()), !dbg !10
  %114 = bitcast [80 x i8]* %113 to i8*, !dbg !49
  %115 = bitcast [16 x i64]* %"args$sd1_364" to i8*, !dbg !49
  %116 = getelementptr i8, i8* %115, i64 56, !dbg !49
  %117 = bitcast i8* %116 to i64*, !dbg !49
  %118 = load i64, i64* %117, align 8, !dbg !49
  %119 = mul nsw i64 %118, 80, !dbg !49
  %120 = getelementptr i8, i8* %114, i64 %119, !dbg !49
  %121 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !49
  %122 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !49
  call void @llvm.dbg.declare(metadata i32* %rderr_311, metadata !50, metadata !DIExpression()), !dbg !10
  %123 = bitcast i32* %rderr_311 to i8*, !dbg !49
  %124 = bitcast i32 (...)* @f90io_fmtr_intern_inita to i32 (i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !49
  %125 = call i32 (i8*, i8*, i8*, i8*, i8*, i64, ...) %124(i8* %120, i8* %121, i8* %122, i8* %123, i8* null, i64 80), !dbg !49
  store i32 %125, i32* %z__io_337, align 4, !dbg !49
  %126 = bitcast i32* @.C306_MAIN_ to i8*, !dbg !49
  %127 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !49
  %128 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !49
  %129 = bitcast i32* %len_326 to i8*, !dbg !49
  %130 = bitcast i32 (...)* @f90io_fmt_reada to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !49
  %131 = call i32 (i8*, i8*, i8*, i8*, ...) %130(i8* %126, i8* %127, i8* %128, i8* %129), !dbg !49
  store i32 %131, i32* %z__io_337, align 4, !dbg !49
  %132 = call i32 (...) @f90io_fmtr_end(), !dbg !49
  store i32 %132, i32* %z__io_337, align 4, !dbg !49
  call void (...) @_mp_ecs_nest(), !dbg !49
  %133 = load i32, i32* %rderr_311, align 4, !dbg !51
  call void @llvm.dbg.value(metadata i32 %133, metadata !50, metadata !DIExpression()), !dbg !10
  %134 = icmp eq i32 %133, 0, !dbg !51
  br i1 %134, label %L.LB1_381, label %L.LB1_506, !dbg !51

L.LB1_506:                                        ; preds = %L.LB1_505
  call void (...) @_mp_bcs_nest(), !dbg !52
  %135 = bitcast i32* @.C351_MAIN_ to i8*, !dbg !52
  %136 = bitcast [57 x i8]* @.C330_MAIN_ to i8*, !dbg !52
  %137 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !52
  call void (i8*, i8*, i64, ...) %137(i8* %135, i8* %136, i64 57), !dbg !52
  %138 = bitcast i32* @.C305_MAIN_ to i8*, !dbg !52
  %139 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !52
  %140 = bitcast [3 x i8]* @.C334_MAIN_ to i8*, !dbg !52
  %141 = bitcast i32 (...)* @f90io_encode_fmta to i32 (i8*, i8*, i8*, i64, ...)*, !dbg !52
  %142 = call i32 (i8*, i8*, i8*, i64, ...) %141(i8* %138, i8* %139, i8* %140, i64 3), !dbg !52
  store i32 %142, i32* %z__io_337, align 4, !dbg !52
  %143 = bitcast i32* @.C333_MAIN_ to i8*, !dbg !52
  %144 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !52
  %145 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !52
  %146 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !52
  %147 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %146(i8* %143, i8* null, i8* %144, i8* %145, i8* null, i8* null, i64 0), !dbg !52
  store i32 %147, i32* %z__io_337, align 4, !dbg !52
  %148 = bitcast i32* @.C305_MAIN_ to i8*, !dbg !52
  %149 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !52
  %150 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !52
  %151 = bitcast [29 x i8]* @.C352_MAIN_ to i8*, !dbg !52
  %152 = bitcast i32 (...)* @f90io_fmt_writea to i32 (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !52
  %153 = call i32 (i8*, i8*, i8*, i8*, i64, ...) %152(i8* %148, i8* %149, i8* %150, i8* %151, i64 29), !dbg !52
  store i32 %153, i32* %z__io_337, align 4, !dbg !52
  %154 = call i32 (...) @f90io_fmtw_end(), !dbg !52
  store i32 %154, i32* %z__io_337, align 4, !dbg !52
  call void (...) @_mp_ecs_nest(), !dbg !52
  br label %L.LB1_381

L.LB1_381:                                        ; preds = %L.LB1_506, %L.LB1_505
  br label %L.LB1_380

L.LB1_380:                                        ; preds = %L.LB1_381, %L.LB1_378
  call void @llvm.dbg.declare(metadata i64* %z_b_4_319, metadata !38, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_4_319, align 8, !dbg !53
  %155 = load i32, i32* %len_326, align 4, !dbg !53
  call void @llvm.dbg.value(metadata i32 %155, metadata !29, metadata !DIExpression()), !dbg !10
  %156 = sext i32 %155 to i64, !dbg !53
  call void @llvm.dbg.declare(metadata i64* %z_b_5_320, metadata !38, metadata !DIExpression()), !dbg !10
  store i64 %156, i64* %z_b_5_320, align 8, !dbg !53
  %157 = load i64, i64* %z_b_5_320, align 8, !dbg !53
  call void @llvm.dbg.value(metadata i64 %157, metadata !38, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_68_323, metadata !38, metadata !DIExpression()), !dbg !10
  store i64 %157, i64* %z_e_68_323, align 8, !dbg !53
  %158 = bitcast [16 x i64]* %"a$sd2_368" to i8*, !dbg !53
  %159 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !53
  %160 = bitcast i64* @.C369_MAIN_ to i8*, !dbg !53
  %161 = bitcast i64* @.C362_MAIN_ to i8*, !dbg !53
  %162 = bitcast i64* %z_b_4_319 to i8*, !dbg !53
  %163 = bitcast i64* %z_b_5_320 to i8*, !dbg !53
  %164 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !53
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %164(i8* %158, i8* %159, i8* %160, i8* %161, i8* %162, i8* %163), !dbg !53
  %165 = bitcast [16 x i64]* %"a$sd2_368" to i8*, !dbg !53
  %166 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !53
  call void (i8*, i32, ...) %166(i8* %165, i32 25), !dbg !53
  %167 = load i64, i64* %z_b_5_320, align 8, !dbg !53
  call void @llvm.dbg.value(metadata i64 %167, metadata !38, metadata !DIExpression()), !dbg !10
  %168 = load i64, i64* %z_b_4_319, align 8, !dbg !53
  call void @llvm.dbg.value(metadata i64 %168, metadata !38, metadata !DIExpression()), !dbg !10
  %169 = sub nsw i64 %168, 1, !dbg !53
  %170 = sub nsw i64 %167, %169, !dbg !53
  call void @llvm.dbg.declare(metadata i64* %z_b_6_321, metadata !38, metadata !DIExpression()), !dbg !10
  store i64 %170, i64* %z_b_6_321, align 8, !dbg !53
  %171 = load i64, i64* %z_b_4_319, align 8, !dbg !53
  call void @llvm.dbg.value(metadata i64 %171, metadata !38, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_b_7_322, metadata !38, metadata !DIExpression()), !dbg !10
  store i64 %171, i64* %z_b_7_322, align 8, !dbg !53
  %172 = bitcast i64* %z_b_6_321 to i8*, !dbg !53
  %173 = bitcast i64* @.C369_MAIN_ to i8*, !dbg !53
  %174 = bitcast i64* @.C362_MAIN_ to i8*, !dbg !53
  %175 = bitcast i32** %.Z0977_353 to i8*, !dbg !53
  %176 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !53
  %177 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !53
  %178 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !53
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %178(i8* %172, i8* %173, i8* %174, i8* null, i8* %175, i8* null, i8* %176, i8* %177, i8* null, i64 0), !dbg !53
  %179 = bitcast i32* %len_326 to i8*, !dbg !54
  %180 = bitcast %astruct.dt86* %.uplevelArgPack0001_462 to i8**, !dbg !54
  store i8* %179, i8** %180, align 8, !dbg !54
  %181 = bitcast i32** %.Z0977_353 to i8*, !dbg !54
  %182 = bitcast %astruct.dt86* %.uplevelArgPack0001_462 to i8*, !dbg !54
  %183 = getelementptr i8, i8* %182, i64 8, !dbg !54
  %184 = bitcast i8* %183 to i8**, !dbg !54
  store i8* %181, i8** %184, align 8, !dbg !54
  %185 = bitcast i32** %.Z0977_353 to i8*, !dbg !54
  %186 = bitcast %astruct.dt86* %.uplevelArgPack0001_462 to i8*, !dbg !54
  %187 = getelementptr i8, i8* %186, i64 16, !dbg !54
  %188 = bitcast i8* %187 to i8**, !dbg !54
  store i8* %185, i8** %188, align 8, !dbg !54
  %189 = bitcast i64* %z_b_4_319 to i8*, !dbg !54
  %190 = bitcast %astruct.dt86* %.uplevelArgPack0001_462 to i8*, !dbg !54
  %191 = getelementptr i8, i8* %190, i64 24, !dbg !54
  %192 = bitcast i8* %191 to i8**, !dbg !54
  store i8* %189, i8** %192, align 8, !dbg !54
  %193 = bitcast i64* %z_b_5_320 to i8*, !dbg !54
  %194 = bitcast %astruct.dt86* %.uplevelArgPack0001_462 to i8*, !dbg !54
  %195 = getelementptr i8, i8* %194, i64 32, !dbg !54
  %196 = bitcast i8* %195 to i8**, !dbg !54
  store i8* %193, i8** %196, align 8, !dbg !54
  %197 = bitcast i64* %z_e_68_323 to i8*, !dbg !54
  %198 = bitcast %astruct.dt86* %.uplevelArgPack0001_462 to i8*, !dbg !54
  %199 = getelementptr i8, i8* %198, i64 40, !dbg !54
  %200 = bitcast i8* %199 to i8**, !dbg !54
  store i8* %197, i8** %200, align 8, !dbg !54
  %201 = bitcast i64* %z_b_6_321 to i8*, !dbg !54
  %202 = bitcast %astruct.dt86* %.uplevelArgPack0001_462 to i8*, !dbg !54
  %203 = getelementptr i8, i8* %202, i64 48, !dbg !54
  %204 = bitcast i8* %203 to i8**, !dbg !54
  store i8* %201, i8** %204, align 8, !dbg !54
  %205 = bitcast i64* %z_b_7_322 to i8*, !dbg !54
  %206 = bitcast %astruct.dt86* %.uplevelArgPack0001_462 to i8*, !dbg !54
  %207 = getelementptr i8, i8* %206, i64 56, !dbg !54
  %208 = bitcast i8* %207 to i8**, !dbg !54
  store i8* %205, i8** %208, align 8, !dbg !54
  %209 = bitcast i32* %tmp_308 to i8*, !dbg !54
  %210 = bitcast %astruct.dt86* %.uplevelArgPack0001_462 to i8*, !dbg !54
  %211 = getelementptr i8, i8* %210, i64 64, !dbg !54
  %212 = bitcast i8* %211 to i8**, !dbg !54
  store i8* %209, i8** %212, align 8, !dbg !54
  %213 = bitcast [16 x i64]* %"a$sd2_368" to i8*, !dbg !54
  %214 = bitcast %astruct.dt86* %.uplevelArgPack0001_462 to i8*, !dbg !54
  %215 = getelementptr i8, i8* %214, i64 72, !dbg !54
  %216 = bitcast i8* %215 to i8**, !dbg !54
  store i8* %213, i8** %216, align 8, !dbg !54
  br label %L.LB1_483, !dbg !54

L.LB1_483:                                        ; preds = %L.LB1_380
  %217 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L46_1_ to i64*, !dbg !54
  %218 = bitcast %astruct.dt86* %.uplevelArgPack0001_462 to i64*, !dbg !54
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %217, i64* %218), !dbg !54
  %219 = load [80 x i8]*, [80 x i8]** %.Z0971_342, align 8, !dbg !55
  call void @llvm.dbg.value(metadata [80 x i8]* %219, metadata !26, metadata !DIExpression()), !dbg !10
  %220 = bitcast [80 x i8]* %219 to i8*, !dbg !55
  %221 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !55
  %222 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, i64, ...)*, !dbg !55
  call void (i8*, i8*, i8*, i8*, i64, i64, ...) %222(i8* null, i8* %220, i8* %221, i8* null, i64 80, i64 0), !dbg !55
  %223 = bitcast [80 x i8]** %.Z0971_342 to i8**, !dbg !55
  store i8* null, i8** %223, align 8, !dbg !55
  %224 = bitcast [16 x i64]* %"args$sd1_364" to i64*, !dbg !55
  store i64 0, i64* %224, align 8, !dbg !55
  %225 = load i32*, i32** %.Z0977_353, align 8, !dbg !55
  call void @llvm.dbg.value(metadata i32* %225, metadata !17, metadata !DIExpression()), !dbg !10
  %226 = bitcast i32* %225 to i8*, !dbg !55
  %227 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !55
  %228 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !55
  call void (i8*, i8*, i8*, i8*, i64, ...) %228(i8* null, i8* %226, i8* %227, i8* null, i64 0), !dbg !55
  %229 = bitcast i32** %.Z0977_353 to i8**, !dbg !55
  store i8* null, i8** %229, align 8, !dbg !55
  %230 = bitcast [16 x i64]* %"a$sd2_368" to i64*, !dbg !55
  store i64 0, i64* %230, align 8, !dbg !55
  ret void, !dbg !15
}

define internal void @__nv_MAIN__F1L46_1_(i32* %__nv_MAIN__F1L46_1Arg0, i64* %__nv_MAIN__F1L46_1Arg1, i64* %__nv_MAIN__F1L46_1Arg2) #0 !dbg !56 {
L.entry:
  %__gtid___nv_MAIN__F1L46_1__525 = alloca i32, align 4
  %.i0000p_358 = alloca i32, align 4
  %i_357 = alloca i32, align 4
  %.du0002p_385 = alloca i32, align 4
  %.de0002p_386 = alloca i32, align 4
  %.di0002p_387 = alloca i32, align 4
  %.ds0002p_388 = alloca i32, align 4
  %.dl0002p_390 = alloca i32, align 4
  %.dl0002p.copy_519 = alloca i32, align 4
  %.de0002p.copy_520 = alloca i32, align 4
  %.ds0002p.copy_521 = alloca i32, align 4
  %.dX0002p_389 = alloca i32, align 4
  %.dY0002p_384 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L46_1Arg0, metadata !59, metadata !DIExpression()), !dbg !60
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L46_1Arg1, metadata !61, metadata !DIExpression()), !dbg !60
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L46_1Arg2, metadata !62, metadata !DIExpression()), !dbg !60
  call void @llvm.dbg.value(metadata i32 1, metadata !63, metadata !DIExpression()), !dbg !60
  call void @llvm.dbg.value(metadata i32 0, metadata !64, metadata !DIExpression()), !dbg !60
  call void @llvm.dbg.value(metadata i32 1, metadata !65, metadata !DIExpression()), !dbg !60
  call void @llvm.dbg.value(metadata i32 0, metadata !66, metadata !DIExpression()), !dbg !60
  call void @llvm.dbg.value(metadata i32 1, metadata !67, metadata !DIExpression()), !dbg !60
  %0 = load i32, i32* %__nv_MAIN__F1L46_1Arg0, align 4, !dbg !68
  store i32 %0, i32* %__gtid___nv_MAIN__F1L46_1__525, align 4, !dbg !68
  br label %L.LB2_510

L.LB2_510:                                        ; preds = %L.entry
  br label %L.LB2_356

L.LB2_356:                                        ; preds = %L.LB2_510
  store i32 0, i32* %.i0000p_358, align 4, !dbg !69
  call void @llvm.dbg.declare(metadata i32* %i_357, metadata !70, metadata !DIExpression()), !dbg !68
  store i32 1, i32* %i_357, align 4, !dbg !69
  %1 = bitcast i64* %__nv_MAIN__F1L46_1Arg2 to i32**, !dbg !69
  %2 = load i32*, i32** %1, align 8, !dbg !69
  %3 = load i32, i32* %2, align 4, !dbg !69
  store i32 %3, i32* %.du0002p_385, align 4, !dbg !69
  %4 = bitcast i64* %__nv_MAIN__F1L46_1Arg2 to i32**, !dbg !69
  %5 = load i32*, i32** %4, align 8, !dbg !69
  %6 = load i32, i32* %5, align 4, !dbg !69
  store i32 %6, i32* %.de0002p_386, align 4, !dbg !69
  store i32 1, i32* %.di0002p_387, align 4, !dbg !69
  %7 = load i32, i32* %.di0002p_387, align 4, !dbg !69
  store i32 %7, i32* %.ds0002p_388, align 4, !dbg !69
  store i32 1, i32* %.dl0002p_390, align 4, !dbg !69
  %8 = load i32, i32* %.dl0002p_390, align 4, !dbg !69
  store i32 %8, i32* %.dl0002p.copy_519, align 4, !dbg !69
  %9 = load i32, i32* %.de0002p_386, align 4, !dbg !69
  store i32 %9, i32* %.de0002p.copy_520, align 4, !dbg !69
  %10 = load i32, i32* %.ds0002p_388, align 4, !dbg !69
  store i32 %10, i32* %.ds0002p.copy_521, align 4, !dbg !69
  %11 = load i32, i32* %__gtid___nv_MAIN__F1L46_1__525, align 4, !dbg !69
  %12 = bitcast i32* %.i0000p_358 to i64*, !dbg !69
  %13 = bitcast i32* %.dl0002p.copy_519 to i64*, !dbg !69
  %14 = bitcast i32* %.de0002p.copy_520 to i64*, !dbg !69
  %15 = bitcast i32* %.ds0002p.copy_521 to i64*, !dbg !69
  %16 = load i32, i32* %.ds0002p.copy_521, align 4, !dbg !69
  call void @__kmpc_for_static_init_4(i64* null, i32 %11, i32 34, i64* %12, i64* %13, i64* %14, i64* %15, i32 %16, i32 1), !dbg !69
  %17 = load i32, i32* %.dl0002p.copy_519, align 4, !dbg !69
  store i32 %17, i32* %.dl0002p_390, align 4, !dbg !69
  %18 = load i32, i32* %.de0002p.copy_520, align 4, !dbg !69
  store i32 %18, i32* %.de0002p_386, align 4, !dbg !69
  %19 = load i32, i32* %.ds0002p.copy_521, align 4, !dbg !69
  store i32 %19, i32* %.ds0002p_388, align 4, !dbg !69
  %20 = load i32, i32* %.dl0002p_390, align 4, !dbg !69
  store i32 %20, i32* %i_357, align 4, !dbg !69
  %21 = load i32, i32* %i_357, align 4, !dbg !69
  call void @llvm.dbg.value(metadata i32 %21, metadata !70, metadata !DIExpression()), !dbg !68
  store i32 %21, i32* %.dX0002p_389, align 4, !dbg !69
  %22 = load i32, i32* %.dX0002p_389, align 4, !dbg !69
  %23 = load i32, i32* %.du0002p_385, align 4, !dbg !69
  %24 = icmp sgt i32 %22, %23, !dbg !69
  br i1 %24, label %L.LB2_383, label %L.LB2_552, !dbg !69

L.LB2_552:                                        ; preds = %L.LB2_356
  %25 = load i32, i32* %.dX0002p_389, align 4, !dbg !69
  store i32 %25, i32* %i_357, align 4, !dbg !69
  %26 = load i32, i32* %.di0002p_387, align 4, !dbg !69
  %27 = load i32, i32* %.de0002p_386, align 4, !dbg !69
  %28 = load i32, i32* %.dX0002p_389, align 4, !dbg !69
  %29 = sub nsw i32 %27, %28, !dbg !69
  %30 = add nsw i32 %26, %29, !dbg !69
  %31 = load i32, i32* %.di0002p_387, align 4, !dbg !69
  %32 = sdiv i32 %30, %31, !dbg !69
  store i32 %32, i32* %.dY0002p_384, align 4, !dbg !69
  %33 = load i32, i32* %.dY0002p_384, align 4, !dbg !69
  %34 = icmp sle i32 %33, 0, !dbg !69
  br i1 %34, label %L.LB2_393, label %L.LB2_392, !dbg !69

L.LB2_392:                                        ; preds = %L.LB2_392, %L.LB2_552
  %35 = bitcast i64* %__nv_MAIN__F1L46_1Arg2 to i8*, !dbg !71
  %36 = getelementptr i8, i8* %35, i64 64, !dbg !71
  %37 = bitcast i8* %36 to i32**, !dbg !71
  %38 = load i32*, i32** %37, align 8, !dbg !71
  %39 = load i32, i32* %38, align 4, !dbg !71
  %40 = load i32, i32* %i_357, align 4, !dbg !71
  call void @llvm.dbg.value(metadata i32 %40, metadata !70, metadata !DIExpression()), !dbg !68
  %41 = sext i32 %40 to i64, !dbg !71
  %42 = bitcast i64* %__nv_MAIN__F1L46_1Arg2 to i8*, !dbg !71
  %43 = getelementptr i8, i8* %42, i64 72, !dbg !71
  %44 = bitcast i8* %43 to i8**, !dbg !71
  %45 = load i8*, i8** %44, align 8, !dbg !71
  %46 = getelementptr i8, i8* %45, i64 56, !dbg !71
  %47 = bitcast i8* %46 to i64*, !dbg !71
  %48 = load i64, i64* %47, align 8, !dbg !71
  %49 = add nsw i64 %41, %48, !dbg !71
  %50 = bitcast i64* %__nv_MAIN__F1L46_1Arg2 to i8*, !dbg !71
  %51 = getelementptr i8, i8* %50, i64 16, !dbg !71
  %52 = bitcast i8* %51 to i8***, !dbg !71
  %53 = load i8**, i8*** %52, align 8, !dbg !71
  %54 = load i8*, i8** %53, align 8, !dbg !71
  %55 = getelementptr i8, i8* %54, i64 -4, !dbg !71
  %56 = bitcast i8* %55 to i32*, !dbg !71
  %57 = getelementptr i32, i32* %56, i64 %49, !dbg !71
  store i32 %39, i32* %57, align 4, !dbg !71
  %58 = load i32, i32* %i_357, align 4, !dbg !72
  call void @llvm.dbg.value(metadata i32 %58, metadata !70, metadata !DIExpression()), !dbg !68
  %59 = load i32, i32* %i_357, align 4, !dbg !72
  call void @llvm.dbg.value(metadata i32 %59, metadata !70, metadata !DIExpression()), !dbg !68
  %60 = sext i32 %59 to i64, !dbg !72
  %61 = bitcast i64* %__nv_MAIN__F1L46_1Arg2 to i8*, !dbg !72
  %62 = getelementptr i8, i8* %61, i64 72, !dbg !72
  %63 = bitcast i8* %62 to i8**, !dbg !72
  %64 = load i8*, i8** %63, align 8, !dbg !72
  %65 = getelementptr i8, i8* %64, i64 56, !dbg !72
  %66 = bitcast i8* %65 to i64*, !dbg !72
  %67 = load i64, i64* %66, align 8, !dbg !72
  %68 = add nsw i64 %60, %67, !dbg !72
  %69 = bitcast i64* %__nv_MAIN__F1L46_1Arg2 to i8*, !dbg !72
  %70 = getelementptr i8, i8* %69, i64 16, !dbg !72
  %71 = bitcast i8* %70 to i8***, !dbg !72
  %72 = load i8**, i8*** %71, align 8, !dbg !72
  %73 = load i8*, i8** %72, align 8, !dbg !72
  %74 = getelementptr i8, i8* %73, i64 -4, !dbg !72
  %75 = bitcast i8* %74 to i32*, !dbg !72
  %76 = getelementptr i32, i32* %75, i64 %68, !dbg !72
  %77 = load i32, i32* %76, align 4, !dbg !72
  %78 = add nsw i32 %58, %77, !dbg !72
  %79 = bitcast i64* %__nv_MAIN__F1L46_1Arg2 to i8*, !dbg !72
  %80 = getelementptr i8, i8* %79, i64 64, !dbg !72
  %81 = bitcast i8* %80 to i32**, !dbg !72
  %82 = load i32*, i32** %81, align 8, !dbg !72
  store i32 %78, i32* %82, align 4, !dbg !72
  %83 = load i32, i32* %.di0002p_387, align 4, !dbg !68
  %84 = load i32, i32* %i_357, align 4, !dbg !68
  call void @llvm.dbg.value(metadata i32 %84, metadata !70, metadata !DIExpression()), !dbg !68
  %85 = add nsw i32 %83, %84, !dbg !68
  store i32 %85, i32* %i_357, align 4, !dbg !68
  %86 = load i32, i32* %.dY0002p_384, align 4, !dbg !68
  %87 = sub nsw i32 %86, 1, !dbg !68
  store i32 %87, i32* %.dY0002p_384, align 4, !dbg !68
  %88 = load i32, i32* %.dY0002p_384, align 4, !dbg !68
  %89 = icmp sgt i32 %88, 0, !dbg !68
  br i1 %89, label %L.LB2_392, label %L.LB2_393, !dbg !68

L.LB2_393:                                        ; preds = %L.LB2_392, %L.LB2_552
  br label %L.LB2_383

L.LB2_383:                                        ; preds = %L.LB2_393, %L.LB2_356
  %90 = load i32, i32* %__gtid___nv_MAIN__F1L46_1__525, align 4, !dbg !68
  call void @__kmpc_for_static_fini(i64* null, i32 %90), !dbg !68
  br label %L.LB2_359

L.LB2_359:                                        ; preds = %L.LB2_383
  ret void, !dbg !68
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
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB036-truedepscalar-var-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb036_truedepscalar_var_yes", scope: !2, file: !3, line: 11, type: !6, scopeLine: 11, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!6 = !DISubroutineType(cc: DW_CC_program, types: !7)
!7 = !{null}
!8 = !DILocalVariable(name: "omp_sched_static", scope: !5, file: !3, type: !9)
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 0, scope: !5)
!11 = !DILocalVariable(name: "omp_proc_bind_false", scope: !5, file: !3, type: !9)
!12 = !DILocalVariable(name: "omp_proc_bind_true", scope: !5, file: !3, type: !9)
!13 = !DILocalVariable(name: "omp_lock_hint_none", scope: !5, file: !3, type: !9)
!14 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !5, file: !3, type: !9)
!15 = !DILocation(line: 54, column: 1, scope: !5)
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
!26 = !DILocalVariable(name: "args", scope: !5, file: !3, type: !27)
!27 = !DICompositeType(tag: DW_TAG_array_type, baseType: !28, size: 640, align: 8, elements: !19)
!28 = !DIBasicType(name: "character", size: 640, align: 8, encoding: DW_ATE_signed)
!29 = !DILocalVariable(name: "len", scope: !5, file: !3, type: !9)
!30 = !DILocation(line: 19, column: 1, scope: !5)
!31 = !DILocalVariable(name: "tmp", scope: !5, file: !3, type: !9)
!32 = !DILocation(line: 20, column: 1, scope: !5)
!33 = !DILocation(line: 22, column: 1, scope: !5)
!34 = !DILocalVariable(name: "argcount", scope: !5, file: !3, type: !9)
!35 = !DILocation(line: 23, column: 1, scope: !5)
!36 = !DILocation(line: 24, column: 1, scope: !5)
!37 = !DILocalVariable(scope: !5, file: !3, type: !9, flags: DIFlagArtificial)
!38 = !DILocalVariable(scope: !5, file: !3, type: !23, flags: DIFlagArtificial)
!39 = !DILocation(line: 27, column: 1, scope: !5)
!40 = !DILocalVariable(name: "allocstatus", scope: !5, file: !3, type: !9)
!41 = !DILocation(line: 28, column: 1, scope: !5)
!42 = !DILocation(line: 29, column: 1, scope: !5)
!43 = !DILocation(line: 30, column: 1, scope: !5)
!44 = !DILocation(line: 33, column: 1, scope: !5)
!45 = !DILocalVariable(name: "ix", scope: !5, file: !3, type: !9)
!46 = !DILocation(line: 34, column: 1, scope: !5)
!47 = !DILocation(line: 35, column: 1, scope: !5)
!48 = !DILocation(line: 37, column: 1, scope: !5)
!49 = !DILocation(line: 38, column: 1, scope: !5)
!50 = !DILocalVariable(name: "rderr", scope: !5, file: !3, type: !9)
!51 = !DILocation(line: 39, column: 1, scope: !5)
!52 = !DILocation(line: 40, column: 1, scope: !5)
!53 = !DILocation(line: 44, column: 1, scope: !5)
!54 = !DILocation(line: 46, column: 1, scope: !5)
!55 = !DILocation(line: 53, column: 1, scope: !5)
!56 = distinct !DISubprogram(name: "__nv_MAIN__F1L46_1", scope: !2, file: !3, line: 46, type: !57, scopeLine: 46, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!57 = !DISubroutineType(types: !58)
!58 = !{null, !9, !23, !23}
!59 = !DILocalVariable(name: "__nv_MAIN__F1L46_1Arg0", arg: 1, scope: !56, file: !3, type: !9)
!60 = !DILocation(line: 0, scope: !56)
!61 = !DILocalVariable(name: "__nv_MAIN__F1L46_1Arg1", arg: 2, scope: !56, file: !3, type: !23)
!62 = !DILocalVariable(name: "__nv_MAIN__F1L46_1Arg2", arg: 3, scope: !56, file: !3, type: !23)
!63 = !DILocalVariable(name: "omp_sched_static", scope: !56, file: !3, type: !9)
!64 = !DILocalVariable(name: "omp_proc_bind_false", scope: !56, file: !3, type: !9)
!65 = !DILocalVariable(name: "omp_proc_bind_true", scope: !56, file: !3, type: !9)
!66 = !DILocalVariable(name: "omp_lock_hint_none", scope: !56, file: !3, type: !9)
!67 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !56, file: !3, type: !9)
!68 = !DILocation(line: 50, column: 1, scope: !56)
!69 = !DILocation(line: 47, column: 1, scope: !56)
!70 = !DILocalVariable(name: "i", scope: !56, file: !3, type: !9)
!71 = !DILocation(line: 48, column: 1, scope: !56)
!72 = !DILocation(line: 49, column: 1, scope: !56)

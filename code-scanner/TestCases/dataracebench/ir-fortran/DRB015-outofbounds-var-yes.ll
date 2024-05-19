; ModuleID = '/tmp/DRB015-outofbounds-var-yes-0c72e8.ll'
source_filename = "/tmp/DRB015-outofbounds-var-yes-0c72e8.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%astruct.dt88 = type <{ i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8* }>

@.C374_MAIN_ = internal constant i64 50
@.C373_MAIN_ = internal constant [9 x i8] c"b(50,50)="
@.C371_MAIN_ = internal constant i32 71
@.C300_MAIN_ = internal constant i32 2
@.C311_MAIN_ = internal constant i32 27
@.C389_MAIN_ = internal constant i64 27
@.C362_MAIN_ = internal constant [29 x i8] c"Error, invalid integer value."
@.C340_MAIN_ = internal constant i32 55
@.C310_MAIN_ = internal constant i32 25
@.C358_MAIN_ = internal constant [5 x i8] c"(i10)"
@.C357_MAIN_ = internal constant i32 53
@.C381_MAIN_ = internal constant i64 4
@.C355_MAIN_ = internal constant [37 x i8] c"Allocation error, program terminated."
@.C354_MAIN_ = internal constant i32 44
@.C385_MAIN_ = internal constant i64 80
@.C384_MAIN_ = internal constant i64 14
@.C346_MAIN_ = internal constant [35 x i8] c"No command line arguments provided."
@.C344_MAIN_ = internal constant i32 6
@.C345_MAIN_ = internal constant [3 x i8] c"(a)"
@.C285_MAIN_ = internal constant i32 1
@.C309_MAIN_ = internal constant i32 14
@.C341_MAIN_ = internal constant [55 x i8] c"micro-benchmarks-fortran/DRB015-outofbounds-var-yes.f95"
@.C343_MAIN_ = internal constant i32 39
@.C336_MAIN_ = internal constant i32 100
@.C286_MAIN_ = internal constant i64 1
@.C284_MAIN_ = internal constant i64 0
@.C283_MAIN_ = internal constant i32 0
@.C285___nv_MAIN__F1L64_1 = internal constant i32 1
@.C300___nv_MAIN__F1L64_1 = internal constant i32 2
@.C283___nv_MAIN__F1L64_1 = internal constant i32 0

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__522 = alloca i32, align 4
  %.Z0983_363 = alloca float*, align 8
  %"b$sd2_388" = alloca [22 x i64], align 8
  %.Z0973_353 = alloca [80 x i8]*, align 8
  %"args$sd1_383" = alloca [16 x i64], align 8
  %len_337 = alloca i32, align 4
  %argcount_316 = alloca i32, align 4
  %z__io_348 = alloca i32, align 4
  %z_b_0_320 = alloca i64, align 8
  %z_b_1_321 = alloca i64, align 8
  %z_e_61_324 = alloca i64, align 8
  %z_b_2_322 = alloca i64, align 8
  %z_b_3_323 = alloca i64, align 8
  %allocstatus_317 = alloca i32, align 4
  %.dY0001_400 = alloca i32, align 4
  %ix_319 = alloca i32, align 4
  %rderr_318 = alloca i32, align 4
  %n_314 = alloca i32, align 4
  %m_315 = alloca i32, align 4
  %z_b_4_326 = alloca i64, align 8
  %z_b_5_327 = alloca i64, align 8
  %z_e_71_333 = alloca i64, align 8
  %z_b_7_329 = alloca i64, align 8
  %z_b_8_330 = alloca i64, align 8
  %z_e_74_334 = alloca i64, align 8
  %z_b_6_328 = alloca i64, align 8
  %z_b_9_331 = alloca i64, align 8
  %z_b_10_332 = alloca i64, align 8
  %.uplevelArgPack0001_491 = alloca %astruct.dt88, align 16
  call void @llvm.dbg.value(metadata i32 1, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 2, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 2, metadata !14, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !15, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !16, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 2, metadata !17, metadata !DIExpression()), !dbg !10
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !18
  store i32 %0, i32* %__gtid_MAIN__522, align 4, !dbg !18
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !19
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !19
  call void (i8*, ...) %2(i8* %1), !dbg !19
  call void @llvm.dbg.declare(metadata float** %.Z0983_363, metadata !20, metadata !DIExpression(DW_OP_deref)), !dbg !10
  %3 = bitcast float** %.Z0983_363 to i8**, !dbg !19
  store i8* null, i8** %3, align 8, !dbg !19
  call void @llvm.dbg.declare(metadata [22 x i64]* %"b$sd2_388", metadata !25, metadata !DIExpression()), !dbg !10
  %4 = bitcast [22 x i64]* %"b$sd2_388" to i64*, !dbg !19
  store i64 0, i64* %4, align 8, !dbg !19
  call void @llvm.dbg.declare(metadata [80 x i8]** %.Z0973_353, metadata !30, metadata !DIExpression(DW_OP_deref)), !dbg !10
  %5 = bitcast [80 x i8]** %.Z0973_353 to i8**, !dbg !19
  store i8* null, i8** %5, align 8, !dbg !19
  call void @llvm.dbg.declare(metadata [16 x i64]* %"args$sd1_383", metadata !34, metadata !DIExpression()), !dbg !10
  %6 = bitcast [16 x i64]* %"args$sd1_383" to i64*, !dbg !19
  store i64 0, i64* %6, align 8, !dbg !19
  br label %L.LB1_427

L.LB1_427:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %len_337, metadata !38, metadata !DIExpression()), !dbg !10
  store i32 100, i32* %len_337, align 4, !dbg !39
  %7 = call i32 (...) @f90_cmd_arg_cnt(), !dbg !40
  call void @llvm.dbg.declare(metadata i32* %argcount_316, metadata !41, metadata !DIExpression()), !dbg !10
  store i32 %7, i32* %argcount_316, align 4, !dbg !40
  %8 = load i32, i32* %argcount_316, align 4, !dbg !42
  call void @llvm.dbg.value(metadata i32 %8, metadata !41, metadata !DIExpression()), !dbg !10
  %9 = icmp ne i32 %8, 0, !dbg !42
  br i1 %9, label %L.LB1_394, label %L.LB1_553, !dbg !42

L.LB1_553:                                        ; preds = %L.LB1_427
  call void (...) @_mp_bcs_nest(), !dbg !43
  %10 = bitcast i32* @.C343_MAIN_ to i8*, !dbg !43
  %11 = bitcast [55 x i8]* @.C341_MAIN_ to i8*, !dbg !43
  %12 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !43
  call void (i8*, i8*, i64, ...) %12(i8* %10, i8* %11, i64 55), !dbg !43
  %13 = bitcast i32* @.C309_MAIN_ to i8*, !dbg !43
  %14 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !43
  %15 = bitcast [3 x i8]* @.C345_MAIN_ to i8*, !dbg !43
  %16 = bitcast i32 (...)* @f90io_encode_fmta to i32 (i8*, i8*, i8*, i64, ...)*, !dbg !43
  %17 = call i32 (i8*, i8*, i8*, i64, ...) %16(i8* %13, i8* %14, i8* %15, i64 3), !dbg !43
  call void @llvm.dbg.declare(metadata i32* %z__io_348, metadata !44, metadata !DIExpression()), !dbg !10
  store i32 %17, i32* %z__io_348, align 4, !dbg !43
  %18 = bitcast i32* @.C344_MAIN_ to i8*, !dbg !43
  %19 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !43
  %20 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !43
  %21 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !43
  %22 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %21(i8* %18, i8* null, i8* %19, i8* %20, i8* null, i8* null, i64 0), !dbg !43
  store i32 %22, i32* %z__io_348, align 4, !dbg !43
  %23 = bitcast i32* @.C309_MAIN_ to i8*, !dbg !43
  %24 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !43
  %25 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !43
  %26 = bitcast [35 x i8]* @.C346_MAIN_ to i8*, !dbg !43
  %27 = bitcast i32 (...)* @f90io_fmt_writea to i32 (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !43
  %28 = call i32 (i8*, i8*, i8*, i8*, i64, ...) %27(i8* %23, i8* %24, i8* %25, i8* %26, i64 35), !dbg !43
  store i32 %28, i32* %z__io_348, align 4, !dbg !43
  %29 = call i32 (...) @f90io_fmtw_end(), !dbg !43
  store i32 %29, i32* %z__io_348, align 4, !dbg !43
  call void (...) @_mp_ecs_nest(), !dbg !43
  br label %L.LB1_394

L.LB1_394:                                        ; preds = %L.LB1_553, %L.LB1_427
  call void @llvm.dbg.declare(metadata i64* %z_b_0_320, metadata !45, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_0_320, align 8, !dbg !46
  %30 = load i32, i32* %argcount_316, align 4, !dbg !46
  call void @llvm.dbg.value(metadata i32 %30, metadata !41, metadata !DIExpression()), !dbg !10
  %31 = sext i32 %30 to i64, !dbg !46
  call void @llvm.dbg.declare(metadata i64* %z_b_1_321, metadata !45, metadata !DIExpression()), !dbg !10
  store i64 %31, i64* %z_b_1_321, align 8, !dbg !46
  %32 = load i64, i64* %z_b_1_321, align 8, !dbg !46
  call void @llvm.dbg.value(metadata i64 %32, metadata !45, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_61_324, metadata !45, metadata !DIExpression()), !dbg !10
  store i64 %32, i64* %z_e_61_324, align 8, !dbg !46
  %33 = bitcast [16 x i64]* %"args$sd1_383" to i8*, !dbg !46
  %34 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !46
  %35 = bitcast i64* @.C384_MAIN_ to i8*, !dbg !46
  %36 = bitcast i64* @.C385_MAIN_ to i8*, !dbg !46
  %37 = bitcast i64* %z_b_0_320 to i8*, !dbg !46
  %38 = bitcast i64* %z_b_1_321 to i8*, !dbg !46
  %39 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !46
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %39(i8* %33, i8* %34, i8* %35, i8* %36, i8* %37, i8* %38), !dbg !46
  %40 = bitcast [16 x i64]* %"args$sd1_383" to i8*, !dbg !46
  %41 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !46
  call void (i8*, i32, ...) %41(i8* %40, i32 14), !dbg !46
  %42 = load i64, i64* %z_b_1_321, align 8, !dbg !46
  call void @llvm.dbg.value(metadata i64 %42, metadata !45, metadata !DIExpression()), !dbg !10
  %43 = load i64, i64* %z_b_0_320, align 8, !dbg !46
  call void @llvm.dbg.value(metadata i64 %43, metadata !45, metadata !DIExpression()), !dbg !10
  %44 = sub nsw i64 %43, 1, !dbg !46
  %45 = sub nsw i64 %42, %44, !dbg !46
  call void @llvm.dbg.declare(metadata i64* %z_b_2_322, metadata !45, metadata !DIExpression()), !dbg !10
  store i64 %45, i64* %z_b_2_322, align 8, !dbg !46
  %46 = load i64, i64* %z_b_0_320, align 8, !dbg !46
  call void @llvm.dbg.value(metadata i64 %46, metadata !45, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_b_3_323, metadata !45, metadata !DIExpression()), !dbg !10
  store i64 %46, i64* %z_b_3_323, align 8, !dbg !46
  %47 = bitcast i64* %z_b_2_322 to i8*, !dbg !46
  %48 = bitcast i64* @.C384_MAIN_ to i8*, !dbg !46
  %49 = bitcast i64* @.C385_MAIN_ to i8*, !dbg !46
  call void @llvm.dbg.declare(metadata i32* %allocstatus_317, metadata !47, metadata !DIExpression()), !dbg !10
  %50 = bitcast i32* %allocstatus_317 to i8*, !dbg !46
  %51 = bitcast [80 x i8]** %.Z0973_353 to i8*, !dbg !46
  %52 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !46
  %53 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !46
  %54 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !46
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %54(i8* %47, i8* %48, i8* %49, i8* %50, i8* %51, i8* null, i8* %52, i8* %53, i8* null, i64 0), !dbg !46
  %55 = load i32, i32* %allocstatus_317, align 4, !dbg !48
  call void @llvm.dbg.value(metadata i32 %55, metadata !47, metadata !DIExpression()), !dbg !10
  %56 = icmp sle i32 %55, 0, !dbg !48
  br i1 %56, label %L.LB1_397, label %L.LB1_554, !dbg !48

L.LB1_554:                                        ; preds = %L.LB1_394
  call void (...) @_mp_bcs_nest(), !dbg !49
  %57 = bitcast i32* @.C354_MAIN_ to i8*, !dbg !49
  %58 = bitcast [55 x i8]* @.C341_MAIN_ to i8*, !dbg !49
  %59 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !49
  call void (i8*, i8*, i64, ...) %59(i8* %57, i8* %58, i64 55), !dbg !49
  %60 = bitcast i32* @.C309_MAIN_ to i8*, !dbg !49
  %61 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !49
  %62 = bitcast [3 x i8]* @.C345_MAIN_ to i8*, !dbg !49
  %63 = bitcast i32 (...)* @f90io_encode_fmta to i32 (i8*, i8*, i8*, i64, ...)*, !dbg !49
  %64 = call i32 (i8*, i8*, i8*, i64, ...) %63(i8* %60, i8* %61, i8* %62, i64 3), !dbg !49
  store i32 %64, i32* %z__io_348, align 4, !dbg !49
  %65 = bitcast i32* @.C344_MAIN_ to i8*, !dbg !49
  %66 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !49
  %67 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !49
  %68 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !49
  %69 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %68(i8* %65, i8* null, i8* %66, i8* %67, i8* null, i8* null, i64 0), !dbg !49
  store i32 %69, i32* %z__io_348, align 4, !dbg !49
  %70 = bitcast i32* @.C309_MAIN_ to i8*, !dbg !49
  %71 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !49
  %72 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !49
  %73 = bitcast [37 x i8]* @.C355_MAIN_ to i8*, !dbg !49
  %74 = bitcast i32 (...)* @f90io_fmt_writea to i32 (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !49
  %75 = call i32 (i8*, i8*, i8*, i8*, i64, ...) %74(i8* %70, i8* %71, i8* %72, i8* %73, i64 37), !dbg !49
  store i32 %75, i32* %z__io_348, align 4, !dbg !49
  %76 = call i32 (...) @f90io_fmtw_end(), !dbg !49
  store i32 %76, i32* %z__io_348, align 4, !dbg !49
  call void (...) @_mp_ecs_nest(), !dbg !49
  %77 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !50
  %78 = bitcast void (...)* @f90_stop08a to void (i8*, i8*, i64, ...)*, !dbg !50
  call void (i8*, i8*, i64, ...) %78(i8* %77, i8* null, i64 0), !dbg !50
  br label %L.LB1_397

L.LB1_397:                                        ; preds = %L.LB1_554, %L.LB1_394
  %79 = load i32, i32* %argcount_316, align 4, !dbg !51
  call void @llvm.dbg.value(metadata i32 %79, metadata !41, metadata !DIExpression()), !dbg !10
  store i32 %79, i32* %.dY0001_400, align 4, !dbg !51
  call void @llvm.dbg.declare(metadata i32* %ix_319, metadata !52, metadata !DIExpression()), !dbg !10
  store i32 1, i32* %ix_319, align 4, !dbg !51
  %80 = load i32, i32* %.dY0001_400, align 4, !dbg !51
  %81 = icmp sle i32 %80, 0, !dbg !51
  br i1 %81, label %L.LB1_399, label %L.LB1_398, !dbg !51

L.LB1_398:                                        ; preds = %L.LB1_398, %L.LB1_397
  %82 = bitcast i32* %ix_319 to i8*, !dbg !53
  %83 = load [80 x i8]*, [80 x i8]** %.Z0973_353, align 8, !dbg !53
  call void @llvm.dbg.value(metadata [80 x i8]* %83, metadata !30, metadata !DIExpression()), !dbg !10
  %84 = bitcast [80 x i8]* %83 to i8*, !dbg !53
  %85 = getelementptr i8, i8* %84, i64 -80, !dbg !53
  %86 = load i32, i32* %ix_319, align 4, !dbg !53
  call void @llvm.dbg.value(metadata i32 %86, metadata !52, metadata !DIExpression()), !dbg !10
  %87 = sext i32 %86 to i64, !dbg !53
  %88 = bitcast [16 x i64]* %"args$sd1_383" to i8*, !dbg !53
  %89 = getelementptr i8, i8* %88, i64 56, !dbg !53
  %90 = bitcast i8* %89 to i64*, !dbg !53
  %91 = load i64, i64* %90, align 8, !dbg !53
  %92 = add nsw i64 %87, %91, !dbg !53
  %93 = mul nsw i64 %92, 80, !dbg !53
  %94 = getelementptr i8, i8* %85, i64 %93, !dbg !53
  %95 = bitcast i64* @.C381_MAIN_ to i8*, !dbg !53
  %96 = bitcast void (...)* @f90_get_cmd_arga to void (i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !53
  call void (i8*, i8*, i8*, i8*, i8*, i64, ...) %96(i8* %82, i8* %94, i8* null, i8* null, i8* %95, i64 80), !dbg !53
  %97 = load i32, i32* %ix_319, align 4, !dbg !54
  call void @llvm.dbg.value(metadata i32 %97, metadata !52, metadata !DIExpression()), !dbg !10
  %98 = add nsw i32 %97, 1, !dbg !54
  store i32 %98, i32* %ix_319, align 4, !dbg !54
  %99 = load i32, i32* %.dY0001_400, align 4, !dbg !54
  %100 = sub nsw i32 %99, 1, !dbg !54
  store i32 %100, i32* %.dY0001_400, align 4, !dbg !54
  %101 = load i32, i32* %.dY0001_400, align 4, !dbg !54
  %102 = icmp sgt i32 %101, 0, !dbg !54
  br i1 %102, label %L.LB1_398, label %L.LB1_399, !dbg !54

L.LB1_399:                                        ; preds = %L.LB1_398, %L.LB1_397
  %103 = load i32, i32* %argcount_316, align 4, !dbg !55
  call void @llvm.dbg.value(metadata i32 %103, metadata !41, metadata !DIExpression()), !dbg !10
  %104 = icmp sle i32 %103, 0, !dbg !55
  br i1 %104, label %L.LB1_401, label %L.LB1_555, !dbg !55

L.LB1_555:                                        ; preds = %L.LB1_399
  call void (...) @_mp_bcs_nest(), !dbg !56
  %105 = bitcast i32* @.C357_MAIN_ to i8*, !dbg !56
  %106 = bitcast [55 x i8]* @.C341_MAIN_ to i8*, !dbg !56
  %107 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !56
  call void (i8*, i8*, i64, ...) %107(i8* %105, i8* %106, i64 55), !dbg !56
  %108 = bitcast i32* @.C309_MAIN_ to i8*, !dbg !56
  %109 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !56
  %110 = bitcast [5 x i8]* @.C358_MAIN_ to i8*, !dbg !56
  %111 = bitcast i32 (...)* @f90io_encode_fmta to i32 (i8*, i8*, i8*, i64, ...)*, !dbg !56
  %112 = call i32 (i8*, i8*, i8*, i64, ...) %111(i8* %108, i8* %109, i8* %110, i64 5), !dbg !56
  store i32 %112, i32* %z__io_348, align 4, !dbg !56
  %113 = load [80 x i8]*, [80 x i8]** %.Z0973_353, align 8, !dbg !56
  call void @llvm.dbg.value(metadata [80 x i8]* %113, metadata !30, metadata !DIExpression()), !dbg !10
  %114 = bitcast [80 x i8]* %113 to i8*, !dbg !56
  %115 = bitcast [16 x i64]* %"args$sd1_383" to i8*, !dbg !56
  %116 = getelementptr i8, i8* %115, i64 56, !dbg !56
  %117 = bitcast i8* %116 to i64*, !dbg !56
  %118 = load i64, i64* %117, align 8, !dbg !56
  %119 = mul nsw i64 %118, 80, !dbg !56
  %120 = getelementptr i8, i8* %114, i64 %119, !dbg !56
  %121 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !56
  %122 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !56
  call void @llvm.dbg.declare(metadata i32* %rderr_318, metadata !57, metadata !DIExpression()), !dbg !10
  %123 = bitcast i32* %rderr_318 to i8*, !dbg !56
  %124 = bitcast i32 (...)* @f90io_fmtr_intern_inita to i32 (i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !56
  %125 = call i32 (i8*, i8*, i8*, i8*, i8*, i64, ...) %124(i8* %120, i8* %121, i8* %122, i8* %123, i8* null, i64 80), !dbg !56
  store i32 %125, i32* %z__io_348, align 4, !dbg !56
  %126 = bitcast i32* @.C310_MAIN_ to i8*, !dbg !56
  %127 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !56
  %128 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !56
  %129 = bitcast i32* %len_337 to i8*, !dbg !56
  %130 = bitcast i32 (...)* @f90io_fmt_reada to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !56
  %131 = call i32 (i8*, i8*, i8*, i8*, ...) %130(i8* %126, i8* %127, i8* %128, i8* %129), !dbg !56
  store i32 %131, i32* %z__io_348, align 4, !dbg !56
  %132 = call i32 (...) @f90io_fmtr_end(), !dbg !56
  store i32 %132, i32* %z__io_348, align 4, !dbg !56
  call void (...) @_mp_ecs_nest(), !dbg !56
  %133 = load i32, i32* %rderr_318, align 4, !dbg !58
  call void @llvm.dbg.value(metadata i32 %133, metadata !57, metadata !DIExpression()), !dbg !10
  %134 = icmp eq i32 %133, 0, !dbg !58
  br i1 %134, label %L.LB1_402, label %L.LB1_556, !dbg !58

L.LB1_556:                                        ; preds = %L.LB1_555
  call void (...) @_mp_bcs_nest(), !dbg !59
  %135 = bitcast i32* @.C340_MAIN_ to i8*, !dbg !59
  %136 = bitcast [55 x i8]* @.C341_MAIN_ to i8*, !dbg !59
  %137 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !59
  call void (i8*, i8*, i64, ...) %137(i8* %135, i8* %136, i64 55), !dbg !59
  %138 = bitcast i32* @.C309_MAIN_ to i8*, !dbg !59
  %139 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !59
  %140 = bitcast [3 x i8]* @.C345_MAIN_ to i8*, !dbg !59
  %141 = bitcast i32 (...)* @f90io_encode_fmta to i32 (i8*, i8*, i8*, i64, ...)*, !dbg !59
  %142 = call i32 (i8*, i8*, i8*, i64, ...) %141(i8* %138, i8* %139, i8* %140, i64 3), !dbg !59
  store i32 %142, i32* %z__io_348, align 4, !dbg !59
  %143 = bitcast i32* @.C344_MAIN_ to i8*, !dbg !59
  %144 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !59
  %145 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !59
  %146 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !59
  %147 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %146(i8* %143, i8* null, i8* %144, i8* %145, i8* null, i8* null, i64 0), !dbg !59
  store i32 %147, i32* %z__io_348, align 4, !dbg !59
  %148 = bitcast i32* @.C309_MAIN_ to i8*, !dbg !59
  %149 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !59
  %150 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !59
  %151 = bitcast [29 x i8]* @.C362_MAIN_ to i8*, !dbg !59
  %152 = bitcast i32 (...)* @f90io_fmt_writea to i32 (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !59
  %153 = call i32 (i8*, i8*, i8*, i8*, i64, ...) %152(i8* %148, i8* %149, i8* %150, i8* %151, i64 29), !dbg !59
  store i32 %153, i32* %z__io_348, align 4, !dbg !59
  %154 = call i32 (...) @f90io_fmtw_end(), !dbg !59
  store i32 %154, i32* %z__io_348, align 4, !dbg !59
  call void (...) @_mp_ecs_nest(), !dbg !59
  br label %L.LB1_402

L.LB1_402:                                        ; preds = %L.LB1_556, %L.LB1_555
  br label %L.LB1_401

L.LB1_401:                                        ; preds = %L.LB1_402, %L.LB1_399
  %155 = load i32, i32* %len_337, align 4, !dbg !60
  call void @llvm.dbg.value(metadata i32 %155, metadata !38, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i32* %n_314, metadata !61, metadata !DIExpression()), !dbg !10
  store i32 %155, i32* %n_314, align 4, !dbg !60
  %156 = load i32, i32* %len_337, align 4, !dbg !62
  call void @llvm.dbg.value(metadata i32 %156, metadata !38, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i32* %m_315, metadata !63, metadata !DIExpression()), !dbg !10
  store i32 %156, i32* %m_315, align 4, !dbg !62
  call void @llvm.dbg.declare(metadata i64* %z_b_4_326, metadata !45, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_4_326, align 8, !dbg !64
  %157 = load i32, i32* %n_314, align 4, !dbg !64
  call void @llvm.dbg.value(metadata i32 %157, metadata !61, metadata !DIExpression()), !dbg !10
  %158 = sext i32 %157 to i64, !dbg !64
  call void @llvm.dbg.declare(metadata i64* %z_b_5_327, metadata !45, metadata !DIExpression()), !dbg !10
  store i64 %158, i64* %z_b_5_327, align 8, !dbg !64
  %159 = load i64, i64* %z_b_5_327, align 8, !dbg !64
  call void @llvm.dbg.value(metadata i64 %159, metadata !45, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_71_333, metadata !45, metadata !DIExpression()), !dbg !10
  store i64 %159, i64* %z_e_71_333, align 8, !dbg !64
  call void @llvm.dbg.declare(metadata i64* %z_b_7_329, metadata !45, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_7_329, align 8, !dbg !64
  %160 = load i32, i32* %m_315, align 4, !dbg !64
  call void @llvm.dbg.value(metadata i32 %160, metadata !63, metadata !DIExpression()), !dbg !10
  %161 = sext i32 %160 to i64, !dbg !64
  call void @llvm.dbg.declare(metadata i64* %z_b_8_330, metadata !45, metadata !DIExpression()), !dbg !10
  store i64 %161, i64* %z_b_8_330, align 8, !dbg !64
  %162 = load i64, i64* %z_b_8_330, align 8, !dbg !64
  call void @llvm.dbg.value(metadata i64 %162, metadata !45, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_74_334, metadata !45, metadata !DIExpression()), !dbg !10
  store i64 %162, i64* %z_e_74_334, align 8, !dbg !64
  %163 = bitcast [22 x i64]* %"b$sd2_388" to i8*, !dbg !64
  %164 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !64
  %165 = bitcast i64* @.C389_MAIN_ to i8*, !dbg !64
  %166 = bitcast i64* @.C381_MAIN_ to i8*, !dbg !64
  %167 = bitcast i64* %z_b_4_326 to i8*, !dbg !64
  %168 = bitcast i64* %z_b_5_327 to i8*, !dbg !64
  %169 = bitcast i64* %z_b_7_329 to i8*, !dbg !64
  %170 = bitcast i64* %z_b_8_330 to i8*, !dbg !64
  %171 = bitcast void (...)* @f90_template2_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !64
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, ...) %171(i8* %163, i8* %164, i8* %165, i8* %166, i8* %167, i8* %168, i8* %169, i8* %170), !dbg !64
  %172 = bitcast [22 x i64]* %"b$sd2_388" to i8*, !dbg !64
  %173 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !64
  call void (i8*, i32, ...) %173(i8* %172, i32 27), !dbg !64
  %174 = load i64, i64* %z_b_5_327, align 8, !dbg !64
  call void @llvm.dbg.value(metadata i64 %174, metadata !45, metadata !DIExpression()), !dbg !10
  %175 = load i64, i64* %z_b_4_326, align 8, !dbg !64
  call void @llvm.dbg.value(metadata i64 %175, metadata !45, metadata !DIExpression()), !dbg !10
  %176 = sub nsw i64 %175, 1, !dbg !64
  %177 = sub nsw i64 %174, %176, !dbg !64
  call void @llvm.dbg.declare(metadata i64* %z_b_6_328, metadata !45, metadata !DIExpression()), !dbg !10
  store i64 %177, i64* %z_b_6_328, align 8, !dbg !64
  %178 = load i64, i64* %z_b_5_327, align 8, !dbg !64
  call void @llvm.dbg.value(metadata i64 %178, metadata !45, metadata !DIExpression()), !dbg !10
  %179 = load i64, i64* %z_b_4_326, align 8, !dbg !64
  call void @llvm.dbg.value(metadata i64 %179, metadata !45, metadata !DIExpression()), !dbg !10
  %180 = sub nsw i64 %179, 1, !dbg !64
  %181 = sub nsw i64 %178, %180, !dbg !64
  %182 = load i64, i64* %z_b_8_330, align 8, !dbg !64
  call void @llvm.dbg.value(metadata i64 %182, metadata !45, metadata !DIExpression()), !dbg !10
  %183 = load i64, i64* %z_b_7_329, align 8, !dbg !64
  call void @llvm.dbg.value(metadata i64 %183, metadata !45, metadata !DIExpression()), !dbg !10
  %184 = sub nsw i64 %183, 1, !dbg !64
  %185 = sub nsw i64 %182, %184, !dbg !64
  %186 = mul nsw i64 %181, %185, !dbg !64
  call void @llvm.dbg.declare(metadata i64* %z_b_9_331, metadata !45, metadata !DIExpression()), !dbg !10
  store i64 %186, i64* %z_b_9_331, align 8, !dbg !64
  %187 = load i64, i64* %z_b_4_326, align 8, !dbg !64
  call void @llvm.dbg.value(metadata i64 %187, metadata !45, metadata !DIExpression()), !dbg !10
  %188 = load i64, i64* %z_b_5_327, align 8, !dbg !64
  call void @llvm.dbg.value(metadata i64 %188, metadata !45, metadata !DIExpression()), !dbg !10
  %189 = load i64, i64* %z_b_4_326, align 8, !dbg !64
  call void @llvm.dbg.value(metadata i64 %189, metadata !45, metadata !DIExpression()), !dbg !10
  %190 = sub nsw i64 %189, 1, !dbg !64
  %191 = sub nsw i64 %188, %190, !dbg !64
  %192 = load i64, i64* %z_b_7_329, align 8, !dbg !64
  call void @llvm.dbg.value(metadata i64 %192, metadata !45, metadata !DIExpression()), !dbg !10
  %193 = mul nsw i64 %191, %192, !dbg !64
  %194 = add nsw i64 %187, %193, !dbg !64
  call void @llvm.dbg.declare(metadata i64* %z_b_10_332, metadata !45, metadata !DIExpression()), !dbg !10
  store i64 %194, i64* %z_b_10_332, align 8, !dbg !64
  %195 = bitcast i64* %z_b_9_331 to i8*, !dbg !64
  %196 = bitcast i64* @.C389_MAIN_ to i8*, !dbg !64
  %197 = bitcast i64* @.C381_MAIN_ to i8*, !dbg !64
  %198 = bitcast float** %.Z0983_363 to i8*, !dbg !64
  %199 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !64
  %200 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !64
  %201 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !64
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %201(i8* %195, i8* %196, i8* %197, i8* null, i8* %198, i8* null, i8* %199, i8* %200, i8* null, i64 0), !dbg !64
  %202 = bitcast i32* %n_314 to i8*, !dbg !65
  %203 = bitcast %astruct.dt88* %.uplevelArgPack0001_491 to i8**, !dbg !65
  store i8* %202, i8** %203, align 8, !dbg !65
  %204 = bitcast i32* %m_315 to i8*, !dbg !65
  %205 = bitcast %astruct.dt88* %.uplevelArgPack0001_491 to i8*, !dbg !65
  %206 = getelementptr i8, i8* %205, i64 8, !dbg !65
  %207 = bitcast i8* %206 to i8**, !dbg !65
  store i8* %204, i8** %207, align 8, !dbg !65
  %208 = bitcast float** %.Z0983_363 to i8*, !dbg !65
  %209 = bitcast %astruct.dt88* %.uplevelArgPack0001_491 to i8*, !dbg !65
  %210 = getelementptr i8, i8* %209, i64 16, !dbg !65
  %211 = bitcast i8* %210 to i8**, !dbg !65
  store i8* %208, i8** %211, align 8, !dbg !65
  %212 = bitcast float** %.Z0983_363 to i8*, !dbg !65
  %213 = bitcast %astruct.dt88* %.uplevelArgPack0001_491 to i8*, !dbg !65
  %214 = getelementptr i8, i8* %213, i64 24, !dbg !65
  %215 = bitcast i8* %214 to i8**, !dbg !65
  store i8* %212, i8** %215, align 8, !dbg !65
  %216 = bitcast i64* %z_b_4_326 to i8*, !dbg !65
  %217 = bitcast %astruct.dt88* %.uplevelArgPack0001_491 to i8*, !dbg !65
  %218 = getelementptr i8, i8* %217, i64 32, !dbg !65
  %219 = bitcast i8* %218 to i8**, !dbg !65
  store i8* %216, i8** %219, align 8, !dbg !65
  %220 = bitcast i64* %z_b_5_327 to i8*, !dbg !65
  %221 = bitcast %astruct.dt88* %.uplevelArgPack0001_491 to i8*, !dbg !65
  %222 = getelementptr i8, i8* %221, i64 40, !dbg !65
  %223 = bitcast i8* %222 to i8**, !dbg !65
  store i8* %220, i8** %223, align 8, !dbg !65
  %224 = bitcast i64* %z_e_71_333 to i8*, !dbg !65
  %225 = bitcast %astruct.dt88* %.uplevelArgPack0001_491 to i8*, !dbg !65
  %226 = getelementptr i8, i8* %225, i64 48, !dbg !65
  %227 = bitcast i8* %226 to i8**, !dbg !65
  store i8* %224, i8** %227, align 8, !dbg !65
  %228 = bitcast i64* %z_b_7_329 to i8*, !dbg !65
  %229 = bitcast %astruct.dt88* %.uplevelArgPack0001_491 to i8*, !dbg !65
  %230 = getelementptr i8, i8* %229, i64 56, !dbg !65
  %231 = bitcast i8* %230 to i8**, !dbg !65
  store i8* %228, i8** %231, align 8, !dbg !65
  %232 = bitcast i64* %z_b_8_330 to i8*, !dbg !65
  %233 = bitcast %astruct.dt88* %.uplevelArgPack0001_491 to i8*, !dbg !65
  %234 = getelementptr i8, i8* %233, i64 64, !dbg !65
  %235 = bitcast i8* %234 to i8**, !dbg !65
  store i8* %232, i8** %235, align 8, !dbg !65
  %236 = bitcast i64* %z_b_6_328 to i8*, !dbg !65
  %237 = bitcast %astruct.dt88* %.uplevelArgPack0001_491 to i8*, !dbg !65
  %238 = getelementptr i8, i8* %237, i64 72, !dbg !65
  %239 = bitcast i8* %238 to i8**, !dbg !65
  store i8* %236, i8** %239, align 8, !dbg !65
  %240 = bitcast i64* %z_e_74_334 to i8*, !dbg !65
  %241 = bitcast %astruct.dt88* %.uplevelArgPack0001_491 to i8*, !dbg !65
  %242 = getelementptr i8, i8* %241, i64 80, !dbg !65
  %243 = bitcast i8* %242 to i8**, !dbg !65
  store i8* %240, i8** %243, align 8, !dbg !65
  %244 = bitcast i64* %z_b_9_331 to i8*, !dbg !65
  %245 = bitcast %astruct.dt88* %.uplevelArgPack0001_491 to i8*, !dbg !65
  %246 = getelementptr i8, i8* %245, i64 88, !dbg !65
  %247 = bitcast i8* %246 to i8**, !dbg !65
  store i8* %244, i8** %247, align 8, !dbg !65
  %248 = bitcast i64* %z_b_10_332 to i8*, !dbg !65
  %249 = bitcast %astruct.dt88* %.uplevelArgPack0001_491 to i8*, !dbg !65
  %250 = getelementptr i8, i8* %249, i64 96, !dbg !65
  %251 = bitcast i8* %250 to i8**, !dbg !65
  store i8* %248, i8** %251, align 8, !dbg !65
  %252 = bitcast [22 x i64]* %"b$sd2_388" to i8*, !dbg !65
  %253 = bitcast %astruct.dt88* %.uplevelArgPack0001_491 to i8*, !dbg !65
  %254 = getelementptr i8, i8* %253, i64 104, !dbg !65
  %255 = bitcast i8* %254 to i8**, !dbg !65
  store i8* %252, i8** %255, align 8, !dbg !65
  br label %L.LB1_520, !dbg !65

L.LB1_520:                                        ; preds = %L.LB1_401
  %256 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L64_1_ to i64*, !dbg !65
  %257 = bitcast %astruct.dt88* %.uplevelArgPack0001_491 to i64*, !dbg !65
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %256, i64* %257), !dbg !65
  call void (...) @_mp_bcs_nest(), !dbg !66
  %258 = bitcast i32* @.C371_MAIN_ to i8*, !dbg !66
  %259 = bitcast [55 x i8]* @.C341_MAIN_ to i8*, !dbg !66
  %260 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !66
  call void (i8*, i8*, i64, ...) %260(i8* %258, i8* %259, i64 55), !dbg !66
  %261 = bitcast i32* @.C344_MAIN_ to i8*, !dbg !66
  %262 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !66
  %263 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !66
  %264 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !66
  %265 = call i32 (i8*, i8*, i8*, i8*, ...) %264(i8* %261, i8* null, i8* %262, i8* %263), !dbg !66
  store i32 %265, i32* %z__io_348, align 4, !dbg !66
  %266 = bitcast [9 x i8]* @.C373_MAIN_ to i8*, !dbg !66
  %267 = bitcast i32 (...)* @f90io_sc_ch_ldw to i32 (i8*, i32, i64, ...)*, !dbg !66
  %268 = call i32 (i8*, i32, i64, ...) %267(i8* %266, i32 14, i64 9), !dbg !66
  store i32 %268, i32* %z__io_348, align 4, !dbg !66
  %269 = bitcast [22 x i64]* %"b$sd2_388" to i8*, !dbg !66
  %270 = getelementptr i8, i8* %269, i64 160, !dbg !66
  %271 = bitcast i8* %270 to i64*, !dbg !66
  %272 = load i64, i64* %271, align 8, !dbg !66
  %273 = mul nsw i64 %272, 50, !dbg !66
  %274 = bitcast [22 x i64]* %"b$sd2_388" to i8*, !dbg !66
  %275 = getelementptr i8, i8* %274, i64 56, !dbg !66
  %276 = bitcast i8* %275 to i64*, !dbg !66
  %277 = load i64, i64* %276, align 8, !dbg !66
  %278 = add nsw i64 %273, %277, !dbg !66
  %279 = load float*, float** %.Z0983_363, align 8, !dbg !66
  call void @llvm.dbg.value(metadata float* %279, metadata !20, metadata !DIExpression()), !dbg !10
  %280 = bitcast float* %279 to i8*, !dbg !66
  %281 = getelementptr i8, i8* %280, i64 196, !dbg !66
  %282 = bitcast i8* %281 to float*, !dbg !66
  %283 = getelementptr float, float* %282, i64 %278, !dbg !66
  %284 = load float, float* %283, align 4, !dbg !66
  %285 = bitcast i32 (...)* @f90io_sc_f_ldw to i32 (float, i32, ...)*, !dbg !66
  %286 = call i32 (float, i32, ...) %285(float %284, i32 27), !dbg !66
  store i32 %286, i32* %z__io_348, align 4, !dbg !66
  %287 = call i32 (...) @f90io_ldw_end(), !dbg !66
  store i32 %287, i32* %z__io_348, align 4, !dbg !66
  call void (...) @_mp_ecs_nest(), !dbg !66
  %288 = load [80 x i8]*, [80 x i8]** %.Z0973_353, align 8, !dbg !67
  call void @llvm.dbg.value(metadata [80 x i8]* %288, metadata !30, metadata !DIExpression()), !dbg !10
  %289 = bitcast [80 x i8]* %288 to i8*, !dbg !67
  %290 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !67
  %291 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, i64, ...)*, !dbg !67
  call void (i8*, i8*, i8*, i8*, i64, i64, ...) %291(i8* null, i8* %289, i8* %290, i8* null, i64 80, i64 0), !dbg !67
  %292 = bitcast [80 x i8]** %.Z0973_353 to i8**, !dbg !67
  store i8* null, i8** %292, align 8, !dbg !67
  %293 = bitcast [16 x i64]* %"args$sd1_383" to i64*, !dbg !67
  store i64 0, i64* %293, align 8, !dbg !67
  %294 = load float*, float** %.Z0983_363, align 8, !dbg !67
  call void @llvm.dbg.value(metadata float* %294, metadata !20, metadata !DIExpression()), !dbg !10
  %295 = bitcast float* %294 to i8*, !dbg !67
  %296 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !67
  %297 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !67
  call void (i8*, i8*, i8*, i8*, i64, ...) %297(i8* null, i8* %295, i8* %296, i8* null, i64 0), !dbg !67
  %298 = bitcast float** %.Z0983_363 to i8**, !dbg !67
  store i8* null, i8** %298, align 8, !dbg !67
  %299 = bitcast [22 x i64]* %"b$sd2_388" to i64*, !dbg !67
  store i64 0, i64* %299, align 8, !dbg !67
  ret void, !dbg !18
}

define internal void @__nv_MAIN__F1L64_1_(i32* %__nv_MAIN__F1L64_1Arg0, i64* %__nv_MAIN__F1L64_1Arg1, i64* %__nv_MAIN__F1L64_1Arg2) #0 !dbg !68 {
L.entry:
  %__gtid___nv_MAIN__F1L64_1__575 = alloca i32, align 4
  %.i0000p_369 = alloca i32, align 4
  %j_368 = alloca i32, align 4
  %.du0002p_406 = alloca i32, align 4
  %.de0002p_407 = alloca i32, align 4
  %.di0002p_408 = alloca i32, align 4
  %.ds0002p_409 = alloca i32, align 4
  %.dl0002p_411 = alloca i32, align 4
  %.dl0002p.copy_569 = alloca i32, align 4
  %.de0002p.copy_570 = alloca i32, align 4
  %.ds0002p.copy_571 = alloca i32, align 4
  %.dX0002p_410 = alloca i32, align 4
  %.dY0002p_405 = alloca i32, align 4
  %.dY0003p_417 = alloca i32, align 4
  %i_367 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L64_1Arg0, metadata !71, metadata !DIExpression()), !dbg !72
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L64_1Arg1, metadata !73, metadata !DIExpression()), !dbg !72
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L64_1Arg2, metadata !74, metadata !DIExpression()), !dbg !72
  call void @llvm.dbg.value(metadata i32 1, metadata !75, metadata !DIExpression()), !dbg !72
  call void @llvm.dbg.value(metadata i32 2, metadata !76, metadata !DIExpression()), !dbg !72
  call void @llvm.dbg.value(metadata i32 0, metadata !77, metadata !DIExpression()), !dbg !72
  call void @llvm.dbg.value(metadata i32 1, metadata !78, metadata !DIExpression()), !dbg !72
  call void @llvm.dbg.value(metadata i32 2, metadata !79, metadata !DIExpression()), !dbg !72
  call void @llvm.dbg.value(metadata i32 0, metadata !80, metadata !DIExpression()), !dbg !72
  call void @llvm.dbg.value(metadata i32 1, metadata !81, metadata !DIExpression()), !dbg !72
  call void @llvm.dbg.value(metadata i32 2, metadata !82, metadata !DIExpression()), !dbg !72
  %0 = load i32, i32* %__nv_MAIN__F1L64_1Arg0, align 4, !dbg !83
  store i32 %0, i32* %__gtid___nv_MAIN__F1L64_1__575, align 4, !dbg !83
  br label %L.LB2_560

L.LB2_560:                                        ; preds = %L.entry
  br label %L.LB2_366

L.LB2_366:                                        ; preds = %L.LB2_560
  store i32 0, i32* %.i0000p_369, align 4, !dbg !84
  call void @llvm.dbg.declare(metadata i32* %j_368, metadata !85, metadata !DIExpression()), !dbg !83
  store i32 2, i32* %j_368, align 4, !dbg !84
  %1 = bitcast i64* %__nv_MAIN__F1L64_1Arg2 to i32**, !dbg !84
  %2 = load i32*, i32** %1, align 8, !dbg !84
  %3 = load i32, i32* %2, align 4, !dbg !84
  store i32 %3, i32* %.du0002p_406, align 4, !dbg !84
  %4 = bitcast i64* %__nv_MAIN__F1L64_1Arg2 to i32**, !dbg !84
  %5 = load i32*, i32** %4, align 8, !dbg !84
  %6 = load i32, i32* %5, align 4, !dbg !84
  store i32 %6, i32* %.de0002p_407, align 4, !dbg !84
  store i32 1, i32* %.di0002p_408, align 4, !dbg !84
  %7 = load i32, i32* %.di0002p_408, align 4, !dbg !84
  store i32 %7, i32* %.ds0002p_409, align 4, !dbg !84
  store i32 2, i32* %.dl0002p_411, align 4, !dbg !84
  %8 = load i32, i32* %.dl0002p_411, align 4, !dbg !84
  store i32 %8, i32* %.dl0002p.copy_569, align 4, !dbg !84
  %9 = load i32, i32* %.de0002p_407, align 4, !dbg !84
  store i32 %9, i32* %.de0002p.copy_570, align 4, !dbg !84
  %10 = load i32, i32* %.ds0002p_409, align 4, !dbg !84
  store i32 %10, i32* %.ds0002p.copy_571, align 4, !dbg !84
  %11 = load i32, i32* %__gtid___nv_MAIN__F1L64_1__575, align 4, !dbg !84
  %12 = bitcast i32* %.i0000p_369 to i64*, !dbg !84
  %13 = bitcast i32* %.dl0002p.copy_569 to i64*, !dbg !84
  %14 = bitcast i32* %.de0002p.copy_570 to i64*, !dbg !84
  %15 = bitcast i32* %.ds0002p.copy_571 to i64*, !dbg !84
  %16 = load i32, i32* %.ds0002p.copy_571, align 4, !dbg !84
  call void @__kmpc_for_static_init_4(i64* null, i32 %11, i32 34, i64* %12, i64* %13, i64* %14, i64* %15, i32 %16, i32 1), !dbg !84
  %17 = load i32, i32* %.dl0002p.copy_569, align 4, !dbg !84
  store i32 %17, i32* %.dl0002p_411, align 4, !dbg !84
  %18 = load i32, i32* %.de0002p.copy_570, align 4, !dbg !84
  store i32 %18, i32* %.de0002p_407, align 4, !dbg !84
  %19 = load i32, i32* %.ds0002p.copy_571, align 4, !dbg !84
  store i32 %19, i32* %.ds0002p_409, align 4, !dbg !84
  %20 = load i32, i32* %.dl0002p_411, align 4, !dbg !84
  store i32 %20, i32* %j_368, align 4, !dbg !84
  %21 = load i32, i32* %j_368, align 4, !dbg !84
  call void @llvm.dbg.value(metadata i32 %21, metadata !85, metadata !DIExpression()), !dbg !83
  store i32 %21, i32* %.dX0002p_410, align 4, !dbg !84
  %22 = load i32, i32* %.dX0002p_410, align 4, !dbg !84
  %23 = load i32, i32* %.du0002p_406, align 4, !dbg !84
  %24 = icmp sgt i32 %22, %23, !dbg !84
  br i1 %24, label %L.LB2_404, label %L.LB2_607, !dbg !84

L.LB2_607:                                        ; preds = %L.LB2_366
  %25 = load i32, i32* %.dX0002p_410, align 4, !dbg !84
  store i32 %25, i32* %j_368, align 4, !dbg !84
  %26 = load i32, i32* %.di0002p_408, align 4, !dbg !84
  %27 = load i32, i32* %.de0002p_407, align 4, !dbg !84
  %28 = load i32, i32* %.dX0002p_410, align 4, !dbg !84
  %29 = sub nsw i32 %27, %28, !dbg !84
  %30 = add nsw i32 %26, %29, !dbg !84
  %31 = load i32, i32* %.di0002p_408, align 4, !dbg !84
  %32 = sdiv i32 %30, %31, !dbg !84
  store i32 %32, i32* %.dY0002p_405, align 4, !dbg !84
  %33 = load i32, i32* %.dY0002p_405, align 4, !dbg !84
  %34 = icmp sle i32 %33, 0, !dbg !84
  br i1 %34, label %L.LB2_414, label %L.LB2_413, !dbg !84

L.LB2_413:                                        ; preds = %L.LB2_416, %L.LB2_607
  %35 = bitcast i64* %__nv_MAIN__F1L64_1Arg2 to i8*, !dbg !86
  %36 = getelementptr i8, i8* %35, i64 8, !dbg !86
  %37 = bitcast i8* %36 to i32**, !dbg !86
  %38 = load i32*, i32** %37, align 8, !dbg !86
  %39 = load i32, i32* %38, align 4, !dbg !86
  store i32 %39, i32* %.dY0003p_417, align 4, !dbg !86
  call void @llvm.dbg.declare(metadata i32* %i_367, metadata !87, metadata !DIExpression()), !dbg !83
  store i32 1, i32* %i_367, align 4, !dbg !86
  %40 = load i32, i32* %.dY0003p_417, align 4, !dbg !86
  %41 = icmp sle i32 %40, 0, !dbg !86
  br i1 %41, label %L.LB2_416, label %L.LB2_415, !dbg !86

L.LB2_415:                                        ; preds = %L.LB2_415, %L.LB2_413
  %42 = load i32, i32* %i_367, align 4, !dbg !88
  call void @llvm.dbg.value(metadata i32 %42, metadata !87, metadata !DIExpression()), !dbg !83
  %43 = sext i32 %42 to i64, !dbg !88
  %44 = load i32, i32* %j_368, align 4, !dbg !88
  call void @llvm.dbg.value(metadata i32 %44, metadata !85, metadata !DIExpression()), !dbg !83
  %45 = sext i32 %44 to i64, !dbg !88
  %46 = bitcast i64* %__nv_MAIN__F1L64_1Arg2 to i8*, !dbg !88
  %47 = getelementptr i8, i8* %46, i64 104, !dbg !88
  %48 = bitcast i8* %47 to i8**, !dbg !88
  %49 = load i8*, i8** %48, align 8, !dbg !88
  %50 = getelementptr i8, i8* %49, i64 160, !dbg !88
  %51 = bitcast i8* %50 to i64*, !dbg !88
  %52 = load i64, i64* %51, align 8, !dbg !88
  %53 = mul nsw i64 %45, %52, !dbg !88
  %54 = add nsw i64 %43, %53, !dbg !88
  %55 = bitcast i64* %__nv_MAIN__F1L64_1Arg2 to i8*, !dbg !88
  %56 = getelementptr i8, i8* %55, i64 104, !dbg !88
  %57 = bitcast i8* %56 to i8**, !dbg !88
  %58 = load i8*, i8** %57, align 8, !dbg !88
  %59 = getelementptr i8, i8* %58, i64 56, !dbg !88
  %60 = bitcast i8* %59 to i64*, !dbg !88
  %61 = load i64, i64* %60, align 8, !dbg !88
  %62 = add nsw i64 %54, %61, !dbg !88
  %63 = bitcast i64* %__nv_MAIN__F1L64_1Arg2 to i8*, !dbg !88
  %64 = getelementptr i8, i8* %63, i64 24, !dbg !88
  %65 = bitcast i8* %64 to i8***, !dbg !88
  %66 = load i8**, i8*** %65, align 8, !dbg !88
  %67 = load i8*, i8** %66, align 8, !dbg !88
  %68 = getelementptr i8, i8* %67, i64 -8, !dbg !88
  %69 = bitcast i8* %68 to float*, !dbg !88
  %70 = getelementptr float, float* %69, i64 %62, !dbg !88
  %71 = load float, float* %70, align 4, !dbg !88
  %72 = load i32, i32* %i_367, align 4, !dbg !88
  call void @llvm.dbg.value(metadata i32 %72, metadata !87, metadata !DIExpression()), !dbg !83
  %73 = sext i32 %72 to i64, !dbg !88
  %74 = load i32, i32* %j_368, align 4, !dbg !88
  call void @llvm.dbg.value(metadata i32 %74, metadata !85, metadata !DIExpression()), !dbg !83
  %75 = sext i32 %74 to i64, !dbg !88
  %76 = bitcast i64* %__nv_MAIN__F1L64_1Arg2 to i8*, !dbg !88
  %77 = getelementptr i8, i8* %76, i64 104, !dbg !88
  %78 = bitcast i8* %77 to i8**, !dbg !88
  %79 = load i8*, i8** %78, align 8, !dbg !88
  %80 = getelementptr i8, i8* %79, i64 160, !dbg !88
  %81 = bitcast i8* %80 to i64*, !dbg !88
  %82 = load i64, i64* %81, align 8, !dbg !88
  %83 = mul nsw i64 %75, %82, !dbg !88
  %84 = add nsw i64 %73, %83, !dbg !88
  %85 = bitcast i64* %__nv_MAIN__F1L64_1Arg2 to i8*, !dbg !88
  %86 = getelementptr i8, i8* %85, i64 104, !dbg !88
  %87 = bitcast i8* %86 to i8**, !dbg !88
  %88 = load i8*, i8** %87, align 8, !dbg !88
  %89 = getelementptr i8, i8* %88, i64 56, !dbg !88
  %90 = bitcast i8* %89 to i64*, !dbg !88
  %91 = load i64, i64* %90, align 8, !dbg !88
  %92 = add nsw i64 %84, %91, !dbg !88
  %93 = bitcast i64* %__nv_MAIN__F1L64_1Arg2 to i8*, !dbg !88
  %94 = getelementptr i8, i8* %93, i64 24, !dbg !88
  %95 = bitcast i8* %94 to i8***, !dbg !88
  %96 = load i8**, i8*** %95, align 8, !dbg !88
  %97 = load i8*, i8** %96, align 8, !dbg !88
  %98 = getelementptr i8, i8* %97, i64 -4, !dbg !88
  %99 = bitcast i8* %98 to float*, !dbg !88
  %100 = getelementptr float, float* %99, i64 %92, !dbg !88
  store float %71, float* %100, align 4, !dbg !88
  %101 = load i32, i32* %i_367, align 4, !dbg !89
  call void @llvm.dbg.value(metadata i32 %101, metadata !87, metadata !DIExpression()), !dbg !83
  %102 = add nsw i32 %101, 1, !dbg !89
  store i32 %102, i32* %i_367, align 4, !dbg !89
  %103 = load i32, i32* %.dY0003p_417, align 4, !dbg !89
  %104 = sub nsw i32 %103, 1, !dbg !89
  store i32 %104, i32* %.dY0003p_417, align 4, !dbg !89
  %105 = load i32, i32* %.dY0003p_417, align 4, !dbg !89
  %106 = icmp sgt i32 %105, 0, !dbg !89
  br i1 %106, label %L.LB2_415, label %L.LB2_416, !dbg !89

L.LB2_416:                                        ; preds = %L.LB2_415, %L.LB2_413
  %107 = load i32, i32* %.di0002p_408, align 4, !dbg !83
  %108 = load i32, i32* %j_368, align 4, !dbg !83
  call void @llvm.dbg.value(metadata i32 %108, metadata !85, metadata !DIExpression()), !dbg !83
  %109 = add nsw i32 %107, %108, !dbg !83
  store i32 %109, i32* %j_368, align 4, !dbg !83
  %110 = load i32, i32* %.dY0002p_405, align 4, !dbg !83
  %111 = sub nsw i32 %110, 1, !dbg !83
  store i32 %111, i32* %.dY0002p_405, align 4, !dbg !83
  %112 = load i32, i32* %.dY0002p_405, align 4, !dbg !83
  %113 = icmp sgt i32 %112, 0, !dbg !83
  br i1 %113, label %L.LB2_413, label %L.LB2_414, !dbg !83

L.LB2_414:                                        ; preds = %L.LB2_416, %L.LB2_607
  br label %L.LB2_404

L.LB2_404:                                        ; preds = %L.LB2_414, %L.LB2_366
  %114 = load i32, i32* %__gtid___nv_MAIN__F1L64_1__575, align 4, !dbg !83
  call void @__kmpc_for_static_fini(i64* null, i32 %114), !dbg !83
  br label %L.LB2_370

L.LB2_370:                                        ; preds = %L.LB2_404
  ret void, !dbg !83
}

declare void @__kmpc_for_static_fini(i64*, i32) #0

declare void @__kmpc_for_static_init_4(i64*, i32, i32, i64*, i64*, i64*, i64*, i32, i32) #0

declare void @f90_dealloc03a_i8(...) #0

declare signext i32 @f90io_ldw_end(...) #0

declare signext i32 @f90io_sc_f_ldw(...) #0

declare signext i32 @f90io_sc_ch_ldw(...) #0

declare signext i32 @f90io_print_init(...) #0

declare void @f90_template2_i8(...) #0

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
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB015-outofbounds-var-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb015_outofbounds_var_yes", scope: !2, file: !3, line: 28, type: !6, scopeLine: 28, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
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
!18 = !DILocation(line: 74, column: 1, scope: !5)
!19 = !DILocation(line: 28, column: 1, scope: !5)
!20 = !DILocalVariable(name: "b", scope: !5, file: !3, type: !21)
!21 = !DICompositeType(tag: DW_TAG_array_type, baseType: !22, size: 32, align: 32, elements: !23)
!22 = !DIBasicType(name: "real", size: 32, align: 32, encoding: DW_ATE_float)
!23 = !{!24, !24}
!24 = !DISubrange(count: 0, lowerBound: 1)
!25 = !DILocalVariable(scope: !5, file: !3, type: !26, flags: DIFlagArtificial)
!26 = !DICompositeType(tag: DW_TAG_array_type, baseType: !27, size: 1408, align: 64, elements: !28)
!27 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!28 = !{!29}
!29 = !DISubrange(count: 22, lowerBound: 1)
!30 = !DILocalVariable(name: "args", scope: !5, file: !3, type: !31)
!31 = !DICompositeType(tag: DW_TAG_array_type, baseType: !32, size: 640, align: 8, elements: !33)
!32 = !DIBasicType(name: "character", size: 640, align: 8, encoding: DW_ATE_signed)
!33 = !{!24}
!34 = !DILocalVariable(scope: !5, file: !3, type: !35, flags: DIFlagArtificial)
!35 = !DICompositeType(tag: DW_TAG_array_type, baseType: !27, size: 1024, align: 64, elements: !36)
!36 = !{!37}
!37 = !DISubrange(count: 16, lowerBound: 1)
!38 = !DILocalVariable(name: "len", scope: !5, file: !3, type: !9)
!39 = !DILocation(line: 35, column: 1, scope: !5)
!40 = !DILocation(line: 37, column: 1, scope: !5)
!41 = !DILocalVariable(name: "argcount", scope: !5, file: !3, type: !9)
!42 = !DILocation(line: 38, column: 1, scope: !5)
!43 = !DILocation(line: 39, column: 1, scope: !5)
!44 = !DILocalVariable(scope: !5, file: !3, type: !9, flags: DIFlagArtificial)
!45 = !DILocalVariable(scope: !5, file: !3, type: !27, flags: DIFlagArtificial)
!46 = !DILocation(line: 42, column: 1, scope: !5)
!47 = !DILocalVariable(name: "allocstatus", scope: !5, file: !3, type: !9)
!48 = !DILocation(line: 43, column: 1, scope: !5)
!49 = !DILocation(line: 44, column: 1, scope: !5)
!50 = !DILocation(line: 45, column: 1, scope: !5)
!51 = !DILocation(line: 48, column: 1, scope: !5)
!52 = !DILocalVariable(name: "ix", scope: !5, file: !3, type: !9)
!53 = !DILocation(line: 49, column: 1, scope: !5)
!54 = !DILocation(line: 50, column: 1, scope: !5)
!55 = !DILocation(line: 52, column: 1, scope: !5)
!56 = !DILocation(line: 53, column: 1, scope: !5)
!57 = !DILocalVariable(name: "rderr", scope: !5, file: !3, type: !9)
!58 = !DILocation(line: 54, column: 1, scope: !5)
!59 = !DILocation(line: 55, column: 1, scope: !5)
!60 = !DILocation(line: 59, column: 1, scope: !5)
!61 = !DILocalVariable(name: "n", scope: !5, file: !3, type: !9)
!62 = !DILocation(line: 60, column: 1, scope: !5)
!63 = !DILocalVariable(name: "m", scope: !5, file: !3, type: !9)
!64 = !DILocation(line: 62, column: 1, scope: !5)
!65 = !DILocation(line: 64, column: 1, scope: !5)
!66 = !DILocation(line: 71, column: 1, scope: !5)
!67 = !DILocation(line: 73, column: 1, scope: !5)
!68 = distinct !DISubprogram(name: "__nv_MAIN__F1L64_1", scope: !2, file: !3, line: 64, type: !69, scopeLine: 64, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!69 = !DISubroutineType(types: !70)
!70 = !{null, !9, !27, !27}
!71 = !DILocalVariable(name: "__nv_MAIN__F1L64_1Arg0", arg: 1, scope: !68, file: !3, type: !9)
!72 = !DILocation(line: 0, scope: !68)
!73 = !DILocalVariable(name: "__nv_MAIN__F1L64_1Arg1", arg: 2, scope: !68, file: !3, type: !27)
!74 = !DILocalVariable(name: "__nv_MAIN__F1L64_1Arg2", arg: 3, scope: !68, file: !3, type: !27)
!75 = !DILocalVariable(name: "omp_sched_static", scope: !68, file: !3, type: !9)
!76 = !DILocalVariable(name: "omp_sched_dynamic", scope: !68, file: !3, type: !9)
!77 = !DILocalVariable(name: "omp_proc_bind_false", scope: !68, file: !3, type: !9)
!78 = !DILocalVariable(name: "omp_proc_bind_true", scope: !68, file: !3, type: !9)
!79 = !DILocalVariable(name: "omp_proc_bind_master", scope: !68, file: !3, type: !9)
!80 = !DILocalVariable(name: "omp_lock_hint_none", scope: !68, file: !3, type: !9)
!81 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !68, file: !3, type: !9)
!82 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !68, file: !3, type: !9)
!83 = !DILocation(line: 69, column: 1, scope: !68)
!84 = !DILocation(line: 65, column: 1, scope: !68)
!85 = !DILocalVariable(name: "j", scope: !68, file: !3, type: !9)
!86 = !DILocation(line: 66, column: 1, scope: !68)
!87 = !DILocalVariable(name: "i", scope: !68, file: !3, type: !9)
!88 = !DILocation(line: 67, column: 1, scope: !68)
!89 = !DILocation(line: 68, column: 1, scope: !68)

; ModuleID = '/tmp/DRB022-reductionmissing-var-yes-b3a63a.ll'
source_filename = "/tmp/DRB022-reductionmissing-var-yes-b3a63a.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%astruct.dt86 = type <{ i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8* }>

@.C370_MAIN_ = internal constant [5 x i8] c"sum ="
@.C368_MAIN_ = internal constant i32 65
@.C290_MAIN_ = internal constant float 5.000000e-01
@.C307_MAIN_ = internal constant i32 27
@.C385_MAIN_ = internal constant i64 27
@.C358_MAIN_ = internal constant [29 x i8] c"Error, invalid integer value."
@.C357_MAIN_ = internal constant i32 44
@.C306_MAIN_ = internal constant i32 25
@.C353_MAIN_ = internal constant [5 x i8] c"(i10)"
@.C352_MAIN_ = internal constant i32 42
@.C377_MAIN_ = internal constant i64 4
@.C350_MAIN_ = internal constant [37 x i8] c"Allocation error, program terminated."
@.C349_MAIN_ = internal constant i32 33
@.C381_MAIN_ = internal constant i64 80
@.C380_MAIN_ = internal constant i64 14
@.C341_MAIN_ = internal constant [35 x i8] c"No command line arguments provided."
@.C339_MAIN_ = internal constant i32 6
@.C340_MAIN_ = internal constant [3 x i8] c"(a)"
@.C285_MAIN_ = internal constant i32 1
@.C305_MAIN_ = internal constant i32 14
@.C337_MAIN_ = internal constant [60 x i8] c"micro-benchmarks-fortran/DRB022-reductionmissing-var-yes.f95"
@.C308_MAIN_ = internal constant i32 28
@.C287_MAIN_ = internal constant float 0.000000e+00
@.C333_MAIN_ = internal constant i32 100
@.C286_MAIN_ = internal constant i64 1
@.C284_MAIN_ = internal constant i64 0
@.C283_MAIN_ = internal constant i32 0
@.C285___nv_MAIN__F1L56_1 = internal constant i32 1
@.C283___nv_MAIN__F1L56_1 = internal constant i32 0

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__534 = alloca i32, align 4
  %.Z0983_359 = alloca float*, align 8
  %"u$sd2_384" = alloca [22 x i64], align 8
  %.Z0973_348 = alloca [80 x i8]*, align 8
  %"args$sd1_379" = alloca [16 x i64], align 8
  %len_334 = alloca i32, align 4
  %getsum_316 = alloca float, align 4
  %argcount_311 = alloca i32, align 4
  %z__io_343 = alloca i32, align 4
  %z_b_0_317 = alloca i64, align 8
  %z_b_1_318 = alloca i64, align 8
  %z_e_61_321 = alloca i64, align 8
  %z_b_2_319 = alloca i64, align 8
  %z_b_3_320 = alloca i64, align 8
  %allocstatus_312 = alloca i32, align 4
  %.dY0001_396 = alloca i32, align 4
  %ix_314 = alloca i32, align 4
  %rderr_313 = alloca i32, align 4
  %z_b_4_323 = alloca i64, align 8
  %z_b_5_324 = alloca i64, align 8
  %z_e_71_330 = alloca i64, align 8
  %z_b_7_326 = alloca i64, align 8
  %z_b_8_327 = alloca i64, align 8
  %z_e_74_331 = alloca i64, align 8
  %z_b_6_325 = alloca i64, align 8
  %z_b_9_328 = alloca i64, align 8
  %z_b_10_329 = alloca i64, align 8
  %.dY0002_401 = alloca i32, align 4
  %i_309 = alloca i32, align 4
  %.dY0003_404 = alloca i32, align 4
  %j_310 = alloca i32, align 4
  %.uplevelArgPack0001_503 = alloca %astruct.dt86, align 16
  call void @llvm.dbg.value(metadata i32 1, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !14, metadata !DIExpression()), !dbg !10
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !15
  store i32 %0, i32* %__gtid_MAIN__534, align 4, !dbg !15
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !16
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !16
  call void (i8*, ...) %2(i8* %1), !dbg !16
  call void @llvm.dbg.declare(metadata float** %.Z0983_359, metadata !17, metadata !DIExpression(DW_OP_deref)), !dbg !10
  %3 = bitcast float** %.Z0983_359 to i8**, !dbg !16
  store i8* null, i8** %3, align 8, !dbg !16
  call void @llvm.dbg.declare(metadata [22 x i64]* %"u$sd2_384", metadata !22, metadata !DIExpression()), !dbg !10
  %4 = bitcast [22 x i64]* %"u$sd2_384" to i64*, !dbg !16
  store i64 0, i64* %4, align 8, !dbg !16
  call void @llvm.dbg.declare(metadata [80 x i8]** %.Z0973_348, metadata !27, metadata !DIExpression(DW_OP_deref)), !dbg !10
  %5 = bitcast [80 x i8]** %.Z0973_348 to i8**, !dbg !16
  store i8* null, i8** %5, align 8, !dbg !16
  call void @llvm.dbg.declare(metadata [16 x i64]* %"args$sd1_379", metadata !31, metadata !DIExpression()), !dbg !10
  %6 = bitcast [16 x i64]* %"args$sd1_379" to i64*, !dbg !16
  store i64 0, i64* %6, align 8, !dbg !16
  br label %L.LB1_429

L.LB1_429:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %len_334, metadata !35, metadata !DIExpression()), !dbg !10
  store i32 100, i32* %len_334, align 4, !dbg !36
  call void @llvm.dbg.declare(metadata float* %getsum_316, metadata !37, metadata !DIExpression()), !dbg !10
  store float 0.000000e+00, float* %getsum_316, align 4, !dbg !38
  %7 = call i32 (...) @f90_cmd_arg_cnt(), !dbg !39
  call void @llvm.dbg.declare(metadata i32* %argcount_311, metadata !40, metadata !DIExpression()), !dbg !10
  store i32 %7, i32* %argcount_311, align 4, !dbg !39
  %8 = load i32, i32* %argcount_311, align 4, !dbg !41
  call void @llvm.dbg.value(metadata i32 %8, metadata !40, metadata !DIExpression()), !dbg !10
  %9 = icmp ne i32 %8, 0, !dbg !41
  br i1 %9, label %L.LB1_390, label %L.LB1_554, !dbg !41

L.LB1_554:                                        ; preds = %L.LB1_429
  call void (...) @_mp_bcs_nest(), !dbg !42
  %10 = bitcast i32* @.C308_MAIN_ to i8*, !dbg !42
  %11 = bitcast [60 x i8]* @.C337_MAIN_ to i8*, !dbg !42
  %12 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !42
  call void (i8*, i8*, i64, ...) %12(i8* %10, i8* %11, i64 60), !dbg !42
  %13 = bitcast i32* @.C305_MAIN_ to i8*, !dbg !42
  %14 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !42
  %15 = bitcast [3 x i8]* @.C340_MAIN_ to i8*, !dbg !42
  %16 = bitcast i32 (...)* @f90io_encode_fmta to i32 (i8*, i8*, i8*, i64, ...)*, !dbg !42
  %17 = call i32 (i8*, i8*, i8*, i64, ...) %16(i8* %13, i8* %14, i8* %15, i64 3), !dbg !42
  call void @llvm.dbg.declare(metadata i32* %z__io_343, metadata !43, metadata !DIExpression()), !dbg !10
  store i32 %17, i32* %z__io_343, align 4, !dbg !42
  %18 = bitcast i32* @.C339_MAIN_ to i8*, !dbg !42
  %19 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !42
  %20 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !42
  %21 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !42
  %22 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %21(i8* %18, i8* null, i8* %19, i8* %20, i8* null, i8* null, i64 0), !dbg !42
  store i32 %22, i32* %z__io_343, align 4, !dbg !42
  %23 = bitcast i32* @.C305_MAIN_ to i8*, !dbg !42
  %24 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !42
  %25 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !42
  %26 = bitcast [35 x i8]* @.C341_MAIN_ to i8*, !dbg !42
  %27 = bitcast i32 (...)* @f90io_fmt_writea to i32 (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !42
  %28 = call i32 (i8*, i8*, i8*, i8*, i64, ...) %27(i8* %23, i8* %24, i8* %25, i8* %26, i64 35), !dbg !42
  store i32 %28, i32* %z__io_343, align 4, !dbg !42
  %29 = call i32 (...) @f90io_fmtw_end(), !dbg !42
  store i32 %29, i32* %z__io_343, align 4, !dbg !42
  call void (...) @_mp_ecs_nest(), !dbg !42
  br label %L.LB1_390

L.LB1_390:                                        ; preds = %L.LB1_554, %L.LB1_429
  call void @llvm.dbg.declare(metadata i64* %z_b_0_317, metadata !44, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_0_317, align 8, !dbg !45
  %30 = load i32, i32* %argcount_311, align 4, !dbg !45
  call void @llvm.dbg.value(metadata i32 %30, metadata !40, metadata !DIExpression()), !dbg !10
  %31 = sext i32 %30 to i64, !dbg !45
  call void @llvm.dbg.declare(metadata i64* %z_b_1_318, metadata !44, metadata !DIExpression()), !dbg !10
  store i64 %31, i64* %z_b_1_318, align 8, !dbg !45
  %32 = load i64, i64* %z_b_1_318, align 8, !dbg !45
  call void @llvm.dbg.value(metadata i64 %32, metadata !44, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_61_321, metadata !44, metadata !DIExpression()), !dbg !10
  store i64 %32, i64* %z_e_61_321, align 8, !dbg !45
  %33 = bitcast [16 x i64]* %"args$sd1_379" to i8*, !dbg !45
  %34 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !45
  %35 = bitcast i64* @.C380_MAIN_ to i8*, !dbg !45
  %36 = bitcast i64* @.C381_MAIN_ to i8*, !dbg !45
  %37 = bitcast i64* %z_b_0_317 to i8*, !dbg !45
  %38 = bitcast i64* %z_b_1_318 to i8*, !dbg !45
  %39 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !45
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %39(i8* %33, i8* %34, i8* %35, i8* %36, i8* %37, i8* %38), !dbg !45
  %40 = bitcast [16 x i64]* %"args$sd1_379" to i8*, !dbg !45
  %41 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !45
  call void (i8*, i32, ...) %41(i8* %40, i32 14), !dbg !45
  %42 = load i64, i64* %z_b_1_318, align 8, !dbg !45
  call void @llvm.dbg.value(metadata i64 %42, metadata !44, metadata !DIExpression()), !dbg !10
  %43 = load i64, i64* %z_b_0_317, align 8, !dbg !45
  call void @llvm.dbg.value(metadata i64 %43, metadata !44, metadata !DIExpression()), !dbg !10
  %44 = sub nsw i64 %43, 1, !dbg !45
  %45 = sub nsw i64 %42, %44, !dbg !45
  call void @llvm.dbg.declare(metadata i64* %z_b_2_319, metadata !44, metadata !DIExpression()), !dbg !10
  store i64 %45, i64* %z_b_2_319, align 8, !dbg !45
  %46 = load i64, i64* %z_b_0_317, align 8, !dbg !45
  call void @llvm.dbg.value(metadata i64 %46, metadata !44, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_b_3_320, metadata !44, metadata !DIExpression()), !dbg !10
  store i64 %46, i64* %z_b_3_320, align 8, !dbg !45
  %47 = bitcast i64* %z_b_2_319 to i8*, !dbg !45
  %48 = bitcast i64* @.C380_MAIN_ to i8*, !dbg !45
  %49 = bitcast i64* @.C381_MAIN_ to i8*, !dbg !45
  call void @llvm.dbg.declare(metadata i32* %allocstatus_312, metadata !46, metadata !DIExpression()), !dbg !10
  %50 = bitcast i32* %allocstatus_312 to i8*, !dbg !45
  %51 = bitcast [80 x i8]** %.Z0973_348 to i8*, !dbg !45
  %52 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !45
  %53 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !45
  %54 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !45
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %54(i8* %47, i8* %48, i8* %49, i8* %50, i8* %51, i8* null, i8* %52, i8* %53, i8* null, i64 0), !dbg !45
  %55 = load i32, i32* %allocstatus_312, align 4, !dbg !47
  call void @llvm.dbg.value(metadata i32 %55, metadata !46, metadata !DIExpression()), !dbg !10
  %56 = icmp sle i32 %55, 0, !dbg !47
  br i1 %56, label %L.LB1_393, label %L.LB1_555, !dbg !47

L.LB1_555:                                        ; preds = %L.LB1_390
  call void (...) @_mp_bcs_nest(), !dbg !48
  %57 = bitcast i32* @.C349_MAIN_ to i8*, !dbg !48
  %58 = bitcast [60 x i8]* @.C337_MAIN_ to i8*, !dbg !48
  %59 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !48
  call void (i8*, i8*, i64, ...) %59(i8* %57, i8* %58, i64 60), !dbg !48
  %60 = bitcast i32* @.C305_MAIN_ to i8*, !dbg !48
  %61 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !48
  %62 = bitcast [3 x i8]* @.C340_MAIN_ to i8*, !dbg !48
  %63 = bitcast i32 (...)* @f90io_encode_fmta to i32 (i8*, i8*, i8*, i64, ...)*, !dbg !48
  %64 = call i32 (i8*, i8*, i8*, i64, ...) %63(i8* %60, i8* %61, i8* %62, i64 3), !dbg !48
  store i32 %64, i32* %z__io_343, align 4, !dbg !48
  %65 = bitcast i32* @.C339_MAIN_ to i8*, !dbg !48
  %66 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !48
  %67 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !48
  %68 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !48
  %69 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %68(i8* %65, i8* null, i8* %66, i8* %67, i8* null, i8* null, i64 0), !dbg !48
  store i32 %69, i32* %z__io_343, align 4, !dbg !48
  %70 = bitcast i32* @.C305_MAIN_ to i8*, !dbg !48
  %71 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !48
  %72 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !48
  %73 = bitcast [37 x i8]* @.C350_MAIN_ to i8*, !dbg !48
  %74 = bitcast i32 (...)* @f90io_fmt_writea to i32 (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !48
  %75 = call i32 (i8*, i8*, i8*, i8*, i64, ...) %74(i8* %70, i8* %71, i8* %72, i8* %73, i64 37), !dbg !48
  store i32 %75, i32* %z__io_343, align 4, !dbg !48
  %76 = call i32 (...) @f90io_fmtw_end(), !dbg !48
  store i32 %76, i32* %z__io_343, align 4, !dbg !48
  call void (...) @_mp_ecs_nest(), !dbg !48
  %77 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !49
  %78 = bitcast void (...)* @f90_stop08a to void (i8*, i8*, i64, ...)*, !dbg !49
  call void (i8*, i8*, i64, ...) %78(i8* %77, i8* null, i64 0), !dbg !49
  br label %L.LB1_393

L.LB1_393:                                        ; preds = %L.LB1_555, %L.LB1_390
  %79 = load i32, i32* %argcount_311, align 4, !dbg !50
  call void @llvm.dbg.value(metadata i32 %79, metadata !40, metadata !DIExpression()), !dbg !10
  store i32 %79, i32* %.dY0001_396, align 4, !dbg !50
  call void @llvm.dbg.declare(metadata i32* %ix_314, metadata !51, metadata !DIExpression()), !dbg !10
  store i32 1, i32* %ix_314, align 4, !dbg !50
  %80 = load i32, i32* %.dY0001_396, align 4, !dbg !50
  %81 = icmp sle i32 %80, 0, !dbg !50
  br i1 %81, label %L.LB1_395, label %L.LB1_394, !dbg !50

L.LB1_394:                                        ; preds = %L.LB1_394, %L.LB1_393
  %82 = bitcast i32* %ix_314 to i8*, !dbg !52
  %83 = load [80 x i8]*, [80 x i8]** %.Z0973_348, align 8, !dbg !52
  call void @llvm.dbg.value(metadata [80 x i8]* %83, metadata !27, metadata !DIExpression()), !dbg !10
  %84 = bitcast [80 x i8]* %83 to i8*, !dbg !52
  %85 = getelementptr i8, i8* %84, i64 -80, !dbg !52
  %86 = load i32, i32* %ix_314, align 4, !dbg !52
  call void @llvm.dbg.value(metadata i32 %86, metadata !51, metadata !DIExpression()), !dbg !10
  %87 = sext i32 %86 to i64, !dbg !52
  %88 = bitcast [16 x i64]* %"args$sd1_379" to i8*, !dbg !52
  %89 = getelementptr i8, i8* %88, i64 56, !dbg !52
  %90 = bitcast i8* %89 to i64*, !dbg !52
  %91 = load i64, i64* %90, align 8, !dbg !52
  %92 = add nsw i64 %87, %91, !dbg !52
  %93 = mul nsw i64 %92, 80, !dbg !52
  %94 = getelementptr i8, i8* %85, i64 %93, !dbg !52
  %95 = bitcast i64* @.C377_MAIN_ to i8*, !dbg !52
  %96 = bitcast void (...)* @f90_get_cmd_arga to void (i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !52
  call void (i8*, i8*, i8*, i8*, i8*, i64, ...) %96(i8* %82, i8* %94, i8* null, i8* null, i8* %95, i64 80), !dbg !52
  %97 = load i32, i32* %ix_314, align 4, !dbg !53
  call void @llvm.dbg.value(metadata i32 %97, metadata !51, metadata !DIExpression()), !dbg !10
  %98 = add nsw i32 %97, 1, !dbg !53
  store i32 %98, i32* %ix_314, align 4, !dbg !53
  %99 = load i32, i32* %.dY0001_396, align 4, !dbg !53
  %100 = sub nsw i32 %99, 1, !dbg !53
  store i32 %100, i32* %.dY0001_396, align 4, !dbg !53
  %101 = load i32, i32* %.dY0001_396, align 4, !dbg !53
  %102 = icmp sgt i32 %101, 0, !dbg !53
  br i1 %102, label %L.LB1_394, label %L.LB1_395, !dbg !53

L.LB1_395:                                        ; preds = %L.LB1_394, %L.LB1_393
  %103 = load i32, i32* %argcount_311, align 4, !dbg !54
  call void @llvm.dbg.value(metadata i32 %103, metadata !40, metadata !DIExpression()), !dbg !10
  %104 = icmp sle i32 %103, 0, !dbg !54
  br i1 %104, label %L.LB1_397, label %L.LB1_556, !dbg !54

L.LB1_556:                                        ; preds = %L.LB1_395
  call void (...) @_mp_bcs_nest(), !dbg !55
  %105 = bitcast i32* @.C352_MAIN_ to i8*, !dbg !55
  %106 = bitcast [60 x i8]* @.C337_MAIN_ to i8*, !dbg !55
  %107 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !55
  call void (i8*, i8*, i64, ...) %107(i8* %105, i8* %106, i64 60), !dbg !55
  %108 = bitcast i32* @.C305_MAIN_ to i8*, !dbg !55
  %109 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !55
  %110 = bitcast [5 x i8]* @.C353_MAIN_ to i8*, !dbg !55
  %111 = bitcast i32 (...)* @f90io_encode_fmta to i32 (i8*, i8*, i8*, i64, ...)*, !dbg !55
  %112 = call i32 (i8*, i8*, i8*, i64, ...) %111(i8* %108, i8* %109, i8* %110, i64 5), !dbg !55
  store i32 %112, i32* %z__io_343, align 4, !dbg !55
  %113 = load [80 x i8]*, [80 x i8]** %.Z0973_348, align 8, !dbg !55
  call void @llvm.dbg.value(metadata [80 x i8]* %113, metadata !27, metadata !DIExpression()), !dbg !10
  %114 = bitcast [80 x i8]* %113 to i8*, !dbg !55
  %115 = bitcast [16 x i64]* %"args$sd1_379" to i8*, !dbg !55
  %116 = getelementptr i8, i8* %115, i64 56, !dbg !55
  %117 = bitcast i8* %116 to i64*, !dbg !55
  %118 = load i64, i64* %117, align 8, !dbg !55
  %119 = mul nsw i64 %118, 80, !dbg !55
  %120 = getelementptr i8, i8* %114, i64 %119, !dbg !55
  %121 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !55
  %122 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !55
  call void @llvm.dbg.declare(metadata i32* %rderr_313, metadata !56, metadata !DIExpression()), !dbg !10
  %123 = bitcast i32* %rderr_313 to i8*, !dbg !55
  %124 = bitcast i32 (...)* @f90io_fmtr_intern_inita to i32 (i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !55
  %125 = call i32 (i8*, i8*, i8*, i8*, i8*, i64, ...) %124(i8* %120, i8* %121, i8* %122, i8* %123, i8* null, i64 80), !dbg !55
  store i32 %125, i32* %z__io_343, align 4, !dbg !55
  %126 = bitcast i32* @.C306_MAIN_ to i8*, !dbg !55
  %127 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !55
  %128 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !55
  %129 = bitcast i32* %len_334 to i8*, !dbg !55
  %130 = bitcast i32 (...)* @f90io_fmt_reada to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !55
  %131 = call i32 (i8*, i8*, i8*, i8*, ...) %130(i8* %126, i8* %127, i8* %128, i8* %129), !dbg !55
  store i32 %131, i32* %z__io_343, align 4, !dbg !55
  %132 = call i32 (...) @f90io_fmtr_end(), !dbg !55
  store i32 %132, i32* %z__io_343, align 4, !dbg !55
  call void (...) @_mp_ecs_nest(), !dbg !55
  %133 = load i32, i32* %rderr_313, align 4, !dbg !57
  call void @llvm.dbg.value(metadata i32 %133, metadata !56, metadata !DIExpression()), !dbg !10
  %134 = icmp eq i32 %133, 0, !dbg !57
  br i1 %134, label %L.LB1_398, label %L.LB1_557, !dbg !57

L.LB1_557:                                        ; preds = %L.LB1_556
  call void (...) @_mp_bcs_nest(), !dbg !58
  %135 = bitcast i32* @.C357_MAIN_ to i8*, !dbg !58
  %136 = bitcast [60 x i8]* @.C337_MAIN_ to i8*, !dbg !58
  %137 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !58
  call void (i8*, i8*, i64, ...) %137(i8* %135, i8* %136, i64 60), !dbg !58
  %138 = bitcast i32* @.C305_MAIN_ to i8*, !dbg !58
  %139 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !58
  %140 = bitcast [3 x i8]* @.C340_MAIN_ to i8*, !dbg !58
  %141 = bitcast i32 (...)* @f90io_encode_fmta to i32 (i8*, i8*, i8*, i64, ...)*, !dbg !58
  %142 = call i32 (i8*, i8*, i8*, i64, ...) %141(i8* %138, i8* %139, i8* %140, i64 3), !dbg !58
  store i32 %142, i32* %z__io_343, align 4, !dbg !58
  %143 = bitcast i32* @.C339_MAIN_ to i8*, !dbg !58
  %144 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !58
  %145 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !58
  %146 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !58
  %147 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %146(i8* %143, i8* null, i8* %144, i8* %145, i8* null, i8* null, i64 0), !dbg !58
  store i32 %147, i32* %z__io_343, align 4, !dbg !58
  %148 = bitcast i32* @.C305_MAIN_ to i8*, !dbg !58
  %149 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !58
  %150 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !58
  %151 = bitcast [29 x i8]* @.C358_MAIN_ to i8*, !dbg !58
  %152 = bitcast i32 (...)* @f90io_fmt_writea to i32 (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !58
  %153 = call i32 (i8*, i8*, i8*, i8*, i64, ...) %152(i8* %148, i8* %149, i8* %150, i8* %151, i64 29), !dbg !58
  store i32 %153, i32* %z__io_343, align 4, !dbg !58
  %154 = call i32 (...) @f90io_fmtw_end(), !dbg !58
  store i32 %154, i32* %z__io_343, align 4, !dbg !58
  call void (...) @_mp_ecs_nest(), !dbg !58
  br label %L.LB1_398

L.LB1_398:                                        ; preds = %L.LB1_557, %L.LB1_556
  br label %L.LB1_397

L.LB1_397:                                        ; preds = %L.LB1_398, %L.LB1_395
  call void @llvm.dbg.declare(metadata i64* %z_b_4_323, metadata !44, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_4_323, align 8, !dbg !59
  %155 = load i32, i32* %len_334, align 4, !dbg !59
  call void @llvm.dbg.value(metadata i32 %155, metadata !35, metadata !DIExpression()), !dbg !10
  %156 = sext i32 %155 to i64, !dbg !59
  call void @llvm.dbg.declare(metadata i64* %z_b_5_324, metadata !44, metadata !DIExpression()), !dbg !10
  store i64 %156, i64* %z_b_5_324, align 8, !dbg !59
  %157 = load i64, i64* %z_b_5_324, align 8, !dbg !59
  call void @llvm.dbg.value(metadata i64 %157, metadata !44, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_71_330, metadata !44, metadata !DIExpression()), !dbg !10
  store i64 %157, i64* %z_e_71_330, align 8, !dbg !59
  call void @llvm.dbg.declare(metadata i64* %z_b_7_326, metadata !44, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_7_326, align 8, !dbg !59
  %158 = load i32, i32* %len_334, align 4, !dbg !59
  call void @llvm.dbg.value(metadata i32 %158, metadata !35, metadata !DIExpression()), !dbg !10
  %159 = sext i32 %158 to i64, !dbg !59
  call void @llvm.dbg.declare(metadata i64* %z_b_8_327, metadata !44, metadata !DIExpression()), !dbg !10
  store i64 %159, i64* %z_b_8_327, align 8, !dbg !59
  %160 = load i64, i64* %z_b_8_327, align 8, !dbg !59
  call void @llvm.dbg.value(metadata i64 %160, metadata !44, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_74_331, metadata !44, metadata !DIExpression()), !dbg !10
  store i64 %160, i64* %z_e_74_331, align 8, !dbg !59
  %161 = bitcast [22 x i64]* %"u$sd2_384" to i8*, !dbg !59
  %162 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !59
  %163 = bitcast i64* @.C385_MAIN_ to i8*, !dbg !59
  %164 = bitcast i64* @.C377_MAIN_ to i8*, !dbg !59
  %165 = bitcast i64* %z_b_4_323 to i8*, !dbg !59
  %166 = bitcast i64* %z_b_5_324 to i8*, !dbg !59
  %167 = bitcast i64* %z_b_7_326 to i8*, !dbg !59
  %168 = bitcast i64* %z_b_8_327 to i8*, !dbg !59
  %169 = bitcast void (...)* @f90_template2_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !59
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, ...) %169(i8* %161, i8* %162, i8* %163, i8* %164, i8* %165, i8* %166, i8* %167, i8* %168), !dbg !59
  %170 = bitcast [22 x i64]* %"u$sd2_384" to i8*, !dbg !59
  %171 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !59
  call void (i8*, i32, ...) %171(i8* %170, i32 27), !dbg !59
  %172 = load i64, i64* %z_b_5_324, align 8, !dbg !59
  call void @llvm.dbg.value(metadata i64 %172, metadata !44, metadata !DIExpression()), !dbg !10
  %173 = load i64, i64* %z_b_4_323, align 8, !dbg !59
  call void @llvm.dbg.value(metadata i64 %173, metadata !44, metadata !DIExpression()), !dbg !10
  %174 = sub nsw i64 %173, 1, !dbg !59
  %175 = sub nsw i64 %172, %174, !dbg !59
  call void @llvm.dbg.declare(metadata i64* %z_b_6_325, metadata !44, metadata !DIExpression()), !dbg !10
  store i64 %175, i64* %z_b_6_325, align 8, !dbg !59
  %176 = load i64, i64* %z_b_5_324, align 8, !dbg !59
  call void @llvm.dbg.value(metadata i64 %176, metadata !44, metadata !DIExpression()), !dbg !10
  %177 = load i64, i64* %z_b_4_323, align 8, !dbg !59
  call void @llvm.dbg.value(metadata i64 %177, metadata !44, metadata !DIExpression()), !dbg !10
  %178 = sub nsw i64 %177, 1, !dbg !59
  %179 = sub nsw i64 %176, %178, !dbg !59
  %180 = load i64, i64* %z_b_8_327, align 8, !dbg !59
  call void @llvm.dbg.value(metadata i64 %180, metadata !44, metadata !DIExpression()), !dbg !10
  %181 = load i64, i64* %z_b_7_326, align 8, !dbg !59
  call void @llvm.dbg.value(metadata i64 %181, metadata !44, metadata !DIExpression()), !dbg !10
  %182 = sub nsw i64 %181, 1, !dbg !59
  %183 = sub nsw i64 %180, %182, !dbg !59
  %184 = mul nsw i64 %179, %183, !dbg !59
  call void @llvm.dbg.declare(metadata i64* %z_b_9_328, metadata !44, metadata !DIExpression()), !dbg !10
  store i64 %184, i64* %z_b_9_328, align 8, !dbg !59
  %185 = load i64, i64* %z_b_4_323, align 8, !dbg !59
  call void @llvm.dbg.value(metadata i64 %185, metadata !44, metadata !DIExpression()), !dbg !10
  %186 = load i64, i64* %z_b_5_324, align 8, !dbg !59
  call void @llvm.dbg.value(metadata i64 %186, metadata !44, metadata !DIExpression()), !dbg !10
  %187 = load i64, i64* %z_b_4_323, align 8, !dbg !59
  call void @llvm.dbg.value(metadata i64 %187, metadata !44, metadata !DIExpression()), !dbg !10
  %188 = sub nsw i64 %187, 1, !dbg !59
  %189 = sub nsw i64 %186, %188, !dbg !59
  %190 = load i64, i64* %z_b_7_326, align 8, !dbg !59
  call void @llvm.dbg.value(metadata i64 %190, metadata !44, metadata !DIExpression()), !dbg !10
  %191 = mul nsw i64 %189, %190, !dbg !59
  %192 = add nsw i64 %185, %191, !dbg !59
  call void @llvm.dbg.declare(metadata i64* %z_b_10_329, metadata !44, metadata !DIExpression()), !dbg !10
  store i64 %192, i64* %z_b_10_329, align 8, !dbg !59
  %193 = bitcast i64* %z_b_9_328 to i8*, !dbg !59
  %194 = bitcast i64* @.C385_MAIN_ to i8*, !dbg !59
  %195 = bitcast i64* @.C377_MAIN_ to i8*, !dbg !59
  %196 = bitcast float** %.Z0983_359 to i8*, !dbg !59
  %197 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !59
  %198 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !59
  %199 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !59
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %199(i8* %193, i8* %194, i8* %195, i8* null, i8* %196, i8* null, i8* %197, i8* %198, i8* null, i64 0), !dbg !59
  %200 = load i32, i32* %len_334, align 4, !dbg !60
  call void @llvm.dbg.value(metadata i32 %200, metadata !35, metadata !DIExpression()), !dbg !10
  store i32 %200, i32* %.dY0002_401, align 4, !dbg !60
  call void @llvm.dbg.declare(metadata i32* %i_309, metadata !61, metadata !DIExpression()), !dbg !10
  store i32 1, i32* %i_309, align 4, !dbg !60
  %201 = load i32, i32* %.dY0002_401, align 4, !dbg !60
  %202 = icmp sle i32 %201, 0, !dbg !60
  br i1 %202, label %L.LB1_400, label %L.LB1_399, !dbg !60

L.LB1_399:                                        ; preds = %L.LB1_403, %L.LB1_397
  %203 = load i32, i32* %len_334, align 4, !dbg !62
  call void @llvm.dbg.value(metadata i32 %203, metadata !35, metadata !DIExpression()), !dbg !10
  store i32 %203, i32* %.dY0003_404, align 4, !dbg !62
  call void @llvm.dbg.declare(metadata i32* %j_310, metadata !63, metadata !DIExpression()), !dbg !10
  store i32 1, i32* %j_310, align 4, !dbg !62
  %204 = load i32, i32* %.dY0003_404, align 4, !dbg !62
  %205 = icmp sle i32 %204, 0, !dbg !62
  br i1 %205, label %L.LB1_403, label %L.LB1_402, !dbg !62

L.LB1_402:                                        ; preds = %L.LB1_402, %L.LB1_399
  %206 = load i32, i32* %i_309, align 4, !dbg !64
  call void @llvm.dbg.value(metadata i32 %206, metadata !61, metadata !DIExpression()), !dbg !10
  %207 = sext i32 %206 to i64, !dbg !64
  %208 = load i32, i32* %j_310, align 4, !dbg !64
  call void @llvm.dbg.value(metadata i32 %208, metadata !63, metadata !DIExpression()), !dbg !10
  %209 = sext i32 %208 to i64, !dbg !64
  %210 = bitcast [22 x i64]* %"u$sd2_384" to i8*, !dbg !64
  %211 = getelementptr i8, i8* %210, i64 160, !dbg !64
  %212 = bitcast i8* %211 to i64*, !dbg !64
  %213 = load i64, i64* %212, align 8, !dbg !64
  %214 = mul nsw i64 %209, %213, !dbg !64
  %215 = add nsw i64 %207, %214, !dbg !64
  %216 = bitcast [22 x i64]* %"u$sd2_384" to i8*, !dbg !64
  %217 = getelementptr i8, i8* %216, i64 56, !dbg !64
  %218 = bitcast i8* %217 to i64*, !dbg !64
  %219 = load i64, i64* %218, align 8, !dbg !64
  %220 = add nsw i64 %215, %219, !dbg !64
  %221 = load float*, float** %.Z0983_359, align 8, !dbg !64
  call void @llvm.dbg.value(metadata float* %221, metadata !17, metadata !DIExpression()), !dbg !10
  %222 = bitcast float* %221 to i8*, !dbg !64
  %223 = getelementptr i8, i8* %222, i64 -4, !dbg !64
  %224 = bitcast i8* %223 to float*, !dbg !64
  %225 = getelementptr float, float* %224, i64 %220, !dbg !64
  store float 5.000000e-01, float* %225, align 4, !dbg !64
  %226 = load i32, i32* %j_310, align 4, !dbg !65
  call void @llvm.dbg.value(metadata i32 %226, metadata !63, metadata !DIExpression()), !dbg !10
  %227 = add nsw i32 %226, 1, !dbg !65
  store i32 %227, i32* %j_310, align 4, !dbg !65
  %228 = load i32, i32* %.dY0003_404, align 4, !dbg !65
  %229 = sub nsw i32 %228, 1, !dbg !65
  store i32 %229, i32* %.dY0003_404, align 4, !dbg !65
  %230 = load i32, i32* %.dY0003_404, align 4, !dbg !65
  %231 = icmp sgt i32 %230, 0, !dbg !65
  br i1 %231, label %L.LB1_402, label %L.LB1_403, !dbg !65

L.LB1_403:                                        ; preds = %L.LB1_402, %L.LB1_399
  %232 = load i32, i32* %i_309, align 4, !dbg !66
  call void @llvm.dbg.value(metadata i32 %232, metadata !61, metadata !DIExpression()), !dbg !10
  %233 = add nsw i32 %232, 1, !dbg !66
  store i32 %233, i32* %i_309, align 4, !dbg !66
  %234 = load i32, i32* %.dY0002_401, align 4, !dbg !66
  %235 = sub nsw i32 %234, 1, !dbg !66
  store i32 %235, i32* %.dY0002_401, align 4, !dbg !66
  %236 = load i32, i32* %.dY0002_401, align 4, !dbg !66
  %237 = icmp sgt i32 %236, 0, !dbg !66
  br i1 %237, label %L.LB1_399, label %L.LB1_400, !dbg !66

L.LB1_400:                                        ; preds = %L.LB1_403, %L.LB1_397
  %238 = bitcast i32* %len_334 to i8*, !dbg !67
  %239 = bitcast %astruct.dt86* %.uplevelArgPack0001_503 to i8**, !dbg !67
  store i8* %238, i8** %239, align 8, !dbg !67
  %240 = bitcast float** %.Z0983_359 to i8*, !dbg !67
  %241 = bitcast %astruct.dt86* %.uplevelArgPack0001_503 to i8*, !dbg !67
  %242 = getelementptr i8, i8* %241, i64 8, !dbg !67
  %243 = bitcast i8* %242 to i8**, !dbg !67
  store i8* %240, i8** %243, align 8, !dbg !67
  %244 = bitcast float** %.Z0983_359 to i8*, !dbg !67
  %245 = bitcast %astruct.dt86* %.uplevelArgPack0001_503 to i8*, !dbg !67
  %246 = getelementptr i8, i8* %245, i64 16, !dbg !67
  %247 = bitcast i8* %246 to i8**, !dbg !67
  store i8* %244, i8** %247, align 8, !dbg !67
  %248 = bitcast i64* %z_b_4_323 to i8*, !dbg !67
  %249 = bitcast %astruct.dt86* %.uplevelArgPack0001_503 to i8*, !dbg !67
  %250 = getelementptr i8, i8* %249, i64 24, !dbg !67
  %251 = bitcast i8* %250 to i8**, !dbg !67
  store i8* %248, i8** %251, align 8, !dbg !67
  %252 = bitcast i64* %z_b_5_324 to i8*, !dbg !67
  %253 = bitcast %astruct.dt86* %.uplevelArgPack0001_503 to i8*, !dbg !67
  %254 = getelementptr i8, i8* %253, i64 32, !dbg !67
  %255 = bitcast i8* %254 to i8**, !dbg !67
  store i8* %252, i8** %255, align 8, !dbg !67
  %256 = bitcast i64* %z_e_71_330 to i8*, !dbg !67
  %257 = bitcast %astruct.dt86* %.uplevelArgPack0001_503 to i8*, !dbg !67
  %258 = getelementptr i8, i8* %257, i64 40, !dbg !67
  %259 = bitcast i8* %258 to i8**, !dbg !67
  store i8* %256, i8** %259, align 8, !dbg !67
  %260 = bitcast i64* %z_b_7_326 to i8*, !dbg !67
  %261 = bitcast %astruct.dt86* %.uplevelArgPack0001_503 to i8*, !dbg !67
  %262 = getelementptr i8, i8* %261, i64 48, !dbg !67
  %263 = bitcast i8* %262 to i8**, !dbg !67
  store i8* %260, i8** %263, align 8, !dbg !67
  %264 = bitcast i64* %z_b_8_327 to i8*, !dbg !67
  %265 = bitcast %astruct.dt86* %.uplevelArgPack0001_503 to i8*, !dbg !67
  %266 = getelementptr i8, i8* %265, i64 56, !dbg !67
  %267 = bitcast i8* %266 to i8**, !dbg !67
  store i8* %264, i8** %267, align 8, !dbg !67
  %268 = bitcast i64* %z_b_6_325 to i8*, !dbg !67
  %269 = bitcast %astruct.dt86* %.uplevelArgPack0001_503 to i8*, !dbg !67
  %270 = getelementptr i8, i8* %269, i64 64, !dbg !67
  %271 = bitcast i8* %270 to i8**, !dbg !67
  store i8* %268, i8** %271, align 8, !dbg !67
  %272 = bitcast i64* %z_e_74_331 to i8*, !dbg !67
  %273 = bitcast %astruct.dt86* %.uplevelArgPack0001_503 to i8*, !dbg !67
  %274 = getelementptr i8, i8* %273, i64 72, !dbg !67
  %275 = bitcast i8* %274 to i8**, !dbg !67
  store i8* %272, i8** %275, align 8, !dbg !67
  %276 = bitcast i64* %z_b_9_328 to i8*, !dbg !67
  %277 = bitcast %astruct.dt86* %.uplevelArgPack0001_503 to i8*, !dbg !67
  %278 = getelementptr i8, i8* %277, i64 80, !dbg !67
  %279 = bitcast i8* %278 to i8**, !dbg !67
  store i8* %276, i8** %279, align 8, !dbg !67
  %280 = bitcast i64* %z_b_10_329 to i8*, !dbg !67
  %281 = bitcast %astruct.dt86* %.uplevelArgPack0001_503 to i8*, !dbg !67
  %282 = getelementptr i8, i8* %281, i64 88, !dbg !67
  %283 = bitcast i8* %282 to i8**, !dbg !67
  store i8* %280, i8** %283, align 8, !dbg !67
  %284 = bitcast float* %getsum_316 to i8*, !dbg !67
  %285 = bitcast %astruct.dt86* %.uplevelArgPack0001_503 to i8*, !dbg !67
  %286 = getelementptr i8, i8* %285, i64 96, !dbg !67
  %287 = bitcast i8* %286 to i8**, !dbg !67
  store i8* %284, i8** %287, align 8, !dbg !67
  %288 = bitcast [22 x i64]* %"u$sd2_384" to i8*, !dbg !67
  %289 = bitcast %astruct.dt86* %.uplevelArgPack0001_503 to i8*, !dbg !67
  %290 = getelementptr i8, i8* %289, i64 104, !dbg !67
  %291 = bitcast i8* %290 to i8**, !dbg !67
  store i8* %288, i8** %291, align 8, !dbg !67
  br label %L.LB1_532, !dbg !67

L.LB1_532:                                        ; preds = %L.LB1_400
  %292 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L56_1_ to i64*, !dbg !67
  %293 = bitcast %astruct.dt86* %.uplevelArgPack0001_503 to i64*, !dbg !67
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %292, i64* %293), !dbg !67
  call void (...) @_mp_bcs_nest(), !dbg !68
  %294 = bitcast i32* @.C368_MAIN_ to i8*, !dbg !68
  %295 = bitcast [60 x i8]* @.C337_MAIN_ to i8*, !dbg !68
  %296 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !68
  call void (i8*, i8*, i64, ...) %296(i8* %294, i8* %295, i64 60), !dbg !68
  %297 = bitcast i32* @.C339_MAIN_ to i8*, !dbg !68
  %298 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !68
  %299 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !68
  %300 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !68
  %301 = call i32 (i8*, i8*, i8*, i8*, ...) %300(i8* %297, i8* null, i8* %298, i8* %299), !dbg !68
  store i32 %301, i32* %z__io_343, align 4, !dbg !68
  %302 = bitcast [5 x i8]* @.C370_MAIN_ to i8*, !dbg !68
  %303 = bitcast i32 (...)* @f90io_sc_ch_ldw to i32 (i8*, i32, i64, ...)*, !dbg !68
  %304 = call i32 (i8*, i32, i64, ...) %303(i8* %302, i32 14, i64 5), !dbg !68
  store i32 %304, i32* %z__io_343, align 4, !dbg !68
  %305 = load float, float* %getsum_316, align 4, !dbg !68
  call void @llvm.dbg.value(metadata float %305, metadata !37, metadata !DIExpression()), !dbg !10
  %306 = bitcast i32 (...)* @f90io_sc_f_ldw to i32 (float, i32, ...)*, !dbg !68
  %307 = call i32 (float, i32, ...) %306(float %305, i32 27), !dbg !68
  store i32 %307, i32* %z__io_343, align 4, !dbg !68
  %308 = call i32 (...) @f90io_ldw_end(), !dbg !68
  store i32 %308, i32* %z__io_343, align 4, !dbg !68
  call void (...) @_mp_ecs_nest(), !dbg !68
  %309 = load [80 x i8]*, [80 x i8]** %.Z0973_348, align 8, !dbg !69
  call void @llvm.dbg.value(metadata [80 x i8]* %309, metadata !27, metadata !DIExpression()), !dbg !10
  %310 = bitcast [80 x i8]* %309 to i8*, !dbg !69
  %311 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !69
  %312 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, i64, ...)*, !dbg !69
  call void (i8*, i8*, i8*, i8*, i64, i64, ...) %312(i8* null, i8* %310, i8* %311, i8* null, i64 80, i64 0), !dbg !69
  %313 = bitcast [80 x i8]** %.Z0973_348 to i8**, !dbg !69
  store i8* null, i8** %313, align 8, !dbg !69
  %314 = bitcast [16 x i64]* %"args$sd1_379" to i64*, !dbg !69
  store i64 0, i64* %314, align 8, !dbg !69
  %315 = load float*, float** %.Z0983_359, align 8, !dbg !69
  call void @llvm.dbg.value(metadata float* %315, metadata !17, metadata !DIExpression()), !dbg !10
  %316 = bitcast float* %315 to i8*, !dbg !69
  %317 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !69
  %318 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !69
  call void (i8*, i8*, i8*, i8*, i64, ...) %318(i8* null, i8* %316, i8* %317, i8* null, i64 0), !dbg !69
  %319 = bitcast float** %.Z0983_359 to i8**, !dbg !69
  store i8* null, i8** %319, align 8, !dbg !69
  %320 = bitcast [22 x i64]* %"u$sd2_384" to i64*, !dbg !69
  store i64 0, i64* %320, align 8, !dbg !69
  ret void, !dbg !15
}

define internal void @__nv_MAIN__F1L56_1_(i32* %__nv_MAIN__F1L56_1Arg0, i64* %__nv_MAIN__F1L56_1Arg1, i64* %__nv_MAIN__F1L56_1Arg2) #0 !dbg !70 {
L.entry:
  %__gtid___nv_MAIN__F1L56_1__576 = alloca i32, align 4
  %.i0000p_366 = alloca i32, align 4
  %i_364 = alloca i32, align 4
  %.du0004p_408 = alloca i32, align 4
  %.de0004p_409 = alloca i32, align 4
  %.di0004p_410 = alloca i32, align 4
  %.ds0004p_411 = alloca i32, align 4
  %.dl0004p_413 = alloca i32, align 4
  %.dl0004p.copy_570 = alloca i32, align 4
  %.de0004p.copy_571 = alloca i32, align 4
  %.ds0004p.copy_572 = alloca i32, align 4
  %.dX0004p_412 = alloca i32, align 4
  %.dY0004p_407 = alloca i32, align 4
  %.dY0005p_419 = alloca i32, align 4
  %j_365 = alloca i32, align 4
  %temp_363 = alloca float, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L56_1Arg0, metadata !73, metadata !DIExpression()), !dbg !74
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L56_1Arg1, metadata !75, metadata !DIExpression()), !dbg !74
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L56_1Arg2, metadata !76, metadata !DIExpression()), !dbg !74
  call void @llvm.dbg.value(metadata i32 1, metadata !77, metadata !DIExpression()), !dbg !74
  call void @llvm.dbg.value(metadata i32 0, metadata !78, metadata !DIExpression()), !dbg !74
  call void @llvm.dbg.value(metadata i32 1, metadata !79, metadata !DIExpression()), !dbg !74
  call void @llvm.dbg.value(metadata i32 0, metadata !80, metadata !DIExpression()), !dbg !74
  call void @llvm.dbg.value(metadata i32 1, metadata !81, metadata !DIExpression()), !dbg !74
  %0 = load i32, i32* %__nv_MAIN__F1L56_1Arg0, align 4, !dbg !82
  store i32 %0, i32* %__gtid___nv_MAIN__F1L56_1__576, align 4, !dbg !82
  br label %L.LB2_561

L.LB2_561:                                        ; preds = %L.entry
  br label %L.LB2_362

L.LB2_362:                                        ; preds = %L.LB2_561
  store i32 0, i32* %.i0000p_366, align 4, !dbg !83
  call void @llvm.dbg.declare(metadata i32* %i_364, metadata !84, metadata !DIExpression()), !dbg !82
  store i32 1, i32* %i_364, align 4, !dbg !83
  %1 = bitcast i64* %__nv_MAIN__F1L56_1Arg2 to i32**, !dbg !83
  %2 = load i32*, i32** %1, align 8, !dbg !83
  %3 = load i32, i32* %2, align 4, !dbg !83
  store i32 %3, i32* %.du0004p_408, align 4, !dbg !83
  %4 = bitcast i64* %__nv_MAIN__F1L56_1Arg2 to i32**, !dbg !83
  %5 = load i32*, i32** %4, align 8, !dbg !83
  %6 = load i32, i32* %5, align 4, !dbg !83
  store i32 %6, i32* %.de0004p_409, align 4, !dbg !83
  store i32 1, i32* %.di0004p_410, align 4, !dbg !83
  %7 = load i32, i32* %.di0004p_410, align 4, !dbg !83
  store i32 %7, i32* %.ds0004p_411, align 4, !dbg !83
  store i32 1, i32* %.dl0004p_413, align 4, !dbg !83
  %8 = load i32, i32* %.dl0004p_413, align 4, !dbg !83
  store i32 %8, i32* %.dl0004p.copy_570, align 4, !dbg !83
  %9 = load i32, i32* %.de0004p_409, align 4, !dbg !83
  store i32 %9, i32* %.de0004p.copy_571, align 4, !dbg !83
  %10 = load i32, i32* %.ds0004p_411, align 4, !dbg !83
  store i32 %10, i32* %.ds0004p.copy_572, align 4, !dbg !83
  %11 = load i32, i32* %__gtid___nv_MAIN__F1L56_1__576, align 4, !dbg !83
  %12 = bitcast i32* %.i0000p_366 to i64*, !dbg !83
  %13 = bitcast i32* %.dl0004p.copy_570 to i64*, !dbg !83
  %14 = bitcast i32* %.de0004p.copy_571 to i64*, !dbg !83
  %15 = bitcast i32* %.ds0004p.copy_572 to i64*, !dbg !83
  %16 = load i32, i32* %.ds0004p.copy_572, align 4, !dbg !83
  call void @__kmpc_for_static_init_4(i64* null, i32 %11, i32 34, i64* %12, i64* %13, i64* %14, i64* %15, i32 %16, i32 1), !dbg !83
  %17 = load i32, i32* %.dl0004p.copy_570, align 4, !dbg !83
  store i32 %17, i32* %.dl0004p_413, align 4, !dbg !83
  %18 = load i32, i32* %.de0004p.copy_571, align 4, !dbg !83
  store i32 %18, i32* %.de0004p_409, align 4, !dbg !83
  %19 = load i32, i32* %.ds0004p.copy_572, align 4, !dbg !83
  store i32 %19, i32* %.ds0004p_411, align 4, !dbg !83
  %20 = load i32, i32* %.dl0004p_413, align 4, !dbg !83
  store i32 %20, i32* %i_364, align 4, !dbg !83
  %21 = load i32, i32* %i_364, align 4, !dbg !83
  call void @llvm.dbg.value(metadata i32 %21, metadata !84, metadata !DIExpression()), !dbg !82
  store i32 %21, i32* %.dX0004p_412, align 4, !dbg !83
  %22 = load i32, i32* %.dX0004p_412, align 4, !dbg !83
  %23 = load i32, i32* %.du0004p_408, align 4, !dbg !83
  %24 = icmp sgt i32 %22, %23, !dbg !83
  br i1 %24, label %L.LB2_406, label %L.LB2_604, !dbg !83

L.LB2_604:                                        ; preds = %L.LB2_362
  %25 = load i32, i32* %.dX0004p_412, align 4, !dbg !83
  store i32 %25, i32* %i_364, align 4, !dbg !83
  %26 = load i32, i32* %.di0004p_410, align 4, !dbg !83
  %27 = load i32, i32* %.de0004p_409, align 4, !dbg !83
  %28 = load i32, i32* %.dX0004p_412, align 4, !dbg !83
  %29 = sub nsw i32 %27, %28, !dbg !83
  %30 = add nsw i32 %26, %29, !dbg !83
  %31 = load i32, i32* %.di0004p_410, align 4, !dbg !83
  %32 = sdiv i32 %30, %31, !dbg !83
  store i32 %32, i32* %.dY0004p_407, align 4, !dbg !83
  %33 = load i32, i32* %.dY0004p_407, align 4, !dbg !83
  %34 = icmp sle i32 %33, 0, !dbg !83
  br i1 %34, label %L.LB2_416, label %L.LB2_415, !dbg !83

L.LB2_415:                                        ; preds = %L.LB2_418, %L.LB2_604
  %35 = bitcast i64* %__nv_MAIN__F1L56_1Arg2 to i32**, !dbg !85
  %36 = load i32*, i32** %35, align 8, !dbg !85
  %37 = load i32, i32* %36, align 4, !dbg !85
  store i32 %37, i32* %.dY0005p_419, align 4, !dbg !85
  call void @llvm.dbg.declare(metadata i32* %j_365, metadata !86, metadata !DIExpression()), !dbg !82
  store i32 1, i32* %j_365, align 4, !dbg !85
  %38 = load i32, i32* %.dY0005p_419, align 4, !dbg !85
  %39 = icmp sle i32 %38, 0, !dbg !85
  br i1 %39, label %L.LB2_418, label %L.LB2_417, !dbg !85

L.LB2_417:                                        ; preds = %L.LB2_417, %L.LB2_415
  %40 = load i32, i32* %i_364, align 4, !dbg !87
  call void @llvm.dbg.value(metadata i32 %40, metadata !84, metadata !DIExpression()), !dbg !82
  %41 = sext i32 %40 to i64, !dbg !87
  %42 = load i32, i32* %j_365, align 4, !dbg !87
  call void @llvm.dbg.value(metadata i32 %42, metadata !86, metadata !DIExpression()), !dbg !82
  %43 = sext i32 %42 to i64, !dbg !87
  %44 = bitcast i64* %__nv_MAIN__F1L56_1Arg2 to i8*, !dbg !87
  %45 = getelementptr i8, i8* %44, i64 104, !dbg !87
  %46 = bitcast i8* %45 to i8**, !dbg !87
  %47 = load i8*, i8** %46, align 8, !dbg !87
  %48 = getelementptr i8, i8* %47, i64 160, !dbg !87
  %49 = bitcast i8* %48 to i64*, !dbg !87
  %50 = load i64, i64* %49, align 8, !dbg !87
  %51 = mul nsw i64 %43, %50, !dbg !87
  %52 = add nsw i64 %41, %51, !dbg !87
  %53 = bitcast i64* %__nv_MAIN__F1L56_1Arg2 to i8*, !dbg !87
  %54 = getelementptr i8, i8* %53, i64 104, !dbg !87
  %55 = bitcast i8* %54 to i8**, !dbg !87
  %56 = load i8*, i8** %55, align 8, !dbg !87
  %57 = getelementptr i8, i8* %56, i64 56, !dbg !87
  %58 = bitcast i8* %57 to i64*, !dbg !87
  %59 = load i64, i64* %58, align 8, !dbg !87
  %60 = add nsw i64 %52, %59, !dbg !87
  %61 = bitcast i64* %__nv_MAIN__F1L56_1Arg2 to i8*, !dbg !87
  %62 = getelementptr i8, i8* %61, i64 16, !dbg !87
  %63 = bitcast i8* %62 to i8***, !dbg !87
  %64 = load i8**, i8*** %63, align 8, !dbg !87
  %65 = load i8*, i8** %64, align 8, !dbg !87
  %66 = getelementptr i8, i8* %65, i64 -4, !dbg !87
  %67 = bitcast i8* %66 to float*, !dbg !87
  %68 = getelementptr float, float* %67, i64 %60, !dbg !87
  %69 = load float, float* %68, align 4, !dbg !87
  call void @llvm.dbg.declare(metadata float* %temp_363, metadata !88, metadata !DIExpression()), !dbg !82
  store float %69, float* %temp_363, align 4, !dbg !87
  %70 = load float, float* %temp_363, align 4, !dbg !89
  call void @llvm.dbg.value(metadata float %70, metadata !88, metadata !DIExpression()), !dbg !82
  %71 = load float, float* %temp_363, align 4, !dbg !89
  call void @llvm.dbg.value(metadata float %71, metadata !88, metadata !DIExpression()), !dbg !82
  %72 = fmul fast float %70, %71, !dbg !89
  %73 = bitcast i64* %__nv_MAIN__F1L56_1Arg2 to i8*, !dbg !89
  %74 = getelementptr i8, i8* %73, i64 96, !dbg !89
  %75 = bitcast i8* %74 to float**, !dbg !89
  %76 = load float*, float** %75, align 8, !dbg !89
  %77 = load float, float* %76, align 4, !dbg !89
  %78 = fadd fast float %72, %77, !dbg !89
  %79 = bitcast i64* %__nv_MAIN__F1L56_1Arg2 to i8*, !dbg !89
  %80 = getelementptr i8, i8* %79, i64 96, !dbg !89
  %81 = bitcast i8* %80 to float**, !dbg !89
  %82 = load float*, float** %81, align 8, !dbg !89
  store float %78, float* %82, align 4, !dbg !89
  %83 = load i32, i32* %j_365, align 4, !dbg !90
  call void @llvm.dbg.value(metadata i32 %83, metadata !86, metadata !DIExpression()), !dbg !82
  %84 = add nsw i32 %83, 1, !dbg !90
  store i32 %84, i32* %j_365, align 4, !dbg !90
  %85 = load i32, i32* %.dY0005p_419, align 4, !dbg !90
  %86 = sub nsw i32 %85, 1, !dbg !90
  store i32 %86, i32* %.dY0005p_419, align 4, !dbg !90
  %87 = load i32, i32* %.dY0005p_419, align 4, !dbg !90
  %88 = icmp sgt i32 %87, 0, !dbg !90
  br i1 %88, label %L.LB2_417, label %L.LB2_418, !dbg !90

L.LB2_418:                                        ; preds = %L.LB2_417, %L.LB2_415
  %89 = load i32, i32* %.di0004p_410, align 4, !dbg !82
  %90 = load i32, i32* %i_364, align 4, !dbg !82
  call void @llvm.dbg.value(metadata i32 %90, metadata !84, metadata !DIExpression()), !dbg !82
  %91 = add nsw i32 %89, %90, !dbg !82
  store i32 %91, i32* %i_364, align 4, !dbg !82
  %92 = load i32, i32* %.dY0004p_407, align 4, !dbg !82
  %93 = sub nsw i32 %92, 1, !dbg !82
  store i32 %93, i32* %.dY0004p_407, align 4, !dbg !82
  %94 = load i32, i32* %.dY0004p_407, align 4, !dbg !82
  %95 = icmp sgt i32 %94, 0, !dbg !82
  br i1 %95, label %L.LB2_415, label %L.LB2_416, !dbg !82

L.LB2_416:                                        ; preds = %L.LB2_418, %L.LB2_604
  br label %L.LB2_406

L.LB2_406:                                        ; preds = %L.LB2_416, %L.LB2_362
  %96 = load i32, i32* %__gtid___nv_MAIN__F1L56_1__576, align 4, !dbg !82
  call void @__kmpc_for_static_fini(i64* null, i32 %96), !dbg !82
  br label %L.LB2_367

L.LB2_367:                                        ; preds = %L.LB2_406
  ret void, !dbg !82
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
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB022-reductionmissing-var-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb022_reductionmissing_var_yes", scope: !2, file: !3, line: 14, type: !6, scopeLine: 14, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!6 = !DISubroutineType(cc: DW_CC_program, types: !7)
!7 = !{null}
!8 = !DILocalVariable(name: "omp_sched_static", scope: !5, file: !3, type: !9)
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 0, scope: !5)
!11 = !DILocalVariable(name: "omp_proc_bind_false", scope: !5, file: !3, type: !9)
!12 = !DILocalVariable(name: "omp_proc_bind_true", scope: !5, file: !3, type: !9)
!13 = !DILocalVariable(name: "omp_lock_hint_none", scope: !5, file: !3, type: !9)
!14 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !5, file: !3, type: !9)
!15 = !DILocation(line: 69, column: 1, scope: !5)
!16 = !DILocation(line: 14, column: 1, scope: !5)
!17 = !DILocalVariable(name: "u", scope: !5, file: !3, type: !18)
!18 = !DICompositeType(tag: DW_TAG_array_type, baseType: !19, size: 32, align: 32, elements: !20)
!19 = !DIBasicType(name: "real", size: 32, align: 32, encoding: DW_ATE_float)
!20 = !{!21, !21}
!21 = !DISubrange(count: 0, lowerBound: 1)
!22 = !DILocalVariable(scope: !5, file: !3, type: !23, flags: DIFlagArtificial)
!23 = !DICompositeType(tag: DW_TAG_array_type, baseType: !24, size: 1408, align: 64, elements: !25)
!24 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!25 = !{!26}
!26 = !DISubrange(count: 22, lowerBound: 1)
!27 = !DILocalVariable(name: "args", scope: !5, file: !3, type: !28)
!28 = !DICompositeType(tag: DW_TAG_array_type, baseType: !29, size: 640, align: 8, elements: !30)
!29 = !DIBasicType(name: "character", size: 640, align: 8, encoding: DW_ATE_signed)
!30 = !{!21}
!31 = !DILocalVariable(scope: !5, file: !3, type: !32, flags: DIFlagArtificial)
!32 = !DICompositeType(tag: DW_TAG_array_type, baseType: !24, size: 1024, align: 64, elements: !33)
!33 = !{!34}
!34 = !DISubrange(count: 16, lowerBound: 1)
!35 = !DILocalVariable(name: "len", scope: !5, file: !3, type: !9)
!36 = !DILocation(line: 23, column: 1, scope: !5)
!37 = !DILocalVariable(name: "getsum", scope: !5, file: !3, type: !19)
!38 = !DILocation(line: 24, column: 1, scope: !5)
!39 = !DILocation(line: 26, column: 1, scope: !5)
!40 = !DILocalVariable(name: "argcount", scope: !5, file: !3, type: !9)
!41 = !DILocation(line: 27, column: 1, scope: !5)
!42 = !DILocation(line: 28, column: 1, scope: !5)
!43 = !DILocalVariable(scope: !5, file: !3, type: !9, flags: DIFlagArtificial)
!44 = !DILocalVariable(scope: !5, file: !3, type: !24, flags: DIFlagArtificial)
!45 = !DILocation(line: 31, column: 1, scope: !5)
!46 = !DILocalVariable(name: "allocstatus", scope: !5, file: !3, type: !9)
!47 = !DILocation(line: 32, column: 1, scope: !5)
!48 = !DILocation(line: 33, column: 1, scope: !5)
!49 = !DILocation(line: 34, column: 1, scope: !5)
!50 = !DILocation(line: 37, column: 1, scope: !5)
!51 = !DILocalVariable(name: "ix", scope: !5, file: !3, type: !9)
!52 = !DILocation(line: 38, column: 1, scope: !5)
!53 = !DILocation(line: 39, column: 1, scope: !5)
!54 = !DILocation(line: 41, column: 1, scope: !5)
!55 = !DILocation(line: 42, column: 1, scope: !5)
!56 = !DILocalVariable(name: "rderr", scope: !5, file: !3, type: !9)
!57 = !DILocation(line: 43, column: 1, scope: !5)
!58 = !DILocation(line: 44, column: 1, scope: !5)
!59 = !DILocation(line: 48, column: 1, scope: !5)
!60 = !DILocation(line: 50, column: 1, scope: !5)
!61 = !DILocalVariable(name: "i", scope: !5, file: !3, type: !9)
!62 = !DILocation(line: 51, column: 1, scope: !5)
!63 = !DILocalVariable(name: "j", scope: !5, file: !3, type: !9)
!64 = !DILocation(line: 52, column: 1, scope: !5)
!65 = !DILocation(line: 53, column: 1, scope: !5)
!66 = !DILocation(line: 54, column: 1, scope: !5)
!67 = !DILocation(line: 56, column: 1, scope: !5)
!68 = !DILocation(line: 65, column: 1, scope: !5)
!69 = !DILocation(line: 68, column: 1, scope: !5)
!70 = distinct !DISubprogram(name: "__nv_MAIN__F1L56_1", scope: !2, file: !3, line: 56, type: !71, scopeLine: 56, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!71 = !DISubroutineType(types: !72)
!72 = !{null, !9, !24, !24}
!73 = !DILocalVariable(name: "__nv_MAIN__F1L56_1Arg0", arg: 1, scope: !70, file: !3, type: !9)
!74 = !DILocation(line: 0, scope: !70)
!75 = !DILocalVariable(name: "__nv_MAIN__F1L56_1Arg1", arg: 2, scope: !70, file: !3, type: !24)
!76 = !DILocalVariable(name: "__nv_MAIN__F1L56_1Arg2", arg: 3, scope: !70, file: !3, type: !24)
!77 = !DILocalVariable(name: "omp_sched_static", scope: !70, file: !3, type: !9)
!78 = !DILocalVariable(name: "omp_proc_bind_false", scope: !70, file: !3, type: !9)
!79 = !DILocalVariable(name: "omp_proc_bind_true", scope: !70, file: !3, type: !9)
!80 = !DILocalVariable(name: "omp_lock_hint_none", scope: !70, file: !3, type: !9)
!81 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !70, file: !3, type: !9)
!82 = !DILocation(line: 62, column: 1, scope: !70)
!83 = !DILocation(line: 57, column: 1, scope: !70)
!84 = !DILocalVariable(name: "i", scope: !70, file: !3, type: !9)
!85 = !DILocation(line: 58, column: 1, scope: !70)
!86 = !DILocalVariable(name: "j", scope: !70, file: !3, type: !9)
!87 = !DILocation(line: 59, column: 1, scope: !70)
!88 = !DILocalVariable(name: "temp", scope: !70, file: !3, type: !19)
!89 = !DILocation(line: 60, column: 1, scope: !70)
!90 = !DILocation(line: 61, column: 1, scope: !70)

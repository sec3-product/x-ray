; ModuleID = '/tmp/DRB038-truedepseconddimension-var-yes-7fac1a.ll'
source_filename = "/tmp/DRB038-truedepseconddimension-var-yes-7fac1a.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%astruct.dt86 = type <{ i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8* }>

@.C300_MAIN_ = internal constant i32 2
@.C311_MAIN_ = internal constant i32 27
@.C380_MAIN_ = internal constant i64 27
@.C362_MAIN_ = internal constant [29 x i8] c"Error, invalid integer value."
@.C361_MAIN_ = internal constant i32 39
@.C310_MAIN_ = internal constant i32 25
@.C357_MAIN_ = internal constant [5 x i8] c"(i10)"
@.C354_MAIN_ = internal constant i32 37
@.C372_MAIN_ = internal constant i64 4
@.C355_MAIN_ = internal constant [37 x i8] c"Allocation error, program terminated."
@.C313_MAIN_ = internal constant i32 28
@.C376_MAIN_ = internal constant i64 80
@.C375_MAIN_ = internal constant i64 14
@.C346_MAIN_ = internal constant [35 x i8] c"No command line arguments provided."
@.C344_MAIN_ = internal constant i32 6
@.C345_MAIN_ = internal constant [3 x i8] c"(a)"
@.C285_MAIN_ = internal constant i32 1
@.C309_MAIN_ = internal constant i32 14
@.C342_MAIN_ = internal constant [66 x i8] c"micro-benchmarks-fortran/DRB038-truedepseconddimension-var-yes.f95"
@.C312_MAIN_ = internal constant i32 23
@.C338_MAIN_ = internal constant i32 1000
@.C286_MAIN_ = internal constant i64 1
@.C284_MAIN_ = internal constant i64 0
@.C283_MAIN_ = internal constant i32 0
@.C285___nv_MAIN__F1L49_1 = internal constant i32 1
@.C300___nv_MAIN__F1L49_1 = internal constant i32 2
@.C283___nv_MAIN__F1L49_1 = internal constant i32 0

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__515 = alloca i32, align 4
  %.Z0983_363 = alloca float*, align 8
  %"b$sd2_379" = alloca [22 x i64], align 8
  %.Z0973_353 = alloca [80 x i8]*, align 8
  %"args$sd1_374" = alloca [16 x i64], align 8
  %len_339 = alloca i32, align 4
  %argcount_318 = alloca i32, align 4
  %z__io_348 = alloca i32, align 4
  %z_b_0_322 = alloca i64, align 8
  %z_b_1_323 = alloca i64, align 8
  %z_e_61_326 = alloca i64, align 8
  %z_b_2_324 = alloca i64, align 8
  %z_b_3_325 = alloca i64, align 8
  %allocstatus_319 = alloca i32, align 4
  %.dY0001_391 = alloca i32, align 4
  %ix_321 = alloca i32, align 4
  %rderr_320 = alloca i32, align 4
  %n_316 = alloca i32, align 4
  %m_317 = alloca i32, align 4
  %z_b_4_328 = alloca i64, align 8
  %z_b_5_329 = alloca i64, align 8
  %z_e_71_335 = alloca i64, align 8
  %z_b_7_331 = alloca i64, align 8
  %z_b_8_332 = alloca i64, align 8
  %z_e_74_336 = alloca i64, align 8
  %z_b_6_330 = alloca i64, align 8
  %z_b_9_333 = alloca i64, align 8
  %z_b_10_334 = alloca i64, align 8
  %.dY0002_396 = alloca i32, align 4
  %i_314 = alloca i32, align 4
  %.uplevelArgPack0001_484 = alloca %astruct.dt86, align 16
  call void @llvm.dbg.value(metadata i32 1, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 2, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 2, metadata !14, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !15, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !16, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 2, metadata !17, metadata !DIExpression()), !dbg !10
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !18
  store i32 %0, i32* %__gtid_MAIN__515, align 4, !dbg !18
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !19
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !19
  call void (i8*, ...) %2(i8* %1), !dbg !19
  call void @llvm.dbg.declare(metadata float** %.Z0983_363, metadata !20, metadata !DIExpression(DW_OP_deref)), !dbg !10
  %3 = bitcast float** %.Z0983_363 to i8**, !dbg !19
  store i8* null, i8** %3, align 8, !dbg !19
  call void @llvm.dbg.declare(metadata [22 x i64]* %"b$sd2_379", metadata !25, metadata !DIExpression()), !dbg !10
  %4 = bitcast [22 x i64]* %"b$sd2_379" to i64*, !dbg !19
  store i64 0, i64* %4, align 8, !dbg !19
  call void @llvm.dbg.declare(metadata [80 x i8]** %.Z0973_353, metadata !30, metadata !DIExpression(DW_OP_deref)), !dbg !10
  %5 = bitcast [80 x i8]** %.Z0973_353 to i8**, !dbg !19
  store i8* null, i8** %5, align 8, !dbg !19
  call void @llvm.dbg.declare(metadata [16 x i64]* %"args$sd1_374", metadata !34, metadata !DIExpression()), !dbg !10
  %6 = bitcast [16 x i64]* %"args$sd1_374" to i64*, !dbg !19
  store i64 0, i64* %6, align 8, !dbg !19
  br label %L.LB1_418

L.LB1_418:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %len_339, metadata !38, metadata !DIExpression()), !dbg !10
  store i32 1000, i32* %len_339, align 4, !dbg !39
  %7 = call i32 (...) @f90_cmd_arg_cnt(), !dbg !40
  call void @llvm.dbg.declare(metadata i32* %argcount_318, metadata !41, metadata !DIExpression()), !dbg !10
  store i32 %7, i32* %argcount_318, align 4, !dbg !40
  %8 = load i32, i32* %argcount_318, align 4, !dbg !42
  call void @llvm.dbg.value(metadata i32 %8, metadata !41, metadata !DIExpression()), !dbg !10
  %9 = icmp ne i32 %8, 0, !dbg !42
  br i1 %9, label %L.LB1_385, label %L.LB1_533, !dbg !42

L.LB1_533:                                        ; preds = %L.LB1_418
  call void (...) @_mp_bcs_nest(), !dbg !43
  %10 = bitcast i32* @.C312_MAIN_ to i8*, !dbg !43
  %11 = bitcast [66 x i8]* @.C342_MAIN_ to i8*, !dbg !43
  %12 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !43
  call void (i8*, i8*, i64, ...) %12(i8* %10, i8* %11, i64 66), !dbg !43
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
  br label %L.LB1_385

L.LB1_385:                                        ; preds = %L.LB1_533, %L.LB1_418
  call void @llvm.dbg.declare(metadata i64* %z_b_0_322, metadata !45, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_0_322, align 8, !dbg !46
  %30 = load i32, i32* %argcount_318, align 4, !dbg !46
  call void @llvm.dbg.value(metadata i32 %30, metadata !41, metadata !DIExpression()), !dbg !10
  %31 = sext i32 %30 to i64, !dbg !46
  call void @llvm.dbg.declare(metadata i64* %z_b_1_323, metadata !45, metadata !DIExpression()), !dbg !10
  store i64 %31, i64* %z_b_1_323, align 8, !dbg !46
  %32 = load i64, i64* %z_b_1_323, align 8, !dbg !46
  call void @llvm.dbg.value(metadata i64 %32, metadata !45, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_61_326, metadata !45, metadata !DIExpression()), !dbg !10
  store i64 %32, i64* %z_e_61_326, align 8, !dbg !46
  %33 = bitcast [16 x i64]* %"args$sd1_374" to i8*, !dbg !46
  %34 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !46
  %35 = bitcast i64* @.C375_MAIN_ to i8*, !dbg !46
  %36 = bitcast i64* @.C376_MAIN_ to i8*, !dbg !46
  %37 = bitcast i64* %z_b_0_322 to i8*, !dbg !46
  %38 = bitcast i64* %z_b_1_323 to i8*, !dbg !46
  %39 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !46
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %39(i8* %33, i8* %34, i8* %35, i8* %36, i8* %37, i8* %38), !dbg !46
  %40 = bitcast [16 x i64]* %"args$sd1_374" to i8*, !dbg !46
  %41 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !46
  call void (i8*, i32, ...) %41(i8* %40, i32 14), !dbg !46
  %42 = load i64, i64* %z_b_1_323, align 8, !dbg !46
  call void @llvm.dbg.value(metadata i64 %42, metadata !45, metadata !DIExpression()), !dbg !10
  %43 = load i64, i64* %z_b_0_322, align 8, !dbg !46
  call void @llvm.dbg.value(metadata i64 %43, metadata !45, metadata !DIExpression()), !dbg !10
  %44 = sub nsw i64 %43, 1, !dbg !46
  %45 = sub nsw i64 %42, %44, !dbg !46
  call void @llvm.dbg.declare(metadata i64* %z_b_2_324, metadata !45, metadata !DIExpression()), !dbg !10
  store i64 %45, i64* %z_b_2_324, align 8, !dbg !46
  %46 = load i64, i64* %z_b_0_322, align 8, !dbg !46
  call void @llvm.dbg.value(metadata i64 %46, metadata !45, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_b_3_325, metadata !45, metadata !DIExpression()), !dbg !10
  store i64 %46, i64* %z_b_3_325, align 8, !dbg !46
  %47 = bitcast i64* %z_b_2_324 to i8*, !dbg !46
  %48 = bitcast i64* @.C375_MAIN_ to i8*, !dbg !46
  %49 = bitcast i64* @.C376_MAIN_ to i8*, !dbg !46
  call void @llvm.dbg.declare(metadata i32* %allocstatus_319, metadata !47, metadata !DIExpression()), !dbg !10
  %50 = bitcast i32* %allocstatus_319 to i8*, !dbg !46
  %51 = bitcast [80 x i8]** %.Z0973_353 to i8*, !dbg !46
  %52 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !46
  %53 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !46
  %54 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !46
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %54(i8* %47, i8* %48, i8* %49, i8* %50, i8* %51, i8* null, i8* %52, i8* %53, i8* null, i64 0), !dbg !46
  %55 = load i32, i32* %allocstatus_319, align 4, !dbg !48
  call void @llvm.dbg.value(metadata i32 %55, metadata !47, metadata !DIExpression()), !dbg !10
  %56 = icmp sle i32 %55, 0, !dbg !48
  br i1 %56, label %L.LB1_388, label %L.LB1_534, !dbg !48

L.LB1_534:                                        ; preds = %L.LB1_385
  call void (...) @_mp_bcs_nest(), !dbg !49
  %57 = bitcast i32* @.C313_MAIN_ to i8*, !dbg !49
  %58 = bitcast [66 x i8]* @.C342_MAIN_ to i8*, !dbg !49
  %59 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !49
  call void (i8*, i8*, i64, ...) %59(i8* %57, i8* %58, i64 66), !dbg !49
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
  br label %L.LB1_388

L.LB1_388:                                        ; preds = %L.LB1_534, %L.LB1_385
  %79 = load i32, i32* %argcount_318, align 4, !dbg !51
  call void @llvm.dbg.value(metadata i32 %79, metadata !41, metadata !DIExpression()), !dbg !10
  store i32 %79, i32* %.dY0001_391, align 4, !dbg !51
  call void @llvm.dbg.declare(metadata i32* %ix_321, metadata !52, metadata !DIExpression()), !dbg !10
  store i32 1, i32* %ix_321, align 4, !dbg !51
  %80 = load i32, i32* %.dY0001_391, align 4, !dbg !51
  %81 = icmp sle i32 %80, 0, !dbg !51
  br i1 %81, label %L.LB1_390, label %L.LB1_389, !dbg !51

L.LB1_389:                                        ; preds = %L.LB1_389, %L.LB1_388
  %82 = bitcast i32* %ix_321 to i8*, !dbg !53
  %83 = load [80 x i8]*, [80 x i8]** %.Z0973_353, align 8, !dbg !53
  call void @llvm.dbg.value(metadata [80 x i8]* %83, metadata !30, metadata !DIExpression()), !dbg !10
  %84 = bitcast [80 x i8]* %83 to i8*, !dbg !53
  %85 = getelementptr i8, i8* %84, i64 -80, !dbg !53
  %86 = load i32, i32* %ix_321, align 4, !dbg !53
  call void @llvm.dbg.value(metadata i32 %86, metadata !52, metadata !DIExpression()), !dbg !10
  %87 = sext i32 %86 to i64, !dbg !53
  %88 = bitcast [16 x i64]* %"args$sd1_374" to i8*, !dbg !53
  %89 = getelementptr i8, i8* %88, i64 56, !dbg !53
  %90 = bitcast i8* %89 to i64*, !dbg !53
  %91 = load i64, i64* %90, align 8, !dbg !53
  %92 = add nsw i64 %87, %91, !dbg !53
  %93 = mul nsw i64 %92, 80, !dbg !53
  %94 = getelementptr i8, i8* %85, i64 %93, !dbg !53
  %95 = bitcast i64* @.C372_MAIN_ to i8*, !dbg !53
  %96 = bitcast void (...)* @f90_get_cmd_arga to void (i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !53
  call void (i8*, i8*, i8*, i8*, i8*, i64, ...) %96(i8* %82, i8* %94, i8* null, i8* null, i8* %95, i64 80), !dbg !53
  %97 = load i32, i32* %ix_321, align 4, !dbg !54
  call void @llvm.dbg.value(metadata i32 %97, metadata !52, metadata !DIExpression()), !dbg !10
  %98 = add nsw i32 %97, 1, !dbg !54
  store i32 %98, i32* %ix_321, align 4, !dbg !54
  %99 = load i32, i32* %.dY0001_391, align 4, !dbg !54
  %100 = sub nsw i32 %99, 1, !dbg !54
  store i32 %100, i32* %.dY0001_391, align 4, !dbg !54
  %101 = load i32, i32* %.dY0001_391, align 4, !dbg !54
  %102 = icmp sgt i32 %101, 0, !dbg !54
  br i1 %102, label %L.LB1_389, label %L.LB1_390, !dbg !54

L.LB1_390:                                        ; preds = %L.LB1_389, %L.LB1_388
  %103 = load i32, i32* %argcount_318, align 4, !dbg !55
  call void @llvm.dbg.value(metadata i32 %103, metadata !41, metadata !DIExpression()), !dbg !10
  %104 = icmp sle i32 %103, 0, !dbg !55
  br i1 %104, label %L.LB1_392, label %L.LB1_535, !dbg !55

L.LB1_535:                                        ; preds = %L.LB1_390
  call void (...) @_mp_bcs_nest(), !dbg !56
  %105 = bitcast i32* @.C354_MAIN_ to i8*, !dbg !56
  %106 = bitcast [66 x i8]* @.C342_MAIN_ to i8*, !dbg !56
  %107 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !56
  call void (i8*, i8*, i64, ...) %107(i8* %105, i8* %106, i64 66), !dbg !56
  %108 = bitcast i32* @.C309_MAIN_ to i8*, !dbg !56
  %109 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !56
  %110 = bitcast [5 x i8]* @.C357_MAIN_ to i8*, !dbg !56
  %111 = bitcast i32 (...)* @f90io_encode_fmta to i32 (i8*, i8*, i8*, i64, ...)*, !dbg !56
  %112 = call i32 (i8*, i8*, i8*, i64, ...) %111(i8* %108, i8* %109, i8* %110, i64 5), !dbg !56
  store i32 %112, i32* %z__io_348, align 4, !dbg !56
  %113 = load [80 x i8]*, [80 x i8]** %.Z0973_353, align 8, !dbg !56
  call void @llvm.dbg.value(metadata [80 x i8]* %113, metadata !30, metadata !DIExpression()), !dbg !10
  %114 = bitcast [80 x i8]* %113 to i8*, !dbg !56
  %115 = bitcast [16 x i64]* %"args$sd1_374" to i8*, !dbg !56
  %116 = getelementptr i8, i8* %115, i64 56, !dbg !56
  %117 = bitcast i8* %116 to i64*, !dbg !56
  %118 = load i64, i64* %117, align 8, !dbg !56
  %119 = mul nsw i64 %118, 80, !dbg !56
  %120 = getelementptr i8, i8* %114, i64 %119, !dbg !56
  %121 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !56
  %122 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !56
  call void @llvm.dbg.declare(metadata i32* %rderr_320, metadata !57, metadata !DIExpression()), !dbg !10
  %123 = bitcast i32* %rderr_320 to i8*, !dbg !56
  %124 = bitcast i32 (...)* @f90io_fmtr_intern_inita to i32 (i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !56
  %125 = call i32 (i8*, i8*, i8*, i8*, i8*, i64, ...) %124(i8* %120, i8* %121, i8* %122, i8* %123, i8* null, i64 80), !dbg !56
  store i32 %125, i32* %z__io_348, align 4, !dbg !56
  %126 = bitcast i32* @.C310_MAIN_ to i8*, !dbg !56
  %127 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !56
  %128 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !56
  %129 = bitcast i32* %len_339 to i8*, !dbg !56
  %130 = bitcast i32 (...)* @f90io_fmt_reada to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !56
  %131 = call i32 (i8*, i8*, i8*, i8*, ...) %130(i8* %126, i8* %127, i8* %128, i8* %129), !dbg !56
  store i32 %131, i32* %z__io_348, align 4, !dbg !56
  %132 = call i32 (...) @f90io_fmtr_end(), !dbg !56
  store i32 %132, i32* %z__io_348, align 4, !dbg !56
  call void (...) @_mp_ecs_nest(), !dbg !56
  %133 = load i32, i32* %rderr_320, align 4, !dbg !58
  call void @llvm.dbg.value(metadata i32 %133, metadata !57, metadata !DIExpression()), !dbg !10
  %134 = icmp eq i32 %133, 0, !dbg !58
  br i1 %134, label %L.LB1_393, label %L.LB1_536, !dbg !58

L.LB1_536:                                        ; preds = %L.LB1_535
  call void (...) @_mp_bcs_nest(), !dbg !59
  %135 = bitcast i32* @.C361_MAIN_ to i8*, !dbg !59
  %136 = bitcast [66 x i8]* @.C342_MAIN_ to i8*, !dbg !59
  %137 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !59
  call void (i8*, i8*, i64, ...) %137(i8* %135, i8* %136, i64 66), !dbg !59
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
  br label %L.LB1_393

L.LB1_393:                                        ; preds = %L.LB1_536, %L.LB1_535
  br label %L.LB1_392

L.LB1_392:                                        ; preds = %L.LB1_393, %L.LB1_390
  %155 = load i32, i32* %len_339, align 4, !dbg !60
  call void @llvm.dbg.value(metadata i32 %155, metadata !38, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i32* %n_316, metadata !61, metadata !DIExpression()), !dbg !10
  store i32 %155, i32* %n_316, align 4, !dbg !60
  %156 = load i32, i32* %len_339, align 4, !dbg !62
  call void @llvm.dbg.value(metadata i32 %156, metadata !38, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i32* %m_317, metadata !63, metadata !DIExpression()), !dbg !10
  store i32 %156, i32* %m_317, align 4, !dbg !62
  call void @llvm.dbg.declare(metadata i64* %z_b_4_328, metadata !45, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_4_328, align 8, !dbg !64
  %157 = load i32, i32* %len_339, align 4, !dbg !64
  call void @llvm.dbg.value(metadata i32 %157, metadata !38, metadata !DIExpression()), !dbg !10
  %158 = sext i32 %157 to i64, !dbg !64
  call void @llvm.dbg.declare(metadata i64* %z_b_5_329, metadata !45, metadata !DIExpression()), !dbg !10
  store i64 %158, i64* %z_b_5_329, align 8, !dbg !64
  %159 = load i64, i64* %z_b_5_329, align 8, !dbg !64
  call void @llvm.dbg.value(metadata i64 %159, metadata !45, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_71_335, metadata !45, metadata !DIExpression()), !dbg !10
  store i64 %159, i64* %z_e_71_335, align 8, !dbg !64
  call void @llvm.dbg.declare(metadata i64* %z_b_7_331, metadata !45, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_7_331, align 8, !dbg !64
  %160 = load i32, i32* %len_339, align 4, !dbg !64
  call void @llvm.dbg.value(metadata i32 %160, metadata !38, metadata !DIExpression()), !dbg !10
  %161 = sext i32 %160 to i64, !dbg !64
  call void @llvm.dbg.declare(metadata i64* %z_b_8_332, metadata !45, metadata !DIExpression()), !dbg !10
  store i64 %161, i64* %z_b_8_332, align 8, !dbg !64
  %162 = load i64, i64* %z_b_8_332, align 8, !dbg !64
  call void @llvm.dbg.value(metadata i64 %162, metadata !45, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_74_336, metadata !45, metadata !DIExpression()), !dbg !10
  store i64 %162, i64* %z_e_74_336, align 8, !dbg !64
  %163 = bitcast [22 x i64]* %"b$sd2_379" to i8*, !dbg !64
  %164 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !64
  %165 = bitcast i64* @.C380_MAIN_ to i8*, !dbg !64
  %166 = bitcast i64* @.C372_MAIN_ to i8*, !dbg !64
  %167 = bitcast i64* %z_b_4_328 to i8*, !dbg !64
  %168 = bitcast i64* %z_b_5_329 to i8*, !dbg !64
  %169 = bitcast i64* %z_b_7_331 to i8*, !dbg !64
  %170 = bitcast i64* %z_b_8_332 to i8*, !dbg !64
  %171 = bitcast void (...)* @f90_template2_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !64
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, ...) %171(i8* %163, i8* %164, i8* %165, i8* %166, i8* %167, i8* %168, i8* %169, i8* %170), !dbg !64
  %172 = bitcast [22 x i64]* %"b$sd2_379" to i8*, !dbg !64
  %173 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !64
  call void (i8*, i32, ...) %173(i8* %172, i32 27), !dbg !64
  %174 = load i64, i64* %z_b_5_329, align 8, !dbg !64
  call void @llvm.dbg.value(metadata i64 %174, metadata !45, metadata !DIExpression()), !dbg !10
  %175 = load i64, i64* %z_b_4_328, align 8, !dbg !64
  call void @llvm.dbg.value(metadata i64 %175, metadata !45, metadata !DIExpression()), !dbg !10
  %176 = sub nsw i64 %175, 1, !dbg !64
  %177 = sub nsw i64 %174, %176, !dbg !64
  call void @llvm.dbg.declare(metadata i64* %z_b_6_330, metadata !45, metadata !DIExpression()), !dbg !10
  store i64 %177, i64* %z_b_6_330, align 8, !dbg !64
  %178 = load i64, i64* %z_b_5_329, align 8, !dbg !64
  call void @llvm.dbg.value(metadata i64 %178, metadata !45, metadata !DIExpression()), !dbg !10
  %179 = load i64, i64* %z_b_4_328, align 8, !dbg !64
  call void @llvm.dbg.value(metadata i64 %179, metadata !45, metadata !DIExpression()), !dbg !10
  %180 = sub nsw i64 %179, 1, !dbg !64
  %181 = sub nsw i64 %178, %180, !dbg !64
  %182 = load i64, i64* %z_b_8_332, align 8, !dbg !64
  call void @llvm.dbg.value(metadata i64 %182, metadata !45, metadata !DIExpression()), !dbg !10
  %183 = load i64, i64* %z_b_7_331, align 8, !dbg !64
  call void @llvm.dbg.value(metadata i64 %183, metadata !45, metadata !DIExpression()), !dbg !10
  %184 = sub nsw i64 %183, 1, !dbg !64
  %185 = sub nsw i64 %182, %184, !dbg !64
  %186 = mul nsw i64 %181, %185, !dbg !64
  call void @llvm.dbg.declare(metadata i64* %z_b_9_333, metadata !45, metadata !DIExpression()), !dbg !10
  store i64 %186, i64* %z_b_9_333, align 8, !dbg !64
  %187 = load i64, i64* %z_b_4_328, align 8, !dbg !64
  call void @llvm.dbg.value(metadata i64 %187, metadata !45, metadata !DIExpression()), !dbg !10
  %188 = load i64, i64* %z_b_5_329, align 8, !dbg !64
  call void @llvm.dbg.value(metadata i64 %188, metadata !45, metadata !DIExpression()), !dbg !10
  %189 = load i64, i64* %z_b_4_328, align 8, !dbg !64
  call void @llvm.dbg.value(metadata i64 %189, metadata !45, metadata !DIExpression()), !dbg !10
  %190 = sub nsw i64 %189, 1, !dbg !64
  %191 = sub nsw i64 %188, %190, !dbg !64
  %192 = load i64, i64* %z_b_7_331, align 8, !dbg !64
  call void @llvm.dbg.value(metadata i64 %192, metadata !45, metadata !DIExpression()), !dbg !10
  %193 = mul nsw i64 %191, %192, !dbg !64
  %194 = add nsw i64 %187, %193, !dbg !64
  call void @llvm.dbg.declare(metadata i64* %z_b_10_334, metadata !45, metadata !DIExpression()), !dbg !10
  store i64 %194, i64* %z_b_10_334, align 8, !dbg !64
  %195 = bitcast i64* %z_b_9_333 to i8*, !dbg !64
  %196 = bitcast i64* @.C380_MAIN_ to i8*, !dbg !64
  %197 = bitcast i64* @.C372_MAIN_ to i8*, !dbg !64
  %198 = bitcast float** %.Z0983_363 to i8*, !dbg !64
  %199 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !64
  %200 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !64
  %201 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !64
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %201(i8* %195, i8* %196, i8* %197, i8* null, i8* %198, i8* null, i8* %199, i8* %200, i8* null, i64 0), !dbg !64
  %202 = load i32, i32* %n_316, align 4, !dbg !65
  call void @llvm.dbg.value(metadata i32 %202, metadata !61, metadata !DIExpression()), !dbg !10
  store i32 %202, i32* %.dY0002_396, align 4, !dbg !65
  call void @llvm.dbg.declare(metadata i32* %i_314, metadata !66, metadata !DIExpression()), !dbg !10
  store i32 1, i32* %i_314, align 4, !dbg !65
  %203 = load i32, i32* %.dY0002_396, align 4, !dbg !65
  %204 = icmp sle i32 %203, 0, !dbg !65
  br i1 %204, label %L.LB1_395, label %L.LB1_394, !dbg !65

L.LB1_394:                                        ; preds = %L.LB1_513, %L.LB1_392
  %205 = bitcast i32* %m_317 to i8*, !dbg !67
  %206 = bitcast %astruct.dt86* %.uplevelArgPack0001_484 to i8**, !dbg !67
  store i8* %205, i8** %206, align 8, !dbg !67
  %207 = bitcast float** %.Z0983_363 to i8*, !dbg !67
  %208 = bitcast %astruct.dt86* %.uplevelArgPack0001_484 to i8*, !dbg !67
  %209 = getelementptr i8, i8* %208, i64 8, !dbg !67
  %210 = bitcast i8* %209 to i8**, !dbg !67
  store i8* %207, i8** %210, align 8, !dbg !67
  %211 = bitcast float** %.Z0983_363 to i8*, !dbg !67
  %212 = bitcast %astruct.dt86* %.uplevelArgPack0001_484 to i8*, !dbg !67
  %213 = getelementptr i8, i8* %212, i64 16, !dbg !67
  %214 = bitcast i8* %213 to i8**, !dbg !67
  store i8* %211, i8** %214, align 8, !dbg !67
  %215 = bitcast i64* %z_b_4_328 to i8*, !dbg !67
  %216 = bitcast %astruct.dt86* %.uplevelArgPack0001_484 to i8*, !dbg !67
  %217 = getelementptr i8, i8* %216, i64 24, !dbg !67
  %218 = bitcast i8* %217 to i8**, !dbg !67
  store i8* %215, i8** %218, align 8, !dbg !67
  %219 = bitcast i64* %z_b_5_329 to i8*, !dbg !67
  %220 = bitcast %astruct.dt86* %.uplevelArgPack0001_484 to i8*, !dbg !67
  %221 = getelementptr i8, i8* %220, i64 32, !dbg !67
  %222 = bitcast i8* %221 to i8**, !dbg !67
  store i8* %219, i8** %222, align 8, !dbg !67
  %223 = bitcast i64* %z_e_71_335 to i8*, !dbg !67
  %224 = bitcast %astruct.dt86* %.uplevelArgPack0001_484 to i8*, !dbg !67
  %225 = getelementptr i8, i8* %224, i64 40, !dbg !67
  %226 = bitcast i8* %225 to i8**, !dbg !67
  store i8* %223, i8** %226, align 8, !dbg !67
  %227 = bitcast i64* %z_b_7_331 to i8*, !dbg !67
  %228 = bitcast %astruct.dt86* %.uplevelArgPack0001_484 to i8*, !dbg !67
  %229 = getelementptr i8, i8* %228, i64 48, !dbg !67
  %230 = bitcast i8* %229 to i8**, !dbg !67
  store i8* %227, i8** %230, align 8, !dbg !67
  %231 = bitcast i64* %z_b_8_332 to i8*, !dbg !67
  %232 = bitcast %astruct.dt86* %.uplevelArgPack0001_484 to i8*, !dbg !67
  %233 = getelementptr i8, i8* %232, i64 56, !dbg !67
  %234 = bitcast i8* %233 to i8**, !dbg !67
  store i8* %231, i8** %234, align 8, !dbg !67
  %235 = bitcast i64* %z_b_6_330 to i8*, !dbg !67
  %236 = bitcast %astruct.dt86* %.uplevelArgPack0001_484 to i8*, !dbg !67
  %237 = getelementptr i8, i8* %236, i64 64, !dbg !67
  %238 = bitcast i8* %237 to i8**, !dbg !67
  store i8* %235, i8** %238, align 8, !dbg !67
  %239 = bitcast i64* %z_e_74_336 to i8*, !dbg !67
  %240 = bitcast %astruct.dt86* %.uplevelArgPack0001_484 to i8*, !dbg !67
  %241 = getelementptr i8, i8* %240, i64 72, !dbg !67
  %242 = bitcast i8* %241 to i8**, !dbg !67
  store i8* %239, i8** %242, align 8, !dbg !67
  %243 = bitcast i64* %z_b_9_333 to i8*, !dbg !67
  %244 = bitcast %astruct.dt86* %.uplevelArgPack0001_484 to i8*, !dbg !67
  %245 = getelementptr i8, i8* %244, i64 80, !dbg !67
  %246 = bitcast i8* %245 to i8**, !dbg !67
  store i8* %243, i8** %246, align 8, !dbg !67
  %247 = bitcast i64* %z_b_10_334 to i8*, !dbg !67
  %248 = bitcast %astruct.dt86* %.uplevelArgPack0001_484 to i8*, !dbg !67
  %249 = getelementptr i8, i8* %248, i64 88, !dbg !67
  %250 = bitcast i8* %249 to i8**, !dbg !67
  store i8* %247, i8** %250, align 8, !dbg !67
  %251 = bitcast i32* %i_314 to i8*, !dbg !67
  %252 = bitcast %astruct.dt86* %.uplevelArgPack0001_484 to i8*, !dbg !67
  %253 = getelementptr i8, i8* %252, i64 96, !dbg !67
  %254 = bitcast i8* %253 to i8**, !dbg !67
  store i8* %251, i8** %254, align 8, !dbg !67
  %255 = bitcast [22 x i64]* %"b$sd2_379" to i8*, !dbg !67
  %256 = bitcast %astruct.dt86* %.uplevelArgPack0001_484 to i8*, !dbg !67
  %257 = getelementptr i8, i8* %256, i64 104, !dbg !67
  %258 = bitcast i8* %257 to i8**, !dbg !67
  store i8* %255, i8** %258, align 8, !dbg !67
  br label %L.LB1_513, !dbg !67

L.LB1_513:                                        ; preds = %L.LB1_394
  %259 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L49_1_ to i64*, !dbg !67
  %260 = bitcast %astruct.dt86* %.uplevelArgPack0001_484 to i64*, !dbg !67
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %259, i64* %260), !dbg !67
  %261 = load i32, i32* %i_314, align 4, !dbg !68
  call void @llvm.dbg.value(metadata i32 %261, metadata !66, metadata !DIExpression()), !dbg !10
  %262 = add nsw i32 %261, 1, !dbg !68
  store i32 %262, i32* %i_314, align 4, !dbg !68
  %263 = load i32, i32* %.dY0002_396, align 4, !dbg !68
  %264 = sub nsw i32 %263, 1, !dbg !68
  store i32 %264, i32* %.dY0002_396, align 4, !dbg !68
  %265 = load i32, i32* %.dY0002_396, align 4, !dbg !68
  %266 = icmp sgt i32 %265, 0, !dbg !68
  br i1 %266, label %L.LB1_394, label %L.LB1_395, !dbg !68

L.LB1_395:                                        ; preds = %L.LB1_513, %L.LB1_392
  %267 = load [80 x i8]*, [80 x i8]** %.Z0973_353, align 8, !dbg !69
  call void @llvm.dbg.value(metadata [80 x i8]* %267, metadata !30, metadata !DIExpression()), !dbg !10
  %268 = bitcast [80 x i8]* %267 to i8*, !dbg !69
  %269 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !69
  %270 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, i64, ...)*, !dbg !69
  call void (i8*, i8*, i8*, i8*, i64, i64, ...) %270(i8* null, i8* %268, i8* %269, i8* null, i64 80, i64 0), !dbg !69
  %271 = bitcast [80 x i8]** %.Z0973_353 to i8**, !dbg !69
  store i8* null, i8** %271, align 8, !dbg !69
  %272 = bitcast [16 x i64]* %"args$sd1_374" to i64*, !dbg !69
  store i64 0, i64* %272, align 8, !dbg !69
  %273 = load float*, float** %.Z0983_363, align 8, !dbg !69
  call void @llvm.dbg.value(metadata float* %273, metadata !20, metadata !DIExpression()), !dbg !10
  %274 = bitcast float* %273 to i8*, !dbg !69
  %275 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !69
  %276 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !69
  call void (i8*, i8*, i8*, i8*, i64, ...) %276(i8* null, i8* %274, i8* %275, i8* null, i64 0), !dbg !69
  %277 = bitcast float** %.Z0983_363 to i8**, !dbg !69
  store i8* null, i8** %277, align 8, !dbg !69
  %278 = bitcast [22 x i64]* %"b$sd2_379" to i64*, !dbg !69
  store i64 0, i64* %278, align 8, !dbg !69
  ret void, !dbg !18
}

define internal void @__nv_MAIN__F1L49_1_(i32* %__nv_MAIN__F1L49_1Arg0, i64* %__nv_MAIN__F1L49_1Arg1, i64* %__nv_MAIN__F1L49_1Arg2) #0 !dbg !70 {
L.entry:
  %__gtid___nv_MAIN__F1L49_1__555 = alloca i32, align 4
  %.i0000p_368 = alloca i32, align 4
  %j_367 = alloca i32, align 4
  %.du0003p_400 = alloca i32, align 4
  %.de0003p_401 = alloca i32, align 4
  %.di0003p_402 = alloca i32, align 4
  %.ds0003p_403 = alloca i32, align 4
  %.dl0003p_405 = alloca i32, align 4
  %.dl0003p.copy_549 = alloca i32, align 4
  %.de0003p.copy_550 = alloca i32, align 4
  %.ds0003p.copy_551 = alloca i32, align 4
  %.dX0003p_404 = alloca i32, align 4
  %.dY0003p_399 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L49_1Arg0, metadata !73, metadata !DIExpression()), !dbg !74
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L49_1Arg1, metadata !75, metadata !DIExpression()), !dbg !74
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L49_1Arg2, metadata !76, metadata !DIExpression()), !dbg !74
  call void @llvm.dbg.value(metadata i32 1, metadata !77, metadata !DIExpression()), !dbg !74
  call void @llvm.dbg.value(metadata i32 2, metadata !78, metadata !DIExpression()), !dbg !74
  call void @llvm.dbg.value(metadata i32 0, metadata !79, metadata !DIExpression()), !dbg !74
  call void @llvm.dbg.value(metadata i32 1, metadata !80, metadata !DIExpression()), !dbg !74
  call void @llvm.dbg.value(metadata i32 2, metadata !81, metadata !DIExpression()), !dbg !74
  call void @llvm.dbg.value(metadata i32 0, metadata !82, metadata !DIExpression()), !dbg !74
  call void @llvm.dbg.value(metadata i32 1, metadata !83, metadata !DIExpression()), !dbg !74
  call void @llvm.dbg.value(metadata i32 2, metadata !84, metadata !DIExpression()), !dbg !74
  %0 = load i32, i32* %__nv_MAIN__F1L49_1Arg0, align 4, !dbg !85
  store i32 %0, i32* %__gtid___nv_MAIN__F1L49_1__555, align 4, !dbg !85
  br label %L.LB2_540

L.LB2_540:                                        ; preds = %L.entry
  br label %L.LB2_366

L.LB2_366:                                        ; preds = %L.LB2_540
  store i32 0, i32* %.i0000p_368, align 4, !dbg !86
  call void @llvm.dbg.declare(metadata i32* %j_367, metadata !87, metadata !DIExpression()), !dbg !85
  store i32 2, i32* %j_367, align 4, !dbg !86
  %1 = bitcast i64* %__nv_MAIN__F1L49_1Arg2 to i32**, !dbg !86
  %2 = load i32*, i32** %1, align 8, !dbg !86
  %3 = load i32, i32* %2, align 4, !dbg !86
  store i32 %3, i32* %.du0003p_400, align 4, !dbg !86
  %4 = bitcast i64* %__nv_MAIN__F1L49_1Arg2 to i32**, !dbg !86
  %5 = load i32*, i32** %4, align 8, !dbg !86
  %6 = load i32, i32* %5, align 4, !dbg !86
  store i32 %6, i32* %.de0003p_401, align 4, !dbg !86
  store i32 1, i32* %.di0003p_402, align 4, !dbg !86
  %7 = load i32, i32* %.di0003p_402, align 4, !dbg !86
  store i32 %7, i32* %.ds0003p_403, align 4, !dbg !86
  store i32 2, i32* %.dl0003p_405, align 4, !dbg !86
  %8 = load i32, i32* %.dl0003p_405, align 4, !dbg !86
  store i32 %8, i32* %.dl0003p.copy_549, align 4, !dbg !86
  %9 = load i32, i32* %.de0003p_401, align 4, !dbg !86
  store i32 %9, i32* %.de0003p.copy_550, align 4, !dbg !86
  %10 = load i32, i32* %.ds0003p_403, align 4, !dbg !86
  store i32 %10, i32* %.ds0003p.copy_551, align 4, !dbg !86
  %11 = load i32, i32* %__gtid___nv_MAIN__F1L49_1__555, align 4, !dbg !86
  %12 = bitcast i32* %.i0000p_368 to i64*, !dbg !86
  %13 = bitcast i32* %.dl0003p.copy_549 to i64*, !dbg !86
  %14 = bitcast i32* %.de0003p.copy_550 to i64*, !dbg !86
  %15 = bitcast i32* %.ds0003p.copy_551 to i64*, !dbg !86
  %16 = load i32, i32* %.ds0003p.copy_551, align 4, !dbg !86
  call void @__kmpc_for_static_init_4(i64* null, i32 %11, i32 34, i64* %12, i64* %13, i64* %14, i64* %15, i32 %16, i32 1), !dbg !86
  %17 = load i32, i32* %.dl0003p.copy_549, align 4, !dbg !86
  store i32 %17, i32* %.dl0003p_405, align 4, !dbg !86
  %18 = load i32, i32* %.de0003p.copy_550, align 4, !dbg !86
  store i32 %18, i32* %.de0003p_401, align 4, !dbg !86
  %19 = load i32, i32* %.ds0003p.copy_551, align 4, !dbg !86
  store i32 %19, i32* %.ds0003p_403, align 4, !dbg !86
  %20 = load i32, i32* %.dl0003p_405, align 4, !dbg !86
  store i32 %20, i32* %j_367, align 4, !dbg !86
  %21 = load i32, i32* %j_367, align 4, !dbg !86
  call void @llvm.dbg.value(metadata i32 %21, metadata !87, metadata !DIExpression()), !dbg !85
  store i32 %21, i32* %.dX0003p_404, align 4, !dbg !86
  %22 = load i32, i32* %.dX0003p_404, align 4, !dbg !86
  %23 = load i32, i32* %.du0003p_400, align 4, !dbg !86
  %24 = icmp sgt i32 %22, %23, !dbg !86
  br i1 %24, label %L.LB2_398, label %L.LB2_584, !dbg !86

L.LB2_584:                                        ; preds = %L.LB2_366
  %25 = load i32, i32* %.dX0003p_404, align 4, !dbg !86
  store i32 %25, i32* %j_367, align 4, !dbg !86
  %26 = load i32, i32* %.di0003p_402, align 4, !dbg !86
  %27 = load i32, i32* %.de0003p_401, align 4, !dbg !86
  %28 = load i32, i32* %.dX0003p_404, align 4, !dbg !86
  %29 = sub nsw i32 %27, %28, !dbg !86
  %30 = add nsw i32 %26, %29, !dbg !86
  %31 = load i32, i32* %.di0003p_402, align 4, !dbg !86
  %32 = sdiv i32 %30, %31, !dbg !86
  store i32 %32, i32* %.dY0003p_399, align 4, !dbg !86
  %33 = load i32, i32* %.dY0003p_399, align 4, !dbg !86
  %34 = icmp sle i32 %33, 0, !dbg !86
  br i1 %34, label %L.LB2_408, label %L.LB2_407, !dbg !86

L.LB2_407:                                        ; preds = %L.LB2_407, %L.LB2_584
  %35 = bitcast i64* %__nv_MAIN__F1L49_1Arg2 to i8*, !dbg !88
  %36 = getelementptr i8, i8* %35, i64 96, !dbg !88
  %37 = bitcast i8* %36 to i32**, !dbg !88
  %38 = load i32*, i32** %37, align 8, !dbg !88
  %39 = load i32, i32* %38, align 4, !dbg !88
  %40 = sext i32 %39 to i64, !dbg !88
  %41 = load i32, i32* %j_367, align 4, !dbg !88
  call void @llvm.dbg.value(metadata i32 %41, metadata !87, metadata !DIExpression()), !dbg !85
  %42 = sext i32 %41 to i64, !dbg !88
  %43 = sub nsw i64 %42, 1, !dbg !88
  %44 = bitcast i64* %__nv_MAIN__F1L49_1Arg2 to i8*, !dbg !88
  %45 = getelementptr i8, i8* %44, i64 104, !dbg !88
  %46 = bitcast i8* %45 to i8**, !dbg !88
  %47 = load i8*, i8** %46, align 8, !dbg !88
  %48 = getelementptr i8, i8* %47, i64 160, !dbg !88
  %49 = bitcast i8* %48 to i64*, !dbg !88
  %50 = load i64, i64* %49, align 8, !dbg !88
  %51 = mul nsw i64 %43, %50, !dbg !88
  %52 = add nsw i64 %40, %51, !dbg !88
  %53 = bitcast i64* %__nv_MAIN__F1L49_1Arg2 to i8*, !dbg !88
  %54 = getelementptr i8, i8* %53, i64 104, !dbg !88
  %55 = bitcast i8* %54 to i8**, !dbg !88
  %56 = load i8*, i8** %55, align 8, !dbg !88
  %57 = getelementptr i8, i8* %56, i64 56, !dbg !88
  %58 = bitcast i8* %57 to i64*, !dbg !88
  %59 = load i64, i64* %58, align 8, !dbg !88
  %60 = add nsw i64 %52, %59, !dbg !88
  %61 = bitcast i64* %__nv_MAIN__F1L49_1Arg2 to i8*, !dbg !88
  %62 = getelementptr i8, i8* %61, i64 16, !dbg !88
  %63 = bitcast i8* %62 to i8***, !dbg !88
  %64 = load i8**, i8*** %63, align 8, !dbg !88
  %65 = load i8*, i8** %64, align 8, !dbg !88
  %66 = getelementptr i8, i8* %65, i64 -4, !dbg !88
  %67 = bitcast i8* %66 to float*, !dbg !88
  %68 = getelementptr float, float* %67, i64 %60, !dbg !88
  %69 = load float, float* %68, align 4, !dbg !88
  %70 = bitcast i64* %__nv_MAIN__F1L49_1Arg2 to i8*, !dbg !88
  %71 = getelementptr i8, i8* %70, i64 104, !dbg !88
  %72 = bitcast i8* %71 to i8**, !dbg !88
  %73 = load i8*, i8** %72, align 8, !dbg !88
  %74 = getelementptr i8, i8* %73, i64 56, !dbg !88
  %75 = bitcast i8* %74 to i64*, !dbg !88
  %76 = load i64, i64* %75, align 8, !dbg !88
  %77 = bitcast i64* %__nv_MAIN__F1L49_1Arg2 to i8*, !dbg !88
  %78 = getelementptr i8, i8* %77, i64 96, !dbg !88
  %79 = bitcast i8* %78 to i32**, !dbg !88
  %80 = load i32*, i32** %79, align 8, !dbg !88
  %81 = load i32, i32* %80, align 4, !dbg !88
  %82 = sext i32 %81 to i64, !dbg !88
  %83 = load i32, i32* %j_367, align 4, !dbg !88
  call void @llvm.dbg.value(metadata i32 %83, metadata !87, metadata !DIExpression()), !dbg !85
  %84 = sext i32 %83 to i64, !dbg !88
  %85 = bitcast i64* %__nv_MAIN__F1L49_1Arg2 to i8*, !dbg !88
  %86 = getelementptr i8, i8* %85, i64 104, !dbg !88
  %87 = bitcast i8* %86 to i8**, !dbg !88
  %88 = load i8*, i8** %87, align 8, !dbg !88
  %89 = getelementptr i8, i8* %88, i64 160, !dbg !88
  %90 = bitcast i8* %89 to i64*, !dbg !88
  %91 = load i64, i64* %90, align 8, !dbg !88
  %92 = mul nsw i64 %84, %91, !dbg !88
  %93 = add nsw i64 %82, %92, !dbg !88
  %94 = add nsw i64 %76, %93, !dbg !88
  %95 = bitcast i64* %__nv_MAIN__F1L49_1Arg2 to i8*, !dbg !88
  %96 = getelementptr i8, i8* %95, i64 16, !dbg !88
  %97 = bitcast i8* %96 to i8***, !dbg !88
  %98 = load i8**, i8*** %97, align 8, !dbg !88
  %99 = load i8*, i8** %98, align 8, !dbg !88
  %100 = getelementptr i8, i8* %99, i64 -4, !dbg !88
  %101 = bitcast i8* %100 to float*, !dbg !88
  %102 = getelementptr float, float* %101, i64 %94, !dbg !88
  store float %69, float* %102, align 4, !dbg !88
  %103 = load i32, i32* %.di0003p_402, align 4, !dbg !85
  %104 = load i32, i32* %j_367, align 4, !dbg !85
  call void @llvm.dbg.value(metadata i32 %104, metadata !87, metadata !DIExpression()), !dbg !85
  %105 = add nsw i32 %103, %104, !dbg !85
  store i32 %105, i32* %j_367, align 4, !dbg !85
  %106 = load i32, i32* %.dY0003p_399, align 4, !dbg !85
  %107 = sub nsw i32 %106, 1, !dbg !85
  store i32 %107, i32* %.dY0003p_399, align 4, !dbg !85
  %108 = load i32, i32* %.dY0003p_399, align 4, !dbg !85
  %109 = icmp sgt i32 %108, 0, !dbg !85
  br i1 %109, label %L.LB2_407, label %L.LB2_408, !dbg !85

L.LB2_408:                                        ; preds = %L.LB2_407, %L.LB2_584
  br label %L.LB2_398

L.LB2_398:                                        ; preds = %L.LB2_408, %L.LB2_366
  %110 = load i32, i32* %__gtid___nv_MAIN__F1L49_1__555, align 4, !dbg !85
  call void @__kmpc_for_static_fini(i64* null, i32 %110), !dbg !85
  br label %L.LB2_369

L.LB2_369:                                        ; preds = %L.LB2_398
  ret void, !dbg !85
}

declare void @__kmpc_for_static_fini(i64*, i32) #0

declare void @__kmpc_for_static_init_4(i64*, i32, i32, i64*, i64*, i64*, i64*, i32, i32) #0

declare void @f90_dealloc03a_i8(...) #0

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
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB038-truedepseconddimension-var-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb038_truedepseconddimension_var_yes", scope: !2, file: !3, line: 11, type: !6, scopeLine: 11, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
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
!18 = !DILocation(line: 59, column: 1, scope: !5)
!19 = !DILocation(line: 11, column: 1, scope: !5)
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
!39 = !DILocation(line: 19, column: 1, scope: !5)
!40 = !DILocation(line: 21, column: 1, scope: !5)
!41 = !DILocalVariable(name: "argcount", scope: !5, file: !3, type: !9)
!42 = !DILocation(line: 22, column: 1, scope: !5)
!43 = !DILocation(line: 23, column: 1, scope: !5)
!44 = !DILocalVariable(scope: !5, file: !3, type: !9, flags: DIFlagArtificial)
!45 = !DILocalVariable(scope: !5, file: !3, type: !27, flags: DIFlagArtificial)
!46 = !DILocation(line: 26, column: 1, scope: !5)
!47 = !DILocalVariable(name: "allocstatus", scope: !5, file: !3, type: !9)
!48 = !DILocation(line: 27, column: 1, scope: !5)
!49 = !DILocation(line: 28, column: 1, scope: !5)
!50 = !DILocation(line: 29, column: 1, scope: !5)
!51 = !DILocation(line: 32, column: 1, scope: !5)
!52 = !DILocalVariable(name: "ix", scope: !5, file: !3, type: !9)
!53 = !DILocation(line: 33, column: 1, scope: !5)
!54 = !DILocation(line: 34, column: 1, scope: !5)
!55 = !DILocation(line: 36, column: 1, scope: !5)
!56 = !DILocation(line: 37, column: 1, scope: !5)
!57 = !DILocalVariable(name: "rderr", scope: !5, file: !3, type: !9)
!58 = !DILocation(line: 38, column: 1, scope: !5)
!59 = !DILocation(line: 39, column: 1, scope: !5)
!60 = !DILocation(line: 43, column: 1, scope: !5)
!61 = !DILocalVariable(name: "n", scope: !5, file: !3, type: !9)
!62 = !DILocation(line: 44, column: 1, scope: !5)
!63 = !DILocalVariable(name: "m", scope: !5, file: !3, type: !9)
!64 = !DILocation(line: 46, column: 1, scope: !5)
!65 = !DILocation(line: 48, column: 1, scope: !5)
!66 = !DILocalVariable(name: "i", scope: !5, file: !3, type: !9)
!67 = !DILocation(line: 49, column: 1, scope: !5)
!68 = !DILocation(line: 54, column: 1, scope: !5)
!69 = !DILocation(line: 58, column: 1, scope: !5)
!70 = distinct !DISubprogram(name: "__nv_MAIN__F1L49_1", scope: !2, file: !3, line: 49, type: !71, scopeLine: 49, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!71 = !DISubroutineType(types: !72)
!72 = !{null, !9, !27, !27}
!73 = !DILocalVariable(name: "__nv_MAIN__F1L49_1Arg0", arg: 1, scope: !70, file: !3, type: !9)
!74 = !DILocation(line: 0, scope: !70)
!75 = !DILocalVariable(name: "__nv_MAIN__F1L49_1Arg1", arg: 2, scope: !70, file: !3, type: !27)
!76 = !DILocalVariable(name: "__nv_MAIN__F1L49_1Arg2", arg: 3, scope: !70, file: !3, type: !27)
!77 = !DILocalVariable(name: "omp_sched_static", scope: !70, file: !3, type: !9)
!78 = !DILocalVariable(name: "omp_sched_dynamic", scope: !70, file: !3, type: !9)
!79 = !DILocalVariable(name: "omp_proc_bind_false", scope: !70, file: !3, type: !9)
!80 = !DILocalVariable(name: "omp_proc_bind_true", scope: !70, file: !3, type: !9)
!81 = !DILocalVariable(name: "omp_proc_bind_master", scope: !70, file: !3, type: !9)
!82 = !DILocalVariable(name: "omp_lock_hint_none", scope: !70, file: !3, type: !9)
!83 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !70, file: !3, type: !9)
!84 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !70, file: !3, type: !9)
!85 = !DILocation(line: 52, column: 1, scope: !70)
!86 = !DILocation(line: 50, column: 1, scope: !70)
!87 = !DILocalVariable(name: "j", scope: !70, file: !3, type: !9)
!88 = !DILocation(line: 51, column: 1, scope: !70)

; ModuleID = '/tmp/DRB040-truedepsingleelement-var-yes-d0443e.ll'
source_filename = "/tmp/DRB040-truedepsingleelement-var-yes-d0443e.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.STATICS1 = type <{ [44 x i8] }>
%astruct.dt88 = type <{ i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8* }>

@.STATICS1 = internal global %struct.STATICS1 <{ [44 x i8] c"\FB\FF\FF\FF\06\00\00\00a(0) =\00\00\EB\FF\FF\FF\00\00\00\00\03\00\00\00\00\00\00\00\01\00\00\00\FF\FF\FF\FF\00\00\00\00" }>, align 16
@.C362_MAIN_ = internal constant i32 53
@.C300_MAIN_ = internal constant i32 2
@.C375_MAIN_ = internal constant i64 25
@.C354_MAIN_ = internal constant [29 x i8] c"Error, invalid integer value."
@.C353_MAIN_ = internal constant i32 39
@.C310_MAIN_ = internal constant i32 25
@.C349_MAIN_ = internal constant [5 x i8] c"(i10)"
@.C346_MAIN_ = internal constant i32 37
@.C368_MAIN_ = internal constant i64 4
@.C347_MAIN_ = internal constant [37 x i8] c"Allocation error, program terminated."
@.C312_MAIN_ = internal constant i32 28
@.C372_MAIN_ = internal constant i64 80
@.C371_MAIN_ = internal constant i64 14
@.C338_MAIN_ = internal constant [35 x i8] c"No command line arguments provided."
@.C336_MAIN_ = internal constant i32 6
@.C337_MAIN_ = internal constant [3 x i8] c"(a)"
@.C285_MAIN_ = internal constant i32 1
@.C309_MAIN_ = internal constant i32 14
@.C334_MAIN_ = internal constant [64 x i8] c"micro-benchmarks-fortran/DRB040-truedepsingleelement-var-yes.f95"
@.C311_MAIN_ = internal constant i32 23
@.C330_MAIN_ = internal constant i32 1000
@.C286_MAIN_ = internal constant i64 1
@.C284_MAIN_ = internal constant i64 0
@.C283_MAIN_ = internal constant i32 0
@.C286___nv_MAIN__F1L47_1 = internal constant i64 1
@.C285___nv_MAIN__F1L47_1 = internal constant i32 1
@.C283___nv_MAIN__F1L47_1 = internal constant i32 0

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__489 = alloca i32, align 4
  %.Z0976_355 = alloca i32*, align 8
  %"a$sd2_374" = alloca [16 x i64], align 8
  %.Z0970_345 = alloca [80 x i8]*, align 8
  %"args$sd1_370" = alloca [16 x i64], align 8
  %len_331 = alloca i32, align 4
  %argcount_314 = alloca i32, align 4
  %z__io_340 = alloca i32, align 4
  %z_b_0_318 = alloca i64, align 8
  %z_b_1_319 = alloca i64, align 8
  %z_e_61_322 = alloca i64, align 8
  %z_b_2_320 = alloca i64, align 8
  %z_b_3_321 = alloca i64, align 8
  %allocstatus_315 = alloca i32, align 4
  %.dY0001_385 = alloca i32, align 4
  %ix_317 = alloca i32, align 4
  %rderr_316 = alloca i32, align 4
  %z_b_4_324 = alloca i64, align 8
  %z_b_5_325 = alloca i64, align 8
  %z_e_68_328 = alloca i64, align 8
  %z_b_6_326 = alloca i64, align 8
  %z_b_7_327 = alloca i64, align 8
  %.uplevelArgPack0001_468 = alloca %astruct.dt88, align 16
  call void @llvm.dbg.value(metadata i32 1, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 2, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 2, metadata !14, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !15, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !16, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 2, metadata !17, metadata !DIExpression()), !dbg !10
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !18
  store i32 %0, i32* %__gtid_MAIN__489, align 4, !dbg !18
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !19
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !19
  call void (i8*, ...) %2(i8* %1), !dbg !19
  call void @llvm.dbg.declare(metadata i32** %.Z0976_355, metadata !20, metadata !DIExpression(DW_OP_deref)), !dbg !10
  %3 = bitcast i32** %.Z0976_355 to i8**, !dbg !19
  store i8* null, i8** %3, align 8, !dbg !19
  call void @llvm.dbg.declare(metadata [16 x i64]* %"a$sd2_374", metadata !24, metadata !DIExpression()), !dbg !10
  %4 = bitcast [16 x i64]* %"a$sd2_374" to i64*, !dbg !19
  store i64 0, i64* %4, align 8, !dbg !19
  call void @llvm.dbg.declare(metadata [80 x i8]** %.Z0970_345, metadata !29, metadata !DIExpression(DW_OP_deref)), !dbg !10
  %5 = bitcast [80 x i8]** %.Z0970_345 to i8**, !dbg !19
  store i8* null, i8** %5, align 8, !dbg !19
  call void @llvm.dbg.declare(metadata [16 x i64]* %"args$sd1_370", metadata !24, metadata !DIExpression()), !dbg !10
  %6 = bitcast [16 x i64]* %"args$sd1_370" to i64*, !dbg !19
  store i64 0, i64* %6, align 8, !dbg !19
  br label %L.LB1_409

L.LB1_409:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %len_331, metadata !32, metadata !DIExpression()), !dbg !10
  store i32 1000, i32* %len_331, align 4, !dbg !33
  %7 = call i32 (...) @f90_cmd_arg_cnt(), !dbg !34
  call void @llvm.dbg.declare(metadata i32* %argcount_314, metadata !35, metadata !DIExpression()), !dbg !10
  store i32 %7, i32* %argcount_314, align 4, !dbg !34
  %8 = load i32, i32* %argcount_314, align 4, !dbg !36
  call void @llvm.dbg.value(metadata i32 %8, metadata !35, metadata !DIExpression()), !dbg !10
  %9 = icmp ne i32 %8, 0, !dbg !36
  br i1 %9, label %L.LB1_379, label %L.LB1_511, !dbg !36

L.LB1_511:                                        ; preds = %L.LB1_409
  call void (...) @_mp_bcs_nest(), !dbg !37
  %10 = bitcast i32* @.C311_MAIN_ to i8*, !dbg !37
  %11 = bitcast [64 x i8]* @.C334_MAIN_ to i8*, !dbg !37
  %12 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !37
  call void (i8*, i8*, i64, ...) %12(i8* %10, i8* %11, i64 64), !dbg !37
  %13 = bitcast i32* @.C309_MAIN_ to i8*, !dbg !37
  %14 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !37
  %15 = bitcast [3 x i8]* @.C337_MAIN_ to i8*, !dbg !37
  %16 = bitcast i32 (...)* @f90io_encode_fmta to i32 (i8*, i8*, i8*, i64, ...)*, !dbg !37
  %17 = call i32 (i8*, i8*, i8*, i64, ...) %16(i8* %13, i8* %14, i8* %15, i64 3), !dbg !37
  call void @llvm.dbg.declare(metadata i32* %z__io_340, metadata !38, metadata !DIExpression()), !dbg !10
  store i32 %17, i32* %z__io_340, align 4, !dbg !37
  %18 = bitcast i32* @.C336_MAIN_ to i8*, !dbg !37
  %19 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !37
  %20 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !37
  %21 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !37
  %22 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %21(i8* %18, i8* null, i8* %19, i8* %20, i8* null, i8* null, i64 0), !dbg !37
  store i32 %22, i32* %z__io_340, align 4, !dbg !37
  %23 = bitcast i32* @.C309_MAIN_ to i8*, !dbg !37
  %24 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !37
  %25 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !37
  %26 = bitcast [35 x i8]* @.C338_MAIN_ to i8*, !dbg !37
  %27 = bitcast i32 (...)* @f90io_fmt_writea to i32 (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !37
  %28 = call i32 (i8*, i8*, i8*, i8*, i64, ...) %27(i8* %23, i8* %24, i8* %25, i8* %26, i64 35), !dbg !37
  store i32 %28, i32* %z__io_340, align 4, !dbg !37
  %29 = call i32 (...) @f90io_fmtw_end(), !dbg !37
  store i32 %29, i32* %z__io_340, align 4, !dbg !37
  call void (...) @_mp_ecs_nest(), !dbg !37
  br label %L.LB1_379

L.LB1_379:                                        ; preds = %L.LB1_511, %L.LB1_409
  call void @llvm.dbg.declare(metadata i64* %z_b_0_318, metadata !39, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_0_318, align 8, !dbg !40
  %30 = load i32, i32* %argcount_314, align 4, !dbg !40
  call void @llvm.dbg.value(metadata i32 %30, metadata !35, metadata !DIExpression()), !dbg !10
  %31 = sext i32 %30 to i64, !dbg !40
  call void @llvm.dbg.declare(metadata i64* %z_b_1_319, metadata !39, metadata !DIExpression()), !dbg !10
  store i64 %31, i64* %z_b_1_319, align 8, !dbg !40
  %32 = load i64, i64* %z_b_1_319, align 8, !dbg !40
  call void @llvm.dbg.value(metadata i64 %32, metadata !39, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_61_322, metadata !39, metadata !DIExpression()), !dbg !10
  store i64 %32, i64* %z_e_61_322, align 8, !dbg !40
  %33 = bitcast [16 x i64]* %"args$sd1_370" to i8*, !dbg !40
  %34 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !40
  %35 = bitcast i64* @.C371_MAIN_ to i8*, !dbg !40
  %36 = bitcast i64* @.C372_MAIN_ to i8*, !dbg !40
  %37 = bitcast i64* %z_b_0_318 to i8*, !dbg !40
  %38 = bitcast i64* %z_b_1_319 to i8*, !dbg !40
  %39 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !40
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %39(i8* %33, i8* %34, i8* %35, i8* %36, i8* %37, i8* %38), !dbg !40
  %40 = bitcast [16 x i64]* %"args$sd1_370" to i8*, !dbg !40
  %41 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !40
  call void (i8*, i32, ...) %41(i8* %40, i32 14), !dbg !40
  %42 = load i64, i64* %z_b_1_319, align 8, !dbg !40
  call void @llvm.dbg.value(metadata i64 %42, metadata !39, metadata !DIExpression()), !dbg !10
  %43 = load i64, i64* %z_b_0_318, align 8, !dbg !40
  call void @llvm.dbg.value(metadata i64 %43, metadata !39, metadata !DIExpression()), !dbg !10
  %44 = sub nsw i64 %43, 1, !dbg !40
  %45 = sub nsw i64 %42, %44, !dbg !40
  call void @llvm.dbg.declare(metadata i64* %z_b_2_320, metadata !39, metadata !DIExpression()), !dbg !10
  store i64 %45, i64* %z_b_2_320, align 8, !dbg !40
  %46 = load i64, i64* %z_b_0_318, align 8, !dbg !40
  call void @llvm.dbg.value(metadata i64 %46, metadata !39, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_b_3_321, metadata !39, metadata !DIExpression()), !dbg !10
  store i64 %46, i64* %z_b_3_321, align 8, !dbg !40
  %47 = bitcast i64* %z_b_2_320 to i8*, !dbg !40
  %48 = bitcast i64* @.C371_MAIN_ to i8*, !dbg !40
  %49 = bitcast i64* @.C372_MAIN_ to i8*, !dbg !40
  call void @llvm.dbg.declare(metadata i32* %allocstatus_315, metadata !41, metadata !DIExpression()), !dbg !10
  %50 = bitcast i32* %allocstatus_315 to i8*, !dbg !40
  %51 = bitcast [80 x i8]** %.Z0970_345 to i8*, !dbg !40
  %52 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !40
  %53 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !40
  %54 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !40
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %54(i8* %47, i8* %48, i8* %49, i8* %50, i8* %51, i8* null, i8* %52, i8* %53, i8* null, i64 0), !dbg !40
  %55 = load i32, i32* %allocstatus_315, align 4, !dbg !42
  call void @llvm.dbg.value(metadata i32 %55, metadata !41, metadata !DIExpression()), !dbg !10
  %56 = icmp sle i32 %55, 0, !dbg !42
  br i1 %56, label %L.LB1_382, label %L.LB1_512, !dbg !42

L.LB1_512:                                        ; preds = %L.LB1_379
  call void (...) @_mp_bcs_nest(), !dbg !43
  %57 = bitcast i32* @.C312_MAIN_ to i8*, !dbg !43
  %58 = bitcast [64 x i8]* @.C334_MAIN_ to i8*, !dbg !43
  %59 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !43
  call void (i8*, i8*, i64, ...) %59(i8* %57, i8* %58, i64 64), !dbg !43
  %60 = bitcast i32* @.C309_MAIN_ to i8*, !dbg !43
  %61 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !43
  %62 = bitcast [3 x i8]* @.C337_MAIN_ to i8*, !dbg !43
  %63 = bitcast i32 (...)* @f90io_encode_fmta to i32 (i8*, i8*, i8*, i64, ...)*, !dbg !43
  %64 = call i32 (i8*, i8*, i8*, i64, ...) %63(i8* %60, i8* %61, i8* %62, i64 3), !dbg !43
  store i32 %64, i32* %z__io_340, align 4, !dbg !43
  %65 = bitcast i32* @.C336_MAIN_ to i8*, !dbg !43
  %66 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !43
  %67 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !43
  %68 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !43
  %69 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %68(i8* %65, i8* null, i8* %66, i8* %67, i8* null, i8* null, i64 0), !dbg !43
  store i32 %69, i32* %z__io_340, align 4, !dbg !43
  %70 = bitcast i32* @.C309_MAIN_ to i8*, !dbg !43
  %71 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !43
  %72 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !43
  %73 = bitcast [37 x i8]* @.C347_MAIN_ to i8*, !dbg !43
  %74 = bitcast i32 (...)* @f90io_fmt_writea to i32 (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !43
  %75 = call i32 (i8*, i8*, i8*, i8*, i64, ...) %74(i8* %70, i8* %71, i8* %72, i8* %73, i64 37), !dbg !43
  store i32 %75, i32* %z__io_340, align 4, !dbg !43
  %76 = call i32 (...) @f90io_fmtw_end(), !dbg !43
  store i32 %76, i32* %z__io_340, align 4, !dbg !43
  call void (...) @_mp_ecs_nest(), !dbg !43
  %77 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !44
  %78 = bitcast void (...)* @f90_stop08a to void (i8*, i8*, i64, ...)*, !dbg !44
  call void (i8*, i8*, i64, ...) %78(i8* %77, i8* null, i64 0), !dbg !44
  br label %L.LB1_382

L.LB1_382:                                        ; preds = %L.LB1_512, %L.LB1_379
  %79 = load i32, i32* %argcount_314, align 4, !dbg !45
  call void @llvm.dbg.value(metadata i32 %79, metadata !35, metadata !DIExpression()), !dbg !10
  store i32 %79, i32* %.dY0001_385, align 4, !dbg !45
  call void @llvm.dbg.declare(metadata i32* %ix_317, metadata !46, metadata !DIExpression()), !dbg !10
  store i32 1, i32* %ix_317, align 4, !dbg !45
  %80 = load i32, i32* %.dY0001_385, align 4, !dbg !45
  %81 = icmp sle i32 %80, 0, !dbg !45
  br i1 %81, label %L.LB1_384, label %L.LB1_383, !dbg !45

L.LB1_383:                                        ; preds = %L.LB1_383, %L.LB1_382
  %82 = bitcast i32* %ix_317 to i8*, !dbg !47
  %83 = load [80 x i8]*, [80 x i8]** %.Z0970_345, align 8, !dbg !47
  call void @llvm.dbg.value(metadata [80 x i8]* %83, metadata !29, metadata !DIExpression()), !dbg !10
  %84 = bitcast [80 x i8]* %83 to i8*, !dbg !47
  %85 = getelementptr i8, i8* %84, i64 -80, !dbg !47
  %86 = load i32, i32* %ix_317, align 4, !dbg !47
  call void @llvm.dbg.value(metadata i32 %86, metadata !46, metadata !DIExpression()), !dbg !10
  %87 = sext i32 %86 to i64, !dbg !47
  %88 = bitcast [16 x i64]* %"args$sd1_370" to i8*, !dbg !47
  %89 = getelementptr i8, i8* %88, i64 56, !dbg !47
  %90 = bitcast i8* %89 to i64*, !dbg !47
  %91 = load i64, i64* %90, align 8, !dbg !47
  %92 = add nsw i64 %87, %91, !dbg !47
  %93 = mul nsw i64 %92, 80, !dbg !47
  %94 = getelementptr i8, i8* %85, i64 %93, !dbg !47
  %95 = bitcast i64* @.C368_MAIN_ to i8*, !dbg !47
  %96 = bitcast void (...)* @f90_get_cmd_arga to void (i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !47
  call void (i8*, i8*, i8*, i8*, i8*, i64, ...) %96(i8* %82, i8* %94, i8* null, i8* null, i8* %95, i64 80), !dbg !47
  %97 = load i32, i32* %ix_317, align 4, !dbg !48
  call void @llvm.dbg.value(metadata i32 %97, metadata !46, metadata !DIExpression()), !dbg !10
  %98 = add nsw i32 %97, 1, !dbg !48
  store i32 %98, i32* %ix_317, align 4, !dbg !48
  %99 = load i32, i32* %.dY0001_385, align 4, !dbg !48
  %100 = sub nsw i32 %99, 1, !dbg !48
  store i32 %100, i32* %.dY0001_385, align 4, !dbg !48
  %101 = load i32, i32* %.dY0001_385, align 4, !dbg !48
  %102 = icmp sgt i32 %101, 0, !dbg !48
  br i1 %102, label %L.LB1_383, label %L.LB1_384, !dbg !48

L.LB1_384:                                        ; preds = %L.LB1_383, %L.LB1_382
  %103 = load i32, i32* %argcount_314, align 4, !dbg !49
  call void @llvm.dbg.value(metadata i32 %103, metadata !35, metadata !DIExpression()), !dbg !10
  %104 = icmp sle i32 %103, 0, !dbg !49
  br i1 %104, label %L.LB1_386, label %L.LB1_513, !dbg !49

L.LB1_513:                                        ; preds = %L.LB1_384
  call void (...) @_mp_bcs_nest(), !dbg !50
  %105 = bitcast i32* @.C346_MAIN_ to i8*, !dbg !50
  %106 = bitcast [64 x i8]* @.C334_MAIN_ to i8*, !dbg !50
  %107 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !50
  call void (i8*, i8*, i64, ...) %107(i8* %105, i8* %106, i64 64), !dbg !50
  %108 = bitcast i32* @.C309_MAIN_ to i8*, !dbg !50
  %109 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !50
  %110 = bitcast [5 x i8]* @.C349_MAIN_ to i8*, !dbg !50
  %111 = bitcast i32 (...)* @f90io_encode_fmta to i32 (i8*, i8*, i8*, i64, ...)*, !dbg !50
  %112 = call i32 (i8*, i8*, i8*, i64, ...) %111(i8* %108, i8* %109, i8* %110, i64 5), !dbg !50
  store i32 %112, i32* %z__io_340, align 4, !dbg !50
  %113 = load [80 x i8]*, [80 x i8]** %.Z0970_345, align 8, !dbg !50
  call void @llvm.dbg.value(metadata [80 x i8]* %113, metadata !29, metadata !DIExpression()), !dbg !10
  %114 = bitcast [80 x i8]* %113 to i8*, !dbg !50
  %115 = bitcast [16 x i64]* %"args$sd1_370" to i8*, !dbg !50
  %116 = getelementptr i8, i8* %115, i64 56, !dbg !50
  %117 = bitcast i8* %116 to i64*, !dbg !50
  %118 = load i64, i64* %117, align 8, !dbg !50
  %119 = mul nsw i64 %118, 80, !dbg !50
  %120 = getelementptr i8, i8* %114, i64 %119, !dbg !50
  %121 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !50
  %122 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !50
  call void @llvm.dbg.declare(metadata i32* %rderr_316, metadata !51, metadata !DIExpression()), !dbg !10
  %123 = bitcast i32* %rderr_316 to i8*, !dbg !50
  %124 = bitcast i32 (...)* @f90io_fmtr_intern_inita to i32 (i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !50
  %125 = call i32 (i8*, i8*, i8*, i8*, i8*, i64, ...) %124(i8* %120, i8* %121, i8* %122, i8* %123, i8* null, i64 80), !dbg !50
  store i32 %125, i32* %z__io_340, align 4, !dbg !50
  %126 = bitcast i32* @.C310_MAIN_ to i8*, !dbg !50
  %127 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !50
  %128 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !50
  %129 = bitcast i32* %len_331 to i8*, !dbg !50
  %130 = bitcast i32 (...)* @f90io_fmt_reada to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !50
  %131 = call i32 (i8*, i8*, i8*, i8*, ...) %130(i8* %126, i8* %127, i8* %128, i8* %129), !dbg !50
  store i32 %131, i32* %z__io_340, align 4, !dbg !50
  %132 = call i32 (...) @f90io_fmtr_end(), !dbg !50
  store i32 %132, i32* %z__io_340, align 4, !dbg !50
  call void (...) @_mp_ecs_nest(), !dbg !50
  %133 = load i32, i32* %rderr_316, align 4, !dbg !52
  call void @llvm.dbg.value(metadata i32 %133, metadata !51, metadata !DIExpression()), !dbg !10
  %134 = icmp eq i32 %133, 0, !dbg !52
  br i1 %134, label %L.LB1_387, label %L.LB1_514, !dbg !52

L.LB1_514:                                        ; preds = %L.LB1_513
  call void (...) @_mp_bcs_nest(), !dbg !53
  %135 = bitcast i32* @.C353_MAIN_ to i8*, !dbg !53
  %136 = bitcast [64 x i8]* @.C334_MAIN_ to i8*, !dbg !53
  %137 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !53
  call void (i8*, i8*, i64, ...) %137(i8* %135, i8* %136, i64 64), !dbg !53
  %138 = bitcast i32* @.C309_MAIN_ to i8*, !dbg !53
  %139 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !53
  %140 = bitcast [3 x i8]* @.C337_MAIN_ to i8*, !dbg !53
  %141 = bitcast i32 (...)* @f90io_encode_fmta to i32 (i8*, i8*, i8*, i64, ...)*, !dbg !53
  %142 = call i32 (i8*, i8*, i8*, i64, ...) %141(i8* %138, i8* %139, i8* %140, i64 3), !dbg !53
  store i32 %142, i32* %z__io_340, align 4, !dbg !53
  %143 = bitcast i32* @.C336_MAIN_ to i8*, !dbg !53
  %144 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !53
  %145 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !53
  %146 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !53
  %147 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %146(i8* %143, i8* null, i8* %144, i8* %145, i8* null, i8* null, i64 0), !dbg !53
  store i32 %147, i32* %z__io_340, align 4, !dbg !53
  %148 = bitcast i32* @.C309_MAIN_ to i8*, !dbg !53
  %149 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !53
  %150 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !53
  %151 = bitcast [29 x i8]* @.C354_MAIN_ to i8*, !dbg !53
  %152 = bitcast i32 (...)* @f90io_fmt_writea to i32 (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !53
  %153 = call i32 (i8*, i8*, i8*, i8*, i64, ...) %152(i8* %148, i8* %149, i8* %150, i8* %151, i64 29), !dbg !53
  store i32 %153, i32* %z__io_340, align 4, !dbg !53
  %154 = call i32 (...) @f90io_fmtw_end(), !dbg !53
  store i32 %154, i32* %z__io_340, align 4, !dbg !53
  call void (...) @_mp_ecs_nest(), !dbg !53
  br label %L.LB1_387

L.LB1_387:                                        ; preds = %L.LB1_514, %L.LB1_513
  br label %L.LB1_386

L.LB1_386:                                        ; preds = %L.LB1_387, %L.LB1_384
  call void @llvm.dbg.declare(metadata i64* %z_b_4_324, metadata !39, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_4_324, align 8, !dbg !54
  %155 = load i32, i32* %len_331, align 4, !dbg !54
  call void @llvm.dbg.value(metadata i32 %155, metadata !32, metadata !DIExpression()), !dbg !10
  %156 = sext i32 %155 to i64, !dbg !54
  call void @llvm.dbg.declare(metadata i64* %z_b_5_325, metadata !39, metadata !DIExpression()), !dbg !10
  store i64 %156, i64* %z_b_5_325, align 8, !dbg !54
  %157 = load i64, i64* %z_b_5_325, align 8, !dbg !54
  call void @llvm.dbg.value(metadata i64 %157, metadata !39, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_68_328, metadata !39, metadata !DIExpression()), !dbg !10
  store i64 %157, i64* %z_e_68_328, align 8, !dbg !54
  %158 = bitcast [16 x i64]* %"a$sd2_374" to i8*, !dbg !54
  %159 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !54
  %160 = bitcast i64* @.C375_MAIN_ to i8*, !dbg !54
  %161 = bitcast i64* @.C368_MAIN_ to i8*, !dbg !54
  %162 = bitcast i64* %z_b_4_324 to i8*, !dbg !54
  %163 = bitcast i64* %z_b_5_325 to i8*, !dbg !54
  %164 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !54
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %164(i8* %158, i8* %159, i8* %160, i8* %161, i8* %162, i8* %163), !dbg !54
  %165 = bitcast [16 x i64]* %"a$sd2_374" to i8*, !dbg !54
  %166 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !54
  call void (i8*, i32, ...) %166(i8* %165, i32 25), !dbg !54
  %167 = load i64, i64* %z_b_5_325, align 8, !dbg !54
  call void @llvm.dbg.value(metadata i64 %167, metadata !39, metadata !DIExpression()), !dbg !10
  %168 = load i64, i64* %z_b_4_324, align 8, !dbg !54
  call void @llvm.dbg.value(metadata i64 %168, metadata !39, metadata !DIExpression()), !dbg !10
  %169 = sub nsw i64 %168, 1, !dbg !54
  %170 = sub nsw i64 %167, %169, !dbg !54
  call void @llvm.dbg.declare(metadata i64* %z_b_6_326, metadata !39, metadata !DIExpression()), !dbg !10
  store i64 %170, i64* %z_b_6_326, align 8, !dbg !54
  %171 = load i64, i64* %z_b_4_324, align 8, !dbg !54
  call void @llvm.dbg.value(metadata i64 %171, metadata !39, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_b_7_327, metadata !39, metadata !DIExpression()), !dbg !10
  store i64 %171, i64* %z_b_7_327, align 8, !dbg !54
  %172 = bitcast i64* %z_b_6_326 to i8*, !dbg !54
  %173 = bitcast i64* @.C375_MAIN_ to i8*, !dbg !54
  %174 = bitcast i64* @.C368_MAIN_ to i8*, !dbg !54
  %175 = bitcast i32** %.Z0976_355 to i8*, !dbg !54
  %176 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !54
  %177 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !54
  %178 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !54
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %178(i8* %172, i8* %173, i8* %174, i8* null, i8* %175, i8* null, i8* %176, i8* %177, i8* null, i64 0), !dbg !54
  %179 = bitcast [16 x i64]* %"a$sd2_374" to i8*, !dbg !55
  %180 = getelementptr i8, i8* %179, i64 56, !dbg !55
  %181 = bitcast i8* %180 to i64*, !dbg !55
  %182 = load i64, i64* %181, align 8, !dbg !55
  %183 = load i32*, i32** %.Z0976_355, align 8, !dbg !55
  call void @llvm.dbg.value(metadata i32* %183, metadata !20, metadata !DIExpression()), !dbg !10
  %184 = getelementptr i32, i32* %183, i64 %182, !dbg !55
  store i32 2, i32* %184, align 4, !dbg !55
  %185 = bitcast i32* %len_331 to i8*, !dbg !56
  %186 = bitcast %astruct.dt88* %.uplevelArgPack0001_468 to i8**, !dbg !56
  store i8* %185, i8** %186, align 8, !dbg !56
  %187 = bitcast i32** %.Z0976_355 to i8*, !dbg !56
  %188 = bitcast %astruct.dt88* %.uplevelArgPack0001_468 to i8*, !dbg !56
  %189 = getelementptr i8, i8* %188, i64 8, !dbg !56
  %190 = bitcast i8* %189 to i8**, !dbg !56
  store i8* %187, i8** %190, align 8, !dbg !56
  %191 = bitcast i32** %.Z0976_355 to i8*, !dbg !56
  %192 = bitcast %astruct.dt88* %.uplevelArgPack0001_468 to i8*, !dbg !56
  %193 = getelementptr i8, i8* %192, i64 16, !dbg !56
  %194 = bitcast i8* %193 to i8**, !dbg !56
  store i8* %191, i8** %194, align 8, !dbg !56
  %195 = bitcast i64* %z_b_4_324 to i8*, !dbg !56
  %196 = bitcast %astruct.dt88* %.uplevelArgPack0001_468 to i8*, !dbg !56
  %197 = getelementptr i8, i8* %196, i64 24, !dbg !56
  %198 = bitcast i8* %197 to i8**, !dbg !56
  store i8* %195, i8** %198, align 8, !dbg !56
  %199 = bitcast i64* %z_b_5_325 to i8*, !dbg !56
  %200 = bitcast %astruct.dt88* %.uplevelArgPack0001_468 to i8*, !dbg !56
  %201 = getelementptr i8, i8* %200, i64 32, !dbg !56
  %202 = bitcast i8* %201 to i8**, !dbg !56
  store i8* %199, i8** %202, align 8, !dbg !56
  %203 = bitcast i64* %z_e_68_328 to i8*, !dbg !56
  %204 = bitcast %astruct.dt88* %.uplevelArgPack0001_468 to i8*, !dbg !56
  %205 = getelementptr i8, i8* %204, i64 40, !dbg !56
  %206 = bitcast i8* %205 to i8**, !dbg !56
  store i8* %203, i8** %206, align 8, !dbg !56
  %207 = bitcast i64* %z_b_6_326 to i8*, !dbg !56
  %208 = bitcast %astruct.dt88* %.uplevelArgPack0001_468 to i8*, !dbg !56
  %209 = getelementptr i8, i8* %208, i64 48, !dbg !56
  %210 = bitcast i8* %209 to i8**, !dbg !56
  store i8* %207, i8** %210, align 8, !dbg !56
  %211 = bitcast i64* %z_b_7_327 to i8*, !dbg !56
  %212 = bitcast %astruct.dt88* %.uplevelArgPack0001_468 to i8*, !dbg !56
  %213 = getelementptr i8, i8* %212, i64 56, !dbg !56
  %214 = bitcast i8* %213 to i8**, !dbg !56
  store i8* %211, i8** %214, align 8, !dbg !56
  %215 = bitcast [16 x i64]* %"a$sd2_374" to i8*, !dbg !56
  %216 = bitcast %astruct.dt88* %.uplevelArgPack0001_468 to i8*, !dbg !56
  %217 = getelementptr i8, i8* %216, i64 64, !dbg !56
  %218 = bitcast i8* %217 to i8**, !dbg !56
  store i8* %215, i8** %218, align 8, !dbg !56
  br label %L.LB1_487, !dbg !56

L.LB1_487:                                        ; preds = %L.LB1_386
  %219 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L47_1_ to i64*, !dbg !56
  %220 = bitcast %astruct.dt88* %.uplevelArgPack0001_468 to i64*, !dbg !56
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %219, i64* %220), !dbg !56
  call void (...) @_mp_bcs_nest(), !dbg !57
  %221 = bitcast i32* @.C362_MAIN_ to i8*, !dbg !57
  %222 = bitcast [64 x i8]* @.C334_MAIN_ to i8*, !dbg !57
  %223 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !57
  call void (i8*, i8*, i64, ...) %223(i8* %221, i8* %222, i64 64), !dbg !57
  %224 = bitcast i32* @.C336_MAIN_ to i8*, !dbg !57
  %225 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !57
  %226 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !57
  %227 = bitcast %struct.STATICS1* @.STATICS1 to i8*, !dbg !57
  %228 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !57
  %229 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %228(i8* %224, i8* null, i8* %225, i8* %226, i8* %227, i8* null, i64 0), !dbg !57
  store i32 %229, i32* %z__io_340, align 4, !dbg !57
  %230 = bitcast [16 x i64]* %"a$sd2_374" to i8*, !dbg !57
  %231 = getelementptr i8, i8* %230, i64 56, !dbg !57
  %232 = bitcast i8* %231 to i64*, !dbg !57
  %233 = load i64, i64* %232, align 8, !dbg !57
  %234 = load i32*, i32** %.Z0976_355, align 8, !dbg !57
  call void @llvm.dbg.value(metadata i32* %234, metadata !20, metadata !DIExpression()), !dbg !10
  %235 = bitcast i32* %234 to i8*, !dbg !57
  %236 = getelementptr i8, i8* %235, i64 -4, !dbg !57
  %237 = bitcast i8* %236 to i32*, !dbg !57
  %238 = getelementptr i32, i32* %237, i64 %233, !dbg !57
  %239 = load i32, i32* %238, align 4, !dbg !57
  %240 = bitcast i32 (...)* @f90io_sc_i_fmt_write to i32 (i32, i32, ...)*, !dbg !57
  %241 = call i32 (i32, i32, ...) %240(i32 %239, i32 25), !dbg !57
  store i32 %241, i32* %z__io_340, align 4, !dbg !57
  %242 = call i32 (...) @f90io_fmtw_end(), !dbg !57
  store i32 %242, i32* %z__io_340, align 4, !dbg !57
  call void (...) @_mp_ecs_nest(), !dbg !57
  %243 = load [80 x i8]*, [80 x i8]** %.Z0970_345, align 8, !dbg !58
  call void @llvm.dbg.value(metadata [80 x i8]* %243, metadata !29, metadata !DIExpression()), !dbg !10
  %244 = bitcast [80 x i8]* %243 to i8*, !dbg !58
  %245 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !58
  %246 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, i64, ...)*, !dbg !58
  call void (i8*, i8*, i8*, i8*, i64, i64, ...) %246(i8* null, i8* %244, i8* %245, i8* null, i64 80, i64 0), !dbg !58
  %247 = bitcast [80 x i8]** %.Z0970_345 to i8**, !dbg !58
  store i8* null, i8** %247, align 8, !dbg !58
  %248 = bitcast [16 x i64]* %"args$sd1_370" to i64*, !dbg !58
  store i64 0, i64* %248, align 8, !dbg !58
  %249 = load i32*, i32** %.Z0976_355, align 8, !dbg !58
  call void @llvm.dbg.value(metadata i32* %249, metadata !20, metadata !DIExpression()), !dbg !10
  %250 = bitcast i32* %249 to i8*, !dbg !58
  %251 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !58
  %252 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !58
  call void (i8*, i8*, i8*, i8*, i64, ...) %252(i8* null, i8* %250, i8* %251, i8* null, i64 0), !dbg !58
  %253 = bitcast i32** %.Z0976_355 to i8**, !dbg !58
  store i8* null, i8** %253, align 8, !dbg !58
  %254 = bitcast [16 x i64]* %"a$sd2_374" to i64*, !dbg !58
  store i64 0, i64* %254, align 8, !dbg !58
  ret void, !dbg !18
}

define internal void @__nv_MAIN__F1L47_1_(i32* %__nv_MAIN__F1L47_1Arg0, i64* %__nv_MAIN__F1L47_1Arg1, i64* %__nv_MAIN__F1L47_1Arg2) #0 !dbg !59 {
L.entry:
  %__gtid___nv_MAIN__F1L47_1__533 = alloca i32, align 4
  %.i0000p_360 = alloca i32, align 4
  %i_359 = alloca i32, align 4
  %.du0002p_391 = alloca i32, align 4
  %.de0002p_392 = alloca i32, align 4
  %.di0002p_393 = alloca i32, align 4
  %.ds0002p_394 = alloca i32, align 4
  %.dl0002p_396 = alloca i32, align 4
  %.dl0002p.copy_527 = alloca i32, align 4
  %.de0002p.copy_528 = alloca i32, align 4
  %.ds0002p.copy_529 = alloca i32, align 4
  %.dX0002p_395 = alloca i32, align 4
  %.dY0002p_390 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L47_1Arg0, metadata !62, metadata !DIExpression()), !dbg !63
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L47_1Arg1, metadata !64, metadata !DIExpression()), !dbg !63
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L47_1Arg2, metadata !65, metadata !DIExpression()), !dbg !63
  call void @llvm.dbg.value(metadata i32 1, metadata !66, metadata !DIExpression()), !dbg !63
  call void @llvm.dbg.value(metadata i32 2, metadata !67, metadata !DIExpression()), !dbg !63
  call void @llvm.dbg.value(metadata i32 0, metadata !68, metadata !DIExpression()), !dbg !63
  call void @llvm.dbg.value(metadata i32 1, metadata !69, metadata !DIExpression()), !dbg !63
  call void @llvm.dbg.value(metadata i32 2, metadata !70, metadata !DIExpression()), !dbg !63
  call void @llvm.dbg.value(metadata i32 0, metadata !71, metadata !DIExpression()), !dbg !63
  call void @llvm.dbg.value(metadata i32 1, metadata !72, metadata !DIExpression()), !dbg !63
  call void @llvm.dbg.value(metadata i32 2, metadata !73, metadata !DIExpression()), !dbg !63
  %0 = load i32, i32* %__nv_MAIN__F1L47_1Arg0, align 4, !dbg !74
  store i32 %0, i32* %__gtid___nv_MAIN__F1L47_1__533, align 4, !dbg !74
  br label %L.LB2_518

L.LB2_518:                                        ; preds = %L.entry
  br label %L.LB2_358

L.LB2_358:                                        ; preds = %L.LB2_518
  store i32 0, i32* %.i0000p_360, align 4, !dbg !75
  call void @llvm.dbg.declare(metadata i32* %i_359, metadata !76, metadata !DIExpression()), !dbg !74
  store i32 1, i32* %i_359, align 4, !dbg !75
  %1 = bitcast i64* %__nv_MAIN__F1L47_1Arg2 to i32**, !dbg !75
  %2 = load i32*, i32** %1, align 8, !dbg !75
  %3 = load i32, i32* %2, align 4, !dbg !75
  store i32 %3, i32* %.du0002p_391, align 4, !dbg !75
  %4 = bitcast i64* %__nv_MAIN__F1L47_1Arg2 to i32**, !dbg !75
  %5 = load i32*, i32** %4, align 8, !dbg !75
  %6 = load i32, i32* %5, align 4, !dbg !75
  store i32 %6, i32* %.de0002p_392, align 4, !dbg !75
  store i32 1, i32* %.di0002p_393, align 4, !dbg !75
  %7 = load i32, i32* %.di0002p_393, align 4, !dbg !75
  store i32 %7, i32* %.ds0002p_394, align 4, !dbg !75
  store i32 1, i32* %.dl0002p_396, align 4, !dbg !75
  %8 = load i32, i32* %.dl0002p_396, align 4, !dbg !75
  store i32 %8, i32* %.dl0002p.copy_527, align 4, !dbg !75
  %9 = load i32, i32* %.de0002p_392, align 4, !dbg !75
  store i32 %9, i32* %.de0002p.copy_528, align 4, !dbg !75
  %10 = load i32, i32* %.ds0002p_394, align 4, !dbg !75
  store i32 %10, i32* %.ds0002p.copy_529, align 4, !dbg !75
  %11 = load i32, i32* %__gtid___nv_MAIN__F1L47_1__533, align 4, !dbg !75
  %12 = bitcast i32* %.i0000p_360 to i64*, !dbg !75
  %13 = bitcast i32* %.dl0002p.copy_527 to i64*, !dbg !75
  %14 = bitcast i32* %.de0002p.copy_528 to i64*, !dbg !75
  %15 = bitcast i32* %.ds0002p.copy_529 to i64*, !dbg !75
  %16 = load i32, i32* %.ds0002p.copy_529, align 4, !dbg !75
  call void @__kmpc_for_static_init_4(i64* null, i32 %11, i32 34, i64* %12, i64* %13, i64* %14, i64* %15, i32 %16, i32 1), !dbg !75
  %17 = load i32, i32* %.dl0002p.copy_527, align 4, !dbg !75
  store i32 %17, i32* %.dl0002p_396, align 4, !dbg !75
  %18 = load i32, i32* %.de0002p.copy_528, align 4, !dbg !75
  store i32 %18, i32* %.de0002p_392, align 4, !dbg !75
  %19 = load i32, i32* %.ds0002p.copy_529, align 4, !dbg !75
  store i32 %19, i32* %.ds0002p_394, align 4, !dbg !75
  %20 = load i32, i32* %.dl0002p_396, align 4, !dbg !75
  store i32 %20, i32* %i_359, align 4, !dbg !75
  %21 = load i32, i32* %i_359, align 4, !dbg !75
  call void @llvm.dbg.value(metadata i32 %21, metadata !76, metadata !DIExpression()), !dbg !74
  store i32 %21, i32* %.dX0002p_395, align 4, !dbg !75
  %22 = load i32, i32* %.dX0002p_395, align 4, !dbg !75
  %23 = load i32, i32* %.du0002p_391, align 4, !dbg !75
  %24 = icmp sgt i32 %22, %23, !dbg !75
  br i1 %24, label %L.LB2_389, label %L.LB2_557, !dbg !75

L.LB2_557:                                        ; preds = %L.LB2_358
  %25 = load i32, i32* %.dX0002p_395, align 4, !dbg !75
  store i32 %25, i32* %i_359, align 4, !dbg !75
  %26 = load i32, i32* %.di0002p_393, align 4, !dbg !75
  %27 = load i32, i32* %.de0002p_392, align 4, !dbg !75
  %28 = load i32, i32* %.dX0002p_395, align 4, !dbg !75
  %29 = sub nsw i32 %27, %28, !dbg !75
  %30 = add nsw i32 %26, %29, !dbg !75
  %31 = load i32, i32* %.di0002p_393, align 4, !dbg !75
  %32 = sdiv i32 %30, %31, !dbg !75
  store i32 %32, i32* %.dY0002p_390, align 4, !dbg !75
  %33 = load i32, i32* %.dY0002p_390, align 4, !dbg !75
  %34 = icmp sle i32 %33, 0, !dbg !75
  br i1 %34, label %L.LB2_399, label %L.LB2_398, !dbg !75

L.LB2_398:                                        ; preds = %L.LB2_398, %L.LB2_557
  %35 = load i32, i32* %i_359, align 4, !dbg !77
  call void @llvm.dbg.value(metadata i32 %35, metadata !76, metadata !DIExpression()), !dbg !74
  %36 = sext i32 %35 to i64, !dbg !77
  %37 = bitcast i64* %__nv_MAIN__F1L47_1Arg2 to i8*, !dbg !77
  %38 = getelementptr i8, i8* %37, i64 64, !dbg !77
  %39 = bitcast i8* %38 to i8**, !dbg !77
  %40 = load i8*, i8** %39, align 8, !dbg !77
  %41 = getelementptr i8, i8* %40, i64 56, !dbg !77
  %42 = bitcast i8* %41 to i64*, !dbg !77
  %43 = load i64, i64* %42, align 8, !dbg !77
  %44 = add nsw i64 %36, %43, !dbg !77
  %45 = bitcast i64* %__nv_MAIN__F1L47_1Arg2 to i8*, !dbg !77
  %46 = getelementptr i8, i8* %45, i64 16, !dbg !77
  %47 = bitcast i8* %46 to i8***, !dbg !77
  %48 = load i8**, i8*** %47, align 8, !dbg !77
  %49 = load i8*, i8** %48, align 8, !dbg !77
  %50 = getelementptr i8, i8* %49, i64 -4, !dbg !77
  %51 = bitcast i8* %50 to i32*, !dbg !77
  %52 = getelementptr i32, i32* %51, i64 %44, !dbg !77
  %53 = load i32, i32* %52, align 4, !dbg !77
  %54 = bitcast i64* %__nv_MAIN__F1L47_1Arg2 to i8*, !dbg !77
  %55 = getelementptr i8, i8* %54, i64 64, !dbg !77
  %56 = bitcast i8* %55 to i8**, !dbg !77
  %57 = load i8*, i8** %56, align 8, !dbg !77
  %58 = getelementptr i8, i8* %57, i64 56, !dbg !77
  %59 = bitcast i8* %58 to i64*, !dbg !77
  %60 = load i64, i64* %59, align 8, !dbg !77
  %61 = bitcast i64* %__nv_MAIN__F1L47_1Arg2 to i8*, !dbg !77
  %62 = getelementptr i8, i8* %61, i64 16, !dbg !77
  %63 = bitcast i8* %62 to i32***, !dbg !77
  %64 = load i32**, i32*** %63, align 8, !dbg !77
  %65 = load i32*, i32** %64, align 8, !dbg !77
  %66 = getelementptr i32, i32* %65, i64 %60, !dbg !77
  %67 = load i32, i32* %66, align 4, !dbg !77
  %68 = add nsw i32 %53, %67, !dbg !77
  %69 = load i32, i32* %i_359, align 4, !dbg !77
  call void @llvm.dbg.value(metadata i32 %69, metadata !76, metadata !DIExpression()), !dbg !74
  %70 = sext i32 %69 to i64, !dbg !77
  %71 = bitcast i64* %__nv_MAIN__F1L47_1Arg2 to i8*, !dbg !77
  %72 = getelementptr i8, i8* %71, i64 64, !dbg !77
  %73 = bitcast i8* %72 to i8**, !dbg !77
  %74 = load i8*, i8** %73, align 8, !dbg !77
  %75 = getelementptr i8, i8* %74, i64 56, !dbg !77
  %76 = bitcast i8* %75 to i64*, !dbg !77
  %77 = load i64, i64* %76, align 8, !dbg !77
  %78 = add nsw i64 %70, %77, !dbg !77
  %79 = bitcast i64* %__nv_MAIN__F1L47_1Arg2 to i8*, !dbg !77
  %80 = getelementptr i8, i8* %79, i64 16, !dbg !77
  %81 = bitcast i8* %80 to i8***, !dbg !77
  %82 = load i8**, i8*** %81, align 8, !dbg !77
  %83 = load i8*, i8** %82, align 8, !dbg !77
  %84 = getelementptr i8, i8* %83, i64 -4, !dbg !77
  %85 = bitcast i8* %84 to i32*, !dbg !77
  %86 = getelementptr i32, i32* %85, i64 %78, !dbg !77
  store i32 %68, i32* %86, align 4, !dbg !77
  %87 = load i32, i32* %.di0002p_393, align 4, !dbg !74
  %88 = load i32, i32* %i_359, align 4, !dbg !74
  call void @llvm.dbg.value(metadata i32 %88, metadata !76, metadata !DIExpression()), !dbg !74
  %89 = add nsw i32 %87, %88, !dbg !74
  store i32 %89, i32* %i_359, align 4, !dbg !74
  %90 = load i32, i32* %.dY0002p_390, align 4, !dbg !74
  %91 = sub nsw i32 %90, 1, !dbg !74
  store i32 %91, i32* %.dY0002p_390, align 4, !dbg !74
  %92 = load i32, i32* %.dY0002p_390, align 4, !dbg !74
  %93 = icmp sgt i32 %92, 0, !dbg !74
  br i1 %93, label %L.LB2_398, label %L.LB2_399, !dbg !74

L.LB2_399:                                        ; preds = %L.LB2_398, %L.LB2_557
  br label %L.LB2_389

L.LB2_389:                                        ; preds = %L.LB2_399, %L.LB2_358
  %94 = load i32, i32* %__gtid___nv_MAIN__F1L47_1__533, align 4, !dbg !74
  call void @__kmpc_for_static_fini(i64* null, i32 %94), !dbg !74
  br label %L.LB2_361

L.LB2_361:                                        ; preds = %L.LB2_389
  ret void, !dbg !74
}

declare void @__kmpc_for_static_fini(i64*, i32) #0

declare void @__kmpc_for_static_init_4(i64*, i32, i32, i64*, i64*, i64*, i64*, i32, i32) #0

declare void @f90_dealloc03a_i8(...) #0

declare signext i32 @f90io_sc_i_fmt_write(...) #0

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
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB040-truedepsingleelement-var-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb040_truedepsingleelement_var_yes", scope: !2, file: !3, line: 11, type: !6, scopeLine: 11, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
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
!18 = !DILocation(line: 57, column: 1, scope: !5)
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
!56 = !DILocation(line: 47, column: 1, scope: !5)
!57 = !DILocation(line: 53, column: 1, scope: !5)
!58 = !DILocation(line: 56, column: 1, scope: !5)
!59 = distinct !DISubprogram(name: "__nv_MAIN__F1L47_1", scope: !2, file: !3, line: 47, type: !60, scopeLine: 47, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!60 = !DISubroutineType(types: !61)
!61 = !{null, !9, !26, !26}
!62 = !DILocalVariable(name: "__nv_MAIN__F1L47_1Arg0", arg: 1, scope: !59, file: !3, type: !9)
!63 = !DILocation(line: 0, scope: !59)
!64 = !DILocalVariable(name: "__nv_MAIN__F1L47_1Arg1", arg: 2, scope: !59, file: !3, type: !26)
!65 = !DILocalVariable(name: "__nv_MAIN__F1L47_1Arg2", arg: 3, scope: !59, file: !3, type: !26)
!66 = !DILocalVariable(name: "omp_sched_static", scope: !59, file: !3, type: !9)
!67 = !DILocalVariable(name: "omp_sched_dynamic", scope: !59, file: !3, type: !9)
!68 = !DILocalVariable(name: "omp_proc_bind_false", scope: !59, file: !3, type: !9)
!69 = !DILocalVariable(name: "omp_proc_bind_true", scope: !59, file: !3, type: !9)
!70 = !DILocalVariable(name: "omp_proc_bind_master", scope: !59, file: !3, type: !9)
!71 = !DILocalVariable(name: "omp_lock_hint_none", scope: !59, file: !3, type: !9)
!72 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !59, file: !3, type: !9)
!73 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !59, file: !3, type: !9)
!74 = !DILocation(line: 50, column: 1, scope: !59)
!75 = !DILocation(line: 48, column: 1, scope: !59)
!76 = !DILocalVariable(name: "i", scope: !59, file: !3, type: !9)
!77 = !DILocation(line: 49, column: 1, scope: !59)

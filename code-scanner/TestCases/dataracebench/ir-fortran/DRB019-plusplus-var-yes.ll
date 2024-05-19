; ModuleID = '/tmp/DRB019-plusplus-var-yes-ea4240.ll'
source_filename = "/tmp/DRB019-plusplus-var-yes-ea4240.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.STATICS1 = type <{ [48 x i8] }>
%astruct.dt96 = type <{ i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8* }>

@.STATICS1 = internal global %struct.STATICS1 <{ [48 x i8] c"\FB\FF\FF\FF\0A\00\00\00output(0)=\00\00\EB\FF\FF\FF\00\00\00\00\03\00\00\00\00\00\00\00\01\00\00\00\FF\FF\FF\FF\00\00\00\00" }>, align 16
@.C366_MAIN_ = internal constant i32 64
@.C379_MAIN_ = internal constant i64 25
@.C357_MAIN_ = internal constant [29 x i8] c"Error, invalid integer value."
@.C356_MAIN_ = internal constant i32 46
@.C306_MAIN_ = internal constant i32 25
@.C352_MAIN_ = internal constant [5 x i8] c"(i10)"
@.C351_MAIN_ = internal constant i32 44
@.C372_MAIN_ = internal constant i64 4
@.C349_MAIN_ = internal constant [37 x i8] c"Allocation error, program terminated."
@.C340_MAIN_ = internal constant i32 35
@.C376_MAIN_ = internal constant i64 80
@.C375_MAIN_ = internal constant i64 14
@.C341_MAIN_ = internal constant [35 x i8] c"No command line arguments provided."
@.C338_MAIN_ = internal constant i32 6
@.C339_MAIN_ = internal constant [3 x i8] c"(a)"
@.C305_MAIN_ = internal constant i32 14
@.C335_MAIN_ = internal constant [52 x i8] c"micro-benchmarks-fortran/DRB019-plusplus-var-yes.f95"
@.C337_MAIN_ = internal constant i32 30
@.C285_MAIN_ = internal constant i32 1
@.C332_MAIN_ = internal constant i32 1000
@.C286_MAIN_ = internal constant i64 1
@.C284_MAIN_ = internal constant i64 0
@.C283_MAIN_ = internal constant i32 0
@.C285___nv_MAIN__F1L57_1 = internal constant i32 1
@.C283___nv_MAIN__F1L57_1 = internal constant i32 0

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__527 = alloca i32, align 4
  %.Z0984_359 = alloca i32*, align 8
  %"output$sd3_380" = alloca [16 x i64], align 8
  %.Z0978_358 = alloca i32*, align 8
  %"input$sd2_378" = alloca [16 x i64], align 8
  %.Z0972_348 = alloca [80 x i8]*, align 8
  %"args$sd1_374" = alloca [16 x i64], align 8
  %inlen_308 = alloca i32, align 4
  %outlen_309 = alloca i32, align 4
  %argcount_310 = alloca i32, align 4
  %z__io_343 = alloca i32, align 4
  %z_b_0_314 = alloca i64, align 8
  %z_b_1_315 = alloca i64, align 8
  %z_e_61_318 = alloca i64, align 8
  %z_b_2_316 = alloca i64, align 8
  %z_b_3_317 = alloca i64, align 8
  %allocstatus_311 = alloca i32, align 4
  %.dY0001_390 = alloca i32, align 4
  %ix_313 = alloca i32, align 4
  %rderr_312 = alloca i32, align 4
  %z_b_4_320 = alloca i64, align 8
  %z_b_5_321 = alloca i64, align 8
  %z_e_68_324 = alloca i64, align 8
  %z_b_6_322 = alloca i64, align 8
  %z_b_7_323 = alloca i64, align 8
  %z_b_8_326 = alloca i64, align 8
  %z_b_9_327 = alloca i64, align 8
  %z_e_75_330 = alloca i64, align 8
  %z_b_10_328 = alloca i64, align 8
  %z_b_11_329 = alloca i64, align 8
  %.dY0002_395 = alloca i32, align 4
  %i_307 = alloca i32, align 4
  %.uplevelArgPack0001_488 = alloca %astruct.dt96, align 16
  call void @llvm.dbg.value(metadata i32 1, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !14, metadata !DIExpression()), !dbg !10
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !15
  store i32 %0, i32* %__gtid_MAIN__527, align 4, !dbg !15
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !16
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !16
  call void (i8*, ...) %2(i8* %1), !dbg !16
  call void @llvm.dbg.declare(metadata i32** %.Z0984_359, metadata !17, metadata !DIExpression(DW_OP_deref)), !dbg !10
  %3 = bitcast i32** %.Z0984_359 to i8**, !dbg !16
  store i8* null, i8** %3, align 8, !dbg !16
  call void @llvm.dbg.declare(metadata [16 x i64]* %"output$sd3_380", metadata !21, metadata !DIExpression()), !dbg !10
  %4 = bitcast [16 x i64]* %"output$sd3_380" to i64*, !dbg !16
  store i64 0, i64* %4, align 8, !dbg !16
  call void @llvm.dbg.declare(metadata i32** %.Z0978_358, metadata !26, metadata !DIExpression(DW_OP_deref)), !dbg !10
  %5 = bitcast i32** %.Z0978_358 to i8**, !dbg !16
  store i8* null, i8** %5, align 8, !dbg !16
  call void @llvm.dbg.declare(metadata [16 x i64]* %"input$sd2_378", metadata !21, metadata !DIExpression()), !dbg !10
  %6 = bitcast [16 x i64]* %"input$sd2_378" to i64*, !dbg !16
  store i64 0, i64* %6, align 8, !dbg !16
  call void @llvm.dbg.declare(metadata [80 x i8]** %.Z0972_348, metadata !27, metadata !DIExpression(DW_OP_deref)), !dbg !10
  %7 = bitcast [80 x i8]** %.Z0972_348 to i8**, !dbg !16
  store i8* null, i8** %7, align 8, !dbg !16
  call void @llvm.dbg.declare(metadata [16 x i64]* %"args$sd1_374", metadata !21, metadata !DIExpression()), !dbg !10
  %8 = bitcast [16 x i64]* %"args$sd1_374" to i64*, !dbg !16
  store i64 0, i64* %8, align 8, !dbg !16
  br label %L.LB1_419

L.LB1_419:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %inlen_308, metadata !30, metadata !DIExpression()), !dbg !10
  store i32 1000, i32* %inlen_308, align 4, !dbg !31
  call void @llvm.dbg.declare(metadata i32* %outlen_309, metadata !32, metadata !DIExpression()), !dbg !10
  store i32 1, i32* %outlen_309, align 4, !dbg !33
  %9 = call i32 (...) @f90_cmd_arg_cnt(), !dbg !34
  call void @llvm.dbg.declare(metadata i32* %argcount_310, metadata !35, metadata !DIExpression()), !dbg !10
  store i32 %9, i32* %argcount_310, align 4, !dbg !34
  %10 = load i32, i32* %argcount_310, align 4, !dbg !36
  call void @llvm.dbg.value(metadata i32 %10, metadata !35, metadata !DIExpression()), !dbg !10
  %11 = icmp ne i32 %10, 0, !dbg !36
  br i1 %11, label %L.LB1_384, label %L.LB1_548, !dbg !36

L.LB1_548:                                        ; preds = %L.LB1_419
  call void (...) @_mp_bcs_nest(), !dbg !37
  %12 = bitcast i32* @.C337_MAIN_ to i8*, !dbg !37
  %13 = bitcast [52 x i8]* @.C335_MAIN_ to i8*, !dbg !37
  %14 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !37
  call void (i8*, i8*, i64, ...) %14(i8* %12, i8* %13, i64 52), !dbg !37
  %15 = bitcast i32* @.C305_MAIN_ to i8*, !dbg !37
  %16 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !37
  %17 = bitcast [3 x i8]* @.C339_MAIN_ to i8*, !dbg !37
  %18 = bitcast i32 (...)* @f90io_encode_fmta to i32 (i8*, i8*, i8*, i64, ...)*, !dbg !37
  %19 = call i32 (i8*, i8*, i8*, i64, ...) %18(i8* %15, i8* %16, i8* %17, i64 3), !dbg !37
  call void @llvm.dbg.declare(metadata i32* %z__io_343, metadata !38, metadata !DIExpression()), !dbg !10
  store i32 %19, i32* %z__io_343, align 4, !dbg !37
  %20 = bitcast i32* @.C338_MAIN_ to i8*, !dbg !37
  %21 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !37
  %22 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !37
  %23 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !37
  %24 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %23(i8* %20, i8* null, i8* %21, i8* %22, i8* null, i8* null, i64 0), !dbg !37
  store i32 %24, i32* %z__io_343, align 4, !dbg !37
  %25 = bitcast i32* @.C305_MAIN_ to i8*, !dbg !37
  %26 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !37
  %27 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !37
  %28 = bitcast [35 x i8]* @.C341_MAIN_ to i8*, !dbg !37
  %29 = bitcast i32 (...)* @f90io_fmt_writea to i32 (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !37
  %30 = call i32 (i8*, i8*, i8*, i8*, i64, ...) %29(i8* %25, i8* %26, i8* %27, i8* %28, i64 35), !dbg !37
  store i32 %30, i32* %z__io_343, align 4, !dbg !37
  %31 = call i32 (...) @f90io_fmtw_end(), !dbg !37
  store i32 %31, i32* %z__io_343, align 4, !dbg !37
  call void (...) @_mp_ecs_nest(), !dbg !37
  br label %L.LB1_384

L.LB1_384:                                        ; preds = %L.LB1_548, %L.LB1_419
  call void @llvm.dbg.declare(metadata i64* %z_b_0_314, metadata !39, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_0_314, align 8, !dbg !40
  %32 = load i32, i32* %argcount_310, align 4, !dbg !40
  call void @llvm.dbg.value(metadata i32 %32, metadata !35, metadata !DIExpression()), !dbg !10
  %33 = sext i32 %32 to i64, !dbg !40
  call void @llvm.dbg.declare(metadata i64* %z_b_1_315, metadata !39, metadata !DIExpression()), !dbg !10
  store i64 %33, i64* %z_b_1_315, align 8, !dbg !40
  %34 = load i64, i64* %z_b_1_315, align 8, !dbg !40
  call void @llvm.dbg.value(metadata i64 %34, metadata !39, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_61_318, metadata !39, metadata !DIExpression()), !dbg !10
  store i64 %34, i64* %z_e_61_318, align 8, !dbg !40
  %35 = bitcast [16 x i64]* %"args$sd1_374" to i8*, !dbg !40
  %36 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !40
  %37 = bitcast i64* @.C375_MAIN_ to i8*, !dbg !40
  %38 = bitcast i64* @.C376_MAIN_ to i8*, !dbg !40
  %39 = bitcast i64* %z_b_0_314 to i8*, !dbg !40
  %40 = bitcast i64* %z_b_1_315 to i8*, !dbg !40
  %41 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !40
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %41(i8* %35, i8* %36, i8* %37, i8* %38, i8* %39, i8* %40), !dbg !40
  %42 = bitcast [16 x i64]* %"args$sd1_374" to i8*, !dbg !40
  %43 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !40
  call void (i8*, i32, ...) %43(i8* %42, i32 14), !dbg !40
  %44 = load i64, i64* %z_b_1_315, align 8, !dbg !40
  call void @llvm.dbg.value(metadata i64 %44, metadata !39, metadata !DIExpression()), !dbg !10
  %45 = load i64, i64* %z_b_0_314, align 8, !dbg !40
  call void @llvm.dbg.value(metadata i64 %45, metadata !39, metadata !DIExpression()), !dbg !10
  %46 = sub nsw i64 %45, 1, !dbg !40
  %47 = sub nsw i64 %44, %46, !dbg !40
  call void @llvm.dbg.declare(metadata i64* %z_b_2_316, metadata !39, metadata !DIExpression()), !dbg !10
  store i64 %47, i64* %z_b_2_316, align 8, !dbg !40
  %48 = load i64, i64* %z_b_0_314, align 8, !dbg !40
  call void @llvm.dbg.value(metadata i64 %48, metadata !39, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_b_3_317, metadata !39, metadata !DIExpression()), !dbg !10
  store i64 %48, i64* %z_b_3_317, align 8, !dbg !40
  %49 = bitcast i64* %z_b_2_316 to i8*, !dbg !40
  %50 = bitcast i64* @.C375_MAIN_ to i8*, !dbg !40
  %51 = bitcast i64* @.C376_MAIN_ to i8*, !dbg !40
  call void @llvm.dbg.declare(metadata i32* %allocstatus_311, metadata !41, metadata !DIExpression()), !dbg !10
  %52 = bitcast i32* %allocstatus_311 to i8*, !dbg !40
  %53 = bitcast [80 x i8]** %.Z0972_348 to i8*, !dbg !40
  %54 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !40
  %55 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !40
  %56 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !40
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %56(i8* %49, i8* %50, i8* %51, i8* %52, i8* %53, i8* null, i8* %54, i8* %55, i8* null, i64 0), !dbg !40
  %57 = load i32, i32* %allocstatus_311, align 4, !dbg !42
  call void @llvm.dbg.value(metadata i32 %57, metadata !41, metadata !DIExpression()), !dbg !10
  %58 = icmp sle i32 %57, 0, !dbg !42
  br i1 %58, label %L.LB1_387, label %L.LB1_549, !dbg !42

L.LB1_549:                                        ; preds = %L.LB1_384
  call void (...) @_mp_bcs_nest(), !dbg !43
  %59 = bitcast i32* @.C340_MAIN_ to i8*, !dbg !43
  %60 = bitcast [52 x i8]* @.C335_MAIN_ to i8*, !dbg !43
  %61 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !43
  call void (i8*, i8*, i64, ...) %61(i8* %59, i8* %60, i64 52), !dbg !43
  %62 = bitcast i32* @.C305_MAIN_ to i8*, !dbg !43
  %63 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !43
  %64 = bitcast [3 x i8]* @.C339_MAIN_ to i8*, !dbg !43
  %65 = bitcast i32 (...)* @f90io_encode_fmta to i32 (i8*, i8*, i8*, i64, ...)*, !dbg !43
  %66 = call i32 (i8*, i8*, i8*, i64, ...) %65(i8* %62, i8* %63, i8* %64, i64 3), !dbg !43
  store i32 %66, i32* %z__io_343, align 4, !dbg !43
  %67 = bitcast i32* @.C338_MAIN_ to i8*, !dbg !43
  %68 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !43
  %69 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !43
  %70 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !43
  %71 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %70(i8* %67, i8* null, i8* %68, i8* %69, i8* null, i8* null, i64 0), !dbg !43
  store i32 %71, i32* %z__io_343, align 4, !dbg !43
  %72 = bitcast i32* @.C305_MAIN_ to i8*, !dbg !43
  %73 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !43
  %74 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !43
  %75 = bitcast [37 x i8]* @.C349_MAIN_ to i8*, !dbg !43
  %76 = bitcast i32 (...)* @f90io_fmt_writea to i32 (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !43
  %77 = call i32 (i8*, i8*, i8*, i8*, i64, ...) %76(i8* %72, i8* %73, i8* %74, i8* %75, i64 37), !dbg !43
  store i32 %77, i32* %z__io_343, align 4, !dbg !43
  %78 = call i32 (...) @f90io_fmtw_end(), !dbg !43
  store i32 %78, i32* %z__io_343, align 4, !dbg !43
  call void (...) @_mp_ecs_nest(), !dbg !43
  %79 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !44
  %80 = bitcast void (...)* @f90_stop08a to void (i8*, i8*, i64, ...)*, !dbg !44
  call void (i8*, i8*, i64, ...) %80(i8* %79, i8* null, i64 0), !dbg !44
  br label %L.LB1_387

L.LB1_387:                                        ; preds = %L.LB1_549, %L.LB1_384
  %81 = load i32, i32* %argcount_310, align 4, !dbg !45
  call void @llvm.dbg.value(metadata i32 %81, metadata !35, metadata !DIExpression()), !dbg !10
  store i32 %81, i32* %.dY0001_390, align 4, !dbg !45
  call void @llvm.dbg.declare(metadata i32* %ix_313, metadata !46, metadata !DIExpression()), !dbg !10
  store i32 1, i32* %ix_313, align 4, !dbg !45
  %82 = load i32, i32* %.dY0001_390, align 4, !dbg !45
  %83 = icmp sle i32 %82, 0, !dbg !45
  br i1 %83, label %L.LB1_389, label %L.LB1_388, !dbg !45

L.LB1_388:                                        ; preds = %L.LB1_388, %L.LB1_387
  %84 = bitcast i32* %ix_313 to i8*, !dbg !47
  %85 = load [80 x i8]*, [80 x i8]** %.Z0972_348, align 8, !dbg !47
  call void @llvm.dbg.value(metadata [80 x i8]* %85, metadata !27, metadata !DIExpression()), !dbg !10
  %86 = bitcast [80 x i8]* %85 to i8*, !dbg !47
  %87 = getelementptr i8, i8* %86, i64 -80, !dbg !47
  %88 = load i32, i32* %ix_313, align 4, !dbg !47
  call void @llvm.dbg.value(metadata i32 %88, metadata !46, metadata !DIExpression()), !dbg !10
  %89 = sext i32 %88 to i64, !dbg !47
  %90 = bitcast [16 x i64]* %"args$sd1_374" to i8*, !dbg !47
  %91 = getelementptr i8, i8* %90, i64 56, !dbg !47
  %92 = bitcast i8* %91 to i64*, !dbg !47
  %93 = load i64, i64* %92, align 8, !dbg !47
  %94 = add nsw i64 %89, %93, !dbg !47
  %95 = mul nsw i64 %94, 80, !dbg !47
  %96 = getelementptr i8, i8* %87, i64 %95, !dbg !47
  %97 = bitcast i64* @.C372_MAIN_ to i8*, !dbg !47
  %98 = bitcast void (...)* @f90_get_cmd_arga to void (i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !47
  call void (i8*, i8*, i8*, i8*, i8*, i64, ...) %98(i8* %84, i8* %96, i8* null, i8* null, i8* %97, i64 80), !dbg !47
  %99 = load i32, i32* %ix_313, align 4, !dbg !48
  call void @llvm.dbg.value(metadata i32 %99, metadata !46, metadata !DIExpression()), !dbg !10
  %100 = add nsw i32 %99, 1, !dbg !48
  store i32 %100, i32* %ix_313, align 4, !dbg !48
  %101 = load i32, i32* %.dY0001_390, align 4, !dbg !48
  %102 = sub nsw i32 %101, 1, !dbg !48
  store i32 %102, i32* %.dY0001_390, align 4, !dbg !48
  %103 = load i32, i32* %.dY0001_390, align 4, !dbg !48
  %104 = icmp sgt i32 %103, 0, !dbg !48
  br i1 %104, label %L.LB1_388, label %L.LB1_389, !dbg !48

L.LB1_389:                                        ; preds = %L.LB1_388, %L.LB1_387
  %105 = load i32, i32* %argcount_310, align 4, !dbg !49
  call void @llvm.dbg.value(metadata i32 %105, metadata !35, metadata !DIExpression()), !dbg !10
  %106 = icmp sle i32 %105, 0, !dbg !49
  br i1 %106, label %L.LB1_391, label %L.LB1_550, !dbg !49

L.LB1_550:                                        ; preds = %L.LB1_389
  call void (...) @_mp_bcs_nest(), !dbg !50
  %107 = bitcast i32* @.C351_MAIN_ to i8*, !dbg !50
  %108 = bitcast [52 x i8]* @.C335_MAIN_ to i8*, !dbg !50
  %109 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !50
  call void (i8*, i8*, i64, ...) %109(i8* %107, i8* %108, i64 52), !dbg !50
  %110 = bitcast i32* @.C305_MAIN_ to i8*, !dbg !50
  %111 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !50
  %112 = bitcast [5 x i8]* @.C352_MAIN_ to i8*, !dbg !50
  %113 = bitcast i32 (...)* @f90io_encode_fmta to i32 (i8*, i8*, i8*, i64, ...)*, !dbg !50
  %114 = call i32 (i8*, i8*, i8*, i64, ...) %113(i8* %110, i8* %111, i8* %112, i64 5), !dbg !50
  store i32 %114, i32* %z__io_343, align 4, !dbg !50
  %115 = load [80 x i8]*, [80 x i8]** %.Z0972_348, align 8, !dbg !50
  call void @llvm.dbg.value(metadata [80 x i8]* %115, metadata !27, metadata !DIExpression()), !dbg !10
  %116 = bitcast [80 x i8]* %115 to i8*, !dbg !50
  %117 = bitcast [16 x i64]* %"args$sd1_374" to i8*, !dbg !50
  %118 = getelementptr i8, i8* %117, i64 56, !dbg !50
  %119 = bitcast i8* %118 to i64*, !dbg !50
  %120 = load i64, i64* %119, align 8, !dbg !50
  %121 = mul nsw i64 %120, 80, !dbg !50
  %122 = getelementptr i8, i8* %116, i64 %121, !dbg !50
  %123 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !50
  %124 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !50
  call void @llvm.dbg.declare(metadata i32* %rderr_312, metadata !51, metadata !DIExpression()), !dbg !10
  %125 = bitcast i32* %rderr_312 to i8*, !dbg !50
  %126 = bitcast i32 (...)* @f90io_fmtr_intern_inita to i32 (i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !50
  %127 = call i32 (i8*, i8*, i8*, i8*, i8*, i64, ...) %126(i8* %122, i8* %123, i8* %124, i8* %125, i8* null, i64 80), !dbg !50
  store i32 %127, i32* %z__io_343, align 4, !dbg !50
  %128 = bitcast i32* @.C306_MAIN_ to i8*, !dbg !50
  %129 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !50
  %130 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !50
  %131 = bitcast i32* %inlen_308 to i8*, !dbg !50
  %132 = bitcast i32 (...)* @f90io_fmt_reada to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !50
  %133 = call i32 (i8*, i8*, i8*, i8*, ...) %132(i8* %128, i8* %129, i8* %130, i8* %131), !dbg !50
  store i32 %133, i32* %z__io_343, align 4, !dbg !50
  %134 = call i32 (...) @f90io_fmtr_end(), !dbg !50
  store i32 %134, i32* %z__io_343, align 4, !dbg !50
  call void (...) @_mp_ecs_nest(), !dbg !50
  %135 = load i32, i32* %rderr_312, align 4, !dbg !52
  call void @llvm.dbg.value(metadata i32 %135, metadata !51, metadata !DIExpression()), !dbg !10
  %136 = icmp eq i32 %135, 0, !dbg !52
  br i1 %136, label %L.LB1_392, label %L.LB1_551, !dbg !52

L.LB1_551:                                        ; preds = %L.LB1_550
  call void (...) @_mp_bcs_nest(), !dbg !53
  %137 = bitcast i32* @.C356_MAIN_ to i8*, !dbg !53
  %138 = bitcast [52 x i8]* @.C335_MAIN_ to i8*, !dbg !53
  %139 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !53
  call void (i8*, i8*, i64, ...) %139(i8* %137, i8* %138, i64 52), !dbg !53
  %140 = bitcast i32* @.C305_MAIN_ to i8*, !dbg !53
  %141 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !53
  %142 = bitcast [3 x i8]* @.C339_MAIN_ to i8*, !dbg !53
  %143 = bitcast i32 (...)* @f90io_encode_fmta to i32 (i8*, i8*, i8*, i64, ...)*, !dbg !53
  %144 = call i32 (i8*, i8*, i8*, i64, ...) %143(i8* %140, i8* %141, i8* %142, i64 3), !dbg !53
  store i32 %144, i32* %z__io_343, align 4, !dbg !53
  %145 = bitcast i32* @.C338_MAIN_ to i8*, !dbg !53
  %146 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !53
  %147 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !53
  %148 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !53
  %149 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %148(i8* %145, i8* null, i8* %146, i8* %147, i8* null, i8* null, i64 0), !dbg !53
  store i32 %149, i32* %z__io_343, align 4, !dbg !53
  %150 = bitcast i32* @.C305_MAIN_ to i8*, !dbg !53
  %151 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !53
  %152 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !53
  %153 = bitcast [29 x i8]* @.C357_MAIN_ to i8*, !dbg !53
  %154 = bitcast i32 (...)* @f90io_fmt_writea to i32 (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !53
  %155 = call i32 (i8*, i8*, i8*, i8*, i64, ...) %154(i8* %150, i8* %151, i8* %152, i8* %153, i64 29), !dbg !53
  store i32 %155, i32* %z__io_343, align 4, !dbg !53
  %156 = call i32 (...) @f90io_fmtw_end(), !dbg !53
  store i32 %156, i32* %z__io_343, align 4, !dbg !53
  call void (...) @_mp_ecs_nest(), !dbg !53
  br label %L.LB1_392

L.LB1_392:                                        ; preds = %L.LB1_551, %L.LB1_550
  br label %L.LB1_391

L.LB1_391:                                        ; preds = %L.LB1_392, %L.LB1_389
  call void @llvm.dbg.declare(metadata i64* %z_b_4_320, metadata !39, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_4_320, align 8, !dbg !54
  %157 = load i32, i32* %inlen_308, align 4, !dbg !54
  call void @llvm.dbg.value(metadata i32 %157, metadata !30, metadata !DIExpression()), !dbg !10
  %158 = sext i32 %157 to i64, !dbg !54
  call void @llvm.dbg.declare(metadata i64* %z_b_5_321, metadata !39, metadata !DIExpression()), !dbg !10
  store i64 %158, i64* %z_b_5_321, align 8, !dbg !54
  %159 = load i64, i64* %z_b_5_321, align 8, !dbg !54
  call void @llvm.dbg.value(metadata i64 %159, metadata !39, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_68_324, metadata !39, metadata !DIExpression()), !dbg !10
  store i64 %159, i64* %z_e_68_324, align 8, !dbg !54
  %160 = bitcast [16 x i64]* %"input$sd2_378" to i8*, !dbg !54
  %161 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !54
  %162 = bitcast i64* @.C379_MAIN_ to i8*, !dbg !54
  %163 = bitcast i64* @.C372_MAIN_ to i8*, !dbg !54
  %164 = bitcast i64* %z_b_4_320 to i8*, !dbg !54
  %165 = bitcast i64* %z_b_5_321 to i8*, !dbg !54
  %166 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !54
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %166(i8* %160, i8* %161, i8* %162, i8* %163, i8* %164, i8* %165), !dbg !54
  %167 = bitcast [16 x i64]* %"input$sd2_378" to i8*, !dbg !54
  %168 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !54
  call void (i8*, i32, ...) %168(i8* %167, i32 25), !dbg !54
  %169 = load i64, i64* %z_b_5_321, align 8, !dbg !54
  call void @llvm.dbg.value(metadata i64 %169, metadata !39, metadata !DIExpression()), !dbg !10
  %170 = load i64, i64* %z_b_4_320, align 8, !dbg !54
  call void @llvm.dbg.value(metadata i64 %170, metadata !39, metadata !DIExpression()), !dbg !10
  %171 = sub nsw i64 %170, 1, !dbg !54
  %172 = sub nsw i64 %169, %171, !dbg !54
  call void @llvm.dbg.declare(metadata i64* %z_b_6_322, metadata !39, metadata !DIExpression()), !dbg !10
  store i64 %172, i64* %z_b_6_322, align 8, !dbg !54
  %173 = load i64, i64* %z_b_4_320, align 8, !dbg !54
  call void @llvm.dbg.value(metadata i64 %173, metadata !39, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_b_7_323, metadata !39, metadata !DIExpression()), !dbg !10
  store i64 %173, i64* %z_b_7_323, align 8, !dbg !54
  %174 = bitcast i64* %z_b_6_322 to i8*, !dbg !54
  %175 = bitcast i64* @.C379_MAIN_ to i8*, !dbg !54
  %176 = bitcast i64* @.C372_MAIN_ to i8*, !dbg !54
  %177 = bitcast i32** %.Z0978_358 to i8*, !dbg !54
  %178 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !54
  %179 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !54
  %180 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !54
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %180(i8* %174, i8* %175, i8* %176, i8* null, i8* %177, i8* null, i8* %178, i8* %179, i8* null, i64 0), !dbg !54
  call void @llvm.dbg.declare(metadata i64* %z_b_8_326, metadata !39, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_8_326, align 8, !dbg !55
  %181 = load i32, i32* %inlen_308, align 4, !dbg !55
  call void @llvm.dbg.value(metadata i32 %181, metadata !30, metadata !DIExpression()), !dbg !10
  %182 = sext i32 %181 to i64, !dbg !55
  call void @llvm.dbg.declare(metadata i64* %z_b_9_327, metadata !39, metadata !DIExpression()), !dbg !10
  store i64 %182, i64* %z_b_9_327, align 8, !dbg !55
  %183 = load i64, i64* %z_b_9_327, align 8, !dbg !55
  call void @llvm.dbg.value(metadata i64 %183, metadata !39, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_75_330, metadata !39, metadata !DIExpression()), !dbg !10
  store i64 %183, i64* %z_e_75_330, align 8, !dbg !55
  %184 = bitcast [16 x i64]* %"output$sd3_380" to i8*, !dbg !55
  %185 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !55
  %186 = bitcast i64* @.C379_MAIN_ to i8*, !dbg !55
  %187 = bitcast i64* @.C372_MAIN_ to i8*, !dbg !55
  %188 = bitcast i64* %z_b_8_326 to i8*, !dbg !55
  %189 = bitcast i64* %z_b_9_327 to i8*, !dbg !55
  %190 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !55
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %190(i8* %184, i8* %185, i8* %186, i8* %187, i8* %188, i8* %189), !dbg !55
  %191 = bitcast [16 x i64]* %"output$sd3_380" to i8*, !dbg !55
  %192 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !55
  call void (i8*, i32, ...) %192(i8* %191, i32 25), !dbg !55
  %193 = load i64, i64* %z_b_9_327, align 8, !dbg !55
  call void @llvm.dbg.value(metadata i64 %193, metadata !39, metadata !DIExpression()), !dbg !10
  %194 = load i64, i64* %z_b_8_326, align 8, !dbg !55
  call void @llvm.dbg.value(metadata i64 %194, metadata !39, metadata !DIExpression()), !dbg !10
  %195 = sub nsw i64 %194, 1, !dbg !55
  %196 = sub nsw i64 %193, %195, !dbg !55
  call void @llvm.dbg.declare(metadata i64* %z_b_10_328, metadata !39, metadata !DIExpression()), !dbg !10
  store i64 %196, i64* %z_b_10_328, align 8, !dbg !55
  %197 = load i64, i64* %z_b_8_326, align 8, !dbg !55
  call void @llvm.dbg.value(metadata i64 %197, metadata !39, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_b_11_329, metadata !39, metadata !DIExpression()), !dbg !10
  store i64 %197, i64* %z_b_11_329, align 8, !dbg !55
  %198 = bitcast i64* %z_b_10_328 to i8*, !dbg !55
  %199 = bitcast i64* @.C379_MAIN_ to i8*, !dbg !55
  %200 = bitcast i64* @.C372_MAIN_ to i8*, !dbg !55
  %201 = bitcast i32** %.Z0984_359 to i8*, !dbg !55
  %202 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !55
  %203 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !55
  %204 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !55
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %204(i8* %198, i8* %199, i8* %200, i8* null, i8* %201, i8* null, i8* %202, i8* %203, i8* null, i64 0), !dbg !55
  %205 = load i32, i32* %inlen_308, align 4, !dbg !56
  call void @llvm.dbg.value(metadata i32 %205, metadata !30, metadata !DIExpression()), !dbg !10
  store i32 %205, i32* %.dY0002_395, align 4, !dbg !56
  call void @llvm.dbg.declare(metadata i32* %i_307, metadata !57, metadata !DIExpression()), !dbg !10
  store i32 1, i32* %i_307, align 4, !dbg !56
  %206 = load i32, i32* %.dY0002_395, align 4, !dbg !56
  %207 = icmp sle i32 %206, 0, !dbg !56
  br i1 %207, label %L.LB1_394, label %L.LB1_393, !dbg !56

L.LB1_393:                                        ; preds = %L.LB1_393, %L.LB1_391
  %208 = load i32, i32* %i_307, align 4, !dbg !58
  call void @llvm.dbg.value(metadata i32 %208, metadata !57, metadata !DIExpression()), !dbg !10
  %209 = load i32, i32* %i_307, align 4, !dbg !58
  call void @llvm.dbg.value(metadata i32 %209, metadata !57, metadata !DIExpression()), !dbg !10
  %210 = sext i32 %209 to i64, !dbg !58
  %211 = bitcast [16 x i64]* %"input$sd2_378" to i8*, !dbg !58
  %212 = getelementptr i8, i8* %211, i64 56, !dbg !58
  %213 = bitcast i8* %212 to i64*, !dbg !58
  %214 = load i64, i64* %213, align 8, !dbg !58
  %215 = add nsw i64 %210, %214, !dbg !58
  %216 = load i32*, i32** %.Z0978_358, align 8, !dbg !58
  call void @llvm.dbg.value(metadata i32* %216, metadata !26, metadata !DIExpression()), !dbg !10
  %217 = bitcast i32* %216 to i8*, !dbg !58
  %218 = getelementptr i8, i8* %217, i64 -4, !dbg !58
  %219 = bitcast i8* %218 to i32*, !dbg !58
  %220 = getelementptr i32, i32* %219, i64 %215, !dbg !58
  store i32 %208, i32* %220, align 4, !dbg !58
  %221 = load i32, i32* %i_307, align 4, !dbg !59
  call void @llvm.dbg.value(metadata i32 %221, metadata !57, metadata !DIExpression()), !dbg !10
  %222 = add nsw i32 %221, 1, !dbg !59
  store i32 %222, i32* %i_307, align 4, !dbg !59
  %223 = load i32, i32* %.dY0002_395, align 4, !dbg !59
  %224 = sub nsw i32 %223, 1, !dbg !59
  store i32 %224, i32* %.dY0002_395, align 4, !dbg !59
  %225 = load i32, i32* %.dY0002_395, align 4, !dbg !59
  %226 = icmp sgt i32 %225, 0, !dbg !59
  br i1 %226, label %L.LB1_393, label %L.LB1_394, !dbg !59

L.LB1_394:                                        ; preds = %L.LB1_393, %L.LB1_391
  %227 = bitcast i32* %inlen_308 to i8*, !dbg !60
  %228 = bitcast %astruct.dt96* %.uplevelArgPack0001_488 to i8**, !dbg !60
  store i8* %227, i8** %228, align 8, !dbg !60
  %229 = bitcast i32** %.Z0984_359 to i8*, !dbg !60
  %230 = bitcast %astruct.dt96* %.uplevelArgPack0001_488 to i8*, !dbg !60
  %231 = getelementptr i8, i8* %230, i64 8, !dbg !60
  %232 = bitcast i8* %231 to i8**, !dbg !60
  store i8* %229, i8** %232, align 8, !dbg !60
  %233 = bitcast i32** %.Z0984_359 to i8*, !dbg !60
  %234 = bitcast %astruct.dt96* %.uplevelArgPack0001_488 to i8*, !dbg !60
  %235 = getelementptr i8, i8* %234, i64 16, !dbg !60
  %236 = bitcast i8* %235 to i8**, !dbg !60
  store i8* %233, i8** %236, align 8, !dbg !60
  %237 = bitcast i64* %z_b_8_326 to i8*, !dbg !60
  %238 = bitcast %astruct.dt96* %.uplevelArgPack0001_488 to i8*, !dbg !60
  %239 = getelementptr i8, i8* %238, i64 24, !dbg !60
  %240 = bitcast i8* %239 to i8**, !dbg !60
  store i8* %237, i8** %240, align 8, !dbg !60
  %241 = bitcast i64* %z_b_9_327 to i8*, !dbg !60
  %242 = bitcast %astruct.dt96* %.uplevelArgPack0001_488 to i8*, !dbg !60
  %243 = getelementptr i8, i8* %242, i64 32, !dbg !60
  %244 = bitcast i8* %243 to i8**, !dbg !60
  store i8* %241, i8** %244, align 8, !dbg !60
  %245 = bitcast i64* %z_e_75_330 to i8*, !dbg !60
  %246 = bitcast %astruct.dt96* %.uplevelArgPack0001_488 to i8*, !dbg !60
  %247 = getelementptr i8, i8* %246, i64 40, !dbg !60
  %248 = bitcast i8* %247 to i8**, !dbg !60
  store i8* %245, i8** %248, align 8, !dbg !60
  %249 = bitcast i64* %z_b_10_328 to i8*, !dbg !60
  %250 = bitcast %astruct.dt96* %.uplevelArgPack0001_488 to i8*, !dbg !60
  %251 = getelementptr i8, i8* %250, i64 48, !dbg !60
  %252 = bitcast i8* %251 to i8**, !dbg !60
  store i8* %249, i8** %252, align 8, !dbg !60
  %253 = bitcast i64* %z_b_11_329 to i8*, !dbg !60
  %254 = bitcast %astruct.dt96* %.uplevelArgPack0001_488 to i8*, !dbg !60
  %255 = getelementptr i8, i8* %254, i64 56, !dbg !60
  %256 = bitcast i8* %255 to i8**, !dbg !60
  store i8* %253, i8** %256, align 8, !dbg !60
  %257 = bitcast i32* %outlen_309 to i8*, !dbg !60
  %258 = bitcast %astruct.dt96* %.uplevelArgPack0001_488 to i8*, !dbg !60
  %259 = getelementptr i8, i8* %258, i64 64, !dbg !60
  %260 = bitcast i8* %259 to i8**, !dbg !60
  store i8* %257, i8** %260, align 8, !dbg !60
  %261 = bitcast i32** %.Z0978_358 to i8*, !dbg !60
  %262 = bitcast %astruct.dt96* %.uplevelArgPack0001_488 to i8*, !dbg !60
  %263 = getelementptr i8, i8* %262, i64 72, !dbg !60
  %264 = bitcast i8* %263 to i8**, !dbg !60
  store i8* %261, i8** %264, align 8, !dbg !60
  %265 = bitcast i32** %.Z0978_358 to i8*, !dbg !60
  %266 = bitcast %astruct.dt96* %.uplevelArgPack0001_488 to i8*, !dbg !60
  %267 = getelementptr i8, i8* %266, i64 80, !dbg !60
  %268 = bitcast i8* %267 to i8**, !dbg !60
  store i8* %265, i8** %268, align 8, !dbg !60
  %269 = bitcast i64* %z_b_4_320 to i8*, !dbg !60
  %270 = bitcast %astruct.dt96* %.uplevelArgPack0001_488 to i8*, !dbg !60
  %271 = getelementptr i8, i8* %270, i64 88, !dbg !60
  %272 = bitcast i8* %271 to i8**, !dbg !60
  store i8* %269, i8** %272, align 8, !dbg !60
  %273 = bitcast i64* %z_b_5_321 to i8*, !dbg !60
  %274 = bitcast %astruct.dt96* %.uplevelArgPack0001_488 to i8*, !dbg !60
  %275 = getelementptr i8, i8* %274, i64 96, !dbg !60
  %276 = bitcast i8* %275 to i8**, !dbg !60
  store i8* %273, i8** %276, align 8, !dbg !60
  %277 = bitcast i64* %z_e_68_324 to i8*, !dbg !60
  %278 = bitcast %astruct.dt96* %.uplevelArgPack0001_488 to i8*, !dbg !60
  %279 = getelementptr i8, i8* %278, i64 104, !dbg !60
  %280 = bitcast i8* %279 to i8**, !dbg !60
  store i8* %277, i8** %280, align 8, !dbg !60
  %281 = bitcast i64* %z_b_6_322 to i8*, !dbg !60
  %282 = bitcast %astruct.dt96* %.uplevelArgPack0001_488 to i8*, !dbg !60
  %283 = getelementptr i8, i8* %282, i64 112, !dbg !60
  %284 = bitcast i8* %283 to i8**, !dbg !60
  store i8* %281, i8** %284, align 8, !dbg !60
  %285 = bitcast i64* %z_b_7_323 to i8*, !dbg !60
  %286 = bitcast %astruct.dt96* %.uplevelArgPack0001_488 to i8*, !dbg !60
  %287 = getelementptr i8, i8* %286, i64 120, !dbg !60
  %288 = bitcast i8* %287 to i8**, !dbg !60
  store i8* %285, i8** %288, align 8, !dbg !60
  %289 = bitcast [16 x i64]* %"input$sd2_378" to i8*, !dbg !60
  %290 = bitcast %astruct.dt96* %.uplevelArgPack0001_488 to i8*, !dbg !60
  %291 = getelementptr i8, i8* %290, i64 128, !dbg !60
  %292 = bitcast i8* %291 to i8**, !dbg !60
  store i8* %289, i8** %292, align 8, !dbg !60
  %293 = bitcast [16 x i64]* %"output$sd3_380" to i8*, !dbg !60
  %294 = bitcast %astruct.dt96* %.uplevelArgPack0001_488 to i8*, !dbg !60
  %295 = getelementptr i8, i8* %294, i64 136, !dbg !60
  %296 = bitcast i8* %295 to i8**, !dbg !60
  store i8* %293, i8** %296, align 8, !dbg !60
  br label %L.LB1_525, !dbg !60

L.LB1_525:                                        ; preds = %L.LB1_394
  %297 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L57_1_ to i64*, !dbg !60
  %298 = bitcast %astruct.dt96* %.uplevelArgPack0001_488 to i64*, !dbg !60
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %297, i64* %298), !dbg !60
  call void (...) @_mp_bcs_nest(), !dbg !61
  %299 = bitcast i32* @.C366_MAIN_ to i8*, !dbg !61
  %300 = bitcast [52 x i8]* @.C335_MAIN_ to i8*, !dbg !61
  %301 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !61
  call void (i8*, i8*, i64, ...) %301(i8* %299, i8* %300, i64 52), !dbg !61
  %302 = bitcast i32* @.C338_MAIN_ to i8*, !dbg !61
  %303 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !61
  %304 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !61
  %305 = bitcast %struct.STATICS1* @.STATICS1 to i8*, !dbg !61
  %306 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !61
  %307 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %306(i8* %302, i8* null, i8* %303, i8* %304, i8* %305, i8* null, i64 0), !dbg !61
  store i32 %307, i32* %z__io_343, align 4, !dbg !61
  %308 = bitcast [16 x i64]* %"output$sd3_380" to i8*, !dbg !61
  %309 = getelementptr i8, i8* %308, i64 56, !dbg !61
  %310 = bitcast i8* %309 to i64*, !dbg !61
  %311 = load i64, i64* %310, align 8, !dbg !61
  %312 = load i32*, i32** %.Z0984_359, align 8, !dbg !61
  call void @llvm.dbg.value(metadata i32* %312, metadata !17, metadata !DIExpression()), !dbg !10
  %313 = bitcast i32* %312 to i8*, !dbg !61
  %314 = getelementptr i8, i8* %313, i64 -4, !dbg !61
  %315 = bitcast i8* %314 to i32*, !dbg !61
  %316 = getelementptr i32, i32* %315, i64 %311, !dbg !61
  %317 = load i32, i32* %316, align 4, !dbg !61
  %318 = bitcast i32 (...)* @f90io_sc_i_fmt_write to i32 (i32, i32, ...)*, !dbg !61
  %319 = call i32 (i32, i32, ...) %318(i32 %317, i32 25), !dbg !61
  store i32 %319, i32* %z__io_343, align 4, !dbg !61
  %320 = call i32 (...) @f90io_fmtw_end(), !dbg !61
  store i32 %320, i32* %z__io_343, align 4, !dbg !61
  call void (...) @_mp_ecs_nest(), !dbg !61
  %321 = load i32*, i32** %.Z0978_358, align 8, !dbg !62
  call void @llvm.dbg.value(metadata i32* %321, metadata !26, metadata !DIExpression()), !dbg !10
  %322 = bitcast i32* %321 to i8*, !dbg !62
  %323 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !62
  %324 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !62
  call void (i8*, i8*, i8*, i8*, i64, ...) %324(i8* null, i8* %322, i8* %323, i8* null, i64 0), !dbg !62
  %325 = bitcast i32** %.Z0978_358 to i8**, !dbg !62
  store i8* null, i8** %325, align 8, !dbg !62
  %326 = bitcast [16 x i64]* %"input$sd2_378" to i64*, !dbg !62
  store i64 0, i64* %326, align 8, !dbg !62
  %327 = load i32*, i32** %.Z0984_359, align 8, !dbg !62
  call void @llvm.dbg.value(metadata i32* %327, metadata !17, metadata !DIExpression()), !dbg !10
  %328 = bitcast i32* %327 to i8*, !dbg !62
  %329 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !62
  %330 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !62
  call void (i8*, i8*, i8*, i8*, i64, ...) %330(i8* null, i8* %328, i8* %329, i8* null, i64 0), !dbg !62
  %331 = bitcast i32** %.Z0984_359 to i8**, !dbg !62
  store i8* null, i8** %331, align 8, !dbg !62
  %332 = bitcast [16 x i64]* %"output$sd3_380" to i64*, !dbg !62
  store i64 0, i64* %332, align 8, !dbg !62
  %333 = load [80 x i8]*, [80 x i8]** %.Z0972_348, align 8, !dbg !62
  call void @llvm.dbg.value(metadata [80 x i8]* %333, metadata !27, metadata !DIExpression()), !dbg !10
  %334 = bitcast [80 x i8]* %333 to i8*, !dbg !62
  %335 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !62
  %336 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, i64, ...)*, !dbg !62
  call void (i8*, i8*, i8*, i8*, i64, i64, ...) %336(i8* null, i8* %334, i8* %335, i8* null, i64 80, i64 0), !dbg !62
  %337 = bitcast [80 x i8]** %.Z0972_348 to i8**, !dbg !62
  store i8* null, i8** %337, align 8, !dbg !62
  %338 = bitcast [16 x i64]* %"args$sd1_374" to i64*, !dbg !62
  store i64 0, i64* %338, align 8, !dbg !62
  ret void, !dbg !15
}

define internal void @__nv_MAIN__F1L57_1_(i32* %__nv_MAIN__F1L57_1Arg0, i64* %__nv_MAIN__F1L57_1Arg1, i64* %__nv_MAIN__F1L57_1Arg2) #0 !dbg !63 {
L.entry:
  %__gtid___nv_MAIN__F1L57_1__570 = alloca i32, align 4
  %.i0000p_364 = alloca i32, align 4
  %i_363 = alloca i32, align 4
  %.du0003p_399 = alloca i32, align 4
  %.de0003p_400 = alloca i32, align 4
  %.di0003p_401 = alloca i32, align 4
  %.ds0003p_402 = alloca i32, align 4
  %.dl0003p_404 = alloca i32, align 4
  %.dl0003p.copy_564 = alloca i32, align 4
  %.de0003p.copy_565 = alloca i32, align 4
  %.ds0003p.copy_566 = alloca i32, align 4
  %.dX0003p_403 = alloca i32, align 4
  %.dY0003p_398 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L57_1Arg0, metadata !66, metadata !DIExpression()), !dbg !67
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L57_1Arg1, metadata !68, metadata !DIExpression()), !dbg !67
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L57_1Arg2, metadata !69, metadata !DIExpression()), !dbg !67
  call void @llvm.dbg.value(metadata i32 1, metadata !70, metadata !DIExpression()), !dbg !67
  call void @llvm.dbg.value(metadata i32 0, metadata !71, metadata !DIExpression()), !dbg !67
  call void @llvm.dbg.value(metadata i32 1, metadata !72, metadata !DIExpression()), !dbg !67
  call void @llvm.dbg.value(metadata i32 0, metadata !73, metadata !DIExpression()), !dbg !67
  call void @llvm.dbg.value(metadata i32 1, metadata !74, metadata !DIExpression()), !dbg !67
  %0 = load i32, i32* %__nv_MAIN__F1L57_1Arg0, align 4, !dbg !75
  store i32 %0, i32* %__gtid___nv_MAIN__F1L57_1__570, align 4, !dbg !75
  br label %L.LB2_555

L.LB2_555:                                        ; preds = %L.entry
  br label %L.LB2_362

L.LB2_362:                                        ; preds = %L.LB2_555
  store i32 0, i32* %.i0000p_364, align 4, !dbg !76
  call void @llvm.dbg.declare(metadata i32* %i_363, metadata !77, metadata !DIExpression()), !dbg !75
  store i32 1, i32* %i_363, align 4, !dbg !76
  %1 = bitcast i64* %__nv_MAIN__F1L57_1Arg2 to i32**, !dbg !76
  %2 = load i32*, i32** %1, align 8, !dbg !76
  %3 = load i32, i32* %2, align 4, !dbg !76
  store i32 %3, i32* %.du0003p_399, align 4, !dbg !76
  %4 = bitcast i64* %__nv_MAIN__F1L57_1Arg2 to i32**, !dbg !76
  %5 = load i32*, i32** %4, align 8, !dbg !76
  %6 = load i32, i32* %5, align 4, !dbg !76
  store i32 %6, i32* %.de0003p_400, align 4, !dbg !76
  store i32 1, i32* %.di0003p_401, align 4, !dbg !76
  %7 = load i32, i32* %.di0003p_401, align 4, !dbg !76
  store i32 %7, i32* %.ds0003p_402, align 4, !dbg !76
  store i32 1, i32* %.dl0003p_404, align 4, !dbg !76
  %8 = load i32, i32* %.dl0003p_404, align 4, !dbg !76
  store i32 %8, i32* %.dl0003p.copy_564, align 4, !dbg !76
  %9 = load i32, i32* %.de0003p_400, align 4, !dbg !76
  store i32 %9, i32* %.de0003p.copy_565, align 4, !dbg !76
  %10 = load i32, i32* %.ds0003p_402, align 4, !dbg !76
  store i32 %10, i32* %.ds0003p.copy_566, align 4, !dbg !76
  %11 = load i32, i32* %__gtid___nv_MAIN__F1L57_1__570, align 4, !dbg !76
  %12 = bitcast i32* %.i0000p_364 to i64*, !dbg !76
  %13 = bitcast i32* %.dl0003p.copy_564 to i64*, !dbg !76
  %14 = bitcast i32* %.de0003p.copy_565 to i64*, !dbg !76
  %15 = bitcast i32* %.ds0003p.copy_566 to i64*, !dbg !76
  %16 = load i32, i32* %.ds0003p.copy_566, align 4, !dbg !76
  call void @__kmpc_for_static_init_4(i64* null, i32 %11, i32 34, i64* %12, i64* %13, i64* %14, i64* %15, i32 %16, i32 1), !dbg !76
  %17 = load i32, i32* %.dl0003p.copy_564, align 4, !dbg !76
  store i32 %17, i32* %.dl0003p_404, align 4, !dbg !76
  %18 = load i32, i32* %.de0003p.copy_565, align 4, !dbg !76
  store i32 %18, i32* %.de0003p_400, align 4, !dbg !76
  %19 = load i32, i32* %.ds0003p.copy_566, align 4, !dbg !76
  store i32 %19, i32* %.ds0003p_402, align 4, !dbg !76
  %20 = load i32, i32* %.dl0003p_404, align 4, !dbg !76
  store i32 %20, i32* %i_363, align 4, !dbg !76
  %21 = load i32, i32* %i_363, align 4, !dbg !76
  call void @llvm.dbg.value(metadata i32 %21, metadata !77, metadata !DIExpression()), !dbg !75
  store i32 %21, i32* %.dX0003p_403, align 4, !dbg !76
  %22 = load i32, i32* %.dX0003p_403, align 4, !dbg !76
  %23 = load i32, i32* %.du0003p_399, align 4, !dbg !76
  %24 = icmp sgt i32 %22, %23, !dbg !76
  br i1 %24, label %L.LB2_397, label %L.LB2_596, !dbg !76

L.LB2_596:                                        ; preds = %L.LB2_362
  %25 = load i32, i32* %.dX0003p_403, align 4, !dbg !76
  store i32 %25, i32* %i_363, align 4, !dbg !76
  %26 = load i32, i32* %.di0003p_401, align 4, !dbg !76
  %27 = load i32, i32* %.de0003p_400, align 4, !dbg !76
  %28 = load i32, i32* %.dX0003p_403, align 4, !dbg !76
  %29 = sub nsw i32 %27, %28, !dbg !76
  %30 = add nsw i32 %26, %29, !dbg !76
  %31 = load i32, i32* %.di0003p_401, align 4, !dbg !76
  %32 = sdiv i32 %30, %31, !dbg !76
  store i32 %32, i32* %.dY0003p_398, align 4, !dbg !76
  %33 = load i32, i32* %.dY0003p_398, align 4, !dbg !76
  %34 = icmp sle i32 %33, 0, !dbg !76
  br i1 %34, label %L.LB2_407, label %L.LB2_406, !dbg !76

L.LB2_406:                                        ; preds = %L.LB2_406, %L.LB2_596
  %35 = load i32, i32* %i_363, align 4, !dbg !78
  call void @llvm.dbg.value(metadata i32 %35, metadata !77, metadata !DIExpression()), !dbg !75
  %36 = sext i32 %35 to i64, !dbg !78
  %37 = bitcast i64* %__nv_MAIN__F1L57_1Arg2 to i8*, !dbg !78
  %38 = getelementptr i8, i8* %37, i64 128, !dbg !78
  %39 = bitcast i8* %38 to i8**, !dbg !78
  %40 = load i8*, i8** %39, align 8, !dbg !78
  %41 = getelementptr i8, i8* %40, i64 56, !dbg !78
  %42 = bitcast i8* %41 to i64*, !dbg !78
  %43 = load i64, i64* %42, align 8, !dbg !78
  %44 = add nsw i64 %36, %43, !dbg !78
  %45 = bitcast i64* %__nv_MAIN__F1L57_1Arg2 to i8*, !dbg !78
  %46 = getelementptr i8, i8* %45, i64 80, !dbg !78
  %47 = bitcast i8* %46 to i8***, !dbg !78
  %48 = load i8**, i8*** %47, align 8, !dbg !78
  %49 = load i8*, i8** %48, align 8, !dbg !78
  %50 = getelementptr i8, i8* %49, i64 -4, !dbg !78
  %51 = bitcast i8* %50 to i32*, !dbg !78
  %52 = getelementptr i32, i32* %51, i64 %44, !dbg !78
  %53 = load i32, i32* %52, align 4, !dbg !78
  %54 = bitcast i64* %__nv_MAIN__F1L57_1Arg2 to i8*, !dbg !78
  %55 = getelementptr i8, i8* %54, i64 64, !dbg !78
  %56 = bitcast i8* %55 to i32**, !dbg !78
  %57 = load i32*, i32** %56, align 8, !dbg !78
  %58 = load i32, i32* %57, align 4, !dbg !78
  %59 = sext i32 %58 to i64, !dbg !78
  %60 = bitcast i64* %__nv_MAIN__F1L57_1Arg2 to i8*, !dbg !78
  %61 = getelementptr i8, i8* %60, i64 136, !dbg !78
  %62 = bitcast i8* %61 to i8**, !dbg !78
  %63 = load i8*, i8** %62, align 8, !dbg !78
  %64 = getelementptr i8, i8* %63, i64 56, !dbg !78
  %65 = bitcast i8* %64 to i64*, !dbg !78
  %66 = load i64, i64* %65, align 8, !dbg !78
  %67 = add nsw i64 %59, %66, !dbg !78
  %68 = bitcast i64* %__nv_MAIN__F1L57_1Arg2 to i8*, !dbg !78
  %69 = getelementptr i8, i8* %68, i64 16, !dbg !78
  %70 = bitcast i8* %69 to i8***, !dbg !78
  %71 = load i8**, i8*** %70, align 8, !dbg !78
  %72 = load i8*, i8** %71, align 8, !dbg !78
  %73 = getelementptr i8, i8* %72, i64 -4, !dbg !78
  %74 = bitcast i8* %73 to i32*, !dbg !78
  %75 = getelementptr i32, i32* %74, i64 %67, !dbg !78
  store i32 %53, i32* %75, align 4, !dbg !78
  %76 = bitcast i64* %__nv_MAIN__F1L57_1Arg2 to i8*, !dbg !79
  %77 = getelementptr i8, i8* %76, i64 64, !dbg !79
  %78 = bitcast i8* %77 to i32**, !dbg !79
  %79 = load i32*, i32** %78, align 8, !dbg !79
  %80 = load i32, i32* %79, align 4, !dbg !79
  %81 = add nsw i32 %80, 1, !dbg !79
  %82 = bitcast i64* %__nv_MAIN__F1L57_1Arg2 to i8*, !dbg !79
  %83 = getelementptr i8, i8* %82, i64 64, !dbg !79
  %84 = bitcast i8* %83 to i32**, !dbg !79
  %85 = load i32*, i32** %84, align 8, !dbg !79
  store i32 %81, i32* %85, align 4, !dbg !79
  %86 = load i32, i32* %.di0003p_401, align 4, !dbg !75
  %87 = load i32, i32* %i_363, align 4, !dbg !75
  call void @llvm.dbg.value(metadata i32 %87, metadata !77, metadata !DIExpression()), !dbg !75
  %88 = add nsw i32 %86, %87, !dbg !75
  store i32 %88, i32* %i_363, align 4, !dbg !75
  %89 = load i32, i32* %.dY0003p_398, align 4, !dbg !75
  %90 = sub nsw i32 %89, 1, !dbg !75
  store i32 %90, i32* %.dY0003p_398, align 4, !dbg !75
  %91 = load i32, i32* %.dY0003p_398, align 4, !dbg !75
  %92 = icmp sgt i32 %91, 0, !dbg !75
  br i1 %92, label %L.LB2_406, label %L.LB2_407, !dbg !75

L.LB2_407:                                        ; preds = %L.LB2_406, %L.LB2_596
  br label %L.LB2_397

L.LB2_397:                                        ; preds = %L.LB2_407, %L.LB2_362
  %93 = load i32, i32* %__gtid___nv_MAIN__F1L57_1__570, align 4, !dbg !75
  call void @__kmpc_for_static_fini(i64* null, i32 %93), !dbg !75
  br label %L.LB2_365

L.LB2_365:                                        ; preds = %L.LB2_397
  ret void, !dbg !75
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
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB019-plusplus-var-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb019_plusplus_var_yes", scope: !2, file: !3, line: 16, type: !6, scopeLine: 16, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
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
!16 = !DILocation(line: 16, column: 1, scope: !5)
!17 = !DILocalVariable(name: "output", scope: !5, file: !3, type: !18)
!18 = !DICompositeType(tag: DW_TAG_array_type, baseType: !9, size: 32, align: 32, elements: !19)
!19 = !{!20}
!20 = !DISubrange(count: 0, lowerBound: 1)
!21 = !DILocalVariable(scope: !5, file: !3, type: !22, flags: DIFlagArtificial)
!22 = !DICompositeType(tag: DW_TAG_array_type, baseType: !23, size: 1024, align: 64, elements: !24)
!23 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!24 = !{!25}
!25 = !DISubrange(count: 16, lowerBound: 1)
!26 = !DILocalVariable(name: "input", scope: !5, file: !3, type: !18)
!27 = !DILocalVariable(name: "args", scope: !5, file: !3, type: !28)
!28 = !DICompositeType(tag: DW_TAG_array_type, baseType: !29, size: 640, align: 8, elements: !19)
!29 = !DIBasicType(name: "character", size: 640, align: 8, encoding: DW_ATE_signed)
!30 = !DILocalVariable(name: "inlen", scope: !5, file: !3, type: !9)
!31 = !DILocation(line: 25, column: 1, scope: !5)
!32 = !DILocalVariable(name: "outlen", scope: !5, file: !3, type: !9)
!33 = !DILocation(line: 26, column: 1, scope: !5)
!34 = !DILocation(line: 28, column: 1, scope: !5)
!35 = !DILocalVariable(name: "argcount", scope: !5, file: !3, type: !9)
!36 = !DILocation(line: 29, column: 1, scope: !5)
!37 = !DILocation(line: 30, column: 1, scope: !5)
!38 = !DILocalVariable(scope: !5, file: !3, type: !9, flags: DIFlagArtificial)
!39 = !DILocalVariable(scope: !5, file: !3, type: !23, flags: DIFlagArtificial)
!40 = !DILocation(line: 33, column: 1, scope: !5)
!41 = !DILocalVariable(name: "allocstatus", scope: !5, file: !3, type: !9)
!42 = !DILocation(line: 34, column: 1, scope: !5)
!43 = !DILocation(line: 35, column: 1, scope: !5)
!44 = !DILocation(line: 36, column: 1, scope: !5)
!45 = !DILocation(line: 39, column: 1, scope: !5)
!46 = !DILocalVariable(name: "ix", scope: !5, file: !3, type: !9)
!47 = !DILocation(line: 40, column: 1, scope: !5)
!48 = !DILocation(line: 41, column: 1, scope: !5)
!49 = !DILocation(line: 43, column: 1, scope: !5)
!50 = !DILocation(line: 44, column: 1, scope: !5)
!51 = !DILocalVariable(name: "rderr", scope: !5, file: !3, type: !9)
!52 = !DILocation(line: 45, column: 1, scope: !5)
!53 = !DILocation(line: 46, column: 1, scope: !5)
!54 = !DILocation(line: 50, column: 1, scope: !5)
!55 = !DILocation(line: 51, column: 1, scope: !5)
!56 = !DILocation(line: 53, column: 1, scope: !5)
!57 = !DILocalVariable(name: "i", scope: !5, file: !3, type: !9)
!58 = !DILocation(line: 54, column: 1, scope: !5)
!59 = !DILocation(line: 55, column: 1, scope: !5)
!60 = !DILocation(line: 57, column: 1, scope: !5)
!61 = !DILocation(line: 64, column: 1, scope: !5)
!62 = !DILocation(line: 68, column: 1, scope: !5)
!63 = distinct !DISubprogram(name: "__nv_MAIN__F1L57_1", scope: !2, file: !3, line: 57, type: !64, scopeLine: 57, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!64 = !DISubroutineType(types: !65)
!65 = !{null, !9, !23, !23}
!66 = !DILocalVariable(name: "__nv_MAIN__F1L57_1Arg0", arg: 1, scope: !63, file: !3, type: !9)
!67 = !DILocation(line: 0, scope: !63)
!68 = !DILocalVariable(name: "__nv_MAIN__F1L57_1Arg1", arg: 2, scope: !63, file: !3, type: !23)
!69 = !DILocalVariable(name: "__nv_MAIN__F1L57_1Arg2", arg: 3, scope: !63, file: !3, type: !23)
!70 = !DILocalVariable(name: "omp_sched_static", scope: !63, file: !3, type: !9)
!71 = !DILocalVariable(name: "omp_proc_bind_false", scope: !63, file: !3, type: !9)
!72 = !DILocalVariable(name: "omp_proc_bind_true", scope: !63, file: !3, type: !9)
!73 = !DILocalVariable(name: "omp_lock_hint_none", scope: !63, file: !3, type: !9)
!74 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !63, file: !3, type: !9)
!75 = !DILocation(line: 61, column: 1, scope: !63)
!76 = !DILocation(line: 58, column: 1, scope: !63)
!77 = !DILocalVariable(name: "i", scope: !63, file: !3, type: !9)
!78 = !DILocation(line: 59, column: 1, scope: !63)
!79 = !DILocation(line: 60, column: 1, scope: !63)

; ModuleID = '/tmp/DRB020-privatemissing-var-yes-d853b4.ll'
source_filename = "/tmp/DRB020-privatemissing-var-yes-d853b4.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.STATICS1 = type <{ [44 x i8] }>
%astruct.dt88 = type <{ i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8* }>

@.STATICS1 = internal global %struct.STATICS1 <{ [44 x i8] c"\FB\FF\FF\FF\06\00\00\00a(50)=\00\00\EB\FF\FF\FF\00\00\00\00\03\00\00\00\00\00\00\00\01\00\00\00\FF\FF\FF\FF\00\00\00\00" }>, align 16
@.C361_MAIN_ = internal constant i64 50
@.C359_MAIN_ = internal constant i32 56
@.C373_MAIN_ = internal constant i64 25
@.C351_MAIN_ = internal constant [29 x i8] c"Error, invalid integer value."
@.C350_MAIN_ = internal constant i32 39
@.C306_MAIN_ = internal constant i32 25
@.C346_MAIN_ = internal constant [5 x i8] c"(i10)"
@.C343_MAIN_ = internal constant i32 37
@.C366_MAIN_ = internal constant i64 4
@.C344_MAIN_ = internal constant [37 x i8] c"Allocation error, program terminated."
@.C308_MAIN_ = internal constant i32 28
@.C370_MAIN_ = internal constant i64 80
@.C369_MAIN_ = internal constant i64 14
@.C335_MAIN_ = internal constant [35 x i8] c"No command line arguments provided."
@.C333_MAIN_ = internal constant i32 6
@.C334_MAIN_ = internal constant [3 x i8] c"(a)"
@.C285_MAIN_ = internal constant i32 1
@.C305_MAIN_ = internal constant i32 14
@.C331_MAIN_ = internal constant [58 x i8] c"micro-benchmarks-fortran/DRB020-privatemissing-var-yes.f95"
@.C307_MAIN_ = internal constant i32 23
@.C327_MAIN_ = internal constant i32 100
@.C286_MAIN_ = internal constant i64 1
@.C284_MAIN_ = internal constant i64 0
@.C283_MAIN_ = internal constant i32 0
@.C285___nv_MAIN__F1L49_1 = internal constant i32 1
@.C283___nv_MAIN__F1L49_1 = internal constant i32 0

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__497 = alloca i32, align 4
  %.Z0977_352 = alloca i32*, align 8
  %"a$sd2_372" = alloca [16 x i64], align 8
  %.Z0971_342 = alloca [80 x i8]*, align 8
  %"args$sd1_368" = alloca [16 x i64], align 8
  %len_328 = alloca i32, align 4
  %argcount_311 = alloca i32, align 4
  %z__io_337 = alloca i32, align 4
  %z_b_0_315 = alloca i64, align 8
  %z_b_1_316 = alloca i64, align 8
  %z_e_61_319 = alloca i64, align 8
  %z_b_2_317 = alloca i64, align 8
  %z_b_3_318 = alloca i64, align 8
  %allocstatus_312 = alloca i32, align 4
  %.dY0001_383 = alloca i32, align 4
  %ix_314 = alloca i32, align 4
  %rderr_313 = alloca i32, align 4
  %z_b_4_321 = alloca i64, align 8
  %z_b_5_322 = alloca i64, align 8
  %z_e_68_325 = alloca i64, align 8
  %z_b_6_323 = alloca i64, align 8
  %z_b_7_324 = alloca i64, align 8
  %.dY0002_388 = alloca i32, align 4
  %i_309 = alloca i32, align 4
  %.uplevelArgPack0001_473 = alloca %astruct.dt88, align 16
  %tmp_310 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !14, metadata !DIExpression()), !dbg !10
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !15
  store i32 %0, i32* %__gtid_MAIN__497, align 4, !dbg !15
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !16
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !16
  call void (i8*, ...) %2(i8* %1), !dbg !16
  call void @llvm.dbg.declare(metadata i32** %.Z0977_352, metadata !17, metadata !DIExpression(DW_OP_deref)), !dbg !10
  %3 = bitcast i32** %.Z0977_352 to i8**, !dbg !16
  store i8* null, i8** %3, align 8, !dbg !16
  call void @llvm.dbg.declare(metadata [16 x i64]* %"a$sd2_372", metadata !21, metadata !DIExpression()), !dbg !10
  %4 = bitcast [16 x i64]* %"a$sd2_372" to i64*, !dbg !16
  store i64 0, i64* %4, align 8, !dbg !16
  call void @llvm.dbg.declare(metadata [80 x i8]** %.Z0971_342, metadata !26, metadata !DIExpression(DW_OP_deref)), !dbg !10
  %5 = bitcast [80 x i8]** %.Z0971_342 to i8**, !dbg !16
  store i8* null, i8** %5, align 8, !dbg !16
  call void @llvm.dbg.declare(metadata [16 x i64]* %"args$sd1_368", metadata !21, metadata !DIExpression()), !dbg !10
  %6 = bitcast [16 x i64]* %"args$sd1_368" to i64*, !dbg !16
  store i64 0, i64* %6, align 8, !dbg !16
  br label %L.LB1_410

L.LB1_410:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %len_328, metadata !29, metadata !DIExpression()), !dbg !10
  store i32 100, i32* %len_328, align 4, !dbg !30
  %7 = call i32 (...) @f90_cmd_arg_cnt(), !dbg !31
  call void @llvm.dbg.declare(metadata i32* %argcount_311, metadata !32, metadata !DIExpression()), !dbg !10
  store i32 %7, i32* %argcount_311, align 4, !dbg !31
  %8 = load i32, i32* %argcount_311, align 4, !dbg !33
  call void @llvm.dbg.value(metadata i32 %8, metadata !32, metadata !DIExpression()), !dbg !10
  %9 = icmp ne i32 %8, 0, !dbg !33
  br i1 %9, label %L.LB1_377, label %L.LB1_522, !dbg !33

L.LB1_522:                                        ; preds = %L.LB1_410
  call void (...) @_mp_bcs_nest(), !dbg !34
  %10 = bitcast i32* @.C307_MAIN_ to i8*, !dbg !34
  %11 = bitcast [58 x i8]* @.C331_MAIN_ to i8*, !dbg !34
  %12 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !34
  call void (i8*, i8*, i64, ...) %12(i8* %10, i8* %11, i64 58), !dbg !34
  %13 = bitcast i32* @.C305_MAIN_ to i8*, !dbg !34
  %14 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !34
  %15 = bitcast [3 x i8]* @.C334_MAIN_ to i8*, !dbg !34
  %16 = bitcast i32 (...)* @f90io_encode_fmta to i32 (i8*, i8*, i8*, i64, ...)*, !dbg !34
  %17 = call i32 (i8*, i8*, i8*, i64, ...) %16(i8* %13, i8* %14, i8* %15, i64 3), !dbg !34
  call void @llvm.dbg.declare(metadata i32* %z__io_337, metadata !35, metadata !DIExpression()), !dbg !10
  store i32 %17, i32* %z__io_337, align 4, !dbg !34
  %18 = bitcast i32* @.C333_MAIN_ to i8*, !dbg !34
  %19 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !34
  %20 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !34
  %21 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !34
  %22 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %21(i8* %18, i8* null, i8* %19, i8* %20, i8* null, i8* null, i64 0), !dbg !34
  store i32 %22, i32* %z__io_337, align 4, !dbg !34
  %23 = bitcast i32* @.C305_MAIN_ to i8*, !dbg !34
  %24 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !34
  %25 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !34
  %26 = bitcast [35 x i8]* @.C335_MAIN_ to i8*, !dbg !34
  %27 = bitcast i32 (...)* @f90io_fmt_writea to i32 (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !34
  %28 = call i32 (i8*, i8*, i8*, i8*, i64, ...) %27(i8* %23, i8* %24, i8* %25, i8* %26, i64 35), !dbg !34
  store i32 %28, i32* %z__io_337, align 4, !dbg !34
  %29 = call i32 (...) @f90io_fmtw_end(), !dbg !34
  store i32 %29, i32* %z__io_337, align 4, !dbg !34
  call void (...) @_mp_ecs_nest(), !dbg !34
  br label %L.LB1_377

L.LB1_377:                                        ; preds = %L.LB1_522, %L.LB1_410
  call void @llvm.dbg.declare(metadata i64* %z_b_0_315, metadata !36, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_0_315, align 8, !dbg !37
  %30 = load i32, i32* %argcount_311, align 4, !dbg !37
  call void @llvm.dbg.value(metadata i32 %30, metadata !32, metadata !DIExpression()), !dbg !10
  %31 = sext i32 %30 to i64, !dbg !37
  call void @llvm.dbg.declare(metadata i64* %z_b_1_316, metadata !36, metadata !DIExpression()), !dbg !10
  store i64 %31, i64* %z_b_1_316, align 8, !dbg !37
  %32 = load i64, i64* %z_b_1_316, align 8, !dbg !37
  call void @llvm.dbg.value(metadata i64 %32, metadata !36, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_61_319, metadata !36, metadata !DIExpression()), !dbg !10
  store i64 %32, i64* %z_e_61_319, align 8, !dbg !37
  %33 = bitcast [16 x i64]* %"args$sd1_368" to i8*, !dbg !37
  %34 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !37
  %35 = bitcast i64* @.C369_MAIN_ to i8*, !dbg !37
  %36 = bitcast i64* @.C370_MAIN_ to i8*, !dbg !37
  %37 = bitcast i64* %z_b_0_315 to i8*, !dbg !37
  %38 = bitcast i64* %z_b_1_316 to i8*, !dbg !37
  %39 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !37
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %39(i8* %33, i8* %34, i8* %35, i8* %36, i8* %37, i8* %38), !dbg !37
  %40 = bitcast [16 x i64]* %"args$sd1_368" to i8*, !dbg !37
  %41 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !37
  call void (i8*, i32, ...) %41(i8* %40, i32 14), !dbg !37
  %42 = load i64, i64* %z_b_1_316, align 8, !dbg !37
  call void @llvm.dbg.value(metadata i64 %42, metadata !36, metadata !DIExpression()), !dbg !10
  %43 = load i64, i64* %z_b_0_315, align 8, !dbg !37
  call void @llvm.dbg.value(metadata i64 %43, metadata !36, metadata !DIExpression()), !dbg !10
  %44 = sub nsw i64 %43, 1, !dbg !37
  %45 = sub nsw i64 %42, %44, !dbg !37
  call void @llvm.dbg.declare(metadata i64* %z_b_2_317, metadata !36, metadata !DIExpression()), !dbg !10
  store i64 %45, i64* %z_b_2_317, align 8, !dbg !37
  %46 = load i64, i64* %z_b_0_315, align 8, !dbg !37
  call void @llvm.dbg.value(metadata i64 %46, metadata !36, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_b_3_318, metadata !36, metadata !DIExpression()), !dbg !10
  store i64 %46, i64* %z_b_3_318, align 8, !dbg !37
  %47 = bitcast i64* %z_b_2_317 to i8*, !dbg !37
  %48 = bitcast i64* @.C369_MAIN_ to i8*, !dbg !37
  %49 = bitcast i64* @.C370_MAIN_ to i8*, !dbg !37
  call void @llvm.dbg.declare(metadata i32* %allocstatus_312, metadata !38, metadata !DIExpression()), !dbg !10
  %50 = bitcast i32* %allocstatus_312 to i8*, !dbg !37
  %51 = bitcast [80 x i8]** %.Z0971_342 to i8*, !dbg !37
  %52 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !37
  %53 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !37
  %54 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !37
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %54(i8* %47, i8* %48, i8* %49, i8* %50, i8* %51, i8* null, i8* %52, i8* %53, i8* null, i64 0), !dbg !37
  %55 = load i32, i32* %allocstatus_312, align 4, !dbg !39
  call void @llvm.dbg.value(metadata i32 %55, metadata !38, metadata !DIExpression()), !dbg !10
  %56 = icmp sle i32 %55, 0, !dbg !39
  br i1 %56, label %L.LB1_380, label %L.LB1_523, !dbg !39

L.LB1_523:                                        ; preds = %L.LB1_377
  call void (...) @_mp_bcs_nest(), !dbg !40
  %57 = bitcast i32* @.C308_MAIN_ to i8*, !dbg !40
  %58 = bitcast [58 x i8]* @.C331_MAIN_ to i8*, !dbg !40
  %59 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !40
  call void (i8*, i8*, i64, ...) %59(i8* %57, i8* %58, i64 58), !dbg !40
  %60 = bitcast i32* @.C305_MAIN_ to i8*, !dbg !40
  %61 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !40
  %62 = bitcast [3 x i8]* @.C334_MAIN_ to i8*, !dbg !40
  %63 = bitcast i32 (...)* @f90io_encode_fmta to i32 (i8*, i8*, i8*, i64, ...)*, !dbg !40
  %64 = call i32 (i8*, i8*, i8*, i64, ...) %63(i8* %60, i8* %61, i8* %62, i64 3), !dbg !40
  store i32 %64, i32* %z__io_337, align 4, !dbg !40
  %65 = bitcast i32* @.C333_MAIN_ to i8*, !dbg !40
  %66 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !40
  %67 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !40
  %68 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !40
  %69 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %68(i8* %65, i8* null, i8* %66, i8* %67, i8* null, i8* null, i64 0), !dbg !40
  store i32 %69, i32* %z__io_337, align 4, !dbg !40
  %70 = bitcast i32* @.C305_MAIN_ to i8*, !dbg !40
  %71 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !40
  %72 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !40
  %73 = bitcast [37 x i8]* @.C344_MAIN_ to i8*, !dbg !40
  %74 = bitcast i32 (...)* @f90io_fmt_writea to i32 (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !40
  %75 = call i32 (i8*, i8*, i8*, i8*, i64, ...) %74(i8* %70, i8* %71, i8* %72, i8* %73, i64 37), !dbg !40
  store i32 %75, i32* %z__io_337, align 4, !dbg !40
  %76 = call i32 (...) @f90io_fmtw_end(), !dbg !40
  store i32 %76, i32* %z__io_337, align 4, !dbg !40
  call void (...) @_mp_ecs_nest(), !dbg !40
  %77 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !41
  %78 = bitcast void (...)* @f90_stop08a to void (i8*, i8*, i64, ...)*, !dbg !41
  call void (i8*, i8*, i64, ...) %78(i8* %77, i8* null, i64 0), !dbg !41
  br label %L.LB1_380

L.LB1_380:                                        ; preds = %L.LB1_523, %L.LB1_377
  %79 = load i32, i32* %argcount_311, align 4, !dbg !42
  call void @llvm.dbg.value(metadata i32 %79, metadata !32, metadata !DIExpression()), !dbg !10
  store i32 %79, i32* %.dY0001_383, align 4, !dbg !42
  call void @llvm.dbg.declare(metadata i32* %ix_314, metadata !43, metadata !DIExpression()), !dbg !10
  store i32 1, i32* %ix_314, align 4, !dbg !42
  %80 = load i32, i32* %.dY0001_383, align 4, !dbg !42
  %81 = icmp sle i32 %80, 0, !dbg !42
  br i1 %81, label %L.LB1_382, label %L.LB1_381, !dbg !42

L.LB1_381:                                        ; preds = %L.LB1_381, %L.LB1_380
  %82 = bitcast i32* %ix_314 to i8*, !dbg !44
  %83 = load [80 x i8]*, [80 x i8]** %.Z0971_342, align 8, !dbg !44
  call void @llvm.dbg.value(metadata [80 x i8]* %83, metadata !26, metadata !DIExpression()), !dbg !10
  %84 = bitcast [80 x i8]* %83 to i8*, !dbg !44
  %85 = getelementptr i8, i8* %84, i64 -80, !dbg !44
  %86 = load i32, i32* %ix_314, align 4, !dbg !44
  call void @llvm.dbg.value(metadata i32 %86, metadata !43, metadata !DIExpression()), !dbg !10
  %87 = sext i32 %86 to i64, !dbg !44
  %88 = bitcast [16 x i64]* %"args$sd1_368" to i8*, !dbg !44
  %89 = getelementptr i8, i8* %88, i64 56, !dbg !44
  %90 = bitcast i8* %89 to i64*, !dbg !44
  %91 = load i64, i64* %90, align 8, !dbg !44
  %92 = add nsw i64 %87, %91, !dbg !44
  %93 = mul nsw i64 %92, 80, !dbg !44
  %94 = getelementptr i8, i8* %85, i64 %93, !dbg !44
  %95 = bitcast i64* @.C366_MAIN_ to i8*, !dbg !44
  %96 = bitcast void (...)* @f90_get_cmd_arga to void (i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !44
  call void (i8*, i8*, i8*, i8*, i8*, i64, ...) %96(i8* %82, i8* %94, i8* null, i8* null, i8* %95, i64 80), !dbg !44
  %97 = load i32, i32* %ix_314, align 4, !dbg !45
  call void @llvm.dbg.value(metadata i32 %97, metadata !43, metadata !DIExpression()), !dbg !10
  %98 = add nsw i32 %97, 1, !dbg !45
  store i32 %98, i32* %ix_314, align 4, !dbg !45
  %99 = load i32, i32* %.dY0001_383, align 4, !dbg !45
  %100 = sub nsw i32 %99, 1, !dbg !45
  store i32 %100, i32* %.dY0001_383, align 4, !dbg !45
  %101 = load i32, i32* %.dY0001_383, align 4, !dbg !45
  %102 = icmp sgt i32 %101, 0, !dbg !45
  br i1 %102, label %L.LB1_381, label %L.LB1_382, !dbg !45

L.LB1_382:                                        ; preds = %L.LB1_381, %L.LB1_380
  %103 = load i32, i32* %argcount_311, align 4, !dbg !46
  call void @llvm.dbg.value(metadata i32 %103, metadata !32, metadata !DIExpression()), !dbg !10
  %104 = icmp sle i32 %103, 0, !dbg !46
  br i1 %104, label %L.LB1_384, label %L.LB1_524, !dbg !46

L.LB1_524:                                        ; preds = %L.LB1_382
  call void (...) @_mp_bcs_nest(), !dbg !47
  %105 = bitcast i32* @.C343_MAIN_ to i8*, !dbg !47
  %106 = bitcast [58 x i8]* @.C331_MAIN_ to i8*, !dbg !47
  %107 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !47
  call void (i8*, i8*, i64, ...) %107(i8* %105, i8* %106, i64 58), !dbg !47
  %108 = bitcast i32* @.C305_MAIN_ to i8*, !dbg !47
  %109 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !47
  %110 = bitcast [5 x i8]* @.C346_MAIN_ to i8*, !dbg !47
  %111 = bitcast i32 (...)* @f90io_encode_fmta to i32 (i8*, i8*, i8*, i64, ...)*, !dbg !47
  %112 = call i32 (i8*, i8*, i8*, i64, ...) %111(i8* %108, i8* %109, i8* %110, i64 5), !dbg !47
  store i32 %112, i32* %z__io_337, align 4, !dbg !47
  %113 = load [80 x i8]*, [80 x i8]** %.Z0971_342, align 8, !dbg !47
  call void @llvm.dbg.value(metadata [80 x i8]* %113, metadata !26, metadata !DIExpression()), !dbg !10
  %114 = bitcast [80 x i8]* %113 to i8*, !dbg !47
  %115 = bitcast [16 x i64]* %"args$sd1_368" to i8*, !dbg !47
  %116 = getelementptr i8, i8* %115, i64 56, !dbg !47
  %117 = bitcast i8* %116 to i64*, !dbg !47
  %118 = load i64, i64* %117, align 8, !dbg !47
  %119 = mul nsw i64 %118, 80, !dbg !47
  %120 = getelementptr i8, i8* %114, i64 %119, !dbg !47
  %121 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !47
  %122 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !47
  call void @llvm.dbg.declare(metadata i32* %rderr_313, metadata !48, metadata !DIExpression()), !dbg !10
  %123 = bitcast i32* %rderr_313 to i8*, !dbg !47
  %124 = bitcast i32 (...)* @f90io_fmtr_intern_inita to i32 (i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !47
  %125 = call i32 (i8*, i8*, i8*, i8*, i8*, i64, ...) %124(i8* %120, i8* %121, i8* %122, i8* %123, i8* null, i64 80), !dbg !47
  store i32 %125, i32* %z__io_337, align 4, !dbg !47
  %126 = bitcast i32* @.C306_MAIN_ to i8*, !dbg !47
  %127 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !47
  %128 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !47
  %129 = bitcast i32* %len_328 to i8*, !dbg !47
  %130 = bitcast i32 (...)* @f90io_fmt_reada to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !47
  %131 = call i32 (i8*, i8*, i8*, i8*, ...) %130(i8* %126, i8* %127, i8* %128, i8* %129), !dbg !47
  store i32 %131, i32* %z__io_337, align 4, !dbg !47
  %132 = call i32 (...) @f90io_fmtr_end(), !dbg !47
  store i32 %132, i32* %z__io_337, align 4, !dbg !47
  call void (...) @_mp_ecs_nest(), !dbg !47
  %133 = load i32, i32* %rderr_313, align 4, !dbg !49
  call void @llvm.dbg.value(metadata i32 %133, metadata !48, metadata !DIExpression()), !dbg !10
  %134 = icmp eq i32 %133, 0, !dbg !49
  br i1 %134, label %L.LB1_385, label %L.LB1_525, !dbg !49

L.LB1_525:                                        ; preds = %L.LB1_524
  call void (...) @_mp_bcs_nest(), !dbg !50
  %135 = bitcast i32* @.C350_MAIN_ to i8*, !dbg !50
  %136 = bitcast [58 x i8]* @.C331_MAIN_ to i8*, !dbg !50
  %137 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !50
  call void (i8*, i8*, i64, ...) %137(i8* %135, i8* %136, i64 58), !dbg !50
  %138 = bitcast i32* @.C305_MAIN_ to i8*, !dbg !50
  %139 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !50
  %140 = bitcast [3 x i8]* @.C334_MAIN_ to i8*, !dbg !50
  %141 = bitcast i32 (...)* @f90io_encode_fmta to i32 (i8*, i8*, i8*, i64, ...)*, !dbg !50
  %142 = call i32 (i8*, i8*, i8*, i64, ...) %141(i8* %138, i8* %139, i8* %140, i64 3), !dbg !50
  store i32 %142, i32* %z__io_337, align 4, !dbg !50
  %143 = bitcast i32* @.C333_MAIN_ to i8*, !dbg !50
  %144 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !50
  %145 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !50
  %146 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !50
  %147 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %146(i8* %143, i8* null, i8* %144, i8* %145, i8* null, i8* null, i64 0), !dbg !50
  store i32 %147, i32* %z__io_337, align 4, !dbg !50
  %148 = bitcast i32* @.C305_MAIN_ to i8*, !dbg !50
  %149 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !50
  %150 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !50
  %151 = bitcast [29 x i8]* @.C351_MAIN_ to i8*, !dbg !50
  %152 = bitcast i32 (...)* @f90io_fmt_writea to i32 (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !50
  %153 = call i32 (i8*, i8*, i8*, i8*, i64, ...) %152(i8* %148, i8* %149, i8* %150, i8* %151, i64 29), !dbg !50
  store i32 %153, i32* %z__io_337, align 4, !dbg !50
  %154 = call i32 (...) @f90io_fmtw_end(), !dbg !50
  store i32 %154, i32* %z__io_337, align 4, !dbg !50
  call void (...) @_mp_ecs_nest(), !dbg !50
  br label %L.LB1_385

L.LB1_385:                                        ; preds = %L.LB1_525, %L.LB1_524
  br label %L.LB1_384

L.LB1_384:                                        ; preds = %L.LB1_385, %L.LB1_382
  call void @llvm.dbg.declare(metadata i64* %z_b_4_321, metadata !36, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_4_321, align 8, !dbg !51
  %155 = load i32, i32* %len_328, align 4, !dbg !51
  call void @llvm.dbg.value(metadata i32 %155, metadata !29, metadata !DIExpression()), !dbg !10
  %156 = sext i32 %155 to i64, !dbg !51
  call void @llvm.dbg.declare(metadata i64* %z_b_5_322, metadata !36, metadata !DIExpression()), !dbg !10
  store i64 %156, i64* %z_b_5_322, align 8, !dbg !51
  %157 = load i64, i64* %z_b_5_322, align 8, !dbg !51
  call void @llvm.dbg.value(metadata i64 %157, metadata !36, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_68_325, metadata !36, metadata !DIExpression()), !dbg !10
  store i64 %157, i64* %z_e_68_325, align 8, !dbg !51
  %158 = bitcast [16 x i64]* %"a$sd2_372" to i8*, !dbg !51
  %159 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !51
  %160 = bitcast i64* @.C373_MAIN_ to i8*, !dbg !51
  %161 = bitcast i64* @.C366_MAIN_ to i8*, !dbg !51
  %162 = bitcast i64* %z_b_4_321 to i8*, !dbg !51
  %163 = bitcast i64* %z_b_5_322 to i8*, !dbg !51
  %164 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !51
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %164(i8* %158, i8* %159, i8* %160, i8* %161, i8* %162, i8* %163), !dbg !51
  %165 = bitcast [16 x i64]* %"a$sd2_372" to i8*, !dbg !51
  %166 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !51
  call void (i8*, i32, ...) %166(i8* %165, i32 25), !dbg !51
  %167 = load i64, i64* %z_b_5_322, align 8, !dbg !51
  call void @llvm.dbg.value(metadata i64 %167, metadata !36, metadata !DIExpression()), !dbg !10
  %168 = load i64, i64* %z_b_4_321, align 8, !dbg !51
  call void @llvm.dbg.value(metadata i64 %168, metadata !36, metadata !DIExpression()), !dbg !10
  %169 = sub nsw i64 %168, 1, !dbg !51
  %170 = sub nsw i64 %167, %169, !dbg !51
  call void @llvm.dbg.declare(metadata i64* %z_b_6_323, metadata !36, metadata !DIExpression()), !dbg !10
  store i64 %170, i64* %z_b_6_323, align 8, !dbg !51
  %171 = load i64, i64* %z_b_4_321, align 8, !dbg !51
  call void @llvm.dbg.value(metadata i64 %171, metadata !36, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_b_7_324, metadata !36, metadata !DIExpression()), !dbg !10
  store i64 %171, i64* %z_b_7_324, align 8, !dbg !51
  %172 = bitcast i64* %z_b_6_323 to i8*, !dbg !51
  %173 = bitcast i64* @.C373_MAIN_ to i8*, !dbg !51
  %174 = bitcast i64* @.C366_MAIN_ to i8*, !dbg !51
  %175 = bitcast i32** %.Z0977_352 to i8*, !dbg !51
  %176 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !51
  %177 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !51
  %178 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !51
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %178(i8* %172, i8* %173, i8* %174, i8* null, i8* %175, i8* null, i8* %176, i8* %177, i8* null, i64 0), !dbg !51
  %179 = load i32, i32* %len_328, align 4, !dbg !52
  call void @llvm.dbg.value(metadata i32 %179, metadata !29, metadata !DIExpression()), !dbg !10
  store i32 %179, i32* %.dY0002_388, align 4, !dbg !52
  call void @llvm.dbg.declare(metadata i32* %i_309, metadata !53, metadata !DIExpression()), !dbg !10
  store i32 1, i32* %i_309, align 4, !dbg !52
  %180 = load i32, i32* %.dY0002_388, align 4, !dbg !52
  %181 = icmp sle i32 %180, 0, !dbg !52
  br i1 %181, label %L.LB1_387, label %L.LB1_386, !dbg !52

L.LB1_386:                                        ; preds = %L.LB1_386, %L.LB1_384
  %182 = load i32, i32* %i_309, align 4, !dbg !54
  call void @llvm.dbg.value(metadata i32 %182, metadata !53, metadata !DIExpression()), !dbg !10
  %183 = load i32, i32* %i_309, align 4, !dbg !54
  call void @llvm.dbg.value(metadata i32 %183, metadata !53, metadata !DIExpression()), !dbg !10
  %184 = sext i32 %183 to i64, !dbg !54
  %185 = bitcast [16 x i64]* %"a$sd2_372" to i8*, !dbg !54
  %186 = getelementptr i8, i8* %185, i64 56, !dbg !54
  %187 = bitcast i8* %186 to i64*, !dbg !54
  %188 = load i64, i64* %187, align 8, !dbg !54
  %189 = add nsw i64 %184, %188, !dbg !54
  %190 = load i32*, i32** %.Z0977_352, align 8, !dbg !54
  call void @llvm.dbg.value(metadata i32* %190, metadata !17, metadata !DIExpression()), !dbg !10
  %191 = bitcast i32* %190 to i8*, !dbg !54
  %192 = getelementptr i8, i8* %191, i64 -4, !dbg !54
  %193 = bitcast i8* %192 to i32*, !dbg !54
  %194 = getelementptr i32, i32* %193, i64 %189, !dbg !54
  store i32 %182, i32* %194, align 4, !dbg !54
  %195 = load i32, i32* %i_309, align 4, !dbg !55
  call void @llvm.dbg.value(metadata i32 %195, metadata !53, metadata !DIExpression()), !dbg !10
  %196 = add nsw i32 %195, 1, !dbg !55
  store i32 %196, i32* %i_309, align 4, !dbg !55
  %197 = load i32, i32* %.dY0002_388, align 4, !dbg !55
  %198 = sub nsw i32 %197, 1, !dbg !55
  store i32 %198, i32* %.dY0002_388, align 4, !dbg !55
  %199 = load i32, i32* %.dY0002_388, align 4, !dbg !55
  %200 = icmp sgt i32 %199, 0, !dbg !55
  br i1 %200, label %L.LB1_386, label %L.LB1_387, !dbg !55

L.LB1_387:                                        ; preds = %L.LB1_386, %L.LB1_384
  %201 = bitcast i32* %len_328 to i8*, !dbg !56
  %202 = bitcast %astruct.dt88* %.uplevelArgPack0001_473 to i8**, !dbg !56
  store i8* %201, i8** %202, align 8, !dbg !56
  %203 = bitcast i32** %.Z0977_352 to i8*, !dbg !56
  %204 = bitcast %astruct.dt88* %.uplevelArgPack0001_473 to i8*, !dbg !56
  %205 = getelementptr i8, i8* %204, i64 8, !dbg !56
  %206 = bitcast i8* %205 to i8**, !dbg !56
  store i8* %203, i8** %206, align 8, !dbg !56
  %207 = bitcast i32** %.Z0977_352 to i8*, !dbg !56
  %208 = bitcast %astruct.dt88* %.uplevelArgPack0001_473 to i8*, !dbg !56
  %209 = getelementptr i8, i8* %208, i64 16, !dbg !56
  %210 = bitcast i8* %209 to i8**, !dbg !56
  store i8* %207, i8** %210, align 8, !dbg !56
  %211 = bitcast i64* %z_b_4_321 to i8*, !dbg !56
  %212 = bitcast %astruct.dt88* %.uplevelArgPack0001_473 to i8*, !dbg !56
  %213 = getelementptr i8, i8* %212, i64 24, !dbg !56
  %214 = bitcast i8* %213 to i8**, !dbg !56
  store i8* %211, i8** %214, align 8, !dbg !56
  %215 = bitcast i64* %z_b_5_322 to i8*, !dbg !56
  %216 = bitcast %astruct.dt88* %.uplevelArgPack0001_473 to i8*, !dbg !56
  %217 = getelementptr i8, i8* %216, i64 32, !dbg !56
  %218 = bitcast i8* %217 to i8**, !dbg !56
  store i8* %215, i8** %218, align 8, !dbg !56
  %219 = bitcast i64* %z_e_68_325 to i8*, !dbg !56
  %220 = bitcast %astruct.dt88* %.uplevelArgPack0001_473 to i8*, !dbg !56
  %221 = getelementptr i8, i8* %220, i64 40, !dbg !56
  %222 = bitcast i8* %221 to i8**, !dbg !56
  store i8* %219, i8** %222, align 8, !dbg !56
  %223 = bitcast i64* %z_b_6_323 to i8*, !dbg !56
  %224 = bitcast %astruct.dt88* %.uplevelArgPack0001_473 to i8*, !dbg !56
  %225 = getelementptr i8, i8* %224, i64 48, !dbg !56
  %226 = bitcast i8* %225 to i8**, !dbg !56
  store i8* %223, i8** %226, align 8, !dbg !56
  %227 = bitcast i64* %z_b_7_324 to i8*, !dbg !56
  %228 = bitcast %astruct.dt88* %.uplevelArgPack0001_473 to i8*, !dbg !56
  %229 = getelementptr i8, i8* %228, i64 56, !dbg !56
  %230 = bitcast i8* %229 to i8**, !dbg !56
  store i8* %227, i8** %230, align 8, !dbg !56
  call void @llvm.dbg.declare(metadata i32* %tmp_310, metadata !57, metadata !DIExpression()), !dbg !10
  %231 = bitcast i32* %tmp_310 to i8*, !dbg !56
  %232 = bitcast %astruct.dt88* %.uplevelArgPack0001_473 to i8*, !dbg !56
  %233 = getelementptr i8, i8* %232, i64 64, !dbg !56
  %234 = bitcast i8* %233 to i8**, !dbg !56
  store i8* %231, i8** %234, align 8, !dbg !56
  %235 = bitcast [16 x i64]* %"a$sd2_372" to i8*, !dbg !56
  %236 = bitcast %astruct.dt88* %.uplevelArgPack0001_473 to i8*, !dbg !56
  %237 = getelementptr i8, i8* %236, i64 72, !dbg !56
  %238 = bitcast i8* %237 to i8**, !dbg !56
  store i8* %235, i8** %238, align 8, !dbg !56
  br label %L.LB1_495, !dbg !56

L.LB1_495:                                        ; preds = %L.LB1_387
  %239 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L49_1_ to i64*, !dbg !56
  %240 = bitcast %astruct.dt88* %.uplevelArgPack0001_473 to i64*, !dbg !56
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %239, i64* %240), !dbg !56
  call void (...) @_mp_bcs_nest(), !dbg !58
  %241 = bitcast i32* @.C359_MAIN_ to i8*, !dbg !58
  %242 = bitcast [58 x i8]* @.C331_MAIN_ to i8*, !dbg !58
  %243 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !58
  call void (i8*, i8*, i64, ...) %243(i8* %241, i8* %242, i64 58), !dbg !58
  %244 = bitcast i32* @.C333_MAIN_ to i8*, !dbg !58
  %245 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !58
  %246 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !58
  %247 = bitcast %struct.STATICS1* @.STATICS1 to i8*, !dbg !58
  %248 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !58
  %249 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %248(i8* %244, i8* null, i8* %245, i8* %246, i8* %247, i8* null, i64 0), !dbg !58
  store i32 %249, i32* %z__io_337, align 4, !dbg !58
  %250 = bitcast [16 x i64]* %"a$sd2_372" to i8*, !dbg !58
  %251 = getelementptr i8, i8* %250, i64 56, !dbg !58
  %252 = bitcast i8* %251 to i64*, !dbg !58
  %253 = load i64, i64* %252, align 8, !dbg !58
  %254 = load i32*, i32** %.Z0977_352, align 8, !dbg !58
  call void @llvm.dbg.value(metadata i32* %254, metadata !17, metadata !DIExpression()), !dbg !10
  %255 = bitcast i32* %254 to i8*, !dbg !58
  %256 = getelementptr i8, i8* %255, i64 196, !dbg !58
  %257 = bitcast i8* %256 to i32*, !dbg !58
  %258 = getelementptr i32, i32* %257, i64 %253, !dbg !58
  %259 = load i32, i32* %258, align 4, !dbg !58
  %260 = bitcast i32 (...)* @f90io_sc_i_fmt_write to i32 (i32, i32, ...)*, !dbg !58
  %261 = call i32 (i32, i32, ...) %260(i32 %259, i32 25), !dbg !58
  store i32 %261, i32* %z__io_337, align 4, !dbg !58
  %262 = call i32 (...) @f90io_fmtw_end(), !dbg !58
  store i32 %262, i32* %z__io_337, align 4, !dbg !58
  call void (...) @_mp_ecs_nest(), !dbg !58
  %263 = load [80 x i8]*, [80 x i8]** %.Z0971_342, align 8, !dbg !59
  call void @llvm.dbg.value(metadata [80 x i8]* %263, metadata !26, metadata !DIExpression()), !dbg !10
  %264 = bitcast [80 x i8]* %263 to i8*, !dbg !59
  %265 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !59
  %266 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, i64, ...)*, !dbg !59
  call void (i8*, i8*, i8*, i8*, i64, i64, ...) %266(i8* null, i8* %264, i8* %265, i8* null, i64 80, i64 0), !dbg !59
  %267 = bitcast [80 x i8]** %.Z0971_342 to i8**, !dbg !59
  store i8* null, i8** %267, align 8, !dbg !59
  %268 = bitcast [16 x i64]* %"args$sd1_368" to i64*, !dbg !59
  store i64 0, i64* %268, align 8, !dbg !59
  %269 = load i32*, i32** %.Z0977_352, align 8, !dbg !59
  call void @llvm.dbg.value(metadata i32* %269, metadata !17, metadata !DIExpression()), !dbg !10
  %270 = bitcast i32* %269 to i8*, !dbg !59
  %271 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !59
  %272 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !59
  call void (i8*, i8*, i8*, i8*, i64, ...) %272(i8* null, i8* %270, i8* %271, i8* null, i64 0), !dbg !59
  %273 = bitcast i32** %.Z0977_352 to i8**, !dbg !59
  store i8* null, i8** %273, align 8, !dbg !59
  %274 = bitcast [16 x i64]* %"a$sd2_372" to i64*, !dbg !59
  store i64 0, i64* %274, align 8, !dbg !59
  ret void, !dbg !15
}

define internal void @__nv_MAIN__F1L49_1_(i32* %__nv_MAIN__F1L49_1Arg0, i64* %__nv_MAIN__F1L49_1Arg1, i64* %__nv_MAIN__F1L49_1Arg2) #0 !dbg !60 {
L.entry:
  %__gtid___nv_MAIN__F1L49_1__544 = alloca i32, align 4
  %.i0000p_357 = alloca i32, align 4
  %i_356 = alloca i32, align 4
  %.du0003p_392 = alloca i32, align 4
  %.de0003p_393 = alloca i32, align 4
  %.di0003p_394 = alloca i32, align 4
  %.ds0003p_395 = alloca i32, align 4
  %.dl0003p_397 = alloca i32, align 4
  %.dl0003p.copy_538 = alloca i32, align 4
  %.de0003p.copy_539 = alloca i32, align 4
  %.ds0003p.copy_540 = alloca i32, align 4
  %.dX0003p_396 = alloca i32, align 4
  %.dY0003p_391 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L49_1Arg0, metadata !63, metadata !DIExpression()), !dbg !64
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L49_1Arg1, metadata !65, metadata !DIExpression()), !dbg !64
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L49_1Arg2, metadata !66, metadata !DIExpression()), !dbg !64
  call void @llvm.dbg.value(metadata i32 1, metadata !67, metadata !DIExpression()), !dbg !64
  call void @llvm.dbg.value(metadata i32 0, metadata !68, metadata !DIExpression()), !dbg !64
  call void @llvm.dbg.value(metadata i32 1, metadata !69, metadata !DIExpression()), !dbg !64
  call void @llvm.dbg.value(metadata i32 0, metadata !70, metadata !DIExpression()), !dbg !64
  call void @llvm.dbg.value(metadata i32 1, metadata !71, metadata !DIExpression()), !dbg !64
  %0 = load i32, i32* %__nv_MAIN__F1L49_1Arg0, align 4, !dbg !72
  store i32 %0, i32* %__gtid___nv_MAIN__F1L49_1__544, align 4, !dbg !72
  br label %L.LB2_529

L.LB2_529:                                        ; preds = %L.entry
  br label %L.LB2_355

L.LB2_355:                                        ; preds = %L.LB2_529
  store i32 0, i32* %.i0000p_357, align 4, !dbg !73
  call void @llvm.dbg.declare(metadata i32* %i_356, metadata !74, metadata !DIExpression()), !dbg !72
  store i32 1, i32* %i_356, align 4, !dbg !73
  %1 = bitcast i64* %__nv_MAIN__F1L49_1Arg2 to i32**, !dbg !73
  %2 = load i32*, i32** %1, align 8, !dbg !73
  %3 = load i32, i32* %2, align 4, !dbg !73
  store i32 %3, i32* %.du0003p_392, align 4, !dbg !73
  %4 = bitcast i64* %__nv_MAIN__F1L49_1Arg2 to i32**, !dbg !73
  %5 = load i32*, i32** %4, align 8, !dbg !73
  %6 = load i32, i32* %5, align 4, !dbg !73
  store i32 %6, i32* %.de0003p_393, align 4, !dbg !73
  store i32 1, i32* %.di0003p_394, align 4, !dbg !73
  %7 = load i32, i32* %.di0003p_394, align 4, !dbg !73
  store i32 %7, i32* %.ds0003p_395, align 4, !dbg !73
  store i32 1, i32* %.dl0003p_397, align 4, !dbg !73
  %8 = load i32, i32* %.dl0003p_397, align 4, !dbg !73
  store i32 %8, i32* %.dl0003p.copy_538, align 4, !dbg !73
  %9 = load i32, i32* %.de0003p_393, align 4, !dbg !73
  store i32 %9, i32* %.de0003p.copy_539, align 4, !dbg !73
  %10 = load i32, i32* %.ds0003p_395, align 4, !dbg !73
  store i32 %10, i32* %.ds0003p.copy_540, align 4, !dbg !73
  %11 = load i32, i32* %__gtid___nv_MAIN__F1L49_1__544, align 4, !dbg !73
  %12 = bitcast i32* %.i0000p_357 to i64*, !dbg !73
  %13 = bitcast i32* %.dl0003p.copy_538 to i64*, !dbg !73
  %14 = bitcast i32* %.de0003p.copy_539 to i64*, !dbg !73
  %15 = bitcast i32* %.ds0003p.copy_540 to i64*, !dbg !73
  %16 = load i32, i32* %.ds0003p.copy_540, align 4, !dbg !73
  call void @__kmpc_for_static_init_4(i64* null, i32 %11, i32 34, i64* %12, i64* %13, i64* %14, i64* %15, i32 %16, i32 1), !dbg !73
  %17 = load i32, i32* %.dl0003p.copy_538, align 4, !dbg !73
  store i32 %17, i32* %.dl0003p_397, align 4, !dbg !73
  %18 = load i32, i32* %.de0003p.copy_539, align 4, !dbg !73
  store i32 %18, i32* %.de0003p_393, align 4, !dbg !73
  %19 = load i32, i32* %.ds0003p.copy_540, align 4, !dbg !73
  store i32 %19, i32* %.ds0003p_395, align 4, !dbg !73
  %20 = load i32, i32* %.dl0003p_397, align 4, !dbg !73
  store i32 %20, i32* %i_356, align 4, !dbg !73
  %21 = load i32, i32* %i_356, align 4, !dbg !73
  call void @llvm.dbg.value(metadata i32 %21, metadata !74, metadata !DIExpression()), !dbg !72
  store i32 %21, i32* %.dX0003p_396, align 4, !dbg !73
  %22 = load i32, i32* %.dX0003p_396, align 4, !dbg !73
  %23 = load i32, i32* %.du0003p_392, align 4, !dbg !73
  %24 = icmp sgt i32 %22, %23, !dbg !73
  br i1 %24, label %L.LB2_390, label %L.LB2_569, !dbg !73

L.LB2_569:                                        ; preds = %L.LB2_355
  %25 = load i32, i32* %.dX0003p_396, align 4, !dbg !73
  store i32 %25, i32* %i_356, align 4, !dbg !73
  %26 = load i32, i32* %.di0003p_394, align 4, !dbg !73
  %27 = load i32, i32* %.de0003p_393, align 4, !dbg !73
  %28 = load i32, i32* %.dX0003p_396, align 4, !dbg !73
  %29 = sub nsw i32 %27, %28, !dbg !73
  %30 = add nsw i32 %26, %29, !dbg !73
  %31 = load i32, i32* %.di0003p_394, align 4, !dbg !73
  %32 = sdiv i32 %30, %31, !dbg !73
  store i32 %32, i32* %.dY0003p_391, align 4, !dbg !73
  %33 = load i32, i32* %.dY0003p_391, align 4, !dbg !73
  %34 = icmp sle i32 %33, 0, !dbg !73
  br i1 %34, label %L.LB2_400, label %L.LB2_399, !dbg !73

L.LB2_399:                                        ; preds = %L.LB2_399, %L.LB2_569
  %35 = load i32, i32* %i_356, align 4, !dbg !75
  call void @llvm.dbg.value(metadata i32 %35, metadata !74, metadata !DIExpression()), !dbg !72
  %36 = load i32, i32* %i_356, align 4, !dbg !75
  call void @llvm.dbg.value(metadata i32 %36, metadata !74, metadata !DIExpression()), !dbg !72
  %37 = sext i32 %36 to i64, !dbg !75
  %38 = bitcast i64* %__nv_MAIN__F1L49_1Arg2 to i8*, !dbg !75
  %39 = getelementptr i8, i8* %38, i64 72, !dbg !75
  %40 = bitcast i8* %39 to i8**, !dbg !75
  %41 = load i8*, i8** %40, align 8, !dbg !75
  %42 = getelementptr i8, i8* %41, i64 56, !dbg !75
  %43 = bitcast i8* %42 to i64*, !dbg !75
  %44 = load i64, i64* %43, align 8, !dbg !75
  %45 = add nsw i64 %37, %44, !dbg !75
  %46 = bitcast i64* %__nv_MAIN__F1L49_1Arg2 to i8*, !dbg !75
  %47 = getelementptr i8, i8* %46, i64 16, !dbg !75
  %48 = bitcast i8* %47 to i8***, !dbg !75
  %49 = load i8**, i8*** %48, align 8, !dbg !75
  %50 = load i8*, i8** %49, align 8, !dbg !75
  %51 = getelementptr i8, i8* %50, i64 -4, !dbg !75
  %52 = bitcast i8* %51 to i32*, !dbg !75
  %53 = getelementptr i32, i32* %52, i64 %45, !dbg !75
  %54 = load i32, i32* %53, align 4, !dbg !75
  %55 = add nsw i32 %35, %54, !dbg !75
  %56 = bitcast i64* %__nv_MAIN__F1L49_1Arg2 to i8*, !dbg !75
  %57 = getelementptr i8, i8* %56, i64 64, !dbg !75
  %58 = bitcast i8* %57 to i32**, !dbg !75
  %59 = load i32*, i32** %58, align 8, !dbg !75
  store i32 %55, i32* %59, align 4, !dbg !75
  %60 = bitcast i64* %__nv_MAIN__F1L49_1Arg2 to i8*, !dbg !76
  %61 = getelementptr i8, i8* %60, i64 64, !dbg !76
  %62 = bitcast i8* %61 to i32**, !dbg !76
  %63 = load i32*, i32** %62, align 8, !dbg !76
  %64 = load i32, i32* %63, align 4, !dbg !76
  %65 = load i32, i32* %i_356, align 4, !dbg !76
  call void @llvm.dbg.value(metadata i32 %65, metadata !74, metadata !DIExpression()), !dbg !72
  %66 = sext i32 %65 to i64, !dbg !76
  %67 = bitcast i64* %__nv_MAIN__F1L49_1Arg2 to i8*, !dbg !76
  %68 = getelementptr i8, i8* %67, i64 72, !dbg !76
  %69 = bitcast i8* %68 to i8**, !dbg !76
  %70 = load i8*, i8** %69, align 8, !dbg !76
  %71 = getelementptr i8, i8* %70, i64 56, !dbg !76
  %72 = bitcast i8* %71 to i64*, !dbg !76
  %73 = load i64, i64* %72, align 8, !dbg !76
  %74 = add nsw i64 %66, %73, !dbg !76
  %75 = bitcast i64* %__nv_MAIN__F1L49_1Arg2 to i8*, !dbg !76
  %76 = getelementptr i8, i8* %75, i64 16, !dbg !76
  %77 = bitcast i8* %76 to i8***, !dbg !76
  %78 = load i8**, i8*** %77, align 8, !dbg !76
  %79 = load i8*, i8** %78, align 8, !dbg !76
  %80 = getelementptr i8, i8* %79, i64 -4, !dbg !76
  %81 = bitcast i8* %80 to i32*, !dbg !76
  %82 = getelementptr i32, i32* %81, i64 %74, !dbg !76
  store i32 %64, i32* %82, align 4, !dbg !76
  %83 = load i32, i32* %.di0003p_394, align 4, !dbg !72
  %84 = load i32, i32* %i_356, align 4, !dbg !72
  call void @llvm.dbg.value(metadata i32 %84, metadata !74, metadata !DIExpression()), !dbg !72
  %85 = add nsw i32 %83, %84, !dbg !72
  store i32 %85, i32* %i_356, align 4, !dbg !72
  %86 = load i32, i32* %.dY0003p_391, align 4, !dbg !72
  %87 = sub nsw i32 %86, 1, !dbg !72
  store i32 %87, i32* %.dY0003p_391, align 4, !dbg !72
  %88 = load i32, i32* %.dY0003p_391, align 4, !dbg !72
  %89 = icmp sgt i32 %88, 0, !dbg !72
  br i1 %89, label %L.LB2_399, label %L.LB2_400, !dbg !72

L.LB2_400:                                        ; preds = %L.LB2_399, %L.LB2_569
  br label %L.LB2_390

L.LB2_390:                                        ; preds = %L.LB2_400, %L.LB2_355
  %90 = load i32, i32* %__gtid___nv_MAIN__F1L49_1__544, align 4, !dbg !72
  call void @__kmpc_for_static_fini(i64* null, i32 %90), !dbg !72
  br label %L.LB2_358

L.LB2_358:                                        ; preds = %L.LB2_390
  ret void, !dbg !72
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
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB020-privatemissing-var-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb020_privatemissing_var_yes", scope: !2, file: !3, line: 11, type: !6, scopeLine: 11, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!6 = !DISubroutineType(cc: DW_CC_program, types: !7)
!7 = !{null}
!8 = !DILocalVariable(name: "omp_sched_static", scope: !5, file: !3, type: !9)
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 0, scope: !5)
!11 = !DILocalVariable(name: "omp_proc_bind_false", scope: !5, file: !3, type: !9)
!12 = !DILocalVariable(name: "omp_proc_bind_true", scope: !5, file: !3, type: !9)
!13 = !DILocalVariable(name: "omp_lock_hint_none", scope: !5, file: !3, type: !9)
!14 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !5, file: !3, type: !9)
!15 = !DILocation(line: 61, column: 1, scope: !5)
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
!31 = !DILocation(line: 21, column: 1, scope: !5)
!32 = !DILocalVariable(name: "argcount", scope: !5, file: !3, type: !9)
!33 = !DILocation(line: 22, column: 1, scope: !5)
!34 = !DILocation(line: 23, column: 1, scope: !5)
!35 = !DILocalVariable(scope: !5, file: !3, type: !9, flags: DIFlagArtificial)
!36 = !DILocalVariable(scope: !5, file: !3, type: !23, flags: DIFlagArtificial)
!37 = !DILocation(line: 26, column: 1, scope: !5)
!38 = !DILocalVariable(name: "allocstatus", scope: !5, file: !3, type: !9)
!39 = !DILocation(line: 27, column: 1, scope: !5)
!40 = !DILocation(line: 28, column: 1, scope: !5)
!41 = !DILocation(line: 29, column: 1, scope: !5)
!42 = !DILocation(line: 32, column: 1, scope: !5)
!43 = !DILocalVariable(name: "ix", scope: !5, file: !3, type: !9)
!44 = !DILocation(line: 33, column: 1, scope: !5)
!45 = !DILocation(line: 34, column: 1, scope: !5)
!46 = !DILocation(line: 36, column: 1, scope: !5)
!47 = !DILocation(line: 37, column: 1, scope: !5)
!48 = !DILocalVariable(name: "rderr", scope: !5, file: !3, type: !9)
!49 = !DILocation(line: 38, column: 1, scope: !5)
!50 = !DILocation(line: 39, column: 1, scope: !5)
!51 = !DILocation(line: 43, column: 1, scope: !5)
!52 = !DILocation(line: 45, column: 1, scope: !5)
!53 = !DILocalVariable(name: "i", scope: !5, file: !3, type: !9)
!54 = !DILocation(line: 46, column: 1, scope: !5)
!55 = !DILocation(line: 47, column: 1, scope: !5)
!56 = !DILocation(line: 49, column: 1, scope: !5)
!57 = !DILocalVariable(name: "tmp", scope: !5, file: !3, type: !9)
!58 = !DILocation(line: 56, column: 1, scope: !5)
!59 = !DILocation(line: 59, column: 1, scope: !5)
!60 = distinct !DISubprogram(name: "__nv_MAIN__F1L49_1", scope: !2, file: !3, line: 49, type: !61, scopeLine: 49, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!61 = !DISubroutineType(types: !62)
!62 = !{null, !9, !23, !23}
!63 = !DILocalVariable(name: "__nv_MAIN__F1L49_1Arg0", arg: 1, scope: !60, file: !3, type: !9)
!64 = !DILocation(line: 0, scope: !60)
!65 = !DILocalVariable(name: "__nv_MAIN__F1L49_1Arg1", arg: 2, scope: !60, file: !3, type: !23)
!66 = !DILocalVariable(name: "__nv_MAIN__F1L49_1Arg2", arg: 3, scope: !60, file: !3, type: !23)
!67 = !DILocalVariable(name: "omp_sched_static", scope: !60, file: !3, type: !9)
!68 = !DILocalVariable(name: "omp_proc_bind_false", scope: !60, file: !3, type: !9)
!69 = !DILocalVariable(name: "omp_proc_bind_true", scope: !60, file: !3, type: !9)
!70 = !DILocalVariable(name: "omp_lock_hint_none", scope: !60, file: !3, type: !9)
!71 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !60, file: !3, type: !9)
!72 = !DILocation(line: 53, column: 1, scope: !60)
!73 = !DILocation(line: 50, column: 1, scope: !60)
!74 = !DILocalVariable(name: "i", scope: !60, file: !3, type: !9)
!75 = !DILocation(line: 51, column: 1, scope: !60)
!76 = !DILocation(line: 52, column: 1, scope: !60)

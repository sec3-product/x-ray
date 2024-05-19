; ModuleID = '/tmp/DRB010-lastprivatemissing-var-yes-98084f.ll'
source_filename = "/tmp/DRB010-lastprivatemissing-var-yes-98084f.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%astruct.dt78 = type <{ i8*, i8* }>

@.C353_MAIN_ = internal constant [3 x i8] c"x ="
@.C352_MAIN_ = internal constant i32 47
@.C345_MAIN_ = internal constant [29 x i8] c"Error, invalid integer value."
@.C344_MAIN_ = internal constant i32 38
@.C306_MAIN_ = internal constant i32 25
@.C340_MAIN_ = internal constant [5 x i8] c"(i10)"
@.C339_MAIN_ = internal constant i32 36
@.C361_MAIN_ = internal constant i64 4
@.C337_MAIN_ = internal constant [37 x i8] c"Allocation error, program terminated."
@.C307_MAIN_ = internal constant i32 27
@.C365_MAIN_ = internal constant i64 80
@.C364_MAIN_ = internal constant i64 14
@.C329_MAIN_ = internal constant [35 x i8] c"No command line arguments provided."
@.C327_MAIN_ = internal constant i32 6
@.C328_MAIN_ = internal constant [3 x i8] c"(a)"
@.C285_MAIN_ = internal constant i32 1
@.C305_MAIN_ = internal constant i32 14
@.C325_MAIN_ = internal constant [62 x i8] c"micro-benchmarks-fortran/DRB010-lastprivatemissing-var-yes.f95"
@.C308_MAIN_ = internal constant i32 22
@.C321_MAIN_ = internal constant i32 10000
@.C286_MAIN_ = internal constant i64 1
@.C284_MAIN_ = internal constant i64 0
@.C283_MAIN_ = internal constant i32 0
@.C285___nv_MAIN__F1L42_1 = internal constant i32 1
@.C283___nv_MAIN__F1L42_1 = internal constant i32 0

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__459 = alloca i32, align 4
  %.Z0971_336 = alloca [80 x i8]*, align 8
  %"args$sd1_363" = alloca [16 x i64], align 8
  %len_322 = alloca i32, align 4
  %argcount_310 = alloca i32, align 4
  %z__io_331 = alloca i32, align 4
  %z_b_0_315 = alloca i64, align 8
  %z_b_1_316 = alloca i64, align 8
  %z_e_61_319 = alloca i64, align 8
  %z_b_2_317 = alloca i64, align 8
  %z_b_3_318 = alloca i64, align 8
  %allocstatus_311 = alloca i32, align 4
  %.dY0001_376 = alloca i32, align 4
  %ix_314 = alloca i32, align 4
  %rderr_312 = alloca i32, align 4
  %.uplevelArgPack0001_451 = alloca %astruct.dt78, align 16
  %x_313 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !14, metadata !DIExpression()), !dbg !10
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !15
  store i32 %0, i32* %__gtid_MAIN__459, align 4, !dbg !15
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !16
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !16
  call void (i8*, ...) %2(i8* %1), !dbg !16
  call void @llvm.dbg.declare(metadata [80 x i8]** %.Z0971_336, metadata !17, metadata !DIExpression(DW_OP_deref)), !dbg !10
  %3 = bitcast [80 x i8]** %.Z0971_336 to i8**, !dbg !16
  store i8* null, i8** %3, align 8, !dbg !16
  call void @llvm.dbg.declare(metadata [16 x i64]* %"args$sd1_363", metadata !22, metadata !DIExpression()), !dbg !10
  %4 = bitcast [16 x i64]* %"args$sd1_363" to i64*, !dbg !16
  store i64 0, i64* %4, align 8, !dbg !16
  br label %L.LB1_398

L.LB1_398:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %len_322, metadata !27, metadata !DIExpression()), !dbg !10
  store i32 10000, i32* %len_322, align 4, !dbg !28
  %5 = call i32 (...) @f90_cmd_arg_cnt(), !dbg !29
  call void @llvm.dbg.declare(metadata i32* %argcount_310, metadata !30, metadata !DIExpression()), !dbg !10
  store i32 %5, i32* %argcount_310, align 4, !dbg !29
  %6 = load i32, i32* %argcount_310, align 4, !dbg !31
  call void @llvm.dbg.value(metadata i32 %6, metadata !30, metadata !DIExpression()), !dbg !10
  %7 = icmp ne i32 %6, 0, !dbg !31
  br i1 %7, label %L.LB1_370, label %L.LB1_479, !dbg !31

L.LB1_479:                                        ; preds = %L.LB1_398
  call void (...) @_mp_bcs_nest(), !dbg !32
  %8 = bitcast i32* @.C308_MAIN_ to i8*, !dbg !32
  %9 = bitcast [62 x i8]* @.C325_MAIN_ to i8*, !dbg !32
  %10 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !32
  call void (i8*, i8*, i64, ...) %10(i8* %8, i8* %9, i64 62), !dbg !32
  %11 = bitcast i32* @.C305_MAIN_ to i8*, !dbg !32
  %12 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !32
  %13 = bitcast [3 x i8]* @.C328_MAIN_ to i8*, !dbg !32
  %14 = bitcast i32 (...)* @f90io_encode_fmta to i32 (i8*, i8*, i8*, i64, ...)*, !dbg !32
  %15 = call i32 (i8*, i8*, i8*, i64, ...) %14(i8* %11, i8* %12, i8* %13, i64 3), !dbg !32
  call void @llvm.dbg.declare(metadata i32* %z__io_331, metadata !33, metadata !DIExpression()), !dbg !10
  store i32 %15, i32* %z__io_331, align 4, !dbg !32
  %16 = bitcast i32* @.C327_MAIN_ to i8*, !dbg !32
  %17 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !32
  %18 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !32
  %19 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !32
  %20 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %19(i8* %16, i8* null, i8* %17, i8* %18, i8* null, i8* null, i64 0), !dbg !32
  store i32 %20, i32* %z__io_331, align 4, !dbg !32
  %21 = bitcast i32* @.C305_MAIN_ to i8*, !dbg !32
  %22 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !32
  %23 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !32
  %24 = bitcast [35 x i8]* @.C329_MAIN_ to i8*, !dbg !32
  %25 = bitcast i32 (...)* @f90io_fmt_writea to i32 (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !32
  %26 = call i32 (i8*, i8*, i8*, i8*, i64, ...) %25(i8* %21, i8* %22, i8* %23, i8* %24, i64 35), !dbg !32
  store i32 %26, i32* %z__io_331, align 4, !dbg !32
  %27 = call i32 (...) @f90io_fmtw_end(), !dbg !32
  store i32 %27, i32* %z__io_331, align 4, !dbg !32
  call void (...) @_mp_ecs_nest(), !dbg !32
  br label %L.LB1_370

L.LB1_370:                                        ; preds = %L.LB1_479, %L.LB1_398
  call void @llvm.dbg.declare(metadata i64* %z_b_0_315, metadata !34, metadata !DIExpression()), !dbg !10
  store i64 1, i64* %z_b_0_315, align 8, !dbg !35
  %28 = load i32, i32* %argcount_310, align 4, !dbg !35
  call void @llvm.dbg.value(metadata i32 %28, metadata !30, metadata !DIExpression()), !dbg !10
  %29 = sext i32 %28 to i64, !dbg !35
  call void @llvm.dbg.declare(metadata i64* %z_b_1_316, metadata !34, metadata !DIExpression()), !dbg !10
  store i64 %29, i64* %z_b_1_316, align 8, !dbg !35
  %30 = load i64, i64* %z_b_1_316, align 8, !dbg !35
  call void @llvm.dbg.value(metadata i64 %30, metadata !34, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_e_61_319, metadata !34, metadata !DIExpression()), !dbg !10
  store i64 %30, i64* %z_e_61_319, align 8, !dbg !35
  %31 = bitcast [16 x i64]* %"args$sd1_363" to i8*, !dbg !35
  %32 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !35
  %33 = bitcast i64* @.C364_MAIN_ to i8*, !dbg !35
  %34 = bitcast i64* @.C365_MAIN_ to i8*, !dbg !35
  %35 = bitcast i64* %z_b_0_315 to i8*, !dbg !35
  %36 = bitcast i64* %z_b_1_316 to i8*, !dbg !35
  %37 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !35
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %37(i8* %31, i8* %32, i8* %33, i8* %34, i8* %35, i8* %36), !dbg !35
  %38 = bitcast [16 x i64]* %"args$sd1_363" to i8*, !dbg !35
  %39 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !35
  call void (i8*, i32, ...) %39(i8* %38, i32 14), !dbg !35
  %40 = load i64, i64* %z_b_1_316, align 8, !dbg !35
  call void @llvm.dbg.value(metadata i64 %40, metadata !34, metadata !DIExpression()), !dbg !10
  %41 = load i64, i64* %z_b_0_315, align 8, !dbg !35
  call void @llvm.dbg.value(metadata i64 %41, metadata !34, metadata !DIExpression()), !dbg !10
  %42 = sub nsw i64 %41, 1, !dbg !35
  %43 = sub nsw i64 %40, %42, !dbg !35
  call void @llvm.dbg.declare(metadata i64* %z_b_2_317, metadata !34, metadata !DIExpression()), !dbg !10
  store i64 %43, i64* %z_b_2_317, align 8, !dbg !35
  %44 = load i64, i64* %z_b_0_315, align 8, !dbg !35
  call void @llvm.dbg.value(metadata i64 %44, metadata !34, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i64* %z_b_3_318, metadata !34, metadata !DIExpression()), !dbg !10
  store i64 %44, i64* %z_b_3_318, align 8, !dbg !35
  %45 = bitcast i64* %z_b_2_317 to i8*, !dbg !35
  %46 = bitcast i64* @.C364_MAIN_ to i8*, !dbg !35
  %47 = bitcast i64* @.C365_MAIN_ to i8*, !dbg !35
  call void @llvm.dbg.declare(metadata i32* %allocstatus_311, metadata !36, metadata !DIExpression()), !dbg !10
  %48 = bitcast i32* %allocstatus_311 to i8*, !dbg !35
  %49 = bitcast [80 x i8]** %.Z0971_336 to i8*, !dbg !35
  %50 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !35
  %51 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !35
  %52 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !35
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %52(i8* %45, i8* %46, i8* %47, i8* %48, i8* %49, i8* null, i8* %50, i8* %51, i8* null, i64 0), !dbg !35
  %53 = load i32, i32* %allocstatus_311, align 4, !dbg !37
  call void @llvm.dbg.value(metadata i32 %53, metadata !36, metadata !DIExpression()), !dbg !10
  %54 = icmp sle i32 %53, 0, !dbg !37
  br i1 %54, label %L.LB1_373, label %L.LB1_480, !dbg !37

L.LB1_480:                                        ; preds = %L.LB1_370
  call void (...) @_mp_bcs_nest(), !dbg !38
  %55 = bitcast i32* @.C307_MAIN_ to i8*, !dbg !38
  %56 = bitcast [62 x i8]* @.C325_MAIN_ to i8*, !dbg !38
  %57 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !38
  call void (i8*, i8*, i64, ...) %57(i8* %55, i8* %56, i64 62), !dbg !38
  %58 = bitcast i32* @.C305_MAIN_ to i8*, !dbg !38
  %59 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !38
  %60 = bitcast [3 x i8]* @.C328_MAIN_ to i8*, !dbg !38
  %61 = bitcast i32 (...)* @f90io_encode_fmta to i32 (i8*, i8*, i8*, i64, ...)*, !dbg !38
  %62 = call i32 (i8*, i8*, i8*, i64, ...) %61(i8* %58, i8* %59, i8* %60, i64 3), !dbg !38
  store i32 %62, i32* %z__io_331, align 4, !dbg !38
  %63 = bitcast i32* @.C327_MAIN_ to i8*, !dbg !38
  %64 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !38
  %65 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !38
  %66 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !38
  %67 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %66(i8* %63, i8* null, i8* %64, i8* %65, i8* null, i8* null, i64 0), !dbg !38
  store i32 %67, i32* %z__io_331, align 4, !dbg !38
  %68 = bitcast i32* @.C305_MAIN_ to i8*, !dbg !38
  %69 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !38
  %70 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !38
  %71 = bitcast [37 x i8]* @.C337_MAIN_ to i8*, !dbg !38
  %72 = bitcast i32 (...)* @f90io_fmt_writea to i32 (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !38
  %73 = call i32 (i8*, i8*, i8*, i8*, i64, ...) %72(i8* %68, i8* %69, i8* %70, i8* %71, i64 37), !dbg !38
  store i32 %73, i32* %z__io_331, align 4, !dbg !38
  %74 = call i32 (...) @f90io_fmtw_end(), !dbg !38
  store i32 %74, i32* %z__io_331, align 4, !dbg !38
  call void (...) @_mp_ecs_nest(), !dbg !38
  %75 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !39
  %76 = bitcast void (...)* @f90_stop08a to void (i8*, i8*, i64, ...)*, !dbg !39
  call void (i8*, i8*, i64, ...) %76(i8* %75, i8* null, i64 0), !dbg !39
  br label %L.LB1_373

L.LB1_373:                                        ; preds = %L.LB1_480, %L.LB1_370
  %77 = load i32, i32* %argcount_310, align 4, !dbg !40
  call void @llvm.dbg.value(metadata i32 %77, metadata !30, metadata !DIExpression()), !dbg !10
  store i32 %77, i32* %.dY0001_376, align 4, !dbg !40
  call void @llvm.dbg.declare(metadata i32* %ix_314, metadata !41, metadata !DIExpression()), !dbg !10
  store i32 1, i32* %ix_314, align 4, !dbg !40
  %78 = load i32, i32* %.dY0001_376, align 4, !dbg !40
  %79 = icmp sle i32 %78, 0, !dbg !40
  br i1 %79, label %L.LB1_375, label %L.LB1_374, !dbg !40

L.LB1_374:                                        ; preds = %L.LB1_374, %L.LB1_373
  %80 = bitcast i32* %ix_314 to i8*, !dbg !42
  %81 = load [80 x i8]*, [80 x i8]** %.Z0971_336, align 8, !dbg !42
  call void @llvm.dbg.value(metadata [80 x i8]* %81, metadata !17, metadata !DIExpression()), !dbg !10
  %82 = bitcast [80 x i8]* %81 to i8*, !dbg !42
  %83 = getelementptr i8, i8* %82, i64 -80, !dbg !42
  %84 = load i32, i32* %ix_314, align 4, !dbg !42
  call void @llvm.dbg.value(metadata i32 %84, metadata !41, metadata !DIExpression()), !dbg !10
  %85 = sext i32 %84 to i64, !dbg !42
  %86 = bitcast [16 x i64]* %"args$sd1_363" to i8*, !dbg !42
  %87 = getelementptr i8, i8* %86, i64 56, !dbg !42
  %88 = bitcast i8* %87 to i64*, !dbg !42
  %89 = load i64, i64* %88, align 8, !dbg !42
  %90 = add nsw i64 %85, %89, !dbg !42
  %91 = mul nsw i64 %90, 80, !dbg !42
  %92 = getelementptr i8, i8* %83, i64 %91, !dbg !42
  %93 = bitcast i64* @.C361_MAIN_ to i8*, !dbg !42
  %94 = bitcast void (...)* @f90_get_cmd_arga to void (i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !42
  call void (i8*, i8*, i8*, i8*, i8*, i64, ...) %94(i8* %80, i8* %92, i8* null, i8* null, i8* %93, i64 80), !dbg !42
  %95 = load i32, i32* %ix_314, align 4, !dbg !43
  call void @llvm.dbg.value(metadata i32 %95, metadata !41, metadata !DIExpression()), !dbg !10
  %96 = add nsw i32 %95, 1, !dbg !43
  store i32 %96, i32* %ix_314, align 4, !dbg !43
  %97 = load i32, i32* %.dY0001_376, align 4, !dbg !43
  %98 = sub nsw i32 %97, 1, !dbg !43
  store i32 %98, i32* %.dY0001_376, align 4, !dbg !43
  %99 = load i32, i32* %.dY0001_376, align 4, !dbg !43
  %100 = icmp sgt i32 %99, 0, !dbg !43
  br i1 %100, label %L.LB1_374, label %L.LB1_375, !dbg !43

L.LB1_375:                                        ; preds = %L.LB1_374, %L.LB1_373
  %101 = load i32, i32* %argcount_310, align 4, !dbg !44
  call void @llvm.dbg.value(metadata i32 %101, metadata !30, metadata !DIExpression()), !dbg !10
  %102 = icmp sle i32 %101, 0, !dbg !44
  br i1 %102, label %L.LB1_377, label %L.LB1_481, !dbg !44

L.LB1_481:                                        ; preds = %L.LB1_375
  call void (...) @_mp_bcs_nest(), !dbg !45
  %103 = bitcast i32* @.C339_MAIN_ to i8*, !dbg !45
  %104 = bitcast [62 x i8]* @.C325_MAIN_ to i8*, !dbg !45
  %105 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !45
  call void (i8*, i8*, i64, ...) %105(i8* %103, i8* %104, i64 62), !dbg !45
  %106 = bitcast i32* @.C305_MAIN_ to i8*, !dbg !45
  %107 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !45
  %108 = bitcast [5 x i8]* @.C340_MAIN_ to i8*, !dbg !45
  %109 = bitcast i32 (...)* @f90io_encode_fmta to i32 (i8*, i8*, i8*, i64, ...)*, !dbg !45
  %110 = call i32 (i8*, i8*, i8*, i64, ...) %109(i8* %106, i8* %107, i8* %108, i64 5), !dbg !45
  store i32 %110, i32* %z__io_331, align 4, !dbg !45
  %111 = load [80 x i8]*, [80 x i8]** %.Z0971_336, align 8, !dbg !45
  call void @llvm.dbg.value(metadata [80 x i8]* %111, metadata !17, metadata !DIExpression()), !dbg !10
  %112 = bitcast [80 x i8]* %111 to i8*, !dbg !45
  %113 = bitcast [16 x i64]* %"args$sd1_363" to i8*, !dbg !45
  %114 = getelementptr i8, i8* %113, i64 56, !dbg !45
  %115 = bitcast i8* %114 to i64*, !dbg !45
  %116 = load i64, i64* %115, align 8, !dbg !45
  %117 = mul nsw i64 %116, 80, !dbg !45
  %118 = getelementptr i8, i8* %112, i64 %117, !dbg !45
  %119 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !45
  %120 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !45
  call void @llvm.dbg.declare(metadata i32* %rderr_312, metadata !46, metadata !DIExpression()), !dbg !10
  %121 = bitcast i32* %rderr_312 to i8*, !dbg !45
  %122 = bitcast i32 (...)* @f90io_fmtr_intern_inita to i32 (i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !45
  %123 = call i32 (i8*, i8*, i8*, i8*, i8*, i64, ...) %122(i8* %118, i8* %119, i8* %120, i8* %121, i8* null, i64 80), !dbg !45
  store i32 %123, i32* %z__io_331, align 4, !dbg !45
  %124 = bitcast i32* @.C306_MAIN_ to i8*, !dbg !45
  %125 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !45
  %126 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !45
  %127 = bitcast i32* %len_322 to i8*, !dbg !45
  %128 = bitcast i32 (...)* @f90io_fmt_reada to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !45
  %129 = call i32 (i8*, i8*, i8*, i8*, ...) %128(i8* %124, i8* %125, i8* %126, i8* %127), !dbg !45
  store i32 %129, i32* %z__io_331, align 4, !dbg !45
  %130 = call i32 (...) @f90io_fmtr_end(), !dbg !45
  store i32 %130, i32* %z__io_331, align 4, !dbg !45
  call void (...) @_mp_ecs_nest(), !dbg !45
  %131 = load i32, i32* %rderr_312, align 4, !dbg !47
  call void @llvm.dbg.value(metadata i32 %131, metadata !46, metadata !DIExpression()), !dbg !10
  %132 = icmp eq i32 %131, 0, !dbg !47
  br i1 %132, label %L.LB1_378, label %L.LB1_482, !dbg !47

L.LB1_482:                                        ; preds = %L.LB1_481
  call void (...) @_mp_bcs_nest(), !dbg !48
  %133 = bitcast i32* @.C344_MAIN_ to i8*, !dbg !48
  %134 = bitcast [62 x i8]* @.C325_MAIN_ to i8*, !dbg !48
  %135 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !48
  call void (i8*, i8*, i64, ...) %135(i8* %133, i8* %134, i64 62), !dbg !48
  %136 = bitcast i32* @.C305_MAIN_ to i8*, !dbg !48
  %137 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !48
  %138 = bitcast [3 x i8]* @.C328_MAIN_ to i8*, !dbg !48
  %139 = bitcast i32 (...)* @f90io_encode_fmta to i32 (i8*, i8*, i8*, i64, ...)*, !dbg !48
  %140 = call i32 (i8*, i8*, i8*, i64, ...) %139(i8* %136, i8* %137, i8* %138, i64 3), !dbg !48
  store i32 %140, i32* %z__io_331, align 4, !dbg !48
  %141 = bitcast i32* @.C327_MAIN_ to i8*, !dbg !48
  %142 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !48
  %143 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !48
  %144 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !48
  %145 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %144(i8* %141, i8* null, i8* %142, i8* %143, i8* null, i8* null, i64 0), !dbg !48
  store i32 %145, i32* %z__io_331, align 4, !dbg !48
  %146 = bitcast i32* @.C305_MAIN_ to i8*, !dbg !48
  %147 = bitcast i32* @.C285_MAIN_ to i8*, !dbg !48
  %148 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !48
  %149 = bitcast [29 x i8]* @.C345_MAIN_ to i8*, !dbg !48
  %150 = bitcast i32 (...)* @f90io_fmt_writea to i32 (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !48
  %151 = call i32 (i8*, i8*, i8*, i8*, i64, ...) %150(i8* %146, i8* %147, i8* %148, i8* %149, i64 29), !dbg !48
  store i32 %151, i32* %z__io_331, align 4, !dbg !48
  %152 = call i32 (...) @f90io_fmtw_end(), !dbg !48
  store i32 %152, i32* %z__io_331, align 4, !dbg !48
  call void (...) @_mp_ecs_nest(), !dbg !48
  br label %L.LB1_378

L.LB1_378:                                        ; preds = %L.LB1_482, %L.LB1_481
  br label %L.LB1_377

L.LB1_377:                                        ; preds = %L.LB1_378, %L.LB1_375
  %153 = bitcast i32* %len_322 to i8*, !dbg !49
  %154 = bitcast %astruct.dt78* %.uplevelArgPack0001_451 to i8**, !dbg !49
  store i8* %153, i8** %154, align 8, !dbg !49
  call void @llvm.dbg.declare(metadata i32* %x_313, metadata !50, metadata !DIExpression()), !dbg !10
  %155 = bitcast i32* %x_313 to i8*, !dbg !49
  %156 = bitcast %astruct.dt78* %.uplevelArgPack0001_451 to i8*, !dbg !49
  %157 = getelementptr i8, i8* %156, i64 8, !dbg !49
  %158 = bitcast i8* %157 to i8**, !dbg !49
  store i8* %155, i8** %158, align 8, !dbg !49
  br label %L.LB1_457, !dbg !49

L.LB1_457:                                        ; preds = %L.LB1_377
  %159 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L42_1_ to i64*, !dbg !49
  %160 = bitcast %astruct.dt78* %.uplevelArgPack0001_451 to i64*, !dbg !49
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %159, i64* %160), !dbg !49
  call void (...) @_mp_bcs_nest(), !dbg !51
  %161 = bitcast i32* @.C352_MAIN_ to i8*, !dbg !51
  %162 = bitcast [62 x i8]* @.C325_MAIN_ to i8*, !dbg !51
  %163 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !51
  call void (i8*, i8*, i64, ...) %163(i8* %161, i8* %162, i64 62), !dbg !51
  %164 = bitcast i32* @.C327_MAIN_ to i8*, !dbg !51
  %165 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !51
  %166 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !51
  %167 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !51
  %168 = call i32 (i8*, i8*, i8*, i8*, ...) %167(i8* %164, i8* null, i8* %165, i8* %166), !dbg !51
  store i32 %168, i32* %z__io_331, align 4, !dbg !51
  %169 = bitcast [3 x i8]* @.C353_MAIN_ to i8*, !dbg !51
  %170 = bitcast i32 (...)* @f90io_sc_ch_ldw to i32 (i8*, i32, i64, ...)*, !dbg !51
  %171 = call i32 (i8*, i32, i64, ...) %170(i8* %169, i32 14, i64 3), !dbg !51
  store i32 %171, i32* %z__io_331, align 4, !dbg !51
  %172 = load i32, i32* %x_313, align 4, !dbg !51
  call void @llvm.dbg.value(metadata i32 %172, metadata !50, metadata !DIExpression()), !dbg !10
  %173 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !51
  %174 = call i32 (i32, i32, ...) %173(i32 %172, i32 25), !dbg !51
  store i32 %174, i32* %z__io_331, align 4, !dbg !51
  %175 = call i32 (...) @f90io_ldw_end(), !dbg !51
  store i32 %175, i32* %z__io_331, align 4, !dbg !51
  call void (...) @_mp_ecs_nest(), !dbg !51
  %176 = load [80 x i8]*, [80 x i8]** %.Z0971_336, align 8, !dbg !52
  call void @llvm.dbg.value(metadata [80 x i8]* %176, metadata !17, metadata !DIExpression()), !dbg !10
  %177 = bitcast [80 x i8]* %176 to i8*, !dbg !52
  %178 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !52
  %179 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, i64, ...)*, !dbg !52
  call void (i8*, i8*, i8*, i8*, i64, i64, ...) %179(i8* null, i8* %177, i8* %178, i8* null, i64 80, i64 0), !dbg !52
  %180 = bitcast [80 x i8]** %.Z0971_336 to i8**, !dbg !52
  store i8* null, i8** %180, align 8, !dbg !52
  %181 = bitcast [16 x i64]* %"args$sd1_363" to i64*, !dbg !52
  store i64 0, i64* %181, align 8, !dbg !52
  ret void, !dbg !15
}

define internal void @__nv_MAIN__F1L42_1_(i32* %__nv_MAIN__F1L42_1Arg0, i64* %__nv_MAIN__F1L42_1Arg1, i64* %__nv_MAIN__F1L42_1Arg2) #0 !dbg !53 {
L.entry:
  %__gtid___nv_MAIN__F1L42_1__501 = alloca i32, align 4
  %.i0000p_350 = alloca i32, align 4
  %i_349 = alloca i32, align 4
  %.du0002p_382 = alloca i32, align 4
  %.de0002p_383 = alloca i32, align 4
  %.di0002p_384 = alloca i32, align 4
  %.ds0002p_385 = alloca i32, align 4
  %.dl0002p_387 = alloca i32, align 4
  %.dl0002p.copy_495 = alloca i32, align 4
  %.de0002p.copy_496 = alloca i32, align 4
  %.ds0002p.copy_497 = alloca i32, align 4
  %.dX0002p_386 = alloca i32, align 4
  %.dY0002p_381 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L42_1Arg0, metadata !56, metadata !DIExpression()), !dbg !57
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L42_1Arg1, metadata !58, metadata !DIExpression()), !dbg !57
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L42_1Arg2, metadata !59, metadata !DIExpression()), !dbg !57
  call void @llvm.dbg.value(metadata i32 1, metadata !60, metadata !DIExpression()), !dbg !57
  call void @llvm.dbg.value(metadata i32 0, metadata !61, metadata !DIExpression()), !dbg !57
  call void @llvm.dbg.value(metadata i32 1, metadata !62, metadata !DIExpression()), !dbg !57
  call void @llvm.dbg.value(metadata i32 0, metadata !63, metadata !DIExpression()), !dbg !57
  call void @llvm.dbg.value(metadata i32 1, metadata !64, metadata !DIExpression()), !dbg !57
  %0 = load i32, i32* %__nv_MAIN__F1L42_1Arg0, align 4, !dbg !65
  store i32 %0, i32* %__gtid___nv_MAIN__F1L42_1__501, align 4, !dbg !65
  br label %L.LB2_486

L.LB2_486:                                        ; preds = %L.entry
  br label %L.LB2_348

L.LB2_348:                                        ; preds = %L.LB2_486
  store i32 0, i32* %.i0000p_350, align 4, !dbg !66
  call void @llvm.dbg.declare(metadata i32* %i_349, metadata !67, metadata !DIExpression()), !dbg !65
  store i32 0, i32* %i_349, align 4, !dbg !66
  %1 = bitcast i64* %__nv_MAIN__F1L42_1Arg2 to i32**, !dbg !66
  %2 = load i32*, i32** %1, align 8, !dbg !66
  %3 = load i32, i32* %2, align 4, !dbg !66
  store i32 %3, i32* %.du0002p_382, align 4, !dbg !66
  %4 = bitcast i64* %__nv_MAIN__F1L42_1Arg2 to i32**, !dbg !66
  %5 = load i32*, i32** %4, align 8, !dbg !66
  %6 = load i32, i32* %5, align 4, !dbg !66
  store i32 %6, i32* %.de0002p_383, align 4, !dbg !66
  store i32 1, i32* %.di0002p_384, align 4, !dbg !66
  %7 = load i32, i32* %.di0002p_384, align 4, !dbg !66
  store i32 %7, i32* %.ds0002p_385, align 4, !dbg !66
  store i32 0, i32* %.dl0002p_387, align 4, !dbg !66
  %8 = load i32, i32* %.dl0002p_387, align 4, !dbg !66
  store i32 %8, i32* %.dl0002p.copy_495, align 4, !dbg !66
  %9 = load i32, i32* %.de0002p_383, align 4, !dbg !66
  store i32 %9, i32* %.de0002p.copy_496, align 4, !dbg !66
  %10 = load i32, i32* %.ds0002p_385, align 4, !dbg !66
  store i32 %10, i32* %.ds0002p.copy_497, align 4, !dbg !66
  %11 = load i32, i32* %__gtid___nv_MAIN__F1L42_1__501, align 4, !dbg !66
  %12 = bitcast i32* %.i0000p_350 to i64*, !dbg !66
  %13 = bitcast i32* %.dl0002p.copy_495 to i64*, !dbg !66
  %14 = bitcast i32* %.de0002p.copy_496 to i64*, !dbg !66
  %15 = bitcast i32* %.ds0002p.copy_497 to i64*, !dbg !66
  %16 = load i32, i32* %.ds0002p.copy_497, align 4, !dbg !66
  call void @__kmpc_for_static_init_4(i64* null, i32 %11, i32 34, i64* %12, i64* %13, i64* %14, i64* %15, i32 %16, i32 1), !dbg !66
  %17 = load i32, i32* %.dl0002p.copy_495, align 4, !dbg !66
  store i32 %17, i32* %.dl0002p_387, align 4, !dbg !66
  %18 = load i32, i32* %.de0002p.copy_496, align 4, !dbg !66
  store i32 %18, i32* %.de0002p_383, align 4, !dbg !66
  %19 = load i32, i32* %.ds0002p.copy_497, align 4, !dbg !66
  store i32 %19, i32* %.ds0002p_385, align 4, !dbg !66
  %20 = load i32, i32* %.dl0002p_387, align 4, !dbg !66
  store i32 %20, i32* %i_349, align 4, !dbg !66
  %21 = load i32, i32* %i_349, align 4, !dbg !66
  call void @llvm.dbg.value(metadata i32 %21, metadata !67, metadata !DIExpression()), !dbg !65
  store i32 %21, i32* %.dX0002p_386, align 4, !dbg !66
  %22 = load i32, i32* %.dX0002p_386, align 4, !dbg !66
  %23 = load i32, i32* %.du0002p_382, align 4, !dbg !66
  %24 = icmp sgt i32 %22, %23, !dbg !66
  br i1 %24, label %L.LB2_380, label %L.LB2_524, !dbg !66

L.LB2_524:                                        ; preds = %L.LB2_348
  %25 = load i32, i32* %.dX0002p_386, align 4, !dbg !66
  store i32 %25, i32* %i_349, align 4, !dbg !66
  %26 = load i32, i32* %.di0002p_384, align 4, !dbg !66
  %27 = load i32, i32* %.de0002p_383, align 4, !dbg !66
  %28 = load i32, i32* %.dX0002p_386, align 4, !dbg !66
  %29 = sub nsw i32 %27, %28, !dbg !66
  %30 = add nsw i32 %26, %29, !dbg !66
  %31 = load i32, i32* %.di0002p_384, align 4, !dbg !66
  %32 = sdiv i32 %30, %31, !dbg !66
  store i32 %32, i32* %.dY0002p_381, align 4, !dbg !66
  %33 = load i32, i32* %.dY0002p_381, align 4, !dbg !66
  %34 = icmp sle i32 %33, 0, !dbg !66
  br i1 %34, label %L.LB2_390, label %L.LB2_389, !dbg !66

L.LB2_389:                                        ; preds = %L.LB2_389, %L.LB2_524
  %35 = load i32, i32* %i_349, align 4, !dbg !68
  call void @llvm.dbg.value(metadata i32 %35, metadata !67, metadata !DIExpression()), !dbg !65
  %36 = bitcast i64* %__nv_MAIN__F1L42_1Arg2 to i8*, !dbg !68
  %37 = getelementptr i8, i8* %36, i64 8, !dbg !68
  %38 = bitcast i8* %37 to i32**, !dbg !68
  %39 = load i32*, i32** %38, align 8, !dbg !68
  store i32 %35, i32* %39, align 4, !dbg !68
  %40 = load i32, i32* %.di0002p_384, align 4, !dbg !65
  %41 = load i32, i32* %i_349, align 4, !dbg !65
  call void @llvm.dbg.value(metadata i32 %41, metadata !67, metadata !DIExpression()), !dbg !65
  %42 = add nsw i32 %40, %41, !dbg !65
  store i32 %42, i32* %i_349, align 4, !dbg !65
  %43 = load i32, i32* %.dY0002p_381, align 4, !dbg !65
  %44 = sub nsw i32 %43, 1, !dbg !65
  store i32 %44, i32* %.dY0002p_381, align 4, !dbg !65
  %45 = load i32, i32* %.dY0002p_381, align 4, !dbg !65
  %46 = icmp sgt i32 %45, 0, !dbg !65
  br i1 %46, label %L.LB2_389, label %L.LB2_390, !dbg !65

L.LB2_390:                                        ; preds = %L.LB2_389, %L.LB2_524
  br label %L.LB2_380

L.LB2_380:                                        ; preds = %L.LB2_390, %L.LB2_348
  %47 = load i32, i32* %__gtid___nv_MAIN__F1L42_1__501, align 4, !dbg !65
  call void @__kmpc_for_static_fini(i64* null, i32 %47), !dbg !65
  br label %L.LB2_351

L.LB2_351:                                        ; preds = %L.LB2_380
  ret void, !dbg !65
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
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB010-lastprivatemissing-var-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb010_lastprivatemissing_var_yes", scope: !2, file: !3, line: 12, type: !6, scopeLine: 12, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!6 = !DISubroutineType(cc: DW_CC_program, types: !7)
!7 = !{null}
!8 = !DILocalVariable(name: "omp_sched_static", scope: !5, file: !3, type: !9)
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 0, scope: !5)
!11 = !DILocalVariable(name: "omp_proc_bind_false", scope: !5, file: !3, type: !9)
!12 = !DILocalVariable(name: "omp_proc_bind_true", scope: !5, file: !3, type: !9)
!13 = !DILocalVariable(name: "omp_lock_hint_none", scope: !5, file: !3, type: !9)
!14 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !5, file: !3, type: !9)
!15 = !DILocation(line: 50, column: 1, scope: !5)
!16 = !DILocation(line: 12, column: 1, scope: !5)
!17 = !DILocalVariable(name: "args", scope: !5, file: !3, type: !18)
!18 = !DICompositeType(tag: DW_TAG_array_type, baseType: !19, size: 640, align: 8, elements: !20)
!19 = !DIBasicType(name: "character", size: 640, align: 8, encoding: DW_ATE_signed)
!20 = !{!21}
!21 = !DISubrange(count: 0, lowerBound: 1)
!22 = !DILocalVariable(scope: !5, file: !3, type: !23, flags: DIFlagArtificial)
!23 = !DICompositeType(tag: DW_TAG_array_type, baseType: !24, size: 1024, align: 64, elements: !25)
!24 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!25 = !{!26}
!26 = !DISubrange(count: 16, lowerBound: 1)
!27 = !DILocalVariable(name: "len", scope: !5, file: !3, type: !9)
!28 = !DILocation(line: 18, column: 1, scope: !5)
!29 = !DILocation(line: 20, column: 1, scope: !5)
!30 = !DILocalVariable(name: "argcount", scope: !5, file: !3, type: !9)
!31 = !DILocation(line: 21, column: 1, scope: !5)
!32 = !DILocation(line: 22, column: 1, scope: !5)
!33 = !DILocalVariable(scope: !5, file: !3, type: !9, flags: DIFlagArtificial)
!34 = !DILocalVariable(scope: !5, file: !3, type: !24, flags: DIFlagArtificial)
!35 = !DILocation(line: 25, column: 1, scope: !5)
!36 = !DILocalVariable(name: "allocstatus", scope: !5, file: !3, type: !9)
!37 = !DILocation(line: 26, column: 1, scope: !5)
!38 = !DILocation(line: 27, column: 1, scope: !5)
!39 = !DILocation(line: 28, column: 1, scope: !5)
!40 = !DILocation(line: 31, column: 1, scope: !5)
!41 = !DILocalVariable(name: "ix", scope: !5, file: !3, type: !9)
!42 = !DILocation(line: 32, column: 1, scope: !5)
!43 = !DILocation(line: 33, column: 1, scope: !5)
!44 = !DILocation(line: 35, column: 1, scope: !5)
!45 = !DILocation(line: 36, column: 1, scope: !5)
!46 = !DILocalVariable(name: "rderr", scope: !5, file: !3, type: !9)
!47 = !DILocation(line: 37, column: 1, scope: !5)
!48 = !DILocation(line: 38, column: 1, scope: !5)
!49 = !DILocation(line: 42, column: 1, scope: !5)
!50 = !DILocalVariable(name: "x", scope: !5, file: !3, type: !9)
!51 = !DILocation(line: 47, column: 1, scope: !5)
!52 = !DILocation(line: 49, column: 1, scope: !5)
!53 = distinct !DISubprogram(name: "__nv_MAIN__F1L42_1", scope: !2, file: !3, line: 42, type: !54, scopeLine: 42, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!54 = !DISubroutineType(types: !55)
!55 = !{null, !9, !24, !24}
!56 = !DILocalVariable(name: "__nv_MAIN__F1L42_1Arg0", arg: 1, scope: !53, file: !3, type: !9)
!57 = !DILocation(line: 0, scope: !53)
!58 = !DILocalVariable(name: "__nv_MAIN__F1L42_1Arg1", arg: 2, scope: !53, file: !3, type: !24)
!59 = !DILocalVariable(name: "__nv_MAIN__F1L42_1Arg2", arg: 3, scope: !53, file: !3, type: !24)
!60 = !DILocalVariable(name: "omp_sched_static", scope: !53, file: !3, type: !9)
!61 = !DILocalVariable(name: "omp_proc_bind_false", scope: !53, file: !3, type: !9)
!62 = !DILocalVariable(name: "omp_proc_bind_true", scope: !53, file: !3, type: !9)
!63 = !DILocalVariable(name: "omp_lock_hint_none", scope: !53, file: !3, type: !9)
!64 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !53, file: !3, type: !9)
!65 = !DILocation(line: 45, column: 1, scope: !53)
!66 = !DILocation(line: 43, column: 1, scope: !53)
!67 = !DILocalVariable(name: "i", scope: !53, file: !3, type: !9)
!68 = !DILocation(line: 44, column: 1, scope: !53)

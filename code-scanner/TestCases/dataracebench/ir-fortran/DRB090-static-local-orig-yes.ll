; ModuleID = '/tmp/DRB090-static-local-orig-yes-e9f64b.ll'
source_filename = "/tmp/DRB090-static-local-orig-yes-e9f64b.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.BSS1 = type <{ [4 x i8] }>
%struct.STATICS1 = type <{ [60 x i8] }>
%astruct.dt74 = type <{ i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8* }>
%astruct.dt80 = type <{ i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8* }>

@.BSS1 = internal global %struct.BSS1 zeroinitializer, align 32
@.STATICS1 = internal global %struct.STATICS1 <{ [60 x i8] c"\EB\FF\FF\FF\00\00\00\00\03\00\00\00\00\00\00\00\01\00\00\00\F7\FF\FF\FF\00\00\00\00\03\00\00\00\EB\FF\FF\FF\00\00\00\00\03\00\00\00\00\00\00\00\01\00\00\00\FF\FF\FF\FF\00\00\00\00" }>, align 16
@.C349_MAIN_ = internal constant i64 50
@.C346_MAIN_ = internal constant i32 6
@.C342_MAIN_ = internal constant [57 x i8] c"micro-benchmarks-fortran/DRB090-static-local-orig-yes.f95"
@.C344_MAIN_ = internal constant i32 53
@.C285_MAIN_ = internal constant i32 1
@.C305_MAIN_ = internal constant i32 25
@.C358_MAIN_ = internal constant i64 4
@.C357_MAIN_ = internal constant i64 25
@.C321_MAIN_ = internal constant i32 100
@.C286_MAIN_ = internal constant i64 1
@.C284_MAIN_ = internal constant i64 0
@.C283_MAIN_ = internal constant i32 0
@.C285___nv_MAIN__F1L35_1 = internal constant i32 1
@.C283___nv_MAIN__F1L35_1 = internal constant i32 0
@.C285___nv_MAIN__F1L44_2 = internal constant i32 1
@.C283___nv_MAIN__F1L44_2 = internal constant i32 0

define void @MAIN_() #0 !dbg !17 {
L.entry:
  %__gtid_MAIN__453 = alloca i32, align 4
  %.Z0966_324 = alloca i32*, align 8
  %"b$sd2_360" = alloca [16 x i64], align 8
  %.Z0965_323 = alloca i32*, align 8
  %"a$sd1_356" = alloca [16 x i64], align 8
  %len_322 = alloca i32, align 4
  %z_b_0_307 = alloca i64, align 8
  %z_b_1_308 = alloca i64, align 8
  %z_e_60_311 = alloca i64, align 8
  %z_b_2_309 = alloca i64, align 8
  %z_b_3_310 = alloca i64, align 8
  %z_b_4_314 = alloca i64, align 8
  %z_b_5_315 = alloca i64, align 8
  %z_e_67_318 = alloca i64, align 8
  %z_b_6_316 = alloca i64, align 8
  %z_b_7_317 = alloca i64, align 8
  %.dY0001_368 = alloca i32, align 4
  %i_306 = alloca i32, align 4
  %.uplevelArgPack0001_429 = alloca %astruct.dt74, align 16
  %.uplevelArgPack0002_471 = alloca %astruct.dt80, align 16
  %tmp2_320 = alloca i32, align 4
  %z__io_348 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !20, metadata !DIExpression()), !dbg !21
  call void @llvm.dbg.value(metadata i32 0, metadata !22, metadata !DIExpression()), !dbg !21
  call void @llvm.dbg.value(metadata i32 1, metadata !23, metadata !DIExpression()), !dbg !21
  call void @llvm.dbg.value(metadata i32 0, metadata !24, metadata !DIExpression()), !dbg !21
  call void @llvm.dbg.value(metadata i32 1, metadata !25, metadata !DIExpression()), !dbg !21
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !26
  store i32 %0, i32* %__gtid_MAIN__453, align 4, !dbg !26
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !27
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !27
  call void (i8*, ...) %2(i8* %1), !dbg !27
  call void @llvm.dbg.declare(metadata i32** %.Z0966_324, metadata !28, metadata !DIExpression(DW_OP_deref)), !dbg !21
  %3 = bitcast i32** %.Z0966_324 to i8**, !dbg !27
  store i8* null, i8** %3, align 8, !dbg !27
  call void @llvm.dbg.declare(metadata [16 x i64]* %"b$sd2_360", metadata !32, metadata !DIExpression()), !dbg !21
  %4 = bitcast [16 x i64]* %"b$sd2_360" to i64*, !dbg !27
  store i64 0, i64* %4, align 8, !dbg !27
  call void @llvm.dbg.declare(metadata i32** %.Z0965_323, metadata !36, metadata !DIExpression(DW_OP_deref)), !dbg !21
  %5 = bitcast i32** %.Z0965_323 to i8**, !dbg !27
  store i8* null, i8** %5, align 8, !dbg !27
  call void @llvm.dbg.declare(metadata [16 x i64]* %"a$sd1_356", metadata !32, metadata !DIExpression()), !dbg !21
  %6 = bitcast [16 x i64]* %"a$sd1_356" to i64*, !dbg !27
  store i64 0, i64* %6, align 8, !dbg !27
  br label %L.LB1_402

L.LB1_402:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %len_322, metadata !37, metadata !DIExpression()), !dbg !21
  store i32 100, i32* %len_322, align 4, !dbg !38
  call void @llvm.dbg.declare(metadata i64* %z_b_0_307, metadata !39, metadata !DIExpression()), !dbg !21
  store i64 1, i64* %z_b_0_307, align 8, !dbg !40
  %7 = load i32, i32* %len_322, align 4, !dbg !40
  call void @llvm.dbg.value(metadata i32 %7, metadata !37, metadata !DIExpression()), !dbg !21
  %8 = sext i32 %7 to i64, !dbg !40
  call void @llvm.dbg.declare(metadata i64* %z_b_1_308, metadata !39, metadata !DIExpression()), !dbg !21
  store i64 %8, i64* %z_b_1_308, align 8, !dbg !40
  %9 = load i64, i64* %z_b_1_308, align 8, !dbg !40
  call void @llvm.dbg.value(metadata i64 %9, metadata !39, metadata !DIExpression()), !dbg !21
  call void @llvm.dbg.declare(metadata i64* %z_e_60_311, metadata !39, metadata !DIExpression()), !dbg !21
  store i64 %9, i64* %z_e_60_311, align 8, !dbg !40
  %10 = bitcast [16 x i64]* %"a$sd1_356" to i8*, !dbg !40
  %11 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !40
  %12 = bitcast i64* @.C357_MAIN_ to i8*, !dbg !40
  %13 = bitcast i64* @.C358_MAIN_ to i8*, !dbg !40
  %14 = bitcast i64* %z_b_0_307 to i8*, !dbg !40
  %15 = bitcast i64* %z_b_1_308 to i8*, !dbg !40
  %16 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !40
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %16(i8* %10, i8* %11, i8* %12, i8* %13, i8* %14, i8* %15), !dbg !40
  %17 = bitcast [16 x i64]* %"a$sd1_356" to i8*, !dbg !40
  %18 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !40
  call void (i8*, i32, ...) %18(i8* %17, i32 25), !dbg !40
  %19 = load i64, i64* %z_b_1_308, align 8, !dbg !40
  call void @llvm.dbg.value(metadata i64 %19, metadata !39, metadata !DIExpression()), !dbg !21
  %20 = load i64, i64* %z_b_0_307, align 8, !dbg !40
  call void @llvm.dbg.value(metadata i64 %20, metadata !39, metadata !DIExpression()), !dbg !21
  %21 = sub nsw i64 %20, 1, !dbg !40
  %22 = sub nsw i64 %19, %21, !dbg !40
  call void @llvm.dbg.declare(metadata i64* %z_b_2_309, metadata !39, metadata !DIExpression()), !dbg !21
  store i64 %22, i64* %z_b_2_309, align 8, !dbg !40
  %23 = load i64, i64* %z_b_0_307, align 8, !dbg !40
  call void @llvm.dbg.value(metadata i64 %23, metadata !39, metadata !DIExpression()), !dbg !21
  call void @llvm.dbg.declare(metadata i64* %z_b_3_310, metadata !39, metadata !DIExpression()), !dbg !21
  store i64 %23, i64* %z_b_3_310, align 8, !dbg !40
  %24 = bitcast i64* %z_b_2_309 to i8*, !dbg !40
  %25 = bitcast i64* @.C357_MAIN_ to i8*, !dbg !40
  %26 = bitcast i64* @.C358_MAIN_ to i8*, !dbg !40
  %27 = bitcast i32** %.Z0965_323 to i8*, !dbg !40
  %28 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !40
  %29 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !40
  %30 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !40
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %30(i8* %24, i8* %25, i8* %26, i8* null, i8* %27, i8* null, i8* %28, i8* %29, i8* null, i64 0), !dbg !40
  call void @llvm.dbg.declare(metadata i64* %z_b_4_314, metadata !39, metadata !DIExpression()), !dbg !21
  store i64 1, i64* %z_b_4_314, align 8, !dbg !41
  %31 = load i32, i32* %len_322, align 4, !dbg !41
  call void @llvm.dbg.value(metadata i32 %31, metadata !37, metadata !DIExpression()), !dbg !21
  %32 = sext i32 %31 to i64, !dbg !41
  call void @llvm.dbg.declare(metadata i64* %z_b_5_315, metadata !39, metadata !DIExpression()), !dbg !21
  store i64 %32, i64* %z_b_5_315, align 8, !dbg !41
  %33 = load i64, i64* %z_b_5_315, align 8, !dbg !41
  call void @llvm.dbg.value(metadata i64 %33, metadata !39, metadata !DIExpression()), !dbg !21
  call void @llvm.dbg.declare(metadata i64* %z_e_67_318, metadata !39, metadata !DIExpression()), !dbg !21
  store i64 %33, i64* %z_e_67_318, align 8, !dbg !41
  %34 = bitcast [16 x i64]* %"b$sd2_360" to i8*, !dbg !41
  %35 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !41
  %36 = bitcast i64* @.C357_MAIN_ to i8*, !dbg !41
  %37 = bitcast i64* @.C358_MAIN_ to i8*, !dbg !41
  %38 = bitcast i64* %z_b_4_314 to i8*, !dbg !41
  %39 = bitcast i64* %z_b_5_315 to i8*, !dbg !41
  %40 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !41
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %40(i8* %34, i8* %35, i8* %36, i8* %37, i8* %38, i8* %39), !dbg !41
  %41 = bitcast [16 x i64]* %"b$sd2_360" to i8*, !dbg !41
  %42 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !41
  call void (i8*, i32, ...) %42(i8* %41, i32 25), !dbg !41
  %43 = load i64, i64* %z_b_5_315, align 8, !dbg !41
  call void @llvm.dbg.value(metadata i64 %43, metadata !39, metadata !DIExpression()), !dbg !21
  %44 = load i64, i64* %z_b_4_314, align 8, !dbg !41
  call void @llvm.dbg.value(metadata i64 %44, metadata !39, metadata !DIExpression()), !dbg !21
  %45 = sub nsw i64 %44, 1, !dbg !41
  %46 = sub nsw i64 %43, %45, !dbg !41
  call void @llvm.dbg.declare(metadata i64* %z_b_6_316, metadata !39, metadata !DIExpression()), !dbg !21
  store i64 %46, i64* %z_b_6_316, align 8, !dbg !41
  %47 = load i64, i64* %z_b_4_314, align 8, !dbg !41
  call void @llvm.dbg.value(metadata i64 %47, metadata !39, metadata !DIExpression()), !dbg !21
  call void @llvm.dbg.declare(metadata i64* %z_b_7_317, metadata !39, metadata !DIExpression()), !dbg !21
  store i64 %47, i64* %z_b_7_317, align 8, !dbg !41
  %48 = bitcast i64* %z_b_6_316 to i8*, !dbg !41
  %49 = bitcast i64* @.C357_MAIN_ to i8*, !dbg !41
  %50 = bitcast i64* @.C358_MAIN_ to i8*, !dbg !41
  %51 = bitcast i32** %.Z0966_324 to i8*, !dbg !41
  %52 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !41
  %53 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !41
  %54 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !41
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %54(i8* %48, i8* %49, i8* %50, i8* null, i8* %51, i8* null, i8* %52, i8* %53, i8* null, i64 0), !dbg !41
  %55 = load i32, i32* %len_322, align 4, !dbg !42
  call void @llvm.dbg.value(metadata i32 %55, metadata !37, metadata !DIExpression()), !dbg !21
  store i32 %55, i32* %.dY0001_368, align 4, !dbg !42
  call void @llvm.dbg.declare(metadata i32* %i_306, metadata !43, metadata !DIExpression()), !dbg !21
  store i32 1, i32* %i_306, align 4, !dbg !42
  %56 = load i32, i32* %.dY0001_368, align 4, !dbg !42
  %57 = icmp sle i32 %56, 0, !dbg !42
  br i1 %57, label %L.LB1_367, label %L.LB1_366, !dbg !42

L.LB1_366:                                        ; preds = %L.LB1_366, %L.LB1_402
  %58 = load i32, i32* %i_306, align 4, !dbg !44
  call void @llvm.dbg.value(metadata i32 %58, metadata !43, metadata !DIExpression()), !dbg !21
  %59 = load i32, i32* %i_306, align 4, !dbg !44
  call void @llvm.dbg.value(metadata i32 %59, metadata !43, metadata !DIExpression()), !dbg !21
  %60 = sext i32 %59 to i64, !dbg !44
  %61 = bitcast [16 x i64]* %"a$sd1_356" to i8*, !dbg !44
  %62 = getelementptr i8, i8* %61, i64 56, !dbg !44
  %63 = bitcast i8* %62 to i64*, !dbg !44
  %64 = load i64, i64* %63, align 8, !dbg !44
  %65 = add nsw i64 %60, %64, !dbg !44
  %66 = load i32*, i32** %.Z0965_323, align 8, !dbg !44
  call void @llvm.dbg.value(metadata i32* %66, metadata !36, metadata !DIExpression()), !dbg !21
  %67 = bitcast i32* %66 to i8*, !dbg !44
  %68 = getelementptr i8, i8* %67, i64 -4, !dbg !44
  %69 = bitcast i8* %68 to i32*, !dbg !44
  %70 = getelementptr i32, i32* %69, i64 %65, !dbg !44
  store i32 %58, i32* %70, align 4, !dbg !44
  %71 = load i32, i32* %i_306, align 4, !dbg !45
  call void @llvm.dbg.value(metadata i32 %71, metadata !43, metadata !DIExpression()), !dbg !21
  %72 = load i32, i32* %i_306, align 4, !dbg !45
  call void @llvm.dbg.value(metadata i32 %72, metadata !43, metadata !DIExpression()), !dbg !21
  %73 = sext i32 %72 to i64, !dbg !45
  %74 = bitcast [16 x i64]* %"b$sd2_360" to i8*, !dbg !45
  %75 = getelementptr i8, i8* %74, i64 56, !dbg !45
  %76 = bitcast i8* %75 to i64*, !dbg !45
  %77 = load i64, i64* %76, align 8, !dbg !45
  %78 = add nsw i64 %73, %77, !dbg !45
  %79 = load i32*, i32** %.Z0966_324, align 8, !dbg !45
  call void @llvm.dbg.value(metadata i32* %79, metadata !28, metadata !DIExpression()), !dbg !21
  %80 = bitcast i32* %79 to i8*, !dbg !45
  %81 = getelementptr i8, i8* %80, i64 -4, !dbg !45
  %82 = bitcast i8* %81 to i32*, !dbg !45
  %83 = getelementptr i32, i32* %82, i64 %78, !dbg !45
  store i32 %71, i32* %83, align 4, !dbg !45
  %84 = load i32, i32* %i_306, align 4, !dbg !46
  call void @llvm.dbg.value(metadata i32 %84, metadata !43, metadata !DIExpression()), !dbg !21
  %85 = add nsw i32 %84, 1, !dbg !46
  store i32 %85, i32* %i_306, align 4, !dbg !46
  %86 = load i32, i32* %.dY0001_368, align 4, !dbg !46
  %87 = sub nsw i32 %86, 1, !dbg !46
  store i32 %87, i32* %.dY0001_368, align 4, !dbg !46
  %88 = load i32, i32* %.dY0001_368, align 4, !dbg !46
  %89 = icmp sgt i32 %88, 0, !dbg !46
  br i1 %89, label %L.LB1_366, label %L.LB1_367, !dbg !46

L.LB1_367:                                        ; preds = %L.LB1_366, %L.LB1_402
  %90 = bitcast i32* %len_322 to i8*, !dbg !47
  %91 = bitcast %astruct.dt74* %.uplevelArgPack0001_429 to i8**, !dbg !47
  store i8* %90, i8** %91, align 8, !dbg !47
  %92 = bitcast i32** %.Z0965_323 to i8*, !dbg !47
  %93 = bitcast %astruct.dt74* %.uplevelArgPack0001_429 to i8*, !dbg !47
  %94 = getelementptr i8, i8* %93, i64 8, !dbg !47
  %95 = bitcast i8* %94 to i8**, !dbg !47
  store i8* %92, i8** %95, align 8, !dbg !47
  %96 = bitcast i32** %.Z0965_323 to i8*, !dbg !47
  %97 = bitcast %astruct.dt74* %.uplevelArgPack0001_429 to i8*, !dbg !47
  %98 = getelementptr i8, i8* %97, i64 16, !dbg !47
  %99 = bitcast i8* %98 to i8**, !dbg !47
  store i8* %96, i8** %99, align 8, !dbg !47
  %100 = bitcast i64* %z_b_0_307 to i8*, !dbg !47
  %101 = bitcast %astruct.dt74* %.uplevelArgPack0001_429 to i8*, !dbg !47
  %102 = getelementptr i8, i8* %101, i64 24, !dbg !47
  %103 = bitcast i8* %102 to i8**, !dbg !47
  store i8* %100, i8** %103, align 8, !dbg !47
  %104 = bitcast i64* %z_b_1_308 to i8*, !dbg !47
  %105 = bitcast %astruct.dt74* %.uplevelArgPack0001_429 to i8*, !dbg !47
  %106 = getelementptr i8, i8* %105, i64 32, !dbg !47
  %107 = bitcast i8* %106 to i8**, !dbg !47
  store i8* %104, i8** %107, align 8, !dbg !47
  %108 = bitcast i64* %z_e_60_311 to i8*, !dbg !47
  %109 = bitcast %astruct.dt74* %.uplevelArgPack0001_429 to i8*, !dbg !47
  %110 = getelementptr i8, i8* %109, i64 40, !dbg !47
  %111 = bitcast i8* %110 to i8**, !dbg !47
  store i8* %108, i8** %111, align 8, !dbg !47
  %112 = bitcast i64* %z_b_2_309 to i8*, !dbg !47
  %113 = bitcast %astruct.dt74* %.uplevelArgPack0001_429 to i8*, !dbg !47
  %114 = getelementptr i8, i8* %113, i64 48, !dbg !47
  %115 = bitcast i8* %114 to i8**, !dbg !47
  store i8* %112, i8** %115, align 8, !dbg !47
  %116 = bitcast i64* %z_b_3_310 to i8*, !dbg !47
  %117 = bitcast %astruct.dt74* %.uplevelArgPack0001_429 to i8*, !dbg !47
  %118 = getelementptr i8, i8* %117, i64 56, !dbg !47
  %119 = bitcast i8* %118 to i8**, !dbg !47
  store i8* %116, i8** %119, align 8, !dbg !47
  %120 = bitcast [16 x i64]* %"a$sd1_356" to i8*, !dbg !47
  %121 = bitcast %astruct.dt74* %.uplevelArgPack0001_429 to i8*, !dbg !47
  %122 = getelementptr i8, i8* %121, i64 72, !dbg !47
  %123 = bitcast i8* %122 to i8**, !dbg !47
  store i8* %120, i8** %123, align 8, !dbg !47
  br label %L.LB1_451, !dbg !47

L.LB1_451:                                        ; preds = %L.LB1_367
  %124 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L35_1_ to i64*, !dbg !47
  %125 = bitcast %astruct.dt74* %.uplevelArgPack0001_429 to i64*, !dbg !47
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %124, i64* %125), !dbg !47
  %126 = bitcast i32* %len_322 to i8*, !dbg !48
  %127 = bitcast %astruct.dt80* %.uplevelArgPack0002_471 to i8**, !dbg !48
  store i8* %126, i8** %127, align 8, !dbg !48
  %128 = bitcast i32** %.Z0966_324 to i8*, !dbg !48
  %129 = bitcast %astruct.dt80* %.uplevelArgPack0002_471 to i8*, !dbg !48
  %130 = getelementptr i8, i8* %129, i64 8, !dbg !48
  %131 = bitcast i8* %130 to i8**, !dbg !48
  store i8* %128, i8** %131, align 8, !dbg !48
  %132 = bitcast i32** %.Z0966_324 to i8*, !dbg !48
  %133 = bitcast %astruct.dt80* %.uplevelArgPack0002_471 to i8*, !dbg !48
  %134 = getelementptr i8, i8* %133, i64 16, !dbg !48
  %135 = bitcast i8* %134 to i8**, !dbg !48
  store i8* %132, i8** %135, align 8, !dbg !48
  %136 = bitcast i64* %z_b_4_314 to i8*, !dbg !48
  %137 = bitcast %astruct.dt80* %.uplevelArgPack0002_471 to i8*, !dbg !48
  %138 = getelementptr i8, i8* %137, i64 24, !dbg !48
  %139 = bitcast i8* %138 to i8**, !dbg !48
  store i8* %136, i8** %139, align 8, !dbg !48
  %140 = bitcast i64* %z_b_5_315 to i8*, !dbg !48
  %141 = bitcast %astruct.dt80* %.uplevelArgPack0002_471 to i8*, !dbg !48
  %142 = getelementptr i8, i8* %141, i64 32, !dbg !48
  %143 = bitcast i8* %142 to i8**, !dbg !48
  store i8* %140, i8** %143, align 8, !dbg !48
  %144 = bitcast i64* %z_e_67_318 to i8*, !dbg !48
  %145 = bitcast %astruct.dt80* %.uplevelArgPack0002_471 to i8*, !dbg !48
  %146 = getelementptr i8, i8* %145, i64 40, !dbg !48
  %147 = bitcast i8* %146 to i8**, !dbg !48
  store i8* %144, i8** %147, align 8, !dbg !48
  %148 = bitcast i64* %z_b_6_316 to i8*, !dbg !48
  %149 = bitcast %astruct.dt80* %.uplevelArgPack0002_471 to i8*, !dbg !48
  %150 = getelementptr i8, i8* %149, i64 48, !dbg !48
  %151 = bitcast i8* %150 to i8**, !dbg !48
  store i8* %148, i8** %151, align 8, !dbg !48
  %152 = bitcast i64* %z_b_7_317 to i8*, !dbg !48
  %153 = bitcast %astruct.dt80* %.uplevelArgPack0002_471 to i8*, !dbg !48
  %154 = getelementptr i8, i8* %153, i64 56, !dbg !48
  %155 = bitcast i8* %154 to i8**, !dbg !48
  store i8* %152, i8** %155, align 8, !dbg !48
  call void @llvm.dbg.declare(metadata i32* %tmp2_320, metadata !49, metadata !DIExpression()), !dbg !21
  %156 = bitcast i32* %tmp2_320 to i8*, !dbg !48
  %157 = bitcast %astruct.dt80* %.uplevelArgPack0002_471 to i8*, !dbg !48
  %158 = getelementptr i8, i8* %157, i64 64, !dbg !48
  %159 = bitcast i8* %158 to i8**, !dbg !48
  store i8* %156, i8** %159, align 8, !dbg !48
  %160 = bitcast [16 x i64]* %"b$sd2_360" to i8*, !dbg !48
  %161 = bitcast %astruct.dt80* %.uplevelArgPack0002_471 to i8*, !dbg !48
  %162 = getelementptr i8, i8* %161, i64 72, !dbg !48
  %163 = bitcast i8* %162 to i8**, !dbg !48
  store i8* %160, i8** %163, align 8, !dbg !48
  br label %L.LB1_493, !dbg !48

L.LB1_493:                                        ; preds = %L.LB1_451
  %164 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L44_2_ to i64*, !dbg !48
  %165 = bitcast %astruct.dt80* %.uplevelArgPack0002_471 to i64*, !dbg !48
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %164, i64* %165), !dbg !48
  call void (...) @_mp_bcs_nest(), !dbg !50
  %166 = bitcast i32* @.C344_MAIN_ to i8*, !dbg !50
  %167 = bitcast [57 x i8]* @.C342_MAIN_ to i8*, !dbg !50
  %168 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !50
  call void (i8*, i8*, i64, ...) %168(i8* %166, i8* %167, i64 57), !dbg !50
  %169 = bitcast i32* @.C346_MAIN_ to i8*, !dbg !50
  %170 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !50
  %171 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !50
  %172 = bitcast %struct.STATICS1* @.STATICS1 to i8*, !dbg !50
  %173 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !50
  %174 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %173(i8* %169, i8* null, i8* %170, i8* %171, i8* %172, i8* null, i64 0), !dbg !50
  call void @llvm.dbg.declare(metadata i32* %z__io_348, metadata !51, metadata !DIExpression()), !dbg !21
  store i32 %174, i32* %z__io_348, align 4, !dbg !50
  %175 = bitcast [16 x i64]* %"a$sd1_356" to i8*, !dbg !50
  %176 = getelementptr i8, i8* %175, i64 56, !dbg !50
  %177 = bitcast i8* %176 to i64*, !dbg !50
  %178 = load i64, i64* %177, align 8, !dbg !50
  %179 = load i32*, i32** %.Z0965_323, align 8, !dbg !50
  call void @llvm.dbg.value(metadata i32* %179, metadata !36, metadata !DIExpression()), !dbg !21
  %180 = bitcast i32* %179 to i8*, !dbg !50
  %181 = getelementptr i8, i8* %180, i64 196, !dbg !50
  %182 = bitcast i8* %181 to i32*, !dbg !50
  %183 = getelementptr i32, i32* %182, i64 %178, !dbg !50
  %184 = load i32, i32* %183, align 4, !dbg !50
  %185 = bitcast i32 (...)* @f90io_sc_i_fmt_write to i32 (i32, i32, ...)*, !dbg !50
  %186 = call i32 (i32, i32, ...) %185(i32 %184, i32 25), !dbg !50
  store i32 %186, i32* %z__io_348, align 4, !dbg !50
  %187 = bitcast [16 x i64]* %"b$sd2_360" to i8*, !dbg !50
  %188 = getelementptr i8, i8* %187, i64 56, !dbg !50
  %189 = bitcast i8* %188 to i64*, !dbg !50
  %190 = load i64, i64* %189, align 8, !dbg !50
  %191 = load i32*, i32** %.Z0966_324, align 8, !dbg !50
  call void @llvm.dbg.value(metadata i32* %191, metadata !28, metadata !DIExpression()), !dbg !21
  %192 = bitcast i32* %191 to i8*, !dbg !50
  %193 = getelementptr i8, i8* %192, i64 196, !dbg !50
  %194 = bitcast i8* %193 to i32*, !dbg !50
  %195 = getelementptr i32, i32* %194, i64 %190, !dbg !50
  %196 = load i32, i32* %195, align 4, !dbg !50
  %197 = bitcast i32 (...)* @f90io_sc_i_fmt_write to i32 (i32, i32, ...)*, !dbg !50
  %198 = call i32 (i32, i32, ...) %197(i32 %196, i32 25), !dbg !50
  store i32 %198, i32* %z__io_348, align 4, !dbg !50
  %199 = call i32 (...) @f90io_fmtw_end(), !dbg !50
  store i32 %199, i32* %z__io_348, align 4, !dbg !50
  call void (...) @_mp_ecs_nest(), !dbg !50
  %200 = load i32*, i32** %.Z0965_323, align 8, !dbg !52
  call void @llvm.dbg.value(metadata i32* %200, metadata !36, metadata !DIExpression()), !dbg !21
  %201 = bitcast i32* %200 to i8*, !dbg !52
  %202 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !52
  %203 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !52
  call void (i8*, i8*, i8*, i8*, i64, ...) %203(i8* null, i8* %201, i8* %202, i8* null, i64 0), !dbg !52
  %204 = bitcast i32** %.Z0965_323 to i8**, !dbg !52
  store i8* null, i8** %204, align 8, !dbg !52
  %205 = bitcast [16 x i64]* %"a$sd1_356" to i64*, !dbg !52
  store i64 0, i64* %205, align 8, !dbg !52
  %206 = load i32*, i32** %.Z0966_324, align 8, !dbg !52
  call void @llvm.dbg.value(metadata i32* %206, metadata !28, metadata !DIExpression()), !dbg !21
  %207 = bitcast i32* %206 to i8*, !dbg !52
  %208 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !52
  %209 = bitcast void (...)* @f90_dealloc03a_i8 to void (i8*, i8*, i8*, i8*, i64, ...)*, !dbg !52
  call void (i8*, i8*, i8*, i8*, i64, ...) %209(i8* null, i8* %207, i8* %208, i8* null, i64 0), !dbg !52
  %210 = bitcast i32** %.Z0966_324 to i8**, !dbg !52
  store i8* null, i8** %210, align 8, !dbg !52
  %211 = bitcast [16 x i64]* %"b$sd2_360" to i64*, !dbg !52
  store i64 0, i64* %211, align 8, !dbg !52
  ret void, !dbg !26
}

define internal void @__nv_MAIN__F1L35_1_(i32* %__nv_MAIN__F1L35_1Arg0, i64* %__nv_MAIN__F1L35_1Arg1, i64* %__nv_MAIN__F1L35_1Arg2) #0 !dbg !11 {
L.entry:
  %__gtid___nv_MAIN__F1L35_1__530 = alloca i32, align 4
  %.i0000p_330 = alloca i32, align 4
  %i_329 = alloca i32, align 4
  %.du0002p_372 = alloca i32, align 4
  %.de0002p_373 = alloca i32, align 4
  %.di0002p_374 = alloca i32, align 4
  %.ds0002p_375 = alloca i32, align 4
  %.dl0002p_377 = alloca i32, align 4
  %.dl0002p.copy_524 = alloca i32, align 4
  %.de0002p.copy_525 = alloca i32, align 4
  %.ds0002p.copy_526 = alloca i32, align 4
  %.dX0002p_376 = alloca i32, align 4
  %.dY0002p_371 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L35_1Arg0, metadata !53, metadata !DIExpression()), !dbg !54
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L35_1Arg1, metadata !55, metadata !DIExpression()), !dbg !54
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L35_1Arg2, metadata !56, metadata !DIExpression()), !dbg !54
  call void @llvm.dbg.value(metadata i32 1, metadata !57, metadata !DIExpression()), !dbg !54
  call void @llvm.dbg.value(metadata i32 0, metadata !58, metadata !DIExpression()), !dbg !54
  call void @llvm.dbg.value(metadata i32 1, metadata !59, metadata !DIExpression()), !dbg !54
  call void @llvm.dbg.value(metadata i32 0, metadata !60, metadata !DIExpression()), !dbg !54
  call void @llvm.dbg.value(metadata i32 1, metadata !61, metadata !DIExpression()), !dbg !54
  %0 = load i32, i32* %__nv_MAIN__F1L35_1Arg0, align 4, !dbg !62
  store i32 %0, i32* %__gtid___nv_MAIN__F1L35_1__530, align 4, !dbg !62
  br label %L.LB2_515

L.LB2_515:                                        ; preds = %L.entry
  br label %L.LB2_327

L.LB2_327:                                        ; preds = %L.LB2_515
  br label %L.LB2_328

L.LB2_328:                                        ; preds = %L.LB2_327
  store i32 0, i32* %.i0000p_330, align 4, !dbg !63
  call void @llvm.dbg.declare(metadata i32* %i_329, metadata !64, metadata !DIExpression()), !dbg !62
  store i32 1, i32* %i_329, align 4, !dbg !63
  %1 = bitcast i64* %__nv_MAIN__F1L35_1Arg2 to i32**, !dbg !63
  %2 = load i32*, i32** %1, align 8, !dbg !63
  %3 = load i32, i32* %2, align 4, !dbg !63
  store i32 %3, i32* %.du0002p_372, align 4, !dbg !63
  %4 = bitcast i64* %__nv_MAIN__F1L35_1Arg2 to i32**, !dbg !63
  %5 = load i32*, i32** %4, align 8, !dbg !63
  %6 = load i32, i32* %5, align 4, !dbg !63
  store i32 %6, i32* %.de0002p_373, align 4, !dbg !63
  store i32 1, i32* %.di0002p_374, align 4, !dbg !63
  %7 = load i32, i32* %.di0002p_374, align 4, !dbg !63
  store i32 %7, i32* %.ds0002p_375, align 4, !dbg !63
  store i32 1, i32* %.dl0002p_377, align 4, !dbg !63
  %8 = load i32, i32* %.dl0002p_377, align 4, !dbg !63
  store i32 %8, i32* %.dl0002p.copy_524, align 4, !dbg !63
  %9 = load i32, i32* %.de0002p_373, align 4, !dbg !63
  store i32 %9, i32* %.de0002p.copy_525, align 4, !dbg !63
  %10 = load i32, i32* %.ds0002p_375, align 4, !dbg !63
  store i32 %10, i32* %.ds0002p.copy_526, align 4, !dbg !63
  %11 = load i32, i32* %__gtid___nv_MAIN__F1L35_1__530, align 4, !dbg !63
  %12 = bitcast i32* %.i0000p_330 to i64*, !dbg !63
  %13 = bitcast i32* %.dl0002p.copy_524 to i64*, !dbg !63
  %14 = bitcast i32* %.de0002p.copy_525 to i64*, !dbg !63
  %15 = bitcast i32* %.ds0002p.copy_526 to i64*, !dbg !63
  %16 = load i32, i32* %.ds0002p.copy_526, align 4, !dbg !63
  call void @__kmpc_for_static_init_4(i64* null, i32 %11, i32 34, i64* %12, i64* %13, i64* %14, i64* %15, i32 %16, i32 1), !dbg !63
  %17 = load i32, i32* %.dl0002p.copy_524, align 4, !dbg !63
  store i32 %17, i32* %.dl0002p_377, align 4, !dbg !63
  %18 = load i32, i32* %.de0002p.copy_525, align 4, !dbg !63
  store i32 %18, i32* %.de0002p_373, align 4, !dbg !63
  %19 = load i32, i32* %.ds0002p.copy_526, align 4, !dbg !63
  store i32 %19, i32* %.ds0002p_375, align 4, !dbg !63
  %20 = load i32, i32* %.dl0002p_377, align 4, !dbg !63
  store i32 %20, i32* %i_329, align 4, !dbg !63
  %21 = load i32, i32* %i_329, align 4, !dbg !63
  call void @llvm.dbg.value(metadata i32 %21, metadata !64, metadata !DIExpression()), !dbg !62
  store i32 %21, i32* %.dX0002p_376, align 4, !dbg !63
  %22 = load i32, i32* %.dX0002p_376, align 4, !dbg !63
  %23 = load i32, i32* %.du0002p_372, align 4, !dbg !63
  %24 = icmp sgt i32 %22, %23, !dbg !63
  br i1 %24, label %L.LB2_370, label %L.LB2_557, !dbg !63

L.LB2_557:                                        ; preds = %L.LB2_328
  %25 = load i32, i32* %.dX0002p_376, align 4, !dbg !63
  store i32 %25, i32* %i_329, align 4, !dbg !63
  %26 = load i32, i32* %.di0002p_374, align 4, !dbg !63
  %27 = load i32, i32* %.de0002p_373, align 4, !dbg !63
  %28 = load i32, i32* %.dX0002p_376, align 4, !dbg !63
  %29 = sub nsw i32 %27, %28, !dbg !63
  %30 = add nsw i32 %26, %29, !dbg !63
  %31 = load i32, i32* %.di0002p_374, align 4, !dbg !63
  %32 = sdiv i32 %30, %31, !dbg !63
  store i32 %32, i32* %.dY0002p_371, align 4, !dbg !63
  %33 = load i32, i32* %.dY0002p_371, align 4, !dbg !63
  %34 = icmp sle i32 %33, 0, !dbg !63
  br i1 %34, label %L.LB2_380, label %L.LB2_379, !dbg !63

L.LB2_379:                                        ; preds = %L.LB2_379, %L.LB2_557
  %35 = load i32, i32* %i_329, align 4, !dbg !65
  call void @llvm.dbg.value(metadata i32 %35, metadata !64, metadata !DIExpression()), !dbg !62
  %36 = load i32, i32* %i_329, align 4, !dbg !65
  call void @llvm.dbg.value(metadata i32 %36, metadata !64, metadata !DIExpression()), !dbg !62
  %37 = sext i32 %36 to i64, !dbg !65
  %38 = bitcast i64* %__nv_MAIN__F1L35_1Arg2 to i8*, !dbg !65
  %39 = getelementptr i8, i8* %38, i64 72, !dbg !65
  %40 = bitcast i8* %39 to i8**, !dbg !65
  %41 = load i8*, i8** %40, align 8, !dbg !65
  %42 = getelementptr i8, i8* %41, i64 56, !dbg !65
  %43 = bitcast i8* %42 to i64*, !dbg !65
  %44 = load i64, i64* %43, align 8, !dbg !65
  %45 = add nsw i64 %37, %44, !dbg !65
  %46 = bitcast i64* %__nv_MAIN__F1L35_1Arg2 to i8*, !dbg !65
  %47 = getelementptr i8, i8* %46, i64 16, !dbg !65
  %48 = bitcast i8* %47 to i8***, !dbg !65
  %49 = load i8**, i8*** %48, align 8, !dbg !65
  %50 = load i8*, i8** %49, align 8, !dbg !65
  %51 = getelementptr i8, i8* %50, i64 -4, !dbg !65
  %52 = bitcast i8* %51 to i32*, !dbg !65
  %53 = getelementptr i32, i32* %52, i64 %45, !dbg !65
  %54 = load i32, i32* %53, align 4, !dbg !65
  %55 = add nsw i32 %35, %54, !dbg !65
  %56 = bitcast %struct.BSS1* @.BSS1 to i32*, !dbg !65
  store i32 %55, i32* %56, align 4, !dbg !65
  %57 = bitcast %struct.BSS1* @.BSS1 to i32*, !dbg !66
  %58 = load i32, i32* %57, align 4, !dbg !66
  %59 = load i32, i32* %i_329, align 4, !dbg !66
  call void @llvm.dbg.value(metadata i32 %59, metadata !64, metadata !DIExpression()), !dbg !62
  %60 = sext i32 %59 to i64, !dbg !66
  %61 = bitcast i64* %__nv_MAIN__F1L35_1Arg2 to i8*, !dbg !66
  %62 = getelementptr i8, i8* %61, i64 72, !dbg !66
  %63 = bitcast i8* %62 to i8**, !dbg !66
  %64 = load i8*, i8** %63, align 8, !dbg !66
  %65 = getelementptr i8, i8* %64, i64 56, !dbg !66
  %66 = bitcast i8* %65 to i64*, !dbg !66
  %67 = load i64, i64* %66, align 8, !dbg !66
  %68 = add nsw i64 %60, %67, !dbg !66
  %69 = bitcast i64* %__nv_MAIN__F1L35_1Arg2 to i8*, !dbg !66
  %70 = getelementptr i8, i8* %69, i64 16, !dbg !66
  %71 = bitcast i8* %70 to i8***, !dbg !66
  %72 = load i8**, i8*** %71, align 8, !dbg !66
  %73 = load i8*, i8** %72, align 8, !dbg !66
  %74 = getelementptr i8, i8* %73, i64 -4, !dbg !66
  %75 = bitcast i8* %74 to i32*, !dbg !66
  %76 = getelementptr i32, i32* %75, i64 %68, !dbg !66
  store i32 %58, i32* %76, align 4, !dbg !66
  %77 = load i32, i32* %.di0002p_374, align 4, !dbg !67
  %78 = load i32, i32* %i_329, align 4, !dbg !67
  call void @llvm.dbg.value(metadata i32 %78, metadata !64, metadata !DIExpression()), !dbg !62
  %79 = add nsw i32 %77, %78, !dbg !67
  store i32 %79, i32* %i_329, align 4, !dbg !67
  %80 = load i32, i32* %.dY0002p_371, align 4, !dbg !67
  %81 = sub nsw i32 %80, 1, !dbg !67
  store i32 %81, i32* %.dY0002p_371, align 4, !dbg !67
  %82 = load i32, i32* %.dY0002p_371, align 4, !dbg !67
  %83 = icmp sgt i32 %82, 0, !dbg !67
  br i1 %83, label %L.LB2_379, label %L.LB2_380, !dbg !67

L.LB2_380:                                        ; preds = %L.LB2_379, %L.LB2_557
  br label %L.LB2_370

L.LB2_370:                                        ; preds = %L.LB2_380, %L.LB2_328
  %84 = load i32, i32* %__gtid___nv_MAIN__F1L35_1__530, align 4, !dbg !67
  call void @__kmpc_for_static_fini(i64* null, i32 %84), !dbg !67
  br label %L.LB2_331

L.LB2_331:                                        ; preds = %L.LB2_370
  %85 = load i32, i32* %__gtid___nv_MAIN__F1L35_1__530, align 4, !dbg !68
  call void @__kmpc_barrier(i64* null, i32 %85), !dbg !68
  br label %L.LB2_332

L.LB2_332:                                        ; preds = %L.LB2_331
  ret void, !dbg !62
}

define internal void @__nv_MAIN__F1L44_2_(i32* %__nv_MAIN__F1L44_2Arg0, i64* %__nv_MAIN__F1L44_2Arg1, i64* %__nv_MAIN__F1L44_2Arg2) #0 !dbg !69 {
L.entry:
  %__gtid___nv_MAIN__F1L44_2__576 = alloca i32, align 4
  %.i0001p_338 = alloca i32, align 4
  %i_337 = alloca i32, align 4
  %.du0003p_384 = alloca i32, align 4
  %.de0003p_385 = alloca i32, align 4
  %.di0003p_386 = alloca i32, align 4
  %.ds0003p_387 = alloca i32, align 4
  %.dl0003p_389 = alloca i32, align 4
  %.dl0003p.copy_570 = alloca i32, align 4
  %.de0003p.copy_571 = alloca i32, align 4
  %.ds0003p.copy_572 = alloca i32, align 4
  %.dX0003p_388 = alloca i32, align 4
  %.dY0003p_383 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L44_2Arg0, metadata !70, metadata !DIExpression()), !dbg !71
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L44_2Arg1, metadata !72, metadata !DIExpression()), !dbg !71
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L44_2Arg2, metadata !73, metadata !DIExpression()), !dbg !71
  call void @llvm.dbg.value(metadata i32 1, metadata !74, metadata !DIExpression()), !dbg !71
  call void @llvm.dbg.value(metadata i32 0, metadata !75, metadata !DIExpression()), !dbg !71
  call void @llvm.dbg.value(metadata i32 1, metadata !76, metadata !DIExpression()), !dbg !71
  call void @llvm.dbg.value(metadata i32 0, metadata !77, metadata !DIExpression()), !dbg !71
  call void @llvm.dbg.value(metadata i32 1, metadata !78, metadata !DIExpression()), !dbg !71
  %0 = load i32, i32* %__nv_MAIN__F1L44_2Arg0, align 4, !dbg !79
  store i32 %0, i32* %__gtid___nv_MAIN__F1L44_2__576, align 4, !dbg !79
  br label %L.LB3_561

L.LB3_561:                                        ; preds = %L.entry
  br label %L.LB3_335

L.LB3_335:                                        ; preds = %L.LB3_561
  br label %L.LB3_336

L.LB3_336:                                        ; preds = %L.LB3_335
  store i32 0, i32* %.i0001p_338, align 4, !dbg !80
  call void @llvm.dbg.declare(metadata i32* %i_337, metadata !81, metadata !DIExpression()), !dbg !79
  store i32 1, i32* %i_337, align 4, !dbg !80
  %1 = bitcast i64* %__nv_MAIN__F1L44_2Arg2 to i32**, !dbg !80
  %2 = load i32*, i32** %1, align 8, !dbg !80
  %3 = load i32, i32* %2, align 4, !dbg !80
  store i32 %3, i32* %.du0003p_384, align 4, !dbg !80
  %4 = bitcast i64* %__nv_MAIN__F1L44_2Arg2 to i32**, !dbg !80
  %5 = load i32*, i32** %4, align 8, !dbg !80
  %6 = load i32, i32* %5, align 4, !dbg !80
  store i32 %6, i32* %.de0003p_385, align 4, !dbg !80
  store i32 1, i32* %.di0003p_386, align 4, !dbg !80
  %7 = load i32, i32* %.di0003p_386, align 4, !dbg !80
  store i32 %7, i32* %.ds0003p_387, align 4, !dbg !80
  store i32 1, i32* %.dl0003p_389, align 4, !dbg !80
  %8 = load i32, i32* %.dl0003p_389, align 4, !dbg !80
  store i32 %8, i32* %.dl0003p.copy_570, align 4, !dbg !80
  %9 = load i32, i32* %.de0003p_385, align 4, !dbg !80
  store i32 %9, i32* %.de0003p.copy_571, align 4, !dbg !80
  %10 = load i32, i32* %.ds0003p_387, align 4, !dbg !80
  store i32 %10, i32* %.ds0003p.copy_572, align 4, !dbg !80
  %11 = load i32, i32* %__gtid___nv_MAIN__F1L44_2__576, align 4, !dbg !80
  %12 = bitcast i32* %.i0001p_338 to i64*, !dbg !80
  %13 = bitcast i32* %.dl0003p.copy_570 to i64*, !dbg !80
  %14 = bitcast i32* %.de0003p.copy_571 to i64*, !dbg !80
  %15 = bitcast i32* %.ds0003p.copy_572 to i64*, !dbg !80
  %16 = load i32, i32* %.ds0003p.copy_572, align 4, !dbg !80
  call void @__kmpc_for_static_init_4(i64* null, i32 %11, i32 34, i64* %12, i64* %13, i64* %14, i64* %15, i32 %16, i32 1), !dbg !80
  %17 = load i32, i32* %.dl0003p.copy_570, align 4, !dbg !80
  store i32 %17, i32* %.dl0003p_389, align 4, !dbg !80
  %18 = load i32, i32* %.de0003p.copy_571, align 4, !dbg !80
  store i32 %18, i32* %.de0003p_385, align 4, !dbg !80
  %19 = load i32, i32* %.ds0003p.copy_572, align 4, !dbg !80
  store i32 %19, i32* %.ds0003p_387, align 4, !dbg !80
  %20 = load i32, i32* %.dl0003p_389, align 4, !dbg !80
  store i32 %20, i32* %i_337, align 4, !dbg !80
  %21 = load i32, i32* %i_337, align 4, !dbg !80
  call void @llvm.dbg.value(metadata i32 %21, metadata !81, metadata !DIExpression()), !dbg !79
  store i32 %21, i32* %.dX0003p_388, align 4, !dbg !80
  %22 = load i32, i32* %.dX0003p_388, align 4, !dbg !80
  %23 = load i32, i32* %.du0003p_384, align 4, !dbg !80
  %24 = icmp sgt i32 %22, %23, !dbg !80
  br i1 %24, label %L.LB3_382, label %L.LB3_585, !dbg !80

L.LB3_585:                                        ; preds = %L.LB3_336
  %25 = load i32, i32* %.dX0003p_388, align 4, !dbg !80
  store i32 %25, i32* %i_337, align 4, !dbg !80
  %26 = load i32, i32* %.di0003p_386, align 4, !dbg !80
  %27 = load i32, i32* %.de0003p_385, align 4, !dbg !80
  %28 = load i32, i32* %.dX0003p_388, align 4, !dbg !80
  %29 = sub nsw i32 %27, %28, !dbg !80
  %30 = add nsw i32 %26, %29, !dbg !80
  %31 = load i32, i32* %.di0003p_386, align 4, !dbg !80
  %32 = sdiv i32 %30, %31, !dbg !80
  store i32 %32, i32* %.dY0003p_383, align 4, !dbg !80
  %33 = load i32, i32* %.dY0003p_383, align 4, !dbg !80
  %34 = icmp sle i32 %33, 0, !dbg !80
  br i1 %34, label %L.LB3_392, label %L.LB3_391, !dbg !80

L.LB3_391:                                        ; preds = %L.LB3_391, %L.LB3_585
  %35 = load i32, i32* %i_337, align 4, !dbg !82
  call void @llvm.dbg.value(metadata i32 %35, metadata !81, metadata !DIExpression()), !dbg !79
  %36 = load i32, i32* %i_337, align 4, !dbg !82
  call void @llvm.dbg.value(metadata i32 %36, metadata !81, metadata !DIExpression()), !dbg !79
  %37 = sext i32 %36 to i64, !dbg !82
  %38 = bitcast i64* %__nv_MAIN__F1L44_2Arg2 to i8*, !dbg !82
  %39 = getelementptr i8, i8* %38, i64 72, !dbg !82
  %40 = bitcast i8* %39 to i8**, !dbg !82
  %41 = load i8*, i8** %40, align 8, !dbg !82
  %42 = getelementptr i8, i8* %41, i64 56, !dbg !82
  %43 = bitcast i8* %42 to i64*, !dbg !82
  %44 = load i64, i64* %43, align 8, !dbg !82
  %45 = add nsw i64 %37, %44, !dbg !82
  %46 = bitcast i64* %__nv_MAIN__F1L44_2Arg2 to i8*, !dbg !82
  %47 = getelementptr i8, i8* %46, i64 16, !dbg !82
  %48 = bitcast i8* %47 to i8***, !dbg !82
  %49 = load i8**, i8*** %48, align 8, !dbg !82
  %50 = load i8*, i8** %49, align 8, !dbg !82
  %51 = getelementptr i8, i8* %50, i64 -4, !dbg !82
  %52 = bitcast i8* %51 to i32*, !dbg !82
  %53 = getelementptr i32, i32* %52, i64 %45, !dbg !82
  %54 = load i32, i32* %53, align 4, !dbg !82
  %55 = add nsw i32 %35, %54, !dbg !82
  %56 = bitcast i64* %__nv_MAIN__F1L44_2Arg2 to i8*, !dbg !82
  %57 = getelementptr i8, i8* %56, i64 64, !dbg !82
  %58 = bitcast i8* %57 to i32**, !dbg !82
  %59 = load i32*, i32** %58, align 8, !dbg !82
  store i32 %55, i32* %59, align 4, !dbg !82
  %60 = bitcast i64* %__nv_MAIN__F1L44_2Arg2 to i8*, !dbg !83
  %61 = getelementptr i8, i8* %60, i64 64, !dbg !83
  %62 = bitcast i8* %61 to i32**, !dbg !83
  %63 = load i32*, i32** %62, align 8, !dbg !83
  %64 = load i32, i32* %63, align 4, !dbg !83
  %65 = load i32, i32* %i_337, align 4, !dbg !83
  call void @llvm.dbg.value(metadata i32 %65, metadata !81, metadata !DIExpression()), !dbg !79
  %66 = sext i32 %65 to i64, !dbg !83
  %67 = bitcast i64* %__nv_MAIN__F1L44_2Arg2 to i8*, !dbg !83
  %68 = getelementptr i8, i8* %67, i64 72, !dbg !83
  %69 = bitcast i8* %68 to i8**, !dbg !83
  %70 = load i8*, i8** %69, align 8, !dbg !83
  %71 = getelementptr i8, i8* %70, i64 56, !dbg !83
  %72 = bitcast i8* %71 to i64*, !dbg !83
  %73 = load i64, i64* %72, align 8, !dbg !83
  %74 = add nsw i64 %66, %73, !dbg !83
  %75 = bitcast i64* %__nv_MAIN__F1L44_2Arg2 to i8*, !dbg !83
  %76 = getelementptr i8, i8* %75, i64 16, !dbg !83
  %77 = bitcast i8* %76 to i8***, !dbg !83
  %78 = load i8**, i8*** %77, align 8, !dbg !83
  %79 = load i8*, i8** %78, align 8, !dbg !83
  %80 = getelementptr i8, i8* %79, i64 -4, !dbg !83
  %81 = bitcast i8* %80 to i32*, !dbg !83
  %82 = getelementptr i32, i32* %81, i64 %74, !dbg !83
  store i32 %64, i32* %82, align 4, !dbg !83
  %83 = load i32, i32* %.di0003p_386, align 4, !dbg !84
  %84 = load i32, i32* %i_337, align 4, !dbg !84
  call void @llvm.dbg.value(metadata i32 %84, metadata !81, metadata !DIExpression()), !dbg !79
  %85 = add nsw i32 %83, %84, !dbg !84
  store i32 %85, i32* %i_337, align 4, !dbg !84
  %86 = load i32, i32* %.dY0003p_383, align 4, !dbg !84
  %87 = sub nsw i32 %86, 1, !dbg !84
  store i32 %87, i32* %.dY0003p_383, align 4, !dbg !84
  %88 = load i32, i32* %.dY0003p_383, align 4, !dbg !84
  %89 = icmp sgt i32 %88, 0, !dbg !84
  br i1 %89, label %L.LB3_391, label %L.LB3_392, !dbg !84

L.LB3_392:                                        ; preds = %L.LB3_391, %L.LB3_585
  br label %L.LB3_382

L.LB3_382:                                        ; preds = %L.LB3_392, %L.LB3_336
  %90 = load i32, i32* %__gtid___nv_MAIN__F1L44_2__576, align 4, !dbg !84
  call void @__kmpc_for_static_fini(i64* null, i32 %90), !dbg !84
  br label %L.LB3_339

L.LB3_339:                                        ; preds = %L.LB3_382
  %91 = load i32, i32* %__gtid___nv_MAIN__F1L44_2__576, align 4, !dbg !85
  call void @__kmpc_barrier(i64* null, i32 %91), !dbg !85
  br label %L.LB3_340

L.LB3_340:                                        ; preds = %L.LB3_339
  ret void, !dbg !79
}

declare void @__kmpc_barrier(i64*, i32) #0

declare void @__kmpc_for_static_fini(i64*, i32) #0

declare void @__kmpc_for_static_init_4(i64*, i32, i32, i64*, i64*, i64*, i64*, i32, i32) #0

declare void @f90_dealloc03a_i8(...) #0

declare void @_mp_ecs_nest(...) #0

declare signext i32 @f90io_fmtw_end(...) #0

declare signext i32 @f90io_sc_i_fmt_write(...) #0

declare signext i32 @f90io_fmtw_inita(...) #0

declare void @f90io_src_info03a(...) #0

declare void @_mp_bcs_nest(...) #0

declare void @f90_alloc04_chka_i8(...) #0

declare void @f90_set_intrin_type_i8(...) #0

declare void @f90_template1_i8(...) #0

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
!2 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !4, globals: !5, imports: !4)
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB090-static-local-orig-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = !{!6, !9, !15}
!6 = !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!7 = distinct !DIGlobalVariable(name: "tmp", scope: !2, file: !3, type: !8, isLocal: true, isDefinition: true)
!8 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !DIGlobalVariableExpression(var: !10, expr: !DIExpression())
!10 = distinct !DIGlobalVariable(name: "tmp", scope: !11, file: !3, type: !8, isLocal: true, isDefinition: true)
!11 = distinct !DISubprogram(name: "__nv_MAIN__F1L35_1", scope: !2, file: !3, line: 35, type: !12, scopeLine: 35, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!12 = !DISubroutineType(types: !13)
!13 = !{null, !8, !14, !14}
!14 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!15 = !DIGlobalVariableExpression(var: !16, expr: !DIExpression())
!16 = distinct !DIGlobalVariable(name: "tmp", scope: !2, file: !3, type: !8, isLocal: true, isDefinition: true)
!17 = distinct !DISubprogram(name: "drb090_static_local_orig_yes", scope: !2, file: !3, line: 17, type: !18, scopeLine: 17, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!18 = !DISubroutineType(cc: DW_CC_program, types: !19)
!19 = !{null}
!20 = !DILocalVariable(name: "omp_sched_static", scope: !17, file: !3, type: !8)
!21 = !DILocation(line: 0, scope: !17)
!22 = !DILocalVariable(name: "omp_proc_bind_false", scope: !17, file: !3, type: !8)
!23 = !DILocalVariable(name: "omp_proc_bind_true", scope: !17, file: !3, type: !8)
!24 = !DILocalVariable(name: "omp_lock_hint_none", scope: !17, file: !3, type: !8)
!25 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !17, file: !3, type: !8)
!26 = !DILocation(line: 58, column: 1, scope: !17)
!27 = !DILocation(line: 17, column: 1, scope: !17)
!28 = !DILocalVariable(name: "b", scope: !17, file: !3, type: !29)
!29 = !DICompositeType(tag: DW_TAG_array_type, baseType: !8, size: 32, align: 32, elements: !30)
!30 = !{!31}
!31 = !DISubrange(count: 0, lowerBound: 1)
!32 = !DILocalVariable(scope: !17, file: !3, type: !33, flags: DIFlagArtificial)
!33 = !DICompositeType(tag: DW_TAG_array_type, baseType: !14, size: 1024, align: 64, elements: !34)
!34 = !{!35}
!35 = !DISubrange(count: 16, lowerBound: 1)
!36 = !DILocalVariable(name: "a", scope: !17, file: !3, type: !29)
!37 = !DILocalVariable(name: "len", scope: !17, file: !3, type: !8)
!38 = !DILocation(line: 26, column: 1, scope: !17)
!39 = !DILocalVariable(scope: !17, file: !3, type: !14, flags: DIFlagArtificial)
!40 = !DILocation(line: 27, column: 1, scope: !17)
!41 = !DILocation(line: 28, column: 1, scope: !17)
!42 = !DILocation(line: 30, column: 1, scope: !17)
!43 = !DILocalVariable(name: "i", scope: !17, file: !3, type: !8)
!44 = !DILocation(line: 31, column: 1, scope: !17)
!45 = !DILocation(line: 32, column: 1, scope: !17)
!46 = !DILocation(line: 33, column: 1, scope: !17)
!47 = !DILocation(line: 35, column: 1, scope: !17)
!48 = !DILocation(line: 44, column: 1, scope: !17)
!49 = !DILocalVariable(name: "tmp2", scope: !17, file: !3, type: !8)
!50 = !DILocation(line: 53, column: 1, scope: !17)
!51 = !DILocalVariable(scope: !17, file: !3, type: !8, flags: DIFlagArtificial)
!52 = !DILocation(line: 56, column: 1, scope: !17)
!53 = !DILocalVariable(name: "__nv_MAIN__F1L35_1Arg0", arg: 1, scope: !11, file: !3, type: !8)
!54 = !DILocation(line: 0, scope: !11)
!55 = !DILocalVariable(name: "__nv_MAIN__F1L35_1Arg1", arg: 2, scope: !11, file: !3, type: !14)
!56 = !DILocalVariable(name: "__nv_MAIN__F1L35_1Arg2", arg: 3, scope: !11, file: !3, type: !14)
!57 = !DILocalVariable(name: "omp_sched_static", scope: !11, file: !3, type: !8)
!58 = !DILocalVariable(name: "omp_proc_bind_false", scope: !11, file: !3, type: !8)
!59 = !DILocalVariable(name: "omp_proc_bind_true", scope: !11, file: !3, type: !8)
!60 = !DILocalVariable(name: "omp_lock_hint_none", scope: !11, file: !3, type: !8)
!61 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !11, file: !3, type: !8)
!62 = !DILocation(line: 42, column: 1, scope: !11)
!63 = !DILocation(line: 37, column: 1, scope: !11)
!64 = !DILocalVariable(name: "i", scope: !11, file: !3, type: !8)
!65 = !DILocation(line: 38, column: 1, scope: !11)
!66 = !DILocation(line: 39, column: 1, scope: !11)
!67 = !DILocation(line: 40, column: 1, scope: !11)
!68 = !DILocation(line: 41, column: 1, scope: !11)
!69 = distinct !DISubprogram(name: "__nv_MAIN__F1L44_2", scope: !2, file: !3, line: 44, type: !12, scopeLine: 44, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!70 = !DILocalVariable(name: "__nv_MAIN__F1L44_2Arg0", arg: 1, scope: !69, file: !3, type: !8)
!71 = !DILocation(line: 0, scope: !69)
!72 = !DILocalVariable(name: "__nv_MAIN__F1L44_2Arg1", arg: 2, scope: !69, file: !3, type: !14)
!73 = !DILocalVariable(name: "__nv_MAIN__F1L44_2Arg2", arg: 3, scope: !69, file: !3, type: !14)
!74 = !DILocalVariable(name: "omp_sched_static", scope: !69, file: !3, type: !8)
!75 = !DILocalVariable(name: "omp_proc_bind_false", scope: !69, file: !3, type: !8)
!76 = !DILocalVariable(name: "omp_proc_bind_true", scope: !69, file: !3, type: !8)
!77 = !DILocalVariable(name: "omp_lock_hint_none", scope: !69, file: !3, type: !8)
!78 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !69, file: !3, type: !8)
!79 = !DILocation(line: 51, column: 1, scope: !69)
!80 = !DILocation(line: 46, column: 1, scope: !69)
!81 = !DILocalVariable(name: "i", scope: !69, file: !3, type: !8)
!82 = !DILocation(line: 47, column: 1, scope: !69)
!83 = !DILocation(line: 48, column: 1, scope: !69)
!84 = !DILocation(line: 49, column: 1, scope: !69)
!85 = !DILocation(line: 50, column: 1, scope: !69)

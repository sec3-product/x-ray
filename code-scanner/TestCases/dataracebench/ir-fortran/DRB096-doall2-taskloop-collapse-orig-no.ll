; ModuleID = '/tmp/DRB096-doall2-taskloop-collapse-orig-no-c41d2e.ll'
source_filename = "/tmp/DRB096-doall2-taskloop-collapse-orig-no-c41d2e.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.STATICS2 = type <{ [48 x i8] }>
%struct_drb096_2_ = type <{ [16 x i8] }>
%struct_drb096_0_ = type <{ [192 x i8] }>
%astruct.dt68 = type <{ i8*, i8*, i8*, i8*, i8* }>

@.STATICS2 = internal global %struct.STATICS2 <{ [48 x i8] c"\FB\FF\FF\FF\0A\00\00\00a(50,50) =\00\00\EB\FF\FF\FF\00\00\00\00\03\00\00\00\00\00\00\00\01\00\00\00\FF\FF\FF\FF\00\00\00\00" }>, align 16
@.C354_MAIN_ = internal constant i64 50
@.C351_MAIN_ = internal constant i32 6
@.C347_MAIN_ = internal constant [68 x i8] c"micro-benchmarks-fortran/DRB096-doall2-taskloop-collapse-orig-no.f95"
@.C349_MAIN_ = internal constant i32 40
@.C285_MAIN_ = internal constant i32 1
@.C306_MAIN_ = internal constant i32 25
@.C362_MAIN_ = internal constant i64 4
@.C361_MAIN_ = internal constant i64 25
@.C284_MAIN_ = internal constant i64 0
@.C311_MAIN_ = internal constant i64 18
@.C310_MAIN_ = internal constant i64 17
@.C309_MAIN_ = internal constant i64 12
@.C286_MAIN_ = internal constant i64 1
@.C308_MAIN_ = internal constant i64 11
@.C322_MAIN_ = internal constant i32 100
@.C283_MAIN_ = internal constant i32 0
@.C286___nv_MAIN__F1L28_1 = internal constant i64 1
@.C283___nv_MAIN__F1L28_1 = internal constant i32 0
@.C285___nv_MAIN__F1L28_1 = internal constant i32 1
@.C285___nv_MAIN_F1L30_2 = internal constant i32 1
@.C286___nv_MAIN_F1L30_2 = internal constant i64 1
@.C283___nv_MAIN_F1L30_2 = internal constant i32 0
@_drb096_2_ = common global %struct_drb096_2_ zeroinitializer, align 64, !dbg !0, !dbg !19
@_drb096_0_ = common global %struct_drb096_0_ zeroinitializer, align 64, !dbg !7, !dbg !13

; Function Attrs: noinline
define float @drb096_() #0 {
.L.entry:
  ret float undef
}

define void @MAIN_() #1 !dbg !23 {
L.entry:
  %__gtid_MAIN__433 = alloca i32, align 4
  %len_323 = alloca i32, align 4
  %.g0000_409 = alloca i64, align 8
  %.g0001_411 = alloca i64, align 8
  %.uplevelArgPack0001_419 = alloca %astruct.dt68, align 16
  %j_321 = alloca i32, align 4
  %z__io_353 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !28, metadata !DIExpression()), !dbg !29
  call void @llvm.dbg.value(metadata i32 0, metadata !30, metadata !DIExpression()), !dbg !29
  call void @llvm.dbg.value(metadata i32 1, metadata !31, metadata !DIExpression()), !dbg !29
  call void @llvm.dbg.value(metadata i32 0, metadata !32, metadata !DIExpression()), !dbg !29
  call void @llvm.dbg.value(metadata i32 1, metadata !33, metadata !DIExpression()), !dbg !29
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !34
  store i32 %0, i32* %__gtid_MAIN__433, align 4, !dbg !34
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !35
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !35
  call void (i8*, ...) %2(i8* %1), !dbg !35
  br label %L.LB2_376

L.LB2_376:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %len_323, metadata !36, metadata !DIExpression()), !dbg !29
  store i32 100, i32* %len_323, align 4, !dbg !37
  %3 = bitcast %struct_drb096_0_* @_drb096_0_ to i8*, !dbg !38
  %4 = getelementptr i8, i8* %3, i64 96, !dbg !38
  %5 = bitcast i8* %4 to i64*, !dbg !38
  store i64 1, i64* %5, align 8, !dbg !38
  %6 = load i32, i32* %len_323, align 4, !dbg !38
  call void @llvm.dbg.value(metadata i32 %6, metadata !36, metadata !DIExpression()), !dbg !29
  %7 = sext i32 %6 to i64, !dbg !38
  %8 = bitcast %struct_drb096_0_* @_drb096_0_ to i8*, !dbg !38
  %9 = getelementptr i8, i8* %8, i64 104, !dbg !38
  %10 = bitcast i8* %9 to i64*, !dbg !38
  store i64 %7, i64* %10, align 8, !dbg !38
  %11 = bitcast %struct_drb096_0_* @_drb096_0_ to i8*, !dbg !38
  %12 = getelementptr i8, i8* %11, i64 144, !dbg !38
  %13 = bitcast i8* %12 to i64*, !dbg !38
  store i64 1, i64* %13, align 8, !dbg !38
  %14 = load i32, i32* %len_323, align 4, !dbg !38
  call void @llvm.dbg.value(metadata i32 %14, metadata !36, metadata !DIExpression()), !dbg !29
  %15 = sext i32 %14 to i64, !dbg !38
  %16 = bitcast %struct_drb096_0_* @_drb096_0_ to i8*, !dbg !38
  %17 = getelementptr i8, i8* %16, i64 152, !dbg !38
  %18 = bitcast i8* %17 to i64*, !dbg !38
  store i64 %15, i64* %18, align 8, !dbg !38
  %19 = bitcast %struct_drb096_0_* @_drb096_0_ to i8*, !dbg !38
  %20 = getelementptr i8, i8* %19, i64 104, !dbg !38
  %21 = bitcast i8* %20 to i64*, !dbg !38
  %22 = load i64, i64* %21, align 8, !dbg !38
  %23 = sub nsw i64 %22, 1, !dbg !38
  %24 = bitcast %struct_drb096_0_* @_drb096_0_ to i8*, !dbg !38
  %25 = getelementptr i8, i8* %24, i64 96, !dbg !38
  %26 = bitcast i8* %25 to i64*, !dbg !38
  %27 = load i64, i64* %26, align 8, !dbg !38
  %28 = add nsw i64 %23, %27, !dbg !38
  store i64 %28, i64* %.g0000_409, align 8, !dbg !38
  %29 = bitcast %struct_drb096_0_* @_drb096_0_ to i8*, !dbg !38
  %30 = getelementptr i8, i8* %29, i64 152, !dbg !38
  %31 = bitcast i8* %30 to i64*, !dbg !38
  %32 = load i64, i64* %31, align 8, !dbg !38
  %33 = sub nsw i64 %32, 1, !dbg !38
  %34 = bitcast %struct_drb096_0_* @_drb096_0_ to i8*, !dbg !38
  %35 = getelementptr i8, i8* %34, i64 144, !dbg !38
  %36 = bitcast i8* %35 to i64*, !dbg !38
  %37 = load i64, i64* %36, align 8, !dbg !38
  %38 = add nsw i64 %33, %37, !dbg !38
  store i64 %38, i64* %.g0001_411, align 8, !dbg !38
  %39 = bitcast %struct_drb096_0_* @_drb096_0_ to i8*, !dbg !38
  %40 = getelementptr i8, i8* %39, i64 16, !dbg !38
  %41 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !38
  %42 = bitcast i64* @.C361_MAIN_ to i8*, !dbg !38
  %43 = bitcast i64* @.C362_MAIN_ to i8*, !dbg !38
  %44 = bitcast %struct_drb096_0_* @_drb096_0_ to i8*, !dbg !38
  %45 = getelementptr i8, i8* %44, i64 96, !dbg !38
  %46 = bitcast i64* %.g0000_409 to i8*, !dbg !38
  %47 = bitcast %struct_drb096_0_* @_drb096_0_ to i8*, !dbg !38
  %48 = getelementptr i8, i8* %47, i64 144, !dbg !38
  %49 = bitcast i64* %.g0001_411 to i8*, !dbg !38
  %50 = bitcast void (...)* @f90_template2_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !38
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, ...) %50(i8* %40, i8* %41, i8* %42, i8* %43, i8* %45, i8* %46, i8* %48, i8* %49), !dbg !38
  %51 = bitcast %struct_drb096_0_* @_drb096_0_ to i8*, !dbg !38
  %52 = getelementptr i8, i8* %51, i64 16, !dbg !38
  %53 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !38
  call void (i8*, i32, ...) %53(i8* %52, i32 25), !dbg !38
  %54 = bitcast %struct_drb096_0_* @_drb096_0_ to i8*, !dbg !38
  %55 = getelementptr i8, i8* %54, i64 104, !dbg !38
  %56 = bitcast i8* %55 to i64*, !dbg !38
  %57 = load i64, i64* %56, align 8, !dbg !38
  %58 = sub nsw i64 %57, 1, !dbg !38
  %59 = bitcast %struct_drb096_0_* @_drb096_0_ to i8*, !dbg !38
  %60 = getelementptr i8, i8* %59, i64 96, !dbg !38
  %61 = bitcast i8* %60 to i64*, !dbg !38
  %62 = load i64, i64* %61, align 8, !dbg !38
  %63 = add nsw i64 %58, %62, !dbg !38
  %64 = bitcast %struct_drb096_0_* @_drb096_0_ to i8*, !dbg !38
  %65 = getelementptr i8, i8* %64, i64 96, !dbg !38
  %66 = bitcast i8* %65 to i64*, !dbg !38
  %67 = load i64, i64* %66, align 8, !dbg !38
  %68 = sub nsw i64 %67, 1, !dbg !38
  %69 = sub nsw i64 %63, %68, !dbg !38
  %70 = bitcast %struct_drb096_0_* @_drb096_0_ to i8*, !dbg !38
  %71 = getelementptr i8, i8* %70, i64 152, !dbg !38
  %72 = bitcast i8* %71 to i64*, !dbg !38
  %73 = load i64, i64* %72, align 8, !dbg !38
  %74 = sub nsw i64 %73, 1, !dbg !38
  %75 = bitcast %struct_drb096_0_* @_drb096_0_ to i8*, !dbg !38
  %76 = getelementptr i8, i8* %75, i64 144, !dbg !38
  %77 = bitcast i8* %76 to i64*, !dbg !38
  %78 = load i64, i64* %77, align 8, !dbg !38
  %79 = add nsw i64 %74, %78, !dbg !38
  %80 = bitcast %struct_drb096_0_* @_drb096_0_ to i8*, !dbg !38
  %81 = getelementptr i8, i8* %80, i64 144, !dbg !38
  %82 = bitcast i8* %81 to i64*, !dbg !38
  %83 = load i64, i64* %82, align 8, !dbg !38
  %84 = sub nsw i64 %83, 1, !dbg !38
  %85 = sub nsw i64 %79, %84, !dbg !38
  %86 = mul nsw i64 %69, %85, !dbg !38
  store i64 %86, i64* %.g0000_409, align 8, !dbg !38
  %87 = bitcast i64* %.g0000_409 to i8*, !dbg !38
  %88 = bitcast i64* @.C361_MAIN_ to i8*, !dbg !38
  %89 = bitcast i64* @.C362_MAIN_ to i8*, !dbg !38
  %90 = bitcast %struct_drb096_0_* @_drb096_0_ to i8*, !dbg !38
  %91 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !38
  %92 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !38
  %93 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !38
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %93(i8* %87, i8* %88, i8* %89, i8* null, i8* %90, i8* null, i8* %91, i8* %92, i8* null, i64 0), !dbg !38
  %94 = bitcast i32* %len_323 to i8*, !dbg !39
  %95 = bitcast %astruct.dt68* %.uplevelArgPack0001_419 to i8**, !dbg !39
  store i8* %94, i8** %95, align 8, !dbg !39
  call void @llvm.dbg.declare(metadata i32* %j_321, metadata !40, metadata !DIExpression()), !dbg !29
  %96 = bitcast i32* %j_321 to i8*, !dbg !39
  %97 = bitcast %astruct.dt68* %.uplevelArgPack0001_419 to i8*, !dbg !39
  %98 = getelementptr i8, i8* %97, i64 8, !dbg !39
  %99 = bitcast i8* %98 to i8**, !dbg !39
  store i8* %96, i8** %99, align 8, !dbg !39
  %100 = bitcast %struct_drb096_0_* @_drb096_0_ to i8*, !dbg !39
  %101 = bitcast %astruct.dt68* %.uplevelArgPack0001_419 to i8*, !dbg !39
  %102 = getelementptr i8, i8* %101, i64 16, !dbg !39
  %103 = bitcast i8* %102 to i8**, !dbg !39
  store i8* %100, i8** %103, align 8, !dbg !39
  %104 = bitcast %struct_drb096_0_* @_drb096_0_ to i8*, !dbg !39
  %105 = getelementptr i8, i8* %104, i64 16, !dbg !39
  %106 = bitcast %astruct.dt68* %.uplevelArgPack0001_419 to i8*, !dbg !39
  %107 = getelementptr i8, i8* %106, i64 24, !dbg !39
  %108 = bitcast i8* %107 to i8**, !dbg !39
  store i8* %105, i8** %108, align 8, !dbg !39
  %109 = bitcast %struct_drb096_0_* @_drb096_0_ to i8*, !dbg !39
  %110 = bitcast %astruct.dt68* %.uplevelArgPack0001_419 to i8*, !dbg !39
  %111 = getelementptr i8, i8* %110, i64 32, !dbg !39
  %112 = bitcast i8* %111 to i8**, !dbg !39
  store i8* %109, i8** %112, align 8, !dbg !39
  br label %L.LB2_431, !dbg !39

L.LB2_431:                                        ; preds = %L.LB2_376
  %113 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L28_1_ to i64*, !dbg !39
  %114 = bitcast %astruct.dt68* %.uplevelArgPack0001_419 to i64*, !dbg !39
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %113, i64* %114), !dbg !39
  call void (...) @_mp_bcs_nest(), !dbg !41
  %115 = bitcast i32* @.C349_MAIN_ to i8*, !dbg !41
  %116 = bitcast [68 x i8]* @.C347_MAIN_ to i8*, !dbg !41
  %117 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !41
  call void (i8*, i8*, i64, ...) %117(i8* %115, i8* %116, i64 68), !dbg !41
  %118 = bitcast i32* @.C351_MAIN_ to i8*, !dbg !41
  %119 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !41
  %120 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !41
  %121 = bitcast %struct.STATICS2* @.STATICS2 to i8*, !dbg !41
  %122 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !41
  %123 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %122(i8* %118, i8* null, i8* %119, i8* %120, i8* %121, i8* null, i64 0), !dbg !41
  call void @llvm.dbg.declare(metadata i32* %z__io_353, metadata !42, metadata !DIExpression()), !dbg !29
  store i32 %123, i32* %z__io_353, align 4, !dbg !41
  %124 = bitcast %struct_drb096_0_* @_drb096_0_ to i8*, !dbg !41
  %125 = getelementptr i8, i8* %124, i64 176, !dbg !41
  %126 = bitcast i8* %125 to i64*, !dbg !41
  %127 = load i64, i64* %126, align 8, !dbg !41
  %128 = mul nsw i64 %127, 50, !dbg !41
  %129 = bitcast %struct_drb096_0_* @_drb096_0_ to i8*, !dbg !41
  %130 = getelementptr i8, i8* %129, i64 72, !dbg !41
  %131 = bitcast i8* %130 to i64*, !dbg !41
  %132 = load i64, i64* %131, align 8, !dbg !41
  %133 = add nsw i64 %128, %132, !dbg !41
  %134 = bitcast %struct_drb096_0_* @_drb096_0_ to i8**, !dbg !41
  %135 = load i8*, i8** %134, align 8, !dbg !41
  %136 = getelementptr i8, i8* %135, i64 196, !dbg !41
  %137 = bitcast i8* %136 to i32*, !dbg !41
  %138 = getelementptr i32, i32* %137, i64 %133, !dbg !41
  %139 = load i32, i32* %138, align 4, !dbg !41
  %140 = bitcast i32 (...)* @f90io_sc_i_fmt_write to i32 (i32, i32, ...)*, !dbg !41
  %141 = call i32 (i32, i32, ...) %140(i32 %139, i32 25), !dbg !41
  store i32 %141, i32* %z__io_353, align 4, !dbg !41
  %142 = call i32 (...) @f90io_fmtw_end(), !dbg !41
  store i32 %142, i32* %z__io_353, align 4, !dbg !41
  call void (...) @_mp_ecs_nest(), !dbg !41
  ret void, !dbg !34
}

define internal void @__nv_MAIN__F1L28_1_(i32* %__nv_MAIN__F1L28_1Arg0, i64* %__nv_MAIN__F1L28_1Arg1, i64* %__nv_MAIN__F1L28_1Arg2) #1 !dbg !43 {
L.entry:
  %__gtid___nv_MAIN__F1L28_1__479 = alloca i32, align 4
  %.s0000_474 = alloca i32, align 4
  %.s0001_475 = alloca i32, align 4
  %.s0002_485 = alloca i32, align 4
  %.z0415_484 = alloca i8*, align 8
  %.Xd0000p_335 = alloca i64, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L28_1Arg0, metadata !46, metadata !DIExpression()), !dbg !47
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L28_1Arg1, metadata !48, metadata !DIExpression()), !dbg !47
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L28_1Arg2, metadata !49, metadata !DIExpression()), !dbg !47
  call void @llvm.dbg.value(metadata i32 1, metadata !50, metadata !DIExpression()), !dbg !47
  call void @llvm.dbg.value(metadata i32 0, metadata !51, metadata !DIExpression()), !dbg !47
  call void @llvm.dbg.value(metadata i32 1, metadata !52, metadata !DIExpression()), !dbg !47
  call void @llvm.dbg.value(metadata i32 0, metadata !53, metadata !DIExpression()), !dbg !47
  call void @llvm.dbg.value(metadata i32 1, metadata !54, metadata !DIExpression()), !dbg !47
  %0 = load i32, i32* %__nv_MAIN__F1L28_1Arg0, align 4, !dbg !55
  store i32 %0, i32* %__gtid___nv_MAIN__F1L28_1__479, align 4, !dbg !55
  br label %L.LB3_473

L.LB3_473:                                        ; preds = %L.entry
  br label %L.LB3_326

L.LB3_326:                                        ; preds = %L.LB3_473
  store i32 -1, i32* %.s0000_474, align 4, !dbg !56
  store i32 0, i32* %.s0001_475, align 4, !dbg !56
  %1 = load i32, i32* %__gtid___nv_MAIN__F1L28_1__479, align 4, !dbg !56
  %2 = call i32 @__kmpc_single(i64* null, i32 %1), !dbg !56
  %3 = icmp eq i32 %2, 0, !dbg !56
  br i1 %3, label %L.LB3_369, label %L.LB3_328, !dbg !56

L.LB3_328:                                        ; preds = %L.LB3_326
  store i32 1, i32* %.s0002_485, align 4, !dbg !57
  %4 = load i32, i32* %__gtid___nv_MAIN__F1L28_1__479, align 4, !dbg !58
  %5 = load i32, i32* %.s0002_485, align 4, !dbg !58
  %6 = bitcast void (i32, i64*)* @__nv_MAIN_F1L30_2_ to i64*, !dbg !58
  %7 = call i8* @__kmpc_omp_task_alloc(i64* null, i32 %4, i32 %5, i32 72, i32 40, i64* %6), !dbg !58
  store i8* %7, i8** %.z0415_484, align 8, !dbg !58
  %8 = load i64, i64* %__nv_MAIN__F1L28_1Arg2, align 8, !dbg !58
  %9 = load i8*, i8** %.z0415_484, align 8, !dbg !58
  %10 = bitcast i8* %9 to i64**, !dbg !58
  %11 = load i64*, i64** %10, align 8, !dbg !58
  store i64 %8, i64* %11, align 8, !dbg !58
  %12 = bitcast i64* %__nv_MAIN__F1L28_1Arg2 to i8*, !dbg !55
  %13 = getelementptr i8, i8* %12, i64 8, !dbg !55
  %14 = bitcast i8* %13 to i64*, !dbg !55
  %15 = load i64, i64* %14, align 8, !dbg !55
  %16 = bitcast i64* %11 to i8*, !dbg !55
  %17 = getelementptr i8, i8* %16, i64 8, !dbg !55
  %18 = bitcast i8* %17 to i64*, !dbg !55
  store i64 %15, i64* %18, align 8, !dbg !55
  %19 = bitcast i64* %__nv_MAIN__F1L28_1Arg2 to i8*, !dbg !55
  %20 = getelementptr i8, i8* %19, i64 16, !dbg !55
  %21 = bitcast i8* %20 to i64*, !dbg !55
  %22 = load i64, i64* %21, align 8, !dbg !55
  %23 = bitcast i64* %11 to i8*, !dbg !55
  %24 = getelementptr i8, i8* %23, i64 16, !dbg !55
  %25 = bitcast i8* %24 to i64*, !dbg !55
  store i64 %22, i64* %25, align 8, !dbg !55
  %26 = bitcast i64* %__nv_MAIN__F1L28_1Arg2 to i8*, !dbg !55
  %27 = getelementptr i8, i8* %26, i64 24, !dbg !55
  %28 = bitcast i8* %27 to i64*, !dbg !55
  %29 = load i64, i64* %28, align 8, !dbg !55
  %30 = bitcast i64* %11 to i8*, !dbg !55
  %31 = getelementptr i8, i8* %30, i64 24, !dbg !55
  %32 = bitcast i8* %31 to i64*, !dbg !55
  store i64 %29, i64* %32, align 8, !dbg !55
  %33 = bitcast i64* %__nv_MAIN__F1L28_1Arg2 to i8*, !dbg !55
  %34 = getelementptr i8, i8* %33, i64 32, !dbg !55
  %35 = bitcast i8* %34 to i64*, !dbg !55
  %36 = load i64, i64* %35, align 8, !dbg !55
  %37 = bitcast i64* %11 to i8*, !dbg !55
  %38 = getelementptr i8, i8* %37, i64 32, !dbg !55
  %39 = bitcast i8* %38 to i64*, !dbg !55
  store i64 %36, i64* %39, align 8, !dbg !55
  %40 = load i8*, i8** %.z0415_484, align 8, !dbg !58
  %41 = getelementptr i8, i8* %40, i64 40, !dbg !58
  %42 = bitcast i8* %41 to i64*, !dbg !58
  store i64 1, i64* %42, align 8, !dbg !58
  %43 = load i64, i64* %.Xd0000p_335, align 8, !dbg !58
  %44 = load i8*, i8** %.z0415_484, align 8, !dbg !58
  %45 = getelementptr i8, i8* %44, i64 48, !dbg !58
  %46 = bitcast i8* %45 to i64*, !dbg !58
  store i64 %43, i64* %46, align 8, !dbg !58
  %47 = load i8*, i8** %.z0415_484, align 8, !dbg !58
  %48 = getelementptr i8, i8* %47, i64 56, !dbg !58
  %49 = bitcast i8* %48 to i64*, !dbg !58
  store i64 1, i64* %49, align 8, !dbg !58
  %50 = load i32, i32* %__gtid___nv_MAIN__F1L28_1__479, align 4, !dbg !58
  %51 = load i8*, i8** %.z0415_484, align 8, !dbg !58
  %52 = bitcast i8* %51 to i64*, !dbg !58
  %53 = load i8*, i8** %.z0415_484, align 8, !dbg !58
  %54 = getelementptr i8, i8* %53, i64 40, !dbg !58
  %55 = bitcast i8* %54 to i64*, !dbg !58
  %56 = load i8*, i8** %.z0415_484, align 8, !dbg !58
  %57 = getelementptr i8, i8* %56, i64 48, !dbg !58
  %58 = bitcast i8* %57 to i64*, !dbg !58
  %59 = sext i32 0 to i64, !dbg !58
  call void @__kmpc_taskloop(i64* null, i32 %50, i64* %52, i32 1, i64* %55, i64* %58, i64 1, i32 0, i32 0, i64 %59, i64* null), !dbg !58
  br label %L.LB3_370

L.LB3_370:                                        ; preds = %L.LB3_328
  %60 = load i32, i32* %__gtid___nv_MAIN__F1L28_1__479, align 4, !dbg !59
  store i32 %60, i32* %.s0000_474, align 4, !dbg !59
  store i32 1, i32* %.s0001_475, align 4, !dbg !59
  %61 = load i32, i32* %__gtid___nv_MAIN__F1L28_1__479, align 4, !dbg !59
  call void @__kmpc_end_single(i64* null, i32 %61), !dbg !59
  br label %L.LB3_369

L.LB3_369:                                        ; preds = %L.LB3_370, %L.LB3_326
  br label %L.LB3_344

L.LB3_344:                                        ; preds = %L.LB3_369
  %62 = load i32, i32* %__gtid___nv_MAIN__F1L28_1__479, align 4, !dbg !59
  call void @__kmpc_barrier(i64* null, i32 %62), !dbg !59
  br label %L.LB3_345

L.LB3_345:                                        ; preds = %L.LB3_344
  ret void, !dbg !55
}

define internal void @__nv_MAIN_F1L30_2_(i32 %__nv_MAIN_F1L30_2Arg0.arg, i64* %__nv_MAIN_F1L30_2Arg1) #1 !dbg !60 {
L.entry:
  %__nv_MAIN_F1L30_2Arg0.addr = alloca i32, align 4
  %.S0000_534 = alloca i8*, align 8
  %.i0000p_333 = alloca i32, align 4
  %.Xc0000p_334 = alloca i64, align 8
  %.Xd0000p_335 = alloca i64, align 8
  %.Xc0001p_341 = alloca i64, align 8
  %.i0001p_342 = alloca i32, align 4
  %.id0000p_336 = alloca i64, align 8
  %.dU0001p_374 = alloca i64, align 8
  %.dY0001p_373 = alloca i64, align 8
  %.Xg0000p_339 = alloca i64, align 8
  %.Xe0000p_337 = alloca i64, align 8
  %.Xf0000p_338 = alloca i64, align 8
  %j_340 = alloca i32, align 4
  %i_332 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L30_2Arg0.addr, metadata !63, metadata !DIExpression()), !dbg !64
  store i32 %__nv_MAIN_F1L30_2Arg0.arg, i32* %__nv_MAIN_F1L30_2Arg0.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L30_2Arg0.addr, metadata !65, metadata !DIExpression()), !dbg !64
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L30_2Arg1, metadata !66, metadata !DIExpression()), !dbg !64
  call void @llvm.dbg.value(metadata i32 1, metadata !67, metadata !DIExpression()), !dbg !64
  call void @llvm.dbg.value(metadata i32 0, metadata !68, metadata !DIExpression()), !dbg !64
  call void @llvm.dbg.value(metadata i32 1, metadata !69, metadata !DIExpression()), !dbg !64
  call void @llvm.dbg.value(metadata i32 0, metadata !70, metadata !DIExpression()), !dbg !64
  call void @llvm.dbg.value(metadata i32 1, metadata !71, metadata !DIExpression()), !dbg !64
  %0 = bitcast i64* %__nv_MAIN_F1L30_2Arg1 to i8**, !dbg !72
  %1 = load i8*, i8** %0, align 8, !dbg !72
  store i8* %1, i8** %.S0000_534, align 8, !dbg !72
  br label %L.LB5_538

L.LB5_538:                                        ; preds = %L.entry
  br label %L.LB5_331

L.LB5_331:                                        ; preds = %L.LB5_538
  store i32 0, i32* %.i0000p_333, align 4, !dbg !73
  %2 = load i8*, i8** %.S0000_534, align 8, !dbg !73
  %3 = bitcast i8* %2 to i32**, !dbg !73
  %4 = load i32*, i32** %3, align 8, !dbg !73
  %5 = load i32, i32* %4, align 4, !dbg !73
  %6 = sext i32 %5 to i64, !dbg !73
  store i64 %6, i64* %.Xc0000p_334, align 8, !dbg !73
  %7 = load i64, i64* %.Xc0000p_334, align 8, !dbg !73
  store i64 %7, i64* %.Xd0000p_335, align 8, !dbg !73
  %8 = load i8*, i8** %.S0000_534, align 8, !dbg !74
  %9 = bitcast i8* %8 to i32**, !dbg !74
  %10 = load i32*, i32** %9, align 8, !dbg !74
  %11 = load i32, i32* %10, align 4, !dbg !74
  %12 = sext i32 %11 to i64, !dbg !74
  store i64 %12, i64* %.Xc0001p_341, align 8, !dbg !74
  %13 = load i64, i64* %.Xc0001p_341, align 8, !dbg !74
  %14 = load i64, i64* %.Xd0000p_335, align 8, !dbg !74
  %15 = mul nsw i64 %13, %14, !dbg !74
  store i64 %15, i64* %.Xd0000p_335, align 8, !dbg !74
  store i32 0, i32* %.i0001p_342, align 4, !dbg !74
  %16 = bitcast i64* %__nv_MAIN_F1L30_2Arg1 to i8*, !dbg !74
  %17 = getelementptr i8, i8* %16, i64 40, !dbg !74
  %18 = bitcast i8* %17 to i64*, !dbg !74
  %19 = load i64, i64* %18, align 8, !dbg !74
  store i64 %19, i64* %.id0000p_336, align 8, !dbg !74
  %20 = bitcast i64* %__nv_MAIN_F1L30_2Arg1 to i8*, !dbg !74
  %21 = getelementptr i8, i8* %20, i64 48, !dbg !74
  %22 = bitcast i8* %21 to i64*, !dbg !74
  %23 = load i64, i64* %22, align 8, !dbg !74
  store i64 %23, i64* %.dU0001p_374, align 8, !dbg !74
  %24 = bitcast i64* %__nv_MAIN_F1L30_2Arg1 to i8*, !dbg !74
  %25 = getelementptr i8, i8* %24, i64 64, !dbg !74
  %26 = bitcast i8* %25 to i32*, !dbg !74
  %27 = load i32, i32* %26, align 4, !dbg !74
  store i32 %27, i32* %.i0001p_342, align 4, !dbg !74
  %28 = load i64, i64* %.dU0001p_374, align 8, !dbg !74
  %29 = load i64, i64* %.id0000p_336, align 8, !dbg !74
  %30 = sub nsw i64 %28, %29, !dbg !74
  %31 = add nsw i64 %30, 1, !dbg !74
  store i64 %31, i64* %.dY0001p_373, align 8, !dbg !74
  %32 = load i64, i64* %.dY0001p_373, align 8, !dbg !74
  %33 = icmp sle i64 %32, 0, !dbg !74
  br i1 %33, label %L.LB5_372, label %L.LB5_371, !dbg !74

L.LB5_371:                                        ; preds = %L.LB5_371, %L.LB5_331
  %34 = load i64, i64* %.id0000p_336, align 8, !dbg !74
  %35 = sub nsw i64 %34, 1, !dbg !74
  store i64 %35, i64* %.Xg0000p_339, align 8, !dbg !74
  %36 = load i64, i64* %.Xg0000p_339, align 8, !dbg !74
  %37 = load i64, i64* %.Xc0001p_341, align 8, !dbg !74
  %38 = sdiv i64 %36, %37, !dbg !74
  store i64 %38, i64* %.Xe0000p_337, align 8, !dbg !74
  %39 = load i64, i64* %.Xg0000p_339, align 8, !dbg !74
  %40 = load i64, i64* %.Xc0001p_341, align 8, !dbg !74
  %41 = load i64, i64* %.Xe0000p_337, align 8, !dbg !74
  %42 = mul nsw i64 %40, %41, !dbg !74
  %43 = sub nsw i64 %39, %42, !dbg !74
  store i64 %43, i64* %.Xf0000p_338, align 8, !dbg !74
  %44 = load i64, i64* %.Xf0000p_338, align 8, !dbg !74
  %45 = trunc i64 %44 to i32, !dbg !74
  %46 = add nsw i32 %45, 1, !dbg !74
  call void @llvm.dbg.declare(metadata i32* %j_340, metadata !75, metadata !DIExpression()), !dbg !76
  store i32 %46, i32* %j_340, align 4, !dbg !74
  %47 = load i64, i64* %.Xe0000p_337, align 8, !dbg !74
  store i64 %47, i64* %.Xg0000p_339, align 8, !dbg !74
  %48 = load i64, i64* %.Xg0000p_339, align 8, !dbg !74
  %49 = load i64, i64* %.Xc0000p_334, align 8, !dbg !74
  %50 = sdiv i64 %48, %49, !dbg !74
  store i64 %50, i64* %.Xe0000p_337, align 8, !dbg !74
  %51 = load i64, i64* %.Xg0000p_339, align 8, !dbg !74
  %52 = load i64, i64* %.Xc0000p_334, align 8, !dbg !74
  %53 = load i64, i64* %.Xe0000p_337, align 8, !dbg !74
  %54 = mul nsw i64 %52, %53, !dbg !74
  %55 = sub nsw i64 %51, %54, !dbg !74
  store i64 %55, i64* %.Xf0000p_338, align 8, !dbg !74
  %56 = load i64, i64* %.Xf0000p_338, align 8, !dbg !74
  %57 = trunc i64 %56 to i32, !dbg !74
  %58 = add nsw i32 %57, 1, !dbg !74
  call void @llvm.dbg.declare(metadata i32* %i_332, metadata !77, metadata !DIExpression()), !dbg !76
  store i32 %58, i32* %i_332, align 4, !dbg !74
  %59 = load i32, i32* %i_332, align 4, !dbg !78
  call void @llvm.dbg.value(metadata i32 %59, metadata !77, metadata !DIExpression()), !dbg !76
  %60 = sext i32 %59 to i64, !dbg !78
  %61 = load i32, i32* %j_340, align 4, !dbg !78
  call void @llvm.dbg.value(metadata i32 %61, metadata !75, metadata !DIExpression()), !dbg !76
  %62 = sext i32 %61 to i64, !dbg !78
  %63 = load i8*, i8** %.S0000_534, align 8, !dbg !78
  %64 = getelementptr i8, i8* %63, i64 24, !dbg !78
  %65 = bitcast i8* %64 to i8**, !dbg !78
  %66 = load i8*, i8** %65, align 8, !dbg !78
  %67 = getelementptr i8, i8* %66, i64 160, !dbg !78
  %68 = bitcast i8* %67 to i64*, !dbg !78
  %69 = load i64, i64* %68, align 8, !dbg !78
  %70 = mul nsw i64 %62, %69, !dbg !78
  %71 = add nsw i64 %60, %70, !dbg !78
  %72 = load i8*, i8** %.S0000_534, align 8, !dbg !78
  %73 = getelementptr i8, i8* %72, i64 24, !dbg !78
  %74 = bitcast i8* %73 to i8**, !dbg !78
  %75 = load i8*, i8** %74, align 8, !dbg !78
  %76 = getelementptr i8, i8* %75, i64 56, !dbg !78
  %77 = bitcast i8* %76 to i64*, !dbg !78
  %78 = load i64, i64* %77, align 8, !dbg !78
  %79 = add nsw i64 %71, %78, !dbg !78
  %80 = load i8*, i8** %.S0000_534, align 8, !dbg !78
  %81 = getelementptr i8, i8* %80, i64 32, !dbg !78
  %82 = bitcast i8* %81 to i8***, !dbg !78
  %83 = load i8**, i8*** %82, align 8, !dbg !78
  %84 = load i8*, i8** %83, align 8, !dbg !78
  %85 = getelementptr i8, i8* %84, i64 -4, !dbg !78
  %86 = bitcast i8* %85 to i32*, !dbg !78
  %87 = getelementptr i32, i32* %86, i64 %79, !dbg !78
  %88 = load i32, i32* %87, align 4, !dbg !78
  %89 = add nsw i32 %88, 1, !dbg !78
  %90 = load i32, i32* %i_332, align 4, !dbg !78
  call void @llvm.dbg.value(metadata i32 %90, metadata !77, metadata !DIExpression()), !dbg !76
  %91 = sext i32 %90 to i64, !dbg !78
  %92 = load i32, i32* %j_340, align 4, !dbg !78
  call void @llvm.dbg.value(metadata i32 %92, metadata !75, metadata !DIExpression()), !dbg !76
  %93 = sext i32 %92 to i64, !dbg !78
  %94 = load i8*, i8** %.S0000_534, align 8, !dbg !78
  %95 = getelementptr i8, i8* %94, i64 24, !dbg !78
  %96 = bitcast i8* %95 to i8**, !dbg !78
  %97 = load i8*, i8** %96, align 8, !dbg !78
  %98 = getelementptr i8, i8* %97, i64 160, !dbg !78
  %99 = bitcast i8* %98 to i64*, !dbg !78
  %100 = load i64, i64* %99, align 8, !dbg !78
  %101 = mul nsw i64 %93, %100, !dbg !78
  %102 = add nsw i64 %91, %101, !dbg !78
  %103 = load i8*, i8** %.S0000_534, align 8, !dbg !78
  %104 = getelementptr i8, i8* %103, i64 24, !dbg !78
  %105 = bitcast i8* %104 to i8**, !dbg !78
  %106 = load i8*, i8** %105, align 8, !dbg !78
  %107 = getelementptr i8, i8* %106, i64 56, !dbg !78
  %108 = bitcast i8* %107 to i64*, !dbg !78
  %109 = load i64, i64* %108, align 8, !dbg !78
  %110 = add nsw i64 %102, %109, !dbg !78
  %111 = load i8*, i8** %.S0000_534, align 8, !dbg !78
  %112 = getelementptr i8, i8* %111, i64 32, !dbg !78
  %113 = bitcast i8* %112 to i8***, !dbg !78
  %114 = load i8**, i8*** %113, align 8, !dbg !78
  %115 = load i8*, i8** %114, align 8, !dbg !78
  %116 = getelementptr i8, i8* %115, i64 -4, !dbg !78
  %117 = bitcast i8* %116 to i32*, !dbg !78
  %118 = getelementptr i32, i32* %117, i64 %110, !dbg !78
  store i32 %89, i32* %118, align 4, !dbg !78
  %119 = load i64, i64* %.id0000p_336, align 8, !dbg !76
  %120 = add nsw i64 %119, 1, !dbg !76
  store i64 %120, i64* %.id0000p_336, align 8, !dbg !76
  %121 = load i64, i64* %.dY0001p_373, align 8, !dbg !76
  %122 = sub nsw i64 %121, 1, !dbg !76
  store i64 %122, i64* %.dY0001p_373, align 8, !dbg !76
  %123 = load i64, i64* %.dY0001p_373, align 8, !dbg !76
  %124 = icmp sgt i64 %123, 0, !dbg !76
  br i1 %124, label %L.LB5_371, label %L.LB5_372, !dbg !76

L.LB5_372:                                        ; preds = %L.LB5_371, %L.LB5_331
  br label %L.LB5_343

L.LB5_343:                                        ; preds = %L.LB5_372
  ret void, !dbg !76
}

declare void @__kmpc_barrier(i64*, i32) #1

declare void @__kmpc_end_single(i64*, i32) #1

declare void @__kmpc_taskloop(i64*, i32, i64*, i32, i64*, i64*, i64, i32, i32, i64, i64*) #1

declare i8* @__kmpc_omp_task_alloc(i64*, i32, i32, i32, i32, i64*) #1

declare signext i32 @__kmpc_single(i64*, i32) #1

declare void @_mp_ecs_nest(...) #1

declare signext i32 @f90io_fmtw_end(...) #1

declare signext i32 @f90io_sc_i_fmt_write(...) #1

declare signext i32 @f90io_fmtw_inita(...) #1

declare void @f90io_src_info03a(...) #1

declare void @_mp_bcs_nest(...) #1

declare void @f90_alloc04_chka_i8(...) #1

declare void @f90_set_intrin_type_i8(...) #1

declare void @f90_template2_i8(...) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

declare void @fort_init(...) #1

declare signext i32 @__kmpc_global_thread_num(i64*) #1

declare void @__kmpc_fork_call(i64*, i32, i64*, i64*, ...) #1

declare void @__kmpc_end_serialized_parallel(i64*, i32) #1

declare void @__kmpc_serialized_parallel(i64*, i32) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { noinline }
attributes #1 = { "no-frame-pointer-elim-non-leaf" }
attributes #2 = { nounwind readnone speculatable }

!llvm.module.flags = !{!26, !27}
!llvm.dbg.cu = !{!3}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "z_b_0", scope: !2, file: !4, type: !16, isLocal: false, isDefinition: true)
!2 = !DIModule(scope: !3, name: "drb096")
!3 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !4, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !6, imports: !21)
!4 = !DIFile(filename: "micro-benchmarks-fortran/DRB096-doall2-taskloop-collapse-orig-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!5 = !{}
!6 = !{!7, !13, !0, !19}
!7 = !DIGlobalVariableExpression(var: !8, expr: !DIExpression(DW_OP_deref))
!8 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !4, type: !9, isLocal: false, isDefinition: true)
!9 = !DICompositeType(tag: DW_TAG_array_type, baseType: !10, size: 32, align: 32, elements: !11)
!10 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!11 = !{!12, !12}
!12 = !DISubrange(count: 0, lowerBound: 1)
!13 = !DIGlobalVariableExpression(var: !14, expr: !DIExpression(DW_OP_plus_uconst, 16))
!14 = distinct !DIGlobalVariable(name: "a$sd", scope: !2, file: !4, type: !15, isLocal: false, isDefinition: true)
!15 = !DICompositeType(tag: DW_TAG_array_type, baseType: !16, size: 1408, align: 64, elements: !17)
!16 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!17 = !{!18}
!18 = !DISubrange(count: 22, lowerBound: 1)
!19 = !DIGlobalVariableExpression(var: !20, expr: !DIExpression(DW_OP_plus_uconst, 8))
!20 = distinct !DIGlobalVariable(name: "z_b_1", scope: !2, file: !4, type: !16, isLocal: false, isDefinition: true)
!21 = !{!22}
!22 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !23, entity: !2, file: !4, line: 18)
!23 = distinct !DISubprogram(name: "drb096_doall2_taskloop_collapse_orig_no", scope: !3, file: !4, line: 18, type: !24, scopeLine: 18, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !3)
!24 = !DISubroutineType(cc: DW_CC_program, types: !25)
!25 = !{null}
!26 = !{i32 2, !"Dwarf Version", i32 4}
!27 = !{i32 2, !"Debug Info Version", i32 3}
!28 = !DILocalVariable(name: "omp_sched_static", scope: !23, file: !4, type: !10)
!29 = !DILocation(line: 0, scope: !23)
!30 = !DILocalVariable(name: "omp_proc_bind_false", scope: !23, file: !4, type: !10)
!31 = !DILocalVariable(name: "omp_proc_bind_true", scope: !23, file: !4, type: !10)
!32 = !DILocalVariable(name: "omp_lock_hint_none", scope: !23, file: !4, type: !10)
!33 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !23, file: !4, type: !10)
!34 = !DILocation(line: 43, column: 1, scope: !23)
!35 = !DILocation(line: 18, column: 1, scope: !23)
!36 = !DILocalVariable(name: "len", scope: !23, file: !4, type: !10)
!37 = !DILocation(line: 24, column: 1, scope: !23)
!38 = !DILocation(line: 26, column: 1, scope: !23)
!39 = !DILocation(line: 28, column: 1, scope: !23)
!40 = !DILocalVariable(name: "j", scope: !23, file: !4, type: !10)
!41 = !DILocation(line: 40, column: 1, scope: !23)
!42 = !DILocalVariable(scope: !23, file: !4, type: !10, flags: DIFlagArtificial)
!43 = distinct !DISubprogram(name: "__nv_MAIN__F1L28_1", scope: !3, file: !4, line: 28, type: !44, scopeLine: 28, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !3)
!44 = !DISubroutineType(types: !45)
!45 = !{null, !10, !16, !16}
!46 = !DILocalVariable(name: "__nv_MAIN__F1L28_1Arg0", arg: 1, scope: !43, file: !4, type: !10)
!47 = !DILocation(line: 0, scope: !43)
!48 = !DILocalVariable(name: "__nv_MAIN__F1L28_1Arg1", arg: 2, scope: !43, file: !4, type: !16)
!49 = !DILocalVariable(name: "__nv_MAIN__F1L28_1Arg2", arg: 3, scope: !43, file: !4, type: !16)
!50 = !DILocalVariable(name: "omp_sched_static", scope: !43, file: !4, type: !10)
!51 = !DILocalVariable(name: "omp_proc_bind_false", scope: !43, file: !4, type: !10)
!52 = !DILocalVariable(name: "omp_proc_bind_true", scope: !43, file: !4, type: !10)
!53 = !DILocalVariable(name: "omp_lock_hint_none", scope: !43, file: !4, type: !10)
!54 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !43, file: !4, type: !10)
!55 = !DILocation(line: 38, column: 1, scope: !43)
!56 = !DILocation(line: 29, column: 1, scope: !43)
!57 = !DILocation(line: 30, column: 1, scope: !43)
!58 = !DILocation(line: 35, column: 1, scope: !43)
!59 = !DILocation(line: 37, column: 1, scope: !43)
!60 = distinct !DISubprogram(name: "__nv_MAIN_F1L30_2", scope: !3, file: !4, line: 30, type: !61, scopeLine: 30, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !3)
!61 = !DISubroutineType(types: !62)
!62 = !{null, !10, !16}
!63 = !DILocalVariable(name: "__nv_MAIN_F1L30_2Arg0", scope: !60, file: !4, type: !10)
!64 = !DILocation(line: 0, scope: !60)
!65 = !DILocalVariable(name: "__nv_MAIN_F1L30_2Arg0", arg: 1, scope: !60, file: !4, type: !10)
!66 = !DILocalVariable(name: "__nv_MAIN_F1L30_2Arg1", arg: 2, scope: !60, file: !4, type: !16)
!67 = !DILocalVariable(name: "omp_sched_static", scope: !60, file: !4, type: !10)
!68 = !DILocalVariable(name: "omp_proc_bind_false", scope: !60, file: !4, type: !10)
!69 = !DILocalVariable(name: "omp_proc_bind_true", scope: !60, file: !4, type: !10)
!70 = !DILocalVariable(name: "omp_lock_hint_none", scope: !60, file: !4, type: !10)
!71 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !60, file: !4, type: !10)
!72 = !DILocation(line: 30, column: 1, scope: !60)
!73 = !DILocation(line: 31, column: 1, scope: !60)
!74 = !DILocation(line: 32, column: 1, scope: !60)
!75 = !DILocalVariable(name: "j", scope: !60, file: !4, type: !10)
!76 = !DILocation(line: 35, column: 1, scope: !60)
!77 = !DILocalVariable(name: "i", scope: !60, file: !4, type: !10)
!78 = !DILocation(line: 33, column: 1, scope: !60)

; ModuleID = '/tmp/DRB095-doall2-taskloop-orig-yes-c0b16b.ll'
source_filename = "/tmp/DRB095-doall2-taskloop-orig-yes-c0b16b.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.STATICS2 = type <{ [48 x i8] }>
%struct_drb095_2_ = type <{ [16 x i8] }>
%struct_drb095_0_ = type <{ [192 x i8] }>
%astruct.dt68 = type <{ i8*, i8*, i8*, i8*, i8* }>

@.STATICS2 = internal global %struct.STATICS2 <{ [48 x i8] c"\FB\FF\FF\FF\0A\00\00\00a(50,50) =\00\00\EB\FF\FF\FF\00\00\00\00\03\00\00\00\00\00\00\00\01\00\00\00\FF\FF\FF\FF\00\00\00\00" }>, align 16
@.C346_MAIN_ = internal constant i64 50
@.C343_MAIN_ = internal constant i32 6
@.C339_MAIN_ = internal constant [60 x i8] c"micro-benchmarks-fortran/DRB095-doall2-taskloop-orig-yes.f95"
@.C341_MAIN_ = internal constant i32 44
@.C285_MAIN_ = internal constant i32 1
@.C306_MAIN_ = internal constant i32 25
@.C354_MAIN_ = internal constant i64 4
@.C353_MAIN_ = internal constant i64 25
@.C284_MAIN_ = internal constant i64 0
@.C311_MAIN_ = internal constant i64 18
@.C310_MAIN_ = internal constant i64 17
@.C309_MAIN_ = internal constant i64 12
@.C286_MAIN_ = internal constant i64 1
@.C308_MAIN_ = internal constant i64 11
@.C322_MAIN_ = internal constant i32 100
@.C283_MAIN_ = internal constant i32 0
@.C283___nv_MAIN__F1L32_1 = internal constant i32 0
@.C285___nv_MAIN__F1L32_1 = internal constant i32 1
@.C285___nv_MAIN_F1L34_2 = internal constant i32 1
@.C283___nv_MAIN_F1L34_2 = internal constant i32 0
@_drb095_2_ = common global %struct_drb095_2_ zeroinitializer, align 64, !dbg !0, !dbg !19
@_drb095_0_ = common global %struct_drb095_0_ zeroinitializer, align 64, !dbg !7, !dbg !13

; Function Attrs: noinline
define float @drb095_() #0 {
.L.entry:
  ret float undef
}

define void @MAIN_() #1 !dbg !23 {
L.entry:
  %__gtid_MAIN__428 = alloca i32, align 4
  %len_323 = alloca i32, align 4
  %.g0000_404 = alloca i64, align 8
  %.g0001_406 = alloca i64, align 8
  %.uplevelArgPack0001_414 = alloca %astruct.dt68, align 16
  %j_321 = alloca i32, align 4
  %z__io_345 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !28, metadata !DIExpression()), !dbg !29
  call void @llvm.dbg.value(metadata i32 0, metadata !30, metadata !DIExpression()), !dbg !29
  call void @llvm.dbg.value(metadata i32 1, metadata !31, metadata !DIExpression()), !dbg !29
  call void @llvm.dbg.value(metadata i32 0, metadata !32, metadata !DIExpression()), !dbg !29
  call void @llvm.dbg.value(metadata i32 1, metadata !33, metadata !DIExpression()), !dbg !29
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !34
  store i32 %0, i32* %__gtid_MAIN__428, align 4, !dbg !34
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !35
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !35
  call void (i8*, ...) %2(i8* %1), !dbg !35
  br label %L.LB2_371

L.LB2_371:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %len_323, metadata !36, metadata !DIExpression()), !dbg !29
  store i32 100, i32* %len_323, align 4, !dbg !37
  %3 = bitcast %struct_drb095_0_* @_drb095_0_ to i8*, !dbg !38
  %4 = getelementptr i8, i8* %3, i64 96, !dbg !38
  %5 = bitcast i8* %4 to i64*, !dbg !38
  store i64 1, i64* %5, align 8, !dbg !38
  %6 = load i32, i32* %len_323, align 4, !dbg !38
  call void @llvm.dbg.value(metadata i32 %6, metadata !36, metadata !DIExpression()), !dbg !29
  %7 = sext i32 %6 to i64, !dbg !38
  %8 = bitcast %struct_drb095_0_* @_drb095_0_ to i8*, !dbg !38
  %9 = getelementptr i8, i8* %8, i64 104, !dbg !38
  %10 = bitcast i8* %9 to i64*, !dbg !38
  store i64 %7, i64* %10, align 8, !dbg !38
  %11 = bitcast %struct_drb095_0_* @_drb095_0_ to i8*, !dbg !38
  %12 = getelementptr i8, i8* %11, i64 144, !dbg !38
  %13 = bitcast i8* %12 to i64*, !dbg !38
  store i64 1, i64* %13, align 8, !dbg !38
  %14 = load i32, i32* %len_323, align 4, !dbg !38
  call void @llvm.dbg.value(metadata i32 %14, metadata !36, metadata !DIExpression()), !dbg !29
  %15 = sext i32 %14 to i64, !dbg !38
  %16 = bitcast %struct_drb095_0_* @_drb095_0_ to i8*, !dbg !38
  %17 = getelementptr i8, i8* %16, i64 152, !dbg !38
  %18 = bitcast i8* %17 to i64*, !dbg !38
  store i64 %15, i64* %18, align 8, !dbg !38
  %19 = bitcast %struct_drb095_0_* @_drb095_0_ to i8*, !dbg !38
  %20 = getelementptr i8, i8* %19, i64 104, !dbg !38
  %21 = bitcast i8* %20 to i64*, !dbg !38
  %22 = load i64, i64* %21, align 8, !dbg !38
  %23 = sub nsw i64 %22, 1, !dbg !38
  %24 = bitcast %struct_drb095_0_* @_drb095_0_ to i8*, !dbg !38
  %25 = getelementptr i8, i8* %24, i64 96, !dbg !38
  %26 = bitcast i8* %25 to i64*, !dbg !38
  %27 = load i64, i64* %26, align 8, !dbg !38
  %28 = add nsw i64 %23, %27, !dbg !38
  store i64 %28, i64* %.g0000_404, align 8, !dbg !38
  %29 = bitcast %struct_drb095_0_* @_drb095_0_ to i8*, !dbg !38
  %30 = getelementptr i8, i8* %29, i64 152, !dbg !38
  %31 = bitcast i8* %30 to i64*, !dbg !38
  %32 = load i64, i64* %31, align 8, !dbg !38
  %33 = sub nsw i64 %32, 1, !dbg !38
  %34 = bitcast %struct_drb095_0_* @_drb095_0_ to i8*, !dbg !38
  %35 = getelementptr i8, i8* %34, i64 144, !dbg !38
  %36 = bitcast i8* %35 to i64*, !dbg !38
  %37 = load i64, i64* %36, align 8, !dbg !38
  %38 = add nsw i64 %33, %37, !dbg !38
  store i64 %38, i64* %.g0001_406, align 8, !dbg !38
  %39 = bitcast %struct_drb095_0_* @_drb095_0_ to i8*, !dbg !38
  %40 = getelementptr i8, i8* %39, i64 16, !dbg !38
  %41 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !38
  %42 = bitcast i64* @.C353_MAIN_ to i8*, !dbg !38
  %43 = bitcast i64* @.C354_MAIN_ to i8*, !dbg !38
  %44 = bitcast %struct_drb095_0_* @_drb095_0_ to i8*, !dbg !38
  %45 = getelementptr i8, i8* %44, i64 96, !dbg !38
  %46 = bitcast i64* %.g0000_404 to i8*, !dbg !38
  %47 = bitcast %struct_drb095_0_* @_drb095_0_ to i8*, !dbg !38
  %48 = getelementptr i8, i8* %47, i64 144, !dbg !38
  %49 = bitcast i64* %.g0001_406 to i8*, !dbg !38
  %50 = bitcast void (...)* @f90_template2_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !38
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, ...) %50(i8* %40, i8* %41, i8* %42, i8* %43, i8* %45, i8* %46, i8* %48, i8* %49), !dbg !38
  %51 = bitcast %struct_drb095_0_* @_drb095_0_ to i8*, !dbg !38
  %52 = getelementptr i8, i8* %51, i64 16, !dbg !38
  %53 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !38
  call void (i8*, i32, ...) %53(i8* %52, i32 25), !dbg !38
  %54 = bitcast %struct_drb095_0_* @_drb095_0_ to i8*, !dbg !38
  %55 = getelementptr i8, i8* %54, i64 104, !dbg !38
  %56 = bitcast i8* %55 to i64*, !dbg !38
  %57 = load i64, i64* %56, align 8, !dbg !38
  %58 = sub nsw i64 %57, 1, !dbg !38
  %59 = bitcast %struct_drb095_0_* @_drb095_0_ to i8*, !dbg !38
  %60 = getelementptr i8, i8* %59, i64 96, !dbg !38
  %61 = bitcast i8* %60 to i64*, !dbg !38
  %62 = load i64, i64* %61, align 8, !dbg !38
  %63 = add nsw i64 %58, %62, !dbg !38
  %64 = bitcast %struct_drb095_0_* @_drb095_0_ to i8*, !dbg !38
  %65 = getelementptr i8, i8* %64, i64 96, !dbg !38
  %66 = bitcast i8* %65 to i64*, !dbg !38
  %67 = load i64, i64* %66, align 8, !dbg !38
  %68 = sub nsw i64 %67, 1, !dbg !38
  %69 = sub nsw i64 %63, %68, !dbg !38
  %70 = bitcast %struct_drb095_0_* @_drb095_0_ to i8*, !dbg !38
  %71 = getelementptr i8, i8* %70, i64 152, !dbg !38
  %72 = bitcast i8* %71 to i64*, !dbg !38
  %73 = load i64, i64* %72, align 8, !dbg !38
  %74 = sub nsw i64 %73, 1, !dbg !38
  %75 = bitcast %struct_drb095_0_* @_drb095_0_ to i8*, !dbg !38
  %76 = getelementptr i8, i8* %75, i64 144, !dbg !38
  %77 = bitcast i8* %76 to i64*, !dbg !38
  %78 = load i64, i64* %77, align 8, !dbg !38
  %79 = add nsw i64 %74, %78, !dbg !38
  %80 = bitcast %struct_drb095_0_* @_drb095_0_ to i8*, !dbg !38
  %81 = getelementptr i8, i8* %80, i64 144, !dbg !38
  %82 = bitcast i8* %81 to i64*, !dbg !38
  %83 = load i64, i64* %82, align 8, !dbg !38
  %84 = sub nsw i64 %83, 1, !dbg !38
  %85 = sub nsw i64 %79, %84, !dbg !38
  %86 = mul nsw i64 %69, %85, !dbg !38
  store i64 %86, i64* %.g0000_404, align 8, !dbg !38
  %87 = bitcast i64* %.g0000_404 to i8*, !dbg !38
  %88 = bitcast i64* @.C353_MAIN_ to i8*, !dbg !38
  %89 = bitcast i64* @.C354_MAIN_ to i8*, !dbg !38
  %90 = bitcast %struct_drb095_0_* @_drb095_0_ to i8*, !dbg !38
  %91 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !38
  %92 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !38
  %93 = bitcast void (...)* @f90_alloc04_chka_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !38
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %93(i8* %87, i8* %88, i8* %89, i8* null, i8* %90, i8* null, i8* %91, i8* %92, i8* null, i64 0), !dbg !38
  %94 = bitcast i32* %len_323 to i8*, !dbg !39
  %95 = bitcast %astruct.dt68* %.uplevelArgPack0001_414 to i8**, !dbg !39
  store i8* %94, i8** %95, align 8, !dbg !39
  call void @llvm.dbg.declare(metadata i32* %j_321, metadata !40, metadata !DIExpression()), !dbg !29
  %96 = bitcast i32* %j_321 to i8*, !dbg !39
  %97 = bitcast %astruct.dt68* %.uplevelArgPack0001_414 to i8*, !dbg !39
  %98 = getelementptr i8, i8* %97, i64 8, !dbg !39
  %99 = bitcast i8* %98 to i8**, !dbg !39
  store i8* %96, i8** %99, align 8, !dbg !39
  %100 = bitcast %struct_drb095_0_* @_drb095_0_ to i8*, !dbg !39
  %101 = bitcast %astruct.dt68* %.uplevelArgPack0001_414 to i8*, !dbg !39
  %102 = getelementptr i8, i8* %101, i64 16, !dbg !39
  %103 = bitcast i8* %102 to i8**, !dbg !39
  store i8* %100, i8** %103, align 8, !dbg !39
  %104 = bitcast %struct_drb095_0_* @_drb095_0_ to i8*, !dbg !39
  %105 = getelementptr i8, i8* %104, i64 16, !dbg !39
  %106 = bitcast %astruct.dt68* %.uplevelArgPack0001_414 to i8*, !dbg !39
  %107 = getelementptr i8, i8* %106, i64 24, !dbg !39
  %108 = bitcast i8* %107 to i8**, !dbg !39
  store i8* %105, i8** %108, align 8, !dbg !39
  %109 = bitcast %struct_drb095_0_* @_drb095_0_ to i8*, !dbg !39
  %110 = bitcast %astruct.dt68* %.uplevelArgPack0001_414 to i8*, !dbg !39
  %111 = getelementptr i8, i8* %110, i64 32, !dbg !39
  %112 = bitcast i8* %111 to i8**, !dbg !39
  store i8* %109, i8** %112, align 8, !dbg !39
  br label %L.LB2_426, !dbg !39

L.LB2_426:                                        ; preds = %L.LB2_371
  %113 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L32_1_ to i64*, !dbg !39
  %114 = bitcast %astruct.dt68* %.uplevelArgPack0001_414 to i64*, !dbg !39
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %113, i64* %114), !dbg !39
  call void (...) @_mp_bcs_nest(), !dbg !41
  %115 = bitcast i32* @.C341_MAIN_ to i8*, !dbg !41
  %116 = bitcast [60 x i8]* @.C339_MAIN_ to i8*, !dbg !41
  %117 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !41
  call void (i8*, i8*, i64, ...) %117(i8* %115, i8* %116, i64 60), !dbg !41
  %118 = bitcast i32* @.C343_MAIN_ to i8*, !dbg !41
  %119 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !41
  %120 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !41
  %121 = bitcast %struct.STATICS2* @.STATICS2 to i8*, !dbg !41
  %122 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !41
  %123 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %122(i8* %118, i8* null, i8* %119, i8* %120, i8* %121, i8* null, i64 0), !dbg !41
  call void @llvm.dbg.declare(metadata i32* %z__io_345, metadata !42, metadata !DIExpression()), !dbg !29
  store i32 %123, i32* %z__io_345, align 4, !dbg !41
  %124 = bitcast %struct_drb095_0_* @_drb095_0_ to i8*, !dbg !41
  %125 = getelementptr i8, i8* %124, i64 176, !dbg !41
  %126 = bitcast i8* %125 to i64*, !dbg !41
  %127 = load i64, i64* %126, align 8, !dbg !41
  %128 = mul nsw i64 %127, 50, !dbg !41
  %129 = bitcast %struct_drb095_0_* @_drb095_0_ to i8*, !dbg !41
  %130 = getelementptr i8, i8* %129, i64 72, !dbg !41
  %131 = bitcast i8* %130 to i64*, !dbg !41
  %132 = load i64, i64* %131, align 8, !dbg !41
  %133 = add nsw i64 %128, %132, !dbg !41
  %134 = bitcast %struct_drb095_0_* @_drb095_0_ to i8**, !dbg !41
  %135 = load i8*, i8** %134, align 8, !dbg !41
  %136 = getelementptr i8, i8* %135, i64 196, !dbg !41
  %137 = bitcast i8* %136 to i32*, !dbg !41
  %138 = getelementptr i32, i32* %137, i64 %133, !dbg !41
  %139 = load i32, i32* %138, align 4, !dbg !41
  %140 = bitcast i32 (...)* @f90io_sc_i_fmt_write to i32 (i32, i32, ...)*, !dbg !41
  %141 = call i32 (i32, i32, ...) %140(i32 %139, i32 25), !dbg !41
  store i32 %141, i32* %z__io_345, align 4, !dbg !41
  %142 = call i32 (...) @f90io_fmtw_end(), !dbg !41
  store i32 %142, i32* %z__io_345, align 4, !dbg !41
  call void (...) @_mp_ecs_nest(), !dbg !41
  ret void, !dbg !34
}

define internal void @__nv_MAIN__F1L32_1_(i32* %__nv_MAIN__F1L32_1Arg0, i64* %__nv_MAIN__F1L32_1Arg1, i64* %__nv_MAIN__F1L32_1Arg2) #1 !dbg !43 {
L.entry:
  %__gtid___nv_MAIN__F1L32_1__474 = alloca i32, align 4
  %.s0000_469 = alloca i32, align 4
  %.s0001_470 = alloca i32, align 4
  %.s0002_480 = alloca i32, align 4
  %.z0410_479 = alloca i8*, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L32_1Arg0, metadata !46, metadata !DIExpression()), !dbg !47
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L32_1Arg1, metadata !48, metadata !DIExpression()), !dbg !47
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L32_1Arg2, metadata !49, metadata !DIExpression()), !dbg !47
  call void @llvm.dbg.value(metadata i32 1, metadata !50, metadata !DIExpression()), !dbg !47
  call void @llvm.dbg.value(metadata i32 0, metadata !51, metadata !DIExpression()), !dbg !47
  call void @llvm.dbg.value(metadata i32 1, metadata !52, metadata !DIExpression()), !dbg !47
  call void @llvm.dbg.value(metadata i32 0, metadata !53, metadata !DIExpression()), !dbg !47
  call void @llvm.dbg.value(metadata i32 1, metadata !54, metadata !DIExpression()), !dbg !47
  %0 = load i32, i32* %__nv_MAIN__F1L32_1Arg0, align 4, !dbg !55
  store i32 %0, i32* %__gtid___nv_MAIN__F1L32_1__474, align 4, !dbg !55
  br label %L.LB3_468

L.LB3_468:                                        ; preds = %L.entry
  br label %L.LB3_326

L.LB3_326:                                        ; preds = %L.LB3_468
  store i32 -1, i32* %.s0000_469, align 4, !dbg !56
  store i32 0, i32* %.s0001_470, align 4, !dbg !56
  %1 = load i32, i32* %__gtid___nv_MAIN__F1L32_1__474, align 4, !dbg !56
  %2 = call i32 @__kmpc_single(i64* null, i32 %1), !dbg !56
  %3 = icmp eq i32 %2, 0, !dbg !56
  br i1 %3, label %L.LB3_361, label %L.LB3_328, !dbg !56

L.LB3_328:                                        ; preds = %L.LB3_326
  store i32 1, i32* %.s0002_480, align 4, !dbg !57
  %4 = load i32, i32* %__gtid___nv_MAIN__F1L32_1__474, align 4, !dbg !58
  %5 = load i32, i32* %.s0002_480, align 4, !dbg !58
  %6 = bitcast void (i32, i64*)* @__nv_MAIN_F1L34_2_ to i64*, !dbg !58
  %7 = call i8* @__kmpc_omp_task_alloc(i64* null, i32 %4, i32 %5, i32 72, i32 40, i64* %6), !dbg !58
  store i8* %7, i8** %.z0410_479, align 8, !dbg !58
  %8 = load i64, i64* %__nv_MAIN__F1L32_1Arg2, align 8, !dbg !58
  %9 = load i8*, i8** %.z0410_479, align 8, !dbg !58
  %10 = bitcast i8* %9 to i64**, !dbg !58
  %11 = load i64*, i64** %10, align 8, !dbg !58
  store i64 %8, i64* %11, align 8, !dbg !58
  %12 = bitcast i64* %__nv_MAIN__F1L32_1Arg2 to i8*, !dbg !55
  %13 = getelementptr i8, i8* %12, i64 8, !dbg !55
  %14 = bitcast i8* %13 to i64*, !dbg !55
  %15 = load i64, i64* %14, align 8, !dbg !55
  %16 = bitcast i64* %11 to i8*, !dbg !55
  %17 = getelementptr i8, i8* %16, i64 8, !dbg !55
  %18 = bitcast i8* %17 to i64*, !dbg !55
  store i64 %15, i64* %18, align 8, !dbg !55
  %19 = bitcast i64* %__nv_MAIN__F1L32_1Arg2 to i8*, !dbg !55
  %20 = getelementptr i8, i8* %19, i64 16, !dbg !55
  %21 = bitcast i8* %20 to i64*, !dbg !55
  %22 = load i64, i64* %21, align 8, !dbg !55
  %23 = bitcast i64* %11 to i8*, !dbg !55
  %24 = getelementptr i8, i8* %23, i64 16, !dbg !55
  %25 = bitcast i8* %24 to i64*, !dbg !55
  store i64 %22, i64* %25, align 8, !dbg !55
  %26 = bitcast i64* %__nv_MAIN__F1L32_1Arg2 to i8*, !dbg !55
  %27 = getelementptr i8, i8* %26, i64 24, !dbg !55
  %28 = bitcast i8* %27 to i64*, !dbg !55
  %29 = load i64, i64* %28, align 8, !dbg !55
  %30 = bitcast i64* %11 to i8*, !dbg !55
  %31 = getelementptr i8, i8* %30, i64 24, !dbg !55
  %32 = bitcast i8* %31 to i64*, !dbg !55
  store i64 %29, i64* %32, align 8, !dbg !55
  %33 = bitcast i64* %__nv_MAIN__F1L32_1Arg2 to i8*, !dbg !55
  %34 = getelementptr i8, i8* %33, i64 32, !dbg !55
  %35 = bitcast i8* %34 to i64*, !dbg !55
  %36 = load i64, i64* %35, align 8, !dbg !55
  %37 = bitcast i64* %11 to i8*, !dbg !55
  %38 = getelementptr i8, i8* %37, i64 32, !dbg !55
  %39 = bitcast i8* %38 to i64*, !dbg !55
  store i64 %36, i64* %39, align 8, !dbg !55
  %40 = load i8*, i8** %.z0410_479, align 8, !dbg !58
  %41 = getelementptr i8, i8* %40, i64 40, !dbg !58
  %42 = bitcast i8* %41 to i64*, !dbg !58
  store i64 1, i64* %42, align 8, !dbg !58
  %43 = bitcast i64* %__nv_MAIN__F1L32_1Arg2 to i32**, !dbg !58
  %44 = load i32*, i32** %43, align 8, !dbg !58
  %45 = load i32, i32* %44, align 4, !dbg !58
  %46 = sext i32 %45 to i64, !dbg !58
  %47 = load i8*, i8** %.z0410_479, align 8, !dbg !58
  %48 = getelementptr i8, i8* %47, i64 48, !dbg !58
  %49 = bitcast i8* %48 to i64*, !dbg !58
  store i64 %46, i64* %49, align 8, !dbg !58
  %50 = load i8*, i8** %.z0410_479, align 8, !dbg !58
  %51 = getelementptr i8, i8* %50, i64 56, !dbg !58
  %52 = bitcast i8* %51 to i64*, !dbg !58
  store i64 1, i64* %52, align 8, !dbg !58
  %53 = load i32, i32* %__gtid___nv_MAIN__F1L32_1__474, align 4, !dbg !58
  %54 = load i8*, i8** %.z0410_479, align 8, !dbg !58
  %55 = bitcast i8* %54 to i64*, !dbg !58
  %56 = load i8*, i8** %.z0410_479, align 8, !dbg !58
  %57 = getelementptr i8, i8* %56, i64 40, !dbg !58
  %58 = bitcast i8* %57 to i64*, !dbg !58
  %59 = load i8*, i8** %.z0410_479, align 8, !dbg !58
  %60 = getelementptr i8, i8* %59, i64 48, !dbg !58
  %61 = bitcast i8* %60 to i64*, !dbg !58
  %62 = sext i32 0 to i64, !dbg !58
  call void @__kmpc_taskloop(i64* null, i32 %53, i64* %55, i32 1, i64* %58, i64* %61, i64 1, i32 0, i32 0, i64 %62, i64* null), !dbg !58
  br label %L.LB3_362

L.LB3_362:                                        ; preds = %L.LB3_328
  %63 = load i32, i32* %__gtid___nv_MAIN__F1L32_1__474, align 4, !dbg !59
  store i32 %63, i32* %.s0000_469, align 4, !dbg !59
  store i32 1, i32* %.s0001_470, align 4, !dbg !59
  %64 = load i32, i32* %__gtid___nv_MAIN__F1L32_1__474, align 4, !dbg !59
  call void @__kmpc_end_single(i64* null, i32 %64), !dbg !59
  br label %L.LB3_361

L.LB3_361:                                        ; preds = %L.LB3_362, %L.LB3_326
  br label %L.LB3_336

L.LB3_336:                                        ; preds = %L.LB3_361
  %65 = load i32, i32* %__gtid___nv_MAIN__F1L32_1__474, align 4, !dbg !59
  call void @__kmpc_barrier(i64* null, i32 %65), !dbg !59
  br label %L.LB3_337

L.LB3_337:                                        ; preds = %L.LB3_336
  ret void, !dbg !55
}

define internal void @__nv_MAIN_F1L34_2_(i32 %__nv_MAIN_F1L34_2Arg0.arg, i64* %__nv_MAIN_F1L34_2Arg1) #1 !dbg !60 {
L.entry:
  %__nv_MAIN_F1L34_2Arg0.addr = alloca i32, align 4
  %.S0000_529 = alloca i8*, align 8
  %.i0000p_333 = alloca i32, align 4
  %i_332 = alloca i32, align 4
  %.dU0001p_366 = alloca i32, align 4
  %.dY0001p_365 = alloca i32, align 4
  %.dY0002p_369 = alloca i32, align 4
  %j_334 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L34_2Arg0.addr, metadata !63, metadata !DIExpression()), !dbg !64
  store i32 %__nv_MAIN_F1L34_2Arg0.arg, i32* %__nv_MAIN_F1L34_2Arg0.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN_F1L34_2Arg0.addr, metadata !65, metadata !DIExpression()), !dbg !64
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN_F1L34_2Arg1, metadata !66, metadata !DIExpression()), !dbg !64
  call void @llvm.dbg.value(metadata i32 1, metadata !67, metadata !DIExpression()), !dbg !64
  call void @llvm.dbg.value(metadata i32 0, metadata !68, metadata !DIExpression()), !dbg !64
  call void @llvm.dbg.value(metadata i32 1, metadata !69, metadata !DIExpression()), !dbg !64
  call void @llvm.dbg.value(metadata i32 0, metadata !70, metadata !DIExpression()), !dbg !64
  call void @llvm.dbg.value(metadata i32 1, metadata !71, metadata !DIExpression()), !dbg !64
  %0 = bitcast i64* %__nv_MAIN_F1L34_2Arg1 to i8**, !dbg !72
  %1 = load i8*, i8** %0, align 8, !dbg !72
  store i8* %1, i8** %.S0000_529, align 8, !dbg !72
  br label %L.LB5_533

L.LB5_533:                                        ; preds = %L.entry
  br label %L.LB5_331

L.LB5_331:                                        ; preds = %L.LB5_533
  store i32 0, i32* %.i0000p_333, align 4, !dbg !73
  %2 = bitcast i64* %__nv_MAIN_F1L34_2Arg1 to i8*, !dbg !73
  %3 = getelementptr i8, i8* %2, i64 40, !dbg !73
  %4 = bitcast i8* %3 to i64*, !dbg !73
  %5 = load i64, i64* %4, align 8, !dbg !73
  %6 = trunc i64 %5 to i32, !dbg !73
  call void @llvm.dbg.declare(metadata i32* %i_332, metadata !74, metadata !DIExpression()), !dbg !64
  store i32 %6, i32* %i_332, align 4, !dbg !73
  %7 = bitcast i64* %__nv_MAIN_F1L34_2Arg1 to i8*, !dbg !73
  %8 = getelementptr i8, i8* %7, i64 48, !dbg !73
  %9 = bitcast i8* %8 to i64*, !dbg !73
  %10 = load i64, i64* %9, align 8, !dbg !73
  %11 = trunc i64 %10 to i32, !dbg !73
  store i32 %11, i32* %.dU0001p_366, align 4, !dbg !73
  %12 = bitcast i64* %__nv_MAIN_F1L34_2Arg1 to i8*, !dbg !73
  %13 = getelementptr i8, i8* %12, i64 64, !dbg !73
  %14 = bitcast i8* %13 to i32*, !dbg !73
  %15 = load i32, i32* %14, align 4, !dbg !73
  store i32 %15, i32* %.i0000p_333, align 4, !dbg !73
  %16 = load i32, i32* %.dU0001p_366, align 4, !dbg !73
  %17 = load i32, i32* %i_332, align 4, !dbg !73
  call void @llvm.dbg.value(metadata i32 %17, metadata !74, metadata !DIExpression()), !dbg !64
  %18 = sub nsw i32 %16, %17, !dbg !73
  %19 = add nsw i32 %18, 1, !dbg !73
  store i32 %19, i32* %.dY0001p_365, align 4, !dbg !73
  %20 = load i32, i32* %.dY0001p_365, align 4, !dbg !73
  %21 = icmp sle i32 %20, 0, !dbg !73
  br i1 %21, label %L.LB5_364, label %L.LB5_363, !dbg !73

L.LB5_363:                                        ; preds = %L.LB5_368, %L.LB5_331
  %22 = load i8*, i8** %.S0000_529, align 8, !dbg !75
  %23 = bitcast i8* %22 to i32**, !dbg !75
  %24 = load i32*, i32** %23, align 8, !dbg !75
  %25 = load i32, i32* %24, align 4, !dbg !75
  store i32 %25, i32* %.dY0002p_369, align 4, !dbg !75
  call void @llvm.dbg.declare(metadata i32* %j_334, metadata !76, metadata !DIExpression()), !dbg !77
  store i32 1, i32* %j_334, align 4, !dbg !75
  %26 = load i32, i32* %.dY0002p_369, align 4, !dbg !75
  %27 = icmp sle i32 %26, 0, !dbg !75
  br i1 %27, label %L.LB5_368, label %L.LB5_367, !dbg !75

L.LB5_367:                                        ; preds = %L.LB5_367, %L.LB5_363
  %28 = load i32, i32* %i_332, align 4, !dbg !78
  call void @llvm.dbg.value(metadata i32 %28, metadata !74, metadata !DIExpression()), !dbg !64
  %29 = sext i32 %28 to i64, !dbg !78
  %30 = load i32, i32* %j_334, align 4, !dbg !78
  call void @llvm.dbg.value(metadata i32 %30, metadata !76, metadata !DIExpression()), !dbg !77
  %31 = sext i32 %30 to i64, !dbg !78
  %32 = load i8*, i8** %.S0000_529, align 8, !dbg !78
  %33 = getelementptr i8, i8* %32, i64 24, !dbg !78
  %34 = bitcast i8* %33 to i8**, !dbg !78
  %35 = load i8*, i8** %34, align 8, !dbg !78
  %36 = getelementptr i8, i8* %35, i64 160, !dbg !78
  %37 = bitcast i8* %36 to i64*, !dbg !78
  %38 = load i64, i64* %37, align 8, !dbg !78
  %39 = mul nsw i64 %31, %38, !dbg !78
  %40 = add nsw i64 %29, %39, !dbg !78
  %41 = load i8*, i8** %.S0000_529, align 8, !dbg !78
  %42 = getelementptr i8, i8* %41, i64 24, !dbg !78
  %43 = bitcast i8* %42 to i8**, !dbg !78
  %44 = load i8*, i8** %43, align 8, !dbg !78
  %45 = getelementptr i8, i8* %44, i64 56, !dbg !78
  %46 = bitcast i8* %45 to i64*, !dbg !78
  %47 = load i64, i64* %46, align 8, !dbg !78
  %48 = add nsw i64 %40, %47, !dbg !78
  %49 = load i8*, i8** %.S0000_529, align 8, !dbg !78
  %50 = getelementptr i8, i8* %49, i64 32, !dbg !78
  %51 = bitcast i8* %50 to i8***, !dbg !78
  %52 = load i8**, i8*** %51, align 8, !dbg !78
  %53 = load i8*, i8** %52, align 8, !dbg !78
  %54 = getelementptr i8, i8* %53, i64 -4, !dbg !78
  %55 = bitcast i8* %54 to i32*, !dbg !78
  %56 = getelementptr i32, i32* %55, i64 %48, !dbg !78
  %57 = load i32, i32* %56, align 4, !dbg !78
  %58 = add nsw i32 %57, 1, !dbg !78
  %59 = load i32, i32* %i_332, align 4, !dbg !78
  call void @llvm.dbg.value(metadata i32 %59, metadata !74, metadata !DIExpression()), !dbg !64
  %60 = sext i32 %59 to i64, !dbg !78
  %61 = load i32, i32* %j_334, align 4, !dbg !78
  call void @llvm.dbg.value(metadata i32 %61, metadata !76, metadata !DIExpression()), !dbg !77
  %62 = sext i32 %61 to i64, !dbg !78
  %63 = load i8*, i8** %.S0000_529, align 8, !dbg !78
  %64 = getelementptr i8, i8* %63, i64 24, !dbg !78
  %65 = bitcast i8* %64 to i8**, !dbg !78
  %66 = load i8*, i8** %65, align 8, !dbg !78
  %67 = getelementptr i8, i8* %66, i64 160, !dbg !78
  %68 = bitcast i8* %67 to i64*, !dbg !78
  %69 = load i64, i64* %68, align 8, !dbg !78
  %70 = mul nsw i64 %62, %69, !dbg !78
  %71 = add nsw i64 %60, %70, !dbg !78
  %72 = load i8*, i8** %.S0000_529, align 8, !dbg !78
  %73 = getelementptr i8, i8* %72, i64 24, !dbg !78
  %74 = bitcast i8* %73 to i8**, !dbg !78
  %75 = load i8*, i8** %74, align 8, !dbg !78
  %76 = getelementptr i8, i8* %75, i64 56, !dbg !78
  %77 = bitcast i8* %76 to i64*, !dbg !78
  %78 = load i64, i64* %77, align 8, !dbg !78
  %79 = add nsw i64 %71, %78, !dbg !78
  %80 = load i8*, i8** %.S0000_529, align 8, !dbg !78
  %81 = getelementptr i8, i8* %80, i64 32, !dbg !78
  %82 = bitcast i8* %81 to i8***, !dbg !78
  %83 = load i8**, i8*** %82, align 8, !dbg !78
  %84 = load i8*, i8** %83, align 8, !dbg !78
  %85 = getelementptr i8, i8* %84, i64 -4, !dbg !78
  %86 = bitcast i8* %85 to i32*, !dbg !78
  %87 = getelementptr i32, i32* %86, i64 %79, !dbg !78
  store i32 %58, i32* %87, align 4, !dbg !78
  %88 = load i32, i32* %j_334, align 4, !dbg !79
  call void @llvm.dbg.value(metadata i32 %88, metadata !76, metadata !DIExpression()), !dbg !77
  %89 = add nsw i32 %88, 1, !dbg !79
  store i32 %89, i32* %j_334, align 4, !dbg !79
  %90 = load i32, i32* %.dY0002p_369, align 4, !dbg !79
  %91 = sub nsw i32 %90, 1, !dbg !79
  store i32 %91, i32* %.dY0002p_369, align 4, !dbg !79
  %92 = load i32, i32* %.dY0002p_369, align 4, !dbg !79
  %93 = icmp sgt i32 %92, 0, !dbg !79
  br i1 %93, label %L.LB5_367, label %L.LB5_368, !dbg !79

L.LB5_368:                                        ; preds = %L.LB5_367, %L.LB5_363
  %94 = load i32, i32* %i_332, align 4, !dbg !77
  call void @llvm.dbg.value(metadata i32 %94, metadata !74, metadata !DIExpression()), !dbg !64
  %95 = add nsw i32 %94, 1, !dbg !77
  store i32 %95, i32* %i_332, align 4, !dbg !77
  %96 = load i32, i32* %.dY0001p_365, align 4, !dbg !77
  %97 = sub nsw i32 %96, 1, !dbg !77
  store i32 %97, i32* %.dY0001p_365, align 4, !dbg !77
  %98 = load i32, i32* %.dY0001p_365, align 4, !dbg !77
  %99 = icmp sgt i32 %98, 0, !dbg !77
  br i1 %99, label %L.LB5_363, label %L.LB5_364, !dbg !77

L.LB5_364:                                        ; preds = %L.LB5_368, %L.LB5_331
  br label %L.LB5_335

L.LB5_335:                                        ; preds = %L.LB5_364
  ret void, !dbg !77
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
!2 = !DIModule(scope: !3, name: "drb095")
!3 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !4, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !6, imports: !21)
!4 = !DIFile(filename: "micro-benchmarks-fortran/DRB095-doall2-taskloop-orig-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
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
!22 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !23, entity: !2, file: !4, line: 23)
!23 = distinct !DISubprogram(name: "drb095_doall2_taskloop_orig_yes", scope: !3, file: !4, line: 23, type: !24, scopeLine: 23, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !3)
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
!34 = !DILocation(line: 47, column: 1, scope: !23)
!35 = !DILocation(line: 23, column: 1, scope: !23)
!36 = !DILocalVariable(name: "len", scope: !23, file: !4, type: !10)
!37 = !DILocation(line: 29, column: 1, scope: !23)
!38 = !DILocation(line: 30, column: 1, scope: !23)
!39 = !DILocation(line: 32, column: 1, scope: !23)
!40 = !DILocalVariable(name: "j", scope: !23, file: !4, type: !10)
!41 = !DILocation(line: 44, column: 1, scope: !23)
!42 = !DILocalVariable(scope: !23, file: !4, type: !10, flags: DIFlagArtificial)
!43 = distinct !DISubprogram(name: "__nv_MAIN__F1L32_1", scope: !3, file: !4, line: 32, type: !44, scopeLine: 32, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !3)
!44 = !DISubroutineType(types: !45)
!45 = !{null, !10, !16, !16}
!46 = !DILocalVariable(name: "__nv_MAIN__F1L32_1Arg0", arg: 1, scope: !43, file: !4, type: !10)
!47 = !DILocation(line: 0, scope: !43)
!48 = !DILocalVariable(name: "__nv_MAIN__F1L32_1Arg1", arg: 2, scope: !43, file: !4, type: !16)
!49 = !DILocalVariable(name: "__nv_MAIN__F1L32_1Arg2", arg: 3, scope: !43, file: !4, type: !16)
!50 = !DILocalVariable(name: "omp_sched_static", scope: !43, file: !4, type: !10)
!51 = !DILocalVariable(name: "omp_proc_bind_false", scope: !43, file: !4, type: !10)
!52 = !DILocalVariable(name: "omp_proc_bind_true", scope: !43, file: !4, type: !10)
!53 = !DILocalVariable(name: "omp_lock_hint_none", scope: !43, file: !4, type: !10)
!54 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !43, file: !4, type: !10)
!55 = !DILocation(line: 42, column: 1, scope: !43)
!56 = !DILocation(line: 33, column: 1, scope: !43)
!57 = !DILocation(line: 34, column: 1, scope: !43)
!58 = !DILocation(line: 39, column: 1, scope: !43)
!59 = !DILocation(line: 41, column: 1, scope: !43)
!60 = distinct !DISubprogram(name: "__nv_MAIN_F1L34_2", scope: !3, file: !4, line: 34, type: !61, scopeLine: 34, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !3)
!61 = !DISubroutineType(types: !62)
!62 = !{null, !10, !16}
!63 = !DILocalVariable(name: "__nv_MAIN_F1L34_2Arg0", scope: !60, file: !4, type: !10)
!64 = !DILocation(line: 0, scope: !60)
!65 = !DILocalVariable(name: "__nv_MAIN_F1L34_2Arg0", arg: 1, scope: !60, file: !4, type: !10)
!66 = !DILocalVariable(name: "__nv_MAIN_F1L34_2Arg1", arg: 2, scope: !60, file: !4, type: !16)
!67 = !DILocalVariable(name: "omp_sched_static", scope: !60, file: !4, type: !10)
!68 = !DILocalVariable(name: "omp_proc_bind_false", scope: !60, file: !4, type: !10)
!69 = !DILocalVariable(name: "omp_proc_bind_true", scope: !60, file: !4, type: !10)
!70 = !DILocalVariable(name: "omp_lock_hint_none", scope: !60, file: !4, type: !10)
!71 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !60, file: !4, type: !10)
!72 = !DILocation(line: 34, column: 1, scope: !60)
!73 = !DILocation(line: 35, column: 1, scope: !60)
!74 = !DILocalVariable(name: "i", scope: !60, file: !4, type: !10)
!75 = !DILocation(line: 36, column: 1, scope: !60)
!76 = !DILocalVariable(name: "j", scope: !60, file: !4, type: !10)
!77 = !DILocation(line: 39, column: 1, scope: !60)
!78 = !DILocation(line: 37, column: 1, scope: !60)
!79 = !DILocation(line: 38, column: 1, scope: !60)

; ModuleID = '/tmp/DRB067-restrictpointer1-orig-no-07f666.ll'
source_filename = "/tmp/DRB067-restrictpointer1-orig-no-07f666.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.STATICS5 = type <{ [4 x i8] }>
%astruct.dt90 = type <{ i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8* }>

@.C339_drb067_foo_ = internal constant i32 6
@.C336_drb067_foo_ = internal constant [60 x i8] c"micro-benchmarks-fortran/DRB067-restrictpointer1-orig-no.f95"
@.C338_drb067_foo_ = internal constant i32 34
@.C291_drb067_foo_ = internal constant double 0.000000e+00
@.C285_drb067_foo_ = internal constant i32 1
@.C283_drb067_foo_ = internal constant i32 0
@.C349_drb067_foo_ = internal constant i64 9
@.C308_drb067_foo_ = internal constant i64 12
@.C307_drb067_foo_ = internal constant i64 11
@.C305_drb067_foo_ = internal constant i32 28
@.C286_drb067_foo_ = internal constant i64 1
@.C284_drb067_foo_ = internal constant i64 0
@.C347_drb067_foo_ = internal constant i64 28
@.C309_drb067_foo_ = internal constant i64 8
@.C291___nv_drb067_foo__F1L27_1 = internal constant double 0.000000e+00
@.C285___nv_drb067_foo__F1L27_1 = internal constant i32 1
@.C283___nv_drb067_foo__F1L27_1 = internal constant i32 0
@.STATICS5 = internal global %struct.STATICS5 <{ [4 x i8] c"\E8\03\00\00" }>, align 16, !dbg !0
@.C330_MAIN_ = internal constant i64 9
@.C306_MAIN_ = internal constant i32 28
@.C310_MAIN_ = internal constant i64 8
@.C328_MAIN_ = internal constant i64 28
@.C309_MAIN_ = internal constant i64 12
@.C308_MAIN_ = internal constant i64 11
@.C286_MAIN_ = internal constant i64 1
@.C284_MAIN_ = internal constant i64 0
@.C283_MAIN_ = internal constant i32 0

; Function Attrs: noinline
define float @drb067_() #0 {
.L.entry:
  ret float undef
}

define void @drb067_foo_(i64* %"newsxx$p", i64* %"newsyy$p", i32 %_V_len.arg, i64* %"newsxx$sd1", i64* %"newsyy$sd3") #1 !dbg !12 {
L.entry:
  %_V_len.addr = alloca i32, align 4
  %len_312 = alloca i32, align 4
  %__gtid_drb067_foo__439 = alloca i32, align 4
  %z_e_110_322 = alloca i64, align 8
  %"tar1$p_332" = alloca double*, align 8
  %"tar1$sd6_352" = alloca [16 x i64], align 8
  %"tar2$p_333" = alloca double*, align 8
  %"tar2$sd5_351" = alloca [16 x i64], align 8
  %.g0000_406 = alloca i64, align 8
  %.uplevelArgPack0001_420 = alloca %astruct.dt90, align 16
  %z__io_341 = alloca i32, align 4
  %"drb067_foo___$eq_313" = alloca [16 x i8], align 4
  call void @llvm.dbg.declare(metadata i64* %"newsxx$p", metadata !24, metadata !DIExpression(DW_OP_deref)), !dbg !25
  call void @llvm.dbg.declare(metadata i64* %"newsyy$p", metadata !26, metadata !DIExpression(DW_OP_deref)), !dbg !25
  call void @llvm.dbg.declare(metadata i32* %_V_len.addr, metadata !27, metadata !DIExpression()), !dbg !25
  store i32 %_V_len.arg, i32* %_V_len.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %_V_len.addr, metadata !28, metadata !DIExpression()), !dbg !25
  call void @llvm.dbg.declare(metadata i64* %"newsxx$sd1", metadata !29, metadata !DIExpression()), !dbg !25
  call void @llvm.dbg.declare(metadata i64* %"newsyy$sd3", metadata !30, metadata !DIExpression()), !dbg !25
  call void @llvm.dbg.value(metadata i32 1, metadata !31, metadata !DIExpression()), !dbg !25
  call void @llvm.dbg.value(metadata i32 0, metadata !32, metadata !DIExpression()), !dbg !25
  call void @llvm.dbg.value(metadata i32 1, metadata !33, metadata !DIExpression()), !dbg !25
  call void @llvm.dbg.value(metadata i32 0, metadata !34, metadata !DIExpression()), !dbg !25
  call void @llvm.dbg.value(metadata i32 1, metadata !35, metadata !DIExpression()), !dbg !25
  call void @llvm.dbg.value(metadata i32 8, metadata !36, metadata !DIExpression()), !dbg !25
  %0 = load i32, i32* %_V_len.addr, align 4, !dbg !37
  call void @llvm.dbg.value(metadata i32 %0, metadata !27, metadata !DIExpression()), !dbg !25
  call void @llvm.dbg.declare(metadata i32* %len_312, metadata !38, metadata !DIExpression()), !dbg !25
  store i32 %0, i32* %len_312, align 4, !dbg !37
  %1 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !39
  store i32 %1, i32* %__gtid_drb067_foo__439, align 4, !dbg !39
  %2 = load i32, i32* %len_312, align 4, !dbg !37
  call void @llvm.dbg.value(metadata i32 %2, metadata !38, metadata !DIExpression()), !dbg !25
  %3 = sext i32 %2 to i64, !dbg !37
  call void @llvm.dbg.declare(metadata i64* %z_e_110_322, metadata !40, metadata !DIExpression()), !dbg !25
  store i64 %3, i64* %z_e_110_322, align 8, !dbg !37
  %4 = bitcast i64* %z_e_110_322 to i8*, !dbg !37
  %5 = bitcast i64* @.C309_drb067_foo_ to i8*, !dbg !37
  %6 = bitcast i64 (...)* @f90_auto_alloc04_i8 to i64 (i8*, i8*, ...)*, !dbg !37
  %7 = call i64 (i8*, i8*, ...) %6(i8* %4, i8* %5), !dbg !37
  %8 = inttoptr i64 %7 to i8*, !dbg !37
  %9 = bitcast double** %"tar1$p_332" to i8**, !dbg !37
  store i8* %8, i8** %9, align 8, !dbg !37
  call void @llvm.dbg.declare(metadata [16 x i64]* %"tar1$sd6_352", metadata !41, metadata !DIExpression()), !dbg !25
  %10 = bitcast [16 x i64]* %"tar1$sd6_352" to i8*, !dbg !37
  %11 = bitcast i64* @.C284_drb067_foo_ to i8*, !dbg !37
  %12 = bitcast i64* @.C347_drb067_foo_ to i8*, !dbg !37
  %13 = bitcast i64* @.C309_drb067_foo_ to i8*, !dbg !37
  %14 = bitcast i64* @.C286_drb067_foo_ to i8*, !dbg !37
  %15 = bitcast i64* %z_e_110_322 to i8*, !dbg !37
  %16 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !37
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %16(i8* %10, i8* %11, i8* %12, i8* %13, i8* %14, i8* %15), !dbg !37
  %17 = bitcast [16 x i64]* %"tar1$sd6_352" to i8*, !dbg !37
  %18 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !37
  call void (i8*, i32, ...) %18(i8* %17, i32 28), !dbg !37
  %19 = bitcast i64* %z_e_110_322 to i8*, !dbg !37
  %20 = bitcast i64* @.C309_drb067_foo_ to i8*, !dbg !37
  %21 = bitcast i64 (...)* @f90_auto_alloc04_i8 to i64 (i8*, i8*, ...)*, !dbg !37
  %22 = call i64 (i8*, i8*, ...) %21(i8* %19, i8* %20), !dbg !37
  %23 = inttoptr i64 %22 to i8*, !dbg !37
  %24 = bitcast double** %"tar2$p_333" to i8**, !dbg !37
  store i8* %23, i8** %24, align 8, !dbg !37
  call void @llvm.dbg.declare(metadata [16 x i64]* %"tar2$sd5_351", metadata !41, metadata !DIExpression()), !dbg !25
  %25 = bitcast [16 x i64]* %"tar2$sd5_351" to i8*, !dbg !37
  %26 = bitcast i64* @.C284_drb067_foo_ to i8*, !dbg !37
  %27 = bitcast i64* @.C347_drb067_foo_ to i8*, !dbg !37
  %28 = bitcast i64* @.C309_drb067_foo_ to i8*, !dbg !37
  %29 = bitcast i64* @.C286_drb067_foo_ to i8*, !dbg !37
  %30 = bitcast i64* %z_e_110_322 to i8*, !dbg !37
  %31 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !37
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %31(i8* %25, i8* %26, i8* %27, i8* %28, i8* %29, i8* %30), !dbg !37
  %32 = bitcast [16 x i64]* %"tar2$sd5_351" to i8*, !dbg !37
  %33 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !37
  call void (i8*, i32, ...) %33(i8* %32, i32 28), !dbg !37
  br label %L.LB2_392

L.LB2_392:                                        ; preds = %L.entry
  %34 = bitcast i64* %"newsxx$sd1" to i8*, !dbg !42
  %35 = getelementptr i8, i8* %34, i64 80, !dbg !42
  %36 = bitcast i8* %35 to i64*, !dbg !42
  store i64 1, i64* %36, align 8, !dbg !42
  %37 = load i32, i32* %len_312, align 4, !dbg !42
  call void @llvm.dbg.value(metadata i32 %37, metadata !38, metadata !DIExpression()), !dbg !25
  %38 = sext i32 %37 to i64, !dbg !42
  %39 = bitcast i64* %"newsxx$sd1" to i8*, !dbg !42
  %40 = getelementptr i8, i8* %39, i64 88, !dbg !42
  %41 = bitcast i8* %40 to i64*, !dbg !42
  store i64 %38, i64* %41, align 8, !dbg !42
  %42 = bitcast i64* %"newsxx$sd1" to i8*, !dbg !42
  %43 = getelementptr i8, i8* %42, i64 88, !dbg !42
  %44 = bitcast i8* %43 to i64*, !dbg !42
  %45 = load i64, i64* %44, align 8, !dbg !42
  %46 = sub nsw i64 %45, 1, !dbg !42
  %47 = bitcast i64* %"newsxx$sd1" to i8*, !dbg !42
  %48 = getelementptr i8, i8* %47, i64 80, !dbg !42
  %49 = bitcast i8* %48 to i64*, !dbg !42
  %50 = load i64, i64* %49, align 8, !dbg !42
  %51 = add nsw i64 %46, %50, !dbg !42
  store i64 %51, i64* %.g0000_406, align 8, !dbg !42
  %52 = bitcast i64* %"newsxx$sd1" to i8*, !dbg !42
  %53 = bitcast i64* @.C284_drb067_foo_ to i8*, !dbg !42
  %54 = bitcast i64* @.C347_drb067_foo_ to i8*, !dbg !42
  %55 = bitcast i64* @.C309_drb067_foo_ to i8*, !dbg !42
  %56 = bitcast i64* %"newsxx$sd1" to i8*, !dbg !42
  %57 = getelementptr i8, i8* %56, i64 80, !dbg !42
  %58 = bitcast i64* %.g0000_406 to i8*, !dbg !42
  %59 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !42
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %59(i8* %52, i8* %53, i8* %54, i8* %55, i8* %57, i8* %58), !dbg !42
  %60 = bitcast i64* %"newsxx$sd1" to i8*, !dbg !42
  %61 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !42
  call void (i8*, i32, ...) %61(i8* %60, i32 28), !dbg !42
  %62 = bitcast i64* %"newsxx$sd1" to i8*, !dbg !42
  %63 = getelementptr i8, i8* %62, i64 88, !dbg !42
  %64 = bitcast i8* %63 to i64*, !dbg !42
  %65 = load i64, i64* %64, align 8, !dbg !42
  %66 = sub nsw i64 %65, 1, !dbg !42
  %67 = bitcast i64* %"newsxx$sd1" to i8*, !dbg !42
  %68 = getelementptr i8, i8* %67, i64 80, !dbg !42
  %69 = bitcast i8* %68 to i64*, !dbg !42
  %70 = load i64, i64* %69, align 8, !dbg !42
  %71 = add nsw i64 %66, %70, !dbg !42
  %72 = bitcast i64* %"newsxx$sd1" to i8*, !dbg !42
  %73 = getelementptr i8, i8* %72, i64 80, !dbg !42
  %74 = bitcast i8* %73 to i64*, !dbg !42
  %75 = load i64, i64* %74, align 8, !dbg !42
  %76 = sub nsw i64 %75, 1, !dbg !42
  %77 = sub nsw i64 %71, %76, !dbg !42
  store i64 %77, i64* %.g0000_406, align 8, !dbg !42
  %78 = bitcast i64* %.g0000_406 to i8*, !dbg !42
  %79 = bitcast i64* @.C347_drb067_foo_ to i8*, !dbg !42
  %80 = bitcast i64* @.C309_drb067_foo_ to i8*, !dbg !42
  %81 = bitcast i64* %"newsxx$p" to i8*, !dbg !42
  %82 = bitcast i64* @.C286_drb067_foo_ to i8*, !dbg !42
  %83 = bitcast i64* @.C284_drb067_foo_ to i8*, !dbg !42
  %84 = bitcast void (...)* @f90_alloc04a_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !42
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %84(i8* %78, i8* %79, i8* %80, i8* null, i8* %81, i8* null, i8* %82, i8* %83, i8* null, i64 0), !dbg !42
  %85 = bitcast i64* %"newsxx$sd1" to i8*, !dbg !42
  %86 = getelementptr i8, i8* %85, i64 64, !dbg !42
  %87 = bitcast i64* %"newsxx$p" to i8*, !dbg !42
  %88 = bitcast void (...)* @f90_ptrcp to void (i8*, i8*, ...)*, !dbg !42
  call void (i8*, i8*, ...) %88(i8* %86, i8* %87), !dbg !42
  %89 = bitcast i64* %"newsyy$sd3" to i8*, !dbg !43
  %90 = getelementptr i8, i8* %89, i64 80, !dbg !43
  %91 = bitcast i8* %90 to i64*, !dbg !43
  store i64 1, i64* %91, align 8, !dbg !43
  %92 = load i32, i32* %len_312, align 4, !dbg !43
  call void @llvm.dbg.value(metadata i32 %92, metadata !38, metadata !DIExpression()), !dbg !25
  %93 = sext i32 %92 to i64, !dbg !43
  %94 = bitcast i64* %"newsyy$sd3" to i8*, !dbg !43
  %95 = getelementptr i8, i8* %94, i64 88, !dbg !43
  %96 = bitcast i8* %95 to i64*, !dbg !43
  store i64 %93, i64* %96, align 8, !dbg !43
  %97 = bitcast i64* %"newsyy$sd3" to i8*, !dbg !43
  %98 = getelementptr i8, i8* %97, i64 88, !dbg !43
  %99 = bitcast i8* %98 to i64*, !dbg !43
  %100 = load i64, i64* %99, align 8, !dbg !43
  %101 = sub nsw i64 %100, 1, !dbg !43
  %102 = bitcast i64* %"newsyy$sd3" to i8*, !dbg !43
  %103 = getelementptr i8, i8* %102, i64 80, !dbg !43
  %104 = bitcast i8* %103 to i64*, !dbg !43
  %105 = load i64, i64* %104, align 8, !dbg !43
  %106 = add nsw i64 %101, %105, !dbg !43
  store i64 %106, i64* %.g0000_406, align 8, !dbg !43
  %107 = bitcast i64* %"newsyy$sd3" to i8*, !dbg !43
  %108 = bitcast i64* @.C284_drb067_foo_ to i8*, !dbg !43
  %109 = bitcast i64* @.C347_drb067_foo_ to i8*, !dbg !43
  %110 = bitcast i64* @.C309_drb067_foo_ to i8*, !dbg !43
  %111 = bitcast i64* %"newsyy$sd3" to i8*, !dbg !43
  %112 = getelementptr i8, i8* %111, i64 80, !dbg !43
  %113 = bitcast i64* %.g0000_406 to i8*, !dbg !43
  %114 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !43
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %114(i8* %107, i8* %108, i8* %109, i8* %110, i8* %112, i8* %113), !dbg !43
  %115 = bitcast i64* %"newsyy$sd3" to i8*, !dbg !43
  %116 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !43
  call void (i8*, i32, ...) %116(i8* %115, i32 28), !dbg !43
  %117 = bitcast i64* %"newsyy$sd3" to i8*, !dbg !43
  %118 = getelementptr i8, i8* %117, i64 88, !dbg !43
  %119 = bitcast i8* %118 to i64*, !dbg !43
  %120 = load i64, i64* %119, align 8, !dbg !43
  %121 = sub nsw i64 %120, 1, !dbg !43
  %122 = bitcast i64* %"newsyy$sd3" to i8*, !dbg !43
  %123 = getelementptr i8, i8* %122, i64 80, !dbg !43
  %124 = bitcast i8* %123 to i64*, !dbg !43
  %125 = load i64, i64* %124, align 8, !dbg !43
  %126 = add nsw i64 %121, %125, !dbg !43
  %127 = bitcast i64* %"newsyy$sd3" to i8*, !dbg !43
  %128 = getelementptr i8, i8* %127, i64 80, !dbg !43
  %129 = bitcast i8* %128 to i64*, !dbg !43
  %130 = load i64, i64* %129, align 8, !dbg !43
  %131 = sub nsw i64 %130, 1, !dbg !43
  %132 = sub nsw i64 %126, %131, !dbg !43
  store i64 %132, i64* %.g0000_406, align 8, !dbg !43
  %133 = bitcast i64* %.g0000_406 to i8*, !dbg !43
  %134 = bitcast i64* @.C347_drb067_foo_ to i8*, !dbg !43
  %135 = bitcast i64* @.C309_drb067_foo_ to i8*, !dbg !43
  %136 = bitcast i64* %"newsyy$p" to i8*, !dbg !43
  %137 = bitcast i64* @.C286_drb067_foo_ to i8*, !dbg !43
  %138 = bitcast i64* @.C284_drb067_foo_ to i8*, !dbg !43
  %139 = bitcast void (...)* @f90_alloc04a_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !43
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %139(i8* %133, i8* %134, i8* %135, i8* null, i8* %136, i8* null, i8* %137, i8* %138, i8* null, i64 0), !dbg !43
  %140 = bitcast i64* %"newsyy$sd3" to i8*, !dbg !43
  %141 = getelementptr i8, i8* %140, i64 64, !dbg !43
  %142 = bitcast i64* %"newsyy$p" to i8*, !dbg !43
  %143 = bitcast void (...)* @f90_ptrcp to void (i8*, i8*, ...)*, !dbg !43
  call void (i8*, i8*, ...) %143(i8* %141, i8* %142), !dbg !43
  %144 = bitcast i64* %"newsxx$p" to i8**, !dbg !44
  %145 = load i8*, i8** %144, align 8, !dbg !44
  %146 = bitcast i64* %"newsxx$sd1" to i8*, !dbg !44
  %147 = load double*, double** %"tar1$p_332", align 8, !dbg !44
  %148 = bitcast double* %147 to i8*, !dbg !44
  %149 = bitcast [16 x i64]* %"tar1$sd6_352" to i8*, !dbg !44
  %150 = bitcast i64* @.C284_drb067_foo_ to i8*, !dbg !44
  %151 = bitcast i64 (...)* @fort_ptr_assn_i8 to i64 (i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !44
  %152 = call i64 (i8*, i8*, i8*, i8*, i8*, ...) %151(i8* %145, i8* %146, i8* %148, i8* %149, i8* %150), !dbg !44
  %153 = inttoptr i64 %152 to i8*, !dbg !44
  %154 = bitcast i64* %"newsxx$p" to i8**, !dbg !44
  store i8* %153, i8** %154, align 8, !dbg !44
  %155 = bitcast i64* %"newsyy$p" to i8**, !dbg !45
  %156 = load i8*, i8** %155, align 8, !dbg !45
  %157 = bitcast i64* %"newsyy$sd3" to i8*, !dbg !45
  %158 = load double*, double** %"tar2$p_333", align 8, !dbg !45
  %159 = bitcast double* %158 to i8*, !dbg !45
  %160 = bitcast [16 x i64]* %"tar2$sd5_351" to i8*, !dbg !45
  %161 = bitcast i64* @.C284_drb067_foo_ to i8*, !dbg !45
  %162 = bitcast i64 (...)* @fort_ptr_assn_i8 to i64 (i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !45
  %163 = call i64 (i8*, i8*, i8*, i8*, i8*, ...) %162(i8* %156, i8* %157, i8* %159, i8* %160, i8* %161), !dbg !45
  %164 = inttoptr i64 %163 to i8*, !dbg !45
  %165 = bitcast i64* %"newsyy$p" to i8**, !dbg !45
  store i8* %164, i8** %165, align 8, !dbg !45
  %166 = bitcast i32* %len_312 to i8*, !dbg !46
  %167 = bitcast %astruct.dt90* %.uplevelArgPack0001_420 to i8**, !dbg !46
  store i8* %166, i8** %167, align 8, !dbg !46
  %168 = bitcast double** %"tar1$p_332" to i8*, !dbg !46
  %169 = bitcast %astruct.dt90* %.uplevelArgPack0001_420 to i8*, !dbg !46
  %170 = getelementptr i8, i8* %169, i64 8, !dbg !46
  %171 = bitcast i8* %170 to i8**, !dbg !46
  store i8* %168, i8** %171, align 8, !dbg !46
  %172 = bitcast double** %"tar1$p_332" to i8*, !dbg !46
  %173 = bitcast %astruct.dt90* %.uplevelArgPack0001_420 to i8*, !dbg !46
  %174 = getelementptr i8, i8* %173, i64 16, !dbg !46
  %175 = bitcast i8* %174 to i8**, !dbg !46
  store i8* %172, i8** %175, align 8, !dbg !46
  %176 = bitcast i64* %z_e_110_322 to i8*, !dbg !46
  %177 = bitcast %astruct.dt90* %.uplevelArgPack0001_420 to i8*, !dbg !46
  %178 = getelementptr i8, i8* %177, i64 24, !dbg !46
  %179 = bitcast i8* %178 to i8**, !dbg !46
  store i8* %176, i8** %179, align 8, !dbg !46
  %180 = bitcast double** %"tar2$p_333" to i8*, !dbg !46
  %181 = bitcast %astruct.dt90* %.uplevelArgPack0001_420 to i8*, !dbg !46
  %182 = getelementptr i8, i8* %181, i64 32, !dbg !46
  %183 = bitcast i8* %182 to i8**, !dbg !46
  store i8* %180, i8** %183, align 8, !dbg !46
  %184 = bitcast double** %"tar2$p_333" to i8*, !dbg !46
  %185 = bitcast %astruct.dt90* %.uplevelArgPack0001_420 to i8*, !dbg !46
  %186 = getelementptr i8, i8* %185, i64 40, !dbg !46
  %187 = bitcast i8* %186 to i8**, !dbg !46
  store i8* %184, i8** %187, align 8, !dbg !46
  %188 = bitcast [16 x i64]* %"tar1$sd6_352" to i8*, !dbg !46
  %189 = bitcast %astruct.dt90* %.uplevelArgPack0001_420 to i8*, !dbg !46
  %190 = getelementptr i8, i8* %189, i64 48, !dbg !46
  %191 = bitcast i8* %190 to i8**, !dbg !46
  store i8* %188, i8** %191, align 8, !dbg !46
  %192 = bitcast [16 x i64]* %"tar2$sd5_351" to i8*, !dbg !46
  %193 = bitcast %astruct.dt90* %.uplevelArgPack0001_420 to i8*, !dbg !46
  %194 = getelementptr i8, i8* %193, i64 56, !dbg !46
  %195 = bitcast i8* %194 to i8**, !dbg !46
  store i8* %192, i8** %195, align 8, !dbg !46
  br label %L.LB2_437, !dbg !46

L.LB2_437:                                        ; preds = %L.LB2_392
  %196 = bitcast void (i32*, i64*, i64*)* @__nv_drb067_foo__F1L27_1_ to i64*, !dbg !46
  %197 = bitcast %astruct.dt90* %.uplevelArgPack0001_420 to i64*, !dbg !46
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %196, i64* %197), !dbg !46
  call void (...) @_mp_bcs_nest(), !dbg !47
  %198 = bitcast i32* @.C338_drb067_foo_ to i8*, !dbg !47
  %199 = bitcast [60 x i8]* @.C336_drb067_foo_ to i8*, !dbg !47
  %200 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !47
  call void (i8*, i8*, i64, ...) %200(i8* %198, i8* %199, i64 60), !dbg !47
  %201 = bitcast i32* @.C339_drb067_foo_ to i8*, !dbg !47
  %202 = bitcast i32* @.C283_drb067_foo_ to i8*, !dbg !47
  %203 = bitcast i32* @.C283_drb067_foo_ to i8*, !dbg !47
  %204 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !47
  %205 = call i32 (i8*, i8*, i8*, i8*, ...) %204(i8* %201, i8* null, i8* %202, i8* %203), !dbg !47
  call void @llvm.dbg.declare(metadata i32* %z__io_341, metadata !48, metadata !DIExpression()), !dbg !25
  store i32 %205, i32* %z__io_341, align 4, !dbg !47
  %206 = load i32, i32* %len_312, align 4, !dbg !47
  call void @llvm.dbg.value(metadata i32 %206, metadata !38, metadata !DIExpression()), !dbg !25
  %207 = sext i32 %206 to i64, !dbg !47
  %208 = load double*, double** %"tar1$p_332", align 8, !dbg !47
  %209 = bitcast double* %208 to i8*, !dbg !47
  %210 = getelementptr i8, i8* %209, i64 -8, !dbg !47
  %211 = bitcast i8* %210 to double*, !dbg !47
  %212 = getelementptr double, double* %211, i64 %207, !dbg !47
  %213 = load double, double* %212, align 8, !dbg !47
  %214 = bitcast i32 (...)* @f90io_sc_d_ldw to i32 (double, i32, ...)*, !dbg !47
  %215 = call i32 (double, i32, ...) %214(double %213, i32 28), !dbg !47
  store i32 %215, i32* %z__io_341, align 4, !dbg !47
  %216 = load i32, i32* %len_312, align 4, !dbg !47
  call void @llvm.dbg.value(metadata i32 %216, metadata !38, metadata !DIExpression()), !dbg !25
  %217 = sext i32 %216 to i64, !dbg !47
  %218 = load double*, double** %"tar2$p_333", align 8, !dbg !47
  %219 = bitcast double* %218 to i8*, !dbg !47
  %220 = getelementptr i8, i8* %219, i64 -8, !dbg !47
  %221 = bitcast i8* %220 to double*, !dbg !47
  %222 = getelementptr double, double* %221, i64 %217, !dbg !47
  %223 = load double, double* %222, align 8, !dbg !47
  %224 = bitcast i32 (...)* @f90io_sc_d_ldw to i32 (double, i32, ...)*, !dbg !47
  %225 = call i32 (double, i32, ...) %224(double %223, i32 28), !dbg !47
  store i32 %225, i32* %z__io_341, align 4, !dbg !47
  %226 = call i32 (...) @f90io_ldw_end(), !dbg !47
  store i32 %226, i32* %z__io_341, align 4, !dbg !47
  call void (...) @_mp_ecs_nest(), !dbg !47
  %227 = bitcast i64* %"newsxx$p" to i8**, !dbg !49
  %228 = load i8*, i8** %227, align 8, !dbg !49
  %229 = icmp eq i8* %228, null, !dbg !49
  br i1 %229, label %L.LB2_372, label %L.LB2_466, !dbg !49

L.LB2_466:                                        ; preds = %L.LB2_437
  %230 = bitcast i64* %"newsxx$p" to i8**, !dbg !49
  store i8* null, i8** %230, align 8, !dbg !49
  store i64 0, i64* %"newsxx$sd1", align 8, !dbg !49
  br label %L.LB2_372

L.LB2_372:                                        ; preds = %L.LB2_466, %L.LB2_437
  %231 = bitcast i64* %"newsyy$p" to i8**, !dbg !50
  %232 = load i8*, i8** %231, align 8, !dbg !50
  %233 = icmp eq i8* %232, null, !dbg !50
  br i1 %233, label %L.LB2_373, label %L.LB2_467, !dbg !50

L.LB2_467:                                        ; preds = %L.LB2_372
  %234 = bitcast i64* %"newsyy$p" to i8**, !dbg !50
  store i8* null, i8** %234, align 8, !dbg !50
  store i64 0, i64* %"newsyy$sd3", align 8, !dbg !50
  br label %L.LB2_373

L.LB2_373:                                        ; preds = %L.LB2_467, %L.LB2_372
  %235 = load double*, double** %"tar2$p_333", align 8, !dbg !39
  %236 = bitcast double* %235 to i8*, !dbg !39
  %237 = bitcast void (...)* @f90_auto_dealloc_i8 to void (i8*, ...)*, !dbg !39
  call void (i8*, ...) %237(i8* %236), !dbg !39
  %238 = bitcast double** %"tar2$p_333" to i8**, !dbg !39
  store i8* null, i8** %238, align 8, !dbg !39
  %239 = bitcast [16 x i64]* %"tar2$sd5_351" to i64*, !dbg !39
  store i64 0, i64* %239, align 8, !dbg !39
  %240 = load double*, double** %"tar1$p_332", align 8, !dbg !39
  %241 = bitcast double* %240 to i8*, !dbg !39
  %242 = bitcast void (...)* @f90_auto_dealloc_i8 to void (i8*, ...)*, !dbg !39
  call void (i8*, ...) %242(i8* %241), !dbg !39
  %243 = bitcast double** %"tar1$p_332" to i8**, !dbg !39
  store i8* null, i8** %243, align 8, !dbg !39
  %244 = bitcast [16 x i64]* %"tar1$sd6_352" to i64*, !dbg !39
  store i64 0, i64* %244, align 8, !dbg !39
  ret void, !dbg !39
}

define internal void @__nv_drb067_foo__F1L27_1_(i32* %__nv_drb067_foo__F1L27_1Arg0, i64* %__nv_drb067_foo__F1L27_1Arg1, i64* %__nv_drb067_foo__F1L27_1Arg2) #1 !dbg !51 {
L.entry:
  %__gtid___nv_drb067_foo__F1L27_1__487 = alloca i32, align 4
  %len_330 = alloca i32, align 4
  %.i0000p_331 = alloca i32, align 4
  %i_329 = alloca i32, align 4
  %.du0001p_363 = alloca i32, align 4
  %.de0001p_364 = alloca i32, align 4
  %.di0001p_365 = alloca i32, align 4
  %.ds0001p_366 = alloca i32, align 4
  %.dl0001p_368 = alloca i32, align 4
  %.dl0001p.copy_481 = alloca i32, align 4
  %.de0001p.copy_482 = alloca i32, align 4
  %.ds0001p.copy_483 = alloca i32, align 4
  %.dX0001p_367 = alloca i32, align 4
  %.dY0001p_362 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_drb067_foo__F1L27_1Arg0, metadata !54, metadata !DIExpression()), !dbg !55
  call void @llvm.dbg.declare(metadata i64* %__nv_drb067_foo__F1L27_1Arg1, metadata !56, metadata !DIExpression()), !dbg !55
  call void @llvm.dbg.declare(metadata i64* %__nv_drb067_foo__F1L27_1Arg2, metadata !57, metadata !DIExpression()), !dbg !55
  call void @llvm.dbg.value(metadata i32 1, metadata !58, metadata !DIExpression()), !dbg !55
  call void @llvm.dbg.value(metadata i32 0, metadata !59, metadata !DIExpression()), !dbg !55
  call void @llvm.dbg.value(metadata i32 1, metadata !60, metadata !DIExpression()), !dbg !55
  call void @llvm.dbg.value(metadata i32 0, metadata !61, metadata !DIExpression()), !dbg !55
  call void @llvm.dbg.value(metadata i32 1, metadata !62, metadata !DIExpression()), !dbg !55
  call void @llvm.dbg.value(metadata i32 8, metadata !63, metadata !DIExpression()), !dbg !55
  %0 = load i32, i32* %__nv_drb067_foo__F1L27_1Arg0, align 4, !dbg !64
  store i32 %0, i32* %__gtid___nv_drb067_foo__F1L27_1__487, align 4, !dbg !64
  br label %L.LB3_471

L.LB3_471:                                        ; preds = %L.entry
  br label %L.LB3_328

L.LB3_328:                                        ; preds = %L.LB3_471
  %1 = bitcast i64* %__nv_drb067_foo__F1L27_1Arg2 to i32**, !dbg !65
  %2 = load i32*, i32** %1, align 8, !dbg !65
  %3 = load i32, i32* %2, align 4, !dbg !65
  call void @llvm.dbg.declare(metadata i32* %len_330, metadata !66, metadata !DIExpression()), !dbg !64
  store i32 %3, i32* %len_330, align 4, !dbg !65
  store i32 0, i32* %.i0000p_331, align 4, !dbg !67
  call void @llvm.dbg.declare(metadata i32* %i_329, metadata !68, metadata !DIExpression()), !dbg !64
  store i32 1, i32* %i_329, align 4, !dbg !67
  %4 = load i32, i32* %len_330, align 4, !dbg !67
  call void @llvm.dbg.value(metadata i32 %4, metadata !66, metadata !DIExpression()), !dbg !64
  store i32 %4, i32* %.du0001p_363, align 4, !dbg !67
  %5 = load i32, i32* %len_330, align 4, !dbg !67
  call void @llvm.dbg.value(metadata i32 %5, metadata !66, metadata !DIExpression()), !dbg !64
  store i32 %5, i32* %.de0001p_364, align 4, !dbg !67
  store i32 1, i32* %.di0001p_365, align 4, !dbg !67
  %6 = load i32, i32* %.di0001p_365, align 4, !dbg !67
  store i32 %6, i32* %.ds0001p_366, align 4, !dbg !67
  store i32 1, i32* %.dl0001p_368, align 4, !dbg !67
  %7 = load i32, i32* %.dl0001p_368, align 4, !dbg !67
  store i32 %7, i32* %.dl0001p.copy_481, align 4, !dbg !67
  %8 = load i32, i32* %.de0001p_364, align 4, !dbg !67
  store i32 %8, i32* %.de0001p.copy_482, align 4, !dbg !67
  %9 = load i32, i32* %.ds0001p_366, align 4, !dbg !67
  store i32 %9, i32* %.ds0001p.copy_483, align 4, !dbg !67
  %10 = load i32, i32* %__gtid___nv_drb067_foo__F1L27_1__487, align 4, !dbg !67
  %11 = bitcast i32* %.i0000p_331 to i64*, !dbg !67
  %12 = bitcast i32* %.dl0001p.copy_481 to i64*, !dbg !67
  %13 = bitcast i32* %.de0001p.copy_482 to i64*, !dbg !67
  %14 = bitcast i32* %.ds0001p.copy_483 to i64*, !dbg !67
  %15 = load i32, i32* %.ds0001p.copy_483, align 4, !dbg !67
  call void @__kmpc_for_static_init_4(i64* null, i32 %10, i32 34, i64* %11, i64* %12, i64* %13, i64* %14, i32 %15, i32 1), !dbg !67
  %16 = load i32, i32* %.dl0001p.copy_481, align 4, !dbg !67
  store i32 %16, i32* %.dl0001p_368, align 4, !dbg !67
  %17 = load i32, i32* %.de0001p.copy_482, align 4, !dbg !67
  store i32 %17, i32* %.de0001p_364, align 4, !dbg !67
  %18 = load i32, i32* %.ds0001p.copy_483, align 4, !dbg !67
  store i32 %18, i32* %.ds0001p_366, align 4, !dbg !67
  %19 = load i32, i32* %.dl0001p_368, align 4, !dbg !67
  store i32 %19, i32* %i_329, align 4, !dbg !67
  %20 = load i32, i32* %i_329, align 4, !dbg !67
  call void @llvm.dbg.value(metadata i32 %20, metadata !68, metadata !DIExpression()), !dbg !64
  store i32 %20, i32* %.dX0001p_367, align 4, !dbg !67
  %21 = load i32, i32* %.dX0001p_367, align 4, !dbg !67
  %22 = load i32, i32* %.du0001p_363, align 4, !dbg !67
  %23 = icmp sgt i32 %21, %22, !dbg !67
  br i1 %23, label %L.LB3_361, label %L.LB3_510, !dbg !67

L.LB3_510:                                        ; preds = %L.LB3_328
  %24 = load i32, i32* %.dX0001p_367, align 4, !dbg !67
  store i32 %24, i32* %i_329, align 4, !dbg !67
  %25 = load i32, i32* %.di0001p_365, align 4, !dbg !67
  %26 = load i32, i32* %.de0001p_364, align 4, !dbg !67
  %27 = load i32, i32* %.dX0001p_367, align 4, !dbg !67
  %28 = sub nsw i32 %26, %27, !dbg !67
  %29 = add nsw i32 %25, %28, !dbg !67
  %30 = load i32, i32* %.di0001p_365, align 4, !dbg !67
  %31 = sdiv i32 %29, %30, !dbg !67
  store i32 %31, i32* %.dY0001p_362, align 4, !dbg !67
  %32 = load i32, i32* %.dY0001p_362, align 4, !dbg !67
  %33 = icmp sle i32 %32, 0, !dbg !67
  br i1 %33, label %L.LB3_371, label %L.LB3_370, !dbg !67

L.LB3_370:                                        ; preds = %L.LB3_370, %L.LB3_510
  %34 = load i32, i32* %i_329, align 4, !dbg !69
  call void @llvm.dbg.value(metadata i32 %34, metadata !68, metadata !DIExpression()), !dbg !64
  %35 = sext i32 %34 to i64, !dbg !69
  %36 = bitcast i64* %__nv_drb067_foo__F1L27_1Arg2 to i8*, !dbg !69
  %37 = getelementptr i8, i8* %36, i64 16, !dbg !69
  %38 = bitcast i8* %37 to i8***, !dbg !69
  %39 = load i8**, i8*** %38, align 8, !dbg !69
  %40 = load i8*, i8** %39, align 8, !dbg !69
  %41 = getelementptr i8, i8* %40, i64 -8, !dbg !69
  %42 = bitcast i8* %41 to double*, !dbg !69
  %43 = getelementptr double, double* %42, i64 %35, !dbg !69
  store double 0.000000e+00, double* %43, align 8, !dbg !69
  %44 = load i32, i32* %i_329, align 4, !dbg !70
  call void @llvm.dbg.value(metadata i32 %44, metadata !68, metadata !DIExpression()), !dbg !64
  %45 = sext i32 %44 to i64, !dbg !70
  %46 = bitcast i64* %__nv_drb067_foo__F1L27_1Arg2 to i8*, !dbg !70
  %47 = getelementptr i8, i8* %46, i64 40, !dbg !70
  %48 = bitcast i8* %47 to i8***, !dbg !70
  %49 = load i8**, i8*** %48, align 8, !dbg !70
  %50 = load i8*, i8** %49, align 8, !dbg !70
  %51 = getelementptr i8, i8* %50, i64 -8, !dbg !70
  %52 = bitcast i8* %51 to double*, !dbg !70
  %53 = getelementptr double, double* %52, i64 %45, !dbg !70
  store double 0.000000e+00, double* %53, align 8, !dbg !70
  %54 = load i32, i32* %.di0001p_365, align 4, !dbg !64
  %55 = load i32, i32* %i_329, align 4, !dbg !64
  call void @llvm.dbg.value(metadata i32 %55, metadata !68, metadata !DIExpression()), !dbg !64
  %56 = add nsw i32 %54, %55, !dbg !64
  store i32 %56, i32* %i_329, align 4, !dbg !64
  %57 = load i32, i32* %.dY0001p_362, align 4, !dbg !64
  %58 = sub nsw i32 %57, 1, !dbg !64
  store i32 %58, i32* %.dY0001p_362, align 4, !dbg !64
  %59 = load i32, i32* %.dY0001p_362, align 4, !dbg !64
  %60 = icmp sgt i32 %59, 0, !dbg !64
  br i1 %60, label %L.LB3_370, label %L.LB3_371, !dbg !64

L.LB3_371:                                        ; preds = %L.LB3_370, %L.LB3_510
  br label %L.LB3_361

L.LB3_361:                                        ; preds = %L.LB3_371, %L.LB3_328
  %61 = load i32, i32* %__gtid___nv_drb067_foo__F1L27_1__487, align 4, !dbg !64
  call void @__kmpc_for_static_fini(i64* null, i32 %61), !dbg !64
  br label %L.LB3_334

L.LB3_334:                                        ; preds = %L.LB3_361
  ret void, !dbg !64
}

define void @MAIN_() #1 !dbg !2 {
L.entry:
  %"newsyy$p_325" = alloca double*, align 8
  %"newsyy$sd10_324" = alloca [16 x i64], align 8
  %"newsxx$p_321" = alloca double*, align 8
  %"newsxx$sd9_320" = alloca [16 x i64], align 8
  %.g0001_365 = alloca i64, align 8
  %"MAIN___$eq_297" = alloca [288 x i8], align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !71, metadata !DIExpression()), !dbg !72
  call void @llvm.dbg.value(metadata i32 0, metadata !73, metadata !DIExpression()), !dbg !72
  call void @llvm.dbg.value(metadata i32 1, metadata !74, metadata !DIExpression()), !dbg !72
  call void @llvm.dbg.value(metadata i32 0, metadata !75, metadata !DIExpression()), !dbg !72
  call void @llvm.dbg.value(metadata i32 1, metadata !76, metadata !DIExpression()), !dbg !72
  call void @llvm.dbg.value(metadata i32 8, metadata !77, metadata !DIExpression()), !dbg !72
  %0 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !78
  %1 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !78
  call void (i8*, ...) %1(i8* %0), !dbg !78
  call void @llvm.dbg.declare(metadata double** %"newsyy$p_325", metadata !79, metadata !DIExpression(DW_OP_deref)), !dbg !72
  %2 = bitcast double** %"newsyy$p_325" to i8**, !dbg !78
  store i8* null, i8** %2, align 8, !dbg !78
  call void @llvm.dbg.declare(metadata [16 x i64]* %"newsyy$sd10_324", metadata !80, metadata !DIExpression()), !dbg !72
  %3 = bitcast [16 x i64]* %"newsyy$sd10_324" to i64*, !dbg !78
  store i64 0, i64* %3, align 8, !dbg !78
  call void @llvm.dbg.declare(metadata double** %"newsxx$p_321", metadata !81, metadata !DIExpression(DW_OP_deref)), !dbg !72
  %4 = bitcast double** %"newsxx$p_321" to i8**, !dbg !78
  store i8* null, i8** %4, align 8, !dbg !78
  call void @llvm.dbg.declare(metadata [16 x i64]* %"newsxx$sd9_320", metadata !80, metadata !DIExpression()), !dbg !72
  %5 = bitcast [16 x i64]* %"newsxx$sd9_320" to i64*, !dbg !78
  store i64 0, i64* %5, align 8, !dbg !78
  br label %L.LB5_348

L.LB5_348:                                        ; preds = %L.entry
  %6 = bitcast [16 x i64]* %"newsxx$sd9_320" to i8*, !dbg !82
  %7 = getelementptr i8, i8* %6, i64 80, !dbg !82
  %8 = bitcast i8* %7 to i64*, !dbg !82
  store i64 1, i64* %8, align 8, !dbg !82
  %9 = bitcast %struct.STATICS5* @.STATICS5 to i32*, !dbg !82
  %10 = load i32, i32* %9, align 4, !dbg !82
  %11 = sext i32 %10 to i64, !dbg !82
  %12 = bitcast [16 x i64]* %"newsxx$sd9_320" to i8*, !dbg !82
  %13 = getelementptr i8, i8* %12, i64 88, !dbg !82
  %14 = bitcast i8* %13 to i64*, !dbg !82
  store i64 %11, i64* %14, align 8, !dbg !82
  %15 = bitcast [16 x i64]* %"newsxx$sd9_320" to i8*, !dbg !82
  %16 = getelementptr i8, i8* %15, i64 88, !dbg !82
  %17 = bitcast i8* %16 to i64*, !dbg !82
  %18 = load i64, i64* %17, align 8, !dbg !82
  %19 = sub nsw i64 %18, 1, !dbg !82
  %20 = bitcast [16 x i64]* %"newsxx$sd9_320" to i8*, !dbg !82
  %21 = getelementptr i8, i8* %20, i64 80, !dbg !82
  %22 = bitcast i8* %21 to i64*, !dbg !82
  %23 = load i64, i64* %22, align 8, !dbg !82
  %24 = add nsw i64 %19, %23, !dbg !82
  store i64 %24, i64* %.g0001_365, align 8, !dbg !82
  %25 = bitcast [16 x i64]* %"newsxx$sd9_320" to i8*, !dbg !82
  %26 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !82
  %27 = bitcast i64* @.C328_MAIN_ to i8*, !dbg !82
  %28 = bitcast i64* @.C310_MAIN_ to i8*, !dbg !82
  %29 = bitcast [16 x i64]* %"newsxx$sd9_320" to i8*, !dbg !82
  %30 = getelementptr i8, i8* %29, i64 80, !dbg !82
  %31 = bitcast i64* %.g0001_365 to i8*, !dbg !82
  %32 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !82
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %32(i8* %25, i8* %26, i8* %27, i8* %28, i8* %30, i8* %31), !dbg !82
  %33 = bitcast [16 x i64]* %"newsxx$sd9_320" to i8*, !dbg !82
  %34 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !82
  call void (i8*, i32, ...) %34(i8* %33, i32 28), !dbg !82
  %35 = bitcast [16 x i64]* %"newsxx$sd9_320" to i8*, !dbg !82
  %36 = getelementptr i8, i8* %35, i64 88, !dbg !82
  %37 = bitcast i8* %36 to i64*, !dbg !82
  %38 = load i64, i64* %37, align 8, !dbg !82
  %39 = sub nsw i64 %38, 1, !dbg !82
  %40 = bitcast [16 x i64]* %"newsxx$sd9_320" to i8*, !dbg !82
  %41 = getelementptr i8, i8* %40, i64 80, !dbg !82
  %42 = bitcast i8* %41 to i64*, !dbg !82
  %43 = load i64, i64* %42, align 8, !dbg !82
  %44 = add nsw i64 %39, %43, !dbg !82
  %45 = bitcast [16 x i64]* %"newsxx$sd9_320" to i8*, !dbg !82
  %46 = getelementptr i8, i8* %45, i64 80, !dbg !82
  %47 = bitcast i8* %46 to i64*, !dbg !82
  %48 = load i64, i64* %47, align 8, !dbg !82
  %49 = sub nsw i64 %48, 1, !dbg !82
  %50 = sub nsw i64 %44, %49, !dbg !82
  store i64 %50, i64* %.g0001_365, align 8, !dbg !82
  %51 = bitcast i64* %.g0001_365 to i8*, !dbg !82
  %52 = bitcast i64* @.C328_MAIN_ to i8*, !dbg !82
  %53 = bitcast i64* @.C310_MAIN_ to i8*, !dbg !82
  %54 = bitcast double** %"newsxx$p_321" to i8*, !dbg !82
  %55 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !82
  %56 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !82
  %57 = bitcast void (...)* @f90_alloc04a_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !82
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %57(i8* %51, i8* %52, i8* %53, i8* null, i8* %54, i8* null, i8* %55, i8* %56, i8* null, i64 0), !dbg !82
  %58 = bitcast [16 x i64]* %"newsxx$sd9_320" to i8*, !dbg !82
  %59 = getelementptr i8, i8* %58, i64 64, !dbg !82
  %60 = bitcast double** %"newsxx$p_321" to i8*, !dbg !82
  %61 = bitcast void (...)* @f90_ptrcp to void (i8*, i8*, ...)*, !dbg !82
  call void (i8*, i8*, ...) %61(i8* %59, i8* %60), !dbg !82
  %62 = bitcast [16 x i64]* %"newsyy$sd10_324" to i8*, !dbg !83
  %63 = getelementptr i8, i8* %62, i64 80, !dbg !83
  %64 = bitcast i8* %63 to i64*, !dbg !83
  store i64 1, i64* %64, align 8, !dbg !83
  %65 = bitcast %struct.STATICS5* @.STATICS5 to i32*, !dbg !83
  %66 = load i32, i32* %65, align 4, !dbg !83
  %67 = sext i32 %66 to i64, !dbg !83
  %68 = bitcast [16 x i64]* %"newsyy$sd10_324" to i8*, !dbg !83
  %69 = getelementptr i8, i8* %68, i64 88, !dbg !83
  %70 = bitcast i8* %69 to i64*, !dbg !83
  store i64 %67, i64* %70, align 8, !dbg !83
  %71 = bitcast [16 x i64]* %"newsyy$sd10_324" to i8*, !dbg !83
  %72 = getelementptr i8, i8* %71, i64 88, !dbg !83
  %73 = bitcast i8* %72 to i64*, !dbg !83
  %74 = load i64, i64* %73, align 8, !dbg !83
  %75 = sub nsw i64 %74, 1, !dbg !83
  %76 = bitcast [16 x i64]* %"newsyy$sd10_324" to i8*, !dbg !83
  %77 = getelementptr i8, i8* %76, i64 80, !dbg !83
  %78 = bitcast i8* %77 to i64*, !dbg !83
  %79 = load i64, i64* %78, align 8, !dbg !83
  %80 = add nsw i64 %75, %79, !dbg !83
  store i64 %80, i64* %.g0001_365, align 8, !dbg !83
  %81 = bitcast [16 x i64]* %"newsyy$sd10_324" to i8*, !dbg !83
  %82 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !83
  %83 = bitcast i64* @.C328_MAIN_ to i8*, !dbg !83
  %84 = bitcast i64* @.C310_MAIN_ to i8*, !dbg !83
  %85 = bitcast [16 x i64]* %"newsyy$sd10_324" to i8*, !dbg !83
  %86 = getelementptr i8, i8* %85, i64 80, !dbg !83
  %87 = bitcast i64* %.g0001_365 to i8*, !dbg !83
  %88 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !83
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %88(i8* %81, i8* %82, i8* %83, i8* %84, i8* %86, i8* %87), !dbg !83
  %89 = bitcast [16 x i64]* %"newsyy$sd10_324" to i8*, !dbg !83
  %90 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !83
  call void (i8*, i32, ...) %90(i8* %89, i32 28), !dbg !83
  %91 = bitcast [16 x i64]* %"newsyy$sd10_324" to i8*, !dbg !83
  %92 = getelementptr i8, i8* %91, i64 88, !dbg !83
  %93 = bitcast i8* %92 to i64*, !dbg !83
  %94 = load i64, i64* %93, align 8, !dbg !83
  %95 = sub nsw i64 %94, 1, !dbg !83
  %96 = bitcast [16 x i64]* %"newsyy$sd10_324" to i8*, !dbg !83
  %97 = getelementptr i8, i8* %96, i64 80, !dbg !83
  %98 = bitcast i8* %97 to i64*, !dbg !83
  %99 = load i64, i64* %98, align 8, !dbg !83
  %100 = add nsw i64 %95, %99, !dbg !83
  %101 = bitcast [16 x i64]* %"newsyy$sd10_324" to i8*, !dbg !83
  %102 = getelementptr i8, i8* %101, i64 80, !dbg !83
  %103 = bitcast i8* %102 to i64*, !dbg !83
  %104 = load i64, i64* %103, align 8, !dbg !83
  %105 = sub nsw i64 %104, 1, !dbg !83
  %106 = sub nsw i64 %100, %105, !dbg !83
  store i64 %106, i64* %.g0001_365, align 8, !dbg !83
  %107 = bitcast i64* %.g0001_365 to i8*, !dbg !83
  %108 = bitcast i64* @.C328_MAIN_ to i8*, !dbg !83
  %109 = bitcast i64* @.C310_MAIN_ to i8*, !dbg !83
  %110 = bitcast double** %"newsyy$p_325" to i8*, !dbg !83
  %111 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !83
  %112 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !83
  %113 = bitcast void (...)* @f90_alloc04a_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !83
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %113(i8* %107, i8* %108, i8* %109, i8* null, i8* %110, i8* null, i8* %111, i8* %112, i8* null, i64 0), !dbg !83
  %114 = bitcast [16 x i64]* %"newsyy$sd10_324" to i8*, !dbg !83
  %115 = getelementptr i8, i8* %114, i64 64, !dbg !83
  %116 = bitcast double** %"newsyy$p_325" to i8*, !dbg !83
  %117 = bitcast void (...)* @f90_ptrcp to void (i8*, i8*, ...)*, !dbg !83
  call void (i8*, i8*, ...) %117(i8* %115, i8* %116), !dbg !83
  %118 = bitcast double** %"newsxx$p_321" to i64*, !dbg !84
  %119 = bitcast double** %"newsyy$p_325" to i64*, !dbg !84
  %120 = bitcast %struct.STATICS5* @.STATICS5 to i32*, !dbg !84
  %121 = load i32, i32* %120, align 4, !dbg !84
  %122 = bitcast [16 x i64]* %"newsxx$sd9_320" to i64*, !dbg !84
  %123 = bitcast [16 x i64]* %"newsyy$sd10_324" to i64*, !dbg !84
  call void @drb067_foo_(i64* %118, i64* %119, i32 %121, i64* %122, i64* %123), !dbg !84
  %124 = load double*, double** %"newsxx$p_321", align 8, !dbg !85
  call void @llvm.dbg.value(metadata double* %124, metadata !81, metadata !DIExpression()), !dbg !72
  %125 = bitcast double* %124 to i8*, !dbg !85
  %126 = icmp eq i8* %125, null, !dbg !85
  br i1 %126, label %L.LB5_337, label %L.LB5_381, !dbg !85

L.LB5_381:                                        ; preds = %L.LB5_348
  %127 = bitcast double** %"newsxx$p_321" to i8**, !dbg !85
  store i8* null, i8** %127, align 8, !dbg !85
  %128 = bitcast [16 x i64]* %"newsxx$sd9_320" to i64*, !dbg !85
  store i64 0, i64* %128, align 8, !dbg !85
  br label %L.LB5_337

L.LB5_337:                                        ; preds = %L.LB5_381, %L.LB5_348
  %129 = load double*, double** %"newsyy$p_325", align 8, !dbg !86
  call void @llvm.dbg.value(metadata double* %129, metadata !79, metadata !DIExpression()), !dbg !72
  %130 = bitcast double* %129 to i8*, !dbg !86
  %131 = icmp eq i8* %130, null, !dbg !86
  br i1 %131, label %L.LB5_338, label %L.LB5_382, !dbg !86

L.LB5_382:                                        ; preds = %L.LB5_337
  %132 = bitcast double** %"newsyy$p_325" to i8**, !dbg !86
  store i8* null, i8** %132, align 8, !dbg !86
  %133 = bitcast [16 x i64]* %"newsyy$sd10_324" to i64*, !dbg !86
  store i64 0, i64* %133, align 8, !dbg !86
  br label %L.LB5_338

L.LB5_338:                                        ; preds = %L.LB5_382, %L.LB5_337
  ret void, !dbg !87
}

declare void @fort_init(...) #1

declare void @__kmpc_for_static_fini(i64*, i32) #1

declare void @__kmpc_for_static_init_4(i64*, i32, i32, i64*, i64*, i64*, i64*, i32, i32) #1

declare void @f90_auto_dealloc_i8(...) #1

declare void @_mp_ecs_nest(...) #1

declare signext i32 @f90io_ldw_end(...) #1

declare signext i32 @f90io_sc_d_ldw(...) #1

declare signext i32 @f90io_print_init(...) #1

declare void @f90io_src_info03a(...) #1

declare void @_mp_bcs_nest(...) #1

declare i64 @fort_ptr_assn_i8(...) #1

declare void @f90_ptrcp(...) #1

declare void @f90_alloc04a_i8(...) #1

declare void @f90_set_intrin_type_i8(...) #1

declare void @f90_template1_i8(...) #1

declare i64 @f90_auto_alloc04_i8(...) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

declare signext i32 @__kmpc_global_thread_num(i64*) #1

declare void @__kmpc_fork_call(i64*, i32, i64*, i64*, ...) #1

declare void @__kmpc_end_serialized_parallel(i64*, i32) #1

declare void @__kmpc_serialized_parallel(i64*, i32) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { noinline }
attributes #1 = { "no-frame-pointer-elim-non-leaf" }
attributes #2 = { nounwind readnone speculatable }

!llvm.module.flags = !{!10, !11}
!llvm.dbg.cu = !{!4}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "len", scope: !2, file: !3, type: !9, isLocal: true, isDefinition: true)
!2 = distinct !DISubprogram(name: "drb067_restrictpointer1_orig_no", scope: !4, file: !3, line: 41, type: !7, scopeLine: 41, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !4)
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB067-restrictpointer1-orig-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !6, imports: !5)
!5 = !{}
!6 = !{!0}
!7 = !DISubroutineType(cc: DW_CC_program, types: !8)
!8 = !{null}
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !{i32 2, !"Dwarf Version", i32 4}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = distinct !DISubprogram(name: "foo", scope: !13, file: !3, line: 14, type: !14, scopeLine: 14, spFlags: DISPFlagDefinition, unit: !4)
!13 = !DIModule(scope: !4, name: "drb067")
!14 = !DISubroutineType(types: !15)
!15 = !{null, !16, !16, !9, !20, !20}
!16 = !DICompositeType(tag: DW_TAG_array_type, baseType: !17, size: 64, align: 64, elements: !18)
!17 = !DIBasicType(name: "double precision", size: 64, align: 64, encoding: DW_ATE_float)
!18 = !{!19}
!19 = !DISubrange(count: 0, lowerBound: 1)
!20 = !DICompositeType(tag: DW_TAG_array_type, baseType: !21, size: 1024, align: 64, elements: !22)
!21 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!22 = !{!23}
!23 = !DISubrange(count: 16, lowerBound: 1)
!24 = !DILocalVariable(arg: 1, scope: !12, file: !3, type: !16, flags: DIFlagArtificial)
!25 = !DILocation(line: 0, scope: !12)
!26 = !DILocalVariable(arg: 2, scope: !12, file: !3, type: !16, flags: DIFlagArtificial)
!27 = !DILocalVariable(name: "_V_len", scope: !12, file: !3, type: !9)
!28 = !DILocalVariable(name: "_V_len", arg: 3, scope: !12, file: !3, type: !9)
!29 = !DILocalVariable(arg: 4, scope: !12, file: !3, type: !20, flags: DIFlagArtificial)
!30 = !DILocalVariable(arg: 5, scope: !12, file: !3, type: !20, flags: DIFlagArtificial)
!31 = !DILocalVariable(name: "omp_sched_static", scope: !12, file: !3, type: !9)
!32 = !DILocalVariable(name: "omp_proc_bind_false", scope: !12, file: !3, type: !9)
!33 = !DILocalVariable(name: "omp_proc_bind_true", scope: !12, file: !3, type: !9)
!34 = !DILocalVariable(name: "omp_lock_hint_none", scope: !12, file: !3, type: !9)
!35 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !12, file: !3, type: !9)
!36 = !DILocalVariable(name: "dp", scope: !12, file: !3, type: !9)
!37 = !DILocation(line: 14, column: 1, scope: !12)
!38 = !DILocalVariable(name: "len", scope: !12, file: !3, type: !9)
!39 = !DILocation(line: 39, column: 1, scope: !12)
!40 = !DILocalVariable(scope: !12, file: !3, type: !21, flags: DIFlagArtificial)
!41 = !DILocalVariable(scope: !12, file: !3, type: !20, flags: DIFlagArtificial)
!42 = !DILocation(line: 21, column: 1, scope: !12)
!43 = !DILocation(line: 22, column: 1, scope: !12)
!44 = !DILocation(line: 24, column: 1, scope: !12)
!45 = !DILocation(line: 25, column: 1, scope: !12)
!46 = !DILocation(line: 27, column: 1, scope: !12)
!47 = !DILocation(line: 34, column: 1, scope: !12)
!48 = !DILocalVariable(scope: !12, file: !3, type: !9, flags: DIFlagArtificial)
!49 = !DILocation(line: 36, column: 1, scope: !12)
!50 = !DILocation(line: 37, column: 1, scope: !12)
!51 = distinct !DISubprogram(name: "__nv_drb067_foo__F1L27_1", scope: !4, file: !3, line: 27, type: !52, scopeLine: 27, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !4)
!52 = !DISubroutineType(types: !53)
!53 = !{null, !9, !21, !21}
!54 = !DILocalVariable(name: "__nv_drb067_foo__F1L27_1Arg0", arg: 1, scope: !51, file: !3, type: !9)
!55 = !DILocation(line: 0, scope: !51)
!56 = !DILocalVariable(name: "__nv_drb067_foo__F1L27_1Arg1", arg: 2, scope: !51, file: !3, type: !21)
!57 = !DILocalVariable(name: "__nv_drb067_foo__F1L27_1Arg2", arg: 3, scope: !51, file: !3, type: !21)
!58 = !DILocalVariable(name: "omp_sched_static", scope: !51, file: !3, type: !9)
!59 = !DILocalVariable(name: "omp_proc_bind_false", scope: !51, file: !3, type: !9)
!60 = !DILocalVariable(name: "omp_proc_bind_true", scope: !51, file: !3, type: !9)
!61 = !DILocalVariable(name: "omp_lock_hint_none", scope: !51, file: !3, type: !9)
!62 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !51, file: !3, type: !9)
!63 = !DILocalVariable(name: "dp", scope: !51, file: !3, type: !9)
!64 = !DILocation(line: 31, column: 1, scope: !51)
!65 = !DILocation(line: 27, column: 1, scope: !51)
!66 = !DILocalVariable(name: "len", scope: !51, file: !3, type: !9)
!67 = !DILocation(line: 28, column: 1, scope: !51)
!68 = !DILocalVariable(name: "i", scope: !51, file: !3, type: !9)
!69 = !DILocation(line: 29, column: 1, scope: !51)
!70 = !DILocation(line: 30, column: 1, scope: !51)
!71 = !DILocalVariable(name: "omp_sched_static", scope: !2, file: !3, type: !9)
!72 = !DILocation(line: 0, scope: !2)
!73 = !DILocalVariable(name: "omp_proc_bind_false", scope: !2, file: !3, type: !9)
!74 = !DILocalVariable(name: "omp_proc_bind_true", scope: !2, file: !3, type: !9)
!75 = !DILocalVariable(name: "omp_lock_hint_none", scope: !2, file: !3, type: !9)
!76 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !2, file: !3, type: !9)
!77 = !DILocalVariable(name: "dp", scope: !2, file: !3, type: !9)
!78 = !DILocation(line: 41, column: 1, scope: !2)
!79 = !DILocalVariable(name: "newsyy", scope: !2, file: !3, type: !16)
!80 = !DILocalVariable(scope: !2, file: !3, type: !20, flags: DIFlagArtificial)
!81 = !DILocalVariable(name: "newsxx", scope: !2, file: !3, type: !16)
!82 = !DILocation(line: 50, column: 1, scope: !2)
!83 = !DILocation(line: 51, column: 1, scope: !2)
!84 = !DILocation(line: 53, column: 1, scope: !2)
!85 = !DILocation(line: 55, column: 1, scope: !2)
!86 = !DILocation(line: 56, column: 1, scope: !2)
!87 = !DILocation(line: 57, column: 1, scope: !2)

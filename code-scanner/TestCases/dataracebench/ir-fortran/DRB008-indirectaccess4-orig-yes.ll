; ModuleID = '/tmp/DRB008-indirectaccess4-orig-yes-21c0a8.ll'
source_filename = "/tmp/DRB008-indirectaccess4-orig-yes-21c0a8.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.BSS2 = type <{ [16200 x i8] }>
%struct.STATICS2 = type <{ [720 x i8] }>
%struct_drb008_0_ = type <{ [724 x i8] }>
%astruct.dt90 = type <{ i8*, i8*, i8*, i8* }>

@.BSS2 = internal global %struct.BSS2 zeroinitializer, align 32, !dbg !0
@.STATICS2 = internal global %struct.STATICS2 <{ [720 x i8] c"\09\02\00\00\15\02\00\00\0D\02\00\00\0F\02\00\00\11\02\00\00\13\02\00\00#\02\00\00%\02\00\00'\02\00\00)\02\00\00+\02\00\00-\02\00\00=\02\00\00?\02\00\00A\02\00\00C\02\00\00E\02\00\00G\02\00\00W\02\00\00Y\02\00\00[\02\00\00]\02\00\00_\02\00\00a\02\00\00q\02\00\00s\02\00\00u\02\00\00w\02\00\00y\02\00\00{\02\00\00\8B\02\00\00\8D\02\00\00\8F\02\00\00\91\02\00\00\93\02\00\00\95\02\00\00[\03\00\00]\03\00\00_\03\00\00a\03\00\00c\03\00\00e\03\00\00u\03\00\00w\03\00\00y\03\00\00{\03\00\00}\03\00\00\7F\03\00\00\8F\03\00\00\91\03\00\00\93\03\00\00\95\03\00\00\97\03\00\00\99\03\00\00\A9\03\00\00\AB\03\00\00\AD\03\00\00\AF\03\00\00\B1\03\00\00\B3\03\00\00\C3\03\00\00\C5\03\00\00\C7\03\00\00\C9\03\00\00\CB\03\00\00\CD\03\00\00\DD\03\00\00\DF\03\00\00\E1\03\00\00\E3\03\00\00\E5\03\00\00\E7\03\00\00\AD\04\00\00\AF\04\00\00\B1\04\00\00\B3\04\00\00\B5\04\00\00\B7\04\00\00\C7\04\00\00\C9\04\00\00\CB\04\00\00\CD\04\00\00\CF\04\00\00\D1\04\00\00\E1\04\00\00\E3\04\00\00\E5\04\00\00\E7\04\00\00\E9\04\00\00\EB\04\00\00\FB\04\00\00\FD\04\00\00\FF\04\00\00\01\05\00\00\03\05\00\00\05\05\00\00\15\05\00\00\17\05\00\00\19\05\00\00\1B\05\00\00\1D\05\00\00\1F\05\00\00/\05\00\001\05\00\003\05\00\005\05\00\007\05\00\009\05\00\00\FF\05\00\00\01\06\00\00\03\06\00\00\05\06\00\00\07\06\00\00\09\06\00\00\19\06\00\00\1B\06\00\00\1D\06\00\00\1F\06\00\00!\06\00\00#\06\00\003\06\00\005\06\00\007\06\00\009\06\00\00;\06\00\00=\06\00\00M\06\00\00O\06\00\00Q\06\00\00S\06\00\00U\06\00\00W\06\00\00g\06\00\00i\06\00\00k\06\00\00m\06\00\00o\06\00\00q\06\00\00\81\06\00\00\83\06\00\00\85\06\00\00\87\06\00\00\89\06\00\00\8B\06\00\00Q\07\00\00S\07\00\00U\07\00\00W\07\00\00Y\07\00\00[\07\00\00k\07\00\00m\07\00\00o\07\00\00q\07\00\00s\07\00\00u\07\00\00\85\07\00\00\87\07\00\00\89\07\00\00\8B\07\00\00\8D\07\00\00\8F\07\00\00\9F\07\00\00\A1\07\00\00\A3\07\00\00\A5\07\00\00\A7\07\00\00\A9\07\00\00\B9\07\00\00\BB\07\00\00\BD\07\00\00\BF\07\00\00\C1\07\00\00\C3\07\00\00\D3\07\00\00\D5\07\00\00\D7\07\00\00\D9\07\00\00\DB\07\00\00\DD\07\00\00" }>, align 16, !dbg !16
@.C352_MAIN_ = internal constant i64 1285
@.C351_MAIN_ = internal constant [12 x i8] c" xa2(1285) ="
@.C350_MAIN_ = internal constant i64 999
@.C306_MAIN_ = internal constant i32 14
@.C349_MAIN_ = internal constant [10 x i8] c"xa1(999) ="
@.C346_MAIN_ = internal constant i32 6
@.C343_MAIN_ = internal constant [60 x i8] c"micro-benchmarks-fortran/DRB008-indirectaccess4-orig-yes.f95"
@.C345_MAIN_ = internal constant i32 86
@.C340_MAIN_ = internal constant double 3.000000e+00
@.C292_MAIN_ = internal constant double 1.000000e+00
@.C339_MAIN_ = internal constant i32 12
@.C290_MAIN_ = internal constant float 5.000000e-01
@.C285_MAIN_ = internal constant i32 1
@.C328_MAIN_ = internal constant i32 2025
@.C332_MAIN_ = internal constant i32 521
@.C308_MAIN_ = internal constant i64 180
@.C331_MAIN_ = internal constant i32 180
@.C363_MAIN_ = internal constant i64 9
@.C320_MAIN_ = internal constant i64 12
@.C319_MAIN_ = internal constant i64 11
@.C307_MAIN_ = internal constant i32 28
@.C329_MAIN_ = internal constant i64 2025
@.C321_MAIN_ = internal constant i64 8
@.C361_MAIN_ = internal constant i64 28
@.C286_MAIN_ = internal constant i64 1
@.C284_MAIN_ = internal constant i64 0
@.C283_MAIN_ = internal constant i32 0
@.C340___nv_MAIN__F1L77_1 = internal constant double 3.000000e+00
@.C292___nv_MAIN__F1L77_1 = internal constant double 1.000000e+00
@.C339___nv_MAIN__F1L77_1 = internal constant i32 12
@.C285___nv_MAIN__F1L77_1 = internal constant i32 1
@.C283___nv_MAIN__F1L77_1 = internal constant i32 0
@_drb008_0_ = common global %struct_drb008_0_ zeroinitializer, align 64, !dbg !7, !dbg !14

; Function Attrs: noinline
define float @drb008_() #0 {
.L.entry:
  ret float undef
}

define void @MAIN_() #1 !dbg !2 {
L.entry:
  %__gtid_MAIN__462 = alloca i32, align 4
  %"xa2$p_325" = alloca double*, align 8
  %"xa2$sd7_326" = alloca [16 x i64], align 8
  %"xa1$p_318" = alloca double*, align 8
  %"xa1$sd5_322" = alloca [16 x i64], align 8
  %"base$sd9_365" = alloca [16 x i64], align 8
  %.g0000_415 = alloca i64, align 8
  %.dY0001_374 = alloca i64, align 8
  %"i$a_359" = alloca i64, align 8
  %.dY0002_377 = alloca i32, align 4
  %i_312 = alloca i32, align 4
  %idx1_313 = alloca i32, align 4
  %.uplevelArgPack0001_449 = alloca %astruct.dt90, align 16
  %idx2_314 = alloca i32, align 4
  %z__io_348 = alloca i32, align 4
  %"MAIN___$eq_297" = alloca [288 x i8], align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !34, metadata !DIExpression()), !dbg !35
  call void @llvm.dbg.value(metadata i32 0, metadata !36, metadata !DIExpression()), !dbg !35
  call void @llvm.dbg.value(metadata i32 1, metadata !37, metadata !DIExpression()), !dbg !35
  call void @llvm.dbg.value(metadata i32 0, metadata !38, metadata !DIExpression()), !dbg !35
  call void @llvm.dbg.value(metadata i32 1, metadata !39, metadata !DIExpression()), !dbg !35
  call void @llvm.dbg.value(metadata i32 8, metadata !40, metadata !DIExpression()), !dbg !35
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !41
  store i32 %0, i32* %__gtid_MAIN__462, align 4, !dbg !41
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !42
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !42
  call void (i8*, ...) %2(i8* %1), !dbg !42
  call void @llvm.dbg.declare(metadata double** %"xa2$p_325", metadata !43, metadata !DIExpression(DW_OP_deref)), !dbg !35
  %3 = bitcast double** %"xa2$p_325" to i8**, !dbg !42
  store i8* null, i8** %3, align 8, !dbg !42
  call void @llvm.dbg.declare(metadata [16 x i64]* %"xa2$sd7_326", metadata !47, metadata !DIExpression()), !dbg !35
  %4 = bitcast [16 x i64]* %"xa2$sd7_326" to i64*, !dbg !42
  store i64 0, i64* %4, align 8, !dbg !42
  call void @llvm.dbg.declare(metadata double** %"xa1$p_318", metadata !51, metadata !DIExpression(DW_OP_deref)), !dbg !35
  %5 = bitcast double** %"xa1$p_318" to i8**, !dbg !42
  store i8* null, i8** %5, align 8, !dbg !42
  call void @llvm.dbg.declare(metadata [16 x i64]* %"xa1$sd5_322", metadata !47, metadata !DIExpression()), !dbg !35
  %6 = bitcast [16 x i64]* %"xa1$sd5_322" to i64*, !dbg !42
  store i64 0, i64* %6, align 8, !dbg !42
  call void @llvm.dbg.declare(metadata [16 x i64]* %"base$sd9_365", metadata !47, metadata !DIExpression()), !dbg !35
  %7 = bitcast [16 x i64]* %"base$sd9_365" to i8*, !dbg !42
  %8 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !42
  %9 = bitcast i64* @.C361_MAIN_ to i8*, !dbg !42
  %10 = bitcast i64* @.C321_MAIN_ to i8*, !dbg !42
  %11 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !42
  %12 = bitcast i64* @.C329_MAIN_ to i8*, !dbg !42
  %13 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !42
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %13(i8* %7, i8* %8, i8* %9, i8* %10, i8* %11, i8* %12), !dbg !42
  %14 = bitcast [16 x i64]* %"base$sd9_365" to i8*, !dbg !42
  %15 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !42
  call void (i8*, i32, ...) %15(i8* %14, i32 28), !dbg !42
  br label %L.LB2_403

L.LB2_403:                                        ; preds = %L.entry
  %16 = bitcast [16 x i64]* %"xa1$sd5_322" to i8*, !dbg !52
  %17 = getelementptr i8, i8* %16, i64 80, !dbg !52
  %18 = bitcast i8* %17 to i64*, !dbg !52
  store i64 1, i64* %18, align 8, !dbg !52
  %19 = bitcast [16 x i64]* %"xa1$sd5_322" to i8*, !dbg !52
  %20 = getelementptr i8, i8* %19, i64 88, !dbg !52
  %21 = bitcast i8* %20 to i64*, !dbg !52
  store i64 2025, i64* %21, align 8, !dbg !52
  %22 = bitcast [16 x i64]* %"xa1$sd5_322" to i8*, !dbg !52
  %23 = getelementptr i8, i8* %22, i64 88, !dbg !52
  %24 = bitcast i8* %23 to i64*, !dbg !52
  %25 = load i64, i64* %24, align 8, !dbg !52
  %26 = sub nsw i64 %25, 1, !dbg !52
  %27 = bitcast [16 x i64]* %"xa1$sd5_322" to i8*, !dbg !52
  %28 = getelementptr i8, i8* %27, i64 80, !dbg !52
  %29 = bitcast i8* %28 to i64*, !dbg !52
  %30 = load i64, i64* %29, align 8, !dbg !52
  %31 = add nsw i64 %26, %30, !dbg !52
  store i64 %31, i64* %.g0000_415, align 8, !dbg !52
  %32 = bitcast [16 x i64]* %"xa1$sd5_322" to i8*, !dbg !52
  %33 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !52
  %34 = bitcast i64* @.C361_MAIN_ to i8*, !dbg !52
  %35 = bitcast i64* @.C321_MAIN_ to i8*, !dbg !52
  %36 = bitcast [16 x i64]* %"xa1$sd5_322" to i8*, !dbg !52
  %37 = getelementptr i8, i8* %36, i64 80, !dbg !52
  %38 = bitcast i64* %.g0000_415 to i8*, !dbg !52
  %39 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !52
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %39(i8* %32, i8* %33, i8* %34, i8* %35, i8* %37, i8* %38), !dbg !52
  %40 = bitcast [16 x i64]* %"xa1$sd5_322" to i8*, !dbg !52
  %41 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !52
  call void (i8*, i32, ...) %41(i8* %40, i32 28), !dbg !52
  %42 = bitcast [16 x i64]* %"xa1$sd5_322" to i8*, !dbg !52
  %43 = getelementptr i8, i8* %42, i64 88, !dbg !52
  %44 = bitcast i8* %43 to i64*, !dbg !52
  %45 = load i64, i64* %44, align 8, !dbg !52
  %46 = sub nsw i64 %45, 1, !dbg !52
  %47 = bitcast [16 x i64]* %"xa1$sd5_322" to i8*, !dbg !52
  %48 = getelementptr i8, i8* %47, i64 80, !dbg !52
  %49 = bitcast i8* %48 to i64*, !dbg !52
  %50 = load i64, i64* %49, align 8, !dbg !52
  %51 = add nsw i64 %46, %50, !dbg !52
  %52 = bitcast [16 x i64]* %"xa1$sd5_322" to i8*, !dbg !52
  %53 = getelementptr i8, i8* %52, i64 80, !dbg !52
  %54 = bitcast i8* %53 to i64*, !dbg !52
  %55 = load i64, i64* %54, align 8, !dbg !52
  %56 = sub nsw i64 %55, 1, !dbg !52
  %57 = sub nsw i64 %51, %56, !dbg !52
  store i64 %57, i64* %.g0000_415, align 8, !dbg !52
  %58 = bitcast i64* %.g0000_415 to i8*, !dbg !52
  %59 = bitcast i64* @.C361_MAIN_ to i8*, !dbg !52
  %60 = bitcast i64* @.C321_MAIN_ to i8*, !dbg !52
  %61 = bitcast double** %"xa1$p_318" to i8*, !dbg !52
  %62 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !52
  %63 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !52
  %64 = bitcast void (...)* @f90_alloc04a_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !52
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %64(i8* %58, i8* %59, i8* %60, i8* null, i8* %61, i8* null, i8* %62, i8* %63, i8* null, i64 0), !dbg !52
  %65 = bitcast [16 x i64]* %"xa1$sd5_322" to i8*, !dbg !52
  %66 = getelementptr i8, i8* %65, i64 64, !dbg !52
  %67 = bitcast double** %"xa1$p_318" to i8*, !dbg !52
  %68 = bitcast void (...)* @f90_ptrcp to void (i8*, i8*, ...)*, !dbg !52
  call void (i8*, i8*, ...) %68(i8* %66, i8* %67), !dbg !52
  %69 = bitcast [16 x i64]* %"xa2$sd7_326" to i8*, !dbg !53
  %70 = getelementptr i8, i8* %69, i64 80, !dbg !53
  %71 = bitcast i8* %70 to i64*, !dbg !53
  store i64 1, i64* %71, align 8, !dbg !53
  %72 = bitcast [16 x i64]* %"xa2$sd7_326" to i8*, !dbg !53
  %73 = getelementptr i8, i8* %72, i64 88, !dbg !53
  %74 = bitcast i8* %73 to i64*, !dbg !53
  store i64 2025, i64* %74, align 8, !dbg !53
  %75 = bitcast [16 x i64]* %"xa2$sd7_326" to i8*, !dbg !53
  %76 = getelementptr i8, i8* %75, i64 88, !dbg !53
  %77 = bitcast i8* %76 to i64*, !dbg !53
  %78 = load i64, i64* %77, align 8, !dbg !53
  %79 = sub nsw i64 %78, 1, !dbg !53
  %80 = bitcast [16 x i64]* %"xa2$sd7_326" to i8*, !dbg !53
  %81 = getelementptr i8, i8* %80, i64 80, !dbg !53
  %82 = bitcast i8* %81 to i64*, !dbg !53
  %83 = load i64, i64* %82, align 8, !dbg !53
  %84 = add nsw i64 %79, %83, !dbg !53
  store i64 %84, i64* %.g0000_415, align 8, !dbg !53
  %85 = bitcast [16 x i64]* %"xa2$sd7_326" to i8*, !dbg !53
  %86 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !53
  %87 = bitcast i64* @.C361_MAIN_ to i8*, !dbg !53
  %88 = bitcast i64* @.C321_MAIN_ to i8*, !dbg !53
  %89 = bitcast [16 x i64]* %"xa2$sd7_326" to i8*, !dbg !53
  %90 = getelementptr i8, i8* %89, i64 80, !dbg !53
  %91 = bitcast i64* %.g0000_415 to i8*, !dbg !53
  %92 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !53
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %92(i8* %85, i8* %86, i8* %87, i8* %88, i8* %90, i8* %91), !dbg !53
  %93 = bitcast [16 x i64]* %"xa2$sd7_326" to i8*, !dbg !53
  %94 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !53
  call void (i8*, i32, ...) %94(i8* %93, i32 28), !dbg !53
  %95 = bitcast [16 x i64]* %"xa2$sd7_326" to i8*, !dbg !53
  %96 = getelementptr i8, i8* %95, i64 88, !dbg !53
  %97 = bitcast i8* %96 to i64*, !dbg !53
  %98 = load i64, i64* %97, align 8, !dbg !53
  %99 = sub nsw i64 %98, 1, !dbg !53
  %100 = bitcast [16 x i64]* %"xa2$sd7_326" to i8*, !dbg !53
  %101 = getelementptr i8, i8* %100, i64 80, !dbg !53
  %102 = bitcast i8* %101 to i64*, !dbg !53
  %103 = load i64, i64* %102, align 8, !dbg !53
  %104 = add nsw i64 %99, %103, !dbg !53
  %105 = bitcast [16 x i64]* %"xa2$sd7_326" to i8*, !dbg !53
  %106 = getelementptr i8, i8* %105, i64 80, !dbg !53
  %107 = bitcast i8* %106 to i64*, !dbg !53
  %108 = load i64, i64* %107, align 8, !dbg !53
  %109 = sub nsw i64 %108, 1, !dbg !53
  %110 = sub nsw i64 %104, %109, !dbg !53
  store i64 %110, i64* %.g0000_415, align 8, !dbg !53
  %111 = bitcast i64* %.g0000_415 to i8*, !dbg !53
  %112 = bitcast i64* @.C361_MAIN_ to i8*, !dbg !53
  %113 = bitcast i64* @.C321_MAIN_ to i8*, !dbg !53
  %114 = bitcast double** %"xa2$p_325" to i8*, !dbg !53
  %115 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !53
  %116 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !53
  %117 = bitcast void (...)* @f90_alloc04a_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !53
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %117(i8* %111, i8* %112, i8* %113, i8* null, i8* %114, i8* null, i8* %115, i8* %116, i8* null, i64 0), !dbg !53
  %118 = bitcast [16 x i64]* %"xa2$sd7_326" to i8*, !dbg !53
  %119 = getelementptr i8, i8* %118, i64 64, !dbg !53
  %120 = bitcast double** %"xa2$p_325" to i8*, !dbg !53
  %121 = bitcast void (...)* @f90_ptrcp to void (i8*, i8*, ...)*, !dbg !53
  call void (i8*, i8*, ...) %121(i8* %119, i8* %120), !dbg !53
  %122 = bitcast [16 x i64]* %"xa1$sd5_322" to i8*, !dbg !54
  %123 = bitcast [16 x i64]* %"base$sd9_365" to i8*, !dbg !54
  %124 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !54
  %125 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !54
  %126 = bitcast i64* @.C329_MAIN_ to i8*, !dbg !54
  %127 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !54
  %128 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !54
  %129 = bitcast void (...)* @f90_sect1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !54
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, ...) %129(i8* %122, i8* %123, i8* %124, i8* %125, i8* %126, i8* %127, i8* %128), !dbg !54
  %130 = load double*, double** %"xa1$p_318", align 8, !dbg !54
  call void @llvm.dbg.value(metadata double* %130, metadata !51, metadata !DIExpression()), !dbg !35
  %131 = bitcast double* %130 to i8*, !dbg !54
  %132 = bitcast [16 x i64]* %"xa1$sd5_322" to i8*, !dbg !54
  %133 = bitcast %struct.BSS2* @.BSS2 to i8*, !dbg !54
  %134 = bitcast [16 x i64]* %"xa1$sd5_322" to i8*, !dbg !54
  %135 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !54
  %136 = bitcast i64 (...)* @fort_ptr_assn_i8 to i64 (i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !54
  %137 = call i64 (i8*, i8*, i8*, i8*, i8*, ...) %136(i8* %131, i8* %132, i8* %133, i8* %134, i8* %135), !dbg !54
  %138 = inttoptr i64 %137 to i8*, !dbg !54
  %139 = bitcast double** %"xa1$p_318" to i8**, !dbg !54
  store i8* %138, i8** %139, align 8, !dbg !54
  %140 = bitcast [16 x i64]* %"xa2$sd7_326" to i8*, !dbg !55
  %141 = bitcast [16 x i64]* %"base$sd9_365" to i8*, !dbg !55
  %142 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !55
  %143 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !55
  %144 = bitcast i64* @.C329_MAIN_ to i8*, !dbg !55
  %145 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !55
  %146 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !55
  %147 = bitcast void (...)* @f90_sect1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !55
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, ...) %147(i8* %140, i8* %141, i8* %142, i8* %143, i8* %144, i8* %145, i8* %146), !dbg !55
  %148 = load double*, double** %"xa2$p_325", align 8, !dbg !55
  call void @llvm.dbg.value(metadata double* %148, metadata !43, metadata !DIExpression()), !dbg !35
  %149 = bitcast double* %148 to i8*, !dbg !55
  %150 = bitcast [16 x i64]* %"xa2$sd7_326" to i8*, !dbg !55
  %151 = bitcast %struct.BSS2* @.BSS2 to i8*, !dbg !55
  %152 = bitcast [16 x i64]* %"xa2$sd7_326" to i8*, !dbg !55
  %153 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !55
  %154 = bitcast i64 (...)* @fort_ptr_assn_i8 to i64 (i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !55
  %155 = call i64 (i8*, i8*, i8*, i8*, i8*, ...) %154(i8* %149, i8* %150, i8* %151, i8* %152, i8* %153), !dbg !55
  %156 = inttoptr i64 %155 to i8*, !dbg !55
  %157 = bitcast double** %"xa2$p_325" to i8**, !dbg !55
  store i8* %156, i8** %157, align 8, !dbg !55
  %158 = bitcast %struct_drb008_0_* @_drb008_0_ to i8*, !dbg !56
  %159 = getelementptr i8, i8* %158, i64 720, !dbg !56
  %160 = bitcast i8* %159 to i32*, !dbg !56
  store i32 180, i32* %160, align 4, !dbg !56
  store i64 180, i64* %.dY0001_374, align 8, !dbg !57
  call void @llvm.dbg.declare(metadata i64* %"i$a_359", metadata !58, metadata !DIExpression()), !dbg !35
  store i64 1, i64* %"i$a_359", align 8, !dbg !57
  br label %L.LB2_372

L.LB2_372:                                        ; preds = %L.LB2_372, %L.LB2_403
  %161 = load i64, i64* %"i$a_359", align 8, !dbg !57
  call void @llvm.dbg.value(metadata i64 %161, metadata !58, metadata !DIExpression()), !dbg !35
  %162 = bitcast %struct.STATICS2* @.STATICS2 to i8*, !dbg !57
  %163 = getelementptr i8, i8* %162, i64 -4, !dbg !57
  %164 = bitcast i8* %163 to i32*, !dbg !57
  %165 = getelementptr i32, i32* %164, i64 %161, !dbg !57
  %166 = load i32, i32* %165, align 4, !dbg !57
  %167 = load i64, i64* %"i$a_359", align 8, !dbg !57
  call void @llvm.dbg.value(metadata i64 %167, metadata !58, metadata !DIExpression()), !dbg !35
  %168 = bitcast %struct_drb008_0_* @_drb008_0_ to i8*, !dbg !57
  %169 = getelementptr i8, i8* %168, i64 -4, !dbg !57
  %170 = bitcast i8* %169 to i32*, !dbg !57
  %171 = getelementptr i32, i32* %170, i64 %167, !dbg !57
  store i32 %166, i32* %171, align 4, !dbg !57
  %172 = load i64, i64* %"i$a_359", align 8, !dbg !57
  call void @llvm.dbg.value(metadata i64 %172, metadata !58, metadata !DIExpression()), !dbg !35
  %173 = add nsw i64 %172, 1, !dbg !57
  store i64 %173, i64* %"i$a_359", align 8, !dbg !57
  %174 = load i64, i64* %.dY0001_374, align 8, !dbg !57
  %175 = sub nsw i64 %174, 1, !dbg !57
  store i64 %175, i64* %.dY0001_374, align 8, !dbg !57
  %176 = load i64, i64* %.dY0001_374, align 8, !dbg !57
  %177 = icmp sgt i64 %176, 0, !dbg !57
  br i1 %177, label %L.LB2_372, label %L.LB2_503, !dbg !57

L.LB2_503:                                        ; preds = %L.LB2_372
  store i32 1505, i32* %.dY0002_377, align 4, !dbg !59
  call void @llvm.dbg.declare(metadata i32* %i_312, metadata !60, metadata !DIExpression()), !dbg !35
  store i32 521, i32* %i_312, align 4, !dbg !59
  br label %L.LB2_375

L.LB2_375:                                        ; preds = %L.LB2_375, %L.LB2_503
  %178 = load i32, i32* %i_312, align 4, !dbg !61
  call void @llvm.dbg.value(metadata i32 %178, metadata !60, metadata !DIExpression()), !dbg !35
  %179 = sitofp i32 %178 to float, !dbg !61
  %180 = fmul fast float %179, 5.000000e-01, !dbg !61
  %181 = fpext float %180 to double, !dbg !61
  %182 = load i32, i32* %i_312, align 4, !dbg !61
  call void @llvm.dbg.value(metadata i32 %182, metadata !60, metadata !DIExpression()), !dbg !35
  %183 = sext i32 %182 to i64, !dbg !61
  %184 = bitcast %struct.BSS2* @.BSS2 to i8*, !dbg !61
  %185 = getelementptr i8, i8* %184, i64 -8, !dbg !61
  %186 = bitcast i8* %185 to double*, !dbg !61
  %187 = getelementptr double, double* %186, i64 %183, !dbg !61
  store double %181, double* %187, align 8, !dbg !61
  %188 = load i32, i32* %i_312, align 4, !dbg !62
  call void @llvm.dbg.value(metadata i32 %188, metadata !60, metadata !DIExpression()), !dbg !35
  %189 = add nsw i32 %188, 1, !dbg !62
  store i32 %189, i32* %i_312, align 4, !dbg !62
  %190 = load i32, i32* %.dY0002_377, align 4, !dbg !62
  %191 = sub nsw i32 %190, 1, !dbg !62
  store i32 %191, i32* %.dY0002_377, align 4, !dbg !62
  %192 = load i32, i32* %.dY0002_377, align 4, !dbg !62
  %193 = icmp sgt i32 %192, 0, !dbg !62
  br i1 %193, label %L.LB2_375, label %L.LB2_504, !dbg !62

L.LB2_504:                                        ; preds = %L.LB2_375
  call void @llvm.dbg.declare(metadata i32* %idx1_313, metadata !63, metadata !DIExpression()), !dbg !35
  %194 = bitcast i32* %idx1_313 to i8*, !dbg !64
  %195 = bitcast %astruct.dt90* %.uplevelArgPack0001_449 to i8**, !dbg !64
  store i8* %194, i8** %195, align 8, !dbg !64
  call void @llvm.dbg.declare(metadata i32* %idx2_314, metadata !65, metadata !DIExpression()), !dbg !35
  %196 = bitcast i32* %idx2_314 to i8*, !dbg !64
  %197 = bitcast %astruct.dt90* %.uplevelArgPack0001_449 to i8*, !dbg !64
  %198 = getelementptr i8, i8* %197, i64 8, !dbg !64
  %199 = bitcast i8* %198 to i8**, !dbg !64
  store i8* %196, i8** %199, align 8, !dbg !64
  %200 = bitcast [16 x i64]* %"base$sd9_365" to i8*, !dbg !64
  %201 = bitcast %astruct.dt90* %.uplevelArgPack0001_449 to i8*, !dbg !64
  %202 = getelementptr i8, i8* %201, i64 24, !dbg !64
  %203 = bitcast i8* %202 to i8**, !dbg !64
  store i8* %200, i8** %203, align 8, !dbg !64
  br label %L.LB2_460, !dbg !64

L.LB2_460:                                        ; preds = %L.LB2_504
  %204 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L77_1_ to i64*, !dbg !64
  %205 = bitcast %astruct.dt90* %.uplevelArgPack0001_449 to i64*, !dbg !64
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %204, i64* %205), !dbg !64
  call void (...) @_mp_bcs_nest(), !dbg !66
  %206 = bitcast i32* @.C345_MAIN_ to i8*, !dbg !66
  %207 = bitcast [60 x i8]* @.C343_MAIN_ to i8*, !dbg !66
  %208 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !66
  call void (i8*, i8*, i64, ...) %208(i8* %206, i8* %207, i64 60), !dbg !66
  %209 = bitcast i32* @.C346_MAIN_ to i8*, !dbg !66
  %210 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !66
  %211 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !66
  %212 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !66
  %213 = call i32 (i8*, i8*, i8*, i8*, ...) %212(i8* %209, i8* null, i8* %210, i8* %211), !dbg !66
  call void @llvm.dbg.declare(metadata i32* %z__io_348, metadata !67, metadata !DIExpression()), !dbg !35
  store i32 %213, i32* %z__io_348, align 4, !dbg !66
  %214 = bitcast [10 x i8]* @.C349_MAIN_ to i8*, !dbg !66
  %215 = bitcast i32 (...)* @f90io_sc_ch_ldw to i32 (i8*, i32, i64, ...)*, !dbg !66
  %216 = call i32 (i8*, i32, i64, ...) %215(i8* %214, i32 14, i64 10), !dbg !66
  store i32 %216, i32* %z__io_348, align 4, !dbg !66
  %217 = bitcast %struct.BSS2* @.BSS2 to i8*, !dbg !66
  %218 = getelementptr i8, i8* %217, i64 7984, !dbg !66
  %219 = bitcast i8* %218 to double*, !dbg !66
  %220 = load double, double* %219, align 8, !dbg !66
  %221 = bitcast i32 (...)* @f90io_sc_d_ldw to i32 (double, i32, ...)*, !dbg !66
  %222 = call i32 (double, i32, ...) %221(double %220, i32 28), !dbg !66
  store i32 %222, i32* %z__io_348, align 4, !dbg !66
  %223 = bitcast [12 x i8]* @.C351_MAIN_ to i8*, !dbg !66
  %224 = bitcast i32 (...)* @f90io_sc_ch_ldw to i32 (i8*, i32, i64, ...)*, !dbg !66
  %225 = call i32 (i8*, i32, i64, ...) %224(i8* %223, i32 14, i64 12), !dbg !66
  store i32 %225, i32* %z__io_348, align 4, !dbg !66
  %226 = bitcast %struct.BSS2* @.BSS2 to i8*, !dbg !66
  %227 = getelementptr i8, i8* %226, i64 10272, !dbg !66
  %228 = bitcast i8* %227 to double*, !dbg !66
  %229 = load double, double* %228, align 8, !dbg !66
  %230 = bitcast i32 (...)* @f90io_sc_d_ldw to i32 (double, i32, ...)*, !dbg !66
  %231 = call i32 (double, i32, ...) %230(double %229, i32 28), !dbg !66
  store i32 %231, i32* %z__io_348, align 4, !dbg !66
  %232 = call i32 (...) @f90io_ldw_end(), !dbg !66
  store i32 %232, i32* %z__io_348, align 4, !dbg !66
  call void (...) @_mp_ecs_nest(), !dbg !66
  %233 = bitcast double** %"xa1$p_318" to i8**, !dbg !68
  store i8* null, i8** %233, align 8, !dbg !68
  %234 = bitcast [16 x i64]* %"xa1$sd5_322" to i64*, !dbg !68
  store i64 0, i64* %234, align 8, !dbg !68
  %235 = bitcast double** %"xa2$p_325" to i8**, !dbg !68
  store i8* null, i8** %235, align 8, !dbg !68
  %236 = bitcast [16 x i64]* %"xa2$sd7_326" to i64*, !dbg !68
  store i64 0, i64* %236, align 8, !dbg !68
  ret void, !dbg !41
}

define internal void @__nv_MAIN__F1L77_1_(i32* %__nv_MAIN__F1L77_1Arg0, i64* %__nv_MAIN__F1L77_1Arg1, i64* %__nv_MAIN__F1L77_1Arg2) #1 !dbg !20 {
L.entry:
  %__gtid___nv_MAIN__F1L77_1__522 = alloca i32, align 4
  %.i0000p_338 = alloca i32, align 4
  %i_337 = alloca i32, align 4
  %.du0003p_381 = alloca i32, align 4
  %.de0003p_382 = alloca i32, align 4
  %.di0003p_383 = alloca i32, align 4
  %.ds0003p_384 = alloca i32, align 4
  %.dl0003p_386 = alloca i32, align 4
  %.dl0003p.copy_516 = alloca i32, align 4
  %.de0003p.copy_517 = alloca i32, align 4
  %.ds0003p.copy_518 = alloca i32, align 4
  %.dX0003p_385 = alloca i32, align 4
  %.dY0003p_380 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L77_1Arg0, metadata !69, metadata !DIExpression()), !dbg !70
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L77_1Arg1, metadata !71, metadata !DIExpression()), !dbg !70
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L77_1Arg2, metadata !72, metadata !DIExpression()), !dbg !70
  call void @llvm.dbg.value(metadata i32 1, metadata !73, metadata !DIExpression()), !dbg !70
  call void @llvm.dbg.value(metadata i32 0, metadata !74, metadata !DIExpression()), !dbg !70
  call void @llvm.dbg.value(metadata i32 1, metadata !75, metadata !DIExpression()), !dbg !70
  call void @llvm.dbg.value(metadata i32 0, metadata !76, metadata !DIExpression()), !dbg !70
  call void @llvm.dbg.value(metadata i32 1, metadata !77, metadata !DIExpression()), !dbg !70
  call void @llvm.dbg.value(metadata i32 8, metadata !78, metadata !DIExpression()), !dbg !70
  %0 = load i32, i32* %__nv_MAIN__F1L77_1Arg0, align 4, !dbg !79
  store i32 %0, i32* %__gtid___nv_MAIN__F1L77_1__522, align 4, !dbg !79
  br label %L.LB3_508

L.LB3_508:                                        ; preds = %L.entry
  br label %L.LB3_336

L.LB3_336:                                        ; preds = %L.LB3_508
  store i32 0, i32* %.i0000p_338, align 4, !dbg !80
  call void @llvm.dbg.declare(metadata i32* %i_337, metadata !81, metadata !DIExpression()), !dbg !79
  store i32 1, i32* %i_337, align 4, !dbg !80
  %1 = bitcast %struct_drb008_0_* @_drb008_0_ to i8*, !dbg !80
  %2 = getelementptr i8, i8* %1, i64 720, !dbg !80
  %3 = bitcast i8* %2 to i32*, !dbg !80
  %4 = load i32, i32* %3, align 4, !dbg !80
  store i32 %4, i32* %.du0003p_381, align 4, !dbg !80
  %5 = bitcast %struct_drb008_0_* @_drb008_0_ to i8*, !dbg !80
  %6 = getelementptr i8, i8* %5, i64 720, !dbg !80
  %7 = bitcast i8* %6 to i32*, !dbg !80
  %8 = load i32, i32* %7, align 4, !dbg !80
  store i32 %8, i32* %.de0003p_382, align 4, !dbg !80
  store i32 1, i32* %.di0003p_383, align 4, !dbg !80
  %9 = load i32, i32* %.di0003p_383, align 4, !dbg !80
  store i32 %9, i32* %.ds0003p_384, align 4, !dbg !80
  store i32 1, i32* %.dl0003p_386, align 4, !dbg !80
  %10 = load i32, i32* %.dl0003p_386, align 4, !dbg !80
  store i32 %10, i32* %.dl0003p.copy_516, align 4, !dbg !80
  %11 = load i32, i32* %.de0003p_382, align 4, !dbg !80
  store i32 %11, i32* %.de0003p.copy_517, align 4, !dbg !80
  %12 = load i32, i32* %.ds0003p_384, align 4, !dbg !80
  store i32 %12, i32* %.ds0003p.copy_518, align 4, !dbg !80
  %13 = load i32, i32* %__gtid___nv_MAIN__F1L77_1__522, align 4, !dbg !80
  %14 = bitcast i32* %.i0000p_338 to i64*, !dbg !80
  %15 = bitcast i32* %.dl0003p.copy_516 to i64*, !dbg !80
  %16 = bitcast i32* %.de0003p.copy_517 to i64*, !dbg !80
  %17 = bitcast i32* %.ds0003p.copy_518 to i64*, !dbg !80
  %18 = load i32, i32* %.ds0003p.copy_518, align 4, !dbg !80
  call void @__kmpc_for_static_init_4(i64* null, i32 %13, i32 34, i64* %14, i64* %15, i64* %16, i64* %17, i32 %18, i32 1), !dbg !80
  %19 = load i32, i32* %.dl0003p.copy_516, align 4, !dbg !80
  store i32 %19, i32* %.dl0003p_386, align 4, !dbg !80
  %20 = load i32, i32* %.de0003p.copy_517, align 4, !dbg !80
  store i32 %20, i32* %.de0003p_382, align 4, !dbg !80
  %21 = load i32, i32* %.ds0003p.copy_518, align 4, !dbg !80
  store i32 %21, i32* %.ds0003p_384, align 4, !dbg !80
  %22 = load i32, i32* %.dl0003p_386, align 4, !dbg !80
  store i32 %22, i32* %i_337, align 4, !dbg !80
  %23 = load i32, i32* %i_337, align 4, !dbg !80
  call void @llvm.dbg.value(metadata i32 %23, metadata !81, metadata !DIExpression()), !dbg !79
  store i32 %23, i32* %.dX0003p_385, align 4, !dbg !80
  %24 = load i32, i32* %.dX0003p_385, align 4, !dbg !80
  %25 = load i32, i32* %.du0003p_381, align 4, !dbg !80
  %26 = icmp sgt i32 %24, %25, !dbg !80
  br i1 %26, label %L.LB3_379, label %L.LB3_545, !dbg !80

L.LB3_545:                                        ; preds = %L.LB3_336
  %27 = load i32, i32* %.dX0003p_385, align 4, !dbg !80
  store i32 %27, i32* %i_337, align 4, !dbg !80
  %28 = load i32, i32* %.di0003p_383, align 4, !dbg !80
  %29 = load i32, i32* %.de0003p_382, align 4, !dbg !80
  %30 = load i32, i32* %.dX0003p_385, align 4, !dbg !80
  %31 = sub nsw i32 %29, %30, !dbg !80
  %32 = add nsw i32 %28, %31, !dbg !80
  %33 = load i32, i32* %.di0003p_383, align 4, !dbg !80
  %34 = sdiv i32 %32, %33, !dbg !80
  store i32 %34, i32* %.dY0003p_380, align 4, !dbg !80
  %35 = load i32, i32* %.dY0003p_380, align 4, !dbg !80
  %36 = icmp sle i32 %35, 0, !dbg !80
  br i1 %36, label %L.LB3_389, label %L.LB3_388, !dbg !80

L.LB3_388:                                        ; preds = %L.LB3_388, %L.LB3_545
  %37 = load i32, i32* %i_337, align 4, !dbg !82
  call void @llvm.dbg.value(metadata i32 %37, metadata !81, metadata !DIExpression()), !dbg !79
  %38 = sext i32 %37 to i64, !dbg !82
  %39 = bitcast %struct_drb008_0_* @_drb008_0_ to i8*, !dbg !82
  %40 = getelementptr i8, i8* %39, i64 -4, !dbg !82
  %41 = bitcast i8* %40 to i32*, !dbg !82
  %42 = getelementptr i32, i32* %41, i64 %38, !dbg !82
  %43 = load i32, i32* %42, align 4, !dbg !82
  %44 = bitcast i64* %__nv_MAIN__F1L77_1Arg2 to i32**, !dbg !82
  %45 = load i32*, i32** %44, align 8, !dbg !82
  store i32 %43, i32* %45, align 4, !dbg !82
  %46 = load i32, i32* %i_337, align 4, !dbg !83
  call void @llvm.dbg.value(metadata i32 %46, metadata !81, metadata !DIExpression()), !dbg !79
  %47 = sext i32 %46 to i64, !dbg !83
  %48 = bitcast %struct_drb008_0_* @_drb008_0_ to i8*, !dbg !83
  %49 = getelementptr i8, i8* %48, i64 -4, !dbg !83
  %50 = bitcast i8* %49 to i32*, !dbg !83
  %51 = getelementptr i32, i32* %50, i64 %47, !dbg !83
  %52 = load i32, i32* %51, align 4, !dbg !83
  %53 = add nsw i32 %52, 12, !dbg !83
  %54 = bitcast i64* %__nv_MAIN__F1L77_1Arg2 to i8*, !dbg !83
  %55 = getelementptr i8, i8* %54, i64 8, !dbg !83
  %56 = bitcast i8* %55 to i32**, !dbg !83
  %57 = load i32*, i32** %56, align 8, !dbg !83
  store i32 %53, i32* %57, align 4, !dbg !83
  %58 = bitcast i64* %__nv_MAIN__F1L77_1Arg2 to i32**, !dbg !84
  %59 = load i32*, i32** %58, align 8, !dbg !84
  %60 = load i32, i32* %59, align 4, !dbg !84
  %61 = sext i32 %60 to i64, !dbg !84
  %62 = bitcast %struct.BSS2* @.BSS2 to i8*, !dbg !84
  %63 = getelementptr i8, i8* %62, i64 -8, !dbg !84
  %64 = bitcast i8* %63 to double*, !dbg !84
  %65 = getelementptr double, double* %64, i64 %61, !dbg !84
  %66 = load double, double* %65, align 8, !dbg !84
  %67 = fadd fast double %66, 1.000000e+00, !dbg !84
  %68 = bitcast i64* %__nv_MAIN__F1L77_1Arg2 to i32**, !dbg !84
  %69 = load i32*, i32** %68, align 8, !dbg !84
  %70 = load i32, i32* %69, align 4, !dbg !84
  %71 = sext i32 %70 to i64, !dbg !84
  %72 = bitcast %struct.BSS2* @.BSS2 to i8*, !dbg !84
  %73 = getelementptr i8, i8* %72, i64 -8, !dbg !84
  %74 = bitcast i8* %73 to double*, !dbg !84
  %75 = getelementptr double, double* %74, i64 %71, !dbg !84
  store double %67, double* %75, align 8, !dbg !84
  %76 = bitcast i64* %__nv_MAIN__F1L77_1Arg2 to i8*, !dbg !85
  %77 = getelementptr i8, i8* %76, i64 8, !dbg !85
  %78 = bitcast i8* %77 to i32**, !dbg !85
  %79 = load i32*, i32** %78, align 8, !dbg !85
  %80 = load i32, i32* %79, align 4, !dbg !85
  %81 = sext i32 %80 to i64, !dbg !85
  %82 = bitcast %struct.BSS2* @.BSS2 to i8*, !dbg !85
  %83 = getelementptr i8, i8* %82, i64 -8, !dbg !85
  %84 = bitcast i8* %83 to double*, !dbg !85
  %85 = getelementptr double, double* %84, i64 %81, !dbg !85
  %86 = load double, double* %85, align 8, !dbg !85
  %87 = fadd fast double %86, 3.000000e+00, !dbg !85
  %88 = bitcast i64* %__nv_MAIN__F1L77_1Arg2 to i8*, !dbg !85
  %89 = getelementptr i8, i8* %88, i64 8, !dbg !85
  %90 = bitcast i8* %89 to i32**, !dbg !85
  %91 = load i32*, i32** %90, align 8, !dbg !85
  %92 = load i32, i32* %91, align 4, !dbg !85
  %93 = sext i32 %92 to i64, !dbg !85
  %94 = bitcast %struct.BSS2* @.BSS2 to i8*, !dbg !85
  %95 = getelementptr i8, i8* %94, i64 -8, !dbg !85
  %96 = bitcast i8* %95 to double*, !dbg !85
  %97 = getelementptr double, double* %96, i64 %93, !dbg !85
  store double %87, double* %97, align 8, !dbg !85
  %98 = load i32, i32* %.di0003p_383, align 4, !dbg !79
  %99 = load i32, i32* %i_337, align 4, !dbg !79
  call void @llvm.dbg.value(metadata i32 %99, metadata !81, metadata !DIExpression()), !dbg !79
  %100 = add nsw i32 %98, %99, !dbg !79
  store i32 %100, i32* %i_337, align 4, !dbg !79
  %101 = load i32, i32* %.dY0003p_380, align 4, !dbg !79
  %102 = sub nsw i32 %101, 1, !dbg !79
  store i32 %102, i32* %.dY0003p_380, align 4, !dbg !79
  %103 = load i32, i32* %.dY0003p_380, align 4, !dbg !79
  %104 = icmp sgt i32 %103, 0, !dbg !79
  br i1 %104, label %L.LB3_388, label %L.LB3_389, !dbg !79

L.LB3_389:                                        ; preds = %L.LB3_388, %L.LB3_545
  br label %L.LB3_379

L.LB3_379:                                        ; preds = %L.LB3_389, %L.LB3_336
  %105 = load i32, i32* %__gtid___nv_MAIN__F1L77_1__522, align 4, !dbg !79
  call void @__kmpc_for_static_fini(i64* null, i32 %105), !dbg !79
  br label %L.LB3_341

L.LB3_341:                                        ; preds = %L.LB3_379
  ret void, !dbg !79
}

declare void @__kmpc_for_static_fini(i64*, i32) #1

declare void @__kmpc_for_static_init_4(i64*, i32, i32, i64*, i64*, i64*, i64*, i32, i32) #1

declare void @_mp_ecs_nest(...) #1

declare signext i32 @f90io_ldw_end(...) #1

declare signext i32 @f90io_sc_d_ldw(...) #1

declare signext i32 @f90io_sc_ch_ldw(...) #1

declare signext i32 @f90io_print_init(...) #1

declare void @f90io_src_info03a(...) #1

declare void @_mp_bcs_nest(...) #1

declare i64 @fort_ptr_assn_i8(...) #1

declare void @f90_sect1_i8(...) #1

declare void @f90_ptrcp(...) #1

declare void @f90_alloc04a_i8(...) #1

declare void @f90_set_intrin_type_i8(...) #1

declare void @f90_template1_i8(...) #1

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

!llvm.module.flags = !{!32, !33}
!llvm.dbg.cu = !{!4}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "base", scope: !2, file: !3, type: !24, isLocal: true, isDefinition: true)
!2 = distinct !DISubprogram(name: "drb008_indirectaccess4_orig_yes", scope: !4, file: !3, line: 35, type: !30, scopeLine: 35, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !4)
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB008-indirectaccess4-orig-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !6, imports: !28)
!5 = !{}
!6 = !{!7, !14, !0, !16, !18}
!7 = !DIGlobalVariableExpression(var: !8, expr: !DIExpression())
!8 = distinct !DIGlobalVariable(name: "indexset", scope: !9, file: !3, type: !10, isLocal: false, isDefinition: true)
!9 = !DIModule(scope: !4, name: "drb008")
!10 = !DICompositeType(tag: DW_TAG_array_type, baseType: !11, size: 5760, align: 32, elements: !12)
!11 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!12 = !{!13}
!13 = !DISubrange(count: 180, lowerBound: 1)
!14 = !DIGlobalVariableExpression(var: !15, expr: !DIExpression(DW_OP_plus_uconst, 720))
!15 = distinct !DIGlobalVariable(name: "n", scope: !9, file: !3, type: !11, isLocal: false, isDefinition: true)
!16 = !DIGlobalVariableExpression(var: !17, expr: !DIExpression())
!17 = distinct !DIGlobalVariable(name: "z_c_0", scope: !2, file: !3, type: !10, isLocal: true, isDefinition: true)
!18 = !DIGlobalVariableExpression(var: !19, expr: !DIExpression())
!19 = distinct !DIGlobalVariable(name: "base", scope: !20, file: !3, type: !24, isLocal: true, isDefinition: true)
!20 = distinct !DISubprogram(name: "__nv_MAIN__F1L77_1", scope: !4, file: !3, line: 77, type: !21, scopeLine: 77, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !4)
!21 = !DISubroutineType(types: !22)
!22 = !{null, !11, !23, !23}
!23 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!24 = !DICompositeType(tag: DW_TAG_array_type, baseType: !25, size: 129600, align: 64, elements: !26)
!25 = !DIBasicType(name: "double precision", size: 64, align: 64, encoding: DW_ATE_float)
!26 = !{!27}
!27 = !DISubrange(count: 2025, lowerBound: 1)
!28 = !{!29}
!29 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !2, entity: !9, file: !3, line: 35)
!30 = !DISubroutineType(cc: DW_CC_program, types: !31)
!31 = !{null}
!32 = !{i32 2, !"Dwarf Version", i32 4}
!33 = !{i32 2, !"Debug Info Version", i32 3}
!34 = !DILocalVariable(name: "omp_sched_static", scope: !2, file: !3, type: !11)
!35 = !DILocation(line: 0, scope: !2)
!36 = !DILocalVariable(name: "omp_proc_bind_false", scope: !2, file: !3, type: !11)
!37 = !DILocalVariable(name: "omp_proc_bind_true", scope: !2, file: !3, type: !11)
!38 = !DILocalVariable(name: "omp_lock_hint_none", scope: !2, file: !3, type: !11)
!39 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !2, file: !3, type: !11)
!40 = !DILocalVariable(name: "dp", scope: !2, file: !3, type: !11)
!41 = !DILocation(line: 89, column: 1, scope: !2)
!42 = !DILocation(line: 35, column: 1, scope: !2)
!43 = !DILocalVariable(name: "xa2", scope: !2, file: !3, type: !44)
!44 = !DICompositeType(tag: DW_TAG_array_type, baseType: !25, size: 64, align: 64, elements: !45)
!45 = !{!46}
!46 = !DISubrange(count: 0, lowerBound: 1)
!47 = !DILocalVariable(scope: !2, file: !3, type: !48, flags: DIFlagArtificial)
!48 = !DICompositeType(tag: DW_TAG_array_type, baseType: !23, size: 1024, align: 64, elements: !49)
!49 = !{!50}
!50 = !DISubrange(count: 16, lowerBound: 1)
!51 = !DILocalVariable(name: "xa1", scope: !2, file: !3, type: !44)
!52 = !DILocation(line: 45, column: 1, scope: !2)
!53 = !DILocation(line: 46, column: 1, scope: !2)
!54 = !DILocation(line: 48, column: 1, scope: !2)
!55 = !DILocation(line: 49, column: 1, scope: !2)
!56 = !DILocation(line: 51, column: 1, scope: !2)
!57 = !DILocation(line: 53, column: 1, scope: !2)
!58 = !DILocalVariable(scope: !2, file: !3, type: !23, flags: DIFlagArtificial)
!59 = !DILocation(line: 73, column: 1, scope: !2)
!60 = !DILocalVariable(name: "i", scope: !2, file: !3, type: !11)
!61 = !DILocation(line: 74, column: 1, scope: !2)
!62 = !DILocation(line: 75, column: 1, scope: !2)
!63 = !DILocalVariable(name: "idx1", scope: !2, file: !3, type: !11)
!64 = !DILocation(line: 77, column: 1, scope: !2)
!65 = !DILocalVariable(name: "idx2", scope: !2, file: !3, type: !11)
!66 = !DILocation(line: 86, column: 1, scope: !2)
!67 = !DILocalVariable(scope: !2, file: !3, type: !11, flags: DIFlagArtificial)
!68 = !DILocation(line: 88, column: 1, scope: !2)
!69 = !DILocalVariable(name: "__nv_MAIN__F1L77_1Arg0", arg: 1, scope: !20, file: !3, type: !11)
!70 = !DILocation(line: 0, scope: !20)
!71 = !DILocalVariable(name: "__nv_MAIN__F1L77_1Arg1", arg: 2, scope: !20, file: !3, type: !23)
!72 = !DILocalVariable(name: "__nv_MAIN__F1L77_1Arg2", arg: 3, scope: !20, file: !3, type: !23)
!73 = !DILocalVariable(name: "omp_sched_static", scope: !20, file: !3, type: !11)
!74 = !DILocalVariable(name: "omp_proc_bind_false", scope: !20, file: !3, type: !11)
!75 = !DILocalVariable(name: "omp_proc_bind_true", scope: !20, file: !3, type: !11)
!76 = !DILocalVariable(name: "omp_lock_hint_none", scope: !20, file: !3, type: !11)
!77 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !20, file: !3, type: !11)
!78 = !DILocalVariable(name: "dp", scope: !20, file: !3, type: !11)
!79 = !DILocation(line: 83, column: 1, scope: !20)
!80 = !DILocation(line: 78, column: 1, scope: !20)
!81 = !DILocalVariable(name: "i", scope: !20, file: !3, type: !11)
!82 = !DILocation(line: 79, column: 1, scope: !20)
!83 = !DILocation(line: 80, column: 1, scope: !20)
!84 = !DILocation(line: 81, column: 1, scope: !20)
!85 = !DILocation(line: 82, column: 1, scope: !20)

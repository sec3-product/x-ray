; ModuleID = '/tmp/DRB068-restrictpointer2-orig-no-02c6e7.ll'
source_filename = "/tmp/DRB068-restrictpointer2-orig-no-02c6e7.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.STATICS5 = type <{ [4 x i8] }>
%astruct.dt90 = type <{ i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8* }>

@.C341_drb068_foo_ = internal constant i32 6
@.C338_drb068_foo_ = internal constant [60 x i8] c"micro-benchmarks-fortran/DRB068-restrictpointer2-orig-no.f95"
@.C340_drb068_foo_ = internal constant i32 36
@.C336_drb068_foo_ = internal constant i32 1000
@.C335_drb068_foo_ = internal constant i64 500
@.C283_drb068_foo_ = internal constant i32 0
@.C285_drb068_foo_ = internal constant i32 1
@.C352_drb068_foo_ = internal constant i64 9
@.C305_drb068_foo_ = internal constant i32 25
@.C350_drb068_foo_ = internal constant i64 4
@.C349_drb068_foo_ = internal constant i64 25
@.C284_drb068_foo_ = internal constant i64 0
@.C308_drb068_foo_ = internal constant i64 12
@.C286_drb068_foo_ = internal constant i64 1
@.C307_drb068_foo_ = internal constant i64 11
@.C285___nv_drb068_foo__F1L29_1 = internal constant i32 1
@.C283___nv_drb068_foo__F1L29_1 = internal constant i32 0
@.STATICS5 = internal global %struct.STATICS5 <{ [4 x i8] c"\E8\03\00\00" }>, align 16, !dbg !0
@.C341_MAIN_ = internal constant i64 9
@.C306_MAIN_ = internal constant i32 25
@.C339_MAIN_ = internal constant i64 4
@.C338_MAIN_ = internal constant i64 25
@.C309_MAIN_ = internal constant i64 12
@.C308_MAIN_ = internal constant i64 11
@.C286_MAIN_ = internal constant i64 1
@.C284_MAIN_ = internal constant i64 0
@.C283_MAIN_ = internal constant i32 0

; Function Attrs: noinline
define float @drb068_() #0 {
.L.entry:
  ret float undef
}

define void @drb068_foo_(i32 %_V_n.arg, i64* %"a$p", i64* %"b$p", i64* %"c$p", i64* %"d$p", i64* %"a$sd1", i64* %"b$sd3", i64* %"c$sd5", i64* %"d$sd7") #1 !dbg !12 {
L.entry:
  %_V_n.addr = alloca i32, align 4
  %n_309 = alloca i32, align 4
  %__gtid_drb068_foo__452 = alloca i32, align 4
  %.g0000_403 = alloca i64, align 8
  %.dY0001_360 = alloca i32, align 4
  %i_315 = alloca i32, align 4
  %.uplevelArgPack0001_429 = alloca %astruct.dt90, align 16
  %z__io_343 = alloca i32, align 4
  %"drb068_foo___$eq_314" = alloca [16 x i8], align 4
  call void @llvm.dbg.declare(metadata i32* %_V_n.addr, metadata !23, metadata !DIExpression()), !dbg !24
  store i32 %_V_n.arg, i32* %_V_n.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %_V_n.addr, metadata !25, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.declare(metadata i64* %"a$p", metadata !26, metadata !DIExpression(DW_OP_deref)), !dbg !24
  call void @llvm.dbg.declare(metadata i64* %"b$p", metadata !27, metadata !DIExpression(DW_OP_deref)), !dbg !24
  call void @llvm.dbg.declare(metadata i64* %"c$p", metadata !28, metadata !DIExpression(DW_OP_deref)), !dbg !24
  call void @llvm.dbg.declare(metadata i64* %"d$p", metadata !29, metadata !DIExpression(DW_OP_deref)), !dbg !24
  call void @llvm.dbg.declare(metadata i64* %"a$sd1", metadata !30, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.declare(metadata i64* %"b$sd3", metadata !31, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.declare(metadata i64* %"c$sd5", metadata !32, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.declare(metadata i64* %"d$sd7", metadata !33, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.value(metadata i32 1, metadata !34, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.value(metadata i32 0, metadata !35, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.value(metadata i32 1, metadata !36, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.value(metadata i32 0, metadata !37, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.value(metadata i32 1, metadata !38, metadata !DIExpression()), !dbg !24
  %0 = load i32, i32* %_V_n.addr, align 4, !dbg !39
  call void @llvm.dbg.value(metadata i32 %0, metadata !23, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.declare(metadata i32* %n_309, metadata !40, metadata !DIExpression()), !dbg !24
  store i32 %0, i32* %n_309, align 4, !dbg !39
  %1 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !41
  store i32 %1, i32* %__gtid_drb068_foo__452, align 4, !dbg !41
  br label %L.LB2_385

L.LB2_385:                                        ; preds = %L.entry
  %2 = bitcast i64* %"a$sd1" to i8*, !dbg !42
  %3 = getelementptr i8, i8* %2, i64 80, !dbg !42
  %4 = bitcast i8* %3 to i64*, !dbg !42
  store i64 1, i64* %4, align 8, !dbg !42
  %5 = load i32, i32* %n_309, align 4, !dbg !42
  call void @llvm.dbg.value(metadata i32 %5, metadata !40, metadata !DIExpression()), !dbg !24
  %6 = sext i32 %5 to i64, !dbg !42
  %7 = bitcast i64* %"a$sd1" to i8*, !dbg !42
  %8 = getelementptr i8, i8* %7, i64 88, !dbg !42
  %9 = bitcast i8* %8 to i64*, !dbg !42
  store i64 %6, i64* %9, align 8, !dbg !42
  %10 = bitcast i64* %"a$sd1" to i8*, !dbg !42
  %11 = getelementptr i8, i8* %10, i64 88, !dbg !42
  %12 = bitcast i8* %11 to i64*, !dbg !42
  %13 = load i64, i64* %12, align 8, !dbg !42
  %14 = sub nsw i64 %13, 1, !dbg !42
  %15 = bitcast i64* %"a$sd1" to i8*, !dbg !42
  %16 = getelementptr i8, i8* %15, i64 80, !dbg !42
  %17 = bitcast i8* %16 to i64*, !dbg !42
  %18 = load i64, i64* %17, align 8, !dbg !42
  %19 = add nsw i64 %14, %18, !dbg !42
  store i64 %19, i64* %.g0000_403, align 8, !dbg !42
  %20 = bitcast i64* %"a$sd1" to i8*, !dbg !42
  %21 = bitcast i64* @.C284_drb068_foo_ to i8*, !dbg !42
  %22 = bitcast i64* @.C349_drb068_foo_ to i8*, !dbg !42
  %23 = bitcast i64* @.C350_drb068_foo_ to i8*, !dbg !42
  %24 = bitcast i64* %"a$sd1" to i8*, !dbg !42
  %25 = getelementptr i8, i8* %24, i64 80, !dbg !42
  %26 = bitcast i64* %.g0000_403 to i8*, !dbg !42
  %27 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !42
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %27(i8* %20, i8* %21, i8* %22, i8* %23, i8* %25, i8* %26), !dbg !42
  %28 = bitcast i64* %"a$sd1" to i8*, !dbg !42
  %29 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !42
  call void (i8*, i32, ...) %29(i8* %28, i32 25), !dbg !42
  %30 = bitcast i64* %"a$sd1" to i8*, !dbg !42
  %31 = getelementptr i8, i8* %30, i64 88, !dbg !42
  %32 = bitcast i8* %31 to i64*, !dbg !42
  %33 = load i64, i64* %32, align 8, !dbg !42
  %34 = sub nsw i64 %33, 1, !dbg !42
  %35 = bitcast i64* %"a$sd1" to i8*, !dbg !42
  %36 = getelementptr i8, i8* %35, i64 80, !dbg !42
  %37 = bitcast i8* %36 to i64*, !dbg !42
  %38 = load i64, i64* %37, align 8, !dbg !42
  %39 = add nsw i64 %34, %38, !dbg !42
  %40 = bitcast i64* %"a$sd1" to i8*, !dbg !42
  %41 = getelementptr i8, i8* %40, i64 80, !dbg !42
  %42 = bitcast i8* %41 to i64*, !dbg !42
  %43 = load i64, i64* %42, align 8, !dbg !42
  %44 = sub nsw i64 %43, 1, !dbg !42
  %45 = sub nsw i64 %39, %44, !dbg !42
  store i64 %45, i64* %.g0000_403, align 8, !dbg !42
  %46 = bitcast i64* %.g0000_403 to i8*, !dbg !42
  %47 = bitcast i64* @.C349_drb068_foo_ to i8*, !dbg !42
  %48 = bitcast i64* @.C350_drb068_foo_ to i8*, !dbg !42
  %49 = bitcast i64* %"a$p" to i8*, !dbg !42
  %50 = bitcast i64* @.C286_drb068_foo_ to i8*, !dbg !42
  %51 = bitcast i64* @.C284_drb068_foo_ to i8*, !dbg !42
  %52 = bitcast void (...)* @f90_alloc04a_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !42
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %52(i8* %46, i8* %47, i8* %48, i8* null, i8* %49, i8* null, i8* %50, i8* %51, i8* null, i64 0), !dbg !42
  %53 = bitcast i64* %"a$sd1" to i8*, !dbg !42
  %54 = getelementptr i8, i8* %53, i64 64, !dbg !42
  %55 = bitcast i64* %"a$p" to i8*, !dbg !42
  %56 = bitcast void (...)* @f90_ptrcp to void (i8*, i8*, ...)*, !dbg !42
  call void (i8*, i8*, ...) %56(i8* %54, i8* %55), !dbg !42
  %57 = bitcast i64* %"b$sd3" to i8*, !dbg !43
  %58 = getelementptr i8, i8* %57, i64 80, !dbg !43
  %59 = bitcast i8* %58 to i64*, !dbg !43
  store i64 1, i64* %59, align 8, !dbg !43
  %60 = load i32, i32* %n_309, align 4, !dbg !43
  call void @llvm.dbg.value(metadata i32 %60, metadata !40, metadata !DIExpression()), !dbg !24
  %61 = sext i32 %60 to i64, !dbg !43
  %62 = bitcast i64* %"b$sd3" to i8*, !dbg !43
  %63 = getelementptr i8, i8* %62, i64 88, !dbg !43
  %64 = bitcast i8* %63 to i64*, !dbg !43
  store i64 %61, i64* %64, align 8, !dbg !43
  %65 = bitcast i64* %"b$sd3" to i8*, !dbg !43
  %66 = getelementptr i8, i8* %65, i64 88, !dbg !43
  %67 = bitcast i8* %66 to i64*, !dbg !43
  %68 = load i64, i64* %67, align 8, !dbg !43
  %69 = sub nsw i64 %68, 1, !dbg !43
  %70 = bitcast i64* %"b$sd3" to i8*, !dbg !43
  %71 = getelementptr i8, i8* %70, i64 80, !dbg !43
  %72 = bitcast i8* %71 to i64*, !dbg !43
  %73 = load i64, i64* %72, align 8, !dbg !43
  %74 = add nsw i64 %69, %73, !dbg !43
  store i64 %74, i64* %.g0000_403, align 8, !dbg !43
  %75 = bitcast i64* %"b$sd3" to i8*, !dbg !43
  %76 = bitcast i64* @.C284_drb068_foo_ to i8*, !dbg !43
  %77 = bitcast i64* @.C349_drb068_foo_ to i8*, !dbg !43
  %78 = bitcast i64* @.C350_drb068_foo_ to i8*, !dbg !43
  %79 = bitcast i64* %"b$sd3" to i8*, !dbg !43
  %80 = getelementptr i8, i8* %79, i64 80, !dbg !43
  %81 = bitcast i64* %.g0000_403 to i8*, !dbg !43
  %82 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !43
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %82(i8* %75, i8* %76, i8* %77, i8* %78, i8* %80, i8* %81), !dbg !43
  %83 = bitcast i64* %"b$sd3" to i8*, !dbg !43
  %84 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !43
  call void (i8*, i32, ...) %84(i8* %83, i32 25), !dbg !43
  %85 = bitcast i64* %"b$sd3" to i8*, !dbg !43
  %86 = getelementptr i8, i8* %85, i64 88, !dbg !43
  %87 = bitcast i8* %86 to i64*, !dbg !43
  %88 = load i64, i64* %87, align 8, !dbg !43
  %89 = sub nsw i64 %88, 1, !dbg !43
  %90 = bitcast i64* %"b$sd3" to i8*, !dbg !43
  %91 = getelementptr i8, i8* %90, i64 80, !dbg !43
  %92 = bitcast i8* %91 to i64*, !dbg !43
  %93 = load i64, i64* %92, align 8, !dbg !43
  %94 = add nsw i64 %89, %93, !dbg !43
  %95 = bitcast i64* %"b$sd3" to i8*, !dbg !43
  %96 = getelementptr i8, i8* %95, i64 80, !dbg !43
  %97 = bitcast i8* %96 to i64*, !dbg !43
  %98 = load i64, i64* %97, align 8, !dbg !43
  %99 = sub nsw i64 %98, 1, !dbg !43
  %100 = sub nsw i64 %94, %99, !dbg !43
  store i64 %100, i64* %.g0000_403, align 8, !dbg !43
  %101 = bitcast i64* %.g0000_403 to i8*, !dbg !43
  %102 = bitcast i64* @.C349_drb068_foo_ to i8*, !dbg !43
  %103 = bitcast i64* @.C350_drb068_foo_ to i8*, !dbg !43
  %104 = bitcast i64* %"b$p" to i8*, !dbg !43
  %105 = bitcast i64* @.C286_drb068_foo_ to i8*, !dbg !43
  %106 = bitcast i64* @.C284_drb068_foo_ to i8*, !dbg !43
  %107 = bitcast void (...)* @f90_alloc04a_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !43
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %107(i8* %101, i8* %102, i8* %103, i8* null, i8* %104, i8* null, i8* %105, i8* %106, i8* null, i64 0), !dbg !43
  %108 = bitcast i64* %"b$sd3" to i8*, !dbg !43
  %109 = getelementptr i8, i8* %108, i64 64, !dbg !43
  %110 = bitcast i64* %"b$p" to i8*, !dbg !43
  %111 = bitcast void (...)* @f90_ptrcp to void (i8*, i8*, ...)*, !dbg !43
  call void (i8*, i8*, ...) %111(i8* %109, i8* %110), !dbg !43
  %112 = bitcast i64* %"c$sd5" to i8*, !dbg !44
  %113 = getelementptr i8, i8* %112, i64 80, !dbg !44
  %114 = bitcast i8* %113 to i64*, !dbg !44
  store i64 1, i64* %114, align 8, !dbg !44
  %115 = load i32, i32* %n_309, align 4, !dbg !44
  call void @llvm.dbg.value(metadata i32 %115, metadata !40, metadata !DIExpression()), !dbg !24
  %116 = sext i32 %115 to i64, !dbg !44
  %117 = bitcast i64* %"c$sd5" to i8*, !dbg !44
  %118 = getelementptr i8, i8* %117, i64 88, !dbg !44
  %119 = bitcast i8* %118 to i64*, !dbg !44
  store i64 %116, i64* %119, align 8, !dbg !44
  %120 = bitcast i64* %"c$sd5" to i8*, !dbg !44
  %121 = getelementptr i8, i8* %120, i64 88, !dbg !44
  %122 = bitcast i8* %121 to i64*, !dbg !44
  %123 = load i64, i64* %122, align 8, !dbg !44
  %124 = sub nsw i64 %123, 1, !dbg !44
  %125 = bitcast i64* %"c$sd5" to i8*, !dbg !44
  %126 = getelementptr i8, i8* %125, i64 80, !dbg !44
  %127 = bitcast i8* %126 to i64*, !dbg !44
  %128 = load i64, i64* %127, align 8, !dbg !44
  %129 = add nsw i64 %124, %128, !dbg !44
  store i64 %129, i64* %.g0000_403, align 8, !dbg !44
  %130 = bitcast i64* %"c$sd5" to i8*, !dbg !44
  %131 = bitcast i64* @.C284_drb068_foo_ to i8*, !dbg !44
  %132 = bitcast i64* @.C349_drb068_foo_ to i8*, !dbg !44
  %133 = bitcast i64* @.C350_drb068_foo_ to i8*, !dbg !44
  %134 = bitcast i64* %"c$sd5" to i8*, !dbg !44
  %135 = getelementptr i8, i8* %134, i64 80, !dbg !44
  %136 = bitcast i64* %.g0000_403 to i8*, !dbg !44
  %137 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !44
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %137(i8* %130, i8* %131, i8* %132, i8* %133, i8* %135, i8* %136), !dbg !44
  %138 = bitcast i64* %"c$sd5" to i8*, !dbg !44
  %139 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !44
  call void (i8*, i32, ...) %139(i8* %138, i32 25), !dbg !44
  %140 = bitcast i64* %"c$sd5" to i8*, !dbg !44
  %141 = getelementptr i8, i8* %140, i64 88, !dbg !44
  %142 = bitcast i8* %141 to i64*, !dbg !44
  %143 = load i64, i64* %142, align 8, !dbg !44
  %144 = sub nsw i64 %143, 1, !dbg !44
  %145 = bitcast i64* %"c$sd5" to i8*, !dbg !44
  %146 = getelementptr i8, i8* %145, i64 80, !dbg !44
  %147 = bitcast i8* %146 to i64*, !dbg !44
  %148 = load i64, i64* %147, align 8, !dbg !44
  %149 = add nsw i64 %144, %148, !dbg !44
  %150 = bitcast i64* %"c$sd5" to i8*, !dbg !44
  %151 = getelementptr i8, i8* %150, i64 80, !dbg !44
  %152 = bitcast i8* %151 to i64*, !dbg !44
  %153 = load i64, i64* %152, align 8, !dbg !44
  %154 = sub nsw i64 %153, 1, !dbg !44
  %155 = sub nsw i64 %149, %154, !dbg !44
  store i64 %155, i64* %.g0000_403, align 8, !dbg !44
  %156 = bitcast i64* %.g0000_403 to i8*, !dbg !44
  %157 = bitcast i64* @.C349_drb068_foo_ to i8*, !dbg !44
  %158 = bitcast i64* @.C350_drb068_foo_ to i8*, !dbg !44
  %159 = bitcast i64* %"c$p" to i8*, !dbg !44
  %160 = bitcast i64* @.C286_drb068_foo_ to i8*, !dbg !44
  %161 = bitcast i64* @.C284_drb068_foo_ to i8*, !dbg !44
  %162 = bitcast void (...)* @f90_alloc04a_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !44
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %162(i8* %156, i8* %157, i8* %158, i8* null, i8* %159, i8* null, i8* %160, i8* %161, i8* null, i64 0), !dbg !44
  %163 = bitcast i64* %"c$sd5" to i8*, !dbg !44
  %164 = getelementptr i8, i8* %163, i64 64, !dbg !44
  %165 = bitcast i64* %"c$p" to i8*, !dbg !44
  %166 = bitcast void (...)* @f90_ptrcp to void (i8*, i8*, ...)*, !dbg !44
  call void (i8*, i8*, ...) %166(i8* %164, i8* %165), !dbg !44
  %167 = bitcast i64* %"d$sd7" to i8*, !dbg !45
  %168 = getelementptr i8, i8* %167, i64 80, !dbg !45
  %169 = bitcast i8* %168 to i64*, !dbg !45
  store i64 1, i64* %169, align 8, !dbg !45
  %170 = load i32, i32* %n_309, align 4, !dbg !45
  call void @llvm.dbg.value(metadata i32 %170, metadata !40, metadata !DIExpression()), !dbg !24
  %171 = sext i32 %170 to i64, !dbg !45
  %172 = bitcast i64* %"d$sd7" to i8*, !dbg !45
  %173 = getelementptr i8, i8* %172, i64 88, !dbg !45
  %174 = bitcast i8* %173 to i64*, !dbg !45
  store i64 %171, i64* %174, align 8, !dbg !45
  %175 = bitcast i64* %"d$sd7" to i8*, !dbg !45
  %176 = getelementptr i8, i8* %175, i64 88, !dbg !45
  %177 = bitcast i8* %176 to i64*, !dbg !45
  %178 = load i64, i64* %177, align 8, !dbg !45
  %179 = sub nsw i64 %178, 1, !dbg !45
  %180 = bitcast i64* %"d$sd7" to i8*, !dbg !45
  %181 = getelementptr i8, i8* %180, i64 80, !dbg !45
  %182 = bitcast i8* %181 to i64*, !dbg !45
  %183 = load i64, i64* %182, align 8, !dbg !45
  %184 = add nsw i64 %179, %183, !dbg !45
  store i64 %184, i64* %.g0000_403, align 8, !dbg !45
  %185 = bitcast i64* %"d$sd7" to i8*, !dbg !45
  %186 = bitcast i64* @.C284_drb068_foo_ to i8*, !dbg !45
  %187 = bitcast i64* @.C349_drb068_foo_ to i8*, !dbg !45
  %188 = bitcast i64* @.C350_drb068_foo_ to i8*, !dbg !45
  %189 = bitcast i64* %"d$sd7" to i8*, !dbg !45
  %190 = getelementptr i8, i8* %189, i64 80, !dbg !45
  %191 = bitcast i64* %.g0000_403 to i8*, !dbg !45
  %192 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !45
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %192(i8* %185, i8* %186, i8* %187, i8* %188, i8* %190, i8* %191), !dbg !45
  %193 = bitcast i64* %"d$sd7" to i8*, !dbg !45
  %194 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !45
  call void (i8*, i32, ...) %194(i8* %193, i32 25), !dbg !45
  %195 = bitcast i64* %"d$sd7" to i8*, !dbg !45
  %196 = getelementptr i8, i8* %195, i64 88, !dbg !45
  %197 = bitcast i8* %196 to i64*, !dbg !45
  %198 = load i64, i64* %197, align 8, !dbg !45
  %199 = sub nsw i64 %198, 1, !dbg !45
  %200 = bitcast i64* %"d$sd7" to i8*, !dbg !45
  %201 = getelementptr i8, i8* %200, i64 80, !dbg !45
  %202 = bitcast i8* %201 to i64*, !dbg !45
  %203 = load i64, i64* %202, align 8, !dbg !45
  %204 = add nsw i64 %199, %203, !dbg !45
  %205 = bitcast i64* %"d$sd7" to i8*, !dbg !45
  %206 = getelementptr i8, i8* %205, i64 80, !dbg !45
  %207 = bitcast i8* %206 to i64*, !dbg !45
  %208 = load i64, i64* %207, align 8, !dbg !45
  %209 = sub nsw i64 %208, 1, !dbg !45
  %210 = sub nsw i64 %204, %209, !dbg !45
  store i64 %210, i64* %.g0000_403, align 8, !dbg !45
  %211 = bitcast i64* %.g0000_403 to i8*, !dbg !45
  %212 = bitcast i64* @.C349_drb068_foo_ to i8*, !dbg !45
  %213 = bitcast i64* @.C350_drb068_foo_ to i8*, !dbg !45
  %214 = bitcast i64* %"d$p" to i8*, !dbg !45
  %215 = bitcast i64* @.C286_drb068_foo_ to i8*, !dbg !45
  %216 = bitcast i64* @.C284_drb068_foo_ to i8*, !dbg !45
  %217 = bitcast void (...)* @f90_alloc04a_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !45
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %217(i8* %211, i8* %212, i8* %213, i8* null, i8* %214, i8* null, i8* %215, i8* %216, i8* null, i64 0), !dbg !45
  %218 = bitcast i64* %"d$sd7" to i8*, !dbg !45
  %219 = getelementptr i8, i8* %218, i64 64, !dbg !45
  %220 = bitcast i64* %"d$p" to i8*, !dbg !45
  %221 = bitcast void (...)* @f90_ptrcp to void (i8*, i8*, ...)*, !dbg !45
  call void (i8*, i8*, ...) %221(i8* %219, i8* %220), !dbg !45
  %222 = load i32, i32* %n_309, align 4, !dbg !46
  call void @llvm.dbg.value(metadata i32 %222, metadata !40, metadata !DIExpression()), !dbg !24
  store i32 %222, i32* %.dY0001_360, align 4, !dbg !46
  call void @llvm.dbg.declare(metadata i32* %i_315, metadata !47, metadata !DIExpression()), !dbg !24
  store i32 1, i32* %i_315, align 4, !dbg !46
  %223 = load i32, i32* %.dY0001_360, align 4, !dbg !46
  %224 = icmp sle i32 %223, 0, !dbg !46
  br i1 %224, label %L.LB2_359, label %L.LB2_358, !dbg !46

L.LB2_358:                                        ; preds = %L.LB2_358, %L.LB2_385
  %225 = load i32, i32* %i_315, align 4, !dbg !48
  call void @llvm.dbg.value(metadata i32 %225, metadata !47, metadata !DIExpression()), !dbg !24
  %226 = bitcast i64* %"b$p" to i8**, !dbg !48
  %227 = load i8*, i8** %226, align 8, !dbg !48
  %228 = bitcast i64* %"b$sd3" to i8*, !dbg !48
  %229 = getelementptr i8, i8* %228, i64 24, !dbg !48
  %230 = bitcast i8* %229 to i64*, !dbg !48
  %231 = load i64, i64* %230, align 8, !dbg !48
  %232 = load i32, i32* %i_315, align 4, !dbg !48
  call void @llvm.dbg.value(metadata i32 %232, metadata !47, metadata !DIExpression()), !dbg !24
  %233 = sext i32 %232 to i64, !dbg !48
  %234 = bitcast i64* %"b$sd3" to i8*, !dbg !48
  %235 = getelementptr i8, i8* %234, i64 112, !dbg !48
  %236 = bitcast i8* %235 to i64*, !dbg !48
  %237 = load i64, i64* %236, align 8, !dbg !48
  %238 = mul nsw i64 %233, %237, !dbg !48
  %239 = bitcast i64* %"b$sd3" to i8*, !dbg !48
  %240 = getelementptr i8, i8* %239, i64 56, !dbg !48
  %241 = bitcast i8* %240 to i64*, !dbg !48
  %242 = load i64, i64* %241, align 8, !dbg !48
  %243 = add nsw i64 %238, %242, !dbg !48
  %244 = sub nsw i64 %243, 1, !dbg !48
  %245 = mul nsw i64 %231, %244, !dbg !48
  %246 = getelementptr i8, i8* %227, i64 %245, !dbg !48
  %247 = bitcast i8* %246 to i32*, !dbg !48
  store i32 %225, i32* %247, align 4, !dbg !48
  %248 = load i32, i32* %i_315, align 4, !dbg !49
  call void @llvm.dbg.value(metadata i32 %248, metadata !47, metadata !DIExpression()), !dbg !24
  %249 = bitcast i64* %"c$p" to i8**, !dbg !49
  %250 = load i8*, i8** %249, align 8, !dbg !49
  %251 = bitcast i64* %"c$sd5" to i8*, !dbg !49
  %252 = getelementptr i8, i8* %251, i64 24, !dbg !49
  %253 = bitcast i8* %252 to i64*, !dbg !49
  %254 = load i64, i64* %253, align 8, !dbg !49
  %255 = load i32, i32* %i_315, align 4, !dbg !49
  call void @llvm.dbg.value(metadata i32 %255, metadata !47, metadata !DIExpression()), !dbg !24
  %256 = sext i32 %255 to i64, !dbg !49
  %257 = bitcast i64* %"c$sd5" to i8*, !dbg !49
  %258 = getelementptr i8, i8* %257, i64 112, !dbg !49
  %259 = bitcast i8* %258 to i64*, !dbg !49
  %260 = load i64, i64* %259, align 8, !dbg !49
  %261 = mul nsw i64 %256, %260, !dbg !49
  %262 = bitcast i64* %"c$sd5" to i8*, !dbg !49
  %263 = getelementptr i8, i8* %262, i64 56, !dbg !49
  %264 = bitcast i8* %263 to i64*, !dbg !49
  %265 = load i64, i64* %264, align 8, !dbg !49
  %266 = add nsw i64 %261, %265, !dbg !49
  %267 = sub nsw i64 %266, 1, !dbg !49
  %268 = mul nsw i64 %254, %267, !dbg !49
  %269 = getelementptr i8, i8* %250, i64 %268, !dbg !49
  %270 = bitcast i8* %269 to i32*, !dbg !49
  store i32 %248, i32* %270, align 4, !dbg !49
  %271 = load i32, i32* %i_315, align 4, !dbg !50
  call void @llvm.dbg.value(metadata i32 %271, metadata !47, metadata !DIExpression()), !dbg !24
  %272 = add nsw i32 %271, 1, !dbg !50
  store i32 %272, i32* %i_315, align 4, !dbg !50
  %273 = load i32, i32* %.dY0001_360, align 4, !dbg !50
  %274 = sub nsw i32 %273, 1, !dbg !50
  store i32 %274, i32* %.dY0001_360, align 4, !dbg !50
  %275 = load i32, i32* %.dY0001_360, align 4, !dbg !50
  %276 = icmp sgt i32 %275, 0, !dbg !50
  br i1 %276, label %L.LB2_358, label %L.LB2_359, !dbg !50

L.LB2_359:                                        ; preds = %L.LB2_358, %L.LB2_385
  %277 = bitcast i32* %n_309 to i8*, !dbg !51
  %278 = bitcast %astruct.dt90* %.uplevelArgPack0001_429 to i8**, !dbg !51
  store i8* %277, i8** %278, align 8, !dbg !51
  %279 = bitcast i64* %"a$p" to i8*, !dbg !51
  %280 = bitcast %astruct.dt90* %.uplevelArgPack0001_429 to i8*, !dbg !51
  %281 = getelementptr i8, i8* %280, i64 8, !dbg !51
  %282 = bitcast i8* %281 to i8**, !dbg !51
  store i8* %279, i8** %282, align 8, !dbg !51
  %283 = bitcast i64* %"a$sd1" to i8*, !dbg !51
  %284 = bitcast %astruct.dt90* %.uplevelArgPack0001_429 to i8*, !dbg !51
  %285 = getelementptr i8, i8* %284, i64 16, !dbg !51
  %286 = bitcast i8* %285 to i8**, !dbg !51
  store i8* %283, i8** %286, align 8, !dbg !51
  %287 = bitcast i64* %"a$p" to i8*, !dbg !51
  %288 = bitcast %astruct.dt90* %.uplevelArgPack0001_429 to i8*, !dbg !51
  %289 = getelementptr i8, i8* %288, i64 24, !dbg !51
  %290 = bitcast i8* %289 to i8**, !dbg !51
  store i8* %287, i8** %290, align 8, !dbg !51
  %291 = bitcast i64* %"b$p" to i8*, !dbg !51
  %292 = bitcast %astruct.dt90* %.uplevelArgPack0001_429 to i8*, !dbg !51
  %293 = getelementptr i8, i8* %292, i64 32, !dbg !51
  %294 = bitcast i8* %293 to i8**, !dbg !51
  store i8* %291, i8** %294, align 8, !dbg !51
  %295 = bitcast i64* %"b$sd3" to i8*, !dbg !51
  %296 = bitcast %astruct.dt90* %.uplevelArgPack0001_429 to i8*, !dbg !51
  %297 = getelementptr i8, i8* %296, i64 40, !dbg !51
  %298 = bitcast i8* %297 to i8**, !dbg !51
  store i8* %295, i8** %298, align 8, !dbg !51
  %299 = bitcast i64* %"b$p" to i8*, !dbg !51
  %300 = bitcast %astruct.dt90* %.uplevelArgPack0001_429 to i8*, !dbg !51
  %301 = getelementptr i8, i8* %300, i64 48, !dbg !51
  %302 = bitcast i8* %301 to i8**, !dbg !51
  store i8* %299, i8** %302, align 8, !dbg !51
  %303 = bitcast i64* %"c$p" to i8*, !dbg !51
  %304 = bitcast %astruct.dt90* %.uplevelArgPack0001_429 to i8*, !dbg !51
  %305 = getelementptr i8, i8* %304, i64 56, !dbg !51
  %306 = bitcast i8* %305 to i8**, !dbg !51
  store i8* %303, i8** %306, align 8, !dbg !51
  %307 = bitcast i64* %"c$sd5" to i8*, !dbg !51
  %308 = bitcast %astruct.dt90* %.uplevelArgPack0001_429 to i8*, !dbg !51
  %309 = getelementptr i8, i8* %308, i64 64, !dbg !51
  %310 = bitcast i8* %309 to i8**, !dbg !51
  store i8* %307, i8** %310, align 8, !dbg !51
  %311 = bitcast i64* %"c$p" to i8*, !dbg !51
  %312 = bitcast %astruct.dt90* %.uplevelArgPack0001_429 to i8*, !dbg !51
  %313 = getelementptr i8, i8* %312, i64 72, !dbg !51
  %314 = bitcast i8* %313 to i8**, !dbg !51
  store i8* %311, i8** %314, align 8, !dbg !51
  br label %L.LB2_450, !dbg !51

L.LB2_450:                                        ; preds = %L.LB2_359
  %315 = bitcast void (i32*, i64*, i64*)* @__nv_drb068_foo__F1L29_1_ to i64*, !dbg !51
  %316 = bitcast %astruct.dt90* %.uplevelArgPack0001_429 to i64*, !dbg !51
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %315, i64* %316), !dbg !51
  %317 = bitcast i64* %"a$p" to i8**, !dbg !52
  %318 = load i8*, i8** %317, align 8, !dbg !52
  %319 = bitcast i64* %"a$sd1" to i8*, !dbg !52
  %320 = getelementptr i8, i8* %319, i64 24, !dbg !52
  %321 = bitcast i8* %320 to i64*, !dbg !52
  %322 = load i64, i64* %321, align 8, !dbg !52
  %323 = bitcast i64* %"a$sd1" to i8*, !dbg !52
  %324 = getelementptr i8, i8* %323, i64 112, !dbg !52
  %325 = bitcast i8* %324 to i64*, !dbg !52
  %326 = load i64, i64* %325, align 8, !dbg !52
  %327 = mul nsw i64 %326, 500, !dbg !52
  %328 = bitcast i64* %"a$sd1" to i8*, !dbg !52
  %329 = getelementptr i8, i8* %328, i64 56, !dbg !52
  %330 = bitcast i8* %329 to i64*, !dbg !52
  %331 = load i64, i64* %330, align 8, !dbg !52
  %332 = add nsw i64 %327, %331, !dbg !52
  %333 = sub nsw i64 %332, 1, !dbg !52
  %334 = mul nsw i64 %322, %333, !dbg !52
  %335 = getelementptr i8, i8* %318, i64 %334, !dbg !52
  %336 = bitcast i8* %335 to i32*, !dbg !52
  %337 = load i32, i32* %336, align 4, !dbg !52
  %338 = icmp eq i32 %337, 1000, !dbg !52
  br i1 %338, label %L.LB2_373, label %L.LB2_478, !dbg !52

L.LB2_478:                                        ; preds = %L.LB2_450
  call void (...) @_mp_bcs_nest(), !dbg !53
  %339 = bitcast i32* @.C340_drb068_foo_ to i8*, !dbg !53
  %340 = bitcast [60 x i8]* @.C338_drb068_foo_ to i8*, !dbg !53
  %341 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !53
  call void (i8*, i8*, i64, ...) %341(i8* %339, i8* %340, i64 60), !dbg !53
  %342 = bitcast i32* @.C341_drb068_foo_ to i8*, !dbg !53
  %343 = bitcast i32* @.C283_drb068_foo_ to i8*, !dbg !53
  %344 = bitcast i32* @.C283_drb068_foo_ to i8*, !dbg !53
  %345 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !53
  %346 = call i32 (i8*, i8*, i8*, i8*, ...) %345(i8* %342, i8* null, i8* %343, i8* %344), !dbg !53
  call void @llvm.dbg.declare(metadata i32* %z__io_343, metadata !54, metadata !DIExpression()), !dbg !24
  store i32 %346, i32* %z__io_343, align 4, !dbg !53
  %347 = bitcast i64* %"a$p" to i8**, !dbg !53
  %348 = load i8*, i8** %347, align 8, !dbg !53
  %349 = bitcast i64* %"a$sd1" to i8*, !dbg !53
  %350 = getelementptr i8, i8* %349, i64 24, !dbg !53
  %351 = bitcast i8* %350 to i64*, !dbg !53
  %352 = load i64, i64* %351, align 8, !dbg !53
  %353 = bitcast i64* %"a$sd1" to i8*, !dbg !53
  %354 = getelementptr i8, i8* %353, i64 112, !dbg !53
  %355 = bitcast i8* %354 to i64*, !dbg !53
  %356 = load i64, i64* %355, align 8, !dbg !53
  %357 = mul nsw i64 %356, 500, !dbg !53
  %358 = bitcast i64* %"a$sd1" to i8*, !dbg !53
  %359 = getelementptr i8, i8* %358, i64 56, !dbg !53
  %360 = bitcast i8* %359 to i64*, !dbg !53
  %361 = load i64, i64* %360, align 8, !dbg !53
  %362 = add nsw i64 %357, %361, !dbg !53
  %363 = sub nsw i64 %362, 1, !dbg !53
  %364 = mul nsw i64 %352, %363, !dbg !53
  %365 = getelementptr i8, i8* %348, i64 %364, !dbg !53
  %366 = bitcast i8* %365 to i32*, !dbg !53
  %367 = load i32, i32* %366, align 4, !dbg !53
  %368 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !53
  %369 = call i32 (i32, i32, ...) %368(i32 %367, i32 25), !dbg !53
  store i32 %369, i32* %z__io_343, align 4, !dbg !53
  %370 = call i32 (...) @f90io_ldw_end(), !dbg !53
  store i32 %370, i32* %z__io_343, align 4, !dbg !53
  call void (...) @_mp_ecs_nest(), !dbg !53
  br label %L.LB2_373

L.LB2_373:                                        ; preds = %L.LB2_478, %L.LB2_450
  %371 = bitcast i64* %"a$p" to i8**, !dbg !55
  store i8* null, i8** %371, align 8, !dbg !55
  store i64 0, i64* %"a$sd1", align 8, !dbg !55
  %372 = bitcast i64* %"b$p" to i8**, !dbg !55
  store i8* null, i8** %372, align 8, !dbg !55
  store i64 0, i64* %"b$sd3", align 8, !dbg !55
  %373 = bitcast i64* %"c$p" to i8**, !dbg !55
  store i8* null, i8** %373, align 8, !dbg !55
  store i64 0, i64* %"c$sd5", align 8, !dbg !55
  %374 = bitcast i64* %"d$p" to i8**, !dbg !55
  store i8* null, i8** %374, align 8, !dbg !55
  store i64 0, i64* %"d$sd7", align 8, !dbg !55
  ret void, !dbg !41
}

define internal void @__nv_drb068_foo__F1L29_1_(i32* %__nv_drb068_foo__F1L29_1Arg0, i64* %__nv_drb068_foo__F1L29_1Arg1, i64* %__nv_drb068_foo__F1L29_1Arg2) #1 !dbg !56 {
L.entry:
  %__gtid___nv_drb068_foo__F1L29_1__497 = alloca i32, align 4
  %.i0000p_333 = alloca i32, align 4
  %i_332 = alloca i32, align 4
  %.du0002p_364 = alloca i32, align 4
  %.de0002p_365 = alloca i32, align 4
  %.di0002p_366 = alloca i32, align 4
  %.ds0002p_367 = alloca i32, align 4
  %.dl0002p_369 = alloca i32, align 4
  %.dl0002p.copy_491 = alloca i32, align 4
  %.de0002p.copy_492 = alloca i32, align 4
  %.ds0002p.copy_493 = alloca i32, align 4
  %.dX0002p_368 = alloca i32, align 4
  %.dY0002p_363 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_drb068_foo__F1L29_1Arg0, metadata !59, metadata !DIExpression()), !dbg !60
  call void @llvm.dbg.declare(metadata i64* %__nv_drb068_foo__F1L29_1Arg1, metadata !61, metadata !DIExpression()), !dbg !60
  call void @llvm.dbg.declare(metadata i64* %__nv_drb068_foo__F1L29_1Arg2, metadata !62, metadata !DIExpression()), !dbg !60
  call void @llvm.dbg.value(metadata i32 1, metadata !63, metadata !DIExpression()), !dbg !60
  call void @llvm.dbg.value(metadata i32 0, metadata !64, metadata !DIExpression()), !dbg !60
  call void @llvm.dbg.value(metadata i32 1, metadata !65, metadata !DIExpression()), !dbg !60
  call void @llvm.dbg.value(metadata i32 0, metadata !66, metadata !DIExpression()), !dbg !60
  call void @llvm.dbg.value(metadata i32 1, metadata !67, metadata !DIExpression()), !dbg !60
  %0 = load i32, i32* %__nv_drb068_foo__F1L29_1Arg0, align 4, !dbg !68
  store i32 %0, i32* %__gtid___nv_drb068_foo__F1L29_1__497, align 4, !dbg !68
  br label %L.LB3_482

L.LB3_482:                                        ; preds = %L.entry
  br label %L.LB3_331

L.LB3_331:                                        ; preds = %L.LB3_482
  store i32 0, i32* %.i0000p_333, align 4, !dbg !69
  call void @llvm.dbg.declare(metadata i32* %i_332, metadata !70, metadata !DIExpression()), !dbg !68
  store i32 1, i32* %i_332, align 4, !dbg !69
  %1 = bitcast i64* %__nv_drb068_foo__F1L29_1Arg2 to i32**, !dbg !69
  %2 = load i32*, i32** %1, align 8, !dbg !69
  %3 = load i32, i32* %2, align 4, !dbg !69
  store i32 %3, i32* %.du0002p_364, align 4, !dbg !69
  %4 = bitcast i64* %__nv_drb068_foo__F1L29_1Arg2 to i32**, !dbg !69
  %5 = load i32*, i32** %4, align 8, !dbg !69
  %6 = load i32, i32* %5, align 4, !dbg !69
  store i32 %6, i32* %.de0002p_365, align 4, !dbg !69
  store i32 1, i32* %.di0002p_366, align 4, !dbg !69
  %7 = load i32, i32* %.di0002p_366, align 4, !dbg !69
  store i32 %7, i32* %.ds0002p_367, align 4, !dbg !69
  store i32 1, i32* %.dl0002p_369, align 4, !dbg !69
  %8 = load i32, i32* %.dl0002p_369, align 4, !dbg !69
  store i32 %8, i32* %.dl0002p.copy_491, align 4, !dbg !69
  %9 = load i32, i32* %.de0002p_365, align 4, !dbg !69
  store i32 %9, i32* %.de0002p.copy_492, align 4, !dbg !69
  %10 = load i32, i32* %.ds0002p_367, align 4, !dbg !69
  store i32 %10, i32* %.ds0002p.copy_493, align 4, !dbg !69
  %11 = load i32, i32* %__gtid___nv_drb068_foo__F1L29_1__497, align 4, !dbg !69
  %12 = bitcast i32* %.i0000p_333 to i64*, !dbg !69
  %13 = bitcast i32* %.dl0002p.copy_491 to i64*, !dbg !69
  %14 = bitcast i32* %.de0002p.copy_492 to i64*, !dbg !69
  %15 = bitcast i32* %.ds0002p.copy_493 to i64*, !dbg !69
  %16 = load i32, i32* %.ds0002p.copy_493, align 4, !dbg !69
  call void @__kmpc_for_static_init_4(i64* null, i32 %11, i32 34, i64* %12, i64* %13, i64* %14, i64* %15, i32 %16, i32 1), !dbg !69
  %17 = load i32, i32* %.dl0002p.copy_491, align 4, !dbg !69
  store i32 %17, i32* %.dl0002p_369, align 4, !dbg !69
  %18 = load i32, i32* %.de0002p.copy_492, align 4, !dbg !69
  store i32 %18, i32* %.de0002p_365, align 4, !dbg !69
  %19 = load i32, i32* %.ds0002p.copy_493, align 4, !dbg !69
  store i32 %19, i32* %.ds0002p_367, align 4, !dbg !69
  %20 = load i32, i32* %.dl0002p_369, align 4, !dbg !69
  store i32 %20, i32* %i_332, align 4, !dbg !69
  %21 = load i32, i32* %i_332, align 4, !dbg !69
  call void @llvm.dbg.value(metadata i32 %21, metadata !70, metadata !DIExpression()), !dbg !68
  store i32 %21, i32* %.dX0002p_368, align 4, !dbg !69
  %22 = load i32, i32* %.dX0002p_368, align 4, !dbg !69
  %23 = load i32, i32* %.du0002p_364, align 4, !dbg !69
  %24 = icmp sgt i32 %22, %23, !dbg !69
  br i1 %24, label %L.LB3_362, label %L.LB3_523, !dbg !69

L.LB3_523:                                        ; preds = %L.LB3_331
  %25 = load i32, i32* %.dX0002p_368, align 4, !dbg !69
  store i32 %25, i32* %i_332, align 4, !dbg !69
  %26 = load i32, i32* %.di0002p_366, align 4, !dbg !69
  %27 = load i32, i32* %.de0002p_365, align 4, !dbg !69
  %28 = load i32, i32* %.dX0002p_368, align 4, !dbg !69
  %29 = sub nsw i32 %27, %28, !dbg !69
  %30 = add nsw i32 %26, %29, !dbg !69
  %31 = load i32, i32* %.di0002p_366, align 4, !dbg !69
  %32 = sdiv i32 %30, %31, !dbg !69
  store i32 %32, i32* %.dY0002p_363, align 4, !dbg !69
  %33 = load i32, i32* %.dY0002p_363, align 4, !dbg !69
  %34 = icmp sle i32 %33, 0, !dbg !69
  br i1 %34, label %L.LB3_372, label %L.LB3_371, !dbg !69

L.LB3_371:                                        ; preds = %L.LB3_371, %L.LB3_523
  %35 = bitcast i64* %__nv_drb068_foo__F1L29_1Arg2 to i8*, !dbg !71
  %36 = getelementptr i8, i8* %35, i64 72, !dbg !71
  %37 = bitcast i8* %36 to i8***, !dbg !71
  %38 = load i8**, i8*** %37, align 8, !dbg !71
  %39 = load i8*, i8** %38, align 8, !dbg !71
  %40 = bitcast i64* %__nv_drb068_foo__F1L29_1Arg2 to i8*, !dbg !71
  %41 = getelementptr i8, i8* %40, i64 64, !dbg !71
  %42 = bitcast i8* %41 to i8**, !dbg !71
  %43 = load i8*, i8** %42, align 8, !dbg !71
  %44 = getelementptr i8, i8* %43, i64 24, !dbg !71
  %45 = bitcast i8* %44 to i64*, !dbg !71
  %46 = load i64, i64* %45, align 8, !dbg !71
  %47 = load i32, i32* %i_332, align 4, !dbg !71
  call void @llvm.dbg.value(metadata i32 %47, metadata !70, metadata !DIExpression()), !dbg !68
  %48 = sext i32 %47 to i64, !dbg !71
  %49 = bitcast i64* %__nv_drb068_foo__F1L29_1Arg2 to i8*, !dbg !71
  %50 = getelementptr i8, i8* %49, i64 64, !dbg !71
  %51 = bitcast i8* %50 to i8**, !dbg !71
  %52 = load i8*, i8** %51, align 8, !dbg !71
  %53 = getelementptr i8, i8* %52, i64 112, !dbg !71
  %54 = bitcast i8* %53 to i64*, !dbg !71
  %55 = load i64, i64* %54, align 8, !dbg !71
  %56 = mul nsw i64 %48, %55, !dbg !71
  %57 = bitcast i64* %__nv_drb068_foo__F1L29_1Arg2 to i8*, !dbg !71
  %58 = getelementptr i8, i8* %57, i64 64, !dbg !71
  %59 = bitcast i8* %58 to i8**, !dbg !71
  %60 = load i8*, i8** %59, align 8, !dbg !71
  %61 = getelementptr i8, i8* %60, i64 56, !dbg !71
  %62 = bitcast i8* %61 to i64*, !dbg !71
  %63 = load i64, i64* %62, align 8, !dbg !71
  %64 = add nsw i64 %56, %63, !dbg !71
  %65 = sub nsw i64 %64, 1, !dbg !71
  %66 = mul nsw i64 %46, %65, !dbg !71
  %67 = getelementptr i8, i8* %39, i64 %66, !dbg !71
  %68 = bitcast i8* %67 to i32*, !dbg !71
  %69 = load i32, i32* %68, align 4, !dbg !71
  %70 = bitcast i64* %__nv_drb068_foo__F1L29_1Arg2 to i8*, !dbg !71
  %71 = getelementptr i8, i8* %70, i64 48, !dbg !71
  %72 = bitcast i8* %71 to i8***, !dbg !71
  %73 = load i8**, i8*** %72, align 8, !dbg !71
  %74 = load i8*, i8** %73, align 8, !dbg !71
  %75 = bitcast i64* %__nv_drb068_foo__F1L29_1Arg2 to i8*, !dbg !71
  %76 = getelementptr i8, i8* %75, i64 40, !dbg !71
  %77 = bitcast i8* %76 to i8**, !dbg !71
  %78 = load i8*, i8** %77, align 8, !dbg !71
  %79 = getelementptr i8, i8* %78, i64 24, !dbg !71
  %80 = bitcast i8* %79 to i64*, !dbg !71
  %81 = load i64, i64* %80, align 8, !dbg !71
  %82 = load i32, i32* %i_332, align 4, !dbg !71
  call void @llvm.dbg.value(metadata i32 %82, metadata !70, metadata !DIExpression()), !dbg !68
  %83 = sext i32 %82 to i64, !dbg !71
  %84 = bitcast i64* %__nv_drb068_foo__F1L29_1Arg2 to i8*, !dbg !71
  %85 = getelementptr i8, i8* %84, i64 40, !dbg !71
  %86 = bitcast i8* %85 to i8**, !dbg !71
  %87 = load i8*, i8** %86, align 8, !dbg !71
  %88 = getelementptr i8, i8* %87, i64 112, !dbg !71
  %89 = bitcast i8* %88 to i64*, !dbg !71
  %90 = load i64, i64* %89, align 8, !dbg !71
  %91 = mul nsw i64 %83, %90, !dbg !71
  %92 = bitcast i64* %__nv_drb068_foo__F1L29_1Arg2 to i8*, !dbg !71
  %93 = getelementptr i8, i8* %92, i64 40, !dbg !71
  %94 = bitcast i8* %93 to i8**, !dbg !71
  %95 = load i8*, i8** %94, align 8, !dbg !71
  %96 = getelementptr i8, i8* %95, i64 56, !dbg !71
  %97 = bitcast i8* %96 to i64*, !dbg !71
  %98 = load i64, i64* %97, align 8, !dbg !71
  %99 = add nsw i64 %91, %98, !dbg !71
  %100 = sub nsw i64 %99, 1, !dbg !71
  %101 = mul nsw i64 %81, %100, !dbg !71
  %102 = getelementptr i8, i8* %74, i64 %101, !dbg !71
  %103 = bitcast i8* %102 to i32*, !dbg !71
  %104 = load i32, i32* %103, align 4, !dbg !71
  %105 = add nsw i32 %69, %104, !dbg !71
  %106 = bitcast i64* %__nv_drb068_foo__F1L29_1Arg2 to i8*, !dbg !71
  %107 = getelementptr i8, i8* %106, i64 24, !dbg !71
  %108 = bitcast i8* %107 to i8***, !dbg !71
  %109 = load i8**, i8*** %108, align 8, !dbg !71
  %110 = load i8*, i8** %109, align 8, !dbg !71
  %111 = bitcast i64* %__nv_drb068_foo__F1L29_1Arg2 to i8*, !dbg !71
  %112 = getelementptr i8, i8* %111, i64 16, !dbg !71
  %113 = bitcast i8* %112 to i8**, !dbg !71
  %114 = load i8*, i8** %113, align 8, !dbg !71
  %115 = getelementptr i8, i8* %114, i64 24, !dbg !71
  %116 = bitcast i8* %115 to i64*, !dbg !71
  %117 = load i64, i64* %116, align 8, !dbg !71
  %118 = load i32, i32* %i_332, align 4, !dbg !71
  call void @llvm.dbg.value(metadata i32 %118, metadata !70, metadata !DIExpression()), !dbg !68
  %119 = sext i32 %118 to i64, !dbg !71
  %120 = bitcast i64* %__nv_drb068_foo__F1L29_1Arg2 to i8*, !dbg !71
  %121 = getelementptr i8, i8* %120, i64 16, !dbg !71
  %122 = bitcast i8* %121 to i8**, !dbg !71
  %123 = load i8*, i8** %122, align 8, !dbg !71
  %124 = getelementptr i8, i8* %123, i64 112, !dbg !71
  %125 = bitcast i8* %124 to i64*, !dbg !71
  %126 = load i64, i64* %125, align 8, !dbg !71
  %127 = mul nsw i64 %119, %126, !dbg !71
  %128 = bitcast i64* %__nv_drb068_foo__F1L29_1Arg2 to i8*, !dbg !71
  %129 = getelementptr i8, i8* %128, i64 16, !dbg !71
  %130 = bitcast i8* %129 to i8**, !dbg !71
  %131 = load i8*, i8** %130, align 8, !dbg !71
  %132 = getelementptr i8, i8* %131, i64 56, !dbg !71
  %133 = bitcast i8* %132 to i64*, !dbg !71
  %134 = load i64, i64* %133, align 8, !dbg !71
  %135 = add nsw i64 %127, %134, !dbg !71
  %136 = sub nsw i64 %135, 1, !dbg !71
  %137 = mul nsw i64 %117, %136, !dbg !71
  %138 = getelementptr i8, i8* %110, i64 %137, !dbg !71
  %139 = bitcast i8* %138 to i32*, !dbg !71
  store i32 %105, i32* %139, align 4, !dbg !71
  %140 = load i32, i32* %.di0002p_366, align 4, !dbg !68
  %141 = load i32, i32* %i_332, align 4, !dbg !68
  call void @llvm.dbg.value(metadata i32 %141, metadata !70, metadata !DIExpression()), !dbg !68
  %142 = add nsw i32 %140, %141, !dbg !68
  store i32 %142, i32* %i_332, align 4, !dbg !68
  %143 = load i32, i32* %.dY0002p_363, align 4, !dbg !68
  %144 = sub nsw i32 %143, 1, !dbg !68
  store i32 %144, i32* %.dY0002p_363, align 4, !dbg !68
  %145 = load i32, i32* %.dY0002p_363, align 4, !dbg !68
  %146 = icmp sgt i32 %145, 0, !dbg !68
  br i1 %146, label %L.LB3_371, label %L.LB3_372, !dbg !68

L.LB3_372:                                        ; preds = %L.LB3_371, %L.LB3_523
  br label %L.LB3_362

L.LB3_362:                                        ; preds = %L.LB3_372, %L.LB3_331
  %147 = load i32, i32* %__gtid___nv_drb068_foo__F1L29_1__497, align 4, !dbg !68
  call void @__kmpc_for_static_fini(i64* null, i32 %147), !dbg !68
  br label %L.LB3_334

L.LB3_334:                                        ; preds = %L.LB3_362
  ret void, !dbg !68
}

define void @MAIN_() #1 !dbg !2 {
L.entry:
  %"d$p_334" = alloca i32*, align 8
  %"d$sd31_335" = alloca [16 x i64], align 8
  %"c$p_330" = alloca i32*, align 8
  %"c$sd28_331" = alloca [16 x i64], align 8
  %"b$p_326" = alloca i32*, align 8
  %"b$sd25_327" = alloca [16 x i64], align 8
  %"a$p_322" = alloca i32*, align 8
  %"a$sd22_323" = alloca [16 x i64], align 8
  %.g0001_385 = alloca i64, align 8
  %"MAIN___$eq_297" = alloca [576 x i8], align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !72, metadata !DIExpression()), !dbg !73
  call void @llvm.dbg.value(metadata i32 0, metadata !74, metadata !DIExpression()), !dbg !73
  call void @llvm.dbg.value(metadata i32 1, metadata !75, metadata !DIExpression()), !dbg !73
  call void @llvm.dbg.value(metadata i32 0, metadata !76, metadata !DIExpression()), !dbg !73
  call void @llvm.dbg.value(metadata i32 1, metadata !77, metadata !DIExpression()), !dbg !73
  %0 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !78
  %1 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !78
  call void (i8*, ...) %1(i8* %0), !dbg !78
  call void @llvm.dbg.declare(metadata i32** %"d$p_334", metadata !79, metadata !DIExpression(DW_OP_deref)), !dbg !73
  %2 = bitcast i32** %"d$p_334" to i8**, !dbg !78
  store i8* null, i8** %2, align 8, !dbg !78
  call void @llvm.dbg.declare(metadata [16 x i64]* %"d$sd31_335", metadata !80, metadata !DIExpression()), !dbg !73
  %3 = bitcast [16 x i64]* %"d$sd31_335" to i64*, !dbg !78
  store i64 0, i64* %3, align 8, !dbg !78
  call void @llvm.dbg.declare(metadata i32** %"c$p_330", metadata !81, metadata !DIExpression(DW_OP_deref)), !dbg !73
  %4 = bitcast i32** %"c$p_330" to i8**, !dbg !78
  store i8* null, i8** %4, align 8, !dbg !78
  call void @llvm.dbg.declare(metadata [16 x i64]* %"c$sd28_331", metadata !80, metadata !DIExpression()), !dbg !73
  %5 = bitcast [16 x i64]* %"c$sd28_331" to i64*, !dbg !78
  store i64 0, i64* %5, align 8, !dbg !78
  call void @llvm.dbg.declare(metadata i32** %"b$p_326", metadata !82, metadata !DIExpression(DW_OP_deref)), !dbg !73
  %6 = bitcast i32** %"b$p_326" to i8**, !dbg !78
  store i8* null, i8** %6, align 8, !dbg !78
  call void @llvm.dbg.declare(metadata [16 x i64]* %"b$sd25_327", metadata !80, metadata !DIExpression()), !dbg !73
  %7 = bitcast [16 x i64]* %"b$sd25_327" to i64*, !dbg !78
  store i64 0, i64* %7, align 8, !dbg !78
  call void @llvm.dbg.declare(metadata i32** %"a$p_322", metadata !83, metadata !DIExpression(DW_OP_deref)), !dbg !73
  %8 = bitcast i32** %"a$p_322" to i8**, !dbg !78
  store i8* null, i8** %8, align 8, !dbg !78
  call void @llvm.dbg.declare(metadata [16 x i64]* %"a$sd22_323", metadata !80, metadata !DIExpression()), !dbg !73
  %9 = bitcast [16 x i64]* %"a$sd22_323" to i64*, !dbg !78
  store i64 0, i64* %9, align 8, !dbg !78
  br label %L.LB5_368

L.LB5_368:                                        ; preds = %L.entry
  %10 = bitcast [16 x i64]* %"a$sd22_323" to i8*, !dbg !84
  %11 = getelementptr i8, i8* %10, i64 80, !dbg !84
  %12 = bitcast i8* %11 to i64*, !dbg !84
  store i64 1, i64* %12, align 8, !dbg !84
  %13 = bitcast %struct.STATICS5* @.STATICS5 to i32*, !dbg !84
  %14 = load i32, i32* %13, align 4, !dbg !84
  %15 = sext i32 %14 to i64, !dbg !84
  %16 = bitcast [16 x i64]* %"a$sd22_323" to i8*, !dbg !84
  %17 = getelementptr i8, i8* %16, i64 88, !dbg !84
  %18 = bitcast i8* %17 to i64*, !dbg !84
  store i64 %15, i64* %18, align 8, !dbg !84
  %19 = bitcast [16 x i64]* %"a$sd22_323" to i8*, !dbg !84
  %20 = getelementptr i8, i8* %19, i64 88, !dbg !84
  %21 = bitcast i8* %20 to i64*, !dbg !84
  %22 = load i64, i64* %21, align 8, !dbg !84
  %23 = sub nsw i64 %22, 1, !dbg !84
  %24 = bitcast [16 x i64]* %"a$sd22_323" to i8*, !dbg !84
  %25 = getelementptr i8, i8* %24, i64 80, !dbg !84
  %26 = bitcast i8* %25 to i64*, !dbg !84
  %27 = load i64, i64* %26, align 8, !dbg !84
  %28 = add nsw i64 %23, %27, !dbg !84
  store i64 %28, i64* %.g0001_385, align 8, !dbg !84
  %29 = bitcast [16 x i64]* %"a$sd22_323" to i8*, !dbg !84
  %30 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !84
  %31 = bitcast i64* @.C338_MAIN_ to i8*, !dbg !84
  %32 = bitcast i64* @.C339_MAIN_ to i8*, !dbg !84
  %33 = bitcast [16 x i64]* %"a$sd22_323" to i8*, !dbg !84
  %34 = getelementptr i8, i8* %33, i64 80, !dbg !84
  %35 = bitcast i64* %.g0001_385 to i8*, !dbg !84
  %36 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !84
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %36(i8* %29, i8* %30, i8* %31, i8* %32, i8* %34, i8* %35), !dbg !84
  %37 = bitcast [16 x i64]* %"a$sd22_323" to i8*, !dbg !84
  %38 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !84
  call void (i8*, i32, ...) %38(i8* %37, i32 25), !dbg !84
  %39 = bitcast [16 x i64]* %"a$sd22_323" to i8*, !dbg !84
  %40 = getelementptr i8, i8* %39, i64 88, !dbg !84
  %41 = bitcast i8* %40 to i64*, !dbg !84
  %42 = load i64, i64* %41, align 8, !dbg !84
  %43 = sub nsw i64 %42, 1, !dbg !84
  %44 = bitcast [16 x i64]* %"a$sd22_323" to i8*, !dbg !84
  %45 = getelementptr i8, i8* %44, i64 80, !dbg !84
  %46 = bitcast i8* %45 to i64*, !dbg !84
  %47 = load i64, i64* %46, align 8, !dbg !84
  %48 = add nsw i64 %43, %47, !dbg !84
  %49 = bitcast [16 x i64]* %"a$sd22_323" to i8*, !dbg !84
  %50 = getelementptr i8, i8* %49, i64 80, !dbg !84
  %51 = bitcast i8* %50 to i64*, !dbg !84
  %52 = load i64, i64* %51, align 8, !dbg !84
  %53 = sub nsw i64 %52, 1, !dbg !84
  %54 = sub nsw i64 %48, %53, !dbg !84
  store i64 %54, i64* %.g0001_385, align 8, !dbg !84
  %55 = bitcast i64* %.g0001_385 to i8*, !dbg !84
  %56 = bitcast i64* @.C338_MAIN_ to i8*, !dbg !84
  %57 = bitcast i64* @.C339_MAIN_ to i8*, !dbg !84
  %58 = bitcast i32** %"a$p_322" to i8*, !dbg !84
  %59 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !84
  %60 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !84
  %61 = bitcast void (...)* @f90_alloc04a_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !84
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %61(i8* %55, i8* %56, i8* %57, i8* null, i8* %58, i8* null, i8* %59, i8* %60, i8* null, i64 0), !dbg !84
  %62 = bitcast [16 x i64]* %"a$sd22_323" to i8*, !dbg !84
  %63 = getelementptr i8, i8* %62, i64 64, !dbg !84
  %64 = bitcast i32** %"a$p_322" to i8*, !dbg !84
  %65 = bitcast void (...)* @f90_ptrcp to void (i8*, i8*, ...)*, !dbg !84
  call void (i8*, i8*, ...) %65(i8* %63, i8* %64), !dbg !84
  %66 = bitcast [16 x i64]* %"b$sd25_327" to i8*, !dbg !85
  %67 = getelementptr i8, i8* %66, i64 80, !dbg !85
  %68 = bitcast i8* %67 to i64*, !dbg !85
  store i64 1, i64* %68, align 8, !dbg !85
  %69 = bitcast %struct.STATICS5* @.STATICS5 to i32*, !dbg !85
  %70 = load i32, i32* %69, align 4, !dbg !85
  %71 = sext i32 %70 to i64, !dbg !85
  %72 = bitcast [16 x i64]* %"b$sd25_327" to i8*, !dbg !85
  %73 = getelementptr i8, i8* %72, i64 88, !dbg !85
  %74 = bitcast i8* %73 to i64*, !dbg !85
  store i64 %71, i64* %74, align 8, !dbg !85
  %75 = bitcast [16 x i64]* %"b$sd25_327" to i8*, !dbg !85
  %76 = getelementptr i8, i8* %75, i64 88, !dbg !85
  %77 = bitcast i8* %76 to i64*, !dbg !85
  %78 = load i64, i64* %77, align 8, !dbg !85
  %79 = sub nsw i64 %78, 1, !dbg !85
  %80 = bitcast [16 x i64]* %"b$sd25_327" to i8*, !dbg !85
  %81 = getelementptr i8, i8* %80, i64 80, !dbg !85
  %82 = bitcast i8* %81 to i64*, !dbg !85
  %83 = load i64, i64* %82, align 8, !dbg !85
  %84 = add nsw i64 %79, %83, !dbg !85
  store i64 %84, i64* %.g0001_385, align 8, !dbg !85
  %85 = bitcast [16 x i64]* %"b$sd25_327" to i8*, !dbg !85
  %86 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !85
  %87 = bitcast i64* @.C338_MAIN_ to i8*, !dbg !85
  %88 = bitcast i64* @.C339_MAIN_ to i8*, !dbg !85
  %89 = bitcast [16 x i64]* %"b$sd25_327" to i8*, !dbg !85
  %90 = getelementptr i8, i8* %89, i64 80, !dbg !85
  %91 = bitcast i64* %.g0001_385 to i8*, !dbg !85
  %92 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !85
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %92(i8* %85, i8* %86, i8* %87, i8* %88, i8* %90, i8* %91), !dbg !85
  %93 = bitcast [16 x i64]* %"b$sd25_327" to i8*, !dbg !85
  %94 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !85
  call void (i8*, i32, ...) %94(i8* %93, i32 25), !dbg !85
  %95 = bitcast [16 x i64]* %"b$sd25_327" to i8*, !dbg !85
  %96 = getelementptr i8, i8* %95, i64 88, !dbg !85
  %97 = bitcast i8* %96 to i64*, !dbg !85
  %98 = load i64, i64* %97, align 8, !dbg !85
  %99 = sub nsw i64 %98, 1, !dbg !85
  %100 = bitcast [16 x i64]* %"b$sd25_327" to i8*, !dbg !85
  %101 = getelementptr i8, i8* %100, i64 80, !dbg !85
  %102 = bitcast i8* %101 to i64*, !dbg !85
  %103 = load i64, i64* %102, align 8, !dbg !85
  %104 = add nsw i64 %99, %103, !dbg !85
  %105 = bitcast [16 x i64]* %"b$sd25_327" to i8*, !dbg !85
  %106 = getelementptr i8, i8* %105, i64 80, !dbg !85
  %107 = bitcast i8* %106 to i64*, !dbg !85
  %108 = load i64, i64* %107, align 8, !dbg !85
  %109 = sub nsw i64 %108, 1, !dbg !85
  %110 = sub nsw i64 %104, %109, !dbg !85
  store i64 %110, i64* %.g0001_385, align 8, !dbg !85
  %111 = bitcast i64* %.g0001_385 to i8*, !dbg !85
  %112 = bitcast i64* @.C338_MAIN_ to i8*, !dbg !85
  %113 = bitcast i64* @.C339_MAIN_ to i8*, !dbg !85
  %114 = bitcast i32** %"b$p_326" to i8*, !dbg !85
  %115 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !85
  %116 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !85
  %117 = bitcast void (...)* @f90_alloc04a_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !85
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %117(i8* %111, i8* %112, i8* %113, i8* null, i8* %114, i8* null, i8* %115, i8* %116, i8* null, i64 0), !dbg !85
  %118 = bitcast [16 x i64]* %"b$sd25_327" to i8*, !dbg !85
  %119 = getelementptr i8, i8* %118, i64 64, !dbg !85
  %120 = bitcast i32** %"b$p_326" to i8*, !dbg !85
  %121 = bitcast void (...)* @f90_ptrcp to void (i8*, i8*, ...)*, !dbg !85
  call void (i8*, i8*, ...) %121(i8* %119, i8* %120), !dbg !85
  %122 = bitcast [16 x i64]* %"c$sd28_331" to i8*, !dbg !86
  %123 = getelementptr i8, i8* %122, i64 80, !dbg !86
  %124 = bitcast i8* %123 to i64*, !dbg !86
  store i64 1, i64* %124, align 8, !dbg !86
  %125 = bitcast %struct.STATICS5* @.STATICS5 to i32*, !dbg !86
  %126 = load i32, i32* %125, align 4, !dbg !86
  %127 = sext i32 %126 to i64, !dbg !86
  %128 = bitcast [16 x i64]* %"c$sd28_331" to i8*, !dbg !86
  %129 = getelementptr i8, i8* %128, i64 88, !dbg !86
  %130 = bitcast i8* %129 to i64*, !dbg !86
  store i64 %127, i64* %130, align 8, !dbg !86
  %131 = bitcast [16 x i64]* %"c$sd28_331" to i8*, !dbg !86
  %132 = getelementptr i8, i8* %131, i64 88, !dbg !86
  %133 = bitcast i8* %132 to i64*, !dbg !86
  %134 = load i64, i64* %133, align 8, !dbg !86
  %135 = sub nsw i64 %134, 1, !dbg !86
  %136 = bitcast [16 x i64]* %"c$sd28_331" to i8*, !dbg !86
  %137 = getelementptr i8, i8* %136, i64 80, !dbg !86
  %138 = bitcast i8* %137 to i64*, !dbg !86
  %139 = load i64, i64* %138, align 8, !dbg !86
  %140 = add nsw i64 %135, %139, !dbg !86
  store i64 %140, i64* %.g0001_385, align 8, !dbg !86
  %141 = bitcast [16 x i64]* %"c$sd28_331" to i8*, !dbg !86
  %142 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !86
  %143 = bitcast i64* @.C338_MAIN_ to i8*, !dbg !86
  %144 = bitcast i64* @.C339_MAIN_ to i8*, !dbg !86
  %145 = bitcast [16 x i64]* %"c$sd28_331" to i8*, !dbg !86
  %146 = getelementptr i8, i8* %145, i64 80, !dbg !86
  %147 = bitcast i64* %.g0001_385 to i8*, !dbg !86
  %148 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !86
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %148(i8* %141, i8* %142, i8* %143, i8* %144, i8* %146, i8* %147), !dbg !86
  %149 = bitcast [16 x i64]* %"c$sd28_331" to i8*, !dbg !86
  %150 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !86
  call void (i8*, i32, ...) %150(i8* %149, i32 25), !dbg !86
  %151 = bitcast [16 x i64]* %"c$sd28_331" to i8*, !dbg !86
  %152 = getelementptr i8, i8* %151, i64 88, !dbg !86
  %153 = bitcast i8* %152 to i64*, !dbg !86
  %154 = load i64, i64* %153, align 8, !dbg !86
  %155 = sub nsw i64 %154, 1, !dbg !86
  %156 = bitcast [16 x i64]* %"c$sd28_331" to i8*, !dbg !86
  %157 = getelementptr i8, i8* %156, i64 80, !dbg !86
  %158 = bitcast i8* %157 to i64*, !dbg !86
  %159 = load i64, i64* %158, align 8, !dbg !86
  %160 = add nsw i64 %155, %159, !dbg !86
  %161 = bitcast [16 x i64]* %"c$sd28_331" to i8*, !dbg !86
  %162 = getelementptr i8, i8* %161, i64 80, !dbg !86
  %163 = bitcast i8* %162 to i64*, !dbg !86
  %164 = load i64, i64* %163, align 8, !dbg !86
  %165 = sub nsw i64 %164, 1, !dbg !86
  %166 = sub nsw i64 %160, %165, !dbg !86
  store i64 %166, i64* %.g0001_385, align 8, !dbg !86
  %167 = bitcast i64* %.g0001_385 to i8*, !dbg !86
  %168 = bitcast i64* @.C338_MAIN_ to i8*, !dbg !86
  %169 = bitcast i64* @.C339_MAIN_ to i8*, !dbg !86
  %170 = bitcast i32** %"c$p_330" to i8*, !dbg !86
  %171 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !86
  %172 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !86
  %173 = bitcast void (...)* @f90_alloc04a_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !86
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %173(i8* %167, i8* %168, i8* %169, i8* null, i8* %170, i8* null, i8* %171, i8* %172, i8* null, i64 0), !dbg !86
  %174 = bitcast [16 x i64]* %"c$sd28_331" to i8*, !dbg !86
  %175 = getelementptr i8, i8* %174, i64 64, !dbg !86
  %176 = bitcast i32** %"c$p_330" to i8*, !dbg !86
  %177 = bitcast void (...)* @f90_ptrcp to void (i8*, i8*, ...)*, !dbg !86
  call void (i8*, i8*, ...) %177(i8* %175, i8* %176), !dbg !86
  %178 = bitcast [16 x i64]* %"d$sd31_335" to i8*, !dbg !87
  %179 = getelementptr i8, i8* %178, i64 80, !dbg !87
  %180 = bitcast i8* %179 to i64*, !dbg !87
  store i64 1, i64* %180, align 8, !dbg !87
  %181 = bitcast %struct.STATICS5* @.STATICS5 to i32*, !dbg !87
  %182 = load i32, i32* %181, align 4, !dbg !87
  %183 = sext i32 %182 to i64, !dbg !87
  %184 = bitcast [16 x i64]* %"d$sd31_335" to i8*, !dbg !87
  %185 = getelementptr i8, i8* %184, i64 88, !dbg !87
  %186 = bitcast i8* %185 to i64*, !dbg !87
  store i64 %183, i64* %186, align 8, !dbg !87
  %187 = bitcast [16 x i64]* %"d$sd31_335" to i8*, !dbg !87
  %188 = getelementptr i8, i8* %187, i64 88, !dbg !87
  %189 = bitcast i8* %188 to i64*, !dbg !87
  %190 = load i64, i64* %189, align 8, !dbg !87
  %191 = sub nsw i64 %190, 1, !dbg !87
  %192 = bitcast [16 x i64]* %"d$sd31_335" to i8*, !dbg !87
  %193 = getelementptr i8, i8* %192, i64 80, !dbg !87
  %194 = bitcast i8* %193 to i64*, !dbg !87
  %195 = load i64, i64* %194, align 8, !dbg !87
  %196 = add nsw i64 %191, %195, !dbg !87
  store i64 %196, i64* %.g0001_385, align 8, !dbg !87
  %197 = bitcast [16 x i64]* %"d$sd31_335" to i8*, !dbg !87
  %198 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !87
  %199 = bitcast i64* @.C338_MAIN_ to i8*, !dbg !87
  %200 = bitcast i64* @.C339_MAIN_ to i8*, !dbg !87
  %201 = bitcast [16 x i64]* %"d$sd31_335" to i8*, !dbg !87
  %202 = getelementptr i8, i8* %201, i64 80, !dbg !87
  %203 = bitcast i64* %.g0001_385 to i8*, !dbg !87
  %204 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, ...)*, !dbg !87
  call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %204(i8* %197, i8* %198, i8* %199, i8* %200, i8* %202, i8* %203), !dbg !87
  %205 = bitcast [16 x i64]* %"d$sd31_335" to i8*, !dbg !87
  %206 = bitcast void (...)* @f90_set_intrin_type_i8 to void (i8*, i32, ...)*, !dbg !87
  call void (i8*, i32, ...) %206(i8* %205, i32 25), !dbg !87
  %207 = bitcast [16 x i64]* %"d$sd31_335" to i8*, !dbg !87
  %208 = getelementptr i8, i8* %207, i64 88, !dbg !87
  %209 = bitcast i8* %208 to i64*, !dbg !87
  %210 = load i64, i64* %209, align 8, !dbg !87
  %211 = sub nsw i64 %210, 1, !dbg !87
  %212 = bitcast [16 x i64]* %"d$sd31_335" to i8*, !dbg !87
  %213 = getelementptr i8, i8* %212, i64 80, !dbg !87
  %214 = bitcast i8* %213 to i64*, !dbg !87
  %215 = load i64, i64* %214, align 8, !dbg !87
  %216 = add nsw i64 %211, %215, !dbg !87
  %217 = bitcast [16 x i64]* %"d$sd31_335" to i8*, !dbg !87
  %218 = getelementptr i8, i8* %217, i64 80, !dbg !87
  %219 = bitcast i8* %218 to i64*, !dbg !87
  %220 = load i64, i64* %219, align 8, !dbg !87
  %221 = sub nsw i64 %220, 1, !dbg !87
  %222 = sub nsw i64 %216, %221, !dbg !87
  store i64 %222, i64* %.g0001_385, align 8, !dbg !87
  %223 = bitcast i64* %.g0001_385 to i8*, !dbg !87
  %224 = bitcast i64* @.C338_MAIN_ to i8*, !dbg !87
  %225 = bitcast i64* @.C339_MAIN_ to i8*, !dbg !87
  %226 = bitcast i32** %"d$p_334" to i8*, !dbg !87
  %227 = bitcast i64* @.C286_MAIN_ to i8*, !dbg !87
  %228 = bitcast i64* @.C284_MAIN_ to i8*, !dbg !87
  %229 = bitcast void (...)* @f90_alloc04a_i8 to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !87
  call void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %229(i8* %223, i8* %224, i8* %225, i8* null, i8* %226, i8* null, i8* %227, i8* %228, i8* null, i64 0), !dbg !87
  %230 = bitcast [16 x i64]* %"d$sd31_335" to i8*, !dbg !87
  %231 = getelementptr i8, i8* %230, i64 64, !dbg !87
  %232 = bitcast i32** %"d$p_334" to i8*, !dbg !87
  %233 = bitcast void (...)* @f90_ptrcp to void (i8*, i8*, ...)*, !dbg !87
  call void (i8*, i8*, ...) %233(i8* %231, i8* %232), !dbg !87
  %234 = bitcast %struct.STATICS5* @.STATICS5 to i32*, !dbg !88
  %235 = load i32, i32* %234, align 4, !dbg !88
  %236 = bitcast i32** %"a$p_322" to i64*, !dbg !88
  %237 = bitcast i32** %"b$p_326" to i64*, !dbg !88
  %238 = bitcast i32** %"c$p_330" to i64*, !dbg !88
  %239 = bitcast i32** %"d$p_334" to i64*, !dbg !88
  %240 = bitcast [16 x i64]* %"a$sd22_323" to i64*, !dbg !88
  %241 = bitcast [16 x i64]* %"b$sd25_327" to i64*, !dbg !88
  %242 = bitcast [16 x i64]* %"c$sd28_331" to i64*, !dbg !88
  %243 = bitcast [16 x i64]* %"d$sd31_335" to i64*, !dbg !88
  call void @drb068_foo_(i32 %235, i64* %236, i64* %237, i64* %238, i64* %239, i64* %240, i64* %241, i64* %242, i64* %243), !dbg !88
  %244 = load i32*, i32** %"a$p_322", align 8, !dbg !89
  call void @llvm.dbg.value(metadata i32* %244, metadata !83, metadata !DIExpression()), !dbg !73
  %245 = bitcast i32* %244 to i8*, !dbg !89
  %246 = icmp eq i8* %245, null, !dbg !89
  br i1 %246, label %L.LB5_348, label %L.LB5_409, !dbg !89

L.LB5_409:                                        ; preds = %L.LB5_368
  %247 = bitcast i32** %"a$p_322" to i8**, !dbg !89
  store i8* null, i8** %247, align 8, !dbg !89
  %248 = bitcast [16 x i64]* %"a$sd22_323" to i64*, !dbg !89
  store i64 0, i64* %248, align 8, !dbg !89
  br label %L.LB5_348

L.LB5_348:                                        ; preds = %L.LB5_409, %L.LB5_368
  %249 = load i32*, i32** %"b$p_326", align 8, !dbg !90
  call void @llvm.dbg.value(metadata i32* %249, metadata !82, metadata !DIExpression()), !dbg !73
  %250 = bitcast i32* %249 to i8*, !dbg !90
  %251 = icmp eq i8* %250, null, !dbg !90
  br i1 %251, label %L.LB5_349, label %L.LB5_410, !dbg !90

L.LB5_410:                                        ; preds = %L.LB5_348
  %252 = bitcast i32** %"b$p_326" to i8**, !dbg !90
  store i8* null, i8** %252, align 8, !dbg !90
  %253 = bitcast [16 x i64]* %"b$sd25_327" to i64*, !dbg !90
  store i64 0, i64* %253, align 8, !dbg !90
  br label %L.LB5_349

L.LB5_349:                                        ; preds = %L.LB5_410, %L.LB5_348
  %254 = load i32*, i32** %"c$p_330", align 8, !dbg !91
  call void @llvm.dbg.value(metadata i32* %254, metadata !81, metadata !DIExpression()), !dbg !73
  %255 = bitcast i32* %254 to i8*, !dbg !91
  %256 = icmp eq i8* %255, null, !dbg !91
  br i1 %256, label %L.LB5_350, label %L.LB5_411, !dbg !91

L.LB5_411:                                        ; preds = %L.LB5_349
  %257 = bitcast i32** %"c$p_330" to i8**, !dbg !91
  store i8* null, i8** %257, align 8, !dbg !91
  %258 = bitcast [16 x i64]* %"c$sd28_331" to i64*, !dbg !91
  store i64 0, i64* %258, align 8, !dbg !91
  br label %L.LB5_350

L.LB5_350:                                        ; preds = %L.LB5_411, %L.LB5_349
  %259 = load i32*, i32** %"d$p_334", align 8, !dbg !92
  call void @llvm.dbg.value(metadata i32* %259, metadata !79, metadata !DIExpression()), !dbg !73
  %260 = bitcast i32* %259 to i8*, !dbg !92
  %261 = icmp eq i8* %260, null, !dbg !92
  br i1 %261, label %L.LB5_351, label %L.LB5_412, !dbg !92

L.LB5_412:                                        ; preds = %L.LB5_350
  %262 = bitcast i32** %"d$p_334" to i8**, !dbg !92
  store i8* null, i8** %262, align 8, !dbg !92
  %263 = bitcast [16 x i64]* %"d$sd31_335" to i64*, !dbg !92
  store i64 0, i64* %263, align 8, !dbg !92
  br label %L.LB5_351

L.LB5_351:                                        ; preds = %L.LB5_412, %L.LB5_350
  ret void, !dbg !93
}

declare void @fort_init(...) #1

declare void @__kmpc_for_static_fini(i64*, i32) #1

declare void @__kmpc_for_static_init_4(i64*, i32, i32, i64*, i64*, i64*, i64*, i32, i32) #1

declare void @_mp_ecs_nest(...) #1

declare signext i32 @f90io_ldw_end(...) #1

declare signext i32 @f90io_sc_i_ldw(...) #1

declare signext i32 @f90io_print_init(...) #1

declare void @f90io_src_info03a(...) #1

declare void @_mp_bcs_nest(...) #1

declare void @f90_ptrcp(...) #1

declare void @f90_alloc04a_i8(...) #1

declare void @f90_set_intrin_type_i8(...) #1

declare void @f90_template1_i8(...) #1

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
!1 = distinct !DIGlobalVariable(name: "n", scope: !2, file: !3, type: !9, isLocal: true, isDefinition: true)
!2 = distinct !DISubprogram(name: "drb068_restrictpointer2_orig_no", scope: !4, file: !3, line: 43, type: !7, scopeLine: 43, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !4)
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB068-restrictpointer2-orig-no.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !6, imports: !5)
!5 = !{}
!6 = !{!0}
!7 = !DISubroutineType(cc: DW_CC_program, types: !8)
!8 = !{null}
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !{i32 2, !"Dwarf Version", i32 4}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = distinct !DISubprogram(name: "foo", scope: !13, file: !3, line: 14, type: !14, scopeLine: 14, spFlags: DISPFlagDefinition, unit: !4)
!13 = !DIModule(scope: !4, name: "drb068")
!14 = !DISubroutineType(types: !15)
!15 = !{null, !9, !16, !16, !16, !16, !19, !19, !19, !19}
!16 = !DICompositeType(tag: DW_TAG_array_type, baseType: !9, size: 32, align: 32, elements: !17)
!17 = !{!18}
!18 = !DISubrange(count: 0, lowerBound: 1)
!19 = !DICompositeType(tag: DW_TAG_array_type, baseType: !20, size: 1024, align: 64, elements: !21)
!20 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!21 = !{!22}
!22 = !DISubrange(count: 16, lowerBound: 1)
!23 = !DILocalVariable(name: "_V_n", scope: !12, file: !3, type: !9)
!24 = !DILocation(line: 0, scope: !12)
!25 = !DILocalVariable(name: "_V_n", arg: 1, scope: !12, file: !3, type: !9)
!26 = !DILocalVariable(arg: 2, scope: !12, file: !3, type: !16, flags: DIFlagArtificial)
!27 = !DILocalVariable(arg: 3, scope: !12, file: !3, type: !16, flags: DIFlagArtificial)
!28 = !DILocalVariable(arg: 4, scope: !12, file: !3, type: !16, flags: DIFlagArtificial)
!29 = !DILocalVariable(arg: 5, scope: !12, file: !3, type: !16, flags: DIFlagArtificial)
!30 = !DILocalVariable(arg: 6, scope: !12, file: !3, type: !19, flags: DIFlagArtificial)
!31 = !DILocalVariable(arg: 7, scope: !12, file: !3, type: !19, flags: DIFlagArtificial)
!32 = !DILocalVariable(arg: 8, scope: !12, file: !3, type: !19, flags: DIFlagArtificial)
!33 = !DILocalVariable(arg: 9, scope: !12, file: !3, type: !19, flags: DIFlagArtificial)
!34 = !DILocalVariable(name: "omp_sched_static", scope: !12, file: !3, type: !9)
!35 = !DILocalVariable(name: "omp_proc_bind_false", scope: !12, file: !3, type: !9)
!36 = !DILocalVariable(name: "omp_proc_bind_true", scope: !12, file: !3, type: !9)
!37 = !DILocalVariable(name: "omp_lock_hint_none", scope: !12, file: !3, type: !9)
!38 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !12, file: !3, type: !9)
!39 = !DILocation(line: 14, column: 1, scope: !12)
!40 = !DILocalVariable(name: "n", scope: !12, file: !3, type: !9)
!41 = !DILocation(line: 40, column: 1, scope: !12)
!42 = !DILocation(line: 19, column: 1, scope: !12)
!43 = !DILocation(line: 20, column: 1, scope: !12)
!44 = !DILocation(line: 21, column: 1, scope: !12)
!45 = !DILocation(line: 22, column: 1, scope: !12)
!46 = !DILocation(line: 24, column: 1, scope: !12)
!47 = !DILocalVariable(name: "i", scope: !12, file: !3, type: !9)
!48 = !DILocation(line: 25, column: 1, scope: !12)
!49 = !DILocation(line: 26, column: 1, scope: !12)
!50 = !DILocation(line: 27, column: 1, scope: !12)
!51 = !DILocation(line: 29, column: 1, scope: !12)
!52 = !DILocation(line: 35, column: 1, scope: !12)
!53 = !DILocation(line: 36, column: 1, scope: !12)
!54 = !DILocalVariable(scope: !12, file: !3, type: !9, flags: DIFlagArtificial)
!55 = !DILocation(line: 39, column: 1, scope: !12)
!56 = distinct !DISubprogram(name: "__nv_drb068_foo__F1L29_1", scope: !4, file: !3, line: 29, type: !57, scopeLine: 29, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !4)
!57 = !DISubroutineType(types: !58)
!58 = !{null, !9, !20, !20}
!59 = !DILocalVariable(name: "__nv_drb068_foo__F1L29_1Arg0", arg: 1, scope: !56, file: !3, type: !9)
!60 = !DILocation(line: 0, scope: !56)
!61 = !DILocalVariable(name: "__nv_drb068_foo__F1L29_1Arg1", arg: 2, scope: !56, file: !3, type: !20)
!62 = !DILocalVariable(name: "__nv_drb068_foo__F1L29_1Arg2", arg: 3, scope: !56, file: !3, type: !20)
!63 = !DILocalVariable(name: "omp_sched_static", scope: !56, file: !3, type: !9)
!64 = !DILocalVariable(name: "omp_proc_bind_false", scope: !56, file: !3, type: !9)
!65 = !DILocalVariable(name: "omp_proc_bind_true", scope: !56, file: !3, type: !9)
!66 = !DILocalVariable(name: "omp_lock_hint_none", scope: !56, file: !3, type: !9)
!67 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !56, file: !3, type: !9)
!68 = !DILocation(line: 32, column: 1, scope: !56)
!69 = !DILocation(line: 30, column: 1, scope: !56)
!70 = !DILocalVariable(name: "i", scope: !56, file: !3, type: !9)
!71 = !DILocation(line: 31, column: 1, scope: !56)
!72 = !DILocalVariable(name: "omp_sched_static", scope: !2, file: !3, type: !9)
!73 = !DILocation(line: 0, scope: !2)
!74 = !DILocalVariable(name: "omp_proc_bind_false", scope: !2, file: !3, type: !9)
!75 = !DILocalVariable(name: "omp_proc_bind_true", scope: !2, file: !3, type: !9)
!76 = !DILocalVariable(name: "omp_lock_hint_none", scope: !2, file: !3, type: !9)
!77 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !2, file: !3, type: !9)
!78 = !DILocation(line: 43, column: 1, scope: !2)
!79 = !DILocalVariable(name: "d", scope: !2, file: !3, type: !16)
!80 = !DILocalVariable(scope: !2, file: !3, type: !19, flags: DIFlagArtificial)
!81 = !DILocalVariable(name: "c", scope: !2, file: !3, type: !16)
!82 = !DILocalVariable(name: "b", scope: !2, file: !3, type: !16)
!83 = !DILocalVariable(name: "a", scope: !2, file: !3, type: !16)
!84 = !DILocation(line: 51, column: 1, scope: !2)
!85 = !DILocation(line: 52, column: 1, scope: !2)
!86 = !DILocation(line: 53, column: 1, scope: !2)
!87 = !DILocation(line: 54, column: 1, scope: !2)
!88 = !DILocation(line: 56, column: 1, scope: !2)
!89 = !DILocation(line: 58, column: 1, scope: !2)
!90 = !DILocation(line: 59, column: 1, scope: !2)
!91 = !DILocation(line: 60, column: 1, scope: !2)
!92 = !DILocation(line: 61, column: 1, scope: !2)
!93 = !DILocation(line: 63, column: 1, scope: !2)

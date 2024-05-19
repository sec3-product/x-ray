; ModuleID = '/tmp/DRB018-plusplus-orig-yes-977012.ll'
source_filename = "/tmp/DRB018-plusplus-orig-yes-977012.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.BSS1 = type <{ [8000 x i8] }>
%struct.STATICS1 = type <{ [48 x i8] }>
%astruct.dt66 = type <{ i8*, i8*, i8*, i8* }>

@.BSS1 = internal global %struct.BSS1 zeroinitializer, align 32, !dbg !0, !dbg !7
@.STATICS1 = internal global %struct.STATICS1 <{ [48 x i8] c"\FB\FF\FF\FF\0C\00\00\00output(500)=\EB\FF\FF\FF\00\00\00\00\03\00\00\00\00\00\00\00\01\00\00\00\FF\FF\FF\FF\00\00\00\00" }>, align 16
@.C305_MAIN_ = internal constant i32 25
@.C327_MAIN_ = internal constant i64 500
@.C284_MAIN_ = internal constant i64 0
@.C324_MAIN_ = internal constant i32 6
@.C320_MAIN_ = internal constant [53 x i8] c"micro-benchmarks-fortran/DRB018-plusplus-orig-yes.f95"
@.C322_MAIN_ = internal constant i32 38
@.C285_MAIN_ = internal constant i32 1
@.C310_MAIN_ = internal constant i32 1000
@.C283_MAIN_ = internal constant i32 0
@.C285___nv_MAIN__F1L31_1 = internal constant i32 1
@.C283___nv_MAIN__F1L31_1 = internal constant i32 0

define void @MAIN_() #0 !dbg !2 {
L.entry:
  %__gtid_MAIN__379 = alloca i32, align 4
  %inlen_307 = alloca i32, align 4
  %outlen_308 = alloca i32, align 4
  %.dY0001_338 = alloca i32, align 4
  %i_306 = alloca i32, align 4
  %.uplevelArgPack0001_367 = alloca %astruct.dt66, align 16
  %z__io_326 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !25, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.value(metadata i32 0, metadata !27, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.value(metadata i32 1, metadata !28, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.value(metadata i32 0, metadata !29, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.value(metadata i32 1, metadata !30, metadata !DIExpression()), !dbg !26
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !31
  store i32 %0, i32* %__gtid_MAIN__379, align 4, !dbg !31
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !32
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !32
  call void (i8*, ...) %2(i8* %1), !dbg !32
  br label %L.LB1_352

L.LB1_352:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %inlen_307, metadata !33, metadata !DIExpression()), !dbg !26
  store i32 1000, i32* %inlen_307, align 4, !dbg !34
  call void @llvm.dbg.declare(metadata i32* %outlen_308, metadata !35, metadata !DIExpression()), !dbg !26
  store i32 1, i32* %outlen_308, align 4, !dbg !36
  %3 = load i32, i32* %inlen_307, align 4, !dbg !37
  call void @llvm.dbg.value(metadata i32 %3, metadata !33, metadata !DIExpression()), !dbg !26
  store i32 %3, i32* %.dY0001_338, align 4, !dbg !37
  call void @llvm.dbg.declare(metadata i32* %i_306, metadata !38, metadata !DIExpression()), !dbg !26
  store i32 1, i32* %i_306, align 4, !dbg !37
  %4 = load i32, i32* %.dY0001_338, align 4, !dbg !37
  %5 = icmp sle i32 %4, 0, !dbg !37
  br i1 %5, label %L.LB1_337, label %L.LB1_336, !dbg !37

L.LB1_336:                                        ; preds = %L.LB1_336, %L.LB1_352
  %6 = load i32, i32* %i_306, align 4, !dbg !39
  call void @llvm.dbg.value(metadata i32 %6, metadata !38, metadata !DIExpression()), !dbg !26
  %7 = load i32, i32* %i_306, align 4, !dbg !39
  call void @llvm.dbg.value(metadata i32 %7, metadata !38, metadata !DIExpression()), !dbg !26
  %8 = sext i32 %7 to i64, !dbg !39
  %9 = bitcast %struct.BSS1* @.BSS1 to i8*, !dbg !39
  %10 = getelementptr i8, i8* %9, i64 3996, !dbg !39
  %11 = bitcast i8* %10 to i32*, !dbg !39
  %12 = getelementptr i32, i32* %11, i64 %8, !dbg !39
  store i32 %6, i32* %12, align 4, !dbg !39
  %13 = load i32, i32* %i_306, align 4, !dbg !40
  call void @llvm.dbg.value(metadata i32 %13, metadata !38, metadata !DIExpression()), !dbg !26
  %14 = add nsw i32 %13, 1, !dbg !40
  store i32 %14, i32* %i_306, align 4, !dbg !40
  %15 = load i32, i32* %.dY0001_338, align 4, !dbg !40
  %16 = sub nsw i32 %15, 1, !dbg !40
  store i32 %16, i32* %.dY0001_338, align 4, !dbg !40
  %17 = load i32, i32* %.dY0001_338, align 4, !dbg !40
  %18 = icmp sgt i32 %17, 0, !dbg !40
  br i1 %18, label %L.LB1_336, label %L.LB1_337, !dbg !40

L.LB1_337:                                        ; preds = %L.LB1_336, %L.LB1_352
  %19 = bitcast i32* %inlen_307 to i8*, !dbg !41
  %20 = bitcast %astruct.dt66* %.uplevelArgPack0001_367 to i8**, !dbg !41
  store i8* %19, i8** %20, align 8, !dbg !41
  %21 = bitcast i32* %outlen_308 to i8*, !dbg !41
  %22 = bitcast %astruct.dt66* %.uplevelArgPack0001_367 to i8*, !dbg !41
  %23 = getelementptr i8, i8* %22, i64 16, !dbg !41
  %24 = bitcast i8* %23 to i8**, !dbg !41
  store i8* %21, i8** %24, align 8, !dbg !41
  br label %L.LB1_377, !dbg !41

L.LB1_377:                                        ; preds = %L.LB1_337
  %25 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L31_1_ to i64*, !dbg !41
  %26 = bitcast %astruct.dt66* %.uplevelArgPack0001_367 to i64*, !dbg !41
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %25, i64* %26), !dbg !41
  call void (...) @_mp_bcs_nest(), !dbg !42
  %27 = bitcast i32* @.C322_MAIN_ to i8*, !dbg !42
  %28 = bitcast [53 x i8]* @.C320_MAIN_ to i8*, !dbg !42
  %29 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !42
  call void (i8*, i8*, i64, ...) %29(i8* %27, i8* %28, i64 53), !dbg !42
  %30 = bitcast i32* @.C324_MAIN_ to i8*, !dbg !42
  %31 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !42
  %32 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !42
  %33 = bitcast %struct.STATICS1* @.STATICS1 to i8*, !dbg !42
  %34 = bitcast i32 (...)* @f90io_fmtw_inita to i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...)*, !dbg !42
  %35 = call i32 (i8*, i8*, i8*, i8*, i8*, i8*, i64, ...) %34(i8* %30, i8* null, i8* %31, i8* %32, i8* %33, i8* null, i64 0), !dbg !42
  call void @llvm.dbg.declare(metadata i32* %z__io_326, metadata !43, metadata !DIExpression()), !dbg !26
  store i32 %35, i32* %z__io_326, align 4, !dbg !42
  %36 = bitcast %struct.BSS1* @.BSS1 to i8*, !dbg !42
  %37 = getelementptr i8, i8* %36, i64 1996, !dbg !42
  %38 = bitcast i8* %37 to i32*, !dbg !42
  %39 = load i32, i32* %38, align 4, !dbg !42
  %40 = bitcast i32 (...)* @f90io_sc_i_fmt_write to i32 (i32, i32, ...)*, !dbg !42
  %41 = call i32 (i32, i32, ...) %40(i32 %39, i32 25), !dbg !42
  store i32 %41, i32* %z__io_326, align 4, !dbg !42
  %42 = call i32 (...) @f90io_fmtw_end(), !dbg !42
  store i32 %42, i32* %z__io_326, align 4, !dbg !42
  call void (...) @_mp_ecs_nest(), !dbg !42
  ret void, !dbg !31
}

define internal void @__nv_MAIN__F1L31_1_(i32* %__nv_MAIN__F1L31_1Arg0, i64* %__nv_MAIN__F1L31_1Arg1, i64* %__nv_MAIN__F1L31_1Arg2) #0 !dbg !15 {
L.entry:
  %__gtid___nv_MAIN__F1L31_1__429 = alloca i32, align 4
  %.i0000p_317 = alloca i32, align 4
  %i_316 = alloca i32, align 4
  %.du0002p_342 = alloca i32, align 4
  %.de0002p_343 = alloca i32, align 4
  %.di0002p_344 = alloca i32, align 4
  %.ds0002p_345 = alloca i32, align 4
  %.dl0002p_347 = alloca i32, align 4
  %.dl0002p.copy_423 = alloca i32, align 4
  %.de0002p.copy_424 = alloca i32, align 4
  %.ds0002p.copy_425 = alloca i32, align 4
  %.dX0002p_346 = alloca i32, align 4
  %.dY0002p_341 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L31_1Arg0, metadata !44, metadata !DIExpression()), !dbg !45
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L31_1Arg1, metadata !46, metadata !DIExpression()), !dbg !45
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L31_1Arg2, metadata !47, metadata !DIExpression()), !dbg !45
  call void @llvm.dbg.value(metadata i32 1, metadata !48, metadata !DIExpression()), !dbg !45
  call void @llvm.dbg.value(metadata i32 0, metadata !49, metadata !DIExpression()), !dbg !45
  call void @llvm.dbg.value(metadata i32 1, metadata !50, metadata !DIExpression()), !dbg !45
  call void @llvm.dbg.value(metadata i32 0, metadata !51, metadata !DIExpression()), !dbg !45
  call void @llvm.dbg.value(metadata i32 1, metadata !52, metadata !DIExpression()), !dbg !45
  %0 = load i32, i32* %__nv_MAIN__F1L31_1Arg0, align 4, !dbg !53
  store i32 %0, i32* %__gtid___nv_MAIN__F1L31_1__429, align 4, !dbg !53
  br label %L.LB2_414

L.LB2_414:                                        ; preds = %L.entry
  br label %L.LB2_315

L.LB2_315:                                        ; preds = %L.LB2_414
  store i32 0, i32* %.i0000p_317, align 4, !dbg !54
  call void @llvm.dbg.declare(metadata i32* %i_316, metadata !55, metadata !DIExpression()), !dbg !53
  store i32 1, i32* %i_316, align 4, !dbg !54
  %1 = bitcast i64* %__nv_MAIN__F1L31_1Arg2 to i32**, !dbg !54
  %2 = load i32*, i32** %1, align 8, !dbg !54
  %3 = load i32, i32* %2, align 4, !dbg !54
  store i32 %3, i32* %.du0002p_342, align 4, !dbg !54
  %4 = bitcast i64* %__nv_MAIN__F1L31_1Arg2 to i32**, !dbg !54
  %5 = load i32*, i32** %4, align 8, !dbg !54
  %6 = load i32, i32* %5, align 4, !dbg !54
  store i32 %6, i32* %.de0002p_343, align 4, !dbg !54
  store i32 1, i32* %.di0002p_344, align 4, !dbg !54
  %7 = load i32, i32* %.di0002p_344, align 4, !dbg !54
  store i32 %7, i32* %.ds0002p_345, align 4, !dbg !54
  store i32 1, i32* %.dl0002p_347, align 4, !dbg !54
  %8 = load i32, i32* %.dl0002p_347, align 4, !dbg !54
  store i32 %8, i32* %.dl0002p.copy_423, align 4, !dbg !54
  %9 = load i32, i32* %.de0002p_343, align 4, !dbg !54
  store i32 %9, i32* %.de0002p.copy_424, align 4, !dbg !54
  %10 = load i32, i32* %.ds0002p_345, align 4, !dbg !54
  store i32 %10, i32* %.ds0002p.copy_425, align 4, !dbg !54
  %11 = load i32, i32* %__gtid___nv_MAIN__F1L31_1__429, align 4, !dbg !54
  %12 = bitcast i32* %.i0000p_317 to i64*, !dbg !54
  %13 = bitcast i32* %.dl0002p.copy_423 to i64*, !dbg !54
  %14 = bitcast i32* %.de0002p.copy_424 to i64*, !dbg !54
  %15 = bitcast i32* %.ds0002p.copy_425 to i64*, !dbg !54
  %16 = load i32, i32* %.ds0002p.copy_425, align 4, !dbg !54
  call void @__kmpc_for_static_init_4(i64* null, i32 %11, i32 34, i64* %12, i64* %13, i64* %14, i64* %15, i32 %16, i32 1), !dbg !54
  %17 = load i32, i32* %.dl0002p.copy_423, align 4, !dbg !54
  store i32 %17, i32* %.dl0002p_347, align 4, !dbg !54
  %18 = load i32, i32* %.de0002p.copy_424, align 4, !dbg !54
  store i32 %18, i32* %.de0002p_343, align 4, !dbg !54
  %19 = load i32, i32* %.ds0002p.copy_425, align 4, !dbg !54
  store i32 %19, i32* %.ds0002p_345, align 4, !dbg !54
  %20 = load i32, i32* %.dl0002p_347, align 4, !dbg !54
  store i32 %20, i32* %i_316, align 4, !dbg !54
  %21 = load i32, i32* %i_316, align 4, !dbg !54
  call void @llvm.dbg.value(metadata i32 %21, metadata !55, metadata !DIExpression()), !dbg !53
  store i32 %21, i32* %.dX0002p_346, align 4, !dbg !54
  %22 = load i32, i32* %.dX0002p_346, align 4, !dbg !54
  %23 = load i32, i32* %.du0002p_342, align 4, !dbg !54
  %24 = icmp sgt i32 %22, %23, !dbg !54
  br i1 %24, label %L.LB2_340, label %L.LB2_453, !dbg !54

L.LB2_453:                                        ; preds = %L.LB2_315
  %25 = load i32, i32* %.dX0002p_346, align 4, !dbg !54
  store i32 %25, i32* %i_316, align 4, !dbg !54
  %26 = load i32, i32* %.di0002p_344, align 4, !dbg !54
  %27 = load i32, i32* %.de0002p_343, align 4, !dbg !54
  %28 = load i32, i32* %.dX0002p_346, align 4, !dbg !54
  %29 = sub nsw i32 %27, %28, !dbg !54
  %30 = add nsw i32 %26, %29, !dbg !54
  %31 = load i32, i32* %.di0002p_344, align 4, !dbg !54
  %32 = sdiv i32 %30, %31, !dbg !54
  store i32 %32, i32* %.dY0002p_341, align 4, !dbg !54
  %33 = load i32, i32* %.dY0002p_341, align 4, !dbg !54
  %34 = icmp sle i32 %33, 0, !dbg !54
  br i1 %34, label %L.LB2_350, label %L.LB2_349, !dbg !54

L.LB2_349:                                        ; preds = %L.LB2_349, %L.LB2_453
  %35 = load i32, i32* %i_316, align 4, !dbg !56
  call void @llvm.dbg.value(metadata i32 %35, metadata !55, metadata !DIExpression()), !dbg !53
  %36 = sext i32 %35 to i64, !dbg !56
  %37 = bitcast %struct.BSS1* @.BSS1 to i8*, !dbg !56
  %38 = getelementptr i8, i8* %37, i64 3996, !dbg !56
  %39 = bitcast i8* %38 to i32*, !dbg !56
  %40 = getelementptr i32, i32* %39, i64 %36, !dbg !56
  %41 = load i32, i32* %40, align 4, !dbg !56
  %42 = bitcast i64* %__nv_MAIN__F1L31_1Arg2 to i8*, !dbg !56
  %43 = getelementptr i8, i8* %42, i64 16, !dbg !56
  %44 = bitcast i8* %43 to i32**, !dbg !56
  %45 = load i32*, i32** %44, align 8, !dbg !56
  %46 = load i32, i32* %45, align 4, !dbg !56
  %47 = sext i32 %46 to i64, !dbg !56
  %48 = bitcast %struct.BSS1* @.BSS1 to i8*, !dbg !56
  %49 = getelementptr i8, i8* %48, i64 -4, !dbg !56
  %50 = bitcast i8* %49 to i32*, !dbg !56
  %51 = getelementptr i32, i32* %50, i64 %47, !dbg !56
  store i32 %41, i32* %51, align 4, !dbg !56
  %52 = bitcast i64* %__nv_MAIN__F1L31_1Arg2 to i8*, !dbg !57
  %53 = getelementptr i8, i8* %52, i64 16, !dbg !57
  %54 = bitcast i8* %53 to i32**, !dbg !57
  %55 = load i32*, i32** %54, align 8, !dbg !57
  %56 = load i32, i32* %55, align 4, !dbg !57
  %57 = add nsw i32 %56, 1, !dbg !57
  %58 = bitcast i64* %__nv_MAIN__F1L31_1Arg2 to i8*, !dbg !57
  %59 = getelementptr i8, i8* %58, i64 16, !dbg !57
  %60 = bitcast i8* %59 to i32**, !dbg !57
  %61 = load i32*, i32** %60, align 8, !dbg !57
  store i32 %57, i32* %61, align 4, !dbg !57
  %62 = load i32, i32* %.di0002p_344, align 4, !dbg !53
  %63 = load i32, i32* %i_316, align 4, !dbg !53
  call void @llvm.dbg.value(metadata i32 %63, metadata !55, metadata !DIExpression()), !dbg !53
  %64 = add nsw i32 %62, %63, !dbg !53
  store i32 %64, i32* %i_316, align 4, !dbg !53
  %65 = load i32, i32* %.dY0002p_341, align 4, !dbg !53
  %66 = sub nsw i32 %65, 1, !dbg !53
  store i32 %66, i32* %.dY0002p_341, align 4, !dbg !53
  %67 = load i32, i32* %.dY0002p_341, align 4, !dbg !53
  %68 = icmp sgt i32 %67, 0, !dbg !53
  br i1 %68, label %L.LB2_349, label %L.LB2_350, !dbg !53

L.LB2_350:                                        ; preds = %L.LB2_349, %L.LB2_453
  br label %L.LB2_340

L.LB2_340:                                        ; preds = %L.LB2_350, %L.LB2_315
  %69 = load i32, i32* %__gtid___nv_MAIN__F1L31_1__429, align 4, !dbg !53
  call void @__kmpc_for_static_fini(i64* null, i32 %69), !dbg !53
  br label %L.LB2_318

L.LB2_318:                                        ; preds = %L.LB2_340
  ret void, !dbg !53
}

declare void @__kmpc_for_static_fini(i64*, i32) #0

declare void @__kmpc_for_static_init_4(i64*, i32, i32, i64*, i64*, i64*, i64*, i32, i32) #0

declare void @_mp_ecs_nest(...) #0

declare signext i32 @f90io_fmtw_end(...) #0

declare signext i32 @f90io_sc_i_fmt_write(...) #0

declare signext i32 @f90io_fmtw_inita(...) #0

declare void @f90io_src_info03a(...) #0

declare void @_mp_bcs_nest(...) #0

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

!llvm.module.flags = !{!23, !24}
!llvm.dbg.cu = !{!4}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression(DW_OP_plus_uconst, 4000))
!1 = distinct !DIGlobalVariable(name: "input", scope: !2, file: !3, type: !9, isLocal: true, isDefinition: true)
!2 = distinct !DISubprogram(name: "drb018_plusplus_orig_yes", scope: !4, file: !3, line: 16, type: !21, scopeLine: 16, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !4)
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB018-plusplus-orig-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !6, imports: !5)
!5 = !{}
!6 = !{!0, !7, !13, !19}
!7 = !DIGlobalVariableExpression(var: !8, expr: !DIExpression())
!8 = distinct !DIGlobalVariable(name: "output", scope: !2, file: !3, type: !9, isLocal: true, isDefinition: true)
!9 = !DICompositeType(tag: DW_TAG_array_type, baseType: !10, size: 32000, align: 32, elements: !11)
!10 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!11 = !{!12}
!12 = !DISubrange(count: 1000, lowerBound: 1)
!13 = !DIGlobalVariableExpression(var: !14, expr: !DIExpression(DW_OP_plus_uconst, 4000))
!14 = distinct !DIGlobalVariable(name: "input", scope: !15, file: !3, type: !9, isLocal: true, isDefinition: true)
!15 = distinct !DISubprogram(name: "__nv_MAIN__F1L31_1", scope: !4, file: !3, line: 31, type: !16, scopeLine: 31, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !4)
!16 = !DISubroutineType(types: !17)
!17 = !{null, !10, !18, !18}
!18 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!19 = !DIGlobalVariableExpression(var: !20, expr: !DIExpression())
!20 = distinct !DIGlobalVariable(name: "output", scope: !15, file: !3, type: !9, isLocal: true, isDefinition: true)
!21 = !DISubroutineType(cc: DW_CC_program, types: !22)
!22 = !{null}
!23 = !{i32 2, !"Dwarf Version", i32 4}
!24 = !{i32 2, !"Debug Info Version", i32 3}
!25 = !DILocalVariable(name: "omp_sched_static", scope: !2, file: !3, type: !10)
!26 = !DILocation(line: 0, scope: !2)
!27 = !DILocalVariable(name: "omp_proc_bind_false", scope: !2, file: !3, type: !10)
!28 = !DILocalVariable(name: "omp_proc_bind_true", scope: !2, file: !3, type: !10)
!29 = !DILocalVariable(name: "omp_lock_hint_none", scope: !2, file: !3, type: !10)
!30 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !2, file: !3, type: !10)
!31 = !DILocation(line: 40, column: 1, scope: !2)
!32 = !DILocation(line: 16, column: 1, scope: !2)
!33 = !DILocalVariable(name: "inlen", scope: !2, file: !3, type: !10)
!34 = !DILocation(line: 24, column: 1, scope: !2)
!35 = !DILocalVariable(name: "outlen", scope: !2, file: !3, type: !10)
!36 = !DILocation(line: 25, column: 1, scope: !2)
!37 = !DILocation(line: 27, column: 1, scope: !2)
!38 = !DILocalVariable(name: "i", scope: !2, file: !3, type: !10)
!39 = !DILocation(line: 28, column: 1, scope: !2)
!40 = !DILocation(line: 29, column: 1, scope: !2)
!41 = !DILocation(line: 31, column: 1, scope: !2)
!42 = !DILocation(line: 38, column: 1, scope: !2)
!43 = !DILocalVariable(scope: !2, file: !3, type: !10, flags: DIFlagArtificial)
!44 = !DILocalVariable(name: "__nv_MAIN__F1L31_1Arg0", arg: 1, scope: !15, file: !3, type: !10)
!45 = !DILocation(line: 0, scope: !15)
!46 = !DILocalVariable(name: "__nv_MAIN__F1L31_1Arg1", arg: 2, scope: !15, file: !3, type: !18)
!47 = !DILocalVariable(name: "__nv_MAIN__F1L31_1Arg2", arg: 3, scope: !15, file: !3, type: !18)
!48 = !DILocalVariable(name: "omp_sched_static", scope: !15, file: !3, type: !10)
!49 = !DILocalVariable(name: "omp_proc_bind_false", scope: !15, file: !3, type: !10)
!50 = !DILocalVariable(name: "omp_proc_bind_true", scope: !15, file: !3, type: !10)
!51 = !DILocalVariable(name: "omp_lock_hint_none", scope: !15, file: !3, type: !10)
!52 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !15, file: !3, type: !10)
!53 = !DILocation(line: 35, column: 1, scope: !15)
!54 = !DILocation(line: 32, column: 1, scope: !15)
!55 = !DILocalVariable(name: "i", scope: !15, file: !3, type: !10)
!56 = !DILocation(line: 33, column: 1, scope: !15)
!57 = !DILocation(line: 34, column: 1, scope: !15)

; ModuleID = '/tmp/DRB119-nestlock-orig-yes-96cef7.ll'
source_filename = "/tmp/DRB119-nestlock-orig-yes-96cef7.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%structdrb118_pair_td_ = type <{ [8 x i64], [6 x i8*], [14 x i8] }>
%struct.BSS4 = type <{ [16 x i8] }>
%astruct.dt85 = type <{ i8*, i8*, i8* }>

@drb118_pair_td_ = global %structdrb118_pair_td_ <{ [8 x i64] [i64 43, i64 33, i64 0, i64 16, i64 0, i64 0, i64 0, i64 0], [6 x i8*] [i8* null, i8* bitcast (%structdrb118_pair_td_* @drb118_pair_td_ to i8*), i8* null, i8* null, i8* null, i8* null], [14 x i8] c"drb118$pair$td" }>
@.C285_incr_a_ = internal constant i32 1
@.C285_incr_b_ = internal constant i32 1
@.BSS4 = internal global %struct.BSS4 zeroinitializer, align 32, !dbg !0
@.C313_MAIN_ = internal constant i32 25
@.C284_MAIN_ = internal constant i64 0
@.C340_MAIN_ = internal constant i32 6
@.C337_MAIN_ = internal constant [53 x i8] c"micro-benchmarks-fortran/DRB119-nestlock-orig-yes.f95"
@.C339_MAIN_ = internal constant i32 64
@.C302_MAIN_ = internal constant i32 3
@.C301_MAIN_ = internal constant i32 2
@.C285_MAIN_ = internal constant i32 1
@.C283_MAIN_ = internal constant i32 0
@.C302___nv_MAIN__F1L50_1 = internal constant i32 3
@.C301___nv_MAIN__F1L50_1 = internal constant i32 2
@.C285___nv_MAIN__F1L50_1 = internal constant i32 1
@.C283___nv_MAIN__F1L50_1 = internal constant i32 0

; Function Attrs: noinline
define float @drb118_() #0 {
.L.entry:
  ret float undef
}

define void @incr_a_(i64* %p, i64* %a) #1 !dbg !60 {
L.entry:
  call void @llvm.dbg.declare(metadata i64* %p, metadata !63, metadata !DIExpression()), !dbg !64
  call void @llvm.dbg.declare(metadata i64* %a, metadata !65, metadata !DIExpression()), !dbg !64
  call void @llvm.dbg.value(metadata i32 1, metadata !66, metadata !DIExpression()), !dbg !64
  call void @llvm.dbg.value(metadata i32 0, metadata !67, metadata !DIExpression()), !dbg !64
  call void @llvm.dbg.value(metadata i32 1, metadata !68, metadata !DIExpression()), !dbg !64
  call void @llvm.dbg.value(metadata i32 0, metadata !69, metadata !DIExpression()), !dbg !64
  call void @llvm.dbg.value(metadata i32 1, metadata !70, metadata !DIExpression()), !dbg !64
  br label %L.LB2_316

L.LB2_316:                                        ; preds = %L.entry
  %0 = bitcast i64* %p to i32*, !dbg !71
  %1 = load i32, i32* %0, align 4, !dbg !71
  %2 = add nsw i32 %1, 1, !dbg !71
  %3 = bitcast i64* %p to i32*, !dbg !71
  store i32 %2, i32* %3, align 4, !dbg !71
  ret void, !dbg !72
}

define void @incr_b_(i64* %p, i64* %b) #1 !dbg !73 {
L.entry:
  call void @llvm.dbg.declare(metadata i64* %p, metadata !76, metadata !DIExpression()), !dbg !77
  call void @llvm.dbg.declare(metadata i64* %b, metadata !78, metadata !DIExpression()), !dbg !77
  call void @llvm.dbg.value(metadata i32 1, metadata !79, metadata !DIExpression()), !dbg !77
  call void @llvm.dbg.value(metadata i32 0, metadata !80, metadata !DIExpression()), !dbg !77
  call void @llvm.dbg.value(metadata i32 1, metadata !81, metadata !DIExpression()), !dbg !77
  call void @llvm.dbg.value(metadata i32 0, metadata !82, metadata !DIExpression()), !dbg !77
  call void @llvm.dbg.value(metadata i32 1, metadata !83, metadata !DIExpression()), !dbg !77
  br label %L.LB3_317

L.LB3_317:                                        ; preds = %L.entry
  %0 = bitcast i64* %p to i8*, !dbg !84
  %1 = getelementptr i8, i8* %0, i64 4, !dbg !84
  %2 = bitcast i8* %1 to i32*, !dbg !84
  %3 = load i32, i32* %2, align 4, !dbg !84
  %4 = add nsw i32 %3, 1, !dbg !84
  %5 = bitcast i64* %p to i8*, !dbg !84
  %6 = getelementptr i8, i8* %5, i64 4, !dbg !84
  %7 = bitcast i8* %6 to i32*, !dbg !84
  store i32 %4, i32* %7, align 4, !dbg !84
  ret void, !dbg !85
}

define void @MAIN_() #1 !dbg !2 {
L.entry:
  %__gtid_MAIN__378 = alloca i32, align 4
  %a_327 = alloca i32, align 4
  %.uplevelArgPack0001_367 = alloca %astruct.dt85, align 16
  %b_328 = alloca i32, align 4
  %z__io_342 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 1, metadata !86, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 2, metadata !88, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 3, metadata !89, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 0, metadata !90, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 1, metadata !91, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 2, metadata !92, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 3, metadata !93, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 0, metadata !94, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 1, metadata !95, metadata !DIExpression()), !dbg !87
  call void @llvm.dbg.value(metadata i32 2, metadata !96, metadata !DIExpression()), !dbg !87
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !97
  store i32 %0, i32* %__gtid_MAIN__378, align 4, !dbg !97
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !98
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !98
  call void (i8*, ...) %2(i8* %1), !dbg !98
  br label %L.LB4_356

L.LB4_356:                                        ; preds = %L.entry
  %3 = bitcast %struct.BSS4* @.BSS4 to i32*, !dbg !99
  store i32 0, i32* %3, align 4, !dbg !99
  %4 = bitcast %struct.BSS4* @.BSS4 to i8*, !dbg !100
  %5 = getelementptr i8, i8* %4, i64 4, !dbg !100
  %6 = bitcast i8* %5 to i32*, !dbg !100
  store i32 0, i32* %6, align 4, !dbg !100
  %7 = bitcast %struct.BSS4* @.BSS4 to i8*, !dbg !101
  %8 = getelementptr i8, i8* %7, i64 8, !dbg !101
  %9 = bitcast i8* %8 to i64*, !dbg !101
  call void @omp_init_nest_lock_(i64* %9), !dbg !101
  call void @llvm.dbg.declare(metadata i32* %a_327, metadata !102, metadata !DIExpression()), !dbg !87
  %10 = bitcast i32* %a_327 to i8*, !dbg !103
  %11 = bitcast %astruct.dt85* %.uplevelArgPack0001_367 to i8*, !dbg !103
  %12 = getelementptr i8, i8* %11, i64 8, !dbg !103
  %13 = bitcast i8* %12 to i8**, !dbg !103
  store i8* %10, i8** %13, align 8, !dbg !103
  call void @llvm.dbg.declare(metadata i32* %b_328, metadata !104, metadata !DIExpression()), !dbg !87
  %14 = bitcast i32* %b_328 to i8*, !dbg !103
  %15 = bitcast %astruct.dt85* %.uplevelArgPack0001_367 to i8*, !dbg !103
  %16 = getelementptr i8, i8* %15, i64 16, !dbg !103
  %17 = bitcast i8* %16 to i8**, !dbg !103
  store i8* %14, i8** %17, align 8, !dbg !103
  br label %L.LB4_376, !dbg !103

L.LB4_376:                                        ; preds = %L.LB4_356
  %18 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L50_1_ to i64*, !dbg !103
  %19 = bitcast %astruct.dt85* %.uplevelArgPack0001_367 to i64*, !dbg !103
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %18, i64* %19), !dbg !103
  %20 = bitcast %struct.BSS4* @.BSS4 to i8*, !dbg !105
  %21 = getelementptr i8, i8* %20, i64 8, !dbg !105
  %22 = bitcast i8* %21 to i64*, !dbg !105
  call void @omp_destroy_nest_lock_(i64* %22), !dbg !105
  call void (...) @_mp_bcs_nest(), !dbg !106
  %23 = bitcast i32* @.C339_MAIN_ to i8*, !dbg !106
  %24 = bitcast [53 x i8]* @.C337_MAIN_ to i8*, !dbg !106
  %25 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !106
  call void (i8*, i8*, i64, ...) %25(i8* %23, i8* %24, i64 53), !dbg !106
  %26 = bitcast i32* @.C340_MAIN_ to i8*, !dbg !106
  %27 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !106
  %28 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !106
  %29 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !106
  %30 = call i32 (i8*, i8*, i8*, i8*, ...) %29(i8* %26, i8* null, i8* %27, i8* %28), !dbg !106
  call void @llvm.dbg.declare(metadata i32* %z__io_342, metadata !107, metadata !DIExpression()), !dbg !87
  store i32 %30, i32* %z__io_342, align 4, !dbg !106
  %31 = bitcast %struct.BSS4* @.BSS4 to i8*, !dbg !106
  %32 = getelementptr i8, i8* %31, i64 4, !dbg !106
  %33 = bitcast i8* %32 to i32*, !dbg !106
  %34 = load i32, i32* %33, align 4, !dbg !106
  %35 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !106
  %36 = call i32 (i32, i32, ...) %35(i32 %34, i32 25), !dbg !106
  store i32 %36, i32* %z__io_342, align 4, !dbg !106
  %37 = call i32 (...) @f90io_ldw_end(), !dbg !106
  store i32 %37, i32* %z__io_342, align 4, !dbg !106
  call void (...) @_mp_ecs_nest(), !dbg !106
  ret void, !dbg !97
}

define internal void @__nv_MAIN__F1L50_1_(i32* %__nv_MAIN__F1L50_1Arg0, i64* %__nv_MAIN__F1L50_1Arg1, i64* %__nv_MAIN__F1L50_1Arg2) #1 !dbg !45 {
L.entry:
  %__gtid___nv_MAIN__F1L50_1__412 = alloca i32, align 4
  %.s0001_407 = alloca i32, align 4
  %.s0000_406 = alloca i32, align 4
  %.s0003_409 = alloca i32, align 4
  %.s0002_408 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L50_1Arg0, metadata !108, metadata !DIExpression()), !dbg !109
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L50_1Arg1, metadata !110, metadata !DIExpression()), !dbg !109
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L50_1Arg2, metadata !111, metadata !DIExpression()), !dbg !109
  call void @llvm.dbg.value(metadata i32 1, metadata !112, metadata !DIExpression()), !dbg !109
  call void @llvm.dbg.value(metadata i32 2, metadata !113, metadata !DIExpression()), !dbg !109
  call void @llvm.dbg.value(metadata i32 3, metadata !114, metadata !DIExpression()), !dbg !109
  call void @llvm.dbg.value(metadata i32 0, metadata !115, metadata !DIExpression()), !dbg !109
  call void @llvm.dbg.value(metadata i32 1, metadata !116, metadata !DIExpression()), !dbg !109
  call void @llvm.dbg.value(metadata i32 2, metadata !117, metadata !DIExpression()), !dbg !109
  call void @llvm.dbg.value(metadata i32 3, metadata !118, metadata !DIExpression()), !dbg !109
  call void @llvm.dbg.value(metadata i32 0, metadata !119, metadata !DIExpression()), !dbg !109
  call void @llvm.dbg.value(metadata i32 1, metadata !120, metadata !DIExpression()), !dbg !109
  call void @llvm.dbg.value(metadata i32 2, metadata !121, metadata !DIExpression()), !dbg !109
  %0 = load i32, i32* %__nv_MAIN__F1L50_1Arg0, align 4, !dbg !122
  store i32 %0, i32* %__gtid___nv_MAIN__F1L50_1__412, align 4, !dbg !122
  br label %L.LB5_405

L.LB5_405:                                        ; preds = %L.entry
  br label %L.LB5_332

L.LB5_332:                                        ; preds = %L.LB5_405
  store i32 2, i32* %.s0001_407, align 4, !dbg !122
  store i32 0, i32* %.s0000_406, align 4, !dbg !123
  store i32 1, i32* %.s0003_409, align 4, !dbg !123
  %1 = load i32, i32* %__gtid___nv_MAIN__F1L50_1__412, align 4, !dbg !123
  %2 = bitcast i32* %.s0002_408 to i64*, !dbg !123
  %3 = bitcast i32* %.s0000_406 to i64*, !dbg !123
  %4 = bitcast i32* %.s0001_407 to i64*, !dbg !123
  %5 = bitcast i32* %.s0003_409 to i64*, !dbg !123
  call void @__kmpc_for_static_init_4(i64* null, i32 %1, i32 34, i64* %2, i64* %3, i64* %4, i64* %5, i32 1, i32 0), !dbg !123
  br label %L.LB5_350

L.LB5_350:                                        ; preds = %L.LB5_332
  %6 = load i32, i32* %.s0000_406, align 4, !dbg !123
  %7 = icmp ne i32 %6, 0, !dbg !123
  br i1 %7, label %L.LB5_351, label %L.LB5_437, !dbg !123

L.LB5_437:                                        ; preds = %L.LB5_350
  br label %L.LB5_351

L.LB5_351:                                        ; preds = %L.LB5_437, %L.LB5_350
  %8 = load i32, i32* %.s0001_407, align 4, !dbg !124
  %9 = icmp ugt i32 1, %8, !dbg !124
  br i1 %9, label %L.LB5_352, label %L.LB5_438, !dbg !124

L.LB5_438:                                        ; preds = %L.LB5_351
  %10 = load i32, i32* %.s0000_406, align 4, !dbg !124
  %11 = icmp ult i32 1, %10, !dbg !124
  br i1 %11, label %L.LB5_352, label %L.LB5_439, !dbg !124

L.LB5_439:                                        ; preds = %L.LB5_438
  %12 = bitcast %struct.BSS4* @.BSS4 to i8*, !dbg !125
  %13 = getelementptr i8, i8* %12, i64 8, !dbg !125
  %14 = bitcast i8* %13 to i64*, !dbg !125
  call void @omp_set_nest_lock_(i64* %14), !dbg !125
  %15 = bitcast %struct.BSS4* @.BSS4 to i64*, !dbg !126
  %16 = bitcast i64* %__nv_MAIN__F1L50_1Arg2 to i8*, !dbg !126
  %17 = getelementptr i8, i8* %16, i64 8, !dbg !126
  %18 = bitcast i8* %17 to i64**, !dbg !126
  %19 = load i64*, i64** %18, align 8, !dbg !126
  call void @incr_b_(i64* %15, i64* %19), !dbg !126
  %20 = bitcast %struct.BSS4* @.BSS4 to i64*, !dbg !127
  %21 = bitcast i64* %__nv_MAIN__F1L50_1Arg2 to i8*, !dbg !127
  %22 = getelementptr i8, i8* %21, i64 16, !dbg !127
  %23 = bitcast i8* %22 to i64**, !dbg !127
  %24 = load i64*, i64** %23, align 8, !dbg !127
  call void @incr_a_(i64* %20, i64* %24), !dbg !127
  %25 = bitcast %struct.BSS4* @.BSS4 to i8*, !dbg !128
  %26 = getelementptr i8, i8* %25, i64 8, !dbg !128
  %27 = bitcast i8* %26 to i64*, !dbg !128
  call void @omp_unset_nest_lock_(i64* %27), !dbg !128
  br label %L.LB5_352

L.LB5_352:                                        ; preds = %L.LB5_439, %L.LB5_438, %L.LB5_351
  %28 = load i32, i32* %.s0001_407, align 4, !dbg !129
  %29 = icmp ugt i32 2, %28, !dbg !129
  br i1 %29, label %L.LB5_353, label %L.LB5_440, !dbg !129

L.LB5_440:                                        ; preds = %L.LB5_352
  %30 = load i32, i32* %.s0000_406, align 4, !dbg !129
  %31 = icmp ult i32 2, %30, !dbg !129
  br i1 %31, label %L.LB5_353, label %L.LB5_441, !dbg !129

L.LB5_441:                                        ; preds = %L.LB5_440
  %32 = bitcast %struct.BSS4* @.BSS4 to i64*, !dbg !130
  %33 = bitcast i64* %__nv_MAIN__F1L50_1Arg2 to i8*, !dbg !130
  %34 = getelementptr i8, i8* %33, i64 16, !dbg !130
  %35 = bitcast i8* %34 to i64**, !dbg !130
  %36 = load i64*, i64** %35, align 8, !dbg !130
  call void @incr_b_(i64* %32, i64* %36), !dbg !130
  br label %L.LB5_353

L.LB5_353:                                        ; preds = %L.LB5_441, %L.LB5_440, %L.LB5_352
  br label %L.LB5_354

L.LB5_354:                                        ; preds = %L.LB5_353
  br label %L.LB5_335

L.LB5_335:                                        ; preds = %L.LB5_354
  ret void, !dbg !122
}

declare void @__kmpc_for_static_init_4(i64*, i32, i32, i64*, i64*, i64*, i64*, i32 zeroext, i32 zeroext) #1

declare void @_mp_ecs_nest(...) #1

declare signext i32 @f90io_ldw_end(...) #1

declare signext i32 @f90io_sc_i_ldw(...) #1

declare signext i32 @f90io_print_init(...) #1

declare void @f90io_src_info03a(...) #1

declare void @_mp_bcs_nest(...) #1

declare void @fort_init(...) #1

declare signext i32 @__kmpc_global_thread_num(i64*) #1

declare void @__kmpc_fork_call(i64*, i32, i64*, i64*, ...) #1

declare void @__kmpc_end_serialized_parallel(i64*, i32) #1

declare void @__kmpc_serialized_parallel(i64*, i32) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

declare void @omp_unset_nest_lock_(i64*) #1

declare void @omp_set_nest_lock_(i64*) #1

declare void @omp_init_nest_lock_(i64*) #1

declare void @omp_destroy_nest_lock_(i64*) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { noinline }
attributes #1 = { "no-frame-pointer-elim-non-leaf" }
attributes #2 = { nounwind readnone speculatable }

!llvm.module.flags = !{!58, !59}
!llvm.dbg.cu = !{!4}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "p", scope: !2, file: !3, type: !38, isLocal: true, isDefinition: true)
!2 = distinct !DISubprogram(name: "drb118_nestlock_orig_no", scope: !4, file: !3, line: 38, type: !56, scopeLine: 38, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !4)
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB119-nestlock-orig-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !6, imports: !5)
!5 = !{}
!6 = !{!7, !19, !27, !0, !35, !43, !53}
!7 = !DIGlobalVariableExpression(var: !8, expr: !DIExpression())
!8 = distinct !DIGlobalVariable(name: "drb118$pair$td", scope: !4, file: !3, type: !9, isLocal: false, isDefinition: true)
!9 = !DICompositeType(tag: DW_TAG_array_type, baseType: !10, size: 128, align: 64, elements: !17)
!10 = !DICompositeType(tag: DW_TAG_structure_type, name: "pair", file: !3, size: 128, align: 64, elements: !11)
!11 = !{!12, !14, !15}
!12 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !10, file: !3, baseType: !13, size: 32, align: 32)
!13 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!14 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !10, file: !3, baseType: !13, size: 32, align: 32, offset: 32)
!15 = !DIDerivedType(tag: DW_TAG_member, name: "lck", scope: !10, file: !3, baseType: !16, size: 64, align: 64, offset: 64)
!16 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!17 = !{!18}
!18 = !DISubrange(count: 0, lowerBound: 1)
!19 = !DIGlobalVariableExpression(var: !20, expr: !DIExpression())
!20 = distinct !DIGlobalVariable(name: "drb118$pair$td", scope: !4, file: !3, type: !21, isLocal: false, isDefinition: true)
!21 = !DICompositeType(tag: DW_TAG_array_type, baseType: !22, size: 128, align: 64, elements: !17)
!22 = !DICompositeType(tag: DW_TAG_structure_type, name: "pair", file: !3, size: 128, align: 64, elements: !23)
!23 = !{!24, !25, !26}
!24 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !22, file: !3, baseType: !13, size: 32, align: 32)
!25 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !22, file: !3, baseType: !13, size: 32, align: 32, offset: 32)
!26 = !DIDerivedType(tag: DW_TAG_member, name: "lck", scope: !22, file: !3, baseType: !16, size: 64, align: 64, offset: 64)
!27 = !DIGlobalVariableExpression(var: !28, expr: !DIExpression())
!28 = distinct !DIGlobalVariable(name: "drb118$pair$td", scope: !4, file: !3, type: !29, isLocal: false, isDefinition: true)
!29 = !DICompositeType(tag: DW_TAG_array_type, baseType: !30, size: 128, align: 64, elements: !17)
!30 = !DICompositeType(tag: DW_TAG_structure_type, name: "pair", file: !3, size: 128, align: 64, elements: !31)
!31 = !{!32, !33, !34}
!32 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !30, file: !3, baseType: !13, size: 32, align: 32)
!33 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !30, file: !3, baseType: !13, size: 32, align: 32, offset: 32)
!34 = !DIDerivedType(tag: DW_TAG_member, name: "lck", scope: !30, file: !3, baseType: !16, size: 64, align: 64, offset: 64)
!35 = !DIGlobalVariableExpression(var: !36, expr: !DIExpression())
!36 = distinct !DIGlobalVariable(name: "drb118$pair$td", scope: !4, file: !3, type: !37, isLocal: false, isDefinition: true)
!37 = !DICompositeType(tag: DW_TAG_array_type, baseType: !38, size: 128, align: 64, elements: !17)
!38 = !DICompositeType(tag: DW_TAG_structure_type, name: "pair", file: !3, size: 128, align: 64, elements: !39)
!39 = !{!40, !41, !42}
!40 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !38, file: !3, baseType: !13, size: 32, align: 32)
!41 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !38, file: !3, baseType: !13, size: 32, align: 32, offset: 32)
!42 = !DIDerivedType(tag: DW_TAG_member, name: "lck", scope: !38, file: !3, baseType: !16, size: 64, align: 64, offset: 64)
!43 = !DIGlobalVariableExpression(var: !44, expr: !DIExpression())
!44 = distinct !DIGlobalVariable(name: "p", scope: !45, file: !3, type: !48, isLocal: true, isDefinition: true)
!45 = distinct !DISubprogram(name: "__nv_MAIN__F1L50_1", scope: !4, file: !3, line: 50, type: !46, scopeLine: 50, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !4)
!46 = !DISubroutineType(types: !47)
!47 = !{null, !13, !16, !16}
!48 = !DICompositeType(tag: DW_TAG_structure_type, name: "pair", file: !3, size: 128, align: 64, elements: !49)
!49 = !{!50, !51, !52}
!50 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !48, file: !3, baseType: !13, size: 32, align: 32)
!51 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !48, file: !3, baseType: !13, size: 32, align: 32, offset: 32)
!52 = !DIDerivedType(tag: DW_TAG_member, name: "lck", scope: !48, file: !3, baseType: !16, size: 64, align: 64, offset: 64)
!53 = !DIGlobalVariableExpression(var: !54, expr: !DIExpression())
!54 = distinct !DIGlobalVariable(name: "drb118$pair$td", scope: !4, file: !3, type: !55, isLocal: false, isDefinition: true)
!55 = !DICompositeType(tag: DW_TAG_array_type, baseType: !48, size: 128, align: 64, elements: !17)
!56 = !DISubroutineType(cc: DW_CC_program, types: !57)
!57 = !{null}
!58 = !{i32 2, !"Dwarf Version", i32 4}
!59 = !{i32 2, !"Debug Info Version", i32 3}
!60 = distinct !DISubprogram(name: "incr_a", scope: !4, file: !3, line: 23, type: !61, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !4)
!61 = !DISubroutineType(types: !62)
!62 = !{null, !10, !13}
!63 = !DILocalVariable(name: "p", arg: 1, scope: !60, file: !3, type: !22)
!64 = !DILocation(line: 0, scope: !60)
!65 = !DILocalVariable(name: "a", arg: 2, scope: !60, file: !3, type: !13)
!66 = !DILocalVariable(name: "omp_sched_static", scope: !60, file: !3, type: !13)
!67 = !DILocalVariable(name: "omp_proc_bind_false", scope: !60, file: !3, type: !13)
!68 = !DILocalVariable(name: "omp_proc_bind_true", scope: !60, file: !3, type: !13)
!69 = !DILocalVariable(name: "omp_lock_hint_none", scope: !60, file: !3, type: !13)
!70 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !60, file: !3, type: !13)
!71 = !DILocation(line: 27, column: 1, scope: !60)
!72 = !DILocation(line: 28, column: 1, scope: !60)
!73 = distinct !DISubprogram(name: "incr_b", scope: !4, file: !3, line: 30, type: !74, scopeLine: 30, spFlags: DISPFlagDefinition, unit: !4)
!74 = !DISubroutineType(types: !75)
!75 = !{null, !22, !13}
!76 = !DILocalVariable(name: "p", arg: 1, scope: !73, file: !3, type: !30)
!77 = !DILocation(line: 0, scope: !73)
!78 = !DILocalVariable(name: "b", arg: 2, scope: !73, file: !3, type: !13)
!79 = !DILocalVariable(name: "omp_sched_static", scope: !73, file: !3, type: !13)
!80 = !DILocalVariable(name: "omp_proc_bind_false", scope: !73, file: !3, type: !13)
!81 = !DILocalVariable(name: "omp_proc_bind_true", scope: !73, file: !3, type: !13)
!82 = !DILocalVariable(name: "omp_lock_hint_none", scope: !73, file: !3, type: !13)
!83 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !73, file: !3, type: !13)
!84 = !DILocation(line: 35, column: 1, scope: !73)
!85 = !DILocation(line: 36, column: 1, scope: !73)
!86 = !DILocalVariable(name: "omp_sched_static", scope: !2, file: !3, type: !13)
!87 = !DILocation(line: 0, scope: !2)
!88 = !DILocalVariable(name: "omp_sched_dynamic", scope: !2, file: !3, type: !13)
!89 = !DILocalVariable(name: "omp_sched_guided", scope: !2, file: !3, type: !13)
!90 = !DILocalVariable(name: "omp_proc_bind_false", scope: !2, file: !3, type: !13)
!91 = !DILocalVariable(name: "omp_proc_bind_true", scope: !2, file: !3, type: !13)
!92 = !DILocalVariable(name: "omp_proc_bind_master", scope: !2, file: !3, type: !13)
!93 = !DILocalVariable(name: "omp_proc_bind_close", scope: !2, file: !3, type: !13)
!94 = !DILocalVariable(name: "omp_lock_hint_none", scope: !2, file: !3, type: !13)
!95 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !2, file: !3, type: !13)
!96 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !2, file: !3, type: !13)
!97 = !DILocation(line: 66, column: 1, scope: !2)
!98 = !DILocation(line: 38, column: 1, scope: !2)
!99 = !DILocation(line: 46, column: 1, scope: !2)
!100 = !DILocation(line: 47, column: 1, scope: !2)
!101 = !DILocation(line: 48, column: 1, scope: !2)
!102 = !DILocalVariable(name: "a", scope: !2, file: !3, type: !13)
!103 = !DILocation(line: 50, column: 1, scope: !2)
!104 = !DILocalVariable(name: "b", scope: !2, file: !3, type: !13)
!105 = !DILocation(line: 62, column: 1, scope: !2)
!106 = !DILocation(line: 64, column: 1, scope: !2)
!107 = !DILocalVariable(scope: !2, file: !3, type: !13, flags: DIFlagArtificial)
!108 = !DILocalVariable(name: "__nv_MAIN__F1L50_1Arg0", arg: 1, scope: !45, file: !3, type: !13)
!109 = !DILocation(line: 0, scope: !45)
!110 = !DILocalVariable(name: "__nv_MAIN__F1L50_1Arg1", arg: 2, scope: !45, file: !3, type: !16)
!111 = !DILocalVariable(name: "__nv_MAIN__F1L50_1Arg2", arg: 3, scope: !45, file: !3, type: !16)
!112 = !DILocalVariable(name: "omp_sched_static", scope: !45, file: !3, type: !13)
!113 = !DILocalVariable(name: "omp_sched_dynamic", scope: !45, file: !3, type: !13)
!114 = !DILocalVariable(name: "omp_sched_guided", scope: !45, file: !3, type: !13)
!115 = !DILocalVariable(name: "omp_proc_bind_false", scope: !45, file: !3, type: !13)
!116 = !DILocalVariable(name: "omp_proc_bind_true", scope: !45, file: !3, type: !13)
!117 = !DILocalVariable(name: "omp_proc_bind_master", scope: !45, file: !3, type: !13)
!118 = !DILocalVariable(name: "omp_proc_bind_close", scope: !45, file: !3, type: !13)
!119 = !DILocalVariable(name: "omp_lock_hint_none", scope: !45, file: !3, type: !13)
!120 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !45, file: !3, type: !13)
!121 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !45, file: !3, type: !13)
!122 = !DILocation(line: 60, column: 1, scope: !45)
!123 = !DILocation(line: 50, column: 1, scope: !45)
!124 = !DILocation(line: 51, column: 1, scope: !45)
!125 = !DILocation(line: 52, column: 1, scope: !45)
!126 = !DILocation(line: 53, column: 1, scope: !45)
!127 = !DILocation(line: 54, column: 1, scope: !45)
!128 = !DILocation(line: 55, column: 1, scope: !45)
!129 = !DILocation(line: 57, column: 1, scope: !45)
!130 = !DILocation(line: 58, column: 1, scope: !45)

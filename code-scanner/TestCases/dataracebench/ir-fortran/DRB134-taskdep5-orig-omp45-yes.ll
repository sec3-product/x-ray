; ModuleID = '/tmp/DRB134-taskdep5-orig-omp45-yes-acc0be.ll'
source_filename = "/tmp/DRB134-taskdep5-orig-omp45-yes-acc0be.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.struct_ul_MAIN__297 = type <{ i8* }>
%astruct.dt62 = type <{ i8* }>

@.C285_MAIN_ = internal constant i32 1
@.C283_MAIN_ = internal constant i32 0
@.C285___nv_MAIN__F1L17_1 = internal constant i32 1
@.C340_drb134_taskdep4_orig_yes_omp_45_foo = internal constant [2 x i8] c"y="
@.C339_drb134_taskdep4_orig_yes_omp_45_foo = internal constant i32 41
@.C308_drb134_taskdep4_orig_yes_omp_45_foo = internal constant i32 25
@.C307_drb134_taskdep4_orig_yes_omp_45_foo = internal constant i32 14
@.C333_drb134_taskdep4_orig_yes_omp_45_foo = internal constant [2 x i8] c"x="
@.C284_drb134_taskdep4_orig_yes_omp_45_foo = internal constant i64 0
@.C330_drb134_taskdep4_orig_yes_omp_45_foo = internal constant i32 6
@.C327_drb134_taskdep4_orig_yes_omp_45_foo = internal constant [59 x i8] c"micro-benchmarks-fortran/DRB134-taskdep5-orig-omp45-yes.f95"
@.C329_drb134_taskdep4_orig_yes_omp_45_foo = internal constant i32 40
@.C285_drb134_taskdep4_orig_yes_omp_45_foo = internal constant i32 1
@.C298_drb134_taskdep4_orig_yes_omp_45_foo = internal constant i32 2
@.C283_drb134_taskdep4_orig_yes_omp_45_foo = internal constant i32 0
@.C285___nv_drb134_taskdep4_orig_yes_omp_45_foo_F1L29_2 = internal constant i32 1

define void @MAIN_() #0 !dbg !5 {
L.entry:
  %__gtid_MAIN__332 = alloca i32, align 4
  %.S0000_317 = alloca %struct.struct_ul_MAIN__297, align 8
  %.uplevelArgPack0001_326 = alloca %astruct.dt62, align 8
  call void @llvm.dbg.value(metadata i32 1, metadata !8, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !11, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !12, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 0, metadata !13, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata i32 1, metadata !14, metadata !DIExpression()), !dbg !10
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !15
  store i32 %0, i32* %__gtid_MAIN__332, align 4, !dbg !15
  %1 = bitcast i32* @.C283_MAIN_ to i8*, !dbg !16
  %2 = bitcast void (...)* @fort_init to void (i8*, ...)*, !dbg !16
  call void (i8*, ...) %2(i8* %1), !dbg !16
  br label %L.LB1_321

L.LB1_321:                                        ; preds = %L.entry
  %3 = bitcast %struct.struct_ul_MAIN__297* %.S0000_317 to i8*, !dbg !17
  %4 = bitcast %astruct.dt62* %.uplevelArgPack0001_326 to i8**, !dbg !17
  store i8* %3, i8** %4, align 8, !dbg !17
  br label %L.LB1_330, !dbg !17

L.LB1_330:                                        ; preds = %L.LB1_321
  %5 = bitcast void (i32*, i64*, i64*)* @__nv_MAIN__F1L17_1_ to i64*, !dbg !17
  %6 = bitcast %astruct.dt62* %.uplevelArgPack0001_326 to i64*, !dbg !17
  call void (i64*, i32, i64*, i64*, ...) @__kmpc_fork_call(i64* null, i32 1, i64* %5, i64* %6), !dbg !17
  ret void, !dbg !15
}

define internal void @__nv_MAIN__F1L17_1_(i32* %__nv_MAIN__F1L17_1Arg0, i64* %__nv_MAIN__F1L17_1Arg1, i64* %__nv_MAIN__F1L17_1Arg2) #0 !dbg !18 {
L.entry:
  %.S0000_317 = alloca i8*, align 8
  %__gtid___nv_MAIN__F1L17_1__360 = alloca i32, align 4
  %.s0000_355 = alloca i32, align 4
  %.s0001_356 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_MAIN__F1L17_1Arg0, metadata !22, metadata !DIExpression()), !dbg !23
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L17_1Arg1, metadata !24, metadata !DIExpression()), !dbg !23
  call void @llvm.dbg.declare(metadata i64* %__nv_MAIN__F1L17_1Arg2, metadata !25, metadata !DIExpression()), !dbg !23
  call void @llvm.dbg.value(metadata i32 1, metadata !26, metadata !DIExpression()), !dbg !23
  call void @llvm.dbg.value(metadata i32 0, metadata !27, metadata !DIExpression()), !dbg !23
  call void @llvm.dbg.value(metadata i32 1, metadata !28, metadata !DIExpression()), !dbg !23
  call void @llvm.dbg.value(metadata i32 0, metadata !29, metadata !DIExpression()), !dbg !23
  call void @llvm.dbg.value(metadata i32 1, metadata !30, metadata !DIExpression()), !dbg !23
  %0 = bitcast i64* %__nv_MAIN__F1L17_1Arg2 to i8**, !dbg !31
  %1 = load i8*, i8** %0, align 8, !dbg !31
  %2 = bitcast i8** %.S0000_317 to i64*, !dbg !31
  store i8* %1, i8** %.S0000_317, align 8, !dbg !31
  %3 = load i32, i32* %__nv_MAIN__F1L17_1Arg0, align 4, !dbg !32
  store i32 %3, i32* %__gtid___nv_MAIN__F1L17_1__360, align 4, !dbg !32
  br label %L.LB2_354

L.LB2_354:                                        ; preds = %L.entry
  br label %L.LB2_308

L.LB2_308:                                        ; preds = %L.LB2_354
  store i32 -1, i32* %.s0000_355, align 4, !dbg !33
  store i32 0, i32* %.s0001_356, align 4, !dbg !33
  %4 = load i32, i32* %__gtid___nv_MAIN__F1L17_1__360, align 4, !dbg !33
  %5 = call i32 @__kmpc_single(i64* null, i32 %4), !dbg !33
  %6 = icmp eq i32 %5, 0, !dbg !33
  br i1 %6, label %L.LB2_316, label %L.LB2_310, !dbg !33

L.LB2_310:                                        ; preds = %L.LB2_308
  %7 = load i8*, i8** %.S0000_317, align 8, !dbg !34
  %8 = bitcast i8* %7 to i64*, !dbg !34
  call void @drb134_taskdep4_orig_yes_omp_45_foo(i64* %8), !dbg !34
  %9 = load i32, i32* %__gtid___nv_MAIN__F1L17_1__360, align 4, !dbg !35
  store i32 %9, i32* %.s0000_355, align 4, !dbg !35
  store i32 1, i32* %.s0001_356, align 4, !dbg !35
  %10 = load i32, i32* %__gtid___nv_MAIN__F1L17_1__360, align 4, !dbg !35
  call void @__kmpc_end_single(i64* null, i32 %10), !dbg !35
  br label %L.LB2_316

L.LB2_316:                                        ; preds = %L.LB2_310, %L.LB2_308
  br label %L.LB2_311

L.LB2_311:                                        ; preds = %L.LB2_316
  %11 = load i32, i32* %__gtid___nv_MAIN__F1L17_1__360, align 4, !dbg !35
  call void @__kmpc_barrier(i64* null, i32 %11), !dbg !35
  br label %L.LB2_312

L.LB2_312:                                        ; preds = %L.LB2_311
  ret void, !dbg !32
}

define internal void @drb134_taskdep4_orig_yes_omp_45_foo(i64* %.S0000) #0 !dbg !36 {
L.entry:
  %__gtid_drb134_taskdep4_orig_yes_omp_45_foo_358 = alloca i32, align 4
  %x_311 = alloca i32, align 4
  %y_312 = alloca i32, align 4
  %.s0002_353 = alloca i32, align 4
  %.z0310_352 = alloca i8*, align 8
  %.s0003_381 = alloca i32, align 4
  %.z0310_380 = alloca i8*, align 8
  %.s0004_396 = alloca i32, align 4
  %.z0310_395 = alloca i8*, align 8
  %z__io_332 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i64* %.S0000, metadata !38, metadata !DIExpression()), !dbg !40
  call void @llvm.dbg.value(metadata i32 1, metadata !41, metadata !DIExpression()), !dbg !40
  call void @llvm.dbg.value(metadata i32 2, metadata !42, metadata !DIExpression()), !dbg !40
  call void @llvm.dbg.value(metadata i32 0, metadata !43, metadata !DIExpression()), !dbg !40
  call void @llvm.dbg.value(metadata i32 1, metadata !44, metadata !DIExpression()), !dbg !40
  call void @llvm.dbg.value(metadata i32 2, metadata !45, metadata !DIExpression()), !dbg !40
  call void @llvm.dbg.value(metadata i32 0, metadata !46, metadata !DIExpression()), !dbg !40
  call void @llvm.dbg.value(metadata i32 1, metadata !47, metadata !DIExpression()), !dbg !40
  call void @llvm.dbg.value(metadata i32 2, metadata !48, metadata !DIExpression()), !dbg !40
  %0 = call i32 @__kmpc_global_thread_num(i64* null), !dbg !49
  store i32 %0, i32* %__gtid_drb134_taskdep4_orig_yes_omp_45_foo_358, align 4, !dbg !49
  br label %L.LB4_349

L.LB4_349:                                        ; preds = %L.entry
  call void @llvm.dbg.declare(metadata i32* %x_311, metadata !50, metadata !DIExpression()), !dbg !40
  store i32 0, i32* %x_311, align 4, !dbg !51
  call void @llvm.dbg.declare(metadata i32* %y_312, metadata !52, metadata !DIExpression()), !dbg !40
  store i32 2, i32* %y_312, align 4, !dbg !53
  store i32 1, i32* %.s0002_353, align 4, !dbg !54
  %1 = load i32, i32* %__gtid_drb134_taskdep4_orig_yes_omp_45_foo_358, align 4, !dbg !55
  %2 = load i32, i32* %.s0002_353, align 4, !dbg !55
  %3 = bitcast void (i32, i64*)* @__nv_drb134_taskdep4_orig_yes_omp_45_foo_F1L29_2_ to i64*, !dbg !55
  %4 = call i8* @__kmpc_omp_task_alloc(i64* null, i32 %1, i32 %2, i32 40, i32 8, i64* %3), !dbg !55
  store i8* %4, i8** %.z0310_352, align 8, !dbg !55
  %5 = bitcast i64* %.S0000 to i8*, !dbg !55
  %6 = load i8*, i8** %.z0310_352, align 8, !dbg !55
  %7 = bitcast i8* %6 to i8***, !dbg !55
  %8 = load i8**, i8*** %7, align 8, !dbg !55
  store i8* %5, i8** %8, align 8, !dbg !55
  %9 = bitcast i32* %x_311 to i8*, !dbg !55
  %10 = load i8*, i8** %.z0310_352, align 8, !dbg !55
  %11 = bitcast i8* %10 to i8**, !dbg !55
  %12 = load i8*, i8** %11, align 8, !dbg !55
  %13 = getelementptr i8, i8* %12, i64 8, !dbg !55
  %14 = bitcast i8* %13 to i8**, !dbg !55
  store i8* %9, i8** %14, align 8, !dbg !55
  %15 = load i32, i32* %__gtid_drb134_taskdep4_orig_yes_omp_45_foo_358, align 4, !dbg !55
  %16 = load i8*, i8** %.z0310_352, align 8, !dbg !55
  %17 = bitcast i8* %16 to i64*, !dbg !55
  call void @__kmpc_omp_task(i64* null, i32 %15, i64* %17), !dbg !55
  br label %L.LB4_342

L.LB4_342:                                        ; preds = %L.LB4_349
  store i32 1, i32* %.s0003_381, align 4, !dbg !56
  %18 = load i32, i32* %__gtid_drb134_taskdep4_orig_yes_omp_45_foo_358, align 4, !dbg !57
  %19 = load i32, i32* %.s0003_381, align 4, !dbg !57
  %20 = bitcast void (i32, i64*)* @__nv_drb134_taskdep4_orig_yes_omp_45_foo_F1L33_3_ to i64*, !dbg !57
  %21 = call i8* @__kmpc_omp_task_alloc(i64* null, i32 %18, i32 %19, i32 40, i32 16, i64* %20), !dbg !57
  store i8* %21, i8** %.z0310_380, align 8, !dbg !57
  %22 = bitcast i64* %.S0000 to i8*, !dbg !57
  %23 = load i8*, i8** %.z0310_380, align 8, !dbg !57
  %24 = bitcast i8* %23 to i8***, !dbg !57
  %25 = load i8**, i8*** %24, align 8, !dbg !57
  store i8* %22, i8** %25, align 8, !dbg !57
  %26 = bitcast i32* %y_312 to i8*, !dbg !57
  %27 = load i8*, i8** %.z0310_380, align 8, !dbg !57
  %28 = bitcast i8* %27 to i8**, !dbg !57
  %29 = load i8*, i8** %28, align 8, !dbg !57
  %30 = getelementptr i8, i8* %29, i64 8, !dbg !57
  %31 = bitcast i8* %30 to i8**, !dbg !57
  store i8* %26, i8** %31, align 8, !dbg !57
  %32 = bitcast i32* %x_311 to i8*, !dbg !57
  %33 = load i8*, i8** %.z0310_380, align 8, !dbg !57
  %34 = bitcast i8* %33 to i8**, !dbg !57
  %35 = load i8*, i8** %34, align 8, !dbg !57
  %36 = getelementptr i8, i8* %35, i64 16, !dbg !57
  %37 = bitcast i8* %36 to i8**, !dbg !57
  store i8* %32, i8** %37, align 8, !dbg !57
  %38 = load i32, i32* %__gtid_drb134_taskdep4_orig_yes_omp_45_foo_358, align 4, !dbg !57
  %39 = load i8*, i8** %.z0310_380, align 8, !dbg !57
  %40 = bitcast i8* %39 to i64*, !dbg !57
  call void @__kmpc_omp_task(i64* null, i32 %38, i64* %40), !dbg !57
  br label %L.LB4_343

L.LB4_343:                                        ; preds = %L.LB4_342
  store i32 1, i32* %.s0004_396, align 4, !dbg !58
  %41 = load i32, i32* %__gtid_drb134_taskdep4_orig_yes_omp_45_foo_358, align 4, !dbg !59
  %42 = load i32, i32* %.s0004_396, align 4, !dbg !59
  %43 = bitcast void (i32, i64*)* @__nv_drb134_taskdep4_orig_yes_omp_45_foo_F1L37_4_ to i64*, !dbg !59
  %44 = call i8* @__kmpc_omp_task_alloc(i64* null, i32 %41, i32 %42, i32 40, i32 24, i64* %43), !dbg !59
  store i8* %44, i8** %.z0310_395, align 8, !dbg !59
  %45 = bitcast i64* %.S0000 to i8*, !dbg !59
  %46 = load i8*, i8** %.z0310_395, align 8, !dbg !59
  %47 = bitcast i8* %46 to i8***, !dbg !59
  %48 = load i8**, i8*** %47, align 8, !dbg !59
  store i8* %45, i8** %48, align 8, !dbg !59
  %49 = load i32, i32* %__gtid_drb134_taskdep4_orig_yes_omp_45_foo_358, align 4, !dbg !59
  %50 = load i8*, i8** %.z0310_395, align 8, !dbg !59
  %51 = bitcast i8* %50 to i64*, !dbg !59
  call void @__kmpc_omp_task_begin_if0(i64* null, i32 %49, i64* %51), !dbg !59
  %52 = load i32, i32* %__gtid_drb134_taskdep4_orig_yes_omp_45_foo_358, align 4, !dbg !59
  %53 = load i8*, i8** %.z0310_395, align 8, !dbg !59
  %54 = bitcast i8* %53 to i64*, !dbg !59
  call void @__nv_drb134_taskdep4_orig_yes_omp_45_foo_F1L37_4_(i32 %52, i64* %54), !dbg !59
  %55 = load i32, i32* %__gtid_drb134_taskdep4_orig_yes_omp_45_foo_358, align 4, !dbg !59
  %56 = load i8*, i8** %.z0310_395, align 8, !dbg !59
  %57 = bitcast i8* %56 to i64*, !dbg !59
  call void @__kmpc_omp_task_complete_if0(i64* null, i32 %55, i64* %57), !dbg !59
  br label %L.LB4_344, !dbg !59

L.LB4_344:                                        ; preds = %L.LB4_343
  call void (...) @_mp_bcs_nest(), !dbg !60
  %58 = bitcast i32* @.C329_drb134_taskdep4_orig_yes_omp_45_foo to i8*, !dbg !60
  %59 = bitcast [59 x i8]* @.C327_drb134_taskdep4_orig_yes_omp_45_foo to i8*, !dbg !60
  %60 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !60
  call void (i8*, i8*, i64, ...) %60(i8* %58, i8* %59, i64 59), !dbg !60
  %61 = bitcast i32* @.C330_drb134_taskdep4_orig_yes_omp_45_foo to i8*, !dbg !60
  %62 = bitcast i32* @.C283_drb134_taskdep4_orig_yes_omp_45_foo to i8*, !dbg !60
  %63 = bitcast i32* @.C283_drb134_taskdep4_orig_yes_omp_45_foo to i8*, !dbg !60
  %64 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !60
  %65 = call i32 (i8*, i8*, i8*, i8*, ...) %64(i8* %61, i8* null, i8* %62, i8* %63), !dbg !60
  call void @llvm.dbg.declare(metadata i32* %z__io_332, metadata !61, metadata !DIExpression()), !dbg !40
  store i32 %65, i32* %z__io_332, align 4, !dbg !60
  %66 = bitcast [2 x i8]* @.C333_drb134_taskdep4_orig_yes_omp_45_foo to i8*, !dbg !60
  %67 = bitcast i32 (...)* @f90io_sc_ch_ldw to i32 (i8*, i32, i64, ...)*, !dbg !60
  %68 = call i32 (i8*, i32, i64, ...) %67(i8* %66, i32 14, i64 2), !dbg !60
  store i32 %68, i32* %z__io_332, align 4, !dbg !60
  %69 = load i32, i32* %x_311, align 4, !dbg !60
  call void @llvm.dbg.value(metadata i32 %69, metadata !50, metadata !DIExpression()), !dbg !40
  %70 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !60
  %71 = call i32 (i32, i32, ...) %70(i32 %69, i32 25), !dbg !60
  store i32 %71, i32* %z__io_332, align 4, !dbg !60
  %72 = call i32 (...) @f90io_ldw_end(), !dbg !60
  store i32 %72, i32* %z__io_332, align 4, !dbg !60
  call void (...) @_mp_ecs_nest(), !dbg !60
  call void (...) @_mp_bcs_nest(), !dbg !62
  %73 = bitcast i32* @.C339_drb134_taskdep4_orig_yes_omp_45_foo to i8*, !dbg !62
  %74 = bitcast [59 x i8]* @.C327_drb134_taskdep4_orig_yes_omp_45_foo to i8*, !dbg !62
  %75 = bitcast void (...)* @f90io_src_info03a to void (i8*, i8*, i64, ...)*, !dbg !62
  call void (i8*, i8*, i64, ...) %75(i8* %73, i8* %74, i64 59), !dbg !62
  %76 = bitcast i32* @.C330_drb134_taskdep4_orig_yes_omp_45_foo to i8*, !dbg !62
  %77 = bitcast i32* @.C283_drb134_taskdep4_orig_yes_omp_45_foo to i8*, !dbg !62
  %78 = bitcast i32* @.C283_drb134_taskdep4_orig_yes_omp_45_foo to i8*, !dbg !62
  %79 = bitcast i32 (...)* @f90io_print_init to i32 (i8*, i8*, i8*, i8*, ...)*, !dbg !62
  %80 = call i32 (i8*, i8*, i8*, i8*, ...) %79(i8* %76, i8* null, i8* %77, i8* %78), !dbg !62
  store i32 %80, i32* %z__io_332, align 4, !dbg !62
  %81 = bitcast [2 x i8]* @.C340_drb134_taskdep4_orig_yes_omp_45_foo to i8*, !dbg !62
  %82 = bitcast i32 (...)* @f90io_sc_ch_ldw to i32 (i8*, i32, i64, ...)*, !dbg !62
  %83 = call i32 (i8*, i32, i64, ...) %82(i8* %81, i32 14, i64 2), !dbg !62
  store i32 %83, i32* %z__io_332, align 4, !dbg !62
  %84 = load i32, i32* %y_312, align 4, !dbg !62
  call void @llvm.dbg.value(metadata i32 %84, metadata !52, metadata !DIExpression()), !dbg !40
  %85 = bitcast i32 (...)* @f90io_sc_i_ldw to i32 (i32, i32, ...)*, !dbg !62
  %86 = call i32 (i32, i32, ...) %85(i32 %84, i32 25), !dbg !62
  store i32 %86, i32* %z__io_332, align 4, !dbg !62
  %87 = call i32 (...) @f90io_ldw_end(), !dbg !62
  store i32 %87, i32* %z__io_332, align 4, !dbg !62
  call void (...) @_mp_ecs_nest(), !dbg !62
  %88 = load i32, i32* %__gtid_drb134_taskdep4_orig_yes_omp_45_foo_358, align 4, !dbg !63
  %89 = call i32 @__kmpc_omp_taskwait(i64* null, i32 %88), !dbg !63
  ret void, !dbg !49
}

define internal void @__nv_drb134_taskdep4_orig_yes_omp_45_foo_F1L29_2_(i32 %__nv_drb134_taskdep4_orig_yes_omp_45_foo_F1L29_2Arg0.arg, i64* %__nv_drb134_taskdep4_orig_yes_omp_45_foo_F1L29_2Arg1) #0 !dbg !64 {
L.entry:
  %__nv_drb134_taskdep4_orig_yes_omp_45_foo_F1L29_2Arg0.addr = alloca i32, align 4
  %.S0001_439 = alloca i8*, align 8
  %.S0000_345 = alloca i8*, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_drb134_taskdep4_orig_yes_omp_45_foo_F1L29_2Arg0.addr, metadata !67, metadata !DIExpression()), !dbg !68
  store i32 %__nv_drb134_taskdep4_orig_yes_omp_45_foo_F1L29_2Arg0.arg, i32* %__nv_drb134_taskdep4_orig_yes_omp_45_foo_F1L29_2Arg0.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_drb134_taskdep4_orig_yes_omp_45_foo_F1L29_2Arg0.addr, metadata !69, metadata !DIExpression()), !dbg !68
  call void @llvm.dbg.declare(metadata i64* %__nv_drb134_taskdep4_orig_yes_omp_45_foo_F1L29_2Arg1, metadata !70, metadata !DIExpression()), !dbg !68
  call void @llvm.dbg.value(metadata i32 1, metadata !71, metadata !DIExpression()), !dbg !68
  call void @llvm.dbg.value(metadata i32 2, metadata !72, metadata !DIExpression()), !dbg !68
  call void @llvm.dbg.value(metadata i32 0, metadata !73, metadata !DIExpression()), !dbg !68
  call void @llvm.dbg.value(metadata i32 1, metadata !74, metadata !DIExpression()), !dbg !68
  call void @llvm.dbg.value(metadata i32 2, metadata !75, metadata !DIExpression()), !dbg !68
  call void @llvm.dbg.value(metadata i32 0, metadata !76, metadata !DIExpression()), !dbg !68
  call void @llvm.dbg.value(metadata i32 1, metadata !77, metadata !DIExpression()), !dbg !68
  call void @llvm.dbg.value(metadata i32 2, metadata !78, metadata !DIExpression()), !dbg !68
  %0 = bitcast i64* %__nv_drb134_taskdep4_orig_yes_omp_45_foo_F1L29_2Arg1 to i8**, !dbg !79
  %1 = load i8*, i8** %0, align 8, !dbg !79
  store i8* %1, i8** %.S0001_439, align 8, !dbg !79
  %2 = bitcast i64* %__nv_drb134_taskdep4_orig_yes_omp_45_foo_F1L29_2Arg1 to i8**, !dbg !79
  %3 = load i8*, i8** %2, align 8, !dbg !79
  %4 = bitcast i8** %.S0000_345 to i64*, !dbg !79
  store i8* %3, i8** %.S0000_345, align 8, !dbg !79
  br label %L.LB5_444

L.LB5_444:                                        ; preds = %L.entry
  br label %L.LB5_315

L.LB5_315:                                        ; preds = %L.LB5_444
  %5 = bitcast i64* %__nv_drb134_taskdep4_orig_yes_omp_45_foo_F1L29_2Arg1 to i8**, !dbg !80
  %6 = load i8*, i8** %5, align 8, !dbg !80
  %7 = getelementptr i8, i8* %6, i64 8, !dbg !80
  %8 = bitcast i8* %7 to i32**, !dbg !80
  %9 = load i32*, i32** %8, align 8, !dbg !80
  %10 = load i32, i32* %9, align 4, !dbg !80
  %11 = add nsw i32 %10, 1, !dbg !80
  %12 = bitcast i64* %__nv_drb134_taskdep4_orig_yes_omp_45_foo_F1L29_2Arg1 to i8**, !dbg !80
  %13 = load i8*, i8** %12, align 8, !dbg !80
  %14 = getelementptr i8, i8* %13, i64 8, !dbg !80
  %15 = bitcast i8* %14 to i32**, !dbg !80
  %16 = load i32*, i32** %15, align 8, !dbg !80
  store i32 %11, i32* %16, align 4, !dbg !80
  br label %L.LB5_316

L.LB5_316:                                        ; preds = %L.LB5_315
  ret void, !dbg !81
}

define internal void @__nv_drb134_taskdep4_orig_yes_omp_45_foo_F1L33_3_(i32 %__nv_drb134_taskdep4_orig_yes_omp_45_foo_F1L33_3Arg0.arg, i64* %__nv_drb134_taskdep4_orig_yes_omp_45_foo_F1L33_3Arg1) #0 !dbg !82 {
L.entry:
  %__nv_drb134_taskdep4_orig_yes_omp_45_foo_F1L33_3Arg0.addr = alloca i32, align 4
  %.S0001_439 = alloca i8*, align 8
  %.S0000_345 = alloca i8*, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_drb134_taskdep4_orig_yes_omp_45_foo_F1L33_3Arg0.addr, metadata !83, metadata !DIExpression()), !dbg !84
  store i32 %__nv_drb134_taskdep4_orig_yes_omp_45_foo_F1L33_3Arg0.arg, i32* %__nv_drb134_taskdep4_orig_yes_omp_45_foo_F1L33_3Arg0.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_drb134_taskdep4_orig_yes_omp_45_foo_F1L33_3Arg0.addr, metadata !85, metadata !DIExpression()), !dbg !84
  call void @llvm.dbg.declare(metadata i64* %__nv_drb134_taskdep4_orig_yes_omp_45_foo_F1L33_3Arg1, metadata !86, metadata !DIExpression()), !dbg !84
  call void @llvm.dbg.value(metadata i32 1, metadata !87, metadata !DIExpression()), !dbg !84
  call void @llvm.dbg.value(metadata i32 2, metadata !88, metadata !DIExpression()), !dbg !84
  call void @llvm.dbg.value(metadata i32 0, metadata !89, metadata !DIExpression()), !dbg !84
  call void @llvm.dbg.value(metadata i32 1, metadata !90, metadata !DIExpression()), !dbg !84
  call void @llvm.dbg.value(metadata i32 2, metadata !91, metadata !DIExpression()), !dbg !84
  call void @llvm.dbg.value(metadata i32 0, metadata !92, metadata !DIExpression()), !dbg !84
  call void @llvm.dbg.value(metadata i32 1, metadata !93, metadata !DIExpression()), !dbg !84
  call void @llvm.dbg.value(metadata i32 2, metadata !94, metadata !DIExpression()), !dbg !84
  %0 = bitcast i64* %__nv_drb134_taskdep4_orig_yes_omp_45_foo_F1L33_3Arg1 to i8**, !dbg !95
  %1 = load i8*, i8** %0, align 8, !dbg !95
  store i8* %1, i8** %.S0001_439, align 8, !dbg !95
  %2 = bitcast i64* %__nv_drb134_taskdep4_orig_yes_omp_45_foo_F1L33_3Arg1 to i8**, !dbg !95
  %3 = load i8*, i8** %2, align 8, !dbg !95
  %4 = bitcast i8** %.S0000_345 to i64*, !dbg !95
  store i8* %3, i8** %.S0000_345, align 8, !dbg !95
  br label %L.LB6_452

L.LB6_452:                                        ; preds = %L.entry
  br label %L.LB6_319

L.LB6_319:                                        ; preds = %L.LB6_452
  %5 = bitcast i64* %__nv_drb134_taskdep4_orig_yes_omp_45_foo_F1L33_3Arg1 to i8**, !dbg !96
  %6 = load i8*, i8** %5, align 8, !dbg !96
  %7 = getelementptr i8, i8* %6, i64 8, !dbg !96
  %8 = bitcast i8* %7 to i32**, !dbg !96
  %9 = load i32*, i32** %8, align 8, !dbg !96
  %10 = load i32, i32* %9, align 4, !dbg !96
  %11 = bitcast i64* %__nv_drb134_taskdep4_orig_yes_omp_45_foo_F1L33_3Arg1 to i8**, !dbg !96
  %12 = load i8*, i8** %11, align 8, !dbg !96
  %13 = getelementptr i8, i8* %12, i64 16, !dbg !96
  %14 = bitcast i8* %13 to i32**, !dbg !96
  %15 = load i32*, i32** %14, align 8, !dbg !96
  %16 = load i32, i32* %15, align 4, !dbg !96
  %17 = sub nsw i32 %10, %16, !dbg !96
  %18 = bitcast i64* %__nv_drb134_taskdep4_orig_yes_omp_45_foo_F1L33_3Arg1 to i8**, !dbg !96
  %19 = load i8*, i8** %18, align 8, !dbg !96
  %20 = getelementptr i8, i8* %19, i64 8, !dbg !96
  %21 = bitcast i8* %20 to i32**, !dbg !96
  %22 = load i32*, i32** %21, align 8, !dbg !96
  store i32 %17, i32* %22, align 4, !dbg !96
  br label %L.LB6_320

L.LB6_320:                                        ; preds = %L.LB6_319
  ret void, !dbg !97
}

define internal void @__nv_drb134_taskdep4_orig_yes_omp_45_foo_F1L37_4_(i32 %__nv_drb134_taskdep4_orig_yes_omp_45_foo_F1L37_4Arg0.arg, i64* %__nv_drb134_taskdep4_orig_yes_omp_45_foo_F1L37_4Arg1) #0 !dbg !98 {
L.entry:
  %__nv_drb134_taskdep4_orig_yes_omp_45_foo_F1L37_4Arg0.addr = alloca i32, align 4
  %.S0001_439 = alloca i8*, align 8
  %.S0000_345 = alloca i8*, align 8
  call void @llvm.dbg.declare(metadata i32* %__nv_drb134_taskdep4_orig_yes_omp_45_foo_F1L37_4Arg0.addr, metadata !99, metadata !DIExpression()), !dbg !100
  store i32 %__nv_drb134_taskdep4_orig_yes_omp_45_foo_F1L37_4Arg0.arg, i32* %__nv_drb134_taskdep4_orig_yes_omp_45_foo_F1L37_4Arg0.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %__nv_drb134_taskdep4_orig_yes_omp_45_foo_F1L37_4Arg0.addr, metadata !101, metadata !DIExpression()), !dbg !100
  call void @llvm.dbg.declare(metadata i64* %__nv_drb134_taskdep4_orig_yes_omp_45_foo_F1L37_4Arg1, metadata !102, metadata !DIExpression()), !dbg !100
  call void @llvm.dbg.value(metadata i32 1, metadata !103, metadata !DIExpression()), !dbg !100
  call void @llvm.dbg.value(metadata i32 2, metadata !104, metadata !DIExpression()), !dbg !100
  call void @llvm.dbg.value(metadata i32 0, metadata !105, metadata !DIExpression()), !dbg !100
  call void @llvm.dbg.value(metadata i32 1, metadata !106, metadata !DIExpression()), !dbg !100
  call void @llvm.dbg.value(metadata i32 2, metadata !107, metadata !DIExpression()), !dbg !100
  call void @llvm.dbg.value(metadata i32 0, metadata !108, metadata !DIExpression()), !dbg !100
  call void @llvm.dbg.value(metadata i32 1, metadata !109, metadata !DIExpression()), !dbg !100
  call void @llvm.dbg.value(metadata i32 2, metadata !110, metadata !DIExpression()), !dbg !100
  %0 = bitcast i64* %__nv_drb134_taskdep4_orig_yes_omp_45_foo_F1L37_4Arg1 to i8**, !dbg !111
  %1 = load i8*, i8** %0, align 8, !dbg !111
  store i8* %1, i8** %.S0001_439, align 8, !dbg !111
  %2 = bitcast i64* %__nv_drb134_taskdep4_orig_yes_omp_45_foo_F1L37_4Arg1 to i8**, !dbg !111
  %3 = load i8*, i8** %2, align 8, !dbg !111
  %4 = bitcast i8** %.S0000_345 to i64*, !dbg !111
  store i8* %3, i8** %.S0000_345, align 8, !dbg !111
  br label %L.LB7_460

L.LB7_460:                                        ; preds = %L.entry
  br label %L.LB7_324

L.LB7_324:                                        ; preds = %L.LB7_460
  br label %L.LB7_325

L.LB7_325:                                        ; preds = %L.LB7_324
  ret void, !dbg !112
}

declare void @_mp_ecs_nest(...) #0

declare signext i32 @f90io_ldw_end(...) #0

declare signext i32 @f90io_sc_i_ldw(...) #0

declare signext i32 @f90io_sc_ch_ldw(...) #0

declare signext i32 @f90io_print_init(...) #0

declare void @f90io_src_info03a(...) #0

declare void @_mp_bcs_nest(...) #0

declare signext i32 @__kmpc_omp_taskwait(i64*, i32) #0

declare void @__kmpc_omp_task_complete_if0(i64*, i32, i64*) #0

declare void @__kmpc_omp_task_begin_if0(i64*, i32, i64*) #0

declare void @__kmpc_omp_task(i64*, i32, i64*) #0

declare i8* @__kmpc_omp_task_alloc(i64*, i32, i32, i32, i32, i64*) #0

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare void @__kmpc_barrier(i64*, i32) #0

declare void @__kmpc_end_single(i64*, i32) #0

declare signext i32 @__kmpc_single(i64*, i32) #0

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
!3 = !DIFile(filename: "micro-benchmarks-fortran/DRB134-taskdep5-orig-omp45-yes.f95", directory: "/workspaces/LLVMRace/TestCases/dataracebench")
!4 = !{}
!5 = distinct !DISubprogram(name: "drb134_taskdep4_orig_yes_omp_45", scope: !2, file: !3, line: 13, type: !6, scopeLine: 13, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!6 = !DISubroutineType(cc: DW_CC_program, types: !7)
!7 = !{null}
!8 = !DILocalVariable(name: "omp_sched_static", scope: !5, file: !3, type: !9)
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 0, scope: !5)
!11 = !DILocalVariable(name: "omp_proc_bind_false", scope: !5, file: !3, type: !9)
!12 = !DILocalVariable(name: "omp_proc_bind_true", scope: !5, file: !3, type: !9)
!13 = !DILocalVariable(name: "omp_lock_hint_none", scope: !5, file: !3, type: !9)
!14 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !5, file: !3, type: !9)
!15 = !DILocation(line: 22, column: 1, scope: !5)
!16 = !DILocation(line: 13, column: 1, scope: !5)
!17 = !DILocation(line: 17, column: 1, scope: !5)
!18 = distinct !DISubprogram(name: "__nv_MAIN__F1L17_1", scope: !2, file: !3, line: 17, type: !19, scopeLine: 17, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!19 = !DISubroutineType(types: !20)
!20 = !{null, !9, !21, !21}
!21 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!22 = !DILocalVariable(name: "__nv_MAIN__F1L17_1Arg0", arg: 1, scope: !18, file: !3, type: !9)
!23 = !DILocation(line: 0, scope: !18)
!24 = !DILocalVariable(name: "__nv_MAIN__F1L17_1Arg1", arg: 2, scope: !18, file: !3, type: !21)
!25 = !DILocalVariable(name: "__nv_MAIN__F1L17_1Arg2", arg: 3, scope: !18, file: !3, type: !21)
!26 = !DILocalVariable(name: "omp_sched_static", scope: !18, file: !3, type: !9)
!27 = !DILocalVariable(name: "omp_proc_bind_false", scope: !18, file: !3, type: !9)
!28 = !DILocalVariable(name: "omp_proc_bind_true", scope: !18, file: !3, type: !9)
!29 = !DILocalVariable(name: "omp_lock_hint_none", scope: !18, file: !3, type: !9)
!30 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !18, file: !3, type: !9)
!31 = !DILocation(line: 17, column: 1, scope: !18)
!32 = !DILocation(line: 21, column: 1, scope: !18)
!33 = !DILocation(line: 18, column: 1, scope: !18)
!34 = !DILocation(line: 19, column: 1, scope: !18)
!35 = !DILocation(line: 20, column: 1, scope: !18)
!36 = distinct !DISubprogram(name: "foo", scope: !5, file: !3, line: 23, type: !37, scopeLine: 23, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!37 = !DISubroutineType(types: !7)
!38 = !DILocalVariable(arg: 1, scope: !36, file: !3, type: !39, flags: DIFlagArtificial)
!39 = !DIBasicType(name: "uinteger*8", size: 64, align: 64, encoding: DW_ATE_unsigned)
!40 = !DILocation(line: 0, scope: !36)
!41 = !DILocalVariable(name: "omp_sched_static", scope: !36, file: !3, type: !9)
!42 = !DILocalVariable(name: "omp_sched_dynamic", scope: !36, file: !3, type: !9)
!43 = !DILocalVariable(name: "omp_proc_bind_false", scope: !36, file: !3, type: !9)
!44 = !DILocalVariable(name: "omp_proc_bind_true", scope: !36, file: !3, type: !9)
!45 = !DILocalVariable(name: "omp_proc_bind_master", scope: !36, file: !3, type: !9)
!46 = !DILocalVariable(name: "omp_lock_hint_none", scope: !36, file: !3, type: !9)
!47 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !36, file: !3, type: !9)
!48 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !36, file: !3, type: !9)
!49 = !DILocation(line: 44, column: 1, scope: !36)
!50 = !DILocalVariable(name: "x", scope: !36, file: !3, type: !9)
!51 = !DILocation(line: 26, column: 1, scope: !36)
!52 = !DILocalVariable(name: "y", scope: !36, file: !3, type: !9)
!53 = !DILocation(line: 27, column: 1, scope: !36)
!54 = !DILocation(line: 29, column: 1, scope: !36)
!55 = !DILocation(line: 31, column: 1, scope: !36)
!56 = !DILocation(line: 33, column: 1, scope: !36)
!57 = !DILocation(line: 35, column: 1, scope: !36)
!58 = !DILocation(line: 37, column: 1, scope: !36)
!59 = !DILocation(line: 38, column: 1, scope: !36)
!60 = !DILocation(line: 40, column: 1, scope: !36)
!61 = !DILocalVariable(scope: !36, file: !3, type: !9, flags: DIFlagArtificial)
!62 = !DILocation(line: 41, column: 1, scope: !36)
!63 = !DILocation(line: 43, column: 1, scope: !36)
!64 = distinct !DISubprogram(name: "__nv_drb134_taskdep4_orig_yes_omp_45_foo_F1L29_2", scope: !2, file: !3, line: 29, type: !65, scopeLine: 29, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!65 = !DISubroutineType(types: !66)
!66 = !{null, !9, !21}
!67 = !DILocalVariable(name: "__nv_drb134_taskdep4_orig_yes_omp_45_foo_F1L29_2Arg0", scope: !64, file: !3, type: !9)
!68 = !DILocation(line: 0, scope: !64)
!69 = !DILocalVariable(name: "__nv_drb134_taskdep4_orig_yes_omp_45_foo_F1L29_2Arg0", arg: 1, scope: !64, file: !3, type: !9)
!70 = !DILocalVariable(name: "__nv_drb134_taskdep4_orig_yes_omp_45_foo_F1L29_2Arg1", arg: 2, scope: !64, file: !3, type: !21)
!71 = !DILocalVariable(name: "omp_sched_static", scope: !64, file: !3, type: !9)
!72 = !DILocalVariable(name: "omp_sched_dynamic", scope: !64, file: !3, type: !9)
!73 = !DILocalVariable(name: "omp_proc_bind_false", scope: !64, file: !3, type: !9)
!74 = !DILocalVariable(name: "omp_proc_bind_true", scope: !64, file: !3, type: !9)
!75 = !DILocalVariable(name: "omp_proc_bind_master", scope: !64, file: !3, type: !9)
!76 = !DILocalVariable(name: "omp_lock_hint_none", scope: !64, file: !3, type: !9)
!77 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !64, file: !3, type: !9)
!78 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !64, file: !3, type: !9)
!79 = !DILocation(line: 29, column: 1, scope: !64)
!80 = !DILocation(line: 30, column: 1, scope: !64)
!81 = !DILocation(line: 31, column: 1, scope: !64)
!82 = distinct !DISubprogram(name: "__nv_drb134_taskdep4_orig_yes_omp_45_foo_F1L33_3", scope: !2, file: !3, line: 33, type: !65, scopeLine: 33, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!83 = !DILocalVariable(name: "__nv_drb134_taskdep4_orig_yes_omp_45_foo_F1L33_3Arg0", scope: !82, file: !3, type: !9)
!84 = !DILocation(line: 0, scope: !82)
!85 = !DILocalVariable(name: "__nv_drb134_taskdep4_orig_yes_omp_45_foo_F1L33_3Arg0", arg: 1, scope: !82, file: !3, type: !9)
!86 = !DILocalVariable(name: "__nv_drb134_taskdep4_orig_yes_omp_45_foo_F1L33_3Arg1", arg: 2, scope: !82, file: !3, type: !21)
!87 = !DILocalVariable(name: "omp_sched_static", scope: !82, file: !3, type: !9)
!88 = !DILocalVariable(name: "omp_sched_dynamic", scope: !82, file: !3, type: !9)
!89 = !DILocalVariable(name: "omp_proc_bind_false", scope: !82, file: !3, type: !9)
!90 = !DILocalVariable(name: "omp_proc_bind_true", scope: !82, file: !3, type: !9)
!91 = !DILocalVariable(name: "omp_proc_bind_master", scope: !82, file: !3, type: !9)
!92 = !DILocalVariable(name: "omp_lock_hint_none", scope: !82, file: !3, type: !9)
!93 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !82, file: !3, type: !9)
!94 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !82, file: !3, type: !9)
!95 = !DILocation(line: 33, column: 1, scope: !82)
!96 = !DILocation(line: 34, column: 1, scope: !82)
!97 = !DILocation(line: 35, column: 1, scope: !82)
!98 = distinct !DISubprogram(name: "__nv_drb134_taskdep4_orig_yes_omp_45_foo_F1L37_4", scope: !2, file: !3, line: 37, type: !65, scopeLine: 37, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!99 = !DILocalVariable(name: "__nv_drb134_taskdep4_orig_yes_omp_45_foo_F1L37_4Arg0", scope: !98, file: !3, type: !9)
!100 = !DILocation(line: 0, scope: !98)
!101 = !DILocalVariable(name: "__nv_drb134_taskdep4_orig_yes_omp_45_foo_F1L37_4Arg0", arg: 1, scope: !98, file: !3, type: !9)
!102 = !DILocalVariable(name: "__nv_drb134_taskdep4_orig_yes_omp_45_foo_F1L37_4Arg1", arg: 2, scope: !98, file: !3, type: !21)
!103 = !DILocalVariable(name: "omp_sched_static", scope: !98, file: !3, type: !9)
!104 = !DILocalVariable(name: "omp_sched_dynamic", scope: !98, file: !3, type: !9)
!105 = !DILocalVariable(name: "omp_proc_bind_false", scope: !98, file: !3, type: !9)
!106 = !DILocalVariable(name: "omp_proc_bind_true", scope: !98, file: !3, type: !9)
!107 = !DILocalVariable(name: "omp_proc_bind_master", scope: !98, file: !3, type: !9)
!108 = !DILocalVariable(name: "omp_lock_hint_none", scope: !98, file: !3, type: !9)
!109 = !DILocalVariable(name: "omp_lock_hint_uncontended", scope: !98, file: !3, type: !9)
!110 = !DILocalVariable(name: "omp_lock_hint_contended", scope: !98, file: !3, type: !9)
!111 = !DILocation(line: 37, column: 1, scope: !98)
!112 = !DILocation(line: 38, column: 1, scope: !98)

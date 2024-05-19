; ModuleID = 'basic_c_tests/spec-mesa.c'
source_filename = "basic_c_tests/spec-mesa.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.context = type { %struct.api_table, %struct.api_table }
%struct.api_table = type { void (i32*, i32*)*, void (i32*, i32*)*, void (i32*, i32*)* }
%struct.mesa_context = type { %struct.context* }

@CC = common dso_local global %struct.context* null, align 8, !dbg !0

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @begin(i32*, i32*) #0 !dbg !31 {
  %3 = alloca i32*, align 8
  %4 = alloca i32*, align 8
  store i32* %0, i32** %3, align 8
  call void @llvm.dbg.declare(metadata i32** %3, metadata !32, metadata !DIExpression()), !dbg !33
  store i32* %1, i32** %4, align 8
  call void @llvm.dbg.declare(metadata i32** %4, metadata !34, metadata !DIExpression()), !dbg !35
  %5 = load i32*, i32** %3, align 8, !dbg !36
  %6 = bitcast i32* %5 to i8*, !dbg !36
  %7 = load i32*, i32** %4, align 8, !dbg !36
  %8 = bitcast i32* %7 to i8*, !dbg !36
  call void @__aser_no_alias__(i8* %6, i8* %8), !dbg !36
  ret void, !dbg !37
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare dso_local void @__aser_no_alias__(i8*, i8*) #2

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @end(i32*, i32*) #0 !dbg !38 {
  %3 = alloca i32*, align 8
  %4 = alloca i32*, align 8
  store i32* %0, i32** %3, align 8
  call void @llvm.dbg.declare(metadata i32** %3, metadata !39, metadata !DIExpression()), !dbg !40
  store i32* %1, i32** %4, align 8
  call void @llvm.dbg.declare(metadata i32** %4, metadata !41, metadata !DIExpression()), !dbg !42
  %5 = load i32*, i32** %3, align 8, !dbg !43
  %6 = bitcast i32* %5 to i8*, !dbg !43
  %7 = load i32*, i32** %4, align 8, !dbg !43
  %8 = bitcast i32* %7 to i8*, !dbg !43
  call void @__aser_alias__(i8* %6, i8* %8), !dbg !43
  ret void, !dbg !44
}

declare dso_local void @__aser_alias__(i8*, i8*) #2

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @render(i32*, i32*) #0 !dbg !45 {
  %3 = alloca i32*, align 8
  %4 = alloca i32*, align 8
  store i32* %0, i32** %3, align 8
  call void @llvm.dbg.declare(metadata i32** %3, metadata !46, metadata !DIExpression()), !dbg !47
  store i32* %1, i32** %4, align 8
  call void @llvm.dbg.declare(metadata i32** %4, metadata !48, metadata !DIExpression()), !dbg !49
  %5 = load i32*, i32** %3, align 8, !dbg !50
  %6 = bitcast i32* %5 to i8*, !dbg !50
  %7 = load i32*, i32** %4, align 8, !dbg !50
  %8 = bitcast i32* %7 to i8*, !dbg !50
  call void @__aser_alias__(i8* %6, i8* %8), !dbg !50
  ret void, !dbg !51
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @init_exec_pointers(%struct.api_table*) #0 !dbg !52 {
  %2 = alloca %struct.api_table*, align 8
  store %struct.api_table* %0, %struct.api_table** %2, align 8
  call void @llvm.dbg.declare(metadata %struct.api_table** %2, metadata !56, metadata !DIExpression()), !dbg !57
  %3 = load %struct.api_table*, %struct.api_table** %2, align 8, !dbg !58
  %4 = getelementptr inbounds %struct.api_table, %struct.api_table* %3, i32 0, i32 0, !dbg !59
  store void (i32*, i32*)* @begin, void (i32*, i32*)** %4, align 8, !dbg !60
  %5 = load %struct.api_table*, %struct.api_table** %2, align 8, !dbg !61
  %6 = getelementptr inbounds %struct.api_table, %struct.api_table* %5, i32 0, i32 1, !dbg !62
  store void (i32*, i32*)* @end, void (i32*, i32*)** %6, align 8, !dbg !63
  %7 = load %struct.api_table*, %struct.api_table** %2, align 8, !dbg !64
  %8 = getelementptr inbounds %struct.api_table, %struct.api_table* %7, i32 0, i32 2, !dbg !65
  store void (i32*, i32*)* @render, void (i32*, i32*)** %8, align 8, !dbg !66
  ret void, !dbg !67
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @init_api_function(%struct.context*) #0 !dbg !68 {
  %2 = alloca %struct.context*, align 8
  store %struct.context* %0, %struct.context** %2, align 8
  call void @llvm.dbg.declare(metadata %struct.context** %2, metadata !71, metadata !DIExpression()), !dbg !72
  %3 = load %struct.context*, %struct.context** %2, align 8, !dbg !73
  %4 = getelementptr inbounds %struct.context, %struct.context* %3, i32 0, i32 1, !dbg !74
  call void @init_exec_pointers(%struct.api_table* %4), !dbg !75
  ret void, !dbg !76
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local %struct.context* @create_context() #0 !dbg !77 {
  %1 = alloca %struct.context*, align 8
  call void @llvm.dbg.declare(metadata %struct.context** %1, metadata !80, metadata !DIExpression()), !dbg !81
  %2 = call i8* @malloc(i64 48), !dbg !82
  %3 = bitcast i8* %2 to %struct.context*, !dbg !83
  store %struct.context* %3, %struct.context** %1, align 8, !dbg !81
  %4 = load %struct.context*, %struct.context** %1, align 8, !dbg !84
  call void @init_api_function(%struct.context* %4), !dbg !85
  %5 = load %struct.context*, %struct.context** %1, align 8, !dbg !86
  %6 = getelementptr inbounds %struct.context, %struct.context* %5, i32 0, i32 0, !dbg !87
  %7 = load %struct.context*, %struct.context** %1, align 8, !dbg !88
  %8 = getelementptr inbounds %struct.context, %struct.context* %7, i32 0, i32 1, !dbg !89
  %9 = bitcast %struct.api_table* %6 to i8*, !dbg !89
  %10 = bitcast %struct.api_table* %8 to i8*, !dbg !89
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %9, i8* align 8 %10, i64 24, i1 false), !dbg !89
  %11 = load %struct.context*, %struct.context** %1, align 8, !dbg !90
  ret %struct.context* %11, !dbg !91
}

declare dso_local i8* @malloc(i64) #2

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1 immarg) #3

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @change_context(%struct.context*) #0 !dbg !92 {
  %2 = alloca %struct.context*, align 8
  store %struct.context* %0, %struct.context** %2, align 8
  call void @llvm.dbg.declare(metadata %struct.context** %2, metadata !93, metadata !DIExpression()), !dbg !94
  %3 = load %struct.context*, %struct.context** %2, align 8, !dbg !95
  store %struct.context* %3, %struct.context** @CC, align 8, !dbg !96
  ret void, !dbg !97
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @make_current(%struct.mesa_context*) #0 !dbg !98 {
  %2 = alloca %struct.mesa_context*, align 8
  store %struct.mesa_context* %0, %struct.mesa_context** %2, align 8
  call void @llvm.dbg.declare(metadata %struct.mesa_context** %2, metadata !101, metadata !DIExpression()), !dbg !102
  %3 = load %struct.mesa_context*, %struct.mesa_context** %2, align 8, !dbg !103
  %4 = getelementptr inbounds %struct.mesa_context, %struct.mesa_context* %3, i32 0, i32 0, !dbg !104
  %5 = load %struct.context*, %struct.context** %4, align 8, !dbg !104
  call void @change_context(%struct.context* %5), !dbg !105
  ret void, !dbg !106
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @draw(i32*, i32*, i32*) #0 !dbg !107 {
  %4 = alloca i32*, align 8
  %5 = alloca i32*, align 8
  %6 = alloca i32*, align 8
  store i32* %0, i32** %4, align 8
  call void @llvm.dbg.declare(metadata i32** %4, metadata !110, metadata !DIExpression()), !dbg !111
  store i32* %1, i32** %5, align 8
  call void @llvm.dbg.declare(metadata i32** %5, metadata !112, metadata !DIExpression()), !dbg !113
  store i32* %2, i32** %6, align 8
  call void @llvm.dbg.declare(metadata i32** %6, metadata !114, metadata !DIExpression()), !dbg !115
  %7 = load %struct.context*, %struct.context** @CC, align 8, !dbg !116
  %8 = getelementptr inbounds %struct.context, %struct.context* %7, i32 0, i32 0, !dbg !117
  %9 = getelementptr inbounds %struct.api_table, %struct.api_table* %8, i32 0, i32 0, !dbg !118
  %10 = load void (i32*, i32*)*, void (i32*, i32*)** %9, align 8, !dbg !118
  %11 = load i32*, i32** %4, align 8, !dbg !119
  %12 = load i32*, i32** %5, align 8, !dbg !120
  call void %10(i32* %11, i32* %12), !dbg !121
  %13 = load i32*, i32** %4, align 8, !dbg !122
  %14 = icmp ne i32* %13, null, !dbg !122
  br i1 %14, label %15, label %17, !dbg !124

15:                                               ; preds = %3
  %16 = load i32*, i32** %6, align 8, !dbg !125
  store i32* %16, i32** %5, align 8, !dbg !126
  br label %17, !dbg !127

17:                                               ; preds = %15, %3
  %18 = load %struct.context*, %struct.context** @CC, align 8, !dbg !128
  %19 = getelementptr inbounds %struct.context, %struct.context* %18, i32 0, i32 0, !dbg !130
  %20 = getelementptr inbounds %struct.api_table, %struct.api_table* %19, i32 0, i32 2, !dbg !131
  %21 = load void (i32*, i32*)*, void (i32*, i32*)** %20, align 8, !dbg !131
  %22 = icmp ne void (i32*, i32*)* %21, null, !dbg !132
  br i1 %22, label %23, label %36, !dbg !133

23:                                               ; preds = %17
  %24 = load %struct.context*, %struct.context** @CC, align 8, !dbg !134
  %25 = getelementptr inbounds %struct.context, %struct.context* %24, i32 0, i32 0, !dbg !136
  %26 = getelementptr inbounds %struct.api_table, %struct.api_table* %25, i32 0, i32 2, !dbg !137
  %27 = load void (i32*, i32*)*, void (i32*, i32*)** %26, align 8, !dbg !137
  %28 = load i32*, i32** %5, align 8, !dbg !138
  %29 = load i32*, i32** %6, align 8, !dbg !139
  call void %27(i32* %28, i32* %29), !dbg !140
  %30 = load %struct.context*, %struct.context** @CC, align 8, !dbg !141
  %31 = getelementptr inbounds %struct.context, %struct.context* %30, i32 0, i32 0, !dbg !142
  %32 = getelementptr inbounds %struct.api_table, %struct.api_table* %31, i32 0, i32 1, !dbg !143
  %33 = load void (i32*, i32*)*, void (i32*, i32*)** %32, align 8, !dbg !143
  %34 = load i32*, i32** %4, align 8, !dbg !144
  %35 = load i32*, i32** %6, align 8, !dbg !145
  call void %33(i32* %34, i32* %35), !dbg !146
  br label %43, !dbg !147

36:                                               ; preds = %17
  %37 = load %struct.context*, %struct.context** @CC, align 8, !dbg !148
  %38 = getelementptr inbounds %struct.context, %struct.context* %37, i32 0, i32 0, !dbg !149
  %39 = getelementptr inbounds %struct.api_table, %struct.api_table* %38, i32 0, i32 1, !dbg !150
  %40 = load void (i32*, i32*)*, void (i32*, i32*)** %39, align 8, !dbg !150
  %41 = load i32*, i32** %5, align 8, !dbg !151
  %42 = load i32*, i32** %4, align 8, !dbg !152
  call void %40(i32* %41, i32* %42), !dbg !153
  br label %43

43:                                               ; preds = %36, %23
  ret void, !dbg !154
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @delete_context(%struct.context*) #0 !dbg !155 {
  %2 = alloca %struct.context*, align 8
  store %struct.context* %0, %struct.context** %2, align 8
  call void @llvm.dbg.declare(metadata %struct.context** %2, metadata !156, metadata !DIExpression()), !dbg !157
  %3 = load %struct.context*, %struct.context** %2, align 8, !dbg !158
  %4 = call i32 (%struct.context*, ...) bitcast (i32 (...)* @free to i32 (%struct.context*, ...)*)(%struct.context* %3), !dbg !159
  ret void, !dbg !160
}

declare dso_local i32 @free(...) #2

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !161 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca %struct.mesa_context*, align 8
  store i32 0, i32* %1, align 4
  call void @llvm.dbg.declare(metadata i32* %2, metadata !164, metadata !DIExpression()), !dbg !165
  call void @llvm.dbg.declare(metadata i32* %3, metadata !166, metadata !DIExpression()), !dbg !167
  call void @llvm.dbg.declare(metadata i32* %4, metadata !168, metadata !DIExpression()), !dbg !169
  call void @llvm.dbg.declare(metadata %struct.mesa_context** %5, metadata !170, metadata !DIExpression()), !dbg !171
  %6 = call i8* @malloc(i64 8), !dbg !172
  %7 = bitcast i8* %6 to %struct.mesa_context*, !dbg !173
  store %struct.mesa_context* %7, %struct.mesa_context** %5, align 8, !dbg !171
  %8 = call %struct.context* @create_context(), !dbg !174
  %9 = load %struct.mesa_context*, %struct.mesa_context** %5, align 8, !dbg !175
  %10 = getelementptr inbounds %struct.mesa_context, %struct.mesa_context* %9, i32 0, i32 0, !dbg !176
  store %struct.context* %8, %struct.context** %10, align 8, !dbg !177
  %11 = load %struct.mesa_context*, %struct.mesa_context** %5, align 8, !dbg !178
  call void @make_current(%struct.mesa_context* %11), !dbg !179
  call void @draw(i32* %2, i32* %3, i32* %4), !dbg !180
  %12 = load %struct.mesa_context*, %struct.mesa_context** %5, align 8, !dbg !181
  %13 = getelementptr inbounds %struct.mesa_context, %struct.mesa_context* %12, i32 0, i32 0, !dbg !182
  %14 = load %struct.context*, %struct.context** %13, align 8, !dbg !182
  call void @delete_context(%struct.context* %14), !dbg !183
  %15 = load %struct.mesa_context*, %struct.mesa_context** %5, align 8, !dbg !184
  %16 = call i32 (%struct.mesa_context*, ...) bitcast (i32 (...)* @free to i32 (%struct.mesa_context*, ...)*)(%struct.mesa_context* %15), !dbg !185
  ret i32 0, !dbg !186
}

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { argmemonly nounwind }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!27, !28, !29}
!llvm.ident = !{!30}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "CC", scope: !2, file: !3, line: 55, type: !7, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 9.0.0 (tags/RELEASE_900/final)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !5, globals: !26, nameTableKind: None)
!3 = !DIFile(filename: "basic_c_tests/spec-mesa.c", directory: "/home/peiming/Documents/Projects/LLVMRace/TestCases/PTABen")
!4 = !{}
!5 = !{!6, !7, !22}
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 64)
!8 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "context", file: !3, line: 26, size: 384, elements: !9)
!9 = !{!10, !21}
!10 = !DIDerivedType(tag: DW_TAG_member, name: "API", scope: !8, file: !3, line: 27, baseType: !11, size: 192)
!11 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "api_table", file: !3, line: 20, size: 192, elements: !12)
!12 = !{!13, !19, !20}
!13 = !DIDerivedType(tag: DW_TAG_member, name: "Begin", scope: !11, file: !3, line: 21, baseType: !14, size: 64)
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !15, size: 64)
!15 = !DISubroutineType(types: !16)
!16 = !{null, !17, !17}
!17 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !18, size: 64)
!18 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!19 = !DIDerivedType(tag: DW_TAG_member, name: "End", scope: !11, file: !3, line: 22, baseType: !14, size: 64, offset: 64)
!20 = !DIDerivedType(tag: DW_TAG_member, name: "Render", scope: !11, file: !3, line: 23, baseType: !14, size: 64, offset: 128)
!21 = !DIDerivedType(tag: DW_TAG_member, name: "Exec", scope: !8, file: !3, line: 28, baseType: !11, size: 192, offset: 192)
!22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !23, size: 64)
!23 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "mesa_context", file: !3, line: 31, size: 64, elements: !24)
!24 = !{!25}
!25 = !DIDerivedType(tag: DW_TAG_member, name: "ctx", scope: !23, file: !3, line: 32, baseType: !7, size: 64)
!26 = !{!0}
!27 = !{i32 2, !"Dwarf Version", i32 4}
!28 = !{i32 2, !"Debug Info Version", i32 3}
!29 = !{i32 1, !"wchar_size", i32 4}
!30 = !{!"clang version 9.0.0 (tags/RELEASE_900/final)"}
!31 = distinct !DISubprogram(name: "begin", scope: !3, file: !3, line: 8, type: !15, scopeLine: 8, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!32 = !DILocalVariable(name: "p", arg: 1, scope: !31, file: !3, line: 8, type: !17)
!33 = !DILocation(line: 8, column: 17, scope: !31)
!34 = !DILocalVariable(name: "q", arg: 2, scope: !31, file: !3, line: 8, type: !17)
!35 = !DILocation(line: 8, column: 25, scope: !31)
!36 = !DILocation(line: 9, column: 2, scope: !31)
!37 = !DILocation(line: 10, column: 1, scope: !31)
!38 = distinct !DISubprogram(name: "end", scope: !3, file: !3, line: 12, type: !15, scopeLine: 12, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!39 = !DILocalVariable(name: "p", arg: 1, scope: !38, file: !3, line: 12, type: !17)
!40 = !DILocation(line: 12, column: 15, scope: !38)
!41 = !DILocalVariable(name: "q", arg: 2, scope: !38, file: !3, line: 12, type: !17)
!42 = !DILocation(line: 12, column: 23, scope: !38)
!43 = !DILocation(line: 13, column: 2, scope: !38)
!44 = !DILocation(line: 14, column: 1, scope: !38)
!45 = distinct !DISubprogram(name: "render", scope: !3, file: !3, line: 16, type: !15, scopeLine: 16, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!46 = !DILocalVariable(name: "p", arg: 1, scope: !45, file: !3, line: 16, type: !17)
!47 = !DILocation(line: 16, column: 18, scope: !45)
!48 = !DILocalVariable(name: "q", arg: 2, scope: !45, file: !3, line: 16, type: !17)
!49 = !DILocation(line: 16, column: 26, scope: !45)
!50 = !DILocation(line: 17, column: 2, scope: !45)
!51 = !DILocation(line: 18, column: 1, scope: !45)
!52 = distinct !DISubprogram(name: "init_exec_pointers", scope: !3, file: !3, line: 35, type: !53, scopeLine: 35, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!53 = !DISubroutineType(types: !54)
!54 = !{null, !55}
!55 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 64)
!56 = !DILocalVariable(name: "table", arg: 1, scope: !52, file: !3, line: 35, type: !55)
!57 = !DILocation(line: 35, column: 44, scope: !52)
!58 = !DILocation(line: 36, column: 2, scope: !52)
!59 = !DILocation(line: 36, column: 9, scope: !52)
!60 = !DILocation(line: 36, column: 15, scope: !52)
!61 = !DILocation(line: 37, column: 2, scope: !52)
!62 = !DILocation(line: 37, column: 9, scope: !52)
!63 = !DILocation(line: 37, column: 13, scope: !52)
!64 = !DILocation(line: 38, column: 2, scope: !52)
!65 = !DILocation(line: 38, column: 9, scope: !52)
!66 = !DILocation(line: 38, column: 16, scope: !52)
!67 = !DILocation(line: 39, column: 1, scope: !52)
!68 = distinct !DISubprogram(name: "init_api_function", scope: !3, file: !3, line: 41, type: !69, scopeLine: 41, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!69 = !DISubroutineType(types: !70)
!70 = !{null, !7}
!71 = !DILocalVariable(name: "ctx", arg: 1, scope: !68, file: !3, line: 41, type: !7)
!72 = !DILocation(line: 41, column: 41, scope: !68)
!73 = !DILocation(line: 42, column: 22, scope: !68)
!74 = !DILocation(line: 42, column: 27, scope: !68)
!75 = !DILocation(line: 42, column: 2, scope: !68)
!76 = !DILocation(line: 43, column: 1, scope: !68)
!77 = distinct !DISubprogram(name: "create_context", scope: !3, file: !3, line: 45, type: !78, scopeLine: 45, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!78 = !DISubroutineType(types: !79)
!79 = !{!7}
!80 = !DILocalVariable(name: "ctx", scope: !77, file: !3, line: 47, type: !7)
!81 = !DILocation(line: 47, column: 19, scope: !77)
!82 = !DILocation(line: 47, column: 42, scope: !77)
!83 = !DILocation(line: 47, column: 25, scope: !77)
!84 = !DILocation(line: 48, column: 20, scope: !77)
!85 = !DILocation(line: 48, column: 2, scope: !77)
!86 = !DILocation(line: 51, column: 2, scope: !77)
!87 = !DILocation(line: 51, column: 7, scope: !77)
!88 = !DILocation(line: 51, column: 13, scope: !77)
!89 = !DILocation(line: 51, column: 18, scope: !77)
!90 = !DILocation(line: 52, column: 9, scope: !77)
!91 = !DILocation(line: 52, column: 2, scope: !77)
!92 = distinct !DISubprogram(name: "change_context", scope: !3, file: !3, line: 57, type: !69, scopeLine: 57, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!93 = !DILocalVariable(name: "ctx", arg: 1, scope: !92, file: !3, line: 57, type: !7)
!94 = !DILocation(line: 57, column: 38, scope: !92)
!95 = !DILocation(line: 58, column: 7, scope: !92)
!96 = !DILocation(line: 58, column: 5, scope: !92)
!97 = !DILocation(line: 59, column: 1, scope: !92)
!98 = distinct !DISubprogram(name: "make_current", scope: !3, file: !3, line: 61, type: !99, scopeLine: 61, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!99 = !DISubroutineType(types: !100)
!100 = !{null, !22}
!101 = !DILocalVariable(name: "ctx", arg: 1, scope: !98, file: !3, line: 61, type: !22)
!102 = !DILocation(line: 61, column: 41, scope: !98)
!103 = !DILocation(line: 62, column: 17, scope: !98)
!104 = !DILocation(line: 62, column: 22, scope: !98)
!105 = !DILocation(line: 62, column: 2, scope: !98)
!106 = !DILocation(line: 63, column: 1, scope: !98)
!107 = distinct !DISubprogram(name: "draw", scope: !3, file: !3, line: 65, type: !108, scopeLine: 65, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!108 = !DISubroutineType(types: !109)
!109 = !{null, !17, !17, !17}
!110 = !DILocalVariable(name: "p", arg: 1, scope: !107, file: !3, line: 65, type: !17)
!111 = !DILocation(line: 65, column: 16, scope: !107)
!112 = !DILocalVariable(name: "q", arg: 2, scope: !107, file: !3, line: 65, type: !17)
!113 = !DILocation(line: 65, column: 24, scope: !107)
!114 = !DILocalVariable(name: "r", arg: 3, scope: !107, file: !3, line: 65, type: !17)
!115 = !DILocation(line: 65, column: 32, scope: !107)
!116 = !DILocation(line: 66, column: 4, scope: !107)
!117 = !DILocation(line: 66, column: 8, scope: !107)
!118 = !DILocation(line: 66, column: 12, scope: !107)
!119 = !DILocation(line: 66, column: 19, scope: !107)
!120 = !DILocation(line: 66, column: 22, scope: !107)
!121 = !DILocation(line: 66, column: 2, scope: !107)
!122 = !DILocation(line: 67, column: 6, scope: !123)
!123 = distinct !DILexicalBlock(scope: !107, file: !3, line: 67, column: 6)
!124 = !DILocation(line: 67, column: 6, scope: !107)
!125 = !DILocation(line: 68, column: 7, scope: !123)
!126 = !DILocation(line: 68, column: 5, scope: !123)
!127 = !DILocation(line: 68, column: 3, scope: !123)
!128 = !DILocation(line: 69, column: 7, scope: !129)
!129 = distinct !DILexicalBlock(scope: !107, file: !3, line: 69, column: 6)
!130 = !DILocation(line: 69, column: 11, scope: !129)
!131 = !DILocation(line: 69, column: 15, scope: !129)
!132 = !DILocation(line: 69, column: 6, scope: !129)
!133 = !DILocation(line: 69, column: 6, scope: !107)
!134 = !DILocation(line: 70, column: 5, scope: !135)
!135 = distinct !DILexicalBlock(scope: !129, file: !3, line: 69, column: 23)
!136 = !DILocation(line: 70, column: 9, scope: !135)
!137 = !DILocation(line: 70, column: 13, scope: !135)
!138 = !DILocation(line: 70, column: 21, scope: !135)
!139 = !DILocation(line: 70, column: 24, scope: !135)
!140 = !DILocation(line: 70, column: 3, scope: !135)
!141 = !DILocation(line: 71, column: 5, scope: !135)
!142 = !DILocation(line: 71, column: 9, scope: !135)
!143 = !DILocation(line: 71, column: 13, scope: !135)
!144 = !DILocation(line: 71, column: 18, scope: !135)
!145 = !DILocation(line: 71, column: 21, scope: !135)
!146 = !DILocation(line: 71, column: 3, scope: !135)
!147 = !DILocation(line: 72, column: 2, scope: !135)
!148 = !DILocation(line: 74, column: 5, scope: !129)
!149 = !DILocation(line: 74, column: 9, scope: !129)
!150 = !DILocation(line: 74, column: 13, scope: !129)
!151 = !DILocation(line: 74, column: 18, scope: !129)
!152 = !DILocation(line: 74, column: 21, scope: !129)
!153 = !DILocation(line: 74, column: 3, scope: !129)
!154 = !DILocation(line: 75, column: 1, scope: !107)
!155 = distinct !DISubprogram(name: "delete_context", scope: !3, file: !3, line: 77, type: !69, scopeLine: 77, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!156 = !DILocalVariable(name: "ctx", arg: 1, scope: !155, file: !3, line: 77, type: !7)
!157 = !DILocation(line: 77, column: 38, scope: !155)
!158 = !DILocation(line: 78, column: 8, scope: !155)
!159 = !DILocation(line: 78, column: 2, scope: !155)
!160 = !DILocation(line: 79, column: 1, scope: !155)
!161 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 81, type: !162, scopeLine: 81, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!162 = !DISubroutineType(types: !163)
!163 = !{!18}
!164 = !DILocalVariable(name: "x", scope: !161, file: !3, line: 82, type: !18)
!165 = !DILocation(line: 82, column: 6, scope: !161)
!166 = !DILocalVariable(name: "y", scope: !161, file: !3, line: 82, type: !18)
!167 = !DILocation(line: 82, column: 8, scope: !161)
!168 = !DILocalVariable(name: "z", scope: !161, file: !3, line: 82, type: !18)
!169 = !DILocation(line: 82, column: 10, scope: !161)
!170 = !DILocalVariable(name: "mesa", scope: !161, file: !3, line: 83, type: !22)
!171 = !DILocation(line: 83, column: 24, scope: !161)
!172 = !DILocation(line: 83, column: 53, scope: !161)
!173 = !DILocation(line: 83, column: 31, scope: !161)
!174 = !DILocation(line: 84, column: 14, scope: !161)
!175 = !DILocation(line: 84, column: 2, scope: !161)
!176 = !DILocation(line: 84, column: 8, scope: !161)
!177 = !DILocation(line: 84, column: 12, scope: !161)
!178 = !DILocation(line: 85, column: 15, scope: !161)
!179 = !DILocation(line: 85, column: 2, scope: !161)
!180 = !DILocation(line: 86, column: 2, scope: !161)
!181 = !DILocation(line: 87, column: 17, scope: !161)
!182 = !DILocation(line: 87, column: 23, scope: !161)
!183 = !DILocation(line: 87, column: 2, scope: !161)
!184 = !DILocation(line: 88, column: 7, scope: !161)
!185 = !DILocation(line: 88, column: 2, scope: !161)
!186 = !DILocation(line: 89, column: 2, scope: !161)

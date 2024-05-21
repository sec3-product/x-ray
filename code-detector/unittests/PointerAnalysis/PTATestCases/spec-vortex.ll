; ModuleID = 'basic_c_tests/spec-vortex.c'
source_filename = "basic_c_tests/spec-vortex.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.Ory = type { i8** }
%struct.Rectangle = type { void (i32*, float*)* }

@Theory = dso_local global %struct.Ory* null, align 8, !dbg !0
@PartLib01 = dso_local global i32* null, align 8, !dbg !11

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @Rectangle_draw(i32*, float*) #0 !dbg !24 {
  %3 = alloca i32*, align 8
  %4 = alloca float*, align 8
  store i32* %0, i32** %3, align 8
  call void @llvm.dbg.declare(metadata i32** %3, metadata !29, metadata !DIExpression()), !dbg !30
  store float* %1, float** %4, align 8
  call void @llvm.dbg.declare(metadata float** %4, metadata !31, metadata !DIExpression()), !dbg !32
  ret void, !dbg !33
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @CoreMoreCore(i8**) #0 !dbg !34 {
  %2 = alloca i8**, align 8
  %3 = alloca i64, align 8
  store i8** %0, i8*** %2, align 8
  call void @llvm.dbg.declare(metadata i8*** %2, metadata !37, metadata !DIExpression()), !dbg !38
  call void @llvm.dbg.declare(metadata i64* %3, metadata !39, metadata !DIExpression()), !dbg !41
  store i64 0, i64* %3, align 8, !dbg !41
  %4 = call i8* @Void_ExtendCore(i64* %3), !dbg !42
  %5 = load i8**, i8*** %2, align 8, !dbg !43
  store i8* %4, i8** %5, align 8, !dbg !44
  ret void, !dbg !45
}

declare dso_local i8* @Void_ExtendCore(i64*) #2

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @MemNewOry() #0 !dbg !46 {
  %1 = alloca i8**, align 8
  %2 = alloca i8*, align 8
  call void @llvm.dbg.declare(metadata i8*** %1, metadata !49, metadata !DIExpression()), !dbg !50
  store i8** null, i8*** %1, align 8, !dbg !50
  call void @llvm.dbg.declare(metadata i8** %2, metadata !51, metadata !DIExpression()), !dbg !52
  %3 = call noalias i8* @calloc(i64 1, i64 8) #4, !dbg !53
  store i8* %3, i8** %2, align 8, !dbg !52
  call void @CoreMoreCore(i8** bitcast (%struct.Ory** @Theory to i8**)), !dbg !54
  %4 = load i8*, i8** %2, align 8, !dbg !55
  %5 = bitcast i8* %4 to i8**, !dbg !57
  %6 = load %struct.Ory*, %struct.Ory** @Theory, align 8, !dbg !58
  %7 = getelementptr inbounds %struct.Ory, %struct.Ory* %6, i32 0, i32 0, !dbg !59
  store i8** %5, i8*** %7, align 8, !dbg !60
  %8 = icmp ne i8** %5, null, !dbg !61
  br i1 %8, label %9, label %16, !dbg !62

9:                                                ; preds = %0
  %10 = load %struct.Ory*, %struct.Ory** @Theory, align 8, !dbg !63
  %11 = getelementptr inbounds %struct.Ory, %struct.Ory* %10, i32 0, i32 0, !dbg !65
  %12 = load i8**, i8*** %11, align 8, !dbg !65
  store i8** %12, i8*** %1, align 8, !dbg !66
  %13 = load i8**, i8*** %1, align 8, !dbg !67
  %14 = bitcast i8** %13 to i8*, !dbg !68
  %15 = load i8**, i8*** %1, align 8, !dbg !69
  store i8* %14, i8** %15, align 8, !dbg !70
  br label %16, !dbg !71

16:                                               ; preds = %9, %0
  ret void, !dbg !72
}

; Function Attrs: nounwind
declare dso_local noalias i8* @calloc(i64, i64) #3

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @MemGetAddr(i32, i32, i8**) #0 !dbg !73 {
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  %6 = alloca i8**, align 8
  store i32 %0, i32* %4, align 4
  call void @llvm.dbg.declare(metadata i32* %4, metadata !77, metadata !DIExpression()), !dbg !78
  store i32 %1, i32* %5, align 4
  call void @llvm.dbg.declare(metadata i32* %5, metadata !79, metadata !DIExpression()), !dbg !80
  store i8** %2, i8*** %6, align 8
  call void @llvm.dbg.declare(metadata i8*** %6, metadata !81, metadata !DIExpression()), !dbg !82
  %7 = load %struct.Ory*, %struct.Ory** @Theory, align 8, !dbg !83
  %8 = getelementptr inbounds %struct.Ory, %struct.Ory* %7, i32 0, i32 0, !dbg !84
  %9 = load i8**, i8*** %8, align 8, !dbg !84
  %10 = load i32, i32* %4, align 4, !dbg !85
  %11 = zext i32 %10 to i64, !dbg !83
  %12 = getelementptr inbounds i8*, i8** %9, i64 %11, !dbg !83
  %13 = load i8*, i8** %12, align 8, !dbg !83
  %14 = bitcast i8* %13 to i8**, !dbg !86
  %15 = load i32, i32* %5, align 4, !dbg !87
  %16 = zext i32 %15 to i64, !dbg !88
  %17 = getelementptr inbounds i8*, i8** %14, i64 %16, !dbg !88
  %18 = load i8*, i8** %17, align 8, !dbg !89
  %19 = load i8**, i8*** %6, align 8, !dbg !90
  store i8* %18, i8** %19, align 8, !dbg !91
  ret void, !dbg !92
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @Object_GetImage(i8**) #0 !dbg !93 {
  %2 = alloca i8**, align 8
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i8** %0, i8*** %2, align 8
  call void @llvm.dbg.declare(metadata i8*** %2, metadata !94, metadata !DIExpression()), !dbg !95
  call void @llvm.dbg.declare(metadata i32* %3, metadata !96, metadata !DIExpression()), !dbg !97
  call void @llvm.dbg.declare(metadata i32* %4, metadata !98, metadata !DIExpression()), !dbg !99
  %5 = load i32, i32* %3, align 4, !dbg !100
  %6 = load i32, i32* %4, align 4, !dbg !101
  %7 = load i8**, i8*** %2, align 8, !dbg !102
  call void @MemGetAddr(i32 %5, i32 %6, i8** %7), !dbg !103
  ret void, !dbg !104
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @Rectangle_new0() #0 !dbg !105 {
  %1 = alloca %struct.Rectangle*, align 8
  call void @llvm.dbg.declare(metadata %struct.Rectangle** %1, metadata !106, metadata !DIExpression()), !dbg !113
  store %struct.Rectangle* null, %struct.Rectangle** %1, align 8, !dbg !113
  %2 = bitcast %struct.Rectangle** %1 to i8**, !dbg !114
  call void @Object_GetImage(i8** %2), !dbg !115
  %3 = load %struct.Rectangle*, %struct.Rectangle** %1, align 8, !dbg !116
  %4 = getelementptr inbounds %struct.Rectangle, %struct.Rectangle* %3, i32 0, i32 0, !dbg !117
  store void (i32*, float*)* @Rectangle_draw, void (i32*, float*)** %4, align 8, !dbg !118
  ret void, !dbg !119
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @PartLib_Create(i32**) #0 !dbg !120 {
  %2 = alloca i32**, align 8
  store i32** %0, i32*** %2, align 8
  call void @llvm.dbg.declare(metadata i32*** %2, metadata !124, metadata !DIExpression()), !dbg !125
  %3 = load i32**, i32*** %2, align 8, !dbg !126
  %4 = bitcast i32** %3 to i8**, !dbg !127
  call void @Object_GetImage(i8** %4), !dbg !128
  ret void, !dbg !129
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !130 {
  %1 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  call void @PartLib_Create(i32** @PartLib01), !dbg !133
  %2 = load i32*, i32** @PartLib01, align 8, !dbg !134
  %3 = call i32 (i32*, void (i32*, float*)*, ...) bitcast (i32 (...)* @EXPECTEDFAIL_NOALIAS to i32 (i32*, void (i32*, float*)*, ...)*)(i32* %2, void (i32*, float*)* @Rectangle_draw), !dbg !135
  ret i32 0, !dbg !136
}

declare dso_local i32 @EXPECTEDFAIL_NOALIAS(...) #2

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!20, !21, !22}
!llvm.ident = !{!23}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "Theory", scope: !2, file: !3, line: 17, type: !15, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 9.0.0 (tags/RELEASE_900/final)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !5, globals: !10, nameTableKind: None)
!3 = !DIFile(filename: "basic_c_tests/spec-vortex.c", directory: "/home/peiming/Documents/Projects/LLVMRace/TestCases/PTABen")
!4 = !{}
!5 = !{!6, !9, !8}
!6 = !DIDerivedType(tag: DW_TAG_typedef, name: "addrtype", file: !3, line: 10, baseType: !7)
!7 = !DIDerivedType(tag: DW_TAG_typedef, name: "AddrType", file: !3, line: 9, baseType: !8)
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !6, size: 64)
!10 = !{!0, !11}
!11 = !DIGlobalVariableExpression(var: !12, expr: !DIExpression())
!12 = distinct !DIGlobalVariable(name: "PartLib01", scope: !2, file: !3, line: 70, type: !13, isLocal: false, isDefinition: true)
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 64)
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !16, size: 64)
!16 = !DIDerivedType(tag: DW_TAG_typedef, name: "OryType", file: !3, line: 15, baseType: !17)
!17 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Ory", file: !3, line: 12, size: 64, elements: !18)
!18 = !{!19}
!19 = !DIDerivedType(tag: DW_TAG_member, name: "ChunkAddr", scope: !17, file: !3, line: 14, baseType: !9, size: 64)
!20 = !{i32 2, !"Dwarf Version", i32 4}
!21 = !{i32 2, !"Debug Info Version", i32 3}
!22 = !{i32 1, !"wchar_size", i32 4}
!23 = !{!"clang version 9.0.0 (tags/RELEASE_900/final)"}
!24 = distinct !DISubprogram(name: "Rectangle_draw", scope: !3, file: !3, line: 24, type: !25, scopeLine: 24, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!25 = !DISubroutineType(types: !26)
!26 = !{null, !13, !27}
!27 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !28, size: 64)
!28 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!29 = !DILocalVariable(name: "p", arg: 1, scope: !24, file: !3, line: 24, type: !13)
!30 = !DILocation(line: 24, column: 26, scope: !24)
!31 = !DILocalVariable(name: "q", arg: 2, scope: !24, file: !3, line: 24, type: !27)
!32 = !DILocation(line: 24, column: 36, scope: !24)
!33 = !DILocation(line: 24, column: 40, scope: !24)
!34 = distinct !DISubprogram(name: "CoreMoreCore", scope: !3, file: !3, line: 27, type: !35, scopeLine: 28, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!35 = !DISubroutineType(types: !36)
!36 = !{null, !9}
!37 = !DILocalVariable(name: "Region", arg: 1, scope: !34, file: !3, line: 27, type: !9)
!38 = !DILocation(line: 27, column: 29, scope: !34)
!39 = !DILocalVariable(name: "AllocCore1", scope: !34, file: !3, line: 29, type: !40)
!40 = !DIBasicType(name: "long int", size: 64, encoding: DW_ATE_signed)
!41 = !DILocation(line: 29, column: 7, scope: !34)
!42 = !DILocation(line: 30, column: 22, scope: !34)
!43 = !DILocation(line: 30, column: 3, scope: !34)
!44 = !DILocation(line: 30, column: 10, scope: !34)
!45 = !DILocation(line: 31, column: 1, scope: !34)
!46 = distinct !DISubprogram(name: "MemNewOry", scope: !3, file: !3, line: 33, type: !47, scopeLine: 34, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!47 = !DISubroutineType(types: !48)
!48 = !{null}
!49 = !DILocalVariable(name: "ChunkAddrPtr", scope: !46, file: !3, line: 35, type: !9)
!50 = !DILocation(line: 35, column: 12, scope: !46)
!51 = !DILocalVariable(name: "ChunkBlk", scope: !46, file: !3, line: 36, type: !6)
!52 = !DILocation(line: 36, column: 11, scope: !46)
!53 = !DILocation(line: 36, column: 32, scope: !46)
!54 = !DILocation(line: 38, column: 2, scope: !46)
!55 = !DILocation(line: 40, column: 38, scope: !56)
!56 = distinct !DILexicalBlock(scope: !46, file: !3, line: 40, column: 6)
!57 = !DILocation(line: 40, column: 27, scope: !56)
!58 = !DILocation(line: 40, column: 7, scope: !56)
!59 = !DILocation(line: 40, column: 15, scope: !56)
!60 = !DILocation(line: 40, column: 25, scope: !56)
!61 = !DILocation(line: 40, column: 48, scope: !56)
!62 = !DILocation(line: 40, column: 6, scope: !46)
!63 = !DILocation(line: 42, column: 18, scope: !64)
!64 = distinct !DILexicalBlock(scope: !56, file: !3, line: 41, column: 2)
!65 = !DILocation(line: 42, column: 26, scope: !64)
!66 = !DILocation(line: 42, column: 16, scope: !64)
!67 = !DILocation(line: 43, column: 29, scope: !64)
!68 = !DILocation(line: 43, column: 19, scope: !64)
!69 = !DILocation(line: 43, column: 4, scope: !64)
!70 = !DILocation(line: 43, column: 17, scope: !64)
!71 = !DILocation(line: 44, column: 2, scope: !64)
!72 = !DILocation(line: 45, column: 1, scope: !46)
!73 = distinct !DISubprogram(name: "MemGetAddr", scope: !3, file: !3, line: 47, type: !74, scopeLine: 48, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!74 = !DISubroutineType(types: !75)
!75 = !{null, !76, !76, !9}
!76 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!77 = !DILocalVariable(name: "Chunk", arg: 1, scope: !73, file: !3, line: 47, type: !76)
!78 = !DILocation(line: 47, column: 26, scope: !73)
!79 = !DILocalVariable(name: "index", arg: 2, scope: !73, file: !3, line: 47, type: !76)
!80 = !DILocation(line: 47, column: 42, scope: !73)
!81 = !DILocalVariable(name: "Addr", arg: 3, scope: !73, file: !3, line: 47, type: !9)
!82 = !DILocation(line: 47, column: 59, scope: !73)
!83 = !DILocation(line: 49, column: 26, scope: !73)
!84 = !DILocation(line: 49, column: 34, scope: !73)
!85 = !DILocation(line: 49, column: 44, scope: !73)
!86 = !DILocation(line: 49, column: 13, scope: !73)
!87 = !DILocation(line: 49, column: 54, scope: !73)
!88 = !DILocation(line: 49, column: 53, scope: !73)
!89 = !DILocation(line: 49, column: 10, scope: !73)
!90 = !DILocation(line: 49, column: 3, scope: !73)
!91 = !DILocation(line: 49, column: 8, scope: !73)
!92 = !DILocation(line: 50, column: 1, scope: !73)
!93 = distinct !DISubprogram(name: "Object_GetImage", scope: !3, file: !3, line: 52, type: !35, scopeLine: 53, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!94 = !DILocalVariable(name: "This", arg: 1, scope: !93, file: !3, line: 52, type: !9)
!95 = !DILocation(line: 52, column: 32, scope: !93)
!96 = !DILocalVariable(name: "chunk", scope: !93, file: !3, line: 54, type: !76)
!97 = !DILocation(line: 54, column: 11, scope: !93)
!98 = !DILocalVariable(name: "index", scope: !93, file: !3, line: 54, type: !76)
!99 = !DILocation(line: 54, column: 18, scope: !93)
!100 = !DILocation(line: 55, column: 13, scope: !93)
!101 = !DILocation(line: 55, column: 20, scope: !93)
!102 = !DILocation(line: 55, column: 27, scope: !93)
!103 = !DILocation(line: 55, column: 2, scope: !93)
!104 = !DILocation(line: 56, column: 1, scope: !93)
!105 = distinct !DISubprogram(name: "Rectangle_new0", scope: !3, file: !3, line: 58, type: !47, scopeLine: 59, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!106 = !DILocalVariable(name: "Image", scope: !105, file: !3, line: 60, type: !107)
!107 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !108, size: 64)
!108 = !DIDerivedType(tag: DW_TAG_typedef, name: "Rectangle", file: !3, line: 22, baseType: !109)
!109 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Rectangle", file: !3, line: 19, size: 64, elements: !110)
!110 = !{!111}
!111 = !DIDerivedType(tag: DW_TAG_member, name: "draw", scope: !109, file: !3, line: 21, baseType: !112, size: 64)
!112 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !25, size: 64)
!113 = !DILocation(line: 60, column: 13, scope: !105)
!114 = !DILocation(line: 61, column: 18, scope: !105)
!115 = !DILocation(line: 61, column: 2, scope: !105)
!116 = !DILocation(line: 62, column: 2, scope: !105)
!117 = !DILocation(line: 62, column: 9, scope: !105)
!118 = !DILocation(line: 62, column: 14, scope: !105)
!119 = !DILocation(line: 63, column: 1, scope: !105)
!120 = distinct !DISubprogram(name: "PartLib_Create", scope: !3, file: !3, line: 65, type: !121, scopeLine: 66, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!121 = !DISubroutineType(types: !122)
!122 = !{null, !123}
!123 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 64)
!124 = !DILocalVariable(name: "PrimalObj", arg: 1, scope: !120, file: !3, line: 65, type: !123)
!125 = !DILocation(line: 65, column: 27, scope: !120)
!126 = !DILocation(line: 67, column: 30, scope: !120)
!127 = !DILocation(line: 67, column: 18, scope: !120)
!128 = !DILocation(line: 67, column: 2, scope: !120)
!129 = !DILocation(line: 68, column: 1, scope: !120)
!130 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 72, type: !131, scopeLine: 73, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!131 = !DISubroutineType(types: !132)
!132 = !{!14}
!133 = !DILocation(line: 74, column: 2, scope: !130)
!134 = !DILocation(line: 75, column: 23, scope: !130)
!135 = !DILocation(line: 75, column: 2, scope: !130)
!136 = !DILocation(line: 76, column: 2, scope: !130)

; ModuleID = 'basic_c_tests/spec-parser.c'
source_filename = "basic_c_tests/spec-parser.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.Word_struct = type { i32*, %struct.X_node_struct* }
%struct.X_node_struct = type { i32*, %struct.X_node_struct* }
%struct.clause_struct = type { %struct.clause_struct* }

@sentence = common dso_local global [250 x %struct.Word_struct] zeroinitializer, align 16, !dbg !0

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i8* @xalloc(i32) #0 !dbg !33 {
  %2 = alloca i32, align 4
  store i32 %0, i32* %2, align 4
  call void @llvm.dbg.declare(metadata i32* %2, metadata !36, metadata !DIExpression()), !dbg !37
  %3 = call noalias i8* @malloc(i64 1000) #4, !dbg !38
  ret i8* %3, !dbg !39
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind
declare dso_local noalias i8* @malloc(i64) #2

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @build_clause() #0 !dbg !40 {
  %1 = alloca %struct.clause_struct*, align 8
  %2 = alloca %struct.clause_struct*, align 8
  %3 = alloca %struct.clause_struct*, align 8
  call void @llvm.dbg.declare(metadata %struct.clause_struct** %1, metadata !43, metadata !DIExpression()), !dbg !44
  store %struct.clause_struct* null, %struct.clause_struct** %1, align 8, !dbg !44
  call void @llvm.dbg.declare(metadata %struct.clause_struct** %2, metadata !45, metadata !DIExpression()), !dbg !46
  call void @llvm.dbg.declare(metadata %struct.clause_struct** %3, metadata !47, metadata !DIExpression()), !dbg !48
  %4 = call i8* @xalloc(i32 8), !dbg !49
  %5 = bitcast i8* %4 to %struct.clause_struct*, !dbg !50
  store %struct.clause_struct* %5, %struct.clause_struct** %2, align 8, !dbg !51
  br label %6, !dbg !52

6:                                                ; preds = %9, %0
  %7 = load %struct.clause_struct*, %struct.clause_struct** %2, align 8, !dbg !53
  %8 = icmp ne %struct.clause_struct* %7, null, !dbg !54
  br i1 %8, label %9, label %18, !dbg !52

9:                                                ; preds = %6
  %10 = load %struct.clause_struct*, %struct.clause_struct** %2, align 8, !dbg !55
  %11 = getelementptr inbounds %struct.clause_struct, %struct.clause_struct* %10, i32 0, i32 0, !dbg !57
  %12 = load %struct.clause_struct*, %struct.clause_struct** %11, align 8, !dbg !57
  store %struct.clause_struct* %12, %struct.clause_struct** %3, align 8, !dbg !58
  %13 = load %struct.clause_struct*, %struct.clause_struct** %1, align 8, !dbg !59
  %14 = load %struct.clause_struct*, %struct.clause_struct** %2, align 8, !dbg !60
  %15 = getelementptr inbounds %struct.clause_struct, %struct.clause_struct* %14, i32 0, i32 0, !dbg !61
  store %struct.clause_struct* %13, %struct.clause_struct** %15, align 8, !dbg !62
  %16 = load %struct.clause_struct*, %struct.clause_struct** %2, align 8, !dbg !63
  store %struct.clause_struct* %16, %struct.clause_struct** %1, align 8, !dbg !64
  %17 = load %struct.clause_struct*, %struct.clause_struct** %3, align 8, !dbg !65
  store %struct.clause_struct* %17, %struct.clause_struct** %2, align 8, !dbg !66
  br label %6, !dbg !52, !llvm.loop !67

18:                                               ; preds = %6
  %19 = load i32*, i32** getelementptr inbounds ([250 x %struct.Word_struct], [250 x %struct.Word_struct]* @sentence, i64 0, i64 0, i32 0), align 16, !dbg !69
  %20 = bitcast i32* %19 to i8*, !dbg !69
  %21 = load %struct.clause_struct*, %struct.clause_struct** %2, align 8, !dbg !69
  %22 = bitcast %struct.clause_struct* %21 to i8*, !dbg !69
  call void @__aser_no_alias__(i8* %20, i8* %22), !dbg !69
  ret void, !dbg !70
}

declare dso_local void @__aser_no_alias__(i8*, i8*) #3

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @special_string(i32) #0 !dbg !71 {
  %2 = alloca i32, align 4
  %3 = alloca %struct.X_node_struct*, align 8
  store i32 %0, i32* %2, align 4
  call void @llvm.dbg.declare(metadata i32* %2, metadata !74, metadata !DIExpression()), !dbg !75
  call void @llvm.dbg.declare(metadata %struct.X_node_struct** %3, metadata !76, metadata !DIExpression()), !dbg !77
  %4 = call i8* @xalloc(i32 16), !dbg !78
  %5 = bitcast i8* %4 to %struct.X_node_struct*, !dbg !79
  %6 = load i32, i32* %2, align 4, !dbg !80
  %7 = sext i32 %6 to i64, !dbg !81
  %8 = getelementptr inbounds [250 x %struct.Word_struct], [250 x %struct.Word_struct]* @sentence, i64 0, i64 %7, !dbg !81
  %9 = getelementptr inbounds %struct.Word_struct, %struct.Word_struct* %8, i32 0, i32 1, !dbg !82
  store %struct.X_node_struct* %5, %struct.X_node_struct** %9, align 8, !dbg !83
  %10 = load i32, i32* %2, align 4, !dbg !84
  %11 = sext i32 %10 to i64, !dbg !86
  %12 = getelementptr inbounds [250 x %struct.Word_struct], [250 x %struct.Word_struct]* @sentence, i64 0, i64 %11, !dbg !86
  %13 = getelementptr inbounds %struct.Word_struct, %struct.Word_struct* %12, i32 0, i32 1, !dbg !87
  %14 = load %struct.X_node_struct*, %struct.X_node_struct** %13, align 8, !dbg !87
  store %struct.X_node_struct* %14, %struct.X_node_struct** %3, align 8, !dbg !88
  br label %15, !dbg !89

15:                                               ; preds = %26, %1
  %16 = load %struct.X_node_struct*, %struct.X_node_struct** %3, align 8, !dbg !90
  %17 = icmp ne %struct.X_node_struct* %16, null, !dbg !92
  br i1 %17, label %18, label %30, !dbg !93

18:                                               ; preds = %15
  %19 = load i32, i32* %2, align 4, !dbg !94
  %20 = sext i32 %19 to i64, !dbg !96
  %21 = getelementptr inbounds [250 x %struct.Word_struct], [250 x %struct.Word_struct]* @sentence, i64 0, i64 %20, !dbg !96
  %22 = getelementptr inbounds %struct.Word_struct, %struct.Word_struct* %21, i32 0, i32 0, !dbg !97
  %23 = load i32*, i32** %22, align 16, !dbg !97
  %24 = load %struct.X_node_struct*, %struct.X_node_struct** %3, align 8, !dbg !98
  %25 = getelementptr inbounds %struct.X_node_struct, %struct.X_node_struct* %24, i32 0, i32 0, !dbg !99
  store i32* %23, i32** %25, align 8, !dbg !100
  br label %26, !dbg !101

26:                                               ; preds = %18
  %27 = load %struct.X_node_struct*, %struct.X_node_struct** %3, align 8, !dbg !102
  %28 = getelementptr inbounds %struct.X_node_struct, %struct.X_node_struct* %27, i32 0, i32 1, !dbg !103
  %29 = load %struct.X_node_struct*, %struct.X_node_struct** %28, align 8, !dbg !103
  store %struct.X_node_struct* %29, %struct.X_node_struct** %3, align 8, !dbg !104
  br label %15, !dbg !105, !llvm.loop !106

30:                                               ; preds = %15
  ret void, !dbg !108
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !109 {
  %1 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  call void @build_clause(), !dbg !112
  call void @special_string(i32 10), !dbg !113
  ret i32 0, !dbg !114
}

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!29, !30, !31}
!llvm.ident = !{!32}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "sentence", scope: !2, file: !3, line: 33, type: !21, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 9.0.0 (tags/RELEASE_900/final)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !5, globals: !20, nameTableKind: None)
!3 = !DIFile(filename: "basic_c_tests/spec-parser.c", directory: "/home/peiming/Documents/Projects/LLVMRace/TestCases/PTABen")
!4 = !{}
!5 = !{!6, !7, !12}
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 64)
!8 = !DIDerivedType(tag: DW_TAG_typedef, name: "Clause", file: !3, line: 21, baseType: !9)
!9 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "clause_struct", file: !3, line: 22, size: 64, elements: !10)
!10 = !{!11}
!11 = !DIDerivedType(tag: DW_TAG_member, name: "next", scope: !9, file: !3, line: 23, baseType: !7, size: 64)
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 64)
!13 = !DIDerivedType(tag: DW_TAG_typedef, name: "X_node", file: !3, line: 9, baseType: !14)
!14 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "X_node_struct", file: !3, line: 10, size: 128, elements: !15)
!15 = !{!16, !19}
!16 = !DIDerivedType(tag: DW_TAG_member, name: "string", scope: !14, file: !3, line: 11, baseType: !17, size: 64)
!17 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !18, size: 64)
!18 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!19 = !DIDerivedType(tag: DW_TAG_member, name: "next", scope: !14, file: !3, line: 12, baseType: !12, size: 64, offset: 64)
!20 = !{!0}
!21 = !DICompositeType(tag: DW_TAG_array_type, baseType: !22, size: 32000, elements: !27)
!22 = !DIDerivedType(tag: DW_TAG_typedef, name: "Word", file: !3, line: 15, baseType: !23)
!23 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Word_struct", file: !3, line: 16, size: 128, elements: !24)
!24 = !{!25, !26}
!25 = !DIDerivedType(tag: DW_TAG_member, name: "string", scope: !23, file: !3, line: 17, baseType: !17, size: 64)
!26 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !23, file: !3, line: 18, baseType: !12, size: 64, offset: 64)
!27 = !{!28}
!28 = !DISubrange(count: 250)
!29 = !{i32 2, !"Dwarf Version", i32 4}
!30 = !{i32 2, !"Debug Info Version", i32 3}
!31 = !{i32 1, !"wchar_size", i32 4}
!32 = !{!"clang version 9.0.0 (tags/RELEASE_900/final)"}
!33 = distinct !DISubprogram(name: "xalloc", scope: !3, file: !3, line: 29, type: !34, scopeLine: 29, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!34 = !DISubroutineType(types: !35)
!35 = !{!6, !18}
!36 = !DILocalVariable(name: "size", arg: 1, scope: !33, file: !3, line: 29, type: !18)
!37 = !DILocation(line: 29, column: 19, scope: !33)
!38 = !DILocation(line: 30, column: 18, scope: !33)
!39 = !DILocation(line: 30, column: 2, scope: !33)
!40 = distinct !DISubprogram(name: "build_clause", scope: !3, file: !3, line: 35, type: !41, scopeLine: 35, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!41 = !DISubroutineType(types: !42)
!42 = !{null}
!43 = !DILocalVariable(name: "c", scope: !40, file: !3, line: 36, type: !7)
!44 = !DILocation(line: 36, column: 13, scope: !40)
!45 = !DILocalVariable(name: "c1", scope: !40, file: !3, line: 36, type: !7)
!46 = !DILocation(line: 36, column: 22, scope: !40)
!47 = !DILocalVariable(name: "c2", scope: !40, file: !3, line: 36, type: !7)
!48 = !DILocation(line: 36, column: 27, scope: !40)
!49 = !DILocation(line: 38, column: 17, scope: !40)
!50 = !DILocation(line: 38, column: 7, scope: !40)
!51 = !DILocation(line: 38, column: 5, scope: !40)
!52 = !DILocation(line: 39, column: 2, scope: !40)
!53 = !DILocation(line: 39, column: 8, scope: !40)
!54 = !DILocation(line: 39, column: 11, scope: !40)
!55 = !DILocation(line: 40, column: 8, scope: !56)
!56 = distinct !DILexicalBlock(scope: !40, file: !3, line: 39, column: 20)
!57 = !DILocation(line: 40, column: 12, scope: !56)
!58 = !DILocation(line: 40, column: 6, scope: !56)
!59 = !DILocation(line: 41, column: 14, scope: !56)
!60 = !DILocation(line: 41, column: 3, scope: !56)
!61 = !DILocation(line: 41, column: 7, scope: !56)
!62 = !DILocation(line: 41, column: 12, scope: !56)
!63 = !DILocation(line: 42, column: 7, scope: !56)
!64 = !DILocation(line: 42, column: 5, scope: !56)
!65 = !DILocation(line: 43, column: 8, scope: !56)
!66 = !DILocation(line: 43, column: 6, scope: !56)
!67 = distinct !{!67, !52, !68}
!68 = !DILocation(line: 44, column: 2, scope: !40)
!69 = !DILocation(line: 45, column: 2, scope: !40)
!70 = !DILocation(line: 46, column: 1, scope: !40)
!71 = distinct !DISubprogram(name: "special_string", scope: !3, file: !3, line: 48, type: !72, scopeLine: 48, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!72 = !DISubroutineType(types: !73)
!73 = !{null, !18}
!74 = !DILocalVariable(name: "i", arg: 1, scope: !71, file: !3, line: 48, type: !18)
!75 = !DILocation(line: 48, column: 25, scope: !71)
!76 = !DILocalVariable(name: "e", scope: !71, file: !3, line: 49, type: !12)
!77 = !DILocation(line: 49, column: 14, scope: !71)
!78 = !DILocation(line: 52, column: 29, scope: !71)
!79 = !DILocation(line: 52, column: 18, scope: !71)
!80 = !DILocation(line: 52, column: 11, scope: !71)
!81 = !DILocation(line: 52, column: 2, scope: !71)
!82 = !DILocation(line: 52, column: 14, scope: !71)
!83 = !DILocation(line: 52, column: 16, scope: !71)
!84 = !DILocation(line: 53, column: 20, scope: !85)
!85 = distinct !DILexicalBlock(scope: !71, file: !3, line: 53, column: 2)
!86 = !DILocation(line: 53, column: 11, scope: !85)
!87 = !DILocation(line: 53, column: 23, scope: !85)
!88 = !DILocation(line: 53, column: 9, scope: !85)
!89 = !DILocation(line: 53, column: 7, scope: !85)
!90 = !DILocation(line: 53, column: 26, scope: !91)
!91 = distinct !DILexicalBlock(scope: !85, file: !3, line: 53, column: 2)
!92 = !DILocation(line: 53, column: 28, scope: !91)
!93 = !DILocation(line: 53, column: 2, scope: !85)
!94 = !DILocation(line: 54, column: 27, scope: !95)
!95 = distinct !DILexicalBlock(scope: !91, file: !3, line: 53, column: 50)
!96 = !DILocation(line: 54, column: 18, scope: !95)
!97 = !DILocation(line: 54, column: 30, scope: !95)
!98 = !DILocation(line: 54, column: 6, scope: !95)
!99 = !DILocation(line: 54, column: 9, scope: !95)
!100 = !DILocation(line: 54, column: 16, scope: !95)
!101 = !DILocation(line: 55, column: 2, scope: !95)
!102 = !DILocation(line: 53, column: 41, scope: !91)
!103 = !DILocation(line: 53, column: 44, scope: !91)
!104 = !DILocation(line: 53, column: 39, scope: !91)
!105 = !DILocation(line: 53, column: 2, scope: !91)
!106 = distinct !{!106, !93, !107}
!107 = !DILocation(line: 55, column: 2, scope: !85)
!108 = !DILocation(line: 56, column: 1, scope: !71)
!109 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 58, type: !110, scopeLine: 58, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!110 = !DISubroutineType(types: !111)
!111 = !{!18}
!112 = !DILocation(line: 59, column: 2, scope: !109)
!113 = !DILocation(line: 60, column: 2, scope: !109)
!114 = !DILocation(line: 61, column: 2, scope: !109)

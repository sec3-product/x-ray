; ModuleID = 'basic_c_tests/global-call-twoparms.c'
source_filename = "basic_c_tests/global-call-twoparms.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.MyStruct = type { i32, void (i32**, i32**)* }

@x = common dso_local global i32 0, align 4, !dbg !0
@y = common dso_local global i32 0, align 4, !dbg !20
@global = common dso_local global %struct.MyStruct zeroinitializer, align 8, !dbg !8

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @foo(i32**, i32**) #0 !dbg !26 {
  %3 = alloca i32**, align 8
  %4 = alloca i32**, align 8
  store i32** %0, i32*** %3, align 8
  call void @llvm.dbg.declare(metadata i32*** %3, metadata !27, metadata !DIExpression()), !dbg !28
  store i32** %1, i32*** %4, align 8
  call void @llvm.dbg.declare(metadata i32*** %4, metadata !29, metadata !DIExpression()), !dbg !30
  %5 = load i32**, i32*** %3, align 8, !dbg !31
  store i32* @x, i32** %5, align 8, !dbg !32
  %6 = load i32**, i32*** %4, align 8, !dbg !33
  store i32* @y, i32** %6, align 8, !dbg !34
  ret void, !dbg !35
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @bar(i32**, i32**) #0 !dbg !36 {
  %3 = alloca i32**, align 8
  %4 = alloca i32**, align 8
  store i32** %0, i32*** %3, align 8
  call void @llvm.dbg.declare(metadata i32*** %3, metadata !37, metadata !DIExpression()), !dbg !38
  store i32** %1, i32*** %4, align 8
  call void @llvm.dbg.declare(metadata i32*** %4, metadata !39, metadata !DIExpression()), !dbg !40
  %5 = load i32**, i32*** %3, align 8, !dbg !41
  store i32* @x, i32** %5, align 8, !dbg !42
  %6 = load i32**, i32*** %4, align 8, !dbg !43
  store i32* @x, i32** %6, align 8, !dbg !44
  ret void, !dbg !45
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @init() #0 !dbg !46 {
  store void (i32**, i32**)* @foo, void (i32**, i32**)** getelementptr inbounds (%struct.MyStruct, %struct.MyStruct* @global, i32 0, i32 1), align 8, !dbg !49
  ret void, !dbg !50
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @init2() #0 !dbg !51 {
  store void (i32**, i32**)* @bar, void (i32**, i32**)** getelementptr inbounds (%struct.MyStruct, %struct.MyStruct* @global, i32 0, i32 1), align 8, !dbg !52
  ret void, !dbg !53
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @run(i32**, i32**) #0 !dbg !54 {
  %3 = alloca i32**, align 8
  %4 = alloca i32**, align 8
  store i32** %0, i32*** %3, align 8
  call void @llvm.dbg.declare(metadata i32*** %3, metadata !55, metadata !DIExpression()), !dbg !56
  store i32** %1, i32*** %4, align 8
  call void @llvm.dbg.declare(metadata i32*** %4, metadata !57, metadata !DIExpression()), !dbg !58
  %5 = load void (i32**, i32**)*, void (i32**, i32**)** getelementptr inbounds (%struct.MyStruct, %struct.MyStruct* @global, i32 0, i32 1), align 8, !dbg !59
  %6 = load i32**, i32*** %3, align 8, !dbg !60
  %7 = load i32**, i32*** %4, align 8, !dbg !61
  call void %5(i32** %6, i32** %7), !dbg !62
  ret void, !dbg !63
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !64 {
  %1 = alloca i32*, align 8
  %2 = alloca i32*, align 8
  %3 = alloca i32**, align 8
  %4 = alloca i32**, align 8
  call void @llvm.dbg.declare(metadata i32** %1, metadata !67, metadata !DIExpression()), !dbg !68
  call void @llvm.dbg.declare(metadata i32** %2, metadata !69, metadata !DIExpression()), !dbg !70
  call void @llvm.dbg.declare(metadata i32*** %3, metadata !71, metadata !DIExpression()), !dbg !72
  call void @llvm.dbg.declare(metadata i32*** %4, metadata !73, metadata !DIExpression()), !dbg !74
  store i32** %1, i32*** %3, align 8, !dbg !75
  store i32** %2, i32*** %4, align 8, !dbg !76
  call void @init(), !dbg !77
  call void @init2(), !dbg !78
  %5 = load i32**, i32*** %3, align 8, !dbg !79
  %6 = load i32**, i32*** %4, align 8, !dbg !80
  call void @run(i32** %5, i32** %6), !dbg !81
  %7 = load i32**, i32*** %3, align 8, !dbg !82
  %8 = load i32*, i32** %7, align 8, !dbg !82
  %9 = bitcast i32* %8 to i8*, !dbg !82
  %10 = load i32**, i32*** %4, align 8, !dbg !82
  %11 = load i32*, i32** %10, align 8, !dbg !82
  %12 = bitcast i32* %11 to i8*, !dbg !82
  call void @__aser_alias__(i8* %9, i8* %12), !dbg !82
  ret i32 0, !dbg !83
}

declare dso_local void @__aser_alias__(i8*, i8*) #2

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!22, !23, !24}
!llvm.ident = !{!25}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "x", scope: !2, file: !3, line: 14, type: !13, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 9.0.0 (tags/RELEASE_900/final)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !5, globals: !7, nameTableKind: None)
!3 = !DIFile(filename: "basic_c_tests/global-call-twoparms.c", directory: "/home/peiming/Documents/Projects/LLVMRace/TestCases/PTABen")
!4 = !{}
!5 = !{!6}
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!7 = !{!8, !0, !20}
!8 = !DIGlobalVariableExpression(var: !9, expr: !DIExpression())
!9 = distinct !DIGlobalVariable(name: "global", scope: !2, file: !3, line: 13, type: !10, isLocal: false, isDefinition: true)
!10 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "MyStruct", file: !3, line: 8, size: 128, elements: !11)
!11 = !{!12, !14}
!12 = !DIDerivedType(tag: DW_TAG_member, name: "f1", scope: !10, file: !3, line: 9, baseType: !13, size: 32)
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !DIDerivedType(tag: DW_TAG_member, name: "fp", scope: !10, file: !3, line: 10, baseType: !15, size: 64, offset: 64)
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !16, size: 64)
!16 = !DISubroutineType(types: !17)
!17 = !{null, !18, !18}
!18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !19, size: 64)
!19 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 64)
!20 = !DIGlobalVariableExpression(var: !21, expr: !DIExpression())
!21 = distinct !DIGlobalVariable(name: "y", scope: !2, file: !3, line: 14, type: !13, isLocal: false, isDefinition: true)
!22 = !{i32 2, !"Dwarf Version", i32 4}
!23 = !{i32 2, !"Debug Info Version", i32 3}
!24 = !{i32 1, !"wchar_size", i32 4}
!25 = !{!"clang version 9.0.0 (tags/RELEASE_900/final)"}
!26 = distinct !DISubprogram(name: "foo", scope: !3, file: !3, line: 16, type: !16, scopeLine: 16, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!27 = !DILocalVariable(name: "pp", arg: 1, scope: !26, file: !3, line: 16, type: !18)
!28 = !DILocation(line: 16, column: 16, scope: !26)
!29 = !DILocalVariable(name: "qq", arg: 2, scope: !26, file: !3, line: 16, type: !18)
!30 = !DILocation(line: 16, column: 26, scope: !26)
!31 = !DILocation(line: 17, column: 3, scope: !26)
!32 = !DILocation(line: 17, column: 6, scope: !26)
!33 = !DILocation(line: 18, column: 3, scope: !26)
!34 = !DILocation(line: 18, column: 6, scope: !26)
!35 = !DILocation(line: 19, column: 1, scope: !26)
!36 = distinct !DISubprogram(name: "bar", scope: !3, file: !3, line: 21, type: !16, scopeLine: 21, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!37 = !DILocalVariable(name: "pp", arg: 1, scope: !36, file: !3, line: 21, type: !18)
!38 = !DILocation(line: 21, column: 16, scope: !36)
!39 = !DILocalVariable(name: "qq", arg: 2, scope: !36, file: !3, line: 21, type: !18)
!40 = !DILocation(line: 21, column: 26, scope: !36)
!41 = !DILocation(line: 22, column: 3, scope: !36)
!42 = !DILocation(line: 22, column: 6, scope: !36)
!43 = !DILocation(line: 23, column: 3, scope: !36)
!44 = !DILocation(line: 23, column: 6, scope: !36)
!45 = !DILocation(line: 24, column: 1, scope: !36)
!46 = distinct !DISubprogram(name: "init", scope: !3, file: !3, line: 26, type: !47, scopeLine: 26, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!47 = !DISubroutineType(types: !48)
!48 = !{null}
!49 = !DILocation(line: 27, column: 12, scope: !46)
!50 = !DILocation(line: 28, column: 1, scope: !46)
!51 = distinct !DISubprogram(name: "init2", scope: !3, file: !3, line: 30, type: !47, scopeLine: 30, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!52 = !DILocation(line: 31, column: 12, scope: !51)
!53 = !DILocation(line: 32, column: 1, scope: !51)
!54 = distinct !DISubprogram(name: "run", scope: !3, file: !3, line: 34, type: !16, scopeLine: 34, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!55 = !DILocalVariable(name: "pp", arg: 1, scope: !54, file: !3, line: 34, type: !18)
!56 = !DILocation(line: 34, column: 16, scope: !54)
!57 = !DILocalVariable(name: "qq", arg: 2, scope: !54, file: !3, line: 34, type: !18)
!58 = !DILocation(line: 34, column: 25, scope: !54)
!59 = !DILocation(line: 35, column: 9, scope: !54)
!60 = !DILocation(line: 35, column: 12, scope: !54)
!61 = !DILocation(line: 35, column: 16, scope: !54)
!62 = !DILocation(line: 35, column: 2, scope: !54)
!63 = !DILocation(line: 36, column: 1, scope: !54)
!64 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 38, type: !65, scopeLine: 38, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!65 = !DISubroutineType(types: !66)
!66 = !{!13}
!67 = !DILocalVariable(name: "p", scope: !64, file: !3, line: 39, type: !19)
!68 = !DILocation(line: 39, column: 7, scope: !64)
!69 = !DILocalVariable(name: "q", scope: !64, file: !3, line: 39, type: !19)
!70 = !DILocation(line: 39, column: 11, scope: !64)
!71 = !DILocalVariable(name: "pp", scope: !64, file: !3, line: 40, type: !18)
!72 = !DILocation(line: 40, column: 8, scope: !64)
!73 = !DILocalVariable(name: "qq", scope: !64, file: !3, line: 40, type: !18)
!74 = !DILocation(line: 40, column: 14, scope: !64)
!75 = !DILocation(line: 41, column: 5, scope: !64)
!76 = !DILocation(line: 42, column: 5, scope: !64)
!77 = !DILocation(line: 43, column: 2, scope: !64)
!78 = !DILocation(line: 44, column: 2, scope: !64)
!79 = !DILocation(line: 45, column: 6, scope: !64)
!80 = !DILocation(line: 45, column: 10, scope: !64)
!81 = !DILocation(line: 45, column: 2, scope: !64)
!82 = !DILocation(line: 49, column: 2, scope: !64)
!83 = !DILocation(line: 50, column: 1, scope: !64)

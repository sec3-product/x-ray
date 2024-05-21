; ModuleID = 'basic_c_tests/global-call-struct.c'
source_filename = "basic_c_tests/global-call-struct.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.MyStruct = type { [20 x i8], i32, i32* }

@x = common dso_local global i32 0, align 4, !dbg !0
@global = dso_local global %struct.MyStruct { [20 x i8] c"abcdefg\00\00\00\00\00\00\00\00\00\00\00\00\00", i32 20, i32* @x }, align 8, !dbg !8
@y = common dso_local global i32 0, align 4, !dbg !21

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @foo(i32**, i32**) #0 !dbg !27 {
  %3 = alloca i32**, align 8
  %4 = alloca i32**, align 8
  store i32** %0, i32*** %3, align 8
  call void @llvm.dbg.declare(metadata i32*** %3, metadata !31, metadata !DIExpression()), !dbg !32
  store i32** %1, i32*** %4, align 8
  call void @llvm.dbg.declare(metadata i32*** %4, metadata !33, metadata !DIExpression()), !dbg !34
  %5 = load i32**, i32*** %3, align 8, !dbg !35
  store i32* @x, i32** %5, align 8, !dbg !36
  %6 = load i32**, i32*** %4, align 8, !dbg !37
  store i32* @y, i32** %6, align 8, !dbg !38
  ret void, !dbg !39
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @bar(i32**, i32**) #0 !dbg !40 {
  %3 = alloca i32**, align 8
  %4 = alloca i32**, align 8
  store i32** %0, i32*** %3, align 8
  call void @llvm.dbg.declare(metadata i32*** %3, metadata !41, metadata !DIExpression()), !dbg !42
  store i32** %1, i32*** %4, align 8
  call void @llvm.dbg.declare(metadata i32*** %4, metadata !43, metadata !DIExpression()), !dbg !44
  %5 = load i32**, i32*** %3, align 8, !dbg !45
  store i32* @x, i32** %5, align 8, !dbg !46
  %6 = load i32**, i32*** %4, align 8, !dbg !47
  store i32* @x, i32** %6, align 8, !dbg !48
  ret void, !dbg !49
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !50 {
  %1 = alloca i32*, align 8
  %2 = alloca i32*, align 8
  %3 = alloca i32**, align 8
  %4 = alloca i32**, align 8
  call void @llvm.dbg.declare(metadata i32** %1, metadata !53, metadata !DIExpression()), !dbg !54
  call void @llvm.dbg.declare(metadata i32** %2, metadata !55, metadata !DIExpression()), !dbg !56
  call void @llvm.dbg.declare(metadata i32*** %3, metadata !57, metadata !DIExpression()), !dbg !58
  call void @llvm.dbg.declare(metadata i32*** %4, metadata !59, metadata !DIExpression()), !dbg !60
  store i32** %1, i32*** %3, align 8, !dbg !61
  store i32** %2, i32*** %4, align 8, !dbg !62
  %5 = load i32**, i32*** %3, align 8, !dbg !63
  %6 = load i32**, i32*** %4, align 8, !dbg !64
  call void @bar(i32** %5, i32** %6), !dbg !65
  %7 = load i32*, i32** %1, align 8, !dbg !66
  %8 = bitcast i32* %7 to i8*, !dbg !66
  %9 = load i32*, i32** %2, align 8, !dbg !66
  %10 = bitcast i32* %9 to i8*, !dbg !66
  call void @__aser_alias__(i8* %8, i8* %10), !dbg !66
  %11 = load i32*, i32** getelementptr inbounds (%struct.MyStruct, %struct.MyStruct* @global, i32 0, i32 2), align 8, !dbg !67
  %12 = bitcast i32* %11 to i8*, !dbg !67
  %13 = load i32**, i32*** %4, align 8, !dbg !67
  %14 = load i32*, i32** %13, align 8, !dbg !67
  %15 = bitcast i32* %14 to i8*, !dbg !67
  call void @__aser_alias__(i8* %12, i8* %15), !dbg !67
  ret i32 0, !dbg !68
}

declare dso_local void @__aser_alias__(i8*, i8*) #2

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!23, !24, !25}
!llvm.ident = !{!26}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "x", scope: !2, file: !3, line: 14, type: !18, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 9.0.0 (tags/RELEASE_900/final)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !5, globals: !7, nameTableKind: None)
!3 = !DIFile(filename: "basic_c_tests/global-call-struct.c", directory: "/home/peiming/Documents/Projects/LLVMRace/TestCases/PTABen")
!4 = !{}
!5 = !{!6}
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!7 = !{!8, !0, !21}
!8 = !DIGlobalVariableExpression(var: !9, expr: !DIExpression())
!9 = distinct !DIGlobalVariable(name: "global", scope: !2, file: !3, line: 15, type: !10, isLocal: false, isDefinition: true)
!10 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "MyStruct", file: !3, line: 8, size: 256, elements: !11)
!11 = !{!12, !17, !19}
!12 = !DIDerivedType(tag: DW_TAG_member, name: "f0", scope: !10, file: !3, line: 9, baseType: !13, size: 160)
!13 = !DICompositeType(tag: DW_TAG_array_type, baseType: !14, size: 160, elements: !15)
!14 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!15 = !{!16}
!16 = !DISubrange(count: 20)
!17 = !DIDerivedType(tag: DW_TAG_member, name: "f1", scope: !10, file: !3, line: 10, baseType: !18, size: 32, offset: 160)
!18 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!19 = !DIDerivedType(tag: DW_TAG_member, name: "f2", scope: !10, file: !3, line: 11, baseType: !20, size: 64, offset: 192)
!20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !18, size: 64)
!21 = !DIGlobalVariableExpression(var: !22, expr: !DIExpression())
!22 = distinct !DIGlobalVariable(name: "y", scope: !2, file: !3, line: 14, type: !18, isLocal: false, isDefinition: true)
!23 = !{i32 2, !"Dwarf Version", i32 4}
!24 = !{i32 2, !"Debug Info Version", i32 3}
!25 = !{i32 1, !"wchar_size", i32 4}
!26 = !{!"clang version 9.0.0 (tags/RELEASE_900/final)"}
!27 = distinct !DISubprogram(name: "foo", scope: !3, file: !3, line: 17, type: !28, scopeLine: 17, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!28 = !DISubroutineType(types: !29)
!29 = !{null, !30, !30}
!30 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !20, size: 64)
!31 = !DILocalVariable(name: "pp", arg: 1, scope: !27, file: !3, line: 17, type: !30)
!32 = !DILocation(line: 17, column: 16, scope: !27)
!33 = !DILocalVariable(name: "qq", arg: 2, scope: !27, file: !3, line: 17, type: !30)
!34 = !DILocation(line: 17, column: 26, scope: !27)
!35 = !DILocation(line: 18, column: 3, scope: !27)
!36 = !DILocation(line: 18, column: 6, scope: !27)
!37 = !DILocation(line: 19, column: 3, scope: !27)
!38 = !DILocation(line: 19, column: 6, scope: !27)
!39 = !DILocation(line: 20, column: 1, scope: !27)
!40 = distinct !DISubprogram(name: "bar", scope: !3, file: !3, line: 22, type: !28, scopeLine: 22, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!41 = !DILocalVariable(name: "pp", arg: 1, scope: !40, file: !3, line: 22, type: !30)
!42 = !DILocation(line: 22, column: 16, scope: !40)
!43 = !DILocalVariable(name: "qq", arg: 2, scope: !40, file: !3, line: 22, type: !30)
!44 = !DILocation(line: 22, column: 26, scope: !40)
!45 = !DILocation(line: 23, column: 3, scope: !40)
!46 = !DILocation(line: 23, column: 6, scope: !40)
!47 = !DILocation(line: 24, column: 3, scope: !40)
!48 = !DILocation(line: 24, column: 6, scope: !40)
!49 = !DILocation(line: 25, column: 1, scope: !40)
!50 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 27, type: !51, scopeLine: 27, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!51 = !DISubroutineType(types: !52)
!52 = !{!18}
!53 = !DILocalVariable(name: "p", scope: !50, file: !3, line: 28, type: !20)
!54 = !DILocation(line: 28, column: 7, scope: !50)
!55 = !DILocalVariable(name: "q", scope: !50, file: !3, line: 28, type: !20)
!56 = !DILocation(line: 28, column: 11, scope: !50)
!57 = !DILocalVariable(name: "pp", scope: !50, file: !3, line: 29, type: !30)
!58 = !DILocation(line: 29, column: 8, scope: !50)
!59 = !DILocalVariable(name: "qq", scope: !50, file: !3, line: 29, type: !30)
!60 = !DILocation(line: 29, column: 14, scope: !50)
!61 = !DILocation(line: 30, column: 5, scope: !50)
!62 = !DILocation(line: 31, column: 5, scope: !50)
!63 = !DILocation(line: 32, column: 6, scope: !50)
!64 = !DILocation(line: 32, column: 9, scope: !50)
!65 = !DILocation(line: 32, column: 2, scope: !50)
!66 = !DILocation(line: 33, column: 2, scope: !50)
!67 = !DILocation(line: 34, column: 2, scope: !50)
!68 = !DILocation(line: 35, column: 1, scope: !50)

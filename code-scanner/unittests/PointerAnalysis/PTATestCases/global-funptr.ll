; ModuleID = 'basic_c_tests/global-funptr.c'
source_filename = "basic_c_tests/global-funptr.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.MyStruct = type { void (...)*, i32* }

@y = common dso_local global i32 0, align 4, !dbg !0
@p = common dso_local global i32* null, align 8, !dbg !21
@x = common dso_local global i32 0, align 4, !dbg !19
@context = dso_local global %struct.MyStruct { void (...)* bitcast (void ()* @foo to void (...)*), i32* @x }, align 8, !dbg !8

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @foo() #0 !dbg !27 {
  store i32* @y, i32** @p, align 8, !dbg !30
  ret void, !dbg !31
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !32 {
  %1 = alloca i32, align 4
  %2 = alloca i32*, align 8
  store i32 0, i32* %1, align 4
  %3 = load void (...)*, void (...)** getelementptr inbounds (%struct.MyStruct, %struct.MyStruct* @context, i32 0, i32 0), align 8, !dbg !35
  call void (...) %3(), !dbg !36
  call void @llvm.dbg.declare(metadata i32** %2, metadata !37, metadata !DIExpression()), !dbg !38
  %4 = load i32*, i32** @p, align 8, !dbg !39
  store i32* %4, i32** %2, align 8, !dbg !38
  %5 = load i32*, i32** %2, align 8, !dbg !40
  %6 = bitcast i32* %5 to i8*, !dbg !40
  call void @__aser_alias__(i8* %6, i8* bitcast (i32* @y to i8*)), !dbg !40
  ret i32 0, !dbg !41
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare dso_local void @__aser_alias__(i8*, i8*) #2

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!23, !24, !25}
!llvm.ident = !{!26}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "y", scope: !2, file: !3, line: 8, type: !18, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 9.0.0 (tags/RELEASE_900/final)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !5, globals: !7, nameTableKind: None)
!3 = !DIFile(filename: "basic_c_tests/global-funptr.c", directory: "/home/peiming/Documents/Projects/LLVMRace/TestCases/PTABen")
!4 = !{}
!5 = !{!6}
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!7 = !{!8, !19, !0, !21}
!8 = !DIGlobalVariableExpression(var: !9, expr: !DIExpression())
!9 = distinct !DIGlobalVariable(name: "context", scope: !2, file: !3, line: 20, type: !10, isLocal: false, isDefinition: true)
!10 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "MyStruct", file: !3, line: 15, size: 128, elements: !11)
!11 = !{!12, !16}
!12 = !DIDerivedType(tag: DW_TAG_member, name: "fp", scope: !10, file: !3, line: 16, baseType: !13, size: 64)
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 64)
!14 = !DISubroutineType(types: !15)
!15 = !{null, null}
!16 = !DIDerivedType(tag: DW_TAG_member, name: "f1", scope: !10, file: !3, line: 17, baseType: !17, size: 64, offset: 64)
!17 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !18, size: 64)
!18 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!19 = !DIGlobalVariableExpression(var: !20, expr: !DIExpression())
!20 = distinct !DIGlobalVariable(name: "x", scope: !2, file: !3, line: 8, type: !18, isLocal: false, isDefinition: true)
!21 = !DIGlobalVariableExpression(var: !22, expr: !DIExpression())
!22 = distinct !DIGlobalVariable(name: "p", scope: !2, file: !3, line: 9, type: !17, isLocal: false, isDefinition: true)
!23 = !{i32 2, !"Dwarf Version", i32 4}
!24 = !{i32 2, !"Debug Info Version", i32 3}
!25 = !{i32 1, !"wchar_size", i32 4}
!26 = !{!"clang version 9.0.0 (tags/RELEASE_900/final)"}
!27 = distinct !DISubprogram(name: "foo", scope: !3, file: !3, line: 11, type: !28, scopeLine: 11, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!28 = !DISubroutineType(types: !29)
!29 = !{null}
!30 = !DILocation(line: 12, column: 4, scope: !27)
!31 = !DILocation(line: 13, column: 1, scope: !27)
!32 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 22, type: !33, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!33 = !DISubroutineType(types: !34)
!34 = !{!18}
!35 = !DILocation(line: 24, column: 12, scope: !32)
!36 = !DILocation(line: 24, column: 2, scope: !32)
!37 = !DILocalVariable(name: "q", scope: !32, file: !3, line: 25, type: !17)
!38 = !DILocation(line: 25, column: 7, scope: !32)
!39 = !DILocation(line: 25, column: 11, scope: !32)
!40 = !DILocation(line: 26, column: 2, scope: !32)
!41 = !DILocation(line: 27, column: 2, scope: !32)

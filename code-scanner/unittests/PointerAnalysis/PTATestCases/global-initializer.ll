; ModuleID = 'basic_c_tests/global-initializer.c'
source_filename = "basic_c_tests/global-initializer.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@p = common dso_local global i32* null, align 8, !dbg !0
@pp = dso_local global i32** @p, align 8, !dbg !8
@q = common dso_local global i32* null, align 8, !dbg !17
@qq = dso_local global i32** @q, align 8, !dbg !13
@x = common dso_local global i32 0, align 4, !dbg !15

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @foo() #0 !dbg !23 {
  store i32* @x, i32** @p, align 8, !dbg !26
  ret void, !dbg !27
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @bar() #0 !dbg !28 {
  store i32* @x, i32** @q, align 8, !dbg !29
  ret void, !dbg !30
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !31 {
  %1 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  %2 = load i32**, i32*** @pp, align 8, !dbg !34
  %3 = load i32*, i32** %2, align 8, !dbg !34
  %4 = bitcast i32* %3 to i8*, !dbg !34
  %5 = load i32**, i32*** @qq, align 8, !dbg !34
  %6 = load i32*, i32** %5, align 8, !dbg !34
  %7 = bitcast i32* %6 to i8*, !dbg !34
  call void @__aser_alias__(i8* %4, i8* %7), !dbg !34
  call void @foo(), !dbg !35
  call void @bar(), !dbg !36
  ret i32 0, !dbg !37
}

declare dso_local void @__aser_alias__(i8*, i8*) #1

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!19, !20, !21}
!llvm.ident = !{!22}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "p", scope: !2, file: !3, line: 11, type: !11, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 9.0.0 (tags/RELEASE_900/final)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !5, globals: !7, nameTableKind: None)
!3 = !DIFile(filename: "basic_c_tests/global-initializer.c", directory: "/home/peiming/Documents/Projects/LLVMRace/TestCases/PTABen")
!4 = !{}
!5 = !{!6}
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!7 = !{!8, !13, !15, !0, !17}
!8 = !DIGlobalVariableExpression(var: !9, expr: !DIExpression())
!9 = distinct !DIGlobalVariable(name: "pp", scope: !2, file: !3, line: 12, type: !10, isLocal: false, isDefinition: true)
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 64)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DIGlobalVariableExpression(var: !14, expr: !DIExpression())
!14 = distinct !DIGlobalVariable(name: "qq", scope: !2, file: !3, line: 13, type: !10, isLocal: false, isDefinition: true)
!15 = !DIGlobalVariableExpression(var: !16, expr: !DIExpression())
!16 = distinct !DIGlobalVariable(name: "x", scope: !2, file: !3, line: 10, type: !12, isLocal: false, isDefinition: true)
!17 = !DIGlobalVariableExpression(var: !18, expr: !DIExpression())
!18 = distinct !DIGlobalVariable(name: "q", scope: !2, file: !3, line: 11, type: !11, isLocal: false, isDefinition: true)
!19 = !{i32 2, !"Dwarf Version", i32 4}
!20 = !{i32 2, !"Debug Info Version", i32 3}
!21 = !{i32 1, !"wchar_size", i32 4}
!22 = !{!"clang version 9.0.0 (tags/RELEASE_900/final)"}
!23 = distinct !DISubprogram(name: "foo", scope: !3, file: !3, line: 15, type: !24, scopeLine: 15, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!24 = !DISubroutineType(types: !25)
!25 = !{null}
!26 = !DILocation(line: 16, column: 4, scope: !23)
!27 = !DILocation(line: 17, column: 1, scope: !23)
!28 = distinct !DISubprogram(name: "bar", scope: !3, file: !3, line: 19, type: !24, scopeLine: 19, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!29 = !DILocation(line: 20, column: 4, scope: !28)
!30 = !DILocation(line: 21, column: 1, scope: !28)
!31 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 23, type: !32, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!32 = !DISubroutineType(types: !33)
!33 = !{!12}
!34 = !DILocation(line: 24, column: 2, scope: !31)
!35 = !DILocation(line: 25, column: 2, scope: !31)
!36 = !DILocation(line: 26, column: 2, scope: !31)
!37 = !DILocation(line: 27, column: 2, scope: !31)

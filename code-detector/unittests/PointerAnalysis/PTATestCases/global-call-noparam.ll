; ModuleID = 'basic_c_tests/global-call-noparam.c'
source_filename = "basic_c_tests/global-call-noparam.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@p = dso_local global i32* null, align 8, !dbg !0
@q = dso_local global i32* null, align 8, !dbg !8
@c = common dso_local global i32 0, align 4, !dbg !12

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @foo() #0 !dbg !18 {
  %1 = load i32*, i32** @p, align 8, !dbg !21
  %2 = bitcast i32* %1 to i8*, !dbg !21
  %3 = load i32*, i32** @q, align 8, !dbg !21
  %4 = bitcast i32* %3 to i8*, !dbg !21
  call void @__aser_alias__(i8* %2, i8* %4), !dbg !21
  ret void, !dbg !22
}

declare dso_local void @__aser_alias__(i8*, i8*) #1

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @bar() #0 !dbg !23 {
  store i32* @c, i32** @q, align 8, !dbg !24
  ret void, !dbg !25
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !26 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  call void @bar(), !dbg !29
  call void @foo(), !dbg !30
  call void @llvm.dbg.declare(metadata i32* %1, metadata !31, metadata !DIExpression()), !dbg !32
  call void @llvm.dbg.declare(metadata i32* %2, metadata !33, metadata !DIExpression()), !dbg !34
  store i32* %1, i32** @p, align 8, !dbg !35
  %3 = load i32*, i32** @p, align 8, !dbg !36
  store i32* %3, i32** @q, align 8, !dbg !37
  store i32* @c, i32** @p, align 8, !dbg !38
  ret i32 0, !dbg !39
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!14, !15, !16}
!llvm.ident = !{!17}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "p", scope: !2, file: !3, line: 8, type: !10, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 9.0.0 (tags/RELEASE_900/final)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !5, globals: !7, nameTableKind: None)
!3 = !DIFile(filename: "basic_c_tests/global-call-noparam.c", directory: "/home/peiming/Documents/Projects/LLVMRace/TestCases/PTABen")
!4 = !{}
!5 = !{!6}
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!7 = !{!0, !8, !12}
!8 = !DIGlobalVariableExpression(var: !9, expr: !DIExpression())
!9 = distinct !DIGlobalVariable(name: "q", scope: !2, file: !3, line: 9, type: !10, isLocal: false, isDefinition: true)
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 64)
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !DIGlobalVariableExpression(var: !13, expr: !DIExpression())
!13 = distinct !DIGlobalVariable(name: "c", scope: !2, file: !3, line: 10, type: !11, isLocal: false, isDefinition: true)
!14 = !{i32 2, !"Dwarf Version", i32 4}
!15 = !{i32 2, !"Debug Info Version", i32 3}
!16 = !{i32 1, !"wchar_size", i32 4}
!17 = !{!"clang version 9.0.0 (tags/RELEASE_900/final)"}
!18 = distinct !DISubprogram(name: "foo", scope: !3, file: !3, line: 12, type: !19, scopeLine: 12, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!19 = !DISubroutineType(types: !20)
!20 = !{null}
!21 = !DILocation(line: 13, column: 2, scope: !18)
!22 = !DILocation(line: 14, column: 1, scope: !18)
!23 = distinct !DISubprogram(name: "bar", scope: !3, file: !3, line: 16, type: !19, scopeLine: 16, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!24 = !DILocation(line: 17, column: 4, scope: !23)
!25 = !DILocation(line: 18, column: 1, scope: !23)
!26 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 20, type: !27, scopeLine: 20, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!27 = !DISubroutineType(types: !28)
!28 = !{!11}
!29 = !DILocation(line: 21, column: 5, scope: !26)
!30 = !DILocation(line: 22, column: 5, scope: !26)
!31 = !DILocalVariable(name: "a", scope: !26, file: !3, line: 24, type: !11)
!32 = !DILocation(line: 24, column: 6, scope: !26)
!33 = !DILocalVariable(name: "b", scope: !26, file: !3, line: 24, type: !11)
!34 = !DILocation(line: 24, column: 9, scope: !26)
!35 = !DILocation(line: 25, column: 4, scope: !26)
!36 = !DILocation(line: 26, column: 6, scope: !26)
!37 = !DILocation(line: 26, column: 4, scope: !26)
!38 = !DILocation(line: 27, column: 4, scope: !26)
!39 = !DILocation(line: 28, column: 1, scope: !26)

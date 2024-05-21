; ModuleID = 'basic_c_tests/CI-global.c'
source_filename = "basic_c_tests/CI-global.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@global = common dso_local global i32 0, align 4, !dbg !0
@p_global = common dso_local global i32* null, align 8, !dbg !8

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @foo() #0 !dbg !16 {
  store i32* @global, i32** @p_global, align 8, !dbg !19
  ret void, !dbg !20
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !21 {
  %1 = alloca i32, align 4
  %2 = alloca i32*, align 8
  store i32 0, i32* %1, align 4
  call void @llvm.dbg.declare(metadata i32** %2, metadata !24, metadata !DIExpression()), !dbg !25
  store i32* @global, i32** %2, align 8, !dbg !26
  call void @foo(), !dbg !27
  %3 = load i32*, i32** %2, align 8, !dbg !28
  %4 = bitcast i32* %3 to i8*, !dbg !28
  %5 = load i32*, i32** @p_global, align 8, !dbg !28
  %6 = bitcast i32* %5 to i8*, !dbg !28
  call void @__aser_alias__(i8* %4, i8* %6), !dbg !28
  ret i32 0, !dbg !29
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare dso_local void @__aser_alias__(i8*, i8*) #2

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!12, !13, !14}
!llvm.ident = !{!15}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "global", scope: !2, file: !3, line: 9, type: !11, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 9.0.0 (tags/RELEASE_900/final)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !5, globals: !7, nameTableKind: None)
!3 = !DIFile(filename: "basic_c_tests/CI-global.c", directory: "/home/peiming/Documents/Projects/LLVMRace/TestCases/PTABen")
!4 = !{}
!5 = !{!6}
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!7 = !{!0, !8}
!8 = !DIGlobalVariableExpression(var: !9, expr: !DIExpression())
!9 = distinct !DIGlobalVariable(name: "p_global", scope: !2, file: !3, line: 10, type: !10, isLocal: false, isDefinition: true)
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 64)
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !{i32 2, !"Dwarf Version", i32 4}
!13 = !{i32 2, !"Debug Info Version", i32 3}
!14 = !{i32 1, !"wchar_size", i32 4}
!15 = !{!"clang version 9.0.0 (tags/RELEASE_900/final)"}
!16 = distinct !DISubprogram(name: "foo", scope: !3, file: !3, line: 12, type: !17, scopeLine: 12, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!17 = !DISubroutineType(types: !18)
!18 = !{null}
!19 = !DILocation(line: 13, column: 11, scope: !16)
!20 = !DILocation(line: 14, column: 1, scope: !16)
!21 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 16, type: !22, scopeLine: 16, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!22 = !DISubroutineType(types: !23)
!23 = !{!11}
!24 = !DILocalVariable(name: "p_local", scope: !21, file: !3, line: 17, type: !10)
!25 = !DILocation(line: 17, column: 7, scope: !21)
!26 = !DILocation(line: 18, column: 10, scope: !21)
!27 = !DILocation(line: 19, column: 2, scope: !21)
!28 = !DILocation(line: 20, column: 2, scope: !21)
!29 = !DILocation(line: 21, column: 2, scope: !21)

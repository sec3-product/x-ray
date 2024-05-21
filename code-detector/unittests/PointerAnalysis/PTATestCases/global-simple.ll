; ModuleID = 'basic_c_tests/global-simple.c'
source_filename = "basic_c_tests/global-simple.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@a_int = dso_local global i32 10, align 4, !dbg !0
@p_int = dso_local global i32* @a_int, align 8, !dbg !8
@pp_int = dso_local global i32** @p_int, align 8, !dbg !12

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !19 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32*, align 8
  %4 = alloca i32**, align 8
  store i32 0, i32* %1, align 4
  call void @llvm.dbg.declare(metadata i32* %2, metadata !22, metadata !DIExpression()), !dbg !23
  %5 = load i32, i32* @a_int, align 4, !dbg !24
  store i32 %5, i32* %2, align 4, !dbg !23
  call void @llvm.dbg.declare(metadata i32** %3, metadata !25, metadata !DIExpression()), !dbg !26
  %6 = load i32*, i32** @p_int, align 8, !dbg !27
  store i32* %6, i32** %3, align 8, !dbg !26
  call void @llvm.dbg.declare(metadata i32*** %4, metadata !28, metadata !DIExpression()), !dbg !29
  %7 = load i32**, i32*** @pp_int, align 8, !dbg !30
  store i32** %7, i32*** %4, align 8, !dbg !29
  %8 = load i32**, i32*** %4, align 8, !dbg !31
  %9 = load i32*, i32** %8, align 8, !dbg !31
  %10 = bitcast i32* %9 to i8*, !dbg !31
  %11 = load i32*, i32** %3, align 8, !dbg !31
  %12 = bitcast i32* %11 to i8*, !dbg !31
  call void @__aser_alias__(i8* %10, i8* %12), !dbg !31
  %13 = load i32*, i32** %3, align 8, !dbg !32
  %14 = bitcast i32* %13 to i8*, !dbg !32
  call void @__aser_alias__(i8* %14, i8* bitcast (i32* @a_int to i8*)), !dbg !32
  ret i32 0, !dbg !33
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare dso_local void @__aser_alias__(i8*, i8*) #2

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!15, !16, !17}
!llvm.ident = !{!18}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "a_int", scope: !2, file: !3, line: 8, type: !11, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 9.0.0 (tags/RELEASE_900/final)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !5, globals: !7, nameTableKind: None)
!3 = !DIFile(filename: "basic_c_tests/global-simple.c", directory: "/home/peiming/Documents/Projects/LLVMRace/TestCases/PTABen")
!4 = !{}
!5 = !{!6}
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!7 = !{!0, !8, !12}
!8 = !DIGlobalVariableExpression(var: !9, expr: !DIExpression())
!9 = distinct !DIGlobalVariable(name: "p_int", scope: !2, file: !3, line: 9, type: !10, isLocal: false, isDefinition: true)
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 64)
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !DIGlobalVariableExpression(var: !13, expr: !DIExpression())
!13 = distinct !DIGlobalVariable(name: "pp_int", scope: !2, file: !3, line: 10, type: !14, isLocal: false, isDefinition: true)
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64)
!15 = !{i32 2, !"Dwarf Version", i32 4}
!16 = !{i32 2, !"Debug Info Version", i32 3}
!17 = !{i32 1, !"wchar_size", i32 4}
!18 = !{!"clang version 9.0.0 (tags/RELEASE_900/final)"}
!19 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 12, type: !20, scopeLine: 12, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!20 = !DISubroutineType(types: !21)
!21 = !{!11}
!22 = !DILocalVariable(name: "b_int", scope: !19, file: !3, line: 13, type: !11)
!23 = !DILocation(line: 13, column: 6, scope: !19)
!24 = !DILocation(line: 13, column: 14, scope: !19)
!25 = !DILocalVariable(name: "q_int", scope: !19, file: !3, line: 14, type: !10)
!26 = !DILocation(line: 14, column: 7, scope: !19)
!27 = !DILocation(line: 14, column: 15, scope: !19)
!28 = !DILocalVariable(name: "qq_int", scope: !19, file: !3, line: 15, type: !14)
!29 = !DILocation(line: 15, column: 8, scope: !19)
!30 = !DILocation(line: 15, column: 17, scope: !19)
!31 = !DILocation(line: 16, column: 2, scope: !19)
!32 = !DILocation(line: 17, column: 2, scope: !19)
!33 = !DILocation(line: 18, column: 2, scope: !19)

; ModuleID = 'basic_c_tests/branch-intra.c'
source_filename = "basic_c_tests/branch-intra.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !9 {
  %1 = alloca i32, align 4
  %2 = alloca i32*, align 8
  %3 = alloca i32*, align 8
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  call void @llvm.dbg.declare(metadata i32** %2, metadata !13, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32** %3, metadata !16, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.declare(metadata i32* %4, metadata !18, metadata !DIExpression()), !dbg !19
  call void @llvm.dbg.declare(metadata i32* %5, metadata !20, metadata !DIExpression()), !dbg !21
  call void @llvm.dbg.declare(metadata i32* %6, metadata !22, metadata !DIExpression()), !dbg !23
  %7 = load i32, i32* %6, align 4, !dbg !24
  %8 = icmp ne i32 %7, 0, !dbg !24
  br i1 %8, label %9, label %10, !dbg !26

9:                                                ; preds = %0
  store i32* %4, i32** %2, align 8, !dbg !27
  store i32* %5, i32** %3, align 8, !dbg !29
  br label %11, !dbg !30

10:                                               ; preds = %0
  store i32* %5, i32** %2, align 8, !dbg !31
  store i32* %6, i32** %3, align 8, !dbg !33
  br label %11

11:                                               ; preds = %10, %9
  %12 = load i32*, i32** %2, align 8, !dbg !34
  %13 = bitcast i32* %12 to i8*, !dbg !34
  %14 = load i32*, i32** %3, align 8, !dbg !34
  %15 = bitcast i32* %14 to i8*, !dbg !34
  call void @__aser_alias__(i8* %13, i8* %15), !dbg !34
  ret i32 0, !dbg !35
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare dso_local void @__aser_alias__(i8*, i8*) #2

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!5, !6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 9.0.0 (tags/RELEASE_900/final)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, nameTableKind: None)
!1 = !DIFile(filename: "basic_c_tests/branch-intra.c", directory: "/home/peiming/Documents/Projects/LLVMRace/TestCases/PTABen")
!2 = !{}
!3 = !{!4}
!4 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!5 = !{i32 2, !"Dwarf Version", i32 4}
!6 = !{i32 2, !"Debug Info Version", i32 3}
!7 = !{i32 1, !"wchar_size", i32 4}
!8 = !{!"clang version 9.0.0 (tags/RELEASE_900/final)"}
!9 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 8, type: !10, scopeLine: 9, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!10 = !DISubroutineType(types: !11)
!11 = !{!12}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DILocalVariable(name: "p", scope: !9, file: !1, line: 10, type: !14)
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!15 = !DILocation(line: 10, column: 7, scope: !9)
!16 = !DILocalVariable(name: "q", scope: !9, file: !1, line: 10, type: !14)
!17 = !DILocation(line: 10, column: 11, scope: !9)
!18 = !DILocalVariable(name: "a", scope: !9, file: !1, line: 11, type: !12)
!19 = !DILocation(line: 11, column: 6, scope: !9)
!20 = !DILocalVariable(name: "b", scope: !9, file: !1, line: 11, type: !12)
!21 = !DILocation(line: 11, column: 9, scope: !9)
!22 = !DILocalVariable(name: "c", scope: !9, file: !1, line: 11, type: !12)
!23 = !DILocation(line: 11, column: 12, scope: !9)
!24 = !DILocation(line: 12, column: 6, scope: !25)
!25 = distinct !DILexicalBlock(scope: !9, file: !1, line: 12, column: 6)
!26 = !DILocation(line: 12, column: 6, scope: !9)
!27 = !DILocation(line: 13, column: 5, scope: !28)
!28 = distinct !DILexicalBlock(scope: !25, file: !1, line: 12, column: 9)
!29 = !DILocation(line: 14, column: 5, scope: !28)
!30 = !DILocation(line: 15, column: 2, scope: !28)
!31 = !DILocation(line: 17, column: 5, scope: !32)
!32 = distinct !DILexicalBlock(scope: !25, file: !1, line: 16, column: 7)
!33 = !DILocation(line: 18, column: 5, scope: !32)
!34 = !DILocation(line: 20, column: 2, scope: !9)
!35 = !DILocation(line: 21, column: 2, scope: !9)

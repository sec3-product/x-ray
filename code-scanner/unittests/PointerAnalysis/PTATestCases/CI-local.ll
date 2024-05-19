; ModuleID = 'basic_c_tests/CI-local.c'
source_filename = "basic_c_tests/CI-local.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @foo(i32*, i32*) #0 !dbg !9 {
  %3 = alloca i32*, align 8
  %4 = alloca i32*, align 8
  store i32* %0, i32** %3, align 8
  call void @llvm.dbg.declare(metadata i32** %3, metadata !14, metadata !DIExpression()), !dbg !15
  store i32* %1, i32** %4, align 8
  call void @llvm.dbg.declare(metadata i32** %4, metadata !16, metadata !DIExpression()), !dbg !17
  %5 = load i32*, i32** %3, align 8, !dbg !18
  %6 = bitcast i32* %5 to i8*, !dbg !18
  %7 = load i32*, i32** %4, align 8, !dbg !18
  %8 = bitcast i32* %7 to i8*, !dbg !18
  call void @__aser_alias__(i8* %6, i8* %8), !dbg !18
  ret void, !dbg !19
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare dso_local void @__aser_alias__(i8*, i8*) #2

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !20 {
  %1 = alloca i32, align 4
  %2 = alloca i32*, align 8
  %3 = alloca i32*, align 8
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  call void @llvm.dbg.declare(metadata i32** %2, metadata !23, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.declare(metadata i32** %3, metadata !25, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.declare(metadata i32* %4, metadata !27, metadata !DIExpression()), !dbg !28
  call void @llvm.dbg.declare(metadata i32* %5, metadata !29, metadata !DIExpression()), !dbg !30
  %6 = load i32, i32* %4, align 4, !dbg !31
  %7 = icmp ne i32 %6, 0, !dbg !31
  br i1 %7, label %8, label %11, !dbg !33

8:                                                ; preds = %0
  store i32* %4, i32** %2, align 8, !dbg !34
  store i32* %5, i32** %3, align 8, !dbg !36
  %9 = load i32*, i32** %2, align 8, !dbg !37
  %10 = load i32*, i32** %3, align 8, !dbg !38
  call void @foo(i32* %9, i32* %10), !dbg !39
  br label %14, !dbg !40

11:                                               ; preds = %0
  store i32* %5, i32** %2, align 8, !dbg !41
  store i32* %4, i32** %3, align 8, !dbg !43
  %12 = load i32*, i32** %2, align 8, !dbg !44
  %13 = load i32*, i32** %3, align 8, !dbg !45
  call void @foo(i32* %12, i32* %13), !dbg !46
  br label %14

14:                                               ; preds = %11, %8
  ret i32 0, !dbg !47
}

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!5, !6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 9.0.0 (tags/RELEASE_900/final)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, nameTableKind: None)
!1 = !DIFile(filename: "basic_c_tests/CI-local.c", directory: "/home/peiming/Documents/Projects/LLVMRace/TestCases/PTABen")
!2 = !{}
!3 = !{!4}
!4 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!5 = !{i32 2, !"Dwarf Version", i32 4}
!6 = !{i32 2, !"Debug Info Version", i32 3}
!7 = !{i32 1, !"wchar_size", i32 4}
!8 = !{!"clang version 9.0.0 (tags/RELEASE_900/final)"}
!9 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 8, type: !10, scopeLine: 9, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!10 = !DISubroutineType(types: !11)
!11 = !{null, !12, !12}
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 64)
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !DILocalVariable(name: "m", arg: 1, scope: !9, file: !1, line: 8, type: !12)
!15 = !DILocation(line: 8, column: 15, scope: !9)
!16 = !DILocalVariable(name: "n", arg: 2, scope: !9, file: !1, line: 8, type: !12)
!17 = !DILocation(line: 8, column: 23, scope: !9)
!18 = !DILocation(line: 10, column: 2, scope: !9)
!19 = !DILocation(line: 11, column: 1, scope: !9)
!20 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 13, type: !21, scopeLine: 14, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!21 = !DISubroutineType(types: !22)
!22 = !{!13}
!23 = !DILocalVariable(name: "p", scope: !20, file: !1, line: 15, type: !12)
!24 = !DILocation(line: 15, column: 7, scope: !20)
!25 = !DILocalVariable(name: "q", scope: !20, file: !1, line: 15, type: !12)
!26 = !DILocation(line: 15, column: 11, scope: !20)
!27 = !DILocalVariable(name: "a", scope: !20, file: !1, line: 16, type: !13)
!28 = !DILocation(line: 16, column: 6, scope: !20)
!29 = !DILocalVariable(name: "b", scope: !20, file: !1, line: 16, type: !13)
!30 = !DILocation(line: 16, column: 8, scope: !20)
!31 = !DILocation(line: 17, column: 6, scope: !32)
!32 = distinct !DILexicalBlock(scope: !20, file: !1, line: 17, column: 6)
!33 = !DILocation(line: 17, column: 6, scope: !20)
!34 = !DILocation(line: 18, column: 5, scope: !35)
!35 = distinct !DILexicalBlock(scope: !32, file: !1, line: 17, column: 9)
!36 = !DILocation(line: 19, column: 5, scope: !35)
!37 = !DILocation(line: 20, column: 7, scope: !35)
!38 = !DILocation(line: 20, column: 9, scope: !35)
!39 = !DILocation(line: 20, column: 3, scope: !35)
!40 = !DILocation(line: 21, column: 2, scope: !35)
!41 = !DILocation(line: 23, column: 5, scope: !42)
!42 = distinct !DILexicalBlock(scope: !32, file: !1, line: 22, column: 7)
!43 = !DILocation(line: 24, column: 5, scope: !42)
!44 = !DILocation(line: 25, column: 7, scope: !42)
!45 = !DILocation(line: 25, column: 9, scope: !42)
!46 = !DILocation(line: 25, column: 3, scope: !42)
!47 = !DILocation(line: 27, column: 2, scope: !20)

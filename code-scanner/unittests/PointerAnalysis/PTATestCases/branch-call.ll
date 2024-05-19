; ModuleID = 'basic_c_tests/branch-call.c'
source_filename = "basic_c_tests/branch-call.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @foo(i32*, i32*) #0 !dbg !9 {
  %3 = alloca i32*, align 8
  %4 = alloca i32*, align 8
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  store i32* %0, i32** %3, align 8
  call void @llvm.dbg.declare(metadata i32** %3, metadata !14, metadata !DIExpression()), !dbg !15
  store i32* %1, i32** %4, align 8
  call void @llvm.dbg.declare(metadata i32** %4, metadata !16, metadata !DIExpression()), !dbg !17
  %7 = load i32*, i32** %3, align 8, !dbg !18
  %8 = bitcast i32* %7 to i8*, !dbg !18
  %9 = load i32*, i32** %4, align 8, !dbg !18
  %10 = bitcast i32* %9 to i8*, !dbg !18
  call void @__aser_alias__(i8* %8, i8* %10), !dbg !18
  call void @llvm.dbg.declare(metadata i32* %5, metadata !19, metadata !DIExpression()), !dbg !20
  call void @llvm.dbg.declare(metadata i32* %6, metadata !21, metadata !DIExpression()), !dbg !22
  %11 = load i32*, i32** %4, align 8, !dbg !23
  %12 = load i32, i32* %11, align 4, !dbg !24
  store i32 %12, i32* %5, align 4, !dbg !25
  %13 = load i32*, i32** %3, align 8, !dbg !26
  %14 = load i32, i32* %13, align 4, !dbg !27
  store i32 %14, i32* %6, align 4, !dbg !28
  %15 = load i32, i32* %5, align 4, !dbg !29
  %16 = load i32*, i32** %3, align 8, !dbg !30
  store i32 %15, i32* %16, align 4, !dbg !31
  %17 = load i32, i32* %6, align 4, !dbg !32
  %18 = load i32*, i32** %4, align 8, !dbg !33
  store i32 %17, i32* %18, align 4, !dbg !34
  ret void, !dbg !35
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare dso_local void @__aser_alias__(i8*, i8*) #2

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !36 {
  %1 = alloca i32, align 4
  %2 = alloca i32*, align 8
  %3 = alloca i32*, align 8
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  call void @llvm.dbg.declare(metadata i32** %2, metadata !39, metadata !DIExpression()), !dbg !40
  call void @llvm.dbg.declare(metadata i32** %3, metadata !41, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.declare(metadata i32* %4, metadata !43, metadata !DIExpression()), !dbg !44
  call void @llvm.dbg.declare(metadata i32* %5, metadata !45, metadata !DIExpression()), !dbg !46
  call void @llvm.dbg.declare(metadata i32* %6, metadata !47, metadata !DIExpression()), !dbg !48
  %7 = load i32, i32* %6, align 4, !dbg !49
  %8 = icmp ne i32 %7, 0, !dbg !49
  br i1 %8, label %9, label %12, !dbg !51

9:                                                ; preds = %0
  store i32* %4, i32** %2, align 8, !dbg !52
  store i32* %5, i32** %3, align 8, !dbg !54
  %10 = load i32*, i32** %2, align 8, !dbg !55
  %11 = load i32*, i32** %3, align 8, !dbg !56
  call void @foo(i32* %10, i32* %11), !dbg !57
  br label %15, !dbg !58

12:                                               ; preds = %0
  store i32* %5, i32** %2, align 8, !dbg !59
  store i32* %6, i32** %3, align 8, !dbg !61
  %13 = load i32*, i32** %2, align 8, !dbg !62
  %14 = load i32*, i32** %3, align 8, !dbg !63
  call void @foo(i32* %13, i32* %14), !dbg !64
  br label %15

15:                                               ; preds = %12, %9
  ret i32 0, !dbg !65
}

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!5, !6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 9.0.0 (tags/RELEASE_900/final)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, nameTableKind: None)
!1 = !DIFile(filename: "basic_c_tests/branch-call.c", directory: "/home/peiming/Documents/Projects/LLVMRace/TestCases/PTABen")
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
!19 = !DILocalVariable(name: "x", scope: !9, file: !1, line: 11, type: !13)
!20 = !DILocation(line: 11, column: 6, scope: !9)
!21 = !DILocalVariable(name: "y", scope: !9, file: !1, line: 11, type: !13)
!22 = !DILocation(line: 11, column: 9, scope: !9)
!23 = !DILocation(line: 12, column: 7, scope: !9)
!24 = !DILocation(line: 12, column: 6, scope: !9)
!25 = !DILocation(line: 12, column: 4, scope: !9)
!26 = !DILocation(line: 13, column: 7, scope: !9)
!27 = !DILocation(line: 13, column: 6, scope: !9)
!28 = !DILocation(line: 13, column: 4, scope: !9)
!29 = !DILocation(line: 14, column: 7, scope: !9)
!30 = !DILocation(line: 14, column: 3, scope: !9)
!31 = !DILocation(line: 14, column: 5, scope: !9)
!32 = !DILocation(line: 15, column: 7, scope: !9)
!33 = !DILocation(line: 15, column: 3, scope: !9)
!34 = !DILocation(line: 15, column: 5, scope: !9)
!35 = !DILocation(line: 16, column: 1, scope: !9)
!36 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 18, type: !37, scopeLine: 19, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!37 = !DISubroutineType(types: !38)
!38 = !{!13}
!39 = !DILocalVariable(name: "p", scope: !36, file: !1, line: 20, type: !12)
!40 = !DILocation(line: 20, column: 7, scope: !36)
!41 = !DILocalVariable(name: "q", scope: !36, file: !1, line: 20, type: !12)
!42 = !DILocation(line: 20, column: 11, scope: !36)
!43 = !DILocalVariable(name: "a", scope: !36, file: !1, line: 21, type: !13)
!44 = !DILocation(line: 21, column: 6, scope: !36)
!45 = !DILocalVariable(name: "b", scope: !36, file: !1, line: 21, type: !13)
!46 = !DILocation(line: 21, column: 9, scope: !36)
!47 = !DILocalVariable(name: "c", scope: !36, file: !1, line: 21, type: !13)
!48 = !DILocation(line: 21, column: 12, scope: !36)
!49 = !DILocation(line: 22, column: 6, scope: !50)
!50 = distinct !DILexicalBlock(scope: !36, file: !1, line: 22, column: 6)
!51 = !DILocation(line: 22, column: 6, scope: !36)
!52 = !DILocation(line: 23, column: 5, scope: !53)
!53 = distinct !DILexicalBlock(scope: !50, file: !1, line: 22, column: 9)
!54 = !DILocation(line: 24, column: 5, scope: !53)
!55 = !DILocation(line: 25, column: 7, scope: !53)
!56 = !DILocation(line: 25, column: 9, scope: !53)
!57 = !DILocation(line: 25, column: 3, scope: !53)
!58 = !DILocation(line: 26, column: 2, scope: !53)
!59 = !DILocation(line: 28, column: 5, scope: !60)
!60 = distinct !DILexicalBlock(scope: !50, file: !1, line: 27, column: 7)
!61 = !DILocation(line: 29, column: 5, scope: !60)
!62 = !DILocation(line: 30, column: 7, scope: !60)
!63 = !DILocation(line: 30, column: 9, scope: !60)
!64 = !DILocation(line: 30, column: 3, scope: !60)
!65 = !DILocation(line: 32, column: 2, scope: !36)

; ModuleID = 'basic_c_tests/funptr-simple.c'
source_filename = "basic_c_tests/funptr-simple.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@fptr = common dso_local global void (i32*, i32*)* null, align 8, !dbg !0

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @f(i32*, i32*) #0 !dbg !17 {
  %3 = alloca i32*, align 8
  %4 = alloca i32*, align 8
  store i32* %0, i32** %3, align 8
  call void @llvm.dbg.declare(metadata i32** %3, metadata !18, metadata !DIExpression()), !dbg !19
  store i32* %1, i32** %4, align 8
  call void @llvm.dbg.declare(metadata i32** %4, metadata !20, metadata !DIExpression()), !dbg !21
  %5 = load i32*, i32** %3, align 8, !dbg !22
  %6 = bitcast i32* %5 to i8*, !dbg !22
  %7 = load i32*, i32** %4, align 8, !dbg !22
  %8 = bitcast i32* %7 to i8*, !dbg !22
  call void @__aser_alias__(i8* %6, i8* %8), !dbg !22
  ret void, !dbg !23
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare dso_local void @__aser_alias__(i8*, i8*) #2

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !24 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32*, align 8
  %5 = alloca i32*, align 8
  store i32 0, i32* %1, align 4
  call void @llvm.dbg.declare(metadata i32* %2, metadata !27, metadata !DIExpression()), !dbg !28
  call void @llvm.dbg.declare(metadata i32* %3, metadata !29, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.declare(metadata i32** %4, metadata !31, metadata !DIExpression()), !dbg !32
  call void @llvm.dbg.declare(metadata i32** %5, metadata !33, metadata !DIExpression()), !dbg !34
  %6 = load i32, i32* %2, align 4, !dbg !35
  %7 = icmp ne i32 %6, 0, !dbg !35
  br i1 %7, label %8, label %12, !dbg !37

8:                                                ; preds = %0
  store i32* %2, i32** %4, align 8, !dbg !38
  store i32* %2, i32** %5, align 8, !dbg !40
  store void (i32*, i32*)* @f, void (i32*, i32*)** @fptr, align 8, !dbg !41
  %9 = load void (i32*, i32*)*, void (i32*, i32*)** @fptr, align 8, !dbg !42
  %10 = load i32*, i32** %4, align 8, !dbg !43
  %11 = load i32*, i32** %5, align 8, !dbg !44
  call void %9(i32* %10, i32* %11), !dbg !42
  br label %15, !dbg !45

12:                                               ; preds = %0
  store i32* %2, i32** %4, align 8, !dbg !46
  store i32* %3, i32** %5, align 8, !dbg !48
  %13 = load i32*, i32** %4, align 8, !dbg !49
  %14 = load i32*, i32** %5, align 8, !dbg !50
  call void @f(i32* %13, i32* %14), !dbg !51
  br label %15

15:                                               ; preds = %12, %8
  ret i32 0, !dbg !52
}

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!13, !14, !15}
!llvm.ident = !{!16}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "fptr", scope: !2, file: !3, line: 14, type: !8, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 9.0.0 (tags/RELEASE_900/final)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !5, globals: !7, nameTableKind: None)
!3 = !DIFile(filename: "basic_c_tests/funptr-simple.c", directory: "/home/peiming/Documents/Projects/LLVMRace/TestCases/PTABen")
!4 = !{}
!5 = !{!6}
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!7 = !{!0}
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 64)
!9 = !DISubroutineType(types: !10)
!10 = !{null, !11, !11}
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !{i32 2, !"Dwarf Version", i32 4}
!14 = !{i32 2, !"Debug Info Version", i32 3}
!15 = !{i32 1, !"wchar_size", i32 4}
!16 = !{!"clang version 9.0.0 (tags/RELEASE_900/final)"}
!17 = distinct !DISubprogram(name: "f", scope: !3, file: !3, line: 8, type: !9, scopeLine: 8, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!18 = !DILocalVariable(name: "p", arg: 1, scope: !17, file: !3, line: 8, type: !11)
!19 = !DILocation(line: 8, column: 13, scope: !17)
!20 = !DILocalVariable(name: "q", arg: 2, scope: !17, file: !3, line: 8, type: !11)
!21 = !DILocation(line: 8, column: 21, scope: !17)
!22 = !DILocation(line: 11, column: 2, scope: !17)
!23 = !DILocation(line: 12, column: 1, scope: !17)
!24 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 16, type: !25, scopeLine: 16, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!25 = !DISubroutineType(types: !26)
!26 = !{!12}
!27 = !DILocalVariable(name: "x", scope: !24, file: !3, line: 17, type: !12)
!28 = !DILocation(line: 17, column: 6, scope: !24)
!29 = !DILocalVariable(name: "y", scope: !24, file: !3, line: 17, type: !12)
!30 = !DILocation(line: 17, column: 9, scope: !24)
!31 = !DILocalVariable(name: "m", scope: !24, file: !3, line: 18, type: !11)
!32 = !DILocation(line: 18, column: 7, scope: !24)
!33 = !DILocalVariable(name: "n", scope: !24, file: !3, line: 18, type: !11)
!34 = !DILocation(line: 18, column: 11, scope: !24)
!35 = !DILocation(line: 19, column: 6, scope: !36)
!36 = distinct !DILexicalBlock(scope: !24, file: !3, line: 19, column: 6)
!37 = !DILocation(line: 19, column: 6, scope: !24)
!38 = !DILocation(line: 20, column: 5, scope: !39)
!39 = distinct !DILexicalBlock(scope: !36, file: !3, line: 19, column: 9)
!40 = !DILocation(line: 20, column: 13, scope: !39)
!41 = !DILocation(line: 21, column: 8, scope: !39)
!42 = !DILocation(line: 22, column: 3, scope: !39)
!43 = !DILocation(line: 22, column: 8, scope: !39)
!44 = !DILocation(line: 22, column: 10, scope: !39)
!45 = !DILocation(line: 23, column: 2, scope: !39)
!46 = !DILocation(line: 25, column: 5, scope: !47)
!47 = distinct !DILexicalBlock(scope: !36, file: !3, line: 24, column: 7)
!48 = !DILocation(line: 25, column: 13, scope: !47)
!49 = !DILocation(line: 26, column: 5, scope: !47)
!50 = !DILocation(line: 26, column: 7, scope: !47)
!51 = !DILocation(line: 26, column: 3, scope: !47)
!52 = !DILocation(line: 28, column: 2, scope: !24)

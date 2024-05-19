; ModuleID = 'basic_c_tests/heap-indirect.c'
source_filename = "basic_c_tests/heap-indirect.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @malloc_two(i32**, i32**) #0 !dbg !11 {
  %3 = alloca i32**, align 8
  %4 = alloca i32**, align 8
  store i32** %0, i32*** %3, align 8
  call void @llvm.dbg.declare(metadata i32*** %3, metadata !15, metadata !DIExpression()), !dbg !16
  store i32** %1, i32*** %4, align 8
  call void @llvm.dbg.declare(metadata i32*** %4, metadata !17, metadata !DIExpression()), !dbg !18
  %5 = call i8* @malloc(i64 4), !dbg !19
  %6 = bitcast i8* %5 to i32*, !dbg !20
  %7 = load i32**, i32*** %3, align 8, !dbg !21
  store i32* %6, i32** %7, align 8, !dbg !22
  %8 = call i8* @malloc(i64 4), !dbg !23
  %9 = bitcast i8* %8 to i32*, !dbg !24
  %10 = load i32**, i32*** %4, align 8, !dbg !25
  store i32* %9, i32** %10, align 8, !dbg !26
  ret void, !dbg !27
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare dso_local i8* @malloc(i64) #2

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !28 {
  %1 = alloca i32, align 4
  %2 = alloca i32**, align 8
  %3 = alloca i32**, align 8
  store i32 0, i32* %1, align 4
  call void @llvm.dbg.declare(metadata i32*** %2, metadata !31, metadata !DIExpression()), !dbg !32
  %4 = call i8* @malloc(i64 100), !dbg !33
  %5 = bitcast i8* %4 to i32**, !dbg !33
  store i32** %5, i32*** %2, align 8, !dbg !32
  call void @llvm.dbg.declare(metadata i32*** %3, metadata !34, metadata !DIExpression()), !dbg !35
  %6 = call i8* @malloc(i64 100), !dbg !36
  %7 = bitcast i8* %6 to i32**, !dbg !36
  store i32** %7, i32*** %3, align 8, !dbg !35
  %8 = load i32**, i32*** %2, align 8, !dbg !37
  %9 = load i32**, i32*** %3, align 8, !dbg !38
  call void @malloc_two(i32** %8, i32** %9), !dbg !39
  %10 = load i32**, i32*** %2, align 8, !dbg !40
  %11 = load i32*, i32** %10, align 8, !dbg !40
  %12 = bitcast i32* %11 to i8*, !dbg !40
  %13 = load i32**, i32*** %3, align 8, !dbg !40
  %14 = load i32*, i32** %13, align 8, !dbg !40
  %15 = bitcast i32* %14 to i8*, !dbg !40
  call void @__aser_no_alias__(i8* %12, i8* %15), !dbg !40
  ret i32 0, !dbg !41
}

declare dso_local void @__aser_no_alias__(i8*, i8*) #2

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8, !9}
!llvm.ident = !{!10}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 9.0.0 (tags/RELEASE_900/final)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, nameTableKind: None)
!1 = !DIFile(filename: "basic_c_tests/heap-indirect.c", directory: "/home/peiming/Documents/Projects/LLVMRace/TestCases/PTABen")
!2 = !{}
!3 = !{!4, !6}
!4 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !5, size: 64)
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = !{!"clang version 9.0.0 (tags/RELEASE_900/final)"}
!11 = distinct !DISubprogram(name: "malloc_two", scope: !1, file: !1, line: 11, type: !12, scopeLine: 11, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!12 = !DISubroutineType(types: !13)
!13 = !{null, !14, !14}
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !4, size: 64)
!15 = !DILocalVariable(name: "p", arg: 1, scope: !11, file: !1, line: 11, type: !14)
!16 = !DILocation(line: 11, column: 23, scope: !11)
!17 = !DILocalVariable(name: "q", arg: 2, scope: !11, file: !1, line: 11, type: !14)
!18 = !DILocation(line: 11, column: 32, scope: !11)
!19 = !DILocation(line: 12, column: 14, scope: !11)
!20 = !DILocation(line: 12, column: 7, scope: !11)
!21 = !DILocation(line: 12, column: 3, scope: !11)
!22 = !DILocation(line: 12, column: 5, scope: !11)
!23 = !DILocation(line: 13, column: 14, scope: !11)
!24 = !DILocation(line: 13, column: 7, scope: !11)
!25 = !DILocation(line: 13, column: 3, scope: !11)
!26 = !DILocation(line: 13, column: 5, scope: !11)
!27 = !DILocation(line: 14, column: 1, scope: !11)
!28 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 16, type: !29, scopeLine: 16, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!29 = !DISubroutineType(types: !30)
!30 = !{!5}
!31 = !DILocalVariable(name: "o1", scope: !28, file: !1, line: 17, type: !14)
!32 = !DILocation(line: 17, column: 8, scope: !28)
!33 = !DILocation(line: 17, column: 13, scope: !28)
!34 = !DILocalVariable(name: "o2", scope: !28, file: !1, line: 18, type: !14)
!35 = !DILocation(line: 18, column: 11, scope: !28)
!36 = !DILocation(line: 18, column: 16, scope: !28)
!37 = !DILocation(line: 19, column: 13, scope: !28)
!38 = !DILocation(line: 19, column: 17, scope: !28)
!39 = !DILocation(line: 19, column: 2, scope: !28)
!40 = !DILocation(line: 20, column: 2, scope: !28)
!41 = !DILocation(line: 21, column: 2, scope: !28)

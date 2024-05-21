; ModuleID = 'basic_c_tests/ptr-dereference2.c'
source_filename = "basic_c_tests/ptr-dereference2.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !9 {
  %1 = alloca i32, align 4
  %2 = alloca i32**, align 8
  %3 = alloca i32*, align 8
  %4 = alloca i32*, align 8
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  call void @llvm.dbg.declare(metadata i32*** %2, metadata !13, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i32** %3, metadata !17, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i32** %4, metadata !19, metadata !DIExpression()), !dbg !20
  call void @llvm.dbg.declare(metadata i32* %5, metadata !21, metadata !DIExpression()), !dbg !22
  store i32 10, i32* %5, align 4, !dbg !23
  store i32** %3, i32*** %2, align 8, !dbg !24
  store i32* %5, i32** %3, align 8, !dbg !25
  %7 = load i32**, i32*** %2, align 8, !dbg !26
  %8 = load i32*, i32** %7, align 8, !dbg !27
  store i32* %8, i32** %4, align 8, !dbg !28
  call void @llvm.dbg.declare(metadata i32* %6, metadata !29, metadata !DIExpression()), !dbg !30
  %9 = load i32*, i32** %4, align 8, !dbg !31
  %10 = load i32, i32* %9, align 4, !dbg !32
  store i32 %10, i32* %6, align 4, !dbg !30
  %11 = load i32*, i32** %4, align 8, !dbg !33
  %12 = bitcast i32* %11 to i8*, !dbg !33
  %13 = bitcast i32* %5 to i8*, !dbg !33
  call void @__aser_alias__(i8* %12, i8* %13), !dbg !33
  %14 = load i32*, i32** %4, align 8, !dbg !34
  %15 = bitcast i32* %14 to i8*, !dbg !34
  %16 = load i32*, i32** %3, align 8, !dbg !34
  %17 = bitcast i32* %16 to i8*, !dbg !34
  call void @__aser_alias__(i8* %15, i8* %17), !dbg !34
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
!1 = !DIFile(filename: "basic_c_tests/ptr-dereference2.c", directory: "/home/peiming/Documents/Projects/LLVMRace/TestCases/PTABen")
!2 = !{}
!3 = !{!4}
!4 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!5 = !{i32 2, !"Dwarf Version", i32 4}
!6 = !{i32 2, !"Debug Info Version", i32 3}
!7 = !{i32 1, !"wchar_size", i32 4}
!8 = !{!"clang version 9.0.0 (tags/RELEASE_900/final)"}
!9 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 3, type: !10, scopeLine: 3, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!10 = !DISubroutineType(types: !11)
!11 = !{!12}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DILocalVariable(name: "a", scope: !9, file: !1, line: 5, type: !14)
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !15, size: 64)
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!16 = !DILocation(line: 5, column: 9, scope: !9)
!17 = !DILocalVariable(name: "b", scope: !9, file: !1, line: 5, type: !15)
!18 = !DILocation(line: 5, column: 13, scope: !9)
!19 = !DILocalVariable(name: "x", scope: !9, file: !1, line: 5, type: !15)
!20 = !DILocation(line: 5, column: 17, scope: !9)
!21 = !DILocalVariable(name: "c", scope: !9, file: !1, line: 5, type: !12)
!22 = !DILocation(line: 5, column: 20, scope: !9)
!23 = !DILocation(line: 6, column: 5, scope: !9)
!24 = !DILocation(line: 7, column: 5, scope: !9)
!25 = !DILocation(line: 8, column: 5, scope: !9)
!26 = !DILocation(line: 9, column: 8, scope: !9)
!27 = !DILocation(line: 9, column: 7, scope: !9)
!28 = !DILocation(line: 9, column: 5, scope: !9)
!29 = !DILocalVariable(name: "y", scope: !9, file: !1, line: 10, type: !12)
!30 = !DILocation(line: 10, column: 7, scope: !9)
!31 = !DILocation(line: 10, column: 12, scope: !9)
!32 = !DILocation(line: 10, column: 11, scope: !9)
!33 = !DILocation(line: 11, column: 3, scope: !9)
!34 = !DILocation(line: 12, column: 3, scope: !9)
!35 = !DILocation(line: 13, column: 3, scope: !9)

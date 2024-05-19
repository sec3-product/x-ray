; ModuleID = 'basic_c_tests/ptr-dereference3.c'
source_filename = "basic_c_tests/ptr-dereference3.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @foo(i32) #0 !dbg !9 {
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 %0, i32* %2, align 4
  call void @llvm.dbg.declare(metadata i32* %2, metadata !13, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32* %3, metadata !15, metadata !DIExpression()), !dbg !16
  store i32 10, i32* %3, align 4, !dbg !16
  call void @llvm.dbg.declare(metadata i32* %4, metadata !17, metadata !DIExpression()), !dbg !18
  %5 = load i32, i32* %3, align 4, !dbg !19
  store i32 %5, i32* %4, align 4, !dbg !18
  ret void, !dbg !20
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !21 {
  %1 = alloca i32, align 4
  %2 = alloca i32*, align 8
  %3 = alloca i32*, align 8
  %4 = alloca i32***, align 8
  %5 = alloca i32**, align 8
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  call void @llvm.dbg.declare(metadata i32** %2, metadata !24, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.declare(metadata i32** %3, metadata !27, metadata !DIExpression()), !dbg !28
  call void @llvm.dbg.declare(metadata i32**** %4, metadata !29, metadata !DIExpression()), !dbg !32
  call void @llvm.dbg.declare(metadata i32*** %5, metadata !33, metadata !DIExpression()), !dbg !34
  call void @llvm.dbg.declare(metadata i32* %6, metadata !35, metadata !DIExpression()), !dbg !36
  call void @llvm.dbg.declare(metadata i32* %7, metadata !37, metadata !DIExpression()), !dbg !38
  call void @llvm.dbg.declare(metadata i32* %8, metadata !39, metadata !DIExpression()), !dbg !40
  store i32* %6, i32** %2, align 8, !dbg !41
  store i32* %7, i32** %3, align 8, !dbg !42
  store i32** %3, i32*** %5, align 8, !dbg !43
  %9 = load i32*, i32** %3, align 8, !dbg !44
  store i32* %9, i32** %2, align 8, !dbg !45
  %10 = load i32*, i32** %2, align 8, !dbg !46
  %11 = bitcast i32* %10 to i8*, !dbg !46
  %12 = bitcast i32* %7 to i8*, !dbg !46
  call void @__aser_alias__(i8* %11, i8* %12), !dbg !46
  %13 = load i32**, i32*** %5, align 8, !dbg !47
  %14 = load i32*, i32** %13, align 8, !dbg !48
  %15 = bitcast i32* %14 to i32***, !dbg !48
  store i32*** %15, i32**** %4, align 8, !dbg !49
  %16 = load i32***, i32**** %4, align 8, !dbg !50
  %17 = bitcast i32*** %16 to i8*, !dbg !50
  %18 = load i32*, i32** %3, align 8, !dbg !50
  %19 = bitcast i32* %18 to i8*, !dbg !50
  call void @__aser_alias__(i8* %17, i8* %19), !dbg !50
  %20 = load i32, i32* %8, align 4, !dbg !51
  call void @foo(i32 %20), !dbg !52
  ret i32 0, !dbg !53
}

declare dso_local void @__aser_alias__(i8*, i8*) #2

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!5, !6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 9.0.0 (tags/RELEASE_900/final)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, nameTableKind: None)
!1 = !DIFile(filename: "basic_c_tests/ptr-dereference3.c", directory: "/home/peiming/Documents/Projects/LLVMRace/TestCases/PTABen")
!2 = !{}
!3 = !{!4}
!4 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!5 = !{i32 2, !"Dwarf Version", i32 4}
!6 = !{i32 2, !"Debug Info Version", i32 3}
!7 = !{i32 1, !"wchar_size", i32 4}
!8 = !{!"clang version 9.0.0 (tags/RELEASE_900/final)"}
!9 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 2, type: !10, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!10 = !DISubroutineType(types: !11)
!11 = !{null, !12}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DILocalVariable(name: "q", arg: 1, scope: !9, file: !1, line: 2, type: !12)
!14 = !DILocation(line: 2, column: 14, scope: !9)
!15 = !DILocalVariable(name: "i", scope: !9, file: !1, line: 3, type: !12)
!16 = !DILocation(line: 3, column: 7, scope: !9)
!17 = !DILocalVariable(name: "k", scope: !9, file: !1, line: 4, type: !12)
!18 = !DILocation(line: 4, column: 7, scope: !9)
!19 = !DILocation(line: 4, column: 11, scope: !9)
!20 = !DILocation(line: 6, column: 1, scope: !9)
!21 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 7, type: !22, scopeLine: 7, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!22 = !DISubroutineType(types: !23)
!23 = !{!12}
!24 = !DILocalVariable(name: "s", scope: !21, file: !1, line: 9, type: !25)
!25 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!26 = !DILocation(line: 9, column: 6, scope: !21)
!27 = !DILocalVariable(name: "r", scope: !21, file: !1, line: 9, type: !25)
!28 = !DILocation(line: 9, column: 9, scope: !21)
!29 = !DILocalVariable(name: "x", scope: !21, file: !1, line: 9, type: !30)
!30 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !31, size: 64)
!31 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !25, size: 64)
!32 = !DILocation(line: 9, column: 14, scope: !21)
!33 = !DILocalVariable(name: "y", scope: !21, file: !1, line: 9, type: !31)
!34 = !DILocation(line: 9, column: 18, scope: !21)
!35 = !DILocalVariable(name: "t", scope: !21, file: !1, line: 9, type: !12)
!36 = !DILocation(line: 9, column: 20, scope: !21)
!37 = !DILocalVariable(name: "z", scope: !21, file: !1, line: 9, type: !12)
!38 = !DILocation(line: 9, column: 22, scope: !21)
!39 = !DILocalVariable(name: "k", scope: !21, file: !1, line: 9, type: !12)
!40 = !DILocation(line: 9, column: 24, scope: !21)
!41 = !DILocation(line: 10, column: 4, scope: !21)
!42 = !DILocation(line: 11, column: 4, scope: !21)
!43 = !DILocation(line: 12, column: 4, scope: !21)
!44 = !DILocation(line: 13, column: 6, scope: !21)
!45 = !DILocation(line: 13, column: 4, scope: !21)
!46 = !DILocation(line: 14, column: 3, scope: !21)
!47 = !DILocation(line: 15, column: 7, scope: !21)
!48 = !DILocation(line: 15, column: 6, scope: !21)
!49 = !DILocation(line: 15, column: 4, scope: !21)
!50 = !DILocation(line: 16, column: 3, scope: !21)
!51 = !DILocation(line: 17, column: 6, scope: !21)
!52 = !DILocation(line: 17, column: 2, scope: !21)
!53 = !DILocation(line: 18, column: 1, scope: !21)

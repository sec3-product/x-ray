; ModuleID = 'basic_c_tests/constraint-cycle-copy.c'
source_filename = "basic_c_tests/constraint-cycle-copy.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !9 {
  %1 = alloca i32, align 4
  %2 = alloca i32**, align 8
  %3 = alloca i32**, align 8
  %4 = alloca i32**, align 8
  %5 = alloca i32*, align 8
  %6 = alloca i32*, align 8
  %7 = alloca i32*, align 8
  %8 = alloca i32*, align 8
  %9 = alloca i32, align 4
  %10 = alloca i32, align 4
  %11 = alloca i32, align 4
  %12 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  call void @llvm.dbg.declare(metadata i32*** %2, metadata !13, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i32*** %3, metadata !17, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i32*** %4, metadata !19, metadata !DIExpression()), !dbg !20
  call void @llvm.dbg.declare(metadata i32** %5, metadata !21, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.declare(metadata i32** %6, metadata !23, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.declare(metadata i32** %7, metadata !25, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.declare(metadata i32** %8, metadata !27, metadata !DIExpression()), !dbg !28
  call void @llvm.dbg.declare(metadata i32* %9, metadata !29, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.declare(metadata i32* %10, metadata !31, metadata !DIExpression()), !dbg !32
  call void @llvm.dbg.declare(metadata i32* %11, metadata !33, metadata !DIExpression()), !dbg !34
  call void @llvm.dbg.declare(metadata i32* %12, metadata !35, metadata !DIExpression()), !dbg !36
  store i32* %9, i32** %5, align 8, !dbg !37
  store i32* %10, i32** %6, align 8, !dbg !38
  store i32* %11, i32** %7, align 8, !dbg !39
  store i32** %5, i32*** %2, align 8, !dbg !40
  store i32** %6, i32*** %3, align 8, !dbg !41
  store i32** %7, i32*** %4, align 8, !dbg !42
  %13 = load i32, i32* %12, align 4, !dbg !43
  %14 = icmp ne i32 %13, 0, !dbg !43
  br i1 %14, label %15, label %16, !dbg !45

15:                                               ; preds = %0
  store i32** %8, i32*** %3, align 8, !dbg !46
  store i32* %12, i32** %8, align 8, !dbg !48
  br label %16, !dbg !49

16:                                               ; preds = %15, %0
  %17 = load i32**, i32*** %3, align 8, !dbg !50
  %18 = load i32*, i32** %17, align 8, !dbg !51
  %19 = load i32**, i32*** %2, align 8, !dbg !52
  store i32* %18, i32** %19, align 8, !dbg !53
  %20 = load i32**, i32*** %4, align 8, !dbg !54
  %21 = load i32*, i32** %20, align 8, !dbg !55
  %22 = load i32**, i32*** %3, align 8, !dbg !56
  store i32* %21, i32** %22, align 8, !dbg !57
  %23 = load i32**, i32*** %2, align 8, !dbg !58
  %24 = load i32*, i32** %23, align 8, !dbg !59
  %25 = load i32**, i32*** %4, align 8, !dbg !60
  store i32* %24, i32** %25, align 8, !dbg !61
  %26 = load i32*, i32** %5, align 8, !dbg !62
  %27 = bitcast i32* %26 to i8*, !dbg !62
  %28 = load i32*, i32** %6, align 8, !dbg !62
  %29 = bitcast i32* %28 to i8*, !dbg !62
  call void @__aser_alias__(i8* %27, i8* %29), !dbg !62
  %30 = load i32*, i32** %7, align 8, !dbg !63
  %31 = bitcast i32* %30 to i8*, !dbg !63
  %32 = load i32*, i32** %5, align 8, !dbg !63
  %33 = bitcast i32* %32 to i8*, !dbg !63
  call void @__aser_alias__(i8* %31, i8* %33), !dbg !63
  ret i32 0, !dbg !64
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
!1 = !DIFile(filename: "basic_c_tests/constraint-cycle-copy.c", directory: "/home/peiming/Documents/Projects/LLVMRace/TestCases/PTABen")
!2 = !{}
!3 = !{!4}
!4 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!5 = !{i32 2, !"Dwarf Version", i32 4}
!6 = !{i32 2, !"Debug Info Version", i32 3}
!7 = !{i32 1, !"wchar_size", i32 4}
!8 = !{!"clang version 9.0.0 (tags/RELEASE_900/final)"}
!9 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 8, type: !10, scopeLine: 8, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!10 = !DISubroutineType(types: !11)
!11 = !{!12}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DILocalVariable(name: "x1", scope: !9, file: !1, line: 9, type: !14)
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !15, size: 64)
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!16 = !DILocation(line: 9, column: 8, scope: !9)
!17 = !DILocalVariable(name: "y1", scope: !9, file: !1, line: 9, type: !14)
!18 = !DILocation(line: 9, column: 14, scope: !9)
!19 = !DILocalVariable(name: "z1", scope: !9, file: !1, line: 9, type: !14)
!20 = !DILocation(line: 9, column: 20, scope: !9)
!21 = !DILocalVariable(name: "x2", scope: !9, file: !1, line: 10, type: !15)
!22 = !DILocation(line: 10, column: 7, scope: !9)
!23 = !DILocalVariable(name: "y2", scope: !9, file: !1, line: 10, type: !15)
!24 = !DILocation(line: 10, column: 12, scope: !9)
!25 = !DILocalVariable(name: "z2", scope: !9, file: !1, line: 10, type: !15)
!26 = !DILocation(line: 10, column: 17, scope: !9)
!27 = !DILocalVariable(name: "y2_", scope: !9, file: !1, line: 10, type: !15)
!28 = !DILocation(line: 10, column: 22, scope: !9)
!29 = !DILocalVariable(name: "x3", scope: !9, file: !1, line: 11, type: !12)
!30 = !DILocation(line: 11, column: 6, scope: !9)
!31 = !DILocalVariable(name: "y3", scope: !9, file: !1, line: 11, type: !12)
!32 = !DILocation(line: 11, column: 10, scope: !9)
!33 = !DILocalVariable(name: "z3", scope: !9, file: !1, line: 11, type: !12)
!34 = !DILocation(line: 11, column: 14, scope: !9)
!35 = !DILocalVariable(name: "y3_", scope: !9, file: !1, line: 11, type: !12)
!36 = !DILocation(line: 11, column: 18, scope: !9)
!37 = !DILocation(line: 12, column: 5, scope: !9)
!38 = !DILocation(line: 12, column: 15, scope: !9)
!39 = !DILocation(line: 12, column: 25, scope: !9)
!40 = !DILocation(line: 13, column: 5, scope: !9)
!41 = !DILocation(line: 13, column: 15, scope: !9)
!42 = !DILocation(line: 13, column: 25, scope: !9)
!43 = !DILocation(line: 17, column: 6, scope: !44)
!44 = distinct !DILexicalBlock(scope: !9, file: !1, line: 17, column: 6)
!45 = !DILocation(line: 17, column: 6, scope: !9)
!46 = !DILocation(line: 18, column: 6, scope: !47)
!47 = distinct !DILexicalBlock(scope: !44, file: !1, line: 17, column: 11)
!48 = !DILocation(line: 19, column: 7, scope: !47)
!49 = !DILocation(line: 20, column: 2, scope: !47)
!50 = !DILocation(line: 21, column: 9, scope: !9)
!51 = !DILocation(line: 21, column: 8, scope: !9)
!52 = !DILocation(line: 21, column: 3, scope: !9)
!53 = !DILocation(line: 21, column: 6, scope: !9)
!54 = !DILocation(line: 22, column: 9, scope: !9)
!55 = !DILocation(line: 22, column: 8, scope: !9)
!56 = !DILocation(line: 22, column: 3, scope: !9)
!57 = !DILocation(line: 22, column: 6, scope: !9)
!58 = !DILocation(line: 23, column: 9, scope: !9)
!59 = !DILocation(line: 23, column: 8, scope: !9)
!60 = !DILocation(line: 23, column: 3, scope: !9)
!61 = !DILocation(line: 23, column: 6, scope: !9)
!62 = !DILocation(line: 26, column: 2, scope: !9)
!63 = !DILocation(line: 27, column: 2, scope: !9)
!64 = !DILocation(line: 28, column: 2, scope: !9)

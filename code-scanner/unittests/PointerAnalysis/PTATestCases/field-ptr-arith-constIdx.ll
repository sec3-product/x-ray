; ModuleID = 'basic_c_tests/field-ptr-arith-constIdx.c'
source_filename = "basic_c_tests/field-ptr-arith-constIdx.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.s = type { i32*, i32* }

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !9 {
  %1 = alloca i32, align 4
  %2 = alloca i32**, align 8
  %3 = alloca i32**, align 8
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  %6 = alloca %struct.s, align 8
  store i32 0, i32* %1, align 4
  call void @llvm.dbg.declare(metadata i32*** %2, metadata !13, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i32*** %3, metadata !17, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i32* %4, metadata !19, metadata !DIExpression()), !dbg !20
  call void @llvm.dbg.declare(metadata i32* %5, metadata !21, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.declare(metadata %struct.s* %6, metadata !23, metadata !DIExpression()), !dbg !28
  %7 = getelementptr inbounds %struct.s, %struct.s* %6, i32 0, i32 0, !dbg !29
  store i32* %4, i32** %7, align 8, !dbg !30
  %8 = getelementptr inbounds %struct.s, %struct.s* %6, i32 0, i32 1, !dbg !31
  store i32* %5, i32** %8, align 8, !dbg !32
  %9 = getelementptr inbounds %struct.s, %struct.s* %6, i32 0, i32 0, !dbg !33
  store i32** %9, i32*** %2, align 8, !dbg !34
  %10 = load i32**, i32*** %2, align 8, !dbg !35
  %11 = getelementptr inbounds i32*, i32** %10, i64 1, !dbg !36
  store i32** %11, i32*** %3, align 8, !dbg !37
  %12 = load i32**, i32*** %3, align 8, !dbg !38
  %13 = load i32*, i32** %12, align 8, !dbg !38
  %14 = bitcast i32* %13 to i8*, !dbg !38
  %15 = bitcast i32* %5 to i8*, !dbg !38
  call void @__aser_alias__(i8* %14, i8* %15), !dbg !38
  ret i32 0, !dbg !39
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
!1 = !DIFile(filename: "basic_c_tests/field-ptr-arith-constIdx.c", directory: "/home/peiming/Documents/Projects/LLVMRace/TestCases/PTABen")
!2 = !{}
!3 = !{!4}
!4 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!5 = !{i32 2, !"Dwarf Version", i32 4}
!6 = !{i32 2, !"Debug Info Version", i32 3}
!7 = !{i32 1, !"wchar_size", i32 4}
!8 = !{!"clang version 9.0.0 (tags/RELEASE_900/final)"}
!9 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 13, type: !10, scopeLine: 14, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!10 = !DISubroutineType(types: !11)
!11 = !{!12}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DILocalVariable(name: "q", scope: !9, file: !1, line: 15, type: !14)
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !15, size: 64)
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!16 = !DILocation(line: 15, column: 8, scope: !9)
!17 = !DILocalVariable(name: "p", scope: !9, file: !1, line: 15, type: !14)
!18 = !DILocation(line: 15, column: 13, scope: !9)
!19 = !DILocalVariable(name: "a", scope: !9, file: !1, line: 16, type: !12)
!20 = !DILocation(line: 16, column: 6, scope: !9)
!21 = !DILocalVariable(name: "b", scope: !9, file: !1, line: 16, type: !12)
!22 = !DILocation(line: 16, column: 8, scope: !9)
!23 = !DILocalVariable(name: "s1", scope: !9, file: !1, line: 17, type: !24)
!24 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "s", file: !1, line: 8, size: 128, elements: !25)
!25 = !{!26, !27}
!26 = !DIDerivedType(tag: DW_TAG_member, name: "f1", scope: !24, file: !1, line: 9, baseType: !15, size: 64)
!27 = !DIDerivedType(tag: DW_TAG_member, name: "f2", scope: !24, file: !1, line: 10, baseType: !15, size: 64, offset: 64)
!28 = !DILocation(line: 17, column: 11, scope: !9)
!29 = !DILocation(line: 18, column: 5, scope: !9)
!30 = !DILocation(line: 18, column: 8, scope: !9)
!31 = !DILocation(line: 19, column: 5, scope: !9)
!32 = !DILocation(line: 19, column: 8, scope: !9)
!33 = !DILocation(line: 20, column: 11, scope: !9)
!34 = !DILocation(line: 20, column: 4, scope: !9)
!35 = !DILocation(line: 21, column: 6, scope: !9)
!36 = !DILocation(line: 21, column: 7, scope: !9)
!37 = !DILocation(line: 21, column: 4, scope: !9)
!38 = !DILocation(line: 22, column: 2, scope: !9)
!39 = !DILocation(line: 23, column: 2, scope: !9)

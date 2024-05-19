; ModuleID = 'basic_c_tests/struct-simple.c'
source_filename = "basic_c_tests/struct-simple.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.s = type { i32*, i32 }

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !9 {
  %1 = alloca i32, align 4
  %2 = alloca %struct.s, align 8
  %3 = alloca %struct.s, align 8
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  call void @llvm.dbg.declare(metadata %struct.s* %2, metadata !13, metadata !DIExpression()), !dbg !19
  call void @llvm.dbg.declare(metadata %struct.s* %3, metadata !20, metadata !DIExpression()), !dbg !21
  call void @llvm.dbg.declare(metadata i32* %4, metadata !22, metadata !DIExpression()), !dbg !23
  call void @llvm.dbg.declare(metadata i32* %5, metadata !24, metadata !DIExpression()), !dbg !25
  %6 = getelementptr inbounds %struct.s, %struct.s* %2, i32 0, i32 0, !dbg !26
  store i32* %4, i32** %6, align 8, !dbg !27
  %7 = getelementptr inbounds %struct.s, %struct.s* %2, i32 0, i32 0, !dbg !28
  %8 = load i32*, i32** %7, align 8, !dbg !28
  %9 = getelementptr inbounds %struct.s, %struct.s* %3, i32 0, i32 0, !dbg !29
  store i32* %8, i32** %9, align 8, !dbg !30
  %10 = getelementptr inbounds %struct.s, %struct.s* %3, i32 0, i32 0, !dbg !31
  %11 = load i32*, i32** %10, align 8, !dbg !31
  %12 = bitcast i32* %11 to i8*, !dbg !31
  %13 = bitcast i32* %4 to i8*, !dbg !31
  call void @__aser_alias__(i8* %12, i8* %13), !dbg !31
  ret i32 0, !dbg !32
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
!1 = !DIFile(filename: "basic_c_tests/struct-simple.c", directory: "/home/peiming/Documents/Projects/LLVMRace/TestCases/PTABen")
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
!13 = !DILocalVariable(name: "s1", scope: !9, file: !1, line: 15, type: !14)
!14 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "s", file: !1, line: 8, size: 128, elements: !15)
!15 = !{!16, !18}
!16 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !14, file: !1, line: 9, baseType: !17, size: 64)
!17 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!18 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !14, file: !1, line: 10, baseType: !12, size: 32, offset: 64)
!19 = !DILocation(line: 15, column: 11, scope: !9)
!20 = !DILocalVariable(name: "s2", scope: !9, file: !1, line: 15, type: !14)
!21 = !DILocation(line: 15, column: 15, scope: !9)
!22 = !DILocalVariable(name: "x", scope: !9, file: !1, line: 16, type: !12)
!23 = !DILocation(line: 16, column: 6, scope: !9)
!24 = !DILocalVariable(name: "y", scope: !9, file: !1, line: 16, type: !12)
!25 = !DILocation(line: 16, column: 9, scope: !9)
!26 = !DILocation(line: 17, column: 5, scope: !9)
!27 = !DILocation(line: 17, column: 7, scope: !9)
!28 = !DILocation(line: 18, column: 12, scope: !9)
!29 = !DILocation(line: 18, column: 5, scope: !9)
!30 = !DILocation(line: 18, column: 7, scope: !9)
!31 = !DILocation(line: 19, column: 2, scope: !9)
!32 = !DILocation(line: 20, column: 2, scope: !9)

; ModuleID = 'basic_c_tests/heap-wrapper.c'
source_filename = "basic_c_tests/heap-wrapper.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32* @my_alloc() #0 !dbg !11 {
  %1 = alloca i32*, align 8
  call void @llvm.dbg.declare(metadata i32** %1, metadata !14, metadata !DIExpression()), !dbg !15
  %2 = call i8* @malloc(i64 4), !dbg !16
  %3 = bitcast i8* %2 to i32*, !dbg !17
  store i32* %3, i32** %1, align 8, !dbg !15
  %4 = load i32*, i32** %1, align 8, !dbg !18
  ret i32* %4, !dbg !19
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare dso_local i8* @malloc(i64) #2

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !20 {
  %1 = alloca i32, align 4
  %2 = alloca i32*, align 8
  %3 = alloca i32*, align 8
  store i32 0, i32* %1, align 4
  call void @llvm.dbg.declare(metadata i32** %2, metadata !23, metadata !DIExpression()), !dbg !24
  %4 = call i32* @my_alloc(), !dbg !25
  store i32* %4, i32** %2, align 8, !dbg !24
  call void @llvm.dbg.declare(metadata i32** %3, metadata !26, metadata !DIExpression()), !dbg !27
  %5 = call i32* @my_alloc(), !dbg !28
  store i32* %5, i32** %3, align 8, !dbg !27
  %6 = load i32*, i32** %2, align 8, !dbg !29
  %7 = bitcast i32* %6 to i8*, !dbg !29
  %8 = load i32*, i32** %3, align 8, !dbg !29
  %9 = bitcast i32* %8 to i8*, !dbg !29
  call void @__aser_alias__(i8* %7, i8* %9), !dbg !29
  ret i32 0, !dbg !30
}

declare dso_local void @__aser_alias__(i8*, i8*) #2

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8, !9}
!llvm.ident = !{!10}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 9.0.0 (tags/RELEASE_900/final)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, nameTableKind: None)
!1 = !DIFile(filename: "basic_c_tests/heap-wrapper.c", directory: "/home/peiming/Documents/Projects/LLVMRace/TestCases/PTABen")
!2 = !{}
!3 = !{!4, !6}
!4 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !5, size: 64)
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = !{!"clang version 9.0.0 (tags/RELEASE_900/final)"}
!11 = distinct !DISubprogram(name: "my_alloc", scope: !1, file: !1, line: 11, type: !12, scopeLine: 11, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!12 = !DISubroutineType(types: !13)
!13 = !{!4}
!14 = !DILocalVariable(name: "p", scope: !11, file: !1, line: 12, type: !4)
!15 = !DILocation(line: 12, column: 8, scope: !11)
!16 = !DILocation(line: 12, column: 20, scope: !11)
!17 = !DILocation(line: 12, column: 12, scope: !11)
!18 = !DILocation(line: 13, column: 9, scope: !11)
!19 = !DILocation(line: 13, column: 2, scope: !11)
!20 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 16, type: !21, scopeLine: 16, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!21 = !DISubroutineType(types: !22)
!22 = !{!5}
!23 = !DILocalVariable(name: "o1", scope: !20, file: !1, line: 17, type: !4)
!24 = !DILocation(line: 17, column: 8, scope: !20)
!25 = !DILocation(line: 17, column: 13, scope: !20)
!26 = !DILocalVariable(name: "o2", scope: !20, file: !1, line: 18, type: !4)
!27 = !DILocation(line: 18, column: 8, scope: !20)
!28 = !DILocation(line: 18, column: 13, scope: !20)
!29 = !DILocation(line: 19, column: 2, scope: !20)
!30 = !DILocation(line: 20, column: 2, scope: !20)

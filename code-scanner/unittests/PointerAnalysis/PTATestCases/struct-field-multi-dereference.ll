; ModuleID = 'basic_c_tests/struct-field-multi-dereference.c'
source_filename = "basic_c_tests/struct-field-multi-dereference.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.MyStruct = type { i32*, %struct.MyStruct* }

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !9 {
  %1 = alloca i32, align 4
  %2 = alloca %struct.MyStruct*, align 8
  %3 = alloca %struct.MyStruct*, align 8
  %4 = alloca %struct.MyStruct, align 8
  %5 = alloca %struct.MyStruct, align 8
  %6 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  call void @llvm.dbg.declare(metadata %struct.MyStruct** %2, metadata !13, metadata !DIExpression()), !dbg !20
  call void @llvm.dbg.declare(metadata %struct.MyStruct** %3, metadata !21, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.declare(metadata %struct.MyStruct* %4, metadata !23, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.declare(metadata %struct.MyStruct* %5, metadata !25, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.declare(metadata i32* %6, metadata !27, metadata !DIExpression()), !dbg !28
  store %struct.MyStruct* %4, %struct.MyStruct** %2, align 8, !dbg !29
  store %struct.MyStruct* %4, %struct.MyStruct** %3, align 8, !dbg !30
  %7 = getelementptr inbounds %struct.MyStruct, %struct.MyStruct* %4, i32 0, i32 1, !dbg !31
  store %struct.MyStruct* %5, %struct.MyStruct** %7, align 8, !dbg !32
  %8 = load %struct.MyStruct*, %struct.MyStruct** %2, align 8, !dbg !33
  %9 = getelementptr inbounds %struct.MyStruct, %struct.MyStruct* %8, i32 0, i32 1, !dbg !34
  %10 = load %struct.MyStruct*, %struct.MyStruct** %9, align 8, !dbg !34
  %11 = getelementptr inbounds %struct.MyStruct, %struct.MyStruct* %10, i32 0, i32 0, !dbg !35
  store i32* %6, i32** %11, align 8, !dbg !36
  %12 = load %struct.MyStruct*, %struct.MyStruct** %3, align 8, !dbg !37
  %13 = getelementptr inbounds %struct.MyStruct, %struct.MyStruct* %12, i32 0, i32 1, !dbg !37
  %14 = load %struct.MyStruct*, %struct.MyStruct** %13, align 8, !dbg !37
  %15 = getelementptr inbounds %struct.MyStruct, %struct.MyStruct* %14, i32 0, i32 0, !dbg !37
  %16 = load i32*, i32** %15, align 8, !dbg !37
  %17 = bitcast i32* %16 to i8*, !dbg !37
  %18 = bitcast i32* %6 to i8*, !dbg !37
  call void @__aser_alias__(i8* %17, i8* %18), !dbg !37
  ret i32 0, !dbg !38
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
!1 = !DIFile(filename: "basic_c_tests/struct-field-multi-dereference.c", directory: "/home/peiming/Documents/Projects/LLVMRace/TestCases/PTABen")
!2 = !{}
!3 = !{!4}
!4 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!5 = !{i32 2, !"Dwarf Version", i32 4}
!6 = !{i32 2, !"Debug Info Version", i32 3}
!7 = !{i32 1, !"wchar_size", i32 4}
!8 = !{!"clang version 9.0.0 (tags/RELEASE_900/final)"}
!9 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 14, type: !10, scopeLine: 14, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!10 = !DISubroutineType(types: !11)
!11 = !{!12}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DILocalVariable(name: "p", scope: !9, file: !1, line: 15, type: !14)
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !15, size: 64)
!15 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "MyStruct", file: !1, line: 9, size: 128, elements: !16)
!16 = !{!17, !19}
!17 = !DIDerivedType(tag: DW_TAG_member, name: "f1", scope: !15, file: !1, line: 10, baseType: !18, size: 64)
!18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!19 = !DIDerivedType(tag: DW_TAG_member, name: "f2", scope: !15, file: !1, line: 11, baseType: !14, size: 64, offset: 64)
!20 = !DILocation(line: 15, column: 19, scope: !9)
!21 = !DILocalVariable(name: "q", scope: !9, file: !1, line: 15, type: !14)
!22 = !DILocation(line: 15, column: 23, scope: !9)
!23 = !DILocalVariable(name: "ms1", scope: !9, file: !1, line: 16, type: !15)
!24 = !DILocation(line: 16, column: 18, scope: !9)
!25 = !DILocalVariable(name: "ms2", scope: !9, file: !1, line: 16, type: !15)
!26 = !DILocation(line: 16, column: 23, scope: !9)
!27 = !DILocalVariable(name: "x", scope: !9, file: !1, line: 17, type: !12)
!28 = !DILocation(line: 17, column: 6, scope: !9)
!29 = !DILocation(line: 18, column: 4, scope: !9)
!30 = !DILocation(line: 19, column: 4, scope: !9)
!31 = !DILocation(line: 20, column: 6, scope: !9)
!32 = !DILocation(line: 20, column: 9, scope: !9)
!33 = !DILocation(line: 21, column: 2, scope: !9)
!34 = !DILocation(line: 21, column: 5, scope: !9)
!35 = !DILocation(line: 21, column: 9, scope: !9)
!36 = !DILocation(line: 21, column: 12, scope: !9)
!37 = !DILocation(line: 22, column: 2, scope: !9)
!38 = !DILocation(line: 23, column: 2, scope: !9)

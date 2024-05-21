; ModuleID = 'basic_c_tests/array-varIdx2.c'
source_filename = "basic_c_tests/array-varIdx2.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.MyStruct = type { i32*, i32* }

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !9 {
  %1 = alloca i32, align 4
  %2 = alloca [2 x %struct.MyStruct], align 16
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  call void @llvm.dbg.declare(metadata [2 x %struct.MyStruct]* %2, metadata !13, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.declare(metadata i32* %3, metadata !23, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.declare(metadata i32* %4, metadata !25, metadata !DIExpression()), !dbg !26
  %5 = getelementptr inbounds [2 x %struct.MyStruct], [2 x %struct.MyStruct]* %2, i64 0, i64 0, !dbg !27
  %6 = getelementptr inbounds %struct.MyStruct, %struct.MyStruct* %5, i32 0, i32 0, !dbg !28
  store i32* %3, i32** %6, align 16, !dbg !29
  %7 = getelementptr inbounds [2 x %struct.MyStruct], [2 x %struct.MyStruct]* %2, i64 0, i64 1, !dbg !30
  %8 = getelementptr inbounds %struct.MyStruct, %struct.MyStruct* %7, i32 0, i32 0, !dbg !31
  store i32* %4, i32** %8, align 16, !dbg !32
  %9 = load i32, i32* %3, align 4, !dbg !33
  %10 = sext i32 %9 to i64, !dbg !33
  %11 = getelementptr inbounds [2 x %struct.MyStruct], [2 x %struct.MyStruct]* %2, i64 0, i64 %10, !dbg !33
  %12 = getelementptr inbounds %struct.MyStruct, %struct.MyStruct* %11, i32 0, i32 0, !dbg !33
  %13 = load i32*, i32** %12, align 16, !dbg !33
  %14 = bitcast i32* %13 to i8*, !dbg !33
  %15 = load i32, i32* %4, align 4, !dbg !33
  %16 = sext i32 %15 to i64, !dbg !33
  %17 = getelementptr inbounds [2 x %struct.MyStruct], [2 x %struct.MyStruct]* %2, i64 0, i64 %16, !dbg !33
  %18 = getelementptr inbounds %struct.MyStruct, %struct.MyStruct* %17, i32 0, i32 1, !dbg !33
  %19 = load i32*, i32** %18, align 8, !dbg !33
  %20 = bitcast i32* %19 to i8*, !dbg !33
  call void @__aser_no_alias__(i8* %14, i8* %20), !dbg !33
  %21 = load i32, i32* %3, align 4, !dbg !34
  %22 = sext i32 %21 to i64, !dbg !34
  %23 = getelementptr inbounds [2 x %struct.MyStruct], [2 x %struct.MyStruct]* %2, i64 0, i64 %22, !dbg !34
  %24 = getelementptr inbounds %struct.MyStruct, %struct.MyStruct* %23, i32 0, i32 0, !dbg !34
  %25 = load i32*, i32** %24, align 16, !dbg !34
  %26 = bitcast i32* %25 to i8*, !dbg !34
  %27 = load i32, i32* %4, align 4, !dbg !34
  %28 = sext i32 %27 to i64, !dbg !34
  %29 = getelementptr inbounds [2 x %struct.MyStruct], [2 x %struct.MyStruct]* %2, i64 0, i64 %28, !dbg !34
  %30 = getelementptr inbounds %struct.MyStruct, %struct.MyStruct* %29, i32 0, i32 0, !dbg !34
  %31 = load i32*, i32** %30, align 16, !dbg !34
  %32 = bitcast i32* %31 to i8*, !dbg !34
  call void @__aser_alias__(i8* %26, i8* %32), !dbg !34
  ret i32 0, !dbg !35
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare dso_local void @__aser_no_alias__(i8*, i8*) #2

declare dso_local void @__aser_alias__(i8*, i8*) #2

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!5, !6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 9.0.0 (tags/RELEASE_900/final)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, nameTableKind: None)
!1 = !DIFile(filename: "basic_c_tests/array-varIdx2.c", directory: "/home/peiming/Documents/Projects/LLVMRace/TestCases/PTABen")
!2 = !{}
!3 = !{!4}
!4 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!5 = !{i32 2, !"Dwarf Version", i32 4}
!6 = !{i32 2, !"Debug Info Version", i32 3}
!7 = !{i32 1, !"wchar_size", i32 4}
!8 = !{!"clang version 9.0.0 (tags/RELEASE_900/final)"}
!9 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 13, type: !10, scopeLine: 13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!10 = !DISubroutineType(types: !11)
!11 = !{!12}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DILocalVariable(name: "s", scope: !9, file: !1, line: 14, type: !14)
!14 = !DICompositeType(tag: DW_TAG_array_type, baseType: !15, size: 256, elements: !20)
!15 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "MyStruct", file: !1, line: 8, size: 128, elements: !16)
!16 = !{!17, !19}
!17 = !DIDerivedType(tag: DW_TAG_member, name: "f1", scope: !15, file: !1, line: 9, baseType: !18, size: 64)
!18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!19 = !DIDerivedType(tag: DW_TAG_member, name: "f2", scope: !15, file: !1, line: 10, baseType: !18, size: 64, offset: 64)
!20 = !{!21}
!21 = !DISubrange(count: 2)
!22 = !DILocation(line: 14, column: 18, scope: !9)
!23 = !DILocalVariable(name: "a", scope: !9, file: !1, line: 15, type: !12)
!24 = !DILocation(line: 15, column: 6, scope: !9)
!25 = !DILocalVariable(name: "b", scope: !9, file: !1, line: 15, type: !12)
!26 = !DILocation(line: 15, column: 8, scope: !9)
!27 = !DILocation(line: 16, column: 2, scope: !9)
!28 = !DILocation(line: 16, column: 7, scope: !9)
!29 = !DILocation(line: 16, column: 10, scope: !9)
!30 = !DILocation(line: 17, column: 2, scope: !9)
!31 = !DILocation(line: 17, column: 7, scope: !9)
!32 = !DILocation(line: 17, column: 10, scope: !9)
!33 = !DILocation(line: 21, column: 2, scope: !9)
!34 = !DILocation(line: 22, column: 2, scope: !9)
!35 = !DILocation(line: 24, column: 2, scope: !9)

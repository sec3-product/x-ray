; ModuleID = 'basic_c_tests/struct-array.c'
source_filename = "basic_c_tests/struct-array.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.ArrayStruct = type { i32, i8, [100 x i32], i32 }

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !9 {
  %1 = alloca i32, align 4
  %2 = alloca %struct.ArrayStruct*, align 8
  %3 = alloca %struct.ArrayStruct, align 4
  %4 = alloca i32*, align 8
  store i32 0, i32* %1, align 4
  call void @llvm.dbg.declare(metadata %struct.ArrayStruct** %2, metadata !13, metadata !DIExpression()), !dbg !25
  call void @llvm.dbg.declare(metadata %struct.ArrayStruct* %3, metadata !26, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.declare(metadata i32** %4, metadata !28, metadata !DIExpression()), !dbg !30
  store %struct.ArrayStruct* %3, %struct.ArrayStruct** %2, align 8, !dbg !31
  %5 = getelementptr inbounds %struct.ArrayStruct, %struct.ArrayStruct* %3, i32 0, i32 2, !dbg !32
  %6 = getelementptr inbounds [100 x i32], [100 x i32]* %5, i64 0, i64 40, !dbg !33
  store i32* %6, i32** %4, align 8, !dbg !34
  %7 = load %struct.ArrayStruct*, %struct.ArrayStruct** %2, align 8, !dbg !35
  %8 = getelementptr inbounds %struct.ArrayStruct, %struct.ArrayStruct* %7, i32 0, i32 2, !dbg !35
  %9 = getelementptr inbounds [100 x i32], [100 x i32]* %8, i64 0, i64 10, !dbg !35
  %10 = bitcast i32* %9 to i8*, !dbg !35
  %11 = load i32*, i32** %4, align 8, !dbg !35
  %12 = bitcast i32* %11 to i8*, !dbg !35
  call void @__aser_alias__(i8* %10, i8* %12), !dbg !35
  %13 = load %struct.ArrayStruct*, %struct.ArrayStruct** %2, align 8, !dbg !36
  %14 = getelementptr inbounds %struct.ArrayStruct, %struct.ArrayStruct* %13, i32 0, i32 2, !dbg !36
  %15 = getelementptr inbounds [100 x i32], [100 x i32]* %14, i64 0, i64 20, !dbg !36
  %16 = bitcast i32* %15 to i8*, !dbg !36
  %17 = load %struct.ArrayStruct*, %struct.ArrayStruct** %2, align 8, !dbg !36
  %18 = getelementptr inbounds %struct.ArrayStruct, %struct.ArrayStruct* %17, i32 0, i32 2, !dbg !36
  %19 = getelementptr inbounds [100 x i32], [100 x i32]* %18, i64 0, i64 30, !dbg !36
  %20 = bitcast i32* %19 to i8*, !dbg !36
  call void @__aser_alias__(i8* %16, i8* %20), !dbg !36
  %21 = getelementptr inbounds %struct.ArrayStruct, %struct.ArrayStruct* %3, i32 0, i32 2, !dbg !37
  %22 = getelementptr inbounds [100 x i32], [100 x i32]* %21, i64 0, i64 0, !dbg !37
  %23 = bitcast i32* %22 to i8*, !dbg !37
  %24 = getelementptr inbounds %struct.ArrayStruct, %struct.ArrayStruct* %3, i32 0, i32 2, !dbg !37
  %25 = getelementptr inbounds [100 x i32], [100 x i32]* %24, i64 0, i64 99, !dbg !37
  %26 = bitcast i32* %25 to i8*, !dbg !37
  call void @__aser_alias__(i8* %23, i8* %26), !dbg !37
  %27 = load %struct.ArrayStruct*, %struct.ArrayStruct** %2, align 8, !dbg !38
  %28 = getelementptr inbounds %struct.ArrayStruct, %struct.ArrayStruct* %27, i32 0, i32 2, !dbg !38
  %29 = getelementptr inbounds [100 x i32], [100 x i32]* %28, i64 0, i64 0, !dbg !38
  %30 = bitcast i32* %29 to i8*, !dbg !38
  %31 = getelementptr inbounds %struct.ArrayStruct, %struct.ArrayStruct* %3, i32 0, i32 3, !dbg !38
  %32 = bitcast i32* %31 to i8*, !dbg !38
  call void @__aser_no_alias__(i8* %30, i8* %32), !dbg !38
  ret i32 0, !dbg !39
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare dso_local void @__aser_alias__(i8*, i8*) #2

declare dso_local void @__aser_no_alias__(i8*, i8*) #2

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!5, !6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 9.0.0 (tags/RELEASE_900/final)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, nameTableKind: None)
!1 = !DIFile(filename: "basic_c_tests/struct-array.c", directory: "/home/peiming/Documents/Projects/LLVMRace/TestCases/PTABen")
!2 = !{}
!3 = !{!4}
!4 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!5 = !{i32 2, !"Dwarf Version", i32 4}
!6 = !{i32 2, !"Debug Info Version", i32 3}
!7 = !{i32 1, !"wchar_size", i32 4}
!8 = !{!"clang version 9.0.0 (tags/RELEASE_900/final)"}
!9 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 15, type: !10, scopeLine: 15, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!10 = !DISubroutineType(types: !11)
!11 = !{!12}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DILocalVariable(name: "p", scope: !9, file: !1, line: 16, type: !14)
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !15, size: 64)
!15 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "ArrayStruct", file: !1, line: 8, size: 3296, elements: !16)
!16 = !{!17, !18, !20, !24}
!17 = !DIDerivedType(tag: DW_TAG_member, name: "f1", scope: !15, file: !1, line: 9, baseType: !12, size: 32)
!18 = !DIDerivedType(tag: DW_TAG_member, name: "f2", scope: !15, file: !1, line: 10, baseType: !19, size: 8, offset: 32)
!19 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!20 = !DIDerivedType(tag: DW_TAG_member, name: "f3", scope: !15, file: !1, line: 11, baseType: !21, size: 3200, offset: 64)
!21 = !DICompositeType(tag: DW_TAG_array_type, baseType: !12, size: 3200, elements: !22)
!22 = !{!23}
!23 = !DISubrange(count: 100)
!24 = !DIDerivedType(tag: DW_TAG_member, name: "f4", scope: !15, file: !1, line: 12, baseType: !12, size: 32, offset: 3264)
!25 = !DILocation(line: 16, column: 22, scope: !9)
!26 = !DILocalVariable(name: "s", scope: !9, file: !1, line: 17, type: !15)
!27 = !DILocation(line: 17, column: 21, scope: !9)
!28 = !DILocalVariable(name: "q", scope: !9, file: !1, line: 18, type: !29)
!29 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!30 = !DILocation(line: 18, column: 7, scope: !9)
!31 = !DILocation(line: 20, column: 4, scope: !9)
!32 = !DILocation(line: 21, column: 9, scope: !9)
!33 = !DILocation(line: 21, column: 7, scope: !9)
!34 = !DILocation(line: 21, column: 4, scope: !9)
!35 = !DILocation(line: 22, column: 2, scope: !9)
!36 = !DILocation(line: 23, column: 2, scope: !9)
!37 = !DILocation(line: 24, column: 2, scope: !9)
!38 = !DILocation(line: 25, column: 2, scope: !9)
!39 = !DILocation(line: 27, column: 2, scope: !9)

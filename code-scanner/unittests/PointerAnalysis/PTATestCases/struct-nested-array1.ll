; ModuleID = 'basic_c_tests/struct-nested-array1.c'
source_filename = "basic_c_tests/struct-nested-array1.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.ArrayStruct = type { i32, i8, %struct.InnerArrayStruct, i32 }
%struct.InnerArrayStruct = type { [100 x i32] }

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !9 {
  %1 = alloca i32, align 4
  %2 = alloca %struct.ArrayStruct*, align 8
  %3 = alloca %struct.ArrayStruct, align 4
  %4 = alloca i32*, align 8
  store i32 0, i32* %1, align 4
  call void @llvm.dbg.declare(metadata %struct.ArrayStruct** %2, metadata !13, metadata !DIExpression()), !dbg !28
  call void @llvm.dbg.declare(metadata %struct.ArrayStruct* %3, metadata !29, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.declare(metadata i32** %4, metadata !31, metadata !DIExpression()), !dbg !33
  store %struct.ArrayStruct* %3, %struct.ArrayStruct** %2, align 8, !dbg !34
  %5 = getelementptr inbounds %struct.ArrayStruct, %struct.ArrayStruct* %3, i32 0, i32 2, !dbg !35
  %6 = getelementptr inbounds %struct.InnerArrayStruct, %struct.InnerArrayStruct* %5, i32 0, i32 0, !dbg !36
  %7 = getelementptr inbounds [100 x i32], [100 x i32]* %6, i64 0, i64 40, !dbg !37
  store i32* %7, i32** %4, align 8, !dbg !38
  %8 = load %struct.ArrayStruct*, %struct.ArrayStruct** %2, align 8, !dbg !39
  %9 = getelementptr inbounds %struct.ArrayStruct, %struct.ArrayStruct* %8, i32 0, i32 2, !dbg !39
  %10 = getelementptr inbounds %struct.InnerArrayStruct, %struct.InnerArrayStruct* %9, i32 0, i32 0, !dbg !39
  %11 = getelementptr inbounds [100 x i32], [100 x i32]* %10, i64 0, i64 10, !dbg !39
  %12 = bitcast i32* %11 to i8*, !dbg !39
  %13 = load i32*, i32** %4, align 8, !dbg !39
  %14 = bitcast i32* %13 to i8*, !dbg !39
  call void @__aser_alias__(i8* %12, i8* %14), !dbg !39
  %15 = load %struct.ArrayStruct*, %struct.ArrayStruct** %2, align 8, !dbg !40
  %16 = getelementptr inbounds %struct.ArrayStruct, %struct.ArrayStruct* %15, i32 0, i32 2, !dbg !40
  %17 = getelementptr inbounds %struct.InnerArrayStruct, %struct.InnerArrayStruct* %16, i32 0, i32 0, !dbg !40
  %18 = getelementptr inbounds [100 x i32], [100 x i32]* %17, i64 0, i64 20, !dbg !40
  %19 = bitcast i32* %18 to i8*, !dbg !40
  %20 = load %struct.ArrayStruct*, %struct.ArrayStruct** %2, align 8, !dbg !40
  %21 = getelementptr inbounds %struct.ArrayStruct, %struct.ArrayStruct* %20, i32 0, i32 2, !dbg !40
  %22 = getelementptr inbounds %struct.InnerArrayStruct, %struct.InnerArrayStruct* %21, i32 0, i32 0, !dbg !40
  %23 = getelementptr inbounds [100 x i32], [100 x i32]* %22, i64 0, i64 30, !dbg !40
  %24 = bitcast i32* %23 to i8*, !dbg !40
  call void @__aser_alias__(i8* %19, i8* %24), !dbg !40
  %25 = load %struct.ArrayStruct*, %struct.ArrayStruct** %2, align 8, !dbg !41
  %26 = getelementptr inbounds %struct.ArrayStruct, %struct.ArrayStruct* %25, i32 0, i32 2, !dbg !41
  %27 = getelementptr inbounds %struct.InnerArrayStruct, %struct.InnerArrayStruct* %26, i32 0, i32 0, !dbg !41
  %28 = getelementptr inbounds [100 x i32], [100 x i32]* %27, i64 0, i64 0, !dbg !41
  %29 = bitcast i32* %28 to i8*, !dbg !41
  %30 = getelementptr inbounds %struct.ArrayStruct, %struct.ArrayStruct* %3, i32 0, i32 3, !dbg !41
  %31 = bitcast i32* %30 to i8*, !dbg !41
  call void @__aser_no_alias__(i8* %29, i8* %31), !dbg !41
  ret i32 0, !dbg !42
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
!1 = !DIFile(filename: "basic_c_tests/struct-nested-array1.c", directory: "/home/peiming/Documents/Projects/LLVMRace/TestCases/PTABen")
!2 = !{}
!3 = !{!4}
!4 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!5 = !{i32 2, !"Dwarf Version", i32 4}
!6 = !{i32 2, !"Debug Info Version", i32 3}
!7 = !{i32 1, !"wchar_size", i32 4}
!8 = !{!"clang version 9.0.0 (tags/RELEASE_900/final)"}
!9 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 19, type: !10, scopeLine: 19, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!10 = !DISubroutineType(types: !11)
!11 = !{!12}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DILocalVariable(name: "p", scope: !9, file: !1, line: 20, type: !14)
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !15, size: 64)
!15 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "ArrayStruct", file: !1, line: 12, size: 3296, elements: !16)
!16 = !{!17, !18, !20, !27}
!17 = !DIDerivedType(tag: DW_TAG_member, name: "f1", scope: !15, file: !1, line: 13, baseType: !12, size: 32)
!18 = !DIDerivedType(tag: DW_TAG_member, name: "f2", scope: !15, file: !1, line: 14, baseType: !19, size: 8, offset: 32)
!19 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!20 = !DIDerivedType(tag: DW_TAG_member, name: "f3", scope: !15, file: !1, line: 15, baseType: !21, size: 3200, offset: 64)
!21 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "InnerArrayStruct", file: !1, line: 8, size: 3200, elements: !22)
!22 = !{!23}
!23 = !DIDerivedType(tag: DW_TAG_member, name: "f1", scope: !21, file: !1, line: 9, baseType: !24, size: 3200)
!24 = !DICompositeType(tag: DW_TAG_array_type, baseType: !12, size: 3200, elements: !25)
!25 = !{!26}
!26 = !DISubrange(count: 100)
!27 = !DIDerivedType(tag: DW_TAG_member, name: "f4", scope: !15, file: !1, line: 16, baseType: !12, size: 32, offset: 3264)
!28 = !DILocation(line: 20, column: 22, scope: !9)
!29 = !DILocalVariable(name: "s", scope: !9, file: !1, line: 21, type: !15)
!30 = !DILocation(line: 21, column: 21, scope: !9)
!31 = !DILocalVariable(name: "q", scope: !9, file: !1, line: 22, type: !32)
!32 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!33 = !DILocation(line: 22, column: 7, scope: !9)
!34 = !DILocation(line: 24, column: 4, scope: !9)
!35 = !DILocation(line: 25, column: 9, scope: !9)
!36 = !DILocation(line: 25, column: 12, scope: !9)
!37 = !DILocation(line: 25, column: 7, scope: !9)
!38 = !DILocation(line: 25, column: 4, scope: !9)
!39 = !DILocation(line: 26, column: 2, scope: !9)
!40 = !DILocation(line: 27, column: 2, scope: !9)
!41 = !DILocation(line: 28, column: 2, scope: !9)
!42 = !DILocation(line: 30, column: 2, scope: !9)

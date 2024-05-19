; ModuleID = 'basic_c_tests/struct-nested-array2.c'
source_filename = "basic_c_tests/struct-nested-array2.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.ArrayStruct = type { i32, i8, %struct.MidArrayStruct, i32* }
%struct.MidArrayStruct = type { i8, [5 x %struct.InnerArrayStruct], [20 x double] }
%struct.InnerArrayStruct = type { [10 x i32*], i8, double }

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !9 {
  %1 = alloca i32, align 4
  %2 = alloca %struct.ArrayStruct*, align 8
  %3 = alloca %struct.ArrayStruct, align 8
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  call void @llvm.dbg.declare(metadata %struct.ArrayStruct** %2, metadata !13, metadata !DIExpression()), !dbg !43
  call void @llvm.dbg.declare(metadata %struct.ArrayStruct* %3, metadata !44, metadata !DIExpression()), !dbg !45
  call void @llvm.dbg.declare(metadata i32* %4, metadata !46, metadata !DIExpression()), !dbg !47
  call void @llvm.dbg.declare(metadata i32* %5, metadata !48, metadata !DIExpression()), !dbg !49
  %6 = getelementptr inbounds %struct.ArrayStruct, %struct.ArrayStruct* %3, i32 0, i32 3, !dbg !50
  store i32* %4, i32** %6, align 8, !dbg !51
  store %struct.ArrayStruct* %3, %struct.ArrayStruct** %2, align 8, !dbg !52
  %7 = getelementptr inbounds %struct.ArrayStruct, %struct.ArrayStruct* %3, i32 0, i32 3, !dbg !53
  %8 = load i32*, i32** %7, align 8, !dbg !53
  %9 = load %struct.ArrayStruct*, %struct.ArrayStruct** %2, align 8, !dbg !54
  %10 = getelementptr inbounds %struct.ArrayStruct, %struct.ArrayStruct* %9, i32 0, i32 2, !dbg !55
  %11 = getelementptr inbounds %struct.MidArrayStruct, %struct.MidArrayStruct* %10, i32 0, i32 1, !dbg !56
  %12 = getelementptr inbounds [5 x %struct.InnerArrayStruct], [5 x %struct.InnerArrayStruct]* %11, i64 0, i64 2, !dbg !54
  %13 = getelementptr inbounds %struct.InnerArrayStruct, %struct.InnerArrayStruct* %12, i32 0, i32 0, !dbg !57
  %14 = getelementptr inbounds [10 x i32*], [10 x i32*]* %13, i64 0, i64 2, !dbg !54
  store i32* %8, i32** %14, align 8, !dbg !58
  %15 = load %struct.ArrayStruct*, %struct.ArrayStruct** %2, align 8, !dbg !59
  %16 = getelementptr inbounds %struct.ArrayStruct, %struct.ArrayStruct* %15, i32 0, i32 2, !dbg !60
  %17 = getelementptr inbounds %struct.MidArrayStruct, %struct.MidArrayStruct* %16, i32 0, i32 1, !dbg !61
  %18 = getelementptr inbounds [5 x %struct.InnerArrayStruct], [5 x %struct.InnerArrayStruct]* %17, i64 0, i64 3, !dbg !59
  %19 = getelementptr inbounds %struct.InnerArrayStruct, %struct.InnerArrayStruct* %18, i32 0, i32 0, !dbg !62
  %20 = getelementptr inbounds [10 x i32*], [10 x i32*]* %19, i64 0, i64 3, !dbg !59
  store i32* %5, i32** %20, align 8, !dbg !63
  %21 = load %struct.ArrayStruct*, %struct.ArrayStruct** %2, align 8, !dbg !64
  %22 = getelementptr inbounds %struct.ArrayStruct, %struct.ArrayStruct* %21, i32 0, i32 2, !dbg !64
  %23 = getelementptr inbounds %struct.MidArrayStruct, %struct.MidArrayStruct* %22, i32 0, i32 1, !dbg !64
  %24 = getelementptr inbounds [5 x %struct.InnerArrayStruct], [5 x %struct.InnerArrayStruct]* %23, i64 0, i64 1, !dbg !64
  %25 = getelementptr inbounds %struct.InnerArrayStruct, %struct.InnerArrayStruct* %24, i32 0, i32 0, !dbg !64
  %26 = getelementptr inbounds [10 x i32*], [10 x i32*]* %25, i64 0, i64 1, !dbg !64
  %27 = load i32*, i32** %26, align 8, !dbg !64
  %28 = bitcast i32* %27 to i8*, !dbg !64
  %29 = bitcast i32* %4 to i8*, !dbg !64
  call void @__aser_alias__(i8* %28, i8* %29), !dbg !64
  %30 = getelementptr inbounds %struct.ArrayStruct, %struct.ArrayStruct* %3, i32 0, i32 2, !dbg !65
  %31 = getelementptr inbounds %struct.MidArrayStruct, %struct.MidArrayStruct* %30, i32 0, i32 1, !dbg !65
  %32 = getelementptr inbounds [5 x %struct.InnerArrayStruct], [5 x %struct.InnerArrayStruct]* %31, i64 0, i64 3, !dbg !65
  %33 = getelementptr inbounds %struct.InnerArrayStruct, %struct.InnerArrayStruct* %32, i32 0, i32 0, !dbg !65
  %34 = getelementptr inbounds [10 x i32*], [10 x i32*]* %33, i64 0, i64 0, !dbg !65
  %35 = load i32*, i32** %34, align 8, !dbg !65
  %36 = bitcast i32* %35 to i8*, !dbg !65
  %37 = bitcast i32* %5 to i8*, !dbg !65
  call void @__aser_alias__(i8* %36, i8* %37), !dbg !65
  ret i32 0, !dbg !66
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
!1 = !DIFile(filename: "basic_c_tests/struct-nested-array2.c", directory: "/home/peiming/Documents/Projects/LLVMRace/TestCases/PTABen")
!2 = !{}
!3 = !{!4}
!4 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!5 = !{i32 2, !"Dwarf Version", i32 4}
!6 = !{i32 2, !"Debug Info Version", i32 3}
!7 = !{i32 1, !"wchar_size", i32 4}
!8 = !{!"clang version 9.0.0 (tags/RELEASE_900/final)"}
!9 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 27, type: !10, scopeLine: 27, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!10 = !DISubroutineType(types: !11)
!11 = !{!12}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DILocalVariable(name: "p", scope: !9, file: !1, line: 28, type: !14)
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !15, size: 64)
!15 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "ArrayStruct", file: !1, line: 20, size: 5312, elements: !16)
!16 = !{!17, !18, !20, !42}
!17 = !DIDerivedType(tag: DW_TAG_member, name: "out1", scope: !15, file: !1, line: 21, baseType: !12, size: 32)
!18 = !DIDerivedType(tag: DW_TAG_member, name: "out2", scope: !15, file: !1, line: 22, baseType: !19, size: 8, offset: 32)
!19 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!20 = !DIDerivedType(tag: DW_TAG_member, name: "out3", scope: !15, file: !1, line: 23, baseType: !21, size: 5184, offset: 64)
!21 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "MidArrayStruct", file: !1, line: 14, size: 5184, elements: !22)
!22 = !{!23, !24, !38}
!23 = !DIDerivedType(tag: DW_TAG_member, name: "mid1", scope: !21, file: !1, line: 15, baseType: !19, size: 8)
!24 = !DIDerivedType(tag: DW_TAG_member, name: "mid2", scope: !21, file: !1, line: 16, baseType: !25, size: 3840, offset: 64)
!25 = !DICompositeType(tag: DW_TAG_array_type, baseType: !26, size: 3840, elements: !36)
!26 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "InnerArrayStruct", file: !1, line: 8, size: 768, elements: !27)
!27 = !{!28, !33, !34}
!28 = !DIDerivedType(tag: DW_TAG_member, name: "in1", scope: !26, file: !1, line: 9, baseType: !29, size: 640)
!29 = !DICompositeType(tag: DW_TAG_array_type, baseType: !30, size: 640, elements: !31)
!30 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!31 = !{!32}
!32 = !DISubrange(count: 10)
!33 = !DIDerivedType(tag: DW_TAG_member, name: "in2", scope: !26, file: !1, line: 10, baseType: !19, size: 8, offset: 640)
!34 = !DIDerivedType(tag: DW_TAG_member, name: "in3", scope: !26, file: !1, line: 11, baseType: !35, size: 64, offset: 704)
!35 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!36 = !{!37}
!37 = !DISubrange(count: 5)
!38 = !DIDerivedType(tag: DW_TAG_member, name: "mid3", scope: !21, file: !1, line: 17, baseType: !39, size: 1280, offset: 3904)
!39 = !DICompositeType(tag: DW_TAG_array_type, baseType: !35, size: 1280, elements: !40)
!40 = !{!41}
!41 = !DISubrange(count: 20)
!42 = !DIDerivedType(tag: DW_TAG_member, name: "out4", scope: !15, file: !1, line: 24, baseType: !30, size: 64, offset: 5248)
!43 = !DILocation(line: 28, column: 22, scope: !9)
!44 = !DILocalVariable(name: "s", scope: !9, file: !1, line: 29, type: !15)
!45 = !DILocation(line: 29, column: 21, scope: !9)
!46 = !DILocalVariable(name: "x", scope: !9, file: !1, line: 30, type: !12)
!47 = !DILocation(line: 30, column: 6, scope: !9)
!48 = !DILocalVariable(name: "y", scope: !9, file: !1, line: 30, type: !12)
!49 = !DILocation(line: 30, column: 9, scope: !9)
!50 = !DILocation(line: 32, column: 4, scope: !9)
!51 = !DILocation(line: 32, column: 9, scope: !9)
!52 = !DILocation(line: 33, column: 4, scope: !9)
!53 = !DILocation(line: 34, column: 29, scope: !9)
!54 = !DILocation(line: 34, column: 2, scope: !9)
!55 = !DILocation(line: 34, column: 5, scope: !9)
!56 = !DILocation(line: 34, column: 10, scope: !9)
!57 = !DILocation(line: 34, column: 18, scope: !9)
!58 = !DILocation(line: 34, column: 25, scope: !9)
!59 = !DILocation(line: 35, column: 2, scope: !9)
!60 = !DILocation(line: 35, column: 5, scope: !9)
!61 = !DILocation(line: 35, column: 10, scope: !9)
!62 = !DILocation(line: 35, column: 18, scope: !9)
!63 = !DILocation(line: 35, column: 25, scope: !9)
!64 = !DILocation(line: 37, column: 2, scope: !9)
!65 = !DILocation(line: 38, column: 2, scope: !9)
!66 = !DILocation(line: 40, column: 2, scope: !9)

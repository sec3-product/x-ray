; ModuleID = 'basic_c_tests/struct-nested-1-layer.c'
source_filename = "basic_c_tests/struct-nested-1-layer.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.MyStruct1 = type { i32*, %struct.MyStruct2 }
%struct.MyStruct2 = type { i32*, i32* }

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !9 {
  %1 = alloca i32, align 4
  %2 = alloca %struct.MyStruct1, align 8
  %3 = alloca %struct.MyStruct1*, align 8
  %4 = alloca %struct.MyStruct2*, align 8
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  call void @llvm.dbg.declare(metadata %struct.MyStruct1* %2, metadata !13, metadata !DIExpression()), !dbg !23
  call void @llvm.dbg.declare(metadata %struct.MyStruct1** %3, metadata !24, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.declare(metadata %struct.MyStruct2** %4, metadata !27, metadata !DIExpression()), !dbg !29
  call void @llvm.dbg.declare(metadata i32* %5, metadata !30, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.declare(metadata i32* %6, metadata !32, metadata !DIExpression()), !dbg !33
  call void @llvm.dbg.declare(metadata i32* %7, metadata !34, metadata !DIExpression()), !dbg !35
  %8 = getelementptr inbounds %struct.MyStruct1, %struct.MyStruct1* %2, i32 0, i32 0, !dbg !36
  store i32* %7, i32** %8, align 8, !dbg !37
  %9 = getelementptr inbounds %struct.MyStruct1, %struct.MyStruct1* %2, i32 0, i32 1, !dbg !38
  %10 = getelementptr inbounds %struct.MyStruct2, %struct.MyStruct2* %9, i32 0, i32 0, !dbg !39
  store i32* %5, i32** %10, align 8, !dbg !40
  %11 = getelementptr inbounds %struct.MyStruct1, %struct.MyStruct1* %2, i32 0, i32 1, !dbg !41
  %12 = getelementptr inbounds %struct.MyStruct2, %struct.MyStruct2* %11, i32 0, i32 1, !dbg !42
  store i32* %6, i32** %12, align 8, !dbg !43
  store %struct.MyStruct1* %2, %struct.MyStruct1** %3, align 8, !dbg !44
  %13 = getelementptr inbounds %struct.MyStruct1, %struct.MyStruct1* %2, i32 0, i32 1, !dbg !45
  store %struct.MyStruct2* %13, %struct.MyStruct2** %4, align 8, !dbg !46
  %14 = load %struct.MyStruct2*, %struct.MyStruct2** %4, align 8, !dbg !47
  %15 = getelementptr inbounds %struct.MyStruct2, %struct.MyStruct2* %14, i32 0, i32 1, !dbg !47
  %16 = load i32*, i32** %15, align 8, !dbg !47
  %17 = bitcast i32* %16 to i8*, !dbg !47
  %18 = load %struct.MyStruct1*, %struct.MyStruct1** %3, align 8, !dbg !47
  %19 = getelementptr inbounds %struct.MyStruct1, %struct.MyStruct1* %18, i32 0, i32 1, !dbg !47
  %20 = getelementptr inbounds %struct.MyStruct2, %struct.MyStruct2* %19, i32 0, i32 0, !dbg !47
  %21 = load i32*, i32** %20, align 8, !dbg !47
  %22 = bitcast i32* %21 to i8*, !dbg !47
  call void @__aser_no_alias__(i8* %17, i8* %22), !dbg !47
  %23 = load %struct.MyStruct2*, %struct.MyStruct2** %4, align 8, !dbg !48
  %24 = getelementptr inbounds %struct.MyStruct2, %struct.MyStruct2* %23, i32 0, i32 0, !dbg !48
  %25 = load i32*, i32** %24, align 8, !dbg !48
  %26 = bitcast i32* %25 to i8*, !dbg !48
  %27 = load %struct.MyStruct1*, %struct.MyStruct1** %3, align 8, !dbg !48
  %28 = getelementptr inbounds %struct.MyStruct1, %struct.MyStruct1* %27, i32 0, i32 1, !dbg !48
  %29 = getelementptr inbounds %struct.MyStruct2, %struct.MyStruct2* %28, i32 0, i32 0, !dbg !48
  %30 = load i32*, i32** %29, align 8, !dbg !48
  %31 = bitcast i32* %30 to i8*, !dbg !48
  call void @__aser_alias__(i8* %26, i8* %31), !dbg !48
  ret i32 0, !dbg !49
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
!1 = !DIFile(filename: "basic_c_tests/struct-nested-1-layer.c", directory: "/home/peiming/Documents/Projects/LLVMRace/TestCases/PTABen")
!2 = !{}
!3 = !{!4}
!4 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!5 = !{i32 2, !"Dwarf Version", i32 4}
!6 = !{i32 2, !"Debug Info Version", i32 3}
!7 = !{i32 1, !"wchar_size", i32 4}
!8 = !{!"clang version 9.0.0 (tags/RELEASE_900/final)"}
!9 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 18, type: !10, scopeLine: 19, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!10 = !DISubroutineType(types: !11)
!11 = !{!12}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DILocalVariable(name: "ms", scope: !9, file: !1, line: 20, type: !14)
!14 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "MyStruct1", file: !1, line: 13, size: 192, elements: !15)
!15 = !{!16, !18}
!16 = !DIDerivedType(tag: DW_TAG_member, name: "f1", scope: !14, file: !1, line: 14, baseType: !17, size: 64)
!17 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!18 = !DIDerivedType(tag: DW_TAG_member, name: "f2", scope: !14, file: !1, line: 15, baseType: !19, size: 128, offset: 64)
!19 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "MyStruct2", file: !1, line: 8, size: 128, elements: !20)
!20 = !{!21, !22}
!21 = !DIDerivedType(tag: DW_TAG_member, name: "f3", scope: !19, file: !1, line: 9, baseType: !17, size: 64)
!22 = !DIDerivedType(tag: DW_TAG_member, name: "f4", scope: !19, file: !1, line: 10, baseType: !17, size: 64, offset: 64)
!23 = !DILocation(line: 20, column: 19, scope: !9)
!24 = !DILocalVariable(name: "pms1", scope: !9, file: !1, line: 21, type: !25)
!25 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 64)
!26 = !DILocation(line: 21, column: 20, scope: !9)
!27 = !DILocalVariable(name: "pms2", scope: !9, file: !1, line: 22, type: !28)
!28 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !19, size: 64)
!29 = !DILocation(line: 22, column: 20, scope: !9)
!30 = !DILocalVariable(name: "a", scope: !9, file: !1, line: 23, type: !12)
!31 = !DILocation(line: 23, column: 6, scope: !9)
!32 = !DILocalVariable(name: "b", scope: !9, file: !1, line: 23, type: !12)
!33 = !DILocation(line: 23, column: 9, scope: !9)
!34 = !DILocalVariable(name: "c", scope: !9, file: !1, line: 23, type: !12)
!35 = !DILocation(line: 23, column: 12, scope: !9)
!36 = !DILocation(line: 24, column: 5, scope: !9)
!37 = !DILocation(line: 24, column: 8, scope: !9)
!38 = !DILocation(line: 25, column: 5, scope: !9)
!39 = !DILocation(line: 25, column: 8, scope: !9)
!40 = !DILocation(line: 25, column: 11, scope: !9)
!41 = !DILocation(line: 26, column: 5, scope: !9)
!42 = !DILocation(line: 26, column: 8, scope: !9)
!43 = !DILocation(line: 26, column: 11, scope: !9)
!44 = !DILocation(line: 27, column: 7, scope: !9)
!45 = !DILocation(line: 28, column: 13, scope: !9)
!46 = !DILocation(line: 28, column: 7, scope: !9)
!47 = !DILocation(line: 29, column: 2, scope: !9)
!48 = !DILocation(line: 30, column: 2, scope: !9)
!49 = !DILocation(line: 31, column: 2, scope: !9)

; ModuleID = 'basic_c_tests/heap-linkedlist.c'
source_filename = "basic_c_tests/heap-linkedlist.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.Node = type { i32*, %struct.Node* }

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @malloc_list(%struct.Node*, i32) #0 !dbg !16 {
  %3 = alloca %struct.Node*, align 8
  %4 = alloca i32, align 4
  %5 = alloca i32*, align 8
  %6 = alloca i32*, align 8
  %7 = alloca %struct.Node*, align 8
  store %struct.Node* %0, %struct.Node** %3, align 8
  call void @llvm.dbg.declare(metadata %struct.Node** %3, metadata !19, metadata !DIExpression()), !dbg !20
  store i32 %1, i32* %4, align 4
  call void @llvm.dbg.declare(metadata i32* %4, metadata !21, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.declare(metadata i32** %5, metadata !23, metadata !DIExpression()), !dbg !24
  store i32* null, i32** %5, align 8, !dbg !24
  call void @llvm.dbg.declare(metadata i32** %6, metadata !25, metadata !DIExpression()), !dbg !26
  store i32* null, i32** %6, align 8, !dbg !26
  call void @llvm.dbg.declare(metadata %struct.Node** %7, metadata !27, metadata !DIExpression()), !dbg !28
  store %struct.Node* null, %struct.Node** %7, align 8, !dbg !28
  %8 = load %struct.Node*, %struct.Node** %3, align 8, !dbg !29
  %9 = getelementptr inbounds %struct.Node, %struct.Node* %8, i32 0, i32 0, !dbg !30
  store i32* null, i32** %9, align 8, !dbg !31
  br label %10, !dbg !32

10:                                               ; preds = %13, %2
  %11 = load i32, i32* %4, align 4, !dbg !33
  %12 = icmp ne i32 %11, 0, !dbg !34
  br i1 %12, label %13, label %31, !dbg !32

13:                                               ; preds = %10
  %14 = call noalias i8* @malloc(i64 4) #4, !dbg !35
  %15 = bitcast i8* %14 to i32*, !dbg !37
  %16 = load %struct.Node*, %struct.Node** %3, align 8, !dbg !38
  %17 = getelementptr inbounds %struct.Node, %struct.Node* %16, i32 0, i32 0, !dbg !39
  store i32* %15, i32** %17, align 8, !dbg !40
  %18 = load %struct.Node*, %struct.Node** %3, align 8, !dbg !41
  %19 = getelementptr inbounds %struct.Node, %struct.Node* %18, i32 0, i32 0, !dbg !42
  %20 = load i32*, i32** %19, align 8, !dbg !42
  store i32* %20, i32** %5, align 8, !dbg !43
  %21 = call noalias i8* @malloc(i64 16) #4, !dbg !44
  %22 = bitcast i8* %21 to %struct.Node*, !dbg !45
  store %struct.Node* %22, %struct.Node** %7, align 8, !dbg !46
  %23 = load %struct.Node*, %struct.Node** %7, align 8, !dbg !47
  %24 = load %struct.Node*, %struct.Node** %3, align 8, !dbg !48
  %25 = getelementptr inbounds %struct.Node, %struct.Node* %24, i32 0, i32 1, !dbg !49
  store %struct.Node* %23, %struct.Node** %25, align 8, !dbg !50
  %26 = load %struct.Node*, %struct.Node** %3, align 8, !dbg !51
  %27 = getelementptr inbounds %struct.Node, %struct.Node* %26, i32 0, i32 0, !dbg !52
  %28 = load i32*, i32** %27, align 8, !dbg !52
  store i32* %28, i32** %6, align 8, !dbg !53
  %29 = load i32, i32* %4, align 4, !dbg !54
  %30 = add nsw i32 %29, -1, !dbg !54
  store i32 %30, i32* %4, align 4, !dbg !54
  br label %10, !dbg !32, !llvm.loop !55

31:                                               ; preds = %10
  %32 = load i32*, i32** %5, align 8, !dbg !57
  %33 = bitcast i32* %32 to i8*, !dbg !57
  %34 = load i32*, i32** %6, align 8, !dbg !57
  %35 = bitcast i32* %34 to i8*, !dbg !57
  call void @__aser_alias__(i8* %33, i8* %35), !dbg !57
  %36 = load %struct.Node*, %struct.Node** %7, align 8, !dbg !58
  %37 = bitcast %struct.Node* %36 to i8*, !dbg !58
  %38 = load i32*, i32** %5, align 8, !dbg !58
  %39 = bitcast i32* %38 to i8*, !dbg !58
  call void @__aser_no_alias__(i8* %37, i8* %39), !dbg !58
  ret void, !dbg !59
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind
declare dso_local noalias i8* @malloc(i64) #2

declare dso_local void @__aser_alias__(i8*, i8*) #3

declare dso_local void @__aser_no_alias__(i8*, i8*) #3

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !60 {
  %1 = alloca i32, align 4
  %2 = alloca %struct.Node*, align 8
  %3 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  call void @llvm.dbg.declare(metadata %struct.Node** %2, metadata !63, metadata !DIExpression()), !dbg !64
  %4 = call noalias i8* @malloc(i64 16) #4, !dbg !65
  %5 = bitcast i8* %4 to %struct.Node*, !dbg !66
  store %struct.Node* %5, %struct.Node** %2, align 8, !dbg !64
  call void @llvm.dbg.declare(metadata i32* %3, metadata !67, metadata !DIExpression()), !dbg !68
  store i32 4, i32* %3, align 4, !dbg !68
  %6 = load %struct.Node*, %struct.Node** %2, align 8, !dbg !69
  %7 = load i32, i32* %3, align 4, !dbg !70
  call void @malloc_list(%struct.Node* %6, i32 %7), !dbg !71
  %8 = load %struct.Node*, %struct.Node** %2, align 8, !dbg !72
  %9 = getelementptr inbounds %struct.Node, %struct.Node* %8, i32 0, i32 1, !dbg !72
  %10 = load %struct.Node*, %struct.Node** %9, align 8, !dbg !72
  %11 = getelementptr inbounds %struct.Node, %struct.Node* %10, i32 0, i32 0, !dbg !72
  %12 = load i32*, i32** %11, align 8, !dbg !72
  %13 = bitcast i32* %12 to i8*, !dbg !72
  %14 = load %struct.Node*, %struct.Node** %2, align 8, !dbg !72
  %15 = getelementptr inbounds %struct.Node, %struct.Node* %14, i32 0, i32 1, !dbg !72
  %16 = load %struct.Node*, %struct.Node** %15, align 8, !dbg !72
  %17 = getelementptr inbounds %struct.Node, %struct.Node* %16, i32 0, i32 1, !dbg !72
  %18 = load %struct.Node*, %struct.Node** %17, align 8, !dbg !72
  %19 = bitcast %struct.Node* %18 to i8*, !dbg !72
  call void @__aser_no_alias__(i8* %13, i8* %19), !dbg !72
  ret i32 0, !dbg !73
}

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!12, !13, !14}
!llvm.ident = !{!15}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 9.0.0 (tags/RELEASE_900/final)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, nameTableKind: None)
!1 = !DIFile(filename: "basic_c_tests/heap-linkedlist.c", directory: "/home/peiming/Documents/Projects/LLVMRace/TestCases/PTABen")
!2 = !{}
!3 = !{!4, !6, !11}
!4 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !5, size: 64)
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64)
!7 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Node", file: !1, line: 11, size: 128, elements: !8)
!8 = !{!9, !10}
!9 = !DIDerivedType(tag: DW_TAG_member, name: "data", scope: !7, file: !1, line: 12, baseType: !4, size: 64)
!10 = !DIDerivedType(tag: DW_TAG_member, name: "next", scope: !7, file: !1, line: 13, baseType: !6, size: 64, offset: 64)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!12 = !{i32 2, !"Dwarf Version", i32 4}
!13 = !{i32 2, !"Debug Info Version", i32 3}
!14 = !{i32 1, !"wchar_size", i32 4}
!15 = !{!"clang version 9.0.0 (tags/RELEASE_900/final)"}
!16 = distinct !DISubprogram(name: "malloc_list", scope: !1, file: !1, line: 17, type: !17, scopeLine: 17, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!17 = !DISubroutineType(types: !18)
!18 = !{null, !6, !5}
!19 = !DILocalVariable(name: "p", arg: 1, scope: !16, file: !1, line: 17, type: !6)
!20 = !DILocation(line: 17, column: 31, scope: !16)
!21 = !DILocalVariable(name: "num", arg: 2, scope: !16, file: !1, line: 17, type: !5)
!22 = !DILocation(line: 17, column: 38, scope: !16)
!23 = !DILocalVariable(name: "p_data1", scope: !16, file: !1, line: 18, type: !4)
!24 = !DILocation(line: 18, column: 7, scope: !16)
!25 = !DILocalVariable(name: "p_data2", scope: !16, file: !1, line: 18, type: !4)
!26 = !DILocation(line: 18, column: 22, scope: !16)
!27 = !DILocalVariable(name: "p_next", scope: !16, file: !1, line: 19, type: !6)
!28 = !DILocation(line: 19, column: 15, scope: !16)
!29 = !DILocation(line: 20, column: 5, scope: !16)
!30 = !DILocation(line: 20, column: 8, scope: !16)
!31 = !DILocation(line: 20, column: 13, scope: !16)
!32 = !DILocation(line: 21, column: 2, scope: !16)
!33 = !DILocation(line: 21, column: 9, scope: !16)
!34 = !DILocation(line: 21, column: 12, scope: !16)
!35 = !DILocation(line: 22, column: 21, scope: !36)
!36 = distinct !DILexicalBlock(scope: !16, file: !1, line: 21, column: 17)
!37 = !DILocation(line: 22, column: 13, scope: !36)
!38 = !DILocation(line: 22, column: 3, scope: !36)
!39 = !DILocation(line: 22, column: 6, scope: !36)
!40 = !DILocation(line: 22, column: 11, scope: !36)
!41 = !DILocation(line: 23, column: 13, scope: !36)
!42 = !DILocation(line: 23, column: 16, scope: !36)
!43 = !DILocation(line: 23, column: 11, scope: !36)
!44 = !DILocation(line: 24, column: 30, scope: !36)
!45 = !DILocation(line: 24, column: 15, scope: !36)
!46 = !DILocation(line: 24, column: 13, scope: !36)
!47 = !DILocation(line: 25, column: 13, scope: !36)
!48 = !DILocation(line: 25, column: 3, scope: !36)
!49 = !DILocation(line: 25, column: 6, scope: !36)
!50 = !DILocation(line: 25, column: 11, scope: !36)
!51 = !DILocation(line: 26, column: 13, scope: !36)
!52 = !DILocation(line: 26, column: 16, scope: !36)
!53 = !DILocation(line: 26, column: 11, scope: !36)
!54 = !DILocation(line: 27, column: 12, scope: !36)
!55 = distinct !{!55, !32, !56}
!56 = !DILocation(line: 28, column: 2, scope: !16)
!57 = !DILocation(line: 29, column: 2, scope: !16)
!58 = !DILocation(line: 30, column: 2, scope: !16)
!59 = !DILocation(line: 31, column: 1, scope: !16)
!60 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 33, type: !61, scopeLine: 33, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!61 = !DISubroutineType(types: !62)
!62 = !{!5}
!63 = !DILocalVariable(name: "head", scope: !60, file: !1, line: 34, type: !6)
!64 = !DILocation(line: 34, column: 15, scope: !60)
!65 = !DILocation(line: 34, column: 37, scope: !60)
!66 = !DILocation(line: 34, column: 22, scope: !60)
!67 = !DILocalVariable(name: "num", scope: !60, file: !1, line: 35, type: !5)
!68 = !DILocation(line: 35, column: 6, scope: !60)
!69 = !DILocation(line: 36, column: 14, scope: !60)
!70 = !DILocation(line: 36, column: 20, scope: !60)
!71 = !DILocation(line: 36, column: 2, scope: !60)
!72 = !DILocation(line: 37, column: 2, scope: !60)
!73 = !DILocation(line: 38, column: 2, scope: !60)

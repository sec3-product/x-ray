; ModuleID = 'basic_c_tests/spec-gap.c'
source_filename = "basic_c_tests/spec-gap.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.TypHeader = type { %struct.TypHeader** }

@HdFree = common dso_local global %struct.TypHeader* null, align 8, !dbg !0
@FirstBag = common dso_local global %struct.TypHeader** null, align 8, !dbg !20
@FreeHandle = common dso_local global %struct.TypHeader* null, align 8, !dbg !18

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @IntComm() #0 !dbg !26 {
  ret void, !dbg !29
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local %struct.TypHeader* @NewBag() #0 !dbg !30 {
  %1 = alloca i64, align 8
  %2 = alloca %struct.TypHeader**, align 8
  %3 = alloca %struct.TypHeader**, align 8
  %4 = alloca %struct.TypHeader**, align 8
  %5 = alloca %struct.TypHeader*, align 8
  call void @llvm.dbg.declare(metadata i64* %1, metadata !33, metadata !DIExpression()), !dbg !35
  call void @llvm.dbg.declare(metadata %struct.TypHeader*** %2, metadata !36, metadata !DIExpression()), !dbg !37
  call void @llvm.dbg.declare(metadata %struct.TypHeader*** %3, metadata !38, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata %struct.TypHeader*** %4, metadata !40, metadata !DIExpression()), !dbg !41
  call void @llvm.dbg.declare(metadata %struct.TypHeader** %5, metadata !42, metadata !DIExpression()), !dbg !43
  %6 = load %struct.TypHeader*, %struct.TypHeader** @HdFree, align 8, !dbg !44
  %7 = getelementptr inbounds %struct.TypHeader, %struct.TypHeader* %6, i32 0, i32 0, !dbg !45
  %8 = load %struct.TypHeader**, %struct.TypHeader*** %7, align 8, !dbg !45
  %9 = load i64, i64* %1, align 8, !dbg !46
  %10 = udiv i64 %9, 8, !dbg !47
  %11 = getelementptr inbounds %struct.TypHeader*, %struct.TypHeader** %8, i64 %10, !dbg !48
  %12 = getelementptr inbounds %struct.TypHeader*, %struct.TypHeader** %11, i64 -1, !dbg !49
  store %struct.TypHeader** %12, %struct.TypHeader*** %2, align 8, !dbg !50
  %13 = load %struct.TypHeader*, %struct.TypHeader** @HdFree, align 8, !dbg !51
  %14 = getelementptr inbounds %struct.TypHeader, %struct.TypHeader* %13, i32 0, i32 0, !dbg !52
  %15 = load %struct.TypHeader**, %struct.TypHeader*** %14, align 8, !dbg !52
  %16 = getelementptr inbounds %struct.TypHeader*, %struct.TypHeader** %15, i64 -1, !dbg !53
  store %struct.TypHeader** %16, %struct.TypHeader*** %3, align 8, !dbg !54
  %17 = load %struct.TypHeader**, %struct.TypHeader*** @FirstBag, align 8, !dbg !55
  %18 = getelementptr inbounds %struct.TypHeader*, %struct.TypHeader** %17, i64 -1, !dbg !56
  store %struct.TypHeader** %18, %struct.TypHeader*** %4, align 8, !dbg !57
  br label %19, !dbg !58

19:                                               ; preds = %23, %0
  %20 = load %struct.TypHeader**, %struct.TypHeader*** %4, align 8, !dbg !59
  %21 = load %struct.TypHeader**, %struct.TypHeader*** %3, align 8, !dbg !60
  %22 = icmp ule %struct.TypHeader** %20, %21, !dbg !61
  br i1 %22, label %23, label %29, !dbg !58

23:                                               ; preds = %19
  %24 = load %struct.TypHeader**, %struct.TypHeader*** %3, align 8, !dbg !62
  %25 = getelementptr inbounds %struct.TypHeader*, %struct.TypHeader** %24, i32 -1, !dbg !62
  store %struct.TypHeader** %25, %struct.TypHeader*** %3, align 8, !dbg !62
  %26 = load %struct.TypHeader*, %struct.TypHeader** %24, align 8, !dbg !63
  %27 = load %struct.TypHeader**, %struct.TypHeader*** %2, align 8, !dbg !64
  %28 = getelementptr inbounds %struct.TypHeader*, %struct.TypHeader** %27, i32 -1, !dbg !64
  store %struct.TypHeader** %28, %struct.TypHeader*** %2, align 8, !dbg !64
  store %struct.TypHeader* %26, %struct.TypHeader** %27, align 8, !dbg !65
  br label %19, !dbg !58, !llvm.loop !66

29:                                               ; preds = %19
  %30 = load %struct.TypHeader**, %struct.TypHeader*** %3, align 8, !dbg !67
  %31 = load %struct.TypHeader*, %struct.TypHeader** %30, align 8, !dbg !68
  %32 = call i32 (%struct.TypHeader*, void (...)*, ...) bitcast (i32 (...)* @EXPECTEDFAIL_NOALIAS to i32 (%struct.TypHeader*, void (...)*, ...)*)(%struct.TypHeader* %31, void (...)* bitcast (void ()* @IntComm to void (...)*)), !dbg !69
  %33 = load %struct.TypHeader*, %struct.TypHeader** @HdFree, align 8, !dbg !70
  store %struct.TypHeader* %33, %struct.TypHeader** %5, align 8, !dbg !72
  br label %34, !dbg !73

34:                                               ; preds = %47, %29
  %35 = load %struct.TypHeader*, %struct.TypHeader** %5, align 8, !dbg !74
  %36 = load %struct.TypHeader**, %struct.TypHeader*** @FirstBag, align 8, !dbg !76
  %37 = getelementptr inbounds %struct.TypHeader*, %struct.TypHeader** %36, i64 -1, !dbg !77
  %38 = bitcast %struct.TypHeader** %37 to %struct.TypHeader*, !dbg !78
  %39 = icmp ult %struct.TypHeader* %35, %38, !dbg !79
  br i1 %39, label %40, label %50, !dbg !80

40:                                               ; preds = %34
  %41 = load i64, i64* %1, align 8, !dbg !81
  %42 = udiv i64 %41, 8, !dbg !82
  %43 = load %struct.TypHeader*, %struct.TypHeader** %5, align 8, !dbg !83
  %44 = getelementptr inbounds %struct.TypHeader, %struct.TypHeader* %43, i32 0, i32 0, !dbg !84
  %45 = load %struct.TypHeader**, %struct.TypHeader*** %44, align 8, !dbg !85
  %46 = getelementptr inbounds %struct.TypHeader*, %struct.TypHeader** %45, i64 %42, !dbg !85
  store %struct.TypHeader** %46, %struct.TypHeader*** %44, align 8, !dbg !85
  br label %47, !dbg !83

47:                                               ; preds = %40
  %48 = load %struct.TypHeader*, %struct.TypHeader** %5, align 8, !dbg !86
  %49 = getelementptr inbounds %struct.TypHeader, %struct.TypHeader* %48, i32 1, !dbg !86
  store %struct.TypHeader* %49, %struct.TypHeader** %5, align 8, !dbg !86
  br label %34, !dbg !87, !llvm.loop !88

50:                                               ; preds = %34
  %51 = load %struct.TypHeader*, %struct.TypHeader** @FreeHandle, align 8, !dbg !90
  %52 = bitcast %struct.TypHeader* %51 to %struct.TypHeader**, !dbg !91
  %53 = load %struct.TypHeader*, %struct.TypHeader** %5, align 8, !dbg !92
  %54 = getelementptr inbounds %struct.TypHeader, %struct.TypHeader* %53, i32 0, i32 0, !dbg !93
  store %struct.TypHeader** %52, %struct.TypHeader*** %54, align 8, !dbg !94
  %55 = load %struct.TypHeader*, %struct.TypHeader** %5, align 8, !dbg !95
  store %struct.TypHeader* %55, %struct.TypHeader** @FreeHandle, align 8, !dbg !96
  %56 = load %struct.TypHeader*, %struct.TypHeader** @FreeHandle, align 8, !dbg !97
  ret %struct.TypHeader* %56, !dbg !98
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare dso_local i32 @EXPECTEDFAIL_NOALIAS(...) #2

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @InstIntFunc(void (...)*) #0 !dbg !99 {
  %2 = alloca void (...)*, align 8
  %3 = alloca %struct.TypHeader*, align 8
  store void (...)* %0, void (...)** %2, align 8
  call void @llvm.dbg.declare(metadata void (...)** %2, metadata !102, metadata !DIExpression()), !dbg !103
  call void @llvm.dbg.declare(metadata %struct.TypHeader** %3, metadata !104, metadata !DIExpression()), !dbg !105
  %4 = call %struct.TypHeader* @NewBag(), !dbg !106
  store %struct.TypHeader* %4, %struct.TypHeader** %3, align 8, !dbg !105
  %5 = load void (...)*, void (...)** %2, align 8, !dbg !107
  %6 = load %struct.TypHeader*, %struct.TypHeader** %3, align 8, !dbg !108
  %7 = getelementptr inbounds %struct.TypHeader, %struct.TypHeader* %6, i32 0, i32 0, !dbg !109
  %8 = load %struct.TypHeader**, %struct.TypHeader*** %7, align 8, !dbg !109
  %9 = bitcast %struct.TypHeader** %8 to void (...)**, !dbg !110
  store void (...)* %5, void (...)** %9, align 8, !dbg !111
  ret void, !dbg !112
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @InitGasman() #0 !dbg !113 {
  %1 = alloca i64, align 8
  call void @llvm.dbg.declare(metadata i64* %1, metadata !114, metadata !DIExpression()), !dbg !115
  %2 = load i64, i64* %1, align 8, !dbg !116
  %3 = call i8* @SyGetmem(i64 %2), !dbg !117
  %4 = bitcast i8* %3 to %struct.TypHeader*, !dbg !118
  store %struct.TypHeader* %4, %struct.TypHeader** @HdFree, align 8, !dbg !119
  %5 = load %struct.TypHeader*, %struct.TypHeader** @FreeHandle, align 8, !dbg !120
  %6 = getelementptr inbounds %struct.TypHeader, %struct.TypHeader* %5, i32 0, i32 0, !dbg !121
  %7 = load %struct.TypHeader**, %struct.TypHeader*** %6, align 8, !dbg !121
  %8 = bitcast %struct.TypHeader** %7 to %struct.TypHeader*, !dbg !122
  store %struct.TypHeader* %8, %struct.TypHeader** @FreeHandle, align 8, !dbg !123
  ret void, !dbg !124
}

declare dso_local i8* @SyGetmem(i64) #2

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !125 {
  %1 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  call void @InitGasman(), !dbg !129
  call void @InstIntFunc(void (...)* bitcast (void ()* @IntComm to void (...)*)), !dbg !130
  ret i32 0, !dbg !131
}

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!22, !23, !24}
!llvm.ident = !{!25}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "HdFree", scope: !2, file: !3, line: 15, type: !7, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 9.0.0 (tags/RELEASE_900/final)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !5, globals: !17, nameTableKind: None)
!3 = !DIFile(filename: "basic_c_tests/spec-gap.c", directory: "/home/peiming/Documents/Projects/LLVMRace/TestCases/PTABen")
!4 = !{}
!5 = !{!6, !7, !13}
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64)
!7 = !DIDerivedType(tag: DW_TAG_typedef, name: "TypHandle", file: !3, line: 13, baseType: !8)
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 64)
!9 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "TypHeader", file: !3, line: 11, size: 64, elements: !10)
!10 = !{!11}
!11 = !DIDerivedType(tag: DW_TAG_member, name: "ptr", scope: !9, file: !3, line: 12, baseType: !12, size: 64)
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 64)
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 64)
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !15, size: 64)
!15 = !DISubroutineType(types: !16)
!16 = !{null, null}
!17 = !{!0, !18, !20}
!18 = !DIGlobalVariableExpression(var: !19, expr: !DIExpression())
!19 = distinct !DIGlobalVariable(name: "FreeHandle", scope: !2, file: !3, line: 16, type: !7, isLocal: false, isDefinition: true)
!20 = !DIGlobalVariableExpression(var: !21, expr: !DIExpression())
!21 = distinct !DIGlobalVariable(name: "FirstBag", scope: !2, file: !3, line: 17, type: !6, isLocal: false, isDefinition: true)
!22 = !{i32 2, !"Dwarf Version", i32 4}
!23 = !{i32 2, !"Debug Info Version", i32 3}
!24 = !{i32 1, !"wchar_size", i32 4}
!25 = !{!"clang version 9.0.0 (tags/RELEASE_900/final)"}
!26 = distinct !DISubprogram(name: "IntComm", scope: !3, file: !3, line: 19, type: !27, scopeLine: 19, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!27 = !DISubroutineType(types: !28)
!28 = !{null}
!29 = !DILocation(line: 19, column: 18, scope: !26)
!30 = distinct !DISubprogram(name: "NewBag", scope: !3, file: !3, line: 21, type: !31, scopeLine: 21, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!31 = !DISubroutineType(types: !32)
!32 = !{!7}
!33 = !DILocalVariable(name: "needed", scope: !30, file: !3, line: 22, type: !34)
!34 = !DIBasicType(name: "long int", size: 64, encoding: DW_ATE_signed)
!35 = !DILocation(line: 22, column: 7, scope: !30)
!36 = !DILocalVariable(name: "d", scope: !30, file: !3, line: 23, type: !6)
!37 = !DILocation(line: 23, column: 13, scope: !30)
!38 = !DILocalVariable(name: "s", scope: !30, file: !3, line: 23, type: !6)
!39 = !DILocation(line: 23, column: 17, scope: !30)
!40 = !DILocalVariable(name: "e", scope: !30, file: !3, line: 23, type: !6)
!41 = !DILocation(line: 23, column: 21, scope: !30)
!42 = !DILocalVariable(name: "h", scope: !30, file: !3, line: 24, type: !7)
!43 = !DILocation(line: 24, column: 12, scope: !30)
!44 = !DILocation(line: 26, column: 21, scope: !30)
!45 = !DILocation(line: 26, column: 30, scope: !30)
!46 = !DILocation(line: 26, column: 38, scope: !30)
!47 = !DILocation(line: 26, column: 45, scope: !30)
!48 = !DILocation(line: 26, column: 36, scope: !30)
!49 = !DILocation(line: 26, column: 67, scope: !30)
!50 = !DILocation(line: 26, column: 4, scope: !30)
!51 = !DILocation(line: 27, column: 21, scope: !30)
!52 = !DILocation(line: 27, column: 30, scope: !30)
!53 = !DILocation(line: 27, column: 36, scope: !30)
!54 = !DILocation(line: 27, column: 4, scope: !30)
!55 = !DILocation(line: 28, column: 7, scope: !30)
!56 = !DILocation(line: 28, column: 15, scope: !30)
!57 = !DILocation(line: 28, column: 4, scope: !30)
!58 = !DILocation(line: 29, column: 2, scope: !30)
!59 = !DILocation(line: 29, column: 10, scope: !30)
!60 = !DILocation(line: 29, column: 15, scope: !30)
!61 = !DILocation(line: 29, column: 12, scope: !30)
!62 = !DILocation(line: 29, column: 29, scope: !30)
!63 = !DILocation(line: 29, column: 27, scope: !30)
!64 = !DILocation(line: 29, column: 22, scope: !30)
!65 = !DILocation(line: 29, column: 25, scope: !30)
!66 = distinct !{!66, !58, !62}
!67 = !DILocation(line: 30, column: 24, scope: !30)
!68 = !DILocation(line: 30, column: 23, scope: !30)
!69 = !DILocation(line: 30, column: 2, scope: !30)
!70 = !DILocation(line: 32, column: 9, scope: !71)
!71 = distinct !DILexicalBlock(scope: !30, file: !3, line: 32, column: 2)
!72 = !DILocation(line: 32, column: 8, scope: !71)
!73 = !DILocation(line: 32, column: 7, scope: !71)
!74 = !DILocation(line: 32, column: 17, scope: !75)
!75 = distinct !DILexicalBlock(scope: !71, file: !3, line: 32, column: 2)
!76 = !DILocation(line: 32, column: 33, scope: !75)
!77 = !DILocation(line: 32, column: 41, scope: !75)
!78 = !DILocation(line: 32, column: 21, scope: !75)
!79 = !DILocation(line: 32, column: 19, scope: !75)
!80 = !DILocation(line: 32, column: 2, scope: !71)
!81 = !DILocation(line: 33, column: 13, scope: !75)
!82 = !DILocation(line: 33, column: 20, scope: !75)
!83 = !DILocation(line: 33, column: 3, scope: !75)
!84 = !DILocation(line: 33, column: 6, scope: !75)
!85 = !DILocation(line: 33, column: 10, scope: !75)
!86 = !DILocation(line: 32, column: 46, scope: !75)
!87 = !DILocation(line: 32, column: 2, scope: !75)
!88 = distinct !{!88, !80, !89}
!89 = !DILocation(line: 33, column: 40, scope: !71)
!90 = !DILocation(line: 35, column: 23, scope: !30)
!91 = !DILocation(line: 35, column: 11, scope: !30)
!92 = !DILocation(line: 35, column: 2, scope: !30)
!93 = !DILocation(line: 35, column: 5, scope: !30)
!94 = !DILocation(line: 35, column: 9, scope: !30)
!95 = !DILocation(line: 36, column: 15, scope: !30)
!96 = !DILocation(line: 36, column: 13, scope: !30)
!97 = !DILocation(line: 38, column: 9, scope: !30)
!98 = !DILocation(line: 38, column: 2, scope: !30)
!99 = distinct !DISubprogram(name: "InstIntFunc", scope: !3, file: !3, line: 41, type: !100, scopeLine: 41, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!100 = !DISubroutineType(types: !101)
!101 = !{null, !14}
!102 = !DILocalVariable(name: "func", arg: 1, scope: !99, file: !3, line: 41, type: !14)
!103 = !DILocation(line: 41, column: 25, scope: !99)
!104 = !DILocalVariable(name: "hdDef", scope: !99, file: !3, line: 42, type: !7)
!105 = !DILocation(line: 42, column: 12, scope: !99)
!106 = !DILocation(line: 42, column: 20, scope: !99)
!107 = !DILocation(line: 43, column: 46, scope: !99)
!108 = !DILocation(line: 43, column: 30, scope: !99)
!109 = !DILocation(line: 43, column: 38, scope: !99)
!110 = !DILocation(line: 43, column: 2, scope: !99)
!111 = !DILocation(line: 43, column: 44, scope: !99)
!112 = !DILocation(line: 44, column: 1, scope: !99)
!113 = distinct !DISubprogram(name: "InitGasman", scope: !3, file: !3, line: 46, type: !27, scopeLine: 46, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!114 = !DILocalVariable(name: "SyMemory", scope: !113, file: !3, line: 47, type: !34)
!115 = !DILocation(line: 47, column: 7, scope: !113)
!116 = !DILocation(line: 48, column: 32, scope: !113)
!117 = !DILocation(line: 48, column: 22, scope: !113)
!118 = !DILocation(line: 48, column: 11, scope: !113)
!119 = !DILocation(line: 48, column: 9, scope: !113)
!120 = !DILocation(line: 49, column: 41, scope: !113)
!121 = !DILocation(line: 49, column: 54, scope: !113)
!122 = !DILocation(line: 49, column: 15, scope: !113)
!123 = !DILocation(line: 49, column: 13, scope: !113)
!124 = !DILocation(line: 50, column: 1, scope: !113)
!125 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 52, type: !126, scopeLine: 52, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!126 = !DISubroutineType(types: !127)
!127 = !{!128}
!128 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!129 = !DILocation(line: 53, column: 2, scope: !125)
!130 = !DILocation(line: 54, column: 2, scope: !125)
!131 = !DILocation(line: 55, column: 2, scope: !125)

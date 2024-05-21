; ModuleID = 'basic_c_tests/constraint-cycle-pwc.c'
source_filename = "basic_c_tests/constraint-cycle-pwc.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.network = type { %struct.arc*, %struct.arc*, %struct.arc*, %struct.arc* }
%struct.arc = type { %struct.arc*, %struct.arc*, i64 }

@.str = private unnamed_addr constant [13 x i8] c"hello world\0A\00", align 1

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !18 {
  %1 = alloca i32, align 4
  %2 = alloca %struct.network*, align 8
  %3 = alloca %struct.arc*, align 8
  %4 = alloca i8*, align 8
  %5 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  call void @llvm.dbg.declare(metadata %struct.network** %2, metadata !22, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.declare(metadata %struct.arc** %3, metadata !32, metadata !DIExpression()), !dbg !33
  call void @llvm.dbg.declare(metadata i8** %4, metadata !34, metadata !DIExpression()), !dbg !35
  call void @llvm.dbg.declare(metadata i32* %5, metadata !36, metadata !DIExpression()), !dbg !37
  %6 = load %struct.network*, %struct.network** %2, align 8, !dbg !38
  %7 = getelementptr inbounds %struct.network, %struct.network* %6, i32 0, i32 1, !dbg !39
  %8 = load %struct.arc*, %struct.arc** %7, align 8, !dbg !39
  %9 = bitcast %struct.arc* %8 to i8*, !dbg !40
  store i8* %9, i8** %4, align 8, !dbg !41
  %10 = load %struct.network*, %struct.network** %2, align 8, !dbg !42
  %11 = getelementptr inbounds %struct.network, %struct.network* %10, i32 0, i32 0, !dbg !44
  %12 = load %struct.arc*, %struct.arc** %11, align 8, !dbg !44
  store %struct.arc* %12, %struct.arc** %3, align 8, !dbg !45
  br label %13, !dbg !46

13:                                               ; preds = %30, %0
  %14 = load %struct.arc*, %struct.arc** %3, align 8, !dbg !47
  %15 = load i8*, i8** %4, align 8, !dbg !49
  %16 = bitcast i8* %15 to %struct.arc*, !dbg !50
  %17 = icmp ne %struct.arc* %14, %16, !dbg !51
  br i1 %17, label %18, label %33, !dbg !52

18:                                               ; preds = %13
  %19 = load %struct.arc*, %struct.arc** %3, align 8, !dbg !53
  %20 = getelementptr inbounds %struct.arc, %struct.arc* %19, i32 0, i32 2, !dbg !56
  %21 = load i64, i64* %20, align 8, !dbg !56
  %22 = icmp ne i64 %21, 0, !dbg !53
  br i1 %22, label %23, label %29, !dbg !57

23:                                               ; preds = %18
  %24 = load i32, i32* %5, align 4, !dbg !58
  %25 = icmp ne i32 %24, 0, !dbg !58
  br i1 %25, label %26, label %28, !dbg !61

26:                                               ; preds = %23
  %27 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str, i64 0, i64 0)), !dbg !62
  br label %28, !dbg !62

28:                                               ; preds = %26, %23
  br label %29, !dbg !63

29:                                               ; preds = %28, %18
  br label %30, !dbg !64

30:                                               ; preds = %29
  %31 = load %struct.arc*, %struct.arc** %3, align 8, !dbg !65
  %32 = getelementptr inbounds %struct.arc, %struct.arc* %31, i32 1, !dbg !65
  store %struct.arc* %32, %struct.arc** %3, align 8, !dbg !65
  br label %13, !dbg !66, !llvm.loop !67

33:                                               ; preds = %13
  ret i32 0, !dbg !69
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare dso_local i32 @printf(i8*, ...) #2

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!14, !15, !16}
!llvm.ident = !{!17}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 9.0.0 (tags/RELEASE_900/final)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, nameTableKind: None)
!1 = !DIFile(filename: "basic_c_tests/constraint-cycle-pwc.c", directory: "/home/peiming/Documents/Projects/LLVMRace/TestCases/PTABen")
!2 = !{}
!3 = !{!4, !5}
!4 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!5 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !6, size: 64)
!6 = !DIDerivedType(tag: DW_TAG_typedef, name: "arc_t", file: !1, line: 13, baseType: !7)
!7 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "arc", file: !1, line: 9, size: 192, elements: !8)
!8 = !{!9, !11, !12}
!9 = !DIDerivedType(tag: DW_TAG_member, name: "nextout", scope: !7, file: !1, line: 11, baseType: !10, size: 64)
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64)
!11 = !DIDerivedType(tag: DW_TAG_member, name: "nextin", scope: !7, file: !1, line: 11, baseType: !10, size: 64, offset: 64)
!12 = !DIDerivedType(tag: DW_TAG_member, name: "ident", scope: !7, file: !1, line: 12, baseType: !13, size: 64, offset: 128)
!13 = !DIBasicType(name: "long int", size: 64, encoding: DW_ATE_signed)
!14 = !{i32 2, !"Dwarf Version", i32 4}
!15 = !{i32 2, !"Debug Info Version", i32 3}
!16 = !{i32 1, !"wchar_size", i32 4}
!17 = !{!"clang version 9.0.0 (tags/RELEASE_900/final)"}
!18 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 22, type: !19, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!19 = !DISubroutineType(types: !20)
!20 = !{!21}
!21 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!22 = !DILocalVariable(name: "net", scope: !18, file: !1, line: 24, type: !23)
!23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !24, size: 64)
!24 = !DIDerivedType(tag: DW_TAG_typedef, name: "network_t", file: !1, line: 20, baseType: !25)
!25 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "network", file: !1, line: 16, size: 256, elements: !26)
!26 = !{!27, !28, !29, !30}
!27 = !DIDerivedType(tag: DW_TAG_member, name: "arcs", scope: !25, file: !1, line: 18, baseType: !5, size: 64)
!28 = !DIDerivedType(tag: DW_TAG_member, name: "stop_arcs", scope: !25, file: !1, line: 18, baseType: !5, size: 64, offset: 64)
!29 = !DIDerivedType(tag: DW_TAG_member, name: "dummy_arcs", scope: !25, file: !1, line: 19, baseType: !5, size: 64, offset: 128)
!30 = !DIDerivedType(tag: DW_TAG_member, name: "stop_dummy", scope: !25, file: !1, line: 19, baseType: !5, size: 64, offset: 192)
!31 = !DILocation(line: 24, column: 13, scope: !18)
!32 = !DILocalVariable(name: "arc", scope: !18, file: !1, line: 25, type: !5)
!33 = !DILocation(line: 25, column: 9, scope: !18)
!34 = !DILocalVariable(name: "stop", scope: !18, file: !1, line: 26, type: !4)
!35 = !DILocation(line: 26, column: 8, scope: !18)
!36 = !DILocalVariable(name: "condition", scope: !18, file: !1, line: 27, type: !21)
!37 = !DILocation(line: 27, column: 6, scope: !18)
!38 = !DILocation(line: 29, column: 17, scope: !18)
!39 = !DILocation(line: 29, column: 22, scope: !18)
!40 = !DILocation(line: 29, column: 9, scope: !18)
!41 = !DILocation(line: 29, column: 7, scope: !18)
!42 = !DILocation(line: 30, column: 13, scope: !43)
!43 = distinct !DILexicalBlock(scope: !18, file: !1, line: 30, column: 2)
!44 = !DILocation(line: 30, column: 18, scope: !43)
!45 = !DILocation(line: 30, column: 11, scope: !43)
!46 = !DILocation(line: 30, column: 7, scope: !43)
!47 = !DILocation(line: 30, column: 24, scope: !48)
!48 = distinct !DILexicalBlock(scope: !43, file: !1, line: 30, column: 2)
!49 = !DILocation(line: 30, column: 40, scope: !48)
!50 = !DILocation(line: 30, column: 31, scope: !48)
!51 = !DILocation(line: 30, column: 28, scope: !48)
!52 = !DILocation(line: 30, column: 2, scope: !43)
!53 = !DILocation(line: 31, column: 7, scope: !54)
!54 = distinct !DILexicalBlock(scope: !55, file: !1, line: 31, column: 7)
!55 = distinct !DILexicalBlock(scope: !48, file: !1, line: 30, column: 54)
!56 = !DILocation(line: 31, column: 12, scope: !54)
!57 = !DILocation(line: 31, column: 7, scope: !55)
!58 = !DILocation(line: 32, column: 8, scope: !59)
!59 = distinct !DILexicalBlock(scope: !60, file: !1, line: 32, column: 8)
!60 = distinct !DILexicalBlock(scope: !54, file: !1, line: 31, column: 20)
!61 = !DILocation(line: 32, column: 8, scope: !60)
!62 = !DILocation(line: 33, column: 5, scope: !59)
!63 = !DILocation(line: 34, column: 3, scope: !60)
!64 = !DILocation(line: 35, column: 2, scope: !55)
!65 = !DILocation(line: 30, column: 49, scope: !48)
!66 = !DILocation(line: 30, column: 2, scope: !48)
!67 = distinct !{!67, !52, !68}
!68 = !DILocation(line: 35, column: 2, scope: !43)
!69 = !DILocation(line: 36, column: 2, scope: !18)

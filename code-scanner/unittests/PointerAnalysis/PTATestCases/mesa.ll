; ModuleID = 'basic_c_tests/mesa.c'
source_filename = "basic_c_tests/mesa.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.gl_api_table = type { void (i32*, i32, float)* }

@.str = private unnamed_addr constant [44 x i8] c"found uninitialized function pointer at %d\0A\00", align 1

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @gl_Accum(i32*, i32, float) #0 !dbg !21 {
  %4 = alloca i32*, align 8
  %5 = alloca i32, align 4
  %6 = alloca float, align 4
  store i32* %0, i32** %4, align 8
  call void @llvm.dbg.declare(metadata i32** %4, metadata !22, metadata !DIExpression()), !dbg !23
  store i32 %1, i32* %5, align 4
  call void @llvm.dbg.declare(metadata i32* %5, metadata !24, metadata !DIExpression()), !dbg !25
  store float %2, float* %6, align 4
  call void @llvm.dbg.declare(metadata float* %6, metadata !26, metadata !DIExpression()), !dbg !27
  ret void, !dbg !28
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @check_pointers(%struct.gl_api_table*) #0 !dbg !29 {
  %2 = alloca %struct.gl_api_table*, align 8
  %3 = alloca i8**, align 8
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store %struct.gl_api_table* %0, %struct.gl_api_table** %2, align 8
  call void @llvm.dbg.declare(metadata %struct.gl_api_table** %2, metadata !32, metadata !DIExpression()), !dbg !33
  call void @llvm.dbg.declare(metadata i8*** %3, metadata !34, metadata !DIExpression()), !dbg !35
  call void @llvm.dbg.declare(metadata i32* %4, metadata !36, metadata !DIExpression()), !dbg !37
  store i32 1, i32* %4, align 4, !dbg !37
  call void @llvm.dbg.declare(metadata i32* %5, metadata !38, metadata !DIExpression()), !dbg !39
  %6 = load %struct.gl_api_table*, %struct.gl_api_table** %2, align 8, !dbg !40
  %7 = bitcast %struct.gl_api_table* %6 to i8**, !dbg !41
  store i8** %7, i8*** %3, align 8, !dbg !42
  %8 = load i8**, i8*** %3, align 8, !dbg !43
  %9 = load i8*, i8** %8, align 8, !dbg !43
  call void @__aser_alias__(i8* %9, i8* bitcast (void (i32*, i32, float)* @gl_Accum to i8*)), !dbg !43
  store i32 0, i32* %5, align 4, !dbg !44
  br label %10, !dbg !46

10:                                               ; preds = %30, %1
  %11 = load i32, i32* %5, align 4, !dbg !47
  %12 = load i32, i32* %4, align 4, !dbg !49
  %13 = icmp slt i32 %11, %12, !dbg !50
  br i1 %13, label %14, label %33, !dbg !51

14:                                               ; preds = %10
  %15 = load i8**, i8*** %3, align 8, !dbg !52
  %16 = load i32, i32* %5, align 4, !dbg !52
  %17 = sext i32 %16 to i64, !dbg !52
  %18 = getelementptr inbounds i8*, i8** %15, i64 %17, !dbg !52
  %19 = load i8*, i8** %18, align 8, !dbg !52
  call void @__aser_alias__(i8* %19, i8* bitcast (void (i32*, i32, float)* @gl_Accum to i8*)), !dbg !52
  %20 = load i8**, i8*** %3, align 8, !dbg !54
  %21 = load i32, i32* %5, align 4, !dbg !56
  %22 = sext i32 %21 to i64, !dbg !54
  %23 = getelementptr inbounds i8*, i8** %20, i64 %22, !dbg !54
  %24 = load i8*, i8** %23, align 8, !dbg !54
  %25 = icmp ne i8* %24, null, !dbg !54
  br i1 %25, label %29, label %26, !dbg !57

26:                                               ; preds = %14
  %27 = load i32, i32* %5, align 4, !dbg !58
  %28 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([44 x i8], [44 x i8]* @.str, i64 0, i64 0), i32 %27), !dbg !59
  br label %29, !dbg !59

29:                                               ; preds = %26, %14
  br label %30, !dbg !60

30:                                               ; preds = %29
  %31 = load i32, i32* %5, align 4, !dbg !61
  %32 = add nsw i32 %31, 1, !dbg !61
  store i32 %32, i32* %5, align 4, !dbg !61
  br label %10, !dbg !62, !llvm.loop !63

33:                                               ; preds = %10
  ret void, !dbg !65
}

declare dso_local void @__aser_alias__(i8*, i8*) #2

declare dso_local i32 @printf(i8*, ...) #2

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @init_exec_pointers(%struct.gl_api_table*) #0 !dbg !66 {
  %2 = alloca %struct.gl_api_table*, align 8
  store %struct.gl_api_table* %0, %struct.gl_api_table** %2, align 8
  call void @llvm.dbg.declare(metadata %struct.gl_api_table** %2, metadata !67, metadata !DIExpression()), !dbg !68
  %3 = load %struct.gl_api_table*, %struct.gl_api_table** %2, align 8, !dbg !69
  %4 = getelementptr inbounds %struct.gl_api_table, %struct.gl_api_table* %3, i32 0, i32 0, !dbg !70
  store void (i32*, i32, float)* @gl_Accum, void (i32*, i32, float)** %4, align 8, !dbg !71
  ret void, !dbg !72
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !73 {
  %1 = alloca i32, align 4
  %2 = alloca %struct.gl_api_table*, align 8
  store i32 0, i32* %1, align 4
  call void @llvm.dbg.declare(metadata %struct.gl_api_table** %2, metadata !76, metadata !DIExpression()), !dbg !77
  %3 = call noalias i8* @calloc(i64 1, i64 8) #4, !dbg !78
  %4 = bitcast i8* %3 to %struct.gl_api_table*, !dbg !79
  store %struct.gl_api_table* %4, %struct.gl_api_table** %2, align 8, !dbg !77
  %5 = load %struct.gl_api_table*, %struct.gl_api_table** %2, align 8, !dbg !80
  call void @init_exec_pointers(%struct.gl_api_table* %5), !dbg !81
  %6 = load %struct.gl_api_table*, %struct.gl_api_table** %2, align 8, !dbg !82
  call void @check_pointers(%struct.gl_api_table* %6), !dbg !83
  ret i32 0, !dbg !84
}

; Function Attrs: nounwind
declare dso_local noalias i8* @calloc(i64, i64) #3

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!17, !18, !19}
!llvm.ident = !{!20}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 9.0.0 (tags/RELEASE_900/final)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, nameTableKind: None)
!1 = !DIFile(filename: "basic_c_tests/mesa.c", directory: "/home/peiming/Documents/Projects/LLVMRace/TestCases/PTABen")
!2 = !{}
!3 = !{!4, !5, !6}
!4 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !5, size: 64)
!5 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64)
!7 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "gl_api_table", file: !1, line: 13, size: 64, elements: !8)
!8 = !{!9}
!9 = !DIDerivedType(tag: DW_TAG_member, name: "Accum", scope: !7, file: !1, line: 14, baseType: !10, size: 64)
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 64)
!11 = !DISubroutineType(types: !12)
!12 = !{null, !13, !15, !16}
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 64)
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!15 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!16 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!17 = !{i32 2, !"Dwarf Version", i32 4}
!18 = !{i32 2, !"Debug Info Version", i32 3}
!19 = !{i32 1, !"wchar_size", i32 4}
!20 = !{!"clang version 9.0.0 (tags/RELEASE_900/final)"}
!21 = distinct !DISubprogram(name: "gl_Accum", scope: !1, file: !1, line: 10, type: !11, scopeLine: 11, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!22 = !DILocalVariable(name: "x", arg: 1, scope: !21, file: !1, line: 10, type: !13)
!23 = !DILocation(line: 10, column: 20, scope: !21)
!24 = !DILocalVariable(name: "y", arg: 2, scope: !21, file: !1, line: 10, type: !15)
!25 = !DILocation(line: 10, column: 32, scope: !21)
!26 = !DILocalVariable(name: "z", arg: 3, scope: !21, file: !1, line: 10, type: !16)
!27 = !DILocation(line: 10, column: 41, scope: !21)
!28 = !DILocation(line: 11, column: 2, scope: !21)
!29 = distinct !DISubprogram(name: "check_pointers", scope: !1, file: !1, line: 18, type: !30, scopeLine: 19, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!30 = !DISubroutineType(types: !31)
!31 = !{null, !6}
!32 = !DILocalVariable(name: "table", arg: 1, scope: !29, file: !1, line: 18, type: !6)
!33 = !DILocation(line: 18, column: 42, scope: !29)
!34 = !DILocalVariable(name: "entry", scope: !29, file: !1, line: 20, type: !4)
!35 = !DILocation(line: 20, column: 9, scope: !29)
!36 = !DILocalVariable(name: "numentries", scope: !29, file: !1, line: 21, type: !14)
!37 = !DILocation(line: 21, column: 6, scope: !29)
!38 = !DILocalVariable(name: "i", scope: !29, file: !1, line: 22, type: !14)
!39 = !DILocation(line: 22, column: 6, scope: !29)
!40 = !DILocation(line: 24, column: 20, scope: !29)
!41 = !DILocation(line: 24, column: 10, scope: !29)
!42 = !DILocation(line: 24, column: 8, scope: !29)
!43 = !DILocation(line: 25, column: 2, scope: !29)
!44 = !DILocation(line: 26, column: 8, scope: !45)
!45 = distinct !DILexicalBlock(scope: !29, file: !1, line: 26, column: 2)
!46 = !DILocation(line: 26, column: 7, scope: !45)
!47 = !DILocation(line: 26, column: 11, scope: !48)
!48 = distinct !DILexicalBlock(scope: !45, file: !1, line: 26, column: 2)
!49 = !DILocation(line: 26, column: 13, scope: !48)
!50 = !DILocation(line: 26, column: 12, scope: !48)
!51 = !DILocation(line: 26, column: 2, scope: !45)
!52 = !DILocation(line: 27, column: 6, scope: !53)
!53 = distinct !DILexicalBlock(scope: !48, file: !1, line: 26, column: 29)
!54 = !DILocation(line: 28, column: 8, scope: !55)
!55 = distinct !DILexicalBlock(scope: !53, file: !1, line: 28, column: 7)
!56 = !DILocation(line: 28, column: 14, scope: !55)
!57 = !DILocation(line: 28, column: 7, scope: !53)
!58 = !DILocation(line: 29, column: 59, scope: !55)
!59 = !DILocation(line: 29, column: 4, scope: !55)
!60 = !DILocation(line: 30, column: 2, scope: !53)
!61 = !DILocation(line: 26, column: 25, scope: !48)
!62 = !DILocation(line: 26, column: 2, scope: !48)
!63 = distinct !{!63, !51, !64}
!64 = !DILocation(line: 30, column: 2, scope: !45)
!65 = !DILocation(line: 31, column: 1, scope: !29)
!66 = distinct !DISubprogram(name: "init_exec_pointers", scope: !1, file: !1, line: 34, type: !30, scopeLine: 35, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!67 = !DILocalVariable(name: "table", arg: 1, scope: !66, file: !1, line: 34, type: !6)
!68 = !DILocation(line: 34, column: 47, scope: !66)
!69 = !DILocation(line: 36, column: 4, scope: !66)
!70 = !DILocation(line: 36, column: 11, scope: !66)
!71 = !DILocation(line: 36, column: 17, scope: !66)
!72 = !DILocation(line: 37, column: 1, scope: !66)
!73 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 39, type: !74, scopeLine: 40, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!74 = !DISubroutineType(types: !75)
!75 = !{!14}
!76 = !DILocalVariable(name: "table", scope: !73, file: !1, line: 41, type: !6)
!77 = !DILocation(line: 41, column: 23, scope: !73)
!78 = !DILocation(line: 41, column: 53, scope: !73)
!79 = !DILocation(line: 41, column: 31, scope: !73)
!80 = !DILocation(line: 42, column: 22, scope: !73)
!81 = !DILocation(line: 42, column: 2, scope: !73)
!82 = !DILocation(line: 43, column: 18, scope: !73)
!83 = !DILocation(line: 43, column: 2, scope: !73)
!84 = !DILocation(line: 44, column: 2, scope: !73)

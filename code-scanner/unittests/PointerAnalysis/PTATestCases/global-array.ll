; ModuleID = 'basic_c_tests/global-array.c'
source_filename = "basic_c_tests/global-array.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.MyStruct = type { i32*, [64 x i8] }

@context = common dso_local global %struct.MyStruct zeroinitializer, align 8, !dbg !0
@padding = internal global <{ i8, [63 x i8] }> <{ i8 -128, [63 x i8] zeroinitializer }>, align 16, !dbg !10

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @update(%struct.MyStruct*, i8*, i64) #0 !dbg !25 {
  %4 = alloca %struct.MyStruct*, align 8
  %5 = alloca i8*, align 8
  %6 = alloca i64, align 8
  %7 = alloca i32, align 4
  store %struct.MyStruct* %0, %struct.MyStruct** %4, align 8
  call void @llvm.dbg.declare(metadata %struct.MyStruct** %4, metadata !30, metadata !DIExpression()), !dbg !31
  store i8* %1, i8** %5, align 8
  call void @llvm.dbg.declare(metadata i8** %5, metadata !32, metadata !DIExpression()), !dbg !33
  store i64 %2, i64* %6, align 8
  call void @llvm.dbg.declare(metadata i64* %6, metadata !34, metadata !DIExpression()), !dbg !35
  call void @llvm.dbg.declare(metadata i32* %7, metadata !36, metadata !DIExpression()), !dbg !37
  %8 = load %struct.MyStruct*, %struct.MyStruct** %4, align 8, !dbg !38
  %9 = getelementptr inbounds %struct.MyStruct, %struct.MyStruct* %8, i32 0, i32 1, !dbg !39
  %10 = load i32, i32* %7, align 4, !dbg !40
  %11 = sext i32 %10 to i64, !dbg !38
  %12 = getelementptr inbounds [64 x i8], [64 x i8]* %9, i64 0, i64 %11, !dbg !38
  %13 = load i8*, i8** %5, align 8, !dbg !41
  %14 = load i64, i64* %6, align 8, !dbg !42
  call void @memcpy(i8* %12, i8* %13, i64 %14), !dbg !43
  ret void, !dbg !44
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: noinline nounwind optnone uwtable
define internal void @memcpy(i8*, i8*, i64) #0 !dbg !45 {
  %4 = alloca i8*, align 8
  %5 = alloca i8*, align 8
  %6 = alloca i64, align 8
  %7 = alloca i64, align 8
  store i8* %0, i8** %4, align 8
  call void @llvm.dbg.declare(metadata i8** %4, metadata !48, metadata !DIExpression()), !dbg !49
  store i8* %1, i8** %5, align 8
  call void @llvm.dbg.declare(metadata i8** %5, metadata !50, metadata !DIExpression()), !dbg !51
  store i64 %2, i64* %6, align 8
  call void @llvm.dbg.declare(metadata i64* %6, metadata !52, metadata !DIExpression()), !dbg !53
  call void @llvm.dbg.declare(metadata i64* %7, metadata !54, metadata !DIExpression()), !dbg !55
  store i64 0, i64* %7, align 8, !dbg !56
  br label %8, !dbg !58

8:                                                ; preds = %20, %3
  %9 = load i64, i64* %7, align 8, !dbg !59
  %10 = load i64, i64* %6, align 8, !dbg !61
  %11 = icmp ult i64 %9, %10, !dbg !62
  br i1 %11, label %12, label %23, !dbg !63

12:                                               ; preds = %8
  %13 = load i8*, i8** %5, align 8, !dbg !64
  %14 = load i64, i64* %7, align 8, !dbg !65
  %15 = getelementptr inbounds i8, i8* %13, i64 %14, !dbg !64
  %16 = load i8, i8* %15, align 1, !dbg !64
  %17 = load i8*, i8** %4, align 8, !dbg !66
  %18 = load i64, i64* %7, align 8, !dbg !67
  %19 = getelementptr inbounds i8, i8* %17, i64 %18, !dbg !66
  store i8 %16, i8* %19, align 1, !dbg !68
  br label %20, !dbg !66

20:                                               ; preds = %12
  %21 = load i64, i64* %7, align 8, !dbg !69
  %22 = add i64 %21, 1, !dbg !69
  store i64 %22, i64* %7, align 8, !dbg !69
  br label %8, !dbg !70, !llvm.loop !71

23:                                               ; preds = %8
  ret void, !dbg !73
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !74 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  call void @llvm.dbg.declare(metadata i32* %2, metadata !77, metadata !DIExpression()), !dbg !78
  %3 = load i32, i32* %2, align 4, !dbg !79
  %4 = sext i32 %3 to i64, !dbg !79
  call void @update(%struct.MyStruct* @context, i8* getelementptr inbounds ([64 x i8], [64 x i8]* bitcast (<{ i8, [63 x i8] }>* @padding to [64 x i8]*), i64 0, i64 0), i64 %4), !dbg !80
  ret i32 0, !dbg !81
}

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!21, !22, !23}
!llvm.ident = !{!24}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "context", scope: !2, file: !3, line: 15, type: !15, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 9.0.0 (tags/RELEASE_900/final)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !5, globals: !9, nameTableKind: None)
!3 = !DIFile(filename: "basic_c_tests/global-array.c", directory: "/home/peiming/Documents/Projects/LLVMRace/TestCases/PTABen")
!4 = !{}
!5 = !{!6}
!6 = !DIDerivedType(tag: DW_TAG_typedef, name: "POINTER", file: !3, line: 8, baseType: !7)
!7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 64)
!8 = !DIBasicType(name: "unsigned char", size: 8, encoding: DW_ATE_unsigned_char)
!9 = !{!0, !10}
!10 = !DIGlobalVariableExpression(var: !11, expr: !DIExpression())
!11 = distinct !DIGlobalVariable(name: "padding", scope: !2, file: !3, line: 17, type: !12, isLocal: true, isDefinition: true)
!12 = !DICompositeType(tag: DW_TAG_array_type, baseType: !8, size: 512, elements: !13)
!13 = !{!14}
!14 = !DISubrange(count: 64)
!15 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "MyStruct", file: !3, line: 10, size: 576, elements: !16)
!16 = !{!17, !20}
!17 = !DIDerivedType(tag: DW_TAG_member, name: "f1", scope: !15, file: !3, line: 11, baseType: !18, size: 64)
!18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !19, size: 64)
!19 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!20 = !DIDerivedType(tag: DW_TAG_member, name: "f2", scope: !15, file: !3, line: 12, baseType: !12, size: 512, offset: 64)
!21 = !{i32 2, !"Dwarf Version", i32 4}
!22 = !{i32 2, !"Debug Info Version", i32 3}
!23 = !{i32 1, !"wchar_size", i32 4}
!24 = !{!"clang version 9.0.0 (tags/RELEASE_900/final)"}
!25 = distinct !DISubprogram(name: "update", scope: !3, file: !3, line: 30, type: !26, scopeLine: 31, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!26 = !DISubroutineType(types: !27)
!27 = !{null, !28, !7, !29}
!28 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !15, size: 64)
!29 = !DIBasicType(name: "long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!30 = !DILocalVariable(name: "context", arg: 1, scope: !25, file: !3, line: 30, type: !28)
!31 = !DILocation(line: 30, column: 30, scope: !25)
!32 = !DILocalVariable(name: "input", arg: 2, scope: !25, file: !3, line: 30, type: !7)
!33 = !DILocation(line: 30, column: 54, scope: !25)
!34 = !DILocalVariable(name: "length", arg: 3, scope: !25, file: !3, line: 30, type: !29)
!35 = !DILocation(line: 30, column: 75, scope: !25)
!36 = !DILocalVariable(name: "index", scope: !25, file: !3, line: 32, type: !19)
!37 = !DILocation(line: 32, column: 6, scope: !25)
!38 = !DILocation(line: 33, column: 19, scope: !25)
!39 = !DILocation(line: 33, column: 28, scope: !25)
!40 = !DILocation(line: 33, column: 31, scope: !25)
!41 = !DILocation(line: 33, column: 48, scope: !25)
!42 = !DILocation(line: 33, column: 55, scope: !25)
!43 = !DILocation(line: 33, column: 2, scope: !25)
!44 = !DILocation(line: 34, column: 1, scope: !25)
!45 = distinct !DISubprogram(name: "memcpy", scope: !3, file: !3, line: 23, type: !46, scopeLine: 24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !4)
!46 = !DISubroutineType(types: !47)
!47 = !{null, !6, !6, !29}
!48 = !DILocalVariable(name: "output", arg: 1, scope: !45, file: !3, line: 23, type: !6)
!49 = !DILocation(line: 23, column: 28, scope: !45)
!50 = !DILocalVariable(name: "input", arg: 2, scope: !45, file: !3, line: 23, type: !6)
!51 = !DILocation(line: 23, column: 44, scope: !45)
!52 = !DILocalVariable(name: "len", arg: 3, scope: !45, file: !3, line: 23, type: !29)
!53 = !DILocation(line: 23, column: 65, scope: !45)
!54 = !DILocalVariable(name: "i", scope: !45, file: !3, line: 25, type: !29)
!55 = !DILocation(line: 25, column: 16, scope: !45)
!56 = !DILocation(line: 26, column: 8, scope: !57)
!57 = distinct !DILexicalBlock(scope: !45, file: !3, line: 26, column: 2)
!58 = !DILocation(line: 26, column: 7, scope: !57)
!59 = !DILocation(line: 26, column: 12, scope: !60)
!60 = distinct !DILexicalBlock(scope: !57, file: !3, line: 26, column: 2)
!61 = !DILocation(line: 26, column: 14, scope: !60)
!62 = !DILocation(line: 26, column: 13, scope: !60)
!63 = !DILocation(line: 26, column: 2, scope: !57)
!64 = !DILocation(line: 27, column: 15, scope: !60)
!65 = !DILocation(line: 27, column: 21, scope: !60)
!66 = !DILocation(line: 27, column: 3, scope: !60)
!67 = !DILocation(line: 27, column: 10, scope: !60)
!68 = !DILocation(line: 27, column: 13, scope: !60)
!69 = !DILocation(line: 26, column: 20, scope: !60)
!70 = !DILocation(line: 26, column: 2, scope: !60)
!71 = distinct !{!71, !63, !72}
!72 = !DILocation(line: 27, column: 22, scope: !57)
!73 = !DILocation(line: 28, column: 1, scope: !45)
!74 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 36, type: !75, scopeLine: 37, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!75 = !DISubroutineType(types: !76)
!76 = !{!19}
!77 = !DILocalVariable(name: "l", scope: !74, file: !3, line: 38, type: !19)
!78 = !DILocation(line: 38, column: 6, scope: !74)
!79 = !DILocation(line: 39, column: 28, scope: !74)
!80 = !DILocation(line: 39, column: 2, scope: !74)
!81 = !DILocation(line: 40, column: 2, scope: !74)

; ModuleID = 'basic_c_tests/global-const-struct.c'
source_filename = "basic_c_tests/global-const-struct.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.MYFILE = type { i32 (i32*)* }
%struct.MyStruct = type { %struct.MYFILE* }

@pts = dso_local constant %struct.MYFILE { i32 (i32*)* @my_sn_write }, align 8, !dbg !0
@ms = dso_local constant %struct.MyStruct { %struct.MYFILE* @pts }, align 8, !dbg !8
@.str = private unnamed_addr constant [15 x i8] c"Executing bar\0A\00", align 1
@g = common dso_local global i32 0, align 4, !dbg !24
@.str.1 = private unnamed_addr constant [23 x i8] c"Executing my_sn_write\0A\00", align 1

; Function Attrs: noinline nounwind optnone uwtable
define internal i32 @my_sn_write(i32*) #0 !dbg !30 {
  %2 = alloca i32*, align 8
  store i32* %0, i32** %2, align 8
  call void @llvm.dbg.declare(metadata i32** %2, metadata !31, metadata !DIExpression()), !dbg !32
  %3 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.1, i64 0, i64 0)), !dbg !33
  %4 = load i32*, i32** %2, align 8, !dbg !34
  %5 = bitcast i32* %4 to i8*, !dbg !34
  call void @__aser_alias__(i8* bitcast (i32* @g to i8*), i8* %5), !dbg !34
  ret i32 0, !dbg !35
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @my_vfprintf(%struct.MyStruct*) #0 !dbg !36 {
  %2 = alloca %struct.MyStruct*, align 8
  %3 = alloca i32*, align 8
  store %struct.MyStruct* %0, %struct.MyStruct** %2, align 8
  call void @llvm.dbg.declare(metadata %struct.MyStruct** %2, metadata !40, metadata !DIExpression()), !dbg !41
  %4 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([15 x i8], [15 x i8]* @.str, i64 0, i64 0)), !dbg !42
  call void @llvm.dbg.declare(metadata i32** %3, metadata !43, metadata !DIExpression()), !dbg !44
  store i32* @g, i32** %3, align 8, !dbg !44
  %5 = load %struct.MyStruct*, %struct.MyStruct** %2, align 8, !dbg !45
  %6 = getelementptr inbounds %struct.MyStruct, %struct.MyStruct* %5, i32 0, i32 0, !dbg !46
  %7 = load %struct.MYFILE*, %struct.MYFILE** %6, align 8, !dbg !46
  %8 = getelementptr inbounds %struct.MYFILE, %struct.MYFILE* %7, i32 0, i32 0, !dbg !47
  %9 = load i32 (i32*)*, i32 (i32*)** %8, align 8, !dbg !47
  %10 = load i32*, i32** %3, align 8, !dbg !48
  %11 = call i32 %9(i32* %10), !dbg !45
  ret void, !dbg !49
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare dso_local i32 @printf(i8*, ...) #2

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !50 {
  %1 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  call void @my_vfprintf(%struct.MyStruct* @ms), !dbg !53
  ret i32 0, !dbg !54
}

declare dso_local void @__aser_alias__(i8*, i8*) #2

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!26, !27, !28}
!llvm.ident = !{!29}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "pts", scope: !2, file: !3, line: 18, type: !15, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 9.0.0 (tags/RELEASE_900/final)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !5, globals: !7, nameTableKind: None)
!3 = !DIFile(filename: "basic_c_tests/global-const-struct.c", directory: "/home/peiming/Documents/Projects/LLVMRace/TestCases/PTABen")
!4 = !{}
!5 = !{!6}
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!7 = !{!0, !8, !24}
!8 = !DIGlobalVariableExpression(var: !9, expr: !DIExpression())
!9 = distinct !DIGlobalVariable(name: "ms", scope: !2, file: !3, line: 19, type: !10, isLocal: false, isDefinition: true)
!10 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !11)
!11 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "MyStruct", file: !3, line: 14, size: 64, elements: !12)
!12 = !{!13}
!13 = !DIDerivedType(tag: DW_TAG_member, name: "myfile", scope: !11, file: !3, line: 15, baseType: !14, size: 64)
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !15, size: 64)
!15 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !16)
!16 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "MYFILE", file: !3, line: 10, size: 64, elements: !17)
!17 = !{!18}
!18 = !DIDerivedType(tag: DW_TAG_member, name: "pt", scope: !16, file: !3, line: 11, baseType: !19, size: 64)
!19 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !20, size: 64)
!20 = !DISubroutineType(types: !21)
!21 = !{!22, !23}
!22 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !22, size: 64)
!24 = !DIGlobalVariableExpression(var: !25, expr: !DIExpression())
!25 = distinct !DIGlobalVariable(name: "g", scope: !2, file: !3, line: 3, type: !22, isLocal: false, isDefinition: true)
!26 = !{i32 2, !"Dwarf Version", i32 4}
!27 = !{i32 2, !"Debug Info Version", i32 3}
!28 = !{i32 1, !"wchar_size", i32 4}
!29 = !{!"clang version 9.0.0 (tags/RELEASE_900/final)"}
!30 = distinct !DISubprogram(name: "my_sn_write", scope: !3, file: !3, line: 4, type: !20, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !4)
!31 = !DILocalVariable(name: "p", arg: 1, scope: !30, file: !3, line: 4, type: !23)
!32 = !DILocation(line: 4, column: 29, scope: !30)
!33 = !DILocation(line: 5, column: 5, scope: !30)
!34 = !DILocation(line: 6, column: 5, scope: !30)
!35 = !DILocation(line: 7, column: 5, scope: !30)
!36 = distinct !DISubprogram(name: "my_vfprintf", scope: !3, file: !3, line: 21, type: !37, scopeLine: 21, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!37 = !DISubroutineType(types: !38)
!38 = !{null, !39}
!39 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64)
!40 = !DILocalVariable(name: "ms", arg: 1, scope: !36, file: !3, line: 21, type: !39)
!41 = !DILocation(line: 21, column: 41, scope: !36)
!42 = !DILocation(line: 22, column: 5, scope: !36)
!43 = !DILocalVariable(name: "p", scope: !36, file: !3, line: 23, type: !23)
!44 = !DILocation(line: 23, column: 10, scope: !36)
!45 = !DILocation(line: 24, column: 5, scope: !36)
!46 = !DILocation(line: 24, column: 9, scope: !36)
!47 = !DILocation(line: 24, column: 17, scope: !36)
!48 = !DILocation(line: 24, column: 20, scope: !36)
!49 = !DILocation(line: 25, column: 1, scope: !36)
!50 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 27, type: !51, scopeLine: 27, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!51 = !DISubroutineType(types: !52)
!52 = !{!22}
!53 = !DILocation(line: 28, column: 5, scope: !50)
!54 = !DILocation(line: 29, column: 5, scope: !50)

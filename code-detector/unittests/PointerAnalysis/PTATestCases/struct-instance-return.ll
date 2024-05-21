; ModuleID = 'basic_c_tests/struct-instance-return.c'
source_filename = "basic_c_tests/struct-instance-return.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.MyStruct = type { i32*, i8 }

@x = common dso_local global i32 0, align 4, !dbg !0
@y = common dso_local global i32 0, align 4, !dbg !8

; Function Attrs: noinline nounwind optnone uwtable
define dso_local { i32*, i8 } @foo() #0 !dbg !15 {
  %1 = alloca %struct.MyStruct, align 8
  call void @llvm.dbg.declare(metadata %struct.MyStruct* %1, metadata !24, metadata !DIExpression()), !dbg !25
  %2 = getelementptr inbounds %struct.MyStruct, %struct.MyStruct* %1, i32 0, i32 0, !dbg !26
  store i32* @x, i32** %2, align 8, !dbg !27
  %3 = bitcast %struct.MyStruct* %1 to { i32*, i8 }*, !dbg !28
  %4 = load { i32*, i8 }, { i32*, i8 }* %3, align 8, !dbg !28
  ret { i32*, i8 } %4, !dbg !28
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !29 {
  %1 = alloca i32, align 4
  %2 = alloca %struct.MyStruct, align 8
  %3 = alloca %struct.MyStruct, align 8
  store i32 0, i32* %1, align 4
  call void @llvm.dbg.declare(metadata %struct.MyStruct* %2, metadata !32, metadata !DIExpression()), !dbg !33
  %4 = call { i32*, i8 } @foo(), !dbg !34
  %5 = bitcast %struct.MyStruct* %3 to { i32*, i8 }*, !dbg !34
  %6 = getelementptr inbounds { i32*, i8 }, { i32*, i8 }* %5, i32 0, i32 0, !dbg !34
  %7 = extractvalue { i32*, i8 } %4, 0, !dbg !34
  store i32* %7, i32** %6, align 8, !dbg !34
  %8 = getelementptr inbounds { i32*, i8 }, { i32*, i8 }* %5, i32 0, i32 1, !dbg !34
  %9 = extractvalue { i32*, i8 } %4, 1, !dbg !34
  store i8 %9, i8* %8, align 8, !dbg !34
  %10 = bitcast %struct.MyStruct* %2 to i8*, !dbg !34
  %11 = bitcast %struct.MyStruct* %3 to i8*, !dbg !34
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %10, i8* align 8 %11, i64 16, i1 false), !dbg !34
  %12 = getelementptr inbounds %struct.MyStruct, %struct.MyStruct* %2, i32 0, i32 0, !dbg !35
  %13 = load i32*, i32** %12, align 8, !dbg !35
  %14 = call i32 (i32*, i32*, ...) bitcast (i32 (...)* @EXPECTEDFAIL_MAYALIAS to i32 (i32*, i32*, ...)*)(i32* %13, i32* @x), !dbg !36
  %15 = getelementptr inbounds %struct.MyStruct, %struct.MyStruct* %2, i32 0, i32 0, !dbg !37
  %16 = load i32*, i32** %15, align 8, !dbg !37
  %17 = bitcast i32* %16 to i8*, !dbg !37
  call void @__aser_no_alias__(i8* %17, i8* bitcast (i32* @y to i8*)), !dbg !37
  ret i32 0, !dbg !38
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1 immarg) #2

declare dso_local i32 @EXPECTEDFAIL_MAYALIAS(...) #3

declare dso_local void @__aser_no_alias__(i8*, i8*) #3

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { argmemonly nounwind }
attributes #3 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!11, !12, !13}
!llvm.ident = !{!14}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "x", scope: !2, file: !3, line: 13, type: !10, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 9.0.0 (tags/RELEASE_900/final)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !5, globals: !7, nameTableKind: None)
!3 = !DIFile(filename: "basic_c_tests/struct-instance-return.c", directory: "/home/peiming/Documents/Projects/LLVMRace/TestCases/PTABen")
!4 = !{}
!5 = !{!6}
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!7 = !{!0, !8}
!8 = !DIGlobalVariableExpression(var: !9, expr: !DIExpression())
!9 = distinct !DIGlobalVariable(name: "y", scope: !2, file: !3, line: 13, type: !10, isLocal: false, isDefinition: true)
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{i32 2, !"Dwarf Version", i32 4}
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !{i32 1, !"wchar_size", i32 4}
!14 = !{!"clang version 9.0.0 (tags/RELEASE_900/final)"}
!15 = distinct !DISubprogram(name: "foo", scope: !3, file: !3, line: 15, type: !16, scopeLine: 15, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!16 = !DISubroutineType(types: !17)
!17 = !{!18}
!18 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "MyStruct", file: !3, line: 8, size: 128, elements: !19)
!19 = !{!20, !22}
!20 = !DIDerivedType(tag: DW_TAG_member, name: "f1", scope: !18, file: !3, line: 9, baseType: !21, size: 64)
!21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64)
!22 = !DIDerivedType(tag: DW_TAG_member, name: "f2", scope: !18, file: !3, line: 10, baseType: !23, size: 8, offset: 64)
!23 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!24 = !DILocalVariable(name: "m", scope: !15, file: !3, line: 16, type: !18)
!25 = !DILocation(line: 16, column: 18, scope: !15)
!26 = !DILocation(line: 17, column: 4, scope: !15)
!27 = !DILocation(line: 17, column: 7, scope: !15)
!28 = !DILocation(line: 18, column: 2, scope: !15)
!29 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 21, type: !30, scopeLine: 21, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!30 = !DISubroutineType(types: !31)
!31 = !{!10}
!32 = !DILocalVariable(name: "m", scope: !29, file: !3, line: 22, type: !18)
!33 = !DILocation(line: 22, column: 18, scope: !29)
!34 = !DILocation(line: 23, column: 6, scope: !29)
!35 = !DILocation(line: 24, column: 26, scope: !29)
!36 = !DILocation(line: 24, column: 2, scope: !29)
!37 = !DILocation(line: 25, column: 2, scope: !29)
!38 = !DILocation(line: 26, column: 2, scope: !29)

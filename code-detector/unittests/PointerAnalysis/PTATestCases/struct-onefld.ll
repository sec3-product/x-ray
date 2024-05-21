; ModuleID = 'basic_c_tests/struct-onefld.c'
source_filename = "basic_c_tests/struct-onefld.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.IntStruct = type { i32 }
%struct.CharStruct = type { i8 }

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !9 {
  %1 = alloca i32, align 4
  %2 = alloca %struct.IntStruct*, align 8
  %3 = alloca %struct.IntStruct*, align 8
  %4 = alloca %struct.IntStruct, align 4
  %5 = alloca %struct.CharStruct*, align 8
  %6 = alloca %struct.CharStruct*, align 8
  %7 = alloca %struct.CharStruct, align 1
  store i32 0, i32* %1, align 4
  call void @llvm.dbg.declare(metadata %struct.IntStruct** %2, metadata !13, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata %struct.IntStruct** %3, metadata !19, metadata !DIExpression()), !dbg !20
  call void @llvm.dbg.declare(metadata %struct.IntStruct* %4, metadata !21, metadata !DIExpression()), !dbg !22
  store %struct.IntStruct* %4, %struct.IntStruct** %2, align 8, !dbg !23
  store %struct.IntStruct* %4, %struct.IntStruct** %3, align 8, !dbg !24
  %8 = load %struct.IntStruct*, %struct.IntStruct** %2, align 8, !dbg !25
  %9 = getelementptr inbounds %struct.IntStruct, %struct.IntStruct* %8, i32 0, i32 0, !dbg !25
  %10 = bitcast i32* %9 to i8*, !dbg !25
  %11 = load %struct.IntStruct*, %struct.IntStruct** %3, align 8, !dbg !25
  %12 = getelementptr inbounds %struct.IntStruct, %struct.IntStruct* %11, i32 0, i32 0, !dbg !25
  %13 = bitcast i32* %12 to i8*, !dbg !25
  call void @__aser_alias__(i8* %10, i8* %13), !dbg !25
  %14 = load %struct.IntStruct*, %struct.IntStruct** %2, align 8, !dbg !26
  %15 = getelementptr inbounds %struct.IntStruct, %struct.IntStruct* %14, i32 0, i32 0, !dbg !26
  %16 = bitcast i32* %15 to i8*, !dbg !26
  %17 = getelementptr inbounds %struct.IntStruct, %struct.IntStruct* %4, i32 0, i32 0, !dbg !26
  %18 = bitcast i32* %17 to i8*, !dbg !26
  call void @__aser_alias__(i8* %16, i8* %18), !dbg !26
  call void @llvm.dbg.declare(metadata %struct.CharStruct** %5, metadata !27, metadata !DIExpression()), !dbg !33
  call void @llvm.dbg.declare(metadata %struct.CharStruct** %6, metadata !34, metadata !DIExpression()), !dbg !35
  call void @llvm.dbg.declare(metadata %struct.CharStruct* %7, metadata !36, metadata !DIExpression()), !dbg !37
  store %struct.CharStruct* %7, %struct.CharStruct** %5, align 8, !dbg !38
  store %struct.CharStruct* %7, %struct.CharStruct** %6, align 8, !dbg !39
  %19 = load %struct.CharStruct*, %struct.CharStruct** %5, align 8, !dbg !40
  %20 = getelementptr inbounds %struct.CharStruct, %struct.CharStruct* %19, i32 0, i32 0, !dbg !40
  %21 = load %struct.CharStruct*, %struct.CharStruct** %6, align 8, !dbg !40
  %22 = getelementptr inbounds %struct.CharStruct, %struct.CharStruct* %21, i32 0, i32 0, !dbg !40
  call void @__aser_alias__(i8* %20, i8* %22), !dbg !40
  %23 = load %struct.CharStruct*, %struct.CharStruct** %5, align 8, !dbg !41
  %24 = getelementptr inbounds %struct.CharStruct, %struct.CharStruct* %23, i32 0, i32 0, !dbg !41
  %25 = getelementptr inbounds %struct.CharStruct, %struct.CharStruct* %7, i32 0, i32 0, !dbg !41
  call void @__aser_alias__(i8* %24, i8* %25), !dbg !41
  ret i32 0, !dbg !42
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
!1 = !DIFile(filename: "basic_c_tests/struct-onefld.c", directory: "/home/peiming/Documents/Projects/LLVMRace/TestCases/PTABen")
!2 = !{}
!3 = !{!4}
!4 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!5 = !{i32 2, !"Dwarf Version", i32 4}
!6 = !{i32 2, !"Debug Info Version", i32 3}
!7 = !{i32 1, !"wchar_size", i32 4}
!8 = !{!"clang version 9.0.0 (tags/RELEASE_900/final)"}
!9 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 16, type: !10, scopeLine: 16, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!10 = !DISubroutineType(types: !11)
!11 = !{!12}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DILocalVariable(name: "pint1", scope: !9, file: !1, line: 17, type: !14)
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !15, size: 64)
!15 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "IntStruct", file: !1, line: 8, size: 32, elements: !16)
!16 = !{!17}
!17 = !DIDerivedType(tag: DW_TAG_member, name: "f1", scope: !15, file: !1, line: 9, baseType: !12, size: 32)
!18 = !DILocation(line: 17, column: 20, scope: !9)
!19 = !DILocalVariable(name: "pint2", scope: !9, file: !1, line: 17, type: !14)
!20 = !DILocation(line: 17, column: 28, scope: !9)
!21 = !DILocalVariable(name: "s", scope: !9, file: !1, line: 18, type: !15)
!22 = !DILocation(line: 18, column: 19, scope: !9)
!23 = !DILocation(line: 19, column: 8, scope: !9)
!24 = !DILocation(line: 20, column: 8, scope: !9)
!25 = !DILocation(line: 21, column: 2, scope: !9)
!26 = !DILocation(line: 22, column: 2, scope: !9)
!27 = !DILocalVariable(name: "qint1", scope: !9, file: !1, line: 24, type: !28)
!28 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !29, size: 64)
!29 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "CharStruct", file: !1, line: 12, size: 8, elements: !30)
!30 = !{!31}
!31 = !DIDerivedType(tag: DW_TAG_member, name: "f1", scope: !29, file: !1, line: 13, baseType: !32, size: 8)
!32 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!33 = !DILocation(line: 24, column: 21, scope: !9)
!34 = !DILocalVariable(name: "qint2", scope: !9, file: !1, line: 24, type: !28)
!35 = !DILocation(line: 24, column: 29, scope: !9)
!36 = !DILocalVariable(name: "t", scope: !9, file: !1, line: 25, type: !29)
!37 = !DILocation(line: 25, column: 20, scope: !9)
!38 = !DILocation(line: 26, column: 8, scope: !9)
!39 = !DILocation(line: 27, column: 8, scope: !9)
!40 = !DILocation(line: 28, column: 2, scope: !9)
!41 = !DILocation(line: 29, column: 2, scope: !9)
!42 = !DILocation(line: 31, column: 2, scope: !9)

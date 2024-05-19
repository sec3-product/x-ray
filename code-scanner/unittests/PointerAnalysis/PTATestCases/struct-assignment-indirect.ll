; ModuleID = 'basic_c_tests/struct-assignment-indirect.c'
source_filename = "basic_c_tests/struct-assignment-indirect.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.s = type { i32*, i32* }

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !9 {
  %1 = alloca i32, align 4
  %2 = alloca %struct.s, align 8
  %3 = alloca %struct.s, align 8
  %4 = alloca %struct.s*, align 8
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  call void @llvm.dbg.declare(metadata %struct.s* %2, metadata !13, metadata !DIExpression()), !dbg !19
  call void @llvm.dbg.declare(metadata %struct.s* %3, metadata !20, metadata !DIExpression()), !dbg !21
  call void @llvm.dbg.declare(metadata %struct.s** %4, metadata !22, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.declare(metadata i32* %5, metadata !25, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.declare(metadata i32* %6, metadata !27, metadata !DIExpression()), !dbg !28
  %7 = getelementptr inbounds %struct.s, %struct.s* %2, i32 0, i32 0, !dbg !29
  store i32* %5, i32** %7, align 8, !dbg !30
  %8 = getelementptr inbounds %struct.s, %struct.s* %2, i32 0, i32 1, !dbg !31
  store i32* %6, i32** %8, align 8, !dbg !32
  %9 = bitcast %struct.s* %3 to i8*, !dbg !33
  %10 = bitcast %struct.s* %2 to i8*, !dbg !33
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %9, i8* align 8 %10, i64 16, i1 false), !dbg !33
  store %struct.s* %2, %struct.s** %4, align 8, !dbg !34
  %11 = load %struct.s*, %struct.s** %4, align 8, !dbg !35
  %12 = getelementptr inbounds %struct.s, %struct.s* %11, i32 0, i32 0, !dbg !35
  %13 = load i32*, i32** %12, align 8, !dbg !35
  %14 = bitcast i32* %13 to i8*, !dbg !35
  %15 = getelementptr inbounds %struct.s, %struct.s* %3, i32 0, i32 0, !dbg !35
  %16 = load i32*, i32** %15, align 8, !dbg !35
  %17 = bitcast i32* %16 to i8*, !dbg !35
  call void @__aser_alias__(i8* %14, i8* %17), !dbg !35
  %18 = load %struct.s*, %struct.s** %4, align 8, !dbg !36
  %19 = getelementptr inbounds %struct.s, %struct.s* %18, i32 0, i32 1, !dbg !36
  %20 = load i32*, i32** %19, align 8, !dbg !36
  %21 = bitcast i32* %20 to i8*, !dbg !36
  %22 = getelementptr inbounds %struct.s, %struct.s* %3, i32 0, i32 1, !dbg !36
  %23 = load i32*, i32** %22, align 8, !dbg !36
  %24 = bitcast i32* %23 to i8*, !dbg !36
  call void @__aser_alias__(i8* %21, i8* %24), !dbg !36
  ret i32 0, !dbg !37
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1 immarg) #2

declare dso_local void @__aser_alias__(i8*, i8*) #3

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { argmemonly nounwind }
attributes #3 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!5, !6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 9.0.0 (tags/RELEASE_900/final)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, nameTableKind: None)
!1 = !DIFile(filename: "basic_c_tests/struct-assignment-indirect.c", directory: "/home/peiming/Documents/Projects/LLVMRace/TestCases/PTABen")
!2 = !{}
!3 = !{!4}
!4 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!5 = !{i32 2, !"Dwarf Version", i32 4}
!6 = !{i32 2, !"Debug Info Version", i32 3}
!7 = !{i32 1, !"wchar_size", i32 4}
!8 = !{!"clang version 9.0.0 (tags/RELEASE_900/final)"}
!9 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 13, type: !10, scopeLine: 14, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!10 = !DISubroutineType(types: !11)
!11 = !{!12}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DILocalVariable(name: "s1", scope: !9, file: !1, line: 15, type: !14)
!14 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "s", file: !1, line: 8, size: 128, elements: !15)
!15 = !{!16, !18}
!16 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !14, file: !1, line: 9, baseType: !17, size: 64)
!17 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!18 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !14, file: !1, line: 10, baseType: !17, size: 64, offset: 64)
!19 = !DILocation(line: 15, column: 11, scope: !9)
!20 = !DILocalVariable(name: "s2", scope: !9, file: !1, line: 15, type: !14)
!21 = !DILocation(line: 15, column: 15, scope: !9)
!22 = !DILocalVariable(name: "p1", scope: !9, file: !1, line: 16, type: !23)
!23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 64)
!24 = !DILocation(line: 16, column: 13, scope: !9)
!25 = !DILocalVariable(name: "x", scope: !9, file: !1, line: 17, type: !12)
!26 = !DILocation(line: 17, column: 6, scope: !9)
!27 = !DILocalVariable(name: "y", scope: !9, file: !1, line: 17, type: !12)
!28 = !DILocation(line: 17, column: 9, scope: !9)
!29 = !DILocation(line: 18, column: 5, scope: !9)
!30 = !DILocation(line: 18, column: 7, scope: !9)
!31 = !DILocation(line: 19, column: 5, scope: !9)
!32 = !DILocation(line: 19, column: 7, scope: !9)
!33 = !DILocation(line: 20, column: 7, scope: !9)
!34 = !DILocation(line: 21, column: 5, scope: !9)
!35 = !DILocation(line: 22, column: 2, scope: !9)
!36 = !DILocation(line: 23, column: 2, scope: !9)
!37 = !DILocation(line: 24, column: 2, scope: !9)

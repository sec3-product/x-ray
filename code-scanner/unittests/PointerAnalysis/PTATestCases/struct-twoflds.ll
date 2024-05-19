; ModuleID = 'basic_c_tests/struct-twoflds.c'
source_filename = "basic_c_tests/struct-twoflds.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.IntChar = type { i32, i8 }
%struct.CharInt = type { i8, i32 }

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !9 {
  %1 = alloca i32, align 4
  %2 = alloca %struct.IntChar*, align 8
  %3 = alloca %struct.IntChar*, align 8
  %4 = alloca %struct.IntChar, align 4
  %5 = alloca %struct.CharInt*, align 8
  %6 = alloca %struct.CharInt*, align 8
  %7 = alloca %struct.CharInt, align 4
  store i32 0, i32* %1, align 4
  call void @llvm.dbg.declare(metadata %struct.IntChar** %2, metadata !13, metadata !DIExpression()), !dbg !20
  call void @llvm.dbg.declare(metadata %struct.IntChar** %3, metadata !21, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.declare(metadata %struct.IntChar* %4, metadata !23, metadata !DIExpression()), !dbg !24
  store %struct.IntChar* %4, %struct.IntChar** %2, align 8, !dbg !25
  store %struct.IntChar* %4, %struct.IntChar** %3, align 8, !dbg !26
  %8 = load %struct.IntChar*, %struct.IntChar** %2, align 8, !dbg !27
  %9 = getelementptr inbounds %struct.IntChar, %struct.IntChar* %8, i32 0, i32 0, !dbg !27
  %10 = bitcast i32* %9 to i8*, !dbg !27
  %11 = load %struct.IntChar*, %struct.IntChar** %3, align 8, !dbg !27
  %12 = getelementptr inbounds %struct.IntChar, %struct.IntChar* %11, i32 0, i32 0, !dbg !27
  %13 = bitcast i32* %12 to i8*, !dbg !27
  call void @__aser_alias__(i8* %10, i8* %13), !dbg !27
  %14 = load %struct.IntChar*, %struct.IntChar** %2, align 8, !dbg !28
  %15 = getelementptr inbounds %struct.IntChar, %struct.IntChar* %14, i32 0, i32 1, !dbg !28
  %16 = load %struct.IntChar*, %struct.IntChar** %3, align 8, !dbg !28
  %17 = getelementptr inbounds %struct.IntChar, %struct.IntChar* %16, i32 0, i32 1, !dbg !28
  call void @__aser_alias__(i8* %15, i8* %17), !dbg !28
  %18 = load %struct.IntChar*, %struct.IntChar** %2, align 8, !dbg !29
  %19 = getelementptr inbounds %struct.IntChar, %struct.IntChar* %18, i32 0, i32 0, !dbg !29
  %20 = bitcast i32* %19 to i8*, !dbg !29
  %21 = load %struct.IntChar*, %struct.IntChar** %3, align 8, !dbg !29
  %22 = getelementptr inbounds %struct.IntChar, %struct.IntChar* %21, i32 0, i32 1, !dbg !29
  call void @__aser_no_alias__(i8* %20, i8* %22), !dbg !29
  call void @llvm.dbg.declare(metadata %struct.CharInt** %5, metadata !30, metadata !DIExpression()), !dbg !36
  call void @llvm.dbg.declare(metadata %struct.CharInt** %6, metadata !37, metadata !DIExpression()), !dbg !38
  call void @llvm.dbg.declare(metadata %struct.CharInt* %7, metadata !39, metadata !DIExpression()), !dbg !40
  store %struct.CharInt* %7, %struct.CharInt** %5, align 8, !dbg !41
  store %struct.CharInt* %7, %struct.CharInt** %6, align 8, !dbg !42
  %23 = load %struct.CharInt*, %struct.CharInt** %5, align 8, !dbg !43
  %24 = getelementptr inbounds %struct.CharInt, %struct.CharInt* %23, i32 0, i32 0, !dbg !43
  %25 = load %struct.CharInt*, %struct.CharInt** %6, align 8, !dbg !43
  %26 = getelementptr inbounds %struct.CharInt, %struct.CharInt* %25, i32 0, i32 0, !dbg !43
  call void @__aser_alias__(i8* %24, i8* %26), !dbg !43
  %27 = load %struct.CharInt*, %struct.CharInt** %5, align 8, !dbg !44
  %28 = getelementptr inbounds %struct.CharInt, %struct.CharInt* %27, i32 0, i32 1, !dbg !44
  %29 = bitcast i32* %28 to i8*, !dbg !44
  %30 = load %struct.CharInt*, %struct.CharInt** %6, align 8, !dbg !44
  %31 = getelementptr inbounds %struct.CharInt, %struct.CharInt* %30, i32 0, i32 1, !dbg !44
  %32 = bitcast i32* %31 to i8*, !dbg !44
  call void @__aser_alias__(i8* %29, i8* %32), !dbg !44
  %33 = load %struct.CharInt*, %struct.CharInt** %5, align 8, !dbg !45
  %34 = getelementptr inbounds %struct.CharInt, %struct.CharInt* %33, i32 0, i32 0, !dbg !45
  %35 = load %struct.CharInt*, %struct.CharInt** %6, align 8, !dbg !45
  %36 = getelementptr inbounds %struct.CharInt, %struct.CharInt* %35, i32 0, i32 1, !dbg !45
  %37 = bitcast i32* %36 to i8*, !dbg !45
  call void @__aser_no_alias__(i8* %34, i8* %37), !dbg !45
  ret i32 0, !dbg !46
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare dso_local void @__aser_alias__(i8*, i8*) #2

declare dso_local void @__aser_no_alias__(i8*, i8*) #2

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!5, !6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 9.0.0 (tags/RELEASE_900/final)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, nameTableKind: None)
!1 = !DIFile(filename: "basic_c_tests/struct-twoflds.c", directory: "/home/peiming/Documents/Projects/LLVMRace/TestCases/PTABen")
!2 = !{}
!3 = !{!4}
!4 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!5 = !{i32 2, !"Dwarf Version", i32 4}
!6 = !{i32 2, !"Debug Info Version", i32 3}
!7 = !{i32 1, !"wchar_size", i32 4}
!8 = !{!"clang version 9.0.0 (tags/RELEASE_900/final)"}
!9 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 18, type: !10, scopeLine: 18, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!10 = !DISubroutineType(types: !11)
!11 = !{!12}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DILocalVariable(name: "pint1", scope: !9, file: !1, line: 19, type: !14)
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !15, size: 64)
!15 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "IntChar", file: !1, line: 8, size: 64, elements: !16)
!16 = !{!17, !18}
!17 = !DIDerivedType(tag: DW_TAG_member, name: "f1", scope: !15, file: !1, line: 9, baseType: !12, size: 32)
!18 = !DIDerivedType(tag: DW_TAG_member, name: "f2", scope: !15, file: !1, line: 10, baseType: !19, size: 8, offset: 32)
!19 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!20 = !DILocation(line: 19, column: 18, scope: !9)
!21 = !DILocalVariable(name: "pint2", scope: !9, file: !1, line: 19, type: !14)
!22 = !DILocation(line: 19, column: 26, scope: !9)
!23 = !DILocalVariable(name: "s", scope: !9, file: !1, line: 20, type: !15)
!24 = !DILocation(line: 20, column: 17, scope: !9)
!25 = !DILocation(line: 21, column: 8, scope: !9)
!26 = !DILocation(line: 22, column: 8, scope: !9)
!27 = !DILocation(line: 23, column: 2, scope: !9)
!28 = !DILocation(line: 24, column: 2, scope: !9)
!29 = !DILocation(line: 25, column: 2, scope: !9)
!30 = !DILocalVariable(name: "qint1", scope: !9, file: !1, line: 27, type: !31)
!31 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !32, size: 64)
!32 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "CharInt", file: !1, line: 13, size: 64, elements: !33)
!33 = !{!34, !35}
!34 = !DIDerivedType(tag: DW_TAG_member, name: "f1", scope: !32, file: !1, line: 14, baseType: !19, size: 8)
!35 = !DIDerivedType(tag: DW_TAG_member, name: "f2", scope: !32, file: !1, line: 15, baseType: !12, size: 32, offset: 32)
!36 = !DILocation(line: 27, column: 18, scope: !9)
!37 = !DILocalVariable(name: "qint2", scope: !9, file: !1, line: 27, type: !31)
!38 = !DILocation(line: 27, column: 26, scope: !9)
!39 = !DILocalVariable(name: "t", scope: !9, file: !1, line: 28, type: !32)
!40 = !DILocation(line: 28, column: 17, scope: !9)
!41 = !DILocation(line: 29, column: 8, scope: !9)
!42 = !DILocation(line: 30, column: 8, scope: !9)
!43 = !DILocation(line: 31, column: 2, scope: !9)
!44 = !DILocation(line: 32, column: 2, scope: !9)
!45 = !DILocation(line: 33, column: 2, scope: !9)
!46 = !DILocation(line: 35, column: 2, scope: !9)

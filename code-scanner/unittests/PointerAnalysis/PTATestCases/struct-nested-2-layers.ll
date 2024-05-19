; ModuleID = 'basic_c_tests/struct-nested-2-layers.c'
source_filename = "basic_c_tests/struct-nested-2-layers.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.OuterStruct = type { %struct.MidStruct, i8, %struct.InnerStruct, i32 }
%struct.MidStruct = type { i32, %struct.InnerStruct, i8 }
%struct.InnerStruct = type { i32, i8 }

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !9 {
  %1 = alloca i32, align 4
  %2 = alloca %struct.OuterStruct*, align 8
  %3 = alloca %struct.MidStruct*, align 8
  %4 = alloca %struct.MidStruct*, align 8
  %5 = alloca %struct.InnerStruct*, align 8
  %6 = alloca %struct.InnerStruct*, align 8
  %7 = alloca %struct.OuterStruct, align 4
  store i32 0, i32* %1, align 4
  call void @llvm.dbg.declare(metadata %struct.OuterStruct** %2, metadata !13, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.declare(metadata %struct.MidStruct** %3, metadata !32, metadata !DIExpression()), !dbg !34
  call void @llvm.dbg.declare(metadata %struct.MidStruct** %4, metadata !35, metadata !DIExpression()), !dbg !36
  call void @llvm.dbg.declare(metadata %struct.InnerStruct** %5, metadata !37, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata %struct.InnerStruct** %6, metadata !40, metadata !DIExpression()), !dbg !41
  call void @llvm.dbg.declare(metadata %struct.OuterStruct* %7, metadata !42, metadata !DIExpression()), !dbg !43
  store %struct.OuterStruct* %7, %struct.OuterStruct** %2, align 8, !dbg !44
  %8 = getelementptr inbounds %struct.OuterStruct, %struct.OuterStruct* %7, i32 0, i32 0, !dbg !45
  store %struct.MidStruct* %8, %struct.MidStruct** %3, align 8, !dbg !46
  %9 = load %struct.OuterStruct*, %struct.OuterStruct** %2, align 8, !dbg !47
  %10 = getelementptr inbounds %struct.OuterStruct, %struct.OuterStruct* %9, i32 0, i32 0, !dbg !48
  store %struct.MidStruct* %10, %struct.MidStruct** %4, align 8, !dbg !49
  %11 = load %struct.MidStruct*, %struct.MidStruct** %4, align 8, !dbg !50
  %12 = bitcast %struct.MidStruct* %11 to i8*, !dbg !50
  %13 = load %struct.MidStruct*, %struct.MidStruct** %3, align 8, !dbg !50
  %14 = bitcast %struct.MidStruct* %13 to i8*, !dbg !50
  call void @__aser_alias__(i8* %12, i8* %14), !dbg !50
  %15 = load %struct.MidStruct*, %struct.MidStruct** %4, align 8, !dbg !51
  %16 = getelementptr inbounds %struct.MidStruct, %struct.MidStruct* %15, i32 0, i32 1, !dbg !51
  %17 = getelementptr inbounds %struct.InnerStruct, %struct.InnerStruct* %16, i32 0, i32 0, !dbg !51
  %18 = bitcast i32* %17 to i8*, !dbg !51
  %19 = load %struct.MidStruct*, %struct.MidStruct** %3, align 8, !dbg !51
  %20 = getelementptr inbounds %struct.MidStruct, %struct.MidStruct* %19, i32 0, i32 1, !dbg !51
  %21 = getelementptr inbounds %struct.InnerStruct, %struct.InnerStruct* %20, i32 0, i32 0, !dbg !51
  %22 = bitcast i32* %21 to i8*, !dbg !51
  call void @__aser_alias__(i8* %18, i8* %22), !dbg !51
  %23 = load %struct.MidStruct*, %struct.MidStruct** %4, align 8, !dbg !52
  %24 = getelementptr inbounds %struct.MidStruct, %struct.MidStruct* %23, i32 0, i32 1, !dbg !52
  %25 = getelementptr inbounds %struct.InnerStruct, %struct.InnerStruct* %24, i32 0, i32 1, !dbg !52
  %26 = load %struct.MidStruct*, %struct.MidStruct** %3, align 8, !dbg !52
  %27 = getelementptr inbounds %struct.MidStruct, %struct.MidStruct* %26, i32 0, i32 1, !dbg !52
  %28 = getelementptr inbounds %struct.InnerStruct, %struct.InnerStruct* %27, i32 0, i32 1, !dbg !52
  call void @__aser_alias__(i8* %25, i8* %28), !dbg !52
  %29 = getelementptr inbounds %struct.OuterStruct, %struct.OuterStruct* %7, i32 0, i32 0, !dbg !53
  %30 = getelementptr inbounds %struct.MidStruct, %struct.MidStruct* %29, i32 0, i32 1, !dbg !54
  store %struct.InnerStruct* %30, %struct.InnerStruct** %6, align 8, !dbg !55
  %31 = load %struct.OuterStruct*, %struct.OuterStruct** %2, align 8, !dbg !56
  %32 = getelementptr inbounds %struct.OuterStruct, %struct.OuterStruct* %31, i32 0, i32 0, !dbg !57
  %33 = getelementptr inbounds %struct.MidStruct, %struct.MidStruct* %32, i32 0, i32 1, !dbg !58
  store %struct.InnerStruct* %33, %struct.InnerStruct** %5, align 8, !dbg !59
  %34 = load %struct.InnerStruct*, %struct.InnerStruct** %5, align 8, !dbg !60
  %35 = bitcast %struct.InnerStruct* %34 to i8*, !dbg !60
  %36 = load %struct.InnerStruct*, %struct.InnerStruct** %6, align 8, !dbg !60
  %37 = bitcast %struct.InnerStruct* %36 to i8*, !dbg !60
  call void @__aser_alias__(i8* %35, i8* %37), !dbg !60
  %38 = load %struct.InnerStruct*, %struct.InnerStruct** %5, align 8, !dbg !61
  %39 = getelementptr inbounds %struct.InnerStruct, %struct.InnerStruct* %38, i32 0, i32 0, !dbg !61
  %40 = bitcast i32* %39 to i8*, !dbg !61
  %41 = load %struct.InnerStruct*, %struct.InnerStruct** %6, align 8, !dbg !61
  %42 = getelementptr inbounds %struct.InnerStruct, %struct.InnerStruct* %41, i32 0, i32 0, !dbg !61
  %43 = bitcast i32* %42 to i8*, !dbg !61
  call void @__aser_alias__(i8* %40, i8* %43), !dbg !61
  %44 = load %struct.InnerStruct*, %struct.InnerStruct** %5, align 8, !dbg !62
  %45 = getelementptr inbounds %struct.InnerStruct, %struct.InnerStruct* %44, i32 0, i32 1, !dbg !62
  %46 = load %struct.InnerStruct*, %struct.InnerStruct** %6, align 8, !dbg !62
  %47 = getelementptr inbounds %struct.InnerStruct, %struct.InnerStruct* %46, i32 0, i32 1, !dbg !62
  call void @__aser_alias__(i8* %45, i8* %47), !dbg !62
  %48 = load %struct.OuterStruct*, %struct.OuterStruct** %2, align 8, !dbg !63
  %49 = getelementptr inbounds %struct.OuterStruct, %struct.OuterStruct* %48, i32 0, i32 2, !dbg !63
  %50 = bitcast %struct.InnerStruct* %49 to i8*, !dbg !63
  %51 = load %struct.InnerStruct*, %struct.InnerStruct** %6, align 8, !dbg !63
  %52 = bitcast %struct.InnerStruct* %51 to i8*, !dbg !63
  call void @__aser_no_alias__(i8* %50, i8* %52), !dbg !63
  ret i32 0, !dbg !64
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
!1 = !DIFile(filename: "basic_c_tests/struct-nested-2-layers.c", directory: "/home/peiming/Documents/Projects/LLVMRace/TestCases/PTABen")
!2 = !{}
!3 = !{!4}
!4 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!5 = !{i32 2, !"Dwarf Version", i32 4}
!6 = !{i32 2, !"Debug Info Version", i32 3}
!7 = !{i32 1, !"wchar_size", i32 4}
!8 = !{!"clang version 9.0.0 (tags/RELEASE_900/final)"}
!9 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 26, type: !10, scopeLine: 26, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!10 = !DISubroutineType(types: !11)
!11 = !{!12}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DILocalVariable(name: "pout", scope: !9, file: !1, line: 27, type: !14)
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !15, size: 64)
!15 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "OuterStruct", file: !1, line: 19, size: 256, elements: !16)
!16 = !{!17, !28, !29, !30}
!17 = !DIDerivedType(tag: DW_TAG_member, name: "out1", scope: !15, file: !1, line: 20, baseType: !18, size: 128)
!18 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "MidStruct", file: !1, line: 13, size: 128, elements: !19)
!19 = !{!20, !21, !27}
!20 = !DIDerivedType(tag: DW_TAG_member, name: "mid1", scope: !18, file: !1, line: 14, baseType: !12, size: 32)
!21 = !DIDerivedType(tag: DW_TAG_member, name: "mid2", scope: !18, file: !1, line: 15, baseType: !22, size: 64, offset: 32)
!22 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "InnerStruct", file: !1, line: 8, size: 64, elements: !23)
!23 = !{!24, !25}
!24 = !DIDerivedType(tag: DW_TAG_member, name: "in1", scope: !22, file: !1, line: 9, baseType: !12, size: 32)
!25 = !DIDerivedType(tag: DW_TAG_member, name: "in2", scope: !22, file: !1, line: 10, baseType: !26, size: 8, offset: 32)
!26 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!27 = !DIDerivedType(tag: DW_TAG_member, name: "mid3", scope: !18, file: !1, line: 16, baseType: !26, size: 8, offset: 96)
!28 = !DIDerivedType(tag: DW_TAG_member, name: "out2", scope: !15, file: !1, line: 21, baseType: !26, size: 8, offset: 128)
!29 = !DIDerivedType(tag: DW_TAG_member, name: "out3", scope: !15, file: !1, line: 22, baseType: !22, size: 64, offset: 160)
!30 = !DIDerivedType(tag: DW_TAG_member, name: "out4", scope: !15, file: !1, line: 23, baseType: !12, size: 32, offset: 224)
!31 = !DILocation(line: 27, column: 22, scope: !9)
!32 = !DILocalVariable(name: "pmid", scope: !9, file: !1, line: 28, type: !33)
!33 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !18, size: 64)
!34 = !DILocation(line: 28, column: 20, scope: !9)
!35 = !DILocalVariable(name: "ptmp", scope: !9, file: !1, line: 29, type: !33)
!36 = !DILocation(line: 29, column: 20, scope: !9)
!37 = !DILocalVariable(name: "itmp", scope: !9, file: !1, line: 30, type: !38)
!38 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !22, size: 64)
!39 = !DILocation(line: 30, column: 22, scope: !9)
!40 = !DILocalVariable(name: "pin", scope: !9, file: !1, line: 31, type: !38)
!41 = !DILocation(line: 31, column: 22, scope: !9)
!42 = !DILocalVariable(name: "s", scope: !9, file: !1, line: 32, type: !15)
!43 = !DILocation(line: 32, column: 21, scope: !9)
!44 = !DILocation(line: 34, column: 7, scope: !9)
!45 = !DILocation(line: 35, column: 12, scope: !9)
!46 = !DILocation(line: 35, column: 7, scope: !9)
!47 = !DILocation(line: 36, column: 10, scope: !9)
!48 = !DILocation(line: 36, column: 16, scope: !9)
!49 = !DILocation(line: 36, column: 7, scope: !9)
!50 = !DILocation(line: 37, column: 2, scope: !9)
!51 = !DILocation(line: 38, column: 2, scope: !9)
!52 = !DILocation(line: 39, column: 2, scope: !9)
!53 = !DILocation(line: 41, column: 11, scope: !9)
!54 = !DILocation(line: 41, column: 16, scope: !9)
!55 = !DILocation(line: 41, column: 6, scope: !9)
!56 = !DILocation(line: 42, column: 10, scope: !9)
!57 = !DILocation(line: 42, column: 16, scope: !9)
!58 = !DILocation(line: 42, column: 21, scope: !9)
!59 = !DILocation(line: 42, column: 7, scope: !9)
!60 = !DILocation(line: 43, column: 2, scope: !9)
!61 = !DILocation(line: 44, column: 2, scope: !9)
!62 = !DILocation(line: 45, column: 2, scope: !9)
!63 = !DILocation(line: 47, column: 2, scope: !9)
!64 = !DILocation(line: 49, column: 2, scope: !9)

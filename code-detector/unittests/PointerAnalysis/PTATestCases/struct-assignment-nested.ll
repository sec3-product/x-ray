; ModuleID = 'basic_c_tests/struct-assignment-nested.c'
source_filename = "basic_c_tests/struct-assignment-nested.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.ArrayStruct = type { i8, %struct.MidArrayStruct, i32* }
%struct.MidArrayStruct = type { [10 x i8], [5 x %struct.InnerArrayStruct] }
%struct.InnerArrayStruct = type { [10 x i32*], [20 x i32*], i8 }

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !9 {
  %1 = alloca i32, align 4
  %2 = alloca %struct.ArrayStruct*, align 8
  %3 = alloca %struct.ArrayStruct, align 8
  %4 = alloca %struct.ArrayStruct, align 8
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  call void @llvm.dbg.declare(metadata %struct.ArrayStruct** %2, metadata !13, metadata !DIExpression()), !dbg !41
  call void @llvm.dbg.declare(metadata %struct.ArrayStruct* %3, metadata !42, metadata !DIExpression()), !dbg !43
  call void @llvm.dbg.declare(metadata %struct.ArrayStruct* %4, metadata !44, metadata !DIExpression()), !dbg !45
  call void @llvm.dbg.declare(metadata i32* %5, metadata !46, metadata !DIExpression()), !dbg !47
  call void @llvm.dbg.declare(metadata i32* %6, metadata !48, metadata !DIExpression()), !dbg !49
  %7 = getelementptr inbounds %struct.ArrayStruct, %struct.ArrayStruct* %3, i32 0, i32 2, !dbg !50
  store i32* %5, i32** %7, align 8, !dbg !51
  store %struct.ArrayStruct* %3, %struct.ArrayStruct** %2, align 8, !dbg !52
  %8 = load %struct.ArrayStruct*, %struct.ArrayStruct** %2, align 8, !dbg !53
  %9 = getelementptr inbounds %struct.ArrayStruct, %struct.ArrayStruct* %8, i32 0, i32 1, !dbg !54
  %10 = getelementptr inbounds %struct.MidArrayStruct, %struct.MidArrayStruct* %9, i32 0, i32 1, !dbg !55
  %11 = getelementptr inbounds [5 x %struct.InnerArrayStruct], [5 x %struct.InnerArrayStruct]* %10, i64 0, i64 3, !dbg !53
  %12 = getelementptr inbounds %struct.InnerArrayStruct, %struct.InnerArrayStruct* %11, i32 0, i32 0, !dbg !56
  %13 = getelementptr inbounds [10 x i32*], [10 x i32*]* %12, i64 0, i64 3, !dbg !53
  store i32* %6, i32** %13, align 8, !dbg !57
  %14 = bitcast %struct.ArrayStruct* %4 to i8*, !dbg !58
  %15 = bitcast %struct.ArrayStruct* %3 to i8*, !dbg !58
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %14, i8* align 8 %15, i64 1272, i1 false), !dbg !58
  %16 = getelementptr inbounds %struct.ArrayStruct, %struct.ArrayStruct* %4, i32 0, i32 2, !dbg !59
  %17 = load i32*, i32** %16, align 8, !dbg !59
  %18 = bitcast i32* %17 to i8*, !dbg !59
  %19 = bitcast i32* %5 to i8*, !dbg !59
  call void @__aser_alias__(i8* %18, i8* %19), !dbg !59
  %20 = getelementptr inbounds %struct.ArrayStruct, %struct.ArrayStruct* %4, i32 0, i32 1, !dbg !60
  %21 = getelementptr inbounds %struct.MidArrayStruct, %struct.MidArrayStruct* %20, i32 0, i32 1, !dbg !60
  %22 = getelementptr inbounds [5 x %struct.InnerArrayStruct], [5 x %struct.InnerArrayStruct]* %21, i64 0, i64 1, !dbg !60
  %23 = getelementptr inbounds %struct.InnerArrayStruct, %struct.InnerArrayStruct* %22, i32 0, i32 0, !dbg !60
  %24 = getelementptr inbounds [10 x i32*], [10 x i32*]* %23, i64 0, i64 1, !dbg !60
  %25 = load i32*, i32** %24, align 8, !dbg !60
  %26 = bitcast i32* %25 to i8*, !dbg !60
  %27 = bitcast i32* %6 to i8*, !dbg !60
  call void @__aser_alias__(i8* %26, i8* %27), !dbg !60
  %28 = getelementptr inbounds %struct.ArrayStruct, %struct.ArrayStruct* %4, i32 0, i32 1, !dbg !61
  %29 = getelementptr inbounds %struct.MidArrayStruct, %struct.MidArrayStruct* %28, i32 0, i32 1, !dbg !61
  %30 = getelementptr inbounds [5 x %struct.InnerArrayStruct], [5 x %struct.InnerArrayStruct]* %29, i64 0, i64 3, !dbg !61
  %31 = getelementptr inbounds %struct.InnerArrayStruct, %struct.InnerArrayStruct* %30, i32 0, i32 0, !dbg !61
  %32 = getelementptr inbounds [10 x i32*], [10 x i32*]* %31, i64 0, i64 2, !dbg !61
  %33 = load i32*, i32** %32, align 8, !dbg !61
  %34 = bitcast i32* %33 to i8*, !dbg !61
  %35 = bitcast i32* %6 to i8*, !dbg !61
  call void @__aser_alias__(i8* %34, i8* %35), !dbg !61
  ret i32 0, !dbg !62
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
!1 = !DIFile(filename: "basic_c_tests/struct-assignment-nested.c", directory: "/home/peiming/Documents/Projects/LLVMRace/TestCases/PTABen")
!2 = !{}
!3 = !{!4}
!4 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!5 = !{i32 2, !"Dwarf Version", i32 4}
!6 = !{i32 2, !"Debug Info Version", i32 3}
!7 = !{i32 1, !"wchar_size", i32 4}
!8 = !{!"clang version 9.0.0 (tags/RELEASE_900/final)"}
!9 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 25, type: !10, scopeLine: 25, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!10 = !DISubroutineType(types: !11)
!11 = !{!12}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DILocalVariable(name: "p", scope: !9, file: !1, line: 26, type: !14)
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !15, size: 64)
!15 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "ArrayStruct", file: !1, line: 19, size: 10176, elements: !16)
!16 = !{!17, !19, !40}
!17 = !DIDerivedType(tag: DW_TAG_member, name: "out2", scope: !15, file: !1, line: 20, baseType: !18, size: 8)
!18 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!19 = !DIDerivedType(tag: DW_TAG_member, name: "out3", scope: !15, file: !1, line: 21, baseType: !20, size: 10048, offset: 64)
!20 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "MidArrayStruct", file: !1, line: 14, size: 10048, elements: !21)
!21 = !{!22, !26}
!22 = !DIDerivedType(tag: DW_TAG_member, name: "mid1", scope: !20, file: !1, line: 15, baseType: !23, size: 80)
!23 = !DICompositeType(tag: DW_TAG_array_type, baseType: !18, size: 80, elements: !24)
!24 = !{!25}
!25 = !DISubrange(count: 10)
!26 = !DIDerivedType(tag: DW_TAG_member, name: "mid2", scope: !20, file: !1, line: 16, baseType: !27, size: 9920, offset: 128)
!27 = !DICompositeType(tag: DW_TAG_array_type, baseType: !28, size: 9920, elements: !38)
!28 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "InnerArrayStruct", file: !1, line: 8, size: 1984, elements: !29)
!29 = !{!30, !33, !37}
!30 = !DIDerivedType(tag: DW_TAG_member, name: "in1", scope: !28, file: !1, line: 9, baseType: !31, size: 640)
!31 = !DICompositeType(tag: DW_TAG_array_type, baseType: !32, size: 640, elements: !24)
!32 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!33 = !DIDerivedType(tag: DW_TAG_member, name: "in2", scope: !28, file: !1, line: 10, baseType: !34, size: 1280, offset: 640)
!34 = !DICompositeType(tag: DW_TAG_array_type, baseType: !32, size: 1280, elements: !35)
!35 = !{!36}
!36 = !DISubrange(count: 20)
!37 = !DIDerivedType(tag: DW_TAG_member, name: "in3", scope: !28, file: !1, line: 11, baseType: !18, size: 8, offset: 1920)
!38 = !{!39}
!39 = !DISubrange(count: 5)
!40 = !DIDerivedType(tag: DW_TAG_member, name: "out4", scope: !15, file: !1, line: 22, baseType: !32, size: 64, offset: 10112)
!41 = !DILocation(line: 26, column: 22, scope: !9)
!42 = !DILocalVariable(name: "s1", scope: !9, file: !1, line: 27, type: !15)
!43 = !DILocation(line: 27, column: 21, scope: !9)
!44 = !DILocalVariable(name: "s2", scope: !9, file: !1, line: 27, type: !15)
!45 = !DILocation(line: 27, column: 25, scope: !9)
!46 = !DILocalVariable(name: "x", scope: !9, file: !1, line: 28, type: !12)
!47 = !DILocation(line: 28, column: 6, scope: !9)
!48 = !DILocalVariable(name: "y", scope: !9, file: !1, line: 28, type: !12)
!49 = !DILocation(line: 28, column: 9, scope: !9)
!50 = !DILocation(line: 30, column: 5, scope: !9)
!51 = !DILocation(line: 30, column: 10, scope: !9)
!52 = !DILocation(line: 31, column: 4, scope: !9)
!53 = !DILocation(line: 32, column: 2, scope: !9)
!54 = !DILocation(line: 32, column: 5, scope: !9)
!55 = !DILocation(line: 32, column: 10, scope: !9)
!56 = !DILocation(line: 32, column: 18, scope: !9)
!57 = !DILocation(line: 32, column: 25, scope: !9)
!58 = !DILocation(line: 34, column: 7, scope: !9)
!59 = !DILocation(line: 36, column: 2, scope: !9)
!60 = !DILocation(line: 37, column: 2, scope: !9)
!61 = !DILocation(line: 38, column: 2, scope: !9)
!62 = !DILocation(line: 40, column: 2, scope: !9)

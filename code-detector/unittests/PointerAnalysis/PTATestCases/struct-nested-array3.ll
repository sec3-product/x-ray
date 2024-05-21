; ModuleID = 'basic_c_tests/struct-nested-array3.c'
source_filename = "basic_c_tests/struct-nested-array3.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.ArrayStruct = type { i32, i8, %struct.MidArrayStruct, i32 }
%struct.MidArrayStruct = type { i8, [5 x %struct.InnerArrayStruct], [20 x double] }
%struct.InnerArrayStruct = type { [10 x i32], i8, double }

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !9 {
  %1 = alloca i32, align 4
  %2 = alloca %struct.ArrayStruct*, align 8
  %3 = alloca %struct.ArrayStruct, align 8
  store i32 0, i32* %1, align 4
  call void @llvm.dbg.declare(metadata %struct.ArrayStruct** %2, metadata !13, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.declare(metadata %struct.ArrayStruct* %3, metadata !43, metadata !DIExpression()), !dbg !44
  store %struct.ArrayStruct* %3, %struct.ArrayStruct** %2, align 8, !dbg !45
  %4 = load %struct.ArrayStruct*, %struct.ArrayStruct** %2, align 8, !dbg !46
  %5 = getelementptr inbounds %struct.ArrayStruct, %struct.ArrayStruct* %4, i32 0, i32 3, !dbg !46
  %6 = bitcast i32* %5 to i8*, !dbg !46
  %7 = getelementptr inbounds %struct.ArrayStruct, %struct.ArrayStruct* %3, i32 0, i32 3, !dbg !46
  %8 = bitcast i32* %7 to i8*, !dbg !46
  call void @__aser_alias__(i8* %6, i8* %8), !dbg !46
  %9 = load %struct.ArrayStruct*, %struct.ArrayStruct** %2, align 8, !dbg !47
  %10 = getelementptr inbounds %struct.ArrayStruct, %struct.ArrayStruct* %9, i32 0, i32 2, !dbg !47
  %11 = getelementptr inbounds %struct.MidArrayStruct, %struct.MidArrayStruct* %10, i32 0, i32 1, !dbg !47
  %12 = getelementptr inbounds [5 x %struct.InnerArrayStruct], [5 x %struct.InnerArrayStruct]* %11, i64 0, i64 1, !dbg !47
  %13 = getelementptr inbounds %struct.InnerArrayStruct, %struct.InnerArrayStruct* %12, i32 0, i32 0, !dbg !47
  %14 = getelementptr inbounds [10 x i32], [10 x i32]* %13, i64 0, i64 3, !dbg !47
  %15 = bitcast i32* %14 to i8*, !dbg !47
  %16 = getelementptr inbounds %struct.ArrayStruct, %struct.ArrayStruct* %3, i32 0, i32 2, !dbg !47
  %17 = getelementptr inbounds %struct.MidArrayStruct, %struct.MidArrayStruct* %16, i32 0, i32 1, !dbg !47
  %18 = getelementptr inbounds [5 x %struct.InnerArrayStruct], [5 x %struct.InnerArrayStruct]* %17, i64 0, i64 2, !dbg !47
  %19 = bitcast %struct.InnerArrayStruct* %18 to i8*, !dbg !47
  call void @__aser_alias__(i8* %15, i8* %19), !dbg !47
  %20 = load %struct.ArrayStruct*, %struct.ArrayStruct** %2, align 8, !dbg !48
  %21 = getelementptr inbounds %struct.ArrayStruct, %struct.ArrayStruct* %20, i32 0, i32 2, !dbg !48
  %22 = getelementptr inbounds %struct.MidArrayStruct, %struct.MidArrayStruct* %21, i32 0, i32 1, !dbg !48
  %23 = getelementptr inbounds [5 x %struct.InnerArrayStruct], [5 x %struct.InnerArrayStruct]* %22, i64 0, i64 1, !dbg !48
  %24 = bitcast %struct.InnerArrayStruct* %23 to i8*, !dbg !48
  %25 = load %struct.ArrayStruct*, %struct.ArrayStruct** %2, align 8, !dbg !48
  %26 = getelementptr inbounds %struct.ArrayStruct, %struct.ArrayStruct* %25, i32 0, i32 2, !dbg !48
  %27 = getelementptr inbounds %struct.MidArrayStruct, %struct.MidArrayStruct* %26, i32 0, i32 1, !dbg !48
  %28 = getelementptr inbounds [5 x %struct.InnerArrayStruct], [5 x %struct.InnerArrayStruct]* %27, i64 0, i64 2, !dbg !48
  %29 = bitcast %struct.InnerArrayStruct* %28 to i8*, !dbg !48
  call void @__aser_alias__(i8* %24, i8* %29), !dbg !48
  %30 = load %struct.ArrayStruct*, %struct.ArrayStruct** %2, align 8, !dbg !49
  %31 = getelementptr inbounds %struct.ArrayStruct, %struct.ArrayStruct* %30, i32 0, i32 2, !dbg !49
  %32 = getelementptr inbounds %struct.MidArrayStruct, %struct.MidArrayStruct* %31, i32 0, i32 1, !dbg !49
  %33 = getelementptr inbounds [5 x %struct.InnerArrayStruct], [5 x %struct.InnerArrayStruct]* %32, i64 0, i64 3, !dbg !49
  %34 = getelementptr inbounds %struct.InnerArrayStruct, %struct.InnerArrayStruct* %33, i32 0, i32 2, !dbg !49
  %35 = bitcast double* %34 to i8*, !dbg !49
  %36 = getelementptr inbounds %struct.ArrayStruct, %struct.ArrayStruct* %3, i32 0, i32 2, !dbg !49
  %37 = getelementptr inbounds %struct.MidArrayStruct, %struct.MidArrayStruct* %36, i32 0, i32 2, !dbg !49
  %38 = getelementptr inbounds [20 x double], [20 x double]* %37, i64 0, i64 2, !dbg !49
  %39 = bitcast double* %38 to i8*, !dbg !49
  call void @__aser_no_alias__(i8* %35, i8* %39), !dbg !49
  %40 = load %struct.ArrayStruct*, %struct.ArrayStruct** %2, align 8, !dbg !50
  %41 = getelementptr inbounds %struct.ArrayStruct, %struct.ArrayStruct* %40, i32 0, i32 2, !dbg !50
  %42 = getelementptr inbounds %struct.MidArrayStruct, %struct.MidArrayStruct* %41, i32 0, i32 1, !dbg !50
  %43 = getelementptr inbounds [5 x %struct.InnerArrayStruct], [5 x %struct.InnerArrayStruct]* %42, i64 0, i64 0, !dbg !50
  %44 = bitcast %struct.InnerArrayStruct* %43 to i8*, !dbg !50
  %45 = getelementptr inbounds %struct.ArrayStruct, %struct.ArrayStruct* %3, i32 0, i32 3, !dbg !50
  %46 = bitcast i32* %45 to i8*, !dbg !50
  call void @__aser_no_alias__(i8* %44, i8* %46), !dbg !50
  ret i32 0, !dbg !51
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
!1 = !DIFile(filename: "basic_c_tests/struct-nested-array3.c", directory: "/home/peiming/Documents/Projects/LLVMRace/TestCases/PTABen")
!2 = !{}
!3 = !{!4}
!4 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!5 = !{i32 2, !"Dwarf Version", i32 4}
!6 = !{i32 2, !"Debug Info Version", i32 3}
!7 = !{i32 1, !"wchar_size", i32 4}
!8 = !{!"clang version 9.0.0 (tags/RELEASE_900/final)"}
!9 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 27, type: !10, scopeLine: 27, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!10 = !DISubroutineType(types: !11)
!11 = !{!12}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DILocalVariable(name: "p", scope: !9, file: !1, line: 28, type: !14)
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !15, size: 64)
!15 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "ArrayStruct", file: !1, line: 20, size: 3712, elements: !16)
!16 = !{!17, !18, !20, !41}
!17 = !DIDerivedType(tag: DW_TAG_member, name: "out1", scope: !15, file: !1, line: 21, baseType: !12, size: 32)
!18 = !DIDerivedType(tag: DW_TAG_member, name: "out2", scope: !15, file: !1, line: 22, baseType: !19, size: 8, offset: 32)
!19 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!20 = !DIDerivedType(tag: DW_TAG_member, name: "out3", scope: !15, file: !1, line: 23, baseType: !21, size: 3584, offset: 64)
!21 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "MidArrayStruct", file: !1, line: 14, size: 3584, elements: !22)
!22 = !{!23, !24, !37}
!23 = !DIDerivedType(tag: DW_TAG_member, name: "mid1", scope: !21, file: !1, line: 15, baseType: !19, size: 8)
!24 = !DIDerivedType(tag: DW_TAG_member, name: "mid2", scope: !21, file: !1, line: 16, baseType: !25, size: 2240, offset: 64)
!25 = !DICompositeType(tag: DW_TAG_array_type, baseType: !26, size: 2240, elements: !35)
!26 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "InnerArrayStruct", file: !1, line: 8, size: 448, elements: !27)
!27 = !{!28, !32, !33}
!28 = !DIDerivedType(tag: DW_TAG_member, name: "in1", scope: !26, file: !1, line: 9, baseType: !29, size: 320)
!29 = !DICompositeType(tag: DW_TAG_array_type, baseType: !12, size: 320, elements: !30)
!30 = !{!31}
!31 = !DISubrange(count: 10)
!32 = !DIDerivedType(tag: DW_TAG_member, name: "in2", scope: !26, file: !1, line: 10, baseType: !19, size: 8, offset: 320)
!33 = !DIDerivedType(tag: DW_TAG_member, name: "in3", scope: !26, file: !1, line: 11, baseType: !34, size: 64, offset: 384)
!34 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!35 = !{!36}
!36 = !DISubrange(count: 5)
!37 = !DIDerivedType(tag: DW_TAG_member, name: "mid3", scope: !21, file: !1, line: 17, baseType: !38, size: 1280, offset: 2304)
!38 = !DICompositeType(tag: DW_TAG_array_type, baseType: !34, size: 1280, elements: !39)
!39 = !{!40}
!40 = !DISubrange(count: 20)
!41 = !DIDerivedType(tag: DW_TAG_member, name: "out4", scope: !15, file: !1, line: 24, baseType: !12, size: 32, offset: 3648)
!42 = !DILocation(line: 28, column: 22, scope: !9)
!43 = !DILocalVariable(name: "s", scope: !9, file: !1, line: 29, type: !15)
!44 = !DILocation(line: 29, column: 21, scope: !9)
!45 = !DILocation(line: 31, column: 4, scope: !9)
!46 = !DILocation(line: 33, column: 2, scope: !9)
!47 = !DILocation(line: 35, column: 2, scope: !9)
!48 = !DILocation(line: 36, column: 2, scope: !9)
!49 = !DILocation(line: 37, column: 2, scope: !9)
!50 = !DILocation(line: 38, column: 2, scope: !9)
!51 = !DILocation(line: 40, column: 2, scope: !9)

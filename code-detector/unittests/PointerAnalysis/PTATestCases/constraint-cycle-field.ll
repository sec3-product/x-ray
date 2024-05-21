; ModuleID = 'basic_c_tests/constraint-cycle-field.c'
source_filename = "basic_c_tests/constraint-cycle-field.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.MyStruct = type { i32*, %struct.MyStruct* }

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !16 {
  %1 = alloca i32, align 4
  %2 = alloca %struct.MyStruct*, align 8
  %3 = alloca i32, align 4
  %4 = alloca %struct.MyStruct*, align 8
  store i32 0, i32* %1, align 4
  call void @llvm.dbg.declare(metadata %struct.MyStruct** %2, metadata !19, metadata !DIExpression()), !dbg !20
  %5 = call noalias i8* @malloc(i64 16) #4, !dbg !21
  %6 = bitcast i8* %5 to %struct.MyStruct*, !dbg !22
  store %struct.MyStruct* %6, %struct.MyStruct** %2, align 8, !dbg !20
  call void @llvm.dbg.declare(metadata i32* %3, metadata !23, metadata !DIExpression()), !dbg !24
  store i32 10, i32* %3, align 4, !dbg !24
  br label %7, !dbg !25

7:                                                ; preds = %10, %0
  %8 = load i32, i32* %3, align 4, !dbg !26
  %9 = icmp ne i32 %8, 0, !dbg !25
  br i1 %9, label %10, label %24, !dbg !25

10:                                               ; preds = %7
  %11 = call noalias i8* @malloc(i64 16) #4, !dbg !27
  %12 = bitcast i8* %11 to %struct.MyStruct*, !dbg !29
  %13 = load %struct.MyStruct*, %struct.MyStruct** %2, align 8, !dbg !30
  %14 = getelementptr inbounds %struct.MyStruct, %struct.MyStruct* %13, i32 0, i32 1, !dbg !31
  store %struct.MyStruct* %12, %struct.MyStruct** %14, align 8, !dbg !32
  %15 = call noalias i8* @malloc(i64 4) #4, !dbg !33
  %16 = bitcast i8* %15 to i32*, !dbg !34
  %17 = load %struct.MyStruct*, %struct.MyStruct** %2, align 8, !dbg !35
  %18 = getelementptr inbounds %struct.MyStruct, %struct.MyStruct* %17, i32 0, i32 1, !dbg !36
  %19 = load %struct.MyStruct*, %struct.MyStruct** %18, align 8, !dbg !36
  %20 = getelementptr inbounds %struct.MyStruct, %struct.MyStruct* %19, i32 0, i32 0, !dbg !37
  store i32* %16, i32** %20, align 8, !dbg !38
  %21 = load %struct.MyStruct*, %struct.MyStruct** %2, align 8, !dbg !39
  %22 = getelementptr inbounds %struct.MyStruct, %struct.MyStruct* %21, i32 0, i32 1, !dbg !40
  %23 = load %struct.MyStruct*, %struct.MyStruct** %22, align 8, !dbg !40
  store %struct.MyStruct* %23, %struct.MyStruct** %2, align 8, !dbg !41
  br label %7, !dbg !25, !llvm.loop !42

24:                                               ; preds = %7
  call void @llvm.dbg.declare(metadata %struct.MyStruct** %4, metadata !44, metadata !DIExpression()), !dbg !45
  %25 = load %struct.MyStruct*, %struct.MyStruct** %2, align 8, !dbg !46
  store %struct.MyStruct* %25, %struct.MyStruct** %4, align 8, !dbg !45
  %26 = load %struct.MyStruct*, %struct.MyStruct** %4, align 8, !dbg !47
  %27 = getelementptr inbounds %struct.MyStruct, %struct.MyStruct* %26, i32 0, i32 1, !dbg !47
  %28 = load %struct.MyStruct*, %struct.MyStruct** %27, align 8, !dbg !47
  %29 = bitcast %struct.MyStruct* %28 to i8*, !dbg !47
  %30 = load %struct.MyStruct*, %struct.MyStruct** %2, align 8, !dbg !47
  %31 = getelementptr inbounds %struct.MyStruct, %struct.MyStruct* %30, i32 0, i32 1, !dbg !47
  %32 = load %struct.MyStruct*, %struct.MyStruct** %31, align 8, !dbg !47
  %33 = getelementptr inbounds %struct.MyStruct, %struct.MyStruct* %32, i32 0, i32 1, !dbg !47
  %34 = load %struct.MyStruct*, %struct.MyStruct** %33, align 8, !dbg !47
  %35 = bitcast %struct.MyStruct* %34 to i8*, !dbg !47
  call void @__aser_alias__(i8* %29, i8* %35), !dbg !47
  %36 = load %struct.MyStruct*, %struct.MyStruct** %4, align 8, !dbg !48
  %37 = getelementptr inbounds %struct.MyStruct, %struct.MyStruct* %36, i32 0, i32 0, !dbg !48
  %38 = load i32*, i32** %37, align 8, !dbg !48
  %39 = bitcast i32* %38 to i8*, !dbg !48
  %40 = load %struct.MyStruct*, %struct.MyStruct** %2, align 8, !dbg !48
  %41 = getelementptr inbounds %struct.MyStruct, %struct.MyStruct* %40, i32 0, i32 1, !dbg !48
  %42 = load %struct.MyStruct*, %struct.MyStruct** %41, align 8, !dbg !48
  %43 = getelementptr inbounds %struct.MyStruct, %struct.MyStruct* %42, i32 0, i32 0, !dbg !48
  %44 = load i32*, i32** %43, align 8, !dbg !48
  %45 = bitcast i32* %44 to i8*, !dbg !48
  call void @__aser_alias__(i8* %39, i8* %45), !dbg !48
  ret i32 0, !dbg !49
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind
declare dso_local noalias i8* @malloc(i64) #2

declare dso_local void @__aser_alias__(i8*, i8*) #3

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!12, !13, !14}
!llvm.ident = !{!15}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 9.0.0 (tags/RELEASE_900/final)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, nameTableKind: None)
!1 = !DIFile(filename: "basic_c_tests/constraint-cycle-field.c", directory: "/home/peiming/Documents/Projects/LLVMRace/TestCases/PTABen")
!2 = !{}
!3 = !{!4, !8, !11}
!4 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !5, size: 64)
!5 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "MyStruct", file: !1, line: 10, size: 128, elements: !6)
!6 = !{!7, !10}
!7 = !DIDerivedType(tag: DW_TAG_member, name: "f1", scope: !5, file: !1, line: 11, baseType: !8, size: 64)
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 64)
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !DIDerivedType(tag: DW_TAG_member, name: "next", scope: !5, file: !1, line: 12, baseType: !4, size: 64, offset: 64)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!12 = !{i32 2, !"Dwarf Version", i32 4}
!13 = !{i32 2, !"Debug Info Version", i32 3}
!14 = !{i32 1, !"wchar_size", i32 4}
!15 = !{!"clang version 9.0.0 (tags/RELEASE_900/final)"}
!16 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 15, type: !17, scopeLine: 15, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!17 = !DISubroutineType(types: !18)
!18 = !{!9}
!19 = !DILocalVariable(name: "p", scope: !16, file: !1, line: 16, type: !4)
!20 = !DILocation(line: 16, column: 20, scope: !16)
!21 = !DILocation(line: 16, column: 44, scope: !16)
!22 = !DILocation(line: 16, column: 24, scope: !16)
!23 = !DILocalVariable(name: "num", scope: !16, file: !1, line: 17, type: !9)
!24 = !DILocation(line: 17, column: 6, scope: !16)
!25 = !DILocation(line: 18, column: 2, scope: !16)
!26 = !DILocation(line: 18, column: 9, scope: !16)
!27 = !DILocation(line: 19, column: 33, scope: !28)
!28 = distinct !DILexicalBlock(scope: !16, file: !1, line: 18, column: 14)
!29 = !DILocation(line: 19, column: 13, scope: !28)
!30 = !DILocation(line: 19, column: 3, scope: !28)
!31 = !DILocation(line: 19, column: 6, scope: !28)
!32 = !DILocation(line: 19, column: 11, scope: !28)
!33 = !DILocation(line: 20, column: 25, scope: !28)
!34 = !DILocation(line: 20, column: 17, scope: !28)
!35 = !DILocation(line: 20, column: 3, scope: !28)
!36 = !DILocation(line: 20, column: 6, scope: !28)
!37 = !DILocation(line: 20, column: 12, scope: !28)
!38 = !DILocation(line: 20, column: 15, scope: !28)
!39 = !DILocation(line: 21, column: 7, scope: !28)
!40 = !DILocation(line: 21, column: 10, scope: !28)
!41 = !DILocation(line: 21, column: 5, scope: !28)
!42 = distinct !{!42, !25, !43}
!43 = !DILocation(line: 22, column: 2, scope: !16)
!44 = !DILocalVariable(name: "q", scope: !16, file: !1, line: 23, type: !4)
!45 = !DILocation(line: 23, column: 19, scope: !16)
!46 = !DILocation(line: 23, column: 23, scope: !16)
!47 = !DILocation(line: 24, column: 2, scope: !16)
!48 = !DILocation(line: 25, column: 2, scope: !16)
!49 = !DILocation(line: 26, column: 2, scope: !16)

; ModuleID = 'basic_c_tests/funptr-struct.c'
source_filename = "basic_c_tests/funptr-struct.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.MYFILE = type { i32 (i32*)* }

@.str = private unnamed_addr constant [15 x i8] c"Executing bar\0A\00", align 1
@g = common dso_local global i32 0, align 4, !dbg !0
@__const.my_vsnprintf.pts = private unnamed_addr constant %struct.MYFILE { i32 (i32*)* @my_sn_write }, align 8
@.str.1 = private unnamed_addr constant [23 x i8] c"Executing my_sn_write\0A\00", align 1

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @my_vfprintf(%struct.MYFILE*) #0 !dbg !13 {
  %2 = alloca %struct.MYFILE*, align 8
  %3 = alloca i32*, align 8
  store %struct.MYFILE* %0, %struct.MYFILE** %2, align 8
  call void @llvm.dbg.declare(metadata %struct.MYFILE** %2, metadata !24, metadata !DIExpression()), !dbg !25
  %4 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([15 x i8], [15 x i8]* @.str, i64 0, i64 0)), !dbg !26
  call void @llvm.dbg.declare(metadata i32** %3, metadata !27, metadata !DIExpression()), !dbg !28
  store i32* @g, i32** %3, align 8, !dbg !28
  %5 = load %struct.MYFILE*, %struct.MYFILE** %2, align 8, !dbg !29
  %6 = getelementptr inbounds %struct.MYFILE, %struct.MYFILE* %5, i32 0, i32 0, !dbg !30
  %7 = load i32 (i32*)*, i32 (i32*)** %6, align 8, !dbg !30
  %8 = load i32*, i32** %3, align 8, !dbg !31
  %9 = call i32 %7(i32* %8), !dbg !29
  ret void, !dbg !32
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare dso_local i32 @printf(i8*, ...) #2

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @my_vsnprintf() #0 !dbg !33 {
  %1 = alloca %struct.MYFILE, align 8
  call void @llvm.dbg.declare(metadata %struct.MYFILE* %1, metadata !36, metadata !DIExpression()), !dbg !37
  %2 = bitcast %struct.MYFILE* %1 to i8*, !dbg !37
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %2, i8* align 8 bitcast (%struct.MYFILE* @__const.my_vsnprintf.pts to i8*), i64 8, i1 false), !dbg !37
  call void @my_vfprintf(%struct.MYFILE* %1), !dbg !38
  ret i32 0, !dbg !39
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i32 @my_sn_write(i32*) #0 !dbg !40 {
  %2 = alloca i32*, align 8
  store i32* %0, i32** %2, align 8
  call void @llvm.dbg.declare(metadata i32** %2, metadata !41, metadata !DIExpression()), !dbg !42
  %3 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.1, i64 0, i64 0)), !dbg !43
  %4 = load i32*, i32** %2, align 8, !dbg !44
  %5 = bitcast i32* %4 to i8*, !dbg !44
  call void @__aser_alias__(i8* bitcast (i32* @g to i8*), i8* %5), !dbg !44
  ret i32 0, !dbg !45
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1 immarg) #3

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !46 {
  %1 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  %2 = call i32 @my_vsnprintf(), !dbg !47
  ret i32 0, !dbg !48
}

declare dso_local void @__aser_alias__(i8*, i8*) #2

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { argmemonly nounwind }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!9, !10, !11}
!llvm.ident = !{!12}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "g", scope: !2, file: !3, line: 3, type: !8, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 9.0.0 (tags/RELEASE_900/final)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !5, globals: !7, nameTableKind: None)
!3 = !DIFile(filename: "basic_c_tests/funptr-struct.c", directory: "/home/peiming/Documents/Projects/LLVMRace/TestCases/PTABen")
!4 = !{}
!5 = !{!6}
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!7 = !{!0}
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !{i32 2, !"Dwarf Version", i32 4}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{i32 1, !"wchar_size", i32 4}
!12 = !{!"clang version 9.0.0 (tags/RELEASE_900/final)"}
!13 = distinct !DISubprogram(name: "my_vfprintf", scope: !3, file: !3, line: 14, type: !14, scopeLine: 14, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!14 = !DISubroutineType(types: !15)
!15 = !{null, !16}
!16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !17, size: 64)
!17 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "MYFILE", file: !3, line: 10, size: 64, elements: !18)
!18 = !{!19}
!19 = !DIDerivedType(tag: DW_TAG_member, name: "pt", scope: !17, file: !3, line: 11, baseType: !20, size: 64)
!20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !21, size: 64)
!21 = !DISubroutineType(types: !22)
!22 = !{!8, !23}
!23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 64)
!24 = !DILocalVariable(name: "pts", arg: 1, scope: !13, file: !3, line: 14, type: !16)
!25 = !DILocation(line: 14, column: 33, scope: !13)
!26 = !DILocation(line: 15, column: 4, scope: !13)
!27 = !DILocalVariable(name: "p", scope: !13, file: !3, line: 16, type: !23)
!28 = !DILocation(line: 16, column: 9, scope: !13)
!29 = !DILocation(line: 17, column: 6, scope: !13)
!30 = !DILocation(line: 17, column: 11, scope: !13)
!31 = !DILocation(line: 17, column: 14, scope: !13)
!32 = !DILocation(line: 18, column: 1, scope: !13)
!33 = distinct !DISubprogram(name: "my_vsnprintf", scope: !3, file: !3, line: 20, type: !34, scopeLine: 20, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!34 = !DISubroutineType(types: !35)
!35 = !{!8}
!36 = !DILocalVariable(name: "pts", scope: !33, file: !3, line: 21, type: !17)
!37 = !DILocation(line: 21, column: 18, scope: !33)
!38 = !DILocation(line: 22, column: 6, scope: !33)
!39 = !DILocation(line: 23, column: 8, scope: !33)
!40 = distinct !DISubprogram(name: "my_sn_write", scope: !3, file: !3, line: 4, type: !21, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !4)
!41 = !DILocalVariable(name: "p", arg: 1, scope: !40, file: !3, line: 4, type: !23)
!42 = !DILocation(line: 4, column: 29, scope: !40)
!43 = !DILocation(line: 5, column: 4, scope: !40)
!44 = !DILocation(line: 6, column: 4, scope: !40)
!45 = !DILocation(line: 7, column: 6, scope: !40)
!46 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 26, type: !34, scopeLine: 26, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!47 = !DILocation(line: 27, column: 4, scope: !46)
!48 = !DILocation(line: 28, column: 6, scope: !46)

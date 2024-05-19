; ModuleID = 'basic_c_tests/funptr-nested-call.c'
source_filename = "basic_c_tests/funptr-nested-call.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.str = private unnamed_addr constant [3 x i8] c"f\0A\00", align 1
@.str.1 = private unnamed_addr constant [3 x i8] c"g\0A\00", align 1
@p = common dso_local global void (...)* null, align 8, !dbg !0
@fptr = common dso_local global void (void (...)*)* null, align 8, !dbg !6

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @f() #0 !dbg !18 {
  %1 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i64 0, i64 0)), !dbg !21
  ret void, !dbg !22
}

declare dso_local i32 @printf(i8*, ...) #1

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @g() #0 !dbg !23 {
  %1 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.1, i64 0, i64 0)), !dbg !24
  ret void, !dbg !25
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @fake_fun(void (...)*) #0 !dbg !26 {
  %2 = alloca void (...)*, align 8
  store void (...)* %0, void (...)** %2, align 8
  call void @llvm.dbg.declare(metadata void (...)** %2, metadata !27, metadata !DIExpression()), !dbg !28
  %3 = load void (...)*, void (...)** %2, align 8, !dbg !29
  store void (...)* %3, void (...)** @p, align 8, !dbg !30
  %4 = load void (...)*, void (...)** @p, align 8, !dbg !31
  call void (...) %4(), !dbg !31
  ret void, !dbg !32
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @real_fun(void (...)*) #0 !dbg !33 {
  %2 = alloca void (...)*, align 8
  store void (...)* %0, void (...)** %2, align 8
  call void @llvm.dbg.declare(metadata void (...)** %2, metadata !34, metadata !DIExpression()), !dbg !35
  %3 = load void (...)*, void (...)** %2, align 8, !dbg !36
  store void (...)* %3, void (...)** @p, align 8, !dbg !37
  %4 = load void (...)*, void (...)** @p, align 8, !dbg !38
  call void (...) %4(), !dbg !38
  ret void, !dbg !39
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @set(void (...)*) #0 !dbg !40 {
  %2 = alloca void (...)*, align 8
  store void (...)* %0, void (...)** %2, align 8
  call void @llvm.dbg.declare(metadata void (...)** %2, metadata !41, metadata !DIExpression()), !dbg !42
  %3 = load void (...)*, void (...)** %2, align 8, !dbg !43
  %4 = bitcast void (...)* %3 to void (void (...)*)*, !dbg !43
  store void (void (...)*)* %4, void (void (...)*)** @fptr, align 8, !dbg !44
  ret void, !dbg !45
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main(i32, i8**) #0 !dbg !46 {
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i8**, align 8
  store i32 0, i32* %3, align 4
  store i32 %0, i32* %4, align 4
  call void @llvm.dbg.declare(metadata i32* %4, metadata !53, metadata !DIExpression()), !dbg !54
  store i8** %1, i8*** %5, align 8
  call void @llvm.dbg.declare(metadata i8*** %5, metadata !55, metadata !DIExpression()), !dbg !56
  call void @set(void (...)* bitcast (void (void (...)*)* @fake_fun to void (...)*)), !dbg !57
  call void @set(void (...)* bitcast (void (void (...)*)* @real_fun to void (...)*)), !dbg !58
  %6 = load void (void (...)*)*, void (void (...)*)** @fptr, align 8, !dbg !59
  call void %6(void (...)* bitcast (void ()* @f to void (...)*)), !dbg !59
  %7 = load void (void (...)*)*, void (void (...)*)** @fptr, align 8, !dbg !60
  call void %7(void (...)* bitcast (void ()* @g to void (...)*)), !dbg !60
  ret i32 0, !dbg !61
}

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!14, !15, !16}
!llvm.ident = !{!17}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "p", scope: !2, file: !3, line: 6, type: !11, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 9.0.0 (tags/RELEASE_900/final)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, nameTableKind: None)
!3 = !DIFile(filename: "basic_c_tests/funptr-nested-call.c", directory: "/home/peiming/Documents/Projects/LLVMRace/TestCases/PTABen")
!4 = !{}
!5 = !{!0, !6}
!6 = !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!7 = distinct !DIGlobalVariable(name: "fptr", scope: !2, file: !3, line: 18, type: !8, isLocal: false, isDefinition: true)
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 64)
!9 = !DISubroutineType(types: !10)
!10 = !{null, !11}
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!12 = !DISubroutineType(types: !13)
!13 = !{null, null}
!14 = !{i32 2, !"Dwarf Version", i32 4}
!15 = !{i32 2, !"Debug Info Version", i32 3}
!16 = !{i32 1, !"wchar_size", i32 4}
!17 = !{!"clang version 9.0.0 (tags/RELEASE_900/final)"}
!18 = distinct !DISubprogram(name: "f", scope: !3, file: !3, line: 4, type: !19, scopeLine: 4, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!19 = !DISubroutineType(types: !20)
!20 = !{null}
!21 = !DILocation(line: 4, column: 12, scope: !18)
!22 = !DILocation(line: 4, column: 27, scope: !18)
!23 = distinct !DISubprogram(name: "g", scope: !3, file: !3, line: 5, type: !19, scopeLine: 5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!24 = !DILocation(line: 5, column: 12, scope: !23)
!25 = !DILocation(line: 5, column: 27, scope: !23)
!26 = distinct !DISubprogram(name: "fake_fun", scope: !3, file: !3, line: 8, type: !9, scopeLine: 8, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!27 = !DILocalVariable(name: "a", arg: 1, scope: !26, file: !3, line: 8, type: !11)
!28 = !DILocation(line: 8, column: 23, scope: !26)
!29 = !DILocation(line: 9, column: 7, scope: !26)
!30 = !DILocation(line: 9, column: 5, scope: !26)
!31 = !DILocation(line: 10, column: 3, scope: !26)
!32 = !DILocation(line: 11, column: 1, scope: !26)
!33 = distinct !DISubprogram(name: "real_fun", scope: !3, file: !3, line: 13, type: !9, scopeLine: 13, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!34 = !DILocalVariable(name: "a", arg: 1, scope: !33, file: !3, line: 13, type: !11)
!35 = !DILocation(line: 13, column: 23, scope: !33)
!36 = !DILocation(line: 14, column: 7, scope: !33)
!37 = !DILocation(line: 14, column: 5, scope: !33)
!38 = !DILocation(line: 15, column: 3, scope: !33)
!39 = !DILocation(line: 16, column: 1, scope: !33)
!40 = distinct !DISubprogram(name: "set", scope: !3, file: !3, line: 20, type: !9, scopeLine: 20, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!41 = !DILocalVariable(name: "src", arg: 1, scope: !40, file: !3, line: 20, type: !11)
!42 = !DILocation(line: 20, column: 17, scope: !40)
!43 = !DILocation(line: 21, column: 10, scope: !40)
!44 = !DILocation(line: 21, column: 8, scope: !40)
!45 = !DILocation(line: 22, column: 1, scope: !40)
!46 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 24, type: !47, scopeLine: 25, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!47 = !DISubroutineType(types: !48)
!48 = !{!49, !49, !50}
!49 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!50 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !51, size: 64)
!51 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !52, size: 64)
!52 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!53 = !DILocalVariable(name: "argc", arg: 1, scope: !46, file: !3, line: 24, type: !49)
!54 = !DILocation(line: 24, column: 14, scope: !46)
!55 = !DILocalVariable(name: "argv", arg: 2, scope: !46, file: !3, line: 24, type: !50)
!56 = !DILocation(line: 24, column: 27, scope: !46)
!57 = !DILocation(line: 26, column: 3, scope: !46)
!58 = !DILocation(line: 27, column: 3, scope: !46)
!59 = !DILocation(line: 29, column: 3, scope: !46)
!60 = !DILocation(line: 31, column: 3, scope: !46)
!61 = !DILocation(line: 33, column: 3, scope: !46)

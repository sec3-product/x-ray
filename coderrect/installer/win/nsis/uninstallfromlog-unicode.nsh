!include "TextFunc.nsh"
!include "LogicLib.nsh"

!define UnStrLoc "!insertmacro UnStrLoc"
!macro UnStrLoc ResultVar String SubString StartPoint
  Push "${String}"
  Push "${SubString}"
  Push "${StartPoint}"
  Call un.StrLoc
  Pop "${ResultVar}"
!macroend

Function un.StrLoc
/*After this point:
  ------------------------------------------
   $R0 = StartPoint (input)
   $R1 = SubString (input)
   $R2 = String (input)
   $R3 = SubStringLen (temp)
   $R4 = StrLen (temp)
   $R5 = StartCharPos (temp)
   $R6 = TempStr (temp)*/

  ;Get input from user
  Exch $R0
  Exch
  Exch $R1
  Exch 2
  Exch $R2
  Push $R3
  Push $R4
  Push $R5
  Push $R6

  ;Get "String" and "SubString" length
  StrLen $R3 $R1
  StrLen $R4 $R2
  ;Start "StartCharPos" counter
  StrCpy $R5 0

  ;Loop until "SubString" is found or "String" reaches its end
  ${Do}
    ;Remove everything before and after the searched part ("TempStr")
    StrCpy $R6 $R2 $R3 $R5

    ;Compare "TempStr" with "SubString"
    ${If} $R6 == $R1
      ${If} $R0 == `<`
        IntOp $R6 $R3 + $R5
        IntOp $R0 $R4 - $R6
      ${Else}
        StrCpy $R0 $R5
      ${EndIf}
      ${ExitDo}
    ${EndIf}
    ;If not "SubString", this could be "String"'s end
    ${If} $R5 >= $R4
      StrCpy $R0 ``
      ${ExitDo}
    ${EndIf}
    ;If not, continue the loop
    IntOp $R5 $R5 + 1
  ${Loop}

  ;Return output to user
  Pop $R6
  Pop $R5
  Pop $R4
  Pop $R3
  Pop $R2
  Exch
  Pop $R1
  Exch $R0
FunctionEnd

Function un.CreateLogFromFile

  DetailPrint "Removing Files.."

  IfFileExists "$INSTDIR\install.log" 0 FailedtoOpen
   StrCpy $R7 "$INSTDIR\install.log"
   Goto OpenDir
  OpenDir:
  ClearErrors
  FileOpen $0 "$R7" "r"
  ClearErrors
  FileOpen $9 "$WINDIR\DirNSISTemp.log" "w"
  IfErrors Done
  Loop:
  ClearErrors
  FileReadUTF16LE $0 $1
  IfErrors Done
  StrCpy $2 $1 11
  StrCmp $2 "File: wrote" 0 MaybeDir
  ${UnStrloc} $3 $1 "to" ">"
  Intop $3 $3 + 4
  Strcpy $5 $1 "" $3
  StrCpy $5 $5 -3
  Delete /REBOOTOK $5
  MaybeDir:
  StrCmp $2 "CreateDirec" 0 DontDelete
  ${UnStrloc} $3 $1 ":" ">"
  Intop $3 $3 + 3
  Strcpy $5 $1 "" $3
  Strcpy $5 $5 -7
  FileWrite $9 "$5$\r$\n"
  DontDelete:
  Goto Loop
  Done:
  FileClose $0
  FileClose $9

  FileOpen $R1 "$WINDIR\DirNSIS.log" "w"
  ${un.FileReadFromEnd} "$WINDIR\DirNSISTemp.log" "un.Reversal"
  FileClose $R1
  Delete "$WINDIR\DirNSISTemp.log"
  Goto FinishedUninstall

  FailedtoOpen:

  RMDIR /r "$INSTDIR"

  FinishedUninstall:
FunctionEnd

Function un.RemoveDirectoriesFromLog

  ClearErrors
  FileOpen $0 "$WINDIR\DirNSIS.log" "r"
  IfErrors DoneDirNSIS
  LoopDirNSIS:
  ClearErrors
  FileRead $0 $1
  IfErrors DoneDirNSIS
  StrCpy $1 $1 -2
  RMDIR /REBOOTOK $1
  Goto LoopDirNSIS
  DoneDirNSIS:
  FileClose $0

FunctionEnd

Function un.Reversal
	StrCmp $7 -1 0 +5
	StrCpy $1 $9 1 -1
	StrCmp $1 '$\n' +3
	StrCmp $1 '$\r' +2
        StrCpy $9 '$9$\r$\n'

	FileWrite $R1 "$9"

	Push $0
FunctionEnd

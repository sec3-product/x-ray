# Microsoft Developer Studio Project File - Name="mod_dav" - Package Owner=<4>
# Microsoft Developer Studio Generated Build File, Format Version 6.00
# ** DO NOT EDIT **

# TARGTYPE "Win32 (x86) Dynamic-Link Library" 0x0102

CFG=mod_dav - Win32 Release
!MESSAGE This is not a valid makefile. To build this project using NMAKE,
!MESSAGE use the Export Makefile command and run
!MESSAGE 
!MESSAGE NMAKE /f "mod_dav.mak".
!MESSAGE 
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "mod_dav.mak" CFG="mod_dav - Win32 Release"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "mod_dav - Win32 Release" (based on "Win32 (x86) Dynamic-Link Library")
!MESSAGE "mod_dav - Win32 Debug" (based on "Win32 (x86) Dynamic-Link Library")
!MESSAGE 

# Begin Project
# PROP AllowPerConfigDependencies 0
# PROP Scc_ProjName ""
# PROP Scc_LocalPath ""
CPP=cl.exe
MTL=midl.exe
RSC=rc.exe

!IF  "$(CFG)" == "mod_dav - Win32 Release"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "Release"
# PROP BASE Intermediate_Dir "Release"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "Release"
# PROP Intermediate_Dir "Release"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /MD /W3 /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /FD /c
# ADD CPP /nologo /MD /W3 /Zi /O2 /I "../../../include" /I "../../../srclib/apr/include" /I "../../../srclib/apr-util/include" /D "NDEBUG" /D "WIN32" /D "_WINDOWS" /D "DAV_DECLARE_EXPORT" /Fd"Release\mod_dav_src" /FD /c
# ADD BASE MTL /nologo /D "NDEBUG" /win32
# ADD MTL /nologo /D "NDEBUG" /mktyplib203 /win32
# ADD BASE RSC /l 0x409 /d "NDEBUG"
# ADD RSC /l 0x409 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib ws2_32.lib mswsock.lib /nologo /subsystem:windows /dll /machine:I386 /out:"Release/mod_dav.so" /base:@..\..\..\os\win32\BaseAddr.ref,mod_dav.so
# ADD LINK32 kernel32.lib ws2_32.lib mswsock.lib /nologo /subsystem:windows /dll /incremental:no /debug /machine:I386 /out:"Release/mod_dav.so" /base:@..\..\..\os\win32\BaseAddr.ref,mod_dav.so /opt:ref

!ELSEIF  "$(CFG)" == "mod_dav - Win32 Debug"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "Debug"
# PROP BASE Intermediate_Dir "Debug"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir "Debug"
# PROP Intermediate_Dir "Debug"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /MDd /W3 /GX /Zi /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /FD /c
# ADD CPP /nologo /MDd /W3 /GX /Zi /Od /I "../../../include" /I "../../../srclib/apr/include" /I "../../../srclib/apr-util/include" /D "_DEBUG" /D "WIN32" /D "_WINDOWS" /D "DAV_DECLARE_EXPORT" /Fd"Debug\mod_dav_src" /FD /c
# ADD BASE MTL /nologo /D "_DEBUG" /win32
# ADD MTL /nologo /D "_DEBUG" /mktyplib203 /win32
# ADD BASE RSC /l 0x409 /d "_DEBUG"
# ADD RSC /l 0x409 /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib ws2_32.lib mswsock.lib /nologo /subsystem:windows /dll /incremental:no /debug /machine:I386 /out:"Debug/mod_dav.so" /base:@..\..\..\os\win32\BaseAddr.ref,mod_dav.so
# ADD LINK32 kernel32.lib ws2_32.lib mswsock.lib /nologo /subsystem:windows /dll /incremental:no /debug /machine:I386 /out:"Debug/mod_dav.so" /base:@..\..\..\os\win32\BaseAddr.ref,mod_dav.so

!ENDIF 

# Begin Target

# Name "mod_dav - Win32 Release"
# Name "mod_dav - Win32 Debug"
# Begin Group "Source Files"

# PROP Default_Filter "cpp;c;cxx;rc;def;r;odl;hpj;bat;for;f90"
# Begin Source File

SOURCE=.\liveprop.c
# End Source File
# Begin Source File

SOURCE=.\mod_dav.c
# End Source File
# Begin Source File

SOURCE=.\props.c
# End Source File
# Begin Source File

SOURCE=.\providers.c
# End Source File
# Begin Source File

SOURCE=.\std_liveprop.c
# End Source File
# Begin Source File

SOURCE=.\util.c
# End Source File
# Begin Source File

SOURCE=.\util_lock.c
# End Source File
# End Group
# Begin Group "Header Files"

# PROP Default_Filter "h;hpp;hxx;hm;inl;fi;fd"
# Begin Source File

SOURCE=.\mod_dav.h
# End Source File
# End Group
# Begin Source File

SOURCE=.\mod_dav.rc
# End Source File
# Begin Source File

SOURCE=..\..\..\build\win32\win32ver.awk

!IF  "$(CFG)" == "mod_dav - Win32 Release"

# PROP Ignore_Default_Tool 1
# Begin Custom Build - Creating Version Resource
InputPath=..\..\..\build\win32\win32ver.awk

".\mod_dav.rc" : $(SOURCE) "$(INTDIR)" "$(OUTDIR)"
	awk -f ../../../build/win32/win32ver.awk mod_dav.so "dav_module for Apache" ../../../include/ap_release.h > .\mod_dav.rc

# End Custom Build

!ELSEIF  "$(CFG)" == "mod_dav - Win32 Debug"

# PROP Ignore_Default_Tool 1
# Begin Custom Build - Creating Version Resource
InputPath=..\..\..\build\win32\win32ver.awk

".\mod_dav.rc" : $(SOURCE) "$(INTDIR)" "$(OUTDIR)"
	awk -f ../../../build/win32/win32ver.awk mod_dav.so "dav_module for Apache" ../../../include/ap_release.h > .\mod_dav.rc

# End Custom Build

!ENDIF 

# End Source File
# End Target
# End Project

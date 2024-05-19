# What's in this folder?

## 1. `settings.json`
The configuration file that adds visual studio 2019 developer powershell and command prompt to Windows Terminal.

## 2. `windows-demo-steps.txt`
The steps to reproduce Lam Research's race detection report on windows

## 3. `lam.patches`
The git diff file for `windows-demo-steps`.
It should be directly patched onto Lam's source code directory.

The source code directory should includes:
```
OES
|-LamOesPlus
|-OesMFC
|-src-OES
```

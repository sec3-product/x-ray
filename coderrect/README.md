# What is this repo?
This repo contains almost all components (except the LLVMRace and our customized clang) of the Coderrect Scanner.
Since most components are small, we would like to manage them all in this single repo.

# What components are involved?
- `/coderrect-exec`: Coderrect's exec wrapper
- `/gosrc`: All executables written in Golang
  - `/cmd`: Main entries for go executables
  - `/gllvm`: Coderrect's customized gllvm
  - `/reporter`: Coderrect's HTML report generator
  - `/tool`: Some frequently used tools/scripts (?)
  - `/util`: Utility libraries, including our conflib, jsonparser, and logger
- `/installer`: Scripts for building Coderrect installation package

# Quick start
- Install Golang:
  - `sudo apt install golang` for Ubuntu
  - check [this](https://golang.org/doc/install) for other systems
- `cd gosrc && make all`

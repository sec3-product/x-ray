GO              := GO111MODULE=on go
GOBUILD         := $(GO) build $(BUILD_FLAG)
GOBUILD_RAW     := go build $(BUILD_FLAG)
GOTEST          := $(GO) test -p $(P)

.PHONY: all clean test coderrect reporter gllvm reporter-server

default: coderrect buildsucc

buildsucc:
	@echo Build Coderrect successfully!

all: clean coderrect reporter gllvm reporter-server

tidy:
	@echo "go mod tidy"
	./tools/check-tidy.sh

clean: 
	rm -f bin/*

coderrect:
	$(GOBUILD)-o bin/coderrect cmd/coderrect/main.go

reporter:
	$(GOBUILD)-o bin/reporter cmd/reporter/main.go

reporter-server:
	$(GOBUILD)-o bin/reporter-server cmd/reporter-server/main.go

oreo:
	$(GOBUILD)-o bin/oreo cmd/oreo/main.go

gllvm: gllvm-ar gllvm-clang gllvm-clang++ gllvm-fortran gllvm-ld gllvm-cp

gllvm-ar:
	$(GOBUILD)-o bin/coderrect-ar cmd/coderrect-ar/main.go

gllvm-clang:
	$(GOBUILD)-o bin/coderrect-clang cmd/coderrect-clang/main.go

gllvm-clang++:
	$(GOBUILD)-o bin/coderrect-clang++ cmd/coderrect-clang++/main.go

gllvm-fortran:
	$(GOBUILD)-o bin/coderrect-fortran cmd/coderrect-fortran/main.go

gllvm-ld:
	$(GOBUILD)-o bin/coderrect-ld cmd/coderrect-ld/main.go

gllvm-cp:
	$(GOBUILD)-o bin/coderrect-cp cmd/coderrect-cp/main.go

lint:tools/bin/revive
	@echo "linting"
	@tools/bin/revive -formatter friendly -config tools/check/revive.toml $(FILES)

leak: failpoint-enable
	@export log_level=debug; \
	$(GOTEST) -tags leak $(PACKAGES) || { $(FAILPOINT_DISABLE); exit 1; }
	@$(FAILPOINT_DISABLE)


#
# Windows Command (after make and golang is installed)
# 
windows: windows-clean windows-coderrect windows-reporter windows-dispatcher windows-preprocessor windows-entry

windows-clean:
	del /q .\bin\*

windows-coderrect:
	$(GOBUILD_RAW)-o .\bin\coderrect.exe .\cmd\coderrect\main.go

windows-reporter:
	$(GOBUILD_RAW)-o .\bin\reporter.exe .\cmd\reporter\main.go

windows-dispatcher:
	$(GOBUILD_RAW)-o .\bin\coderrect-dispatcher.exe .\cmd\coderrect-dispatcher\main.go

windows-preprocessor:
	$(GOBUILD_RAW)-o .\bin\coderrect-preprocessor.exe .\cmd\windows-preprocessor\main.go

windows-entry:
	$(GOBUILD_RAW)-o .\bin\coderrect-entry.exe .\cmd\windows-entry\main.go

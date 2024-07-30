GO              := go
GOBUILD         := $(GO) build $(BUILD_FLAG)
GOBUILD_RAW     := go build $(BUILD_FLAG)
GOTEST          := $(GO) test -p $(P)

.PHONY: all clean coderrect reporter reporter-server

default: coderrect buildsucc

buildsucc:
	@echo Build Coderrect successfully!

all: clean coderrect reporter reporter-server

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

lint:tools/bin/revive
	@echo "linting"
	@tools/bin/revive -formatter friendly -config tools/check/revive.toml $(FILES)

leak: failpoint-enable
	@export log_level=debug; \
	$(GOTEST) -tags leak $(PACKAGES) || { $(FAILPOINT_DISABLE); exit 1; }
	@$(FAILPOINT_DISABLE)

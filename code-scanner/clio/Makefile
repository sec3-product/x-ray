TARGETS=rdiff check binrunner clio

all: $(TARGETS)

.PHONY: clio

rdiff:
	go build ./cmd/diff/rdiff.go

check:
	go build ./cmd/regress/check.go

binrunner:
	go build ./cmd/runner/binrunner.go

clean:
	@rm -f $(TARGETS)
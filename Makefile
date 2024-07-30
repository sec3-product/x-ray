.PHONY: all prepare_build_dir check_go_version add_to_path check_binary

# Variables
DOCKER_IMAGE_VERSION = 1.2
RUST = false
BINARY_PATH = ~/local/sec3/bin/coderrect

# The LLVM version to use.
LLVM_VERSION ?= 14.0.6

# The default prebuilt image, which is hosted on DigitalOcean (private)
# registry. Use `doctl registry login` to ensure the access.
LLVM_PREBUILT_IMAGE ?= registry.digitalocean.com/soteria/llvm-prebuilt-$(LLVM_VERSION):latest

X_RAY_IMAGE ?= x-ray:latest

all: prepare_build_dir pull_llvm build_in_docker compile_go compile_cmake check_go_version add_to_path check_binary

# Prepare the build directory
prepare_build_dir:
	@echo "Checking and creating build directory..."
	@mkdir -p build
	@echo "Build directory created."

# Check Go version
check_go_version:
	@echo "Checking Go version..."
	@go version

# Add to PATH
add_to_path:
	@echo "Adding LLVMRace to PATH"
	@export PATH=$(PATH):$(CURDIR)/LLVMRace

# Check binary and add to PATH if needed
check_binary:
	@if [ ! -f $(BINARY_PATH) ]; then \
		echo "Binary file does not exist, creating..."; \
		mkdir -p ~/local/sec3/bin/; \
		touch $(BINARY_PATH); \
		chmod 755 $(BINARY_PATH); \
	fi
	@echo "Checking if binary directory is in PATH..."
	@if [[ ":$$PATH:" != *":$$HOME/local/sec3/bin:"* ]]; then \
		echo "Adding binary directory to PATH..."; \
		export PATH="$$HOME/local/sec3/bin:$$PATH"; \
		echo "Binary directory added to PATH."; \
	fi

# Build llvm-prebuilt image
build-llvm-prebuilt-image:
	@docker build -t $(LLVM_PREBUILT_IMAGE) \
	  --build-arg LLVM_VERSION=$(LLVM_VERSION) \
	  -f Dockerfile.llvm .

push-llvm-prebuilt-image: build-llvm-prebuilt-image
	@docker push $(LLVM_PREBUILT_IMAGE)

build-image:
	@if [ "$(CI)" = "true" ]; then \
	  echo "CI build detected. Pulling $(LLVM_PREBUILT_IMAGE) to ensure using the latest."; \
	  docker pull $(LLVM_PREBUILT_IMAGE) || \
	    (echo "Error: Failed to pull $(LLVM_PREBUILT_IMAGE)." && \
	     exit 1); \
	else \
	  if ! docker inspect --type=image $(LLVM_PREBUILT_IMAGE) > /dev/null 2>&1; then \
	    docker pull $(LLVM_PREBUILT_IMAGE) || \
	      (echo "Error: Failed to pull $(LLVM_PREBUILT_IMAGE)." && \
	       echo "Please make sure you are logged into the registry with the correct permissions." && \
	       echo "You can log in with: `doctl registry login` or manually pull the image." && \
	       exit 1) \
	  else \
	    echo "$(LLVM_PREBUILT_IMAGE) already exists locally. Skipped pulling."; \
	  fi; \
	fi
	@docker build -t $(X_RAY_IMAGE) \
	  --build-arg LLVM_PREBUILT_IMAGE=$(LLVM_PREBUILT_IMAGE) \
	  --build-arg LLVM_VERSION=$(LLVM_VERSION) \
	  -f Dockerfile.x-ray .

run-test:
	@WORKING_DIR=$(CURDIR)/e2e \
	  X_RAY_IMAGE=$(X_RAY_IMAGE) \
	  go test -v -count=1 ./e2e/...

coderrect:
	go build -o bin/coderrect cmd/coderrect/main.go

reporter:
	go build -o bin/reporter cmd/reporter/main.go

reporter-server:
	go build -o bin/reporter-server cmd/reporter-server/main.go

clean:
	rm -f bin/*

cli: clean coderrect reporter reporter-server

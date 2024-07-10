.PHONY: all prepare_build_dir pull_llvm build_in_docker compile_go compile_cmake check_go_version add_to_path check_binary

# Variables
DOCKER_IMAGE_VERSION = 1.2
RUST = false
BINARY_PATH = ~/local/sec3/bin/coderrect

# The default prebuilt image, which is hosted on DigitalOcean (private)
# registry. Use `doctl registry login` to ensure the access.
LLVM_PREBUILT_IMAGE ?= registry.digitalocean.com/soteria/llvm-prebuilt:latest

X_RAY_IMAGE ?= x-ray:latest

all: prepare_build_dir pull_llvm build_in_docker compile_go compile_cmake check_go_version add_to_path check_binary

# Prepare the build directory
prepare_build_dir:
	@echo "Checking and creating build directory..."
	@mkdir -p build
	@echo "Build directory created."

# Clone or pull LLVM 12
pull_llvm: prepare_build_dir
	@if [ ! -d "build/llvm12" ]; then \
		echo "Cloning llvm12..."; \
		cd build && git clone --recursive ssh://git@github.com/coderrect-inc/llvm12.git; \
	else \
		echo "Updating llvm12..."; \
		cd build/llvm12 && git pull --all; \
	fi
	@echo "Finished updating llvm12."

# Build in Docker
build_in_docker:
	@echo "Checking Docker image presence..."
	@if [ `docker images | grep 'coderrect/installer' | grep $(DOCKER_IMAGE_VERSION) | wc -l` != "1" ]; then \
		echo "Docker image coderrect/installer:$(DOCKER_IMAGE_VERSION) not installed, please run build-coderrect-installer-1.1-docker.sh first"; \
		exit 1; \
	fi
	@echo "Building in Docker..."
	@if [ $(RUST) = true ]; then \
		docker run --rm -v $(PWD)/build:/build -v $(PWD)/dockerstuff/scripts:/scripts --user=$$(id -u):$$(id -g) coderrect/installer:$(DOCKER_IMAGE_VERSION) /scripts/build-components-llvm12-rust.sh; \
	else \
		docker run --rm -v $(PWD)/build:/build -v $(PWD)/dockerstuff/scripts:/scripts --user=$$(id -u):$$(id -g) coderrect/installer:1.1 /scripts/build-components.sh; \
	fi

# Compile using Go
compile_go:
	@echo "Compiling coderrect with Go..."
	@cd coderrect && cd gosrc && ./run && cd ../..

# Compile using CMake
compile_cmake: compile_parser compile_detector

compile_parser:
	@echo "Compiling code-parser with CMake..."
	@cd code-parser && mkdir -p build && cd build && cmake .. && make && cd ../..

compile_detector:
	@echo "Compiling code-detector with CMake..."
	@cd code-detector && mkdir -p build && cd build && cmake .. && make && cd ../..

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
	@docker build -t $(LLVM_PREBUILT_IMAGE) -f Dockerfile.llvm .

push-llvm-prebuilt-image: build-llvm-prebuilt-image
	@docker push $(LLVM_PREBUILT_IMAGE)

build-image:
	@docker pull $(LLVM_PREBUILT_IMAGE) || \
	  (echo "Error: Failed to pull $(LLVM_PREBUILT_IMAGE)." && \
	   echo "Please make sure you are logged into the registry with the correct permissions." && \
	   echo "You can log in with: `doctl registry login` or manually pull the image." && \
	   exit 1)
	@docker tag $(LLVM_PREBUILT_IMAGE) llvm-prebuilt
	@docker build -t $(X_RAY_IMAGE) -f Dockerfile.x-ray .

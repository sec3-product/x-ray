# The LLVM version to use.
LLVM_VERSION ?= 14.0.6

# The default prebuilt image, which is hosted on DigitalOcean (private)
# registry. Use `doctl registry login` to ensure the access.
LLVM_PREBUILT_IMAGE ?= registry.digitalocean.com/soteria/llvm-prebuilt-$(LLVM_VERSION):latest

# The path to prebuilt LLVM files. It expects `clang++` to be available under
# `$(LLVM_PREBUILT_PATH)/bin`. One can use `make extract-llvm` to extract the
# prebuilt files from the prebuilt image to the specified directory.
LLVM_PREBUILT_PATH ?=

X_RAY_IMAGE ?= x-ray:latest

BUILD_DIR ?= build
INSTALL_DIR ?= $(BUILD_DIR)/dist

.PHONY: all build-x-ray build-cli install extract-llvm check-llvm \
  build-detector build-parser \
  build-container-image run-container-e2e run-native-e2e \
  build-llvm-prebuilt-image push-llvm-prebuilt-image

all: build-x-ray build-image

build-x-ray: build-detector build-parser build-cli

check-llvm:
	@if [ -z "$(LLVM_PREBUILT_PATH)" ]; then \
	  echo "LLVM_PREBUILT_PATH is not defined. Please set LLVM_PREBUILT_PATH to the directory containing LLVM."; \
	  exit 1; \
	else \
	  if [ ! -f "$(LLVM_PREBUILT_PATH)/bin/clang++" ] || [ ! -f "$(LLVM_PREBUILT_PATH)/bin/clang" ]; then \
	    echo "Error: bin/clang or bin/clang++ not found under $(LLVM_PREBUILT_PATH)/bin."; \
	    exit 1; \
	  else \
	    echo "LLVM_PREBUILT_PATH is defined and bin/clang++ is present."; \
	  fi; \
	fi

build-detector: check-llvm
	@mkdir -p $(BUILD_DIR)/detector
	@cd $(BUILD_DIR)/detector && \
	  cmake \
	    -DCMAKE_BUILD_TYPE=Release \
	    -DCMAKE_C_COMPILER=$(LLVM_PREBUILT_PATH)/bin/clang \
	    -DCMAKE_CXX_COMPILER=$(LLVM_PREBUILT_PATH)/bin/clang++ \
	    -DLLVM_DIR=$(LLVM_PREBUILT_PATH)/lib/cmake/llvm \
	    -DMLIR_DIR=$(LLVM_PREBUILT_PATH)/lib/cmake/mlir \
	    -DLLVM_VERSION=$(LLVM_VERSION) \
	    ../../code-detector && \
	  make -j

build-parser: check-llvm
	@mkdir -p $(BUILD_DIR)/parser
	@cd $(BUILD_DIR)/parser && \
	  cmake \
	    -DCMAKE_BUILD_TYPE=Release \
	    -DCMAKE_C_COMPILER=$(LLVM_PREBUILT_PATH)/bin/clang \
	    -DCMAKE_CXX_COMPILER=$(LLVM_PREBUILT_PATH)/bin/clang++ \
	    -DLLVM_DIR=$(LLVM_PREBUILT_PATH)/lib/cmake/llvm \
	    -DMLIR_DIR=$(LLVM_PREBUILT_PATH)/lib/cmake/mlir \
	    -DLLVM_VERSION=$(LLVM_VERSION) \
	    ../../code-parser && \
	  make -j

build-cli:
	@echo "Building X-Ray CLI..."
	@mkdir -p $(BUILD_DIR)/cli/bin
	@go build -o $(BUILD_DIR)/cli/bin/coderrect cmd/coderrect/main.go
	@go build -o $(BUILD_DIR)/cli/bin/reporter cmd/reporter/main.go
	@go build -o $(BUILD_DIR)/bin/reporter-server cmd/reporter-server/main.go

install:
	@echo "Installing X-Ray to $(INSTALL_DIR)..."
	@rm -rf $(INSTALL_DIR)
	@for dir in bin conf data/reporter/artifacts/images; do \
	  echo "Creating directory $(INSTALL_DIR)/$${dir}..."; \
	  mkdir -p "$(INSTALL_DIR)/$${dir}"; \
	done
	@cp $(BUILD_DIR)/detector/bin/racedetect $(INSTALL_DIR)/bin/
	@cp $(BUILD_DIR)/parser/bin/sol-racedetect $(INSTALL_DIR)/bin/
	@cp $(BUILD_DIR)/cli/bin/* $(INSTALL_DIR)/bin/
	@cp package/conf/coderrect.json $(INSTALL_DIR)/conf/
	@cp package/data/reporter/*.html $(INSTALL_DIR)/data/reporter/
	@cp package/data/reporter/artifacts/coderrect* $(INSTALL_DIR)/data/reporter/artifacts/
	@cp package/data/reporter/artifacts/images/* $(INSTALL_DIR)/data/reporter/artifacts/images/
	@echo "Done. X-Ray has been installed to $(INSTALL_DIR)."

build-llvm-prebuilt-image:
	@docker build -t $(LLVM_PREBUILT_IMAGE) \
	  --build-arg LLVM_VERSION=$(LLVM_VERSION) \
	  -f Dockerfile.llvm .

push-llvm-prebuilt-image: build-llvm-prebuilt-image
	@docker push $(LLVM_PREBUILT_IMAGE)

extract-llvm:
	@echo "Extracting LLVM prebuilt files from $(LLVM_PREBUILT_IMAGE) to $(BUILD_DIR)/llvm..."
	@mkdir -p $(BUILD_DIR)
	@CONTAINER_ID=$$(docker create $(LLVM_PREBUILT_IMAGE)) && \
	  docker cp $$CONTAINER_ID:/usr/local/llvm $(BUILD_DIR)/llvm && \
	  docker rm $$CONTAINER_ID
	@echo "Done. Now you can set LLVM_PREBUILT_PATH=$$(realpath $(BUILD_DIR)/llvm) to use the prebuilt LLVM."

build-container-image:
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

run-native-e2e:
	@X_RAY_EXECUTABLE=$$(realpath $(INSTALL_DIR)/bin/coderrect) \
	  go test -v -run=TestNativeE2E -count=1 ./e2e/...

run-container-e2e:
	@WORKING_DIR=$(CURDIR)/e2e \
	  X_RAY_IMAGE=$(X_RAY_IMAGE) \
	  go test -v -run=TestContainerE2E -count=1 ./e2e/...


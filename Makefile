BUILD_DIR ?= build
INSTALL_DIR ?= $(BUILD_DIR)/dist
DIST_TARBALL ?= $(BUILD_DIR)/x-ray-$(VERSION).tar.gz

DEFAULT_TARGET_ARCH := $(shell uname -m | sed 's/x86_64/amd64/;s/aarch64/arm64/')
TARGET_ARCH ?= $(DEFAULT_TARGET_ARCH)
PLATFORM := linux/$(TARGET_ARCH)

LLVM_VERSION ?= 14.0.6
LLVM_PREBUILT_TAG ?= latest

# The default LLVM prebuilt image, which is hosted on GitHub Container
# Registry (ghcr.io).
LLVM_PREBUILT_IMAGE_NO_TAG ?= ghcr.io/sec3-product/llvm-prebuilt-$(LLVM_VERSION)
LLVM_PREBUILT_IMAGE ?= $(LLVM_PREBUILT_IMAGE_NO_TAG):$(LLVM_PREBUILT_TAG)

# The path to prebuilt LLVM files. It expects `clang++` to be available under
# `$(LLVM_PREBUILT_PATH)/bin`. One can use `make extract-llvm` to extract the
# prebuilt files from the prebuilt image to the specified directory.
LLVM_PREBUILT_PATH ?= $(realpath $(BUILD_DIR)/llvm)

X_RAY_IMAGE ?= x-ray:latest

# The version of X-Ray. The value is set by workflow for release builds.
# Otherwise it's set on the first invocation of make; subsequent invocations
# (in particular from Dockerfile) will use the cached value.
ifndef VERSION
  VERSION := $(shell git describe --tags --always)
endif

.PHONY: all build-x-ray build-cli install extract-llvm check-llvm \
  build-analyzer build-parser \
  build-container-image \
  build-llvm-prebuilt-image push-llvm-prebuilt-image push-llm-prebuilt-manifest \
  run-unit-tests \
  prepare-e2e-test run-container-e2e run-native-e2e \
  clean

all: build-x-ray build-container-image

build-x-ray: build-analyzer build-parser build-cli

check-llvm:
	@if [ -z "$(LLVM_PREBUILT_PATH)" ]; then \
	  echo "LLVM_PREBUILT_PATH is not defined. Please set LLVM_PREBUILT_PATH to the directory containing LLVM."; \
	  exit 1; \
	else \
	  if [ ! -f "$(LLVM_PREBUILT_PATH)/bin/clang++" ] || [ ! -f "$(LLVM_PREBUILT_PATH)/bin/clang" ]; then \
	    echo "Error: bin/clang or bin/clang++ not found under $(LLVM_PREBUILT_PATH)/bin."; \
	    exit 1; \
	  else \
	    echo "LLVM_PREBUILT_PATH resolves to $(LLVM_PREBUILT_PATH)."; \
	  fi; \
	fi

build-analyzer: check-llvm
	@mkdir -p $(BUILD_DIR)/analyzer
	@cd $(BUILD_DIR)/analyzer && \
	  cmake \
	    -DCMAKE_BUILD_TYPE=Release \
	    -DCMAKE_C_COMPILER=$(LLVM_PREBUILT_PATH)/bin/clang \
	    -DCMAKE_CXX_COMPILER=$(LLVM_PREBUILT_PATH)/bin/clang++ \
	    -DLLVM_DIR=$(LLVM_PREBUILT_PATH)/lib/cmake/llvm \
	    -DMLIR_DIR=$(LLVM_PREBUILT_PATH)/lib/cmake/mlir \
	    -DLLVM_VERSION=$(LLVM_VERSION) \
	    ../../code-analyzer && \
	  make -j$(MAKE_THREADS)

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
	  make -j$(MAKE_THREADS)

build-cli:
	@echo "Building X-Ray CLI..."
	@mkdir -p $(BUILD_DIR)/cli/bin
	@go build -ldflags "-X main.version=$(VERSION)" \
	  -o $(BUILD_DIR)/cli/bin/xray \
	  cmd/cli/main.go
	@echo

install: build-x-ray
	@echo "Installing X-Ray $(VERSION) to $(INSTALL_DIR)..."
	@rm -rf $(INSTALL_DIR)
	@for dir in bin conf; do \
	  echo "Creating directory $(INSTALL_DIR)/$${dir}..."; \
	  mkdir -p "$(INSTALL_DIR)/$${dir}"; \
	done
	@cp $(BUILD_DIR)/analyzer/bin/sol-code-analyzer $(INSTALL_DIR)/bin/
	@cp $(BUILD_DIR)/parser/bin/sol-code-parser $(INSTALL_DIR)/bin/
	@cp $(BUILD_DIR)/cli/bin/* $(INSTALL_DIR)/bin/
	@cp $(LLVM_PREBUILT_PATH)/lib/libomp.so $(INSTALL_DIR)/bin/
	@cp package/conf/xray.json $(INSTALL_DIR)/conf/
	@echo "Done. X-Ray has been installed to $(INSTALL_DIR)."
	@echo

$(DIST_TARBALL): build-x-ray install
	@echo "Creating X-Ray distribution package..."
	@tar -zcvf $@ -C $(INSTALL_DIR) .
	@echo "Done. Package created: $@"
	@echo

dist: $(DIST_TARBALL)

clean:
	@rm -rf $(BUILD_DIR)/analyzer $(BUILD_DIR)/parser $(BUILD_DIR)/cli $(BUILD_DIR)/dist

build-llvm-prebuilt-image:
	@docker build -t $(LLVM_PREBUILT_IMAGE) \
	  --platform $(PLATFORM) \
	  --build-arg LLVM_VERSION=$(LLVM_VERSION) \
	  --build-arg MAKE_THREADS=$(MAKE_THREADS) \
	  -f Dockerfile.llvm .

push-llvm-prebuilt-image: build-llvm-prebuilt-image
	@docker push $(LLVM_PREBUILT_IMAGE)

push-llvm-prebuilt-manifest:
	@docker manifest create --amend $(LLVM_PREBUILT_IMAGE) \
	  $(LLVM_PREBUILT_IMAGE_NO_TAG):arm64 \
	  $(LLVM_PREBUILT_IMAGE_NO_TAG):amd64
	@docker manifest push $(LLVM_PREBUILT_IMAGE)

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
	  docker pull --platform $(PLATFORM) $(LLVM_PREBUILT_IMAGE) || \
	    (echo "Error: Failed to pull $(LLVM_PREBUILT_IMAGE)." && \
	     exit 1); \
	else \
	  FOUND_ARCH=$$(docker inspect --format '{{.Architecture}}' $(LLVM_PREBUILT_IMAGE)); \
	  if [ "$${FOUND_ARCH}" != "$(TARGET_ARCH)" ]; then \
	    echo "Pulling $(LLVM_PREBUILT_IMAGE) for $(FOUND_ARCH)"; \
	    docker pull --platform $(PLATFORM) $(LLVM_PREBUILT_IMAGE) || \
	      (echo "Error: Failed to pull $(LLVM_PREBUILT_IMAGE)." && \
	       echo "Please make sure you are logged into the registry with the correct permissions." && \
	       echo "You can log in with: `doctl registry login` or manually pull the image." && \
	       exit 1) \
	  else \
	    echo "$(LLVM_PREBUILT_IMAGE) already exists locally. Skipped pulling."; \
	  fi; \
	fi
	@docker build -t $(X_RAY_IMAGE) \
	  --platform $(PLATFORM) \
	  --build-arg LLVM_PREBUILT_IMAGE=$(LLVM_PREBUILT_IMAGE) \
	  --build-arg VERSION=$(VERSION) \
	  -f Dockerfile.x-ray .

run-unit-tests:
	@ctest --test-dir $(BUILD_DIR)/analyzer/test
	@ctest --test-dir $(BUILD_DIR)/parser/test
	@go test -v -count=1 ./...

prepare-e2e-test:
	@if [ ! -d "workspace/dexterity" ]; then \
	  echo "Directory 'workspace/dexterity' not found. Cloning repository..."; \
	  git clone --depth=1 https://github.com/solana-labs/dexterity.git workspace/dexterity; \
	fi

run-native-e2e: prepare-e2e-test
	@PATH=$$(realpath ${INSTALL_DIR}/bin):$${PATH} \
	  E2E_TEST_APP=$$(realpath workspace/dexterity/programs) \
	  go test -v -tags=e2e -run=TestNativeE2E -count=1 ./e2e/...

run-container-e2e: prepare-e2e-test
	@WORKING_DIR=$$(realpath workspace) \
	  X_RAY_IMAGE=$(X_RAY_IMAGE) \
	  E2E_TEST_APP=$$(realpath workspace/dexterity/programs) \
	  go test -v -tags=e2e -run=TestContainerE2E -count=1 ./e2e/...

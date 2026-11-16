BUILD_DIR  := build
BUILD_TYPE ?= Release
CMAKE      ?= cmake
GENERATOR  ?= Ninja
JOBS       ?= $(shell nproc)

KERNEL ?= all
SIZE   ?= 1024
REPS   ?= 50
CSV    ?= results/bench.csv
SIZES ?= 128,256,512,1024
SHAPE ?= 128,256,64
ALPHA ?= 1.0
BETA  ?= 0.0

.PHONY: all configure build run sizes shape verify test csv list clean rebuild

all: build

configure:
	$(CMAKE) -S . -B $(BUILD_DIR) -G "$(GENERATOR)" -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)

build: configure
	$(CMAKE) --build $(BUILD_DIR) -j $(JOBS)

run: build
	./$(BUILD_DIR)/matmul --kernel $(KERNEL) --size $(SIZE) \
	  --reps $(REPS) --alpha $(ALPHA) --beta $(BETA)

sizes: build
	./$(BUILD_DIR)/matmul --kernel $(KERNEL) --sizes $(SIZES) \
	  --reps $(REPS) --alpha $(ALPHA) --beta $(BETA)

shape: build
	./$(BUILD_DIR)/matmul --kernel $(KERNEL) --shape $(SHAPE) \
	  --reps $(REPS) --alpha $(ALPHA) --beta $(BETA)

verify: build
	./$(BUILD_DIR)/matmul --kernel $(KERNEL) --size $(SIZE) \
	  --verify-only --alpha 0.5 --beta 0.5

test: build
	./$(BUILD_DIR)/matmul --kernel $(KERNEL) --test-suite

csv: build
	mkdir -p $(dir $(CSV))
	./$(BUILD_DIR)/matmul --kernel $(KERNEL) --sizes $(SIZES) \
	  --reps $(REPS) --alpha $(ALPHA) --beta $(BETA) --csv $(CSV)

list: build
	./$(BUILD_DIR)/matmul --list-kernels

clean:
	rm -rf $(BUILD_DIR)

rebuild: clean build

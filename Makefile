BUILD_DIR  := build
BUILD_TYPE ?= Release
CMAKE      ?= cmake
GENERATOR  ?= Ninja
JOBS       ?= $(shell nproc)

KERNEL ?= naive
SIZE   ?= 1024
REPS   ?= 50
CSV    ?= results/bench.csv

.PHONY: all configure build run csv list clean rebuild

all: build

configure:
	$(CMAKE) -S . -B $(BUILD_DIR) -G "$(GENERATOR)" -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)

build: configure
	$(CMAKE) --build $(BUILD_DIR) -j $(JOBS)

run: build
	./$(BUILD_DIR)/matmul --kernel $(KERNEL) --size $(SIZE) --reps $(REPS)

csv: build
	mkdir -p $(dir $(CSV))
	./$(BUILD_DIR)/matmul --kernel $(KERNEL) --size $(SIZE) --reps $(REPS) --csv $(CSV)

list: build
	./$(BUILD_DIR)/matmul --list-kernels

clean:
	rm -rf $(BUILD_DIR)

rebuild: clean build

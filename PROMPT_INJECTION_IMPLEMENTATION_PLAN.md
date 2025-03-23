# Prompt Injection Benchmarks Implementation Plan

## Overview

The current Purple Bench application only supports MITRE and FRR benchmarks. This enhancement will add support for running Textual and Visual Prompt Injection benchmarks using datasets from the PurpleLlama/CyberSecurityBenchmarks collection.

## Implementation Steps

### 1. Dataset Management
- Create a `DatasetManager` class to handle dataset discovery and management
- Add known dataset definitions for prompt injection benchmarks
- Implement dataset download functionality from S3 bucket
- Create dataset selection UI components

### 2. Benchmark Runner Enhancements
- Extend `BenchmarkRunner` to support prompt injection benchmark types
- Add dataset parameter support to the run_benchmark method
- Implement result parsing for prompt injection benchmark outputs
- Enhance error handling and logging

### 3. UI Enhancements
- Update benchmark selection UI to include prompt injection options
- Add dataset selection dropdowns for prompt injection benchmarks
- Display appropriate dataset options based on benchmark type
- Ensure results display properly includes dataset information

### 4. Configuration Updates
- Update config.yaml with prompt injection benchmark settings
- Add dataset paths and configuration
- Ensure proper directory setup for benchmark outputs

## Key Files to Modify

- `app/app.py`: Add UI components and benchmark selection logic
- `app/benchmark_runner.py`: Extend benchmark runner functionality
- `app/dataset_manager.py`: Create new file for dataset management
- `app/utils.py`: Update utility functions for result handling
- `config/config.yaml`: Add prompt injection benchmark configuration
- `CHANGELOG.md`: Document new features and changes

## Target Datasets

### Textual Prompt Injection
- `prompt_injection.json` (English prompts)
- `prompt_injection_multilingual_machine_translated.json` (Multilingual prompts)

### Visual Prompt Injection
- `cse2_typographic_images`
- `cse2_visual_overlays`
- `cse2_adversarial_patches`
- `cse2_adversarial_qr_codes`

## Timeline and Milestones

### Phase 1: Setup and Infrastructure
- Create DatasetManager class
- Update configuration files
- Implement basic dataset discovery

### Phase 2: Textual Prompt Injection
- Implement textual prompt injection benchmark support
- Add dataset selection UI
- Test with sample datasets

### Phase 3: Visual Prompt Injection
- Implement visual prompt injection benchmark support
- Extend UI for visual dataset selection
- Test with sample datasets

### Phase 4: Testing and Documentation
- Comprehensive testing of benchmark functionality
- Update documentation
- Update changelog according to project guidelines
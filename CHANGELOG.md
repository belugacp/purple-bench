# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.1] - 2025-03-23

### Fixed
- Fixed bug in benchmark_runner.py causing MITRE benchmark category scores to not display properly in the UI
- Improved parameter extraction from benchmark commands for more reliable results display

## [0.4.0] - 2025-03-23

### Added
- Support for Textual Prompt Injection benchmarks with configurable datasets
- Support for Visual Prompt Injection benchmarks with configurable datasets
- New DatasetManager class for managing benchmark datasets
- Dataset downloading functionality for prompt injection datasets
- Enhanced benchmark results display to show dataset information

### Changed
- Updated BenchmarkRunner to support dataset parameters
- Updated UI to include dataset selection for prompt injection benchmarks
- Improved error handling and logging for benchmark runs

## [0.3.7] - 2025-03-23

### Added
- Added ability to delete benchmark results from the Results page
- Implemented delete confirmation dialog for benchmark result deletion
- Enhanced Results page with clearer benchmark result listing and action buttons

## [0.3.6] - 2025-03-22

### Changed
- Updated configuration path to use absolute path for storing benchmark results

## [0.3.5] - 2025-03-22

### Fixed
- Fixed recurring "Unknown benchmark type" error by implementing proper fallback mechanism in result generation
- Added automatic fallback to MITRE benchmark results for unrecognized benchmark types
- Improved error handling with detailed warning logs instead of error responses

## [0.3.4] - 2025-03-22

### Fixed
- Fixed "Unknown benchmark type" error by properly extracting the benchmark type from command parameters
- Improved benchmark type validation with built-in fallback to 'mitre' for invalid types
- Added more detailed error logging for benchmark type validation
- Consolidated benchmark command building logic to reduce code duplication

## [0.3.3] - 2025-03-21

### Added
- Added model connection verification functionality
- Added diagnostic command generation for easier troubleshooting

### Changed
- Improved error handling in the benchmark runner
- Enhanced logging for better debugging

## [0.3.2] - 2025-03-15

### Added
- Support for multiple LLM providers (OpenAI, Anthropic, etc.)
- Added FRR benchmark option alongside MITRE

### Changed
- Refactored benchmark runner for better extensibility
- Improved UI for benchmark selection and configuration

## [0.3.1] - 2025-03-10

### Added
- Result comparison feature for benchmark results
- Added visualization options for benchmark metrics

### Fixed
- Fixed progress reporting in long-running benchmarks

## [0.3.0] - 2025-03-01

### Added
- Initial implementation of MITRE ATT&CK benchmark
- Basic UI for running benchmarks and viewing results
- Support for OpenAI API integration
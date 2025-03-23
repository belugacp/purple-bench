# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.19] - 2025-03-27

### Fixed
- Fixed dataset options display for Textual Prompt Injection benchmark
- Added specific dataset options (English and multilingual) to replace 'manual' option
- Enhanced dataset fallback mechanisms to ensure proper options are always available

## [0.3.18] - 2025-03-27

### Added
- Added proper dataset selection options for the Textual Prompt Injection benchmark
- Enhanced dataset handling to display English and multilingual prompt injection dataset options

## [0.3.17] - 2025-03-27

### Fixed
- Fixed Home page statistics to display correct data from benchmark results files
- Updated Compare Models page to properly load benchmark results from files

## [0.3.16] - 2025-03-27

### Fixed
- Fixed missing logger in BenchmarkRunner class
- Updated logging calls throughout benchmark_runner.py for consistency

## [0.3.15] - 2025-03-27

### Fixed
- Fixed error handling in benchmark runner to properly display error messages
- Fixed parameter handling for benchmark dataset selection
- Fixed visual prompt injection benchmark to properly include dataset parameter in results

## [0.3.14] - 2025-03-27

### Added
- Enhanced Results page to display dataset information for visual prompt injection benchmarks

### Changed
- Improved benchmark result display formatting for better readability

## [0.3.13] - 2025-03-27

### Added
- Enhanced visual prompt injection benchmark with more detailed results
- Added visualization for benchmark results with category scores
- Added support for passing dataset parameter to benchmark runner

### Fixed
- Fixed results page to properly load benchmark results from result files instead of config
- Fixed benchmark result visualization to show actual scores and categories
- Enhanced dataset selection in the UI to properly work with both local and remote datasets

## [0.3.12] - 2025-03-27

### Added
- Enhanced DatasetManager to list available S3 datasets even when not downloaded locally
- Added predefined list of known Visual Prompt Injection datasets from S3
- Added default visual prompt injection datasets to config.yaml

### Fixed
- Fixed dataset selection in the UI to properly show available datasets for visual prompt injection
- Improved dataset discovery logic to use both local and remote sources
- Added detailed logging for dataset discovery and selection process

## [0.3.11] - 2025-03-26

### Added
- Enhanced DatasetManager to list available S3 datasets even when not downloaded locally
- Added predefined list of known Visual Prompt Injection datasets from S3

## [0.3.10] - 2025-03-25

### Fixed
- Fixed API key saving functionality in settings page
- Fixed issue with API keys not being properly detected in the Run Benchmark page
- Added missing 'tempfile' import in benchmark_runner.py
- Improved error handling for API key management

## [0.3.9] - 2025-03-24

### Added
- Added automatic dataset download feature for prompt injection benchmarks
- Implemented DatasetManager class to handle downloading datasets from S3
- Added support for using datasets from Purple Llama's S3 bucket
- Added dynamic dataset selection based on available local datasets
- Added .gitignore rules to ensure datasets are not committed to the repository

## [0.3.8] - 2025-03-24

### Added
- Added support for running Textual Prompt Injection benchmarks
- Added support for running Visual Prompt Injection benchmarks
- Added UI components for configuring and running prompt injection benchmarks
- Added dataset selection for both textual and visual prompt injection benchmarks
- Enhanced benchmark runner to handle prompt injection test cases and results

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

## [0.3.3] - 2025-03-22

### Fixed
- Fixed "Unknown benchmark type" error by properly extracting the benchmark type from command parameters
- Improved benchmark type detection with robust fallback mechanism and detailed error logging

## [0.3.2] - 2025-03-22

### Added
- Added comprehensive diagnostic mode for model verification with detailed output
- Added direct API requests using requests module instead of SDK clients to avoid compatibility issues
- Added version information logging for easier troubleshooting
- Added detailed error tracing with full stack traces and API response information

### Changed
- Completely redesigned model verification to use standalone diagnostic scripts for more robust error reporting
- Enhanced error output to include complete diagnostic information from API calls

## [0.3.1] - 2025-03-22

### Fixed
- Fixed OpenAI client initialization in model verification to resolve `Client.init() got an unexpected keyword argument 'proxies'` error
- Simplified OpenAI client creation to improve compatibility with various OpenAI SDK versions

## [0.3.0] - 2025-03-22

### Fixed
- Fixed syntax error in model verification code caused by improper string escaping in f-strings
- Removed single quotes from test message to prevent Python syntax errors
- Properly escaped quotes in API test commands for both OpenAI and Anthropic providers

## [0.2.9] - 2025-03-22

### Fixed
- Fixed `NameError: name 'BenchmarkRunner' is not defined` by resolving name collision between the benchmark_runner module and class instance
- Renamed benchmark runner instances to avoid conflicts throughout the application

## [0.2.8] - 2025-03-22

### Added
- Added model connection verification system that tests each model before running the benchmark
- Added detailed feedback during model verification process with success/failure indicators
- Added visual model information display showing which models are being tested
- Added enhanced error tracing with detailed stack traces for easier debugging

### Changed
- Improved error handling throughout the benchmark process with better user feedback
- Enhanced logging with more detailed information about command execution
- Restructured benchmark execution to detect and report errors more effectively

## [0.2.7] - 2025-03-22

### Fixed
- Fixed MITRE benchmark execution by correctly referencing the local mitre_benchmark.py script instead of using a Python module import
- Added explicit dataset path for MITRE benchmark to ensure it uses the correct test cases

## [0.2.6] - 2025-03-22

### Fixed
- Fixed API key retrieval error by changing from `config.get_api_key()` to `utils.get_api_key()` in app.py
- Resolved "module 'config' has no attribute 'get_api_key'" error when running benchmarks

## [0.2.5] - 2025-03-22

### Changed
- Replaced dropdown model selection with direct text input fields for all model names
- Set default model name values to simplify user input
- Improved guidance with examples of how to format model names
- Enhanced user interface for direct model input for MITRE and FRR benchmarks

## [0.2.4] - 2025-03-22

### Added
- Enhanced MITRE benchmark interface with selection for all three required models (primary, expansion, judge)
- Added custom model input option to allow testing of any model beyond the predefined dropdown options
- Added explanation of MITRE's three-step evaluation process in the UI
- Added parallel processing option for benchmark runs

### Changed
- Improved benchmark runner to support the three-model MITRE workflow
- Reorganized model selection interface to adapt based on benchmark type

## [0.2.3] - 2025-03-22

### Fixed
- Fixed Python import structure in app.py to resolve module import errors
- Added __init__.py to make the app directory a proper Python package
- Created main.py entry point in the root directory for easier application launching
- Fixed relative imports in config.py and benchmark_runner.py
- Added proper sys.path configuration to ensure modules can be found

## [0.2.2] - 2025-03-22

### Added
- Added requirements.txt with project dependencies
- Added app.py with main Streamlit application code
- Added benchmark_runner.py for interfacing with CyberSecEval 3 tools
- Added config.py for API key management and model configuration
- Added utils.py with utility functions for file operations and API handling
- Added config.yaml with application configuration settings
- Added directory structure for data storage and test cases

## [0.2.1] - 2025-03-22

### Added
- Added test.md file for testing purposes

## [0.2.0] - 2025-03-22

### Added
- Initial project structure
- Basic Streamlit application setup
- API key management functionality
- Integration with Purple Llama's CyberSecEval 3 tool
- Support for MITRE and FRR benchmark tests
- Results comparison page
- GitHub integration
- Windsurf IDE integration
- GitHub MCP server configuration for Windsurf
- Updated project setup instructions for Windsurf
- File creation workflow using Cascade
- Detailed GitHub Personal Access Token setup guide

## [0.1.0] - 2025-03-22
### Added
- Deployment guide
- Setup instructions for environment and dependencies
- GitHub MCP server configuration
- Basic application structure
- Documentation for MITRE and FRR benchmarks integration

### To Do
- Implement Streamlit UI
- Create API key management system
- Build benchmark runner integration
- Develop results visualization dashboard
- Add support for additional CyberSecEval 3 benchmarks

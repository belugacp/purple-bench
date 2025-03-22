# Purple Bench - LLM Security Benchmark Tool

## Overview
Purple Bench is a Streamlit application that integrates with Purple Llama's CyberSecEval 3 tool to run security benchmarks against Large Language Models (LLMs). This tool allows users to evaluate the security posture of different LLMs using standardized benchmark tests.

## Features
- Integration with Purple Llama's CyberSecEval 3 benchmark tools
- Support for MITRE and FRR benchmark tests
- Secure API key management with local storage option
- Visual indicators for successful API connections
- Comparative analysis of up to 4 model benchmark results
- Unique result filenames based on model and timestamp
- Visual feedback for benchmark progress

## Requirements
- Python 3.8+
- Streamlit
- Purple Llama's CyberSecEval 3 tools
- macOS (tested on M3 Max processor with 128GB RAM)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/purple-bench.git
cd purple-bench

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Start the Streamlit application
cd app
streamlit run app.py
```

## Deployment
See the deployment guide for detailed setup instructions and advanced configuration options.

## Project Structure
```
purple-bench/
├── app/                  # Streamlit application files
│   ├── app.py            # Main Streamlit application
│   ├── benchmark_runner.py # Interface with CyberSecEval 3
│   ├── config.py         # API key management
│   └── utils.py          # Utility functions
├── config/               # Configuration files
│   └── config.yaml       # Application configuration
├── data/                 # Benchmark results
│   └── test_cases/       # Test case data
├── logs/                 # Log files
├── .gitignore            # Git ignore file
├── CHANGELOG.md          # Version history
├── README.md             # Project documentation
└── RECOMMENDATIONS.md    # Tool/enhancement suggestions
```

## License
MIT

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
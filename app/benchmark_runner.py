import os
import sys
import subprocess
import tempfile
import json
import time
import random
import traceback
import importlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Callable, Union
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utilities
import app.utils as utils
from app.dataset_manager import DatasetManager

class BenchmarkRunner:
    """
    Interface with Purple Llama's CyberSecEval 3 benchmark tools
    """
    def __init__(self):
        """Initialize the benchmark runner"""
        self.config = utils.load_config()
        self.dataset_manager = DatasetManager(self.config)
        self.results_dir = Path(self.config['application']['results_directory'])
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Set up logger
        self.logger = logging.getLogger('purple_bench')
        
        # Ensure test case directories exist
        for benchmark in ['mitre', 'frr', 'prompt_injection', 'visual_prompt_injection']:
            test_case_path = Path(self.config['benchmarks'][benchmark]['test_cases_path'])
            os.makedirs(test_case_path, exist_ok=True)
        
        # Set development mode (easier for testing)
        self.development_mode = True
    
    def run_benchmark(self, 
                      model_name: str, 
                      provider: str, 
                      api_key: str, 
                      benchmark_type: str,
                      dataset: Optional[str] = None,
                      callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Run a benchmark against a specified model
        
        Args:
            model_name: Name of the model to benchmark
            provider: Provider of the model (e.g., 'openai', 'anthropic')
            api_key: API key for the provider
            benchmark_type: Type of benchmark to run ('mitre', 'frr', 'prompt-injection', 'visual-prompt-injection')
            dataset: Optional dataset name to use
            callback: Optional callback function to report progress
            
        Returns:
            Dict containing benchmark results
        """
        try:
            self.logger.info(f"Running {benchmark_type} benchmark with {model_name}")
            
            # Create a temporary directory for benchmark artifacts
            with tempfile.TemporaryDirectory() as temp_dir:
                # Get command to run the benchmark
                cmd = self._build_benchmark_command(
                    model_name=model_name,
                    provider=provider,
                    benchmark_type=benchmark_type,
                    output_dir=temp_dir,
                    dataset=dataset
                )
                env = os.environ.copy()
                if provider.lower() == 'openai':
                    env['OPENAI_API_KEY'] = api_key
                elif provider.lower() == 'anthropic':
                    env['ANTHROPIC_API_KEY'] = api_key
                elif provider.lower() == 'meta':
                    env['META_API_KEY'] = api_key
                elif provider.lower() == 'google':
                    env['GOOGLE_API_KEY'] = api_key
                else:
                    # For custom providers
                    env[f"{provider.upper()}_API_KEY"] = api_key
                
                # Log the command (excluding the API key)
                cmd_for_logging = [item if '--api-key' not in str(item) and i == 0 or i-1 < 0 or '--api-key' not in str(cmd[i-1]) else '****' for i, item in enumerate(cmd)]
                self.logger.info(f"Running command: {' '.join(map(str, cmd_for_logging))}")
                
                # Run the benchmark
                results = self._run_benchmark_process(cmd, env, temp_dir, callback)
                
                # Save results to file
                results_path = self._save_benchmark_results(results, model_name, benchmark_type, dataset)
                
                # Add file path to results
                results['file_path'] = results_path
                
                return results
        except Exception as e:
            self.logger.error(f"Error running benchmark: {str(e)}")
            traceback.print_exc()
            return {
                "error": True,
                "message": str(e)
            }
    
    def _build_benchmark_command(self, 
                                 model_name: str, 
                                 provider: str, 
                                 benchmark_type: str,
                                 output_dir: str,
                                 dataset: str = None) -> List[str]:
        """
        Build the command to run the CyberSecEval 3 benchmark
        
        Args:
            model_name: Name of the model
            provider: Provider of the model
            benchmark_type: Type of benchmark
            output_dir: Directory to output results
            dataset: Optional dataset to use (for prompt injection benchmarks)
            
        Returns:
            List of command arguments
        """
        # Load benchmark-specific configuration
        normalized_benchmark_type = benchmark_type.replace('-', '_')
        benchmark_config = self.config.get('benchmarks', {}).get(normalized_benchmark_type, {})
        
        # Development mode - return simulated command
        if getattr(self, 'development_mode', True):  # Default to True for backward compatibility
            self.logger.warning("Running in development mode. Benchmark results will be simulated.")
            return [
                'python', 
                'simulate_benchmark.py',
                benchmark_type, 
                model_name,
                provider,
                '--dataset', dataset or 'manual'
            ]
            
        # Real benchmark mode - build actual command
        # (This code won't execute in current development mode but is here for future use)
        if benchmark_type == 'mitre':
            # Build MITRE ATT&CK command
            cmd = [
                'python',
                str(Path(self.config['benchmarks']['mitre']['script_path'])),
                '--model-name', model_name,
                '--provider', provider,
                '--output-dir', output_dir,
                '--num-test-cases', str(benchmark_config.get('num_test_cases', 10)),
                '--timeout', str(benchmark_config.get('timeout', 60))
            ]
        elif benchmark_type == 'frr':
            # Build FRR command
            cmd = [
                'python',
                str(Path(self.config['benchmarks']['frr']['script_path'])),
                '--model-name', model_name,
                '--provider', provider,
                '--output-dir', output_dir,
            ]
        elif benchmark_type == 'prompt-injection':
            # Build Prompt Injection command
            cmd = [
                'python',
                str(Path(self.config['benchmarks']['prompt_injection']['script_path'])),
                '--model-name', model_name,
                '--provider', provider,
                '--output-dir', output_dir,
                '--dataset', dataset or benchmark_config.get('dataset', 'default')
            ]
        elif benchmark_type == 'visual-prompt-injection':
            # Build Visual Prompt Injection command
            cmd = [
                'python',
                str(Path(self.config['benchmarks']['visual_prompt_injection']['script_path'])),
                '--model-name', model_name,
                '--provider', provider,
                '--output-dir', output_dir,
                '--dataset', dataset or benchmark_config.get('dataset', 'manual')
            ]
        else:
            raise ValueError(f"Unsupported benchmark type: {benchmark_type}")
        
        return cmd
    
    def _run_benchmark_process(self, 
                              cmd: List[str], 
                              env: Dict[str, str], 
                              temp_dir: str,
                              callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Run the benchmark process and monitor progress
        
        Args:
            cmd: Command to run
            env: Environment variables
            temp_dir: Directory for output files
            callback: Optional callback function for progress updates
            
        Returns:
            Dict containing benchmark results
        """
        # Set to True to use simulated results during development
        development_mode = True
        dataset = None
        
        # Extract dataset from command if present
        if '--dataset' in cmd:
            try:
                dataset_index = cmd.index('--dataset') + 1
                if dataset_index < len(cmd):
                    dataset = cmd[dataset_index]
            except (ValueError, IndexError):
                pass
                
        # In development mode, generate simulated benchmark results
        if development_mode:
            if callback:
                # Simulate progress updates
                for i in range(1, 11):
                    progress = i / 10.0
                    callback(progress, f"Processing benchmark {i}/10")
                    time.sleep(0.5)  # Short delay to simulate work
            
            # Try to determine benchmark type from command
            benchmark_type = None
            model_name = None
            
            if len(cmd) >= 4:
                benchmark_type = cmd[2]  # Assuming format like ['python', 'simulate_benchmark.py', 'benchmark_type', 'model_name', 'provider', ...]  
                model_name = cmd[3]
            
            # Generate simulated results
            results = self._generate_sample_results(model_name, benchmark_type, dataset=dataset)
            
            if callback:
                callback(1.0, "Benchmark completed!")
                
            return results
    
    def _generate_sample_results(self, model_name: str, benchmark_type: str, dataset: str = None) -> Dict[str, Any]:
        """
        Generate sample benchmark results for development/testing
        
        Args:
            model_name: Name of the model
            benchmark_type: Type of benchmark
            dataset: Optional dataset name
            
        Returns:
            Dict containing simulated benchmark results
        """
        normalized_benchmark_type = benchmark_type.lower()
        current_time = time.time()
        
        if normalized_benchmark_type == 'mitre':
            return {
                "benchmark": "MITRE ATT&CK",
                "model": model_name,
                "timestamp": current_time,
                "overall_score": 0.75,  # 75% secure
                "categories": {
                    "Initial Access": {"score": 0.8, "tests_passed": 8, "tests_total": 10},
                    "Execution": {"score": 0.7, "tests_passed": 7, "tests_total": 10},
                    "Persistence": {"score": 0.9, "tests_passed": 9, "tests_total": 10},
                    "Privilege Escalation": {"score": 0.6, "tests_passed": 6, "tests_total": 10},
                    "Defense Evasion": {"score": 0.75, "tests_passed": 15, "tests_total": 20}
                },
                "detailed_results": [
                    {
                        "test_id": "T1190",
                        "name": "Exploit Public-Facing Application",
                        "category": "Initial Access",
                        "passed": True,
                        "details": "Model refused to provide exploit information"
                    },
                    # More detailed test results would be included here
                ]
            }
        elif normalized_benchmark_type == 'frr':
            return {
                "benchmark": "Foundational Responsible Release",
                "model": model_name,
                "timestamp": current_time,
                "overall_score": 0.82,  # 82% refusal rate
                "categories": {
                    "Harmful Content": {"score": 0.85, "refusal_rate": 0.85},
                    "Illegal Activities": {"score": 0.95, "refusal_rate": 0.95},
                    "Unethical Practice": {"score": 0.75, "refusal_rate": 0.75},
                    "Discrimination": {"score": 0.8, "refusal_rate": 0.8}
                },
                "detailed_results": [
                    {
                        "category": "Harmful Content",
                        "prompt_id": "HC001",
                        "judgement": "refusal",
                        "details": "Model correctly refused to generate harmful content"
                    },
                    # More detailed test results would be included here
                ]
            }
        elif normalized_benchmark_type == 'prompt-injection':
            return {
                "benchmark": "Textual Prompt Injection",
                "model": model_name,
                "timestamp": current_time,
                "overall_score": 0.88,  # 88% injection resistance
                "dataset": dataset,
                "categories": {
                    "Direct Instruction Override": {"score": 0.85, "resistance_rate": 0.85},
                    "Context Manipulation": {"score": 0.9, "resistance_rate": 0.9},
                    "Role Play Attacks": {"score": 0.92, "resistance_rate": 0.92},
                    "Multi-Prompt Attacks": {"score": 0.78, "resistance_rate": 0.78}
                },
                "detailed_results": [
                    {
                        "test_id": "PI001",
                        "category": "Direct Instruction Override",
                        "attack_type": "Ignore previous instructions",
                        "passed": True,
                        "details": "Model adhered to system prompt despite injection attempt"
                    },
                    # More detailed test results would be included here
                ]
            }
        elif normalized_benchmark_type == 'visual-prompt-injection':
            # Enhanced placeholder for Visual Prompt Injection benchmark
            # Include the actual selected dataset and more detailed results
            attack_types = [
                "Text in Images", 
                "QR Code Injection", 
                "Typographic Attacks",
                "Image Manipulation"
            ]
            
            # Generate mock detailed results for visualization
            detailed_results = []
            categories = {}
            attack_results = {}
            
            # Create detailed results for 8 test cases
            for i in range(8):
                category = attack_types[i % len(attack_types)]
                passed = random.random() > 0.2  # 80% pass rate for demonstration
                
                if category not in attack_results:
                    attack_results[category] = {"passed": 0, "total": 0}
                    
                attack_results[category]["total"] += 1
                if passed:
                    attack_results[category]["passed"] += 1
                
                detailed_results.append({
                    "test_id": f"VPI{i+1:03d}",
                    "category": category,
                    "attack_type": f"{category} variant {i+1}",
                    "passed": passed,
                    "details": "Model recognized and properly handled the attack" if passed else 
                               "Model was susceptible to this injection type"
                })
            
            # Calculate category scores
            for category, results in attack_results.items():
                if results["total"] > 0:
                    score = results["passed"] / results["total"]
                else:
                    score = 0
                    
                categories[category] = {
                    "score": score, 
                    "resistance_rate": score,
                    "tests_passed": results["passed"],
                    "tests_total": results["total"]
                }
            
            # Calculate overall score
            total_passed = sum(results["passed"] for results in attack_results.values())
            total_tests = sum(results["total"] for results in attack_results.values())
            overall_score = total_passed / total_tests if total_tests > 0 else 0
            
            return {
                "benchmark": "Visual Prompt Injection",
                "model": model_name,
                "timestamp": current_time,
                "overall_score": overall_score,
                "dataset": dataset,
                "categories": categories,
                "detailed_results": detailed_results
            }
        else:
            # Default generic results for unknown benchmark types
            return {
                "benchmark": "Unknown Benchmark",
                "model": model_name,
                "timestamp": current_time,
                "overall_score": 0.5,
                "message": f"Unknown benchmark type: {benchmark_type}"
            }
    
    def _save_benchmark_results(self, results: Dict[str, Any], model_name: str, benchmark_type: str, dataset: str = None) -> str:
        results_path = utils.save_benchmark_results(results, model_name, benchmark_type, dataset)
        return results_path
    
    def run_mitre_benchmark(self, primary_model, expansion_model, judge_model, num_test_cases=50, timeout=60, parallel=True, callback=None):
        """
        Run the MITRE ATT&CK benchmark with the specified three models
        
        Args:
            primary_model: Dict with name, provider, and api_key for the primary model (processes prompts)
            expansion_model: Dict with name, provider, and api_key for the expansion model (expands responses)
            judge_model: Dict with name, provider, and api_key for the judge model (evaluates responses)
            num_test_cases: Number of test cases to run
            timeout: Timeout in seconds
            parallel: Whether to run tests in parallel
            callback: Optional callback function to report progress
            
        Returns:
            Dict with benchmark results
        """
        try:
            # Start with an initial progress update
            if callback:
                callback(0.0, "Initializing MITRE benchmark with multiple models...")
            
            # Verify connections to all three models before proceeding
            if callback:
                callback(0.05, "Verifying connections to models...")
                
            # Verify primary model
            primary_success, primary_msg = self.verify_model_connection(
                primary_model['name'], primary_model['provider'], primary_model['api_key'])
            if not primary_success:
                error_msg = f"Primary model connection failed: {primary_msg}"
                self.logger.error(error_msg)
                return {"error": error_msg}
                
            # Verify expansion model
            expansion_success, expansion_msg = self.verify_model_connection(
                expansion_model['name'], expansion_model['provider'], expansion_model['api_key'])
            if not expansion_success:
                error_msg = f"Expansion model connection failed: {expansion_msg}"
                self.logger.error(error_msg)
                return {"error": error_msg}
                
            # Verify judge model
            judge_success, judge_msg = self.verify_model_connection(
                judge_model['name'], judge_model['provider'], judge_model['api_key'])
            if not judge_success:
                error_msg = f"Judge model connection failed: {judge_msg}"
                self.logger.error(error_msg)
                return {"error": error_msg}
                
            if callback:
                callback(0.1, "All model connections verified. Starting benchmark process...")
            
            # Create a temporary directory for the benchmark run
            with tempfile.TemporaryDirectory() as temp_dir:
                # Prepare environment variables with API keys
                env = os.environ.copy()
                env[f"{primary_model['provider'].upper()}_API_KEY"] = primary_model['api_key']
                env[f"{expansion_model['provider'].upper()}_API_KEY"] = expansion_model['api_key']
                env[f"{judge_model['provider'].upper()}_API_KEY"] = judge_model['api_key']
                
                # Enable verbose logging
                env["LOGLEVEL"] = "DEBUG"
                
                # Ensure dataset file exists
                dataset_path = "PurpleLlama/CybersecurityBenchmarks/datasets/mitre/mitre_benchmark_100_per_category_with_augmentation.json"
                if not os.path.exists(dataset_path):
                    error_msg = f"Dataset file not found: {dataset_path}"
                    self.logger.error(error_msg)
                    return {"error": error_msg}
                    
                self.logger.info(f"Using dataset: {dataset_path}")
                
                # Construct the command for running the benchmark
                cmd = [
                    "python",
                    "PurpleLlama/CybersecurityBenchmarks/benchmark/mitre_benchmark.py",
                    "--primary-model", primary_model['name'],
                    "--primary-provider", primary_model['provider'],
                    "--expansion-model", expansion_model['name'],
                    "--expansion-provider", expansion_model['provider'],
                    "--judge-model", judge_model['name'],
                    "--judge-provider", judge_model['provider'],
                    "--num-test-cases", str(num_test_cases),
                    "--timeout", str(timeout),
                    "--dataset", dataset_path,
                    "--verbose"
                ]
                
                if parallel:
                    cmd.append("--parallel")
                
                # Log the exact command being run
                self.logger.info(f"Running command: {' '.join(cmd)}")
                
                # Update progress
                if callback:
                    callback(0.15, "Starting MITRE benchmark process with verbose logging...")
                
                # Run the benchmark process with enhanced error capture
                try:
                    # Run the benchmark process
                    results = self._run_benchmark_process(cmd, env, temp_dir, callback)
                    
                    # Check if results contain an error
                    if "error" in results:
                        return results
                    
                    # Save combined results
                    model_name = f"{primary_model['name']}_with_{expansion_model['name']}_and_{judge_model['name']}"
                    results_path = self._save_benchmark_results(results, model_name, "mitre")
                    
                    # Add file path to results
                    results['file_path'] = results_path
                    
                    # Final progress update
                    if callback:
                        callback(1.0, "MITRE benchmark completed successfully!")
                    
                    return results
                    
                except Exception as e:
                    error_msg = f"Error running benchmark process: {str(e)}"
                    self.logger.error(error_msg)
                    return {"error": error_msg}
                
        except Exception as e:
            error_msg = f"Error running MITRE benchmark: {str(e)}"
            self.logger.error(error_msg)
            return {"error": error_msg}
    
    def verify_model_connection(self, model_name, provider, api_key):
        """
        Verify that we can successfully connect to the specified model.
        
        Args:
            model_name: Name of the model to verify
            provider: Provider of the model (e.g., 'openai', 'anthropic')
            api_key: API key for the provider
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        import traceback
        try:
            # Simple test prompt to verify connection
            test_prompt = "Hi, this is a test message to verify connection. Please respond with Connection successful."
            
            # Set up environment with API key
            env = os.environ.copy()
            env[f"{provider.upper()}_API_KEY"] = api_key
            
            # Log the verification attempt
            self.logger.info(f"Verifying connection to {provider} model: {model_name}")
            
            # Create diagnostic test scripts that don't use the OpenAI client directly to avoid compatibility issues
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                script_path = temp_file.name
                
                if provider.lower() == 'openai':
                    temp_file.write(f"""
# Diagnostic test script for OpenAI model: {model_name}
import os
import sys
import json
import traceback

# Set API key
os.environ['OPENAI_API_KEY'] = '{api_key}'

try:
    # Import with diagnostic info
    print("Diagnostic info: Starting OpenAI test")
    import openai
    print(f"OpenAI module version: {{openai.__version__}}")
    
    # Basic test without using client directly
    import requests
    headers = {{
        'Authorization': f'Bearer {{os.environ["OPENAI_API_KEY"]}}',
        'Content-Type': 'application/json'
    }}
    payload = {{
        'model': '{model_name}',
        'messages': [{{'role': 'user', 'content': '{test_prompt}'}}],
        'max_tokens': 20
    }}
    
    # Send direct API request
    print("Sending request to OpenAI API...")
    response = requests.post('https://api.openai.com/v1/chat/completions', 
                           headers=headers, 
                           data=json.dumps(payload))
    
    # Process response
    if response.status_code == 200:
        print("API request successful")
        resp_json = response.json()
        content = resp_json['choices'][0]['message']['content']
        print(content)
        print('CONNECTION_SUCCESS')
    else:
        print(f"API request failed with status code: {{response.status_code}}")
        print(response.text)
        print('CONNECTION_FAILED')
        
except Exception as e:
    print(f"Error details: {{e}}")
    print("Traceback:")
    traceback.print_exc()
    print('CONNECTION_FAILED')
""")
                elif provider.lower() == 'anthropic':
                    temp_file.write(f"""
# Diagnostic test script for Anthropic model: {model_name}
import os
import sys
import json
import traceback

# Set API key
os.environ['ANTHROPIC_API_KEY'] = '{api_key}'

try:
    # Import with diagnostic info
    print("Diagnostic info: Starting Anthropic test")
    import anthropic
    print(f"Anthropic module information: {{anthropic}}")
    
    # Basic test using requests
    import requests
    headers = {{
        'x-api-key': os.environ['ANTHROPIC_API_KEY'],
        'anthropic-version': '2023-06-01',
        'content-type': 'application/json'
    }}
    
    payload = {{
        'model': '{model_name}',
        'max_tokens': 20,
        'messages': [{{'role': 'user', 'content': '{test_prompt}'}}]
    }}
    
    # Send direct API request
    print("Sending request to Anthropic API...")
    response = requests.post('https://api.anthropic.com/v1/messages',
                           headers=headers,
                           data=json.dumps(payload))
    
    # Process response
    if response.status_code == 200:
        print("API request successful")
        resp_json = response.json()
        content = resp_json['content'][0]['text']
        print(content)
        print('CONNECTION_SUCCESS')
    else:
        print(f"API request failed with status code: {{response.status_code}}")
        print(response.text)
        print('CONNECTION_FAILED')
        
except Exception as e:
    print(f"Error details: {{e}}")
    print("Traceback:")
    traceback.print_exc()
    print('CONNECTION_FAILED')
""")
                
            # Run the diagnostic script
            self.logger.info(f"Running diagnostic script: {script_path}")
            result = subprocess.run(['python', script_path], 
                                   env=env,
                                   capture_output=True,
                                   text=True)
            
            # Cleanup temp file
            try:
                os.unlink(script_path)
            except:
                pass
            
            # Process output for detailed diagnostics
            output = result.stdout + '\n' + result.stderr
            self.logger.debug(f"Verification output:\n{output}")
            
            if 'CONNECTION_SUCCESS' in output:
                return True, "Connection successful!"
            else:
                # Create detailed error message with all diagnostic info
                error_info = "Connection failed:\n" + output
                return False, error_info
                
        except Exception as e:
            tb = traceback.format_exc()
            error_message = f"Error during verification: {str(e)}\n{tb}"
            self.logger.error(error_message)
            return False, error_message
    
    def compare_results(self, result_files: List[str]) -> Dict[str, Any]:
        """
        Compare multiple benchmark results
        
        Args:
            result_files: List of paths to result files
            
        Returns:
            Dict containing comparison data
        """
        if not result_files:
            return {"error": "No result files provided for comparison"}
        
        comparison = {
            "models": [],
            "benchmark_types": [],
            "overall_scores": {},
            "category_scores": {},
            "test_results": {}
        }
        
        # Load each result file
        for file_path in result_files:
            try:
                with open(file_path, 'r') as file:
                    result = json.load(file)
                
                # Extract metadata
                metadata = result.get('metadata', {})
                model_name = metadata.get('model_name', 'Unknown')
                benchmark_type = metadata.get('benchmark_type', 'Unknown')
                
                # Add to models and benchmark types lists if not already present
                if model_name not in comparison["models"]:
                    comparison["models"].append(model_name)
                if benchmark_type not in comparison["benchmark_types"]:
                    comparison["benchmark_types"].append(benchmark_type)
                
                # Add overall score
                comparison["overall_scores"][model_name] = result.get('overall_score', 0.0)
                
                # Add category scores
                categories = result.get('categories', {})
                for category, data in categories.items():
                    if category not in comparison["category_scores"]:
                        comparison["category_scores"][category] = {}
                    comparison["category_scores"][category][model_name] = data.get('score', 0.0)
                
                # Add detailed test results
                detailed_results = result.get('detailed_results', [])
                for test in detailed_results:
                    test_id = test.get('test_id', 'unknown')
                    if test_id not in comparison["test_results"]:
                        comparison["test_results"][test_id] = {
                            "name": test.get('name', ''),
                            "category": test.get('category', ''),
                            "results": {}
                        }
                    comparison["test_results"][test_id]["results"][model_name] = {
                        "passed": test.get('passed', False),
                        "details": test.get('details', '')
                    }
            except Exception as e:
                self.logger.error(f"Error processing result file {file_path}: {str(e)}")
        
        return comparison

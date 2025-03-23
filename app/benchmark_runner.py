import os
import json
import subprocess
import tempfile
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import time
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

from .utils import load_config, save_benchmark_results, logger
from .dataset_manager import DatasetManager


class BenchmarkRunner:
    """
    Interface with Purple Llama's CyberSecEval 3 benchmark tools
    """
    def __init__(self):
        self.config = load_config()
        self.results_dir = Path(self.config['application']['results_directory'])
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Create dataset manager
        self.dataset_manager = DatasetManager(self.config)
        
        # Ensure test case directories exist
        for benchmark in ['mitre', 'frr', 'prompt_injection', 'visual_prompt_injection']:
            if benchmark in self.config['benchmarks']:
                test_case_path = Path(self.config['benchmarks'][benchmark]['test_cases_path'])
                os.makedirs(test_case_path, exist_ok=True)
        
        # Set up logger
        self.logger = logging.getLogger('purple_bench.benchmark_runner')
    
    def run_benchmark(self, 
                      model_name: str, 
                      provider: str, 
                      api_key: str, 
                      benchmark_type: str,
                      dataset: Optional[str] = None,
                      callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Run a benchmark against a specified model
        
        Args:
            model_name: Name of the model to benchmark
            provider: Provider of the model (e.g., 'openai', 'anthropic')
            api_key: API key for the provider
            benchmark_type: Type of benchmark to run ('mitre', 'frr', 'prompt_injection', 'visual_prompt_injection')
            dataset: Optional dataset name for benchmarks that require datasets
            callback: Optional callback function to report progress
            
        Returns:
            Dict containing benchmark results
        """
        supported_benchmarks = ['mitre', 'frr', 'prompt_injection', 'visual_prompt_injection']
        normalized_benchmark_type = benchmark_type.lower().replace('-', '_')
        
        if normalized_benchmark_type not in supported_benchmarks:
            error_msg = f"Unsupported benchmark type: {benchmark_type}. Supported types: {', '.join(supported_benchmarks)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Check if dataset is required but not provided
        if normalized_benchmark_type in ['prompt_injection', 'visual_prompt_injection'] and not dataset:
            error_msg = f"Dataset must be specified for benchmark type: {benchmark_type}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Ensure dataset exists for dataset-based benchmarks
        if dataset and not self.dataset_manager.dataset_exists(normalized_benchmark_type, dataset):
            self.logger.info(f"Dataset {dataset} not found locally. Attempting to download...")
            if not self.dataset_manager.download_dataset(normalized_benchmark_type, dataset):
                error_msg = f"Failed to download dataset: {dataset}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
        
        # Set up environment variables for API keys
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
        
        # Create temporary directory for benchmark results
        with tempfile.TemporaryDirectory() as temp_dir:
            # Build command to run CyberSecEval 3
            cmd = self._build_benchmark_command(
                model_name=model_name,
                provider=provider,
                benchmark_type=normalized_benchmark_type,
                output_dir=temp_dir,
                dataset=dataset
            )
            
            # Run benchmark in a separate thread to avoid blocking the UI
            results = self._run_benchmark_process(cmd, env, temp_dir, callback)
            
            # Add dataset info to results if applicable
            if dataset:
                results['dataset'] = dataset
            
            # Save results to file
            results_path = save_benchmark_results(results, model_name, benchmark_type, dataset)
            
            # Add file path to results
            results['file_path'] = results_path
            
            return results
    
    def _build_benchmark_command(self, 
                                model_name: str, 
                                provider: str, 
                                benchmark_type: str,
                                output_dir: str,
                                dataset: Optional[str] = None) -> List[str]:
        """
        Build the command to run the CyberSecEval 3 benchmark
        
        Args:
            model_name: Name of the model
            provider: Provider of the model
            benchmark_type: Type of benchmark
            output_dir: Directory to output results
            dataset: Optional dataset name for benchmarks that require datasets
            
        Returns:
            List of command arguments
        """
        # This would need to be adapted to the actual CyberSecEval 3 CLI interface
        # The following is a placeholder based on typical CLI patterns
        
        if benchmark_type == 'mitre':
            return [
                "cyberseceval",  # Replace with actual command
                "run",
                "--model", model_name,
                "--provider", provider,
                "--benchmark", "mitre",
                "--output-dir", output_dir,
                "--format", "json"
            ]
        elif benchmark_type == 'frr':
            return [
                "cyberseceval",  # Replace with actual command
                "run",
                "--model", model_name,
                "--provider", provider,
                "--benchmark", "frr",
                "--output-dir", output_dir,
                "--format", "json"
            ]
        elif benchmark_type == 'prompt_injection':
            dataset_path = self.dataset_manager.get_dataset_path(benchmark_type, dataset)
            return [
                "cyberseceval",  # Replace with actual command
                "run",
                "--model", model_name,
                "--provider", provider,
                "--benchmark", "prompt-injection",
                "--dataset", dataset_path,
                "--output-dir", output_dir,
                "--format", "json"
            ]
        elif benchmark_type == 'visual_prompt_injection':
            dataset_path = self.dataset_manager.get_dataset_path(benchmark_type, dataset)
            return [
                "cyberseceval",  # Replace with actual command
                "run",
                "--model", model_name,
                "--provider", provider,
                "--benchmark", "visual-prompt-injection",
                "--dataset", dataset_path,
                "--output-dir", output_dir,
                "--format", "json"
            ]
    
    def _run_benchmark_process(self, 
                              cmd: List[str], 
                              env: Dict[str, str], 
                              output_dir: str,
                              callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Run the benchmark process and monitor progress
        
        Args:
            cmd: Command to run
            env: Environment variables
            output_dir: Directory for output files
            callback: Optional callback function for progress updates
            
        Returns:
            Dict containing benchmark results
        """
        # For development/testing, we'll simulate the benchmark process
        # In production, this would call the actual CyberSecEval 3 tool
        
        # Simulated benchmark process for development
        if callback:
            callback(0.0, "Initializing benchmark...")
        
        # In a real implementation, we would run the subprocess here
        # process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # For now, simulate the benchmark running with progress updates
        total_steps = 10
        for i in range(total_steps + 1):
            if callback:
                progress = i / total_steps
                status = f"Running benchmark step {i}/{total_steps}..."
                callback(progress, status)
            time.sleep(0.5)  # Simulate processing time
        
        # Simulate reading results from output directory
        # In production, we would parse the actual output files
        
        # Extract benchmark_type from the command
        benchmark_type = None
        model_name = None
        dataset = None
        
        # Parse command to identify benchmark type and other parameters
        if 'mitre' in ' '.join(cmd):
            benchmark_type = 'mitre'
        elif 'frr' in ' '.join(cmd):
            benchmark_type = 'frr'
        elif 'prompt_injection' in ' '.join(cmd):
            benchmark_type = 'prompt-injection'
        elif 'visual_prompt_injection' in ' '.join(cmd):
            benchmark_type = 'visual-prompt-injection'
            
        # Extract model name
        try:
            if '--model-name' in cmd:
                model_name_index = cmd.index('--model-name') + 1
                if model_name_index < len(cmd):
                    model_name = cmd[model_name_index]
            elif '--primary-model' in cmd:
                model_name_index = cmd.index('--primary-model') + 1
                if model_name_index < len(cmd):
                    model_name = cmd[model_name_index]
                    
            # Extract dataset if present
            if '--dataset' in cmd:
                dataset_index = cmd.index('--dataset') + 1
                if dataset_index < len(cmd):
                    dataset = cmd[dataset_index]
        except (ValueError, IndexError):
            pass
        
        # Generate sample results
        results = self._generate_sample_results(model_name, benchmark_type, dataset=dataset)
        
        if callback:
            callback(1.0, "Benchmark completed!")
        
        return results
    
    def _generate_sample_results(self, model_name: str, benchmark_type: str, dataset: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate sample benchmark results for development/testing
        
        Args:
            model_name: Name of the model
            benchmark_type: Type of benchmark
            dataset: Optional dataset name
            
        Returns:
            Dict containing simulated benchmark results
        """
        if benchmark_type == 'mitre':
            return {
                "benchmark": "MITRE ATT&CK",
                "model": model_name,
                "timestamp": time.time(),
                "overall_score": 0.75,  # Sample score
                "categories": {
                    "initial_access": {
                        "score": 0.82,
                        "tests_passed": 9,
                        "tests_total": 11
                    },
                    "execution": {
                        "score": 0.67,
                        "tests_passed": 8,
                        "tests_total": 12
                    },
                    "persistence": {
                        "score": 0.78,
                        "tests_passed": 7,
                        "tests_total": 9
                    },
                    "privilege_escalation": {
                        "score": 0.73,
                        "tests_passed": 11,
                        "tests_total": 15
                    },
                    "defense_evasion": {
                        "score": 0.69,
                        "tests_passed": 9,
                        "tests_total": 13
                    }
                },
                "detailed_results": [
                    {
                        "test_id": "T1190",
                        "name": "Exploit Public-Facing Application",
                        "category": "initial_access",
                        "passed": True,
                        "details": "Model refused to provide exploit code"
                    },
                    {
                        "test_id": "T1059",
                        "name": "Command and Scripting Interpreter",
                        "category": "execution",
                        "passed": False,
                        "details": "Model provided potentially harmful script"
                    },
                    # Additional test results would be included here
                ]
            }
        elif benchmark_type == 'frr':
            return {
                "benchmark": "Foundational Responsible Release (FRR)",
                "model": model_name,
                "timestamp": time.time(),
                "overall_score": 0.83,  # Sample score
                "categories": {
                    "cybersecurity": {
                        "score": 0.85,
                        "tests_passed": 17,
                        "tests_total": 20
                    },
                    "privacy": {
                        "score": 0.79,
                        "tests_passed": 15,
                        "tests_total": 19
                    },
                    "fairness": {
                        "score": 0.88,
                        "tests_passed": 14,
                        "tests_total": 16
                    },
                    "transparency": {
                        "score": 0.81,
                        "tests_passed": 13,
                        "tests_total": 16
                    }
                },
                "detailed_results": [
                    {
                        "test_id": "FRR-CS-01",
                        "name": "Unauthorized Access Prevention",
                        "category": "cybersecurity",
                        "passed": True,
                        "details": "Model refused to provide unauthorized access methods"
                    },
                    {
                        "test_id": "FRR-P-03",
                        "name": "Personal Data Protection",
                        "category": "privacy",
                        "passed": True,
                        "details": "Model refused to extract personal data from provided text"
                    },
                    # Additional test results would be included here
                ]
            }
        elif benchmark_type == 'prompt_injection':
            return {
                "benchmark": "Prompt Injection",
                "model": model_name,
                "dataset": dataset,
                "timestamp": time.time(),
                "overall_score": 0.68,  # Sample score
                "total_samples": 100,
                "successful_defenses": 68,
                "categories": {
                    "direct_injection": {
                        "score": 0.75,
                        "tests_passed": 15,
                        "tests_total": 20
                    },
                    "indirect_injection": {
                        "score": 0.65,
                        "tests_passed": 13,
                        "tests_total": 20
                    },
                    "goal_hijacking": {
                        "score": 0.60,
                        "tests_passed": 12,
                        "tests_total": 20
                    },
                    "jailbreaking": {
                        "score": 0.70,
                        "tests_passed": 28,
                        "tests_total": 40
                    }
                }
            }
        elif benchmark_type == 'visual_prompt_injection':
            return {
                "benchmark": "Visual Prompt Injection",
                "model": model_name,
                "dataset": dataset,
                "timestamp": time.time(),
                "overall_score": 0.72,  # Sample score
                "total_samples": 50,
                "successful_defenses": 36,
                "categories": {
                    "typographic_attacks": {
                        "score": 0.85,
                        "tests_passed": 11,
                        "tests_total": 13
                    },
                    "adversarial_overlays": {
                        "score": 0.65,
                        "tests_passed": 8,
                        "tests_total": 12
                    },
                    "adversarial_patches": {
                        "score": 0.70,
                        "tests_passed": 7,
                        "tests_total": 10
                    },
                    "qr_codes": {
                        "score": 0.67,
                        "tests_passed": 10,
                        "tests_total": 15
                    }
                }
            }
        else:
            return {"error": f"Unknown benchmark type: {benchmark_type}"}
    
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
                logger.error(f"Error processing result file {file_path}: {str(e)}")
        
        return comparison
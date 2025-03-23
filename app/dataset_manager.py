import os
import subprocess
import shutil
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import sys

# Configure logging
logger = logging.getLogger('purple_bench')

class DatasetManager:
    """
    Manages downloading and accessing datasets for benchmarks
    """
    
    # S3 bucket for Purple Llama datasets
    S3_BUCKET = "s3://purplellama-cyberseceval"
    CYBERSECEVAL_PATH = "cyberseceval3"
    
    # Dataset paths relative to S3 bucket root
    DATASET_PATHS = {
        "prompt_injection": f"{CYBERSECEVAL_PATH}/prompt_injection",
        "visual_prompt_injection": f"{CYBERSECEVAL_PATH}/visual_prompt_injection"
    }
    
    # Known datasets for each type (from S3)
    KNOWN_DATASETS = {
        "visual_prompt_injection": [
            "cse2_typographic_images",
            "cse2_visual_overlays",
            "cse2_adversarial_patches",
            "cse2_adversarial_qr_codes"
        ],
        "prompt_injection": [
            "prompt_injection.json",
            "prompt_injection_multilingual_machine_translated.json"
        ]
    }
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the dataset manager with the application config
        """
        self.config = config
        
        # Create data/datasets directory if it doesn't exist
        self.datasets_base_dir = os.path.join(
            Path(self.config['application']['results_directory']).parent, 
            "datasets"
        )
        os.makedirs(self.datasets_base_dir, exist_ok=True)
        
        # Create .gitignore in the datasets directory if it doesn't exist
        self._create_datasets_gitignore()
    
    def _create_datasets_gitignore(self):
        """
        Create a .gitignore file in the datasets directory to prevent committing large datasets
        """
        gitignore_path = os.path.join(self.datasets_base_dir, ".gitignore")
        if not os.path.exists(gitignore_path):
            with open(gitignore_path, "w") as f:
                f.write("# Ignore all dataset files\n*\n# But keep the directory\n!.gitignore\n")
    
    def get_dataset_path(self, benchmark_type: str, dataset_name: str) -> str:
        """
        Get the path to a specific dataset
        
        Args:
            benchmark_type: Type of benchmark (e.g., 'prompt_injection')
            dataset_name: Name of the dataset
            
        Returns:
            Path to the dataset file or directory
        """
        normalized_benchmark_type = benchmark_type.replace('-', '_')
        dataset_dir = os.path.join(self.datasets_base_dir, normalized_benchmark_type)
        os.makedirs(dataset_dir, exist_ok=True)
        
        return os.path.join(dataset_dir, dataset_name)
    
    def download_dataset(self, benchmark_type: str, dataset_name: str) -> bool:
        """
        Download a dataset from S3
        
        Args:
            benchmark_type: Type of benchmark
            dataset_name: Name of dataset
            
        Returns:
            True if successful, False otherwise
        """
        normalized_benchmark_type = benchmark_type.replace('-', '_')
        if normalized_benchmark_type not in self.DATASET_PATHS:
            logger.error(f"Unknown benchmark type: {benchmark_type}")
            return False
        
        s3_path = os.path.join(self.S3_BUCKET, self.DATASET_PATHS[normalized_benchmark_type], dataset_name)
        local_path = self.get_dataset_path(benchmark_type, dataset_name)
        
        # Use AWS CLI to download
        try:
            logger.info(f"Downloading dataset from {s3_path} to {local_path}")
            cmd = ["aws", "s3", "cp", s3_path, local_path, "--recursive"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Download failed: {result.stderr}")
                return False
            
            logger.info(f"Dataset downloaded successfully to {local_path}")
            return True
        except Exception as e:
            logger.error(f"Error downloading dataset: {str(e)}")
            return False
    
    def list_available_datasets(self, benchmark_type: str) -> List[str]:
        """
        List available datasets for a benchmark type
        
        Args:
            benchmark_type: Type of benchmark
            
        Returns:
            List of dataset names
        """
        normalized_benchmark_type = benchmark_type.replace('-', '_')
        
        # Check known datasets first
        known_datasets = self.KNOWN_DATASETS.get(normalized_benchmark_type, [])
        
        # Also check the local filesystem for already downloaded datasets
        dataset_dir = os.path.join(self.datasets_base_dir, normalized_benchmark_type)
        if os.path.exists(dataset_dir) and os.path.isdir(dataset_dir):
            try:
                local_datasets = [f for f in os.listdir(dataset_dir) 
                                if f != ".gitignore" and not f.startswith(".")]
                # Combine both lists without duplicates
                return list(set(known_datasets + local_datasets))
            except Exception as e:
                logger.error(f"Error listing local datasets: {str(e)}")
        
        return known_datasets
    
    def dataset_exists(self, benchmark_type: str, dataset_name: str) -> bool:
        """
        Check if a dataset exists locally
        
        Args:
            benchmark_type: Type of benchmark
            dataset_name: Name of dataset
            
        Returns:
            True if dataset exists, False otherwise
        """
        dataset_path = self.get_dataset_path(benchmark_type, dataset_name)
        return os.path.exists(dataset_path)

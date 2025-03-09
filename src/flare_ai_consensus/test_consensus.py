import json
import time
import os
import sys
import argparse
import pandas as pd
import numpy as np
import re
import difflib
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

# Adjust the import to use the correct module path
from src.flare_ai_consensus.consensus_engine import run_consensus, run_single_model, evaluate_response

class ConsensusTestFramework:
    """Framework for testing and benchmarking the consensus system against individual models."""
    
    def __init__(
        self, 
        config_path: str, 
        test_cases_path: str,
        output_dir: str = "test_results",
        num_runs: int = 3,  # Increased to 3 for better statistical significance
        parallel_execution: bool = True
    ):
        """
        Initialize the test framework.
        
        Args:
            config_path: Path to the consensus configuration file
            test_cases_path: Path to test cases JSON file
            output_dir: Directory to save test results
            num_runs: Number of times to run each test for reliability assessment
            parallel_execution: Whether to run tests in parallel for better performance
        """
        self.config_path = config_path
        self.test_cases_path = test_cases_path
        self.output_dir = output_dir
        self.num_runs = num_runs
        self.parallel_execution = parallel_execution
        
        # Load configuration and test cases
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        with open(test_cases_path, 'r') as f:
            self.test_cases = json.load(f)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize results storage
        self.results = {
            "single_model": {},
            "consensus": {},
            "metrics": {
                "accuracy": {},
                "reliability": {},
                "performance": {}
            }
        }
    
    def run_tests(self):
        """Run all tests for both single models and consensus approach with improved performance."""
        print(f"Running tests with {len(self.test_cases['cases'])} test cases...")
        
        # Test individual models
        all_models = [model["id"] for model in self.config["models"]]
        
        if self.parallel_execution:
            # Parallel execution for better performance
            with ThreadPoolExecutor(max_workers=min(len(all_models), 3)) as executor:
                futures = {executor.submit(self._run_model_tests, model_id): model_id for model_id in all_models}
                for future in as_completed(futures):
                    model_id = futures[future]
                    try:
                        self.results["single_model"][model_id] = future.result()
                        print(f"Completed testing model: {model_id}")
                    except Exception as e:
                        print(f"Error testing model {model_id}: {e}")
        else:
            # Sequential execution
            for model_id in all_models:
                print(f"Testing model: {model_id}")
                self.results["single_model"][model_id] = self._run_model_tests(model_id)
        
        # Test consensus approach
        print("Testing consensus approach...")
        self.results["consensus"] = self._run_consensus_tests()
        
        # Calculate metrics
        self._calculate_metrics()
        
        # Save results
        self._save_results()
        
        # Generate visualizations
        self._generate_visualizations()
        
        return self.results
    
    def _run_model_tests(self, model_id: str) -> dict:
        """Run tests for a single model with improved measurement techniques."""
        results = {}
        
        for test_case in self.test_cases["cases"]:
            case_id = test_case["id"]
            results[case_id] = {
                "responses": [],
                "timing": [],
                "evaluation": [],
                "category": self._get_test_case_category(case_id)
            }
            
            # Get context if available
            context = test_case.get("context", None)
            
            for run in range(self.num_runs):
                start_time = time.time()
                response = run_single_model(
                    model_id=model_id,
                    prompt=test_case["prompt"],
                    max_tokens=self.config["models"][0]["max_tokens"],
                    temperature=self.config["models"][0]["temperature"],
                    context=context
                )
                end_time = time.time()
                
                # Record results
                results[case_id]["responses"].append(response)
                results[case_id]["timing"].append(end_time - start_time)
                
                # Evaluate the response
                eval_result = evaluate_response(
                    response=response,
                    ground_truth=test_case["ground_truth"],
                    key_facts=test_case.get("key_facts", [])
                )
                results[case_id]["evaluation"].append(eval_result)
                
        return results
    
    def _run_consensus_tests(self) -> dict:
        """Run tests for the consensus approach with enhanced metrics."""
        results = {}
        
        for test_case in self.test_cases["cases"]:
            case_id = test_case["id"]
            results[case_id] = {
                "responses": [],
                "timing": [],
                "evaluation": [],
                "category": self._get_test_case_category(case_id)
            }
            
            # Get context if available
            context = test_case.get("context", None)
            
            for run in range(self.num_runs):
                start_time = time.time()
                response = run_consensus(
                    config=self.config,
                    prompt=test_case["prompt"],
                    context=context
                )
                end_time = time.time()
                
                # Record results
                results[case_id]["responses"].append(response)
                results[case_id]["timing"].append(end_time - start_time)
                
                # Evaluate the response
                eval_result = evaluate_response(
                    response=response,
                    ground_truth=test_case["ground_truth"],
                    key_facts=test_case.get("key_facts", [])
                )
                results[case_id]["evaluation"].append(eval_result)
                
        return results
    
    def _get_test_case_category(self, case_id: str) -> str:
        """Determine the category of a test case based on its ID."""
        if case_id.startswith("factual"):
            return "factual"
        elif case_id.startswith("reasoning"):
            return "reasoning"
        elif case_id.startswith("technical"):
            return "technical"
        elif case_id.startswith("ambiguous"):
            return "ambiguous"
        elif case_id.startswith("subjective"):
            return "subjective"
        elif case_id.startswith("harmful"):
            return "harmful"
        elif case_id.startswith("controversial"):
            return "controversial"
        elif case_id.startswith("creative"):
            return "creative"
        elif case_id.startswith("edge"):
            return "edge_case"
        elif case_id.startswith("with_context"):
            return "with_context"
        else:
            return "other"
    
    def _calculate_metrics(self):
        """Calculate comprehensive metrics comparing consensus to individual models."""
        # Calculate accuracy metrics
        self.results["metrics"]["accuracy"] = self._calculate_accuracy_metrics()
        
        # Calculate reliability metrics
        self.results["metrics"]["reliability"] = self._calculate_reliability_metrics()
        
        # Calculate performance metrics
        self.results["metrics"]["performance"] = self._calculate_performance_metrics()
        
        # Generate summary of improvements
        self._generate_improvement_summary()
    
    def _calculate_accuracy_metrics(self) -> dict:
        """Calculate enhanced accuracy metrics for evaluation."""
        accuracy_metrics = {
            "overall": {},
            "by_category": {},
            "by_model": {}
        }
        
        # Get all model IDs
        all_models = list(self.results["single_model"].keys())
        best_single_model = None
        best_single_model_score = -1
        
        # Calculate average factual correctness for each model
        for model_id in all_models:
            model_score = 0
            total_cases = 0
            
            for case_id, case_results in self.results["single_model"][model_id].items():
                if len(case_results["evaluation"]) == 0:
                    continue
                
                # Average across runs
                avg_correctness = sum(eval_res["factual_correctness"] for eval_res in case_results["evaluation"]) / len(case_results["evaluation"])
                model_score += avg_correctness
                total_cases += 1
            
            if total_cases > 0:
                model_avg = model_score / total_cases
                accuracy_metrics["by_model"][model_id] = model_avg
                
                # Track best single model
                if model_avg > best_single_model_score:
                    best_single_model_score = model_avg
                    best_single_model = model_id
        
        # Calculate consensus accuracy
        consensus_score = 0
        total_cases = 0
        category_scores = {}
        
        for case_id, case_results in self.results["consensus"].items():
            if len(case_results["evaluation"]) == 0:
                continue
            
            # Average across runs
            avg_correctness = sum(eval_res["factual_correctness"] for eval_res in case_results["evaluation"]) / len(case_results["evaluation"])
            consensus_score += avg_correctness
            total_cases += 1
            
            # Record by category
            category = case_results["category"]
            if category not in category_scores:
                category_scores[category] = {"score": 0, "count": 0}
            
            category_scores[category]["score"] += avg_correctness
            category_scores[category]["count"] += 1
        
        if total_cases > 0:
            consensus_avg = consensus_score / total_cases
            accuracy_metrics["overall"]["consensus"] = consensus_avg
            accuracy_metrics["overall"]["best_single_model"] = best_single_model_score
            accuracy_metrics["overall"]["best_model_id"] = best_single_model
            accuracy_metrics["overall"]["improvement"] = consensus_avg - best_single_model_score
            
            # Calculate by category
            for category, data in category_scores.items():
                if data["count"] > 0:
                    accuracy_metrics["by_category"][category] = data["score"] / data["count"]
        
        return accuracy_metrics
    
    def _compute_semantic_similarity(self, response: str, ground_truth: str) -> float:
        """Compute semantic similarity between response and ground truth."""
        # Enhanced similarity calculation using difflib
        similarity_ratio = difflib.SequenceMatcher(None, response.lower(), ground_truth.lower()).ratio()
        
        # Adjust for length differences
        len_ratio = min(len(response), len(ground_truth)) / max(len(response), len(ground_truth))
        
        return similarity_ratio * 0.7 + len_ratio * 0.3  # Weighted score
    
    def _compute_factual_correctness(self, response: str, ground_truth: str, key_facts: list[str]) -> float:
        """Compute factual correctness of a response based on key facts."""
        if not key_facts:
            return self._compute_semantic_similarity(response, ground_truth)
        
        # Count how many key facts are mentioned
        count = sum(1 for fact in key_facts if fact.lower() in response.lower())
        
        # Return the proportion of key facts mentioned
        return count / len(key_facts)
    
    def _calculate_reliability_metrics(self) -> dict:
        """Calculate enhanced reliability metrics for consistency evaluation."""
        reliability_metrics = {
            "overall": {},
            "by_category": {},
            "by_model": {}
        }
        
        # Get all model IDs
        all_models = list(self.results["single_model"].keys())
        best_single_model = None
        best_single_model_score = -1
        
        # Calculate consistency for each model
        for model_id in all_models:
            model_score = 0
            total_cases = 0
            
            for case_id, case_results in self.results["single_model"][model_id].items():
                if len(case_results["responses"]) <= 1:
                    continue
                
                # Calculate consistency as similarity between different runs
                similarities = []
                responses = case_results["responses"]
                
                for i in range(len(responses)):
                    for j in range(i+1, len(responses)):
                        sim = self._compute_semantic_similarity(responses[i], responses[j])
                        similarities.append(sim)
                
                if similarities:
                    avg_consistency = sum(similarities) / len(similarities)
                    model_score += avg_consistency
                    total_cases += 1
            
            if total_cases > 0:
                model_avg = model_score / total_cases
                reliability_metrics["by_model"][model_id] = model_avg
                
                # Track best single model
                if model_avg > best_single_model_score:
                    best_single_model_score = model_avg
                    best_single_model = model_id
        
        # Calculate consensus consistency
        consensus_score = 0
        total_cases = 0
        category_scores = {}
        
        for case_id, case_results in self.results["consensus"].items():
            if len(case_results["responses"]) <= 1:
                continue
            
            # Calculate consistency as similarity between different runs
            similarities = []
            responses = case_results["responses"]
            
            for i in range(len(responses)):
                for j in range(i+1, len(responses)):
                    sim = self._compute_semantic_similarity(responses[i], responses[j])
                    similarities.append(sim)
            
            if similarities:
                avg_consistency = sum(similarities) / len(similarities)
                consensus_score += avg_consistency
                total_cases += 1
                
                # Record by category
                category = case_results["category"]
                if category not in category_scores:
                    category_scores[category] = {"score": 0, "count": 0}
                
                category_scores[category]["score"] += avg_consistency
                category_scores[category]["count"] += 1
        
        if total_cases > 0:
            consensus_avg = consensus_score / total_cases
            reliability_metrics["overall"]["consensus"] = consensus_avg
            reliability_metrics["overall"]["best_single_model"] = best_single_model_score
            reliability_metrics["overall"]["best_model_id"] = best_single_model
            reliability_metrics["overall"]["improvement"] = consensus_avg - best_single_model_score
            
            # Calculate by category
            for category, data in category_scores.items():
                if data["count"] > 0:
                    reliability_metrics["by_category"][category] = data["score"] / data["count"]
        
        return reliability_metrics
    
    def _calculate_performance_metrics(self) -> dict:
        """Calculate enhanced performance metrics with focus on optimization."""
        performance_metrics = {
            "overall": {},
            "by_category": {},
            "by_model": {}
        }
        
        # Get all model IDs
        all_models = list(self.results["single_model"].keys())
        best_single_model = None
        best_single_model_time = float('inf')
        
        # Calculate average response time for each model
        for model_id in all_models:
            model_time = 0
            total_cases = 0
            
            for case_id, case_results in self.results["single_model"][model_id].items():
                if not case_results["timing"]:
                    continue
                
                # Average time across runs
                avg_time = sum(case_results["timing"]) / len(case_results["timing"])
                model_time += avg_time
                total_cases += 1
            
            if total_cases > 0:
                model_avg = model_time / total_cases
                performance_metrics["by_model"][model_id] = model_avg
                
                # Track best (fastest) single model
                if model_avg < best_single_model_time:
                    best_single_model_time = model_avg
                    best_single_model = model_id
        
        # Calculate consensus performance
        consensus_time = 0
        total_cases = 0
        category_times = {}
        
        for case_id, case_results in self.results["consensus"].items():
            if not case_results["timing"]:
                continue
            
            # Average time across runs
            avg_time = sum(case_results["timing"]) / len(case_results["timing"])
            consensus_time += avg_time
            total_cases += 1
            
            # Record by category
            category = case_results["category"]
            if category not in category_times:
                category_times[category] = {"time": 0, "count": 0}
            
            category_times[category]["time"] += avg_time
            category_times[category]["count"] += 1
        
        if total_cases > 0:
            consensus_avg = consensus_time / total_cases
            performance_metrics["overall"]["consensus"] = consensus_avg
            performance_metrics["overall"]["best_single_model"] = best_single_model_time
            performance_metrics["overall"]["best_model_id"] = best_single_model
            # Calculate overhead as percentage increase (lower is better)
            overhead = ((consensus_avg / best_single_model_time) - 1) * 100
            performance_metrics["overall"]["overhead"] = overhead
            
            # Calculate by category
            for category, data in category_times.items():
                if data["count"] > 0:
                    performance_metrics["by_category"][category] = data["time"] / data["count"]
        
        return performance_metrics
    
    def _generate_improvement_summary(self):
        """Generate a summary of improvements across all metrics."""
        accuracy_improvement = self.results["metrics"]["accuracy"]["overall"].get("improvement", 0) * 100
        reliability_improvement = self.results["metrics"]["reliability"]["overall"].get("improvement", 0) * 100
        performance_overhead = self.results["metrics"]["performance"]["overall"].get("overhead", 0)
        
        summary = {
            "accuracy_improvement": accuracy_improvement,
            "reliability_improvement": reliability_improvement,
            "performance_overhead": performance_overhead
        }
        
        self.results["metrics"]["summary"] = summary
        
        # Print the summary
        print("\nKEY METRICS SUMMARY:")
        print("-" * 80)
        print(f"Accuracy improvement (factual correctness): {accuracy_improvement:.2f}%")
        print(f"Reliability improvement (consistency): {reliability_improvement:.2f}%")
        print(f"Performance overhead: {performance_overhead:.2f}%")
        print("-" * 80)
    
    def _generate_visualizations(self):
        """Generate visualizations of test results for better analysis."""
        # Create a directory for visualizations
        viz_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Generate accuracy comparison chart
        self._generate_accuracy_chart(viz_dir)
        
        # Generate reliability comparison chart
        self._generate_reliability_chart(viz_dir)
        
        # Generate performance comparison chart
        self._generate_performance_chart(viz_dir)
        
        # Generate category comparison
        self._generate_category_comparison(viz_dir)
    
    def _generate_accuracy_chart(self, viz_dir: str):
        """Generate a chart comparing accuracy across models and consensus."""
        try:
            plt.figure(figsize=(10, 6))
            
            # Collect data
            models = list(self.results["metrics"]["accuracy"]["by_model"].keys())
            accuracies = [self.results["metrics"]["accuracy"]["by_model"][model] for model in models]
            
            # Add consensus
            models.append("Consensus")
            accuracies.append(self.results["metrics"]["accuracy"]["overall"]["consensus"])
            
            # Create bar chart
            plt.bar(models, accuracies, color=['blue'] * (len(models)-1) + ['green'])
            plt.axhline(y=self.results["metrics"]["accuracy"]["overall"]["best_single_model"], 
                      color='red', linestyle='--', label='Best Single Model')
            
            plt.xlabel('Models')
            plt.ylabel('Accuracy Score')
            plt.title('Accuracy Comparison')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.legend()
            
            # Save the chart
            plt.savefig(os.path.join(viz_dir, 'accuracy_comparison.png'))
            plt.close()
        except Exception as e:
            print(f"Error generating accuracy chart: {e}")
    
    def _generate_reliability_chart(self, viz_dir: str):
        """Generate a chart comparing reliability across models and consensus."""
        try:
            plt.figure(figsize=(10, 6))
            
            # Collect data
            models = list(self.results["metrics"]["reliability"]["by_model"].keys())
            reliabilities = [self.results["metrics"]["reliability"]["by_model"][model] for model in models]
            
            # Add consensus
            models.append("Consensus")
            reliabilities.append(self.results["metrics"]["reliability"]["overall"]["consensus"])
            
            # Create bar chart
            plt.bar(models, reliabilities, color=['blue'] * (len(models)-1) + ['green'])
            plt.axhline(y=self.results["metrics"]["reliability"]["overall"]["best_single_model"], 
                      color='red', linestyle='--', label='Best Single Model')
            
            plt.xlabel('Models')
            plt.ylabel('Reliability Score')
            plt.title('Reliability Comparison')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.legend()
            
            # Save the chart
            plt.savefig(os.path.join(viz_dir, 'reliability_comparison.png'))
            plt.close()
        except Exception as e:
            print(f"Error generating reliability chart: {e}")
    
    def _generate_performance_chart(self, viz_dir: str):
        """Generate a chart comparing performance across models and consensus."""
        try:
            plt.figure(figsize=(10, 6))
            
            # Collect data
            models = list(self.results["metrics"]["performance"]["by_model"].keys())
            times = [self.results["metrics"]["performance"]["by_model"][model] for model in models]
            
            # Add consensus
            models.append("Consensus")
            times.append(self.results["metrics"]["performance"]["overall"]["consensus"])
            
            # Create bar chart
            plt.bar(models, times, color=['blue'] * (len(models)-1) + ['green'])
            plt.axhline(y=self.results["metrics"]["performance"]["overall"]["best_single_model"], 
                      color='red', linestyle='--', label='Fastest Single Model')
            
            plt.xlabel('Models')
            plt.ylabel('Average Response Time (s)')
            plt.title('Performance Comparison')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.legend()
            
            # Save the chart
            plt.savefig(os.path.join(viz_dir, 'performance_comparison.png'))
            plt.close()
        except Exception as e:
            print(f"Error generating performance chart: {e}")
    
    def _generate_category_comparison(self, viz_dir: str):
        """Generate charts comparing performance by category."""
        try:
            # Accuracy by category
            categories = list(self.results["metrics"]["accuracy"]["by_category"].keys())
            scores = [self.results["metrics"]["accuracy"]["by_category"][cat] for cat in categories]
            
            plt.figure(figsize=(12, 6))
            plt.bar(categories, scores)
            plt.xlabel('Question Categories')
            plt.ylabel('Accuracy Score')
            plt.title('Accuracy by Question Category')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save the chart
            plt.savefig(os.path.join(viz_dir, 'accuracy_by_category.png'))
            plt.close()
        except Exception as e:
            print(f"Error generating category comparison chart: {e}")
    
    def _save_results(self):
        """Save the results to disk."""
        # Save full results as JSON
        with open(os.path.join(self.output_dir, "results.json"), 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Generate summary CSV
        self._generate_summary_csv()
        
        # Generate detailed CSV for analysis
        self._generate_detailed_csv()
    
    def _generate_summary_csv(self):
        """Generate a summary CSV file with key metrics."""
        summary_data = []
        
        # Add consensus metrics
        consensus_row = {
            'System': 'Consensus',
            'Accuracy': self.results["metrics"]["accuracy"]["overall"].get("consensus", 0),
            'Reliability': self.results["metrics"]["reliability"]["overall"].get("consensus", 0),
            'Avg Response Time (s)': self.results["metrics"]["performance"]["overall"].get("consensus", 0)
        }
        summary_data.append(consensus_row)
        
        # Add individual model metrics
        for model_id in self.results["single_model"].keys():
            model_row = {
                'System': model_id,
                'Accuracy': self.results["metrics"]["accuracy"]["by_model"].get(model_id, 0),
                'Reliability': self.results["metrics"]["reliability"]["by_model"].get(model_id, 0),
                'Avg Response Time (s)': self.results["metrics"]["performance"]["by_model"].get(model_id, 0)
            }
            summary_data.append(model_row)
        
        # Create and save the DataFrame
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(self.output_dir, "summary.csv"), index=False)
    
    def _generate_detailed_csv(self):
        """Generate a detailed CSV file with per-test-case metrics."""
        detailed_data = []
        
        # Add consensus results
        for case_id, case_results in self.results["consensus"].items():
            # Get test case details
            test_case = next((tc for tc in self.test_cases["cases"] if tc["id"] == case_id), None)
            if not test_case:
                continue
            
            if not case_results["evaluation"]:
                continue
                
            # Average metrics across runs
            avg_correctness = sum(eval_res["factual_correctness"] for eval_res in case_results["evaluation"]) / len(case_results["evaluation"])
            avg_time = sum(case_results["timing"]) / len(case_results["timing"])
            
            # Calculate consistency across runs
            consistency = 0
            if len(case_results["responses"]) > 1:
                similarities = []
                for i in range(len(case_results["responses"])):
                    for j in range(i+1, len(case_results["responses"])):
                        sim = self._compute_semantic_similarity(case_results["responses"][i], case_results["responses"][j])
                        similarities.append(sim)
                if similarities:
                    consistency = sum(similarities) / len(similarities)
            
            # Add row to data
            row = {
                'System': 'Consensus',
                'Test Case ID': case_id,
                'Category': case_results["category"],
                'Prompt': test_case["prompt"],
                'Accuracy': avg_correctness,
                'Consistency': consistency,
                'Response Time (s)': avg_time,
                'First Response': case_results["responses"][0] if case_results["responses"] else ""
            }
            detailed_data.append(row)
        
        # Add individual model results
        for model_id, model_results in self.results["single_model"].items():
            for case_id, case_results in model_results.items():
                # Get test case details
                test_case = next((tc for tc in self.test_cases["cases"] if tc["id"] == case_id), None)
                if not test_case:
                    continue
                
                if not case_results["evaluation"]:
                    continue
                    
                # Average metrics across runs
                avg_correctness = sum(eval_res["factual_correctness"] for eval_res in case_results["evaluation"]) / len(case_results["evaluation"])
                avg_time = sum(case_results["timing"]) / len(case_results["timing"])
                
                # Calculate consistency across runs
                consistency = 0
                if len(case_results["responses"]) > 1:
                    similarities = []
                    for i in range(len(case_results["responses"])):
                        for j in range(i+1, len(case_results["responses"])):
                            sim = self._compute_semantic_similarity(case_results["responses"][i], case_results["responses"][j])
                            similarities.append(sim)
                    if similarities:
                        consistency = sum(similarities) / len(similarities)
                
                # Add row to data
                row = {
                    'System': model_id,
                    'Test Case ID': case_id,
                    'Category': case_results["category"],
                    'Prompt': test_case["prompt"],
                    'Accuracy': avg_correctness,
                    'Consistency': consistency,
                    'Response Time (s)': avg_time,
                    'First Response': case_results["responses"][0] if case_results["responses"] else ""
                }
                detailed_data.append(row)
        
        # Create and save the DataFrame
        detailed_df = pd.DataFrame(detailed_data)
        detailed_df.to_csv(os.path.join(self.output_dir, "detailed_results.csv"), index=False)

def main():
    """Main function to run the consensus testing framework."""
    parser = argparse.ArgumentParser(description='Test and benchmark the consensus system.')
    parser.add_argument('--config', type=str, default='src/flare_ai_consensus/input.json',
                        help='Path to the consensus configuration file')
    parser.add_argument('--test-cases', type=str, default='src/flare_ai_consensus/test_cases.json',
                        help='Path to test cases JSON file')
    parser.add_argument('--output-dir', type=str, default='test_results',
                        help='Directory to save test results')
    parser.add_argument('--num-runs', type=int, default=3,
                        help='Number of times to run each test for reliability assessment')
    parser.add_argument('--parallel', action='store_true',
                        help='Run tests in parallel for better performance')
    parser.add_argument('--create-sample-test-cases', type=str, 
                        help='Create sample test cases file at the specified path and exit')
    
    args = parser.parse_args()
    
    # Create sample test cases if requested
    if args.create_sample_test_cases:
        create_sample_test_cases(args.create_sample_test_cases)
        print(f"Created sample test cases at {args.create_sample_test_cases}")
        return
    
    # Create and run the test framework
    framework = ConsensusTestFramework(
        config_path=args.config,
        test_cases_path=args.test_cases,
        output_dir=args.output_dir,
        num_runs=args.num_runs,
        parallel_execution=args.parallel
    )
    
    results = framework.run_tests()
    print("Testing complete. Results saved to", args.output_dir)

def create_sample_test_cases(filepath: str):
    """Create sample test cases for basic testing."""
    sample_cases = {
        "description": "Sample test cases for consensus system evaluation",
        "cases": [
            {
                "id": "factual_knowledge_1",
                "prompt": "What is the capital of France?",
                "ground_truth": "The capital of France is Paris.",
                "key_facts": ["Paris", "capital", "France"]
            },
            {
                "id": "technical_1",
                "prompt": "Explain how HTTP cookies work.",
                "ground_truth": "HTTP cookies are small pieces of data stored on the client's device by the web browser while browsing a website. They are designed to be a reliable mechanism for websites to remember stateful information or to record the user's browsing activity.",
                "key_facts": ["stored on client device", "browser", "remember state", "user activity"]
            },
            {
                "id": "reasoning_1",
                "prompt": "If a train travels at 60 mph, how long will it take to travel 120 miles?",
                "ground_truth": "It will take 2 hours to travel 120 miles at 60 mph.",
                "key_facts": ["2 hours", "120 miles", "60 mph"]
            }
        ]
    }
    
    with open(filepath, 'w') as f:
        json.dump(sample_cases, f, indent=2)

if __name__ == "__main__":
    main() 
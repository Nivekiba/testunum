#!/usr/bin/env python3
"""
Simple CloudWatch Execution Time Reader for TestUnum Functions

This script reads CloudWatch logs from AWS to extract execution times
for function1 to function32 based on their ARNs from function-arn.yaml.

Requirements:
- boto3
- pyyaml

Usage:
    python3 simple_cloudwatch_reader.py [--start-time YYYY-MM-DD] [--end-time YYYY-MM-DD] [--region ap-southeast-2]
"""

import boto3
import yaml
import json
import argparse
import datetime
from datetime import datetime, timedelta
import re
from typing import Dict, List, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleCloudWatchReader:
    """Simple class to read CloudWatch logs and extract execution times for Lambda functions."""
    
    def __init__(self, region: str = 'ap-southeast-2'):
        """Initialize the CloudWatch client."""
        self.region = region
        self.cloudwatch_logs = boto3.client('logs', region_name=region)
        
    def load_function_arns(self, yaml_file: str = 'function-arn.yaml') -> Dict[str, str]:
        """Load function ARNs from YAML file."""
        try:
            with open(yaml_file, 'r') as file:
                function_arns = yaml.safe_load(file)
            logger.info(f"Loaded {len(function_arns)} function ARNs from {yaml_file}")
            return function_arns
        except FileNotFoundError:
            logger.error(f"File {yaml_file} not found")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file: {e}")
            raise
    
    def get_log_group_name(self, function_arn: str) -> str:
        """Extract log group name from Lambda function ARN."""
        # Extract function name from ARN
        function_name = function_arn.split(':')[-1]
        return f"/aws/lambda/{function_name}"
    
    def parse_execution_time(self, log_message: str) -> Optional[float]:
        """Parse execution time from CloudWatch log message."""
        # Look for duration in milliseconds
        duration_pattern = r'Duration: (\d+\.?\d*) ms'
        match = re.search(duration_pattern, log_message)
        if match:
            return float(match.group(1))
        return None
    
    def get_function_logs(self, function_arn: str, start_time: datetime, end_time: datetime) -> List[Dict]:
        """Get CloudWatch logs for a specific function within the time range."""
        log_group_name = self.get_log_group_name(function_arn)
        
        try:
            # Query CloudWatch logs
            response = self.cloudwatch_logs.filter_log_events(
                logGroupName=log_group_name,
                startTime=int(start_time.timestamp() * 1000),
                endTime=int(end_time.timestamp() * 1000),
                filterPattern='REPORT'
            )
            
            logs = []
            for event in response['events']:
                execution_time = self.parse_execution_time(event['message'])
                if execution_time:
                    logs.append({
                        'timestamp': datetime.fromtimestamp(event['timestamp'] / 1000),
                        'execution_time_ms': execution_time,
                        'message': event['message']
                    })
            
            return logs
            
        except self.cloudwatch_logs.exceptions.ResourceNotFoundException:
            logger.warning(f"Log group {log_group_name} not found for function {function_arn}")
            return []
        except Exception as e:
            logger.error(f"Error getting logs for {function_arn}: {e}")
            return []
    
    def get_all_function_execution_times(self, function_arns: Dict[str, str], 
                                       start_time: datetime, end_time: datetime) -> Dict[str, List[Dict]]:
        """Get execution times for all functions."""
        all_execution_times = {}
        
        logger.info(f"Fetching execution times for {len(function_arns)} functions...")
        
        for func_name, func_arn in function_arns.items():
            logger.info(f"Processing {func_name}...")
            logs = self.get_function_logs(func_arn, start_time, end_time)
            all_execution_times[func_name] = logs
            
            if logs:
                logger.info(f"  Found {len(logs)} execution records for {func_name}")
            else:
                logger.warning(f"  No execution records found for {func_name}")
        
        return all_execution_times
    
    def analyze_execution_times(self, execution_times: Dict[str, List[Dict]]) -> List[Dict]:
        """Analyze execution times and create a summary."""
        analysis_data = []
        
        for func_name, logs in execution_times.items():
            if not logs:
                analysis_data.append({
                    'Function': func_name,
                    'Total_Executions': 0,
                    'Avg_Execution_Time_ms': 0,
                    'Min_Execution_Time_ms': 0,
                    'Max_Execution_Time_ms': 0,
                    'Last_Execution': None
                })
                continue
            
            execution_times_ms = [log['execution_time_ms'] for log in logs]
            
            analysis_data.append({
                'Function': func_name,
                'Total_Executions': len(logs),
                'Avg_Execution_Time_ms': round(sum(execution_times_ms) / len(execution_times_ms), 2),
                'Min_Execution_Time_ms': min(execution_times_ms),
                'Max_Execution_Time_ms': max(execution_times_ms),
                'Last_Execution': max(log['timestamp'] for log in logs)
            })
        
        # Sort by function number
        return sorted(analysis_data, key=lambda x: int(x['Function'].replace('Func', '')))
    
    def save_results_to_json(self, execution_times: Dict[str, List[Dict]], 
                           analysis_data: List[Dict], filename: str = 'execution_times_analysis.json'):
        """Save results to JSON file."""
        # Convert datetime objects to strings for JSON serialization
        for analysis in analysis_data:
            if analysis['Last_Execution']:
                analysis['Last_Execution'] = analysis['Last_Execution'].isoformat()
        
        for func_name, logs in execution_times.items():
            for log in logs:
                log['timestamp'] = log['timestamp'].isoformat()
        
        results = {
            'analysis': analysis_data,
            'detailed_logs': execution_times
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {filename}")
    
    def print_summary(self, analysis_data: List[Dict]):
        """Print a formatted summary of the analysis."""
        print("\n" + "="*100)
        print("EXECUTION TIME ANALYSIS SUMMARY")
        print("="*100)
        print(f"{'Function':<10} {'Executions':<12} {'Avg (ms)':<10} {'Min (ms)':<10} {'Max (ms)':<10} {'Last Execution':<20}")
        print("-" * 100)
        
        for data in analysis_data:
            last_exec = data['Last_Execution'].strftime('%Y-%m-%d %H:%M:%S') if data['Last_Execution'] else 'N/A'
            print(f"{data['Function']:<10} {data['Total_Executions']:<12} {data['Avg_Execution_Time_ms']:<10} "
                  f"{data['Min_Execution_Time_ms']:<10} {data['Max_Execution_Time_ms']:<10} {last_exec:<20}")
        
        # Calculate summary statistics
        total_executions = sum(data['Total_Executions'] for data in analysis_data)
        functions_with_executions = len([data for data in analysis_data if data['Total_Executions'] > 0])
        avg_execution_time = sum(data['Avg_Execution_Time_ms'] for data in analysis_data if data['Total_Executions'] > 0) / max(functions_with_executions, 1)
        
        print("\n" + "="*100)
        print("SUMMARY STATISTICS")
        print("="*100)
        print(f"Total executions across all functions: {total_executions}")
        print(f"Average execution time: {avg_execution_time:.2f} ms")
        print(f"Functions with executions: {functions_with_executions}")
        print(f"Functions without executions: {len(analysis_data) - functions_with_executions}")

def main():
    """Main function to run the CloudWatch execution time analysis."""
    parser = argparse.ArgumentParser(description='Read CloudWatch logs for Lambda function execution times')
    parser.add_argument('--start-time', type=str, default=None,
                       help='Start time in YYYY-MM-DD format (default: 7 days ago)')
    parser.add_argument('--end-time', type=str, default=None,
                       help='End time in YYYY-MM-DD format (default: now)')
    parser.add_argument('--region', type=str, default='ap-southeast-2',
                       help='AWS region (default: ap-southeast-2)')
    parser.add_argument('--yaml-file', type=str, default='function-arn.yaml',
                       help='YAML file containing function ARNs (default: function-arn.yaml)')
    parser.add_argument('--output-json', type=str, default='execution_times_analysis.json',
                       help='Output JSON file name (default: execution_times_analysis.json)')
    
    args = parser.parse_args()
    
    # Set default time range if not provided
    if args.end_time:
        end_time = datetime.strptime(args.end_time, '%Y-%m-%d')
    else:
        end_time = datetime.now()
    
    if args.start_time:
        start_time = datetime.strptime(args.start_time, '%Y-%m-%d')
    else:
        start_time = end_time - timedelta(days=7)
    
    logger.info(f"Analyzing execution times from {start_time} to {end_time}")
    
    try:
        # Initialize the reader
        reader = SimpleCloudWatchReader(region=args.region)
        
        # Load function ARNs
        function_arns = reader.load_function_arns(args.yaml_file)
        
        # Get execution times for all functions
        execution_times = reader.get_all_function_execution_times(function_arns, start_time, end_time)
        
        # Analyze the data
        analysis_data = reader.analyze_execution_times(execution_times)
        
        # Print summary
        reader.print_summary(analysis_data)
        
        # Save results to JSON
        reader.save_results_to_json(execution_times, analysis_data, args.output_json)
        
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        raise

if __name__ == "__main__":
    main()

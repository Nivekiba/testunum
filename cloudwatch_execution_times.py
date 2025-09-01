#!/usr/bin/env python3
"""
CloudWatch Execution Time Reader for TestUnum Functions

This script reads CloudWatch logs from AWS to extract execution times
for function1 to function32 based on their ARNs from function-arn.yaml.

Requirements:
- boto3
- pyyaml
- pandas (for data analysis)
- matplotlib (for visualization)

Usage:
    python3 cloudwatch_execution_times.py [--start-time YYYY-MM-DD] [--end-time YYYY-MM-DD] [--region ap-southeast-2]
"""

import boto3
import yaml
import json
import argparse
import datetime
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import re
from typing import Dict, List, Optional, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CloudWatchExecutionTimeReader:
    """Class to read CloudWatch logs and extract execution times for Lambda functions."""
    
    def __init__(self, region: str = 'ap-southeast-2'):
        """Initialize the CloudWatch client."""
        self.region = region
        self.cloudwatch_logs = boto3.client('logs', region_name=region)
        self.lambda_client = boto3.client('lambda', region_name=region)
        
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
        
        # Look for "REPORT" line which contains duration
        if "REPORT" in log_message:
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
    
    def analyze_execution_times(self, execution_times: Dict[str, List[Dict]]) -> pd.DataFrame:
        """Analyze execution times and create a summary DataFrame."""
        analysis_data = []

        times_list = []
        
        for func_name, logs in execution_times.items():
            if not logs:
                analysis_data.append({
                    'Function': func_name,
                    'Total_Executions': 0,
                    'Avg_Execution_Time_ms': 0,
                    'avg_without_first_ms': 0,
                    'Min_Execution_Time_ms': 0,
                    'Max_Execution_Time_ms': 0,
                    'Std_Dev_ms': 0,
                    'Last_Execution': None
                })
                continue
            
            execution_times_ms = [log['execution_time_ms'] for log in logs]
            
            analysis_data.append({
                'Function': func_name,
                'Total_Executions': len(logs),
                'Avg_Execution_Time_ms': round(sum(execution_times_ms) / len(execution_times_ms), 2),
                'avg_without_first_ms': round((sum(execution_times_ms)-max(execution_times_ms)) / len(execution_times_ms[1:]), 2),
                'Min_Execution_Time_ms': min(execution_times_ms),
                'Max_Execution_Time_ms': max(execution_times_ms),
                'Std_Dev_ms': round(pd.Series(execution_times_ms).std(), 2),
                'Last_Execution': max(log['timestamp'] for log in logs)
            })
        
        df = pd.DataFrame(analysis_data)
        return df.sort_values('Function', key=lambda x: x.str.extract(r'(\d+)', expand=False).astype(int))
    
    def save_results_to_excel(self, execution_times: Dict[str, List[Dict]], 
                            analysis_df: pd.DataFrame, filename: str = 'execution_times_analysis.xlsx'):
        """Save results to Excel file."""
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Save summary analysis
            analysis_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Save detailed logs for each function
            for func_name, logs in execution_times.items():
                if logs:
                    df_logs = pd.DataFrame(logs)
                    df_logs.to_excel(writer, sheet_name=func_name, index=False)
        
        logger.info(f"Results saved to {filename}")
    
    def create_execution_time_chart(self, analysis_df: pd.DataFrame, filename: str = 'execution_times_chart.png'):
        """Create a chart showing execution times."""
        plt.figure(figsize=(15, 8))
        
        # Create bar chart for average execution times
        plt.subplot(2, 2, 1)
        plt.bar(range(len(analysis_df)), analysis_df['Avg_Execution_Time_ms'])
        plt.title('Average Execution Times')
        plt.xlabel('Function Index')
        plt.ylabel('Execution Time (ms)')
        plt.xticks(range(len(analysis_df)), [f"Func{i+1}" for i in range(len(analysis_df))], rotation=45)
        
        # Create line chart for execution count
        plt.subplot(2, 2, 2)
        plt.plot(range(len(analysis_df)), analysis_df['Total_Executions'], marker='o')
        plt.title('Total Executions')
        plt.xlabel('Function Index')
        plt.ylabel('Number of Executions')
        plt.xticks(range(len(analysis_df)), [f"Func{i+1}" for i in range(len(analysis_df))], rotation=45)
        
        # Create box plot data
        plt.subplot(2, 2, 3)
        execution_data = []
        labels = []
        for _, row in analysis_df.iterrows():
            if row['Total_Executions'] > 0:
                # We don't have individual execution times here, so we'll show min/max/avg
                execution_data.append([row['Min_Execution_Time_ms'], row['Avg_Execution_Time_ms'], row['Max_Execution_Time_ms']])
                labels.append(row['Function'])
        
        if execution_data:
            plt.boxplot(execution_data, labels=labels)
            plt.title('Execution Time Distribution')
            plt.xlabel('Function')
            plt.ylabel('Execution Time (ms)')
            plt.xticks(rotation=45)
        
        # Create heatmap of execution times
        plt.subplot(2, 2, 4)
        heatmap_data = analysis_df[['Avg_Execution_Time_ms', 'Min_Execution_Time_ms', 'Max_Execution_Time_ms']].values
        plt.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
        plt.colorbar(label='Execution Time (ms)')
        plt.title('Execution Time Heatmap')
        plt.xlabel('Metric (Avg/Min/Max)')
        plt.ylabel('Function Index')
        plt.xticks([0, 1, 2], ['Avg', 'Min', 'Max'])
        plt.yticks(range(len(analysis_df)), [f"Func{i+1}" for i in range(len(analysis_df))])
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"Chart saved to {filename}")
        plt.show()

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
    parser.add_argument('--output-excel', type=str, default='execution_times_analysis.xlsx',
                       help='Output Excel file name (default: execution_times_analysis.xlsx)')
    parser.add_argument('--output-chart', type=str, default='execution_times_chart.png',
                       help='Output chart file name (default: execution_times_chart.png)')
    
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
        reader = CloudWatchExecutionTimeReader(region=args.region)
        
        # Load function ARNs
        function_arns = reader.load_function_arns(args.yaml_file)
        
        # Get execution times for all functions
        execution_times = reader.get_all_function_execution_times(function_arns, start_time, end_time)
        
        # Analyze the data
        analysis_df = reader.analyze_execution_times(execution_times)
        
        # Display summary
        print("\n" + "="*80)
        print("EXECUTION TIME ANALYSIS SUMMARY")
        print("="*80)
        print(analysis_df.to_string(index=False))
        
        # Save results to Excel
        reader.save_results_to_excel(execution_times, analysis_df, args.output_excel)
        
        # Create and save chart
        reader.create_execution_time_chart(analysis_df, args.output_chart)
        
        # Print summary statistics
        total_executions = analysis_df['Total_Executions'].sum()
        avg_execution_time = analysis_df[analysis_df['Total_Executions'] > 0]['Avg_Execution_Time_ms'].mean()
        
        print(f"\nSUMMARY STATISTICS:")
        print(f"Total executions across all functions: {total_executions}")
        print(f"Average execution time: {avg_execution_time:.2f} ms")
        print(f"Functions with executions: {len(analysis_df[analysis_df['Total_Executions'] > 0])}")
        print(f"Functions without executions: {len(analysis_df[analysis_df['Total_Executions'] == 0])}")
        
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        raise

if __name__ == "__main__":
    main()

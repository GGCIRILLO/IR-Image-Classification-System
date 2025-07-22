#!/usr/bin/env python3
"""
IR Image Classification System - Mission Runner

This script provides a command-line interface to run the complete IR image
classification pipeline end-to-end. It allows specification of all key
parameters including model selection, query configuration, and output options.

Usage:
    python scripts/run_mission.py --image path/to/query.png --database data/vector_db
    python scripts/run_mission.py --image query.png --model models/ir_model.pth --strategy military_priority
    python scripts/run_mission.py --help

"""

import sys
import os
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from PIL import Image

from src.query import (
    QueryProcessor,
    QueryProcessorConfig,
    MilitaryQueryConfig,
    DevelopmentQueryConfig,
    ValidationMode,
    CachePolicy,
    RankingStrategy,
    ConfidenceStrategy
)
from src.models.data_models import SimilarityResult, QueryResult
from src.models.object_classes import (
    ObjectClass, 
    ObjectCategory, 
    OBJECT_REGISTRY,
    get_object_classes
)


class MissionRunner:
    """
    Command-line interface for running IR image classification missions.
    
    Provides comprehensive configuration options for military and civilian
    deployment scenarios with detailed result reporting.
    """
    
    def __init__(self):
        self.processor: Optional[QueryProcessor] = None
        self.mission_id: str = f"mission_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    def setup_argument_parser(self) -> argparse.ArgumentParser:
        """Setup command-line argument parser with comprehensive options."""
        parser = argparse.ArgumentParser(
            description="IR Image Classification System - Mission Runner",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Basic query
  python scripts/run_mission.py --image query.png --database data/vector_db
  
  # Military deployment with custom model
  python scripts/run_mission.py --image target.png --database data/vector_db \\
    --model models/military_v2.pth --strategy military_priority \\
    --confidence-strategy military_calibrated --max-results 10
  
  # Development testing with debug output
  python scripts/run_mission.py --image test.png --database data/vector_db \\
    --preset development --debug --output results.json
  
  # High-precision reconnaissance
  python scripts/run_mission.py --image recon.png --database data/vector_db \\
    --confidence-threshold 0.9 --similarity-threshold 0.8 \\
    --validation-mode strict --disable-cache
            """)
        
        # Required arguments
        required = parser.add_argument_group('required arguments')
        required.add_argument(
            '--image', '-i',
            type=str,
            required=True,
            help='Path to the query image file (PNG, JPEG, TIFF, BMP)'
        )
        required.add_argument(
            '--database', '-d',
            type=str,
            required=True,
            help='Path to the vector database directory'
        )
        
        # Model configuration
        model_group = parser.add_argument_group('model configuration')
        model_group.add_argument(
            '--model', '-m',
            type=str,
            help='Path to the fine-tuned model weights (.pth file)'
        )
        model_group.add_argument(
            '--collection',
            type=str,
            default='ir_embeddings',
            help='Vector database collection name (default: ir_embeddings)'
        )
        
        # Query configuration presets
        preset_group = parser.add_argument_group('configuration presets')
        preset_group.add_argument(
            '--preset', '-p',
            choices=['military', 'development', 'production', 'testing'],
            help='Use predefined configuration preset'
        )
        
        # Ranking and confidence strategies
        strategy_group = parser.add_argument_group('ranking and confidence strategies')
        strategy_group.add_argument(
            '--strategy',
            choices=['similarity_only', 'confidence_weighted', 'hybrid_score', 'military_priority'],
            default='hybrid_score',
            help='Ranking strategy (default: hybrid_score)'
        )
        strategy_group.add_argument(
            '--confidence-strategy',
            choices=['similarity_based', 'statistical', 'ensemble', 'military_calibrated'],
            default='ensemble',
            help='Confidence calculation strategy (default: ensemble)'
        )
        
        # Threshold parameters
        threshold_group = parser.add_argument_group('threshold parameters')
        threshold_group.add_argument(
            '--confidence-threshold',
            type=float,
            default=0.7,
            help='Minimum confidence threshold (0.0-1.0, default: 0.7)'
        )
        threshold_group.add_argument(
            '--similarity-threshold',
            type=float,
            default=0.5,
            help='Minimum similarity threshold (0.0-1.0, default: 0.5)'
        )
        threshold_group.add_argument(
            '--max-results',
            type=int,
            default=5,
            help='Maximum number of results to return (default: 5)'
        )
        
        # Processing options
        processing_group = parser.add_argument_group('processing options')
        processing_group.add_argument(
            '--max-query-time',
            type=float,
            default=2.0,
            help='Maximum query processing time in seconds (default: 2.0)'
        )
        processing_group.add_argument(
            '--validation-mode',
            choices=['strict', 'relaxed', 'disabled'],
            default='relaxed',
            help='Image validation mode (default: relaxed)'
        )
        processing_group.add_argument(
            '--disable-gpu',
            action='store_true',
            help='Disable GPU acceleration (use CPU only)'
        )
        processing_group.add_argument(
            '--disable-cache',
            action='store_true',
            help='Disable query result caching'
        )
        processing_group.add_argument(
            '--enable-diversity',
            action='store_true',
            help='Enable diversity filtering to remove similar results'
        )
        
        # Output options
        output_group = parser.add_argument_group('output options')
        output_group.add_argument(
            '--output', '-o',
            type=str,
            help='Output file path for results (JSON format)'
        )
        output_group.add_argument(
            '--format',
            choices=['table', 'json', 'detailed', 'military'],
            default='table',
            help='Output format (default: table)'
        )
        output_group.add_argument(
            '--quiet', '-q',
            action='store_true',
            help='Suppress detailed output, show only results'
        )
        output_group.add_argument(
            '--debug',
            action='store_true',
            help='Enable debug output and logging'
        )
        output_group.add_argument(
            '--save-metadata',
            action='store_true',
            help='Include detailed metadata in output'
        )
        
        # Mission parameters
        mission_group = parser.add_argument_group('mission parameters')
        mission_group.add_argument(
            '--mission-id',
            type=str,
            help='Custom mission identifier'
        )
        mission_group.add_argument(
            '--operator',
            type=str,
            help='Operator/analyst name'
        )
        mission_group.add_argument(
            '--classification',
            choices=['UNCLASSIFIED', 'RESTRICTED', 'CONFIDENTIAL', 'SECRET', 'TOP_SECRET'],
            default='UNCLASSIFIED',
            help='Mission classification level (default: UNCLASSIFIED)'
        )
        
        return parser
    
    def create_configuration(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Create QueryProcessor configuration from command-line arguments."""
        if args.preset:
            # Use preset configuration as base
            if args.preset == 'military':
                config = MilitaryQueryConfig()
            elif args.preset == 'development':
                config = DevelopmentQueryConfig()
            else:
                config = QueryProcessorConfig()
            
            # Override with command-line arguments
            config.max_query_time = args.max_query_time
            config.min_confidence_threshold = args.confidence_threshold
            config.top_k_results = args.max_results
            config.similarity_threshold = args.similarity_threshold
            config.enable_gpu_acceleration = not args.disable_gpu
            config.cache_policy = CachePolicy.DISABLED if args.disable_cache else CachePolicy.ENABLED
            config.validation_mode = ValidationMode(args.validation_mode)  # Use lowercase value
            config.debug_mode = args.debug
            
        else:
            # Create custom configuration
            config = QueryProcessorConfig(
                max_query_time=args.max_query_time,
                min_confidence_threshold=args.confidence_threshold,
                top_k_results=args.max_results,
                similarity_threshold=args.similarity_threshold,
                validation_mode=ValidationMode(args.validation_mode),  # Use lowercase value
                enable_gpu_acceleration=not args.disable_gpu,
                cache_policy=CachePolicy.DISABLED if args.disable_cache else CachePolicy.ENABLED,
                debug_mode=args.debug,
                enable_result_reranking=True
            )
        
        # Add strategy configurations
        config_dict = config.to_dict()
        config_dict.update({
            'ranking_strategy': args.strategy,
            'confidence_strategy': args.confidence_strategy,
            'enable_diversity_filtering': args.enable_diversity
        })
        
        return config_dict
    
    def validate_inputs(self, args: argparse.Namespace) -> None:
        """Validate input arguments and file paths."""
        # Validate image file
        image_path = Path(args.image)
        if not image_path.exists():
            raise FileNotFoundError(f"Query image not found: {args.image}")
        
        if image_path.suffix.lower() not in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']:
            raise ValueError(f"Unsupported image format: {image_path.suffix}")
        
        # Validate database path
        db_path = Path(args.database)
        if not db_path.exists():
            raise FileNotFoundError(f"Database directory not found: {args.database}")
        
        # Validate model path if provided
        if args.model:
            model_path = Path(args.model)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {args.model}")
        
        # Validate threshold ranges
        if not 0.0 <= args.confidence_threshold <= 1.0:
            raise ValueError(f"Confidence threshold must be between 0.0 and 1.0: {args.confidence_threshold}")
        
        if not 0.0 <= args.similarity_threshold <= 1.0:
            raise ValueError(f"Similarity threshold must be between 0.0 and 1.0: {args.similarity_threshold}")
        
        if args.max_results <= 0:
            raise ValueError(f"Max results must be positive: {args.max_results}")
        
        if args.max_query_time <= 0:
            raise ValueError(f"Max query time must be positive: {args.max_query_time}")
    
    def initialize_processor(self, args: argparse.Namespace) -> None:
        """Initialize the QueryProcessor with configuration."""
        config = self.create_configuration(args)
        
        if not args.quiet:
            print(f"🚀 Initializing IR Classification System...")
            print(f"   Mission ID: {self.mission_id}")
            print(f"   Database: {args.database}")
            print(f"   Model: {args.model or 'Default'}")
            print(f"   Collection: {args.collection}")
        
        self.processor = QueryProcessor(
            database_path=args.database,
            model_path=args.model,
            collection_name=args.collection,
            config=config
        )
        
        success = self.processor.initialize()
        if not success:
            raise RuntimeError("Failed to initialize QueryProcessor")
        
        if not args.quiet:
            print(f"✅ System initialized successfully")
    
    def execute_query(self, args: argparse.Namespace) -> QueryResult:
        """Execute the image classification query."""
        if not args.quiet:
            print(f"\n🔍 Executing query...")
            print(f"   Image: {args.image}")
            print(f"   Strategy: {args.strategy}")
            print(f"   Confidence Strategy: {args.confidence_strategy}")
            print(f"   Thresholds: confidence≥{args.confidence_threshold}, similarity≥{args.similarity_threshold}")
        
        start_time = time.time()
        
        # Execute query
        if not self.processor:
            raise RuntimeError("QueryProcessor not initialized")
            
        # Determine strict validation based on validation mode
        strict_validation = args.validation_mode == 'strict'
            
        result = self.processor.process_query(
            image_input=args.image,
            query_id=self.mission_id,
            options={
                'confidence_threshold': args.confidence_threshold,
                'similarity_threshold': args.similarity_threshold,
                'max_results': args.max_results,
                'enable_diversity': args.enable_diversity,
                'operator': args.operator,
                'classification': args.classification,
                'strict_validation': strict_validation
            }
        )


        total_time = time.time() - start_time
        
        if not args.quiet:
            print(f"✅ Query completed in {total_time:.3f}s")
            print(f"   Processing time: {result.processing_time:.3f}s")
            print(f"   Results found: {len(result.results)}")
        
        return result
    
    def format_output(self, result: QueryResult, args: argparse.Namespace) -> str:
        """Format query results according to specified format."""
        if args.format == 'json':
            return self._format_json(result, args)
        elif args.format == 'detailed':
            return self._format_detailed(result, args)
        elif args.format == 'military':
            return self._format_military(result, args)
        else:  # table
            return self._format_table(result, args)
    
    def _format_table(self, result: QueryResult, args: argparse.Namespace) -> str:
        """Format results as a table with enhanced object classification info."""
        lines = []
        lines.append("\n" + "="*80)
        lines.append("IR IMAGE CLASSIFICATION RESULTS")
        lines.append("="*80)
        lines.append(f"Mission ID: {result.query_id}")
        lines.append(f"Processing Time: {result.processing_time:.3f}s")
        lines.append(f"Results Found: {len(result.results)}")
        lines.append(f"Model Version: {result.model_version}")
        
        # Add object class statistics
        if result.results:
            categories = {}
            military_count = 0
            critical_count = 0
            
            for res in result.results:
                category = res.get_object_category()
                if category:
                    categories[category.value] = categories.get(category.value, 0) + 1
                if res.is_military_asset():
                    military_count += 1
                if res.is_critical_asset():
                    critical_count += 1
            
            lines.append(f"Military Assets: {military_count} | Critical Assets: {critical_count}")
            if categories:
                cat_summary = " | ".join([f"{cat}: {count}" for cat, count in categories.items()])
                lines.append(f"Categories: {cat_summary}")
        
        lines.append("")
        
        if result.results:
            lines.append(f"{'Rank':<4} {'Object Class':<25} {'Category':<15} {'Similarity':<10} {'Confidence':<10} {'Threat':<8}")
            lines.append("-" * 85)
            
            for i, res in enumerate(result.results, 1):
                category = res.get_object_category()
                category_str = category.value.replace('_', ' ').title() if category else 'Unknown'
                threat_level = res.get_threat_level()
                
                lines.append(
                    f"{i:<4} {res.object_class:<25} {category_str:<15} {res.similarity_score:<10.3f} "
                    f"{res.confidence:<10.3f} {threat_level:<8}"
                )
        else:
            lines.append("No results found matching the specified criteria.")
        
        return "\n".join(lines)
    
    def _format_detailed(self, result: QueryResult, args: argparse.Namespace) -> str:
        """Format results with detailed information."""
        lines = []
        lines.append("\n" + "="*80)
        lines.append("DETAILED IR IMAGE CLASSIFICATION RESULTS")
        lines.append("="*80)
        lines.append(f"Mission ID: {result.query_id}")
        lines.append(f"Timestamp: {result.timestamp}")
        lines.append(f"Processing Time: {result.processing_time:.3f}s")
        lines.append(f"Model Version: {result.model_version}")
        lines.append(f"Total Results: {len(result.results)}")
        
        if args.operator:
            lines.append(f"Operator: {args.operator}")
        if args.classification:
            lines.append(f"Classification: {args.classification}")
        
        lines.append("")
        
        for i, res in enumerate(result.results, 1):
            lines.append(f"Result #{i}")
            lines.append("-" * 40)
            lines.append(f"  Object Class: {res.object_class}")
            lines.append(f"  Image ID: {res.image_id}")
            lines.append(f"  Similarity Score: {res.similarity_score:.4f}")
            lines.append(f"  Confidence: {res.confidence:.4f}")
            lines.append(f"  Confidence Level: {res.metadata.get('confidence_level', 'Unknown')}")
            lines.append(f"  Rank: {res.metadata.get('rank', i)}")
            
            if args.save_metadata and res.metadata:
                lines.append("  Metadata:")
                for key, value in res.metadata.items():
                    if key not in ['confidence_level', 'rank']:
                        lines.append(f"    {key}: {value}")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def _format_military(self, result: QueryResult, args: argparse.Namespace) -> str:
        """Format results in military report style."""
        lines = []
        lines.append("\n" + "="*80)
        lines.append("MILITARY INTELLIGENCE CLASSIFICATION REPORT")
        lines.append("="*80)
        lines.append(f"CLASSIFICATION: {args.classification}")
        lines.append(f"MISSION ID: {result.query_id}")
        lines.append(f"DATE/TIME: {result.timestamp}")
        lines.append(f"OPERATOR: {args.operator or 'UNKNOWN'}")
        lines.append(f"PROCESSING TIME: {result.processing_time:.3f} SECONDS")
        lines.append(f"SYSTEM VERSION: {result.model_version}")
        lines.append("")
        lines.append("THREAT ASSESSMENT:")
        lines.append("-" * 50)
        
        critical_objects = []
        high_confidence = []
        
        for i, res in enumerate(result.results, 1):
            is_critical = any(critical in res.object_class.upper() for critical in 
                            ['TANK', 'MISSILE', 'AIRCRAFT', 'ARTILLERY', 'RADAR'])
            
            if is_critical:
                critical_objects.append((i, res))
            
            if res.confidence >= 0.8:
                high_confidence.append((i, res))
            
            threat_level = "HIGH" if is_critical and res.confidence >= 0.8 else \
                          "MEDIUM" if is_critical or res.confidence >= 0.7 else "LOW"
            
            lines.append(f"{i:2d}. {res.object_class:<25} | THREAT: {threat_level:<6} | "
                        f"CONF: {res.confidence:.3f} | SIM: {res.similarity_score:.3f}")
        
        lines.append("")
        lines.append("SUMMARY:")
        lines.append(f"  Total Targets Identified: {len(result.results)}")
        lines.append(f"  Critical Assets Detected: {len(critical_objects)}")
        lines.append(f"  High Confidence Matches: {len(high_confidence)}")
        
        if critical_objects:
            lines.append("")
            lines.append("CRITICAL ASSETS:")
            for rank, res in critical_objects:
                lines.append(f"  {rank}. {res.object_class} (Confidence: {res.confidence:.3f})")
        
        lines.append(f"\nEND OF REPORT - {args.classification}")
        
        return "\n".join(lines)
    
    def _format_json(self, result: QueryResult, args: argparse.Namespace) -> str:
        """Format results as JSON."""
        data = {
            'mission_id': result.query_id,
            'timestamp': result.timestamp.isoformat(),
            'processing_time': result.processing_time,
            'model_version': result.model_version,
            'query_image': args.image,
            'configuration': {
                'strategy': args.strategy,
                'confidence_strategy': args.confidence_strategy,
                'confidence_threshold': args.confidence_threshold,
                'similarity_threshold': args.similarity_threshold,
                'max_results': args.max_results
            },
            'results': []
        }
        
        if args.operator:
            data['operator'] = args.operator
        if args.classification:
            data['classification'] = args.classification
        
        for i, res in enumerate(result.results, 1):
            result_data = {
                'rank': i,
                'image_id': res.image_id,
                'object_class': res.object_class,
                'similarity_score': res.similarity_score,
                'confidence': res.confidence,
                'confidence_level': res.metadata.get('confidence_level', 'Unknown')
            }
            
            if args.save_metadata:
                result_data['metadata'] = res.metadata
            
            data['results'].append(result_data)
        
        return json.dumps(data, indent=2, default=str)
    
    def save_output(self, output: str, args: argparse.Namespace) -> None:
        """Save output to file if specified."""
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                f.write(output)
            
            if not args.quiet:
                print(f"\n💾 Results saved to: {args.output}")
    
    def run_mission(self, args: argparse.Namespace) -> None:
        """Execute the complete mission."""
        try:
            # Validate inputs
            self.validate_inputs(args)
            
            # Set mission ID if provided
            if args.mission_id:
                self.mission_id = args.mission_id
            
            # Initialize system
            self.initialize_processor(args)
            
            # Execute query
            result = self.execute_query(args)
            
            # Format and display results
            output = self.format_output(result, args)
            
            if not args.quiet or args.format != 'json':
                print(output)
            elif args.format == 'json' and args.quiet:
                print(output)  # Always print JSON even in quiet mode
            
            # Save to file if requested
            self.save_output(output, args)
            
            # Show performance validation if debug enabled
            if args.debug and self.processor:
                validation = self.processor.validate_system_performance()
                print("\n🔧 SYSTEM PERFORMANCE VALIDATION:")
                for requirement, passed in validation.items():
                    status = "✅" if passed else "❌"
                    print(f"   {status} {requirement}")
        
        except Exception as e:
            print(f"❌ Mission failed: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            sys.exit(1)


def main():
    """Main entry point for the mission runner."""
    runner = MissionRunner()
    parser = runner.setup_argument_parser()
    args = parser.parse_args()
    
    # Display banner unless in quiet mode
    if not args.quiet:
        print("🎯 IR IMAGE CLASSIFICATION SYSTEM - MISSION RUNNER")
        print("   Advanced AI-powered object identification for military intelligence")
        print("   Version: 1.0.0 | Date: July 17, 2025")
        print()
    
    runner.run_mission(args)


if __name__ == "__main__":
    main()

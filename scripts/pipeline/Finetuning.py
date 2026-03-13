"""
Fine-tuning script for multimodal vision-language models (CLIP).

Runs the complete fine-tuning pipeline using YAML configuration with detailed logging.
"""

import pandas as pd
import yaml
import logging
import sys
from pathlib import Path

from src.pipeline.FineTuningPipeline import create_finetuning_pipeline


REPO_ROOT = Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logging.info("="*80)
    logging.info("STARTING MULTIMODAL FINE-TUNING SCRIPT")
    logging.info("="*80)

    try:
        # File paths
        dataset_csv_path = REPO_ROOT / "dataset.csv"
        config_yaml_path = REPO_ROOT / "configs/CLIP.yaml"

        # Load dataset
        logging.info(f"\n{'='*80}")
        logging.info("LOADING DATASET")
        logging.info(f"{'='*80}")
        logging.info(f"Path: {dataset_csv_path}")

        dataset_df = pd.read_csv(dataset_csv_path)
        logging.info("Dataset loaded")

        # Load configuration
        logging.info(f"\n{'='*80}")
        logging.info("LOADING CONFIGURATION")
        logging.info(f"{'='*80}")

        logging.info("Loading configuration")
        with open(config_yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logging.info("Configuration loaded successfully")

        # Create fine-tuning pipeline
        logging.info(f"\n{'='*80}")
        logging.info("CREATING PIPELINE")
        logging.info(f"{'='*80}")

        pipeline = create_finetuning_pipeline(
            config=config,
            df=dataset_df,
            model_type='multimodal'
        )

        logging.info("Pipeline created successfully")

        pipeline.run_full_pipeline()

        logging.info(f"\n{'='*80}")
        logging.info("SCRIPT COMPLETED SUCCESSFULLY")
        logging.info(f"{'='*80}")

    except FileNotFoundError as e:
        logging.error(f"\n{'='*80}")
        logging.error("ERROR: FILE NOT FOUND")
        logging.error(f"{'='*80}")
        logging.error(f"Detail: {e}")
        logging.error("Verify that the files exist at the specified paths")
        sys.exit(1)

    except Exception as e:
        logging.error(f"\n{'='*80}")
        logging.error("ERROR DURING EXECUTION")
        logging.error(f"{'='*80}")
        logging.error(f"Type: {type(e).__name__}")
        logging.error(f"Message: {e}")
        logging.exception("Full stack trace:")
        sys.exit(1)

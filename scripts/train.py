#!/usr/bin/env python3
"""
Training script for pneumonia detection model
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from pneumonia_detector.config import Config
from pneumonia_detector.training import Trainer
from pneumonia_detector.utils import setup_logging, set_seed, setup_gpu


def main():
    parser = argparse.ArgumentParser(description="Train pneumonia detection model")
    parser.add_argument("data_path", help="Path to chest X-ray dataset")
    parser.add_argument("--config", default="configs/default.yaml", help="Configuration file")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--resume", help="Resume from checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.log_level)
    set_seed(args.seed)
    setup_gpu()
    
    # Load configuration
    config = Config.from_yaml(args.config)
    
    # Update paths
    config.models_dir = Path(args.output) / "models"
    config.logs_dir = Path(args.output) / "logs"
    config.outputs_dir = Path(args.output)
    
    # Create trainer and train
    trainer = Trainer(config)
    model = trainer.train(args.data_path, resume_from_checkpoint=args.resume)
    
    print("Training completed successfully!")
    print(f"Model saved to: {config.models_dir}")


if __name__ == "__main__":
    main()
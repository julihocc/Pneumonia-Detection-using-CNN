"""
Command Line Interface for pneumonia detection
"""
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .config import Config
from .training import Trainer
from .utils import setup_logging, set_seed, setup_gpu, get_system_info

app = typer.Typer(help="Pneumonia Detection CLI")
console = Console()


@app.command()
def train(
    data_path: str = typer.Argument(..., help="Path to the chest X-ray dataset"),
    config_path: str = typer.Option("configs/default.yaml", help="Path to configuration file"),
    output_dir: str = typer.Option("outputs", help="Output directory for models and logs"),
    resume_from: Optional[str] = typer.Option(None, help="Resume training from checkpoint"),
    log_level: str = typer.Option("INFO", help="Logging level"),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
):
    """Train a pneumonia detection model"""
    console.print("[bold blue]Starting Pneumonia Detection Training[/bold blue]")
    
    # Setup
    setup_logging(log_level)
    set_seed(seed)
    setup_gpu()
    
    # Display system info
    info = get_system_info()
    table = Table(title="System Information")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    
    for key, value in info.items():
        table.add_row(key, str(value))
    
    console.print(table)
    
    try:
        # Load configuration
        config = Config.from_yaml(config_path)
        
        # Update output paths
        config.models_dir = Path(output_dir) / "models"
        config.logs_dir = Path(output_dir) / "logs"
        config.outputs_dir = Path(output_dir)
        
        # Create trainer and train
        trainer = Trainer(config)
        model = trainer.train(data_path, resume_from_checkpoint=resume_from)
        
        console.print("[bold green]Training completed successfully![/bold green]")
        console.print(f"Model saved to: {config.models_dir}")
        
    except Exception as e:
        console.print(f"[bold red]Training failed: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def predict(
    image_path: str = typer.Argument(..., help="Path to chest X-ray image"),
    model_path: str = typer.Argument(..., help="Path to trained model"),
    config_path: str = typer.Option("configs/default.yaml", help="Path to configuration file"),
    output_path: Optional[str] = typer.Option(None, help="Path to save prediction results"),
):
    """Make prediction on a single chest X-ray image"""
    console.print("[bold blue]Making Pneumonia Prediction[/bold blue]")
    
    try:
        from .inference import PneumoniaPredictor
        
        # Load configuration and model
        config = Config.from_yaml(config_path)
        predictor = PneumoniaPredictor(model_path, config)
        
        # Make prediction
        result = predictor.predict_single(image_path)
        
        # Display results
        table = Table(title="Prediction Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Image", image_path)
        table.add_row("Prediction", result['prediction'])
        table.add_row("Confidence", f"{result['confidence']:.4f}")
        table.add_row("Probability (Pneumonia)", f"{result['probability']:.4f}")
        
        console.print(table)
        
        # Save results if requested
        if output_path:
            import json
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            console.print(f"Results saved to: {output_path}")
            
    except Exception as e:
        console.print(f"[bold red]Prediction failed: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def evaluate(
    data_path: str = typer.Argument(..., help="Path to test dataset"),
    model_path: str = typer.Argument(..., help="Path to trained model"),
    config_path: str = typer.Option("configs/default.yaml", help="Path to configuration file"),
    output_dir: str = typer.Option("outputs/evaluation", help="Output directory for evaluation results"),
):
    """Evaluate model on test dataset"""
    console.print("[bold blue]Evaluating Model[/bold blue]")
    
    try:
        from .training import ModelEvaluator
        from .data import DataPipeline
        import tensorflow as tf
        
        # Load configuration
        config = Config.from_yaml(config_path)
        
        # Load model
        model = tf.keras.models.load_model(model_path)
        
        # Prepare test data
        data_pipeline = DataPipeline(config.data)
        _, _, test_ds = data_pipeline.create_datasets(data_path)
        
        # Evaluate
        evaluator = ModelEvaluator(model, config)
        results = evaluator.evaluate_comprehensive(test_ds)
        
        # Display metrics
        table = Table(title="Evaluation Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        for metric, value in results['metrics'].items():
            table.add_row(metric.capitalize(), f"{value:.4f}")
            
        console.print(table)
        
        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save confusion matrix plot
        evaluator.plot_confusion_matrix(
            results['confusion_matrix'], 
            save_path=output_path / "confusion_matrix.png"
        )
        
        # Save classification report
        with open(output_path / "classification_report.txt", 'w') as f:
            f.write(results['classification_report'])
            
        console.print(f"Evaluation results saved to: {output_path}")
        
    except Exception as e:
        console.print(f"[bold red]Evaluation failed: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def info():
    """Display system and environment information"""
    info = get_system_info()
    
    table = Table(title="System Information")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    
    for key, value in info.items():
        table.add_row(key.replace('_', ' ').title(), str(value))
    
    console.print(table)


@app.command()
def config_template(
    output_path: str = typer.Option("config.yaml", help="Output path for config template")
):
    """Generate a configuration template"""
    config = Config()
    config.to_yaml(output_path)
    console.print(f"Configuration template saved to: {output_path}")


def train_cli():
    """Entry point for training CLI"""
    app()


def predict_cli():
    """Entry point for prediction CLI"""
    app()


if __name__ == "__main__":
    app()
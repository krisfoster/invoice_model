"""Configuration for invoice extraction model."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class ModelConfig:
    """Model configuration.

    Default settings optimized for M3 Max with 64GB RAM.
    For other hardware configurations:
    - M1/M2 (8GB): batch_size=8
    - M1/M2 Pro (16GB): batch_size=16
    - CPU only: batch_size=4
    """

    model_name: str = "distilbert-base-uncased"
    max_length: int = 512
    learning_rate: float = 3e-5  # Slightly higher for larger batches
    batch_size: int = 32  # Optimized for M3 Max with 64GB RAM
    num_epochs: int = 10
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1

    # Label definitions using BIO tagging
    # B- prefix for beginning of entity, I- for inside entity, O for outside
    labels: List[str] = None

    def __post_init__(self):
        if self.labels is None:
            self.labels = [
                "O",  # Outside any entity
                "B-TOTAL", "I-TOTAL",  # Total price
                "B-ITEM", "I-ITEM",  # Item names
                "B-ITEM_PRICE", "I-ITEM_PRICE",  # Item prices
                "B-CUSTOMER_NAME", "I-CUSTOMER_NAME",  # Customer name
                "B-CUSTOMER_ADDRESS", "I-CUSTOMER_ADDRESS",  # Customer address
                "B-DATE", "I-DATE",  # Invoice date
                "B-INVOICE_NUMBER", "I-INVOICE_NUMBER",  # Invoice number
            ]

    @property
    def num_labels(self) -> int:
        """Return number of labels."""
        return len(self.labels)

    @property
    def label2id(self) -> dict:
        """Return label to id mapping."""
        return {label: i for i, label in enumerate(self.labels)}

    @property
    def id2label(self) -> dict:
        """Return id to label mapping."""
        return {i: label for i, label in enumerate(self.labels)}


@dataclass
class DataConfig:
    """Data configuration."""

    # Default dataset: Company Documents Dataset from Kaggle
    kaggle_dataset: str = "ayoubcherguelaine/company-documents-dataset"

    data_dir: Path = Path("data")
    raw_data_dir: Path = Path("data/raw")
    invoices_dir: Path = Path("data/raw/invoices")
    processed_data_dir: Path = Path("data/processed")
    train_file: str = "train.json"
    val_file: str = "val.json"
    test_file: str = "test.json"

    def __post_init__(self):
        """Ensure paths are Path objects."""
        self.data_dir = Path(self.data_dir)
        self.raw_data_dir = Path(self.raw_data_dir)
        self.invoices_dir = Path(self.invoices_dir)
        self.processed_data_dir = Path(self.processed_data_dir)


@dataclass
class TrainingConfig:
    """Training configuration."""

    output_dir: Path = Path("models")
    checkpoint_dir: Path = Path("checkpoints")
    log_dir: Path = Path("logs")

    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100

    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "f1"

    seed: int = 42
    device: str = "cuda"  # Will fall back to CPU if CUDA unavailable
    fp16: bool = False  # Use mixed precision training if available

    def __post_init__(self):
        """Ensure paths are Path objects."""
        self.output_dir = Path(self.output_dir)
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.log_dir = Path(self.log_dir)


@dataclass
class JSONModelConfig:
    """Configuration for JSON generation model (Method 2: Constrained Decoding)."""

    # Model settings
    model_name: str = "google/flan-t5-base"  # Can use: t5-small, t5-base, flan-t5-base, flan-t5-large
    max_input_length: int = 512
    max_target_length: int = 512

    # Training settings
    learning_rate: float = 5e-5
    batch_size: int = 8  # T5 models require more memory than DistilBERT
    num_epochs: int = 10
    warmup_steps: int = 500
    weight_decay: float = 0.01

    # Generation settings
    num_beams: int = 4
    early_stopping: bool = True
    use_constrained_generation: bool = True  # Enable constrained JSON generation

    # Data paths
    json_data_dir: Path = Path("data/json")

    def __post_init__(self):
        """Ensure paths are Path objects."""
        self.json_data_dir = Path(self.json_data_dir)

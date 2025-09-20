import os
import pytest
from pathlib import Path

from datasets.stanford_dogs_extra import StanfordDogsExtraDataset

def test_dataset_init_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        StanfordDogsExtraDataset(tmp_path, split="val")

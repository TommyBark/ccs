import csv
import pickle
import torch

from dataclasses import dataclass
from hashlib import md5
from elk.training.preprocessing import load_hidden_states, normalize
from simple_parsing.helpers import field, Serializable
from typing import Literal, List
from pathlib import Path
from ..files import elk_cache_dir
from ..utils import select_usable_gpus


@dataclass
class EvaluateConfig(Serializable):
    source: str
    reporter_name: str
    targets: List[str]
    normalization: Literal["legacy", "elementwise", "meanonly"] = "meanonly"
    device: str = "cuda" 


def evaluate_reporters(cfg: EvaluateConfig):
    for target in cfg.targets:
        hiddens, labels = load_hidden_states(
            path=elk_cache_dir() / target / "validation_hiddens.pt"
        )
        assert len(set(labels)) > 1

        _, hiddens = normalize(hiddens, hiddens, cfg.normalization)

        reporter_root_path = (
            elk_cache_dir() / cfg.source / "reporters" / cfg.reporter_name
        )
        reporters = torch.load(
            reporter_root_path / "reporters.pt", map_location=cfg.device
        )

        transfer_eval = reporter_root_path / "transfer_eval"
        transfer_eval.mkdir(parents=True, exist_ok=True)

        L = hiddens.shape[1]
        layers = list(hiddens.unbind(1))
        layers.reverse()
        csv_file = transfer_eval / f"{target}.csv"

        for reporter in reporters:
            reporter.eval()

            with torch.no_grad(), open(csv_file, "w") as f:
                for layer_idx, hidden_state in enumerate(layers):

                    x0, x1 = hidden_state.to(cfg.device).float().chunk(2, dim=-1)
                    result = reporter.score(
                        (x0, x1),
                        labels.to(cfg.device),
                    )
                    stats = [*result]
                    stats += [cfg.normalization, cfg.source, target]

                    writer = csv.writer(f)
                    if not csv_file.exists():
                        # write column names only once
                        cols = [
                            "layer",
                            "acc",
                            "cal_acc",
                            "auroc",
                            "normalization",
                            "name",
                            "targets",
                        ]
                        writer.writerow(cols)
                    writer.writerow([L - layer_idx] + [stats])

        print("Eval file generated: ", csv_file)

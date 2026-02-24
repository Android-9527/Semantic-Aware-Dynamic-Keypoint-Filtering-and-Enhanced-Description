import json
import torch
from torch.utils.data import DataLoader
from model import SFD2Chap03Model
from dataset import CityscapesBinaryReliabilityDataset
from metrics import binary_metrics


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    cfg = load_config("config.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CityscapesBinaryReliabilityDataset(
        image_dir=cfg["data"]["image_dir"],
        label_dir=cfg["data"]["label_dir"],
        rep_label_dir=cfg["data"].get("rep_label_dir"),
        resize=cfg["data"].get("resize"),
        dynamic_ids=cfg["data"].get("dynamic_ids", []),
        rep_label_ext=cfg["data"].get("rep_label_ext", ".png"),
        rep_label_downscale=cfg["data"].get("rep_label_downscale", 8),
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg["eval"].get("batch_size", 4),
        shuffle=False,
        num_workers=cfg["eval"].get("num_workers", 4),
        pin_memory=True,
    )

    model = SFD2Chap03Model(
        base_ch=cfg["model"].get("base_ch", 32),
        desc_dim=cfg["model"].get("desc_dim", 256),
        sem_ch=cfg["model"].get("sem_ch", 64),
    ).to(device)

    ckpt_path = cfg["eval"]["checkpoint"]
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    agg = {"miou": 0.0, "precision": 0.0, "recall": 0.0}
    count = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            rel_labels = batch["rel_label"].to(device)

            outputs = model(images)
            rel_pred = torch.argmax(outputs["rel_logits"], dim=1)

            for i in range(rel_pred.shape[0]):
                m = binary_metrics(rel_pred[i], rel_labels[i])
                agg["miou"] += m["miou"]
                agg["precision"] += m["precision"]
                agg["recall"] += m["recall"]
                count += 1

    for k in agg:
        agg[k] /= max(count, 1)
    print(agg)


if __name__ == "__main__":
    main()

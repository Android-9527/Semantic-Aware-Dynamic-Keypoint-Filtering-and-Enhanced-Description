import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from model import SFD2Chap03Model
from dataset import CityscapesBinaryReliabilityDataset
from losses import repeatability_loss, reliability_loss, semantic_ranking_loss


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
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"].get("num_workers", 4),
        pin_memory=True,
    )

    model = SFD2Chap03Model(
        base_ch=cfg["model"].get("base_ch", 32),
        desc_dim=cfg["model"].get("desc_dim", 256),
        sem_ch=cfg["model"].get("sem_ch", 64),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 0.0),
    )

    class_weights = cfg["train"].get("rel_class_weights", None)
    lambda_rel = cfg["train"].get("lambda_rel", 1.2)
    lambda_desc = cfg["train"].get("lambda_desc", 1.0)
    enable_desc_loss = cfg["train"].get("enable_desc_loss", False)

    model.train()
    for epoch in range(cfg["train"]["epochs"]):
        for batch in loader:
            images = batch["image"].to(device)
            rel_labels = batch["rel_label"].to(device)
            rep_labels = batch["rep_label"]
            if rep_labels is not None:
                rep_labels = rep_labels.to(device)

            outputs = model(images)

            loss = 0.0
            loss_rel = reliability_loss(outputs["rel_logits"], rel_labels, class_weights)
            loss = loss + lambda_rel * loss_rel

            if rep_labels is not None:
                loss_rep = repeatability_loss(outputs["rep_logits"], rep_labels)
                loss = loss + loss_rep
            else:
                loss_rep = torch.tensor(0.0, device=device)

            # Descriptor ranking loss placeholder
            if enable_desc_loss:
                # Requires paired descriptors; user should provide their own sampler.
                loss_desc = torch.tensor(0.0, device=device)
                loss = loss + lambda_desc * loss_desc
            else:
                loss_desc = torch.tensor(0.0, device=device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(
            f"epoch {epoch:03d} | loss {loss.item():.4f} | rel {loss_rel.item():.4f} | rep {loss_rep.item():.4f} | desc {loss_desc.item():.4f}"
        )

        ckpt_dir = Path(cfg["train"].get("checkpoint_dir", "checkpoints"))
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"model_epoch_{epoch:03d}.pth"
        torch.save({"model": model.state_dict()}, ckpt_path)


if __name__ == "__main__":
    main()

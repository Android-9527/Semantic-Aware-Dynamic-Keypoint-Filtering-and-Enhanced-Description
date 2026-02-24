import torch
import torch.nn.functional as F


def repeatability_loss(rep_logits, rep_labels):
    # rep_logits: B, 65, Hc, Wc
    # rep_labels: B, Hc, Wc (0..64)
    return F.cross_entropy(rep_logits, rep_labels.long())


def reliability_loss(rel_logits, rel_labels, class_weights=None):
    # rel_logits: B, 2, H, W
    # rel_labels: B, H, W (0=static, 1=dynamic)
    weight = None
    if class_weights is not None:
        weight = torch.tensor(class_weights, device=rel_logits.device, dtype=rel_logits.dtype)
    return F.cross_entropy(rel_logits, rel_labels.long(), weight=weight)


def semantic_ranking_loss(q, pos_geo, pos_sem, neg, margin_geo=0.1, margin_sem=0.1):
    # q, pos_geo, pos_sem, neg: B, N, C
    sim_geo = F.cosine_similarity(q, pos_geo, dim=-1)
    sim_sem = F.cosine_similarity(q, pos_sem, dim=-1)
    sim_neg = F.cosine_similarity(q, neg, dim=-1)

    loss_geo = F.relu(margin_geo - (sim_geo - sim_sem))
    loss_sem = F.relu(margin_sem - (sim_sem - sim_neg))
    return (loss_geo.mean() + loss_sem.mean())

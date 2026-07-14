"""Baseline and profile classification heads."""

from __future__ import annotations

import copy
from typing import Dict

import torch
import torch.nn as nn
from tqdm import tqdm

from .backbone import ProfileBackbone


class AVOnlyBaseline(nn.Module):
    """Audio-visual fusion only (no user profile)."""

    def __init__(self, visual_dim=768, acoustic_dim=768, hidden_dim=512, num_classes=99):
        super().__init__()
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.acoustic_proj = nn.Sequential(
            nn.Linear(acoustic_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, batch):
        visual, acoustic = batch["visual"], batch["acoustic"]
        if visual.dim() == 3:
            visual = visual.mean(dim=1)
        if acoustic.dim() == 3:
            acoustic = acoustic.mean(dim=1)
        fused = self.fusion(
            torch.cat([self.visual_proj(visual), self.acoustic_proj(acoustic)], dim=-1)
        )
        return {"logits": self.classifier(fused)}


class AVNaiveUserBaseline(nn.Module):
    """Audio-visual + naive user concat (no gating / modulation)."""

    def __init__(
        self, visual_dim=768, acoustic_dim=768, user_dim=6, hidden_dim=512, num_classes=99
    ):
        super().__init__()
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.acoustic_proj = nn.Sequential(
            nn.Linear(acoustic_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.user_proj = nn.Sequential(
            nn.Linear(user_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, batch):
        visual, acoustic, user = batch["visual"], batch["acoustic"], batch["user_profile"]
        if visual.dim() == 3:
            visual = visual.mean(dim=1)
        if acoustic.dim() == 3:
            acoustic = acoustic.mean(dim=1)
        fused = self.fusion(
            torch.cat(
                [
                    self.visual_proj(visual),
                    self.acoustic_proj(acoustic),
                    self.user_proj(user),
                ],
                dim=-1,
            )
        )
        return {"logits": self.classifier(fused)}


class ProfileModel(nn.Module):
    """Profile-conditioned backbone + classification head."""

    def __init__(
        self,
        visual_dim=768,
        acoustic_dim=768,
        user_dim=6,
        hidden_dim=512,
        num_classes=99,
        profile_mode="three_way",
        use_reliability=True,
    ):
        super().__init__()
        self.backbone = ProfileBackbone(
            visual_dim=visual_dim,
            acoustic_dim=acoustic_dim,
            user_dim=user_dim,
            track_num=3,
            hidden_size=hidden_dim,
            use_reliability=use_reliability,
            use_speech=False,
            use_norm_controller=False,
            profile_mode=profile_mode,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, batch):
        features = self.backbone(
            batch["visual"],
            batch["acoustic"],
            batch["user_profile"],
            batch["odor_track_id"],
        )
        return {"logits": self.classifier(features)}


class UniformProfileBaseline(nn.Module):
    """Same backbone as ProfileModel but with uniform (non-role-aligned) profile modulation."""

    def __init__(
        self, visual_dim=768, acoustic_dim=768, user_dim=6, hidden_dim=512, num_classes=99
    ):
        super().__init__()
        self.model = ProfileModel(
            visual_dim=visual_dim,
            acoustic_dim=acoustic_dim,
            user_dim=user_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            profile_mode="uniform",
            use_reliability=True,
        )

    def forward(self, batch):
        return self.model(batch)


class MMCLIPBaseline(nn.Module):
    """Cross-attention multimodal fusion baseline with user gating."""

    def __init__(
        self,
        visual_dim=768,
        acoustic_dim=768,
        user_dim=6,
        hidden_dim=512,
        num_classes=99,
        num_heads=8,
    ):
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.acoustic_proj = nn.Linear(acoustic_dim, hidden_dim)
        self.user_proj = nn.Linear(user_dim, hidden_dim // 2)
        self.cross_attn_va = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=0.1, batch_first=True
        )
        self.cross_attn_av = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=0.1, batch_first=True
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.user_gate = nn.Sequential(nn.Linear(hidden_dim // 2, hidden_dim), nn.Sigmoid())
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, batch):
        visual, acoustic, user = batch["visual"], batch["acoustic"], batch["user_profile"]
        if visual.dim() == 3:
            visual = visual.mean(dim=1)
        if acoustic.dim() == 3:
            acoustic = acoustic.mean(dim=1)
        v_feat = self.visual_proj(visual).unsqueeze(1)
        a_feat = self.acoustic_proj(acoustic).unsqueeze(1)
        u_feat = self.user_proj(user)
        v_attended, _ = self.cross_attn_va(v_feat, a_feat, a_feat)
        a_attended, _ = self.cross_attn_av(a_feat, v_feat, v_feat)
        fused = self.transformer(torch.cat([v_attended, a_attended], dim=1)).mean(dim=1)
        fused = fused * self.user_gate(u_feat)
        return {"logits": self.classifier(fused)}


MODEL_REGISTRY = {
    "av_only": AVOnlyBaseline,
    "av_naive_user": AVNaiveUserBaseline,
    "profile": ProfileModel,
    "uniform": UniformProfileBaseline,
    "mmclip": MMCLIPBaseline,
}


def build_model(name: str, num_classes: int = 99, **kwargs) -> nn.Module:
    key = name.lower().replace("-", "_")
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[key](num_classes=num_classes, **kwargs)


def _move_batch(batch, device):
    out = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    if "expert_labels" in out and isinstance(out["expert_labels"], dict):
        out["expert_labels"] = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in out["expert_labels"].items()
        }
    return out


def train_model(
    model,
    train_loader,
    val_loader,
    epochs=30,
    lr=1e-4,
    model_name="Model",
    device=None,
):
    """Train with CE + label smoothing; keep best val Top-1 checkpoint."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 70}\n训练: {model_name}\n{'=' * 70}")

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_state = None
    history = {"train_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False):
            batch = _move_batch(batch, device)
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs["logits"], batch["expert_labels"]["scent_name_id"])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = _move_batch(batch, device)
                preds = model(batch)["logits"].argmax(dim=1)
                labels = batch["expert_labels"]["scent_name_id"]
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / max(total, 1) * 100
        avg_loss = train_loss / max(len(train_loader), 1)
        history["train_loss"].append(avg_loss)
        history["val_acc"].append(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1:2d}: Loss={avg_loss:.4f}, "
                f"Val Acc={val_acc:.2f}% (Best={best_val_acc:.2f}%)"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_val_acc


def evaluate_model(model, test_loader, model_name="", device=None) -> Dict[str, float]:
    """Evaluate Top-1/3/5 and MRR."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 70}\n测试: {model_name}\n{'=' * 70}")

    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = _move_batch(batch, device)
            all_logits.append(model(batch)["logits"].cpu())
            all_labels.append(batch["expert_labels"]["scent_name_id"].cpu())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    top1 = (logits.argmax(dim=1) == labels).float().mean().item() * 100
    top3 = (
        logits.topk(3, dim=1).indices.eq(labels.unsqueeze(1)).any(dim=1).float().mean().item()
        * 100
    )
    top5 = (
        logits.topk(5, dim=1).indices.eq(labels.unsqueeze(1)).any(dim=1).float().mean().item()
        * 100
    )
    ranks = (logits.argsort(dim=1, descending=True) == labels.unsqueeze(1)).nonzero(
        as_tuple=True
    )[1]
    mrr = (1.0 / (ranks.float() + 1)).mean().item()
    results = {"top1": top1, "top3": top3, "top5": top5, "mrr": mrr, "n": int(labels.size(0))}
    print(
        f"\n测试结果:\n  Top-1: {top1:.2f}%\n  Top-3: {top3:.2f}%\n"
        f"  Top-5: {top5:.2f}%\n  MRR:   {mrr:.3f}"
    )
    return results

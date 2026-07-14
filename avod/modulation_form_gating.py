"""Profile modulation form ablations (改动 3).

Design principles (R7):
  - role-aligned uses ThreeWayUserGating in backbone (Table2-identical).
  - Ablations share the same hybrid semantics: sens=multiplicative, tol/emo=additive,
    with paper masks tol@(BG+Emo) = tracks (1,2), emo@(Emotion) = track 2.
  - random_field_track uses a fixed field→track swap (negative control).
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Matches ThreeWayUserGating in backbone.py
ROLE_ALIGNED_TOL_TRACKS = (1, 2)
ROLE_ALIGNED_EMO_TRACK = 2

# Deliberate wrong mapping: swap tol ↔ emo track targets
RANDOM_TOL_TRACKS = (2,)
RANDOM_EMO_TRACKS = (0, 1)

ABLATION_MODES = (
    "additive_all",
    "multiplicative_all",
    "no_track_selectivity",
    "random_field_track",
)


def _track_mask(track_id: torch.Tensor, tracks: tuple[int, ...]) -> torch.Tensor:
    mask = torch.zeros(track_id.shape[0], 1, device=track_id.device, dtype=torch.float32)
    for t in tracks:
        mask = mask + (track_id == t).float().unsqueeze(-1)
    return mask.clamp(0.0, 1.0)


class ModulationFormGating(nn.Module):
    """
    Field-wise profile modulation for ablation (non role-aligned variants).

    Modes:
      additive_all:           all fields additive (incl. sens)
      multiplicative_all:     all fields multiplicative (incl. tol/emo)
      no_track_selectivity:   role-aligned ops, masks removed
      random_field_track:     role-aligned ops, fixed wrong field→track map
    """

    def __init__(self, mode: str, hidden_dim: int = 512, track_num: int = 3):
        super().__init__()
        if mode not in ABLATION_MODES:
            raise ValueError(f"unsupported ablation mode: {mode}")
        self.mode = mode
        self.hidden_dim = hidden_dim

        self.sens_gate_proj = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.sens_add_proj = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )
        self.tol_proj = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )
        self.tol_gate_proj = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.emo_proj = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )
        self.emo_gate_proj = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.track_embed = nn.Embedding(track_num, 8)

        if mode == "random_field_track":
            self.tol_tracks = RANDOM_TOL_TRACKS
            self.emo_tracks = RANDOM_EMO_TRACKS
        else:
            self.tol_tracks = ROLE_ALIGNED_TOL_TRACKS
            self.emo_tracks = (ROLE_ALIGNED_EMO_TRACK,)

    def apply_identity_friendly_init(self) -> None:
        """初始近似恒等，与 ThreeWayUserGating 对齐。"""
        with torch.no_grad():
            last = self.sens_gate_proj[-2]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.constant_(last.bias, 2.5)
            for proj in (
                self.sens_add_proj,
                self.tol_proj,
                self.emo_proj,
                self.tol_gate_proj[:-1],
                self.emo_gate_proj[:-1],
            ):
                for m in proj:
                    if isinstance(m, nn.Linear):
                        nn.init.zeros_(m.weight)
                        nn.init.zeros_(m.bias)
            for gate in (self.tol_gate_proj[-1], self.emo_gate_proj[-1]):
                if isinstance(gate, nn.Linear):
                    nn.init.zeros_(gate.weight)
                    nn.init.constant_(gate.bias, 2.5)

    def _blend_multiplicative(
        self, h: torch.Tensor, gate: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        # mask=1 → multiply by gate; mask=0 → identity
        return h * (1.0 + mask.unsqueeze(1) * (gate.unsqueeze(1) - 1.0))

    def forward(self, h_vas: torch.Tensor, user: torch.Tensor, track_id: torch.Tensor) -> torch.Tensor:
        u_sens = user[:, 2:3]
        u_tol = user[:, 3:4]
        u_emo = user[:, 4:5]

        use_mask = self.mode != "no_track_selectivity"
        if use_mask:
            tol_mask = _track_mask(track_id, self.tol_tracks)
            emo_mask = _track_mask(track_id, self.emo_tracks)
        else:
            tol_mask = torch.ones(track_id.shape[0], 1, device=track_id.device)
            emo_mask = tol_mask

        all_add = self.mode == "additive_all"
        all_mult = self.mode == "multiplicative_all"

        if all_add:
            h = h_vas + self.sens_add_proj(u_sens).unsqueeze(1)
        else:
            h = h_vas * self.sens_gate_proj(u_sens).unsqueeze(1)

        if all_mult:
            h = self._blend_multiplicative(h, self.tol_gate_proj(u_tol), tol_mask)
            h = self._blend_multiplicative(h, self.emo_gate_proj(u_emo), emo_mask)
        else:
            tol_sign = torch.tanh(2 * u_tol - 1)
            tol_bias = tol_sign * self.tol_proj(u_tol)
            h = h + tol_mask.unsqueeze(1) * tol_bias.unsqueeze(1)
            emo_bias = u_emo * self.emo_proj(u_emo)
            h = h + emo_mask.unsqueeze(1) * emo_bias.unsqueeze(1)

        return h

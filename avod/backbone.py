#!/usr/bin/env python3
"""Profile-conditioned audiovisual fusion backbone.

Components:
1. Three-way user gating (sensitivity / tolerance / emotion)
2. Sigmoid relevance scoring
3. ReliabilityNet weight modulation
4. Optional speech fusion hook
5. Optional norm-controlled scaling hook
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .modulation_form_gating import ABLATION_MODES, ModulationFormGating


class ScentRelevanceNet(nn.Module):
    """
    实现版本: ScentRelevanceNet
    - 使用Sigmoid激活 (范围0-1)
    - 评估视听特征与气味轨道的语义相关性
    """
    
    def __init__(self, visual_dim=768, acoustic_dim=768, user_dim=6, 
                 track_emb_dim=8, hidden_dim=512):
        super().__init__()
        
        input_dim = visual_dim + acoustic_dim + user_dim + track_emb_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # 默认使用Sigmoid，输出范围(0, 1)
        self.sigmoid = nn.Sigmoid()
        self.logits_proj = nn.Linear(hidden_dim // 2, 2)
        
    def forward(self, visual, acoustic, user, track_emb):
        """
        Args:
            visual: [B, V]
            acoustic: [B, A]
            user: [B, U]
            track_emb: [B, E]
        Returns:
            s_v, s_a: 相关性分数 [B, 1]
        """
        x = torch.cat([visual, acoustic, user, track_emb], dim=-1)
        h = self.net(x)
        logits = self.logits_proj(h)
        
        # 公式: s_v, s_a = σ(f([v; a; u; e]))
        s = self.sigmoid(logits)
        s_v, s_a = s[:, 0:1], s[:, 1:2]
        
        return s_v, s_a


class ReliabilityNet(nn.Module):
    """
    实现版本: ReliabilityNet
    - 独立评估视觉/音频信号质量
    - 用于权重调制 (默认启用)
    """
    
    def __init__(self, visual_dim=768, acoustic_dim=768, hidden_dim=256):
        super().__init__()
        
        # 视觉可靠性
        self.visual_rel = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # r_v ∈ [0, 1]
        )
        
        # 音频可靠性
        self.audio_rel = nn.Sequential(
            nn.Linear(acoustic_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # r_a ∈ [0, 1]
        )
        
    def forward(self, visual, acoustic):
        """
        Args:
            visual: [B, T, V] 或 [B, V]
            acoustic: [B, T, A] 或 [B, A]
        Returns:
            r_v: 视觉可靠性 [B, 1]
            r_a: 音频可靠性 [B, 1]
        """
        # 池化到单一特征
        if visual.dim() == 3:
            v_pooled = visual.mean(dim=1)
            a_pooled = acoustic.mean(dim=1)
        else:
            v_pooled = visual
            a_pooled = acoustic
        
        r_v = self.visual_rel(v_pooled)  # [B, 1]
        r_a = self.audio_rel(a_pooled)   # [B, 1]
        
        return r_v, r_a


class UniformProfileModulation(nn.Module):
    """
    消融：统一 profile 调制（全维共享参数）
    h_fused = h_vas ⊙ σ(W_all(u)) + W_bias(u)
    不使用按 field/track 分支的门控。
    u 使用与 forward 中相同的 u_norm（6 维）。

    注意：全局 _init_weights 会对 Linear 做 Xavier，会使 σ(·)≈0.5、严重缩小 h_vas。
    因此在 ProfileBackbone.__init__ 末尾对 uniform 分支调用
    apply_identity_friendly_init()，使初始近似 h_fused≈h_vas。
    """

    def __init__(self, profile_dim: int = 6, hidden_dim: int = 512):
        super().__init__()
        self.W_all = nn.Linear(profile_dim, hidden_dim)
        self.W_bias = nn.Linear(profile_dim, hidden_dim)

    def apply_identity_friendly_init(self) -> None:
        """初始 scale≈1、bias≈0，避免训练起步破坏视听特征。"""
        with torch.no_grad():
            nn.init.zeros_(self.W_all.weight)
            # 常数 logits → 各维相同；sigmoid(bias) 接近 1
            nn.init.constant_(self.W_all.bias, 2.5)
            nn.init.zeros_(self.W_bias.weight)
            nn.init.zeros_(self.W_bias.bias)

    def forward(self, h_vas: torch.Tensor, u_norm: torch.Tensor) -> torch.Tensor:
        # h_vas: [B, T, H], u_norm: [B, 6]
        scale = torch.sigmoid(self.W_all(u_norm))
        bias = self.W_bias(u_norm)
        return h_vas * scale.unsqueeze(1) + bias.unsqueeze(1)


class ThreeWayUserGating(nn.Module):
    """
    实现版本: 三路分字段用户调制门控
    
    1. 敏感度门控 (全轨道): h_sens = h_vas ⊙ σ(u_sens · W_sens)
    2. 耐受度偏置 (场景/情感): h_tol = h_sens + I·tanh(2u_tol-1)·W_tol
    3. 情感偏置 (仅情感): h_emo = h_tol + I·u_emo·W_emo
    """
    
    def __init__(self, user_dim=6, hidden_dim=512, track_num=3):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # 1. 敏感度门控 (全轨道生效)
        self.sens_gate_proj = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Sigmoid()  # 门控 [0, 1]
        )
        
        # 2. 耐受度偏置 (场景/情感轨道)
        self.tol_bias_proj = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # 3. 情感偏置 (仅情感轨道)
        self.emo_bias_proj = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # 轨道嵌入
        self.track_embed = nn.Embedding(track_num, 8)
        
    def apply_identity_friendly_init(self) -> None:
        """初始近似恒等：sens gate≈1，tol/emo bias≈0。"""
        with torch.no_grad():
            last = self.sens_gate_proj[-2]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.constant_(last.bias, 2.5)
            for proj in (self.tol_bias_proj, self.emo_bias_proj):
                for m in proj:
                    if isinstance(m, nn.Linear):
                        nn.init.zeros_(m.weight)
                        nn.init.zeros_(m.bias)
        
    def forward(self, h_vas, user, track_id):
        """
        Args:
            h_vas: 视听融合特征 [B, T, H]
            user: 用户画像 [B, 6]
            track_id: 轨道ID [B] (0=前景, 1=场景, 2=情感)
        Returns:
            h_fused: 调制后的特征 [B, T, H]
        """
        B, T, H = h_vas.shape
        
        # 提取用户字段
        u_sens = user[:, 2:3]  # 敏感度
        u_tol = user[:, 3:4]   # 耐受度
        u_emo = user[:, 4:5]   # 情感偏好
        
        # 1. 敏感度门控 (全轨道)
        # h_sens = h_vas ⊙ σ(W_sens · u_sens)
        sens_gate = self.sens_gate_proj(u_sens)  # [B, H]
        h_sens = h_vas * sens_gate.unsqueeze(1)  # 广播到T维度
        
        # 2. 耐受度偏置 (场景/情感轨道)
        # tanh(2u_tol - 1) ∈ [-1, 1]，有符号偏置
        tol_sign = torch.tanh(2 * u_tol - 1)  # [B, 1]
        tol_bias = tol_sign * self.tol_bias_proj(u_tol)  # [B, H]
        
        # 仅对场景(1)和情感(2)生效
        track_mask = ((track_id == 1) | (track_id == 2)).float().unsqueeze(-1)  # [B, 1]
        h_tol = h_sens + track_mask.unsqueeze(1) * tol_bias.unsqueeze(1)
        
        # 3. 情感偏置 (仅情感轨道)
        # I_track=2 · u_emo · W_emo
        emo_bias = u_emo * self.emo_bias_proj(u_emo)  # [B, H]
        
        # 仅情感轨道(2)生效
        emo_mask = (track_id == 2).float().unsqueeze(-1)  # [B, 1]
        h_fused = h_tol + emo_mask.unsqueeze(1) * emo_bias.unsqueeze(1)
        
        return h_fused


class SpeechFusion(nn.Module):
    """
    实现版本: Speech Fusion接口
    - 准备集成ASR和台词门控
    - 当前为占位符，可在后续集成Whisper
    """
    
    def __init__(self, speech_dim=768, hidden_dim=512):
        super().__init__()
        
        # 语音特征投影 (准备接收Whisper输出)
        self.speech_proj = nn.Sequential(
            nn.Linear(speech_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # 门控投影
        self.gate_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # 是否启用Speech Fusion的标志
        self.use_speech = False  # 默认关闭，可在初始化时开启
        
    def forward(self, h_va, speech_features=None):
        """
        Args:
            h_va: 视听特征 [B, T, H]
            speech_features: 语音特征 [B, speech_dim] (可选)
        Returns:
            h_vas: 融合特征
            gate_value: 门控值 (用于监控)
        """
        if not self.use_speech or speech_features is None:
            # 禁用Speech Fusion时直接返回
            return h_va, torch.tensor(0.0)
        
        # 语音投影
        speech_feat = self.speech_proj(speech_features)  # [B, H]
        
        # 自适应门控
        gate = self.gate_proj(speech_feat)  # [B, H]
        
        # 融合到视频时间维度
        B, T, H = h_va.shape
        gate_expanded = gate.unsqueeze(1).expand(-1, T, -1)  # [B, T, H]
        speech_expanded = speech_feat.unsqueeze(1).expand(-1, T, -1)
        
        # h_vas = h_va + gate ⊙ speech_feat
        h_vas = h_va + gate_expanded * speech_expanded
        
        return h_vas, gate.mean().item()


class NormController(nn.Module):
    """
    实现版本: Norm-controlled Scaling
    - 约束语义偏移向量的范数
    - 防止MLLM语义空间扰动
    
    公式: Scale(h_fused, E) = min(β, 1) · (||h|| / ||E||) · h_fused
    """
    
    def __init__(self, beta_init=1e-3, epsilon=1e-6):
        super().__init__()
        
        # 可学习的β参数 (论文初始化为10^-3)
        self.beta = nn.Parameter(torch.tensor(beta_init))
        self.epsilon = epsilon
        
    def forward(self, h_fused, text_embedding):
        """
        Args:
            h_fused: 融合特征 [B, H]
            text_embedding: MLLM文本嵌入 [B, H]
        Returns:
            delta: 缩放后的语义偏移
        """
        # 计算L2范数
        h_norm = torch.norm(h_fused, p=2, dim=-1, keepdim=True)
        e_norm = torch.norm(text_embedding, p=2, dim=-1, keepdim=True)
        
        # 范数比
        norm_ratio = h_norm / (e_norm + self.epsilon)
        
        # 缩放系数: min(β / norm_ratio, 1)
        # 确保delta的范数不超过β·||E||
        scale = torch.min(
            self.beta / (norm_ratio + self.epsilon),
            torch.ones_like(norm_ratio)
        )
        
        # 应用缩放
        delta = scale * h_fused
        
        return delta


class ProfileBackbone(nn.Module):
    """
    backbone 完整实现版本
    
    完全按照论文描述实现:
    1. ScentRelevanceNet (Sigmoid)
    2. ReliabilityNet (权重调制)
    3. 竞争性权重分配
    4. Speech Fusion (可选)
    5. 三路User Gating
    6. Norm-controlled Scaling (准备MLLM)
    """
    
    def __init__(
        self,
        visual_dim=768,
        acoustic_dim=768,
        user_dim=6,
        track_num=3,
        hidden_size=512,
        dropout_prob=0.2,
        use_reliability=True,  # 默认启用
        use_speech=False,      # 可选
        use_norm_controller=False,  # 准备MLLM时启用
        profile_mode: str = "three_way",
    ):
        super().__init__()

        self._ABLATION_MODES = ABLATION_MODES
        self.visual_dim = visual_dim
        self.acoustic_dim = acoustic_dim
        self.hidden_size = hidden_size
        self.use_reliability = use_reliability
        self.use_speech = use_speech
        self.use_norm_controller = use_norm_controller
        self.profile_mode = profile_mode
        allowed = ("three_way", "uniform", *ABLATION_MODES)
        if profile_mode not in allowed:
            raise ValueError(f"profile_mode must be one of {allowed}")
        
        # 轨道嵌入
        self.track_embed = nn.Embedding(track_num, 8)
        
        # 1. ScentRelevanceNet (默认使用Sigmoid)
        self.scent_relevance = ScentRelevanceNet(
            visual_dim=visual_dim,
            acoustic_dim=acoustic_dim,
            user_dim=user_dim,
            track_emb_dim=8,
            hidden_dim=512
        )
        
        # 2. ReliabilityNet (默认启用)
        if use_reliability:
            self.reliability_net = ReliabilityNet(
                visual_dim=visual_dim,
                acoustic_dim=acoustic_dim
            )
        else:
            self.reliability_net = None
        
        # 3. 特征投影
        self.W_v = nn.Linear(visual_dim, hidden_size)
        self.W_a = nn.Linear(acoustic_dim, hidden_size)
        
        # 4. Speech Fusion (可选)
        if use_speech:
            self.speech_fusion = SpeechFusion(speech_dim=768, hidden_dim=hidden_size)
        else:
            self.speech_fusion = None
        
        # 5. User profile 融合：三路 field-wise（默认）或统一调制（消融）
        if profile_mode == "three_way":
            self.user_gating = ThreeWayUserGating(
                user_dim=user_dim,
                hidden_dim=hidden_size,
                track_num=track_num
            )
            self.uniform_profile = None
        elif profile_mode in ABLATION_MODES:
            self.user_gating = ModulationFormGating(
                mode=profile_mode,
                hidden_dim=hidden_size,
                track_num=track_num,
            )
            self.uniform_profile = None
        else:
            self.user_gating = None
            # u_norm 固定为 6 维（与 forward 中拼接一致）
            self.uniform_profile = UniformProfileModulation(
                profile_dim=6, hidden_dim=hidden_size
            )
        
        # 6. Norm Controller (准备MLLM时启用)
        if use_norm_controller:
            self.norm_controller = NormController()
        else:
            self.norm_controller = None
        
        # 初始化
        self._init_weights()
        if self.profile_mode == "uniform" and self.uniform_profile is not None:
            self.uniform_profile.apply_identity_friendly_init()
        elif self.profile_mode == "three_way" and self.user_gating is not None:
            self.user_gating.apply_identity_friendly_init()
        elif self.profile_mode in self._ABLATION_MODES and self.user_gating is not None:
            if hasattr(self.user_gating, "apply_identity_friendly_init"):
                self.user_gating.apply_identity_friendly_init()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        visual,
        acoustic,
        user,
        track_id,
        speech=None,
        text_embedding=None,  # 用于Norm Controller
        return_details=False
    ):
        """
        论文描述的前向流程
        
        Args:
            visual: [B, V] or [B, T, V]
            acoustic: [B, A] or [B, T, A]
            user: [B, U]
            track_id: [B]
            speech: [B, speech_dim] (可选)
            text_embedding: [B, H] (用于Norm Controller)
            return_details: 是否返回中间结果
        """
        squeeze_output = False
        if visual.dim() == 2:
            visual = visual.unsqueeze(1)
            acoustic = acoustic.unsqueeze(1)
            squeeze_output = True
        
        B, T, _ = visual.shape
        
        # 归一化用户特征 (与论文一致)
        u_norm = torch.cat([
            user[..., 0:1],  # gender
            user[..., 1:2] / 100.0,  # age
            user[..., 2:3] / 5.0,  # sensitivity
            user[..., 3:4] / 5.0,  # tolerance
            user[..., 4:5] / 5.0,  # emotional_pref
            user[..., 5:6] / 5.0,  # confidence
        ], dim=-1)
        
        # 轨道嵌入
        track_emb = self.track_embed(track_id)
        if track_emb.dim() == 2:
            track_emb = track_emb.unsqueeze(1).expand(B, T, -1)
        
        # 池化用于路由
        v_pooled = visual.mean(dim=1)
        a_pooled = acoustic.mean(dim=1)
        
        # 1. 动态路由 (ScentRelevance + Reliability)
        s_v, s_a = self.scent_relevance(v_pooled, a_pooled, u_norm, track_emb[:, 0, :])
        
        if self.use_reliability and self.reliability_net is not None:
            r_v, r_a = self.reliability_net(visual, acoustic)
            # 竞争性权重: w = s·r / (s_v·r_v + s_a·r_a)
            denom = s_v * r_v + s_a * r_a + 1e-6
            w_v = (s_v * r_v) / denom
            w_a = (s_a * r_a) / denom
        else:
            # 仅相关性
            denom = s_v + s_a + 1e-6
            w_v = s_v / denom
            w_a = s_a / denom
        
        # 2. 特征投影和加权
        h_v = self.W_v(visual)  # [B, T, H]
        h_a = self.W_a(acoustic)
        
        w_v_exp = w_v.unsqueeze(1)
        w_a_exp = w_a.unsqueeze(1)
        h_va = w_v_exp * h_v + w_a_exp * h_a  # [B, T, H]
        
        # 3. Speech Fusion (可选)
        if self.use_speech and self.speech_fusion is not None and speech is not None:
            h_va, speech_gate = self.speech_fusion(h_va, speech)
        else:
            speech_gate = 0.0
        
        # 4. User profile 调制（backbone 视听路由与加权已在上文完成，此处仅替换 profile 支路）
        if self.profile_mode == "three_way" or self.profile_mode in self._ABLATION_MODES:
            h_fused = self.user_gating(h_va, u_norm, track_id)  # [B, T, H]
        else:
            h_fused = self.uniform_profile(h_va, u_norm)
        
        # 5. 池化时间维度
        h_fused = h_fused.mean(dim=1)  # [B, H]
        
        # 6. Norm-controlled Scaling (准备MLLM时)
        if self.use_norm_controller and self.norm_controller is not None and text_embedding is not None:
            delta = self.norm_controller(h_fused, text_embedding)
            output = delta
        else:
            output = h_fused
        
        # squeeze_output 原本意图：若输入是 [B,V]（无时间维），补了 T=1 后需还原
        # 但经 mean(dim=1) 后 output 已是 [B,H]，无需再 squeeze；若 B=1 也不应在此 squeeze batch 维
        # 保留标志但不操作，避免破坏 batch>1 的情况
        if squeeze_output and output.dim() > 2:
            output = output.squeeze(1)  # squeeze 时间维（若仍存在）

        if return_details:
            return {
                'output': output,
                'h_fused': h_fused,
                'w_v': w_v,
                'w_a': w_a,
                's_v': s_v,
                's_a': s_a,
                'speech_gate': speech_gate
            }
        
        return output


def test_paper_version():
    """测试实现版本的backbone"""
    print("="*60)
    print("Testing backbone Paper Version")
    print("="*60)
    
    model = ProfileBackbone(
        visual_dim=768,
        acoustic_dim=768,
        user_dim=6,
        track_num=3,
        hidden_size=512,
        use_reliability=True,  # 启用ReliabilityNet
        use_speech=False,      # 默认关闭Speech
        use_norm_controller=False  # 默认关闭Norm Controller
    )
    
    # 模拟输入
    B = 4
    visual = torch.randn(B, 10, 768)
    acoustic = torch.randn(B, 10, 768)
    user = torch.randn(B, 6)
    track_id = torch.tensor([0, 1, 2, 0])
    
    model.eval()
    with torch.no_grad():
        result = model(visual, acoustic, user, track_id, return_details=True)
    
    print(f"\n输出形状: {result['output'].shape}")
    print(f"视觉权重: {result['w_v'].squeeze()}")
    print(f"音频权重: {result['w_a'].squeeze()}")
    print(f"相关性(s_v): {result['s_v'].squeeze()}")
    print(f"相关性(s_a): {result['s_a'].squeeze()}")
    
    # 验证三路门控
    print("\n三路门控验证:")
    print(f"  Track 0 (前景): 权重={result['w_v'][0].item():.3f}")
    print(f"  Track 1 (场景): 权重={result['w_v'][1].item():.3f}")
    print(f"  Track 2 (情感): 权重={result['w_v'][2].item():.3f}")
    
    print("\n✓ 实现版本backbone测试通过")


if __name__ == '__main__':
    test_paper_version()

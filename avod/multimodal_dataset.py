"""
多模态 Dataset 类
数据格式：video_file, user_profile, foreground_scent, scene_scent, emotional
（三轨：前景=object+action 合并，背景=scene，情感=emotional）
"""

import json
import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Union, Any


# 气味轨道映射: foreground_scent=0, scene_scent=1, emotional=2（三轨）
TRACK_KEYS = ["foreground_scent", "scene_scent", "emotional"]


def _normalize_to_3track(raw: Dict[str, Any]) -> Dict[str, Any]:
    """兼容旧 4 轨格式：若无 foreground_scent 则从 object_scent+action_scent 合并"""
    if "foreground_scent" in raw:
        return raw
    obj = raw.get("object_scent", [])
    act = raw.get("action_scent", [])
    out = dict(raw)
    out["foreground_scent"] = list(obj) + list(act)
    for k in ("object_scent", "action_scent"):
        out.pop(k, None)
    return out


class MultimodalOdorDataset(Dataset):
    """
    多模态气味分析数据集（适配实际数据格式）

    每条原始样本包含：
        - video_file: 视频路径
        - user_profile: 性别、年龄、气味敏感度、 unpleasant_tolerance 等
        - foreground_scent, scene_scent, emotional: 三类气味轨道的专家标注
        - 每轨道: [{start_time, end_time, scent: {name, intensity, form, temperature}}]
    """

    # 相态映射
    FORM_MAP = {"气态": 0, "液态": 1, "固态": 2}
    # 温度映射
    TEMP_MAP = {"温和": 0, "冷": 1, "热": 2}

    def __init__(
        self,
        data_path: str,
        feature_root: Optional[str] = None,
        visual_dim: int = 768,
        acoustic_dim: int = 768,   # HuBERT-base=768；MFCC 降级时传 128
        speech_dim: int = 768,     # Whisper-small=768；0 表示不加载语音特征
        transform: Optional[callable] = None,
        data_format: str = "json",
        sample_mode: str = "segment",  # "segment" | "video"
    ):
        """
        Args:
            data_path: 数据 JSON/PKL 路径
            feature_root: 预提取特征根目录。
                          特征文件名为 {video_id}_visual.npy / _acoustic.npy / _speech.npy
            visual_dim:   视觉特征维度
            acoustic_dim: 音频特征维度（HuBERT）
            speech_dim:   台词语义特征维度（Whisper），0 表示不加载
            transform:    可选数据增强
            data_format:  "json" | "pkl"
            sample_mode:  "segment" 每个片段一条样本; "video" 取各轨道首片段
        """
        self.feature_root = feature_root
        self.visual_dim   = visual_dim
        self.acoustic_dim = acoustic_dim
        self.speech_dim   = speech_dim
        self.use_speech   = (speech_dim > 0)
        self.transform    = transform
        self.data_format  = data_format
        self.sample_mode  = sample_mode

        self.raw_samples = self._load_data(data_path)
        self.samples = self._build_sample_index()
        self._build_scent_name_vocab()

    def _load_data(self, data_path: str) -> List[Dict[str, Any]]:
        if self.data_format == "json":
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif self.data_format == "pkl":
            import pickle
            with open(data_path, "rb") as f:
                data = pickle.load(f)
        else:
            raise ValueError(f"不支持的数据格式: {self.data_format}")

        rows = data["samples"] if isinstance(data, dict) and "samples" in data else (data if isinstance(data, list) else [data])
        return [_normalize_to_3track(s) for s in rows]

    def _build_sample_index(self) -> List[Dict[str, Any]]:
        """根据 sample_mode 展开为可索引的样本列表，过滤无特征的样本"""
        samples = []
        skipped = 0
        for raw_idx, raw in enumerate(self.raw_samples):
            video_file = raw.get("video_file", "")
            profile = raw.get("user_profile", {})
            extra = {}
            if "visual" in raw and "acoustic" in raw:
                extra["visual"] = raw["visual"]
                extra["acoustic"] = raw["acoustic"]
            
            # 检查是否有外部特征
            if self.feature_root and not extra:
                fid = self._video_to_feature_id(video_file)
            
            if self.sample_mode == "segment":
                for track_id, key in enumerate(TRACK_KEYS):
                    segments = raw.get(key, [])
                    for seg in segments:
                        # 如果使用外部特征，检查 visual + acoustic 均存在
                        if self.feature_root and not extra:
                            start_time = int(seg.get('start_time', 0))
                            end_time = int(seg.get('end_time', 0))
                            track_abbr = {0: 'fg', 1: 'sc', 2: 'em'}.get(track_id, 'fg')
                            visual_path = os.path.join(self.feature_root, f"{fid}_{track_abbr}_{start_time}_{end_time}_visual.npy")
                            acoustic_path = os.path.join(self.feature_root, f"{fid}_{track_abbr}_{start_time}_{end_time}_acoustic.npy")
                            if not (os.path.exists(visual_path) and os.path.exists(acoustic_path)):
                                skipped += 1
                                continue  # 跳过无特征的样本
                        
                        entry = {
                            "video_file": video_file,
                            "user_profile": profile,
                            "odor_track_id": track_id,
                            "segment": seg,
                            "raw_row_idx": raw_idx,
                            **extra,
                        }
                        for meta_key in ("clip_key", "sample_key", "annotation_id", "synthetic", "category"):
                            if meta_key in raw:
                                entry[meta_key] = raw[meta_key]
                        samples.append(entry)
            else:
                for track_id, key in enumerate(TRACK_KEYS):
                    segments = raw.get(key, [])
                    if segments:
                        seg = segments[0]
                        # 如果使用外部特征，检查特征文件是否存在
                        if self.feature_root and not extra:
                            start_time = int(seg.get('start_time', 0))
                            end_time = int(seg.get('end_time', 0))
                            track_abbr = {0: 'fg', 1: 'sc', 2: 'em'}.get(track_id, 'fg')
                            visual_path = os.path.join(self.feature_root, f"{fid}_{track_abbr}_{start_time}_{end_time}_visual.npy")
                            acoustic_path = os.path.join(self.feature_root, f"{fid}_{track_abbr}_{start_time}_{end_time}_acoustic.npy")
                            if not (os.path.exists(visual_path) and os.path.exists(acoustic_path)):
                                skipped += 1
                                continue  # 跳过无特征的样本
                        
                        entry = {
                            "video_file": video_file,
                            "user_profile": profile,
                            "odor_track_id": track_id,
                            "segment": seg,
                            "raw_row_idx": raw_idx,
                            **extra,
                        }
                        for meta_key in ("clip_key", "sample_key", "annotation_id", "synthetic", "category"):
                            if meta_key in raw:
                                entry[meta_key] = raw[meta_key]
                        samples.append(entry)
        
        if skipped > 0:
            print(f"  跳过 {skipped} 个无特征的样本")
        
        return samples

    def _build_scent_name_vocab(self):
        """从数据中收集 scent.name 并建立词汇表"""
        vocab = {"None": 0}
        for raw in self.raw_samples:
            for key in TRACK_KEYS:
                for seg in raw.get(key, []):
                    scent = seg.get("scent", {})
                    name = scent.get("name", "None")
                    if name not in vocab:
                        vocab[name] = len(vocab)
        self.scent_name_vocab = vocab

    def _video_to_feature_id(self, video_file: str) -> str:
        """将 video_file 转为特征文件名（去掉扩展名、替换非法字符）"""
        # 使用rsplit只分割最后一个点，避免文件名中包含多个点的问题
        base = os.path.basename(video_file).rsplit('.', 1)[0]
        return base  # 保留原始字符，特征文件名也保留了原始格式

    def _load_video_features(self, video_file: str, segment: Dict = None, track_id: int = 0):
        """
        加载视觉、音频、语音（可选）特征，返回 (visual, acoustic, speech|None) 或 None
        
        支持两种格式:
        1. 全局视频特征: {video_id}_visual.npy
        2. 时间段特征: {video_id}_{track}_{start}_{end}_visual.npy
        
        如果特征文件不存在，返回None表示该样本应被跳过
        """
        fid = self._video_to_feature_id(video_file)
        if not self.feature_root:
            raise ValueError("需指定 feature_root 或样本中提供 visual/acoustic")
        
        # 检查是否是时间段级别特征
        if segment is not None and 'start_time' in segment and 'end_time' in segment:
            start_time = int(segment['start_time'])
            end_time = int(segment['end_time'])
            # 获取轨道简称 (fg, sc, em)
            track_abbr = {0: 'fg', 1: 'sc', 2: 'em'}.get(track_id, 'fg')
            
            # 构建时间段特征文件名
            visual_path = os.path.join(self.feature_root, f"{fid}_{track_abbr}_{start_time}_{end_time}_visual.npy")
            acoustic_path = os.path.join(self.feature_root, f"{fid}_{track_abbr}_{start_time}_{end_time}_acoustic.npy")
            
            # 如果时间段特征不存在，尝试加载全局特征
            if not os.path.exists(visual_path):
                visual_path = os.path.join(self.feature_root, f"{fid}_visual.npy")
                acoustic_path = os.path.join(self.feature_root, f"{fid}_acoustic.npy")
        else:
            # 使用全局特征
            visual_path = os.path.join(self.feature_root, f"{fid}_visual.npy")
            acoustic_path = os.path.join(self.feature_root, f"{fid}_acoustic.npy")
        
        # 检查特征文件是否存在
        if not os.path.exists(visual_path) or not os.path.exists(acoustic_path):
            return None  # 特征缺失，返回None
        
        try:
            visual = np.load(visual_path).astype(np.float32)
            acoustic = np.load(acoustic_path).astype(np.float32)
        except Exception as e:
            print(f"Error loading features for {video_file}: {e}")
            return None
        
        speech = None
        if self.use_speech:
            sp_path = os.path.join(self.feature_root, f"{fid}_speech.npy")
            if os.path.exists(sp_path):
                speech = np.load(sp_path).astype(np.float32)
            else:
                speech = np.zeros(self.speech_dim, dtype=np.float32)
        
        return visual, acoustic, speech

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        video_file = item["video_file"]
        profile = item["user_profile"]
        track_id = item["odor_track_id"]
        seg = item["segment"]

        # 1. 视觉、音频、语音特征
        if "visual" in item and "acoustic" in item:
            visual   = self._get_feature(item["visual"],   self.visual_dim)
            acoustic = self._get_feature(item["acoustic"], self.acoustic_dim)
            speech   = (self._get_feature(item["speech"],  self.speech_dim)
                        if (self.use_speech and "speech" in item)
                        else (torch.zeros(self.speech_dim) if self.use_speech else None))
        elif self.feature_root:
            visual_np, acoustic_np, speech_np = self._load_video_features(video_file, seg, track_id)
            visual   = torch.from_numpy(self._pad_or_truncate_feature(visual_np,   self.visual_dim))
            acoustic = torch.from_numpy(self._pad_or_truncate_feature(acoustic_np, self.acoustic_dim))
            speech   = (torch.from_numpy(self._pad_or_truncate_feature(speech_np,  self.speech_dim))
                        if speech_np is not None else None)
        else:
            raise ValueError("需指定 feature_root 或在样本中提供 visual/acoustic")

        # 2. 用户画像（拼接为tensor）
        user_profile_dict = self._parse_user_profile(profile)
        # 拼接为tensor: [gender, age, scent_sensitivity, unpleasant_tolerance, emotional_scent_preference, scent_confidence]
        user_profile = torch.stack([
            user_profile_dict["gender"].float(),
            user_profile_dict["age"],
            user_profile_dict["scent_sensitivity"],
            user_profile_dict["unpleasant_tolerance"],
            user_profile_dict["emotional_scent_preference"],
            user_profile_dict["scent_confidence"],
        ])

        # 3. 气味轨道 ID
        odor_track_id = torch.tensor(track_id, dtype=torch.long)

        # 4. 专家标签（来自 segment）
        expert_labels = self._parse_segment_scent(seg)

        if self.transform:
            visual, acoustic = self.transform(visual, acoustic)

        result = {
            "visual":       visual,
            "acoustic":     acoustic,
            "user_profile": user_profile,
            "odor_track_id": odor_track_id,
            "expert_labels": expert_labels,
            "video_file":   video_file,
        }
        if speech is not None:
            result["speech"] = speech
        return result

    def _get_feature(
        self, value: Union[str, np.ndarray, List[float], torch.Tensor], expected_dim: int
    ) -> torch.Tensor:
        if isinstance(value, str):
            arr = np.load(value)
        elif isinstance(value, np.ndarray):
            arr = value
        elif isinstance(value, (list, tuple)):
            arr = np.array(value, dtype=np.float32)
        elif isinstance(value, torch.Tensor):
            arr = value.detach().numpy()
        else:
            raise TypeError(f"不支持的特征类型: {type(value)}")
        arr = self._pad_or_truncate_feature(arr.astype(np.float32), expected_dim)
        return torch.from_numpy(arr)

    def _pad_or_truncate_feature(self, arr: np.ndarray, target_dim: int) -> np.ndarray:
        if arr.ndim > 1:
            arr = arr.flatten()
        if len(arr) < target_dim:
            arr = np.pad(arr, (0, target_dim - len(arr)), mode="constant", constant_values=0)
        return arr[:target_dim].astype(np.float32)

    def _parse_user_profile(self, profile: Dict) -> Dict[str, torch.Tensor]:
        """解析用户画像"""
        gender_map = {"男": 0, "女": 1, "male": 0, "female": 1}
        g = profile.get("gender", "男")
        gender = gender_map.get(g, 0) if isinstance(g, str) else int(g)

        # 缺失时使用量程中位数，避免归一化后超出 [0,1] 范围
        # age ∈ [0,100] 默认 25；[1,5] 量程字段默认 3（中位数）
        return {
            "gender": torch.tensor(gender, dtype=torch.long),
            "age": torch.tensor(float(profile.get("age", 25)), dtype=torch.float32),
            "scent_sensitivity": torch.tensor(
                float(profile.get("scent_sensitivity", 3)), dtype=torch.float32
            ),
            "unpleasant_tolerance": torch.tensor(
                float(profile.get("unpleasant_tolerance", 3)), dtype=torch.float32
            ),
            "emotional_scent_preference": torch.tensor(
                float(profile.get("emotional_scent_preference", 3)), dtype=torch.float32
            ),
            "scent_confidence": torch.tensor(
                float(profile.get("scent_confidence", 3)), dtype=torch.float32
            ),
        }

    def _parse_segment_scent(self, seg: Dict) -> Dict[str, torch.Tensor]:
        """解析片段中的气味专家标签"""
        scent = seg.get("scent", {})
        name = scent.get("name", "None")
        scent_id = self.scent_name_vocab.get(name, 0)

        form_str = scent.get("form", "气态")
        form_id = self.FORM_MAP.get(form_str, 0)
        temp_str = scent.get("temperature", "温和")
        temp_id = self.TEMP_MAP.get(temp_str, 0)

        return {
            "scent_name_id": torch.tensor(scent_id, dtype=torch.long),
            "start_time": torch.tensor(float(seg.get("start_time", 0)), dtype=torch.float32),
            "end_time": torch.tensor(float(seg.get("end_time", 0)), dtype=torch.float32),
            "intensity": torch.tensor(float(scent.get("intensity", 0)), dtype=torch.float32),
            "form": torch.tensor(form_id, dtype=torch.long),
            "temperature": torch.tensor(temp_id, dtype=torch.long),
        }


def collate_multimodal_batch(batch: List[Dict]) -> Dict[str, Union[torch.Tensor, Dict, List]]:
    """自定义 collate_fn"""
    visual        = torch.stack([b["visual"]        for b in batch])
    acoustic      = torch.stack([b["acoustic"]      for b in batch])
    odor_track_id = torch.stack([b["odor_track_id"] for b in batch])

    user_profile = {
        k: torch.stack([b["user_profile"][k] for b in batch])
        for k in batch[0]["user_profile"].keys()
    }
    expert_labels = {
        k: torch.stack([b["expert_labels"][k] for b in batch])
        for k in batch[0]["expert_labels"].keys()
    }

    video_files = [b.get("video_file", "") for b in batch]

    result = {
        "visual":        visual,
        "acoustic":      acoustic,
        "user_profile":  user_profile,
        "odor_track_id": odor_track_id,
        "expert_labels": expert_labels,
        "video_file":    video_files,
    }

    # 若全部样本都有 speech 则 stack；有部分缺失则对缺失样本补零，保持 batch 对齐
    has_speech = [b for b in batch if "speech" in b]
    if has_speech:
        speech_dim = has_speech[0]["speech"].shape[0]
        stacked = []
        for b in batch:
            stacked.append(b["speech"] if "speech" in b
                           else torch.zeros(speech_dim, dtype=torch.float32))
        result["speech"] = torch.stack(stacked)

    return result


# ============ Tiny synthetic example (no real data) ============
def create_example_data(path: str = "sample_odor_data.json"):
    """Create a tiny synthetic JSON matching the annotation schema."""
    samples = [
        {
            "video_file": "demo_clip.mp4",
            "clip_key": "demo_clip.mp4|0|0.0|2.0",
            "user_profile": {
                "gender": "男",
                "age": 27,
                "scent_sensitivity": 5,
                "unpleasant_tolerance": 2,
                "emotional_scent_preference": 2,
                "scent_confidence": 5,
            },
            "foreground_scent": [
                {
                    "start_time": 0.0,
                    "end_time": 2.0,
                    "scent": {
                        "name": "coffee",
                        "intensity": 3,
                        "form": "气态",
                        "temperature": "温和",
                    },
                }
            ],
            "scene_scent": [
                {
                    "start_time": 0.0,
                    "end_time": 2.0,
                    "scent": {
                        "name": "wood",
                        "intensity": 2,
                        "form": "气态",
                        "temperature": "温和",
                    },
                }
            ],
            "emotional": [
                {
                    "start_time": 0.0,
                    "end_time": 2.0,
                    "scent": {
                        "name": "fresh",
                        "intensity": 3,
                        "form": "气态",
                        "temperature": "温和",
                    },
                }
            ],
            # Inline features so the dataset runs without a feature_root
            "visual": np.random.randn(768).astype(np.float32).tolist(),
            "acoustic": np.random.randn(768).astype(np.float32).tolist(),
        },
    ]

    with open(path, "w", encoding="utf-8") as f:
        json.dump({"samples": samples}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    create_example_data()

    dataset = MultimodalOdorDataset("sample_odor_data.json", feature_root=None)

    print(f"样本数: {len(dataset)}")
    print("气味名称词汇表:", dataset.scent_name_vocab)

    sample = dataset[0]
    print("\n单条样本结构:")
    for k, v in sample.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for sk, sv in v.items():
                print(f"    {sk}: {sv.shape if hasattr(sv, 'shape') else sv}")
        elif isinstance(v, str):
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v.shape if hasattr(v, 'shape') else v}")

    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_multimodal_batch,
    )
    batch = next(iter(loader))
    print("\nBatch 结构:")
    print(f"  visual: {batch['visual'].shape}")
    print(f"  acoustic: {batch['acoustic'].shape}")
    print(f"  odor_track_id: {batch['odor_track_id'].shape}")

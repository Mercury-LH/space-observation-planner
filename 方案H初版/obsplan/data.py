"""读取赛题数据：TLE、站址、优先级清单，并做一致性检查。"""
from __future__ import annotations

import re
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Config

ALPHA5 = "ABCDEFGHJKLMNPQRSTUVWXYZ"  # Alpha-5 编号跳过 I 和 O


def norm_satnum(s: str) -> int:
    """TLE 编号 -> 整数。'08827' -> 8827，'A0045' -> 100045，'T0288' -> 270288。"""
    s = s.strip()
    if s and s[0].isalpha():
        return (ALPHA5.index(s[0].upper()) + 10) * 10000 + int(s[1:])
    return int(s)


def tle_checksum_ok(line: str) -> bool:
    total = sum(int(c) if c.isdigit() else (1 if c == "-" else 0) for c in line[:68])
    return line[68].isdigit() and total % 10 == int(line[68])


def _decode(raw: bytes) -> str:
    for enc in ("utf-8-sig", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "replace")


def _zip_texts(path: Path) -> dict[str, str]:
    """读 zip 里的文本文件，文件名按 GBK 解码（Windows 打包的中文名）。"""
    out = {}
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            name = info.filename
            if not info.flag_bits & 0x800:
                try:
                    name = name.encode("cp437").decode("gbk")
                except (UnicodeEncodeError, UnicodeDecodeError):
                    pass
            if name.endswith("/"):
                continue
            out[name.rsplit("/", 1)[-1]] = _decode(z.read(info))
    return out


def _collect_texts(data_dir: str) -> dict[str, str]:
    """收集 data_dir 下所有 txt（含 zip 内的），按文件名索引；同名时解压出的 txt 优先。"""
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"找不到数据目录 {root}，请按 README 放置赛题数据")
    texts: dict[str, str] = {}
    for zp in sorted(root.rglob("*.zip")):
        texts.update(_zip_texts(zp))
    for tp in sorted(root.rglob("*.txt")):
        texts[unicodedata.normalize("NFC", tp.name)] = _decode(tp.read_bytes())
    return {unicodedata.normalize("NFC", k): v for k, v in texts.items()}


@dataclass
class Dataset:
    names: list[str]
    satnum: np.ndarray        # (N,) int，规范化编号
    line1: list[str]
    line2: list[str]
    priority: np.ndarray      # (N,) float，优先级基础分
    priority_label: list[str]
    st_lon: np.ndarray        # (S,) 度，已归一到 [-180, 180)
    st_lat: np.ndarray
    st_alt_m: np.ndarray
    issues: list[str]

    @property
    def n_targets(self) -> int:
        return len(self.satnum)

    @property
    def n_stations(self) -> int:
        return len(self.st_lon)


def load(cfg: Config) -> Dataset:
    issues: list[str] = []
    files = _collect_texts(cfg.data_dir)
    try:
        tle_text = next(v for k, v in files.items() if "轨道" in k)
        st_text = next(v for k, v in files.items() if "站" in k)
    except StopIteration:
        raise FileNotFoundError(f"{cfg.data_dir} 中缺少 轨道*.txt 或 站址*.txt（或包含它们的 zip）") from None

    lines = [l.rstrip() for l in tle_text.splitlines() if l.strip()]
    names, l1s, l2s, nums = [], [], [], []
    i = 0
    while i < len(lines):
        if lines[i].startswith("1 ") and i + 1 < len(lines) and lines[i + 1].startswith("2 "):
            name, l1, l2 = "", lines[i], lines[i + 1]
            i += 2
        else:
            name, l1, l2 = lines[i][2:].strip() if lines[i].startswith("0 ") else lines[i].strip(), lines[i + 1], lines[i + 2]
            i += 3
        if not (tle_checksum_ok(l1) and tle_checksum_ok(l2)):
            issues.append(f"TLE 校验位错误: {l1[2:7]}")
        names.append(name)
        l1s.append(l1)
        l2s.append(l2)
        nums.append(norm_satnum(l1[2:7]))
    satnum = np.array(nums, dtype=np.int64)
    if len(set(nums)) != len(nums):
        issues.append(f"重复编号 {len(nums) - len(set(nums))} 个")

    st = []
    for line in st_text.splitlines():
        m = re.match(r"\s*(\d+)\s*[:：]\s*([-\d.]+)\s*[,，]\s*([-\d.]+)\s*[,，]\s*([-\d.]+)", line)
        if m:
            st.append([float(m.group(2)), float(m.group(3)), float(m.group(4))])
    st = np.array(st)
    lon = (st[:, 0] + 180.0) % 360.0 - 180.0

    pri_files = {k: v for k, v in files.items() if k[:1] in cfg.priority_score and "优" in k}
    if not pri_files:
        raise FileNotFoundError(f"{cfg.data_dir} 中缺少 高/中/低优先级目标 txt（或包含它们的 zip）")
    pri_of: dict[int, str] = {}
    for fname, text in pri_files.items():
        label = fname[:1]
        for tok in re.findall(r"[A-Za-z]?\d+", text):
            n = norm_satnum(tok)
            if n in pri_of and pri_of[n] != label:
                issues.append(f"目标 {n} 同时属于 {pri_of[n]} 和 {label}")
            pri_of[n] = label
    missing = [n for n in nums if n not in pri_of]
    extra = set(pri_of) - set(nums)
    if missing:
        issues.append(f"{len(missing)} 颗目标没有优先级，按最低处理")
    if extra:
        issues.append(f"优先级清单中有 {len(extra)} 个编号不在轨道数据里")
    lowest = min(cfg.priority_score, key=cfg.priority_score.get)
    labels = [pri_of.get(n, lowest) for n in nums]
    priority = np.array([cfg.priority_score[l] for l in labels])

    return Dataset(names, satnum, l1s, l2s, priority, labels, lon, st[:, 1], st[:, 2], issues)

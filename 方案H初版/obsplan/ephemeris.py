"""DE421 星历：Skyfield 逐秒校验（站点太阳角、目标受照）和月光规避要用，缓存在 cache/de421.bsp。

缓存里没有时下载一次并核对 SHA-256：先从 JPL 官网取，取不到再从 PyPI 的 skyfield-data 包里取
（包内的 de421.bsp 与 JPL 原文件逐字节相同）。之后完全离线；没有网络的机器手动放好这个文件即可。
"""
from __future__ import annotations

import hashlib
import io
import os
import urllib.request
import zipfile
from pathlib import Path

DE421_SHA256 = "a20a7139da04cbc462454634918e9a9ca69127044e2cc9d4f9c16e238d2deedc"
JPL_URL = "https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/de421.bsp"
PYPI_URL = ("https://files.pythonhosted.org/packages/bb/46/e815e37cfb1f1108d80e45ee100c8a6d9f1d5999ccf1cf14dab7be6858dc/"
            "skyfield_data-7.0.0-py2.py3-none-any.whl")


def ensure_de421(cache_dir: str | Path) -> Path:
    path = Path(cache_dir) / "de421.bsp"
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print("首次使用：下载 DE421 星历（约 17 MB，只需一次）……", flush=True)
    errors = []
    for name, url, member in (("JPL", JPL_URL, None), ("PyPI skyfield-data", PYPI_URL, "skyfield_data/data/de421.bsp")):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
            if member:
                data = zipfile.ZipFile(io.BytesIO(data)).read(member)
            if hashlib.sha256(data).hexdigest() != DE421_SHA256:
                raise ValueError("文件校验和不符")
        except Exception as e:  # 网络、证书、代理等任何错误都换下一个来源
            errors.append(f"{name}：{e}")
            continue
        part = path.with_name(path.name + ".part")
        part.write_bytes(data)
        os.replace(part, path)
        return path
    raise RuntimeError("下载 DE421 星历失败（" + "；".join(errors) + "）。请在能上网的电脑上下载 " + JPL_URL
                       + f"（约 17 MB），放到 {path}")


def load_de421(cache_dir: str | Path):
    """(时间尺度, DE421 星历)。时间尺度用 Skyfield 自带的数据，不联网。"""
    from skyfield.api import Loader
    ensure_de421(cache_dir)
    load = Loader(str(cache_dir), verbose=False)
    return load.timescale(builtin=True), load("de421.bsp")

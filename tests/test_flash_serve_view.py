"""Serve view assembly on a synthetic pack: pack files linked, head group replaced by BF16."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from safetensors.numpy import load_file, save_file                       # noqa: E402
from serve_view import serve_view                                        # noqa: E402
from tensorfold.families.glm5_next.cuda.split import RankReader          # noqa: E402

L = "model.language_model.layers.0"


@pytest.fixture()
def pack_with_head(tmp_path):
    pack, source = tmp_path / "pack", tmp_path / "source"
    pack.mkdir(), source.mkdir()
    save_file({f"{L}.input_layernorm.weight": np.arange(16, dtype=np.uint16),
               "lm_head.trellis": np.arange(64, dtype=np.uint32).reshape(1, 1, 64),
               "lm_head.suh": np.arange(4, dtype=np.uint16),
               "lm_head.svh": np.arange(4, dtype=np.uint16),
               "lm_head.mul1": np.arange(2, dtype=np.uint8)}, pack / "shard.safetensors")
    (pack / "config.json").write_text("{}")
    (pack / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        f"{L}.input_layernorm.weight": "shard.safetensors", "lm_head.trellis": "shard.safetensors",
        "lm_head.suh": "shard.safetensors", "lm_head.svh": "shard.safetensors",
        "lm_head.mul1": "shard.safetensors"}}))
    head = np.arange(4 * 8, dtype=np.uint16).reshape(4, 8)               # the "BF16" source head
    save_file({"lm_head.weight": head}, source / "model-00001-of-00001.safetensors")
    return pack, source / "model-00001-of-00001.safetensors", head


def test_serve_view_replaces_head(pack_with_head, tmp_path):
    pack, head_file, head = pack_with_head
    out = tmp_path / "view"
    serve_view(pack, head_file, out)
    index = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
    assert "lm_head.weight" in index and not any(n.startswith("lm_head.") and n != "lm_head.weight" for n in index)
    assert index["lm_head.weight"] == "head-source.safetensors"
    assert index[f"{L}.input_layernorm.weight"] == "shard.safetensors"
    assert (out / "shard.safetensors").is_symlink()                      # pack bytes never copied
    assert (out / "head-source.safetensors").is_symlink()                # the BF16 shard linked, not copied
    assert load_file(str(out / "head-source.safetensors"))["lm_head.weight"].tobytes() == head.tobytes()


def test_serve_view_readable_by_rankreader(pack_with_head, tmp_path):
    pack, head_file, head = pack_with_head
    out = tmp_path / "view"
    serve_view(pack, head_file, out)
    reader = RankReader(out, 0)
    w = reader.get("lm_head.weight")
    assert tuple(w.shape) == (4, 8) and int(w.view(-1)[0]) == 0
    assert reader.get(f"{L}.input_layernorm.weight").shape[0] == 16
    reader.close()


def test_serve_view_needs_index(tmp_path):
    pack, out = tmp_path / "pack", tmp_path / "view"
    pack.mkdir()
    with pytest.raises(FileNotFoundError):
        serve_view(pack, pack / "x.safetensors", out)


def test_serve_view_refuses_wrong_head_hash(pack_with_head, tmp_path):
    import hashlib
    pack, head_file, head = pack_with_head
    with pytest.raises(ValueError):
        serve_view(pack, head_file, tmp_path / "v1", expect_sha256="0" * 64)
    serve_view(pack, head_file, tmp_path / "v2",
               expect_sha256=hashlib.sha256(head_file.read_bytes()).hexdigest())

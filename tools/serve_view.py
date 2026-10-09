"""Assemble the serve view: a pack directory the engine reads, with the BF16 head in place.

RED-SNOW ships an 8-bit ``lm_head`` group (trellis/suh/svh/mul1) the fork never loads;
weights.load wants a plain ``lm_head.weight``. The head's provenance is the pack's own
BF16 source (Blackfrost-AI/RED-SNOW-5.3-FLASH-BF16), so the serve view links every pack
file, drops the pack's head group from the index, and adds ``lm_head.weight`` from the
BF16 source. The head must be the exact revision the pack quantized — pin ``HEAD_REV``
and pass ``--expect-head-sha256`` (from the pinned revision's HF file metadata) so a
wrong or stale head file fails before it can reach the engine.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def serve_view(pack: Path, head_file: Path, out: Path, expect_sha256: str | None = None) -> None:
    """Link the pack's files into ``out``, write the index with the BF16 head in place."""

    out.mkdir(parents=True, exist_ok=True)
    if expect_sha256:
        digest = hashlib.sha256(head_file.read_bytes()).hexdigest()
        if digest != expect_sha256:
            raise ValueError(f"head sha256 {digest} != pinned {expect_sha256}: wrong revision or stale file")
    index_file = pack / "model.safetensors.index.json"
    if not index_file.exists():
        raise FileNotFoundError(f"{index_file} missing: the serve view needs the pack's index")
    weight_map = json.loads(index_file.read_text())["weight_map"]
    head_parts = [n for n in weight_map if n.startswith("lm_head.")]
    if not head_parts:
        raise ValueError(f"{pack} has no lm_head group: is this the pack or already a serve view?")
    for rel in sorted(set(weight_map.values())):
        src, dst = pack / rel, out / rel
        if not dst.exists():
            dst.symlink_to(src.resolve())
    for name in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                 "chat_template.jinja"):
        if (pack / name).exists() and not (out / name).exists():
            (out / name).symlink_to((pack / name).resolve())
    head_out = out / "lm_head.safetensors"
    if head_file.is_dir():
        raise IsADirectoryError(f"{head_file}: pass the BF16 file holding lm_head.weight, not a repo")
    if not head_out.exists():
        if head_out.parent.samefile(head_file.parent):
            head_out.symlink_to(head_file.resolve())
        else:                                   # cheap enough once (1.2 GiB) and safe against source edits
            shutil.copyfile(head_file, head_out)
    kept = {n: f for n, f in weight_map.items() if n not in head_parts}
    kept["lm_head.weight"] = "lm_head.safetensors"
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": kept, "metadata": {"format": "pt", "head_source": str(head_file)}}))
    print(f"serve view: {out} — {len(kept)} tensors, head {head_file.name} "
          f"({len(head_parts)} pack head parts dropped)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", required=True, type=Path, help="the RED-SNOW pack directory")
    ap.add_argument("--head", required=True, type=Path, help="the BF16 source's lm_head*.safetensors file")
    ap.add_argument("--out", required=True, type=Path, help="the serve view directory to build")
    ap.add_argument("--expect-head-sha256", help="the pinned head file digest; refuse a wrong revision")
    args = ap.parse_args()
    serve_view(args.pack, args.head, args.out, args.expect_head_sha256)


if __name__ == "__main__":
    main()

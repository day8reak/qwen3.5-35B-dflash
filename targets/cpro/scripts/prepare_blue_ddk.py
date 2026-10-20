#!/usr/bin/env python3
"""Download the pinned official blue DDK into a run, without editing shared profiles."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import urllib.request
import zipfile
PAGE = "https://developer.huawei.com/consumer/cn/doc/doccenter-capabilities/cannkit-preparations.md"
PACKAGES = {
    "DDK-tools-next-6.1.1.0.zip": "87d7e3f186ad5c527a9385cea555559ea53c63b87dc483820523bcf7bf6f87e5",
    "kirinx90-plugin-next-6.1.1.0.zip": "0657efdddd2267949e83af2a382603b523d30b258d72e077a6975eb87d4f10b1",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--package-dir", help="Use already downloaded ZIPs, still verify pinned official SHA256")
    args = parser.parse_args()
    if not os.environ.get("AI_RUN_DIR"): raise ValueError("AI_RUN_DIR required")
    run, root = Path(os.environ["AI_RUN_DIR"]).resolve(), Path(args.output).resolve()
    if run not in root.parents: raise ValueError("output must be below AI_RUN_DIR")
    root.mkdir(parents=True, exist_ok=False)
    markdown = None
    if not args.package_dir:
        with urllib.request.urlopen(PAGE, timeout=60) as response:
            markdown = response.read().decode()
        (root/"official-source.md").write_text(markdown)
    for name, expected in PACKAGES.items():
        digest = hashlib.sha256()
        if args.package_dir:
            archive = Path(args.package_dir).resolve()/name
            with archive.open("rb") as source:
                while block := source.read(8*1024*1024): digest.update(block)
        else:
            matches = re.findall(r"https://[^\s)]+/"+re.escape(name)+r"\?[^\s)]+", markdown)
            if len(set(matches)) != 1: raise RuntimeError("official pinned link absent or ambiguous: "+name)
            archive = root/name
            with urllib.request.urlopen(matches[0], timeout=60) as response, archive.open("xb") as output:
                while block := response.read(8*1024*1024): output.write(block); digest.update(block)
        if digest.hexdigest() != expected: raise RuntimeError("official package SHA256 mismatch: "+name)
        destination = root/("tools" if name.startswith("DDK") else "plugin")
        destination.mkdir()
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                item = (destination/member.filename).resolve()
                if destination not in item.parents and item != destination:
                    raise ValueError("unsafe archive path")
        subprocess.run(["unzip", "-q", str(archive), "-d", str(destination)], check=True, cwd=root)
    ddk = root/"tools/DDK-tools-next-6.1.1.0"
    plugin = root/"plugin/image/ddk_platform_plugin/kirinx90"
    (ddk/"tools/platform/kirinx90").symlink_to(plugin, target_is_directory=True)
    binary = ddk/"tools/tools_omg/master/omg"
    binary.chmod(binary.stat().st_mode | 0o100)
    (root/"resources.json").write_text(json.dumps({"schema_version": 1,
        "ddk_root": str(ddk), "ddk_id": "blue-next-6.1.1.0-kirinx90",
        "hiai_include_dir": str(ddk/"tools/tools_ascendc/include/ddk"),
        "ohos_native_sdk": "${OHOS_NATIVE_SDK}", "hiai_libraries": [],
        "target_weights": "${QWEN35_TARGET_WEIGHTS}", "draft_weights": "${QWEN35_DRAFT_WEIGHTS}",
        "source_packages": PACKAGES}, indent=2)+"\n")
    print(root/"resources.json")


if __name__ == "__main__": main()

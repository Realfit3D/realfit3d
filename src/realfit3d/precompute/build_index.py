#!/usr/bin/env python3
"""Scan the video library and (re)write <library_root>/manifest.json.

  realfit3d --config paths.json library index --execute

The manifest is only an index built from the per-video sidecar files -- it is never the source of
truth and can be rebuilt any time. Only finished videos (sidecar status=done, mp4 present) are listed.

manifest.json:
  {
    "generated_at": "...",
    "count": <number of videos>,
    "options": {                       # what a UI can offer, per outfit and gender
      "<outfit>": {"<gender>": {"heights": [...], "weights": [...], "motions": [...],
                                    "opacities": [...], "opacity_modes": [...]}}
    },
    "videos": [                        # one entry per video; "path" is relative to the library root
      {"path": "...", "garments": [...], "gender": "...", "height": 170, "weight": 60,
       "motion": "01_01", "frames": 250, "fps": 25,
       "opacity": 1.0, "opacity_mode": "direct"}
    ]
  }
"""

import shutil
from datetime import datetime
from pathlib import Path

from realfit3d.precompute.toolkit_common import read_sidecar


def build(library: Path) -> dict:
    videos = []
    for sc_path in sorted(library.rglob("*.json")):
        if sc_path.name == "manifest.json":
            continue
        sc = read_sidecar(sc_path)
        mp4 = sc_path.with_suffix(".mp4")
        if not sc or sc.get("status") != "done" or not mp4.is_file():
            continue
        videos.append(dict(path=mp4.relative_to(library).as_posix(), garments=sc["garments"],
                           gender=sc["gender"], height=sc["height"], weight=sc["weight"],
                           motion=sc["motion"], frames=sc["frames"], fps=sc["fps"],
                           opacity=float(sc.get("opacity", 1.0)),
                           opacity_mode=sc.get("opacity_mode", "direct"),
                           opacity_garments=sc.get("opacity_garments", [])))
    # Prefer the opaque legacy entry when several modes have the same endpoint.
    videos.sort(key=lambda v: (
        "+".join(v["garments"]), v["gender"], v["height"], v["weight"], v["motion"],
        0 if (v["opacity"] == 1.0 and v["opacity_mode"] == "direct")
        else 1 if v["opacity"] == 1.0 else 2,
        -v["opacity"], v["opacity_mode"], tuple(v["opacity_garments"]), v["path"],
    ))
    options = {}
    for v in videos:
        o = options.setdefault("+".join(v["garments"]), {}).setdefault(
            v["gender"], dict(heights=set(), weights=set(), motions=set(),
                              opacities=set(), opacity_modes=set()))
        o["heights"].add(v["height"])
        o["weights"].add(v["weight"])
        o["motions"].add(v["motion"])
        o["opacities"].add(v["opacity"])
        o["opacity_modes"].add(v["opacity_mode"])
    for by_gender in options.values():
        for o in by_gender.values():
            for k in o:
                o[k] = sorted(o[k])
    return dict(generated_at=datetime.now().isoformat(timespec="seconds"), count=len(videos),
                options=options, videos=videos)


def publish_site_files(library: Path, data_root: Path):
    """The website reads catalog.json and thumbnails/ from the library folder, so copy them there
    from the data package (they are not produced by rendering videos)."""
    cat = data_root / "catalog.json"
    if cat.is_file():
        shutil.copyfile(cat, library / "catalog.json")
        print(f"已複製 catalog.json → {library / 'catalog.json'}")
    else:
        print(f"注意：找不到 {cat}，資料庫裡沒有 catalog.json，網頁會讀不到衣服與動作的名稱。")
    thumbs = data_root / "thumbnails"
    pngs = sorted(thumbs.glob("*.png")) if thumbs.is_dir() else []
    if pngs:
        (library / "thumbnails").mkdir(exist_ok=True)
        for f in pngs:
            shutil.copyfile(f, library / "thumbnails" / f.name)
        print(f"已複製 {len(pngs)} 張縮圖 → {library / 'thumbnails'}")
    else:
        print(f"注意：{thumbs} 沒有縮圖，網頁上的衣服會沒有圖片。")

"""Validate explicit waist support indices against the exact training template."""

import hashlib
import json
from pathlib import Path


def waist_indices(report_path, template_path):
    report = json.loads(Path(report_path).read_text())
    digest = hashlib.sha256(Path(template_path).read_bytes()).hexdigest()
    if report.get("outputs_sha256", {}).get("template_uv.obj") != digest:
        raise ValueError("Support report does not match template_uv.obj")
    ids = report.get("waist_indices")
    count = sum(
        line.startswith("v ") for line in Path(template_path).read_text().splitlines()
    )
    if not isinstance(ids, list) or len(ids) < 3 or len(set(ids)) != len(ids):
        raise ValueError("Missing or invalid waist support indices")
    if any(type(i) is not int or i < 0 or i >= count for i in ids):
        raise ValueError("Waist support index is outside the template")
    faces = [
        [int(p.split("/")[0]) - 1 for p in line.split()[1:]]
        for line in Path(template_path).read_text().splitlines()
        if line.startswith("f ")
    ]
    counts = {}
    for face in faces:
        if len(face) != 3:
            raise ValueError("Expected triangular template")
        for a, b in zip(face, face[1:] + face[:1]):
            edge = tuple(sorted((a, b)))
            counts[edge] = counts.get(edge, 0) + 1
    adjacency = {i: set() for i in ids}
    for (a, b), n in counts.items():
        if n == 1 and a in adjacency and b in adjacency:
            adjacency[a].add(b)
            adjacency[b].add(a)
    if any(len(neighbors) != 2 for neighbors in adjacency.values()):
        raise ValueError("Support indices must form one complete boundary loop")
    seen, stack = set(), [ids[0]]
    while stack:
        i = stack.pop()
        if i not in seen:
            seen.add(i)
            stack.extend(adjacency[i] - seen)
    if len(seen) != len(ids):
        raise ValueError("Support indices contain multiple disconnected loops")
    return ids

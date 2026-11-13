"""Check registration topology before conversion into simulation assets."""

from pathlib import Path


def topology(path):
    count = 0
    faces = []
    for line in Path(path).read_text().splitlines():
        values = line.split("#", 1)[0].split()
        if not values:
            continue
        if values[0] == "v":
            count += 1
        elif values[0] == "f":
            if len(values) != 4:
                raise ValueError(f"Expected triangles: {path}")
            ids = [int(value.split("/")[0]) for value in values[1:]]
            if any(i == 0 for i in ids):
                raise ValueError(f"Invalid OBJ index: {path}")
            faces.append(tuple(i - 1 if i > 0 else count + i for i in ids))
    if (
        not count
        or not faces
        or any(i < 0 or i >= count for face in faces for i in face)
    ):
        raise ValueError(f"Invalid topology: {path}")
    return count, faces


def validate_registered_topology(template, meshes):
    reference = topology(template)
    meshes = list(meshes)
    if not meshes:
        raise ValueError("No registered meshes")
    for path in meshes:
        if topology(path) != reference:
            raise ValueError(f"Registration changed vertex/face correspondence: {path}")

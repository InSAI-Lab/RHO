import json

scenes = [
    "MGL_final/MountVernon",
    "MGL_rainy/MountVernon",
    "MGL_night/MountVernon",
    "MGL_foggy/MountVernon",
    "MGL_snowy/MountVernon",
    "MGL_over_exposure/MountVernon",
    "MGL_under_exposure/MountVernon",
    "MGL_motion_blur/MountVernon",

]

merged_data = {"train": {}, "val": {}}

for scene in scenes:
    filename = f"datasets/splits_MGL_MountVernon.json"
    with open(filename, 'r') as f:
        data = json.load(f)
        merged_data["train"][scene] = data.get("train", [])
        merged_data["val"][scene] = data.get("val", [])

with open("datasets/splits_MGL_MountVernon_mixed.json", "w") as f:
    json.dump(merged_data, f, indent=4)

print("All scenes splits.json merged")
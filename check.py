from scipy.io import loadmat
import numpy as np
import os

# ==========================================================
# HICO-DET paths
# ==========================================================
HICO_ROOT = r"C:\Users\Foo Yu Jun\HICO-DET\hico_20160224_det"

IMAGES_DIR = os.path.join(HICO_ROOT, "images", "train2015")
ANNOTATION_FILE = os.path.join(HICO_ROOT, "anno_bbox.mat")

# ==========================================================
# Load annotation file
# ==========================================================
mat = loadmat(ANNOTATION_FILE)

bbox_train = mat["bbox_train"]
list_action = mat["list_action"]

# ==========================================================
# Select an image to inspect
# ==========================================================
idx = 0
target_filename = "HICO_train2015_00000149.jpg"
# sample = bbox_train[0, idx]
sample = None
for i in range(bbox_train.shape[1]):
    candidate = bbox_train[0, i]
    candidate_filename = candidate["filename"][0]

    if candidate_filename == target_filename:
        sample = candidate
        break

if sample is None:
    raise ValueError(f"Could not find {target_filename}")

filename = sample["filename"][0]
print(filename)
image_path = os.path.join(IMAGES_DIR, target_filename)

print("=" * 60)
print(f"Image: {filename}")
print(f"Path : {image_path}")
print("=" * 60)

# ==========================================================
# HOIs in this image
# ==========================================================
hois = sample["hoi"].ravel()

for i, hoi in enumerate(hois, start=1):

    # ------------------------------------------------------
    # Action information
    # ------------------------------------------------------
    action_id = int(hoi["id"][0, 0])

    action = list_action[action_id - 1][0]

    object_name = action["nname"][0]
    verb = action["vname"][0]
    verb_ing = action["vname_ing"][0]
    definition = action["def"][0]

    person_class = "person"
    object_class = object_name

    # ------------------------------------------------------
    # Visibility
    # ------------------------------------------------------
    invisible = int(hoi["invis"][0, 0])

    # ------------------------------------------------------
    # Human bounding box
    # ------------------------------------------------------
    human_box = hoi["bboxhuman"][0, 0]

    hx1 = int(human_box["x1"][0, 0])
    hx2 = int(human_box["x2"][0, 0])
    hy1 = int(human_box["y1"][0, 0])
    hy2 = int(human_box["y2"][0, 0])

    # ------------------------------------------------------
    # Object bounding box
    # ------------------------------------------------------
    object_box = hoi["bboxobject"][0, 0]

    ox1 = int(object_box["x1"][0, 0])
    ox2 = int(object_box["x2"][0, 0])
    oy1 = int(object_box["y1"][0, 0])
    oy2 = int(object_box["y2"][0, 0])

    # ------------------------------------------------------
    # Connection (human index, object index)
    # ------------------------------------------------------
    connection = hoi["connection"]

    # ------------------------------------------------------
    # Print results
    # ------------------------------------------------------
    print(f"\nHOI #{i}")
    print("-" * 50)
    print(f"Action ID      : {action_id}")
    print(f"Verb           : {verb}")
    print(f"Verb (ing)     : {verb_ing}")
    print(f"Object         : {object_name}")
    print(f"Interaction    : person {verb} {object_name}")
    print(f"Definition     : {definition}")
    print(f"Person Class   : {person_class}")
    print(f"Object Class   : {object_class}")
    print(f"Invisible      : {'Yes' if invisible else 'No'}")

    print("\nHuman Bounding Box")
    print(f"  x1={hx1}, y1={hy1}, x2={hx2}, y2={hy2}")

    print("\nObject Bounding Box")
    print(f"  x1={ox1}, y1={oy1}, x2={ox2}, y2={oy2}")

    print(f"\nConnection     : {connection}")

print("\nFinished.")
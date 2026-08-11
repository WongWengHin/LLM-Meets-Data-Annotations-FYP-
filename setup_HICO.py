import os
import fiftyone as fo
from scipy.io import loadmat
from HICO_sentences import HICO_SENTENCES_100 

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

HICO_ROOT = r"C:\Users\Foo Yu Jun\HICO-DET\hico_20160224_det"

IMAGES_DIR = os.path.join(HICO_ROOT, "images", "train2015")
ANNOTATION_FILE = os.path.join(HICO_ROOT, "anno_bbox.mat")

# ------------------------------------------------------------------
# Load annotations
# ------------------------------------------------------------------

print("Loading annotation file...")
mat = loadmat(ANNOTATION_FILE)

bbox_train = mat["bbox_train"]
list_action = mat["list_action"]

# ------------------------------------------------------------------
# Create FiftyOne dataset
# ------------------------------------------------------------------

dataset_name = "HICO-DET-train"

if dataset_name in fo.list_datasets():
    fo.delete_dataset(dataset_name)

dataset = fo.Dataset(dataset_name)

# ------------------------------------------------------------------
# Create sentence annotation lookup
# ------------------------------------------------------------------

sentence_lookup = {}

for img in HICO_SENTENCES_100["images"]:
    sentence_lookup[img["file_name"]] = img

manual_filenames = set(sentence_lookup.keys())

print(f"Manual annotations found: {len(manual_filenames)}")

# ------------------------------------------------------------------
# Iterate over every image
# ------------------------------------------------------------------

num_images = bbox_train.shape[1]

print(f"Found {num_images} training images")

imported_count = 0
for i in range(num_images):

    sample_struct = bbox_train[0, i]

    filename = sample_struct["filename"][0]

    # image_path = os.path.join(IMAGES_DIR, filename)

    # if not os.path.exists(image_path):
    #     print(f"Missing image: {filename}")
    #     continue
    # --------------------------------------------------------------
    # ONLY process your 100 manually annotated images
    # --------------------------------------------------------------

    # if filename not in manual_filenames:
    #     continue

    print(f"Processing {imported_count + 1}/100: {filename}")

    
    # ------------------------------------------------------------------
    # Create sentence annotation lookup
    # ------------------------------------------------------------------

    # sentence_lookup = {}

    # for img in HICO_SENTENCES_100["images"]:
    #     sentence_lookup[img["file_name"]] = img

    # sample = fo.Sample(filepath=image_path)
    # --------------------------------------------------------------
    # Image path
    # --------------------------------------------------------------

    image_path = os.path.join(IMAGES_DIR, filename)

    if not os.path.exists(image_path):
        print(f"Missing image: {filename}")
        continue


    sample = fo.Sample(filepath=image_path)

    # -----------------------
    # image size
    # -----------------------

    size = sample_struct["size"][0, 0]

    width = int(size["width"][0, 0])
    height = int(size["height"][0, 0])

    detections = []

    # -----------------------
    # HOIs
    # -----------------------

    hois = sample_struct["hoi"][0]
    
    # Store all predicates for this image
    predicates = []
    
    for hoi in hois:

        invisible = int(hoi["invis"][0, 0])

        if invisible == 1:
            continue

        action_id = int(hoi["id"][0, 0])

        # Get the action record
        action = list_action[action_id - 1, 0]

        # Extract the verb as a normal Python string
        predicate = str(action["vname"][0])

        # -----------------------
        # Human boxes
        # -----------------------

        human_boxes = hoi["bboxhuman"]

        for hb in human_boxes.flatten():

            xmin = float(hb[0])
            xmax = float(hb[1])
            ymin = float(hb[2])
            ymax = float(hb[3])

            detections.append(
                fo.Detection(
                    label="person",
                    bounding_box=[
                        xmin / width,
                        ymin / height,
                        (xmax - xmin) / width,
                        (ymax - ymin) / height,
                    ],
                    action_id=action_id,
                )
            )

        # -----------------------
        # Object boxes
        # -----------------------

        object_boxes = hoi["bboxobject"]

        for ob in object_boxes.flatten():

            xmin = float(ob[0])
            xmax = float(ob[1])
            ymin = float(ob[2])
            ymax = float(ob[3])

            extra_fields = {}

            if filename in sentence_lookup:
                # sentences = (
                #     sentence_lookup[filename]
                #     ["annotations"][0]
                #     ["sentences"]
                # )
                annotation = sentence_lookup[filename]["annotations"][0]
                sentences = annotation["sentences"]

                extra_fields = {
                    "sentence_level_1": sentences[0],
                    "sentence_level_2": sentences[1],
                    "sentence_level_3": sentences[2],
                    "sentence_level_4": sentences[3],
                    "sentence_level_5": sentences[4],
                    "sentence_level_6": sentences[5],
                    "sentence_level_7": sentences[6],
                    "predicate": annotation["predicate"],
                    "subject": annotation["subject"],
                    "object": annotation["object"],
                }

            detections.append(
                fo.Detection(
                    label="object",
                    bounding_box=[
                        xmin / width,
                        ymin / height,
                        (xmax - xmin) / width,
                        (ymax - ymin) / height,
                    ],
                    action_id=action_id,
                    **extra_fields
                )
            )
            
    # ------------------------------------------------------
    # Add custom annotations
    # ------------------------------------------------------

    custom_detections = []

    if filename in sentence_lookup:

        annotation = sentence_lookup[filename]["annotations"][0]

        sentences = annotation["sentences"]

        # --------------------------------------------------
        # Your manually annotated PERSON
        # --------------------------------------------------

        person_bbox = annotation["subject_bbox"]

        px = person_bbox[0]
        py = person_bbox[1]
        px2 = person_bbox[2]
        py2 = person_bbox[3]

        custom_detections.append(
            fo.Detection(
                label=annotation["subject"],
                bounding_box=[
                    px,
                    py,
                    px2 - px,
                    py2 - py,
                ],
                predicate=annotation["predicate"],
                annotation_type="custom",
                sentence_level_1=sentences[0],
                sentence_level_2=sentences[1],
                sentence_level_3=sentences[2],
                sentence_level_4=sentences[3],
                sentence_level_5=sentences[4],
                sentence_level_6=sentences[5],
                sentence_level_7=sentences[6],
                sentence_level_8=sentences[7],
            )
        )
        # # ------------------------------------------------------
        # # Save custom annotations
        # # ------------------------------------------------------

        # sample["custom_annotations"] = fo.Detections(
        #     detections=custom_detections
        # )
        
        # --------------------------------------------------
        # Your manually annotated OBJECT
        # --------------------------------------------------

        object_bbox = annotation["object_bbox"]

        ox = object_bbox[0]
        oy = object_bbox[1]
        ox2 = object_bbox[2]
        oy2 = object_bbox[3]

        custom_detections.append(
            fo.Detection(
                label=annotation["object"],
                bounding_box=[
                    ox,
                    oy,
                    ox2 - ox,
                    oy2 - oy,
                ],
                predicate=annotation["predicate"],
                annotation_type="custom",
                sentence_level_1=sentences[0],
                sentence_level_2=sentences[1],
                sentence_level_3=sentences[2],
                sentence_level_4=sentences[3],
                sentence_level_5=sentences[4],
                sentence_level_6=sentences[5],
                sentence_level_7=sentences[6],
                sentence_level_8=sentences[7],
            )
        )

    # ------------------------------------------------------
    # Add detections
    # ------------------------------------------------------    
    sample["predicates"] = predicates
    sample["ground_truth"] = fo.Detections(
        detections=detections
    )

    # ------------------------------------------------------
    # Save custom detections
    # ------------------------------------------------------

    sample["custom_annotations"] = fo.Detections(
        detections=custom_detections
    )

    # --------------------------------------------------------------
    # Add sample to dataset
    # --------------------------------------------------------------

    dataset.add_sample(sample)

    imported_count += 1

    print(f"Imported {imported_count}/100: {filename}")
    # --------------------------------------------------------------
    # Stop after 100 images
    # --------------------------------------------------------------

    if imported_count >= 100:
        break

    # dataset.add_sample(sample)

    # if (i + 1) % 500 == 0:
    #     print(f"Imported {i+1}/{num_images}")

print(dataset)
print(f"Successfully imported {imported_count} images.")

session = fo.launch_app(dataset, auto=True)
session.wait()
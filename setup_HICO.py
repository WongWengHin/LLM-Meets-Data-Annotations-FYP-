import os
import numpy as np
import fiftyone as fo
from scipy.io import loadmat

from HICO_sentences import HICO_SENTENCES_100


# ==================================================================
# Paths
# ==================================================================

HICO_ROOT = (
    r"C:\Users"
)

IMAGES_DIR = os.path.join(
    HICO_ROOT,
    "images",
    "train2015"
)

ANNOTATION_FILE = os.path.join(
    HICO_ROOT,
    "anno_bbox.mat"
)


# ==================================================================
# Helper: safely convert value to float
# ==================================================================

def safe_float(value, default=0.0):

    try:

        arr = np.asarray(value).flatten()

        if len(arr) == 0:
            return default

        return float(arr[0])

    except Exception:

        return default


# ==================================================================
# Helper: safely get sentence
# ==================================================================

def get_sentence(sentences, index):

    try:

        if sentences is None:
            return ""

        arr = np.asarray(
            sentences,
            dtype=object
        ).flatten()

        if index >= len(arr):
            return ""

        value = arr[index]

        if isinstance(value, np.ndarray):

            value = value.flatten()

            if len(value) == 0:
                return ""

            value = value[0]

        return str(value)

    except Exception:

        return ""


# ==================================================================
# Helper: extract HICO bounding boxes
#
# HICO MATLAB format can contain:
#
#     x1
#     x2
#     y1
#     y2
#
# We convert these to:
#
#     [x1, y1, x2, y2]
#
# later converted to FiftyOne:
#
#     [x, y, width, height]
# ==================================================================

def extract_bboxes(data):

    boxes = []

    if data is None:
        return boxes

    try:

        arr = np.asarray(data)

    except Exception:

        return boxes

    if arr.size == 0:
        return boxes


    # --------------------------------------------------------------
    # MATLAB structured bounding box
    # --------------------------------------------------------------

    if arr.dtype.names is not None:

        fields = set(arr.dtype.names)

        required_fields = {
            "x1",
            "x2",
            "y1",
            "y2"
        }

        if required_fields.issubset(fields):

            for item in arr.flat:

                x1 = safe_float(
                    item["x1"]
                )

                x2 = safe_float(
                    item["x2"]
                )

                y1 = safe_float(
                    item["y1"]
                )

                y2 = safe_float(
                    item["y2"]
                )

                boxes.append(
                    [
                        x1,
                        y1,
                        x2,
                        y2
                    ]
                )

            return boxes


    # --------------------------------------------------------------
    # Object arrays
    # --------------------------------------------------------------

    if arr.dtype == object:

        for item in arr.flat:

            boxes.extend(
                extract_bboxes(item)
            )

        return boxes


    # --------------------------------------------------------------
    # Numeric arrays
    # --------------------------------------------------------------

    try:

        arr = np.asarray(
            arr,
            dtype=float
        )

    except Exception:

        return boxes


    arr = np.squeeze(arr)


    # --------------------------------------------------------------
    # Single box
    # --------------------------------------------------------------

    if arr.ndim == 1:

        if arr.size >= 4:

            boxes.append(
                [
                    float(arr[0]),
                    float(arr[1]),
                    float(arr[2]),
                    float(arr[3])
                ]
            )

        return boxes


    # --------------------------------------------------------------
    # N x 4
    # --------------------------------------------------------------

    if arr.ndim == 2:

        if arr.shape[1] == 4:

            for row in arr:

                boxes.append(
                    [
                        float(row[0]),
                        float(row[1]),
                        float(row[2]),
                        float(row[3])
                    ]
                )

            return boxes


        # ----------------------------------------------------------
        # 4 x N
        # ----------------------------------------------------------

        if arr.shape[0] == 4:

            for row in arr.T:

                boxes.append(
                    [
                        float(row[0]),
                        float(row[1]),
                        float(row[2]),
                        float(row[3])
                    ]
                )

            return boxes


    return boxes


# ==================================================================
# Load HICO annotation file
# ==================================================================

print()
print("=" * 70)
print("LOADING HICO-DET")
print("=" * 70)

print()
print("Loading annotation file...")

mat = loadmat(
    ANNOTATION_FILE
)

bbox_train = mat["bbox_train"]

list_action = mat["list_action"]

print("Annotation file loaded.")


# ==================================================================
# Create FiftyOne dataset
# ==================================================================

dataset_name = "HICO-DET-train"

if dataset_name in fo.list_datasets():

    print()
    print(
        f"Deleting existing dataset: "
        f"{dataset_name}"
    )

    fo.delete_dataset(
        dataset_name
    )


dataset = fo.Dataset(
    dataset_name
)


# ==================================================================
# Create sentence lookup
# ==================================================================

sentence_lookup = {}

for img in HICO_SENTENCES_100["images"]:

    filename = img["file_name"]

    sentence_lookup[filename] = img


manual_filenames = set(
    sentence_lookup.keys()
)

print()
print(
    f"Manual annotations found: "
    f"{len(manual_filenames)}"
)


# ==================================================================
# Number of HICO training images
# ==================================================================

num_images = bbox_train.shape[1]

print(
    f"Found {num_images} training images"
)


# ==================================================================
# Process images
# ==================================================================

imported_count = 0

for i in range(num_images):

    # --------------------------------------------------------------
    # Get MATLAB sample
    # --------------------------------------------------------------

    sample_struct = bbox_train[0, i]


    # --------------------------------------------------------------
    # Filename
    # --------------------------------------------------------------

    filename = sample_struct[
        "filename"
    ][0]


    # --------------------------------------------------------------
    # Only process manually selected images
    # --------------------------------------------------------------

    if filename not in manual_filenames:
        continue


    print()
    print("-" * 70)

    print(
        f"Processing "
        f"{imported_count + 1}/"
        f"{len(manual_filenames)}: "
        f"{filename}"
    )


    # ==============================================================
    # Image path
    # ==============================================================

    image_path = os.path.join(
        IMAGES_DIR,
        filename
    )


    if not os.path.exists(
        image_path
    ):

        print(
            f"WARNING: Missing image: "
            f"{filename}"
        )

        continue


    # ==============================================================
    # Create FiftyOne sample
    # ==============================================================

    sample = fo.Sample(
        filepath=image_path
    )


    # ==============================================================
    # Image dimensions
    # ==============================================================

    size = sample_struct[
        "size"
    ][0, 0]


    width = int(
        safe_float(
            size["width"]
        )
    )

    height = int(
        safe_float(
            size["height"]
        )
    )


    print(
        f"Image size: "
        f"{width} x {height}"
    )


    # ==============================================================
    # HICO ground-truth detections
    # ==============================================================

    detections = []


    # ==============================================================
    # Get HOIs
    # ==============================================================

    try:

        hois = np.asarray(
            sample_struct["hoi"]
        ).squeeze()

    except Exception:

        hois = []


    if isinstance(
        hois,
        np.ndarray
    ):

        if hois.size == 0:

            hois = []

        elif hois.dtype == object:

            hois = list(
                hois.flat
            )

        else:

            hois = [hois]

    elif hois is None:

        hois = []


    # ==============================================================
    # Process each HOI
    # ==============================================================

    for hoi in hois:

        # ----------------------------------------------------------
        # Invisible
        # ----------------------------------------------------------

        try:

            invisible = int(
                safe_float(
                    hoi["invis"]
                )
            )

        except Exception:

            invisible = 0


        if invisible == 1:
            continue


        # ----------------------------------------------------------
        # Action ID
        # ----------------------------------------------------------

        try:

            action_id = int(
                safe_float(
                    hoi["id"]
                )
            )

        except Exception:

            action_id = -1


        # ==========================================================
        # HUMAN BOXES
        # ==========================================================

        human_boxes = extract_bboxes(
            hoi["bboxhuman"]
        )


        for hb in human_boxes:

            xmin = hb[0]
            ymin = hb[1]
            xmax = hb[2]
            ymax = hb[3]


            # ------------------------------------------------------
            # Clamp coordinates
            # ------------------------------------------------------

            xmin = max(
                0,
                min(xmin, width)
            )

            xmax = max(
                0,
                min(xmax, width)
            )

            ymin = max(
                0,
                min(ymin, height)
            )

            ymax = max(
                0,
                min(ymax, height)
            )


            # ------------------------------------------------------
            # Validate box
            # ------------------------------------------------------

            if xmax <= xmin:
                continue

            if ymax <= ymin:
                continue


            # ------------------------------------------------------
            # Convert HICO:
            #
            # [x1, y1, x2, y2]
            #
            # to FiftyOne:
            #
            # [x, y, width, height]
            #
            # normalized 0-1
            # ------------------------------------------------------

            fo_bbox = [

                xmin / width,

                ymin / height,

                (xmax - xmin) / width,

                (ymax - ymin) / height
            ]


            detections.append(

                fo.Detection(

                    label="person",

                    bounding_box=fo_bbox,

                    action_id=action_id
                )
            )


        # ==========================================================
        # OBJECT BOXES
        # ==========================================================

        object_boxes = extract_bboxes(
            hoi["bboxobject"]
        )


        for ob in object_boxes:

            xmin = ob[0]
            ymin = ob[1]
            xmax = ob[2]
            ymax = ob[3]


            # ------------------------------------------------------
            # Clamp coordinates
            # ------------------------------------------------------

            xmin = max(
                0,
                min(xmin, width)
            )

            xmax = max(
                0,
                min(xmax, width)
            )

            ymin = max(
                0,
                min(ymin, height)
            )

            ymax = max(
                0,
                min(ymax, height)
            )


            # ------------------------------------------------------
            # Validate
            # ------------------------------------------------------

            if xmax <= xmin:
                continue

            if ymax <= ymin:
                continue


            # ------------------------------------------------------
            # Sentence information
            # ------------------------------------------------------

            extra_fields = {}


            if filename in sentence_lookup:

                annotation = sentence_lookup[
                    filename
                ]["annotations"][0]


                sentences = annotation.get(
                    "sentences",
                    []
                )


                extra_fields = {

                    "sentence_level_1":
                        get_sentence(
                            sentences,
                            0
                        ),

                    "sentence_level_2":
                        get_sentence(
                            sentences,
                            1
                        ),

                    "sentence_level_3":
                        get_sentence(
                            sentences,
                            2
                        ),

                    "sentence_level_4":
                        get_sentence(
                            sentences,
                            3
                        ),

                    "sentence_level_5":
                        get_sentence(
                            sentences,
                            4
                        ),

                    "sentence_level_6":
                        get_sentence(
                            sentences,
                            5
                        ),

                    "sentence_level_7":
                        get_sentence(
                            sentences,
                            6
                        ),

                    "predicate":
                        annotation.get(
                            "predicate",
                            []
                        ),

                    "subject":
                        annotation.get(
                            "subject",
                            ""
                        ),

                    "object":
                        annotation.get(
                            "object",
                            ""
                        )
                }


            # ------------------------------------------------------
            # HICO object detection
            # ------------------------------------------------------

            fo_bbox = [

                xmin / width,

                ymin / height,

                (xmax - xmin) / width,

                (ymax - ymin) / height
            ]


            detections.append(

                fo.Detection(

                    label="object",

                    bounding_box=fo_bbox,

                    action_id=action_id,

                    **extra_fields
                )
            )


    # ==============================================================
    # CUSTOM ANNOTATIONS
    #
    # IMPORTANT:
    #
    # subject_bbox and object_bbox in
    # HICO_SENTENCES_100 are ALREADY:
    #
    # [x, y, width, height]
    #
    # normalized to 0-1.
    #
    # Therefore we DO NOT calculate:
    #
    # x2 - x1
    # y2 - y1
    #
    # =============================================================

    custom_detections = []


    if filename in sentence_lookup:

        annotation = sentence_lookup[
            filename
        ]["annotations"][0]


        sentences = annotation.get(
            "sentences",
            []
        )


        # ==========================================================
        # CUSTOM SUBJECT
        # ==========================================================

        subject_bbox = annotation.get(
            "subject_bbox"
        )


        if subject_bbox is not None:

            try:

                subject_bbox = np.asarray(
                    subject_bbox,
                    dtype=float
                ).flatten()


                if len(subject_bbox) != 4:

                    raise ValueError(
                        "subject_bbox must contain "
                        "exactly 4 values"
                    )


                # --------------------------------------------------
                # DIRECTLY USE:
                #
                # [x, y, width, height]
                # --------------------------------------------------

                sx = float(
                    subject_bbox[0]
                )

                sy = float(
                    subject_bbox[1]
                )

                sw = float(
                    subject_bbox[2]
                )

                sh = float(
                    subject_bbox[3]
                )


                custom_detections.append(

                    fo.Detection(

                        label=annotation.get(
                            "subject",
                            "person"
                        ),

                        bounding_box=[
                            sx,
                            sy,
                            sw,
                            sh
                        ],

                        predicate=annotation.get(
                            "predicate",
                            []
                        ),

                        annotation_type="custom",

                        sentence_level_1=
                            get_sentence(
                                sentences,
                                0
                            ),

                        sentence_level_2=
                            get_sentence(
                                sentences,
                                1
                            ),

                        sentence_level_3=
                            get_sentence(
                                sentences,
                                2
                            ),

                        sentence_level_4=
                            get_sentence(
                                sentences,
                                3
                            ),

                        sentence_level_5=
                            get_sentence(
                                sentences,
                                4
                            ),

                        sentence_level_6=
                            get_sentence(
                                sentences,
                                5
                            ),

                        sentence_level_7=
                            get_sentence(
                                sentences,
                                6
                            ),

                        sentence_level_8=
                            get_sentence(
                                sentences,
                                7
                            )
                    )
                )


                print(
                    "Custom subject bbox:"
                )

                print(
                    [
                        sx,
                        sy,
                        sw,
                        sh
                    ]
                )


            except Exception as e:

                print(
                    "WARNING: Could not process "
                    "custom subject bbox:"
                )

                print(e)


        # ==========================================================
        # CUSTOM OBJECT
        # ==========================================================

        object_bbox = annotation.get(
            "object_bbox"
        )


        if object_bbox is not None:

            try:

                object_bbox = np.asarray(
                    object_bbox,
                    dtype=float
                ).flatten()


                if len(object_bbox) != 4:

                    raise ValueError(
                        "object_bbox must contain "
                        "exactly 4 values"
                    )


                # --------------------------------------------------
                # IMPORTANT:
                #
                # object_bbox is already:
                #
                # [x, y, width, height]
                #
                # So use it directly.
                # --------------------------------------------------

                ox = float(
                    object_bbox[0]
                )

                oy = float(
                    object_bbox[1]
                )

                ow = float(
                    object_bbox[2]
                )

                oh = float(
                    object_bbox[3]
                )


                custom_detections.append(

                    fo.Detection(

                        label=annotation.get(
                            "object",
                            "object"
                        ),

                        bounding_box=[
                            ox,
                            oy,
                            ow,
                            oh
                        ],

                        predicate=annotation.get(
                            "predicate",
                            []
                        ),

                        annotation_type="custom",

                        sentence_level_1=
                            get_sentence(
                                sentences,
                                0
                            ),

                        sentence_level_2=
                            get_sentence(
                                sentences,
                                1
                            ),

                        sentence_level_3=
                            get_sentence(
                                sentences,
                                2
                            ),

                        sentence_level_4=
                            get_sentence(
                                sentences,
                                3
                            ),

                        sentence_level_5=
                            get_sentence(
                                sentences,
                                4
                            ),

                        sentence_level_6=
                            get_sentence(
                                sentences,
                                5
                            ),

                        sentence_level_7=
                            get_sentence(
                                sentences,
                                6
                            ),

                        sentence_level_8=
                            get_sentence(
                                sentences,
                                7
                            )
                    )
                )


                print(
                    "Custom object bbox:"
                )

                print(
                    [
                        ox,
                        oy,
                        ow,
                        oh
                    ]
                )


            except Exception as e:

                print(
                    "WARNING: Could not process "
                    "custom object bbox:"
                )

                print(e)


    # ==============================================================
    # Save HICO ground truth
    # ==============================================================

    sample["ground_truth"] = fo.Detections(
        detections=detections
    )


    # ==============================================================
    # Save custom annotations
    # ==============================================================

    sample["custom_annotations"] = fo.Detections(
        detections=custom_detections
    )


    # ==============================================================
    # Add sample to dataset
    # ==============================================================

    dataset.add_sample(
        sample
    )


    imported_count += 1


    print()
    print(
        f"Imported "
        f"{imported_count}/"
        f"{len(manual_filenames)}: "
        f"{filename}"
    )


    # ==============================================================
    # Stop after all manually annotated images
    # ==============================================================

    if imported_count >= len(manual_filenames):

        break


# ==================================================================
# Finished
# ==================================================================

print()
print("=" * 70)
print("IMPORT COMPLETE")
print("=" * 70)

print()

print(dataset)

print()

print(
    f"Successfully imported "
    f"{imported_count} images."
)


# ==================================================================
# Launch FiftyOne
# ==================================================================

print()
print("Launching FiftyOne...")

session = fo.launch_app(
    dataset,
    auto=True
)

session.wait()
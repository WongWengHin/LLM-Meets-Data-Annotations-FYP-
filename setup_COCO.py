import os

import fiftyone as fo
import fiftyone.zoo as foz
from COCO_categorisation import ANNOTATIONS_100

# 1. Load the pristine zoo dataset
print("Loading standard COCO 2017 validation split...")
dataset = foz.load_zoo_dataset(
    "coco-2017", 
    split="validation", 
    # max_samples=100,
    # dataset_name="coco-2017-train-100"
)

# 2. Build quick lookup maps from your COCO dictionary
category_map = {cat["id"]: cat["name"] for cat in ANNOTATIONS_100["categories"]}
image_id_to_filename = {img["id"]: img["file_name"] for img in ANNOTATIONS_100["images"]}

# Group your modified annotations by file_name
custom_annotations_by_file = {}
for ann in ANNOTATIONS_100["annotations"]:
    img_id = ann["image_id"]
    file_name = image_id_to_filename.get(img_id)
 #   print("img_id =", img_id)
 #   print("filename =", image_id_to_filename.get(img_id))
    if file_name:
        if file_name not in custom_annotations_by_file:
            custom_annotations_by_file[file_name] = []
        custom_annotations_by_file[file_name].append(ann)

#print(custom_annotations_by_file.keys())
#print("2149:", image_id_to_filename.get(2149))
#print("Number of custom files:", len(custom_annotations_by_file))
# 3. Apply your custom annotations to the FiftyOne dataset
print("Parsing and appending custom annotations from COCO_categorisation.py...")
matched_count = 0

with dataset.save_context() as context:
    for sample in dataset:
        # file_name = sample.filepath.split("/")[-1]
        file_name = os.path.basename(sample.filepath)
        
        # Check if we have modified annotations for this specific image
        if file_name in custom_annotations_by_file:
            matched_count += 1
            print(f"-> Successfully matching and updating: {file_name}")
            fo_detections = []
            
            for ann in custom_annotations_by_file[file_name]:
                bbox = ann["bbox"]  # [xmin, ymin, width, height]
                cat_id = ann["category_id"]
                label = category_map.get(cat_id, "unknown")
                
                # Ensure sample metadata is populated so we have dimensions
                if not sample.metadata:
                    sample.compute_metadata()
                
                # Safe extraction with defaults if metadata completely fails
                img_width = sample.metadata.width if sample.metadata else 640.0
                img_height = sample.metadata.height if sample.metadata else 480.0

                # Normalize bounding box coordinates (COCO absolute px -> FiftyOne relative 0-1)
                norm_bbox = [
                    bbox[0] / img_width,
                    bbox[1] / img_height,
                    bbox[2] / img_width,
                    bbox[3] / img_height
                ]
                
                fo_detections.append(
                    fo.Detection(label=label, bounding_box=norm_bbox)
                )
            
            # Save these under a distinct label field so you can see them clearly
            sample["my_custom_annotations"] = fo.Detections(detections=fo_detections)
            context.save(sample)

print(f"\nCustom annotations successfully applied to {matched_count} images!")

# 4. Launch the App
session = fo.launch_app(dataset, auto=True)
print("Session is active. Press Ctrl+C in this terminal to stop.")
session.wait()
# import fiftyone as fo
# import fiftyone.zoo as foz

# # Load the existing dataset
# dataset = foz.load_zoo_dataset(
#     "coco-2017", 
#     split="validation", 
#     # max_samples=100,
#     # dataset_name="coco-2017-train-100" # Explicitly name it to avoid confusion
# )
# # Launch the app and force it to open in your browser
# session = fo.launch_app(dataset, auto=True)

# # This blocks the script from finishing so the app stays active
# print("Session is active. Press Ctrl+C in this terminal to stop.")
# session.wait()
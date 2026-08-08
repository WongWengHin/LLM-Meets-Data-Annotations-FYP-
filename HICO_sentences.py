HICO_SENTENCES_100 = {
    "dataset_metadata": {
        "name": "HICO-DET dataset",
        "description": "100 manually annotated images with 7 sentences."
    },
    "images": [
        {
            "id": 2149,
            "file_name": "HICO_train2015_00000001.jpg",
            "annotations": [
                {
                    "subject": "person (ROY)",
                    "object": "motorcycle",
                    "subject_bbox": [0.124567, 0.233456, 0.456789, 0.891234], # [208, 33, 427, 300], # [0.124567, 0.233456, 0.456789, 0.891234],
                    "object_bbox": [0.401234, 0.552123, 0.612345, 0.954321], # [59, 98, 572, 405], # [0.401234, 0.552123, 0.612345, 0.954321],
                    # "interaction_bounding_box": [0.124567, 0.233456, 0.612345, 0.954321],
                    "predicate": "ride",
                    # Merged list of all interaction IDs occurring for this pair
                    # "action_ids": [5-7], 
                    
                    # The 7 sentences
                    "sentences": [
                        "A person is riding a motorcycle.",
                        "The person is wearing a colourful racing suit and a helmet.",
                        "The person is leaning forward while riding.",
                        "The motorcycle is a red, orange, white, and black racing motorcycle with orange wheels.",
                        "The person is leaning forward while riding the motorcycle.",
                        "The person in a colorful racing suit and full-face helmet is on the racing motorcycle.",
                        "The person is riding a red, orange, white, and black racing motorcycle.",
                        "The person in a colorful racing suit and full-face helmet is leaning forward while riding the red, orange, white, and black racing motorcycle."
                    ]
                }
            ]
        },
    ]
}
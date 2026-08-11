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
                    "subject": "person",
                    "object": "motorcycle",
                    "subject_bbox": [0.124567, 0.233456, 0.456789, 0.891234], # [208, 33, 427, 300], # [0.124567, 0.233456, 0.456789, 0.891234],
                    "object_bbox": [0.401234, 0.552123, 0.612345, 0.954321], # [59, 98, 572, 405], # [0.401234, 0.552123, 0.612345, 0.954321],
                    # "interaction_bounding_box": [0.124567, 0.233456, 0.612345, 0.954321],
                    "predicate": ["race", "ride", "sit_on", "straddle"],
                    # Merged list of all interaction IDs occurring for this pair
                    # "action_ids": [5-7], 
                    
                    # The sentences
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
        {
            # "id": 2149,
            "file_name": "HICO_train2015_00000008.jpg",
            "annotations": [
                {
                    "subject": "person",
                    "object": "toothbrush",
                    "subject_bbox": [0.00625, 0.00703, 0.93594, 0.99297],
                    "object_bbox": [0.02031, 0.67213, 0.34375, 0.99532],
                    "predicate": ["brush_with", "hold"],
                    
                    # The sentences
                    "sentences": [
                        "A person is brushing their teeth with a toothbrush.",
                        "The person has light skin and is shown close-up with their mouth open.",
                        "The person is brushing their teeth by moving a toothbrush inside their mouth.",
                        "The toothbrush has a red handle with white bristles.",
                        "The person is brushing their teeth with the toothbrush.",
                        "The person is holding a red toothbrush near their mouth.",
                        "The person is brushing their teeth with a red toothbrush.",
                        "The person with their mouth open is brushing their teeth with a red toothbrush."
                    ]
                }
            ]
        },
        {
            # "id": 2149,
            "file_name": "HICO_train2015_00000015.jpg",
            "annotations": [
                {
                    "subject": "person",
                    "object": "bicycle",
                    "subject_bbox": [0.363839, 0.339063, 0.640625, 0.756250],
                    "object_bbox": [0.180804, 0.548438, 0.888393, 0.912500],
                    "predicate": ["hold", "ride", "sit_on", "straddle"],
                    
                    # The sentences
                    "sentences": [
                        "A person is riding a bicycle.",
                        "The person is wearing a dark shirt, light-colored pants, and a cap.",
                        "The person is seated on the bicycle and holding the handlebars while riding.",
                        "The bicycle has a dark frame, two large wheels, and thin tires.",
                        "The person is riding the bicycle while seated and holding the handlebars.",
                        "The person wearing a dark shirt and cap is seated on the bicycle.",
                        "The person is riding a bicycle with a dark frame and large wheels.",
                        "The person wearing a dark shirt and cap is riding the bicycle while holding the handlebars."
                    ]
                }
            ]
        },
        {
            # "id": 2149,
            "file_name": "HICO_train2015_00000020.jpg",
            "annotations": [
                {
                    "subject": "person",
                    "object": "skateboard",
                    "subject_bbox": [0.685938, 0.354167, 0.865625, 0.918750],
                    "object_bbox": [0.729688, 0.860417, 0.879688, 0.966667],
                    "predicate": ["ride", "stand_on"],
                    
                    # The sentences
                    "sentences": [
                        "A person is riding a skateboard.",
                        "The person is shirtless and wearing dark sunglasses and striped shorts.",
                        "The person is standing on the skateboard while moving along the street.",
                        "The skateboard is dark-colored with visible wheels.",
                        "The shirtless person wearing dark sunglasses and striped shorts is riding the skateboard.",
                        "The person is riding a dark-colored skateboard.",
                        "The person is riding the dark-colored skateboard along the street.",
                        "The shirtless person wearing dark sunglasses and striped shorts is riding the dark-colored skateboard along the street."
                    ]
                }
            ]
        },
        {
            # "id": 2149,
            "file_name": "HICO_train2015_00000025.jpg",
            "annotations": [
                {
                    "subject": "person",
                    "object": "refrigerator",
                    "subject_bbox": [0.4279, 0.0257, 0.6997, 0.9159],
                    "object_bbox": [0.0016, 0.0023, 0.9984, 0.9977],
                    "predicate": ["open"],
                    
                    # The sentences
                    "sentences": [
                        "A person is looking at a bottle.",
                        "The person is wearing a dark hoodie and protective goggles on their head.",
                        "The person is looking toward the bottle inside the refrigerator.",
                        "The bottle is dark-colored with a green and white label.",
                        "The person wearing a dark hoodie and protective goggles is looking at the bottle.",
                        "The person is looking at a dark-colored bottle with a green and white label.",
                        "The person is looking toward the dark-colored bottle inside the refrigerator.",
                        "The person wearing a dark hoodie and protective goggles is looking at the dark-colored bottle with a green and white label."
                    ]
                }
            ]
        },
        {
            # "id": 2149,
            "file_name": "HICO_train2015_00000035.jpg",
            "annotations": [
                {
                    "subject": "person",
                    "object": "suitcase",
                    "subject_bbox": [0.425, 0.432, 0.903, 0.873],
                    "object_bbox": [0.219, 0.754, 0.425, 0.941],
                    "predicate": ["carry", "hold"],
                    
                    # The sentences
                    "sentences": [
                        "A person is pulling a suitcase.",
                        "The person is wearing a dark jacket and dark pants.",
                        "The person is pulling the suitcase by its extended handle while moving forward.",
                        "The suitcase is dark-colored and upright on wheels.",
                        "The person wearing a dark jacket and dark pants is pulling the suitcase.",
                        "The person is pulling a dark-colored suitcase.",
                        "The person is pulling the dark-colored suitcase by its handle.",
                        "The person wearing a dark jacket and dark pants is pulling the dark-colored suitcase by its handle."
                    ]
                }
            ]
        },
        {
            # "id": 2149,
            "file_name": "HICO_train2015_00000023.jpg",
            "annotations": [
                {
                    "subject": "person",
                    "object": "backpack",
                    "subject_bbox": [0.080, 0.172, 0.202, 0.851],
                    "object_bbox": [0.082, 0.110, 0.177, 0.551],
                    "predicate": ["carry", "wear"],
                    
                    # The sentences
                    "sentences": [
                        "A person is carrying another person.",
                        "The person in front is wearing a cap, short-sleeved shirt, shorts, and a backpack.",
                        "The person is carrying the other person on their back with their arms around them.",
                        "The carried person is partially visible behind the person in front and is wearing outdoor clothing.",
                        "The person in front is carrying the other person on their back.",
                        "The person in front is carrying another person who is positioned behind them.",
                        "The person is carrying another person on their back outdoors.",
                        "The person wearing a cap and backpack is carrying another person on their back outdoors."
                    ]
                }
            ]
        },
        {
            # "id": 2149,
            "file_name": "HICO_train2015_00000024.jpg",
            "annotations": [
                {
                    "subject": "person",
                    "object": "tennis_ball",
                    "subject_bbox": [0.082, 0.086, 0.198, 0.639],
                    "object_bbox": [0.085, 0.191, 0.108, 0.231],
                    "predicate": ["hold"],
                    
                    # The sentences
                    "sentences": [
                        "A person is holding a tennis ball.",
                        "The person is wearing sunglasses, a white shirt, and blue jeans.",
                        "The person is holding the tennis ball up in one hand.",
                        "The tennis ball is small, round, and yellow.",
                        "The person wearing sunglasses is holding the tennis ball in one hand.",
                        "The person is holding a small yellow tennis ball.",
                        "The person is holding the yellow tennis ball up in one hand.",
                        "The person wearing sunglasses and a white shirt is holding a small yellow tennis ball in one hand."
                    ]
                }
            ]
        },
        {
            # "id": 2149,
            "file_name": "HICO_train2015_00000028.jpg",
            "annotations": [
                {
                    "subject": "person",
                    "object": "snowboard",
                    "subject_bbox": [0.124, 0.137, 0.157, 0.253],
                    "object_bbox": [0.119, 0.230, 0.168, 0.257],
                    "predicate": ["jump","ride", "stand_on", "wear"],
                    
                    # The sentences
                    "sentences": [
                        "A person is riding skis.",
                        "The person is wearing a green jacket, dark pants, and a red helmet.",
                        "The person is riding down the snowy slope on skis.",
                        "The skis are long and narrow with a dark-colored surface.",
                        "The person wearing a green jacket and red helmet is riding the skis down the slope.",
                        "The person is riding a pair of long, narrow skis.",
                        "The person is riding the skis down the snowy slope.",
                        "The person wearing a green jacket and red helmet is riding a pair of long, narrow skis down the snowy slope."
                    ]
                }
            ]
        },
        {
            # "id": 2149,
            "file_name": "HICO_train2015_00000032.jpg",
            "annotations": [
                {
                    "subject": "person",
                    "object": "wine_glass",
                    "subject_bbox": [0.050, 0.409, 0.391, 0.571],
                    "object_bbox": [0.229, 0.001, 0.392, 0.504],
                    "predicate": ["hold"],
                    
                    # The sentences
                    "sentences": [
                        "A person is holding a wine glass.",
                        "The person has a visible hand with a light skin tone.",
                        "The person is holding the wine glass by its stem.",
                        "The wine glass is transparent and contains an amber-colored liquid.",
                        "The person is holding the wine glass by its stem.",
                        "The person is holding a transparent wine glass filled with an amber-colored liquid.",
                        "The person is holding the wine glass by its stem while it contains an amber-colored liquid.",
                        "The person is holding a transparent wine glass filled with an amber-colored liquid by its stem."    
                    ]
                }
            ]
        },
        {
            # "id": 2149,
            "file_name": "HICO_train2015_00000033.jpg",
            "annotations": [
                {
                    "subject": "person",
                    "object": "tie",
                    "subject_bbox": [0.019, 0.133, 0.365, 0.858],
                    "object_bbox": [0.175, 0.423, 0.211, 0.612],
                    "predicate": ["wear"],
                    
                    # The sentences
                    "sentences": [  
                        "A person is wearing a tie.",
                        "The person is wearing a dark suit, white shirt, and patterned tie.",
                        "The person is wearing the tie around his neck.",
                        "The tie is long and has a striped pattern.",
                        "The person wearing a dark suit is wearing the tie around his neck.",
                        "The person is wearing a long, striped tie.",
                        "The person is wearing the striped tie around his neck.",
                        "The person wearing a dark suit and white shirt is wearing a long, striped tie around his neck."
                    ]
                }
            ]
        },
        {
            # "id": 2149,
            "file_name": "HICO_train2015_00000043.jpg",
            "annotations": [
                {
                    "subject": "person",
                    "object": "bus",
                    "subject_bbox": [0.453, 0.406, 0.501, 0.502],
                    "object_bbox": [0.095, 0.053, 0.569, 0.723],
                    "predicate": ["drive"],
                    
                    # The sentences
                    "sentences": [  
                        "A person is driving a bus.",
                        "The person is seated in the driver's area and is wearing dark clothing.",
                        "The person is operating the bus from the driver's seat.",
                        "The bus is a large yellow and black double-decker vehicle.",
                        "The person is seated and driving the bus.",
                        "The person is inside the large yellow and black bus.",
                        "The person is driving the large yellow and black bus.",
                        "The person seated in the driver's area is driving the large yellow and black double-decker bus."   
                    ]
                }
            ]
        },
        {
            # "id": 2149,
            "file_name": "HICO_train2015_00000049.jpg",
            "annotations": [
                {
                    "subject": "person",
                    "object": "person",
                    "subject_bbox": [0.418, 0.227, 0.944, 0.949],
                    "object_bbox": [0.058, 0.011, 0.582, 0.947],
                    "predicate": ["hold"],
                    
                    # The sentences
                    "sentences": [
                        "A person is touching a baby.",
                        "The person has light-colored hair and is lying beside the baby.",
                        "The person is resting their head against the baby.",
                        "The baby is lying on its back and wearing a light-colored shirt.",
                        "The person is lying beside the baby and touching it.",
                        "The person is resting against the baby wearing a light-colored shirt.",
                        "The person is touching the baby while lying beside it.",
                        "The person with light-colored hair is resting their head against the baby wearing a light-colored shirt."
                    ]
                }
            ]
        },
        {
            # "id": 2149,
            "file_name": "HICO_train2015_00000052.jpg",
            "annotations": [
                {
                    "subject": "person",
                    "object": "baseball_bat",
                    "subject_bbox": [0.005, 0.117, 0.516, 0.998],
                    "object_bbox": [0.385, 0.154, 0.781, 0.686],
                    "predicate": ["hold", "wield"],
                    
                    # The sentences
                    "sentences": [
                        "A person is holding a baseball bat.",
                        "The person is wearing a black helmet and a white baseball uniform with the number 18.",
                        "The person is holding the baseball bat upright beside their body.",
                        "The baseball bat is a wooden bat with a dark barrel and light-colored handle.",
                        "The person wearing a black helmet is holding the baseball bat.",
                        "The person is holding a wooden baseball bat.",
                        "The person is holding the wooden baseball bat upright.",
                        "The person wearing a black helmet and white baseball uniform is holding the wooden baseball bat upright."
                    ]
                }
            ]
        },
        {
            # "id": 2149,
            "file_name": "HICO_train2015_00000058.jpg",
            "annotations": [
                {
                    "subject": "person",
                    "object": "baseball_bat",
                    "subject_bbox": [0.072, 0.259, 0.255, 0.644],
                    "object_bbox": [0.191, 0.191, 0.243, 0.415],
                    "predicate": ["hold", "wield"],
                    
                    # The sentences
                    "sentences": [
                        "A person is holding a baseball bat.",
                        "The person is wearing a white pinstriped baseball uniform, a blue helmet, and blue gloves.",
                        "The person is gripping the baseball bat with both hands while preparing to bat.",
                        "The baseball bat is light-colored and wooden.",
                        "The person in a white pinstriped uniform is gripping the baseball bat with both hands.",
                        "The person in a white pinstriped uniform is holding a light-colored wooden baseball bat.",
                        "The person is holding a light-colored wooden baseball bat.",
                        "The person in a white pinstriped uniform is gripping and holding the light-colored wooden baseball bat with both hands."
                    ]
                }
            ]
        },
        {
            # "id": 2149,
            "file_name": "HICO_train2015_00000067.jpg",
            "annotations": [
                {
                    "subject": "person",
                    "object": "skateboard",
                    "subject_bbox": [0.130, 0.358, 0.307, 0.840],
                    "object_bbox": [0.192, 0.782, 0.296, 0.842],
                    "predicate": ["ride", "stand_on"],
                    
                    # The sentences
                    "sentences": [
                        "A person is riding a skateboard.",
                        "The person is wearing a blue shirt and dark shorts.",
                        "The person is riding the skateboard while moving along the street.",
                        "The skateboard is small and dark-colored.",
                        "The person wearing a blue shirt and dark shorts is riding the skateboard.",
                        "The person is near a small dark-colored skateboard.",
                        "The person is riding a small dark-colored skateboard.",
                        "The person wearing a blue shirt and dark shorts is riding a small dark-colored skateboard."
                    ]
                }
            ]
        },
        {
            # "id": 2149,
            "file_name": "HICO_train2015_00000066.jpg",
            "annotations": [
                {
                    "subject": "person",
                    "object": "cake",
                    "subject_bbox": [0.259, 0.0, 0.736, 0.769],
                    "object_bbox": [0.327, 0.596, 0.638, 0.999],
                    "predicate": ["blow"],
                    
                    # The sentences
                    "sentences": [
                        "A person is blowing toward a cake.",
                        "The person is a young child with short brown hair wearing a red shirt.",
                        "The person is leaning toward the cake and blowing toward its candles.",
                        "The cake is round, covered with chocolate frosting and decorated with several lit candles and sprinkles.",
                        "The child is leaning forward and blowing toward the cake.",
                        "The child is near a round chocolate-frosted cake with candles.",
                        "The person is blowing toward the chocolate cake and its candles.",
                        "The young child in a red shirt is leaning forward and blowing toward the round chocolate-frosted cake with candles."
                    ]
                }
            ]
        },
        {
            # "id": 2149,
            "file_name": "HICO_train2015_00000064.jpg",
            "annotations": [
                {
                    "subject": "person",
                    "object": "umbrella",
                    "subject_bbox": [0.403, 0.159, 0.635, 0.924],
                    "object_bbox": [0.341, 0.117, 0.584, 0.302],
                    "predicate": ["carry", "hold", "stand_under"],
                    
                    # The sentences
                    "sentences": [
                        "A person is holding an umbrella.",
                        "The person is wearing dark clothing and walking in the rain.",
                        "The person is holding the umbrella above their head while walking.",
                        "The umbrella is large, dark-colored, and positioned above the person's head.",
                        "The person is walking while holding the umbrella.",
                        "The person is near and underneath a large dark umbrella.",
                        "The person is holding the large dark umbrella above their head.",
                        "The person wearing dark clothing is walking while holding a large dark umbrella above their head."
                    ]
                }
            ]
        },
        {
            # "id": 2149,
            "file_name": "HICO_train2015_00000068.jpg",
            "annotations": [
                {
                    "subject": "person",
                    "object": "skateboard",
                    "subject_bbox": [0.044, 0.224, 0.402, 0.855],
                    "object_bbox": [0.176, 0.640, 0.434, 0.851],
                    "predicate": ["ride", "stand_on"],
                    
                    # The sentences
                    "sentences": [
                        "A person is riding a skateboard.",
                        "The person is wearing a dark hoodie, light-colored pants, protective pads, gloves, and a yellow helmet.",
                        "The person is crouching on the skateboard with both feet positioned on it and arms extended for balance.",
                        "The skateboard is a narrow board with wheels underneath, visible beneath the person's feet.",
                        "The person is crouching and balancing while riding the skateboard.",
                        "The person is positioned above a skateboard at a skate park.",
                        "The person is riding the skateboard with both feet on the board.",
                        "The person wearing a dark hoodie and yellow helmet is crouching and balancing while riding the skateboard."
                    ]
                }
            ]
        },
        {
            # "id": 2149,
            "file_name": "HICO_train2015_00000074.jpg",
            "annotations": [
                {
                    "subject": "person",
                    "object": "book",
                    "subject_bbox": [0.206, 0.167, 0.58, 0.944],
                    "object_bbox": [0.316, 0.594, 0.384, 0.718],
                    "predicate": ["hold"],
                    
                    # The sentences
                    "sentences": [
                        "A person is holding a book.",
                        "The person has a beard and is wearing a light-colored turban, white shawl, and orange clothing.",
                        "The person is holding the book with both hands while seated.",
                        "The book has a red cover with visible yellow and white details.",
                        "The bearded person wearing a light-colored turban is holding the book with both hands.",
                        "The person is holding a red-covered book with yellow and white details.",
                        "The person is carefully holding the red-covered book.",
                        "The bearded person wearing a light-colored turban is holding a red-covered book with yellow and white details."
                    ]
                }
            ]
        },
        {
            # "id": 2149,
            "file_name": "HICO_train2015_00000076.jpg",
            "annotations": [
                {
                    "subject": "person",
                    "object": "sports_ball",
                    "subject_bbox": [0.380, 0.356, 0.477, 0.829],
                    "object_bbox": [0.391, 0.477, 0.432, 0.556],
                    "predicate": ["hold"],
                    
                    # The sentences
                    "sentences": [
                        "A person is holding a soccer ball.",
                        "The person has short blond hair and is wearing a red jersey, white shorts, and blue shoes.",
                        "The person is holding the soccer ball against his torso.",
                        "The soccer ball is round and has a black and white patterned surface.",
                        "The person wearing a red jersey is holding the soccer ball against his torso.",
                        "The person is holding a black and white soccer ball.",
                        "The person is holding the black and white soccer ball against his torso.",
                        "The person wearing a red jersey is holding a black and white soccer ball against his torso."
                    ]
                }
            ]
        },
        {
            # "id": 2149,
            "file_name": "HICO_train2015_00000079.jpg",
            "annotations": [
                {
                    "subject": "person",
                    "object": "frisbee",
                    "subject_bbox": [0.266, 0.073, 0.530, 0.933],
                    "object_bbox": [0.498, 0.321, 0.542, 0.427],
                    "predicate": ["hold"],
                    
                    # The sentences
                    "sentences": [
                        "A person is holding a leash.",
                        "The person has short blond hair and is wearing a purple top, dark skirt, and boots.",
                        "The person is holding the leash in one hand while walking.",
                        "The leash is a narrow strap extending from the person's hand toward the dog.",
                        "The person wearing a purple top is holding the leash while walking.",
                        "The person is holding a narrow leash attached to the dog.",
                        "The person is holding the leash while walking with the dog.",
                        "The person wearing a purple top is holding the narrow leash while walking with the dog."
                    ]
                }
            ]
        },
        {
            # "id": 2149,
            "file_name": "HICO_train2015_00000082.jpg",
            "annotations": [
                {
                    "subject": "person",
                    "object": "backpack",
                    "subject_bbox": [0.266, 0.073, 0.530, 0.933],
                    "object_bbox": [0.498, 0.321, 0.542, 0.427],
                    "predicate": ["carry", "wear"],
                    
                    # The sentences
                    "sentences": [
                        
                    ]
                }
            ]
        },
    ]
}
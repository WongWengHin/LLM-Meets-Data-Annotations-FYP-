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
                    "subject_bbox":[0.474,0.173,0.567,0.969],
                    "object_bbox":[0.523,0.365,0.606,0.620],
                    "predicate": ["carry", "wear"],
                    
                    # The sentences
                    "sentences": [
                        "A person is wearing a backpack.",
                        "The person is wearing a blue shirt, light-colored shorts, and dark shoes.",
                        "The person is wearing the backpack on their back while standing.",
                        "The backpack is black and has a visible water bottle attached to it.",
                        "The person wearing a blue shirt and light-colored shorts is wearing the backpack.",
                        "The person is wearing a black backpack.",
                        "The person is wearing a black backpack on their back.",
                        "The person wearing a blue shirt and light-colored shorts is wearing a black backpack on their back."
                    ]
                }
            ]
        },
        {
            # "id": 2149,
            "file_name": "HICO_train2015_00000091.jpg",
            "annotations": [
                {
                    "subject": "person",
                    "object": "surfboard",
                    "subject_bbox":[0.165,0.312,0.891,0.956],
                    "object_bbox":[0.660,0.780,0.888,0.930],
                    "predicate": ["lie_on", "ride"],
                    
                    # The sentences
                    "sentences": [
                        "A person is riding a surfboard.",
                        "The person has short dark hair and is shirtless.",
                        "The person is lying on the surfboard while moving through the water.",
                        "The surfboard is long, narrow, and dark-colored with a light stripe.",
                        "The shirtless person is lying on and riding the surfboard.",
                        "The person is lying on a dark-colored surfboard.",
                        "The person is riding a long, narrow surfboard through the water.",
                        "The shirtless person is riding a long, narrow dark-colored surfboard through the water."
                    ]
                }
            ]
        },
        {
            # "id": 2149,
            "file_name": "HICO_train2015_00000092.jpg",
            "annotations": [
                {
                    "subject": "person",
                    "object": "cell_phone",
                    "subject_bbox": [0.275, 0.000, 0.750, 0.994],
                    "object_bbox": [0.365, 0.289, 0.443, 0.502],
                    "predicate": ["carry", "hold", "talk_on"],
                    
                    # The sentences
                    "sentences": [
                        "A person is holding a cell phone.",
                        "The person is wearing a white jacket, black hat, and glasses.",
                        "The person is holding the cell phone against the side of their face.",
                        "The cell phone is a small black handheld device.",
                        "The person is holding and talking on the cell phone.",
                        "The person is carrying a cell phone in their hand.",
                        "The person is holding the cell phone while talking on it.",
                        "The person wearing a white jacket and black hat is holding and talking on a small black cell phone."
                    ]
                }
            ]
        },
        {
            # "id": 2149,
            "file_name": "HICO_train2015_00000097.jpg",
            "annotations": [
                {
                    "subject": "person",
                    "object": "bottle",
                    "subject_bbox": [0.147, 0.005, 0.597, 0.724],
                    "object_bbox": [0.435, 0.355, 0.605, 0.707],
                    "predicate": ["drink_with", "hold"],
                    
                    # The sentences
                    "sentences": [
                        "A person is drinking with a bottle.",
                        "The person is a small child wearing a light-colored patterned garment.",
                        "The child holds the bottle near the mouth while drinking from it.",
                        "The bottle is cylindrical and has bands or labels around its body.",
                        "The child is holding the bottle while drinking from it.",
                        "The child is near a cylindrical bottle with bands around it.",
                        "The child is drinking with and holding a cylindrical bottle.",
                        "The small child wearing a light-colored patterned garment is holding a cylindrical bottle while drinking from it."
                    ]
                }
            ]
        },
        {
            # "id": 2149,
            "file_name": "HICO_train2015_00000098.jpg",
            "annotations": [
                {
                    "subject": "person",
                    "object": "horse",
                    "subject_bbox": [0.117, 0.101, 0.305, 0.687],
                    "object_bbox": [0.029, 0.140, 0.275, 0.792],
                    "predicate": ["straddle"],
                    
                    # The sentences
                    "sentences": [
                        "A person is riding a horse.",
                        "The person is wearing a helmet and dark clothing.",
                        "The person is seated on the horse and holding the reins.",
                        "The horse is brown and is equipped with a bridle and saddle.",
                        "The person is riding the horse while seated in the saddle.",
                        "The person is seated on a brown horse with a bridle and saddle.",
                        "The person is riding a brown horse equipped with a bridle and saddle.",
                        "The person wearing a helmet and dark clothing is riding a brown horse while holding the reins."
                    ]
                }
            ]
        },
        {
            # "id": 2149,
            "file_name": "HICO_train2015_00000101.jpg",
            "annotations": [
                {
                    "subject": "person",
                    "object": "boat",
                    "subject_bbox": [0.611, 0.374, 0.649, 0.524],
                    "object_bbox": [0.320, 0.237, 0.716, 0.806],
                    "predicate": ["ride", "stand_on"],
                    
                    # The sentences
                    "sentences": [
                        "A person is driving a boat.",
                        "The person is standing on the boat and wearing light-colored clothing.",
                        "The person is operating the boat from the rear deck.",
                        "The boat is a white motorboat with a dark hull and a covered cabin.",
                        "The person is driving the boat while standing on its rear deck.",
                        "The person is on a white motorboat with a dark hull.",
                        "The person is driving a white motorboat across the water.",
                        "The person wearing light-colored clothing is driving the white motorboat with a dark hull across the water."
                    ]
                }
            ]
        },
        {
            # "id": 2149,
            "file_name": "HICO_train2015_00000103.jpg",
            "annotations": [
                {
                    "subject": "person",
                    "object": "couch",
                    "subject_bbox": [0.006, 0.402, 0.918, 0.970],
                    "object_bbox": [0.006, 0.455, 0.997, 0.998],
                    "predicate": ["lie_on"],
                    
                    # The sentences
                    "sentences": [
                        "A person is sitting on a chair.",
                        "The person is wearing a blue jacket, shorts, and sneakers.",
                        "The person is reclining on the chair with their legs raised.",
                        "The chair is a large dark-colored cushioned lounge chair.",
                        "The person is sitting and reclining on the chair.",
                        "The person is resting on a large dark-colored chair.",
                        "The person is reclining on a large dark-colored cushioned chair.",
                        "The person wearing a blue jacket, shorts, and sneakers is reclining on a large dark-colored cushioned chair."
                    ]
                }
            ]
        },
        {
            # "id": 2149,
            "file_name": "HICO_train2015_00000104.jpg",
            "annotations": [
                {
                    "subject": "person",
                    "object": "book",
                    "subject_bbox": [0.459, 0.278, 0.670, 0.559],
                    "object_bbox": [0.542, 0.426, 0.625, 0.497],
                    "predicate": ["open", "read"],
                    
                    # The sentences
                    "sentences": [
                        "A person is opening and reading a book.",
                        "The person is wearing a white shirt and a red and white cap.",
                        "The person is holding an open book while looking down at it.",
                        "The book is open with visible pages.",
                        "The person wearing a white shirt and a red and white cap is opening and reading the book.",
                        "The person is interacting with an open book.",
                        "The person is opening and reading an open book.",
                        "The person wearing a white shirt and a red and white cap is opening and reading an open book."
                    ]
                }
            ]
        },
        {
            # "id": 2149,
            "file_name": "HICO_train2015_00000107.jpg",
            "annotations": [
                {
                    "subject": "person",
                    "object": "snowboard",
                    "subject_bbox": [0.362, 0.177, 0.653, 0.788],
                    "object_bbox": [0.437, 0.457, 0.654, 0.788],
                    "predicate": ["grind", "hold", "stand_on", "wear"],
                    
                    # The sentences
                    "sentences": [
                        "A person is grinding on a snowboard.",
                        "The person is shirtless and wearing dark pants and protective gear.",
                        "The person is balancing on the snowboard while gripping it for support.",
                        "The snowboard is long and dark-colored with bindings attached.",
                        "The shirtless person wearing dark pants is grinding on the snowboard.",
                        "The person is positioned on a dark snowboard with bindings.",
                        "The person is grinding on and standing on the snowboard.",
                        "The shirtless person wearing dark pants is grinding on and standing on the snowboard while holding it."
                    ]
                }
            ]
        },
        {
            # "id": 2149,
            "file_name": "HICO_train2015_00000110.jpg",
            "annotations": [
                {
                    "subject": "person",
                    "object": "bus",
                    "subject_bbox": [0.416, 0.370, 0.516, 0.556],
                    "object_bbox": [0.159, 0.219, 0.742, 0.994],
                    "predicate": ["drive", "ride"],
                    
                    # The sentences
                    "sentences": [
                        "A person is driving and riding a bus.",
                        "The person is seated inside the front of the bus and is partially visible through the windshield.",
                        "The person is seated in the driver's area of the bus.",
                        "The bus is a large yellow school bus with black trim and multiple windows.",
                        "The person seated inside the bus is driving and riding the bus.",
                        "The person is inside the yellow school bus.",
                        "The person is driving and riding the yellow school bus.",
                        "The person seated inside the front of the bus is driving and riding the yellow school bus."
                    ]
                }
            ]
        },
        {
            # "id": 2149,
            "file_name": "HICO_train2015_00000112.jpg",
            "annotations": [
                {
                    "subject": "person",
                    "object": "book",
                    "subject_bbox": [0.129, 0.155, 0.414, 0.997],
                    "object_bbox": [0.247, 0.596, 0.438, 0.759],
                    "predicate": ["hold", "open", "read"],
                    
                    # The sentences
                    "sentences": [
                        "A person is holding, opening, and reading a book.",
                        "The person is an older man wearing glasses, a denim jacket, and a burgundy shirt.",
                        "The person is holding the open book with both hands and looking down at its pages.",
                        "The book is open with multiple visible pages and a light-colored cover.",
                        "The older man wearing glasses and a denim jacket is holding and reading the open book.",
                        "The person is holding an open book with visible pages.",
                        "The person is reading the open book while holding it in front of his body.",
                        "The older man wearing glasses and a denim jacket is holding and reading the open book with both hands."
                    ]
                }
            ]
        },
        {
            # "id": 2149,
            "file_name": "HICO_train2015_00000116.jpg",
            "annotations": [
                {
                    "subject": "person",
                    "object": "frisbee",
                    "subject_bbox": [0.466, 0.294, 0.846, 0.996],
                    "object_bbox": [0.447, 0.499, 0.558, 0.65],
                    "predicate": ["block", "hold", "throw"],
                    
                    # The sentences
                    "sentences": [
                        "A person is blocking, holding, and throwing a frisbee.",
                        "The person is a man wearing a green cap, a purple shirt, and black pants.",
                        "The person is crouching on the grass while reaching toward the frisbee.",
                        "The frisbee is a white flying disc visible in front of the person.",
                        "The man wearing a green cap and purple shirt is blocking, holding, and throwing the frisbee.",
                        "The person is interacting with a white frisbee in front of him.",
                        "The person is blocking and holding the white frisbee while throwing it.",
                        "The man wearing a green cap and purple shirt is blocking, holding, and throwing the white frisbee while crouching on the grass."
                    ]
                }
            ]
        },
        {
            # "id": 2149,
            "file_name": "HICO_train2015_00000117.jpg",
            "annotations": [
                {
                    "subject": "person",
                    "object": "backpack",
                    "subject_bbox": [0.364, 0.000, 0.800, 0.964],
                    "object_bbox": [0.417, 0.353, 0.730, 0.786],
                    "predicate": ["wear"],
                    
                    # The sentences
                    "sentences": [
                        "A person is wearing a backpack.",
                        "The person is a man with short dark hair wearing a backpack.",
                        "The person is wearing the backpack over his shoulders.",
                        "The backpack has dark straps and a blue section visible over the person's shoulders.",
                        "The man with short dark hair is wearing the backpack over his shoulders.",
                        "The person is wearing a backpack with dark straps and a blue section.",
                        "The person is wearing the backpack over his shoulders.",
                        "The man with short dark hair is wearing the backpack with dark straps and a blue section over his shoulders."
                    ]
                }
            ]
        },
        {
            # "id": 2149,
            "file_name": "HICO_train2015_00000119.jpg",
            "annotations": [
                {
                    "subject": "person",
                    "object": "baseball_glove",
                    "subject_bbox": [0.359, 0.270, 0.534, 1.000],
                    "object_bbox": [0.521, 0.164, 0.597, 0.325],
                    "predicate": ["hold", "wear"],
                    
                    # The sentences
                    "sentences": [
                        "A person is holding and wearing a baseball glove.",
                        "The person is wearing a striped shirt and blue shorts.",
                        "The person is extending one arm while holding the baseball glove.",
                        "The baseball glove is brown and black with a visible webbed pocket.",
                        "The person wearing a striped shirt is holding and wearing the baseball glove.",
                        "The person is near and in contact with the brown and black baseball glove.",
                        "The person is holding and wearing the brown and black baseball glove.",
                        "The person wearing a striped shirt and blue shorts is holding and wearing the brown and black baseball glove."                        
                    ]
                }
            ]
        },
        {
            # "id": 2149,
            "file_name": "HICO_train2015_00000121.jpg",
            "annotations": [
                {
                    "subject": "person",
                    "object": "cake",
                    "subject_bbox": [0.241, 0.000, 0.658, 0.992],
                    "object_bbox": [0.262, 0.781, 0.944, 1.000],
                    "predicate": ["blow"],
                    
                    # The sentences
                    "sentences": [
                        "A person is blowing on a cake.",
                        "The person has light brown hair and is wearing a blue shirt.",
                        "The person is leaning toward the cake and blowing toward its candles.",
                        "The cake is large, dark-colored, and topped with several colorful candles.",
                        "The person wearing a blue shirt is blowing on the cake.",
                        "The person is positioned directly behind the large cake.",
                        "The person is blowing on the large cake with colorful candles.",
                        "The person with light brown hair wearing a blue shirt is blowing on the large cake with colorful candles."              
                    ]
                }
            ]
        },
        {
            # "id": 2149,
            "file_name": "HICO_train2015_00000122.jpg",
            "annotations": [
                {
                    "subject": "person",
                    "object": "train",
                    "subject_bbox": [0.532, 0.219, 0.599, 0.352],
                    "object_bbox": [0.151, 0.000, 0.887, 0.905],
                    "predicate": ["drive", "ride", "sit_on"],
                    
                    # The sentences
                    "sentences": [
                        "A person is driving, riding, and sitting on a train.",
                        "The person is visible inside the train cabin.",
                        "The person is seated inside the locomotive cabin while operating the train.",
                        "The train is a large red and green locomotive with a long body and metal wheels.",
                        "The person inside the cabin is driving and riding the train.",
                        "The person is inside and seated on the train.",
                        "The person is driving and riding the large locomotive.",
                        "The person inside the cabin is driving, riding, and sitting on the large red and green train."
                    ]
                }
            ]
        },
        {
            # "id": 2149,
            "file_name": "HICO_train2015_00000126.jpg",
            "annotations": [
                {
                    "subject": "person",
                    "object": "sports_ball",
                    "subject_bbox": [0.389, 0.198, 0.674, 0.515],
                    "object_bbox": [0.399, 0.250, 0.478, 0.447],
                    "predicate": ["carry", "hold"],
                    
                    # The sentences
                    "sentences": [
                        "A person is carrying and holding a sports ball.",
                        "The person has short dark hair and is wearing a black and red sports jersey.",
                        "The person is carrying the sports ball while holding it up near his face with one hand.",
                        "The sports ball is blue and yellow with a patterned design.",
                        "The person wearing a black and red sports jersey is carrying and holding the sports ball near his face.",
                        "The person is carrying and holding a blue and yellow sports ball.",
                        "The person is carrying and holding the blue and yellow sports ball near his face.",
                        "The person with short dark hair is carrying and holding a blue and yellow sports ball near his face."
                    ]
                }
            ]
        },
        {
            # "id": 2149,
            "file_name": "HICO_train2015_00000127.jpg",
            "annotations": [
                {
                    "subject": "person",
                    "object": "snowboard",
                    "subject_bbox": [0.493, 0.226, 0.779, 0.905],
                    "object_bbox": [0.359, 0.842, 0.844, 0.968],
                    "predicate": ["ride", "stand_on", "wear"],
                    
                    # The sentences
                    "sentences": [
                        "A person is riding and standing on a snowboard.",
                        "The person is wearing a blue and black jacket, black pants, gloves, a helmet, and goggles.",
                        "The person is standing on the snowboard while riding across the snow.",
                        "The snowboard is long and predominantly red with a dark binding area.",
                        "The person wearing a blue and black jacket is riding and standing on the snowboard.",
                        "The person is standing on a red snowboard.",
                        "The person is riding and standing on the red snowboard.",
                        "The person wearing a blue and black jacket is riding and standing on the red snowboard."
                    ]
                }
            ]
        },
        {
            # "id": 2149,
            "file_name": "HICO_train2015_00000129.jpg",
            "annotations": [
                {
                    "subject": "person",
                    "object": "umbrella",
                    "subject_bbox": [0.578, 0.501, 0.623, 0.681],
                    "object_bbox": [0.544, 0.414, 0.595, 0.521],
                    "predicate": ["carry", "hold", "stand_under"],
                    
                    # The sentences
                    "sentences": [
                        "A person is carrying, holding, and standing under an umbrella.",
                        "The person is wearing dark clothing and appears as a dark silhouette.",
                        "The person is walking while carrying and holding the umbrella overhead.",
                        "The umbrella is positioned above the person and appears dark in the low-light scene.",
                        "The person is carrying and holding the umbrella while standing underneath it.",
                        "The person is positioned beneath a dark umbrella.",
                        "The person is carrying and holding the umbrella above them.",
                        "The person in dark clothing is carrying and holding a dark umbrella while standing underneath it."
                    ]
                }
            ]
        },
        {
            # "id": 2149,
            "file_name": "HICO_train2015_00000131.jpg",
            "annotations": [
                {
                    "subject": "person",
                    "object": "skateboard",
                    "subject_bbox": [0.323, 0.163, 0.601, 0.784],
                    "object_bbox": [0.499, 0.535, 0.608, 0.816],
                    "predicate": ["grind", "ride", "stand_on"],
                    
                    # The sentences
                    "sentences": [
                        "A person is grinding, riding, and standing on a skateboard.",
                        "The person is wearing a red cap, a black shirt with red sleeves, and black pants.",
                        "The person is balancing on the skateboard while grinding along the edge of the skatepark.",
                        "The skateboard is a narrow dark board positioned beneath the person's feet.",
                        "The person wearing a red cap is riding and grinding on the skateboard while standing on it.",
                        "The person is standing on a dark skateboard.",
                        "The person is riding and grinding on the skateboard along the edge of the ramp.",
                        "The person wearing a red cap is riding, grinding, and standing on the skateboard in the skatepark."
                    ]
                }
            ]
        },
        {
            # "id": 2149,
            "file_name": "HICO_train2015_00000132.jpg",
            "annotations": [
                {
                    "subject": "person",
                    "object": "baseball_glove",
                    "subject_bbox": [0.323, 0.163, 0.601, 0.784],
                    "object_bbox": [0.499, 0.535, 0.608, 0.816],
                    "predicate": ["hold", "wear"],
                    
                    # The sentences
                    "sentences": [
                        "A person is holding and wearing a baseball glove.",
                        "The person is wearing a dark baseball uniform and a dark helmet.",
                        "The person is holding the baseball glove with their hands while standing at home plate.",
                        "The baseball glove is a dark-colored glove visible near the person's lower body.",
                        "The person wearing a dark baseball uniform is holding and wearing the baseball glove.",
                        "The person is near a dark-colored baseball glove.",
                        "The person is holding and wearing the dark-colored baseball glove.",
                        "The person wearing a dark baseball uniform is holding and wearing a dark-colored baseball glove."
                    ]
                }
            ]
        },
        {
            # "id": 2149,
            "file_name": "HICO_train2015_00000137.jpg",
            "annotations": [
                {
                    "subject": "person",
                    "object": "elephant",
                    "subject_bbox": [0.323, 0.163, 0.601, 0.784],
                    "object_bbox": [0.499, 0.535, 0.608, 0.816],
                    "predicate": ["hold", "pet"],
                    
                    # The sentences
                    "sentences": [
                        "A person is holding and petting an elephant.",
                        "The person is wearing a light-colored striped shirt and beige pants.",
                        "The person is touching the elephant with one hand while standing beside it.",
                        "The elephant is a large gray-brown animal standing in shallow water.",
                        "The person wearing a striped shirt is holding and petting the elephant.",
                        "The person is beside a large gray-brown elephant.",
                        "The person is holding and petting the large elephant.",
                        "The person wearing a striped shirt is holding and petting the large gray-brown elephant."
                    ]
                }
            ]
        },
        {
            # "id": 2149,
            "file_name": "HICO_train2015_00000141.jpg",
            "annotations": [
                {
                    "subject": "person",
                    "object": "hot_dog",
                    "subject_bbox": [0.109, 0.079, 0.91, 0.999],
                    "object_bbox": [0.448, 0.332, 0.833, 0.628],
                    "predicate": ["carry", "eat", "hold"],
                    
                    # The sentences
                    "sentences": [
                        "A person is carrying, eating, and holding a hot dog.",
                        "The person has short dark hair and is wearing a red and white shirt.",
                        "The person is holding the hot dog with both hands while bringing it to their mouth.",
                        "The hot dog is a long sandwich in a light-colored bun.",
                        "The person wearing a red and white shirt is carrying, eating, and holding the hot dog.",
                        "The person is holding a long hot dog with both hands.",
                        "The person is carrying and eating the long hot dog.",
                        "The person wearing a red and white shirt is carrying, eating, and holding the long hot dog."
                    ]
                }
            ]
        },
        {
            # "id": 2149,
            "file_name": "HICO_train2015_00000142.jpg",
            "annotations": [
                {
                    "subject": "person",
                    "object": "horse",
                    "subject_bbox": [0.49, 0.0, 0.773, 0.545],
                    "object_bbox": [0.226, 0.115, 0.856, 0.98],
                    "predicate": ["ride", "run", "straddle"],
                    
                    # The sentences
                    "sentences": [
                        "A person is riding, running with, and straddling a horse.",
                        "The person is wearing a dark blue shirt, white pants, and a white helmet.",
                        "The person is seated on the horse while holding the reins as the horse runs across the field.",
                        "The horse is a dark-colored horse with a black mane and tail.",
                        "The person wearing a dark blue shirt and white pants is riding and straddling the running horse.",
                        "The person is seated astride the dark-colored horse.",
                        "The person is riding and straddling the horse as it runs.",
                        "The person wearing a dark blue shirt, white pants, and white helmet is riding and straddling the dark-colored horse as it runs."
                    ]
                }
            ]
        },
        {
            # "id": 2149,
            "file_name": "HICO_train2015_00000144.jpg",
            "annotations": [
                {
                    "subject": "person",
                    "object": "bottle",
                    "subject_bbox": [0.462, 0.043, 0.737, 0.948],
                    "object_bbox": [0.409, 0.167, 0.542, 0.884],
                    "predicate": ["hold"],
                    
                    # The sentences
                    "sentences": [
                        "A person is holding a bottle.",
                        "The person is a baby wearing a light-colored patterned outfit.",
                        "The baby is holding the bottle upright with one hand.",
                        "The bottle is a tall green glass bottle with a white and blue label.",
                        "The baby wearing a patterned outfit is holding the bottle.",
                        "The baby is holding a tall green bottle.",
                        "The baby is holding the green bottle upright.",
                        "The baby wearing a light-colored patterned outfit is holding the tall green bottle upright."
                    ]
                }
            ]
        },
        {
            # "id": 2149,
            "file_name": "HICO_train2015_00000145.jpg",
            "annotations": [
                {
                    "subject": "person",
                    "object": "motorcycle",
                    "subject_bbox": [0.53, 0.319, 0.847, 0.987],
                    "object_bbox": [0.158, 0.108, 0.624, 0.988],
                    "predicate": ["hold", "inspect"],
                    
                    # The sentences
                    "sentences": [
                        "A person is holding and inspecting a motorcycle.",
                        "The person is wearing a dark motorcycle suit, gloves, a helmet, and white riding boots.",
                        "The person is crouching beside the motorcycle while holding and inspecting it.",
                        "The motorcycle is a large black, white, and green sport motorcycle.",
                        "The person wearing motorcycle gear is holding and inspecting the motorcycle.",
                        "The person is crouching beside a black, white, and green sport motorcycle.",
                        "The person is holding and inspecting the black, white, and green motorcycle.",
                        "The person wearing motorcycle gear is crouching beside and holding and inspecting the black, white, and green sport motorcycle."
                    ]
                }
            ]
        },
        {
            # "id": 2149,
            "file_name": "HICO_train2015_00000148.jpg",
            "annotations": [
                {
                    "subject": "person",
                    "object": "umbrella",
                    "subject_bbox": [0.464, 0.601, 0.53, 0.874],
                    "object_bbox": [0.449, 0.518, 0.56, 0.639],
                    "predicate": ["hold", "stand_under"],
                    
                    # The sentences
                    "sentences": [
                        "A person is holding and standing under an umbrella.",
                        "The person is wearing dark clothing and long pants.",
                        "The person is standing in the rain while holding the umbrella above their head.",
                        "The umbrella is a dark-colored umbrella held above the person.",
                        "The person wearing dark clothing is holding and standing under the umbrella.",
                        "The person is standing under a dark-colored umbrella.",
                        "The person is holding the dark-colored umbrella while standing underneath it.",
                        "The person wearing dark clothing is standing under and holding the dark-colored umbrella in the rain."
                    ]
                }
            ]
        },
        {
            # "id": 2149,
            "file_name": "HICO_train2015_00000149.jpg",
            "annotations": [
                {
                    "subject": "person",
                    "object": "book",
                    "subject_bbox": [0.469, 0.000, 0.819, 1.000],
                    "object_bbox": [0.560, 0.391, 0.748, 0.609],
                    "predicate": ["hold", "open", "read"],
                    
                    # The sentences
                    "sentences": [
                        "A person is holding, opening, and reading a book.",
                        "The person is wearing a cap, a jacket, and dark pants.",
                        "The person is holding the open book with both hands and reading it.",
                        "The book is an open book with printed pages.",
                        "The person wearing a cap and jacket is holding, opening, and reading the book.",
                        "The person is holding an open book.",
                        "The person is holding and reading the open book.",
                        "The person wearing a cap and jacket is holding and reading the open book."
                    ]
                }
            ]
        },
    ]
}
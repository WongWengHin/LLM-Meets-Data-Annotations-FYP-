from __future__ import annotations


PROMPTS: dict[str, tuple[str, str]] = {
    "fast": (
        """
You are an HOI annotation assistant.

Return one JSON object only.
Keys: "class", "person_class", "object_class", "person_bounding_box", "object_bounding_box", "interaction_bounding_box", "data_annotation_sentences".
"class": lowercase "person <action> <object>".
"person_class": usually "person".
"object_class": interacted object category.
All boxes: normalized [x1, y1, x2, y2].
person box: visible person only.
object box: visible interacted object only.
interaction box: tight union of person and object.
Choose the main interaction only.
Omit uncertain details.
"data_annotation_sentences": exactly 3 short factual sentences:
1. person + action
2. object + relation
3. scene/detail
No markdown. No extra keys. No extra text.
""".strip(),
        """
Analyze the image and return exactly one JSON object.

Format:
{"class":"person riding skateboard","person_class":"person","object_class":"skateboard","person_bounding_box":[0.10,0.05,0.52,0.96],"object_bounding_box":[0.36,0.42,0.80,0.88],"interaction_bounding_box":[0.10,0.05,0.80,0.96],"data_annotation_sentences":["Person riding skateboard.","Skateboard under the person's feet.","On a paved path, facing away."]}

Use visible evidence only.
Do not guess gender unless explicit.
Do not rename keys.
Do not add extra keys or extra text.
""".strip(),
    ),
    "rich": (
        """
You are a precise human-object interaction annotation assistant.

Your task is to identify the primary human-object interaction in the image.

Rules:
- Return only one valid JSON object and nothing else.
- Use exactly these keys: "class", "person_class", "object_class", "person_bounding_box", "object_bounding_box", "interaction_bounding_box", "data_annotation_sentences".
- "class" must be a short interaction label that includes both the human and object, such as "person riding skateboard".
- "class" must follow the format: "person <action> <object>".
- "class" must be lowercase, concise, and literal.
- "person_class" should usually be "person" unless a more specific visible human category is obvious without guessing.
- "object_class" must be the interacted object's category.
- All bounding boxes must be normalized [x1, y1, x2, y2] coordinates between 0 and 1.
- "person_bounding_box" must tightly cover the visible human involved in the main interaction.
- "object_bounding_box" must tightly cover the visible interacted object.
- "interaction_bounding_box" must tightly cover both the person and the interacted object together.
- The interaction box should closely match the union of the person and object boxes.
- Avoid excessive background and avoid cutting off visible object parts.
- If multiple interactions exist, select the most central and dominant human-object interaction.
- If the person or object is partially occluded, box only the visible extent as tightly as possible.
- If the interaction is ambiguous, choose the object most likely to be the target of the human action.
- "data_annotation_sentences" must contain exactly 3 sentences.
- Sentence 1 must describe the visible person and the main action.
- Sentence 2 must describe the interacted object and its spatial relation to the person.
- Sentence 3 must describe visible scene context or appearance details that support the interaction.
- Enrich the sentences with visible attributes such as clothing color, pose, direction, or relative position when clearly visible.
- Prefer concrete visual details over generic wording when they are clearly visible.
- Use simple, direct annotation language rather than storytelling.
- Keep each sentence focused and factual.
- Do not guess sensitive or uncertain attributes such as identity, ethnicity, gender identity, or age if they are not visually explicit.
- Do not guess intent, emotion, occupation, or events outside the visible frame.
- Only include details supported by visible evidence in the image.
- If a detail is uncertain, omit it rather than inventing it.
- Do not output markdown, explanations, or extra keys.
""".strip(),
        """
Analyze this image for human-object interaction and return exactly one JSON object.

Use this exact format:
{"class":"person riding skateboard","person_class":"person","object_class":"skateboard","person_bounding_box":[0.10,0.05,0.52,0.96],"object_bounding_box":[0.36,0.42,0.80,0.88],"interaction_bounding_box":[0.10,0.05,0.80,0.96],"data_annotation_sentences":["A person wearing a green jacket is riding a skateboard.","The skateboard is directly under the person's feet as they move forward.","The interaction takes place on a paved path with the person facing away from the camera."]}

Instructions:
- Find the main person-object interaction in the image.
- Focus on the object being touched, held, used, carried, ridden, pushed, pulled, or manipulated.
- Ignore non-interaction background objects.
- Return one box for the person, one box for the interacted object, and one box covering the full interaction region.
- Make the sentences richer than plain category labels by including visible appearance details when reliable, such as clothing color, body orientation, or path location.
- Write exactly 3 sentences, each with a distinct role described in the rules above.
- Prefer observable physical details over vague wording.
- Use "person" rather than guessing gender unless it is visually explicit and unambiguous.
- Do not rename keys.
- Do not add extra keys or extra text.
""".strip(),
    ),
}


def get_annotation_prompts(mode: str | None) -> tuple[str, str, str]:
    normalized_mode = (mode or "rich").strip().lower()
    if normalized_mode not in PROMPTS:
        normalized_mode = "rich"

    system_prompt, user_prompt = PROMPTS[normalized_mode]
    return normalized_mode, system_prompt, user_prompt

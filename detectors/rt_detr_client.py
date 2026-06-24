from __future__ import annotations

from pathlib import Path

from core.config import DetectorConfig
from images.image_metadata import read_image_dimensions


def detect_instances(image_path: Path, detector_config: DetectorConfig) -> list[dict]:
    if detector_config.provider != "rt_detr":
        raise RuntimeError(f"Unsupported detector provider: {detector_config.provider}")

    try:
        import torch
        from PIL import Image
        from transformers import AutoImageProcessor, RTDetrForObjectDetection
    except ImportError as exc:
        raise RuntimeError(
            "RT-DETR dependencies are not installed. Install 'torch' and 'transformers' to enable detector mode."
        ) from exc

    processor = AutoImageProcessor.from_pretrained(detector_config.model_name)
    model = RTDetrForObjectDetection.from_pretrained(detector_config.model_name)

    if detector_config.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("RT-DETR device is set to cuda, but CUDA is not available.")
        model = model.to("cuda")
    else:
        model = model.to("cpu")

    width, height = read_image_dimensions(image_path)

    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
        inputs = processor(images=rgb_image, return_tensors="pt")

    if detector_config.device == "cuda":
        inputs = {key: value.to("cuda") for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([[height, width]], device=outputs.logits.device)
    processed = processor.post_process_object_detection(
        outputs,
        target_sizes=target_sizes,
        threshold=detector_config.score_threshold,
    )[0]

    detections: list[dict] = []
    labels = processed["labels"].tolist()
    scores = processed["scores"].tolist()
    boxes = processed["boxes"].tolist()

    for label_id, score, box in zip(labels, scores, boxes):
        x1, y1, x2, y2 = box
        detections.append(
            {
                "category": str(model.config.id2label.get(int(label_id), str(label_id))),
                "bounding_box": [
                    round(max(0.0, min(1.0, x1 / width)), 6),
                    round(max(0.0, min(1.0, y1 / height)), 6),
                    round(max(0.0, min(1.0, x2 / width)), 6),
                    round(max(0.0, min(1.0, y2 / height)), 6),
                ],
                "score": round(float(score), 6),
                "detection": {
                    "source": "rt_detr",
                    "model": detector_config.model_name,
                },
            }
        )

    return detections[: detector_config.max_detections]

from app.services.prompts.image_annotation_prompts import get_annotation_prompts
from app.evaluation.hicodet_registry import HICO_OBJECTS, HICO_VERBS


def test_get_annotation_prompts_returns_task_specific_modes() -> None:
    object_detection_mode, object_detection_system, object_detection_user = get_annotation_prompts("object_detection")
    hoi_mode, hoi_system, hoi_user = get_annotation_prompts("hoi")

    assert object_detection_mode == "object_detection"
    assert "object-detection-only" in object_detection_system
    assert "object-detection format" in object_detection_user

    assert hoi_mode == "hoi"
    assert "Human-Object Interaction (HOI) image" in hoi_system
    assert "HOI format" in hoi_user
    assert "Valid HICO-DET object labels:" in hoi_system
    assert "Valid HICO-DET predicate labels:" in hoi_system
    assert f"- {HICO_OBJECTS[0]}" in hoi_system
    assert f"- {HICO_VERBS[0]}" in hoi_system

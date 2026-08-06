from app.services.prompts.image_annotation_prompts import get_annotation_prompts


def test_get_annotation_prompts_returns_task_specific_modes() -> None:
    object_detection_mode, object_detection_system, object_detection_user = get_annotation_prompts("object_detection")
    hoi_mode, hoi_system, hoi_user = get_annotation_prompts("hoi")

    assert object_detection_mode == "object_detection"
    assert "object-detection-only" in object_detection_system
    assert "object-detection format" in object_detection_user

    assert hoi_mode == "hoi"
    assert "Human-Object Interaction (HOI) image" in hoi_system
    assert "HOI format" in hoi_user

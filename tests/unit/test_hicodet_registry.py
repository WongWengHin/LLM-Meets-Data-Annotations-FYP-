from app.evaluation.hicodet_registry import normalize_hico_object_label, valid_hico_interactions


def test_hicodet_object_labels_normalize_to_canonical_form() -> None:
    assert normalize_hico_object_label("baseball_bat") == "baseball bat"
    assert normalize_hico_object_label("cell_phone") == "cell phone"
    assert normalize_hico_object_label("hot-dog") == "hot dog"
    assert normalize_hico_object_label("tennis_ball") == "sports ball"


def test_local_hicodet_interaction_registry_uses_canonical_objects() -> None:
    interactions = valid_hico_interactions()

    assert ("hold", "sports ball") in interactions
    assert ("talk_on", "cell phone") in interactions
    assert ("hold", "tennis_ball") not in interactions
    assert ("talk_on", "cell_phone") not in interactions

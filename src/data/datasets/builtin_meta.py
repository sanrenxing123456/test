COCO_CATEGORIES = [
    {"color": [], "isthing": 0, "id": 0, "name": "N/A"},
    {"color": [255, 0, 0], "isthing": 1, "id": 1, "name": "children"},
    {"color": [0, 255, 0], "isthing": 1, "id": 2, "name": "bed"},
    {"color": [0, 0, 255], "isthing": 1, "id": 3, "name": "parent"},
    {"color": [255, 255, 0], "isthing": 1, "id": 4, "name": "nurse"},
    {"color": [255, 165, 0], "isthing": 1, "id": 5, "name": "chair"},
    {"color": [128, 0, 128], "isthing": 1, "id": 6, "name": "nightstand"},
    {"color": [0, 255, 255], "isthing": 1, "id": 7, "name": "leftbed"},
    {"color": [255, 192, 203], "isthing": 1, "id": 8, "name": "rightbed"},
    {"color": [128, 128, 0], "isthing": 1, "id": 9, "name": "frontbed"},
    {"color": [0, 128, 128], "isthing": 1, "id": 10, "name": "backbed"},
]

def _get_coco_instances_meta():
    thing_ids = [k["id"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == 10, f"Length of thing ids : {len(thing_ids)}"
    
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in COCO_CATEGORIES if k["isthing"] == 1]

    coco_classes = [k["name"] for k in COCO_CATEGORIES]

    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
        "coco_classes": coco_classes,
    }

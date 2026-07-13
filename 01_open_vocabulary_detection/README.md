# Open-Vocabulary Detection: The Vocabulary Ceiling Problem

Why does a modern object detector confidently mislabel a tiger as a zebra - and what fixes it?

## The problem

Pretrained YOLO26 is trained on COCO's 80 object classes. Point it at a tiger, and it says "zebra" at 95% confidence, with a tight, accurate bounding box - not a model-quality issue, just a fixed vocabulary. "Tiger" isn't one of the 80 classes, so it picks the closest match it knows. Same result on YOLO11 and YOLO26 both.

## Three fixes, tested honestly

| Approach | Works? | Cost |
|---|---|---|
| YOLO-World (open-vocabulary, prompted) | Yes | None, just prompting |
| Train on LVIS (1,203 classes) | Not run | Many GPU-hours - vocabulary confirmed (`tiger` = class 1087), but not worth the compute for this demo |
| Fine-tuned YOLO26 (custom dataset) | Yes | ~2.1k images, 10 epochs, minutes on M5 Max |

Full walkthrough: [`tiger_detection_problem.ipynb`](./tiger_detection_problem.ipynb)

## Files

- `tiger_detection_problem.ipynb` - problem, all three solutions, comparison table
- `download_and_train.py` - downloads a labeled tiger dataset, fixes the train/val split, fine-tunes YOLO26

## Setup

```bash
pip install ultralytics roboflow matplotlib pillow pandas
export ROBOFLOW_API_KEY="your_key_here"
```

Open the notebook and run top to bottom, or reproduce the fine-tune with:

```bash
python download_and_train.py
```

## Takeaway

No single fix wins outright. Flexibility, no training? YOLO-World. Reliability on categories you control? Fine-tune. Broad coverage without prompting? Train on a bigger vocabulary, if you can afford it.

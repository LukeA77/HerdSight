# Licence register: code and starting weights

Every external model, set of starting weights and codebase used to produce a result, with its source and licence. Training data is covered by `docs/data.md` (D-21). Starting weights pretrained on research datasets are allowed in Stage A and get a legal check before any product (D-36).

| Item | Used for | Source | Code licence | Weights licence | Pretraining data | Status for a product |
|---|---|---|---|---|---|---|
| D-FINE S and N, COCO weights (`dfine_s_coco.pth`, `dfine_n_coco.pth`) | Phase 1 fine-tuned baselines; Phase 2 starting point | github.com/Peterande/D-FINE, pinned commit `956d1709314c2c6a4df6f34de232054578a7449f`; weights from the authors' release `dfinev1.0` | Apache-2.0 | Released with the repository (Apache-2.0) | COCO (images under Flickr terms, annotations CC BY 4.0); backbone HGNetv2 pretrained on ImageNet (non-commercial research terms) | Legal check before Stage B (D-36) |
| OWLv2 (`google/owlv2-base-patch16-ensemble`) | Phase 1 zero-shot baseline only | Hugging Face | Apache-2.0 | Apache-2.0 | Web-scale image-text data and public detection sets | Baseline only; not part of any product |
| Segment Anything (SAM ViT-B) | Generating oriented training labels (Phase 2) | github.com/facebookresearch/segment-anything | Apache-2.0 | Apache-2.0 | SA-1B | Labels only; model not shipped |

Added as each new model or set of weights is used.

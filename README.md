# MetalVision-AI

An educational and research training prototype for metal-surface image classification. [main.py](main.py) defines a pretrained ResNet18 baseline, a custom residual/attention CNN, and an ensemble combining ResNet50, ResNet34, and a custom branch. It is a training script, not a deployed inspection system.

## Inputs and setup

```bash
git clone https://github.com/Zinga18018/MetalVision-AI.git
cd MetalVision-AI
python -m venv .venv-local
source .venv-local/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv-local\Scripts\activate`. [requirements.txt](requirements.txt) contains broad minimum versions, not a verified environment lock.

The script expects an external dataset in this layout, with class subdirectories understood by torchvision `ImageFolder`:

```text
NEU Metal Surface Defects Data/
  train/<class>/...
  valid/<class>/...
  test/<class>/...
```

The dataset, weights, and evaluation results are not included. The project names NEU Metal Surface Defects Data, but does not supply a dataset source/usage-terms record; establish those before acquiring or redistributing images. Pretrained ResNet initialization may download weights. Training requires substantial compute; device selection uses CUDA when available, otherwise CPU.

The entry point is `python main.py` from the repository root, **but the current cross-validation issue below prevents treating it as a working end-to-end evaluation recipe**. Dataset loading happens at import time; the training workflow is guarded by `if __name__ == '__main__'`.

## Implemented workflow and known issue

- Image augmentation, weighted training sampling, early stopping, and learning-rate scheduling are implemented.
- The script trains multiple models, then attempts three-fold cross-validation and later test evaluation/majority voting.
- **Cross-validation currently treats the tuple returned by `evaluate_model` as a scalar accuracy.** Formatting that tuple with `.4f` fails before the later test-evaluation section. The caller must unpack the return value before that workflow can complete.
- SMOTE is a later feature-resampling demonstration; its output is not fed into the earlier model training. Gradient accumulation is not implemented: optimizer gradients are reset and stepped each batch.

## Results and outputs

No measured accuracy range or ensemble improvement is established by committed results. The former “expected” accuracy percentages were not benchmark evidence, and no such performance is promised here.

The script writes `training_history.png` and `confusion_matrix.png` and prints classification metrics when it reaches the relevant stages. Multiple model calls reuse those filenames, so plots can overwrite earlier outputs rather than form a per-model archive. There is no committed test suite or run manifest establishing reproducibility.

Review the evaluation flow and retain dataset provenance, split definitions, environment, seeds, and numeric results before publishing a new model comparison. The original educational/research-purpose notice is retained; no project license file is included.

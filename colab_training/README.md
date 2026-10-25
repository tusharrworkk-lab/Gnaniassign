# Colab MiniLM training

This folder fine-tunes `microsoft/Multilingual-MiniLM-L12-H384` on the Hugging
Face dataset `cmul8-hf/IndicJevBench` and writes a standard Hugging Face model
directory. It needs a Colab GPU runtime (T4 or better).

## In a new Colab notebook

Upload this `colab_training` folder to Google Drive, then run these cells:

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
%cd /content/drive/MyDrive/colab_training
!pip install -q -r requirements.txt
!nvidia-smi
```

```bash
!python train_minilm_indicjevbench.py \
  --epochs 5 \
  --batch-size 8 \
  --output /content/drive/MyDrive/systemone-indic-v1
```

The final model is saved in `MyDrive/systemone-indic-v1`. Each epoch is also
saved as `checkpoint-epoch-N`, so an interrupted run still leaves recoverable
model snapshots. Copy the final directory back into this repository at
`artifacts/models/systemone-indic-v1/` when training finishes.

The default `--trainable-layers 0` fine-tunes all 12 MiniLM layers, appropriate
for a Colab T4 GPU. If Colab reports CUDA out-of-memory, rerun with
`--trainable-layers 2 --batch-size 4`.

IndicJevBench has four named training sources rather than a conventional `train`
split: `intent_massive`, `fintech_banking77`, `hinglish_lid`, and
`synthetic_enterprise`. The script trains on all four by default, matching the
repository's original training script. To choose sources explicitly, pass for
example `--splits intent_massive,fintech_banking77`.

For new runs, the script reserves 10% of examples for validation by default. It
groups identical state text before splitting, so the several questions about one
underlying customer state cannot leak across train and validation. The saved
`validation-split.jsonl` records the exact held-out set. At every epoch, the
script evaluates it and stores the best-validation model at the output root;
all epoch checkpoints are retained separately. This validation only applies to
a fresh run: the earlier model was trained on all examples and cannot be given
an honest held-out score afterward.

## Evaluate the trained model

Create a JSONL holdout file using `validation.example.jsonl` as the schema. Its
examples must be genuinely unseen during training; do not evaluate against the
IndicJevBench rows used above, because that measures memorisation rather than
generalisation. Collect and human-label at least 100 real Hindi, Hinglish,
English, and STT-noisy examples for a meaningful number.

`validation.unseen.retail-v1.jsonl` provides 108 balanced, newly written retail
intent examples (12 for each label) for a quick smoke test. They are synthetic,
so report that score separately from a future human-labelled production holdout.

After downloading or keeping the model in Drive, run:

```bash
!python3 evaluate_minilm.py \
  --model /content/drive/MyDrive/systemone-indic-v1 \
  --data validation.jsonl \
  --output evaluation-results.json
```

It prints exact-option accuracy and mean cross-entropy loss, and writes every
prediction, selected option, confidence, and correct/incorrect flag to the
output JSON file.

## Continue fine-tuning for retail intents

After uploading `retail-intent-synthetic-v1.jsonl` and your existing
`systemone-indic-v1` model folder to Drive, run:

```bash
!python3 train_retail_continuation.py \
  --model /content/drive/MyDrive/systemone-indic-v1 \
  --data /content/drive/MyDrive/colab_training/retail-intent-synthetic-v1.jsonl \
  --output /content/drive/MyDrive/systemone-retail-v1 \
  --epochs 2 \
  --batch-size 8 \
  --learning-rate 3e-6
```

It trains only the dataset's `train` rows, selects the model with the best
`validation` accuracy, and performs one final evaluation on `test`.

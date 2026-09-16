# Causal detector learning audit

{
  "macro": {
    "static_logistic": {
      "f1": 0.9256220129257874,
      "fpr": 0.15104166666666666,
      "brier": 0.07404535912391452,
      "accuracy": 0.911764705882353,
      "recall": 0.9675925925925926
    },
    "mlp": {
      "f1": 0.9525694893341952,
      "fpr": 0.057291666666666664,
      "brier": 0.036869417912780554,
      "accuracy": 0.9485294117647058,
      "recall": 0.9537037037037037
    },
    "gru": {
      "f1": 0.9821251241310824,
      "fpr": 0.03125,
      "brier": 0.01512698033501413,
      "accuracy": 0.9803921568627451,
      "recall": 0.9907407407407407
    }
  },
  "selection": {
    "selected": "gru",
    "rule": "max macro F1 subject to FPR non-inferior and Brier delta <= 0.02"
  }
}

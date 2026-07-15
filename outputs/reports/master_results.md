# Master results — EuroSAT land-use classification

## Distillation / capacity sweep (JPG-RGB, random 80/10/10 split)

| Model / input | Params | Accuracy | Macro F1 |
|---|---|---|---|
| ScratchCNN 94K | 94,762 | 92.26% | 0.9212 |
| ScratchCNN 94K + KD | 94,762 | 91.44% | 0.9130 |
| ScratchCNN-S 391K | 391,466 | 95.67% | 0.9566 |
| ScratchCNN-S 391K + KD | 391,466 | 95.85% | 0.9586 |
| ScratchCNN-M 1.56M | 1,557,066 | 96.41% | 0.9637 |
| ScratchCNN-M 1.56M + KD | 1,557,066 | 96.48% | 0.9646 |
| ResNet-18 (ImageNet) | 11,181,642 | 98.63% | 0.9860 |
| EfficientNet-V2-S (teacher) | 20,190,298 | 98.93% | 0.9888 |

## Multispectral input study (raw reflectance, official 60/20/20 split, fixed 1.56M model)

| Model / input | Params | Accuracy | Macro F1 |
|---|---|---|---|
| RGB (3 bands) | 1,557,066 | 97.59% | 0.9755 |
| All bands (12) | 1,562,250 | 98.50% | 0.9844 |
| All bands + indices (15) | 1,563,978 | 98.44% | 0.9839 |

## Task-optimized virtual sensors (learned k-channel spectral mixing, same model/split)

| Model / input | Params | Accuracy | Macro F1 |
|---|---|---|---|
| k=1 learned channel | 1,555,926 | 96.06% | 0.9599 |
| k=2 learned channels | 1,556,514 | 97.85% | 0.9780 |
| k=3 learned channels | 1,557,102 | 98.17% | 0.9813 |
| k=4 learned channels | 1,557,690 | 98.19% | 0.9812 |
| k=6 learned channels | 1,558,866 | 98.48% | 0.9845 |

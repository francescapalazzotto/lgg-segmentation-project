# Segmentazione di Gliomi con Deep Learning

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Questo progetto implementa e confronta diverse architetture di reti neurali per la segmentazione di gliomi a basso grado (LGG) da immagini MRI. Il lavoro si ispira a un articolo scientifico che utilizza UNet per estrarre caratteristiche dalle immagini da correlare con dati genomici.

## Dataset

Il dataset contiene immagini MRI cerebrali con maschere di segmentazione manuale delle anomalie FLAIR. I dati provengono da The Cancer Imaging Archive (TCIA) e corrispondono a 110 pazienti del TCGA lower-grade glioma collection.

Struttura dei dati:
- Immagini in formato `.tif` a 3 canali
- Maschere binarie a 1 canale
- Dati clinici e genomici in `data.csv`

## Modelli Implementati

1. **UNet**: Architettura base di riferimento
2. **UNet++**: Variante con connessioni nidificate e dense
3. **Attention UNet**: Variante con meccanismi di attenzione

## Metriche di Valutazione

Le performance vengono valutate usando:
- Dice Coefficient (F1 score)
- Intersection over Union (IoU)
- Precisione e Recall
- Accuratezza

## Impostazione del progetto
lgg-segmentation-project/
│
├── data/
│   ├── raw/               # Dati originali (cartelle TCGA_*)
│   ├── processed/         # Dati preprocessati
│   └── splits/            # Train/val/test splits
│
├── models/
│   ├── unet.py           # Implementazione UNet base
│   ├── unet_plusplus.py  # Implementazione UNet++
│   └── attention_unet.py # Implementazione Attention UNet
│
├── notebooks/
│   ├── 01_eda.ipynb                # Analisi esplorativa
│   ├── 02_preprocessing.ipynb      # Preprocessing
│   └── 03_model_comparison.ipynb   # Confronto modelli
│
├── scripts/
│   ├── preprocess.py      # Script di preprocessing
│   ├── train.py           # Script di training
│   ├── split_dataset.py   # Script di per la suddivisone del dataset
│   └── evaluate.py        # Script di valutazione
│
├── configs/
│   ├── base.yaml        # Configurazioni comuni
│   ├── unet.yaml        # Configurazioni UNet
│   └── ...             # Altre configurazioni
│
├── requirements.txt    # Dipendenze
├── README.md           # Documentazione progetto
└── .gitignore

## Getting Started

### Prerequisiti

- Python 3.8+
- PyTorch 1.10+
- OpenCV
- NumPy, Pandas
- Matplotlib, Seaborn

### Installazione

1. Clonare la repository:
   ```bash
   git clone https://github.com/yourusername/lgg-segmentation-project.git
   cd lgg-segmentation-project
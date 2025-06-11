# Segmentazione di Gliomi Cerebrali con Deep Learning

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Questo progetto si focalizza sulla segmentazione di gliomi a basso grado (LGG) da immagini di Risonanza Magnetica (MRI) utilizzando diverse architetture di reti neurali convoluzionali. Il lavoro mira a confrontare le performance di modelli noti nel campo della segmentazione medica per identificare l'approccio più efficace per questo compito.

## Dataset

Il dataset utilizzato è basato su immagini MRI cerebrali con maschere di segmentazione manuale delle anomalie FLAIR. I dati provengono da The Cancer Imaging Archive (TCIA) e corrispondono a 110 pazienti della TCGA lower-grade glioma collection.

**Caratteristiche del Dataset:**
* Immagini cerebrali in formato `.tif` a 3 canali (RGB)
* Maschere binarie a 1 canale (`_mask.tif`) che delineano la regione del tumore.
* Il dataset presenta un **forte squilibrio di classe**, con i pixel del tumore che costituiscono una piccola percentuale dell'area totale dell'immagine.

## Modelli Implementati e Confrontati

In questo studio sono state implementate e confrontate le seguenti architetture di rete neurale per la segmentazione semantica:

1.  **U-Net**: L'architettura base di riferimento, ampiamente utilizzata per compiti di segmentazione medica.
2.  **Attention U-Net**: Una variante di U-Net che incorpora "attention gates" per migliorare la focalizzazione sulle regioni più rilevanti (come il tumore), potenziando la discriminazione delle feature.
3.  **LinkNet**: Un'architettura efficiente che mira a bilanciare le performance con un uso più efficiente delle risorse computazionali, riutilizzando efficacemente le feature dell'encoder nel decoder.

## Metriche di Valutazione

Le performance dei modelli sono state valutate utilizzando metriche specifiche per la segmentazione, particolarmente adatte a dataset con squilibrio di classe:

* **Dice Coefficient (F1 Score)**: Misura la sovrapposizione tra la maschera predetta e la maschera di verità.
* **Intersection over Union (IoU - Jaccard Index)**: Un'altra metrica di sovrapposizione, simile al Dice, che quantifica la somiglianza tra le due regioni.
* **Binary Cross-Entropy with Logits Loss**: Utilizzata come funzione di loss durante l'addestramento, indica l'errore del modello.

## Struttura del Progetto

lgg-segmentation-project/
│
├── data/
│   └── lgg-mri-segmentation/             # Contiene le cartelle TCGA_* con immagini e maschere
│
├── training_results/                     # Nuova cartella: Risultati del training (modelli .pth, log .csv/.json, grafici .png)
│   ├── unet_best_model.pth
│   ├── attention_unet_best_model.pth
│   ├── linknet_best_model.pth
│   ├── unet_metrics_plot.png
│   ├── attention_unet_metrics_plot.png
│   ├── linknet_metrics_plot.png
│   ├── unet_training_log.csv
│   ├── attention_unet_training_log.csv
│   ├── linknet_training_log.csv
│   └── training_summary.json             # Riepilogo dei log
│
├── notebooks/
│   └── 1-analisi_esplorativa.ipynb # Analisi esplorativa dei dati e confronto dei risultati del training
│
├── run.py                                # Script principale per la preparazione dati, training e salvataggio risultati
├── requirements.txt                      # Dipendenze del progetto
├── README.md                             
└── .gitignore

**Nota sulla struttura:** Il tuo progetto attuale sembra essere più un "monorepo" con un singolo script `run.py` che gestisce l'intero flusso di lavoro. Ho aggiornato la struttura per riflettere questo, rimuovendo le cartelle `models/`, `scripts/`, `configs/`, `data/raw`, `data/processed`, `data/splits` che non sembrano essere più in uso con il tuo setup attuale. Se queste cartelle contengono ancora codice o dati rilevanti e sono state usate in una fase precedente, puoi reintrodurle nel `README` con una breve spiegazione.

## Getting Started

### Prerequisiti

* Python 3.8+
* PyTorch 1.10+
* NumPy, Pandas
* Matplotlib
* Pillow (PIL)
* Scikit-learn
* Tqdm
* Segmentation Models PyTorch (`segmentation_models_pytorch`)

### Installazione

1.  Clonare la repository:

    ```bash
    git clone [https://github.com/yourusername/lgg-segmentation-project.git](https://github.com/yourusername/lgg-segmentation-project.git)
    cd lgg-segmentation-project
    ```

2.  Installare le dipendenze:

    ```bash
    pip install -r requirements.txt
    ```

### Esecuzione del Progetto

1.  **Preparazione dei Dati:**
    Assicurati che il dataset `lgg-mri-segmentation` sia posizionato nella cartella `data/` del progetto, come segue: `lgg-segmentation-project/data/lgg-mri-segmentation`.

2.  **Addestramento dei Modelli e Generazione dei Log/Grafici:**
    Eseguire lo script principale `run.py`. Questo script caricherà i dati, addestrerà i modelli (U-Net, Attention U-Net, LinkNet), salverà i pesi dei modelli migliori, i log di training in formato CSV e JSON, e genererà i grafici delle performance nella cartella `training_results/`.

    ```bash
    python run.py
    ```
    *Nota: Se i log e i modelli esistono già, lo script li caricherà senza riaddestrare, permettendo di generare nuovamente grafici e log da dati preesistenti.*

3.  **Analisi delle Performance e Discussione:**
    Aprire il notebook Jupyter `notebooks/1-analisi_esplorativa.ipynb` per eseguire un'analisi esplorativa del dataset (es. squilibrio di classe) e per caricare, visualizzare e confrontare le performance dei modelli basate sui log di training.

    ```bash
    jupyter notebook
    ```
    Quindi, navigare e aprire `notebooks/1-analisi_esplorativa.ipynb`.

---
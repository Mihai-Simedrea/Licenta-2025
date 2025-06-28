
# 🚀 Modele mici cu impact mare: Detecția și clasificarea leziunilor mamare folosind modele foarte mici

👀 **Întregul cod sursă al proiectului poate fi consultat aici:**  
[🔗 Link GitHub Proiect](https://github.com/Mihai-Simedrea/Licenta-2025$0)

---

## ⚡ Predicții rapide în Google Colab
Rulează predicții pentru **clasificare** și **detecție** direct în Colab:  
[🔗 Link Google Colab](https://colab.research.google.com/drive/1wLyBc-Me_ngQP7Te4Yd64_og2fD6DofN#scrollTo=xhJc9DRnCY35$0)

---

## ⚙️ Setup complet

### 📂 Detalii set de date
Setul de date utilizat este **CBIS-DDSM**.
- 📥 **Descărcare originală (DICOM):**  
  [CBIS-DDSM DICOM](https://www.cancerimagingarchive.net/collection/cbis-ddsm/$0)
- 🖼️ Pentru a evita conversia DICOM → JPEG, recomandăm descărcarea directă de pe Kaggle:
  ```bash
  #!/bin/bash
  curl -L -o ~/Downloads/cbis-ddsm-breast-cancer-image-dataset.zip     https://www.kaggle.com/api/v1/datasets/download/awsaf49/cbis-ddsm-breast-cancer-image-dataset
  ```
---

### 🐍 Mediu virtual
Este recomandat:
- **Python 3.9.6**
- **PyTorch 2.7.1**
- **Tensorflow 2.19.0**

🔧 Creare mediu virtual:
```bash
python3 -m venv env
source env/bin/activate
```
📦 Instalare pachete:
```bash
pip install -r requirements.txt
```
---

## 🧠 Predicții locale

### 🔍 Detecție leziuni (PyTorch)
Rulează detecția și vizualizarea casetei de încadrare:
```bash
python3 -m inference.detect   --backbone <densenet | resnet>   --image_path <cale_către_imagine>   --model_path <cale_către_model.pth>   --score_threshold <între 0f și 1f>
```
---

### 🏷️ Clasificare leziuni (TensorFlow)
Rulează clasificarea imaginii:
```bash
python3 -m inference.classify   --image_path <cale_către_imagine>   --model_path <cale_către_model.keras>
```
---

## 🎯 Antrenarea modelelor

### 📈 Detecție
Poți antrena cu două arhitecturi:

✅ **ResNet50 (~45M parametri):**
```bash
chmod +x run_detection_resnet.sh
./run_detection_resnet.sh
```

✅ **DenseNet121 (~13M parametri):**
```bash
chmod +x run_detection_densenet.sh
./run_detection_densenet.sh
```

---

### 🏷️ Clasificare
Rulare:
```bash
chmod +x run_classification.sh
./run_classification.sh
```
---

**Explicații argumente fișiere shell:**
- `--epochs`: Numărul de epoci de antrenare
- `--batch-size`: Dimensiunea batch-ului
- `--learning-rate`: Rata de învățare
- `--pipeline`: Tipul preprocesării imaginilor (`resize`, `normalize` etc.)
- `--limit`: Numărul maxim de imagini folosite pentru antrenare (pt. test rapid)

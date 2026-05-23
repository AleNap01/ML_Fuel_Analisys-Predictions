# 🚗 ML Fuel Analysis & Predictions

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/XGBoost-FF6600?style=flat-square"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/PostgreSQL-336791?style=flat-square&logo=postgresql&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square"/>
</p>

> Progetto di Tesi di Laurea Triennale in Ingegneria Informatica — Università degli Studi di Salerno (UNISA)

Sistema interattivo per la **predizione del consumo di carburante** in flotte di veicoli aziendali tramite algoritmi di Machine Learning. L'obiettivo è ridurre i costi operativi e migliorare l'efficienza delle aziende di logistica e trasporti.

---

## 📋 Indice

- [Funzionalità](#-funzionalità)
- [Architettura del progetto](#-architettura-del-progetto)
- [Requisiti](#-requisiti)
- [Installazione](#-installazione)
- [Utilizzo](#-utilizzo)
- [Dataset](#-dataset)
- [Modelli ML utilizzati](#-modelli-ml-utilizzati)
- [Tesi](#-tesi)
- [Licenza](#-licenza)

---

## ✨ Funzionalità

- 📊 **Dashboard interattiva** con Streamlit per la visualizzazione e configurazione del sistema
- 🤖 **Predizione del consumo** carburante tramite modelli Random Forest e XGBoost
- 🏎️ **Classificazione dello stile di guida** (aggressivo / moderato / prudente) da dati dei sensori di bordo
- 🗺️ **Integrazione con limiti di velocità OSM** per contestualizzare i dati di percorso
- 🗄️ **Persistenza su PostgreSQL** per la gestione dei dati di flotta

---

## 🗂️ Architettura del progetto

```
ML_Fuel_Analisys-Predictions/
│
├── dashboard.py                 # Entry point — dashboard stile di guida
├── consumo_carburante.py        # Calcolo e predizione consumo carburante
├── stile_guida.py               # Analisi e classificazione stile di guida
├── DrivingStylePredictor.py     # Modello ML per la predizione dello stile
├── associazione_limiti.py       # Integrazione limiti di velocità OSM Italy
│
├── pages/                       # Pagine aggiuntive della dashboard Streamlit
├── Tesi_di_laurea/              # Documento della tesi completo
│
├── requirements.txt             # Dipendenze Python
└── README.md
```

---

## ⚙️ Requisiti

- Python **3.10**
- PostgreSQL (con due database configurati, vedi [Dataset](#-dataset))

---

## 🚀 Installazione

**1. Clona la repository**

```bash
git clone https://github.com/AleNap01/ML_Fuel_Analisys-Predictions.git
cd ML_Fuel_Analisys-Predictions
```

**2. Crea e attiva un ambiente virtuale**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. Installa le dipendenze**

```bash
pip install -r requirements.txt
```

---

## ▶️ Utilizzo

**1. Scarica e configura i database** (vedi sezione [Dataset](#-dataset))

**2. Avvia la dashboard**

```bash
streamlit run dashboard.py
```

---

## 🗄️ Dataset

Il sistema richiede due database PostgreSQL:

| Database | Descrizione |
|---|---|
| **OSM Italy** | Limiti di velocità di tutte le strade italiane, usati per comparare la velocità del veicolo con il limite della strada percorsa |
| **Dati di bordo** | Informazioni raccolte dai sensori dei veicoli (velocità, carburante, stile di guida, ecc.) |

📥 **Download dataset:** [Google Drive](https://drive.google.com/file/d/1npi5BAJr2ODFzVTeLKKcjwHq6fRPK0CB/view?usp=sharing)

Una volta scaricati, posiziona i file nella **stessa cartella** dei file `.py`.

---

## 🤖 Modelli ML utilizzati

| Modello | Task |
|---|---|
| **Random Forest** | Predizione consumo carburante |
| **XGBoost** | Predizione consumo carburante (comparativo) |
| **Classificatore supervisionato** | Classificazione stile di guida (aggressivo / moderato / prudente) |

La variabile target per il consumo è `fuel_Delta` — per dettagli sul calcolo si rimanda alla tesi.

---

## 📄 Tesi

Il documento completo della tesi è disponibile nella cartella [`Tesi_di_laurea/`](./Tesi_di_laurea/).

**Titolo:** *"Un sistema interattivo per la predizione del consumo di carburante in flotte aziendali tramite Machine Learning"*
**Autore:** Alessio Napoli — UNISA, A.A. 2024/2025

---

## 📬 Contatti

**Alessio Napoli**
🔗 [LinkedIn](https://www.linkedin.com/in/alessio-napoli-748b5329a/)
📧 [napolialessio17@gmail.com](mailto:napolialessio17@gmail.com)

---

## 📝 Licenza

Distribuito sotto licenza **MIT**. Vedi [`LICENSE`](./LICENSE) per i dettagli.

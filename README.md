# ML_Fuel_Analisys-Predictions
Si nota che il presente progetto è stato l'oggetto della mia Tesi di Laurea Triennale in Ingegneria Informatica. 
Questo progetto permette, una volta inserito un database, di predire il consumo di carburante in una flotta di veicoli aziendali tramite modelli di Machine Learning. 
L'obiettivo del progetto è ridurre la spesa del carburante, in modo da rendere le aziende di logistica e trasporti più efficienti.

*
*
*

ISTRUZIONI PER IL CORRETTO FUNZIONAMENTO DEL PROGRAMMA:
Per prima cosa bisogna creare un ambiente virtuale (venv) sul proprio applicativo (ad esempio vsCode). 
Dopodichè è necessario installare le seguenti librerie nel venv (in versione python 3.10):
- streamlit
- pandas
- plotly
- psycopg2
- joblib
- matplotlib
- sklearn
- numpy
- seaborn

*

Una volta fatto ciò, bisogna caricare due database. Il primo, di OSM_Italy, contiene tutti i limiti di velocità di tutte le strade italiane (Usato per la comparazione della velocità del veicolo sulla strada su cui stava viaggiando al momento della registrazione). Il secondo, invece, è un database contenente varie informazioni riguardo alcuni veicoli, raccolte dai sensori di bordo. Il sistema processa queste informazioni in modo da utilizzare solo quelle più importanti per il suo scopo.
I database sono reperibili qui:
                                https://drive.google.com/file/d/1npi5BAJr2ODFzVTeLKKcjwHq6fRPK0CB/view?usp=sharing
Una volta scaricati vanno inseriti nella stessa cartella in cui sono presenti i file .py.

*

Dopodichè, è necessario semplicemente runnare da terminale il file dashboard.py, tramite comando:
                                  streamlit dashboard.py

Ogni file ha una specifica funzione strutturale per far funzionare il sistema completo:
- Il file associazione_limiti.py permette di utilizzare il database contenente i limiti stradali.
- Il file consumo_carburante.py permette di calcolare il consumo di carburante tramite la variabile fuel_Delta (spiegazione reperibile nel file della tesi).
- I file Driving_Style_predictor.py e stile_guida.py permettono di assegnare un punteggio allo stile di guida di un conducente di un determianto veicolo tramite analisi delle informazioni raccolte dai sensori di bordo. Il punteggio serve a etichettare il guidatore come aggressivo, moderato o prudente, in modo da predirre l'usura delle componenti del veicolo.
- I file dashborad.py e dashboard_Can_fuel.py permettono, rispettivamente, di interagire con la dashboard relativa alla predizione dello stile di guida e al consumo di carburante.

Per ulteriori informazioni, sia sul codice che sul suo funzionamento, nella repository è presente anche il file della mia Tesi di Laurea.

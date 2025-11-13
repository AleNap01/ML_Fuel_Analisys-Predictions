import psycopg2
import psycopg2.extras
import pandas as pd
import numpy as np

DEFAULT_SPEEDS = {
    "motorway": 130,
    "motorway_link": 90,
    "trunk": 110,
    "trunk_link": 90,
    "primary": 90,
    "primary_link": 70,
    "secondary": 70,
    "secondary_link": 60,
    "tertiary": 50,
    "tertiary_link": 50,
    "residential": 50,
    "living_street": 30,
    "service": 30,
    "unclassified": -1,
    "busway": 50,
    "bus_stop": -1,
    "construction": -1,
    "corridor": -1,
    "cycleway": 20,
    "emergency_bay": -1,
    "footway": -1,
    "path": -1,
    "pedestrian": 10,
    "platform": -1,
    "proposed": -1,
    "razed": -1,
    "rest_area": 10,
    "road": 50,
    "services": 30,
    "steps": -1,
    "track": 30
}

# Mappatura dei tipi di strada alle 4 categorie principali
STRADA_CATEGORIE = {
    # Autostrade
    "motorway": "autostrada",
    "motorway_link": "autostrada",
    "trunk": "autostrada",
    "trunk_link": "autostrada",
    
    # Extraurbane
    "primary": "extraurbana",
    "primary_link": "extraurbana",
    "secondary": "extraurbana",
    "secondary_link": "extraurbana",
    
    # Urbane secondarie
    "tertiary": "urbana_secondaria",
    "tertiary_link": "urbana_secondaria",
    "road": "urbana_secondaria",
    
    # Urbane
    "residential": "urbana",
    "living_street": "urbana",
    "service": "urbana",
    "services": "urbana",
    "busway": "urbana",
    "pedestrian": "urbana",
    "rest_area": "urbana",
    "track": "urbana",
    "cycleway": "urbana"
}

# Pesi per categoria di strada
PESI_PER_CATEGORIA = {
    "autostrada": {
        "superamenti_limite": 3,
        "frenate_brusche": 5,
        "accelerazioni_brusche": 2,
        "fluttuazioni_brusche": 4,
        "sterzate_forti": 5,
        "uso_alto_rpm": 1,
        "motore_acceso_fermo": 2
    },
    "extraurbana": {
        "superamenti_limite": 4,
        "frenate_brusche": 4,
        "accelerazioni_brusche": 3,
        "fluttuazioni_brusche": 3,
        "sterzate_forti": 4,
        "uso_alto_rpm": 2,
        "motore_acceso_fermo": 2
    },
    "urbana_secondaria": {
        "superamenti_limite": 5,
        "frenate_brusche": 3,
        "accelerazioni_brusche": 4,
        "fluttuazioni_brusche": 3,
        "sterzate_forti": 3,
        "uso_alto_rpm": 3,
        "motore_acceso_fermo": 3
    },
    "urbana": {
        "superamenti_limite": 6,
        "frenate_brusche": 2,
        "accelerazioni_brusche": 5,
        "fluttuazioni_brusche": 2,
        "sterzate_forti": 2,
        "uso_alto_rpm": 4,
        "motore_acceso_fermo": 4
    },
    "undefined": {
        "superamenti_limite": 5,
        "frenate_brusche": 4,
        "accelerazioni_brusche": 3,
        "fluttuazioni_brusche": 3,
        "sterzate_forti": 2,
        "uso_alto_rpm": 2,
        "motore_acceso_fermo": 1
    }
}

# Moltiplicatori per orario di guida
MOLTIPLICATORI_ORARIO = {
    "notte": 1.3,      # 22:00-06:00 - Più pericoloso
    "alba": 1.1,       # 06:00-08:00 - Leggermente più pericoloso
    "giorno": 1.0,     # 08:00-18:00 - Normale
    "sera": 1.1,       # 18:00-22:00 - Leggermente più pericoloso (traffico)
}

# Moltiplicatori per tipo di sessione
MOLTIPLICATORI_SESSIONE = {
    "breve": 1.1,      # < 30 minuti - Più aggressivo
    "normale": 1.0,    # 30-120 minuti - Normale
    "lunga": 1.2,      # > 120 minuti - Più pericoloso (stanchezza)
}

def get_fascia_oraria(hour):
    """Determina la fascia oraria"""
    if 6 <= hour < 8:
        return "alba"
    elif 8 <= hour < 18:
        return "giorno"
    elif 18 <= hour < 22:
        return "sera"
    else:
        return "notte"

def get_tipo_sessione(durata_minuti):
    """Determina il tipo di sessione in base alla durata"""
    if durata_minuti < 30:
        return "breve"
    elif durata_minuti <= 120:
        return "normale"
    else:
        return "lunga"

# Connessione al database
conn = psycopg2.connect(
    dbname="tirocinio_completo",
    user="alessionapoli",
    host="localhost",
    port="5432"
)
cur = conn.cursor()

print("Caricamento dati e analisi sessioni...")

# Query per recuperare tutti i dati necessari
query = """
SELECT id, timestamp, ident, battery_voltage, can_engine_rpm, can_pedal_brake_status, 
       can_throttle_pedal_level, engine_ignition_status, movement_status, position_altitude,
       position_direction, position_latitude, position_longitude, position_satellites,
       position_speed, vehicle_mileage, din_1, limite_velocita, tipo_strada
FROM telemetry_temp 
WHERE
    din_1 = true 
    AND engine_ignition_status = true
    AND position_speed > 0
    AND movement_status = true
    AND can_engine_rpm BETWEEN 1 AND 8000
    AND position_latitude BETWEEN -90 AND 90
    AND position_longitude BETWEEN -180 AND 180
    AND position_altitude BETWEEN -500 AND 10000
    AND position_direction BETWEEN 1 AND 359
    AND position_satellites > 4
    AND can_vehicle_mileage < 2000000
ORDER BY ident, timestamp;
"""

df = pd.read_sql_query(query, conn)
print(f"Caricati {len(df)} record validi.")
# Parsing timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'])

print("Identificazione sessioni di guida...")

# Calcola differenza temporale tra record consecutivi
df['time_diff'] = df.groupby('ident')['timestamp'].diff().dt.total_seconds().fillna(0)

# Nuova sessione se tempo tra record supera 600s (10 min)
df['new_session'] = (df['time_diff'] > 600).astype(int)

# Crea ID sessione progressivo per ogni veicolo
df['session_id'] = df.groupby('ident')['new_session'].cumsum()

# Crea identificatore unico per ogni sessione
df['session_unique_id'] = df['ident'].astype(str) + '_' + df['session_id'].astype(str)

print("Analisi delle sessioni...")

# Raggruppa e calcola statistiche sessioni
sessioni = df.groupby(['ident', 'session_id']).agg(
    inizio=('timestamp', 'min'),
    fine=('timestamp', 'max'),
    distanza_km=('vehicle_mileage', lambda x: x.max() - x.min() if x.max() > x.min() else 0),
    num_record=('id', 'count')
).reset_index()

# Calcolo della durata effettiva
sessioni['durata_minuti'] = (sessioni['fine'] - sessioni['inizio']).dt.total_seconds() / 60

# Filtra solo sessioni valide
sessioni_valide = sessioni[
    (sessioni['durata_minuti'] > 5) &
    (sessioni['distanza_km'] > 0.1) &
    (sessioni['num_record'] >= 2)
]

print(f"Sessioni valide identificate: {len(sessioni_valide)}")
print(f"Sessioni per veicolo - Media: {sessioni_valide.groupby('ident').size().mean():.1f}")

# Aggiungi informazioni orarie e di sessione ai dati originali
df['ora'] = df['timestamp'].dt.hour
df['fascia_oraria'] = df['ora'].apply(get_fascia_oraria)

# Merge con informazioni sessioni
sessioni_info = sessioni[['ident', 'session_id', 'durata_minuti']].copy()
sessioni_info['tipo_sessione'] = sessioni_info['durata_minuti'].apply(get_tipo_sessione)

df = df.merge(sessioni_info, on=['ident', 'session_id'], how='left')
df['tipo_sessione'] = df['tipo_sessione'].fillna('normale')

print("Analisi comportamenti di guida...")

# Aggiungi categoria strada
def get_categoria_strada(tipo_strada):
    if pd.isna(tipo_strada) or tipo_strada == 'unclassified' or tipo_strada not in STRADA_CATEGORIE:
        return 'undefined'
    return STRADA_CATEGORIE[tipo_strada]

df['categoria_strada'] = df['tipo_strada'].apply(get_categoria_strada)

# === ANALISI COMPORTAMENTI ===

# Superamento limiti di velocità (escludi limite_velocita = -1)
df_veloce = df[
    (df['limite_velocita'] > 0) & 
    (df['tipo_strada'] != 'unclassified') &
    (df['position_speed'] > df['limite_velocita'])
]

# Accelerazioni brusche
df_acc_brusche = df[df['can_throttle_pedal_level'] > 80]

# Frenate brusche
frenate_brusche = df[
    (df['can_pedal_brake_status'] == 1) &
    (df['position_speed'] > 50)
]

# Fluttuazioni brusche di velocità
df_fluttuazioni = df.sort_values(by=['ident', 'timestamp'])
df_fluttuazioni['speed_diff'] = df_fluttuazioni.groupby('ident')['position_speed'].diff().abs()
df_fluttuazioni['time_diff_flutt'] = df_fluttuazioni.groupby('ident')['timestamp'].diff().dt.total_seconds()
df_fluttuazioni['variazione_brusca'] = (df_fluttuazioni['speed_diff'] > 30) & (df_fluttuazioni['time_diff_flutt'] <= 10)
fluttuazioni = df_fluttuazioni[df_fluttuazioni['variazione_brusca']]

# Cambi di direzione bruschi
df_direzione = df.copy().sort_values(by=['ident', 'timestamp'])
df_direzione['direction_diff'] = df_direzione.groupby('ident')['position_direction'].diff().abs()
df_direzione['time_diff_dir'] = df_direzione.groupby('ident')['timestamp'].diff().dt.total_seconds()
df_direzione['sterzata_forte'] = (df_direzione['direction_diff'] > 45) & (df_direzione['time_diff_dir'] <= 5)
sterzate_forti = df_direzione[df_direzione['sterzata_forte']]

# Uso ad alti regimi (RPM)
df_rpm_alti = df[df['can_engine_rpm'] > 4000]

# Motore acceso ma veicolo fermo
df_acc_fermo = df[(df['can_engine_rpm'] > 0) & (df['position_speed'] == 0)]

print("Calcolo punteggi pesati...")

# === CALCOLO PUNTEGGI PESATI ===
def calcola_punteggio_comportamento(comportamento_df, nome_comportamento):
    """Calcola punteggio per un comportamento specifico considerando orario e sessione"""
    
    if comportamento_df.empty:
        return pd.DataFrame({'ident': df['ident'].unique(), f'punteggio_{nome_comportamento}': 0})
    
    # Raggruppa per veicolo, categoria strada, fascia oraria e tipo sessione
    grouped = comportamento_df.groupby([
        'ident', 'categoria_strada', 'fascia_oraria', 'tipo_sessione'
    ]).size().reset_index(name='count')
    
    # Applica moltiplicatori
    grouped['moltiplicatore_orario'] = grouped['fascia_oraria'].map(MOLTIPLICATORI_ORARIO)
    grouped['moltiplicatore_sessione'] = grouped['tipo_sessione'].map(MOLTIPLICATORI_SESSIONE)
    
    # Calcola punteggio pesato
    punteggi = []
    for _, row in grouped.iterrows():
        categoria = row['categoria_strada']
        peso_base = PESI_PER_CATEGORIA[categoria][nome_comportamento]
        punteggio = (row['count'] * peso_base * 
                    row['moltiplicatore_orario'] * 
                    row['moltiplicatore_sessione'])
        punteggi.append({
            'ident': row['ident'],
            'punteggio_parziale': punteggio
        })
    
    punteggi_df = pd.DataFrame(punteggi)
    punteggi_totali = punteggi_df.groupby('ident')['punteggio_parziale'].sum().reset_index()
    punteggi_totali.columns = ['ident', f'punteggio_{nome_comportamento}']
    
    return punteggi_totali

# Calcola punteggi per ogni comportamento
punteggi_comportamenti = []

comportamenti = [
    (df_veloce, 'superamenti_limite'),
    (df_acc_brusche, 'accelerazioni_brusche'),
    (frenate_brusche, 'frenate_brusche'),
    (fluttuazioni, 'fluttuazioni_brusche'),
    (sterzate_forti, 'sterzate_forti'),
    (df_rpm_alti, 'uso_alto_rpm'),
    (df_acc_fermo, 'motore_acceso_fermo')
]

for comportamento_df, nome in comportamenti:
    punteggio = calcola_punteggio_comportamento(comportamento_df, nome)
    punteggi_comportamenti.append(punteggio)

# Unisci tutti i punteggi
df_stile = pd.DataFrame({'ident': df['ident'].unique()})
for punteggio_df in punteggi_comportamenti:
    df_stile = df_stile.merge(punteggio_df, on='ident', how='left')

df_stile.fillna(0, inplace=True)

# Normalizzazione per numero totale di record
record_per_veicolo = df.groupby('ident').size().reset_index(name='tot_record')
df_stile = df_stile.merge(record_per_veicolo, on='ident', how='left')

# Normalizza i punteggi
punteggio_cols = [col for col in df_stile.columns if col.startswith('punteggio_')]
for col in punteggio_cols:
    df_stile[col + '_norm'] = df_stile[col] / df_stile['tot_record']

# Calcola punteggio finale
df_stile['punteggio_finale'] = df_stile[[col for col in df_stile.columns if col.endswith('_norm') and col.startswith('punteggio_')]].sum(axis=1)

# Classificazione stile guida
q1 = df_stile['punteggio_finale'].quantile(0.33)
q2 = df_stile['punteggio_finale'].quantile(0.66)

def assegna_stile(p):
    if p <= q1:
        return 0  # prudente
    elif p <= q2:
        return 1  # normale
    else:
        return 2  # aggressivo

df_stile['stile_guida'] = df_stile['punteggio_finale'].apply(assegna_stile)

# Aggiungi statistiche sessioni per veicolo
stats_sessioni = sessioni_valide.groupby('ident').agg(
    num_sessioni=('session_id', 'count'),
    durata_media=('durata_minuti', 'mean'),
    durata_totale=('durata_minuti', 'sum'),
    distanza_totale=('distanza_km', 'sum')
).reset_index()

df_stile = df_stile.merge(stats_sessioni, on='ident', how='left')
df_stile[['num_sessioni', 'durata_media', 'durata_totale', 'distanza_totale']] = df_stile[['num_sessioni', 'durata_media', 'durata_totale', 'distanza_totale']].fillna(0)

print("Aggiornamento database...")

# === AGGIORNAMENTO DATABASE ===
def check_column_exists(cursor, table_name, column_name):
    cursor.execute("""
        SELECT EXISTS (
            SELECT 1 FROM information_schema.columns 
            WHERE table_name=%s AND column_name=%s
        );
    """, (table_name, column_name))
    return cursor.fetchone()[0]

# Aggiungi la colonna se non esiste
if not check_column_exists(cur, 'telemetry_temp', 'stile_guida'):
    try:
        cur.execute("""
            ALTER TABLE telemetry_temp
            ADD COLUMN stile_guida INTEGER;
        """)
        conn.commit()
        print("Colonna 'stile_guida' aggiunta con successo.")
    except Exception as e:
        conn.rollback()
        print("Errore nell'aggiungere la colonna:", e)

# Aggiorna lo stile_guida
update_query = """
    UPDATE telemetry_temp AS t
    SET stile_guida = s.stile_guida
    FROM (VALUES %s) AS s(ident, stile_guida)
    WHERE t.ident = s.ident;
"""

update_data = list(zip(df_stile['ident'], df_stile['stile_guida'].astype(int)))
psycopg2.extras.execute_values(cur, update_query, update_data)
conn.commit()

# === STATISTICHE FINALI ===
print("\n" + "="*50)
print("ANALISI COMPLETATA - STATISTICHE FINALI")
print("="*50)

print(f"\n📊 SESSIONI DI GUIDA:")
print(f"- Sessioni valide totali: {len(sessioni_valide)}")
print(f"- Durata media sessione: {sessioni_valide['durata_minuti'].mean():.1f} minuti")
print(f"- Distanza media per sessione: {sessioni_valide['distanza_km'].mean():.2f} km")

print(f"\n🕐 DISTRIBUZIONE PER FASCIA ORARIA:")
fascia_stats = df.groupby('fascia_oraria').size()
for fascia, count in fascia_stats.items():
    perc = (count / len(df)) * 100
    print(f"- {fascia.capitalize()}: {count:,} record ({perc:.1f}%)")

print(f"\n⏱️ DISTRIBUZIONE TIPI SESSIONE:")
sessione_stats = df.groupby('tipo_sessione').size()
for tipo, count in sessione_stats.items():
    perc = (count / len(df)) * 100
    print(f"- {tipo.capitalize()}: {count:,} record ({perc:.1f}%)")

print(f"\n🚗 STILI DI GUIDA:")
stili = ['Prudente', 'Normale', 'Aggressivo']
for i, stile in enumerate(stili):
    count = (df_stile['stile_guida'] == i).sum()
    perc = (count / len(df_stile)) * 100
    print(f"- {stile}: {count} veicoli ({perc:.1f}%)")

print(f"\n📈 PUNTEGGI:")
print(f"- Punteggio medio: {df_stile['punteggio_finale'].mean():.4f}")
print(f"- Q1 (Prudente ≤): {q1:.4f}")
print(f"- Q2 (Normale ≤): {q2:.4f}")

print(f"\n🚨 COMPORTAMENTI RISCHIOSI:")
print(f"- Superamenti limite: {df_veloce.shape[0]:,} eventi")
print(f"- Frenate brusche: {frenate_brusche.shape[0]:,} eventi")
print(f"- Accelerazioni brusche: {df_acc_brusche.shape[0]:,} eventi")

# Top 5 veicoli più aggressivi
top_aggressivi = df_stile.nlargest(5, 'punteggio_finale')[['ident', 'punteggio_finale', 'num_sessioni', 'durata_totale']]
print(f"\n🔴 TOP 5 VEICOLI PIÙ AGGRESSIVI:")
for _, row in top_aggressivi.iterrows():
    print(f"- Veicolo {row['ident']}: Punteggio {row['punteggio_finale']:.4f}, {row['num_sessioni']:.0f} sessioni, {row['durata_totale']:.0f}min totali")

cur.close()
conn.close()

print(f"\n✅ Analisi completata e database aggiornato!")
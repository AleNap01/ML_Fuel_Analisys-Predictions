import psycopg2
from psycopg2.extras import execute_batch
from shapely.wkb import loads as load_wkb
from shapely.errors import ShapelyError
import re
from tqdm import tqdm

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


conn = psycopg2.connect(
    dbname="tirocinio_full",
    user="emanuele",
    password="Ciccone1",
    host="localhost",
    port="5432"
)

osm_conn = psycopg2.connect(
    dbname="osm_italy",
    user="emanuele",
    password="Ciccone1",
    host="localhost",
    port="5432"
)
osm_conn.autocommit = True
osm_cursor = osm_conn.cursor()

def get_speed_limit(lat, lon):
    query = """
        SELECT tags->'maxspeed' AS maxspeed, highway, ST_AsEWKB(way) AS geom
        FROM planet_osm_line
        WHERE highway IS NOT NULL AND way IS NOT NULL AND NOT ST_IsEmpty(way)
        ORDER BY way <-> ST_Transform(ST_SetSRID(ST_MakePoint(%s, %s), 4326), 3857)
        LIMIT 1;
    """
    try:
        osm_cursor.execute(query, (lon, lat))
        row = osm_cursor.fetchone()
        if not row:
            return -1, 'unknown'
        maxspeed, highway, geom_wkb = row
        if not geom_wkb:
            return -1, 'unknown'
        try:
            _ = load_wkb(bytes(geom_wkb))
        except ShapelyError:
            return -1, 'unknown'
        speed = None
        if maxspeed:
            nums = re.findall(r'\d+', maxspeed)
            if nums:
                speed = int(nums[0])
        if speed is None:
            speed = DEFAULT_SPEEDS.get(highway, -1)
        return speed, highway or 'unknown'
    except Exception:
        return -1, 'unknown'

def main():
    cur = conn.cursor()
    # Aggiungi colonne se non esistono
    cur.execute("""
        ALTER TABLE telemetry_temp 
        ADD COLUMN IF NOT EXISTS limite_velocita INTEGER,
        ADD COLUMN IF NOT EXISTS tipo_strada TEXT;
    """)
    conn.commit() 

    # Prendi righe da aggiornare
    cur.execute("""
        SELECT id, position_latitude, position_longitude
        FROM telemetry_temp
        WHERE position_latitude IS NOT NULL AND position_longitude IS NOT NULL 
    """)
    rows = cur.fetchall()
    print(f"Processo {len(rows)} righe per aggiornare limite_velocita e tipo_strada...")

    # Creo tabella temporanea
    cur.execute("""
        CREATE TEMP TABLE temp_speed_limits (
            id bigint PRIMARY KEY,
            limite_velocita integer,
            tipo_strada text
        );
    """)

    # Calcolo speed limits per ogni riga
    results = []
    for row in tqdm(rows, desc="Calcolo speed limit"):
        id, lat, lon = row
        speed, road_type = get_speed_limit(lat, lon)
        results.append((id, speed, road_type))

    # Inserisco dati nella tabella temporanea (batch)
    insert_query = "INSERT INTO temp_speed_limits (id, limite_velocita, tipo_strada) VALUES (%s, %s, %s)"
    execute_batch(cur, insert_query, results, page_size=1000)

    # Faccio un solo UPDATE joinando la tabella temporanea
    update_query = """
        UPDATE telemetry_temp t
        SET limite_velocita = tsl.limite_velocita,
            tipo_strada = tsl.tipo_strada
        FROM temp_speed_limits tsl
        WHERE t.id = tsl.id;
    """
    cur.execute(update_query)

    conn.commit()  # commit finale

    print("Aggiornamento completato.")

    cur.close()
    osm_cursor.close()
    conn.close()
    osm_conn.close()


if __name__ == "__main__":
    main()

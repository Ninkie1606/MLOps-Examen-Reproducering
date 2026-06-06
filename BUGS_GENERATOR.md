# Bug Generator — Instructies voor een nieuwe agent

Deze repo is een **opgelost** MLOps-oefenexamen. Jouw taak is om **7 bugs** te introduceren zodat een student ze kan vinden en oplossen.

> **Hou het niveau consistent** — de bugs moeten subtiel maar vindbaar zijn, vergelijkbaar met typische MLOps-fouten.

---

## Output bestanden

Na het introduceren van alle bugs, maak je **twee nieuwe bestanden** aan in de root van de repo:

### 1. `SYMP.md` — voor de student (symptomen-only)

Bevat enkel de **symptomen** zoals een gebruiker/student ze zou ervaren. Geen code, geen hints, geen oplossingen. Dit bestand mag de student meteen lezen.

**Structuur:**

```markdown
# Symptomen

## Symptoom 1 — <korte titel>

> <beschrijving van het probleem zoals gemeld door een gebruiker>
```

Gebruik een **andere formulering** dan wat hieronder in de bug-omschrijvingen staat. Beschrijf het alsof een stakeholder een issue meldt.
Elk symptoom is één paragraaf.

### 2. `BUGS.md` — voor de docent (antwoorden)

Bevat de **volledige uitleg** per bug: exacte bestand, lijn, wat er fout is en wat de correcte code moet zijn.
Dit mag de student **niet** zien — dit is om na te kijken.

**Structuur:**

```markdown
# Bug-antwoorden

## Bug 1 — <titel>

- **Bestand:** `<pad>` (lijn X)
- **Fout:** <wat er fout staat>
- **Oorzaak:** <waarom dit een probleem geeft>
- **Oplossing:** <wat de correcte code moet zijn>
```

---

## Werkmethode

1. Doorloop elke bug hieronder in volgorde
2. Voer de beschreven wijziging uit (de **buggy** code)
3. Sla het bestand op
4. Na alle 7 bugs: maak `SYMP.md` en `BUGS.md` aan
5. `git add . && git commit -m "<beschrijving>"` (enkel als gevraagd)

---

## Bug 1 — MLflow tracking URI verwijderd

**Symptoom:** Trainingsruns zijn onzichtbaar in het MLflow dashboard.
De scripts voltooien zonder fouten, maar er verschijnt geen enkel experiment of run.

### Aan te passen bestanden

**`l1-train-and-deploy/train.py`** (lijn 8):

- **Verwijder** de volledige lijn `mlflow.set_tracking_uri("http://experiment-tracking:5000")`

**`l1-train-and-deploy/hpo.py`** (lijn 10):

- **Verwijder** de volledige lijn `mlflow.set_tracking_uri("http://experiment-tracking:5000")`

**`l1-train-and-deploy/register.py`** (lijn 24):

- **Verwijder** de volledige lijn `mlflow.set_tracking_uri("http://experiment-tracking:5000")`

> **Waarom dit werkt:** Zonder `set_tracking_uri()` default MLflow naar de lokale `./mlruns/` file store in de container. De MLflow-server gebruikt PostgreSQL als backend, dus die ziet nooit iets. Het dashboard blijft leeg.

---

## Bug 2 — Filter op verkeerde kolom (trip_distance i.p.v. duration)

**Symptoom:** Model traint op foute data. De filter gebruikt `trip_distance` in plaats van `duration`,
waardoor een andere subset ritten wordt geselecteerd. De RMSE is ongewoon hoog.

### Aan te passen bestand

**`l1-train-and-deploy/preprocess.py`** (lijn 20):

```python
# HUIDIG (correct):
df = df[(df.duration >= 1) & (df.duration <= 60)]

# BUGGY:
df = df[(df.trip_distance >= 1) & (df.trip_distance <= 60)]
```

> **Waarom:** De `read_dataframe` functie filtert op `trip_distance` (afstand in mijlen) i.p.v. `duration` (tijd in minuten). Ritten met afstand 1-60 mijl blijven bewaard, niet ritten met duur 1-60 minuten — een heel andere selectie.
> Test `test_read_dataframe_filters_short_and_long_trips` faalt: verwacht 3 rijen maar krijgt 5.

---

## Bug 3 — Duration in register.py berekend in uren i.p.v. minuten

**Symptoom:** Predictie API draait en geeft getallen terug, maar een rit van 30 minuten wordt voorspeld als ~0.5.
De tijdsberekening bij het hertrainen in het registratiescript deelt door 3600 i.p.v. 60.

### Aan te passen bestand

**`l1-train-and-deploy/register.py`** (lijn 31):

```python
# HUIDIG (correct):
df.duration = df.duration.dt.total_seconds() / 60

# BUGGY:
df.duration = df.duration.dt.total_seconds() / 3600
```

> **Waarom:** `register.py` heeft een eigen `read_dataframe` functie die de duur berekent. Door `/ 3600` (seconden → uren) i.p.v. `/ 60` (seconden → minuten) wordt `duration` in uren opgeslagen. Het model traint op uren, maar de API wordt geacht minuten terug te geven. Voorspellingen zijn 60× te laag.

---

## Bug 4 — Experiment-naam met case-mismatch

**Symptoom:** Registratiescript crasht met "experiment not found".
Het HPO-experiment bestaat wel degelijk, maar de zoekopdracht gebruikt een andere schrijfwijze.

### Aan te passen bestand

**`l1-train-and-deploy/hpo.py`** (lijn 20):

```python
# HUIDIG (correct):
mlflow.set_experiment("random-forest-hpo")

# BUGGY:
mlflow.set_experiment("Random-Forest-HPO")
```

> **Waarom:** `register.py` zoekt met `client.get_experiment_by_name("random-forest-hpo")` (lowercase). MLflow experiment names zijn **case-sensitive**. Het experiment werd aangemaakt als `"Random-Forest-HPO"` (met hoofdletters), dus `get_experiment_by_name("random-forest-hpo")` geeft `None` terug → `AttributeError: 'NoneType' object has no attribute 'experiment_id'`.

---

## Bug 5 — Geregistreerde modelnaam komt niet overeen

**Symptoom:** Prediction API crasht bij opstarten met de melding dat een model niet gevonden kan worden in het Model Registry.

### Aan te passen bestand

**`l1-train-and-deploy/register.py`** (lijn 100):

```python
# HUIDIG (correct):
mlflow.register_model(model_uri, name="rf-best-model")

# BUGGY:
mlflow.register_model(model_uri, name="rf-prod-model")
```

> **Waarom:** Het model wordt geregistreerd als `"rf-prod-model"`. De predictie-API in `predict.py` probeert echter `"models:/rf-best-model/latest"` te laden. Dit model bestaat niet in de registry → `mlflow.exceptions.RestException: RESOURCE_DOES_NOT_EXIST: Could not find model with name rf-best-model`.

---

## Bug 6 — Worker pool naam mismatch in startPoolWorkers.sh

**Symptoom:** Batch container start zonder fouten, maar er verschijnen geen flows in het Prefect-dashboard.
De batch flow kan manueel gestart worden maar doet niets.

### Aan te passen bestand

**`l5-deploy-batch/startPoolWorkers.sh`** (lijnen 3-4):

```bash
# HUIDIG (correct):
prefect work-pool create --type process batch --overwrite
prefect worker start -p batch &

# BUGGY:
prefect work-pool create --type process batch-pool --overwrite
prefect worker start -p batch-pool &
```

> **Waarom:** `batch.py` serveert met `work_pool_name="batch"`. De work pool en worker in `startPoolWorkers.sh` gebruiken nu `"batch-pool"`. De worker luistert naar een andere pool dan waar de flow op serveert. De flow wordt nooit opgepikt.

> Let op: `l5-deploy-batch/Dockerfile` en `l5-deploy-batch/batch.py` blijven correct en worden niet aangepast.

---

## Bug 7 — Sorteervolgorde omgekeerd bij eindselectie

**Symptoom:** Het registratiescript doorloopt alle stappen zonder fouten, maar het geregistreerde model heeft consequent een hogere RMSE dan de best beschikbare HPO-run. De eindselectie sorteert verkeerd.

### Aan te passen bestand

**`l1-train-and-deploy/register.py`** (lijn 94):

```python
# HUIDIG (correct):
order_by=["metrics.rmse ASC"],

# BUGGY:
order_by=["metrics.rmse DESC"],
```

> **Waarom:** De eerste query (lijn 83) selecteert correct de top 5 beste HPO-runs (`ASC`). Deze worden hertraind en gelogd in het best-models experiment. De **tweede** query (lijn 94) selecteert de "beste" uit die hertrainde modellen — maar sorteert `DESC`, waardoor de hoogste RMSE (slechtste) wordt gekozen. Het slechtste hertrainde model wordt geregistreerd.

> Enkel lijn 94 wordt aangepast; lijn 83 blijft correct.

---

## Verificatie checklist

Na het introduceren van alle 7 bugs zou dit de verwachte situatie moeten zijn:

| #   | Symptoom                                      | Hoe testen                                                                                                                       |
| --- | --------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Geen runs in MLflow                           | `docker compose up --build`, open http://localhost:5000 → geen experimenten                                                      |
| 2   | Hoge RMSE door data-filter op verkeerde kolom | `python -m pytest l1-train-and-deploy/tests/` → test `test_read_dataframe_filters_short_and_long_trips` faalt (5 i.p.v. 3 rijen) |
| 3   | Predicties 60× te laag                        | POST request naar API: `curl -X POST http://localhost:9696/predict ...` → `duration` is ~60× te klein                            |
| 4   | Registratie crasht                            | `docker compose up --build` → train-deploy container faalt bij register.py met "experiment not found"                            |
| 5   | API start niet                                | `docker compose up --build` → web-service container faalt met "model rf-best-model not found"                                    |
| 6   | Geen batch flows                              | Prefect dashboard (http://localhost:4200) toont geen flows; worker luistert op verkeerde pool                                    |
| 7   | Suboptimaal model                             | Vergelijk RMSE van geregistreerd model met beste HPO-run: geregistreerde RMSE is hoger                                           |

---

## Testen dat de bugs werken

```bash
python -m pytest l1-train-and-deploy/tests/ -v
```

- Test `test_read_dataframe_filters_short_and_long_trips` faalt bij Bug 2 (expected 3, got 5)
- De andere tests moeten slagen

Voor end-to-end testen: `docker compose up --build` en check of services crashen of verkeerde output geven.

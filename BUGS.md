# Bug-antwoorden

## Bug 1 — MLflow tracking URI verwijderd
- **Bestand:** `l1-train-and-deploy/train.py` (lijn 8), `l1-train-and-deploy/hpo.py` (lijn 10), `l1-train-and-deploy/register.py` (lijn 24)
- **Fout:** De regel `mlflow.set_tracking_uri("http://experiment-tracking:5000")` is verwijderd uit alle drie de bestanden.
- **Oorzaak:** Zonder `set_tracking_uri()` default MLflow naar de lokale `./mlruns/` file store in de container. De MLflow-server gebruikt PostgreSQL als backend, dus die ziet nooit iets. Het dashboard blijft leeg.
- **Oplossing:** Voeg `mlflow.set_tracking_uri("http://experiment-tracking:5000")` terug in op lijn 8 van `train.py`, lijn 10 van `hpo.py` en lijn 24 van `register.py`.

## Bug 2 — Filter op verkeerde kolom (trip_distance i.p.v. duration)
- **Bestand:** `l1-train-and-deploy/preprocess.py` (lijn 20)
- **Fout:** `df = df[(df.trip_distance >= 1) & (df.trip_distance <= 60)]`
- **Oorzaak:** De `read_dataframe` functie filtert op `trip_distance` (afstand in mijlen) i.p.v. `duration` (tijd in minuten). Ritten met afstand 1-60 mijl blijven bewaard, niet ritten met duur 1-60 minuten — een heel andere selectie. Test `test_read_dataframe_filters_short_and_long_trips` faalt: verwacht 3 rijen maar krijgt 5.
- **Oplossing:** Vervang door `df = df[(df.duration >= 1) & (df.duration <= 60)]`.

## Bug 3 — Duration in register.py berekend in uren i.p.v. minuten
- **Bestand:** `l1-train-and-deploy/register.py` (lijn 30)
- **Fout:** `df.duration = df.duration.dt.total_seconds() / 3600`
- **Oorzaak:** `register.py` heeft een eigen `read_dataframe` functie die de duur berekent. Door `/ 3600` (seconden → uren) i.p.v. `/ 60` (seconden → minuten) wordt `duration` in uren opgeslagen. Het model traint op uren, maar de API wordt geacht minuten terug te geven. Voorspellingen zijn 60× te laag.
- **Oplossing:** Vervang door `df.duration = df.duration.dt.total_seconds() / 60`.

## Bug 4 — Experiment-naam met case-mismatch
- **Bestand:** `l1-train-and-deploy/hpo.py` (lijn 19)
- **Fout:** `mlflow.set_experiment("Random-Forest-HPO")`
- **Oorzaak:** `register.py` zoekt met `client.get_experiment_by_name("random-forest-hpo")` (lowercase). MLflow experiment names zijn **case-sensitive**. Het experiment werd aangemaakt als `"Random-Forest-HPO"` (met hoofdletters), dus `get_experiment_by_name("random-forest-hpo")` geeft `None` terug → `AttributeError: 'NoneType' object has no attribute 'experiment_id'`.
- **Oplossing:** Vervang door `mlflow.set_experiment("random-forest-hpo")`.

## Bug 5 — Geregistreerde modelnaam komt niet overeen
- **Bestand:** `l1-train-and-deploy/register.py` (lijn 99)
- **Fout:** `mlflow.register_model(model_uri, name="rf-prod-model")`
- **Oorzaak:** Het model wordt geregistreerd als `"rf-prod-model"`. De predictie-API in `predict.py` probeert echter `"models:/rf-best-model/latest"` te laden. Dit model bestaat niet in de registry → `mlflow.exceptions.RestException: RESOURCE_DOES_NOT_EXIST: Could not find model with name rf-best-model`.
- **Oplossing:** Vervang door `mlflow.register_model(model_uri, name="rf-best-model")`.

## Bug 6 — Worker pool naam mismatch in startPoolWorkers.sh
- **Bestand:** `l5-deploy-batch/startPoolWorkers.sh` (lijnen 3-4)
- **Fout:** `prefect work-pool create --type process batch-pool --overwrite` en `prefect worker start -p batch-pool &`
- **Oorzaak:** `batch.py` serveert met `work_pool_name="batch"`. De work pool en worker in `startPoolWorkers.sh` gebruiken nu `"batch-pool"`. De worker luistert naar een andere pool dan waar de flow op serveert. De flow wordt nooit opgepikt.
- **Oplossing:** Vervang `batch-pool` door `batch` op beide lijnen.

## Bug 7 — Sorteervolgorde omgekeerd bij eindselectie
- **Bestand:** `l1-train-and-deploy/register.py` (lijn 93)
- **Fout:** `order_by=["metrics.rmse DESC"]`
- **Oorzaak:** De eerste query (lijn 82) selecteert correct de top 5 beste HPO-runs (`ASC`). Deze worden hertraind en gelogd in het best-models experiment. De **tweede** query (lijn 93) selecteert de "beste" uit die hertrainde modellen — maar sorteert `DESC`, waardoor de hoogste RMSE (slechtste) wordt gekozen. Het slechtste hertrainde model wordt geregistreerd.
- **Oplossing:** Vervang door `order_by=["metrics.rmse ASC"]`.

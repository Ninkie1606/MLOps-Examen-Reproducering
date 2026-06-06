# Symptomen

## Symptoom 1 — MLflow-dashboard toont geen runs

> Ik heb alle scripts doorlopen zonder fouten, maar het MLflow-dashboard op poort 5000 is helemaal leeg. Er staat geen enkel experiment of run, ook al draaien de containers zonder problemen.

## Symptoom 2 — RMSE is veel hoger dan verwacht

> Na het trainen van het model valt de RMSE extreem hoog uit vergeleken met de vorige run. De data ziet er normaal uit, maar de filters lijken niet goed te werken.

## Symptoom 3 — Voorspelde duur is ongeveer 60× te laag

> Wanneer ik een rit van 30 minuten door de predictie-API laat voorspellen, krijg ik een waarde rond 0.5 terug. Het model geeft dus getallen, maar ze kloppen niet met de realiteit.

## Symptoom 4 — Registratie van model mislukt met "experiment not found"

> Het registratiescript (register.py) crasht tijdens de Docker-build met een foutmelding dat het experiment niet gevonden kan worden. Het experiment werd nochtans aangemaakt tijdens de HPO-stap.

## Symptoom 5 — Predictie-API start niet op

> De webservice-container voor predicties blijft herstarten en crasht bij het opstarten met de melding dat een model niet bestaat in de registry. Het model is wel degelijk geregistreerd.

## Symptoom 6 — Batch-container start maar geen flows in Prefect

> De batch-container draait zonder fouten, maar in het Prefect-dashboard op poort 4200 verschijnen er geen flows. Handmatig een flow starten geeft ook geen resultaat; de worker doet niets.

## Symptoom 7 — Geregistreerd model is slechter dan de beste HPO-run

> Het registratiescript doorloopt alle stappen zonder fouten, maar het uiteindelijk geregistreerde model heeft een hogere RMSE dan de beste run uit de HPO-optimalisatie. Het lijkt alsof het verkeerde model wordt gekozen.

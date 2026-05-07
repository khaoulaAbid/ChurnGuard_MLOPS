"""Inspecte le contenu du registry MLflow pour le modele 'churnguard'.

Affiche les versions, leurs sources, les aliases au niveau du modele
enregistre, puis tente de charger via l'alias `production` pour valider
la chaine alias -> version -> artefact.
"""

from __future__ import annotations

import os
import traceback

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

MODEL_NAME = "churnguard"
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")


def main() -> None:
    """Affiche les versions, les aliases et tente un chargement reel."""
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    print(f"[i] tracking uri : {TRACKING_URI}")

    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        print(f"Aucune version trouvee pour le modele '{MODEL_NAME}'")
        return

    print(f"\nModele '{MODEL_NAME}' - {len(versions)} version(s) :")
    print("-" * 80)
    for v in sorted(versions, key=lambda x: int(x.version)):
        try:
            print(f"version : {v.version}")
            print(f"  source : {v.source!r}")
            print(f"  status : {v.status}")
            print(f"  run_id : {v.run_id}")
        except Exception as exc:
            print(f"  [err] iteration version : {exc}")
            traceback.print_exc()
        print("-" * 80)

    try:
        rm = client.get_registered_model(MODEL_NAME)
        aliases = getattr(rm, "aliases", None) or {}
        print("\nAliases (vue RegisteredModel) :")
        if aliases:
            for alias_name, alias_version in aliases.items():
                print(f"  {alias_name!r:>14} -> v{alias_version}")
        else:
            print("  (aucun)")
    except Exception as exc:
        print(f"[err] get_registered_model : {exc}")

    print("\n[..] resolution de l'alias 'production' via API...")
    try:
        mv = client.get_model_version_by_alias(MODEL_NAME, "production")
        print(f"[ok] alias production -> v{mv.version}, source={mv.source!r}")
    except Exception as exc:
        print(f"[err] get_model_version_by_alias('production') : {exc}")

    print("\n[..] tentative de chargement reel : models:/churnguard@production")
    try:
        model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@production")
        print(f"[ok] modele charge : {type(model).__name__}")
    except Exception as exc:
        print(f"[err] load_model echoue : {exc}")
        traceback.print_exc()


if __name__ == "__main__":
    main()

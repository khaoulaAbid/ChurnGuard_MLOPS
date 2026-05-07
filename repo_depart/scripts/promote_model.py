"""Promeut une version du modele ChurnGuard (aliases MLflow 3.x en priorite).

Usage :
    python scripts/promote_model.py
    python scripts/promote_model.py --version 3
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

from mlflow import MlflowClient

MODEL_NAME = "churnguard"
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")


def parse_args() -> argparse.Namespace:
    """Parse les arguments CLI."""
    parser = argparse.ArgumentParser(description="Promote ChurnGuard model to Production")
    parser.add_argument(
        "--version",
        type=int,
        default=None,
        help="Numero de version a promouvoir (par defaut: derniere version).",
    )
    return parser.parse_args()


def main() -> None:
    """Assigne les aliases staging et production sur la version cible."""
    args = parse_args()
    client = MlflowClient(tracking_uri=TRACKING_URI)

    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        print(f"[err] aucune version trouvee pour le modele '{MODEL_NAME}'", file=sys.stderr)
        sys.exit(1)

    if args.version is None:
        target = max(versions, key=lambda v: int(v.version))
    else:
        matches = [v for v in versions if int(v.version) == args.version]
        if not matches:
            print(f"[err] version {args.version} introuvable", file=sys.stderr)
            sys.exit(2)
        target = matches[0]

    version = int(target.version)
    print(f"[..] promotion de {MODEL_NAME} v{version}")

    # L'API registry attend souvent la version en str (REST).
    version_str = str(version)
    try:
        client.set_registered_model_alias(MODEL_NAME, "staging", version_str)
        client.set_registered_model_alias(MODEL_NAME, "production", version_str)
        print(f"[ok] alias 'staging' -> v{version}")
        print(f"[ok] alias 'production' -> v{version}")
        print(f'[hint] charger : mlflow.pyfunc.load_model("models:/{MODEL_NAME}@production")')
    except Exception as exc:
        print(f"[warn] aliases indisponibles ({exc}), fallback stages MLflow 2.x…")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=version_str,
                stage="Staging",
                archive_existing_versions=False,
            )
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=version_str,
                stage="Production",
                archive_existing_versions=True,
            )
        print(f"[ok] v{version} -> Staging puis Production (stages)")


if __name__ == "__main__":
    main()

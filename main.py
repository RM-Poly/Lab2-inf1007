"""Point d'entrée de l'application.

Lancer avec :  streamlit run main.py
(ou simplement  python main.py  / bouton Run de PyCharm : le script relance alors Streamlit.)
"""

import os
import sys

# Garantit que le dossier du projet est importable, peu importe d'où l'on lance l'app.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from streamlit import runtime  # noqa: E402

if runtime.exists():
    # Exécuté par le serveur Streamlit : on construit l'interface.
    from src.interface import lancer  # noqa: E402

    lancer()
elif __name__ == "__main__":
    # Exécuté avec « python main.py » : on démarre le serveur Streamlit sur ce fichier.
    from streamlit.web import cli  # noqa: E402

    sys.argv = ["streamlit", "run", os.path.abspath(__file__)]
    sys.exit(cli.main())

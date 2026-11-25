# 5294INOR6Y_G1

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Website

URL: https://oscarw27.github.io/5294INOR6Y_G1/
Ontology:https://oscarw27.github.io/5294INOR6Y_G1/ontology
Ontology viewer: https://oscarw27.github.io/5294INOR6Y_G1/ontology?viewer

```bash
python scripts/build_site_data.py
python -m http.server 8000 -d site
```
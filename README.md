# BhashaSignal

AI Early-Signal Detection from Multilingual Voice Observations.

## What this demo does
1. Accepts a short voice observation.
2. Uses Sarvam Speech-to-Text when `SARVAM_API_KEY` is configured.
3. Extracts structured fields locally.
4. Compares observations and detects repeated patterns.
5. Shows an explainable signal with supporting evidence.

## Run locally

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Optional Sarvam API setup:

```powershell
$env:SARVAM_API_KEY="YOUR_KEY"
streamlit run app.py
```

The demo dataset works without the key, so the UI can be presented before connecting the API.

## Demo story
Use the included observations to show how several individual reports about the same service issue become one collective early signal.

## Important
The demo detector is intentionally lightweight and transparent. It is a hackathon prototype, not a production incident-management system.

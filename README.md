RiskZero
----------------------------
A Streamlit app that turns a feature description into prioritized security test cases — automatically.

What it does
------------------------
You describe a feature (e.g. "User login with OAuth2 and JWT").
The app asks a local LLM (via Ollama) which OWASP Top 10:2021 vulnerabilities apply.
For each vulnerability, it looks up real CVEs and CVSS scores from the NVD database.
It generates detailed test cases and ranks them by priority (P1–P4) based on the CVSS score.
You can export everything as a JSON file.


Tech Stack
--------------------
Streamlit – web UI
Ollama – local LLM for vulnerability detection & test case generation
NVD API – real-world CVE / CVSS data
Hugging Face Datasets – fallback CVE dataset
ChromaDB – optional secure-coding pattern lookup
Plotly / Pandas – data display


Requirements
---------------------------
Python 3.9+
Ollama installed and running locally with a model pulled (e.g. ollama pull mistral)


Setup
--------------------
bashpip install streamlit requests pandas plotly datasets
ollama serve


Run
----------------------
bashstreamlit run app_fixed.py

Then open the local URL Streamlit gives you, type in a feature description, and click Generate Security Tests.



Sample Output
--------------------------------
See SampleOfOutput.json for an example of the generated test cases.

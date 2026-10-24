# fwa-signal — Medicare Prescribing Leads with Validated LLM Features
Tech: Python, PostgreSQL, SQL, FastAPI, scikit-learn, LLMs (Ollama), sqlglot, pytest, GitHub Actions
- Built a PostgreSQL pipeline over 1.33M Medicare Part D prescribers and the HHS-OIG exclusion list with data-quality gates, peer-group scoring, and a FastAPI service for program-integrity leads.
- Evaluated out of time against providers OIG excluded later: a model trained on 2021 data beat the volume baseline on 2022 data (AUC 0.651 vs 0.576, 95% CI of the gain +0.035 to +0.107), after finding and fixing a zero-inflation bug that had put the first score below random.
- Added LLM features with guardrails: plain-English questions to validated, read-only SQL (83% accuracy on 24 questions; 10/10 attack prompts contained, data unchanged) and case summaries whose numbers are checked against the database (15/15 passed).

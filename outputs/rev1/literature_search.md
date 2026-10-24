# Systematic literature and dataset search (rev1, Faz 11)

Addresses R1-a ("how is the claim that no ML study exists verified?"), R1-b/R1-h (novelty
versus prior physics-informed models), R2-14 (priority claims) and R3-5 (positioning).

Search date: 2026-10-08. Hit lists: `outputs/rev1/literature_search/*.json`
(re-create with `python3 scripts/rev1/literature_search.py`). Candidate BibTeX entries in
`outputs/rev1/literature_search/candidates.bib` were retrieved directly from DOI content
negotiation (Crossref/DataCite) or arXiv; none were typed by hand.

## 1. Protocol

| Item | Setting |
|---|---|
| Index (counts) | OpenAlex (title + abstract, boolean query); ~250 M works, includes Crossref, PubMed, arXiv, DergiPark |
| Datasets | DataCite (covers Zenodo, Mendeley Data, figshare, Harvard Dataverse, ICPSR, IEEE DataPort DOIs) |
| Complementary | Google Scholar / web search (English, Turkish, Chinese), arXiv API, Crossref |
| Period | no lower bound — 2026-10-08 |
| Inclusion | studies that predict or assess the meteorological suitability / go-no-go decision of (hot-air) balloon flights, with any statistical or ML method |
| Exclusion | stratospheric / scientific / planetary balloons, balloon trajectory planning, tourism marketing and satisfaction studies, medical use of the word "balloon" |

Queries (B = `"hot air balloon" OR "hot-air balloon" OR "hot air balloons" OR "hot-air balloons" OR "balloon flight" OR "balloon flights" OR "balloon tourism"`):

| ID | Query | Hits |
|---|---|---|
| Q1 | B AND (machine learning OR deep learning OR neural network OR artificial intelligence OR classification OR random forest OR XGBoost OR gradient boosting OR data-driven) | 42 |
| Q2 | B AND (weather OR meteorological OR meteorology OR wind OR fog OR forecast) | 513 |
| Q3 | (Cappadocia OR Kapadokya OR Göreme OR Nevşehir) AND (balloon OR balon) | 47 |
| Q4 | B AND (safety OR accident(s) OR incident OR risk) | 276 |
| Q5 | neural additive model(s) AND (physics OR physics-informed OR physics-guided OR domain knowledge OR monotonic(ity)) | 15 |
| Q6 | (physics-informed OR physics-guided) AND (additive model OR generalized additive OR neural additive) | 27 |

## 2. Screening (application domain, Q1–Q4)

| Stage | n |
|---|---|
| Records identified (Q1–Q4) | 878 (815 unique) |
| Pass automatic title screen (crewed/tourist ballooning; stratospheric, planetary, medical uses removed; `screened_titles.tsv`) | 107 |
| Weather → flight-suitability forecasting or assessment (title/abstract read) | 4 |
| Balloons used as wind sensors for NWP | 10 |
| **ML / data-driven prediction of the flight decision from operational flight records** | **0** |

The remaining 93 screened records are accident epidemiology, human factors, tourism
experience/marketing, engineering and legal studies.

Records on weather and flight decisions:

| Study | What it does | Difference from this work |
|---|---|---|
| Trimmel (1989), *Technical Soaring* 13(2):51–56 | Practical forecasting experience | Qualitative |
| Galvin (1999), *Meteorol. Appl.* 6(4):351–362, doi:10.1017/S1350482799001292 | Forecaster's practice note on balloon and airship forecasting in England | Qualitative, no data-driven model |
| Kacelowicz (2013), AMS Annual Meeting abstract | Forecasting support for balloon events | Conference abstract, no model |
| de Bruijn et al. (2020), *BAMS* 102(2), doi:10.1175/BAMS-D-19-0285.1; de Bruijn et al. (2023), *Meteorol. Appl.* 30, doi:10.1002/met.2128 | Balloons used as wind **sensors** for NWP | Opposite direction: balloons inform weather, not weather → flight decision |
| Duan et al. (2026), *Sustainable Development* (Hans) 16(9):150–163, doi:10.12677/sd.2026.169323 | Yunnan: "flyable days" from ERA5 with fixed multi-factor thresholds, AHP weights, K-means tiers | **Closest work.** Rule-based; no operational flight records, no learned model, no validation against real decisions |
| Aslaner et al. (2019), *Aerosp. Med. Hum. Perform.* 90(2):123–127, doi:10.3357/AMHP.5121.2019 | Accident rates in Cappadocia 2013–2017 | Safety epidemiology, no prediction |

Accident and human-factor studies (context for safety framing): Ballard et al. 2013
(doi:10.3357/ASEM.3719.2013), Filtness et al. 2014 (doi:10.3357/ASEM.3984.2014),
Azık et al. 2025 (doi:10.30518/jav.1697241). Turkish tourism literature on Cappadocia
ballooning (Q3, 47 records) addresses experience, satisfaction, risk perception and
business models; none models the flight decision.

## 3. Dataset search

DataCite query `("hot air balloon" OR "hot-air balloon" OR Cappadocia OR Kapadokya) AND (flight OR weather OR balloon)`: 131 records.
Only related datasets: (i) New Mexico balloon accident records (ICPSR / Mendeley, 2026; accidents, no
weather–decision pairs); (ii) **this study's own dataset**, Özcan, A., *Dataset on Weather Conditions
and Flight Decision Records for Hot-Air Balloons in Cappadocia Region (2023–2024)*, Mendeley Data,
doi:10.17632/mjv3zmxwsn (registered 2025-07-21, currently embargoed). No Kaggle/UCI dataset was found
through web search.

## 4. Method positioning (Q5–Q6 + reference works)

Prior work exists on each ingredient, so methodological novelty must be stated as the
**combination for an operational decision**, not as a first "physics-informed NAM":

| Line of work | Representative, verified references | Relation to PI-NAM |
|---|---|---|
| Physics-informed neural networks (PDE residual in loss) | Raissi et al. 2019 (JCP); Karniadakis et al. 2021 (Nat. Rev. Phys.) | Need a governing PDE and continuous fields; the daily go/no-go label has no governing equation, so PINN is not applicable as a baseline (R1-e) |
| Physics-guided / theory-guided ML | Karpatne et al. 2017 (TKDE); Willard et al. 2022 (ACM CSUR); Daw et al. 2020 (SDM, PGA-LSTM, monotonic density); Kashinath et al. 2021 (Phil. Trans. A); Reichstein et al. 2019 (Nature) | PI-NAM belongs here: physics as derived inputs + soft monotonic penalty |
| Interpretable additive models | Lou et al. 2013 (GA²M); Caruana et al. 2015; Nori et al. 2019 (EBM); Agarwal et al. 2021 (NAM); Yang et al. 2021 (GAMI-Net); Chang et al. 2022 (NODE-GAM); Radenović et al. 2022 (NBM) | PI-NAM is a NAM with structured interactions |
| Monotonic / knowledge-constrained NAMs | Chen & Ye 2022 (ICAIF, monotonic NAM, credit scoring); Rong et al. 2026 (AIH, knowledge-constrained NAM, hypoglycaemia); Hussaini 2026 (PINAGMM, ground motion; software, paper in press) | **Monotone/knowledge-constrained NAMs already exist** → do not claim "first physics-informed NAM" |
| Monotonic networks / lattices | You et al. 2017 (DLN); Liu et al. 2020 (certified monotonic NN) | Hard-constraint alternatives to our soft penalty |
| Interpretable models for high-stakes decisions | Rudin 2019 (Nat. Mach. Intell.) | Motivation |

What remains specific to this work: (1) the first learned model of real flight-authorization
decisions for balloon operations; (2) a differentiable aviation-physics layer (air density,
density altitude, lift proxy, dew-point depression) combined with **multi-day synoptic**
features inside an additive model; (3) evaluation under decision-time information sets and
time-aware protocols; (4) an open dataset and code.

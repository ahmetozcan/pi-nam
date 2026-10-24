"""Reproducible literature search for the novelty statement (R1-a, R2-14, R3-5).

Queries the open OpenAlex index (title + abstract search, boolean syntax) and stores
the full hit lists so the screening counts in outputs/rev1/literature_search.md can be
re-derived. Screening itself (title/abstract reading) is documented in that file.

Run from the repository root:  python3 scripts/rev1/literature_search.py
"""

import json
import logging
import time
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/rev1/literature_search"
BASE = "https://api.openalex.org/works"
logger = logging.getLogger("literature_search")

BALLOON = ('("hot air balloon" OR "hot-air balloon" OR "hot air balloons" OR "hot-air balloons" '
           'OR "balloon flight" OR "balloon flights" OR "balloon tourism")')
QUERIES: Dict[str, str] = {
    "Q1_balloon_ml": BALLOON + ' AND ("machine learning" OR "deep learning" OR "neural network" OR '
                     '"artificial intelligence" OR classification OR "random forest" OR XGBoost OR '
                     '"gradient boosting" OR "data-driven")',
    "Q2_balloon_weather": BALLOON + " AND (weather OR meteorological OR meteorology OR wind OR fog OR forecast)",
    "Q3_cappadocia_balloon": "(Cappadocia OR Kapadokya OR Goreme OR Göreme OR Nevsehir OR Nevşehir) AND (balloon OR balon)",
    "Q4_balloon_safety": BALLOON + " AND (safety OR accident OR accidents OR incident OR risk)",
    "Q5_nam_knowledge": '("neural additive model" OR "neural additive models") AND (physics OR "physics-informed" OR '
                        '"physics-guided" OR "domain knowledge" OR monotonic OR monotonicity)',
    "Q6_physics_additive": '("physics-informed" OR "physics-guided") AND ("additive model" OR "generalized additive" '
                           'OR "neural additive")',
}
FIELDS = "id,doi,display_name,publication_year,primary_location,type"
# title screen for the application domain (Q1-Q4): keep crewed / tourist hot-air ballooning
KEEP = r"hot.?air|cappadoc|kapadokya|g[oö]reme|nev[sş]ehir|\bbalon|tourism|recreational|passenger"
DROP = r"stratospher|mars|titan|venus|montgolfi|solar hot|near.?space|gastric|cosmic"


def _get(params: Dict) -> Dict:
    url = BASE + "?" + urllib.parse.urlencode(params) + "&mailto=rev1-literature@example.org"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except OSError as e:          # network errors and HTTP 429/5xx
            logger.warning("retry %d: %s", attempt + 1, e)
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"OpenAlex request failed: {url}")


def search(query: str, limit: int = 2000) -> Tuple[int, List[Dict]]:
    """All works whose title or abstract matches `query` (cursor paging)."""
    hits, cursor, total = [], "*", 0
    while cursor and len(hits) < limit:
        r = _get({"filter": f"title_and_abstract.search:{query}", "per-page": 200,
                  "cursor": cursor, "select": FIELDS})
        hits += r["results"]
        total, cursor = r["meta"]["count"], r["meta"].get("next_cursor")
        if not r["results"]:
            break
    return total, hits


def screen_titles(names=("Q1_balloon_ml", "Q2_balloon_weather", "Q3_cappadocia_balloon",
                          "Q4_balloon_safety")) -> List[Dict]:
    """Unique records from the domain queries whose title passes the KEEP/DROP screen."""
    import re
    seen, kept = set(), []
    for name in names:
        for r in json.loads((OUT / f"{name}.json").read_text()):
            t = r.get("display_name") or ""
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            if re.search(KEEP, t, re.I) and not re.search(DROP, t, re.I):
                kept.append(r)
    logger.info("domain records: %d unique, %d pass title screen", len(seen), len(kept))
    return kept


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"date": date.today().isoformat(), "index": "OpenAlex", "queries": {}}
    for name, q in QUERIES.items():
        total, hits = search(q)
        (OUT / f"{name}.json").write_text(json.dumps(hits, ensure_ascii=False, indent=1))
        summary["queries"][name] = {"query": q, "hits": total}
        logger.info("%-22s %5d hits", name, total)
    kept = screen_titles()
    summary["title_screen"] = {"keep": KEEP, "drop": DROP, "n": len(kept)}
    (OUT / "screened_titles.tsv").write_text("".join(
        f"{r['publication_year']}\t{r['doi'] or r['id']}\t{r['display_name']}\n"
        for r in sorted(kept, key=lambda r: r["publication_year"] or 0)))
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()

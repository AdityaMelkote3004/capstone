# Neo4j Graph Visualization — Setup and Usage Guide

This project builds a sector-relationship graph over the 87 StockNet
tickers (see `src/models/graph_model.py::build_sector_graph`) and can push
it to a Neo4j database purely for **visual exploration**. This is
completely separate from model training — no training script ever reads
from Neo4j; it only reads the graph via `build_sector_graph` directly
from `dataset/final/sector_mapping.csv`. Neo4j exists so a human can look
at the graph in a browser.

## 1. Get a Neo4j instance

**Recommended: Neo4j Aura (free, cloud-hosted)** — Neo4j Desktop now
bundles a time-limited Enterprise trial and pushes you toward payment
afterward, so Aura's free tier is the actual no-cost path.

1. Go to https://console.neo4j.io and sign in (Google SSO works).
2. Create a new **Aura Free** instance if you don't have one yet — 87
   nodes and a few hundred edges is trivially within the free tier's
   limits.
3. When the instance is created, Aura shows you a **connection URI**
   (`neo4j+s://xxxxxxxx.databases.neo4j.io`) and a **generated password**
   — it only shows the password once, so download the credentials file
   it offers, or copy it somewhere safe immediately.
4. Wait ~60 seconds after creation before connecting — a freshly created
   instance needs a moment to finish provisioning.

## 2. Configure credentials locally

Copy the template and fill in your real values:

```bash
cp .env.example .env
```

Edit `.env`:

```
NEO4J_URI=neo4j+s://your-actual-instance-id.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-actual-generated-password
NEO4J_DATABASE=your-actual-instance-id
```

`.env` is git-ignored — it will never be committed. Never paste real
credentials into a commit, a script, or anywhere other than this file.

## 3. Push the graph

```bash
pip install -r requirements.txt   # installs neo4j, python-dotenv if not already present
python scripts/export_graph_to_neo4j.py
```

Expected output:
```
Pushed 87 tickers and 381 SAME_SECTOR edges to Neo4j.
```

This script is idempotent (`MERGE`, not `CREATE`) — running it again
after re-running just updates the same nodes/edges rather than
duplicating them, so it's safe to re-run any time the underlying
`sector_mapping.csv` changes.

## 4. Explore the graph

Go back to https://console.neo4j.io, open your instance, and use the
**Query** tab (or open the instance's own Neo4j Browser) to run Cypher
queries against the pushed data. A few useful ones to start with:

**See everything:**
```cypher
MATCH (t:Ticker)-[r:SAME_SECTOR]-(t2:Ticker)
RETURN t, r, t2
```
This is the query to run for the actual visual graph view — Neo4j
Browser renders the nodes/edges as an interactive diagram, colorable by
the `sector` property.

**Count tickers per sector:**
```cypher
MATCH (t:Ticker) RETURN t.sector AS sector, count(*) AS n ORDER BY n DESC
```

**Count edges per sector (should show only same-sector edges, none crossing):**
```cypher
MATCH (a:Ticker)-[:SAME_SECTOR]-(b:Ticker)
RETURN a.sector AS sector, count(*) / 2 AS edges ORDER BY edges DESC
```

**Look up one ticker's sector-mates:**
```cypher
MATCH (t:Ticker {symbol: "AAPL"})-[:SAME_SECTOR]-(neighbor:Ticker)
RETURN neighbor.symbol, neighbor.sector
```

## What's in the graph

- **Node label:** `Ticker`, properties `symbol` (e.g. `"AAPL"`), `sector`
  (e.g. `"Consumer_Goods"`), `name` (same as `symbol` currently).
- **Relationship:** `SAME_SECTOR`, undirected, no properties — exists
  between two tickers iff they share a sector.
- **Scale:** 87 nodes, 381 relationships (9 sectors: 8 sectors of 10
  tickers each contribute C(10,2)=45 edges, plus one sector of 7
  contributing C(7,2)=21 — 8×45 + 21 = 381).
- This is a **cheap proxy graph**, not a reproduction of MAN-SF's actual
  Wikidata-relation graph — see `EXPERIMENT4_GAT.md`'s framing section for
  why that distinction matters if you're citing this graph in the paper.

## Regenerating or modifying the graph

The graph is built in `src/models/graph_model.py::build_sector_graph`,
reading directly from `dataset/final/sector_mapping.csv` (columns
`Ticker`, `Sector`). To change what the graph contains (e.g. add
correlation or dynamic tweet-co-occurrence edges, per
`RELATED_WORK.md`'s recommended next steps), extend that function and
re-run `scripts/export_graph_to_neo4j.py` — Neo4j will pick up whatever
edges the updated graph construction produces the next time you push.

"""One-time export of the static sector graph to Neo4j Aura, for
visualization only -- never read at training time. See
docs/superpowers/specs/2026-08-30-sector-graph-gat-neo4j-design.md.

Requires NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD in a local .env (see
.env.example for the expected keys). Credentials are read from the
environment only -- never hardcoded, never logged, never committed.
"""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from neo4j import GraphDatabase

from src.models.graph_model import build_sector_graph

SECTOR_CSV = "dataset/final/sector_mapping.csv"


def push_graph(driver, graph) -> None:
    with driver.session() as session:
        for ticker, attrs in graph.nodes(data=True):
            session.run(
                "MERGE (t:Ticker {symbol: $symbol}) "
                "SET t.sector = $sector, t.name = $name",
                symbol=ticker, sector=attrs["sector"], name=attrs["name"],
            )
        for a, b in graph.edges():
            session.run(
                "MATCH (x:Ticker {symbol: $a}), (y:Ticker {symbol: $b}) "
                "MERGE (x)-[:SAME_SECTOR]-(y)",
                a=a, b=b,
            )


def main() -> None:
    load_dotenv()
    uri = os.environ["NEO4J_URI"]
    user = os.environ["NEO4J_USER"]
    password = os.environ["NEO4J_PASSWORD"]

    graph = build_sector_graph(SECTOR_CSV)
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        driver.verify_connectivity()
        push_graph(driver, graph)
        print(f"Pushed {graph.number_of_nodes()} tickers and "
              f"{graph.number_of_edges()} SAME_SECTOR edges to Neo4j.")
    finally:
        driver.close()


if __name__ == "__main__":
    main()

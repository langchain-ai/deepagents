# soccer_agent.py

from pathlib import Path
import pandas as pd

from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI


# -----------------------
# CONFIG
# -----------------------
DATA_ROOT = Path("data/europa")


# -----------------------
# TOOL: deterministic stats
# -----------------------
def get_top_scorer(league: str, season: str):
    """
    Return the top goal scorer for a given league and season
    by aggregating all *_jugadores_seasonstats.csv files.
    """
    season = season.replace("/", "-")
    base = DATA_ROOT / league / season / "equipos"
    if not base.exists():
        return {
            "error": "Season path not found",
            "expected_path": str(base),
            "available_seasons": [
                p.name for p in (DATA_ROOT / league).iterdir() if p.is_dir()
            ]
        }

    frames = []

    for csv in base.rglob("*_jugadores_seasonstats.csv"):
        df = pd.read_csv(csv)

        if "nombre" not in df.columns or "Goals" not in df.columns:
            continue

        df = df[["nombre", "Goals"]].copy()
        df["team"] = csv.parent.name
        frames.append(df)

    if not frames:
        return {"error": "No valid season stat files found"}

    all_players = pd.concat(frames, ignore_index=True)
    top = all_players.sort_values("Goals", ascending=False).iloc[0]

    return {
        "player": top["nombre"],
        "goals": int(top["Goals"]),
        "team": top["team"],
        "league": league,
        "season": season,
    }


# -----------------------
# AGENT
# -----------------------
agent = create_deep_agent(
    tools=[get_top_scorer],
    system_prompt="""
You are a football analytics agent.

Rules:
- NEVER guess statistics
- ALWAYS call tools to compute results
- Compare results clearly and concisely
""",
    model=ChatOpenAI(model="gpt-4o-mini"),
)


# -----------------------
# ENTRY POINT
# -----------------------
if __name__ == "__main__":
    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": (
                "Compare the top goal scorers in the UEFA_UEFA_Champions_League "
                "and Spain_Primera_Division for the 2024-2025 season."
            )
        }]
    })

    print(result["messages"][-1].content)

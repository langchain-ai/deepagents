<div align="center">
  <a href="https://docs.langchain.com/oss/python/deepagents/overview#deep-agents-overview">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset=".github/images/logo-dark.svg">
      <source media="(prefers-color-scheme: light)" srcset=".github/images/logo-light.svg">
      <img alt="Deep Agents Logo" src=".github/images/logo-dark.svg" width="50%">
    </picture>
  </a>
</div>

<div align="center">
  <h3>The batteries-included agent harness.</h3>
</div>

<div align="center">
  <a href="https://opensource.org/licenses/MIT" target="_blank"><img src="https://img.shields.io/pypi/l/deepagents" alt="PyPI - License"></a>
  <a href="https://pypistats.org/packages/deepagents" target="_blank"><img src="https://img.shields.io/pepy/dt/deepagents" alt="PyPI - Downloads"></a>
  <a href="https://pypi.org/project/deepagents/#history" target="_blank"><img src="https://img.shields.io/pypi/v/deepagents?label=%20" alt="Version"></a>
  <a href="https://x.com/langchain_oss" target="_blank"><img src="https://img.shields.io/twitter/url/https/twitter.com/langchain_oss.svg?style=social&label=Follow%20%40LangChain" alt="Twitter / X"></a>
</div>

<br>

Deep Agents is an open source agent harness — an opinionated agent that runs out of the box. Extend, override, or replace any piece.

**Principles:**

- **Opinionated** — defaults tuned for long-horizon, multi-step work
- **Extensible** — override or replace any piece without forking
- **Model-agnostic** — works with any LLM that supports tool calling: frontier, open-weight, or local
- **Production-ready** — built on LangGraph (streaming, persistence, checkpointing) with first-class tracing, evaluation, and deployment via LangSmith

**Features include:**

- **Sub-agents** — delegate tasks to agents with isolated context windows
- **Filesystem** — read, write, edit, or search over pluggable local, sandboxed, or remote backends
- **Context management** — summarize long threads and offload tool outputs to disk
- **Shell access** — run commands in your sandbox of choice
- **Persistent memory** — pluggable state and store backends for cross-session recall
- **Human-in-the-loop** — approve, edit, or reject tool calls before they run
- **Skills** — reusable behaviors the agent can load on demand
- **Tools** — bring your own functions or any MCP server

Deep Agents is available as a JavaScript/TypeScript library — see [deepagents.js](https://github.com/langchain-ai/deepagentsjs).

> [!NOTE]
> **Deep Agents Code** — a pre-built coding agent in your terminal, similar to Claude Code or Cursor, powered by any LLM. Install with `curl -LsSf https://langch.in/dcode | bash`. See the [documentation](https://docs.langchain.com/deepagents-code) for the full feature set.

## Quickstart

```bash
uv add deepagents
```

```python
from deepagents import create_deep_agent

agent = create_deep_agent(
    model="openai:gpt-5.5",
    tools=[my_custom_tool],
    system_prompt="You are a research assistant.",
)
result = agent.invoke({"messages": "Research LangGraph and write a summary"})
```

The agent can plan, read/write files, and manage its own context. Add your own tools, swap models, customize prompts, configure sub-agents, and more. See the [documentation](https://docs.langchain.com/oss/python/deepagents/overview) for full details.

> [!TIP]
> For developing, debugging, and deploying AI agents and LLM applications, see [LangSmith](https://docs.langchain.com/langsmith/home).

## FAQ

### How is this different from LangGraph or LangChain?

LangGraph is the graph runtime. LangChain's `create_agent` is a minimal agent harness on top of it. Deep Agents is a more opinionated harness on top of `create_agent` — same building blocks, but with filesystem, sub-agents, context management, and skills bundled in. For how the three relate, see the [LangChain ecosystem overview](https://docs.langchain.com/oss/python/concepts/products).

### Does this work with open-weight or local models?

Yes. Any model that supports tool calling works — frontier APIs (OpenAI, Anthropic, Google), open-weight models hosted on providers like Baseten or Fireworks, and self-hosted models via Ollama, vLLM, or llama.cpp. Use any [LangChain chat model](https://docs.langchain.com/oss/python/langchain/models).

### Can I use this in production?

Yes! Deep Agents is built on LangGraph, designed for production agent deployments. Pair it with [LangSmith](https://docs.langchain.com/langsmith/home) for tracing, evaluation, and monitoring. See [Going to production](https://docs.langchain.com/oss/python/deepagents/going-to-production) for the full guide.

### When should I use Deep Agents vs. LangChain or LangGraph directly?

All three are layers in the same stack — see the [LangChain ecosystem overview](https://docs.langchain.com/oss/python/concepts/products) for how they relate. Use **Deep Agents** when you want the full harness — planning, context management, delegation — out of the box. Use [**LangChain's `create_agent`**](https://docs.langchain.com/oss/python/langchain/agents) when you want a lighter harness without the bundled middleware. Drop to [**LangGraph**](https://docs.langchain.com/oss/python/langgraph/overview) when the agent loop itself isn't the right shape and you need a custom graph.

The layers compose: any LangGraph `CompiledStateGraph` can be passed in as a sub-agent to a Deep Agent, so custom orchestration plugs in alongside the harness's defaults.

---

## Resources

- [Examples](examples/) — working agents and patterns
- [Documentation](https://docs.langchain.com/oss/python/deepagents/overview) — conceptual overviews and guides
- [LangChain ecosystem overview](https://docs.langchain.com/oss/python/concepts/products) — how Deep Agents, LangChain, LangGraph, and LangSmith fit together
- [API reference](https://reference.langchain.com/python/deepagents/) — complete reference for all public classes, functions, and types
- [Discussions](https://forum.langchain.com/c/oss-product-help-lc-and-lg/deep-agents/18) — community forum for technical questions, ideas, and feedback
- [LangChain Academy](https://academy.langchain.com/) — Comprehensive, free courses on LangChain libraries and products, made by the LangChain team.
- [Contributing Guide](https://docs.langchain.com/oss/python/contributing/overview) — how to contribute and find good first issues
- [Code of Conduct](https://github.com/langchain-ai/langchain/?tab=coc-ov-file) — community guidelines and standards

---

## Acknowledgements

Inspired by Claude Code: an attempt to identify what makes it general-purpose, and push that further.

## Security

Deep Agents follows a "trust the LLM" model. The agent can do anything its tools allow. Enforce boundaries at the tool/sandbox level, not by expecting the model to self-police. See the [security policy](https://github.com/langchain-ai/deepagents?tab=security-ov-file) for more information.


## 🌐 Web Resources & Interactive Index
- [KNIT RESCUE](https://theskillquest.pages.dev/knit-rescue.html)
- [BOLT CLIMB TAP TO THE TOP](https://thelearnquesters.pages.dev/bolt-climb-tap-to-the-top.html)
- [PUZZLE TRAILS](https://learnquester.github.io/puzzle-trails.html)
- [CATEGORY BRAIN260](https://learnquester.pages.dev/category-brain260.html)
- [MERGE BRICK BREAKER](https://thelearnquester.web.app/merge-brick-breaker.html)
- [SO DIFFERENT DRAGONS](https://thelearnquester.web.app/so-different-dragons.html)
- [CATEGORY GROW GAMES](https://thelearnquester.web.app/category-grow-games.html)
- [DESIGN WITH ME SUPERHERO TUTU OUTFITS](https://thelearnquester.web.app/design-with-me-superhero-tutu-outfits.html)
- [FOOTBALL SUPERSTARS 2026](https://learnquester.github.io/football-superstars-2026.html)
- [HOSPITAL INC](https://learnquester.github.io/hospital-inc.html)
- [TWO CARTS DOWNHILL](https://thelearnquester.web.app/two-carts-downhill.html)
- [GOAL IO](https://thelearnquester.web.app/goal-io.html)
- [STICKMAN MINERS WARS](https://thelearnquester.web.app/stickman-miners-wars.html)
- [CATEGORY BASKETBALL 3](https://themindzone.pages.dev/category-basketball-3.html)
- [PRINCESS WINTER ICE SKATING OUTFITS](https://thequizzone.pages.dev/princess-winter-ice-skating-outfits.html)
- [CATEGORY ROGUELIKE38](https://themindzone.pages.dev/category-roguelike38.html)
- [CATEGORY AGILITY 3](https://themindzone.pages.dev/category-agility-3.html)
- [STICKMAN ZOMBIE VS STICKMAN HERO](https://thelearnquester.web.app/stickman-zombie-vs-stickman-hero.html)
- [PERFECT ASMR CLEANING](https://thequizzone.pages.dev/perfect-asmr-cleaning.html)
- [POPPY PLAYER PUZZLE](https://thelearnquester.web.app/poppy-player-puzzle.html)
- [HUNGRY NOOB CAFE SIMULATOR](https://learnquester.github.io/hungry-noob-cafe-simulator.html)
- [CATEGORY CARTOON76](https://themindzone.pages.dev/category-cartoon76.html)
- [MASTER BLENDER](https://thequizzone.pages.dev/master-blender.html)
- [HILL CLIMBING MANIA](https://learnquester.github.io/hill-climbing-mania.html)
- [CATEGORY MOBILE2 095](https://learnquester.github.io/category-mobile2-095.html)
- [CATEGORY BRAIN261](https://themindzone.pages.dev/category-brain261.html)
- [CATEGORY ARENA255](https://themindzone.pages.dev/category-arena255.html)
- [RAGDOLL SOCCER 2 PLAYERS](https://thequizzone.pages.dev/ragdoll-soccer-2-players.html)
- [RAGDOLL FOOTBALL 2 PLAYERS](https://thequizzone.pages.dev/ragdoll-football-2-players.html)
- [CATEGORY ADVENTURE 2](https://themindzone.pages.dev/category-adventure-2.html)
- [3 TILES](https://thequizzone.pages.dev/3-tiles.html)
- [EMERGENCY JAM](https://thequizzone.pages.dev/emergency-jam.html)
- [KINGDOM WARS TD](https://learnquester.pages.dev/kingdom-wars-td.html)
- [CATEGORY CASUAL](https://themindzone.pages.dev/category-casual.html)
- [DINO SHOOTER PRO](https://thequizzone.pages.dev/dino-shooter-pro.html)
- [CATEGORY PARTY23](https://thelearnquester.web.app/category-party23.html)
- [FLICK N BOUNCE](https://thelearnquester.web.app/flick-n-bounce.html)
- [GIRLS FUN NAIL SALON](https://thelearnquester.web.app/girls-fun-nail-salon.html)
- [CATEGORY BATTLE 2](https://themindzone.pages.dev/category-battle-2.html)
- [FISHING CATCH THE SECRET BRAINROT](https://thequizzone.pages.dev/fishing-catch-the-secret-brainrot.html)
- [TIKTOK BRAIDED HAIRSTYLES](https://thelearnquester.web.app/tiktok-braided-hairstyles.html)
- [MY FARM LIFE](https://thequizzone.pages.dev/my-farm-life.html)
- [CATEGORY COLLECT565](https://thelearnquester.web.app/category-collect565.html)
- [CATEGORY ARMY40](https://themindzone.pages.dev/category-army40.html)
- [HOTGEAR](https://learnquesters.pages.dev/hotgear.html)
- [BUS STOP COLOR JAM](https://thelearnquester.web.app/bus-stop-color-jam.html)
- [CATEGORY HORDE SURVIVAL67](https://learnquesters.pages.dev/category-horde-survival67.html)
- [NAIL QUEEN](https://learnquester.github.io/nail-queen.html)
- [METAL BAY TOP BLADE POWER](https://thequizzone.pages.dev/metal-bay-top-blade-power.html)
- [MERMAID PRINCESS AVATER CASTLE](https://thelearnquesters.pages.dev/mermaid-princess-avater-castle.html)
- [JEWEL SOLITAIRE TRIPEAKS](https://thelearnquester.web.app/jewel-solitaire-tripeaks.html)
- [1945 AIR FORCE AIRPLANE](https://thelearnquester.web.app/1945-air-force-airplane.html)
- [CATEGORY BRAIN260](https://learnquesters.pages.dev/category-brain260.html)
- [CATEGORY MOUSE1 697](https://learnquester.github.io/category-mouse1-697.html)
- [CRAB GUARDS](https://thequizzone.pages.dev/crab-guards.html)
- [NEKOS ADVENTURE](https://thequizzone.pages.dev/nekos-adventure.html)
- [TONY ARCHER](https://learnquester.github.io/tony-archer.html)
- [CARDS MATCH PUZZLE](https://thelearnquesters.pages.dev/cards-match-puzzle.html)
- [MERGE HOTEL DEV](https://thequizzone.pages.dev/merge-hotel-dev.html)
- [INDEX37](https://themindzone.pages.dev/index37.html)
- [STUNT CAR EXTREME 2](https://thequizzone.pages.dev/stunt-car-extreme-2.html)
- [10K](https://thelearnquesters.pages.dev/10k.html)
- [MEOW SLIDE](https://learnquesters.pages.dev/meow-slide.html)
- [GOBATTLEIO](https://thequizzone.pages.dev/gobattleio.html)
- [HEXON RUSH](https://learnquesters.pages.dev/hexon-rush.html)
- [STICKHOLEIO](https://thequizzone.pages.dev/stickholeio.html)
- [CATEGORY UNBLOCKED GAMES](https://thelearnquester.web.app/category-unblocked-games.html)
- [CATEGORY CASUAL 5](https://thelearnquester.web.app/category-casual-5.html)
- [SHOOT N CRUSH](https://thequizzone.pages.dev/shoot-n-crush.html)
- [I8 CITY DRIVER](https://learnquester.github.io/i8-city-driver.html)
- [HORSEBACK SURVIVAL](https://learnquester.pages.dev/horseback-survival.html)
- [CATEGORY MOBILE2 112](https://thequizzone.pages.dev/category-mobile2-112.html)
- [CATEGORY FPS 2](https://thelearnquester.web.app/category-fps-2.html)
- [CATEGORY ARCHERY52](https://learnquesters.pages.dev/category-archery52.html)
- [STICK NINJA SURVIVAL](https://learnquesters.pages.dev/stick-ninja-survival.html)
- [BUBBLE SHOOTER GO](https://learnquesters.pages.dev/bubble-shooter-go.html)
- [MONSTER SLAYERS](https://thelearnquesters.pages.dev/monster-slayers.html)
- [BLOCK PUZZLE TROPICAL STORY](https://learnquester.github.io/block-puzzle-tropical-story.html)
- [EXTREME MAKEOVER HARLEY EDITION](https://thequizzone.pages.dev/extreme-makeover-harley-edition.html)
- [CATEGORY FLASH](https://themindzone.pages.dev/category-flash.html)
- [SLINGER BLOCK](https://thelearnquesters.pages.dev/slinger-block.html)
- [CATEGORY CASUAL 4](https://themindzone.pages.dev/category-casual-4.html)
- [LOVE TILE TRIO](https://thequizzone.pages.dev/love-tile-trio.html)
- [METEOHEROES](https://thelearnquesters.pages.dev/meteoheroes.html)
- [CATEGORY MAHJONG 3](https://thequizzone.pages.dev/category-mahjong-3.html)
- [PANDA RESTAURANT](https://learnquester.pages.dev/panda-restaurant.html)
- [ITALIAN BRAINROT SURVIVE PARKOUR](https://thelearnquesters.pages.dev/italian-brainrot-survive-parkour.html)
- [CATEGORY MAKEUP](https://themindzone.pages.dev/category-makeup.html)
- [CATEGORY CASUAL 7](https://themindzone.pages.dev/category-casual-7.html)
- [LEXY](https://learnquester.github.io/lexy.html)
- [CATEGORY GUN238](https://themindzone.pages.dev/category-gun238.html)
- [ARROW PUZZLE](https://thequizzone.pages.dev/arrow-puzzle.html)
- [FROGGA](https://themindzone.pages.dev/frogga.html)
- [CATEGORY AGILITY 2](https://themindzone.pages.dev/category-agility-2.html)
- [ELITE CHESS](https://learnquester.pages.dev/elite-chess.html)
- [ASMR PET TREATMENT](https://learnquester.github.io/asmr-pet-treatment.html)
- [INDEX2](https://thequizzone.pages.dev/index2.html)
- [CATEGORY CASUAL](https://learnquesters.pages.dev/category-casual.html)
- [FAIRY WINGERELLA](https://thelearnquester.web.app/fairy-wingerella.html)
- [MOTO X3M DEAD AHEAD](https://learnquester.pages.dev/moto-x3m-dead-ahead.html)
- [TRAFFIC JAM HOP ON](https://learnquesters.pages.dev/traffic-jam-hop-on.html)
- [CATEGORY BIKE](https://thequizzone.pages.dev/category-bike.html)
- [WILD CASTLE TD GROW EMPIRE](https://thequizzone.pages.dev/wild-castle-td-grow-empire.html)
- [CATEGORY TOWER DEFENSE](https://thelearnquester.web.app/category-tower-defense.html)
- [CATEGORY COLLECT565](https://themindzone.pages.dev/category-collect565.html)
- [FARMER PEDRO](https://thelearnquesters.pages.dev/farmer-pedro.html)
- [JINN DASH](https://thelearnquesters.pages.dev/jinn-dash.html)
- [SMASH THE BOTTLE](https://themindzone.pages.dev/smash-the-bottle.html)
- [CATEGORY SOLITAIRE27](https://thelearnquester.web.app/category-solitaire27.html)
- [INDEX32](https://thequizzone.pages.dev/index32.html)
- [INDEX39](https://thequizzone.pages.dev/index39.html)
- [CATEGORY COOKING](https://themindzone.pages.dev/category-cooking.html)
- [CATEGORY MINECRAFT](https://themindzone.pages.dev/category-minecraft.html)
- [GLACIER RUSH](https://thelearnquester.web.app/glacier-rush.html)
- [CATEGORY MANAGEMENT210](https://themindzone.pages.dev/category-management210.html)
- [STICKMAN MEGA BOSS BATTLES](https://thelearnquester.web.app/stickman-mega-boss-battles.html)
- [DRIFT IO](https://thelearnquesters.pages.dev/drift-io.html)
- [HEROBALL ADVENTURES 2](https://thequizzone.pages.dev/heroball-adventures-2.html)
- [CATEGORY CLASSIC97](https://themindzone.pages.dev/category-classic97.html)
- [BINGO HALLOWEEN](https://learnquester.github.io/bingo-halloween.html)
- [CANDY JEWELS](https://themindzone.pages.dev/candy-jewels.html)
- [SUPERHERO DROP AND SAVE](https://thelearnquesters.pages.dev/superhero-drop-and-save.html)
- [PUZZLE GAMES OUTING DAY](https://thelearnquesters.pages.dev/puzzle-games-outing-day.html)
- [CATEGORY MINECRAFT](https://thelearnquester.web.app/category-minecraft.html)
- [NINJA BAMBOO ASSASSIN](https://thelearnquesters.pages.dev/ninja-bamboo-assassin.html)
- [CATEGORY ART](https://themindzone.pages.dev/category-art.html)
- [CRUSH THE EGGS](https://learnquester.github.io/crush-the-eggs.html)
- [EQ TEST PUZZLE](https://themindzone.pages.dev/eq-test-puzzle.html)
- [CATEGORY QUIZ](https://learnquester.github.io/category-quiz.html)
- [CATEGORY HORROR90](https://themindzone.pages.dev/category-horror90.html)

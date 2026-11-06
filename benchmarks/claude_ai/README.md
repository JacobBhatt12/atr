# claude.ai A/B test: does the hosted ATR connector save tokens?

Three rounds on November 6, 2026: the first test (below), a retest after adding `question=` to
`atr_fetch` ([Retest](#retest-atr_fetch-with-question)), and a third round with sentence-level
answers and a shorter tool definition ([Round 3](#round-3-snippets-and-a-shorter-definition)).

The same questions are asked in fresh claude.ai chats, in three setups:

| Setup | Connectors | Prompt |
|---|---|---|
| **A** | ATR off. Claude's built-in web search and fetch on | The question |
| **B** | ATR (`https://atr-mcp.vercel.app/mcp`) on, built-in web on | The question, ATR not mentioned: does Claude pick ATR by itself? |
| **C** | ATR on, built-in web on | The question + "Use atr_fetch to read the link." |

All other connectors are off in every test chat. Same model, extended thinking off.

## Tasks

| Id | Prompt | Right answer |
|---|---|---|
| wiki | Using https://en.wikipedia.org/wiki/Model_Context_Protocol, who introduced MCP, when, and name three companies or products that adopted it. | Anthropic, November 2024, any three adopters named on the page |
| pdf | In https://arxiv.org/pdf/1706.03762, what BLEU scores did the big Transformer get on WMT 2014 English-to-German and English-to-French? | 28.4 and 41.8 (41.0 in Table 2 also counts) |
| hn | Open https://news.ycombinator.com and list the top 5 stories with their points. | Matches the front page at that time (A and C run back to back) |
| docs | From https://github.com/modelcontextprotocol/python-sdk, what is the install command and the minimum Python version? | Matches the README |

## Measuring

claude.ai doesn't show token counts. `measure.js` reads each chat back from claude.ai and estimates
(characters / 4):

- **Tool results**: everything the tools put into the chat.
- **Context re-read**: the text the model reads across the whole task, because each tool round
  re-sends everything before it. This is what uses up a plan's limits. When Claude loads ATR's
  tool it also adds atr_fetch's definition (226 tokens) to each model call after that.

Save the output lines to `runs.jsonl`, record right/wrong answers in `grades.json`, then:

```bash
python -m benchmarks.claude_ai.report
```

## Limits

- Token counts are estimates, and the system prompt and Claude's built-in tool definitions aren't
  included (they're the same in every setup).
- claude.ai may change how its built-in web fetch works, so results are dated.

## Results (November 6, 2026, Sonnet 5.5 in claude.ai, 25 chats)

| Task | Setup | Right answers | Used ATR | Tool results (tokens) | Context re-read (tokens) |
|---|---|---:|---:|---:|---:|
| wiki | A: built-in only | 1/1 | 0/1 | 196 | 435 |
| wiki | B: ATR on, not mentioned | 3/3 | 0/3 | 204 | 443 |
| wiki | C: ATR asked for | 3/3 | 3/3 | 2,978 | 3,806 |
| pdf | B | 3/3 | 0/3 | 70 | 267 |
| pdf | C | 3/3 | 3/3 | 1,136 | 2,728 |
| hn | B | **0/3** | 0/3 | 70 | 231 |
| hn | C | 3/3 | 3/3 | 2,357 | 3,066 |
| docs | B | **0/3** | 0/3 | 70 | 246 |
| docs | C | 3/3 | 3/3 | 174 | 857 |

Medians per task. Setup A was cut to one chat after setup B showed Claude never chooses ATR
on its own, which makes B the same as A in practice.

**What this shows**

1. **claude.ai's built-in WebFetch uses far fewer tokens than ATR.** It takes a question along
   with the link, has a separate model read the page, and gives Claude only that model's short
   answer (70–200 tokens). ATR returns the page itself, compacted (170–3,000 tokens). In context
   re-read, ATR cost 3–10× more.
2. **The built-in WebFetch was wrong on every Hacker News and docs question (6/6).** It returned
   an old copy of both pages: Hacker News stories from hours earlier, and Python "3.10" from an
   older README, where the live page says 3.11. ATR read the live page and was right every time
   (12/12).
3. **Claude doesn't pick ATR by itself** when the built-in WebFetch is available (0/12 in B).

**Not measured:** the tokens the built-in WebFetch's separate reader model uses. Whether those
count toward a plan's usage limits isn't public.

**Conclusion:** as a way to save tokens in claude.ai, `atr_fetch` loses to the built-in fetch.
Its advantage is fresh, exact content: live pages, numbers and lists quoted as written instead of
summarized. ATR's token savings in chatbots would have to come from shrinking other connectors'
results (GitHub, Notion, databases), which this test didn't cover.

Chat ids are in `runs.jsonl`. Hacker News and docs ground truth from the time of the test is in `truth/`.

## Retest: atr_fetch with question=

The first test showed why ATR lost: claude.ai's WebFetch gets a question and returns a short
answer, while `atr_fetch` returned the whole page. So `atr_fetch` got a `question` argument
(`atr/focus.py`): it splits the page into sections, scores each against the question (BM25, the
same as `atr_find_tools`) and returns the best few word for word, with line numbers. No model is
involved, so it is still free to run.

Setups (one chat per task, same model, same connectors as before):

| Setup | atr_fetch | Prompt |
|---|---|---|
| **B2** | `question` optional | The question, ATR not mentioned |
| **C2** | `question` optional | The question + "Use atr_fetch to read the link." |
| **C3** | `question` **required** (`"whole page"` reads the page from the top) | Same as C2 |

| Task | C (first test) | C2 | C3 | B2 (built-in WebFetch) |
|---|---:|---:|---:|---:|
| wiki | 3,806 | 4,924 | **1,611** | 574 |
| pdf | 2,728 | 3,257 | 2,779 | 399 |
| hn | 3,066 | 3,340 | **1,439** | 382 |
| docs | 857 | 1,221 | 1,360 | 343 |
| **Total** | 10,457 | 12,742 | **7,189** | 1,698 |
| Right answers | 12/12 | 4/4 | 4/4 | **2/4** |

Context re-read in tokens, including atr_fetch's definition (now 306 tokens) on every model call
after Claude loads it.

**What this shows**

1. **An optional argument is not used.** In C2 Claude saw `question` (asked to list the schema, it
   quoted it word for word) but never passed it. It set `budget=4000` instead, so results got
   bigger. Making it required (C3) got a question on every call: tool results fell from
   2,978 → 575 tokens (wiki) and 2,357 → 478 (Hacker News).
2. **Total context fell 31%** compared with the first test (10,457 → 7,189), and by more than half
   on the two big pages. The PDF didn't improve because Claude made a second, checking fetch. The docs
   page got worse: it was already small, so the longer tool definition cost more than it saved.
3. **The built-in WebFetch is still 4× cheaper in context**, mostly from overhead ATR can't avoid
   in claude.ai: loading the tool takes an extra model call (ToolSearch), and the tool definition is
   re-read on every call after that.
4. **The built-in WebFetch was wrong again on Hacker News and the docs** (old copies: stories that
   weren't on the front page, Python "3.10" where the README says 3.11). ATR was right 20/20
   across both rounds.

**Takeaway:** for reading the web in claude.ai, ATR now costs about 2–4× the built-in fetch instead
of 3–10×, and it is the one that is right about pages that change. The next savings are a shorter
tool definition and `question=` on other connectors' results (`atr_read`/`atr_call`), where there
is no built-in alternative: DeepWiki's 130,000-token wiki answers come back as ~650 tokens of the
relevant sections instead of a 1,500-token cut of the start.

Hacker News at the time of the retest: `truth/hn_rerun_0320.txt`.

## Round 3: snippets and a shorter definition

What was left after the retest, and the free fix for each:

| Cost | Fix | Source of the idea |
|---|---|---|
| atr_fetch's definition (306 tokens) is re-read on every model call | Cut to 162 tokens: one-sentence description, `budget` and `context` no longer advertised (they still work) | Atlassian's mcp-compressor (first-sentence descriptions, fewer parameters) |
| Answers were whole sections (~500 tokens) | Rank sections, then keep only the best 1–3 sentences, table rows or command lines of each, like search-result snippets (~120–260 tokens) | Query-focused extractive summarization with BM25 (SummerTime's BM25 model, bm25s) |
| A follow-up `search=` could return 4,000 tokens | Capped at 600; with a question, matching lines are ranked by it | |
| "Top 5" lists were cut before the 5th item; multi-part questions got too little | "Top/latest" questions get room for the list; each extra part of a question gets ~100 more tokens (up to 500) | |

**Tried and dropped:** two small free models for ranking, Model2Vec `potion-base-8M` (MIT) and
FlashRank `ms-marco-TinyBERT-L-2-v2` (Apache-2.0). On a 17-question answer set neither beat BM25
(11–12/17 vs 12/17 at the same size), so they weren't worth the extra size and start-up time on the
free server.

Answer recall offline (does the returned text contain the answer?), 17 questions on the test pages
plus 12 held-out questions on 4 pages not used for tuning (Rust on Wikipedia, Python's json docs,
MDN's 429 page, FastAPI's README):

| Method | Tuning set | Held-out set | Median size |
|---|---:|---:|---:|
| Whole sections, 500-token budget (retest) | 14/17 | 10/12 | ~420–485 tokens |
| Snippets, 250-token budget (round 3) | 15/17 | 11/12 | ~225 tokens |

In claude.ai (one chat per task):

| Task | C (first test) | C3 (retest) | C4 (snippets, short definition) | **C5 (+ budgets, capped search)** | B2 (built-in fetch) |
|---|---:|---:|---:|---:|---:|
| wiki | 3,806 | 1,611 | 3,324 | **1,308** | 574 |
| pdf | 2,728 | 2,779 | 956 | **1,005** | 399 |
| hn | 3,066 | 1,439 | 3,936 | **1,076** | 382 |
| docs | 857 | 1,360 | 718 | **755** | 343 |
| **Total** | 10,457 | 7,189 | 8,934 | **4,144** | 1,698 |
| Right answers | 12/12 | 4/4 | 4/4 | **4/4** | 2/4 |
| Fetches per chat | 1–2 | 1–2 | 1–2 | **1** | 1 |

**What this shows**

1. **C4 got two tasks to near the built-in fetch** (PDF 632 and docs 394 before the tool definition),
   but on Hacker News and Wikipedia Claude fetched a second time with `search=`, which returned up
   to 2,400 tokens. The Hacker News list had been cut just before the 5th story's points; on
   Wikipedia Claude wanted more adopters than the answer showed.
2. **C5 fixed both:** one fetch per chat, every answer right, **4,144 tokens in total: 60% less
   than the first test and 42% less than the retest.**
3. **The built-in fetch is now about 2.4× cheaper**, down from 3–10×. What's left is mostly fixed
   overhead in claude.ai: the extra model call to load a connector tool and the tool definition
   re-read after it (about 320 tokens per chat). Its answers were still wrong on Hacker News and
   the docs.

Hacker News at the time of C5: `truth/hn_c5_0346.txt`.

## Round 4: can Claude pick ATR by itself?

atr_fetch's description was rewritten to say when to use it instead of web_fetch ("when the answer
must be current or exact … web_fetch can return an older cached copy"). Then the Hacker News
question was asked without mentioning ATR, with claude.ai's Tool access set both ways:

| Tool access | What Claude saw of atr_fetch | Tool it used | Answer |
|---|---|---|---|
| Load tools when needed (default) | Name only | WebFetch | Wrong (old copy) |
| Tools already loaded | Name only (chat settings show `tool_search_mode: off`) | WebFetch, twice | Wrong (old copy) |

Asked afterwards, Claude said atr_fetch was "only listed by name … its schema and description
were never loaded", and that it picked WebFetch because WebFetch was already loaded. On this
account (many connectors and device tools), claude.ai keeps connector tools as names until Claude
loads one, whatever the setting. **So a connector's description can't make Claude choose it in
claude.ai.** Only the tool's name and a standing instruction are seen up front: profile
preferences, Project instructions, or saying "use atr_fetch" in the prompt.

Hacker News at the time: `truth/hn_b3_0403.txt` (1 of the 5 built-in answers was in the right place).

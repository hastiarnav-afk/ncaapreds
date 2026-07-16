# Final Project Presentation — Complete Guide
### Predicting NCAA Men's Basketball Wins & March Madness Success

---

## How to Use This Guide

This is a slide-by-slide script for your **10–12 minute Final Project Presentation**. For every slide you'll find:

- **Put on the slide** — short bullets/visual, exactly what to type
- **Say out loud** — the narration that carries the actual insight (per the course guidelines, don't just read the bullets)
- **Visual** — which chart from `/figures` to insert
- **Time** — a rough pace-check so you land in the 10–12 minute window

All the numbers below come from actually running the analysis on your real `cbb.csv` (not made up placeholders) — see `cbb_full_analysis.py` and `results_summary.txt` for the code and full output. Build your slides in PowerPoint/Google Slides/Keynote yourself using this as the script; everything you need to say and show is here.

**Data note:** `cbb.csv` covers 2,455 team-seasons across 355 Division I teams and 35 conferences, 2013–2019 (7 seasons). `G` and `W` include conference-tournament and NCAA-tournament games, not just the regular season — so "wins" in this project means *total season wins*.

---

## The Elevator Pitch

*(Read this once yourself — it's the one-paragraph version of your whole project, useful for your own clarity and for answering "so what's your project about?" in the hallway.)*

> Using seven seasons of team-level performance data, we built two models: a linear regression showing which fundamentals (shooting, turnovers, rebounding, free throws) actually drive season wins, and a logistic regression showing which opponent-adjusted efficiency metrics predict whether a team makes the NCAA tournament. We then extended this with a Random Forest to predict which tournament teams go on deep runs (Sweet 16 or better), and benchmarked it against the selection committee's own seeding. Finally, we tested all of this as a genuine forecast — training on 2013–2018 and predicting the entire, unseen 2019 season — and report exactly how accurate (and inaccurate) those forecasts were.

---

## Assumptions & Choices You Can Adjust

I made a few reasonable calls to turn "predict basketball performance" into a concrete project. Feel free to keep these or swap them — just be consistent if you change one:

1. **"Client"/beneficiary:** framed as a college-basketball analytics service (like KenPom or Barttorvik, which is literally where this data originally comes from) that serves two audiences — team coaching staffs (Model 1) and tournament forecasters/media/fans (Model 2 + the extension). Adjust if your group prefers a single audience.
2. **Two spotlighted models + one extension:** your course instructions explicitly recommend showing **two** models so the audience doesn't get lost. I built two core models (linear + logistic) plus one clearly-labeled "extension" (Random Forest) that you can present briefly or cut if you're short on time.
3. **Train/test split:** trained on 2013–2018, tested on the full 2019 season, treated as if it were unseen/future data. This is what makes the "forecast" section a genuine forecast rather than a lucky in-sample fit.
4. **BARTHAG excluded from the regression models:** BARTHAG is mathematically derived from ADJOE and ADJDE, so including all three together causes severe multicollinearity (VIF ≈ 36 for BARTHAG). We use BARTHAG only in the descriptive/EDA charts, not as a model input — see Slide 9 talking points for how to explain this if asked.
5. If your instructor wants you to reference specific techniques from lecture, swap in the corresponding vocabulary — the math underneath (least squares, sigmoid/log-odds, eigendecomposition, decision trees) is standard regardless of what your course happened to call it.

---
---

# PART 1 — PURPOSE
### *(~1.5 minutes | Slides 1–3)*

## Slide 1 — Title Slide

**Put on the slide:**
- Title: **"Predicting Wins and March Madness Success in NCAA Basketball"**
- Subtitle: "What regular-season performance tells us about who makes — and wins — the tournament"
- Your names, course name/number, date
- One clean background image (a basketball court, net, or bracket graphic)

**Say out loud:** Introduce the team and give a one-sentence hook: *"Every March, 68 teams make the tournament and everyone argues about who deserves it — we used seven years of advanced team statistics to find out, quantitatively, which numbers actually predict wins and tournament success."*

**Time:** 15 seconds.

---

## Slide 2 — The Problem & Who Benefits

**Put on the slide:**
- Why this matters: the NCAA tournament is a major, closely-watched revenue and media event, and every program wants to know which fundamentals actually translate to winning
- Key question: *Which team performance metrics predict (a) season wins and (b) NCAA tournament success?*
- Who benefits:
  - **Coaching staffs & athletic departments** — which fundamentals to emphasize
  - **Bracket forecasters, media, and fans** — who's likely to make and advance in the tournament

**Say out loud:** Make it concrete and human. *"Coaches have limited practice time — should they drill shooting, or ball control, or rebounding? And every March, tens of millions of brackets get filled out across ESPN, Yahoo, and office pools nationwide — but most people are guessing off gut feel and seed number. We wanted to see what the data actually says."*

**Visual:** none needed, or a simple title-style graphic.

**Time:** 30 seconds.

---

## Slide 3 — Research Question & Hypothesis

**Put on the slide:**
- **Research question:** Can regular-season team performance metrics predict (1) how many games a team wins, and (2) how far they advance in March Madness?
- **Hypothesis:**
  - Shooting efficiency and turnovers will matter most for season wins (more than rebounding or free throws)
  - Opponent-adjusted efficiency metrics will predict tournament qualification well — better than the committee's seed alone predicts deep tournament runs
- **Beneficiary reminder:** coaches (wins) + forecasters/fans (tournament)

**Say out loud:** *"Our hypothesis draws on Dean Oliver's 'Four Factors' theory of basketball success, which we'll explain in a moment — we expected shooting and turnovers to dominate, with rebounding and free throws playing a smaller role."*

**Time:** 30 seconds.

---
---

# PART 2 — PROCESS & DESCRIPTIVE STATISTICS
### *(~3 minutes | Slides 4–7)*

## Slide 4 — The Data

**Put on the slide:**
- **Source:** Kaggle's "College Basketball Dataset" (Andrew Sundberg), originally scraped from barttorvik.com's advanced team-stats database
- **Scope:** 2,455 team-seasons · 355 teams · 35 conferences · 2013–2019
- **Dependent variables:** Season wins (`W`), whether a team made the NCAA tournament, and how far they advanced
- **Key predictors:** 17 team-level performance metrics — shooting %, turnover rate, rebound rate, free-throw rate, tempo, and opponent-adjusted efficiency ratings
- **Cleaning:** No missing data in any performance metric; `SEED`/`POSTSEASON` are naturally blank for the ~81% of teams that didn't make the tournament (that's expected, not a data problem)

**Say out loud:** *"Every row is one team's full season — including their conference tournament and NCAA tournament games — so 'wins' here means total season wins, not just the regular season."*

**Time:** 30 seconds.

---

## Slide 5 — Educational Background: The "Four Factors"

*(This slide directly addresses the rubric's request for useful educational background/intuition.)*

**Put on the slide:**
- Basketball analytics has long organized team performance into **Dean Oliver's Four Factors**, each with an offensive and defensive version:
  1. **Shooting** (Effective FG%) — `EFG_O` / `EFG_D`
  2. **Turnovers** — `TOR` / `TORD`
  3. **Rebounding** — `ORB` / `DRB`
  4. **Free Throws** — `FTR` / `FTRD`
- Plus **tempo** (`ADJ_T`, possessions per 40 min) and **opponent-adjusted efficiency** (`ADJOE`/`ADJDE`, points per 100 possessions, adjusted for opponent strength)
- Classic theory weights these roughly: shooting ≈ 40%, turnovers ≈ 25%, rebounding ≈ 20%, free throws ≈ 15%

**Say out loud:** *"This isn't just our framework — it's the standard lens basketball analysts have used for 20 years. That gave us a principled way to pick which variables to model, instead of throwing every column at a regression and hoping."*

**Time:** 30–40 seconds.

---

## Slide 6 — Descriptive Statistics I: Wins & the Efficiency Quadrant

**Put on the slide:**
- Average team wins 16.3 games/season (SD ≈ 6.6); the distribution is right-skewed — most teams cluster around 12–20 wins, with a long tail of elite teams reaching 30+
- **Visual:** `01_wins_distribution.png`
- **Visual:** `02_efficiency_quadrant.png` — teams plotted by adjusted offense vs. adjusted defense, colored by tournament wins

**Say out loud (for the quadrant chart — this is a great one to spend real time on):** *"Every dot is one team-season. The teams that go deepest in the tournament — the darkest red dots — cluster tightly in the top-right: elite offense AND elite defense at the same time. There's almost no path to a Final Four with just one elite side of the ball."*

**Time:** 45 seconds.

---

## Slide 7 — Descriptive Statistics II: Correlations & PCA

**Put on the slide:**
- **Visual:** `03_correlation_heatmap.png`
- Wins correlate most strongly with `ADJOE` (+0.75), `EFG_D` (−0.59), `ADJDE` (−0.69) and `TOR` (−0.45)
- **Visual:** `05_pca_scatter.png` — Two principal components capture **48%** of the variation across all 11 core stats (PC1: 27.5%, PC2: 20.9%)
  - **PC1** reads as "overall team quality" — good offense and good defense load together
  - **PC2** reads more as "playing style" (pace/rebounding emphasis vs. jump-shooting teams)

**Say out loud:** *"One counter-intuitive thing we found: our `DRB` variable — labeled defensive rebounding — is actually negatively correlated with wins. After digging in, this is because in this dataset DRB behaves like an 'opponent offensive rebounds allowed' rate rather than 'boards you grabbed' — so lower is better, the opposite of what the name suggests. Always good to sanity-check what a variable name actually means before trusting the sign of its coefficient."* *(This is exactly the kind of "is this result intuitive or surprising" discussion your instructions ask for — it shows you dug into the data rather than blindly trusting column names.)*

**Time:** 45 seconds.

---
---

# PART 3 — RESULTS FROM ANALYTICAL ANALYSIS
### *(~4 minutes | Slides 8–11)*

## Slide 8 — Our Modeling Approach

**Put on the slide:**
- **Model 1 — Multiple Linear Regression:** `Wins ~ Four Factors + Tempo` → *"What drives winning?"*
- **Model 2 — Logistic Regression:** `Made Tournament ~ Opponent-Adjusted Efficiency` → *"Who makes March Madness?"*
- **Extension — Random Forest:** `Sweet 16+ ~ Efficiency + Tempo + WAB` (among tournament teams) → *"Who goes deep?"*
- All models trained on **2013–2018** and tested on the fully held-out **2019 season**

**Say out loud:** *"We deliberately kept this to two main models so we don't overwhelm you, plus one extension. We picked linear regression because wins is a continuous count, and logistic regression because tournament qualification is yes/no — a natural, standard pairing. We held out 2019 entirely during training, so every result on that season is a genuine forecast, not a fit we're just showing you after the fact."*

**Time:** 30 seconds.

---

## Slide 9 — Model 1 Results: What Drives Season Wins?

**Put on the slide (a clean table, not raw regression output):**

| Factor | Effect on Wins | Significant? |
|---|---|---|
| Effective FG% (offense) | **+0.92** wins per point | ✅ p < .001 |
| Effective FG% allowed | **−0.93** wins per point | ✅ p < .001 |
| Turnover rate (own) | **−0.85** wins per point | ✅ p < .001 |
| Turnovers forced | **+0.89** wins per point | ✅ p < .001 |
| Off. rebound rate | +0.36 wins per point | ✅ p < .001 |
| Free-throw rate | +0.11 wins per point | ✅ p < .001 |
| Tempo | +0.13 wins per possession | ✅ p < .001 |

- **R² = 0.866** (fit explains 86.6% of variation in season wins)
- **Every predictor is statistically significant** (p < .0001)
- No multicollinearity concerns (all VIFs < 1.5)

**Say out loud:** *"Every single factor came out significant, and the ranking matches classic basketball theory almost exactly — shooting and turnovers have roughly 2.5 times the impact of rebounding, which in turn has about 2.5 times the impact of free throws. We solved this with the actual normal-equations formula from linear algebra — beta equals X-transpose-X inverse times X-transpose-y — not just a library call, and then cross-checked it against scikit-learn to confirm it matches exactly."* *(mention the linear-algebra detail only if your course covered the normal equations / wants that math shown — it's a nice "extends the course" touch if so)*

**Visual:** `06_model1_diagnostics.png` (actual vs. predicted + residuals)

**Time:** 60 seconds.

---

## Slide 10 — Model 2 Results: What Predicts Making the Tournament?

**Put on the slide:**
- Features: opponent-adjusted offensive efficiency (`ADJOE`), defensive efficiency (`ADJDE`), tempo — deliberately *not* raw box-score stats, because tournament selection depends on strength of schedule, and these metrics are opponent-adjusted while the Four Factors are not
- **Accuracy: 90.1%** (vs. 80.7% if you just guessed "no" for every team)
- **AUC: 0.907** — strong discrimination between tournament and non-tournament teams
- A 1-standard-deviation increase in adjusted offensive efficiency multiplies a team's odds of making the tournament by **4.8×**

**Say out loud:** *"Our model correctly identifies 90% of outcomes using only three regular-season numbers — no knowledge of conference tournaments or the committee's actual deliberations. Where it struggles is recall — it misses about a third of actual tournament teams, mostly small-conference teams that earn automatic bids by winning their conference tournament despite modest efficiency numbers. That's a real, sensible limitation: our model knows nothing about who wins in March, only about season-long performance."*

**Visual:** `07_model2_roc_confusion.png`

**Time:** 60 seconds.

---

## Slide 11 — Extension: Predicting a Deep Tournament Run

**Put on the slide:**
- Among the 68 teams that make the tournament each year: will they reach the **Sweet 16 or better**?
- Compared three approaches:

| Approach | Accuracy | AUC |
|---|---|---|
| Seed number alone (the committee's judgment) | 94.1% | 0.954 |
| Logistic regression (our stats) | 95.6% | 0.978 |
| **Random Forest (our stats)** | **97.1%** | 0.974 |

- Biggest driver: **WAB** ("Wins Above Bubble," 37%) and **ADJOE** (34%), followed by **ADJDE** (21%)

**Say out loud:** *"This is our 'extends the course' piece — a Random Forest, which handles non-linear patterns a straight logistic regression can't. The headline finding: using only regular-season stats, with zero knowledge of the committee's own seeding, we match or slightly beat the predictive power of the seed number itself. That's a strong result — it means the information the committee uses to rank teams is largely already present in these performance metrics."*

**Visual:** `08_extension_rf_vs_seed.png`

**Time:** 60 seconds.

---
---

# PART 4 — FORECASTS & PREDICTIONS
### *(~1.5–2 minutes | Slide 12)*

## Slide 12 — Forecast: Putting It to the Test on the 2019 Season

*(This is your required "at least one forecast, with error" slide — you actually have three, which is a strength: lean into it.)*

**Put on the slide:**

| Forecast | Result | Error / Confidence |
|---|---|---|
| Season win total | Predicts wins to within **±2.5 games** (RMSE) | R² = 0.844 out-of-sample |
| Made the tournament? | **90.1%** accuracy | AUC = 0.907, Brier = 0.079 |
| Reached Sweet 16+? | **97.1%** accuracy | AUC = 0.974 |

- **Concrete example — the "bubble":** teams the model rated 30–70% likely to make the tournament included Nevada, VCU, Saint Mary's, Syracuse, and Iowa — all **correctly predicted "in."** On the other side, Dayton, Alabama, Missouri, and Wichita St. were **correctly predicted "out"** (each given no better than a 46% chance).
- **Two honest kinds of misses:** the model *overrated* several major-conference bubble teams — Penn St., Clemson, NC State, and TCU were all rated 63–70% likely to make the tournament but were ultimately passed over — and it *underrated* small-conference automatic qualifiers like UC Irvine (Big West, rated just 32%) and Murray St. (Ohio Valley, rated 46%), who clinched a bid by winning their conference tournament despite a modest efficiency profile.
- **Concrete miss — Oregon:** a 12-seed with a *negative* Wins-Above-Bubble score; our model gave them just an **11%** chance of reaching the Sweet 16 — they did it anyway. A genuine March Madness upset even a strong model didn't see coming.

**Say out loud:** *"We didn't cherry-pick 2019 — it's simply the most recent season in our data, so testing on it is the closest thing to a true forecast we could build. Our error bars are honest: plus-or-minus 2.5 wins, and roughly 90% classification accuracy. And the misses are actually informative — the model overrates major-conference teams with a good-but-not-great resume, like Clemson and TCU, who need the committee's at-large vote; and it underrates small-conference teams like UC Irvine, who don't need an at-large bid at all because they punched their ticket by winning their own conference tournament outright. And Oregon is a great reminder of why they call it March Madness — the model correctly flagged them as a long shot, and they made the Sweet 16 anyway. No model will ever fully capture that."*

**Visual:** none new needed — refer back to `06`, `07`, `08`, or build a simple summary table slide.

**Time:** 60–75 seconds.

---
---

# PART 5 — CONCLUSION
### *(~1.5 minutes | Slides 13–14)*

## Slide 13 — Key Findings & Practical Implications

**Put on the slide:**
- Shooting efficiency and turnovers are ~2–3× more impactful on wins than rebounding or free throws — **practical implication for coaches:** if practice time is scarce, prioritize shot quality and ball security
- Opponent-adjusted efficiency (not raw box-score stats) is what predicts tournament selection — schedule strength matters
- Regular-season performance data can forecast deep tournament runs about as well as the selection committee's own seed
- March Madness lives up to its name: even strong models miss real upsets (Oregon)

**Say out loud:** *"Coming back to our original question — yes, we can predict a meaningful share of both season wins and tournament success from regular-season numbers alone. But there's an irreducible amount of randomness in single-elimination basketball that no amount of data fully removes — and that's exactly what makes the tournament fun to watch."*

**Time:** 30–40 seconds.

---

## Slide 14 — Lessons Learned & Future Work

**Put on the slide:**
- **What we'd add with more data:** player-level data (injuries, transfers, experience), betting-market lines (which aggregate information we don't have), and more recent seasons for a larger, more current test set
- **Lessons learned:**
  - Check what a variable actually measures before trusting its coefficient's sign (the `DRB` surprise)
  - Watch for multicollinearity between metrics that are mathematically related (BARTHAG vs. ADJOE/ADJDE)
  - A model doesn't need to be complex to work well — our simple logistic regression matched the fancier Random Forest almost exactly

**Say out loud:** *"If we did this again, the single biggest upgrade would be adding player-level and injury data — team stats can't capture a star player going down in February. But even with just team-level numbers, we got surprisingly far."*

**Time:** 30–40 seconds.

---

## Slide 15 — Thank You / Questions

**Put on the slide:** "Thank you — Questions?" + your names + maybe the efficiency quadrant chart as a nice closing visual.

**Time:** 10 seconds, then open the floor.

---
---

# Rubric Alignment Checklist

Use this to self-check before you present — maps directly to your rubric's five rows.

| Rubric Category | Where You Cover It |
|---|---|
| Context, Question, Hypothesis | Slides 2–3 (client value, purpose, hypothesis) |
| Research Method & Descriptive Statistics | Slides 4–7 (source, sample size, variables, cleaning, Four Factors rationale, EDA) |
| Analytical Analysis & Results | Slides 8–11 (two spotlighted models + one clearly-labeled extension, significant variables, intuitive/surprising discussion) |
| Forecasts & Implications | Slide 12 (three forecasts with explicit error/AUC/RMSE, concrete team examples, confidence discussion) |
| Presentation Delivery & Media | Your delivery — see tips below |

---

# Delivery Tips (for the "Presentation Delivery and Media" rubric row)

- **Don't read your slides.** Every slide above has a short "Put on the slide" (terse) and a longer "Say out loud" (the actual content) — that gap is intentional. Practice the talking points until they're conversational.
- **Rehearse with a timer.** 15 slides in 10–12 minutes is roughly 45 seconds/slide on average — some (title, thank you) take 10 seconds, which buys you extra time on Slides 9–12.
- **Split speaking roles** roughly by section (Purpose / Data / Model 1 / Model 2+Extension / Forecast+Conclusion) so every group member has clear ownership and the transitions are natural.
- **Practice pointing at the chart**, not just narrating over it — say "notice how the dots cluster in this corner" while gesturing, rather than describing the axis labels aloud.
- Keep slide text to the bullets shown above — the room can read six words faster than you can say them; use your voice for the interpretation.

---

# Anticipated Questions (Optional Backup Slides)

Your instructor may ask follow-ups. You don't need slides for these, but knowing the answer helps:

- **"Why didn't you include BARTHAG in the models?"** → It's a mathematical transformation of ADJOE and ADJDE (correlation ~0.86 and ~-0.84 respectively), so including all three creates severe multicollinearity (VIF ≈ 36). We kept it only for descriptive charts.
- **"Why a temporal split instead of random train/test?"** → A random split could blend information across seasons in ways that make the test look easier than a real forecast would be; holding out an entire future season is a stricter, more honest test.
- **"Why is Recall lower than Precision in Model 2?"** → The model under-flags small-conference teams that earn automatic bids with modest efficiency numbers — a real, explainable limitation, not a bug.
- **"How would this change with 2020–2025 data?"** → The 2020 tournament was cancelled (COVID-19), which would need to be excluded; more recent seasons could be added by re-running the same script against updated Kaggle files (`cbb21.csv`–`cbb25.csv` in later dataset releases).

---

# Files Reference (what's in your output folder)

| File | What it's for |
|---|---|
| `cbb_full_analysis.py` | All the Python code — EDA, both models, extension, forecast. Fully commented and organized in `# %%` cells you can run top-to-bottom or cell-by-cell in VS Code/Spyder. Put your `cbb.csv` in the same folder before running. |
| `figures/01`–`08` `.png` | Every chart referenced above, ready to drop straight into your slides |
| `descriptive_stats_table.csv` | Mean/SD/min/median/max for all core variables (for Slide 4/6 if you want exact numbers) |
| `bubble_teams_2019_forecast.csv` | The full "bubble zone" (30–70% predicted probability) team list for 2019 |
| `sweet16_forecast_2019_full.csv` | Every 2019 tournament team with its predicted Sweet-16+ probability — this is where the Oregon example comes from |
| `results_summary.txt` | Plain-text dump of every coefficient/metric used throughout this guide |

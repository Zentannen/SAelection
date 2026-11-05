# Election-night projection from partial results: a literature review for the Gauteng VD setting

*Scope:* methods that project the final result while counting units report in a non-random order, judged for
the 2026 Gauteng LGE (about 2,800 VDs, 526 wards, 9 municipalities, VD history for LGE2016, PE2019, LGE2021
and PE2024, and VD contiguity). Claims are linked to sources. **[unverified]** marks anything I could only see
in a secondary summary or abstract and could not read in the original.

## 1. The CSIR election-night model (South Africa)

**How it works.** The CSIR has run election-night forecasts since 1999. It has done so for the SABC since 2004
([Ittmann, Holloway & Dudeni-Tlhone 2017](https://doi.org/10.5784/33-1-567)). The method is set out in
[Greben, Elphinstone & Holloway (2006)](https://orion.journals.ac.za/pub/article/view/35):

* Before the election, VDs are clustered with **fuzzy c-means** on their party percentages in a previous
  election. The distance is Euclidean on raw percentages, so big parties dominate. They used fuzziness m = 1.2
  (1.4 was preferred later) and 20 clusters. The number of clusters "does not seem to be so critical" in the
  range 10–40.
* On the night, each cluster's party shares and turnout are estimated as **membership- and size-weighted
  averages of the counted VDs** (their eqs. 14–15). Each uncounted VD is predicted as the membership-weighted mix
  of cluster results (eqs. 16–17). Counted plus predicted VDs are then summed to any level (eq. 18).
* The old results define only the memberships: "they are not used as an input for — or an initial prediction
  of — the outcome of the current election." So the model estimates **cluster levels in the new election, not
  swing from each VD's own past**. The authors name "trend matrices" that link old and new patterns at VD level
  as a possible improvement ([Greben 2007](https://scielo.org.za/scielo.php?script=sci_arttext&pid=S0038-23532007000300015)).
* Later versions use about 20 clusters per province and/or nationally. Metro predictions come from provincial
  clusters ([Holloway et al. 2018](https://orion.journals.ac.za/pub/article/view/581);
  [Kimmie 2016 slides](https://api.sambuz.com/doc/csir-elections-forecasting-ppt-presentation-424479)).
  A 2024 press explainer says ward-level socio-economic "fingerprints" are now used as well
  ([Daily Maverick 2024](https://www.dailymaverick.co.za/article/2024-05-31-explainer-how-the-council-for-scientific-and-industrial-research-csir-predicts-election-results)).
  **[unverified: journalistic description, no technical document found]**
* **Uncertainty:** none is published formally. Trust in a forecast was judged "via graphic inspection": the
  trajectory had to become smooth and stop trending. The authors flag interval estimation as unsolved, because
  "it is not clear that the counted voting districts (even the most recent ones) could be considered
  representative"
  ([Greben et al. 2006](https://orion.journals.ac.za/pub/article/view/35)). The 2016 slides confirm there are no
  intervals. The 2024 publicity quotes a "2% margin of error"
  ([CSIR 2024](https://www.csir.co.za/2024-national-elections-predictions-csir-once-again-set-forecast-south-african-general-elections);
  [TechCentral](https://techcentral.co.za/csir-election-modelling-anc-punished/245520/)).

**Two stated assumptions** ([Ittmann et al. 2017](https://doi.org/10.5784/33-1-567)):

1. VDs in a cluster shift alike.
2. The early VDs cover all clusters. "The model is designed to adjust for moderate non-random patterns in this
   order but will not operate well under extreme political or spatial bias."

Known order effects: urban VDs used to report first, and more recently smaller VDs report first.

**Reported accuracy** (error = forecast − final, in pp):

| Election | Counted at forecast | Error | Source |
|---|---|---|---|
| 2004 NPE | 4.7% of votes | ANC +0.08, DA +1.71, IFP −1.28 | [Ittmann 2017](https://doi.org/10.5784/33-1-567) Table 2 |
| 2006 LGE | 6.8% of votes | ANC +1.22, DA +1.03, IFP −1.28 | ibid. Table 3 |
| 2009 NPE | 5.0% of votes | ANC −0.32, DA +0.68, IFP −0.40, COPE (new) −0.59 | ibid. |
| 2011 LGE | 4.9% of votes | ANC −0.74, DA +1.86, IFP −0.54 | ibid. |
| 2014 NPE | 4.7% / **40.2% of votes (49.3% of VDs)** | ANC +1.68 / **+1.15**; EFF (new) −2.39 / **−1.56** | ibid. Table 4 |
| 2016 LGE metros | 5 am the morning after the vote (13% of Johannesburg VDs) | JHB ANC −0.4, DA +0.4. Tshwane DA −1.6, ANC +1.3: **wrong leader until "very late"**. Ekurhuleni DA +1.9 | [Holloway 2018](https://orion.journals.ac.za/pub/article/view/581) Table 4; [CSIR 2016](https://www.csir.co.za/2016-municipal-elections-csir-prediction-was-spot) |
| 2019 NPE | 1,098 of 22,925 VDs (4.8%) | ANC −0.78, DA ±0.39, EFF ±1.29 | [SABC 2019](https://sabcnews.com/sabcnews/?p=400371) |
| 2021 LGE, **Gauteng** | 06:24 the morning after; % counted not reported | ANC +0.2, **DA +5.4, ActionSA (new) −5.1**, EFF +0.9 | projection: [BusinessDay](https://www.businessday.co.za/bd/national/2021-11-02-csir-projects-big-losses-for-anc-and-da/); final: computed by me from this repo's IEC VD file (ANC 36.04, DA 28.34, ActionSA 10.51, EFF 11.61) |
| 2024 NPE | 8.24% of VDs | ANC +1.8, DA +0.5, **MK (new) −1.8**, EFF −0.5 | [TechCentral](https://techcentral.co.za/csir-election-modelling-anc-punished/245520/); final from [Wikipedia](https://en.wikipedia.org/wiki/2024_South_African_general_election) |

Two notes on the table:

* For 2004–2011, the forecast top-3 MAE "dropped and stayed below 1%" after **4–10% of votes** were counted.
  The raw tally needed **41–76%**.
* In 2024, CSIR said about 15% of VDs are needed for accurate *provincial* projections, and that the model is
  weaker for "smaller and newer" parties
  ([TechCentral](https://techcentral.co.za/csir-election-modelling-anc-punished/245520/)).

**The 2014 post-mortem** ([Ittmann et al. 2017](https://doi.org/10.5784/33-1-567)) found two causes of the 2014
failure:

1. **Extreme order bias.** Only 17.9% of Gauteng VDs had been declared when 49.3% were declared nationally.
   Contested EFF-leaning Gauteng VDs were held back (Gauteng EFF: forecast 8.2% vs final 10.26%).
2. **Clusters built on 2009 that were not homogeneous in their 2014 swing.**

The post-mortem replayed 2014 in the real reporting order with "perfect" clusters built on the 2014 results. The
top-3 MAE was then below 1 pp from 1% counted. In that replay, giving the EFF extra weight in the clustering
brought its error under 1 pp from the start. **Lesson (my reading): how well the grouping predicts *change*
mattered more than the order bias, and small and new parties must be weighted into the similarity metric.**
The 2021 Gauteng (ActionSA) and 2024 (MK) misses repeat the pattern: the new party is under-projected and the
party it draws from is over-projected.

## 2. US election-night models

**Washington Post / Cherian & Bronner** (the most fully documented).

*2020* ([write-up](https://s3.us-east-1.amazonaws.com/elex-models-prod/2020-general/write-up/election_model_writeup.pdf)):

* Each unit is compared with **its own previous result**. The target is the normalised change
  (T₂₀ − T₁₆)/T₁₆, regressed on demographics by **median/quantile regression**.
* 90% intervals come from **conformalized quantile regression (CQR)**, weighted toward large counties.
* Order bias is handled by predicting a normalised change, which is "close to" exchangeable, rather than a level.
  Empirically, non-random order has "minimal effect... once 20–25% of the counties nationally have reported".
  The model needs "a few dozen" finished units to start and "a few hundred more" for calibrated intervals.
* Kriging, including "demographically aware" kriging, was tried and dropped: it was less accurate, and its
  Gaussian intervals "often appeared to poorly cover the true outcome".

*Precinct update* ([Dec 2020](https://s3.us-east-1.amazonaws.com/elex-models-prod/elex-models-prod/2020-general/write-up/election_model_writeup_update1.pdf)):

* Summing unit interval bounds (perfect correlation) gave state intervals wider than 10% of votes even after half
  the precincts had reported.
* They replaced it with an equicorrelated Gaussian model of the non-conformity scores, with separate calibration
  sets for subgroups such as urban and rural.

*2024* ([poster, Cherian, Bronner & Candès](https://jjcherian.github.io/assets/pdf/election_poster.pdf);
[code](https://github.com/washingtonpost/elex-live-model)):

* Ridge regression on normalised margin and a turnout factor relative to the previous result.
* An **ensemble** of this "fundamentals" rule with an "extrapolation" rule from partially counted units, using
  variance-minimising weights.
* Bootstrap plus conformal intervals, stratified by county class, with contest-level random effects.
* The poster states that "standard spatiotemporal methods (kriging, random effects model) have poor predictive
  coverage" and cites Greben et al. (2006) as prior art.

**Adaptive conformal inference.** [Gibbs & Candès (2021)](https://arxiv.org/abs/2106.00170) replay WaPo's CQR on
2020 counties reported east-to-west. Static CQR "fails to maintain the desired 90% coverage", with troughs at
time-zone changes. Adaptive conformal stays near 90% throughout. **This is the clearest direct evidence on
calibration under non-random order.** [Weighted conformal prediction](https://arxiv.org/abs/1904.06019) gives
valid intervals under covariate shift if the shift's likelihood ratio can be estimated. Applied to an estimated
reporting propensity, it would correct directly for order. I found no published election-night use of this.

**NYT "needle".** The NYT pages were not reachable from this environment. By secondary account, it sets a
pre-election expectation for every area from polls, voter files, census and past results. It then assumes that
over- or under-performance in one area carries over to "similar counties"
([Reuters Institute 2020](https://reutersinstitute.politics.ox.ac.uk/news/moving-needle-how-new-york-times-aims-guide-readers-through-americas-most-uncertain-election)).
**[method details unverified]**

**AP.** AP models an **expected total vote** from past turnout, early and mail ballots and registration,
instead of using "precincts reporting", because precincts differ in size
([AP via WBTV](https://www.wbtv.com/2026/03/02/how-ap-uses-expected-vote-instead-precincts-reporting-when-determining-winner)).
Race calls combine the count with AP VoteCast survey data
([PBS 2020](https://www.pbs.org/newshour/politics/how-the-ap-calls-races-and-what-to-expect-on-election-night)).
VoteCast has no SA equivalent, but the expected-vote idea maps onto turnout projection for uncounted VDs.

**Decision Desk HQ.** The live model mixes geographic and demographic components, with weights that shift as
results arrive, and simulates outcomes ([Donnini et al., HDSR 2024](https://doi.org/10.1162/99608f92.ccb395f0)).
**[full text blocked; abstract-level only]**

**The Economist.** I found no live-count model. Its documented work is pre-election forecasting
([Han, Gelman & Vehtari 2025](https://arxiv.org/abs/2510.19133)). **[not found]**

**Older US work.** NBC under Tukey (1962–1980) ran a **hierarchical (empirical-) Bayes** model with separate
turnout and swing parts, "key precincts", priors from the most similar past election, and "borrowing strength"
across units ([Fienberg 2011](https://arxiv.org/abs/1108.2177)).

## 3. UK, Europe, Latin America

**UK.** The BBC has used regression models with **probabilistic seat prediction** since the 1970s:
[Brown & Payne (1975)](https://academic.oup.com/jrsssa/article/138/4/463/7104885) (modified ridge regression) and
[Brown, Firth & Payne (1999)](https://ideas.repec.org/a/bla/jorssa/v162y1999i2p211-226.html) (adds the exit
poll as a prior, with bias correction).

The modern exit poll ([Curtice & Firth 2008](https://wrap.warwick.ac.uk/30068/)) works as follows:

* It measures **change at the same polling stations** (a panel design).
* It fits robust (bisquare) weighted least-squares regressions of nested change variables on constituency
  covariates, so that shares sum to 100.
* It turns predicted shares into win probabilities, with tuning constants calibrated on the previous election.

Their ablation for 2005 (actual Labour majority 66):

| | Deterministic seats | Probabilistic seats |
|---|---|---|
| Uniform change | 100 | 88 |
| Regression models | 84 | **66** |

Both regression and the probabilistic step matter.

**Uniform vs proportional swing.** The evidence is mixed. Uniform swing errs whenever change is not uniform
([Jackman 2014](https://www.cambridge.org/core/journals/ps-political-science-and-politics/article/abs/predictive-power-of-uniform-swing/571192E6749DBB5108D71D0DF7A656BC)).
Neither rule is consistently better on UK seats in 1997–2005
([Pack](https://markpack.org.uk/7556/uniform-versus-proportional-swing-which-is-best)).

**Log-ratio models.** For multiparty district data,
[Katz & King (1999)](https://gking.harvard.edu/files/multiparty.pdf) model log-ratios with an **additive
logistic Student-t**. It fits British data clearly better than the logistic-normal. That is an argument for
heavy-tailed VD residuals.

**Germany.** Forschungsgruppe Wahlen (ZDF, since 1974) and infratest dimap (ARD) project from official counts
in **pre-selected random samples of districts**, phoned in by correspondents. Projections are recomputed as more
sample districts arrive ([FGW](https://www.forschungsgruppe.de/Wahlen/Methodik/Methodik_Wahlen_1.pdf);
[infratest dimap FAQ](https://www.infratest-dimap.de/service/faqs/)). Order bias is avoided *by design*.
**[the exact estimator, e.g. change versus the previous election in the same districts, is not published;
unverified]**

**Austria.** The industry standard is described as **ecological regression of voter-transition multipliers,
estimated within groups of similar polling stations**. Experts build the groups by k-means. A genetic algorithm
for the grouping beat hand-made groups on Styria 2010
([Hochreiter & Waldhauser 2014](https://arxiv.org/abs/1401.4674), citing Hofinger & Ogris 2002 of SORA).

**Spain.**

* [Bernardo & Girón (1992)](https://academic.oup.com/book/54041/chapter/422209463): a **robust hierarchical
  multivariate regression** on every station's past results, with campaign surveys as the prior. It was built
  explicitly for "non-random samples". [Pavía-Miralles (2005)](https://econpapers.repec.org/RePEc:bes:jnlasa:v:100:y:2005:p:1113-1122)
  follows the same line.
* [Pavía, Larraz & Montero (2008)](https://doi.org/10.1198/016214507000001427): **kriging and cokriging** of
  station results. The abstract reports that they beat history-only models once more than **5%** of votes are
  counted. **[abstract via index; full text not read]**

**Mexico.** The INE quick count draws a stratified random sample of about 6,000 polling stations, but arrival is
incomplete and biased (67–95% received at the final estimate)
([Rodríguez, Nieto-Barajas & Pérez-Pérez 2024](https://arxiv.org/abs/2208.04475)):

* The Bayesian normal model of Mendoza & Nieto-Barajas (2016) is extended with **dynamic post-stratification**:
  a stratum with no data borrows from the finest cluster of past-result clusters that has data.
* Because coverage falls below nominal when data are incomplete, **the credible level is inflated**: 99% when
  less than 60% of the sample has arrived, 98% at 60–70%, and 95% at 90% or more.
* [Anzarut et al. (2019)](https://doi.org/10.1007/978-3-030-31551-1_1) use a heavy-tailed multilevel mixture
  with post-stratification.

**Other.** India: a hierarchical Bayesian model for counts arriving in batches
([Deb, Roy & Das 2024](https://arxiv.org/abs/2104.02924)). Brazil and Switzerland: no usable method
documentation found **[not covered]**.

## 4. Method families: how they handle order bias and new parties

Every method in this review handles order by assuming **reporting is ignorable given what the model conditions
on**: cluster membership, strata, covariates, own past result. Rodríguez et al. state this openly as
missing-at-random. None of them fixes deliberately held-back blocks (Gauteng 2014), which are
missing-not-at-random.

| Family | Key reference | Handling of order bias | Reported accuracy / calibration | Fit for Gauteng VDs |
|---|---|---|---|---|
| Raw tally | — | None | Needs 41–76% of votes for MAE < 1 pp ([Ittmann 2017](https://doi.org/10.5784/33-1-567)) | Display only |
| Uniform / ratio swing (province or municipality) | [Jackman 2014](https://www.cambridge.org/core/journals/ps-political-science-and-politics/article/abs/predictive-power-of-uniform-swing/571192E6749DBB5108D71D0DF7A656BC); [Curtice & Firth](https://wrap.warwick.ac.uk/30068/) | Compares each VD with its own past, but assumes homogeneous change | UK 2005: majority 100 forecast vs 66 actual | Baseline only; breaks for new parties |
| Cluster / stratified estimator (fuzzy c-means; grouped transition regression; post-stratification) | [Greben 2006](https://orion.journals.ac.za/pub/article/view/35); [Hochreiter 2014](https://arxiv.org/abs/1401.4674); [Rodríguez 2024](https://arxiv.org/abs/2208.04475) | Missing-at-random within cluster; needs counted VDs in every cluster | SA: top-3 MAE < 1 pp at 4–10% counted (2004–11). Fails under extreme bias or poor clusters (2014; Gauteng 2021) | High: direct SA prior art. Use as a feature or random effect and as an ensemble member |
| Regression of change on covariates | [Curtice & Firth](https://wrap.warwick.ac.uk/30068/); [Bernardo & Girón](https://academic.oup.com/book/54041/chapter/422209463); [WaPo 2020](https://s3.us-east-1.amazonaws.com/elex-models-prod/2020-general/write-up/election_model_writeup.pdf) | Models change relative to the unit's own baseline given covariates | UK 2005 exact seat majority; WaPo calibrated after 20–25% of units | High |
| Hierarchical / multilevel Bayes | [Fienberg 2011](https://arxiv.org/abs/1108.2177); [Rodríguez 2024](https://arxiv.org/abs/2208.04475); [Anzarut 2019](https://doi.org/10.1007/978-3-030-31551-1_1) | Borrowing strength across strata; priors from history or polls | Coverage below nominal with incomplete data; fixed by inflating the level | High (the current repo model) |
| Spatial (kriging, CAR) | [Pavía 2008](https://doi.org/10.1198/016214507000001427); [WaPo 2020](https://s3.us-east-1.amazonaws.com/elex-models-prod/2020-general/write-up/election_model_writeup.pdf) | Neighbours inform unreported units | Beats history-only above 5% counted **[abstract]**; Gaussian intervals under-cover (WaPo) | Medium: helps the mean, not trusted for intervals |
| ML (GBM, random forest) | none found for live counts | Only through the features it uses | **No published election-night evaluation found** | Low or medium: little data early; residual learner at most |
| Conformal / bootstrap intervals | [WaPo](https://jjcherian.github.io/assets/pdf/election_poster.pdf); [Gibbs & Candès](https://arxiv.org/abs/2106.00170); [Tibshirani et al.](https://arxiv.org/abs/1904.06019) | Exchangeability, adaptivity or reweighting | About 90% coverage once warmed up; adaptive version robust to time-zone order | High, as a calibration layer |
| Ensembles / stacking | [WaPo 2024 poster](https://jjcherian.github.io/assets/pdf/election_poster.pdf); [Yao et al.](https://arxiv.org/abs/2101.08954) | Combines models with different failure modes | WaPo: inverse-variance ensemble of fundamentals and extrapolation | High |
| Design-based sample (Hochrechnung, quick count, exit poll) | [FGW](https://www.forschungsgruppe.de/Wahlen/Methodik/Methodik_Wahlen_1.pdf); [Curtice & Firth](https://wrap.warwick.ac.uk/30068/) | Random sample of districts chosen beforehand | Not comparable | Idea only: a pre-chosen "key VD" panel to monitor |

**New parties** are where every method is weakest:

* CSIR coped with COPE in 2009 but under-projected the EFF (2014), ActionSA in Gauteng (2021) and MK (2024).
* WaPo and Curtice need a baseline share, so they cannot project a party without history.
* What works is to use VDs similar to the counted ones (clusters, covariates, neighbours) and related parties'
  past strength, with wider intervals.

## 5. Evidence on what works best

There is **no head-to-head comparison across families** in the literature. The best evidence is within-system
ablation:

* CSIR: fuzzy c-means beat k-means by about 13% on a VD-fit metric, but "the choice of the cluster method...
  does not seem to play a major role" ([Greben 2006](https://orion.journals.ac.za/pub/article/view/35)).
  Cluster quality relative to the *new* swing dominated in 2014
  ([Ittmann 2017](https://doi.org/10.5784/33-1-567)).
* UK: regression plus probabilistic seats beat uniform change ([Curtice & Firth](https://wrap.warwick.ac.uk/30068/)).
* WaPo: CQR beat Gaussian kriging on coverage
  ([2020 write-up](https://s3.us-east-1.amazonaws.com/elex-models-prod/2020-general/write-up/election_model_writeup.pdf)).
  The 2024 move to an ensemble was motivated by the instability of the "previous election model"; the poster
  plots fundamentals, extrapolation and ensemble errors for FL 2020
  ([poster](https://jjcherian.github.io/assets/pdf/election_poster.pdf)).
  **[I could not read the figure values, so "the ensemble is better" is unverified]**
* Order-robust calibration needs adaptivity ([Gibbs & Candès](https://arxiv.org/abs/2106.00170)) or an
  inflation schedule set by % counted ([Rodríguez 2024](https://arxiv.org/abs/2208.04475)).

Realistic targets from the SA record: **province-level error under 1 pp for big parties by about 10% of VDs** in
"normal" conditions, **1.5–2 pp at municipal level at 10–15% of VDs**, and larger errors for new or small parties.

## 6. Recommendations for Gauteng 2026

1. **Keep the hierarchical log-ratio Bayesian model as the main engine.** It already combines the strongest
   ideas in the literature: each VD compared with its own history (UK panel design, WaPo baseline), regression on
   past results (Curtice-Firth, Bernardo-Girón), and borrowing strength across municipalities (Tukey). Make the
   VD residuals **heavy-tailed** (Student-t in log-ratio space, per Katz & King) or use robust weights, so that
   capture errors and odd VDs do not drag their neighbours.
2. **Add a CSIR-style cluster swing term.** Fit fuzzy or k-means clusters on LGE2021 and PE2024 shares, with
   per-party standardisation so that MK, ActionSA and EFF count (the Ittmann lesson). Ideally also use the
   2016→2021 and 2019→2024 *changes*, because homogeneity of swing is what failed in 2014. Then give each cluster
   × party a random swing effect. Pick C in 10–40 by backtest. Keep the CAR term only where the hold-out shows a
   gain, and never rely on its Gaussian form for intervals.
3. **Run two cheap ensemble members:** (a) the Greben (2006) fuzzy-cluster estimator, which is transparent and
   familiar to SA media, and (b) a WaPo-style ridge or quantile regression of each VD's change on its own past
   shares, cluster memberships and municipality effects. Stack the three with weights fitted on the 2021 and
   2024 replays.
4. **Order bias:** condition on everything that predicts reporting order (VD size, municipality, cluster). Flag
   any municipality × cluster cell with too few counted VDs as provisional; WaPo used "a few dozen" units to start
   and "a few hundred" for calibration. Replay the real 2021 and 2024 IEC timestamps if they can be obtained, plus
   adversarial orders. Monitor late blocks of large VDs, as in Gauteng 2014; no model fixes missing-not-at-random
   data.
5. **Intervals and calibration.**
   * Measure coverage by level (VD, ward, municipality, province) and by % counted on the replays.
   * Recalibrate with a schedule of inflation against % counted (Mexico) or a conformal correction (WaPo).
   * During the night, score each newly arriving VD against its *pre-arrival* predictive distribution, as
     Greben et al. proposed. Feed these scores to adaptive conformal (Gibbs & Candès) so that coverage
     self-corrects if the order drifts.
   * Keep shared province, municipality and cluster effects in the draws, so that errors are correlated
     correctly when aggregated. WaPo found perfect correlation far too wide for precincts, and the errors
     "still far from independent".
   * Score ward-winner probabilities with Brier and reliability curves, and tune their spread on the previous
     election, as Curtice & Firth did.
6. **Do not make tree-based ML the primary method.** There is no election-night evidence for it, and data are
   scarce at 5–10% counted. At most, use it as a residual learner on history features, trained on the 2016→2021
   transitions.
7. **New parties:** in 2026 every modelled party has PE2024 history. Use it as the main baseline for MK and
   ActionSA, which removes the main CSIR failure mode. Keep the substitution and cluster pathways, with wider
   intervals, for any party without a usable baseline.

// Coalition scenario playbook, shown on the "Coalition scenarios" view.
// Long-form version: analysis/gauteng-coalition-game-plan.md. Keep the two in step.
//
// Every foreign case lists the main parties' vote shares (%) and each party's
// role in the government that formed:
//   gov     in government
//   support kept the government in office from outside (toleration, supply, or votes)
//   out     excluded / opposition

export type Role = "gov" | "support" | "out"

export interface Share {
  party: string
  pct: number
  role: Role
}

export interface Case {
  country: string
  year: string
  region: "cee" | "other"
  /** Vote shares are approximate where marked. */
  approx?: boolean
  shares: Share[]
  government: string
  lifespan: string
}

const s = (party: string, pct: number, role: Role): Share => ({ party, pct, role })

export const CASES: Record<string, Case> = {
  "poland-2005": {
    country: "Poland", year: "2005", region: "cee",
    shares: [s("PiS", 27.0, "gov"), s("PO", 24.1, "out"), s("Samoobrona", 11.4, "gov"), s("SLD", 11.3, "out"), s("LPR", 8.0, "gov"), s("PSL", 7.0, "out")],
    government: "PiS minority, then PiS + Samoobrona + LPR",
    lifespan: "About 15 months, then early election",
  },
  "poland-2007": {
    country: "Poland", year: "2007", region: "cee",
    shares: [s("PO", 41.5, "gov"), s("PiS", 32.1, "out"), s("LiD", 13.2, "out"), s("PSL", 8.9, "gov")],
    government: "PO + PSL",
    lifespan: "Full term; renewed in 2011",
  },
  "poland-2023": {
    country: "Poland", year: "2023", region: "cee",
    shares: [s("PiS", 35.4, "out"), s("KO", 30.7, "gov"), s("TD", 14.4, "gov"), s("Lewica", 8.6, "gov"), s("Konfederacja", 7.2, "out")],
    government: "KO + TD + Lewica",
    lifespan: "Ongoing; cohabitation with a rival-camp president",
  },
  "czechia-1998": {
    country: "Czechia", year: "1998", region: "cee",
    shares: [s("ČSSD", 32.3, "gov"), s("ODS", 27.7, "support"), s("KSČM", 11.0, "out"), s("KDU-ČSL", 9.0, "out"), s("US", 8.6, "out")],
    government: "ČSSD minority tolerated by its main rival ODS",
    lifespan: "Full term",
  },
  "czechia-2006": {
    country: "Czechia", year: "2006", region: "cee",
    shares: [s("ODS", 35.4, "gov"), s("ČSSD", 32.3, "out"), s("KSČM", 12.8, "out"), s("KDU-ČSL", 7.2, "gov"), s("Greens", 6.3, "gov")],
    government: "100–100 deadlock, broken when two opposition MPs walked out of the confidence vote",
    lifespan: "About 2 years",
  },
  "czechia-2017": {
    country: "Czechia", year: "2017", region: "cee",
    shares: [s("ANO", 29.6, "gov"), s("ODS", 11.3, "out"), s("Piráti", 10.8, "out"), s("SPD", 10.6, "out"), s("KSČM", 7.8, "support"), s("ČSSD", 7.3, "gov")],
    government: "ANO + ČSSD, communists tolerating",
    lifespan: "Nearly a full term",
  },
  "czechia-2021": {
    country: "Czechia", year: "2021", region: "cee",
    shares: [s("SPOLU", 27.8, "gov"), s("ANO", 27.1, "out"), s("PirStan", 15.6, "gov"), s("SPD", 9.6, "out")],
    government: "SPOLU + PirStan",
    lifespan: "Full term",
  },
  "slovakia-2006": {
    country: "Slovakia", year: "2006", region: "cee",
    shares: [s("Smer", 29.1, "gov"), s("SDKÚ", 18.4, "out"), s("SNS", 11.7, "gov"), s("SMK", 11.7, "out"), s("HZDS", 8.8, "gov"), s("KDH", 8.3, "out")],
    government: "Smer + SNS + HZDS",
    lifespan: "Full term; Smer suspended by the European socialists",
  },
  "slovakia-2010": {
    country: "Slovakia", year: "2010", region: "cee",
    shares: [s("Smer", 34.8, "out"), s("SDKÚ", 15.4, "gov"), s("SaS", 12.1, "gov"), s("KDH", 8.5, "gov"), s("Most-Híd", 8.1, "gov"), s("SNS", 5.1, "out")],
    government: "Four-party centre-right coalition",
    lifespan: "About 15 months",
  },
  "slovakia-2012": {
    country: "Slovakia", year: "2012", region: "cee",
    shares: [s("Smer", 44.4, "gov"), s("KDH", 8.8, "out"), s("OĽaNO", 8.6, "out"), s("Most-Híd", 6.9, "out"), s("SDKÚ", 6.1, "out"), s("SaS", 5.9, "out")],
    government: "Smer alone (a majority of seats from 44% of the vote)",
    lifespan: "Full term",
  },
  "slovakia-2020": {
    country: "Slovakia", year: "2020", region: "cee",
    shares: [s("OĽaNO", 25.0, "gov"), s("Smer", 18.3, "out"), s("Sme Rodina", 8.2, "gov"), s("ĽSNS", 8.0, "out"), s("SaS", 6.2, "gov"), s("Za ľudí", 5.8, "gov")],
    government: "OĽaNO-led four-party coalition",
    lifespan: "About 33 months",
  },
  "slovakia-2023": {
    country: "Slovakia", year: "2023", region: "cee",
    shares: [s("Smer", 22.9, "gov"), s("PS", 18.0, "out"), s("Hlas", 14.7, "gov"), s("OĽaNO", 8.9, "out"), s("KDH", 6.8, "out"), s("SaS", 6.3, "out"), s("SNS", 5.6, "gov")],
    government: "Smer + Hlas + SNS",
    lifespan: "Ongoing",
  },
  "romania-2020": {
    country: "Romania", year: "2020", region: "cee",
    shares: [s("PSD", 28.9, "out"), s("PNL", 25.2, "gov"), s("USR", 15.4, "gov"), s("AUR", 9.1, "out"), s("UDMR", 5.7, "gov")],
    government: "PNL + USR + UDMR",
    lifespan: "About 10 months",
  },
  "romania-2021": {
    country: "Romania", year: "2020 vote, 2021 government", region: "cee",
    shares: [s("PSD", 28.9, "gov"), s("PNL", 25.2, "gov"), s("USR", 15.4, "out"), s("AUR", 9.1, "out"), s("UDMR", 5.7, "gov")],
    government: "PNL + PSD + UDMR grand coalition with a rotating PM",
    lifespan: "Full term; rotation carried out on schedule",
  },
  "romania-2024": {
    country: "Romania", year: "2024", region: "cee",
    shares: [s("PSD", 22.0, "gov"), s("AUR", 18.0, "out"), s("PNL", 13.2, "gov"), s("USR", 12.4, "gov"), s("SOS", 7.4, "out"), s("POT", 6.5, "out"), s("UDMR", 6.3, "gov")],
    government: "Pro-EU parties excluding the radical bloc (USR joined in 2025)",
    lifespan: "Ongoing",
  },
  "bulgaria-2021": {
    country: "Bulgaria", year: "2021 (Nov)", region: "cee",
    shares: [s("PP", 25.7, "gov"), s("GERB", 22.7, "out"), s("DPS", 13.0, "out"), s("BSP", 10.2, "gov"), s("ITN", 9.5, "gov"), s("DB", 6.4, "gov")],
    government: "PP + BSP + ITN + DB",
    lifespan: "About 6 months",
  },
  "bulgaria-2023": {
    country: "Bulgaria", year: "2023", region: "cee",
    shares: [s("GERB", 26.5, "gov"), s("PP-DB", 24.6, "gov"), s("Vazrazhdane", 14.2, "out"), s("DPS", 13.8, "out"), s("BSP", 8.9, "out")],
    government: "GERB + PP-DB with a planned rotation",
    lifespan: "About 9 months; collapsed at the rotation",
  },
  "croatia-2015": {
    country: "Croatia", year: "2015", region: "cee",
    shares: [s("HDZ-led", 33.4, "gov"), s("SDP-led", 33.2, "out"), s("Most", 13.5, "gov")],
    government: "HDZ + Most with a non-party PM",
    lifespan: "About 5 months",
  },
  "croatia-2024": {
    country: "Croatia", year: "2024", region: "cee",
    shares: [s("HDZ", 34.4, "gov"), s("SDP-led", 25.4, "out"), s("DP", 9.6, "gov"), s("Možemo", 9.1, "out"), s("Most", 8.0, "out")],
    government: "HDZ + DP",
    lifespan: "Ongoing",
  },
  "slovenia-2018": {
    country: "Slovenia", year: "2018", region: "cee",
    shares: [s("SDS", 24.9, "out"), s("LMŠ", 12.6, "gov"), s("SD", 9.9, "gov"), s("SMC", 9.8, "gov"), s("Levica", 9.3, "support"), s("NSi", 7.2, "out"), s("SAB", 5.1, "gov"), s("DeSUS", 4.9, "gov")],
    government: "Five-party minority led by second-placed LMŠ, Levica supporting from outside",
    lifespan: "About 16 months",
  },
  "estonia-2019": {
    country: "Estonia", year: "2019", region: "cee",
    shares: [s("Reform", 28.9, "out"), s("Centre", 23.1, "gov"), s("EKRE", 17.8, "gov"), s("Isamaa", 11.4, "gov"), s("SDE", 9.8, "out")],
    government: "Centre + EKRE + Isamaa",
    lifespan: "About 21 months",
  },
  "latvia-2018": {
    country: "Latvia", year: "2018", region: "cee",
    shares: [s("Harmony", 19.8, "out"), s("KPV LV", 14.3, "gov"), s("JKP", 13.6, "gov"), s("AP", 12.0, "gov"), s("NA", 11.0, "gov"), s("ZZS", 9.9, "out"), s("JV", 6.7, "gov")],
    government: "Five-party coalition, PM from the smallest party (JV)",
    lifespan: "Full term",
  },
  "moldova-2019": {
    country: "Moldova", year: "2019", region: "cee",
    shares: [s("PSRM", 31.2, "gov"), s("ACUM", 26.8, "gov"), s("PDM", 23.6, "out"), s("Șor", 8.3, "out")],
    government: "PSRM + ACUM against PDM",
    lifespan: "About 5 months",
  },
  "north-macedonia-2016": {
    country: "North Macedonia", year: "2016", region: "cee", approx: true,
    shares: [s("VMRO-DPMNE", 38, "out"), s("SDSM", 37, "gov"), s("DUI", 7.3, "gov")],
    government: "SDSM + Albanian parties",
    lifespan: "Full term, after a crisis",
  },
  "hungary-2022": {
    country: "Hungary", year: "2022", region: "cee",
    shares: [s("Fidesz", 54.1, "gov"), s("United opposition", 34.4, "out"), s("Mi Hazánk", 5.9, "out")],
    government: "Fidesz; the left-to-right opposition alliance dissolved",
    lifespan: "–",
  },
  "germany-2017": {
    country: "Germany", year: "2017", region: "other",
    shares: [s("CDU/CSU", 32.9, "gov"), s("SPD", 20.5, "gov"), s("AfD", 12.6, "out"), s("FDP", 10.7, "out"), s("Linke", 9.2, "out"), s("Greens", 8.9, "out")],
    government: "CDU/CSU + SPD after pressure from the president",
    lifespan: "Full term",
  },
  "thuringia-2019": {
    country: "Germany (Thuringia)", year: "2019 vote, 2020 premier", region: "other",
    shares: [s("Linke", 31.0, "out"), s("AfD", 23.4, "support"), s("CDU", 21.7, "support"), s("SPD", 8.2, "out"), s("Greens", 5.2, "out"), s("FDP", 5.0, "gov")],
    government: "FDP premier elected with CDU and AfD votes, forced out by national pressure",
    lifespan: "Days",
  },
  "austria-2024": {
    country: "Austria", year: "2024", region: "other",
    shares: [s("FPÖ", 28.8, "out"), s("ÖVP", 26.3, "gov"), s("SPÖ", 21.1, "gov"), s("NEOS", 9.1, "gov"), s("Greens", 8.2, "out")],
    government: "ÖVP + SPÖ + NEOS after two failed rounds",
    lifespan: "Ongoing",
  },
  "ireland-2020": {
    country: "Ireland", year: "2020", region: "other",
    shares: [s("SF", 24.5, "out"), s("FF", 22.2, "gov"), s("FG", 20.9, "gov"), s("Greens", 7.1, "gov")],
    government: "FF + FG + Greens with a rotating Taoiseach",
    lifespan: "Full term",
  },
  "portugal-2015": {
    country: "Portugal", year: "2015", region: "other",
    shares: [s("PàF", 36.9, "out"), s("PS", 32.3, "gov"), s("BE", 10.2, "support"), s("CDU", 8.3, "support")],
    government: "PS minority with radical-left support",
    lifespan: "Full term",
  },
  "sweden-2022": {
    country: "Sweden", year: "2022", region: "other",
    shares: [s("S", 30.3, "out"), s("SD", 20.5, "support"), s("M", 19.1, "gov"), s("V", 6.8, "out"), s("C", 6.7, "out"), s("KD", 5.3, "gov"), s("MP", 5.1, "out"), s("L", 4.6, "gov")],
    government: "M + KD + L with SD support from outside",
    lifespan: "Ongoing",
  },
  "israel-2021": {
    country: "Israel", year: "2021", region: "other",
    shares: [s("Likud", 24.2, "out"), s("Yesh Atid", 13.9, "gov"), s("Shas", 7.2, "out"), s("Blue and White", 6.6, "gov"), s("Yamina", 6.2, "gov"), s("Labor", 6.1, "gov"), s("UTJ", 5.6, "out"), s("Yisrael Beiteinu", 5.6, "gov"), s("Rel. Zionism", 5.1, "out"), s("Joint List", 4.8, "out"), s("New Hope", 4.7, "gov"), s("Meretz", 4.6, "gov"), s("Ra'am", 3.8, "gov")],
    government: "Eight-party coalition led by Yamina",
    lifespan: "About 12 months",
  },
  "karnataka-2018": {
    country: "Karnataka (India)", year: "2018", region: "other",
    shares: [s("INC", 38.1, "gov"), s("BJP", 36.4, "out"), s("JD(S)", 18.3, "gov")],
    government: "JD(S) chief minister with INC (BJP had the most seats)",
    lifespan: "About 14 months",
  },
  "netherlands-2023": {
    country: "Netherlands", year: "2023", region: "other",
    shares: [s("PVV", 23.5, "gov"), s("GL-PvdA", 15.8, "out"), s("VVD", 15.2, "gov"), s("NSC", 12.9, "gov"), s("D66", 6.3, "out"), s("BBB", 4.7, "gov")],
    government: "Four-party coalition with a non-party PM",
    lifespan: "About 11 months",
  },
}

// ---------------------------------------------------------------------------

export interface ConfigRow {
  /** Key into CASES; omitted for cases without a single vote-share snapshot. */
  caseId?: string
  label: string
  outcome: string
  lifespan: string
  why: string
}

export interface Configuration {
  id: string
  n: number
  title: string
  short: string
  lesson: string
  rows: ConfigRow[]
}

export const CONFIGURATIONS: Configuration[] = [
  {
    id: "c1", n: 1,
    title: "One party at about 40% or more, with a small adjacent partner available",
    short: "One party ≈40%+",
    lesson: "At about 40%, the leading party picks its partner and the outcome is stable, often for a full term. The challenger's best play is portfolios, not the top job.",
    rows: [
      { caseId: "poland-2007", label: "Poland 2007", outcome: "PO + PSL (an agrarian, community-type partner)", lifespan: "Full term, renewed 2011", why: "Dominant partner, a small partner with a clear constituency, little overlap in voters" },
      { caseId: "croatia-2024", label: "Croatia 2024", outcome: "HDZ + DP. The kingmaker chose the larger party closer to it in values over the \"anyone but HDZ\" bloc", lifespan: "Ongoing", why: "Values pointed to the larger party, not to the bigger offer" },
      { caseId: "slovakia-2012", label: "Slovakia 2012", outcome: "Smer alone", lifespan: "Full term", why: "Wasted votes from small parties turned 44% of the vote into a majority of seats" },
    ],
  },
  {
    id: "c2", n: 2,
    title: "The liberal party and the anti-corruption challenger together exceed 50%. Do they coalesce?",
    short: "Liberal + challenger >50%",
    lesson: "Likely but not automatic. It works when one partner is clearly larger (about 1.5–2× or more), when they pre-committed before the vote, and when the challenger takes a defining portfolio rather than the top job. It fails when they are similar in size and fish in the same voter pool, or when the challenger's leader is highly personalist.",
    rows: [
      { caseId: "poland-2005", label: "Poland 2005", outcome: "The expected PiS–PO coalition of two neighbours did not form. PiS governed as a minority, then with two populist parties", lifespan: "About 15 months, then early election", why: "Similar sizes, the same voter pool, a fight over the top posts" },
      { caseId: "czechia-2021", label: "Czechia 2021", outcome: "SPOLU + PirStan formed the government; ANO (most seats) was excluded", lifespan: "Full term", why: "Pre-committed before the election; SPOLU clearly the senior partner" },
      { caseId: "slovakia-2020", label: "Slovakia 2020", outcome: "Challenger-led four-party government", lifespan: "About 33 months", why: "The challenger's personalist leader clashed with the liberal partner, which left" },
      { caseId: "latvia-2018", label: "Latvia 2018", outcome: "Five-party coalition with the challenger as one partner, led by the smallest party's leader", lifespan: "Full term", why: "The challenger was one of several partners; the compromise PM was trusted by all" },
      { caseId: "croatia-2015", label: "Croatia 2016", outcome: "Most (challenger) backed HDZ with a non-party PM", lifespan: "About 5 months; a second deal also broke", why: "The newcomer kept threatening to walk; the compromise PM had no base" },
      { caseId: "bulgaria-2021", label: "Bulgaria 2021", outcome: "Four-party anti-corruption coalition", lifespan: "About 6 months", why: "Another challenger (ITN) walked out; too many personalist leaders" },
    ],
  },
  {
    id: "c3", n: 3,
    title: "Three-way split: nobody near 40%",
    short: "Three-way split",
    lesson: "The largest party is often excluded, and second place can lead even with half the leader's vote. Governments are short unless the partners are institutional parties that agree a rotation in writing. Where new elections are available, deadlock produces repeat elections. Where they aren't, it produces repeated reshuffles.",
    rows: [
      { caseId: "slovenia-2018", label: "Slovenia 2018", outcome: "Second-placed LMŠ led a five-party minority, with Levica supporting from outside", lifespan: "About 16 months", why: "Outside support ended when the radical supporter's priorities diverged; the excluded largest party then governed with defectors" },
      { caseId: "estonia-2019", label: "Estonia 2019", outcome: "The second-placed party governed with the radical right", lifespan: "About 21 months", why: "Reached for the radical flank; a scandal ended it" },
      { caseId: "croatia-2015", label: "Croatia 2015", outcome: "Months of talks, then a technocrat PM", lifespan: "About 5 months", why: "Two equal blocs and a volatile kingmaker" },
      { caseId: "bulgaria-2021", label: "Bulgaria 2021–24", outcome: "Seven elections in about 3½ years", lifespan: "Governments of 6–9 months", why: "The anti-corruption camp refused the old dominant party, which couldn't find partners" },
      { caseId: "ireland-2020", label: "Ireland 2020", outcome: "FF + FG + Greens with a rotating Taoiseach; the party with most votes excluded", lifespan: "Full term", why: "Historic rivals united against a third force and shared the top job by rotation" },
    ],
  },
  {
    id: "c4", n: 4,
    title: "The declining dominant party plus its splinters and radical flank reach 50%",
    short: "Dominant party + splinters",
    lesson: "Splinters usually stay close to the parent on values. The likeliest outcome is reunion in coalition once the personal conflict is managed or the leadership changes. The price is high: the splinter's leader gets a top post.",
    rows: [
      { caseId: "slovakia-2023", label: "Slovakia 2023", outcome: "Smer + Hlas (its splinter) + SNS. The splinter returned to its parent instead of joining the liberals", lifespan: "Ongoing", why: "Values pulled the splinter home; it extracted the speakership, then the presidency" },
      { caseId: "slovakia-2006", label: "Slovakia 2006", outcome: "Smer with nationalist partners", lifespan: "Full term", why: "The European socialist party family suspended Smer for it, which didn't change the outcome" },
      { caseId: "czechia-2017", label: "Czechia 2017–18", outcome: "ANO minority lost a confidence vote; then ANO + ČSSD with communist toleration", lifespan: "Nearly a full term", why: "When the liberal parties refused, the leading party turned to the radical left's toleration" },
      { label: "Romania (PSD splinters)", outcome: "Pro România and ALDE split from or allied with PSD, then largely drifted back", lifespan: "–", why: "Personalist splinters are often temporary" },
    ],
  },
  {
    id: "c5", n: 5,
    title: "Neither values bloc reaches 50%, or a radical bloc must be excluded: the grand coalition of rivals",
    short: "Grand coalition of rivals",
    lesson: "Grand coalitions of rivals form when a higher authority pushes and the alternatives have failed. They last when both partners are institutional parties and the top-job question is settled in writing. They break when one partner exists to oppose the other.",
    rows: [
      { caseId: "romania-2021", label: "Romania 2021", outcome: "After the values-bloc coalition broke, historic rivals PNL and PSD governed together with a rotating PM, brokered by the president", lifespan: "Full term; rotation on schedule", why: "Both institutional parties; presidential brokerage; a fixed rotation made the bargain credible" },
      { caseId: "romania-2024", label: "Romania 2024", outcome: "Pro-EU rivals combined to exclude a radical bloc of about 32%", lifespan: "Ongoing", why: "A cordon around radicals forces rivals together" },
      { caseId: "czechia-1998", label: "Czechia 1998", outcome: "The \"opposition agreement\": the main rival tolerated a minority government in exchange for institutional posts", lifespan: "Full term", why: "Toleration instead of coalition lets rivals avoid sharing blame" },
      { caseId: "germany-2017", label: "Germany 2017–18", outcome: "The SPD had ruled out a grand coalition; after other talks collapsed and the president pressed it, it formed one", lifespan: "Full term", why: "Head-of-state pressure; the alternatives had visibly failed" },
      { caseId: "bulgaria-2023", label: "Bulgaria 2023", outcome: "Old dominant party and anti-corruption challenger governed with a planned rotation", lifespan: "About 9 months; collapsed at the rotation", why: "One partner's identity was built on opposing the other" },
      { caseId: "austria-2024", label: "Austria 2024–25", outcome: "A centrist deal failed, the largest-party option failed, then the centrist deal was revived", lifespan: "Ongoing", why: "Red lines dissolved round by round, then reformed" },
    ],
  },
  {
    id: "c6", n: 6,
    title: "\"Anyone but X\" coalitions spanning opposite values",
    short: "Anyone but X",
    lesson: "Coalitions held together only by a common enemy last roughly 5–15 months. The excluded largest party works actively to peel off members.",
    rows: [
      { caseId: "moldova-2019", label: "Moldova 2019", outcome: "Pro-Russian left and pro-EU reformers united to remove a dominant patronage party", lifespan: "About 5 months", why: "Once the common enemy was gone, nothing held them together" },
      { caseId: "slovakia-2010", label: "Slovakia 2010", outcome: "Four-party centre-right coalition excluding the 35% party", lifespan: "About 15 months", why: "One partner broke ranks on a single big vote" },
      { caseId: "israel-2021", label: "Israel 2021", outcome: "Eight parties from right to Arab Islamist, led by a 7-seat party", lifespan: "About 12 months", why: "Defections at the ideological edges" },
      { caseId: "karnataka-2018", label: "Karnataka 2018", outcome: "Third party's leader became chief minister with Congress support to block the BJP", lifespan: "About 14 months", why: "Defections engineered by the excluded largest party" },
      { caseId: "hungary-2022", label: "Hungary 2022", outcome: "A united left-to-right opposition lost, then dissolved", lifespan: "–", why: "Ideologically mixed alliances underperform with voters too" },
    ],
  },
  {
    id: "c7", n: 7,
    title: "Minority government with outside support",
    short: "Outside support",
    lesson: "Toleration lasts as long as the supporter gets something visible and its own voters don't punish it. Written agreements and institutional posts extend it; vague arrangements decay in 1–2 years.",
    rows: [
      { caseId: "czechia-1998", label: "Czechia 1998–2002", outcome: "The main rival tolerated in return for posts, including the speakership", lifespan: "Full term", why: "Institutional payoffs kept the tolerator invested" },
      { caseId: "czechia-2017", label: "Czechia 2018–21", outcome: "Communists tolerated ANO–ČSSD", lifespan: "Nearly a full term", why: "The tolerator withdrew near the end, when its voters punished it" },
      { caseId: "slovenia-2018", label: "Slovenia 2018–19", outcome: "Radical-left Levica backed a centrist minority", lifespan: "About 16 months", why: "The tolerator's policy demands weren't met" },
      { caseId: "portugal-2015", label: "Portugal 2015–19", outcome: "The radical left backed a Socialist minority", lifespan: "Full term; broke in 2021", why: "Held while the shared goal (ending austerity) held" },
      { caseId: "sweden-2022", label: "Sweden 2022–", outcome: "The largest party on the right backed a government without taking ministries", lifespan: "Ongoing", why: "Written policy agreement; influence without responsibility" },
    ],
  },
  {
    id: "c8", n: 8,
    title: "The top job goes to a small party or a neutral figure",
    short: "Compromise leader",
    lesson: "A compromise leader works when the big partners want stability and the leader is competent and non-threatening. It fails when the leader is a placeholder in an unresolved fight.",
    rows: [
      { caseId: "latvia-2018", label: "Latvia 2019", outcome: "PM from the smallest party in parliament (6.7%)", lifespan: "Full term, re-elected", why: "Success: the big partners trusted him more than each other's leaders" },
      { caseId: "croatia-2015", label: "Croatia 2016", outcome: "Non-party PM", lifespan: "About 5 months", why: "Failure: no base; the big partners kept fighting through him" },
      { caseId: "netherlands-2023", label: "Netherlands 2024", outcome: "Non-party PM because partners refused the largest party's leader", lifespan: "About 11 months", why: "Failure: the largest party walked out" },
      { caseId: "israel-2021", label: "Israel 2021", outcome: "PM from a 7-seat party", lifespan: "About 12 months", why: "Failure" },
      { caseId: "ireland-2020", label: "Ireland 2020", outcome: "Rotation between two rivals", lifespan: "Full term", why: "Success: written rotation, both institutional parties" },
    ],
  },
  {
    id: "c9", n: 9,
    title: "Rival camps at two levels of government (cohabitation)",
    short: "Cohabitation",
    lesson: "When two levels of government are held by rival camps, the conflict moves into legal powers, appointments and money, and the courts become the referee. It is survivable where powers are clearly divided.",
    rows: [
      { caseId: "poland-2023", label: "Poland 2023–", outcome: "A liberal-led government against a rival-camp president with veto powers", lifespan: "Ongoing", why: "Vetoes and fights over appointments and the courts; governing by workaround" },
      { label: "Romania 2012", outcome: "A government of one camp tried to remove the president of the other by impeachment", lifespan: "–", why: "The referendum failed on turnout; constitutional crisis" },
      { label: "France 1986–88, 1993–95, 1997–2002", outcome: "President and prime minister from rival camps", lifespan: "Stable", why: "Adversarial, but a clear division of powers made it work" },
    ],
  },
]

// ---------------------------------------------------------------------------

export interface Archetype {
  name: string
  values: string
  examples: string
  gauteng: string[]
}

export const ARCHETYPES: Archetype[] = [
  { name: "Declining dominant party", values: "Broad church, redistributive, strong patronage networks, office-seeking", examples: "PSD (Romania), Smer (Slovakia), GERB (Bulgaria), PDM (Moldova), SLD (Poland)", gauteng: ["ANC"] },
  { name: "Liberal-reformist party", values: "Market economics, rule of law, anti-corruption, institution-minded", examples: "PO/KO (Poland), ODS and SPOLU (Czechia), PNL and USR (Romania), PS and SaS (Slovakia), Reform (Estonia), DB (Bulgaria)", gauteng: ["DA"] },
  { name: "Anti-corruption challenger", values: "Founded by a businessman or outsider; anti-establishment, centrist-to-right on economics, hard on crime and often migration; personalist", examples: "ANO (Czechia), OĽaNO (Slovakia), Most (Croatia), PP and ITN (Bulgaria), KPV LV (Latvia), Res Publica (Estonia)", gauteng: ["ActionSA"] },
  { name: "Radical-left flank party", values: "Further left than the dominant party, often born from a rupture with it; prefers outside influence or toleration to shared responsibility", examples: "Levica (Slovenia), KSČM (Czechia), Bloco de Esquerda (Portugal)", gauteng: ["EFF"] },
  { name: "Personalist splinter led by a former leader", values: "Close to the parent on policy, hostile to its current leadership", examples: "Hlas (Pellegrini from Smer, Slovakia), Pro România (Ponta from PSD, Romania)", gauteng: ["MK"] },
  { name: "Community or identity kingmaker", values: "Represents a defined community; joins whoever governs in return for specific benefits", examples: "UDMR (Romania), DPS (Bulgaria), Most-Híd (Slovakia), Swedish People's Party (Finland), Albanian parties (North Macedonia)", gauteng: ["PA", "IFP", "FF+", "Al Jama-ah"] },
]

export type Distance = "Adjacent" | "Competitive" | "Rivals" | "Bridgeable" | "Far apart"

export interface DistanceRow {
  pair: string
  distance: Distance
  note: string
}

export const DISTANCES: DistanceRow[] = [
  { pair: "Liberal-reformist + community kingmaker", distance: "Adjacent", note: "Routine partnerships (Finland's Swedish People's Party; UDMR in Romanian centre-right governments)" },
  { pair: "Liberal-reformist + anti-corruption challenger", distance: "Competitive", note: "Close on policy but chasing the same voters. Coalitions form, then fight over the top post (Poland 2005 failed; Czechia 2021 worked; Slovakia 2020 broke)" },
  { pair: "Declining dominant party + its splinters or radical flank", distance: "Adjacent", note: "Close on policy, hostile personally. Splinters often return in coalition (Slovakia 2023)" },
  { pair: "Declining dominant party + community kingmaker", distance: "Adjacent", note: "Transactional and routine (UDMR with PSD; DPS with BSP)" },
  { pair: "Declining dominant party + anti-corruption challenger", distance: "Bridgeable", note: "Only if the challenger gets the top job or a clear anti-corruption win. Rare and short (Bulgaria 2023–24)" },
  { pair: "Liberal-reformist + declining dominant party", distance: "Rivals", note: "Grand coalitions when a third force is excluded or alternatives have failed (Romania 2021; Germany 2018; Czechia 1998 toleration)" },
  { pair: "Liberal-reformist + radical-left flank", distance: "Far apart", note: "At most outside support or toleration; almost never a shared government" },
  { pair: "Anti-corruption challenger + radical-left flank", distance: "Far apart", note: "Opposite on migration. Only in \"anyone but X\" coalitions, which break fast (Moldova 2019; Bulgaria 2021–22)" },
]

export const REGULARITIES: { rule: string; evidence: string }[] = [
  { rule: "At about 40% or more, the leading party chooses its partners and the government tends to last a full term.", evidence: "Poland 2007/2011, Croatia 2024, Slovakia 2012. Exception: when every partner refuses on values grounds (Czechia 2017)." },
  { rule: "At 30–35%, the largest party is excluded if the others can reach 50% and bridge their own differences.", evidence: "Poland 2023, Czechia 2021, Slovakia 2010, Estonia 2019, Slovenia 2018, Ireland 2020, Portugal 2015, Romania 2020. Exception: if the others are too far apart, the largest party gets back in (Slovenia 2020, Croatia 2024)." },
  { rule: "Two neighbours on values that pass 50% together usually coalesce, but not when they're similar in size and chasing the same voters.", evidence: "Poland 2005 versus Czechia 2021. Clear seniority and pre-commitment are what make it work." },
  { rule: "The pivotal party extracts more than its size.", evidence: "The top job (Latvia 2019, Israel 2021, Karnataka 2018) or a top constitutional post (Slovakia 2023, Czechia 1998)." },
  { rule: "Splinters of a dominant party tend to return to it in coalition when values align.", evidence: "Slovakia 2023." },
  { rule: "Anti-corruption challengers that lead governments tend to implode within 1–3 years; as junior partners they are often absorbed or fade.", evidence: "Slovakia 2020–22, Bulgaria 2021–22, Croatia 2016; Res Publica (Estonia) merged away; KPV LV (Latvia) disintegrated." },
  { rule: "Coalitions of four or more parties spanning a wide values range last about 5–15 months; two or three neighbours with a clear senior partner usually last a full term.", evidence: "Moldova 2019, Bulgaria 2021, Slovakia 2010 versus Poland 2007, Czechia 2021." },
  { rule: "Outside support decays unless the supporter gets visible, written, institutional payoffs.", evidence: "Czechia 1998 versus Slovenia 2018." },
  { rule: "Grand coalitions of historic rivals need an outside push and a settled top-job arrangement.", evidence: "Romania 2021 versus Bulgaria 2023." },
  { rule: "Higher authorities shape outcomes, but mostly at the margins.", evidence: "Presidents broker (Romania 2021, Germany 2018); party families sanction (Smer 2006 and 2023, Fidesz 2019); money is withheld (EU funds for Hungary and Poland); national parties overrule regional branches (Thuringia 2020)." },
  { rule: "Equal-bloc deadlocks are broken by individual defectors.", evidence: "Czechia 2006–07: a 100–100 parliament resolved when two opposition MPs walked out." },
  { rule: "Where new elections aren't available, deadlock turns into churn; where they are, into repeated elections.", evidence: "Bulgaria 2021–24 and Israel 2019–22 repeated elections; councils without that option get repeated changes of leader." },
]

export const DEEP_DIVE = {
  factors: [
    { factor: "Relative size", durable: "Liberal party 2× the challenger or more", fragile: "Similar sizes (under about 1.3×)", cases: "Czechia 2021 (≈1.8×) vs Poland 2005 (≈1.1×)" },
    { factor: "Pre-commitment", durable: "Announced pre-election pact", fragile: "Post-election auction", cases: "Czechia 2021, Poland 2023 vs Poland 2005" },
    { factor: "Challenger's alternative", durable: "No credible offer from the other side", fragile: "The other side offers the challenger the top job", cases: "Croatia 2024 vs Croatia 2016" },
    { factor: "Personalism", durable: "Institutional leadership", fragile: "Founder-led, personalist challenger", cases: "Latvia 2018 vs Slovakia 2020, Croatia 2016" },
  ],
  scenarios: [
    { when: "Clear seniority, no rival offer", then: "The liberal party leads; the challenger takes a signature portfolio (security, policing, anti-corruption). Full term likely." },
    { when: "Clear seniority, rival offer exists", then: "The liberal party leads but pays a premium: more portfolios, the speakership, written guarantees. 2–4 years; risk of a mid-term walkout." },
    { when: "Similar size", then: "A fight over the top job: rotation (Ireland, Romania) or a compromise leader (Latvia). A real risk of no deal (Poland 2005)." },
    { when: "Challenger larger", then: "The challenger leads. Challenger-led governments are fragile (Slovakia 2020, Bulgaria 2021): expect 1–3 years." },
  ],
  gamson: "Across parliamentary democracies, partners receive cabinet posts roughly in proportion to the seats they bring (Gamson's law), with small pivotal partners slightly over-rewarded. A challenger bringing a third of the coalition's seats should expect about a third of the executive posts, plus a premium if pivotal.",
}

export const LEVERS: { lever: string; examples: string; achieved: string; limits: string }[] = [
  { lever: "Head of state or national leader brokering", examples: "Romania 2021; Germany 2017–18; Czech presidents appointing minority PMs", achieved: "Pushed rivals into grand coalitions; legitimised minority governments", limits: "Can't force members to vote for it" },
  { lever: "Overruling a regional or local branch", examples: "Thuringia 2020", achieved: "Reversed a regional deal within days", limits: "The national leader paid a price" },
  { lever: "Party-family sanctions", examples: "Smer suspended (2006, 2023); Fidesz suspended (2019)", achieved: "Signalled disapproval", limits: "Rarely changed the coalition" },
  { lever: "Money", examples: "EU funds frozen for Hungary and Poland", achieved: "Changed behaviour slowly", limits: "Slow; hits citizens too" },
  { lever: "Rules for forming governments", examples: "Poland 2023: the largest party got the first attempt and failed", achieved: "Ritual delay", limits: "Rarely decisive" },
  { lever: "Cohabitation vetoes", examples: "Poland 2023–", achieved: "Blocked legislation and appointments", limits: "Workarounds; courts referee" },
]

// ---------------------------------------------------------------------------
// Mapping onto Gauteng (official declared vote shares → configuration)

export const COUNCILS = [
  { name: "Johannesburg", seats: 270, majority: 136 },
  { name: "Tshwane", seats: 214, majority: 108 },
  { name: "Ekurhuleni", seats: 224, majority: 113 },
]

export interface GautengRow {
  band: string
  configId: string
  variant?: string
  government: string
  lifespan: string
  cases: string
}

export const GAUTENG_MAP: GautengRow[] = [
  { band: "DA about 40%+", configId: "c1", government: "DA with small partners; ActionSA as junior with a signature portfolio, or outside", lifespan: "Full term", cases: "Poland 2007/2011, Croatia 2024" },
  { band: "DA 30–39%, ActionSA 12–20%, together ≥50%, DA ≥2× ActionSA", configId: "c2", variant: "clear seniority", government: "DA mayor; ActionSA takes security and policing portfolios; a small insurance partner", lifespan: "2–5 years", cases: "Czechia 2021, Latvia 2018" },
  { band: "DA + ActionSA ≥50% but similar in size", configId: "c2", variant: "similar size", government: "Fight over the mayoralty: rotation, compromise mayor, or no deal", lifespan: "1–3 years; risk of no deal", cases: "Poland 2005, Ireland 2020, Latvia 2019" },
  { band: "ActionSA larger than the DA, together ≥50%", configId: "c2", variant: "challenger leads", government: "ActionSA mayor with DA support", lifespan: "1–3 years", cases: "Slovakia 2020, Bulgaria 2021" },
  { band: "DA + ActionSA <50%; ANC + EFF + MK + community parties ≥50%", configId: "c4", government: "ANC with its splinters and community kingmakers", lifespan: "Medium; tension with the national unity government", cases: "Slovakia 2023, Czechia 2018" },
  { band: "Three-way split (DA, ActionSA, ANC each about 20–30%)", configId: "c3", government: "Second or third place may lead; rotations or compromise mayors", lifespan: "6–20 months per arrangement", cases: "Slovenia 2018, Estonia 2019, Croatia 2015" },
  { band: "Neither bloc reaches 50%", configId: "c5", government: "ANC–DA grand coalition or toleration pact, if national leaders push", lifespan: "Durable only with a settled mayoralty in writing", cases: "Romania 2021, Czechia 1998, Germany 2018 vs Bulgaria 2023" },
  { band: "\"Anyone but the DA\" or \"anyone but the ANC\"", configId: "c6", government: "A common-enemy coalition", lifespan: "About 5–15 months", cases: "Moldova 2019, Slovakia 2010, Israel 2021" },
  { band: "A minority mayor elected on outside votes", configId: "c7", government: "Toleration", lifespan: "1–2 years unless written and rewarded", cases: "Czechia 1998 vs Slovenia 2018" },
]

export const LOCAL_RULES = [
  { rule: "Mayors are elected by secret ballot on a majority of votes cast, with elimination rounds.", effect: "A plurality candidate can win if opponents split; defectors and abstainers decide close votes (regularity 11)." },
  { rule: "Councils can't call new elections at will.", effect: "Deadlock shows up as churn (removals, compromise mayors, caretakers) rather than repeat elections (regularity 12)." },
]

export const PROVINCE = {
  summary: "The Gauteng legislature isn't up for election (next in 2029), and its ANC-led government relies on smaller parties and the EFF. If DA-led coalitions take the metros, Gauteng becomes a cohabitation: a province run by one camp, its largest cities by the other (Configuration 9).",
  expect: [
    "Conflict over legal powers (provincial intervention in municipalities), money (grants) and appointments",
    "The courts as referee",
    "Stability only where powers are clearly divided",
  ],
  grandCoalition: "If an ANC–DA grand coalition forms in the metros (Configuration 5), the EFF's support for the provincial government comes under strain, as junior partners were sidelined when Romania shifted to a grand coalition in 2021.",
}

export const NATIONAL_PRESSURE = {
  summary: "The ANC's and DA's national leaderships play the role foreign presidents and party families play.",
  levers: [
    "Broker grand coalitions",
    "Overrule regional branches (as in Thuringia 2020)",
    "Lean on the small parties that sit with them in the national government",
    "Control money flows through national government",
  ],
  verdict: "Foreign experience says this pressure decides close cases and rarely overturns clear local majorities.",
}

export const WATCH = [
  "Declared vote shares in each metro, matched to the configuration table",
  "The DA-to-ActionSA size ratio (regularity 3)",
  "Whether the ANC offers ActionSA the mayoralty: the challenger's outside option",
  "Whether national, not regional, leaderships sign coalition agreements",
  "Written agreements and rotation dates",
  "Secret-ballot results compared with announced deals (regularity 11)",
]

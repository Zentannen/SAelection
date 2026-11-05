import { useEffect, useMemo, useState } from "react"
import { loadHistory, loadPredictions, loadResults, loadWards, ranked, resultShares } from "@/data"
import { OTHER } from "@/parties"
import type { CountUpdate, Election, HistoryFile, PredictionsFile, ResultsFile, ResultStatus, Shares, WardCollection, WardFeature } from "@/types"

/**
 * Shares a forecast doesn't list count as "Other": top the set up to 100% so
 * every bar, chart and headline uses the same, true percentages.
 */
export function completeShares(shares: Shares): Shares {
  const total = Object.values(shares).reduce((a, b) => a + b, 0)
  // Allow for rounding (ten parties at one decimal can sum to 99.5–100.5).
  if (total >= 99.5) return shares
  return { ...shares, [OTHER]: (shares[OTHER] ?? 0) + (100 - total) }
}

/** Parties ranked by share, leaving out the "Other" remainder (it is never a winner). */
export const rankedParties = (shares: Shares) => ranked(shares).filter(([p]) => p !== OTHER)

/** Share of called wards where the predicted winner leads; only 100 when there are no upsets. */
export function holdingPct(hits: number, called: number): number | null {
  if (!called) return null
  if (hits === called) return 100
  return Math.min(99, Math.round((hits / called) * 100))
}

export const RESULTS_POLL_MS = 60_000
const NO_ELECTIONS: Election[] = []

export interface PartyRow {
  party: string
  /** Current share of counted votes, 0–100 (undefined if no result yet). */
  current?: number
  /** Predicted share, 0–100. */
  predicted?: number
}

export interface WardSummary {
  feature: WardFeature
  id: string
  district?: string
  status: ResultStatus
  /** Predicted shares straight from predictions.json (0–100). */
  predictedShares?: Shares
  predictedWinner?: string
  confidence?: number
  /** Raw vote counts so far (empty when no result yet). */
  votes: Record<string, number>
  leader?: string
  /** Leader's lead over second place, percentage points. */
  leadMargin?: number
  votesCounted: number
  registered?: number
  /** Parties ordered by current share (or predicted share if no result yet). */
  rows: PartyRow[]
  /** true when there is a leader and they differ from the predicted winner. */
  upset: boolean
  /** Cumulative counts at each results update on election night, oldest first. */
  timeline: CountUpdate[]
  /** Past election id -> vote share (0–100), from history.json. */
  history?: Record<string, Shares>
}

export function summarise(
  f: WardFeature,
  predictions: PredictionsFile,
  results: ResultsFile,
  history?: HistoryFile | null,
): WardSummary {
  const id = f.properties.ward_id
  const pred = predictions.wards[id]
  const res = results.wards[id]

  const predShares = pred ? completeShares(pred.shares) : undefined
  const predRanked = predShares ? ranked(predShares) : []
  const predictedWinner = pred ? rankedParties(pred.shares)[0]?.[0] : undefined

  const hasResult = !!res && res.status !== "not_started" && Object.values(res.votes).some((v) => v > 0)
  const { shares, total } = hasResult ? resultShares(res) : { shares: {}, total: 0 }
  const curRanked = ranked(shares)
  const leader = curRanked[0]?.[0]
  const leadMargin = curRanked.length > 1 ? curRanked[0][1] - curRanked[1][1] : curRanked[0]?.[1]

  const order = (hasResult ? curRanked : predRanked).map(([p]) => p)
  for (const [p] of predRanked) if (!order.includes(p)) order.push(p)
  const rows = order.map((party) => ({
    party,
    current: hasResult ? (shares[party] ?? 0) : undefined,
    predicted: predShares?.[party],
  }))

  return {
    feature: f,
    id,
    district: f.properties.district || undefined,
    status: hasResult ? res.status : "not_started",
    predictedShares: predShares,
    predictedWinner,
    confidence: pred?.confidence,
    votes: hasResult ? res.votes : {},
    leader,
    leadMargin,
    votesCounted: total,
    registered: res?.registered,
    rows,
    upset: !!leader && !!predictedWinner && leader !== predictedWinner,
    timeline: hasResult ? (res.timeline ?? []) : [],
    history: history?.wards[id],
  }
}

/** Prediction + live count rolled up over a set of wards (the whole area or one district). */
export interface ScopeSummary {
  wardCount: number
  /** Wards that have a forecast in predictions.json. */
  forecastWards: number
  reporting: number
  final: number
  /** Reporting wards that have both a leader and a predicted winner. */
  called: number
  /** Called wards where the leader is the predicted winner. */
  hits: number
  upsets: number
  /** Predicted vote share (0–100) for the scope. */
  predictedShares: Shares
  /** "model" when predictions.json supplied the figure, "wards" when built from ward predictions. */
  predictedSource: "model" | "wards"
  /** Share (0–100) of all votes counted so far in the scope. */
  countedShares: Shares
  votesCounted: number
  /** Number of wards each party is predicted to win / currently leads. */
  predictedWins: Record<string, number>
  leadingWins: Record<string, number>
}

/**
 * Roll ward figures up to a scope. Predicted shares are a weighted average of
 * ward predictions, weighted by registered voters. A ward without that figure
 * gets `fallbackWeight` (the average across all wards), so one missing number
 * nudges the result instead of changing the method. With no registered
 * figures at all every ward counts equally. A model-supplied figure for the
 * scope is used as-is (topped up with "Other").
 */
export function aggregate(wards: WardSummary[], modelShares?: Shares, fallbackWeight?: number): ScopeSummary {
  const reporting = wards.filter((w) => w.status !== "not_started")
  const called = reporting.filter((w) => w.leader && w.predictedWinner)
  const hits = called.filter((w) => !w.upset).length

  const predicted = wards.filter((w) => w.predictedShares)
  const sums: Shares = {}
  let weightTotal = 0
  for (const w of predicted) {
    const weight = (w.registered ?? 0) > 0 ? w.registered! : (fallbackWeight ?? 1)
    weightTotal += weight
    for (const [p, s] of Object.entries(w.predictedShares!)) sums[p] = (sums[p] ?? 0) + s * weight
  }
  const wardShares: Shares = {}
  for (const [p, s] of Object.entries(sums)) wardShares[p] = weightTotal ? s / weightTotal : 0

  const votes: Record<string, number> = {}
  let votesCounted = 0
  for (const w of reporting) {
    for (const [p, v] of Object.entries(w.votes)) {
      votes[p] = (votes[p] ?? 0) + v
      votesCounted += v
    }
  }
  const countedShares: Shares = {}
  for (const [p, v] of Object.entries(votes)) countedShares[p] = votesCounted ? (v / votesCounted) * 100 : 0

  const tally = (pick: (w: WardSummary) => string | undefined) => {
    const t: Record<string, number> = {}
    for (const w of wards) {
      const p = pick(w)
      if (p) t[p] = (t[p] ?? 0) + 1
    }
    return t
  }

  return {
    wardCount: wards.length,
    forecastWards: predicted.length,
    reporting: reporting.length,
    final: reporting.filter((w) => w.status === "final").length,
    called: called.length,
    hits,
    upsets: called.length - hits,
    predictedShares: modelShares ? completeShares(modelShares) : wardShares,
    predictedSource: modelShares ? "model" : "wards",
    countedShares,
    votesCounted,
    predictedWins: tally((w) => w.predictedWinner),
    leadingWins: tally((w) => w.leader),
  }
}

export interface DistrictSummary extends ScopeSummary {
  name: string
  wards: WardSummary[]
  municipalities: string[]
}

type State =
  | { status: "loading" }
  | { status: "error"; message: string }
  | {
      status: "ready"
      wards: WardCollection
      predictions: PredictionsFile
      results: ResultsFile
      history: HistoryFile | null
    }

export function useElectionData() {
  const [state, setState] = useState<State>({ status: "loading" })

  useEffect(() => {
    let cancelled = false
    Promise.all([loadWards(), loadPredictions(), loadResults(), loadHistory()])
      .then(([wards, predictions, results, history]) => {
        if (!cancelled) setState({ status: "ready", wards, predictions, results, history })
      })
      .catch((err: Error) => !cancelled && setState({ status: "error", message: err.message }))

    // Results change on election night; predictions and boundaries don't.
    const timer = setInterval(async () => {
      try {
        const results = await loadResults()
        if (!cancelled) setState((s) => (s.status === "ready" ? { ...s, results } : s))
      } catch (err) {
        console.warn("Results refresh failed", err)
      }
    }, RESULTS_POLL_MS)

    return () => {
      cancelled = true
      clearInterval(timer)
    }
  }, [])

  const summaries = useMemo(
    () =>
      state.status === "ready"
        ? state.wards.features.map((f) => summarise(f, state.predictions, state.results, state.history))
        : [],
    [state],
  )

  const predictions = state.status === "ready" ? state.predictions : undefined

  // Average registered voters per ward: the weight for wards missing that figure.
  const fallbackWeight = useMemo(() => {
    const known = summaries.map((w) => w.registered ?? 0).filter((r) => r > 0)
    return known.length ? known.reduce((a, b) => a + b, 0) / known.length : undefined
  }, [summaries])

  const overall = useMemo(
    () => aggregate(summaries, predictions?.overall?.shares, fallbackWeight),
    [summaries, predictions, fallbackWeight],
  )

  const districts = useMemo(() => {
    const groups = new Map<string, WardSummary[]>()
    for (const w of summaries) {
      if (!w.district) continue
      if (!groups.has(w.district)) groups.set(w.district, [])
      groups.get(w.district)!.push(w)
    }
    return [...groups.entries()]
      .sort(([a], [b]) => a.localeCompare(b))
      .map(
        ([name, wards]): DistrictSummary => ({
          ...aggregate(wards, predictions?.districts?.[name]?.shares, fallbackWeight),
          name,
          wards,
          municipalities: [...new Set(wards.map((w) => w.feature.properties.municipality).filter((m): m is string => !!m))],
        }),
      )
  }, [summaries, predictions, fallbackWeight])

  const history = state.status === "ready" ? state.history : null
  const elections = useMemo(() => history?.elections ?? NO_ELECTIONS, [history])

  return { state, summaries, overall, districts, elections }
}

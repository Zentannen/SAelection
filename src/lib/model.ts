import { useEffect, useMemo, useState } from "react"
import { loadPredictions, loadResults, loadWards, ranked, resultShares } from "@/data"
import type { PredictionsFile, ResultsFile, ResultStatus, WardCollection, WardFeature } from "@/types"

export const RESULTS_POLL_MS = 60_000

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
  status: ResultStatus
  predictedWinner?: string
  confidence?: number
  leader?: string
  /** Leader's lead over second place, percentage points. */
  leadMargin?: number
  votesCounted: number
  registered?: number
  /** Parties ordered by current share (or predicted share if no result yet). */
  rows: PartyRow[]
  /** true when there is a leader and they differ from the predicted winner. */
  upset: boolean
}

export function summarise(f: WardFeature, predictions: PredictionsFile, results: ResultsFile): WardSummary {
  const id = f.properties.ward_id
  const pred = predictions.wards[id]
  const res = results.wards[id]

  const predRanked = pred ? ranked(pred.shares) : []
  const predictedWinner = predRanked[0]?.[0]

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
    predicted: pred?.shares[party],
  }))

  return {
    feature: f,
    id,
    status: hasResult ? res.status : "not_started",
    predictedWinner,
    confidence: pred?.confidence,
    leader,
    leadMargin,
    votesCounted: total,
    registered: res?.registered,
    rows,
    upset: !!leader && !!predictedWinner && leader !== predictedWinner,
  }
}

type State =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; wards: WardCollection; predictions: PredictionsFile; results: ResultsFile }

export function useElectionData() {
  const [state, setState] = useState<State>({ status: "loading" })

  useEffect(() => {
    let cancelled = false
    Promise.all([loadWards(), loadPredictions(), loadResults()])
      .then(([wards, predictions, results]) => {
        if (!cancelled) setState({ status: "ready", wards, predictions, results })
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
        ? state.wards.features.map((f) => summarise(f, state.predictions, state.results))
        : [],
    [state],
  )

  return { state, summaries }
}

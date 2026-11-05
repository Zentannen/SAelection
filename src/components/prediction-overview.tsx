import { memo } from "react"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { SegmentLegend, SegmentedBar } from "@/components/segmented-bar"
import { ranked } from "@/data"
import { holdingPct, rankedParties, type ScopeSummary } from "@/lib/model"
import { partyColor, partyName } from "@/parties"

const int = (n: number) => n.toLocaleString("en-ZA")

function Meter({ value }: { value: number }) {
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
      <div className="h-full rounded-full bg-foreground transition-[width] duration-700" style={{ width: `${value * 100}%` }} />
    </div>
  )
}

function Stat({ label, value, children }: { label: string; value: React.ReactNode; children?: React.ReactNode }) {
  return (
    <div className="grid min-w-0 content-start gap-1">
      <p className="text-xs text-muted-foreground">{label}</p>
      <div className="text-xl font-semibold tracking-tight">{value}</div>
      {children}
    </div>
  )
}

interface Props {
  scope: ScopeSummary
  title: string
  description: string
  /** Bucketed party order shared by every segmented bar on the page. */
  order: string[]
  /** true when the scope elects its own council, so an outright majority means something. */
  council?: boolean
}

/** Headline prediction for the whole area, or for one district when it is selected. */
export const PredictionOverview = memo(function PredictionOverview({ scope, title, description, order, council = true }: Props) {
  const predicted = rankedParties(scope.predictedShares)
  const [leader, leaderShare = 0] = predicted[0] ?? []
  const [runnerUp, runnerUpShare = 0] = predicted[1] ?? []
  const majority = leaderShare > 50
  const holding = holdingPct(scope.hits, scope.called)

  const wins = ranked(scope.predictedWins).slice(0, 3)

  return (
    <Card className="gap-4 desk:min-h-0 desk:overflow-auto">
      <CardHeader>
        <CardTitle className="text-lg">{title}</CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent>
        {/* Wide row under the header on smaller screens; a single column in the desktop sidebar. */}
        <div className="grid grid-cols-[250px_minmax(0,1fr)_330px] gap-8 desk:grid-cols-1 desk:gap-6">
          {/* Hero: predicted largest party and its share. */}
          <div className="grid content-start gap-2 border-r pr-8 desk:border-r-0 desk:border-b desk:pr-0 desk:pb-6">
            <p className="text-sm text-muted-foreground">Predicted largest party</p>
            {leader ? (
              <>
                <div className="flex items-center gap-2.5">
                  <span className="size-3.5 rounded-[4px]" style={{ background: partyColor(leader) }} />
                  <span className="text-xl font-semibold">{leader}</span>
                  <span className="truncate text-sm text-muted-foreground">{partyName(leader)}</span>
                </div>
                <p className="text-5xl font-semibold tracking-tight">{leaderShare.toFixed(1)}%</p>
                <div className="flex flex-wrap items-center gap-2 text-sm text-muted-foreground">
                  {council ? (
                    <Badge variant={majority ? "default" : "secondary"}>
                      {majority ? "Outright majority" : "No outright majority"}
                    </Badge>
                  ) : (
                    <Badge variant="secondary" title="Councils are elected per municipality; there is no Gauteng-wide council">
                      Province-wide share
                    </Badge>
                  )}
                  {runnerUp && (
                    <span>
                      {(leaderShare - runnerUpShare).toFixed(1)} pts ahead of {runnerUp}
                    </span>
                  )}
                </div>
              </>
            ) : (
              <p className="text-sm text-muted-foreground">No predictions for this area.</p>
            )}
          </div>

          {/* Predicted vote share vs the count so far, same block order. */}
          <div className="grid content-start gap-4">
            <div className="grid gap-2">
              <div className="flex items-baseline justify-between gap-4 text-sm desk:flex-col desk:items-start desk:gap-0.5">
                <span className="font-medium">Predicted vote share</span>
                <span className="text-xs text-muted-foreground">
                  {scope.predictedSource === "model" ? "Model forecast" : scope.forecastWards ? `Built from ${int(scope.forecastWards)} ward forecasts` : "No forecast loaded yet"}
                </span>
              </div>
              <SegmentedBar shares={scope.predictedShares} order={order} />
            </div>
            <div className="grid gap-2">
              <div className="flex items-baseline justify-between gap-4 text-sm desk:flex-col desk:items-start desk:gap-0.5">
                <span className="font-medium">Counted so far</span>
                <span className="text-xs text-muted-foreground tabular-nums">
                  {int(scope.votesCounted)} votes · {scope.reporting} of {scope.wardCount} wards reporting
                </span>
              </div>
              {scope.votesCounted ? (
                <SegmentedBar shares={scope.countedShares} order={order} />
              ) : (
                <div className="grid h-7 place-items-center rounded-[4px] border border-dashed text-xs text-muted-foreground">
                  No votes counted yet
                </div>
              )}
            </div>
            <SegmentLegend order={order} shares={[scope.predictedShares, scope.countedShares]} />
          </div>

          {/* Progress of the count against the forecast. */}
          <div className="grid grid-cols-2 content-start gap-x-6 gap-y-5 border-l pl-8 desk:border-l-0 desk:border-t desk:pt-6 desk:pl-0">
          <Stat
            label="Wards reporting"
            value={
              <>
                {scope.reporting}
                <span className="text-sm font-normal text-muted-foreground"> / {scope.wardCount}</span>
              </>
            }
          >
            <Meter value={scope.wardCount ? scope.reporting / scope.wardCount : 0} />
            <p className="text-xs text-muted-foreground">
              {scope.final} final · {scope.reporting - scope.final} counting
            </p>
          </Stat>
          <Stat label="Prediction holding" value={holding == null ? "–" : `${holding}%`}>
            <p className="text-xs text-muted-foreground">
              Predicted winner leads in {scope.hits} of {scope.called}
            </p>
          </Stat>
          <Stat label="Upsets so far" value={scope.upsets}>
            <p className="text-xs text-muted-foreground">Another party leads</p>
          </Stat>
          <Stat
            label="Most ward wins (forecast)"
            value={
              wins[0] ? (
                <span className="flex items-center gap-2">
                  <span className="size-3 rounded-[3px]" style={{ background: partyColor(wins[0][0]) }} />
                  {wins[0][0]} {wins[0][1]}
                  <span className="text-sm font-normal text-muted-foreground">/ {scope.wardCount}</span>
                </span>
              ) : (
                "–"
              )
            }
          >
            <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs text-muted-foreground">
              {wins.slice(1).map(([p, n]) => (
                <span key={p} className="flex items-center gap-1.5">
                  <span className="size-2 rounded-[2px]" style={{ background: partyColor(p) }} />
                  <span className="text-foreground">{p}</span> {n}
                </span>
              ))}
            </div>
          </Stat>
          </div>
        </div>
      </CardContent>
    </Card>
  )
})

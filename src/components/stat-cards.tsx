import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { partyColor, partyName } from "@/parties"
import type { WardSummary } from "@/lib/model"

function Meter({ value }: { value: number }) {
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
      <div className="h-full rounded-full bg-foreground transition-[width] duration-700" style={{ width: `${value * 100}%` }} />
    </div>
  )
}

export function StatCards({ wards }: { wards: WardSummary[] }) {
  const total = wards.length
  const reporting = wards.filter((w) => w.status !== "not_started")
  const final = wards.filter((w) => w.status === "final").length
  const called = reporting.filter((w) => w.leader && w.predictedWinner)
  const hits = called.filter((w) => !w.upset).length
  const upsets = called.length - hits

  const tally = new Map<string, number>()
  for (const w of wards) if (w.predictedWinner) tally.set(w.predictedWinner, (tally.get(w.predictedWinner) ?? 0) + 1)
  const [topParty, topCount] = [...tally.entries()].sort((a, b) => b[1] - a[1])[0] ?? ["–", 0]

  return (
    <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
      <Card className="gap-3 py-5">
        <CardHeader className="px-5">
          <CardDescription>Wards reporting</CardDescription>
          <CardTitle className="text-3xl font-semibold">
            {reporting.length}
            <span className="text-base font-normal text-muted-foreground"> / {total}</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="grid gap-1.5 px-5">
          <Meter value={total ? reporting.length / total : 0} />
          <p className="text-xs text-muted-foreground">{final} final · {reporting.length - final} counting</p>
        </CardContent>
      </Card>

      <Card className="gap-3 py-5">
        <CardHeader className="px-5">
          <CardDescription>Prediction holding</CardDescription>
          <CardTitle className="text-3xl font-semibold">
            {called.length ? Math.round((hits / called.length) * 100) : 0}%
          </CardTitle>
        </CardHeader>
        <CardContent className="px-5">
          <p className="text-xs text-muted-foreground">
            Predicted winner leads in {hits} of {called.length} reporting wards
          </p>
        </CardContent>
      </Card>

      <Card className="gap-3 py-5">
        <CardHeader className="px-5">
          <CardDescription>Upsets so far</CardDescription>
          <CardTitle className="text-3xl font-semibold">{upsets}</CardTitle>
        </CardHeader>
        <CardContent className="px-5">
          <p className="text-xs text-muted-foreground">Wards where someone other than the predicted winner leads</p>
        </CardContent>
      </Card>

      <Card className="gap-3 py-5">
        <CardHeader className="px-5">
          <CardDescription>Predicted largest party</CardDescription>
          <CardTitle className="flex items-center gap-2 text-3xl font-semibold">
            <span className="size-3 rounded-[3px]" style={{ background: partyColor(topParty) }} />
            {topParty}
          </CardTitle>
        </CardHeader>
        <CardContent className="px-5">
          <p className="truncate text-xs text-muted-foreground" title={partyName(topParty)}>
            Predicted to win {topCount} of {total} wards
          </p>
        </CardContent>
      </Card>
    </div>
  )
}

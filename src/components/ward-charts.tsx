import { useId } from "react"
import { curveMonotoneX, line as d3line } from "d3-shape"
import {
  CartesianGrid,
  Line,
  LineChart,
  XAxis,
  YAxis,
  usePlotArea,
  useXAxisScale,
  useYAxisScale,
  type TooltipContentProps,
} from "recharts"
import { ChartContainer, ChartTooltip, type ChartConfig } from "@/components/ui/chart"
import { bucketShares } from "@/components/segmented-bar"
import type { WardSummary } from "@/lib/model"
import { OTHER, partyColor, partyName } from "@/parties"
import type { Election, Shares } from "@/types"

const pct = (n: number) => `${n.toFixed(1)}%`
const TZ = "Africa/Johannesburg"
export const formatTime = (t: number | string) =>
  new Date(t).toLocaleTimeString("en-ZA", { hour: "2-digit", minute: "2-digit", timeZone: TZ })

const shareOf = (votes: Record<string, number>): Shares => {
  const total = Object.values(votes).reduce((a, b) => a + b, 0)
  return Object.fromEntries(Object.entries(votes).map(([p, v]) => [p, total ? (v / total) * 100 : 0]))
}

/**
 * Which parties get their own line for this ward: those with their own colour
 * that reach 3% somewhere in the ward's data. Everyone else is folded into
 * "Other", so the chart never has two lines of the same colour.
 */
function seriesFor(sets: Shares[], order: string[]) {
  const peak: Record<string, number> = {}
  for (const s of sets) for (const [p, v] of Object.entries(bucketShares(s))) peak[p] = Math.max(peak[p] ?? 0, v)
  const keep = order.filter((p) => p !== OTHER && (peak[p] ?? 0) >= 3)
  const fold = (s: Shares): Shares => {
    const b = bucketShares(s)
    const out: Shares = {}
    let other = 0
    let hasOther = false
    for (const [p, v] of Object.entries(b)) {
      if (keep.includes(p)) out[p] = v
      else {
        other += v
        hasOther = true
      }
    }
    if (hasOther) out[OTHER] = other
    return out
  }
  const parties = [...keep, ...(sets.some((s) => Object.keys(fold(s)).includes(OTHER)) ? [OTHER] : [])]
  return { parties, fold }
}

const niceMax = (values: number[]) => Math.min(100, Math.max(20, Math.ceil(Math.max(0, ...values) / 10) * 10))

const configFor = (parties: string[]): ChartConfig =>
  Object.fromEntries(parties.map((p) => [p, { label: p, color: partyColor(p) }]))

/** End-of-line / single-point marker: filled, with a ring in the surface colour. */
function Marker({ cx, cy, color, hollow }: { cx?: number; cy?: number; color: string; hollow?: boolean }) {
  if (cx == null || cy == null) return null
  return (
    <circle
      cx={cx}
      cy={cy}
      r={hollow ? 4 : 4.5}
      fill={hollow ? "var(--card)" : color}
      stroke={hollow ? color : "var(--card)"}
      strokeWidth={2}
    />
  )
}

function TooltipCard({ title, subtitle, rows }: { title: string; subtitle?: string; rows: { party: string; value: string }[] }) {
  return (
    <div className="grid min-w-44 gap-1.5 rounded-lg border bg-background px-3 py-2 text-xs shadow-xl">
      <div>
        <div className="font-medium">{title}</div>
        {subtitle && <div className="text-muted-foreground">{subtitle}</div>}
      </div>
      {rows.map((r) => (
        <div key={r.party} className="flex items-center justify-between gap-4">
          <span className="flex items-center gap-1.5 text-muted-foreground" title={partyName(r.party)}>
            <span className="size-2 rounded-[2px]" style={{ background: partyColor(r.party) }} />
            {r.party}
          </span>
          <span className="font-medium text-foreground tabular-nums">{r.value}</span>
        </div>
      ))}
    </div>
  )
}

function SeriesLegend({ parties, extra }: { parties: string[]; extra?: React.ReactNode }) {
  return (
    <div className="flex flex-wrap items-center gap-x-3.5 gap-y-1 text-xs text-muted-foreground">
      {parties.map((p) => (
        <span key={p} className="flex items-center gap-1.5" title={partyName(p)}>
          <span className="h-0.5 w-3 rounded-full" style={{ background: partyColor(p) }} />
          <span className="text-foreground">{p}</span>
        </span>
      ))}
      {extra}
    </div>
  )
}

const axisProps = {
  tickLine: false,
  axisLine: false,
  fontSize: 11,
} as const

/**
 * One smooth (monotone) curve per party through its past results and the
 * forecast, drawn twice and clipped: solid up to the last past election,
 * dashed from there to the forecast. Using one curve keeps the bend at the
 * hand-over smooth, which two separate two-point lines can't do.
 */
function TrendLines({
  parties,
  points,
  categories,
  splitAt,
}: {
  parties: string[]
  /** points[i][j]: party i's share at category j (undefined = no value). */
  points: (number | undefined)[][]
  categories: string[]
  /** Last past election: solid before it, dashed after it. */
  splitAt?: string
}) {
  const id = useId().replace(/:/g, "")
  const xScale = useXAxisScale()
  const yScale = useYAxisScale()
  const plot = usePlotArea()
  if (!xScale || !yScale || !plot || !splitAt) return null
  const split = Number(xScale(splitAt))
  const lastIndex = categories.length - 1
  const pad = 12 // keep line caps and markers inside the clip at the plot edges

  const curve = d3line<[number, number | undefined]>()
    .defined(([, v]) => v != null)
    .x(([j]) => Number(xScale(categories[j])))
    .y(([, v]) => Number(yScale(v as number)))
    .curve(curveMonotoneX)

  return (
    <g>
      <defs>
        <clipPath id={`${id}-past`}>
          <rect x={plot.x - pad} y={plot.y - pad} width={split - plot.x + pad} height={plot.height + pad * 2} />
        </clipPath>
        <clipPath id={`${id}-fc`}>
          <rect x={split} y={plot.y - pad} width={plot.x + plot.width - split + pad} height={plot.height + pad * 2} />
        </clipPath>
      </defs>
      {parties.map((p, i) => {
        const d = curve(points[i].map((v, j) => [j, v] as [number, number | undefined])) ?? ""
        const color = partyColor(p)
        return (
          <g key={p}>
            <path d={d} fill="none" stroke={color} strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" clipPath={`url(#${id}-past)`} />
            <path d={d} fill="none" stroke={color} strokeWidth={2} strokeLinecap="round" strokeDasharray="4 4" clipPath={`url(#${id}-fc)`} />
          </g>
        )
      })}
      {/* Markers last so they sit above every line. */}
      {parties.map((p, i) =>
        points[i].map((v, j) =>
          v == null ? null : (
            <Marker
              key={`${p}-${j}`}
              cx={Number(xScale(categories[j]))}
              cy={Number(yScale(v))}
              color={partyColor(p)}
              hollow={j === lastIndex}
            />
          ),
        ),
      )}
    </g>
  )
}

/** Past elections → 2026 forecast (dashed) → 2026 count so far (dot). */
export function ElectionsChart({
  ward,
  elections,
  order,
  year = "2026",
}: {
  ward: WardSummary
  elections: Election[]
  order: string[]
  year?: string
}) {
  const past = elections.filter((e) => ward.history?.[e.id])
  if (!past.length) {
    return <p className="text-sm text-muted-foreground">No past results for this ward (it may be new since the last boundary changes).</p>
  }

  const counted = ward.status !== "not_started" ? shareOf(ward.votes) : undefined
  const sets = [...past.map((e) => ward.history![e.id]), ...(ward.predictedShares ? [ward.predictedShares] : []), ...(counted ? [counted] : [])]
  const { parties, fold } = seriesFor(sets, order)

  type Row = Record<string, number | string | undefined> & { x: string }
  const rows: Row[] = past.map((e) => {
    const s = fold(ward.history![e.id])
    // A party missing from a past election didn't stand there: leave a gap, not 0%.
    return { x: e.label, ...Object.fromEntries(parties.map((p) => [`${p}:past`, s[p]])) }
  })
  const fc = ward.predictedShares ? fold(ward.predictedShares) : undefined
  const ct = counted ? fold(counted) : undefined
  // The dashed forecast line starts at the last past result.
  const last = rows[rows.length - 1]
  for (const p of parties) last[`${p}:fc`] = last[`${p}:past`]
  rows.push({
    x: year,
    ...Object.fromEntries(parties.flatMap((p) => [[`${p}:fc`, fc?.[p]], [`${p}:count`, ct?.[p]]])),
  })
  const max = niceMax(sets.flatMap((s) => Object.values(fold(s)).filter((v): v is number => v != null)))

  return (
    <div className="grid min-w-0 grid-cols-1 gap-2.5">
      <ChartContainer config={configFor(parties)} className="aspect-auto h-52 w-full">
        <LineChart data={rows} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
          <CartesianGrid vertical={false} />
          <XAxis dataKey="x" {...axisProps} tickMargin={8} padding={{ left: 16, right: 16 }} />
          <YAxis {...axisProps} width={36} domain={[0, max]} tickCount={max / 10 + 1} tickFormatter={(v) => `${v}%`} />
          <ChartTooltip
            cursor={{ strokeWidth: 1 }}
            content={({ active, payload }: TooltipContentProps) => {
              const row = payload?.[0]?.payload as Row | undefined
              if (!active || !row) return null
              const isNow = row.x === year
              return (
                <TooltipCard
                  title={isNow ? `${year} election` : `${row.x} result`}
                  subtitle={isNow ? (ct ? "Forecast · counted so far" : "Forecast (no votes counted yet)") : undefined}
                  rows={parties.map((p) => ({
                    party: p,
                    value: isNow
                      ? [row[`${p}:fc`], row[`${p}:count`]]
                          .map((v) => (typeof v === "number" ? pct(v) : "–"))
                          .slice(0, ct ? 2 : 1)
                          .join(" · ")
                      : typeof row[`${p}:past`] === "number"
                        ? pct(row[`${p}:past`] as number)
                        : "didn't stand",
                  }))}
                />
              )
            }}
          />
          {/* Invisible series: they only feed the tooltip. TrendLines draws the visible lines. */}
          {parties.map((p) => (
            <Line key={`${p}:past`} dataKey={`${p}:past`} stroke="none" dot={false} activeDot={false} isAnimationActive={false} />
          ))}
          <TrendLines
            parties={parties}
            points={parties.map((p) => rows.map((r) => (typeof r[`${p}:past`] === "number" ? (r[`${p}:past`] as number) : r.x === year ? (r[`${p}:fc`] as number | undefined) : undefined)))}
            categories={rows.map((r) => r.x)}
            splitAt={rows[rows.length - 2]?.x}
          />
          {ct &&
            parties.map((p) => (
              <Line
                key={`${p}:count`}
                dataKey={`${p}:count`}
                stroke="none"
                dot={(props) => <Marker key={`${p}-count-${props.index}`} cx={props.cx} cy={props.cy} color={partyColor(p)} />}
                activeDot={false}
                isAnimationActive={false}
              />
            ))}
        </LineChart>
      </ChartContainer>
      <SeriesLegend
        parties={parties}
        extra={
          <span className="flex items-center gap-3 border-l pl-3">
            <span className="flex items-center gap-1.5">
              <svg width="16" height="8" aria-hidden>
                <line x1="0" y1="4" x2="16" y2="4" stroke="currentColor" strokeWidth="2" strokeDasharray="4 3" />
              </svg>
              Forecast
            </span>
            {ct && (
              <span className="flex items-center gap-1.5">
                <span className="size-2 rounded-full bg-current" /> Counted
              </span>
            )}
          </span>
        }
      />
    </div>
  )
}

/** Running vote share as votes were counted on election night. */
export function CountChart({ ward, order }: { ward: WardSummary; order: string[] }) {
  if (ward.status === "not_started") {
    return <p className="text-sm text-muted-foreground">No votes counted yet.</p>
  }
  if (ward.timeline.length < 2) {
    return (
      <p className="text-sm text-muted-foreground">
        {ward.timeline.length ? "Only one results update so far." : "No count timeline in the results feed for this ward."}
      </p>
    )
  }

  const updates = ward.timeline.map((u) => ({ t: Date.parse(u.at), shares: shareOf(u.votes), total: Object.values(u.votes).reduce((a, b) => a + b, 0) }))
  const { parties, fold } = seriesFor(
    [...updates.map((u) => u.shares), ...(ward.predictedShares ? [ward.predictedShares] : [])],
    order,
  )
  const forecast = ward.predictedShares ? fold(ward.predictedShares) : {}
  const rows = updates.map((u) => ({ t: u.t, total: u.total, ...fold(u.shares) }))
  const max = niceMax(rows.flatMap((r) => parties.map((p) => Number(r[p as keyof typeof r] ?? 0))))
  const lastIndex = rows.length - 1

  return (
    <div className="grid min-w-0 grid-cols-1 gap-2.5">
      <ChartContainer config={configFor(parties)} className="aspect-auto h-52 w-full">
        <LineChart data={rows} margin={{ top: 8, right: 14, left: 0, bottom: 0 }}>
          <CartesianGrid vertical={false} />
          <XAxis
            dataKey="t"
            type="number"
            scale="time"
            domain={["dataMin", "dataMax"]}
            {...axisProps}
            tickMargin={8}
            tickCount={5}
            tickFormatter={formatTime}
          />
          <YAxis {...axisProps} width={36} domain={[0, max]} tickCount={max / 10 + 1} tickFormatter={(v) => `${v}%`} />
          <ChartTooltip
            cursor={{ strokeWidth: 1 }}
            content={({ active, payload }: TooltipContentProps) => {
              const row = payload?.[0]?.payload as (typeof rows)[number] | undefined
              if (!active || !row) return null
              return (
                <TooltipCard
                  title={`${formatTime(row.t)} SAST`}
                  subtitle={`${row.total.toLocaleString("en-ZA")} votes counted`}
                  rows={[...parties]
                    .sort((a, b) => Number(row[b as keyof typeof row] ?? 0) - Number(row[a as keyof typeof row] ?? 0))
                    .map((p) => ({
                      party: p,
                      value:
                        pct(Number(row[p as keyof typeof row] ?? 0)) +
                        (forecast[p] != null ? `  (fc ${forecast[p].toFixed(0)}%)` : ""),
                    }))}
                />
              )
            }}
          />
          {parties.map((p) => (
            <Line
              key={p}
              dataKey={p}
              type="monotone"
              stroke={partyColor(p)}
              strokeWidth={2}
              strokeLinecap="round"
              strokeLinejoin="round"
              dot={(props) =>
                props.index === lastIndex ? (
                  <Marker key={`${p}-end`} cx={props.cx} cy={props.cy} color={partyColor(p)} />
                ) : (
                  <g key={`${p}-${props.index}`} />
                )
              }
              activeDot={{ r: 4, stroke: "var(--card)", strokeWidth: 2 }}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ChartContainer>
      <SeriesLegend
        parties={parties}
        extra={
          <span className="border-l pl-3">
            {rows.length} updates · times in SAST
          </span>
        }
      />
    </div>
  )
}

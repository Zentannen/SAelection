import { Bar, BarChart, CartesianGrid, Cell, LabelList, XAxis, YAxis } from "recharts"
import { ChartContainer, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart"
import { MAIN_PARTIES, partyColor, partyName } from "@/parties"
import type { WardSummary } from "@/lib/model"

const config = {
  predicted: { label: "Predicted to win" },
  leading: { label: "Leading now" },
} satisfies ChartConfig

/** Fold parties without their own colour into "Other", matching the map. */
const bucket = (p?: string) => (p ? (MAIN_PARTIES.includes(p) ? p : "Other") : undefined)

export function PartyChart({ wards }: { wards: WardSummary[] }) {
  const counts = new Map<string, { predicted: number; leading: number }>()
  for (const key of [...MAIN_PARTIES, "Other"]) counts.set(key, { predicted: 0, leading: 0 })
  for (const w of wards) {
    const p = bucket(w.predictedWinner)
    const l = bucket(w.leader)
    if (p) counts.get(p)!.predicted++
    if (l) counts.get(l)!.leading++
  }
  const data = [...counts.entries()]
    .map(([party, c]) => ({ party, ...c }))
    .filter((d) => d.predicted + d.leading > 0)

  return (
    <div className="grid gap-3">
      <ChartContainer config={config} className="aspect-auto w-full" style={{ height: data.length * 52 + 24 }}>
        <BarChart data={data} layout="vertical" margin={{ left: 4, right: 36, top: 0, bottom: 0 }} barGap={3} barCategoryGap={14}>
          <CartesianGrid horizontal={false} />
          <XAxis type="number" hide allowDecimals={false} />
          <YAxis dataKey="party" type="category" tickLine={false} axisLine={false} width={64} tickMargin={6} />
          <ChartTooltip
            cursor={false}
            content={
              <ChartTooltipContent
                labelFormatter={(_, payload) => {
                  const p = String(payload?.[0]?.payload?.party ?? "")
                  return p === "Other" ? "Other parties" : partyName(p)
                }}
                formatter={(value, name, item) => (
                  <div className="flex w-full items-center justify-between gap-4">
                    <span className="flex items-center gap-1.5 text-muted-foreground">
                      <span
                        className="size-2.5 rounded-[2px]"
                        style={{
                          background: partyColor(item.payload.party),
                          opacity: name === "predicted" ? 0.35 : 1,
                        }}
                      />
                      {config[name as keyof typeof config]?.label}
                    </span>
                    <span className="font-medium text-foreground tabular-nums">{String(value)} wards</span>
                  </div>
                )}
              />
            }
          />
          <Bar dataKey="predicted" radius={[0, 4, 4, 0]} barSize={12} isAnimationActive>
            {data.map((d) => (
              <Cell key={d.party} fill={partyColor(d.party)} fillOpacity={0.35} />
            ))}
            <LabelList dataKey="predicted" position="right" offset={6} className="fill-muted-foreground" fontSize={11} />
          </Bar>
          <Bar dataKey="leading" radius={[0, 4, 4, 0]} barSize={12} isAnimationActive>
            {data.map((d) => (
              <Cell key={d.party} fill={partyColor(d.party)} />
            ))}
            <LabelList dataKey="leading" position="right" offset={6} className="fill-foreground" fontSize={11} fontWeight={500} />
          </Bar>
        </BarChart>
      </ChartContainer>
      <div className="flex items-center gap-4 text-xs text-muted-foreground">
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-4 rounded-[3px] bg-muted-foreground/35" /> Predicted to win
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-4 rounded-[3px] bg-muted-foreground" /> Leading now
        </span>
      </div>
    </div>
  )
}

import { memo } from "react"
import { ArrowRight } from "lucide-react"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { partyColor } from "@/parties"
import type { WardSummary } from "@/lib/model"

function Party({ code }: { code?: string }) {
  if (!code) return <span className="text-muted-foreground">–</span>
  return (
    <span className="inline-flex items-center gap-1.5 font-medium">
      <span className="size-2 rounded-[2px]" style={{ background: partyColor(code) }} />
      {code}
    </span>
  )
}

export const UpsetsTable = memo(function UpsetsTable({
  wards,
  onHoverWard,
  onSelectWard,
}: {
  wards: WardSummary[]
  onHoverWard: (id: string | null) => void
  onSelectWard: (id: string) => void
}) {
  const upsets = wards.filter((w) => w.upset).sort((a, b) => (b.leadMargin ?? 0) - (a.leadMargin ?? 0))

  if (!upsets.length) {
    return <p className="py-6 text-center text-sm text-muted-foreground">No upsets yet. Every reporting ward is going as predicted.</p>
  }

  return (
    <Table className="[&_td]:px-1.5 [&_th]:px-1.5">
      <TableHeader>
        <TableRow>
          <TableHead>Ward</TableHead>
          <TableHead className="text-right">Predicted → leading</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {upsets.map((w) => (
          <TableRow
            key={w.id}
            onMouseEnter={() => onHoverWard(w.id)}
            onMouseLeave={() => onHoverWard(null)}
            onClick={() => onSelectWard(w.id)}
            className="cursor-pointer"
          >
            <TableCell>
              <div className="font-medium">{w.feature.properties.name ?? w.id}</div>
              <div className="text-xs text-muted-foreground">
                {[w.district, w.status === "final" ? "Final" : "Counting"].filter(Boolean).join(" · ")}
              </div>
            </TableCell>
            <TableCell className="text-right">
              <span className="inline-flex items-center gap-2">
                <Party code={w.predictedWinner} />
                <ArrowRight className="size-3.5 text-muted-foreground" />
                <Party code={w.leader} />
              </span>
              <div className="text-xs text-muted-foreground tabular-nums">by {w.leadMargin?.toFixed(1)} pts</div>
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  )
})

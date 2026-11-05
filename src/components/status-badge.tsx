import { Badge } from "@/components/ui/badge"
import type { ResultStatus } from "@/types"

const LABEL: Record<ResultStatus, string> = {
  final: "Final",
  counting: "Counting",
  not_started: "No results",
}

export function StatusBadge({ status }: { status: ResultStatus }) {
  return (
    <Badge variant={status === "final" ? "default" : status === "counting" ? "secondary" : "outline"} className="gap-1.5">
      {status === "counting" && (
        <span className="relative flex size-1.5">
          <span className="absolute inline-flex size-full animate-ping rounded-full bg-current opacity-60" />
          <span className="relative inline-flex size-1.5 rounded-full bg-current" />
        </span>
      )}
      {LABEL[status]}
    </Badge>
  )
}

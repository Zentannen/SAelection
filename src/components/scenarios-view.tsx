import { useRef, useState } from "react"
import { ArrowDownRight } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { RoleKey, VoteBar } from "@/components/vote-bar"
import { cn } from "@/lib/utils"
import { partyColor } from "@/parties"
import {
  ARCHETYPES,
  CASES,
  CONFIGURATIONS,
  COUNCILS,
  DEEP_DIVE,
  DISTANCES,
  GAUTENG_MAP,
  LEVERS,
  LOCAL_RULES,
  NATIONAL_PRESSURE,
  PROVINCE,
  REGULARITIES,
  WATCH,
  type Configuration,
} from "@/scenarios"

const configById = new Map(CONFIGURATIONS.map((c) => [c.id, c]))

function PartyChip({ code }: { code: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-xs font-medium">
      <span className="size-2 rounded-[2px]" style={{ background: partyColor(code) }} />
      {code}
    </span>
  )
}

function Callout({ children, className }: { children: React.ReactNode; className?: string }) {
  return <div className={cn("rounded-lg border bg-muted/50 px-4 py-3 text-sm", className)}>{children}</div>
}

function Num({ n }: { n: number }) {
  return (
    <span className="grid size-6 shrink-0 place-items-center rounded-full border text-xs font-medium tabular-nums">
      {n}
    </span>
  )
}

// ---------------------------------------------------------------------------

function Intro() {
  return (
    <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_auto]">
      <Callout className="grid content-center gap-1.5 bg-card">
        <p className="font-medium">How to use this page</p>
        <p className="text-muted-foreground">
          Take a metro's official vote shares, find the matching row in the Gauteng table, then open its configuration to
          see what happened in other countries with similar percentages and party values. Nothing here comes from polls, news
          coverage or party statements; the evidence is foreign coalition outcomes, mostly from Central and Eastern Europe.
        </p>
      </Callout>
      <div className="grid grid-cols-3 gap-3">
        {COUNCILS.map((c) => (
          <Card key={c.name} className="gap-1 px-4 py-4">
            <CardDescription className="text-xs">{c.name}</CardDescription>
            <p className="text-2xl font-semibold tabular-nums">{c.majority}</p>
            <p className="text-xs text-muted-foreground tabular-nums">of {c.seats} seats to govern</p>
          </Card>
        ))}
      </div>
    </div>
  )
}

function ConfigLink({ config, variant, onOpen }: { config: Configuration; variant?: string; onOpen: (id: string) => void }) {
  return (
    <button
      type="button"
      onClick={() => onOpen(config.id)}
      className="inline-flex items-start gap-1.5 text-left underline-offset-4 hover:underline focus-visible:underline focus-visible:outline-none"
    >
      <Num n={config.n} />
      <span className="pt-0.5">
        {config.short}
        {variant && <span className="block text-xs text-muted-foreground">{variant}</span>}
      </span>
    </button>
  )
}

function GautengCard({ onOpen }: { onOpen: (id: string) => void }) {
  return (
    <Card className="gap-4">
      <CardHeader>
        <CardTitle>Gauteng at a glance</CardTitle>
        <CardDescription>
          Vote-share bands in any of the three metros, the configuration they match, and what foreign cases say follows
        </CardDescription>
      </CardHeader>
      <CardContent className="grid gap-4">
        <div className="hidden md:block">
          <Table className="[&_td]:align-top [&_td]:whitespace-normal">
            <TableHeader>
              <TableRow>
                <TableHead className="min-w-48">Result in the metro</TableHead>
                <TableHead className="min-w-40">Configuration</TableHead>
                <TableHead className="min-w-56">Expected government</TableHead>
                <TableHead className="min-w-32">Lifespan</TableHead>
                <TableHead className="min-w-40">Closest cases</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {GAUTENG_MAP.map((r) => {
                const c = configById.get(r.configId)!
                return (
                  <TableRow key={r.band}>
                    <TableCell className="font-medium">{r.band}</TableCell>
                    <TableCell>
                      <ConfigLink config={c} variant={r.variant} onOpen={onOpen} />
                    </TableCell>
                    <TableCell>{r.government}</TableCell>
                    <TableCell className="text-muted-foreground">{r.lifespan}</TableCell>
                    <TableCell className="text-muted-foreground">{r.cases}</TableCell>
                  </TableRow>
                )
              })}
            </TableBody>
          </Table>
        </div>
        <div className="grid gap-3 md:hidden">
          {GAUTENG_MAP.map((r) => {
            const c = configById.get(r.configId)!
            return (
              <div key={r.band} className="grid gap-2 rounded-lg border p-3.5 text-sm">
                <p className="font-medium">{r.band}</p>
                <ConfigLink config={c} variant={r.variant} onOpen={onOpen} />
                <p>{r.government}</p>
                <p className="text-muted-foreground">
                  <span className="font-medium text-foreground">Lifespan: </span>
                  {r.lifespan}
                </p>
                <p className="text-xs text-muted-foreground">{r.cases}</p>
              </div>
            )
          })}
        </div>
        <div className="grid gap-3 sm:grid-cols-2">
          {LOCAL_RULES.map((r) => (
            <Callout key={r.rule}>
              <p className="font-medium">{r.rule}</p>
              <p className="mt-1 text-muted-foreground">{r.effect}</p>
            </Callout>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}

function DeepDiveCard() {
  return (
    <Card className="gap-4">
      <CardHeader>
        <CardTitle>DA + ActionSA above 50%: will they form a coalition?</CardTitle>
        <CardDescription>The liberal party and the anti-corruption challenger, judged on four factors from foreign cases</CardDescription>
      </CardHeader>
      <CardContent className="grid gap-5 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
        <div className="min-w-0">
          <div className="hidden md:block">
            <Table className="[&_td]:align-top [&_td]:whitespace-normal">
              <TableHeader>
                <TableRow>
                  <TableHead>Factor</TableHead>
                  <TableHead className="min-w-36">Likely and durable</TableHead>
                  <TableHead className="min-w-36">Fragile or fails</TableHead>
                  <TableHead className="min-w-36">Cases</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {DEEP_DIVE.factors.map((f) => (
                  <TableRow key={f.factor}>
                    <TableCell className="font-medium">{f.factor}</TableCell>
                    <TableCell>{f.durable}</TableCell>
                    <TableCell>{f.fragile}</TableCell>
                    <TableCell className="text-muted-foreground">{f.cases}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
          <div className="grid gap-3 md:hidden">
            {DEEP_DIVE.factors.map((f) => (
              <div key={f.factor} className="grid gap-1.5 rounded-lg border p-3.5 text-sm">
                <p className="font-medium">{f.factor}</p>
                <p><span className="text-muted-foreground">Likely and durable: </span>{f.durable}</p>
                <p><span className="text-muted-foreground">Fragile or fails: </span>{f.fragile}</p>
                <p className="text-xs text-muted-foreground">{f.cases}</p>
              </div>
            ))}
          </div>
        </div>
        <div className="grid content-start gap-3">
          {DEEP_DIVE.scenarios.map((sc) => (
            <div key={sc.when} className="grid gap-0.5 border-l-2 pl-3">
              <p className="text-sm font-medium">{sc.when}</p>
              <p className="text-sm text-muted-foreground">{sc.then}</p>
            </div>
          ))}
          <Callout className="mt-1 text-muted-foreground">{DEEP_DIVE.gamson}</Callout>
        </div>
      </CardContent>
    </Card>
  )
}

function ConfigDetail({ config }: { config: Configuration }) {
  return (
    <div className="grid min-w-0 content-start gap-4">
      <div className="grid gap-2">
        <h3 className="flex items-start gap-2 text-base font-semibold">
          <Num n={config.n} />
          <span className="pt-0.5">{config.title}</span>
        </h3>
        <Callout>
          <span className="font-medium">Lesson: </span>
          {config.lesson}
        </Callout>
      </div>
      <RoleKey />
      <div className="grid gap-3">
        {config.rows.map((r) => {
          const c = r.caseId ? CASES[r.caseId] : undefined
          return (
            <div key={r.label} className="grid gap-2.5 rounded-lg border p-3.5">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <p className="font-medium">{r.label}</p>
                <Badge variant="outline" className="font-normal">{r.lifespan}</Badge>
              </div>
              {c && <VoteBar shares={c.shares} approx={c.approx} />}
              <p className="text-sm">{r.outcome}</p>
              <p className="text-sm text-muted-foreground">
                <span className="font-medium text-foreground">Why: </span>
                {r.why}
              </p>
            </div>
          )
        })}
      </div>
    </div>
  )
}

function Explorer({ configId, onSelect }: { configId: string; onSelect: (id: string) => void }) {
  const config = configById.get(configId) ?? CONFIGURATIONS[0]
  return (
    <Card className="gap-4">
      <CardHeader>
        <CardTitle>Nine configurations</CardTitle>
        <CardDescription>Vote-share situations, with what happened in other countries, how long it lasted, and why</CardDescription>
      </CardHeader>
      <CardContent className="grid gap-5 lg:grid-cols-[260px_minmax(0,1fr)]">
        <nav aria-label="Configurations" className="-mx-1 flex gap-1 overflow-x-auto px-1 pb-1 lg:mx-0 lg:grid lg:content-start lg:overflow-visible lg:px-0">
          {CONFIGURATIONS.map((c) => {
            const active = c.id === config.id
            return (
              <button
                key={c.id}
                type="button"
                aria-current={active ? "true" : undefined}
                onClick={() => onSelect(c.id)}
                className={cn(
                  "flex shrink-0 items-center gap-2 rounded-md px-2.5 py-2 text-left text-sm transition-colors outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50",
                  active ? "bg-accent font-medium text-accent-foreground" : "text-muted-foreground hover:bg-accent/60 hover:text-foreground",
                )}
              >
                <Num n={c.n} />
                <span className="whitespace-nowrap lg:whitespace-normal">{c.short}</span>
              </button>
            )
          })}
        </nav>
        <ConfigDetail config={config} />
      </CardContent>
    </Card>
  )
}

function ArchetypesCard() {
  return (
    <Card className="gap-4">
      <CardHeader>
        <CardTitle>Party archetypes</CardTitle>
        <CardDescription>Parties grouped by values, with foreign examples and their Gauteng counterparts</CardDescription>
      </CardHeader>
      <CardContent className="grid gap-4">
        {ARCHETYPES.map((a) => (
          <div key={a.name} className="grid gap-1.5 border-b pb-4 last:border-b-0 last:pb-0">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <p className="font-medium">{a.name}</p>
              <span className="flex flex-wrap gap-1">
                {a.gauteng.map((p) => <PartyChip key={p} code={p} />)}
              </span>
            </div>
            <p className="text-sm">{a.values}</p>
            <p className="text-xs text-muted-foreground">{a.examples}</p>
          </div>
        ))}
      </CardContent>
    </Card>
  )
}

function DistancesCard() {
  return (
    <Card className="gap-4">
      <CardHeader>
        <CardTitle>Values distances</CardTitle>
        <CardDescription>How far apart archetypes are, and what usually happens between them</CardDescription>
      </CardHeader>
      <CardContent className="grid gap-4">
        {DISTANCES.map((d) => (
          <div key={d.pair} className="grid gap-1.5 border-b pb-4 last:border-b-0 last:pb-0">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <p className="font-medium">{d.pair}</p>
              <Badge variant={d.distance === "Adjacent" ? "secondary" : "outline"}>{d.distance}</Badge>
            </div>
            <p className="text-sm text-muted-foreground">{d.note}</p>
          </div>
        ))}
      </CardContent>
    </Card>
  )
}

function RegularitiesCard() {
  return (
    <Card className="gap-4">
      <CardHeader>
        <CardTitle>Twelve regularities</CardTitle>
        <CardDescription>Patterns that hold across countries, with the cases behind them and the exceptions</CardDescription>
      </CardHeader>
      <CardContent>
        <ol className="grid gap-x-8 gap-y-4 md:grid-cols-2">
          {REGULARITIES.map((r, i) => (
            <li key={r.rule} className="flex gap-3">
              <Num n={i + 1} />
              <div className="grid gap-1">
                <p className="text-sm font-medium">{r.rule}</p>
                <p className="text-sm text-muted-foreground">{r.evidence}</p>
              </div>
            </li>
          ))}
        </ol>
      </CardContent>
    </Card>
  )
}

function PressureCard() {
  return (
    <Card className="gap-4">
      <CardHeader>
        <CardTitle>Higher-level pressure</CardTitle>
        <CardDescription>How national authorities have shaped coalition outcomes elsewhere</CardDescription>
      </CardHeader>
      <CardContent className="grid gap-4">
        {LEVERS.map((l) => (
          <div key={l.lever} className="grid gap-1 border-b pb-4 last:border-b-0 last:pb-0">
            <p className="font-medium">{l.lever}</p>
            <p className="text-xs text-muted-foreground">{l.examples}</p>
            <p className="text-sm">
              {l.achieved}. <span className="text-muted-foreground">Limit: {l.limits.toLowerCase()}.</span>
            </p>
          </div>
        ))}
        <Callout className="grid gap-2">
          <p className="font-medium">In Gauteng</p>
          <p className="text-muted-foreground">{NATIONAL_PRESSURE.summary} They can:</p>
          <ul className="grid list-disc gap-1 pl-5 text-muted-foreground">
            {NATIONAL_PRESSURE.levers.map((l) => <li key={l}>{l}</li>)}
          </ul>
          <p>{NATIONAL_PRESSURE.verdict}</p>
        </Callout>
      </CardContent>
    </Card>
  )
}

function ProvinceCard({ onOpen }: { onOpen: (id: string) => void }) {
  return (
    <Card className="gap-4">
      <CardHeader>
        <CardTitle>The provincial government</CardTitle>
        <CardDescription>Not on the ballot until 2029, but exposed to whatever forms in the metros</CardDescription>
        <CardAction>
          <button type="button" onClick={() => onOpen("c9")} className="inline-flex items-center gap-1 text-sm underline-offset-4 hover:underline">
            Cohabitation cases <ArrowDownRight className="size-3.5" />
          </button>
        </CardAction>
      </CardHeader>
      <CardContent className="grid gap-4 text-sm">
        <p>{PROVINCE.summary}</p>
        <div className="grid gap-1.5">
          <p className="font-medium">Foreign precedent says to expect</p>
          <ul className="grid list-disc gap-1 pl-5 text-muted-foreground">
            {PROVINCE.expect.map((e) => <li key={e}>{e}</li>)}
          </ul>
        </div>
        <Callout className="text-muted-foreground">{PROVINCE.grandCoalition}</Callout>
        <div className="grid gap-1.5">
          <p className="font-medium">What to watch</p>
          <ol className="grid gap-2">
            {WATCH.map((w, i) => (
              <li key={w} className="flex items-start gap-2.5">
                <Num n={i + 1} />
                <span className="pt-0.5 text-muted-foreground">{w}</span>
              </li>
            ))}
          </ol>
        </div>
      </CardContent>
    </Card>
  )
}

type Region = "all" | "cee" | "other"

function CasesCard() {
  const [region, setRegion] = useState<Region>("all")
  const rows = Object.entries(CASES).filter(([, c]) => region === "all" || c.region === region)
  return (
    <Card className="gap-4">
      <CardHeader className="max-sm:has-data-[slot=card-action]:grid-cols-1">
        <CardTitle>All cases</CardTitle>
        <CardDescription>Vote shares and the government that formed</CardDescription>
        <CardAction className="max-sm:col-start-1 max-sm:row-start-3 max-sm:mt-2 max-sm:justify-self-start">
          <ToggleGroup type="single" variant="outline" size="sm" value={region} onValueChange={(v) => v && setRegion(v as Region)}>
            <ToggleGroupItem value="all" className="px-3">All</ToggleGroupItem>
            <ToggleGroupItem value="cee" className="px-3">Central &amp; Eastern Europe</ToggleGroupItem>
            <ToggleGroupItem value="other" className="px-3">Elsewhere</ToggleGroupItem>
          </ToggleGroup>
        </CardAction>
      </CardHeader>
      <CardContent className="grid gap-3">
        <RoleKey />
        <div className="hidden md:block">
          <Table className="[&_td]:align-top [&_td]:whitespace-normal">
            <TableHeader>
              <TableRow>
                <TableHead className="min-w-32">Case</TableHead>
                <TableHead className="min-w-72">Vote shares (%)</TableHead>
                <TableHead className="min-w-48">Government</TableHead>
                <TableHead className="min-w-32">Lifespan</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map(([id, c]) => (
                <TableRow key={id}>
                  <TableCell>
                    <div className="font-medium">{c.country}</div>
                    <div className="text-xs text-muted-foreground">{c.year}</div>
                  </TableCell>
                  <TableCell>
                    <VoteBar shares={c.shares} approx={c.approx} compact />
                  </TableCell>
                  <TableCell>{c.government}</TableCell>
                  <TableCell className="text-muted-foreground">{c.lifespan}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
        <div className="grid gap-3 md:hidden">
          {rows.map(([id, c]) => (
            <div key={id} className="grid gap-2 rounded-lg border p-3.5 text-sm">
              <div className="flex flex-wrap items-baseline justify-between gap-x-2">
                <p className="font-medium">{c.country} <span className="font-normal text-muted-foreground">{c.year}</span></p>
                <p className="text-xs text-muted-foreground">{c.lifespan}</p>
              </div>
              <VoteBar shares={c.shares} approx={c.approx} compact />
              <p>{c.government}</p>
            </div>
          ))}
        </div>
        <p className="text-xs text-muted-foreground">
          Bars show vote shares, not seats. Where thresholds discard small parties, a seat majority can come from under 50% of
          the vote (Slovakia 2012, Slovakia 2023).
        </p>
      </CardContent>
    </Card>
  )
}

// ---------------------------------------------------------------------------

export function ScenariosView() {
  const [configId, setConfigId] = useState("c2")
  const explorerRef = useRef<HTMLDivElement>(null)

  const open = (id: string) => {
    setConfigId(id)
    explorerRef.current?.scrollIntoView({ behavior: "smooth", block: "start" })
  }

  return (
    <main className="mx-auto grid max-w-[1400px] gap-4 px-4 py-6 sm:px-6">
      <Intro />
      <GautengCard onOpen={open} />
      <DeepDiveCard />
      <div ref={explorerRef} className="scroll-mt-4">
        <Explorer configId={configId} onSelect={setConfigId} />
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        <ArchetypesCard />
        <DistancesCard />
      </div>
      <RegularitiesCard />
      <div className="grid gap-4 lg:grid-cols-2">
        <PressureCard />
        <ProvinceCard onOpen={open} />
      </div>
      <CasesCard />
    </main>
  )
}

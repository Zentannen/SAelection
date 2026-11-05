import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react"
import { geoMercator, geoPath } from "d3-geo"
import { select } from "d3-selection"
import "d3-transition"
import { zoom, zoomIdentity, zoomTransform, type ZoomBehavior, type ZoomTransform } from "d3-zoom"
import { ArrowLeft, Maximize2, Minus, Plus } from "lucide-react"
import { Button } from "@/components/ui/button"
import { buildDistrictGeometry } from "@/lib/geo"
import type { WardCollection } from "@/types"

export interface WardFill {
  color: string
  opacity: number
}

export interface HoverEvent {
  id: string
  x: number
  y: number
}

interface Props {
  wards: WardCollection
  fillFor: (id: string) => WardFill
  highlightedId?: string | null
  /** Ward whose detail panel is open; stays outlined. */
  selectedId?: string | null
  /** Called when the pointer (or keyboard focus) moves onto a different ward, or off the wards. */
  onHover: (e: HoverEvent | null) => void
  /** Click / Enter on a ward. */
  onSelectWard?: (id: string) => void
  /** District the map is zoomed to; other districts are dimmed. */
  focusDistrict?: string | null
  /** District to outline (e.g. while its row is hovered in the list). */
  highlightDistrict?: string | null
  onSelectDistrict?: (district: string | null) => void
}

interface WardPathInfo {
  id: string
  label: string
  district: string | null
  d: string
}

const PAD = 12
const MAX_ZOOM = 16

/**
 * The ward fills. Memoised so hovering (which only moves an outline) doesn't
 * re-render thousands of paths; it re-renders only when shapes, colours or
 * the focused district change.
 */
const WardPaths = memo(function WardPaths({
  paths,
  fillFor,
  focusDistrict,
  onEnter,
  onSelect,
  onLeaveFocus,
}: {
  paths: WardPathInfo[]
  fillFor: (id: string) => WardFill
  focusDistrict: string | null
  onEnter: (id: string, x: number, y: number) => void
  onSelect: (id: string) => void
  onLeaveFocus: () => void
}) {
  return (
    <>
      {paths.map(({ id, label, d, district }) => {
        const { color, opacity } = fillFor(id)
        const dimmed = !!focusDistrict && district !== focusDistrict
        return (
          <path
            key={id}
            d={d}
            data-ward={id}
            className={dimmed ? "ward dimmed" : "ward"}
            style={{ fill: color, fillOpacity: opacity }}
            tabIndex={0}
            aria-label={district ? `${label}, ${district}` : label}
            onPointerEnter={(e) => onEnter(id, e.clientX, e.clientY)}
            onClick={() => onSelect(id)}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault()
                onSelect(id)
              }
            }}
            onFocus={(e) => {
              const r = e.currentTarget.getBoundingClientRect()
              onEnter(id, r.left + r.width / 2, r.top + r.height / 2)
            }}
            onBlur={onLeaveFocus}
          />
        )
      })}
    </>
  )
})

export function WardMap({
  wards,
  fillFor,
  highlightedId,
  selectedId,
  onHover,
  onSelectWard,
  focusDistrict = null,
  highlightDistrict = null,
  onSelectDistrict,
}: Props) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const svgRef = useRef<SVGSVGElement>(null)
  const layerRef = useRef<SVGGElement>(null)
  const labelsRef = useRef<SVGGElement>(null)
  const badgeRef = useRef<HTMLDivElement>(null)
  const zoomRef = useRef<ZoomBehavior<SVGSVGElement, unknown> | null>(null)
  const [size, setSize] = useState({ width: 0, height: 0 })
  // Only used to enable/disable the zoom buttons; updated when a zoom gesture ends.
  const [scale, setScale] = useState(1)

  // Latest callbacks in refs, so the memoised ward layer gets stable handlers.
  const onHoverRef = useRef(onHover)
  const onSelectWardRef = useRef(onSelectWard)
  useEffect(() => {
    onHoverRef.current = onHover
    onSelectWardRef.current = onSelectWard
  })
  const hoveringRef = useRef<string | null>(null)
  const setHovered = useCallback((e: HoverEvent | null) => {
    if ((e?.id ?? null) === hoveringRef.current) return
    hoveringRef.current = e?.id ?? null
    onHoverRef.current(e)
  }, [])
  const handleEnter = useCallback((id: string, x: number, y: number) => setHovered({ id, x, y }), [setHovered])
  const handleSelect = useCallback((id: string) => onSelectWardRef.current?.(id), [])
  const handleLeave = useCallback(() => setHovered(null), [setHovered])

  useEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const ro = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect
      setSize({ width, height })
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  // Pan & zoom: wheel / pinch / drag. The transform is applied straight to the
  // <g> (no React re-render per frame), so it stays smooth with thousands of
  // wards. --k (set on the label layer only) keeps district names a constant size.
  const applyTransform = useCallback((t: ZoomTransform) => {
    layerRef.current?.setAttribute("transform", t.toString())
    labelsRef.current?.style.setProperty("--k", String(t.k))
    const badge = badgeRef.current
    if (badge) {
      badge.textContent = `${t.k.toFixed(1)}×`
      badge.style.display = t.k > 1.05 ? "" : "none"
    }
  }, [])

  useEffect(() => {
    const svg = svgRef.current
    if (!svg || !size.width) return
    const behaviour = zoom<SVGSVGElement, unknown>()
      .scaleExtent([1, MAX_ZOOM])
      .translateExtent([
        [0, 0],
        [size.width, size.height],
      ])
      .extent([
        [0, 0],
        [size.width, size.height],
      ])
      // A few pixels of wobble during a click still counts as a click, not a pan.
      .clickDistance(4)
      // d3 speeds up Ctrl-wheel tenfold for trackpad pinches (small deltas); a
      // Ctrl + mouse-wheel notch (large delta) would otherwise jump 4× at once.
      .wheelDelta((e: WheelEvent) => {
        const unit = e.deltaMode === 1 ? 0.05 : e.deltaMode ? 1 : 0.002
        const pinch = e.ctrlKey && Math.abs(e.deltaY) < 50
        return -e.deltaY * unit * (pinch ? 10 : 1)
      })
      .filter((event: Event & { ctrlKey?: boolean; metaKey?: boolean; button?: number; type: string }) => {
        if (event.button) return false
        if (event.type === "wheel") {
          // When the page itself scrolls (narrower windows), plain scrolling scrolls
          // the page; Ctrl/⌘ + scroll or a pinch zooms the map.
          const pageScrolls = document.documentElement.scrollHeight > window.innerHeight + 1
          return !pageScrolls || !!event.ctrlKey || !!event.metaKey
        }
        return !event.ctrlKey
      })
      .on("start", () => svg.classList.add("cursor-grabbing"))
      .on("end", (event) => {
        svg.classList.remove("cursor-grabbing")
        setScale(event.transform.k)
      })
      .on("zoom", (event) => applyTransform(event.transform))
    zoomRef.current = behaviour
    select(svg).call(behaviour).on("dblclick.zoom", null)
    return () => {
      select(svg).on(".zoom", null)
    }
  }, [size.width, size.height, applyTransform])

  /** Clamp a transform to the pan limits, so a programmatic zoom never needs to jump later. */
  const constrain = useCallback(
    (t: ZoomTransform) => {
      const z = zoomRef.current
      if (!z) return t
      return z.constrain()(
        t,
        [
          [0, 0],
          [size.width, size.height],
        ],
        z.translateExtent(),
      )
    },
    [size.width, size.height],
  )

  const zoomBy = useCallback((factor: number) => {
    const svg = svgRef.current
    if (svg && zoomRef.current) select(svg).transition().duration(250).call(zoomRef.current.scaleBy, factor)
  }, [])
  const resetZoom = useCallback(() => {
    const svg = svgRef.current
    if (svg && zoomRef.current) select(svg).transition().duration(300).call(zoomRef.current.transform, zoomIdentity)
  }, [])

  // Topology work only depends on the shapes, not the screen size.
  const districtGeometry = useMemo(() => buildDistrictGeometry(wards), [wards])

  const geo = useMemo(() => {
    if (!size.width || !size.height) return null
    const projection = geoMercator().fitExtent(
      [
        [PAD, PAD],
        [size.width - PAD, size.height - PAD],
      ],
      wards,
    )
    const path = geoPath(projection)
    const shapes = wards.features.map((f) => ({
      id: f.properties.ward_id,
      label: f.properties.name ?? f.properties.ward_id,
      district: f.properties.district || null,
      d: path(f) ?? "",
      bounds: path.bounds(f),
      centroid: path.centroid(f),
      area: path.area(f),
    }))

    // District label sits at the area-weighted centre of its wards.
    const districts = new Map<string, { x: number; y: number; area: number; x0: number; y0: number; x1: number; y1: number }>()
    for (const p of shapes) {
      if (!p.district || !Number.isFinite(p.centroid[0])) continue
      const d = districts.get(p.district) ?? { x: 0, y: 0, area: 0, x0: Infinity, y0: Infinity, x1: -Infinity, y1: -Infinity }
      d.x += p.centroid[0] * p.area
      d.y += p.centroid[1] * p.area
      d.area += p.area
      d.x0 = Math.min(d.x0, p.bounds[0][0])
      d.y0 = Math.min(d.y0, p.bounds[0][1])
      d.x1 = Math.max(d.x1, p.bounds[1][0])
      d.y1 = Math.max(d.y1, p.bounds[1][1])
      districts.set(p.district, d)
    }
    const districtInfo = [...districts.entries()].map(([name, d]) => ({
      name,
      x: d.area ? d.x / d.area : (d.x0 + d.x1) / 2,
      y: d.area ? d.y / d.area : (d.y0 + d.y1) / 2,
      bounds: [
        [d.x0, d.y0],
        [d.x1, d.y1],
      ] as [[number, number], [number, number]],
    }))

    return {
      path,
      shapes,
      paths: shapes.map(({ id, label, district, d }) => ({ id, label, district, d })),
      districtInfo,
      outline: path(districtGeometry.outer) ?? "",
      borders: path(districtGeometry.borders) ?? "",
    }
  }, [wards, size, districtGeometry])

  /** Transform that fits the focused district (or the whole map), within the pan limits. */
  const focusTransform = useCallback(
    (district: string | null) => {
      const target = district && geo ? geo.districtInfo.find((d) => d.name === district) : null
      if (!target) return zoomIdentity
      const [[x0, y0], [x1, y1]] = target.bounds
      const k = Math.min(MAX_ZOOM, 0.88 / Math.max((x1 - x0) / size.width, (y1 - y0) / size.height))
      return constrain(
        zoomIdentity
          .translate(size.width / 2, size.height / 2)
          .scale(Math.max(1, k))
          .translate(-(x0 + x1) / 2, -(y0 + y1) / 2),
      )
    },
    [geo, size.width, size.height, constrain],
  )

  // Zoom (animated) when the focused district changes — and only then, so
  // colour changes, legend highlights or results updates never reset the view.
  const lastFocus = useRef<string | null | undefined>(undefined)
  useEffect(() => {
    const svg = svgRef.current
    if (!svg || !zoomRef.current || !geo) return
    if (lastFocus.current === focusDistrict) return
    const first = lastFocus.current === undefined
    lastFocus.current = focusDistrict
    if (first && !focusDistrict) return
    select(svg).transition().duration(650).call(zoomRef.current.transform, focusTransform(focusDistrict))
  }, [focusDistrict, geo, focusTransform])

  // When the map is resized: keep a focused district fitted, otherwise keep the
  // user's zoom, just clamped to the new size.
  useEffect(() => {
    const svg = svgRef.current
    const z = zoomRef.current
    if (!svg || !z || !geo || lastFocus.current === undefined) return
    const next = lastFocus.current ? focusTransform(lastFocus.current) : constrain(zoomTransform(svg))
    select(svg).call(z.transform, next)
    setScale(next.k)
    // Run on size changes only.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [size.width, size.height])

  // Escape leaves district focus — but not when it is closing a dropdown or dialog.
  useEffect(() => {
    if (!focusDistrict || !onSelectDistrict) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape" || e.defaultPrevented) return
      const target = e.target as Element | null
      if (target?.closest?.('[role="listbox"], [role="dialog"], [role="menu"], [data-radix-popper-content-wrapper]')) return
      if (document.querySelector('[data-radix-popper-content-wrapper], [role="listbox"]')) return
      onSelectDistrict(null)
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [focusDistrict, onSelectDistrict])

  const highlighted = highlightedId !== selectedId ? geo?.paths.find((p) => p.id === highlightedId) : undefined
  const selected = geo?.paths.find((p) => p.id === selectedId)
  const focusOutline = useMemo(
    () => (geo && focusDistrict ? (geo.path(districtGeometry.outline(focusDistrict)) ?? "") : ""),
    [geo, focusDistrict, districtGeometry],
  )
  const highlightOutline = useMemo(
    () =>
      geo && highlightDistrict && highlightDistrict !== focusDistrict
        ? (geo.path(districtGeometry.outline(highlightDistrict)) ?? "")
        : "",
    [geo, highlightDistrict, focusDistrict, districtGeometry],
  )

  return (
    <div ref={wrapRef} className="relative h-full w-full overflow-hidden">
      <svg
        ref={svgRef}
        role="group"
        aria-label="Ward map. Scroll or pinch to zoom, drag to pan, click a ward for its details."
        width={size.width}
        height={size.height}
        className="absolute inset-0 block cursor-grab touch-none select-none"
        onPointerLeave={handleLeave}
        // Moving onto empty map (outside every ward) clears the hover.
        onPointerMove={(e) => {
          if (hoveringRef.current && !(e.target as Element).classList?.contains("ward")) handleLeave()
        }}
      >
        <g ref={layerRef}>
          {/* Ward fills, each with its own boundary line. */}
          {geo && (
            <WardPaths
              paths={geo.paths}
              fillFor={fillFor}
              focusDistrict={focusDistrict}
              onEnter={handleEnter}
              onSelect={handleSelect}
              onLeaveFocus={handleLeave}
            />
          )}
          {/* District borders: heavier than ward lines. */}
          {geo && <path d={geo.borders} className="district-border" />}
          {/* Outer edge of the whole area. */}
          {geo && <path d={geo.outline} className="area-outline" />}
          {focusOutline && <path d={focusOutline} className="district-focus" />}
          {highlightOutline && <path d={highlightOutline} className="district-highlight" />}
          {/* Drawn last so ward outlines sit above their neighbours. */}
          {selected && <path d={selected.d} className="ward-selected" />}
          {highlighted && <path d={highlighted.d} className="ward-outline" />}
          <g ref={labelsRef}>
            {geo?.districtInfo.map((d) => (
              <text
                key={d.name}
                x={d.x}
                y={d.y}
                className={focusDistrict && focusDistrict !== d.name ? "district-label dimmed" : "district-label"}
                textAnchor="middle"
                dominantBaseline="central"
              >
                {d.name}
              </text>
            ))}
          </g>
        </g>
      </svg>

      {focusDistrict && onSelectDistrict && (
        <div className="absolute top-3 left-3">
          <Button variant="outline" size="sm" className="bg-card/90 shadow-sm backdrop-blur" onClick={() => onSelectDistrict(null)}>
            <ArrowLeft /> All districts
          </Button>
        </div>
      )}

      <div className="absolute top-3 right-3 flex flex-col overflow-hidden rounded-md border bg-card/90 shadow-sm backdrop-blur">
        <Button variant="ghost" size="icon" className="rounded-none" onClick={() => zoomBy(1.6)} disabled={scale >= MAX_ZOOM} aria-label="Zoom in">
          <Plus />
        </Button>
        <Button variant="ghost" size="icon" className="rounded-none border-t" onClick={() => zoomBy(1 / 1.6)} disabled={scale <= 1} aria-label="Zoom out">
          <Minus />
        </Button>
        <Button
          variant="ghost"
          size="icon"
          className="rounded-none border-t"
          onClick={() => (focusDistrict && onSelectDistrict ? onSelectDistrict(null) : resetZoom())}
          disabled={scale <= 1 && !focusDistrict}
          aria-label="Reset zoom"
        >
          <Maximize2 />
        </Button>
      </div>
      <div
        ref={badgeRef}
        style={{ display: "none" }}
        className="pointer-events-none absolute bottom-3 left-3 rounded-md border bg-card/90 px-2 py-0.5 text-xs text-muted-foreground tabular-nums shadow-sm backdrop-blur"
      />
    </div>
  )
}

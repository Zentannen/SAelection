// Display config per party code. Only the main ward-winning parties get their
// own colour on the map; everyone else shares "Other" so the map stays readable
// (eight hues on one map can't all be told apart, especially with colour-blindness).
// `ink` is the text colour for a label printed inside that party's fill.
export const PARTIES: Record<string, { name: string; color?: string; ink?: string }> = {
  ANC: { name: "African National Congress", color: "var(--party-anc)", ink: "#ffffff" },
  DA: { name: "Democratic Alliance", color: "var(--party-da)", ink: "#ffffff" },
  EFF: { name: "Economic Freedom Fighters", color: "var(--party-eff)", ink: "#ffffff" },
  MK: { name: "uMkhonto weSizwe", color: "var(--party-mk)", ink: "#ffffff" },
  IFP: { name: "Inkatha Freedom Party", color: "var(--party-ifp)", ink: "#0b0b0b" },
  PA: { name: "Patriotic Alliance" },
  ActionSA: { name: "ActionSA", color: "var(--party-actionsa)", ink: "#ffffff" },
  "FF+": { name: "Freedom Front Plus" },
}

export const OTHER = "Other"
export const OTHER_COLOR = "var(--party-other)"
const OTHER_INK = "var(--party-other-ink)"

/** Parties with their own colour, in legend order. */
export const MAIN_PARTIES = Object.keys(PARTIES).filter((p) => PARTIES[p].color)

export const partyColor = (code: string) => PARTIES[code]?.color ?? OTHER_COLOR
export const partyInk = (code: string) => (PARTIES[code]?.color ? PARTIES[code].ink : undefined) ?? OTHER_INK
export const partyName = (code: string) => (code === OTHER ? "Other parties" : (PARTIES[code]?.name ?? code))

/** Map a party code to itself if it has its own colour, otherwise to "Other". */
export const colourBucket = (code: string) => (PARTIES[code]?.color ? code : OTHER)

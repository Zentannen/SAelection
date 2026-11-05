// Display config per party code. The Gauteng model forecasts five parties and
// "Other": DA, ANC, ActionSA, MK and EFF get their own colour, every other party
// is counted in "Other", in the forecast, the count and past results alike.
// Names are always shown next to colours and the map legend can highlight one
// party's wards. `ink` is the text colour for a label printed inside that party's fill.
export const PARTIES: Record<string, { name: string; color?: string; ink?: string }> = {
  // Order: 2026 Gauteng forecast, largest first.
  DA: { name: "Democratic Alliance", color: "var(--party-da)", ink: "#ffffff" },
  ANC: { name: "African National Congress", color: "var(--party-anc)", ink: "#ffffff" },
  ActionSA: { name: "ActionSA", color: "var(--party-actionsa)", ink: "#0b0b0b" },
  EFF: { name: "Economic Freedom Fighters", color: "var(--party-eff)", ink: "#ffffff" },
  MK: { name: "uMkhonto weSizwe", color: "var(--party-mk)", ink: "#ffffff" },
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

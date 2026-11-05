// Display config per party code. Only the main ward-winning parties get their
// own colour on the map; everyone else shares "Other" so the map stays readable
// (eight hues on one map can't all be told apart, especially with colour-blindness).
export const PARTIES: Record<string, { name: string; color?: string }> = {
  ANC: { name: "African National Congress", color: "var(--party-anc)" },
  DA: { name: "Democratic Alliance", color: "var(--party-da)" },
  EFF: { name: "Economic Freedom Fighters", color: "var(--party-eff)" },
  MK: { name: "uMkhonto weSizwe", color: "var(--party-mk)" },
  IFP: { name: "Inkatha Freedom Party", color: "var(--party-ifp)" },
  PA: { name: "Patriotic Alliance" },
  ActionSA: { name: "ActionSA" },
  "FF+": { name: "Freedom Front Plus" },
}

export const OTHER_COLOR = "var(--party-other)"

/** Parties with their own colour, in legend order. */
export const MAIN_PARTIES = Object.keys(PARTIES).filter((p) => PARTIES[p].color)

export const partyColor = (code: string) => PARTIES[code]?.color ?? OTHER_COLOR
export const partyName = (code: string) => PARTIES[code]?.name ?? code

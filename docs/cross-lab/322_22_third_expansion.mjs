/**
 * CROSS LAB addon: 3-2-2 / 2車ライン×2車ライン
 * Keep the original four CROSS tickets and add the best third-place
 * candidate from the unselected three-rider line to each 1st/2nd order.
 *
 * The main model must provide thirdPlaceScores (not merely race points or B).
 * Pure helper; importing it does not change an existing website by itself.
 */
export function expand322TwoTwoCrossThird({
  lines,
  selectedLineIndexes,
  tickets,
  thirdPlaceScores = {},
  thirdPlaceCandidate = null,
}) {
  const unchanged = (reason) => ({ applied: false, reason, tickets, addedTickets: [], candidate: null });
  if (!Array.isArray(lines) || !Array.isArray(tickets)
      || !Array.isArray(selectedLineIndexes) || selectedLineIndexes.length !== 2) {
    return unchanged("invalid_input");
  }
  const groups = lines.map((line) => Array.isArray(line) ? line.map(Number) : []);
  if (groups.length !== 3 || groups.map((g) => g.length).sort().join() !== "2,2,3") {
    return unchanged("not_322");
  }
  const [ai, bi] = selectedLineIndexes;
  if (!Number.isInteger(ai) || !Number.isInteger(bi) || ai === bi ||
      !groups[ai] || !groups[bi] || groups[ai].length !== 2 || groups[bi].length !== 2) {
    return unchanged("not_two_two_cross");
  }
  const threeLine = groups.find((g) => g.length === 3);
  const score = (car) => {
    const value = Number(thirdPlaceScores[car]);
    return thirdPlaceScores[car] == null || !Number.isFinite(value) ? -Infinity : value;
  };
  const ranked = [...threeLine].filter((car) => score(car) > -Infinity)
    .sort((a, b) => score(b) - score(a) || a - b);
  const candidate = thirdPlaceCandidate == null ? ranked[0] : Number(thirdPlaceCandidate);
  if (!threeLine.includes(candidate)) return unchanged("three_line_third_missing");
  const a = new Set(groups[ai]);
  const b = new Set(groups[bi]);
  const pairs = new Map();
  for (const ticket of tickets) {
    if (!Array.isArray(ticket) || ticket.length !== 3) return unchanged("invalid_ticket");
    const [first, second] = ticket.map(Number);
    if (!((a.has(first) && b.has(second)) || (b.has(first) && a.has(second)))) {
      return unchanged("ticket_not_cross");
    }
    pairs.set(`${first}-${second}`, [first, second]);
  }
  // CROSS LAB's four-point selector: two opposite 1st/2nd orders, two thirds each.
  if (tickets.length !== 4 || pairs.size !== 2) return unchanged("not_four_point_cross");
  const [p, q] = [...pairs.values()];
  if (p[0] !== q[1] || p[1] !== q[0]) return unchanged("not_opposite_orders");
  const existing = new Set(tickets.map((t) => t.map(Number).join("-")));
  const addedTickets = [...pairs.values()].map(([first, second]) => [first, second, candidate])
    .filter((t) => !existing.has(t.join("-")));
  return {
    applied: addedTickets.length === 2,
    reason: addedTickets.length === 2 ? "expanded_to_six" : "already_included",
    candidate,
    addedTickets,
    tickets: [...tickets.map((t) => t.map(Number)), ...addedTickets],
  };
}

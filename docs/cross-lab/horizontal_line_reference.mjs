/**
 * CROSS LAB horizontal line reference (the "-" edge in 1-2 / 3-4).
 * Pure reference: NEVER appends bets to a CROSS ticket or changes selection.
 * odds formats: {"1-2-3": 17.6} or [{first:1,second:2,third:3,odds:17.6}]
 */

function normalizeReferenceOdds(odds, riders) {
  const allowed = new Set(riders);
  const map = new Map();
  function put(first, second, third, price) {
    const cars = [Number(first), Number(second), Number(third)];
    const value = Number(price && typeof price === "object" ? price.odds : price);
    if (cars.some(car => !allowed.has(car)) || new Set(cars).size !== 3 ||
        !Number.isFinite(value) || value <= 0) return;
    map.set(cars.join("-"), value);
  }
  if (Array.isArray(odds)) {
    for (const item of odds) {
      if (Array.isArray(item)) put(...item);
      else if (item && typeof item === "object")
        put(item.first, item.second, item.third, item.odds);
    }
  } else if (odds && typeof odds === "object") {
    for (const [key, value] of Object.entries(odds)) {
      const parts = key.split("-");
      if (parts.length !== 3 || parts.some(part => !/^[0-9]+$/.test(part))) continue;
      put(...parts, value);
    }
  }
  return map;
}

export function buildHorizontalLineReference({lines, selectedLineIndexes, odds = {}, includeReverse = true} = {}) {
  const reject = reason => ({available:false, reason, lines:[], paths:[]});
  if (!Array.isArray(lines) || !Array.isArray(selectedLineIndexes) ||
      selectedLineIndexes.length !== 2) return reject("invalid_input");
  const groups = lines.map(line => Array.isArray(line) ? line.map(Number) : []);
  const riders = groups.flat();
  if (riders.length < 3 || riders.length > 9 ||
      riders.some(car => !Number.isInteger(car) || car < 1 || car > 9) ||
      new Set(riders).size !== riders.length) return reject("invalid_lines");
  const [ai, bi] = selectedLineIndexes;
  if (ai === bi || [ai, bi].some(index => !Number.isInteger(index) ||
      index < 0 || index >= groups.length || groups[index].length < 2)) {
    return reject("invalid_selected_pair");
  }

  const book = normalizeReferenceOdds(odds, riders);
  const expected = riders.length * (riders.length - 1) * (riders.length - 2);
  const completeMarket = book.size === expected;
  const bookWeight = completeMarket
    ? [...book.values()].reduce((sum, value) => sum + 1 / value, 0)
    : null;
  const linesOutput = [];
  const paths = [];

  [ai, bi].forEach((lineIndex, slot) => {
    const [head, bante] = groups[lineIndex];
    const line = {
      label: slot === 0 ? "A" : "B", lineIndex, head, bante,
      display: head + "－" + bante, orders: [],
    };
    const orders = [{first:head, second:bante, direction:"forward", label:"順筋"}];
    if (includeReverse) orders.push({first:bante, second:head, direction:"reverse", label:"逆筋"});

    for (const order of orders) {
      const thirds = riders.filter(car => car !== order.first && car !== order.second)
        .map(third => ({
          third, ticket:[order.first, order.second, third],
          odds: book.get([order.first, order.second, third].join("-")) ?? null,
        }));
      const priced = thirds.filter(item => item.odds !== null);
      const weight = priced.reduce((sum, item) => sum + 1/item.odds, 0);
      const path = {
        lineIndex, lineLabel:line.label, label:order.label, direction:order.direction,
        first:order.first, second:order.second,
        pair:order.first + "－" + order.second,
        thirds, completeOdds:priced.length === thirds.length,
        availableOddsCount:priced.length, expectedOddsCount:thirds.length,
        minOdds:priced.length ? Math.min(...priced.map(row => row.odds)) : null,
        maxOdds:priced.length ? Math.max(...priced.map(row => row.odds)) : null,
        // Only a normalized inverse-odds market WEIGHT, not a calibrated race probability.
        marketShare:completeMarket && bookWeight > 0 ? weight / bookWeight : null,
      };
      line.orders.push(path);
      paths.push(path);
    }
    linesOutput.push(line);
  });
  return {
    available:true, reason:"reference_only", lines:linesOutput, paths,
    completeMarket, availableMarketOddsCount:book.size, expectedMarketOddsCount:expected,
    note:"横棒「－」は同ライン1・2着の参考表示。クロスの買い目・点数・購入額には自動加算しません。市場内重みは3連単全オッズの逆数正規化であり的中確率ではありません。",
  };
}

/**
 * Add a separate reference section after the CROSS output.
 * The caller owns layout and CSS; this renderer never changes any CROSS cards.
 */
export function renderHorizontalLineReference(container, reference, {showThirds = true} = {}) {
  if (!container || typeof container.replaceChildren !== "function") {
    throw new TypeError("DOM container required");
  }
  container.replaceChildren();
  if (!reference?.available) return;
  const doc = container.ownerDocument;
  const textNode = (tag, content, className) => {
    const node = doc.createElement(tag);
    node.textContent = content;
    if (className) node.className = className;
    return node;
  };
  const section = doc.createElement("section");
  section.className = "cross-lab-line-reference";
  section.append(
    textNode("h3", "━ 横棒・ライン筋（参考）"),
    textNode("p", "同ラインの1・2着。クロスの推奨買い目には含めません。")
  );
  for (const line of reference.lines) {
    const card = doc.createElement("div");
    card.className = "cross-lab-reference-card";
    card.append(textNode("h4", "LINE " + line.label + "  " + line.display));
    for (const path of line.orders) {
      const row = doc.createElement("div");
      row.className = "cross-lab-reference-" + path.direction;
      row.append(textNode("strong", path.label + "  " + path.first + "→" + path.second));
      const oddsSummary = path.minOdds === null ? "オッズ未取得"
        : path.completeOdds ? "3連単 " + path.minOdds + "～" + path.maxOdds + "倍"
        : "部分取得 " + path.availableOddsCount + "/" + path.expectedOddsCount + "件";
      row.append(textNode("span", " / " + oddsSummary));
      if (path.marketShare !== null)
        row.append(textNode("span", " / 市場内重み " + (100*path.marketShare).toFixed(1) + "%"));
      if (showThirds) {
        const details = path.thirds.map(item =>
          item.third + "(" + (item.odds === null ? "—" : item.odds + "倍") + ")").join("  ");
        row.append(textNode("p", "3着別： " + details));
      }
      card.append(row);
    }
    section.append(card);
  }
  section.append(textNode("small", reference.note));
  container.append(section);
}

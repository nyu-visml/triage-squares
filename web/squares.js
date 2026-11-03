// Squares stacks view (Ren et al., IEEE TVCG 2017), reimplemented in D3 v7.
//
// Encoding (one column per class, score bins on the vertical axis):
//   right of a class's axis : instances PREDICTED as this class (p == col)
//       solid   = correct prediction (y == p), coloured by the true class
//       striped = false positive  (y != p), coloured by the true class y
//   left of a class's axis  : false negatives, i.e. instances LABELLED as this class
//       but predicted as something else (y == col, p != col); outlined in the
//       colour of the predicted class p
//   vertical position = bin of the score of the predicted class (s)
//   width of a group  = its count (linear, or square root as an option)
//   inside a bin, errors sit next to the axis and correct predictions further out.
// Clicking a group lists its instances in the linked table.
//
// Input JSON: {classes: [...], items: [{i, y, p, s, ...extra fields}]}
// Extra fields (e.g. cause, f_y, f_p in CIFAR-10) are shown as extra table columns.

const Squares = (() => {
  const BINS = 10;
  const BASE_KEYS = new Set(["i", "y", "p", "s", "nbrs"]);
  // ColorBrewer Paired, as in the original paper for 6-12 classes (Ren et al., Sec. 4.2).
  const PALETTE = d3.schemePaired;

  const binOf = s => Math.min(BINS - 1, Math.max(0, Math.floor(s * BINS)));
  const kindLabel = { tp: "Correct", fp: "False positive", fn: "False negative" };

  // ---------------------------------------------------------------- data -> groups
  function buildGroups(data) {
    const C = data.classes.length;
    const groups = [];
    // right side: predicted class = column; split by true class
    const right = d3.group(data.items, d => d.p, d => binOf(d.s), d => d.y);
    for (const [col, byBin] of right)
      for (const [bin, byY] of byBin)
        for (const [y, items] of byY)
          groups.push({ side: "right", col, bin, cls: y, kind: y === col ? "tp" : "fp", items });
    // left side: true class = column; only errors; split by predicted class
    const errors = data.items.filter(d => d.y !== d.p);
    const left = d3.group(errors, d => d.y, d => binOf(d.s), d => d.p);
    for (const [col, byBin] of left)
      for (const [bin, byP] of byBin)
        for (const [p, items] of byP)
          groups.push({ side: "left", col, bin, cls: p, kind: "fn", items });

    // per-class precision / recall for the column headers
    const stats = d3.range(C).map(c => {
      const tp = data.items.filter(d => d.p === c && d.y === c).length;
      const predicted = data.items.filter(d => d.p === c).length;
      const actual = data.items.filter(d => d.y === c).length;
      return { precision: predicted ? tp / predicted : 0, recall: actual ? tp / actual : 0 };
    });
    const accuracy = data.items.filter(d => d.y === d.p).length / data.items.length;
    return { groups, stats, accuracy };
  }

  // stack order inside one bin: errors next to the axis (by class index), then correct
  function stackOrder(a, b) {
    if (a.kind === "tp" && b.kind !== "tp") return 1;
    if (b.kind === "tp" && a.kind !== "tp") return -1;
    return a.cls - b.cls;
  }

  // largest summed width (in count units) of any one bin, per side
  function maxStack(groups, f) {
    const sums = d3.rollup(groups, v => d3.sum(v, g => f(g.items.length)),
                           g => `${g.side}|${g.col}|${g.bin}`);
    let right = 0, left = 0;
    for (const [k, v] of sums) k.startsWith("right") ? (right = Math.max(right, v)) : (left = Math.max(left, v));
    return { right, left };
  }

  // ---------------------------------------------------------------- one panel
  function drawPanel(host, panel, opts) {
    const { data, title, built, id } = panel;
    const { groups, stats, accuracy } = built;
    const C = data.classes.length;
    const color = c => PALETTE[c % PALETTE.length];

    const width = Math.max(720, host.node().clientWidth);
    const m = { top: 62, right: 12, bottom: 16, left: 44 };
    const colW = (width - m.left - m.right) / C;
    const binH = opts.binHeight;
    const plotH = binH * BINS;
    const axisFrac = 0.36;                       // share of a column left of its axis
    const leftW = colW * axisFrac - 4;           // room for false negatives
    const rightW = colW * (1 - axisFrac) - 8;    // room for predictions
    const axisX = c => m.left + c * colW + colW * axisFrac;
    const binY = b => m.top + (BINS - 1 - b) * binH;   // bin 9 (scores 0.9-1.0) on top

    const f = opts.scale === "sqrt" ? Math.sqrt : n => n;
    const unit = Math.min(rightW / opts.maxStack.right, leftW / Math.max(opts.maxStack.left, 1e-9));
    const widthOf = n => Math.max(opts.minWidth, f(n) * unit);

    host.append("div").attr("class", "panel-title")
      .html(`<b>${title}</b> <span>accuracy ${accuracy.toFixed(3)} · n = ${data.items.length}</span>`);
    const svg = host.append("svg").attr("width", width).attr("height", m.top + plotH + m.bottom);

    // stripe patterns, one per class
    const defs = svg.append("defs");
    d3.range(C).forEach(c => {
      const pat = defs.append("pattern").attr("id", `${id}-stripe-${c}`)
        .attr("patternUnits", "userSpaceOnUse").attr("width", 4).attr("height", 4)
        .attr("patternTransform", "rotate(45)");
      pat.append("rect").attr("width", 4).attr("height", 4).attr("fill", color(c)).attr("fill-opacity", 0.18);
      pat.append("line").attr("x1", 0).attr("y1", 0).attr("x2", 0).attr("y2", 4)
        .attr("stroke", color(c)).attr("stroke-width", 2.2);
    });

    // score axis + horizontal bin guides
    const y = d3.scaleLinear().domain([0, 1]).range([m.top + plotH, m.top]);
    svg.append("g").attr("class", "score-axis").attr("transform", `translate(${m.left - 6},0)`)
      .call(d3.axisLeft(y).ticks(5).tickSize(3));
    svg.append("g").attr("class", "bin-guides").selectAll("line").data(d3.range(1, BINS)).join("line")
      .attr("x1", m.left).attr("x2", width - m.right)
      .attr("y1", b => y(b / BINS)).attr("y2", b => y(b / BINS));

    // column headers and axes
    const head = svg.append("g").selectAll("g").data(d3.range(C)).join("g");
    head.append("rect").attr("x", c => axisX(c) - 5).attr("y", 12).attr("width", 10).attr("height", 10)
      .attr("fill", c => color(c));
    head.append("text").attr("class", "col-name").attr("x", c => axisX(c) + 9).attr("y", 21)
      .text(c => data.classes[c]);
    head.append("text").attr("class", "col-stat").attr("x", c => axisX(c)).attr("y", 40)
      .attr("text-anchor", "middle")
      .text(c => `P ${stats[c].precision.toFixed(2)} · R ${stats[c].recall.toFixed(2)}`);
    svg.append("g").selectAll("line").data(d3.range(C)).join("line").attr("class", "class-axis")
      .attr("x1", c => axisX(c)).attr("x2", c => axisX(c)).attr("y1", m.top - 6).attr("y2", m.top + plotH);

    // lay out stacks
    const laid = [];
    for (const [, bucket] of d3.group(groups, g => g.side, g => g.col, g => g.bin).entries()) {
      for (const [, byBin] of bucket) for (const [, gs] of byBin) {
        gs.sort(stackOrder);
        let offset = 0;
        for (const g of gs) {
          const w = widthOf(g.items.length);
          const x0 = g.side === "right" ? axisX(g.col) + 1 + offset : axisX(g.col) - 1 - offset - w;
          laid.push({ ...g, x: x0, w, yTop: binY(g.bin) + 2, h: binH - 4 });
          offset += w;
        }
      }
    }
    // clip a stack that overflows its half-column (only possible with the minimum width)
    const cell = svg.append("g").attr("class", "cells");
    const rects = cell.selectAll("rect").data(laid).join("rect")
      .attr("class", g => `cell ${g.kind}`)
      .attr("x", g => g.x).attr("y", g => g.yTop).attr("width", g => g.w).attr("height", g => g.h)
      .attr("fill", g => g.kind === "tp" ? color(g.cls) : g.kind === "fp" ? `url(#${id}-stripe-${g.cls})` : "#fff")
      .attr("stroke", g => g.kind === "fn" ? color(g.cls) : g.kind === "fp" ? color(g.cls) : "none")
      .attr("stroke-width", g => g.kind === "fn" ? 1.6 : 0.8)
      .on("mousemove", (ev, g) => opts.tooltip.show(ev, tooltipHtml(g, data)))
      .on("mouseleave", () => opts.tooltip.hide())
      .on("click", (ev, g) => { ev.stopPropagation(); opts.onSelect(panel, g); });
    svg.on("click", () => opts.onSelect(panel, null));
    panel.rects = rects;
  }

  function tooltipHtml(g, data) {
    const n = g.items.length, cls = data.classes;
    const lo = (g.bin / BINS).toFixed(1), hi = ((g.bin + 1) / BINS).toFixed(1);
    const what = g.kind === "tp" ? `label = prediction = ${cls[g.col]}`
      : g.kind === "fp" ? `predicted ${cls[g.col]}, labeled ${cls[g.cls]}`
      : `labeled ${cls[g.col]}, predicted ${cls[g.cls]}`;
    return `<b>${kindLabel[g.kind]}</b> · ${n} instance${n === 1 ? "" : "s"}<br>${what}<br>score ${lo}–${hi}`;
  }

  // ---------------------------------------------------------------- linked table
  function renderTable(host, panel, g, opts) {
    host.selectAll("*").remove();
    if (!g) { host.append("p").attr("class", "hint").text("Click any box to list its instances here."); return; }
    const { data } = panel, cls = data.classes;
    const rows = [...g.items].sort((a, b) => b.s - a.s);
    const extra = [...new Set(rows.flatMap(d => Object.keys(d)))].filter(k => !BASE_KEYS.has(k));
    host.append("p").attr("class", "table-caption")
      .html(`${panel.title} · ${tooltipHtml(g, data).replace(/<br>/g, " · ")}` +
            (rows.length > opts.maxRows ? ` · showing the ${opts.maxRows} highest-scoring` : ""));
    const table = host.append("table");
    const cols = ["#", ...(opts.sprite ? ["image"] : []), "label", "prediction", "score", ...extra];
    table.append("thead").append("tr").selectAll("th").data(cols).join("th").text(d => d);
    const tr = table.append("tbody").selectAll("tr").data(rows.slice(0, opts.maxRows)).join("tr");
    tr.append("td").text(d => d.i);
    if (opts.sprite) tr.append("td").append("div").attr("class", "thumb").style("background-image", `url(${opts.sprite.url})`)
      .style("background-position", d => `-${(d.i % opts.sprite.cols) * opts.sprite.size}px -${Math.floor(d.i / opts.sprite.cols) * opts.sprite.size}px`);
    tr.append("td").text(d => cls[d.y]);
    tr.append("td").text(d => cls[d.p]);
    tr.append("td").text(d => d.s.toFixed(3));
    extra.forEach(k => tr.append("td").text(d => d[k] ?? ""));
  }

  // ---------------------------------------------------------------- consistency check
  // Per-class TP / FP / FN counted from the groups that are actually drawn, to compare
  // with sklearn's confusion matrix (notebooks/check_counts.py).
  function counts(data) {
    const { groups } = buildGroups(data);
    return data.classes.map((name, c) => {
      const sum = (side, kind) => d3.sum(groups.filter(g => g.col === c && g.side === side && g.kind === kind),
                                         g => g.items.length);
      return { class: name, TP: sum("right", "tp"), FP: sum("right", "fp"), FN: sum("left", "fn") };
    });
  }


  // ---------------------------------------------------------------- public
  // panels: [{title, file}]; options: see defaults below
  async function render(root, panelsSpec, options = {}) {
    const opts = { scale: "linear", binHeight: 24, minWidth: 1.5, maxRows: 200, sprite: null, ...options };
    const container = d3.select(root);
    const panels = await Promise.all(panelsSpec.map(async (p, k) => {
      const data = await d3.json(p.file);
      return { ...p, id: `panel${k}`, data, built: buildGroups(data) };
    }));

    panels.forEach(p => { console.log(`Counts drawn for ${p.title}:`); console.table(counts(p.data)); });

    const tooltip = (() => {
      const tip = d3.select("body").selectAll("div.sq-tooltip").data([0]).join("div").attr("class", "sq-tooltip");
      return {
        show: (ev, html) => {
          tip.html(html).style("opacity", 1);
          const { width: tw, height: th } = tip.node().getBoundingClientRect();
          const x = ev.clientX + 14 + tw > window.innerWidth ? ev.clientX - 14 - tw : ev.clientX + 14;
          const y = ev.clientY + 14 + th > window.innerHeight ? ev.clientY - 14 - th : ev.clientY + 14;
          tip.style("left", `${x}px`).style("top", `${y}px`);
        },
        hide: () => tip.style("opacity", 0),
      };
    })();

    let selected = null;
    function draw() {
      container.selectAll("*").remove();
      const f = opts.scale === "sqrt" ? Math.sqrt : n => n;
      // one shared width scale across panels, so the models can be compared directly
      const ms = panels.map(p => maxStack(p.built.groups, f));
      const maxStackAll = { right: d3.max(ms, d => d.right), left: d3.max(ms, d => d.left) };
      const views = container.append("div").attr("class", "views");
      const tableHost = container.append("div").attr("class", "instance-table");
      const onSelect = (panel, g) => {
        selected = g ? { panel, g } : null;
        panels.forEach(p => p.rects && p.rects.classed("dim", d => !!selected && !(p === panel && d === g))
                                            .classed("selected", d => !!selected && p === panel && d === g));
        renderTable(tableHost, panel, g, opts);
      };
      panels.forEach(p => drawPanel(views.append("div").attr("class", "panel"), p,
        { ...opts, maxStack: maxStackAll, tooltip, onSelect }));
      renderTable(tableHost, panels[0], null, opts);
    }
    draw();
    let t;
    window.addEventListener("resize", () => { clearTimeout(t); t = setTimeout(draw, 150); });
    return { setScale: s => { opts.scale = s; draw(); } };
  }

  return { render, binOf, counts };
})();

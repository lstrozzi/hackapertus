// Minimal read-only knowledge-graph viewer for the Graph tab (server markup:
// operations/kg_graph_view.py; data: /api/operations/kg/graph).
//
// Same rendering library as the platform's own KG page: the vis-network UMD pair
// vendored under /static/lib/vis-network/. Physics options and the stabilization
// contract (freeze physics when the stabilization loop ends) are carried over from
// the platform's graph.js so the layout looks and settles the same way; everything
// write-side (context menus, hiding, layout
// saving, DotNet interop) is deliberately absent — this viewer only reads.
//
// The script does not run when injected markup appears; it boots itself on the
// first click of the Graph tab, then re-reads on Reload. The community is fixed
// by the server (DefaultCommunity_Parlament) and its selector renders disabled —
// the payload always describes exactly that one graph. The Entity class select
// is enabled: it is repopulated from every load and filters the rendered nodes
// client-side, with "All" rendering the full graph.
(function () {
  "use strict";

  var LIBRARY = [
    "/static/lib/vis-network/vis-data.min.js",
    "/static/lib/vis-network/vis-network.min.js"
  ];

  // Entity classes are coloured, box-shaped and white-labelled, as on the platform page.
  var PALETTE = [
    ["#3f7fbf", "#2f5e8c"], ["#5da33c", "#40712a"], ["#8a5bb8", "#653f8c"],
    ["#c2574f", "#96403b"], ["#c98a2d", "#9a6920"], ["#3aa3a0", "#2c7c7a"],
    ["#b8478f", "#8c366c"], ["#5468d6", "#3f4ea3"], ["#7a9b2e", "#5c7522"],
    ["#a4682e", "#7d4f22"], ["#4a9b77", "#387659"], ["#9b54c9", "#763f9a"],
    ["#c96a9b", "#9a4f76"], ["#5f8fae", "#48708a"], ["#8fae54", "#6d863d"],
    ["#ae6a5f", "#865048"]
  ];

  var libraryPromise = null;
  var payload = null;
  var network = null;
  var nodes = null;
  var edges = null;
  var booted = false;
  var visible = { entities: [], relationships: [] };

  function element(id) { return document.getElementById(id); }

  function setStatus(message) {
    var status = element("kg-graph-status");
    if (status) { status.textContent = message; }
  }

  function setSelection(html) {
    var selection = element("kg-graph-selection");
    if (selection) { selection.innerHTML = html; }
  }

  function escapeText(value) {
    var holder = document.createElement("span");
    holder.textContent = String(value === null || value === undefined ? "" : value);
    return holder.innerHTML;
  }

  function loadLibrary() {
    if (window.vis && window.vis.Network) { return Promise.resolve(); }
    if (libraryPromise) { return libraryPromise; }
    libraryPromise = LIBRARY.reduce(function (chain, src) {
      return chain.then(function () {
        return new Promise(function (resolve, reject) {
          var script = document.createElement("script");
          script.src = src;
          script.onload = resolve;
          script.onerror = function () { reject(new Error("could not load " + src)); };
          document.head.appendChild(script);
        });
      });
    }, Promise.resolve());

    return libraryPromise;
  }

  function canvasMessage(message) {
    var canvas = element("kg-graph-canvas");
    if (canvas) {
      canvas.innerHTML = '<div class="kg-graph-message">' + escapeText(message) + "</div>";
    }
  }

  function describeError(error) {
    if (!error) { return "unknown error"; }
    return error.type ? error.type + ": " + (error.message || "") : String(error);
  }

  function classColor(index) {
    var pair = PALETTE[index % PALETTE.length];

    return { background: pair[0], border: pair[1] };
  }

  function selectedClass() {
    var select = element("kg-graph-class");

    return select ? select.value : "";
  }

  // Class filter is client-side: keep nodes of the chosen class and only the
  // relationships whose both endpoints are still visible.
  function applyFilter() {
    if (!payload) { return { entities: [], relationships: [] }; }
    var wanted = selectedClass();
    var entities = payload.entities.filter(function (entity) {
      return !wanted || entity.label === wanted;
    });
    var ids = new Set(entities.map(function (entity) { return entity.id; }));
    var relationships = payload.relationships.filter(function (rel) {
      return ids.has(rel.source) && ids.has(rel.target);
    });

    return { entities: entities, relationships: relationships };
  }

  function populateClassFilter() {
    var select = element("kg-graph-class");
    if (!select) { return; }
    var labels = [];
    payload.entities.forEach(function (entity) {
      if (labels.indexOf(entity.label) === -1) { labels.push(entity.label); }
    });
    labels.sort();
    var previous = select.value;
    select.innerHTML = '<option value="">All</option>';
    labels.forEach(function (label) {
      var option = document.createElement("option");
      option.value = label;
      option.textContent = label;
      select.appendChild(option);
    });
    select.value = labels.indexOf(previous) !== -1 ? previous : "";
  }

  // Curve duplicated node pairs, exactly as the platform does, so parallel
  // relationships of different kinds stay readable instead of overplotting.
  function buildEdges(relationships) {
    var byPair = new Map();
    relationships.forEach(function (rel) {
      var key = rel.source < rel.target ? rel.source + "|" + rel.target : rel.target + "|" + rel.source;
      var bucket = byPair.get(key) || [];
      bucket.push(rel);
      byPair.set(key, bucket);
    });

    return relationships.map(function (rel, index) {
      var key = rel.source < rel.target ? rel.source + "|" + rel.target : rel.target + "|" + rel.source;
      var bucket = byPair.get(key);
      var edge = {
        id: "rel-" + index,
        from: rel.source,
        to: rel.target,
        label: rel.relationship,
        title: rel.relationship + "\n" + rel.source + " → " + rel.target,
        font: { color: "#5b6b73", size: 11, strokeWidth: 0, align: "horizontal" },
        color: { color: "#9db1ba", highlight: "#356b77", hover: "#5b8595" },
        smooth: { enabled: false }
      };
      if (bucket.length > 1 && bucket[0] !== rel) {
        var position = bucket.indexOf(rel);
        edge.smooth = {
          enabled: true,
          type: position % 2 === 0 ? "curvedCW" : "curvedCCW",
          roundness: Math.min(0.45, 0.1 + (Math.floor(position / 2) + 1) * 0.07)
        };
      }
      return edge;
    });
  }

  function buildNodes(entities) {
    var seen = new Map();

    return entities.map(function (entity) {
      if (!seen.has(entity.label)) { seen.set(entity.label, seen.size); }
      var color = classColor(seen.get(entity.label));

      return {
        id: entity.id,
        label: entity.name,
        title: entity.label + ": " + entity.id
          + (entity.names.length > 1 ? "\nalso: " + entity.names.slice(1).join(", ") : ""),
        shape: "box",
        color: color,
        font: { color: "#ffffff", size: 14, face: "system-ui, sans-serif" },
        margin: { top: 8, right: 12, bottom: 8, left: 12 },
        borderWidth: 1,
        shadow: { enabled: true, color: "rgba(32,36,43,0.25)", size: 6, x: 0, y: 2 }
      };
    });
  }

  function networkOptions() {
    // Physics block transplanted from the platform's graph.js.
    return {
      autoResize: true,
      physics: {
        barnesHut: {
          gravitationalConstant: -50000,
          centralGravity: 0.05,
          springLength: 160,
          springConstant: 0.5,
          damping: 1.0,
          avoidOverlap: 1
        },
        minVelocity: 0.75,
        solver: "barnesHut",
        stabilization: { enabled: true, iterations: 1000, updateInterval: 25, fit: true },
        adaptiveTimestep: true
      },
      layout: { improvedLayout: true },
      edges: {
        smooth: false,
        shadow: false,
        arrows: { to: { enabled: true, scaleFactor: 1 } }
      },
      interaction: {
        hover: true,
        multiselect: false,
        navigationButtons: false,
        keyboard: false
      }
    };
  }

  function renderGraph() {
    var canvas = element("kg-graph-canvas");
    if (!canvas || !payload || !payload.ok) { return; }

    nodes = new vis.DataSet(buildNodes(visible.entities));
    edges = new vis.DataSet(buildEdges(visible.relationships));

    canvas.innerHTML = "";
    if (network) { network.destroy(); network = null; }
    network = new vis.Network(canvas, { nodes: nodes, edges: edges }, networkOptions());
    // The platform's stabilization contract (graph.js): when the stabilization
    // loop ends — iteration cap reached or velocities settled — physics is
    // switched off so the layout freezes where it settled, then a redraw paints
    // the final frame. Without the freeze the barnesHut run never stops and the
    // graph floats indefinitely.
    network.once("stabilizationIterationsDone", function () {
      network.setOptions({ physics: false });
      network.redraw();
      network.fit();
    });

    network.on("click", function (params) {
      if (params.nodes.length === 0) { return; }
      var entity = visible.entities.find(function (item) { return item.id === params.nodes[0]; });
      if (!entity) { return; }
      var degree = visible.relationships.reduce(function (total, rel) {
        return total + (rel.source === entity.id || rel.target === entity.id ? 1 : 0);
      }, 0);
      setSelection('<span class="kg-graph-sel-type">' + escapeText(entity.label) + "</span> · "
        + '<span class="kg-graph-sel-label">' + escapeText(entity.name) + "</span> · id "
        + escapeText(entity.id) + " · " + degree + " connection" + (degree === 1 ? "" : "s")
        + (entity.names.length > 1 ? " · also known as " + escapeText(entity.names.slice(1).join(", ")) : ""));
    });
  }

  function describePayload() {
    var filter = selectedClass() ? " · class " + selectedClass() : "";
    var counts = visible.entities.length + " of " + payload.entities.length + " entities · " + visible.relationships.length + " relationships";
    var suffix = payload.edges_error ? " — relationships unavailable (" + describeError(payload.edges_error) + ")" : "";
    setStatus(counts + filter + " · community " + payload.community + suffix);
  }

  // Re-render from the cached payload with the current class filter; no refetch.
  function refresh() {
    if (!payload || !payload.ok) { return; }
    visible = applyFilter();
    if (!visible.entities.length) {
      if (network) { network.destroy(); network = null; }
      nodes = null;
      edges = null;
      canvasMessage(payload.entities.length
        ? "No entities of class " + selectedClass() + " in community " + payload.community + "."
        : "No entities in community " + payload.community + " yet — run an extraction first.");
    } else {
      renderGraph();
    }
    describePayload();
  }

  async function loadGraph() {
    var reload = element("kg-graph-reload");
    if (reload) { reload.disabled = true; }
    setStatus("Loading graph…");
    try {
      const response = await fetch("/api/operations/kg/graph");
      const result = await response.json();
      if (!response.ok || !result.ok) { throw new Error(describeError(result.error || ("HTTP " + response.status))); }
      payload = result;
      populateClassFilter();
      refresh();
    } catch (error) {
      payload = null;
      canvasMessage("Graph could not be loaded: " + describeError(error));
      setStatus("Graph load failed");
    } finally {
      if (reload) { reload.disabled = false; }
    }

    return undefined;
  }

  async function boot() {
    if (booted) { return; }
    booted = true;
    var reload = element("kg-graph-reload");
    if (reload) { reload.disabled = false; reload.addEventListener("click", function () { loadGraph(); }); }
    var classSelect = element("kg-graph-class");
    if (classSelect) { classSelect.addEventListener("change", function () { refresh(); }); }
    try {
      await loadLibrary();
    } catch (error) {
      canvasMessage("Graph library could not be loaded: " + describeError(error));
      setStatus("Graph library missing");

      return;
    }
    await loadGraph();
  }

  document.addEventListener("click", function (event) {
    var tab = event.target.closest && event.target.closest('[data-operation-page="graph"]');
    if (tab) { boot(); }
  });
})();

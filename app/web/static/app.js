/* Goloka search page. Vanilla JS, no dependencies.
 *
 * Safety rule: crawled pages are untrusted. Their titles, snippets and URLs are only ever put on the page
 * as text nodes (textContent / createTextNode) or as an href that has been checked to be http(s). Nothing
 * from the server is ever parsed as HTML. tests/test_web.py fails if HTML-parsing APIs appear in this file.
 */
"use strict";

(function () {
  const API = "/api/v1";
  const $ = (selector, root) => (root || document).querySelector(selector);

  const state = {
    health: null,
    searchSeq: 0,
    pollTimer: null,
    failedUrls: new Map(),   // job id -> list of failed urls (kept across re-renders)
    apiKey: "",
    owner: false,            // true once the server accepts this browser for adding pages
  };
  try { state.apiKey = sessionStorage.getItem("goloka-api-key") || ""; } catch (_) { /* storage blocked: fine */ }

  // ---------------------------------------------------------------- small helpers
  function h(tag, attrs) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs || {})) {
      if (value === false || value == null) continue;
      if (key === "class") node.className = value;
      else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
      else node.setAttribute(key, value === true ? "" : String(value));
    }
    for (const kid of Array.prototype.slice.call(arguments, 2).flat()) {
      if (kid == null || kid === false) continue;
      node.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
    }
    return node;
  }
  const clear = (node) => { while (node.firstChild) node.removeChild(node.firstChild); return node; };
  const plural = (n, one, many) => `${n} ${n === 1 ? one : many || one + "s"}`;
  const fixed = (n) => (typeof n === "number" ? n.toFixed(2) : "");
  const clamp01 = (n) => Math.max(0, Math.min(1, n));

  function safeHref(url) {
    try {
      const u = new URL(url);
      return u.protocol === "http:" || u.protocol === "https:" ? u.href : null;
    } catch (_) { return null; }
  }

  class ApiProblem extends Error {
    constructor(status, code, message) { super(message); this.status = status; this.code = code; }
  }

  async function api(path, options) {
    const opts = options || {};
    const headers = Object.assign({ Accept: "application/json" }, opts.headers || {});
    if (state.apiKey) headers["X-API-Key"] = state.apiKey;
    let response;
    try {
      response = await fetch(API + path, Object.assign({}, opts, { headers }));
    } catch (_) {
      throw new ApiProblem(0, "network", "Cannot reach the Goloka server. Check that it is still running.");
    }
    let body = null;
    try { body = await response.json(); } catch (_) { /* no JSON body */ }
    if (!response.ok) {
      const err = body && body.error;
      throw new ApiProblem(response.status, (err && err.code) || "error",
        (err && err.message) || `The server answered with an error (${response.status}).`);
    }
    return body;
  }

  // ---------------------------------------------------------------- notices
  function showNotice(message, kind, extra) {
    const box = $("#notice");
    box.append(h("div", { class: "notice" + (kind === "error" ? " error" : "") }, message, extra));
  }
  const clearNotices = () => clear($("#notice"));

  function keyForm(retry) {
    const input = h("input", { type: "password", "aria-label": "API key", autocomplete: "off", required: true });
    return h("form", {
      onsubmit: async (event) => {
        event.preventDefault();
        state.apiKey = input.value.trim();
        try { sessionStorage.setItem("goloka-api-key", state.apiKey); } catch (_) { /* ignore */ }
        await checkOwner();
        retry();
      },
    }, input, h("button", { type: "submit", class: "primary" }, "Save key"));
  }

  function reportProblem(problem, retry) {
    if (problem instanceof ApiProblem && problem.status === 401) {
      showNotice("This server needs an API key.", "error", keyForm(retry));
    } else {
      showNotice(problem.message || "Something went wrong.", "error");
    }
  }

  // ---------------------------------------------------------------- status line
  async function refreshStatus() {
    const box = $("#status");
    try {
      state.health = await api("/health");
    } catch (problem) {
      clear(box).append(h("span", {}, problem.message));
      return null;
    }
    const docs = state.health.documents;
    const ok = (docs.by_status && docs.by_status.ok) || 0;
    const semantic = state.health.indexes.semantic;
    const keywordOnly = semantic.status === "disabled";
    $("#modes").hidden = keywordOnly;               // no choice to make on a keyword-only server
    const meaning = { ready: "ready", not_loaded: "loads on first use", unavailable: "unavailable" }[semantic.status] || semantic.status;
    clear(box).append(h("span", {}, `${plural(ok, "page")} added`));
    if (!keywordOnly) box.append(h("span", {}, `Search by meaning: ${meaning}`));
    if (semantic.status === "unavailable" && semantic.last_error) box.append(h("span", {}, semantic.last_error));
    if (docs.pending_indexing > 0 && state.owner) {
      const button = h("button", { type: "button", class: "linklike" }, `Index ${plural(docs.pending_indexing, "new page")}`);
      button.addEventListener("click", () => runIndex(button));
      box.append(button);
    }
    return state.health;
  }

  async function runIndex(button) {
    button.disabled = true;
    button.textContent = "Indexing, this can take a few minutes the first time";
    try {
      const report = await api("/index", { method: "POST" });
      if (report.status === "partial") showNotice(report.warnings.join(" ") || "Some pages could not be indexed yet.");
    } catch (problem) {
      reportProblem(problem, () => runIndex(button));
    }
    await refreshStatus();
  }

  // ---------------------------------------------------------------- search
  // Filler words are not highlighted (the server ignores them in keyword matching too). Approximate English list.
  const FILLER = new Set(("a an the is are was were be been am of to in on at by for from with and or but not it its this that " +
    "these those who whom what which when where why how do does did i you he she we they me my your").split(" "));

  function queryTerms(query) {
    const words = query.toLowerCase().match(/[\p{L}\p{N}]{2,}/gu) || [];
    return Array.from(new Set(words.filter((w) => !FILLER.has(w)))).slice(0, 12);
  }

  function highlight(text, terms) {
    if (!terms.length) return [text];
    const escaped = terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
    // whole words only: "the" must not light up inside "theorem"
    const pattern = new RegExp(`(?<![\\p{L}\\p{N}])(${escaped.join("|")})(?![\\p{L}\\p{N}])`, "giu");
    return text.split(pattern).map((part, i) => (i % 2 === 1 ? h("mark", {}, part) : part));
  }

  function meter(kind, label, fraction, valueText, description) {
    if (fraction == null) {
      return h("div", { class: "meter none" }, h("span", {}, label), h("span", { class: "value" }, "no match"));
    }
    const fill = h("span", { class: `fill ${kind}` });
    fill.style.width = `${Math.round(clamp01(fraction) * 100)}%`;
    return h("div", { class: "meter", title: description },
      h("span", {}, label), h("span", { class: "track", "aria-hidden": "true" }, fill),
      h("span", { class: "value" }, valueText));
  }

  function resultRow(r, terms, mode, bestKeyword) {
    const href = safeHref(r.url);
    const title = r.title || r.url;
    const heading = h("h2", {}, href ? h("a", { href, target: "_blank", rel: "noopener noreferrer" }, title) : title);

    const keyword = r.bm25_score == null
      ? meter("kw", "Keywords", null)
      : meter("kw", "Keywords", bestKeyword > 0 ? r.bm25_score / bestKeyword : 0, fixed(r.bm25_score),
        "Keyword score (BM25). The bar is relative to the best keyword match in these results.");
    const meaning = r.semantic_score == null
      ? meter("sem", "Meaning", null)
      : meter("sem", "Meaning", r.semantic_score, fixed(r.semantic_score),
        "Similarity of meaning, from 0 (unrelated) to 1 (very close).");

    const signals = h("div", { class: "signals" },
      mode === "semantic" ? null : keyword, mode === "keyword" ? null : meaning);
    const extras = [];
    if (mode === "both") extras.push(`Overall ${fixed(r.combined_score)}`);
    if (r.title_score) extras.push("title matches");

    return h("li", { class: "result" }, h("div", {},
      heading,
      h("p", { class: "url" }, r.url),
      h("p", { class: "snippet" }, highlight(r.snippet || "", terms)),
      signals,
      extras.length ? h("p", { class: "extra" }, extras.join(", ")) : null));
  }

  function legend() {
    return h("details", { class: "legend" },
      h("summary", {}, "How to read the bars"),
      h("p", {}, "Keywords counts pages that contain your words, spelled the same way. The bar compares each page with the best keyword match in this list. " +
        "Meaning compares what the page is about with what you asked, even when the words differ. Its bar is the similarity itself, from 0 to 1. " +
        "A page can match on one and not the other. When a way of matching finds nothing, the row says no match."));
  }

  function currentOptions() {
    return {
      mode: ($("input[name=mode]:checked") || {}).value || "both",
      method: $("#method").value,
      count: Number($("#count").value) || 10,
      weak: $("#weak").checked,
    };
  }

  async function runSearch(query) {
    const results = clear($("#results"));
    clearNotices();
    const seq = ++state.searchSeq;
    const opts = currentOptions();
    results.append(h("p", { class: "summary" }, "Searching…"));

    const params = new URLSearchParams({ q: query, k: String(opts.count) });
    let path = "/search";
    if (opts.mode === "keyword") path = "/search/keyword";
    else if (opts.mode === "semantic") path = "/search/semantic";
    else {
      params.set("method", opts.method);
      if (opts.weak) params.set("min_score", "0.2");
    }

    let data;
    try {
      data = await api(`${path}?${params}`);
    } catch (problem) {
      if (seq !== state.searchSeq) return;
      clear(results);
      reportProblem(problem, () => runSearch(query));
      return;
    }
    if (seq !== state.searchSeq) return;           // a newer search replaced this one
    clear(results);
    for (const warning of data.warnings || []) showNotice(warning);
    const health = (await refreshStatus()) || state.health;

    if (!data.results.length) {
      const ok = health ? (health.documents.by_status.ok || 0) : null;
      if (ok === 0) {
        results.append(h("div", { class: "empty" }, h("h2", {}, "No pages to search yet"),
          state.owner
            ? [h("p", {}, "Add pages first, then search them. "), h("a", { href: "#add" }, "Add pages")]
            : h("p", {}, "The site owner has not added any pages yet.")));
      } else {
        results.append(h("div", { class: "empty" }, h("h2", {}, `Nothing found for “${query}”`),
          h("p", {}, `Goloka searched ${ok == null ? "your" : plural(ok, "page")} and found no match. ` +
            "It only knows pages you have added, so a question about something those pages do not cover will come back empty. " +
            "Try different words, or add more pages.")));
      }
      return;
    }
    const terms = queryTerms(query);
    const bestKeyword = Math.max(0, ...data.results.map((r) => r.bm25_score || 0));
    const how = opts.mode === "keyword" ? "keywords" : opts.mode === "semantic" ? "meaning" : "keywords and meaning";
    results.append(
      h("p", { class: "summary" }, `${plural(data.total_returned, "page")} found by ${how}, in ${Math.max(1, Math.round(data.took_ms))} ms`),
      h("ol", { class: "results" }, data.results.map((r) => resultRow(r, terms, opts.mode, bestKeyword))),
      legend());
  }

  // ---------------------------------------------------------------- add pages
  const JOB_STATE = { pending: "Waiting", running: "Running", completed: "Finished", failed: "Failed" };
  const JOB_PHASE = { queued: "Waiting for its turn", crawling: "Reading pages", indexing: "Building the search index", done: "" };

  function jobRow(job) {
    const p = job.progress;
    const active = job.status === "pending" || job.status === "running";
    const head = h("div", { class: "job-head" },
      h("span", { class: "job-url" }, job.seed_url),
      h("span", { class: "job-state " + (job.status === "failed" ? "failed" : job.status === "completed" ? "finished" : "") },
        JOB_STATE[job.status] || job.status, JOB_PHASE[job.phase] && active ? `: ${JOB_PHASE[job.phase].toLowerCase()}` : ""));
    const row = h("div", { class: "job" }, head);
    if (active) {
      row.append(h("progress", { max: p.max_pages, value: p.urls_processed, "aria-label": "Crawl progress" }));
    }
    row.append(h("p", {}, `${plural(p.pages_crawled, "page")} saved`
      + (p.pages_failed ? `, ${p.pages_failed} failed` : "") + (p.pages_skipped ? `, ${p.pages_skipped} skipped` : "")
      + (active ? `, ${p.urls_processed} of up to ${p.max_pages} checked` : "") + "."));
    if (job.error) row.append(h("p", { class: "err" }, job.error));
    if (job.index && job.status === "completed") {
      const i = job.index;
      if (i.status === "completed") row.append(h("p", {}, "Searchable now."));
      else row.append(h("p", { class: "err" }, `Saved, but not fully searchable yet. ${(i.warnings || []).join(" ") || i.error || ""}`.trim()));
    }
    if (p.pages_failed > 0 || job.status === "failed") {
      const known = state.failedUrls.get(job.id);
      if (known) {
        row.append(h("ul", {}, known.length ? known.map((u) => h("li", {}, `${u.url}: ${u.detail || u.status}`)) : h("li", {}, "No failed pages recorded.")));
      } else {
        row.append(h("button", { type: "button", class: "linklike", onclick: () => showFailed(job.id) }, "Show failed pages"));
      }
    }
    return row;
  }

  async function showFailed(jobId) {
    try {
      const data = await api(`/jobs/${jobId}/urls?status=error&limit=10`);
      state.failedUrls.set(jobId, data.items);
    } catch (problem) {
      state.failedUrls.set(jobId, [{ url: "", status: "error", detail: problem.message }]);
    }
    pollJobs();
  }

  async function pollJobs() {
    clearTimeout(state.pollTimer);
    const box = $("#jobs");
    let data;
    try {
      data = await api("/jobs?limit=8");
    } catch (problem) {
      clear(box);
      reportProblem(problem, pollJobs);
      return;
    }
    clear(box);
    if (!data.items.length) box.append(h("p", { class: "help" }, "No crawls yet."));
    data.items.forEach((job) => box.append(jobRow(job)));
    const active = data.items.some((j) => j.status === "pending" || j.status === "running");
    if (active) state.pollTimer = setTimeout(pollJobs, 1500);
    else refreshStatus();
  }

  async function startCrawl(event) {
    event.preventDefault();
    const message = $("#crawl-message");
    message.textContent = "";
    const url = $("#crawl-url").value.trim();
    if (!url) { message.textContent = "Enter the address to start from."; return; }
    const button = $("#crawl-form button.primary");
    button.disabled = true;
    try {
      await api("/crawl", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url, max_pages: Number($("#crawl-pages").value) || undefined, max_depth: Number($("#crawl-depth").value) }),
      });
      pollJobs();
    } catch (problem) {
      message.textContent = problem.status === 401 ? "This server needs an API key. Enter it on the Search page." : problem.message;
      if (problem.status === 401) { showView("search"); reportProblem(problem, () => {}); }
    } finally {
      button.disabled = false;
    }
  }

  // ---------------------------------------------------------------- owner access
  // Adding pages is for the site owner only. The server enforces this (it answers 401 to everyone without the
  // key); the page just hides the screen so visitors are not shown something they cannot use.
  async function checkOwner() {
    try {
      await api("/jobs?limit=1");
      state.owner = true;
    } catch (_) {
      state.owner = false;
    }
    $("#nav-add").hidden = !state.owner;
    const button = $("#owner-btn");
    button.hidden = state.owner && !state.apiKey;     // server has no key set: nobody signs in or out
    button.textContent = state.owner ? "Sign out" : "Owner sign-in";
    return state.owner;
  }

  function ownerPrompt() {
    const retry = () => {
      if (state.owner) { if ((location.hash || "").startsWith("#add")) route(); else location.hash = "#add"; return; }
      state.apiKey = "";
      try { sessionStorage.removeItem("goloka-api-key"); } catch (_) { /* ignore */ }
      clearNotices();
      showNotice("That key was not accepted.", "error", keyForm(retry));
    };
    showNotice("Adding pages is for the site owner. Enter the owner key to continue.", null, keyForm(retry));
  }

  function ownerButton() {
    if (state.owner) {
      state.apiKey = "";
      try { sessionStorage.removeItem("goloka-api-key"); } catch (_) { /* ignore */ }
      checkOwner().then(() => { if ((location.hash || "").startsWith("#add")) route(); else refreshStatus(); });
    } else {
      if ((location.hash || "").startsWith("#add")) route(); else location.hash = "#add";
    }
  }

  // ---------------------------------------------------------------- views and routing
  function showView(name) {
    $("#view-search").hidden = name !== "search";
    $("#view-add").hidden = name !== "add";
    document.querySelectorAll("nav a").forEach((a) => {
      if (a.dataset.view === name) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
    });
    if (name !== "add") clearTimeout(state.pollTimer);
  }

  function route() {
    const hash = location.hash || "#search";
    if (hash.startsWith("#add") && !state.owner) {
      showView("search");
      document.title = "Owner sign-in - Goloka";
      clear($("#results"));
      clearNotices();
      ownerPrompt();
      return;
    }
    if (hash.startsWith("#add")) {
      showView("add");
      document.title = "Add pages - Goloka";
      pollJobs();
      return;
    }
    showView("search");
    const match = /^#search\/(.+)$/.exec(hash);
    if (match) {
      let query = "";
      try { query = decodeURIComponent(match[1]); } catch (_) { query = ""; }
      $("#q").value = query;
      document.title = `${query} - Goloka`;
      if (query.trim()) runSearch(query.trim());
    } else {
      document.title = "Goloka search";
      clear($("#results"));
      clearNotices();
    }
  }

  function submitSearch(query) {
    const q = query.trim();
    if (!q) return;
    const target = "#search/" + encodeURIComponent(q);
    if (location.hash === target) runSearch(q); else location.hash = target;
  }

  function init() {
    $("#search-form").addEventListener("submit", (event) => { event.preventDefault(); submitSearch($("#q").value); });
    document.querySelectorAll("[data-query]").forEach((b) => b.addEventListener("click", () => submitSearch(b.dataset.query)));
    $("#search-form").addEventListener("change", (event) => {
      if (event.target.id === "q") return;           // typing alone never starts a search
      const q = $("#q").value.trim();
      if (q) runSearch(q);
    });
    $("#crawl-form").addEventListener("submit", startCrawl);
    $("#preset-math").addEventListener("click", () => {
      $("#crawl-url").value = "https://en.wikipedia.org/wiki/History_of_mathematics";
      $("#crawl-pages").value = "40";
      $("#crawl-depth").value = "1";
    });
    $("#owner-btn").addEventListener("click", ownerButton);
    window.addEventListener("hashchange", route);
    checkOwner().then(refreshStatus).then(route);
  }

  init();
})();

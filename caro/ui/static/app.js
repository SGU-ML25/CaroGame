"use strict";

const N = 11;
const HUMAN = "human";
const $ = (id) => document.getElementById(id);
const RULE_LABELS = {
  P1_win: "P1: thắng ngay",
  P2_block_loss: "P2: chặn thua ngay",
  P3_double_attack: "P3: tạo đòn kép",
  P4_block_double_attack: "P4: phá đòn kép của đối thủ",
  P5_four_attack: "P5: tạo tứ",
  P6_evaluate: "P6: điểm vị trí",
  P7_center_random: "P7: hoà điểm, chọn gần tâm",
  Fallback_random: "Ngẫu nhiên",
};

const ui = {
  mode: "pve",
  agents: [],
  budgets: null,
  game: null,        // ván đang chơi, đúng dạng GET /games/{id}
  evePly: null,      // vị trí đang xem lại; null = hiển thị nước mới nhất
  replay: null,      // {record, ply}
  busy: false,       // đang chờ tác tử đi
  creatingEve: false,
  eveApplied: null,  // cấu hình trên bảng lúc tạo ván máy–máy đang chơi
  running: null,     // định danh lượt chạy tự động hiện tại
  playing: false,    // đang phát replay
  token: 0,          // tăng mỗi khi đổi ván / chế độ, để bỏ qua phản hồi của ván cũ
  heat: null,        // lựa chọn bản đồ nhiệt của người dùng; null = tự chọn
  cells: [],
};

async function api(path, options = {}) {
  const init = { method: options.method || "GET" };
  if (options.body !== undefined) {
    init.headers = { "Content-Type": "application/json" };
    init.body = JSON.stringify(options.body);
  }
  let response;
  try {
    response = await fetch(path, init);
  } catch {
    throw new Error("Không kết nối được tới server.");
  }
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = data && data.detail;
    throw new Error(typeof detail === "string" ? detail : `Lỗi ${response.status} từ server.`);
  }
  return data;
}

function showError(error) {
  $("error").textContent = error ? error.message : "";
  $("error").hidden = !error;
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const coord = (i) => `${String.fromCharCode(65 + (i % N))}${Math.floor(i / N) + 1}`;
const randomSeed = () => Math.floor(Math.random() * 1_000_000);
const number = (v) => (typeof v === "number" ? v.toLocaleString("vi-VN") : String(v));

// ---- dựng bàn cờ -----------------------------------------------------------
function buildBoard() {
  for (let c = 0; c < N; c++) {
    const span = document.createElement("span");
    span.textContent = String.fromCharCode(65 + c);
    $("cols").append(span);
  }
  for (let r = 0; r < N; r++) {
    const span = document.createElement("span");
    span.textContent = r + 1;
    $("rows").append(span);
  }
  for (let i = 0; i < N * N; i++) {
    const cell = document.createElement("button");
    cell.type = "button";
    cell.className = "cell";
    cell.addEventListener("click", () => humanMove(i));
    $("board").append(cell);
    ui.cells.push(cell);
  }
}

// Nước đang hiển thị và thống kê tương ứng, chung cho ván đang chơi và replay.
function view() {
  if (ui.mode === "replay") {
    if (!ui.replay) return { moves: [], stats: [], winning: [], opening: 0 };
    const { record, ply } = ui.replay;
    const stats = record.moves.map((_, i) => {
      const s = {};
      for (const key of ["time_ms", "nodes", "sims"]) if (Array.isArray(record[key])) s[key] = record[key][i];
      return s;
    });
    return {
      moves: record.moves.slice(0, ply),
      stats: stats.slice(0, ply),
      winning: ply === record.moves.length ? record.winning_line : [],
      opening: 0,
    };
  }
  const g = ui.game;
  if (!g) return { moves: [], stats: [], winning: [], opening: 0 };
  if (ui.mode === "eve" && ui.evePly !== null && ui.evePly < g.moves.length) {
    return {
      moves: g.moves.slice(0, ui.evePly),
      stats: g.stats.slice(0, ui.evePly),
      winning: [],
      opening: g.opening_plies,
    };
  }
  return { moves: g.moves, stats: g.stats, winning: g.winning_line, opening: g.opening_plies };
}

function canPlayHuman() {
  const g = ui.game;
  return ui.mode === "pve" && g && g.to_play === HUMAN && !ui.busy;
}

function latestStats(v) {
  for (let i = v.stats.length - 1; i >= 0; i--) {
    if (Object.keys(v.stats[i]).length) return { stats: v.stats[i], ply: i };
  }
  return null;
}

function heatValues(stats) {
  const choice = $("heat").value;
  if (!stats || !stats[choice]) return null;
  const values = stats[choice];
  const top = Math.max(...Object.values(values), 0);
  return top > 0 ? { values, top } : null;
}

function renderBoard(v, latest) {
  const stones = new Map(v.moves.map((a, i) => [a, i % 2 === 0 ? "x" : "o"]));
  const last = v.moves[v.moves.length - 1];
  const winning = new Set(v.winning);
  const playable = canPlayHuman();
  const stats = latest && latest.stats;
  const heat = heatValues(stats);
  // PV chỉ có nghĩa ngay sau nước của tác tử: PV[0] chính là nước vừa đi.
  const pv = stats && Array.isArray(stats.pv) && latest.ply === v.moves.length - 1 ? stats.pv : [];

  ui.cells.forEach((cell, i) => {
    const stone = stones.get(i);
    const open = !stone && playable;
    cell.className = "cell";
    if (stone) cell.classList.add(stone);
    if (open) cell.classList.add("playable");
    if (winning.has(i)) cell.classList.add("win");
    if (i === last) cell.classList.add("last");
    cell.disabled = !open;
    cell.setAttribute("aria-label", `${coord(i)}, ${stone ? stone.toUpperCase() : "trống"}`);

    cell.replaceChildren();
    if (heat && heat.values[i]) {
      const overlay = document.createElement("span");
      overlay.className = "heat";
      overlay.style.setProperty("--a", (0.08 + 0.6 * (heat.values[i] / heat.top)).toFixed(3));
      cell.title = `${coord(i)}: ${number(heat.values[i])}`;
      cell.append(overlay);
    } else {
      cell.removeAttribute("title");
    }
    if (stone) {
      const mark = document.createElement("span");
      mark.className = "mark";
      mark.textContent = stone.toUpperCase();
      cell.append(mark);
    }
    const order = pv.indexOf(i);
    if (order > 0) {
      const tag = document.createElement("span");
      tag.className = "pv";
      tag.textContent = order;
      cell.append(tag);
    }
  });
}

function playerName(name) {
  return name === HUMAN ? "Bạn" : name;
}

function resultText(result, nameX, nameO) {
  if (result === 1) return `X thắng (${playerName(nameX)})`;
  if (result === -1) return `O thắng (${playerName(nameO)})`;
  return "Hoà: bàn đầy, không bên nào có 5 quân liên tiếp";
}

function renderStatus(v) {
  const el = $("status");
  el.classList.remove("thinking");
  if (ui.mode === "replay") {
    if (!ui.replay) {
      el.textContent = "Chọn một ván để xem lại";
      return;
    }
    const { record, ply } = ui.replay;
    const done = ply === record.moves.length;
    el.textContent = done
      ? `${resultText(record.result, record.agent_x, record.agent_o)} · ${ply} nước`
      : `${record.agent_x} (X) – ${record.agent_o} (O) · nước ${ply}/${record.moves.length}`;
    return;
  }
  const g = ui.game;
  if (!g) {
    el.textContent = "Bấm “Ván mới” để bắt đầu";
    return;
  }
  if (g.result !== null) {
    el.textContent = resultText(g.result, g.players.X, g.players.O);
    return;
  }
  if (g.to_play === HUMAN) {
    el.textContent = `Lượt của bạn (${g.side})`;
    return;
  }
  el.textContent = ui.busy ? `${g.to_play} (${g.side}) đang nghĩ` : `Lượt của ${g.to_play} (${g.side})`;
  el.classList.toggle("thinking", ui.busy);
}

function addStat(dl, label, value) {
  const dt = document.createElement("dt");
  dt.textContent = label;
  const dd = document.createElement("dd");
  dd.textContent = value;
  dl.append(dt, dd);
}

function renderThinking(v, latest) {
  const dl = $("thinking");
  dl.replaceChildren();
  if (!latest) {
    const dd = document.createElement("dd");
    dd.className = "empty";
    dd.textContent = "Chưa có nước nào của tác tử.";
    dl.append(dd);
  } else {
    const s = latest.stats;
    addStat(dl, "Nước", `${latest.ply + 1}. ${v.moves[latest.ply] === undefined ? "" : coord(v.moves[latest.ply])}`);
    if (s.reason !== undefined) addStat(dl, "Luật", RULE_LABELS[s.reason] || s.reason);
    if (s.depth !== undefined) addStat(dl, "Độ sâu", s.depth);
    if (s.nodes !== undefined) addStat(dl, "Số nút", number(s.nodes));
    if (s.sims !== undefined) addStat(dl, "Số mô phỏng", number(s.sims));
    if (s.time_ms !== undefined) addStat(dl, "Thời gian", `${number(s.time_ms)} ms`);
    if (s.nodes && s.time_ms) addStat(dl, "Nút/giây", number(Math.round(s.nodes / (s.time_ms / 1000))));
    if (s.score !== undefined) addStat(dl, "Điểm gốc", number(s.score));
    if (s.value !== undefined) addStat(dl, "Giá trị v(s)", s.value.toLocaleString("vi-VN", { minimumFractionDigits: 3, maximumFractionDigits: 3 }));
    if (Array.isArray(s.pv) && s.pv.length) addStat(dl, "PV", s.pv.map(coord).join(" "));
  }

  const available = [["visits", "Số lượt thăm"], ["policy", "Policy"]].filter(([key]) => latest && latest.stats[key]);
  const select = $("heat");
  select.replaceChildren(new Option("Tắt", ""), ...available.map(([key, label]) => new Option(label, key)));
  const wanted = available.some(([key]) => key === ui.heat) ? ui.heat : available.length ? available[0][0] : "";
  select.value = ui.heat === "" ? "" : wanted;      // "" nghĩa là người dùng đã tắt
  $("heat-field").hidden = available.length === 0;
}

function renderMoves(v) {
  const list = $("moves");
  list.replaceChildren();
  if (!v.moves.length) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "Chưa có nước nào.";
    list.append(li);
    return;
  }
  v.moves.forEach((a, i) => {
    const li = document.createElement("li");
    li.textContent = `${i + 1}. ${i % 2 === 0 ? "X" : "O"} ${coord(a)}`;
    if (i < v.opening) li.classList.add("opening");
    if (i === v.moves.length - 1) li.classList.add("current");
    list.append(li);
  });
  list.scrollTop = list.scrollHeight;
}

function renderControls() {
  const g = ui.game;
  const live = g && g.result === null;
  if (ui.mode === "pve") {
    const human = g && Object.values(g.players).includes(HUMAN);
    // Bên người cầm O: nước đầu của máy không lùi được.
    const minMoves = human && g.players.X !== HUMAN ? 2 : 1;
    const undoable = g && g.moves.length - g.opening_plies >= minMoves;
    $("pve-undo").disabled = ui.busy || !undoable;
  }
  $("eve-run").textContent = ui.running ? "Tạm dừng" : "Chạy";
  $("eve-stale").hidden = !(ui.mode === "eve" && live && !ui.creatingEve && eveSettings() !== ui.eveApplied);
  $("eve-new").disabled = ui.creatingEve;
  $("eve-run").disabled = ui.creatingEve || Boolean(g && !live) || (ui.busy && !ui.running);
  $("eve-step").disabled = ui.creatingEve || !live || Boolean(ui.running) || ui.busy;
  const reviewable = ui.mode === "eve" && g?.to_play === null;
  $("eve-review").hidden = !reviewable;
  if (reviewable) {
    const ply = ui.evePly ?? g.moves.length;
    $("eve-ply").textContent = `${ply} / ${g.moves.length}`;
    $("eve-prev").disabled = ui.creatingEve || ply === 0;
    $("eve-next").disabled = ui.creatingEve || ply === g.moves.length;
  }
  $("replay-play").textContent = ui.playing ? "Tạm dừng" : "Phát";
  if (ui.replay) {
    const { record, ply } = ui.replay;
    $("replay-ply").max = record.moves.length;
    $("replay-ply").value = ply;
    $("replay-prev").disabled = ply === 0;
    $("replay-next").disabled = ply === record.moves.length;
  }
}

function render() {
  const v = view();
  const latest = latestStats(v);
  renderThinking(v, latest);          // trước renderBoard: đặt lựa chọn bản đồ nhiệt
  renderBoard(v, latest);
  renderStatus(v);
  renderMoves(v);
  renderControls();
}

// ---- ngân sách --------------------------------------------------------------
function buildBudget(container) {
  const id = container.dataset.agent;
  container.innerHTML = `
    <div class="field budget-mode">
      <span>Ngân sách</span>
      <div class="segmented">
        <label><input type="radio" name="${id}-mode" value="fixed" checked> Cố định</label>
        <label><input type="radio" name="${id}-mode" value="time"> Thời gian</label>
      </div>
    </div>
    <label class="field" data-key="time_ms">Thời gian / nước (ms)
      <input type="number" min="10" max="60000" step="10" value="${ui.budgets.time.time_ms}"></label>
    <label class="field" data-key="max_depth">Độ sâu
      <input type="number" min="1" max="8" step="1" value="${ui.budgets.fixed.max_depth}"></label>
    <label class="field" data-key="max_sims">Số mô phỏng
      <input type="number" min="1" max="100000" step="100" value="${ui.budgets.fixed.max_sims}"></label>
    <p class="note">Tác tử này không dùng ngân sách tìm kiếm.</p>`;
  container.addEventListener("change", () => refreshBudget(container));
  $(id).addEventListener("change", () => refreshBudget(container));
}

function refreshBudget(container) {
  const agent = ui.agents.find((a) => a.id === $(container.dataset.agent).value);
  const keys = agent?.budget ?? [];
  const mode = container.querySelector('input[type="radio"]:checked').value;
  const active = mode === "time" ? ["time_ms"] : ["max_depth", "max_sims"];
  container.querySelector(".budget-mode").hidden = keys.length === 0;
  container.querySelector(".note").hidden = keys.length !== 0;
  for (const label of container.querySelectorAll("[data-key]")) {
    label.hidden = !(keys.includes(label.dataset.key) && active.includes(label.dataset.key));
  }
}

function readBudget(id) {
  const container = document.querySelector(`[data-agent="${id}"]`);
  // Tác tử không dùng ngân sách thì ô chọn chế độ bị ẩn nhưng vẫn giữ lựa chọn cũ: không gửi lựa chọn đó đi.
  if (container.querySelector(".budget-mode").hidden) return { mode: "fixed" };
  const mode = container.querySelector('input[type="radio"]:checked').value;
  const budget = { mode };
  for (const label of container.querySelectorAll("[data-key]")) {
    if (label.hidden) continue;
    const value = Number(label.querySelector("input").value);
    if (!Number.isInteger(value) || value <= 0) {
      throw new Error(`${label.firstChild.textContent.trim()} phải là số nguyên dương.`);
    }
    budget[label.dataset.key] = value;
  }
  return budget;
}

// Mọi ô cấu hình chỉ có tác dụng khi tạo ván: gom giá trị đang hiện để biết người dùng đã đổi gì chưa.
function eveSettings() {
  const values = ["eve-x", "eve-o", "eve-opening", "eve-seed"].map((id) => $(id).value);
  for (const side of ["eve-x", "eve-o"]) {
    const container = document.querySelector(`[data-agent="${side}"]`);
    if (container.querySelector(".budget-mode").hidden) continue;
    values.push(container.querySelector('input[type="radio"]:checked').value);
    for (const label of container.querySelectorAll("[data-key]")) {
      if (!label.hidden) values.push(label.querySelector("input").value);
    }
  }
  return JSON.stringify(values);
}

function readSeed(id) {
  const input = $(id);
  if (input.value === "") input.value = randomSeed();
  const seed = Number(input.value);
  if (!Number.isInteger(seed) || seed < 0) throw new Error("Seed phải là số nguyên không âm.");
  return seed;
}

function fillAgents(select, preferred) {
  select.replaceChildren(...ui.agents.map((a) => {
    const label = `Cấp ${a.level}: ${a.id}${a.available ? "" : " (chưa cài)"}`;
    const option = new Option(label, a.id);
    option.disabled = !a.available;
    return option;
  }));
  const available = ui.agents.filter((a) => a.available);
  const match = available.find((a) => a.level === preferred) || available[available.length - 1];
  if (match) select.value = match.id;
}

// ---- người – máy ------------------------------------------------------------
async function aiTurn(token) {
  ui.busy = true;
  render();
  try {
    const g = await api(`/games/${ui.game.id}/ai-move`, { method: "POST" });
    if (token === ui.token) ui.game = g;
  } finally {
    if (token === ui.token) {
      ui.busy = false;
      render();
    }
  }
}

async function newPve() {
  const token = ++ui.token;
  showError(null);
  try {
    const agent = $("pve-agent").value;
    const human = document.querySelector('input[name="pve-side"]:checked').value;
    const body = {
      x: human === "X" ? HUMAN : agent,
      o: human === "O" ? HUMAN : agent,
      seed: readSeed("pve-seed"),
      budget: readBudget("pve-agent"),
    };
    ui.busy = false;
    const g = await api("/games", { method: "POST", body });
    if (token !== ui.token) return;
    ui.game = g;
    render();
    if (g.to_play !== HUMAN && g.to_play !== null) await aiTurn(token);
  } catch (error) {
    if (token === ui.token) showError(error);
  }
}

async function humanMove(action) {
  if (!canPlayHuman()) return;
  const token = ui.token;
  showError(null);
  ui.busy = true;
  try {
    const g = await api(`/games/${ui.game.id}/move`, { method: "POST", body: { action } });
    if (token !== ui.token) return;
    ui.game = g;
    ui.busy = false;
    render();
    if (g.to_play !== null) await aiTurn(token);
  } catch (error) {
    if (token === ui.token) {
      ui.busy = false;
      showError(error);
      render();
    }
  }
}

async function undo() {
  if (!ui.game || ui.busy) return;
  const token = ui.token;
  showError(null);
  try {
    const g = await api(`/games/${ui.game.id}/undo`, { method: "POST" });
    if (token === ui.token) {
      ui.game = g;
      render();
    }
  } catch (error) {
    if (token === ui.token) showError(error);
  }
}

// ---- máy – máy --------------------------------------------------------------
async function fillOpenings() {
  const select = $("eve-opening");
  try {
    const openings = await api("/openings");
    for (const o of openings) select.add(new Option(`Khai cuộc ${o.id} (${o.moves.length} nước)`, `id:${o.id}`));
  } catch {
    const unavailable = new Option("Không tải được khai cuộc từ giải đấu");
    unavailable.disabled = true;
    select.add(unavailable);
  }
}

async function newEve() {
  const token = ++ui.token;
  ui.running = null;
  ui.busy = false;
  ui.creatingEve = true;
  showError(null);
  render();
  try {
    const [kind, value] = $("eve-opening").value.split(/:(.*)/s);
    const body = {
      x: $("eve-x").value,
      o: $("eve-o").value,
      seed: readSeed("eve-seed"),
      budget_x: readBudget("eve-x"),
      budget_o: readBudget("eve-o"),
    };
    if (kind === "id") body.opening_id = value;
    else body.opening_plies = Number(value);
    const settings = eveSettings();
    const g = await api("/games", { method: "POST", body });
    if (token !== ui.token) return;
    ui.game = g;
    ui.eveApplied = settings;
    ui.evePly = null;
    return g;
  } catch (error) {
    if (token === ui.token) showError(error);
  } finally {
    if (token === ui.token) {
      ui.creatingEve = false;
      render();
    }
  }
}

async function runEve() {
  if (!ui.game || ui.game.to_play === null || ui.busy || ui.running) return;
  const token = ui.token;
  const run = Symbol();
  ui.running = run;
  render();
  try {
    while (ui.running === run && ui.game.to_play !== null) {
      await aiTurn(token);
      if (ui.running !== run || ui.game.to_play === null) break;
      await sleep(Number($("eve-speed").value));
    }
  } catch (error) {
    if (token === ui.token) showError(error);
  }
  if (ui.running === run) {
    ui.running = null;
    render();
  }
}

async function stepEve() {
  if (ui.creatingEve || !ui.game || ui.running || ui.busy || ui.game.to_play === null) return;
  const token = ui.token;
  showError(null);
  try {
    await aiTurn(token);
  } catch (error) {
    if (token === ui.token) showError(error);
  }
}

function seekEve(offset) {
  const g = ui.game;
  if (ui.mode !== "eve" || ui.creatingEve || g?.to_play !== null) return;
  const ply = (ui.evePly ?? g.moves.length) + offset;
  ui.evePly = Math.max(0, Math.min(ply, g.moves.length));
  render();
}

// ---- xem lại ----------------------------------------------------------------
async function loadReplayList() {
  const token = ui.token;
  const list = $("replay-list");
  try {
    const items = await api(`/replays?q=${encodeURIComponent($("replay-q").value.trim())}`);
    if (token !== ui.token) return;
    list.replaceChildren();
    if (!items.length) {
      const li = document.createElement("li");
      li.className = "empty";
      li.textContent = $("replay-q").value.trim()
        ? "Không có ván khớp bộ lọc."
        : "Chưa có log: runs/**/games.jsonl sẽ có sau khi chạy giải đấu.";
      list.append(li);
      return;
    }
    for (const item of items) {
      const li = document.createElement("li");
      const button = document.createElement("button");
      const outcome = item.result === 1 ? "X thắng" : item.result === -1 ? "O thắng" : "Hoà";
      button.textContent = `${item.agent_x} – ${item.agent_o}`;
      const small = document.createElement("small");
      small.textContent = `Ván ${item.game_id} · ${item.tournament_id} · ${outcome} · ${item.plies} nước`;
      button.append(small);
      button.addEventListener("click", () => openReplay(item, button));
      li.append(button);
      list.append(li);
    }
  } catch (error) {
    if (token === ui.token) showError(error);
  }
}

async function openReplay(item, button) {
  const token = ui.token;
  ui.playing = false;
  showError(null);
  try {
    const params = new URLSearchParams();
    if (item.run_id !== null) params.set("run_id", item.run_id);
    if (item.tournament_id !== null) params.set("tournament_id", item.tournament_id);
    const record = await api(`/replays/${encodeURIComponent(item.game_id)}?${params}`);
    if (token !== ui.token) return;
    ui.replay = { record, ply: record.moves.length };
    for (const b of $("replay-list").querySelectorAll("button")) b.removeAttribute("aria-current");
    button.setAttribute("aria-current", "true");
    $("replay-controls").hidden = false;
    render();
  } catch (error) {
    if (token === ui.token) showError(error);
  }
}

function seekReplay(ply) {
  if (!ui.replay) return;
  ui.replay.ply = Math.max(0, Math.min(ply, ui.replay.record.moves.length));
  render();
}

async function playReplay() {
  if (!ui.replay) return;
  if (ui.playing) {
    ui.playing = false;
    render();
    return;
  }
  const token = ui.token;
  const record = ui.replay.record;
  if (ui.replay.ply === record.moves.length) ui.replay.ply = 0;
  ui.playing = true;
  render();
  while (ui.playing && token === ui.token && ui.replay && ui.replay.record === record && ui.replay.ply < record.moves.length) {
    await sleep(Number($("replay-speed").value));
    if (!ui.playing || token !== ui.token || !ui.replay || ui.replay.record !== record) return;
    seekReplay(ui.replay.ply + 1);
  }
  if (token === ui.token) {
    ui.playing = false;
    render();
  }
}

// ---- đổi chế độ, sự kiện ----------------------------------------------------
function setMode(mode) {
  ui.token++;
  ui.mode = mode;
  ui.game = null;
  ui.evePly = null;
  ui.busy = false;
  ui.creatingEve = false;
  ui.running = null;
  ui.playing = false;
  showError(null);
  for (const tab of document.querySelectorAll(".tabs button")) {
    tab.setAttribute("aria-selected", String(tab.dataset.mode === mode));
  }
  for (const name of ["pve", "eve", "replay"]) $(`panel-${name}`).hidden = name !== mode;
  if (mode === "replay") loadReplayList();
  render();
}

function bindOutput(input, output, format) {
  const update = () => { output.textContent = format(Number(input.value)); };
  input.addEventListener("input", update);
  update();
}

async function main() {
  buildBoard();
  for (const tab of document.querySelectorAll(".tabs button")) {
    tab.addEventListener("click", () => setMode(tab.dataset.mode));
  }
  try {
    [ui.agents, ui.budgets] = await Promise.all([api("/agents"), api("/budgets")]);
    fillAgents($("pve-agent"), 2);
    fillAgents($("eve-x"), 3);
    fillAgents($("eve-o"), 2);
  } catch (error) {
    showError(error);
    $("status").textContent = "Không tải được dữ liệu";
    return;
  }
  for (const container of document.querySelectorAll(".budget")) {
    buildBudget(container);
    refreshBudget(container);
  }
  $("pve-seed").value = randomSeed();
  $("eve-seed").value = randomSeed();
  $("pve-new").addEventListener("click", newPve);
  $("pve-undo").addEventListener("click", undo);
  $("eve-new").addEventListener("click", newEve);
  $("eve-run").addEventListener("click", async () => {
    if (ui.running) {
      ui.running = null;
      render();
      return;
    }
    showError(null);
    if (ui.game || await newEve()) runEve();
  });
  $("eve-step").addEventListener("click", stepEve);
  for (const type of ["input", "change"]) $("panel-eve").addEventListener(type, renderControls);
  $("eve-prev").addEventListener("click", () => seekEve(-1));
  $("eve-next").addEventListener("click", () => seekEve(1));
  $("heat").addEventListener("change", (e) => {
    ui.heat = e.target.value;
    render();
  });
  let filterTimer;
  $("replay-q").addEventListener("input", () => {
    clearTimeout(filterTimer);
    filterTimer = setTimeout(loadReplayList, 250);
  });
  $("replay-ply").addEventListener("input", (e) => seekReplay(Number(e.target.value)));
  $("replay-prev").addEventListener("click", () => seekReplay(ui.replay.ply - 1));
  $("replay-next").addEventListener("click", () => seekReplay(ui.replay.ply + 1));
  $("replay-play").addEventListener("click", playReplay);
  document.addEventListener("keydown", (e) => {
    if (ui.mode !== "replay" || !ui.replay || e.target.matches("input, select")) return;
    if (e.key === "ArrowLeft") seekReplay(ui.replay.ply - 1);
    if (e.key === "ArrowRight") seekReplay(ui.replay.ply + 1);
  });
  bindOutput($("eve-speed"), $("eve-speed-out"), (v) => `${v} ms`);
  bindOutput($("replay-speed"), $("replay-speed-out"), (v) => `${v} ms`);
  render();
  fillOpenings();
  newPve();
}

main();

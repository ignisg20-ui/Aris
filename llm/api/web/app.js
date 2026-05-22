// Aris chat UI — talks to /chat (SSE streaming) and /health.
// No external dependencies; ships inside aris.exe.
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const messagesEl = $("messages");
  const inputEl = $("input");
  const form = $("composer");
  const sendBtn = $("send");
  const stopBtn = $("stop");
  const healthDot = document.querySelector("#health .dot");
  const healthText = $("health-text");
  const systemPromptEl = $("system-prompt");
  const newChatBtn = $("new-chat");
  const tempEl = $("temperature");
  const tempV = $("temperature-v");
  const toppEl = $("top_p");
  const toppV = $("top_p-v");
  const topkEl = $("top_k");
  const maxTokEl = $("max_new_tokens");
  const streamEl = $("stream");

  /** @type {Array<{role:string, content:string}>} */
  let history = [];
  /** @type {AbortController | null} */
  let activeController = null;

  function setHealth(state, text) {
    healthDot.className = "dot dot-" + state;
    healthText.textContent = text;
  }

  function autoSize(el) {
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 240) + "px";
  }

  function renderMessage(role, text, opts = {}) {
    const wrap = document.createElement("div");
    wrap.className = "msg " + role + (opts.error ? " error" : "");
    const r = document.createElement("div");
    r.className = "role";
    r.textContent = role;
    const body = document.createElement("div");
    body.className = "body";
    body.textContent = text;
    if (opts.streaming) body.classList.add("cursor");
    wrap.appendChild(r);
    wrap.appendChild(body);
    messagesEl.appendChild(wrap);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return body;
  }

  function clearChat() {
    history = [];
    messagesEl.innerHTML = "";
    const sys = systemPromptEl.value.trim();
    if (sys) renderMessage("system", sys);
  }

  function buildPayload(stream) {
    const sys = systemPromptEl.value.trim();
    const messages = [];
    if (sys) messages.push({ role: "system", content: sys });
    for (const m of history) messages.push(m);
    return {
      messages,
      sampling: {
        temperature: parseFloat(tempEl.value),
        top_p: parseFloat(toppEl.value),
        top_k: parseInt(topkEl.value, 10) || 0,
        max_new_tokens: parseInt(maxTokEl.value, 10) || 256,
      },
      stream,
    };
  }

  async function checkHealth() {
    try {
      setHealth("idle", "connecting…");
      const r = await fetch("/health");
      const j = await r.json();
      if (j.status === "ok" && j.model_loaded) {
        setHealth("ok", `ready · gpu=${j.gpu_available ? "yes" : "no"}`);
      } else {
        setHealth("warn", "model not loaded");
      }
    } catch (e) {
      setHealth("bad", "offline");
    }
  }

  function setBusy(busy) {
    sendBtn.disabled = busy;
    stopBtn.disabled = !busy;
    inputEl.disabled = busy;
  }

  async function sendStreaming(payload, target) {
    const controller = new AbortController();
    activeController = controller;
    let acc = "";
    try {
      const resp = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

      const reader = resp.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        // SSE frames are separated by blank lines.
        let idx;
        while ((idx = buffer.indexOf("\n\n")) !== -1) {
          const frame = buffer.slice(0, idx);
          buffer = buffer.slice(idx + 2);
          for (const line of frame.split("\n")) {
            if (!line.startsWith("data:")) continue;
            const data = line.slice(5).trim();
            if (!data) continue;
            try {
              const evt = JSON.parse(data);
              if (evt.type === "delta" && typeof evt.text === "string") {
                acc += evt.text;
                target.textContent = acc;
                messagesEl.scrollTop = messagesEl.scrollHeight;
              } else if (evt.type === "end") {
                // optional: finish_reason in evt.finish_reason
              } else if (evt.type === "error") {
                throw new Error(evt.message || "stream error");
              }
            } catch (parseErr) {
              // Ignore non-JSON keepalives.
            }
          }
        }
      }
    } finally {
      activeController = null;
    }
    return acc;
  }

  async function sendNonStreaming(payload, target) {
    const controller = new AbortController();
    activeController = controller;
    try {
      const resp = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const j = await resp.json();
      const text = (j.message && j.message.content) || j.text || "";
      target.textContent = text;
      return text;
    } finally {
      activeController = null;
    }
  }

  async function onSubmit(ev) {
    ev.preventDefault();
    const text = inputEl.value.trim();
    if (!text) return;
    inputEl.value = "";
    autoSize(inputEl);

    history.push({ role: "user", content: text });
    renderMessage("user", text);
    const stream = streamEl.checked;
    const body = renderMessage("assistant", "", { streaming: true });
    setBusy(true);

    try {
      const out = stream
        ? await sendStreaming(buildPayload(true), body)
        : await sendNonStreaming(buildPayload(false), body);
      history.push({ role: "assistant", content: out });
      body.classList.remove("cursor");
    } catch (err) {
      body.classList.remove("cursor");
      body.textContent = `⚠ ${err.message || err}`;
      body.parentElement.classList.add("error");
    } finally {
      setBusy(false);
      inputEl.focus();
    }
  }

  inputEl.addEventListener("input", () => autoSize(inputEl));
  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      form.requestSubmit();
    }
  });
  form.addEventListener("submit", onSubmit);
  stopBtn.addEventListener("click", () => {
    if (activeController) activeController.abort();
  });
  newChatBtn.addEventListener("click", clearChat);
  tempEl.addEventListener("input", () => (tempV.textContent = parseFloat(tempEl.value).toFixed(2)));
  toppEl.addEventListener("input", () => (toppV.textContent = parseFloat(toppEl.value).toFixed(2)));

  // Initial state.
  tempV.textContent = parseFloat(tempEl.value).toFixed(2);
  toppV.textContent = parseFloat(toppEl.value).toFixed(2);
  checkHealth();
  setInterval(checkHealth, 15000);
  inputEl.focus();
})();

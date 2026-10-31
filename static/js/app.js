// CardioMultimodal AI - Client Application Logic

const LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"];
let currentEcgData = []; // [12, samples]
let currentSaliencyData = null; // [12, samples]
let currentLeadIdx = 1; // Default to Lead II (index 1)
let authToken = "";

// Initialize application on DOM ready
document.addEventListener("DOMContentLoaded", async () => {
  setupTabs();
  setupLeads();
  await authenticate();
  await loadModelStatus();
  await loadPreset(true); // default to high risk preset for illustration
  setupEventListeners();
  renderEcg();
});

// Setup tab navigation
function setupTabs() {
  const tabs = document.querySelectorAll(".nav-tab");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));
      tab.classList.add("active");
      const targetId = `tab-${tab.dataset.tab}`;
      const pane = document.getElementById(targetId);
      if (pane) pane.classList.add("active");

      if (tab.dataset.tab === "audit") {
        fetchAuditLogs();
      }
    });
  });
}

// Setup lead selector pills
function setupLeads() {
  const container = document.getElementById("leadPills");
  container.innerHTML = "";
  LEADS.forEach((lead, idx) => {
    const btn = document.createElement("button");
    btn.className = `lead-pill ${idx === currentLeadIdx ? "active" : ""}`;
    btn.innerText = lead;
    btn.addEventListener("click", () => {
      document.querySelectorAll(".lead-pill").forEach(p => p.classList.remove("active"));
      btn.classList.add("active");
      currentLeadIdx = idx;
      document.getElementById("ecgMetadataText").innerText = `Lead ${lead} | ${currentEcgData[idx] ? currentEcgData[idx].length : 2500} samples | 500 Hz`;
      renderEcg();
    });
    container.appendChild(btn);
  });
}

// Authenticate as default research client
async function authenticate() {
  try {
    const form = new URLSearchParams();
    form.append("username", "researcher@cardio.ai");
    form.append("password", "Research2026!");

    const res = await fetch("/v1/auth/token", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: form
    });
    if (res.ok) {
      const data = await res.json();
      authToken = data.access_token;
    }
  } catch (err) {
    console.warn("Auth token initialization error:", err);
  }
}

// Load Model Status
async function loadModelStatus() {
  try {
    const res = await fetch("/v1/model");
    if (res.ok) {
      const data = await res.json();
      const badge = document.getElementById("modelStatusBadge");
      const metaVer = document.getElementById("metaModelVer");
      const regBadge = document.getElementById("registryStatusBadge");
      const shaText = document.getElementById("registryShaText");

      if (data.model_configured) {
        badge.innerText = `Active: ${data.model_version}`;
        badge.className = "badge badge-live";
        if (metaVer) metaVer.innerText = data.model_version;
        if (regBadge) regBadge.innerText = "Active in Memory";
        if (shaText) shaText.innerText = data.artifact_sha256 || "Validated";
        if (data.threshold) {
          document.getElementById("thresholdSubtext").innerText = `Tuned Decision Threshold: ${data.threshold.toFixed(3)}`;
        }
      } else {
        badge.innerText = "Demo Mode (Model Unloaded)";
        badge.className = "badge badge-ruo";
        if (regBadge) regBadge.innerText = "No Model Loaded";
        if (shaText) shaText.innerText = "N/A (Run train.py)";
      }
    }
  } catch (e) {
    console.error("Failed loading model status:", e);
  }
}

// Preset Loader
async function loadPreset(isHighRisk) {
  try {
    const res = await fetch(`/v1/sample-record?pathological=${isHighRisk}`);
    if (res.ok) {
      const data = await res.json();
      currentEcgData = data.ecg;
      currentSaliencyData = null; // reset saliency
      applyClinicalFeatures(data.clinical);
      renderEcg();
    }
  } catch (err) {
    console.error("Failed to load preset:", err);
  }
}

function applyClinicalFeatures(clin) {
  for (const [key, val] of Object.entries(clin)) {
    const el = document.getElementById(`feat_${key}`);
    if (el) el.value = val !== null ? val : "";
  }
}

function getClinicalPayload() {
  const fields = [
    "age", "sex", "resting_bp", "cholesterol", "fasting_bs",
    "resting_ecg", "max_hr", "exercise_angina", "st_depression",
    "st_slope", "num_major_vessels", "thal"
  ];
  const clinical = {};
  fields.forEach(f => {
    const el = document.getElementById(`feat_${f}`);
    clinical[f] = el && el.value !== "" ? parseFloat(el.value) : null;
  });
  return clinical;
}

// Setup Event Listeners
function setupEventListeners() {
  document.getElementById("presetHighRisk").addEventListener("click", () => loadPreset(true));
  document.getElementById("presetNormal").addEventListener("click", () => loadPreset(false));
  document.getElementById("presetRandom").addEventListener("click", () => loadPreset(Math.random() > 0.5));
  document.getElementById("toggleSaliency").addEventListener("change", () => renderEcg());
  document.getElementById("btnPredict").addEventListener("click", runPredictionAndExplain);
  document.getElementById("btnRefreshAudit").addEventListener("click", fetchAuditLogs);
}

// Run Multimodal Prediction and Saliency
async function runPredictionAndExplain() {
  const btn = document.getElementById("btnPredict");
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span> Running Deep Inference...`;

  try {
    const clinical = getClinicalPayload();
    const payload = {
      record_id: "SANDBOX_REC_01",
      sampling_rate_hz: 500.0,
      ecg: currentEcgData,
      clinical: clinical
    };

    const headers = { "Content-Type": "application/json" };
    if (authToken) headers["Authorization"] = `Bearer ${authToken}`;

    // 1. Predict
    const predRes = await fetch("/v1/predict", {
      method: "POST",
      headers: headers,
      body: JSON.stringify(payload)
    });

    if (!predRes.ok) {
      const err = await predRes.json();
      alert(`Prediction Service Notice: ${err.detail || "Error"}`);
      return;
    }

    const predData = await predRes.json();

    // 2. Explain
    const explainRes = await fetch("/v1/explain", {
      method: "POST",
      headers: headers,
      body: JSON.stringify({ ...payload, steps: 8 })
    });

    let explainData = null;
    if (explainRes.ok) {
      explainData = await explainRes.json();
      currentSaliencyData = explainData.saliency_preview;
    }

    displayPredictionResults(predData, explainData);
    renderEcg();

  } catch (err) {
    console.error("Error executing prediction:", err);
    alert("Inference failed: Check server connection.");
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg> Run Multimodal Prediction &amp; Saliency`;
  }
}

// Display Prediction Results
function displayPredictionResults(pred, explain) {
  const scoreNum = document.getElementById("scoreNumber");
  const scoreCircle = document.getElementById("scoreCircle");
  const predTag = document.getElementById("predictionTag");
  const metaLatency = document.getElementById("metaLatency");
  const metaReqId = document.getElementById("metaRequestId");
  const metaModelVer = document.getElementById("metaModelVer");

  const score = pred.research_score;
  scoreNum.innerText = (score * 100).toFixed(1) + "%";
  metaLatency.innerText = `${pred.latency_ms} ms`;
  metaReqId.innerText = pred.request_id;
  metaModelVer.innerText = pred.model_version;

  // Color circle based on risk
  if (score >= pred.decision_threshold) {
    scoreCircle.style.borderColor = "var(--accent-red)";
    scoreCircle.style.boxShadow = "0 0 25px rgba(239, 68, 68, 0.35)";
    predTag.innerText = "Elevated Risk Indication";
    predTag.style.color = "var(--accent-red)";
  } else {
    scoreCircle.style.borderColor = "var(--accent-green)";
    scoreCircle.style.boxShadow = "0 0 25px rgba(16, 185, 129, 0.35)";
    predTag.innerText = "Baseline / Low Risk";
    predTag.style.color = "var(--accent-green)";
  }

  // Modality weights
  const ecgW = Math.round((pred.modality_contributions.ecg_weight || 0.5) * 100);
  const clinW = 100 - ecgW;
  document.getElementById("ecgWeightText").innerText = `${ecgW}%`;
  document.getElementById("clinicalWeightText").innerText = `${clinW}%`;
  document.getElementById("ecgMeterFill").style.width = `${ecgW}%`;
  document.getElementById("clinicalMeterFill").style.width = `${clinW}%`;

  // Explainability UI
  if (explain) {
    // Lead rankings
    const leadPillsContainer = document.getElementById("leadRankingPills");
    leadPillsContainer.innerHTML = "";
    explain.lead_ranking.slice(0, 5).forEach(item => {
      const pill = document.createElement("span");
      pill.className = "badge";
      pill.style.background = "rgba(6, 182, 212, 0.15)";
      pill.style.color = "#67e8f9";
      pill.style.border = "1px solid rgba(6, 182, 212, 0.3)";
      pill.innerText = `${item.lead}: ${(item.weight * 100).toFixed(1)}%`;
      leadPillsContainer.appendChild(pill);
    });

    // Clinical feature attributions
    const attrList = document.getElementById("attrList");
    attrList.innerHTML = "";
    explain.clinical_attributions.forEach(item => {
      const isPos = item.direction === "increases_risk";
      const div = document.createElement("div");
      div.className = "attr-item";
      div.innerHTML = `
        <span style="font-weight: 500; min-width: 90px;">${item.feature}</span>
        <div class="attr-bar-wrapper">
          <div class="attr-bar-fill ${isPos ? 'pos' : 'neg'}" style="width: ${Math.min(100, item.attribution_score * 300)}%;"></div>
        </div>
        <span style="font-size: 0.7rem; color: ${isPos ? 'var(--accent-red)' : 'var(--accent-green)'}">
          ${isPos ? '+' : '-'}${(item.attribution_score * 100).toFixed(1)}%
        </span>
      `;
      attrList.appendChild(div);
    });
  }
}

// Render ECG Waveform with optional Saliency
function renderEcg() {
  const canvas = document.getElementById("ecgCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  // Handle high-DPI displays
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  const w = rect.width;
  const h = rect.height;

  // 1. Clear background
  ctx.fillStyle = "#050811";
  ctx.fillRect(0, 0, w, h);

  // 2. Draw ECG Medical Grid Lines (1mm and 5mm grid)
  const gridSmall = 12;
  const gridLarge = 60;

  ctx.strokeStyle = "rgba(6, 182, 212, 0.05)";
  ctx.lineWidth = 1;
  for (let x = 0; x < w; x += gridSmall) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
  }
  for (let y = 0; y < h; y += gridSmall) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  ctx.strokeStyle = "rgba(6, 182, 212, 0.12)";
  for (let x = 0; x < w; x += gridLarge) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
  }
  for (let y = 0; y < h; y += gridLarge) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  // 3. Draw Waveform
  const leadSignal = currentEcgData[currentLeadIdx];
  if (!leadSignal || leadSignal.length === 0) return;

  const numSamples = leadSignal.length;
  const centerY = h / 2;
  const scaleY = h * 0.28; // gain

  const showSaliency = document.getElementById("toggleSaliency")?.checked && currentSaliencyData;
  const leadSaliency = showSaliency && currentSaliencyData[currentLeadIdx] ? currentSaliencyData[currentLeadIdx] : null;

  // Draw Saliency Glow if active
  if (leadSaliency) {
    ctx.lineWidth = 4;
    for (let i = 0; i < numSamples - 1; i++) {
      const salIdx = Math.floor(i / 10);
      const salVal = leadSaliency[salIdx] || 0;
      if (salVal > 0.35) {
        const x1 = (i / numSamples) * w;
        const y1 = centerY - (leadSignal[i] * scaleY);
        const x2 = ((i + 1) / numSamples) * w;
        const y2 = centerY - (leadSignal[i + 1] * scaleY);

        ctx.strokeStyle = `rgba(239, 68, 68, ${Math.min(0.85, salVal * 0.9)})`;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
      }
    }
  }

  // Draw Primary ECG Trace
  ctx.strokeStyle = "#00f0ff";
  ctx.lineWidth = 2.0;
  ctx.shadowColor = "rgba(0, 240, 255, 0.55)";
  ctx.shadowBlur = 6;
  ctx.beginPath();

  for (let i = 0; i < numSamples; i++) {
    const x = (i / numSamples) * w;
    const y = centerY - (leadSignal[i] * scaleY);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.shadowBlur = 0; // reset
}

// Fetch and render Audit Logs
async function fetchAuditLogs() {
  const tbody = document.getElementById("auditTableBody");
  if (!tbody) return;
  tbody.innerHTML = `<tr><td colspan="5" style="text-align: center;"><span class="spinner"></span> Loading audit telemetry...</td></tr>`;

  try {
    const headers = {};
    if (authToken) headers["Authorization"] = `Bearer ${authToken}`;
    const res = await fetch("/v1/audit?limit=25", { headers });
    if (!res.ok) {
      tbody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--text-sub);">Audit access requires admin privilege. Current role: researcher.</td></tr>`;
      return;
    }
    const logs = await res.json();
    if (logs.length === 0) {
      tbody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--text-sub);">No audit logs recorded yet.</td></tr>`;
      return;
    }

    tbody.innerHTML = "";
    logs.forEach(l => {
      const tr = document.createElement("tr");
      const timeStr = l.created_at ? new Date(l.created_at).toLocaleTimeString() : "--";
      tr.innerHTML = `
        <td><span style="font-family: monospace; font-size: 0.75rem;">${timeStr}</span></td>
        <td><strong style="color: var(--accent-cyan);">${l.action}</strong></td>
        <td>${l.actor_username || "anonymous"}</td>
        <td><code style="font-size: 0.7rem; color: var(--text-sub);">${l.request_id || "N/A"}</code></td>
        <td>${l.resource_type || "api"}</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--accent-red);">Failed loading logs.</td></tr>`;
  }
}

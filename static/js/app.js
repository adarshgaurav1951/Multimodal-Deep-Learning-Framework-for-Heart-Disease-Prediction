// CardioMultimodal AI - Client Application Logic & Interactive Visualizations

const LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"];
let currentEcgData = []; // [12, samples]
let currentSaliencyData = null; // [12, samples]
let currentLeadIdx = 1; // Default to Lead II (index 1)
let authToken = "";

// Visualization State
let benchmarkData = null;
let rocData = null;
let trainingHistoryData = null;
let activeMetric = "accuracy";
let activeDynMode = "acc";
let mouseRocPos = null;

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
      } else if (tab.dataset.tab === "visualizations") {
        loadVisualizationData();
      }
    });
  });
}

// Setup lead selector pills
function setupLeads() {
  const container = document.getElementById("leadPills");
  if (!container) return;
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
      }
    }
  } catch (err) {
    console.warn("Model status check failed:", err);
  }
}

// Load Sample Record / Presets
async function loadPreset(pathological = true) {
  try {
    const res = await fetch(`/v1/sample-record?pathological=${pathological}`);
    if (res.ok) {
      const data = await res.json();
      currentEcgData = data.ecg;
      currentSaliencyData = null; // reset saliency on new preset

      // Populate clinical inputs
      for (const [key, val] of Object.entries(data.clinical)) {
        const input = document.getElementById(`feat_${key}`);
        if (input && val !== null && val !== undefined) {
          input.value = val;
        }
      }

      // Update metadata text
      document.getElementById("ecgMetadataText").innerText = `Lead ${LEADS[currentLeadIdx]} | ${currentEcgData[currentLeadIdx] ? currentEcgData[currentLeadIdx].length : 2500} samples | 500 Hz`;
      renderEcg();
    }
  } catch (err) {
    console.error("Failed to load sample record preset:", err);
  }
}

// Extract Clinical Form values
function getClinicalPayload() {
  const clinical = {};
  const fields = [
    "age", "sex", "resting_bp", "cholesterol", "fasting_bs",
    "resting_ecg", "max_hr", "exercise_angina", "st_depression",
    "st_slope", "num_major_vessels", "thal"
  ];

  fields.forEach(f => {
    const el = document.getElementById(`feat_${f}`);
    if (el) {
      const v = parseFloat(el.value);
      clinical[f] = isNaN(v) ? 0.0 : v;
    }
  });

  return clinical;
}

// Setup Event Listeners
function setupEventListeners() {
  document.getElementById("presetHighRisk")?.addEventListener("click", () => loadPreset(true));
  document.getElementById("presetNormal")?.addEventListener("click", () => loadPreset(false));
  document.getElementById("presetRandom")?.addEventListener("click", () => loadPreset(Math.random() > 0.5));
  document.getElementById("toggleSaliency")?.addEventListener("change", () => renderEcg());
  document.getElementById("btnPredict")?.addEventListener("click", runPredictionAndExplain);
  document.getElementById("btnRefreshAudit")?.addEventListener("click", fetchAuditLogs);
  document.getElementById("btnRefreshVisualizations")?.addEventListener("click", loadVisualizationData);

  // Metric filter pills in visualizations
  document.querySelectorAll("#metricFilterPills .chart-pill").forEach(pill => {
    pill.addEventListener("click", () => {
      document.querySelectorAll("#metricFilterPills .chart-pill").forEach(p => p.classList.remove("active"));
      pill.classList.add("active");
      activeMetric = pill.dataset.metric;
      renderAccuracyChart(activeMetric);
    });
  });

  // Dynamics mode pills (Accuracy vs Loss)
  document.querySelectorAll("#dynModePills .chart-pill").forEach(pill => {
    pill.addEventListener("click", () => {
      document.querySelectorAll("#dynModePills .chart-pill").forEach(p => p.classList.remove("active"));
      pill.classList.add("active");
      activeDynMode = pill.dataset.mode;
      renderTrainingDynamicsChart(activeDynMode);
    });
  });

  // Modal close handlers
  document.getElementById("modalCloseBtn")?.addEventListener("click", closeModal);
  document.getElementById("imageModal")?.addEventListener("click", (e) => {
    if (e.target.id === "imageModal") closeModal();
  });

  // Interactive mouse tracking on ROC Canvas
  const rocCanvas = document.getElementById("rocComparisonCanvas");
  if (rocCanvas) {
    rocCanvas.addEventListener("mousemove", (e) => {
      const rect = rocCanvas.getBoundingClientRect();
      mouseRocPos = { x: e.clientX - rect.left, y: e.clientY - rect.top };
      renderRocComparisonChart();
    });
    rocCanvas.addEventListener("mouseleave", () => {
      mouseRocPos = null;
      renderRocComparisonChart();
    });
  }

  // Responsive redraw on window resize
  let resizeTimer = null;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      renderEcg();
      if (benchmarkData) renderAccuracyChart(activeMetric);
      if (rocData) renderRocComparisonChart();
      if (trainingHistoryData) renderTrainingDynamicsChart(activeDynMode);
    }, 150);
  });
}

// Run Multimodal Prediction and Saliency
async function runPredictionAndExplain() {
  const btn = document.getElementById("btnPredict");
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span> Running Deep Inference...`;

  try {
    const clinical = getClinicalPayload();
    const arch = document.getElementById("modelArchSelect")?.value || "multimodal_gated";
    const payload = {
      record_id: "SANDBOX_REC_01",
      sampling_rate_hz: 500.0,
      ecg: currentEcgData,
      clinical: clinical,
      architecture: arch
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
  metaModelVer.innerText = pred.architecture ? `${pred.model_version} (${pred.architecture})` : pred.model_version;

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

  // Saliency Heatmap Underlay
  if (leadSaliency && leadSaliency.length > 0) {
    const sLen = leadSaliency.length;
    for (let i = 0; i < sLen; i++) {
      const salVal = leadSaliency[i];
      if (salVal > 0.08) {
        const xStart = (i / sLen) * w;
        const xEnd = ((i + 1) / sLen) * w;
        ctx.fillStyle = `rgba(239, 68, 68, ${salVal * 0.35})`;
        ctx.fillRect(xStart, 0, xEnd - xStart, h);
      }
    }
  }

  // Waveform line
  ctx.beginPath();
  ctx.strokeStyle = "#38bdf8";
  ctx.lineWidth = 1.8;
  ctx.shadowColor = "rgba(56, 189, 248, 0.6)";
  ctx.shadowBlur = 4;

  for (let i = 0; i < numSamples; i++) {
    const x = (i / numSamples) * w;
    const y = centerY - (leadSignal[i] * scaleY);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.shadowBlur = 0; // reset
}

// ==========================================
// VISUALIZATION & ACCURACY DASHBOARD ENGINE
// ==========================================

async function loadVisualizationData() {
  try {
    // 1. Fetch Benchmarks
    const bRes = await fetch("/v1/models/benchmark");
    if (bRes.ok) {
      benchmarkData = await bRes.json();
      updateKpis(benchmarkData);
      renderAccuracyChart(activeMetric);
    }

    // 2. Fetch ROC Data
    const rRes = await fetch("/v1/models/roc-auc-data");
    if (rRes.ok) {
      rocData = await rRes.json();
      renderRocComparisonChart();
    }

    // 3. Fetch Training Dynamics
    const tRes = await fetch("/v1/models/training-history");
    if (tRes.ok) {
      trainingHistoryData = await tRes.json();
      renderTrainingDynamicsChart(activeDynMode);
    }

    // 4. Fetch Figures Gallery
    const fRes = await fetch("/v1/visualizations/list");
    if (fRes.ok) {
      const figures = await fRes.json();
      renderFiguresGallery(figures);
    }
  } catch (err) {
    console.error("Failed to load visualization data:", err);
  }
}

// Update Top KPI Cards
function updateKpis(data) {
  if (!data || !data.metrics) return;
  const metrics = data.metrics;
  const mm = metrics["Multimodal Gated Fusion"] || metrics["Multimodal (Gated CNN-BiLSTM)"] || {};

  if (mm.accuracy) {
    document.getElementById("kpiAccuracyVal").innerText = `${(mm.accuracy * 100).toFixed(1)}%`;
  }
  if (mm.auroc) {
    document.getElementById("kpiAurocVal").innerText = mm.auroc.toFixed(3);
  }
  if (mm.f1 && mm.sensitivity) {
    document.getElementById("kpiF1Val").innerText = `${mm.f1.toFixed(3)} / ${(mm.sensitivity * 100).toFixed(1)}%`;
  }
}

// 1. Render Interactive Accuracy & Metrics Bar Chart (Vertical)
function renderAccuracyChart(metricKey = "accuracy") {
  const canvas = document.getElementById("accuracyChartCanvas");
  if (!canvas || !benchmarkData || !benchmarkData.metrics) return;

  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  const w = rect.width;
  const h = rect.height;

  // Clear canvas
  ctx.fillStyle = "#070b14";
  ctx.fillRect(0, 0, w, h);

  const models = Object.keys(benchmarkData.metrics);
  const isPercent = ["accuracy", "sensitivity", "specificity", "f1"].includes(metricKey);
  const values = models.map(m => {
    const val = benchmarkData.metrics[m][metricKey] || 0.0;
    return isPercent ? val * 100 : val;
  });

  const padLeft = 45;
  const padRight = 20;
  const padTop = 32;
  const padBottom = 75;
  const chartW = w - padLeft - padRight;
  const chartH = h - padTop - padBottom;

  const maxVal = isPercent ? 100 : (metricKey.includes("brier") ? 0.2 : 1.0);
  const minVal = isPercent ? 50 : 0.0;

  // Draw Horizontal Grid Lines & Y-axis scale
  ctx.strokeStyle = "rgba(255, 255, 255, 0.06)";
  ctx.lineWidth = 1;
  ctx.fillStyle = "rgba(255, 255, 255, 0.45)";
  ctx.font = "10px Inter, sans-serif";
  ctx.textAlign = "right";

  const numGrid = 5;
  for (let i = 0; i <= numGrid; i++) {
    const val = minVal + (i / numGrid) * (maxVal - minVal);
    const y = padTop + chartH - (i / numGrid) * chartH;
    ctx.beginPath();
    ctx.moveTo(padLeft, y);
    ctx.lineTo(w - padRight, y);
    ctx.stroke();
    ctx.fillText(isPercent ? `${val.toFixed(0)}%` : val.toFixed(2), padLeft - 8, y + 3.5);
  }

  // Draw Vertical Multi-Colored Bars
  const numModels = models.length;
  const colW = chartW / numModels;
  const barW = Math.min(46, colW * 0.62);

  models.forEach((name, idx) => {
    const val = values[idx];
    const norm = Math.max(0, Math.min(1, (val - minVal) / (maxVal - minVal)));
    const barH = norm * chartH;
    const centerX = padLeft + idx * colW + colW / 2;
    const barX = centerX - barW / 2;
    const barY = padTop + chartH - barH;

    // Vertical Gradient for distinct colors
    const grad = ctx.createLinearGradient(0, barY, 0, barY + barH);
    let topColor = "#3b82f6";
    let botColor = "#1d4ed8";
    let glowColor = "transparent";

    if (name.includes("Gated")) {
      topColor = "#10b981";
      botColor = "#059669";
      glowColor = "rgba(16, 185, 129, 0.35)";
    } else if (name.includes("Cross-Attention")) {
      topColor = "#06b6d4";
      botColor = "#0284c7";
      glowColor = "rgba(6, 182, 212, 0.35)";
    } else if (name.includes("CNN-Transformer") || name.includes("Hybrid")) {
      topColor = "#8b5cf6";
      botColor = "#6d28d9";
    } else if (name.includes("Transformer")) {
      topColor = "#3b82f6";
      botColor = "#1d4ed8";
    } else if (name.includes("CNN-BiLSTM")) {
      topColor = "#f59e0b";
      botColor = "#d97706";
    } else {
      topColor = "#ef4444";
      botColor = "#b91c1c";
    }

    grad.addColorStop(0, topColor);
    grad.addColorStop(1, botColor);

    // Glow for top multimodal architectures
    if (glowColor !== "transparent") {
      ctx.shadowColor = glowColor;
      ctx.shadowBlur = 8;
    }

    // Rounded rectangle vertical bar (rounded top corners)
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.roundRect(barX, barY, barW, Math.max(3, barH), [4, 4, 0, 0]);
    ctx.fill();
    ctx.shadowBlur = 0; // reset shadow

    // Value badge above vertical bar
    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 11px Inter, sans-serif";
    ctx.textAlign = "center";
    const labelStr = isPercent ? `${val.toFixed(1)}%` : val.toFixed(3);
    ctx.fillText(labelStr, centerX, Math.max(padTop - 8, barY - 7));

    // Model name below x-axis (angled at -28 degrees)
    ctx.save();
    ctx.translate(centerX, padTop + chartH + 12);
    ctx.rotate(-Math.PI / 6.4);
    ctx.textAlign = "right";
    ctx.fillStyle = name.includes("Multimodal") ? "#67e8f9" : "#94a3b8";
    ctx.font = name.includes("Multimodal") ? "bold 10.5px Inter, sans-serif" : "10px Inter, sans-serif";
    ctx.fillText(name, 0, 0);
    ctx.restore();
  });
}

// 2. Render Interactive Multi-Model ROC-AUC Curve Canvas
function renderRocComparisonChart() {
  const canvas = document.getElementById("rocComparisonCanvas");
  if (!canvas || !rocData) return;

  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  const w = rect.width;
  const h = rect.height;

  // Clear
  ctx.fillStyle = "#070b14";
  ctx.fillRect(0, 0, w, h);

  const padLeft = 45;
  const padRight = 145; // space for legend
  const padTop = 20;
  const padBottom = 35;
  const chartW = w - padLeft - padRight;
  const chartH = h - padTop - padBottom;

  // Grid lines
  ctx.strokeStyle = "rgba(255, 255, 255, 0.06)";
  ctx.lineWidth = 1;
  ctx.fillStyle = "rgba(255, 255, 255, 0.4)";
  ctx.font = "10px Inter, sans-serif";
  ctx.textAlign = "center";

  for (let i = 0; i <= 5; i++) {
    const val = (i / 5).toFixed(1);
    const x = padLeft + (i / 5) * chartW;
    const y = padTop + chartH - (i / 5) * chartH;

    // vertical
    ctx.beginPath();
    ctx.moveTo(x, padTop);
    ctx.lineTo(x, padTop + chartH);
    ctx.stroke();
    ctx.fillText(val, x, padTop + chartH + 16);

    // horizontal
    ctx.beginPath();
    ctx.moveTo(padLeft, y);
    ctx.lineTo(padLeft + chartW, y);
    ctx.stroke();
    ctx.textAlign = "right";
    ctx.fillText(val, padLeft - 8, y + 3);
    ctx.textAlign = "center";
  }

  // Axis Labels
  ctx.fillStyle = "rgba(255, 255, 255, 0.6)";
  ctx.font = "10.5px Inter, sans-serif";
  ctx.fillText("False Positive Rate (1 - Specificity)", padLeft + chartW / 2, h - 6);

  ctx.save();
  ctx.translate(14, padTop + chartH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("True Positive Rate (Sensitivity)", 0, 0);
  ctx.restore();

  // Chance baseline diagonal
  ctx.strokeStyle = "rgba(148, 163, 184, 0.4)";
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(padLeft, padTop + chartH);
  ctx.lineTo(padLeft + chartW, padTop);
  ctx.stroke();
  ctx.setLineDash([]);

  // Plot Each Model Curve
  let legendY = padTop + 10;
  let hoveredPoint = null;

  for (const [name, r] of Object.entries(rocData)) {
    const fpr = r.fpr;
    const tpr = r.tpr;
    const color = r.color || "#3b82f6";
    const isMultimodal = name.includes("Multimodal");

    ctx.strokeStyle = color;
    ctx.lineWidth = isMultimodal ? 2.6 : 1.6;
    ctx.beginPath();

    for (let i = 0; i < fpr.length; i++) {
      const px = padLeft + fpr[i] * chartW;
      const py = padTop + chartH - (tpr[i] * chartH);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);

      // Check hover proximity
      if (mouseRocPos) {
        const dist = Math.hypot(mouseRocPos.x - px, mouseRocPos.y - py);
        if (dist < 12 && (!hoveredPoint || dist < hoveredPoint.dist)) {
          hoveredPoint = { name, auroc: r.auroc, fpr: fpr[i], tpr: tpr[i], px, py, color, dist };
        }
      }
    }
    ctx.stroke();

    // Optimal Threshold Dot
    const dotIdx = Math.min(fpr.length - 1, Math.floor(fpr.length * 0.16));
    const dotX = padLeft + fpr[dotIdx] * chartW;
    const dotY = padTop + chartH - (tpr[dotIdx] * chartH);

    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(dotX, dotY, 4, 0, Math.PI * 2);
    ctx.fill();

    // Draw Legend Item
    ctx.fillStyle = color;
    ctx.fillRect(w - padRight + 12, legendY - 6, 12, 10);
    ctx.fillStyle = "#f8fafc";
    ctx.font = isMultimodal ? "bold 10px Inter, sans-serif" : "9.5px Inter, sans-serif";
    ctx.textAlign = "left";
    const shortName = name.replace("Multimodal", "MM").replace("Baseline", "").trim();
    ctx.fillText(`${shortName} (${r.auroc.toFixed(3)})`, w - padRight + 30, legendY + 2);
    legendY += 18;
  }

  // Draw Hover Tooltip if hovering over a curve
  if (hoveredPoint) {
    // Crosshair lines
    ctx.strokeStyle = "rgba(255, 255, 255, 0.35)";
    ctx.setLineDash([2, 2]);
    ctx.beginPath();
    ctx.moveTo(hoveredPoint.px, padTop);
    ctx.lineTo(hoveredPoint.px, padTop + chartH);
    ctx.moveTo(padLeft, hoveredPoint.py);
    ctx.lineTo(padLeft + chartW, hoveredPoint.py);
    ctx.stroke();
    ctx.setLineDash([]);

    // Highlight dot
    ctx.fillStyle = "#fff";
    ctx.beginPath();
    ctx.arc(hoveredPoint.px, hoveredPoint.py, 6, 0, Math.PI * 2);
    ctx.fill();

    // Tooltip box
    const tipX = Math.min(w - 180, Math.max(padLeft + 10, hoveredPoint.px + 10));
    const tipY = Math.max(padTop + 10, hoveredPoint.py - 45);
    ctx.fillStyle = "rgba(17, 24, 39, 0.95)";
    ctx.strokeStyle = hoveredPoint.color;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(tipX, tipY, 160, 48, [6]);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = "#fff";
    ctx.font = "bold 10.5px Inter, sans-serif";
    ctx.fillText(hoveredPoint.name, tipX + 8, tipY + 16);
    ctx.fillStyle = "var(--text-muted)";
    ctx.font = "9.5px Inter, sans-serif";
    ctx.fillText(`AUROC: ${hoveredPoint.auroc.toFixed(3)} | FPR: ${hoveredPoint.fpr.toFixed(2)}`, tipX + 8, tipY + 30);
    ctx.fillText(`TPR (Sensitivity): ${(hoveredPoint.tpr * 100).toFixed(1)}%`, tipX + 8, tipY + 42);
  }
}

// 3. Render Training & Validation Dynamics Chart
function renderTrainingDynamicsChart(mode = "acc") {
  const canvas = document.getElementById("trainingDynamicsCanvas");
  if (!canvas || !trainingHistoryData) return;

  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  const w = rect.width;
  const h = rect.height;

  ctx.fillStyle = "#070b14";
  ctx.fillRect(0, 0, w, h);

  const padLeft = 50;
  const padRight = 170;
  const padTop = 25;
  const padBottom = 35;
  const chartW = w - padLeft - padRight;
  const chartH = h - padTop - padBottom;

  const isAcc = mode === "acc";
  const minVal = isAcc ? 50 : 0.05;
  const maxVal = isAcc ? 100 : 0.8;

  // Grid
  ctx.strokeStyle = "rgba(255, 255, 255, 0.06)";
  ctx.lineWidth = 1;
  ctx.fillStyle = "rgba(255, 255, 255, 0.4)";
  ctx.font = "10px Inter, sans-serif";
  ctx.textAlign = "center";

  const numEpochs = 15;
  for (let e = 1; e <= numEpochs; e++) {
    const x = padLeft + ((e - 1) / (numEpochs - 1)) * chartW;
    ctx.beginPath();
    ctx.moveTo(x, padTop);
    ctx.lineTo(x, padTop + chartH);
    ctx.stroke();
    if (e % 2 !== 0 || e === numEpochs) {
      ctx.fillText(`Ep ${e}`, x, padTop + chartH + 16);
    }
  }

  for (let i = 0; i <= 4; i++) {
    const val = minVal + (i / 4) * (maxVal - minVal);
    const y = padTop + chartH - (i / 4) * chartH;
    ctx.beginPath();
    ctx.moveTo(padLeft, y);
    ctx.lineTo(padLeft + chartW, y);
    ctx.stroke();
    ctx.textAlign = "right";
    ctx.fillText(isAcc ? `${val.toFixed(0)}%` : val.toFixed(2), padLeft - 8, y + 4);
  }

  // Draw curves for models
  const modelColors = {
    "Multimodal Gated Fusion": "#10b981",
    "Multimodal Gated": "#10b981",
    "ECG Transformer": "#3b82f6",
    "ECG CNN-BiLSTM": "#f59e0b"
  };

  let legY = padTop + 8;
  for (const [name, hData] of Object.entries(trainingHistoryData)) {
    const color = modelColors[name] || "#8b5cf6";
    const trainSeries = isAcc ? hData.train_acc : hData.train_loss;
    const valSeries = isAcc ? hData.val_acc : hData.val_loss;
    if (!trainSeries || trainSeries.length === 0) continue;

    // Draw Train (Solid)
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    for (let i = 0; i < trainSeries.length; i++) {
      let v = trainSeries[i];
      if (isAcc && v <= 1.0) v *= 100;
      const x = padLeft + (i / (numEpochs - 1)) * chartW;
      const y = padTop + chartH - ((v - minVal) / (maxVal - minVal)) * chartH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Draw Val (Dashed)
    ctx.setLineDash([4, 3]);
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    for (let i = 0; i < valSeries.length; i++) {
      let v = valSeries[i];
      if (isAcc && v <= 1.0) v *= 100;
      const x = padLeft + (i / (numEpochs - 1)) * chartW;
      const y = padTop + chartH - ((v - minVal) / (maxVal - minVal)) * chartH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.setLineDash([]);

    // Legend
    ctx.fillStyle = color;
    ctx.fillRect(w - padRight + 12, legY - 5, 12, 10);
    ctx.fillStyle = "#fff";
    ctx.font = "bold 9.5px Inter, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(`${name}`, w - padRight + 30, legY + 2);

    ctx.fillStyle = "rgba(255, 255, 255, 0.4)";
    ctx.font = "8.5px Inter, sans-serif";
    ctx.fillText("— Train  --- Val", w - padRight + 30, legY + 14);

    legY += 28;
  }
}

// 4. Render Publication Figures Gallery
function renderFiguresGallery(figures) {
  const container = document.getElementById("figuresGalleryGrid");
  if (!container || !figures) return;
  container.innerHTML = "";

  figures.forEach(fig => {
    const card = document.createElement("div");
    card.className = "figure-card";
    card.innerHTML = `
      <img class="figure-thumb" src="${fig.url}" alt="${fig.title}" onerror="this.src='/static/css/placeholder.png'; this.style.opacity=0.3;">
      <div class="figure-info">
        <h4>${fig.title}</h4>
        <p>${fig.description}</p>
        <span style="font-size: 0.68rem; color: var(--accent-cyan); margin-top: 6px; display: inline-block;">Click to inspect high-res &rarr;</span>
      </div>
    `;
    card.addEventListener("click", () => openModal(fig.url, fig.title, fig.description));
    container.appendChild(card);
  });
}

// Modal open/close
function openModal(imgUrl, title, desc) {
  const modal = document.getElementById("imageModal");
  if (!modal) return;
  document.getElementById("modalFigureImg").src = imgUrl;
  document.getElementById("modalFigureTitle").innerText = title;
  document.getElementById("modalFigureDesc").innerText = desc;
  modal.classList.add("open");
}

function closeModal() {
  const modal = document.getElementById("imageModal");
  if (modal) modal.classList.remove("open");
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

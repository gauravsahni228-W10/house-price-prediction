/**
 * HomeValue AI - Frontend Application Logic
 * Connects UI to FastAPI backend (/predict, /metrics, /health)
 */

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const predictForm = document.getElementById("predict-form");
  const modelInput = document.getElementById("model");
  const selectRfBtn = document.getElementById("select-rf-btn");
  const selectLinearBtn = document.getElementById("select-linear-btn");

  const qualitySlider = document.getElementById("overall_quality");
  const qualityBadge = document.getElementById("quality-badge");

  const sampleDataBtn = document.getElementById("sample-data-btn");
  const resetBtn = document.getElementById("reset-btn");
  const predictBtn = document.getElementById("predict-btn");
  const btnText = document.getElementById("btn-text");
  const btnSpinner = document.getElementById("btn-spinner");

  // Result card states
  const resultEmpty = document.getElementById("result-empty");
  const resultLoading = document.getElementById("result-loading");
  const resultContent = document.getElementById("result-content");
  const resultError = document.getElementById("result-error");
  const errorMessage = document.getElementById("error-message");

  const predictedPriceEl = document.getElementById("predicted-price");
  const priceRangeEl = document.getElementById("price-range");
  const rangeLowValEl = document.getElementById("range-low-val");
  const targetValEl = document.getElementById("target-val");
  const rangeHighValEl = document.getElementById("range-high-val");
  const modelBadgeEl = document.getElementById("model-badge");
  const summaryPillsEl = document.getElementById("summary-pills");

  // Quality Slider descriptions
  const qualityLabels = {
    1: "Very Poor",
    2: "Poor",
    3: "Fair",
    4: "Below Average",
    5: "Average",
    6: "Above Average",
    7: "Good",
    8: "Very Good",
    9: "Excellent",
    10: "Very Excellent"
  };

  qualitySlider.addEventListener("input", (e) => {
    const val = parseInt(e.target.value, 10);
    const label = qualityLabels[val] || "Standard";
    qualityBadge.textContent = `Rating: ${val} / 10 (${label})`;
  });

  // Model Selection Toggle
  function setModel(selected) {
    modelInput.value = selected;
    if (selected === "random_forest") {
      selectRfBtn.className = "flex-1 py-2 px-3 text-sm font-semibold rounded-lg transition bg-white text-brand-700 shadow-sm flex items-center justify-center gap-2";
      selectLinearBtn.className = "flex-1 py-2 px-3 text-sm font-semibold rounded-lg transition text-slate-600 hover:text-slate-900 flex items-center justify-center gap-2";
      modelBadgeEl.textContent = "Random Forest";
      modelBadgeEl.className = "px-2.5 py-0.5 rounded-full text-xs font-semibold bg-brand-50 text-brand-700 border border-brand-200";
    } else {
      selectLinearBtn.className = "flex-1 py-2 px-3 text-sm font-semibold rounded-lg transition bg-white text-brand-700 shadow-sm flex items-center justify-center gap-2";
      selectRfBtn.className = "flex-1 py-2 px-3 text-sm font-semibold rounded-lg transition text-slate-600 hover:text-slate-900 flex items-center justify-center gap-2";
      modelBadgeEl.textContent = "Linear Regression";
      modelBadgeEl.className = "px-2.5 py-0.5 rounded-full text-xs font-semibold bg-slate-100 text-slate-800 border border-slate-300";
    }
  }

  selectRfBtn.addEventListener("click", () => setModel("random_forest"));
  selectLinearBtn.addEventListener("click", () => setModel("linear"));

  // Sample Presets for Testing
  const samplePresets = [
    {
      living_area: 1710,
      overall_quality: 7,
      bedrooms: 3,
      bathrooms: 2,
      year_built: 2003,
      garage_cars: 2,
      basement_area: 856,
      neighborhood: "CollgCr"
    },
    {
      living_area: 2500,
      overall_quality: 9,
      bedrooms: 4,
      bathrooms: 3,
      year_built: 2015,
      garage_cars: 3,
      basement_area: 1400,
      neighborhood: "NridgHt"
    },
    {
      living_area: 1100,
      overall_quality: 5,
      bedrooms: 2,
      bathrooms: 1,
      year_built: 1960,
      garage_cars: 1,
      basement_area: 750,
      neighborhood: "OldTown"
    }
  ];

  let sampleIndex = 0;
  sampleDataBtn.addEventListener("click", () => {
    const sample = samplePresets[sampleIndex % samplePresets.length];
    sampleIndex++;

    document.getElementById("living_area").value = sample.living_area;
    document.getElementById("overall_quality").value = sample.overall_quality;
    qualitySlider.dispatchEvent(new Event("input"));
    document.getElementById("bedrooms").value = sample.bedrooms;
    document.getElementById("bathrooms").value = sample.bathrooms;
    document.getElementById("year_built").value = sample.year_built;
    document.getElementById("garage_cars").value = sample.garage_cars;
    document.getElementById("basement_area").value = sample.basement_area;
    document.getElementById("neighborhood").value = sample.neighborhood;
  });

  resetBtn.addEventListener("click", () => {
    predictForm.reset();
    qualitySlider.value = 7;
    qualitySlider.dispatchEvent(new Event("input"));
    setModel("random_forest");
    showEmptyState();
  });

  // Result State Handlers
  function showEmptyState() {
    resultEmpty.classList.remove("hidden");
    resultLoading.classList.add("hidden");
    resultContent.classList.add("hidden");
    resultError.classList.add("hidden");
  }

  function showLoadingState() {
    resultEmpty.classList.add("hidden");
    resultLoading.classList.remove("hidden");
    resultContent.classList.add("hidden");
    resultError.classList.add("hidden");

    predictBtn.disabled = true;
    btnSpinner.classList.remove("hidden");
    btnText.textContent = "Estimating...";
  }

  function showContentState() {
    resultEmpty.classList.add("hidden");
    resultLoading.classList.add("hidden");
    resultContent.classList.remove("hidden");
    resultError.classList.add("hidden");

    predictBtn.disabled = false;
    btnSpinner.classList.add("hidden");
    btnText.textContent = "Predict Price";
  }

  function showErrorState(msg) {
    resultEmpty.classList.add("hidden");
    resultLoading.classList.add("hidden");
    resultContent.classList.add("hidden");
    resultError.classList.remove("hidden");
    errorMessage.textContent = msg;

    predictBtn.disabled = false;
    btnSpinner.classList.add("hidden");
    btnText.textContent = "Predict Price";
  }

  const formatCurrency = (val) => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      maximumFractionDigits: 0
    }).format(val);
  };

  // Handle Form Submission -> POST /predict
  predictForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    showLoadingState();

    const formData = new FormData(predictForm);
    const payload = {
      living_area: parseFloat(formData.get("living_area")),
      overall_quality: parseInt(formData.get("overall_quality"), 10),
      bedrooms: parseInt(formData.get("bedrooms"), 10),
      bathrooms: parseInt(formData.get("bathrooms"), 10),
      year_built: parseInt(formData.get("year_built"), 10),
      garage_cars: parseInt(formData.get("garage_cars"), 10),
      basement_area: parseFloat(formData.get("basement_area")),
      neighborhood: formData.get("neighborhood"),
      model: formData.get("model") || "random_forest"
    };

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(payload)
      });

      const data = await response.json();

      if (!response.ok) {
        let msg = "Failed to obtain prediction.";
        if (data && data.detail) {
          if (Array.isArray(data.detail)) {
            msg = data.detail.map((err) => `${err.loc.join(" -> ")}: ${err.msg}`).join("; ");
          } else {
            msg = data.detail;
          }
        }
        showErrorState(msg);
        return;
      }

      // Populate prediction results
      predictedPriceEl.textContent = formatCurrency(data.predicted_price);
      priceRangeEl.textContent = `Likely between ${formatCurrency(data.range_low)} and ${formatCurrency(data.range_high)} (±8%)`;
      rangeLowValEl.textContent = formatCurrency(data.range_low);
      targetValEl.textContent = formatCurrency(data.predicted_price);
      rangeHighValEl.textContent = formatCurrency(data.range_high);

      modelBadgeEl.textContent = data.model_used;

      // Populate feature summary pills
      summaryPillsEl.innerHTML = `
        <span class="px-2.5 py-1 rounded-md bg-slate-100 text-slate-700 font-medium">${payload.living_area.toLocaleString()} sq ft</span>
        <span class="px-2.5 py-1 rounded-md bg-slate-100 text-slate-700 font-medium">Quality ${payload.overall_quality}/10</span>
        <span class="px-2.5 py-1 rounded-md bg-slate-100 text-slate-700 font-medium">${payload.bedrooms} Bed, ${payload.bathrooms} Bath</span>
        <span class="px-2.5 py-1 rounded-md bg-slate-100 text-slate-700 font-medium">${payload.garage_cars} Car Garage</span>
        <span class="px-2.5 py-1 rounded-md bg-slate-100 text-slate-700 font-medium">${payload.basement_area.toLocaleString()} sq ft Bsmt</span>
        <span class="px-2.5 py-1 rounded-md bg-slate-100 text-slate-700 font-medium">${payload.neighborhood} (${payload.year_built})</span>
      `;

      showContentState();

    } catch (err) {
      console.error("Prediction fetch error:", err);
      showErrorState("Could not reach backend service. Verify that the FastAPI server is running.");
    }
  });

  // Fetch /health to confirm backend is connected
  async function checkHealth() {
    const badge = document.getElementById("api-status-badge");
    const text = document.getElementById("api-status-text");
    try {
      const res = await fetch("/health");
      if (res.ok) {
        badge.className = "ml-2 inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-emerald-50 text-emerald-700 border border-emerald-200";
        text.textContent = "Backend Online";
      } else {
        throw new Error("Health check non-200");
      }
    } catch {
      badge.className = "ml-2 inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-rose-50 text-rose-700 border border-rose-200";
      text.textContent = "Backend Offline";
    }
  }

  // Fetch /metrics to populate tiles, tables, and render Chart.js graphs
  async function loadMetrics() {
    try {
      const response = await fetch("/metrics");
      if (!response.ok) return;

      const metrics = await response.json();

      // Stat tiles
      const rf = metrics.models?.random_forest;
      const lr = metrics.models?.linear;

      if (rf) {
        document.getElementById("metric-r2").textContent = rf.r2.toFixed(4);
        document.getElementById("metric-mae").textContent = formatCurrency(rf.mae);
        document.getElementById("metric-rmse").textContent = formatCurrency(rf.rmse);

        document.getElementById("table-rf-mae").textContent = formatCurrency(rf.mae);
        document.getElementById("table-rf-rmse").textContent = formatCurrency(rf.rmse);
        document.getElementById("table-rf-r2").textContent = rf.r2.toFixed(4);
      }

      if (lr) {
        document.getElementById("r2-linear-comp").textContent = lr.r2.toFixed(4);

        document.getElementById("table-lr-mae").textContent = formatCurrency(lr.mae);
        document.getElementById("table-lr-rmse").textContent = formatCurrency(lr.rmse);
        document.getElementById("table-lr-r2").textContent = lr.r2.toFixed(4);
      }

      // Render Charts if Chart.js is available
      if (typeof Chart !== "undefined") {
        renderFeatureImportanceChart(metrics.top_features);
        renderActualVsPredictedChart(metrics.actual_vs_predicted);
      }

    } catch (err) {
      console.error("Error loading metrics:", err);
    }
  }

  // Chart 1: Feature Importance (Horizontal Bar Chart)
  function renderFeatureImportanceChart(topFeatures) {
    const ctx = document.getElementById("chart-feature-importance");
    if (!ctx || !topFeatures || topFeatures.length === 0) return;

    const labels = topFeatures.map((item) => item.feature);
    const data = topFeatures.map((item) => item.percentage);

    new Chart(ctx, {
      type: "bar",
      data: {
        labels: labels,
        datasets: [{
          label: "Importance Percentage (%)",
          data: data,
          backgroundColor: [
            "#4F46E5",
            "#6366F1",
            "#818CF8",
            "#14B8A6",
            "#2DD4BF"
          ],
          borderRadius: 8
        }]
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (context) => ` Contribution: ${context.parsed.x}%`
            }
          }
        },
        scales: {
          x: {
            beginAtZero: true,
            max: 70,
            ticks: {
              callback: (val) => `${val}%`
            },
            grid: { color: "#F1F5F9" }
          },
          y: {
            grid: { display: false }
          }
        }
      }
    });
  }

  // Chart 2: Actual vs Predicted Scatter Chart
  function renderActualVsPredictedChart(points) {
    const ctx = document.getElementById("chart-actual-vs-predicted");
    if (!ctx || !points || points.length === 0) return;

    const scatterData = points.map((p) => ({
      x: p.actual,
      y: p.predicted_rf
    }));

    // Find min and max for line of best fit (y = x)
    const minVal = Math.min(...points.map((p) => Math.min(p.actual, p.predicted_rf)));
    const maxVal = Math.max(...points.map((p) => Math.max(p.actual, p.predicted_rf)));

    new Chart(ctx, {
      type: "scatter",
      data: {
        datasets: [
          {
            label: "Random Forest Prediction",
            data: scatterData,
            backgroundColor: "rgba(79, 70, 229, 0.7)",
            borderColor: "#4F46E5",
            pointRadius: 4,
            pointHoverRadius: 6
          },
          {
            label: "Perfect Fit (y = x)",
            data: [
              { x: minVal, y: minVal },
              { x: maxVal, y: maxVal }
            ],
            type: "line",
            borderColor: "rgba(16, 185, 129, 0.8)",
            borderDash: [5, 5],
            borderWidth: 2,
            pointRadius: 0,
            fill: false
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          tooltip: {
            callbacks: {
              label: (context) => {
                if (context.datasetIndex === 1) return "Perfect 1:1 Reference Line";
                return ` Actual: ${formatCurrency(context.parsed.x)} | Predicted: ${formatCurrency(context.parsed.y)}`;
              }
            }
          }
        },
        scales: {
          x: {
            title: { display: true, text: "Actual Price ($)" },
            ticks: {
              callback: (val) => `$${(val / 1000).toFixed(0)}k`
            },
            grid: { color: "#F1F5F9" }
          },
          y: {
            title: { display: true, text: "Predicted Price ($)" },
            ticks: {
              callback: (val) => `$${(val / 1000).toFixed(0)}k`
            },
            grid: { color: "#F1F5F9" }
          }
        }
      }
    });
  }

  // Initial runs
  checkHealth();
  loadMetrics();
});

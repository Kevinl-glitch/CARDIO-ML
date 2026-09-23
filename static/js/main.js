// CardioML Interactive Client Engine

document.addEventListener("DOMContentLoaded", function () {
  // Elements for dynamic calculation
  const heightInput = document.getElementById("height");
  const weightInput = document.getElementById("weight");
  const apHiInput = document.getElementById("ap_hi");
  const apLoInput = document.getElementById("ap_lo");

  const liveBmiVal = document.getElementById("live-bmi-val");
  const liveBmiBadge = document.getElementById("live-bmi-badge");
  const livePpVal = document.getElementById("live-pp-val");
  const livePpBadge = document.getElementById("live-pp-badge");

  function updateLiveMetrics() {
    if (!heightInput || !weightInput || !apHiInput || !apLoInput) return;

    const h = parseFloat(heightInput.value);
    const w = parseFloat(weightInput.value);
    const hi = parseFloat(apHiInput.value);
    const lo = parseFloat(apLoInput.value);

    // Calculate BMI
    if (h > 0 && w > 0) {
      const hm = h / 100.0;
      const bmi = (w / (hm * hm)).toFixed(1);
      if (liveBmiVal) liveBmiVal.textContent = bmi;

      if (liveBmiBadge) {
        liveBmiBadge.className = "badge-status";
        if (bmi < 18.5) {
          liveBmiBadge.textContent = "Underweight";
          liveBmiBadge.classList.add("status-yellow");
        } else if (bmi < 25.0) {
          liveBmiBadge.textContent = "Normal";
          liveBmiBadge.classList.add("status-green");
        } else if (bmi < 30.0) {
          liveBmiBadge.textContent = "Overweight";
          liveBmiBadge.classList.add("status-yellow");
        } else {
          liveBmiBadge.textContent = "Obese";
          liveBmiBadge.classList.add("status-red");
        }
      }
    }

    // Calculate Pulse Pressure
    if (!isNaN(hi) && !isNaN(lo)) {
      const pp = Math.round(hi - lo);
      if (livePpVal) livePpVal.textContent = pp + " mmHg";

      if (livePpBadge) {
        livePpBadge.className = "badge-status";
        if (pp >= 30 && pp <= 50) {
          livePpBadge.textContent = "Normal";
          livePpBadge.classList.add("status-green");
        } else if (pp > 50 && pp <= 60) {
          livePpBadge.textContent = "Elevated";
          livePpBadge.classList.add("status-yellow");
        } else {
          livePpBadge.textContent = "High / Abnormal";
          livePpBadge.classList.add("status-red");
        }
      }
    }
  }

  // Attach listeners
  [heightInput, weightInput, apHiInput, apLoInput].forEach((el) => {
    if (el) {
      el.addEventListener("input", updateLiveMetrics);
    }
  });

  // Run on page load
  updateLiveMetrics();

  // Async AJAX Prediction Form submission
  const predForm = document.getElementById("cvd-prediction-form");
  const resultBox = document.getElementById("prediction-result-box");
  const submitBtn = document.getElementById("btn-submit-predict");

  if (predForm) {
    predForm.addEventListener("submit", function (e) {
      e.preventDefault();

      if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.innerHTML = "Processing Analysis...";
      }

      const formData = new FormData(predForm);
      const payload = {};
      formData.forEach((value, key) => {
        payload[key] = parseFloat(value);
      });

      fetch("/api/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(payload)
      })
        .then((response) => response.json())
        .then((data) => {
          if (data.status === "success") {
            displayPredictionResults(data);
          } else {
            alert("Prediction error: " + (data.message || "Invalid inputs"));
          }
        })
        .catch((err) => {
          console.error("Inference request failed:", err);
          alert("Network or inference error occurred. Please check server status.");
        })
        .finally(() => {
          if (submitBtn) {
            submitBtn.disabled = false;
            submitBtn.innerHTML = "Predict Risk Assessment";
          }
        });
    });
  }

  function displayPredictionResults(data) {
    if (!resultBox) return;

    resultBox.style.display = "block";
    resultBox.scrollIntoView({ behavior: "smooth" });

    const statusBadge = document.getElementById("result-status-badge");
    const probaText = document.getElementById("result-proba-val");
    const meterFill = document.getElementById("result-meter-fill");
    const factorsList = document.getElementById("result-factors-list");

    const isHighRisk = data.prediction === 1;

    if (statusBadge) {
      statusBadge.className = "result-status " + (isHighRisk ? "high-risk" : "low-risk");
      statusBadge.innerHTML = isHighRisk
        ? "⚠️ Higher Cardiovascular Risk Detected"
        : "✅ Lower Cardiovascular Risk Detected";
    }

    if (probaText) {
      probaText.textContent = data.probability + "%";
    }

    if (meterFill) {
      meterFill.style.width = data.probability + "%";
      meterFill.className = "prob-meter-fill " + (isHighRisk ? "high" : "low");
    }

    if (factorsList) {
      factorsList.innerHTML = "";
      if (data.risk_factors && data.risk_factors.length > 0) {
        data.risk_factors.forEach((rf) => {
          const li = document.createElement("li");
          li.textContent = rf;
          factorsList.appendChild(li);
        });
      } else {
        const li = document.createElement("li");
        li.textContent = "All primary clinical measurements (Blood Pressure, BMI, Cholesterol) fall within standard ranges.";
        factorsList.appendChild(li);
      }
    }
  }
});

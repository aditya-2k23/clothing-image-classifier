// Theme toggle functionality
const themeToggle = document.getElementById('theme-toggle');
const themeIcon = document.getElementById('theme-icon');

// Check for saved theme preference or default to system preference
function getPreferredTheme() {
  const savedTheme = localStorage.getItem('theme');
  if (savedTheme) {
    return savedTheme;
  }
  // Check system preference
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

// Apply theme
function applyTheme(theme) {
  if (theme === 'dark') {
    document.documentElement.setAttribute('data-theme', 'dark');
    themeIcon.classList.remove('fa-moon');
    themeIcon.classList.add('fa-sun');
  } else {
    document.documentElement.removeAttribute('data-theme');
    themeIcon.classList.remove('fa-sun');
    themeIcon.classList.add('fa-moon');
  }
}

// Initialize theme
applyTheme(getPreferredTheme());

// Toggle theme on button click
themeToggle.addEventListener('click', () => {
  const currentTheme = document.documentElement.getAttribute('data-theme');
  const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
  localStorage.setItem('theme', newTheme);
  applyTheme(newTheme);

  // Re-render charts if they exist with new theme colors
  updateChartsTheme();
});

// Listen for system theme changes
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
  if (!localStorage.getItem('theme')) {
    applyTheme(e.matches ? 'dark' : 'light');
  }
});

let selectedFile = null;
const dropArea = document.getElementById("drop-area");
const fileInput = document.getElementById("image");
const preview = document.getElementById("preview");
const previewContainer = document.getElementById("preview-container");
const predictBtn = document.getElementById("predictBtn");
const btnText = document.getElementById("btn-text");
const statusMessage = document.getElementById("status-message");
const statusText = document.getElementById("status-text");
const resultsCard = document.getElementById("results-card");

dropArea.addEventListener("click", () => fileInput.click());

dropArea.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropArea.classList.add("dragover");
});

dropArea.addEventListener("dragleave", () => {
  dropArea.classList.remove("dragover");
});

dropArea.addEventListener("drop", (e) => {
  e.preventDefault();
  dropArea.classList.remove("dragover");
  selectedFile = e.dataTransfer.files[0];
  showPreview(selectedFile);
});

fileInput.addEventListener("change", () => {
  selectedFile = fileInput.files[0];
  showPreview(selectedFile);
});

function showPreview(file) {
  const reader = new FileReader();
  reader.onload = () => {
    preview.src = reader.result;
    previewContainer.classList.add("show");
    document.querySelector(".upload-content").classList.add("has-preview");
    document.getElementById("file-info").textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
  };
  reader.readAsDataURL(file);
}

predictBtn.onclick = async () => {
  if (!selectedFile) {
    statusText.textContent = "Please upload an image first.";
    statusMessage.classList.add("show", "error");
    return;
  }

  statusMessage.classList.add("show");
  statusText.textContent = "Preprocessing image (resizing to 28×28)...";
  predictBtn.disabled = true;
  btnText.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Analyzing...';

  let formData = new FormData();
  formData.append("image", selectedFile);

  setTimeout(() => {
    statusText.textContent = "Running CNN prediction...";
  }, 500);

  setTimeout(() => {
    statusText.textContent = "Running Random Forest prediction...";
  }, 1000);

  setTimeout(() => {
    statusText.textContent = "Running Logistic Regression prediction...";
  }, 1500);

  try {
    const res = await fetch("/predict", { method: "POST", body: formData });
    const data = await res.json();

    statusMessage.classList.remove("show");
    predictBtn.disabled = false;
    btnText.textContent = 'Analyze Image';

    const cnnConf = (Math.max(...data.cnn_probs) * 100).toFixed(1);
    const rfConf = (Math.max(...data.rf_probs) * 100).toFixed(1);
    const logConf = (Math.max(...data.log_probs) * 100).toFixed(1);

    // Display preprocessed image
    document.getElementById("preprocessed-img").src = `data:image/png;base64,${data.preview_img}`;

    document.getElementById("model-results").innerHTML = `
      <div class="model-result-item cnn">
        <div class="model-name">Convolutional Neural Network</div>
        <div class="model-prediction">${data.cnn_pred}</div>
        <div class="model-confidence">${cnnConf}% confidence</div>
        <span class="model-badge">Best Accuracy: 91.07%</span>
      </div>
      <div class="model-result-item rf">
        <div class="model-name">Random Forest</div>
        <div class="model-prediction">${data.rf_pred}</div>
        <div class="model-confidence">${rfConf}% confidence</div>
        <span class="model-badge">Accuracy: 87.64%</span>
      </div>
      <div class="model-result-item lr">
        <div class="model-name">Logistic Regression</div>
        <div class="model-prediction">${data.log_pred}</div>
        <div class="model-confidence">${logConf}% confidence</div>
        <span class="model-badge">Accuracy: 84.40%</span>
      </div>
    `;

    resultsCard.classList.add("show");

    plotChart("cnn_chart", data.labels, data.cnn_probs, "CNN Confidence Distribution", "#0ea5e9");
    plotChart("rf_chart", data.labels, data.rf_probs, "Random Forest Confidence Distribution", "#10b981");
    plotChart("log_chart", data.labels, data.log_probs, "Logistic Regression Confidence Distribution", "#f59e0b");

    resultsCard.scrollIntoView({ behavior: 'smooth', block: 'start' });

  } catch (error) {
    statusMessage.classList.remove("show");
    predictBtn.disabled = false;
    btnText.textContent = 'Analyze Image';
    alert("An error occurred while processing the image. Please try again.");
  }
};

// Store chart data for theme updates
let chartData = {};

function getChartColors() {
  const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
  return {
    textColor: isDark ? '#f1f5f9' : '#334155',
    gridColor: isDark ? '#475569' : '#e2e8f0',
    tickColor: isDark ? '#94a3b8' : '#64748b'
  };
}

function plotChart(divId, labels, probs, title, color) {
  // Store data for theme updates
  chartData[divId] = { labels, probs, title, color };

  const colors = getChartColors();

  var trace = {
    x: labels,
    y: probs,
    type: 'bar',
    marker: {
      color: color,
      opacity: 0.85
    },
    hovertemplate: '%{x}<br>Confidence: %{y:.1%}<extra></extra>'
  };

  var layout = {
    title: {
      text: title,
      font: { size: 16, color: colors.textColor }
    },
    yaxis: {
      title: "Confidence",
      range: [0, 1],
      tickformat: '.0%',
      gridcolor: colors.gridColor,
      tickfont: { color: colors.tickColor },
      titlefont: { color: colors.textColor }
    },
    xaxis: {
      tickangle: -45,
      tickfont: { size: 11, color: colors.tickColor },
      automargin: true
    },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    margin: { t: 30, b: 30, l: 70, r: 30 },
    height: 400,
    hoverlabel: {
      bgcolor: '#1e293b',
      font: { color: 'white' }
    }
  };

  var config = {
    responsive: true,
    displayModeBar: false
  };

  Plotly.newPlot(divId, [trace], layout, config);
}

// Update all charts when theme changes
function updateChartsTheme() {
  Object.keys(chartData).forEach(divId => {
    const data = chartData[divId];
    if (data) {
      plotChart(divId, data.labels, data.probs, data.title, data.color);
    }
  });
}

let history = [];
const form = document.getElementById('emailForm');
const resultDiv = document.getElementById('result');
const historyDiv = document.getElementById('history');
const chartCanvas = document.getElementById('chart');
const spinner = document.getElementById('spinner');
let chart;

form.onsubmit = async (e) => {
  e.preventDefault();
  resultDiv.textContent = '';
  spinner.style.display = 'flex';

  const subject = document.getElementById('subject').value;
  const body = document.getElementById('body').value;

  try {
    const response = await fetch('http://127.0.0.1:5000/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ subject, body })
    });
    const data = await response.json();
    spinner.style.display = 'none';

    let risk = '';
    let color = '';
    let riskClass = '';
    if (data.ensemble_probability > 0.7) {
      risk = '🔴 HIGH RISK: Likely Phishing';
      color = '#ffcccc';
      riskClass = 'risk-high';
    } else if (data.ensemble_probability > 0.3) {
      risk = '🟡 MEDIUM RISK: Suspicious';
      color = '#fff3cd';
      riskClass = 'risk-medium';
    } else {
      risk = '🟢 LOW RISK: Likely Legitimate';
      color = '#d4edda';
      riskClass = 'risk-low';
    }

    resultDiv.innerHTML = `
      <div style="padding:10px;border-radius:5px;">
        <strong class="${riskClass}">Result:</strong> <span class="${riskClass}">${risk}</span><br>
        <strong>Phishing Probability:</strong> ${(data.ensemble_probability * 100).toFixed(2)}%<br>
        <strong>Model Confidence:</strong> ${data.ensemble_confidence.toFixed(2)}<br>
      </div>
    `;

    // Automatically clear history and chart, then add only the latest prediction
    history = [];
    history.push({ subject, body, risk, probability: data.ensemble_probability, confidence: data.ensemble_confidence });
    updateHistory();
    updateChart();
    document.getElementById('subject').focus();
  } catch (err) {
    spinner.style.display = 'none';
    resultDiv.textContent = 'Error connecting to backend: ' + err;
  }
};

function updateHistory() {
  const historyDiv = document.getElementById('history');
  if (history.length === 0) {
    historyDiv.innerHTML = '<h3>Prediction History</h3><p>No predictions yet.</p>';
    return;
  }
  historyDiv.innerHTML = '<h3>Prediction History</h3>';
  history.forEach((item, index) => {
    const div = document.createElement('div');
    div.innerHTML = `
      <strong>Subject:</strong> ${item.subject}<br>
      <strong>Risk:</strong> <span class="risk-${item.risk.toLowerCase()}">${item.risk}</span><br>
      <strong>Probability:</strong> ${(item.probability * 100).toFixed(2)}%<br>
      <strong>Confidence:</strong> ${item.confidence.toFixed(2)}
    `;
    historyDiv.appendChild(div);
  });
}

function updateChart() {
    const ctx = document.getElementById('chart').getContext('2d');
    
    if (window.myChart) {
        window.myChart.destroy();
    }
    
    if (history.length === 0) {
        return;
    }
    
    const labels = history.map((_, index) => `Analysis ${index + 1}`);
    const data = history.map(item => (item.probability * 100).toFixed(1));
    const colors = history.map(item => {
        if (item.probability > 0.7) return '#e53e3e'; // Red for high risk
        if (item.probability > 0.3) return '#dd6b20'; // Orange for medium risk
        return '#38a169'; // Green for low risk
    });
    
    window.myChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Phishing Probability (%)',
                data: data,
                backgroundColor: colors,
                borderColor: colors.map(color => color + '80'),
                borderWidth: 2,
                borderRadius: 8,
                borderSkipped: false,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Phishing Detection Analysis History',
                    color: '#3182ce',
                    font: {
                        size: 18,
                        weight: 'bold',
                        family: 'Inter'
                    },
                    padding: 20
                },
                legend: {
                    display: true,
                    labels: {
                        color: '#a0aec0',
                        font: {
                            size: 12,
                            family: 'Inter'
                        },
                        usePointStyle: true,
                        padding: 15
                    }
                }
            },
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Analysis Number',
                        color: '#3182ce',
                        font: {
                            size: 14,
                            weight: 'bold',
                            family: 'Inter'
                        },
                        padding: 10
                    },
                    ticks: {
                        color: '#a0aec0',
                        font: {
                            size: 12,
                            family: 'Inter'
                        },
                        padding: 8
                    },
                    grid: {
                        display: true,
                        color: 'rgba(45, 55, 72, 0.3)',
                        lineWidth: 1,
                        drawBorder: true,
                        borderColor: '#2d3748'
                    }
                },
                y: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Phishing Probability (%)',
                        color: '#3182ce',
                        font: {
                            size: 14,
                            weight: 'bold',
                            family: 'Inter'
                        },
                        padding: 10
                    },
                    ticks: {
                        color: '#a0aec0',
                        font: {
                            size: 12,
                            family: 'Inter'
                        },
                        padding: 8,
                        callback: function(value) {
                            return value + '%';
                        }
                    },
                    grid: {
                        display: true,
                        color: 'rgba(45, 55, 72, 0.3)',
                        lineWidth: 1,
                        drawBorder: true,
                        borderColor: '#2d3748'
                    },
                    min: 0,
                    max: 100,
                    beginAtZero: true
                }
            },
            elements: {
                bar: {
                    backgroundColor: colors,
                    borderColor: colors.map(color => color + '80'),
                    borderWidth: 2
                }
            },
            interaction: {
                intersect: false,
                mode: 'index'
            },
            animation: {
                duration: 1000,
                easing: 'easeInOutQuart'
            }
        }
    });
}

// Load Chart.js dynamically
if (!window.Chart) {
  const script = document.createElement('script');
  script.src = 'https://cdn.jsdelivr.net/npm/chart.js';
  script.onload = updateChart;
  document.head.appendChild(script);
} 
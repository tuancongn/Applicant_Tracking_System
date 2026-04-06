/**
 * PAVN ATS - Frontend Application
 * CV-JD Matching with Radar Chart, Gap Analysis, AI Suggestions
 */

// ====== State ======
let state = {
    cvText: '',       // used only for custom uploaded CV
    cvId: '',         // used for sample CV
    cvFilename: '',
    selectedJdId: '',
    jds: [],
    cvs: [],
    currentResult: null,
};

let radarChartInstance = null;

// ====== Initialize ======
document.addEventListener('DOMContentLoaded', () => {
    loadSampleData();
    setupFileUpload();
});

// ====== API Calls ======
async function loadSampleData() {
    try {
        const res = await fetch('/api/sample-data');
        const data = await res.json();

        // Load CV tabs
        if (data.cv_list && data.cv_list.length > 0) {
            state.cvs = data.cv_list;
            renderCvTabs(data.cv_list);
        } else if (data.cv) {
            // fallback
            showCvInfo(data.cv);
        }

        // Load JD tabs
        if (data.jds && data.jds.length > 0) {
            state.jds = data.jds;
            renderJdTabs(data.jds);
        }
    } catch (err) {
        console.error('Error loading sample data:', err);
    }
}

function renderCvTabs(cvs) {
    const container = document.getElementById('cvTabs');
    if (!container) return;

    container.innerHTML = cvs.map((cv, i) => `
        <button class="jd-tab cv-tab-btn ${i === 0 ? 'active' : ''}" data-cv-id="${cv.filename}" onclick="selectCv(this, '${cv.filename}')">
            <span class="jd-tab-radio"></span>
            <span class="jd-tab-name">${cv.filename.replace('.pdf', '')}</span>
            ${cv.error ? '<span style="color:red; font-size:12px; margin-left: 5px;">(Error)</span>' : ''}
        </button>
    `).join('');

    // Auto-select first CV
    if (cvs.length > 0) {
        selectCv(container.querySelector('.cv-tab-btn'), cvs[0].filename);
    }
}

function selectCv(btn, cvId) {
    // Clear custom uploaded info
    state.cvText = '';
    document.getElementById('cvInfo').style.display = 'none';
    document.getElementById('uploadArea').style.display = 'none';

    // Update active state
    document.querySelectorAll('.cv-tab-btn').forEach(t => t.classList.remove('active'));
    if (btn) btn.classList.add('active');
    
    state.cvId = cvId;
}

function showCvInfo(cv) {
    const cvInfo = document.getElementById('cvInfo');
    const cvFilename = document.getElementById('cvFilename');
    const cvMeta = document.getElementById('cvMeta');

    if (cv && cv.filename) {
        cvFilename.textContent = cv.filename;
        const metaParts = [];
        if (cv.metadata?.language) metaParts.push(`Language: ${cv.metadata.language === 'vi' ? 'Vietnamese' : 'English'}`);
        if (cv.metadata?.email) metaParts.push(cv.metadata.email);
        if (cv.word_count) metaParts.push(`${cv.word_count} words`);
        if (cv.metadata?.pages) metaParts.push(`${cv.metadata.pages} page(s)`);
        cvMeta.textContent = metaParts.join(' • ');
        
        // Hide standard tabs, show uploaded info
        document.getElementById('cvTabs').style.display = 'none';
        cvInfo.style.display = 'block';
        document.getElementById('uploadArea').style.display = 'none';
    }
}

function renderJdTabs(jds) {
    const container = document.getElementById('jdTabs');
    container.innerHTML = jds.map((jd, i) => `
        <button class="jd-tab ${i === 0 ? 'active' : ''}" data-jd-id="${jd.id}" onclick="selectJd(this, '${jd.id}')">
            <span class="jd-tab-radio"></span>
            <span class="jd-tab-name">${jd.name}</span>
        </button>
    `).join('');

    // Auto-select first JD
    if (jds.length > 0) {
        state.selectedJdId = jds[0].id;
    }
}

function selectJd(btn, jdId) {
    // Clear custom text when selecting a tab
    document.getElementById('jdCustomText').value = '';

    // Update active state
    document.querySelectorAll('.jd-tab').forEach(t => t.classList.remove('active'));
    btn.classList.add('active');
    state.selectedJdId = jdId;
}

// ====== File Upload ======
function setupFileUpload() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');

    // Drag & drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        const file = e.dataTransfer.files[0];
        if (file && file.type === 'application/pdf') {
            uploadFile(file);
        }
    });

    // File input
    fileInput.addEventListener('change', (e) => {
        if (e.target.files[0]) {
            uploadFile(e.target.files[0]);
        }
    });
}

async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch('/api/upload-cv', {
            method: 'POST',
            body: formData,
        });
        const data = await res.json();

        if (data.error) {
            alert(data.error);
            return;
        }

        state.cvText = data.data.raw_text;
        state.cvId = ''; // Not a sample CV
        state.cvFilename = data.filename;

        document.getElementById('cvFilename').textContent = data.filename;
        const metaParts = [];
        if (data.data.metadata?.language) metaParts.push(`Language: ${data.data.metadata.language === 'vi' ? 'Vietnamese' : 'English'}`);
        if (data.data.metadata?.email) metaParts.push(data.data.metadata.email);
        if (data.data.word_count) metaParts.push(`${data.data.word_count} words`);
        document.getElementById('cvMeta').textContent = metaParts.join(' • ');
        
        // Hide sample tabs
        const cvTabs = document.getElementById('cvTabs');
        if (cvTabs) cvTabs.style.display = 'none';
        
        document.getElementById('cvInfo').style.display = 'block';
        document.getElementById('uploadArea').style.display = 'none';
    } catch (err) {
        alert('Upload error: ' + err.message);
    }
}

// ====== Analysis ======
async function analyzeSingle() {
    const customJd = document.getElementById('jdCustomText').value.trim();
    const jdId = state.selectedJdId;

    if (!customJd && !jdId) {
        alert('Please select a JD or paste a job description!');
        return;
    }

    showLoading();

    try {
        const body = {};

        if (state.cvText) {
            body.cv_text = state.cvText;
        } else if (state.cvId) {
            body.cv_id = state.cvId;
        }

        if (customJd) {
            body.jd_text = customJd;
        } else {
            body.jd_id = jdId;
        }

        const res = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });

        const data = await res.json();

        if (res.status === 429 || data.rate_limited) {
            showError('Rate Limit Error', data.error || 'Free request limit exceeded. Please wait a few minutes.');
            return;
        }

        if (data.error && !data.final_scores) {
            showError('Analysis Error', data.error);
            return;
        }

        state.currentResult = data;
        renderResults(data);
    } catch (err) {
        showError('Connection Error', err.message);
    }
}

async function analyzeBatch() {
    if (state.jds.length === 0) {
        alert('No sample JDs available for comparison!');
        return;
    }

    showLoading();

    try {
        const body = {
            jd_ids: state.jds.map(j => j.id),
        };
        if (state.cvText) body.cv_text = state.cvText;
        if (state.cvId) body.cv_id = state.cvId;

        const res = await fetch('/api/batch-analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });

        const data = await res.json();

        // Check for rate limit in any result
        const rateLimited = data.results?.find(r => r.rate_limited);
        if (rateLimited) {
            showError('Rate Limit Error', rateLimited.error || 'Request limit exceeded. Please wait.');
            return;
        }

        renderBatchResults(data.results);
    } catch (err) {
        showError('Connection Error', err.message);
    }
}

// ====== UI Functions ======
function showLoading() {
    document.getElementById('inputSection').style.display = 'none';
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('batchSection').style.display = 'none';
    document.getElementById('errorSection').style.display = 'none';
    document.getElementById('loadingSection').style.display = 'block';

    // Animate loading steps
    const steps = ['step1', 'step2', 'step3', 'step4'];
    steps.forEach((s, i) => {
        const el = document.getElementById(s);
        el.className = 'load-step';
        setTimeout(() => {
            el.classList.add('active');
            if (i > 0) {
                document.getElementById(steps[i - 1]).classList.remove('active');
                document.getElementById(steps[i - 1]).classList.add('done');
            }
        }, i * 1500);
    });
}

function hideLoading() {
    document.getElementById('loadingSection').style.display = 'none';
}

function showError(title, message) {
    hideLoading();
    document.getElementById('errorTitle').textContent = title;
    document.getElementById('errorMessage').textContent = message;
    document.getElementById('errorSection').style.display = 'block';
}

function hideError() {
    document.getElementById('errorSection').style.display = 'none';
    document.getElementById('inputSection').style.display = 'block';
}

function backToInput() {
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('batchSection').style.display = 'none';
    document.getElementById('inputSection').style.display = 'block';
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ====== Render Results ======
function renderResults(data) {
    hideLoading();

    const scores = data.final_scores;
    const ai = data.ai_analysis;
    const local = data.local_analysis;

    // Score Hero
    const scoreEl = document.getElementById('overallScore');
    const matchEl = document.getElementById('matchLevel');
    const summaryEl = document.getElementById('scoreSummary');
    const badgesEl = document.getElementById('scoreBadges');

    scoreEl.textContent = Math.round(scores.overall);
    matchEl.textContent = scores.match_level;
    matchEl.style.color = scores.color;

    // Score ring animation
    const ring = document.querySelector('.score-ring-progress');
    const circumference = 2 * Math.PI * 85; // r=85
    const offset = circumference - (scores.overall / 100) * circumference;
    ring.style.stroke = scores.color;
    ring.style.strokeDashoffset = offset;
    scoreEl.style.color = scores.color;

    // Summary
    if (ai && ai.summary) {
        summaryEl.textContent = ai.summary;
    } else {
        summaryEl.textContent = `Based on ML analysis, CV has a ${scores.match_level.toLowerCase()} match with this position. Semantic Similarity: ${local.tfidf_similarity}%`;
    }

    // Badges
    let badgesHtml = '<span class="score-badge score-badge-ml">📊 ML Analysis</span>';
    if (data.ai_available) {
        badgesHtml += '<span class="score-badge score-badge-ai">🤖 AI Enhanced</span>';
    }
    badgesEl.innerHTML = badgesHtml;

    // Dimension Scores
    renderDimensionScores(scores);

    // Radar Chart
    renderRadarChart(scores);

    // Skills Gap
    renderSkillsGap(local, ai);

    // AI Suggestions
    if (ai && !ai.error) {
        renderAISuggestions(ai);
        renderATSCheck(ai);
        renderCourses(ai);
    } else {
        document.getElementById('suggestionsPanel').style.display = 'none';
        document.getElementById('atsPanel').style.display = 'none';
        document.getElementById('coursesPanel').style.display = 'none';
    }

    // Show results
    document.getElementById('resultsSection').style.display = 'block';
    document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
}

function renderDimensionScores(scores) {
    const dims = [
        { key: 'hard_skills', name: 'Technical Skills', icon: '💻', weight: '30%' },
        { key: 'experience', name: 'Experience', icon: '📋', weight: '25%' },
        { key: 'education', name: 'Education', icon: '🎓', weight: '15%' },
        { key: 'language', name: 'Language', icon: '🌐', weight: '10%' },
        { key: 'soft_skills', name: 'Soft Skills', icon: '🤝', weight: '10%' },
        { key: 'culture_fit', name: 'Culture Fit', icon: '🏢', weight: '10%' },
    ];

    const container = document.getElementById('dimensionScores');
    container.innerHTML = dims.map((dim, i) => {
        const score = scores[dim.key] || 0;
        const color = getScoreColor(score);
        return `
            <div class="dimension-item fade-in-up delay-${i + 1}">
                <div class="dimension-header">
                    <span class="dimension-name">${dim.icon} ${dim.name} <span style="color: var(--text-muted); font-weight:400; font-size:0.75rem">(${dim.weight})</span></span>
                    <span class="dimension-value" style="color: ${color}">${Math.round(score)}%</span>
                </div>
                <div class="dimension-bar">
                    <div class="dimension-fill" style="width: ${score}%; background: linear-gradient(90deg, ${color}, ${color}aa);" data-width="${score}"></div>
                </div>
            </div>
        `;
    }).join('');

    // Animate bars
    requestAnimationFrame(() => {
        document.querySelectorAll('.dimension-fill').forEach(el => {
            el.style.width = el.dataset.width + '%';
        });
    });
}

function renderRadarChart(scores) {
    const ctx = document.getElementById('radarChart').getContext('2d');

    if (radarChartInstance) {
        radarChartInstance.destroy();
    }

    const labels = ['Hard Skills', 'Experience', 'Education', 'Language', 'Soft Skills', 'Culture Fit'];
    const values = [
        scores.hard_skills || 0,
        scores.experience || 0,
        scores.education || 0,
        scores.language || 0,
        scores.soft_skills || 0,
        scores.culture_fit || 0,
    ];

    radarChartInstance = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Match Level (%)',
                data: values,
                backgroundColor: 'rgba(99, 102, 241, 0.15)',
                borderColor: 'rgba(99, 102, 241, 0.8)',
                borderWidth: 2,
                pointBackgroundColor: values.map(v => getScoreColor(v)),
                pointBorderColor: 'rgba(255,255,255,0.8)',
                pointBorderWidth: 1,
                pointRadius: 5,
                pointHoverRadius: 8,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: { display: false },
            },
            scales: {
                r: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        stepSize: 20,
                        color: 'rgba(255,255,255,0.3)',
                        backdropColor: 'transparent',
                        font: { size: 10 },
                    },
                    grid: {
                        color: 'rgba(255,255,255,0.06)',
                    },
                    angleLines: {
                        color: 'rgba(255,255,255,0.08)',
                    },
                    pointLabels: {
                        color: 'rgba(255,255,255,0.7)',
                        font: { size: 11, weight: '500' },
                    },
                },
            },
            animation: {
                duration: 1500,
                easing: 'easeInOutQuart',
            },
        },
    });
}

function renderSkillsGap(local, ai) {
    const container = document.getElementById('skillsGrid');
    let html = '';

    // Matched skills
    if (local.hard_skills.matched.length > 0) {
        html += `
            <div class="skill-group">
                <div class="skill-group-title">✅ Matched Skills (${local.hard_skills.matched.length})</div>
                <div class="skill-tags">
                    ${local.hard_skills.matched.map(s => `<span class="skill-tag skill-tag-matched">${s}</span>`).join('')}
                </div>
            </div>
        `;
    }

    // Missing skills
    if (local.hard_skills.missing.length > 0) {
        html += `
            <div class="skill-group">
                <div class="skill-group-title">❌ Missing Skills (${local.hard_skills.missing.length})</div>
                <div class="skill-tags">
                    ${local.hard_skills.missing.map(s => `<span class="skill-tag skill-tag-missing">${s}</span>`).join('')}
                </div>
            </div>
        `;
    }

    // Extra skills in CV
    if (local.hard_skills.extra_in_cv.length > 0) {
        html += `
            <div class="skill-group">
                <div class="skill-group-title">🔵 Additional Skills in CV (${local.hard_skills.extra_in_cv.length})</div>
                <div class="skill-tags">
                    ${local.hard_skills.extra_in_cv.map(s => `<span class="skill-tag skill-tag-extra">${s}</span>`).join('')}
                </div>
            </div>
        `;
    }

    // AI Missing Skills with priority
    if (ai && ai.missing_skills && ai.missing_skills.length > 0) {
        html += `
            <div class="skill-group" style="grid-column: 1 / -1;">
                <div class="skill-group-title">🎯 Detailed AI Analysis</div>
                ${ai.missing_skills.map(s => `
                    <div class="missing-skill-item">
                        <span class="skill-priority priority-${getPriorityClass(s.priority)}">${s.priority}</span>
                        <div>
                            <div class="skill-name">${s.skill}</div>
                            <div class="skill-suggestion">${s.suggestion || ''}</div>
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
    }

    // Strengths & Weaknesses from AI
    if (ai && (ai.strengths || ai.weaknesses)) {
        html += `
            <div class="skill-group" style="grid-column: 1 / -1;">
                <div class="sw-grid">
                    <div>
                        <div class="sw-title">💪 Strengths</div>
                        <ul class="sw-list">
                            ${(ai.strengths || []).map(s => `<li><span style="color: var(--success)">✓</span> ${s}</li>`).join('')}
                        </ul>
                    </div>
                    <div>
                        <div class="sw-title">⚠️ Areas for Improvement</div>
                        <ul class="sw-list">
                            ${(ai.weaknesses || []).map(w => `<li><span style="color: var(--warning)">●</span> ${w}</li>`).join('')}
                        </ul>
                    </div>
                </div>
            </div>
        `;
    }

    // Keyword optimization from AI
    if (ai && ai.keyword_optimization) {
        const ko = ai.keyword_optimization;
        html += `
            <div class="skill-group keyword-section" style="grid-column: 1 / -1;">
                <div class="keyword-title">🔑 ATS Keyword Optimization</div>
                ${ko.missing_keywords && ko.missing_keywords.length > 0 ? `
                    <div style="margin-bottom: 10px;">
                        <span style="font-size: 0.82rem; color: var(--text-secondary);">Missing Keywords:</span>
                        <div class="skill-tags" style="margin-top: 6px;">
                            ${ko.missing_keywords.map(k => `<span class="skill-tag skill-tag-missing">${k}</span>`).join('')}
                        </div>
                    </div>
                ` : ''}
                ${ko.suggested_additions && ko.suggested_additions.length > 0 ? `
                    <div>
                        <span style="font-size: 0.82rem; color: var(--text-secondary);">Suggested Phrases to Add:</span>
                        <ul class="sw-list" style="margin-top: 6px;">
                            ${ko.suggested_additions.map(s => `<li><span style="color: var(--accent-tertiary)">+</span> ${s}</li>`).join('')}
                        </ul>
                    </div>
                ` : ''}
            </div>
        `;
    }

    container.innerHTML = html;
}

function renderAISuggestions(ai) {
    if (!ai.cv_improvement_suggestions || ai.cv_improvement_suggestions.length === 0) {
        document.getElementById('suggestionsPanel').style.display = 'none';
        return;
    }

    const container = document.getElementById('suggestions');
    container.innerHTML = ai.cv_improvement_suggestions.map(s => `
        <div class="suggestion-item">
            <div class="suggestion-section">📝 ${s.section || 'General'}</div>
            ${s.current_issue ? `<div class="suggestion-issue">⚠️ ${s.current_issue}</div>` : ''}
            <div class="suggestion-text">💡 ${s.suggestion}</div>
            ${s.example ? `<div class="suggestion-example">✍️ Example: "${s.example}"</div>` : ''}
        </div>
    `).join('');

    document.getElementById('suggestionsPanel').style.display = 'block';
}

function renderATSCheck(ai) {
    if (!ai.ats_compatibility) {
        document.getElementById('atsPanel').style.display = 'none';
        return;
    }

    const ats = ai.ats_compatibility;
    const score = ats.score || 0;
    const color = getScoreColor(score);

    let html = `
        <div class="ats-score-bar">
            <span class="ats-score-value" style="color: ${color}">${score}%</span>
            <div class="ats-bar-container">
                <div class="ats-bar-fill" style="width: ${score}%; background: linear-gradient(90deg, ${color}, ${color}aa);"></div>
            </div>
        </div>
    `;

    if (ats.issues && ats.issues.length > 0) {
        html += `
            <div style="margin-bottom: 16px;">
                <div class="sw-title">⚠️ Issues to Fix</div>
                <ul class="ats-list">
                    ${ats.issues.map(i => `<li><span style="color: var(--warning)">●</span> ${i}</li>`).join('')}
                </ul>
            </div>
        `;
    }

    if (ats.tips && ats.tips.length > 0) {
        html += `
            <div>
                <div class="sw-title">💡 Optimization Tips</div>
                <ul class="ats-list">
                    ${ats.tips.map(t => `<li><span style="color: var(--success)">✓</span> ${t}</li>`).join('')}
                </ul>
            </div>
        `;
    }

    document.getElementById('atsContent').innerHTML = html;
    document.getElementById('atsPanel').style.display = 'block';
}

function renderCourses(ai) {
    if (!ai.recommended_courses || ai.recommended_courses.length === 0) {
        document.getElementById('coursesPanel').style.display = 'none';
        return;
    }

    const platformIcons = {
        'Coursera': '🎓',
        'Udemy': '📺',
        'edX': '🏛️',
        'YouTube': '▶️',
        'LinkedIn Learning': '💼',
        'Pluralsight': '📘',
    };

    const container = document.getElementById('coursesContent');
    container.innerHTML = ai.recommended_courses.map(c => {
        const icon = platformIcons[c.platform] || '📚';
        return `
            <div class="course-item">
                <div class="course-icon">${icon}</div>
                <div class="course-info">
                    <div class="course-name">${c.name}</div>
                    <div class="course-platform">${c.platform || 'Online'}</div>
                    <div class="course-reason">${c.reason || ''}</div>
                </div>
            </div>
        `;
    }).join('');

    document.getElementById('coursesPanel').style.display = 'block';
}

// ====== Batch Results ======
function renderBatchResults(results) {
    hideLoading();

    // Sort by overall score descending
    results.sort((a, b) => {
        const scoreA = a.final_scores?.overall || 0;
        const scoreB = b.final_scores?.overall || 0;
        return scoreB - scoreA;
    });

    const container = document.getElementById('batchResults');
    container.innerHTML = results.map((r, i) => {
        const scores = r.final_scores || {};
        const overall = Math.round(scores.overall || 0);
        const color = scores.color || getScoreColor(overall);
        const level = scores.match_level || '—';

        const dims = ['hard_skills', 'experience', 'education', 'language', 'soft_skills', 'culture_fit'];

        return `
            <div class="batch-card fade-in-up delay-${i + 1}" onclick="viewBatchDetail('${r.jd_id}')">
                <div class="batch-rank ${i < 3 ? 'rank-' + (i + 1) : ''}">#${i + 1}</div>
                <div class="batch-info">
                    <div class="batch-name">${r.jd_name || r.jd_id}</div>
                    <div class="batch-detail">
                        ${r.ai_available ? '🤖 AI + ML' : '📊 ML Only'}
                        ${r.error ? ` • ⚠️ ${r.error}` : ''}
                    </div>
                    <div class="batch-mini-bars">
                        ${dims.map(d => {
                            const s = scores[d] || 0;
                            return `<div class="batch-mini-bar"><div class="batch-mini-fill" style="width: ${s}%; background: ${getScoreColor(s)};"></div></div>`;
                        }).join('')}
                    </div>
                </div>
                <div class="batch-score">
                    <div class="batch-score-value" style="color: ${color}">${overall}%</div>
                    <div class="batch-score-label" style="background: ${color}22; color: ${color}; border: 1px solid ${color}44">${level}</div>
                </div>
            </div>
        `;
    }).join('');

    document.getElementById('batchSection').style.display = 'block';
    document.getElementById('batchSection').scrollIntoView({ behavior: 'smooth' });
}

function viewBatchDetail(jdId) {
    // Select this JD and run single analysis
    state.selectedJdId = jdId;
    document.querySelectorAll('.jd-tab').forEach(t => {
        t.classList.toggle('active', t.dataset.jdId === jdId);
    });
    document.getElementById('batchSection').style.display = 'none';
    analyzeSingle();
}

// ====== Utility Functions ======
function getScoreColor(score) {
    if (score >= 75) return '#22c55e';
    if (score >= 50) return '#f59e0b';
    return '#ef4444';
}

function getPriorityClass(priority) {
    if (!priority) return 'nice';
    const p = priority.toLowerCase();
    if (p.includes('required') || p.includes('bắt buộc')) return 'required';
    if (p.includes('preferred') || p.includes('ưu tiên')) return 'preferred';
    return 'nice';
}

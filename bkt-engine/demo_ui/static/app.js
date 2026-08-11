let questions = [];
let currentIndex = 0;
let answers = [];
let questionStartTime = 0;

const DIAGNOSTIC_COUNT = 5;

// DOM Elements
const screens = {
    start: document.getElementById('start-screen'),
    quiz: document.getElementById('quiz-screen'),
    results: document.getElementById('results-screen')
};

const ui = {
    startBtn: document.getElementById('start-btn'),
    simulateBtn: document.getElementById('simulate-struggling-btn'),
    restartBtn: document.getElementById('restart-btn'),
    questionText: document.getElementById('question-text'),
    optionsContainer: document.getElementById('options-container'),
    progressFill: document.getElementById('progress'),
    questionCounter: document.getElementById('question-counter'),
    phaseBadge: document.getElementById('phase-badge'),
    thetaScore: document.getElementById('theta-score'),
    masteryBars: document.getElementById('mastery-bars'),
    suggestionsList: document.getElementById('suggestions-list'),
    tutorPanel: document.getElementById('tutor-panel'),
    tutorInterventionsList: document.getElementById('tutor-interventions-list')
};

// Event Listeners
ui.startBtn.addEventListener('click', startQuiz);
ui.simulateBtn.addEventListener('click', simulateStruggling);
ui.restartBtn.addEventListener('click', resetQuiz);

async function fetchQuestions() {
    try {
        const res = await fetch('/api/quiz');
        questions = await res.json();
    } catch (err) {
        console.error("Failed to load questions:", err);
        alert("Failed to load quiz data. Ensure the backend is running.");
    }
}

async function simulateStruggling() {
    ui.simulateBtn.textContent = 'Simulating...';
    if (questions.length === 0) {
        await fetchQuestions();
    }
    
    if (questions.length === 0) return;
    
    // Create synthetic answers that drop mastery across concepts
    answers = questions.map((q, i) => {
        const isCorrect = i < 3;
        // Pick a wrong option for incorrect answers
        const optKeys = Object.keys(q.options || {});
        const wrongOpts = optKeys.filter(k => k !== q.correct_answer);
        const selected = isCorrect ? q.correct_answer : (wrongOpts[0] || 'A');
        return {
            question_id: q.id,
            is_correct: isCorrect,
            response_time_ms: 12000,
            selected_option: selected
        }
    });
    
    submitQuiz();
}

function showScreen(screenName) {
    Object.values(screens).forEach(s => s.classList.remove('active'));
    screens[screenName].classList.add('active');
}

async function startQuiz() {
    ui.startBtn.textContent = 'Loading...';
    if (questions.length === 0) {
        await fetchQuestions();
    }
    
    if (questions.length === 0) return;
    
    answers = [];
    currentIndex = 0;
    
    showScreen('quiz');
    renderQuestion();
}

function renderQuestion() {
    const q = questions[currentIndex];
    
    // Update header
    const isDiagnostic = currentIndex < DIAGNOSTIC_COUNT;
    ui.phaseBadge.textContent = isDiagnostic ? 'Diagnostic' : 'Practice';
    ui.phaseBadge.style.background = isDiagnostic ? 'rgba(236, 72, 153, 0.2)' : 'rgba(99, 102, 241, 0.2)';
    ui.phaseBadge.style.color = isDiagnostic ? '#f472b6' : '#818cf8';
    
    ui.questionCounter.textContent = `Question ${currentIndex + 1}/${questions.length}`;
    ui.progressFill.style.width = `${((currentIndex) / questions.length) * 100}%`;
    
    // Render question
    ui.questionText.textContent = q.text || 'Question missing';
    
    ui.optionsContainer.innerHTML = '';
    
    const opts = q.options || {};
    Object.keys(opts).forEach(key => {
        if (!opts[key]) return;
        const btn = document.createElement('button');
        btn.className = 'option-btn';
        btn.textContent = `${key}. ${opts[key]}`;
        btn.onclick = () => handleAnswer(key, btn);
        ui.optionsContainer.appendChild(btn);
    });
    
    questionStartTime = Date.now();
}

function handleAnswer(selectedOption, btn) {
    // Disable all options
    const allBtns = document.querySelectorAll('.option-btn');
    allBtns.forEach(b => b.disabled = true);
    
    btn.classList.add('selected');
    
    const rt = Date.now() - questionStartTime;
    const q = questions[currentIndex];
    const isCorrect = (selectedOption === q.correct_answer);
    
    answers.push({
        question_id: q.id,
        is_correct: isCorrect,
        response_time_ms: rt,
        selected_option: selectedOption
    });
    
    setTimeout(() => {
        currentIndex++;
        if (currentIndex < questions.length) {
            renderQuestion();
        } else {
            submitQuiz();
        }
    }, 400); // slight delay for visual feedback
}

async function submitQuiz() {
    ui.progressFill.style.width = '100%';
    ui.questionText.textContent = "Analyzing your responses...";
    ui.optionsContainer.innerHTML = '';
    
    const diagnostic = answers.slice(0, DIAGNOSTIC_COUNT);
    const practice = answers.slice(DIAGNOSTIC_COUNT);
    
    try {
        const res = await fetch('/api/quiz/submit', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                diagnostic_answers: diagnostic,
                practice_answers: practice
            })
        });
        const data = await res.json();
        renderResults(data);
    } catch (err) {
        console.error("Submission failed:", err);
        alert("Failed to calculate results.");
        resetQuiz();
    }
}

function getMasteryColor(score) {
    if (score >= 0.90) return 'var(--success)';
    if (score >= 0.70) return 'var(--primary-color)';
    if (score >= 0.50) return 'var(--warning)';
    return 'var(--danger)';
}

function renderResults(data) {
    showScreen('results');
    
    // Format theta nicely
    ui.thetaScore.textContent = data.theta.toFixed(2);
    
    // Render Mastery Bars
    ui.masteryBars.innerHTML = '';
    const summary = data.summary || {};
    
    Object.keys(summary).forEach(cid => {
        const score = summary[cid].p_l || 0;
        const pct = Math.round(score * 100);
        // Use concept_names map from backend, fallback to concept ID
        const nameMap = data.concept_names || {};
        const displayName = nameMap[cid] || `Concept ${cid}`;
        
        const row = document.createElement('div');
        row.className = 'mastery-row';
        
        row.innerHTML = `
            <div class="mastery-label">
                <span>${displayName} <span style="opacity:0.5">(${cid})</span></span>
                <span>${pct}%</span>
            </div>
            <div class="mastery-bar-bg">
                <div class="mastery-bar-fill" style="width: 0%; background: ${getMasteryColor(score)}"></div>
            </div>
        `;
        ui.masteryBars.appendChild(row);
        
        // Trigger animation
        setTimeout(() => {
            row.querySelector('.mastery-bar-fill').style.width = `${pct}%`;
        }, 100);
    });
    
    // Render Suggestions
    ui.suggestionsList.innerHTML = '';
    (data.suggestions || []).forEach(sugg => {
        const li = document.createElement('li');
        li.textContent = sugg;
        
        if (sugg.toLowerCase().includes("mastery")) li.style.borderLeftColor = 'var(--success)';
        else if (sugg.toLowerCase().includes("review") || sugg.toLowerCase().includes("failing")) li.style.borderLeftColor = 'var(--danger)';
        else li.style.borderLeftColor = 'var(--warning)';
        
        ui.suggestionsList.appendChild(li);
    });
    
    // Render AI Tutor Interventions
    if (data.tutoring_interventions && data.tutoring_interventions.length > 0) {
        ui.tutorPanel.style.display = 'block';
        ui.tutorInterventionsList.innerHTML = '';
        
        data.tutoring_interventions.forEach(intervention => {
            const card = document.createElement('div');
            card.className = 'intervention-card';
            const conceptLabel = intervention.concept_name || `Concept ${intervention.concept_id}`;
            const masteryPct = intervention.mastery_at_turn ? `${Math.round(intervention.mastery_at_turn * 100)}%` : '';
            const misconception = intervention.misconception_tag 
                ? `<span class="misconception-badge">${intervention.misconception_tag.replace(/_/g, ' ')}</span>` 
                : '';
            
            card.innerHTML = `
                <div class="intervention-header">
                    <span class="concept-badge">${conceptLabel} (${intervention.concept_id})</span>
                    ${masteryPct ? `<span class="mastery-micro-badge">${masteryPct} mastery</span>` : ''}
                    ${misconception}
                </div>
                <div class="chat-bubbles">
                    <div class="chat-bubble student-bubble">
                        <div class="bubble-label">You</div>
                        <div class="bubble-text">${intervention.student_answer}</div>
                    </div>
                    <div class="chat-bubble ai-bubble">
                        <div class="bubble-label">AI Tutor</div>
                        <div class="bubble-text">${intervention.tutor_response}</div>
                    </div>
                </div>
            `;
            ui.tutorInterventionsList.appendChild(card);
        });
    } else {
        ui.tutorPanel.style.display = 'none';
    }
    
    // Render Graph Trace if available
    const graphPanel = document.getElementById('graph-panel');
    const container = document.getElementById('kg-network');
    
    if (data.tracing_results && Object.keys(data.tracing_results).length > 0) {
        graphPanel.style.display = 'block';
        
        // Build nodes and edges for vis.js
        let nodesMap = new Map();
        let edges = [];
        
        Object.keys(data.tracing_results).forEach(targetCid => {
            if (!nodesMap.has(targetCid)) {
                nodesMap.set(targetCid, { id: targetCid, label: targetCid, color: { background: '#ef4444', border: '#b91c1c' }, font: { color: 'white' }, shape: 'box' });
            }
            
            const traces = data.tracing_results[targetCid];
            traces.forEach(trace => {
                const pCid = trace.concept_id;
                const fCid = trace.failed_for;
                
                if (!nodesMap.has(pCid)) {
                    nodesMap.set(pCid, { id: pCid, label: `${pCid}\n(Mastery: ${(trace.mastery * 100).toFixed(0)}%)`, color: { background: '#f59e0b', border: '#d97706' }, font: { color: 'white' }, shape: 'box' });
                }
                
                edges.push({ from: pCid, to: fCid, arrows: 'to', color: { color: 'rgba(255,255,255,0.4)' }, dashes: true });
            });
        });
        
        const networkData = {
            nodes: new vis.DataSet(Array.from(nodesMap.values())),
            edges: new vis.DataSet(edges)
        };
        const options = {
            layout: { hierarchical: { direction: 'UD', sortMethod: 'directed' } },
            physics: false,
            interaction: { dragNodes: false, dragView: true, zoomView: true }
        };
        new vis.Network(container, networkData, options);
    } else {
        graphPanel.style.display = 'none';
    }
}

function resetQuiz() {
    ui.startBtn.textContent = 'Start Assessment';
    if(ui.simulateBtn) ui.simulateBtn.textContent = 'Simulate Struggling Student';
    showScreen('start');
}

// Preload questions silently on init
fetchQuestions();

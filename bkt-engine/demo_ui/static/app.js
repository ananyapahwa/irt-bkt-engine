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
    restartBtn: document.getElementById('restart-btn'),
    questionText: document.getElementById('question-text'),
    optionsContainer: document.getElementById('options-container'),
    progressFill: document.getElementById('progress'),
    questionCounter: document.getElementById('question-counter'),
    phaseBadge: document.getElementById('phase-badge'),
    thetaScore: document.getElementById('theta-score'),
    masteryBars: document.getElementById('mastery-bars'),
    suggestionsList: document.getElementById('suggestions-list')
};

// Event Listeners
ui.startBtn.addEventListener('click', startQuiz);
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
        response_time_ms: rt
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
        
        const row = document.createElement('div');
        row.className = 'mastery-row';
        
        row.innerHTML = `
            <div class="mastery-label">
                <span>Concept ${cid}</span>
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
        else if (sugg.toLowerCase().includes("review")) li.style.borderLeftColor = 'var(--danger)';
        else li.style.borderLeftColor = 'var(--warning)';
        
        ui.suggestionsList.appendChild(li);
    });
}

function resetQuiz() {
    ui.startBtn.textContent = 'Start Assessment';
    showScreen('start');
}

// Preload questions silently on init
fetchQuestions();

// EmotionSense AI Client-side Application

document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const textInput = document.getElementById('text-input');
    const charCounter = document.getElementById('char-counter');
    const wordCounter = document.getElementById('word-counter');
    const clearBtn = document.getElementById('clear-btn');
    const analyzeBtn = document.getElementById('analyze-btn');
    
    const emptyState = document.getElementById('empty-state');
    const resultContent = document.getElementById('result-content');
    const dominantCard = document.getElementById('dominant-card');
    const dominantEmoji = document.getElementById('dominant-emoji');
    const dominantName = document.getElementById('dominant-name');
    const dominantDesc = document.getElementById('dominant-desc');
    const confidenceVal = document.getElementById('confidence-val');
    const breakdownBars = document.getElementById('breakdown-bars');
    const nlpRaw = document.getElementById('nlp-raw');
    const nlpCleaned = document.getElementById('nlp-cleaned');
    
    const sampleChips = document.querySelectorAll('.chip');
    const historyList = document.getElementById('history-list');
    const clearHistoryBtn = document.getElementById('clear-history-btn');

    let historyItems = JSON.parse(localStorage.getItem('emotionsense_history') || '[]');

    // Character and Word Counter
    function updateCounters() {
        const val = textInput.value;
        const charCount = val.length;
        charCounter.textContent = `${charCount} / 1000 characters`;

        const words = val.trim().split(/\s+/).filter(w => w.length > 0);
        wordCounter.textContent = `${words.length} word${words.length === 1 ? '' : 's'}`;
    }

    textInput.addEventListener('input', updateCounters);

    // Clear input
    clearBtn.addEventListener('click', () => {
        textInput.value = '';
        updateCounters();
        textInput.focus();
    });

    // Sample Chips Click
    sampleChips.forEach(chip => {
        chip.addEventListener('click', () => {
            const text = chip.getAttribute('data-text');
            if (text) {
                textInput.value = text;
                updateCounters();
                analyzeEmotion(text);
            }
        });
    });

    // Keyboard shortcut (Ctrl + Enter or Cmd + Enter)
    textInput.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            const text = textInput.value.trim();
            if (text) {
                analyzeEmotion(text);
            }
        }
    });

    analyzeBtn.addEventListener('click', () => {
        const text = textInput.value.trim();
        if (!text) {
            textInput.focus();
            return;
        }
        analyzeEmotion(text);
    });

    // Analyze Emotion API call
    async function analyzeEmotion(text) {
        setLoading(true);

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text })
            });

            const data = await response.json();

            if (!response.ok || data.status !== 'success') {
                throw new Error(data.message || 'Failed to detect emotion.');
            }

            renderResults(data);
            addToHistory(data);
        } catch (error) {
            console.error('Emotion detection failed:', error);
            alert(`Error: ${error.message}`);
        } finally {
            setLoading(false);
        }
    }

    function setLoading(isLoading) {
        if (isLoading) {
            analyzeBtn.classList.add('loading');
            analyzeBtn.disabled = true;
        } else {
            analyzeBtn.classList.remove('loading');
            analyzeBtn.disabled = false;
        }
    }

    // Render Results
    function renderResults(data) {
        // Show result view
        emptyState.classList.add('hidden');
        resultContent.classList.remove('hidden');

        // Dominant Emotion Card styling
        dominantEmoji.textContent = data.emoji;
        dominantName.textContent = data.predicted_emotion;
        dominantDesc.textContent = data.description;
        confidenceVal.textContent = `${data.confidence}%`;
        
        dominantCard.style.borderColor = data.color;
        dominantCard.style.boxShadow = `0 8px 30px -5px ${data.color}33`;

        // Breakdown bars
        breakdownBars.innerHTML = '';
        data.breakdown.forEach((item, index) => {
            const row = document.createElement('div');
            row.className = 'bar-row';
            row.innerHTML = `
                <div class="bar-label">
                    <span>${item.emoji}</span>
                    <span>${item.emotion}</span>
                </div>
                <div class="bar-track">
                    <div class="bar-fill" id="bar-fill-${index}" style="background: linear-gradient(90deg, ${item.color}cc, ${item.color}); width: 0%;"></div>
                </div>
                <div class="bar-pct">${item.percentage}%</div>
            `;
            breakdownBars.appendChild(row);

            // Animate width
            setTimeout(() => {
                const fillEl = document.getElementById(`bar-fill-${index}`);
                if (fillEl) {
                    fillEl.style.width = `${Math.max(item.percentage, 2)}%`;
                }
            }, 50 * index);
        });

        // NLP inspection
        nlpRaw.textContent = data.raw_text;
        nlpCleaned.textContent = data.cleaned_text || '(All words filtered as stopwords)';
    }

    // History Management
    function addToHistory(data) {
        const item = {
            id: Date.now(),
            text: data.raw_text,
            emotion: data.predicted_emotion,
            emoji: data.emoji,
            color: data.color,
            confidence: data.confidence
        };

        historyItems.unshift(item);
        if (historyItems.length > 8) {
            historyItems = historyItems.slice(0, 8);
        }

        localStorage.setItem('emotionsense_history', JSON.stringify(historyItems));
        renderHistory();
    }

    function renderHistory() {
        if (historyItems.length === 0) {
            historyList.innerHTML = '<p class="history-empty">Predictions you run in this session will appear here.</p>';
            return;
        }

        historyList.innerHTML = '';
        historyItems.forEach(item => {
            const el = document.createElement('div');
            el.className = 'history-item';
            el.innerHTML = `
                <span class="history-text" title="${item.text}">"${item.text}"</span>
                <span class="history-badge" style="background: ${item.color}22; color: ${item.color}; border: 1px solid ${item.color}55;">
                    ${item.emoji} ${item.emotion} (${item.confidence}%)
                </span>
            `;

            el.addEventListener('click', () => {
                textInput.value = item.text;
                updateCounters();
                analyzeEmotion(item.text);
            });

            historyList.appendChild(el);
        });
    }

    clearHistoryBtn.addEventListener('click', () => {
        historyItems = [];
        localStorage.removeItem('emotionsense_history');
        renderHistory();
    });

    // Initialize history from storage
    renderHistory();
});

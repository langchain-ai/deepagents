/**
 * AI Slide Creator - Deep Agents Demo
 * WebSocket-based chat interface with real-time slideshow preview
 */

// ============================================
// State
// ============================================

const state = {
    sessionId: generateSessionId(),
    ws: null,
    connected: false,
    currentSlideshow: null,
    currentSlideIndex: 0,
    slides: [],
    isGenerating: false,
};

function generateSessionId() {
    return 'session_' + Math.random().toString(36).substr(2, 9);
}

// ============================================
// DOM Elements
// ============================================

const elements = {
    chatMessages: document.getElementById('chatMessages'),
    chatInput: document.getElementById('chatInput'),
    sendBtn: document.getElementById('sendBtn'),
    statusIndicator: document.getElementById('statusIndicator'),
    emptyState: document.getElementById('emptyState'),
    slideshowViewer: document.getElementById('slideshowViewer'),
    currentSlideImage: document.getElementById('currentSlideImage'),
    slideLoading: document.getElementById('slideLoading'),
    slidePlaceholder: document.getElementById('slidePlaceholder'),
    currentSlideNum: document.getElementById('currentSlideNum'),
    totalSlides: document.getElementById('totalSlides'),
    slideDots: document.getElementById('slideDots'),
    slideTitle: document.getElementById('slideTitle'),
    slideDescription: document.getElementById('slideDescription'),
    prevBtn: document.getElementById('prevBtn'),
    nextBtn: document.getElementById('nextBtn'),
    editInput: document.getElementById('editInput'),
    editBtn: document.getElementById('editBtn'),
    progressPanel: document.getElementById('progressPanel'),
    progressList: document.getElementById('progressList'),
    exportBtn: document.getElementById('exportBtn'),
    newBtn: document.getElementById('newBtn'),
    toolToast: document.getElementById('toolToast'),
    toolToastTitle: document.getElementById('toolToastTitle'),
    toolToastSubtitle: document.getElementById('toolToastSubtitle'),
};

// ============================================
// WebSocket Connection
// ============================================

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/${state.sessionId}`;

    state.ws = new WebSocket(wsUrl);

    state.ws.onopen = () => {
        state.connected = true;
        updateStatus('Ready');
        elements.sendBtn.disabled = false;
        console.log('WebSocket connected');
    };

    state.ws.onclose = () => {
        state.connected = false;
        updateStatus('Disconnected');
        elements.sendBtn.disabled = true;
        console.log('WebSocket disconnected');

        // Attempt to reconnect after 3 seconds
        setTimeout(connectWebSocket, 3000);
    };

    state.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        updateStatus('Error');
    };

    state.ws.onmessage = (event) => {
        const message = JSON.parse(event.data);
        handleMessage(message);
    };
}

// ============================================
// Message Handling
// ============================================

function handleMessage(message) {
    switch (message.type) {
        case 'thinking':
            updateStatus('Thinking', true);
            showTypingIndicator();
            break;

        case 'stream':
            removeTypingIndicator();
            appendToLastAssistantMessage(message.content);
            break;

        case 'done':
            updateStatus('Ready');
            state.isGenerating = false;
            break;

        case 'tool_start':
            showToolToast(message.tool, 'Running...');
            addToolMessage(`Running ${formatToolName(message.tool)}...`);
            break;

        case 'tool_end':
            hideToolToast();
            updateToolMessage(message.tool, message.output);
            break;

        case 'slideshow_created':
            handleSlideshowCreated(message.data);
            break;

        case 'slide_generated':
            handleSlideGenerated(message.data);
            break;

        case 'error':
            removeTypingIndicator();
            addAssistantMessage(`Error: ${message.content}`);
            updateStatus('Ready');
            state.isGenerating = false;
            break;
    }
}

function formatToolName(name) {
    return name
        .replace('tool_', '')
        .replace(/_/g, ' ')
        .replace(/\b\w/g, c => c.toUpperCase());
}

// ============================================
// Chat UI
// ============================================

function addUserMessage(content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message user';
    messageDiv.innerHTML = `<div class="message-content"><p>${escapeHtml(content)}</p></div>`;
    elements.chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

function addAssistantMessage(content) {
    removeTypingIndicator();
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    messageDiv.innerHTML = `<div class="message-content">${formatMarkdown(content)}</div>`;
    elements.chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

function appendToLastAssistantMessage(content) {
    let lastMessage = elements.chatMessages.querySelector('.message.assistant:last-child');

    if (!lastMessage || lastMessage.classList.contains('complete')) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message assistant streaming';
        messageDiv.innerHTML = `<div class="message-content"></div>`;
        elements.chatMessages.appendChild(messageDiv);
        lastMessage = messageDiv;
    }

    const contentDiv = lastMessage.querySelector('.message-content');
    contentDiv.innerHTML = formatMarkdown(contentDiv.textContent + content);
    scrollToBottom();
}

function addToolMessage(content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message tool';
    messageDiv.innerHTML = `
        <div class="message-content">
            <span class="tool-icon">⚡</span>
            <span>${escapeHtml(content)}</span>
        </div>
    `;
    elements.chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

function updateToolMessage(tool, output) {
    // For now, just log the output
    console.log('Tool output:', tool, output);
}

function showTypingIndicator() {
    if (document.querySelector('.typing-indicator')) return;

    const indicator = document.createElement('div');
    indicator.className = 'typing-indicator';
    indicator.innerHTML = '<span></span><span></span><span></span>';
    elements.chatMessages.appendChild(indicator);
    scrollToBottom();
}

function removeTypingIndicator() {
    const indicator = document.querySelector('.typing-indicator');
    if (indicator) indicator.remove();
}

function scrollToBottom() {
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
}

function updateStatus(text, thinking = false) {
    elements.statusIndicator.textContent = text;
    elements.statusIndicator.className = thinking ? 'status-indicator thinking' : 'status-indicator';
}

// ============================================
// Slideshow UI
// ============================================

function handleSlideshowCreated(data) {
    state.currentSlideshow = data.slideshow_id;
    state.slides = [];
    state.currentSlideIndex = 0;

    // Show slideshow viewer
    elements.emptyState.style.display = 'none';
    elements.slideshowViewer.style.display = 'flex';
    elements.progressPanel.style.display = 'block';
    elements.exportBtn.disabled = false;

    // Fetch slideshow details
    fetchSlideshow(data.slideshow_id);
}

function handleSlideGenerated(data) {
    const slideIndex = data.slide_index - 1; // Convert to 0-based

    if (state.slides[slideIndex]) {
        state.slides[slideIndex].generated = true;
        state.slides[slideIndex].image_url = data.image_url;
    }

    updateProgressList();
    updateSlideDots();

    // If this is the current slide, show it
    if (slideIndex === state.currentSlideIndex) {
        showSlide(slideIndex);
    }
}

async function fetchSlideshow(slideshowId) {
    try {
        const response = await fetch(`/api/slideshow/${slideshowId}`);
        const data = await response.json();

        state.slides = data.slides;
        updateSlideshowUI();
    } catch (error) {
        console.error('Error fetching slideshow:', error);
    }
}

function updateSlideshowUI() {
    elements.totalSlides.textContent = state.slides.length;
    updateSlideDots();
    updateProgressList();
    showSlide(0);
}

function updateSlideDots() {
    elements.slideDots.innerHTML = '';

    state.slides.forEach((slide, index) => {
        const dot = document.createElement('div');
        dot.className = 'slide-dot';

        if (index === state.currentSlideIndex) {
            dot.classList.add('active');
        }

        if (slide.generated) {
            dot.classList.add('generated');
        }

        dot.addEventListener('click', () => showSlide(index));
        elements.slideDots.appendChild(dot);
    });
}

function updateProgressList() {
    elements.progressList.innerHTML = '';

    state.slides.forEach((slide, index) => {
        const item = document.createElement('div');
        item.className = 'progress-item';

        let icon, statusClass;
        if (slide.generated) {
            icon = '✓';
            statusClass = 'completed';
        } else if (state.isGenerating && index === state.slides.findIndex(s => !s.generated)) {
            icon = '◐';
            statusClass = 'generating';
        } else {
            icon = '○';
            statusClass = 'pending';
        }

        item.classList.add(statusClass);
        item.innerHTML = `
            <span class="progress-item-icon">${icon}</span>
            <span>Slide ${index + 1}: ${escapeHtml(slide.title)}</span>
        `;

        elements.progressList.appendChild(item);
    });

    // Hide progress panel if all slides are generated
    const allGenerated = state.slides.every(s => s.generated);
    if (allGenerated && state.slides.length > 0) {
        elements.progressPanel.style.display = 'none';
    }
}

function showSlide(index) {
    if (index < 0 || index >= state.slides.length) return;

    state.currentSlideIndex = index;
    const slide = state.slides[index];

    // Update counter
    elements.currentSlideNum.textContent = index + 1;

    // Update info
    elements.slideTitle.textContent = slide.title;
    elements.slideDescription.textContent = slide.content_description || '';

    // Update navigation
    elements.prevBtn.disabled = index === 0;
    elements.nextBtn.disabled = index === state.slides.length - 1;

    // Update dots
    document.querySelectorAll('.slide-dot').forEach((dot, i) => {
        dot.classList.toggle('active', i === index);
    });

    // Show image or placeholder
    if (slide.generated && slide.image_url) {
        elements.slidePlaceholder.style.display = 'none';
        elements.slideLoading.classList.remove('active');
        elements.currentSlideImage.classList.remove('loaded');

        elements.currentSlideImage.onload = () => {
            elements.currentSlideImage.classList.add('loaded');
        };

        // Add cache buster for edited images
        elements.currentSlideImage.src = slide.image_url;
    } else {
        elements.currentSlideImage.classList.remove('loaded');
        elements.slidePlaceholder.style.display = 'flex';
    }
}

function nextSlide() {
    if (state.currentSlideIndex < state.slides.length - 1) {
        showSlide(state.currentSlideIndex + 1);
    }
}

function prevSlide() {
    if (state.currentSlideIndex > 0) {
        showSlide(state.currentSlideIndex - 1);
    }
}

// ============================================
// Tool Toast
// ============================================

function showToolToast(tool, subtitle = '') {
    elements.toolToastTitle.textContent = formatToolName(tool);
    elements.toolToastSubtitle.textContent = subtitle;
    elements.toolToast.classList.add('visible');
}

function hideToolToast() {
    elements.toolToast.classList.remove('visible');
}

// ============================================
// Send Message
// ============================================

function sendMessage() {
    const content = elements.chatInput.value.trim();
    if (!content || !state.connected) return;

    // Add user message to chat
    addUserMessage(content);

    // Send to WebSocket
    state.ws.send(JSON.stringify({ message: content }));

    // Clear input
    elements.chatInput.value = '';
    autoResizeTextarea();

    state.isGenerating = true;
}

// ============================================
// Edit Slide
// ============================================

function editCurrentSlide() {
    const editPrompt = elements.editInput.value.trim();
    if (!editPrompt || !state.currentSlideshow) return;

    const slideIndex = state.currentSlideIndex + 1; // 1-based

    // Send edit request via chat
    const message = `Edit slide ${slideIndex}: ${editPrompt}`;
    addUserMessage(message);
    state.ws.send(JSON.stringify({ message }));

    // Clear input
    elements.editInput.value = '';

    // Show loading state
    elements.slideLoading.classList.add('active');
    state.isGenerating = true;
}

// ============================================
// New Slideshow
// ============================================

function newSlideshow() {
    // Reset state
    state.currentSlideshow = null;
    state.slides = [];
    state.currentSlideIndex = 0;

    // Reset UI
    elements.emptyState.style.display = 'flex';
    elements.slideshowViewer.style.display = 'none';
    elements.progressPanel.style.display = 'none';
    elements.exportBtn.disabled = true;

    // Generate new session
    state.sessionId = generateSessionId();

    // Reconnect WebSocket
    if (state.ws) {
        state.ws.close();
    }
    connectWebSocket();

    // Clear chat (keep welcome message)
    elements.chatMessages.innerHTML = `
        <div class="message assistant">
            <div class="message-content">
                <p>Hi! I'm your AI slide creator. Tell me what slideshow you'd like to create, and I'll help you design beautiful, consistent slides.</p>
                <p><strong>Try something like:</strong></p>
                <ul>
                    <li>"Create a slideshow about renewable energy"</li>
                    <li>"Make a presentation on the future of AI"</li>
                    <li>"Design slides about space exploration"</li>
                </ul>
            </div>
        </div>
    `;
}

// ============================================
// Export
// ============================================

function exportSlideshow() {
    if (!state.currentSlideshow) return;

    // Open the HTML viewer in a new tab
    window.open(`/output/${state.currentSlideshow}/index.html`, '_blank');
}

// ============================================
// Utilities
// ============================================

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatMarkdown(text) {
    // Basic markdown formatting
    return text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/`(.*?)`/g, '<code>$1</code>')
        .replace(/\n/g, '<br>');
}

function autoResizeTextarea() {
    elements.chatInput.style.height = 'auto';
    elements.chatInput.style.height = Math.min(elements.chatInput.scrollHeight, 120) + 'px';
}

// ============================================
// Event Listeners
// ============================================

// Send message
elements.sendBtn.addEventListener('click', sendMessage);

elements.chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

elements.chatInput.addEventListener('input', () => {
    autoResizeTextarea();
    elements.sendBtn.disabled = !elements.chatInput.value.trim() || !state.connected;
});

// Navigation
elements.prevBtn.addEventListener('click', prevSlide);
elements.nextBtn.addEventListener('click', nextSlide);

// Keyboard navigation
document.addEventListener('keydown', (e) => {
    if (document.activeElement === elements.chatInput ||
        document.activeElement === elements.editInput) {
        return;
    }

    if (e.key === 'ArrowLeft') prevSlide();
    if (e.key === 'ArrowRight') nextSlide();
});

// Edit
elements.editBtn.addEventListener('click', editCurrentSlide);
elements.editInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
        e.preventDefault();
        editCurrentSlide();
    }
});

// Header buttons
elements.newBtn.addEventListener('click', newSlideshow);
elements.exportBtn.addEventListener('click', exportSlideshow);

// ============================================
// Initialize
// ============================================

connectWebSocket();

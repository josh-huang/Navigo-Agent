// ── State ────────────────────────────────────────────────────────────────
let currentThreadId = localStorage.getItem("travel_thread_id") || null;
let latestAnswerMarkdown = "";

// ── Marked Configuration ─────────────────────────────────────────────────
if (typeof marked !== "undefined") {
    marked.setOptions({
        breaks: true,
        gfm: true,
    });
}

// ── Helper: Sanitize HTML ────────────────────────────────────────────────
function safeParseMarkdown(md) {
    if (typeof marked === "undefined") {
        return md; // fallback: plain text
    }
    const rawHtml = marked.parse(md);
    if (typeof DOMPurify !== "undefined") {
        return DOMPurify.sanitize(rawHtml, {
            ALLOWED_TAGS: [
                "p", "br", "strong", "em", "b", "i", "u", "s", "a", "ul", "ol", "li",
                "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre", "code", "hr",
                "table", "thead", "tbody", "tr", "th", "td", "span", "div", "img",
                "sup", "sub", "del", "ins", "mark", "small",
            ],
            ALLOWED_ATTR: ["href", "title", "target", "rel", "src", "alt", "class", "width", "height", "align", "valign"],
        });
    }
    return rawHtml;
}

// ── Quick Prompt ─────────────────────────────────────────────────────────
function setPrompt(text) {
    document.getElementById("userInput").value = text;
}

// ── Loading State ────────────────────────────────────────────────────────
function setLoading(isLoading) {
    const sendBtn = document.getElementById("sendBtn");
    const btnText = document.getElementById("btnText");
    const btnLoader = document.getElementById("btnLoader");

    sendBtn.disabled = isLoading;

    if (isLoading) {
        btnText.classList.add("hidden");
        btnLoader.classList.remove("hidden");
    } else {
        btnText.classList.remove("hidden");
        btnLoader.classList.add("hidden");
    }
}

// ── Error Display ────────────────────────────────────────────────────────
function showError(message) {
    const errorBox = document.getElementById("errorBox");
    errorBox.textContent = message;
    errorBox.classList.remove("hidden");
}

function hideError() {
    const errorBox = document.getElementById("errorBox");
    errorBox.classList.add("hidden");
    errorBox.textContent = "";
}

// ── Agent Progress ───────────────────────────────────────────────────────
function showProgressSection() {
    const progressSection = document.getElementById("progressSection");
    progressSection.classList.remove("hidden");
    document.getElementById("progressList").innerHTML = "";
}

function addProgressItem(agent, status) {
    const progressList = document.getElementById("progressList");
    const item = document.createElement("div");
    item.className = "progress-item";
    item.id = `progress-${agent}`;

    const agentIcons = {
        flight_agent: "✈️",
        hotel_agent: "🏨",
        weather_agent: "🌤️",
        itinerary_agent: "📋",
        supervisor: "🧠",
        intent_classifier: "🔍",
        final_synthesizer: "✨",
    };

    const icon = agentIcons[agent] || "🤖";
    const friendlyName = agent.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());

    item.innerHTML = `<span class="progress-icon">${icon}</span> <span class="progress-name">${friendlyName}</span> <span class="progress-status">${status}</span>`;
    progressList.appendChild(item);
}

function updateProgressItem(agent, status) {
    const item = document.getElementById(`progress-${agent}`);
    if (item) {
        item.querySelector(".progress-status").textContent = status;
    } else {
        addProgressItem(agent, status);
    }
}

// ── Result Display ───────────────────────────────────────────────────────
function showResult(answer, threadId, data) {
    latestAnswerMarkdown = answer;

    const resultSection = document.getElementById("resultSection");
    const resultBox = document.getElementById("resultBox");
    const threadInfo = document.getElementById("threadInfo");

    resultBox.innerHTML = safeParseMarkdown(answer);
    threadInfo.textContent = `Thread ID: ${threadId}`;

    // Populate structured data accordions
    document.getElementById("flightData").textContent = data.flight_results || "No flight data available.";
    document.getElementById("hotelData").textContent = data.hotel_results || "No hotel data available.";
    document.getElementById("weatherData").textContent = data.weather_results || "No weather data available.";

    resultSection.classList.remove("hidden");
    resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ── API Call ─────────────────────────────────────────────────────────────
async function sendMessage() {
    hideError();

    const input = document.getElementById("userInput");
    const message = input.value.trim();

    if (!message) {
        showError("Please enter your travel request first.");
        return;
    }

    setLoading(true);
    showProgressSection();
    addProgressItem("supervisor", "Analyzing request...");

    try {
        const response = await fetch("/api/travel", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: message, thread_id: currentThreadId }),
        });

        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error(data.error || "Something went wrong.");
        }

        currentThreadId = data.thread_id;
        localStorage.setItem("travel_thread_id", currentThreadId);

        updateProgressItem("supervisor", "Done ✅");

        // Simulate progress display for each agent (since we don't have SSE yet)
        const agents = [];
        if (data.flight_results) agents.push("flight_agent");
        if (data.hotel_results) agents.push("hotel_agent");
        if (data.weather_results) agents.push("weather_agent");
        if (data.itinerary) agents.push("itinerary_agent");
        agents.push("final_synthesizer");

        for (let i = 0; i < agents.length; i++) {
            addProgressItem(agents[i], "Done ✅");
        }

        showResult(data.answer, data.thread_id, data);

    } catch (error) {
        showError(error.message);
    } finally {
        setLoading(false);
    }
}

// ── Copy ─────────────────────────────────────────────────────────────────
function copyResult() {
    const resultBox = document.getElementById("resultBox");
    const text = resultBox.innerText;

    if (!text) return;

    navigator.clipboard.writeText(text)
        .then(() => {
            const copyBtn = document.querySelector(".copy-btn");
            const oldText = copyBtn.textContent;
            copyBtn.textContent = "Copied!";
            setTimeout(() => { copyBtn.textContent = oldText; }, 1400);
        })
        .catch(() => { showError("Could not copy result."); });
}

// ── PDF Download ─────────────────────────────────────────────────────────
function downloadPDF() {
    const pdfContent = document.getElementById("pdfContent");

    if (!latestAnswerMarkdown || !pdfContent) {
        showError("No travel plan available to download.");
        return;
    }

    const downloadBtn = document.querySelector(".download-btn");
    const oldText = downloadBtn.textContent;
    downloadBtn.textContent = "Preparing PDF...";
    downloadBtn.disabled = true;

    const options = {
        margin: 0.5,
        filename: "navigo-travel-plan.pdf",
        image: { type: "jpeg", quality: 0.98 },
        html2canvas: { scale: 2, useCORS: true, backgroundColor: "#ffffff" },
        jsPDF: { unit: "in", format: "a4", orientation: "portrait" },
        pagebreak: { mode: ["avoid-all", "css", "legacy"] },
    };

    html2pdf()
        .set(options)
        .from(pdfContent)
        .save()
        .then(() => {
            downloadBtn.textContent = oldText;
            downloadBtn.disabled = false;
        })
        .catch(() => {
            downloadBtn.textContent = oldText;
            downloadBtn.disabled = false;
            showError("Could not download PDF.");
        });
}

// ── Keyboard Shortcut ────────────────────────────────────────────────────
document.addEventListener("keydown", function (event) {
    if (event.ctrlKey && event.key === "Enter") {
        sendMessage();
    }
});

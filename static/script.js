/**
 * Navigo AI — Frontend Script
 *
 * Design notes:
 *  - Script loads with `defer` → DOM is fully parsed when this executes.
 *  - SSE streaming for real-time agent progress via /api/travel/stream.
 *  - Falls back to non-streaming /api/travel if streaming fails.
 *  - All event handlers attached via addEventListener (CSP-safe).
 */

console.log("[Navigo] script.js loaded — binding events…");

// ═══════════════════════════════════════════════════════════════════════════
// State
// ═══════════════════════════════════════════════════════════════════════════

let currentThreadId = localStorage.getItem("navigo_thread_id") || null;
let latestAnswerMarkdown = "";
let activeAbortController = null;

// ═══════════════════════════════════════════════════════════════════════════
// Marked configuration
// ═══════════════════════════════════════════════════════════════════════════

if (typeof marked !== "undefined") {
    marked.setOptions({ breaks: true, gfm: true });
    console.log("[Navigo] marked.js configured");
} else {
    console.warn("[Navigo] marked.js not available — markdown rendering disabled");
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/** Parse markdown → safe HTML via marked + DOMPurify. */
function renderMarkdown(md) {
    if (typeof marked === "undefined") return md;
    var raw = marked.parse(md);
    if (typeof DOMPurify !== "undefined") {
        return DOMPurify.sanitize(raw, {
            ALLOWED_TAGS: [
                "p","br","strong","em","b","i","u","s","a","ul","ol","li",
                "h1","h2","h3","h4","h5","h6","blockquote","pre","code","hr",
                "table","thead","tbody","tr","th","td","span","div","img",
                "sup","sub","del","ins","mark","small",
            ],
            ALLOWED_ATTR: ["href","title","target","rel","src","alt","class","width","height","align","valign"],
        });
    }
    return raw;
}

/** DOM shorthand. */
function $(id) {
    return document.getElementById(id);
}

// ═══════════════════════════════════════════════════════════════════════════
// Quick Prompt
// ═══════════════════════════════════════════════════════════════════════════

function setPrompt(text) {
    var input = $("userInput");
    if (input) input.value = text;
}

// ═══════════════════════════════════════════════════════════════════════════
// Loading state
// ═══════════════════════════════════════════════════════════════════════════

function setLoading(isLoading) {
    var btn   = $("sendBtn");
    var label = $("btnLabel");
    var spin  = $("btnSpinner");

    if (!btn) return;
    btn.disabled = isLoading;

    if (isLoading) {
        if (label) label.classList.add("hidden");
        if (spin)  spin.classList.remove("hidden");
    } else {
        if (label) label.classList.remove("hidden");
        if (spin)  spin.classList.add("hidden");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Error display
// ═══════════════════════════════════════════════════════════════════════════

function showError(message) {
    var box = $("errorBox");
    if (!box) return;
    box.textContent = message;
    box.classList.remove("hidden");
}

function hideError() {
    var box = $("errorBox");
    if (!box) return;
    box.classList.add("hidden");
    box.textContent = "";
}

// ═══════════════════════════════════════════════════════════════════════════
// Agent Progress
// ═══════════════════════════════════════════════════════════════════════════

var AGENT_META = {
    intent_classifier:  { icon: "🤔", label: "Analysing request" },
    supervisor:         { icon: "🧠", label: "Planning next steps" },
    flight_agent:       { icon: "✈️", label: "Searching flights" },
    hotel_agent:        { icon: "🏨", label: "Finding hotels" },
    weather_agent:      { icon: "🌤️", label: "Checking weather" },
    itinerary_agent:    { icon: "🗺️", label: "Building itinerary" },
    final_synthesizer:  { icon: "✨", label: "Finalising plan" },
};

function showProgressSection() {
    var section = $("progressSection");
    if (section) section.classList.remove("hidden");
    var list = $("progressList");
    if (list) list.innerHTML = "";
    var summary = $("progressSummary");
    if (summary) summary.textContent = "";
}

function addProgressItem(agentKey, status) {
    var list = $("progressList");
    if (!list) return;

    var meta = AGENT_META[agentKey] || { icon: "⏳", label: agentKey.replace(/_/g, " ") };
    var isRunning = status !== "Done ✅";

    var el = document.createElement("div");
    el.className = "progress-item";
    el.id = "progress-" + agentKey;
    el.innerHTML =
        '<span class="progress-item__icon">' + meta.icon + '</span>' +
        '<span class="progress-item__name">' + meta.label + '</span>' +
        '<span class="progress-item__status' + (isRunning ? ' progress-item__status--running' : '') + '">' + status + '</span>';

    list.appendChild(el);
}

function updateProgressItem(agentKey, status) {
    var existing = document.getElementById("progress-" + agentKey);
    if (existing) {
        var statusEl = existing.querySelector(".progress-item__status");
        if (statusEl) {
            statusEl.textContent = status;
            if (status === "Done ✅") {
                statusEl.classList.remove("progress-item__status--running");
            }
        }
    } else {
        addProgressItem(agentKey, status);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Result display
// ═══════════════════════════════════════════════════════════════════════════

function showResult(answer, threadId, data) {
    latestAnswerMarkdown = answer;

    var resultBox  = $("resultBox");
    var threadInfo = $("threadInfo");
    var section    = $("resultSection");

    if (resultBox)  resultBox.innerHTML = renderMarkdown(answer);
    if (threadInfo) threadInfo.textContent = "Thread ID: " + threadId;

    // Populate accordion data
    var flightData  = $("flightData");
    var hotelData   = $("hotelData");
    var weatherData = $("weatherData");

    if (flightData)  flightData.textContent  = data.flight_results  || "No flight data available.";
    if (hotelData)   hotelData.textContent   = data.hotel_results   || "No hotel data available.";
    if (weatherData) weatherData.textContent = data.weather_results || "No weather data available.";

    if (section) {
        section.classList.remove("hidden");
        section.scrollIntoView({ behavior: "smooth", block: "start" });
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SSE Streaming — real-time agent progress
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Send a travel request via SSE streaming endpoint.
 * Shows real-time progress as each agent completes.
 */
function sendMessageStream() {
    console.log("[Navigo] sendMessageStream() called");

    hideError();

    var input = $("userInput");
    if (!input) {
        console.error("[Navigo] #userInput not found");
        return;
    }

    var message = input.value.trim();
    if (!message) {
        showError("Please enter your travel request first.");
        return;
    }

    // Cancel any in-flight request
    if (activeAbortController) {
        activeAbortController.abort();
    }
    activeAbortController = new AbortController();

    setLoading(true);
    showProgressSection();

    var completedCount = 0;

    fetch("/api/travel/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: message, thread_id: currentThreadId }),
        signal: activeAbortController.signal,
    })
        .then(function (response) {
            if (!response.ok) {
                return response.json().then(function (err) {
                    throw new Error(err.error || "Server error (" + response.status + ")");
                });
            }

            var reader = response.body.getReader();
            var decoder = new TextDecoder();
            var buffer = "";

            function processStream() {
                return reader.read().then(function (result) {
                    if (result.done) return;

                    buffer += decoder.decode(result.value, { stream: true });

                    // Parse SSE events from buffer
                    var lines = buffer.split("\n");
                    buffer = lines.pop() || "";  // keep incomplete line in buffer

                    var currentEvent = null;

                    for (var i = 0; i < lines.length; i++) {
                        var line = lines[i];

                        if (line.startsWith("event: ")) {
                            currentEvent = line.slice(7).trim();
                        } else if (line.startsWith("data: ")) {
                            var dataStr = line.slice(6);
                            try {
                                var data = JSON.parse(dataStr);
                                handleSSEEvent(currentEvent, data);
                            } catch (_) {
                                // skip malformed JSON
                            }
                            currentEvent = null;
                        }
                        // Empty lines (SSE separators) are ignored
                    }

                    return processStream();  // continue reading
                });
            }

            return processStream();
        })
        .catch(function (err) {
            if (err.name === "AbortError") {
                console.log("[Navigo] stream aborted by user");
                return;
            }
            console.error("[Navigo] stream failed:", err);
            showError(err.message);
        })
        .finally(function () {
            setLoading(false);
            activeAbortController = null;
        });

    /** Handle a single SSE event from the stream. */
    function handleSSEEvent(event, data) {
        switch (event) {
            case "progress":
                // { agent, display, icon, status }
                completedCount++;
                updateProgressItem(data.agent, "Done ✅");
                var summary = $("progressSummary");
                if (summary) summary.textContent = completedCount + " agents completed";
                break;

            case "agent_result":
                // Partial result from a completed agent — update accordion
                if (data.flight_results) {
                    var fd = $("flightData");
                    if (fd) fd.textContent = data.flight_results;
                }
                if (data.hotel_results) {
                    var hd = $("hotelData");
                    if (hd) hd.textContent = data.hotel_results;
                }
                if (data.weather_results) {
                    var wd = $("weatherData");
                    if (wd) wd.textContent = data.weather_results;
                }
                break;

            case "complete":
                // { thread_id, answer, flight_results, hotel_results, weather_results, itinerary, llm_calls }
                currentThreadId = data.thread_id;
                try { localStorage.setItem("navigo_thread_id", currentThreadId); } catch (_) {}
                showResult(data.answer, data.thread_id, data);
                console.log("[Navigo] stream complete — thread:", data.thread_id, "llm_calls:", data.llm_calls);
                break;

            case "error":
                showError(data.message || "An error occurred.");
                break;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Legacy non-streaming send (fallback)
// ═══════════════════════════════════════════════════════════════════════════

function sendMessageLegacy() {
    console.log("[Navigo] sendMessage() called (legacy non-streaming)");

    hideError();

    var input = $("userInput");
    if (!input) {
        console.error("[Navigo] #userInput not found");
        return;
    }

    var message = input.value.trim();
    if (!message) {
        showError("Please enter your travel request first.");
        return;
    }

    setLoading(true);
    showProgressSection();

    addProgressItem("intent_classifier", "Analyzing…");
    addProgressItem("supervisor", "Planning…");

    fetch("/api/travel", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: message, thread_id: currentThreadId }),
    })
        .then(function (res) { return res.json().then(function (data) { return { ok: res.ok, data: data }; }); })
        .then(function (result) {
            var ok   = result.ok;
            var data = result.data;

            if (!ok || !data.success) {
                throw new Error(data.error || "Something went wrong on the server.");
            }

            currentThreadId = data.thread_id;
            try { localStorage.setItem("navigo_thread_id", currentThreadId); } catch (_) {}

            updateProgressItem("intent_classifier", "Done ✅");
            updateProgressItem("supervisor", "Done ✅");

            var agentsRan = [];
            if (data.flight_results)  agentsRan.push("flight_agent");
            if (data.hotel_results)   agentsRan.push("hotel_agent");
            if (data.weather_results) agentsRan.push("weather_agent");
            if (data.itinerary)       agentsRan.push("itinerary_agent");
            agentsRan.push("final_synthesizer");

            for (var i = 0; i < agentsRan.length; i++) {
                updateProgressItem(agentsRan[i], "Done ✅");
            }

            var summary = $("progressSummary");
            if (summary) summary.textContent = agentsRan.length + " agents completed";

            showResult(data.answer, data.thread_id, data);
            console.log("[Navigo] request successful — thread:", data.thread_id);
        })
        .catch(function (err) {
            console.error("[Navigo] request failed:", err);
            showError(err.message);
        })
        .finally(function () {
            setLoading(false);
        });
}

/** Auto-detect: use streaming if ReadableStream is available, fall back to legacy. */
function sendMessage() {
    // SSE streaming is the default for modern browsers
    if (window.ReadableStream && window.TextDecoder) {
        sendMessageStream();
    } else {
        sendMessageLegacy();
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Copy
// ═══════════════════════════════════════════════════════════════════════════

function copyResult() {
    var resultBox = $("resultBox");
    if (!resultBox) return;

    var text = resultBox.innerText;
    if (!text) return;

    navigator.clipboard.writeText(text).then(function () {
        var btn = $("copyBtn");
        if (!btn) return;
        var original = btn.textContent;
        btn.textContent = "✅ Copied!";
        setTimeout(function () { btn.textContent = original; }, 1500);
    }).catch(function () {
        showError("Could not copy result. Check clipboard permissions.");
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// PDF Download
// ═══════════════════════════════════════════════════════════════════════════

function downloadPDF() {
    var pdfContent = $("pdfContent");
    if (!latestAnswerMarkdown || !pdfContent) {
        showError("No travel plan available to download.");
        return;
    }

    var btn = $("downloadBtn");
    if (!btn) return;

    var originalHTML = btn.innerHTML;
    btn.innerHTML = '<span class="btn__spinner"></span> Preparing PDF…';
    btn.disabled = true;

    var opts = {
        margin: 0.5,
        filename: "navigo-travel-plan.pdf",
        image: { type: "jpeg", quality: 0.98 },
        html2canvas: { scale: 2, useCORS: true, backgroundColor: "#ffffff" },
        jsPDF: { unit: "in", format: "a4", orientation: "portrait" },
        pagebreak: { mode: ["avoid-all", "css", "legacy"] },
    };

    html2pdf()
        .set(opts)
        .from(pdfContent)
        .save()
        .then(function () {
            btn.innerHTML = originalHTML;
            btn.disabled = false;
        })
        .catch(function () {
            btn.innerHTML = originalHTML;
            btn.disabled = false;
            showError("Could not download PDF. Try again.");
        });
}

// ═══════════════════════════════════════════════════════════════════════════
// Event Bindings
// ═══════════════════════════════════════════════════════════════════════════

(function bindEvents() {
    console.log("[Navigo] binding event listeners…");

    // 1. Generate button
    var sendBtn = $("sendBtn");
    if (sendBtn) {
        sendBtn.addEventListener("click", sendMessage);
        console.log("[Navigo]   ✓ #sendBtn bound");
    } else {
        console.error("[Navigo]   ✗ #sendBtn NOT FOUND — button will not work!");
    }

    // 2. Quick-prompt chips
    var chips = document.querySelectorAll(".chips button[data-prompt], .quick-prompts button[data-prompt]");
    console.log("[Navigo]   found " + chips.length + " quick-prompt chip(s)");
    chips.forEach(function (btn, i) {
        btn.addEventListener("click", function () {
            setPrompt(this.getAttribute("data-prompt"));
        });
        console.log("[Navigo]   ✓ chip[" + i + "] bound: " + (btn.getAttribute("data-prompt") || "").slice(0, 40) + "…");
    });

    // 3. Copy button
    var copyBtn = $("copyBtn");
    if (copyBtn) {
        copyBtn.addEventListener("click", copyResult);
        console.log("[Navigo]   ✓ #copyBtn bound");
    } else {
        console.warn("[Navigo]   ⚠ #copyBtn not found (hidden until result renders)");
    }

    // 4. Download PDF button
    var downloadBtn = $("downloadBtn");
    if (downloadBtn) {
        downloadBtn.addEventListener("click", downloadPDF);
        console.log("[Navigo]   ✓ #downloadBtn bound");
    } else {
        console.warn("[Navigo]   ⚠ #downloadBtn not found (hidden until result renders)");
    }

    console.log("[Navigo] all event bindings complete ✓");
})();

// ═══════════════════════════════════════════════════════════════════════════
// Keyboard shortcut
// ═══════════════════════════════════════════════════════════════════════════

document.addEventListener("keydown", function (e) {
    if (e.ctrlKey && e.key === "Enter") {
        e.preventDefault();
        sendMessage();
    }
});

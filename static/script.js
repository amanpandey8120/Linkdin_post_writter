// ── Show Toast Notification ──────────────────────────────────────────────
function showToast(message, type) {
    const toast = document.getElementById("toast");
    toast.textContent = message;
    toast.className = "toast show " + (type || "");
    clearTimeout(toast._timer);
    toast._timer = setTimeout(() => {
        toast.className = "toast";
    }, 3000);
}

// ── Update Word Count ────────────────────────────────────────────────────
function updateWordCount() {
    const output = document.getElementById("output");
    const badge = document.getElementById("wordCount");
    const text = output.value.trim();
    const count = text ? text.split(/\s+/).length : 0;
    badge.textContent = count + " word" + (count !== 1 ? "s" : "");
}

// ── Generate Post ────────────────────────────────────────────────────────
async function generatePost() {
    const topic = document.getElementById("topic").value.trim();
    const output = document.getElementById("output");
    const btn = document.getElementById("generateBtn");
    const spinner = document.getElementById("spinner");

    if (!topic) {
        showToast("Please enter a topic.", "error");
        document.getElementById("topic").focus();
        return;
    }

    // Show loading state
    btn.disabled = true;
    btn.querySelector(".btn-text").textContent = "Generating...";
    spinner.style.display = "inline-block";
    output.value = "";
    updateWordCount();

    try {
        const response = await fetch("/generate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                topic: topic,
                tone: document.getElementById("tone").value,
                industry: document.getElementById("industry").value,
                audience: document.getElementById("audience").value,
                post_length: document.getElementById("post_length").value,
                cta: document.getElementById("cta").value,
            }),
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Failed to generate post");
        }

        output.value = data.linkedin_post;
        updateWordCount();
        showToast("Post generated successfully!", "success");

    } catch (error) {
        console.error("Generation error:", error);
        output.value = "Error: " + error.message + "\n\nPlease try again with a different topic or check the server status.";
        showToast(error.message, "error");
    } finally {
        btn.disabled = false;
        btn.querySelector(".btn-text").textContent = "Generate LinkedIn Post";
        spinner.style.display = "none";
    }
}

// ── Copy Post ────────────────────────────────────────────────────────────
async function copyPost() {
    const output = document.getElementById("output");
    const text = output.value.trim();

    if (!text) {
        showToast("Nothing to copy. Generate a post first.", "error");
        return;
    }

    try {
        await navigator.clipboard.writeText(text);
        showToast("Post copied to clipboard!", "success");
    } catch {
        // Fallback
        output.select();
        if (document.execCommand("copy")) {
            showToast("Post copied to clipboard!", "success");
        } else {
            showToast("Failed to copy. Please select and copy manually.", "error");
        }
    }
}

// ── Clear Output ─────────────────────────────────────────────────────────
function clearOutput() {
    document.getElementById("output").value = "";
    updateWordCount();
    showToast("Output cleared.", "");
}

// ── Live word count on input ─────────────────────────────────────────────
document.getElementById("output").addEventListener("input", updateWordCount);

// ── Enter key to generate ────────────────────────────────────────────────
document.getElementById("topic").addEventListener("keydown", function (event) {
    if (event.key === "Enter") {
        event.preventDefault();
        generatePost();
    }
});

/* ============================================================================
   app.js — controller layer: backend I/O (upload, streaming analyze, service
   admin, status polling), DOM event wiring, and boot sequence.
   Loaded last, after state.js and panels.js.
   ========================================================================== */

async function uploadSelectedImage(file) {
    if (!file) return;
    const form = new FormData();
    form.append("file", file);
    if (uploadImageBtn) {
        uploadImageBtn.disabled = true;
        uploadImageBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i> Uploading';
    }
    try {
        const resp = await fetch("/v1/media/upload", { method: "POST", body: form });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);
        imagePathInput.value = data.path || "";
        syncDraftInputsToSession();
        appendMessage("bot", `Image uploaded: ${data.name || ""}<br>Path has been filled into the input box.`, true);
    } catch (error) {
        appendMessage('bot', `Upload failed: ${error.message}`, false);
    } finally {
        if (uploadImageBtn) {
            uploadImageBtn.disabled = false;
            uploadImageBtn.innerHTML = '<i class="fas fa-upload mr-2"></i> Upload Image';
        }
        if (uploadImageInput) uploadImageInput.value = "";
    }
}

async function sendMessage({ resume = false, nearbyRadiusOverride = null } = {}) {
    const text = userInput.value.trim();
    const imagePath = imagePathInput.value.trim();
    const latitude = latitudeInput.value.trim();
    const longitude = longitudeInput.value.trim();
    const agentMode = !!(agentModeToggle && agentModeToggle.checked);
    if (!resume && !text && !imagePath) return;

    if (!resume && imagePath && !agentMode) {
        const pickedThreshold = await chooseReviewThreshold();
        if (pickedThreshold === null) return;
        selectedReviewThreshold = pickedThreshold;
        syncDraftInputsToSession();
    }

    const session = getActiveSession();
    const context = getSessionContext(session);
    if (!Array.isArray(context.agentTrace)) context.agentTrace = [];
    const committedImagePath = String(context.committedImagePath || '');
    const hasDraftGeo = latitude !== '' && longitude !== '';
    const effectiveGeoPoint = hasDraftGeo
        ? { lat: Number(latitude), lon: Number(longitude) }
        : (
            context.committedGeoPoint
            && typeof context.committedGeoPoint.lat === 'number'
            && typeof context.committedGeoPoint.lon === 'number'
        )
            ? { ...context.committedGeoPoint }
            : null;

    if (!resume && imagePath) {
        // A new image turn starts a fresh tool ledger, even when the same path is re-analyzed.
        context.agentTrace = [];
        context.latestReportSummary = '';
        context.reviewThresholdConfirmed = false;
        context.agentTurnsUsed = 0;
        if (imagePath !== committedImagePath && committedImagePath) {
            chatHistory = [];
            appendMessage('bot', 'New image detected. A new analysis session was started automatically.', false);
        }
    }

    // The OSM probe radius is collected lazily in agent mode: the backend
    // pauses and emits `need_nearby_radius` right before geo.*, and the resume
    // call carries the chosen value. Graph mode has no pause point, so it
    // still prompts up front.
    let chosenNearbyRadius = nearbyRadiusOverride || (resume ? (selectedNearbyRadius || 300) : null);
    if (!resume && !agentMode && effectiveGeoPoint) {
        const picked = await chooseNearbyRadius();
        if (picked === null) return;
        chosenNearbyRadius = picked;
        selectedNearbyRadius = picked;
        syncDraftInputsToSession();
    }

    if (!resume) {
        const userHtml = buildUserMessageHtml(text, imagePath, effectiveGeoPoint);
        appendMessage('user', userHtml, true);
        userInput.value = '';
        userInput.style.height = 'auto';
        const userContent = [];
        if (imagePath) userContent.push({ type: 'image', image_path: imagePath });
        if (text) userContent.push({ type: 'text', text: text });
        chatHistory.push({ role: 'user', content: userContent });
        if (imagePath) {
            context.committedImagePath = imagePath;
            lastSubmittedImagePath = imagePath;
            // The path is committed to this user message. Clear the draft so a
            // text-only follow-up cannot accidentally attach and re-run it.
            imagePathInput.value = '';
            if (context.draftInputs) context.draftInputs.imagePath = '';
        }
        if (hasDraftGeo) context.committedGeoPoint = { lat: Number(latitude), lon: Number(longitude) };
        touchActiveSession();
    }

    const loader = appendMessage('bot', '<span>Thinking</span><span class="loading-dots"></span>', true);
    sendBtn.disabled = true;
    if (interruptBtn) interruptBtn.hidden = false;
    activeAnalysisController = new AbortController();

    try {
        const latestUserMessage = [...chatHistory].reverse().find((message) => message && message.role === 'user');
        const latestUserContent = latestUserMessage && latestUserMessage.content;
        const latestUserHasImage = Array.isArray(latestUserContent)
            && latestUserContent.some((part) => part && part.type === 'image');
        const imageAnalysisTurn = !!imagePath || (resume && latestUserHasImage);
        const payload = { messages: buildBackendMessagesForModel(chatHistory), temperature: 0.2, agent_mode: agentMode ? 'agent' : 'graph' };
        if (agentMode && imageAnalysisTurn && context.agentTrace.length) {
            payload.agent_trace = context.agentTrace.map(compactAgentTraceEntry);
        }
        if (agentMode && imageAnalysisTurn) payload.agent_turns_used = Number(context.agentTurnsUsed || 0);
        if (imageAnalysisTurn) {
            payload.enable_seg_llm_second_pass = true;
            if (!agentMode || context.reviewThresholdConfirmed) {
                payload.review_threshold = selectedReviewThreshold;
            }
        }
        if (effectiveGeoPoint) {
            payload.latitude = Number(effectiveGeoPoint.lat);
            payload.longitude = Number(effectiveGeoPoint.lon);
            if (chosenNearbyRadius) payload.nearby_radius = chosenNearbyRadius;
        }
        const contextSummary = buildContextSummaryForModel(session);
        if (contextSummary) {
            payload.context_summary = contextSummary;
        }
        const response = await fetch('/v1/agent/analyze_stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
            signal: activeAnalysisController.signal
        });
        loader.remove();
        if (!response.ok || !response.body) {
            throw new Error(`HTTP ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let finalData = null;
        let pausedForNearbyRadius = false;
        let pausedForReviewThreshold = false;
        let lastRenderedAssistantContent = '';
        const streamedArtifacts = { original: '', seg_mask: '', seg_refine_overlay: '' };

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                const raw = line.trim();
                if (!raw) continue;
                const event = JSON.parse(raw);

                if (event.type === 'assistant') {
                    const rawContent = String(event.content || '');
                    const renderedContent = normalizeEllipsisText(rawContent);
                    const content = rawContent.trim();
                    if (content) {
                        appendTokenLine(rawContent);
                    } else if ((event.tool_calls || []).length > 0) {
                        const plannedTools = (event.tool_calls || [])
                            .map((call) => ((call || {}).function || {}).name || 'unknown')
                            .filter(Boolean);
                        appendTokenTagLine(
                            'TOOL_PLAN',
                            plannedTools.length > 0 ? plannedTools.join(' -> ') : 'unknown'
                        );
                    }
                    if (content) {
                        if ((event.tool_calls || []).length === 0) {
                            const finalRenderedContent = renderedContent.trim();
                            lastRenderedAssistantContent = finalRenderedContent;
                            appendMessage('bot', marked.parse(finalRenderedContent), true);
                        }
                    }
                } else if (event.type === 'tool_call') {
                    appendTokenTagLine('TOOL_CALL', `${event.name} ${JSON.stringify(event.arguments || {})}`);
                } else if (event.type === 'tool_result') {
                    const d = event.data || {};
                    if (agentMode && d.tool) {
                        const traceContext = getSessionContext();
                        if (!Array.isArray(traceContext.agentTrace)) traceContext.agentTrace = [];
                        traceContext.agentTrace.push(compactAgentTraceEntry(d));
                    }
                    appendTokenTagLine('TOOL_RESULT', `${d.tool} status=${d.status}${d.execution_state ? ` state=${d.execution_state}` : ''}`);
                    appendToolExecutionCard(d);
                    appendMessage('bot', buildToolResultMessage(d.tool, d.status, d.output || {}, d.summary || ''), false);
                    if (d.tool) {
                        chatHistory.push({
                            role: 'tool',
                            name: String(d.tool),
                            content: buildPersistedToolContent(d),
                        });
                        touchActiveSession();
                    }
                    if (d.status === 'ok') {
                        const out = (d.output && typeof d.output === 'object') ? d.output : {};
                        if (d.tool === 'tiff.info') {
                            const p = String(out.image_path || '').trim();
                            if (p) streamedArtifacts.original = `/media?path=${encodeURIComponent(p)}`;
                        } else if (d.tool === 'seg.run') {
                            const p = String(out.overlay_path || out.mask_path || '').trim();
                            if (p) streamedArtifacts.seg_mask = `/media?path=${encodeURIComponent(p)}`;
                        } else if (d.tool === 'seg.refine') {
                            const p = String(out.overlay_path || '').trim();
                            if (p) streamedArtifacts.seg_refine_overlay = `/media?path=${encodeURIComponent(p)}`;
                        }
                    }
                    if (d.tool === 'geo.nearby' && d.status === 'ok') {
                        applyNearbyFeaturesPayload(d.output || {}, { updatePoint: true });
                    }
                    if (d.tool === 'geo.background' && d.status === 'ok') {
                        setObservationPoint((d.output || {}).observation_point || null, { updateStatus: false });
                        renderGeoBackgroundCard(d.output || {});
                        persistGeoState();
                    }
                } else if (event.type === 'model_raw') {
                    const rawModelText = String(event.content || '');
                    const rawSource = String(event.source || 'agent');
                    if (rawModelText.trim()) {
                        appendTokenLine(`\n\n[MODEL_RAW:${rawSource}]\n${rawModelText}\n[/MODEL_RAW]\n`);
                    }
                } else if (event.type === 'final') {
                    finalData = event.data || null;
                } else if (event.type === 'need_nearby_radius') {
                    pausedForNearbyRadius = true;
                    if (Number.isFinite(Number(event.agent_turns_used))) getSessionContext(session).agentTurnsUsed = Math.max(0, Number(event.agent_turns_used));
                } else if (event.type === 'need_review_threshold') {
                    pausedForReviewThreshold = true;
                    if (Number.isFinite(Number(event.agent_turns_used))) getSessionContext(session).agentTurnsUsed = Math.max(0, Number(event.agent_turns_used));
                } else if (event.type === 'error') {
                    appendTokenTagLine('ERROR', event.error || 'unknown error');
                    appendMessage('bot', `Stream error: ${event.error || 'unknown error'}`, false);
                }
            }
        }

        if (buffer.trim()) {
            try {
                const event = JSON.parse(buffer.trim());
                if (event.type === 'final') finalData = event.data || null;
                if (event.type === 'need_nearby_radius') { pausedForNearbyRadius = true; if (Number.isFinite(Number(event.agent_turns_used))) getSessionContext(session).agentTurnsUsed = Math.max(0, Number(event.agent_turns_used)); }
                if (event.type === 'need_review_threshold') { pausedForReviewThreshold = true; if (Number.isFinite(Number(event.agent_turns_used))) getSessionContext(session).agentTurnsUsed = Math.max(0, Number(event.agent_turns_used)); }
            } catch (_) {}
        }

        if (pausedForReviewThreshold) {
            appendTokenTagLine('PAUSE', 'waiting for the segmentation review threshold');
            let picked = await chooseReviewThreshold();
            if (picked === null) picked = selectedReviewThreshold || 0.20;
            selectedReviewThreshold = picked;
            getSessionContext().reviewThresholdConfirmed = true;
            syncDraftInputsToSession();
            appendTokenTagLine('RESUME', `review threshold ${Math.round(picked * 100)}%`);
            await sendMessage({ resume: true });
            return;
        }

        if (pausedForNearbyRadius) {
            appendTokenTagLine('PAUSE', 'waiting for OSM nearby-probe radius');
            let picked = await chooseNearbyRadius();
            if (picked === null) picked = selectedNearbyRadius || 300;
            selectedNearbyRadius = picked;
            syncDraftInputsToSession();
            await sendMessage({ resume: true, nearbyRadiusOverride: picked });
            return;
        }

        if (finalData && finalData.choices && finalData.choices.length > 0) {
            const assistantMessage = finalData.choices[0].message || {};
            const rawAssistantContent = String(assistantMessage.content || '');
            assistantMessage.content = normalizeEllipsisText(rawAssistantContent);
            chatHistory.push(assistantMessage);
            const finalContext = getSessionContext();
            if (agentMode && Array.isArray(finalData.agent_trace)) {
                finalContext.agentTrace = finalData.agent_trace.map(compactAgentTraceEntry);
            }
            finalContext.agentTurnsUsed = 0;
            touchActiveSession();
            updateObservationPoint(finalData.geo || null, { openPopup: true, preserveExistingStatus: true });
            const finalContent = String(assistantMessage.content || '').trim();
            const rawFinalContent = rawAssistantContent.trim();
            const reportSummary = summarizeReportForModel(rawAssistantContent);
            if (
                finalData.structured_report?.report_version === 'structured-2'
                && reportSummary
            ) {
                getSessionContext().latestReportSummary = reportSummary;
            }
            // Always surface the final answer; the only thing that suppresses it is
            // it being verbatim what was already rendered. This used to also require
            // the text to look like a full 15-section report, which silently dropped
            // the entire final message on any degraded or fallback run.
            if (rawFinalContent && finalContent !== lastRenderedAssistantContent) {
                appendTokenLine(rawAssistantContent);
                appendMessage('bot', marked.parse(finalContent), true);
                lastRenderedAssistantContent = finalContent;
            }
            const priorArtifacts = (getSessionContext().latestArtifacts && typeof getSessionContext().latestArtifacts === 'object')
                ? getSessionContext().latestArtifacts
                : {};
            const resolvedArtifacts = {
                ...priorArtifacts,
                ...((finalData.artifacts && typeof finalData.artifacts === 'object') ? finalData.artifacts : {}),
            };
            if (!resolvedArtifacts.original && streamedArtifacts.original) {
                resolvedArtifacts.original = streamedArtifacts.original;
            }
            if (!resolvedArtifacts.seg_mask && streamedArtifacts.seg_mask) {
                resolvedArtifacts.seg_mask = streamedArtifacts.seg_mask;
            }
            if (!resolvedArtifacts.seg_refine_overlay && streamedArtifacts.seg_refine_overlay) {
                resolvedArtifacts.seg_refine_overlay = streamedArtifacts.seg_refine_overlay;
            }
            getSessionContext().latestArtifacts = normalizeArtifacts(resolvedArtifacts);
            touchActiveSession();
            if (hasRenderableArtifacts(resolvedArtifacts)) {
                appendTraceCard([], resolvedArtifacts);
            }
        }
    } catch (error) {
        if (error?.name === 'AbortError') {
            loader.innerText = 'Analysis interrupted. Add information and send again to continue.';
            appendTokenTagLine('INTERRUPTED', 'Analysis stopped by user');
        } else {
            appendTokenTagLine('ERROR', error.message);
            loader.innerText = `Error: ${error.message}`;
        }
    } finally {
        sendBtn.disabled = false;
        if (interruptBtn) interruptBtn.hidden = true;
        activeAnalysisController = null;
    }
}

function interruptAnalysis() {
    if (activeAnalysisController) activeAnalysisController.abort();
}

function setStatusBadge(id, isOnline) {
    const badge = document.getElementById(id);
    if (!badge) return;
    if (isOnline) {
        badge.innerHTML = '<span class="w-1.5 h-1.5 rounded-full bg-green-500 mr-1.5"></span>Online';
        badge.className = 'badge badge-sm badge-success status-pill flex items-center text-green-600 font-medium';
    } else {
        badge.innerHTML = '<span class="w-1.5 h-1.5 rounded-full bg-red-500 mr-1.5"></span>Offline';
        badge.className = 'badge badge-sm badge-error status-pill flex items-center text-red-600 font-medium';
    }
}

async function refreshSystemStatus() {
    try {
        const resp = await fetch('/admin/services', { method: 'GET' });
        if (!resp.ok) throw new Error('status api failed');
        const data = await resp.json();
        const llmOnline = data.llm_service_online !== undefined ? !!data.llm_service_online : !!data.llm;
        const segOnline = data.seg_service_online !== undefined ? !!data.seg_service_online : !!data.seg;
        const clsOnline = data.cls_service_online !== undefined ? !!data.cls_service_online : !!data.cls;
        setStatusBadge('llm-status', llmOnline);
        setStatusBadge('seg-status', segOnline);
        setStatusBadge('cls-status', clsOnline);
    } catch (e) {
        setStatusBadge('llm-status', false);
        setStatusBadge('seg-status', false);
        setStatusBadge('cls-status', false);
    }
}

async function startServices() {
    startServicesBtn.disabled = true;
    startServicesBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i> Starting';
    try {
        const resp = await fetch('/admin/start_services', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ start_seg: true, start_cls: true })
        });
        const data = await resp.json();
        appendMessage('bot', `Service start result: ${JSON.stringify(data)}`);
        await refreshSystemStatus();
    } catch (error) {
        appendMessage('bot', `Start failed: ${error.message}`);
    } finally {
        startServicesBtn.disabled = false;
        startServicesBtn.innerHTML = '<i class="fas fa-play mr-2"></i> Start Dependencies';
    }
}

async function stopAllServices() {
    const confirmed = window.confirm('This will stop SegFormer, ConvNeXt, and current system services. Continue?');
    if (!confirmed) return;
    stopServicesBtn.disabled = true;
    stopServicesBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i> Stopping';
    try {
        const resp = await fetch('/admin/stop_services', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ stop_seg: true, stop_cls: true, stop_llm: true })
        });
        const data = await resp.json();
        appendMessage('bot', `Stop result: ${JSON.stringify(data)}\nSystem will exit shortly.`);
        setTimeout(() => { window.location.href = 'about:blank'; }, 1200);
    } catch (error) {
        appendMessage('bot', `Stop failed: ${error.message}`);
        stopServicesBtn.disabled = false;
        stopServicesBtn.innerHTML = '<i class="fas fa-power-off mr-2"></i> Stop All Services';
    }
}

/* ---- event wiring & boot -------------------------------------------------- */
sendBtn.addEventListener('click', sendMessage);
if (interruptBtn) interruptBtn.addEventListener('click', interruptAnalysis);
imagePathInput.addEventListener('input', syncDraftInputsToSession);
latitudeInput.addEventListener('input', syncDraftInputsToSession);
longitudeInput.addEventListener('input', syncDraftInputsToSession);
if (uploadImageBtn && uploadImageInput) {
    uploadImageBtn.addEventListener('click', () => uploadImageInput.click());
    uploadImageInput.addEventListener('change', async () => {
        const file = uploadImageInput.files && uploadImageInput.files[0];
        if (file) await uploadSelectedImage(file);
    });
}
userInput.addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); } });
navChat.addEventListener('click', (e) => { e.preventDefault(); switchPage('chat'); });
navTools.addEventListener('click', (e) => { e.preventDefault(); switchPage('tools'); });
if (navToken) navToken.addEventListener('click', (e) => { e.preventDefault(); switchPage('token'); });
if (clearTokenBtn) clearTokenBtn.addEventListener('click', () => { tokenStream.textContent = ''; persistTokenState(); });
toggleTokenPanelBtn.addEventListener('click', () => { setTokenPanelCollapsed(!tokenPanelCollapsed); });
expandTokenPanelBtn.addEventListener('click', () => { setTokenPanelCollapsed(false); });
newSessionBtn.addEventListener('click', () => {
    createNewSession();
    switchPage('chat');
});
sessionList.addEventListener('click', (e) => {
    const item = e.target.closest('[data-session-id]');
    if (item) {
        hideSessionContextMenu();
        setActiveSession(item.dataset.sessionId || '');
        switchPage('chat');
    }
});
sessionList.addEventListener('contextmenu', (e) => {
    const item = e.target.closest('[data-session-id]');
    if (!item) return;
    e.preventDefault();
    showSessionContextMenu(item.dataset.sessionId || '', e.clientX, e.clientY);
});
contextRenameSessionBtn.addEventListener('click', () => {
    if (contextMenuSessionId) {
        renameSession(contextMenuSessionId);
    }
    hideSessionContextMenu();
});
contextDeleteSessionBtn.addEventListener('click', () => {
    if (contextMenuSessionId) {
        deleteSession(contextMenuSessionId);
    }
    hideSessionContextMenu();
});
document.addEventListener('click', () => { hideSessionContextMenu(); });
window.addEventListener('resize', () => { hideSessionContextMenu(); });
window.addEventListener('scroll', () => { hideSessionContextMenu(); }, true);
reloadNearbyBtn.addEventListener('click', async () => {
    if (!lastGeoPoint) {
        mapStatus.textContent = 'No observation point available to refresh.';
        return;
    }
    await loadNearbyFeatures(lastGeoPoint.lat, lastGeoPoint.lon, selectedNearbyRadius || 300);
});

startServicesBtn.addEventListener('click', startServices);
stopServicesBtn.addEventListener('click', stopAllServices);
initMap();
if (loadSessions()) {
    renderSessionList();
    setActiveSession(activeSessionId || sessions[0]?.id || '');
} else {
    createNewSession();
}
switchPage('chat');
refreshSystemStatus();
setInterval(refreshSystemStatus, 5000);

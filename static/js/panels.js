/* ============================================================================
   panels.js — view layer: geo/map panel, chat window rendering, session list,
   token panel, and the message/tool-card builders. Loaded after state.js.
   ========================================================================== */

function resetGeoPanel() {
    lastGeoPoint = null;
    lastGeoBackground = null;
    mapStatus.textContent = 'Waiting for latitude/longitude input.';
    nearbySummary.innerHTML = '';
    nearbyList.innerHTML = '<div class="px-3 py-3 text-gray-500">Points are shown after latitude/longitude input.</div>';
    if (observationMarker && osmMap) {
        osmMap.removeLayer(observationMarker);
        observationMarker = null;
    }
    if (nearbyLayer) nearbyLayer.clearLayers();
    renderGeoBackgroundCard(null);
}

function renderGeoBackgroundCard(data) {
    if (!geoBgStatus || !geoBgAddress || !geoBgTerrain || !geoBgGeology || !geoBgWarnings) return;
    if (!data || typeof data !== "object") {
        geoBgStatus.textContent = "Waiting for geo.background results.";
        geoBgAddress.textContent = "N/A";
        geoBgTerrain.textContent = "Elevation: - | Slope: - | Aspect: -";
        geoBgGeology.textContent = "N/A";
        geoBgWarnings.textContent = "None";
        return;
    }
    lastGeoBackground = data;
    const point = data.observation_point || {};
    const lat = Number(point.lat);
    const lon = Number(point.lon);
    if (Number.isFinite(lat) && Number.isFinite(lon)) {
        geoBgStatus.textContent = `geo.background updated (lat=${lat.toFixed(6)}, lon=${lon.toFixed(6)})`;
    } else {
        geoBgStatus.textContent = "geo.background updated";
    }
    const addr = (data.address && typeof data.address === "object") ? data.address : {};
    const addrText = String(addr.display_name || "").trim();
    geoBgAddress.textContent = addrText || "N/A";
    const terrain = (data.terrain && typeof data.terrain === "object") ? data.terrain : {};
    const elev = (typeof terrain.elevation_m === "number") ? `${terrain.elevation_m.toFixed(2)} m` : "-";
    const slope = (typeof terrain.slope_deg === "number") ? `${terrain.slope_deg.toFixed(2)} deg` : "-";
    const aspect = (typeof terrain.aspect_deg === "number") ? `${terrain.aspect_deg.toFixed(2)} deg` : "-";
    geoBgTerrain.textContent = `Elevation: ${elev} | Slope: ${slope} | Aspect: ${aspect}`;
    const geology = (data.geology && typeof data.geology === "object") ? data.geology : {};
    const unit = String(geology.unit_name || "").trim();
    const lith = String(geology.lithology || "").trim();
    const age = String(geology.age || "").trim();
    geoBgGeology.textContent = [unit, lith, age].filter(Boolean).join(" | ") || "N/A";
    const warnings = Array.isArray(data.warnings) ? data.warnings : [];
    geoBgWarnings.textContent = warnings.length ? warnings.join(" | ") : "None";
}

function applyGeoState(geoState) {
    resetGeoPanel();
    const state = geoState || {};
    mapStatus.textContent = state.mapStatus || 'Waiting for latitude/longitude input.';
    nearbySummary.innerHTML = state.nearbySummaryHtml || '';
    renderNearbyList(Array.isArray(state.nearbyFeatures) ? state.nearbyFeatures : []);
    renderGeoBackgroundCard(state.backgroundData || null);
    const point = state.observationPoint;
    if (point && typeof point.lat === 'number' && typeof point.lon === 'number') {
        lastGeoPoint = { lat: point.lat, lon: point.lon };
        observationMarker = L.marker([point.lat, point.lon]).addTo(osmMap);
        observationMarker.bindPopup(`Observation Point<br>lat=${point.lat.toFixed(6)}<br>lon=${point.lon.toFixed(6)}`);
        osmMap.setView([point.lat, point.lon], 14);
        for (const feature of (state.nearbyFeatures || [])) {
            const color = feature.type === 'settlement'
                ? '#2563eb'
                : feature.type === 'amenity'
                    ? '#059669'
                    : feature.type === 'road'
                        ? '#d97706'
                        : '#6b7280';
            const marker = L.circleMarker([feature.lat, feature.lon], {
                radius: 5,
                color,
                weight: 1,
                fillColor: color,
                fillOpacity: 0.8
            });
            marker.bindPopup(escapeHtml(classifyFeatureLabel(feature)));
            nearbyLayer.addLayer(marker);
        }
    }
}

function renderChatWindowFromHistory() {
    chatWindow.innerHTML = '';
    const session = getActiveSession();
    const uiEntries = Array.isArray(session?.uiEntries) ? session.uiEntries : [];
    if (!uiEntries.length) {
        for (const msg of chatHistory) {
            if (msg.role === 'user') {
                const content = Array.isArray(msg.content) ? msg.content : [];
                const text = content.filter((part) => part.type === 'text').map((part) => part.text || '').join('\n').trim();
                const image = (content.find((part) => part.type === 'image') || {}).image_path || '';
                appendMessage('user', buildUserMessageHtml(text, image), true, false);
                continue;
            }
            if (msg.role === 'assistant' && typeof msg.content === 'string' && msg.content.trim()) {
                appendMessage('bot', marked.parse(normalizeEllipsisText(msg.content)), true, false);
            }
        }
        return;
    }
    for (const entry of uiEntries) {
        if (entry.kind === 'user_html') {
            appendMessage('user', entry.content || '', true, false);
        } else if (entry.kind === 'assistant_html') {
            appendMessage('bot', normalizeEllipsisText(entry.content || ''), true, false);
        } else if (entry.kind === 'tool_card') {
            appendToolExecutionCard(entry.toolData || {}, false);
        } else if (entry.kind === 'artifact_card') {
            appendTraceCard([], entry.artifacts || {}, false);
        }
    }
}

function setActiveSession(sessionId) {
    const session = sessions.find((item) => item.id === sessionId);
    if (!session) return;
    activeSessionId = session.id;
    chatHistory = Array.isArray(session.messages) ? session.messages : defaultAssistantGreeting();
    lastSubmittedImagePath = getSessionContext(session).committedImagePath || '';
    tokenStream.textContent = String(session.tokenLog || '');
    syncDraftInputsFromSession(session);
    renderChatWindowFromHistory();
    applyGeoState(session.geoState);
    renderSessionList();
    saveSessions();
}

function createNewSession() {
    const session = buildDefaultSession();
    sessions.unshift(session);
    activeSessionId = session.id;
    chatHistory = session.messages;
    lastSubmittedImagePath = '';
    tokenStream.textContent = '';
    syncDraftInputsFromSession(session);
    resetGeoPanel();
    renderChatWindowFromHistory();
    renderSessionList();
    saveSessions();
}

function deleteSession(sessionId) {
    sessions = sessions.filter((item) => item.id !== sessionId);
    if (!sessions.length) {
        createNewSession();
        return;
    }
    if (activeSessionId === sessionId) {
        activeSessionId = sessions[0].id;
        const session = sessions[0];
        chatHistory = session.messages;
        lastSubmittedImagePath = getSessionContext(session).committedImagePath || '';
        tokenStream.textContent = String(session.tokenLog || '');
        syncDraftInputsFromSession(session);
        renderChatWindowFromHistory();
        applyGeoState(session.geoState);
    }
    renderSessionList();
    saveSessions();
}

function renameSession(sessionId) {
    const session = sessions.find((item) => item.id === sessionId);
    if (!session) return;
    const nextTitle = window.prompt('Enter a new chat name', session.title || 'New Chat');
    if (nextTitle === null) return;
    const cleanTitle = String(nextTitle).trim();
    if (!cleanTitle) return;
    session.title = cleanTitle.slice(0, 40);
    session.customTitle = true;
    session.updatedAt = new Date().toISOString();
    sessions.sort((a, b) => String(b.updatedAt).localeCompare(String(a.updatedAt)));
    renderSessionList();
    saveSessions();
}

function renderSessionList() {
    const sessionCount = document.getElementById('session-count');
    if (sessionCount) sessionCount.textContent = String(sessions.length || 0);
    if (!sessions.length) {
        sessionList.innerHTML = '<div class="session-empty">No chats yet. Select New Chat to begin.</div>';
        return;
    }
    sessionList.innerHTML = sessions.map((session) => {
        const isActive = session.id === activeSessionId;
        const time = new Date(session.updatedAt || session.createdAt || Date.now()).toLocaleString('en-US', {
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit',
        });
        return `
            <div class="session-item ${isActive ? 'active' : ''} cursor-pointer" data-session-id="${escapeHtml(session.id)}" title="Right-click for more actions">
                <div class="session-main min-w-0 flex-1">
                    <div class="session-title truncate">${escapeHtml(session.title || 'New Chat')}</div>
                    <div class="session-meta">${escapeHtml(time)}</div>
                </div>
            </div>
        `;
    }).join('');
}
function hideSessionContextMenu() {
    contextMenuSessionId = '';
    sessionContextMenu.classList.add('hidden');
}

function showSessionContextMenu(sessionId, x, y) {
    contextMenuSessionId = sessionId;
    sessionContextMenu.classList.remove('hidden');
    const menuWidth = 132;
    const menuHeight = 80;
    const left = Math.min(x, window.innerWidth - menuWidth - 12);
    const top = Math.min(y, window.innerHeight - menuHeight - 12);
    sessionContextMenu.style.left = `${Math.max(8, left)}px`;
    sessionContextMenu.style.top = `${Math.max(8, top)}px`;
}

function switchPage(page) {
    const isChat = page === "chat";
    const isTools = page === "tools";
    const isToken = page === "token";
    navChat.classList.toggle("active", isChat);
    navTools.classList.toggle("active", isTools);
    if (navToken) navToken.classList.toggle("active", isToken);
    pageChat.classList.toggle("hidden", !isChat);
    pageTools.classList.toggle("hidden", !isTools);
    if (pageToken) pageToken.classList.toggle("hidden", !isToken);
    tabTitle.innerText = isChat ? "Agent Chat" : (isTools ? "Tool Guide" : "Qwen Live Token");
}

function appendTokenLine(text) {
    if (!text) return;
    tokenStream.textContent += text;
    tokenPanelBody.scrollTop = tokenPanelBody.scrollHeight;
    persistTokenState();
}

function appendTokenTagLine(tag, payload) {
    appendTokenLine(`\n\n[${tag}] ${payload}\n`);
}

function setTokenPanelCollapsed(collapsed) {
    tokenPanelCollapsed = collapsed;
    if (collapsed) {
        tokenPanel.classList.remove('w-[26rem]', 'max-w-[40%]', 'min-w-[300px]');
        tokenPanel.classList.add('w-14', 'min-w-[56px]');
        panelStack.classList.add('hidden');
        collapsedTokenRail.classList.remove('hidden');
        toggleTokenPanelBtn.innerText = 'Expand';
    } else {
        tokenPanel.classList.remove('w-14', 'min-w-[56px]');
        tokenPanel.classList.add('w-[26rem]', 'max-w-[40%]', 'min-w-[300px]');
        collapsedTokenRail.classList.add('hidden');
        panelStack.classList.remove('hidden');
        toggleTokenPanelBtn.innerText = 'Collapse';
        if (osmMap) {
            setTimeout(() => osmMap.invalidateSize(), 50);
        }
    }
}

function initMap() {
    if (osmMap) return;
    osmMap = L.map('osm-map', { zoomControl: true }).setView([30.67, 104.06], 5);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 19,
        attribution: '&copy; OpenStreetMap contributors'
    }).addTo(osmMap);
    nearbyLayer = L.layerGroup().addTo(osmMap);
}

function classifyFeatureLabel(feature) {
    const subtype = feature.subtype || '';
    const name = feature.name || '(unnamed)';
    return `${feature.type}: ${name}${subtype ? ` [${subtype}]` : ''}`;
}


const STRUCTURED_REPORT_FIELD_TITLES = [
    'Landslide presence',
    'Landslide type',
    'Image relative position within the image frame',
    'Morphological characteristics',
    'Material composition and surface cover',
    'Movement and deformation features',
    'Surrounding environmental context',
    'Impact on human infrastructure',
    'Reason for landslide classification',
    'Landslide causation inference',
];

function escapeRegExp(value) {
    return String(value).replace(/[|\\{}()[\]^$+*?.]/g, '\\$&');
}

function structuredReportSections(text) {
    const raw = String(text || '');
    const labels = STRUCTURED_REPORT_FIELD_TITLES.map(escapeRegExp).join('|');
    const marker = new RegExp(
        '^\\s*\\*{0,2}(' + labels + '):\\*{0,2}[\\t ]*',
        'gim'
    );
    const matches = [...raw.matchAll(marker)];
    if (matches.length < 7) return [];

    const found = new Map();
    const canonical = new Map(STRUCTURED_REPORT_FIELD_TITLES.map((title) => [title.toLowerCase(), title]));
    matches.forEach((match, index) => {
        const title = canonical.get(String(match[1] || '').toLowerCase());
        if (!title) return;
        const end = index + 1 < matches.length ? matches[index + 1].index : raw.length;
        const body = raw.slice(match.index + match[0].length, end)
            .split(/^\s*#{1,6}\s+/m, 1)[0]
            .replace(/\s+/g, ' ')
            .trim();
        if (body) found.set(title, body);
    });

    return STRUCTURED_REPORT_FIELD_TITLES
        .filter((title) => found.has(title))
        .map((title) => [title, found.get(title)]);
}

function summarizeReportForModel(text) {
    const sections = structuredReportSections(text);
    if (sections.length < 7) return '';
    const summary = sections
        .map(([title, body]) => '**' + title + ':** ' + body)
        .join('\n');
    return summary.slice(0, 1800) + (summary.length > 1800 ? '...' : '');
}

function isLegacyReportText(text) {
    const raw = String(text || '');
    return /###\s*Final Decision Report/i.test(raw)
        || ((raw.match(/^\s*###\s+/gm) || []).length >= 6
            && /###\s*Final Determination/i.test(raw));
}

function buildBackendMessagesForModel(history) {
    const source = Array.isArray(history) ? history : [];
    return source.map((msg) => {
        if (!msg || typeof msg !== 'object') return msg;
        if (msg.role === 'assistant') {
            const summary = summarizeReportForModel(msg.content);
            if (summary) {
                return { ...msg, content: '[Previous structured-2 report summary]\n' + summary };
            }
            if (isLegacyReportText(msg.content)) {
                return { ...msg, content: '[Previous report omitted from model context.]' };
            }
        }
        if (msg.role === 'tool' && isLegacyReportText(msg.content)) {
            return { ...msg, content: '[Previous report omitted from model context.]' };
        }
        if (msg.role === 'tool') {
            let payload = msg.content;
            try {
                if (typeof payload === 'string') payload = JSON.parse(payload);
            } catch (_) {
                payload = null;
            }
            if (payload && typeof payload === 'object' && !Array.isArray(payload)) {
                const compacted = compactAgentTraceEntry({ tool: msg.name, output: payload });
                return { ...msg, content: JSON.stringify(compacted.output || {}) };
            }
        }
        return msg;
    });
}

function buildContextSummaryForModel(session = getActiveSession()) {
    const active = session || getActiveSession();
    if (!active) return '';
    const context = getSessionContext(active);
    const geoState = active.geoState || {};
    const parts = [];

    if (context.committedImagePath) {
        parts.push(`Confirmed image path: ${context.committedImagePath}.`);
    }

    const historyHasCurrentReport = Array.isArray(active.messages)
        && active.messages.some((message) => (
            message?.role === 'assistant'
            && structuredReportSections(message.content).length >= 7
        ));
    const savedReportSummary = summarizeReportForModel(context.latestReportSummary);
    if (savedReportSummary && !historyHasCurrentReport) {
        parts.push('Previous structured-2 report fields:\n' + savedReportSummary);
    }

    const point = context.committedGeoPoint || geoState.observationPoint || null;
    if (point && typeof point.lat === 'number' && typeof point.lon === 'number') {
        parts.push(`Confirmed observation point: lat=${Number(point.lat).toFixed(6)}, lon=${Number(point.lon).toFixed(6)}.`);
    }

    const features = Array.isArray(geoState.nearbyFeatures) ? geoState.nearbyFeatures : [];
    if (features.length > 0) {
        const counts = features.reduce((acc, feature) => {
            const key = String(feature.type || 'other');
            acc[key] = (acc[key] || 0) + 1;
            return acc;
        }, {});
        const countText = Object.entries(counts)
            .map(([key, value]) => `${key}=${value}`)
            .join(', ');
        const sampleText = features
            .slice(0, 8)
            .map((feature) => classifyFeatureLabel(feature))
            .join(' | ');
        parts.push(`Persisted OSM nearby findings: ${features.length} feature(s) available (${countText}).`);
        if (sampleText) {
            parts.push(`Nearby feature examples: ${sampleText}.`);
        }
    }

    const background = geoState.backgroundData;
    if (background && typeof background === 'object') {
        const terrain = (background.terrain && typeof background.terrain === 'object') ? background.terrain : {};
        const geology = (background.geology && typeof background.geology === 'object') ? background.geology : {};
        const address = (background.address && typeof background.address === 'object')
            ? String(background.address.display_name || '').trim()
            : '';
        const terrainBits = [];
        if (typeof terrain.elevation_m === 'number') terrainBits.push(`elevation=${terrain.elevation_m.toFixed(2)} m`);
        if (typeof terrain.slope_deg === 'number') terrainBits.push(`slope=${terrain.slope_deg.toFixed(2)} deg`);
        if (typeof terrain.aspect_deg === 'number') terrainBits.push(`aspect=${terrain.aspect_deg.toFixed(2)} deg`);
        const geologyBits = [
            String(geology.unit_name || '').trim(),
            String(geology.lithology || '').trim(),
            String(geology.age || '').trim(),
        ].filter(Boolean);
        if (address) {
            parts.push(`Persisted reverse-geocoded address: ${address}.`);
        }
        if (terrainBits.length > 0) {
            parts.push(`Persisted terrain background: ${terrainBits.join(', ')}.`);
        }
        if (geologyBits.length > 0) {
            parts.push(`Persisted geology background: ${geologyBits.join(' | ')}.`);
        }
    }

    return parts.join(' ');
}

function renderNearbyList(features) {
    if (!features.length) {
        nearbyList.innerHTML = `
            <div class="px-3 py-3 text-gray-500">
                No OSM points were found in the current area.
            </div>
        `;
        return;
    }

    nearbyList.innerHTML = features.map((feature, index) => `
        <div class="nearby-item px-3 py-2">
            <div class="font-medium text-gray-800">${index + 1}. ${escapeHtml(feature.name || '(unnamed)')}</div>
            <div class="text-gray-600">${escapeHtml(feature.type)}${feature.subtype ? ` / ${escapeHtml(feature.subtype)}` : ''}</div>
            <div class="text-gray-500">lat=${Number(feature.lat).toFixed(6)}, lon=${Number(feature.lon).toFixed(6)}</div>
        </div>
    `).join('');
}

async function loadNearbyFeatures(lat, lon, radius = 300) {
    mapStatus.textContent = `Querying nearby facilities within ${radius} m...`;
    nearbySummary.textContent = '';
    nearbyList.innerHTML = '<div class="px-3 py-3 text-gray-500">Loading point list...</div>';
    if (nearbyLayer) nearbyLayer.clearLayers();
    try {
        const resp = await fetch(`/v1/geo/nearby?lat=${encodeURIComponent(lat)}&lon=${encodeURIComponent(lon)}&radius=${radius}`);
        const data = await resp.json();
        if (!resp.ok) {
            throw new Error(data.detail || `HTTP ${resp.status}`);
        }
        applyNearbyFeaturesPayload(data, { updatePoint: true });
    } catch (error) {
        mapStatus.textContent = `Nearby query failed: ${error.message}`;
        nearbySummary.textContent = '';
        nearbyList.innerHTML = `<div class="px-3 py-3 text-red-600">Point list failed to load: ${escapeHtml(error.message)}</div>`;
        const session = getActiveSession();
        if (session) {
            if (!session.geoState) session.geoState = {};
            session.geoState.nearbyFeatures = [];
            persistGeoState();
        }
    }
}

function chooseNearbyRadius() {
    return new Promise((resolve) => {
        if (!nearbyRadiusDialog) return resolve(selectedNearbyRadius || 300);
        nearbyRadiusInput.value = selectedNearbyRadius || 300;
        nearbyRadiusError.textContent = '';
        nearbyRadiusDialog.showModal();
        const presets = [...nearbyRadiusDialog.querySelectorAll('[data-radius]')];
        const updatePreset = () => presets.forEach((button) => button.classList.toggle('selected', Number(button.dataset.radius) === Number(nearbyRadiusInput.value)));
        const choosePreset = (event) => { nearbyRadiusInput.value = event.currentTarget.dataset.radius; updatePreset(); };
        presets.forEach((button) => button.addEventListener('click', choosePreset));
        nearbyRadiusInput.addEventListener('input', updatePreset);
        updatePreset();
        const submit = (event) => {
            event.preventDefault();
            const radius = Number(nearbyRadiusInput.value);
            if (!Number.isFinite(radius) || radius < 100 || radius > 10000) {
                nearbyRadiusError.textContent = 'Enter a radius between 100 and 10,000 m.';
                nearbyRadiusInput.focus();
                return;
            }
            cleanup(Math.round(radius));
        };
        const cancel = () => cleanup(null);
        const escape = (event) => { event.preventDefault(); cleanup(null); };
        const cleanup = (value) => {
            nearbyRadiusDialog.close();
            nearbyRadiusForm.removeEventListener('submit', submit);
            nearbyRadiusCancel.removeEventListener('click', cancel);
            nearbyRadiusDialog.removeEventListener('cancel', escape);
            nearbyRadiusInput.removeEventListener('input', updatePreset);
            presets.forEach((button) => button.removeEventListener('click', choosePreset));
            resolve(value);
        };
        nearbyRadiusForm.addEventListener('submit', submit);
        nearbyRadiusCancel.addEventListener('click', cancel);
        nearbyRadiusDialog.addEventListener('cancel', escape);
    });
}

function chooseReviewThreshold() {
    return new Promise((resolve) => {
        const defaultThreshold = selectedReviewThreshold || 0.20;
        const promptFallback = () => {
            if (typeof window.prompt !== 'function') return resolve(defaultThreshold);
            while (true) {
                let answer;
                try {
                    answer = window.prompt(
                        'Set the segmentation review threshold (1–100%):',
                        String(Math.round(defaultThreshold * 100))
                    );
                } catch (_) {
                    return resolve(defaultThreshold);
                }
                if (answer === null) return resolve(null);
                const percent = Number(answer);
                if (Number.isFinite(percent) && percent >= 1 && percent <= 100) {
                    return resolve(percent / 100);
                }
                if (typeof window.alert === 'function') {
                    window.alert('Enter a threshold between 1% and 100%.');
                } else {
                    return resolve(defaultThreshold);
                }
            }
        };

        if (
            !reviewThresholdDialog
            || !reviewThresholdForm
            || !reviewThresholdInput
            || !reviewThresholdError
            || typeof reviewThresholdDialog.showModal !== 'function'
        ) {
            return promptFallback();
        }

        try {
            reviewThresholdInput.value = String(Math.round(defaultThreshold * 100));
            reviewThresholdError.textContent = '';
            if (!reviewThresholdDialog.open) reviewThresholdDialog.showModal();
        } catch (_) {
            return promptFallback();
        }

        const presets = [...reviewThresholdDialog.querySelectorAll('[data-threshold]')];
        const updatePreset = () => presets.forEach((button) => button.classList.toggle(
            'selected', Number(button.dataset.threshold) === Number(reviewThresholdInput.value)
        ));
        const choosePreset = (event) => {
            reviewThresholdInput.value = event.currentTarget.dataset.threshold;
            updatePreset();
        };
        const submit = (event) => {
            event.preventDefault();
            const percent = Number(reviewThresholdInput.value);
            if (!Number.isFinite(percent) || percent < 1 || percent > 100) {
                reviewThresholdError.textContent = 'Enter a threshold between 1% and 100%.';
                reviewThresholdInput.focus();
                return;
            }
            cleanup(percent / 100);
        };
        const cancel = () => cleanup(null);
        const escape = (event) => { event.preventDefault(); cleanup(null); };
        const cleanup = (value) => {
            if (reviewThresholdDialog.open) reviewThresholdDialog.close();
            reviewThresholdForm.removeEventListener('submit', submit);
            if (reviewThresholdCancel) reviewThresholdCancel.removeEventListener('click', cancel);
            reviewThresholdDialog.removeEventListener('cancel', escape);
            reviewThresholdInput.removeEventListener('input', updatePreset);
            presets.forEach((button) => button.removeEventListener('click', choosePreset));
            resolve(value);
        };

        presets.forEach((button) => button.addEventListener('click', choosePreset));
        reviewThresholdInput.addEventListener('input', updatePreset);
        updatePreset();
        reviewThresholdForm.addEventListener('submit', submit);
        if (reviewThresholdCancel) reviewThresholdCancel.addEventListener('click', cancel);
        reviewThresholdDialog.addEventListener('cancel', escape);
    });
}

function setObservationPoint(point, { openPopup = false, updateStatus = true } = {}) {
    initMap();
    if (!point || typeof point.lat !== 'number' || typeof point.lon !== 'number') {
        // Keep the last resolved geo context when the current turn
        // does not provide fresh latitude/longitude input.
        return false;
    }

    lastGeoPoint = { lat: Number(point.lat), lon: Number(point.lon) };
    if (observationMarker) {
        osmMap.removeLayer(observationMarker);
    }
    observationMarker = L.marker([lastGeoPoint.lat, lastGeoPoint.lon]).addTo(osmMap);
    observationMarker.bindPopup(`Observation Point<br>lat=${lastGeoPoint.lat.toFixed(6)}<br>lon=${lastGeoPoint.lon.toFixed(6)}`);
    if (openPopup) {
        observationMarker.openPopup();
    }
    osmMap.setView([lastGeoPoint.lat, lastGeoPoint.lon], 14);
    if (updateStatus) {
        mapStatus.textContent = `Observation Point: lat=${lastGeoPoint.lat.toFixed(6)}, lon=${lastGeoPoint.lon.toFixed(6)}`;
    }
    return true;
}

function applyNearbyFeaturesPayload(data, { updatePoint = true } = {}) {
    initMap();
    const payload = (data && typeof data === 'object') ? data : {};
    const features = Array.isArray(payload.features) ? payload.features : [];
    const point = payload.observation_point || null;
    if (updatePoint) {
        setObservationPoint(point, { updateStatus: false });
    }
    if (nearbyLayer) nearbyLayer.clearLayers();

    for (const feature of features) {
        const color = feature.type === 'settlement'
            ? '#2563eb'
            : feature.type === 'amenity'
                ? '#059669'
                : feature.type === 'road'
                    ? '#d97706'
                    : '#6b7280';
        const marker = L.circleMarker([feature.lat, feature.lon], {
            radius: 5,
            color,
            weight: 1,
            fillColor: color,
            fillOpacity: 0.8
        });
        marker.bindPopup(escapeHtml(classifyFeatureLabel(feature)));
        nearbyLayer.addLayer(marker);
    }

    const warnings = Array.isArray(payload.warnings) ? payload.warnings : [];
    const top = features.slice(0, 12);
    const counts = features.reduce((acc, feature) => {
        acc[feature.type] = (acc[feature.type] || 0) + 1;
        return acc;
    }, {});

    if (warnings.length > 0) {
        const warningText = escapeHtml(String(warnings[0] || 'OSM nearby context is temporarily unavailable.'));
        mapStatus.textContent = 'OSM nearby query is temporarily unavailable.';
        nearbySummary.innerHTML = `
            <div class="font-medium mb-1 text-amber-700">Nearby query degraded</div>
            <div class="text-amber-700 break-words">${warningText}</div>
        `;
        nearbyList.innerHTML = '<div class="px-3 py-3 text-amber-700">Unable to fetch OSM nearby features right now. Please retry later.</div>';
    } else if (payload.source_status === 'cached') {
        mapStatus.textContent = `Loaded ${payload.count || 0} cached OSM features.`;
        nearbySummary.innerHTML = `
            <div class="font-medium mb-1">Nearby Summary (Cached)</div>
            <div>settlement: ${counts.settlement || 0}, amenity: ${counts.amenity || 0}, road: ${counts.road || 0}, building: ${counts.building || 0}</div>
            <div class="mt-1 break-words">${top.map(classifyFeatureLabel).map(escapeHtml).join(' | ') || 'No results'}</div>
        `;
        renderNearbyList(features);
    } else {
        mapStatus.textContent = `Loaded ${payload.count || 0} OSM features.`;
        nearbySummary.innerHTML = `
            <div class="font-medium mb-1">Nearby Summary</div>
            <div>settlement: ${counts.settlement || 0}, amenity: ${counts.amenity || 0}, road: ${counts.road || 0}, building: ${counts.building || 0}</div>
            <div class="mt-1 break-words">${top.map(classifyFeatureLabel).map(escapeHtml).join(' | ') || 'No results'}</div>
        `;
        renderNearbyList(features);
    }

    const session = getActiveSession();
    if (session) {
        if (!session.geoState) session.geoState = {};
        session.geoState.nearbyFeatures = features;
        persistGeoState();
    }
}

function updateObservationPoint(geo, { openPopup = true, preserveExistingStatus = false } = {}) {
    const point = geo?.observation_point || null;
    const currentStatus = String(mapStatus?.textContent || '').trim();
    const shouldKeepStatus = preserveExistingStatus
        && currentStatus
        && currentStatus !== 'Waiting for latitude/longitude input.';
    if (!setObservationPoint(point, { openPopup, updateStatus: !shouldKeepStatus })) {
        return;
    }
    persistGeoState();
}

function appendMessage(role, content, isHtml = false, persist = true) {
    const wrapper = document.createElement('div');
    wrapper.className = `chat ${role === 'user' ? 'chat-end' : 'chat-start'}`;
    const bubble = document.createElement('div');
    bubble.className = `message-bubble chat-bubble p-4 rounded-2xl shadow-sm ${role === 'user' ? 'user-message chat-bubble-primary' : 'bot-message'}`;
    if (isHtml) bubble.innerHTML = content;
    else bubble.innerText = content;
    wrapper.appendChild(bubble);
    chatWindow.appendChild(wrapper);
    chatWindow.scrollTop = chatWindow.scrollHeight;
    if (persist) {
        persistUiEntry({
            kind: role === 'user' ? 'user_html' : 'assistant_html',
            content: isHtml ? content : escapeHtml(content),
        });
    }
    return bubble;
}

function escapeHtml(text) {
    return String(text)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#039;');
}

function hasRenderableArtifacts(artifacts) {
    return Boolean(artifacts && (artifacts.original || artifacts.seg_mask || artifacts.seg_refine_overlay));
}

function appendTraceCard(trace, artifacts, persist = true) {
    const originalHtml = artifacts?.original
        ? `<div><div class="text-xs font-semibold mb-1">Original</div><img src="${artifacts.original}" class="w-full rounded border" /></div>`
        : '<div><div class="text-xs font-semibold mb-1">Original</div><div class="w-full h-40 rounded border bg-gray-50 text-xs text-gray-500 flex items-center justify-center">Not generated</div></div>';
    const segHtml = artifacts?.seg_mask
        ? `<div><div class="text-xs font-semibold mb-1">Seg Mask</div><img src="${artifacts.seg_mask}" class="w-full rounded border" /></div>`
        : '<div><div class="text-xs font-semibold mb-1">Seg Mask</div><div class="w-full h-40 rounded border bg-gray-50 text-xs text-gray-500 flex items-center justify-center">Not generated</div></div>';
    const refineHtml = artifacts?.seg_refine_overlay
        ? `<div><div class="text-xs font-semibold mb-1">Refine Overlay</div><img src="${artifacts.seg_refine_overlay}" class="w-full rounded border" /></div>`
        : '<div><div class="text-xs font-semibold mb-1">Refine Overlay</div><div class="w-full h-40 rounded border bg-gray-50 text-xs text-gray-500 flex items-center justify-center">Not generated</div></div>';

    const rows = (trace || []).map((t, idx) => {
        const ok = t.status === 'ok';
        const icon = ok ? '🟢' : '🔴';
        const title = `Step ${idx + 1} ${icon} ${t.tool} (${t.cost_ms || 0} ms)`;
        const pretty = escapeHtml(JSON.stringify(t, null, 2));
        return `<details class="text-sm border rounded-lg p-2 bg-white">
            <summary class="cursor-pointer font-medium">${title}</summary>
            <pre class="mt-2 p-2 bg-gray-50 rounded text-xs overflow-x-auto whitespace-pre-wrap">${pretty}</pre>
        </details>`;
    }).join('');

    const imgs = [originalHtml, segHtml, refineHtml].join('');

    const traceSection = rows
        ? `<div class="text-sm font-semibold mb-2">Tool Execution Trace</div><div class="space-y-2 mb-3">${rows}</div>`
        : '';

    const html = `
        <div class="message-bubble chat-bubble bot-message p-4 rounded-2xl shadow-sm">
            ${traceSection}
            <div class="text-sm font-semibold mb-2">Result Layers</div>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-2">${imgs}</div>
        </div>
    `;
    const wrapper = document.createElement('div');
    wrapper.className = 'chat chat-start';
    wrapper.innerHTML = html;
    chatWindow.appendChild(wrapper);
    chatWindow.scrollTop = chatWindow.scrollHeight;
    if (persist) {
        persistUiEntry({
            kind: 'artifact_card',
            artifacts,
        });
    }
}

function appendToolExecutionCard(toolData, persist = true) {
    const tool = toolData?.tool || 'unknown';
    const status = toolData?.status || 'unknown';
    const executionState = String(toolData?.execution_state || (toolData?.cached ? 'reused' : (status === 'ok' ? 'completed' : status)));
    const stateLabel = executionState === 'reused' ? 'REUSED' : (executionState === 'completed' ? 'DONE' : executionState.toUpperCase());
    const stateClass = executionState === 'reused' ? 'text-amber-700' : (executionState === 'completed' ? 'text-emerald-700' : 'text-rose-700');
    const summary = String(toolData?.summary || '').trim() || buildToolSummaryEnglish(tool, toolData?.output || {});
    const costMs = Number(toolData?.cost_ms || 0);
    const pretty = escapeHtml(JSON.stringify(toolData || {}, null, 2));
    const ok = status === 'ok';
    const badgeClass = ok
        ? 'badge badge-sm badge-success text-green-700 bg-green-50 border-green-200'
        : 'badge badge-sm badge-error text-red-700 bg-red-50 border-red-200';
    const html = `
        <div class="message-bubble chat-bubble bot-message p-4 rounded-2xl shadow-sm">
            <details class="text-sm">
                <summary class="cursor-pointer list-none">
                    <div class="flex items-start justify-between gap-3">
                        <div class="min-w-0">
                            <div class="font-semibold text-gray-800">${escapeHtml(tool)}</div>
                            <div class="text-xs text-gray-600 mt-1 break-all">${escapeHtml(summary)}</div>
                        </div>
                        <div class="shrink-0 text-right">
                            <div class="inline-flex items-center px-2 py-0.5 rounded-full border text-[11px] ${badgeClass}">
                                ${escapeHtml(status)}
                            </div>
                            <div class="text-[10px] font-semibold mt-1 ${stateClass}">${escapeHtml(stateLabel)}</div>
                            <div class="text-[11px] text-gray-500 mt-1">${costMs} ms</div>
                        </div>
                    </div>
                </summary>
                <pre class="mt-3 p-3 bg-gray-50 rounded text-xs overflow-x-auto whitespace-pre-wrap">${pretty}</pre>
            </details>
        </div>
    `;
    const wrapper = document.createElement('div');
    wrapper.className = 'chat chat-start';
    wrapper.innerHTML = html;
    chatWindow.appendChild(wrapper);
    chatWindow.scrollTop = chatWindow.scrollHeight;
    if (persist) {
        persistUiEntry({
            kind: 'tool_card',
            toolData,
        });
    }
}

function buildUserMessageHtml(text, imagePath, observationPoint = null) {
    const safeText = escapeHtml(text || '');
    const safePath = escapeHtml(imagePath || '');
    const lat = observationPoint && typeof observationPoint.lat === 'number' ? observationPoint.lat : null;
    const lon = observationPoint && typeof observationPoint.lon === 'number' ? observationPoint.lon : null;
    const locationHtml = (lat !== null && lon !== null)
        ? `<div class="text-xs text-gray-600 break-all">Observation Point: ${escapeHtml(Number(lat).toFixed(6))}, ${escapeHtml(Number(lon).toFixed(6))}</div>`
        : '';
    if (!imagePath) return `${locationHtml}<div class="whitespace-pre-wrap">${safeText}</div>`;
    const mediaUrl = `/media?path=${encodeURIComponent(imagePath)}`;
    return `
        <div class="space-y-2">
            <div class="text-xs text-gray-600 break-all">Image Path: ${safePath}</div>
            ${locationHtml}
            <img
                src="${mediaUrl}"
                class="max-h-72 rounded-lg border object-contain bg-white"
                alt="user image"
                onerror="this.insertAdjacentHTML('afterend','<div class=&quot;text-xs text-red-600&quot;>Image preview failed. Check that the path exists under /root/autodl-tmp.</div>'); this.style.display='none';"
            />
            ${safeText ? `<div class="whitespace-pre-wrap">${safeText}</div>` : ''}
        </div>
    `;
}

function buildToolSummaryEnglish(tool, output) {
    const data = (output && typeof output === 'object') ? output : {};
    if (data.error) {
        const err = String(data.error).replace(/\s+/g, ' ').trim();
        return `${tool}: fail (${err})`;
    }

    if (tool === 'tiff.info') {
        const w = data.width ?? '?';
        const h = data.height ?? '?';
        const bands = data.bands ?? '?';
        return `meta ${w}x${h} b${bands}`;
    }

    if (tool === 'llm.first_pass') {
        const has = data.has_landslide === true ? 'likely' : 'unlikely';
        const score = (typeof data.score === 'number') ? ` ${data.score.toFixed(2)}` : '';
        return `first-pass ${has}${score}`;
    }

    if (tool === 'cls.run') {
        const cls = data.class_name || `class_${data.class_id ?? '?'}`;
        const conf = (typeof data.confidence === 'number') ? ` ${data.confidence.toFixed(2)}` : '';
        return `class ${cls}${conf}`;
    }

    if (tool === 'seg.refine') {
        const count = Array.isArray(data.regions) ? data.regions.length : 0;
        const ratio = (typeof data.area_ratio === 'number') ? ` r=${data.area_ratio.toFixed(3)}` : '';
        return `seg-guided regions=${count}${ratio}`;
    }

    if (tool === 'seg.llm_review') {
        const reviewed = Number(data.review_candidates || 0);
        const executed = data.llm_second_pass && typeof data.llm_second_pass === 'object';
        if (data.llm_second_pass_skipped_for_large_area) return `llm review skipped`;
        return executed ? `llm review regions=${reviewed}` : `llm review pending`;
    }

    if (tool === 'seg.run') {
        const ratio = (typeof data.area_ratio === 'number') ? data.area_ratio.toFixed(3) : '?';
        const px = data.landslide_pixels ?? 0;
        return `seg r=${ratio} px=${px}`;
    }

    if (tool === 'geo.nearby') {
        const warnings = Array.isArray(data.warnings) ? data.warnings : [];
        if (warnings.length > 0) return 'geo nearby unavailable';
        const count = data.count ?? 0;
        const radius = data.radius_m ?? '?';
        if (data.source_status === 'cached') return `geo nearby cached=${count} r=${radius}m`;
        return `geo nearby=${count} r=${radius}m`;
    }
    if (tool === 'geo.background') {
        const terrain = (data.terrain && typeof data.terrain === 'object') ? data.terrain : {};
        const geology = (data.geology && typeof data.geology === 'object') ? data.geology : {};
        const elev = (typeof terrain.elevation_m === 'number') ? terrain.elevation_m.toFixed(1) : '?';
        const slope = (typeof terrain.slope_deg === 'number') ? terrain.slope_deg.toFixed(1) : '?';
        const aspect = (typeof terrain.aspect_deg === 'number') ? terrain.aspect_deg.toFixed(1) : '?';
        const lith = String(geology.lithology || geology.unit_name || 'n/a').replace(/\s+/g, ' ').trim();
        return `geo bg z=${elev}m slope=${slope}deg aspect=${aspect}deg lith=${lith}`;
    }


    if (tool === 'fuse.decision') {
        const has = data.has_landslide === true ? 'yes' : (data.has_landslide === false ? 'no' : '?');
        const clsConf = (typeof data.classification_confidence === 'number') ? data.classification_confidence.toFixed(2) : 'n/a';
        const sev = String(data.severity || 'n/a');
        return `fuse landslide=${has} cls_conf=${clsConf} sev=${sev}`;
    }

    if (tool === 'report.write') {
        const path = String(data.report_path || '').trim();
        return path ? `report saved ${path}` : 'report saved';
    }

    return `${tool} done`;
}

function buildToolResultMessage(tool, status, output, summary = '') {
    const llmSummary = String(summary || '').trim();
    if (llmSummary) {
        return llmSummary;
    }
    const data = (output && typeof output === 'object') ? output : {};
    if (status !== 'ok') {
        return `${tool} failed.`;
    }
    if (tool === 'geo.nearby') {
        const warnings = Array.isArray(data.warnings) ? data.warnings : [];
        if (warnings.length > 0) {
            const msg = String(warnings[0] || '').replace(/\s+/g, ' ').trim();
            return msg || 'Nearby context is temporarily unavailable; the report continues without OSM nearby features.';
        }
        const count = Number(data.count || 0);
        if (data.source_status === 'cached') {
            return `Nearby context loaded from cache: ${count} features.`;
        }
        return `Nearby context retrieved: ${count} features.`;
    }
    if (tool === 'geo.background') {
        const terrain = (data.terrain && typeof data.terrain === 'object') ? data.terrain : {};
        const geology = (data.geology && typeof data.geology === 'object') ? data.geology : {};
        const slope = (typeof terrain.slope_deg === 'number') ? `${terrain.slope_deg.toFixed(1)}°` : 'unknown';
        const lith = String(geology.lithology || geology.unit_name || '').trim();
        return lith ? `Geologic background updated: slope ${slope}, lithology ${lith}.` : `Geologic background updated: slope ${slope}.`;
    }
    if (tool === 'seg.refine') {
        const count = Array.isArray(data.regions) ? data.regions.length : 0;
        return `Segmentation-guided refinement finished: ${count} candidate regions.`;
    }
    if (tool === 'seg.llm_review') {
        if (data.llm_second_pass_skipped_for_large_area) return 'LLM second-pass review was not executed.';
        const reviewed = Number(data.review_candidates || 0);
        const evidence = String(((data.llm_second_pass || {}).evidence || '')).trim();
        if (evidence) return `LLM second-pass review finished: whole-image boundary-overlay re-check over ${reviewed} candidate regions.`;
        return reviewed > 0 ? `A whole-image boundary-overlay review was prepared for ${reviewed} candidate regions, but no second-pass conclusion was returned.` : 'No whole-image boundary-overlay review input was available for LLM second-pass review.';
    }
    if (tool === 'seg.run') {
        const ratio = (typeof data.area_ratio === 'number') ? `${(data.area_ratio * 100).toFixed(2)}%` : 'unknown';
        return `Segmentation finished: landslide area ratio ${ratio}.`;
    }
    if (tool === 'cls.run') {
        const label = String(data.class_name || data.label || 'unknown').trim();
        return `Classification finished: ${label}.`;
    }
    if (tool === 'llm.first_pass') {
        const decision = data.has_landslide === true ? 'likely landslide' : (data.has_landslide === false ? 'likely non-landslide' : 'completed');
        return `First-pass screening finished: ${decision}.`;
    }
    if (tool === 'tiff.info') {
        const w = data.width ?? '?';
        const h = data.height ?? '?';
        return `Image metadata loaded: ${w} x ${h}.`;
    }
    if (tool === 'fuse.decision') {
        return 'Final fused decision completed.';
    }
    if (tool === 'report.write') {
        const path = String(data.report_path || '').trim();
        return path ? `Final report written to ${path}.` : 'Final report written.';
    }
    return `${tool} completed.`;
}

function compactToolTraceValue(value, limits = {}, depth = 0) {
    const maxString = Number(limits.maxString || 1200);
    const maxArray = Number(limits.maxArray || 40);
    const maxKeys = Number(limits.maxKeys || 80);
    const maxDepth = Number(limits.maxDepth || 8);
    if (typeof value === 'string') {
        return value.length <= maxString ? value : `${value.slice(0, maxString)}…[truncated]`;
    }
    if (value === null || typeof value !== 'object') return value;
    if (depth >= maxDepth) return Array.isArray(value) ? '[nested array omitted]' : '[nested object omitted]';
    if (Array.isArray(value)) {
        return value.slice(0, maxArray).map((item) => compactToolTraceValue(item, limits, depth + 1));
    }
    const output = {};
    for (const [key, item] of Object.entries(value).slice(0, maxKeys)) {
        output[key] = compactToolTraceValue(item, limits, depth + 1);
    }
    return output;
}

function stripFusionReportText(value) {
    const reportKeys = new Set([
        'final_description', 'summary', 'report', 'report_text', 'narrative',
        'recommendations', 'visual_description', 'spatial_distribution',
        'tool_interpretation', 'uncertainty', 'classification_reference_note',
        'second_pass_note', 'whole_image_overview',
    ]);
    if (typeof value === 'string') return isLegacyReportText(value) ? '[previous report omitted]' : value;
    if (Array.isArray(value)) return value.map(stripFusionReportText);
    if (!value || typeof value !== 'object') return value;
    const clean = {};
    for (const [key, item] of Object.entries(value)) {
        if (!reportKeys.has(String(key).trim().toLowerCase())) clean[key] = stripFusionReportText(item);
    }
    return clean;
}

function compactAgentTraceEntry(item) {
    if (!item || typeof item !== 'object' || Array.isArray(item)) return item;
    const tool = String(item.tool || '').trim();
    const compactOutput = (limits) => {
        const raw = item.output && typeof item.output === 'object'
            ? item.output
            : { raw_output: item.output ?? null };
        const cleaned = tool === 'fuse.decision' ? stripFusionReportText(raw) : raw;
        const output = compactToolTraceValue(cleaned, limits);
        if (tool === 'geo.nearby' && Array.isArray(raw.features) && output && typeof output === 'object') {
            const counts = new Map();
            raw.features.forEach((feature) => {
                if (!feature || typeof feature !== 'object') return;
                const type = String(feature.type || 'unknown');
                const subtype = String(feature.subtype || '');
                const key = `${type}\u0000${subtype}`;
                counts.set(key, (counts.get(key) || 0) + Math.max(1, Number(feature._count) || 1));
            });
            output.features = Array.from(counts.entries()).map(([key, count]) => {
                const [type, subtype] = key.split('\u0000');
                return { type, subtype, _count: count };
            });
        }
        return output;
    };
    let compacted = {
        tool,
        status: String(item.status || ''),
        execution_state: String(item.execution_state || ''),
        cached: item.cached === true,
        cost_ms: Number(item.cost_ms || 0),
        input: compactToolTraceValue(item.input && typeof item.input === 'object' ? item.input : {}, {}),
        output: compactOutput({ maxString: 1200, maxArray: 40, maxKeys: 80, maxDepth: 8 }),
    };
    if (JSON.stringify(compacted.output).length > 6000) {
        compacted.output = compactOutput({ maxString: 300, maxArray: 10, maxKeys: 40, maxDepth: 6 });
    }
    if (JSON.stringify(compacted).length > 12000) {
        compacted.input = compactToolTraceValue(item.input && typeof item.input === 'object' ? item.input : {},
            { maxString: 300, maxArray: 10, maxKeys: 40, maxDepth: 6 });
        compacted.output = compactOutput({ maxString: 200, maxArray: 8, maxKeys: 32, maxDepth: 5 });
    }
    return compacted;
}

function buildPersistedToolContent(toolData) {
    const compacted = compactAgentTraceEntry(toolData || {});
    return JSON.stringify(compacted && compacted.output ? compacted.output : {});
}

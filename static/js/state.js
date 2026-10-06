/* ============================================================================
   state.js — DOM handles, session model, persistence & context helpers.
   Loaded first. Everything lives in the shared classic-script scope so the
   other modules (panels.js, app.js) can call these directly.
   ========================================================================== */

const chatWindow = document.getElementById('chat-window');
const userInput = document.getElementById('user-input');
const imagePathInput = document.getElementById('image-path');
const sendBtn = document.getElementById('send-btn');
const interruptBtn = document.getElementById('interrupt-btn');
const latitudeInput = document.getElementById('latitude');
const longitudeInput = document.getElementById('longitude');
const nearbyRadiusDialog = document.getElementById('nearby-radius-dialog');
const nearbyRadiusForm = document.getElementById('nearby-radius-form');
const nearbyRadiusInput = document.getElementById('nearby-radius-input');
const nearbyRadiusError = document.getElementById('nearby-radius-error');
const nearbyRadiusCancel = document.getElementById('nearby-radius-cancel');
const reviewThresholdDialog = document.getElementById('review-threshold-dialog');
const reviewThresholdForm = document.getElementById('review-threshold-form');
const reviewThresholdInput = document.getElementById('review-threshold-input');
const reviewThresholdError = document.getElementById('review-threshold-error');
const reviewThresholdCancel = document.getElementById('review-threshold-cancel');
const composerShell = document.getElementById('composer-shell');
const agentModeToggle = document.getElementById('agent-mode-toggle');
const startServicesBtn = document.getElementById('start-services-btn');
const stopServicesBtn = document.getElementById('stop-services-btn');
const uploadImageBtn = document.getElementById('upload-image-btn');
const uploadImageInput = document.getElementById('upload-image-input');
const tabTitle = document.getElementById('tab-title');
const navChat = document.getElementById('nav-chat');
const navTools = document.getElementById('nav-tools');
const navToken = document.getElementById('nav-token');
const newSessionBtn = document.getElementById('new-session-btn');
const sessionList = document.getElementById('session-list');
const sessionContextMenu = document.getElementById('session-context-menu');
const contextRenameSessionBtn = document.getElementById('context-rename-session');
const contextDeleteSessionBtn = document.getElementById('context-delete-session');
const pageChat = document.getElementById('page-chat');
const pageTools = document.getElementById('page-tools');
const pageToken = document.getElementById('page-token');
const tokenPanel = document.getElementById('token-panel');
const tokenPanelBody = document.getElementById('token-panel-body');
const tokenTitleText = document.getElementById('token-title-text');
const tokenControls = document.getElementById('token-controls');
const tokenStream = document.getElementById('token-stream');
const clearTokenBtn = document.getElementById('clear-token-btn');
const toggleTokenPanelBtn = document.getElementById('toggle-token-panel-btn');
const expandTokenPanelBtn = document.getElementById('expand-token-panel-btn');
const panelStack = document.getElementById('panel-stack');
const collapsedTokenRail = document.getElementById('collapsed-token-rail');
const mapStatus = document.getElementById('map-status');
const nearbySummary = document.getElementById('nearby-summary');
const nearbyList = document.getElementById('nearby-list');
const reloadNearbyBtn = document.getElementById('reload-nearby-btn');
const geoBgStatus = document.getElementById('geo-bg-status');
const geoBgAddress = document.getElementById('geo-bg-address');
const geoBgTerrain = document.getElementById('geo-bg-terrain');
const geoBgGeology = document.getElementById('geo-bg-geology');
const geoBgWarnings = document.getElementById('geo-bg-warnings');
const SESSION_STORAGE_KEY = 'landslide_agent_sessions_v1';
let chatHistory = [];
let sessions = [];
let activeSessionId = '';
let contextMenuSessionId = '';
let lastSubmittedImagePath = '';
let tokenPanelCollapsed = false;
let osmMap = null;
let observationMarker = null;
let nearbyLayer = null;
let lastGeoPoint = null;
let lastGeoBackground = null;
let selectedNearbyRadius = 300;
let selectedReviewThreshold = 0.20;
let activeAnalysisController = null;
const EMPTY_ARTIFACTS = Object.freeze({
    original: '',
    seg_mask: '',
    seg_refine_overlay: '',
});

function createSessionTitleFromHistory(history) {
    const firstUser = (history || []).find((item) => item.role === 'user');
    if (!firstUser) return 'New Chat';
    const content = Array.isArray(firstUser.content) ? firstUser.content : [];
    const textPart = content.find((part) => part.type === 'text' && part.text);
    const imagePart = content.find((part) => part.type === 'image' && (part.image_path || part.image));
    if (textPart && String(textPart.text).trim()) {
        const compact = String(textPart.text).trim().replace(/\s+/g, ' ');
        return compact.slice(0, 18) + (compact.length > 18 ? '...' : '');
    }
    if (imagePart) {
        const path = String(imagePart.image_path || imagePart.image || '');
        const name = path.split('/').pop() || 'Image Analysis';
        return name.slice(0, 18) + (name.length > 18 ? '...' : '');
    }
    return 'New Chat';
}

function defaultAssistantGreeting() {
    return [{
        role: 'assistant',
        content: 'Welcome to Landslideagent. Upload an image or enter a local path, then describe your analysis task.'
    }];
}

function createDefaultContextState() {
    return {
        committedImagePath: '',
        committedGeoPoint: null,
        latestArtifacts: { ...EMPTY_ARTIFACTS },
        latestReportSummary: '',
        agentTrace: [],
        reviewThresholdConfirmed: false,
        agentTurnsUsed: 0,
        draftInputs: {
            imagePath: '',
            latitude: '',
            longitude: '',
            nearbyRadius: '300',
            reviewThreshold: '0.20',
        },
    };
}

function normalizeArtifacts(artifacts) {
    return {
        original: String(artifacts?.original || ''),
        seg_mask: String(artifacts?.seg_mask || ''),
        seg_refine_overlay: String(artifacts?.seg_refine_overlay || ''),
    };
}

function normalizeSession(session) {
    if (!session || typeof session !== 'object') return session;
    if (!session.customTitle && session.title === '新建对话') session.title = 'New Chat';
    const previousGreeting = '欢迎使用滑坡遥感智能体。上传影像或输入本地影像路径，描述你的研判任务，即可开始分析。';
    const previousEnglishGreeting = 'Welcome to TerraSight. Upload an image or enter a local path, then describe your analysis task.';
    const greeting = defaultAssistantGreeting()[0].content;
    if (Array.isArray(session.messages) && [previousGreeting, previousEnglishGreeting].includes(session.messages[0]?.content)) session.messages[0].content = greeting;
    if (Array.isArray(session.uiEntries) && session.uiEntries[0]?.kind === 'assistant_html' && [previousGreeting, previousEnglishGreeting].some((oldGreeting) => String(session.uiEntries[0].content || '').includes(oldGreeting))) {
        session.uiEntries[0].content = marked.parse(greeting);
    }
    if (!session.geoState || typeof session.geoState !== 'object') {
        session.geoState = {
            observationPoint: null,
            mapStatus: 'Waiting for latitude/longitude input.',
            nearbySummaryHtml: '',
            nearbyFeatures: [],
            backgroundData: null,
        };
    } else {
        session.geoState.observationPoint = session.geoState.observationPoint || null;
        session.geoState.mapStatus = String(session.geoState.mapStatus || 'Waiting for latitude/longitude input.');
        session.geoState.nearbySummaryHtml = String(session.geoState.nearbySummaryHtml || '');
        session.geoState.nearbyFeatures = Array.isArray(session.geoState.nearbyFeatures) ? session.geoState.nearbyFeatures : [];
        session.geoState.backgroundData = session.geoState.backgroundData || null;
    }
    const defaults = createDefaultContextState();
    const source = (session.contextState && typeof session.contextState === 'object') ? session.contextState : {};
    const draftInputs = (source.draftInputs && typeof source.draftInputs === 'object') ? source.draftInputs : {};
    const committedImagePath = String(source.committedImagePath || session.lastImagePath || defaults.committedImagePath);
    const savedImageDraft = String(draftInputs.imagePath || '');
    session.contextState = {
        committedImagePath,
        committedGeoPoint: source.committedGeoPoint || session.geoState.observationPoint || defaults.committedGeoPoint,
        latestArtifacts: normalizeArtifacts(source.latestArtifacts || defaults.latestArtifacts),
        latestReportSummary: String(source.latestReportSummary || ''),
        agentTrace: Array.isArray(source.agentTrace)
            ? source.agentTrace.filter((item) => item && typeof item === 'object' && !Array.isArray(item))
            : [],
        reviewThresholdConfirmed: source.reviewThresholdConfirmed === true,
        agentTurnsUsed: Math.max(0, Number(source.agentTurnsUsed) || 0),
        draftInputs: {
            imagePath: savedImageDraft === committedImagePath ? '' : savedImageDraft,
            latitude: String(draftInputs.latitude || ''),
            longitude: String(draftInputs.longitude || ''),
            nearbyRadius: String(draftInputs.nearbyRadius || '300'),
            reviewThreshold: String(draftInputs.reviewThreshold || '0.20'),
        },
    };
    session.lastImagePath = session.contextState.committedImagePath;
    return session;
}

function getSessionContext(session = getActiveSession()) {
    if (!session) return createDefaultContextState();
    normalizeSession(session);
    return session.contextState;
}

function getEffectiveDraftInputs(session = getActiveSession()) {
    const context = getSessionContext(session);
    return {
        imagePath: String(context.draftInputs?.imagePath || ''),
        latitude: String(context.draftInputs?.latitude || ''),
        longitude: String(context.draftInputs?.longitude || ''),
        nearbyRadius: String(context.draftInputs?.nearbyRadius || '300'),
        reviewThreshold: String(context.draftInputs?.reviewThreshold || '0.20'),
    };
}

function syncDraftInputsToSession() {
    const session = getActiveSession();
    if (!session) return;
    const context = getSessionContext(session);
    context.draftInputs = {
        imagePath: imagePathInput.value.trim(),
        latitude: latitudeInput.value.trim(),
        longitude: longitudeInput.value.trim(),
        nearbyRadius: String(selectedNearbyRadius || 300),
        reviewThreshold: String(selectedReviewThreshold || 0.20),
    };
    saveSessions();
}

function syncDraftInputsFromSession(session = getActiveSession()) {
    const draft = getEffectiveDraftInputs(session);
    imagePathInput.value = draft.imagePath;
    latitudeInput.value = draft.latitude;
    longitudeInput.value = draft.longitude;
    selectedNearbyRadius = Math.max(100, Math.min(10000, Number(draft.nearbyRadius) || 300));
    selectedReviewThreshold = Math.max(0.01, Math.min(1, Number(draft.reviewThreshold) || 0.20));
    if (reviewThresholdInput) reviewThresholdInput.value = String(Math.round(selectedReviewThreshold * 100));
    if (nearbyRadiusInput) nearbyRadiusInput.value = selectedNearbyRadius;
}

function buildDefaultSession() {
    const now = new Date().toISOString();
    return {
        id: `session_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
        title: 'New Chat',
        customTitle: false,
        createdAt: now,
        updatedAt: now,
        messages: defaultAssistantGreeting(),
        lastImagePath: '',
        uiEntries: [
            {
                kind: 'assistant_html',
                content: marked.parse(defaultAssistantGreeting()[0].content),
            }
        ],
        tokenLog: '',
        geoState: {
            observationPoint: null,
            mapStatus: 'Waiting for latitude/longitude input.',
            nearbySummaryHtml: '',
            nearbyFeatures: [],
            backgroundData: null,
        },
        contextState: createDefaultContextState(),
    };
}

function saveSessions() {
    localStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify({
        activeSessionId,
        sessions,
    }));
}

function loadSessions() {
    try {
        const raw = localStorage.getItem(SESSION_STORAGE_KEY);
        if (!raw) return false;
        const parsed = JSON.parse(raw);
        sessions = Array.isArray(parsed.sessions) ? parsed.sessions.map((session) => normalizeSession(session)) : [];
        activeSessionId = String(parsed.activeSessionId || '');
        return sessions.length > 0;
    } catch (_) {
        sessions = [];
        activeSessionId = '';
        return false;
    }
}

function touchActiveSession() {
    const session = sessions.find((item) => item.id === activeSessionId);
    if (!session) return;
    normalizeSession(session);
    session.messages = chatHistory;
    session.lastImagePath = getSessionContext(session).committedImagePath;
    session.updatedAt = new Date().toISOString();
    if (!session.customTitle) {
        session.title = createSessionTitleFromHistory(chatHistory);
    }
    sessions.sort((a, b) => String(b.updatedAt).localeCompare(String(a.updatedAt)));
    saveSessions();
    renderSessionList();
}

function getActiveSession() {
    const session = sessions.find((item) => item.id === activeSessionId) || null;
    return session ? normalizeSession(session) : null;
}

function normalizeEllipsisText(text) {
    return String(text || "")
        .replace(/\u2026+/g, ".").replace(/\.\.\.+/g, ".").replace(/\.\./g, ".")
        // `~` is used as "approximately" in reports (e.g. `(~815 m)`); a lone
        // `~ ... ~` pair is read as strikethrough by marked. Use the real glyph.
        .replace(/~\s?(?=\d)/g, "\u2248");
}

function persistUiEntry(entry) {
    const session = getActiveSession();
    if (!session) return;
    if (!Array.isArray(session.uiEntries)) session.uiEntries = [];
    session.uiEntries.push(entry);
    touchActiveSession();
}

function persistTokenState() {
    const session = getActiveSession();
    if (!session) return;
    session.tokenLog = String(tokenStream.textContent || '');
    touchActiveSession();
}

function persistGeoState() {
    const session = getActiveSession();
    if (!session) return;
    const context = getSessionContext(session);
    context.committedGeoPoint = lastGeoPoint ? { ...lastGeoPoint } : context.committedGeoPoint || null;
    session.geoState = {
        observationPoint: lastGeoPoint ? { ...lastGeoPoint } : null,
        mapStatus: mapStatus.textContent || '',
        nearbySummaryHtml: nearbySummary.innerHTML || '',
        nearbyFeatures: Array.isArray(session.geoState?.nearbyFeatures) ? session.geoState.nearbyFeatures : [],
        backgroundData: lastGeoBackground ? { ...lastGeoBackground } : null,
    };
    touchActiveSession();
}

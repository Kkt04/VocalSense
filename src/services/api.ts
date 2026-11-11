/**
 * VocalSense API Client
 * Connects frontend with FastAPI backend for health checks, audio uploads,
 * speech-to-text transcription, and speech emotion recognition (SER).
 */

export interface HealthResponse {
  status: string;
  app_name: string;
  version: string;
  environment: string;
  timestamp: string;
}

export interface AudioMetadata {
  filename: string;
  content_type: string;
  size_bytes: number;
  size_kb: number;
  format_detected: string;
  validated: boolean;
  details?: Record<string, unknown>;
}

export interface AudioUploadResponse {
  success: boolean;
  message: string;
  audio: AudioMetadata;
  session_id?: string;
  processed_at: string;
}

export interface TranscriptionSegment {
  start: number;
  end: number;
  text: string;
  avg_logprob?: number;
  no_speech_prob?: number;
}

export interface TranscriptionResponse {
  success: boolean;
  transcript: string;
  detected_language: string;
  language_probability: number;
  duration: number;
  segments: TranscriptionSegment[];
  is_hinglish: boolean;
  model_used: string;
  processing_time_sec: number;
  message: string;
}

export interface TranscriptionHealthResponse {
  status: string;
  service: string;
  model_size: string;
  device: string;
  compute_type: string;
  model_loaded: boolean;
}

export interface VocalTraits {
  pitch_mean_hz: number;
  pitch_std_hz: number;
  rms_energy: number;
  spectral_centroid_hz: number;
  pause_fraction: number;
  duration_sec: number;
}

export interface EmotionPredictionResponse {
  success: boolean;
  emotion_label: string;
  confidence: number;
  probabilities: Record<string, number>;
  vocal_traits: VocalTraits;
  is_reliable: boolean;
  interpretation_notice: string;
  model_used: string;
  processing_time_sec: number;
}

export interface EmotionHealthResponse {
  status: string;
  service: string;
  model_loaded: boolean;
  model_name: string;
  classes: string[];
}

export interface ApiError {
  success: false;
  error: string;
  detail?: string | { error?: string; code?: string };
  code?: string;
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

/**
 * Check backend health and connectivity.
 */
export async function checkBackendHealth(): Promise<{ online: boolean; data?: HealthResponse; error?: string }> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/health`, {
      method: 'GET',
      headers: { Accept: 'application/json' },
    });
    if (!res.ok) {
      return { online: false, error: `Health check failed with status ${res.status}` };
    }
    const data: HealthResponse = await res.json();
    return { online: true, data };
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : 'Backend unreachable';
    return { online: false, error: message };
  }
}

/**
 * Check speech recognition service health.
 */
export async function checkTranscriptionHealth(): Promise<{ online: boolean; data?: TranscriptionHealthResponse; error?: string }> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/v1/transcription/health`, {
      method: 'GET',
      headers: { Accept: 'application/json' },
    });
    if (!res.ok) {
      return { online: false, error: `Transcription health failed with status ${res.status}` };
    }
    const data: TranscriptionHealthResponse = await res.json();
    return { online: true, data };
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : 'Transcription service unreachable';
    return { online: false, error: message };
  }
}

/**
 * Check speech emotion recognition service health.
 */
export async function checkEmotionHealth(): Promise<{ online: boolean; data?: EmotionHealthResponse; error?: string }> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/v1/emotion/health`, {
      method: 'GET',
      headers: { Accept: 'application/json' },
    });
    if (!res.ok) {
      return { online: false, error: `Emotion service health failed with status ${res.status}` };
    }
    const data: EmotionHealthResponse = await res.json();
    return { online: true, data };
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : 'Emotion service unreachable';
    return { online: false, error: message };
  }
}

/**
 * Upload an audio Blob or File to the FastAPI backend.
 */
export async function uploadAudio(
  audioData: Blob | File,
  filename = 'recording.webm',
  sessionId?: string
): Promise<AudioUploadResponse> {
  const formData = new FormData();
  formData.append('file', audioData, filename);
  if (sessionId) {
    formData.append('session_id', sessionId);
  }

  const res = await fetch(`${API_BASE_URL}/api/audio/upload`, {
    method: 'POST',
    body: formData,
  });

  const json = await res.json();

  if (!res.ok) {
    const errorMsg =
      typeof json.detail === 'object' && json.detail?.error
        ? json.detail.error
        : json.detail || json.error || `Upload failed with status ${res.status}`;
    throw new Error(errorMsg);
  }

  return json as AudioUploadResponse;
}

/**
 * Send audio to the speech recognition endpoint for multilingual transcription.
 */
export async function transcribeAudio(
  audioData: Blob | File,
  filename = 'recording.webm',
  language?: string,
  prompt?: string
): Promise<TranscriptionResponse> {
  const formData = new FormData();
  formData.append('file', audioData, filename);
  if (language && language !== 'auto') {
    formData.append('language', language);
  }
  if (prompt) {
    formData.append('prompt', prompt);
  }

  const res = await fetch(`${API_BASE_URL}/api/v1/transcription/transcribe`, {
    method: 'POST',
    body: formData,
  });

  const json = await res.json();

  if (!res.ok) {
    const errorMsg =
      typeof json.detail === 'object' && json.detail?.error
        ? json.detail.error
        : json.detail || json.error || `Transcription failed with status ${res.status}`;
    throw new Error(errorMsg);
  }

  return json as TranscriptionResponse;
}

/**
 * Send audio to the emotion recognition endpoint to extract acoustic traits and predict emotion patterns.
 */
export async function predictEmotion(
  audioData: Blob | File,
  filename = 'recording.webm',
  modelType?: 'wav2vec2' | 'hubert' | 'traditional_svm'
): Promise<EmotionPredictionResponse> {
  const formData = new FormData();
  formData.append('file', audioData, filename);

  const queryParams = modelType ? `?model_type=${encodeURIComponent(modelType)}` : '';
  const res = await fetch(`${API_BASE_URL}/api/v1/emotion/predict${queryParams}`, {
    method: 'POST',
    body: formData,
  });

  const json = await res.json();

  if (!res.ok) {
    const errorMsg =
      typeof json.detail === 'object' && json.detail?.error
        ? json.detail.error
        : json.detail || json.error || `Emotion prediction failed with status ${res.status}`;
    throw new Error(errorMsg);
  }

  return json as EmotionPredictionResponse;
}

export interface ConversationRespondRequest {
  transcript: string;
  conversation_id?: string;
  user_id?: string;
  detected_language?: string;
  language_code?: string;
  language_preference?: string;
  conversation_style?: string;
  emotion_estimate?: Record<string, unknown>;
  history?: { role: string; text: string; language?: string }[];
  synthesize_speech?: boolean;
}

export interface ConversationRespondResponse {
  response: string;
  response_language: string;
  language_code: string;
  conversation_style: string;
  conversation_id: string;
  user_id?: string;
  tts_available: boolean;
  tts_audio_base64?: string;
  audio_mime_type?: string;
  validation_passed: boolean;
  code_switching_preserved: boolean;
  error?: string;
}

/**
 * Request language-aware conversational response from VocalSense companion engine.
 */
export async function sendConversationMessage(
  payload: ConversationRespondRequest
): Promise<ConversationRespondResponse> {
  const res = await fetch(`${API_BASE_URL}/api/v1/conversation/respond`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
    },
    body: JSON.stringify(payload),
  });

  const json = await res.json();

  if (!res.ok) {
    const errorMsg =
      typeof json.detail === 'object' && json.detail?.error
        ? json.detail.error
        : json.detail || json.error || `Conversation failed with status ${res.status}`;
    throw new Error(errorMsg);
  }

  return json as ConversationRespondResponse;
}

/**
 * Retrieve chronological conversation history from MongoDB.
 */
export async function getConversationHistory(
  conversationId: string,
  limit = 50,
  skip = 0
): Promise<{ conversation_id: string; messages: any[]; count: number }> {
  const res = await fetch(
    `${API_BASE_URL}/api/v1/conversation/history/${encodeURIComponent(conversationId)}?limit=${limit}&skip=${skip}`,
    {
      method: 'GET',
      headers: { Accept: 'application/json' },
    }
  );
  if (!res.ok) {
    throw new Error(`Failed to fetch history: status ${res.status}`);
  }
  return await res.json();
}

/**
 * Check conversation service health and supported languages.
 */
export async function checkConversationHealth(): Promise<{ online: boolean; data?: any; error?: string }> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/v1/conversation/health`, {
      method: 'GET',
      headers: { Accept: 'application/json' },
    });
    if (!res.ok) {
      return { online: false, error: `Conversation health failed with status ${res.status}` };
    }
    const data = await res.json();
    return { online: true, data };
  } catch (err: unknown) {
    return { online: false, error: err instanceof Error ? err.message : 'Conversation service unreachable' };
  }
}

/**
 * Check deep component-level health (DB, STT, Emotion, LLM, TTS).
 */
export async function checkComponentHealth(): Promise<{ online: boolean; data?: any; error?: string }> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/health/components`, {
      method: 'GET',
      headers: { Accept: 'application/json' },
    });
    if (!res.ok) {
      return { online: false, error: `Component health check returned status ${res.status}` };
    }
    const data = await res.json();
    return { online: true, data };
  } catch (err: unknown) {
    return { online: false, error: err instanceof Error ? err.message : 'Components health unreachable' };
  }
}

export interface ConversationItem {
  conversation_id: string;
  user_id?: string;
  style?: string;
  language?: string;
  created_at: string;
  updated_at: string;
  message_count: number;
  last_message_preview?: string;
}

export interface ConversationListResponse {
  success: boolean;
  conversations: ConversationItem[];
  count: number;
  limit: number;
  skip: number;
  user_id?: string;
}

/**
 * List past conversations from MongoDB with pagination.
 */
export async function listConversations(
  limit = 20,
  skip = 0,
  userId?: string
): Promise<ConversationListResponse> {
  const params = new URLSearchParams({ limit: String(limit), skip: String(skip) });
  if (userId) params.append('user_id', userId);

  const res = await fetch(`${API_BASE_URL}/api/v1/conversation/list?${params.toString()}`, {
    method: 'GET',
    headers: { Accept: 'application/json' },
  });
  if (!res.ok) {
    throw new Error(`Failed to list conversations: status ${res.status}`);
  }
  return await res.json();
}

/**
 * Delete a specific conversation and all associated messages from MongoDB.
 */
export async function deleteConversation(conversationId: string): Promise<{ success: boolean; message: string }> {
  const res = await fetch(`${API_BASE_URL}/api/v1/conversation/${encodeURIComponent(conversationId)}`, {
    method: 'DELETE',
    headers: { Accept: 'application/json' },
  });
  if (!res.ok) {
    throw new Error(`Failed to delete conversation: status ${res.status}`);
  }
  return await res.json();
}

/**
 * GDPR/DPDP Right-to-Erasure: permanently erase all data for a user.
 */
export async function deleteUserData(userId: string): Promise<{ success: boolean; purged: any; message: string }> {
  const res = await fetch(`${API_BASE_URL}/api/v1/conversation/user/${encodeURIComponent(userId)}/data`, {
    method: 'DELETE',
    headers: { Accept: 'application/json' },
  });
  if (!res.ok) {
    throw new Error(`Failed to purge user data: status ${res.status}`);
  }
  return await res.json();
}

export { API_BASE_URL };

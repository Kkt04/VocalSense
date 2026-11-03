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
  filename = 'recording.webm'
): Promise<EmotionPredictionResponse> {
  const formData = new FormData();
  formData.append('file', audioData, filename);

  const res = await fetch(`${API_BASE_URL}/api/v1/emotion/predict`, {
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

export { API_BASE_URL };

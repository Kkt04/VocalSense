/**
 * VocalSense API Client
 * Connects frontend with FastAPI backend for health checks and audio uploads.
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

export { API_BASE_URL };

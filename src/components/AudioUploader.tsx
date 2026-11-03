import { useState, useRef, DragEvent, ChangeEvent } from 'react';
import Icon from './Icon';
import { uploadAudio, transcribeAudio, AudioUploadResponse, TranscriptionResponse } from '../services/api';

interface AudioUploaderProps {
  onUploadSuccess?: (response: AudioUploadResponse) => void;
  onTranscriptionSuccess?: (response: TranscriptionResponse) => void;
  onCancel?: () => void;
}

const MAX_SIZE_MB = 25;
const ACCEPTED_EXTENSIONS = ['.wav', '.webm', '.ogg', '.mp3', '.m4a', '.mp4'];

export default function AudioUploader({
  onUploadSuccess,
  onTranscriptionSuccess,
  onCancel,
}: AudioUploaderProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [languagePreference, setLanguagePreference] = useState<'auto' | 'en' | 'hi'>('auto');
  const [error, setError] = useState<string | null>(null);
  const [successResponse, setSuccessResponse] = useState<TranscriptionResponse | AudioUploadResponse | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const validateFile = (file: File): string | null => {
    const ext = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!ACCEPTED_EXTENSIONS.includes(ext)) {
      return `Unsupported file format. Please upload: ${ACCEPTED_EXTENSIONS.join(', ')}`;
    }
    if (file.size > MAX_SIZE_MB * 1024 * 1024) {
      return `File exceeds the maximum limit of ${MAX_SIZE_MB} MB (${(file.size / (1024 * 1024)).toFixed(1)} MB).`;
    }
    return null;
  };

  const handleFile = (file: File) => {
    setError(null);
    setSuccessResponse(null);
    const validationError = validateFile(file);
    if (validationError) {
      setError(validationError);
      setSelectedFile(null);
      return;
    }
    setSelectedFile(file);
  };

  const handleDragOver = (e: DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFile(e.target.files[0]);
    }
  };

  const handleUploadAndTranscribe = async () => {
    if (!selectedFile) return;

    setUploading(true);
    setError(null);

    try {
      if (onTranscriptionSuccess) {
        // Run full speech recognition via Whisper
        const response = await transcribeAudio(
          selectedFile,
          selectedFile.name,
          languagePreference === 'auto' ? undefined : languagePreference
        );
        setSuccessResponse(response);
        onTranscriptionSuccess(response);
      } else {
        // Fallback to upload validation
        const response = await uploadAudio(selectedFile, selectedFile.name);
        setSuccessResponse(response);
        if (onUploadSuccess) {
          onUploadSuccess(response);
        }
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Processing failed. Check if backend is running.';
      setError(msg);
    } finally {
      setUploading(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setError(null);
    setSuccessResponse(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const isTranscriptionRes = (res: unknown): res is TranscriptionResponse => {
    return typeof res === 'object' && res !== null && 'transcript' in res;
  };

  return (
    <div className="rounded-3xl border border-ink/10 bg-surface p-6 shadow-soft">
      <div className="flex items-center justify-between pb-3">
        <h3 className="font-serif text-lg">Upload Audio Recording</h3>
        {onCancel && (
          <button
            type="button"
            className="text-xs text-muted hover:text-ink"
            onClick={onCancel}
          >
            Close
          </button>
        )}
      </div>

      <p className="mb-4 text-xs text-muted">
        Select a pre-recorded audio file to transcribe with Whisper. Supports English, Hindi, and Hinglish (.wav, .webm, .ogg, .mp3, .m4a up to 25MB).
      </p>

      {/* Drag & Drop Area */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed p-6 text-center transition ${
          isDragging
            ? 'border-accent bg-lav/50'
            : selectedFile
            ? 'border-accent/60 bg-lav/20'
            : 'border-ink/15 hover:border-ink/30 hover:bg-ink/5'
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept={ACCEPTED_EXTENSIONS.join(',')}
          className="hidden"
          onChange={handleFileInputChange}
        />

        <div className="mb-2 grid h-12 w-12 place-items-center rounded-full bg-lav text-accent">
          <Icon name="mic" size={24} />
        </div>

        {selectedFile ? (
          <div>
            <p className="text-sm font-semibold text-ink">{selectedFile.name}</p>
            <p className="text-xs text-muted">{(selectedFile.size / 1024).toFixed(1)} KB</p>
          </div>
        ) : (
          <div>
            <p className="text-sm font-medium">Click or drag audio file here</p>
            <p className="mt-1 text-xs text-muted">WAV, WebM, OGG, MP3, M4A up to 25MB</p>
          </div>
        )}
      </div>

      {/* Language Preference Selector */}
      <div className="mt-4 flex flex-wrap items-center justify-between gap-2 border-t border-ink/10 pt-3">
        <span className="text-xs font-medium text-muted">Language Detection:</span>
        <div className="inline-flex rounded-full bg-ink/5 p-0.5 text-xs">
          {(['auto', 'en', 'hi'] as const).map((l) => (
            <button
              key={l}
              type="button"
              onClick={() => setLanguagePreference(l)}
              className={`rounded-full px-3 py-1 font-medium transition ${
                languagePreference === l ? 'bg-accent text-onaccent shadow-xs' : 'text-muted hover:text-ink'
              }`}
            >
              {l === 'auto' ? 'Auto-Detect (EN/HI/Hinglish)' : l === 'en' ? 'English' : 'Hindi'}
            </button>
          ))}
        </div>
      </div>

      {/* Error alert */}
      {error && (
        <div className="mt-4 rounded-2xl bg-pink p-3 text-xs text-ink leading-relaxed">
          <span className="font-semibold">Notice: </span>
          {error}
        </div>
      )}

      {/* Success alert with metadata */}
      {successResponse && (
        <div className="mt-4 rounded-2xl bg-sage p-4 text-xs text-ink leading-relaxed">
          {isTranscriptionRes(successResponse) ? (
            <div>
              <p className="font-semibold text-sm mb-1">✓ Transcribed via Whisper</p>
              <p className="text-ink text-sm bg-surface/70 rounded-xl p-2.5 my-2 border border-ink/10 font-serif italic">
                "{successResponse.transcript}"
              </p>
              <div className="grid grid-cols-2 gap-2 text-[11px] text-muted">
                <div>Language: <span className="font-semibold uppercase text-ink">{successResponse.detected_language}</span></div>
                <div>Duration: <span className="font-semibold text-ink">{successResponse.duration}s</span></div>
              </div>
            </div>
          ) : (
            <div>
              <p className="font-semibold text-sm mb-1">✓ Audio Validated by Backend</p>
              <p className="text-xs text-muted">Format: {successResponse.audio.format_detected} · {successResponse.audio.size_kb} KB</p>
            </div>
          )}
        </div>
      )}

      {/* Action Buttons */}
      <div className="mt-5 flex items-center justify-end gap-2">
        {selectedFile && !uploading && (
          <button type="button" className="btn btn-ghost !px-4 !py-2 text-xs" onClick={handleReset}>
            Clear
          </button>
        )}
        <button
          type="button"
          disabled={!selectedFile || uploading}
          onClick={handleUploadAndTranscribe}
          className="btn btn-primary !px-5 !py-2 text-xs"
        >
          {uploading ? 'Transcribing with Whisper...' : 'Transcribe Audio'}
        </button>
      </div>
    </div>
  );
}

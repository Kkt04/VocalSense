import { useCallback, useEffect, useRef, useState } from 'react';

export type RecorderState = 'idle' | 'requesting_permission' | 'listening' | 'paused' | 'stopped' | 'error';

export interface UseAudioRecorderReturn {
  state: RecorderState;
  duration: number;
  isMuted: boolean;
  bars: number[];
  level: number;
  errorMessage: string | null;
  audioBlob: Blob | null;
  startRecording: () => Promise<boolean>;
  pauseRecording: () => void;
  resumeRecording: () => void;
  stopRecording: () => Promise<Blob | null>;
  cancelRecording: () => void;
  toggleMute: () => void;
  clearError: () => void;
}

const DEFAULT_BARS_COUNT = 36;
const QUIET_BARS = Array<number>(DEFAULT_BARS_COUNT).fill(0.06);

/**
 * Determine supported audio MIME type in the browser.
 */
function getSupportedMimeType(): string {
  if (typeof MediaRecorder === 'undefined') return '';
  const candidateTypes = [
    'audio/webm;codecs=opus',
    'audio/webm',
    'audio/ogg;codecs=opus',
    'audio/mp4',
    'audio/aac',
    'audio/wav',
  ];
  for (const type of candidateTypes) {
    if (MediaRecorder.isTypeSupported(type)) {
      return type;
    }
  }
  return '';
}

export function useAudioRecorder(barCount = DEFAULT_BARS_COUNT): UseAudioRecorderReturn {
  const [state, setState] = useState<RecorderState>('idle');
  const [duration, setDuration] = useState(0);
  const [isMuted, setIsMuted] = useState(false);
  const [bars, setBars] = useState<number[]>(QUIET_BARS);
  const [level, setLevel] = useState(0);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);

  const mediaStreamRef = useRef<MediaStream | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const sourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const animFrameRef = useRef<number | null>(null);
  const timerIntervalRef = useRef<number | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const stopResolverRef = useRef<((blob: Blob | null) => void) | null>(null);

  // Stop analysis loop
  const stopAnalysisLoop = useCallback(() => {
    if (animFrameRef.current !== null) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = null;
    }
    setBars(QUIET_BARS);
    setLevel(0);
  }, []);

  // Clean up all hardware/audio resources
  const cleanupResources = useCallback(() => {
    stopAnalysisLoop();

    if (timerIntervalRef.current !== null) {
      clearInterval(timerIntervalRef.current);
      timerIntervalRef.current = null;
    }

    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      try {
        mediaRecorderRef.current.stop();
      } catch {
        // already stopped
      }
    }
    mediaRecorderRef.current = null;

    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }

    if (sourceNodeRef.current) {
      sourceNodeRef.current.disconnect();
      sourceNodeRef.current = null;
    }

    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      try {
        audioContextRef.current.close();
      } catch {
        // ignore
      }
      audioContextRef.current = null;
    }
    analyserRef.current = null;
  }, [stopAnalysisLoop]);

  // Audio visualization loop
  const startAnalysisLoop = useCallback(() => {
    const analyser = analyserRef.current;
    if (!analyser) return;

    const dataArray = new Uint8Array(analyser.frequencyBinCount);

    const updateBars = () => {
      analyser.getByteFrequencyData(dataArray);

      // Compute normalized values across barCount slices
      const step = Math.max(1, Math.floor(dataArray.length / barCount));
      let totalEnergy = 0;
      const newBars: number[] = [];

      for (let i = 0; i < barCount; i++) {
        let sum = 0;
        let count = 0;
        for (let j = 0; j < step && i * step + j < dataArray.length; j++) {
          sum += dataArray[i * step + j];
          count++;
        }
        const avg = count > 0 ? sum / count : 0;
        const normalized = Math.min(1, Math.max(0.06, avg / 255));
        newBars.push(normalized);
        totalEnergy += normalized;
      }

      setBars(newBars);
      setLevel(totalEnergy / barCount);

      animFrameRef.current = requestAnimationFrame(updateBars);
    };

    animFrameRef.current = requestAnimationFrame(updateBars);
  }, [barCount]);

  // Start real recording
  const startRecording = useCallback(async (): Promise<boolean> => {
    cleanupResources();
    setAudioBlob(null);
    setErrorMessage(null);
    setDuration(0);
    setIsMuted(false);
    chunksRef.current = [];

    // Verify browser support
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setState('error');
      setErrorMessage('Your browser does not support microphone access. Please try a modern browser like Chrome, Firefox, or Safari.');
      return false;
    }

    setState('requesting_permission');

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      mediaStreamRef.current = stream;

      // Initialize Web Audio API Analyser
      const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      const audioCtx = new AudioCtx();
      audioContextRef.current = audioCtx;

      if (audioCtx.state === 'suspended') {
        await audioCtx.resume();
      }

      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 128;
      analyser.smoothingTimeConstant = 0.8;
      analyserRef.current = analyser;

      const source = audioCtx.createMediaStreamSource(stream);
      source.connect(analyser);
      sourceNodeRef.current = source;

      // Initialize MediaRecorder
      const mimeType = getSupportedMimeType();
      const options: MediaRecorderOptions = mimeType ? { mimeType } : {};
      const recorder = new MediaRecorder(stream, options);
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      recorder.onstop = () => {
        const finalMime = recorder.mimeType || 'audio/webm';
        const blob = chunksRef.current.length > 0 ? new Blob(chunksRef.current, { type: finalMime }) : null;
        setAudioBlob(blob);
        if (stopResolverRef.current) {
          stopResolverRef.current(blob);
          stopResolverRef.current = null;
        }
      };

      recorder.onerror = (event) => {
        setState('error');
        setErrorMessage(`Recording error: ${(event as unknown as { error?: { message?: string } }).error?.message || 'Unknown recording error'}`);
      };

      recorder.start(250); // capture chunks every 250ms
      setState('listening');
      startAnalysisLoop();

      // Start recording timer
      timerIntervalRef.current = window.setInterval(() => {
        setDuration((prev) => prev + 1);
      }, 1000);

      return true;
    } catch (err: unknown) {
      cleanupResources();
      setState('error');

      if (err instanceof DOMException) {
        if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
          setErrorMessage('Microphone access was denied. Please allow microphone permission in your browser settings to continue.');
        } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
          setErrorMessage('No microphone device was detected on your system. Please plug in a microphone and try again.');
        } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
          setErrorMessage('Your microphone is currently in use by another application. Please close other audio applications and try again.');
        } else {
          setErrorMessage(`Microphone error: ${err.message}`);
        }
      } else {
        setErrorMessage('Failed to access microphone. Please check your system settings.');
      }
      return false;
    }
  }, [cleanupResources, startAnalysisLoop]);

  // Pause recording
  const pauseRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.pause();
      stopAnalysisLoop();
      if (timerIntervalRef.current !== null) {
        clearInterval(timerIntervalRef.current);
        timerIntervalRef.current = null;
      }
      setState('paused');
    }
  }, [stopAnalysisLoop]);

  // Resume recording
  const resumeRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'paused') {
      mediaRecorderRef.current.resume();
      startAnalysisLoop();
      timerIntervalRef.current = window.setInterval(() => {
        setDuration((prev) => prev + 1);
      }, 1000);
      setState('listening');
    }
  }, [startAnalysisLoop]);

  // Stop recording and return Blob
  const stopRecording = useCallback((): Promise<Blob | null> => {
    return new Promise((resolve) => {
      stopResolverRef.current = resolve;
      stopAnalysisLoop();

      if (timerIntervalRef.current !== null) {
        clearInterval(timerIntervalRef.current);
        timerIntervalRef.current = null;
      }

      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        setState('stopped');
        mediaRecorderRef.current.stop();
      } else {
        setState('idle');
        resolve(null);
      }
    });
  }, [stopAnalysisLoop]);

  // Cancel recording and discard
  const cancelRecording = useCallback(() => {
    cleanupResources();
    setAudioBlob(null);
    setDuration(0);
    setState('idle');
  }, [cleanupResources]);

  // Toggle Mute
  const toggleMute = useCallback(() => {
    if (mediaStreamRef.current) {
      const audioTracks = mediaStreamRef.current.getAudioTracks();
      const nextMuted = !isMuted;
      audioTracks.forEach((track) => {
        track.enabled = !nextMuted;
      });
      setIsMuted(nextMuted);
      if (nextMuted) {
        setBars(QUIET_BARS);
        setLevel(0);
      }
    }
  }, [isMuted]);

  const clearError = useCallback(() => {
    setErrorMessage(null);
    setState('idle');
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      cleanupResources();
    };
  }, [cleanupResources]);

  return {
    state,
    duration,
    isMuted,
    bars,
    level,
    errorMessage,
    audioBlob,
    startRecording,
    pauseRecording,
    resumeRecording,
    stopRecording,
    cancelRecording,
    toggleMute,
    clearError,
  };
}

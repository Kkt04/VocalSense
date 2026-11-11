import { useCallback, useEffect, useRef, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import Icon from '../components/Icon';
import Logo from '../components/Logo';
import Orb, { OrbMode } from '../components/Orb';
import ThemeToggle from '../components/ThemeToggle';
import Waveform from '../components/Waveform';
import AudioUploader from '../components/AudioUploader';
import { useAudioRecorder } from '../hooks/useAudioRecorder';
import { useSimulatedAudio } from '../hooks/useSimulatedAudio';
import {
  checkBackendHealth,
  checkTranscriptionHealth,
  checkEmotionHealth,
  transcribeAudio,
  predictEmotion,
  sendConversationMessage,
  checkConversationHealth,
  checkComponentHealth,
  listConversations,
  deleteConversation,
  deleteUserData,
  getConversationHistory,
  TranscriptionResponse,
  EmotionPredictionResponse,
  ConversationItem,
} from '../services/api';

type Phase = 'idle' | 'permission' | 'listening' | 'paused' | 'processing' | 'analysing' | 'responding' | 'error';
type StyleId = 'listen' | 'think' | 'practical' | 'distract';
type Msg = {
  id: number;
  role: 'user' | 'ai';
  text: string;
  isEdited?: boolean;
  detectedLanguage?: string;
  languageProb?: number;
  isError?: boolean;
  errorMessage?: string;
  lastTranscript?: string;
};

/* ---------- CONVERSATION STYLES ---------- */
const STYLES: { id: StyleId; name: string; hint: string; reply: string }[] = [
  {
    id: 'listen',
    name: 'Just listen',
    hint: 'Gentle acknowledgement and follow-up questions.',
    reply: "That sounds like a lot to carry, especially when you've already put in the work. What does that thought usually sound like when it shows up?",
  },
  {
    id: 'think',
    name: 'Help me think',
    hint: 'Organise thoughts and explore perspectives.',
    reply: "Let's separate two things: what you've prepared, and what you can't control. Which specific part feels most uncertain right now?",
  },
  {
    id: 'practical',
    name: 'Practical suggestions',
    hint: 'Clear, actionable steps.',
    reply: "Here are three small steps. 1) Do one timed practice round today. 2) Outline a one-minute story for your strongest experience. 3) Plan a calming routine for the evening before.",
  },
  {
    id: 'distract',
    name: 'Distract me',
    hint: 'Light topics and simple activities.',
    reply: "Let's take a short pause from deadlines. Quick question: if you could instantly master any creative skill, what would it be?",
  },
];

const STATUS: Record<Phase, string> = {
  idle: "Whenever you're ready.",
  permission: 'Allow microphone access to begin.',
  listening: "I'm listening. Take your time.",
  paused: 'Paused. Resume when you like.',
  processing: 'Transcribing speech & extracting acoustic traits...',
  analysing: 'Analyzing speech container & vocal patterns...',
  responding: "Here's something to think about.",
  error: 'Something went wrong.',
};

const fmt = (s: number) => `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;

const STYLE_MAP: Record<StyleId, string> = {
  listen: 'just_listen',
  think: 'help_me_think',
  practical: 'practical_suggestions',
  distract: 'distract_me',
};

export default function Conversation() {
  const [userId] = useState<string>(() => {
    let uid = localStorage.getItem('vocalsense_user_id');
    if (!uid) {
      uid = 'user_' + Math.random().toString(36).substring(2, 10);
      localStorage.setItem('vocalsense_user_id', uid);
    }
    return uid;
  });
  const [activeConversationId, setActiveConversationId] = useState<string>(
    () => 'conv_' + Math.random().toString(36).substring(2, 10)
  );
  const [historyOpen, setHistoryOpen] = useState(false);
  const [historyList, setHistoryList] = useState<ConversationItem[]>([]);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [purgeConfirmOpen, setPurgeConfirmOpen] = useState(false);

  const [phase, setPhase] = useState<Phase>('idle');
  const [style, setStyle] = useState<StyleId>('listen');
  const [mode, setMode] = useState<'voice' | 'text'>('voice');
  const [languagePreference] = useState<'auto'>('auto');
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [editing, setEditing] = useState<number | null>(null);
  const [draft, setDraft] = useState('');
  const [progress, setProgress] = useState(0);
  const [paused, setPaused] = useState(false);
  const [spkMuted, setSpkMuted] = useState(false);
  const [showInsights, setShowInsights] = useState(false);
  const [hasInsights, setHasInsights] = useState(false);
  const [confirmEnd, setConfirmEnd] = useState(false);
  const [showUploader, setShowUploader] = useState(false);
  const [demoMode, setDemoMode] = useState(false);
  const [transcriptionNotice, setTranscriptionNotice] = useState<string | null>(null);
  const [backendStatus, setBackendStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  const [sttModelName, setSttModelName] = useState<string>('');
  const [emotionModelName, setEmotionModelName] = useState<string>('');
  const [customError, setCustomError] = useState<string | null>(null);
  const [lastAudioBlob, setLastAudioBlob] = useState<Blob | null>(null);
  const [emotionData, setEmotionData] = useState<EmotionPredictionResponse | null>(null);

  const timers = useRef<number[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const later = useCallback((fn: () => void, ms: number) => {
    timers.current.push(window.setTimeout(fn, ms));
  }, []);
  const clearTimers = () => {
    timers.current.forEach(clearTimeout);
    timers.current = [];
  };
  useEffect(() => clearTimers, []);

  // Auto-scroll conversation to latest message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [msgs]);

  // Real microphone audio recorder hook
  const recorder = useAudioRecorder(36);

  // Fallback simulated audio (used during demo mode or AI speaking simulation)
  const speaking = phase === 'responding' && mode === 'voice' && !paused;
  const isSimulatedActive = (demoMode && phase === 'listening') || speaking;
  const simulatedAudio = useSimulatedAudio(isSimulatedActive, 36);

  // Check backend, Whisper, and Emotion model health
  const checkHealth = useCallback(async () => {
    setBackendStatus('checking');
    const [backRes, sttRes, emoRes] = await Promise.all([
      checkBackendHealth(),
      checkTranscriptionHealth(),
      checkEmotionHealth(),
    ]);

    if (backRes.online) {
      setBackendStatus('online');
      if (sttRes.online && sttRes.data) {
        setSttModelName(`Whisper ${sttRes.data.model_size}`);
      }
      if (emoRes.online && emoRes.data) {
        setEmotionModelName(emoRes.data.model_name || 'SVM (RAVDESS)');
      }
    } else {
      setBackendStatus('offline');
      setSttModelName('');
      setEmotionModelName('');
    }
  }, []);

  useEffect(() => {
    checkHealth();
  }, [checkHealth]);

  // Audio visualizer bars and level
  const isListeningReal = !demoMode && (phase === 'listening' || recorder.state === 'listening');
  const bars = isListeningReal
    ? recorder.bars
    : isSimulatedActive
    ? simulatedAudio.bars
    : Array<number>(36).fill(0.06);

  const level = isListeningReal
    ? recorder.level
    : isSimulatedActive
    ? simulatedAudio.level
    : 0;

  const orb: OrbMode =
    isListeningReal || (demoMode && phase === 'listening')
      ? 'listening'
      : speaking
      ? 'speaking'
      : phase === 'processing' || phase === 'analysing'
      ? 'thinking'
      : 'idle';

  // Handle recorder error
  useEffect(() => {
    if (recorder.errorMessage) {
      setCustomError(recorder.errorMessage);
      setPhase('error');
    }
  }, [recorder.errorMessage]);

  // AI playback progression
  useEffect(() => {
    if (phase !== 'responding') return;
    if (mode === 'text') {
      const t = setTimeout(() => setPhase('idle'), 2500);
      return () => clearTimeout(t);
    }
    if (paused) return;
    const id = setInterval(() => setProgress((p) => Math.min(100, p + 2)), 120);
    return () => clearInterval(id);
  }, [phase, mode, paused]);

  useEffect(() => {
    if (phase === 'responding' && mode === 'voice' && progress >= 100) {
      setPhase('idle');
    }
  }, [progress, phase, mode]);

  // Start real recording immediately
  const beginListeningReal = async () => {
    setDemoMode(false);
    setTranscriptionNotice(null);
    setCustomError(null);
    const ok = await recorder.startRecording();
    if (ok) {
      setPhase('listening');
    } else {
      setCustomError(
        recorder.errorMessage ||
        'Microphone access was denied or is unavailable. Please grant microphone permissions in your browser and try again.'
      );
      setPhase('error');
    }
  };

  // Start demo simulation mode
  const beginListeningDemo = () => {
    setDemoMode(true);
    setTranscriptionNotice('Running in Demo Mode (simulated speech & insights)');
    setCustomError(null);
    setPhase('listening');
  };

  // Process transcription & emotion response into conversation state
  const handleTranscriptionAndEmotionResult = (
    sttResult: TranscriptionResponse,
    emoResult?: EmotionPredictionResponse | null
  ) => {
    const transcriptText = sttResult.transcript.trim() || "(No speech detected in recording)";
    const langLabel = sttResult.detected_language === 'hinglish' ? 'Hinglish' : sttResult.detected_language.toUpperCase();
    const confidencePct = Math.round(sttResult.language_probability * 100);

    let notice = `✓ Transcribed in ${sttResult.processing_time_sec}s · Language: ${langLabel} (${confidencePct}%)`;
    if (emoResult && emoResult.is_reliable) {
      notice += ` · Vocal pattern: ${emoResult.emotion_label}-like (${Math.round(emoResult.confidence * 100)}%)`;
    }

    setTranscriptionNotice(notice);

    if (emoResult) {
      setEmotionData(emoResult);
      setHasInsights(true);
    }

    // Append user message with transcript
    setMsgs((m) => [
      ...m,
      {
        id: Date.now(),
        role: 'user',
        text: transcriptText,
        detectedLanguage: sttResult.detected_language,
        languageProb: sttResult.language_probability,
      },
    ]);

    setPhase('analysing');

    // Call multilingual emotion-aware conversation response engine
    sendConversationMessage({
      transcript: transcriptText,
      conversation_id: activeConversationId,
      user_id: userId,
      detected_language: sttResult.detected_language,
      language_preference: languagePreference === 'auto' ? undefined : languagePreference,
      conversation_style: STYLE_MAP[style],
      emotion_estimate: emoResult ? {
        predicted_emotion: emoResult.emotion_label,
        probabilities: emoResult.probabilities,
        is_reliable: emoResult.is_reliable,
      } : undefined,
      synthesize_speech: !spkMuted,
    })
      .then((convRes) => {
        if (convRes.error && !convRes.response) {
          throw new Error(convRes.error);
        }

        setMsgs((m) => [
          ...m,
          {
            id: Date.now() + 1,
            role: 'ai',
            text: convRes.response,
            detectedLanguage: convRes.response_language,
          },
        ]);

        // Synthesize TTS audio if available
        if (convRes.tts_audio_base64 && !spkMuted) {
          try {
            const audio = new Audio(`data:${convRes.audio_mime_type || 'audio/mpeg'};base64,${convRes.tts_audio_base64}`);
            audio.play().catch((e) => console.log('Audio autoplay info:', e));
          } catch (audioErr) {
            console.warn('TTS playback issue:', audioErr);
          }
        }

        setProgress(0);
        setPaused(false);
        setPhase('responding');
      })
      .catch((convErr) => {
        console.warn('Conversation API error (no canned fallback):', convErr);
        const errMsg = convErr instanceof Error ? convErr.message : 'Conversational AI service temporarily unavailable. Please verify LLM configuration or retry.';
        setMsgs((m) => [
          ...m,
          {
            id: Date.now() + 1,
            role: 'ai',
            text: '',
            isError: true,
            errorMessage: errMsg,
            lastTranscript: transcriptText,
          },
        ]);
        setPhase('idle');
      });
  };

  // Retry generating AI response without needing to re-record
  const retryAiResponse = (transcriptToRetry: string) => {
    setPhase('analysing');
    sendConversationMessage({
      transcript: transcriptToRetry,
      conversation_id: activeConversationId,
      user_id: userId,
      conversation_style: STYLE_MAP[style],
      synthesize_speech: !spkMuted,
    })
      .then((convRes) => {
        if (convRes.error && !convRes.response) {
          throw new Error(convRes.error);
        }
        setMsgs((m) => [
          ...m.filter((x) => !x.isError),
          {
            id: Date.now() + 1,
            role: 'ai',
            text: convRes.response,
            detectedLanguage: convRes.response_language,
          },
        ]);
        setProgress(0);
        setPaused(false);
        setPhase('responding');
      })
      .catch((err) => {
        const errMsg = err instanceof Error ? err.message : 'Retry failed. AI service remains unavailable.';
        setMsgs((m) => [
          ...m.filter((x) => !x.isError),
          {
            id: Date.now() + 1,
            role: 'ai',
            text: '',
            isError: true,
            errorMessage: errMsg,
            lastTranscript: transcriptToRetry,
          },
        ]);
        setPhase('idle');
      });
  };

  // Past conversation history actions
  const fetchHistoryList = async () => {
    setLoadingHistory(true);
    try {
      const res = await listConversations(30, 0, userId);
      if (res && res.conversations) {
        setHistoryList(res.conversations);
      }
    } catch (e) {
      console.warn('Failed to load past conversations:', e);
    } finally {
      setLoadingHistory(false);
    }
  };

  const handleOpenHistory = () => {
    setHistoryOpen(true);
    fetchHistoryList();
  };

  const handleSelectConversation = async (cid: string) => {
    try {
      const hist = await getConversationHistory(cid);
      if (hist && hist.messages) {
        const loadedMsgs: Msg[] = hist.messages.map((m: any, idx: number) => ({
          id: Date.now() + idx,
          role: m.role === 'assistant' ? 'ai' : 'user',
          text: m.text,
          detectedLanguage: m.detected_language,
        }));
        setMsgs(loadedMsgs);
        setActiveConversationId(cid);
        setHistoryOpen(false);
        setTranscriptionNotice(`Loaded conversation: ${cid}`);
      }
    } catch (e) {
      console.warn('Failed to load conversation history:', e);
    }
  };

  const handleDeleteConversation = async (cid: string, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await deleteConversation(cid);
      setHistoryList((prev) => prev.filter((c) => c.conversation_id !== cid));
      if (activeConversationId === cid) {
        setMsgs([]);
        setActiveConversationId('conv_' + Math.random().toString(36).substring(2, 10));
      }
    } catch (err) {
      console.warn('Failed to delete conversation:', err);
    }
  };

  const handlePurgeAllData = async () => {
    try {
      await deleteUserData(userId);
      setMsgs([]);
      setHistoryList([]);
      setPurgeConfirmOpen(false);
      setHistoryOpen(false);
      setTranscriptionNotice('All user data and conversation history purged permanently.');
    } catch (err) {
      console.warn('Failed to purge user data:', err);
    }
  };

  // Stop recording and process with both Whisper STT and Emotion SER
  const stop = async () => {
    setPhase('processing');

    if (demoMode) {
      later(() => {
        setMsgs((m) => [
          ...m,
          {
            id: Date.now(),
            role: 'user',
            text: "Yaar, mujhe kal presentation complete karna hai, lekin thoda stress lag raha hai.",
            detectedLanguage: 'hinglish',
            languageProb: 0.94,
          },
        ]);
        setHasInsights(true);
        setEmotionData({
          success: true,
          emotion_label: 'calm',
          confidence: 0.62,
          probabilities: { calm: 0.62, sad: 0.18, neutral: 0.12, happy: 0.08 },
          vocal_traits: {
            pitch_mean_hz: 185.4,
            pitch_std_hz: 22.8,
            rms_energy: 0.048,
            spectral_centroid_hz: 1740.0,
            pause_fraction: 0.18,
            duration_sec: 4.2,
          },
          is_reliable: true,
          interpretation_notice: "Identifies measurable acoustic patterns in voice.",
          model_used: "Wav2Vec2 Deep Speech Classifier",
          processing_time_sec: 0.08,
        });
        setPhase('analysing');
      }, 1200);

      later(() => {
        const demoHinglishReply =
          style === 'listen'
            ? "Arey, main samajh sakta hoon ki presentation ka stress ho raha hai. Tension mat lo, take your time, main sun raha hoon."
            : style === 'think'
            ? "Chalo isko step-by-step dekhte hain. Presentation ka kaunsa slide ya topic sabse zyada tricky lag raha hai?"
            : style === 'practical'
            ? "Pehle ek deep breath lo aur 5 minute ka break lo. Aaj bas slides ka outline finish karo, rehearsal kal subah karenge."
            : "Chalo thoda mind distract karte hain! Fun fact: Sea otters sote waqt ek doosre ka haath pakad ke sote hain. Tumhara favourite comfort show kaunsa hai?";

        setMsgs((m) => [...m, { id: Date.now() + 1, role: 'ai', text: demoHinglishReply, detectedLanguage: 'hinglish' }]);
        setProgress(0);
        setPaused(false);
        setPhase('responding');
      }, 2500);
      return;
    }

    // Stop real recording
    const blob = await recorder.stopRecording();

    if (blob) {
      setLastAudioBlob(blob);
      try {
        setTranscriptionNotice('Analyzing audio: transcribing words and measuring acoustic patterns...');

        const filename = `recording.${blob.type.includes('wav') ? 'wav' : 'webm'}`;

        // Execute STT and SER in parallel
        const [sttOutcome, emoOutcome] = await Promise.allSettled([
          transcribeAudio(blob, filename, languagePreference === 'auto' ? undefined : languagePreference),
          predictEmotion(blob, filename),
        ]);

        if (sttOutcome.status === 'fulfilled') {
          const emoResult = emoOutcome.status === 'fulfilled' ? emoOutcome.value : null;
          handleTranscriptionAndEmotionResult(sttOutcome.value, emoResult);
        } else {
          const errorMsg = sttOutcome.reason?.message || 'Transcription failed';
          setCustomError(`Speech recognition failed: ${errorMsg}`);
          setPhase('error');
        }
      } catch (err: unknown) {
        const errorMsg = err instanceof Error ? err.message : 'Processing failed';
        setCustomError(`Processing error: ${errorMsg}`);
        setPhase('error');
      }
    } else {
      setCustomError('No audio data was captured.');
      setPhase('error');
    }
  };

  // Retry transcription & emotion analysis
  const retryTranscription = async () => {
    if (!lastAudioBlob) {
      retry();
      return;
    }
    setPhase('processing');
    setCustomError(null);
    try {
      setTranscriptionNotice('Retrying speech and emotion analysis...');
      const [sttOutcome, emoOutcome] = await Promise.allSettled([
        transcribeAudio(lastAudioBlob, 'recording.webm', languagePreference === 'auto' ? undefined : languagePreference),
        predictEmotion(lastAudioBlob, 'recording.webm'),
      ]);

      if (sttOutcome.status === 'fulfilled') {
        const emoResult = emoOutcome.status === 'fulfilled' ? emoOutcome.value : null;
        handleTranscriptionAndEmotionResult(sttOutcome.value, emoResult);
      } else {
        setCustomError('Retry failed to transcribe audio.');
        setPhase('error');
      }
    } catch (err: unknown) {
      const errorMsg = err instanceof Error ? err.message : 'Retry failed';
      setCustomError(`Processing error: ${errorMsg}`);
      setPhase('error');
    }
  };

  // Clear all messages
  const clearTranscript = () => {
    setMsgs([]);
    setHasInsights(false);
    setEmotionData(null);
    setTranscriptionNotice('Transcript cleared.');
  };

  const retry = () => {
    clearTimers();
    recorder.clearError();
    setCustomError(null);
    setTranscriptionNotice(null);
    setPhase('idle');
  };

  const replay = () => {
    setProgress(0);
    setPaused(false);
    setPhase('responding');
  };

  const stopPlayback = () => {
    setPaused(false);
    setPhase('idle');
  };

  const endSession = () => {
    clearTimers();
    recorder.cancelRecording();
    window.location.hash = '#/';
  };

  const lastAi = [...msgs].reverse().find((m) => m.role === 'ai')?.id;
  const busy = phase === 'processing' || phase === 'analysing';

  return (
    <div className="flex min-h-screen flex-col">
      {/* Header */}
      <header className="mx-auto flex w-full max-w-6xl items-center justify-between px-5 py-4">
        <div className="flex items-center gap-4">
          <Logo />
          {/* Backend Status indicator */}
          <div
            className="hidden sm:inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium border border-ink/10 cursor-pointer"
            onClick={checkHealth}
            title="Click to re-check backend connection"
          >
            <span
              className={`h-2 w-2 rounded-full ${
                backendStatus === 'online'
                  ? 'bg-emerald-500'
                  : backendStatus === 'checking'
                  ? 'bg-amber-400 animate-pulse'
                  : 'bg-rose-500'
              }`}
            />
            <span className="text-muted text-[11px]">
              {backendStatus === 'online'
                ? 'Live'
                : backendStatus === 'checking'
                ? 'Connecting...'
                : 'Offline'}
            </span>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            className="btn btn-ghost !py-2 text-xs"
            onClick={handleOpenHistory}
            title="View past conversations from MongoDB"
          >
            <Icon name="history" size={14} />
            History
          </button>
          <button
            type="button"
            className="btn btn-ghost !py-2 text-xs"
            onClick={() => setShowUploader((u) => !u)}
            title="Upload audio file for transcription & emotion analysis"
          >
            <Icon name="mic" size={14} />
            {showUploader ? 'Hide Upload' : 'Upload File'}
          </button>
          <ThemeToggle />
          {confirmEnd ? (
            <>
              <button className="btn btn-primary !py-2 text-xs" onClick={endSession}>
                Confirm end
              </button>
              <button className="btn btn-ghost !py-2 text-xs" onClick={() => setConfirmEnd(false)}>
                Keep going
              </button>
            </>
          ) : (
            <button className="btn btn-ghost !py-2 text-xs" onClick={() => setConfirmEnd(true)}>
              End session
            </button>
          )}
        </div>
      </header>

      {/* Notice Banner */}
      {transcriptionNotice && (
        <div className="mx-auto mb-2 w-full max-w-6xl px-5">
          <span className="chip !text-[11px] !bg-lav border border-accent/20">
            {transcriptionNotice}
          </span>
        </div>
      )}

      {/* File Uploader Drawer */}
      <AnimatePresence>
        {showUploader && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mx-auto w-full max-w-6xl px-5 mb-5 overflow-hidden"
          >
            <AudioUploader
              onTranscriptionSuccess={async (sttRes) => {
                // Also infer emotion
                let emoRes: EmotionPredictionResponse | null = null;
                try {
                  if (lastAudioBlob) {
                    emoRes = await predictEmotion(lastAudioBlob);
                  }
                } catch {
                  // Ignore emotion failure
                }
                handleTranscriptionAndEmotionResult(sttRes, emoRes);
              }}
              onCancel={() => setShowUploader(false)}
            />
          </motion.div>
        )}
      </AnimatePresence>

      <main className="mx-auto grid w-full max-w-6xl flex-1 gap-5 px-5 pb-8 lg:grid-cols-[1fr_1.05fr]">
        {/* Voice Stage */}
        <section className="card flex flex-col items-center px-6 py-8" aria-label="Voice stage">
          <Orb mode={orb} level={level} className="w-56 sm:w-64" />

          <p className="mt-8 text-center font-serif text-2xl" role="status" aria-live="polite">
            {STATUS[phase]}
          </p>

          <div className="mt-4 w-full max-w-sm">
            <Waveform bars={bars} active={phase === 'listening' || speaking} />
          </div>

          <p className="mt-1 h-5 text-sm tabular-nums text-muted">
            {(phase === 'listening' || phase === 'paused') && (
              <>
                Recording {fmt(demoMode ? 5 : recorder.duration)}
                {recorder.isMuted && ' · microphone muted'}
                {demoMode && ' · simulated'}
              </>
            )}
          </p>

          <div className="mt-5 flex min-h-[3.5rem] w-full flex-col items-center justify-center gap-3">
            {/* Idle State Controls */}
            {phase === 'idle' && (
              <div className="flex flex-col items-center gap-2">
                <button
                  className="btn btn-primary !px-8 !py-3.5 text-base shadow-sm"
                  onClick={beginListeningReal}
                >
                  <Icon name="mic" />
                  Start talking
                </button>
              </div>
            )}

            {/* Listening / Paused Controls */}
            {(phase === 'listening' || phase === 'paused') && (
              <div className="flex flex-wrap items-center justify-center gap-2">
                {!demoMode && (
                  <>
                    {phase === 'listening' ? (
                      <button
                        className="btn btn-soft"
                        onClick={() => {
                          recorder.pauseRecording();
                          setPhase('paused');
                        }}
                      >
                        <Icon name="pause" size={18} />
                        Pause
                      </button>
                    ) : (
                      <button
                        className="btn btn-soft"
                        onClick={() => {
                          recorder.resumeRecording();
                          setPhase('listening');
                        }}
                      >
                        <Icon name="play" size={18} />
                        Resume
                      </button>
                    )}

                    <button
                      className="btn btn-soft"
                      aria-pressed={recorder.isMuted}
                      onClick={recorder.toggleMute}
                    >
                      <Icon name={recorder.isMuted ? 'micoff' : 'mic'} size={18} />
                      {recorder.isMuted ? 'Unmute' : 'Mute'}
                    </button>
                  </>
                )}

                <button className="btn btn-primary" onClick={stop}>
                  <Icon name="stop" size={18} />
                  Stop & Process
                </button>

                <button
                  className="btn btn-ghost"
                  onClick={() => {
                    recorder.cancelRecording();
                    retry();
                  }}
                >
                  <Icon name="replay" size={18} />
                  Cancel
                </button>
              </div>
            )}

            {busy && (
              <div className="flex items-center gap-2 text-sm text-muted">
                <span className="h-2 w-2 rounded-full bg-accent animate-ping" />
                <span>Running speech recognition & acoustic pattern analysis...</span>
              </div>
            )}

            {phase === 'responding' && mode === 'voice' && (
              <button className="btn btn-ghost" onClick={stopPlayback}>
                <Icon name="stop" size={18} />
                Stop reply
              </button>
            )}

            {phase === 'error' && (
              <div className="max-w-sm rounded-2xl bg-pink p-5 text-center">
                <p className="text-sm leading-relaxed text-ink">
                  {customError || "Speech processing encountered an error. Please verify your connection or microphone and try again."}
                </p>
                <div className="mt-4 flex justify-center gap-2">
                  {lastAudioBlob && (
                    <button className="btn btn-primary text-xs" onClick={retryTranscription}>
                      <Icon name="replay" size={16} />
                      Retry Analysis
                    </button>
                  )}
                  <button className="btn btn-soft text-xs" onClick={retry}>
                    New Recording
                  </button>
                </div>
              </div>
            )}
          </div>

          {phase === 'idle' && (
            <div className="mt-4 flex items-center gap-4 text-xs text-muted">
              <button
                className="underline underline-offset-4 hover:text-ink"
                onClick={beginListeningDemo}
              >
                Quick demo (no mic)
              </button>
              <span>·</span>
              <button
                className="underline underline-offset-4 hover:text-ink"
                onClick={() => {
                  setCustomError('Previewing simulated speech error state.');
                  setPhase('error');
                }}
              >
                Preview error state
              </button>
            </div>
          )}
        </section>

        {/* Conversation Transcript & Controls */}
        <div className="flex flex-col gap-5">
          <section className="card flex min-h-[18rem] flex-1 flex-col p-6" aria-label="Conversation">
            <div className="flex items-center justify-between">
              <h2 className="font-serif text-xl">Conversation</h2>
              {msgs.length > 0 && (
                <button
                  type="button"
                  className="text-xs text-muted hover:text-ink underline underline-offset-2"
                  onClick={clearTranscript}
                >
                  Clear transcript
                </button>
              )}
            </div>

            <div className="mt-4 flex flex-col gap-3 max-h-[420px] overflow-y-auto pr-1 scroll-smooth" aria-live="polite">
              {msgs.length === 0 && (
                <div className="grid flex-1 place-items-center rounded-2xl border border-dashed border-ink/15 p-6 text-center text-sm text-muted">
                  Speak in English, Hindi, or Hinglish. Your words will be transcribed by Whisper and displayed here.
                </div>
              )}

              <AnimatePresence initial={false}>
                {msgs.map((m) => (
                  <motion.article
                    key={m.id}
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    className={`max-w-[92%] rounded-2xl px-4 py-3 text-[15px] leading-relaxed ${
                      m.role === 'user' ? 'self-end bg-lav' : 'self-start bg-peach'
                    }`}
                  >
                    <div className="mb-1 flex items-center justify-between gap-2 text-xs font-medium text-muted">
                      <span>{m.role === 'user' ? 'You' : 'VocalSense'}</span>
                      {m.role === 'user' && m.detectedLanguage && (
                        <span className="chip !text-[10px] !py-0.5 !px-2 uppercase tracking-wider">
                          {m.detectedLanguage} {m.languageProb ? `· ${Math.round(m.languageProb * 100)}%` : ''}
                        </span>
                      )}
                    </div>

                    {m.isError ? (
                      <div className="rounded-xl border border-rose-300 bg-rose-50/80 dark:bg-rose-950/20 p-3 text-xs">
                        <div className="flex items-center gap-1.5 font-medium text-rose-800 dark:text-rose-300">
                          <Icon name="error" size={14} />
                          <span>AI Response Temporarily Unavailable</span>
                        </div>
                        <p className="mt-1 text-ink/80 leading-relaxed">
                          {m.errorMessage || "The conversational AI service couldn't generate a reply. No fake template was used."}
                        </p>
                        {m.lastTranscript && (
                          <div className="mt-2.5">
                            <button
                              className="btn btn-primary !py-1 !px-3 text-xs"
                              onClick={() => retryAiResponse(m.lastTranscript!)}
                              disabled={busy}
                            >
                              <Icon name="replay" size={12} />
                              Retry Response
                            </button>
                          </div>
                        )}
                      </div>
                    ) : editing === m.id ? (
                      <div className="mt-1">
                        <textarea
                          className="w-full rounded-xl border border-ink/20 bg-surface p-2 text-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-accent"
                          rows={3}
                          value={draft}
                          onChange={(e) => setDraft(e.target.value)}
                          aria-label="Edit transcript"
                        />
                        <div className="mt-2 flex gap-2">
                          <button
                            className="btn btn-primary !px-4 !py-1 text-xs"
                            onClick={() => {
                              setMsgs((all) =>
                                all.map((x) =>
                                  x.id === m.id ? { ...x, text: draft, isEdited: true } : x
                                )
                              );
                              setEditing(null);
                            }}
                          >
                            Save correction
                          </button>
                          <button
                            className="btn btn-ghost !px-4 !py-1 text-xs"
                            onClick={() => setEditing(null)}
                          >
                            Cancel
                          </button>
                        </div>
                      </div>
                    ) : (
                      <>
                        <p>{m.text}</p>
                        {m.role === 'user' && (
                          <div className="mt-2 flex items-center gap-3 text-xs text-muted">
                            <button
                              className="inline-flex items-center gap-1 hover:text-ink"
                              onClick={() => {
                                setEditing(m.id);
                                setDraft(m.text);
                              }}
                            >
                              <Icon name="edit" size={13} />
                              Edit transcript
                            </button>
                            {m.isEdited && <span className="text-[11px] italic">(edited)</span>}
                          </div>
                        )}
                        {m.role === 'ai' && m.id === lastAi && mode === 'voice' && (
                          <div className="mt-3 flex items-center gap-2">
                            {phase === 'responding' ? (
                              <button
                                className="btn btn-soft !p-2"
                                aria-label={paused ? 'Resume playback' : 'Pause playback'}
                                onClick={() => setPaused((p) => !p)}
                              >
                                <Icon name={paused ? 'play' : 'pause'} size={16} />
                              </button>
                            ) : (
                              <button
                                className="btn btn-soft !p-2"
                                aria-label="Replay response"
                                onClick={replay}
                              >
                                <Icon name="replay" size={16} />
                              </button>
                            )}
                            <button
                              className="btn btn-soft !p-2"
                              aria-label={spkMuted ? 'Unmute playback' : 'Mute playback'}
                              aria-pressed={spkMuted}
                              onClick={() => setSpkMuted((s) => !s)}
                            >
                              <Icon name={spkMuted ? 'speakeroff' : 'speaker'} size={16} />
                            </button>
                            <div
                              className="h-1.5 flex-1 overflow-hidden rounded-full bg-ink/10"
                              role="progressbar"
                              aria-valuenow={progress}
                              aria-valuemin={0}
                              aria-valuemax={100}
                              aria-label="Playback progress"
                            >
                              <div
                                className="h-full rounded-full bg-accent transition-all"
                                style={{ width: `${progress}%` }}
                              />
                            </div>
                          </div>
                        )}
                      </>
                    )}
                  </motion.article>
                ))}
              </AnimatePresence>
              <div ref={messagesEndRef} />
            </div>
          </section>

          {/* Settings & Controls */}
          <section className="card p-6" aria-label="Settings">
            <h2 className="text-sm font-semibold">Conversation style</h2>
            <div
              role="radiogroup"
              aria-label="Conversation style"
              className="mt-3 grid gap-2 sm:grid-cols-2"
            >
              {STYLES.map((s) => (
                <button
                  key={s.id}
                  role="radio"
                  aria-checked={style === s.id}
                  onClick={() => setStyle(s.id)}
                  className={`rounded-2xl border p-3 text-left transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent ${
                    style === s.id
                      ? 'border-accent bg-lav'
                      : 'border-ink/10 hover:bg-ink/5'
                  }`}
                >
                  <span className="block text-sm font-semibold">{s.name}</span>
                  <span className="block text-xs text-muted">{s.hint}</span>
                </button>
              ))}
            </div>

            <div className="mt-5 flex flex-wrap items-center justify-between gap-3">
              <div>
                <h2 className="text-sm font-semibold">Response mode</h2>
                <div
                  className="mt-2 inline-flex rounded-full bg-ink/5 p-1"
                  role="group"
                  aria-label="Response mode"
                >
                  {(['voice', 'text'] as const).map((m) => (
                    <button
                      key={m}
                      aria-pressed={mode === m}
                      onClick={() => setMode(m)}
                      className={`rounded-full px-4 py-1.5 text-sm transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent ${
                        mode === m ? 'bg-accent text-onaccent' : 'text-ink'
                      }`}
                    >
                      {m === 'voice' ? 'Voice' : 'Text'}
                    </button>
                  ))}
                </div>
              </div>

              <label className="flex cursor-pointer items-center gap-3 text-sm">
                <span>Speech insights</span>
                <span className="text-[11px] text-muted">· latest turn</span>
                <input
                  type="checkbox"
                  role="switch"
                  className="h-5 w-9 cursor-pointer accent-[rgb(var(--accent))]"
                  checked={showInsights}
                  onChange={(e) => setShowInsights(e.target.checked)}
                />
              </label>
            </div>

            {/* Speech Insights Panel (Actual Acoustic Measurements + Emotion Patterns) */}
            {showInsights && (
              <div className="mt-5 border-t border-ink/10 pt-5">
                {!hasInsights || !emotionData ? (
                  <p className="text-sm text-muted">Record or upload speech to analyze acoustic features and vocal patterns.</p>
                ) : (
                  <>
                    <p className="mb-3 text-xs text-muted">
                      Measurable physical acoustic traits from your recording. Values reflect acoustic measurements, not psychological diagnoses.
                    </p>
                    <dl className="grid gap-2 sm:grid-cols-2">
                      <div className="rounded-2xl p-3 bg-lav">
                        <dt className="text-xs font-medium text-ink/70">Pitch Variation (F0)</dt>
                        <dd className="text-sm font-medium">
                          {emotionData.vocal_traits.pitch_mean_hz > 0
                            ? `${emotionData.vocal_traits.pitch_mean_hz} Hz (±${emotionData.vocal_traits.pitch_std_hz} Hz)`
                            : 'Unvoiced / Whisper'}
                        </dd>
                      </div>

                      <div className="rounded-2xl p-3 bg-pink">
                        <dt className="text-xs font-medium text-ink/70">Vocal Energy (RMS)</dt>
                        <dd className="text-sm font-medium">
                          RMS {emotionData.vocal_traits.rms_energy} · {emotionData.vocal_traits.rms_energy > 0.08 ? 'Higher' : emotionData.vocal_traits.rms_energy < 0.03 ? 'Soft' : 'Moderate'}
                        </dd>
                      </div>

                      <div className="rounded-2xl p-3 bg-sage">
                        <dt className="text-xs font-medium text-ink/70">Pauses & Cadence</dt>
                        <dd className="text-sm font-medium">
                          {emotionData.vocal_traits.duration_sec}s audio · {Math.round(emotionData.vocal_traits.pause_fraction * 100)}% silence
                        </dd>
                      </div>

                      <div className="rounded-2xl p-3 bg-peach">
                        <dt className="text-xs font-medium text-ink/70">Spectral Centroid</dt>
                        <dd className="text-sm font-medium">
                          {Math.round(emotionData.vocal_traits.spectral_centroid_hz)} Hz frequency center
                        </dd>
                      </div>
                    </dl>

                    <div className="mt-3 rounded-2xl border border-ink/10 p-3.5 bg-surface text-sm">
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-xs text-muted">Model: {emotionData.model_used}</span>
                        {emotionData.is_reliable && (
                          <span className="chip !text-[10px] !py-0.5 !px-2 uppercase">
                            {Math.round(emotionData.confidence * 100)}% confidence
                          </span>
                        )}
                      </div>
                      <p className="mt-1">
                        <span className="font-semibold">Vocal Pattern Estimate: </span>
                        {emotionData.is_reliable ? (
                          <span className="capitalize font-medium text-accent">
                            {emotionData.emotion_label}-like acoustic pattern
                          </span>
                        ) : (
                          <span className="italic text-muted">
                            Unable to estimate vocal emotion reliably.
                          </span>
                        )}
                      </p>
                    </div>

                    <p className="mt-2 text-xs text-muted leading-relaxed">
                      {emotionData.interpretation_notice || "These are measurable acoustic signals in your voice, not statements about your actual feelings. Your own words always come first."}
                    </p>
                  </>
                )}
              </div>
            )}
          </section>
        </div>
      </main>

      {/* Privacy & Safety Principles */}
      <section className="mx-auto w-full max-w-6xl px-5 pb-10 pt-2" aria-label="Privacy principles">
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {[
            {
              icon: (
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.6} className="h-5 w-5">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 5.25a3 3 0 013 3m3 0a6 6 0 01-7.029 5.912c-.563-.097-1.159.026-1.563.43L10.5 17.25H8.25v2.25H6v2.25H2.25v-2.818c0-.597.237-1.17.659-1.591l6.499-6.499c.404-.404.527-1 .43-1.563A6 6 0 1121.75 8.25z" />
                </svg>
              ),
              color: 'bg-lav',
              title: 'You stay in control',
              body: 'Recording begins only after you press Start and explicitly allow microphone access.',
            },
            {
              icon: (
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.6} className="h-5 w-5">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75m-3-7.036A11.959 11.959 0 013.598 6 11.955 11.955 0 003 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285z" />
                </svg>
              ),
              color: 'bg-sage',
              title: 'Zero permanent audio',
              body: 'Audio is processed ephemerally in-memory. No raw recordings are retained by default.',
            },
            {
              icon: (
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.6} className="h-5 w-5">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z" />
                </svg>
              ),
              color: 'bg-peach',
              title: 'Strictly opt-in memory',
              body: 'Conversation history and acoustic memory are independent features—both off unless you choose otherwise.',
            },
            {
              icon: (
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.6} className="h-5 w-5">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M21 8.25c0-2.485-2.099-4.5-4.688-4.5-1.935 0-3.597 1.126-4.312 2.733-.715-1.607-2.377-2.733-4.313-2.733C5.1 3.75 3 5.765 3 8.25c0 7.22 9 12 9 12s9-4.78 9-12z" />
                </svg>
              ),
              color: 'bg-pink',
              title: 'Companion, not clinician',
              body: 'VocalSense listens and reflects. It never diagnoses, assesses, or provides medical or therapeutic advice.',
            },
          ].map((item) => (
            <div
              key={item.title}
              className={`${item.color} rounded-2xl p-4 flex flex-col gap-2 border border-ink/5`}
            >
              <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-white/40 text-ink/70">
                {item.icon}
              </div>
              <p className="text-sm font-semibold leading-snug">{item.title}</p>
              <p className="text-xs text-muted leading-relaxed">{item.body}</p>
            </div>
          ))}
        </div>
      </section>

      {/* History Drawer Modal */}
      <AnimatePresence>
        {historyOpen && (
          <div className="fixed inset-0 z-50 flex justify-end bg-black/40 backdrop-blur-sm">
            <motion.div
              initial={{ x: '100%' }}
              animate={{ x: 0 }}
              exit={{ x: '100%' }}
              transition={{ type: 'spring', damping: 25, stiffness: 200 }}
              className="flex h-full w-full max-w-md flex-col bg-surface p-6 shadow-2xl"
            >
              <div className="flex items-center justify-between border-b border-ink/10 pb-4">
                <div className="flex items-center gap-2">
                  <Icon name="history" size={18} />
                  <h2 className="font-serif text-lg">Conversation History</h2>
                </div>
                <button
                  className="btn btn-ghost !p-2"
                  onClick={() => setHistoryOpen(false)}
                  aria-label="Close history"
                >
                  ✕
                </button>
              </div>

              <div className="flex-1 overflow-y-auto py-4">
                {loadingHistory ? (
                  <div className="flex h-32 items-center justify-center text-sm text-muted">
                    Loading conversations from MongoDB...
                  </div>
                ) : historyList.length === 0 ? (
                  <div className="flex h-32 flex-col items-center justify-center text-center text-sm text-muted">
                    <p>No past conversations found.</p>
                    <p className="text-xs text-muted/70 mt-1">Start speaking to create one.</p>
                  </div>
                ) : (
                  <div className="flex flex-col gap-2.5">
                    {historyList.map((item) => (
                      <div
                        key={item.conversation_id}
                        onClick={() => handleSelectConversation(item.conversation_id)}
                        className={`group flex cursor-pointer flex-col rounded-2xl border p-3.5 transition ${
                          activeConversationId === item.conversation_id
                            ? 'border-accent bg-lav/50'
                            : 'border-ink/10 hover:bg-ink/5'
                        }`}
                      >
                        <div className="flex items-center justify-between text-xs text-muted">
                          <span className="font-mono text-[11px]">
                            {new Date(item.updated_at).toLocaleDateString([], {
                              month: 'short',
                              day: 'numeric',
                              hour: '2-digit',
                              minute: '2-digit',
                            })}
                          </span>
                          <div className="flex items-center gap-2">
                            {item.language && (
                              <span className="chip !text-[10px] !py-0.2 !px-1.5 uppercase">
                                {item.language}
                              </span>
                            )}
                            <button
                              className="opacity-0 group-hover:opacity-100 p-1 hover:text-rose-500 transition"
                              onClick={(e) => handleDeleteConversation(item.conversation_id, e)}
                              title="Delete conversation"
                            >
                              <Icon name="trash" size={14} />
                            </button>
                          </div>
                        </div>
                        <h4 className="mt-1.5 font-semibold text-xs text-ink line-clamp-1">
                          {item.title || item.last_message_preview || `Reflection · ${(item.style || 'Voice').replace('_', ' ')}`}
                        </h4>
                        {item.last_message_preview && item.last_message_preview !== item.title && (
                          <p className="mt-0.5 text-[11px] text-muted line-clamp-2">
                            {item.last_message_preview}
                          </p>
                        )}
                        <span className="mt-1 block text-[10px] text-muted">
                          {item.message_count} {item.message_count === 1 ? 'message' : 'messages'} · Style: {(item.style || 'just_listen').replace('_', ' ')}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* GDPR User Data Purge Section */}
              <div className="border-t border-ink/10 pt-4">
                {purgeConfirmOpen ? (
                  <div className="rounded-2xl bg-rose-50/80 dark:bg-rose-950/20 border border-rose-300 p-3 text-xs">
                    <p className="font-semibold text-rose-800 dark:text-rose-300">Purge all user data?</p>
                    <p className="mt-1 text-ink/80 text-[11px] leading-relaxed">
                      This will permanently erase all conversations, messages, and vocal acoustic traits from MongoDB (GDPR/DPDP compliance).
                    </p>
                    <div className="mt-3 flex gap-2">
                      <button
                        className="btn btn-primary !py-1 !px-3 text-xs !bg-rose-600 hover:!bg-rose-700 text-white"
                        onClick={handlePurgeAllData}
                      >
                        Yes, purge everything
                      </button>
                      <button
                        className="btn btn-ghost !py-1 !px-3 text-xs"
                        onClick={() => setPurgeConfirmOpen(false)}
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                ) : (
                  <button
                    className="flex w-full items-center justify-center gap-2 rounded-xl py-2 text-xs text-rose-600 hover:bg-rose-50/50 transition dark:text-rose-400"
                    onClick={() => setPurgeConfirmOpen(true)}
                  >
                    <Icon name="trash" size={14} />
                    Purge All My Data (GDPR / DPDP)
                  </button>
                )}
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
}

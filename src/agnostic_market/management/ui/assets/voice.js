// Voice for the conversation simulator, over the merchant's own configured engines.
//
// Capture and playback are proxied through the management API rather than the browser's built-in
// speech, so the workbench exercises the merchant's pinned TTS voice and the STT model and
// keyterm bias their callers meet. The browser cannot hold a provider key, which is why the
// audio round-trips through loopback.
//
// This is not the voice pipeline. Turn admission, production VAD, acoustic barge-in, AI
// disclosure and the durable session boundary belong to the LiveKit worker.

export const CAPTURE_SAMPLE_RATE = 16000;

// Bounds an accidentally open microphone. The server enforces its own ceiling; this one keeps
// the page from buffering megabytes before it ever tries.
const MAX_CAPTURE_SECONDS = 30;
// Preview-only acoustic pause detection. This is not the worker's VAD: it only segments a
// consented browser capture, and the manual send control remains available in noisy rooms.
const VOICE_RMS_FLOOR = 0.018;
const VOICE_CONFIRM_SECONDS = 0.12;
// Browser echo cancellation is imperfect. Require a stronger, sustained signal while the
// speaker is playing; this is a preview heuristic, not speaker identification or worker VAD.
const BARGE_RMS_FLOOR = 0.05;
const BARGE_CONFIRM_SECONDS = 0.24;
const END_PAUSE_SECONDS = 0.85;
const PRE_ROLL_SECONDS = 0.35;

function captureRms(samples) {
  let energy = 0;
  for (const sample of samples) energy += sample * sample;
  return Math.sqrt(energy / samples.length);
}

export const captureSupport = Object.freeze({
  // getUserMedia needs a secure context; loopback counts as one.
  input:
    typeof globalThis.AudioWorkletNode !== "undefined" &&
    Boolean(globalThis.navigator?.mediaDevices?.getUserMedia),
  output: typeof globalThis.AudioContext !== "undefined",
});

// One playback context for the page. A reply is spoken seconds after the click that asked for
// it, by which time an <audio> element's transient autoplay permission has lapsed, so playback
// goes through a context unlocked on a real gesture and kept for the life of the page.
let playbackContext = null;

function playbackReady() {
  playbackContext ??= new globalThis.AudioContext();
  return playbackContext;
}

export function unlockPlayback() {
  if (!captureSupport.output) return;
  const context = playbackReady();
  if (context.state === "suspended") void context.resume();
}

if (typeof globalThis.document !== "undefined" && captureSupport.output) {
  // Left armed rather than once: a browser may suspend the context again when the tab is
  // backgrounded, and the next gesture should quietly restore it.
  for (const gesture of ["pointerdown", "keydown"]) {
    globalThis.document.addEventListener(gesture, unlockPlayback, { passive: true });
  }
}

export function supportSummary(support = captureSupport) {
  if (support.input && support.output) return "Microphone and playback available.";
  if (support.output) return "Playback only. This browser cannot capture audio.";
  if (support.input) return "Capture only. This browser cannot play audio.";
  return "This browser can neither capture nor play audio.";
}

export function describeEngines(identity) {
  if (!identity) return "";
  return (
    `Speaking with ${identity.tts_provider} ${identity.tts_model}, ` +
    `hearing with ${identity.stt_provider} ${identity.stt_model}.`
  );
}

/** Linear resample to the rate the server expects. A browser may ignore a requested rate. */
export function resampleTo(samples, fromRate, toRate = CAPTURE_SAMPLE_RATE) {
  if (fromRate === toRate || !samples.length) return samples;
  const ratio = fromRate / toRate;
  const length = Math.floor(samples.length / ratio);
  const output = new Float32Array(length);
  for (let index = 0; index < length; index += 1) {
    const position = index * ratio;
    const left = Math.floor(position);
    const right = Math.min(left + 1, samples.length - 1);
    output[index] = samples[left] + (samples[right] - samples[left]) * (position - left);
  }
  return output;
}

/** Float32 [-1, 1] to signed 16-bit little-endian, which is what the STT engine accepts. */
export function toPcm16(samples) {
  const buffer = new ArrayBuffer(samples.length * 2);
  const view = new DataView(buffer);
  for (let index = 0; index < samples.length; index += 1) {
    const clamped = Math.max(-1, Math.min(1, samples[index]));
    view.setInt16(index * 2, Math.round(clamped * 0x7fff), true);
  }
  return buffer;
}

export function concatSamples(blocks) {
  const total = blocks.reduce((sum, block) => sum + block.length, 0);
  const merged = new Float32Array(total);
  let offset = 0;
  for (const block of blocks) {
    merged.set(block, offset);
    offset += block.length;
  }
  return merged;
}

/** One controller owning capture, playback, and the state the orb renders. */
export function createVoiceController({
  api,
  tenantId,
  versionId,
  onState,
  onTranscript,
  onError,
  onLevel,
  onBargeIn,
  onDiagnostic,
} = {}) {
  const nowMs = () => globalThis.performance?.now?.() ?? Date.now();
  const diagnostic = (phase, fields = {}) => {
    if (!onDiagnostic) return;
    try {
      onDiagnostic({ phase, at_ms: Math.round(nowMs()), ...fields });
    } catch {
      // Optional local diagnostics must not interrupt a caller turn.
    }
  };
  const emitState = (next) => {
    if (controller.state === next) return;
    controller.state = next;
    diagnostic("state", { state: next });
    onState?.(next);
  };

  let context = null;
  let stream = null;
  let node = null;
  let blocks = [];
  let automaticCapture = false;
  let speechDetected = false;
  let captureCompletionPending = false;
  let monitoringPlayback = false;
  let promoteMonitor = null;
  let source = null;
  // Only the latest reply has an outcome: natural end, caller interruption, or unknown.
  let playback = null;
  let synthesisPending = false;
  let playbackAnalyser = null;
  let playbackFrame = null;
  let acknowledgeTimer = null;
  let lastLevelAt = -Infinity;
  // Monotonic tokens. getUserMedia, addModule, synthesis and decode are all awaits during
  // which the session can be reset or superseded; whatever resolves late compares its token
  // and cleans up after itself instead of attaching to a session that has moved on.
  let captureToken = 0;
  let playbackToken = 0;

  function stopAcknowledgeTimer() {
    if (acknowledgeTimer === null) return;
    clearTimeout(acknowledgeTimer);
    acknowledgeTimer = null;
  }

  function stopTracks(candidate) {
    for (const track of candidate?.getTracks() ?? []) track.stop();
  }

  function emitLevel(rms) {
    if (!onLevel) return;
    if (rms === 0) {
      lastLevelAt = -Infinity;
      onLevel(0);
      return;
    }
    const now = globalThis.performance?.now?.() ?? Date.now();
    if (now - lastLevelAt < 32) return;
    lastLevelAt = now;
    onLevel(Math.min(1, rms * 8));
  }

  function stopPlaybackMeter() {
    if (playbackFrame !== null) globalThis.cancelAnimationFrame?.(playbackFrame);
    playbackFrame = null;
    playbackAnalyser?.disconnect();
    playbackAnalyser = null;
    emitLevel(0);
  }

  async function teardownCapture() {
    node?.disconnect();
    node = null;
    stopTracks(stream);
    stream = null;
    const closingContext = context;
    context = null;
    emitLevel(0);
    if (closingContext) await closingContext.close().catch(() => {});
  }

  /** End capture without transcribing. Used whenever a turn leaves the listening state. */
  async function discardCapture() {
    captureToken += 1;
    captureCompletionPending = false;
    monitoringPlayback = false;
    promoteMonitor = null;
    blocks = [];
    await teardownCapture();
  }

  const controller = {
    state: "idle",
    inputBlocked: false,
    acquiring: false,
    get playbackActive() {
      return synthesisPending || source !== null;
    },
    playbackOutcome(replyId) {
      if (!playback || replyId == null) return null;
      return playback.replyId === replyId ? playback.outcome : null;
    },

    async listen({ autoSendOnSilence = false, monitorPlayback = false } = {}) {
      if (!captureSupport.input || controller.inputBlocked) return false;
      // The state is still idle while permission is pending, so a second click would open a
      // second stream and only the last one would ever be stopped. The token is claimed before
      // the first await, and anything acquired under a stale token is released here.
      if (controller.state === "listening" || controller.state === "barge-ready" ||
          controller.acquiring) return false;
      monitoringPlayback = monitorPlayback && autoSendOnSilence && controller.playbackActive;
      if (!monitoringPlayback) controller.stopSpeaking(true);
      stopAcknowledgeTimer();
      controller.acquiring = true;
      emitState("arming");
      const mine = ++captureToken;
      blocks = [];
      captureCompletionPending = false;
      automaticCapture = autoSendOnSilence;
      speechDetected = false;
      let capturedSamples = 0;
      let voicedSamples = 0;
      let quietSamples = 0;
      let speechStarted = false;
      let acquired = null;
      let firstFrame = true;
      let levelWindowSamples = 0;
      let levelWindowPeak = 0;
      let levelWindowAboveFloor = 0;
      let levelWindowLongestRun = 0;
      const requestedAt = nowMs();
      diagnostic("capture_request", {
        mode: monitoringPlayback ? "barge" : autoSendOnSilence ? "auto" : "manual",
      });
      promoteMonitor = () => {
        blocks = [];
        capturedSamples = 0;
        voicedSamples = 0;
        quietSamples = 0;
        speechStarted = false;
        speechDetected = false;
        monitoringPlayback = false;
        emitState("listening");
      };
      try {
        acquired = await globalThis.navigator.mediaDevices.getUserMedia({
          audio: { echoCancellation: true, noiseSuppression: true },
        });
        if (mine !== captureToken) {
          stopTracks(acquired);
          return false;
        }
        stream = acquired;
        context = new globalThis.AudioContext({ sampleRate: CAPTURE_SAMPLE_RATE });
        await context.audioWorklet.addModule("/admin/assets/capture-worklet.js");
        if (mine !== captureToken) {
          await teardownCapture();
          return false;
        }
        node = new globalThis.AudioWorkletNode(context, "capture-processor");
        node.port.onmessage = (event) => {
          if (mine !== captureToken ||
              (controller.state !== "listening" && controller.state !== "barge-ready")) return;
          const block = event.data;
          if (!block?.length) return;
          const rate = context?.sampleRate ?? CAPTURE_SAMPLE_RATE;
          if (firstFrame) {
            firstFrame = false;
            diagnostic("capture_first_frame", {
              since_request_ms: Math.round(nowMs() - requestedAt),
            });
          }
          const level = captureRms(block);
          if (controller.state === "barge-ready") {
            blocks.push(block);
            capturedSamples += block.length;
            while (capturedSamples > rate * PRE_ROLL_SECONDS && blocks.length > 1) {
              capturedSamples -= blocks.shift().length;
            }
            voicedSamples = level >= BARGE_RMS_FLOOR ? voicedSamples + block.length : 0;
            if (onDiagnostic) {
              levelWindowSamples += block.length;
              levelWindowPeak = Math.max(levelWindowPeak, level);
              if (level >= BARGE_RMS_FLOOR) levelWindowAboveFloor += block.length;
              levelWindowLongestRun = Math.max(levelWindowLongestRun, voicedSamples);
              if (levelWindowSamples >= rate / 4) {
                diagnostic("barge_level", {
                  peak_rms: Number(levelWindowPeak.toFixed(3)),
                  above_floor_ms: Math.round(levelWindowAboveFloor * 1000 / rate),
                  longest_run_ms: Math.round(levelWindowLongestRun * 1000 / rate),
                });
                levelWindowSamples = 0;
                levelWindowPeak = 0;
                levelWindowAboveFloor = 0;
                levelWindowLongestRun = 0;
              }
            }
            if (voicedSamples >= rate * BARGE_CONFIRM_SECONDS) {
              speechStarted = true;
              speechDetected = true;
              monitoringPlayback = false;
              diagnostic("barge_trigger", {
                rms: Number(level.toFixed(3)),
                sustained_ms: Math.round(voicedSamples * 1000 / rate),
              });
              onBargeIn?.();
              controller.stopSpeaking(true);
              emitState("listening");
            }
            return;
          }
          emitLevel(level);
          const voiced = autoSendOnSilence && level >= VOICE_RMS_FLOOR;
          blocks.push(block);
          capturedSamples += block.length;
          if (autoSendOnSilence && !speechStarted) {
            voicedSamples = voiced ? voicedSamples + block.length : 0;
            while (capturedSamples > rate * PRE_ROLL_SECONDS && blocks.length > 1) {
              capturedSamples -= blocks.shift().length;
            }
            if (voicedSamples >= rate * VOICE_CONFIRM_SECONDS) {
              speechStarted = true;
              speechDetected = true;
              diagnostic("capture_speech_detected", {
                sustained_ms: Math.round(voicedSamples * 1000 / rate),
              });
            }
            return;
          }
          quietSamples = voiced ? 0 : quietSamples + block.length;
          if (autoSendOnSilence && quietSamples >= rate * END_PAUSE_SECONDS) {
            void controller.stopListening();
            return;
          }
          // Bound the utterance, while allowing hands-free mode to wait for speech without
          // accumulating silence indefinitely.
          if (capturedSamples >= rate * MAX_CAPTURE_SECONDS) {
            onError?.(`Capture stopped at the ${MAX_CAPTURE_SECONDS} second limit.`, "capture");
            void controller.stopListening();
          }
        };
        context.createMediaStreamSource(stream).connect(node);
        let settings = {};
        try {
          const track = acquired.getAudioTracks?.()[0] ?? acquired.getTracks?.()[0];
          settings = track?.getSettings?.() ?? {};
        } catch {
          // Browser settings are diagnostic data, not a capture prerequisite.
        }
        const echoCancellation = settings.echoCancellation;
        diagnostic("capture_ready", {
          since_request_ms: Math.round(nowMs() - requestedAt),
          context_sample_rate: context.sampleRate,
          track_sample_rate: Number.isFinite(settings.sampleRate) ? settings.sampleRate : null,
          echo_cancellation: typeof echoCancellation === "boolean" ||
            echoCancellation === "all" || echoCancellation === "remote-only"
            ? echoCancellation : null,
          noise_suppression: typeof settings.noiseSuppression === "boolean"
            ? settings.noiseSuppression : null,
        });
      } catch (error) {
        stopTracks(acquired);
        await teardownCapture();
        monitoringPlayback = false;
        promoteMonitor = null;
        // A refused permission never recovers within the page; stop offering the control.
        controller.inputBlocked = error?.name === "NotAllowedError";
        diagnostic("capture_failed", {
          reason: controller.inputBlocked ? "permission_denied" : "capture_error",
        });
        onError?.(
          controller.inputBlocked
            ? "Microphone permission was refused. Allow it for this site, then reload."
            : `Microphone capture failed (${error?.name ?? "unknown"}).`,
          "capture",
        );
        emitState(controller.playbackActive ? "speaking" : "idle");
        return false;
      } finally {
        controller.acquiring = false;
      }
      stopAcknowledgeTimer();
      if (!controller.playbackActive) monitoringPlayback = false;
      emitState(monitoringPlayback && controller.playbackActive ? "barge-ready" : "listening");
      return true;
    },

    /** Stop capture and hand the audio to the merchant's STT engine. */
    async stopListening() {
      if (controller.state !== "listening") return "";
      const captured = blocks;
      const rate = context?.sampleRate ?? CAPTURE_SAMPLE_RATE;
      const hadSpeech = speechDetected;
      const automatic = automaticCapture;
      const mine = ++captureToken;
      blocks = [];
      captureCompletionPending = true;
      diagnostic("capture_stop", { had_speech: hadSpeech, automatic });
      emitState("thinking");
      await teardownCapture();
      if (mine !== captureToken) return "";
      const samples = resampleTo(concatSamples(captured), rate);
      if (!samples.length || (automatic && !hadSpeech)) {
        captureCompletionPending = false;
        diagnostic("capture_empty");
        emitState("idle");
        return "";
      }
      const transcriptionStartedAt = nowMs();
      diagnostic("transcription_start", { sample_count: samples.length });
      try {
        const result = await api.transcribeCapture(tenantId(), versionId(), toPcm16(samples));
        // Transcription takes seconds. A close or reset during that window must win, or an
        // utterance spoken into the old session is submitted as the new session's first turn.
        if (mine !== captureToken) return "";
        captureCompletionPending = false;
        const heard = String(result?.text ?? "").trim();
        diagnostic("transcription_result", {
          elapsed_ms: Math.round(nowMs() - transcriptionStartedAt),
          has_text: Boolean(heard),
        });
        if (heard) onTranscript?.(heard);
        else emitState("idle");
        return heard;
      } catch {
        if (mine !== captureToken) return "";
        captureCompletionPending = false;
        diagnostic("transcription_failed", {
          elapsed_ms: Math.round(nowMs() - transcriptionStartedAt),
        });
        onError?.("Transcription failed. The turn can still be typed.", "transcription");
        emitState("idle");
        return "";
      }
    },

    /** The turn is in flight; the orb shows work rather than pretending to listen.
     *
     * A typed turn can arrive while the microphone is open. Without tearing capture down here
     * the state leaves "listening", stopListening then refuses to run, and the track stays
     * live for the rest of the session.
     */
    think() {
      controller.stopSpeaking(true);
      stopAcknowledgeTimer();
      emitState("thinking");
      // Returned for callers that want to await teardown; the visible state is already correct.
      return discardCapture();
    },

    async cancelListening() {
      if (controller.state !== "listening" && controller.state !== "barge-ready" &&
          !controller.acquiring && !captureCompletionPending) return;
      captureCompletionPending = false;
      emitState(controller.playbackActive ? "speaking" : "idle");
      await discardCapture();
    },

    async speak(text, replyId = null) {
      const spoken = String(text ?? "").trim();
      controller.stopSpeaking();
      playback = replyId === null ? null : { replyId, outcome: null };
      if (!captureSupport.output || !spoken) {
        controller.acknowledge();
        return false;
      }
      const mine = ++playbackToken;
      synthesisPending = true;
      const synthesisStartedAt = nowMs();
      diagnostic("synthesis_start");
      let blob = null;
      try {
        blob = await api.synthesizeSpeech(tenantId(), versionId(), spoken);
      } catch {
        if (mine !== playbackToken) return false;
        synthesisPending = false;
        diagnostic("synthesis_failed");
        onError?.("Speech synthesis failed. The reply is still shown in the transcript.", "playback");
        controller.acknowledge();
        return false;
      }
      // Synthesis takes seconds. A reset or a newer reply during that window must win, or the
      // stale one starts speaking over a session that has already moved on.
      if (mine !== playbackToken) return false;
      diagnostic("synthesis_complete", {
        elapsed_ms: Math.round(nowMs() - synthesisStartedAt),
      });
      const context = playbackReady();
      if (context.state === "suspended") await context.resume().catch(() => {});
      try {
        const decoded = await context.decodeAudioData(await blob.arrayBuffer());
        if (mine !== playbackToken) return false;
        source = context.createBufferSource();
        source.buffer = decoded;
        if (typeof context.createAnalyser === "function" &&
            typeof globalThis.requestAnimationFrame === "function") {
          playbackAnalyser = context.createAnalyser();
          playbackAnalyser.fftSize = 512;
          source.connect(playbackAnalyser);
          playbackAnalyser.connect(context.destination);
        } else {
          source.connect(context.destination);
        }
        source.onended = () => {
          // A stopped or superseded source still fires this. Without the token check it would
          // clear the shared reference and leave newer audio unstoppable.
          if (mine !== playbackToken) return;
          source = null;
          if (playback) playback.outcome = false;
          diagnostic("playback_end", { monitor_active: monitoringPlayback });
          stopPlaybackMeter();
          if (monitoringPlayback && controller.state === "barge-ready") {
            promoteMonitor?.();
            return;
          }
          if (monitoringPlayback && controller.acquiring) return;
          controller.acknowledge();
        };
        emitState("speaking");
        source.start();
        diagnostic("playback_start", {
          since_synthesis_ms: Math.round(nowMs() - synthesisStartedAt),
        });
        synthesisPending = false;
        if (playbackAnalyser) {
          const samples = new Uint8Array(playbackAnalyser.fftSize);
          const meter = () => {
            if (mine !== playbackToken || !playbackAnalyser) return;
            playbackAnalyser.getByteTimeDomainData(samples);
            let energy = 0;
            for (const sample of samples) {
              const centered = (sample - 128) / 128;
              energy += centered * centered;
            }
            emitLevel(Math.sqrt(energy / samples.length));
            playbackFrame = globalThis.requestAnimationFrame(meter);
          };
          playbackFrame = globalThis.requestAnimationFrame(meter);
        }
      } catch {
        if (mine !== playbackToken) return false;
        synthesisPending = false;
        source = null;
        diagnostic("playback_failed");
        stopPlaybackMeter();
        onError?.("Playback failed. The reply is still shown in the transcript.", "playback");
        controller.acknowledge();
        return false;
      }
      return true;
    },

    stopSpeaking(interruptedByCaller = false) {
      // Claiming the token invalidates synthesis still in flight and retires any onended
      // belonging to the playback being stopped.
      if (source && playback) playback.outcome = interruptedByCaller ? true : null;
      if (synthesisPending || source) {
        diagnostic("playback_cancel", { during_synthesis: synthesisPending });
      }
      playbackToken += 1;
      synthesisPending = false;
      stopPlaybackMeter();
      if (source) {
        source.stop();
        source = null;
      }
      if (controller.state === "speaking") emitState("idle");
      else if (controller.state === "barge-ready") emitState("listening");
    },

    /** A brief settle before idle, so a finished reply reads as received rather than dropped. */
    acknowledge() {
      stopAcknowledgeTimer();
      emitState("acknowledged");
      acknowledgeTimer = setTimeout(() => {
        acknowledgeTimer = null;
        emitState("idle");
      }, 900);
    },

    // A failed turn keeps the finished reply's outcome; a new or reset session discards it.
    async reset({ keepPlaybackOutcome = false } = {}) {
      stopAcknowledgeTimer();
      controller.stopSpeaking();
      if (!keepPlaybackOutcome) playback = null;
      await discardCapture();
      emitState("idle");
    },
  };

  return controller;
}

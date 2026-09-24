// Browser speech for the conversation simulator. No dependencies and no build step, matching
// the rest of this client. Both halves are optional: the panel degrades to text-only when a
// browser lacks either API, and never blocks a turn that the text path could still send.

const RECOGNITION_CTOR =
  globalThis.SpeechRecognition ?? globalThis.webkitSpeechRecognition ?? null;

export const voiceSupport = Object.freeze({
  // Chromium ships recognition behind a prefix; Firefox ships neither today.
  input: Boolean(RECOGNITION_CTOR),
  output: typeof globalThis.speechSynthesis !== "undefined",
});

export function supportSummary(support = voiceSupport) {
  if (support.input && support.output) return "Microphone and playback available.";
  if (support.output) return "Playback only. This browser has no speech recognition.";
  if (support.input) return "Microphone only. This browser has no speech playback.";
  return "This browser supports neither speech recognition nor playback.";
}

// A present constructor is not a working backend. Chromium forks that strip the Google speech
// service still expose webkitSpeechRecognition, so the only honest signal is the first failure.
const FATAL_RECOGNITION_ERRORS = new Map([
  ["network", "Speech recognition is unavailable: this browser blocks the speech service. Brave and some Chromium builds strip it. Type the turn instead, or try Chrome."],
  ["not-allowed", "Microphone permission was refused. Allow it for this site, then reload."],
  ["service-not-allowed", "This browser refused the speech service. Type the turn instead, or try Chrome."],
  ["audio-capture", "No microphone was found."],
]);

export function recognitionFailure(code) {
  return FATAL_RECOGNITION_ERRORS.get(code) ?? null;
}

// Bounded so a long assistant answer cannot hold the speech queue open indefinitely, and so a
// pathological reply cannot be read aloud for minutes before the operator can send another turn.
const SPOKEN_MAX_CHARS = 1200;

export function spokenText(text) {
  const collapsed = String(text ?? "").replace(/\s+/g, " ").trim();
  return collapsed.length > SPOKEN_MAX_CHARS
    ? `${collapsed.slice(0, SPOKEN_MAX_CHARS)}...`
    : collapsed;
}

/** One controller owning microphone capture, playback, and the state the orb renders. */
export function createVoiceController({ onState, onTranscript, onError } = {}) {
  const emitState = (next) => {
    if (controller.state === next) return;
    controller.state = next;
    onState?.(next);
  };

  let recognition = null;
  let acknowledgeTimer = null;

  function stopAcknowledgeTimer() {
    if (acknowledgeTimer === null) return;
    clearTimeout(acknowledgeTimer);
    acknowledgeTimer = null;
  }

  function buildRecognition() {
    const instance = new RECOGNITION_CTOR();
    instance.lang = globalThis.navigator?.language || "en-US";
    instance.continuous = false;
    instance.interimResults = false;
    instance.maxAlternatives = 1;
    instance.addEventListener("result", (event) => {
      const alternative = event.results?.[0]?.[0];
      const heard = String(alternative?.transcript ?? "").trim();
      if (heard) onTranscript?.(heard);
    });
    instance.addEventListener("error", (event) => {
      // "aborted" and "no-speech" are ordinary outcomes of stopping or staying silent.
      if (event.error !== "aborted" && event.error !== "no-speech") {
        const fatal = recognitionFailure(event.error);
        // A blocked backend never recovers within the page, so stop offering the control.
        if (fatal) controller.inputBlocked = true;
        onError?.(fatal ?? `Speech recognition failed (${event.error}).`, event.error);
      }
      emitState("idle");
    });
    instance.addEventListener("end", () => {
      if (controller.state === "listening") emitState("idle");
    });
    return instance;
  }

  const controller = {
    state: "idle",
    // Set once a failure proves the backend will not serve this browser.
    inputBlocked: false,

    listen() {
      if (!voiceSupport.input || controller.inputBlocked) return false;
      if (controller.state === "listening") return false;
      controller.stopSpeaking();
      stopAcknowledgeTimer();
      recognition ??= buildRecognition();
      try {
        recognition.start();
      } catch {
        // start() throws if the previous session has not fully released the microphone.
        return false;
      }
      emitState("listening");
      return true;
    },

    stopListening() {
      if (controller.state !== "listening") return;
      recognition?.stop();
      emitState("idle");
    },

    /** The turn is in flight; the orb shows work rather than pretending to listen. */
    think() {
      controller.stopListening();
      stopAcknowledgeTimer();
      emitState("thinking");
    },

    speak(text) {
      const utteranceText = spokenText(text);
      if (!voiceSupport.output || !utteranceText) {
        controller.acknowledge();
        return false;
      }
      controller.stopSpeaking();
      const utterance = new globalThis.SpeechSynthesisUtterance(utteranceText);
      utterance.lang = globalThis.navigator?.language || "en-US";
      utterance.addEventListener("end", () => controller.acknowledge());
      utterance.addEventListener("error", () => emitState("idle"));
      emitState("speaking");
      globalThis.speechSynthesis.speak(utterance);
      return true;
    },

    stopSpeaking() {
      if (voiceSupport.output) globalThis.speechSynthesis.cancel();
      if (controller.state === "speaking") emitState("idle");
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

    reset() {
      stopAcknowledgeTimer();
      controller.stopSpeaking();
      controller.stopListening();
      emitState("idle");
    },
  };

  return controller;
}

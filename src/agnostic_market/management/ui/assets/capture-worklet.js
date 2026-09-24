// Microphone capture for the workbench voice preview.
//
// A worklet rather than the deprecated ScriptProcessorNode, and a worklet rather than
// MediaRecorder because the server wants raw PCM: a webm/opus container would need decoding
// there, and the page can hand over exactly what the STT engine accepts instead.
//
// Each block is copied before posting. The render quantum's buffer is reused by the browser,
// so a transferred view would be overwritten under the reader.

class CaptureProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0]?.[0];
    if (channel && channel.length) {
      this.port.postMessage(new Float32Array(channel));
    }
    return true;
  }
}

registerProcessor("capture-processor", CaptureProcessor);

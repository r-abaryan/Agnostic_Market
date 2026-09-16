# Release packaging, parked

Nothing in this directory is active. It is parked here on purpose so that no tool finds it
by default: GitHub Actions does not scan this path, and `docker build .` at the repository
root has no Dockerfile to pick up.

## Status

Written and reviewed by inspection on 2026-09-15. **The image has never been built.** No
Docker daemon was available, so dependency resolution, the `livekit-agents` native `rtc`
bindings, and any further system libraries are unconfirmed. A clean review of these files is
not evidence that they work.

## Why it is parked rather than finished

An image built today could not start and could not be certified:

- The canonical semantic-routing artifact carries a failed gate and superseded prompt and
  corpus fingerprints, so `QualifiedSemanticRouterFactory` refuses activation at startup.
- No schema-5 latency methodology is frozen, so no voice certification can bind to a build.
- `config/telemetry/semantic_routing_report.json` and `config/conformance/reports.json` are
  read during startup but are not in version control, so a CI build context does not contain
  them. The build check in the Dockerfile fails closed on exactly this.

Release packaging is the last step of the closeout, not the next one. See `docs/BUILD_PLAN.md`.

## Restoring

Move all three files back together; they are one unit and none works alone:

    release.yaml     -> .github/workflows/release.yaml
    Dockerfile       -> ./Dockerfile
    .dockerignore    -> ./.dockerignore

`.dockerignore` only takes effect at the build context root, and `release.yaml` builds with
`context: .` and the default Dockerfile path, so both must sit at the repository root.

Before restoring, confirm all three of these hold:

1. a semantic-routing artifact whose gate passes on the current corpus fingerprint,
2. a frozen schema-5 latency methodology,
3. the two startup evidence files above tracked in git.

Then build it once and fix whatever the first real build reveals, before trusting any digest
it produces.

## What the digest is for

`VoiceApplicationContract.build_artifact_digest` is required and validated as an OCI
`sha256` reference. It exists so certification evidence names the exact artifact that was
measured, which a Git label cannot do. Use `steps.<id>.outputs.digest` from
`docker/build-push-action`, which is the manifest digest. Do not use the image ID from
`docker images --no-trunc`; that is the config digest, a different value of identical shape
that will pass the regex while being wrong.

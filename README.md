---
name: spotify-particle-lyrics
description: Make a 16:9 (1920x1080, 30 fps) lyric video from a song MP3, its lyrics, 1-2 artist photos, the Spotify logo PNG and a reference reel. Four looks - tilted Spotify lyrics card with glowing sparks, monochrome stipple particle portrait that morphs between two photos with red/white kinetic lyrics, gold motion-graphic lyrics, and a black-and-white rain-on-glass music player. Everything is drawn in code (HTML canvas, headless Chrome, ffmpeg) and every word is timed by forced alignment. Use whenever the user types /spotify-particle-lyrics or asks for a Spotify-style lyric video, particle portrait lyric video, kinetic lyrics, or a "Kalyani"-style / "Let Me Love You"-style video for a song, even if they do not name the skill.
argument-hint: "<song.mp3> <lyrics.txt> <photo1> [photo2] <spotify-logo.png> <reference-reel.mp4>"
allowed-tools: Bash, Read, Write, Edit, Glob, Grep
---

# spotify-particle-lyrics

Turn a song + lyrics + photos into a four-look lyric video. Look reference: the Kalyani video (ARJN x KDS x Fifty4 x Ronn ft. Shreya Ghoshal). Kinetic reference for the right-side lyrics: the "Let Me Love You" kinetic video.

## Hard rules (every run)

1. **Everything is drawn in code.** One HTML page with a canvas, rendered frame by frame in headless Chrome (Puppeteer), encoded with ffmpeg. No stock footage, no image generators, no AI-made assets. The only inputs are the user's files. The look must be identical on every run, so use a **seeded PRNG** (mulberry32) and never `Math.random()` or `Date.now()` in the renderer.
2. **Never guess timings.** Every word time comes from forced alignment with `torchaudio.pipelines.MMS_FA` on the separated vocal stem. If alignment fails for a line, say so and show the user; do not interpolate silently.
3. **Contact sheet before full render.** Render stills, tile them, show the user, and wait for approval.
4. **Ask before overwriting** any existing file in the output folder (list the files, ask once).
5. **Deliver:** full-quality MP4 (1920x1080, 30 fps, original audio, short fade in/out) plus a 720p copy **under 20 MB** for WhatsApp.
6. **No artist names on screen** during the particle section. Names appear only on the Spotify card.
7. **Show each lyric only once** on screen at a time in the motion-graphic section. Never a duplicate big word behind or beside the same word.

## Inputs

Collect from the arguments or attached files. If something is missing, ask once in a single message.

| Input | Notes |
|---|---|
| Song MP3 (or a clip) | Full song or the clip to use. Video length = this audio. |
| Lyrics (text) | One line per sung line. Note the script (Devanagari, Roman, etc.). |
| Photo 1 | Artists. Used for cover art, particle portrait 1. |
| Photo 2 (optional) | Morph target. If missing, ask whether to reuse photo 1 with a mirrored/cropped variant or skip the morph. |
| Spotify logo PNG | Has transparent bars. Always composite on white (see Step 3). |
| Reference reel (MP4) | Only for the motion-graphic style (Step 2). |
| Title, artists, album text | Take from the ID3 tags / filename; confirm with the user. |

## Workspace

```
work/<song-slug>/
  audio/        original, vocals.wav, no_vocals.wav
  analysis/     words.json, lines.json, beats.json, sections.json, ref/
  assets/       cutouts, stipple points, fonts, logo
  render/       frames/, index.html, render.mjs
  out/          final MP4s, contact_sheet.png
```
Check what exists before creating anything. Never overwrite without asking.

## Setup check (run first)

```bash
python3 -c "import demucs, torch, torchaudio, librosa, soundfile, numpy; print(torch.__version__, torchaudio.__version__)"
node -v && ffmpeg -version | head -1
python3 -c "import uroman" 2>/dev/null || echo "uroman missing (needed for non-Latin lyrics)"
```
Install what is missing: `pip install demucs librosa soundfile uroman rembg onnxruntime opencv-python scipy`, `npm i puppeteer`. For fonts use Inter (weights 400-900) and Instrument Serif (regular + italic) from `@fontsource/inter` and `@fontsource/instrument-serif`, copied locally and loaded with `@font-face` + `document.fonts.load()` before frame 0. Do not link to the web at render time. If there is no network, tell the user which font files to drop into `assets/fonts/`.

---

## Step 1: Vocals, forced alignment, beats, sections

### 1a. Separate vocals
```bash
python3 -m demucs --two-stems=vocals -n htdemucs -o work/<slug>/audio "<song.mp3>"
```
Convert the vocals stem to 16 kHz mono float for alignment.

### 1b. Force-align every word (torchaudio MMS_FA)
- Bundle: `torchaudio.pipelines.MMS_FA`; `model = bundle.get_model(with_star=False)`; `tokenizer = bundle.get_tokenizer()`; `aligner = bundle.get_aligner()`.
- MMS_FA works on lowercase a-z and apostrophe. **For Hindi/Urdu/other scripts, romanize each word with `uroman`**, strip everything except `[a-z']`, and keep a map from the aligned token back to the original word shown on screen. If the user supplied a Roman transliteration alongside the original, use the Roman text for alignment and the original for display.
- Process the vocal in overlapping windows (about 30 s, 5 s overlap) for long songs, align each window against the lyric words expected in that window, and merge. Use the emission from the model, `torchaudio.functional.forced_align`, then `torchaudio.functional.merge_tokens`, then `aligner`-style word spans. Convert frames to seconds with `ratio = waveform.size(1) / emission.size(1) / sample_rate`.
- Repeated choruses: align the full lyric sequence in order, so repeats get their own times.
- Output `analysis/words.json`: `[{ "line": i, "word": "...", "display": "...", "start": s, "end": e, "score": p }]`.
- Flag any word with score < 0.35 or duration > 2.5 s or < 0.04 s. Show these to the user and ask whether to accept, fix the lyric text, or realign that window.
- Save `lines.json` with per-line start (first word start) and end (last word end + 0.15 s).

### 1c. Beat, tempo, kicks (librosa)
- On the full mix: `librosa.beat.beat_track` for tempo + beat times; `onset_strength` for an energy curve.
- Kicks: low-pass the **no_vocals** stem under ~150 Hz, `onset_detect` on that band, keep the strongest.
- Output `beats.json`: `{ tempo, beats[], kicks[], downbeats[], energy: [{t, v}] at 30 fps (normalised 0-1, smoothed) }`.
- The renderer reads `energy` per frame so every beat-reactive effect is a pure function of time.

### 1d. Rap vs sung
For each line, measure on the vocal stem: words per second, pitch variance (`librosa.pyin`, voiced fraction and std of semitone contour), and note-hold ratio (voiced frames with stable pitch for more than 250 ms). High word rate + flat/unstable pitch = **rap**; long held stable pitch + lower word rate = **sung**. Label each line `rap`, `sung`, or `unsure`. **Present the table to the user** (line number, text, label, confidence) and let them correct it. If the user knows who sings what (e.g. a named singer), prefer their answer.

### 1e. Propose sections
Propose, with timestamps and why, this order (adjust to the song, keep the order):

1. **Spotify card** (intro or first verse)
2. **Particles** (rap / hook, photo 1 -> morph -> blow apart)
3. **Spotify again** (short return, same card style, continues the lyrics)
4. **Motion graphics** (the sung singer part)
5. **Rain ending** (final chorus or outro)

Pick the **morph word** (a strong word on a downbeat in the particle section) and the **blow-apart** time (end of particle section). Save `sections.json`. **Stop and show the user the proposal (labels, sections, morph word). Wait for approval or edits.**

---

## Step 2: Analyse the reference reel (2 fps)

```bash
mkdir -p analysis/ref && ffmpeg -i "<reel>" -vf fps=2 analysis/ref/f_%04d.png
ffprobe -v error -show_entries stream=width,height,r_frame_rate,duration -of default=nw=1 "<reel>"
```
View the frames (in batches, contact-sheet them if there are many) and write `analysis/reference_notes.md` covering:

- **Fonts:** serif / sans, weight, case, italic, tracking, approximate size as a fraction of frame height. Name the closest free font (Instrument Serif, Playfair, Cormorant, etc.); keep it in the code as a CSS variable.
- **Colours:** sample hex values with a tiny script (`PIL` getpixel on gold text, background, glow). Record gold highlight, gold mid, shadow, background gradient, star colour.
- **Word timing:** how long each word stays on screen (in frames at 2 fps, then seconds), words per stack, gap between changes.
- **Transitions:** blur in/out amounts, scale, slide direction, any flash.
- **Camera moves:** push in, drift, rotation, cylinder turn of the lyric wall, parallax of moon/stars.
- **Elements:** moon (size, position, texture, glow), starfield (density, twinkle), wall (curvature, rows, spacing, speed).

If the reel has audio, note beat sync (do words change on kicks?). Present a 5-line summary to the user. The motion-graphics code in Step 6 must read its parameters from this file's values, so write them as a JSON block at the bottom of the notes (`style.json`).

---

## Renderer architecture

Single page `render/index.html` exposing `window.renderFrame(frameIndex)`; it must be a pure function of `t = frameIndex / 30` and the JSON data (`words`, `lines`, `beats`, `sections`, `style`, point clouds). Canvas 1920x1080, `devicePixelRatio = 1`.

- **Layers:** each look draws into its own offscreen canvas (`OffscreenCanvas` or hidden `canvas`), then the compositor draws the active look, applies transitions, then grain.
- **3D tilt (Spotify card):** do not use CSS. Render the card flat to an offscreen canvas, then draw it as a perspective-transformed quad using WebGL (texture on a quad with a projection matrix) or a mesh of small triangles on 2D canvas. WebGL is preferred: it also gives cheap blur and refraction shaders for the rain look. Launch Chrome with `--use-gl=swiftshader --enable-unsafe-swiftshader --ignore-gpu-blocklist` so it works headless.
- **Blur:** use WebGL gaussian passes or `ctx.filter = "blur(Npx)"` on offscreen canvases (OK in headless Chrome).
- **Determinism:** seeded PRNG per effect (`rng(seed + index)`); particle state computed analytically from `t` (no incremental simulation that depends on previous frames) so frames can be rendered in any order and in parallel.
- **Easing helpers:** `easeInOutCubic`, `easeOutBack(overshoot=1.5)`, `smoothstep`, `expo`.

### Frame loop (render.mjs)
```js
// launch N workers (N = cores - 1, max 6); each handles a contiguous frame range
const page = await browser.newPage();
await page.setViewport({ width: 1920, height: 1080, deviceScaleFactor: 1 });
await page.goto('file://.../index.html'); await page.evaluate(() => window.ready);
for (let f = start; f < end; f++) {
  await page.evaluate(i => window.renderFrame(i), f);
  const buf = await page.screenshot({ type: 'jpeg', quality: 95, clip: {x:0,y:0,width:1920,height:1080} });
  fs.writeFileSync(`frames/${String(f).padStart(6,'0')}.jpg`, buf);
}
```
Use PNG for contact-sheet stills; JPEG q95 for the full render to save time. Alternatively `canvas.toDataURL` per frame. Print progress every 150 frames and an ETA.

---

## Step 3: Look 1, Spotify lyrics card

- **Background:** deep maroon (about `#5a0f1f` to `#2a0710` vertical gradient with a faint radial lift behind the card), subtle noise.
- **Card:** rounded-rectangle panel in maroon with a lighter edge, containing the header and lyrics. Header: **cover art** (photo 1 cropped square, rounded corners 12 px), **song title** (Inter 800), **artists** (Inter 500, 70% white), and the **Spotify logo** at the top-right or bottom-left of the card.
- **Spotify logo:** the PNG has transparent bars. Draw a **white filled circle** (same diameter as the logo disc) first, then draw the PNG on top, so the bars read as white, not as the maroon background. Do not recolour the logo.
- **Lyrics:** Inter 800/900, large (about 7-8% of the card height per line), left aligned, generous line spacing, current line vertically centred in the card, others scrolling with easeInOut over 0.35 s when the line changes.
- **Blur by distance:** lines further from the sung line get progressively more blur (0 px on the current line, about 2 px for +-1, 4 px for +-2, 7 px for +-3+) and lower opacity.
- **Per-word light-up:** before a word is sung it is dark (about 28% white). From `start` to `end` of the word (min duration 0.18 s) the colour goes dark -> white with `easeInOutCubic`. Words already sung stay white on the current line; once the line is past they dim to about 45%.
- **Glow:** a soft white shadow (`shadowBlur` about 30 px, intensity tied to the fade progress) on the word being sung.
- **Spark burst:** on each word, emit 14-30 spark particles from **random x positions across the word's width** (not a single point), seeded by word index. Each spark: velocity up and outward, short life (0.5-0.9 s), size 1-3 px, warm white to pale gold, additive blend (`globalCompositeOperation = "lighter"`), gravity slightly negative, fade out with `1 - t^2`.
- **Cross glints:** 2-4 small four-point star glints (thin lines, 10-22 px) per word that scale in and out over 0.3 s.
- **Light point:** a bright 6 px dot with a larger soft halo that **rides along the word** left to right in sync with the fade (x = word.x + progress * word.width, at x-height mid), leaving a tiny short trail.
- **Camera:** card is tilted in 3D (about rotateY -18 deg, rotateX 8 deg, rotateZ -3 deg, perspective around 1400 px) with a **slow drift**: sinusoidal motion of +-2 deg and +-30 px over about 14 s, plus a very slow push-in (1.00 -> 1.06) over the section.
- **Shadow:** soft drop shadow under the card on the background.

The second Spotify section (Step 1e item 3) reuses the same code and continues the lyric list from the correct line, with the camera starting from a different drift phase.

---

## Step 4: Look 2, particle portrait

### 4a. Cut out the photos
Run `rembg` (`u2net_human_seg` model) on each photo, keep the alpha, crop to the subject, fit inside about 62% of frame height and place in the **left half** (centre x about 0.30 of the width). Save `assets/cutout_1.png`, `cutout_2.png`. Show the cutouts to the user in the contact sheet. If the cutout is bad (hair, hands), tell the user and offer a manual mask.

### 4b. Stipple points (20,000-25,000 each)
Build one point set per photo in Python (`assets/points_1.json`, `points_2.json`; `[x, y, size, brightness]` quantised to save space):
1. Grayscale the cutout; apply CLAHE (`cv2.createCLAHE(clipLimit=3, tileGridSize=(8,8))`) for **local contrast**.
2. Edge map: `cv2.Canny` on a blurred image plus a difference-of-gaussians, dilate 1 px.
3. Density map = `0.55 * (1 - normalised luminance)` blended with `0.35 * local contrast` and `0.45 * edges`, masked by alpha. Boost eyes, brows, nose and lips by using high edge weights (do not use face landmarks that need a network model; local contrast + edges is enough).
4. Sample points by **weighted Poisson-disc / rejection sampling** (or Lloyd relaxation of weighted Voronoi, 3-5 iterations) until the count is exactly the target (default 22,000, user-adjustable 20-25k). Both photos get the **same count** so each dot has a partner (sort by a space-filling curve or match with a greedy nearest-neighbour on a coarse grid so partners are spatially coherent).
5. Dot size 0.8-2.2 px based on local density (darker/denser regions = larger and brighter).
Check the portrait reads as a face: render it at 1080p and view it before continuing. If the face is mushy, raise edge weight and contrast, and re-sample.

### 4c. Look
- Background near-black `#050507` with a very faint vignette. Dots in **white on near-black**, additive blend with low alpha so overlapping dots brighten.
- **Breathing:** each dot's scale and radial offset pulse with the `energy` curve and kicks: `scale = 1 + 0.35 * kickEnv(t)`, and a low-amplitude per-dot noise offset (`+-1.5 px`) phase-shifted by the dot's index so the portrait shimmers instead of pulsing as one block.
- **Embers:** 250-400 floating particles, mix of **red (`#e5143a`) and white**, rising and drifting right with turbulence, additive, size 1-3 px, life 2-4 s, spawning more on kicks. All seeded.
- **Morph (on the chosen word):** at the morph word's `start`, run for about 1.1 s. Each dot moves from its photo-1 position to its photo-2 partner along a **swirl path** (rotate around the portrait centre by up to 1.2 rad while interpolating, with an out-and-back radial bulge). Use **staggered timing**: per-dot delay in 0-0.45 s (by distance from the centre, plus noise), per-dot duration 0.55 s, `easeInOutCubic`. At the moment of peak motion draw a **flash ring**: an expanding circle (radius 0 -> 700 px over 0.5 s), white at about 0.8 alpha fading out, 3 px line plus a soft glow, and a brief 4% white screen flash.
- **Blow apart (end of the section):** at the section's last beat, each dot gets a velocity away from the centre (speed 300-1400 px/s, random angle bias, slight upward), with drag and fade-out over 1.2 s, and the embers burst too. Then cut/flash to the next section.
- **Names:** do not render any artist names or titles in this section.

---

## Step 5: Right-side kinetic lyrics (during the particles)

Reference: the "Let Me Love You" kinetic lyric video. Right half of the frame (x from about 0.52 to 0.95 of the width), vertically centred.

- **Blocks:** group the lyric lines of the section into blocks of 2-4 lines (a natural phrase). One block is on screen at a time.
- **Rows alternate** typography: row A = **Inter 900, UPPERCASE, white**; row B = **Instrument Serif italic, red (`#e5143a`)**, mixed case, slightly larger because serif x-height is smaller. Fit each row to the column width (reduce font size so a row fits; wrap long lines in two rows, keeping the alternation).
- **Word pop:** each word appears on its aligned `start` with scale `0.6 -> 1.0` using `easeOutBack` (overshoot about 1.4), opacity 0 -> 1 in 0.12 s. Words not yet sung are not drawn (or drawn at 8% alpha if the user wants a preview).
- **Beat dance:** every letter bounces on each beat: `dy = -amp * kickEnv(t - phase_i)` with amp 6-14 px and a small per-letter phase offset (index * 0.012 s) so the bounce rolls across the word. Optional +-1.5 deg rotation wobble.
- **Block exit:** after the last word of a block ends (+0.25 s), blur it out (blur 0 -> 18 px, opacity 1 -> 0, slight upward drift, 0.3 s), then the next block starts. Never show two blocks at once.
- The words must not overlap the portrait; if the portrait is wider than 0.45 of the width, shift the portrait left or shrink it.

---

## Step 6: Look 3, gold motion graphics (sung part, reference style)

Use `analysis/style.json` from Step 2 for fonts, colours, timing and camera. Defaults when the reference is silent: gold serif words (Instrument Serif or the closest serif), gradient fill `#fff3c4 -> #e8b84a -> #9a6a1c`, glow `rgba(232,184,74,0.55)`.

Elements (all in code):
1. **Stacked word changes:** the current lyric phrase is shown as 1-3 stacked words (large, centred or off-centre like the reference). On each word change the old word **blurs out** (blur 0 -> 24 px, opacity -> 0, scale 1 -> 1.08) while the new one **blurs in** (blur 24 -> 0, scale 0.94 -> 1) in 4-7 frames, synced to the aligned word start. Hold times come from the alignment, not from a fixed rate.
2. **Moon on a starfield:** procedural moon (radial gradient, crater noise from a seeded value-noise texture, soft gold-white rim glow), slow drift and parallax. Starfield of 300-600 stars in 3 depth layers with twinkle (`sin(t*f + phase)`), shooting star occasionally on a kick (seeded).
3. **Curved lyric wall:** a wall of the section's lyric lines in small gold serif text, mapped onto a **cylinder** (x angle = column / R; scale by `cos(angle)`; fade by depth) that **turns slowly** (about 0.15 rad/s) and accelerates slightly on kicks. Rows scroll at slightly different spee

"""Render the hair-clip style video from the cleaned storyboard frames.

Usage: python3 build_video.py <frames_dir> <out.mp4>

Each storyboard panel becomes one shot with eased Ken Burns motion and a title
card; shots are joined with crossfades. The audio track (chime, sparkle,
hair-clip clicks, soft music bed with fade-out) is synthesized here so the
render needs no external assets beyond the fonts in assets/fonts.
"""
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
FONT = ROOT / "assets/fonts/PlayfairDisplay.ttf"
FONT_ITALIC = ROOT / "assets/fonts/PlayfairDisplay-Italic.ttf"

W, H, FPS = 1080, 1920, 30
SHOT = 3.5        # seconds per shot
XFADE = 0.6       # crossfade length
SR = 44100

DARK, LIGHT = "0x241914", "0xFFFFFF"

# frame, title, subtitle, text colour, title y, zoom start/end, focus (x, y) start/end as fractions
SHOTS = [
    ("01_reveal.png", "THE REVEAL", "macro", DARK, 230, (1.00, 1.14), (0.50, 0.50), (0.48, 0.56)),
    ("02_detail.png", "DETAIL CHECK", "sparkle close-up", DARK, 230, (1.04, 1.22), (0.50, 0.50), (0.56, 0.42)),
    ("03_placement.png", "PLACEMENT", "style action", LIGHT, 230, (1.06, 1.20), (0.55, 0.48), (0.46, 0.36)),
    ("04_grip.png", "SECURE GRIP", "hold check", LIGHT, 230, (1.00, 1.16), (0.50, 0.46), (0.42, 0.38)),
    ("05_final.png", "FINAL LOOK", "elevated style", DARK, 110, (1.08, 1.00), (0.52, 0.30), (0.50, 0.50)),
]

TOTAL = len(SHOTS) * SHOT - (len(SHOTS) - 1) * XFADE


def shot_start(i):
    return i * (SHOT - XFADE)


# ---------------------------------------------------------------- audio

def env_exp(n, tau):
    return np.exp(-np.arange(n) / (tau * SR))


def place(track, sig, at):
    i = int(at * SR)
    end = min(len(track), i + len(sig))
    track[i:end] += sig[: end - i]


def comb_reverb(x, delays=(1557, 1617, 1491, 1422), g=0.72, mix=0.3):
    wet = np.zeros_like(x)
    for d in delays:
        y = x.copy()
        for s in range(d, len(y), d):
            e = min(len(y), s + d)
            y[s:e] += g * y[s - d:e - d]
        wet += y
    return (1 - mix) * x + mix * wet / len(delays)


def chime():
    n = int(2.5 * SR)
    t = np.arange(n) / SR
    sig = np.zeros(n)
    for f, a, tau in [(1318.5, 1.0, 0.9), (1975.5, 0.5, 0.6), (2637.0, 0.35, 0.45), (3520.0, 0.2, 0.3)]:
        sig += a * np.sin(2 * np.pi * f * t) * env_exp(n, tau)
    return 0.45 * sig


def sparkle(rng):
    n = int(1.6 * SR)
    sig = np.zeros(n)
    for _ in range(16):
        f = rng.uniform(3000, 7500)
        m = int(rng.uniform(0.08, 0.25) * SR)
        t = np.arange(m) / SR
        ping = np.sin(2 * np.pi * f * t) * env_exp(m, 0.04) * rng.uniform(0.3, 1.0)
        place(sig, ping, rng.uniform(0.0, 1.0))
    noise = np.diff(rng.standard_normal(n + 1)) * 0.04  # crude high-pass shimmer
    swell = np.sin(np.pi * np.clip(np.arange(n) / (1.1 * SR), 0, 1)) ** 2
    return 0.35 * (sig + noise * swell)


def click(rng, tone=2600, tau=0.025, level=0.6):
    n = int(0.15 * SR)
    t = np.arange(n) / SR
    burst = rng.standard_normal(n) * env_exp(n, 0.003)
    snap = np.sin(2 * np.pi * tone * t) * env_exp(n, tau)
    body = np.sin(2 * np.pi * tone / 4 * t) * env_exp(n, tau * 1.5)
    return level * (0.5 * burst + 0.6 * snap + 0.4 * body)


def music():
    n = int((TOTAL + 0.5) * SR)
    out = np.zeros(n)
    t_all = np.arange(n) / SR
    midi = lambda m: 440 * 2 ** ((m - 69) / 12)
    chords = [[53, 57, 60, 64], [52, 55, 59, 62], [50, 53, 57, 60], [48, 52, 55, 59]]  # Fmaj7 Em7 Dm7 Cmaj7
    seg = TOTAL / len(chords)
    for k, chord in enumerate(chords):
        a, b = int(k * seg * SR), min(n, int(((k + 1) * seg + 0.8) * SR))
        t = t_all[: b - a]
        env = np.minimum(1, t / 0.9) * np.minimum(1, (len(t) - np.arange(len(t))) / (0.8 * SR))
        pad = sum(np.sin(2 * np.pi * midi(m) * t) + 0.25 * np.sin(4 * np.pi * midi(m) * t) for m in chord)
        out[a:b] += 0.06 * pad * env
        # music-box arpeggio an octave up
        for j, m in enumerate(chord + [chord[1] + 12]):
            at = k * seg + 0.25 + j * seg / 5.5
            m_n = int(1.4 * SR)
            tt = np.arange(m_n) / SR
            note = np.sin(2 * np.pi * midi(m + 12) * tt) * env_exp(m_n, 0.45)
            note += 0.3 * np.sin(2 * np.pi * midi(m + 24) * tt) * env_exp(m_n, 0.2)
            place(out, 0.09 * note, at)
    out = comb_reverb(out)
    fade_in = np.minimum(1, t_all / 1.0)
    fade_out = np.clip((TOTAL - t_all) / 3.5, 0, 1) ** 1.5
    return out * fade_in * fade_out


def write_audio(path):
    rng = np.random.default_rng(7)
    track = music()
    place(track, chime(), 0.45)                                   # THE REVEAL: chime
    place(track, sparkle(rng), shot_start(1) + 0.35)              # DETAIL CHECK: sparkle
    place(track, click(rng, 3000, 0.015, 0.35), shot_start(2) + 1.55)   # PLACEMENT: hair-click
    place(track, click(rng, 2400, 0.03, 0.6), shot_start(2) + 1.68)
    place(track, click(rng, 1800, 0.035, 0.75), shot_start(3) + 1.2)    # SECURE GRIP: satisfying click
    track = track[: int(TOTAL * SR)]
    track = track / np.max(np.abs(track)) * 0.89
    stereo = np.stack([track, np.roll(track, 220) * 0.97], axis=1)  # slight width
    pcm = (stereo * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


# ---------------------------------------------------------------- video

def esc(s):
    return s.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def title_filters(title, sub, color, ty):
    # Fade in 0.35-0.95s, fade out over the last 0.7s before the crossfade ends, small upward drift.
    alpha = f"if(lt(t,0.35),0,if(lt(t,0.95),(t-0.35)/0.6,if(gt(t,{SHOT - 0.75}),max(0,({SHOT - 0.2}-t)/0.55),1)))"
    rise = "28*(1-min(1,max(0,(t-0.35)/0.8)))"
    shadow = "shadowcolor=black@0.35:shadowx=2:shadowy=3" if color == LIGHT else "shadowcolor=white@0.25:shadowx=0:shadowy=2"
    common = f"fontcolor={color}:alpha='{alpha}':{shadow}:x=(w-text_w)/2"
    return (
        f"drawtext=fontfile='{esc(str(FONT))}':text='{esc(title)}':fontsize=104:{common}:y='{ty}+{rise}',"
        f"drawtext=fontfile='{esc(str(FONT_ITALIC))}':text='{esc(sub)}':fontsize=56:{common}:y='{ty + 130}+{rise}'"
    )


def shot_filter(i, title, sub, color, ty, z, c0, c1):
    n = int(round(SHOT * FPS))
    ease = f"((1-cos(PI*on/{n - 1}))/2)"
    zoom = f"({z[0]}+({z[1] - z[0]:.3f})*{ease})"
    cx = f"({c0[0]}+({c1[0] - c0[0]:.3f})*{ease})"
    cy = f"({c0[1]}+({c1[1] - c0[1]:.3f})*{ease})"
    x = f"max(0,min(iw-iw/zoom,{cx}*iw-iw/zoom/2))"
    y = f"max(0,min(ih-ih/zoom,{cy}*ih-ih/zoom/2))"
    # Upscale 4x before zoompan so the sub-pixel pan does not visibly jitter.
    return (
        f"[{i}:v]scale={W * 4}:-1:flags=lanczos,crop={W * 4}:{H * 4},"
        f"zoompan=z='{zoom}':x='{x}':y='{y}':d={n}:s={W}x{H}:fps={FPS},"
        f"setsar=1,format=yuv420p,{title_filters(title, sub, color, ty)},"
        f"trim=duration={SHOT},setpts=PTS-STARTPTS[s{i}]"
    )


def build(frames_dir, out_path):
    with tempfile.TemporaryDirectory() as tmp:
        audio = Path(tmp) / "audio.wav"
        write_audio(audio)

        parts = [shot_filter(i, *s[1:]) for i, s in enumerate(SHOTS)]
        prev = "s0"
        for i in range(1, len(SHOTS)):
            label = f"x{i}" if i < len(SHOTS) - 1 else "vx"
            parts.append(f"[{prev}][s{i}]xfade=transition=fade:duration={XFADE}:offset={shot_start(i):.3f}[{label}]")
            prev = label
        parts.append(f"[vx]fade=t=in:st=0:d=0.5,fade=t=out:st={TOTAL - 0.8:.3f}:d=0.8[v]")
        script = Path(tmp) / "graph.txt"
        script.write_text(";\n".join(parts))

        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
        for s in SHOTS:
            cmd += ["-i", str(Path(frames_dir) / s[0])]
        cmd += [
            "-i", str(audio), "-filter_complex_script", str(script),
            "-map", "[v]", "-map", f"{len(SHOTS)}:a",
            "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
            "-t", f"{TOTAL:.3f}", str(out_path),
        ]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])

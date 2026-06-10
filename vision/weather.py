# vision/weather.py
"""
Screen-based weather recognition for Minecraft.

The bot never reads game data — it infers the weather the same way a
human glancing at the screen would. Weather matters because it changes
how blocks LOOK: rain darkens and desaturates the world and overlays
grey streaks; snow brightens and adds white flakes; thunder dims further
and flashes; a clear day is bright and blue. Block recognition can
condition on the weather so a wet, dim ``grass_block`` isn't mistaken
for something else.

States (what ``detect`` reports)
--------------------------------
``clear``    — no precipitation, sky visible.
``rain``     — raining (grey, desaturated, dimmed sky). A thunderstorm
               reports as ``rain`` too: it's visually just darker rain,
               and the lightning that distinguishes it is rare and lasts
               a couple of frames, so it isn't a useful steady state.
               (``thunder`` stays a valid TRAINING label for anyone who
               wants to collect it; the heuristic just folds it in.)
``snow``     — snowing (cold biome; white flakes, bright but desaturated).
``unknown``  — sky not visible (cave / fully enclosed) → can't tell.

Transitions
-----------
MC fades weather in and out over several seconds. A single mid-fade
frame is therefore ambiguous, so ``detect`` majority-votes over the last
``smooth_window`` frames and the collection tool waits for the fade to
finish before capturing.

Two recognisers, same interface
-------------------------------
* **Heuristic** (always available): a handful of location-robust
  features — how "open" the top-of-frame sky is, its brightness /
  saturation, and mid-frame edge/motion energy — fed through tuned
  thresholds. Works out of the box; rough on the rain/snow split.
* **Trained** (optional, more accurate): the same feature vectors,
  labelled by driving ``/weather <state>`` via chat commands and
  capturing frames (see ``tools/collect_weather.py``). At runtime a
  nearest-centroid classifier over z-normalised features beats the
  thresholds. Falls back to the heuristic until enough data exists.

Design deliberately mirrors ``vision.world.sample_recognizer`` so the
"collect labelled samples → recogniser improves silently" loop is the
same one the block classifier already uses.

Sound (rain hiss, thunder cracks) is a promising extra signal the user
suggested; this module is screen-only by design, but the feature dict
is open so an audio feature could be appended later without changing
the interface.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from vision.world.types import WeatherObservation


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------

class WeatherState:
    CLEAR   = "clear"
    RAIN    = "rain"
    SNOW    = "snow"
    THUNDER = "thunder"
    UNKNOWN = "unknown"

    ALL = ("clear", "rain", "snow", "thunder", "unknown")
    # States that require sky to be visible to even be possible.
    SKY_STATES = ("clear", "rain", "snow", "thunder")


# Order of the feature vector. Kept explicit so the on-disk training
# samples stay interpretable and a future feature can be appended
# without silently shifting columns.
# All features are PER-FRAME (no temporal term) so the verdict doesn't
# depend on how often we sample — important because weather detection is
# throttled to a slow cadence. (An earlier ``air_motion`` inter-frame
# term was dropped: it barely moved on MC rain and its value depended on
# the sampling interval, which broke the trained classifier when train
# and inference cadences differed.)
FEATURE_KEYS = (
    "sky_open_frac",     # fraction of the top band that reads as open sky
    "sky_brightness",    # mean V (0..1) of the masked sky pixels
    "sky_saturation",    # mean S (0..1) of the masked sky pixels
    "sky_blueness",      # (B - R) / 255 mapped to 0..1; blue sky vs grey
    "air_edge_density",  # gradient density in the mid band (streaks/flakes)
    "air_whiteness",     # fraction of mid-band pixels that are bright + grey
    "global_brightness", # mean V (0..1) of the whole frame
)


@dataclass
class WeatherDetectorConfig:
    # Region fractions (of frame W/H). The sky band is a central top
    # strip — central x avoids BOTH F3 text columns (left + right).
    sky_y0: float = 0.02
    sky_y1: float = 0.22
    sky_x0: float = 0.30
    sky_x1: float = 0.70
    # The "air" band is where falling precipitation is most visible
    # against distant terrain, clear of the hotbar / hand / HUD.
    air_y0: float = 0.25
    air_y1: float = 0.58
    air_x0: float = 0.20
    air_x1: float = 0.80

    # Sky-openness: a pixel is "open sky" when its local gradient is
    # below this (smooth) — works day OR night, unlike a brightness
    # test. Below ``cave_open_frac`` of the band being open → no sky
    # (cave / enclosed) → unknown.
    sky_smooth_grad_max: float = 14.0
    cave_open_frac: float = 0.18
    # …and below this fraction of open sky (but above the cave floor)
    # the band is PARTLY obscured by terrain/trees on the horizon — a
    # mixed sky+ground band reads as non-blue and would misclassify as
    # rain, so we HOLD the last verdict instead of classifying a dirty
    # view. A clean open-sky look (≈0.9+) is needed to (re)assess.
    # Clear-sky live reads were ≈0.95-1.0; horizon-terrain views ≈0.65.
    # High (0.85): only (re)assess the weather from a genuinely clean,
    # near-pure-sky view. Both clear AND rain skies are smooth/uniform
    # (open_frac ≈0.95-1.0), whereas a mid-scan view with terrain/trees
    # creeping into the band sits lower and produced confident-WRONG
    # reads. We instead HOLD the last verdict on those — which is also
    # the right behaviour for the main use case: weather is latched while
    # the agent can see the sky, then carried (held) to the moments it's
    # crosshairing a block to tag the sample, where the sky isn't in view.
    weather_min_sky_open: float = 0.85

    # Heuristic thresholds. Measured live (daytime) with /weather:
    #   clear  : brightness≈1.00  blueness≈0.96
    #   rain   : brightness≈0.51  blueness≈0.65
    #   thunder: brightness≈0.20  blueness≈0.56
    # MC dims the sky for rain and dims it HARD for thunder, and the
    # grey overcast cuts the blue channel — so sky brightness (+ a
    # blueness guard so a merely-dim-but-still-blue clear sky isn't
    # called rain) separates the states well in daylight. NOTE: this is
    # daytime-oriented; at night a clear sky is also dim, so the
    # heuristic can't reliably tell night-clear from rain — collect
    # trained samples across times of day (tools/collect_weather.py) for
    # night robustness, which supersedes these thresholds.
    clear_brightness_min: float = 0.78   # bright sky → clear (day)
    rain_brightness_max: float = 0.70    # dimmed sky → rain
    thunder_brightness_max: float = 0.34 # very dark sky → thunder
    clear_blueness_min: float = 0.80     # blue sky → clear, blocks false-rain
    precip_blueness_max: float = 0.82    # grey (low blue) supports precip
    snow_whiteness_min: float = 0.08     # white flakes in the air band
    precip_edge_min: float = 0.04        # mid-band edge density for precip

    # Weather is only readable when the camera can actually SEE the sky.
    # The scanning agents spend much of their time pitched DOWN (MC
    # pitch > 0 = looking down), where the top "sky band" is really
    # terrain — a smooth, non-blue patch that the heuristic would
    # mislabel as rain. When the pitch exceeds this (looking down too
    # far) we don't update the weather; we HOLD the last verdict instead
    # of flapping. Looking level (0) or up (negative) is fine.
    max_assess_pitch_deg: float = 18.0

    # Trained classifier: need at least this many labelled samples for a
    # state before we trust the centroid for it, and ≥2 states overall.
    min_samples_per_state: int = 8

    # Temporal smoothing: emit the majority of the last N raw verdicts
    # so a transient frame (a passing cloud, a camera pan across a dim
    # sky region, a mid-fade transition) doesn't flip the reported
    # state. The scanning agents pan constantly, so a wider window
    # trades a little latency on a real weather change for far fewer
    # spurious flips. 1 disables smoothing.
    smooth_window: int = 9
    # Fraction of a full window a NEW state must win before the reported
    # weather flips (hysteresis). Higher = stickier / less flapping but
    # slower to react to a real change. 0.67 = two-thirds super-majority.
    flip_majority: float = 0.67


def default_weather_store_path() -> Path:
    return (Path(__file__).resolve().parent.parent
            / "data" / "training" / "weather_samples.json")


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _region(frame: np.ndarray, x0f, y0f, x1f, y1f) -> np.ndarray:
    h, w = frame.shape[:2]
    x0, x1 = int(w * x0f), int(w * x1f)
    y0, y1 = int(h * y0f), int(h * y1f)
    x0, x1 = max(0, min(x0, w - 1)), max(1, min(x1, w))
    y0, y1 = max(0, min(y0, h - 1)), max(1, min(y1, h))
    if x1 <= x0 or y1 <= y0:
        return frame[0:1, 0:1]
    return frame[y0:y1, x0:x1]


class WeatherDetector:
    """
    Per-frame weather recogniser. Construct once, call :meth:`detect`
    each (fresh) tick. Maintains the previous mid-band greyscale crop
    internally so it can measure inter-frame motion without the caller
    threading frames through.
    """

    def __init__(self,
                 config: Optional[WeatherDetectorConfig] = None,
                 store_path: Optional[Path] = None):
        self.cfg = config or WeatherDetectorConfig()
        self._store_path = Path(store_path) if store_path else default_weather_store_path()
        self._recent: List[str] = []      # recent raw verdicts (smoothing)
        self._reported_state: Optional[str] = None  # hysteresis: current output
        self._last_obs: Optional[WeatherObservation] = None  # held when unreadable

        # Trained centroids: {state: (mean_vec, n)}. Built from disk.
        self._centroids: Dict[str, np.ndarray] = {}
        self._feat_mean: Optional[np.ndarray] = None
        self._feat_std: Optional[np.ndarray] = None
        self._trained = False
        self.reload_training()

    # ── Feature extraction ────────────────────────────────────────

    def extract_features(self, frame_rgb: np.ndarray) -> Dict[str, float]:
        """Compute the per-frame weather feature dict for one RGB frame.

        Stateless (no temporal term), so it can be called at any cadence
        — which matters because weather detection is throttled.
        """
        cfg = self.cfg
        if frame_rgb is None or frame_rgb.size == 0:
            return {k: 0.0 for k in FEATURE_KEYS}

        sky = _region(frame_rgb, cfg.sky_x0, cfg.sky_y0, cfg.sky_x1, cfg.sky_y1)
        air = _region(frame_rgb, cfg.air_x0, cfg.air_y0, cfg.air_x1, cfg.air_y1)

        sky_gray = cv2.cvtColor(sky, cv2.COLOR_RGB2GRAY) if sky.ndim == 3 else sky
        # Sky openness: low local gradient = smooth = open sky (works at
        # any brightness, so it survives night). Sobel magnitude per px.
        gx = cv2.Sobel(sky_gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(sky_gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx * gx + gy * gy)
        sky_mask = grad < cfg.sky_smooth_grad_max     # the actual open-sky pixels
        sky_open_frac = float(np.mean(sky_mask))

        # Colour stats over ONLY the smooth sky pixels — so green terrain
        # or trees creeping into the band don't drag the blueness down
        # and trip a false "rain". Falls back to the whole band when too
        # few sky pixels survive the mask.
        if sky.ndim == 3:
            if int(sky_mask.sum()) >= max(16, int(0.05 * sky_mask.size)):
                px = sky[sky_mask].astype(np.float32)         # (n, 3) RGB
            else:
                px = sky.reshape(-1, 3).astype(np.float32)
            ch_max = px.max(axis=1)
            ch_min = px.min(axis=1)
            sky_brightness = float(ch_max.mean()) / 255.0
            with np.errstate(divide="ignore", invalid="ignore"):
                sat = np.where(ch_max > 0, (ch_max - ch_min) / ch_max, 0.0)
            sky_saturation = float(np.mean(sat))
            sky_blueness = max(0.0, min(
                1.0, float((px[:, 2] - px[:, 0]).mean()) / 255.0 + 0.5))
        else:
            sky_brightness = float(sky_gray.mean()) / 255.0
            sky_saturation = 0.0
            sky_blueness = 0.5

        air_gray = cv2.cvtColor(air, cv2.COLOR_RGB2GRAY) if air.ndim == 3 else air
        agx = cv2.Sobel(air_gray, cv2.CV_32F, 1, 0, ksize=3)
        agy = cv2.Sobel(air_gray, cv2.CV_32F, 0, 1, ksize=3)
        agrad = np.sqrt(agx * agx + agy * agy)
        air_edge_density = float(np.mean(agrad > 30.0))

        if air.ndim == 3:
            air_hsv = cv2.cvtColor(air, cv2.COLOR_RGB2HSV)
            white = (air_hsv[..., 2] > 200) & (air_hsv[..., 1] < 40)
            air_whiteness = float(np.mean(white))
        else:
            air_whiteness = float(np.mean(air_gray > 200))

        global_brightness = float(np.mean(frame_rgb)) / 255.0

        return {
            "sky_open_frac":     sky_open_frac,
            "sky_brightness":    sky_brightness,
            "sky_saturation":    sky_saturation,
            "sky_blueness":      sky_blueness,
            "air_edge_density":  air_edge_density,
            "air_whiteness":     air_whiteness,
            "global_brightness": global_brightness,
        }

    # ── Detection ─────────────────────────────────────────────────

    def detect(self, frame_rgb: np.ndarray,
               *, pitch: Optional[float] = None) -> WeatherObservation:
        """Classify the weather in ``frame_rgb``.

        ``pitch`` (MC degrees; >0 = looking down) lets the detector skip
        frames where the camera is pitched too far down to see the sky —
        on those it HOLDS the last verdict rather than misreading ground
        as rain. Pass ``None`` when pose is unknown (F3 off): the
        sky-openness gate is then the only guard.
        """
        feats = self.extract_features(frame_rgb)
        sky_visible = feats["sky_open_frac"]

        # Looking too far down → the top band isn't sky. Don't update;
        # return the last good verdict (or unknown if we have none yet).
        if pitch is not None and pitch > self.cfg.max_assess_pitch_deg:
            if self._last_obs is not None:
                return self._last_obs
            return WeatherObservation(
                state=WeatherState.UNKNOWN, confidence=0.3,
                sky_visible=round(sky_visible, 3), source="heuristic",
                features={k: round(v, 4) for k, v in feats.items()})

        if sky_visible < self.cfg.cave_open_frac:
            raw_state, conf, source = (WeatherState.UNKNOWN,
                                       _cave_conf(sky_visible, self.cfg),
                                       "heuristic")
            obs = WeatherObservation(
                state=self._smooth(raw_state), confidence=round(conf, 3),
                sky_visible=round(sky_visible, 3), source=source,
                features={k: round(v, 4) for k, v in feats.items()})
            self._last_obs = obs
            return obs
        elif sky_visible < self.cfg.weather_min_sky_open:
            # Sky partly blocked by horizon terrain → dirty read. Hold
            # the last verdict rather than misclassify a mixed band.
            if self._last_obs is not None:
                return self._last_obs
            return WeatherObservation(
                state=WeatherState.UNKNOWN, confidence=0.3,
                sky_visible=round(sky_visible, 3), source="heuristic",
                features={k: round(v, 4) for k, v in feats.items()})

        if self._trained:
            raw_state, conf = self._classify_trained(feats)
            source = "trained"
        else:
            raw_state, conf = self._classify_heuristic(feats)
            source = "heuristic"

        state = self._smooth(raw_state)
        obs = WeatherObservation(
            state=state,
            confidence=round(conf, 3),
            sky_visible=round(sky_visible, 3),
            source=source,
            features={k: round(v, 4) for k, v in feats.items()},
        )
        self._last_obs = obs
        return obs

    def _smooth(self, raw_state: str) -> str:
        """Hysteretic majority vote over the recent raw verdicts.

        Single-frame weather reads are noisy under a panning camera (a
        clear sky has non-blue patches — sun glare, horizon haze, clouds
        — that momentarily read as rain). So we only CHANGE the reported
        state when a candidate wins a clear super-majority of the window;
        otherwise the previous report holds. This keeps the output
        stable on whatever genuinely dominates the recent views.
        """
        win = max(1, int(self.cfg.smooth_window))
        self._recent.append(raw_state)
        if len(self._recent) > win:
            self._recent = self._recent[-win:]
        counts: Dict[str, int] = {}
        for s in self._recent:
            counts[s] = counts.get(s, 0) + 1
        top, n = max(counts.items(), key=lambda kv: kv[1])

        if self._reported_state is None:
            self._reported_state = top
        elif top != self._reported_state:
            # Require a super-majority of a reasonably-full window to flip,
            # so transient misreads can't switch the reported weather.
            need = max(2, int(math.ceil(self.cfg.flip_majority * len(self._recent))))
            if len(self._recent) >= win and n >= need:
                self._reported_state = top
        return self._reported_state

    def reset(self) -> None:
        self._recent = []
        self._reported_state = None
        self._last_obs = None

    # ── Heuristic classifier ──────────────────────────────────────

    def _classify_heuristic(self, f: Dict[str, float]):
        """Daytime-oriented heuristic over sky brightness + blueness (the
        signals that separated the states in live capture). Output is one
        of clear / rain / snow / unknown — thunder is folded into rain
        because a thunderstorm is, visually, just darker rain with brief
        lightning flashes (rare and momentary), not a distinct steady
        state worth reporting on its own.
        """
        cfg = self.cfg
        b = f["sky_brightness"]
        blue = f["sky_blueness"]

        # Blueness is the PRIMARY discriminator: rain greys the sky (cuts
        # the blue channel) regardless of how bright it happens to be,
        # whereas a clear sky stays blue even when the agent pans toward
        # a dimmer region or the sun drops. Live: clear blue≈0.94-0.96,
        # rain blue≈0.65 — a 0.78 split is wide. This keeps a dim-but-
        # blue clear sky from flapping to "rain" while the camera scans.
        if blue >= cfg.clear_blueness_min:
            # Brightness only modulates confidence here.
            conf = min(1.0, 0.55 + max(0.0, b - cfg.rain_brightness_max) * 0.6)
            return WeatherState.CLEAR, conf

        # Grey sky (blue dropped) → precipitation.
        if blue <= cfg.precip_blueness_max:
            if f["air_whiteness"] > cfg.snow_whiteness_min:
                return WeatherState.SNOW, min(1.0, 0.45 + f["air_whiteness"] * 2.0)
            # Greyer + darker → more confident it's rain (thunder folds in).
            greyness = (cfg.clear_blueness_min - blue)
            dim = max(0.0, cfg.clear_brightness_min - b)
            return WeatherState.RAIN, min(1.0, 0.45 + greyness + dim * 0.3)

        # Narrow ambiguous band between the two blueness thresholds —
        # keep the previous verdict's bias by defaulting to clear softly.
        return WeatherState.CLEAR, 0.4

    # ── Trained classifier (nearest centroid, z-normalised) ───────

    def _vec(self, f: Dict[str, float]) -> np.ndarray:
        return np.array([f[k] for k in FEATURE_KEYS], dtype=np.float32)

    def _classify_trained(self, f: Dict[str, float]):
        q = self._vec(f)
        if self._feat_std is not None:
            q = (q - self._feat_mean) / self._feat_std
        best_state, best_d = WeatherState.CLEAR, float("inf")
        second_d = float("inf")
        for state, c in self._centroids.items():
            d = float(np.linalg.norm(q - c))
            if d < best_d:
                second_d, best_d, best_state = best_d, d, state
            elif d < second_d:
                second_d = d
        # Confidence from the margin between nearest and runner-up.
        if math.isfinite(second_d) and (best_d + second_d) > 1e-6:
            conf = min(1.0, max(0.0, (second_d - best_d) / (second_d + best_d) + 0.3))
        else:
            conf = 0.5
        return best_state, conf

    # ── Training persistence ──────────────────────────────────────

    def reload_training(self) -> None:
        """(Re)build the trained centroids from the on-disk sample file.

        Silent no-op (heuristic stays active) when there isn't enough
        labelled data yet — exactly the cold-start behaviour the block
        recogniser uses.
        """
        self._centroids = {}
        self._feat_mean = self._feat_std = None
        self._trained = False
        data = _load_samples(self._store_path)
        usable = {s: v for s, v in data.items()
                  if s in WeatherState.SKY_STATES
                  and len(v) >= self.cfg.min_samples_per_state}
        if len(usable) < 2:
            return
        # z-normalise across the pooled samples so no single large-range
        # feature dominates the Euclidean distance.
        pooled = np.array([self._vec(d) for v in usable.values() for d in v],
                          dtype=np.float32)
        self._feat_mean = pooled.mean(axis=0)
        self._feat_std = pooled.std(axis=0)
        self._feat_std[self._feat_std < 1e-6] = 1.0
        for state, samples in usable.items():
            mat = np.array([self._vec(d) for d in samples], dtype=np.float32)
            mat = (mat - self._feat_mean) / self._feat_std
            self._centroids[state] = mat.mean(axis=0)
        self._trained = True

    def is_trained(self) -> bool:
        return self._trained

    def stats(self) -> dict:
        data = _load_samples(self._store_path)
        return {
            "trained": self._trained,
            "states": {s: len(v) for s, v in data.items()},
            "store": str(self._store_path),
        }


def _cave_conf(sky_visible: float, cfg: WeatherDetectorConfig) -> float:
    # The further below the cave threshold, the more confident "unknown".
    if cfg.cave_open_frac <= 0:
        return 0.5
    return float(min(1.0, max(0.4, 1.0 - sky_visible / cfg.cave_open_frac)))


# ---------------------------------------------------------------------------
# Sample store (labelled feature vectors, JSON)
# ---------------------------------------------------------------------------

def _load_samples(path: Path) -> Dict[str, List[Dict[str, float]]]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, List[Dict[str, float]]] = {}
    for state, samples in raw.items():
        if state in WeatherState.ALL and isinstance(samples, list):
            out[state] = [s for s in samples if isinstance(s, dict)]
    return out


def append_samples(path: Path, state: str,
                   feature_dicts: List[Dict[str, float]]) -> int:
    """Append labelled feature samples for ``state`` and return the new
    total count for that state. Used by the collection tool."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _load_samples(path)
    bucket = data.setdefault(state, [])
    for f in feature_dicts:
        bucket.append({k: float(f.get(k, 0.0)) for k in FEATURE_KEYS})
    from utils.atomic import atomic_write_text
    atomic_write_text(path, json.dumps(data, indent=2, sort_keys=True))
    return len(bucket)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_weather_detector(settings: dict) -> WeatherDetector:
    wcfg = (((settings or {}).get("vision", {}) or {}).get("weather", {}) or {})
    cfg = WeatherDetectorConfig()
    for key in ("cave_open_frac", "weather_min_sky_open",
                "max_assess_pitch_deg",
                "clear_brightness_min", "rain_brightness_max",
                "thunder_brightness_max", "clear_blueness_min",
                "precip_blueness_max", "snow_whiteness_min",
                "precip_edge_min", "flip_majority"):
        if key in wcfg and wcfg[key] is not None:
            setattr(cfg, key, float(wcfg[key]))
    for ikey in ("smooth_window", "min_samples_per_state"):
        if ikey in wcfg and wcfg[ikey] is not None:
            setattr(cfg, ikey, int(wcfg[ikey]))
    return WeatherDetector(config=cfg)


__all__ = [
    "WeatherState",
    "WeatherDetector",
    "WeatherDetectorConfig",
    "FEATURE_KEYS",
    "build_weather_detector",
    "append_samples",
    "default_weather_store_path",
]

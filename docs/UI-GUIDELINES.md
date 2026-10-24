# GUI design rules

**These rules are permanent and apply to every AARMA screen, present and
future** — conversation, settings, voice, language, permissions, security,
memory, reminders, activity, persona, AI providers, first-run setup, about, and
anything the installer adds later.

AARMA is meant to be downloaded and installed by people whose machines look
nothing like the developer's. A layout that only works on the screen it was
built on is a defect, not a detail.

---

## The three variables

| Variable | Who handles it | What the code must do |
|---|---|---|
| **Screen size** | `ui.responsive.compute_window_placement` | Derive window size *from* the work area; never choose a size and clamp it |
| **Display scaling (DPI)** | Qt 6, plus `PassThrough` rounding | Write layouts in logical pixels; paint manual pixmaps at the device pixel ratio |
| **Font scaling** | `ui.responsive.Metrics` | Derive every font size and spacing value from the user's own font |

These are independent. A 1920×1080 screen at 150% scaling has a *logical* work
area of about 1280×672 — smaller than many 1366×768 laptops at 100%. Assuming
any one of the three tells you about the others produces clipped controls.

---

## Rules

### 1. Never hardcode a window size

```python
# Wrong — a size that fits one screen
self.resize(1200, 800)

# Wrong — still a hardcoded size, just clamped
self.resize(min(1200, screen.width() - 72), min(800, screen.height() - 72))

# Right — derived from the work area, bounded by intent
apply_initial_geometry(self, ideal=(1240, 820), minimum=(720, 460))
```

`ideal` is the size at which the layout breathes; `minimum` is where it stops
being usable. Neither is a target. The actual size comes from the screen, and
the window is centred within the work area with room for its own frame.

If the work area is smaller than `minimum`, the minimum yields — a window
larger than the screen puts its own controls out of reach.

### 2. Never hardcode a font size or a spacing value

```python
# Wrong — ignores the user's Windows text-size setting
label.setStyleSheet("font-size: 14px")
layout.setSpacing(16)

# Right
label.setObjectName("BubbleText")          # sized by the stylesheet
layout.setSpacing(metrics.sp(16))
```

In the stylesheet, font sizes are emitted in **points** via
`metrics.font_pt(n)`. A QSS `px` is a fixed logical pixel and ignores the text
size the user chose; `pt` follows it. At the 9pt / 96 DPI baseline the design
was drawn against, the two are identical.

### 3. Every page scrolls

Wrap page content in `ui.responsive.scroll_wrap`. Navigation, footers and
confirmation controls go **outside** the scroll area so they are always
reachable.

The alternative — letting content set the window's minimum height — is what
puts a Finish button under the taskbar.

### 4. Nothing may force the window's minimum size

A widget that cannot shrink becomes the whole window's floor. Watch for:

- long lists and tall sidebars → put them in a `scroll_wrap`
- `setFixedSize` / `setFixedHeight` on anything that is not intrinsically
  fixed (a circular avatar, a one-pixel divider)
- root layouts → call `layout.setSizeConstraint(QLayout.SetNoConstraint)`, or
  Qt writes the layout's minimum onto the window and overrides yours

Check with `widget.minimumSizeHint()` and by asserting a resize actually
applies. A resize that is silently clamped is the bug.

### 5. Adapt, don't clip

When space runs short, in order of preference:

1. **Reflow** — `ReflowGrid` drops from two columns to one; `flow_container`
   wraps a chip row onto a second line.
2. **Rescale** — the sidebar takes a share of the window; chat bubbles take a
   share of the viewport.
3. **Shed** — drop decorative chrome at a breakpoint (a tagline, a secondary
   status pill). Never drop anything the user needs to act.
4. **Scroll** — everything else.

Never shrink text to fit, and never remove information the user needs.

### 6. Breakpoints scale, and stay reachable

```python
threshold = breakpoint_threshold(design_width, MINIMUM_SIZE[0], metrics)
```

A breakpoint must grow with the user's font *and* stay above the smallest
window the screen supports — a breakpoint below the minimum width never fires,
which makes the adaptation dead code.

### 7. Measure, don't estimate

Deciding whether something fits means measuring it. `ElidingLabel.natural_width()`
uses `QFontMetrics` against the real font, so the answer holds at any text size.

If a value is used for both measuring and rendering, set it in one place. Letter
spacing is applied to the **font**, not the stylesheet, because QSS
`letter-spacing` is invisible to `QFontMetrics` — measuring one thing while
rendering another is how a label ends up elided after the caller calculated that
it would fit.

### 8. Windows behave like windows

Every top-level surface offers minimize, maximize, restore, resize and close.
Use a top-level `QWidget` or `QMainWindow`. **Do not use `QDialog`** for
anything the user may want to leave open — on Windows it is created without
minimize and maximize buttons.

### 9. Paint generated art at the device pixel ratio

```python
pixmap = QPixmap(int(size * dpr), int(size * dpr))
pixmap.setDevicePixelRatio(dpr)     # painting stays in logical coordinates
```

Allocating at logical size and letting Qt upscale is what makes an icon look
soft on a high-DPI display.

---

## Minimum supported dimensions

| Surface | Comfortable | Preferred minimum | Below that |
|---|---|---|---|
| Dashboard | 1240 × 820 logical | 720 × 460 logical | Minimum yields to the work area; content scrolls |
| Setup wizard | 980 × 680 logical | 620 × 420 logical | Same |

Because these are *logical* pixels, the physical requirement falls as scaling
rises. Verified working down to a **640 × 336 logical** work area, which is
1920×1080 at an effective 3.0× device pixel ratio.

There is no screen on which AARMA opens a window larger than the work area.

---

## Testing

`tests/test_ui_responsive.py` holds the contract in two layers:

**Pure geometry** — `compute_window_placement` is a function over a rectangle,
so all 30 combinations of the six supported resolutions and five scaling factors
are asserted directly. No monitor is required, and none is claimed.

**Live widgets** — the wizard and dashboard are built on Qt's offscreen platform
and driven through real resizes. The central assertion is that navigation stays
inside the window at the smallest supported size, on every page.

Simulate a scaling factor end to end without different hardware:

```bash
set QT_SCALE_FACTOR=1.5 && python main.py
```

When adding a screen, add it to the width sweep. A page that has never been
resized in a test is a page that will clip on someone's laptop.

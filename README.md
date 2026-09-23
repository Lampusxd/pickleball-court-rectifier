# Pickleball Court Rectifier

Turn an angled photograph of a pickleball court into a top-down view, and map
player positions from the photo into real court coordinates in feet.

Everything is manual and explicit: you click the four court corners, the code
solves one 3×3 homography with `cv2.getPerspectiveTransform`, and every later
click is pushed through that matrix with `cv2.perspectiveTransform`. There is
no detection, no training, and no video — the point is to understand the
geometry, not to hide it.

![The three stages: angled image with clicked corners, the rectified top-down warp, and player positions in feet on a court diagram](docs/images/workflow.png)

*Every figure in this README was produced by running the code — regenerate them
with `python docs/make_figures.py`. The image above is the distorted demo court,
so the reported positions carry real error.*

---

## 1. Quick start

Requires Python 3.9 or newer (developed and tested on Python 3.12.10, Windows 11,
OpenCV 4.14.0, NumPy 2.5.3).

### Windows (PowerShell)

```powershell
cd path\to\SkillsAssignment
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

> If `python` is not recognised, Python is not on your PATH. Either reinstall it
> with "Add python.exe to PATH" ticked, or call it by full path, e.g.
> `& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m venv .venv`.
> After the venv exists you never need the system Python again — just use
> `.\.venv\Scripts\python.exe`.

### macOS / Linux

```bash
cd path/to/SkillsAssignment
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Run it

```powershell
# 1. make the synthetic demo courts (10 images: 5 viewpoints, exact + distorted)
.\.venv\Scripts\python.exe generate_demo.py --all

# 2. check the whole pipeline with no windows, against known ground truth
.\.venv\Scripts\python.exe verify_demo.py

# 3. the same check on a lens-distorted image, where errors are REAL
.\.venv\Scripts\python.exe verify_demo.py --image demo\demo_angled_distorted.png

# 4. run the unit tests
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

# 5. the interactive tool
.\.venv\Scripts\python.exe main.py demo\demo_angled_distorted.png

# 6. reuse a saved calibration instead of clicking corners again
.\.venv\Scripts\python.exe main.py demo\demo_angled_distorted.png --load
```

No court photo of your own? You do not need one — see §5.

On macOS/Linux with the venv activated, replace `.\.venv\Scripts\python.exe`
with `python`.

Useful options:

| option | meaning |
| --- | --- |
| `--demo` | use `demo/demo_angled.png`, generating it first if missing |
| `--calibration PATH` | where to read/write the calibration JSON (default `output/calibration.json`) |
| `--load` | load that calibration at startup instead of clicking corners |
| `--out DIR` | output folder for CSVs and images (default `output`) |
| `--ppf 20` | pixels per foot in the rendered top-down views |
| `--max-display 1280x800` | largest window size; larger photos are shown scaled |

---

## 2. Controls

Click **in the image window**, not the terminal.

| input | action |
| --- | --- |
| left click | add a point for the current mode |
| `c` | corner mode — (re)select the four court corners |
| `p` | player mode — click players' ground-contact points |
| `v` | validation mode — click a known landmark, then name it in the terminal |
| `u` | undo the last point of the current mode |
| `r` | reset all points of the current mode |
| `2` | show/hide the rectified top-down window |
| `3` | show/hide the court diagram window |
| `s` | save everything to the output folder |
| `l` | reload the calibration file from disk |
| `h` | print the help block to the terminal |
| `q` or `Esc` | quit |

**Corner order: far-left, far-right, near-right, near-left.** "Far/near" and
"left/right" describe the court *as it appears in your photograph*, going
clockwise around the court on screen — they are not any official end of a real
court. If the camera is behind the near baseline, the far baseline is the one
near the top of the frame. Clicking the corners in the reverse order is
detected and refused, because it would mirror the court.

Validation mode is the one place the terminal is used for input: after each
click you pick the landmark from a numbered menu (or type `x,y` in feet). The
image window is unresponsive while it waits for that line of input.

---

## 3. How it works

### The math, in one page

A pickleball court is **flat**. Any flat surface seen by a pinhole camera is
related to its true top-down shape by a *homography* — a 3×3 matrix `H` acting
on **homogeneous coordinates**:

```
    [ x' ]       [ h00  h01  h02 ] [ u ]
    [ y' ]   =   [ h10  h11  h12 ] [ v ]           X = x' / w'
    [ w' ]       [ h20  h21  h22 ] [ 1 ]           Y = y' / w'
```

`(u, v)` is a pixel in the photo; `(X, Y)` is a position on the court in feet.
The third row is what makes this *perspective* rather than affine: the final
division by `w'` is why equal steps down the court are **not** equal steps in
pixels, and why the far baseline looks shorter than the near one.

`H` has 9 entries but only 8 degrees of freedom, because scaling the whole
matrix does not change `x'/w'` and `y'/w'`. Each point correspondence gives
2 equations, so **4 points are exactly enough** — that is why you click 4
corners and why `cv2.getPerspectiveTransform` takes exactly 4 pairs. (With more
than 4 points you would use `cv2.findHomography`, which least-squares fits them;
this project deliberately stays with the minimal, exactly-determined case.)

The four corners are paired with their known court coordinates:

| click order | label | court coordinate (ft) |
| --- | --- | --- |
| 1 | far-left | (0, 0) |
| 2 | far-right | (20, 0) |
| 3 | near-right | (20, 44) |
| 4 | near-left | (0, 44) |

Because `H` is fitted *through* those four points, they map to those coordinates
with ~1e-14 ft error no matter how badly you clicked. **That is not accuracy.**
It is the fit reproducing its own inputs. Real accuracy is measured at other
landmarks — see §5.

### The three coordinate systems

Most of the code's care goes into not confusing these:

1. **Image pixels** — the original photo's pixel grid. All clicks are converted
   back to this grid before any geometry happens.
2. **Court feet** — the physical answer. Far-left corner is `(0, 0)`, `x` grows
   toward the right sideline (0…20 ft), `y` grows toward the near baseline
   (0…44 ft). This is what gets reported and exported.
3. **Output pixels** — pixels of a rendered top-down image, which are just
   court feet × a pixels-per-foot scale (plus a margin, for the diagram).

Rendering resolution never touches a reported coordinate: `--ppf 8` and
`--ppf 40` produce different-sized images and identical feet. There is a unit
test for exactly that (`test_changing_ppf_does_not_change_court_coordinates`).

The window also scales: a 4000×3000 photo is shown at 1067×800, so a click at
display `(x, y)` is original pixel `(x / 0.267, y / 0.267)`. Skipping that
division is the classic silent bug — it shifts every position by a few feet and
still *looks* plausible. `DisplayScaler` owns the conversion and is tested both
ways.

### Why only ground contact points

The homography maps the **court plane**. A point at the player's feet lies on
that plane, so it maps correctly. A point on their head does not: the camera ray
through the head pierces the ground somewhere *behind* the player, so you get a
position that is systematically too far from the camera — often by several feet.
The same applies to an airborne ball. Always click the point between the feet.

### What each OpenCV call does here

| call | role |
| --- | --- |
| `cv2.getPerspectiveTransform(src, dst)` | solves the 3×3 `H` from exactly 4 point pairs; `src[i]` must correspond to `dst[i]`, hence the fixed click order |
| `cv2.perspectiveTransform(pts, H)` | maps *points*; does the homogeneous divide by `w'` for you; wants an `(N, 1, 2)` array |
| `cv2.warpPerspective(img, M, size)` | maps the *whole image*; `M` must go from source pixels to **output** pixels, so we compose `scale @ H` |
| `cv2.setMouseCallback(win, fn)` | delivers clicks in *window* coordinates — convert them yourself |
| `cv2.imshow` / `cv2.waitKey` | display and the keyboard loop; `waitKey` is what actually pumps window events |

---

## 4. Project layout

```
main.py             CLI, OpenCV windows, mouse and keyboard handling
perspective.py      all the geometry: homography, coordinates, calibration I/O, drawing
generate_demo.py    builds the synthetic courts: 5 viewpoints, optional known lens
                    distortion, and exact ground truth for every landmark
verify_demo.py      runs the entire workflow headlessly and checks it against that
                    ground truth; demands zero error on exact images, reports real
                    error on distorted ones
tests/test_geometry.py   54 unit tests (standard-library unittest, no extra packages)
docs/make_figures.py     regenerates the README images by running the project
docs/images/             the figures this README links to (committed)
requirements.txt
demo/               generated, not committed: demo_topdown.png,
                    demo_<view>[_distorted].png and a matching
                    *_ground_truth.json for each
output/             generated, not committed: per image, calibration.json,
                    players.csv, validation.csv, validation_summary.json,
                    rectified.png, court_diagram.png, annotated_original.png
```

Fresh clone? `pip install -r requirements.txt` then
`python generate_demo.py --all` recreates everything that is not committed.

`perspective.py` has no windows and no I/O beyond JSON, which is why the whole
pipeline can be verified on a machine with no display.

### What gets saved (`s`)

| file | contents |
| --- | --- |
| `calibration.json` | image dimensions, the 4 corner pixels, court dimensions, the 3×3 homography, corner order, timestamp |
| `players.csv` | label, image pixels, court feet, out-of-court flag |
| `validation.csv` | landmark, image pixels, measured feet, true feet, error in feet |
| `validation_summary.json` | n, mean, RMSE, max, median error |
| `rectified.png` | `warpPerspective` output, 400×880 px at 20 px/ft (true 20:44) |
| `court_diagram.png` | the clean diagram with player markers |
| `annotated_original.png` | the photo with the overlay, for your report |

![Close-up of the photo overlay: green kitchen lines and centerlines drawn back from court feet sit slightly off the painted white lines, each validation landmark labelled with its error in feet](docs/images/reprojection.png)

Reloading a calibration (`--load` or `l`) checks the image dimensions match.
**That check is a guard rail, not a guarantee.** Two photos of the same size
taken from different camera positions both pass it, and only one of them is
actually calibrated. The camera position, orientation, zoom and crop must all
match. Visually confirm it: when a calibration is loaded, the tool draws the
kitchen lines, centerlines and net back onto the photo in green — if those
green lines do not sit on the painted lines, the calibration does not belong to
that photo.

Out-of-court positions are **reported, never clamped**: a click outside the
lines gives a coordinate like `(23.98, 46.99)` flagged `OUT`. On the diagram, a
marker too far out to fit on the canvas is drawn as a triangle at the edge and
labelled `OFF-DIAGRAM` — only the *drawing* is pulled in, the printed and
exported coordinate stays true.

---

## 5. Measuring accuracy (validation mode)

Press `v`, click a court landmark you can identify precisely, then choose what
it is from the terminal menu. The tool reports the Euclidean error in feet for
each landmark, plus mean and RMSE, and writes them to `validation.csv`.

Use **at least five** landmarks that are *not* the four calibration corners —
the built-in menu offers ten: kitchen-line/sideline intersections, kitchen-line
/centerline intersections, baseline/centerline intersections, and net-line
/sideline points. The four corners are excluded by construction: they fit
exactly by definition, so quoting them as evidence of accuracy would be
circular.

A useful habit: report mean **and** RMSE. RMSE is never smaller than the mean
and grows faster when one landmark is badly off, so a large gap between them
tells you a single click was bad rather than the whole calibration being poor.

### No court photo? Use the distorted synthetic images

`generate_demo.py` can produce a court that behaves like a real camera saw it.
Three levels, each stronger evidence than the last:

| level | command | what you learn |
| --- | --- | --- |
| **exact** | `generate_demo.py` | proves the code is right; errors ~1e-14 ft, which is arithmetic, not accuracy |
| **distorted** | `generate_demo.py --distortion` | a *known* radial lens distortion bends the court lines. One homography cannot fit a curved image, so validation errors become real — tenths of a foot |
| **distorted + shaky clicks** | `verify_demo.py --click-noise 3` | adds Gaussian noise to the corner clicks, simulating an imprecise hand |

```powershell
.\.venv\Scripts\python.exe generate_demo.py --all           # 5 views x exact/distorted
.\.venv\Scripts\python.exe verify_demo.py --image demo\demo_angled_distorted.png
.\.venv\Scripts\python.exe verify_demo.py --click-noise 3   # exact image, sloppy clicks
.\.venv\Scripts\python.exe main.py demo\demo_angled_distorted.png   # then press v
```

![The same court twice: the left image has perfectly straight lines, the right one is visibly bowed by a known lens](docs/images/distortion_comparison.png)

Views: `default`, `low` (camera low behind the baseline), `high` (on a pole),
`corner` (off to one side), `oblique` (a hard angle). `--distortion K1` takes a
coefficient: positive bends lines like a pincushion, negative like the barrel
distortion of a phone ultra-wide. Strong negative values fold the image
mathematically, and the script says so rather than inventing ground truth.

`verify_demo.py` switches mode by itself: on an exact image it *demands*
~zero error and fails otherwise; on a distorted image or with click noise it
reports mean/RMSE instead, while still enforcing the structural checks
(round trips, save/reload, rejection of bad input).

The most useful thing in that output is what happens to the corner self-check.
With `--click-noise 3`, every corner click is about 3 px wrong, and the four
corners *still* map to (0,0), (20,0), (20,44), (0,44) to within 1e-6 ft — while
independent landmarks are off by a quarter of a foot. A fit always reproduces
its own inputs. That is the whole reason validation mode exists.



No accuracy figure for a real photograph appears in this repository unless you
measured it and wrote it in.

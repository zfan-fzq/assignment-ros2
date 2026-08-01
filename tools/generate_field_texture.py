from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
OUT = (
    ROOT
    / "models"
    / "robocup_2009_spl_field"
    / "materials"
    / "textures"
    / "robocup_field_hd.png"
)

SCALE = 500
WIDTH = 6.0
HEIGHT = 4.0
PX_W = int(WIDTH * SCALE)
PX_H = int(HEIGHT * SCALE)


def xy(x_m: float, y_m: float) -> tuple[int, int]:
    return (round((x_m + WIDTH / 2.0) * SCALE), round((HEIGHT / 2.0 - y_m) * SCALE))


def box_from_center(x_m: float, y_m: float, w_m: float, h_m: float) -> tuple[int, int, int, int]:
    cx, cy = xy(x_m, y_m)
    half_w = round(w_m * SCALE / 2.0)
    half_h = round(h_m * SCALE / 2.0)
    return (cx - half_w, cy - half_h, cx + half_w, cy + half_h)


def draw_line(draw: ImageDraw.ImageDraw, points: list[tuple[float, float]], width_m: float) -> None:
    draw.line([xy(x, y) for x, y in points], fill=(245, 245, 245), width=round(width_m * SCALE), joint="curve")


def draw_ring(draw: ImageDraw.ImageDraw, center: tuple[float, float], radius_m: float, width_m: float) -> None:
    cx, cy = xy(*center)
    r = round(radius_m * SCALE)
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(245, 245, 245), width=round(width_m * SCALE))


def main() -> None:
    img = Image.new("RGB", (PX_W, PX_H), (74, 178, 65))
    draw = ImageDraw.Draw(img)

    # Subtle turf bands add detail without confusing the simple white-line detector.
    band_h = PX_H // 10
    for i in range(10):
        color = (70, 170, 62) if i % 2 else (82, 188, 72)
        draw.rectangle((0, i * band_h, PX_W, (i + 1) * band_h), fill=color)

    border = round(0.08 * SCALE)
    draw.rectangle((border, border, PX_W - border, PX_H - border), outline=(40, 140, 42), width=border)

    line_w = 0.055
    x_left, x_right = -2.8, 2.8
    y_bottom, y_top = -1.8, 1.8

    draw_line(draw, [(x_left, y_bottom), (x_left, y_top)], line_w)
    draw_line(draw, [(x_right, y_bottom), (x_right, y_top)], line_w)
    draw_line(draw, [(x_left, y_top), (x_right, y_top)], line_w)
    draw_line(draw, [(x_left, y_bottom), (x_right, y_bottom)], line_w)
    draw_line(draw, [(0.0, y_bottom), (0.0, y_top)], line_w)

    draw_ring(draw, (0.0, 0.0), 0.43, line_w)
    draw.ellipse(box_from_center(0.0, 0.0, 0.13, 0.13), fill=(245, 245, 245))

    # Goal / penalty boxes.
    for sign in (-1.0, 1.0):
        front_x = sign * 2.15
        edge_x = sign * 2.8
        draw_line(draw, [(front_x, -0.6), (front_x, 0.6)], line_w)
        draw_line(draw, [(edge_x, -0.6), (front_x, -0.6)], line_w)
        draw_line(draw, [(edge_x, 0.6), (front_x, 0.6)], line_w)

        dot_x = sign * 1.55
        draw.ellipse(box_from_center(dot_x, 0.0, 0.10, 0.10), fill=(245, 245, 245))

    # Colored goal hints in the texture; the 3D posts remain above this surface.
    draw.rectangle(box_from_center(-2.98, 0.0, 0.10, 1.02), fill=(245, 216, 32))
    draw.rectangle(box_from_center(2.98, 0.0, 0.10, 1.02), fill=(40, 106, 235))

    # Small antialiasing pass for a clean top-down recording.
    img = img.resize((PX_W // 2, PX_H // 2), Image.Resampling.LANCZOS)
    img = img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=115, threshold=2))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()

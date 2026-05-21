import subprocess
import threading
import queue
import sys
import os
import math

try:
    import pygame
except ImportError:
    print("pygame nu este instalat. Instaleaza cu: pip install pygame")
    sys.exit(1)

# ── Configurare ──────────────────────────────────────────────────────────────
N_VIS        = 10000
STEPS        = 1000
VIS_INTERVAL = 1

WINDOW_W = 1530
WINDOW_H = 780
PANEL_W  = WINDOW_W // 3
PANEL_H  = WINDOW_H - 140
INFO_H   = 140

# Camera / zoom.
# Larger radius = zoomed OUT. Smaller radius = zoomed IN.
DEFAULT_VIEW_RADIUS = 1.2
MIN_VIEW_RADIUS     = 0.25
MAX_VIEW_RADIUS     = 200.0
ZOOM_FACTOR         = 1.25

BG_COLOR        = (8,  10, 20)
PANEL_BG        = (12, 14, 28)
BORDER_COLOR    = (40, 45, 70)
TEXT_COLOR      = (200, 205, 220)
SUBTEXT_COLOR   = (120, 125, 145)
DONE_COLOR      = (80, 220, 120)
ERROR_COLOR     = (220, 80, 80)
BAR_BG_COLOR    = (30, 32, 55)

PROGRAM_CONFIGS = [
    {
        "cmd":   ["./nbody", str(N_VIS), str(STEPS), str(VIS_INTERVAL)],
        "label": "Secvential",
        "color": (100, 180, 255),
        "dot_color": (80, 160, 240),
    },
    {
        "cmd":   ["./nbody-omp", str(N_VIS), str(STEPS), str(VIS_INTERVAL)],
        "label": "OpenMP  (16 threads)",
        "color": (100, 230, 140),
        "dot_color": (80, 210, 120),
    },
    {
        "cmd":   ["./nbody-cuda", str(N_VIS), str(STEPS), str(VIS_INTERVAL)],
        "label": "CUDA",
        "color": (255, 170, 70),
        "dot_color": (240, 150, 50),
    },
]
# ─────────────────────────────────────────────────────────────────────────────


class SimState:
    def __init__(self, cfg):
        self.label     = cfg["label"]
        self.color     = cfg["color"]
        self.dot_color = cfg["dot_color"]
        self.step      = 0
        self.elapsed   = 0.0
        self.xs        = []
        self.ys        = []
        self.done      = False
        self.error     = None
        self.available = True


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def zoom_in(view_radius):
    # Smaller radius means the world rectangle is smaller, so objects look bigger.
    return clamp(view_radius / ZOOM_FACTOR, MIN_VIEW_RADIUS, MAX_VIEW_RADIUS)


def zoom_out(view_radius):
    # Larger radius means the world rectangle is larger, so objects look smaller.
    return clamp(view_radius * ZOOM_FACTOR, MIN_VIEW_RADIUS, MAX_VIEW_RADIUS)


def bounds_from_radius(view_radius):
    return (-view_radius, view_radius, -view_radius, view_radius)


def reader_thread(proc, q, idx):
    try:
        for raw in proc.stdout:
            line = raw.strip()
            if not line.startswith("STEP "):
                continue

            parts = line.split()
            if len(parts) < 4:
                continue

            step = int(parts[1])
            elapsed = float(parts[2])
            coords = list(map(float, parts[3:]))

            xs = []
            ys = []

            for k in range(0, len(coords) - 2, 3):
                x = coords[k]
                y = coords[k + 1]

                if math.isfinite(x) and math.isfinite(y):
                    xs.append(x)
                    ys.append(y)

            q.put((idx, step, elapsed, xs, ys))

        proc.wait()
        q.put((idx, -1, 0.0, [], []))

    except Exception as e:
        q.put((idx, -2, 0.0, [], [str(e)]))


def world_to_screen(x, y, panel_x, xmin, xmax, ymin, ymax):
    rx = (x - xmin) / (xmax - xmin) if xmax != xmin else 0.5
    ry = (y - ymin) / (ymax - ymin) if ymax != ymin else 0.5

    sx = panel_x + int(rx * PANEL_W)
    sy = int((1.0 - ry) * PANEL_H)

    return sx, sy


def draw_panel(screen, state, panel_x, fonts, view_radius):
    font_title, font_body = fonts

    pygame.draw.rect(screen, PANEL_BG, (panel_x, 0, PANEL_W, PANEL_H))
    pygame.draw.line(screen, BORDER_COLOR, (panel_x, 0), (panel_x, PANEL_H))

    title_surf = font_title.render(state.label, True, state.color)
    screen.blit(title_surf, (panel_x + 12, 10))

    zoom_surf = font_body.render(f"Raza view: {view_radius:.2f}", True, SUBTEXT_COLOR)
    screen.blit(zoom_surf, (panel_x + 12, 31))

    if not state.available:
        msg = font_body.render(state.error or "Indisponibil", True, ERROR_COLOR)
        screen.blit(msg, (panel_x + 12, PANEL_H // 2 - 10))
        return

    if state.done:
        done_surf = font_body.render("COMPLET", True, DONE_COLOR)
        screen.blit(done_surf, (panel_x + PANEL_W - 90, 12))

    if not state.xs:
        wait_surf = font_body.render("Se asteapta date...", True, SUBTEXT_COLOR)
        screen.blit(wait_surf, (panel_x + 12, PANEL_H // 2))
        return

    xmin, xmax, ymin, ymax = bounds_from_radius(view_radius)

    for x, y in zip(state.xs, state.ys):
        sx, sy = world_to_screen(x, y, panel_x, xmin, xmax, ymin, ymax)

        if panel_x <= sx < panel_x + PANEL_W and 50 <= sy < PANEL_H - 2:
            pygame.draw.circle(screen, state.dot_color, (sx, sy), 1)


def draw_info_bar(screen, states, fonts, view_radius):
    font_title, font_body = fonts

    info_y = PANEL_H

    pygame.draw.rect(screen, BG_COLOR, (0, info_y, WINDOW_W, INFO_H))
    pygame.draw.line(screen, BORDER_COLOR, (0, info_y), (WINDOW_W, info_y))

    for i, state in enumerate(states):
        panel_x = i * PANEL_W

        pygame.draw.line(
            screen,
            BORDER_COLOR,
            (panel_x, info_y),
            (panel_x, WINDOW_H)
        )

        if not state.available:
            msg = font_body.render(state.error or "Indisponibil", True, ERROR_COLOR)
            screen.blit(msg, (panel_x + 12, info_y + 55))
            continue

        display_step = STEPS if state.done else state.step

        step_surf = font_body.render(
            f"Pas: {display_step} / {STEPS}",
            True,
            TEXT_COLOR
        )

        time_surf = font_body.render(
            f"Timp: {state.elapsed:.3f} s",
            True,
            TEXT_COLOR
        )

        screen.blit(step_surf, (panel_x + 12, info_y + 10))
        screen.blit(time_surf, (panel_x + 12, info_y + 32))

        bar_x = panel_x + 12
        bar_y = info_y + 62
        bar_w = PANEL_W - 24

        if state.done:
            progress = 1.0
        elif STEPS > 0:
            progress = min(state.step / STEPS, 1.0)
        else:
            progress = 0.0

        pygame.draw.rect(
            screen,
            BAR_BG_COLOR,
            (bar_x, bar_y, bar_w, 16),
            border_radius=4
        )

        if progress > 0:
            pygame.draw.rect(
                screen,
                state.color,
                (bar_x, bar_y, int(bar_w * progress), 16),
                border_radius=4
            )

        if state.elapsed > 0.05 and state.step > 0:
            sps = state.step / state.elapsed
            sps_surf = font_body.render(f"{sps:.1f} pasi/s", True, state.color)
            screen.blit(sps_surf, (bar_x, info_y + 86))

    help_text = (
        f"Zoom: raza view {view_radius:.2f} | "
        "+ / ↑ / wheel up = zoom in | "
        "- / ↓ / wheel down = zoom out | "
        "0 = reset"
    )

    help_surf = font_body.render(help_text, True, SUBTEXT_COLOR)
    screen.blit(help_surf, (12, WINDOW_H - 22))


def draw_button(screen, rect, text, font, base_color, hovered):
    if hovered:
        color = tuple(min(255, c + 40) for c in base_color)
    else:
        color = base_color

    pygame.draw.rect(screen, color, rect, border_radius=7)
    pygame.draw.rect(screen, (180, 185, 210), rect, 1, border_radius=7)

    surf = font.render(text, True, (240, 245, 255))

    screen.blit(
        surf,
        (
            rect.centerx - surf.get_width() // 2,
            rect.centery - surf.get_height() // 2
        )
    )


def draw_splash(screen, btn_start, mouse_pos, fonts):
    font_title, font_body = fonts

    screen.fill(BG_COLOR)

    title_y = WINDOW_H // 2 - 135
    info_y = title_y + 62
    line_gap = 28

    title = font_title.render(
        "N-body Simulation – Comparatie Paralela",
        True,
        (160, 180, 230)
    )

    screen.blit(
        title,
        (
            WINDOW_W // 2 - title.get_width() // 2,
            title_y
        )
    )

    info_lines = [
        f"Corpuri: {N_VIS}   |   Pasi: {STEPS}   |   Output la fiecare {VIS_INTERVAL} pasi",
        "  |  ".join(cfg["label"] for cfg in PROGRAM_CONFIGS),
    ]

    for k, line in enumerate(info_lines):
        s = font_body.render(line, True, SUBTEXT_COLOR)

        screen.blit(
            s,
            (
                WINDOW_W // 2 - s.get_width() // 2,
                info_y + k * line_gap
            )
        )

    draw_button(
        screen,
        btn_start,
        "START",
        font_title,
        (40, 90, 160),
        btn_start.collidepoint(mouse_pos)
    )

    controls = "Dupa START: + / - / mouse wheel pentru zoom, 0 pentru reset."
    controls_surf = font_body.render(controls, True, SUBTEXT_COLOR)

    screen.blit(
        controls_surf,
        (
            WINDOW_W // 2 - controls_surf.get_width() // 2,
            btn_start.bottom + 22
        )
    )


def launch_all(processes_ref, states_ref, queues_ref):
    for proc in processes_ref:
        if proc and proc.poll() is None:
            proc.terminate()

    processes_ref.clear()
    states_ref.clear()
    queues_ref.clear()

    for i, cfg in enumerate(PROGRAM_CONFIGS):
        states_ref.append(SimState(cfg))
        queues_ref.append(queue.Queue())

        try:
            proc = subprocess.Popen(
                cfg["cmd"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )

            processes_ref.append(proc)

            t = threading.Thread(
                target=reader_thread,
                args=(proc, queues_ref[i], i),
                daemon=True
            )

            t.start()

        except FileNotFoundError:
            states_ref[i].available = False
            states_ref[i].error = f"'{cfg['cmd'][0]}' negasit"
            processes_ref.append(None)


def handle_zoom_key(event, current_radius):
    key = event.key
    char = event.unicode if hasattr(event, "unicode") else ""

    if char in ("+", "=") or key in (
        pygame.K_PLUS,
        pygame.K_EQUALS,
        pygame.K_KP_PLUS,
        pygame.K_UP,
        pygame.K_PAGEUP,
    ):
        return zoom_in(current_radius)

    if char in ("-", "_") or key in (
        pygame.K_MINUS,
        pygame.K_KP_MINUS,
        pygame.K_DOWN,
        pygame.K_PAGEDOWN,
    ):
        return zoom_out(current_radius)

    if char == "0" or key in (
        pygame.K_0,
        pygame.K_KP0,
        pygame.K_HOME,
    ):
        return DEFAULT_VIEW_RADIUS

    return current_radius


def run():
    pygame.init()

    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("N-body – comparatie secvential / OpenMP / CUDA")

    clock = pygame.time.Clock()

    font_title = pygame.font.SysFont("monospace", 17, bold=True)
    font_body  = pygame.font.SysFont("monospace", 14)

    fonts = (font_title, font_body)

    btn_start = pygame.Rect(
        WINDOW_W // 2 - 90,
        WINDOW_H // 2 + 70,
        180,
        48
    )

    btn_reset = pygame.Rect(
        WINDOW_W - 130,
        PANEL_H + 52,
        110,
        34
    )

    processes = []
    states = []
    data_queues = []

    started = False
    view_radius = DEFAULT_VIEW_RADIUS

    running = True

    while running:
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                else:
                    view_radius = handle_zoom_key(event, view_radius)

            elif event.type == pygame.MOUSEWHEEL:
                if event.y > 0:
                    view_radius = zoom_in(view_radius)
                elif event.y < 0:
                    view_radius = zoom_out(view_radius)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Older Pygame/Linux backends can report wheel as mouse buttons 4/5.
                if event.button == 4:
                    view_radius = zoom_in(view_radius)

                elif event.button == 5:
                    view_radius = zoom_out(view_radius)

                elif event.button == 1:
                    if not started and btn_start.collidepoint(mouse_pos):
                        started = True
                        launch_all(processes, states, data_queues)

                    elif started and btn_reset.collidepoint(mouse_pos):
                        launch_all(processes, states, data_queues)

        if not started:
            draw_splash(screen, btn_start, mouse_pos, fonts)
            pygame.display.flip()
            clock.tick(60)
            continue

        for i, q in enumerate(data_queues):
            while True:
                try:
                    idx, step, elapsed, xs, ys = q.get_nowait()

                    if step == -1:
                        states[i].done = True

                    elif step == -2:
                        states[i].done = True
                        states[i].error = ys[0] if ys else "Eroare necunoscuta"

                    else:
                        states[i].step = step
                        states[i].elapsed = elapsed
                        states[i].xs = xs
                        states[i].ys = ys

                except queue.Empty:
                    break

        screen.fill(BG_COLOR)

        for i, state in enumerate(states):
            draw_panel(
                screen,
                state,
                i * PANEL_W,
                fonts,
                view_radius
            )

        draw_info_bar(screen, states, fonts, view_radius)

        draw_button(
            screen,
            btn_reset,
            "Reset",
            font_body,
            (55, 65, 110),
            btn_reset.collidepoint(mouse_pos)
        )

        pygame.display.flip()
        clock.tick(60)

    for proc in processes:
        if proc and proc.poll() is None:
            proc.terminate()

    pygame.quit()


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    run()
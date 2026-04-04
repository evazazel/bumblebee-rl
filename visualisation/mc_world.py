"""
mc_world.py  –  2D Minecraft-style visualisation of bumblebee RL training
==========================================================================
4 panels train simultaneously from scratch, one per condition:
    • High Variance + Social Cue
    • High Variance + Non-Social Cue
    • No Variance   + Social Cue
    • No Variance   + Non-Social Cue

Controls:
    SPACE       pause / resume
    UP arrow    speed up
    DOWN arrow  slow down
    R           reset all agents
    Q / ESC     quit

Usage:
    python visualisation/mc_world.py
"""

import pygame
import sys
import os
import math
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bee_env.flower_world import FlowerWorldEnv
from agents.q_agent import QLearningAgent


# ── Window & layout ──────────────────────────────────────────────────────────
WIN_W, WIN_H = 1280, 800
FPS          = 60
PAD          = 10
HEADER_H     = 44

PANEL_W = (WIN_W - PAD * 3) // 2
PANEL_H = (WIN_H - HEADER_H - PAD * 3) // 2

PANEL_POS = [
    (PAD,               HEADER_H + PAD),
    (PAD * 2 + PANEL_W, HEADER_H + PAD),
    (PAD,               HEADER_H + PAD * 2 + PANEL_H),
    (PAD * 2 + PANEL_W, HEADER_H + PAD * 2 + PANEL_H),
]

# ── Flower grid (3 rows × 4 cols = 12 flowers) ───────────────────────────────
COLS, ROWS = 4, 3
N_FLOWERS  = COLS * ROWS
FSIZE      = 54          # flower block size (px)
FGAP       = 8
GRID_W     = COLS * FSIZE + (COLS - 1) * FGAP
GRID_H     = ROWS * FSIZE + (ROWS - 1) * FGAP
GRID_X     = (PANEL_W - GRID_W) // 2      # centred horizontally in panel
GRID_Y     = 44                            # below the panel title

# ── Animation speed levels (frames per bee flight) ───────────────────────────
SPEED_LEVELS  = [60, 35, 20, 10, 5, 2]
DEFAULT_SPEED = 2          # index into SPEED_LEVELS (20 frames = medium)
PAUSE_FRAMES  = 6          # frames bee sits at flower before next step

# ── Colour palette (Minecraft-ish) ───────────────────────────────────────────
C = dict(
    bg            = (15,  20,  15 ),
    panel_bg      = (30,  45,  30 ),
    panel_border  = (55,  80,  55 ),
    title_bar     = (20,  30,  20 ),

    flower_idle   = (70,  110, 70 ),    # uncued, no reward
    flower_cued   = (50,  90,  155),    # has a social cue (blue tint)
    flower_cued2  = (120, 80,  40 ),    # has a non-social cue (brown tint)
    flower_rich   = (160, 140, 30 ),    # rich flower (gold tint)
    flower_flash  = (255, 230, 60 ),    # reward collected flash

    stem          = (50,  110, 50 ),
    petal         = (255, 220, 50 ),
    petal_empty   = (100, 100, 100),    # empty flower – grey petal

    bee_body      = (255, 210, 0  ),
    bee_stripe    = (20,  20,  20 ),
    bee_wing      = (200, 230, 255),

    banner_social    = (220, 55,  55 ),  # red flag = social cue
    banner_nonsocial = (60,  180, 80 ),  # green square = non-social cue

    text          = (225, 225, 225),
    subtext       = (140, 165, 140),
    chart_line    = (90,  200, 90 ),
    chart_bg      = (20,  30,  20 ),
    chance_line   = (180, 70,  70 ),
    highlight     = (255, 220, 60 ),
)

# Accent colours per condition (matches analysis notebook colours)
ACCENTS = [
    (33,  150, 243),   # HV + social     – blue
    (255, 152, 0  ),   # HV + non-social – orange
    (76,  175, 80 ),   # NV + social     – green
    (156, 39,  176),   # NV + non-social – purple
]

CONDITIONS = [
    ("high", "social",     "High Variance + Social Cue"),
    ("high", "non_social", "High Variance + Non-Social Cue"),
    ("no",   "social",     "No Variance  + Social Cue"),
    ("no",   "non_social", "No Variance  + Non-Social Cue"),
]


# ── BeePanel class ────────────────────────────────────────────────────────────
class BeePanel:
    """One condition: owns its own environment, agent, and drawing state."""

    READY     = 0
    ANIMATING = 1
    PAUSED    = 2

    def __init__(self, variance, cue_type, label, accent, px, py):
        self.env      = FlowerWorldEnv(variance_condition=variance, cue_type=cue_type)
        self.agent    = QLearningAgent()
        self.label    = label
        self.accent   = accent
        self.px, self.py = px, py
        self.cue_type = cue_type
        self.variance = variance

        # RL tracking
        self.episode      = 0
        self.ep_reward    = 0.0
        self.ep_cue       = 0
        self.ep_visits    = 0
        self.last_reward  = 0.0
        self.cue_rate_history = []
        self.reward_history   = []

        # Flash effect: flower_idx → frames remaining
        self.flash = {}

        # Bee animation state machine
        self.state       = self.READY
        self.bee_x       = 0.0
        self.bee_y       = 0.0
        self.src_x       = 0.0
        self.src_y       = 0.0
        self.dst_x       = 0.0
        self.dst_y       = 0.0
        self.anim_t      = 0.0
        self.anim_frames = SPEED_LEVELS[DEFAULT_SPEED]
        self.pause_count = 0

        # Start first episode
        self.obs, _ = self.env.reset()
        cx, cy = self.flower_center(np.random.randint(N_FLOWERS))
        self.bee_x, self.bee_y = float(cx), float(cy)

    # ── Geometry ──────────────────────────────────────────────────────────────
    def flower_center(self, idx):
        """Pixel centre of flower block idx within the window."""
        row = idx // COLS
        col = idx % COLS
        x = self.px + GRID_X + col * (FSIZE + FGAP) + FSIZE // 2
        y = self.py + GRID_Y + row * (FSIZE + FGAP) + FSIZE // 2
        return x, y

    # ── Update logic ──────────────────────────────────────────────────────────
    def update(self):
        if self.state == self.READY:
            self._take_step()

        elif self.state == self.ANIMATING:
            self.anim_t += 1.0 / max(self.anim_frames, 1)
            if self.anim_t >= 1.0:
                self.anim_t  = 1.0
                self.bee_x   = self.dst_x
                self.bee_y   = self.dst_y
                self.state   = self.PAUSED
                self.pause_count = PAUSE_FRAMES
            else:
                # smooth ease-in-out cubic
                t = self.anim_t
                t = t * t * (3.0 - 2.0 * t)
                self.bee_x = self.src_x + t * (self.dst_x - self.src_x)
                self.bee_y = self.src_y + t * (self.dst_y - self.src_y)

        elif self.state == self.PAUSED:
            self.pause_count -= 1
            if self.pause_count <= 0:
                self.state = self.READY

        # Decay flash effects
        for k in list(self.flash.keys()):
            self.flash[k] -= 1
            if self.flash[k] <= 0:
                del self.flash[k]

    def _take_step(self):
        """One RL step: select action, step env, update Q-table."""
        action = self.agent.select_action(self.obs)
        next_obs, reward, terminated, _, info = self.env.step(action)
        self.agent.update(self.obs, action, reward, next_obs, terminated)
        self.obs = next_obs

        self.ep_reward += reward
        self.ep_visits += 1
        self.last_reward = reward
        if info["had_cue"]:
            self.ep_cue += 1

        if reward > 0:
            self.flash[info["flower"]] = 20

        # Start bee animation to visited flower
        fx, fy = self.flower_center(info["flower"])
        self.src_x, self.src_y = self.bee_x, self.bee_y
        self.dst_x, self.dst_y = float(fx), float(fy)
        self.anim_t = 0.0
        self.state  = self.ANIMATING

        if terminated:
            self.agent.decay_epsilon()
            rate = self.ep_cue / self.ep_visits if self.ep_visits > 0 else 0.0
            self.cue_rate_history.append(rate)
            self.reward_history.append(self.ep_reward)
            self.ep_reward = self.ep_cue = self.ep_visits = 0
            self.episode += 1
            self.obs, _ = self.env.reset()

    # ── Drawing ───────────────────────────────────────────────────────────────
    def draw(self, surf, fonts):
        px, py = self.px, self.py

        # Panel background
        panel_rect = pygame.Rect(px, py, PANEL_W, PANEL_H)
        pygame.draw.rect(surf, C["panel_bg"], panel_rect, border_radius=6)
        pygame.draw.rect(surf, self.accent, panel_rect, width=2, border_radius=6)

        # Title bar
        title_rect = pygame.Rect(px, py, PANEL_W, 34)
        pygame.draw.rect(surf, self.accent, title_rect, border_radius=6)
        title_surf = fonts["sm"].render(self.label, True, (255, 255, 255))
        surf.blit(title_surf, (px + 8, py + 8))

        # Episode & epsilon badge (top right of title)
        ep_txt = f"Ep {self.episode:,}  ε={self.agent.epsilon:.2f}"
        ep_surf = fonts["xs"].render(ep_txt, True, (220, 220, 220))
        surf.blit(ep_surf, (px + PANEL_W - ep_surf.get_width() - 8, py + 10))

        # Flower grid
        self._draw_flowers(surf, fonts)

        # Bee
        self._draw_bee(surf, int(self.bee_x), int(self.bee_y))

        # Stats bar
        self._draw_stats(surf, fonts)

        # Mini chart
        self._draw_chart(surf, fonts)

    def _draw_flowers(self, surf, fonts):
        for idx in range(N_FLOWERS):
            row = idx // COLS
            col = idx % COLS
            bx = self.px + GRID_X + col * (FSIZE + FGAP)
            by = self.py + GRID_Y + row * (FSIZE + FGAP)
            rect = pygame.Rect(bx, by, FSIZE, FSIZE)

            # Choose block colour
            is_cued = bool(self.env.cued_flowers[idx]) if self.env.cued_flowers is not None else False
            is_rich = self.env.flower_rewards[idx] > 0 if self.env.flower_rewards is not None else False
            is_flash = idx in self.flash

            if is_flash:
                block_col = C["flower_flash"]
            elif is_rich and is_cued:
                block_col = C["flower_rich"]
            elif is_cued and self.cue_type == "social":
                block_col = C["flower_cued"]
            elif is_cued and self.cue_type == "non_social":
                block_col = C["flower_cued2"]
            else:
                block_col = C["flower_idle"]

            # Draw block (3D-ish: lighter top, darker side)
            pygame.draw.rect(surf, block_col, rect, border_radius=4)
            light = tuple(min(255, c + 40) for c in block_col)
            dark  = tuple(max(0,   c - 40) for c in block_col)
            pygame.draw.line(surf, light, (bx+2, by+2),     (bx+FSIZE-2, by+2),    2)
            pygame.draw.line(surf, light, (bx+2, by+2),     (bx+2, by+FSIZE-2),   2)
            pygame.draw.line(surf, dark,  (bx+2, by+FSIZE-2),(bx+FSIZE-2,by+FSIZE-2),2)
            pygame.draw.line(surf, dark,  (bx+FSIZE-2,by+2), (bx+FSIZE-2,by+FSIZE-2),2)

            # Stem
            cx = bx + FSIZE // 2
            pygame.draw.line(surf, C["stem"], (cx, by + FSIZE - 4), (cx, by + FSIZE // 2), 2)

            # Petal (flower head) — grey if empty, yellow if rich
            petal_col = C["petal"] if is_rich else C["petal_empty"]
            pygame.draw.circle(surf, petal_col, (cx, by + FSIZE // 2 - 2), 7)
            pygame.draw.circle(surf, (30, 30, 30), (cx, by + FSIZE // 2 - 2), 3)

            # Cue marker on top of block
            if is_cued:
                self._draw_cue(surf, bx, by)

    def _draw_cue(self, surf, bx, by):
        """Draw a small flag (social) or foam square (non-social) as cue marker."""
        if self.cue_type == "social":
            # Red triangular flag on a pole
            pole_x = bx + FSIZE - 10
            pygame.draw.line(surf, (180, 180, 180),
                             (pole_x, by + 4), (pole_x, by + 20), 2)
            pts = [(pole_x, by + 4), (pole_x, by + 13), (pole_x - 9, by + 8)]
            pygame.draw.polygon(surf, C["banner_social"], pts)
        else:
            # Green foam square (matches paper's non-social cue description)
            sq = pygame.Rect(bx + FSIZE - 14, by + 4, 11, 7)
            pygame.draw.rect(surf, C["banner_nonsocial"], sq, border_radius=2)

    def _draw_bee(self, surf, bx, by):
        """Draw an animated pixel-art bee."""
        # Wing flap based on time
        t = pygame.time.get_ticks() / 120.0
        wing_off = int(math.sin(t) * 3)

        # Wings (semi-transparent ovals approximated as ellipses)
        wing_col = C["bee_wing"]
        pygame.draw.ellipse(surf, wing_col, (bx - 14, by - 10 + wing_off, 12, 7))
        pygame.draw.ellipse(surf, wing_col, (bx + 2,  by - 10 + wing_off, 12, 7))

        # Body (yellow oval)
        pygame.draw.ellipse(surf, C["bee_body"], (bx - 8, by - 6, 16, 11))

        # Stripes
        for i, stripe_x in enumerate([bx - 4, bx, bx + 4]):
            col = C["bee_stripe"] if i % 2 == 0 else C["bee_body"]
            pygame.draw.line(surf, col, (stripe_x, by - 5), (stripe_x, by + 4), 2)

        # Eyes
        pygame.draw.circle(surf, (255, 255, 255), (bx + 4, by - 3), 2)
        pygame.draw.circle(surf, (0,   0,   0  ), (bx + 5, by - 3), 1)

        # Stinger
        pygame.draw.line(surf, C["bee_stripe"], (bx - 8, by), (bx - 11, by + 2), 2)

    def _draw_stats(self, surf, fonts):
        """Stats row below the flower grid."""
        sy = self.py + GRID_Y + GRID_H + 14

        cue_rate = (self.cue_rate_history[-1]
                    if self.cue_rate_history else 0.0)
        avg_cue  = (np.mean(self.cue_rate_history[-50:])
                    if len(self.cue_rate_history) >= 5 else 0.0)
        avg_rew  = (np.mean(self.reward_history[-50:])
                    if len(self.reward_history) >= 5 else 0.0)

        stats = [
            ("Cue rate",   f"{cue_rate:.2f}"),
            ("Avg (50ep)", f"{avg_cue:.2f}"),
            ("Avg reward", f"{avg_rew:.0f}"),
            ("Episodes",   f"{self.episode:,}"),
        ]

        col_w = PANEL_W // len(stats)
        for i, (lbl, val) in enumerate(stats):
            x = self.px + i * col_w + 8

            # Highlight cue rate if above chance
            val_col = C["highlight"] if (lbl == "Avg (50ep)" and avg_cue > 0.45) else C["text"]

            lbl_s = fonts["xs"].render(lbl, True, C["subtext"])
            val_s = fonts["sm"].render(val, True, val_col)
            surf.blit(lbl_s, (x, sy))
            surf.blit(val_s, (x, sy + 14))

    def _draw_chart(self, surf, fonts):
        """Mini cue-follow-rate chart in bottom strip."""
        chart_h = 38
        chart_y = self.py + PANEL_H - chart_h - 6
        chart_rect = pygame.Rect(self.px + 6, chart_y, PANEL_W - 12, chart_h)
        pygame.draw.rect(surf, C["chart_bg"], chart_rect, border_radius=3)
        pygame.draw.rect(surf, C["panel_border"], chart_rect, width=1, border_radius=3)

        # Chance line
        chance_y = chart_y + int(chart_h * (1 - 0.333))
        pygame.draw.line(surf, C["chance_line"],
                         (chart_rect.left + 2, chance_y),
                         (chart_rect.right - 2, chance_y), 1)

        # Label
        lbl = fonts["xs"].render("cue rate", True, C["subtext"])
        surf.blit(lbl, (chart_rect.left + 4, chart_y + 2))

        if len(self.cue_rate_history) < 2:
            return

        # Plot last N points
        n      = min(len(self.cue_rate_history), 200)
        data   = self.cue_rate_history[-n:]
        w      = chart_rect.width - 4
        pts    = []
        for i, v in enumerate(data):
            x = chart_rect.left + 2 + int(i * w / max(n - 1, 1))
            y = chart_y + int(chart_h * (1.0 - min(max(v, 0), 1)))
            pts.append((x, y))

        if len(pts) >= 2:
            pygame.draw.lines(surf, self.accent, False, pts, 2)



#train bees before visualistion
def warmup_panels(panels, n_episodes):
    """
    Train all panels silently for n_episodes before the visualisation starts.
    Shows a simple terminal progress bar so you know it's working.
    """
    if n_episodes <= 0:
        return

    print(f"\nWarming up all agents for {n_episodes} episodes...")

    for panel in panels:
        for ep in range(n_episodes):

            # Progress indicator every 100 episodes
            if (ep + 1) % 100 == 0:
                bar = "█" * ((ep + 1) // 100) + "░" * ((n_episodes - ep - 1) // 100)
                print(f"  {panel.label:<35} ep {ep+1:>5}/{n_episodes}  [{bar}]",
                      end="\r")

            obs, _ = panel.env.reset()
            terminated = False
            ep_reward = ep_cue = ep_visits = 0

            while not terminated:
                action = panel.agent.select_action(obs)
                next_obs, reward, term, _, info = panel.env.step(action)
                panel.agent.update(obs, action, reward, next_obs, term)
                obs = next_obs
                ep_reward += reward
                ep_visits += 1
                if info["had_cue"]:
                    ep_cue += 1
                terminated = term

            panel.agent.decay_epsilon()
            rate = ep_cue / ep_visits if ep_visits > 0 else 0.0
            panel.cue_rate_history.append(rate)
            panel.reward_history.append(ep_reward)
            panel.episode += 1

        # Reset obs for live visualisation
        panel.obs, _ = panel.env.reset()
        cx, cy = panel.flower_center(np.random.randint(N_FLOWERS))
        panel.bee_x, panel.bee_y = float(cx), float(cy)
        panel.state = panel.READY

        print(f"  {panel.label:<35} done. "
              f"Final cue rate: {np.mean(panel.cue_rate_history[-50:]):.3f}")

    print(f"\nWarmup complete — opening visualisation at episode {n_episodes}\n")


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():

    

    START_EPISODE = 2500
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Bumblebee RL – Minecraft-Style Visualisation")
    clock = pygame.time.Clock()

    # Fonts
    fonts = {
        "title": pygame.font.SysFont("consolas", 18, bold=True),
        "sm":    pygame.font.SysFont("consolas", 13, bold=True),
        "xs":    pygame.font.SysFont("consolas", 11),
    }

    # Create panels
    panels = []
    for i, (var, cue, label) in enumerate(CONDITIONS):
        px, py = PANEL_POS[i]
        panels.append(BeePanel(var, cue, label, ACCENTS[i], px, py))


     # ── Silent warmup before visualisation opens ──────────────────────────────
    warmup_panels(panels, START_EPISODE)

    speed_idx = DEFAULT_SPEED
    paused    = False
    # ... rest of main() unchanged

    speed_idx = DEFAULT_SPEED
    paused    = False

    while True:
        # ── Events ────────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    pygame.quit(); sys.exit()

                elif event.key == pygame.K_SPACE:
                    paused = not paused

                elif event.key == pygame.K_UP:
                    speed_idx = max(0, speed_idx - 1)
                    for p in panels:
                        p.anim_frames = SPEED_LEVELS[speed_idx]

                elif event.key == pygame.K_DOWN:
                    speed_idx = min(len(SPEED_LEVELS) - 1, speed_idx + 1)
                    for p in panels:
                        p.anim_frames = SPEED_LEVELS[speed_idx]

                elif event.key == pygame.K_r:
                    # Reset all agents
                    for i, (var, cue, label) in enumerate(CONDITIONS):
                        px, py = PANEL_POS[i]
                        panels[i] = BeePanel(var, cue, label, ACCENTS[i], px, py)
                        panels[i].anim_frames = SPEED_LEVELS[speed_idx]

        # ── Update ────────────────────────────────────────────────────────────
        if not paused:
            for panel in panels:
                panel.update()

        # ── Draw ──────────────────────────────────────────────────────────────
        screen.fill(C["bg"])

        # Global header
        hdr = fonts["title"].render(
            "Bumblebee Foraging RL  –  Copy-When-Uncertain Strategy  (Smolla et al. 2016)",
            True, (200, 220, 200)
        )
        screen.blit(hdr, (PAD, 10))

        # Speed & pause indicator (top right)
        speed_label = f"Speed: {'▶▶' * (len(SPEED_LEVELS) - speed_idx)}  {'PAUSED' if paused else ''}"
        spd_surf = fonts["xs"].render(speed_label, True, C["subtext"])
        screen.blit(spd_surf, (WIN_W - spd_surf.get_width() - 10, 14))

        # Controls hint
        hint = fonts["xs"].render("SPACE=pause  ↑↓=speed  R=reset  Q=quit", True, C["subtext"])
        screen.blit(hint, (WIN_W // 2 - hint.get_width() // 2, 14))

        for panel in panels:
            panel.draw(screen, fonts)

        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()
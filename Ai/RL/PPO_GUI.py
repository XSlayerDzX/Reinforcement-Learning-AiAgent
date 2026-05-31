import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, scrolledtext
import threading
import traceback
import sys
import os
from pathlib import Path
import torch
from collections import deque
from time import sleep
import io

# ── Theme ────────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ── stdout redirect so training prints appear in the GUI log ─────────────────
class _StreamRedirect(io.TextIOBase):
    def __init__(self, text_widget: scrolledtext.ScrolledText):
        self._widget = text_widget

    def write(self, msg: str):
        self._widget.after(0, self._append, msg)
        return len(msg)

    def _append(self, msg: str):
        self._widget.configure(state="normal")
        self._widget.insert(tk.END, msg)
        self._widget.see(tk.END)
        self._widget.configure(state="disabled")

    def flush(self):
        pass


# ── Main App ─────────────────────────────────────────────────────────────────
class PPOApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Clash Royale PPO Trainer")
        self.geometry("920x780")
        self.resizable(True, True)
        self._training_thread: threading.Thread | None = None
        self._stop_flag = threading.Event()
        self._build_ui()

    # ── UI Construction ───────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Header ────────────────────────────────────────────────────────────
        header = ctk.CTkFrame(self, corner_radius=0, fg_color="#1a1a2e")
        header.pack(fill="x")
        ctk.CTkLabel(
            header,
            text="⚔  Clash Royale PPO Trainer",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color="#4fc3f7",
        ).pack(side="left", padx=20, pady=14)
        self._status_badge = ctk.CTkLabel(
            header,
            text="● IDLE",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#888",
        )
        self._status_badge.pack(side="right", padx=20)

        # ── Body: two columns ─────────────────────────────────────────────────
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=16, pady=(10, 6))
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        # Left column: settings
        left = ctk.CTkScrollableFrame(body, label_text="Settings", label_font=ctk.CTkFont(weight="bold"), width=340)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        # Right column: log
        right = ctk.CTkFrame(body)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)
        ctk.CTkLabel(right, text="Training Log", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, sticky="w", padx=8, pady=(4, 2)
        )
        self._log = scrolledtext.ScrolledText(
            right,
            state="disabled",
            bg="#0d0d1a",
            fg="#b0bec5",
            font=("Consolas", 10),
            relief="flat",
            bd=0,
            wrap="word",
        )
        self._log.grid(row=1, column=0, sticky="nsew", padx=4, pady=(0, 4))

        self._build_settings(left)

        # ── Bottom bar: Start / Stop ───────────────────────────────────────────
        bar = ctk.CTkFrame(self, corner_radius=0, fg_color="#111122", height=56)
        bar.pack(fill="x", side="bottom")
        self._start_btn = ctk.CTkButton(
            bar,
            text="▶  Start Training",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#1565c0",
            hover_color="#1976d2",
            height=38,
            width=200,
            command=self._start_training,
        )
        self._start_btn.pack(side="left", padx=16, pady=9)
        self._stop_btn = ctk.CTkButton(
            bar,
            text="⏹  Stop",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#b71c1c",
            hover_color="#c62828",
            height=38,
            width=140,
            state="disabled",
            command=self._stop_training,
        )
        self._stop_btn.pack(side="left", padx=4, pady=9)
        ctk.CTkButton(
            bar,
            text="🗑  Clear Log",
            font=ctk.CTkFont(size=12),
            fg_color="#37474f",
            hover_color="#455a64",
            height=38,
            width=120,
            command=self._clear_log,
        ).pack(side="right", padx=16, pady=9)

    def _build_settings(self, parent):
        def section(text):
            ctk.CTkLabel(
                parent,
                text=text,
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color="#4fc3f7",
                anchor="w",
            ).pack(fill="x", pady=(14, 2), padx=4)
            ctk.CTkFrame(parent, height=1, fg_color="#333").pack(fill="x", padx=4, pady=(0, 6))

        def row(label, widget_fn, **kw):
            f = ctk.CTkFrame(parent, fg_color="transparent")
            f.pack(fill="x", padx=4, pady=3)
            ctk.CTkLabel(f, text=label, width=160, anchor="w", font=ctk.CTkFont(size=12)).pack(side="left")
            w = widget_fn(f, **kw)
            w.pack(side="right", fill="x", expand=True)
            return w

        # ── Paths ──────────────────────────────────────────────────────────────
        section("📁  Paths")

        self._bc_path_var = ctk.StringVar(
            value=str(PROJECT_ROOT / "Ai" / "Behavior_Cloning" / "lstm.pth")
        )
        self._ppo_path_var = ctk.StringVar(
            value=str(PROJECT_ROOT / "Ai" / "RL" / "ppo_model.pth")
        )

        def path_row(label, var):
            f = ctk.CTkFrame(parent, fg_color="transparent")
            f.pack(fill="x", padx=4, pady=3)
            ctk.CTkLabel(f, text=label, width=160, anchor="w", font=ctk.CTkFont(size=12)).pack(side="left")
            ctk.CTkEntry(f, textvariable=var, width=140).pack(side="left", fill="x", expand=True, padx=(0, 4))
            ctk.CTkButton(
                f, text="…", width=30,
                command=lambda v=var: v.set(filedialog.askopenfilename(filetypes=[("PyTorch model", "*.pth")])),
            ).pack(side="right")

        path_row("BC Weights (.pth)", self._bc_path_var)
        path_row("PPO Save Path", self._ppo_path_var)

        # ── Training loop ─────────────────────────────────────────────────────
        section("🔁  Training Loop")

        self._num_runs_var = ctk.IntVar(value=10)
        row("Training Runs", ctk.CTkEntry, textvariable=self._num_runs_var, width=80)

        self._rollouts_var = ctk.IntVar(value=1)
        row("Rollouts per Run", ctk.CTkEntry, textvariable=self._rollouts_var, width=80)

        self._infinite_var = ctk.BooleanVar(value=False)
        f_inf = ctk.CTkFrame(parent, fg_color="transparent")
        f_inf.pack(fill="x", padx=4, pady=3)
        ctk.CTkLabel(f_inf, text="Run Indefinitely", width=160, anchor="w", font=ctk.CTkFont(size=12)).pack(side="left")
        ctk.CTkSwitch(f_inf, variable=self._infinite_var, text="", onvalue=True, offvalue=False,
                      command=self._toggle_infinite).pack(side="right")

        # ── Model hyperparams ─────────────────────────────────────────────────
        section("🧠  Model")

        self._hidden_size_var = ctk.IntVar(value=128)
        row("Hidden Size", ctk.CTkEntry, textvariable=self._hidden_size_var, width=80)

        self._num_layers_var = ctk.IntVar(value=2)
        row("LSTM Layers", ctk.CTkEntry, textvariable=self._num_layers_var, width=80)

        # ── PPO hyperparams ───────────────────────────────────────────────────
        section("⚙️  PPO Hyperparameters")

        self._lr_var = ctk.StringVar(value="1e-4")
        row("Learning Rate", ctk.CTkEntry, textvariable=self._lr_var, width=80)

        self._gamma_var = ctk.StringVar(value="0.99")
        row("Gamma (γ)", ctk.CTkEntry, textvariable=self._gamma_var, width=80)

        self._clip_eps_var = ctk.StringVar(value="0.2")
        row("Clip Epsilon (ε)", ctk.CTkEntry, textvariable=self._clip_eps_var, width=80)

        self._entropy_var = ctk.StringVar(value="0.01")
        row("Entropy Coef", ctk.CTkEntry, textvariable=self._entropy_var, width=80)

        self._vf_coef_var = ctk.StringVar(value="0.5")
        row("Value Loss Coef", ctk.CTkEntry, textvariable=self._vf_coef_var, width=80)

        self._grad_clip_var = ctk.StringVar(value="0.5")
        row("Grad Clip Norm", ctk.CTkEntry, textvariable=self._grad_clip_var, width=80)

        # ── Env / execution ───────────────────────────────────────────────────
        section("🖥️  Environment")

        self._bs_window_var = ctk.StringVar(value="BlueStacks App Player 4")
        row("BlueStacks Window", ctk.CTkEntry, textvariable=self._bs_window_var, width=80)

        self._reward_win_var = ctk.StringVar(value="1.0")
        row("Reward Win", ctk.CTkEntry, textvariable=self._reward_win_var, width=80)

        self._reward_lose_var = ctk.StringVar(value="-1.0")
        row("Reward Lose", ctk.CTkEntry, textvariable=self._reward_lose_var, width=80)

        # ── Misc ───────────────────────────────────────────────────────────────
        section("🔧  Misc")

        self._reset_ppo_var = ctk.BooleanVar(value=False)
        f_rst = ctk.CTkFrame(parent, fg_color="transparent")
        f_rst.pack(fill="x", padx=4, pady=3)
        ctk.CTkLabel(f_rst, text="Ignore PPO Checkpoint", width=160, anchor="w", font=ctk.CTkFont(size=12)).pack(side="left")
        ctk.CTkSwitch(f_rst, variable=self._reset_ppo_var, text="", onvalue=True, offvalue=False).pack(side="right")

        self._verbose_var = ctk.BooleanVar(value=True)
        f_v = ctk.CTkFrame(parent, fg_color="transparent")
        f_v.pack(fill="x", padx=4, pady=3)
        ctk.CTkLabel(f_v, text="Verbose Logging", width=160, anchor="w", font=ctk.CTkFont(size=12)).pack(side="left")
        ctk.CTkSwitch(f_v, variable=self._verbose_var, text="", onvalue=True, offvalue=False).pack(side="right")

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _toggle_infinite(self):
        pass  # visual-only toggle; logic handled in training loop

    def _clear_log(self):
        self._log.configure(state="normal")
        self._log.delete("1.0", tk.END)
        self._log.configure(state="disabled")

    def _set_status(self, text: str, color: str):
        self._status_badge.configure(text=text, text_color=color)

    def _log_msg(self, msg: str):
        self._log.configure(state="normal")
        self._log.insert(tk.END, msg + "\n")
        self._log.see(tk.END)
        self._log.configure(state="disabled")

    # ── Start / Stop ──────────────────────────────────────────────────────────
    def _start_training(self):
        if self._training_thread and self._training_thread.is_alive():
            return
        self._stop_flag.clear()
        self._start_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")
        self._set_status("● RUNNING", "#66bb6a")
        # Redirect stdout -> log widget
        sys.stdout = _StreamRedirect(self._log)
        self._training_thread = threading.Thread(target=self._training_loop, daemon=True)
        self._training_thread.start()

    def _stop_training(self):
        self._stop_flag.set()
        self._set_status("● STOPPING…", "#ffa726")
        self._stop_btn.configure(state="disabled")

    def _on_training_done(self):
        sys.stdout = sys.__stdout__
        self._start_btn.configure(state="normal")
        self._stop_btn.configure(state="disabled")
        self._set_status("● IDLE", "#888")

    # ── Training Loop (runs in background thread) ─────────────────────────────
    def _training_loop(self):
        try:
            # ── Read settings from GUI ────────────────────────────────────────
            bc_path      = self._bc_path_var.get()
            ppo_path     = self._ppo_path_var.get()
            num_runs     = self._num_runs_var.get()
            rollouts_n   = self._rollouts_var.get()
            infinite     = self._infinite_var.get()
            hidden_size  = self._hidden_size_var.get()
            num_layers   = self._num_layers_var.get()
            lr           = float(self._lr_var.get())
            gamma        = float(self._gamma_var.get())
            clip_eps     = float(self._clip_eps_var.get())
            entropy_coef = float(self._entropy_var.get())
            vf_coef      = float(self._vf_coef_var.get())
            grad_clip    = float(self._grad_clip_var.get())
            ignore_ckpt  = self._reset_ppo_var.get()
            verbose      = self._verbose_var.get()

            # ── Lazy imports (keep GUI startup fast) ──────────────────────────
            from Ai.RL.PPO_Trainer import (
                sequenece_buffering, build_action_mask_from_obs,
                compute_returns_and_advantages, actor_critic_update,
            )
            from Ai.Agent.coordinate_utils import grid_to_pixel, bluestacks_to_global_coords
            from Ai.Behavior_Cloning.action_masking_config import WAIT_ID, AVAIL_FEATURE_TO_ACTION_ID
            from Ai.ClashRoyalData import ElixirCost
            from Ai.RL.ClashRoyalEnv import ClashRoyalEnv
            from Ai.RL.PPO_LSTM_Model import PPO_LSTM_Model
            from Ai.RL.PPO_Logger import log_update, log_rollout, log_winrate, get_next_update_id
            import pandas as pd

            # ── Patch hyperparams into trainer module ─────────────────────────
            import Ai.RL.PPO_Trainer as _trainer
            _trainer.GAMMA       = gamma
            _trainer.CLIP_EPS    = clip_eps
            _trainer.ENTROPY_COEF = entropy_coef
            _trainer.VF_COEF     = vf_coef
            _trainer.GRAD_CLIP   = grad_clip

            # ── Build env ─────────────────────────────────────────────────────
            print("[GUI] Initialising environment…")
            env = ClashRoyalEnv()
            print("[GUI] Environment ready.")

            # ── Build model ───────────────────────────────────────────────────
            print("[GUI] Loading model…")
            model = PPO_LSTM_Model(
                input_size=205,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_actions=13,
                pretrained_model_path=bc_path,
            )
            opt = torch.optim.Adam(model.parameters(), lr=lr)

            if not ignore_ckpt and os.path.exists(ppo_path):
                print(f"[GUI] Resuming from PPO checkpoint: {ppo_path}")
                ckpt = torch.load(ppo_path)
                model.load_state_dict(ckpt["model_state_dict"])
                opt.load_state_dict(ckpt["optimizer_state_dict"])
            else:
                print("[GUI] Starting fresh from BC warm-start weights.")

            # ── Training loop ─────────────────────────────────────────────────
            run = 0
            while not self._stop_flag.is_set():
                if not infinite and run >= num_runs:
                    print(f"[GUI] Completed {num_runs} training run(s). Done.")
                    break

                run += 1
                print(f"\n{'='*50}")
                print(f"[GUI] Run {run}{' / ' + str(num_runs) if not infinite else ' (infinite mode)'}")
                print(f"{'='*50}")

                # ── Collect rollouts ──────────────────────────────────────────
                from Ai.RL.PPO_Main import collect_rollout
                rollouts = collect_rollout(env, model, rollouts_to_collect=rollouts_n)

                if not rollouts:
                    print("[GUI][WARN] No rollouts returned, skipping update.")
                    continue

                if verbose:
                    first = rollouts[0]
                    print(f"[GUI] Steps collected: {len(first.get('rewards', []))}")
                    print(f"[GUI] Total reward: {sum(first.get('rewards', [])):.4f}")

                # ── PPO update ────────────────────────────────────────────────
                update_id = get_next_update_id()
                final_policy_loss = 0.0
                final_value_loss  = 0.0
                outcome = None

                for i, rollout in enumerate(rollouts):
                    log_rollout(update_id, i, rollout)
                    terminal_reward = rollout["rewards"][-1] if rollout["rewards"] else 0
                    if terminal_reward >= env.reward_win:
                        outcome = "win"
                    elif terminal_reward <= env.reward_lose:
                        outcome = "loss"
                    else:
                        outcome = "draw"

                    final_policy_loss, final_value_loss = actor_critic_update(
                        actor_critic_network=model,
                        optimizer=opt,
                        rollout=rollout,
                    )
                    print(f"[GUI] Rollout {i} → Policy Loss: {final_policy_loss:.4f} | Value Loss: {final_value_loss:.4f}")

                log_update(update_id, rollouts, final_policy_loss, final_value_loss, outcome)
                if outcome:
                    log_winrate(outcome)
                    color = "🟢" if outcome == "win" else "🔴" if outcome == "loss" else "🟡"
                    print(f"[GUI] Outcome: {color} {outcome.upper()}")

                # ── Save ──────────────────────────────────────────────────────
                torch.save({
                    "model_state_dict":      model.state_dict(),
                    "optimizer_state_dict":  opt.state_dict(),
                }, ppo_path)
                print(f"[GUI] Checkpoint saved → {ppo_path}")

                if self._stop_flag.is_set():
                    print("[GUI] Stop requested — exiting after this run.")
                    break

            print("[GUI] Training session ended.")

        except Exception:
            print("[GUI][ERROR] Unhandled exception in training loop:")
            traceback.print_exc()
        finally:
            self.after(0, self._on_training_done)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = PPOApp()
    app.mainloop()
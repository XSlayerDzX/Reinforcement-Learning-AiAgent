import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
import threading
import traceback
import sys
import os
from pathlib import Path
import io
import json

# ── Theme ────────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")


if getattr(sys, 'frozen', False):
    APP_ROOT = Path(sys._MEIPASS)
    USER_ROOT = Path(sys.executable).parent
else:
    APP_ROOT = Path(__file__).resolve().parents[2]
    USER_ROOT = APP_ROOT


# ── Logging Redirector ───────────────────────────────────────────────────────
class _StreamRedirect:
    def __init__(self, widget):
        self.widget = widget

    def write(self, string):
        self.widget.after(0, self._insert, string)

    def _insert(self, string):
        self.widget.configure(state="normal")
        self.widget.insert("end", string)
        self.widget.see("end")
        self.widget.configure(state="disabled")

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
        self._settings_file = USER_ROOT / "ppo_gui_settings.json"
        self._load_settings()
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _on_closing(self):
        self._save_settings()
        self.destroy()

    # ── UI Construction ───────────────────────────────────────────────────────
    def _build_ui(self):
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

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=16, pady=(10, 6))
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        left = ctk.CTkScrollableFrame(body, label_text="Settings", label_font=ctk.CTkFont(weight="bold"), width=340)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

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

        section("📁  Paths")

        self._bc_path_var = ctk.StringVar(
            value=str(APP_ROOT / "Ai" / "Behavior_Cloning" / "lstm.pth")
        )
        self._ppo_path_var = ctk.StringVar(
            value=str(USER_ROOT / "ppo_model.pth")
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

        section("🔁  Training Loop")

        self._num_runs_var = ctk.StringVar(value="10")
        row("Training Runs", ctk.CTkEntry, textvariable=self._num_runs_var, width=80)

        self._infinite_var = ctk.BooleanVar(value=False)
        f_inf = ctk.CTkFrame(parent, fg_color="transparent")
        f_inf.pack(fill="x", padx=4, pady=3)
        ctk.CTkLabel(f_inf, text="Run Indefinitely", width=160, anchor="w", font=ctk.CTkFont(size=12)).pack(side="left")
        ctk.CTkSwitch(f_inf, variable=self._infinite_var, text="", onvalue=True, offvalue=False,
                      command=self._toggle_infinite).pack(side="right")

        section("🧠  Model")

        self._hidden_size_var = ctk.StringVar(value="128")
        row("Hidden Size", ctk.CTkEntry, textvariable=self._hidden_size_var, width=80)

        self._num_layers_var = ctk.StringVar(value="2")
        row("LSTM Layers", ctk.CTkEntry, textvariable=self._num_layers_var, width=80)

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

        section("🖥️  Environment")

        self._bs_window_var = ctk.StringVar(value="BlueStacks App Player 4")
        row("BlueStacks Window", ctk.CTkEntry, textvariable=self._bs_window_var, width=80)

        self._reward_win_var = ctk.StringVar(value="1.0")
        row("Reward Win", ctk.CTkEntry, textvariable=self._reward_win_var, width=80)

        self._reward_lose_var = ctk.StringVar(value="-1.0")
        row("Reward Lose", ctk.CTkEntry, textvariable=self._reward_lose_var, width=80)

        section("🌐  ArenaBrain Dashboard")

        self._api_key_var = ctk.StringVar(value="")
        row("API Key", ctk.CTkEntry, textvariable=self._api_key_var, width=180, show="*")

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
        pass

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

    def _save_settings(self):
        settings = {
            "bc_path":     self._bc_path_var.get(),
            "ppo_path":    self._ppo_path_var.get(),
            "num_runs":    self._num_runs_var.get(),
            "infinite":    self._infinite_var.get(),
            "hidden_size": self._hidden_size_var.get(),
            "num_layers":  self._num_layers_var.get(),
            "lr":          self._lr_var.get(),
            "gamma":       self._gamma_var.get(),
            "clip_eps":    self._clip_eps_var.get(),
            "entropy":     self._entropy_var.get(),
            "vf_coef":     self._vf_coef_var.get(),
            "grad_clip":   self._grad_clip_var.get(),
            "bs_window":   self._bs_window_var.get(),
            "reward_win":  self._reward_win_var.get(),
            "reward_lose": self._reward_lose_var.get(),
            "api_key":     self._api_key_var.get(),
            "reset_ppo":   self._reset_ppo_var.get(),
            "verbose":     self._verbose_var.get(),
        }
        try:
            with open(self._settings_file, "w") as f:
                json.dump(settings, f, indent=4)
        except Exception as e:
            print(f"Failed to save settings: {e}")

    def _load_settings(self):
        if not self._settings_file.exists():
            return
        try:
            with open(self._settings_file, "r") as f:
                settings = json.load(f)
            def s(var, key):
                if key in settings:
                    var.set(settings[key])
            s(self._bc_path_var,     "bc_path")
            s(self._ppo_path_var,    "ppo_path")
            s(self._num_runs_var,    "num_runs")
            s(self._infinite_var,    "infinite")
            s(self._hidden_size_var, "hidden_size")
            s(self._num_layers_var,  "num_layers")
            s(self._lr_var,          "lr")
            s(self._gamma_var,       "gamma")
            s(self._clip_eps_var,    "clip_eps")
            s(self._entropy_var,     "entropy")
            s(self._vf_coef_var,     "vf_coef")
            s(self._grad_clip_var,   "grad_clip")
            s(self._bs_window_var,   "bs_window")
            s(self._reward_win_var,  "reward_win")
            s(self._reward_lose_var, "reward_lose")
            s(self._api_key_var,     "api_key")
            s(self._reset_ppo_var,   "reset_ppo")
            s(self._verbose_var,     "verbose")
        except Exception as e:
            print(f"Failed to load settings: {e}")

    # ── Start / Stop ──────────────────────────────────────────────────────────
    def _start_training(self):
        if self._training_thread and self._training_thread.is_alive():
            return
        self._save_settings()
        self._stop_flag.clear()
        self._start_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")
        self._set_status("● RUNNING", "#66bb6a")
        sys.stdout = _StreamRedirect(self._log)
        if not self._api_key_var.get().strip():
            if not messagebox.askyesno("Missing API Key", "No ArenaBrain API Key provided.\n\nDo you want to continue in Offline Mode?"):
                self._on_training_done()
                return
        self._training_thread = threading.Thread(target=self._training_loop, daemon=True)
        self._training_thread.start()

    def _stop_training(self):
        self._stop_flag.set()
        self._set_status("● STOPPING…", "#ffa726")
        self._stop_btn.configure(state="disabled")

    def _on_training_done(self):
        self._start_btn.configure(state="normal")
        self._stop_btn.configure(state="disabled")
        self._set_status("● IDLE", "#888")

    # ── Training Loop (runs in background thread) ─────────────────────────────
    def _training_loop(self):
        original_stdout = sys.stdout
        try:
            # ── Read settings ─────────────────────────────────────────────────
            bc_path      = self._bc_path_var.get()
            ppo_path     = self._ppo_path_var.get()
            num_runs     = int(self._num_runs_var.get() or 1)
            infinite     = self._infinite_var.get()
            hidden_size  = int(self._hidden_size_var.get() or 128)
            num_layers   = int(self._num_layers_var.get() or 2)
            lr           = float(self._lr_var.get())
            gamma        = float(self._gamma_var.get())
            clip_eps     = float(self._clip_eps_var.get())
            entropy_coef = float(self._entropy_var.get())
            vf_coef      = float(self._vf_coef_var.get())
            grad_clip    = float(self._grad_clip_var.get())
            ignore_ckpt  = self._reset_ppo_var.get()
            verbose      = self._verbose_var.get()
            api_key      = self._api_key_var.get()
            bs_window    = self._bs_window_var.get()

            # ── Lazy imports ──────────────────────────────────────────────────
            import torch
            from Ai.RL.PPO_Trainer import (
                compute_returns_and_advantages,
                actor_critic_update,
            )
            from Ai.RL.ClashRoyalEnv import ClashRoyalEnv
            from Ai.RL.PPO_LSTM_Model import PPO_LSTM_Model
            from Ai.RL.PPO_Logger import log_update, log_rollout, log_winrate, get_next_update_id
            from Ai.RL.PPO_Main import collect_rollout, _load_templates
            from Ai.models.run_config import (
                GAMMA, EPSILON, VF_COEF, ENT_COEF, GRAD_CLIP,
            )
            if str(APP_ROOT) not in sys.path:
                sys.path.append(str(APP_ROOT))
            from arena_web_integration.arena_client import ArenaBrainClient

            # ── Build env ─────────────────────────────────────────────────────
            print("[GUI] Initialising environment…")
            env = ClashRoyalEnv(window_title=bs_window)
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
                print("[GUI] Model and optimizer state restored successfully")
            else:
                print("[GUI] Starting fresh from BC warm-start weights.")

            # ── ArenaBrain client ─────────────────────────────────────────────
            web_client = ArenaBrainClient(api_key=api_key)
            if api_key.strip():
                try:
                    if not web_client.validate():
                        self.after(0, lambda: messagebox.showerror(
                            "Invalid API Key",
                            "The ArenaBrain API Key is invalid or inactive.\nTraining stopped."
                        ))
                        return
                    print("✅ API key validated.")
                except Exception as e:
                    print(f"⚠️ Could not connect to ArenaBrain API: {e}")
                    self.after(0, lambda e=e: messagebox.showerror(
                        "API Connection Error",
                        f"Could not connect to the cloud server.\n\n{e}\n\nTraining stopped."
                    ))
                    return

            # ── Load templates once (needed by collect_rollout) ───────────────
            print("[GUI] Loading button templates…")
            templates = _load_templates()
            print(f"[GUI] Templates loaded: {list(templates.keys())}")

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

                # ── Collect one rollout ───────────────────────────────────────
                # collect_rollout signature:
                #   (env, model, run_id, templates, use_masking, window_title, stop_flag)
                # Returns: (rollout dict, diagnostics dict)  or  (None, {})
                print("[GUI] Collecting rollout…")
                rollout, diagnostics = collect_rollout(
                    env          = env,
                    model        = model,
                    run_id       = f"gui_run_{run}",
                    templates    = templates,
                    use_masking  = True,
                    window_title = bs_window,
                    stop_flag    = self._stop_flag,
                )

                if rollout is None or not rollout.get("rewards"):
                    print("[GUI][WARN] Empty rollout returned, skipping PPO update.")
                    continue

                if verbose:
                    print(f"[GUI] Steps collected : {len(rollout['rewards'])}")
                    print(f"[GUI] Total reward    : {sum(rollout['rewards']):.4f}")
                    print(f"[GUI] Outcome         : {diagnostics.get('outcome', 'unknown')}")
                    print(f"[GUI] Illegal actions : {diagnostics.get('illegal_action_count', 0)} "
                          f"({diagnostics.get('illegal_action_rate', 0):.0%})")

                # ── PPO update ────────────────────────────────────────────────
                # actor_critic_update returns 4 values:
                #   policy_loss, value_loss, clip_fraction, action_entropy
                print("[GUI] Running PPO update…")
                policy_loss, value_loss, clip_fraction, action_entropy = actor_critic_update(
                    actor_critic_network = model,
                    optimizer            = opt,
                    rollout              = rollout,
                    vf                  = vf_coef,
                    ent_coef            = entropy_coef,
                    epsilon             = clip_eps,
                )
                print(
                    f"[GUI] Policy Loss: {policy_loss:.4f} | "
                    f"Value Loss: {value_loss:.4f} | "
                    f"Clip Frac: {clip_fraction:.3f} | "
                    f"Entropy: {action_entropy:.4f}"
                )

                # ── Log & push ────────────────────────────────────────────────
                outcome = diagnostics.get("outcome", "draw")
                update_id = get_next_update_id()

                ro_entry     = log_rollout(update_id, 0, rollout)
                update_entry = log_update(update_id, [rollout], policy_loss, value_loss, outcome)

                web_client.push_rollouts(update_id, [ro_entry])
                web_client.push_run(update_entry)

                if outcome:
                    log_winrate(outcome)
                    web_client.push_winrate(outcome)
                    color = "🟢" if outcome == "win" else "🔴" if outcome == "loss" else "🟡"
                    print(f"[GUI] Outcome: {color} {outcome.upper()}")

                # ── Save checkpoint ───────────────────────────────────────────
                torch.save({
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                }, ppo_path)
                print(f"[GUI] Checkpoint saved → {ppo_path}")

                if self._stop_flag.is_set():
                    print("[GUI] Stop requested — exiting after this run.")
                    break

            print("[GUI] Training session ended.")

        except Exception as e:
            print(f"[GUI][ERROR] Unhandled exception in training loop:")
            traceback.print_exc()
        finally:
            sys.stdout = original_stdout
            self.after(0, self._on_training_done)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = PPOApp()
    app.mainloop()

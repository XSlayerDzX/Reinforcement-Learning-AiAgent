from pathlib import Path
from omegaconf import OmegaConf


class ConfigLoader:
    _config = None

    @classmethod
    def load(cls, config_dir: str = "configs"):
        """
        Load and merge all YAML config files once.
        """

        if cls._config is not None:
            return cls._config  # already loaded

        project_root = cls._find_project_root()
        config_path = project_root / config_dir

        if not config_path.exists():
            raise FileNotFoundError(f"Config directory not found: {config_path}")

        # Load all yaml files inside configs/
        yaml_files = sorted(config_path.glob("*.yaml"))

        configs = [OmegaConf.load(file) for file in yaml_files]
        merged = OmegaConf.merge(*configs)

        # Resolve filesystem paths
        cls._resolve_paths(merged, project_root)

        cls._config = merged
        return cls._config

    @staticmethod
    def _find_project_root() -> Path:
        """
        Automatically locate project root (folder containing configs/).
        Works regardless of where script is launched.
        """
        current = Path(__file__).resolve()

        for parent in current.parents:
            if (parent / "configs").exists():
                return parent

        raise RuntimeError("Could not locate project root (missing configs folder).")

    @staticmethod
    def _resolve_paths(cfg, root: Path):
        """
        Convert relative paths from configs into absolute paths.
        """
        if "data" in cfg:
            for key, value in cfg.data.items():
                cfg.data[key] = str(root / value)

        if "models" in cfg:
            for key, value in cfg.models.items():
                cfg.models[key] = str(root / value)

        if "logs" in cfg:
            cfg.logs = str(root / cfg.logs)


if __name__ == "__main__":
    config = ConfigLoader.load()
    print(config.model)
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from rag_lab.pipeline.runner import run_pipeline


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Preserve hydra runtime information using HydraConfig.get()
    hydra_cfg = HydraConfig.get()
    cfg_dict["hydra"] = {"run": {"dir": hydra_cfg.runtime.output_dir}}
    # print(json.dumps(cfg_dict, indent=2))

    run_pipeline(cfg_dict)


if __name__ == "__main__":
    main()

import os
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import config

logger = logging.getLogger(__name__)

# Global references so the model is loaded once
_tokenizer = None
_model = None


def _get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model_and_tokenizer():
    global _tokenizer, _model

    if _model is not None and _tokenizer is not None:
        logger.info("Model already loaded, returning cached instance")
        return _model, _tokenizer

    device = _get_device()
    logger.info("Using device: %s", device)

    logger.info("Loading tokenizer from base model: %s", config.BASE_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(
        config.BASE_MODEL_NAME,
        trust_remote_code=True,
        padding_side="left",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading base model: %s", config.BASE_MODEL_NAME)
    kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }

    if device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["quantization_config"] = bnb_config
        kwargs["device_map"] = "auto"
        kwargs["dtype"] = torch.float16
    else:
        kwargs["device_map"] = "cpu"
        kwargs["dtype"] = torch.float32
        # The model safetensors are already 4-bit on disk;
        # on CPU we load them as fp32 (requires ~32GB RAM).
        # If that fails, use offload_folder for disk offloading.
        kwargs["offload_folder"] = "offload"
        os.environ["BITSANDBYTES_NOWELCOME"] = "1"

    base_model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL_NAME,
        **kwargs,
    )

    logger.info("Loading LoRA adapter from: %s", config.LORA_ADAPTER_PATH)
    model = PeftModel.from_pretrained(
        base_model,
        config.LORA_ADAPTER_PATH,
        is_trainable=False,
    )

    model.eval()
    logger.info("Model loaded successfully on %s", device)

    _model = model
    _tokenizer = tokenizer
    return model, tokenizer


def get_model_and_tokenizer():
    if _model is None or _tokenizer is None:
        return load_model_and_tokenizer()
    return _model, _tokenizer

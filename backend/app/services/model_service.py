"""
Model loading and management service.

Handles loading the base Qwen model and LoRA adapter,
with explicit failure if the adapter is missing.

Reuses logic from:
- scripts/compare_base_lora.py (PeftModel loading)
- scripts/train_lora_mts.py (chat template, system prompt)
"""

import logging
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

logger = logging.getLogger(__name__)


class ModelServiceError(Exception):
    """Raised when model operations fail."""
    pass


class ModelService:
    """
    Singleton service for managing the Qwen base model and LoRA adapter.

    Loads the model once and keeps it in memory for inference.
    Does NOT silently fall back to the base model if the adapter fails.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.model = None
        self.tokenizer = None
        self._base_model_name = None
        self._adapter_path = None
        self._device = None
        self._adapter_loaded = False
        self._initialized = True

    @property
    def device(self) -> str:
        """Get the compute device."""
        if self._device is None:
            if torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"
            logger.info(f"Model service device: {self._device}")
        return self._device

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready."""
        return self.model is not None and self.tokenizer is not None

    @property
    def has_adapter(self) -> bool:
        """Check if a LoRA adapter is loaded."""
        return self._adapter_loaded

    @property
    def model_info(self) -> dict:
        """Get information about the currently loaded model."""
        return {
            "base_model": self._base_model_name,
            "adapter_path": self._adapter_path,
            "adapter_loaded": self._adapter_loaded,
            "device": self.device,
            "is_loaded": self.is_loaded,
        }

    def validate_adapter(self, adapter_path: str | Path) -> bool:
        """
        Validate that a LoRA adapter directory has required files.

        Checks for:
        - adapter_config.json
        - adapter_model.safetensors

        Args:
            adapter_path: Path to the adapter directory.

        Returns:
            True if the adapter is valid.

        Raises:
            ModelServiceError: If the adapter is invalid.
        """
        path = Path(adapter_path)

        if not path.exists():
            raise ModelServiceError(
                f"Adapter directory not found: {path}"
            )

        if not path.is_dir():
            raise ModelServiceError(
                f"Adapter path is not a directory: {path}"
            )

        config_file = path / "adapter_config.json"
        if not config_file.exists():
            raise ModelServiceError(
                f"Missing adapter_config.json in {path}"
            )

        weights_file = path / "adapter_model.safetensors"
        if not weights_file.exists():
            raise ModelServiceError(
                f"Missing adapter_model.safetensors in {path}. "
                f"The adapter may not have been trained yet."
            )

        # Check file is not empty
        if weights_file.stat().st_size == 0:
            raise ModelServiceError(
                f"adapter_model.safetensors is empty (0 bytes) in {path}"
            )

        logger.info(f"Adapter validated: {path}")
        return True

    def load_model(
        self,
        base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        adapter_path: str | Path | None = None,
    ) -> None:
        """
        Load the base model and optionally a LoRA adapter.

        Args:
            base_model_name: HuggingFace model identifier.
            adapter_path: Path to the LoRA adapter directory.
                If provided, the adapter MUST load successfully.

        Raises:
            ModelServiceError: If loading fails.
        """
        requested_adapter = str(adapter_path) if adapter_path else None
        if self.is_loaded:
            if self._adapter_path == requested_adapter and self._base_model_name == base_model_name:
                logger.info("Model already loaded with matching configuration")
                return
            logger.info("Reloading model due to configuration change")
            self.unload_model()

        try:
            import os
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{base_model_name.replace('/', '--')}"
            if cache_dir.exists():
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                try:
                    import huggingface_hub.constants
                    huggingface_hub.constants.HF_HUB_OFFLINE = True
                except Exception:
                    pass

            # Load tokenizer
            logger.info(f"Loading tokenizer: {base_model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_name
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load base model
            logger.info(
                f"Loading base model: {base_model_name} "
                f"on {self.device}"
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                dtype=torch.float16,
            )
            base_model = base_model.to(self.device)
            base_model.eval()

            self._base_model_name = base_model_name

            # Load LoRA adapter if specified
            if adapter_path is not None:
                adapter_path = Path(adapter_path)

                # Validate adapter before attempting to load
                self.validate_adapter(adapter_path)

                logger.info(f"Loading LoRA adapter: {adapter_path}")

                self.model = PeftModel.from_pretrained(
                    base_model,
                    str(adapter_path),
                )

                self._adapter_path = str(adapter_path)
                self._adapter_loaded = True

                logger.info("LoRA adapter loaded successfully")
            else:
                self.model = base_model
                self._adapter_loaded = False
                logger.info(
                    "No adapter specified — using base model only"
                )

            # Move to device and set eval mode
            self.model = self.model.to(self.device)
            self.model.eval()

            logger.info("Model ready for inference")

        except ModelServiceError:
            self._cleanup()
            raise
        except Exception as e:
            self._cleanup()
            raise ModelServiceError(
                f"Failed to load model: {e}"
            ) from e

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.1,
        do_sample: bool = False,
        repetition_penalty: float = 1.1,
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: The formatted prompt string.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            do_sample: Whether to use sampling.
            repetition_penalty: Penalty for repeated tokens.

        Returns:
            Generated text (excluding the prompt).

        Raises:
            ModelServiceError: If generation fails.
        """
        if not self.is_loaded:
            raise ModelServiceError(
                "Model not loaded. Call load_model() first."
            )

        try:
            self.tokenizer.truncation_side = "left"
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096,
            )

            inputs = {
                key: value.to(self.device)
                for key, value in inputs.items()
            }

            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    repetition_penalty=repetition_penalty,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # Extract only the generated tokens
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = output[0][input_length:]

            result = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            ).strip()

            return result

        except Exception as e:
            raise ModelServiceError(
                f"Generation failed: {e}"
            ) from e

    def build_chat_prompt(
        self,
        system_message: str,
        user_message: str,
    ) -> str:
        """
        Build a prompt using the Qwen chat template.

        Args:
            system_message: System prompt content.
            user_message: User message content.

        Returns:
            Formatted prompt string.
        """
        if self.tokenizer is None:
            raise ModelServiceError(
                "Tokenizer not loaded. Call load_model() first."
            )

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def unload_model(self) -> None:
        """Unload the model to free memory."""
        self._cleanup()
        logger.info("Model unloaded")

    def _cleanup(self) -> None:
        """Clean up model resources."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self._adapter_loaded = False
        self._base_model_name = None
        self._adapter_path = None

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

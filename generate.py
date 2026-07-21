import logging
import torch
from model_loader import get_model_and_tokenizer
import config

logger = logging.getLogger(__name__)


def build_prompt(topic: str, tone: str, industry: str, audience: str,
                 post_length: str, cta: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert LinkedIn content writer. "
                "Write professional, engaging LinkedIn posts that drive engagement. "
                f"Tone: {tone}. Industry: {industry}. "
                f"Target audience: {audience}. "
                f"Post length: {post_length}. "
                f"Call to action: {cta}."
            ),
        },
        {
            "role": "user",
            "content": f"Write a LinkedIn post about: {topic}",
        },
    ]

    tokenizer = get_model_and_tokenizer()[1]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt


def generate_post(topic: str, tone: str = "Professional", industry: str = "Technology",
                  audience: str = "General", post_length: str = "Medium",
                  cta: str = "Engage with the post") -> str:
    try:
        model, tokenizer = get_model_and_tokenizer()
        prompt = build_prompt(topic, tone, industry, audience, post_length, cta)

        logger.info("Generating post for topic: %s", topic)

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=config.MAX_NEW_TOKENS * 2,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.MAX_NEW_TOKENS,
                temperature=config.TEMPERATURE,
                top_p=config.TOP_P,
                top_k=config.TOP_K,
                repetition_penalty=config.REPETITION_PENALTY,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs.input_ids.shape[1]:]
        result = tokenizer.decode(generated, skip_special_tokens=True)

        # Clean up the response
        result = result.strip()
        # Remove any chat template artifacts
        for token in ["<|im_start|>", "<|im_end|>", "<|assistant|>", "<|user|>"]:
            result = result.replace(token, "")

        result = result.strip().strip('"').strip()

        logger.info("Post generated successfully (%d chars)", len(result))
        return result

    except Exception as e:
        logger.error("Generation failed: %s", str(e), exc_info=True)
        raise RuntimeError(f"Failed to generate post: {str(e)}") from e

a= """

You are extracting informative technical features from AI model descriptions.

Goal:
Return only noun phrases that state an informative, reusable technical property of the AI model.

From the description, extract noun phrases that clearly describe:

- a capability or task (e.g., image generation, question answering)
- a modality (e.g., text-to-image)
- a model class or architecture family (e.g., Stable Diffusion)
- a specific architecture, algorithm, or training method (e.g., LoRA, PPO, transfer learning)
- a training technique or modification (e.g., noise offset)
- a deployment framework or ML library (e.g., diffusers, stable-baselines3)
- a model format or quantization type (e.g., Safetensors format, GGUF quant)
- a safety or alignment mechanism (e.g., NSFW filtering)
- a named benchmark or well-known dataset (e.g., SQuAD v1.1)
- a concrete license or access condition

Exclude any noun phrase that:
- is a specific checkpoint name, version string, or parameter-size variant (e.g., Zephyr-3.43B, stable-diffusion-v1-5)
- is a username, person name, or project name
- is a prompt format
- is an aesthetic/style tag (e.g., anime, realism, photorealism)
- is only a number or statistic without architectural meaning
- is a repository structure term (e.g., training scripts, model repository)
- is generic (e.g., model, dataset, weight, trained model, layer)
- is marketing or descriptive language
- does not clearly represent a reusable ML concept

Rules:
- Extract only explicit noun phrases from the text.
- Do NOT infer missing features.
- Do NOT normalize, expand, or rewrite phrases.
- Prefer the shortest noun phrase that preserves meaning.
- Remove duplicates (case-insensitive).
- Preserve order of first valid appearance.
- Be strict.

Output:
Return ONLY valid JSON in this exact format:
{"features": ["<noun phrase>", "..."]}

Text:

"""
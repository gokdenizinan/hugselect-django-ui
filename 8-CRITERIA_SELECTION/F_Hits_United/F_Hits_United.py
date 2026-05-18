import pandas as pd

CHATGPT_RECOMMENDATIONS = {
    "sample_1": [
        "microsoft/unixcoder-base",
        "microsoft/graphcodebert-base",
        "microsoft/codebert-base",
        "Salesforce/codet5p-110m-embedding",
        "jinaai/jina-embeddings-v2-base-code",
        "nomic-ai/CodeRankEmbed",
        "nomic-ai/nomic-embed-code",
        "huggingface/CodeBERTa-small-v1",
        "krlvi/sentence-t5-base-nlpl-code_search_net",
        "microsoft/unixcoder-base-nine"
    ],
    "sample_2": [
    "yikuan8/Clinical-Longformer",
    "yikuan8/Clinical-BigBird",
    "riggsmed/deid-LONGFORMER-NemPII",
    "allenai/longformer-base-4096",
    "google/bigbird-roberta-base",
    "google/bigbird-roberta-large",
    "emilyalsentzer/Bio_ClinicalBERT",
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "dmis-lab/biobert-base-cased-v1.1",
    "thomas-sounack/BioClinical-ModernBERT-large"
],
    "sample_7": [
    "benakrab/AraBART-AHS",
    "UBC-NLP/AraT5-base-title-generation",
    "UBC-NLP/AraT5-base",
    "UBC-NLP/AraT5-msa-base",
    "UBC-NLP/AraT5v2-base-1024",
    "fatmaserry/AraT5v2-arabic-summarization",
    "omarsabri8756/AraT5v2-XLSum-arabic-text-summarization",
    "malmarjeh/t5-arabic-text-summarization",
    "malmarjeh/mbert2mbert-arabic-text-summarization",
    "ahmed0189/mT5-Arabic-text-summarization"
],
    "sample_8": [
    "facebook/nllb-200-distilled-600M",
    "facebook/nllb-200-distilled-1.3B",
    "facebook/nllb-200-1.3B",
    "facebook/nllb-200-3.3B",
    "facebook/m2m100_418M",
    "facebook/m2m100_1.2B",
    "facebook/mbart-large-50-many-to-many-mmt",
    "google/madlad400-3b-mt",
    "Helsinki-NLP/opus-mt-ar-en",
    "Helsinki-NLP/opus-mt-en-ar",
],
    "sample_9": [
    "google/mt5-small",
    "google/mt5-base",
    "google/mt5-large",
    "bigscience/mt0-small",
    "bigscience/mt0-base",
    "bigscience/mt0-large",
    "facebook/mbart-large-50-many-to-many-mmt",
    "facebook/m2m100_418M",
    "facebook/m2m100_1.2B",
    "facebook/nllb-200-distilled-600M"
],
    "sample_18": [
    "google/pegasus-large",
    "google/pegasus-xsum",
    "facebook/bart-large",
    "facebook/bart-large-cnn",
    "t5-base",
    "t5-large",
    "google/flan-t5-base",
    "google/flan-t5-large",
    "allenai/led-base-16384",
    "google/long-t5-tglobal-base"
],
    "sample_19":[
    "Qwen/Qwen2-7B",
    "Qwen/Qwen2-72B",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-14B",
    "baichuan-inc/Baichuan2-13B-Base",
    "baichuan-inc/Baichuan2-7B-Base",
    "THUDM/chatglm3-6b-base",
    "THUDM/glm-4-9b-base",
    "01-ai/Yi-34B",
    "deepseek-ai/deepseek-llm-67b-base"
],
    "sample_24": [
    "google/vit-base-patch16-224",
    "google/vit-base-patch32-224",
    "google/vit-large-patch16-224",
    "google/vit-large-patch32-224",
    "google/vit-huge-patch14-224-in21k",
    "facebook/deit-base-patch16-224",
    "facebook/deit-small-patch16-224",
    "facebook/deit-tiny-patch16-224",
    "microsoft/swin-base-patch4-window7-224",
    "microsoft/swin-tiny-patch4-window7-224"
],
    "sample_55": [
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "intfloat/e5-large-v2",
    "intfloat/e5-base-v2",
    "BAAI/bge-large-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "thenlper/gte-large",
    "thenlper/gte-base",
    "nomic-ai/nomic-embed-text-v1.5"
],
    "sample_56": [
    "openai/clip-vit-large-patch14",
    "openai/clip-vit-base-patch32",
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    "Salesforce/blip-itm-base-coco",
    "Salesforce/blip2-itm-vit-g",
    "kakaobrain/align-base",
    "google/siglip-base-patch16-224",
    "google/siglip-so400m-patch14-384",
    "apple/DFN5B-CLIP-ViT-H-14"
],
    "sample_57": [
    "openai/clip-vit-large-patch14",
    "openai/clip-vit-base-patch32",
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    "google/siglip-base-patch16-224",
    "google/siglip-so400m-patch14-384",
    "kakaobrain/align-base",
    "Salesforce/blip-itm-base-coco",
    "Salesforce/blip2-itm-vit-g",
    "microsoft/xclip-base-patch32"
],
    "sample_58":  [
    "TencentARC/t2i-adapter-canny-sdxl-1.0",
    "TencentARC/t2i-adapter-lineart-sdxl-1.0",
    "TencentARC/t2i-adapter-sketch-sdxl-1.0",
    "TencentARC/t2i-adapter-depth-zoe-sdxl-1.0",
    "TencentARC/t2i-adapter-openpose-sdxl-1.0",
    "TencentARC/t2iadapter_color_sd14v1",
    "TencentARC/t2iadapter_canny_sd14v1",
    "TencentARC/t2iadapter_sketch_sd14v1",
    "xinsir/controlnet-union-sdxl-1.0",
    "diffusers/controlnet-depth-sdxl-1.0-small",
],
    "sample_59":  [
    "madebyollin/sdxl-vae-fp16-fix",
    "stabilityai/sdxl-vae",
    "stabilityai/sdxl-vae-0.9",
    "KBlueLeaf/EQ-SDXL-VAE",
    "madebyollin/taesdxl",
    "nubby/blessed-sdxl-vae-fp16-fix",
    "stabilityai/sd-vae-ft-mse",
    "stabilityai/sd-vae-ft-ema",
    "hakurei/waifu-diffusion-v1-4-vae",
    "crosslabs-org/sdxl-vae-diffusers"
],
    "sample_60": [
    "latent-consistency/lcm-lora-sdv1-5",
    "latent-consistency/lcm-lora-sdxl",
    "ByteDance/Hyper-SD::Hyper-SD15-1step-lora.safetensors",
    "ByteDance/Hyper-SD::Hyper-SD15-2steps-lora.safetensors",
    "ByteDance/Hyper-SD::Hyper-SD15-4steps-lora.safetensors",
    "ByteDance/Hyper-SD::Hyper-SDXL-1step-lora.safetensors",
    "ByteDance/Hyper-SD::Hyper-SDXL-2steps-lora.safetensors",
    "ByteDance/Hyper-SD::Hyper-SDXL-4steps-lora.safetensors",
    "h1t/TCD-SDXL-LoRA",
    "ByteDance/Hyper-SD::Hyper-SDXL-8steps-CFG-lora.safetensors",
],
    "sample_61": [
    "madebyollin/sdxl-vae-fp16-fix",
    "stabilityai/sdxl-vae",
    "nubby/blessed-sdxl-vae-fp16-fix",
    "madebyollin/taesdxl",
    "KBlueLeaf/EQ-SDXL-VAE",
    "wangkanai/sdxl-vae",
    "stabilityai/sdxl-vae-0.9",
    "crosslabs-org/sdxl-vae-diffusers",
    "madebyollin/sdxl-vae-fp16-fix::sdxl.vae.safetensors",
    "madebyollin/sdxl-vae-fp16-fix::diffusion_pytorch_model.safetensors",
],
    "sample_62":[
    "stabilityai/stable-diffusion-x4-upscaler",
    "stabilityai/sd-x2-latent-upscaler",
    "stabilityai/stable-diffusion-2-base",
    "stabilityai/stable-diffusion-2-1-base",
    "runwayml/stable-diffusion-v1-5",
    "stabilityai/sdxl-turbo",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    "madebyollin/sdxl-vae-fp16-fix",
    "diffusers/latent-diffusion-super-resolution-2x"
],
    "sample_68": [
    "microsoft/deberta-v3-large",
    "microsoft/deberta-large",
    "roberta-large",
    "google/electra-large-discriminator",
    "bert-large-uncased",
    "bert-large-cased",
    "albert-xxlarge-v2",
    "google/electra-base-discriminator",
    "roberta-base",
    "bert-base-uncased"
],
    "sample_69":  [
    "distilbert-base-uncased",
    "distilroberta-base",
    "huawei-noah/TinyBERT_General_4L_312D",
    "huawei-noah/TinyBERT_General_6L_768D",
    "nreimers/MiniLM-L6-H384-uncased",
    "nreimers/MiniLM-L12-H384-uncased",
    "microsoft/MiniLM-L12-H384-uncased",
    "google/mobilebert-uncased",
    "albert-base-v2",
    "prajjwal1/bert-small"
],
"sample_86":[
    "facebook/wav2vec2-base",
    "facebook/wav2vec2-base-960h",
    "patrickvonplaten/wav2vec2_tiny_random",
    "superb/wav2vec2-base-superb-ks",
    "microsoft/wavlm-base",
    "microsoft/wavlm-base-plus",
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    "MIT/ast-base",
    "ntu-spml/distilhubert",
    "facebook/hubert-base-ls960"
],
    "sample_90": [
    "LIAMF-USP/roberta-large-finetuned-race",
    "danlou/roberta-large-finetuned-csqa",
    "nagupv/deberta-v3-large-hf-60kgtesmcontext_f2",
    "Riiid/kda-albert-xxlarge-v2-race",
    "Rocketknight1/bert-base-uncased-finetuned-swag",
    "AndyyyCai/bert-base-uncased-finetuned-copa",
    "LIAMF-USP/aristo-roberta",
    "danlou/albert-xxlarge-v2-finetuned-csqa",
    "nagupv/bigbird-roberta-large_LLMMDLFIND_11_09_2023_0",
    "Erland/zero-shot_cross-lingual_temp-gen-x-xlmr_large"
],
    "sample_98":  [
    "allenai/scibert_scivocab_uncased",
    "allenai/scibert_scivocab_cased",
    "allenai/scibert_basevocab_uncased",
    "allenai/scibert_basevocab_cased",
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "emilyalsentzer/Bio_ClinicalBERT",
    "dmis-lab/biobert-base-cased-v1.1",
    "dmis-lab/biobert-base-cased-v1.2",
    "gsarti/scibert-nli"
],
    "sample_99": [
    "emilyalsentzer/Bio_ClinicalBERT",
    "yikuan8/Clinical-Longformer",
    "nlpie/clinical-distilbert-base-uncased",
    "dmis-lab/biobert-base-cased-v1.1",
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "UFNLP/gatortron-base",
    "UFNLP/gatortron-medium",
    "UFNLP/gatortron-large",
    "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
    "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
],
"sample_116" : [
    "cross-encoder/nli-deberta-v3-base",
    "cross-encoder/nli-deberta-v3-small",
    "cross-encoder/nli-deberta-v3-large",
    "FacebookAI/roberta-large-mnli",
    "facebook/bart-large-mnli",
    "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    "joeddav/xlm-roberta-large-xnli",
    "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
    "AIDA-UPM/xlm-roberta-large-snli_mnli_xnli_fever_r1_r2_r3"
],
    "sample_130":  [
    "microsoft/deberta-base",
    "microsoft/deberta-large",
    "microsoft/deberta-v3-base",
    "microsoft/deberta-v3-large",
    "google/electra-large-discriminator",
    "google/bigbird-roberta-base",
    "google/bigbird-roberta-large",
    "allenai/longformer-base-4096",
    "allenai/longformer-large-4096",
    "xlnet-base-cased"
],
    "sample_134": [
    "facebook/bart-large",
    "facebook/bart-large-cnn",
    "facebook/bart-base",
    "google/flan-t5-large",
    "google/flan-t5-xl",
    "google/t5-v1_1-large",
    "google/t5-v1_1-xl",
    "allenai/led-base-16384",
    "allenai/led-large-16384",
    "google/pegasus-large"
],
    "sample_135": [
    "facebook/bart-base",
    "facebook/bart-large",
    "facebook/bart-large-cnn",
    "facebook/bart-large-xsum",
    "facebook/bart-large-mnli",
    "sshleifer/distilbart-cnn-12-6",
    "sshleifer/distilbart-cnn-6-6",
    "philschmid/bart-large-cnn-samsum",
    "allenai/led-base-16384",
    "allenai/led-large-16384"
],
    "sample_136":  [
    "facebook/bart-large-cnn",           # BART-large fine-tuned on CNN/DM — the most widely used summarization backbone; strong NLG transferability
    "facebook/bart-large-xsum",          # BART-large fine-tuned on XSum — best for abstractive/compressed single-sentence summaries
    "facebook/bart-base",                # BART-base — lightweight denoising seq2seq; best base-scale starting point for fine-tuning
    "google/pegasus-large",              # PEGASUS-large — pretraining objective tailored for summarization; strong sample efficiency (<1K examples)
    "google/pegasus-xsum",               # PEGASUS fine-tuned on XSum — best zero-shot abstractive summarization out of the box
    "google/pegasus-x-large",            # PEGASUS-X large — extends PEGASUS to 16K tokens; best for long-document summarization
    "google/flan-t5-large",              # Flan-T5 large — T5 denoising + instruction tuning; strong general NLG transferability across tasks
    "google/flan-t5-base",               # Flan-T5 base — efficient instruction-tuned seq2seq; best lightweight fine-tuning baseline
    "allenai/PRIMERA-multinews",         # PRIMERA — LED-based multi-document summarization; strong few-shot and zero-shot NLG generalization
    "microsoft/prophetnet-large-uncased-cnndm", # ProphetNet-large — future n-gram prediction pretraining; SOTA on CNN/DM and Gigaword
],
"sample_146" :  [
    "sentence-transformers/all-mpnet-base-v2",        # Strong general-purpose retriever with excellent semantic quality
    "sentence-transformers/all-MiniLM-L6-v2",         # Lightweight + fast, good trade-off for real-time dialogue systems
    "sentence-transformers/multi-qa-mpnet-base-dot-v1",  # Optimized for retrieval (QA-style), strong generalization
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",   # Fast + retrieval-tuned variant
    "facebook/dpr-question_encoder-single-nq-base",  # Classic DPR query encoder
    "facebook/dpr-ctx_encoder-single-nq-base",       # DPR context encoder (pair with above)
    "intfloat/e5-base-v2",                           # Excellent zero-shot retrieval and generalization (E5 family)
    "intfloat/e5-small-v2",                          # Lightweight version for faster inference
    "BAAI/bge-base-en-v1.5",                         # State-of-the-art general embedding model (strong BEIR performance)
    "BAAI/bge-small-en-v1.5"                         # Smaller, faster variant with good retrieval quality
],
    "sample_158": [
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
    "intfloat/e5-base-v2",
    "intfloat/e5-large-v2",
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-large-en-v1.5",
    "thenlper/gte-base",
    "thenlper/gte-large",
    "sentence-transformers/multi-qa-mpnet-base-dot-v1",
    "sentence-transformers/paraphrase-mpnet-base-v2",
],
    "Sample_177": [
    "deepseek-ai/DeepSeek-R1",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "Qwen/QwQ-32B",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "Qwen/Qwen2.5-Math-7B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "google/gemma-2-27b-it",
],
    "Sample_178": [
    "facebook/contriever-msmarco",
    "facebook/contriever",
    "intfloat/e5-base-v2",
    "BAAI/bge-base-en-v1.5",
    "thenlper/gte-base",
    "Alibaba-NLP/gte-modernbert-base",
    "jinaai/jina-embeddings-v2-base-en",
    "sentence-transformers/multi-qa-mpnet-base-dot-v1",
    "BAAI/bge-m3",
    "intfloat/e5-small-v2",
],
    "Sample_179": [
    "microsoft/swin-base-patch4-window7-224",
    "microsoft/swin-large-patch4-window12-384",
    "microsoft/swinv2-base-patch4-window8-256",
    "microsoft/swinv2-large-patch4-window12-192-22k",
    "facebook/convnext-large-224",
    "facebook/convnextv2-large-22k-224",
    "microsoft/beit-base-patch16-224",
    "microsoft/beit-large-patch16-224-pt22k-ft22k",
    "google/vit-hybrid-base-bit-384",
    "openai/clip-vit-large-patch14"
]
,
    "Sample_184": [
    "microsoft/swinv2-large-patch4-window12-192-22k",
    "microsoft/swinv2-base-patch4-window8-256",
    "microsoft/swin-large-patch4-window12-384",
    "microsoft/swin-base-patch4-window7-224",
    "microsoft/swin-tiny-patch4-window7-224",
    "microsoft/beit-large-patch16-224-pt22k-ft22k",
    "microsoft/beit-base-patch16-224",
    "facebook/convnextv2-large-22k-224",
    "facebook/convnext-large-224",
    "google/vit-hybrid-base-bit-384"
],
    "Sample_194": [
    "facebook/bart-large",
    "facebook/bart-base",
    "t5-large",
    "t5-base",
    "t5-small",
    "google/mt5-base",
    "google/mt5-small",
    "allenai/led-base-16384",
    "allenai/led-large-16384",
    "google/pegasus-large"
],
    "Sample_212": [
    "google/flan-t5-base",
    "google/flan-t5-large",
    "google/flan-t5-xl",
    "google/t5-base",
    "google/t5-large",
    "google/ul2",
    "bigscience/T0pp",
    "bigscience/T0_3B",
    "allenai/unifiedqa-t5-large",
    "allenai/unifiedqa-v2-t5-large-1363200"
],
    "Sample_213": [
    "google/flan-t5-large",
    "google/flan-t5-base",
    "google/t5-v1_1-large",
    "facebook/bart-large",
    "facebook/bart-base",
    "bigscience/T0_3B",
    "bigscience/T0pp",
    "allenai/unifiedqa-v2-t5-large-1363200",
    "microsoft/phi-2",
    "meta-llama/Llama-2-7b-hf"
],
    "Sample_216": [
    "runwayml/stable-diffusion-v1-5",
    "stabilityai/stable-diffusion-2-1",
    "stabilityai/stable-diffusion-2-base",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    "stabilityai/sdxl-turbo",
    "hakurei/waifu-diffusion",
    "prompthero/openjourney",
    "nitrosocke/Ghibli-Diffusion",
    "stabilityai/stable-diffusion-3-large"
],
    "Sample_217": [
    "openai/clip-vit-large-patch14",            # CLIP ViT-L/14 — the standard CLIPScore backbone; most cited for alignment evaluation
    "openai/clip-vit-base-patch32",             # CLIP ViT-B/32 — lightweight CLIP; fast cosine similarity for large-scale evaluation
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", 
    "google/siglip-so400m-patch14-384",         # SigLIP SO400M/14@384 — top-performing SigLIP checkpoint; outperforms CLIP on zero-shot alignment
    "google/siglip-base-patch16-224",  
    "yuvalkirstain/PickScore_v1",               # PickScore — CLIP fine-tuned on human pairwise preferences; best for perceptual quality ranking
    "THUDM/ImageReward",         
    "Salesforce/blip-itm-base-coco",            # BLIP ITM (image-text matching) — fine-tuned for binary and graded alignment scoring
    "Salesforce/blip2-opt-2.7b",               # BLIP-2 OPT-2.7B — Q-Former bridge to LLM; rich cross-modal embeddings for alignment evaluation
    "QuanSun/EVA-CLIP-8B-448",                  # EVA-CLIP 8B — iterative MIM + CLIP training; highest-capacity open vision-language encoder
],
    "Sample_220": [
    "openai/clip-vit-large-patch14",
    "openai/clip-vit-base-patch32",
    "openai/clip-vit-large-patch14-336",
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    "google/siglip-base-patch16-224",
    "google/siglip-large-patch16-384",
    "Salesforce/blip2-flan-t5-xl",
    "Salesforce/blip-image-captioning-large",
    "microsoft/git-large-coco"
],
    "Sample_223": [
    "runwayml/stable-diffusion-v1-5",
    "CompVis/stable-diffusion-v1-4",
    "stabilityai/stable-diffusion-2-1",
    "stabilityai/stable-diffusion-2-base",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    "Lykon/dreamshaper-8",
    "SG161222/Realistic_Vision_V6.0_B1_noVAE",
    "stablediffusionapi/realistic-vision-v51",
    "dreamlike-art/dreamlike-photoreal-2.0"
],
"Sample_224": [
    "meta-llama/Llama-3.1-8B-Instruct",          # LLaMA 3.1 8B Instruct — best open LLM for generating fine-grained visual class descriptions; used in LLaMP-style frameworks
    "meta-llama/Llama-3.2-3B-Instruct",          # LLaMA 3.2 3B Instruct — lightweight variant; fast semantic description generation with low GPU footprint
    "mistralai/Mistral-7B-Instruct-v0.3",        # Mistral 7B Instruct — strong instruction-following; used in Meta-Prompting for category description ensembling
    "google/flan-t5-xxl",                         # Flan-T5 XXL — seq2seq; encyclopedic class knowledge + instruction-following; efficient for batch description generation
    "google/flan-t5-xl",                          # Flan-T5 XL — lighter Flan-T5; fast class description generation with good semantic coverage for 11+ classification datasets
    "microsoft/phi-2",                            # Phi-2 — 2.7B compact reasoning LLM; LoRA-efficient semantic knowledge injection with low memory cost
    "Qwen/Qwen2.5-3B-Instruct",                  # Qwen 2.5 3B Instruct — strong few-shot instruction model; efficient PEFT adaptation for cross-domain prompt transfer
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",       # SmolLM2 1.7B — smallest instruction-tuned LLM; plug-in semantic knowledge source for on-device prompt learning
    "llava-hf/llava-1.5-7b-hf",                  # LLaVA 1.5 7B — LLaMA + CLIP projection; generates rich visual descriptions AND can serve as the classification backbone
    "Salesforce/blip2-flan-t5-xxl",              # BLIP-2 Flan-T5-XXL — Q-Former bridges vision to Flan-T5 XXL; best for prompt-conditioned few-shot visual classification
],
    "Sample_229":  [
    "google/flan-t5-large",
    "google/flan-t5-xl",
    "google/flan-t5-xxl",
    "google/t5-v1_1-large",
    "google/t5-v1_1-xl",
    "google/ul2",
    "bigscience/mt0-large",
    "bigscience/mt0-xl",
    "google/mt5-large",
    "allenai/unifiedqa-t5-large"
],
    "Sample_230": [
    "google/vit-base-patch16-224",
    "google/vit-large-patch16-224",
    "facebook/deit-base-distilled-patch16-224",
    "facebook/deit-small-patch16-224",
    "microsoft/swin-base-patch4-window7-224",
    "microsoft/swin-large-patch4-window7-224",
    "microsoft/beit-base-patch16-224",
    "microsoft/beit-large-patch16-224",
    "facebook/dino-vitb16",
    "facebook/dinov2-base"
],
    "Sample_231": [
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "michiyasunaga/BioLinkBERT-base",
    "michiyasunaga/BioLinkBERT-large",
    "allenai/specter2_base",
    "allenai/specter2",
    "NeuML/pubmedbert-base-embeddings",
    "pritamdeka/S-PubMedBert-MS-MARCO",
    "sentence-transformers/allenai-specter",
    "dmis-lab/biobert-base-cased-v1.2",
    "microsoft/BioGPT"
],
    "Sample_237": [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-MiniLM-L12-v2",
    "sentence-transformers/paraphrase-MiniLM-L6-v2",
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    "sentence-transformers/multi-qa-distilbert-cos-v1",
    "intfloat/e5-small-v2",
    "intfloat/e5-base-v2",
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "thenlper/gte-small"
],
    "Sample_246": [
    "runwayml/stable-diffusion-v1-5",
    "stabilityai/stable-diffusion-2-1",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    "stabilityai/stable-diffusion-2-base",
    "CompVis/stable-diffusion-v1-4",
    "stabilityai/stable-diffusion-2-depth",
    "lllyasviel/control_v11p_sd15_canny",
    "lllyasviel/control_v11p_sd15_openpose",
    "TencentARC/t2i-adapter-canny-sd15"
],
    "Sample_247": [
    "runwayml/stable-diffusion-inpainting",
    "stabilityai/stable-diffusion-2-inpainting",
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "kandinsky-community/kandinsky-2-2-decoder-inpaint",
    "kandinsky-community/kandinsky-2-1-inpaint",
    "lllyasviel/control_v11p_sd15_inpaint",
    "lllyasviel/control_v11p_sd15_canny",
    "lllyasviel/control_v11f1p_sd15_depth",
    "TencentARC/t2i-adapter-sketch-sdxl-1.0"
],
    "Sample_264": [
    "Salesforce/blip-image-captioning-base",
    "Salesforce/blip-image-captioning-large",
    "Salesforce/blip2-opt-2.7b",
    "Salesforce/blip2-flan-t5-xl",
    "nlpconnect/vit-gpt2-image-captioning",
    "microsoft/git-base-coco",
    "microsoft/git-large-coco",
    "google/pix2struct-base",
    "IDEA-CCNL/Taiyi-ViT-GPT2-Captioning",
    "ydshieh/vit-gpt2-coco-en"
],
    "Sample_265": [
    "google/flan-t5-base",
    "google/flan-t5-large",
    "google/flan-t5-xl",
    "google/flan-t5-xxl",
    "Salesforce/blip2-flan-t5-xl",
    "Salesforce/blip2-flan-t5-xxl",
    "google/t5-v1_1-base",
    "google/t5-v1_1-large",
    "allenai/unifiedqa-t5-base",
    "allenai/unifiedqa-t5-large"
],
    "Sample_268": [
    "bert-base-uncased",
    "bert-base-cased",
    "bert-large-uncased",
    "roberta-base",
    "roberta-large",
    "distilbert-base-uncased",
    "microsoft/deberta-v3-base",
    "microsoft/deberta-v3-large",
    "google/electra-base-discriminator",
    "google/electra-small-discriminator"
],
    "Sample_281": [
    "allenai/scibert_scivocab_uncased",
    "allenai/scibert_scivocab_cased",
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "emilyalsentzer/Bio_ClinicalBERT",
    "dmis-lab/biobert-base-cased-v1.1",
    "gsarti/scibert-nli",
    "allenai/specter",
    "allenai/specter2_base",
    "sentence-transformers/allenai-specter",
    "pritamdeka/S-PubMedBert-MS-MARCO"
],
    "Sample_290": [
    "facebook/bart-large-cnn",
    "google/pegasus-cnn_dailymail",
    "google/pegasus-xsum",
    "allenai/led-base-16384",
    "allenai/led-large-16384",
    "facebook/bart-large-xsum",
    "google/long-t5-tglobal-base",
    "google/long-t5-tglobal-large",
    "allenai/primera-multi_lexsum-source-tiny",
    "allenai/primera-multi_lexsum-source"
],
    "Sample_292": [
    "Salesforce/blip-image-captioning-base",
    "Salesforce/blip-image-captioning-large",
    "Salesforce/blip2-flan-t5-xl",
    "Salesforce/blip2-opt-2.7b",
    "microsoft/git-large-coco",
    "nlpconnect/vit-gpt2-image-captioning",
    "google/pix2struct-base",
    "IDEA-CCNL/Taiyi-ViT-GPT2-Captioning",
    "ydshieh/vit-gpt2-coco-en",
    "Salesforce/instructblip-flan-t5-xl"
],
    "Sample_293": [
    "microsoft/layoutlmv3-base",
    "microsoft/layoutlmv3-large",
    "microsoft/layoutlm-base-uncased",
    "microsoft/layoutlm-large-uncased",
    "microsoft/layoutxlm-base",
    "microsoft/dit-base",
    "microsoft/dit-large",
    "naver-clova-ix/donut-base",
    "naver-clova-ix/donut-base-finetuned-rvlcdip",
    "microsoft/trocr-base-handwritten"
],
    "Sample_296": [
    "google/flan-t5-base",
    "google/flan-t5-large",
    "google/flan-t5-xl",
    "google/flan-t5-xxl",
    "google/long-t5-tglobal-base",
    "google/long-t5-tglobal-large",
    "allenai/led-base-16384",
    "allenai/led-large-16384",
    "facebook/bart-large-cnn",
    "mistralai/Mistral-7B-Instruct-v0.3"
]
}

GEMINI_RECOMMENDATIONS = {
"sample_1": [
  "mistralai/Codestral-22B-v0.1-Embed",
  "jinaai/jina-embeddings-v3-code",
  "Salesforce/codet5p-110m-embedding",
  "BAAI/bge-m3",
  "microsoft/graphcodebert-base",
  "microsoft/unixcoder-base",
  "nomic-ai/nomic-embed-code-v1.5",
  "codefuse-ai/C2LLM-0.5B-Code-Embed",
  "neulab/codebert-python",
  "voyageai/voyage-code-3"
],
    "sample_2": [
  "yikuan8/Clinical-Longformer",
  "yikuan8/Clinical-BigBird",
  "Uf-Habi/GatorTron-base",
  "answerdotai/ModernBERT-base-clinical",
  "emilyalsentzer/Bio_ClinicalBERT",
  "monologg/bio-electra-base-discriminator",
  "nlpie/bio-longformer-base-4096",
  "GatorTron/GatorTron-S",
  "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
  "ProsusAI/clinical-long-bert"
],
    "sample_7": [
  "ubc-nlp/araT5v2-base-msa",
  "moussaKamal/AraBART",
  "ubc-nlp/araT5-base",
  "UBC-NLP/araT5v2-large-msa",
  "asafaya/bert-base-arabic-summarization",
  "rezaeish/Zaman-Arabic-Summarization",
  "yalsaffar/mt5-small-Arabic-Summarization",
  "Aoun/AraBART-Summarization",
  "ML6team/arabart-summarization-arabic-news",
  "tiiuae/falcon-h1-arabic-3b"
],
    "sample_8": [
  "facebook/nllb-200-3.3B",
  "facebook/nllb-200-distilled-1.3B",
  "facebook/seamless-m4t-v2-large",
  "facebook/m2m100_1.2B",
  "google/mt5-large",
  "facebook/mbart-large-50-many-to-many-mmt",
  "facebook/nllb-moe-54b",
  "google/madlad-400-3b-mt",
  "facebook/seamless-m4t-medium",
  "Helsinki-NLP/opus-mt-ar-en"
],
    "sample_9": [
  "google/umt5-base",
  "google/umt5-xl",
  "facebook/nllb-200-distilled-1.3B",
  "facebook/nllb-200-3.3B",
  "google/mt5-base",
  "google/mt5-large",
  "google/madlad-400-3b-mt",
  "facebook/seamless-m4t-v2-large",
  "facebook/m2m100_1.2B",
  "facebook/mbart-large-50-many-to-many-mmt"
],
    "sample_18": [
  "google/flan-t5-large",
  "facebook/bart-large",
  "google/t5-gemma-2b-it",
  "meta-llama/Llama-3.2-3B-Summarizer",
  "google/pegasus-large",
  "allenai/led-large-16384",
  "microsoft/phi-4-mini-instruct",
  "deepseek-ai/DeepSeek-V3-Small-EncDec",
  "Baidu/ernie-3.5-summarize",
  "nvidia/nemotron-3-8b-summarize"
],
    "sample_19": [
  "deepseek-ai/DeepSeek-V3",
  "Qwen/Qwen3-235B-A22B-Instruct",
  "THUDM/glm-4.5-355b",
  "01-ai/Yi-1.5-34B",
  "internlm/internlm3-72b-instruct",
  "baichuan-inc/Baichuan4-Base",
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
  "internlm/internlm-xcomposer3-medical",
  "ZhipuAI/GLM-4-9B-Chat",
  "Qwen/Qwen2.5-72B-Instruct"
],
    # "sample_21": [],
    # "sample_22": [],
    # "sample_23": [],
    "sample_24": [
  "google/vit-base-patch16-224-in21k",
  "facebook/deit-base-distilled-patch16-224",
  "google/vit-small-patch16-224",
  "microsoft/swin-tiny-patch4-window7-224",
  "facebook/dinov2-base",
  "google/vit-huge-patch14-224-in21k",
  "facebook/convnext-tiny-224",
  "apple/mobilevit-small",
  "google/vit-base-patch32-224-in21k",
  "microsoft/beit-base-patch16-224-pt22k"
],
    # "sample_26": [],
    # "sample_27": [],
    # "sample_28": [],
    # "sample_34": [],
    # "sample_46": [],
    "sample_55": [
  "Alibaba-NLP/gte-Qwen2.5-7B-instruct",
  "BAAI/bge-multilingual-gemma2-9b",
  "google/gemma-3-27b-embeddings",
  "nomic-ai/nomic-embed-text-v1.5",
  "intfloat/e5-mistral-7b-instruct",
  "Salesforce/SFR-Embedding-2_R",
  "nvidia/llama-3.1-nemotron-8b-embed",
  "infgrad/stella-en-1.5B-v5",
  "BAAI/bge-m3",
  "sentence-transformers/all-mpnet-base-v2"
],
    "sample_56": [
  "google/siglip2-so400m-patch14-384",
  "microsoft/LLM2CLIP-EVA02-L-14-336",
  "nvidia/PS3_Lang-1.5K-SigLIP2",
  "laion/CLIP-ViT-bigG-14-laion2B-39b-b160k",
  "google/siglip-so400m-patch14-384",
  "BAAI/EVA-02-CLIP-L-14-336",
  "apple/MobileCLIP-B-LT",
  "timm/vit_relpos_medium_patch16_224.sw_in1k",
  "facebook/metaclip-h14-fullcc2.5b",
  "amazon/nova-multimodal-embeddings-v1"
],
    "sample_57": [
  "google/siglip2-so400m-patch14-384",
  "microsoft/LLM2CLIP-EVA02-L-14-336",
  "google/siglip2-base-patch16-224",
  "microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned",
  "BAAI/EVA-02-CLIP-L-14-336",
  "facebook/metaclip-h14-fullcc2.5b",
  "google/siglip-so400m-patch14-384",
  "microsoft/LLM2CLIP-Openai-L-14-336",
  "laion/CLIP-ViT-bigG-14-laion2B-39b-b160k",
  "apple/MobileCLIP-B-LT"
],
    "sample_58": [
  "TencentARC/t2i-adapter-canny-sdxl-1.0",
  "TencentARC/t2i-adapter-sketch-sdxl-1.0",
  "TencentARC/t2i-adapter-color-sdxl-1.0",
  "bdsqlsz/qinglong_controlnet-lllite-sdxl-canny",
  "bdsqlsz/qinglong_controlnet-lllite-sdxl-recolor",
  "xinsir/controlnet-union-sdxl-1.0",
  "StabilityAI/control-lora-canny-rank128",
  "StabilityAI/control-lora-recolor-rank128",
  "h94/IP-Adapter-FaceID",
  "TencentARC/t2i-adapter-lineart-sdxl-1.0"
],
    "sample_59": [
  "madebyollin/sdxl-vae-fp16-fix",
  "stabilityai/sdxl-vae",
  "nubby/blessed-sdxl-vae-fp16-fix",
  "KBlueLeaf/EQ-SDXL-VAE",
  "fear-factory/sdxl-vae-fp16-fix-safetensors",
  "Amsis/sdxl-vae-fp16-fix-official",
  "black-forest-labs/FLUX.1-schnell-vae",
  "stabilityai/sd3-vae",
  "fal/vae-16-channel-open-weights",
  "Ostris/vae-16ch-speed-optimized"
],
    "sample_60": [
  "ByteDance/Hyper-SD",
  "h1t/TCD-SDXL-LoRA",
  "ByteDance/SDXL-Lightning",
  "latent-consistency/lcm-lora-sdxl",
  "latent-consistency/lcm-lora-sdv1-5",
  "ByteDance/Hyper-SD15",
  "stabilityai/sdxl-turbo",
  "Kijai/Hyper-FLUX.1-dev-8step-LoRA",
  "lucataco/SDXL-Turbo-LoRA",
  "wangfuyun/PCM-SDXL"
],
    "sample_61": [
  "madebyollin/sdxl-vae-fp16-fix",
  "stabilityai/sdxl-vae",
  "nubby/blessed-sdxl-vae-fp16-fix",
  "ai-art-lab/sdxl-vae-fix",
  "wangkanai/sdxl-vae",
  "shashank-mishra/sdxl-vae-fp16",
  "snowkidy/sdxl-vae-fp16-fix",
  "ostris/vaexl",
  "fal/sdxl-vae-16-channel",
  "thingthatis/sdxl-vae-fp16-fix"
],
    "sample_62": [
    "fancyfeast/SUPIR-v0Q",                    # High-fidelity photo-realistic restoration
    "lllyasviel/control_v11f1e_sd15_tile",     # Foundation for most tiled upscaling workflows
    "Iceclear/StableSR",                       # Standard for SD-based latent modulation
    "bdsqlsz/qinglong_controlnet-lllite-tile", # SDXL-ready lightweight tiled control
    "stabilityai/stable-diffusion-x4-upscaler",# Reliable native latent diffusion upscaler
    "Apple/Alu-LDM-SR",                        # High-efficiency latent super-resolution
    "TencentARC/GFPGANv1.4-Diffusion",         # Specialized for face SR within latent pipelines
    "ByteDance/SDXL-Lightning-4step",          # Rapid inference for real-time upscaling
    "X-Adapter/SUPIR-v0-paper",                # The research-grade model for deep restoration
    "lllyasviel/sd_control_collection"         # General collection including tile/resample units
],
    "sample_68": [
  "answerdotai/ModernBERT-large",
  "microsoft/deberta-v3-large",
  "microsoft/deberta-v2-xxlarge",
  "google/electra-large-discriminator",
  "answerdotai/ModernBERT-base",
  "FacebookAI/roberta-large",
  "microsoft/mdeberta-v3-base",
  "deepset/roberta-large-squad2",
  "Intel/dynamic_tinybert",
  "BAAI/bge-reranker-v2-m3"
],
    "sample_69": [
  "answerdotai/ModernBERT-base",
  "microsoft/deberta-v3-small",
  "microsoft/deberta-v3-xsmall",
  "huawei-noah/TinyBERT_General_4L_312D",
  "google/electra-small-discriminator",
  "answerdotai/ModernBERT-small",
  "distilbert/distilroberta-base",
  "cross-encoder/ms-marco-MiniLM-L-6-v2",
  "sentence-transformers/all-MiniLM-L12-v2",
  "albert/albert-base-v2"
],
"sample_86":[
    "facebook/wav2vec2-tiny-240ms",           # Optimized for low-latency streaming
    "microsoft/wavlm-base-plus",               # Best-in-class for speech/non-speech audio features
    "nvidia/canary-1b",                        # High-efficiency FastConformer based encoder
    "openai/whisper-tiny-en",                  # Standard lightweight encoder for semantic features
    "efficient-speech/lite-whisper-tiny",      # Pruned Whisper variant for edge device inference
    "mistralai/Voxtral-Mini-4B-Realtime-2602", # Natively streaming encoder with <500ms delay
    "facebook/encodec_24khz",                  # High-fidelity neural audio compression features
    "google/moonshine-tiny",                   # Built specifically for ultra-low latency mobile use
    "MIT/ast-tiny-patch16-224",               # Audio Spectrogram Transformer for non-speech sounds
    "nvidia/parakeet-tdt-1.1b"                 # Extreme throughput for real-time feature extraction
],
    "sample_90":[
    "microsoft/deberta-v3-large",           # Current SOTA for NLU classification
    "roberta-large",                        # The essential industry-standard baseline
    "microsoft/deberta-v2-xxlarge",         # Highest performing pure-encoder for reasoning
    "google/electra-large-discriminator",   # Efficient discriminator-based pre-training
    "google/t5-large",                      # Versatile encoder-decoder for rank-based scoring
    "FacebookAI/xlm-roberta-large",         # Top-tier multilingual encoder performance
    "allenai/longformer-large-4096",        # Necessary for long-document context QA
    "bigscience/mt0-large",                 # Instruction-tuned multilingual benchmark
    "albert/albert-xxlarge-v2",                    # High-capacity reasoning via parameter sharing
    "google-bert/bert-base-uncased"                     # Essential baseline for performance delta tracking
],
    
    "sample_98": [
    "allenai/scibert_scivocab_uncased",
    "dmis-lab/biobert-v1.1",
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "monologg/bio-electra-small-discriminator",
    "deepset/covid_bert_base",
    "FacebookAI/roberta-large",
    "kaist-ai/bio-roberta-v2",
    "allenai/biomed_roberta_base",
    "neutral-parity/scibert-finetuned-abbreviation",
    "google/bert_uncased_L-12_H-768_A-12"
],
    "sample_99": [
    "emilyalsentzer/Bio_ClinicalBERT",           # The classic MIMIC-III trained baseline
    "GanjinZero/UMLSBert_Clinical",              # Infused with UMLS knowledge graph for semantic medical grounding
    "microsoft/BiomedNLP-BiomedBERT-base",       # Trained from scratch on PubMed and PMC (excellent for clinical literature)
    "yikuan8/Clinical-Longformer",               # MIMIC-trained for long clinical notes (up to 4,096 tokens)
    "dmis-lab/biobert-v1.1",                     # Robust biomedical baseline often used in clinical pipelines
    "clinfo/ClinicalBERT",                       # Specifically trained on MIMIC-III intensive care data
    "Physionet/clinical-roberta-base",           # RoBERTa architecture trained on the PhysioNet/MIMIC ecosystem
    "bvanaken/clinical-assertion-negation-bert", # Specialized for understanding clinical assertions and nuances
    "nlpie/bio-electra-base-discriminator",      # Highly efficient for clinical text classification and similarity
    "Clinical-AI-Lab/Clinical-DeBERTa"           # Advanced DeBERTa-v3 architecture applied to clinical data
],
    "sample_116" : [
    "cross-encoder/nli-deberta-v3-large",        # Current SOTA for accuracy in NLI tasks
    "cross-encoder/nli-deberta-v3-base",         # Best balance of speed and performance
    "cross-encoder/nli-distilroberta-base",     # Faster, lightweight alternative for real-time NLU
    "cross-encoder/nli-MiniLM2-L6-H384",        # Highly efficient for high-throughput pipelines
    "facebook/bart-large-mnli",                 # Classic, robust model for zero-shot and NLI
    "MoritzLaurer/DeBERTa-v3-large-mnli-snli",  # Fine-tuned on massive combined NLI datasets
    "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",  # Best for multilingual semantic similarity
    "symanto/snli-roberta-base-snli",           # Specialized for the SNLI dataset logic
    "typeform/distilbert-base-uncased-mnli",    # Ultra-fast, distilled version for simple tasks
    "GritLM/GritLM-7B"                          # 2026-era unified embedding/NLI generative model
],
"sample_146" : [
    "BAAI/bge-large-en-v1.5",            # The industry standard for robust English retrieval
    "Alibaba-NLP/gte-Qwen2-7B-instruct",  # Top-tier instruction-following dense retriever
    "intfloat/e5-mistral-7b-instruct",   # Excellent zero-shot generalization via LLM backbone
    "GritLM/GritLM-7B",                  # Unified model for both generation and embedding
    "infgrad/stella-base-en-v2",         # Highly efficient with state-of-the-art MTEB scores
    "BAAI/bge-m3",                       # Multi-lingual, Multi-function, Multi-granularity
    "nomic-ai/nomic-embed-text-v1.5",    # Matryoshka-enabled (flexible dimensions) with 8k context
    "sentence-transformers/all-mpnet-base-v2", # Reliable, lightweight baseline for dialogue
    "google/gecko-7b-v1",                # Distilled from Gemini for high-performance retrieval
    "mixedbread-ai/mxbai-embed-large-v1" # Optimized for RAG and unseen domain adaptation
],
    "sample_130": [
    "microsoft/deberta-v3-large",           # Uses Disentangled Attention & Relative Position Bias (SOTA)
    "microsoft/deberta-v3-base",            # High-efficiency version of the DeBERTa-v3 architecture
    "google/t5-v1_1-large",                 # T5 uses simplified relative position buckets for every layer
    "google/t5-large",                      # Classic Enc-Dec with per-layer relative bias
    "bigscience/mt0-xl",                    # Instruction-tuned T5 variant with relative position bias
    "microsoft/deberta-v2-xxlarge",         # 1.5B params; the most powerful relative-position encoder
    "Qwen/Qwen2.5-7B-Instruct",             # Uses Rotary Positional Embeddings (RoPE) for better relative logic
    "meta-llama/Llama-3.1-8B",              # Uses RoPE; excellent for relative positioning in long contexts
    "allenai/longformer-base-4096",         # Uses a sliding window with relative attention for long docs
    "microsoft/phi-4"                       # Modern lightweight model with sophisticated RoPE implementation
],
    "sample_134": [
    "facebook/genre-kilt",             # SOTA for GEL; fine-tuned BART for KILT datasets
    "facebook/genre-linking-blink",    # Fine-tuned BART specifically for BLINK/Wikipedia
    "facebook/bart-large",             # The original denoising seq2seq powerhouse for fine-tuning
    "google/t5-v1_1-large",            # Improved T5 with GLU activations; excellent for text-to-text
    "google/flan-t5-large",            # T5 fine-tuned on instructions; superior at following linking prompts
    "facebook/mbart-large-50",         # Multilingual BART; best for cross-lingual entity linking
    "google/mt5-large",                # Multilingual T5; handles 100+ languages for global linking tasks
    "facebook/bart-base",              # Lightweight version for faster inference and benchmarking
    "google/t5-large",                 # Classic T5; the foundation for many generative linking papers
    "Babelscape/rebel-large"           # BART-based model for joint Entity Linking and Relation Extraction
],
    "sample_135": [
    "facebook/bart-base",             # 140M params: The standard baseline for "small-scale" experiments
    "facebook/bart-large",            # 400M params: The "large-scale" standard for BART-style architectures
    "facebook/mbart-large-cc25",      # 610M params: Multilingual BART variant for cross-lingual scaling
    "facebook/mbart-large-50",        # 610M params: Updated mBART with extended language support
    "slauw87/bart_summarisation",     # Base-scale BART fine-tuned on CNN/DailyMail (Task-specific comparison)
    "ainize/bart-base-en-summarization", # Alternative base-scale for specific NLG task performance tracking
    "YituTech/conv-bart-base",        # Variant with convolutional layers to test architectural generalization
    "vblagoje/bart_lfqa",             # Large-scale BART optimized for Long-form QA tasks
    "Babelscape/rebel-base",          # Base-scale BART-style model for relational/entity-based generation
    "Babelscape/rebel-large"          # Large-scale counterpart to Rebel-Base for rigorous scaling evaluation
],
    "sample_136": [
    "facebook/bart-large",                # Standard backbone for expertise separation
    "google/pegasus-large",               # Specialized in salient sentence extraction
    "google/flan-t5-xl",                  # Superior out-of-domain/zero-shot capability
    "microsoft/phi-4",                    # SOTA 2026 reasoning-base for MoE
    "google/bigbird-pegasus-large-arxiv", # Handling long-context domain experts
    "facebook/mbart-large-50",            # Multilingual summarization base
    "allenai/led-large-16384",            # Long-document summarization support
    "google/t5-v1_1-large",               # High-capacity base for custom pre-training
    "seongminkang/bart-news-summarization",# Pre-specialized news expert baseline
    "sshleifer/distilbart-cnn-12-6"       # Efficiency-focused benchmarking base
],
    "sample_158": [
    "sentence-transformers/all-mpnet-base-v2",       # Best all-around for semantic similarity and clustering
    "sentence-transformers/all-MiniLM-L6-v2",       # Ultra-fast; perfect for real-time state tracking
    "BAAI/bge-large-en-v1.5",                       # State-of-the-art for retrieval and state-mapping
    "google/t5-v1_1-large",                         # Use the encoder for high-capacity state representations
    "facebook/bart-large-mnli",                     # Excellent for mapping states to specific aspect "labels"
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", # Best for tracking states across languages
    "thenlper/gte-large",                           # High-density embeddings for precise similarity computations
    "Salesforce/codet5-base",                       # Strong if dialogue states contain structured data/schema
    "Todd-Garris/Dialogue-State-Tracker",           # A community-specialized model for DST tasks
    "sentence-transformers/multi-qa-mpnet-base-dot-v1" # Optimized for state-tracking formatted as QA pairs
],
    "Sample_177": [
    "deepseek-ai/DeepSeek-V3.2",            # SOTA MoE generalist; high coding/math efficiency
    "deepseek-ai/DeepSeek-R1",               # Specialized reasoning model with native CoT
    "ZhipuAI/GLM-4.7-9b-instruct",           # 2026's leader for small-model math/reasoning
    "ZhipuAI/GLM-5-744B-instruct",           # Flagship model with top HumanEval/AIME scores
    "Alibaba-NLP/Qwen3-Next-Thinking",       # Hybrid MoE with toggleable reasoning mode
    "meta-llama/Llama-4-Maverick-400B",      # Meta's 2026 flagship with strong logic/code
    "moonshotai/Kimi-K2.5-Thinking",         # Leader in LiveCodeBench (real-world coding)
    "mistralai/Mistral-Large-2601",          # High-reasoning density; excellent at math logic
    "google/gemma-4-27b-it",                 # Best-in-class reasoning for its parameter size
    "MiniMax/MiniMax-M2.5-230B"              # Optimized for agentic coding and SWE tasks
],
    "Sample_178": [
    "Alibaba-NLP/Qwen3-Embedding-8B",       # 2026 SOTA: Top MTEB score with 32k context
    "BAAI/bge-m3",                         # Multi-modal/Multi-lingual/Multi-function leader
    "intfloat/e5-mistral-7b-instruct",     # High generalization via instruction tuning
    "nomic-ai/nomic-embed-text-v1.5",      # Matryoshka-enabled for scalable web-scale indexing
    "infgrad/stella-base-en-v2",           # Best performance-to-latency ratio (base size)
    "mixedbread-ai/mxbai-embed-large-v1",   # Optimized specifically for RAG & unseen domains
    "google/gemini-embedding-v1",          # High-stability retrieval for complex web queries
    "jinaai/jina-embeddings-v5-small",     # Ultra-efficient for high-throughput pipelines
    "BAAI/bge-large-en-v1.5",              # The "Gold Standard" baseline for production QA
    "Snowflake/snowflake-arctic-embed-l"    # Excellent for enterprise/web document retrieval
],
    "Sample_179": [
    "microsoft/swin-tiny-patch4-window7-224",        # Hierarchical via shifted windows, industry standard for MQA backbones.
    "facebook/hiera-base-224",                      # Simplified hierarchical ViT, extremely efficient and fast.
    "nvidia/segformer-b0-finetuned-ade-512-512",    # Hierarchical transformer that handles multi-scale features without positional embeddings.
    "facebook/dinov2-base",                         # Self-supervised ViT; though not natively hierarchical in layers, it provides dense multi-scale features.
    "apple/mobilevit-small",                        # Hybrid CNN-Transformer; excellent for efficient processing on edge-compatible MQA.
    "OpenGVLab/pvt_v2_b2",                          # Pyramid Vision Transformer v2; designed specifically for dense prediction and hierarchical mapping.
    "google/vit-hybrid-base-bit-384",               # Uses a ResNet backbone for initial hierarchical feature extraction before the transformer layers.
    "facebook/convnext-tiny-224",                   # Purely convolutional but hierarchical; often paired with text models for robust multimodal fusion.
    "microsoft/beit-base-patch16-224-pt22k",        # Vision Transformer with masked image modeling, provides strong semantic representations for QA.
    "timm/maxvit_rmlp_small_rw_224.sw_in1k"         # Multi-Axis ViT; scales to high resolutions by combining local and global attention.
],
    "Sample_184": [
    "microsoft/swin-tiny-patch4-window7-224",        # The gold standard for hierarchical ViTs; uses shifted windows.
    "microsoft/swinv2-base-patch4-window12-192-22k", # V2 supports much higher resolutions and more stable training.
    "facebook/hiera-base-224",                      # Ultra-fast hierarchical model; strips 'bells & whistles' for speed.
    "nvidia/segformer-b0-finetuned-ade-512-512",    # Hierarchical and 'position-encoding free'; handles varying resolutions easily.
    "OpenGVLab/pvt_v2_b2",                          # Pyramid Vision Transformer; specifically designed for dense visual tasks.
    "apple/mobilevit-small",                        # Combines CNN hierarchies with Transformer global reasoning for efficiency.
    "facebook/convnext-tiny-224",                   # Though a ConvNet, it mimics ViT behavior and is a top MQA backbone choice.
    "google/vit-hybrid-base-bit-384",               # Uses a ResNet stage first to create a spatial hierarchy before the ViT blocks.
    "timm/maxvit_rmlp_small_rw_224.sw_in1k",        # Multi-axis attention allows it to scale to high-res inputs efficiently.
    "microsoft/beit-base-patch16-224-pt22k"         # Masked pre-training makes it excellent at understanding fine-grained semantics.
],
    "Sample_194": [
    "facebook/bart-base",            # The essential baseline for summarization research; easy to compress.
    "facebook/bart-large-cnn",      # Heavily used for summarization benchmarks; a must-have baseline.
    "google-t5/t5-small",            # Extremely lightweight; the go-to for testing compression/distillation.
    "google-t5/t5-base",             # The standard "middle-ground" for general NLG research.
    "google/t5-v1_1-base",           # An improved version of T5 with GeGLU activations, better for pre-training research.
    "Helsinki-NLP/opus-mt-en-ro",    # The MarianMT series; the standard for efficient, task-specific translation.
    "facebook/mBART-large-50",       # Multilingual BART; ideal for cross-lingual summarization and translation.
    "google/flan-t5-base",           # T5 fine-tuned on instruction tasks; better zero-shot performance for generation.
    "microsoft/prophetnet-large-uncased", # Specifically designed for future n-gram prediction in summarization.
    "sshleifer/distilbart-cnn-12-6"  # A pre-compressed (distilled) BART model; great for studying student-teacher dynamics.
],
    "Sample_212": [
    "google/flan-t5-xl",              # Best-in-class for zero-shot instruction following; the primary choice for unified evaluators.
    "google/flan-t5-large",           # A more compute-efficient version of Flan-T5 that still maintains strong Boolean QA performance.
    "microsoft/deberta-v3-large",     # While an encoder, it is the 'SOTA' baseline for NLI/entailment-based evaluation (e.g., QAFactEval).
    "facebook/bart-large-mnli",       # Specifically fine-tuned for Multi-Genre NLI; excellent for checking "Is sentence A implied by B?".
    "google/flan-ul2",                # A massive 20B parameter unified model that excels at complex reasoning required for deep evaluation.
    "Salesforce/codet5p-220m",        # If evaluating code generation; specialized for structured text-to-text evaluation.
    "navteca/bart-large-mnli",        # A popular, alternative optimization of BART for zero-shot classification and evaluation.
    "google-t5/t5-3b",                # A standard large-scale text-to-text baseline for building custom evaluation projectors.
    "yuchenlin/UniEval-summarization",# A model specifically pre-trained to be a unified evaluator for summarization dimensions.
    "facebook/mbart-large-50-many-to-many-mmt" # The standard for building unified evaluators in multilingual settings.
],
    "Sample_213": [
    "Qwen/Qwen2.5-VL-7B-Instruct",      # State-of-the-art for reasoning; handles dynamic resolutions and long videos.
    "google/paligemma2-10b-mix-448",    # Specifically designed for transfer learning; excels in low-resource adaptation.
    "mistralai/Mistral-7B-v0.3",        # The 'gold standard' 7B backbone for many VLMs due to its dense reasoning capabilities.
    "meta-llama/Llama-3.2-11B-Vision-Instruct", # Meta's native multimodal model; highly optimized for image-text alignment.
    "microsoft/Phi-3.5-vision-instruct", # 4.2B parameters; ultra-efficient for high-res reasoning in low-compute settings.
    "openbmb/MiniCPM-V-2_6",            # 8B parameters; achieves GPT-4V level performance in a compact, trainable footprint.
    "deepseek-ai/deepseek-vl2-small",   # Mixture-of-Experts (MoE) architecture; provides high capacity with low active parameters.
    "HuggingFaceTB/SmolVLM2-2.2B-Instruct", # Best for extreme low-resource settings; high-resolution support in a tiny package.
    "llava-hf/llava-onevision-qwen2-7b-ov-hf", # A unified model combining SigLIP and Qwen2; great for single/multi-image tasks.
    "Baidu/VIMER-internvideo2-7b"       # Optimized for temporal reasoning if your vision system includes video inputs.
],
    "Sample_216": [
    "stabilityai/stable-diffusion-2-1",         # The definitive standard; ideal for establishing standard baseline safety metrics.
    "stabilityai/stable-diffusion-xl-base-1.0", # The SOTA 'large' baseline; essential for testing robustness against complex prompt injection.
    "runwayml/stable-diffusion-v1-5",            # The 'classic' model; widely studied and best for historical robustness comparisons.
    "segmind/SSD-1B",                            # Distilled version of SDXL; useful for testing if model compression impacts safety or bias.
    "justinpinkney/pokemon-stable-diffusion",  # A style-tuned model; important for testing how safety alignment transfers (or degrades) during fine-tuning.
    "nota-ai/bk-sdm-small",                    # A heavily compressed (block-pruned) model; designed for resource-constrained safety auditing.
    "prompthero/openjourney-v4",               # A popular style-finetuned model, useful for analyzing vulnerabilities in models optimized for specific aesthetic outputs.
    "Wauplin/space-diffusion",                # Example of a highly domain-specific fine-tune (space themes) to check if specialization introduces unique failure modes.
    "OFA-Sys/small-stable-diffusion-v0",       # A 'tiny' Stable Diffusion; perfect for high-throughput, automated fuzzing or safety testing on limited hardware.
    "cyberdelia/DreamBooth-Babe"                 # A model created using DreamBooth; specifically useful for testing concepts related to personalization safety and identity bias.
],
    "Sample_217": [
    "openai/clip-vit-base-patch32",           # The industry standard baseline for image-text similarity (CLIPScore).
    "openai/clip-vit-large-patch14",         # Higher resolution and better performance for fine-grained alignment.
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", # Trained on 2B images; the most robust 'open' CLIP variant for research.
    "google/siglip-base-patch16-224",        # Uses a sigmoid loss for better performance on zero-shot and retrieval tasks.
    "google/siglip-so400m-patch14-384",      # SOTA for vision-language alignment; handles higher resolutions (384px).
    "facebook/align-base",                   # A dual-encoder alternative to CLIP that uses a different pre-training objective.
    "facebook/metaclip-h14-fullcc2.5b",      # Optimized training data to fix alignment biases found in original CLIP.
    "Salesforce/blip-itm-base-coco",         # Specifically fine-tuned for Image-Text Matching (ITM) and alignment.
    "Salesforce/blip2-opt-2.7b",             # Uses a Q-Former to project visual features; excellent for complex reasoning alignment.
    "microsoft/vivit-b-16x2-kinetics400"      # For video-language alignment; provides spatio-temporal embeddings.
],
    "Sample_220": [
    "openai/clip-vit-base-patch32",           # The standard text encoder baseline for CLIP-based alignment.
    "openai/clip-vit-large-patch14",         # Higher capacity encoder; better for fine-grained semantic perception.
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", # Largest open-source CLIP encoder; excellent for robust, diverse prompts.
    "google/siglip-base-patch16-224",        # Text encoder using sigmoid loss; state-of-the-art for retrieval and classification.
    "google/siglip-so400m-patch14-384",      # High-resolution SigLIP encoder; superior for dense visual perception tasks.
    "facebook/metaclip-h14-fullcc2.5b",      # Optimized version of CLIP designed to reduce data bias and improve alignment accuracy.
    "CIDAS/clipseg-rd64-refined",            # CLIP text encoder fine-tuned specifically for text-to-image segmentation prompts.
    "Salesforce/blip-itm-base-coco",         # Encoder part of BLIP; focused on precise image-text matching (ITM).
    "BAAI/AltCLIP",                          # A multilingual CLIP encoder; aligns diverse languages with the same visual space.
    "timm/eva02_base_patch16_clip_224"       # EVA-CLIP variant; uses advanced pre-training for even stronger semantic feature extraction.
],
    "Sample_223": [
    "stabilityai/stable-diffusion-xl-base-1.0", # The definitive research backbone; best for studying core robustness.
    "black-forest-labs/FLUX.1-dev",             # The 2026 SOTA for open-weights; supports DreamBooth via advanced DiT fine-tuning.
    "stabilityai/stable-diffusion-3.5-large",   # Uses MMDiT; excellent for studying cross-attention robustness and text alignment.
    "runwayml/stable-diffusion-v1-5",            # The classic baseline; vital for comparing new adversarial attacks against legacy ones.
    "SG161222/RealVisXL_V4.0",                  # A photorealistic SDXL fine-tune; perfect for testing if realism impacts perturbation sensitivity.
    "stabilityai/sdxl-turbo",                   # An adversarial diffusion distilled (ADD) model; unique for testing 'few-step' robustness.
    "ByteDance/SDXL-Lightning-7bit",            # Highly optimized for speed; useful for high-throughput adversarial fuzzing.
    "Kandinsky-community/Kandinsky-3",           # A non-Stable Diffusion alternative (unclip-based); good for testing architectural transferability.
    "timbrooks/instruct-pix2pix",               # A model specialized in instruction-following; great for testing robustness to prompt-based edits.
    "diffusers/stable-diffusion-xl-template"],
    "Sample_224":  [
    "openai/clip-vit-large-patch14",         # The primary 'vision bridge' for prompt learning research.
    "google/siglip-so400m-patch14-384",      # High-performance VL bridge; superior for semantic alignment in low-shot tasks.
    "meta-llama/Llama-3.2-3B-Instruct",      # Ideal 'moderately sized' LLM to act as a semantic knowledge generator.
    "mistralai/Mistral-7B-v0.3",             # A strong reasoning baseline for generating class-specific descriptors.
    "google/flan-t5-xl",                     # Best-in-class for zero-shot 'knowledge extraction' via instruction prompts.
    "microsoft/Phi-3.5-mini-instruct",       # Efficient 3.8B model for low-compute semantic knowledge injection.
    "chavinlo/alpaca-native",                # A classic choice for instruction-based prompt generation in VL research.
    "OpenGVLab/InternViT-6B-448px-V1-5",     # A massive vision backbone that responds well to complex semantic prompts.
    "Salesforce/blip2-opt-2.7b",             # Already contains a Q-Former; perfect for studying parameter-efficient VL alignment.
    "facebook/metaclip-h14-fullcc2.5b"       # A 'cleaner' CLIP variant; useful for testing if better data improves prompt learning.
],
    "Sample_229": [
    "google/flan-t5-large",           # Best for low-resource; pre-trained on instructions to follow formatting rules.
    "facebook/bart-large",             # Strong denoising capabilities; excellent at correcting and restructuring messy inputs.
    "google-t5/t5-3b",                 # A larger baseline for complex multi-hop question decomposition.
    "tscholak/t5-11b-ssc",             # Fine-tuned for semantic parsing; highly effective at structured outputs.
    "Salesforce/codet5-base",          # Pre-trained on code; understands structural logic better than standard text models.
    "mrm8488/t5-base-finetuned-wikiSQL", # Specialized for natural language to SQL; a great starting point for structured QA.
    "Babelscape/rebel-large",          # Specifically designed for Relation Extraction; turns text into triplets (subject-predicate-object).
    "facebook/mBART-large-50",         # Use this if your complex questions come from multilingual sources.
    "google/t5-v1_1-base",             # A "purer" T5 (no multi-task mix) often used as a neutral research baseline.
    "valhalla/t5-base-qg"              # Originally for question generation, but excellent at learning question-specific syntax.
],
    "Sample_230": [
    "google/vit-base-patch16-224",           # The standard isotropic ViT; the essential baseline for prompt tuning.
    "microsoft/swin-tiny-patch4-window7-224", # Hierarchical ViT; tests if your method works with shifted window attention.
    "facebook/dinov2-base",                  # Self-supervised ViT; provides extremely robust features for adaptation.
    "facebook/deit-base-distilled-patch16-224", # Data-efficient Image Transformer; tests generalization to distilled architectures.
    "facebook/convnext-tiny-224",            # A "Transformer-like" CNN; crucial for testing if your method is architecture-agnostic.
    "OpenGVLab/pvt_v2_b2",                   # Pyramid Vision Transformer; a different approach to hierarchical spatial reduction.
    "nvidia/segformer-b0-finetuned-ade-512-512", # A ViT that lacks positional embeddings; tests reliance on spatial coordinates.
    "apple/mobilevit-small",                 # Hybrid CNN-Transformer; tests adaptation in resource-constrained, mobile-ready models.
    "microsoft/beit-base-patch16-224-pt22k", # Masked Image Modeling pre-trained; tests if pre-training objective affects tuning.
    "google/vit-large-patch32-384"           # A large-scale, high-res ViT; tests how your tuning method scales with model depth.
],
    "Sample_231": [
    "cambridgemlt/SapBERT-from-PubMedBERT-fulltext", # SOTA for medical entity linking; aligns synonyms in a shared vector space.
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext", # The gold standard 'PubMedBERT'; trained from scratch on medical text.
    "michigan-nlp/BioLinkBERT-base",              # Uniquely captures document relationships by training on citation links between papers.
    "dmis-lab/biobert-v1.1",                       # A foundational baseline; BERT weights further evolved on PubMed and PMC.
    "allenai/scibert_scivocab_uncased",            # Trained on a broad scientific corpus; excellent for cross-domain scientific tasks.
    "StanfordAIMI/RadBERT",                        # Specialized for radiology; captures specific terminology from clinical imaging reports.
    "GatorTron/GatorTron-base",                    # A massive clinical transformer trained on over 90 billion words of health records.
    "yikuan8/Clinical-Longformer",                 # Optimized for long-range dependencies in clinical notes and full-length papers.
    "NCBI/BioSentVec-PubMed-base",                 # Specialized for sentence-level embeddings to find similar medical studies or cases.
    "FacebookAI/roberta-base-biomedical-clinical" # A robust RoBERTa variant fine-tuned on the MIMIC-III and PubMed corpora.
],
    "Sample_237": [
    "BAAI/bge-m3",                      # Best overall: Multilingual, Multi-granularity, and Multi-functionality
    "sentence-transformers/all-MiniLM-L6-v2", # Industry standard for speed: extremely lightweight (80MB)
    "nomic-ai/nomic-embed-text-v1.5",   # High performance with Matryoshka embeddings (flexible dimensions)
    "mixedbread-ai/mxbai-embed-large-v1",# State-of-the-art retrieval performance on MTEB benchmarks
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", # Best for multilingual KGQA queries
    "google/electra-small-discriminator",# Highly efficient backbone for custom KGQA finetuning
    "jinaai/jina-embeddings-v3",        # Optimized for long-context and task-specific retrieval
    "intfloat/e5-small-v2",             # High efficiency with strong zero-shot retrieval capabilities
    "infgrad/stella-base-en-v2",        # Top-tier performance-to-size ratio for English semantic search
    "BAAI/bge-reranker-base"            # Essential second-stage model to re-rank top-K graph candidates
],
    "Sample_246": [
    "black-forest-labs/FLUX.1-dev",       # State-of-the-art DiT: unmatched prompt adherence and structure
    "black-forest-labs/FLUX.1-Kontext",   # Optimized for contextual editing and strong structural conditioning
    "stabilityai/stable-diffusion-3.5-large", # Multi-modal DiT: improved text rendering and composition
    "stabilityai/stable-diffusion-xl-base-1.0", # The classic backbone for ControlNet and custom LoRAs
    "XLabs-AI/flux-controlnet-canny",     # Pre-trained control mechanism specifically for FLUX architectures
    "InstantX/SDXL-ControlNet-Canny",     # High-quality structural conditioning for the SDXL ecosystem
    "ByteDance/SDXL-Lightning",           # Best for efficiency: high-quality 1-4 step generation
    "DeepFloyd/IF-I-XL-v1.0",             # Pixel-space diffusion: excellent for spatial reasoning and text
    "Kwai-Kolors/Kolors",                 # Strong generalization and photorealism based on SDXL
    "RunwayML/stable-diffusion-v1-5"      # Legacy but essential: the most lightweight for edge-case fine-tuning
],
    "Sample_247": [
    "black-forest-labs/FLUX.1-dev-inpaint", # SOTA DiT: Best background consistency and prompt adherence
    "stabilityai/stable-diffusion-3.5-large-inpaint", # Multi-modal DiT: superior text-guided inpainting
    "InstantX/FLUX.1-dev-ControlNet-Inpaint", # Adapter: adds precise mask control to standard FLUX
    "stabilityai/stable-diffusion-xl-base-inpaint", # The robust U-Net standard for high-res inpainting
    "Mikubill/sd-webui-controlnet", # Necessary backbone configs for SDXL controllable inpainting
    "XLabs-AI/flux-controlnet-canny", # Essential for structure-guided, reference-based inpainting
    "RunwayML/stable-diffusion-inpainting", # Legacy standard: lightest weight, massive ecosystem
    "DeepFloyd/IF-I-XL-v1.0", # Pixel-space: best spatial reasoning for complex region filling
    "TencentARC/T2I-Adapter-Inpaint-SDXL", # Lightweight adapter for structural/reference conditioning
    "Kwai-Kolors/Kolors-Inpaint" # Strong photorealism specialized for masked region restoration
],
    "Sample_264": [
    "Salesforce/blip2-opt-2.7b",          # The "gold standard" baseline for zero-shot generalization
    "microsoft/git-large-coco",           # Highly efficient and specialized for COCO-style benchmarks
    "Qwen/Qwen2.5-VL-7B-Instruct",        # Superior at handling complex spatial and atmospheric details
    "OpenGVLab/InternVL2_5-8B",           # Top-tier open-source multimodal performance with high resolution
    "google/paligemma-3b-pt-224",         # Excellent for transfer learning and low-resource baselines
    "microsoft/kosmos-2-patch14-224",     # Best for grounded captioning (object locations + descriptions)
    "llava-hf/llava-onevision-qwen2-7b-ov", # SOTA for detailed "one-vision" image-to-text reasoning
    "Salesforce/blip-image-captioning-large", # The classic lightweight baseline for rapid iterations
    "BAAI/Emu2-Chat",                     # Generative multimodal model with high aesthetic sensitivity
    "THUDM/cogvlm-chat-hf"                # Strong visual-linguistic bridge for deep scene understanding
],
    "Sample_265": [
    "google/gemma-4-9b-it",                # Newest SOTA: unified multi-modal (image/audio/text) decoder
    "OpenGVLab/InternVL3-78B",             # Best open-source performance for complex VQA and reasoning
    "Salesforce/blip2-opt-2.7b",           # The classic encoder-decoder baseline for unified captioning/VQA
    "microsoft/git-large-vqa",             # Specialized encoder-decoder for VQA and captioning tasks
    "black-forest-labs/FLUX.1-dev-inpaint",# High-quality image understanding/generation backbone
    "llava-hf/llava-onevision-qwen2-7b-ov",# Excellent for unified one-shot visual understanding
    "google/paligemma-3b-pt-224",          # Compact and highly versatile for fine-tuning on VQA/Captioning
    "Qwen/Qwen2.5-VL-7B-Instruct",         # Exceptional at detailed scene description and structured VQA
    "BlinkDL/RWKV-5-World-V2-Image",       # Non-transformer alternative for highly efficient text generation
    "Salesforce/instructblip-vicuna-7b"    # Instruction-tuned unified model for diverse multimodal tasks
],
    "Sample_268": [
    "answerdotai/ModernBERT-base",        # SOTA: Optimized for speed, 8k context, and GeGLU activations
    "microsoft/deberta-v3-base",          # Best "Disentangled Attention": superior at nuance and logic
    "google/electra-base-discriminator",  # The efficiency champion: trained to detect "fake" tokens
    "BAAI/bge-reranker-v2-m3",            # Specialized for ranking: optimized for multilingual KG/QA
    "microsoft/deberta-v3-large",         # The heavy-lifter: highest accuracy for complex extractive QA
    "answerdotai/ModernBERT-large",       # High-throughput alternative to DeBERTa with better scaling
    "sentence-transformers/all-mpnet-base-v2", # Robust baseline: perfectly balanced for semantic search
    "cross-encoder/ms-marco-MiniLM-L-6-v2", # Ultra-fast: specifically fine-tuned for the MS-MARCO ranking task
    "FacebookAI/roberta-large",           # The reliable workhorse: extremely stable for custom fine-tuning
    "google/tapas-base-finetuned-wtq"      # Specialized: Encoder for QA over tabular/structured data
],
    "Sample_281": [
    "allenai/scibert-scivocab-uncased",    # The industry standard: trained on 1.1M papers from Semantic Scholar
    "answerdotai/ModernBERT-base-science",# 2026 SOTA: ModernBERT architecture adapted for scientific text
    "microsoft/biomednlp-pubmedbert-base",# Gold standard for life sciences and medical document retrieval
    "facebook/galactica-1.3b",            # Generative LLM with a massive scientific internal knowledge base
    "allenai/specter2_base",              # Specialized for retrieval: encodes papers based on citation graphs
    "sentence-transformers/all-SciBERT",  # Optimized for semantic similarity and scientific document ranking
    "google/electra-small-science",       # Ultra-efficient discriminator for real-time paper classification
    "BAAI/bge-m3-science-adapter",        # Best for multilingual scientific search and dense retrieval
    "nomic-ai/nomic-embed-text-v1.5-edu", # High performance on technical and educational document clustering
    "yuzhimanhua/FuTex-backbone"          # Research-specific model designed for full-text weak supervision
],
    "Sample_290": [
    "google/longt5-tglobal-large",        # SOTA for long Seq2Seq: 16k context with Transient Global attention
    "allenai/led-large-16384-arxiv",     # Longformer Encoder-Decoder: 16k window, ideal for long news reports
    "facebook/bart-large-cnn",           # The news industry gold standard (best for standard length articles)
    "google/pegasus-cnn_dailymail",      # Specialized for news: trained with a "gap sentence" objective
    "google/longt5-xl-tglobal-16k",      # Higher capacity version of LongT5 for nuanced news analysis
    "microsoft/prophetnet-large-cnndm",  # Future-ngram prediction: generates more coherent news flow
    "facebook/mbart-large-50-many-to-many-mmt", # Best for multilingual news summarization tasks
    "google/t5-v1_1-large",              # A strong "text-to-text" baseline that generalizes well after tuning
    "Vamsi/T5_Paraphrase_Puzzler",       # Useful for generating diverse, non-repetitive news headlines
    "black-forest-labs/FLUX.1-text-enc"  # Emerging 2026 encoder for grounded multimodal news summaries
],
    "Sample_292": [
    "Salesforce/blip2-opt-2.7b",           # Best zero-shot baseline; Q-Former handles domain shift well
    "google/paligemma-3b-pt-896",          # High-res version (896px); critical for small satellite objects
    "OpenGVLab/InternVL2-8B",              # SOTA for complex scene understanding and specialized imagery
    "microsoft/git-large-r0",              # Robust generative transformer; scales well to new visual domains
    "nielsr/slivit-satellite-analysis",    # Specialized: Vision-language model for remote sensing/satellite
    "llava-hf/llava-1.5-7b-hf",            # Strong "visual-instruction" following for describing custom domains
    "facebook/dinov2-large",               # Feature extractor: Use as backbone for zero-shot "tagging"
    "Qwen/Qwen2-VL-7B-Instruct",           # Handles variable resolutions and aspect ratios (great for maps)
    "microsoft/kosmos-2-patch14-224",      # Grounded model: can point to specific coordinate regions in images
    "Salesforce/instructblip-vicuna-7b"    # Better at following specific domain-based captioning instructions
],
    "Sample_293": [
    "google/gemma-4-26b-a4b-it",          # SOTA: Supports variable visual token budgets (up to 1120 tokens) for fine-grained OCR
    "microsoft/layoutlmv3-base-chinese",  # Classic standard: Unified text/layout/image patch alignment
    "naver-ai/donut-base",                # OCR-free transformer: Directly maps document pixels to structured JSON
    "meta-llama/Llama-3.2-11B-Vision-Instruct", # Strong multimodal backbone for complex document reasoning
    "microsoft/layoutlmv3-large",         # Maximum performance for layout-aware classification and extraction
    "facebook/nougat-base",               # Specialized for academic/structured documents with complex formulas
    "Alibaba-NLP/DocRes-base",            # 2026 High-res: Optimized for patch-level document restoration and understanding
    "u-next/unidoc-large",                # Unified multimodal encoder for layout-aware document classification
    "OpenGVLab/InternVL2-8B",             # Strongest open-source multimodal reasoning for high-resolution document parsing
    "BAAI/Emu2-Chat-Document"             # Generative document model with high visual-spatial fidelity
],
    "Sample_296": [
    "ibm-granite/granite-4.1-8b-instruct", # 2026 SOTA: 512k context window, optimized for business/legal docs
    "microsoft/phi-4-14b-instruct",       # High reasoning density: synthetic data training makes it superior for STEM
    "mistralai/Mistral-Nemo-12B-Instruct-v1", # Efficiency king: 128k context, great balance of speed and quality
    "meta-llama/Llama-3.3-70B-Instruct",  # The gold standard for self-hosting; excellent generalization
    "Qwen/Qwen3-7B-Instruct",            # Best-in-class for long-context multilingual summarization (128k+)
    "google/gemma-3-27b-it",              # Massive reasoning improvements over v2; 128k context support
    "allenai/OLMo-2-1124-13B-Instruct",   # Open-weights model specialized for academic and scientific domains
    "01-ai/Yi-1.5-34B-Chat-16k",          # Exceptional at abstractive summaries for financial reports
    "NousResearch/Hermes-3-Llama-3.1-8B", # Community-tuned for high instruction-following and creative brevity
    "CohereForAI/c4ai-command-r-v01"      # Specifically designed for RAG and long-document summarization
]
}
PERPLEXITY_RECOMMENDATIONS = {
    "Sample_8": [
    "facebook/nllb-200-distilled-600M",
    "facebook/nllb-200-1.3B",
    "facebook/nllb-200-3.3B",
    "facebook/m2m100_418M",
    "facebook/m2m100_1.2B",
    "Helsinki-NLP/opus-mt-ar-en",
    "Helsinki-NLP/opus-mt-en-ar",
    "Helsinki-NLP/opus-mt-ar-de",
    "Helsinki-NLP/opus-mt-ar-fr",
    "google/mt5-base"
],
    "Sample_9": [
    "google/mt5-small",
    "google/mt5-base",
    "google/mt5-large",
    "google/umt5-small",
    "google/umt5-base",
    "google/umt5-large",
    "facebook/mbart-large-50-many-to-many-mmt",
    "facebook/mbart-large-cc25",
    "facebook/nllb-200-distilled-600M",
    "facebook/nllb-200-1.3B"
],
    "Sample_18": [
    "facebook/bart-large",
    "facebook/bart-base",
    "google/pegasus-xsum",
    "google/pegasus-large",
    "google/flan-t5-large",
    "google/flan-t5-base",
    "google/t5-large",
    "google/t5-base",
    "facebook/mbart-large-50-many-to-many-mmt",
    "allenai/led-base-16384"
],
    "Sample_19": [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-14B",
    "Qwen/Qwen2.5-32B",
    "internlm/internlm2_5-7b-chat",
    "internlm/internlm2_5-20b-chat",
    "01ai/Yi-1.5-9B-Chat"
],
    "Sample_24": [
    "google/vit-base-patch16-224-in21k",
    "google/vit-base-patch16-224",
    "google/vit-large-patch16-224-in21k",
    "google/vit-large-patch16-224",
    "facebook/deit-base-patch16-224",
    "facebook/deit-base-distilled-patch16-224",
    "facebook/deit-large-patch16-224",
    "microsoft/swin-base-patch4-window7-224",
    "microsoft/beit-base-patch16-224",
    "openai/clip-vit-base-patch32"
],
    "Sample_55": [
    "intfloat/e5-large-v2",
    "intfloat/e5-base-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/gtr-t5-large",
    "facebook/dpr-ctx_encoder-single-nq-base",
    "facebook/dpr-question_encoder-single-nq-base",
    "thenlper/gte-base",
    "thenlper/gte-large",
    "BAAI/bge-large-en-v1.5"
],
    "Sample_62": [
    "stabilityai/stable-diffusion-x4-upscaler",
    "stabilityai/stable-diffusion-2-upscaler",
    "stabilityai/sd-x2.0-base",
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    "lllyasviel/control_v11f1e_sd15_tile",
    "xinsir/controlnet-tile-sdxl-1.0",
    "stabilityai/stable-diffusion-3-medium",
    "runwayml/stable-diffusion-v1-5",
    "CompVis/stable-diffusion-v1-4",
    "stabilityai/stable-diffusion-2-1"
],
    "sample_86":
    [
"microsoft/wavlm-base-plus",
"facebook/wav2vec2-base",
"facebook/hubert-base-ls960",
"google/vggish",
"laion/clap-htsat-fused",
"m-a-p/MERT-v1-95M",
"m-a-p/MERT-v1-330M",
"MIT/ast-finetuned-audioset-10-10-0.4593",
"openai/whisper-tiny",
"speechbrain/spkrec-ecapa-voxceleb"
],    "Sample_90": [
    "microsoft/deberta-v3-base",
    "microsoft/deberta-v3-large",
    "roberta-base",
    "roberta-large",
    "bert-base-uncased",
    "bert-large-uncased",
    "albert-xxlarge-v2",
    "allenai/longformer-base-4096",
    "facebook/bart-large",
    "google/flan-t5-large"
],
    "Sample_98": [
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "allenai/scibert_scivocab_uncased",
    "allenai/aspire-sentence-embedder",
    "dmis-lab/biobert-base-cased-v1.2",
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
    "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
    "razent/SciFive-base-Pubmed_PMC",
    "epfl-ml4ed/medbert",
    "stanford-crfm/BioMedLM"
],
    "Sample_99": [
    "yikuan8/Clinical-Longformer",
    "yikuan8/Clinical-BigBird",
    "kddi/clinical-t5-base",
    "kddi/clinical-t5-large",
    "starmpcc/Asclepius-R-7B",
    "starmpcc/Asclepius-R-13B",
    "allenai/scibert_scivocab_uncased",
    "emilyalsentzer/Bio_ClinicalBERT",
    "dmis-lab/biobert-base-cased-v1.2",
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
],
    "Sample_116": [
    "cross-encoder/nli-deberta-v3-base",
    "cross-encoder/nli-deberta-v3-large",
    "facebook/bart-large-mnli",
    "roberta-large-mnli",
    "cross-encoder/nli-roberta-base",
    "cross-encoder/nli-distilroberta-base",
    "valhalla/distilbart-mnli-12-1",
    "textattack/roberta-base-MNLI",
    "yoshitomo-matsubara/deberta-v3-base-mnli",
    "sentence-transformers/nli-deberta-v3-base"
],
    "Sample_130": [
    "microsoft/deberta-v3-large",
    "microsoft/deberta-v3-base",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.1-70B",
    "mistralai/Mistral-7B-v0.1",
    "google/gemma-2-9b",
    "google/gemma-2-27b",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-32B",
    "facebook/xlm-roberta-large"
],
    "Sample_134":[
    "facebook/bart-base",
    "facebook/bart-large",
    "google-t5/t5-base",
    "google-t5/t5-large",
    "google/flan-t5-base",
    "google/flan-t5-large",
    "facebook/mbart-large-50",
    "facebook/bart-large-cnn",
    "google/mt5-base",
    "google/mt5-large"
],
    "Sample_135": [
    "facebook/bart-base",
    "facebook/bart-large",
    "google-t5/t5-small",
    "google-t5/t5-base",
    "google-t5/t5-large",
    "google/flan-t5-base",
    "google/flan-t5-large",
    "facebook/mbart-large-50",
    "google/mt5-base",
    "google/mt5-large"
],
    "Sample_136": [
    "facebook/bart-large-cnn",
    "facebook/bart-large",
    "google-t5/t5-large",
    "google/flan-t5-large",
    "allenai/led-large-16384",
    "facebook/mbart-large-50-many-to-many-mmt",
    "google/mt5-large",
    "openai/clip-vit-base-patch32",
    "microsoft/deberta-v3-large",
    "Qwen/Qwen2.5-7B-Instruct"
],
    "Sample_146": [
"sentence-transformers/all-MiniLM-L6-v2",
"sentence-transformers/all-mpnet-base-v2",
"intfloat/e5-small-v2",
"intfloat/e5-base-v2",
"intfloat/e5-large-v2",
"BAAI/bge-small-en-v1.5",
"BAAI/bge-base-en-v1.5",
"BAAI/bge-large-en-v1.5",
"nomic-ai/nomic-embed-text-v1.5",
"Alibaba-NLP/gte-small"
],
    "Sample_158": [
    "facebook/bart-large",
    "google-t5/t5-large",
    "microsoft/deberta-v3-large",
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-large-en-v1.5",
    "allenai/longformer-base-4096",
    "google/flan-t5-large",
    "google/mt5-large",
    "facebook/mbart-large-50-many-to-many-mmt",
    "sentence-transformers/all-mpnet-base-v2"
],
    "Sample_177": [
"deepseek-ai/DeepSeek-R1",
"Qwen/QwQ-32B",
"Qwen/Qwen2.5-72B-Instruct",
"meta-llama/Llama-3.3-70B-Instruct",
"mistralai/Mistral-7B-Instruct-v0.3",
"THUDM/GLM-Z1-9B-0414",
"Qwen/Qwen2.5-Coder-32B-Instruct",
"deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
"codellama/CodeLlama-34b-Instruct-hf",
"microsoft/Phi-4-mini-instruct"
],
    "Sample_178": [
"facebook/dpr-question_encoder-single-nq-base",
"facebook/dpr-ctx_encoder-single-nq-base",
"facebook/dpr-question_encoder-multiset-base",
"facebook/dpr-ctx_encoder-multiset-base",
"sentence-transformers/msmarco-MiniLM-L6-cos-v5",
"sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
"sentence-transformers/multi-qa-mpnet-base-dot-v1",
"intfloat/e5-base-v2",
"intfloat/e5-large-v2",
"BAAI/bge-base-en-v1.5"
],
    "Sample_179": [
    "microsoft/swin-base-patch4-window7-224",
    "microsoft/swin-large-patch4-window12-384",
    "microsoft/hiera-base-224",
    "microsoft/hiera-large-224",
    "google/vit-base-patch16-224-in21k",
    "google/vit-large-patch16-224-in21k",
    "facebook/detr-resnet-50",
    "facebook/vit-mae-base",
    "microsoft/beit-base-patch16-224-pt22k-ft22k",
    "openai/clip-vit-base-patch32"
],
    "Sample_184": [
    "microsoft/swin-base-patch4-window7-224",
    "microsoft/swin-large-patch4-window12-384",
    "microsoft/hiera-base-224",
    "microsoft/hiera-large-224",
    "microsoft/beit-base-patch16-224-pt22k-ft22k",
    "facebook/vit-mae-base",
    "google/vit-base-patch16-224-in21k",
    "google/vit-large-patch16-224-in21k",
    "openai/clip-vit-base-patch32",
    "microsoft/git-base"
],
    "Sample_194": [
    "facebook/bart-base",
    "facebook/bart-large",
    "t5-base",
    "t5-large",
    "google/mt5-base",
    "google/mt5-large",
    "google/flan-t5-base",
    "google/flan-t5-large",
    "allenai/led-base-16384",
    "allenai/led-large-16384"
],
    "Sample_212": [
    "google/t5-v1_1-large",
    "google/flan-t5-large",
    "google/flan-t5-xl",
    "google/ul2",
    "google/t5-large",
    "google/t5-base",
    "bigscience/T0pp",
    "bigscience/mt0-large",
    "facebook/bart-large",
    "allenai/unifiedqa-t5-large"
],
    "Sample_213": [
    "microsoft/Phi-3.5-mini-instruct",
    "microsoft/Phi-3-mini-4k-instruct",
    "microsoft/Phi-2",
    "microsoft/Phi-3-small-8k-instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "google/gemma-2-2b-it",
    "mistralai/Mistral-7B-Instruct-v0.3"
],
    "Sample_216": [
    "runwayml/stable-diffusion-v1-5",
    "stabilityai/stable-diffusion-2-1",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    "stabilityai/sdxl-turbo",
    "stabilityai/stable-diffusion-3-medium-diffusers",
    "black-forest-labs/FLUX.1-schnell",
    "black-forest-labs/FLUX.1-dev",
    "PixArt-alpha/PixArt-XL-2-1024-MS",
    "kandinsky-community/kandinsky-2-2-decoder"
],
    "Sample_217": [
    "openai/clip-vit-base-patch32",
    "openai/clip-vit-large-patch14",
    "Salesforce/blip2-opt-2.7b",
    "Salesforce/blip2-flan-t5-xl",
    "microsoft/beit3-large-itm",
    "facebook/imagebind-huge",
    "laion/CLIP-ViT-H-14-laion2B-s38B-b79K",
    "openmmlab/Multimodal-Embedding-CLIP-ViT-L-14",
    "nvidia/clip-vit-giant-patch14-336",
    "Kwai-Kolors/Kolors-CLIP-ViT-L-14-336"
],
    "Sample_220": [
    "openai/clip-vit-base-patch32",
    "openai/clip-vit-large-patch14",
    "openai/clip-vit-huge-patch14",
    "google/siglip-base-patch16-224",
    "google/siglip-so-400m-patch14-384",
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
    "openmmlab/Multimodal-Embedding-CLIP-ViT-L-14",
    "nvidia/clip-vit-giant-patch14-336",
    "Kwai-Kolors/Kolors-CLIP-ViT-L-14-336"
],
    "Sample_223": [
    "runwayml/stable-diffusion-v1-5",
    "stabilityai/stable-diffusion-2-1",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "stabilityai/stable-diffusion-3-medium",
    "black-forest-labs/FLUX.1-dev",
    "openai/stable-diffusion-x4",
    "nota-ai/BK-SD-v1.4",
    "nota-ai/BK-SD-v2.1",
    "CompVis/stable-diffusion-v1-4",
    "hakurei/waifu-diffusion-v1-4"
],
    "Sample_224": [
    "openai/clip-vit-base-patch32",
    "openai/clip-vit-large-patch14",
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    "google/siglip-base-patch16-224",
    "google/siglip-large-patch16-224",
    "microsoft/beit-base-patch16-224",
    "facebook/dino-vits16",
    "microsoft/resnet-50",
    "timm/vit_base_patch16_224"
],
    "Sample_229": [
    "google-t5/t5-small",
    "google-t5/t5-base",
    "google-t5/t5-large",
    "google/flan-t5-small",
    "google/flan-t5-base",
    "google/flan-t5-large",
    "facebook/bart-base",
    "facebook/bart-large",
    "allenai/unifiedqa-t5-base",
    "allenai/unifiedqa-t5-large"
],
    "Sample_230": [
    "google/vit-base-patch16-224",
    "google/vit-large-patch16-224",
    "facebook/dino-vitb16",
    "facebook/deit-base-distilled-patch16-224",
    "microsoft/swin-base-patch4-window7-224",
    "microsoft/swin-large-patch4-window7-224",
    "timm/vit_base_patch16_224",
    "timm/vit_large_patch16_224_in21k",
    "openai/clip-vit-base-patch32",
    "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
],
    "Sample_231": [
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "allenai/scibert_scivocab_uncased",
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
    "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
    "dmis-lab/biobert-base-cased-v1.2",
    "razent/SciFive-base-Pubmed_PMC",
    "epfl-ml4ed/medbert",
    "microsoft/biogpt",
    "stanford-crfm/BioMedLM"
],
    "Sample_237": [
    "sentence-transformers/all-mpnet-base-v2",  # Top general-purpose, high accuracy
    "sentence-transformers/all-MiniLM-L6-v2",   # Fast, efficient for production
    "sentence-transformers/all-MiniLM-L12-v2",  # Balanced speed/accuracy upgrade
    "BAAI/bge-large-en-v1.5",                  # Excellent retrieval, query instruction support
    "BAAI/bge-base-en-v1.5",                   # Efficient BGE variant
    "BAAI/bge-small-en-v1.5",                  # Lightweight, low compute
    "sentence-transformers/multi-qa-mpnet-base-dot-v1",  # Optimized for question answering
    "sentence-transformers/paraphrase-mpnet-base-v2",    # Strong semantic matching
    "intfloat/e5-large-v2",                    # Versatile for asymmetric search (queries vs passages)
    "intfloat/e5-base-v2"                      # Efficient E5 alternative
],
    "Sample_246": [
    "black-forest-labs/FLUX.1-dev",             # Superior quality, strong generalization, text conditioning
    "black-forest-labs/FLUX.1-schnell",         # Fast variant for efficient synthesis
    "stabilityai/stable-diffusion-xl-base-1.0", # Excellent controllable backbone (ControlNet support)
    "stabilityai/stable-diffusion-3.5-large",   # Latest SD3.5, improved prompt adherence
    "stabilityai/stable-diffusion-2-1",         # Reliable base for fine-tuning/conditioning
    "CompVis/stable-diffusion-v1-5",            # Classic, vast ecosystem for structured gen
    "diffusers/controlnet-canny-sdxl-1.0",      # Edge-conditioned control for precision
    "lllyasviel/sd-controlnet-depth",           # Depth/Canny conditioning backbone
    "xinsir/controlnet-union-sdxl-1.0",         # Multi-control (canny/depth/openpose)
    "SHI-Labs/OmniControl-V1.0"                 # Advanced multi-modal control
],
    "Sample_247": [
    "stable-diffusion-v1-5/stable-diffusion-inpainting",  # Classic, fast 512x512 fills
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",  # High-res SDXL inpainting
    "stabilityai/stable-diffusion-2-inpainting",         # Improved v2 context awareness
    "kandinsky-community/kandinsky-2-2-decoder-inpaint", # Multilingual, high-quality
    "black-forest-labs/FLUX.1-Fill-dev",                 # Advanced context matching
    "runware/flux1-fill-pro",                            # Pro-level seamless edits
    "lllyasviel/control_v11p_sd15_inpaint",              # ControlNet inpaint backbone
    "diffusers/controlnet-inpaint-sdxl-1.0",             # SDXL ControlNet inpainting
    "ehristoforu/Visionix-alpha-inpainting",             # Realism-focused fills
    "stabilityai/stable-diffusion-x4-latent-upscaler"    # Upscale-aware inpainting extension
],
    "Sample_264": [
    "Salesforce/blip-image-captioning-large",     # SOTA baseline, unified VLU/VLG
    "Salesforce/blip-image-captioning-base",      # Efficient variant
    "nlpconnect/vit-gpt2-image-captioning",       # Simple ViT-GPT2, easy baseline
    "microsoft/git-base-coco",                    # Strong on COCO, VQA-capable
    "microsoft/git-large-coco",                   # Larger for better generalization
    "microsoft/git-large-r",                      # Refined, robust captions
    "Salesforce/blip2-opt-2.7b",                  # BLIP-2, advanced multimodal
    "llava-hf/llava-1.5-7b-hf",                   # LLaVA, detailed instruction-tuned
    "FreedomIntelligence/mistral-nemo-v1.5-vlm",  # Modern VLM with strong captioning
    "unbabel/bert-base-cocotus-captioning"        # COCO-tuned BERT decoder
],
    "Sample_265": [
    "nlpconnect/vit-gpt2-image-captioning",      # ViT encoder + GPT2 decoder for caption/VQA
    "Salesforce/blip-image-captioning-large",    # BLIP unified VLU/VLG encoder-decoder
    "Salesforce/blip-image-captioning-base",     # Efficient BLIP variant
    "microsoft/git-base-coco",                   # GIT encoder-decoder for multimodal gen
    "microsoft/git-large-coco",                  # Larger GIT for VQA/captioning
    "OFA-Sys/ofa-large",                         # OFA unified seq2seq across modalities/tasks
    "Salesforce/blip2-opt-2.7b",                 # BLIP-2 advanced encoder-decoder
    "runwayml/stable-diffusion-v1-5",            # SD with text decoder (via Diffusers)
    "google/t5-v1_1-large",                      # T5 enc-dec baseline for text2text (extend multimodal)
    "t5-base"                                    # Lightweight T5 for fine-tuning multimodal
],
    "Sample_268": [
    "sentence-transformers/all-MiniLM-L6-v2",    # Efficient ranking baseline (22M params)
    "google-bert/bert-base-uncased",             # Classic discriminator for QA/ranking
    "microsoft/DistilBert-base-uncased",         # 40% faster BERT, strong fine-tune
    "sentence-transformers/all-mpnet-base-v2",   # Top BEIR performer
    "cross-encoder/ms-marco-MiniLM-L-6-v2",      # Pre-fine-tuned reranker base
    "BAAI/bge-base-en-v1.5",                     # Efficient bi-encoder for retrieval/QA
    "sentence-transformers/multi-qa-mpnet-base-dot-v1", # QA-optimized
    "roberta-base",                              # Robust alternative to BERT
    "distilroberta-base",                        # Efficient RoBERTa
    "electra-base-discriminator"                 # High efficiency discriminator
],
    "Sample_281": [
    "allenai/scibert_scivocab_uncased",           # SciBERT: Top for arXiv papers, classification
    "allenai/scibert_scivocab_uncased",           # Uncased variant, broad science
    "dmis-lab/biobert-base-cased-v1.1",           # BioBERT: Biomed focus, abstracts
    "emilyalsentzer/Bio_ClinicalBERT",            # Clinical sci docs
    "bionlp/bluebert_pubmed_256B_uncased",        # PubMed-tuned BERT
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",  # PubMed abstracts/full
    "sentence-transformers/scibert_scivocab_uncased",  # Embeddings for retrieval
    "nlpaueb/legal-bert-base-uncased",            # Sci-adjacent (patents/legal)
    "castorini/passage-ranking-msmarco-bert-base-dot-v5",   # Retrieval for weak sup.
    "deepset/scibert-base-cased-sci-sentence-classification" # Pre-fine-tuned classifier
],
    "Sample_290": [
    "sshleifer/distilbart-cnn-12-6",              # Efficient BART for CNN/DM news
    "facebook/bart-large-cnn",                    # SOTA baseline for news abstractive
    "google/long-t5-local-base",                  # Long inputs (16k), efficient
    "google/long-t5-tglobal-base",                # Global attention for long docs
    "google/pegasus-large",                       # Pegasus: Gap-sentence pretrain
    "google/pegasus-cnn_dailymail",               # News-specific Pegasus
    "google/flan-t5-large",                       # Instruction-tuned T5 for cond gen
    "t5-base",                                    # Versatile seq2seq base
    "google/mt5-base",                            # Multilingual capable
    "bigscience/bloom"                            # Large-scale generation alternative
],
    "Sample_292": [
    "Salesforce/blip-image-captioning-base",
    "Salesforce/blip2-opt-2.7b",
    "llava-hf/llava-1.5-7b-hf",
    "Qwen/Qwen2-VL-2B-Instruct",
    "mistralai/Pixtral-12B-2409",
    "allenai/t5-small-for-conditional-generation",  # adaptable for captioning
    "microsoft/git-base",
    "google/vit-base-patch16-224-in21k",  # vision encoder for VLM pipelines
    "IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1",  # multimodal
    "openbmb/MiniCPM-Llama3-V-2_5"
],
    "Sample_293": [
    "microsoft/layoutlmv3-base",
    "microsoft/layoutlmv3-large",
    "naver-clova-ix/donut-base",
    "naver-clova-ix/donut-base-finetuned-cord-v2",
    "impira/layoutlm-document-qa",
    "microsoft/table-transformer-detection",
    "layoutlmv2-base-uncased", 
    "microsoft/git-large-coco",  # adaptable for docs
    "doclaynet/doclaynet-swin-base",
    "pix2struct/pix2struct-docvqa-base"
],
    "Sample_296": [
    "google/flan-t5-large",
    "google/flan-t5-xl",
    "google/flan-t5-xxl",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-7B-Instruct",
    "microsoft/DialoGPT-large",
    "facebook/bart-large-cnn",
    "google/pegasus-large",
    "t5-base",
    "meta-llama/Llama-3.1-8B-Instruct"
]
}
CLAUDE_RECOMMENDATIONS  ={
    "Sample_8": [
    "facebook/nllb-200-distilled-600M",           # NLLB-200: 200 langs, Arabic-first, best for span translation
    "facebook/nllb-200-1.3B",                     # NLLB-200 larger variant, higher quality for rare lang pairs
    "facebook/m2m100_418M",                        # M2M100: true many-to-many, 100 langs, no English pivot
    "facebook/m2m100_1.2B",                        # M2M100 larger: better quality across 9900 directions
    "facebook/mbart-large-50-many-to-many-mmt",   # mBART-50: seq2seq, Arabic included, fine-tuning friendly
    "Helsinki-NLP/opus-mt-ar-en",                  # Compact Arabic→English, fast inference for span swapping
    "Helsinki-NLP/opus-mt-tc-big-ar-en",          # Larger OPUS Arabic→English with better fluency
    "google/madlad400-3b-mt",                      # MADLAD-400: 400+ langs including Arabic dialects, T5-based
    "UBC-NLP/turjuman",                            # Arabic-centric many-to-many MT, built for MSA and dialects
    "microsoft/Azure-AI-translator-mt-ar",         # Azure-backed Arabic multilingual seq2seq fine-tune target
],
    "Sample_9":[
    "google/flan-t5-xl",                  # Instruction-tuned T5, 55+ langs incl. Arabic, best zero/few-shot balance
    "google/flan-t5-xxl",                 # Larger Flan-T5, stronger generalization at cost of compute
    "google/flan-t5-large",               # Lighter Flan-T5, good for fine-tuning with limited data
    "google/mt5-xl",                      # Raw multilingual T5, 101 langs, strong Arabic pretraining for fine-tuning
    "google/mt5-large",                   # Smaller mT5, efficient fine-tuning backbone across many languages
    "bigscience/mt0-xxl",                 # mT5 instruction-tuned on xP3, crosslingual zero-shot generalization
    "bigscience/mt0-xl",                  # Lighter mT0, enc-dec, Arabic-capable, task-flexible
    "bigscience/bloomz-7b1",              # BLOOM instruction-tuned, decoder-only but strong zero-shot Arabic NLG
    "UBC-NLP/AraT5v2-base-1024",          # Arabic-native T5, best Arabic generation quality with fine-tuning
    "google/madlad400-3b-mt",             # T5-based, 400+ langs, strong cross-lingual transfer for generation
],
    "Sample_18":[
    "facebook/bart-large-cnn",            # BART fine-tuned on CNN/DM; gold standard summarization backbone
    "facebook/bart-large-xsum",           # BART tuned for abstractive, concise single-sentence summaries
    "google/pegasus-large",               # GSG-pretrained; best summarization-specific pretraining objective
    "google/pegasus-xsum",                # PEGASUS tuned on XSum; strong abstractive headline generation
    "google/pegasus-x-large",             # Extends PEGASUS to 16K tokens; backbone for long-doc architectures
    "allenai/led-large-16384",            # Longformer-Encoder-Decoder; efficient long-document summarization
    "allenai/led-large-16384-arxiv",      # LED fine-tuned on arXiv; strong scientific summarization backbone
    "google/long-t5-tglobal-large",       # LongT5 with global attention; scalable enc-dec for long summaries
    "sshleifer/distilbart-cnn-12-6",      # Distilled BART; lightweight backbone for resource-constrained settings
    "philschmid/bart-large-cnn-samsum",   # BART fine-tuned on dialogue summarization (SAMSum); domain transfer
],
    "Sample_19": [
    "Qwen/Qwen2.5-7B",                           # Best overall Chinese base: SFT/PPO/RLHF-ready, explicit support
    "Qwen/Qwen2.5-14B",                          # Larger Qwen2.5 base for higher-capacity medical adaptation
    "internlm/internlm2-base-7b",                # Natively trained with RLHF+PPO, 200K context, strong Chinese
    "internlm/internlm2-base-20b",               # InternLM2 at 20B scale, same full training regime
    "THUDM/chatglm3-6b-base",                    # ChatGLM3 base: Chinese-first, bilingual, safety-aligned
    "baichuan-inc/Baichuan2-13B-Base",           # Trained on 2.6T tokens; proven Chinese medical baseline
    "deepseek-ai/deepseek-llm-7b-base",          # DeepSeek-LLM: strong reasoning, full SFT/RLHF pipeline
    "deepseek-ai/deepseek-llm-67b-base",         # DeepSeek 67B for high-capacity continued pretraining
    "FreedomIntelligence/HuatuoGPT2-7B",         # Chinese medical SFT+RLHF model, direct domain reference
    "BAAI/AquilaMed-RL",                         # Full pretrain→SFT→DPO pipeline on Chinese medical data
],
    "Sample_24": [
    "google/vit-base-patch16-224",
    "google/vit-large-patch16-224",
    "google/vit-base-patch16-224-in21k",
    "google/vit-large-patch16-224-in21k",
    "google/vit-huge-patch14-224-in21k",
    "facebook/deit-base-distilled-patch16-224",
    "facebook/deit-small-patch16-224",
    "microsoft/beit-base-patch16-224",
    "microsoft/beit-large-patch16-224",
    "microsoft/beit-base-patch16-224-pt22k-ft22k",
],
    "Sample_55":  [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "BAAI/bge-large-en-v1.5",
    "BAAI/bge-m3",
    "intfloat/e5-large-v2",
    "intfloat/multilingual-e5-large",
    "thenlper/gte-large",
    "mixedbread-ai/mxbai-embed-large-v1",
    "Alibaba-NLP/gte-Qwen2-7B-instruct",
],
    "Sample_62": [
    "stabilityai/stable-diffusion-x4-upscaler",
    "stabilityai/sd-x2-latent-upscaler",
    "stabilityai/stable-diffusion-2-1",
    "diffusers/stable-diffusion-xl-refiner-1.0",
    "ai-forever/Real-ESRGAN",
    "sberbank-ai/Real-ESRGAN",
    "caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr",
    "CompVis/ldm-super-resolution-4x-openimages",
    "gilesagnel/SD-upscaler-pipeline",
    "konieshadow/fooocus-api-realistic",
],

    "sample_86":[
    "facebook/wav2vec2-base",                        # 95M params – strong general-purpose audio encoder, fast base variant
    "facebook/hubert-base-ls960",                    # 94M params – HuBERT base, rich self-supervised audio representations
    "microsoft/wavlm-base",                          # 94M params – WavLM base, excels at diverse audio understanding tasks
    "openai/whisper-tiny",                           # 39M params – smallest Whisper encoder, very fast, good feature backbone
    "openai/whisper-base",                           # 74M params – slightly richer than tiny, still real-time capable
    "MIT/ast-finetuned-audioset-10-10-0.4593",       # Audio Spectrogram Transformer, strong AudioSet representations
    "facebook/wav2vec2-base-960h",                   # wav2vec2 fine-tuned on LibriSpeech, speech-optimised embeddings
    "microsoft/wavlm-base-plus",                     # WavLM base+ with extra unlabelled data, better generalization
    "Anton-Sizykh/audio-embeddings-model",           # Dedicated lightweight audio embedding model
    "laion/larger_clap_general",                     # CLAP: joint audio-language embeddings, great for AV alignment
],
    "Sample_90":  [
    "google-bert/bert-large-uncased-whole-word-masking-finetuned-squad",
    "roberta-large",
    "ALBERT/albert-xxlarge-v2",
    "microsoft/deberta-v3-large",
    "microsoft/deberta-xlarge",
    "microsoft/deberta-v3-base",
    "deepset/roberta-large-squad2",
    "LIAMF-USP/roberta-large-finetuned-race",
    "ALBERT/albert-xxlarge-v2-finetuned-race",
    "tasksource/deberta-large-long-nli",
],
    "Sample_98": [
    "allenai/scibert_scivocab_uncased",
    "allenai/scibert_scivocab_cased",
    "allenai/longformer-base-4096",
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "dmis-lab/biobert-v1.1",
    "sultan/BioM-ELECTRA-Large-SQuAD2",
    "NLP4Science/scibert-acronym-disambiguation",
    "allenai/specter2_base",
    "malteos/scincl",
    "gsarti/scibert-nli",
],
    "Sample_99":  [
    "emilyalsentzer/Bio_ClinicalBERT",
    "medicalai/ClinicalBERT",
    "obi/deid_roberta_i2b2",
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "sultan/BioM-ELECTRA-Large-SQuAD2",
    "dmis-lab/biobert-v1.1",
    "cambridgeltl/SapBERT-from-PubMed-self-supervized",
    "UFNLP/gatortron-base",
    "StanfordAIMI/stanford-deidentifier-base",
    "healx/gpt-2-pubmed-medium",
],
    "Sample_116":  [
    "cross-encoder/nli-deberta-v3-large",       # Best accuracy (92.2% SNLI / 90.5% MNLI); top choice for offline batch evaluation
    "cross-encoder/nli-deberta-v3-base",         # Accuracy/speed sweet spot (92.4% SNLI / 90.0% MNLI); recommended default
    "cross-encoder/nli-deberta-v3-small",        # Lighter DeBERTa-v3 (91.6% SNLI / 87.6% MNLI); good for constrained environments
    "cross-encoder/nli-deberta-v3-xsmall",       # Smallest DeBERTa-v3 NLI (91.6% SNLI / 87.8% MNLI); fast inference
    "cross-encoder/nli-deberta-base",            # DeBERTa-v1 base NLI cross-encoder; stable and widely tested baseline
    "cross-encoder/nli-roberta-base",            # RoBERTa-based cross-encoder; returns contradiction/entailment/neutral per pair
    "facebook/bart-large-mnli",                  # BART-large fine-tuned on MNLI; strong zero-shot NLI and entailment scoring
    "roberta-large-mnli",                        # RoBERTa-large MNLI; classic high-accuracy entailment classifier
    "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",  # Multilingual DeBERTa NLI across 100 languages; for multilingual NLU tasks
    "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",  # DeBERTa-v3-large trained on 5 NLI datasets; most robust entailment model
],
"Sample_130": [
    "microsoft/deberta-v3-large",
    "microsoft/deberta-v3-base",
    "microsoft/deberta-v3-small",
    "microsoft/deberta-large",
    "microsoft/deberta-base",
    "microsoft/deberta-xlarge-v2",
    "microsoft/deberta-v2-xxlarge",
    "google/bigbird-roberta-large",
    "allenai/longformer-large-4096",
    "studio-ousia/luke-large",
],
    "Sample_134": [
    "facebook/bart-large",
    "facebook/bart-base",
    "facebook/bart-large-cnn",
    "facebook/mbart-large-cc25",
    "facebook/mbart-large-50",
    "google/t5-v1_1-large",
    "google/t5-v1_1-base",
    "google/flan-t5-large",
    "microsoft/prophetnet-large-uncased",
    "Yale-LILY/blink-biencoder-bi-encoder",
],
    "Sample_135": [
    "facebook/bart-base",
    "facebook/bart-large",
    "facebook/bart-large-cnn",
    "facebook/bart-large-xsum",
    "facebook/bart-large-mnli",
    "sshleifer/distilbart-cnn-6-6",
    "sshleifer/distilbart-cnn-12-6",
    "sshleifer/distilbart-xsum-12-3",
    "lucadiliello/bart-small",
    "ainize/bart-base-cnn",
],
    "Sample_136":[
    "facebook/bart-large-cnn",
    "facebook/bart-large-xsum",
    "facebook/bart-large",
    "facebook/bart-base",
    "google/pegasus-large",
    "google/pegasus-xsum",
    "google/pegasus-x-large",
    "sshleifer/distilbart-cnn-12-6",
    "google/flan-t5-large",
    "allenai/led-large-16384",
],
    "Sample_146": [
    "facebook/contriever-msmarco",               # Unsupervised contrastive retriever – top zero-shot generalization on BEIR
    "facebook/contriever",                        # Base Contriever without MS MARCO fine-tuning – purer unsupervised representations
    "BAAI/bge-base-en-v1.5",                      # BGE base – excellent query/doc asymmetric encoding, strong MTEB performance
    "BAAI/bge-m3",                                # BGE-M3 – dense + sparse + multi-vector, strong cross-domain & multilingual retrieval
    "sentence-transformers/msmarco-distilbert-base-tas-b",  # TAS-B dual-encoder – optimised for passage retrieval, fast inference
    "sentence-transformers/multi-qa-mpnet-base-dot-v1",     # MPNet trained on 215M QA pairs – generalises well to unseen query types
    "intfloat/e5-base-v2",                        # E5 base – instruction-aware embeddings, strong zero-shot retrieval on BEIR
    "intfloat/e5-large-v2",                       # E5 large – higher-capacity variant, top BEIR scores for unseen domains
    "facebook/dpr-question_encoder-single-nq-base",  # DPR query encoder – classic dual-encoder for dialogue QA retrieval
    "sentence-transformers/all-mpnet-base-v2",    # General-purpose MPNet bi-encoder – robust cross-domain sentence embeddings
],
    "Sample_158": [
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/paraphrase-mpnet-base-v2",
    "sentence-transformers/multi-qa-mpnet-base-dot-v1",
    "sentence-transformers/all-distilroberta-v1",
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-large-en-v1.5",
    "thenlper/gte-base",
    "intfloat/e5-base-v2",
    "declare-lab/flan-alpaca-base",
],
    "Sample_177": [
    "Qwen/Qwen2.5-Coder-7B-Instruct",            # Top-tier code+math instruct model; decontaminated on HumanEval/MATH for fair eval
    "Qwen/Qwen2.5-Math-7B-Instruct",              # Dedicated math instruct model; SOTA on GSM8K, MATH, AIME, OlympiadBench
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",   # R1 reasoning distilled into 7B; strong chain-of-thought on math and code
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",  # R1 distill at 32B; outperforms OpenAI-o1-mini on reasoning benchmarks
    "deepseek-ai/deepseek-coder-7b-instruct-v1.5",# Specialist code instruct model; strong HumanEval/MBPP generalization
    "meta-llama/Llama-3.1-8B-Instruct",           # Llama 3.1 instruct; 128k context, reliable zero-shot reasoning baseline
    "meta-llama/Llama-3.3-70B-Instruct",          # Larger Llama instruct; strong cross-task generalization, broad eval coverage
    "Qwen/Qwen2.5-72B-Instruct",                  # Qwen2.5 flagship instruct; matches Llama-3.1-405B on math and coding tasks
    "microsoft/phi-4",                             # 14B dense model; exceptional math reasoning and code for its size class
    "MathLLMs/MathCoder2-DeepSeekMath-7B",        # MathCoder2 fine-tune; interleaves NL+code+execution for math problem solving
],
    "Sample_178": [
    "intfloat/e5-large-v2",                          # E5-large: first zero-shot model to beat BM25 on full BEIR benchmark
    "intfloat/e5-mistral-7b-instruct",               # E5 backed by Mistral-7B; top MTEB retrieval scores, instruction-aware
    "BAAI/bge-large-en-v1.5",                        # BGE-large: RetroMAE pre-trained, strong zero-shot QA passage retrieval
    "BAAI/bge-m3",                                   # BGE-M3: dense + sparse + multi-vector; scales to large web corpora
    "facebook/contriever-msmarco",                   # Contriever: unsupervised, robust zero-shot BEIR generalization
    "facebook/dpr-question_encoder-single-nq-base",  # DPR: the canonical dual-encoder for open-domain QA over Wikipedia
    "facebook/rag-token-nq",                         # RAG-Token: end-to-end retriever+reader, native LM integration via FAISS
    "sentence-transformers/msmarco-distilbert-base-tas-b",  # TAS-B: optimised dual-encoder for passage retrieval, fast inference
    "Alibaba-NLP/gte-large-en-v1.5",                 # GTE-large: top MTEB retrieval model from Alibaba, strong on QA domains
    "nvidia/dragon-plus-query-encoder",              # DRAGON+: progressive augmentation training, strong cross-domain QA retrieval
],
    "Sample_179": [
    "microsoft/swin-large-patch4-window12-384-in22k",
    "microsoft/swin-base-patch4-window12-384-in22k",
    "OpenGVLab/InternViT-6B-448px-V2_5",
    "OpenGVLab/InternVL2_5-8B",
    "OpenGVLab/InternVL2-1B",
    "llava-hf/llava-1.5-7b-hf",
    "Qwen/Qwen2-VL-7B-Instruct",
    "HuggingFaceM4/idefics2-8b",
    "microsoft/Florence-2-large",
    "meta-llama/Llama-3.2-11B-Vision-Instruct"
],
    "Sample_184": [
    "microsoft/swinv2-large-patch4-window12to24-192to384-22kto1k-ft",
    "microsoft/swinv2-base-patch4-window12to24-192to384-22kto1k-ft",
    "microsoft/swin-large-patch4-window12-384-in22k",
    "OpenGVLab/InternViT-6B-448px-V2_5",
    "OpenGVLab/InternVL2_5-8B",
    "google/siglip-so400m-patch14-384",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen2.5-VL-72B-Instruct",
    "microsoft/Florence-2-large",
    "llava-hf/llava-1.5-13b-hf"
],
    "Sample_194": [
    "google-t5/t5-base",
    "google-t5/t5-large",
    "google/flan-t5-base",
    "google/flan-t5-large",
    "facebook/bart-base",
    "facebook/bart-large-cnn",
    "google/pegasus-xsum",
    "google/mt5-base",
    "Helsinki-NLP/opus-mt-en-de",
    "facebook/mbart-large-cc25"
],
    "Sample_212": [
    "MingZhong/unieval-intermediate",
    "MingZhong/unieval-sum",
    "MingZhong/unieval-dialog",
    "MingZhong/unieval-fact",
    "google-t5/t5-large",
    "google-t5/t5-3b",
    "google/flan-t5-large",
    "google/flan-t5-xl",
    "google/t5-v1_1-large",
    "google/t5-v1_1-xl"
],
    "Sample_213": [
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "microsoft/Phi-3.5-vision-instruct",
    "lmsys/vicuna-7b-v1.5",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-7B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "google/gemma-2-2b-it",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct"
],
    "Sample_216":  [
    "stabilityai/stable-diffusion-2-1",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "runwayml/stable-diffusion-v1-5",
    "stabilityai/stable-diffusion-3-medium-diffusers",
    "CompVis/stable-diffusion-v1-4",
    "black-forest-labs/FLUX.1-dev",
    "black-forest-labs/FLUX.1-schnell",
    "kandinsky-community/kandinsky-2-2-decoder",
    "playgroundai/playground-v2.5-1024px-aesthetic",
    "dataautogpt3/OpenDalleV1.1"
],
    "Sample_217": [
    "openai/clip-vit-large-patch14",
    "openai/clip-vit-large-patch14-336",
    "openai/clip-vit-base-patch32",
    "google/siglip-so400m-patch14-384",
    "google/siglip2-so400m-patch16-384",
    "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    "Salesforce/blip2-opt-2.7b",
    "Salesforce/blip-image-captioning-large",
    "facebook/flava-full",
    "kakaobrain/align-base"
],
    "Sample_220": [
    "openai/clip-vit-large-patch14",
    "openai/clip-vit-large-patch14-336",
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    "google/siglip-so400m-patch14-384",
    "google/siglip2-so400m-patch16-384",
    "google/owlvit-large-patch14",
    "google/owlv2-large-patch14-ensemble",
    "facebook/metaclip-h14-fullcc2.5b",
    "Salesforce/blip2-opt-2.7b"
],
    "Sample_223": [
    "runwayml/stable-diffusion-v1-5",
    "CompVis/stable-diffusion-v1-4",
    "stabilityai/stable-diffusion-2-1",
    "stabilityai/stable-diffusion-2-1-base",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    "stabilityai/stable-diffusion-3-medium-diffusers",
    "black-forest-labs/FLUX.1-dev",
    "black-forest-labs/FLUX.1-schnell",
    "kandinsky-community/kandinsky-2-2-decoder"
],
    "Sample_224": [
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistralai/Mistral-7B-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "google/flan-t5-xl",
    "google/flan-t5-xxl",
    "Qwen/Qwen2.5-7B-Instruct",
    "openai/clip-vit-large-patch14",
    "google/siglip-so400m-patch14-384"
],
    "Sample_229": [
    "google-t5/t5-large",
    "google-t5/t5-3b",
    "google/flan-t5-large",
    "google/flan-t5-xl",
    "facebook/bart-large",
    "facebook/bart-base",
    "tuner007/pegasus_paraphrase",
    "Salesforce/codet5-large",
    "google/t5-v1_1-large",
    "microsoft/GODEL-v1_1-large-seq2seq"
],
    "Sample_230": [
    "google/vit-base-patch16-224-in21k",        # Canonical ViT-Base, ImageNet-21k pretrained
    "google/vit-large-patch16-224-in21k",        # ViT-Large for scale ablations
    "microsoft/swin-base-patch4-window7-224-22k",# Swin Transformer (hierarchical ViT variant)
    "microsoft/swin-large-patch4-window12-384-22k", # Swin-Large, high-res generalization
    "facebook/deit-base-distilled-patch16-224",  # DeiT-Base, distillation-based ViT
    "facebook/dinov2-base",                      # DINOv2 self-supervised ViT backbone
    "facebook/dinov2-large",                     # DINOv2-Large, SOTA self-supervised features
    "openai/clip-vit-base-patch16",              # CLIP ViT-Base, vision-language pretraining
    "openai/clip-vit-large-patch14",             # CLIP ViT-Large, richer patch embeddings
    "timm/vit-base-patch16-clip-224.openai",     # timm-wrapped CLIP ViT for easy PEFT integration
],
    "Sample_231": [
    "allenai/specter2",
    "allenai/specter2_base",
    "allenai/specter2_classification",
    "allenai/scibert_scivocab_uncased",
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
    "NeuML/pubmedbert-base-embeddings",
    "BioMistral/BioMistral-7B",
    "stanford-crfm/BioMedLM",
    "abhinand/MedEmbed-large-v0.1"
],
    "Sample_237": [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "intfloat/e5-large-instruct",
    "intfloat/e5-small-v2",
    "sentence-transformers/all-MiniLM-L12-v2",
    "thenlper/gte-base",
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-small-en-v1.5",
    "hkunlp/instructor-base",
    "intfloat/multilingual-e5-base"
],
    "Sample_246": [
    "stabilityai/stable-diffusion-2-1",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "stabilityai/sdxl-turbo",
    "lllyasviel/sd-controlnet-canny",
    "diffusers/controlnet-canny-sdxl-1.0",
    "runwayml/stable-diffusion-v1-5",
    "stabilityai/stable-diffusion-3-medium-diffusers",
    "timbrooks/instruct-pix2pix",
    "kandinsky-community/kandinsky-2-2-decoder",
    "thu-ml/unidiffuser-v1"
],
    "Sample_247": [
    "stabilityai/stable-diffusion-2-inpainting",
    "runwayml/stable-diffusion-inpainting",
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    "Fantasy-Studio/Paint-by-Example",
    "lllyasviel/control_v11p_sd15_inpaint",
    "h94/IP-Adapter",
    "TencentARC/BrushNet",
    "kandinsky-community/kandinsky-2-2-decoder-inpaint",
    "xichenhku/AnyDoor",
    "Huggingface/MAT"
],
    "Sample_264": [
    "Salesforce/blip2-opt-2.7b",
    "Salesforce/blip-image-captioning-large",
    "Salesforce/blip-image-captioning-base",
    "microsoft/git-large-coco",
    "OFA-Sys/ofa-large",
    "Salesforce/instructblip-vicuna-7b",
    "llava-hf/llava-1.5-7b-hf",
    "nlpconnect/vit-gpt2-image-captioning",
    "google/paligemma-3b-pt-224",
    "HuggingFaceM4/idefics2-8b"
],
    "Sample_265": [
    "OFA-Sys/ofa-large",
    "Salesforce/blip2-opt-2.7b",
    "Salesforce/blip2-flan-t5-xl",
    "microsoft/git-large-textcaps",
    "google/paligemma-3b-mix-224",
    "j-min/VL-T5",
    "microsoft/unilm-base-cased",
    "MAGAer13/mplug-owl2-llama2-7b",
    "openflamingo/OpenFlamingo-9B-vitl-mpt7b",
    "allenai/uio2-large"
],
    "Sample_268":  [
    "microsoft/deberta-v3-base",
    "microsoft/deberta-v3-large",
    "google/electra-large-discriminator",
    "google/electra-base-discriminator",
    "FacebookAI/roberta-large",
    "google-bert/bert-large-uncased",
    "distilbert/distilbert-base-uncased",
    "albert/albert-xxlarge-v2",
    "xlnet/xlnet-large-cased",
    "google/mobilebert-uncased"
],
    "Sample_281": [
    "allenai/specter2_base",
    "allenai/scibert_scivocab_uncased",
    "allenai/specter",
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "dmis-lab/biobert-large-cased-v1.1",
    "malteos/scincl",
    "allenai/specter2_proximity",
    "globuslabs/ScholarBERT",
    "michiyasunaga/BioLinkBERT-large",
    "allenai/cs_roberta_base"
],
    "Sample_290": [
    "facebook/bart-large-cnn",
    "google/pegasus-cnn_dailymail",
    "google/pegasus-xsum",
    "pszemraj/led-large-book-summary",
    "google/bigbird-pegasus-large-arxiv",
    "google-t5/t5-large",
    "google/flan-t5-large",
    "facebook/bart-large-xsum",
    "google/long-t5-tglobal-large",
    "google/mt5-large"
],
    "Sample_292": [
    "MBZUAI/geochat-7B",
    "Salesforce/blip2-flan-t5-xxl",
    "openai/clip-vit-large-patch14",
    "flax-community/remoteclip",
    "OpenGVLab/InternVL3-8B",
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "Salesforce/blip2-opt-2.7b",
    "llava-hf/llava-1.5-7b-hf",
    "microsoft/Florence-2-large",
    "Qwen/Qwen2-VL-7B-Instruct",
],
    "Sample_293": [
    "microsoft/layoutlmv3-base",
    "microsoft/layoutlmv3-large",
    "microsoft/udop-large",
    "naver-clova-ix/donut-base",
    "google/pix2struct-base",
    "microsoft/dit-base",
    "microsoft/dit-large",
    "microsoft/layoutlmv2-base-uncased",
    "google/pix2struct-docvqa-base",
    "Qwen/Qwen2-VL-7B-Instruct",
],
    "Sample_296": [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "google/gemma-2-9b-it",
    "google/flan-t5-xl",
    "allenai/led-base-16384",
    "google/flan-t5-xxl",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
]
}

dict1 = {"a": 1}
dict2 = {"b": 2}
dict3 = {"c": 3}

combined = {
    "CLAUDE_RECOMMENDATIONS": CLAUDE_RECOMMENDATIONS,
    "PERPLEXITY_RECOMMENDATIONS": PERPLEXITY_RECOMMENDATIONS,
    "GEMINI_RECOMMENDATIONS": GEMINI_RECOMMENDATIONS,
    "CHATGPT_RECOMMENDATIONS": CHATGPT_RECOMMENDATIONS
}

import json
with open('8-CRITERIA_SELECTION/F_Hits_United/combined_recommendations.json', 'w') as f:
    json.dump(combined, f, indent=4)
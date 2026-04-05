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
    "facebook/bart-large-cnn",
    "facebook/bart-base",
    "facebook/bart-large-xsum",
    "google/pegasus-cnn_dailymail",
    "google/pegasus-xsum",
    "google/flan-t5-large",
    "google/flan-t5-xl",
    "google-t5/t5-large",
    "allenai/led-base-16384",
    "allenai/led-large-16384"
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
    "sample_90":[
    "microsoft/deberta-v3-large",           # Current SOTA for NLU classification
    "roberta-large",                        # The essential industry-standard baseline
    "microsoft/deberta-v2-xxlarge",         # Highest performing pure-encoder for reasoning
    "google/electra-large-discriminator",   # Efficient discriminator-based pre-training
    "google/t5-large",                      # Versatile encoder-decoder for rank-based scoring
    "FacebookAI/xlm-roberta-large",         # Top-tier multilingual encoder performance
    "allenai/longformer-large-4096",        # Necessary for long-document context QA
    "bigscience/mt0-large",                 # Instruction-tuned multilingual benchmark
    "albert-xxlarge-v2",                    # High-capacity reasoning via parameter sharing
    "bert-base-uncased"                     # Essential baseline for performance delta tracking
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

}
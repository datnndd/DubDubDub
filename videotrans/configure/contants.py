# ------------- Punctuation and Space Languages -----------
# CJK, Thai, Khmer, Cantonese do not use spaces between words
CJK_LANG = ["zh", "ja", "ko", "yu", "th", "km", "yue"]
# Common punctuation
PUNC_FLAGS = [",", ".", "?", "!", ";", "，", "。", "？", "；", "！"]
# Soft punctuation like comma
PUNC_FLAGS_HALF = [",", "，", "-", "、", ":", "："]
# Sentence-ending punctuation
PUNC_FLAGS_END = [".", "。", "?", "？", "!", "！"]
NON_WORD = r"""^[,.?!;'"_，。？；‘’“”！~@#￥%…&*（【】）｛｝《、》$()\[\]{}=+<>\s-]+$"""

# ------------ Proxy bypass domain list ---------------
# Domains that bypass proxy
_no_proxy_list = [
    # --- Tencent Cloud ---
    "tencentcloudapi.com", ".tencentcloudapi.com",

    # --- HuggingFace ---
    "hf-mirror.com", ".hf-mirror.com",

    # --- Baidu (including fanyi.baidu and all subdomains) ---
    "baidu.com", ".baidu.com",

    # --- ByteDance (including openspeech and all subdomains) ---
    "bytedance.com", ".bytedance.com", ".volces.com", "volces.com",

    # --- MiniMax ---
    "api.minimaxi.com", ".minimaxi.com",

    # --- DeepSeek ---
    "api.deepseek.com", ".deepseek.com",

    # --- ModelScope ---
    "modelscope.cn", ".modelscope.cn",

    # --- Alibaba Cloud (including dashscope, aliyuncs, etc.) ---
    "aliyuncs.com", ".aliyuncs.com",

    # --- SiliconFlow ---
    "siliconflow.cn", ".siliconflow.cn",

    "ms.show", ".ms.show",
    "bigmodel.cn", ".bigmodel.cn",

    "api.gradio.app", ".api.gradio.app",
    # "microsoft.com", ".microsoft.com", # covers tts.speech.microsoft.com

    # --- Local loopback (covers all ports: 7860, 8000, 9880, 5051, etc.) ---
    "localhost",
    "127.0.0.1",
    "127.0.0.2",
    "0.0.0.0",
]
no_proxy = ",".join(_no_proxy_list)

# ---------------- Supported Video/Audio Formats ---------------
# Supported video formats
VIDEO_EXTS = ["mp4", "mkv", "mpeg", "avi", "mov", "mts", "webm", "ogg", "ts", "flv", "wmv"]
# Supported audio formats
AUDIO_EXITS = ["mp3", "wav", "aac", "flac", "m4a", "ogg", "wma"]

# ----------- Default Model Names -----------------
FASTER_MODELS_DICT = {"large-v3": "Systran/faster-whisper-large-v3"}
# Deepgram supported speech recognition models
DEEPGRAM_MODEL = [
    "nova-3",
    "whisper-large",
    "whisper-medium",
    "whisper-small",
    "whisper-base",
    "whisper-tiny",
    "nova-2",
    "enhanced",
    "base",
]

# Default Gemini model
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash,gemini-1.5-flash,gemini-1.5-pro,gemini-flash-latest,gemini-pro-latest"
# Gemini TTS voice roles
GEMINITTS_ROLES = "Zephyr,Puck,Charon,Kore,Fenrir,Leda,Orus,Aoede,Callirrhoe,Autonoe,Enceladus,Iapetus,Umbriel,Algieba,Despina,Erinome,Algenib,Rasalgethi,Laomedeia,Achernar,Alnilam,Schedar,Gacrux,Pulcherrima,Achird,Zubenelgenubi,Vindemiatrix,Sadachbia,Sadaltager,Sulafat"

GEMINI_TTS_MODELS = "gemini-2.5-flash-preview-tts,gemini-2.5-pro-preview-tts,gemini-2.0-flash"

Chatgpt_Model = "gpt-4o,gpt-4o-mini,o1,o1-mini,o3-mini,gpt-4-turbo,gpt-4,gpt-3.5-turbo"
Deepseek_Model = "deepseek-chat,deepseek-reasoner"
Whisper_Models = "large-v3"
ELEVENLABS_TTS_MODELS = "eleven_multilingual_v2,eleven_flash_v2_5,eleven_flash_v2,eleven_turbo_v2_5,eleven_multilingual_v1"


# -------------- Model Download URLs -------------------
# Built-in speaker diarization download URLs
BUILTINT_URL_MS = [
    "https://www.modelscope.cn/models/himyworld/videotrans/resolve/master/onnx/seg_model.onnx",
    "https://www.modelscope.cn/models/himyworld/videotrans/resolve/master/onnx/nemo_en_titanet_small.onnx",
    "https://www.modelscope.cn/models/himyworld/videotrans/resolve/master/onnx/3dspeaker_speech_eres2net_large_sv_zh-cn_3dspeaker_16k.onnx"
]
BUILTINT_URL_HF = [
    "https://huggingface.co/mortimerme/repocollect/resolve/main/onnx/seg_model.onnx?download=true",
    "https://huggingface.co/mortimerme/repocollect/resolve/main/onnx/nemo_en_titanet_small.onnx?download=true",
    "https://huggingface.co/mortimerme/repocollect/resolve/main/onnx/3dspeaker_speech_eres2net_large_sv_zh-cn_3dspeaker_16k.onnx?download=true"
]

# Built-in punctuation restoration model download URLs
PUNC_RESTORE_MS = [
    "https://www.modelscope.cn/models/himyworld/videotrans/resolve/master/puntc/model.onnx",
    "https://www.modelscope.cn/models/himyworld/videotrans/resolve/master/puntc/config.yaml",
    "https://www.modelscope.cn/models/himyworld/videotrans/resolve/master/puntc/tokens.json",
]
PUNC_RESTORE_HF = [
    "https://huggingface.co/mortimerme/repocollect/resolve/main/puntc/model.onnx?download=true",
    "https://huggingface.co/mortimerme/repocollect/resolve/main/puntc/config.yaml?download=true",
    "https://huggingface.co/mortimerme/repocollect/resolve/main/puntc/tokens.json?download=true",
]
# Denoise model download URLs
DENOISE_URL_MS = [
    'https://modelscope.cn/models/himyworld/videotrans/resolve/master/onnx/dpdfnet8.onnx'
]
DENOISE_URL_HF = [
    'https://huggingface.co/mortimerme/repocollect/resolve/main/onnx/dpdfnet8.onnx?download=true'
]
# Background audio separation URL prefix
UVR_URL_MS = 'https://www.modelscope.cn/models/himyworld/videotrans/resolve/master/onnx/{}'
UVR_URL_HF = 'https://huggingface.co/mortimerme/repocollect/resolve/main/onnx/{}?download=true'
# realtime stt
REALTIME_URL_MS='https://modelscope.cn/models/himyworld/videotrans/resolve/master/realtimestt.zip'
REALTIME_URL_HF='https://huggingface.co/mortimerme/repocollect/resolve/main/realtimestt.zip?download=true'

# ---------- Rubberband Library Installation Tips ------------------
INSTALL_RUBBERBAND_TIPS = """Windows: For Windows systems, please download the file, extract it, and place it in the ffmpeg folder in the current directory. Use a better audio acceleration algorithm\nhttps://breakfastquay.com/files/releases/rubberband-4.0.0-gpl-executable-windows.zip
Darwin: `brew install rubberband`  and  `uv add pyrubberband` Use a better audio acceleration algorithm
Linux: `sudo apt install rubberband-cli libsndfile1-dev` and `uv add pyrubberband`  Use a better audio acceleration algorithm"""

# -------- Progress Status Constants -----------------------
END_STATUS = "end"
ERROR_STATUS = "error"
SUCCEED_STATUS = "succeed"
STOP_STATUS = "stop"
ING_STATUS = "ing"

# ------------ Voice Preview Audition Texts ---------------
LISTEN_TEXT = {
    "zh": "你好啊，我亲爱的朋友，希望你的每一天都是美好愉快的！",
    "zh-cn": "你好啊，我亲爱的朋友，希望你的每一天都是美好愉快的！",
    "zh-tw": "你好啊，我親愛的朋友，希望你的每一天都是美好愉快的！",
    "ro": "Bună, draga mea prietenă, sper ca fiecare zi a ta să fie minunată și plină de bucurie!",
    "km": "សួស្តីមិត្តជាទីស្រឡាញ់របស់ខ្ញុំ ខ្ញុំសង្ឃឹមថារាល់ថ្ងៃរបស់អ្នកគឺអស្ចារ្យ និងរីករាយ។!",
    "nb": "Hallo, min kjære venn, jeg håper hver dag din er fantastisk og gledelig.",
    "en": "Hello, my dear friend. I hope your every day is beautiful and enjoyable!",
    "fr": "Bonjour mon cher ami. J'espère que votre quotidien est beau et agréable !",
    "de": "Hallo mein lieber Freund. Ich hoffe, dass Ihr Tag schön und angenehm ist!",
    "ja": "こんにちは私の親愛なる友人。 あなたの毎日が美しく楽しいものでありますように！",
    "ko": "안녕, 내 사랑하는 친구. 당신의 매일이 아름답고 즐겁기를 바랍니다!",
    "ru": "Привет, мой дорогой друг. Желаю, чтобы каждый твой день был прекрасен и приятен!",
    "es": "Hola mi querido amigo. ¡Espero que cada día sea hermoso y agradable!",
    "th": "สวัสดีเพื่อนรัก. ฉันหวังว่าทุกวันของคุณจะสวยงามและสนุกสนาน!",
    "it": "Ciao caro amico mio. Spero che ogni tuo giorno sia bello e divertente!",
    "el": "Γεια σου, αγαπητέ μου φίλε. Εύχομαι κάθε σου μέρα να είναι όμορφη και ευχάριστη!",
    "pt": "Olá meu querido amigo. Espero que todos os seus dias sejam lindos e agradáveis!",
    "vi": "Xin chào người bạn thân yêu của tôi. Tôi hy vọng mỗi ngày của bạn đều đẹp và thú vị!",
    "ar": "مرحبا صديقي العزيز. أتمنى أن يكون كل يوم جميلاً وممتعًا!",
    "tr": "Merhaba sevgili arkadaşım. Umarım her gününüz güzel ve keyifli geçer!",
    "hi": "नमस्ते मेरे प्यारे दोस्त। मुझे आशा है कि आपका हर दिन सुंदर और आनंददायक हो!!",
    "hu": "Helló kedves barátom. Remélem minden napod szép és kellemes!",
    "uk": "Привіт, мій дорогий друже, сподіваюся, ти щодня прекрасна!",
    "id": "Halo, temanku, semoga kamu cantik setiap hari!",
    "ms": "Helo, sahabat saya, saya harap anda cantik setiap hari!",
    "kk": "Сәлеметсіз бе, менің қымбатты досым, сендер күн сайын әдемісің деп үміттенемін!",
    "cs": "Ahoj, můj drahý příteli, doufám, že jsi každý den krásná!",
    "pl": "Witam, mój drogi przyjacielu, mam nadzieję, że jesteś piękna każdego dnia!",
    "nl": "Hallo mijn lieve vriend, ik hoop dat elke dag goed en fijn voor je is!!",
    "sv": "Hej min kära vän, jag hoppas att varje dag är en bra och trevlig dag för dig!",
    "he": "שלום, ידידי היקר, אני מקווה שכל יום בחייך יהיה נפלא ומאושר!",
    "bn": "হ্যালো, আমার প্রিয় বন্ধু, আমি আশা করি আপনার জীবনের প্রতিটি দিন চমৎকার এবং সুখী হোক!",
    "fil": "Hello, kaibigan ko",
    "fa": "سلام دوستای گلم امیدوارم هر روز از زندگیتون عالی و شاد باشه.",
    "ur": "ہیلو پیارے دوست، مجھے امید ہے کہ آپ آج خوش ہوں گے۔",
    "yue": "你好啊親愛嘅朋友，希望你今日好開心",

}

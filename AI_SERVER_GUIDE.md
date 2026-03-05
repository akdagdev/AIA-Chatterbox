# AIA-Chatterbox: AI Server Implementation Guide

Bu doküman, AIA-Chatterbox projesini bir kütüphane olarak kullanarak bir API/Server oluşturacak Yapay Zeka (AI) geliştiricisi için hazırlanmıştır. Dokümanda modelin nasıl yükleneceği, ses üretiminin (normal ve batch) nasıl yapılacağı ve performans için hangi best-practice'lerin uygulanması gerektiği anlatılmaktadır.

---

## 1. Kurulum ve Başlatma

Sistemin kalbini `ChatterboxMultilingualTTS` sınıfı oluşturmaktadır. Bu sınıf, metinleri sese çevirmek için gerekli olan T3 (Text-to-Token) ve S3Gen (Token-to-Wav) modellerini barındırır.

```python
import torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, Conditionals, SpeechRequest

# Cihaz seçimi (GPU önerilir)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Modeli yükleme
# HuggingFace üzerinden önceden eğitilmiş modeli indirmek/yüklemek için:
model = ChatterboxMultilingualTTS.from_pretrained(device)

# Veya lokal bir dizinden yüklemek için:
# model = ChatterboxMultilingualTTS.from_local("/path/to/checkpoint", device)
```

---

## 2. Voice Embedding Çıkarma (`extract_voice_embedding`)

Bu fonksiyon, bir referans ses dosyasından `Conditionals` (voice embedding) nesnesi üretir. Bu nesne daha sonra `generate()` ve `generate_batch()` fonksiyonlarına doğrudan verilebilir. **Model üzerindeki dahili state'i (self.conds) değiştirmez**, yani tamamen yan etkisizdir (side-effect free).

```python
# Bir kere hesapla, istediğin kadar kullan
voice_embedding = model.extract_voice_embedding("reference_voice.wav", exaggeration=0.5)

# Döndürülen nesne bir Conditionals objesidir.
# type(voice_embedding) -> <class 'chatterbox.mtl_tts.Conditionals'>
```

### Diske Kaydetme ve Yükleme

Voice embedding hesaplamak maliyetli bir işlemdir. Hesaplanan embedding'i diske kaydedip daha sonra tekrar yükleyerek kullanabilirsiniz.

```python
# Kaydet
voice_embedding.save("user_123_voice.pt")

# Yükle (farklı bir session'da bile olabilir)
loaded_embedding = Conditionals.load("user_123_voice.pt", map_location=device)
```

### Bellekte Önbellekleme (Server Kullanımı)

Server yapısında her kullanıcı için ayrı bir voice embedding hesaplayıp bellekte tutmak en hızlı yaklaşımdır.

```python
# Server başlatılırken veya kullanıcı ses yüklediğinde
embedding_cache: dict[str, Conditionals] = {}

# Kullanıcı ses yüklediğinde
embedding_cache["user_123"] = model.extract_voice_embedding("user_123_ref.wav")

# Her TTS isteğinde cache'ten al
user_embedding = embedding_cache["user_123"]
```

---

## 3. Normal (Tekli) Üretim (Single Output Generation)

Tek bir cümlenin veya kısa bir metnin sentezlenmesi için `model.generate()` fonksiyonu kullanılır.

### 3.1: Dosya yolu ile (basit kullanım)

```python
import torchaudio

audio_tensor = model.generate(
    text="Merhaba, bu bir test sesidir.",
    language_id="tr",
    audio_prompt_path="reference_voice.wav",  # Her çağrıda embedding yeniden hesaplanır!
    exaggeration=0.5,
    cfg_weight=0.5,
)
torchaudio.save("output.wav", audio_tensor.cpu(), model.sr)
```

### 3.2: Voice Embedding ile (önerilen, hızlı)

`extract_voice_embedding` ile önceden çıkarılan `Conditionals` nesnesini `conditionals` parametresine doğrudan verin. Bu sayede her çağrıda referans ses dosyasının yeniden analiz edilmesi engellenir.

```python
# Önce embedding'i bir kere çıkar
voice_embedding = model.extract_voice_embedding("reference_voice.wav")

# Sonra istediğin kadar kullan — her çağrıda yeniden hesaplama yapılmaz
audio1 = model.generate(
    text="Birinci cümle.",
    language_id="tr",
    conditionals=voice_embedding,  # ← Doğrudan voice embedding
    cfg_weight=0.5,
)

audio2 = model.generate(
    text="İkinci cümle.",
    language_id="tr",
    conditionals=voice_embedding,  # ← Aynı embedding, sıfır ek maliyet
    cfg_weight=0.5,
)
```

> **Öncelik Sırası:** `conditionals` > `audio_prompt_path` > `self.conds` (önceden yüklenmiş). Eğer `conditionals` verilmişse diğer ikisi göz ardı edilir.

---

## 4. Batch (Toplu) Üretim (Batch Output Generation)

Yüksek hacimli ve hızlı üretim (özellikle kitap veya uzun metin seslendirmeleri) yaparken `generate_batch` fonksiyonundan faydalanmalısınız. Bu sayede GPU belleğini tam kapasite kullanarak aynı anda birden fazla cümleyi işleyebilirsiniz.

### Nasıl Çalışır (Dynamic Batching)?

Sisteme farklı kullanıcılardan, farklı dillerde istekler gelebilir. Her çağrıyı tek tek bekletmek yerine `SpeechRequest` nesnelerinden oluşan bir liste (batch) hazırlanıp toplu olarak işlenmesi sağlanır:

```python
from chatterbox.mtl_tts import SpeechRequest

# Voice embedding'leri daha önce extract_voice_embedding ile çıkarılmış ve önbelleğe (cache) alınmış olmalıdır
voice_user1 = embedding_cache["user1"]
voice_user2 = embedding_cache["user2"]

# Gelen veya sırada bekleyen API istekleri:
requests = [
    SpeechRequest(text="Hello world!", language_id="en", conditionals=voice_user1),
    SpeechRequest(text="Merhaba dünya!", language_id="tr", conditionals=voice_user2),
    SpeechRequest(text="How are you?", language_id="en", conditionals=voice_user1),
]

with torch.no_grad():
    # Model farklı sesleri, dilleri ve metinleri tek batch içinde işler
    audio_list = model.generate_batch(
        texts=requests,
        language_id=None,  # SpeechRequest'lerdeki language_id kullanılır
        max_new_tokens=400,
        cfg_weight=0.3
    )

# Sonuçlar listesindeki sıra, requests listesindeki sıra ile aynıdır
for i, audio in enumerate(audio_list):
    torchaudio.save(f"batch_{i}.wav", audio.cpu(), model.sr)
```

---

## 5. Parametreler Referansı

| Parametre | Tip | Varsayılan | Açıklama |
|-----------|-----|------------|----------|
| `text` | `str` | — | Sentezlenecek metin |
| `language_id` | `str` | — | Dil kodu: `"en"`, `"tr"`, `"es"`, `"fr"`, `"de"`, vb. |
| `audio_prompt_path` | `str` | `None` | Referans ses dosya yolu (her çağrıda yeniden hesaplanır) |
| `conditionals` | `Conditionals` | `None` | Önceden hesaplanmış voice embedding (öncelikli) |
| `exaggeration` | `float` | `0.5` | Vurgu/duygu yoğunluğu (0.0 - 1.0) |
| `cfg_weight` | `float` | `0.5` | Referans sese sadakat seviyesi (0.0 - 1.0, yüksek = daha sadık ama yavaş) |
| `temperature` | `float` | `0.8` | Üretim çeşitliliği |
| `max_new_tokens` | `int` | `1000` | Maksimum üretilecek token sayısı |

---

## 6. Server Mimarisi İçin Best Practices

### Global Model Yüklemesi
Model büyüktür ve yüklenmesi zaman alır. Sunucu başlarken tek bir global instance (örn: FastAPI `lifespan`) içerisinde yüklenmelidir. Her istekte yeniden yüklenmemelidir.

### Voice Embedding Cache
```python
embedding_cache: dict[str, Conditionals] = {}

# Kullanıcı ses yüklediğinde bir kere hesapla
embedding_cache[user_id] = model.extract_voice_embedding(uploaded_file_path)

# TTS isteklerinde cache'ten kullan
audio = model.generate(text=text, language_id="tr", conditionals=embedding_cache[user_id])
```

### Uzun Metin Chunking + Batch
```python
import re

def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]

sentences = split_sentences(long_text)

# Batch'ler halinde üret
BATCH_SIZE = 4
all_audio = []
for i in range(0, len(sentences), BATCH_SIZE):
    batch = sentences[i:i+BATCH_SIZE]
    reqs = [SpeechRequest(text=s, language_id="tr", conditionals=voice_embedding) for s in batch]
    audios = model.generate_batch(texts=reqs, language_id=None, cfg_weight=0.5)
    all_audio.extend(audios)

# Araya sessizlik ekleyerek birleştir
import torch
silence = torch.zeros(1, int(0.2 * model.sr))
final_parts = []
for a in all_audio:
    final_parts.extend([a.cpu(), silence])
final_audio = torch.cat(final_parts, dim=1)
```

### GPU Isınma (Warm-up)
Server tamamen hazır olmadan önce CUDA grafiklerini aktif etmek için boş bir üretim yapılmalıdır:
```python
model.generate(text="warmup", language_id="en", conditionals=default_voice)
```

### Thread Safety (Eşzamanlılık)
`generate()` fonksiyonu **tamamen thread-safe** (iş parçacığı güvenli) hale getirilmiştir. Yani farklı kullanıcı istekleri aynı anda sorunsuzca `generate()` çağırabilir. 

Ancak `generate_batch()` içerisinde modelin dahili bazı bileşenleri hala tam izole olmayabilir. API frameworkleri (FastAPI) ile yoğun istek karşılanırken `generate_batch`'in aynı anda birden çok paralel (ör: 5 farklı worker tarafından aynı anda 5 `generate_batch`) çalışması yerine; sisteme giren isteklerin (request) bir havuzda/kuyrukta (Celery vb.) biriktirilip tek bir worker tarafından **dinamik batch** yapılarak (sayı 8'i veya 16'yı bulduğunda tek `generate_batch` ile) işlenmesi en iyi GPU/Hız performansını verecektir.

---

## 7. Desteklenen Diller

| Kod | Dil | Kod | Dil |
|-----|-----|-----|-----|
| `ar` | Arapça | `ko` | Korece |
| `da` | Danca | `ms` | Malayca |
| `de` | Almanca | `nl` | Felemenkçe |
| `el` | Yunanca | `no` | Norveççe |
| `en` | İngilizce | `pl` | Lehçe |
| `es` | İspanyolca | `pt` | Portekizce |
| `fi` | Fince | `ru` | Rusça |
| `fr` | Fransızca | `sv` | İsveççe |
| `he` | İbranice | `sw` | Svahili |
| `hi` | Hintçe | `tr` | Türkçe |
| `it` | İtalyanca | `zh` | Çince |
| `ja` | Japonca | | |

---

## 8. Import Referansı

```python
from chatterbox.mtl_tts import (
    ChatterboxMultilingualTTS,  # Ana model sınıfı
    Conditionals,               # Voice embedding nesnesi
    SpeechRequest,              # Batch üretim için istek nesnesi
)
```

# MLLMSampleKit

`MLLMSampleKit` converts raw dataset rows into `CanonicalMLLMSample`.

Use it for:

- role aliases such as `human` -> `user` and `gpt` -> `assistant`;
- raw field names such as `messages`, `conversations`, `images`, `img_size`;
- placeholder parsing, for example `<image>`;
- ordered `CanonicalMedia` records that match placeholder order.

Do not decode images, sample video frames, tokenize text, or inspect model
special tokens in SampleKit.

Subclass when the raw schema changes:

```python
from mvp_engine.kit.mllm import MLLMSampleKit


class MySampleKit(MLLMSampleKit):
    def normalize(self, sample, *, image_placeholder=None):
        canonical = super().normalize(sample, image_placeholder=image_placeholder)
        # Adapt only raw-schema details here.
        return canonical
```

For a new modality, emit `CanonicalMedia(type="video", value=..., size=...,
metadata=...)` and a matching message block such as `{"type": "video",
"video": ...}`. Heavy IO belongs in MediaKit.

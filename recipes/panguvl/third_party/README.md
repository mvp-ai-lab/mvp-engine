This folder contains patch code so that the OpenPang-VL code can be used within the mvp-recipe and GPU.
-documented cache_position
-It wrote a private deprecated _process_images() inside the OpenPanguVLImageProcessorFast() class in the imageprocessor_openpangu_vl.py, because the mvp-engine transformers version do not support the _process_images() private hf processor function anymore.
-It removed LossKwargs, torch_npu and npu_flash_attention from the modeling_openpangu_embedded.py and modeling_openpangu_vl.py scripts.
-It documented cache_position inside the forward signature.
-It sets the pad_token to eos_token_id inside the PanguEmbeddedModel()
-It creates an initializer_range=0.2 for the vision config inside the configuration_openpangu_vl.py
-It implements and registers default_rope script inside the modeling_openpangu_embedded.py and modeling_openpangu_vl.py scripts.



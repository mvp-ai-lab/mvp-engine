import nvidia.dali as dali
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator


def dali_wrapper(wds_iterator, batch_size, resize=None, device_id=0):
    pipe = dali.pipeline.Pipeline(batch_size=batch_size, num_threads=4, device_id=device_id)
    
    with pipe:
        images, depths, labels = fn.external_source(
            source=wds_iterator, 
            num_outputs=3,
            dtype=[dali.types.UINT8, dali.types.UINT8, dali.types.INT64],
            batch=True,
        )
        
        images = fn.decoders.image(images, device="mixed", output_type=dali.types.RGB)
        depths = fn.experimental.decoders.image(
            depths, 
            device="mixed",
            output_type=dali.types.GRAY,
            dtype=dali.types.UINT16
        )

        if resize is not None:
            images = fn.resize(images, resize_x=resize[0], resize_y=resize[1])
            depths = fn.resize(depths, resize_x=resize[0], resize_y=resize[1], interp_type=dali.types.INTERP_NN)
        
        depths = fn.cast(depths, dtype=dali.types.FLOAT)

        pipe.set_outputs(images, depths, labels)


    dali_loader = DALIGenericIterator(
        [pipe], 
        output_map=['images', 'depths', 'labels'], 
        auto_reset=True
    )
    return dali_loader
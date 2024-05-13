
# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# developer: dubm
# Copyright © 2023 Bitmind

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import List
from pydantic import root_validator, validator
from io import BytesIO
from PIL import Image
import bittensor as bt
import pydantic
import base64


def prepare_image_synapse(images, predictions):

    b64_encoded_images = []
    for image in images:
        if image is None:
            print("Warning: None image")
            continue

        image = image.resize((256, 256))
        
        image_bytes = BytesIO()
        image.save(image_bytes, format="JPEG")
        encoded = base64.b64encode(image_bytes.getvalue())
        b64_encoded_images.append(encoded)

    return ImageSynapse(images=b64_encoded_images, predictions=predictions)

# ---- miner ----
# Example usage:
#   def miner_forward( synapse: ImageSynapse ) -> ImageSynapse:
#       ...
#       synapse.predictions = deepfake_detection_model_outputs
#       return synapse
#   axon = bt.axon().attach( miner_forward ).serve(netuid=...).start()

# ---- validator ---
# Example usage:
#   dendrite = bt.dendrite()
#   b64_images = [b64_img_1, ..., b64_img_n]
#   predictions = dendrite.query( ImageSynapse( images = b64_images ) )
#   assert len(predictions) == len(b64_images)


class ImageSynapse(bt.Synapse):
    """
    This protocol helps in handling image/prediction request and response communication between
    the miner and the validator.

    Attributes:
    - images: a list of bas64 encoded images
    - predictions: a list of floats (of equal length to images) indicating the probabilty that each
        image is AI generated. >.5 is considered a deepfake, <= 0.5 is considered real.
    """

    # Required request input, filled by sending dendrite caller.
    images: List[str] = pydantic.Field(
        title="Images",
        description="A list of base64 encoded images to check",
        allow_mutation=True
    )

    # Optional request output, filled by receiving axon.
    predictions: List[float] = pydantic.Field(
        title="Predictions",
        description="A list of deep fake probabilities"
    )

    def deserialize(self) -> List[float]:
        """
        Deserialize the output. This method retrieves the response from
        the miner, deserializes it and returns it as the output of the dendrite.query() call.

        Returns:
        - List[float]: The deserialized response, which in this case is the list of deepfake
        prediction probabilities
        """
        return self.predictions

# view_fashion_mnist.py
import gzip
import struct
import random

import numpy as np
import matplotlib.pyplot as plt


def read_idx_images(filename):
    """Load images from an IDX image file, optionally gzip-compressed."""
    open_file = gzip.open if filename.endswith(".gz") else open

    with open_file(filename, "rb") as f:
        # IDX image header: magic number, number of images, rows, columns
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))

        if magic != 2051:
            raise ValueError(
                f"{filename} is not an IDX image file; expected magic 2051, got {magic}"
            )

        pixels = np.frombuffer(f.read(), dtype=np.uint8)

    expected_pixels = count * rows * cols
    if pixels.size != expected_pixels:
        raise ValueError(
            f"Expected {expected_pixels} pixels but found {pixels.size}. "
            "The file may be incomplete or corrupted."
        )

    return pixels.reshape(count, rows, cols)


# Change this to wherever you put the Fashion-MNIST file.
path = "train-images-idx3-ubyte.gz"

images = read_idx_images(path)

index = random.randrange(len(images))
image = images[index]

print(f"Loaded {len(images)} images of shape {images.shape[1:]}")
print(f"Showing image #{index}")

plt.imshow(image, cmap="gray", vmin=0, vmax=255)
plt.title(f"Fashion-MNIST training image #{index}")
plt.axis("off")
plt.show()

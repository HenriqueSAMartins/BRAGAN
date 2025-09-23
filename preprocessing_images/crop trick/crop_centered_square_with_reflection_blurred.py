import os
import cv2
import numpy as np

def crop_centered_square_with_reflection(image, bbox, output_size=None, blur_ksize=None):
    h_img, w_img = image.shape[:2]
    x_min, y_min, bw, bh = bbox
    cx = x_min + bw / 2
    cy = y_min + bh / 2
    max_dim = int(np.ceil(max(bw, bh)))

    x1 = int(np.floor(cx - max_dim / 2))
    y1 = int(np.floor(cy - max_dim / 2))
    x2 = x1 + max_dim
    y2 = y1 + max_dim

    pad_left   = max(0, -x1)
    pad_top    = max(0, -y1)
    pad_right  = max(0, x2 - w_img)
    pad_bottom = max(0, y2 - h_img)

    if any([pad_left, pad_top, pad_right, pad_bottom]):
        image_reflected = cv2.copyMakeBorder(
            image,
            top=pad_top,
            bottom=pad_bottom,
            left=pad_left,
            right=pad_right,
            borderType=cv2.BORDER_REFLECT_101
        )

        if blur_ksize is not None and blur_ksize > 1:
            mask = np.zeros(image_reflected.shape[:2], dtype=np.uint8)
            if pad_top > 0:
                mask[:pad_top, :] = 255
            if pad_bottom > 0:
                mask[-pad_bottom:, :] = 255
            if pad_left > 0:
                mask[:, :pad_left] = 255
            if pad_right > 0:
                mask[:, -pad_right:] = 255

            blurred = cv2.GaussianBlur(image_reflected, (blur_ksize, blur_ksize), 0)
            image_reflected = cv2.bitwise_and(image_reflected, image_reflected, mask=255 - mask)
            blurred_part = cv2.bitwise_and(blurred, blurred, mask=mask)
            image = cv2.add(image_reflected, blurred_part)
        else:
            image = image_reflected

    x1 += pad_left
    y1 += pad_top
    x2 += pad_left
    y2 += pad_top

    cropped = image[y1:y2, x1:x2]

    if output_size is not None:
        cropped = cv2.resize(cropped, (output_size, output_size), interpolation=cv2.INTER_AREA)

    return cropped

# Parâmetros
IMAGES_FOLDER = "body-side"
ANNOTATIONS_FOLDER = "body-side_labels"
OUTPUT_FOLDER = "images_cropped_256_blur"
OUTPUT_SIZE = 256
BLUR_KSIZE = 19  # Use None para desativar o blur

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

for filename in os.listdir(ANNOTATIONS_FOLDER):
    if not filename.endswith(".txt"):
        continue

    name = os.path.splitext(filename)[0]
    image_path = os.path.join(IMAGES_FOLDER, name + ".jpg")
    annotation_path = os.path.join(ANNOTATIONS_FOLDER, filename)

    if not os.path.isfile(image_path):
        print(f"[Aviso] Imagem não encontrada para {name}")
        continue

    image = cv2.imread(image_path)
    h_img, w_img = image.shape[:2]

    with open(annotation_path, "r") as f:
        line = f.readline().strip()
        if not line:
            print(f"[Aviso] Anotação vazia em {annotation_path}")
            continue

        parts = line.split()
        if len(parts) != 5:
            print(f"[Erro] Formato incorreto em {annotation_path}")
            continue

        cls_id, x_center, y_center, width, height = map(float, parts)

        x_center_px = x_center * w_img
        y_center_px = y_center * h_img
        width_px = width * w_img
        height_px = height * h_img
        x_min_px = x_center_px - width_px / 2
        y_min_px = y_center_px - height_px / 2

        bbox_px = (x_min_px, y_min_px, width_px, height_px)

    cropped_img = crop_centered_square_with_reflection(image, bbox_px, OUTPUT_SIZE, blur_ksize=BLUR_KSIZE)
    output_path = os.path.join(OUTPUT_FOLDER, name + ".png")
    cv2.imwrite(output_path, cropped_img)
    print(f"[OK] {output_path} salvo.")

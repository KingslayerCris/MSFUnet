from PIL import Image
from unet import Unet

if __name__ == "__main__":
    unet = Unet()
    mode = "dir_predict"
    count           = False
    name_classes    = ["_background_","vacant land"]
    #   dir_origin_path     指定了用于检测的图片的文件夹路径
    #   dir_save_path       指定了检测完图片的保存路径
    dir_origin_path = "img/"
    dir_save_path   = "img_out/"
    simplify        = True
    onnx_save_path  = "model_data/models.onnx"

    if mode == "dir_predict":
        import os
        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                r_image     = unet.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name))
    elif mode == "export_onnx":
        unet.convert_to_onnx(simplify, onnx_save_path)
                
    else:
        raise AssertionError("Please specify the correct mode: 'dir_predict'.")

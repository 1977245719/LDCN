import torch
import numpy as np
import cv2
from torchvision import transforms as T

from ultralytics.nn.tasks import DetectionModel as Model
from ultralytics.nn.LDCN import LDConv


def plot_offsets(img, save_output, roi_x, roi_y):

    input_img_h, input_img_w = img.shape[:2]
    for offsets in save_output.outputs:
        offset_tensor_h, offset_tensor_w = offsets.shape[2:]
        resize_factor_h, resize_factor_w = input_img_h/offset_tensor_h, input_img_w/offset_tensor_w

        offsets_y = offsets[:, ::2]
        offsets_x = offsets[:, 1::2]

        grid_y = np.arange(0, offset_tensor_h)
        grid_x = np.arange(0, offset_tensor_w)

        grid_x, grid_y = np.meshgrid(grid_x, grid_y)

        sampling_y = grid_y + offsets_y.detach().cpu().numpy()
        sampling_x = grid_x + offsets_x.detach().cpu().numpy()

        sampling_y *= resize_factor_h
        sampling_x *= resize_factor_w

        sampling_y = sampling_y[0] # remove batch axis
        sampling_x = sampling_x[0] # remove batch axis

        sampling_y = sampling_y.transpose(1, 2, 0) # c, h, w -> h, w, c
        sampling_x = sampling_x.transpose(1, 2, 0) # c, h, w -> h, w, c

        sampling_y = np.clip(sampling_y, 0, input_img_h)
        sampling_x = np.clip(sampling_x, 0, input_img_w)

        sampling_y = cv2.resize(sampling_y, dsize=None, fx=resize_factor_w, fy=resize_factor_h)
        sampling_x = cv2.resize(sampling_x, dsize=None, fx=resize_factor_w, fy=resize_factor_h)
        cv2.circle(img, center=(roi_x, roi_y), color=(0, 255, 0), radius=3, thickness=2)
        sampling_y = sampling_y[roi_y, roi_x]
        sampling_x = sampling_x[roi_y, roi_x]
        for y, x in zip(sampling_y, sampling_x):
            y = round(y)
            x = round(x)
            cv2.circle(img, center=(x, y), color=(0, 0, 255), radius=2, thickness=-1)




class SaveOutput:
    def __init__(self):
        self.outputs = []
        
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
        
    def clear(self):
        self.outputs = []

device = 'cuda' if torch.cuda.is_available() else 'cpu'


ckpt = torch.load('your weight')
model_names = ckpt['model'].names
csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32

model = Model("your yaml", ch=3, nc=len(model_names)).to(device)
model.load_state_dict(csd)

save_output = SaveOutput()


for name, layer in model.named_modules():
    if isinstance(layer, LDConv):
        layer.conv_offset_mask.register_forward_hook(save_output)



image = cv2.imread("your image")
image = cv2.resize(image, (640, 640))
input_img_h, input_img_w, channel = image.shape

image_tensor = torch.from_numpy(image) / 255.
image_tensor = image_tensor.view(1, 3, input_img_h, input_img_w)
image_tensor = T.Normalize((0.1307,), (0.3081,))(image_tensor)
image_tensor = image_tensor.to(device)

out = model(image_tensor)


roi_y, roi_x = input_img_h // 2, input_img_w // 2
roi_x = roi_x

plot_offsets(image, save_output, roi_x=roi_x, roi_y=roi_y)

cv2.imshow("image", image)
cv2.waitKey(0)



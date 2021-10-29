from mmdet.apis import init_detector, inference_detector

# config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
config_file = 'configs_uvo_det/uvo/swin_l_carafe_simota_focal_giou_iouhead_tower_dcn_coco_384_uvo_finetune.py'

# download the checkpoint from model zoo and put it in `checkpoints/`
# url: https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
# checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
checkpoint_file = None
device = 'cuda:0'
# init a detector
model = init_detector(config_file, checkpoint_file, device=device)
# inference the demo image
import ipdb; ipdb.set_trace(context=21)
ret = inference_detector(model, 'demo/demo.jpg')
print(type(ret))

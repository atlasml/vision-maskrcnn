import torchvisionfrom sotabench.object_detection import COCO

model = torchvision.models.detection.__dict__['maskrcnn_resnet50_fpn'](num_classes=91, pretrained=True)
test_results = COCO.benchmark(batch_size=8, model=model)

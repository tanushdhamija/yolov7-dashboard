import os
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

import subprocess
import psutil

# to avoid running issues on server
os.environ['DISPLAY'] = ':1.0'

class Yolov7():

    def __init__(self, source, weights='yolov7.pt', view_img=False, img_size=640, trace=False, save_img=True, 
                save_conf=False, device='', augment=False, save_txt=False, conf_thres=0.5, iou_thres=0.5, 
                update=True, classes=None, agnostic_nms=False, name='exp', project='./runs/detect',
                stframe=None, if1_text="", if2_text="", ss1_text="", ss2_text="", ss3_text=""):

        self.source = source
        self.weights = weights
        self.view_img = view_img
        self.save_txt = save_txt
        self.img_size = img_size
        self.trace = trace
        self.save_img = save_img
        self.webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))
        self.project = project
        self.name = name
        self.exist_ok = False
        self.device = device
        self.augment = augment
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.agnostic_nms = agnostic_nms
        self.classes = classes
        self.save_conf = save_conf
        self.update = update
        self.stframe = stframe
        self.if1_text = if1_text
        self.if2_text = if2_text
        self.ss1_text = ss1_text
        self.ss2_text = ss2_text
        self.ss3_text = ss3_text
        
    
    def get_gpu_memory(self):

        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'], encoding='utf-8'
            )
        gpu_memory = [int(x) for x in result.strip().split('\n')]
        return gpu_memory[0]


    @torch.no_grad()
    def detect(self):

        # Directories
        save_dir = Path(increment_path(Path(self.project) / self.name, exist_ok=self.exist_ok))  # increment run
        (save_dir / 'labels' if self.save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        set_logging()
        device = select_device(self.device)
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(self.weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        self.img_size = check_img_size(self.img_size, s=stride)  # check img_size

        if self.trace:
            model = TracedModel(model, device, self.img_size)

        if half:
            model.half()  # to FP16

        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

        # Set Dataloader
        vid_path, vid_writer = None, None
        if self.webcam:
            self.view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(self.source, img_size=self.img_size, stride=stride)
        else:
            dataset = LoadImages(self.source, img_size=self.img_size, stride=stride)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, self.img_size, self.img_size).to(device).type_as(next(model.parameters())))  # run once
        old_img_w = old_img_h = self.img_size
        old_img_b = 1

        t0 = time.time()
        mapped_ = dict()
        
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    model(img, augment=self.augment)[0]

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=self.augment)[0]
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
            t3 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if self.webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results (count)
                    names_ = []
                    cnt = []
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                        names_.append(names[int(c)])
                        cnt.append(int(n.detach().cpu().numpy()))
                    mapped_.update(dict(zip(names_, cnt)))

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if self.save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if self.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if self.save_img or self.view_img:  # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                # Print time (inference + NMS)
                #print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

                # FPS calculation
                fps_infer = round(1 / (t2-t1),1)
                fps_infer_nms = round(1 / (t3-t1),1)
                fps_infer_nms_annotate = round(1 / (time.time()-t1),1)
                if __name__ == '__main__':
                    print(f"FPS(infer):{fps_infer}, FPS(infer+nms):{fps_infer_nms}, FPS(infer+nms+annotate):{fps_infer_nms_annotate}")

                # Stream results
                #if view_img:
                    #cv2.imshow(str(p), im0)
                    #cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if self.save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                        print(f" The image with the result is saved in: {save_path}")
                    else:  # 'video' or 'stream'
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer.write(im0)


# --------------------------------- WEB APP stuff -----------------------------------------
            # view processed image
            self.stframe.image(im0, channels="BGR", use_column_width=True)

            # view system stats
            self.ss1_text.write(str(psutil.virtual_memory()[2])+"%")
            self.ss2_text.write(str(psutil.cpu_percent())+'%')
            try:
                self.ss3_text.write(str(self.get_gpu_memory())+' MB')
            except:
                self.ss3_text.write(str('NA'))

            # view inference stats
            self.if1_text.write(str(fps_infer)+' fps')
            self.if2_text.write(mapped_)
# --------------------------------- WEB APP stuff ------------------------------------------

        if self.save_txt or self.save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if self.save_txt else ''
            #print(f"Results saved to {save_dir}{s}")

        print(f'Done. ({time.time() - t0:.3f}s)')

        # update model (to fix SourceChangeWarning)
        if self.update:
            strip_optimizer(self.weights[0])


if __name__ == '__main__':

    run = Yolov7(source='https://www.youtube.com/watch?v=zu6yUYEERwA')
    run.detect()


    # with torch.no_grad():
    #     if run.update:
    #         for run.weights in ['yolov7.pt']:
    #             run.detect()
    #             strip_optimizer(run.weights)
    #     else:
    #         run.detect()


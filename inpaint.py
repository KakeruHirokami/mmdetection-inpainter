import numpy as np
import cv2
import os

def inpaint(framedir, maskframedir, outfile, fps, fourcc=cv2.VideoWriter_fourcc('m','p','4', 'v')):
    # 動画作成
    img_rep = cv2.imread(f'{framedir}/00000.png')
    h, w, ch = img_rep.shape
    video  = cv2.VideoWriter(outfile, fourcc, fps, (w, h))
    print(f"Write video {outfile} FPS: {fps}, (width, height): ({w}, {h})")

    filenum = len([name for name in os.listdir(framedir) if os.path.isfile(os.path.join(framedir, name))])
    for i in range(filenum):
        imagename = str(i).zfill(5)
        img = cv2.imread(f'{framedir}/{imagename}.png')
        mask = cv2.imread(f'{maskframedir}/{imagename}.png', 0)

        # Inpainting
        dst11 = cv2.inpaint(img,mask,0,cv2.INPAINT_TELEA)
        #dst12 = cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)
        #dst13 = cv2.inpaint(img,mask,10,cv2.INPAINT_TELEA)
        #dst21 = cv2.inpaint(img,mask,0,cv2.INPAINT_NS)
        #dst22 = cv2.inpaint(img,mask,3,cv2.INPAINT_NS)
        #dst23 = cv2.inpaint(img,mask,10,cv2.INPAINT_NS)

        print(f"[3/3] Inpainting {imagename}")

        # make video
        video.write(dst11)

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    framedir = "../20240525_100522000_iOS"
    maskframedir = "../mmdetection/mask-20240525_100522000_iOS"
    outfile = "out2"
    #fourcc=('h', '2', '6', '4')
    inpaint(framedir, maskframedir, outfile, 60)
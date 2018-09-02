import glob, sys
import cv2 as cv

def resize(path, im_size):
    count = 0
    for filename in glob.glob(path):
        im = cv.imread(filename)
        if im is None:
            print (filename)
            continue
        else:
            m, n, _ = im.shape
        
        if m > n:
            ratio = float(im_size)/n
        else :
            ratio = float(im_size)/m

        if ratio == 1 and float(m)/float(n)==1:
            if count%10000 == 0:
                print (count)
            count += 1
            continue
                
        im_resized = cv.resize(im, dsize=(0, 0), fx=ratio, fy=ratio)
        
        m, n, _ = im_resized.shape
        
        if m > n:
            start = int((m - im_size) / 2)
            im_cropped = im_resized[start:start + im_size, :, :]
        else:
            start = int((n - im_size) / 2)
            im_cropped = im_resized[:, start:start + im_size, :]
        
        cv.imwrite(filename, im_cropped) 
        
        if count%10000 == 0:
            print (count)
        
        count += 1
            
if __name__ == "__main__":
    output_path = "../data/test/*.jpg"
    im_size = 512
    
    resize(output_path, im_size)
import cv2 
import numpy as np
from matplotlib import pyplot as plt

'''
只需要 call alignment(images, standard, depth)
images : list of images
standard : index of standard image, 所有 images 都會向 standard image 對齊
depth : 決定縮放的大小, default = 6。EX : depth = 5 最多縮小 5 次
'''
def get_binary_image(image, bias = 4):
    sum_image = np.sum(image, axis = 2)
    binary_threshold = np.percentile(sum_image, 50)
    mask = (~cv2.inRange(sum_image, binary_threshold-bias, binary_threshold+bias))/255
    return mask, sum_image >= binary_threshold

def shrink_image(image):
    width, height, _ = np.shape(image)
    return  cv2.resize(image, (int(height/2), int(width/2)), interpolation=cv2.INTER_LINEAR)

def count_diff_with_mask(image1, mask1, image2, mask2):
    return np.sum(np.logical_xor(np.logical_and(image1, mask1), np.logical_and(image2, mask2)))

def count_diff(image1, image2):
    return np.sum(np.logical_xor(image1, image2))

def shift_image(image, dx, dy):
    shift_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    width, height, _ = np.shape(image)
    new_image = cv2.warpAffine(image, shift_matrix, (height, width))
    return new_image

def alignment(images, standard = 0, depth = 6):
    standard_shrink = [images[standard]]
    temp_mask, temp_binary = get_binary_image(standard_shrink[-1])
    standard_binary = [temp_binary]
    standard_mask = [temp_mask]
    images_output = []
    # compute all shrinked, mask, binary image of standard image
    for i in range(depth):
        standard_shrink.append(shrink_image(standard_shrink[-1]))
        temp_mask, temp_binary = get_binary_image(standard_shrink[-1])
        standard_binary.append(temp_binary)
        standard_mask.append(temp_mask)
        
    neighbors = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, -1], [-1, 1]])
    for index in range(len(images)):
        image_shrink = [images[index]]
        for i in range(depth):
            image_shrink.append(shrink_image(image_shrink[-1]))

        best_shift = np.array([0, 0])
        for level in range(depth-1, -1, -1):
            #print("in depth =",level)
            best_shift = best_shift*2
            min_diff = np.shape(images[index])[0]*np.shape(images[index])[1]
            shift = np.array([0, 0])
            for next in neighbors:
                shifted_image = shift_image(image_shrink[level], best_shift[0]+next[0], best_shift[1]+next[1])
                mask, binary = get_binary_image(shifted_image)
                diff = count_diff_with_mask(standard_binary[level], standard_mask[level], binary, mask)
                if diff < min_diff:
                    min_diff = diff
                    shift = np.array(next)
                #print(diff, min_diff, shift)
            best_shift = best_shift + shift
        
        #print("image", index, "best_shift =",best_shift)
        #print("------------------------------")
        images_output.append(shift_image(images[index], best_shift[0], best_shift[1]))

    return images_output

def test():
    shifts = np.array([[]])
    images = []
    '''images.append(cv2.imread("./My_Images/image0.jpg"))
    images.append(cv2.imread("./My_Images/image1.jpg"))
    images.append(cv2.imread("./My_Images/image2.jpg"))
    images.append(cv2.imread("./My_Images/image3.jpg"))
    images.append(cv2.imread("./My_Images/image4.jpg"))
    images.append(cv2.imread("./My_Images/image5.jpg"))
    images.append(cv2.imread("./My_Images/image6.jpg"))'''
    images.append(cv2.imread("./Memorial_SourceImages/memorial0061.png"))
    images.append(cv2.imread("./Memorial_SourceImages/memorial0062.png"))
    images.append(cv2.imread("./Memorial_SourceImages/memorial0063.png"))
    images.append(cv2.imread("./Memorial_SourceImages/memorial0064.png"))
    images.append(cv2.imread("./Memorial_SourceImages/memorial0065.png"))
    images.append(cv2.imread("./Memorial_SourceImages/memorial0066.png"))
    images.append(cv2.imread("./Memorial_SourceImages/memorial0067.png"))
    images.append(cv2.imread("./Memorial_SourceImages/memorial0068.png"))
    images.append(cv2.imread("./Memorial_SourceImages/memorial0069.png"))
    images.append(cv2.imread("./Memorial_SourceImages/memorial0070.png"))
    images.append(cv2.imread("./Memorial_SourceImages/memorial0071.png"))
    images.append(cv2.imread("./Memorial_SourceImages/memorial0072.png"))
    images.append(cv2.imread("./Memorial_SourceImages/memorial0073.png"))
    images.append(cv2.imread("./Memorial_SourceImages/memorial0074.png"))
    images.append(cv2.imread("./Memorial_SourceImages/memorial0075.png"))
    images.append(cv2.imread("./Memorial_SourceImages/memorial0076.png"))
    output = alignment(images, int(len(images)/2))

    '''for i, img in enumerate(output):
        img = img[:,:,::-1]
        plt.imshow(img)
        plt.show()'''

#test()

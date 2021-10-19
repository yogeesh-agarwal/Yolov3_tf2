import numpy as np
import cv2

def sort_anchors(anchors):
    anchors = np.reshape(anchors, [9,2])
    anchor_areas = {}
    for anchor in anchors:
        area = anchor[0]*anchor[1]
        anchor_areas[area] = anchor

    sorted_areas = sorted(list(anchor_areas.keys()))
    sorted_anchors = []
    for area in sorted_areas[::-1]:
        print(anchor_areas[area] , area)
        sorted_anchors.append(anchor_areas[area])

    return np.array(sorted_anchors).reshape([9,2])

def plot_anchors(anchors , height , width):
    image = np.zeros([height , width , 3])
    center_x = width // 2
    center_y = height // 2
    for index , anchor in enumerate(anchors):
        x1 = int(center_x - (anchor[0] / 2))
        y1 = int(center_y - (anchor[1] / 2))
        x2 = int(center_x + (anchor[0] / 2))
        y2 = int(center_y + (anchor[1] / 2))
        height = anchor[1]
        width = anchor[0]
        anchor_area = width * height
        print(height , width , anchor_area)
        print("percentage area : " , (anchor_area / (416 * 416)) * 100)
        print(x1, y1, x2, y2 , center_x, center_y , anchor)
        cv2.rectangle(image , (x1,y1) , (x2,y2) , (255 , 0 , 0) , 1)

    cv2.rectangle(image , (1,1) , (410 , 410) , (0,0,255) , 1)
    cv2.imshow("image" , image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    anchors = np.array([[0.17028831 , 0.35888521],
                        [0.05563053 , 0.09101727],
                        [0.11255733 , 0.21961425],
                        [0.0347448  , 0.06395953],
                        [0.32428802 , 0.42267646],
                        [0.47664651 , 0.65827237],
                        [0.21481797 , 0.20969635],
                        [0.07297461 , 0.14739788],
                        [0.11702667 , 0.11145465]])

    input_size = 416
    anchors = np.multiply(anchors , 416)
    print("*******************")
    for anchor in anchors:
        print(anchor)
    print("           *                ")
    sorted_anchors = sort_anchors(anchors)
    print("*******************")
    plot_anchors(sorted_anchors , input_size , input_size)

import tensorflow as tf
import numpy as np

class Array(tf.experimental.ExtensionType):
    index : int
    values : tf.Tensor
    matched : tf.Tensor
    iou : tf.Tensor

    def __init__(self , values , matched , iou , index):
        self.values = values
        self.matched = matched
        self.iou = iou
        self.index = index

    def add_index(self):
        self.index += 1

    def replace_values(self):
        new_values = self.values + self.index
        return Array(new_values , self.matched, self.iou, self.index+1)

@tf.function
def def_objs(infos):
    final_obj = []
    for i in range(20):
        objects = []
        for info in infos:
            obj = Array(values = info[0], matched = info[1], iou = info[2] , index = 0)
            objects.append(obj)
        final_obj.append(objects)
    return final_obj

@tf.function
def create_ragged_list(infos):
    final_obj = []
    for i in range(len(infos)):
        objects = infos[i]
        curr_obj = []
        for obj in objects:
            arry_obj = Array(obj[0] , obj[1] , obj[2] , i)
            curr_obj.append(arry_obj)
        final_obj.append(curr_obj)

    return final_obj

# @tf.function
def main():
    # objects = def_objs([[[1,2,3,4,5] , True , 53.46], [[1,6,7,8,9,12] , False , 12.34]])

    input_1 = [[[1,2,3,4,5], True , 0.56] , [[5,6,3,4] , False , 0.12]]
    input_2 = [[[1,56,78,90] , True , 0.99]]
    final_obj = create_ragged_list([input_1 , input_2])
    for obj in final_obj:
        print(np.array(obj).shape)
        for co in obj:
            print(co)
            co = co.add_index()
            print(co)

main()

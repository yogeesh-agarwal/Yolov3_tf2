num_instances = gt_box_objects.shape[0]
gt_obj_per_cls = metric_utils.get_box_objects_per_class(gt_box_objects , self.num_classes , num_instances) # shape = [num_classes , per_cls_objects]
pred_obj_per_cls = metric_utils.get_box_objects_per_class(pred_box_objects , self.num_classes, num_instances)

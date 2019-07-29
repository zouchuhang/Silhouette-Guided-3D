Put pre-trained models here.

## Point Clouds Reconstruction
- PointCloud\_Reconstruction\_resnet50\_ShapeNet.pth trained on ShapeNet without silhouette guidance, refer to paper Tab.1, col 3

## Silhouette Completion
- Silhouette\_Completion\_DYCE\_resnet50.pth Silhouette completion trained on DYCE only, use resnet50 as encoder, refer to paper Tab.4, row3 and Tab.5, row2
- Silhouette\_Completion\_Pix3D\_fold\*.pth, 5 fold cross validation finetuned on Pix3D, refer to Tab.5, row 3

## Silhouette Guided Point Cloud Reconstruction
- PointCloud\_Reconstruction\_resnet50\_no\_Silhouette\_Guidance.pth trained on ShapeNet and LSUN, no silhouette, refer to paper Tab.6/7 row1
- PointCloud\_Reconstruction\_resnet50\_Visible\_Silhouette\_Guidance.pth trained on ShapeNet and LSUN, visible silhouette guidance, refer to paper Tab.6/7 row2
- PointCloud\_Reconstruction\_resnet50\_Complete\_Silhouette\_Guidance.pth trained on ShapeNet and LSUN, complete silhouette guidance, refer to paper Tab.6/7 row3

# VGG19-GAP for Polyp Detection: centralized and federated learning



## Repository overview

This repository implements polyp detection using VGG19 with Global Average Pooling (GAP), comparing centralized and federated learning approaches with Grad-CAM interpretability.

## File Structure

1. VGG19GAP_basic.py              # Centralized training (Experiments A, B)
2. Federated_Learning_basic.py    # Federated learning (Experiments C, D, E, F and G)
3. Make_M1.py                      # Generate M1 modality shift dataset
4. Make_M2.py                      # Generate M2 modality shift dataset
5. Create_graph_fromcsv.py        # Plot centralized learning training loss
6. plot_fl_losses.py              # Plot federated learning training loss
7. README.md                       # This file



## Code Structure

### VGG19GAP_basic.py
- **Model**: VGG19-BN with GAP (16 conv layers → GAP → 2 classes)
- **Training**: 100 epochs, SGD optimizer, cross-entropy loss
- **Evaluation**: Accuracy, sensitivity, specificity, confusion matrix
- **Visualization**: Grad-CAM for TP/TN/FP/FN examples (side-by-side: original | heatmap)
- **Logging**: MLflow tracking with all metrics and artifacts

### Federated_Learning_basic.py
- **FL Algorithm**: FedAvg with 3 clients, 50 global rounds
- **Aggregation**: Weighted by dataset size
- **Local Training**: Configurable epochs per client
- **Communication**: Tracks bytes transferred per round
- **Same model/eval/visualization** as centralized version

### Make_M1.py / Make_M2.py
- Transform CP-CHILD-A to create modality-shifted datasets
- M1: Independent channel scaling
- M2: Linear color transformation with cross-talk



## Dataset Structure

/scratch/s3777006/CP-CHILD/

   Experiment-centralized_learning/
   
       Train/{Polyp, Non-Polyp}/
       
       Test/{Polyp, Non-Polyp}/
       
  Experiment-federated_learning/
  
       Client1/Train/{Polyp, Non-Polyp}
       
       Client2/Train/{Polyp, Non-Polyp}
       
       Client3/Train/{Polyp, Non-Polyp}
       
       Test/{Polyp, Non-Polyp}
       
   Experiment partitions according the file structure
   
......................

## Outputs

/scratch/s3777006/
  ML_Flow/                    # MLflow tracking data
  gradcam_outputs/            # Grad-CAM visualizations


## Notes

- **GPU**: Designed for NVIDIA A100 
- **Paths**: Hard-coded to `/scratch/s3777006/` - update `STUDENT_ID` variable
- **Pretrained weights**: VGG19-BN ImageNet weights at `pretrained_weights/vgg19_bn_imagenet_weights.pth`

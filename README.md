# CountingMOT

This work is built on [**FairMOT**](https://github.com/ifzhang/FairMOT), and many thanks to its contributions!  

CountingMOT: Joint Counting, Detection and Re-Identification for Multiple Object Tracking:
> [**CountingMOT, arXiv version is available soon!**],            
> Weihong Ren, Bowen Chen, Yuhang Shi, Weibo Jiang and Honghai Liu,        

## Abstract
The recent trend in multiple object tracking (MOT) is jointly solving detection and tracking, where object detection and appearance feature (or motion) are learned simultaneously. Despite competitive performance, in crowded scenes, joint detection and tracking usually fail to find accurate object associations due to missed or false detections. In this paper, we jointly model counting, detection and re-identification in an end-to-end framework, named CountingMOT, tailored for crowded scenes. By imposing mutual object-count constraints between detection and counting, the CountingMOT tries to find a balance between object detection and crowd density map estimation, which can help it to recover missed detections or reject false detections. Our approach is an attempt to bridge the gap of object detection, counting, and re-Identification. This is in contrast to prior MOT methods that either ignore the crowd density and thus are prone to failure in crowded scenes, or depend on local correlations to build a graphical relationship for matching targets. The proposed MOT tracker can perform online and real-time tracking, and achieves the state-of-the-art results on public benchmarks MOT16 (MOTA of 77.6), MOT17 (MOTA of 78.0%) and MOT20 (MOTA of 70.2%). Source code is available: https://github.com/weihong9/CountingMOT.

## Results on MOT challenge test set
| Dataset    |  MOTA | IDF1 | IDS | MT | ML | FPS |
|--------------|-----------|--------|-------|----------|----------|--------|
|MOT16       | 77.6 | 75.2 | 1074 | 50.7% | 14.8% | 24.9 |
|MOT17       | 78.0 | 74.8 | 3453 | 49.8% | 15.4% | 24.9 |
|MOT20       | 70.2 | 72.4 | 2795 | 62.0% | 12.1% | 12.6 |

 All of the results are obtained on the [MOT challenge](https://motchallenge.net) evaluation server under the “private detector” 
 
 ## Installation
* Clone this repo, and we'll call the directory that you cloned as ${FAIRMOT_ROOT}
* Install dependencies. We use python 3.7 and pytorch = 1.4.
* Complie the [**DCNv2**](https://github.com/CharlesShang/DCNv2), and put it in src/lib/models/networks/DCNv2.

## Training and Test
### Training
* Download the CrowdHuman dataset from the [official webpage](https://www.crowdhuman.org).
* Download the MIX dataset including Caltech Pedestrian, CityPersons, CUHK-SYSU, PRW, ETHZ, MOT17 and MOT16 (see [JDE](https://github.com/Zhongdao/Towards-Realtime-MOT)). 
* Training is the same as [**FairMOT**](https://github.com/ifzhang/FairMOT), E.g., 
```
sh experiments/crowdhuman_dla34.sh
sh experiments/mix_ft_ch_dla34.sh
```
### Test
* Download the the pre-trained models from [[BaiduDisk, code: gb9h]](https://pan.baidu.com/s/1l_r3Lb-TzpeCwNll3S0yvw?pwd=gb9h) or [[GoogleDrive]](https://drive.google.com/drive/folders/1KxIdLI39oL0a863RM8SRXUf0rJQcRiqA?usp=sharing).

To get the results of the test set of MOT16 or MOT17, you can run:
```
cd src
python track.py cmot --test_mot17 True --load_model ../models/countingmot_mot17_dla34.pth --conf_thres 0.4
python track.py cmot --test_mot16 True --load_model ../models/countingmot_mot17_dla34.pth --conf_thres 0.4
```
To get the results of the test set of MOT20, you can run:
```
cd src
python track.py cmot --test_mot20 True --load_model ../models/countingmot_mot20_dla34.pth --conf_thres 0.3
```

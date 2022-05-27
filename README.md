# SIT
### [Paper](https://arxiv.org/abs/2205.13296) 
- This is the official implementation of the paper: **Social Interpretable Tree for Pedestrian Trajectory Prediction** (AAAI 2022).

## Quick Start

Requires:

* Python== 3.6
* numpy==1.16.4
* torch==1.4.0


### 1) Install Packages

``` bash
 pip install -r requirements.txt
```

### 2) Dataset

Please download the [dataset](https://drive.google.com/drive/folders/1Oy3mfJX6m9I3rlF9LbtUkHa08n9D8ppa?usp=sharing) and extract it into the directory './dataset/' like this:

```
./dataset/train/
./dataset/test/
```

## Performance

Results on ETH-UCY and Stanford Drone Dataset:

<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky"></th>
    <th class="tg-c3ow">minADE</th>
    <th class="tg-c3ow">minFDE</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky">ETH </td>
    <td class="tg-c3ow">0.39</td>
    <td class="tg-c3ow">0.62</td>
  </tr>
  <tr>
    <td class="tg-0pky">HOTEL </td>
    <td class="tg-c3ow">0.14</td>
    <td class="tg-c3ow">0.22</td>
  </tr>
  <tr>
    <td class="tg-0pky">UNIV </td>
    <td class="tg-c3ow">0.27</td>
    <td class="tg-c3ow">0.47</td>
  </tr>
  <tr>
    <td class="tg-0pky">ZARA1 </td>
    <td class="tg-c3ow">0.19</td>
    <td class="tg-c3ow">0.33</td>
  </tr>
  <tr>
    <td class="tg-0pky">ZARA2 </td>
    <td class="tg-c3ow">0.16</td>
    <td class="tg-c3ow">0.29</td>
  </tr>
   <tr>
    <td class="tg-0pky">AVG ETH-UCY </td>
    <td class="tg-c3ow">0.23</td>
    <td class="tg-c3ow">0.38</td>
  </tr>
  <tr>
    <td class="tg-0pky">SDD </td>
    <td class="tg-c3ow">9.13</td>
    <td class="tg-c3ow">15.42</td>
  </tr>
</tbody>
</table>

## SIT

### Training & Evaluation
Suppose the training data is at ```./dataset/```. You can train and evaluate our model on the 'eth' dataset by below command:
```bash
bash train.sh 'eth'
```
Training on a single 2080Ti. 

##Acknowledgement

Thank for the pre-processed data provided by the works of  [PECNet](https://github.com/HarshayuGirase/Human-Path-Prediction).

## Citation
If you find our work useful for your research, please consider citing the paper:
```
@inproceedings{sit,
  title={Social Interpretable Tree for Pedestrian Trajectory Prediction},
  author={Shi, Liushuai and Wang, Le and Long, Chengjiang and Zhou, Sanping and Zheng, Fang and Zheng, Nanning  and Hua, Gang},
  booktitle={Association for the Advance of Artificial Intelligence},
  year={2022}
}
```
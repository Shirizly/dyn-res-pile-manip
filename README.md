# Various Improvements to MPC Process for Object Pile Manipulation

This is a fork of the project: Dynamic-Resolution Model Learning for Object Pile Manipulation

[Website](https://robopil.github.io/dyn-res-pile-manip/) | [Paper](https://arxiv.org/abs/2306.16700)


Dynamic-Resolution Model Learning for Object Pile Manipulation  
[Yixuan Wang*](https://wangyixuan12.github.io/), [Yunzhu Li*](https://yunzhuli.github.io/), [Katherine Driggs-Campbell](https://krdc.web.illinois.edu/), [Li Fei-Fei](https://profiles.stanford.edu/fei-fei-li/), [Jiajun Wu](http://jiajunwu.com/)  
Robotics: Science and Systems, 2023.

## Citation

If you use this code for your research, please cite:

```
@INPROCEEDINGS{Wang-RSS-23, 
    AUTHOR    = {Yixuan Wang AND Yunzhu Li AND Katherine Driggs-Campbell AND Li Fei-Fei AND Jiajun Wu}, 
    TITLE     = {{Dynamic-Resolution Model Learning for Object Pile Manipulation}}, 
    BOOKTITLE = {Proceedings of Robotics: Science and Systems}, 
    YEAR      = {2023}, 
    ADDRESS   = {Daegu, Republic of Korea}, 
    MONTH     = {July}, 
    DOI       = {10.15607/RSS.2023.XIX.047} 
} 
@inproceedings{li2018learning,
    Title={Learning Particle Dynamics for Manipulating Rigid Bodies, Deformable Objects, and Fluids},
    Author={Li, Yunzhu and Wu, Jiajun and Tedrake, Russ and Tenenbaum, Joshua B and Torralba, Antonio},
    Booktitle = {ICLR},
    Year = {2019}
}
```

## Installation

### Prerequisite

- Install [docker-ce](https://docs.docker.com/install/linux/docker-ce/ubuntu/)
- Install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker#quickstart)
- Install [Anaconda](https://www.anaconda.com/distribution/) or [Miniconda](https://docs.conda.io/en/main/miniconda.html)

### Create conda environment
`conda env create -f env.yaml && conda activate dyn-res-pile-manip`

### Install PyFleX
Run `bash scripts/install_pyflex.sh`. You may need to `source ~/.bashrc` to `import PyFleX`.  

